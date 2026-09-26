"""`model_family == "catboost"` = обученная скалярная регрессия задержки.

Классификатор/ранкер/многомерный выход не экспортируется и не загружается как
predictor задержки — ни через exporter, ни через вручную собранный bundle
(байты классификатора грузятся и через `CatBoostRegressor`, поэтому проверяется
сохранённый в модели objective и форма выхода, а не класс обёртки).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool
from conftest import make_manifest, synthetic_feature_frame
from fastapi.testclient import TestClient

from mostransport_ml.artifacts.bundle import write_bundle
from mostransport_ml.features.schema import FEATURE_NAMES
from mostransport_ml.inference.model_families import (
    ModelTaskError,
    validate_catboost_delay_regressor,
)
from mostransport_ml.inference.predictor import ArtifactLoadError, ArtifactPredictor, export_bundle
from mostransport_ml.serving import artifact_app

FIXTURE = Path(__file__).parent / "fixtures" / "contract_v1_request.json"
COMMON = dict(
    iterations=5, depth=2, random_seed=0, thread_count=1, verbose=False, allow_writing_files=False
)
X = synthetic_feature_frame()
Y = X.iloc[:, 0].to_numpy()
GROUPS = np.repeat(np.arange(len(X) // 8), 8)


def regressor(loss="RMSE", target=Y):
    return CatBoostRegressor(**COMMON, loss_function=loss).fit(X, target)


def classifier(loss="Logloss"):
    labels = (Y > 0).astype(int) if loss == "Logloss" else np.digitize(Y, [-0.5, 0.5])
    return CatBoostClassifier(**COMMON, loss_function=loss).fit(X, labels)


def ranker(loss="YetiRank"):
    label = np.abs(np.round(Y)) if loss == "YetiRank" else Y
    return CatBoostRanker(**COMMON, loss_function=loss).fit(Pool(X, label=label, group_id=GROUPS))


def raw_bytes(model) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.cbm")
        model.save_model(path)
        return Path(path).read_bytes()


def bypass_bundle(tmp_path, model, name="bundle") -> Path:
    """Bundle, собранный вручную в обход exporter'а (хеши при этом корректны)."""
    directory = tmp_path / name
    write_bundle(directory, make_manifest(), raw_bytes(model), "model.cbm")
    return directory


def load_reason(directory) -> str:
    with pytest.raises(ArtifactLoadError) as excinfo:
        ArtifactPredictor.load(directory)
    return excinfo.value.reason


# ------------------------------------------------------------------ accepted


@pytest.mark.parametrize(
    "loss", ["RMSE", "MAE", "Quantile:alpha=0.8", "Huber:delta=2", "Expectile:alpha=0.4"]
)
def test_scalar_catboost_regression_is_accepted(tmp_path, loss):
    model = regressor(loss)
    export_bundle(tmp_path / "b", model, make_manifest())
    predictor = ArtifactPredictor.load(tmp_path / "b")
    assert predictor.is_ready() is True
    validate_catboost_delay_regressor(model)


# ------------------------------------------------------------------ export side


@pytest.mark.parametrize(
    "model_factory",
    [classifier, lambda: classifier("MultiClass"), ranker, lambda: ranker("QueryRMSE")],
    ids=["classifier-logloss", "classifier-multiclass", "ranker-yetirank", "ranker-queryrmse"],
)
def test_classifier_and_ranker_are_rejected_at_export(tmp_path, model_factory):
    with pytest.raises(ModelTaskError):
        export_bundle(tmp_path / "b", model_factory(), make_manifest())
    assert not (tmp_path / "b").exists()


def test_generic_catboost_is_rejected_at_export_even_with_regression_objective(tmp_path):
    generic = CatBoost(dict(COMMON, loss_function="RMSE")).fit(X, Y)
    with pytest.raises(ModelTaskError, match="CatBoostRegressor"):
        export_bundle(tmp_path / "b", generic, make_manifest())


def test_unfitted_regressor_is_rejected_at_export(tmp_path):
    with pytest.raises(ModelTaskError, match="not fitted"):
        export_bundle(tmp_path / "b", CatBoostRegressor(**COMMON), make_manifest())


@pytest.mark.parametrize(
    ("loss", "target"),
    [
        ("MultiRMSE", np.c_[Y, -Y]),
        ("RMSEWithUncertainty", Y),
        ("Cox", np.where(Y > 0, 1.0, -1.0) * (np.abs(Y) + 0.1)),
    ],
)
def test_non_scalar_or_non_target_scale_regressors_are_rejected_at_export(tmp_path, loss, target):
    with pytest.raises(ModelTaskError):
        export_bundle(tmp_path / "b", regressor(loss, target), make_manifest())


# ------------------------------------------------------------------ load side (bypassing exporter)


@pytest.mark.parametrize(
    "model_factory",
    [
        classifier,
        lambda: classifier("MultiClass"),
        ranker,
        lambda: ranker("QueryRMSE"),
        lambda: regressor("MultiRMSE", np.c_[Y, -Y]),
        lambda: regressor("RMSEWithUncertainty"),
        lambda: regressor("Cox", np.where(Y > 0, 1.0, -1.0) * (np.abs(Y) + 0.1)),
    ],
    ids=["logloss", "multiclass", "yetirank", "queryrmse", "multirmse", "uncertainty", "cox"],
)
def test_bundles_bypassing_exporter_are_rejected_at_load(tmp_path, model_factory):
    assert load_reason(bypass_bundle(tmp_path, model_factory())) == "model_task_incompatible"


def test_classifier_bytes_load_through_regressor_wrapper_but_objective_is_detected():
    loaded = CatBoostRegressor()
    loaded.load_model(blob=raw_bytes(classifier()))  # обёртка не защищает
    assert loaded.get_all_params()["loss_function"] == "Logloss"
    with pytest.raises(ModelTaskError, match="Logloss"):
        validate_catboost_delay_regressor(loaded)


def test_generic_catboost_bytes_with_scalar_regression_objective_load(tmp_path):
    # В .cbm нет Python-класса — только objective; RMSE — скалярная регрессия.
    generic = CatBoost(dict(COMMON, loss_function="RMSE")).fit(X, Y)
    assert ArtifactPredictor.load(bypass_bundle(tmp_path, generic)).is_ready() is True


class _StubModel:
    def __init__(self, params, fitted=True):
        self._params, self._fitted = params, fitted
        self.feature_names_ = list(FEATURE_NAMES)

    def is_fitted(self):
        return self._fitted

    def get_all_params(self):
        if isinstance(self._params, Exception):
            raise self._params
        return self._params

    def predict(self, data):
        return np.zeros(len(data))


@pytest.mark.parametrize(
    "stub",
    [
        _StubModel({"loss_function": "PythonUserDefinedPerObject"}),
        _StubModel({}),
        _StubModel(RuntimeError("no params")),
        _StubModel({"loss_function": "RMSE"}, fitted=False),
    ],
    ids=["custom-objective", "missing-objective", "params-error", "unfitted"],
)
def test_unverifiable_models_are_rejected(stub):
    with pytest.raises(ModelTaskError):
        validate_catboost_delay_regressor(stub)


# ------------------------------------------ end-to-end: never ready, no fake delay


@pytest.mark.parametrize("model_factory", [classifier, ranker], ids=["classifier", "ranker"])
def test_service_with_task_incompatible_bundle_never_returns_a_delay(
    tmp_path, caplog, model_factory
):
    bundle = bypass_bundle(tmp_path, model_factory())
    with caplog.at_level(logging.INFO, logger="mostransport_ml.serving"):
        client = TestClient(artifact_app.create_artifact_app(bundle))
    assert client.get("/ready").status_code == 503
    assert client.get("/ready").json()["ready"] is False
    response = client.post("/api/v1/predict", json=json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert response.status_code == 503
    assert "delay_seconds" not in response.text
    assert "reason=model_task_incompatible" in caplog.text
    assert str(tmp_path) not in caplog.text
