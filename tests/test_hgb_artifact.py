"""P2: HGB H0, family `hist_gradient_boosting` (skops) и runtime-safe-v1 artifact inference.

Все модели крошечные и синтетические. Главные acceptance-тесты: полный
roundtrip HGB-bundle (train → skops → Artifact Bundle v1 → ArtifactPredictor →
P1-проекция → HGB) и HTTP E2E на реальном artifact'е без mock-predictor'а.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from conftest import (
    TINY_HGB_PARAMS,
    make_hgb_manifest,
    make_manifest,
    synthetic_feature_frame,
    synthetic_runtime_safe_frame,
    train_tiny_hgb,
)
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from test_inference_predictor import batch_one, multi_point_batch
from test_ml_context import packet, runtime_request

from mostransport_ml.artifacts.bundle import (
    BUNDLE_FILENAME,
    ArtifactBundleError,
    load_bundle,
    sha256_bytes,
    write_bundle,
)
from mostransport_ml.artifacts.manifest import ArtifactCompatibilityError, validate_compatibility
from mostransport_ml.features import context as context_module
from mostransport_ml.features.adapters import runtime_context
from mostransport_ml.features.context import (
    build_features_from_context,
    features_for_schema,
    project_runtime_safe_features,
)
from mostransport_ml.features.schema import (
    FEATURE_NAMES,
    RUNTIME_SAFE_EXCLUDED_FEATURES,
    RUNTIME_SAFE_FEATURE_NAMES,
    SUPPORTED_FEATURE_SCHEMA_VERSIONS,
    UnsupportedFeatureSchemaError,
    feature_names_for_schema,
)
from mostransport_ml.inference import model_families
from mostransport_ml.inference.model_families import (
    HGB_TRUSTED_SKOPS_TYPES,
    MODEL_FAMILIES,
    ModelFamilyError,
    ModelTaskError,
)
from mostransport_ml.inference.predictor import ArtifactLoadError, ArtifactPredictor, export_bundle
from mostransport_ml.models.hgb_v1 import HGB_H0_PARAMS, fit_h0, make_regressor
from mostransport_ml.serving import artifact_app

HGB = MODEL_FAMILIES["hist_gradient_boosting"]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "contract_v1_request.json"
TREE_PREDICTOR = "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"


def target_for(frame: pd.DataFrame) -> np.ndarray:
    return 20.0 * np.nan_to_num(frame["cur_dev_s"].to_numpy()) + np.arange(len(frame)) % 7


# ------------------------------------------------------------------ H0 production config


def test_h0_config_is_exact():
    assert dict(HGB_H0_PARAMS) == {
        "loss": "absolute_error",
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
    }


def test_h0_config_is_immutable():
    with pytest.raises(TypeError):
        HGB_H0_PARAMS["max_iter"] = 10  # type: ignore[index]


def test_make_regressor_is_exact_h0():
    model = make_regressor()
    assert type(model) is HistGradientBoostingRegressor
    params = model.get_params()
    assert {k: params[k] for k in HGB_H0_PARAMS} == dict(HGB_H0_PARAMS)


def test_fit_h0_is_deterministic_and_accepts_nan():
    X = synthetic_runtime_safe_frame(n_rows=120, seed=3)
    assert X.isna().any().any()
    first, second = fit_h0(X, target_for(X)), fit_h0(X, target_for(X))
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert list(first.feature_names_in_) == list(RUNTIME_SAFE_FEATURE_NAMES)
    assert np.isfinite(first.predict(X)).all()


@pytest.mark.parametrize(
    "drift",
    [
        lambda X: X[list(X.columns)[::-1]],
        lambda X: X.drop(columns="speed_last"),
        lambda X: X.assign(rows_5m=1.0),
        lambda X: synthetic_feature_frame(n_rows=len(X)),
    ],
    ids=["reordered", "missing", "extra_raw_count", "tabular_v1"],
)
def test_fit_h0_rejects_schema_drift(drift):
    X = synthetic_runtime_safe_frame(n_rows=40)
    with pytest.raises(ValueError, match="runtime-safe-v1"):
        fit_h0(drift(X), target_for(X))


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_fit_h0_rejects_infinity_in_features(value):
    X = synthetic_runtime_safe_frame(n_rows=40)
    X.iloc[3, 7] = value
    with pytest.raises(ValueError, match="infinity"):
        fit_h0(X, target_for(X))


def test_fit_h0_rejects_non_numeric_features():
    X = synthetic_runtime_safe_frame(n_rows=40)
    with pytest.raises(ValueError, match="numeric"):
        fit_h0(X.assign(manual_fill=X["manual_fill"] > 0), target_for(X))
    with pytest.raises(TypeError):
        fit_h0(X.to_numpy(), target_for(X))


@pytest.mark.parametrize(
    "bad_y",
    [
        lambda y: np.where(np.arange(len(y)) == 2, np.nan, y),
        lambda y: np.where(np.arange(len(y)) == 2, np.inf, y),
        lambda y: y[:-1],
        lambda y: y.astype(str),
        lambda y: y > 0,
    ],
    ids=["nan", "inf", "length", "strings", "bool"],
)
def test_fit_h0_rejects_bad_target(bad_y):
    X = synthetic_runtime_safe_frame(n_rows=40)
    with pytest.raises(ValueError):
        fit_h0(X, bad_y(target_for(X)))


# ------------------------------------------------------------------ family: serialization


def test_hgb_family_contract():
    assert HGB.name == "hist_gradient_boosting"
    assert HGB.model_filename == "model.skops"
    assert HGB_TRUSTED_SKOPS_TYPES == (TREE_PREDICTOR,)


def test_fitted_hgb_roundtrip_is_exact_and_needs_only_the_frozen_trust(tiny_hgb_model):
    blob = HGB.serialize(tiny_hgb_model)
    assert sio.get_untrusted_types(data=blob) == [TREE_PREDICTOR]
    loaded = HGB.load(blob)
    assert type(loaded) is HistGradientBoostingRegressor
    assert HGB.feature_names(loaded) == list(RUNTIME_SAFE_FEATURE_NAMES)
    X = synthetic_runtime_safe_frame(n_rows=50, seed=9)
    assert X.isna().any().any()
    np.testing.assert_allclose(loaded.predict(X), tiny_hgb_model.predict(X), rtol=0, atol=1e-12)


def test_unfitted_hgb_cannot_be_exported():
    with pytest.raises(ModelTaskError, match="not fitted"):
        HGB.serialize(HistGradientBoostingRegressor())


def test_classifier_cannot_masquerade_as_hgb_regression():
    X = synthetic_runtime_safe_frame()
    classifier = HistGradientBoostingClassifier(max_iter=5).fit(X, target_for(X) > 10)
    with pytest.raises(ModelTaskError, match="HistGradientBoostingRegressor"):
        HGB.serialize(classifier)
    with pytest.raises(ModelTaskError, match="HistGradientBoostingRegressor"):
        HGB.load(sio.dumps(classifier))


def test_unrelated_estimator_and_wrappers_cannot_masquerade(tiny_hgb_model):
    X = synthetic_runtime_safe_frame().fillna(0.0)
    linear = LinearRegression().fit(X, target_for(X))
    with pytest.raises(ModelTaskError):
        HGB.serialize(linear)
    with pytest.raises(ModelTaskError):
        HGB.load(sio.dumps(linear))
    with pytest.raises(ModelTaskError):
        HGB.load(sio.dumps({"model": tiny_hgb_model}))


class OverriddenPredict(HistGradientBoostingRegressor):
    def predict(self, X):
        return np.zeros(len(X))


def test_subclass_with_overridden_predict_is_rejected():
    model = train_tiny_hgb(estimator=OverriddenPredict(**TINY_HGB_PARAMS))
    with pytest.raises(ModelTaskError, match="OverriddenPredict"):
        HGB.serialize(model)
    with pytest.raises(ModelFamilyError, match="untrusted types"):
        HGB.load(sio.dumps(model))


@pytest.mark.parametrize("blob", [b"", b"not a zip archive", b"PK\x03\x04garbage"])
def test_corrupted_bytes_fail_as_load_error(blob):
    with pytest.raises(ModelFamilyError) as excinfo:
        HGB.load(blob)
    assert not isinstance(excinfo.value, ModelTaskError)


def test_truncated_model_fails_as_load_error(tiny_hgb_model):
    blob = HGB.serialize(tiny_hgb_model)
    with pytest.raises(ModelFamilyError) as excinfo:
        HGB.load(blob[: len(blob) // 2])
    assert not isinstance(excinfo.value, ModelTaskError)


CONSTRUCTED: list[object] = []


class Payload:
    """Чужой тип внутри skops-файла: не должен быть сконструирован при загрузке."""

    def __setstate__(self, state):
        CONSTRUCTED.append(state)
        self.__dict__.update(state)


def test_unexpected_serialized_type_is_rejected_without_constructing_it(tiny_hgb_model):
    blob = sio.dumps({"model": tiny_hgb_model, "payload": Payload()})
    assert any(t.endswith("Payload") for t in sio.get_untrusted_types(data=blob))
    CONSTRUCTED.clear()
    with pytest.raises(ModelFamilyError, match="untrusted types"):
        HGB.load(blob)
    assert CONSTRUCTED == []


def tampered(model, field: str, value, *, leaf: bool = False) -> bytes:
    bad = copy.deepcopy(model)
    nodes = bad._predictors[0][0].nodes
    node = int(np.flatnonzero(nodes["is_leaf"] == 1)[0]) if leaf else 0
    assert leaf or nodes["is_leaf"][0] == 0  # корень — сплит
    nodes[field][node] = value
    return sio.dumps(bad)


@pytest.mark.parametrize(
    ("field", "value", "leaf"),
    [
        ("left", 10**6, False),
        ("right", 0, False),  # ссылка на себя → цикл
        ("feature_idx", 29, False),
        ("feature_idx", -1, False),
        ("is_categorical", 1, False),
        ("value", np.nan, True),
    ],
)
def test_structurally_invalid_trees_are_rejected_before_any_predict(
    tiny_hgb_model, monkeypatch, field, value, leaf
):
    blob = tampered(tiny_hgb_model, field, value, leaf=leaf)
    calls = []
    original = HistGradientBoostingRegressor.predict
    monkeypatch.setattr(
        HistGradientBoostingRegressor,
        "predict",
        lambda self, X: calls.append(1) or original(self, X),
    )
    with pytest.raises(ModelFamilyError):
        HGB.load(blob)
    assert calls == []


def test_non_finite_baseline_is_rejected(tiny_hgb_model):
    bad = copy.deepcopy(tiny_hgb_model)
    bad._baseline_prediction = np.array([[np.nan]])
    with pytest.raises(ModelFamilyError, match="baseline"):
        HGB.serialize(bad)
    with pytest.raises(ModelFamilyError, match="baseline"):
        HGB.load(sio.dumps(bad))


def test_model_without_feature_names_is_rejected():
    X = synthetic_runtime_safe_frame()
    unnamed = HistGradientBoostingRegressor(**TINY_HGB_PARAMS).fit(X.to_numpy(), target_for(X))
    with pytest.raises(ModelTaskError, match="feature names"):
        HGB.serialize(unnamed)
    with pytest.raises(ModelTaskError, match="feature names"):
        HGB.load(sio.dumps(unnamed))


# ------------------------------------------------------------------ feature schemas


def test_supported_feature_schema_lookup():
    assert sorted(SUPPORTED_FEATURE_SCHEMA_VERSIONS) == ["runtime-safe-v1", "tabular-v1"]
    assert feature_names_for_schema("tabular-v1") == FEATURE_NAMES
    assert feature_names_for_schema("runtime-safe-v1") == RUNTIME_SAFE_FEATURE_NAMES
    for unknown in ("tabular-v2", "", None, ["tabular-v1"]):
        with pytest.raises(UnsupportedFeatureSchemaError):
            feature_names_for_schema(unknown)  # type: ignore[arg-type]


def test_features_for_schema_routes_runtime_safe_through_p1_projection(monkeypatch):
    tabular = build_features_from_context(batch_one())
    calls = []
    real = context_module.project_runtime_safe_features
    monkeypatch.setattr(
        context_module, "project_runtime_safe_features", lambda f: calls.append(f) or real(f)
    )
    projected = features_for_schema(tabular, "runtime-safe-v1")
    assert len(calls) == 1 and calls[0] is tabular
    pd.testing.assert_frame_equal(projected, real(tabular), check_exact=True)
    assert features_for_schema(tabular, "tabular-v1") is tabular
    with pytest.raises(UnsupportedFeatureSchemaError):
        features_for_schema(tabular, "runtime-safe-v2")
    with pytest.raises(ValueError):
        features_for_schema(tabular[list(FEATURE_NAMES)[::-1]], "tabular-v1")


def test_compatibility_accepts_exactly_one_explicit_mode():
    hgb_manifest = make_hgb_manifest()
    supported = SUPPORTED_FEATURE_SCHEMA_VERSIONS
    assert validate_compatibility(hgb_manifest, supported_feature_schema_versions=supported)
    assert validate_compatibility(make_manifest(), supported_feature_schema_versions=supported)
    with pytest.raises(ArtifactCompatibilityError, match="expected 'tabular-v1'"):
        validate_compatibility(hgb_manifest, expected_feature_schema_version="tabular-v1")
    with pytest.raises(ArtifactCompatibilityError, match="one of"):
        validate_compatibility(
            make_manifest(feature_schema_version="tabular-v2"),
            supported_feature_schema_versions=supported,
        )
    for kwargs in (
        {},
        {
            "expected_feature_schema_version": "tabular-v1",
            "supported_feature_schema_versions": supported,
        },
        {"supported_feature_schema_versions": "tabular-v1"},
        {"supported_feature_schema_versions": []},
        {"supported_feature_schema_versions": ["tabular-v1", ""]},
    ):
        with pytest.raises(ValueError):
            validate_compatibility(hgb_manifest, **kwargs)


def test_load_bundle_requires_an_explicit_schema_mode(hgb_bundle):
    directory = hgb_bundle()
    with pytest.raises(ValueError):
        load_bundle(directory, supported_model_families=MODEL_FAMILIES)
    verified = load_bundle(
        directory,
        supported_model_families=MODEL_FAMILIES,
        supported_feature_schema_versions=SUPPORTED_FEATURE_SCHEMA_VERSIONS,
    )
    assert verified.manifest.feature_schema_version == "runtime-safe-v1"


# ------------------------------------------------------------------ bundle × schema matrix


def load_reason(directory, **kwargs) -> str:
    with pytest.raises(ArtifactLoadError) as excinfo:
        ArtifactPredictor.load(directory, **kwargs)
    return excinfo.value.reason


def handcrafted(directory, model, manifest, family: str = "hist_gradient_boosting"):
    """Bundle в обход export_bundle (как у злоумышленника/ошибки): проверяет loader."""
    spec = MODEL_FAMILIES[family]
    write_bundle(directory, manifest, spec.serialize(model), spec.model_filename)
    return directory


def test_hgb_runtime_safe_bundle_exports_and_loads(hgb_bundle):
    predictor = ArtifactPredictor.load(hgb_bundle())
    assert predictor.manifest.model_family == "hist_gradient_boosting"
    assert predictor.feature_schema_version() == "runtime-safe-v1"
    assert predictor.feature_names == RUNTIME_SAFE_FEATURE_NAMES


@pytest.mark.parametrize(
    "frame",
    [
        synthetic_feature_frame(),
        synthetic_runtime_safe_frame()[list(RUNTIME_SAFE_FEATURE_NAMES)[::-1]],
    ],
    ids=["hgb_on_37", "hgb_on_reordered_29"],
)
def test_hgb_with_wrong_features_for_runtime_safe_manifest_is_rejected(tmp_path, frame):
    model = train_tiny_hgb(frame)
    with pytest.raises(ArtifactBundleError, match="runtime-safe-v1"):
        export_bundle(tmp_path / "export", model, make_hgb_manifest())
    assert not (tmp_path / "export").exists()
    handcrafted(tmp_path / "b", model, make_hgb_manifest())
    assert load_reason(tmp_path / "b") == "model_feature_mismatch"


def test_hgb_29_with_tabular_manifest_is_rejected(tmp_path, tiny_hgb_model):
    manifest = make_hgb_manifest(feature_schema_version="tabular-v1")
    with pytest.raises(ArtifactBundleError, match="tabular-v1"):
        export_bundle(tmp_path / "export", tiny_hgb_model, manifest)
    handcrafted(tmp_path / "b", tiny_hgb_model, manifest)
    assert load_reason(tmp_path / "b") == "model_feature_mismatch"


def test_catboost_37_with_runtime_safe_manifest_is_rejected(tmp_path, tiny_catboost_model):
    manifest = make_manifest(feature_schema_version="runtime-safe-v1")
    with pytest.raises(ArtifactBundleError, match="runtime-safe-v1"):
        export_bundle(tmp_path / "export", tiny_catboost_model, manifest)
    handcrafted(tmp_path / "b", tiny_catboost_model, manifest, family="catboost")
    assert load_reason(tmp_path / "b") == "model_feature_mismatch"


def test_unknown_feature_schema_is_rejected_before_the_model_is_parsed(
    tmp_path, tiny_hgb_model, monkeypatch
):
    manifest = make_hgb_manifest(feature_schema_version="runtime-safe-v2")
    with pytest.raises(ArtifactCompatibilityError):
        export_bundle(tmp_path / "export", tiny_hgb_model, manifest)
    handcrafted(tmp_path / "b", tiny_hgb_model, manifest)
    calls = []
    patched = type(HGB)(
        name=HGB.name,
        model_filename=HGB.model_filename,
        serialize=HGB.serialize,
        load=lambda data: calls.append(data),
        feature_names=HGB.feature_names,
    )
    monkeypatch.setitem(model_families.MODEL_FAMILIES, "hist_gradient_boosting", patched)
    assert load_reason(tmp_path / "b") == "incompatible_feature_schema"
    assert load_reason(tmp_path / "b", expected_feature_schema_version="runtime-safe-v2") == (
        "incompatible_feature_schema"
    )
    assert calls == []


def test_explicit_loader_pin(hgb_bundle, catboost_bundle):
    hgb, catboost = hgb_bundle(), catboost_bundle()
    assert load_reason(hgb, expected_feature_schema_version="tabular-v1") == (
        "incompatible_feature_schema"
    )
    assert load_reason(catboost, expected_feature_schema_version="runtime-safe-v1") == (
        "incompatible_feature_schema"
    )
    pinned = ArtifactPredictor.load(hgb, expected_feature_schema_version="runtime-safe-v1")
    assert pinned.feature_schema_version() == "runtime-safe-v1"
    legacy = ArtifactPredictor.load(catboost, expected_feature_schema_version="tabular-v1")
    assert legacy.feature_schema_version() == "tabular-v1"


def test_default_loader_accepts_both_supported_generations(hgb_bundle, catboost_bundle):
    hgb = ArtifactPredictor.load(hgb_bundle())
    legacy = ArtifactPredictor.load(catboost_bundle())
    assert (hgb.manifest.model_family, hgb.feature_schema_version()) == (
        "hist_gradient_boosting",
        "runtime-safe-v1",
    )
    assert (legacy.manifest.model_family, legacy.feature_schema_version()) == (
        "catboost",
        "tabular-v1",
    )
    assert legacy.feature_names == FEATURE_NAMES
    batch = batch_one()
    for predictor in (hgb, legacy):
        assert np.isfinite(predictor.predict(batch)).all()


# ------------------------------------------------------------------ predictor


class Recorder:
    """Прозрачная обёртка модели: запоминает ровно то, что получила модель."""

    def __init__(self, model) -> None:
        self.model, self.frames = model, []

    def predict(self, data):
        self.frames.append(data.copy())
        return self.model.predict(data)


def recording(predictor: ArtifactPredictor) -> Recorder:
    recorder = Recorder(predictor._model)
    predictor._model = recorder
    return recorder


def test_runtime_safe_predictor_receives_exactly_the_p1_projection(hgb_bundle):
    predictor = ArtifactPredictor.load(hgb_bundle())
    recorder = recording(predictor)
    batch = multi_point_batch()
    predictor.predict(batch)
    (frame,) = recorder.frames
    assert frame.shape == (3, 29)
    assert tuple(frame.columns) == RUNTIME_SAFE_FEATURE_NAMES
    assert not set(RUNTIME_SAFE_EXCLUDED_FEATURES) & set(frame.columns)
    expected = project_runtime_safe_features(build_features_from_context(batch))
    pd.testing.assert_frame_equal(frame, expected, check_exact=True)


def test_legacy_predictor_receives_exactly_tabular_v1(catboost_bundle):
    predictor = ArtifactPredictor.load(catboost_bundle())
    recorder = recording(predictor)
    batch = multi_point_batch()
    predictor.predict(batch)
    (frame,) = recorder.frames
    assert frame.shape == (3, 37)
    pd.testing.assert_frame_equal(frame, build_features_from_context(batch), check_exact=True)


def test_runtime_safe_predictions_keep_batch_point_order(hgb_bundle, tiny_hgb_model):
    predictor = ArtifactPredictor.load(hgb_bundle())
    recorder = recording(predictor)
    batch = multi_point_batch()
    predictions = predictor.predict(batch)
    assert (
        list(recorder.frames[0].index) == [p.point_id for p in batch.points] == ["p3", "p0", "p7"]
    )
    expected = tiny_hgb_model.predict(
        project_runtime_safe_features(build_features_from_context(batch))
    )
    np.testing.assert_array_equal(predictions, expected)
    assert len(set(predictions.tolist())) > 1


def test_nan_survives_projection_into_hgb(hgb_bundle, tiny_hgb_model):
    predictor = ArtifactPredictor.load(hgb_bundle())
    recorder = recording(predictor)
    batch = runtime_context(runtime_request([packet(-400, speed=12.0)]))  # редкая история
    prediction = predictor.predict(batch)
    projected = project_runtime_safe_features(build_features_from_context(batch))
    nan_columns = projected.columns[projected.isna().iloc[0]].tolist()
    assert {"speed_mean_1m", "speed_std_3m"} <= set(nan_columns)
    frame = recorder.frames[0]
    assert frame.columns[frame.isna().iloc[0]].tolist() == nan_columns
    np.testing.assert_array_equal(prediction, tiny_hgb_model.predict(projected))
    assert np.isfinite(prediction).all()


def test_hgb_residual_uses_canonical_current_deviation(hgb_bundle, tiny_hgb_model):
    batch = batch_one()
    projected = project_runtime_safe_features(build_features_from_context(batch))
    raw = tiny_hgb_model.predict(projected)
    direct = ArtifactPredictor.load(hgb_bundle(target_formulation="direct")).predict(batch)
    residual = ArtifactPredictor.load(hgb_bundle(target_formulation="residual")).predict(batch)
    np.testing.assert_array_equal(direct, raw)
    np.testing.assert_allclose(residual, raw + batch.points[0].current_deviation_s, atol=1e-9)


# ------------------------------------------------------------------ main roundtrip acceptance


def test_hgb_artifact_roundtrip_acceptance(tmp_path):
    frame = synthetic_runtime_safe_frame(n_rows=96, seed=21)
    assert frame.isna().any().any()
    model = train_tiny_hgb(frame)
    batch = multi_point_batch()
    projected = project_runtime_safe_features(build_features_from_context(batch))
    before_train, before_batch = model.predict(frame), model.predict(projected)

    directory = tmp_path / "hgb"
    descriptor = export_bundle(directory, model, make_hgb_manifest())
    assert sorted(p.name for p in directory.iterdir()) == [
        "bundle.json",
        "manifest.json",
        "model.skops",
    ]
    predictor = ArtifactPredictor.load(directory)
    np.testing.assert_allclose(predictor._model.predict(frame), before_train, rtol=0, atol=1e-12)
    np.testing.assert_allclose(predictor.predict(batch), before_batch, rtol=0, atol=1e-12)

    assert predictor.manifest.model_family == "hist_gradient_boosting"
    assert predictor.manifest.feature_schema_version == "runtime-safe-v1"
    assert predictor.manifest.artifact_schema_version == "artifact-manifest-v1"
    assert descriptor.bundle_schema_version == "artifact-bundle-v1"
    assert predictor.model_version() == "synthetic-hgb-0001"
    assert predictor.feature_schema_version() == "runtime-safe-v1"
    assert predictor.bundle_sha256 == descriptor.fingerprint()
    assert predictor.bundle_sha256 == sha256_bytes((directory / BUNDLE_FILENAME).read_bytes())

    model_file = directory / "model.skops"
    model_file.write_bytes(model_file.read_bytes() + b"x")
    assert load_reason(directory) == "bundle_invalid"
    corrupted = b"not a skops archive"
    model_file.write_bytes(corrupted)
    data = json.loads((directory / BUNDLE_FILENAME).read_bytes())
    data["model_sha256"] = sha256_bytes(corrupted)
    (directory / BUNDLE_FILENAME).write_text(json.dumps(data))
    assert load_reason(directory) == "model_load_failed"


# ------------------------------------------------------------------ serving E2E (real artifact)


def test_hgb_serving_end_to_end(hgb_bundle, tiny_hgb_model):
    client = TestClient(artifact_app.create_artifact_app(hgb_bundle()))
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "ready": True,
        "model_version": "synthetic-hgb-0001",
        "feature_schema_version": "runtime-safe-v1",
    }

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["model_version"] == "synthetic-hgb-0001"
    assert body["feature_schema_version"] == "runtime-safe-v1"
    delay = body["prediction"]["delay_seconds"]
    assert isinstance(delay, float) and math.isfinite(delay)
    planned = datetime.fromisoformat(payload["schedule_context"]["target_time_begin"])
    target_time = datetime.fromisoformat(body["prediction"]["target_time"])
    assert abs(target_time - (planned + timedelta(seconds=delay))) <= timedelta(microseconds=1)

    # Ответ — это настоящий загруженный HGB на P1-проекции того же запроса.
    batch = runtime_context(payload)
    expected = tiny_hgb_model.predict(
        project_runtime_safe_features(build_features_from_context(batch))
    )
    assert delay == float(expected[0])


def test_catboost_serving_still_reports_tabular_v1(catboost_bundle):
    client = TestClient(artifact_app.create_artifact_app(catboost_bundle()))
    assert client.get("/ready").json()["feature_schema_version"] == "tabular-v1"
    response = client.post("/api/v1/predict", json=json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert response.status_code == 200
    assert response.json()["feature_schema_version"] == "tabular-v1"
