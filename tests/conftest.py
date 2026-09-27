"""Общие синтетические fixtures для artifact/inference/serving тестов.

Никаких организаторских данных: крошечные CatBoost (`tabular-v1`) и HGB
(`runtime-safe-v1`, с NaN) обучаются на случайных значениях только ради
проверки инфраструктуры.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.artifacts.manifest import ArtifactManifest, EvaluationSummary
from mostransport_ml.features.schema import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    RUNTIME_SAFE_FEATURE_NAMES,
    RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
)

TINY_CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "iterations": 25,
    "depth": 3,
    "learning_rate": 0.3,
    "random_seed": 7,
    "thread_count": 1,
    "verbose": False,
    "allow_writing_files": False,
}
TINY_HGB_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.2,
    "max_iter": 25,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 5,
    "early_stopping": False,
    "random_state": 7,
}


def synthetic_feature_frame(n_rows: int = 64, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n_rows, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    frame.loc[::5, "speed_trend_5m"] = np.nan
    return frame


def synthetic_runtime_safe_frame(n_rows: int = 64, seed: int = 0) -> pd.DataFrame:
    """Синтетическая матрица ровно `runtime-safe-v1` (29) с легитимными NaN."""
    frame = synthetic_feature_frame(n_rows, seed)[list(RUNTIME_SAFE_FEATURE_NAMES)].copy()
    frame.loc[::4, "latest_valid_gps_lag_s"] = np.nan
    return frame


def train_tiny_hgb(frame: pd.DataFrame | None = None, estimator=None):
    """Крошечная обученная HGB-модель; по умолчанию на `synthetic_runtime_safe_frame()`."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    frame = synthetic_runtime_safe_frame() if frame is None else frame
    values = np.asarray(frame, dtype=float)
    target = 30.0 * np.nan_to_num(values[:, 0]) + 5.0 * np.nan_to_num(values[:, 5])
    model = estimator if estimator is not None else HistGradientBoostingRegressor(**TINY_HGB_PARAMS)
    return model.fit(frame, target)


def make_hgb_manifest(**overrides) -> ArtifactManifest:
    fields = {
        "model_version": "synthetic-hgb-0001",
        "model_family": "hist_gradient_boosting",
        "feature_schema_version": RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
        "model_params": dict(TINY_HGB_PARAMS),
    }
    fields.update(overrides)
    return make_manifest(**fields)


def train_tiny_catboost(columns: list[str] | None = None):
    from catboost import CatBoostRegressor

    frame = synthetic_feature_frame()
    if columns is not None:
        frame.columns = columns
    target = 30.0 * frame.iloc[:, 0].to_numpy() + 5.0 * frame.iloc[:, 5].to_numpy()
    return CatBoostRegressor(**TINY_CATBOOST_PARAMS).fit(frame, target)


def make_manifest(**overrides) -> ArtifactManifest:
    fields = {
        "artifact_schema_version": "artifact-manifest-v1",
        "model_version": "synthetic-infra-0001",
        "model_family": "catboost",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_formulation": "direct",
        "train_regime": "synthetic_fixture",
        "model_params": dict(TINY_CATBOOST_PARAMS),
        "training_data_provenance": {"source": "synthetic-test-fixture"},
        "code_provenance": {"tests": "conftest"},
        "evaluation": EvaluationSummary(split="synthetic", metric="mae", value=1.0, n_rows=64),
        "created_at": datetime(2026, 9, 26, 0, 0, tzinfo=UTC),
    }
    fields.update(overrides)
    return ArtifactManifest(**fields)


@pytest.fixture(scope="session")
def tiny_catboost_model():
    return train_tiny_catboost()


@pytest.fixture
def catboost_bundle(tmp_path, tiny_catboost_model):
    """Фабрика реальных CatBoost bundle: `catboost_bundle(target_formulation=...)` → путь."""
    from mostransport_ml.inference.predictor import export_bundle

    counter = iter(range(1000))

    def factory(**manifest_overrides):
        directory = tmp_path / f"bundle-{next(counter)}"
        export_bundle(directory, tiny_catboost_model, make_manifest(**manifest_overrides))
        return directory

    return factory


@pytest.fixture(scope="session")
def tiny_hgb_model():
    return train_tiny_hgb()


@pytest.fixture
def hgb_bundle(tmp_path, tiny_hgb_model):
    """Фабрика реальных HGB `runtime-safe-v1` bundle: `hgb_bundle(**manifest_overrides)` → путь."""
    from mostransport_ml.inference.predictor import export_bundle

    counter = iter(range(1000))

    def factory(**manifest_overrides):
        directory = tmp_path / f"hgb-bundle-{next(counter)}"
        export_bundle(directory, tiny_hgb_model, make_hgb_manifest(**manifest_overrides))
        return directory

    return factory
