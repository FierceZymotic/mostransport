"""Общие синтетические fixtures для artifact/inference/serving тестов.

Никаких организаторских данных: крошечная CatBoost-модель обучается на
случайных значениях в схеме `tabular-v1` только ради проверки инфраструктуры.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.artifacts.manifest import ArtifactManifest, EvaluationSummary
from mostransport_ml.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

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


def synthetic_feature_frame(n_rows: int = 64, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n_rows, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    frame.loc[::5, "speed_trend_5m"] = np.nan
    return frame


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
