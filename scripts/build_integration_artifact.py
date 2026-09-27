#!/usr/bin/env python3
"""INTEGRATION TEST ONLY — NOT FOR SUBMISSION — NOT A QUALITY MODEL.

Строит крошечный Artifact Bundle v1 на синтетических данных для проверки
HTTP-интеграции Backend → ML (`POST /api/v1/predict`). Официальный датасет не
читается.

- по умолчанию (`--model-family catboost`): `CatBoostRegressor`, схема
  `tabular-v1`, байтово детерминированный bundle,
  `model_version = integration-fixture-v1`;
- `--model-family hist_gradient_boosting`: HGB с конфигурацией H0, схема
  `runtime-safe-v1` (синтетические признаки с NaN),
  `model_version = integration-fixture-hgb-v1`; байты `model.skops` не
  детерминированы (skops пишет zip с временем создания).

Запуск:
    uv run python scripts/build_integration_artifact.py \
        --output /tmp/mostransport-integration-artifact

`--replace` разрешает перезаписать каталог, только если там лежит ранее
построенный integration-fixture bundle.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mostransport_ml.artifacts.bundle import (
    BUNDLE_FILENAME,
    MANIFEST_FILENAME,
    BundleDescriptor,
)
from mostransport_ml.artifacts.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactManifest,
    EvaluationSummary,
)
from mostransport_ml.data.official import DATASET_ENV_VAR
from mostransport_ml.features.schema import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    RUNTIME_SAFE_FEATURE_NAMES,
    RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
)
from mostransport_ml.inference.predictor import export_bundle
from mostransport_ml.models.hgb_v1 import HGB_H0_PARAMS, fit_h0

INTEGRATION_MODEL_VERSION = "integration-fixture-v1"
INTEGRATION_HGB_MODEL_VERSION = "integration-fixture-hgb-v1"
INTEGRATION_MODEL_FAMILIES = ("catboost", "hist_gradient_boosting")
INTEGRATION_TRAIN_REGIME = "integration_fixture_synthetic"
FIXED_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)
N_ROWS = 256
SEED = 20260101
CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "iterations": 30,
    "depth": 3,
    "learning_rate": 0.2,
    "random_seed": SEED,
    "thread_count": 1,
    "verbose": False,
    "allow_writing_files": False,
}
_MODEL_FILENAMES = frozenset({"model.cbm", "model.skops"})
_FIXTURE_MODEL_VERSIONS = frozenset({INTEGRATION_MODEL_VERSION, INTEGRATION_HGB_MODEL_VERSION})


class IntegrationArtifactError(RuntimeError):
    """Каталог назначения небезопасен или занят чем-то, кроме integration fixture."""


def synthetic_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Бессмысленные синтетические данные в схеме tabular-v1 (не организаторские)."""
    rng = np.random.default_rng(SEED)
    features = pd.DataFrame(
        rng.normal(size=(N_ROWS, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES)
    )
    features["cur_dev_s"] = rng.normal(60.0, 120.0, N_ROWS)
    features["horizon_minutes"] = rng.uniform(10.5, 15.0, N_ROWS)
    target = features["cur_dev_s"].to_numpy() + rng.normal(0.0, 30.0, N_ROWS)
    return features, target


def synthetic_runtime_safe_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Те же синтетические данные в схеме runtime-safe-v1, часть ячеек — NaN."""
    features, target = synthetic_training_data()
    runtime_safe = features[list(RUNTIME_SAFE_FEATURE_NAMES)].copy()
    runtime_safe.iloc[::11, runtime_safe.columns.get_loc("speed_trend_5m")] = np.nan
    runtime_safe.iloc[::17, runtime_safe.columns.get_loc("latest_valid_gps_lag_s")] = np.nan
    return runtime_safe, target


def _check_output(output: Path, replace: bool) -> None:
    dataset = os.environ.get(DATASET_ENV_VAR)
    if dataset and output.resolve().is_relative_to(Path(dataset).expanduser().resolve()):
        raise IntegrationArtifactError("output must not be inside the official dataset")
    if not output.exists() or (output.is_dir() and not any(output.iterdir())):
        return
    if not replace:
        raise IntegrationArtifactError("output exists and is not empty (use --replace)")
    entries = {p.name for p in output.iterdir()} if output.is_dir() else set()
    manifest_path = output / MANIFEST_FILENAME
    is_fixture = (
        entries <= {BUNDLE_FILENAME, MANIFEST_FILENAME, *_MODEL_FILENAMES}
        and manifest_path.is_file()
        and ArtifactManifest.from_json(manifest_path.read_bytes()).model_version
        in _FIXTURE_MODEL_VERSIONS
    )
    if not is_fixture:
        raise IntegrationArtifactError("--replace only overwrites an integration-fixture bundle")
    shutil.rmtree(output)


def _manifest(**fields) -> ArtifactManifest:
    n_rows = fields.pop("n_rows")
    return ArtifactManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        target_formulation="direct",
        train_regime=INTEGRATION_TRAIN_REGIME,
        training_data_provenance={
            "source": "synthetic",
            "purpose": "INTEGRATION TEST ONLY - NOT FOR SUBMISSION - NOT A QUALITY MODEL",
            "seed": SEED,
            "n_rows": n_rows,
        },
        code_provenance={"generator": "scripts/build_integration_artifact.py"},
        created_at=FIXED_TIMESTAMP,
        **fields,
    )


def _build_hgb(output: Path) -> BundleDescriptor:
    features, target = synthetic_runtime_safe_training_data()
    model = fit_h0(features, target)
    in_sample_mae = round(float(np.mean(np.abs(model.predict(features) - target))), 6)
    manifest = _manifest(
        model_version=INTEGRATION_HGB_MODEL_VERSION,
        model_family="hist_gradient_boosting",
        feature_schema_version=RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
        model_params=dict(HGB_H0_PARAMS),
        evaluation=EvaluationSummary(
            split="synthetic_train", metric="mae", value=in_sample_mae, n_rows=len(target)
        ),
        n_rows=len(target),
    )
    return export_bundle(output, model, manifest)


def build_integration_artifact(
    output: str | Path, *, replace: bool = False, model_family: str = "catboost"
) -> BundleDescriptor:
    if model_family not in INTEGRATION_MODEL_FAMILIES:
        raise ValueError(f"model_family must be one of {INTEGRATION_MODEL_FAMILIES}")
    output = Path(output)
    _check_output(output, replace)
    if model_family == "hist_gradient_boosting":
        return _build_hgb(output)

    from catboost import CatBoostRegressor

    features, target = synthetic_training_data()
    model = CatBoostRegressor(**CATBOOST_PARAMS).fit(features, target)
    # Детерминированные байты модели: CatBoost кладёт в .cbm случайный guid и время.
    metadata = model.get_metadata()
    metadata["model_guid"] = INTEGRATION_MODEL_VERSION
    metadata["train_finish_time"] = FIXED_TIMESTAMP.strftime("%Y-%m-%dT%H:%M:%SZ")
    in_sample_mae = round(float(np.mean(np.abs(model.predict(features) - target))), 6)
    manifest = ArtifactManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        model_version=INTEGRATION_MODEL_VERSION,
        model_family="catboost",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        target_formulation="direct",
        train_regime=INTEGRATION_TRAIN_REGIME,
        model_params=CATBOOST_PARAMS,
        training_data_provenance={
            "source": "synthetic",
            "purpose": "INTEGRATION TEST ONLY - NOT FOR SUBMISSION - NOT A QUALITY MODEL",
            "seed": SEED,
            "n_rows": N_ROWS,
        },
        code_provenance={"generator": "scripts/build_integration_artifact.py"},
        evaluation=EvaluationSummary(
            split="synthetic_train", metric="mae", value=in_sample_mae, n_rows=N_ROWS
        ),
        created_at=FIXED_TIMESTAMP,
    )
    return export_bundle(output, model, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="INTEGRATION TEST ONLY artifact bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--model-family", choices=INTEGRATION_MODEL_FAMILIES, default="catboost")
    args = parser.parse_args(argv)
    try:
        descriptor = build_integration_artifact(
            args.output, replace=args.replace, model_family=args.model_family
        )
    except IntegrationArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    manifest = ArtifactManifest.from_json((args.output / MANIFEST_FILENAME).read_bytes())
    print("INTEGRATION TEST ONLY - NOT FOR SUBMISSION - NOT A QUALITY MODEL")
    print(f"artifact: {args.output}")
    print(f"model_family: {manifest.model_family}")
    print(f"model_version: {manifest.model_version}")
    print(f"feature_schema_version: {manifest.feature_schema_version}")
    print(f"bundle_sha256: {descriptor.fingerprint()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
