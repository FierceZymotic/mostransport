"""Artifact-backed predictor: Artifact Bundle v1 + общий Feature Builder + reconstruction.

Один и тот же путь для runtime (Contract v1 → runtime adapter) и offline
(official данные → offline adapter):

    CanonicalBatch → build_features_from_context (tabular-v1) → model.predict
        → final_prediction(target_formulation) → delay_seconds

Никакой второй feature-логики, выбора модели, clipping, калибровки,
вероятностей или reason здесь нет.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mostransport_ml.artifacts.bundle import (
    ArtifactBundleError,
    BundleDescriptor,
    VerifiedBundle,
    load_bundle,
    write_bundle,
)
from mostransport_ml.artifacts.manifest import (
    ArtifactCompatibilityError,
    ArtifactManifest,
    ManifestValidationError,
    UnsupportedArtifactSchemaError,
    validate_compatibility,
)
from mostransport_ml.features.context import CanonicalBatch, build_features_from_context
from mostransport_ml.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from mostransport_ml.inference.model_families import (
    MODEL_FAMILIES,
    LoadedModel,
    ModelFamilyError,
    ModelTaskError,
)
from mostransport_ml.target.formulation import FORMULATIONS, final_prediction


class ArtifactLoadError(RuntimeError):
    """Artifact не может обслуживать инференс. `reason` — короткий безопасный код."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PredictionOutputError(RuntimeError):
    """Модель вернула выход неверной формы или нефинитные значения."""


def _check_model_features(family_name: str, model: object) -> None:
    names = MODEL_FAMILIES[family_name].feature_names(model)
    if names is not None and list(names) != list(FEATURE_NAMES):
        raise ArtifactLoadError("model_feature_mismatch")


class ArtifactPredictor:
    """Готовый к инференсу predictor поверх проверенного Artifact Bundle v1."""

    def __init__(self, verified: VerifiedBundle, model: LoadedModel) -> None:
        self._verified = verified
        self._model = model

    @classmethod
    def load(
        cls,
        bundle_dir: str | Path,
        *,
        expected_feature_schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> ArtifactPredictor:
        """Проверить bundle и загрузить модель. Все проверки — до загрузки модели."""
        try:
            verified = load_bundle(
                bundle_dir,
                expected_feature_schema_version=expected_feature_schema_version,
                supported_model_families=MODEL_FAMILIES,
            )
        except ArtifactBundleError:
            raise ArtifactLoadError("bundle_invalid") from None
        except UnsupportedArtifactSchemaError:
            raise ArtifactLoadError("unsupported_manifest_schema") from None
        except ArtifactCompatibilityError:
            raise ArtifactLoadError("incompatible_feature_schema") from None
        except ManifestValidationError:
            raise ArtifactLoadError("manifest_invalid") from None
        manifest = verified.manifest
        if manifest.target_formulation not in FORMULATIONS:
            raise ArtifactLoadError("unsupported_target_formulation")
        try:
            model = MODEL_FAMILIES[manifest.model_family].load(verified.model_bytes)
        except ModelTaskError:
            raise ArtifactLoadError("model_task_incompatible") from None
        except ModelFamilyError:
            raise ArtifactLoadError("model_load_failed") from None
        _check_model_features(manifest.model_family, model)
        return cls(verified, model)

    # ------------------------------------------------------------------ metadata

    @property
    def manifest(self) -> ArtifactManifest:
        return self._verified.manifest

    @property
    def bundle_sha256(self) -> str:
        return self._verified.bundle_sha256

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return self.manifest.model_version

    def feature_schema_version(self) -> str:
        return self.manifest.feature_schema_version

    # ------------------------------------------------------------------ inference

    def predict(self, batch: CanonicalBatch) -> np.ndarray:
        """Итоговые `delay_seconds` для каждой точки batch, в порядке `batch.points`."""
        if not isinstance(batch, CanonicalBatch):
            raise TypeError("batch must be a CanonicalBatch")
        if not batch.points:
            return np.empty(0, dtype=float)
        features = build_features_from_context(batch)
        if tuple(features.columns) != FEATURE_NAMES:
            raise PredictionOutputError("feature columns differ from tabular-v1")
        if list(features.index) != [p.point_id for p in batch.points]:
            raise PredictionOutputError("feature rows differ from batch point order")
        raw = np.asarray(self._model.predict(features), dtype=float).reshape(-1)
        if raw.shape != (len(batch.points),):
            raise PredictionOutputError("model returned an unexpected number of outputs")
        try:
            final = final_prediction(
                raw, features["cur_dev_s"].to_numpy(), self.manifest.target_formulation
            )
        except ValueError:
            raise PredictionOutputError("model output or current deviation is not finite") from None
        return final


def export_bundle(
    directory: str | Path, model: object, manifest: ArtifactManifest
) -> BundleDescriptor:
    """Сериализовать обученную модель и записать Artifact Bundle v1.

    Проверяет, что manifest совместим с `tabular-v1`, family поддерживается,
    формулировка известна, модель — скалярная регрессия задержки (для
    `catboost`: обученный `CatBoostRegressor`, см. `model_families`) и обучена
    ровно на `FEATURE_NAMES`. Ошибка задачи модели — `ModelTaskError`.
    """
    validate_compatibility(manifest, expected_feature_schema_version=FEATURE_SCHEMA_VERSION)
    family = MODEL_FAMILIES.get(manifest.model_family)
    if family is None:
        raise ArtifactBundleError(f"model_family {manifest.model_family!r} is not supported")
    if manifest.target_formulation not in FORMULATIONS:
        raise ArtifactBundleError(f"unsupported target_formulation {manifest.target_formulation!r}")
    model_bytes = family.serialize(model)
    names = family.feature_names(model)
    if names is None or list(names) != list(FEATURE_NAMES):
        raise ArtifactBundleError("model must be trained on exactly tabular-v1 FEATURE_NAMES")
    return write_bundle(directory, manifest, model_bytes, family.model_filename)
