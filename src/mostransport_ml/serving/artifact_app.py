"""Production composition root: FastAPI + ArtifactPredictor из Artifact Bundle v1.

Запуск:

    MOSTRANSPORT_ARTIFACT_DIR=/path/to/bundle \\
        uv run uvicorn mostransport_ml.serving.artifact_app:create_app_from_env --factory

Если bundle не настроен, повреждён или несовместим, сервис всё равно
стартует, но `/ready` и `/api/v1/predict` отвечают 503; в лог пишется
только короткий код причины (без путей и текста исключений).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI

from mostransport_ml.features.context import CanonicalBatch
from mostransport_ml.inference.predictor import ArtifactLoadError, ArtifactPredictor
from mostransport_ml.serving.app import create_app
from mostransport_ml.serving.service import Predictor, PredictorNotReadyError

ARTIFACT_DIR_ENV_VAR = "MOSTRANSPORT_ARTIFACT_DIR"

logger = logging.getLogger("mostransport_ml.serving")


class UnavailablePredictor:
    """Predictor, который никогда не готов; `reason` — безопасный код причины."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def is_ready(self) -> bool:
        return False

    def model_version(self) -> str:
        raise PredictorNotReadyError(self.reason)

    def feature_schema_version(self) -> str:
        raise PredictorNotReadyError(self.reason)

    def predict(self, batch: CanonicalBatch) -> list[float]:
        raise PredictorNotReadyError(self.reason)


def load_predictor(bundle_dir: str | Path | None) -> Predictor:
    if not bundle_dir:
        logger.error("artifact_unavailable reason=artifact_not_configured")
        return UnavailablePredictor("artifact_not_configured")
    try:
        predictor = ArtifactPredictor.load(bundle_dir)
    except ArtifactLoadError as exc:
        logger.error("artifact_unavailable reason=%s", exc.reason)
        return UnavailablePredictor(exc.reason)
    except Exception as exc:
        logger.error("artifact_unavailable reason=unexpected error_type=%s", type(exc).__name__)
        return UnavailablePredictor("unexpected")
    logger.info(
        "artifact_loaded model_version=%s bundle_sha256=%s",
        predictor.model_version(),
        predictor.bundle_sha256,
    )
    return predictor


def create_artifact_app(bundle_dir: str | Path | None) -> FastAPI:
    return create_app(load_predictor(bundle_dir))


def create_app_from_env() -> FastAPI:
    return create_artifact_app(os.environ.get(ARTIFACT_DIR_ENV_VAR))
