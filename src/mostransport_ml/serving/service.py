"""Inference orchestration Contract v1 и заменяемая граница Predictor.

    FastAPI → InferenceService → runtime adapter → CanonicalBatch → Predictor
        → Contract v1 response

`InferenceService` не считает признаки и не содержит модель: он проверяет
readiness, нормализует запрос общим runtime-адаптером (тем же, что
покрыт parity-тестами с offline-путём), вызывает predictor и строго
валидирует его выход. Predictor получает неизменяемый `CanonicalBatch`,
поэтому не может подменить запрос.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol

import pandas as pd

from mostransport_ml.features.adapters import ContextValidationError, runtime_context
from mostransport_ml.features.builder import ForbiddenInputColumnError
from mostransport_ml.features.context import CanonicalBatch
from mostransport_ml.serving.schemas import PredictionV1, PredictRequestV1, PredictResponseV1


class PredictorNotReadyError(RuntimeError):
    """Инференс запрошен до готовности predictor'а."""


class PredictorContractError(RuntimeError):
    """Predictor нарушил контракт: тип readiness/версий, число или конечность выходов."""


class InvalidPredictionContextError(ValueError):
    """Запрос прошёл схему, но не может быть нормализован в канонический контекст."""


class Predictor(Protocol):
    """Граница между serving и моделью (реализации: `ArtifactPredictor`, `MockPredictor`)."""

    def is_ready(self) -> bool:
        """Ровно `bool`."""
        ...

    def model_version(self) -> str:
        """Непустая строка; вызывается только у готового predictor'а."""
        ...

    def feature_schema_version(self) -> str:
        """Непустая строка; вызывается только у готового predictor'а."""
        ...

    def predict(self, batch: CanonicalBatch) -> Sequence[float]:
        """Ровно одна итоговая задержка (секунды) на точку, в порядке `batch.points`."""
        ...


def utc_now() -> datetime:
    return datetime.now(UTC)


def _non_empty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise PredictorContractError(f"predictor.{name}() must return a non-empty str")
    return value


class InferenceService:
    def __init__(self, predictor: Predictor, clock: Callable[[], datetime] = utc_now) -> None:
        self._predictor = predictor
        self._clock = clock

    def is_ready(self) -> bool:
        value = self._predictor.is_ready()
        if type(value) is not bool:
            raise PredictorContractError("predictor.is_ready() must return exactly bool")
        return value

    def model_version(self) -> str:
        return _non_empty_str(self._predictor.model_version(), "model_version")

    def feature_schema_version(self) -> str:
        return _non_empty_str(self._predictor.feature_schema_version(), "feature_schema_version")

    def _generated_at(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise PredictorContractError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def predict(self, request: PredictRequestV1) -> PredictResponseV1:
        if not self.is_ready():
            raise PredictorNotReadyError("predictor is not ready")
        model_version = self.model_version()
        feature_schema_version = self.feature_schema_version()

        try:
            batch = runtime_context(request.model_dump(mode="json"))
        except (ContextValidationError, ForbiddenInputColumnError):
            raise InvalidPredictionContextError(
                "request cannot form a prediction context"
            ) from None

        outputs = list(self._predictor.predict(batch))
        if len(outputs) != len(batch.points):
            raise PredictorContractError("predictor returned a wrong number of predictions")
        delay = outputs[0]
        if (
            isinstance(delay, bool)
            or not isinstance(delay, int | float)
            or not math.isfinite(delay)
        ):
            raise PredictorContractError("predictor returned a non-finite or non-numeric delay")
        delay = float(delay)

        point = batch.points[0]
        target_time = (
            (point.target_time_begin + pd.Timedelta(seconds=delay))
            .round("us")
            .tz_localize("UTC")
            .to_pydatetime()
        )
        return PredictResponseV1(
            request_id=request.request_id,
            prediction=PredictionV1(delay_seconds=delay, target_time=target_time, reason=None),
            generated_at=self._generated_at(),
            model_version=model_version,
            feature_schema_version=feature_schema_version,
        )
