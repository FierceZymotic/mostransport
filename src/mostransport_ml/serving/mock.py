"""Детерминированный mock-predictor.

Существует для интеграции с backend/frontend и serving-тестов до
появления реальной модели, и остаётся полезным после — как безопасная,
не требующая зависимостей локальная fixture, а не одноразовый scaffolding.
Никогда не читает transport-specific ключи из `context`, никогда не
использует случайность, никогда не бросает исключение по собственной
инициативе.
"""

from __future__ import annotations

from mostransport_ml.serving.schemas import PredictionBatchRequest, PredictionStatus
from mostransport_ml.serving.service import RawPrediction

MOCK_MODEL_VERSION = "mock-v0"
MOCK_PREDICTED_DELAY = 0.0


class MockPredictor:
    """Всегда готовый predictor, возвращающий фиксированную задержку для
    каждого vehicle.

    Должен подключаться только явно (см. `mock_app.py`) — ничто в
    `serving/app.py` не выбирает его по умолчанию.
    """

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return MOCK_MODEL_VERSION

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        return [
            RawPrediction(
                vehicle_id=vehicle.vehicle_id,
                predicted_delay=MOCK_PREDICTED_DELAY,
                status=PredictionStatus.OK,
            )
            for vehicle in request.vehicles
        ]
