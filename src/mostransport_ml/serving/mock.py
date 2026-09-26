"""Детерминированный mock-predictor для локальной интеграции (Contract v1).

Всегда готов, возвращает фиксированную задержку для каждой точки, не
считает признаки и не использует случайность. Подключается только явно
(`mock_app.py`); реальный путь — `artifact_app.py` (Artifact Bundle v1).
"""

from __future__ import annotations

from mostransport_ml.features.context import CanonicalBatch

MOCK_MODEL_VERSION = "mock-v1"
MOCK_FEATURE_SCHEMA_VERSION = "mock"
MOCK_PREDICTED_DELAY = 0.0


class MockPredictor:
    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return MOCK_MODEL_VERSION

    def feature_schema_version(self) -> str:
        return MOCK_FEATURE_SCHEMA_VERSION

    def predict(self, batch: CanonicalBatch) -> list[float]:
        return [MOCK_PREDICTED_DELAY for _ in batch.points]
