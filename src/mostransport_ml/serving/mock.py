"""Deterministic mock predictor.

Exists for backend/frontend integration and serving tests before a real
model exists, and stays useful afterward as a safe, dependency-free local
fixture — it is not throwaway scaffolding. Never reads transport-specific
keys out of `context`, never uses randomness, never raises on its own.
"""

from __future__ import annotations

from mostransport_ml.serving.schemas import PredictionBatchRequest, PredictionStatus
from mostransport_ml.serving.service import RawPrediction

MOCK_MODEL_VERSION = "mock-v0"
MOCK_PREDICTED_DELAY = 0.0


class MockPredictor:
    """Always-ready predictor that returns a fixed delay for every vehicle.

    Must be wired in explicitly (see `mock_app.py`) — nothing in
    `serving/app.py` defaults to this on its own.
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
