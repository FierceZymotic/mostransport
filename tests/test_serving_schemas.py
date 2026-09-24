"""Тесты инвариантов на уровне схемы: status <-> predicted_delay, границы horizon.

Конструируют Pydantic-модели напрямую (без HTTP-слоя) — быстрое,
детерминированное покрытие инвариантов, заданных в `serving/schemas.py`.
HTTP-level аналоги — в `tests/test_serving_prediction.py`, включая
случаи (NaN/Infinity в сыром JSON запроса), где поведение на уровне
клиента и на уровне HTTP по-настоящему различается.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mostransport_ml.serving.schemas import (
    PredictionBatchRequest,
    PredictionStatus,
    VehiclePrediction,
)


def _base_request(**overrides):
    payload = {
        "prediction_time": "2026-01-01T00:00:00Z",
        "vehicles": [{"vehicle_id": "x", "context": {}}],
    }
    payload.update(overrides)
    return payload


# --- инвариант status <-> predicted_delay ------------------------------------


def test_ok_with_finite_delay_is_valid():
    prediction = VehiclePrediction(vehicle_id="x", predicted_delay=1.5, status=PredictionStatus.OK)
    assert prediction.predicted_delay == 1.5


def test_ok_with_none_delay_is_rejected():
    with pytest.raises(ValidationError):
        VehiclePrediction(vehicle_id="x", predicted_delay=None, status=PredictionStatus.OK)


def test_insufficient_data_with_none_delay_is_valid():
    prediction = VehiclePrediction(
        vehicle_id="x", predicted_delay=None, status=PredictionStatus.INSUFFICIENT_DATA
    )
    assert prediction.predicted_delay is None


def test_insufficient_data_with_number_is_rejected():
    with pytest.raises(ValidationError):
        VehiclePrediction(
            vehicle_id="x", predicted_delay=1.0, status=PredictionStatus.INSUFFICIENT_DATA
        )


def test_error_status_with_none_delay_is_valid():
    prediction = VehiclePrediction(
        vehicle_id="x", predicted_delay=None, status=PredictionStatus.ERROR
    )
    assert prediction.predicted_delay is None


def test_error_status_with_number_is_rejected():
    with pytest.raises(ValidationError):
        VehiclePrediction(vehicle_id="x", predicted_delay=1.0, status=PredictionStatus.ERROR)


def test_ok_with_nan_delay_is_rejected():
    with pytest.raises(ValidationError):
        VehiclePrediction(vehicle_id="x", predicted_delay=float("nan"), status=PredictionStatus.OK)


def test_ok_with_infinite_delay_is_rejected():
    with pytest.raises(ValidationError):
        VehiclePrediction(vehicle_id="x", predicted_delay=float("inf"), status=PredictionStatus.OK)


# --- horizon_minutes: None, либо конечное и > 0 -------------------------------


def test_horizon_none_is_valid():
    request = PredictionBatchRequest(**_base_request())
    assert request.horizon_minutes is None


def test_horizon_positive_is_valid():
    request = PredictionBatchRequest(**_base_request(horizon_minutes=15))
    assert request.horizon_minutes == 15


def test_horizon_zero_is_rejected():
    with pytest.raises(ValidationError):
        PredictionBatchRequest(**_base_request(horizon_minutes=0))


def test_horizon_negative_is_rejected():
    with pytest.raises(ValidationError):
        PredictionBatchRequest(**_base_request(horizon_minutes=-1))


def test_horizon_nan_is_rejected():
    with pytest.raises(ValidationError):
        PredictionBatchRequest(**_base_request(horizon_minutes=float("nan")))


def test_horizon_positive_infinity_is_rejected():
    with pytest.raises(ValidationError):
        PredictionBatchRequest(**_base_request(horizon_minutes=float("inf")))


def test_horizon_negative_infinity_is_rejected():
    with pytest.raises(ValidationError):
        PredictionBatchRequest(**_base_request(horizon_minutes=float("-inf")))
