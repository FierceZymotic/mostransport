"""Tests for GET /health, GET /ready, and basic OpenAPI shape."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from mostransport_ml.serving.app import create_app
from mostransport_ml.serving.mock import MOCK_MODEL_VERSION, MockPredictor
from mostransport_ml.serving.schemas import PredictionBatchRequest


class _NotReadyPredictor:
    """Fake predictor that is never ready, to exercise /ready's 503 path."""

    def is_ready(self) -> bool:
        return False

    def model_version(self) -> str:
        return "not-ready-v0"

    def predict_batch(self, request: PredictionBatchRequest):
        raise AssertionError("predict_batch must not be called while not ready")


class _ReadyCheckRaisesPredictor:
    """Fake predictor whose `is_ready()` itself raises."""

    def __init__(self, secret: str) -> None:
        self._secret = secret

    def is_ready(self) -> bool:
        raise RuntimeError(self._secret)

    def model_version(self) -> str:
        return "ready-check-raises-v0"

    def predict_batch(self, request: PredictionBatchRequest):
        raise AssertionError("predict_batch must not be called")


class _ModelVersionRaisesPredictor:
    """Fake predictor that reports ready but whose `model_version()` raises."""

    def __init__(self, secret: str) -> None:
        self._secret = secret

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        raise RuntimeError(self._secret)

    def predict_batch(self, request: PredictionBatchRequest):
        raise AssertionError("predict_batch must not be called")


class _NonBoolReadyPredictor:
    """Fake predictor whose `is_ready()` returns something other than `bool`
    — e.g. a truthy string, which `bool(...)` would silently accept."""

    def __init__(self, value: object) -> None:
        self._value = value

    def is_ready(self) -> bool:
        return self._value  # type: ignore[return-value]

    def model_version(self) -> str:
        return "non-bool-ready-v0"

    def predict_batch(self, request: PredictionBatchRequest):
        raise AssertionError("predict_batch must not be called")


class _InvalidModelVersionPredictor:
    """Fake predictor that reports ready but whose `model_version()` returns
    something other than a non-empty `str`."""

    def __init__(self, value: object) -> None:
        self._value = value

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return self._value  # type: ignore[return-value]

    def predict_batch(self, request: PredictionBatchRequest):
        raise AssertionError("predict_batch must not be called")


def test_health_returns_200_and_valid_json():
    client = TestClient(create_app(MockPredictor()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_is_true_for_mock_predictor():
    client = TestClient(create_app(MockPredictor()))
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["model_version"] == MOCK_MODEL_VERSION


def test_ready_returns_503_when_predictor_not_ready():
    client = TestClient(create_app(_NotReadyPredictor()))
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False


def test_ready_returns_503_when_is_ready_raises(caplog):
    secret = "SECRET_READY_456"
    client = TestClient(create_app(_ReadyCheckRaisesPredictor(secret)))

    with caplog.at_level(logging.DEBUG, logger="mostransport_ml.serving"):
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body == {"ready": False, "model_version": None}
    assert secret not in response.text
    assert secret not in caplog.text


def test_ready_returns_503_when_model_version_raises(caplog):
    secret = "SECRET_MODEL_VERSION_789"
    client = TestClient(create_app(_ModelVersionRaisesPredictor(secret)))

    with caplog.at_level(logging.DEBUG, logger="mostransport_ml.serving"):
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body == {"ready": False, "model_version": None}
    assert secret not in response.text
    assert secret not in caplog.text


# --- runtime predictor contract: is_ready() must be exactly bool,
# model_version() must be a non-empty str when ready ------------------------


def test_is_ready_string_false_is_rejected_not_treated_as_ready():
    """A truthy string like "false" must not be silently accepted via
    `bool(...)` — that would make /ready say "ready" while nothing backs it."""
    client = TestClient(create_app(_NonBoolReadyPredictor("false")))
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"ready": False, "model_version": None}


def test_is_ready_string_false_also_fails_prediction_path():
    client = TestClient(create_app(_NonBoolReadyPredictor("false")))
    payload = {
        "prediction_time": "2026-01-01T00:00:00Z",
        "vehicles": [{"vehicle_id": "x", "context": {}}],
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert 500 <= response.status_code < 600


def test_is_ready_int_one_is_rejected():
    client = TestClient(create_app(_NonBoolReadyPredictor(1)))
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"ready": False, "model_version": None}


def test_model_version_none_while_ready_is_rejected():
    client = TestClient(create_app(_InvalidModelVersionPredictor(None)))
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"ready": False, "model_version": None}


def test_model_version_integer_is_rejected():
    client = TestClient(create_app(_InvalidModelVersionPredictor(123)))
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"ready": False, "model_version": None}


def test_model_version_empty_string_is_rejected():
    client = TestClient(create_app(_InvalidModelVersionPredictor("")))
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"ready": False, "model_version": None}


def test_model_version_valid_non_empty_string_works():
    client = TestClient(create_app(_InvalidModelVersionPredictor("real-v1")))
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True, "model_version": "real-v1"}


def test_invalid_predictor_contract_secret_absent_from_response_and_logs(caplog):
    """Even the invalid VALUE a broken predictor returns must not leak — the
    contract-violation exception message embeds it, but only the exception
    TYPE is ever logged, never its message."""
    secret = "SECRET_READY_VALUE_999"
    client = TestClient(create_app(_NonBoolReadyPredictor(secret)))

    with caplog.at_level(logging.DEBUG, logger="mostransport_ml.serving"):
        response = client.get("/ready")

    assert response.status_code == 503
    assert secret not in response.text
    assert secret not in caplog.text


def test_openapi_lists_expected_paths():
    client = TestClient(create_app(MockPredictor()))
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/ready" in paths
    assert "/api/v1/predict/batch" in paths
