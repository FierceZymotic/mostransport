"""Тесты безопасной обработки ошибок: исключения predictor'а и его
невалидный output никогда не должны утечь traceback'ом, эхом входа
запроса или дойти до клиента как некорректный JSON — и никогда не должны
появиться в server-side логах."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from mostransport_ml.serving.app import create_app
from mostransport_ml.serving.schemas import PredictionBatchRequest, PredictionStatus
from mostransport_ml.serving.service import RawPrediction


class _ExplodingPredictor:
    """Fake predictor, всегда бросающий исключение — проверка 5xx safety net."""

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "exploding-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        raise RuntimeError("simulated predictor crash: /home/fz/should-not-leak")


class _FixedOutputPredictor:
    """Fake predictor, возвращающий фиксированную (возможно нефинитную) задержку."""

    def __init__(self, predicted_delay: float) -> None:
        self._predicted_delay = predicted_delay

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "fixed-output-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        return [
            RawPrediction(
                vehicle_id=vehicle.vehicle_id,
                predicted_delay=self._predicted_delay,
                status=PredictionStatus.OK,
            )
            for vehicle in request.vehicles
        ]


class _ReorderingPredictor:
    """Fake predictor, возвращающий корректные predictions, но в другом порядке."""

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "reordering-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        return [
            RawPrediction(
                vehicle_id=vehicle.vehicle_id, predicted_delay=0.0, status=PredictionStatus.OK
            )
            for vehicle in reversed(request.vehicles)
        ]


class _WrongIdPredictor:
    """Fake predictor с верным count/order, но неверным id."""

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "wrong-id-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        predictions = [
            RawPrediction(
                vehicle_id=vehicle.vehicle_id, predicted_delay=0.0, status=PredictionStatus.OK
            )
            for vehicle in request.vehicles
        ]
        predictions[-1] = RawPrediction(
            vehicle_id="wrong-id", predicted_delay=0.0, status=PredictionStatus.OK
        )
        return predictions


class _WrongCountPredictor:
    """Fake predictor, возвращающий слишком много (extra > 0) или слишком
    мало (extra < 0) predictions относительно входного batch'а."""

    def __init__(self, extra: int) -> None:
        self._extra = extra

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "wrong-count-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        predictions = [
            RawPrediction(
                vehicle_id=vehicle.vehicle_id, predicted_delay=0.0, status=PredictionStatus.OK
            )
            for vehicle in request.vehicles
        ]
        if self._extra > 0:
            predictions.extend(
                RawPrediction(
                    vehicle_id=f"extra-{i}", predicted_delay=0.0, status=PredictionStatus.OK
                )
                for i in range(self._extra)
            )
        elif self._extra < 0:
            predictions = predictions[: self._extra]
        return predictions


class _ClearingPredictor:
    """Fake predictor, очищающий `request.vehicles` на месте и возвращающий
    соответствующий (но теперь бессмысленный) пустой список predictions."""

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "clearing-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        request.vehicles.clear()
        return []


class _IdRewritingPredictor:
    """Fake predictor, переписывающий id vehicle прямо в самом request'е, а
    затем (нечестно) возвращающий prediction, совпадающий с его же правкой."""

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "id-rewriting-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        request.vehicles[0].vehicle_id = "rewritten-id"
        return [
            RawPrediction(
                vehicle_id=vehicle.vehicle_id, predicted_delay=0.0, status=PredictionStatus.OK
            )
            for vehicle in request.vehicles
        ]


class _OrderMutatingPredictor:
    """Fake predictor, разворачивающий `request.vehicles` на месте, а затем
    (последовательно, со своей точки зрения) возвращающий соответствующие
    predictions."""

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "order-mutating-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        request.vehicles.reverse()
        return [
            RawPrediction(
                vehicle_id=vehicle.vehicle_id, predicted_delay=0.0, status=PredictionStatus.OK
            )
            for vehicle in request.vehicles
        ]


class _PredictionTimeRewritingPredictor:
    """Fake predictor, переписывающий `request.prediction_time` на месте."""

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "time-rewriting-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        request.prediction_time = datetime(2099, 1, 1, tzinfo=UTC)
        return [
            RawPrediction(
                vehicle_id=vehicle.vehicle_id, predicted_delay=0.0, status=PredictionStatus.OK
            )
            for vehicle in request.vehicles
        ]


class _HorizonRewritingPredictor:
    """Fake predictor, переписывающий `request.horizon_minutes` на месте."""

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "horizon-rewriting-v0"

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        request.horizon_minutes = 999
        return [
            RawPrediction(
                vehicle_id=vehicle.vehicle_id, predicted_delay=0.0, status=PredictionStatus.OK
            )
            for vehicle in request.vehicles
        ]


def _payload(*vehicle_ids: str):
    ids = vehicle_ids or ("synthetic-1",)
    return {
        "prediction_time": "2026-01-01T00:00:00Z",
        "vehicles": [{"vehicle_id": vid, "context": {"synthetic": True}} for vid in ids],
    }


def test_predictor_exception_returns_safe_5xx_without_traceback_or_input_echo():
    client = TestClient(create_app(_ExplodingPredictor()))
    response = client.post("/api/v1/predict/batch", json=_payload())

    assert 500 <= response.status_code < 600
    body_text = response.text
    assert "Traceback" not in body_text
    assert "/home/fz/should-not-leak" not in body_text
    assert "synthetic" not in body_text  # содержимое запроса (id/context) не эхо́ится


def test_predictor_exception_secret_message_absent_from_response_and_logs(caplog):
    secret = "SECRET_PREDICTOR_123"

    class _SecretExplodingPredictor:
        def is_ready(self) -> bool:
            return True

        def model_version(self) -> str:
            return "secret-exploding-v0"

        def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
            raise RuntimeError(secret)

    client = TestClient(create_app(_SecretExplodingPredictor()))
    with caplog.at_level(logging.DEBUG, logger="mostransport_ml.serving"):
        response = client.post("/api/v1/predict/batch", json=_payload())

    assert 500 <= response.status_code < 600
    assert secret not in response.text
    assert secret not in caplog.text
    # ТИП ошибки по-прежнему безопасно логировать — он не несёт данных запроса.
    assert "RuntimeError" in caplog.text


def test_prediction_order_mismatch_returns_safe_500():
    client = TestClient(create_app(_ReorderingPredictor()))
    response = client.post("/api/v1/predict/batch", json=_payload("a", "b"))

    assert 500 <= response.status_code < 600
    assert "predictor produced invalid output" in response.text


def test_prediction_id_mismatch_returns_safe_500():
    client = TestClient(create_app(_WrongIdPredictor()))
    response = client.post("/api/v1/predict/batch", json=_payload("a", "b"))

    assert 500 <= response.status_code < 600
    assert "predictor produced invalid output" in response.text


def test_prediction_excess_count_returns_safe_500():
    client = TestClient(create_app(_WrongCountPredictor(extra=1)))
    response = client.post("/api/v1/predict/batch", json=_payload("a", "b"))

    assert 500 <= response.status_code < 600


def test_prediction_missing_count_returns_safe_500():
    client = TestClient(create_app(_WrongCountPredictor(extra=-1)))
    response = client.post("/api/v1/predict/batch", json=_payload("a", "b"))

    assert 500 <= response.status_code < 600


def test_prediction_exact_correspondence_still_returns_200():
    """Проверка контракта не должна отклонять predictor, который делает всё правильно."""
    client = TestClient(create_app(_FixedOutputPredictor(0.0)))
    response = client.post("/api/v1/predict/batch", json=_payload("a", "b"))

    assert response.status_code == 200
    assert [p["vehicle_id"] for p in response.json()["predictions"]] == ["a", "b"]


def test_predictor_nan_output_returns_safe_5xx_with_strict_json():
    client = TestClient(create_app(_FixedOutputPredictor(float("nan"))))
    response = client.post("/api/v1/predict/batch", json=_payload())

    assert 500 <= response.status_code < 600
    json.loads(response.text)  # обязан быть стандартным, парсящимся JSON — без голого токена NaN
    assert "NaN" not in response.text


def test_predictor_positive_infinity_output_returns_safe_5xx():
    client = TestClient(create_app(_FixedOutputPredictor(float("inf"))))
    response = client.post("/api/v1/predict/batch", json=_payload())

    assert 500 <= response.status_code < 600
    json.loads(response.text)
    assert "Infinity" not in response.text


def test_predictor_negative_infinity_output_returns_safe_5xx():
    client = TestClient(create_app(_FixedOutputPredictor(float("-inf"))))
    response = client.post("/api/v1/predict/batch", json=_payload())

    assert 500 <= response.status_code < 600
    json.loads(response.text)
    assert "Infinity" not in response.text


# --- request-mutation protection: predictor не должен суметь задним числом
# переопределить, чем был "request" ---------------------------------


def test_predictor_clearing_vehicles_returns_safe_500_not_empty_200():
    client = TestClient(create_app(_ClearingPredictor()))
    response = client.post("/api/v1/predict/batch", json=_payload("a"))

    # Уязвимое поведение выглядело бы как 200 с predictions=[].
    assert 500 <= response.status_code < 600
    assert "predictor produced invalid output" in response.text


def test_predictor_rewriting_vehicle_id_returns_safe_500():
    client = TestClient(create_app(_IdRewritingPredictor()))
    response = client.post("/api/v1/predict/batch", json=_payload("a"))

    assert 500 <= response.status_code < 600
    assert "rewritten-id" not in response.text


def test_predictor_mutating_vehicle_order_returns_safe_500():
    client = TestClient(create_app(_OrderMutatingPredictor()))
    response = client.post("/api/v1/predict/batch", json=_payload("a", "b"))

    assert 500 <= response.status_code < 600


def test_predictor_rewriting_prediction_time_returns_safe_500():
    client = TestClient(create_app(_PredictionTimeRewritingPredictor()))
    response = client.post("/api/v1/predict/batch", json=_payload("a"))

    assert 500 <= response.status_code < 600
    assert "2099" not in response.text


def test_predictor_rewriting_horizon_minutes_returns_safe_500():
    client = TestClient(create_app(_HorizonRewritingPredictor()))
    response = client.post("/api/v1/predict/batch", json=_payload("a") | {"horizon_minutes": 15})

    assert 500 <= response.status_code < 600
    assert "999" not in response.text


def test_non_mutating_predictor_still_returns_200():
    """Проверка мутации не должна отклонять predictor, ведущий себя корректно."""
    payload = _payload("a", "b") | {"horizon_minutes": 15}
    client = TestClient(create_app(_FixedOutputPredictor(0.0)))
    response = client.post("/api/v1/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_time"] == "2026-01-01T00:00:00Z"
    assert body["horizon_minutes"] == 15.0


def test_response_uses_original_request_envelope_not_mutated_values():
    """Даже в случаях мутации то, что *действительно* доходит до клиента
    (если вообще доходит), никогда не должно нести переписанные predictor'ом
    значения — здесь это подтверждается через safe-failure путь, поскольку
    такие predictor'ы отклоняются целиком, вместо того чтобы вообще
    сформировать ответ."""
    client = TestClient(create_app(_PredictionTimeRewritingPredictor()))
    response = client.post("/api/v1/predict/batch", json=_payload("a") | {"horizon_minutes": 15})
    assert 500 <= response.status_code < 600
    # Ни одно поле тела ответа не могло утечь мутированный конверт, потому
    # что успешного ответа вообще не существует.
    assert response.json() == {"detail": "predictor produced invalid output"}
