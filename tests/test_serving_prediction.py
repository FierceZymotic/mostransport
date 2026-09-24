"""Тесты для POST /api/v1/predict/batch: happy path и валидация запроса.

В test-payload'ах используются только синтетические, generic имена полей
(например, `vehicle_id`, `context: {"synthetic": True}`) — ничто здесь не
выдумывает и не предполагает реальную organizer/transport-схему.
"""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from mostransport_ml.serving.app import create_app
from mostransport_ml.serving.mock import MOCK_MODEL_VERSION, MOCK_PREDICTED_DELAY, MockPredictor


def _client() -> TestClient:
    return TestClient(create_app(MockPredictor()))


def _valid_payload(**overrides):
    payload = {
        "prediction_time": "2026-01-01T00:00:00Z",
        "horizon_minutes": 15,
        "vehicles": [
            {"vehicle_id": "synthetic-1", "context": {"synthetic": True}},
            {"vehicle_id": "synthetic-2", "context": {"synthetic": True}},
        ],
    }
    payload.update(overrides)
    return payload


def test_batch_prediction_happy_path():
    response = _client().post("/api/v1/predict/batch", json=_valid_payload())
    assert response.status_code == 200
    body = response.json()

    assert body["model_version"] == MOCK_MODEL_VERSION
    assert len(body["predictions"]) == 2
    # Порядок входа и vehicle_id сохраняются.
    assert [p["vehicle_id"] for p in body["predictions"]] == ["synthetic-1", "synthetic-2"]
    for prediction in body["predictions"]:
        assert prediction["status"] == "ok"
        assert prediction["predicted_delay"] == MOCK_PREDICTED_DELAY
        assert math.isfinite(prediction["predicted_delay"])


def test_horizon_minutes_is_optional():
    payload = _valid_payload()
    del payload["horizon_minutes"]
    response = _client().post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    assert response.json()["horizon_minutes"] is None


def test_single_vehicle_batch_accepted():
    payload = _valid_payload(vehicles=[{"vehicle_id": "synthetic-1", "context": {}}])
    response = _client().post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_empty_vehicles_rejected():
    response = _client().post("/api/v1/predict/batch", json=_valid_payload(vehicles=[]))
    assert response.status_code == 422


def test_non_object_context_rejected():
    payload = _valid_payload()
    payload["vehicles"][0]["context"] = "not-an-object"
    response = _client().post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 422


def test_missing_context_rejected():
    payload = _valid_payload()
    del payload["vehicles"][0]["context"]
    response = _client().post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 422


def test_unknown_top_level_field_rejected():
    payload = _valid_payload()
    payload["unexpected_field"] = "nope"
    response = _client().post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 422


def test_unknown_vehicle_field_rejected():
    payload = _valid_payload()
    payload["vehicles"][0]["unexpected_field"] = "nope"
    response = _client().post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 422


# --- horizon_minutes: None либо конечное число > 0, через реальный HTTP -----


def test_horizon_zero_rejected_over_http():
    response = _client().post("/api/v1/predict/batch", json=_valid_payload(horizon_minutes=0))
    assert response.status_code == 422


def test_horizon_negative_rejected_over_http():
    response = _client().post("/api/v1/predict/batch", json=_valid_payload(horizon_minutes=-1))
    assert response.status_code == 422


def _post_raw_json(client: TestClient, raw_body: str):
    """Отправить вручную собранную JSON-строку, минуя собственный JSON-энкодер
    test client'а. `TestClient(...).post(json=...)` отказывается даже
    сконструировать запрос с NaN/Infinity (его энкодер бросает исключение на
    стороне клиента ещё до отправки) — поэтому единственный способ
    проверить, как сервер сам обрабатывает буквальные токены NaN/Infinity в
    теле запроса, — отправить сырые байты напрямую, как это сделал бы
    не-Python клиент."""
    return client.post(
        "/api/v1/predict/batch", content=raw_body, headers={"Content-Type": "application/json"}
    )


def test_horizon_nan_in_raw_request_body_is_rejected_not_500():
    raw_body = (
        '{"prediction_time": "2026-01-01T00:00:00Z", "horizon_minutes": NaN, '
        '"vehicles": [{"vehicle_id": "x", "context": {}}]}'
    )
    response = _post_raw_json(_client(), raw_body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "finite_number"


def test_horizon_positive_infinity_in_raw_request_body_is_rejected_not_500():
    raw_body = (
        '{"prediction_time": "2026-01-01T00:00:00Z", "horizon_minutes": Infinity, '
        '"vehicles": [{"vehicle_id": "x", "context": {}}]}'
    )
    response = _post_raw_json(_client(), raw_body)
    assert response.status_code == 422


def test_horizon_negative_infinity_in_raw_request_body_is_rejected_not_500():
    raw_body = (
        '{"prediction_time": "2026-01-01T00:00:00Z", "horizon_minutes": -Infinity, '
        '"vehicles": [{"vehicle_id": "x", "context": {}}]}'
    )
    response = _post_raw_json(_client(), raw_body)
    assert response.status_code == 422


# --- sanitized 422: невалидные значения запроса никогда не эхо́ятся обратно --


def test_sanitized_422_does_not_echo_invalid_context_value():
    payload = _valid_payload()
    payload["vehicles"][0]["context"] = "VERY_SECRET_CONTEXT_VALUE"
    response = _client().post("/api/v1/predict/batch", json=payload)

    assert response.status_code == 422
    assert "VERY_SECRET_CONTEXT_VALUE" not in response.text
    # Всё ещё достаточно информативно для отладки на стороне backend'а.
    error = response.json()["detail"][0]
    assert set(error.keys()) == {"loc", "msg", "type"}
    assert "context" in error["loc"]


def test_sanitized_422_does_not_echo_unknown_field_value():
    payload = _valid_payload()
    payload["unexpected_field"] = "ANOTHER_SECRET_TOPLEVEL_VALUE"
    response = _client().post("/api/v1/predict/batch", json=payload)

    assert response.status_code == 422
    assert "ANOTHER_SECRET_TOPLEVEL_VALUE" not in response.text


# --- sanitized 422 `loc`: неизвестное ИМЯ ПОЛЯ — тоже данные запроса,
# а не только проблемное значение --------------------------------------------


def test_sanitized_422_redacts_unknown_top_level_field_name_in_loc():
    payload = _valid_payload()
    payload["SECRET_FIELD_NAME_123"] = "anything"
    response = _client().post("/api/v1/predict/batch", json=payload)

    assert response.status_code == 422
    assert "SECRET_FIELD_NAME_123" not in response.text
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "<field>"]


def test_sanitized_422_redacts_unknown_vehicle_field_name_in_loc():
    payload = _valid_payload()
    payload["vehicles"][0]["SECRET_VEHICLE_FIELD_456"] = "anything"
    response = _client().post("/api/v1/predict/batch", json=payload)

    assert response.status_code == 422
    assert "SECRET_VEHICLE_FIELD_456" not in response.text
    error = response.json()["detail"][0]
    # Индекс списка (структурный маркер, не данные запроса) сохраняется;
    # редактируется только неизвестное имя поля.
    assert error["loc"] == ["body", "vehicles", 0, "<field>"]


def test_sanitized_422_keeps_known_field_name_useful_for_debugging():
    response = _client().post("/api/v1/predict/batch", json=_valid_payload(horizon_minutes=0))

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert "horizon_minutes" in error["loc"]


def test_sanitized_422_body_never_contains_any_secret_marker():
    payload = _valid_payload()
    payload["SECRET_FIELD_NAME_123"] = "anything"
    payload["vehicles"][0]["context"] = "SECRET_CONTEXT_VALUE"
    response = _client().post("/api/v1/predict/batch", json=payload)

    assert response.status_code == 422
    assert "SECRET_FIELD_NAME_123" not in response.text
    assert "SECRET_CONTEXT_VALUE" not in response.text
