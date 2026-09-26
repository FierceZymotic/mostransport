"""HTTP Contract v1: `POST /api/v1/predict`, `/ready`, `/health`, санитизация и OpenAPI."""

from __future__ import annotations

import copy
import json
import logging
from datetime import UTC, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from mostransport_ml.serving.app import create_app
from mostransport_ml.serving.mock import MockPredictor

FIXED_NOW = datetime(2026, 1, 6, 3, 35, 2, 500000, tzinfo=UTC)
SECRET = "SECRET_MARKER_7f3a"

REQUEST = {
    "request_id": "8b0c2f5e-4c1b-4f0e-9d1e-2a6f0c1d9e77",
    "prediction_time": "2026-01-06T03:35:00Z",
    "vehicle_context": {"unit_id": "unit-1", "tr_id": "131672", "route_id": "route-7"},
    "schedule_context": {
        "target_action_id": "53700172828",
        "target_time_begin": "2026-01-06T03:47:00Z",
        "target_lat": 55.75,
        "target_lon": 37.61,
        "current_deviation_seconds": 274,
        "manual_fill": False,
    },
    "telemetry": [
        {
            "event_time": "2026-01-06T03:34:30Z",
            "lat": 55.74,
            "lon": 37.6,
            "location_valid": True,
            "speed": 32.0,
        },
        {
            "event_time": "2026-01-06T03:35:00Z",
            "lat": 55.741,
            "lon": 37.601,
            "location_valid": True,
            "speed": 30.0,
        },
    ],
}


class FixedDelayPredictor:
    def __init__(self, delay=320.5, *, ready=True, version="model-x", schema="tabular-v1"):
        self.delay, self.ready, self.version, self.schema = delay, ready, version, schema
        self.batches = []

    def is_ready(self):
        return self.ready

    def model_version(self):
        return self.version

    def feature_schema_version(self):
        return self.schema

    def predict(self, batch):
        self.batches.append(batch)
        return [self.delay for _ in batch.points]


def client(predictor=None, clock=lambda: FIXED_NOW):
    return TestClient(create_app(predictor or FixedDelayPredictor(), clock=clock))


def post(payload, predictor=None, **kwargs):
    return client(predictor, **kwargs).post("/api/v1/predict", json=payload)


# ------------------------------------------------------------------ success


def test_canonical_success_response_exact_shape():
    response = post(REQUEST)
    assert response.status_code == 200
    assert response.json() == {
        "request_id": REQUEST["request_id"],
        "status": "success",
        "prediction": {
            "delay_seconds": 320.5,
            "target_time": "2026-01-06T03:52:20.500000Z",
            "reason": None,
        },
        "generated_at": "2026-01-06T03:35:02.500000Z",
        "model_version": "model-x",
        "feature_schema_version": "tabular-v1",
    }


def test_target_time_is_planned_plus_predicted_delay_including_negative():
    body = post(REQUEST, FixedDelayPredictor(delay=-60.0)).json()
    assert body["prediction"]["target_time"] == "2026-01-06T03:46:00Z"


def test_generated_at_is_normalized_to_utc():
    moscow = timezone(timedelta(hours=3))
    body = post(REQUEST, clock=lambda: datetime(2026, 1, 6, 6, 35, 2, tzinfo=moscow)).json()
    assert body["generated_at"] == "2026-01-06T03:35:02Z"


def test_request_reaches_predictor_as_canonical_naive_utc_batch():
    predictor = FixedDelayPredictor()
    payload = copy.deepcopy(REQUEST)
    payload["prediction_time"] = "2026-01-06T06:35:00+03:00"
    payload["telemetry"].append(
        {
            "event_time": "2026-01-06T03:35:01Z",
            "lat": 1.0,
            "lon": 1.0,
            "location_valid": True,
            "speed": 1.0,
        }
    )
    assert post(payload, predictor).status_code == 200
    batch = predictor.batches[0]
    point = batch.points[0]
    assert str(point.prediction_time) == "2026-01-06 03:35:00"
    assert (point.point_id, point.tr_id, point.unit_id, point.route_id) == (
        REQUEST["request_id"],
        "131672",
        "unit-1",
        "route-7",
    )
    assert point.current_deviation_s == 274.0
    assert len(batch.telemetry) == 2  # будущий пакет отброшен до predictor'а


def test_extra_raw_packet_fields_are_accepted_and_ignored():
    payload = copy.deepcopy(REQUEST)
    payload["telemetry"][0]["heading"] = 90
    payload["telemetry"][0]["ndtp_raw"] = {"x": 1}
    assert post(payload).status_code == 200


def test_null_gps_fields_are_accepted():
    payload = copy.deepcopy(REQUEST)
    payload["telemetry"][0].update(lat=None, lon=None, location_valid=None)
    assert post(payload).status_code == 200


def test_empty_telemetry_history_is_accepted():
    payload = copy.deepcopy(REQUEST)
    payload["telemetry"] = []
    assert post(payload).status_code == 200


def test_mock_app_path_still_works():
    response = TestClient(create_app(MockPredictor())).post("/api/v1/predict", json=REQUEST)
    assert response.status_code == 200 and response.json()["prediction"]["delay_seconds"] == 0.0


# ------------------------------------------------------------------ malformed input → sanitized 422


def mutated(path, value=None, *, delete=False):
    payload = copy.deepcopy(REQUEST)
    target = payload
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return payload


MALFORMED = [
    mutated(("request_id",), delete=True),
    mutated(("request_id",), ""),
    mutated(("prediction_time",), "2026-01-06T03:35:00"),
    mutated(("prediction_time",), 1767670500),
    mutated(("prediction_time",), "not-a-date"),
    mutated(("schedule_context", "target_time_begin"), "2026-01-06T03:47:00"),
    mutated(("telemetry", 0, "event_time"), "2026-01-06T03:34:30"),
    mutated(("vehicle_context", "tr_id"), 131672),
    mutated(("vehicle_context", "route_id"), delete=True),
    mutated(("schedule_context", "current_deviation_seconds"), "274"),
    mutated(("schedule_context", "current_deviation_seconds"), True),
    mutated(("schedule_context", "manual_fill"), "false"),
    mutated(("schedule_context", "manual_fill"), None),
    mutated(("schedule_context", "target_lat"), 91.0),
    mutated(("schedule_context", "target_lon"), None),
    mutated(("schedule_context", "target_time_begin"), "2026-01-06T03:45:00Z"),  # horizon 10
    mutated(("schedule_context", "target_time_begin"), "2026-01-06T03:50:01Z"),  # horizon > 15
    mutated(("telemetry",), delete=True),
    mutated(("telemetry", 0, "speed"), delete=True),
    mutated(("telemetry", 0, "location_valid"), "yes"),
    mutated(("telemetry", 0, "time_fact_begin"), "2026-01-06T03:40:00Z"),
    mutated(("schedule_context", "target_delay_s"), 10),
    mutated(("horizon_minutes",), 12),
    mutated(("vehicles",), []),
]


@pytest.mark.parametrize("payload", MALFORMED)
def test_malformed_requests_are_rejected_with_sanitized_422(payload):
    predictor = FixedDelayPredictor()
    response = post(payload, predictor)
    assert response.status_code == 422
    for item in response.json()["detail"]:
        assert set(item) == {"loc", "msg", "type"}
    assert predictor.batches == []


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_in_raw_body_rejected(constant):
    raw = json.dumps(REQUEST).replace(
        '"current_deviation_seconds": 274', f'"current_deviation_seconds": {constant}'
    )
    response = client().post(
        "/api/v1/predict", content=raw, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


# ------------------------------------------------------------------ wire invariant: speed


@pytest.mark.parametrize("speed", [32.0, 32, 0, 0.0, 119.5])
def test_speed_finite_number_is_accepted(speed):
    payload = mutated(("telemetry", 0, "speed"), speed)
    assert post(payload).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        mutated(("telemetry", 0, "speed"), None),
        mutated(("telemetry", 0, "speed"), delete=True),
        mutated(("telemetry", 0, "speed"), True),
        mutated(("telemetry", 0, "speed"), False),
        mutated(("telemetry", 0, "speed"), "32"),
        mutated(("telemetry", 1, "speed"), None),
    ],
    ids=["null", "missing", "true", "false", "string", "null-second-packet"],
)
def test_speed_must_be_a_finite_number(payload):
    predictor = FixedDelayPredictor()
    response = post(payload, predictor)
    assert response.status_code == 422
    assert any(item["loc"][-1] == "speed" for item in response.json()["detail"])
    assert predictor.batches == []


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_speed_non_finite_raw_json_is_rejected(constant):
    raw = json.dumps(REQUEST).replace('"speed": 32.0', f'"speed": {constant}', 1)
    assert constant in raw
    response = client().post(
        "/api/v1/predict", content=raw, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_null_gps_coordinates_remain_allowed_with_numeric_speed():
    payload = copy.deepcopy(REQUEST)
    payload["telemetry"][0].update(lat=None, lon=None, location_valid=None, speed=0.0)
    assert post(payload).status_code == 200


def test_422_never_echoes_values_or_unknown_field_names():
    payload = copy.deepcopy(REQUEST)
    payload[SECRET] = SECRET
    payload["vehicle_context"]["tr_id"] = {SECRET: SECRET}
    payload["schedule_context"]["manual_fill"] = SECRET
    response = post(payload)
    assert response.status_code == 422
    assert SECRET not in response.text
    locs = [item["loc"] for item in response.json()["detail"]]
    assert ["body", "<field>"] in locs
    assert ["body", "schedule_context", "manual_fill"] in locs


# ------------------------------------------------------------------ readiness / failures


def test_not_ready_predictor_gives_503_on_ready_and_predict():
    predictor = FixedDelayPredictor(ready=False)
    c = client(predictor)
    assert c.get("/ready").status_code == 503
    assert c.get("/ready").json() == {
        "ready": False,
        "model_version": None,
        "feature_schema_version": None,
    }
    response = c.post("/api/v1/predict", json=REQUEST)
    assert response.status_code == 503 and response.json() == {"detail": "predictor is not ready"}
    assert predictor.batches == []


def test_ready_exposes_versions():
    assert client().get("/ready").json() == {
        "ready": True,
        "model_version": "model-x",
        "feature_schema_version": "tabular-v1",
    }


@pytest.mark.parametrize(
    "predictor",
    [
        FixedDelayPredictor(ready="false"),
        FixedDelayPredictor(ready=1),
        FixedDelayPredictor(version=""),
        FixedDelayPredictor(version=None),
        FixedDelayPredictor(schema=""),
    ],
)
def test_predictor_contract_violations_fail_safely(predictor):
    c = client(predictor)
    assert c.get("/ready").status_code == 503
    assert c.post("/api/v1/predict", json=REQUEST).status_code in (500, 503)


@pytest.mark.parametrize("delay", [float("nan"), float("inf"), "320", True, None])
def test_invalid_predictor_output_gives_safe_500(delay):
    response = post(REQUEST, FixedDelayPredictor(delay=delay))
    assert response.status_code == 500
    assert response.json() == {"detail": "internal error during inference"}


def test_wrong_prediction_count_gives_safe_500():
    class TwoOutputs(FixedDelayPredictor):
        def predict(self, batch):
            return [1.0, 2.0]

    assert post(REQUEST, TwoOutputs()).status_code == 500


def test_naive_clock_gives_safe_500():
    assert post(REQUEST, clock=lambda: datetime(2026, 1, 6)).status_code == 500


def test_predictor_exception_is_sanitized_in_response_and_logs(caplog):
    class Exploding(FixedDelayPredictor):
        def predict(self, batch):
            raise RuntimeError(f"{SECRET} /home/fz/should-not-leak")

    with caplog.at_level(logging.DEBUG, logger="mostransport_ml.serving"):
        response = post(REQUEST, Exploding())
    assert response.status_code == 500
    for text in (response.text, caplog.text):
        assert SECRET not in text and "/home/fz" not in text and "Traceback" not in text
    assert "RuntimeError" in caplog.text


def test_request_payload_is_not_logged(caplog):
    payload = copy.deepcopy(REQUEST)
    payload["telemetry"][0]["raw_marker"] = SECRET
    payload["telemetry"][0]["lat"] = 55.123456789
    with caplog.at_level(logging.DEBUG, logger="mostransport_ml.serving"):
        assert post(payload).status_code == 200
    assert SECRET not in caplog.text and "55.123456789" not in caplog.text
    assert REQUEST["request_id"] not in caplog.text


def test_health():
    assert client().get("/health").json() == {"status": "ok"}


# ------------------------------------------------------------------ OpenAPI


def test_openapi_exposes_only_contract_v1_paths_and_schemas():
    spec = client().get("/openapi.json").json()
    assert sorted(spec["paths"]) == ["/api/v1/predict", "/health", "/ready"]
    predict = spec["paths"]["/api/v1/predict"]["post"]
    assert sorted(predict["responses"]) == ["200", "422", "500", "503"]
    schemas = spec["components"]["schemas"]
    assert "PredictRequestV1" in schemas and "PredictResponseV1" in schemas
    for legacy in ("PredictionBatchRequest", "VehicleRequest", "HTTPValidationError"):
        assert legacy not in schemas
    ref = predict["responses"]["422"]["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/ValidationErrorResponse")
    assert set(schemas["ErrorDetail"]["properties"]) == {"loc", "msg", "type"}
    assert schemas["PredictionV1"]["properties"]["reason"]["type"] == "null"
    packet = schemas["TelemetryPacketV1"]
    assert set(packet["required"]) == {"event_time", "lat", "lon", "location_valid", "speed"}
    assert packet["properties"]["speed"]["type"] == "number"
    assert "anyOf" not in packet["properties"]["speed"]  # не nullable
    assert "anyOf" in packet["properties"]["lat"]  # null GPS допустим
