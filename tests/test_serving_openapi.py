"""OpenAPI must match the actual runtime contract — this is what Andrey's
backend would use as the integration contract. These check specific
sections of the published schema, not a full snapshot.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from mostransport_ml.serving.app import create_app
from mostransport_ml.serving.mock import MockPredictor


def _openapi() -> dict:
    client = TestClient(create_app(MockPredictor()))
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def _resolve_schema(spec: dict, schema: dict) -> dict:
    """Follow one `$ref` (if present) into `components/schemas`."""
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return spec["components"]["schemas"][name]
    return schema


def _response_schema(spec: dict, path: str, method: str, status: str) -> dict:
    response = spec["paths"][path][method]["responses"][status]
    schema = response["content"]["application/json"]["schema"]
    return _resolve_schema(spec, schema)


def test_health_documents_success_schema():
    spec = _openapi()
    responses = spec["paths"]["/health"]["get"]["responses"]
    assert "200" in responses
    schema = _response_schema(spec, "/health", "get", "200")
    assert "status" in schema["properties"]


def test_ready_documents_200_and_503():
    spec = _openapi()
    responses = spec["paths"]["/ready"]["get"]["responses"]
    assert "200" in responses
    assert "503" in responses


def test_ready_200_and_503_schemas_match_ready_response():
    spec = _openapi()
    for status in ("200", "503"):
        schema = _response_schema(spec, "/ready", "get", status)
        assert set(schema["properties"].keys()) == {"ready", "model_version"}


def test_predict_documents_200_422_500_503():
    spec = _openapi()
    responses = spec["paths"]["/api/v1/predict/batch"]["post"]["responses"]
    assert set(responses.keys()) >= {"200", "422", "500", "503"}


def test_predict_200_schema_is_prediction_batch_response():
    spec = _openapi()
    schema = _response_schema(spec, "/api/v1/predict/batch", "post", "200")
    assert set(schema["properties"].keys()) == {
        "prediction_time",
        "horizon_minutes",
        "model_version",
        "target_name",
        "target_unit",
        "predictions",
    }


def test_predict_422_uses_sanitized_schema():
    spec = _openapi()
    schema = _response_schema(spec, "/api/v1/predict/batch", "post", "422")
    assert schema["title"] == "ValidationErrorResponse"
    assert set(schema["properties"].keys()) == {"detail"}


def test_predict_422_error_items_have_no_input_or_ctx():
    spec = _openapi()
    schema = _response_schema(spec, "/api/v1/predict/batch", "post", "422")
    error_item_schema = _resolve_schema(spec, schema["properties"]["detail"]["items"])

    properties = set(error_item_schema["properties"].keys())
    assert properties == {"loc", "msg", "type"}
    assert "input" not in properties
    assert "ctx" not in properties


def test_predict_500_and_503_use_generic_error_response():
    spec = _openapi()
    for status in ("500", "503"):
        schema = _response_schema(spec, "/api/v1/predict/batch", "post", status)
        assert schema["title"] == "ErrorResponse"
        assert set(schema["properties"].keys()) == {"detail"}


def test_no_default_fastapi_validation_error_schema_is_published():
    """The custom 422 handler fully replaces FastAPI's default
    HTTPValidationError/ValidationError components — they must not appear
    at all, since they would advertise the unsanitized `input`/`ctx` shape."""
    spec = _openapi()
    schema_names = set(spec["components"]["schemas"].keys())
    assert "HTTPValidationError" not in schema_names
    assert "ValidationError" not in schema_names
