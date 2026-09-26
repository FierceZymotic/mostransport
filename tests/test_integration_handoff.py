"""Backend handoff: integration-only artifact, Contract v1 smoke, замена artifact через env.

INTEGRATION TEST ONLY: `integration-fixture-v1` — синтетическая модель для
проверки HTTP-интеграции, не для submission и не модель качества.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from mostransport_ml.artifacts.bundle import BUNDLE_FILENAME, MANIFEST_FILENAME
from mostransport_ml.data.official import DATASET_ENV_VAR
from mostransport_ml.features.adapters import runtime_context
from mostransport_ml.inference.predictor import ArtifactPredictor
from mostransport_ml.serving import artifact_app
from mostransport_ml.serving.app import create_app
from mostransport_ml.serving.mock import MockPredictor

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "contract_v1_request.json"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_integration_artifact")
smoke = load_script("smoke_contract_v1")


@pytest.fixture(scope="module")
def integration_bundle(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("integration") / "artifact"
    builder.build_integration_artifact(directory)
    return directory


def client_fetch(client: TestClient):
    def fetch(method, path, body):
        response = (
            client.request(method, path, json=body)
            if body is not None
            else client.request(method, path)
        )
        return response.status_code, response.json()

    return fetch


# ------------------------------------------------------------------ integration artifact


def test_integration_artifact_is_a_valid_labelled_regression_bundle(integration_bundle):
    predictor = ArtifactPredictor.load(integration_bundle)
    manifest = predictor.manifest
    assert predictor.is_ready() is True
    assert manifest.model_version == "integration-fixture-v1"
    assert (manifest.model_family, manifest.feature_schema_version) == ("catboost", "tabular-v1")
    assert manifest.target_formulation == "direct"
    assert manifest.train_regime == "integration_fixture_synthetic"
    assert "NOT FOR SUBMISSION" in manifest.training_data_provenance["purpose"]
    assert manifest.training_data_provenance["source"] == "synthetic"
    batch = runtime_context(json.loads(FIXTURE.read_text(encoding="utf-8")))
    delay = predictor.predict(batch)
    assert delay.shape == (1,) and np.isfinite(delay).all()


def test_integration_artifact_is_byte_deterministic(tmp_path, integration_bundle):
    builder.build_integration_artifact(tmp_path / "again")
    for name in (BUNDLE_FILENAME, MANIFEST_FILENAME, "model.cbm"):
        assert (tmp_path / "again" / name).read_bytes() == (integration_bundle / name).read_bytes()


def test_integration_artifact_output_guards(tmp_path, monkeypatch):
    target = tmp_path / "artifact"
    assert builder.main(["--output", str(target)]) == 0
    assert builder.main(["--output", str(target)]) == 2  # занято, без --replace
    assert builder.main(["--output", str(target), "--replace"]) == 0

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me")
    assert builder.main(["--output", str(foreign), "--replace"]) == 2
    assert (foreign / "notes.txt").read_text() == "keep me"

    monkeypatch.setenv(DATASET_ENV_VAR, str(tmp_path / "dataset"))
    assert builder.main(["--output", str(tmp_path / "dataset" / "artifact")]) == 2
    assert not (tmp_path / "dataset" / "artifact").exists()


# ------------------------------------------------------------------ Contract v1 smoke


def test_smoke_passes_against_integration_artifact_service(integration_bundle, capsys):
    client = TestClient(artifact_app.create_artifact_app(integration_bundle))
    assert smoke.main([], fetch=client_fetch(client)) == 0
    output = capsys.readouterr().out
    assert "model_version=integration-fixture-v1" in output
    assert "contract v1 smoke: OK" in output


def test_smoke_fails_when_service_is_not_ready_or_wrong_schema(tmp_path):
    not_ready = TestClient(artifact_app.create_artifact_app(None))
    assert smoke.main([], fetch=client_fetch(not_ready)) == 1
    mock = TestClient(create_app(MockPredictor()))  # feature_schema_version = "mock"
    assert smoke.main([], fetch=client_fetch(mock)) == 1


def test_smoke_detects_broken_response_semantics():
    request = json.loads(FIXTURE.read_text(encoding="utf-8"))
    good = {
        "request_id": request["request_id"],
        "status": "success",
        "prediction": {
            "delay_seconds": 60.0,
            "target_time": "2026-01-06T03:48:00Z",
            "reason": None,
        },
        "generated_at": "2026-01-06T03:35:01Z",
        "model_version": "m",
        "feature_schema_version": "tabular-v1",
    }
    assert smoke.check_prediction(request, 200, good, expected_model="m") == 60.0
    broken = [
        {**good, "request_id": "other"},
        {**good, "prediction": {**good["prediction"], "reason": "x"}},
        {**good, "prediction": {**good["prediction"], "target_time": "2026-01-06T03:47:00Z"}},
        {**good, "generated_at": "2026-01-06T03:35:01"},
        {**good, "feature_schema_version": "tabular-v2"},
        {**good, "prediction": {**good["prediction"], "delay_seconds": True}},
    ]
    for body in broken:
        with pytest.raises(smoke.SmokeFailure):
            smoke.check_prediction(request, 200, body, expected_model="m")


def test_live_http_smoke_against_uvicorn(integration_bundle):
    server = uvicorn.Server(
        uvicorn.Config(
            artifact_app.create_artifact_app(integration_bundle),
            host="127.0.0.1",
            port=0,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        assert smoke.main(["--base-url", f"http://127.0.0.1:{port}"]) == 0
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ------------------------------------------ artifact replacement = env var only


def test_replacing_artifact_needs_only_the_env_var(
    monkeypatch, integration_bundle, catboost_bundle
):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    responses = []
    for bundle in (integration_bundle, catboost_bundle(target_formulation="residual")):
        monkeypatch.setenv(artifact_app.ARTIFACT_DIR_ENV_VAR, str(bundle))
        client = TestClient(artifact_app.create_app_from_env())
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        responses.append(response.json())
    assert [r["model_version"] for r in responses] == [
        "integration-fixture-v1",
        "synthetic-infra-0001",
    ]
    assert set(responses[0]) == set(responses[1])
    assert set(responses[0]["prediction"]) == set(responses[1]["prediction"])
