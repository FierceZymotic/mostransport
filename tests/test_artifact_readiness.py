"""Readiness must mean "model artifact loaded and compatible"; deployment defaults point at the final HGB bundle."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mostransport_ml.serving.artifact_app import create_artifact_app

ROOT = Path(__file__).resolve().parents[1]


def test_committed_integration_fixture_is_not_ready():
    # ./ml-artifact holds bundle.json + manifest.json but its model file is git-ignored:
    # it can never load, so the service must say "not ready" and refuse to predict.
    client = TestClient(create_artifact_app(ROOT / "ml-artifact"))
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 503 and ready.json()["ready"] is False
    assert client.post("/api/v1/predict", json={}).status_code in (422, 503)


def test_compose_and_image_healthchecks_probe_readiness_not_liveness():
    compose = (ROOT / "docker-compose.yml").read_text()
    ml_block = compose.split("\n  postgres:")[0]
    assert "8000/ready" in ml_block and "8000/health" not in ml_block
    assert "./artifacts/hgb-h0-runtime-safe-v1-group-a-v1" in ml_block
    assert "8000/ready" in (ROOT / "Dockerfile").read_text()


def test_env_example_does_not_override_the_final_artifact_with_the_fixture():
    # `cp .env.example .env` is the documented setup step; its value wins over the compose default.
    env = dict(line.split("=", 1) for line in (ROOT / ".env.example").read_text().splitlines()
               if line and not line.startswith("#") and "=" in line)
    assert env["ML_ARTIFACT_DIR"] == "./artifacts/hgb-h0-runtime-safe-v1-group-a-v1"
    assert env["SCHEDULE_FACT_SOURCE"] == "none"
    assert env["DEMO_CLOCK_MODE"] in {"off", "day_shift", "translate"}


def test_backend_compose_env_is_explicit_about_clock_and_fact_source():
    compose = (ROOT / "docker-compose.yml").read_text()
    backend = compose.split("\n  backend:")[1].split("\n  frontend:")[0]
    for name in ("DEMO_CLOCK_MODE", "DEMO_CLOCK_TARGET_DAY", "SCHEDULE_FACT_SOURCE"):
        assert f"{name}: ${{{name}:-" in backend
    assert "SCHEDULE_FACT_SOURCE: ${SCHEDULE_FACT_SOURCE:-none}" in backend
    assert "condition: service_started" in backend.split("postgres:")[0]
