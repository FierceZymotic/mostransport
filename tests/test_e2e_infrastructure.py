"""E2E: Contract v1 JSON → HTTP → CanonicalBatch → tabular-v1 → CatBoost из bundle → ответ.

Плюс межпутевой parity: одна и та же логическая точка через HTTP и через
offline validate/submission даёт побитово одинаковый прогноз из одного bundle.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_validate_submission import T0, write_validate_dataset

from mostransport_ml.data.official import load_validate_inputs
from mostransport_ml.features.adapters import runtime_context
from mostransport_ml.features.context import build_features_from_context
from mostransport_ml.inference.predictor import ArtifactPredictor
from mostransport_ml.inference.submission import generate_validate_submission
from mostransport_ml.serving import artifact_app
from mostransport_ml.serving.app import create_app

FIXTURE = Path(__file__).parent / "fixtures" / "contract_v1_request.json"
NOW = datetime(2026, 1, 6, 3, 35, 1, tzinfo=UTC)


def iso_z(ts: pd.Timestamp) -> str:
    return ts.tz_localize("UTC").isoformat().replace("+00:00", "Z")


@pytest.mark.parametrize("formulation", ["direct", "residual"])
def test_runtime_http_e2e_with_real_catboost_bundle(
    catboost_bundle, tiny_catboost_model, formulation
):
    bundle = catboost_bundle(target_formulation=formulation)
    predictor = artifact_app.load_predictor(bundle)
    assert isinstance(predictor, ArtifactPredictor)
    client = TestClient(create_app(predictor, clock=lambda: NOW))
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert client.get("/ready").json() == {
        "ready": True,
        "model_version": "synthetic-infra-0001",
        "feature_schema_version": "tabular-v1",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    features = build_features_from_context(runtime_context(payload))
    raw = float(tiny_catboost_model.predict(features)[0])
    expected = raw + 274.0 if formulation == "residual" else raw
    assert body["prediction"]["delay_seconds"] == expected
    expected_target = (pd.Timestamp("2026-01-06 03:47:00") + pd.Timedelta(seconds=expected)).round(
        "us"
    )
    assert body["prediction"]["target_time"].endswith("Z")
    assert pd.Timestamp(body["prediction"]["target_time"]) == expected_target.tz_localize("UTC")
    assert body["request_id"] == payload["request_id"]
    assert body["model_version"] == predictor.manifest.model_version == "synthetic-infra-0001"
    assert body["feature_schema_version"] == predictor.manifest.feature_schema_version
    assert body["prediction"]["reason"] is None
    assert body["generated_at"] == "2026-01-06T03:35:01Z"


def test_same_bundle_same_point_same_prediction_via_http_and_submission(tmp_path, catboost_bundle):
    bundle = catboost_bundle(target_formulation="residual")
    root = write_validate_dataset(tmp_path / "ds")
    output = tmp_path / "submission.csv"
    generate_validate_submission(root, bundle, output)
    offline = pd.read_csv(output, sep=";", dtype={"sample_id": str}).set_index("sample_id")

    inputs = load_validate_inputs(root)
    point = inputs.points.set_index("sample_id").loc["131672_a"]
    telemetry = inputs.telemetry[inputs.telemetry["tr_id"] == 131672]
    payload = {
        "request_id": "131672_a",
        "prediction_time": iso_z(point["T"]),
        "vehicle_context": {"unit_id": "1", "tr_id": "131672", "route_id": "r"},
        "schedule_context": {
            "target_action_id": "900",
            "target_time_begin": iso_z(point["target_time_begin"]),
            "target_lat": 55.75,
            "target_lon": 37.61,
            "current_deviation_seconds": float(point["cur_dev_s"]),
            "manual_fill": False,
        },
        "telemetry": [
            {
                "event_time": iso_z(row.event_time),
                "lat": None if pd.isna(row.lat) else float(row.lat),
                "lon": None if pd.isna(row.lon) else float(row.lon),
                "location_valid": None if pd.isna(row.location_valid) else bool(row.location_valid),
                "speed": None if pd.isna(row.speed) else float(row.speed),
            }
            for row in telemetry.itertuples()
        ],
    }
    client = TestClient(artifact_app.create_artifact_app(bundle))
    body = client.post("/api/v1/predict", json=payload).json()
    assert body["prediction"]["delay_seconds"] == offline.loc["131672_a", "prediction"]
    assert point["T"] == T0


def test_artifact_app_unconfigured_and_corrupted_bundle_are_not_ready(
    tmp_path, catboost_bundle, caplog
):
    with caplog.at_level(logging.INFO, logger="mostransport_ml.serving"):
        unconfigured = TestClient(artifact_app.create_artifact_app(None))
        bundle = catboost_bundle()
        (bundle / "model.cbm").write_bytes(b"corrupted")
        corrupted = TestClient(artifact_app.create_artifact_app(bundle))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for client in (unconfigured, corrupted):
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200
        response = client.post("/api/v1/predict", json=fixture)
        assert response.status_code == 503
        assert response.json() == {"detail": "predictor is not ready"}
    assert "reason=artifact_not_configured" in caplog.text
    assert "reason=bundle_invalid" in caplog.text
    assert str(tmp_path) not in caplog.text


def test_incompatible_bundle_is_a_readiness_failure(tmp_path, tiny_catboost_model):
    from conftest import make_manifest

    from mostransport_ml.artifacts.bundle import write_bundle
    from mostransport_ml.inference.model_families import MODEL_FAMILIES

    blob = MODEL_FAMILIES["catboost"].serialize(tiny_catboost_model)
    write_bundle(
        tmp_path / "b", make_manifest(feature_schema_version="tabular-v2"), blob, "model.cbm"
    )
    client = TestClient(artifact_app.create_artifact_app(tmp_path / "b"))
    assert client.get("/ready").json()["ready"] is False


def test_env_factory_reads_artifact_dir(monkeypatch, catboost_bundle):
    monkeypatch.setenv(artifact_app.ARTIFACT_DIR_ENV_VAR, str(catboost_bundle()))
    assert TestClient(artifact_app.create_app_from_env()).get("/ready").status_code == 200
    monkeypatch.delenv(artifact_app.ARTIFACT_DIR_ENV_VAR)
    assert TestClient(artifact_app.create_app_from_env()).get("/ready").status_code == 503
