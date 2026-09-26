#!/usr/bin/env python3
"""Integration smoke для Backend → ML Contract v1 (не проверка качества модели).

Против запущенного ML-сервиса: `GET /health`, `GET /ready`, затем
`POST /api/v1/predict` с каноническим примером запроса и проверка формы и
семантики ответа. Только stdlib; локальные HTTP-прокси игнорируются.

Запуск:
    uv run python scripts/smoke_contract_v1.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_PAYLOAD = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "contract_v1_request.json"
)
EXPECTED_FEATURE_SCHEMA_VERSION = "tabular-v1"

Fetch = Callable[[str, str, dict | None], tuple[int, Any]]


class SmokeFailure(AssertionError):
    pass


def http_fetch(base_url: str, timeout: float = 30.0) -> Fetch:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def fetch(method: str, path: str, body: dict | None) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")

    return fetch


def _aware(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise SmokeFailure(f"{name} must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SmokeFailure(f"{name} must be timezone-aware")
    return parsed


def check_prediction(request: dict, status: int, body: Any, *, expected_model: str | None) -> float:
    """Проверить ответ Contract v1; вернуть delay_seconds."""
    if status != 200 or not isinstance(body, dict):
        raise SmokeFailure(f"POST /api/v1/predict returned HTTP {status}")
    if body.get("request_id") != request["request_id"]:
        raise SmokeFailure("request_id is not echoed unchanged")
    if body.get("status") != "success":
        raise SmokeFailure("status must be 'success'")
    prediction = body.get("prediction")
    if not isinstance(prediction, dict):
        raise SmokeFailure("prediction object is missing")
    delay = prediction.get("delay_seconds")
    if isinstance(delay, bool) or not isinstance(delay, int | float) or not math.isfinite(delay):
        raise SmokeFailure("prediction.delay_seconds must be a finite number")
    if "reason" not in prediction or prediction["reason"] is not None:
        raise SmokeFailure("prediction.reason must be null")
    target_time = _aware(prediction.get("target_time"), "prediction.target_time")
    planned = datetime.fromisoformat(request["schedule_context"]["target_time_begin"])
    if abs(target_time - (planned + timedelta(seconds=delay))) > timedelta(microseconds=1):
        raise SmokeFailure("prediction.target_time != target_time_begin + delay_seconds")
    _aware(body.get("generated_at"), "generated_at")
    if not isinstance(body.get("model_version"), str) or not body["model_version"]:
        raise SmokeFailure("model_version must be a non-empty string")
    if expected_model is not None and body["model_version"] != expected_model:
        raise SmokeFailure("model_version differs from /ready")
    if body.get("feature_schema_version") != EXPECTED_FEATURE_SCHEMA_VERSION:
        raise SmokeFailure(f"feature_schema_version must be {EXPECTED_FEATURE_SCHEMA_VERSION}")
    return float(delay)


def run_smoke(fetch: Fetch, request: dict) -> list[str]:
    lines = []
    status, body = fetch("GET", "/health", None)
    if status != 200 or body != {"status": "ok"}:
        raise SmokeFailure(f"GET /health returned HTTP {status}")
    lines.append("GET /health: ok")

    status, body = fetch("GET", "/ready", None)
    if status != 200 or not isinstance(body, dict) or body.get("ready") is not True:
        raise SmokeFailure(f"GET /ready returned HTTP {status} (artifact not loaded?)")
    if body.get("feature_schema_version") != EXPECTED_FEATURE_SCHEMA_VERSION:
        raise SmokeFailure(
            f"/ready feature_schema_version must be {EXPECTED_FEATURE_SCHEMA_VERSION}"
        )
    model_version = body.get("model_version")
    lines.append(f"GET /ready: ready model_version={model_version}")

    status, body = fetch("POST", "/api/v1/predict", request)
    delay = check_prediction(request, status, body, expected_model=model_version)
    lines.append(
        f"POST /api/v1/predict: ok delay_seconds={delay} "
        f"target_time={body['prediction']['target_time']} generated_at={body['generated_at']}"
    )
    return lines


def main(argv: list[str] | None = None, fetch: Fetch | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contract v1 integration smoke")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    args = parser.parse_args(argv)
    request = json.loads(args.payload.read_text(encoding="utf-8"))
    try:
        for line in run_smoke(fetch or http_fetch(args.base_url), request):
            print(line)
    except (SmokeFailure, urllib.error.URLError, OSError, ValueError) as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    print("contract v1 smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
