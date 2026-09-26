"""Канонический Contract v1 пример: JSON → runtime adapter → CanonicalBatch → tabular-v1.

Fixture — корректный пример producer'а (T = 2026-01-06T03:35:00Z), ровно по
правилу истории: все пакеты `(T-15m, T]` + последний пакет + последний
strict-valid GPS пакет `<= T`:

  T-40m  strict-valid GPS — якорь старше окна
  T-10m, T-4m, T-30s  пакеты окна (GPS невалиден)
  T, T   одинаковое время: speed 11, затем 22 (последний пакет — внутри окна)

Граничные случаи (пакет ровно в T-15m, будущий пакет, null-координата)
строятся в тестах как варианты этого же fixture.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.features.adapters import runtime_context
from mostransport_ml.features.context import build_features_from_context
from mostransport_ml.features.schema import FEATURE_NAMES
from mostransport_ml.features.spatial import haversine_m
from mostransport_ml.serving.schemas import PredictRequestV1

FIXTURE = Path(__file__).parent / "fixtures" / "contract_v1_request.json"
PACKET_FIELDS = {"event_time", "lat", "lon", "location_valid", "speed"}


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def features_of(payload: dict) -> pd.Series:
    frame = build_features_from_context(runtime_context(payload))
    assert tuple(frame.columns) == FEATURE_NAMES
    return frame.iloc[0]


def with_packet(payload: dict, packet: dict) -> dict:
    extended = copy.deepcopy(payload)
    extended["telemetry"].append(packet)
    return extended


def to_offset(payload: dict, hours: int) -> dict:
    tz = timezone(timedelta(hours=hours))

    def convert(value: str) -> str:
        return datetime.fromisoformat(value).astimezone(tz).isoformat()

    shifted = copy.deepcopy(payload)
    shifted["prediction_time"] = convert(shifted["prediction_time"])
    shifted["schedule_context"]["target_time_begin"] = convert(
        shifted["schedule_context"]["target_time_begin"]
    )
    for packet in shifted["telemetry"]:
        packet["event_time"] = convert(packet["event_time"])
    return shifted


# ------------------------------------------ fixture is a correct producer example


def test_fixture_is_a_valid_contract_v1_request():
    request = PredictRequestV1.model_validate(load_fixture())
    assert request.prediction_time.utcoffset() == timedelta(0)
    assert len(request.telemetry) == 6
    for packet in load_fixture()["telemetry"]:
        assert set(packet) == PACKET_FIELDS  # только поля Contract v1
        assert isinstance(packet["speed"], float) and math.isfinite(packet["speed"])
        assert packet["event_time"].endswith("Z")


def test_fixture_follows_producer_history_rule():
    payload = load_fixture()
    t = datetime.fromisoformat(payload["prediction_time"])
    times = [datetime.fromisoformat(p["event_time"]) for p in payload["telemetry"]]
    assert max(times) <= t  # producer: event_time <= prediction_time, будущих пакетов нет
    in_window = [
        p for p, e in zip(payload["telemetry"], times, strict=True) if e > t - timedelta(minutes=15)
    ]
    outside = [
        p
        for p, e in zip(payload["telemetry"], times, strict=True)
        if e <= t - timedelta(minutes=15)
    ]
    # вне окна — только якорь: последний strict-valid GPS <= T (в окне валидного GPS нет)
    assert len(outside) == 1 and outside[0]["location_valid"] is True
    assert all(not p["location_valid"] for p in in_window)
    assert times == sorted(times)


def test_canonical_batch_semantics():
    batch = runtime_context(load_fixture())
    point = batch.points[0]
    assert str(point.prediction_time) == "2026-01-06 03:35:00"
    assert str(point.target_time_begin) == "2026-01-06 03:47:00"
    assert (point.current_deviation_s, point.manual_fill) == (274.0, False)
    assert batch.telemetry.speed.tolist() == [30.0, 10.0, 0.0, 8.0, 11.0, 22.0]
    assert batch.telemetry.event_time.max() == np.datetime64("2026-01-06T03:35:00", "ns")


def test_time_windows_and_packet_at_T():
    row = features_of(load_fixture())
    assert row["horizon_minutes"] == 12.0
    assert row["latest_packet_lag_s"] == 0.0  # event_time == T включён
    assert row["rows_15m"] == 5.0
    assert row["rows_5m"] == 4.0
    assert row["rows_1m"] == 3.0
    assert row["valid_gps_count_15m"] == 0.0
    assert row["valid_gps_ratio_15m"] == 0.0


def test_old_strict_valid_gps_anchor_is_used():
    row = features_of(load_fixture())
    assert row["latest_valid_gps_lag_s"] == 2400.0
    assert (row["latest_valid_lat"], row["latest_valid_lon"]) == (55.7401, 37.6002)
    expected = haversine_m(37.6002, 55.7401, 37.6184, 55.7512)
    assert row["distance_to_target_m"] == pytest.approx(float(expected), rel=0, abs=1e-9)


def test_equal_event_time_keeps_list_order():
    payload = load_fixture()
    assert features_of(payload)["speed_last"] == 22.0
    payload["telemetry"][4], payload["telemetry"][5] = (
        payload["telemetry"][5],
        payload["telemetry"][4],
    )
    assert features_of(payload)["speed_last"] == 11.0


# ------------------------------------------------------------------ boundary variants


def test_packet_exactly_at_window_start_is_excluded_from_rolling_window():
    base = features_of(load_fixture())
    boundary = {
        "event_time": "2026-01-06T03:20:00Z",  # ровно T-15m
        "lat": 55.7402,
        "lon": 37.6003,
        "location_valid": False,
        "speed": 5.0,
    }
    variant = with_packet(load_fixture(), boundary)
    variant["telemetry"].sort(key=lambda p: p["event_time"])
    pd.testing.assert_series_equal(features_of(variant), base, check_exact=True)
    assert features_of(variant)["rows_15m"] == 5.0


def test_future_packet_is_dropped_defense_in_depth():
    base = features_of(load_fixture())
    future = {
        "event_time": "2026-01-06T03:35:20Z",  # producer так не шлёт; ML всё равно отбрасывает
        "lat": 55.7599,
        "lon": 37.6199,
        "location_valid": True,
        "speed": 99.0,
    }
    variant = with_packet(load_fixture(), future)
    assert len(runtime_context(variant).telemetry) == 6
    pd.testing.assert_series_equal(features_of(variant), base, check_exact=True)


def test_null_coordinate_is_not_strict_valid_gps():
    variant = load_fixture()
    variant["telemetry"][1].update(location_valid=True, lat=None)
    row = features_of(variant)
    assert row["valid_gps_count_15m"] == 0.0
    assert row["latest_valid_gps_lag_s"] == 2400.0


@pytest.mark.parametrize("hours", [3, -5, 0])
def test_equivalent_offset_times_give_identical_features(hours):
    pd.testing.assert_series_equal(
        features_of(to_offset(load_fixture(), hours)), features_of(load_fixture()), check_exact=True
    )
