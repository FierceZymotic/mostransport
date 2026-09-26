"""Contract v1 timestamps: aware ISO-8601 → UTC → naive-UTC (представление frozen M1).

Официальные naive-времена датасета — UTC wall-clock без метки пояса, поэтому
runtime-время `2026-01-06T03:35:00Z` обязано совпасть с offline `2026-01-06 03:35:00`.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.features.adapters import (
    ContextValidationError,
    offline_context,
    runtime_context,
)
from mostransport_ml.features.context import build_features_from_context

T_NAIVE_UTC = pd.Timestamp("2026-01-06 03:35:00")


def request(prediction_time: str, target_time: str, packet_times: list[str]) -> dict:
    return {
        "request_id": "r1",
        "prediction_time": prediction_time,
        "vehicle_context": {"unit_id": "u", "tr_id": "131672", "route_id": "r"},
        "schedule_context": {
            "target_action_id": "a",
            "target_time_begin": target_time,
            "target_lat": 55.75,
            "target_lon": 37.61,
            "current_deviation_seconds": 274,
            "manual_fill": False,
        },
        "telemetry": [
            {"event_time": t, "lat": 55.74, "lon": 37.60, "location_valid": True, "speed": 30.0}
            for t in packet_times
        ],
    }


Z_REQUEST = request(
    "2026-01-06T03:35:00Z",
    "2026-01-06T03:47:00Z",
    ["2026-01-06T03:33:10.5Z", "2026-01-06T03:35:00Z"],
)
PLUS_THREE = request(
    "2026-01-06T06:35:00+03:00",
    "2026-01-06T06:47:00+03:00",
    ["2026-01-06T06:33:10.5+03:00", "2026-01-06T06:35:00+03:00"],
)
MINUS_FIVE = request(
    "2026-01-05T22:35:00-05:00",
    "2026-01-05T22:47:00-05:00",
    ["2026-01-05T22:33:10.5-05:00", "2026-01-05T22:35:00-05:00"],
)
EXPLICIT_UTC_OFFSET = request(
    "2026-01-06T03:35:00+00:00",
    "2026-01-06T03:47:00+00:00",
    ["2026-01-06T03:33:10.5+00:00", "2026-01-06T03:35:00+00:00"],
)


@pytest.mark.parametrize("req", [Z_REQUEST, PLUS_THREE, MINUS_FIVE, EXPLICIT_UTC_OFFSET])
def test_aware_timestamps_normalize_to_naive_utc(req):
    batch = runtime_context(req)
    point = batch.points[0]
    assert point.prediction_time == T_NAIVE_UTC
    assert point.prediction_time.tzinfo is None
    assert point.target_time_begin == pd.Timestamp("2026-01-06 03:47:00")
    assert list(batch.telemetry.event_time) == [
        np.datetime64("2026-01-06T03:33:10.500000000"),
        np.datetime64("2026-01-06T03:35:00.000000000"),
    ]


def test_equivalent_instants_give_identical_features():
    features = [
        build_features_from_context(runtime_context(r))
        for r in (Z_REQUEST, PLUS_THREE, MINUS_FIVE, EXPLICIT_UTC_OFFSET)
    ]
    for other in features[1:]:
        pd.testing.assert_frame_equal(other, features[0], check_exact=True)
    assert features[0].loc["r1", "T_hour"] == 3.0  # UTC wall-clock, как в официальном датасете
    assert features[0].loc["r1", "horizon_minutes"] == 12.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.__setitem__("prediction_time", "2026-01-06T03:35:00"),
        lambda r: r["schedule_context"].__setitem__("target_time_begin", "2026-01-06T03:47:00"),
        lambda r: r["telemetry"][1].__setitem__("event_time", "2026-01-06T03:35:00"),
    ],
    ids=["prediction_time", "target_time_begin", "telemetry.event_time"],
)
def test_naive_runtime_timestamps_are_rejected(mutate):
    req = copy.deepcopy(Z_REQUEST)
    mutate(req)
    with pytest.raises(ContextValidationError, match="timezone-naive"):
        runtime_context(req)


def test_future_cutoff_uses_normalized_instants():
    # 06:35:01+03:00 — это T+1s в UTC: пакет должен быть отброшен.
    req = copy.deepcopy(Z_REQUEST)
    req["telemetry"].append(
        {
            "event_time": "2026-01-06T06:35:01+03:00",
            "lat": 1.0,
            "lon": 1.0,
            "location_valid": True,
            "speed": 99.0,
        }
    )
    assert len(runtime_context(req).telemetry) == 2


def test_offline_naive_utc_path_is_unchanged_and_matches_runtime():
    points = pd.DataFrame(
        {
            "sample_id": ["r1"],
            "tr_id": [131672],
            "T": pd.to_datetime([T_NAIVE_UTC]).astype("datetime64[ns]"),
            "target_stop_id": [1],
            "target_time_begin": pd.to_datetime(["2026-01-06 03:47:00"]).astype("datetime64[ns]"),
            "cur_dev_s": [274.0],
        }
    )
    telemetry = pd.DataFrame(
        {
            "tr_id": [131672, 131672],
            "event_time": pd.to_datetime(
                [pd.Timestamp("2026-01-06 03:33:10.5"), pd.Timestamp("2026-01-06 03:35:00")]
            ).astype("datetime64[ns]"),
            "location_valid": pd.array([True, True], dtype="boolean"),
            "lon": [37.60, 37.60],
            "lat": [55.74, 55.74],
            "speed": [30.0, 30.0],
        }
    )
    plan = pd.DataFrame(
        {
            "tt_action_item_id": [1],
            "tr_id": [131672],
            "geom": ["POINT (37.61 55.75)"],
            "manual_fill": [False],
        }
    )
    offline_batch = offline_context(points, telemetry, plan)
    assert offline_batch.points[0].prediction_time == T_NAIVE_UTC  # offline значения не сдвигаются
    offline = build_features_from_context(offline_batch)
    runtime = build_features_from_context(runtime_context(PLUS_THREE))
    pd.testing.assert_frame_equal(runtime, offline, check_exact=True)
