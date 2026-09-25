"""Тесты point-in-time Feature Builder `tabular-v1` на крошечных синтетических данных."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.features.builder import (
    ForbiddenInputColumnError,
    build_features,
    gps_valid_mask,
)
from mostransport_ml.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

T0 = pd.Timestamp("2026-01-06 10:00:00")


def make_points(rows: list[tuple[str, int, pd.Timestamp]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [r[0] for r in rows],
            "tr_id": [r[1] for r in rows],
            "T": pd.to_datetime([r[2] for r in rows]),
            "target_stop_id": [1000 + r[1] for r in rows],
            "target_time_begin": pd.to_datetime([r[2] + pd.Timedelta(minutes=12) for r in rows]),
            "cur_dev_s": [30.0] * len(rows),
        }
    )


def make_telemetry(rows: list[dict]) -> pd.DataFrame:
    base = {"location_valid": True, "lon": 37.60, "lat": 55.75, "speed": 20.0}
    frame = pd.DataFrame([{**base, **r} for r in rows])
    frame["event_time"] = pd.to_datetime(frame["event_time"])
    return frame[["tr_id", "event_time", "location_valid", "lon", "lat", "speed"]]


def make_plan(tr_ids: list[int], geom: str = "POINT (37.61 55.76)") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tt_action_item_id": [1000 + t for t in tr_ids],
            "tr_id": tr_ids,
            "geom": [geom] * len(tr_ids),
            "manual_fill": [True] * len(tr_ids),
        }
    )


def test_output_schema_is_exact_and_versioned():
    points = make_points([("a", 1, T0)])
    telemetry = make_telemetry([{"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=30)}])
    features = build_features(points, telemetry, make_plan([1]))
    assert FEATURE_SCHEMA_VERSION == "tabular-v1"
    assert tuple(features.columns) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert features.index.name == "sample_id"
    assert "sample_id" not in features.columns
    for banned in ("tr_id", "target_stop_id", "unit_id", "packet_id", "building_address"):
        assert banned not in features.columns
    assert all(dtype == np.float64 for dtype in features.dtypes)


def test_future_packet_does_not_change_features():
    points = make_points([("a", 1, T0)])
    past = {"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=30), "speed": 10.0}
    future_a = {"tr_id": 1, "event_time": T0 + pd.Timedelta(seconds=1), "speed": 99.0}
    future_b = {**future_a, "speed": 0.0, "lon": 10.0, "lat": 10.0, "location_valid": False}
    plan = make_plan([1])

    only_past = build_features(points, make_telemetry([past]), plan)
    with_future_a = build_features(points, make_telemetry([past, future_a]), plan)
    with_future_b = build_features(points, make_telemetry([past, future_b]), plan)

    pd.testing.assert_frame_equal(only_past, with_future_a)
    pd.testing.assert_frame_equal(only_past, with_future_b)
    assert only_past.loc["a", "speed_last"] == 10.0
    assert only_past.loc["a", "latest_packet_lag_s"] == 30.0


def test_packet_exactly_at_T_is_included():
    points = make_points([("a", 1, T0)])
    telemetry = make_telemetry(
        [
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=30), "speed": 10.0},
            {"tr_id": 1, "event_time": T0, "speed": 40.0},
        ]
    )
    features = build_features(points, telemetry, make_plan([1]))
    assert features.loc["a", "latest_packet_lag_s"] == 0.0
    assert features.loc["a", "speed_last"] == 40.0
    assert features.loc["a", "rows_1m"] == 2.0


@pytest.mark.parametrize("window", [1, 3, 5, 10, 15])
def test_window_is_half_open_excluding_T_minus_w(window):
    points = make_points([("a", 1, T0)])
    telemetry = make_telemetry(
        [
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(minutes=window)},
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(minutes=window) + pd.Timedelta(seconds=1)},
            {"tr_id": 1, "event_time": T0},
        ]
    )
    features = build_features(points, telemetry, make_plan([1]))
    assert features.loc["a", f"rows_{window}m"] == 2.0


@pytest.mark.parametrize("forbidden", ["time_fact_begin", "target_delay_s", "target_class"])
@pytest.mark.parametrize("which", ["points", "telemetry", "schedule_plan"])
def test_forbidden_columns_fail_fast(forbidden, which):
    inputs = {
        "points": make_points([("a", 1, T0)]),
        "telemetry": make_telemetry([{"tr_id": 1, "event_time": T0}]),
        "schedule_plan": make_plan([1]),
    }
    inputs[which] = inputs[which].assign(**{forbidden: 1.0})
    with pytest.raises(ForbiddenInputColumnError, match=forbidden):
        build_features(**inputs)


def test_point_order_is_preserved():
    points = make_points([("B", 2, T0), ("A", 1, T0), ("C", 3, T0)])
    telemetry = make_telemetry(
        [
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=10), "speed": 1.0},
            {"tr_id": 2, "event_time": T0 - pd.Timedelta(seconds=20), "speed": 2.0},
            {"tr_id": 3, "event_time": T0 - pd.Timedelta(seconds=30), "speed": 3.0},
        ]
    )
    features = build_features(points, telemetry, make_plan([1, 2, 3]))
    assert list(features.index) == ["B", "A", "C"]
    assert features["speed_last"].tolist() == [2.0, 1.0, 3.0]
    assert features["latest_packet_lag_s"].tolist() == [20.0, 10.0, 30.0]


def test_telemetry_input_order_does_not_matter():
    points = make_points([("a", 1, T0)])
    rows = [
        {"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=s), "speed": float(s)}
        for s in (100, 5, 50, 250)
    ]
    ordered = build_features(points, make_telemetry(rows), make_plan([1]))
    shuffled = build_features(points, make_telemetry(rows[::-1]), make_plan([1]))
    pd.testing.assert_frame_equal(ordered, shuffled)
    assert ordered.loc["a", "speed_last"] == 5.0


def test_gps_valid_mask_is_strict():
    telemetry = pd.DataFrame(
        {
            "location_valid": [True, False, True, True, None],
            "lon": [37.6, 37.6, np.nan, 37.6, 37.6],
            "lat": [55.7, 55.7, 55.7, np.nan, 55.7],
        }
    )
    assert gps_valid_mask(telemetry).tolist() == [True, False, False, False, False]


def test_gps_features_use_strict_mask():
    points = make_points([("a", 1, T0)])
    telemetry = make_telemetry(
        [
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=40), "lon": 37.50, "lat": 55.70},
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=30), "location_valid": False},
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=20), "lon": np.nan},
        ]
    )
    features = build_features(points, telemetry, make_plan([1]))
    row = features.loc["a"]
    assert row["latest_valid_gps_lag_s"] == 40.0
    assert (row["latest_valid_lon"], row["latest_valid_lat"]) == (37.50, 55.70)
    assert row["valid_gps_count_5m"] == 1.0
    assert row["valid_gps_ratio_5m"] == pytest.approx(1 / 3)


def test_missing_history_returns_predictable_values():
    points = make_points([("a", 1, T0), ("b", 2, T0)])
    telemetry = make_telemetry([{"tr_id": 2, "event_time": T0 + pd.Timedelta(minutes=1)}])
    features = build_features(points, telemetry, make_plan([1, 2]))
    for sample in ("a", "b"):  # "b" has only future telemetry
        row = features.loc[sample]
        for name in FEATURE_NAMES:
            if name.startswith(("rows_", "valid_gps_count_")):
                assert row[name] == 0.0, name
        for name in (
            "latest_packet_lag_s",
            "latest_valid_gps_lag_s",
            "valid_gps_ratio_5m",
            "speed_last",
            "speed_mean_5m",
            "speed_std_5m",
            "zero_speed_fraction_5m",
            "speed_trend_5m",
            "latest_valid_lon",
            "distance_to_target_m",
        ):
            assert math.isnan(row[name]), name
        assert row["cur_dev_s"] == 30.0
        assert not math.isnan(row["target_lon"])


def test_speed_statistics_and_trend():
    points = make_points([("a", 1, T0)])
    # 4 минуты равномерного разгона: +2 км/ч в минуту, плюс null speed, который игнорируется.
    rows = [
        {"tr_id": 1, "event_time": T0 - pd.Timedelta(minutes=m), "speed": 20.0 - 2.0 * m}
        for m in (4, 3, 2, 1, 0)
    ]
    rows.append({"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=30), "speed": np.nan})
    features = build_features(points, make_telemetry(rows), make_plan([1]))
    row = features.loc["a"]
    assert row["speed_trend_5m"] == pytest.approx(2.0)
    assert row["speed_mean_5m"] == pytest.approx(16.0)
    assert row["speed_min_5m"] == 12.0
    assert row["speed_max_5m"] == 20.0
    assert row["speed_std_5m"] == pytest.approx(np.std([12, 14, 16, 18, 20], ddof=1))
    assert row["speed_last"] == 20.0
    assert row["rows_5m"] == 6.0


def test_speed_trend_needs_two_observations_and_zero_fraction():
    points = make_points([("a", 1, T0)])
    telemetry = make_telemetry(
        [
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(seconds=10), "speed": 0.0},
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(minutes=8), "speed": 0.0},
            {"tr_id": 1, "event_time": T0 - pd.Timedelta(minutes=9), "speed": 30.0},
        ]
    )
    row = build_features(points, telemetry, make_plan([1])).loc["a"]
    assert math.isnan(row["speed_trend_5m"])
    assert math.isnan(row["speed_std_5m"])
    assert row["zero_speed_fraction_5m"] == 1.0
    assert row["zero_speed_fraction_10m"] == pytest.approx(2 / 3)


def test_point_features_and_distance():
    t = pd.Timestamp("2026-01-06 13:45:30")
    points = make_points([("a", 1, t)])
    telemetry = make_telemetry([{"tr_id": 1, "event_time": t, "lon": 37.61, "lat": 55.76}])
    row = build_features(points, telemetry, make_plan([1])).loc["a"]
    assert (row["T_hour"], row["T_minute"]) == (13.0, 45.0)
    assert row["T_minutes_since_midnight"] == pytest.approx(13 * 60 + 45.5)
    assert row["horizon_minutes"] == 12.0
    assert row["distance_to_target_m"] == pytest.approx(0.0, abs=1e-6)
    assert row["manual_fill"] == 1.0


def test_horizon_outside_official_window_is_rejected():
    points = make_points([("a", 1, T0)])
    points["target_time_begin"] = T0 + pd.Timedelta(minutes=10)
    with pytest.raises(ValueError, match="horizon"):
        build_features(points, make_telemetry([{"tr_id": 1, "event_time": T0}]), make_plan([1]))


def test_missing_plan_row_gives_nan_and_malformed_geom_fails():
    points = make_points([("a", 1, T0)])
    telemetry = make_telemetry([{"tr_id": 1, "event_time": T0}])
    row = build_features(points, telemetry, make_plan([2])).loc["a"]
    assert math.isnan(row["target_lon"]) and math.isnan(row["manual_fill"])
    assert math.isnan(row["distance_to_target_m"])
    with pytest.raises(ValueError, match="Malformed geom"):
        build_features(points, telemetry, make_plan([1], geom="LINESTRING (1 2, 3 4)"))


def test_string_timestamps_are_rejected():
    points = make_points([("a", 1, T0)])
    points["T"] = points["T"].astype(str)
    with pytest.raises(TypeError):
        build_features(points, make_telemetry([{"tr_id": 1, "event_time": T0}]), make_plan([1]))
