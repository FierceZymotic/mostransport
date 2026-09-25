"""M2-I1: канонический ML-контекст и parity offline ↔ runtime (Contract v1).

Каждый сценарий — одна логическая prediction point, выраженная дважды:
как offline-кадры (форма `load_official_split`) и как runtime-запрос
Contract v1. Оба пути должны дать побитово одинаковые признаки
`tabular-v1` через один и тот же замороженный builder, совпадающие с
прямым M1-путём `build_features`. Все данные синтетические.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.features.adapters import (
    ContextValidationError,
    offline_context,
    runtime_context,
)
from mostransport_ml.features.builder import ForbiddenInputColumnError, build_features
from mostransport_ml.features.context import (
    CanonicalPoint,
    build_features_from_context,
    to_builder_inputs,
)
from mostransport_ml.features.schema import FEATURE_NAMES

T = pd.Timestamp("2026-01-06 10:00:00")
TR_ID = 501
TARGET_ACTION = 90001
TARGET_LON, TARGET_LAT = 37.61234567, 55.7654321


def packet(offset_s: float, *, valid=True, lon=37.6, lat=55.75, speed=20.0) -> dict:
    return {
        "event_time": T + pd.Timedelta(seconds=offset_s),
        "location_valid": valid,
        "lon": lon,
        "lat": lat,
        "speed": speed,
    }


def offline_frames(packets, *, horizon_min=12.0, cur_dev=45.0, manual_fill=True):
    points = pd.DataFrame(
        {
            "sample_id": ["p1"],
            "tr_id": [TR_ID],
            "T": pd.to_datetime([T]).astype("datetime64[ns]"),
            "target_stop_id": [TARGET_ACTION],
            "target_time_begin": pd.to_datetime([T + pd.Timedelta(minutes=horizon_min)]).astype(
                "datetime64[ns]"
            ),
            "cur_dev_s": [cur_dev],
        }
    )
    other_vehicle = [dict(packet(-30, speed=99.0), tr_id=777), dict(packet(-5), tr_id=777)]
    rows = [dict(p, tr_id=TR_ID) for p in packets] + other_vehicle
    telemetry = pd.DataFrame(
        {
            "tr_id": pd.Series([r["tr_id"] for r in rows], dtype="int64"),
            "event_time": pd.to_datetime([r["event_time"] for r in rows]).astype("datetime64[ns]"),
            "location_valid": pd.array([r["location_valid"] for r in rows], dtype="boolean"),
            "lon": pd.Series([r["lon"] for r in rows], dtype="float64"),
            "lat": pd.Series([r["lat"] for r in rows], dtype="float64"),
            "speed": pd.Series([r["speed"] for r in rows], dtype="float64"),
        }
    )
    plan = pd.DataFrame(
        {
            "tt_action_item_id": [TARGET_ACTION, TARGET_ACTION + 1],
            "tr_id": [TR_ID, TR_ID],
            "geom": [f"POINT ({TARGET_LON} {TARGET_LAT})", "POINT (1 1)"],
            "manual_fill": [manual_fill, False],
        }
    )
    return points, telemetry, plan


def _json_number(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def runtime_request(packets, *, horizon_min=12.0, cur_dev=45.0, manual_fill=True):
    return {
        "request_id": "p1",
        "prediction_time": T.isoformat(),
        "vehicle_context": {"unit_id": "u-17", "tr_id": str(TR_ID), "route_id": "r-42"},
        "schedule_context": {
            "target_action_id": str(TARGET_ACTION),
            "target_time_begin": (T + pd.Timedelta(minutes=horizon_min)).isoformat(),
            "target_lat": TARGET_LAT,
            "target_lon": TARGET_LON,
            "current_deviation_seconds": cur_dev,
            "manual_fill": manual_fill,
        },
        "telemetry": [
            {
                "event_time": p["event_time"].isoformat(),
                "location_valid": p["location_valid"],
                "lat": _json_number(p["lat"]),
                "lon": _json_number(p["lon"]),
                "speed": _json_number(p["speed"]),
                "heading": 90.0,
                "raw_ndtp_field": "ignored",
            }
            for p in packets
        ],
    }


def assert_parity(packets, **kwargs) -> pd.DataFrame:
    """Path A (offline → canonical → builder) == Path B (runtime → canonical → builder)."""
    frames = offline_frames(packets, **kwargs)
    x_m1 = build_features(*frames)
    x_offline = build_features_from_context(offline_context(*frames))
    x_runtime = build_features_from_context(runtime_context(runtime_request(packets, **kwargs)))

    for x in (x_offline, x_runtime):
        assert tuple(x.columns) == FEATURE_NAMES
        assert list(x.index) == ["p1"]
    pd.testing.assert_frame_equal(x_offline, x_m1, check_exact=True)
    pd.testing.assert_frame_equal(x_runtime, x_offline, check_exact=True)
    return x_runtime


# ------------------------------------------------------------------ parity cases


def test_parity_normal_history():
    packets = [
        packet(-s, speed=10.0 + (s % 7), lon=37.6 + s * 1e-5, lat=55.75 - s * 1e-5)
        for s in range(1200, -1, -10)
    ]
    x = assert_parity(packets)
    assert x.loc["p1", "rows_15m"] == 90.0
    assert not x.loc["p1"].isna().any()


def test_parity_sparse_history():
    x = assert_parity([packet(-840, speed=5.0), packet(-400, speed=12.0), packet(-30, speed=0.0)])
    assert x.loc["p1", "rows_1m"] == 1.0
    assert math.isnan(x.loc["p1", "speed_std_3m"])


def test_parity_missing_and_invalid_gps():
    packets = [
        packet(-600),
        packet(-300, valid=False),
        packet(-200, valid=None),
        packet(-100, lon=None, lat=None, speed=None),
        packet(-50, valid=True, lat=None),
    ]
    x = assert_parity(packets)
    assert x.loc["p1", "latest_valid_gps_lag_s"] == 600.0
    assert x.loc["p1", "valid_gps_count_5m"] == 0.0
    assert x.loc["p1", "valid_gps_ratio_5m"] == 0.0


def test_parity_zero_speed():
    x = assert_parity([packet(-s, speed=0.0) for s in (280, 200, 120, 40)])
    assert x.loc["p1", "zero_speed_fraction_5m"] == 1.0
    assert x.loc["p1", "speed_trend_5m"] == 0.0


def test_parity_packet_exactly_at_T_is_used():
    x = assert_parity([packet(-30, speed=10.0), packet(0, speed=40.0)])
    assert x.loc["p1", "latest_packet_lag_s"] == 0.0
    assert x.loc["p1", "speed_last"] == 40.0


def test_parity_future_packet_has_no_influence():
    base = [packet(-30, speed=10.0)]
    future = base + [packet(1, speed=99.0, lon=10.0, lat=10.0), packet(3600, valid=False)]
    pd.testing.assert_frame_equal(assert_parity(base), assert_parity(future), check_exact=True)
    batch = runtime_context(runtime_request(future))
    assert len(batch.telemetry) == 1  # будущие пакеты не попадают в канонический контекст


@pytest.mark.parametrize("window", [1, 3, 5, 10, 15])
def test_parity_packet_exactly_at_window_boundary(window):
    boundary = -60 * window
    x = assert_parity([packet(boundary), packet(boundary + 1), packet(0)])
    assert x.loc["p1", f"rows_{window}m"] == 2.0  # T-w исключён, T включён


def test_parity_empty_history():
    x = assert_parity([])
    assert x.loc["p1", "rows_15m"] == 0.0
    assert math.isnan(x.loc["p1", "latest_packet_lag_s"])
    assert x.loc["p1", "target_lon"] == TARGET_LON


def test_parity_other_timestamps_and_values():
    x = assert_parity([packet(-90.123456)], horizon_min=14.5, cur_dev=-17.25, manual_fill=False)
    assert x.loc["p1", "horizon_minutes"] == 14.5
    assert x.loc["p1", "manual_fill"] == 0.0
    assert x.loc["p1", "latest_packet_lag_s"] == pytest.approx(90.123456)


# ------------------------------------------------------------------ history > 15 minutes


OLD_HISTORY = [
    packet(-2400, valid=True, lon=37.50, lat=55.70, speed=33.0),  # T-40m: последний valid GPS
    packet(-1800, valid=False, lon=None, lat=None, speed=None),  # T-30m
    *[packet(-s, valid=False, lon=None, lat=None, speed=None) for s in (840, 600, 300, 60)],
]


def test_unbounded_features_depend_on_history_older_than_15_minutes():
    x = assert_parity(OLD_HISTORY)
    row = x.loc["p1"]
    assert row["latest_valid_gps_lag_s"] == 2400.0
    assert (row["latest_valid_lon"], row["latest_valid_lat"]) == (37.50, 55.70)
    assert row["speed_last"] == 33.0
    assert not math.isnan(row["distance_to_target_m"])
    assert row["valid_gps_count_15m"] == 0.0


def test_runtime_adapter_does_not_truncate_history():
    batch = runtime_context(runtime_request(OLD_HISTORY))
    assert len(batch.telemetry) == len(OLD_HISTORY)
    assert batch.telemetry.event_time.min() == np.datetime64(T - pd.Timedelta(minutes=40), "ns")


def test_fifteen_minute_truncation_would_create_skew():
    full = assert_parity(OLD_HISTORY)
    truncated = [p for p in OLD_HISTORY if p["event_time"] > T - pd.Timedelta(minutes=15)]
    cut = build_features_from_context(runtime_context(runtime_request(truncated)))
    changed = sorted(c for c in FEATURE_NAMES if not full[c].equals(cut[c]))
    assert changed == sorted(
        [
            "latest_valid_gps_lag_s",
            "latest_valid_lon",
            "latest_valid_lat",
            "distance_to_target_m",
            "speed_last",
        ]
    )


def test_window_plus_latest_anchors_is_sufficient_history():
    """Минимальная точная история: (T-15m, T] + последний пакет, последний strict-valid
    GPS и последний пакет с non-null speed на момент <= T (даже если они старше)."""
    rich = [
        packet(-5000, speed=1.0),
        packet(-3000, valid=True, lon=37.51, lat=55.71, speed=None),  # последний valid GPS
        packet(-2500, valid=False, lon=37.0, lat=55.0, speed=7.0),  # последний non-null speed
        packet(-2000, valid=False, lon=None, lat=None, speed=None),  # последний пакет до окна
        *[packet(-s, valid=False, lon=None, lat=None, speed=None) for s in (800, 200)],
    ]
    anchors = [rich[1], rich[2]]  # последний пакет уже внутри окна
    compact = anchors + [p for p in rich if p["event_time"] > T - pd.Timedelta(minutes=15)]
    full = assert_parity(rich)
    pd.testing.assert_frame_equal(assert_parity(compact), full, check_exact=True)


# ------------------------------------------------------------------ runtime validation


def _with(path: tuple[str, ...], value, *, delete=False):
    request = runtime_request([packet(-30)])
    target = request
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return request


@pytest.mark.parametrize(
    "path",
    [
        ("request_id",),
        ("prediction_time",),
        ("vehicle_context",),
        ("vehicle_context", "tr_id"),
        ("vehicle_context", "unit_id"),
        ("vehicle_context", "route_id"),
        ("schedule_context",),
        ("schedule_context", "target_action_id"),
        ("schedule_context", "target_time_begin"),
        ("schedule_context", "target_lat"),
        ("schedule_context", "target_lon"),
        ("schedule_context", "current_deviation_seconds"),
        ("schedule_context", "manual_fill"),
        ("telemetry",),
    ],
)
def test_missing_required_context_fails(path):
    with pytest.raises(ContextValidationError, match=path[-1]):
        runtime_context(_with(path, None, delete=True))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("request_id",), ""),
        (("prediction_time",), "yesterday"),
        (("prediction_time",), "2026-01-06T10:00:00+03:00"),
        (("prediction_time",), "2026-01-06T10:00:00Z"),
        (("vehicle_context", "tr_id"), 501),
        (("schedule_context", "target_lat"), None),
        (("schedule_context", "target_lat"), 95.0),
        (("schedule_context", "target_lon"), "37.6"),
        (("schedule_context", "current_deviation_seconds"), None),
        (("schedule_context", "current_deviation_seconds"), float("nan")),
        (("schedule_context", "current_deviation_seconds"), True),
        (("schedule_context", "manual_fill"), None),
        (("schedule_context", "manual_fill"), "true"),
        (("telemetry",), {"event_time": "2026-01-06T09:59:00"}),
    ],
)
def test_malformed_context_fails(path, value):
    with pytest.raises(ContextValidationError):
        runtime_context(_with(path, value))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_time", None),
        ("event_time", "2026-01-06T09:59:00Z"),
        ("location_valid", "yes"),
        ("lat", "55.7"),
        ("speed", float("inf")),
    ],
)
def test_malformed_packet_fails(field, value):
    request = runtime_request([packet(-30)])
    request["telemetry"][0][field] = value
    with pytest.raises(ContextValidationError, match=field):
        runtime_context(request)


@pytest.mark.parametrize("field", ["event_time", "location_valid", "lat", "lon", "speed"])
def test_missing_packet_field_fails(field):
    request = runtime_request([packet(-30)])
    del request["telemetry"][0][field]
    with pytest.raises(ContextValidationError, match=field):
        runtime_context(request)


def test_horizon_outside_official_window_fails_in_shared_builder():
    batch = runtime_context(runtime_request([packet(-30)], horizon_min=10.0))
    with pytest.raises(ValueError, match="horizon"):
        build_features_from_context(batch)


@pytest.mark.parametrize("where", ["top", "vehicle_context", "schedule_context", "packet"])
@pytest.mark.parametrize("forbidden", ["time_fact_begin", "target_delay_s", "target_class"])
def test_forbidden_fields_fail_fast_anywhere_in_request(where, forbidden):
    request = runtime_request([packet(-30)])
    target = {
        "top": request,
        "vehicle_context": request["vehicle_context"],
        "schedule_context": request["schedule_context"],
        "packet": request["telemetry"][0],
    }[where]
    target[forbidden] = 1
    with pytest.raises(ForbiddenInputColumnError, match=forbidden):
        runtime_context(request)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_offline_adapter_rejects_forbidden_columns(which):
    frames = list(offline_frames([packet(-30)]))
    frames[which] = frames[which].assign(time_fact_begin=1)
    with pytest.raises(ForbiddenInputColumnError):
        offline_context(*frames)


# ------------------------------------------------------------------ canonical context


def test_identity_fields_are_traceability_only():
    base = build_features_from_context(runtime_context(runtime_request([packet(-30)])))
    request = runtime_request([packet(-30)])
    request["vehicle_context"].update(unit_id="other-unit", route_id="other-route")
    request["schedule_context"]["target_action_id"] = "another-action"
    changed = build_features_from_context(runtime_context(request))
    pd.testing.assert_frame_equal(base, changed, check_exact=True)
    for identity in ("unit_id", "route_id", "tr_id", "target_action_id", "request_id"):
        assert identity not in FEATURE_NAMES


def test_runtime_identity_is_carried_for_traceability():
    point = runtime_context(runtime_request([packet(-30)])).points[0]
    assert (point.point_id, point.tr_id, point.unit_id, point.route_id) == (
        "p1",
        "501",
        "u-17",
        "r-42",
    )
    assert point.target_action_id == str(TARGET_ACTION)


def test_equal_timestamps_keep_source_order_in_both_paths():
    tied = [packet(-30, speed=10.0), packet(-30, speed=20.0)]
    assert assert_parity(tied).loc["p1", "speed_last"] == 20.0
    assert assert_parity(tied[::-1]).loc["p1", "speed_last"] == 10.0


def test_canonical_context_is_immutable():
    batch = runtime_context(runtime_request([packet(-30)]))
    with pytest.raises(FrozenInstanceError):
        batch.points[0].current_deviation_s = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        batch.telemetry.speed[0] = 0.0


def test_canonical_point_validation():
    kwargs = {
        "point_id": "p",
        "tr_id": "1",
        "prediction_time": T,
        "target_action_id": "a",
        "target_time_begin": T + pd.Timedelta(minutes=12),
        "target_lon": 37.6,
        "target_lat": 55.7,
        "current_deviation_s": 0.0,
        "manual_fill": True,
    }
    CanonicalPoint(**kwargs)
    with pytest.raises(ValueError, match="both"):
        CanonicalPoint(**{**kwargs, "target_lat": math.nan})
    with pytest.raises(ValueError, match="naive"):
        CanonicalPoint(**{**kwargs, "prediction_time": T.tz_localize("UTC")})


@pytest.mark.parametrize("coordinate", [37.43070705, 1e-05, -0.0, 0.1 + 0.2, 179.99999999999997])
def test_target_coordinates_round_trip_exactly_through_builder_boundary(coordinate):
    batch = runtime_context(runtime_request([packet(-30)]))
    point = batch.points[0]
    moved = CanonicalPoint(**{**point.__dict__, "target_lon": coordinate})
    _, _, plan = to_builder_inputs(type(batch)(points=(moved,), telemetry=batch.telemetry))
    features = build_features_from_context(type(batch)(points=(moved,), telemetry=batch.telemetry))
    assert plan.loc[0, "geom"].startswith("POINT (")
    assert features.loc["p1", "target_lon"] == coordinate


def test_offline_missing_plan_keeps_m1_nan_semantics():
    points, telemetry, plan = offline_frames([packet(-30)])
    plan = plan.iloc[1:]  # целевой план отсутствует
    x_m1 = build_features(points, telemetry, plan)
    x_canonical = build_features_from_context(offline_context(points, telemetry, plan))
    pd.testing.assert_frame_equal(x_canonical, x_m1, check_exact=True)
    assert math.isnan(x_canonical.loc["p1", "target_lon"])
    assert math.isnan(x_canonical.loc["p1", "manual_fill"])


def test_offline_multi_point_order_matches_m1():
    points, telemetry, plan = offline_frames([packet(-s) for s in (500, 100, 0)])
    extra = points.assign(sample_id="p0", T=points["T"] - pd.Timedelta(minutes=5)).assign(
        target_time_begin=points["target_time_begin"] - pd.Timedelta(minutes=5)
    )
    multi = pd.concat([points, extra], ignore_index=True)
    x_m1 = build_features(multi, telemetry, plan)
    x_canonical = build_features_from_context(offline_context(multi, telemetry, plan))
    assert list(x_canonical.index) == ["p1", "p0"]
    pd.testing.assert_frame_equal(x_canonical, x_m1, check_exact=True)
