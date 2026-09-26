"""P1: frozen `runtime-safe-v1`, строгая проекция 37 → 29 и offline/runtime parity.

Главный acceptance-тест: одно логическое prediction state, выраженное как
offline-кадры (с явным safe deviation override) и как Contract v1
`PredictRequestV1`, даёт один и тот же 29-мерный вектор `runtime-safe-v1`
через общий канонический builder. Все данные синтетические.
"""

from __future__ import annotations

import math
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_feature_frame

from mostransport_ml.data.safe_deviation import safe_current_deviation_seconds
from mostransport_ml.features.adapters import offline_context, runtime_context
from mostransport_ml.features.context import (
    build_features_from_context,
    project_runtime_safe_features,
)
from mostransport_ml.features.schema import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    RUNTIME_SAFE_EXCLUDED_FEATURES,
    RUNTIME_SAFE_FEATURE_NAMES,
    RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
)
from mostransport_ml.serving.schemas import PredictRequestV1
from mostransport_ml.serving.service import InferenceService

LEGACY_TABULAR_V1 = (
    "cur_dev_s",
    "horizon_minutes",
    "T_hour",
    "T_minute",
    "T_minutes_since_midnight",
    "latest_packet_lag_s",
    "latest_valid_gps_lag_s",
    "rows_1m",
    "rows_3m",
    "rows_5m",
    "rows_10m",
    "rows_15m",
    "valid_gps_count_5m",
    "valid_gps_count_10m",
    "valid_gps_count_15m",
    "valid_gps_ratio_5m",
    "valid_gps_ratio_10m",
    "valid_gps_ratio_15m",
    "speed_last",
    "speed_mean_1m",
    "speed_mean_3m",
    "speed_mean_5m",
    "speed_mean_10m",
    "speed_std_3m",
    "speed_std_5m",
    "speed_std_10m",
    "speed_min_5m",
    "speed_max_5m",
    "zero_speed_fraction_5m",
    "zero_speed_fraction_10m",
    "speed_trend_5m",
    "target_lon",
    "target_lat",
    "latest_valid_lon",
    "latest_valid_lat",
    "distance_to_target_m",
    "manual_fill",
)
FROZEN_RUNTIME_SAFE_V1 = (
    "cur_dev_s",
    "horizon_minutes",
    "T_hour",
    "T_minute",
    "T_minutes_since_midnight",
    "latest_packet_lag_s",
    "latest_valid_gps_lag_s",
    "valid_gps_ratio_5m",
    "valid_gps_ratio_10m",
    "valid_gps_ratio_15m",
    "speed_last",
    "speed_mean_1m",
    "speed_mean_3m",
    "speed_mean_5m",
    "speed_mean_10m",
    "speed_std_3m",
    "speed_std_5m",
    "speed_std_10m",
    "speed_min_5m",
    "speed_max_5m",
    "zero_speed_fraction_5m",
    "zero_speed_fraction_10m",
    "speed_trend_5m",
    "target_lon",
    "target_lat",
    "latest_valid_lon",
    "latest_valid_lat",
    "distance_to_target_m",
    "manual_fill",
)
RAW_COUNTS = (
    "rows_1m",
    "rows_3m",
    "rows_5m",
    "rows_10m",
    "rows_15m",
    "valid_gps_count_5m",
    "valid_gps_count_10m",
    "valid_gps_count_15m",
)


# ------------------------------------------------------------------ schema


def test_runtime_safe_schema_version():
    assert RUNTIME_SAFE_FEATURE_SCHEMA_VERSION == "runtime-safe-v1"


def test_runtime_safe_schema_has_exactly_29_features_in_frozen_order():
    assert len(RUNTIME_SAFE_FEATURE_NAMES) == 29
    assert len(set(RUNTIME_SAFE_FEATURE_NAMES)) == 29
    assert RUNTIME_SAFE_FEATURE_NAMES == FROZEN_RUNTIME_SAFE_V1


def test_raw_counts_are_excluded_and_everything_else_is_kept_in_order():
    assert RUNTIME_SAFE_EXCLUDED_FEATURES == RAW_COUNTS
    assert not set(RAW_COUNTS) & set(RUNTIME_SAFE_FEATURE_NAMES)
    kept = tuple(name for name in FEATURE_NAMES if name not in RAW_COUNTS)
    assert kept == RUNTIME_SAFE_FEATURE_NAMES


def test_legacy_tabular_v1_is_unchanged():
    assert FEATURE_SCHEMA_VERSION == "tabular-v1"
    assert FEATURE_NAMES == LEGACY_TABULAR_V1
    assert len(FEATURE_NAMES) == 37


# ------------------------------------------------------------------ projection


def tabular_frame() -> pd.DataFrame:
    frame = synthetic_feature_frame(n_rows=6, seed=11)
    frame.index = pd.Index([f"s{i}" for i in range(6)], name="sample_id")
    frame.loc["s2", "latest_valid_lon"] = np.nan
    return frame


def test_projection_selects_exact_runtime_safe_columns_without_changing_values():
    source = tabular_frame()
    projected = project_runtime_safe_features(source)
    assert tuple(projected.columns) == RUNTIME_SAFE_FEATURE_NAMES
    assert projected.index.equals(source.index)
    pd.testing.assert_frame_equal(
        projected, source[list(RUNTIME_SAFE_FEATURE_NAMES)], check_exact=True
    )
    assert math.isnan(projected.loc["s2", "latest_valid_lon"])  # NaN не заполняется
    assert projected["speed_trend_5m"].isna().sum() == source["speed_trend_5m"].isna().sum()


def test_projection_does_not_mutate_source_and_returns_independent_frame():
    source = tabular_frame()
    before = source.copy(deep=True)
    projected = project_runtime_safe_features(source)
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    projected.iloc[:, :] = 0.0
    pd.testing.assert_frame_equal(source, before, check_exact=True)


def test_projection_rejects_missing_feature():
    with pytest.raises(ValueError, match="missing columns: \\['speed_last'\\]"):
        project_runtime_safe_features(tabular_frame().drop(columns="speed_last"))


def test_projection_rejects_missing_raw_count_too():
    with pytest.raises(ValueError, match="missing columns: \\['rows_5m'\\]"):
        project_runtime_safe_features(tabular_frame().drop(columns="rows_5m"))


def test_projection_rejects_unexpected_feature():
    with pytest.raises(ValueError, match="unexpected columns: \\['stop_duration_s'\\]"):
        project_runtime_safe_features(tabular_frame().assign(stop_duration_s=1.0))


def test_projection_rejects_reordered_columns():
    source = tabular_frame()
    swapped = list(source.columns)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="order"):
        project_runtime_safe_features(source[swapped])


def test_projection_rejects_duplicate_columns():
    source = tabular_frame()
    duplicated = pd.concat([source, source[["cur_dev_s"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate columns: \\['cur_dev_s'\\]"):
        project_runtime_safe_features(duplicated)


def test_projection_rejects_duplicate_that_replaces_a_feature():
    source = tabular_frame()
    renamed = source.rename(columns={"speed_last": "speed_mean_1m"})
    assert len(renamed.columns) == 37
    with pytest.raises(ValueError, match="duplicate columns"):
        project_runtime_safe_features(renamed)


def test_projection_requires_a_dataframe():
    with pytest.raises(TypeError):
        project_runtime_safe_features(tabular_frame().to_numpy())


# ------------------------------------------------------------------ parity fixture

SAMPLE_ID = "131672_1767693600"
TR_ID = 131672
T_UTC = pd.Timestamp("2026-01-06 10:00:00")  # naive UTC wall-clock, как в датасете
MSK = timezone(timedelta(hours=3))
TARGET_ACTION = 53700172828
TARGET_TIME = T_UTC + pd.Timedelta(minutes=12.5)
TARGET_LON, TARGET_LAT = 37.61234567, 55.7654321
OFFICIAL_CUR_DEV = 999.0  # official-значение в points; override должен его заменить


def iso_msk(ts: pd.Timestamp) -> str:
    """Contract v1: timezone-aware ISO-8601 (здесь +03:00, не UTC)."""
    return ts.tz_localize("UTC").tz_convert(MSK).isoformat()


def history() -> list[dict]:
    """21 минута telemetry с шагом 12 с + последний пакет за 7 с до T.

    Скорость меняется, есть нули и почти нули; есть невалидный GPS
    (`location_valid` false и null, координаты null) — последний пакет тоже
    невалиден, поэтому lag пакета и lag валидного GPS различаются.
    """
    packets = []
    for k, offset in enumerate(range(-1260, 0, 12)):
        speed = round(18.0 + 14.0 * math.sin(k / 4.0), 1)
        if k % 9 == 4:
            speed = 0.0
        elif k % 13 == 6:
            speed = 0.4
        valid: bool | None = k % 7 != 3
        if k % 11 == 5:
            valid = None
        has_coordinates = valid is not False or k % 2 == 0
        packets.append(
            {
                "offset_s": offset,
                "location_valid": valid,
                "lon": 37.5 + k * 1.1e-4 if has_coordinates else None,
                "lat": 55.70 + k * 0.6e-4 if has_coordinates else None,
                "speed": speed,
            }
        )
    packets.append(
        {"offset_s": -7, "location_valid": False, "lon": None, "lat": None, "speed": 11.5}
    )
    return packets


FUTURE_PACKETS = [
    {"offset_s": 1e-6, "location_valid": True, "lon": 170.0, "lat": -80.0, "speed": 999.0},
    {"offset_s": 30, "location_valid": True, "lon": -170.0, "lat": 80.0, "speed": 500.0},
]


def schedule_facts() -> pd.DataFrame:
    """Factual schedule того же ТС: последнее подтверждённое событие <= T даёт 137 с."""
    rows = [
        (53700172820, -20 * 60, -20 * 60 + 70),
        (53700172821, -9 * 60, -9 * 60 + 137),
        (53700172822, 2 * 60, 2 * 60 + 100),  # факт после T — не участвует
        (TARGET_ACTION, 12.5 * 60, None),  # ещё не подтверждено
    ]
    return pd.DataFrame(
        {
            "tr_id": [TR_ID] * len(rows),
            "tt_action_item_id": [r[0] for r in rows],
            "time_begin": [T_UTC + pd.Timedelta(seconds=r[1]) for r in rows],
            "time_fact_begin": pd.to_datetime(
                [None if r[2] is None else T_UTC + pd.Timedelta(seconds=r[2]) for r in rows]
            ),
        }
    )


def offline_frames(packets: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = pd.DataFrame(
        {
            "sample_id": [SAMPLE_ID],
            "tr_id": [TR_ID],
            "T": pd.to_datetime([T_UTC]).astype("datetime64[ns]"),
            "target_stop_id": [TARGET_ACTION],
            "target_time_begin": pd.to_datetime([TARGET_TIME]).astype("datetime64[ns]"),
            "cur_dev_s": [OFFICIAL_CUR_DEV],
        }
    )
    decoy = [  # другое ТС с экстремальными значениями в то же время
        {"offset_s": -30, "location_valid": True, "lon": 10.0, "lat": 10.0, "speed": 300.0},
        {"offset_s": -3, "location_valid": True, "lon": 11.0, "lat": 11.0, "speed": 0.0},
    ]
    rows = [dict(p, tr_id=TR_ID) for p in packets] + [dict(p, tr_id=777) for p in decoy]
    telemetry = pd.DataFrame(
        {
            "tr_id": pd.Series([r["tr_id"] for r in rows], dtype="int64"),
            "event_time": pd.to_datetime(
                [T_UTC + pd.Timedelta(seconds=r["offset_s"]) for r in rows]
            ).astype("datetime64[ns]"),
            "location_valid": pd.array([r["location_valid"] for r in rows], dtype="boolean"),
            "lon": pd.Series([r["lon"] for r in rows], dtype="float64"),
            "lat": pd.Series([r["lat"] for r in rows], dtype="float64"),
            "speed": pd.Series([r["speed"] for r in rows], dtype="float64"),
        }
    )
    plan = pd.DataFrame(
        {
            "tt_action_item_id": [TARGET_ACTION, 53700172821],
            "tr_id": [TR_ID, TR_ID],
            "geom": [f"POINT ({TARGET_LON} {TARGET_LAT})", "POINT (37.5 55.7)"],
            "manual_fill": [True, False],
        }
    )
    return points, telemetry, plan


def runtime_payload(packets: list[dict], current_deviation_seconds: float) -> dict:
    return {
        "request_id": SAMPLE_ID,
        "prediction_time": iso_msk(T_UTC),
        "vehicle_context": {"unit_id": "unit-17", "tr_id": str(TR_ID), "route_id": "route-7"},
        "schedule_context": {
            "target_action_id": str(TARGET_ACTION),
            "target_time_begin": iso_msk(TARGET_TIME),
            "target_lat": TARGET_LAT,
            "target_lon": TARGET_LON,
            "current_deviation_seconds": current_deviation_seconds,
            "manual_fill": True,
        },
        "telemetry": [
            {
                "event_time": iso_msk(T_UTC + pd.Timedelta(seconds=p["offset_s"])),
                "location_valid": p["location_valid"],
                "lat": p["lat"],
                "lon": p["lon"],
                "speed": p["speed"],
                "heading": 90.0,
            }
            for p in packets
        ],
    }


def offline_path(packets: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    points, telemetry, plan = offline_frames(packets)
    safe = safe_current_deviation_seconds(points, schedule_facts())
    batch = offline_context(points, telemetry, plan, current_deviation_seconds=safe)
    features = build_features_from_context(batch)
    return features, project_runtime_safe_features(features)


def runtime_path(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    request = PredictRequestV1.model_validate(payload)
    batch = runtime_context(request.model_dump(mode="json"))  # как InferenceService
    features = build_features_from_context(batch)
    return features, project_runtime_safe_features(features)


def safe_value() -> float:
    points, _, _ = offline_frames(history())
    return float(safe_current_deviation_seconds(points, schedule_facts()).loc[SAMPLE_ID])


# ------------------------------------------------------------------ parity


def test_offline_and_runtime_give_the_same_runtime_safe_vector():
    assert safe_value() == 137.0
    offline_37, offline_29 = offline_path(history())
    runtime_37, runtime_29 = runtime_path(runtime_payload(history(), safe_value()))

    assert tuple(offline_29.columns) == tuple(runtime_29.columns) == RUNTIME_SAFE_FEATURE_NAMES
    assert list(offline_29.index) == list(runtime_29.index) == [SAMPLE_ID]
    pd.testing.assert_frame_equal(runtime_37, offline_37, check_exact=True)
    pd.testing.assert_frame_equal(runtime_29, offline_29, check_exact=True)
    np.testing.assert_allclose(
        offline_29.to_numpy(), runtime_29.to_numpy(), rtol=0, atol=1e-12, equal_nan=True
    )


def test_parity_fixture_exercises_the_features():
    _, x = runtime_path(runtime_payload(history(), safe_value()))
    row = x.loc[SAMPLE_ID]
    assert not row.isna().any()  # сравниваются реальные значения, а не NaN
    assert row["cur_dev_s"] == 137.0  # safe deviation, не official 999
    assert row["horizon_minutes"] == 12.5
    assert (row["T_hour"], row["T_minute"]) == (10.0, 0.0)  # +03:00 → naive UTC
    assert row["latest_packet_lag_s"] == 7.0
    assert row["latest_valid_gps_lag_s"] > row["latest_packet_lag_s"]
    for window in ("5m", "10m", "15m"):
        assert 0.0 < row[f"valid_gps_ratio_{window}"] < 1.0
    for window in ("5m", "10m"):
        assert 0.0 < row[f"zero_speed_fraction_{window}"] < 1.0
    assert row["speed_min_5m"] == 0.0 < row["speed_max_5m"]
    assert row["speed_std_3m"] > 0.0
    assert row["speed_trend_5m"] != 0.0
    assert row["speed_last"] == 11.5
    assert (row["target_lon"], row["target_lat"]) == (TARGET_LON, TARGET_LAT)
    assert row["distance_to_target_m"] > 0.0
    assert row["manual_fill"] == 1.0
    assert len({row[f"speed_mean_{w}"] for w in ("1m", "3m", "5m", "10m")}) == 4


def test_offline_override_wins_over_official_cur_dev_s():
    points, telemetry, plan = offline_frames(history())
    legacy = build_features_from_context(offline_context(points, telemetry, plan))
    offline_37, _ = offline_path(history())
    assert legacy.loc[SAMPLE_ID, "cur_dev_s"] == OFFICIAL_CUR_DEV
    assert offline_37.loc[SAMPLE_ID, "cur_dev_s"] == 137.0
    pd.testing.assert_frame_equal(
        legacy.drop(columns="cur_dev_s"), offline_37.drop(columns="cur_dev_s"), check_exact=True
    )


# ------------------------------------------------------------------ future telemetry


def test_future_runtime_telemetry_cannot_influence_features():
    legal = runtime_payload(history(), safe_value())
    with_future = runtime_payload(FUTURE_PACKETS[:1] + history() + FUTURE_PACKETS[1:], safe_value())
    assert len(with_future["telemetry"]) == len(legal["telemetry"]) + 2

    legal_37, legal_29 = runtime_path(legal)
    future_37, future_29 = runtime_path(with_future)
    pd.testing.assert_frame_equal(future_37, legal_37, check_exact=True)
    pd.testing.assert_frame_equal(future_29, legal_29, check_exact=True)

    legal_batch = runtime_context(PredictRequestV1.model_validate(legal).model_dump(mode="json"))
    future_batch = runtime_context(
        PredictRequestV1.model_validate(with_future).model_dump(mode="json")
    )
    assert len(future_batch.telemetry) == len(legal_batch.telemetry)  # будущие пакеты отброшены
    np.testing.assert_array_equal(future_batch.telemetry.speed, legal_batch.telemetry.speed)


def test_future_offline_telemetry_cannot_influence_features():
    legal_37, legal_29 = offline_path(history())
    future_37, future_29 = offline_path(history() + FUTURE_PACKETS)
    pd.testing.assert_frame_equal(future_37, legal_37, check_exact=True)
    pd.testing.assert_frame_equal(future_29, legal_29, check_exact=True)


# ------------------------------------------------------------------ runtime deviation mapping


class CapturingPredictor:
    def __init__(self) -> None:
        self.batches = []

    def is_ready(self) -> bool:
        return True

    def model_version(self) -> str:
        return "capture"

    def feature_schema_version(self) -> str:
        return FEATURE_SCHEMA_VERSION

    def predict(self, batch):
        self.batches.append(batch)
        return [0.0]


@pytest.mark.parametrize("deviation", [-123.5, 0, 274, 0.001])
def test_runtime_current_deviation_seconds_reaches_cur_dev_s(deviation):
    predictor = CapturingPredictor()
    request = PredictRequestV1.model_validate(runtime_payload(history(), deviation))
    InferenceService(predictor).predict(request)
    (batch,) = predictor.batches
    assert batch.points[0].current_deviation_s == float(deviation)
    features = build_features_from_context(batch)
    assert features.loc[SAMPLE_ID, "cur_dev_s"] == float(deviation)
    runtime_safe = project_runtime_safe_features(features)
    assert runtime_safe.loc[SAMPLE_ID, "cur_dev_s"] == float(deviation)
    assert list(runtime_safe.columns).count("cur_dev_s") == 1  # второго поля отклонения нет
