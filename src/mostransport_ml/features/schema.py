"""Зафиксированная схема признаков `tabular-v1`.

Порядок `FEATURE_NAMES` — часть контракта: модель обучается и применяется
к колонкам ровно в этом порядке. Идентичность точки (`sample_id`) —
provenance, а не признак.
"""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = "tabular-v1"

FEATURE_NAMES: tuple[str, ...] = (
    # point
    "cur_dev_s",
    "horizon_minutes",
    "T_hour",
    "T_minute",
    "T_minutes_since_midnight",
    # freshness / observation quality
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
    # speed
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
    # spatial
    "target_lon",
    "target_lat",
    "latest_valid_lon",
    "latest_valid_lat",
    "distance_to_target_m",
    # schedule (planned)
    "manual_fill",
)

# Поля, которые не имеют права попадать ни в один вход Feature Builder'а.
FORBIDDEN_INPUT_COLUMNS: frozenset[str] = frozenset(
    {"time_fact_begin", "target_delay_s", "target_class"}
)

HORIZON_MIN_EXCLUSIVE_MINUTES = 10.0
HORIZON_MAX_INCLUSIVE_MINUTES = 15.0
