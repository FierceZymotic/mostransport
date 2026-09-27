"""Зафиксированные схемы признаков: `tabular-v1` и `runtime-safe-v1`.

Порядок `FEATURE_NAMES` — часть контракта: модель обучается и применяется
к колонкам ровно в этом порядке. Идентичность точки (`sample_id`) —
provenance, а не признак.

`runtime-safe-v1` не вычисляется отдельно: это строгая проекция выхода того
же канонического builder'а (`features.context.project_runtime_safe_features`).
Какую схему потребляет модель, объявляет её проверенный manifest
(`feature_schema_version`); `feature_names_for_schema` — единственный lookup
поддерживаемых схем.
"""

from __future__ import annotations

from types import MappingProxyType

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

RUNTIME_SAFE_FEATURE_SCHEMA_VERSION = "runtime-safe-v1"

# Восемь raw packet-count признаков `tabular-v1`: их распределение зависит от
# частоты telemetry (исторический CSV ~12–15 с между пакетами; runtime cadence
# не гарантирован и может отличаться), поэтому в `runtime-safe-v1` они не входят.
RUNTIME_SAFE_EXCLUDED_FEATURES: tuple[str, ...] = (
    "rows_1m",
    "rows_3m",
    "rows_5m",
    "rows_10m",
    "rows_15m",
    "valid_gps_count_5m",
    "valid_gps_count_10m",
    "valid_gps_count_15m",
)

# `runtime-safe-v1` (research F0): 29 признаков `tabular-v1` без raw counts, в
# исходном относительном порядке. Значения — ровно значения `tabular-v1`.
#
# Семантика `cur_dev_s` в этой схеме: point-in-time-safe текущее отклонение,
# переданное в каноническую prediction point (`CanonicalPoint.current_deviation_s`),
# а НЕ official historical `cur_dev_s` датасета. Источник offline —
# `data.safe_deviation` (только факты с `time_fact_begin <= T`), переданный через
# `offline_context(..., current_deviation_seconds=...)`; источник runtime —
# `PredictRequestV1.schedule_context.current_deviation_seconds`.
RUNTIME_SAFE_FEATURE_NAMES: tuple[str, ...] = (
    # point
    "cur_dev_s",
    "horizon_minutes",
    "T_hour",
    "T_minute",
    "T_minutes_since_midnight",
    # freshness / observation quality
    "latest_packet_lag_s",
    "latest_valid_gps_lag_s",
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

# Схемы признаков, которые этот код умеет подавать модели: версия → точный
# упорядоченный список признаков. Других схем нет.
SUPPORTED_FEATURE_SCHEMAS = MappingProxyType(
    {
        FEATURE_SCHEMA_VERSION: FEATURE_NAMES,
        RUNTIME_SAFE_FEATURE_SCHEMA_VERSION: RUNTIME_SAFE_FEATURE_NAMES,
    }
)
SUPPORTED_FEATURE_SCHEMA_VERSIONS: frozenset[str] = frozenset(SUPPORTED_FEATURE_SCHEMAS)


class UnsupportedFeatureSchemaError(ValueError):
    """`feature_schema_version` не поддерживается этим кодом."""


def feature_names_for_schema(version: str) -> tuple[str, ...]:
    """Точный упорядоченный список признаков поддерживаемой схемы; иначе ошибка."""
    if not isinstance(version, str) or version not in SUPPORTED_FEATURE_SCHEMAS:
        raise UnsupportedFeatureSchemaError(
            f"unsupported feature_schema_version {version!r}; "
            f"supported: {sorted(SUPPORTED_FEATURE_SCHEMA_VERSIONS)}"
        )
    return SUPPORTED_FEATURE_SCHEMAS[version]
