"""Канонический ML-контекст prediction point и единственный вход в Feature Builder.

Offline-данные и runtime-запрос Backend → ML Contract v1 нормализуются в одно
и то же представление (`CanonicalBatch`), которое затем подаётся в тот же
замороженный `build_features` через `build_features_from_context`. Здесь нет
ни одного признака: только типы, их валидация и упаковка в входной формат
builder'а.

Поля идентичности (`point_id`, `tr_id`, `target_action_id`, `unit_id`,
`route_id`) нужны для группировки telemetry и трассировки и не становятся
признаками.

Историческая зависимость `tabular-v1` (проверено по коду builder'а): все
оконные признаки используют не более 15 минут `(T-15m, T]`, но
`latest_packet_lag_s`, `latest_valid_gps_lag_s`, `latest_valid_lon/lat`
(→ `distance_to_target_m`) и `speed_last` берут последнее наблюдение
`<= T` без нижней границы. Поэтому telemetry здесь не обрезается по окну.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mostransport_ml.features.builder import build_features
from mostransport_ml.features.schema import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    RUNTIME_SAFE_FEATURE_NAMES,
    RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
    feature_names_for_schema,
)


def _naive_timestamp(value: pd.Timestamp, name: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or value is pd.NaT:
        raise TypeError(f"{name} must be a pandas Timestamp")
    if value.tzinfo is not None:
        raise ValueError(f"{name} must be timezone-naive")
    return value.as_unit("ns")


def _readonly(values: np.ndarray) -> np.ndarray:
    values = np.array(values, copy=True)
    values.flags.writeable = False
    return values


@dataclass(frozen=True)
class CanonicalPoint:
    """Одна логическая prediction point, независимо от источника.

    `target_lon`/`target_lat` — либо оба конечные, либо оба NaN (плановый
    контекст недоступен — возможно только offline, как и в M1).
    `manual_fill` — `None`, если плановый контекст недоступен.
    """

    point_id: str
    tr_id: str
    prediction_time: pd.Timestamp
    target_action_id: str
    target_time_begin: pd.Timestamp
    target_lon: float
    target_lat: float
    current_deviation_s: float
    manual_fill: bool | None
    unit_id: str | None = None
    route_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("point_id", "tr_id", "target_action_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("prediction_time", "target_time_begin"):
            object.__setattr__(self, name, _naive_timestamp(getattr(self, name), name))
        lon, lat = float(self.target_lon), float(self.target_lat)
        if math.isnan(lon) != math.isnan(lat):
            raise ValueError("target_lon and target_lat must be both known or both NaN")
        if not math.isnan(lon) and not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise ValueError("target coordinates out of range")
        object.__setattr__(self, "target_lon", lon)
        object.__setattr__(self, "target_lat", lat)
        object.__setattr__(self, "current_deviation_s", float(self.current_deviation_s))
        if self.manual_fill is not None and not isinstance(self.manual_fill, bool):
            raise TypeError("manual_fill must be bool or None")


@dataclass(frozen=True)
class CanonicalTelemetry:
    """Колоночная история telemetry (любые ТС, любое время), только поля M1.

    Порядок строк сохраняется как у источника: при равных `event_time`
    builder выбирает «последнее» наблюдение по этому порядку.
    `location_valid` — True/False/None (None трактуется builder'ом как False).
    """

    tr_id: np.ndarray
    event_time: np.ndarray
    location_valid: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    speed: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.tr_id)
        tr_id = np.asarray(self.tr_id, dtype=object)
        if not all(isinstance(v, str) and v for v in tr_id.tolist()):
            raise ValueError("telemetry.tr_id must contain non-empty strings")
        event_time = np.asarray(self.event_time)
        if event_time.dtype.kind != "M":
            raise TypeError("telemetry.event_time must be datetime64")
        event_time = event_time.astype("datetime64[ns]")
        if np.isnat(event_time).any():
            raise ValueError("telemetry.event_time contains missing timestamps")
        location_valid = np.asarray(self.location_valid, dtype=object)
        if not all(v is None or isinstance(v, bool) for v in location_valid.tolist()):
            raise TypeError("telemetry.location_valid must contain bool or None")
        columns = {
            "tr_id": tr_id,
            "event_time": event_time,
            "location_valid": location_valid,
            "lon": np.asarray(self.lon, dtype=float),
            "lat": np.asarray(self.lat, dtype=float),
            "speed": np.asarray(self.speed, dtype=float),
        }
        for name, values in columns.items():
            if values.shape != (n,):
                raise ValueError(f"telemetry.{name} must be 1-D with length {n}")
            object.__setattr__(self, name, _readonly(values))

    def __len__(self) -> int:
        return len(self.tr_id)


@dataclass(frozen=True)
class CanonicalBatch:
    """Упорядоченные prediction points + общая для них история telemetry."""

    points: tuple[CanonicalPoint, ...]
    telemetry: CanonicalTelemetry

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        ids = [p.point_id for p in self.points]
        if len(ids) != len(set(ids)):
            raise ValueError("point_id must be unique within a batch")


def _wkt_point(lon: float, lat: float) -> str | None:
    # repr(float) — кратчайшее точное представление: float(repr(x)) == x,
    # поэтому координаты проходят через WKT без потерь.
    if math.isnan(lon):
        return None
    return f"POINT ({float(lon)!r} {float(lat)!r})"


def to_builder_inputs(batch: CanonicalBatch) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Упаковать канонический контекст в входные DataFrame'ы замороженного builder'а.

    Плановый контекст передаётся строкой плана на каждую точку с
    синтетическим ключом (позиция точки), поэтому соответствие
    точка → цель однозначно и не зависит от идентификаторов источника.
    """
    points = batch.points
    keys = np.arange(len(points), dtype=np.int64)
    point_frame = pd.DataFrame(
        {
            "sample_id": [p.point_id for p in points],
            "tr_id": [p.tr_id for p in points],
            "T": pd.to_datetime([p.prediction_time for p in points]).astype("datetime64[ns]"),
            "target_stop_id": keys,
            "target_time_begin": pd.to_datetime([p.target_time_begin for p in points]).astype(
                "datetime64[ns]"
            ),
            "cur_dev_s": np.array([p.current_deviation_s for p in points], dtype=float),
        }
    )
    plan_frame = pd.DataFrame(
        {
            "tt_action_item_id": keys,
            "tr_id": [p.tr_id for p in points],
            "geom": pd.Series(
                [_wkt_point(p.target_lon, p.target_lat) for p in points], dtype=object
            ),
            "manual_fill": pd.Series([p.manual_fill for p in points], dtype=object),
        }
    )
    tel = batch.telemetry
    telemetry_frame = pd.DataFrame(
        {
            "tr_id": tel.tr_id,
            "event_time": tel.event_time,
            "location_valid": tel.location_valid,
            "lon": tel.lon,
            "lat": tel.lat,
            "speed": tel.speed,
        }
    )
    return point_frame, telemetry_frame, plan_frame


def build_features_from_context(batch: CanonicalBatch) -> pd.DataFrame:
    """Единственный путь от канонического контекста к признакам `tabular-v1`."""
    return build_features(*to_builder_inputs(batch))


def project_runtime_safe_features(features: pd.DataFrame) -> pd.DataFrame:
    """Единственная проекция `tabular-v1` (37) → `runtime-safe-v1` (29).

    Вход — таблица признаков канонического builder'а: колонки строго
    `FEATURE_NAMES` в замороженном порядке, без дублей. Любой дрейф схемы
    (отсутствующая, лишняя, переставленная или повторённая колонка) —
    `ValueError`. Проекция только выбирает колонки `RUNTIME_SAFE_FEATURE_NAMES`:
    ничего не вычисляет, не меняет значения и dtype, не заполняет NaN, сохраняет
    индекс и не мутирует вход (возвращается новая таблица).
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    columns = features.columns
    duplicated = sorted(map(str, columns[columns.duplicated()].unique()))
    if duplicated:
        raise ValueError(f"tabular-v1 features contain duplicate columns: {duplicated}")
    missing = [name for name in FEATURE_NAMES if name not in columns]
    if missing:
        raise ValueError(f"tabular-v1 features are missing columns: {missing}")
    unexpected = [name for name in columns if name not in FEATURE_NAMES]
    if unexpected:
        raise ValueError(f"tabular-v1 features contain unexpected columns: {unexpected}")
    if tuple(columns) != FEATURE_NAMES:
        raise ValueError("tabular-v1 feature columns are not in the frozen FEATURE_NAMES order")
    return features.loc[:, list(RUNTIME_SAFE_FEATURE_NAMES)].copy()


def features_for_schema(
    tabular_features: pd.DataFrame, feature_schema_version: str
) -> pd.DataFrame:
    """Канонические признаки `tabular-v1` → признаки схемы, которую потребляет модель.

    `tabular-v1` возвращается как есть (после проверки точных колонок);
    `runtime-safe-v1` — только через `project_runtime_safe_features`.
    Неподдерживаемая схема — `UnsupportedFeatureSchemaError`.
    """
    feature_names_for_schema(feature_schema_version)
    if feature_schema_version == RUNTIME_SAFE_FEATURE_SCHEMA_VERSION:
        return project_runtime_safe_features(tabular_features)
    if feature_schema_version == FEATURE_SCHEMA_VERSION:
        if not isinstance(tabular_features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if tuple(tabular_features.columns) != FEATURE_NAMES:
            raise ValueError("tabular-v1 feature columns differ from the frozen FEATURE_NAMES")
        return tabular_features
    raise RuntimeError(f"no transformation for supported schema {feature_schema_version!r}")
