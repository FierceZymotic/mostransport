"""Нормализующие адаптеры: offline-данные и Contract v1 → `CanonicalBatch`.

Адаптеры не считают признаки, не обрезают историю окном и не вычисляют
доменные факты Backend'а (matching, текущее отклонение, eligibility): они
только проверяют и переупаковывают уже готовые значения.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from mostransport_ml.features.builder import ForbiddenInputColumnError
from mostransport_ml.features.context import CanonicalBatch, CanonicalPoint, CanonicalTelemetry
from mostransport_ml.features.schema import FORBIDDEN_INPUT_COLUMNS
from mostransport_ml.features.spatial import parse_point_wkt

OFFLINE_POINT_COLUMNS = (
    "sample_id",
    "tr_id",
    "T",
    "target_stop_id",
    "target_time_begin",
    "cur_dev_s",
)
OFFLINE_TELEMETRY_COLUMNS = ("tr_id", "event_time", "location_valid", "lon", "lat", "speed")
OFFLINE_PLAN_COLUMNS = ("tt_action_item_id", "tr_id", "geom", "manual_fill")

RUNTIME_TOP_LEVEL = (
    "request_id",
    "prediction_time",
    "vehicle_context",
    "schedule_context",
    "telemetry",
)
RUNTIME_VEHICLE_FIELDS = ("unit_id", "tr_id", "route_id")
RUNTIME_SCHEDULE_FIELDS = (
    "target_action_id",
    "target_time_begin",
    "target_lat",
    "target_lon",
    "current_deviation_seconds",
    "manual_fill",
)
# Поля пакета, которые читает tabular-v1. Остальные поля (heading и любые
# дополнительные сырые поля) допускаются и игнорируются.
RUNTIME_PACKET_FIELDS = ("event_time", "location_valid", "lat", "lon", "speed")


class ContextValidationError(ValueError):
    """Runtime-запрос не содержит корректного контекста для прогноза.

    Сообщение содержит путь к полю и причину, но не значения запроса.
    """


# --------------------------------------------------------------------------- offline


def _reject_forbidden_columns(frame: pd.DataFrame, name: str) -> None:
    forbidden = sorted(FORBIDDEN_INPUT_COLUMNS & set(frame.columns))
    if forbidden:
        raise ForbiddenInputColumnError(
            f"{name} contains forbidden factual/target columns: {forbidden}"
        )


def _require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _optional_bool(value: Any) -> bool | None:
    return None if pd.isna(value) else bool(value)


def _examples(values: Iterable[Any]) -> list[str]:
    return sorted(map(str, values))[:5]


def _deviation_by_sample_id(
    sample_ids: pd.Series, override: pd.Series | Mapping[Any, Any]
) -> dict[Any, float]:
    """Проверить override текущего отклонения и вернуть `sample_id → value`.

    Сопоставление только по `sample_id`: ровно одно конечное число на каждую
    точку, без отсутствующих и лишних ключей. Позиционного выравнивания нет.
    """
    if isinstance(override, pd.Series):
        keys, values = override.index.tolist(), override.tolist()
    elif isinstance(override, Mapping):
        keys, values = list(override.keys()), list(override.values())
    else:
        raise TypeError(
            "current_deviation_seconds must be a pandas Series or a mapping keyed by sample_id"
        )
    if sample_ids.isna().any():
        raise ValueError("points.sample_id contains missing values")
    if sample_ids.duplicated().any():
        raise ValueError(
            "points.sample_id must be unique to align current_deviation_seconds; duplicated: "
            f"{_examples(sample_ids[sample_ids.duplicated()].unique())}"
        )
    key_index = pd.Index(keys, dtype=object)
    if key_index.has_duplicates:
        raise ValueError(
            "current_deviation_seconds has duplicated sample_id keys: "
            f"{_examples(key_index[key_index.duplicated()].unique())}"
        )
    for value in values:
        if (
            isinstance(value, bool | np.bool_)
            or not isinstance(value, int | float | np.integer | np.floating)
            or not math.isfinite(value)
        ):
            raise ValueError("current_deviation_seconds must contain only finite numbers")
    lookup = dict(zip(keys, (float(v) for v in values), strict=True))
    expected = sample_ids.tolist()
    missing = [s for s in expected if s not in lookup]
    if missing:
        raise ValueError(
            f"current_deviation_seconds is missing {len(missing)} sample_id(s): "
            f"{_examples(missing)}"
        )
    extra = set(lookup) - set(expected)
    if extra:
        raise ValueError(
            f"current_deviation_seconds has {len(extra)} sample_id(s) that are not "
            f"prediction points: {_examples(extra)}"
        )
    return lookup


def offline_context(
    points: pd.DataFrame,
    telemetry: pd.DataFrame,
    schedule_plan: pd.DataFrame,
    *,
    current_deviation_seconds: pd.Series | Mapping[Any, Any] | None = None,
) -> CanonicalBatch:
    """Официальные offline-кадры (как из `load_official_split`) → `CanonicalBatch`.

    Плановый контекст целевой остановки ищется по `(target_stop_id, tr_id)`
    так же, как в M1; отсутствующий план даёт NaN-координаты и
    `manual_fill=None`. Telemetry передаётся целиком в исходном порядке:
    cutoff `event_time <= T` для каждой точки применяет builder.

    Текущее отклонение точки (`CanonicalPoint.current_deviation_s` → признак
    `cur_dev_s`) по умолчанию берётся из `points.cur_dev_s`, как в M1.
    Keyword-only `current_deviation_seconds` явно заменяет его (например
    point-in-time-safe значениями `data.safe_deviation`): Series или mapping
    `sample_id → секунды`, сопоставляемые строго по `sample_id`, ровно одно
    конечное число на точку. В этом режиме `points.cur_dev_s` не читается и
    не обязателен.
    """
    for frame, name in (
        (points, "points"),
        (telemetry, "telemetry"),
        (schedule_plan, "schedule_plan"),
    ):
        _reject_forbidden_columns(frame, name)
    point_columns = OFFLINE_POINT_COLUMNS
    if current_deviation_seconds is not None:
        point_columns = tuple(c for c in OFFLINE_POINT_COLUMNS if c != "cur_dev_s")
    _require_columns(points, point_columns, "points")
    _require_columns(telemetry, OFFLINE_TELEMETRY_COLUMNS, "telemetry")
    _require_columns(schedule_plan, OFFLINE_PLAN_COLUMNS, "schedule_plan")
    deviation_override = (
        None
        if current_deviation_seconds is None
        else _deviation_by_sample_id(points["sample_id"], current_deviation_seconds)
    )

    plan = schedule_plan[list(OFFLINE_PLAN_COLUMNS)]
    if plan.duplicated(["tt_action_item_id", "tr_id"]).any():
        raise ValueError("schedule_plan has duplicated (tt_action_item_id, tr_id) keys")
    merged = points[list(point_columns)].merge(
        plan,
        left_on=["target_stop_id", "tr_id"],
        right_on=["tt_action_item_id", "tr_id"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(points):
        raise RuntimeError("plan context merge changed the number of points")

    canonical_points = []
    for row in merged.itertuples(index=False):
        lon, lat = parse_point_wkt(row.geom)
        canonical_points.append(
            CanonicalPoint(
                point_id=str(row.sample_id),
                tr_id=str(row.tr_id),
                prediction_time=pd.Timestamp(row.T),
                target_action_id=str(row.target_stop_id),
                target_time_begin=pd.Timestamp(row.target_time_begin),
                target_lon=lon,
                target_lat=lat,
                current_deviation_s=(
                    row.cur_dev_s
                    if deviation_override is None
                    else deviation_override[row.sample_id]
                ),
                manual_fill=_optional_bool(row.manual_fill),
            )
        )

    if not pd.api.types.is_datetime64_dtype(telemetry["event_time"]):
        raise TypeError("telemetry.event_time must be a timezone-naive datetime64 column")
    canonical_telemetry = CanonicalTelemetry(
        tr_id=np.array([str(v) for v in telemetry["tr_id"].tolist()], dtype=object),
        event_time=telemetry["event_time"].to_numpy(dtype="datetime64[ns]"),
        location_valid=np.array(
            [_optional_bool(v) for v in telemetry["location_valid"].tolist()], dtype=object
        ),
        lon=telemetry["lon"].to_numpy(dtype=float),
        lat=telemetry["lat"].to_numpy(dtype=float),
        speed=telemetry["speed"].to_numpy(dtype=float),
    )
    return CanonicalBatch(points=tuple(canonical_points), telemetry=canonical_telemetry)


# --------------------------------------------------------------------------- runtime


def _find_forbidden_keys(value: Any, path: str) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_INPUT_COLUMNS:
                yield child_path
            yield from _find_forbidden_keys(child, child_path)
    elif isinstance(value, list | tuple):
        for i, child in enumerate(value):
            yield from _find_forbidden_keys(child, f"{path}[{i}]")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContextValidationError(f"{path}: must be an object")
    return value


def _field(obj: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        raise ContextValidationError(f"{path}.{key}: required field is missing")
    return obj[key]


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextValidationError(f"{path}: must be a non-empty string")
    return value


def _timestamp(value: Any, path: str) -> pd.Timestamp:
    """ISO-8601 строка или datetime; только timezone-aware → naive UTC.

    Официальные naive-времена датасета — это UTC wall-clock без метки
    пояса, и builder работает в этом представлении. Runtime-время обязано
    быть aware (например `...Z` или `...+03:00`): оно переводится в UTC и
    лишается метки пояса, поэтому эквивалентные моменты дают одно и то же
    каноническое значение. Naive runtime-время неоднозначно и отклоняется.
    """
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ContextValidationError(f"{path}: must be an ISO-8601 datetime") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ContextValidationError(f"{path}: must be an ISO-8601 datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContextValidationError(
            f"{path}: timezone-naive datetimes are ambiguous; send timezone-aware ISO-8601 "
            "(e.g. 2026-01-06T03:35:00Z)"
        )
    return pd.Timestamp(parsed).tz_convert("UTC").tz_localize(None).as_unit("ns")


def _number(value: Any, path: str, *, nullable: bool) -> float:
    if value is None and nullable:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContextValidationError(f"{path}: must be a number" + (" or null" if nullable else ""))
    number = float(value)
    if not math.isfinite(number):
        raise ContextValidationError(f"{path}: must be finite")
    return number


def _bool(value: Any, path: str, *, nullable: bool) -> bool | None:
    if value is None and nullable:
        return None
    if not isinstance(value, bool):
        raise ContextValidationError(
            f"{path}: must be a boolean" + (" or null" if nullable else "")
        )
    return value


def runtime_context(request: Mapping[str, Any]) -> CanonicalBatch:
    """Backend → ML Contract v1 запрос → `CanonicalBatch` из одной точки.

    Point-in-time: пакеты с `event_time > prediction_time` отбрасываются
    здесь (и дополнительно отсекаются builder'ом); `event_time ==
    prediction_time` сохраняется. История не обрезается никаким окном.
    """
    request = _mapping(request, "request")
    forbidden = list(_find_forbidden_keys(request, ""))
    if forbidden:
        raise ForbiddenInputColumnError(
            f"request contains forbidden factual/target fields: {forbidden}"
        )
    for key in RUNTIME_TOP_LEVEL:
        _field(request, key, "request")

    request_id = _string(request["request_id"], "request.request_id")
    prediction_time = _timestamp(request["prediction_time"], "request.prediction_time")

    vehicle = _mapping(request["vehicle_context"], "request.vehicle_context")
    unit_id, tr_id, route_id = (
        _string(_field(vehicle, k, "request.vehicle_context"), f"request.vehicle_context.{k}")
        for k in RUNTIME_VEHICLE_FIELDS
    )

    schedule = _mapping(request["schedule_context"], "request.schedule_context")
    sp = "request.schedule_context"
    target_action_id = _string(_field(schedule, "target_action_id", sp), f"{sp}.target_action_id")
    target_time_begin = _timestamp(
        _field(schedule, "target_time_begin", sp), f"{sp}.target_time_begin"
    )
    target_lat = _number(_field(schedule, "target_lat", sp), f"{sp}.target_lat", nullable=False)
    target_lon = _number(_field(schedule, "target_lon", sp), f"{sp}.target_lon", nullable=False)
    if not (-90.0 <= target_lat <= 90.0 and -180.0 <= target_lon <= 180.0):
        raise ContextValidationError(f"{sp}: target coordinates out of range")
    current_deviation_s = _number(
        _field(schedule, "current_deviation_seconds", sp),
        f"{sp}.current_deviation_seconds",
        nullable=False,
    )
    manual_fill = _bool(_field(schedule, "manual_fill", sp), f"{sp}.manual_fill", nullable=False)

    packets = request["telemetry"]
    if not isinstance(packets, list):
        raise ContextValidationError("request.telemetry: must be an array")
    event_times, location_valid, lon, lat, speed = [], [], [], [], []
    for i, raw_packet in enumerate(packets):
        path = f"request.telemetry[{i}]"
        packet = _mapping(raw_packet, path)
        for key in RUNTIME_PACKET_FIELDS:
            _field(packet, key, path)
        event_time = _timestamp(packet["event_time"], f"{path}.event_time")
        values = (
            _bool(packet["location_valid"], f"{path}.location_valid", nullable=True),
            _number(packet["lon"], f"{path}.lon", nullable=True),
            _number(packet["lat"], f"{path}.lat", nullable=True),
            _number(packet["speed"], f"{path}.speed", nullable=True),
        )
        if event_time > prediction_time:
            continue  # будущий пакет не может влиять на контекст точки
        event_times.append(event_time.to_datetime64())
        location_valid.append(values[0])
        lon.append(values[1])
        lat.append(values[2])
        speed.append(values[3])

    point = CanonicalPoint(
        point_id=request_id,
        tr_id=tr_id,
        prediction_time=prediction_time,
        target_action_id=target_action_id,
        target_time_begin=target_time_begin,
        target_lon=target_lon,
        target_lat=target_lat,
        current_deviation_s=current_deviation_s,
        manual_fill=manual_fill,
        unit_id=unit_id,
        route_id=route_id,
    )
    telemetry = CanonicalTelemetry(
        tr_id=np.array([tr_id] * len(event_times), dtype=object),
        event_time=np.array(event_times, dtype="datetime64[ns]"),
        location_valid=np.array(location_valid, dtype=object),
        lon=np.array(lon, dtype=float),
        lat=np.array(lat, dtype=float),
        speed=np.array(speed, dtype=float),
    )
    return CanonicalBatch(points=(point,), telemetry=telemetry)
