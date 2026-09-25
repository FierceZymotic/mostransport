"""Point-in-time Feature Builder `tabular-v1`.

Builder не читает файлы и не знает, откуда пришли данные: он получает
prediction points, telemetry и плановый schedule-контекст явно. Это
позволяет позже применить тот же код online (точка + недавнее окно
telemetry + плановый контекст) без второй реализации признаков.

Speed/zero-speed признаки здесь — model-specific преобразования сырой
telemetry, а не попытка дублировать будущие domain-агрегаты backend'а
(segment speed, dwell и т.п.).

Инварианты, обеспечиваемые кодом:
- используется только telemetry с `event_time <= T`, даже если вызывающий
  передал более поздние строки;
- окно шириной w — полуинтервал `(T - w, T]`;
- входы с `time_fact_begin`/`target_delay_s`/`target_class` отклоняются;
- порядок строк выхода = порядок входных точек, индекс — `sample_id`;
- колонки выхода = `FEATURE_NAMES` в зафиксированном порядке.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from mostransport_ml.features.schema import (
    FEATURE_NAMES,
    FORBIDDEN_INPUT_COLUMNS,
    HORIZON_MAX_INCLUSIVE_MINUTES,
    HORIZON_MIN_EXCLUSIVE_MINUTES,
)
from mostransport_ml.features.spatial import haversine_m, parse_point_wkt

POINT_INPUT_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "tr_id",
    "T",
    "target_stop_id",
    "target_time_begin",
    "cur_dev_s",
)
TELEMETRY_INPUT_COLUMNS: tuple[str, ...] = (
    "tr_id",
    "event_time",
    "location_valid",
    "lon",
    "lat",
    "speed",
)
PLAN_INPUT_COLUMNS: tuple[str, ...] = ("tt_action_item_id", "tr_id", "geom", "manual_fill")

ROW_WINDOWS_MIN: tuple[int, ...] = (1, 3, 5, 10, 15)
GPS_WINDOWS_MIN: tuple[int, ...] = (5, 10, 15)
SPEED_MEAN_WINDOWS_MIN: tuple[int, ...] = (1, 3, 5, 10)
SPEED_STD_WINDOWS_MIN: tuple[int, ...] = (3, 5, 10)
ZERO_SPEED_WINDOWS_MIN: tuple[int, ...] = (5, 10)

_NON_TELEMETRY_FEATURES = frozenset(
    {
        "cur_dev_s",
        "horizon_minutes",
        "T_hour",
        "T_minute",
        "T_minutes_since_midnight",
        "target_lon",
        "target_lat",
        "distance_to_target_m",
        "manual_fill",
    }
)
TELEMETRY_FEATURE_NAMES: tuple[str, ...] = tuple(
    n for n in FEATURE_NAMES if n not in _NON_TELEMETRY_FEATURES
)

_NS_PER_S = 1_000_000_000
_NS_PER_MIN = 60 * _NS_PER_S


class ForbiddenInputColumnError(ValueError):
    """Вход Feature Builder'а содержит factual/target-колонку."""


def gps_valid_mask(telemetry: pd.DataFrame) -> np.ndarray:
    """Строгая валидность GPS: `location_valid & lon.notna() & lat.notna()`."""
    location_valid = (
        telemetry["location_valid"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    return (
        location_valid & telemetry["lon"].notna().to_numpy() & telemetry["lat"].notna().to_numpy()
    )


def _reject_forbidden(df: pd.DataFrame, name: str) -> None:
    forbidden = sorted(FORBIDDEN_INPUT_COLUMNS & set(df.columns))
    if forbidden:
        raise ForbiddenInputColumnError(
            f"{name} contains forbidden factual/target columns: {forbidden}"
        )


def _require(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _datetime_ns(series: pd.Series, name: str) -> np.ndarray:
    if not pd.api.types.is_datetime64_dtype(series):
        raise TypeError(f"{name} must be a timezone-naive datetime64 column, got {series.dtype}")
    if series.isna().any():
        raise ValueError(f"{name} contains missing timestamps")
    return series.to_numpy(dtype="datetime64[ns]").view("int64")


def _vehicle_bounds(tr_sorted: np.ndarray) -> dict:
    if tr_sorted.size == 0:
        return {}
    starts = np.concatenate(([0], np.flatnonzero(tr_sorted[1:] != tr_sorted[:-1]) + 1))
    ends = np.concatenate((starts[1:], [tr_sorted.size]))
    return {tr_sorted[s]: (int(s), int(e)) for s, e in zip(starts, ends, strict=True)}


def _speed_trend_per_min(t_min: np.ndarray, speed: np.ndarray) -> float:
    """OLS-наклон speed по времени (мин относительно T), км/ч в минуту."""
    if speed.size < 2:
        return math.nan
    t_centered = t_min - t_min.mean()
    denom = float(np.sum(t_centered**2))
    if denom == 0.0:
        return math.nan
    return float(np.sum(t_centered * (speed - speed.mean())) / denom)


def _point_features(
    t_ns: int, ev: np.ndarray, gv: np.ndarray, sp: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> dict[str, float]:
    """Признаки одной точки по истории одного ТС (массивы отсортированы по времени)."""
    end = int(np.searchsorted(ev, t_ns, side="right"))  # жёсткий cutoff event_time <= T
    ev, gv, sp, lon, lat = ev[:end], gv[:end], sp[:end], lon[:end], lat[:end]
    out: dict[str, float] = {}

    def window_start(minutes: int) -> int:
        return int(np.searchsorted(ev, t_ns - minutes * _NS_PER_MIN, side="right"))

    out["latest_packet_lag_s"] = (t_ns - ev[-1]) / _NS_PER_S if end else math.nan
    valid_idx = np.flatnonzero(gv)
    if valid_idx.size:
        last_valid = valid_idx[-1]
        out["latest_valid_gps_lag_s"] = (t_ns - ev[last_valid]) / _NS_PER_S
        out["latest_valid_lon"] = float(lon[last_valid])
        out["latest_valid_lat"] = float(lat[last_valid])
    else:
        out["latest_valid_gps_lag_s"] = math.nan
        out["latest_valid_lon"] = math.nan
        out["latest_valid_lat"] = math.nan

    for w in ROW_WINDOWS_MIN:
        out[f"rows_{w}m"] = float(end - window_start(w))
    for w in GPS_WINDOWS_MIN:
        start = window_start(w)
        n_rows = end - start
        n_valid = int(gv[start:].sum())
        out[f"valid_gps_count_{w}m"] = float(n_valid)
        out[f"valid_gps_ratio_{w}m"] = n_valid / n_rows if n_rows else math.nan

    speed_idx = np.flatnonzero(~np.isnan(sp))
    out["speed_last"] = float(sp[speed_idx[-1]]) if speed_idx.size else math.nan

    def window_speeds(minutes: int) -> tuple[np.ndarray, np.ndarray]:
        start = window_start(minutes)
        s, t = sp[start:], ev[start:]
        keep = ~np.isnan(s)
        return s[keep], t[keep]

    for w in SPEED_MEAN_WINDOWS_MIN:
        s, _ = window_speeds(w)
        out[f"speed_mean_{w}m"] = float(s.mean()) if s.size else math.nan
    for w in SPEED_STD_WINDOWS_MIN:
        s, _ = window_speeds(w)
        out[f"speed_std_{w}m"] = float(s.std(ddof=1)) if s.size >= 2 else math.nan
    s5, t5 = window_speeds(5)
    out["speed_min_5m"] = float(s5.min()) if s5.size else math.nan
    out["speed_max_5m"] = float(s5.max()) if s5.size else math.nan
    out["speed_trend_5m"] = _speed_trend_per_min((t5 - t_ns) / _NS_PER_MIN, s5)
    for w in ZERO_SPEED_WINDOWS_MIN:
        s, _ = window_speeds(w)
        out[f"zero_speed_fraction_{w}m"] = float(np.mean(s == 0.0)) if s.size else math.nan
    return out


def _plan_context(points: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Плановый контекст целевой остановки точки: (target_lon, target_lat, manual_fill)."""
    plan = plan[list(PLAN_INPUT_COLUMNS)]
    if plan.duplicated(["tt_action_item_id", "tr_id"]).any():
        raise ValueError("schedule_plan has duplicated (tt_action_item_id, tr_id) keys")
    merged = points[["target_stop_id", "tr_id"]].merge(
        plan,
        left_on=["target_stop_id", "tr_id"],
        right_on=["tt_action_item_id", "tr_id"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(points):
        raise RuntimeError("plan context merge changed the number of points")
    coords = [parse_point_wkt(v) for v in merged["geom"].tolist()]
    manual_fill = (
        merged["manual_fill"]
        .astype("boolean")
        .astype("Float64")
        .to_numpy(dtype=float, na_value=np.nan)
    )
    return pd.DataFrame(
        {
            "target_lon": [c[0] for c in coords],
            "target_lat": [c[1] for c in coords],
            "manual_fill": manual_fill,
        }
    )


def build_features(
    points: pd.DataFrame, telemetry: pd.DataFrame, schedule_plan: pd.DataFrame
) -> pd.DataFrame:
    """Построить один ряд `FEATURE_NAMES` на каждую prediction point.

    `points`: `POINT_INPUT_COLUMNS` (`target_stop_id` используется только
    для поиска планового контекста, в признаки не попадает).
    `telemetry`: `TELEMETRY_INPUT_COLUMNS`, может содержать любые ТС и время.
    `schedule_plan`: `PLAN_INPUT_COLUMNS`, только плановые поля.

    Возвращает DataFrame с индексом `sample_id` (порядок входа сохранён) и
    колонками строго `FEATURE_NAMES`.
    """
    for df, name in (
        (points, "points"),
        (telemetry, "telemetry"),
        (schedule_plan, "schedule_plan"),
    ):
        _reject_forbidden(df, name)
    _require(points, POINT_INPUT_COLUMNS, "points")
    _require(telemetry, TELEMETRY_INPUT_COLUMNS, "telemetry")
    _require(schedule_plan, PLAN_INPUT_COLUMNS, "schedule_plan")
    if points["sample_id"].duplicated().any():
        raise ValueError("points.sample_id must be unique")

    t_ns = _datetime_ns(points["T"], "points.T")
    target_ns = _datetime_ns(points["target_time_begin"], "points.target_time_begin")
    horizon = (target_ns - t_ns) / _NS_PER_MIN
    out_of_range = ~(
        (horizon > HORIZON_MIN_EXCLUSIVE_MINUTES) & (horizon <= HORIZON_MAX_INCLUSIVE_MINUTES)
    )
    if out_of_range.any():
        bad = points.loc[out_of_range, "sample_id"].tolist()[:5]
        raise ValueError(f"horizon_minutes outside (10, 15] for sample_id(s): {bad}")

    ev_all = _datetime_ns(telemetry["event_time"], "telemetry.event_time")
    tr_all = telemetry["tr_id"].to_numpy()
    order = np.lexsort((ev_all, tr_all))
    tr_sorted = tr_all[order]
    ev = ev_all[order]
    gv = gps_valid_mask(telemetry)[order]
    sp = telemetry["speed"].to_numpy(dtype=float)[order]
    lon = telemetry["lon"].to_numpy(dtype=float)[order]
    lat = telemetry["lat"].to_numpy(dtype=float)[order]
    bounds = _vehicle_bounds(tr_sorted)

    empty_i = np.empty(0, dtype=np.int64)
    empty_b = np.empty(0, dtype=bool)
    empty_f = np.empty(0, dtype=float)
    rows = []
    for tr_id, point_t in zip(points["tr_id"].tolist(), t_ns.tolist(), strict=True):
        span = bounds.get(tr_id)
        if span is None:
            rows.append(_point_features(point_t, empty_i, empty_b, empty_f, empty_f, empty_f))
            continue
        s, e = span
        rows.append(_point_features(point_t, ev[s:e], gv[s:e], sp[s:e], lon[s:e], lat[s:e]))

    telemetry_features = pd.DataFrame.from_records(rows, columns=list(TELEMETRY_FEATURE_NAMES))
    plan_features = _plan_context(points, schedule_plan)
    t_index = pd.DatetimeIndex(points["T"])

    features = pd.DataFrame(
        {
            "cur_dev_s": points["cur_dev_s"].to_numpy(dtype=float),
            "horizon_minutes": horizon,
            "T_hour": t_index.hour.to_numpy(dtype=float),
            "T_minute": t_index.minute.to_numpy(dtype=float),
            "T_minutes_since_midnight": (
                t_index.hour * 60 + t_index.minute + t_index.second / 60.0
            ).to_numpy(dtype=float),
        }
    )
    features = pd.concat([features, telemetry_features, plan_features], axis=1)
    features["distance_to_target_m"] = haversine_m(
        features["latest_valid_lon"],
        features["latest_valid_lat"],
        features["target_lon"],
        features["target_lat"],
    )
    features.index = pd.Index(points["sample_id"].to_numpy(), name="sample_id")
    return features[list(FEATURE_NAMES)].astype(float)
