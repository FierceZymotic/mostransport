"""Пространственные helpers: WKT `POINT (lon lat)` и haversine-дистанция."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

EARTH_MEAN_RADIUS_M = 6_371_008.8

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_POINT_RE = re.compile(rf"^\s*POINT\s*\(\s*({_NUMBER})\s+({_NUMBER})\s*\)\s*$", re.IGNORECASE)
_POINT_EMPTY_RE = re.compile(r"^\s*POINT\s+EMPTY\s*$", re.IGNORECASE)


def parse_point_wkt(value: Any) -> tuple[float, float]:
    """Разобрать официальный `POINT (lon lat)` в `(lon, lat)`.

    Семантика:
    - отсутствующее значение (None/NaN/пустая строка/`POINT EMPTY`) →
      `(nan, nan)`;
    - любое другое значение, не являющееся корректным `POINT (lon lat)` с
      lon ∈ [-180, 180] и lat ∈ [-90, 90] → `ValueError` (fail fast, чтобы
      испорченная геометрия не превращалась молча в признак).
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (math.nan, math.nan)
    if not isinstance(value, str):
        raise ValueError(f"geom must be a WKT string, got {type(value).__name__}")
    if not value.strip() or _POINT_EMPTY_RE.match(value):
        return (math.nan, math.nan)
    match = _POINT_RE.match(value)
    if match is None:
        raise ValueError(f"Malformed geom, expected 'POINT (lon lat)': {value!r}")
    lon, lat = float(match.group(1)), float(match.group(2))
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError(f"geom coordinates out of range: {value!r}")
    return (lon, lat)


def haversine_m(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> np.ndarray:
    """Дистанция по большому кругу в метрах; NaN на входе → NaN на выходе."""
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2)
    )
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * EARTH_MEAN_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
