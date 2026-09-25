"""Тесты разбора WKT `POINT (lon lat)` и haversine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mostransport_ml.features.spatial import haversine_m, parse_point_wkt


def test_parse_official_point_format():
    assert parse_point_wkt("POINT (37.43070705 55.8040083)") == (37.43070705, 55.8040083)
    assert parse_point_wkt("  point(-1.5e1  2)  ") == (-15.0, 2.0)


@pytest.mark.parametrize("missing", [None, math.nan, "", "   ", "POINT EMPTY"])
def test_missing_geom_is_nan(missing):
    lon, lat = parse_point_wkt(missing)
    assert math.isnan(lon) and math.isnan(lat)


@pytest.mark.parametrize(
    "malformed",
    ["POINT (37.6)", "POINT (a b)", "LINESTRING (1 2, 3 4)", "37.6 55.7", "POINT (200 55)", 42],
)
def test_malformed_geom_raises(malformed):
    with pytest.raises(ValueError):
        parse_point_wkt(malformed)


def test_haversine_identical_points_is_zero():
    assert haversine_m(37.6, 55.75, 37.6, 55.75) == pytest.approx(0.0, abs=1e-9)


def test_haversine_known_distance_and_nan():
    one_degree_lat = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert one_degree_lat == pytest.approx(111_195.08, rel=1e-5)
    result = haversine_m([0.0, np.nan], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    assert result[0] == pytest.approx(one_degree_lat)
    assert math.isnan(result[1])
