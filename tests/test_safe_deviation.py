"""P1: offline point-in-time-safe current deviation и его явная подача в offline_context.

Все данные синтетические. Эталон семантики — research safe proxy: последнее
подтверждённое событие того же `tr_id` с `time_fact_begin <= T`; ничья по
факту → максимальный числовой `tt_action_item_id` → последняя строка CSV;
нет событий → 0.0.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.data.safe_deviation import (
    SAFE_DEVIATION_NAME,
    SafeDeviationInputError,
    safe_current_deviation_seconds,
)
from mostransport_ml.features.adapters import offline_context
from mostransport_ml.features.builder import ForbiddenInputColumnError, build_features
from mostransport_ml.features.context import build_features_from_context
from mostransport_ml.features.schema import FEATURE_NAMES

T = pd.Timestamp("2026-01-06 10:00:00")


def at(seconds: float) -> pd.Timestamp:
    return T + pd.Timedelta(seconds=seconds)


def facts_frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (tr_id, tt_action_item_id, time_begin, time_fact_begin | None)."""
    return pd.DataFrame(
        {
            "tr_id": [r[0] for r in rows],
            "tt_action_item_id": pd.Series([r[1] for r in rows], dtype=object),
            "time_begin": pd.to_datetime([r[2] for r in rows]),
            "time_fact_begin": pd.to_datetime([r[3] for r in rows]),
        }
    )


def points_frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (sample_id, tr_id, T)."""
    return pd.DataFrame(
        {
            "sample_id": [r[0] for r in rows],
            "tr_id": [r[1] for r in rows],
            "T": pd.to_datetime([r[2] for r in rows]),
        }
    )


def single(facts_rows: list[tuple], t: pd.Timestamp = T) -> float:
    result = safe_current_deviation_seconds(points_frame([("p", 1, t)]), facts_frame(facts_rows))
    return float(result.loc["p"])


def reference_safe_deviation(points: pd.DataFrame, facts: pd.DataFrame) -> tuple[pd.Series, int]:
    """Независимый brute-force эталон в форме research-кода (sort + last).

    Возвращает значения и число точек, где последний факт был ничьей.
    """
    facts = facts.assign(
        source_row=np.arange(len(facts)), numeric_id=facts["tt_action_item_id"].map(int)
    )
    values, tied_points = {}, 0
    for p in points.itertuples(index=False):
        eligible = facts[(facts.tr_id == p.tr_id) & (facts.time_fact_begin <= p.T)]
        if eligible.empty:
            values[p.sample_id] = 0.0
            continue
        tied_points += int((eligible.time_fact_begin == eligible.time_fact_begin.max()).sum() > 1)
        chosen = eligible.sort_values(
            ["time_fact_begin", "numeric_id", "source_row"], kind="stable"
        ).iloc[-1]
        values[p.sample_id] = (chosen.time_fact_begin - chosen.time_begin).total_seconds()
    return pd.Series(values, name=SAFE_DEVIATION_NAME, dtype=float), tied_points


# ------------------------------------------------------------------ semantics matrix


def test_no_confirmed_event_at_or_before_T_gives_zero():
    rows = [
        (1, 10, at(-600), None),  # не подтверждено
        (1, 11, at(-300), at(1)),  # подтверждено только после T
    ]
    assert single(rows) == 0.0


def test_one_eligible_event_gives_exact_delay():
    assert single([(1, 10, at(-600), at(-510.25))]) == 89.75


def test_latest_fact_wins_not_latest_plan_or_last_row():
    rows = [
        (1, 12, at(-300), at(-120)),  # поздний план, но факт раньше: 180
        (1, 10, at(-900), at(-60)),  # последний факт <= T: 840
        (1, 11, at(-600), at(-400)),  # последняя строка: 200
    ]
    assert single(rows) == 840.0


def test_event_exactly_at_T_is_eligible():
    assert single([(1, 10, at(-600), at(-300)), (1, 11, at(-60), at(0))]) == 60.0


def test_future_fact_with_huge_delay_is_ignored():
    rows = [
        (1, 10, at(-600), at(-300)),
        (1, 11, at(-60), at(0.001)),  # на 1 мс позже T
        (1, 12, at(-3000), at(5000)),  # огромная задержка в будущем
    ]
    assert single(rows) == 300.0


def test_latest_fact_tie_uses_numeric_action_id_not_lexicographic():
    rows = [
        (1, "10", at(-500), at(-60)),  # 440 — численно больший ID
        (1, "9", at(-100), at(-60)),  # 40 — лексикографически больший
    ]
    assert "9" > "10"
    assert single(rows) == 440.0
    assert single(rows[::-1]) == 440.0


def test_latest_fact_tie_with_integer_ids():
    assert single([(1, 100, at(-500), at(-60)), (1, 99, at(-100), at(-60))]) == 440.0


def test_same_fact_and_numeric_id_uses_later_source_row():
    rows = [
        (1, 7, at(-500), at(-60)),
        (1, "007", at(-200), at(-60)),  # тот же числовой ID, более поздняя строка
    ]
    assert single(rows) == 140.0
    assert single(rows[::-1]) == 440.0


def test_negative_deviation_is_preserved():
    assert single([(1, 10, at(-60), at(-105.5))]) == -45.5


def test_positive_deviation_is_preserved_without_clamp():
    assert single([(1, 10, at(-3600), at(-2999.75))]) == 600.25


def test_future_rows_never_participate_even_in_tie_break():
    base = [(1, 10, at(-600), at(-300))]
    future_ties = [(1, "not-a-number", at(-900), at(30)), (1, "x", at(-50), at(30))]
    assert single(base + future_ties) == single(base) == 300.0


def test_tie_break_needing_unparseable_action_id_fails_loudly():
    rows = [(1, "abc", at(-500), at(-60)), (1, "10", at(-100), at(-60))]
    with pytest.raises(SafeDeviationInputError, match="tt_action_item_id"):
        single(rows)


def test_unparseable_action_id_is_fine_when_tie_break_is_not_needed():
    assert single([(1, "abc", at(-500), at(-60)), (1, "10", at(-100), at(-90))]) == 440.0


@pytest.mark.parametrize("bad_id", [None, 1.5, True, "1e3", ""])
def test_non_integer_action_ids_are_rejected_in_a_tie(bad_id):
    rows = [(1, bad_id, at(-500), at(-60)), (1, 3, at(-100), at(-60))]
    with pytest.raises(SafeDeviationInputError, match="tt_action_item_id"):
        single(rows)


def test_integral_float_action_ids_are_numeric():
    assert single([(1, 10.0, at(-500), at(-60)), (1, 9, at(-100), at(-60))]) == 440.0


# ------------------------------------------------------------------ alignment and output


MULTI_FACTS = [
    (1, 10, at(-900), at(-800)),  # 100
    (1, 11, at(-400), at(-100)),  # 300
    (2, 20, at(-700), at(-730)),  # -30
    (3, 30, at(-60), at(120)),  # только будущее
]
MULTI_POINTS = [
    ("a", 1, at(0)),
    ("b", 1, at(-500)),
    ("c", 2, at(0)),
    ("d", 3, at(0)),
    ("e", 1, at(-900)),
]
MULTI_EXPECTED = {"a": 300.0, "b": 100.0, "c": -30.0, "d": 0.0, "e": 0.0}


def test_multiple_points_are_aligned_by_sample_id():
    result = safe_current_deviation_seconds(points_frame(MULTI_POINTS), facts_frame(MULTI_FACTS))
    assert result.name == SAFE_DEVIATION_NAME
    assert result.index.name == "sample_id"
    assert result.dtype == np.float64
    assert list(result.index) == ["a", "b", "c", "d", "e"]
    assert result.to_dict() == MULTI_EXPECTED


def test_shuffled_points_give_same_value_per_sample_id():
    points = points_frame(MULTI_POINTS)
    shuffled = points.sample(frac=1.0, random_state=3)
    assert list(shuffled["sample_id"]) != list(points["sample_id"])
    result = safe_current_deviation_seconds(shuffled, facts_frame(MULTI_FACTS))
    assert list(result.index) == list(shuffled["sample_id"])
    assert result.to_dict() == MULTI_EXPECTED


def test_facts_filtered_by_vehicle_keep_the_same_result():
    facts = facts_frame(MULTI_FACTS)
    points = points_frame(MULTI_POINTS)
    full = safe_current_deviation_seconds(points, facts)
    vehicle_1 = points[points.tr_id == 1]
    filtered = safe_current_deviation_seconds(vehicle_1, facts[facts.tr_id == 1])
    pd.testing.assert_series_equal(filtered, full.loc[list(vehicle_1["sample_id"])])


def test_matches_research_reference_on_randomized_ties():
    rng = np.random.default_rng(42)
    fact_rows, point_rows = [], []
    for vehicle in (101, 102, 103, 104):
        for _ in range(60):
            plan = at(60 * int(rng.integers(-120, 0)))  # минутная сетка
            fact = plan + pd.Timedelta(seconds=30 * int(rng.integers(-10, 30)))  # сетка → ничьи
            confirmed = rng.random() > 0.1
            action = int(rng.integers(1, 12))  # малый диапазон → ничьи по ID
            fact_rows.append((vehicle, action, plan, fact if confirmed else None))
        for k in range(40):
            point_rows.append((f"{vehicle}-{k}", vehicle, at(30 * int(rng.integers(-240, 60)))))
    facts, points = facts_frame(fact_rows), points_frame(point_rows)
    expected, tied_points = reference_safe_deviation(points, facts)
    result = safe_current_deviation_seconds(points, facts)
    pd.testing.assert_series_equal(result, expected, check_names=False, check_index_type=False)
    assert (result != 0.0).sum() > 50  # фикстура действительно выбирает события
    assert tied_points >= 10  # и действительно проверяет tie-break

    # Строки с фактом позже всех T (экстремальные задержки, мусорные ID) ничего не меняют.
    late = at(30 * 60 + 1)
    future = facts_frame(
        [(v, "garbage", late - pd.Timedelta(hours=5), late) for v in (101, 102, 103, 104)] * 2
    )
    with_future = pd.concat([facts, future], ignore_index=True)
    pd.testing.assert_series_equal(safe_current_deviation_seconds(points, with_future), result)


# ------------------------------------------------------------------ input validation


def test_duplicate_prediction_sample_id_fails():
    points = points_frame([("p", 1, T), ("p", 1, at(-60))])
    with pytest.raises(SafeDeviationInputError, match="unique"):
        safe_current_deviation_seconds(points, facts_frame(MULTI_FACTS))


@pytest.mark.parametrize("column", ["sample_id", "tr_id", "T"])
def test_missing_point_column_fails(column):
    points = points_frame([("p", 1, T)]).drop(columns=column)
    with pytest.raises(SafeDeviationInputError, match=column):
        safe_current_deviation_seconds(points, facts_frame(MULTI_FACTS))


@pytest.mark.parametrize("column", ["tr_id", "tt_action_item_id", "time_begin", "time_fact_begin"])
def test_missing_fact_column_fails(column):
    facts = facts_frame(MULTI_FACTS).drop(columns=column)
    with pytest.raises(SafeDeviationInputError, match=column):
        safe_current_deviation_seconds(points_frame([("p", 1, T)]), facts)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("sample_id", None, "sample_id"),
        ("tr_id", None, "tr_id"),
        ("tr_id", 1.0, "tr_id"),
        ("T", pd.NaT, "missing"),
        ("T", "not a time", "unparseable"),
    ],
)
def test_bad_point_values_fail(column, value, match):
    points = points_frame([("p", 1, T), ("q", 1, T)]).astype({column: object})
    points.loc[1, column] = value
    with pytest.raises(SafeDeviationInputError, match=match):
        safe_current_deviation_seconds(points, facts_frame(MULTI_FACTS))


def test_timezone_aware_T_is_rejected():
    points = points_frame([("p", 1, T)])
    points["T"] = points["T"].dt.tz_localize("UTC")
    with pytest.raises(SafeDeviationInputError, match="naive"):
        safe_current_deviation_seconds(points, facts_frame(MULTI_FACTS))


def test_numeric_timestamps_are_rejected_instead_of_guessing_units():
    points = points_frame([("p", 1, T)])
    points["T"] = points["T"].astype("int64")
    with pytest.raises(SafeDeviationInputError, match="not numbers"):
        safe_current_deviation_seconds(points, facts_frame(MULTI_FACTS))


def test_ambiguous_non_iso_timestamps_are_rejected_instead_of_guessing_day_first():
    points = points_frame([("p", 1, T)]).astype({"T": object})
    points.loc[0, "T"] = "01/06/2026 10:00:00"  # 6 января или 1 июня?
    with pytest.raises(SafeDeviationInputError, match="unparseable"):
        safe_current_deviation_seconds(points, facts_frame(MULTI_FACTS))


def test_iso_strings_with_mixed_precision_are_accepted():
    facts = facts_frame(MULTI_FACTS).astype({"time_begin": object, "time_fact_begin": object})
    facts["time_begin"] = facts["time_begin"].map(lambda ts: ts.isoformat(sep=" "))
    facts.loc[0, "time_fact_begin"] = "2026-01-06T09:46:40"
    facts.loc[1, "time_fact_begin"] = "2026-01-06 09:58:20.000000000"
    result = safe_current_deviation_seconds(points_frame(MULTI_POINTS), facts)
    assert result.to_dict() == MULTI_EXPECTED


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("time_begin", None, "missing"),
        ("time_fact_begin", "31/31/2026 xx", "unparseable"),
        ("tr_id", None, "tr_id"),
    ],
)
def test_bad_fact_values_fail(column, value, match):
    facts = facts_frame(MULTI_FACTS).astype({column: object})
    facts.loc[1, column] = value
    with pytest.raises(SafeDeviationInputError, match=match):
        safe_current_deviation_seconds(points_frame([("p", 1, T)]), facts)


def test_string_timestamps_are_parsed_consistently():
    facts = facts_frame(MULTI_FACTS)
    as_text = facts.assign(
        time_begin=facts["time_begin"].dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
        time_fact_begin=facts["time_fact_begin"].dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
    )
    points = points_frame(MULTI_POINTS)
    pd.testing.assert_series_equal(
        safe_current_deviation_seconds(points, as_text),
        safe_current_deviation_seconds(points, facts),
    )


def test_reordered_fact_rows_are_rejected():
    facts = facts_frame(MULTI_FACTS).sort_values("time_fact_begin")
    with pytest.raises(SafeDeviationInputError, match="source"):
        safe_current_deviation_seconds(points_frame([("p", 1, T)]), facts)


def test_point_vehicle_without_any_schedule_rows_fails():
    with pytest.raises(SafeDeviationInputError, match="no rows"):
        safe_current_deviation_seconds(points_frame([("p", 999, T)]), facts_frame(MULTI_FACTS))


def test_integer_and_string_vehicle_ids_share_the_canonical_key():
    facts = facts_frame(MULTI_FACTS).astype({"tr_id": str})
    result = safe_current_deviation_seconds(points_frame(MULTI_POINTS), facts)
    assert result.to_dict() == MULTI_EXPECTED


def test_inputs_are_not_mutated():
    points, facts = points_frame(MULTI_POINTS), facts_frame(MULTI_FACTS)
    before = points.copy(), facts.copy()
    safe_current_deviation_seconds(points, facts)
    pd.testing.assert_frame_equal(points, before[0])
    pd.testing.assert_frame_equal(facts, before[1])


# ------------------------------------------------------------------ offline_context override


def override_frames():
    """Три offline-точки двух ТС: points (+official cur_dev_s), telemetry, plan."""
    points = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "tr_id": [1, 1, 2],
            "T": pd.to_datetime([T, at(300), T]),
            "target_stop_id": [11, 12, 21],
            "target_time_begin": pd.to_datetime([at(720), at(1020), at(840)]),
            "cur_dev_s": [5.0, 6.0, 7.0],  # official-значения, которые override заменяет
        }
    )
    offsets = range(-900, 301, 30)
    telemetry = pd.DataFrame(
        {
            "tr_id": [v for v in (1, 2) for _ in offsets],
            "event_time": pd.to_datetime([at(s) for _ in (1, 2) for s in offsets]),
            "location_valid": pd.array([s % 90 != 0 for _ in (1, 2) for s in offsets]),
            "lon": [37.6 + s * 1e-6 for _ in (1, 2) for s in offsets],
            "lat": [55.7 + s * 1e-6 for _ in (1, 2) for s in offsets],
            "speed": [float(abs(s) % 40) for _ in (1, 2) for s in offsets],
        }
    )
    plan = pd.DataFrame(
        {
            "tt_action_item_id": [11, 12, 21],
            "tr_id": [1, 1, 2],
            "geom": ["POINT (37.61 55.71)", "POINT (37.62 55.72)", "POINT (37.63 55.73)"],
            "manual_fill": [True, False, True],
        }
    )
    return points, telemetry, plan


SAFE = pd.Series({"s1": -12.5, "s2": 0.0, "s3": 431.0})


def test_default_offline_context_keeps_official_cur_dev_s():
    frames = override_frames()
    x = build_features_from_context(offline_context(*frames))
    pd.testing.assert_frame_equal(x, build_features(*frames), check_exact=True)
    assert x["cur_dev_s"].to_dict() == {"s1": 5.0, "s2": 6.0, "s3": 7.0}


def test_override_replaces_only_cur_dev_s():
    frames = override_frames()
    legacy = build_features_from_context(offline_context(*frames))
    safe = build_features_from_context(offline_context(*frames, current_deviation_seconds=SAFE))
    assert tuple(safe.columns) == FEATURE_NAMES
    assert safe["cur_dev_s"].to_dict() == SAFE.to_dict()
    pd.testing.assert_frame_equal(
        safe.drop(columns="cur_dev_s"), legacy.drop(columns="cur_dev_s"), check_exact=True
    )


def test_override_reaches_canonical_point_current_deviation():
    batch = offline_context(*override_frames(), current_deviation_seconds=SAFE)
    assert {p.point_id: p.current_deviation_s for p in batch.points} == SAFE.to_dict()


def test_override_alignment_is_by_sample_id_not_position():
    frames = override_frames()
    reversed_series = SAFE.iloc[::-1]
    as_mapping = {"s3": 431.0, "s1": -12.5, "s2": 0.0}
    expected = build_features_from_context(offline_context(*frames, current_deviation_seconds=SAFE))
    for override in (reversed_series, as_mapping):
        x = build_features_from_context(
            offline_context(*frames, current_deviation_seconds=override)
        )
        pd.testing.assert_frame_equal(x, expected, check_exact=True)


def test_override_from_safe_helper_on_shuffled_points():
    points, telemetry, plan = override_frames()
    facts = facts_frame([(1, 11, at(-400), at(-100)), (2, 21, at(-200), at(-260))])
    shuffled = points.iloc[[2, 0, 1]]
    safe = safe_current_deviation_seconds(shuffled, facts)
    x = build_features_from_context(
        offline_context(shuffled, telemetry, plan, current_deviation_seconds=safe)
    )
    assert list(x.index) == ["s3", "s1", "s2"]
    assert x["cur_dev_s"].to_dict() == {"s3": -60.0, "s1": 300.0, "s2": 300.0}


def test_override_makes_official_cur_dev_s_column_optional():
    points, telemetry, plan = override_frames()
    without = points.drop(columns="cur_dev_s")
    x = build_features_from_context(
        offline_context(without, telemetry, plan, current_deviation_seconds=SAFE)
    )
    assert x["cur_dev_s"].to_dict() == SAFE.to_dict()
    with pytest.raises(ValueError, match="cur_dev_s"):
        offline_context(without, telemetry, plan)


def test_override_duplicate_prediction_sample_id_fails():
    points, telemetry, plan = override_frames()
    points.loc[2, "sample_id"] = "s1"
    with pytest.raises(ValueError, match="unique"):
        offline_context(points, telemetry, plan, current_deviation_seconds=SAFE)


def test_override_duplicate_keys_fail():
    duplicated = pd.Series([1.0, 2.0, 3.0, 4.0], index=["s1", "s2", "s3", "s1"])
    with pytest.raises(ValueError, match="duplicated"):
        offline_context(*override_frames(), current_deviation_seconds=duplicated)


def test_override_missing_sample_id_fails():
    with pytest.raises(ValueError, match="missing 1 sample_id"):
        offline_context(*override_frames(), current_deviation_seconds=SAFE.drop("s2"))


def test_override_extra_sample_id_fails():
    extra = pd.concat([SAFE, pd.Series({"s4": 1.0})])
    with pytest.raises(ValueError, match="not prediction points"):
        offline_context(*override_frames(), current_deviation_seconds=extra)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, "12.5", True])
def test_override_non_finite_or_non_numeric_value_fails(bad):
    override = pd.Series({"s1": -12.5, "s2": bad, "s3": 431.0}, dtype=object)
    with pytest.raises(ValueError, match="finite numbers"):
        offline_context(*override_frames(), current_deviation_seconds=override)


def test_override_must_be_keyed_container():
    with pytest.raises(TypeError, match="keyed by sample_id"):
        offline_context(*override_frames(), current_deviation_seconds=[-12.5, 0.0, 431.0])


def test_override_does_not_mutate_inputs():
    frames = override_frames()
    before = [f.copy() for f in frames]
    safe = SAFE.copy()
    offline_context(*frames, current_deviation_seconds=safe)
    for frame, copy in zip(frames, before, strict=True):
        pd.testing.assert_frame_equal(frame, copy)
    pd.testing.assert_series_equal(safe, SAFE)


def test_factual_schedule_cannot_enter_the_feature_builder():
    points, telemetry, _ = override_frames()
    facts = facts_frame([(1, 11, at(-400), at(-100))]).assign(geom=None, manual_fill=True)
    with pytest.raises(ForbiddenInputColumnError, match="time_fact_begin"):
        offline_context(points, telemetry, facts, current_deviation_seconds=SAFE)
