"""Тесты безопасного official adapter на крошечных синтетических CSV (не organizer data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mostransport_ml.data.official import (
    DATASET_ENV_VAR,
    SCHEDULE_PLAN_COLUMNS,
    OfficialDataSchemaError,
    load_official_split,
    load_schedule_plan,
    load_shared_real_vehicle_ids,
    resolve_dataset_root,
)
from mostransport_ml.features.schema import FORBIDDEN_INPUT_COLUMNS

LABELS = pd.DataFrame(
    {
        "sample_id": ["s1", "s2"],
        "tr_id": [7, 8],
        "T": ["2026-01-06 02:10:00", "2026-01-06 02:15:00"],
        "target_stop_id": [501, 502],
        "target_time_begin": ["2026-01-06 02:22:00.000000000", "2026-01-06 02:26:00.000000000"],
        "cur_dev_s": [0.0, 60.0],
        "target_delay_s": [-25.0, 130.0],
        "target_class": ["ontime", "late"],
    }
)
TRAFFIC = pd.DataFrame(
    {
        "packet_id": [-1, -2],
        "tr_id": [7, 8],
        "unit_id": [1, 2],
        "event_time": ["2026-01-06 02:09:31.462764", "2026-01-06 02:14:00.000001"],
        "device_event_id": [0, 0],
        "location_valid": [True, False],
        "gps_time": ["2026-01-06 02:09:31", None],
        "lon": [37.6, None],
        "lat": [55.7, None],
        "alt": [150.0, None],
        "speed": [20.0, None],
        "heading": [90.0, None],
        "receive_time": ["2026-01-06 02:09:31.5", "2026-01-06 02:14:00.1"],
        "is_hist_data": [False, False],
    }
)
SCHEDULE_WITH_FACT = pd.DataFrame(
    {
        "tt_action_item_id": [501, 502],
        "time_begin": ["2026-01-06 02:22:00", "2026-01-06 02:26:00"],
        "time_fact_begin": ["2026-01-06 02:21:35", "2026-01-06 02:28:10"],
        "order_date": ["2026-01-06", "2026-01-06"],
        "manual_fill": [True, False],
        "tr_id": [7, 8],
        "geom": ["POINT (37.61 55.71)", "POINT (37.62 55.72)"],
        "building_address": ["ул. Тестовая, д.1", "ул. Тестовая, д.2"],
    }
)
# Официальный test schedule (с фактом): его плановые `tr_id` = {7, 9} намеренно
# отличаются от train `tr_id` = {7, 8}, чтобы источник real-набора был различим.
TEST_SCHEDULE_WITH_FACT = pd.DataFrame(
    {
        "tt_action_item_id": [601, 602, 603],
        "time_begin": ["2026-01-06 06:36:00", "2026-01-06 06:41:00", "2026-01-06 06:41:00"],
        "time_fact_begin": ["2026-01-06 06:37:00", "2026-01-06 06:40:00", "2026-01-06 06:45:00"],
        "order_date": ["2026-01-06"] * 3,
        "manual_fill": [True] * 3,
        "tr_id": [7, 7, 9],
        "geom": ["POINT (37.4 55.8)", "POINT (37.5 55.8)", "POINT (37.5 55.8)"],
        "building_address": ["a", "b", "c"],
    }
)
PLAN_ONLY = pd.DataFrame(
    {
        "tt_action_item_id": [601, 602, 603],
        "time_begin": ["2026-01-06 06:36:00", "2026-01-06 06:41:00", "2026-01-06 06:41:00"],
        "order_date": ["2026-01-06"] * 3,
        "manual_fill": [True] * 3,
        "tr_id": [7, 7, 9],
        "geom": ["POINT (37.4 55.8)", "POINT (37.5 55.8)", "POINT (37.5 55.8)"],
        "building_address": ["a", "b", "c"],
    }
)


def write_dataset(
    root: Path,
    schedule: pd.DataFrame = SCHEDULE_WITH_FACT,
    test_schedule: pd.DataFrame = TEST_SCHEDULE_WITH_FACT,
) -> Path:
    """Крошечный train+test датасет; файлов validate намеренно нет."""
    files = {
        "labels/labels_train.csv": LABELS,
        "train/traffic.csv": TRAFFIC,
        "train/schedule.csv": schedule,
        "labels/labels_test.csv": LABELS,
        "test/traffic.csv": TRAFFIC,
        "test/schedule.csv": test_schedule,
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        content.to_csv(path, index=False)
    return root


def test_split_separates_target_and_never_loads_factual_schedule(tmp_path):
    split = load_official_split(write_dataset(tmp_path), "train")

    assert "time_fact_begin" not in split.schedule_plan.columns
    assert tuple(split.schedule_plan.columns) == SCHEDULE_PLAN_COLUMNS
    for frame in (split.points, split.telemetry, split.schedule_plan):
        assert not FORBIDDEN_INPUT_COLUMNS & set(frame.columns)

    assert split.points["sample_id"].tolist() == ["s1", "s2"]
    assert split.target.to_dict() == {"s1": -25.0, "s2": 130.0}
    assert pd.api.types.is_datetime64_dtype(split.points["T"])
    assert pd.api.types.is_datetime64_dtype(split.telemetry["event_time"])
    for banned in ("packet_id", "unit_id", "device_event_id"):
        assert banned not in split.telemetry.columns


def test_missing_required_column_gives_structured_error(tmp_path):
    broken = SCHEDULE_WITH_FACT.rename(columns={"geom": "geometry"})
    root = write_dataset(tmp_path, schedule=broken)
    with pytest.raises(OfficialDataSchemaError) as excinfo:
        load_official_split(root, "train")
    assert excinfo.value.missing == ["geom"]
    assert excinfo.value.path.name == "schedule.csv"


def test_schedule_plan_loader_accepts_file_without_fact(tmp_path):
    path = tmp_path / "plan_only.csv"
    PLAN_ONLY.to_csv(path, index=False)
    plan = load_schedule_plan(path)
    assert tuple(plan.columns) == SCHEDULE_PLAN_COLUMNS


def test_validate_split_has_no_modeling_loader(tmp_path):
    with pytest.raises(ValueError, match="Unsupported split"):
        load_official_split(write_dataset(tmp_path), "validate")  # type: ignore[arg-type]


def test_shared_real_vehicle_ids_come_from_test_planned_schedule(tmp_path):
    root = write_dataset(tmp_path)
    assert not (root / "validate").exists()
    # {7, 9} — плановые tr_id test schedule, а не train {7, 8}.
    assert load_shared_real_vehicle_ids(root) == frozenset({7, 9})


def test_shared_real_vehicle_ids_ignore_factual_schedule_column(tmp_path):
    without_fact = TEST_SCHEDULE_WITH_FACT.drop(columns=["time_fact_begin"])
    root = write_dataset(tmp_path, test_schedule=without_fact)
    assert load_shared_real_vehicle_ids(root) == frozenset({7, 9})


def test_resolve_dataset_root(tmp_path, monkeypatch):
    monkeypatch.delenv(DATASET_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match=DATASET_ENV_VAR):
        resolve_dataset_root()
    monkeypatch.setenv(DATASET_ENV_VAR, str(tmp_path))
    assert resolve_dataset_root() == tmp_path
    assert resolve_dataset_root(tmp_path / ".") == tmp_path / "."
    with pytest.raises(FileNotFoundError):
        resolve_dataset_root(tmp_path / "missing")
