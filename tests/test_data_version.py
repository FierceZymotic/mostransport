"""Регрессия R2: provenance-fingerprint schedule не зависит от factual-колонок."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from test_official_loader import TEST_SCHEDULE_WITH_FACT, write_dataset

from mostransport_ml.data.official import (
    FACTUAL_SCHEDULE_COLUMNS,
    SCHEDULE_PLAN_COLUMNS,
    planned_schedule_fingerprint,
)

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_offline_baseline.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_offline_baseline", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(tmp_path: Path, frame: pd.DataFrame, name: str = "schedule.csv") -> str:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return planned_schedule_fingerprint(path)


def with_changed_fact(frame: pd.DataFrame) -> pd.DataFrame:
    changed = frame.copy()
    changed["time_fact_begin"] = ["2026-01-06 09:00:00"] * len(changed)
    return changed


def test_changed_time_fact_begin_keeps_fingerprint(tmp_path):
    original = fingerprint(tmp_path, TEST_SCHEDULE_WITH_FACT, "a.csv")
    changed = fingerprint(tmp_path, with_changed_fact(TEST_SCHEDULE_WITH_FACT), "b.csv")
    assert original == changed


def test_fingerprint_cannot_ingest_factual_columns(tmp_path):
    # Файл с factual-колонкой и файл, где её нет вовсе, дают один и тот же fingerprint:
    # значит factual-колонка не входит в сериализуемое представление.
    with_fact = fingerprint(tmp_path, TEST_SCHEDULE_WITH_FACT, "a.csv")
    no_fact = TEST_SCHEDULE_WITH_FACT.drop(columns=sorted(FACTUAL_SCHEDULE_COLUMNS))
    assert fingerprint(tmp_path, no_fact, "b.csv") == with_fact
    assert not FACTUAL_SCHEDULE_COLUMNS & set(SCHEDULE_PLAN_COLUMNS)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("tt_action_item_id", 999),
        ("time_begin", "2026-01-06 06:37:00"),
        ("order_date", "2026-01-07"),
        ("manual_fill", False),
        ("tr_id", 8),
        ("geom", "POINT (37.41 55.8)"),
        ("building_address", "z"),
    ],
)
def test_changed_planned_value_changes_fingerprint(tmp_path, column, value):
    assert column in SCHEDULE_PLAN_COLUMNS
    changed = TEST_SCHEDULE_WITH_FACT.copy()
    changed.loc[0, column] = value
    assert fingerprint(tmp_path, TEST_SCHEDULE_WITH_FACT, "a.csv") != fingerprint(
        tmp_path, changed, "b.csv"
    )


def test_fingerprint_is_deterministic(tmp_path):
    path = tmp_path / "schedule.csv"
    TEST_SCHEDULE_WITH_FACT.to_csv(path, index=False)
    assert planned_schedule_fingerprint(path) == planned_schedule_fingerprint(path)


def test_runner_data_version_ignores_factual_schedule_bytes(tmp_path):
    runner = load_runner()
    base = runner.dataset_version(write_dataset(tmp_path / "base"))
    fact_changed = runner.dataset_version(
        write_dataset(tmp_path / "fact", test_schedule=with_changed_fact(TEST_SCHEDULE_WITH_FACT))
    )
    planned = TEST_SCHEDULE_WITH_FACT.copy()
    planned.loc[0, "time_begin"] = "2026-01-06 06:37:00"
    planned_changed = runner.dataset_version(
        write_dataset(tmp_path / "plan", test_schedule=planned)
    )

    assert base == fact_changed
    assert base != planned_changed
    assert base.startswith("official-planned-sha256:")
    assert not (tmp_path / "base" / "validate").exists()
