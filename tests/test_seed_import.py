"""DB seed importers (scripts/import_db_seed.py).

psycopg is a Backend-side dependency and is not installed in the ML environment, so the
module is loaded with a stub and a recording connection; only the SQL parameters are checked.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from datetime import timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class _Cursor:
    def __init__(self, log):
        self.log, self.rowcount = log, 1

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Conn:
    def __init__(self):
        self.log, self.commits = [], 0

    def cursor(self):
        return _Cursor(self.log)

    def commit(self):
        self.commits += 1


@pytest.fixture()
def seed(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace(Connection=object))
    spec = importlib.util.spec_from_file_location("import_db_seed_under_test", ROOT / "scripts" / "import_db_seed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vehicle_identity_is_one_to_one_and_excludes_synthetic_trips(seed, tmp_path):
    csv = tmp_path / "traffic.csv"
    csv.write_text("tr_id,unit_id,speed\n134040,1105498,1\n134040,1105498,2\n9000003,777,0\n122048,26,5\n")
    conn = _Conn()
    assert seed.import_vehicle_identity(conn, csv) == 2
    params = sorted(p for _, p in conn.log)
    assert params == [("1105498", "134040"), ("26", "122048")]
    assert all("ON CONFLICT (unit_id)" in sql for sql, _ in conn.log)
    assert conn.commits == 1


def test_vehicle_identity_refuses_ambiguous_mapping(seed, tmp_path):
    csv = tmp_path / "traffic.csv"
    csv.write_text("tr_id,unit_id\n1,10\n2,10\n")
    conn = _Conn()
    with pytest.raises(SystemExit):
        seed.import_vehicle_identity(conn, csv)
    assert conn.log == []


def test_replay_facts_skip_missing_and_are_explicit_utc(seed, tmp_path):
    csv = tmp_path / "train" / "schedule.csv"
    csv.parent.mkdir()
    csv.write_text("tt_action_item_id,tr_id,time_fact_begin\n10,1,2026-01-06 03:00:05\n11,1,\n")
    conn = _Conn()
    assert seed.import_replay_facts(conn, csv) == 1
    (sql, (when, action_id)), = conn.log
    assert sql.startswith("UPDATE schedule_actions SET time_fact_begin")
    assert action_id == "10"
    assert when.tzinfo is not None and when.utcoffset() == timezone.utc.utcoffset(None)
    assert when.isoformat() == "2026-01-06T03:00:05+00:00"
    # idempotent: the same statements again
    again = _Conn()
    seed.import_replay_facts(again, csv)
    assert again.log == conn.log


@pytest.mark.parametrize("split", ["test", "validate"])
def test_replay_facts_refuse_test_and_validate_splits(seed, tmp_path, split):
    csv = tmp_path / split / "schedule.csv"
    csv.parent.mkdir()
    csv.write_text("tt_action_item_id,tr_id,time_fact_begin\n10,1,2026-01-06 03:00:05\n")
    conn = _Conn()
    with pytest.raises(SystemExit):
        seed.import_replay_facts(conn, csv)
    assert conn.log == []


def test_plan_and_telemetry_times_are_bound_as_aware_utc(seed):
    # The same instants as the frozen contract (naive organizer time == UTC), independent of
    # the database session TimeZone.
    conn = _Conn()
    assert seed.import_schedule_actions(conn) > 0
    plan_times = [params[2] for _, params in conn.log]
    assert all(t.utcoffset() == timezone.utc.utcoffset(None) for t in plan_times)
    assert plan_times[0].isoformat() == "2026-01-06T06:36:00+00:00"
    conn = _Conn()
    assert seed.import_telemetry(conn) == 115
    tele_times = [params[2] for _, params in conn.log]
    assert all(t.utcoffset() == timezone.utc.utcoffset(None) for t in tele_times)
    assert tele_times[0].isoformat() == "2026-01-06T02:47:50+00:00"
