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
