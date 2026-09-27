"""Shared Backend <-> ML conformance cases (the same JSON drives backend/test/conformance-fixtures.spec.ts)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mostransport_ml.data.safe_deviation import safe_current_deviation_seconds

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "backend_ml_conformance.json").read_text())


def _naive(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)


@pytest.mark.parametrize("case", FIXTURE["current_deviation"], ids=lambda c: c["name"])
def test_p1_safe_deviation_matches_shared_fixture(case):
    facts = pd.DataFrame(
        {
            "tr_id": ["X"] * len(case["rows"]),
            "tt_action_item_id": [r["id"] for r in case["rows"]],
            "time_begin": [_naive(r["plan"]) for r in case["rows"]],
            "time_fact_begin": [None if r["fact"] is None else _naive(r["fact"]) for r in case["rows"]],
        }
    )
    facts["time_fact_begin"] = pd.to_datetime(facts["time_fact_begin"])
    points = pd.DataFrame({"sample_id": ["p"], "tr_id": ["X"], "T": [_naive(case["T"])]})
    assert safe_current_deviation_seconds(points, facts).iloc[0] == case["expected_seconds"]


def classify_target_actions(rows: list[dict], prediction_time: pd.Timestamp) -> tuple[str, str | None]:
    """Python mirror of backend classifyTargetActions (docs §5.1): used by the parity harness."""
    lo, hi = prediction_time + pd.Timedelta(minutes=10), prediction_time + pd.Timedelta(minutes=15)
    horizon = [r for r in rows if lo < pd.Timestamp(r["plan"]) <= hi]
    if not horizon:
        return "none", None
    earliest = min(pd.Timestamp(r["plan"]) for r in horizon)
    at = sorted((r for r in horizon if pd.Timestamp(r["plan"]) == earliest), key=lambda r: int(r["id"]))
    if len({(r["lat"], r["lon"], r["manual_fill"]) for r in at}) > 1:
        return "ambiguous", None
    return "ok", at[0]["id"]


@pytest.mark.parametrize("case", FIXTURE["target_selection"], ids=lambda c: c["name"])
def test_target_selection_mirror_matches_shared_fixture(case):
    status, action_id = classify_target_actions(case["rows"], pd.Timestamp(case["T"]))
    assert (status, action_id) == (case["expected_status"], case["expected_id"])
