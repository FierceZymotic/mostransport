"""Безопасный validate inference и запись submission (синтетический official-like датасет)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.data import official
from mostransport_ml.data.official import (
    OfficialDataSchemaError,
    load_sample_submission_ids,
    load_validate_inputs,
)
from mostransport_ml.features.adapters import offline_context
from mostransport_ml.features.context import build_features_from_context
from mostransport_ml.features.schema import FEATURE_NAMES
from mostransport_ml.inference.predictor import ArtifactPredictor
from mostransport_ml.inference.submission import (
    SubmissionError,
    build_submission,
    generate_validate_submission,
    write_submission,
)

T0 = pd.Timestamp("2026-01-06 03:35:00")


def write_validate_dataset(root: Path, *, decoy_fact: bool = True) -> Path:
    points = pd.DataFrame(
        {
            "sample_id": ["131672_a", "131672_b", "122048_c"],
            "tr_id": [131672, 131672, 122048],
            "T": [T0, T0 + pd.Timedelta(minutes=5), T0],
            "target_stop_id": [900, 901, 950],
            "target_time_begin": [
                T0 + pd.Timedelta(minutes=12),
                T0 + pd.Timedelta(minutes=16),
                T0 + pd.Timedelta(minutes=15),
            ],
            "cur_dev_s": [274.0, 300.0, -15.0],
        }
    )
    rows = []
    for tr_id in (131672, 122048):
        for s in range(1500, -600, -15):  # включая пакеты после T (должны игнорироваться)
            rows.append(
                {
                    "packet_id": s,
                    "tr_id": tr_id,
                    "unit_id": 1,
                    "event_time": T0 - pd.Timedelta(seconds=s),
                    "device_event_id": 0,
                    "location_valid": s % 45 != 0,
                    "gps_time": None,
                    "lon": 37.6 + s * 1e-5,
                    "lat": 55.7 + s * 1e-5,
                    "alt": 150.0,
                    "speed": float(abs(s) % 40),
                    "heading": 90.0,
                    "receive_time": None,
                    "is_hist_data": False,
                }
            )
    plan = pd.DataFrame(
        {
            "tt_action_item_id": [900, 901, 950],
            "time_begin": [T0 + pd.Timedelta(minutes=m) for m in (12, 16, 15)],
            "order_date": ["2026-01-06"] * 3,
            "manual_fill": [False, True, False],
            "tr_id": [131672, 131672, 122048],
            "geom": ["POINT (37.61 55.75)", "POINT (37.62 55.76)", "POINT (37.5 55.8)"],
            "building_address": ["a", "b", "c"],
        }
    )
    if decoy_fact:  # даже если бы в плановом файле был факт — он не читается
        plan["time_fact_begin"] = T0 + pd.Timedelta(hours=9)
    template = pd.DataFrame(
        {"sample_id": ["122048_c", "131672_a", "131672_b"], "prediction": [0.0, 1.0, 2.0]}
    )
    (root / "validate").mkdir(parents=True)
    points.to_csv(root / "validate/points.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "validate/traffic.csv", index=False)
    plan.to_csv(root / "validate/schedule_plan.csv", index=False)
    template.to_csv(root / "sample_submission.csv", sep=";", index=False)
    return root


# ------------------------------------------------------------------ loader safety


def test_validate_inputs_use_only_validate_files_and_planned_schedule(tmp_path, monkeypatch):
    root = write_validate_dataset(tmp_path / "ds")
    opened: list[str] = []
    real_read_csv = pd.read_csv

    def spy(path, *args, **kwargs):
        opened.append(Path(path).relative_to(root).as_posix())
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(official.pd, "read_csv", spy)
    inputs = load_validate_inputs(root)
    load_sample_submission_ids(root)

    assert set(opened) == {
        "validate/points.csv",
        "validate/traffic.csv",
        "validate/schedule_plan.csv",
        "sample_submission.csv",
    }
    assert "time_fact_begin" not in inputs.schedule_plan.columns
    for frame in (inputs.points, inputs.telemetry, inputs.schedule_plan):
        assert not {"time_fact_begin", "target_delay_s", "target_class"} & set(frame.columns)
    assert not (root / "test").exists() and not (root / "labels").exists()


def test_validate_points_with_target_columns_are_rejected(tmp_path):
    root = write_validate_dataset(tmp_path / "ds")
    points = pd.read_csv(root / "validate/points.csv")
    points["target_delay_s"] = 1.0
    points.to_csv(root / "validate/points.csv", index=False)
    with pytest.raises(OfficialDataSchemaError):
        load_validate_inputs(root)


def test_validate_features_build_with_exact_schema(tmp_path):
    inputs = load_validate_inputs(write_validate_dataset(tmp_path / "ds"))
    features = build_features_from_context(
        offline_context(inputs.points, inputs.telemetry, inputs.schedule_plan)
    )
    assert tuple(features.columns) == FEATURE_NAMES
    assert list(features.index) == ["131672_a", "131672_b", "122048_c"]
    assert features.loc["131672_a", "latest_packet_lag_s"] == 0.0


def test_sample_submission_template_is_strict(tmp_path):
    root = write_validate_dataset(tmp_path / "ds")
    assert load_sample_submission_ids(root) == ("122048_c", "131672_a", "131672_b")
    (root / "sample_submission.csv").write_text("sample_id,prediction\na,1\n")
    with pytest.raises(OfficialDataSchemaError):
        load_sample_submission_ids(root)


# ------------------------------------------------------------------ submission builder / writer


def test_build_submission_follows_template_order():
    frame = build_submission(["b", "a", "c"], ["a", "b", "c"], [1.0, 2.0, 3.0])
    assert frame.to_dict("list") == {"sample_id": ["b", "a", "c"], "prediction": [2.0, 1.0, 3.0]}


@pytest.mark.parametrize(
    ("required", "points", "values"),
    [
        (["a", "a"], ["a", "b"], [1.0, 2.0]),
        (["a", "b"], ["a", "a"], [1.0, 2.0]),
        (["a", "b"], ["a"], [1.0]),
        (["a"], ["a", "b"], [1.0, 2.0]),
        (["a", "b"], ["a", "b"], [1.0, float("nan")]),
        (["a", "b"], ["a", "b"], [1.0, float("inf")]),
        (["a", "b"], ["a", "b"], [1.0]),
        ([], [], []),
    ],
)
def test_build_submission_rejects_invalid_inputs(required, points, values):
    with pytest.raises(SubmissionError):
        build_submission(required, points, values)


def test_write_submission_exact_format(tmp_path):
    path = tmp_path / "submission.csv"
    write_submission(build_submission(["x_1", "y_2"], ["y_2", "x_1"], [-3.5, 12.25]), path)
    assert path.read_text() == "sample_id;prediction\nx_1;12.25\ny_2;-3.5\n"
    with pytest.raises(SubmissionError):
        write_submission(pd.DataFrame({"id": ["a"], "prediction": [1.0]}), tmp_path / "bad.csv")


# ------------------------------------------------------------------ offline E2E


def test_offline_e2e_validate_submission(tmp_path, catboost_bundle):
    for formulation in ("direct", "residual"):
        root = write_validate_dataset(tmp_path / f"ds-{formulation}")
        bundle = catboost_bundle(target_formulation=formulation)
        output = tmp_path / f"submission-{formulation}.csv"

        summary = generate_validate_submission(root, bundle, output)

        written = pd.read_csv(output, sep=";", dtype={"sample_id": str})
        assert output.read_text().splitlines()[0] == "sample_id;prediction"
        assert list(written.columns) == ["sample_id", "prediction"]
        assert written["sample_id"].tolist() == ["122048_c", "131672_a", "131672_b"]
        assert summary.n_rows == 3 and summary.output_name == output.name
        assert np.all(np.isfinite(written["prediction"].to_numpy()))

        inputs = load_validate_inputs(root)
        batch = offline_context(inputs.points, inputs.telemetry, inputs.schedule_plan)
        expected = dict(
            zip(
                [p.point_id for p in batch.points],
                ArtifactPredictor.load(bundle).predict(batch),
                strict=True,
            )
        )
        assert written.set_index("sample_id")["prediction"].to_dict() == expected


def test_submission_is_never_written_into_dataset(tmp_path, catboost_bundle):
    root = write_validate_dataset(tmp_path / "ds")
    with pytest.raises(SubmissionError, match="inside the official dataset"):
        generate_validate_submission(root, catboost_bundle(), root / "submission.csv")
