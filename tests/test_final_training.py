"""P3: финальный training path HGB H0 на синтетическом наборе официальной структуры.

Никаких организаторских данных. Синтетический датасет повторяет раскладку
официального: train (labels/traffic/schedule с фактом), test/schedule.csv (план
+ мусорный факт), а также файлы, которые training path читать НЕ должен
(`labels_test`, `test/traffic.csv`, `validate/**`, `sample_submission.csv`) —
они недоступны на чтение и не должны появиться среди вызовов `read_csv`.
Реальный research gate (77.34…) проверяется только официальным прогоном.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.artifacts.bundle import BUNDLE_FILENAME, MANIFEST_FILENAME, sha256_bytes
from mostransport_ml.data.official import (
    SCHEDULE_PLAN_COLUMNS,
    OfficialDataSchemaError,
    load_official_split,
)
from mostransport_ml.data.safe_deviation import (
    SCHEDULE_FACT_COLUMNS,
    load_train_schedule_facts,
    schedule_facts_fingerprint,
)
from mostransport_ml.features.schema import (
    RUNTIME_SAFE_EXCLUDED_FEATURES,
    RUNTIME_SAFE_FEATURE_NAMES,
)
from mostransport_ml.inference.predictor import ArtifactPredictor
from mostransport_ml.models.hgb_v1 import HGB_H0_PARAMS, fit_h0
from mostransport_ml.training import final_hgb
from mostransport_ml.training.final_hgb import (
    FINAL_MODEL_VERSION,
    OOF_TOLERANCE,
    RESEARCH_REFERENCE_OOF_MAE,
    OOFGateError,
    TrainingDataError,
    TrainingOutputError,
    align_target,
    check_oof_gate,
    check_output_location,
    evaluate_h0_oof,
    package_source_files,
    prepare_group_a_training,
    select_group_a,
    source_tree_sha256,
    train_final_artifact,
)

REPO = Path(__file__).resolve().parents[1]
# Group A задаётся только test-планом; ID намеренно без «диапазонной» логики:
# 9000003 — в A, маленькие/«реальные на вид» ID — в B.
A_IDS = (501, 9000003, 77, 130001, 42, 610)
B_IDS = (134000, 7)
DAY = pd.Timestamp("2026-01-06")
FORBIDDEN = (
    "labels/labels_test.csv",
    "test/traffic.csv",
    "validate/points.csv",
    "validate/traffic.csv",
    "validate/schedule_plan.csv",
    "sample_submission.csv",
)
OFFICIAL_CUR_DEV = 99_999.0  # official cur_dev_s в labels — не должен попасть в X


def _vehicle_schedule(tr_id: int, rng: np.random.Generator) -> pd.DataFrame:
    plans = [DAY + pd.Timedelta(hours=6, minutes=3 * k) for k in range(61)]
    delays = np.cumsum(rng.normal(0.0, 25.0, len(plans))).round()
    facts = [p + pd.Timedelta(seconds=float(d)) for p, d in zip(plans, delays, strict=True)]
    facts = [
        None if p > DAY + pd.Timedelta(hours=8, minutes=45) else f
        for p, f in zip(plans, facts, strict=True)
    ]
    base_lon, base_lat = 37.4 + (tr_id % 97) * 1e-3, 55.6 + (tr_id % 89) * 1e-3
    return pd.DataFrame(
        {
            "tt_action_item_id": [tr_id * 1000 + k for k in range(len(plans))],
            "time_begin": [p.strftime("%Y-%m-%d %H:%M:%S.000000000") for p in plans],
            "time_fact_begin": [
                None if f is None else f.strftime("%Y-%m-%d %H:%M:%S.000000000") for f in facts
            ],
            "order_date": DAY.strftime("%Y-%m-%d"),
            "manual_fill": [bool(k % 3 == 0) for k in range(len(plans))],
            "tr_id": tr_id,
            "geom": [
                f"POINT ({base_lon + k * 2e-4} {base_lat + k * 1e-4})" for k in range(len(plans))
            ],
            "building_address": [f"stop {k}" for k in range(len(plans))],
        }
    )


def _vehicle_points(tr_id: int, schedule: pd.DataFrame, rng) -> pd.DataFrame:
    plans = pd.to_datetime(schedule["time_begin"])
    facts = pd.to_datetime(schedule["time_fact_begin"])
    rows = []
    for k in range(24):
        T = DAY + pd.Timedelta(hours=6, minutes=20 + 5 * k)
        window = (plans > T + pd.Timedelta(minutes=10)) & (plans <= T + pd.Timedelta(minutes=15))
        i = int(np.flatnonzero(window.to_numpy())[0])
        rows.append(
            {
                "sample_id": f"{tr_id}_{int(T.timestamp())}",
                "tr_id": tr_id,
                "T": T.strftime("%Y-%m-%d %H:%M:%S"),
                "target_stop_id": int(schedule["tt_action_item_id"].iloc[i]),
                "target_time_begin": schedule["time_begin"].iloc[i],
                "cur_dev_s": OFFICIAL_CUR_DEV + rng.normal(),
                "target_delay_s": (facts.iloc[i] - plans.iloc[i]).total_seconds(),
                "target_class": "ontime",
            }
        )
    return pd.DataFrame(rows)


def _vehicle_telemetry(tr_id: int, rng) -> pd.DataFrame:
    times = [DAY + pd.Timedelta(hours=5, minutes=55, seconds=20 * k) for k in range(495)]
    speed = np.clip(
        20 + 15 * np.sin(np.arange(len(times)) / 9) + rng.normal(0, 3, len(times)), 0, None
    )
    valid = rng.random(len(times)) > 0.1
    return pd.DataFrame(
        {
            "packet_id": np.arange(len(times)),
            "tr_id": tr_id,
            "event_time": [t.strftime("%Y-%m-%d %H:%M:%S.%f") for t in times],
            "location_valid": valid,
            "lon": [37.4 + (tr_id % 97) * 1e-3 + k * 1e-5 for k in range(len(times))],
            "lat": [55.6 + (tr_id % 89) * 1e-3 + k * 5e-6 for k in range(len(times))],
            "speed": speed.round(1),
        }
    )


def write_synthetic_official_dataset(root: Path) -> Path:
    rng = np.random.default_rng(20260106)
    schedules, points, telemetry = [], [], []
    for tr_id in (*A_IDS, *B_IDS):
        schedule = _vehicle_schedule(tr_id, rng)
        schedules.append(schedule)
        points.append(_vehicle_points(tr_id, schedule, rng))
        telemetry.append(_vehicle_telemetry(tr_id, rng))
    labels = pd.concat(points, ignore_index=True).sample(frac=1.0, random_state=3)  # порядок файла
    train_schedule = pd.concat(schedules, ignore_index=True)
    test_schedule = train_schedule[train_schedule["tr_id"].isin(A_IDS)].assign(
        time_fact_begin="NOT A TIMESTAMP — must never be parsed"
    )
    files = {
        "labels/labels_train.csv": labels,
        "train/traffic.csv": pd.concat(telemetry, ignore_index=True),
        "train/schedule.csv": train_schedule,
        "test/schedule.csv": test_schedule,
    }
    for rel, frame in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    for rel in FORBIDDEN:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("FORBIDDEN FOR TRAINING\n")
        path.chmod(0)
    return root


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = write_synthetic_official_dataset(tmp_path_factory.mktemp("official") / "dataset")
    yield root
    for rel in FORBIDDEN:
        (root / rel).chmod(0o600)


@pytest.fixture(scope="module")
def prepared(dataset):
    return prepare_group_a_training(dataset)


@pytest.fixture
def csv_reads(monkeypatch):
    """Все вызовы `pandas.read_csv`: (путь, usecols, nrows)."""
    calls = []
    real = pd.read_csv

    def recording(path, *args, **kwargs):
        calls.append((Path(path).resolve(), kwargs.get("usecols"), kwargs.get("nrows")))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording)
    return calls


def assert_training_reads_are_allowed(calls, root: Path) -> None:
    root = root.resolve()
    touched = {path.relative_to(root).as_posix() for path, _, _ in calls}
    assert not touched & set(FORBIDDEN)
    assert not any(name.startswith("validate/") for name in touched)
    for path, usecols, nrows in calls:
        name = path.relative_to(root).as_posix()
        if nrows == 0:
            continue  # только заголовок (проверка схемы)
        if name == "test/schedule.csv":
            assert set(usecols) <= set(SCHEDULE_PLAN_COLUMNS)  # planned-only
        elif name == "train/schedule.csv":
            assert set(usecols) in (set(SCHEDULE_PLAN_COLUMNS), set(SCHEDULE_FACT_COLUMNS))
        else:
            assert name in {"labels/labels_train.csv", "train/traffic.csv"}


# ------------------------------------------------------------------ factual loader


def test_train_facts_loader_reads_only_fact_columns_in_csv_order(dataset, csv_reads):
    facts = load_train_schedule_facts(dataset)
    assert tuple(facts.columns) == SCHEDULE_FACT_COLUMNS
    assert facts.index.equals(pd.RangeIndex(len(facts)))
    raw = pd.read_csv(dataset / "train/schedule.csv")
    assert (
        facts["tt_action_item_id"].tolist() == raw["tt_action_item_id"].tolist()
    )  # без сортировки
    assert str(facts["time_begin"].dtype) == "datetime64[ns]"
    assert facts["time_fact_begin"].isna().any() and facts["time_fact_begin"].notna().any()
    data_reads = [c for c in csv_reads if c[2] != 0 and c[0].name == "schedule.csv"]
    assert data_reads[0][1] == list(SCHEDULE_FACT_COLUMNS)
    assert all(c[0].parent.name == "train" for c in data_reads)


def test_train_facts_loader_has_no_other_split_and_validates_schema(tmp_path):
    (tmp_path / "test").mkdir()
    _vehicle_schedule(1, np.random.default_rng(0)).to_csv(tmp_path / "test/schedule.csv")
    with pytest.raises(FileNotFoundError):
        load_train_schedule_facts(tmp_path)  # test-факты этим loader'ом не прочитать
    (tmp_path / "train").mkdir()
    schedule = _vehicle_schedule(1, np.random.default_rng(0)).drop(columns="time_fact_begin")
    schedule.to_csv(tmp_path / "train/schedule.csv", index=False)
    with pytest.raises(OfficialDataSchemaError):
        load_train_schedule_facts(tmp_path)


def test_generic_official_loader_remains_factual_free(dataset):
    train = load_official_split(dataset, "train")
    assert "time_fact_begin" not in train.schedule_plan.columns


# ------------------------------------------------------------------ Group A and target


def test_group_a_comes_from_planned_real_universe_without_id_heuristics(dataset):
    train = load_official_split(dataset, "train")
    selected = select_group_a(train.points, frozenset(A_IDS))
    assert set(selected["tr_id"]) == set(A_IDS)
    assert not set(selected["tr_id"]) & set(B_IDS)
    file_order = [
        s for s, t in zip(train.points.sample_id, train.points.tr_id, strict=True) if t in A_IDS
    ]
    assert selected["sample_id"].tolist() == file_order


def test_select_group_a_rejects_bad_inputs(dataset):
    points = load_official_split(dataset, "train").points
    with pytest.raises(TrainingDataError, match="empty"):
        select_group_a(points, frozenset())
    with pytest.raises(TrainingDataError, match="empty"):
        select_group_a(points, frozenset({123456789}))
    duplicated = pd.concat([points, points.iloc[:1]], ignore_index=True)
    with pytest.raises(TrainingDataError, match="unique"):
        select_group_a(duplicated, frozenset(A_IDS))


def test_target_is_aligned_by_sample_id_not_position():
    target = pd.Series([10.0, 20.0, 30.0], index=["c", "a", "b"])
    np.testing.assert_array_equal(align_target(["a", "b", "c"], target), [20.0, 30.0, 10.0])
    shuffled = target.sample(frac=1.0, random_state=1)
    np.testing.assert_array_equal(align_target(["b", "c"], shuffled), [30.0, 10.0])


@pytest.mark.parametrize(
    ("ids", "target", "match"),
    [
        (["a", "x"], pd.Series([1.0, 2.0], index=["a", "b"]), "missing"),
        (["a", "a"], pd.Series([1.0, 2.0], index=["a", "b"]), "unique"),
        (["a"], pd.Series([1.0, 2.0], index=["a", "a"]), "unique"),
        (["a"], pd.Series([np.nan], index=["a"]), "finite"),
        (["a"], pd.Series([np.inf], index=["a"]), "finite"),
    ],
)
def test_target_alignment_failures(ids, target, match):
    with pytest.raises(TrainingDataError, match=match):
        align_target(ids, target)


def test_prepared_group_a_matrix(dataset, prepared):
    train = load_official_split(dataset, "train")
    group_a = train.points[train.points.tr_id.isin(A_IDS)]
    assert prepared.n_rows == len(group_a) == 6 * 24 and prepared.n_tr_id == len(A_IDS)
    assert prepared.n_train_points == len(train.points)
    assert tuple(prepared.X.columns) == RUNTIME_SAFE_FEATURE_NAMES and prepared.X.shape[1] == 29
    assert not set(RUNTIME_SAFE_EXCLUDED_FEATURES) & set(prepared.X.columns)
    assert prepared.X.index.tolist() == group_a["sample_id"].tolist()
    np.testing.assert_array_equal(prepared.y, train.target.loc[group_a["sample_id"]].to_numpy())
    np.testing.assert_array_equal(prepared.groups, group_a["tr_id"].to_numpy())
    np.testing.assert_array_equal(
        prepared.X["cur_dev_s"].to_numpy(), prepared.safe_deviation.to_numpy()
    )
    assert prepared.X["cur_dev_s"].abs().max() < OFFICIAL_CUR_DEV / 10  # official не использован
    assert "cur_dev_s" not in prepared.points.columns
    assert not np.isinf(prepared.X.to_numpy()).any()
    assert [p.point_id for p in prepared.batch.points] == prepared.X.index.tolist()


def test_prepare_enforces_expected_invariants(dataset):
    with pytest.raises(TrainingDataError, match="rows"):
        prepare_group_a_training(dataset, expected_rows=1141)
    with pytest.raises(TrainingDataError, match="tr_id"):
        prepare_group_a_training(dataset, expected_tr_ids=13)


# ------------------------------------------------------------------ OOF and gate


def test_oof_isolates_groups_and_predicts_every_row_once(prepared):
    from sklearn.model_selection import GroupKFold

    oof = evaluate_h0_oof(prepared.X, prepared.y, prepared.groups)
    assert (oof.n_rows, oof.n_groups, oof.n_splits) == (144, 6, 5)
    assert np.isfinite(oof.predictions).all()
    assert sum(f.n_validation for f in oof.folds) == oof.n_rows
    held_out = [tr for f in oof.folds for tr in f.validation_tr_ids]
    assert sorted(held_out) == sorted(A_IDS)  # каждый tr_id валидируется ровно один раз
    splits = list(GroupKFold(n_splits=5).split(prepared.X, prepared.y, prepared.groups))
    for fold, (train_idx, valid_idx) in zip(oof.folds, splits, strict=True):
        assert not set(prepared.groups[train_idx]) & set(prepared.groups[valid_idx])
        assert fold.validation_tr_ids == tuple(sorted(set(prepared.groups[valid_idx].tolist())))
    assert oof.mae == pytest.approx(np.mean(np.abs(prepared.y - oof.predictions)), abs=1e-12)


def test_oof_non_finite_prediction_fails(prepared, monkeypatch):
    class NanModel:
        def predict(self, X):
            return np.full(len(X), np.nan)

    monkeypatch.setattr(final_hgb, "fit_h0", lambda X, y: NanModel())
    with pytest.raises(RuntimeError, match="finite"):
        evaluate_h0_oof(prepared.X, prepared.y, prepared.groups)


def test_oof_gate_reference_and_decisions():
    assert RESEARCH_REFERENCE_OOF_MAE == 77.34108978455868
    assert OOF_TOLERANCE == 1e-9
    assert check_oof_gate(RESEARCH_REFERENCE_OOF_MAE) == 0.0
    assert check_oof_gate(RESEARCH_REFERENCE_OOF_MAE + 5e-10) <= OOF_TOLERANCE
    for bad in (RESEARCH_REFERENCE_OOF_MAE + 2e-9, 77.341090, 77.0, float("nan")):
        with pytest.raises(OOFGateError):
            check_oof_gate(bad)


def test_gate_failure_blocks_final_fit_and_export(dataset, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(final_hgb, "fit_final_h0", lambda *a: calls.append("fit"))
    monkeypatch.setattr(final_hgb, "export_bundle", lambda *a: calls.append("export"))
    report = tmp_path / "reports" / "fail.json"
    with pytest.raises(OOFGateError):
        train_final_artifact(
            dataset,
            tmp_path / "artifact",
            report_path=report,
            expected_rows=None,
            expected_tr_ids=None,
            reference=0.0,
        )
    assert calls == []
    assert not (tmp_path / "artifact").exists()
    content = json.loads(report.read_text())
    assert content["status"] == "FAIL" and content["bundle_sha256"] is None


# ------------------------------------------------------------------ final artifact


@pytest.fixture(scope="module")
def final_result(dataset, prepared, tmp_path_factory):
    reference = evaluate_h0_oof(prepared.X, prepared.y, prepared.groups).mae  # только синтетика
    out = tmp_path_factory.mktemp("final")
    entrypoint = REPO / "scripts" / "train_final_hgb.py"
    result = train_final_artifact(
        dataset,
        out / "artifact",
        entrypoint=entrypoint,
        report_path=out / "reports" / "p3.json",
        expected_rows=None,
        expected_tr_ids=None,
        reference=reference,
    )
    return result, out


def test_final_manifest_fields(final_result, prepared):
    result, _ = final_result
    manifest = result.manifest
    assert manifest.artifact_schema_version == "artifact-manifest-v1"
    assert manifest.model_version == FINAL_MODEL_VERSION == "hgb-h0-runtime-safe-v1-group-a-v1"
    assert manifest.model_family == "hist_gradient_boosting"
    assert manifest.feature_schema_version == "runtime-safe-v1"
    assert manifest.target_formulation == "direct"
    assert manifest.train_regime == "group_a_only_hgb_h0_full_fit"
    assert dict(manifest.model_params) == dict(HGB_H0_PARAMS)
    evaluation = manifest.evaluation
    assert (evaluation.split, evaluation.metric) == ("group_a_groupkfold5_oof", "mae")
    assert evaluation.value == result.oof.mae and evaluation.n_rows == prepared.n_rows
    data = manifest.training_data_provenance
    assert data["n_rows"] == 144 and data["n_tr_id"] == 6 and data["non_group_a_rows_used"] == 0
    assert data["official_cur_dev_s_used"] is False
    assert data["group_a_sample_ids_sha256"] == prepared.sample_ids_sha256
    assert data["train_schedule_facts_fingerprint"] == prepared.schedule_facts_fingerprint
    code = manifest.code_provenance
    assert code["training_entrypoint"] == "scripts/train_final_hgb.py"
    assert len(code["source_tree_sha256"]) == 64
    assert {"git_head", "git_dirty", "scikit_learn", "skops"} <= set(code)


def test_final_artifact_reloads_from_disk_with_prediction_parity(final_result, prepared):
    result, out = final_result
    directory = out / "artifact"
    assert sorted(p.name for p in directory.iterdir()) == [
        BUNDLE_FILENAME,
        MANIFEST_FILENAME,
        "model.skops",
    ]
    assert result.bundle_sha256 == sha256_bytes((directory / BUNDLE_FILENAME).read_bytes())
    predictor = ArtifactPredictor.load(directory)
    assert predictor.bundle_sha256 == result.bundle_sha256
    expected = fit_h0(prepared.X, prepared.y).predict(prepared.X)  # детерминированный refit
    np.testing.assert_allclose(predictor.predict(prepared.batch), expected, rtol=0, atol=1e-12)


def test_manifest_and_report_contain_no_paths_or_rows(final_result, dataset):
    result, out = final_result
    manifest_text = (out / "artifact" / MANIFEST_FILENAME).read_text()
    report_text = (out / "reports" / "p3.json").read_text()
    for text in (manifest_text, report_text):
        for secret in (str(dataset), str(out), str(REPO), str(Path.home())):
            assert secret not in text
        assert 'target_delay_s":' not in text  # ни одного значения target
    report = json.loads(report_text)
    assert report["status"] == "PASS" and report["bundle_sha256"] == result.bundle_sha256
    assert report["n_rows"] == 144 and len(report["folds"]) == 5
    assert all(
        set(fold) == {"fold", "n_validation", "validation_tr_ids", "mae"}
        for fold in report["folds"]
    )


def test_replace_only_overwrites_a_previous_final_artifact(
    final_result, dataset, prepared, tmp_path
):
    _, out = final_result
    reference = final_result[0].oof.mae
    with pytest.raises(TrainingOutputError, match="--replace"):
        final_hgb._check_artifact_dir(out / "artifact", replace=False)
    assert final_hgb._check_artifact_dir(out / "artifact", replace=True) is True
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep")
    with pytest.raises(TrainingOutputError, match="previous final"):
        train_final_artifact(
            dataset,
            foreign,
            replace=True,
            expected_rows=None,
            expected_tr_ids=None,
            reference=reference,
        )
    assert (foreign / "notes.txt").read_text() == "keep"


# ------------------------------------------------------------------ provenance and outputs


def test_source_tree_fingerprint_is_deterministic(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    first = source_tree_sha256({"src/a.py": a, "src/b.py": b})
    assert first == source_tree_sha256({"src/b.py": b, "src/a.py": a})
    assert first != source_tree_sha256({"src/a.py": a, "src/c.py": b})  # имя входит в отпечаток
    b.write_text("y = 3\n")
    assert first != source_tree_sha256({"src/a.py": a, "src/b.py": b})
    files = package_source_files()
    assert "src/mostransport_ml/training/final_hgb.py" in files
    assert all(name.startswith("src/mostransport_ml/") for name in files)
    assert not any("__pycache__" in name for name in files)
    assert source_tree_sha256(files) == source_tree_sha256(package_source_files())


def test_output_location_guards(dataset, tmp_path):
    with pytest.raises(TrainingOutputError, match="dataset"):
        check_output_location(dataset / "artifact", dataset)
    assert (
        check_output_location(tmp_path / "artifact", dataset) == (tmp_path / "artifact").resolve()
    )
    if (REPO / ".git").exists():
        with pytest.raises(TrainingOutputError, match="git-ignored"):
            check_output_location(REPO / "src" / "final-artifact", dataset)
        check_output_location(REPO / "artifacts" / FINAL_MODEL_VERSION, dataset)


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="needs a git checkout")
@pytest.mark.parametrize("outcome", [1, 128, OSError("git vanished")])
def test_output_location_fails_closed_unless_git_confirms_ignored(dataset, monkeypatch, outcome):
    ignored_path = REPO / "artifacts" / FINAL_MODEL_VERSION
    check_output_location(ignored_path, dataset)  # реальный git: 0 → принято
    real_run = subprocess.run

    def fake_run(args, *rest, **kwargs):
        if "check-ignore" in args:
            if isinstance(outcome, Exception):
                raise outcome
            return subprocess.CompletedProcess(args, outcome)
        return real_run(args, *rest, **kwargs)

    monkeypatch.setattr(final_hgb.subprocess, "run", fake_run)
    with pytest.raises(TrainingOutputError, match="git-ignored"):
        check_output_location(ignored_path, dataset)


def test_report_inside_artifact_dir_is_rejected_before_any_work(
    dataset, final_result, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(final_hgb, "prepare_group_a_training", lambda *a, **k: calls.append(1))
    fresh = tmp_path / "fresh"
    for report in (fresh, fresh / "report.json", fresh / "reports" / "p3.json"):
        with pytest.raises(TrainingOutputError, match="report"):
            train_final_artifact(dataset, fresh, report_path=report, expected_rows=None)
    assert not fresh.exists() and calls == []

    # Уже принятый bundle с --replace и отчётом внутри: отказ, bundle не тронут.
    existing = tmp_path / "existing"
    shutil.copytree(final_result[1] / "artifact", existing)
    before = {p.name: p.read_bytes() for p in existing.iterdir()}
    with pytest.raises(TrainingOutputError, match="report"):
        train_final_artifact(dataset, existing, report_path=existing / "report.json", replace=True)
    assert {p.name: p.read_bytes() for p in existing.iterdir()} == before
    assert ArtifactPredictor.load(existing).model_version() == FINAL_MODEL_VERSION
    assert calls == []


def test_schedule_facts_fingerprint_covers_exactly_the_consumed_facts(dataset, prepared, tmp_path):
    facts = load_train_schedule_facts(dataset)
    fingerprint = schedule_facts_fingerprint(facts)
    assert fingerprint == prepared.schedule_facts_fingerprint
    assert fingerprint == schedule_facts_fingerprint(facts.copy())
    assert fingerprint == schedule_facts_fingerprint(facts.assign(extra_column=1))
    raw = pd.read_csv(dataset / "train/schedule.csv", usecols=list(SCHEDULE_FACT_COLUMNS))
    assert fingerprint == schedule_facts_fingerprint(raw)  # строки == разобранные времена

    changed = facts.copy()
    confirmed = int(changed["time_fact_begin"].first_valid_index())
    changed.loc[confirmed, "time_fact_begin"] += pd.Timedelta(seconds=1)
    assert schedule_facts_fingerprint(changed) != fingerprint
    assert schedule_facts_fingerprint(facts.iloc[::-1].reset_index(drop=True)) != fingerprint

    # Изменение непотребляемых колонок schedule (адрес, geom) не меняет отпечаток.
    schedule = pd.read_csv(dataset / "train/schedule.csv")
    other = tmp_path / "other"
    (other / "train").mkdir(parents=True)
    schedule.assign(
        building_address="changed address", geom="POINT (1 1)", order_date="2026-01-07"
    ).to_csv(other / "train/schedule.csv", index=False)
    assert schedule_facts_fingerprint(load_train_schedule_facts(other)) == fingerprint


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="needs a git checkout")
def test_generated_official_outputs_are_git_ignored():
    paths = [
        f"artifacts/{FINAL_MODEL_VERSION}/model.skops",
        f"artifacts/{FINAL_MODEL_VERSION}/manifest.json",
        "artifacts/reports/p3-hgb-h0-group-a-v1.json",
        "submission-hgb-h0-runtime-safe-v1.csv",
    ]
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", *paths], capture_output=True, text=True
    )
    assert sorted(result.stdout.split()) == sorted(paths)


# ------------------------------------------------------------------ forbidden access


def test_training_path_never_reads_forbidden_files(dataset, csv_reads, tmp_path):
    if os.geteuid() != 0:  # недоступны на чтение: любое обращение упало бы
        for rel in FORBIDDEN:
            with pytest.raises(PermissionError):
                (dataset / rel).read_bytes()
    train_final_artifact(
        dataset,
        tmp_path / "artifact",
        expected_rows=None,
        expected_tr_ids=None,
        tolerance=float("inf"),  # только для проверки доступа к файлам
    )
    assert csv_reads
    assert_training_reads_are_allowed(csv_reads, dataset)
    test_schedule_reads = [
        c for c in csv_reads if c[0] == (dataset / "test/schedule.csv").resolve()
    ]
    assert test_schedule_reads and all(
        n == 0 or "time_fact_begin" not in cols for _, cols, n in test_schedule_reads
    )


def test_cli_enforces_official_invariants_before_any_artifact(dataset, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "train_final_hgb", REPO / "scripts/train_final_hgb.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    code = cli.main(["--dataset-root", str(dataset), "--artifact-dir", str(tmp_path / "a")])
    assert code == 2
    assert "expected 1141" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()
