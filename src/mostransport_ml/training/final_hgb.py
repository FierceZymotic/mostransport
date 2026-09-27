"""Финальный production training path: official Group A → HGB H0 → Artifact Bundle v1.

Research закрыт; модуль воспроизводит выбранную модель production-путём:

    official dataset
      → Group A: train points, чей `tr_id` входит в planned-only real universe
        (`load_shared_real_vehicle_ids`; ID не захардкожены, группа B исключена)
      → target по `sample_id` (labels_train)
      → факты TRAIN schedule (`load_train_schedule_facts`)
        → `safe_current_deviation_seconds` (P1)
      → `offline_context(..., current_deviation_seconds=safe)`
        (official `cur_dev_s` физически убирается из points)
      → `build_features_from_context` (tabular-v1, 37)
      → `project_runtime_safe_features` (runtime-safe-v1, 29)
      → GroupKFold(5, groups=tr_id) OOF HGB H0 → gate против research reference
      → `fit_h0` на всей Group A → ArtifactManifest v1 → `export_bundle`
      → свежая загрузка bundle с диска → parity прогнозов.

Никогда не читаются: `labels/labels_test.csv`, `test/traffic.csv`, `validate/**`,
факт test schedule (из `test/schedule.csv` — только плановые поля для real
universe). Отчёт и manifest содержат только агрегаты/отпечатки, без строк
организаторских данных и абсолютных путей.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

import mostransport_ml
from mostransport_ml.artifacts.bundle import BUNDLE_FILENAME, MANIFEST_FILENAME, sha256_bytes
from mostransport_ml.artifacts.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactManifest,
    EvaluationSummary,
)
from mostransport_ml.data.official import (
    load_official_split,
    load_shared_real_vehicle_ids,
    planned_schedule_fingerprint,
    resolve_dataset_root,
)
from mostransport_ml.data.safe_deviation import (
    SAFE_DEVIATION_SEMANTICS,
    TRAIN_SCHEDULE_FILE,
    load_train_schedule_facts,
    safe_current_deviation_seconds,
    schedule_facts_fingerprint,
)
from mostransport_ml.evaluation.metrics import mae
from mostransport_ml.features.adapters import offline_context
from mostransport_ml.features.context import (
    CanonicalBatch,
    build_features_from_context,
    project_runtime_safe_features,
)
from mostransport_ml.features.schema import (
    RUNTIME_SAFE_FEATURE_NAMES,
    RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
)
from mostransport_ml.inference.model_families import MODEL_FAMILIES
from mostransport_ml.inference.predictor import ArtifactPredictor, export_bundle
from mostransport_ml.models.hgb_v1 import (
    HGB_H0_PARAMS,
    HGB_H0_TARGET_FORMULATION,
    fit_h0,
)

FINAL_MODEL_VERSION = "hgb-h0-runtime-safe-v1-group-a-v1"
FINAL_MODEL_FAMILY = "hist_gradient_boosting"
FINAL_TRAIN_REGIME = "group_a_only_hgb_h0_full_fit"
OOF_SPLIT = "group_a_groupkfold5_oof"
N_SPLITS = 5

# Research reference (read-only, вне репозитория): notebook
# `model_selection_runtime_safe`, кандидат "HGB direct", набор F0 — pooled OOF MAE,
# сохранённый полной точностью (`model_selection_summary.csv`).
RESEARCH_REFERENCE_OOF_MAE = 77.34108978455868
# Тот же код и те же версии (scikit-learn 1.9.1): допустим только шум порядка
# суммирования (ulp при MAE ≈ 77 — ~1.4e-14), поэтому допуск узкий.
OOF_TOLERANCE = 1e-9

# Инварианты официального датасета (Dataset Evidence v1 / research).
OFFICIAL_GROUP_A_ROWS = 1141
OFFICIAL_GROUP_A_TR_IDS = 13

TRAINING_ENTRYPOINT = "scripts/train_final_hgb.py"
SOURCE_TREE_SCOPE = "src/mostransport_ml/**/*.py + scripts/train_final_hgb.py"
# Что считается «relevant working tree» для git_dirty.
GIT_DIRTY_SCOPE = ("src/mostransport_ml", TRAINING_ENTRYPOINT, "pyproject.toml", "uv.lock")
PARITY_ATOL = 1e-12

_FINAL_BUNDLE_ENTRIES = frozenset(
    {BUNDLE_FILENAME, MANIFEST_FILENAME, MODEL_FAMILIES[FINAL_MODEL_FAMILY].model_filename}
)


class TrainingDataError(ValueError):
    """Официальные training-данные не проходят структурные инварианты."""


class TrainingOutputError(ValueError):
    """Небезопасное или занятое место для artifact/отчёта."""


class OOFGateError(RuntimeError):
    """OOF MAE не воспроизводит research reference: финальный fit/export запрещён."""


class ArtifactParityError(RuntimeError):
    """Прогнозы загруженного artifact'а отличаются от обученной модели."""


# --------------------------------------------------------------------------- data


def select_group_a(points: pd.DataFrame, real_tr_ids: Iterable[Any]) -> pd.DataFrame:
    """Train points, чей `tr_id` входит в real universe; порядок файла сохраняется."""
    real = frozenset(real_tr_ids)
    if not real:
        raise TrainingDataError("real tr_id universe is empty")
    sample_ids = points["sample_id"]
    if sample_ids.isna().any() or sample_ids.duplicated().any():
        raise TrainingDataError("train sample_id must be present and unique")
    selected = points[points["tr_id"].isin(real)].reset_index(drop=True)
    if selected.empty:
        raise TrainingDataError("Group A is empty: no train point belongs to the real universe")
    return selected


def align_target(sample_ids: Sequence[Any], target: pd.Series) -> np.ndarray:
    """Target строго по `sample_id` (не по позиции) в порядке `sample_ids`."""
    if not target.index.is_unique:
        raise TrainingDataError("target index (sample_id) must be unique")
    ids = pd.Index(list(sample_ids))
    if ids.has_duplicates:
        raise TrainingDataError("selected sample_id must be unique")
    missing = ids.difference(target.index)
    if len(missing):
        raise TrainingDataError(f"target is missing for {len(missing)} selected sample_id(s)")
    values = target.reindex(ids).to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise TrainingDataError("target must be finite for every selected sample_id")
    return values


def build_runtime_safe_matrix(
    points: pd.DataFrame,
    telemetry: pd.DataFrame,
    schedule_plan: pd.DataFrame,
    schedule_facts: pd.DataFrame,
) -> tuple[CanonicalBatch, pd.DataFrame, pd.Series]:
    """Safe deviation (P1) → канонический builder → P1-проекция `runtime-safe-v1`."""
    safe = safe_current_deviation_seconds(points, schedule_facts)
    # Official `cur_dev_s` убирается физически: override — единственный источник.
    point_frame = points.drop(columns=["cur_dev_s"], errors="ignore")
    batch = offline_context(point_frame, telemetry, schedule_plan, current_deviation_seconds=safe)
    X = project_runtime_safe_features(build_features_from_context(batch))
    if tuple(X.columns) != RUNTIME_SAFE_FEATURE_NAMES:
        raise TrainingDataError("training matrix columns differ from runtime-safe-v1")
    if X.index.tolist() != points["sample_id"].astype(str).tolist():
        raise TrainingDataError("training matrix rows differ from Group A sample_id order")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise TrainingDataError("training matrix contains +/-infinity")
    if not np.array_equal(X["cur_dev_s"].to_numpy(), safe.to_numpy()):
        raise TrainingDataError("cur_dev_s differs from the safe current deviation")
    return batch, X, safe


def sample_ids_sha256(sample_ids: Iterable[Any]) -> str:
    """Отпечаток упорядоченного списка `sample_id` (каждый id + перевод строки)."""
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8") + b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class GroupATrainingData:
    """Group A: точки (без official `cur_dev_s`), канонический batch, X (29), y, группы."""

    points: pd.DataFrame
    batch: CanonicalBatch
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    safe_deviation: pd.Series
    n_train_points: int
    schedule_facts_fingerprint: str

    @property
    def n_rows(self) -> int:
        return len(self.y)

    @property
    def n_tr_id(self) -> int:
        return len(set(self.groups.tolist()))

    @property
    def sample_ids_sha256(self) -> str:
        return sample_ids_sha256(self.X.index)

    @property
    def nan_cells(self) -> int:
        return int(self.X.isna().sum().sum())


def prepare_group_a_training(
    dataset_root: str | Path,
    *,
    expected_rows: int | None = None,
    expected_tr_ids: int | None = None,
) -> GroupATrainingData:
    """Official TRAIN → Group A → (batch, X runtime-safe-v1, y по sample_id, groups)."""
    root = Path(dataset_root)
    train = load_official_split(root, "train")
    real_tr_ids = load_shared_real_vehicle_ids(root)  # только плановые поля test schedule
    points = select_group_a(train.points, real_tr_ids)
    n_tr_id = points["tr_id"].nunique()
    if expected_rows is not None and len(points) != expected_rows:
        raise TrainingDataError(f"Group A has {len(points)} rows, expected {expected_rows}")
    if expected_tr_ids is not None and n_tr_id != expected_tr_ids:
        raise TrainingDataError(f"Group A has {n_tr_id} tr_id, expected {expected_tr_ids}")
    y = align_target(points["sample_id"].tolist(), train.target)
    facts = load_train_schedule_facts(root)
    batch, X, safe = build_runtime_safe_matrix(points, train.telemetry, train.schedule_plan, facts)
    return GroupATrainingData(
        points=points.drop(columns=["cur_dev_s"]),
        batch=batch,
        X=X,
        y=y,
        groups=points["tr_id"].to_numpy(),
        safe_deviation=safe,
        n_train_points=len(train.points),
        schedule_facts_fingerprint=schedule_facts_fingerprint(facts),
    )


# --------------------------------------------------------------------------- OOF


@dataclass(frozen=True)
class FoldResult:
    index: int
    n_validation: int
    validation_tr_ids: tuple[Any, ...]
    mae: float


@dataclass(frozen=True)
class OOFResult:
    mae: float
    predictions: np.ndarray
    folds: tuple[FoldResult, ...]
    n_rows: int
    n_groups: int
    n_splits: int


def evaluate_h0_oof(X: pd.DataFrame, y: Any, groups: Any, *, n_splits: int = N_SPLITS) -> OOFResult:
    """`GroupKFold(n_splits)` по `tr_id`: H0 без весов, early stopping и пост-обработки."""
    target = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if not (len(X) == len(target) == len(groups)):
        raise ValueError("X, y and groups must have the same number of rows")
    oof = np.full(len(target), np.nan)
    predicted = np.zeros(len(target), dtype=int)
    folds = []
    splitter = GroupKFold(n_splits=n_splits)
    for k, (train_idx, valid_idx) in enumerate(splitter.split(X, target, groups), start=1):
        train_groups, valid_groups = (
            set(groups[train_idx].tolist()),
            set(groups[valid_idx].tolist()),
        )
        if train_groups & valid_groups:
            raise RuntimeError(f"fold {k}: tr_id present in both train and validation")
        model = fit_h0(X.iloc[train_idx], target[train_idx])
        prediction = np.asarray(model.predict(X.iloc[valid_idx]), dtype=float)
        if prediction.shape != (len(valid_idx),) or not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"fold {k}: predictions are not one finite value per row")
        oof[valid_idx] = prediction
        predicted[valid_idx] += 1
        folds.append(
            FoldResult(
                index=k,
                n_validation=len(valid_idx),
                validation_tr_ids=tuple(sorted(valid_groups)),
                mae=mae(target[valid_idx], prediction),
            )
        )
    if not np.all(predicted == 1):
        raise RuntimeError("every row must be predicted exactly once out of fold")
    return OOFResult(
        mae=mae(target, oof),
        predictions=oof,
        folds=tuple(folds),
        n_rows=len(target),
        n_groups=len(set(groups.tolist())),
        n_splits=n_splits,
    )


def check_oof_gate(
    actual: float,
    *,
    reference: float = RESEARCH_REFERENCE_OOF_MAE,
    tolerance: float = OOF_TOLERANCE,
) -> float:
    """Вернуть |actual − reference|; вне допуска — `OOFGateError` (fit/export запрещены)."""
    delta = abs(float(actual) - float(reference))
    if not np.isfinite(delta) or delta > tolerance:
        raise OOFGateError(
            f"OOF MAE {actual!r} does not reproduce research reference {reference!r} "
            f"(|delta|={delta:.3e} > {tolerance:.1e})"
        )
    return delta


def fit_final_h0(X: pd.DataFrame, y: Any):
    """Финальный H0 на всех строках Group A (та же training-граница `fit_h0`)."""
    model = fit_h0(X, y)
    params = model.get_params()
    if {key: params[key] for key in HGB_H0_PARAMS} != dict(HGB_H0_PARAMS):
        raise RuntimeError("final model parameters differ from HGB H0")
    return model


# --------------------------------------------------------------------------- provenance


def package_source_files() -> dict[str, Path]:
    """Все `.py` пакета `mostransport_ml` под стабильными относительными именами."""
    package = Path(mostransport_ml.__file__).resolve().parent
    return {
        f"src/mostransport_ml/{path.relative_to(package).as_posix()}": path
        for path in sorted(package.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def source_tree_sha256(files: Mapping[str, Path]) -> str:
    """SHA-256 по (относительное имя, размер, байты) файлов в сортировке по имени."""
    digest = hashlib.sha256()
    for name in sorted(files):
        data = Path(files[name]).read_bytes()
        digest.update(name.encode("utf-8") + b"\0" + str(len(data)).encode("ascii") + b"\0")
        digest.update(data)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _repo_root(start: Path) -> Path | None:
    top = _git(start, "rev-parse", "--show-toplevel")
    return Path(top) if top else None


def build_code_provenance(entrypoint: str | Path | None = None) -> dict[str, Any]:
    """Честная привязка к реальному коду: HEAD, dirty и SHA-256 фактических исходников.

    Абсолютных путей нет: только относительные имена и хеши. `git_dirty=True`
    означает, что artifact собран кодом, отличным от HEAD (например, до коммита).
    """
    files = package_source_files()
    if entrypoint is not None:
        files[TRAINING_ENTRYPOINT] = Path(entrypoint)
    repo = _repo_root(Path(mostransport_ml.__file__).resolve().parent)
    head = _git(repo, "rev-parse", "HEAD") if repo else None
    status = (
        _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *GIT_DIRTY_SCOPE)
        if repo
        else None
    )
    import skops

    return {
        "git_head": head or "unavailable",
        "git_dirty": None if status is None else bool(status),
        "git_dirty_scope": list(GIT_DIRTY_SCOPE),
        "training_entrypoint": (
            TRAINING_ENTRYPOINT if entrypoint is not None else "mostransport_ml.training.final_hgb"
        ),
        "source_tree_sha256": source_tree_sha256(files),
        "source_tree_scope": SOURCE_TREE_SCOPE,
        "source_file_count": len(files),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "skops": skops.__version__,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_data_provenance(
    dataset_root: str | Path, data: GroupATrainingData
) -> dict[str, Any]:
    """Только агрегаты и отпечатки TRAIN-входов; без путей, строк и значений target."""
    root = Path(dataset_root)
    return {
        "source": "official_train_group_a",
        "selection": (
            "train prediction points whose tr_id is in the planned-only official "
            "test-schedule tr_id universe (load_shared_real_vehicle_ids)"
        ),
        "n_rows": data.n_rows,
        "n_tr_id": data.n_tr_id,
        "n_train_points_total": data.n_train_points,
        "non_group_a_rows_used": 0,
        "group_a_sample_ids_sha256": data.sample_ids_sha256,
        "labels_train_sha256": _file_sha256(root / "labels" / "labels_train.csv"),
        "train_traffic_sha256": _file_sha256(root / "train" / "traffic.csv"),
        "train_schedule_plan_fingerprint": planned_schedule_fingerprint(root / TRAIN_SCHEDULE_FILE),
        "train_schedule_facts_fingerprint": data.schedule_facts_fingerprint,
        "target": "target_delay_s",
        "current_deviation": SAFE_DEVIATION_SEMANTICS,
        "official_cur_dev_s_used": False,
        "x_nan_cells": data.nan_cells,
    }


def build_final_manifest(
    oof: OOFResult,
    training_data_provenance: Mapping[str, Any],
    code_provenance: Mapping[str, Any],
    *,
    created_at: datetime | None = None,
) -> ArtifactManifest:
    """ArtifactManifest v1 финального artifact'а; evaluation — OOF, не in-sample."""
    return ArtifactManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        model_version=FINAL_MODEL_VERSION,
        model_family=FINAL_MODEL_FAMILY,
        feature_schema_version=RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
        target_formulation=HGB_H0_TARGET_FORMULATION,
        train_regime=FINAL_TRAIN_REGIME,
        model_params=dict(HGB_H0_PARAMS),
        training_data_provenance=dict(training_data_provenance),
        code_provenance=dict(code_provenance),
        evaluation=EvaluationSummary(
            split=OOF_SPLIT, metric="mae", value=oof.mae, n_rows=oof.n_rows
        ),
        created_at=created_at or datetime.now(UTC),
    )


# --------------------------------------------------------------------------- outputs


def check_output_location(path: str | Path, dataset_root: str | Path) -> Path:
    """Выход не внутри официального датасета и не отслеживаемый git'ом путь репозитория."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(Path(dataset_root).resolve()):
        raise TrainingOutputError("output must not be inside the official dataset")
    anchor = next((p for p in (resolved, *resolved.parents) if p.exists()), None)
    repo = _repo_root(anchor if anchor.is_dir() else anchor.parent) if anchor else None
    if repo is not None and resolved.is_relative_to(repo.resolve()):
        # Fail closed: принимается только явный ответ git «ignored» (код 0);
        # 1 (не ignored), fatal/ошибка или отсутствие git — отказ.
        try:
            returncode = subprocess.run(
                ["git", "-C", str(repo), "check-ignore", "-q", str(resolved)],
                capture_output=True,
            ).returncode
        except OSError:
            returncode = None
        if returncode != 0:
            raise TrainingOutputError(
                "output inside the repository must be confirmed git-ignored "
                f"(e.g. artifacts/...); git check-ignore returned {returncode}"
            )
    return resolved


def _check_artifact_dir(artifact_dir: Path, replace: bool) -> bool:
    """True, если каталог занят прежним финальным artifact'ом и его можно заменить."""
    if not artifact_dir.exists() or (artifact_dir.is_dir() and not any(artifact_dir.iterdir())):
        return False
    if not replace:
        raise TrainingOutputError("artifact directory exists and is not empty (use --replace)")
    entries = {p.name for p in artifact_dir.iterdir()} if artifact_dir.is_dir() else set()
    manifest_path = artifact_dir / MANIFEST_FILENAME
    is_final = (
        entries <= _FINAL_BUNDLE_ENTRIES
        and manifest_path.is_file()
        and ArtifactManifest.from_json(manifest_path.read_bytes()).model_version
        == FINAL_MODEL_VERSION
    )
    if not is_final:
        raise TrainingOutputError("--replace only overwrites a previous final HGB artifact")
    return True


# --------------------------------------------------------------------------- orchestration


@dataclass(frozen=True)
class FinalTrainingResult:
    data: GroupATrainingData
    oof: OOFResult
    oof_delta: float
    manifest: ArtifactManifest
    bundle_sha256: str
    artifact_dir: Path
    report: dict[str, Any]


def build_report(
    data: GroupATrainingData,
    oof: OOFResult,
    *,
    reference: float,
    tolerance: float,
    status: str,
    code_provenance: Mapping[str, Any],
    bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Агрегатный отчёт OOF/artifact: без строк данных, target и абсолютных путей."""
    return {
        "status": status,
        "model_version": FINAL_MODEL_VERSION,
        "model_family": FINAL_MODEL_FAMILY,
        "feature_schema_version": RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
        "n_features": len(RUNTIME_SAFE_FEATURE_NAMES),
        "target_formulation": HGB_H0_TARGET_FORMULATION,
        "h0_params": dict(HGB_H0_PARAMS),
        "reference_oof_mae": reference,
        "actual_oof_mae": oof.mae,
        "abs_delta": abs(oof.mae - reference),
        "tolerance": tolerance,
        "n_rows": oof.n_rows,
        "n_groups": oof.n_groups,
        "n_splits": oof.n_splits,
        "folds": [
            {
                "fold": fold.index,
                "n_validation": fold.n_validation,
                "validation_tr_ids": list(fold.validation_tr_ids),
                "mae": fold.mae,
            }
            for fold in oof.folds
        ],
        "group_a_sample_ids_sha256": data.sample_ids_sha256,
        "x_nan_cells": data.nan_cells,
        "bundle_sha256": bundle_sha256,
        "code_provenance": dict(code_provenance),
    }


def _write_report(report: Mapping[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_final_artifact(
    dataset_root: str | Path | None,
    artifact_dir: str | Path,
    *,
    entrypoint: str | Path | None = None,
    report_path: str | Path | None = None,
    replace: bool = False,
    expected_rows: int | None = OFFICIAL_GROUP_A_ROWS,
    expected_tr_ids: int | None = OFFICIAL_GROUP_A_TR_IDS,
    reference: float = RESEARCH_REFERENCE_OOF_MAE,
    tolerance: float = OOF_TOLERANCE,
) -> FinalTrainingResult:
    """Полный P3-путь; bundle создаётся только после прохождения OOF gate.

    `expected_*`, `reference`, `tolerance` переопределяются только в тестах на
    синтетике; CLI использует официальные инварианты и research reference.
    """
    root = resolve_dataset_root(dataset_root)
    artifact_dir = check_output_location(artifact_dir, root)
    report_file = None if report_path is None else check_output_location(report_path, root)
    if report_file is not None and report_file.is_relative_to(artifact_dir):
        # Bundle v1 содержит ровно bundle.json, manifest.json, model.skops: отчёт
        # внутри него (или на его месте) сделал бы принятый bundle недействительным.
        raise TrainingOutputError("report must not be the artifact directory or inside it")
    replace_existing = _check_artifact_dir(artifact_dir, replace)

    data = prepare_group_a_training(
        root, expected_rows=expected_rows, expected_tr_ids=expected_tr_ids
    )
    oof = evaluate_h0_oof(data.X, data.y, data.groups)
    code_provenance = build_code_provenance(entrypoint)
    try:
        delta = check_oof_gate(oof.mae, reference=reference, tolerance=tolerance)
    except OOFGateError:
        report = build_report(
            data,
            oof,
            reference=reference,
            tolerance=tolerance,
            status="FAIL",
            code_provenance=code_provenance,
        )
        _write_report(report, report_file)
        raise

    model = fit_final_h0(data.X, data.y)
    expected = np.asarray(model.predict(data.X), dtype=float)
    manifest = build_final_manifest(
        oof, build_training_data_provenance(root, data), code_provenance
    )
    if replace_existing:
        shutil.rmtree(artifact_dir)
    descriptor = export_bundle(artifact_dir, model, manifest)
    del model  # дальше — только artifact, загруженный с диска

    predictor = ArtifactPredictor.load(artifact_dir)
    loaded = predictor.manifest
    if (
        loaded.model_version,
        loaded.model_family,
        loaded.feature_schema_version,
        loaded.target_formulation,
    ) != (
        FINAL_MODEL_VERSION,
        FINAL_MODEL_FAMILY,
        RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
        HGB_H0_TARGET_FORMULATION,
    ):
        raise ArtifactParityError("loaded artifact metadata differs from the final model")
    bundle_sha = sha256_bytes((artifact_dir / BUNDLE_FILENAME).read_bytes())
    if not (predictor.bundle_sha256 == descriptor.fingerprint() == bundle_sha):
        raise ArtifactParityError("bundle SHA-256 differs between export and load")
    actual = np.asarray(predictor.predict(data.batch), dtype=float)
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=0.0, atol=PARITY_ATOL
    ):
        raise ArtifactParityError("loaded artifact predictions differ from the trained model")
    reloaded = np.asarray(ArtifactPredictor.load(artifact_dir).predict(data.batch), dtype=float)
    if not np.array_equal(reloaded, actual):
        raise ArtifactParityError("a fresh load of the artifact predicts differently")

    report = build_report(
        data,
        oof,
        reference=reference,
        tolerance=tolerance,
        status="PASS",
        code_provenance=code_provenance,
        bundle_sha256=bundle_sha,
    )
    _write_report(report, report_file)
    return FinalTrainingResult(
        data=data,
        oof=oof,
        oof_delta=delta,
        manifest=loaded,
        bundle_sha256=bundle_sha,
        artifact_dir=artifact_dir,
        report=report,
    )
