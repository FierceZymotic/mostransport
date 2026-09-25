#!/usr/bin/env python3
"""M1 — leakage-safe tabular baseline v1 на официальном датасете.

Шаги (конфигурация намеренно фиксирована, без HPO):
1. загрузить train/test через безопасный official adapter
   (schedule — только плановые поля, `time_fact_begin` не читается);
2. построить признаки `tabular-v1` одним и тем же Feature Builder'ом;
3. воспроизвести baselines на labels_test и сверить с Dataset Evidence v1;
4. выполнить ровно 6 заранее определённых CatBoost-экспериментов
   (3 train regime × 2 formulation); сетка и конфигурация зафиксированы
   заранее, labels_test не используется для early stopping/HPO/отбора во
   время fit — обученные модели оцениваются на нём после fit;
5. записать метаданные каждого эксперимента через ExperimentLogger.

Файлы validate не читаются. Структурный real-набор ТС (группа A) берётся
из плановых полей официального test schedule.

Запуск:
    MOSTRANSPORT_DATASET=/path/to/dataset uv run python scripts/run_offline_baseline.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from pathlib import Path

import catboost
import numpy as np
import pandas as pd

from mostransport_ml.data.official import (
    load_official_split,
    load_shared_real_vehicle_ids,
    planned_schedule_fingerprint,
    resolve_dataset_root,
)
from mostransport_ml.evaluation.baseline import MedianBaselineRegressor
from mostransport_ml.evaluation.metrics import mae
from mostransport_ml.experiments.log import ExperimentLogger, ExperimentRecord
from mostransport_ml.features.builder import build_features
from mostransport_ml.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from mostransport_ml.models.catboost_v1 import CATBOOST_V1_PARAMS, fit_predict
from mostransport_ml.models.regimes import (
    M1_EXPERIMENTS,
    assign_train_groups,
    select_training_rows,
)
from mostransport_ml.target.formulation import TARGET_SPEC, final_prediction, training_target

MODEL_NAME = "catboost_v1"
DEFAULT_LOG_PATH = Path("experiments/m1_runs.jsonl")

# Dataset Evidence v1 (tag dataset-evidence-v1): контрольные значения для сверки,
# а не результат — baselines пересчитываются заново ниже.
EVIDENCE_BASELINE_MAE = {"zero": 103.3371, "train_median": 100.8725, "cur_dev_s": 93.3598}
BASELINE_TOLERANCE_S = 1e-3
EXPECTED_SNAPSHOT = {
    "n_train_points": 4434,
    "n_test_points": 353,
    "n_real_vehicle_ids": 13,
    "n_candidate_vehicle_ids": 26,
}

RAW_BYTES_FILES = (
    "labels/labels_train.csv",
    "labels/labels_test.csv",
    "train/traffic.csv",
    "test/traffic.csv",
)
# Schedule-файлы содержат factual `time_fact_begin`, поэтому в fingerprint входит
# только их плановое allowlist-представление, а не сырые байты.
PLANNED_SCHEDULE_FILES = (
    "train/schedule.csv",
    "test/schedule.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_version(root: Path) -> str:
    """Отпечаток входов: сырые байты labels/telemetry + плановое представление schedule."""
    combined = hashlib.sha256()
    for rel in RAW_BYTES_FILES:
        combined.update(f"{rel}:raw:{_sha256(root / rel)}\n".encode())
    for rel in PLANNED_SCHEDULE_FILES:
        combined.update(f"{rel}:planned:{planned_schedule_fingerprint(root / rel)}\n".encode())
    return f"official-planned-sha256:{combined.hexdigest()[:16]}"


def check_snapshot(actual: dict[str, int]) -> list[str]:
    return [
        f"{key}: expected {expected}, got {actual[key]}"
        for key, expected in EXPECTED_SNAPSHOT.items()
        if actual[key] != expected
    ]


def mae_by_horizon(y_true: np.ndarray, y_pred: np.ndarray, horizon: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame({"err": np.abs(y_true - y_pred), "h": np.ceil(horizon).astype(int)})
    return {str(h): float(v) for h, v in frame.groupby("h")["err"].mean().items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", help="official dataset root (else $MOSTRANSPORT_DATASET)")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args(argv)

    root = resolve_dataset_root(args.dataset_root)
    data_version = dataset_version(root)
    train = load_official_split(root, "train")
    test = load_official_split(root, "test")
    real_ids = load_shared_real_vehicle_ids(root)

    train_ids = set(train.points["tr_id"].tolist())
    snapshot = {
        "n_train_points": len(train.points),
        "n_test_points": len(test.points),
        "n_real_vehicle_ids": len(real_ids & train_ids),
        "n_candidate_vehicle_ids": len(train_ids - real_ids),
    }
    problems = check_snapshot(snapshot)
    if problems:
        print("STOP: dataset snapshot differs from Dataset Evidence v1:", *problems, sep="\n  ")
        return 2

    X_train = build_features(train.points, train.telemetry, train.schedule_plan)
    X_test = build_features(test.points, test.telemetry, test.schedule_plan)
    y_train = train.target.loc[X_train.index].to_numpy(dtype=float)
    y_test = test.target.loc[X_test.index].to_numpy(dtype=float)
    cur_dev_train = X_train["cur_dev_s"].to_numpy()
    cur_dev_test = X_test["cur_dev_s"].to_numpy()

    median_model = MedianBaselineRegressor().fit(X_train, y_train)
    baselines = {
        "zero": mae(y_test, np.zeros_like(y_test)),
        "train_median": mae(y_test, median_model.predict(X_test)),
        "cur_dev_s": mae(y_test, cur_dev_test),
    }
    print(f"dataset root: {root}\ndata_version: {data_version}")
    print(f"features: {FEATURE_SCHEMA_VERSION}, n={len(FEATURE_NAMES)}")
    print(f"train median target: {median_model.median_}")
    print("baselines on labels_test (MAE, s):")
    mismatches = []
    for name, value in baselines.items():
        reference = EVIDENCE_BASELINE_MAE[name]
        print(f"  {name:<13} {value:.4f}   (evidence {reference:.4f})")
        if abs(value - reference) > BASELINE_TOLERANCE_S:
            mismatches.append(name)
    if mismatches:
        print(f"STOP: baselines differ from Dataset Evidence v1: {mismatches}")
        return 2

    groups = assign_train_groups(train.points["tr_id"].to_numpy(), real_ids)
    batch_id = uuid.uuid4().hex
    logger = ExperimentLogger(args.log_path)
    horizon_test = X_test["horizon_minutes"].to_numpy()
    results = []

    for spec in M1_EXPERIMENTS:
        selection = select_training_rows(groups, spec.train_regime)
        mask = selection.mask
        y_fit = training_target(y_train[mask], cur_dev_train[mask], spec.formulation)
        weights = None if selection.sample_weight is None else selection.sample_weight[mask]
        raw = fit_predict(X_train.loc[mask], y_fit, X_test, sample_weight=weights)
        prediction = final_prediction(raw, cur_dev_test, spec.formulation)

        test_mae = mae(y_test, prediction)
        delta = test_mae - baselines["cur_dev_s"]
        params = {
            "m1_batch_id": batch_id,
            "experiment_id": spec.experiment_id,
            "train_regime": spec.train_regime,
            "formulation": spec.formulation,
            "n_train_rows": int(mask.sum()),
            "n_group_A": selection.n_group_a,
            "n_group_B": selection.n_group_b,
            "sample_weight_policy": selection.weight_policy,
            "group_B_weight": selection.group_b_weight,
            "n_features": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "catboost_params": dict(CATBOOST_V1_PARAMS),
            "catboost_version": catboost.__version__,
            "random_seed": CATBOOST_V1_PARAMS["random_seed"],
            "n_test_rows": len(y_test),
            "test_mae_s": test_mae,
            "delta_mae_vs_cur_dev_s": delta,
            "baseline_test_mae_s": baselines,
            "diagnostic_median_abs_error_s": float(np.median(np.abs(y_test - prediction))),
            "diagnostic_mae_by_horizon_ceil_min": mae_by_horizon(y_test, prediction, horizon_test),
        }
        logger.log(
            ExperimentRecord(
                model_name=MODEL_NAME,
                validation_mae=test_mae,
                data_version=data_version,
                target_version=TARGET_SPEC.version,
                feature_version=FEATURE_SCHEMA_VERSION,
                params=params,
                notes=(
                    "M1 predefined experiment; validation_mae = MAE on official "
                    "labels/labels_test.csv, computed once after fit (not used for fit, "
                    "early stopping or tuning)."
                ),
            )
        )
        results.append(params)

    table = pd.DataFrame(results)[
        [
            "experiment_id",
            "train_regime",
            "formulation",
            "n_train_rows",
            "test_mae_s",
            "delta_mae_vs_cur_dev_s",
        ]
    ]
    print(f"\nM1 experiments (batch {batch_id}):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    balanced = next(r for r in results if r["train_regime"] == "balanced_group_weight")
    print(
        f"\ngroups: A={balanced['n_group_A']} rows, B={balanced['n_group_B']} rows; "
        f"balanced group B weight = {balanced['group_B_weight']:.6f}"
    )
    lowest = min(results, key=lambda r: r["test_mae_s"])
    print(
        f"lowest MAE among six predefined M1 experiments: {lowest['experiment_id']} "
        f"({lowest['test_mae_s']:.4f} s)"
    )
    print(f"experiment log: {args.log_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
