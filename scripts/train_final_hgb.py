#!/usr/bin/env python3
"""Финальный production HGB H0 artifact из official TRAIN Group A (P3).

Один воспроизводимый путь: Group A (planned-only real tr_id universe) →
safe current deviation (P1) → канонический builder → `runtime-safe-v1` →
GroupKFold(5) OOF gate против research reference → финальный fit H0 на всей
Group A → Artifact Bundle v1 (`model.skops`) → загрузка с диска и parity.
Bundle создаётся только если OOF gate пройден. `labels_test`, `test/traffic.csv`,
`validate/**` и факт test schedule не читаются.

Запуск:
    MOSTRANSPORT_DATASET=/path/to/dataset uv run python scripts/train_final_hgb.py \\
        --artifact-dir artifacts/hgb-h0-runtime-safe-v1-group-a-v1 \\
        --report artifacts/reports/p3-hgb-h0-group-a-v1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mostransport_ml.data.official import resolve_dataset_root
from mostransport_ml.training.final_hgb import (
    FINAL_MODEL_VERSION,
    OOFGateError,
    TrainingDataError,
    TrainingOutputError,
    train_final_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", help="official dataset root (else $MOSTRANSPORT_DATASET)")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help=f"bundle output, e.g. artifacts/{FINAL_MODEL_VERSION}",
    )
    parser.add_argument("--report", type=Path, help="aggregate JSON report (no organizer rows)")
    parser.add_argument(
        "--replace", action="store_true", help="overwrite a previous final HGB artifact"
    )
    args = parser.parse_args(argv)

    try:
        result = train_final_artifact(
            resolve_dataset_root(args.dataset_root),
            args.artifact_dir,
            entrypoint=Path(__file__),
            report_path=args.report,
            replace=args.replace,
        )
    except OOFGateError as exc:
        print(f"OOF GATE FAILED — no artifact written: {exc}", file=sys.stderr)
        return 3
    except (TrainingDataError, TrainingOutputError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    oof, manifest, data = result.oof, result.manifest, result.data
    provenance = manifest.code_provenance
    print(f"Group A: rows={data.n_rows} tr_id={data.n_tr_id} (non-Group-A rows used: 0)")
    shape = f"{data.X.shape[0]}x{data.X.shape[1]}"
    print(f"X: {shape} {manifest.feature_schema_version} NaN cells={data.nan_cells}")
    print(
        f"OOF MAE={oof.mae!r} reference={result.report['reference_oof_mae']!r} "
        f"|delta|={result.oof_delta:.3e} → PASS"
    )
    for fold in oof.folds:
        tr_ids = ",".join(map(str, fold.validation_tr_ids))
        print(f"  fold {fold.index}: n_validation={fold.n_validation} tr_id=[{tr_ids}]", end="")
        print(f" MAE={fold.mae:.6f}")
    family = f"{manifest.model_family}, {manifest.target_formulation}"
    print(f"model_version: {manifest.model_version} ({family})")
    print(f"artifact: {args.artifact_dir}")
    print(f"bundle_sha256: {result.bundle_sha256}")
    print(f"source_tree_sha256: {provenance['source_tree_sha256']}")
    print(f"git_head: {provenance['git_head']} git_dirty: {provenance['git_dirty']}")
    print("load-back from disk: PASS; direct model vs ArtifactPredictor parity: PASS")
    if args.report is not None:
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
