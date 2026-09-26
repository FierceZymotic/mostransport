#!/usr/bin/env python3
"""Validate submission из Artifact Bundle v1.

Использует только validate/points.csv, validate/traffic.csv,
validate/schedule_plan.csv и sample_submission.csv; пишет `sample_id;prediction`
в порядке шаблона. Модель не обучается и не выбирается — используется ровно
переданный bundle.

Запуск:
    MOSTRANSPORT_DATASET=/path/to/dataset uv run python scripts/make_submission.py \\
        --artifact-dir artifacts/<bundle> --output submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mostransport_ml.data.official import resolve_dataset_root
from mostransport_ml.inference.submission import generate_validate_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", help="official dataset root (else $MOSTRANSPORT_DATASET)")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = generate_validate_submission(
        resolve_dataset_root(args.dataset_root), args.artifact_dir, args.output
    )
    print(
        f"submission written: {summary.output_name} rows={summary.n_rows} "
        f"model_version={summary.model_version} bundle_sha256={summary.bundle_sha256}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
