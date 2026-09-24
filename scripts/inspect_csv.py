#!/usr/bin/env python3
"""First-hour, schema-agnostic CSV inspection CLI.

Examples
--------
    uv run python scripts/inspect_csv.py --path data/raw/organizer.csv
    uv run python scripts/inspect_csv.py --path data/raw/organizer.csv \\
        --nrows 5000 --time-column event_time
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mostransport_ml.data.inspection import inspect_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path, help="Path to the CSV file to inspect.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit the number of rows read, for a fast look at a large file.",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="Optional column name to additionally report the min/max time range of.",
    )
    args = parser.parse_args()

    report = inspect_csv(args.path, nrows=args.nrows, time_column=args.time_column)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
