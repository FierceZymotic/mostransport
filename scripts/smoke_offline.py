#!/usr/bin/env python3
"""End-to-end smoke test для offline-пайплайна.

Строит маленький синтетический dataframe (без организаторских данных),
затем прогоняет его через: temporal split -> median baseline fit ->
predict -> MAE. Проверяет только то, что offline-инфраструктура собрана
вместе и исполняема — не то, что какая-либо реальная модель хороша.

Запуск:
    uv run python scripts/smoke_offline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mostransport_ml.evaluation.baseline import MedianBaselineRegressor
from mostransport_ml.evaluation.metrics import mae
from mostransport_ml.evaluation.temporal import split_by_time_boundaries


def build_synthetic_dataframe(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="min")
    target = rng.normal(loc=5.0, scale=2.0, size=n_rows)
    return pd.DataFrame({"timestamp": timestamps, "target": target})


def main() -> int:
    df = build_synthetic_dataframe()

    split = split_by_time_boundaries(
        df,
        time_column="timestamp",
        train_end="2026-01-01T04:00:00",
        validation_end="2026-01-01T04:50:00",
    )
    print(
        f"split sizes -> train={len(split.train)} "
        f"validation={len(split.validation)} test={len(split.test)}"
    )

    model = MedianBaselineRegressor()
    model.fit(split.train, split.train["target"])

    predictions = model.predict(split.validation)
    validation_mae = mae(split.validation["target"].to_numpy(), predictions)

    print(f"median baseline (train) = {model.median_:.4f}")
    print(f"validation MAE          = {validation_mae:.4f}")
    print("smoke_offline: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
