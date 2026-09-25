"""Безопасный adapter официального датасета организаторов (offline, CSV).

Граница ответственности: только чтение и валидация схемы. Никакого
feature engineering здесь нет — `features/` получает уже загруженные
DataFrame'ы явно и не знает про файлы.

Leakage-гарантии, обеспечиваемые именно здесь:

- schedule читается строго по allowlist плановых колонок через
  `usecols` — `time_fact_begin` физически не попадает в память;
- target (`target_delay_s`) отделяется от prediction points и
  возвращается отдельной Series; `target_class` не читается вовсе;
- для validate нет никакого loader'а: M1 не читает файлы validate;
- structural real-набор `tr_id` берётся из плановых полей официального
  test schedule (тем же allowlist-loader'ом);
- fingerprint schedule для provenance строится только по плановому
  allowlist-представлению, поэтому factual-колонки на него не влияют.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATASET_ENV_VAR = "MOSTRANSPORT_DATASET"

Split = Literal["train", "test"]

POINT_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "tr_id",
    "T",
    "target_stop_id",
    "target_time_begin",
    "cur_dev_s",
)
TARGET_COLUMN = "target_delay_s"

TELEMETRY_COLUMNS: tuple[str, ...] = (
    "tr_id",
    "event_time",
    "location_valid",
    "lon",
    "lat",
    "speed",
)

SCHEDULE_PLAN_COLUMNS: tuple[str, ...] = (
    "tt_action_item_id",
    "time_begin",
    "order_date",
    "manual_fill",
    "tr_id",
    "geom",
    "building_address",
)

FACTUAL_SCHEDULE_COLUMNS: frozenset[str] = frozenset({"time_fact_begin"})

_SPLIT_FILES: dict[str, dict[str, str]] = {
    "train": {
        "labels": "labels/labels_train.csv",
        "telemetry": "train/traffic.csv",
        "schedule": "train/schedule.csv",
    },
    "test": {
        "labels": "labels/labels_test.csv",
        "telemetry": "test/traffic.csv",
        "schedule": "test/schedule.csv",
    },
}


class OfficialDataSchemaError(ValueError):
    """Файл официального датасета не содержит обязательных колонок."""

    def __init__(self, path: Path, missing: list[str], available: list[str]) -> None:
        self.path = path
        self.missing = missing
        self.available = available
        super().__init__(
            f"{path}: missing required columns {missing}; available columns: {available}"
        )


@dataclass(frozen=True)
class OfficialSplit:
    """Загруженный split: prediction points отдельно от target.

    `points` — одна строка на prediction point, без target-колонок.
    `target` — `target_delay_s`, индексирован `sample_id`.
    `telemetry` — только колонки `TELEMETRY_COLUMNS`.
    `schedule_plan` — только плановые колонки `SCHEDULE_PLAN_COLUMNS`.
    """

    name: str
    points: pd.DataFrame
    target: pd.Series
    telemetry: pd.DataFrame
    schedule_plan: pd.DataFrame


def resolve_dataset_root(explicit: str | Path | None = None) -> Path:
    """Явный путь > env `MOSTRANSPORT_DATASET`; иначе понятная ошибка."""
    raw = explicit if explicit is not None else os.environ.get(DATASET_ENV_VAR)
    if not raw:
        raise ValueError(
            f"Dataset root is not configured: pass it explicitly or set {DATASET_ENV_VAR}"
        )
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")
    return root


def _require_columns(path: Path, required: tuple[str, ...]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Official dataset file not found: {path}")
    available = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in required if c not in available]
    if missing:
        raise OfficialDataSchemaError(path, missing, available)


def _as_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).astype("datetime64[ns]")


def load_points_with_target(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Прочитать labels-файл: вернуть (points без target, target по sample_id)."""
    required = (*POINT_COLUMNS, TARGET_COLUMN)
    _require_columns(path, required)
    labels = pd.read_csv(path, usecols=list(required))
    labels["T"] = _as_ns(labels["T"])
    labels["target_time_begin"] = _as_ns(labels["target_time_begin"])

    if labels["sample_id"].duplicated().any():
        raise ValueError(f"{path}: sample_id is not unique")
    target_values = labels[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.all(np.isfinite(target_values)):
        raise ValueError(f"{path}: {TARGET_COLUMN} contains NaN or infinite values")

    points = labels[list(POINT_COLUMNS)].reset_index(drop=True)
    target = pd.Series(
        target_values, index=pd.Index(labels["sample_id"], name="sample_id"), name=TARGET_COLUMN
    )
    return points, target


def load_telemetry(path: Path) -> pd.DataFrame:
    _require_columns(path, TELEMETRY_COLUMNS)
    telemetry = pd.read_csv(
        path,
        usecols=list(TELEMETRY_COLUMNS),
        dtype={
            "tr_id": "int64",
            "location_valid": "boolean",
            "lon": "float64",
            "lat": "float64",
            "speed": "float64",
        },
    )
    telemetry["event_time"] = _as_ns(telemetry["event_time"])
    return telemetry[list(TELEMETRY_COLUMNS)]


def load_schedule_plan(path: Path) -> pd.DataFrame:
    """Только плановые поля schedule; `time_fact_begin` не читается (usecols)."""
    _require_columns(path, SCHEDULE_PLAN_COLUMNS)
    plan = pd.read_csv(path, usecols=list(SCHEDULE_PLAN_COLUMNS))
    leaked = FACTUAL_SCHEDULE_COLUMNS & set(plan.columns)
    if leaked:  # защитный backstop: usecols уже гарантирует это
        raise RuntimeError(f"{path}: factual schedule columns loaded: {sorted(leaked)}")
    plan["time_begin"] = _as_ns(plan["time_begin"])
    plan["order_date"] = _as_ns(plan["order_date"])
    return plan[list(SCHEDULE_PLAN_COLUMNS)]


def planned_schedule_fingerprint(path: Path) -> str:
    """SHA-256 планового представления schedule-файла.

    Строится из того же allowlist-представления, что и `load_schedule_plan`
    (колонки `SCHEDULE_PLAN_COLUMNS` в фиксированном порядке, строки в
    порядке файла), сериализованного детерминированно в CSV. Поэтому
    fingerprint не зависит от factual-колонок (`time_fact_begin`), но
    меняется при изменении любого планового значения.
    """
    serialized = load_schedule_plan(path).to_csv(
        index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S.%f"
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_official_split(root: str | Path, split: Split) -> OfficialSplit:
    """Загрузить train или test. Для validate modeling-loader'а нет намеренно."""
    if split not in _SPLIT_FILES:
        raise ValueError(f"Unsupported split {split!r}; expected one of {sorted(_SPLIT_FILES)}")
    root = Path(root)
    files = _SPLIT_FILES[split]
    points, target = load_points_with_target(root / files["labels"])
    return OfficialSplit(
        name=split,
        points=points,
        target=target,
        telemetry=load_telemetry(root / files["telemetry"]),
        schedule_plan=load_schedule_plan(root / files["schedule"]),
    )


def load_shared_real_vehicle_ids(root: str | Path) -> frozenset[int]:
    """Structural real-набор `tr_id`: ТС официального test schedule.

    Читается через `load_schedule_plan` (только плановые поля) — без
    prediction points, target, factual schedule и без файлов validate.
    """
    plan = load_schedule_plan(Path(root) / _SPLIT_FILES["test"]["schedule"])
    return frozenset(int(v) for v in plan["tr_id"].unique())
