"""Offline point-in-time-safe текущее отклонение по factual schedule (training-side).

Воспроизводит зафиксированную research-семантику safe deviation для
`runtime-safe-v1`. Для каждой prediction point `(sample_id, tr_id, T)`:

1. кандидаты — factual-строки schedule того же `tr_id` с
   `time_fact_begin <= T` (строки без факта не подтверждены и не участвуют);
2. кандидатов нет → `0.0`;
3. иначе берётся последнее подтверждённое событие: максимальный
   `time_fact_begin`; при ничьей — максимальный ЧИСЛОВОЙ `tt_action_item_id`
   (`"10" > "9"`); если и он совпадает — последняя строка в исходном порядке
   строк (порядок CSV);
4. результат — `(time_fact_begin − time_begin)` в секундах, знак сохраняется,
   без clamp.

Tie-break — детерминированное research-правило, а не восстановленный
tie-break организаторов или Backend'а.

Граница leakage: factual schedule существует только offline. Модуль
возвращает только значения отклонения (Series по `sample_id`) для
`offline_context(..., current_deviation_seconds=...)`. Сами факты не
передаются ни в Feature Builder (он отклоняет `time_fact_begin`), ни в
runtime-контекст и не становятся признаками. Строки с `time_fact_begin > T`
не участвуют в выборе ни на одном шаге (в том числе в разборе ID при
ничьей), что дополнительно проверяется для каждого выбранного события.
Факты `test/schedule.csv` нельзя применять к точкам validate (Dataset
Evidence v1: они раскрывают hidden target validate).

Порядок строк `schedule_facts` считается исходным порядком CSV. Поэтому индекс
обязан быть уникальным и возрастающим (например, `RangeIndex` от
`pd.read_csv(path, usecols=SCHEDULE_FACT_COLUMNS)` после любой фильтрации
строк). Переупорядоченный кадр отклоняется, а не молча используется.

`load_train_schedule_facts` — единственный loader фактов: training-only, жёстко
только `train/schedule.csv` (без параметра split, поэтому факты test/validate им
не прочитать), только `SCHEDULE_FACT_COLUMNS`, порядок CSV сохранён. Общие
official loaders (`data/official.py`) по-прежнему фактов не читают.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mostransport_ml.data.official import OfficialDataSchemaError

POINT_KEY_COLUMNS: tuple[str, ...] = ("sample_id", "tr_id", "T")
SCHEDULE_FACT_COLUMNS: tuple[str, ...] = (
    "tr_id",
    "tt_action_item_id",
    "time_begin",
    "time_fact_begin",
)
SAFE_DEVIATION_NAME = "safe_current_deviation_seconds"
TRAIN_SCHEDULE_FILE = "train/schedule.csv"
# Стабильный идентификатор семантики (для provenance artifact'а).
SAFE_DEVIATION_SEMANTICS = (
    "p1-safe-deviation-v1: latest confirmed fact <= T of same tr_id; "
    "tie: max numeric tt_action_item_id, then last CSV row; none: 0.0; no clamp"
)

_NS_PER_S = 1_000_000_000
_MAX_EXACT_FLOAT_INT = 2**53
_INTEGER_RE = re.compile(r"[+-]?[0-9]+")


class SafeDeviationInputError(ValueError):
    """Вход safe deviation неоднозначен или повреждён; молча продолжать нельзя."""


def _require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SafeDeviationInputError(f"{name} is missing required columns: {missing}")


def _vehicle_keys(values: pd.Series, name: str) -> list[str]:
    """`tr_id` → канонический строковый ключ (как `CanonicalPoint.tr_id`)."""
    keys = []
    for value in values.tolist():
        valid = isinstance(value, int | np.integer | str) and not isinstance(value, bool | np.bool_)
        if not valid or (isinstance(value, str) and not value):
            raise SafeDeviationInputError(f"{name} must contain non-empty strings or integers")
        keys.append(str(value))
    return keys


def _naive_ns(values: pd.Series, name: str, *, allow_missing: bool) -> np.ndarray:
    """Разобрать timestamps в naive `datetime64[ns]`.

    Строки — только ISO-8601 (без угадывания day-first/month-first); числа
    (неизвестная единица) и timezone-aware значения — ошибка.
    """
    numeric = pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
    if numeric and values.notna().any():
        raise SafeDeviationInputError(f"{name} must contain datetimes, not numbers")
    try:
        parsed = pd.to_datetime(values, errors="raise", format="ISO8601")
    except (ValueError, TypeError, OverflowError) as exc:
        raise SafeDeviationInputError(f"{name} contains unparseable timestamps") from exc
    if getattr(parsed.dtype, "tz", None) is not None:
        raise SafeDeviationInputError(f"{name} must be timezone-naive (official UTC wall-clock)")
    try:
        ns = parsed.astype("datetime64[ns]").to_numpy()
    except (ValueError, TypeError, OverflowError) as exc:
        raise SafeDeviationInputError(f"{name} contains unparseable timestamps") from exc
    if not allow_missing and np.isnat(ns).any():
        raise SafeDeviationInputError(f"{name} contains missing timestamps")
    return ns


def _numeric_action_id(value: Any) -> int:
    """Числовое значение `tt_action_item_id` для tie-break; иначе явная ошибка."""
    if not isinstance(value, bool | np.bool_):
        if isinstance(value, int | np.integer):
            return int(value)
        if (
            isinstance(value, float | np.floating)
            and math.isfinite(value)
            and float(value).is_integer()
            and abs(value) <= _MAX_EXACT_FLOAT_INT
        ):
            return int(value)
        if isinstance(value, str) and _INTEGER_RE.fullmatch(value):
            return int(value)
    raise SafeDeviationInputError(
        f"tt_action_item_id {value!r} cannot be interpreted as an integer for the "
        "latest-fact tie-break"
    )


def safe_current_deviation_seconds(points: pd.DataFrame, schedule_facts: pd.DataFrame) -> pd.Series:
    """Point-in-time-safe текущее отклонение (секунды) для каждой prediction point.

    `points`: колонки `sample_id` (уникальный, без пропусков), `tr_id`, `T`
    (naive UTC); остальные колонки игнорируются. `schedule_facts`: колонки
    `SCHEDULE_FACT_COLUMNS` в исходном порядке строк (см. модуль). Для
    каждого `tr_id` точки должна быть хотя бы одна строка schedule, иначе
    вход считается несогласованным.

    Возвращает `float64` Series с индексом `sample_id` в порядке `points`;
    передаётся в `offline_context(..., current_deviation_seconds=...)`.
    """
    _require_columns(points, POINT_KEY_COLUMNS, "points")
    _require_columns(schedule_facts, SCHEDULE_FACT_COLUMNS, "schedule_facts")

    sample_ids = points["sample_id"]
    if sample_ids.isna().any():
        raise SafeDeviationInputError("points.sample_id contains missing values")
    if sample_ids.duplicated().any():
        raise SafeDeviationInputError("points.sample_id must be unique")
    point_vehicles = _vehicle_keys(points["tr_id"], "points.tr_id")
    point_t = _naive_ns(points["T"], "points.T", allow_missing=False).view("int64")

    order = schedule_facts.index
    if not (order.is_unique and order.is_monotonic_increasing):
        raise SafeDeviationInputError(
            "schedule_facts rows must stay in source (CSV) order: "
            "the index must be unique and increasing"
        )
    fact_vehicles = np.asarray(_vehicle_keys(schedule_facts["tr_id"], "schedule_facts.tr_id"))
    plan_ns = _naive_ns(
        schedule_facts["time_begin"], "schedule_facts.time_begin", allow_missing=False
    ).view("int64")
    fact_dt = _naive_ns(
        schedule_facts["time_fact_begin"], "schedule_facts.time_fact_begin", allow_missing=True
    )
    confirmed = ~np.isnat(fact_dt)
    fact_ns = fact_dt.view("int64")
    action_ids = schedule_facts["tt_action_item_id"].tolist()

    absent = sorted(set(point_vehicles) - set(fact_vehicles.tolist()))
    if absent:
        raise SafeDeviationInputError(
            f"schedule_facts has no rows for tr_id(s) of prediction points: {absent[:5]}"
        )
    # Подтверждённые события каждого ТС: позиции строк, отсортированные по
    # (time_fact_begin, исходная позиция) — ничья по факту остаётся в порядке CSV.
    events: dict[str, np.ndarray] = {}
    for vehicle in set(point_vehicles):
        rows = np.flatnonzero((fact_vehicles == vehicle) & confirmed)
        events[vehicle] = rows[np.lexsort((rows, fact_ns[rows]))]

    deviations = np.empty(len(point_vehicles), dtype=float)
    for i, (vehicle, t_ns) in enumerate(zip(point_vehicles, point_t.tolist(), strict=True)):
        rows = events[vehicle]
        sorted_facts = fact_ns[rows]
        n_eligible = int(np.searchsorted(sorted_facts, t_ns, side="right"))
        if n_eligible == 0:
            deviations[i] = 0.0
            continue
        latest = sorted_facts[n_eligible - 1]
        tied = rows[int(np.searchsorted(sorted_facts, latest, side="left")) : n_eligible]
        if len(tied) == 1:
            chosen = int(tied[0])
        else:
            numeric = [_numeric_action_id(action_ids[r]) for r in tied]
            best = max(numeric)
            chosen = int([r for r, n in zip(tied, numeric, strict=True) if n == best][-1])
        if fact_ns[chosen] > t_ns:  # инвариант leakage; не должен срабатывать
            raise RuntimeError("safe deviation selected a factual event after T")
        deviations[i] = (int(fact_ns[chosen]) - int(plan_ns[chosen])) / _NS_PER_S

    return pd.Series(
        deviations,
        index=pd.Index(sample_ids, name="sample_id"),
        name=SAFE_DEVIATION_NAME,
        dtype=float,
    )


def load_train_schedule_facts(root: str | Path) -> pd.DataFrame:
    """Factual schedule TRAIN (training-only) для `safe_current_deviation_seconds`.

    Читается только `<root>/train/schedule.csv`, только `SCHEDULE_FACT_COLUMNS`
    (`usecols`), без сортировки: индекс — `RangeIndex` в исходном порядке строк
    CSV. Времена разбираются тем же правилом, что и в helper'е (ISO-8601,
    naive, `datetime64[ns]`; `time_begin` обязателен, `time_fact_begin` может
    отсутствовать). Нельзя передавать в Feature Builder и runtime-контекст.
    """
    path = Path(root) / TRAIN_SCHEDULE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Official dataset file not found: {TRAIN_SCHEDULE_FILE}")
    available = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in SCHEDULE_FACT_COLUMNS if c not in available]
    if missing:
        raise OfficialDataSchemaError(Path(TRAIN_SCHEDULE_FILE), missing, available)
    facts = pd.read_csv(path, usecols=list(SCHEDULE_FACT_COLUMNS))[list(SCHEDULE_FACT_COLUMNS)]
    facts.index = pd.RangeIndex(len(facts))
    facts["time_begin"] = _naive_ns(
        facts["time_begin"], "train schedule time_begin", allow_missing=False
    )
    facts["time_fact_begin"] = _naive_ns(
        facts["time_fact_begin"], "train schedule time_fact_begin", allow_missing=True
    )
    return facts


def schedule_facts_fingerprint(schedule_facts: pd.DataFrame) -> str:
    """SHA-256 фактового представления, которое потребляет safe deviation.

    Хешируются только `SCHEDULE_FACT_COLUMNS` в порядке строк кадра (порядок CSV
    от `load_train_schedule_facts`); остальные колонки не влияют. ID — как
    строки, времена — naive `datetime64[ns]` в целых наносекундах (пропуск —
    пустое поле), поэтому строковое и разобранное представления одного
    содержимого дают один отпечаток.
    """
    _require_columns(schedule_facts, SCHEDULE_FACT_COLUMNS, "schedule_facts")
    columns = {
        "tr_id": [str(v) for v in schedule_facts["tr_id"].tolist()],
        "tt_action_item_id": [str(v) for v in schedule_facts["tt_action_item_id"].tolist()],
    }
    for name in ("time_begin", "time_fact_begin"):
        values = _naive_ns(schedule_facts[name], f"schedule_facts.{name}", allow_missing=True)
        missing = np.isnat(values)
        columns[name] = [
            "" if gap else str(ns)
            for gap, ns in zip(missing, values.view("int64").tolist(), strict=True)
        ]
    digest = hashlib.sha256((",".join(SCHEDULE_FACT_COLUMNS) + "\n").encode("utf-8"))
    for row in zip(*(columns[c] for c in SCHEDULE_FACT_COLUMNS), strict=True):
        digest.update((",".join(row) + "\n").encode("utf-8"))
    return digest.hexdigest()
