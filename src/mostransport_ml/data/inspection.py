"""Generic, schema-agnostic инспекция CSV/dataframe.

Этот модуль ничего не знает о transport-specific именах колонок. Он
рассчитан на то, чтобы быть первым, что запускается на свежеполученном
organizer CSV — до какой-либо canonicalization или работы над target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ColumnProfile:
    """Профиль одной колонки."""

    name: str
    dtype: str
    missing_count: int
    missing_ratio: float
    unique_count: int | None
    numeric_summary: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "missing_count": self.missing_count,
            "missing_ratio": self.missing_ratio,
            "unique_count": self.unique_count,
            "numeric_summary": self.numeric_summary,
        }


@dataclass(frozen=True)
class InspectionReport:
    """Результат профилирования CSV-файла или dataframe."""

    file: Path
    rows_inspected: int
    n_columns: int
    columns: list[ColumnProfile]
    duplicate_row_count: int
    time_column: str | None = None
    time_range: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": str(self.file),
            "rows_inspected": self.rows_inspected,
            "n_columns": self.n_columns,
            "duplicate_row_count": self.duplicate_row_count,
            "time_column": self.time_column,
            "time_range": list(self.time_range) if self.time_range else None,
            "columns": [c.to_dict() for c in self.columns],
        }


def inspect_csv(
    path: str | Path,
    *,
    nrows: int | None = None,
    time_column: str | None = None,
    unique_count_max_rows: int = 200_000,
) -> InspectionReport:
    """Профилировать CSV-файл, не предполагая никакой конкретной схемы.

    Параметры
    ---------
    path:
        Путь к CSV-файлу.
    nrows:
        Опциональное ограничение на число читаемых строк (передаётся в
        ``pandas.read_csv``). Полезно для быстрого первого взгляда на
        большой файл.
    time_column:
        Опциональное имя колонки, для которой дополнительно вывести
        min/max диапазон. Никогда не выводится автоматически — должно
        быть явно задано вызывающим, поскольку реальное имя колонки
        времени пока не известно.
    unique_count_max_rows:
        Пропустить (потенциально дорогой) подсчёт уникальных значений
        колонки, если прочитано больше строк, чем этот порог — чтобы
        инспекция в первый час оставалась дешёвой на больших файлах.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path, nrows=nrows)
    return inspect_dataframe(
        df,
        file=file_path,
        time_column=time_column,
        unique_count_max_rows=unique_count_max_rows,
    )


def inspect_dataframe(
    df: pd.DataFrame,
    *,
    file: str | Path | None = None,
    time_column: str | None = None,
    unique_count_max_rows: int = 200_000,
) -> InspectionReport:
    """Профилировать уже загруженный dataframe. Общая логика для `inspect_csv` и тестов."""
    n_rows = len(df)
    compute_unique = n_rows <= unique_count_max_rows

    columns: list[ColumnProfile] = []
    for name in df.columns:
        series = df[name]
        missing_count = int(series.isna().sum())
        missing_ratio = float(missing_count / n_rows) if n_rows else 0.0
        unique_count = int(series.nunique(dropna=True)) if compute_unique else None

        numeric_summary: dict[str, float] | None = None
        if pd.api.types.is_numeric_dtype(series):
            described = series.describe()
            numeric_summary = {
                stat: float(described[stat])
                for stat in ("min", "max", "mean", "std")
                if stat in described.index and pd.notna(described[stat])
            }

        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                unique_count=unique_count,
                numeric_summary=numeric_summary,
            )
        )

    duplicate_row_count = int(df.duplicated().sum())

    time_range: tuple[str, str] | None = None
    if time_column is not None:
        if time_column not in df.columns:
            raise KeyError(f"time_column {time_column!r} not found in columns: {list(df.columns)}")
        parsed = pd.to_datetime(df[time_column], errors="raise", utc=False)
        if parsed.notna().any():
            time_range = (str(parsed.min()), str(parsed.max()))

    return InspectionReport(
        file=Path(file) if file is not None else Path("<in-memory>"),
        rows_inspected=n_rows,
        n_columns=len(df.columns),
        columns=columns,
        duplicate_row_count=duplicate_row_count,
        time_column=time_column,
        time_range=time_range,
    )
