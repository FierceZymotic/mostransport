"""Dataset manifest: метаданные воспроизводимости, никогда не сырые строки.

`DatasetManifest` фиксирует, чем *является* датасет, чтобы на него можно
было сослаться из эксперимента и позже воспроизвести. Каждый source-файл
несёт свою собственную схему/row count/time range — файлы не считаются
имеющими общую схему, поскольку организаторы могут выдать несколько CSV
разной формы (например, telemetry.csv рядом со schedule.csv). Manifest
намеренно никогда не хранит реальное содержимое строк.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceFile:
    """Метаданные воспроизводимости для одного source-файла.

    Схема, row count и time range описывают ИМЕННО ЭТОТ файл — никогда не
    считается, что они применимы к другим файлам той же версии датасета.
    """

    path: str
    size_bytes: int
    sha256: str
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    row_count: int | None
    time_column: str | None = None
    time_range: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "columns": list(self.columns),
            "dtypes": self.dtypes,
            "row_count": self.row_count,
            "time_column": self.time_column,
            "time_range": list(self.time_range) if self.time_range else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceFile:
        time_range = data.get("time_range")
        return cls(
            path=data["path"],
            size_bytes=data["size_bytes"],
            sha256=data["sha256"],
            columns=tuple(data["columns"]),
            dtypes=dict(data["dtypes"]),
            row_count=data.get("row_count"),
            time_column=data.get("time_column"),
            time_range=tuple(time_range) if time_range else None,
        )


@dataclass(frozen=True)
class DatasetManifest:
    """Метаданные воспроизводимости для версии датасета.

    На уровне датасета: `label` версии и время создания. Всё, что связано
    со схемой (колонки, dtypes, row count, time range), живёт по-файлово
    в `source_files`, поскольку версия датасета может объединять файлы с
    разными схемами.
    """

    label: str
    source_files: tuple[SourceFile, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source_files": [f.to_dict() for f in self.source_files],
            "created_at": self.created_at,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        source_files = tuple(SourceFile.from_dict(sf) for sf in data["source_files"])
        return cls(
            label=data["label"],
            source_files=source_files,
            created_at=data["created_at"],
        )

    @classmethod
    def from_json(cls, text: str) -> DatasetManifest:
        return cls.from_dict(json.loads(text))


def _build_source_file(
    path: Path,
    *,
    header_columns: tuple[str, ...],
    count_rows: bool,
    time_column: str | None,
) -> SourceFile:
    import pandas as pd

    row_count: int | None = None
    time_range: tuple[str, str] | None = None

    if count_rows or time_column is not None:
        full_df = pd.read_csv(path)
        dtypes = {str(c): str(full_df[c].dtype) for c in full_df.columns}
        if count_rows:
            row_count = len(full_df)
        if time_column is not None:
            parsed = pd.to_datetime(full_df[time_column], errors="raise", utc=False)
            if parsed.notna().any():
                time_range = (str(parsed.min()), str(parsed.max()))
    else:
        header_df = pd.read_csv(path, nrows=0)
        dtypes = {str(c): str(header_df[c].dtype) for c in header_df.columns}

    return SourceFile(
        path=str(path),
        size_bytes=path.stat().st_size,
        sha256=_sha256_of_file(path),
        columns=header_columns,
        dtypes=dtypes,
        row_count=row_count,
        time_column=time_column,
        time_range=time_range,
    )


def build_manifest(
    paths: str | Path | list[str | Path],
    *,
    label: str,
    count_rows: bool = True,
    time_column: str | None = None,
) -> DatasetManifest:
    """Построить `DatasetManifest` из одного или нескольких CSV-файлов.

    Каждый файл получает свои собственные метаданные схемы/row-count/
    time-range — файлы не обязаны иметь общую схему. Персистентно
    сохраняются только метаданные (размер, hash, схема, опциональный row
    count / time range); manifest никогда не хранит данные строк.

    Параметры
    ---------
    paths:
        Один путь к CSV либо список путей, относящихся к одной версии
        датасета. Они могут иметь разные схемы.
    label:
        Короткий человекочитаемый label версии (например,
        "organizer-2026-09-23").
    count_rows:
        Загружать ли каждый файл целиком, чтобы посчитать точный row
        count. Установите False, чтобы пропустить это для очень больших
        файлов.
    time_column:
        Опциональное имя колонки, для которой дополнительно записать
        min/max диапазон — применяется к тем файлам, где она реально
        есть. Никогда не выводится автоматически — должно быть явно
        задано вызывающим. Бросает исключение, если ни один из файлов её
        не содержит.
    """
    path_list = [Path(paths)] if isinstance(paths, str | Path) else [Path(p) for p in paths]
    for p in path_list:
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")

    import pandas as pd

    header_columns = {p: tuple(str(c) for c in pd.read_csv(p, nrows=0).columns) for p in path_list}

    if time_column is not None and not any(time_column in cols for cols in header_columns.values()):
        raise KeyError(
            f"time_column {time_column!r} not found in any source file: "
            f"{[str(p) for p in path_list]}"
        )

    source_files = tuple(
        _build_source_file(
            p,
            header_columns=header_columns[p],
            count_rows=count_rows,
            time_column=time_column if time_column in header_columns[p] else None,
        )
        for p in path_list
    )

    return DatasetManifest(
        label=label,
        source_files=source_files,
        created_at=datetime.now(UTC).isoformat(),
    )
