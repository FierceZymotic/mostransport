"""Dataset manifest: reproducibility metadata, never raw row data.

A `DatasetManifest` records what a dataset *is* so an experiment can point at
it and be reproduced later. Each source file carries its own schema/row
count/time range — files are not assumed to share a schema, since organizers
may ship several CSVs with different shapes (e.g. telemetry.csv alongside
schedule.csv). The manifest intentionally never stores actual row contents.
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
    """Reproducibility metadata for a single source file.

    Schema, row count, and time range describe THIS file only — they are
    never assumed to apply to other files in the same dataset version.
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
    """Reproducibility metadata for a dataset version.

    Dataset-level: a version `label` and a creation timestamp. Everything
    schema-shaped (columns, dtypes, row count, time range) lives per-file on
    `source_files`, since a dataset version can bundle files with different
    schemas.
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
    """Build a `DatasetManifest` from one or more CSV files.

    Each file gets its own schema/row-count/time-range metadata — files are
    not required to share a schema. Only metadata is retained persistently
    (size, hash, schema, optional row count / time range); the manifest never
    stores row data.

    Parameters
    ----------
    paths:
        One CSV path, or a list of CSV paths belonging to the same dataset
        version. They may have different schemas.
    label:
        A short human-assigned version label (e.g. "organizer-2026-09-23").
    count_rows:
        Whether to load each file fully to compute an exact row count.
        Set False to skip this for very large files.
    time_column:
        Optional column name to additionally record the min/max range of,
        applied to whichever files actually contain it. Never inferred —
        must be explicitly supplied by the caller. Raises if none of the
        given files contain it at all.
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
