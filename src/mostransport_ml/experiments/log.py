"""Append-only JSONL experiment log.

Deliberately not MLflow: a single, dependency-free writer that's enough for
a 48-hour hackathon. Only `model_name` and `validation_mae` are required —
everything else can be filled in incrementally as inputs become known.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _validate_and_normalize_mae(value: float | None) -> float | None:
    """`None` (not yet known) passes through; any other value must be a
    finite number. Rejects NaN/+Infinity/-Infinity, which are not valid JSON
    and would make the experiment log unreliable to read back."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"validation_mae must be numeric or None, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"validation_mae must be finite (no NaN/Infinity), got {numeric!r}")
    return numeric


@dataclass(frozen=True)
class ExperimentRecord:
    model_name: str
    validation_mae: float | None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    data_version: str | None = None
    target_version: str | None = None
    feature_version: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "validation_mae", _validate_and_normalize_mae(self.validation_mae))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentLogger:
    """Appends `ExperimentRecord`s to a JSONL file, one record per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: ExperimentRecord) -> None:
        # allow_nan=False is a defensive backstop: ExperimentRecord already
        # rejects non-finite validation_mae, but this keeps the JSONL output
        # strictly valid JSON even if a non-finite value ever reached here
        # through some other field.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")

    def log_run(
        self,
        *,
        model_name: str,
        validation_mae: float | None,
        data_version: str | None = None,
        target_version: str | None = None,
        feature_version: str | None = None,
        params: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> ExperimentRecord:
        """Convenience constructor + log in one call. Returns the record logged."""
        record = ExperimentRecord(
            model_name=model_name,
            validation_mae=validation_mae,
            data_version=data_version,
            target_version=target_version,
            feature_version=feature_version,
            params=params or {},
            notes=notes,
        )
        self.log(record)
        return record
