"""Validate inference и детерминированная запись submission `sample_id;prediction`.

Путь: official validate (points + traffic + schedule_plan) → offline adapter →
CanonicalBatch → тот же ArtifactPredictor, что в serving → submission в
порядке `sample_submission.csv`. Target validate не существует и не строится;
test schedule (с фактом) здесь не читается.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mostransport_ml.data.official import (
    SUBMISSION_COLUMNS,
    load_sample_submission_ids,
    load_validate_inputs,
)
from mostransport_ml.features.adapters import offline_context
from mostransport_ml.inference.predictor import ArtifactPredictor


class SubmissionError(ValueError):
    """Submission не может быть собран корректно."""


@dataclass(frozen=True)
class SubmissionSummary:
    n_rows: int
    model_version: str
    bundle_sha256: str
    output_name: str


def build_submission(
    required_ids: Sequence[str], point_ids: Sequence[str], predictions: Sequence[float]
) -> pd.DataFrame:
    """Ровно одна строка на каждый `required_ids` в его порядке, конечные значения."""
    required = list(required_ids)
    points = list(point_ids)
    values = np.asarray(predictions, dtype=float).reshape(-1)
    if not required:
        raise SubmissionError("required sample_id list is empty")
    if len(set(required)) != len(required):
        raise SubmissionError("required sample_id list has duplicates")
    if len(set(points)) != len(points):
        raise SubmissionError("predicted sample_id list has duplicates")
    if len(points) != len(values):
        raise SubmissionError("number of predictions differs from number of points")
    missing, extra = set(required) - set(points), set(points) - set(required)
    if missing or extra:
        raise SubmissionError(
            f"sample_id mismatch: {len(missing)} missing, {len(extra)} unexpected"
        )
    if not np.all(np.isfinite(values)):
        raise SubmissionError("predictions must be finite")
    by_id = dict(zip(points, values.tolist(), strict=True))
    return pd.DataFrame(
        {"sample_id": required, "prediction": [by_id[i] for i in required]},
        columns=list(SUBMISSION_COLUMNS),
    )


def write_submission(frame: pd.DataFrame, path: str | Path) -> None:
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        raise SubmissionError(f"columns must be exactly {SUBMISSION_COLUMNS}")
    if frame["sample_id"].duplicated().any() or frame["sample_id"].isna().any():
        raise SubmissionError("sample_id must be present and unique")
    if not all(math.isfinite(v) for v in frame["prediction"].tolist()):
        raise SubmissionError("predictions must be finite")
    frame.to_csv(path, sep=";", index=False, lineterminator="\n")


def generate_validate_submission(
    dataset_root: str | Path, bundle_dir: str | Path, output_path: str | Path
) -> SubmissionSummary:
    root = Path(dataset_root).resolve()
    output = Path(output_path).resolve()
    if output.is_relative_to(root):
        raise SubmissionError("output must not be written inside the official dataset")
    predictor = ArtifactPredictor.load(bundle_dir)
    inputs = load_validate_inputs(root)
    required = load_sample_submission_ids(root)
    batch = offline_context(inputs.points, inputs.telemetry, inputs.schedule_plan)
    predictions = predictor.predict(batch)
    frame = build_submission(required, [p.point_id for p in batch.points], predictions)
    write_submission(frame, output)
    return SubmissionSummary(
        n_rows=len(frame),
        model_version=predictor.model_version(),
        bundle_sha256=predictor.bundle_sha256,
        output_name=output.name,
    )
