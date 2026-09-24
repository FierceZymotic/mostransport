"""Strict temporal (chronological) train/validation/test split.

No random splitting anywhere in this module — for early-delay prediction,
getting the chronology right matters more than almost anything else in the
evaluation setup. Purge/embargo windows and walk-forward CV are deliberately
out of scope until the real target/horizon are known (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_time_boundaries(
    dataframe: pd.DataFrame,
    time_column: str,
    train_end: Any,
    validation_end: Any,
) -> TemporalSplit:
    """Split `dataframe` chronologically into train/validation/test.

    - train: time_column < train_end
    - validation: train_end <= time_column < validation_end
    - test: time_column >= validation_end

    Parameters
    ----------
    dataframe:
        Input data. Does not need to be pre-sorted.
    time_column:
        Name of the column to split on. Must be present and parseable as a
        datetime (or already be one).
    train_end, validation_end:
        Boundary timestamps (anything `pandas.Timestamp` accepts). Must
        satisfy train_end < validation_end.

    Raises
    ------
    KeyError
        If `time_column` is not present in `dataframe`.
    ValueError
        If boundaries are invalid or ordered wrong, or timezone-awareness is
        inconsistent between the boundaries and/or the time column.
    """
    if time_column not in dataframe.columns:
        raise KeyError(
            f"time_column {time_column!r} not found in columns: {list(dataframe.columns)}"
        )

    times = pd.to_datetime(dataframe[time_column], errors="raise")

    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)

    if (train_end_ts.tzinfo is None) != (validation_end_ts.tzinfo is None):
        raise ValueError(
            "train_end and validation_end must both be timezone-aware or both "
            f"timezone-naive, got {train_end_ts!r} and {validation_end_ts!r}"
        )
    if train_end_ts >= validation_end_ts:
        raise ValueError(
            f"train_end ({train_end_ts}) must be strictly before "
            f"validation_end ({validation_end_ts})"
        )

    column_has_tz = times.dt.tz is not None
    boundaries_have_tz = train_end_ts.tzinfo is not None

    if column_has_tz != boundaries_have_tz:
        raise ValueError(
            f"Timezone mismatch: column {time_column!r} is "
            f"{'timezone-aware' if column_has_tz else 'timezone-naive'}, but the supplied "
            f"boundaries are {'timezone-aware' if boundaries_have_tz else 'timezone-naive'}. "
            "Make both aware or both naive before splitting."
        )

    train_mask = times < train_end_ts
    validation_mask = (times >= train_end_ts) & (times < validation_end_ts)
    test_mask = times >= validation_end_ts

    train = dataframe.loc[train_mask]
    validation = dataframe.loc[validation_mask]
    test = dataframe.loc[test_mask]

    n_total = len(dataframe)
    n_split = len(train) + len(validation) + len(test)
    if n_split != n_total:
        raise ValueError(
            f"Split lost or duplicated rows: {n_total} input rows, {n_split} across "
            "train/validation/test. This should not happen — please report it."
        )

    return TemporalSplit(train=train, validation=validation, test=test)
