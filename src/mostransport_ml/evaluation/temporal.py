"""Строгий temporal (хронологический) train/validation/test split.

Никакого random split нигде в этом модуле — для раннего прогнозирования
задержек правильная хронология важнее почти всего остального в схеме
оценки. Purge/embargo-окна и walk-forward CV намеренно вне scope, пока не
станут известны реальные target/horizon (см. docs/ARCHITECTURE.md).
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
    """Разбить `dataframe` хронологически на train/validation/test.

    - train: time_column < train_end
    - validation: train_end <= time_column < validation_end
    - test: time_column >= validation_end

    Параметры
    ---------
    dataframe:
        Входные данные. Не обязаны быть заранее отсортированы.
    time_column:
        Имя колонки, по которой делается split. Обязана присутствовать и
        парситься как datetime (либо уже быть им).
    train_end, validation_end:
        Границы-timestamp'ы (подходит всё, что принимает
        `pandas.Timestamp`). Обязано выполняться train_end < validation_end.

    Исключения
    ----------
    KeyError
        Если `time_column` отсутствует в `dataframe`.
    ValueError
        Если границы невалидны или заданы в неверном порядке, либо
        timezone-awareness несогласована между границами и/или колонкой
        времени.
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
