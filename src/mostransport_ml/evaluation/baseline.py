"""Простой константный baseline: предсказывает медиану target на train.

Даёт честный baseline MAE в первый час после появления target, задолго
до того, как появится реальный feature engineering или модель.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MedianBaselineRegressor:
    """Предсказывает медиану `y`, увиденную во время `fit`, полностью игнорируя `X`."""

    def __init__(self) -> None:
        self.median_: float | None = None

    def fit(self, X: Any, y: Any) -> MedianBaselineRegressor:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.size == 0:
            raise ValueError("Cannot fit MedianBaselineRegressor on empty y")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains NaN or infinite values")
        self.median_ = float(np.median(y_arr))
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("MedianBaselineRegressor must be fit before predict")
        return np.full(shape=len(X), fill_value=self.median_, dtype=float)
