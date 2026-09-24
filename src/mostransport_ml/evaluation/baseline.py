"""Simple constant baseline: predict the median training-set target value.

Gives an honest baseline MAE within the first hour after the target exists,
well before any real feature engineering or modeling is in place.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MedianBaselineRegressor:
    """Predicts the median of `y` seen during `fit`, ignoring `X` entirely."""

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
