"""Evaluation metrics. MAE is the organizers' confirmed primary metric."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error, with explicit shape/finiteness validation.

    A thin wrapper over `sklearn.metrics.mean_absolute_error` that refuses to
    silently produce a number from mismatched shapes or non-finite values.

    Raises
    ------
    ValueError
        If shapes disagree, inputs are empty, or contain NaN/inf values.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true_arr.shape} vs {y_pred_arr.shape}"
        )
    if y_true_arr.size == 0:
        raise ValueError("y_true/y_pred must not be empty")
    if not np.all(np.isfinite(y_true_arr)):
        raise ValueError("y_true contains NaN or infinite values")
    if not np.all(np.isfinite(y_pred_arr)):
        raise ValueError("y_pred contains NaN or infinite values")

    return float(mean_absolute_error(y_true_arr, y_pred_arr))
