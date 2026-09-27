"""Зафиксированная production-конфигурация HistGradientBoosting H0.

Research закрыт: production-кандидат — `HistGradientBoostingRegressor`,
DIRECT `target_delay_s`, схема `runtime-safe-v1` (F0, 29 признаков),
обучение только на Group A. H0 — консервативная фиксированная конфигурация,
а не доказанный глобальный оптимум. Никакого HPO, early stopping или
eval_set. Модуль задаёт только конфиг и строгую training-границу; финальное
обучение на официальных данных — отдельный этап (P3).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mostransport_ml.features.schema import (
    RUNTIME_SAFE_FEATURE_NAMES,
    RUNTIME_SAFE_FEATURE_SCHEMA_VERSION,
)

HGB_H0_FEATURE_SCHEMA_VERSION = RUNTIME_SAFE_FEATURE_SCHEMA_VERSION
HGB_H0_TARGET_FORMULATION = "direct"

HGB_H0_PARAMS: MappingProxyType[str, Any] = MappingProxyType(
    {
        "loss": "absolute_error",
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
    }
)


def make_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**HGB_H0_PARAMS)


def _check_features(X: pd.DataFrame) -> None:
    """Ровно `runtime-safe-v1` в замороженном порядке; NaN можно, ±inf и нечисла — нет."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with runtime-safe-v1 columns")
    if tuple(X.columns) != RUNTIME_SAFE_FEATURE_NAMES:
        raise ValueError(
            "X columns must be exactly runtime-safe-v1 RUNTIME_SAFE_FEATURE_NAMES in order"
        )
    for name, dtype in X.dtypes.items():
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            raise ValueError(f"feature {name!r} must be numeric")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("X contains +/-infinity; only finite values or NaN are allowed")


def _check_target(y: Any, n_rows: int) -> np.ndarray:
    values = np.asarray(y)
    if values.dtype.kind not in "iuf":
        raise ValueError("y must be numeric")
    values = values.astype(float)
    if values.shape != (n_rows,):
        raise ValueError(f"y must be 1-D with {n_rows} values")
    if not np.all(np.isfinite(values)):
        raise ValueError("y must be finite")
    return values


def fit_h0(X: pd.DataFrame, y: Any) -> HistGradientBoostingRegressor:
    """Обучить H0 DIRECT на `runtime-safe-v1` через строгую training-границу."""
    _check_features(X)
    target = _check_target(y, len(X))
    return make_regressor().fit(X, target)
