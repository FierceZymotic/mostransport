"""Единственная зафиксированная конфигурация CatBoost для M1.

Никакого HPO, early stopping или eval_set: модель обучается ровно
`iterations` итераций и затем применяется. Оценочные данные сюда не
передаются до окончания fit.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CATBOOST_V1_PARAMS: MappingProxyType[str, Any] = MappingProxyType(
    {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
    }
)


def make_regressor() -> CatBoostRegressor:
    return CatBoostRegressor(**CATBOOST_V1_PARAMS)


def fit_predict(
    X_train: pd.DataFrame,
    y_train: Any,
    X_eval: pd.DataFrame,
    sample_weight: Any | None = None,
) -> np.ndarray:
    """Обучить на train (без eval_set) и вернуть сырой выход модели на X_eval."""
    if list(X_train.columns) != list(X_eval.columns):
        raise ValueError("X_train and X_eval must have identical ordered feature columns")
    model = make_regressor()
    model.fit(X_train, np.asarray(y_train, dtype=float), sample_weight=sample_weight)
    return np.asarray(model.predict(X_eval), dtype=float)
