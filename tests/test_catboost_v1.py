"""Тесты фиксированной CatBoost-конфигурации M1 на крошечных синтетических данных."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mostransport_ml.models.catboost_v1 import CATBOOST_V1_PARAMS, fit_predict, make_regressor


def tiny_dataset(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=80), "x2": rng.normal(size=80)})
    X.loc[::7, "x2"] = np.nan
    y = 3.0 * X["x1"].to_numpy() + rng.normal(scale=0.1, size=80)
    return X.iloc[:60], y[:60], X.iloc[60:]


def test_config_is_the_single_fixed_m1_config():
    assert dict(CATBOOST_V1_PARAMS) == {
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
    with pytest.raises(TypeError):
        CATBOOST_V1_PARAMS["depth"] = 8  # type: ignore[index]
    params = make_regressor().get_params()
    assert params["iterations"] == 500 and params["random_seed"] == 42


def test_fit_predict_is_deterministic():
    X_train, y_train, X_eval = tiny_dataset()
    weights = np.linspace(0.5, 1.5, len(y_train))
    first = fit_predict(X_train, y_train, X_eval, sample_weight=weights)
    second = fit_predict(X_train, y_train, X_eval, sample_weight=weights)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)
    assert first.shape == (len(X_eval),)
    assert np.all(np.isfinite(first))


def test_feature_columns_must_match():
    X_train, y_train, X_eval = tiny_dataset()
    with pytest.raises(ValueError, match="identical ordered feature columns"):
        fit_predict(X_train, y_train, X_eval[["x2", "x1"]])
