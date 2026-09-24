import numpy as np
import pytest

from mostransport_ml.evaluation.baseline import MedianBaselineRegressor


def test_median_baseline_predicts_constant_median():
    model = MedianBaselineRegressor()
    model.fit(X=list(range(5)), y=[1.0, 2.0, 3.0, 4.0, 100.0])
    assert model.median_ == pytest.approx(3.0)

    predictions = model.predict(list(range(3)))
    assert predictions.shape == (3,)
    assert np.all(predictions == pytest.approx(3.0))


def test_median_baseline_requires_fit_before_predict():
    model = MedianBaselineRegressor()
    with pytest.raises(RuntimeError):
        model.predict([1, 2, 3])


def test_median_baseline_rejects_empty_y():
    model = MedianBaselineRegressor()
    with pytest.raises(ValueError):
        model.fit(X=[], y=[])


def test_median_baseline_rejects_nan_y():
    model = MedianBaselineRegressor()
    with pytest.raises(ValueError):
        model.fit(X=[1, 2], y=[1.0, np.nan])
