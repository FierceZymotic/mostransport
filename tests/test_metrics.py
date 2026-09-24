import numpy as np
import pytest

from mostransport_ml.evaluation.metrics import mae


def test_mae_basic():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 5.0]
    assert mae(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_mae_perfect_prediction_is_zero():
    y = [1.0, 2.0, 3.0]
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0, 2.0, 3.0])


def test_mae_empty_raises():
    with pytest.raises(ValueError):
        mae([], [])


def test_mae_nan_in_y_true_raises():
    with pytest.raises(ValueError):
        mae([1.0, np.nan], [1.0, 2.0])


def test_mae_inf_in_y_pred_raises():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0, np.inf])
