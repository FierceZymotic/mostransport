"""Тесты DIRECT/RESIDUAL формулировок target (чистая математика, без CatBoost)."""

from __future__ import annotations

import numpy as np
import pytest

from mostransport_ml.target.formulation import (
    FORMULATIONS,
    TARGET_SPEC,
    final_prediction,
    training_target,
)

TARGET = np.array([120.0, -30.0, 0.0])
CUR_DEV = np.array([100.0, 10.0, -5.0])
OUTPUT = np.array([7.0, -2.0, 1.5])


def test_formulations_are_exactly_direct_and_residual():
    assert FORMULATIONS == ("direct", "residual")
    assert TARGET_SPEC.name == "target_delay_s"
    assert TARGET_SPEC.unit == "seconds"


def test_direct_formulation_is_identity():
    np.testing.assert_array_equal(training_target(TARGET, CUR_DEV, "direct"), TARGET)
    np.testing.assert_array_equal(final_prediction(OUTPUT, CUR_DEV, "direct"), OUTPUT)


def test_residual_formulation():
    np.testing.assert_array_equal(training_target(TARGET, CUR_DEV, "residual"), TARGET - CUR_DEV)
    np.testing.assert_array_equal(final_prediction(OUTPUT, CUR_DEV, "residual"), CUR_DEV + OUTPUT)


def test_residual_round_trip_recovers_target():
    residual = training_target(TARGET, CUR_DEV, "residual")
    np.testing.assert_allclose(final_prediction(residual, CUR_DEV, "residual"), TARGET)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="formulation"):
        training_target(TARGET, CUR_DEV, "clipped")
    with pytest.raises(ValueError, match="NaN"):
        training_target(np.array([np.nan]), np.array([0.0]), "residual")
    with pytest.raises(ValueError, match="shape"):
        final_prediction(OUTPUT, CUR_DEV[:2], "residual")
