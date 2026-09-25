"""Тесты структурных train-групп, режимов обучения и фиксированной сетки M1."""

from __future__ import annotations

import numpy as np
import pytest

from mostransport_ml.models.regimes import (
    GROUP_A,
    GROUP_B,
    M1_EXPERIMENTS,
    TRAIN_REGIMES,
    assign_train_groups,
    select_training_rows,
)


def test_groups_are_structural_not_an_id_range():
    # 9000003 входит в real-набор → группа A, несмотря на "synthetic-looking" ID;
    # 5 отсутствует в real-наборе → группа B, несмотря на "обычный" ID.
    tr_ids = np.array([10, 9000003, 5, 10, 9000020])
    groups = assign_train_groups(tr_ids, {10, 9000003})
    assert groups.tolist() == [GROUP_A, GROUP_A, GROUP_B, GROUP_A, GROUP_B]


def test_empty_real_set_is_rejected():
    with pytest.raises(ValueError):
        assign_train_groups([1, 2], set())


GROUPS = np.array([GROUP_A, GROUP_B, GROUP_B, GROUP_A, GROUP_B, GROUP_B, GROUP_B, GROUP_B])


def test_real_only_selects_group_a_without_weights():
    selection = select_training_rows(GROUPS, "real_only")
    assert selection.mask.tolist() == (GROUPS == GROUP_A).tolist()
    assert selection.sample_weight is None
    assert (selection.n_group_a, selection.n_group_b) == (2, 0)


def test_all_train_uses_every_row_uniformly():
    selection = select_training_rows(GROUPS, "all_train")
    assert selection.mask.all()
    assert selection.sample_weight is None
    assert (selection.n_group_a, selection.n_group_b) == (2, 6)


def test_balanced_weights_equalise_group_totals():
    selection = select_training_rows(GROUPS, "balanced_group_weight")
    weights = selection.sample_weight
    assert selection.mask.all()
    assert selection.group_b_weight == pytest.approx(2 / 6)
    assert np.all(weights[GROUPS == GROUP_A] == 1.0)
    assert weights[GROUPS == GROUP_A].sum() == pytest.approx(weights[GROUPS == GROUP_B].sum())


def test_unknown_regime_or_group_is_rejected():
    with pytest.raises(ValueError):
        select_training_rows(GROUPS, "weighted_0.3")
    with pytest.raises(ValueError):
        select_training_rows(np.array(["synthetic"]), "all_train")


def test_m1_grid_is_exactly_six_predefined_experiments():
    assert TRAIN_REGIMES == ("real_only", "all_train", "balanced_group_weight")
    combos = [(e.train_regime, e.formulation) for e in M1_EXPERIMENTS]
    assert len(M1_EXPERIMENTS) == 6
    assert len(set(combos)) == 6
    assert set(combos) == {(r, f) for r in TRAIN_REGIMES for f in ("direct", "residual")}
    assert len({e.experiment_id for e in M1_EXPERIMENTS}) == 6
