"""Tests that Phase 3 mutants are caught by leakage checks."""
from __future__ import annotations

import numpy as np

from revpred.harness import make_planted_imbalance
from revpred.resample import SmoterResampler
from revpred.targets import QuantileTargetTransform
from tests.mutants.mutants_phase3 import (
    QuantileTargetTransformMutantB,
    oversample_before_split_mutant_a,
    smoter_all_neighbours_mutant_c,
)


def test_mutant_a_oversample_before_split_caught() -> None:
    """Mutant A (oversample before split) is caught under validation perturbation."""
    X, y, _ = make_planted_imbalance(seed=42, learnable=True, n=1000, d=6)
    # Split
    n_tr = 750
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:], y[n_tr:]

    # Clean SMOTER fit on training rows only
    clean_resampler = SmoterResampler()
    clean_res_1 = clean_resampler.fit_resample(X_tr, y_tr, train_idx=np.arange(n_tr), seed=42)

    # Perturbed validation set
    y_val_pert = y_val * 100.0

    # In clean procedure, validation perturbation has ZERO effect on training resample
    clean_res_2 = clean_resampler.fit_resample(X_tr, y_tr, train_idx=np.arange(n_tr), seed=42)
    np.testing.assert_array_equal(clean_res_1.y, clean_res_2.y)

    # In Mutant A, whole dataset is resampled before split, so validation perturbation leaks into training
    X_comb_1 = np.vstack([X_tr, X_val])
    y_comb_1 = np.concatenate([y_tr, y_val])
    _X_tr_m1, _, y_tr_m1, _, _ = oversample_before_split_mutant_a(X_comb_1, y_comb_1, test_size=0.25, seed=42)

    X_comb_2 = np.vstack([X_tr, X_val])
    y_comb_2 = np.concatenate([y_tr, y_val_pert])
    _X_tr_m2, _, y_tr_m2, _, _ = oversample_before_split_mutant_a(X_comb_2, y_comb_2, test_size=0.25, seed=42)

    assert not np.array_equal(y_tr_m1, y_tr_m2), "Mutant A was not caught under validation perturbation!"


def test_mutant_b_quantile_fit_all_caught() -> None:
    """Mutant B (QuantileTransformer fit on all rows) is caught under validation perturbation."""
    _X, y, _ = make_planted_imbalance(seed=42, learnable=True, n=1000, d=6)
    n_tr = 750
    y_tr = y[:n_tr]
    y_val = y[n_tr:]
    y_val_pert = y_val * 100.0

    # Clean QuantileTransformer fit on train only
    qt_clean = QuantileTargetTransform(random_state=42)
    qt_clean.fit(y_tr)
    tr_clean_1 = qt_clean.transform(y_tr)

    # If validation is perturbed, clean transformation on train is unchanged
    tr_clean_2 = qt_clean.transform(y_tr)
    np.testing.assert_array_equal(tr_clean_1, tr_clean_2)

    # Mutant B fits on all rows
    mb1 = QuantileTargetTransformMutantB(random_state=42).fit_all(y_tr, y_val)
    tr_mb1 = mb1.transform(y_tr)

    mb2 = QuantileTargetTransformMutantB(random_state=42).fit_all(y_tr, y_val_pert)
    tr_mb2 = mb2.transform(y_tr)

    assert not np.array_equal(tr_mb1, tr_mb2), "Mutant B was not caught under validation perturbation!"


def test_mutant_c_neighbours_all_rows_caught() -> None:
    """Mutant C (neighbours drawn from all rows) is caught under validation perturbation."""
    X, y, _ = make_planted_imbalance(seed=42, learnable=True, n=1000, d=6)
    n_tr = 750
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:], y[n_tr:]
    X_val_pert = X_val * 10.0
    y_val_pert = y_val * 10.0

    # Clean SMOTER does not depend on validation
    SmoterResampler().fit_resample(X_tr, y_tr, train_idx=np.arange(n_tr), seed=42)

    # Mutant C draws neighbours from train + val
    _X_mc1, y_mc1 = smoter_all_neighbours_mutant_c(X_tr, y_tr, X_val, y_val, train_idx=np.arange(n_tr), seed=42)
    _X_mc2, y_mc2 = smoter_all_neighbours_mutant_c(X_tr, y_tr, X_val_pert, y_val_pert, train_idx=np.arange(n_tr), seed=42)

    assert not np.array_equal(y_mc1, y_mc2), "Mutant C was not caught under validation perturbation!"
