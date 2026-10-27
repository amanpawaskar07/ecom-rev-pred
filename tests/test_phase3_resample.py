"""Tests for Phase 3 SMOTER-style oversampling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gates import lib as gl
from revpred.resample import TrainGuard, fit_resample


@pytest.fixture
def synthetic_toy_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create toy dataset with known rare rows, categorical, and binary columns."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "num1": rng.standard_normal(n),
        "num2": rng.standard_normal(n) * 2.0 + 1.0,
        "bin1": rng.integers(0, 2, size=n),
        "cat1": rng.choice(["A", "B", "C"], size=n),
    })
    # Create target with top 10% rare rows (> 100.0)
    y = rng.uniform(1.0, 10.0, size=n)
    rare_idx = rng.choice(n, size=20, replace=False)
    y[rare_idx] = rng.uniform(100.0, 200.0, size=20)
    train_idx = np.arange(n)
    return df, y, train_idx


def test_smoter_interpolation_invariants(synthetic_toy_data: tuple[pd.DataFrame, np.ndarray, np.ndarray]) -> None:
    """Synthetic y equals lambda interpolation within tolerance, and numeric features lie on segment."""
    df, y, train_idx = synthetic_toy_data
    tol = float(gl.thr("phase3.smoter_interp_abs_tol"))

    res = fit_resample(df, y, train_idx=train_idx, seed=42)
    X_res, y_res, parent_ids, lambdas = res

    n_orig = len(df)
    n_syn = res.n_syn
    assert n_syn > 0, "No synthetic rows generated."
    assert len(parent_ids) == n_syn
    assert len(lambdas) == n_syn

    for i in range(n_syn):
        syn_idx = n_orig + i
        id_a, id_b = parent_ids[i]
        lam = lambdas[i]

        # Invariant 1: y_syn = y_a + lam * (y_b - y_a)
        expected_y = y[id_a] + lam * (y[id_b] - y[id_a])
        actual_y = y_res[syn_idx]
        assert abs(actual_y - expected_y) <= tol, f"y interp error {abs(actual_y - expected_y)} > {tol}"

        # Invariant 2: numeric features lie on segment between parents
        for num_col in ["num1", "num2"]:
            val_a = df.loc[id_a, num_col]
            val_b = df.loc[id_b, num_col]
            val_syn = X_res.loc[syn_idx, num_col]

            expected_num = val_a + lam * (val_b - val_a)
            assert abs(val_syn - expected_num) <= 1e-12
            min_bound = min(val_a, val_b) - 1e-12
            max_bound = max(val_a, val_b) + 1e-12
            assert min_bound <= val_syn <= max_bound

        # Invariant 3: categorical and binary features copied from parent a
        assert X_res.loc[syn_idx, "bin1"] == df.loc[id_a, "bin1"]
        assert X_res.loc[syn_idx, "cat1"] == df.loc[id_a, "cat1"]


def test_smoter_rare_share_after(synthetic_toy_data: tuple[pd.DataFrame, np.ndarray, np.ndarray]) -> None:
    """Rare share after resampling equals 30% +/- 1 row."""
    df, y, train_idx = synthetic_toy_data
    res = fit_resample(df, y, train_idx=train_idx, seed=42)

    tail_thr = res.tail_thr
    n_rare_after = np.sum(res.y >= tail_thr)
    n_total_after = len(res.y)
    expected_rare_count = 0.30 * n_total_after

    assert abs(n_rare_after - expected_rare_count) <= 1.0, (
        f"Rare count after {n_rare_after} differs from 30% of {n_total_after} ({expected_rare_count}) by > 1 row."
    )


def test_smoter_original_rows_first_and_unchanged(synthetic_toy_data: tuple[pd.DataFrame, np.ndarray, np.ndarray]) -> None:
    """Original rows come first in resampled output and are completely unchanged."""
    df, y, train_idx = synthetic_toy_data
    res = fit_resample(df, y, train_idx=train_idx, seed=42)

    n_orig = len(df)
    pd.testing.assert_frame_equal(res.X.iloc[:n_orig], df)
    np.testing.assert_array_equal(res.y[:n_orig], y)


def test_smoter_determinism(synthetic_toy_data: tuple[pd.DataFrame, np.ndarray, np.ndarray]) -> None:
    """Fixed seed produces bit-for-bit identical output."""
    df, y, train_idx = synthetic_toy_data
    res1 = fit_resample(df, y, train_idx=train_idx, seed=42)
    res2 = fit_resample(df, y, train_idx=train_idx, seed=42)

    pd.testing.assert_frame_equal(res1.X, res2.X)
    np.testing.assert_array_equal(res1.y, res2.y)
    assert res1.parent_ids == res2.parent_ids
    np.testing.assert_array_equal(res1.lambdas, res2.lambdas)


def test_smoter_train_guard_outside_index_raises(synthetic_toy_data: tuple[pd.DataFrame, np.ndarray, np.ndarray]) -> None:
    """TrainGuard asserts that train_idx is a subset of the allowed indices, raising if not."""
    df, y, train_idx = synthetic_toy_data
    allowed_indices = set(range(100))  # Only first 100 are allowed

    guard = TrainGuard(allowed_indices)
    with pytest.raises(ValueError, match="TrainGuard violation"):
        fit_resample(df, y, train_idx=train_idx, seed=42, guard=guard)


def test_smoter_fewer_than_6_rare_rows_skips() -> None:
    """Fewer than 6 rare rows in a fold -> returns unchanged and records skip."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({"x": rng.standard_normal(n)})
    # All zero revenue except 4 rows
    y = np.zeros(n)
    y[:4] = [10.0, 20.0, 30.0, 40.0]
    train_idx = np.arange(n)

    res = fit_resample(df, y, train_idx=train_idx, seed=42)
    assert res.skipped is True
    assert res.n_syn == 0
    assert len(res.parent_ids) == 0
    assert len(res.X) == n
