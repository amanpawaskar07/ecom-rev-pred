"""Tests for Phase 3 planted imbalance harness."""
from __future__ import annotations

import numpy as np

from revpred.harness import make_planted_imbalance


def test_harness_dimensions_and_properties() -> None:
    """Test output shapes, types, and minimum value guarantee."""
    X, y, truth = make_planted_imbalance(seed=42, learnable=True, n=6000, d=12)

    assert X.shape == (6000, 12)
    assert y.shape == (6000,)
    assert np.all(y >= 0.0)
    assert np.isclose(np.min(y), 0.0)
    assert truth.learnable is True
    assert len(truth.burst_mask) == 6000


def test_harness_burst_fractions() -> None:
    """Burst set B is ~3.5% of rows for both learnable and unlearnable cases."""
    _, _, truth_learn = make_planted_imbalance(seed=42, learnable=True, n=6000, d=12)
    frac_learn = np.mean(truth_learn.burst_mask)
    assert 0.025 <= frac_learn <= 0.055, f"Learnable burst frac {frac_learn} not ~3.5%"

    _, _, truth_unlearn = make_planted_imbalance(seed=42, learnable=False, n=6000, d=12)
    frac_unlearn = np.mean(truth_unlearn.burst_mask)
    assert np.isclose(frac_unlearn, 0.035), f"Unlearnable burst frac {frac_unlearn} != 0.035"


def test_harness_determinism() -> None:
    """Fixed seed gives identical outputs."""
    X1, y1, _ = make_planted_imbalance(seed=123, learnable=True)
    X2, y2, _ = make_planted_imbalance(seed=123, learnable=True)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
