"""Tests for Phase 3 target transformations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gates import lib as gl
from revpred.targets import (
    Log1pTargetTransform,
    QuantileTargetTransform,
    get_target_transform,
)


@pytest.fixture
def panel_dev_targets() -> np.ndarray:
    """Load dev revenue targets."""
    df = pd.read_parquet("data/processed/panel_dev.parquet")
    return df["revenue"].to_numpy(dtype=float)


def test_quantile_roundtrip_positive_rows(panel_dev_targets: np.ndarray) -> None:
    """Quantile round trip on train rows with revenue > 0: max relative error <= threshold."""
    y = panel_dev_targets
    pos = y > 0
    assert np.any(pos), "No positive revenue rows found."

    qt = QuantileTargetTransform(random_state=42)
    qt.fit(y)
    y_trans = qt.transform(y)
    y_inv = qt.inverse(y_trans)

    # Relative error on positive rows
    rel_errors = np.abs(y[pos] - y_inv[pos]) / y[pos]
    max_rel_err = float(np.max(rel_errors))
    max_allowed = float(gl.thr("phase3.roundtrip_max_rel_err"))

    assert max_rel_err <= max_allowed, f"Max relative error {max_rel_err} > {max_allowed}"


def test_quantile_zero_point_mass(panel_dev_targets: np.ndarray) -> None:
    """The zero point mass maps back to 0 within 1e-9."""
    y = panel_dev_targets
    zero_mask = y == 0.0
    assert np.any(zero_mask), "No zero revenue rows found."

    qt = QuantileTargetTransform(random_state=42)
    qt.fit(y)
    y_trans = qt.transform(y)
    y_inv = qt.inverse(y_trans)

    max_zero_err = float(np.max(np.abs(y_inv[zero_mask])))
    assert max_zero_err <= 1e-9, f"Zero point mass max error {max_zero_err} > 1e-9"


def test_quantile_inverse_monotonic_grid(panel_dev_targets: np.ndarray) -> None:
    """Inverse is non-decreasing on a grid across the transformed domain."""
    y = panel_dev_targets
    qt = QuantileTargetTransform(random_state=42)
    qt.fit(y)
    y_trans = qt.transform(y)

    grid = np.linspace(y_trans.min() - 2.0, y_trans.max() + 2.0, 5000)
    inv_grid = qt.inverse(grid)

    diffs = np.diff(inv_grid)
    assert not np.any(diffs < -1e-12), f"Inverse is not non-decreasing: min diff = {np.min(diffs)}"


def test_quantile_inverse_non_negative(panel_dev_targets: np.ndarray) -> None:
    """Inverse transformation is strictly >= 0."""
    y = panel_dev_targets
    qt = QuantileTargetTransform(random_state=42)
    qt.fit(y)

    extreme_inputs = np.array([-1000.0, -10.0, -5.0, 0.0, 5.0, 10.0, 1000.0])
    inv_vals = qt.inverse(extreme_inputs)
    assert np.all(inv_vals >= 0.0), f"Inverse returned negative values: {inv_vals[inv_vals < 0]}"


def test_log1p_properties(panel_dev_targets: np.ndarray) -> None:
    """Test Log1pTargetTransform round trip and non-negativity."""
    y = panel_dev_targets
    t = Log1pTargetTransform()
    t.fit(y)
    y_trans = t.transform(y)
    y_inv = t.inverse(y_trans)

    pos = y > 0
    rel_errors = np.abs(y[pos] - y_inv[pos]) / y[pos]
    assert np.max(rel_errors) < 1e-12
    assert np.all(y_inv >= 0.0)
    assert np.all(y_inv[y == 0.0] == 0.0)


def test_factory_get_target_transform() -> None:
    """Test get_target_transform factory."""
    t0 = get_target_transform("log1p")
    assert isinstance(t0, Log1pTargetTransform)
    t1 = get_target_transform("quantile")
    assert isinstance(t1, QuantileTargetTransform)

    with pytest.raises(ValueError, match="Unknown target transform"):
        get_target_transform("unsupported_transform")


def test_target_transform_negative_raises() -> None:
    """Negative values during fit raise ValueError."""
    bad_y = np.array([10.0, -5.0, 20.0])
    with pytest.raises(ValueError):
        Log1pTargetTransform().fit(bad_y)

    with pytest.raises(ValueError):
        QuantileTargetTransform().fit(bad_y)
