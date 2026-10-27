"""Planted imbalance harness and linear truth harness with fixed mathematical specifications.

Pre-registered benchmarks for validating imbalanced regression treatments
and stacking behavior against known, planted data-generating processes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PlantedTruth(dict):
    """Container for ground truth planted metadata."""

    def __init__(
        self,
        burst_mask: np.ndarray,
        learnable: bool,
        min_y: float,
        y_clean: np.ndarray,
    ) -> None:
        super().__init__(
            burst_mask=burst_mask,
            learnable=learnable,
            min_y=min_y,
            y_clean=y_clean,
        )
        self.burst_mask = burst_mask
        self.learnable = learnable
        self.min_y = min_y
        self.y_clean = y_clean


def make_planted_imbalance(
    seed: int = 42,
    learnable: bool = True,
    n: int = 6000,
    d: int = 12,
) -> tuple[np.ndarray, np.ndarray, PlantedTruth]:
    """Generate planted imbalance regression dataset.

    Specification (FIXED, constants must never be altered):
    - X ~ N(0,1)^(n x d)
    - y = 1.0*X1 + 0.5*X2 + 0.3*N(0,1)
    - Burst set B:
        learnable=True:  B = (X3 > 1.4) & (X5 > 0) (about 3.5% of rows)
        learnable=False: B = a random 3.5% of rows independent of X
    - For rows in B:
        y += 3.0 + 1.5*X4 (learnable)
        y += 3.0 + 1.5*N(0,1) (unlearnable)
    - y = y - min(y)
    Returns: X, y, truth.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    eps = rng.standard_normal(n)
    y = 1.0 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * eps
    y_clean = 1.0 * X[:, 0] + 0.5 * X[:, 1]

    if learnable:
        # B = (X3 > 1.4) & (X5 > 0) [0-indexed: col 2 and col 4]
        B = (X[:, 2] > 1.4) & (X[:, 4] > 0)
        y[B] += 3.0 + 1.5 * X[B, 3]  # [0-indexed: col 3]
        y_clean[B] += 3.0 + 1.5 * X[B, 3]
    else:
        # random 3.5% of rows independent of X
        n_burst = round(0.035 * n)
        burst_idx = rng.choice(n, size=n_burst, replace=False)
        B = np.zeros(n, dtype=bool)
        B[burst_idx] = True
        noise_burst = rng.standard_normal(n_burst)
        y[B] += 3.0 + 1.5 * noise_burst
        y_clean[B] += 3.0

    min_y = float(np.min(y))
    y = y - min_y
    y_clean = y_clean - min_y

    truth = PlantedTruth(
        burst_mask=B,
        learnable=learnable,
        min_y=min_y,
        y_clean=y_clean,
    )
    return X, y, truth


def make_linear_truth(
    seed: int = 42,
    n: int = 4000,
    d: int = 10,
) -> pd.DataFrame:
    """Generate linear ground-truth dataset for stacking validation.

    Specification (FIXED, constants must never be altered):
    - X ~ N(0,1)^(n x d)
    - y = X @ beta + 0.5 * N(0,1)
    - beta = [2, -1.5, 1, 0.5, 0, 0, 0, 0, 0, 0]
    - rows ordered into 20 equal "month" blocks for temporal folds.
    Returns: pd.DataFrame with features x_0..x_{d-1}, revenue, y, month, row_id, series_id.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta = np.zeros(d, dtype=float)
    fixed_beta = [2.0, -1.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    beta[: min(d, len(fixed_beta))] = fixed_beta[: min(d, len(fixed_beta))]

    eps = rng.standard_normal(n)
    y = X @ beta + 0.5 * eps

    n_blocks = 20
    rows_per_block = n // n_blocks
    month_names = [f"m_{i:02d}" for i in range(n_blocks)]
    months = np.repeat(month_names, rows_per_block)
    if len(months) < n:
        remainder = n - len(months)
        months = np.concatenate([months, np.full(remainder, month_names[-1])])

    col_names = [f"x_{i}" for i in range(d)]
    df = pd.DataFrame(X, columns=col_names)
    df["revenue"] = y
    df["y"] = y
    df["month"] = months
    df["row_id"] = np.arange(n)
    df["series_id"] = np.arange(n)
    df.attrs["beta"] = beta.tolist()
    df.attrs["seed"] = seed
    return df
