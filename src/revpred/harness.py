"""Planted imbalance harness with fixed mathematical specification.

Pre-registered benchmark for validating imbalanced regression treatments
against known, planted data-generating processes.
"""
from __future__ import annotations

import numpy as np


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
