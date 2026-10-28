"""Planted-truth test harness for SHAP explanation verification."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_planted_linear(
    seed: int = 42,
    n: int = 5000,
    noise_std: float = 0.3,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate planted linear dataset with known feature effects and correlated features.

    Features:
        X1, X3, X4, X5, X6 ~ N(0, 1)
        X2 = 0.6 * X1 + 0.8 * N(0, 1)  (corr(X1, X2) = 0.6)
        C ~ Uniform({A, B, C, D, E}) with effect e:
            A: -0.8, B: -0.4, C: 0.0, D: 0.4, E: 0.8
        f = 2.0 * X1 - 1.0 * X2 + 0.5 * X3 + 0.0 * X4 + e(C)
        X5, X6, X4 carry no signal.
        y = f + noise_std * N(0, 1)

    True importance:
        |b_j| * std(x_j) for numeric features, std of e(C) for C.
    """
    rng = np.random.default_rng(seed)

    x1 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    x5 = rng.standard_normal(n)
    x6 = rng.standard_normal(n)
    x2 = 0.6 * x1 + 0.8 * rng.standard_normal(n)

    c_levels = np.array(["A", "B", "C", "D", "E"])
    c = rng.choice(c_levels, size=n)
    e_map = {"A": -0.8, "B": -0.4, "C": 0.0, "D": 0.4, "E": 0.8}
    e = np.array([e_map[val] for val in c])

    f = 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + 0.0 * x4 + e
    if noise_std > 0.0:
        y = f + noise_std * rng.standard_normal(n)
    else:
        y = f.copy()

    df = pd.DataFrame({
        "X1": x1,
        "X2": x2,
        "X3": x3,
        "X4": x4,
        "X5": x5,
        "X6": x6,
        "C": c,
        "f": f,
        "y": y,
    })

    true_importance = {
        "X1": 2.0 * float(np.std(x1)),
        "X2": 1.0 * float(np.std(x2)),
        "X3": 0.5 * float(np.std(x3)),
        "X4": 0.0,
        "X5": 0.0,
        "X6": 0.0,
        "C": float(np.std(e)),
    }

    return df, true_importance


def make_planted_interaction(
    seed: int = 42,
    n: int = 5000,
    noise_std: float = 0.3,
) -> pd.DataFrame:
    """Generate planted interaction dataset where y = 2.0 * X1 + 1.5 * X2 * X3 + noise.

    A linear ridge model cannot represent the X2 * X3 interaction; SHAP explains
    the linear model, so X2 and X3 receive near-zero attribution.
    """
    rng = np.random.default_rng(seed)

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    x5 = rng.standard_normal(n)
    x6 = rng.standard_normal(n)

    y = 2.0 * x1 + 1.5 * x2 * x3 + noise_std * rng.standard_normal(n)

    return pd.DataFrame({
        "X1": x1,
        "X2": x2,
        "X3": x3,
        "X4": x4,
        "X5": x5,
        "X6": x6,
        "y": y,
    })

