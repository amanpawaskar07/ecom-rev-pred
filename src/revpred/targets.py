"""Target transformation classes for regression under severe skew and point masses at zero.

Target transforms in cost order:
  1. log1p: Default rival (np.log1p / np.expm1, clipped at 0).
  2. quantile: QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, n_train),
     subsample=None) fit on training target only; inverse clipped at 0.

Point Mass at Zero:
  Zero revenue represents an exact point mass (structural zero / no sales).
  In QuantileTransformer, all zero values share identical empirical CDF values.
  When mapped to a normal output distribution, ties at zero map to the lower-tail normal quantile
  (e.g., -5.199 for typical sample sizes). During inverse transformation, any prediction at or below
  this value maps back to 0.0, preserving the zero point mass within float precision (< 1e-9).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.preprocessing import QuantileTransformer


class BaseTargetTransform(ABC):
    """Abstract base class for target transformations."""

    @abstractmethod
    def fit(self, y: Any) -> BaseTargetTransform:
        """Fit transformation state on training targets only."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, y: Any) -> np.ndarray:
        """Transform target values to transformed scale."""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, y_trans: Any) -> np.ndarray:
        """Inverse transform values back to original revenue scale, clipped >= 0."""
        raise NotImplementedError

    def inverse_transform(self, y_trans: Any) -> np.ndarray:
        """Alias for inverse() to maintain compatibility with scikit-learn API."""
        return self.inverse(y_trans)

    def fit_transform(self, y: Any) -> np.ndarray:
        """Fit and transform target."""
        return self.fit(y).transform(y)


class Log1pTargetTransform(BaseTargetTransform):
    """Standard log1p transformation: log(1 + y) with expm1 inverse clipped >= 0."""

    def __init__(self) -> None:
        self.is_fitted_: bool = False

    def fit(self, y: Any) -> Log1pTargetTransform:
        y_arr = np.asarray(y, dtype=float)
        if (y_arr < 0).any():
            raise ValueError("Revenue targets cannot be negative for Log1pTargetTransform.")
        self.is_fitted_ = True
        return self

    def transform(self, y: Any) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        return np.log1p(np.maximum(y_arr, 0.0))

    def inverse(self, y_trans: Any) -> np.ndarray:
        p_arr = np.asarray(y_trans, dtype=float)
        return np.clip(np.expm1(p_arr), 0.0, None)


class QuantileTargetTransform(BaseTargetTransform):
    """Quantile transformation mapping targets to a standard normal distribution.

    Uses QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, n_train),
    subsample=None) fit strictly on training target values.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.transformer_: QuantileTransformer | None = None
        self.n_train_: int | None = None
        self.zero_val_transformed_: float | None = None

    def fit(self, y: Any) -> QuantileTargetTransform:
        y_arr = np.asarray(y, dtype=float).ravel()
        if (y_arr < 0).any():
            raise ValueError("Revenue targets cannot be negative for QuantileTargetTransform.")

        n_train = len(y_arr)
        if n_train == 0:
            raise ValueError("Cannot fit QuantileTargetTransform on empty target array.")

        self.n_train_ = n_train
        n_quantiles = min(1000, n_train)

        self.transformer_ = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=None,
            random_state=self.random_state,
        )
        self.transformer_.fit(y_arr.reshape(-1, 1))

        # Record mapped value of exact zero
        z_trans = self.transformer_.transform(np.array([[0.0]]))[0, 0]
        self.zero_val_transformed_ = float(z_trans)

        return self

    def transform(self, y: Any) -> np.ndarray:
        if self.transformer_ is None:
            raise RuntimeError("QuantileTargetTransform must be fitted before transform.")
        y_arr = np.asarray(y, dtype=float).ravel()
        y_2d = np.maximum(y_arr, 0.0).reshape(-1, 1)
        return self.transformer_.transform(y_2d).ravel()

    def inverse(self, y_trans: Any) -> np.ndarray:
        if self.transformer_ is None:
            raise RuntimeError("QuantileTargetTransform must be fitted before inverse.")
        p_arr = np.asarray(y_trans, dtype=float).ravel().reshape(-1, 1)
        raw_inv = self.transformer_.inverse_transform(p_arr).ravel()
        # Clip at 0.0 and ensure point mass maps back exactly to 0 within float precision
        clipped = np.clip(raw_inv, 0.0, None)
        if self.zero_val_transformed_ is not None:
            clipped[p_arr.ravel() <= self.zero_val_transformed_] = 0.0
        return clipped


def get_target_transform(name: str, seed: int = 42) -> BaseTargetTransform:
    """Factory to retrieve target transform instance by name."""
    name_clean = name.strip().lower()
    if name_clean in ("log1p", "t0", "rival"):
        return Log1pTargetTransform()
    elif name_clean in ("quantile", "t1", "quant"):
        return QuantileTargetTransform(random_state=seed)
    else:
        raise ValueError(f"Unknown target transform name: '{name}'. Expected 'log1p' or 'quantile'.")
