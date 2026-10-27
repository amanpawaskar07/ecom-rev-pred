"""Model specifications, fixed hyperparameter factories, and ablation wrappers."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from revpred.config import get_project_config, get_thresholds


def make_xgb(params: dict[str, Any] | None = None) -> xgb.XGBRegressor:
    """Instantiate XGBoost regressor with fixed default hyperparameters from config.

    Objective: reg:squarederror. Fixed seed, n_jobs=1.
    """
    cfg = get_project_config()
    thr = get_thresholds()
    default_params = dict(cfg.get("xgb_default", {}))

    seed = int(thr["global"]["seed"])
    default_params.setdefault("random_state", seed)
    default_params.setdefault("n_jobs", 1)
    default_params.setdefault("objective", "reg:squarederror")

    if params:
        default_params.update(params)

    return xgb.XGBRegressor(**default_params)


def make_rf(params: dict[str, Any] | None = None) -> RandomForestRegressor:
    """Instantiate Random Forest regressor with fixed default hyperparameters from config.

    Fixed seed, n_jobs=1.
    """
    cfg = get_project_config()
    thr = get_thresholds()
    default_params = dict(cfg.get("rf_default", {}))

    seed = int(thr["global"]["seed"])
    default_params.setdefault("random_state", seed)
    default_params.setdefault("n_jobs", 1)

    if params:
        default_params.update(params)

    return RandomForestRegressor(**default_params)


def make_ridge(params: dict[str, Any] | None = None) -> Pipeline:
    """Instantiate Ridge regressor pipeline with StandardScaler and fixed hyperparameters."""
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])

    ridge_params = {"alpha": 1.0, "random_state": seed}
    if params:
        ridge_params.update(params)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(**ridge_params)),
    ])


class ModelSpec:
    """Model wrapper encapsulating name, feature column list, target transformation,
    and underlying estimator.
    """

    def __init__(
        self,
        name: str,
        features: list[str],
        target_transform: str | tuple[Callable[[Any], Any], Callable[[Any], Any]] = "log1p",
        estimator: Any = None,
    ):
        self.name = name
        self.features = list(features)
        self.target_transform = target_transform
        if estimator is None:
            self.estimator = make_xgb()
        else:
            self.estimator = estimator

        if self.target_transform == "log1p":
            self.transform_y: Callable[[Any], Any] = np.log1p
            self.inverse_y: Callable[[Any], Any] = lambda p: np.clip(np.expm1(p), 0.0, None)
        elif isinstance(self.target_transform, tuple):
            self.transform_y, self.inverse_y = self.target_transform
        elif callable(self.target_transform):
            self.transform_y = self.target_transform
            self.inverse_y = lambda p: p
        else:
            raise ValueError(f"Unsupported target_transform: {self.target_transform}")

        self.fitted_estimator_: Any = None

    def fit(self, train_df: pd.DataFrame) -> ModelSpec:
        """Fit estimator on train_df using specified features and transformed target."""
        X = train_df[self.features].to_numpy(dtype=float)
        y = self.transform_y(train_df["revenue"].to_numpy(dtype=float))
        self.fitted_estimator_ = clone(self.estimator)
        self.fitted_estimator_.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on df using specified features and inverted target transformation."""
        if self.fitted_estimator_ is None:
            raise RuntimeError("ModelSpec must be fit before predict can be called.")
        X = df[self.features].to_numpy(dtype=float)
        preds_raw = self.fitted_estimator_.predict(X)
        return self.inverse_y(np.asarray(preds_raw, dtype=float))

    def __call__(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
        self.fit(train_df)
        return self.predict(val_df)
