"""Model specifications, fixed hyperparameter factories, and ablation wrappers."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

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


class FullRidgeRegressor(BaseEstimator, RegressorMixin):
    """Full Ridge regressor replicating Phase 1 baseline Ridge.

    Transforms log_features with log1p and StandardScaler, num_features with StandardScaler,
    and cat_features with OneHotEncoder(handle_unknown='ignore', sparse_output=False).
    Fits Ridge on y_log with alpha selected from [0.1, 1.0, 10.0, 100.0] via TimeSeriesSplit(3).
    Rows are sorted by month inside fit. Unweighted.
    """

    LOG_FEATURES: ClassVar[list[str]] = [
        "lag_rev_1",
        "lag_rev_2",
        "lag_rev_3",
        "roll_mean_rev_3",
        "lag_lines_1",
        "lag_lines_2",
        "lag_lines_3",
        "trailing_price_3",
    ]
    NUM_FEATURES: ClassVar[list[str]] = [
        "active_months_before",
        "months_since_first_sale",
        "month_of_year",
    ]
    CAT_FEATURES: ClassVar[list[str]] = ["region", "category"]

    def __init__(
        self,
        alphas: Sequence[float] | None = None,
        alpha: float | None = None,
        cv_splits: int = 3,
        random_state: int | None = None,
    ):
        self.alpha = alpha
        if alpha is not None and alphas is None:
            self.alphas = (alpha,)
        elif alphas is not None:
            self.alphas = alphas
        else:
            self.alphas = (0.1, 1.0, 10.0, 100.0)
        self.cv_splits = cv_splits
        self.random_state = random_state

    def fit(self, df: pd.DataFrame, y_log: np.ndarray | pd.Series | Sequence[float]) -> FullRidgeRegressor:
        """Fit preprocessor and Ridge with TimeSeriesSplit tuning on y_log."""
        thr = get_thresholds()
        seed = int(self.random_state if self.random_state is not None else thr["global"]["seed"])

        y_arr = np.asarray(y_log, dtype=float)

        if "month" in df.columns:
            df_temp = df.copy()
            df_temp["_y_target"] = y_arr
            train_sorted = df_temp.sort_values("month").reset_index(drop=True)
            y_train_log = train_sorted["_y_target"].to_numpy(dtype=float)
            df_sorted = train_sorted.drop(columns=["_y_target"])
        else:
            df_sorted = df.copy().reset_index(drop=True)
            y_train_log = y_arr

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "log",
                    Pipeline([
                        ("log1p", FunctionTransformer(np.log1p, validate=False)),
                        ("scaler", StandardScaler()),
                    ]),
                    self.LOG_FEATURES,
                ),
                ("num", StandardScaler(), self.NUM_FEATURES),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.CAT_FEATURES,
                ),
            ]
        )

        X_train = preprocessor.fit_transform(df_sorted)

        if self.alpha is not None:
            best_ridge = Ridge(alpha=self.alpha, random_state=seed)
            best_ridge.fit(X_train, y_train_log)
            self.preprocessor_ = preprocessor
            self.best_estimator_ = best_ridge
            return self

        cv_inner = TimeSeriesSplit(n_splits=self.cv_splits)
        grid = GridSearchCV(
            estimator=Ridge(random_state=seed),
            param_grid={"alpha": list(self.alphas)},
            cv=cv_inner,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X_train, y_train_log)

        self.preprocessor_ = preprocessor
        self.grid_ = grid
        self.best_estimator_ = grid.best_estimator_
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict log-scale predictions on df."""
        if not hasattr(self, "best_estimator_"):
            raise RuntimeError("FullRidgeRegressor must be fitted before calling predict.")
        X_val = self.preprocessor_.transform(df)
        preds_log = self.best_estimator_.predict(X_val)
        return np.asarray(preds_log, dtype=float)
