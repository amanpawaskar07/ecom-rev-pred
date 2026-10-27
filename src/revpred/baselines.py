"""Baseline models: mean, persistence, and ridge regression with CV evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from revpred.config import get_thresholds
from revpred.evaluate import evaluate_oof, run_cv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


def fit_predict_mean(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """Predict expm1 of training fold mean of log1p revenue."""
    log_y = np.log1p(train_df["revenue"].to_numpy(dtype=float))
    mean_val = float(np.expm1(np.mean(log_y)))
    return np.full(len(val_df), mean_val, dtype=float)


def fit_predict_persistence(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """Predict using lag_rev_1 (realized revenue of the previous month)."""
    return np.clip(val_df["lag_rev_1"].to_numpy(dtype=float), 0.0, None)


def fit_predict_ridge(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """Fit Ridge regression on base features with log1p transforms, StandardScaler, and TimeSeriesSplit(3)."""
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])

    log_features = [
        "lag_rev_1",
        "lag_rev_2",
        "lag_rev_3",
        "roll_mean_rev_3",
        "lag_lines_1",
        "lag_lines_2",
        "lag_lines_3",
        "trailing_price_3",
    ]
    num_features = ["active_months_before", "months_since_first_sale", "month_of_year"]
    cat_features = ["region", "category"]

    # Ensure train rows are sorted chronologically by month
    train_sorted = train_df.sort_values("month").reset_index(drop=True)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "log",
                Pipeline([
                    ("log1p", FunctionTransformer(np.log1p, validate=False)),
                    ("scaler", StandardScaler()),
                ]),
                log_features,
            ),
            ("num", StandardScaler(), num_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(train_sorted)
    y_train_log = np.log1p(train_sorted["revenue"].to_numpy(dtype=float))

    X_val = preprocessor.transform(val_df)

    cv_inner = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        estimator=Ridge(random_state=seed),
        param_grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
        cv=cv_inner,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X_train, y_train_log)

    preds_log = grid.best_estimator_.predict(X_val)
    return np.clip(np.expm1(preds_log), 0.0, None)


def run_all_baselines(
    panel_dev: pd.DataFrame,
    reports_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run cross-validation for all three baselines, save OOFs and baselines_cv.json."""
    rep_path = Path(reports_dir) if reports_dir else REPORTS_DIR
    oof_dir = rep_path / "oof"

    baselines_spec = {
        "baseline_mean": fit_predict_mean,
        "baseline_persistence": fit_predict_persistence,
        "baseline_ridge": fit_predict_ridge,
    }

    results: dict[str, Any] = {}

    for name, fn in baselines_spec.items():
        oof_df = run_cv(fit_predict_fn=fn, panel_dev=panel_dev, name=name, oof_dir=oof_dir)
        metrics = evaluate_oof(oof_df)
        results[name] = metrics

    rep_path.mkdir(parents=True, exist_ok=True)
    with open(rep_path / "baselines_cv.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results