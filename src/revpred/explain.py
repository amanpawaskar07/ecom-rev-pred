"""SHAP explanation engine for FullRidgeRegressor and linear models."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def groups_for(preprocessor: ColumnTransformer) -> list[str]:
    """Map every transformed column to ONE original feature from ColumnTransformer structure.

    Derived from the ColumnTransformer structure (not by parsing column names).
    Asserts that:
    1. Each transformed column is in exactly one group.
    2. Each numeric feature has exactly one column.
    """
    groups: list[str] = []
    numeric_features: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if transformer in ("drop", None) or (name == "remainder" and transformer == "drop"):
            continue

        if transformer == "passthrough":
            for col in columns:
                groups.append(str(col))
                numeric_features.append(str(col))
            continue

        # Check for Pipeline
        actual_transformer = transformer
        if isinstance(transformer, Pipeline):
            for _, step in transformer.steps:
                if hasattr(step, "categories_"):
                    actual_transformer = step
                    break
            else:
                # 1-to-1 pipeline (e.g. log1p + StandardScaler)
                for col in columns:
                    groups.append(str(col))
                    numeric_features.append(str(col))
                continue

        # Check for OneHotEncoder
        if hasattr(actual_transformer, "categories_"):
            for i, col in enumerate(columns):
                n_cats = len(actual_transformer.categories_[i])
                drop_idx = getattr(actual_transformer, "drop_idx_", None)
                if drop_idx is not None and drop_idx[i] is not None:
                    n_cats -= 1
                for _ in range(n_cats):
                    groups.append(str(col))
        else:
            # 1-to-1 transformer (e.g. StandardScaler)
            for col in columns:
                groups.append(str(col))
                numeric_features.append(str(col))

    # Assertions
    for num_feat in numeric_features:
        count = groups.count(num_feat)
        assert count == 1, f"Numeric feature '{num_feat}' has {count} columns, expected exactly 1."

    return groups


def check_group_coverage(preprocessor: ColumnTransformer, groups: Sequence[str]) -> bool:
    """Verify that groups cover every transformed column with no drops, gaps, or invalid entries."""
    try:
        expected_groups = groups_for(preprocessor)
        if len(groups) != len(expected_groups):
            return False
        return list(groups) == expected_groups
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def linear_shap(
    Z: np.ndarray,
    coef: np.ndarray,
    intercept: float | np.ndarray,
    z_bg_mean: np.ndarray,
    groups: Sequence[str] | dict[str, list[int]],
) -> tuple[np.ndarray, float]:
    """Exact SHAP values for a linear model on transformed features, aggregated to original groups.

    phi_j = coef_j * (z_j - mean_background z_j)
    base_value = intercept + coef . mean_background z

    Parameters:
        Z: Transformed feature matrix [n x p]
        coef: Linear model coefficients [p]
        intercept: Model intercept scalar
        z_bg_mean: Background mean transformed feature vector [p]
        groups: Mapping from transformed columns to original features

    Returns:
        phi: Group-aggregated SHAP matrix [n x n_groups]
        base_value: Scalar base value
    """
    Z_arr = np.asarray(Z, dtype=float)
    coef_arr = np.asarray(coef, dtype=float).ravel()
    z_bg_arr = np.asarray(z_bg_mean, dtype=float).ravel()
    inter_val = float(np.asarray(intercept).item())

    if Z_arr.shape[1] != coef_arr.shape[0]:
        raise ValueError(f"Z columns ({Z_arr.shape[1]}) != coef size ({coef_arr.shape[0]})")
    if z_bg_arr.shape[0] != coef_arr.shape[0]:
        raise ValueError(f"z_bg_mean size ({z_bg_arr.shape[0]}) != coef size ({coef_arr.shape[0]})")

    base_value = float(inter_val + np.dot(coef_arr, z_bg_arr))
    phi_trans = coef_arr * (Z_arr - z_bg_arr)

    if isinstance(groups, dict):
        group_names = list(groups.keys())
        group_to_indices = groups
    else:
        group_names = list(dict.fromkeys(groups))
        group_to_indices = {g: [i for i, grp in enumerate(groups) if grp == g] for g in group_names}

    phi = np.zeros((Z_arr.shape[0], len(group_names)), dtype=float)
    for idx, g in enumerate(group_names):
        cols = group_to_indices[g]
        phi[:, idx] = np.sum(phi_trans[:, cols], axis=1)

    return phi, base_value


def explain_rows(
    model: Any,
    rows_df: pd.DataFrame,
    background_df: pd.DataFrame,
) -> tuple[np.ndarray, float, list[str]]:
    """Explain rows for a fitted FullRidgeRegressor using background_df.

    Returns:
        phi: [n x n_groups]
        base_value: float
        group_names: list of unique group names
    """
    if not hasattr(model, "preprocessor_") or not hasattr(model, "best_estimator_"):
        raise ValueError("model must be a fitted FullRidgeRegressor with preprocessor_ and best_estimator_.")

    Z = model.preprocessor_.transform(rows_df)
    Z_bg = model.preprocessor_.transform(background_df)
    z_bg_mean = np.mean(Z_bg, axis=0)

    coef = model.best_estimator_.coef_
    intercept = model.best_estimator_.intercept_
    groups = groups_for(model.preprocessor_)
    group_names = list(dict.fromkeys(groups))

    phi, base_value = linear_shap(Z, coef, intercept, z_bg_mean, groups)
    return phi, base_value, group_names


def permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_log_val: np.ndarray | pd.Series | Sequence[float],
    n_repeats: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Compute permutation importance in ORIGINAL feature space using RMSLE as metric."""
    y_log_arr = np.asarray(y_log_val, dtype=float)
    pred_base_log = model.predict(X_val)
    base_rmsle = float(np.sqrt(np.mean((np.maximum(pred_base_log, 0.0) - np.maximum(y_log_arr, 0.0)) ** 2)))

    groups = groups_for(model.preprocessor_)
    feature_names = list(dict.fromkeys(groups))

    rng = np.random.default_rng(seed)
    scores: dict[str, float] = {}

    for feat in feature_names:
        diffs: list[float] = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            perm_idx = rng.permutation(len(X_val))
            X_perm[feat] = X_val[feat].iloc[perm_idx].values
            pred_perm_log = model.predict(X_perm)
            perm_rmsle = float(
                np.sqrt(np.mean((np.maximum(pred_perm_log, 0.0) - np.maximum(y_log_arr, 0.0)) ** 2))
            )
            diffs.append(perm_rmsle - base_rmsle)
        scores[feat] = float(np.mean(diffs))

    return scores


def explain_row(
    model: Any,
    payload_row: pd.DataFrame | pd.Series | dict[str, Any],
    background_stats: dict[str, Any],
    top_k: int | None = None,
) -> dict[str, Any]:
    """Single-row explanation using only background_stats = {feature_names, z_mean, base_value}."""
    if isinstance(payload_row, dict):
        df_row = pd.DataFrame([payload_row])
    elif isinstance(payload_row, pd.Series):
        df_row = pd.DataFrame([payload_row.to_dict()])
    else:
        df_row = payload_row.copy()

    Z = model.preprocessor_.transform(df_row)
    z_mean = np.asarray(background_stats["z_mean"], dtype=float)
    base_value = float(background_stats["base_value"])
    coef = model.best_estimator_.coef_
    intercept = model.best_estimator_.intercept_

    groups = background_stats.get("group_map")
    if groups is None:
        groups = groups_for(model.preprocessor_)

    group_names = background_stats.get("feature_names")
    if group_names is None:
        group_names = list(dict.fromkeys(groups))

    phi, _ = linear_shap(Z, coef, intercept, z_mean, groups)
    phi_row = phi[0]
    prediction_log = float(base_value + np.sum(phi_row))

    attributions = {g: float(val) for g, val in zip(group_names, phi_row)}
    sorted_features = sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True)

    if top_k is not None:
        top_features = sorted_features[:top_k]
    else:
        top_features = sorted_features

    return {
        "base_value": base_value,
        "prediction_log": prediction_log,
        "prediction_rev": float(np.clip(np.expm1(prediction_log), 0.0, None)),
        "attributions": attributions,
        "top_features": top_features,
        "group_names": group_names,
        "phi": phi_row,
    }


def check_background_disjoint(val_row_ids: Sequence[Any], bg_row_ids: Sequence[Any]) -> bool:
    """Check that validation row IDs and background row IDs are strictly disjoint."""
    set_val = set(val_row_ids)
    set_bg = set(bg_row_ids)
    return len(set_val.intersection(set_bg)) == 0


def check_model_matches_served(model: Any, expected_coef_sha: str) -> bool:
    """Verify that model coefficients match the served model SHA256."""
    if not hasattr(model, "best_estimator_") or not hasattr(model.best_estimator_, "coef_"):
        return False
    actual_sha = hashlib.sha256(np.ascontiguousarray(model.best_estimator_.coef_).tobytes()).hexdigest()
    return actual_sha == expected_coef_sha


def check_additivity(
    phi: np.ndarray,
    base_value: float,
    preds_log: np.ndarray,
    max_abs_err: float = 1e-3,
) -> bool:
    """Check that sum of SHAP attributions plus base_value equals predictions in log1p space."""
    err = np.max(np.abs(np.sum(phi, axis=1) + base_value - preds_log))
    return bool(err <= max_abs_err)
