"""Temporal expanding-window stacking regressor."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from gates import lib as gl
from revpred.config import get_project_config
from revpred.models import FullRidgeRegressor, make_rf, make_ridge, make_xgb
from revpred.targets import Log1pTargetTransform


def _resolve_estimator(spec: Any, default_role: str = "base") -> Any:
    """Resolve estimator from string identifier, callable, or existing estimator instance."""
    if isinstance(spec, str):
        cfg = get_project_config()
        key = spec.lower()
        if key == "ridge":
            return make_ridge()
        if key == "rf":
            return make_rf()
        if key == "xgb":
            meta_params = cfg.get("stack", {}).get("meta_params", {})
            return make_xgb(params=meta_params)
        raise ValueError(f"Unknown estimator string identifier: {spec}")
    if callable(spec) and not hasattr(spec, "fit"):
        return spec()
    return spec


def _fit_with_optional_sample_weight(
    estimator: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit estimator, passing sample_weight appropriately if supported and provided."""
    if sample_weight is None:
        estimator.fit(X, y)
        return estimator

    if isinstance(estimator, Pipeline):
        fitted = False
        for step_name in estimator.named_steps:
            if step_name == "ridge":
                kw = {f"{step_name}__sample_weight": sample_weight}
                estimator.fit(X, y, **kw)
                fitted = True
                break
        if not fitted:
            estimator.fit(X, y)
        return estimator

    try:
        estimator.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        estimator.fit(X, y)
    return estimator


class TemporalStacker:
    """Temporal expanding-window stacking model.

    Generates out-of-fold (OOF) base predictions strictly from earlier months,
    trains a meta-learner on the OOF base predictions matrix, and refits all
    base models on the entire training set for subsequent inference.
    """

    def __init__(
        self,
        bases: Sequence[Any] | None = None,
        meta: Any | None = None,
        min_inner_train_months: int | None = None,
        features: list[str] | None = None,
        target_col: str = "revenue",
        month_col: str = "month",
        row_id_col: str = "row_id",
        target_transform: str | Any | None = "log1p",
        resampler: str | None = "weights",
        cache: dict[Any, Any] | None = None,
    ):
        cfg = get_project_config()
        stack_cfg = cfg.get("stack", {})

        if bases is None:
            configured_bases = stack_cfg.get("bases", ["ridge", "rf"])
            raw_bases = list(configured_bases)
        else:
            raw_bases = list(bases)

        self.bases: list[Any] = []
        self.base_features_: list[list[str] | None] = []
        for b in raw_bases:
            if isinstance(b, (tuple, list)) and len(b) == 2 and (b[1] is None or isinstance(b[1], (list, tuple))):
                est_spec, feat_spec = b
                self.bases.append(_resolve_estimator(est_spec, "base"))
                self.base_features_.append(list(feat_spec) if feat_spec is not None else None)
            else:
                self.bases.append(_resolve_estimator(b, "base"))
                self.base_features_.append(None)

        if meta is None:
            configured_meta = stack_cfg.get("meta", "xgb")
            self.meta = _resolve_estimator(configured_meta, "meta")
        else:
            self.meta = _resolve_estimator(meta, "meta")

        if min_inner_train_months is None:
            self.min_inner_train_months = int(stack_cfg.get("min_inner_train_months", 3))
        else:
            self.min_inner_train_months = int(min_inner_train_months)

        self.features = list(features) if features is not None else None
        self.features_: list[str] = []
        self.target_col = target_col
        self.month_col = month_col
        self.row_id_col = row_id_col
        self.target_transform = target_transform
        self.resampler = resampler
        self.cache = cache

        self.fitted_bases_: list[Any] = []
        self.fitted_meta_: Any = None
        self.oof_log_: pd.DataFrame | None = None
        self.transform_y_: Callable[[Any], Any] = lambda y: np.asarray(y, dtype=float)
        self.inverse_y_: Callable[[Any], Any] = lambda p: np.asarray(p, dtype=float)
        self._init_target_transform()

    def _init_target_transform(self) -> None:
        if self.target_transform is None or self.target_transform == "none":
            self.transform_y_ = lambda y: np.asarray(y, dtype=float)
            self.inverse_y_ = lambda p: np.asarray(p, dtype=float)
        elif self.target_transform == "log1p":
            self.transform_y_ = np.log1p
            self.inverse_y_ = lambda p: np.clip(np.expm1(p), 0.0, None)
        elif isinstance(self.target_transform, Log1pTargetTransform):
            self.transform_y_ = self.target_transform.transform
            self.inverse_y_ = self.target_transform.inverse
        elif isinstance(self.target_transform, tuple):
            self.transform_y_, self.inverse_y_ = self.target_transform
        else:
            raise ValueError(f"Unsupported target_transform: {self.target_transform}")

    def _get_base_X(self, df: pd.DataFrame, b_idx: int) -> pd.DataFrame | np.ndarray:
        feats = self.base_features_[b_idx]
        if feats is not None:
            base_model = self.bases[b_idx]
            if hasattr(base_model, "LOG_FEATURES") or isinstance(base_model, FullRidgeRegressor):
                cols = list(feats)
                if self.month_col in df.columns and self.month_col not in cols:
                    cols.append(self.month_col)
                return df[cols]
            return df[feats].to_numpy(dtype=float)
        return df[self.features_].to_numpy(dtype=float)

    @property
    def oof_log(self) -> pd.DataFrame:
        """Expose out-of-fold fit tracking log (row_id, month, fit_max_month)."""
        if self.oof_log_ is None:
            raise RuntimeError("TemporalStacker has not been fit yet.")
        return self.oof_log_

    def fit(self, train_df: pd.DataFrame) -> TemporalStacker:
        """Fit temporal stacker on train_df.

        For each inner month j with at least min_inner_train_months earlier train months:
          - fit every base model on rows with month strictly earlier than j
          - predict rows of month j -> OOF base predictions (transformed-target scale)
          - record per OOF row: fit_max_month (latest month used to fit predicting bases)
        Train meta learner on OOF matrix only (base predictions as features),
        with target transform and resampler applied to meta-training rows only.
        Refit every base model on all train rows for inference.
        """
        if self.features is not None:
            self.features_ = list(self.features)
        else:
            forbidden = {
                self.row_id_col,
                "series_id",
                "entity",
                "category",
                "region",
                self.month_col,
                self.target_col,
            }
            self.features_ = [
                c for c in train_df.columns if c not in forbidden and not c.startswith("aux_")
            ]

        unique_months = sorted(train_df[self.month_col].unique())
        n_months = len(unique_months)
        if n_months <= self.min_inner_train_months:
            raise ValueError(
                f"TemporalStacker requires more than {self.min_inner_train_months} months in train_df, "
                f"found {n_months}: {unique_months}"
            )

        oof_pred_blocks: list[np.ndarray] = []
        oof_meta_row_dfs: list[pd.DataFrame] = []
        oof_log_records: list[dict[str, Any]] = []

        # 1. Inner temporal expanding-window folds
        for j_idx in range(self.min_inner_train_months, n_months):
            j_month = unique_months[j_idx]
            earlier_months = tuple(unique_months[:j_idx])
            fit_max_month = unique_months[j_idx - 1]

            inner_train = train_df[train_df[self.month_col].isin(earlier_months)]
            inner_val = train_df[train_df[self.month_col] == j_month]

            y_in_tr_trans = self.transform_y_(inner_train[self.target_col].to_numpy(dtype=float))

            base_preds_j: list[np.ndarray] = []
            for b_idx, base_spec in enumerate(self.bases):
                X_in_tr = self._get_base_X(inner_train, b_idx)
                X_in_val = self._get_base_X(inner_val, b_idx)

                base_params = (
                    tuple(sorted((k, str(v)) for k, v in base_spec.get_params().items()))
                    if hasattr(base_spec, "get_params")
                    else ()
                )
                cache_key = (b_idx, earlier_months, base_params)
                if self.cache is not None and cache_key in self.cache:
                    base_model = self.cache[cache_key]
                else:
                    base_model = clone(base_spec)
                    base_model.fit(X_in_tr, y_in_tr_trans)
                    if self.cache is not None:
                        self.cache[cache_key] = base_model

                p_b = base_model.predict(X_in_val)
                base_preds_j.append(np.asarray(p_b, dtype=float))

            P_j = np.column_stack(base_preds_j)
            oof_pred_blocks.append(P_j)
            oof_meta_row_dfs.append(inner_val)

            row_ids = (
                inner_val[self.row_id_col].values
                if self.row_id_col in inner_val.columns
                else inner_val.index.values
            )
            for rid in row_ids:
                oof_log_records.append({
                    "row_id": rid,
                    "month": j_month,
                    "fit_max_month": fit_max_month,
                })

        # 2. Form meta-training matrix and record OOF log
        X_meta = np.vstack(oof_pred_blocks)
        meta_rows_df = pd.concat(oof_meta_row_dfs, ignore_index=True)
        self.oof_log_ = pd.DataFrame(oof_log_records)

        # 3. Train meta learner on OOF matrix only
        y_meta_raw = meta_rows_df[self.target_col].to_numpy(dtype=float)
        y_meta_trans = self.transform_y_(y_meta_raw)

        self.fitted_meta_ = clone(self.meta)
        if self.resampler == "weights":
            tail_thr = gl.tail_threshold(y_meta_raw)
            sample_weights = np.where(y_meta_raw >= tail_thr, 3.0, 1.0)
            _fit_with_optional_sample_weight(
                self.fitted_meta_, X_meta, y_meta_trans, sample_weight=sample_weights
            )
        else:
            _fit_with_optional_sample_weight(self.fitted_meta_, X_meta, y_meta_trans, sample_weight=None)

        # 4. Refit every base model on all train rows for inference
        y_all_trans = self.transform_y_(train_df[self.target_col].to_numpy(dtype=float))
        all_train_months = tuple(unique_months)

        self.fitted_bases_ = []
        for b_idx, base_spec in enumerate(self.bases):
            X_all = self._get_base_X(train_df, b_idx)
            base_params = (
                tuple(sorted((k, str(v)) for k, v in base_spec.get_params().items()))
                if hasattr(base_spec, "get_params")
                else ()
            )
            outer_cache_key = ("refit", b_idx, all_train_months, base_params)
            if self.cache is not None and outer_cache_key in self.cache:
                refit_base = self.cache[outer_cache_key]
            else:
                refit_base = clone(base_spec)
                refit_base.fit(X_all, y_all_trans)
                if self.cache is not None:
                    self.cache[outer_cache_key] = refit_base
            self.fitted_bases_.append(refit_base)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on df: bases -> meta -> inverted target."""
        if not self.fitted_bases_ or self.fitted_meta_ is None:
            raise RuntimeError("TemporalStacker must be fitted before predict can be called.")

        base_preds = []
        for b_idx, base_model in enumerate(self.fitted_bases_):
            X_val = self._get_base_X(df, b_idx)
            base_preds.append(np.asarray(base_model.predict(X_val), dtype=float))

        X_meta_val = np.column_stack(base_preds)
        meta_preds_trans = self.fitted_meta_.predict(X_meta_val)
        return self.inverse_y_(np.asarray(meta_preds_trans, dtype=float))

    def __call__(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
        self.fit(train_df)
        return self.predict(val_df)
