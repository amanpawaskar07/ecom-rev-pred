"""Phase 4 mutants for leakage and causal testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from revpred.stack import TemporalStacker


class ShuffledKFoldStackerMutantA(TemporalStacker):
    """Mutant (a): OOF base predictions generated via shuffled KFold (LEAK: mixes future months into base fits)."""

    def fit(self, train_df: pd.DataFrame) -> ShuffledKFoldStackerMutantA:
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

        n = len(train_df)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        P_oof = np.zeros((n, len(self.bases)), dtype=float)
        fit_max_months = [None] * n

        X = train_df[self.features_].to_numpy(dtype=float)
        y_trans = self.transform_y_(train_df[self.target_col].to_numpy(dtype=float))
        months = train_df[self.month_col].to_numpy()

        for k_tr_idx, k_val_idx in kf.split(train_df):
            # LEAK: k_tr_idx contains rows from later months than rows in k_val_idx
            X_tr, y_tr = X[k_tr_idx], y_trans[k_tr_idx]
            X_val = X[k_val_idx]
            max_m = str(np.max(months[k_tr_idx]))

            for b_idx, base_spec in enumerate(self.bases):
                base_model = clone(base_spec)
                base_model.fit(X_tr, y_tr)
                P_oof[k_val_idx, b_idx] = base_model.predict(X_val)

            for idx in k_val_idx:
                fit_max_months[idx] = max_m

        row_ids = (
            train_df[self.row_id_col].values
            if self.row_id_col in train_df.columns
            else train_df.index.values
        )
        self.oof_log_ = pd.DataFrame({
            "row_id": row_ids,
            "month": months,
            "fit_max_month": fit_max_months,
        })

        # Fit meta on this leaked OOF matrix
        self.fitted_meta_ = clone(self.meta)
        self.fitted_meta_.fit(P_oof, y_trans)

        # Refit bases on all rows
        self.fitted_bases_ = []
        for base_spec in self.bases:
            refit_base = clone(base_spec)
            refit_base.fit(X, y_trans)
            self.fitted_bases_.append(refit_base)

        return self


class AllRowsFitStackerMutantB(TemporalStacker):
    """Mutant (b): Base models fit on ALL rows and predict ALL rows to form 'OOF' (LEAK)."""

    def fit(self, train_df: pd.DataFrame) -> AllRowsFitStackerMutantB:
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

        X = train_df[self.features_].to_numpy(dtype=float)
        y_trans = self.transform_y_(train_df[self.target_col].to_numpy(dtype=float))
        overall_max_month = str(train_df[self.month_col].max())

        # LEAK: fit bases on all rows
        base_preds = []
        self.fitted_bases_ = []
        for base_spec in self.bases:
            base_model = clone(base_spec)
            base_model.fit(X, y_trans)
            self.fitted_bases_.append(base_model)
            base_preds.append(base_model.predict(X))

        P_all = np.column_stack(base_preds)

        row_ids = (
            train_df[self.row_id_col].values
            if self.row_id_col in train_df.columns
            else train_df.index.values
        )
        self.oof_log_ = pd.DataFrame({
            "row_id": row_ids,
            "month": train_df[self.month_col].values,
            "fit_max_month": np.full(len(train_df), overall_max_month),
        })

        self.fitted_meta_ = clone(self.meta)
        self.fitted_meta_.fit(P_all, y_trans)
        return self


class InSampleMetaStackerMutantC(TemporalStacker):
    """Mutant (c): Meta learner fit on in-sample base predictions (no OOF split at all) (LEAK)."""

    def fit(self, train_df: pd.DataFrame) -> InSampleMetaStackerMutantC:
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

        X = train_df[self.features_].to_numpy(dtype=float)
        y_trans = self.transform_y_(train_df[self.target_col].to_numpy(dtype=float))
        overall_max_month = str(train_df[self.month_col].max())

        # Fit bases on all rows
        self.fitted_bases_ = []
        base_in_sample_preds = []
        for base_spec in self.bases:
            base_model = clone(base_spec)
            base_model.fit(X, y_trans)
            self.fitted_bases_.append(base_model)
            base_in_sample_preds.append(base_model.predict(X))

        # LEAK: Meta is trained on in-sample base predictions instead of OOF
        P_in_sample = np.column_stack(base_in_sample_preds)

        row_ids = (
            train_df[self.row_id_col].values
            if self.row_id_col in train_df.columns
            else train_df.index.values
        )
        self.oof_log_ = pd.DataFrame({
            "row_id": row_ids,
            "month": train_df[self.month_col].values,
            "fit_max_month": np.full(len(train_df), overall_max_month),
        })

        self.fitted_meta_ = clone(self.meta)
        self.fitted_meta_.fit(P_in_sample, y_trans)
        return self
