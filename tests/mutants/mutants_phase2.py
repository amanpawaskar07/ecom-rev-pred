"""Phase 2 mutants for leakage and causal testing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from revpred.features import cat_region_level_table
from revpred.panel import get_canonical_window

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class FeatureBuilderMutantA:
    """Mutant (a): edges and P90 fit on ALL rows (train + validation), ignoring fit guards."""

    def __init__(
        self,
        cat_region_df: pd.DataFrame | None = None,
        order_lines: pd.DataFrame | None = None,
        val_month: str | None = None,
    ):
        self.val_month = val_month
        self.cat_region_df = cat_region_df
        self.order_lines = order_lines
        self.price_quintiles_: np.ndarray | None = None
        self.p90_: float | None = None
        self.is_fitted_: bool = False

    def _ensure_cat_region_table(self) -> pd.DataFrame:
        if self.cat_region_df is not None:
            return self.cat_region_df
        if self.order_lines is not None:
            self.cat_region_df = cat_region_level_table(self.order_lines)
            return self.cat_region_df
        ol_path = REPO_ROOT / "data" / "processed" / "order_lines.parquet"
        if ol_path.exists():
            ol = pd.read_parquet(ol_path)
            self.cat_region_df = cat_region_level_table(ol)
            return self.cat_region_df
        raise ValueError("Cannot compute cat_region_level without cat_region_df or order_lines.")

    def fit(self, train_df: pd.DataFrame, val_month: str | None = None) -> FeatureBuilderMutantA:
        # LEAK: Does not check fit guard, fits on whatever dataframe is passed (even if it contains val rows)
        pos_prices = train_df.loc[train_df["trailing_price_3"] > 0, "trailing_price_3"].to_numpy(dtype=float)
        if len(pos_prices) > 0:
            self.price_quintiles_ = np.percentile(pos_prices, [20, 40, 60, 80])
        else:
            self.price_quintiles_ = np.array([0.0, 0.0, 0.0, 0.0])

        cum_rev = train_df["cum_rev_6"].to_numpy(dtype=float)
        self.p90_ = float(np.percentile(cum_rev, 90)) if len(cum_rev) > 0 else 0.0

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FeatureBuilderMutantA must be fit before transform.")

        out = df.copy()
        trailing = out["trailing_price_3"].to_numpy(dtype=float)
        pos_mask = trailing > 0
        buckets = np.zeros(len(out), dtype=int)
        if np.any(pos_mask) and self.price_quintiles_ is not None:
            buckets[pos_mask] = np.digitize(trailing[pos_mask], self.price_quintiles_, right=True) + 1
        out["price_bucket"] = buckets

        cum = out["cum_rev_6"].to_numpy(dtype=float)
        out["high_value_flag"] = (cum >= self.p90_).astype(int)

        crl_table = self._ensure_cat_region_table()
        if "cat_region_level" in out.columns:
            out = out.drop(columns=["cat_region_level"])
        out = out.merge(crl_table, on=["category", "region", "month"], how="left")
        out["cat_region_level"] = out["cat_region_level"].fillna(0.0).astype(float)
        return out


def cat_region_level_table_mutant_b(
    order_lines: pd.DataFrame,
    window: list[str] | None = None,
) -> pd.DataFrame:
    """Mutant (b): level includes month t (months in [window_start, t]), leaking current revenue."""
    if window is None:
        window = get_canonical_window(order_lines)

    df_win = order_lines[order_lines["month"].isin(window)]

    cell_rev = (
        df_win.groupby(["category", "region", "month"])["price"]
        .sum()
        .unstack(level="month", fill_value=0.0)
        .reindex(columns=window, fill_value=0.0)
    )
    log1p_cell_rev = np.log1p(cell_rev)

    records: list[pd.DataFrame] = []
    for idx, t in enumerate(window):
        # LEAK: includes month t (:idx + 1) instead of strictly past months (:idx)
        means = log1p_cell_rev.iloc[:, : idx + 1].mean(axis=1)
        df_t = means.reset_index(name="cat_region_level")
        df_t["month"] = t
        records.append(df_t)

    table = pd.concat(records, ignore_index=True)
    return table[["category", "region", "month", "cat_region_level"]]


def shuffle_feature_across_months_mutant_c(
    df: pd.DataFrame,
    feature_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Mutant (c): permute feature_col globally across all rows (mixes months)."""
    df_out = df.copy()
    rng = np.random.default_rng(seed)
    vals = df_out[feature_col].to_numpy()
    df_out[feature_col] = rng.permutation(vals)
    return df_out
