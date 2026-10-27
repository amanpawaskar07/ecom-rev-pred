"""Feature engineering and negative control utilities for Phase 2."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from revpred.panel import get_canonical_window

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_INPUT_COLUMNS: list[str] = [
    "lag_rev_1",
    "lag_rev_2",
    "lag_rev_3",
    "lag_lines_1",
    "lag_lines_2",
    "lag_lines_3",
    "roll_mean_rev_3",
    "trailing_price_3",
    "active_months_before",
    "months_since_first_sale",
    "month_of_year",
    "cum_rev_6",
    "category",
    "region",
    "month",
]


def cat_region_level_table(
    order_lines: pd.DataFrame,
    window: list[str] | None = None,
) -> pd.DataFrame:
    """Compute expanding mean table of log1p cell revenue strictly for past months m in [window_start, t-1].

    Where R_cell(m) = total revenue of the (category, region) cell in month m across all entities (0 if none),
    computed from order_lines. No fitting, strictly past months.
    Returns DataFrame keyed by (category, region, month) with column cat_region_level.
    """
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
        if idx == 0:
            means = pd.Series(0.0, index=log1p_cell_rev.index)
        else:
            means = log1p_cell_rev.iloc[:, :idx].mean(axis=1)
        df_t = means.reset_index(name="cat_region_level")
        df_t["month"] = t
        records.append(df_t)

    table = pd.concat(records, ignore_index=True)
    return table[["category", "region", "month", "cat_region_level"]]


class FeatureBuilder:
    """Stateful transformer learning bucket edges and P90 from training rows only.

    Applies price_bucket, high_value_flag, and cat_region_level.
    """

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

    def fit(self, train_df: pd.DataFrame, val_month: str | None = None) -> FeatureBuilder:
        """Fit bucket edges and P90 on training fold rows only.

        Raises ValueError if train_df contains any row at or after the first validation month.
        """
        active_val_month = val_month or self.val_month
        if (
            active_val_month is not None
            and "month" in train_df.columns
            and (train_df["month"] >= active_val_month).any()
        ):
            raise ValueError(
                f"Fit guard violation: train_df contains rows at or after first validation month '{active_val_month}'."
            )

        if "trailing_price_3" not in train_df.columns:
            raise KeyError("train_df missing required column 'trailing_price_3'")
        pos_prices = train_df.loc[train_df["trailing_price_3"] > 0, "trailing_price_3"].to_numpy(dtype=float)
        if len(pos_prices) > 0:
            self.price_quintiles_ = np.percentile(pos_prices, [20, 40, 60, 80])
        else:
            self.price_quintiles_ = np.array([0.0, 0.0, 0.0, 0.0])

        if "cum_rev_6" not in train_df.columns:
            raise KeyError("train_df missing required column 'cum_rev_6'")
        cum_rev = train_df["cum_rev_6"].to_numpy(dtype=float)
        self.p90_ = float(np.percentile(cum_rev, 90)) if len(cum_rev) > 0 else 0.0

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe by adding the three engineered columns."""
        if not self.is_fitted_:
            raise RuntimeError("FeatureBuilder must be fit before transform can be called.")

        out = df.copy()

        # 1. price_bucket: 1..5 by quintile edges if trailing_price_3 > 0, else 0
        trailing = out["trailing_price_3"].to_numpy(dtype=float)
        pos_mask = trailing > 0
        buckets = np.zeros(len(out), dtype=int)
        if np.any(pos_mask) and self.price_quintiles_ is not None:
            buckets[pos_mask] = np.digitize(trailing[pos_mask], self.price_quintiles_, right=True) + 1
        out["price_bucket"] = buckets

        # 2. high_value_flag: 1 if cum_rev_6 >= P90, else 0
        cum = out["cum_rev_6"].to_numpy(dtype=float)
        out["high_value_flag"] = (cum >= self.p90_).astype(int)

        # 3. cat_region_level: expanding mean from order_lines
        crl_table = self._ensure_cat_region_table()
        if "cat_region_level" in out.columns:
            out = out.drop(columns=["cat_region_level"])

        out = out.merge(
            crl_table,
            on=["category", "region", "month"],
            how="left",
        )
        out["cat_region_level"] = out["cat_region_level"].fillna(0.0).astype(float)

        return out

    def fit_transform(self, train_df: pd.DataFrame, val_month: str | None = None) -> pd.DataFrame:
        return self.fit(train_df, val_month=val_month).transform(train_df)


def shuffle_feature_within_month(
    df: pd.DataFrame,
    feature_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Permute feature_col across rows within each month (seeded).

    Preserves each month's sorted value multiset.
    """
    df_out = df.copy()
    rng = np.random.default_rng(seed)
    for m in df_out["month"].unique():
        mask = df_out["month"] == m
        vals = df_out.loc[mask, feature_col].to_numpy()
        perm = rng.permutation(vals)
        df_out.loc[mask, feature_col] = perm
    return df_out
