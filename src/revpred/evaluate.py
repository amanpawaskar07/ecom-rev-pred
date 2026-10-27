"""Cross-validation evaluation harness and OOF prediction generator."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gates.lib import tail_threshold
from revpred.config import get_project_config
from revpred.cv import expanding_window_splits

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OOF_DIR = REPO_ROOT / "reports" / "oof"


def run_cv(
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray | pd.Series | list[float]],
    panel_dev: pd.DataFrame,
    name: str,
    oof_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Run expanding-window CV on panel_dev, produce OOF frame, save to reports/oof/<name>.parquet."""
    cfg = get_project_config()
    min_train = int(cfg["cv"]["min_train_months"])
    val_months = int(cfg["cv"]["val_months"])

    splits = expanding_window_splits(
        panel_dev["month"],
        min_train_months=min_train,
        val_months=val_months,
    )

    if not splits:
        raise ValueError("No valid CV splits could be generated for panel_dev.")

    out_oof_dir = Path(oof_dir) if oof_dir else OOF_DIR
    out_oof_dir.mkdir(parents=True, exist_ok=True)

    fold_dfs: list[pd.DataFrame] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = panel_dev.iloc[train_idx].copy().reset_index(drop=True)
        val_df = panel_dev.iloc[val_idx].copy().reset_index(drop=True)

        # tail_thr per fold = quantile(train revenue > 0, 0.90)
        tail_thr_val = tail_threshold(train_df["revenue"].to_numpy())

        # Fit and predict
        preds = fit_predict_fn(train_df, val_df)
        preds_arr = np.clip(np.asarray(preds, dtype=float), 0.0, None)

        fold_oof = pd.DataFrame({
            "row_id": val_df["row_id"].values,
            "series_id": val_df["series_id"].values,
            "cluster": val_df["series_id"].values,
            "month": val_df["month"].values,
            "fold": fold_idx,
            "y": val_df["revenue"].values,
            "pred": preds_arr,
            "tail_thr": tail_thr_val,
        })
        fold_dfs.append(fold_oof)

    oof_df = pd.concat(fold_dfs, ignore_index=True)

    # Save to reports/oof/<name>.parquet
    out_file = out_oof_dir / f"{name}.parquet"
    oof_df.to_parquet(out_file, index=False)

    return oof_df


def evaluate_oof(oof_df: pd.DataFrame) -> dict[str, Any]:
    """Compute per-fold and pooled primary and secondary metrics using gates.lib functions."""
    from gates.lib import mae, rmsle, tail_mae, wape

    folds = sorted(oof_df["fold"].unique())
    fold_metrics: dict[str, dict[str, float]] = {}

    for f in folds:
        f_sub = oof_df[oof_df["fold"] == f]
        y_f = f_sub["y"].to_numpy()
        p_f = f_sub["pred"].to_numpy()
        thr_f = float(f_sub["tail_thr"].iloc[0])

        fold_metrics[f"fold_{f}"] = {
            "rmsle": rmsle(y_f, p_f),
            "mae": mae(y_f, p_f),
            "wape": wape(y_f, p_f),
            "tail_mae": tail_mae(y_f, p_f, thr_f),
        }

    y_pool = oof_df["y"].to_numpy()
    p_pool = oof_df["pred"].to_numpy()

    # Pooled tail MAE: evaluate over rows where y >= row's fold tail_thr
    tail_mask = y_pool >= oof_df["tail_thr"].to_numpy()
    if np.any(tail_mask):
        pooled_tail_mae = float(np.mean(np.abs(p_pool[tail_mask] - y_pool[tail_mask])))
    else:
        pooled_tail_mae = 0.0

    pooled_metrics = {
        "rmsle": rmsle(y_pool, p_pool),
        "mae": mae(y_pool, p_pool),
        "wape": wape(y_pool, p_pool),
        "tail_mae": pooled_tail_mae,
    }

    return {
        "pooled": pooled_metrics,
        "per_fold": fold_metrics,
    }