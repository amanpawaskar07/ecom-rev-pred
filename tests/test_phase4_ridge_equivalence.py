"""Equivalence test: FullRidgeRegressor must replicate baseline Ridge exactly (<= 1e-9 abs diff on all 20,628 rows)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from revpred.config import get_project_config
from revpred.cv import expanding_window_splits
from revpred.models import FullRidgeRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_full_ridge_replicates_baseline_ridge() -> None:
    panel_path = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"
    baseline_path = REPO_ROOT / "reports" / "oof" / "baseline_ridge.parquet"

    if not panel_path.exists() or not baseline_path.exists():
        pytest.skip("Required parquet files missing")

    panel_dev = pd.read_parquet(panel_path)
    baseline_oof = pd.read_parquet(baseline_path)

    cfg = get_project_config()
    min_train = int(cfg["cv"]["min_train_months"])
    val_months = int(cfg["cv"]["val_months"])

    splits = expanding_window_splits(
        panel_dev["month"],
        min_train_months=min_train,
        val_months=val_months,
    )

    pred_records: list[dict[str, Any]] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        train_df = panel_dev.iloc[tr_idx].copy().reset_index(drop=True)
        val_df = panel_dev.iloc[val_idx].copy().reset_index(drop=True)

        y_tr_log = np.log1p(train_df["revenue"].to_numpy(dtype=float))

        model = FullRidgeRegressor()
        model.fit(train_df, y_tr_log)

        preds_log = model.predict(val_df)
        preds = np.clip(np.expm1(preds_log), 0.0, None)

        for rid, p in zip(val_df["row_id"].values, preds):
            pred_records.append({"row_id": rid, "pred_full_ridge": p})

    df_preds = pd.DataFrame(pred_records)
    assert len(df_preds) == 20628, f"Expected 20,628 rows, got {len(df_preds)}"

    merged = df_preds.merge(baseline_oof[["row_id", "pred"]], on="row_id", how="inner")
    assert len(merged) == 20628, f"Merged row count mismatch: {len(merged)} vs 20628"

    abs_diff = np.abs(merged["pred_full_ridge"].to_numpy() - merged["pred"].to_numpy())
    max_abs_diff = float(np.max(abs_diff))

    print(f"FullRidgeRegressor vs Baseline Ridge: max_abs_diff = {max_abs_diff:.4e}")
    assert max_abs_diff <= 1e-9, (
        f"Max absolute difference {max_abs_diff:.4e} exceeds 1e-9 threshold!"
    )
