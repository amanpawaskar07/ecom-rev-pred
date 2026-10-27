"""Tests for Phase 4 temporal stacking and linear truth harness."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from revpred.config import get_project_config, get_thresholds
from revpred.cv import expanding_window_splits
from revpred.harness import make_linear_truth
from revpred.stack import TemporalStacker

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_oof_causality_violations_zero() -> None:
    """For every OOF row: fit_max_month < row month (violations = 0)."""
    n = 600
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "month": np.repeat(["2017-01", "2017-02", "2017-03", "2017-04", "2017-05", "2017-06"], 100),
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "revenue": np.random.exponential(100, size=n),
    })

    stacker = TemporalStacker(min_inner_train_months=3, features=["f1", "f2"])
    stacker.fit(df)

    oof_log = stacker.oof_log
    assert len(oof_log) == 300  # months 4, 5, 6
    violations = int((oof_log["fit_max_month"] >= oof_log["month"]).sum())
    assert violations == 0, f"Found {violations} causality violations!"


def test_forbidden_splitters_not_in_stack() -> None:
    """stack.py does not use KFold, ShuffleSplit or StackingRegressor."""
    stack_file = REPO_ROOT / "src" / "revpred" / "stack.py"
    content = stack_file.read_text(encoding="utf-8")
    forbidden = ["KFold", "ShuffleSplit", "StackingRegressor"]
    for word in forbidden:
        assert word not in content, f"Forbidden splitter '{word}' found in stack.py!"


def test_meta_train_rows_min_threshold_on_real_data() -> None:
    """Meta training rows per outer fold >= phase4.min_meta_train_rows (1000)."""
    panel_path = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"
    if not panel_path.exists():
        pytest.skip("panel_dev.parquet not found")

    panel_dev = pd.read_parquet(panel_path)
    cfg = get_project_config()
    thr = get_thresholds()
    min_meta_rows_thr = int(thr["phase4"]["min_meta_train_rows"])

    min_train = int(cfg["cv"]["min_train_months"])
    val_months = int(cfg["cv"]["val_months"])
    splits = expanding_window_splits(
        panel_dev["month"],
        min_train_months=min_train,
        val_months=val_months,
    )

    base_features = list(cfg["features"]["base"])

    # Test on fold 0 (the smallest outer fold)
    tr_idx, _ = splits[0]
    tr_df = panel_dev.iloc[tr_idx].copy().reset_index(drop=True)

    stacker = TemporalStacker(
        min_inner_train_months=int(cfg["stack"]["min_inner_train_months"]),
        features=base_features,
    )
    stacker.fit(tr_df)

    n_meta_rows = len(stacker.oof_log)
    assert n_meta_rows >= min_meta_rows_thr, (
        f"Meta training rows {n_meta_rows} is below threshold {min_meta_rows_thr}!"
    )


def test_linear_truth_harness_properties() -> None:
    """Linear truth dataset matches fixed specification."""
    df = make_linear_truth(seed=42, n=4000, d=10)
    assert df.shape == (4000, 15)
    assert df["month"].nunique() == 20
    assert (df["month"].value_counts() == 200).all()
    assert "revenue" in df.columns
    assert "y" in df.columns
    for i in range(10):
        assert f"x_{i}" in df.columns
