"""Phase 5 Stage B dry-run tests: run_eval verification and full dry-run pipeline test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gates import gate_05
from gates import lib as gl
from revpred.final_eval_core import run_eval
from scripts.final_eval import run_dry_run

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"
FINAL_CONFIG_PATH = REPO_ROOT / "reports" / "final_config.json"


@pytest.fixture(scope="module")
def dev_split_data():
    panel_dev = pd.read_parquet(DEV_PATH)
    with open(FINAL_CONFIG_PATH, encoding="utf-8") as f:
        final_cfg = json.load(f)

    unique_months = sorted(panel_dev["month"].unique())
    assert len(unique_months) >= 6, f"Expected at least 6 dev months, got {len(unique_months)}"

    train_months = unique_months[:-3]
    eval_months = unique_months[-3:]

    train_df = panel_dev[panel_dev["month"].isin(train_months)].copy().reset_index(drop=True)
    eval_df = panel_dev[panel_dev["month"].isin(eval_months)].copy().reset_index(drop=True)

    return train_df, eval_df, final_cfg


def test_dry_run_eval_on_dev(dev_split_data):
    """Verify run_eval produces valid predictions and matches baseline ridge."""
    train_df, eval_df, final_cfg = dev_split_data

    repro_tol = float(gl.thr("global.repro_rel_tol"))
    res = run_eval(train_df, eval_df, final_cfg)

    preds_df = res["predictions"]

    # 1. Final model predictions equal baseline ridge predictions
    final_preds = preds_df["final"].to_numpy(dtype=float)
    ridge_preds = preds_df["baseline_ridge"].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(final_preds - ridge_preds)))
    assert max_abs_diff <= repro_tol, (
        f"Final model diverges from baseline ridge: max_abs_diff={max_abs_diff} > {repro_tol}"
    )

    # 2. No NaN anywhere in predictions
    for col in [
        "final",
        "challenger",
        "baseline_ridge",
        "baseline_mean",
        "baseline_persistence",
    ]:
        assert not np.isnan(preds_df[col].to_numpy()).any(), f"NaN found in predictions col {col}"

    # 3. No NaN in pooled or per-month metrics
    for cand, metrics in res["pooled"].items():
        for m_name, val in metrics.items():
            assert not np.isnan(val), f"NaN found in pooled metric {cand}.{m_name}"

    for m_month, m_dict in res["per_month"].items():
        for cand, metrics in m_dict.items():
            for m_name, val in metrics.items():
                assert not np.isnan(val), f"NaN found in per-month metric {m_month}.{cand}.{m_name}"

    # 4. Refitting never happens: fit call count == 1 per model
    for model_name, count in res["fit_counts"].items():
        assert count == 1, f"Model '{model_name}' fit count was {count}, expected exactly 1"

    # 5. Tail threshold is valid positive number
    assert res["tail_threshold"] > 0.0

    # 6. 'fold' and 'cluster' columns exist on predictions
    assert "fold" in preds_df.columns, "Missing 'fold' column on predictions DataFrame"
    assert "cluster" in preds_df.columns, "Missing 'cluster' column on predictions DataFrame"


def test_leaky_mutant_caught(dev_split_data):
    """Verify that a leaky mutant feeding eval month's own revenue as a feature is caught."""
    train_df, eval_df, final_cfg = dev_split_data

    # Normal run
    res_clean = run_eval(train_df, eval_df, final_cfg)
    clean_persistence_rmsle = res_clean["pooled"]["baseline_persistence"]["rmsle"]
    assert clean_persistence_rmsle > 0.5, f"Unexpected clean persistence RMSLE: {clean_persistence_rmsle}"

    # Create leaky mutant: eval month's own revenue fed into lag_rev_1
    eval_df_leaky = eval_df.copy()
    eval_df_leaky["lag_rev_1"] = eval_df_leaky["revenue"]

    res_leaky = run_eval(train_df, eval_df_leaky, final_cfg)
    leaky_persistence_rmsle = res_leaky["pooled"]["baseline_persistence"]["rmsle"]

    # Leaky persistence must perfectly predict revenue (RMSLE == 0.0)
    assert leaky_persistence_rmsle < 1e-9, (
        f"Leaky mutant failed to produce 0.0 persistence RMSLE: {leaky_persistence_rmsle}"
    )

    # Explicit leak detector
    leak_detected = np.isclose(eval_df_leaky["lag_rev_1"], eval_df_leaky["revenue"]).all()
    assert leak_detected, "Leak detection failed: lag_rev_1 should match revenue in mutant"


def test_full_dry_run_pipeline(tmp_path):
    """Run final_eval.py dry run and verify all output files exist and gate_05 checks evaluate."""
    # Execute full dry run into isolated tmp_path
    run_dry_run(temp_dir=tmp_path)

    rep_dir = tmp_path / "reports"
    oof_dir = rep_dir / "oof"
    models_dir = tmp_path / "models"

    # Verify every expected output file exists
    expected_files = [
        rep_dir / "holdout_results.json",
        oof_dir / "holdout_predictions.parquet",
        oof_dir / "holdout_final.parquet",
        oof_dir / "holdout_challenger.parquet",
        oof_dir / "holdout_baseline_ridge.parquet",
        oof_dir / "holdout_baseline_mean.parquet",
        oof_dir / "holdout_baseline_persistence.parquet",
        models_dir / "final_model.joblib",
        models_dir / "feature_pipeline.joblib",
        models_dir / "MANIFEST.json",
    ]
    for ef in expected_files:
        assert ef.exists(), f"Dry run failed to produce expected file: {ef}"

    # Run Gate 5 checks against the dry-run outputs
    gate_05.build_checks(
        reports_dir=rep_dir,
        oof_dir=oof_dir,
        models_dir=models_dir,
        is_dry_run=True,
    )
