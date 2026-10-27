"""Tests for baselines, CV evaluation, and metrics reporting."""
import json
from pathlib import Path

import pandas as pd
import pytest

from gates.lib import mae, rmsle, wape
from revpred.baselines import run_all_baselines

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def panel_dev():
    return pd.read_parquet("data/processed/panel_dev.parquet")


def test_baselines_run_and_oof_schema(panel_dev):
    run_all_baselines(panel_dev)

    for baseline_name in ["baseline_mean", "baseline_persistence", "baseline_ridge"]:
        oof_path = REPO_ROOT / "reports" / "oof" / f"{baseline_name}.parquet"
        assert oof_path.exists(), f"Missing OOF file: {oof_path}"
        oof = pd.read_parquet(oof_path)

        # Check required schema from AGENTS.md section 6
        expected_cols = ["row_id", "series_id", "month", "fold", "y", "pred", "tail_thr"]
        for col in expected_cols:
            assert col in oof.columns, f"Missing column {col} in {baseline_name}"

        assert len(oof) == len(panel_dev[panel_dev["month"] >= "2017-10"])  # 8 val folds starting at 2017-10
        assert (oof["pred"] >= 0).all()
        assert not oof["pred"].isna().any()

    # Verify baselines_cv.json
    cv_json_path = REPO_ROOT / "reports" / "baselines_cv.json"
    assert cv_json_path.exists()
    with open(cv_json_path, encoding="utf-8") as f:
        data = json.load(f)

    for b in ["baseline_mean", "baseline_persistence", "baseline_ridge"]:
        assert b in data
        assert "pooled" in data[b]
        assert "rmsle" in data[b]["pooled"]
        assert "mae" in data[b]["pooled"]
        assert "wape" in data[b]["pooled"]
        assert "tail_mae" in data[b]["pooled"]

        # Check recompute consistency
        oof = pd.read_parquet(REPO_ROOT / "reports" / "oof" / f"{b}.parquet")
        y = oof["y"].to_numpy()
        p = oof["pred"].to_numpy()
        assert abs(rmsle(y, p) - data[b]["pooled"]["rmsle"]) < 1e-6
        assert abs(mae(y, p) - data[b]["pooled"]["mae"]) < 1e-6
        assert abs(wape(y, p) - data[b]["pooled"]["wape"]) < 1e-6

    # Sanity: Ridge beats Mean on dev CV
    assert data["baseline_ridge"]["pooled"]["rmsle"] < data["baseline_mean"]["pooled"]["rmsle"]