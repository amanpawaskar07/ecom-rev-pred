"""Tests for src/revpred/data.py."""
from pathlib import Path

import pandas as pd
import pytest

from revpred.data import build_order_lines, compute_window, manifest


def test_manifest_real_data():
    m = manifest()
    assert "orders" in m
    assert m["orders"]["row_count"] == 99441
    assert m["items"]["row_count"] == 112650
    assert m["customers"]["row_count"] == 99441
    assert m["products"]["row_count"] == 32951
    assert m["translation"]["row_count"] == 71
    assert (Path("data/raw/MANIFEST.json")).exists()


def test_manifest_mismatch_fails(tmp_path, monkeypatch):
    test_file = tmp_path / "orders.csv"
    test_file.write_text("header\nrow1\n", encoding="utf-8")
    # Point raw_dir to tmp_path with bad row count
    fake_cfg = {
        "raw_dir": str(tmp_path),
        "files": {"orders": "orders.csv"},
        "expected_rowcounts": {"orders": 999},
    }
    monkeypatch.setattr("revpred.data.get_project_config", lambda: fake_cfg)
    with pytest.raises(ValueError, match="Row count mismatch"):
        manifest(raw_dir=tmp_path, manifest_path=tmp_path / "MANIFEST.json")


def test_compute_window_logic():
    # 20 months qualifying, 2 trailing/leading bad months
    monthly_orders = {
        "2016-10": 10,
        "2017-01": 1000,
        "2017-02": 1100,
        "2017-03": 1200,
        "2017-04": 1300,
        "2017-05": 1400,
        "2017-06": 1500,
        "2017-07": 1600,
        "2017-08": 1700,
        "2017-09": 1800,
        "2017-10": 1900,
        "2017-11": 2000,
        "2017-12": 2100,
        "2018-01": 2200,
        "2018-02": 2300,
        "2018-03": 2400,
        "2018-04": 2500,
        "2018-05": 2600,
        "2018-06": 2700,
        "2018-07": 2800,
        "2018-08": 2900,
        "2018-09": 5,
    }
    res = compute_window(monthly_orders, min_order_frac_of_median=0.10, lag_depth=3, holdout_months=3)
    assert res["window_start"] == "2017-01"
    assert res["window_end"] == "2018-08"
    assert len(res["window_months"]) == 20
    assert "2016-10" in res["dropped_months"]
    assert "2018-09" in res["dropped_months"]
    assert len(res["modeled_months"]) == 17
    assert len(res["dev_modeled_months"]) == 14
    assert len(res["holdout_modeled_months"]) == 3


def test_build_order_lines():
    df, profile = build_order_lines()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 112101

    expected_cols = ["order_id", "order_item_id", "product_id", "category", "region", "month", "price"]
    assert list(df.columns) == expected_cols
    assert (df["price"] >= 0).all()
    assert not df["category"].isna().any()
    assert not df["region"].isna().any()
    assert profile["unmatched_stats"]["unmatched_revenue"] == 0.0
    assert profile["unmatched_stats"]["unmapped_states"] == 0

    assert (Path("data/processed/order_lines.parquet")).exists()
    assert (Path("reports/data_profile.json")).exists()