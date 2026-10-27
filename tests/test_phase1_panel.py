"""Tests for panel construction, truncation invariance, causal entry, and CV splitter."""
import pytest

from revpred.config import get_thresholds
from revpred.cv import expanding_window_splits
from revpred.data import load_order_lines
from revpred.panel import FEATURES, build_panel, get_canonical_window


@pytest.fixture(scope="module")
def order_lines():
    return load_order_lines()


def test_features_constant():
    # Target and aux_* are never features
    assert "revenue" not in FEATURES
    assert "aux_lines" not in FEATURES
    assert "aux_active" not in FEATURES
    assert "cum_rev_6" not in FEATURES
    assert "row_id" not in FEATURES
    assert "series_id" not in FEATURES
    assert "month" not in FEATURES
    assert "lag_rev_1" in FEATURES
    assert "region" in FEATURES
    assert "category" in FEATURES


def test_truncation_invariance(order_lines):
    window = get_canonical_window(order_lines)
    panel_full = build_panel(order_lines, entity="product_id", window=window)

    # 3 seeded cut months
    cut_months = ["2017-07", "2017-11", "2018-04"]

    num_cols = [
        "revenue", "aux_lines", "aux_active", "cum_rev_6",
        "lag_rev_1", "lag_rev_2", "lag_rev_3",
        "lag_lines_1", "lag_lines_2", "lag_lines_3",
        "roll_mean_rev_3", "trailing_price_3",
        "active_months_before", "months_since_first_sale", "month_of_year",
    ]

    for cut_m in cut_months:
        # (a) Lines with month >= cut_m deleted
        ol_a = order_lines[order_lines["month"] < cut_m].copy()
        panel_a = build_panel(ol_a, entity="product_id", window=window)

        sub_full_a = panel_full[panel_full["month"] < cut_m].sort_values("row_id").reset_index(drop=True)
        sub_a = panel_a[panel_a["month"] < cut_m].sort_values("row_id").reset_index(drop=True)

        assert len(sub_full_a) == len(sub_a), f"Cut {cut_m} row counts differ for (a)"
        assert (sub_full_a["row_id"].values == sub_a["row_id"].values).all()

        max_diff_a = (sub_full_a[num_cols] - sub_a[num_cols]).abs().max().max()
        assert max_diff_a == 0.0, f"Cut {cut_m} max abs diff {max_diff_a} > 0 for (a)"

        # (b) Price x100 for month >= cut_m
        ol_b = order_lines.copy()
        ol_b.loc[ol_b["month"] >= cut_m, "price"] *= 100.0
        panel_b = build_panel(ol_b, entity="product_id", window=window)

        sub_full_b = panel_full[panel_full["month"] < cut_m].sort_values("row_id").reset_index(drop=True)
        sub_b = panel_b[panel_b["month"] < cut_m].sort_values("row_id").reset_index(drop=True)

        assert len(sub_full_b) == len(sub_b), f"Cut {cut_m} row counts differ for (b)"
        assert (sub_full_b["row_id"].values == sub_b["row_id"].values).all()

        max_diff_b = (sub_full_b[num_cols] - sub_b[num_cols]).abs().max().max()
        assert max_diff_b == 0.0, f"Cut {cut_m} max abs diff {max_diff_b} > 0 for (b)"


def test_causal_entry(order_lines):
    window = get_canonical_window(order_lines)
    panel_full = build_panel(order_lines, entity="product_id", window=window)

    for cut_m in ["2017-07", "2017-11", "2018-04"]:
        ol_cut = order_lines[order_lines["month"] < cut_m].copy()
        panel_cut = build_panel(ol_cut, entity="product_id", window=window)

        # For every month t < cut_m, the series set must be identical
        months_to_check = [m for m in window[3:] if m < cut_m]
        for m in months_to_check:
            series_full = set(panel_full[panel_full["month"] == m]["series_id"])
            series_cut = set(panel_cut[panel_cut["month"] == m]["series_id"])
            assert series_full == series_cut, f"Series set at month {m} differs between full and cut at {cut_m}"


def test_target_conditioned_inclusion_guard(order_lines):
    window = get_canonical_window(order_lines)
    panel = build_panel(order_lines, entity="product_id", window=window)

    # Share of rows with revenue > 0 is strictly < 1 (zero-revenue rows are included once entered)
    pos_share = (panel["revenue"] > 0).mean()
    assert 0.0 < pos_share < 1.0

    # No series has a row before its entry month
    thr = get_thresholds()
    min_active = int(thr["phase1"]["min_active_months_before_entry"])
    assert (panel["active_months_before"] >= min_active).all()


def test_cv_splitter():
    months_seq = [
        "2017-04", "2017-04", "2017-05", "2017-05", "2017-06",
        "2017-07", "2017-08", "2017-09", "2017-10", "2017-11",
    ]
    splits = expanding_window_splits(months_seq, min_train_months=3, val_months=1)
    assert len(splits) == 5

    for train_idx, val_idx in splits:
        train_m = [months_seq[i] for i in train_idx]
        val_m = [months_seq[i] for i in val_idx]
        assert max(train_m) < min(val_m)
        assert len(set(train_idx).intersection(set(val_idx))) == 0