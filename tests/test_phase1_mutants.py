"""Tests that mutants in tests/mutants/mutants_phase1.py are rejected."""
from unittest.mock import patch

import pytest

from revpred.data import load_order_lines
from revpred.panel import build_panel, get_canonical_window, split_and_save_panel
from tests.mutants.mutants_phase1 import (
    build_panel_mutant_a,
    build_panel_mutant_b,
    split_and_save_panel_mutant_c,
)


@pytest.fixture(scope="module")
def order_lines():
    return load_order_lines()


def test_mutant_a_lag_leak_rejected(order_lines):
    """Mutant (a): lag features use the current month. Must make test fail."""
    # Production code test: lag_rev_1 is not current month revenue
    p_normal = build_panel(order_lines, entity="product_id")
    diff_normal = (p_normal["lag_rev_1"] - p_normal["revenue"]).abs().sum()
    assert diff_normal > 0.0, "Production code must not leak current revenue into lag"

    # Mutant code: fails the leakage check (lag_rev_1 == revenue)
    p_mutant = build_panel_mutant_a(order_lines)
    diff_mutant = (p_mutant["lag_rev_1"] - p_mutant["revenue"]).abs().sum()
    assert diff_mutant == 0.0, "Mutant (a) failed to exhibit current-month leak"


def test_mutant_b_causal_entry_rejected(order_lines):
    """Mutant (b): entry computed from full-data active counts. Must make test fail."""
    window = get_canonical_window(order_lines)
    cut_m = "2017-07"
    ol_cut = order_lines[order_lines["month"] < cut_m].copy()

    # Production code test: causal entry invariance holds
    p_full = build_panel(order_lines, entity="product_id", window=window)
    p_cut = build_panel(ol_cut, entity="product_id", window=window)
    s_full = set(p_full[p_full["month"] == "2017-06"]["series_id"])
    s_cut = set(p_cut[p_cut["month"] == "2017-06"]["series_id"])
    assert s_full == s_cut, "Production code must satisfy causal entry invariance"

    # Mutant code: fails causal entry (future sales leak into past entry decisions)
    mb_full = build_panel_mutant_b(order_lines, window=window)
    mb_cut = build_panel_mutant_b(ol_cut, window=window)
    s_mb_full = set(mb_full[mb_full["month"] == "2017-06"]["series_id"])
    s_mb_cut = set(mb_cut[mb_cut["month"] == "2017-06"]["series_id"])
    assert s_mb_full != s_mb_cut, "Mutant (b) failed to exhibit causal entry violation"


def test_mutant_c_dev_truncation_rejected(order_lines, tmp_path):
    """Mutant (c): dev panel built from full data then sliced. Must make test fail."""
    # Production code: calls build_panel on truncated order_lines (month <= 2018-05)
    calls_normal = []
    with patch("revpred.panel.build_panel", side_effect=build_panel) as mock_bp:
        split_and_save_panel(
            order_lines,
            entity="product_id",
            processed_dir=tmp_path / "proc",
            holdout_dir=tmp_path / "hold",
            reports_dir=tmp_path / "rep",
        )
        for call in mock_bp.call_args_list:
            ol_arg = call.args[0]
            calls_normal.append(str(ol_arg["month"].max()))

    assert "2018-05" in calls_normal, "Production code must build dev panel from truncated order_lines"

    # Mutant code: never calls build_panel on truncated order_lines
    calls_mutant = []
    with patch("tests.mutants.mutants_phase1.build_panel", side_effect=build_panel) as mock_bp_c:
        split_and_save_panel_mutant_c(
            order_lines,
            entity="product_id",
            processed_dir=tmp_path / "proc_m",
            holdout_dir=tmp_path / "hold_m",
            reports_dir=tmp_path / "rep_m",
        )
        for call in mock_bp_c.call_args_list:
            ol_arg = call.args[0]
            calls_mutant.append(str(ol_arg["month"].max()))

    assert "2018-05" not in calls_mutant, "Mutant (c) failed to exhibit full-data-then-slice behavior"