"""Tests for Phase 2 engineered features, causality, leakage guards, and mutants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revpred.data import load_order_lines
from revpred.features import (
    FeatureBuilder,
    cat_region_level_table,
    shuffle_feature_within_month,
)
from tests.mutants.mutants_phase2 import (
    FeatureBuilderMutantA,
    cat_region_level_table_mutant_b,
    shuffle_feature_across_months_mutant_c,
)


@pytest.fixture(scope="module")
def order_lines():
    return load_order_lines()


@pytest.fixture(scope="module")
def panel_dev():
    return pd.read_parquet("data/processed/panel_dev.parquet")


CUT_MONTHS = ["2017-07", "2017-11", "2018-04"]


def test_perturbation_invariance(order_lines, panel_dev):
    """Perturbation: for each cut month, perturbing validation-side inputs leaves fitted state bit-identical."""
    for cut_m in CUT_MONTHS:
        train_df = panel_dev[panel_dev["month"] < cut_m].copy()
        val_df = panel_dev[panel_dev["month"] >= cut_m].copy()

        # Fit on unperturbed train
        fb_clean = FeatureBuilder(order_lines=order_lines, val_month=cut_m)
        fb_clean.fit(train_df)

        # Perturb validation inputs
        val_perturbed = val_df.copy()
        val_perturbed["cum_rev_6"] *= 100.0
        val_perturbed["trailing_price_3"] *= 100.0
        ol_perturbed = order_lines[order_lines["month"] < cut_m].copy()

        fb_perturbed = FeatureBuilder(order_lines=ol_perturbed, val_month=cut_m)
        fb_perturbed.fit(train_df)

        # Assert fitted state (edges, P90) is bit-identical
        assert np.array_equal(fb_clean.price_quintiles_, fb_perturbed.price_quintiles_), (
            f"price_quintiles_ differed under perturbation at cut month {cut_m}"
        )
        assert fb_clean.p90_ == fb_perturbed.p90_, (
            f"p90_ differed under perturbation at cut month {cut_m}"
        )


def test_cell_level_causality(order_lines):
    """Cell-level causality: cat_region_level at month t is unchanged when order_lines >= t deleted/inflated."""
    crl_orig = cat_region_level_table(order_lines)

    for cut_m in CUT_MONTHS:
        # 1. Delete order lines with month >= cut_m
        ol_del = order_lines[order_lines["month"] < cut_m].copy()
        crl_del = cat_region_level_table(ol_del)

        orig_t = crl_orig[crl_orig["month"] == cut_m][["category", "region", "cat_region_level"]]
        del_t = crl_del[crl_del["month"] == cut_m][["category", "region", "cat_region_level"]]

        merged_del = orig_t.merge(del_t, on=["category", "region"], how="outer", suffixes=("_orig", "_del"))
        merged_del["cat_region_level_orig"] = merged_del["cat_region_level_orig"].fillna(0.0)
        merged_del["cat_region_level_del"] = merged_del["cat_region_level_del"].fillna(0.0)

        diff_del = (merged_del["cat_region_level_orig"] - merged_del["cat_region_level_del"]).abs().max()
        assert diff_del == 0.0, f"cat_region_level changed after deleting future orders at {cut_m}: max diff {diff_del}"

        # 2. Inflate order lines with month >= cut_m by 100x
        ol_inf = order_lines.copy()
        ol_inf.loc[ol_inf["month"] >= cut_m, "price"] *= 100.0
        crl_inf = cat_region_level_table(ol_inf)

        inf_t = crl_inf[crl_inf["month"] == cut_m][["category", "region", "cat_region_level"]]
        merged_inf = orig_t.merge(inf_t, on=["category", "region"], how="outer", suffixes=("_orig", "_inf"))
        merged_inf["cat_region_level_orig"] = merged_inf["cat_region_level_orig"].fillna(0.0)
        merged_inf["cat_region_level_inf"] = merged_inf["cat_region_level_inf"].fillna(0.0)

        diff_inf = (merged_inf["cat_region_level_orig"] - merged_inf["cat_region_level_inf"]).abs().max()
        assert diff_inf == 0.0, f"cat_region_level changed after inflating future orders at {cut_m}: max diff {diff_inf}"


def test_fit_guard_raises_on_validation_rows(panel_dev):
    """Fit guard: FeatureBuilder.fit raises when given any row at or after first validation month."""
    fb = FeatureBuilder(val_month="2017-10")
    with pytest.raises(ValueError, match="Fit guard violation"):
        fb.fit(panel_dev)


def test_within_month_shuffle_preserves_multiset(order_lines, panel_dev):
    """Within-month shuffle preserves each month's sorted value multiset."""
    crl = cat_region_level_table(order_lines)
    fb = FeatureBuilder(cat_region_df=crl, val_month="2017-10")
    fb.fit(panel_dev[panel_dev["month"] < "2017-10"])
    trans = fb.transform(panel_dev)

    for col in ["price_bucket", "high_value_flag", "cat_region_level"]:
        shuffled = shuffle_feature_within_month(trans, col, seed=42)
        for m in trans["month"].unique():
            orig_vals = np.sort(trans.loc[trans["month"] == m, col].to_numpy())
            shuf_vals = np.sort(shuffled.loc[shuffled["month"] == m, col].to_numpy())
            assert np.array_equal(orig_vals, shuf_vals), (
                f"Within-month multiset violated for {col} in month {m}"
            )


def test_mutant_a_rejected(panel_dev):
    """Mutant (a): edges fit on all rows fails under perturbation / violates isolation."""
    cut_m = "2017-07"
    train_clean = panel_dev[panel_dev["month"] < cut_m].copy()
    val_clean = panel_dev[panel_dev["month"] >= cut_m].copy()
    full_clean = pd.concat([train_clean, val_clean], ignore_index=True)

    val_perturbed = val_clean.copy()
    val_perturbed["trailing_price_3"] *= 100.0
    val_perturbed["cum_rev_6"] *= 100.0
    full_perturbed = pd.concat([train_clean, val_perturbed], ignore_index=True)

    fb_m = FeatureBuilderMutantA()
    fb_m.fit(full_clean)
    edges_clean = fb_m.price_quintiles_.copy()
    p90_clean = fb_m.p90_

    fb_m.fit(full_perturbed)
    edges_pert = fb_m.price_quintiles_.copy()
    p90_pert = fb_m.p90_

    assert not np.array_equal(edges_clean, edges_pert) or p90_clean != p90_pert, (
        "Mutant (a) failed to exhibit leakage under validation perturbation"
    )


def test_mutant_b_rejected(order_lines):
    """Mutant (b): level includes month t fails cell-level causality check."""
    cut_m = "2017-07"
    crl_m_orig = cat_region_level_table_mutant_b(order_lines)

    ol_del = order_lines[order_lines["month"] < cut_m].copy()
    crl_m_del = cat_region_level_table_mutant_b(ol_del)

    orig_t = crl_m_orig[crl_m_orig["month"] == cut_m][["category", "region", "cat_region_level"]]
    del_t = crl_m_del[crl_m_del["month"] == cut_m][["category", "region", "cat_region_level"]]

    merged = orig_t.merge(del_t, on=["category", "region"], how="outer", suffixes=("_orig", "_del"))
    merged["cat_region_level_orig"] = merged["cat_region_level_orig"].fillna(0.0)
    merged["cat_region_level_del"] = merged["cat_region_level_del"].fillna(0.0)

    diff = (merged["cat_region_level_orig"] - merged["cat_region_level_del"]).abs().max()
    assert diff > 0.0, "Mutant (b) failed to be caught by cell causality test"


def test_mutant_c_rejected(order_lines, panel_dev):
    """Mutant (c): global shuffle fails within-month multiset check."""
    crl = cat_region_level_table(order_lines)
    fb = FeatureBuilder(cat_region_df=crl, val_month="2017-10")
    fb.fit(panel_dev[panel_dev["month"] < "2017-10"])
    trans = fb.transform(panel_dev)

    shuf_c = shuffle_feature_across_months_mutant_c(trans, "price_bucket", seed=42)
    mismatch_found = False
    for m in trans["month"].unique():
        orig_vals = np.sort(trans.loc[trans["month"] == m, "price_bucket"].to_numpy())
        shuf_vals = np.sort(shuf_c.loc[shuf_c["month"] == m, "price_bucket"].to_numpy())
        if not np.array_equal(orig_vals, shuf_vals):
            mismatch_found = True
            break
    assert mismatch_found, "Mutant (c) failed to be caught by within-month multiset test"
