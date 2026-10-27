"""Gate 2: Feature engineering, perturbation invariance, mutants, and negative controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates import lib as gl
from revpred.data import load_order_lines
from revpred.features import (
    FeatureBuilder,
    cat_region_level_table,
)
from tests.mutants.mutants_phase2 import (
    FeatureBuilderMutantA,
    cat_region_level_table_mutant_b,
    shuffle_feature_across_months_mutant_c,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"

CUT_MONTHS = ["2017-07", "2017-11", "2018-04"]


def build_checks() -> None:
    """Build and evaluate all Gate 2 checks."""
    order_lines = load_order_lines()
    panel_dev = pd.read_parquet(REPO_ROOT / "data" / "processed" / "panel_dev.parquet")

    # =========================================================================
    # 1. Perturbation Invariance & Cell Causality
    # =========================================================================
    perturbation_max_abs_diff = 0.0
    crl_orig = cat_region_level_table(order_lines)

    for cut_m in CUT_MONTHS:
        train_df = panel_dev[panel_dev["month"] < cut_m].copy()
        val_df = panel_dev[panel_dev["month"] >= cut_m].copy()

        # Clean fit
        fb_clean = FeatureBuilder(order_lines=order_lines, val_month=cut_m)
        fb_clean.fit(train_df)

        # Perturbed validation inputs
        val_perturbed = val_df.copy()
        val_perturbed["cum_rev_6"] *= 100.0
        val_perturbed["trailing_price_3"] *= 100.0
        ol_perturbed = order_lines[order_lines["month"] < cut_m].copy()

        fb_perturbed = FeatureBuilder(order_lines=ol_perturbed, val_month=cut_m)
        fb_perturbed.fit(train_df)

        d_edges = float(np.abs(fb_clean.price_quintiles_ - fb_perturbed.price_quintiles_).max())
        d_p90 = float(abs(fb_clean.p90_ - fb_perturbed.p90_))
        perturbation_max_abs_diff = max(perturbation_max_abs_diff, d_edges, d_p90)

        # Cell-level causality at cut month t
        ol_del = order_lines[order_lines["month"] < cut_m].copy()
        crl_del = cat_region_level_table(ol_del)

        orig_t = crl_orig[crl_orig["month"] == cut_m][["category", "region", "cat_region_level"]]
        del_t = crl_del[crl_del["month"] == cut_m][["category", "region", "cat_region_level"]]

        merged_del = orig_t.merge(del_t, on=["category", "region"], how="outer", suffixes=("_orig", "_del"))
        merged_del["cat_region_level_orig"] = merged_del["cat_region_level_orig"].fillna(0.0)
        merged_del["cat_region_level_del"] = merged_del["cat_region_level_del"].fillna(0.0)
        d_crl_del = float((merged_del["cat_region_level_orig"] - merged_del["cat_region_level_del"]).abs().max())

        ol_inf = order_lines.copy()
        ol_inf.loc[ol_inf["month"] >= cut_m, "price"] *= 100.0
        crl_inf = cat_region_level_table(ol_inf)

        inf_t = crl_inf[crl_inf["month"] == cut_m][["category", "region", "cat_region_level"]]
        merged_inf = orig_t.merge(inf_t, on=["category", "region"], how="outer", suffixes=("_orig", "_inf"))
        merged_inf["cat_region_level_orig"] = merged_inf["cat_region_level_orig"].fillna(0.0)
        merged_inf["cat_region_level_inf"] = merged_inf["cat_region_level_inf"].fillna(0.0)
        d_crl_inf = float((merged_inf["cat_region_level_orig"] - merged_inf["cat_region_level_inf"]).abs().max())

        perturbation_max_abs_diff = max(perturbation_max_abs_diff, d_crl_del, d_crl_inf)

    gl.check("perturbation_max_abs_diff", perturbation_max_abs_diff, "<=", "phase2.perturbation_max_abs_diff")

    # =========================================================================
    # 2. Mutants Detection Fraction
    # =========================================================================
    mutants_caught = 0

    # Mutant A: edges fit on all rows leaks under perturbation
    cut_test = "2017-07"
    tr_clean = panel_dev[panel_dev["month"] < cut_test].copy()
    val_clean = panel_dev[panel_dev["month"] >= cut_test].copy()
    val_pert = val_clean.copy()
    val_pert["trailing_price_3"] *= 100.0
    val_pert["cum_rev_6"] *= 100.0

    full_c = pd.concat([tr_clean, val_clean], ignore_index=True)
    full_p = pd.concat([tr_clean, val_pert], ignore_index=True)

    fb_ma = FeatureBuilderMutantA()
    fb_ma.fit(full_c)
    edges_ma_c = fb_ma.price_quintiles_.copy()
    p90_ma_c = fb_ma.p90_
    fb_ma.fit(full_p)
    edges_ma_p = fb_ma.price_quintiles_.copy()
    p90_ma_p = fb_ma.p90_

    if not np.array_equal(edges_ma_c, edges_ma_p) or p90_ma_c != p90_ma_p:
        mutants_caught += 1

    # Mutant B: level includes month t
    crl_mb_orig = cat_region_level_table_mutant_b(order_lines)
    ol_mb_del = order_lines[order_lines["month"] < cut_test].copy()
    crl_mb_del = cat_region_level_table_mutant_b(ol_mb_del)
    orig_mb_t = crl_mb_orig[crl_mb_orig["month"] == cut_test][["category", "region", "cat_region_level"]]
    del_mb_t = crl_mb_del[crl_mb_del["month"] == cut_test][["category", "region", "cat_region_level"]]
    merged_mb = orig_mb_t.merge(del_mb_t, on=["category", "region"], how="outer", suffixes=("_orig", "_del"))
    merged_mb["cat_region_level_orig"] = merged_mb["cat_region_level_orig"].fillna(0.0)
    merged_mb["cat_region_level_del"] = merged_mb["cat_region_level_del"].fillna(0.0)
    d_mb = float((merged_mb["cat_region_level_orig"] - merged_mb["cat_region_level_del"]).abs().max())
    if d_mb > 0.0:
        mutants_caught += 1

    # Mutant C: shuffle across months
    fb_mc = FeatureBuilder(cat_region_df=crl_orig, val_month="2017-10")
    fb_mc.fit(panel_dev[panel_dev["month"] < "2017-10"])
    dev_trans = fb_mc.transform(panel_dev)
    shuf_mc = shuffle_feature_across_months_mutant_c(dev_trans, "price_bucket", seed=42)
    mc_caught = False
    for m in dev_trans["month"].unique():
        orig_v = np.sort(dev_trans.loc[dev_trans["month"] == m, "price_bucket"].to_numpy())
        shuf_v = np.sort(shuf_mc.loc[shuf_mc["month"] == m, "price_bucket"].to_numpy())
        if not np.array_equal(orig_v, shuf_v):
            mc_caught = True
            break
    if mc_caught:
        mutants_caught += 1

    mutants_caught_frac = mutants_caught / 3.0
    gl.check("mutants_caught_frac", mutants_caught_frac, ">=", "phase2.mutant_detection_min_frac")

    # =========================================================================
    # 3. Controls & Real Feature Decisions
    # =========================================================================
    with open(REPORTS_DIR / "feature_decision.json", encoding="utf-8") as f:
        stored_decisions = json.load(f)

    f0_oof = pd.read_parquet(OOF_DIR / "feat_f0.parquet")
    features = ["price_bucket", "high_value_flag", "cat_region_level"]

    ctrl_files = {
        "price_bucket": "feat_c_price_bucket.parquet",
        "high_value_flag": "feat_c_high_value_flag.parquet",
        "cat_region_level": "feat_c_cat_region_level.parquet",
    }

    real_files = {
        "price_bucket": "feat_f1_price_bucket.parquet",
        "high_value_flag": "feat_f2_high_value_flag.parquet",
        "cat_region_level": "feat_f3_cat_region_level.parquet",
    }

    recomputed_kept = []
    recomputed_dropped = []
    recomputed_matches = 1

    real_results = {}
    ctrl_results = {}

    for f in features:
        # Check control
        ctrl_oof = pd.read_parquet(OOF_DIR / ctrl_files[f])
        ctrl_pair_df = pd.DataFrame({
            "cluster": ctrl_oof["cluster"],
            "fold": ctrl_oof["fold"],
            "y": ctrl_oof["y"],
            "tail_thr": ctrl_oof["tail_thr"],
            "pred_ctrl": ctrl_oof["pred"],
            "pred_f0": f0_oof["pred"],
        })
        res_ctrl = gl.decide_pair(ctrl_pair_df, advanced_col="pred_ctrl", rival_col="pred_f0", metric="rmsle")
        ctrl_results[f] = res_ctrl

        c_gain = float(res_ctrl["relative_gain"])
        c_ci_zero = 1 if (res_ctrl["ci_lo"] <= 0.0 <= res_ctrl["ci_hi"]) else 0

        gl.check(f"control_gain_rel_{f}", c_gain, "<=", "phase2.shuffle_control_max_rel_gain")
        gl.check(f"control_ci_includes_zero_{f}", c_ci_zero, "==", "global.bool_true")

        # Check real feature
        real_oof = pd.read_parquet(OOF_DIR / real_files[f])
        real_pair_df = pd.DataFrame({
            "cluster": real_oof["cluster"],
            "fold": real_oof["fold"],
            "y": real_oof["y"],
            "tail_thr": real_oof["tail_thr"],
            "pred_real": real_oof["pred"],
            "pred_f0": f0_oof["pred"],
        })
        res_real = gl.decide_pair(real_pair_df, advanced_col="pred_real", rival_col="pred_f0", metric="rmsle")
        real_results[f] = res_real

        adv_wins = bool(res_real["advanced_wins"])
        if adv_wins:
            recomputed_kept.append(f)
        else:
            recomputed_dropped.append(f)

        stored_f = stored_decisions["decisions"][f]
        if stored_f["advanced_wins"] != adv_wins or stored_f["kept"] != adv_wins:
            recomputed_matches = 0

    if (
        recomputed_kept != stored_decisions.get("kept", [])
        or recomputed_dropped != stored_decisions.get("dropped", [])
    ):
        recomputed_matches = 0

    gl.check("decision_recompute_matches", recomputed_matches, "==", "global.bool_true")

    # Kept features all have ci_lo > 0
    kept_all_ci_lo_gt_0 = 1
    for f in recomputed_kept:
        if real_results[f]["ci_lo"] <= 0:
            kept_all_ci_lo_gt_0 = 0
    gl.check("kept_features_all_have_ci_lo_gt_0", kept_all_ci_lo_gt_0, "==", "global.bool_true")

    # =========================================================================
    # 4. Informative Output
    # =========================================================================
    for f in features:
        rr = real_results[f]
        cr = ctrl_results[f]
        decision_str = "kept" if f in recomputed_kept else "dropped"
        gl.info(f"{f}_real_gain_rel", round(rr["relative_gain"], 6))
        gl.info(f"{f}_real_ci", [round(rr["ci_lo"], 6), round(rr["ci_hi"], 6)])
        gl.info(f"{f}_control_gain_rel", round(cr["relative_gain"], 6))
        gl.info(f"{f}_control_ci", [round(cr["ci_lo"], 6), round(cr["ci_hi"], 6)])
        gl.info(f"{f}_decision", decision_str)


def main() -> None:
    gl.run_gate("2", build_checks)


if __name__ == "__main__":
    main()
