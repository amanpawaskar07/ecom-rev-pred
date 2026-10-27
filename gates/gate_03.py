"""Gate 3: Quantile target transformation, SMOTER resampling, harness validation, and decisions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates import lib as gl
from revpred.harness import make_planted_imbalance
from revpred.resample import SmoterResampler
from revpred.targets import QuantileTargetTransform
from tests.mutants.mutants_phase3 import (
    QuantileTargetTransformMutantB,
    oversample_before_split_mutant_a,
    smoter_all_neighbours_mutant_c,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"


def build_checks() -> None:
    """Build and evaluate all Gate 3 checks."""
    # =========================================================================
    # 1. Quantile Roundtrip and Point Mass Invariants
    # =========================================================================
    panel_dev = pd.read_parquet(REPO_ROOT / "data" / "processed" / "panel_dev.parquet")
    y_dev = panel_dev["revenue"].to_numpy(dtype=float)
    pos_mask = y_dev > 0

    qt = QuantileTargetTransform(random_state=42)
    qt.fit(y_dev)
    y_trans = qt.transform(y_dev)
    y_inv = qt.inverse(y_trans)

    rel_errs = np.abs(y_dev[pos_mask] - y_inv[pos_mask]) / y_dev[pos_mask]
    roundtrip_max_rel_err = float(np.max(rel_errs))
    gl.check("roundtrip_max_rel_err", roundtrip_max_rel_err, "<=", "phase3.roundtrip_max_rel_err")

    # =========================================================================
    # 2. SMOTER Invariants
    # =========================================================================
    X_toy, y_toy, _ = make_planted_imbalance(seed=42, learnable=True, n=1000, d=6)
    train_idx_toy = np.arange(len(X_toy))
    res_toy = SmoterResampler().fit_resample(X_toy, y_toy, train_idx=train_idx_toy, seed=42)

    smoter_interp_max_abs_err = 0.0
    n_orig = len(X_toy)
    for i in range(res_toy.n_syn):
        id_a, id_b = res_toy.parent_ids[i]
        lam = res_toy.lambdas[i]
        expected_y = y_toy[id_a] + lam * (y_toy[id_b] - y_toy[id_a])
        actual_y = res_toy.y[n_orig + i]
        err = abs(actual_y - expected_y)
        smoter_interp_max_abs_err = max(smoter_interp_max_abs_err, err)

    gl.check("smoter_interp_max_abs_err", smoter_interp_max_abs_err, "<=", "phase3.smoter_interp_abs_tol")

    # =========================================================================
    # 3. Mutant Detection Fraction
    # =========================================================================
    mutants_caught = 0

    # Mutant A: oversample before split caught under validation perturbation
    n_tr = 750
    X_tr, y_tr = X_toy[:n_tr], y_toy[:n_tr]
    X_val, y_val = X_toy[n_tr:], y_toy[n_tr:]
    y_val_pert = y_val * 100.0

    X_comb_1 = np.vstack([X_tr, X_val])
    y_comb_1 = np.concatenate([y_tr, y_val])
    _, _, y_tr_m1, _, _ = oversample_before_split_mutant_a(X_comb_1, y_comb_1, test_size=0.25, seed=42)

    X_comb_2 = np.vstack([X_tr, X_val])
    y_comb_2 = np.concatenate([y_tr, y_val_pert])
    _, _, y_tr_m2, _, _ = oversample_before_split_mutant_a(X_comb_2, y_comb_2, test_size=0.25, seed=42)

    if not np.array_equal(y_tr_m1, y_tr_m2):
        mutants_caught += 1

    # Mutant B: quantile fit on all rows caught under validation perturbation
    mb1 = QuantileTargetTransformMutantB(random_state=42).fit_all(y_tr, y_val)
    tr_mb1 = mb1.transform(y_tr)
    mb2 = QuantileTargetTransformMutantB(random_state=42).fit_all(y_tr, y_val_pert)
    tr_mb2 = mb2.transform(y_tr)
    if not np.array_equal(tr_mb1, tr_mb2):
        mutants_caught += 1

    # Mutant C: neighbours drawn from all rows caught under validation perturbation
    X_val_pert = X_val * 10.0
    _, y_mc1 = smoter_all_neighbours_mutant_c(X_tr, y_tr, X_val, y_val, train_idx=np.arange(n_tr), seed=42)
    _, y_mc2 = smoter_all_neighbours_mutant_c(X_tr, y_tr, X_val_pert, y_val_pert, train_idx=np.arange(n_tr), seed=42)
    if not np.array_equal(y_mc1, y_mc2):
        mutants_caught += 1

    mutants_caught_frac = mutants_caught / 3.0
    gl.check("mutants_caught_frac", mutants_caught_frac, ">=", "phase3.mutant_detection_min_frac")

    # =========================================================================
    # 4. Harness Imbalance Results
    # =========================================================================
    with open(REPORTS_DIR / "harness_imbalance.json", encoding="utf-8") as f:
        harness_rep = json.load(f)

    tail_bias_reduction_rel = harness_rep["learnable"]["mean_tail_bias_reduction_rel"]
    tail_mae_gain_rel = harness_rep["learnable"]["mean_tail_mae_gain_rel"]
    bulk_rmsle_worsening_rel = harness_rep["learnable"]["mean_bulk_rmsle_worsening_rel"]
    unlearnable_tail_mae_gain_rel = harness_rep["unlearnable"]["mean_tail_mae_gain_rel"]
    leak_mutant_inflation_ratio = harness_rep["leak_mutant"]["leak_mutant_inflation_ratio"]

    gl.check("tail_bias_reduction_rel", tail_bias_reduction_rel, ">=", "phase3.planted_tail_bias_reduction_min_rel")
    gl.check("tail_mae_gain_rel", tail_mae_gain_rel, ">=", "phase3.planted_tail_mae_gain_min_rel")
    gl.check("bulk_rmsle_worsening_rel", bulk_rmsle_worsening_rel, "<=", "metrics.bulk_noninferiority_rel")
    gl.check("unlearnable_tail_mae_gain_rel", unlearnable_tail_mae_gain_rel, "<=", "phase3.null_tail_gain_max_rel")
    gl.check("leak_mutant_inflation_ratio", leak_mutant_inflation_ratio, ">=", "phase3.leak_mutant_inflation_min_ratio")

    # =========================================================================
    # 5. Decision Recompute from OOF Parquets
    # =========================================================================
    with open(REPORTS_DIR / "imbalance_decision.json", encoding="utf-8") as f:
        stored_imb = json.load(f)

    # 5a. Recompute target decision
    oof_t0 = pd.read_parquet(OOF_DIR / "imb_t0_log1p.parquet")
    oof_t1 = pd.read_parquet(OOF_DIR / "imb_t1_quantile.parquet")
    target_pair_df = pd.DataFrame({
        "cluster": oof_t0["cluster"],
        "fold": oof_t0["fold"],
        "y": oof_t0["y"],
        "tail_thr": oof_t0["tail_thr"],
        "pred_adv": oof_t1["pred"],
        "pred_rival": oof_t0["pred"],
    })
    recomputed_target_res = gl.decide_pair(
        target_pair_df, advanced_col="pred_adv", rival_col="pred_rival", metric="rmsle"
    )
    recomputed_target_winner = "quantile" if recomputed_target_res["advanced_wins"] else "log1p"

    # 5b. Recompute resampling decision
    oof_r0 = pd.read_parquet(OOF_DIR / "imb_r0_none.parquet")
    oof_r1 = pd.read_parquet(OOF_DIR / "imb_r1_weights.parquet")
    oof_r2 = pd.read_parquet(OOF_DIR / "imb_r2_smoter.parquet")

    resample_df = pd.DataFrame({
        "cluster": oof_r0["cluster"],
        "fold": oof_r0["fold"],
        "y": oof_r0["y"],
        "tail_thr": oof_r0["tail_thr"],
        "none": oof_r0["pred"],
        "weights": oof_r1["pred"],
        "smoter": oof_r2["pred"],
    })
    candidates = ["none", "weights", "smoter"]
    recomputed_among = gl.decide_among(
        resample_df, candidates_in_cost_order=candidates, metric="tail_mae"
    )

    # Bulk non-inferiority
    folds = np.sort(resample_df["fold"].unique())
    rmsle_by_cand = {}
    for c in candidates:
        scores = [gl.rmsle(resample_df.loc[resample_df["fold"] == f, "y"], resample_df.loc[resample_df["fold"] == f, c]) for f in folds]
        rmsle_by_cand[c] = float(np.mean(scores))
    best_primary = min(rmsle_by_cand.values())

    bulk_threshold = float(gl.thr("metrics.bulk_noninferiority_rel"))
    recomputed_resampler = recomputed_among.selected
    curr_idx = candidates.index(recomputed_resampler)
    while curr_idx >= 0:
        cand = candidates[curr_idx]
        worsening = (rmsle_by_cand[cand] - best_primary) / best_primary
        if worsening <= bulk_threshold:
            recomputed_resampler = cand
            break
        curr_idx -= 1
    if curr_idx < 0:
        recomputed_resampler = "none"

    recomputed_matches = 1
    if (
        recomputed_target_winner != stored_imb["winning_target"]
        or recomputed_resampler != stored_imb["chosen_resampler"]
    ):
        recomputed_matches = 0

    gl.check("decision_recompute_matches", recomputed_matches, "==", "global.bool_true")

    # =========================================================================
    # 6. Informative Output
    # =========================================================================
    gl.info("target_real_gain_rel", round(float(recomputed_target_res["relative_gain"]), 6))
    gl.info("target_real_ci", [round(float(recomputed_target_res["ci_lo"]), 6), round(float(recomputed_target_res["ci_hi"]), 6)])
    gl.info("chosen_target_transform", stored_imb["winning_target"])

    weights_ev = recomputed_among.evidence["pairwise_vs_best"].get("weights", {})
    if "relative_gain" in weights_ev:
        gl.info("resample_weights_tail_gain_rel", round(float(weights_ev["relative_gain"]), 6))
        gl.info("resample_weights_tail_ci", [round(float(weights_ev["ci_lo"]), 6), round(float(weights_ev["ci_hi"]), 6)])
    else:
        gl.info("resample_weights_tail_gain_rel", 0.0)
        gl.info("resample_weights_tail_ci", [0.0, 0.0])

    smoter_ev = recomputed_among.evidence["pairwise_vs_best"].get("none", {})
    if "relative_gain" in smoter_ev:
        gl.info("resample_smoter_tail_gain_rel", round(float(smoter_ev["relative_gain"]), 6))
        gl.info("resample_smoter_tail_ci", [round(float(smoter_ev["ci_lo"]), 6), round(float(smoter_ev["ci_hi"]), 6)])
    else:
        gl.info("resample_smoter_tail_gain_rel", 0.0)
        gl.info("resample_smoter_tail_ci", [0.0, 0.0])

    gl.info("chosen_resampler", stored_imb["chosen_resampler"])


def main() -> None:
    gl.run_gate("3", build_checks)


if __name__ == "__main__":
    main()
