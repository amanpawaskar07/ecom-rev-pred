"""Gate 4: Temporal stacking, OOF causality, harness validation, and decision rules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates import lib as gl
from tests.mutants.mutants_phase4 import (
    AllRowsFitStackerMutantB,
    InSampleMetaStackerMutantC,
    ShuffledKFoldStackerMutantA,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"


def build_checks() -> None:
    # =========================================================================
    # 1. OOF Causality & Min Meta Train Rows
    # =========================================================================
    ablation_stack_path = REPORTS_DIR / "ablation_stack.json"
    with open(ablation_stack_path, encoding="utf-8") as f:
        ablation_rep = json.load(f)

    oof_causality_violations = ablation_rep["oof_causality_violations"]
    min_meta_train_rows = ablation_rep["min_meta_train_rows"]

    gl.check(
        "oof_causality_violations",
        oof_causality_violations,
        "<=",
        "phase4.oof_causality_violations_max",
    )
    gl.check(
        "min_meta_train_rows",
        min_meta_train_rows,
        ">=",
        "phase4.min_meta_train_rows",
    )

    # =========================================================================
    # 2. Mutant Detection Fraction
    # =========================================================================
    months = ["2017-01", "2017-02", "2017-03", "2017-04", "2017-05"]
    n_per_m = 60
    n_toy = len(months) * n_per_m
    rng = np.random.default_rng(42)
    df_toy = pd.DataFrame({
        "row_id": np.arange(n_toy),
        "month": np.repeat(months, n_per_m),
        "f1": rng.standard_normal(n_toy),
        "f2": rng.standard_normal(n_toy),
        "revenue": np.abs(rng.standard_normal(n_toy)) * 100.0,
    })

    mutants_caught = 0

    # Mutant A: Shuffled KFold
    m_a = ShuffledKFoldStackerMutantA(min_inner_train_months=3, features=["f1", "f2"]).fit(df_toy)
    viol_a = int((m_a.oof_log["fit_max_month"] >= m_a.oof_log["month"]).sum())
    if viol_a > 0:
        mutants_caught += 1

    # Mutant B: All rows fit
    m_b = AllRowsFitStackerMutantB(min_inner_train_months=3, features=["f1", "f2"]).fit(df_toy)
    viol_b = int((m_b.oof_log["fit_max_month"] >= m_b.oof_log["month"]).sum())
    if viol_b > 0:
        mutants_caught += 1

    # Mutant C: In-sample meta
    m_c = InSampleMetaStackerMutantC(min_inner_train_months=3, features=["f1", "f2"]).fit(df_toy)
    viol_c = int((m_c.oof_log["fit_max_month"] >= m_c.oof_log["month"]).sum())
    if viol_c > 0:
        mutants_caught += 1

    mutants_caught_frac = mutants_caught / 3.0
    gl.check("mutants_caught_frac", mutants_caught_frac, ">=", "phase4.mutant_detection_min_frac")

    # =========================================================================
    # 3. Linear-Truth Harness Stack Gain
    # =========================================================================
    harness_path = REPORTS_DIR / "harness_stack.json"
    with open(harness_path, encoding="utf-8") as f:
        harness_rep = json.load(f)

    linear_truth_stack_gain_rel = harness_rep["linear_truth_stack_gain_rel"]
    gl.check(
        "linear_truth_stack_gain_rel",
        linear_truth_stack_gain_rel,
        "<=",
        "phase4.linear_truth_stack_gain_max_rel",
    )

    # =========================================================================
    # 4. Forbidden Splitters in stack.py
    # =========================================================================
    stack_code = (REPO_ROOT / "src" / "revpred" / "stack.py").read_text(encoding="utf-8")
    forbidden_terms = ["KFold", "ShuffleSplit", "StackingRegressor"]
    forbidden_count = sum(term in stack_code for term in forbidden_terms)
    gl.check("forbidden_splitters_in_stack", forbidden_count, "==", "global.zero")

    # =========================================================================
    # 5. Decision Recompute from OOF Parquets
    # =========================================================================
    stack_dec_path = REPORTS_DIR / "stack_decision.json"
    with open(stack_dec_path, encoding="utf-8") as f:
        stored_dec = json.load(f)

    candidates = [
        "ridge",
        "xgb",
        "rf",
        "mean_blend",
        "stack_ridge_meta",
        "stack_xgb_meta",
    ]

    oof_dict: dict[str, pd.DataFrame] = {}
    for c in candidates:
        fn = f"{c}.parquet" if c.startswith("stack_") else f"stack_{c}.parquet"
        oof_dict[c] = pd.read_parquet(OOF_DIR / fn)

    df_eval = pd.DataFrame({
        "cluster": oof_dict["ridge"]["cluster"],
        "fold": oof_dict["ridge"]["fold"],
        "y": oof_dict["ridge"]["y"],
        "tail_thr": oof_dict["ridge"]["tail_thr"],
        "ridge": oof_dict["ridge"]["pred"],
        "xgb": oof_dict["xgb"]["pred"],
        "rf": oof_dict["rf"]["pred"],
        "mean_blend": oof_dict["mean_blend"]["pred"],
        "stack_ridge_meta": oof_dict["stack_ridge_meta"]["pred"],
        "stack_xgb_meta": oof_dict["stack_xgb_meta"]["pred"],
    })

    recomputed_decision = gl.decide_among(
        df_eval,
        candidates_in_cost_order=candidates,
        metric="rmsle",
    )

    recomputed_matches = 1 if recomputed_decision.selected == stored_dec["default_model"] else 0
    gl.check("decision_recompute_matches", recomputed_matches, "==", "global.bool_true")

    # =========================================================================
    # 5b. Ridge Rival Equivalence to Baseline Ridge
    # =========================================================================
    stack_ridge_path = OOF_DIR / "stack_ridge.parquet"
    baseline_ridge_path = OOF_DIR / "baseline_ridge.parquet"

    df_stack_ridge = pd.read_parquet(stack_ridge_path)
    df_baseline_ridge = pd.read_parquet(baseline_ridge_path)

    merged_ridge = df_stack_ridge[["row_id", "pred"]].merge(
        df_baseline_ridge[["row_id", "pred"]],
        on="row_id",
        suffixes=("_stack", "_baseline"),
    )
    ridge_rival_vs_baseline_max_abs_diff = float(
        np.max(np.abs(merged_ridge["pred_stack"] - merged_ridge["pred_baseline"]))
    )
    gl.check(
        "ridge_rival_vs_baseline_max_abs_diff",
        ridge_rival_vs_baseline_max_abs_diff,
        "<=",
        "global.repro_rel_tol",
    )

    # =========================================================================
    # 6. Informative Output Table
    # =========================================================================
    best_cand = recomputed_decision.evidence["best_candidate"]
    gl.info("best_candidate_by_metric", best_cand)
    gl.info("default_model", recomputed_decision.selected)

    for c in candidates:
        metrics = ablation_rep["metrics"][c]["pooled"]
        mean_fold_m = recomputed_decision.evidence["mean_fold_metrics"][c]
        ev_best = recomputed_decision.evidence["pairwise_vs_best"].get(c, {})
        rel_gain_best = ev_best.get("relative_gain", 0.0)
        ci_lo = ev_best.get("ci_lo", 0.0)
        ci_hi = ev_best.get("ci_hi", 0.0)
        fav_frac = ev_best.get("folds_favoring_frac", 0.0)

        gl.info(
            f"candidate_{c}",
            f"pooled_rmsle={metrics['rmsle']:.4f} mae={metrics['mae']:.2f} wape={metrics['wape']:.4f} "
            f"tail_mae={metrics['tail_mae']:.2f} | mean_fold_rmsle={mean_fold_m:.4f} | "
            f"vs_best(gain={rel_gain_best:.4f}, ci=[{ci_lo:.4f}, {ci_hi:.4f}], folds_fav={fav_frac:.2f})",
        )

    # Specific pairwise info for stack comparisons
    p_xgb = ablation_rep["pairwise_vs_stack_xgb_meta"]["stack_xgb_meta vs xgb"]
    gl.info("stack_xgb_meta_vs_xgb_gain_rel", round(float(p_xgb["relative_gain"]), 6))
    gl.info("stack_xgb_meta_vs_xgb_ci", [round(float(p_xgb["ci_lo"]), 6), round(float(p_xgb["ci_hi"]), 6)])

    p_ridge = ablation_rep["pairwise_vs_stack_xgb_meta"]["stack_xgb_meta vs ridge"]
    gl.info("stack_xgb_meta_vs_ridge_gain_rel", round(float(p_ridge["relative_gain"]), 6))
    gl.info("stack_xgb_meta_vs_ridge_ci", [round(float(p_ridge["ci_lo"]), 6), round(float(p_ridge["ci_hi"]), 6)])


def main() -> None:
    gl.run_gate("4", build_checks)


if __name__ == "__main__":
    main()
