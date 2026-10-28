"""Gate 6: SHAP explanations, planted-truth recovery, permutation cross-check, noise probe, and mutants."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

from gates import lib as gl
from revpred.explain import check_group_coverage, groups_for
from tests.mutants.mutants_phase6 import (
    run_mutant_a_validation_as_background,
    run_mutant_b_refit_model_mismatch,
    run_mutant_c_additivity_in_revenue_space,
    run_mutant_d_drop_one_hot_level,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
SHAP_DIR = REPORTS_DIR / "shap"
MODELS_DIR = REPO_ROOT / "models"


def build_checks() -> None:
    # =========================================================================
    # 1. Oracle Exactness on Planted Linear (Zero Noise)
    # =========================================================================
    oracle_phi = np.load(SHAP_DIR / "oracle_phi.npy")
    oracle_truth = np.load(SHAP_DIR / "oracle_truth.npy")
    oracle_err = float(np.max(np.abs(oracle_phi - oracle_truth)))
    gl.check("oracle_shap_max_abs_err", oracle_err, "<=", "phase6.oracle_shap_max_abs")

    # =========================================================================
    # 2. Library Cross-Check on Real Model
    # =========================================================================
    lib_shap = np.load(SHAP_DIR / "library_shap.npy")
    lib_linear = np.load(SHAP_DIR / "library_linear_shap.npy")
    lib_diff = float(np.max(np.abs(lib_shap - lib_linear)))
    gl.check("library_vs_analytic_max_abs", lib_diff, "<=", "phase6.oracle_shap_max_abs")

    # =========================================================================
    # 3. Additivity on Served Model (300 Dev Rows)
    # =========================================================================
    phi = np.load(SHAP_DIR / "phi.npy")
    base = float(np.load(SHAP_DIR / "base.npy")[0])
    preds = np.load(SHAP_DIR / "preds.npy")
    additivity_err = float(np.max(np.abs(np.sum(phi, axis=1) + base - preds)))
    gl.check("additivity_max_abs_err", additivity_err, "<=", "phase6.additivity_max_abs")

    # =========================================================================
    # 4. Learned Recovery on Planted Linear (Noise = 0.3, Ridge)
    # =========================================================================
    planted_phi = np.load(SHAP_DIR / "planted_linear_phi.npy")
    planted_true_imp = np.load(SHAP_DIR / "planted_linear_true_imp.npy")
    mean_abs_planted = np.mean(np.abs(planted_phi), axis=0)

    spr_planted = float(spearmanr(mean_abs_planted, planted_true_imp).statistic)
    gl.check("planted_rank_spearman", spr_planted, ">=", "phase6.planted_rank_spearman_min")

    # Noise features: indices 3, 4, 5 (X4, X5, X6)
    noise_share = float((mean_abs_planted[3] + mean_abs_planted[4] + mean_abs_planted[5]) / np.sum(mean_abs_planted))
    gl.check("planted_noise_share", noise_share, "<=", "phase6.planted_noise_share_max")

    # =========================================================================
    # 5. Interaction Blindness on Planted Interaction
    # =========================================================================
    inter_phi = np.load(SHAP_DIR / "planted_interaction_phi.npy")
    mean_abs_inter = np.mean(np.abs(inter_phi), axis=0)
    # Interaction features: indices 1, 2 (X2, X3)
    inter_share = float((mean_abs_inter[1] + mean_abs_inter[2]) / np.sum(mean_abs_inter))
    gl.check("interaction_blind_share", inter_share, "<=", "phase6.planted_noise_share_max")

    # =========================================================================
    # 6. Permutation Cross-Check on Model (b) Validation Rows
    # =========================================================================
    model_b_phi = np.load(SHAP_DIR / "model_b_phi.npy")
    model_b_perm = np.load(SHAP_DIR / "model_b_perm_importance.npy")
    with open(SHAP_DIR / "model_b_groups.json", encoding="utf-8") as f:
        groups_b = json.load(f)

    mean_abs_b = np.mean(np.abs(model_b_phi), axis=0)
    spr_perm = float(spearmanr(mean_abs_b, model_b_perm).statistic)
    gl.check("perm_vs_shap_spearman", spr_perm, ">=", "phase6.perm_vs_shap_spearman_min")

    shap_ranked = [g for _, g in sorted(zip(mean_abs_b, groups_b), reverse=True)]
    perm_ranked = [g for _, g in sorted(zip(model_b_perm, groups_b), reverse=True)]
    top3_overlap = len(set(shap_ranked[:3]).intersection(set(perm_ranked[:3])))
    gl.check("top3_overlap", top3_overlap, ">=", "phase6.top3_overlap_min")

    # =========================================================================
    # 7. Noise Probe on Model (b)
    # =========================================================================
    probe_phi = np.load(SHAP_DIR / "probe_phi.npy")
    with open(SHAP_DIR / "probe_groups.json", encoding="utf-8") as f:
        probe_groups = json.load(f)

    mean_abs_probe = np.mean(np.abs(probe_phi), axis=0)
    probe_scores = {g: float(v) for g, v in zip(probe_groups, mean_abs_probe)}
    sorted_probe = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)

    top_feature_score = sorted_probe[0][1]
    noise_scores = [probe_scores["noise_1"], probe_scores["noise_2"], probe_scores["noise_3"]]
    noise_max_rel = float(max(noise_scores) / top_feature_score)
    gl.check("noise_max_rel_to_top", noise_max_rel, "<=", "phase6.noise_feature_max_rel_to_top")

    ranked_probe_names = [g for g, _ in sorted_probe]
    top_5_probe = ranked_probe_names[:5]
    noise_in_top_5 = sum(1 for n in ["noise_1", "noise_2", "noise_3"] if n in top_5_probe)
    gl.check("noise_in_top_k", noise_in_top_5, "==", "global.zero")

    # =========================================================================
    # 8. Group Coverage and Preprocessor Structure
    # =========================================================================
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    groups_served = groups_for(model.preprocessor_)
    coverage_ok = 1 if check_group_coverage(model.preprocessor_, groups_served) else 0
    gl.check("group_coverage_ok", coverage_ok, "==", "global.bool_true")

    # =========================================================================
    # 9. Explain Manifest Integrity
    # =========================================================================
    manifest_path = MODELS_DIR / "EXPLAIN_MANIFEST.json"
    manifest_ok = 0
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        files = manifest.get("files", {})
        if len(files) == 3:
            all_valid = True
            for rel_path, exp_sha in files.items():
                p = REPO_ROOT / rel_path
                if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != exp_sha:
                    all_valid = False
                    break
            if all_valid:
                manifest_ok = 1
    gl.check("explain_manifest_ok", manifest_ok, "==", "global.bool_true")

    # =========================================================================
    # 10. Mutants Detection
    # =========================================================================
    row_ids = np.load(SHAP_DIR / "row_ids.npy", allow_pickle=True)
    expected_coef_sha = hashlib.sha256(
        np.ascontiguousarray(model.best_estimator_.coef_).tobytes()
    ).hexdigest()

    caught_a = 1 if not run_mutant_a_validation_as_background(row_ids) else 0
    caught_b = 1 if not run_mutant_b_refit_model_mismatch(model, expected_coef_sha) else 0
    caught_c = 1 if not run_mutant_c_additivity_in_revenue_space(phi, base, preds) else 0
    caught_d = 1 if not run_mutant_d_drop_one_hot_level(model.preprocessor_) else 0

    mutants_caught_frac = (caught_a + caught_b + caught_c + caught_d) / 4.0
    gl.check("mutants_caught_frac", mutants_caught_frac, ">=", "phase6.mutant_detection_min_frac")

    # =========================================================================
    # 11. Holdout Ledger Recovery Touches Check
    # =========================================================================
    ledger_path = REPORTS_DIR / "holdout_ledger.jsonl"
    lines = (
        [line.strip() for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if ledger_path.exists()
        else []
    )
    holdout_recovery_touches = len(lines) - 1
    gl.check("holdout_recovery_touches", holdout_recovery_touches, "<=", "phase5.holdout_recovery_touches_max")

    # =========================================================================
    # 12. Informative Outputs
    # =========================================================================
    with open(SHAP_DIR / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    mean_abs_served = np.mean(np.abs(phi), axis=0)
    top_10 = sorted(
        zip(feature_names, [round(float(v), 6) for v in mean_abs_served]),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    gl.info("top_10_features", [[g, v] for g, v in top_10])

    lag_names = ["lag_rev_1", "lag_rev_2", "lag_rev_3", "roll_mean_rev_3"]
    feat_to_val = dict(zip(feature_names, mean_abs_served))
    lag_total = round(float(sum(feat_to_val.get(f, 0.0) for f in lag_names)), 6)
    gl.info("lag_group_total", lag_total)

    with open(MODELS_DIR / "explain_background.json", encoding="utf-8") as f:
        bg_json = json.load(f)
    coef_vals = bg_json["coef"]
    group_map = bg_json["group_map"]
    coef_signs = {}
    for g in feature_names:
        g_c = [c for c, grp in zip(coef_vals, group_map) if grp == g]
        if all(c > 0 for c in g_c):
            coef_signs[g] = "+"
        elif all(c < 0 for c in g_c):
            coef_signs[g] = "-"
        else:
            coef_signs[g] = "mixed"
    gl.info("coefficient_signs", coef_signs)


def main() -> None:
    gl.run_gate("6", build_checks)


if __name__ == "__main__":
    main()

