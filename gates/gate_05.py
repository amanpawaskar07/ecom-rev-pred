"""Gate 5: Sealed holdout evaluation, recovery tracking, non-inferiority verification, and artifact validation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from gates import lib as gl

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"
MODELS_DIR = REPO_ROOT / "models"


def build_checks(
    reports_dir: Path | None = None,
    oof_dir: Path | None = None,
    models_dir: Path | None = None,
    is_dry_run: bool = False,
) -> None:
    rep_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    o_dir = Path(oof_dir) if oof_dir else OOF_DIR
    m_dir = Path(models_dir) if models_dir else MODELS_DIR

    # =========================================================================
    # 1. Holdout Recovery Ledger Checks
    # =========================================================================
    ledger_path = rep_dir / "holdout_ledger.jsonl"
    if not is_dry_run:
        lines = (
            [line.strip() for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if ledger_path.exists()
            else []
        )
        holdout_recovery_touches = len(lines) - 1
        gl.check("holdout_recovery_touches", holdout_recovery_touches, "<=", "phase5.holdout_recovery_touches_max")

        first_entry = json.loads(lines[0]) if len(lines) > 0 else {}
        first_touch_logged = 1 if first_entry.get("purpose") == "final_eval" else 0
        gl.check("first_touch_logged", first_touch_logged, "==", "global.bool_true")

        # First touch RMSLE reproducibility verification
        crash_file = REPORTS_DIR / "holdout_first_touch_crash.json"
        with open(crash_file, encoding="utf-8") as f:
            crash_data = json.load(f)
        crash_vals = crash_data["pooled_rmsle"]

        holdout_results_path = rep_dir / "holdout_results.json"
        with open(holdout_results_path, encoding="utf-8") as f:
            holdout_results = json.load(f)
        recovery_pooled = holdout_results.get("metrics", {}).get("pooled", {})

        first_touch_rmsle_max_abs_diff = max(
            abs(float(recovery_pooled[cand]["rmsle"]) - float(crash_vals[cand]))
            for cand in crash_vals
        )
        gl.check(
            "first_touch_rmsle_max_abs_diff",
            first_touch_rmsle_max_abs_diff,
            "<=",
            "phase5.first_touch_rmsle_repro_abs",
        )

        holdout_file = REPO_ROOT / "data" / "holdout" / "panel_holdout.parquet"
        seal_file = rep_dir / "holdout_seal.json" if (rep_dir / "holdout_seal.json").exists() else REPORTS_DIR / "holdout_seal.json"
        with open(seal_file, encoding="utf-8") as f:
            seal_data = json.load(f)

        actual_sha = hashlib.sha256(holdout_file.read_bytes()).hexdigest()
        holdout_seal_ok = 1 if actual_sha == seal_data["sha256"] else 0
        gl.check("holdout_seal_ok", holdout_seal_ok, "==", "global.bool_true")
    else:
        # In dry run mode, ledger checks evaluate trivially against mocked values
        gl.check("holdout_recovery_touches", 0, "<=", "phase5.holdout_recovery_touches_max")
        gl.check("first_touch_logged", 1, "==", "global.bool_true")
        gl.check("first_touch_rmsle_max_abs_diff", 0.0, "<=", "phase5.first_touch_rmsle_repro_abs")
        gl.check("holdout_seal_ok", 1, "==", "global.bool_true")
        with open(rep_dir / "holdout_results.json", encoding="utf-8") as f:
            holdout_results = json.load(f)

    # =========================================================================
    # 2. Final Evaluation Tag Cleanliness
    # =========================================================================
    res_diff = subprocess.run(
        ["git", "diff", "--name-only", "phase5-frozen"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    diff_files = [line.strip() for line in res_diff.stdout.strip().splitlines() if line.strip()]

    allowed_prefixes = ("reports/", "models/")
    allowed_new_files = {
        "AGENTS.md",
        "gates/thresholds.yaml",
        "gates/thresholds.sha256",
        "src/revpred/final_eval_core.py",
        "tests/test_phase5_final_eval.py",
        "scripts/final_eval.py",
        "gates/gate_05.py",
    }

    final_eval_tag_clean = 1
    for f_path in diff_files:
        if f_path.startswith(allowed_prefixes) or f_path in allowed_new_files:
            continue
        final_eval_tag_clean = 0
        break

    gl.check("final_eval_tag_clean", final_eval_tag_clean, "==", "global.bool_true")

    # =========================================================================
    # 3. Baseline Beat Margins
    # =========================================================================
    pairwise = holdout_results.get("pairwise", {})

    # Final vs Mean
    p_mean = pairwise.get("final vs baseline_mean", {})
    gl.check("beat_margin_rel_mean", float(p_mean.get("relative_gain", 0.0)), ">=", "phase5.baseline_beat_margin_rel")
    gl.check("gain_ci_lo_mean", float(p_mean.get("ci_lo", 0.0)), ">", "global.zero")
    gl.check("months_favoring_frac_mean", float(p_mean.get("months_favoring_frac", 0.0)), ">=", "global.folds_favoring_frac")

    # Final vs Persistence
    p_pers = pairwise.get("final vs baseline_persistence", {})
    gl.check("beat_margin_rel_persistence", float(p_pers.get("relative_gain", 0.0)), ">=", "phase5.baseline_beat_margin_rel")
    gl.check("gain_ci_lo_persistence", float(p_pers.get("ci_lo", 0.0)), ">", "global.zero")
    gl.check("months_favoring_frac_persistence", float(p_pers.get("months_favoring_frac", 0.0)), ">=", "global.folds_favoring_frac")

    # =========================================================================
    # 4. Ridge Non-inferiority Worsening Recompute
    # =========================================================================
    df_final = pd.read_parquet(o_dir / "holdout_final.parquet")
    df_ridge = pd.read_parquet(o_dir / "holdout_baseline_ridge.parquet")

    final_rmsle = gl.rmsle(df_final["y"], df_final["pred"])
    ridge_rmsle = gl.rmsle(df_ridge["y"], df_ridge["pred"])
    ridge_worsening_rel = float((final_rmsle - ridge_rmsle) / ridge_rmsle) if ridge_rmsle > 0 else 0.0

    gl.check("ridge_worsening_rel", ridge_worsening_rel, "<=", "phase5.ridge_noninferiority_max_rel")

    # =========================================================================
    # 5. Metrics Recompute Max Relative Difference
    # =========================================================================
    candidates = ["final", "challenger", "baseline_ridge", "baseline_mean", "baseline_persistence"]
    stored_metrics = holdout_results.get("metrics", {})
    stored_pooled = stored_metrics.get("pooled", {})
    stored_per_m = stored_metrics.get("per_month", {})
    tail_thr = float(holdout_results.get("tail_threshold", 0.0))

    max_rel_diff = 0.0

    for cand in candidates:
        cand_oof = pd.read_parquet(o_dir / f"holdout_{cand}.parquet")
        y_all = cand_oof["y"].to_numpy(dtype=float)
        p_all = cand_oof["pred"].to_numpy(dtype=float)

        c_rmsle = gl.rmsle(y_all, p_all)
        c_mae = gl.mae(y_all, p_all)
        c_wape = gl.wape(y_all, p_all)
        c_tail_mae = gl.tail_mae(y_all, p_all, tail_thr)

        st_p = stored_pooled.get(cand, {})
        for metric_name, computed_val in [
            ("rmsle", c_rmsle),
            ("mae", c_mae),
            ("wape", c_wape),
            ("tail_mae", c_tail_mae),
        ]:
            stored_val = float(st_p.get(metric_name, 0.0))
            if stored_val > 0:
                rel_d = abs(computed_val - stored_val) / stored_val
                max_rel_diff = max(max_rel_diff, rel_d)

        # Per-month
        for m, m_group in cand_oof.groupby("month"):
            y_m = m_group["y"].to_numpy(dtype=float)
            p_m = m_group["pred"].to_numpy(dtype=float)
            cm_rmsle = gl.rmsle(y_m, p_m)
            cm_mae = gl.mae(y_m, p_m)
            cm_wape = gl.wape(y_m, p_m)
            cm_tail_mae = gl.tail_mae(y_m, p_m, tail_thr)

            st_m = stored_per_m.get(str(m), {}).get(cand, {})
            for metric_name, computed_val in [
                ("rmsle", cm_rmsle),
                ("mae", cm_mae),
                ("wape", cm_wape),
                ("tail_mae", cm_tail_mae),
            ]:
                stored_val = float(st_m.get(metric_name, 0.0))
                if stored_val > 0:
                    rel_d = abs(computed_val - stored_val) / stored_val
                    max_rel_diff = max(max_rel_diff, rel_d)

    gl.check("metrics_recompute_max_rel_diff", max_rel_diff, "<=", "global.repro_rel_tol")

    # =========================================================================
    # 6. Model Manifest Integrity
    # =========================================================================
    manifest_path = m_dir / "MANIFEST.json"
    manifest_ok = 0
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        files_valid = True
        for m_file, expected_sha in manifest.get("files", {}).items():
            # In dry-run mode, look in m_dir
            if is_dry_run:
                mf_path = m_dir / Path(m_file).name
            else:
                mf_path = REPO_ROOT / m_file
            if not mf_path.exists() or hashlib.sha256(mf_path.read_bytes()).hexdigest() != expected_sha:
                files_valid = False
                break

        versions = manifest.get("versions", {})
        has_versions = bool(
            versions.get("python")
            and versions.get("scikit_learn")
            and versions.get("xgboost")
            and versions.get("shap")
        )

        final_cfg_path = rep_dir / "final_config.json" if (rep_dir / "final_config.json").exists() else REPORTS_DIR / "final_config.json"
        with open(final_cfg_path, encoding="utf-8") as f:
            final_cfg = json.load(f)

        cfg_sha_match = manifest.get("config_sha256") == final_cfg.get("sha256_config")
        src_sha_match = manifest.get("sha256_src") == final_cfg.get("sha256_src")

        if files_valid and has_versions and cfg_sha_match and src_sha_match:
            manifest_ok = 1

    gl.check("model_manifest_ok", manifest_ok, "==", "global.bool_true")

    # =========================================================================
    # 7. Non-Gating Informative Outputs
    # =========================================================================
    p_chall_ridge = pairwise.get("challenger vs baseline_ridge", {})
    gl.info("challenger_vs_baseline_ridge_gain_rel", round(float(p_chall_ridge.get("relative_gain", 0.0)), 6))
    gl.info(
        "challenger_vs_baseline_ridge_ci",
        [round(float(p_chall_ridge.get("ci_lo", 0.0)), 6), round(float(p_chall_ridge.get("ci_hi", 0.0)), 6)],
    )
    gl.info("challenger_vs_baseline_ridge_months_fav", p_chall_ridge.get("months_favoring"))

    p_chall_final = pairwise.get("challenger vs final", {})
    gl.info("challenger_vs_final_gain_rel", round(float(p_chall_final.get("relative_gain", 0.0)), 6))
    gl.info(
        "challenger_vs_final_ci",
        [round(float(p_chall_final.get("ci_lo", 0.0)), 6), round(float(p_chall_final.get("ci_hi", 0.0)), 6)],
    )
    gl.info("challenger_vs_final_months_fav", p_chall_final.get("months_favoring"))

    gl.info("claim_final_vs_ridge", holdout_results.get("claim_final_vs_ridge"))
    gl.info("claim_challenger_vs_ridge", holdout_results.get("claim_challenger_vs_ridge"))


def main() -> None:
    gl.run_gate("5", build_checks)


if __name__ == "__main__":
    main()
