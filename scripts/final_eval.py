"""Phase 5 Stage B: Sealed holdout evaluation and artifact packaging.

This script executes THE SINGLE AUTHORIZED HOLDOUT RECOVERY TOUCH.
Pre-flight checks verify repository integrity against phase5-frozen tag before holdout access.
Supports `--dry-run` to execute the full evaluation pipeline on dev stand-in rows without touching holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb

from gates import lib as gl
from revpred.config import get_project_config, get_thresholds
from revpred.final_eval_core import run_eval
from revpred.holdout import load_holdout

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"
MODELS_DIR = REPO_ROOT / "models"
FINAL_CONFIG_PATH = REPORTS_DIR / "final_config.json"
DEV_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"


def verify_preflight_checks(is_dry_run: bool = False) -> dict[str, Any]:
    """Verify repository status, clean git diff vs phase5-frozen, holdout ledger, and checksums."""
    print("=== Pre-flight Verifications ===")

    # (a) Verify tag phase5-frozen exists
    res_tag = subprocess.run(
        ["git", "tag", "-l", "phase5-frozen"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if "phase5-frozen" not in res_tag.stdout.strip().split():
        print("ERROR: Tag 'phase5-frozen' does not exist! Exit code 2.")
        sys.exit(2)
    print("  [OK] Tag phase5-frozen exists.")

    # (b) Verify git diff phase5-frozen -- src config reports/final_config.json is empty
    res_diff = subprocess.run(
        ["git", "diff", "phase5-frozen", "--", "src", "config", "reports/final_config.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res_diff.stdout.strip():
        print("ERROR: Tracked files under src, config, or final_config.json differ from phase5-frozen!")
        print(res_diff.stdout[:500])
        sys.exit(2)
    print("  [OK] git diff vs phase5-frozen is clean.")

    # (c) Ledger and output file checks
    ledger_path = REPORTS_DIR / "holdout_ledger.jsonl"
    results_path = REPORTS_DIR / "holdout_results.json"

    if not is_dry_run:
        # Recovery touch requires exactly ONE prior line in holdout_ledger.jsonl with purpose 'final_eval'
        if not ledger_path.exists():
            print("ERROR: reports/holdout_ledger.jsonl does not exist! Expected exactly 1 prior line. Exit code 2.")
            sys.exit(2)

        lines = [line.strip() for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            print(f"ERROR: reports/holdout_ledger.jsonl contains {len(lines)} lines, expected exactly 1! Exit code 2.")
            sys.exit(2)

        first_entry = json.loads(lines[0])
        if first_entry.get("purpose") != "final_eval":
            print(f"ERROR: First ledger entry purpose is '{first_entry.get('purpose')}', expected 'final_eval'! Exit code 2.")
            sys.exit(2)
        print("  [OK] holdout_ledger.jsonl verified (exactly 1 prior touch with purpose 'final_eval').")

        # Must have no existing holdout_results.json
        if results_path.exists():
            print(f"ERROR: {results_path} already exists! Exit code 2.")
            sys.exit(2)
        print("  [OK] reports/holdout_results.json does not exist.")
    else:
        print("  [DRY-RUN] Skipping ledger enforcement; no holdout will be read.")

    # (d) Verify final_config_hashes_match_files recomputes OK
    with open(FINAL_CONFIG_PATH, encoding="utf-8") as f:
        final_cfg = json.load(f)

    recorded_hashes = final_cfg.get("file_hashes", {})
    for rel_path, expected_sha in recorded_hashes.items():
        fp = REPO_ROOT / rel_path
        if not fp.exists():
            print(f"ERROR: File {rel_path} in file_hashes does not exist! Exit code 2.")
            sys.exit(2)
        actual_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            print(f"ERROR: File {rel_path} checksum mismatch: {actual_sha} != {expected_sha}! Exit code 2.")
            sys.exit(2)

    manifest = "".join(f"{k}:{v}\n" for k, v in sorted(recorded_hashes.items()))
    recomputed_sha256 = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    if recomputed_sha256 != final_cfg.get("sha256"):
        print(f"ERROR: sha256 mismatch: {recomputed_sha256} != {final_cfg.get('sha256')}! Exit code 2.")
        sys.exit(2)

    print("  [OK] final_config hashes verified against disk.")
    return final_cfg


def evaluate_claim(
    rel_gain: float,
    ci_lo: float,
    months_fav_frac: float,
    beat_margin: float,
    folds_fav_frac: float,
) -> str:
    """Pre-registered decision logic for claiming 'beats' vs 'does not beat'."""
    if rel_gain >= beat_margin and ci_lo > 0 and months_fav_frac >= folds_fav_frac:
        return "beats"
    return "does not beat"


def execute_pipeline(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    final_cfg: dict[str, Any],
    reports_out_dir: Path,
    oof_out_dir: Path,
    models_out_dir: Path,
    is_dry_run: bool = False,
) -> dict[str, Any]:
    """Execute complete evaluation, bootstrap, breakdown, claim generation, and artifact saving."""
    cfg = get_project_config()
    thr = get_thresholds()

    beat_margin = float(thr["phase5"]["baseline_beat_margin_rel"])
    folds_fav_frac = float(thr["global"]["folds_favoring_frac"])
    n_bootstrap = int(thr["global"]["n_bootstrap"])
    seed = int(thr["global"]["seed"])

    print("Running run_eval...")
    eval_result = run_eval(
        train_df=train_df,
        eval_df=eval_df,
        final_config=final_cfg,
    )

    preds_df = eval_result["predictions"]
    pooled_metrics = eval_result["pooled"]
    per_month_metrics = eval_result["per_month"]
    tail_thr = eval_result["tail_threshold"]
    fitted_models = eval_result["models"]

    print("\n--- Pooled Metrics ---")
    for m_name, m_vals in pooled_metrics.items():
        print(
            f"  {m_name:22s}: RMSLE={m_vals['rmsle']:.4f}  MAE={m_vals['mae']:.2f}  "
            f"WAPE={m_vals['wape']:.4f}  Tail_MAE={m_vals['tail_mae']:.2f}"
        )

    # Pairwise paired cluster bootstrap evaluations
    print("\nRunning paired cluster bootstrap over product series...")
    unique_eval_months = sorted(preds_df["month"].unique())
    n_eval_months = len(unique_eval_months)

    pairs = [
        ("final", "baseline_mean"),
        ("final", "baseline_persistence"),
        ("final", "baseline_ridge"),
        ("challenger", "baseline_ridge"),
        ("challenger", "final"),
    ]

    pairwise_results: dict[str, Any] = {}

    for cand_a, cand_b in pairs:
        mean_diff, ci_lo, ci_hi, per_fold_diff, folds_fav = gl.paired_cluster_bootstrap(
            preds_df,
            metric="rmsle",
            a_col=cand_a,
            b_col=cand_b,
            n=n_bootstrap,
            seed=seed,
        )

        m_a_vals = [per_month_metrics[m][cand_a]["rmsle"] for m in unique_eval_months]
        m_b_vals = [per_month_metrics[m][cand_b]["rmsle"] for m in unique_eval_months]
        per_month_gains = [(mb - ma) / mb for ma, mb in zip(m_a_vals, m_b_vals)]

        mean_b = float(np.mean(m_b_vals))
        mean_a = float(np.mean(m_a_vals))
        rel_gain = float((mean_b - mean_a) / mean_b) if mean_b > 0 else 0.0
        gain_ci_lo = float(ci_lo / mean_b) if mean_b > 0 else 0.0
        gain_ci_hi = float(ci_hi / mean_b) if mean_b > 0 else 0.0
        months_fav_frac = float(folds_fav / n_eval_months)

        pair_key = f"{cand_a} vs {cand_b}"
        pairwise_results[pair_key] = {
            "candidate": cand_a,
            "rival": cand_b,
            "relative_gain": rel_gain,
            "ci_lo": gain_ci_lo,
            "ci_hi": gain_ci_hi,
            "mean_diff": float(mean_diff),
            "diff_ci_lo": float(ci_lo),
            "diff_ci_hi": float(ci_hi),
            "months_favoring": int(folds_fav),
            "total_months": int(n_eval_months),
            "months_favoring_frac": months_fav_frac,
            "per_month_gains": [float(g) for g in per_month_gains],
            "mean_candidate_rmsle": mean_a,
            "mean_rival_rmsle": mean_b,
        }
        print(
            f"  {pair_key:32s}: gain={rel_gain:+.4f}  CI=[{gain_ci_lo:+.4f}, {gain_ci_hi:+.4f}]  "
            f"months_fav={folds_fav}/{n_eval_months} ({months_fav_frac:.2f})"
        )

    # Baseline equivalence & claim strings
    max_abs_ridge_diff = float(np.max(np.abs(preds_df["final"] - preds_df["baseline_ridge"])))
    final_is_baseline_equivalent = bool(max_abs_ridge_diff <= float(thr["global"]["repro_rel_tol"]))

    p_final_ridge = pairwise_results["final vs baseline_ridge"]
    claim_final_vs_ridge = evaluate_claim(
        p_final_ridge["relative_gain"],
        p_final_ridge["ci_lo"],
        p_final_ridge["months_favoring_frac"],
        beat_margin=beat_margin,
        folds_fav_frac=folds_fav_frac,
    )

    p_chall_ridge = pairwise_results["challenger vs baseline_ridge"]
    claim_challenger_vs_ridge = evaluate_claim(
        p_chall_ridge["relative_gain"],
        p_chall_ridge["ci_lo"],
        p_chall_ridge["months_favoring_frac"],
        beat_margin=beat_margin,
        folds_fav_frac=folds_fav_frac,
    )

    print(f"\nFinal is baseline equivalent: {final_is_baseline_equivalent}")
    print(f"Claim final vs ridge: {claim_final_vs_ridge}")
    print(f"Claim challenger vs ridge: {claim_challenger_vs_ridge}")

    # Error breakdown by region
    print("\nComputing error breakdowns...")
    breakdown_by_region: dict[str, dict[str, dict[str, float]]] = {}
    for reg in sorted(preds_df["region"].dropna().unique()):
        sub_r = preds_df[preds_df["region"] == reg]
        y_r = sub_r["y"].to_numpy(dtype=float)
        r_dict: dict[str, dict[str, float]] = {}
        for cand in ["final", "challenger", "baseline_ridge", "baseline_mean", "baseline_persistence"]:
            p_r = sub_r[cand].to_numpy(dtype=float)
            r_dict[cand] = {
                "rmsle": gl.rmsle(y_r, p_r),
                "mae": gl.mae(y_r, p_r),
                "wape": gl.wape(y_r, p_r),
                "tail_mae": gl.tail_mae(y_r, p_r, tail_thr),
            }
        breakdown_by_region[str(reg)] = r_dict

    # Error breakdown by top-10 categories by train revenue
    top10_cats = train_df.groupby("category")["revenue"].sum().nlargest(10).index.tolist()
    breakdown_by_cat: dict[str, dict[str, dict[str, float]]] = {}
    for cat in top10_cats:
        sub_c = preds_df[preds_df["category"] == cat]
        if len(sub_c) == 0:
            continue
        y_c = sub_c["y"].to_numpy(dtype=float)
        c_dict: dict[str, dict[str, float]] = {}
        for cand in ["final", "challenger", "baseline_ridge", "baseline_mean", "baseline_persistence"]:
            p_c = sub_c[cand].to_numpy(dtype=float)
            c_dict[cand] = {
                "rmsle": gl.rmsle(y_c, p_c),
                "mae": gl.mae(y_c, p_c),
                "wape": gl.wape(y_c, p_c),
                "tail_mae": gl.tail_mae(y_c, p_c, tail_thr),
            }
        breakdown_by_cat[str(cat)] = c_dict

    # Write OOF parquet predictions
    print("\nSaving OOF parquet predictions...")
    oof_out_dir.mkdir(parents=True, exist_ok=True)
    preds_df.to_parquet(oof_out_dir / "holdout_predictions.parquet", index=False)

    for cand in ["final", "challenger", "baseline_ridge", "baseline_mean", "baseline_persistence"]:
        df_cand_oof = pd.DataFrame({
            "row_id": preds_df["row_id"].values,
            "series_id": preds_df["series_id"].values,
            "cluster": preds_df["cluster"].values,
            "month": preds_df["month"].values,
            "fold": preds_df["month"].values,
            "y": preds_df["y"].values,
            "tail_thr": preds_df["tail_thr"].values,
            "pred": preds_df[cand].values,
        })
        p_path = oof_out_dir / f"holdout_{cand}.parquet"
        df_cand_oof.to_parquet(p_path, index=False)
        print(f"  Saved {p_path}")

    # Save served models and manifest (challenger is NOT saved as served model)
    print("\nSaving model artifacts and manifest...")
    models_out_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = models_out_dir / "final_model.joblib"
    joblib.dump(fitted_models["final"], final_model_path)
    print(f"  Saved {final_model_path}")

    feature_pipeline_path = models_out_dir / "feature_pipeline.joblib"
    feature_pipeline_payload = {
        "features": list(final_cfg["features"]),
        "categorical_features": ["region", "category"],
        "categorical_levels": {
            "region": sorted(train_df["region"].dropna().unique().tolist()),
            "category": sorted(train_df["category"].dropna().unique().tolist()),
        },
        "tail_threshold": tail_thr,
        "history_window": {
            "min_train_months": int(cfg["cv"]["min_train_months"]),
            "lag_depth": int(cfg["lag_depth"]),
            "dev_months": sorted(train_df["month"].unique().tolist()),
            "holdout_months": unique_eval_months,
        },
    }
    joblib.dump(feature_pipeline_payload, feature_pipeline_path)
    print(f"  Saved {feature_pipeline_path}")

    manifest_path = models_out_dir / "MANIFEST.json"
    manifest_data = {
        "files": {
            "models/final_model.joblib": hashlib.sha256(final_model_path.read_bytes()).hexdigest(),
            "models/feature_pipeline.joblib": hashlib.sha256(feature_pipeline_path.read_bytes()).hexdigest(),
        },
        "versions": {
            "python": sys.version,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "shap": shap.__version__,
        },
        "config_sha256": final_cfg["sha256_config"],
        "sha256_src": final_cfg["sha256_src"],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"  Saved {manifest_path}")

    # Write holdout_results.json
    results_data = {
        "tail_threshold": tail_thr,
        "holdout_months": unique_eval_months,
        "holdout_rows": len(preds_df),
        "metrics": {
            "pooled": pooled_metrics,
            "per_month": per_month_metrics,
        },
        "pairwise": pairwise_results,
        "error_breakdown": {
            "by_region": breakdown_by_region,
            "by_category_top10": breakdown_by_cat,
        },
        "final_is_baseline_equivalent": final_is_baseline_equivalent,
        "claim_final_vs_ridge": claim_final_vs_ridge,
        "claim_challenger_vs_ridge": claim_challenger_vs_ridge,
    }

    results_out_path = reports_out_dir / "holdout_results.json"
    reports_out_dir.mkdir(parents=True, exist_ok=True)
    with open(results_out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved {results_out_path}")

    return results_data


def run_dry_run(temp_dir: Path | None = None) -> None:
    """Execute full evaluation pipeline on DEV stand-in rows with NO holdout read."""
    print("=== STARTING FULL DRY-RUN (DEV DATA STAND-IN) ===")
    final_cfg = verify_preflight_checks(is_dry_run=True)

    panel_dev = pd.read_parquet(DEV_PATH)
    unique_months = sorted(panel_dev["month"].unique())
    train_months = unique_months[:-3]
    eval_months = unique_months[-3:]

    print(f"Dev data: {len(unique_months)} total months.")
    print(f"Stand-in Train: {len(train_months)} months ({train_months[0]}..{train_months[-1]})")
    print(f"Stand-in Eval:  {len(eval_months)} months ({eval_months[0]}..{eval_months[-1]})")

    train_df = panel_dev[panel_dev["month"].isin(train_months)].copy().reset_index(drop=True)
    eval_df = panel_dev[panel_dev["month"].isin(eval_months)].copy().reset_index(drop=True)

    target_dir = temp_dir or Path(tempfile.mkdtemp(prefix="dry_run_eval_"))
    rep_dir = target_dir / "reports"
    oof_dir = rep_dir / "oof"
    models_dir = target_dir / "models"

    print(f"Target dry-run directory: {target_dir}")

    results_data = execute_pipeline(
        train_df=train_df,
        eval_df=eval_df,
        final_cfg=final_cfg,
        reports_out_dir=rep_dir,
        oof_out_dir=oof_dir,
        models_out_dir=models_dir,
        is_dry_run=True,
    )

    print("\nDry-run artifact verification:")
    expected_files = [
        rep_dir / "holdout_results.json",
        oof_dir / "holdout_predictions.parquet",
        oof_dir / "holdout_final.parquet",
        oof_dir / "holdout_challenger.parquet",
        oof_dir / "holdout_baseline_ridge.parquet",
        oof_dir / "holdout_baseline_mean.parquet",
        oof_dir / "holdout_baseline_persistence.parquet",
        models_dir / "final_model.joblib",
        models_dir / "feature_pipeline.joblib",
        models_dir / "MANIFEST.json",
    ]
    for ef in expected_files:
        assert ef.exists(), f"Missing expected dry-run file: {ef}"
        print(f"  [EXISTS] {ef.name}")

    print("\n=== DRY-RUN COMPLETED SUCCESSFULLY ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Stage B Holdout Evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Execute dry run on dev data without holdout access")
    parser.add_argument("--temp-dir", type=str, default=None, help="Custom output directory for dry run")
    args = parser.parse_args()

    if args.dry_run:
        custom_dir = Path(args.temp_dir) if args.temp_dir else None
        run_dry_run(custom_dir)
        return

    # Real run
    final_cfg = verify_preflight_checks(is_dry_run=False)

    print("\nLoading dev data from panel_dev.parquet...")
    panel_dev = pd.read_parquet(DEV_PATH)

    print("Executing single authorized recovery load: load_holdout('final_eval_recovery')...")
    try:
        panel_holdout = load_holdout("final_eval_recovery")
        print(f"Successfully loaded holdout data ({len(panel_holdout)} rows).")

        execute_pipeline(
            train_df=panel_dev,
            eval_df=panel_holdout,
            final_cfg=final_cfg,
            reports_out_dir=REPORTS_DIR,
            oof_out_dir=OOF_DIR,
            models_out_dir=MODELS_DIR,
            is_dry_run=False,
        )
        print("\nHoldout evaluation completed successfully.")

    except Exception:
        traceback.print_exc()
        print("\nSTATUS: STOP-HUMAN. Exception raised during holdout evaluation!")
        sys.exit(1)


if __name__ == "__main__":
    main()
