"""Phase 5 Stage A: Random search hyperparameter tuning for Ridge and Stacked models.

Tuning budget: phase5.tuning_budget_configs (24) per candidate.
Candidates:
  1. tuned_ridge: FullRidgeRegressor (unweighted, log1p target), alpha log-uniform [0.01, 100.0].
  2. tuned_stack: stack_ridge_meta (Ridge on base+region+cat, RF on numeric features; Ridge meta on log1p, weighted),
                  RF params (n_estimators 100-400, max_depth 6-20, min_samples_leaf 1-10),
                  meta Ridge alpha log-uniform [0.01, 100.0].
Decision:
  Cost order [baseline_ridge, tuned_ridge, tuned_stack] via gates.lib.decide_among on Phase 1 CV folds.
Outputs:
  - reports/tuning_results.json
  - reports/final_config.json
  - reports/oof/tuned_ridge.parquet
  - reports/oof/tuned_stack.parquet
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gates import lib as gl
from revpred.config import get_project_config, get_thresholds
from revpred.cv import expanding_window_splits
from revpred.evaluate import evaluate_oof
from revpred.models import FullRidgeRegressor, make_rf, make_ridge
from revpred.stack import TemporalStacker

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"
DATA_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"


def compute_dir_hashes(dir_names: list[str]) -> dict[str, str]:
    """Compute sha256 for all relevant files in given directories."""
    file_hashes: dict[str, str] = {}
    for d in dir_names:
        dir_path = REPO_ROOT / d
        if not dir_path.exists():
            continue
        for p in sorted(dir_path.rglob("*")):
            if (
                p.is_file()
                and not p.name.endswith(".pyc")
                and "__pycache__" not in p.parts
                and not p.name.endswith(".gitkeep")
                and "revpred.egg-info" not in p.parts
            ):
                rel = str(p.relative_to(REPO_ROOT).as_posix())
                file_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return file_hashes


def compute_composite_sha(file_hashes: dict[str, str]) -> str:
    """Compute a single aggregate sha256 from sorted file hashes."""
    manifest = "".join(f"{k}:{v}\n" for k, v in sorted(file_hashes.items()))
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def run_tuning() -> None:
    print("=== Phase 5 Stage A: Tuning and Freeze ===")
    cfg = get_project_config()
    thr = get_thresholds()

    seed = int(thr["global"]["seed"])
    budget = int(thr["phase5"]["tuning_budget_configs"])
    primary_metric = thr["metrics"]["primary"]

    print(f"Loading data from {DATA_PATH}...")
    panel_dev = pd.read_parquet(DATA_PATH)

    # Feature definitions
    feat_dec_path = REPORTS_DIR / "feature_decision.json"
    if feat_dec_path.exists():
        with open(feat_dec_path, encoding="utf-8") as f:
            feat_dec = json.load(f)
        kept_features = list(feat_dec.get("kept", []))
    else:
        kept_features = []

    base_features = list(cfg["features"]["base"])
    active_features = base_features + [f for f in kept_features if f in panel_dev.columns]
    ridge_feats = base_features + ["region", "category"]
    rf_feats = active_features

    min_train = int(cfg["cv"]["min_train_months"])
    val_months = int(cfg["cv"]["val_months"])
    splits = expanding_window_splits(
        panel_dev["month"],
        min_train_months=min_train,
        val_months=val_months,
    )
    n_folds = len(splits)
    print(f"Configured {n_folds} expanding-window dev folds. Tuning budget: {budget} configs per candidate.")

    # =========================================================================
    # 1. Candidate A: tuned_ridge (FullRidgeRegressor, alpha only)
    # =========================================================================
    print("\n--- Tuning Candidate A: tuned_ridge ---")
    rng_ridge = np.random.default_rng(seed)
    # Log-uniform in [0.01, 100.0]
    log_alphas = rng_ridge.uniform(np.log10(0.01), np.log10(100.0), size=budget)
    ridge_configs = [
        {"config_id": i, "alpha": float(10 ** log_a)}
        for i, log_a in enumerate(log_alphas)
    ]

    ridge_results: list[dict[str, Any]] = []
    ridge_best_mean_fold_rmsle = float("inf")
    ridge_best_config: dict[str, Any] | None = None
    ridge_best_oof_df: pd.DataFrame | None = None

    for c_idx, r_cfg in enumerate(ridge_configs):
        alpha = r_cfg["alpha"]
        fold_records: list[dict[str, Any]] = []
        oof_dfs: list[pd.DataFrame] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            tr = panel_dev.iloc[tr_idx].copy().reset_index(drop=True)
            val = panel_dev.iloc[val_idx].copy().reset_index(drop=True)
            tail_thr = gl.tail_threshold(tr["revenue"].to_numpy())

            y_tr_trans = np.log1p(tr["revenue"].to_numpy(dtype=float))
            model = FullRidgeRegressor(alpha=alpha, random_state=seed)
            model.fit(tr, y_tr_trans)

            pred_log = model.predict(val)
            pred = np.clip(np.expm1(pred_log), 0.0, None)

            f_rmsle = gl.rmsle(val["revenue"], pred)
            f_mae = gl.mae(val["revenue"], pred)
            f_wape = gl.wape(val["revenue"], pred)
            f_tail_mae = gl.tail_mae(val["revenue"], pred, tail_thr)

            fold_records.append({
                "fold": fold_idx,
                "rmsle": f_rmsle,
                "mae": f_mae,
                "wape": f_wape,
                "tail_mae": f_tail_mae,
            })

            oof_dfs.append(pd.DataFrame({
                "row_id": val["row_id"].values,
                "series_id": val["series_id"].values,
                "cluster": val["series_id"].values,
                "month": val["month"].values,
                "fold": fold_idx,
                "y": val["revenue"].values,
                "tail_thr": tail_thr,
                "pred": pred,
            }))

        mean_fold_rmsle = float(np.mean([fr["rmsle"] for fr in fold_records]))
        df_oof_cand = pd.concat(oof_dfs, ignore_index=True)
        pooled_metrics = evaluate_oof(df_oof_cand)["pooled"]

        record = {
            "config_id": c_idx,
            "params": {"alpha": alpha},
            "mean_fold_rmsle": mean_fold_rmsle,
            "pooled": pooled_metrics,
            "fold_metrics": fold_records,
        }
        ridge_results.append(record)

        if mean_fold_rmsle < ridge_best_mean_fold_rmsle:
            ridge_best_mean_fold_rmsle = mean_fold_rmsle
            ridge_best_config = record
            ridge_best_oof_df = df_oof_cand

        print(f"[Ridge Config {c_idx+1:02d}/{budget}] alpha={alpha:8.4f} | mean_fold_rmsle={mean_fold_rmsle:.4f} | pooled_rmsle={pooled_metrics['rmsle']:.4f}")

    assert ridge_best_config is not None and ridge_best_oof_df is not None
    print(f"Best tuned_ridge: Config {ridge_best_config['config_id']} (alpha={ridge_best_config['params']['alpha']:.4f}) with mean_fold_rmsle={ridge_best_mean_fold_rmsle:.4f}")

    # Save tuned_ridge OOF parquet
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    ridge_oof_path = OOF_DIR / "tuned_ridge.parquet"
    ridge_best_oof_df.to_parquet(ridge_oof_path, index=False)
    print(f"Saved {ridge_oof_path} ({len(ridge_best_oof_df)} rows)")

    # =========================================================================
    # 2. Candidate B: tuned_stack (stack_ridge_meta, RF params + meta alpha)
    # =========================================================================
    print("\n--- Tuning Candidate B: tuned_stack ---")
    rng_stack = np.random.default_rng(seed)
    stack_configs = []
    for i in range(budget):
        n_est = int(rng_stack.integers(100, 401))
        max_d = int(rng_stack.integers(6, 21))
        min_leaf = int(rng_stack.integers(1, 11))
        meta_alpha = float(10 ** rng_stack.uniform(np.log10(0.01), np.log10(100.0)))
        stack_configs.append({
            "config_id": i,
            "rf_params": {
                "n_estimators": n_est,
                "max_depth": max_d,
                "min_samples_leaf": min_leaf,
            },
            "meta_alpha": meta_alpha,
        })

    stack_results: list[dict[str, Any]] = []
    stack_best_mean_fold_rmsle = float("inf")
    stack_best_config: dict[str, Any] | None = None
    stack_best_oof_df: pd.DataFrame | None = None

    # Shared cache across configs: Base 0 (FullRidgeRegressor) fits are reused across configs and folds
    inner_base_cache: dict[Any, Any] = {}

    for c_idx, s_cfg in enumerate(stack_configs):
        t0 = time.time()
        rf_params = s_cfg["rf_params"]
        meta_alpha = s_cfg["meta_alpha"]
        fold_records = []
        oof_dfs = []

        stack_bases = [
            (FullRidgeRegressor(), ridge_feats),
            (make_rf(rf_params), rf_feats),
        ]

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            tr = panel_dev.iloc[tr_idx].copy().reset_index(drop=True)
            val = panel_dev.iloc[val_idx].copy().reset_index(drop=True)
            tail_thr = gl.tail_threshold(tr["revenue"].to_numpy())

            stacker = TemporalStacker(
                bases=stack_bases,
                meta=make_ridge({"alpha": meta_alpha}),
                min_inner_train_months=int(cfg["stack"]["min_inner_train_months"]),
                features=active_features,
                target_transform="log1p",
                resampler="weights",
                cache=inner_base_cache,
            )
            stacker.fit(tr)
            pred = stacker.predict(val)

            f_rmsle = gl.rmsle(val["revenue"], pred)
            f_mae = gl.mae(val["revenue"], pred)
            f_wape = gl.wape(val["revenue"], pred)
            f_tail_mae = gl.tail_mae(val["revenue"], pred, tail_thr)

            fold_records.append({
                "fold": fold_idx,
                "rmsle": f_rmsle,
                "mae": f_mae,
                "wape": f_wape,
                "tail_mae": f_tail_mae,
            })

            oof_dfs.append(pd.DataFrame({
                "row_id": val["row_id"].values,
                "series_id": val["series_id"].values,
                "cluster": val["series_id"].values,
                "month": val["month"].values,
                "fold": fold_idx,
                "y": val["revenue"].values,
                "tail_thr": tail_thr,
                "pred": pred,
            }))

        mean_fold_rmsle = float(np.mean([fr["rmsle"] for fr in fold_records]))
        df_oof_cand = pd.concat(oof_dfs, ignore_index=True)
        pooled_metrics = evaluate_oof(df_oof_cand)["pooled"]
        elapsed = time.time() - t0

        record = {
            "config_id": c_idx,
            "params": {"rf_params": rf_params, "meta_alpha": meta_alpha},
            "mean_fold_rmsle": mean_fold_rmsle,
            "pooled": pooled_metrics,
            "fold_metrics": fold_records,
        }
        stack_results.append(record)

        if mean_fold_rmsle < stack_best_mean_fold_rmsle:
            stack_best_mean_fold_rmsle = mean_fold_rmsle
            stack_best_config = record
            stack_best_oof_df = df_oof_cand

        print(
            f"[Stack Config {c_idx+1:02d}/{budget}] n_est={rf_params['n_estimators']} "
            f"d={rf_params['max_depth']:02d} leaf={rf_params['min_samples_leaf']:02d} "
            f"alpha={meta_alpha:7.3f} | mean_fold_rmsle={mean_fold_rmsle:.4f} "
            f"pooled={pooled_metrics['rmsle']:.4f} ({elapsed:.1f}s)"
        )

    assert stack_best_config is not None and stack_best_oof_df is not None
    print(
        f"Best tuned_stack: Config {stack_best_config['config_id']} "
        f"(params={stack_best_config['params']}) with mean_fold_rmsle={stack_best_mean_fold_rmsle:.4f}"
    )

    # Save tuned_stack OOF parquet
    stack_oof_path = OOF_DIR / "tuned_stack.parquet"
    stack_best_oof_df.to_parquet(stack_oof_path, index=False)
    print(f"Saved {stack_oof_path} ({len(stack_best_oof_df)} rows)")

    # =========================================================================
    # 3. Decision Rule: decide_among [baseline_ridge, tuned_ridge, tuned_stack]
    # =========================================================================
    print("\n--- Running Decision Rule: decide_among ---")
    baseline_oof_path = OOF_DIR / "baseline_ridge.parquet"
    assert baseline_oof_path.exists(), f"Missing {baseline_oof_path}"

    df_base = pd.read_parquet(baseline_oof_path)
    df_eval = pd.DataFrame({
        "cluster": df_base["cluster"],
        "fold": df_base["fold"],
        "y": df_base["y"],
        "tail_thr": df_base["tail_thr"],
        "baseline_ridge": df_base["pred"],
        "tuned_ridge": ridge_best_oof_df["pred"],
        "tuned_stack": stack_best_oof_df["pred"],
    })

    candidates_order = ["baseline_ridge", "tuned_ridge", "tuned_stack"]
    decision_result = gl.decide_among(
        df_eval,
        candidates_in_cost_order=candidates_order,
        metric=primary_metric,
    )
    final_model = decision_result.selected
    challenger_model = "tuned_stack"  # Always frozen as challenger

    print(f"Decision: selected '{final_model}' among {candidates_order}")
    print(f"Best by metric: {decision_result.evidence['best_candidate']}")
    print(f"Mean fold metrics: {decision_result.evidence['mean_fold_metrics']}")
    for k, v in decision_result.evidence["pairwise_vs_best"].items():
        print(f"Pairwise vs best ({k}): {v}")

    # Determine final hyperparameters
    if final_model == "baseline_ridge":
        final_hyperparameters = {
            "model_type": "FullRidgeRegressor",
            "alphas": [0.1, 1.0, 10.0, 100.0],
            "cv_splits": 3,
            "description": "Baseline-equivalent FullRidgeRegressor with internal TimeSeriesSplit(3)",
        }
        resampler_choice = "none"
    elif final_model == "tuned_ridge":
        final_hyperparameters = {
            "model_type": "FullRidgeRegressor",
            "alpha": ridge_best_config["params"]["alpha"],
            "description": "Tuned FullRidgeRegressor with fixed alpha",
        }
        resampler_choice = "none"
    else:  # tuned_stack
        final_hyperparameters = {
            "model_type": "TemporalStacker",
            "rf_params": stack_best_config["params"]["rf_params"],
            "meta_alpha": stack_best_config["params"]["meta_alpha"],
            "description": "Tuned TemporalStacker with RF base and Ridge meta",
        }
        resampler_choice = "weights"

    challenger_hyperparameters = {
        "model_type": "TemporalStacker",
        "rf_params": stack_best_config["params"]["rf_params"],
        "meta_alpha": stack_best_config["params"]["meta_alpha"],
        "description": "Tuned TemporalStacker pre-registered challenger",
    }

    # =========================================================================
    # 4. Stability Check: rerun final-config CV with phase5.n_seeds seeds
    # =========================================================================
    print("\n--- Running Stability Check ---")
    n_seeds = int(thr["phase5"]["n_seeds"])
    seeds = [seed + i for i in range(n_seeds)]
    pooled_rmsles: list[float] = []

    print(f"Evaluating final model '{final_model}' across {n_seeds} seeds: {seeds}")

    for s in seeds:
        seed_oof_preds: list[np.ndarray] = []
        seed_oof_y: list[np.ndarray] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            tr = panel_dev.iloc[tr_idx].copy().reset_index(drop=True)
            val = panel_dev.iloc[val_idx].copy().reset_index(drop=True)

            if final_model == "baseline_ridge":
                y_tr_trans = np.log1p(tr["revenue"].to_numpy(dtype=float))
                model = FullRidgeRegressor(random_state=s)
                model.fit(tr, y_tr_trans)
                p = np.clip(np.expm1(model.predict(val)), 0.0, None)
            elif final_model == "tuned_ridge":
                y_tr_trans = np.log1p(tr["revenue"].to_numpy(dtype=float))
                model = FullRidgeRegressor(alpha=final_hyperparameters["alpha"], random_state=s)
                model.fit(tr, y_tr_trans)
                p = np.clip(np.expm1(model.predict(val)), 0.0, None)
            else:  # tuned_stack
                st_bases = [
                    (FullRidgeRegressor(random_state=s), ridge_feats),
                    (make_rf(dict(final_hyperparameters["rf_params"], random_state=s)), rf_feats),
                ]
                st = TemporalStacker(
                    bases=st_bases,
                    meta=make_ridge({"alpha": final_hyperparameters["meta_alpha"], "random_state": s}),
                    min_inner_train_months=int(cfg["stack"]["min_inner_train_months"]),
                    features=active_features,
                    target_transform="log1p",
                    resampler="weights",
                )
                st.fit(tr)
                p = st.predict(val)

            seed_oof_preds.append(p)
            seed_oof_y.append(val["revenue"].values)

        all_y = np.concatenate(seed_oof_y)
        all_p = np.concatenate(seed_oof_preds)
        s_rmsle = gl.rmsle(all_y, all_p)
        pooled_rmsles.append(float(s_rmsle))
        print(f"Seed {s}: pooled_rmsle = {s_rmsle:.6f}")

    mean_rmsle = float(np.mean(pooled_rmsles))
    seed_spread_rel = float((max(pooled_rmsles) - min(pooled_rmsles)) / mean_rmsle)
    max_allowed_spread = float(thr["phase5"]["seed_spread_max_rel"])
    print(f"Seed spread rel: {seed_spread_rel:.6f} (max allowed: {max_allowed_spread:.6f})")

    # =========================================================================
    # 5. Cryptographic Checksums & Report Generation
    # =========================================================================
    print("\n--- Computing Hashes and Writing Reports ---")
    file_hashes = compute_dir_hashes(["config", "src"])
    tree_sha = compute_composite_sha(file_hashes)
    config_sha = compute_composite_sha({k: v for k, v in file_hashes.items() if k.startswith("config/")})
    src_sha = compute_composite_sha({k: v for k, v in file_hashes.items() if k.startswith("src/")})

    tuning_results_report = {
        "budget_configs": budget,
        "seed": seed,
        "ridge": {
            "candidate": "tuned_ridge",
            "best_config_id": ridge_best_config["config_id"],
            "best_params": ridge_best_config["params"],
            "best_mean_fold_rmsle": ridge_best_mean_fold_rmsle,
            "best_pooled": ridge_best_config["pooled"],
            "configs": ridge_results,
        },
        "stack": {
            "candidate": "tuned_stack",
            "best_config_id": stack_best_config["config_id"],
            "best_params": stack_best_config["params"],
            "best_mean_fold_rmsle": stack_best_mean_fold_rmsle,
            "best_pooled": stack_best_config["pooled"],
            "configs": stack_results,
        },
    }

    tuning_results_path = REPORTS_DIR / "tuning_results.json"
    with open(tuning_results_path, "w", encoding="utf-8") as f:
        json.dump(tuning_results_report, f, indent=2)
    print(f"Saved {tuning_results_path}")

    final_config_report = {
        "final_model": final_model,
        "final_hyperparameters": final_hyperparameters,
        "challenger_model": challenger_model,
        "challenger_hyperparameters": challenger_hyperparameters,
        "challenger_frozen": 1,
        "features": active_features,
        "target_transform": "log1p",
        "resampler": resampler_choice,
        "seed": seed,
        "decision_evidence": decision_result.evidence,
        "stability": {
            "n_seeds": n_seeds,
            "seeds": seeds,
            "pooled_rmsles": pooled_rmsles,
            "seed_spread_rel": seed_spread_rel,
        },
        "sha256": tree_sha,
        "sha256_config": config_sha,
        "sha256_src": src_sha,
        "file_hashes": file_hashes,
    }

    final_config_path = REPORTS_DIR / "final_config.json"
    with open(final_config_path, "w", encoding="utf-8") as f:
        json.dump(final_config_report, f, indent=2)
    print(f"Saved {final_config_path}")

    print("\nPhase 5 Stage A tuning complete.")


if __name__ == "__main__":
    run_tuning()