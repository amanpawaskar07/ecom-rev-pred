"""Phase 5 Stage B: Core evaluation engine for Final, Challenger, and Baseline models.

Strictly causal, one-step-ahead evaluation over unseen evaluation rows.
Rebuilds challenger and final model from final_config.json specifications.
Fits each model exactly once on train_df; never refits during evaluation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gates import lib as gl
from revpred.models import FullRidgeRegressor, make_rf, make_ridge
from revpred.stack import TemporalStacker


def build_challenger_from_config(
    challenger_hyp: dict[str, Any],
    seed: int = 42,
) -> TemporalStacker:
    """Rebuild TemporalStacker challenger strictly from JSON hyperparameters."""
    bases: list[tuple[Any, list[str]]] = []
    for b in challenger_hyp["bases"]:
        m_type = b["model_type"]
        feats = list(b["features"])
        if m_type == "FullRidgeRegressor":
            base_model = FullRidgeRegressor(random_state=seed)
        elif m_type in ("RandomForest", "RandomForestRegressor"):
            rf_params = dict(b.get("params", challenger_hyp.get("rf_params", {})))
            rf_params.setdefault("random_state", seed)
            rf_params.setdefault("n_jobs", 1)
            base_model = make_rf(rf_params)
        else:
            raise ValueError(f"Unknown base model type: {m_type}")
        bases.append((base_model, feats))

    meta_spec = challenger_hyp.get("meta", {})
    if isinstance(meta_spec, dict):
        meta_alpha = float(meta_spec.get("alpha", challenger_hyp.get("meta_alpha", 1.0)))
    else:
        meta_alpha = float(challenger_hyp.get("meta_alpha", 1.0))

    meta_model = make_ridge({"alpha": meta_alpha, "random_state": seed})
    min_inner = int(challenger_hyp.get("min_inner_train_months", 3))
    active_feats = list(challenger_hyp.get("features", []))
    target_transform = challenger_hyp.get("target_transform", "log1p")
    resampler = challenger_hyp.get("resampler", "weights")

    return TemporalStacker(
        bases=bases,
        meta=meta_model,
        min_inner_train_months=min_inner,
        features=active_feats,
        target_transform=target_transform,
        resampler=resampler,
    )


def run_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    final_config: dict[str, Any],
) -> dict[str, Any]:
    """Train Final, Challenger, and Baselines on train_df; predict eval_df one-step-ahead.

    Models evaluated:
      - final: FullRidgeRegressor from final_config.json
      - challenger: TemporalStacker rebuilt from final_config.json
      - baseline_mean: mean of log1p train revenue
      - baseline_persistence: lag_rev_1
      - baseline_ridge: baseline-equivalent FullRidgeRegressor

    Evaluation:
      - ONE tail threshold computed from positive train_df revenue.
      - Predicts month by month, one-step-ahead; no refitting.
    """
    seed = int(final_config.get("seed", 42))
    fit_counts: dict[str, int] = {
        "final": 0,
        "challenger": 0,
        "baseline_ridge": 0,
        "baseline_mean": 0,
        "baseline_persistence": 0,
    }

    # 1. Single tail threshold from train_df
    tail_thr = float(gl.tail_threshold(train_df["revenue"].to_numpy(dtype=float)))

    # 2. Train baseline_mean
    y_tr_raw = train_df["revenue"].to_numpy(dtype=float)
    y_tr_log = np.log1p(y_tr_raw)
    mean_val = float(np.expm1(np.mean(y_tr_log)))
    fit_counts["baseline_mean"] += 1
    fit_counts["baseline_persistence"] += 1  # Persistence has no parameters to fit

    # 3. Train baseline_ridge
    baseline_ridge = FullRidgeRegressor(
        alphas=(0.1, 1.0, 10.0, 100.0), cv_splits=3, random_state=seed
    )
    baseline_ridge.fit(train_df, y_tr_log)
    fit_counts["baseline_ridge"] += 1

    # 4. Train final model
    final_hyp = final_config.get("final_hyperparameters", {})
    if final_hyp.get("model_type") == "FullRidgeRegressor":
        final_alphas = final_hyp.get("alphas")
        final_alpha = final_hyp.get("alpha")
        final_splits = int(final_hyp.get("cv_splits", 3))
        final_model = FullRidgeRegressor(
            alphas=final_alphas,
            alpha=final_alpha,
            cv_splits=final_splits,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported final model type: {final_hyp.get('model_type')}")
    final_model.fit(train_df, y_tr_log)
    fit_counts["final"] += 1

    # 5. Train challenger model
    challenger_hyp = final_config["challenger_hyperparameters"]
    challenger_model = build_challenger_from_config(challenger_hyp, seed=seed)
    challenger_model.fit(train_df)
    fit_counts["challenger"] += 1

    # 6. Predict eval_df month by month (strictly one-step-ahead; never refit)
    unique_months = sorted(eval_df["month"].unique())
    pred_blocks: list[pd.DataFrame] = []
    per_month_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for m in unique_months:
        eval_m = eval_df[eval_df["month"] == m].copy().reset_index(drop=True)
        y_m = eval_m["revenue"].to_numpy(dtype=float)

        p_final = np.clip(np.expm1(final_model.predict(eval_m)), 0.0, None)
        p_ridge = np.clip(np.expm1(baseline_ridge.predict(eval_m)), 0.0, None)
        p_mean = np.full(len(eval_m), mean_val, dtype=float)
        p_pers = np.clip(eval_m["lag_rev_1"].to_numpy(dtype=float), 0.0, None)
        p_challenger = np.clip(challenger_model.predict(eval_m), 0.0, None)

        m_preds_df = pd.DataFrame({
            "row_id": eval_m["row_id"].values,
            "series_id": eval_m["series_id"].values,
            "cluster": eval_m["series_id"].values,
            "category": eval_m["category"].values if "category" in eval_m.columns else "",
            "region": eval_m["region"].values if "region" in eval_m.columns else "",
            "month": eval_m["month"].values,
            "fold": eval_m["month"].values,
            "y": y_m,
            "tail_thr": tail_thr,
            "final": p_final,
            "challenger": p_challenger,
            "baseline_ridge": p_ridge,
            "baseline_mean": p_mean,
            "baseline_persistence": p_pers,
        })
        pred_blocks.append(m_preds_df)

        # Per-month metrics
        m_results: dict[str, dict[str, float]] = {}
        for cand, p in [
            ("final", p_final),
            ("challenger", p_challenger),
            ("baseline_ridge", p_ridge),
            ("baseline_mean", p_mean),
            ("baseline_persistence", p_pers),
        ]:
            m_results[cand] = {
                "rmsle": gl.rmsle(y_m, p),
                "mae": gl.mae(y_m, p),
                "wape": gl.wape(y_m, p),
                "tail_mae": gl.tail_mae(y_m, p, tail_thr),
            }
        per_month_metrics[m] = m_results

    df_preds = pd.concat(pred_blocks, ignore_index=True)
    y_all = df_preds["y"].to_numpy(dtype=float)

    # 7. Pooled metrics over all eval rows
    pooled_metrics: dict[str, dict[str, float]] = {}
    for cand in [
        "final",
        "challenger",
        "baseline_ridge",
        "baseline_mean",
        "baseline_persistence",
    ]:
        p_all = df_preds[cand].to_numpy(dtype=float)
        pooled_metrics[cand] = {
            "rmsle": gl.rmsle(y_all, p_all),
            "mae": gl.mae(y_all, p_all),
            "wape": gl.wape(y_all, p_all),
            "tail_mae": gl.tail_mae(y_all, p_all, tail_thr),
        }

    return {
        "predictions": df_preds,
        "pooled": pooled_metrics,
        "per_month": per_month_metrics,
        "tail_threshold": tail_thr,
        "fit_counts": fit_counts,
        "models": {
            "final": final_model,
            "challenger": challenger_model,
            "baseline_ridge": baseline_ridge,
        },
    }

