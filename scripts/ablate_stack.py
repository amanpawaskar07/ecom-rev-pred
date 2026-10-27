"""Run Phase 4 candidate ablations across expanding-window CV folds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gates import lib as gl
from revpred.config import get_project_config, get_thresholds
from revpred.cv import expanding_window_splits
from revpred.evaluate import evaluate_oof
from revpred.models import make_rf, make_ridge, make_xgb
from revpred.stack import TemporalStacker

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"


def run_stack_ablations(
    panel_dev: pd.DataFrame | None = None,
    reports_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute all 6 candidates in cost order across CV folds and compute statistical decisions."""
    cfg = get_project_config()
    thr = get_thresholds()
    primary_metric = str(thr["metrics"]["primary"])

    rep_path = Path(reports_dir) if reports_dir else REPORTS_DIR
    oof_path = rep_path / "oof"
    rep_path.mkdir(parents=True, exist_ok=True)
    oof_path.mkdir(parents=True, exist_ok=True)

    if panel_dev is None:
        panel_dev = pd.read_parquet(REPO_ROOT / "data" / "processed" / "panel_dev.parquet")

    # Inherit feature decision: base features only
    feat_dec_path = rep_path / "feature_decision.json"
    if feat_dec_path.exists():
        with open(feat_dec_path, encoding="utf-8") as f:
            feat_dec = json.load(f)
        kept_features = list(feat_dec.get("kept", []))
    else:
        kept_features = []

    base_features = list(cfg["features"]["base"])
    active_features = base_features + [f for f in kept_features if f in panel_dev.columns]

    min_train = int(cfg["cv"]["min_train_months"])
    val_months = int(cfg["cv"]["val_months"])
    splits = expanding_window_splits(
        panel_dev["month"],
        min_train_months=min_train,
        val_months=val_months,
    )

    candidates = [
        "ridge",
        "xgb",
        "rf",
        "mean_blend",
        "stack_ridge_meta",
        "stack_xgb_meta",
    ]

    candidate_oof_lists: dict[str, list[pd.DataFrame]] = {c: [] for c in candidates}
    inner_base_cache: dict[tuple[int, tuple[str, ...]], Any] = {}

    total_causality_violations = 0
    min_meta_train_rows = float("inf")

    print(f"Running Phase 4 ablations across {len(splits)} expanding-window folds...")

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        tr = panel_dev.iloc[tr_idx].copy().reset_index(drop=True)
        val = panel_dev.iloc[val_idx].copy().reset_index(drop=True)
        tail_thr = gl.tail_threshold(tr["revenue"].to_numpy())

        # Phase 3 decisions: log1p target, sample weights (3.0 for rare rows, else 1.0)
        y_tr_trans = np.log1p(tr["revenue"].to_numpy(dtype=float))
        w = np.where(tr["revenue"] >= tail_thr, 3.0, 1.0)

        X_tr = tr[active_features]
        X_val = val[active_features]

        # 1. Candidate 0: Ridge
        r = make_ridge()
        r.fit(X_tr, y_tr_trans, ridge__sample_weight=w)
        pred_ridge = np.clip(np.expm1(r.predict(X_val)), 0.0, None)

        # 2. Candidate 1: XGBoost
        m_xgb = make_xgb()
        m_xgb.fit(X_tr, y_tr_trans, sample_weight=w)
        pred_xgb = np.clip(np.expm1(m_xgb.predict(X_val)), 0.0, None)

        # 3. Candidate 1: Random Forest
        m_rf = make_rf()
        m_rf.fit(X_tr, y_tr_trans, sample_weight=w)
        pred_rf = np.clip(np.expm1(m_rf.predict(X_val)), 0.0, None)

        # 4. Candidate 2: Mean Blend of ridge + rf + xgb
        pred_mean_blend = (pred_ridge + pred_rf + pred_xgb) / 3.0

        # 5. Candidate 3: Temporal Stacker with Ridge meta
        stack_ridge = TemporalStacker(
            bases=["ridge", "rf"],
            meta=make_ridge(),
            min_inner_train_months=int(cfg["stack"]["min_inner_train_months"]),
            features=active_features,
            target_transform="log1p",
            resampler="weights",
            cache=inner_base_cache,
        )
        stack_ridge.fit(tr)
        pred_stack_ridge = stack_ridge.predict(val)

        # 6. Candidate 4: Temporal Stacker with XGB meta
        stack_xgb = TemporalStacker(
            bases=["ridge", "rf"],
            meta=make_xgb(params=cfg["stack"]["meta_params"]),
            min_inner_train_months=int(cfg["stack"]["min_inner_train_months"]),
            features=active_features,
            target_transform="log1p",
            resampler="weights",
            cache=inner_base_cache,
        )
        stack_xgb.fit(tr)
        pred_stack_xgb = stack_xgb.predict(val)

        # Causality verification on stacker OOF logs
        log_r = stack_ridge.oof_log
        log_x = stack_xgb.oof_log
        viol_r = int((log_r["fit_max_month"] >= log_r["month"]).sum())
        viol_x = int((log_x["fit_max_month"] >= log_x["month"]).sum())
        total_causality_violations += (viol_r + viol_x)

        n_meta = len(log_x)
        min_meta_train_rows = min(min_meta_train_rows, n_meta)

        preds_dict = {
            "ridge": pred_ridge,
            "xgb": pred_xgb,
            "rf": pred_rf,
            "mean_blend": pred_mean_blend,
            "stack_ridge_meta": pred_stack_ridge,
            "stack_xgb_meta": pred_stack_xgb,
        }

        common = {
            "row_id": val["row_id"].values,
            "series_id": val["series_id"].values,
            "cluster": val["series_id"].values,
            "month": val["month"].values,
            "fold": fold_idx,
            "y": val["revenue"].values,
            "tail_thr": tail_thr,
        }

        for c in candidates:
            df_fold = pd.DataFrame({**common, "pred": preds_dict[c]})
            candidate_oof_lists[c].append(df_fold)

        print(f"Fold {fold_idx + 1}/{len(splits)}: val={val['month'].iloc[0]}, n_val={len(val)}, n_meta_train={n_meta}")

    # Concatenate OOF dataframes and save parquets
    oof_dfs: dict[str, pd.DataFrame] = {}
    for c in candidates:
        df_oof = pd.concat(candidate_oof_lists[c], ignore_index=True)
        out_parquet = oof_path / (f"{c}.parquet" if c.startswith("stack_") else f"stack_{c}.parquet")
        df_oof.to_parquet(out_parquet, index=False)
        oof_dfs[c] = df_oof
        print(f"Saved {out_parquet} ({len(df_oof)} rows)")

    # Form comparison DataFrame for decisions
    df_eval = pd.DataFrame({
        "cluster": oof_dfs["ridge"]["cluster"],
        "fold": oof_dfs["ridge"]["fold"],
        "y": oof_dfs["ridge"]["y"],
        "tail_thr": oof_dfs["ridge"]["tail_thr"],
        "ridge": oof_dfs["ridge"]["pred"],
        "xgb": oof_dfs["xgb"]["pred"],
        "rf": oof_dfs["rf"]["pred"],
        "mean_blend": oof_dfs["mean_blend"]["pred"],
        "stack_ridge_meta": oof_dfs["stack_ridge_meta"]["pred"],
        "stack_xgb_meta": oof_dfs["stack_xgb_meta"]["pred"],
    })

    # Evaluate metrics for all candidates
    candidate_metrics: dict[str, Any] = {}
    for c in candidates:
        candidate_metrics[c] = evaluate_oof(oof_dfs[c])

    # 1. decide_among in cost order
    decision_result = gl.decide_among(
        df_eval,
        candidates_in_cost_order=candidates,
        metric=primary_metric,
    )
    default_model = decision_result.selected

    # 2. decide_pair: stack_xgb_meta vs each rival individually
    pairwise_vs_stack_xgb: dict[str, Any] = {}
    rivals = [c for c in candidates if c != "stack_xgb_meta"]
    for rival in rivals:
        pair_ev = gl.decide_pair(
            df_eval,
            advanced_col="stack_xgb_meta",
            rival_col=rival,
            metric=primary_metric,
        )
        pairwise_vs_stack_xgb[f"stack_xgb_meta vs {rival}"] = pair_ev

    ablation_report = {
        "metrics": candidate_metrics,
        "candidates_in_cost_order": candidates,
        "decide_among": decision_result.evidence,
        "pairwise_vs_stack_xgb_meta": pairwise_vs_stack_xgb,
        "default_model": default_model,
        "oof_causality_violations": total_causality_violations,
        "min_meta_train_rows": int(min_meta_train_rows),
    }

    ablation_out = rep_path / "ablation_stack.json"
    ablation_out.write_text(json.dumps(ablation_report, indent=2), encoding="utf-8")
    print(f"Saved {ablation_out}")

    decision_report = {
        "default_model": default_model,
        "candidates_in_cost_order": candidates,
        "decide_among_selected": default_model,
        "best_candidate": decision_result.evidence["best_candidate"],
        "mean_fold_metrics": decision_result.evidence["mean_fold_metrics"],
        "evidence": decision_result.evidence,
    }

    decision_out = rep_path / "stack_decision.json"
    decision_out.write_text(json.dumps(decision_report, indent=2), encoding="utf-8")
    print(f"Saved {decision_out} (default_model={default_model})")

    return ablation_report


def main() -> None:
    print("Running Phase 4 stacking ablations and decision rules...")
    run_stack_ablations()
    print("Stacking ablations complete.")


if __name__ == "__main__":
    main()
