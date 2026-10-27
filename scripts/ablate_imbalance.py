"""Phase 3: Target transform and SMOTER resampling ablations on real dev data.

Ablations:
  1. Target: T0 log1p vs T1 quantile (resampling none).
  2. Resampling: with the winning target:
       R0 none, R1 weights, R2 smoter.
Decisions via gates.lib:
  - Target: decide_pair(T1 adv, T0 rival) on primary metric (rmsle).
  - Resampling: decide_among([none, weights, smoter]) on tail_mae, subject to bulk non-inferiority:
    chosen option's primary metric must be within metrics.bulk_noninferiority_rel of the best
    primary among the three, else take the next cheaper option.

Saves:
  - reports/oof/imb_t0_log1p.parquet
  - reports/oof/imb_t1_quantile.parquet
  - reports/oof/imb_r0_none.parquet
  - reports/oof/imb_r1_weights.parquet
  - reports/oof/imb_r2_smoter.parquet
  - reports/ablation_target.json
  - reports/ablation_resample.json
  - reports/imbalance_decision.json
"""
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
from revpred.models import make_xgb
from revpred.resample import SmoterResampler, TrainGuard
from revpred.targets import Log1pTargetTransform, QuantileTargetTransform

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"


def run_target_ablation(
    panel_dev: pd.DataFrame,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    """Run Target Ablation: T0 log1p vs T1 quantile (resampling none)."""
    oof_t0: list[pd.DataFrame] = []
    oof_t1: list[pd.DataFrame] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        tr = panel_dev.iloc[tr_idx].copy().reset_index(drop=True)
        val = panel_dev.iloc[val_idx].copy().reset_index(drop=True)
        tail_thr = gl.tail_threshold(tr["revenue"].to_numpy())

        # T0: log1p
        t0 = Log1pTargetTransform()
        t0.fit(tr["revenue"])
        y_tr_t0 = t0.transform(tr["revenue"])
        m0 = make_xgb().fit(tr[features], y_tr_t0)
        p0 = t0.inverse(m0.predict(val[features]))

        # T1: quantile
        t1 = QuantileTargetTransform(random_state=seed)
        t1.fit(tr["revenue"])
        y_tr_t1 = t1.transform(tr["revenue"])
        m1 = make_xgb().fit(tr[features], y_tr_t1)
        p1 = t1.inverse(m1.predict(val[features]))

        common = {
            "row_id": val["row_id"].values,
            "series_id": val["series_id"].values,
            "cluster": val["series_id"].values,
            "month": val["month"].values,
            "fold": fold_idx,
            "y": val["revenue"].values,
            "tail_thr": tail_thr,
        }
        oof_t0.append(pd.DataFrame({**common, "pred": p0}))
        oof_t1.append(pd.DataFrame({**common, "pred": p1}))

    df_t0 = pd.concat(oof_t0, ignore_index=True)
    df_t1 = pd.concat(oof_t1, ignore_index=True)

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df_t0.to_parquet(OOF_DIR / "imb_t0_log1p.parquet", index=False)
    df_t1.to_parquet(OOF_DIR / "imb_t1_quantile.parquet", index=False)

    pair_df = pd.DataFrame({
        "cluster": df_t0["cluster"],
        "fold": df_t0["fold"],
        "y": df_t0["y"],
        "tail_thr": df_t0["tail_thr"],
        "pred_adv": df_t1["pred"],
        "pred_rival": df_t0["pred"],
    })

    thr = get_thresholds()
    primary_metric = str(thr["metrics"]["primary"])
    target_decision = gl.decide_pair(
        pair_df,
        advanced_col="pred_adv",
        rival_col="pred_rival",
        metric=primary_metric,
    )

    winning_target = "quantile" if target_decision["advanced_wins"] else "log1p"

    ablation_target_report = {
        "metrics": {
            "t0_log1p": evaluate_oof(df_t0),
            "t1_quantile": evaluate_oof(df_t1),
        },
        "decision": target_decision,
        "winning_target": winning_target,
    }
    with open(REPORTS_DIR / "ablation_target.json", "w", encoding="utf-8") as f:
        json.dump(ablation_target_report, f, indent=2)

    return df_t0, df_t1, ablation_target_report, winning_target


def run_resample_ablation(
    panel_dev: pd.DataFrame,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    winning_target: str,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], str]:
    """Run Resampling Ablation: R0 none, R1 weights, R2 smoter with winning target."""
    oof_r0: list[pd.DataFrame] = []
    oof_r1: list[pd.DataFrame] = []
    oof_r2: list[pd.DataFrame] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        tr = panel_dev.iloc[tr_idx].copy().reset_index(drop=True)
        val = panel_dev.iloc[val_idx].copy().reset_index(drop=True)
        tail_thr = gl.tail_threshold(tr["revenue"].to_numpy())

        if winning_target == "quantile":
            t = QuantileTargetTransform(random_state=seed)
        else:
            t = Log1pTargetTransform()
        t.fit(tr["revenue"])
        y_tr_trans = t.transform(tr["revenue"])

        # R0: none
        m0 = make_xgb().fit(tr[features], y_tr_trans)
        p0 = t.inverse(m0.predict(val[features]))

        # R1: weights (3.0 for rare rows, else 1.0)
        w = np.where(tr["revenue"] >= tail_thr, 3.0, 1.0)
        m1 = make_xgb().fit(tr[features], y_tr_trans, sample_weight=w)
        p1 = t.inverse(m1.predict(val[features]))

        # R2: smoter
        guard = TrainGuard(tr.index)
        res = SmoterResampler(guard=guard).fit_resample(
            X=tr[features],
            y=y_tr_trans,
            train_idx=tr.index,
            seed=seed,
            tail_thr=tail_thr,
            y_raw=tr["revenue"].values,
        )
        m2 = make_xgb().fit(res.X, res.y)
        p2 = t.inverse(m2.predict(val[features]))

        common = {
            "row_id": val["row_id"].values,
            "series_id": val["series_id"].values,
            "cluster": val["series_id"].values,
            "month": val["month"].values,
            "fold": fold_idx,
            "y": val["revenue"].values,
            "tail_thr": tail_thr,
        }
        oof_r0.append(pd.DataFrame({**common, "pred": p0}))
        oof_r1.append(pd.DataFrame({**common, "pred": p1}))
        oof_r2.append(pd.DataFrame({**common, "pred": p2}))

    df_r0 = pd.concat(oof_r0, ignore_index=True)
    df_r1 = pd.concat(oof_r1, ignore_index=True)
    df_r2 = pd.concat(oof_r2, ignore_index=True)

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df_r0.to_parquet(OOF_DIR / "imb_r0_none.parquet", index=False)
    df_r1.to_parquet(OOF_DIR / "imb_r1_weights.parquet", index=False)
    df_r2.to_parquet(OOF_DIR / "imb_r2_smoter.parquet", index=False)

    df_resample = pd.DataFrame({
        "cluster": df_r0["cluster"],
        "fold": df_r0["fold"],
        "y": df_r0["y"],
        "tail_thr": df_r0["tail_thr"],
        "none": df_r0["pred"],
        "weights": df_r1["pred"],
        "smoter": df_r2["pred"],
    })

    candidates_in_cost_order = ["none", "weights", "smoter"]
    res_among = gl.decide_among(
        df_resample,
        candidates_in_cost_order=candidates_in_cost_order,
        metric="tail_mae",
    )

    # Bulk non-inferiority check
    thr = get_thresholds()
    bulk_noninferiority_rel = float(thr["metrics"]["bulk_noninferiority_rel"])

    folds = np.sort(df_resample["fold"].unique())
    rmsle_scores: dict[str, float] = {}
    for c in candidates_in_cost_order:
        scores = [gl.rmsle(df_resample.loc[df_resample["fold"] == f, "y"], df_resample.loc[df_resample["fold"] == f, c]) for f in folds]
        rmsle_scores[c] = float(np.mean(scores))

    best_primary = min(rmsle_scores.values())

    initial_selected = res_among.selected
    final_selected = initial_selected
    current_idx = candidates_in_cost_order.index(initial_selected)

    while current_idx >= 0:
        cand = candidates_in_cost_order[current_idx]
        rel_worsening = (rmsle_scores[cand] - best_primary) / best_primary
        if rel_worsening <= bulk_noninferiority_rel:
            final_selected = cand
            break
        current_idx -= 1

    if current_idx < 0:
        final_selected = "none"

    ablation_resample_report = {
        "metrics": {
            "r0_none": evaluate_oof(df_r0),
            "r1_weights": evaluate_oof(df_r1),
            "r2_smoter": evaluate_oof(df_r2),
        },
        "rmsle_by_candidate": rmsle_scores,
        "best_primary_candidate": min(candidates_in_cost_order, key=lambda c: rmsle_scores[c]),
        "decide_among_tail_mae": res_among.evidence,
        "initial_selected": initial_selected,
        "final_selected": final_selected,
        "bulk_noninferiority_rel_threshold": bulk_noninferiority_rel,
    }
    with open(REPORTS_DIR / "ablation_resample.json", "w", encoding="utf-8") as f:
        json.dump(ablation_resample_report, f, indent=2)

    return {"r0_none": df_r0, "r1_weights": df_r1, "r2_smoter": df_r2}, ablation_resample_report, final_selected


def main() -> None:
    panel_dev = pd.read_parquet(REPO_ROOT / "data" / "processed" / "panel_dev.parquet")
    cfg = get_project_config()
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])

    feat_dec_path = REPORTS_DIR / "feature_decision.json"
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

    print("Running target ablation (T0 log1p vs T1 quantile)...")
    _, _, target_rep, winning_target = run_target_ablation(
        panel_dev, active_features, splits, seed
    )
    print(f"Target ablation complete. Winner: {winning_target}")

    print(f"Running resampling ablation (R0 none, R1 weights, R2 smoter) with winning target '{winning_target}'...")
    _, resample_rep, chosen_resampler = run_resample_ablation(
        panel_dev, active_features, splits, winning_target, seed
    )
    print(f"Resampling ablation complete. Chosen: {chosen_resampler}")

    imbalance_decision = {
        "winning_target": winning_target,
        "target_decision": target_rep["decision"],
        "chosen_resampler": chosen_resampler,
        "resample_decision": resample_rep,
    }
    with open(REPORTS_DIR / "imbalance_decision.json", "w", encoding="utf-8") as f:
        json.dump(imbalance_decision, f, indent=2)

    print("Imbalance decision saved to reports/imbalance_decision.json")


if __name__ == "__main__":
    main()
