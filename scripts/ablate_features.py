"""Run Phase 2 feature ablations, negative controls, and decision rules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from gates.lib import decide_pair, tail_threshold
from revpred.config import get_project_config, get_thresholds
from revpred.cv import expanding_window_splits
from revpred.evaluate import evaluate_oof
from revpred.features import (
    FeatureBuilder,
    cat_region_level_table,
    shuffle_feature_within_month,
)
from revpred.models import ModelSpec, make_xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"
CONFIG_DIR = REPO_ROOT / "config"


def run_feature_ablations(
    panel_dev: pd.DataFrame | None = None,
    order_lines: pd.DataFrame | None = None,
    reports_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute feature ablations and shuffled controls across CV folds, producing OOFs and reports."""
    cfg = get_project_config()
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])

    rep_path = Path(reports_dir) if reports_dir else REPORTS_DIR
    oof_path = rep_path / "oof"
    rep_path.mkdir(parents=True, exist_ok=True)
    oof_path.mkdir(parents=True, exist_ok=True)

    if panel_dev is None:
        panel_dev = pd.read_parquet(REPO_ROOT / "data" / "processed" / "panel_dev.parquet")
    if order_lines is None:
        order_lines = pd.read_parquet(REPO_ROOT / "data" / "processed" / "order_lines.parquet")

    base_features = list(cfg["features"]["base"])
    eng_features = list(cfg["features"]["engineered"])

    crl_table = cat_region_level_table(order_lines)

    min_train = int(cfg["cv"]["min_train_months"])
    val_months = int(cfg["cv"]["val_months"])
    splits = expanding_window_splits(
        panel_dev["month"],
        min_train_months=min_train,
        val_months=val_months,
    )

    variants = {
        "feat_f0": base_features,
        "feat_f1_price_bucket": base_features + ["price_bucket"],
        "feat_f2_high_value_flag": base_features + ["high_value_flag"],
        "feat_f3_cat_region_level": base_features + ["cat_region_level"],
        "feat_f4_all": base_features + eng_features,
    }

    controls = {
        "feat_c_price_bucket": "price_bucket",
        "feat_c_high_value_flag": "high_value_flag",
        "feat_c_cat_region_level": "cat_region_level",
    }

    # Pre-transform fold dataframes with FeatureBuilder (fit on train only)
    fold_data = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        tr = panel_dev.iloc[train_idx].copy().reset_index(drop=True)
        val = panel_dev.iloc[val_idx].copy().reset_index(drop=True)
        val_m = str(val["month"].min())

        fb = FeatureBuilder(cat_region_df=crl_table, val_month=val_m)
        fb.fit(tr)
        tr_trans = fb.transform(tr)
        val_trans = fb.transform(val)
        tail_thr_val = tail_threshold(tr["revenue"].to_numpy())

        fold_data.append({
            "fold": fold_idx,
            "tr": tr_trans,
            "val": val_trans,
            "tail_thr": tail_thr_val,
        })

    oof_dfs: dict[str, pd.DataFrame] = {}

    # Run variants
    for var_name, feats in variants.items():
        dfs = []
        for fd in fold_data:
            tr = fd["tr"]
            val = fd["val"]
            spec = ModelSpec(name=var_name, features=feats, estimator=make_xgb())
            preds = spec(tr, val)

            dfs.append(pd.DataFrame({
                "row_id": val["row_id"].values,
                "series_id": val["series_id"].values,
                "cluster": val["series_id"].values,
                "month": val["month"].values,
                "fold": fd["fold"],
                "y": val["revenue"].values,
                "pred": preds,
                "tail_thr": fd["tail_thr"],
            }))
        oof_df = pd.concat(dfs, ignore_index=True)
        oof_df.to_parquet(oof_path / f"{var_name}.parquet", index=False)
        oof_dfs[var_name] = oof_df

    # Run controls
    for ctrl_name, col in controls.items():
        dfs = []
        for fd in fold_data:
            tr = fd["tr"].copy()
            val = fd["val"].copy()

            tr_shuf = shuffle_feature_within_month(tr, col, seed=seed)
            val_shuf = shuffle_feature_within_month(val, col, seed=seed)

            spec = ModelSpec(name=ctrl_name, features=base_features + [col], estimator=make_xgb())
            preds = spec(tr_shuf, val_shuf)

            dfs.append(pd.DataFrame({
                "row_id": val["row_id"].values,
                "series_id": val["series_id"].values,
                "cluster": val["series_id"].values,
                "month": val["month"].values,
                "fold": fd["fold"],
                "y": val["revenue"].values,
                "pred": preds,
                "tail_thr": fd["tail_thr"],
            }))
        oof_df = pd.concat(dfs, ignore_index=True)
        oof_df.to_parquet(oof_path / f"{ctrl_name}.parquet", index=False)
        oof_dfs[ctrl_name] = oof_df

    # Evaluate metrics and decisions
    f0_oof = oof_dfs["feat_f0"]
    ablation_results: dict[str, Any] = {
        "metrics": {},
        "decisions": {},
        "controls": {},
    }

    for name, oof in oof_dfs.items():
        ablation_results["metrics"][name] = evaluate_oof(oof)

    decisions: dict[str, Any] = {}
    kept_list: list[str] = []
    dropped_list: list[str] = []

    feature_variant_map = {
        "price_bucket": "feat_f1_price_bucket",
        "high_value_flag": "feat_f2_high_value_flag",
        "cat_region_level": "feat_f3_cat_region_level",
    }
    feature_ctrl_map = {
        "price_bucket": "feat_c_price_bucket",
        "high_value_flag": "feat_c_high_value_flag",
        "cat_region_level": "feat_c_cat_region_level",
    }

    min_effect = float(thr["global"]["min_effect_rel"])
    min_folds_frac = float(thr["global"]["folds_favoring_frac"])

    for feat, var_name in feature_variant_map.items():
        oof_feat = oof_dfs[var_name]
        pair_df = pd.DataFrame({
            "cluster": oof_feat["cluster"],
            "fold": oof_feat["fold"],
            "y": oof_feat["y"],
            "tail_thr": oof_feat["tail_thr"],
            "pred_adv": oof_feat["pred"],
            "pred_f0": f0_oof["pred"],
        })
        pair_res = decide_pair(pair_df, advanced_col="pred_adv", rival_col="pred_f0", metric="rmsle")
        ablation_results["decisions"][feat] = pair_res

        adv_wins = bool(pair_res["advanced_wins"])
        rel_gain = float(pair_res["relative_gain"])
        ci_lo = float(pair_res["ci_lo"])
        ci_hi = float(pair_res["ci_hi"])
        fav_frac = float(pair_res["folds_favoring_frac"])

        if adv_wins:
            kept_list.append(feat)
            reason = f"relative_gain ({rel_gain:.4f}) >= {min_effect} and ci_lo ({ci_lo:.4f}) > 0 and folds_favoring_frac ({fav_frac:.2f}) >= {min_folds_frac}"
        else:
            dropped_list.append(feat)
            reasons = []
            if rel_gain < min_effect:
                reasons.append(f"relative_gain ({rel_gain:.4f}) < min_effect_rel ({min_effect})")
            if ci_lo <= 0:
                reasons.append(f"ci_lo ({ci_lo:.4f}) <= 0")
            if fav_frac < min_folds_frac:
                reasons.append(f"folds_favoring_frac ({fav_frac:.2f}) < {min_folds_frac}")
            reason = "; ".join(reasons)

        decisions[feat] = {
            "kept": adv_wins,
            "reason": reason,
            "relative_gain": rel_gain,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "folds_favoring_adv": pair_res["folds_favoring_adv"],
            "n_folds": pair_res["n_folds"],
            "folds_favoring_frac": fav_frac,
            "advanced_wins": adv_wins,
        }

    # Controls decision analysis
    for feat, ctrl_name in feature_ctrl_map.items():
        oof_ctrl = oof_dfs[ctrl_name]
        ctrl_pair_df = pd.DataFrame({
            "cluster": oof_ctrl["cluster"],
            "fold": oof_ctrl["fold"],
            "y": oof_ctrl["y"],
            "tail_thr": oof_ctrl["tail_thr"],
            "pred_ctrl": oof_ctrl["pred"],
            "pred_f0": f0_oof["pred"],
        })
        ctrl_res = decide_pair(ctrl_pair_df, advanced_col="pred_ctrl", rival_col="pred_f0", metric="rmsle")
        ci_includes_zero = bool(ctrl_res["ci_lo"] <= 0.0 <= ctrl_res["ci_hi"])
        ctrl_res["ci_includes_zero"] = ci_includes_zero
        ablation_results["controls"][feat] = ctrl_res

    # Save reports/ablation_features.json
    with open(rep_path / "ablation_features.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # Save reports/feature_decision.json
    feature_decision = {
        "decisions": decisions,
        "kept": kept_list,
        "dropped": dropped_list,
    }
    with open(rep_path / "feature_decision.json", "w", encoding="utf-8") as f:
        json.dump(feature_decision, f, indent=2)

    # Save config/features_selected.yaml
    selected_yaml = {
        "kept": kept_list,
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR / "features_selected.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(selected_yaml, f, sort_keys=False)

    return feature_decision


def main() -> None:
    feature_decision = run_feature_ablations()
    print("Feature ablation run complete.")
    print("Kept features:", feature_decision["kept"])
    print("Dropped features:", feature_decision["dropped"])


if __name__ == "__main__":
    main()
