"""Phase 6 SHAP analysis, cross-checks, noise probe, and report generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder

from revpred.config import get_project_config, get_thresholds
from revpred.cv import expanding_window_splits
from revpred.explain import (
    explain_row,
    explain_rows,
    groups_for,
    linear_shap,
    permutation_importance,
)
from revpred.harness_explain import make_planted_interaction, make_planted_linear
from revpred.models import FullRidgeRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
SHAP_DIR = REPORTS_DIR / "shap"
MODELS_DIR = REPO_ROOT / "models"
DEV_PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"


class ProbedFullRidgeRegressor(FullRidgeRegressor):
    """Subclass of FullRidgeRegressor extending NUM_FEATURES for noise probe."""

    NUM_FEATURES: ClassVar[list[str]] = list(FullRidgeRegressor.NUM_FEATURES) + [
        "noise_1",
        "noise_2",
        "noise_3",
        "permuted_top",
    ]


def run_explain_pipeline() -> None:
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])
    cfg = get_project_config()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading served model and panel data...")
    served_model = joblib.load(MODELS_DIR / "final_model.joblib")
    with open(MODELS_DIR / "explain_background.json", encoding="utf-8") as f:
        bg_stats = json.load(f)

    dev_df = pd.read_parquet(DEV_PANEL_PATH)
    splits = expanding_window_splits(
        dev_df["month"],
        min_train_months=int(cfg["cv"]["min_train_months"]),
        val_months=int(cfg["cv"]["val_months"]),
    )
    last_train_idx, last_val_idx = splits[-1]
    train_fold = dev_df.iloc[last_train_idx].copy().reset_index(drop=True)
    val_fold = dev_df.iloc[last_val_idx].copy().reset_index(drop=True)

    # 300 sampled validation rows with global seed
    val_300 = val_fold.sample(n=300, random_state=seed).copy().reset_index(drop=True)
    # 100 training background rows
    bg_100 = train_fold.sample(n=100, random_state=seed).copy().reset_index(drop=True)

    # =========================================================================
    # 1. Served Model Explanations (API style, dev background)
    # =========================================================================
    print("Computing served model SHAP explanations on 300 dev rows...")
    Z_val_300 = served_model.preprocessor_.transform(val_300)
    coef = served_model.best_estimator_.coef_
    intercept = served_model.best_estimator_.intercept_
    z_bg_mean = np.asarray(bg_stats["z_mean"], dtype=float)
    served_groups = bg_stats["group_map"]
    unique_served_groups = bg_stats["feature_names"]

    phi_served, base_served = linear_shap(Z_val_300, coef, intercept, z_bg_mean, served_groups)
    preds_served = served_model.predict(val_300)

    # Save served model arrays
    np.save(SHAP_DIR / "phi.npy", phi_served)
    np.save(SHAP_DIR / "base.npy", np.array([base_served]))
    np.save(SHAP_DIR / "preds.npy", preds_served)
    np.save(SHAP_DIR / "row_ids.npy", val_300["row_id"].to_numpy())
    with open(SHAP_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(unique_served_groups, f, indent=2)

    # Global summary
    mean_abs_shap = np.mean(np.abs(phi_served), axis=0)
    signed_mean_shap = np.mean(phi_served, axis=0)
    mean_abs_dict = {g: float(v) for g, v in zip(unique_served_groups, mean_abs_shap)}
    signed_mean_dict = {g: float(v) for g, v in zip(unique_served_groups, signed_mean_shap)}

    sorted_features = sorted(mean_abs_dict.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_features[:10]

    lag_features = ["lag_rev_1", "lag_rev_2", "lag_rev_3", "roll_mean_rev_3"]
    lag_group_total = float(sum(mean_abs_dict.get(f, 0.0) for f in lag_features))

    # Coefficient table and signs
    coef_table = []
    group_signs = {}
    for idx_c, (c_val, g_name) in enumerate(zip(coef, served_groups)):
        coef_table.append({
            "column_index": idx_c,
            "group": g_name,
            "coefficient": float(c_val),
        })

    for g in unique_served_groups:
        g_coefs = [c_val for c_val, grp in zip(coef, served_groups) if grp == g]
        if all(c > 0 for c in g_coefs):
            group_signs[g] = "+"
        elif all(c < 0 for c in g_coefs):
            group_signs[g] = "-"
        else:
            group_signs[g] = "mixed"

    shap_global_rep = {
        "mean_abs_shap": mean_abs_dict,
        "signed_mean_shap": signed_mean_dict,
        "top_10_features": top_10,
        "lag_group_total": lag_group_total,
        "coefficient_signs": group_signs,
        "coefficient_table": coef_table,
    }
    with open(REPORTS_DIR / "shap_global.json", "w", encoding="utf-8") as f:
        json.dump(shap_global_rep, f, indent=2)
    print("Saved reports/shap_global.json")

    # =========================================================================
    # 2. Local Examples (5 dev rows)
    # =========================================================================
    print("Generating 5 local explanation examples...")
    local_examples = []
    # Pick 5 rows with varied revenue: highest, lowest, median, and two quartiles
    sorted_val_indices = np.argsort(val_300["revenue"].to_numpy())
    sample_indices = [
        int(sorted_val_indices[-1]),  # max
        int(sorted_val_indices[len(sorted_val_indices) // 4]),
        int(sorted_val_indices[len(sorted_val_indices) // 2]),
        int(sorted_val_indices[3 * len(sorted_val_indices) // 4]),
        int(sorted_val_indices[0]),   # min
    ]
    for r_idx in sample_indices:
        row_dict = val_300.iloc[r_idx].to_dict()
        exp = explain_row(served_model, row_dict, bg_stats, top_k=5)
        local_examples.append({
            "row_id": str(row_dict["row_id"]),
            "series_id": str(row_dict["series_id"]),
            "month": str(row_dict["month"]),
            "revenue": float(row_dict["revenue"]),
            "prediction_log": exp["prediction_log"],
            "prediction_rev": exp["prediction_rev"],
            "base_value": exp["base_value"],
            "attributions": exp["attributions"],
            "top_features": exp["top_features"],
        })

    with open(REPORTS_DIR / "shap_local_examples.json", "w", encoding="utf-8") as f:
        json.dump(local_examples, f, indent=2)
    print("Saved reports/shap_local_examples.json")

    # =========================================================================
    # 3. Model (b): Refit on Training Rows, Validation Cross-Check
    # =========================================================================
    print("Refitting model (b) on training rows for permutation cross-check...")
    model_b = FullRidgeRegressor(cv_splits=3, random_state=seed)
    y_log_train = np.log1p(train_fold["revenue"].to_numpy(dtype=float))
    model_b.fit(train_fold, y_log_train)

    phi_b, _, groups_b = explain_rows(model_b, val_300, bg_100)
    y_log_val = np.log1p(val_300["revenue"].to_numpy(dtype=float))
    perm_importances = permutation_importance(model_b, val_300, y_log_val, n_repeats=10, seed=seed)

    perm_array = np.array([perm_importances[g] for g in groups_b], dtype=float)
    np.save(SHAP_DIR / "model_b_phi.npy", phi_b)
    np.save(SHAP_DIR / "model_b_perm_importance.npy", perm_array)
    with open(SHAP_DIR / "model_b_groups.json", "w", encoding="utf-8") as f:
        json.dump(groups_b, f, indent=2)

    mean_abs_b = np.mean(np.abs(phi_b), axis=0)
    spr_res = spearmanr(mean_abs_b, perm_array)
    spearman_corr = float(spr_res.statistic)

    shap_ranked = [g for _, g in sorted(zip(mean_abs_b, groups_b), reverse=True)]
    perm_ranked = [g for _, g in sorted(zip(perm_array, groups_b), reverse=True)]

    top3_shap = shap_ranked[:3]
    top3_perm = perm_ranked[:3]
    top3_overlap = len(set(top3_shap).intersection(set(top3_perm)))

    crosscheck_rep = {
        "spearman_rank_correlation": spearman_corr,
        "top3_overlap": top3_overlap,
        "top3_shap": top3_shap,
        "top3_perm": top3_perm,
        "shap_ranks": {g: i + 1 for i, g in enumerate(shap_ranked)},
        "perm_ranks": {g: i + 1 for i, g in enumerate(perm_ranked)},
        "shap_values": {g: float(v) for g, v in zip(groups_b, mean_abs_b)},
        "perm_values": perm_importances,
    }
    with open(REPORTS_DIR / "shap_crosscheck.json", "w", encoding="utf-8") as f:
        json.dump(crosscheck_rep, f, indent=2)
    print("Saved reports/shap_crosscheck.json")

    # =========================================================================
    # 4. Noise Probe on Model (b)
    # =========================================================================
    print("Running noise probe with injected N(0, 1) and permuted top feature...")
    top_feature = shap_ranked[0]
    rng_probe = np.random.default_rng(seed)

    train_probe = train_fold.copy()
    train_probe["noise_1"] = rng_probe.standard_normal(len(train_probe))
    train_probe["noise_2"] = rng_probe.standard_normal(len(train_probe))
    train_probe["noise_3"] = rng_probe.standard_normal(len(train_probe))
    train_probe["permuted_top"] = rng_probe.permutation(train_probe[top_feature].values)

    val_probe = val_300.copy()
    val_probe["noise_1"] = rng_probe.standard_normal(len(val_probe))
    val_probe["noise_2"] = rng_probe.standard_normal(len(val_probe))
    val_probe["noise_3"] = rng_probe.standard_normal(len(val_probe))
    val_probe["permuted_top"] = rng_probe.permutation(val_probe[top_feature].values)

    bg_probe = bg_100.copy()
    bg_probe["noise_1"] = rng_probe.standard_normal(len(bg_probe))
    bg_probe["noise_2"] = rng_probe.standard_normal(len(bg_probe))
    bg_probe["noise_3"] = rng_probe.standard_normal(len(bg_probe))
    bg_probe["permuted_top"] = rng_probe.permutation(bg_probe[top_feature].values)

    probed_model = ProbedFullRidgeRegressor(cv_splits=3, random_state=seed)
    y_log_train_p = np.log1p(train_probe["revenue"].to_numpy(dtype=float))
    probed_model.fit(train_probe, y_log_train_p)

    phi_probe, _, groups_probe = explain_rows(probed_model, val_probe, bg_probe)
    np.save(SHAP_DIR / "probe_phi.npy", phi_probe)
    with open(SHAP_DIR / "probe_groups.json", "w", encoding="utf-8") as f:
        json.dump(groups_probe, f, indent=2)

    mean_abs_probe = np.mean(np.abs(phi_probe), axis=0)
    probe_dict = {g: float(v) for g, v in zip(groups_probe, mean_abs_probe)}
    sorted_probe = sorted(probe_dict.items(), key=lambda x: x[1], reverse=True)
    ranked_probe_names = [g for g, _ in sorted_probe]

    top_feature_val = sorted_probe[0][1]
    noise_vals = {
        "noise_1": probe_dict["noise_1"],
        "noise_2": probe_dict["noise_2"],
        "noise_3": probe_dict["noise_3"],
    }
    max_noise_rel = float(max(noise_vals.values()) / top_feature_val)

    top_5_names = ranked_probe_names[:5]
    noise_in_top_k = sum(1 for n in ["noise_1", "noise_2", "noise_3"] if n in top_5_names)
    perm_top_rank = ranked_probe_names.index("permuted_top") + 1

    probe_rep = {
        "top_feature": sorted_probe[0][0],
        "top_feature_mean_abs_shap": top_feature_val,
        "max_noise_rel_to_top": max_noise_rel,
        "noise_in_top_k": noise_in_top_k,
        "noise_features": {
            k: {"mean_abs_shap": v, "rel_to_top": float(v / top_feature_val)}
            for k, v in noise_vals.items()
        },
        "permuted_top": {
            "source_feature": top_feature,
            "rank": perm_top_rank,
            "total_features": len(groups_probe),
            "mean_abs_shap": probe_dict["permuted_top"],
        },
        "all_features_ranked": sorted_probe,
    }
    with open(REPORTS_DIR / "shap_probe.json", "w", encoding="utf-8") as f:
        json.dump(probe_rep, f, indent=2)
    print("Saved reports/shap_probe.json")

    # =========================================================================
    # 5. Planted Truth Harnesses
    # =========================================================================
    print("Running planted truth harnesses...")
    # Oracle exactness (make_planted_linear without noise)
    df_oracle, _ = make_planted_linear(seed=seed, n=5000, noise_std=0.0)
    oracle_preprocessor = ColumnTransformer([
        ("num", "passthrough", ["X1", "X2", "X3", "X4", "X5", "X6"]),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["C"]),
    ])
    Z_oracle = oracle_preprocessor.fit_transform(df_oracle)
    reg_oracle = LinearRegression().fit(Z_oracle, df_oracle["f"].to_numpy())
    z_oracle_bg = np.mean(Z_oracle, axis=0)

    oracle_groups = groups_for(oracle_preprocessor)
    oracle_phi, _ = linear_shap(
        Z_oracle,
        reg_oracle.coef_,
        reg_oracle.intercept_,
        z_oracle_bg,
        oracle_groups,
    )

    # Analytic truth:
    # X1: 2.0, X2: -1.0, X3: 0.5, X4..X6: 0.0, C: e - e_mean
    e_map = {"A": -0.8, "B": -0.4, "C": 0.0, "D": 0.4, "E": 0.8}
    e_arr = np.array([e_map[val] for val in df_oracle["C"]])
    e_mean = float(np.mean(e_arr))
    x1_arr = df_oracle["X1"].to_numpy()
    x2_arr = df_oracle["X2"].to_numpy()
    x3_arr = df_oracle["X3"].to_numpy()
    x4_arr = df_oracle["X4"].to_numpy()
    x5_arr = df_oracle["X5"].to_numpy()
    x6_arr = df_oracle["X6"].to_numpy()

    oracle_truth = np.column_stack([
        2.0 * (x1_arr - float(np.mean(x1_arr))),
        -1.0 * (x2_arr - float(np.mean(x2_arr))),
        0.5 * (x3_arr - float(np.mean(x3_arr))),
        0.0 * (x4_arr - float(np.mean(x4_arr))),
        0.0 * (x5_arr - float(np.mean(x5_arr))),
        0.0 * (x6_arr - float(np.mean(x6_arr))),
        e_arr - e_mean,
    ])

    np.save(SHAP_DIR / "oracle_phi.npy", oracle_phi)
    np.save(SHAP_DIR / "oracle_truth.npy", oracle_truth)

    # Planted linear with noise = 0.3, Ridge model
    df_planted, true_imp_dict = make_planted_linear(seed=seed, n=5000, noise_std=0.3)
    planted_preprocessor = ColumnTransformer([
        ("num", "passthrough", ["X1", "X2", "X3", "X4", "X5", "X6"]),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["C"]),
    ])
    Z_planted = planted_preprocessor.fit_transform(df_planted)
    ridge_planted = Ridge(alpha=1.0, random_state=seed).fit(Z_planted, df_planted["y"].to_numpy())
    planted_groups = groups_for(planted_preprocessor)

    phi_planted, _ = linear_shap(
        Z_planted,
        ridge_planted.coef_,
        ridge_planted.intercept_,
        np.mean(Z_planted, axis=0),
        planted_groups,
    )
    true_imp_arr = np.array([true_imp_dict[g] for g in ["X1", "X2", "X3", "X4", "X5", "X6", "C"]])

    np.save(SHAP_DIR / "planted_linear_phi.npy", phi_planted)
    np.save(SHAP_DIR / "planted_linear_true_imp.npy", true_imp_arr)
    with open(SHAP_DIR / "planted_linear_groups.json", "w", encoding="utf-8") as f:
        json.dump(["X1", "X2", "X3", "X4", "X5", "X6", "C"], f, indent=2)

    # Planted interaction with Ridge
    df_inter = make_planted_interaction(seed=seed, n=5000)
    X_inter = df_inter[["X1", "X2", "X3", "X4", "X5", "X6"]].to_numpy()
    ridge_inter = Ridge(alpha=1.0, random_state=seed).fit(X_inter, df_inter["y"].to_numpy())
    inter_groups = ["X1", "X2", "X3", "X4", "X5", "X6"]
    phi_inter, _ = linear_shap(
        X_inter,
        ridge_inter.coef_,
        ridge_inter.intercept_,
        np.mean(X_inter, axis=0),
        inter_groups,
    )
    np.save(SHAP_DIR / "planted_interaction_phi.npy", phi_inter)
    with open(SHAP_DIR / "planted_interaction_groups.json", "w", encoding="utf-8") as f:
        json.dump(inter_groups, f, indent=2)

    # =========================================================================
    # 6. Library Cross-Check on Real Model
    # =========================================================================
    print("Running shap.LinearExplainer cross-check on served model...")
    Z_all_dev = served_model.preprocessor_.transform(dev_df)
    masker = shap.maskers.Independent(Z_all_dev, max_samples=len(Z_all_dev))
    explainer = shap.LinearExplainer(served_model.best_estimator_, masker)
    lib_shap_res = explainer(Z_val_300)

    # Group aggregation
    lib_group_shap = np.column_stack([
        np.sum(lib_shap_res.values[:, [i for i, g in enumerate(served_groups) if g == u]], axis=1)
        for u in unique_served_groups
    ])

    np.save(SHAP_DIR / "library_shap.npy", lib_group_shap)
    np.save(SHAP_DIR / "library_linear_shap.npy", phi_served)

    print("All Phase 6 explanation artifacts generated successfully.")


if __name__ == "__main__":
    run_explain_pipeline()
