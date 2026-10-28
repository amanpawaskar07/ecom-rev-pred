"""Tests for Phase 6: SHAP explanations, planted truth, permutation cross-check, and noise probe."""
from __future__ import annotations

import hashlib
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
    check_additivity,
    check_group_coverage,
    explain_row,
    explain_rows,
    groups_for,
    linear_shap,
    permutation_importance,
)
from revpred.harness_explain import make_planted_interaction, make_planted_linear
from revpred.models import FullRidgeRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DEV_PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"


def test_oracle_exactness() -> None:
    """Oracle exactness: linear_shap matches analytic truth within oracle_shap_max_abs."""
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])
    tol = float(thr["phase6"]["oracle_shap_max_abs"])

    df, _ = make_planted_linear(seed=seed, n=5000, noise_std=0.0)
    preprocessor = ColumnTransformer([
        ("num", "passthrough", ["X1", "X2", "X3", "X4", "X5", "X6"]),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["C"]),
    ])
    Z = preprocessor.fit_transform(df)
    reg = LinearRegression().fit(Z, df["f"].to_numpy())
    z_bg = np.mean(Z, axis=0)

    groups = groups_for(preprocessor)
    phi, _ = linear_shap(Z, reg.coef_, reg.intercept_, z_bg, groups)

    # Analytic truth
    e_map = {"A": -0.8, "B": -0.4, "C": 0.0, "D": 0.4, "E": 0.8}
    e_arr = np.array([e_map[val] for val in df["C"]])
    e_mean = float(np.mean(e_arr))

    x1_arr = df["X1"].to_numpy()
    x2_arr = df["X2"].to_numpy()
    x3_arr = df["X3"].to_numpy()
    x4_arr = df["X4"].to_numpy()
    x5_arr = df["X5"].to_numpy()
    x6_arr = df["X6"].to_numpy()

    phi_truth = np.column_stack([
        2.0 * (x1_arr - float(np.mean(x1_arr))),
        -1.0 * (x2_arr - float(np.mean(x2_arr))),
        0.5 * (x3_arr - float(np.mean(x3_arr))),
        0.0 * (x4_arr - float(np.mean(x4_arr))),
        0.0 * (x5_arr - float(np.mean(x5_arr))),
        0.0 * (x6_arr - float(np.mean(x6_arr))),
        e_arr - e_mean,
    ])

    max_err = float(np.max(np.abs(phi - phi_truth)))
    assert max_err <= tol, f"Oracle SHAP max abs error {max_err} > {tol}"


def test_library_crosscheck_real_model() -> None:
    """Library cross-check: group-aggregated shap.LinearExplainer vs linear_shap on served model."""
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])
    tol = float(thr["phase6"]["oracle_shap_max_abs"])

    model = joblib.load(MODELS_DIR / "final_model.joblib")
    dev_df = pd.read_parquet(DEV_PANEL_PATH)
    sample = dev_df.sample(n=100, random_state=seed)

    Z_all = model.preprocessor_.transform(dev_df)
    Z_sample = model.preprocessor_.transform(sample)

    masker = shap.maskers.Independent(Z_all, max_samples=len(Z_all))
    explainer = shap.LinearExplainer(model.best_estimator_, masker)
    lib_res = explainer(Z_sample)

    groups = groups_for(model.preprocessor_)
    unique_groups = list(dict.fromkeys(groups))
    lib_group_shap = np.column_stack([
        np.sum(lib_res.values[:, [i for i, g in enumerate(groups) if g == u]], axis=1)
        for u in unique_groups
    ])

    phi_analytic, base_analytic = linear_shap(
        Z_sample,
        model.best_estimator_.coef_,
        model.best_estimator_.intercept_,
        np.mean(Z_all, axis=0),
        groups,
    )

    max_diff = float(np.max(np.abs(lib_group_shap - phi_analytic)))
    assert max_diff <= tol, f"Library vs analytic max abs diff {max_diff} > {tol}"

    base_diff = float(abs(explainer.expected_value - base_analytic))
    assert base_diff <= tol, f"Base value diff {base_diff} > {tol}"


def test_learned_recovery_planted_linear() -> None:
    """Learned recovery on make_planted_linear: rank correlation and noise share."""
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])
    min_spearman = float(thr["phase6"]["planted_rank_spearman_min"])
    max_noise_share = float(thr["phase6"]["planted_noise_share_max"])

    df, true_imp_dict = make_planted_linear(seed=seed, n=5000, noise_std=0.3)
    preprocessor = ColumnTransformer([
        ("num", "passthrough", ["X1", "X2", "X3", "X4", "X5", "X6"]),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["C"]),
    ])
    Z = preprocessor.fit_transform(df)
    ridge = Ridge(alpha=1.0, random_state=seed).fit(Z, df["y"].to_numpy())

    groups = groups_for(preprocessor)
    phi, _ = linear_shap(Z, ridge.coef_, ridge.intercept_, np.mean(Z, axis=0), groups)

    group_names = ["X1", "X2", "X3", "X4", "X5", "X6", "C"]
    mean_abs = np.mean(np.abs(phi), axis=0)
    true_imp = [true_imp_dict[g] for g in group_names]

    spr = float(spearmanr(mean_abs, true_imp).statistic)
    assert spr >= min_spearman, f"Planted rank spearman {spr} < {min_spearman}"

    # Noise share of X4, X5, X6 (indices 3, 4, 5)
    noise_share = float((mean_abs[3] + mean_abs[4] + mean_abs[5]) / np.sum(mean_abs))
    assert noise_share <= max_noise_share, f"Planted noise share {noise_share} > {max_noise_share}"


def test_interaction_blindness() -> None:
    """Interaction blindness: Ridge cannot represent X2*X3; combined share <= threshold."""
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])
    max_share = float(thr["phase6"]["planted_noise_share_max"])

    df = make_planted_interaction(seed=seed, n=5000)
    X = df[["X1", "X2", "X3", "X4", "X5", "X6"]].to_numpy()
    ridge = Ridge(alpha=1.0, random_state=seed).fit(X, df["y"].to_numpy())

    phi, _ = linear_shap(X, ridge.coef_, ridge.intercept_, np.mean(X, axis=0), ["X1", "X2", "X3", "X4", "X5", "X6"])
    mean_abs = np.mean(np.abs(phi), axis=0)

    # Combined share of X2 (index 1) and X3 (index 2)
    x2_x3_share = float((mean_abs[1] + mean_abs[2]) / np.sum(mean_abs))
    assert x2_x3_share <= max_share, f"Interaction combined share {x2_x3_share} > {max_share}"


def test_additivity_served_model() -> None:
    """Additivity on served model: max |sum(phi) + base - prediction| <= additivity_max_abs."""
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])
    tol = float(thr["phase6"]["additivity_max_abs"])

    model = joblib.load(MODELS_DIR / "final_model.joblib")
    with open(MODELS_DIR / "explain_background.json", encoding="utf-8") as f:
        bg_stats = json.load(f)

    dev_df = pd.read_parquet(DEV_PANEL_PATH)
    sample = dev_df.sample(n=300, random_state=seed)

    Z_sample = model.preprocessor_.transform(sample)
    coef = model.best_estimator_.coef_
    intercept = model.best_estimator_.intercept_
    z_mean = np.asarray(bg_stats["z_mean"], dtype=float)
    groups = bg_stats["group_map"]

    phi, base_value = linear_shap(Z_sample, coef, intercept, z_mean, groups)
    preds_log = model.predict(sample)

    assert check_additivity(phi, base_value, preds_log, max_abs_err=tol)


def test_permutation_crosscheck() -> None:
    """Permutation cross-check on model (b): Spearman rank and top-3 overlap."""
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])
    cfg = get_project_config()
    min_spearman = float(thr["phase6"]["perm_vs_shap_spearman_min"])
    min_overlap = int(thr["phase6"]["top3_overlap_min"])

    dev_df = pd.read_parquet(DEV_PANEL_PATH)
    splits = expanding_window_splits(
        dev_df["month"],
        min_train_months=int(cfg["cv"]["min_train_months"]),
        val_months=int(cfg["cv"]["val_months"]),
    )
    last_train_idx, last_val_idx = splits[-1]
    train_fold = dev_df.iloc[last_train_idx].copy().reset_index(drop=True)
    val_fold = dev_df.iloc[last_val_idx].copy().reset_index(drop=True)

    val_300 = val_fold.sample(n=300, random_state=seed).copy().reset_index(drop=True)
    bg_100 = train_fold.sample(n=100, random_state=seed).copy().reset_index(drop=True)

    model_b = FullRidgeRegressor(cv_splits=3, random_state=seed)
    y_log_train = np.log1p(train_fold["revenue"].to_numpy(dtype=float))
    model_b.fit(train_fold, y_log_train)

    phi_b, _, groups_b = explain_rows(model_b, val_300, bg_100)
    y_log_val = np.log1p(val_300["revenue"].to_numpy(dtype=float))
    perm_importances = permutation_importance(model_b, val_300, y_log_val, n_repeats=10, seed=seed)

    mean_abs_b = np.mean(np.abs(phi_b), axis=0)
    perm_arr = np.array([perm_importances[g] for g in groups_b])

    spr = float(spearmanr(mean_abs_b, perm_arr).statistic)
    assert spr >= min_spearman, f"Perm vs SHAP spearman {spr} < {min_spearman}"

    shap_ranked = [g for _, g in sorted(zip(mean_abs_b, groups_b), reverse=True)]
    perm_ranked = [g for _, g in sorted(zip(perm_arr, groups_b), reverse=True)]

    top3_shap = set(shap_ranked[:3])
    top3_perm = set(perm_ranked[:3])
    overlap = len(top3_shap.intersection(top3_perm))
    assert overlap >= min_overlap, f"Top-3 overlap {overlap} < {min_overlap}"


def test_noise_probe() -> None:
    """Noise probe: injected noise features rank low and carry negligible attribution."""
    thr = get_thresholds()
    seed = int(thr["global"]["seed"])
    cfg = get_project_config()
    max_noise_rel = float(thr["phase6"]["noise_feature_max_rel_to_top"])
    forbidden_k = int(thr["phase6"]["noise_feature_top_k_forbidden"])

    dev_df = pd.read_parquet(DEV_PANEL_PATH)
    splits = expanding_window_splits(
        dev_df["month"],
        min_train_months=int(cfg["cv"]["min_train_months"]),
        val_months=int(cfg["cv"]["val_months"]),
    )
    last_train_idx, last_val_idx = splits[-1]
    train_fold = dev_df.iloc[last_train_idx].copy().reset_index(drop=True)
    val_fold = dev_df.iloc[last_val_idx].copy().reset_index(drop=True)

    val_300 = val_fold.sample(n=300, random_state=seed).copy().reset_index(drop=True)
    bg_100 = train_fold.sample(n=100, random_state=seed).copy().reset_index(drop=True)

    top_feature = "roll_mean_rev_3"
    rng = np.random.default_rng(seed)

    class ProbedModel(FullRidgeRegressor):
        NUM_FEATURES: ClassVar[list[str]] = list(FullRidgeRegressor.NUM_FEATURES) + [
            "noise_1",
            "noise_2",
            "noise_3",
            "permuted_top",
        ]

    train_p = train_fold.copy()
    train_p["noise_1"] = rng.standard_normal(len(train_p))
    train_p["noise_2"] = rng.standard_normal(len(train_p))
    train_p["noise_3"] = rng.standard_normal(len(train_p))
    train_p["permuted_top"] = rng.permutation(train_p[top_feature].values)

    val_p = val_300.copy()
    val_p["noise_1"] = rng.standard_normal(len(val_p))
    val_p["noise_2"] = rng.standard_normal(len(val_p))
    val_p["noise_3"] = rng.standard_normal(len(val_p))
    val_p["permuted_top"] = rng.permutation(val_p[top_feature].values)

    bg_p = bg_100.copy()
    bg_p["noise_1"] = rng.standard_normal(len(bg_p))
    bg_p["noise_2"] = rng.standard_normal(len(bg_p))
    bg_p["noise_3"] = rng.standard_normal(len(bg_p))
    bg_p["permuted_top"] = rng.permutation(bg_p[top_feature].values)

    model_p = ProbedModel(cv_splits=3, random_state=seed)
    model_p.fit(train_p, np.log1p(train_p["revenue"].to_numpy(dtype=float)))

    phi_p, _, groups_p = explain_rows(model_p, val_p, bg_p)
    mean_abs_p = np.mean(np.abs(phi_p), axis=0)
    score_map = {g: float(v) for g, v in zip(groups_p, mean_abs_p)}

    sorted_scores = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    top_score = sorted_scores[0][1]

    noise_scores = [score_map["noise_1"], score_map["noise_2"], score_map["noise_3"]]
    actual_max_rel = max(noise_scores) / top_score
    assert actual_max_rel <= max_noise_rel, f"Noise max rel to top {actual_max_rel} > {max_noise_rel}"

    ranked_names = [g for g, _ in sorted_scores]
    top_k_names = ranked_names[:forbidden_k]
    noise_in_top_k = sum(1 for n in ["noise_1", "noise_2", "noise_3"] if n in top_k_names)
    assert noise_in_top_k == 0, f"Found {noise_in_top_k} noise features in top {forbidden_k}"


def test_group_coverage_structure() -> None:
    """Group coverage: all transformed columns are assigned to valid groups with no gaps."""
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    groups = groups_for(model.preprocessor_)
    assert len(groups) == 83
    assert check_group_coverage(model.preprocessor_, groups)


def test_explain_row() -> None:
    """explain_row produces accurate single-row explanations with valid top_k sorting."""
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    with open(MODELS_DIR / "explain_background.json", encoding="utf-8") as f:
        bg_stats = json.load(f)

    dev_df = pd.read_parquet(DEV_PANEL_PATH)
    sample_row = dev_df.iloc[0].to_dict()

    exp = explain_row(model, sample_row, bg_stats, top_k=5)
    assert "base_value" in exp
    assert "prediction_log" in exp
    assert "attributions" in exp
    assert len(exp["top_features"]) == 5

    # Prediction matches model
    df_single = pd.DataFrame([sample_row])
    expected_pred = float(model.predict(df_single)[0])
    assert abs(exp["prediction_log"] - expected_pred) < 1e-6


def test_explain_manifest_integrity() -> None:
    """EXPLAIN_MANIFEST.json matches sha256 of all bound artifacts."""
    manifest_path = MODELS_DIR / "EXPLAIN_MANIFEST.json"
    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest.get("files", {})
    assert len(files) == 3
    for rel_path, expected_sha in files.items():
        actual_sha = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Hash mismatch for {rel_path}"
