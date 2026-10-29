"""End-to-end harness test for Phases 1-6 pipelines using synthetic planted data."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.app import app
from gates import lib as gl
from revpred.artifacts import load_bundle
from revpred.cv import expanding_window_splits
from revpred.explain import explain_rows
from revpred.features import FeatureBuilder
from revpred.harness_e2e import make_synthetic_orders, reconcile_synthetic_csvs
from revpred.models import FullRidgeRegressor
from revpred.panel import build_panel

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR = REPO_ROOT / "reports" / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def run_harness_e2e() -> bool:
    results: dict[str, Any] = {}
    temp_dir = Path(tempfile.mkdtemp(prefix="harness_e2e_", dir=SCRATCH_DIR))

    try:
        # Step 1: Generate synthetic CSVs
        months = [f"2017-{m:02d}" for m in range(1, 13)] + ["2018-01", "2018-02"]
        csv_paths = make_synthetic_orders(temp_dir, seed=42, months=months, n_products=30)
        results["synthetic_csvs_count"] = len(csv_paths)
        assert len(csv_paths) == 5, f"Expected 5 CSVs, got {len(csv_paths)}"

        # Step 2: Reconciliation
        df_ol, recon_stats = reconcile_synthetic_csvs(temp_dir)
        recon_tol = float(gl.thr("phase1.recon_abs_tol_brl"))
        reconciliation_ok = bool(
            recon_stats["recon_abs_diff"] <= recon_tol
            and recon_stats["unmatched_rev"] == 0.0
            and recon_stats["n_order_lines"] > 0
        )
        results["reconciliation_ok"] = reconciliation_ok
        results["recon_abs_diff"] = recon_stats["recon_abs_diff"]
        results["n_order_lines"] = recon_stats["n_order_lines"]

        # Step 3: Panel creation & Truncation Invariance
        # Build panel on full synthetic data
        panel_full = build_panel(df_ol, entity="product_id", window=months)
        results["panel_rows"] = len(panel_full)
        panel_ok = bool(len(panel_full) > 0 and "revenue" in panel_full.columns)
        results["panel_ok"] = panel_ok

        # Truncation invariance: truncate at 2018-01
        cut_month = "2018-01"
        df_ol_cut = df_ol[df_ol["month"] < cut_month].copy()
        panel_cut = build_panel(df_ol_cut, entity="product_id", window=months)

        # Check rows for historical months strictly before cut_month
        history_months = [m for m in months[3:] if m < "2017-12"]
        p_full_hist = (
            panel_full[panel_full["month"].isin(history_months)]
            .sort_values(["month", "series_id"])
            .reset_index(drop=True)
        )
        p_cut_hist = (
            panel_cut[panel_cut["month"].isin(history_months)]
            .sort_values(["month", "series_id"])
            .reset_index(drop=True)
        )

        num_cols = ["lag_rev_1", "lag_rev_2", "lag_rev_3", "roll_mean_rev_3", "revenue"]
        diff_truncation = float(
            (p_full_hist[num_cols].to_numpy() - p_cut_hist[num_cols].to_numpy()).max()
        )
        truncation_invariance_ok = bool(
            len(p_full_hist) == len(p_cut_hist) and len(p_full_hist) > 0 and diff_truncation == 0.0
        )
        results["truncation_invariance_ok"] = truncation_invariance_ok
        results["truncation_diff"] = diff_truncation

        # Step 4: Features
        # Fit FeatureBuilder on training months, transform validation month
        fb = FeatureBuilder(order_lines=df_ol)
        train_months = months[3:9]  # 6 months
        val_month = months[9]       # 1 month

        train_df = panel_full[panel_full["month"].isin(train_months)].copy()
        val_df = panel_full[panel_full["month"] == val_month].copy()

        fb.fit(train_df, val_month=val_month)
        train_feat = fb.transform(train_df)
        val_feat = fb.transform(val_df)

        expected_feats = {"price_bucket", "high_value_flag", "cat_region_level"}
        features_ok = bool(
            expected_feats.issubset(train_feat.columns)
            and expected_feats.issubset(val_feat.columns)
            and not train_feat[list(expected_feats)].isna().any().any()
            and not val_feat[list(expected_feats)].isna().any().any()
        )
        results["features_ok"] = features_ok

        # Step 5: CV Splits
        splits = expanding_window_splits(panel_full["month"], min_train_months=6, val_months=1)
        cv_ok = bool(len(splits) >= 1)
        for t_idx, v_idx in splits:
            t_m = panel_full.iloc[t_idx]["month"]
            v_m = panel_full.iloc[v_idx]["month"]
            if t_m.max() >= v_m.min() or len(set(t_idx).intersection(set(v_idx))) > 0:
                cv_ok = False
        results["cv_splits_count"] = len(splits)
        results["cv_ok"] = cv_ok

        # Step 6: Linear SHAP additivity
        # Fit FullRidgeRegressor on train_df
        y_train = train_df["revenue"].to_numpy()
        model = FullRidgeRegressor(cv_splits=3, random_state=42)
        model.fit(train_df, np.log1p(y_train))

        bg_df = train_df.head(20).copy()
        eval_df = val_df.head(20).copy()

        phi, base_val, _grp_names = explain_rows(model, eval_df, bg_df)
        pred_log = model.predict(eval_df)
        shap_recon = np.sum(phi, axis=1) + base_val
        max_shap_additivity_err = float(np.max(np.abs(shap_recon - pred_log)))
        additivity_max_abs = float(gl.thr("phase6.additivity_max_abs"))

        shap_additivity_ok = bool(max_shap_additivity_err <= additivity_max_abs)
        results["max_shap_additivity_err"] = max_shap_additivity_err
        results["shap_additivity_ok"] = shap_additivity_ok

        # Step 7: Prediction Parity on 50 seeded rows inside api_meta feature ranges
        # Using committed model bundle and local TestClient
        models_dir = REPO_ROOT / "models"
        bundle = load_bundle(models_dir)
        direct_model = joblib.load(models_dir / "final_model.joblib")

        with open(models_dir / "api_meta.json", encoding="utf-8") as f:
            api_meta = json.load(f)

        rng = np.random.default_rng(int(gl.thr("global.seed")))
        seeded_rows = []
        n_seeded = 50
        categories = api_meta["categories"]
        regions = api_meta["regions"]
        ranges = api_meta["feature_ranges"]

        for _ in range(n_seeded):
            row: dict[str, Any] = {}
            for feat, bounds in ranges.items():
                min_v = bounds["min"]
                max_v = bounds["max"]
                if feat in {"active_months_before", "months_since_first_sale", "month_of_year"}:
                    row[feat] = int(rng.integers(int(min_v), int(max_v) + 1))
                else:
                    row[feat] = float(rng.uniform(min_v, max_v))
            row["region"] = str(rng.choice(regions))
            row["category"] = str(rng.choice(categories))
            seeded_rows.append(row)

        max_parity_diff_bundle = 0.0
        max_parity_diff_api = 0.0
        parity_tol = float(gl.thr("phase7.parity_max_abs_brl"))

        with TestClient(app) as client:
            for s_row in seeded_rows:
                # Direct joblib prediction
                df_single = pd.DataFrame([s_row])
                direct_log = float(direct_model.predict(df_single)[0])
                direct_pred = float(np.clip(np.expm1(direct_log), 0.0, None))

                # Bundle prediction
                bundle_pred = float(bundle.predict_revenue(s_row)[0])
                diff_b = abs(bundle_pred - direct_pred)
                max_parity_diff_bundle = max(max_parity_diff_bundle, diff_b)

                # API prediction
                resp = client.post("/predict", json=s_row)
                assert resp.status_code == 200, f"API returned {resp.status_code}: {resp.text}"
                api_pred = float(resp.json()["predicted_revenue_brl"])
                diff_api = abs(api_pred - direct_pred)
                max_parity_diff_api = max(max_parity_diff_api, diff_api)

        parity_50_rows_ok = bool(
            max_parity_diff_bundle <= parity_tol and max_parity_diff_api <= parity_tol
        )
        results["max_parity_diff_bundle"] = max_parity_diff_bundle
        results["max_parity_diff_api"] = max_parity_diff_api
        results["parity_50_rows_ok"] = parity_50_rows_ok

        all_ok = bool(
            reconciliation_ok
            and panel_ok
            and truncation_invariance_ok
            and features_ok
            and cv_ok
            and shap_additivity_ok
            and parity_50_rows_ok
        )
        results["harness_e2e_ok"] = all_ok

        # Print key=value lines
        for k, v in results.items():
            print(f"{k}={v}")

        # Save summary to scratch
        out_summary = SCRATCH_DIR / "harness_e2e_summary.json"
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return all_ok

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    ok = run_harness_e2e()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
