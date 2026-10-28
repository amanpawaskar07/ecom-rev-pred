"""Parity evaluation between FastAPI service, Bundle, and direct joblib model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

import gates.lib as gl
from revpred.artifacts import load_bundle

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DEV_PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"


def run_parity(
    base_url: str,
    n: int = 500,
    explain_n: int = 50,
) -> tuple[float, float, float]:
    """Run parity verification and return (max_diff_bundle, max_diff_direct, max_additivity_err)."""
    seed = int(gl.thr("global.seed"))

    df_dev = pd.read_parquet(DEV_PANEL_PATH)
    sample_df = df_dev.sample(n=n, random_state=seed)

    bundle = load_bundle(MODELS_DIR)
    direct_model = joblib.load(MODELS_DIR / "final_model.joblib")

    numeric_features = [
        "lag_rev_1",
        "lag_rev_2",
        "lag_rev_3",
        "lag_lines_1",
        "lag_lines_2",
        "lag_lines_3",
        "roll_mean_rev_3",
        "trailing_price_3",
        "active_months_before",
        "months_since_first_sale",
        "month_of_year",
    ]
    int_features = {"active_months_before", "months_since_first_sale", "month_of_year"}

    max_diff_bundle = 0.0
    max_diff_direct = 0.0
    max_additivity_err = 0.0

    url = base_url.rstrip("/")
    with httpx.Client(timeout=30.0) as client:
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            payload: dict[str, object] = {}
            for feat in numeric_features:
                val = row[feat]
                if feat in int_features:
                    payload[feat] = int(val)
                else:
                    payload[feat] = float(val)
            payload["region"] = str(row["region"])
            payload["category"] = str(row["category"])

            # 1. API prediction
            resp = client.post(f"{url}/predict", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"POST /predict failed with status {resp.status_code}: {resp.text}")
            api_pred_brl = float(resp.json()["predicted_revenue_brl"])

            # 2. Bundle prediction
            bundle_pred_brl = float(bundle.predict_revenue(payload)[0])
            diff_bundle = abs(api_pred_brl - bundle_pred_brl)
            max_diff_bundle = max(max_diff_bundle, diff_bundle)

            # 3. Direct joblib prediction
            row_df = pd.DataFrame([payload])
            direct_log = float(direct_model.predict(row_df)[0])
            direct_pred_brl = float(np.clip(np.expm1(direct_log), 0.0, None))
            diff_direct = abs(api_pred_brl - direct_pred_brl)
            max_diff_direct = max(max_diff_direct, diff_direct)

            # 4. Explain check on first explain_n rows
            if idx < explain_n:
                explain_payload = dict(payload)
                explain_payload["top_k"] = 15
                resp_exp = client.post(f"{url}/explain", json=explain_payload)
                if resp_exp.status_code != 200:
                    raise RuntimeError(f"POST /explain failed with status {resp_exp.status_code}: {resp_exp.text}")
                exp_data = resp_exp.json()
                base_val = float(exp_data["base_value"])
                pred_log = float(exp_data["prediction_log1p"])
                other_shap = float(exp_data["other_shap"])
                sum_shap = float(sum(c["shap"] for c in exp_data["contributions"]))

                additivity_err = abs(base_val + sum_shap + other_shap - pred_log)
                max_additivity_err = max(max_additivity_err, additivity_err)

    print(f"Parity checked rows: {n}")
    print(f"Max abs diff (API vs Bundle) [BRL]: {max_diff_bundle:.2e}")
    print(f"Max abs diff (API vs Direct Joblib) [BRL]: {max_diff_direct:.2e}")
    print(f"Explain checked rows: {explain_n}")
    print(f"Max explain additivity error: {max_additivity_err:.2e}")

    return max_diff_bundle, max_diff_direct, max_additivity_err


def main() -> None:
    parser = argparse.ArgumentParser(description="Check parity between API, Bundle, and direct joblib model.")
    parser.add_argument("--url", type=str, required=True, help="Base URL of the running API server.")
    parser.add_argument("--n", type=int, default=500, help="Number of dev rows to test.")
    args = parser.parse_args()

    max_b, max_d, max_a = run_parity(args.url, n=args.n)
    max_allowed = float(gl.thr("phase7.parity_max_abs_brl"))
    max_additivity_allowed = float(gl.thr("phase6.additivity_max_abs"))

    if max_b > max_allowed or max_d > max_allowed or max_a > max_additivity_allowed:
        print("Parity or additivity threshold exceeded!", file=sys.stderr)
        sys.exit(1)
    print("Parity checks PASSED")


if __name__ == "__main__":
    main()
