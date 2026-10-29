"""Compare predictions between container API, local uvicorn API, and in-process bundle."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

from gates import lib as gl
from revpred.artifacts import load_bundle

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DEV_PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(url: str, timeout_s: float = 15.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except (httpx.HTTPError, ConnectionError, OSError):
            time.sleep(0.1)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout_s}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare container vs local API predictions.")
    parser.add_argument("--container-url", type=str, required=True, help="URL of running container API.")
    parser.add_argument("--local-url", type=str, default=None, help="URL of running local API. If omitted, starts one.")
    parser.add_argument("--n", type=int, default=500, help="Number of rows to test.")
    parser.add_argument("--out-dir", type=str, default="reports", help="Output directory.")
    args = parser.parse_args()

    local_proc: subprocess.Popen | None = None
    local_url = args.local_url

    if local_url is None:
        port = find_free_port()
        sub_env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
        local_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            cwd=REPO_ROOT,
            env=sub_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        local_url = f"http://127.0.0.1:{port}"

    try:
        wait_for_server(local_url, timeout_s=15.0)
        wait_for_server(args.container_url, timeout_s=15.0)

        seed = int(gl.thr("global.seed"))
        df_dev = pd.read_parquet(DEV_PANEL_PATH)
        sample_df = df_dev.sample(n=args.n, random_state=seed)

        bundle = load_bundle(MODELS_DIR)

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
        max_diff_local = 0.0
        records: list[dict[str, float]] = []

        c_url = args.container_url.rstrip("/")
        l_url = local_url.rstrip("/")

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

                # Container prediction
                r_c = client.post(f"{c_url}/predict", json=payload)
                if r_c.status_code != 200:
                    raise RuntimeError(f"Container predict failed: {r_c.status_code} {r_c.text}")
                pred_c = float(r_c.json()["predicted_revenue_brl"])

                # Local API prediction
                r_l = client.post(f"{l_url}/predict", json=payload)
                if r_l.status_code != 200:
                    raise RuntimeError(f"Local predict failed: {r_l.status_code} {r_l.text}")
                pred_l = float(r_l.json()["predicted_revenue_brl"])

                # Bundle prediction
                pred_b = float(bundle.predict_revenue(payload)[0])

                diff_bundle = abs(pred_c - pred_b)
                diff_local = abs(pred_c - pred_l)

                max_diff_bundle = max(max_diff_bundle, diff_bundle)
                max_diff_local = max(max_diff_local, diff_local)

                records.append({
                    "idx": idx,
                    "container_pred": pred_c,
                    "local_pred": pred_l,
                    "bundle_pred": pred_b,
                    "diff_local": diff_local,
                    "diff_bundle": diff_bundle,
                })

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "container_preds.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n": len(records),
                    "max_diff_bundle": max_diff_bundle,
                    "max_diff_local": max_diff_local,
                    "records": records,
                },
                f,
                indent=2,
            )

        print(f"container_parity_max_abs_brl={max_diff_bundle:.2e}")
        print(f"container_vs_local_api_max_abs_brl={max_diff_local:.2e}")

    finally:
        if local_proc is not None:
            local_proc.terminate()
            try:
                local_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                local_proc.kill()
                local_proc.wait()


if __name__ == "__main__":
    main()

