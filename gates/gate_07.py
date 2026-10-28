"""Gate 7: FastAPI service with parity, latency, schema integrity, and tamper evidence."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

from api.app import app
from gates import lib as gl
from revpred.artifacts import ArtifactTampered, load_bundle

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
SNAPSHOT_PATH = REPO_ROOT / "tests" / "snapshots" / "openapi.json"
DEV_PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"


def find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(url: str, timeout_s: float = 15.0) -> None:
    """Wait until server responds ok on /health."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except (httpx.HTTPError, ConnectionError, OSError):
            time.sleep(0.1)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout_s}s")


def build_checks() -> None:
    seed = int(gl.thr("global.seed"))
    target_parity_rows = int(gl.thr("phase7.parity_rows"))
    bench_requests = int(gl.thr("phase7.bench_requests"))

    # =========================================================================
    # 1. Start uvicorn server on free port
    # =========================================================================
    port = find_free_port()
    proc = subprocess.Popen(
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"

    try:
        wait_for_server(url, timeout_s=15.0)

        with httpx.Client(timeout=30.0) as client:
            # =====================================================================
            # 2. Health & Manifest Check
            # =====================================================================
            r_health = client.get(f"{url}/health")
            health_sha_matches = 0
            if r_health.status_code == 200:
                health_data = r_health.json()
                api_model_sha = health_data.get("model_sha")
                manifest_path = MODELS_DIR / "MANIFEST.json"
                if manifest_path.exists():
                    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    exp_sha = manifest_data.get("files", {}).get("models/final_model.joblib")
                    if api_model_sha and exp_sha and api_model_sha == exp_sha:
                        health_sha_matches = 1
            gl.check("health_sha_matches_manifest", health_sha_matches, "==", "global.bool_true")

            # =====================================================================
            # 3. OpenAPI Snapshot Match
            # =====================================================================
            openapi_matches = 0
            if SNAPSHOT_PATH.exists():
                expected_openapi = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
                actual_openapi = app.openapi()
                if actual_openapi == expected_openapi:
                    openapi_matches = 1
            gl.check("openapi_snapshot_matches", openapi_matches, "==", "global.bool_true")

            # =====================================================================
            # 4. Parity and Additivity Checks (n = phase7.parity_rows)
            # =====================================================================
            df_dev = pd.read_parquet(DEV_PANEL_PATH)
            sample_df = df_dev.sample(n=target_parity_rows, random_state=seed)

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
            checked_parity_count = 0

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

                resp = client.post(f"{url}/predict", json=payload)
                if resp.status_code != 200:
                    continue

                checked_parity_count += 1
                api_pred_brl = float(resp.json()["predicted_revenue_brl"])

                bundle_pred_brl = float(bundle.predict_revenue(payload)[0])
                diff_b = abs(api_pred_brl - bundle_pred_brl)
                max_diff_bundle = max(max_diff_bundle, diff_b)

                row_df = pd.DataFrame([payload])
                direct_log = float(direct_model.predict(row_df)[0])
                direct_pred_brl = float(np.clip(np.expm1(direct_log), 0.0, None))
                diff_d = abs(api_pred_brl - direct_pred_brl)
                max_diff_direct = max(max_diff_direct, diff_d)

                if idx < 50:
                    exp_payload = dict(payload)
                    exp_payload["top_k"] = 15
                    resp_exp = client.post(f"{url}/explain", json=exp_payload)
                    if resp_exp.status_code == 200:
                        exp_data = resp_exp.json()
                        base_val = float(exp_data["base_value"])
                        pred_log = float(exp_data["prediction_log1p"])
                        other_shap = float(exp_data["other_shap"])
                        sum_shap = float(sum(c["shap"] for c in exp_data["contributions"]))
                        add_err = abs(base_val + sum_shap + other_shap - pred_log)
                        max_additivity_err = max(max_additivity_err, add_err)

            gl.check("parity_rows", checked_parity_count, ">=", "phase7.parity_rows")
            gl.check("parity_max_abs_brl", max_diff_bundle, "<=", "phase7.parity_max_abs_brl")
            gl.check("parity_vs_direct_joblib_max_abs_brl", max_diff_direct, "<=", "phase7.parity_max_abs_brl")
            gl.check("explain_additivity_max_abs", max_additivity_err, "<=", "phase6.additivity_max_abs")

            # =====================================================================
            # 5. Invalid Payloads (>= phase7.invalid_payload_cases_min)
            # =====================================================================
            meta = client.get(f"{url}/meta").json()
            valid_p = dict(meta["example_payloads"][0])

            invalid_tests: list[tuple[str, str, dict[str, object] | str]] = [
                ("predict", "negative revenue", {**valid_p, "lag_rev_1": -1.0}),
                ("predict", "unknown region", {**valid_p, "region": "Atlantis"}),
                ("predict", "unknown category", {**valid_p, "category": "flying_cars"}),
                ("predict", "month_of_year 13", {**valid_p, "month_of_year": 13}),
                ("predict", "missing field", {k: v for k, v in valid_p.items() if k != "lag_rev_1"}),
                ("predict", "extra field", {**valid_p, "forbidden_extra": "bad"}),
                ("predict", "wrong type string", {**valid_p, "lag_rev_1": "not_a_float"}),
                (
                    "predict_raw",
                    "NaN value",
                    json.dumps(valid_p).replace(str(valid_p["lag_rev_1"]), "NaN", 1),
                ),
                ("predict", "above 10x ceiling", {**valid_p, "lag_rev_1": 1e12}),
                ("explain", "top_k 0", {**valid_p, "top_k": 0}),
                ("explain", "top_k 16", {**valid_p, "top_k": 16}),
            ]

            rejected_count = 0
            non_422_count = 0
            for endpoint, _, p_data in invalid_tests:
                if endpoint == "predict":
                    r = client.post(f"{url}/predict", json=p_data)
                elif endpoint == "explain":
                    r = client.post(f"{url}/explain", json=p_data)
                elif endpoint == "predict_raw":
                    r = client.post(
                        f"{url}/predict",
                        content=str(p_data),
                        headers={"content-type": "application/json"},
                    )
                else:
                    continue

                if r.status_code == 422:
                    rejected_count += 1
                else:
                    non_422_count += 1

            gl.check("invalid_cases_rejected_count", rejected_count, ">=", "phase7.invalid_payload_cases_min")
            gl.check("invalid_cases_not_422", non_422_count, "==", "global.zero")

            # =====================================================================
            # 6. Determinism Check
            # =====================================================================
            r1 = client.post(f"{url}/predict", json=valid_p)
            r2 = client.post(f"{url}/predict", json=valid_p)
            det_predict_ok = r1.status_code == 200 and r1.json() == r2.json()

            exp_valid = {**valid_p, "top_k": 5}
            e1 = client.post(f"{url}/explain", json=exp_valid)
            e2 = client.post(f"{url}/explain", json=exp_valid)
            det_explain_ok = e1.status_code == 200 and e1.json() == e2.json()

            determinism_ok = 1 if (det_predict_ok and det_explain_ok) else 0
            gl.check("determinism_ok", determinism_ok, "==", "global.bool_true")

            # =====================================================================
            # 7. Latency Benchmark (n = phase7.bench_requests)
            # =====================================================================
            # Warmup
            for _ in range(20):
                client.post(f"{url}/predict", json=valid_p)
                client.post(f"{url}/explain", json=exp_valid)

            # Measure predict
            lat_predict: list[float] = []
            for _ in range(bench_requests):
                t0 = time.perf_counter()
                client.post(f"{url}/predict", json=valid_p)
                t1 = time.perf_counter()
                lat_predict.append((t1 - t0) * 1000.0)

            # Measure explain
            lat_explain: list[float] = []
            for _ in range(bench_requests):
                t0 = time.perf_counter()
                client.post(f"{url}/explain", json=exp_valid)
                t1 = time.perf_counter()
                lat_explain.append((t1 - t0) * 1000.0)

            pred_p50 = float(np.percentile(lat_predict, 50))
            pred_p95 = float(np.percentile(lat_predict, 95))
            pred_p99 = float(np.percentile(lat_predict, 99))

            exp_p50 = float(np.percentile(lat_explain, 50))
            exp_p95 = float(np.percentile(lat_explain, 95))
            exp_p99 = float(np.percentile(lat_explain, 99))

            gl.check("predict_p95_ms", pred_p95, "<=", "phase7.predict_p95_ms_max")
            gl.check("explain_p95_ms", exp_p95, "<=", "phase7.explain_p95_ms_max")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    # =========================================================================
    # 8. Tamper Check
    # =========================================================================
    tamper_ok = 1
    # 8a: final_model.joblib
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(MODELS_DIR, td, dirs_exist_ok=True)
        m_file = Path(td) / "final_model.joblib"
        raw = bytearray(m_file.read_bytes())
        raw[10] = (raw[10] + 1) % 256
        m_file.write_bytes(raw)
        try:
            load_bundle(td)
            tamper_ok = 0
        except ArtifactTampered:
            pass
        except (ValueError, KeyError, TypeError, OSError):
            tamper_ok = 0

    # 8b: api_meta.json
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(MODELS_DIR, td, dirs_exist_ok=True)
        m_file = Path(td) / "api_meta.json"
        raw = bytearray(m_file.read_bytes())
        raw[10] = (raw[10] + 1) % 256
        m_file.write_bytes(raw)
        try:
            load_bundle(td)
            tamper_ok = 0
        except ArtifactTampered:
            pass
        except (ValueError, KeyError, TypeError, OSError):
            tamper_ok = 0

    # 8c: explain_background.json
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(MODELS_DIR, td, dirs_exist_ok=True)
        m_file = Path(td) / "explain_background.json"
        raw = bytearray(m_file.read_bytes())
        raw[10] = (raw[10] + 1) % 256
        m_file.write_bytes(raw)
        try:
            load_bundle(td)
            tamper_ok = 0
        except ArtifactTampered:
            pass
        except (ValueError, KeyError, TypeError, OSError):
            tamper_ok = 0

    gl.check("tamper_rejected", tamper_ok, "==", "global.bool_true")

    # =========================================================================
    # 9. Static Forbidden Patterns in api/
    # =========================================================================
    forbidden = [".fit(", "model_selection", "read_parquet", "data/"]
    forbidden_hits = 0
    for py_file in (REPO_ROOT / "api").rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for pat in forbidden:
            if pat in content:
                forbidden_hits += 1

    gl.check("api_forbidden_patterns", forbidden_hits, "==", "global.zero")

    # =========================================================================
    # 10. Holdout Ledger Recovery Touches Check
    # =========================================================================
    ledger_path = REPORTS_DIR / "holdout_ledger.jsonl"
    lines = (
        [line.strip() for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if ledger_path.exists()
        else []
    )
    holdout_recovery_touches = len(lines) - 1
    gl.check("holdout_recovery_touches", holdout_recovery_touches, "<=", "phase5.holdout_recovery_touches_max")

    # =========================================================================
    # 11. Informative Outputs
    # =========================================================================
    gl.info("predict_latency_ms", {"p50": round(pred_p50, 2), "p95": round(pred_p95, 2), "p99": round(pred_p99, 2)})
    gl.info("explain_latency_ms", {"p50": round(exp_p50, 2), "p95": round(exp_p95, 2), "p99": round(exp_p99, 2)})
    gl.info("cpu_count", os.cpu_count())
    gl.info("python_version", sys.version)


def main() -> None:
    gl.run_gate("7", build_checks)


if __name__ == "__main__":
    main()
