"""Gate 9: Docker Compose with container-vs-local parity and security checks."""
from __future__ import annotations

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
REPORTS_DIR = REPO_ROOT / "reports"
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
    # =========================================================================
    # 1. Start local uvicorn API server on free port
    # =========================================================================
    local_port = find_free_port()
    sub_env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONHASHSEED="0")
    local_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(local_port),
            "--workers",
            "1",
        ],
        cwd=REPO_ROOT,
        env=sub_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    local_url = f"http://127.0.0.1:{local_port}"

    scratch_dir = REPORTS_DIR / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    parsed_kv: dict[str, str] = {}

    try:
        wait_for_server(local_url, timeout_s=15.0)

        # =====================================================================
        # 2. Invoke scripts/container_check.sh
        # =====================================================================
        script_env = dict(
            os.environ,
            OUT_DIR="reports/scratch",
            LOCAL_API_URL=local_url,
            PYTHONPATH=str(REPO_ROOT),
            PYTHONHASHSEED="0",
        )
        proc_script = subprocess.run(
            ["bash", "scripts/container_check.sh"],
            cwd=REPO_ROOT,
            env=script_env,
            capture_output=True,
            text=True,
            check=False,
        )

        for line in proc_script.stdout.splitlines():
            line_str = line.strip()
            if "=" in line_str and not line_str.startswith("#"):
                k, v = line_str.split("=", 1)
                parsed_kv[k.strip()] = v.strip()

        # =====================================================================
        # 3. Independent Measurements
        # =====================================================================
        # Verify container predictions independently against local API and bundle
        container_preds_file = scratch_dir / "container_preds.json"
        assert container_preds_file.exists(), f"Missing {container_preds_file}"
        with open(container_preds_file, encoding="utf-8") as f:
            container_pred_data = json.load(f)

        records = container_pred_data.get("records", [])
        assert len(records) >= 500, f"Expected at least 500 records, got {len(records)}"

        seed = int(gl.thr("global.seed"))
        df_dev = pd.read_parquet(DEV_PANEL_PATH)
        sample_df = df_dev.sample(n=len(records), random_state=seed)

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

        indep_max_bundle_diff = 0.0
        indep_max_local_diff = 0.0

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

                r_local = client.post(f"{local_url}/predict", json=payload)
                assert r_local.status_code == 200, f"Local predict failed: {r_local.text}"
                local_pred_brl = float(r_local.json()["predicted_revenue_brl"])

                bundle_pred_brl = float(bundle.predict_revenue(payload)[0])
                c_pred_brl = float(records[idx]["container_pred"])

                diff_b = abs(c_pred_brl - bundle_pred_brl)
                diff_l = abs(c_pred_brl - local_pred_brl)

                indep_max_bundle_diff = max(indep_max_bundle_diff, diff_b)
                indep_max_local_diff = max(indep_max_local_diff, diff_l)

        # Check containers_healthy_within_s
        healthy_time = float(parsed_kv.get("containers_healthy_within_s", 999.0))
        gl.check("containers_healthy_within_s", healthy_time, "<=", "phase9.healthcheck_timeout_s")

        # Check container_parity_max_abs_brl
        parsed_bundle_diff = float(parsed_kv.get("container_parity_max_abs_brl", 999.0))
        max_bundle_diff = max(indep_max_bundle_diff, parsed_bundle_diff)
        gl.check("container_parity_max_abs_brl", max_bundle_diff, "<=", "phase9.container_parity_max_abs_brl")

        # Check container_vs_local_api_max_abs_brl
        parsed_local_diff = float(parsed_kv.get("container_vs_local_api_max_abs_brl", 999.0))
        max_local_diff = max(indep_max_local_diff, parsed_local_diff)
        gl.check("container_vs_local_api_max_abs_brl", max_local_diff, "<=", "phase9.container_parity_max_abs_brl")

        # Check container_model_sha_matches_local
        container_model_sha = parsed_kv.get("container_model_sha", "")
        manifest_path = MODELS_DIR / "MANIFEST.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest_data.get("files", {}).get("models/final_model.joblib", "")
        sha_matches = 1 if container_model_sha and expected_sha and container_model_sha == expected_sha else 0
        gl.check("container_model_sha_matches_local", sha_matches, "==", "global.bool_true")

        # Check predict_p95_ms_container
        bench_file = scratch_dir / "bench_api_container.json"
        assert bench_file.exists(), f"Missing {bench_file}"
        with open(bench_file, encoding="utf-8") as f:
            bench_data = json.load(f)
        p95_container = float(bench_data["predict"]["p95_ms"])
        gl.check("predict_p95_ms_container", p95_container, "<=", "phase9.predict_p95_ms_max")

        # Check ui_health_ok
        ui_ok = int(parsed_kv.get("ui_health_ok", 0))
        gl.check("ui_health_ok", ui_ok, "==", "global.bool_true")

        # =====================================================================
        # 4. Forbidden paths in images
        # =====================================================================
        images = ["revpred-api:p9", "revpred-ui:p9"]
        forbidden_hits = 0
        for img in images:
            proc_find = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--entrypoint",
                    "sh",
                    img,
                    "-c",
                    "find /app -name data -o -name *.csv -o -name *.parquet -o -name tests -o -name .git",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            hits = [h for h in proc_find.stdout.strip().splitlines() if h.strip()]
            forbidden_hits += len(hits)

        gl.check("forbidden_paths_in_images", forbidden_hits, "==", "global.zero")

        # =====================================================================
        # 5. Image runs as non-root (uid != 0)
        # =====================================================================
        non_root_ok = 1
        for img in images:
            proc_id = subprocess.run(
                ["docker", "run", "--rm", "--entrypoint", "id", img, "-u"],
                capture_output=True,
                text=True,
                check=False,
            )
            uid_str = proc_id.stdout.strip()
            if uid_str == "0" or uid_str == "root" or not uid_str:
                non_root_ok = 0
        gl.check("image_runs_as_nonroot", non_root_ok, "==", "global.bool_true")

        # =====================================================================
        # 6. Compose down leaves no containers
        # =====================================================================
        proc_ps = subprocess.run(
            ["docker", "compose", "-p", "revpred-p9", "ps", "-aq"],
            capture_output=True,
            text=True,
            check=False,
        )
        remaining_containers = len([c for c in proc_ps.stdout.strip().splitlines() if c.strip()])
        compose_leaves_none = 1 if remaining_containers == 0 else 0
        gl.check("compose_down_leaves_no_containers", compose_leaves_none, "==", "global.bool_true")

        # =========================================================================
        # 7. Holdout Ledger Recovery Touches Check
        # =========================================================================
        ledger_path = REPORTS_DIR / "holdout_ledger.jsonl"
        lines = (
            [line.strip() for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if ledger_path.exists()
            else []
        )
        holdout_recovery_touches = len(lines) - 1
        gl.check("holdout_recovery_touches", holdout_recovery_touches, "<=", "phase5.holdout_recovery_touches_max")

        # =====================================================================
        # 8. Informative Outputs
        # =====================================================================
        image_sizes: dict[str, str] = {}
        for img in images:
            proc_insp = subprocess.run(
                ["docker", "image", "inspect", img],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc_insp.returncode == 0:
                insp_data = json.loads(proc_insp.stdout)
                size_mb = insp_data[0].get("Size", 0) / (1024 * 1024)
                image_sizes[img] = f"{size_mb:.2f} MB"

        gl.info("image_sizes", image_sizes)
        gl.info(
            "predict_latency_ms_container",
            {
                "p50": round(bench_data["predict"]["p50_ms"], 2),
                "p95": round(bench_data["predict"]["p95_ms"], 2),
                "p99": round(bench_data["predict"]["p99_ms"], 2),
            },
        )
        proc_img_py = subprocess.run(
            ["docker", "run", "--rm", "revpred-api:p9", "python", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        gl.info("python_version_image", proc_img_py.stdout.strip() or "unknown")

    finally:
        local_proc.terminate()
        try:
            local_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            local_proc.kill()
            local_proc.wait()


def main() -> None:
    gl.run_gate("9", build_checks)


if __name__ == "__main__":
    main()
