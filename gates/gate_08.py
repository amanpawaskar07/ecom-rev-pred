"""Gate 8: Streamlit UI with AppTest scenarios, parity, and safety checks."""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from streamlit.testing.v1 import AppTest

from gates import lib as gl

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(REPO_ROOT / "ui" / "app.py")


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
    port = find_free_port()
    sub_env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
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
        env=sub_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    apptest_exceptions_count = 0

    try:
        wait_for_server(url, timeout_s=15.0)
        os.environ["API_URL"] = url

        # =========================================================================
        # 1. Run AppTest test scenarios via pytest
        # =========================================================================
        pytest_env = dict(os.environ, API_URL=url, PYTHONPATH=str(REPO_ROOT))
        proc_pytest = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_ui.py", "-q", "--tb=no"],
            cwd=REPO_ROOT,
            env=pytest_env,
            capture_output=True,
            text=True,
            check=False,
        )
        passed_m = re.search(r"(\d+)\s+passed", proc_pytest.stdout)
        failed_m = re.search(r"(\d+)\s+failed", proc_pytest.stdout)
        n_passed = int(passed_m.group(1)) if passed_m else 0
        n_failed = int(failed_m.group(1)) if failed_m else 0
        if n_failed > 0 or proc_pytest.returncode != 0:
            apptest_exceptions_count += n_failed if n_failed > 0 else 1

        gl.check("apptest_scenarios_passed", n_passed, ">=", "phase8.apptest_scenarios_min")

        # =========================================================================
        # 2. Compare 20 payloads: 3 examples + 17 seeded jitters
        # =========================================================================
        with httpx.Client(timeout=15.0) as client:
            meta = client.get(f"{url}/meta").json()
            openapi = client.get(f"{url}/openapi.json").json()

            examples = meta.get("examples") or meta.get("example_payloads", [])
            payloads: list[dict[str, Any]] = list(examples[:3])

            regions = meta["regions"]
            categories = meta["categories"]
            feature_ranges = meta["feature_ranges"]
            props = openapi.get("components", {}).get("schemas", {}).get("PredictRequest", {}).get("properties", {})

            seed = int(gl.thr("global.seed"))
            rng = np.random.default_rng(seed)

            for _ in range(17):
                jitter_p: dict[str, Any] = {
                    "region": str(rng.choice(regions)),
                    "category": str(rng.choice(categories)),
                }
                for feat, limits in feature_ranges.items():
                    f_min = limits["min"]
                    f_max = limits["max"]
                    p_type = props.get(feat, {}).get("type", "number")
                    if p_type == "integer":
                        jitter_p[feat] = int(rng.integers(int(f_min), int(f_max) + 1))
                    else:
                        jitter_p[feat] = round(float(rng.uniform(f_min, f_max)), 2)
                payloads.append(jitter_p)

            max_abs_diff = 0.0
            for p in payloads:
                # Direct API prediction
                r_dir = client.post(f"{url}/predict", json=p)
                if r_dir.status_code != 200:
                    raise RuntimeError(f"Direct predict failed with status {r_dir.status_code}: {r_dir.text}")
                expected_brl = round(r_dir.json()["predicted_revenue_brl"], 2)

                # UI prediction via AppTest
                at = AppTest.from_file(APP_PATH).run()
                apptest_exceptions_count += len(at.exception)

                at.selectbox(key="f_region").select(p["region"])
                at.selectbox(key="f_category").select(p["category"])
                for feat in feature_ranges:
                    at.number_input(key=f"f_{feat}").set_value(p[feat])

                at.button(key="btn_predict").click().run()
                apptest_exceptions_count += len(at.exception)

                corpus = " ".join(
                    [m.value for m in at.metric]
                    + [m.value for m in at.markdown]
                    + [t.value for t in at.text]
                )
                match = re.search(r"BRL\s+([\d,]+\.?\d*)", corpus)
                if not match:
                    raise ValueError(f"Could not parse displayed BRL from UI output: {corpus}")
                displayed_brl = float(match.group(1).replace(",", ""))

                diff = abs(displayed_brl - expected_brl)
                max_abs_diff = max(max_abs_diff, diff)

            gl.check("ui_display_vs_api_max_abs", max_abs_diff, "<=", "phase8.ui_display_vs_api_max_abs")

        # =========================================================================
        # 3. AppTest exceptions check
        # =========================================================================
        gl.check("apptest_exceptions", apptest_exceptions_count, "==", "global.zero")

        # =========================================================================
        # 4. Forbidden imports and patterns in ui/
        # =========================================================================
        forbidden_modules = [
            "import revpred",
            "from revpred",
            "import sklearn",
            "from sklearn",
            "import xgboost",
            "from xgboost",
            "import shap",
            "from shap",
            "import joblib",
            "from joblib",
            "import subprocess",
            "from subprocess",
            "import pickle",
            "from pickle",
        ]
        forbidden_patterns = [
            "read_parquet",
            ".fit(",
            "models/",
            "data/",
        ]
        ui_dir = REPO_ROOT / "ui"
        forbidden_hits = 0
        for py_file in ui_dir.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            for mod in forbidden_modules:
                if mod in text:
                    forbidden_hits += 1
            for pat in forbidden_patterns:
                if pat in text:
                    forbidden_hits += 1

        gl.check("forbidden_imports_in_ui", forbidden_hits, "==", "global.zero")

        # =========================================================================
        # 5. API down traceback check
        # =========================================================================
        dead_port = find_free_port()
        old_url = os.environ.get("API_URL")
        try:
            os.environ["API_URL"] = f"http://127.0.0.1:{dead_port}"
            at_dead = AppTest.from_file(APP_PATH).run()
            corpus_dead = " ".join(
                [e.value for e in at_dead.error]
                + [m.value for m in at_dead.markdown]
                + [t.value for t in at_dead.text]
            )
            traceback_shown = 1 if "traceback" in corpus_dead.lower() else 0
        finally:
            if old_url:
                os.environ["API_URL"] = old_url
            else:
                os.environ.pop("API_URL", None)

        gl.check("api_down_traceback_shown", traceback_shown, "==", "global.zero")

        # =========================================================================
        # 6. Holdout ledger recovery touches check
        # =========================================================================
        ledger_path = REPO_ROOT / "reports" / "holdout_ledger.jsonl"
        lines = (
            [
                line.strip()
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if ledger_path.exists()
            else []
        )
        holdout_recovery_touches = len(lines) - 1
        gl.check("holdout_recovery_touches", holdout_recovery_touches, "<=", "phase5.holdout_recovery_touches_max")

        # =========================================================================
        # 7. Informative Outputs
        # =========================================================================
        gl.info("scenarios_run", n_passed + n_failed)
        gl.info("payloads_compared", len(payloads))
        gl.info("python_version", sys.version)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        os.environ.pop("API_URL", None)


def main() -> None:
    gl.run_gate("8", build_checks)


if __name__ == "__main__":
    main()
