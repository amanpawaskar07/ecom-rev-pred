"""Tests for Streamlit UI (Phase 8)."""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(REPO_ROOT / "ui" / "app.py")


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


@pytest.fixture(scope="session")
def running_api():
    existing_url = os.environ.get("API_URL")
    if existing_url:
        yield existing_url
        return

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
    try:
        wait_for_server(url)
        os.environ["API_URL"] = url
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        os.environ.pop("API_URL", None)


def test_scenario_1_first_render(running_api: str) -> None:
    os.environ["API_URL"] = running_api
    at = AppTest.from_file(APP_PATH).run()
    assert len(at.exception) == 0, f"Exceptions occurred: {at.exception}"
    assert len(at.error) == 0, f"Errors occurred: {at.error}"


def test_scenario_2_load_example_predict(running_api: str) -> None:
    os.environ["API_URL"] = running_api
    at = AppTest.from_file(APP_PATH).run()
    assert len(at.exception) == 0
    at.selectbox(key="sel_example").select(1).run()
    at.button(key="btn_load_example").click().run()
    assert len(at.exception) == 0
    at.button(key="btn_predict").click().run()
    assert len(at.exception) == 0

    all_text = " ".join(
        [m.value for m in at.metric]
        + [m.value for m in at.markdown]
        + [t.value for t in at.text]
    )
    match = re.search(r"BRL\s+([\d,]+\.?\d*)", all_text)
    assert match is not None, f"BRL text not found in: {all_text}"
    val = float(match.group(1).replace(",", ""))
    assert val >= 0.0, f"Revenue should be non-negative, got {val}"


def test_scenario_3_explain(running_api: str) -> None:
    os.environ["API_URL"] = running_api
    at = AppTest.from_file(APP_PATH).run()
    assert len(at.exception) == 0
    at.button(key="btn_explain").click().run()
    assert len(at.exception) == 0

    charts = at.get("vega_lite_chart")
    assert len(charts) > 0 or any(e.type == "vega_lite_chart" for e in at.main)

    all_text = " ".join([m.value for m in at.markdown] + [t.value for t in at.text])
    assert "base value" in all_text.lower()
    assert "prediction (log1p)" in all_text.lower()


def test_scenario_4_invalid_input(running_api: str) -> None:
    os.environ["API_URL"] = running_api
    at = AppTest.from_file(APP_PATH).run()
    assert len(at.exception) == 0
    at.number_input(key="f_month_of_year").set_value(13).run()
    at.button(key="btn_predict").click().run()
    assert len(at.exception) == 0
    assert len(at.error) == 1, f"Expected 1 error, got {len(at.error)}"
    err_msg = at.error[0].value
    assert len(err_msg.strip().splitlines()) == 1, f"Error message should be one line: {err_msg}"


def test_scenario_5_api_down() -> None:
    dead_port = find_free_port()
    old_env = os.environ.get("API_URL")
    try:
        os.environ["API_URL"] = f"http://127.0.0.1:{dead_port}"
        at = AppTest.from_file(APP_PATH).run()
        assert len(at.exception) == 0, f"Expected no exceptions, got: {at.exception}"
        assert len(at.error) >= 1, "Expected st.error shown when API is down"
        all_text = " ".join(
            [e.value for e in at.error]
            + [m.value for m in at.markdown]
            + [t.value for t in at.text]
        )
        assert "traceback" not in all_text.lower(), f"Traceback found in output: {all_text}"
    finally:
        if old_env is not None:
            os.environ["API_URL"] = old_env
        else:
            os.environ.pop("API_URL", None)


def test_scenario_6_value_above_dev_max(running_api: str) -> None:
    os.environ["API_URL"] = running_api
    resp = httpx.get(f"{running_api}/meta")
    assert resp.status_code == 200
    meta = resp.json()
    ranges = meta["feature_ranges"]
    d_max = ranges["lag_rev_1"]["max"]
    ceiling = ranges["lag_rev_1"]["ceiling"]
    test_val = d_max + 100.0
    assert test_val <= ceiling

    at = AppTest.from_file(APP_PATH).run()
    assert len(at.exception) == 0
    at.number_input(key="f_lag_rev_1").set_value(test_val).run()
    at.button(key="btn_predict").click().run()
    assert len(at.exception) == 0
    assert len(at.warning) >= 1, "Expected st.warning for outside_dev_range"
    warn_text = " ".join([w.value for w in at.warning])
    assert "lag_rev_1" in warn_text
    assert "linear model extrapolates without limit" in warn_text


def test_scenario_7_displayed_brl_equals_direct_api(running_api: str) -> None:
    os.environ["API_URL"] = running_api
    resp = httpx.get(f"{running_api}/meta")
    assert resp.status_code == 200
    meta = resp.json()
    examples = meta.get("examples") or meta.get("example_payloads", [])
    example = examples[0]

    # Direct API call
    dir_resp = httpx.post(f"{running_api}/predict", json=example)
    assert dir_resp.status_code == 200
    direct_brl = round(dir_resp.json()["predicted_revenue_brl"], 2)

    # UI interaction
    at = AppTest.from_file(APP_PATH).run()
    assert len(at.exception) == 0
    at.selectbox(key="sel_example").select(1).run()
    at.button(key="btn_load_example").click().run()
    at.button(key="btn_predict").click().run()
    assert len(at.exception) == 0

    all_text = " ".join(
        [m.value for m in at.metric]
        + [m.value for m in at.markdown]
        + [t.value for t in at.text]
    )
    match = re.search(r"BRL\s+([\d,]+\.?\d*)", all_text)
    assert match is not None, f"BRL text not found in: {all_text}"
    displayed_brl = float(match.group(1).replace(",", ""))
    assert displayed_brl == direct_brl


def test_forbidden_patterns_in_ui() -> None:
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
    hits: list[str] = []
    for f in ui_dir.rglob("*.py"):
        content = f.read_text(encoding="utf-8")
        for mod in forbidden_modules:
            if mod in content:
                hits.append(f"{f.name}: forbidden import {mod}")
        for pat in forbidden_patterns:
            if pat in content:
                hits.append(f"{f.name}: forbidden pattern {pat}")
    assert len(hits) == 0, f"Found forbidden patterns in ui/: {hits}"
