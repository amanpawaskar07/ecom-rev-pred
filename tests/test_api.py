"""Tests for FastAPI service, tamper detection, OpenAPI snapshot, and validation rules."""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from api.app import app
from revpred.artifacts import ArtifactTampered, load_bundle

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
SNAPSHOT_PATH = REPO_ROOT / "tests" / "snapshots" / "openapi.json"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_payload(client: TestClient) -> dict[str, object]:
    res = client.get("/meta")
    assert res.status_code == 200
    meta = res.json()
    return dict(meta["example_payloads"][0])


def test_invalid_payload_cases(client: TestClient, valid_payload: dict[str, object]) -> None:
    """Verify >= 8 distinct invalid payloads return 422."""
    cases: list[tuple[str, str, dict[str, object] | str]] = []

    # 1. Negative revenue
    p1 = dict(valid_payload)
    p1["lag_rev_1"] = -1.0
    cases.append(("predict", "negative revenue", p1))

    # 2. Unknown region
    p2 = dict(valid_payload)
    p2["region"] = "Atlantis"
    cases.append(("predict", "unknown region", p2))

    # 3. Unknown category
    p3 = dict(valid_payload)
    p3["category"] = "flying_cars"
    cases.append(("predict", "unknown category", p3))

    # 4. month_of_year 13
    p4 = dict(valid_payload)
    p4["month_of_year"] = 13
    cases.append(("predict", "month_of_year 13", p4))

    # 5. Missing required field
    p5 = dict(valid_payload)
    del p5["lag_rev_1"]
    cases.append(("predict", "missing field", p5))

    # 6. Extra field forbidden
    p6 = dict(valid_payload)
    p6["forbidden_extra_param"] = "bad"
    cases.append(("predict", "extra field", p6))

    # 7. Wrong type (string for a float)
    p7 = dict(valid_payload)
    p7["lag_rev_1"] = "not_a_float"
    cases.append(("predict", "wrong type string", p7))

    # 8. NaN
    raw_nan = json.dumps(valid_payload).replace(str(valid_payload["lag_rev_1"]), "NaN", 1)
    cases.append(("predict_raw", "NaN value", raw_nan))

    # 9. Value above 10x ceiling (e.g. 1e12)
    p9 = dict(valid_payload)
    p9["lag_rev_1"] = 1e12
    cases.append(("predict", "above 10x ceiling", p9))

    # 10. top_k = 0 on /explain
    p10 = dict(valid_payload)
    p10["top_k"] = 0
    cases.append(("explain", "top_k 0", p10))

    # 11. top_k = 16 on /explain
    p11 = dict(valid_payload)
    p11["top_k"] = 16
    cases.append(("explain", "top_k 16", p11))

    assert len(cases) >= 8, f"Expected >= 8 invalid cases, got {len(cases)}"

    for endpoint, label, payload in cases:
        if endpoint == "predict":
            resp = client.post("/predict", json=payload)
        elif endpoint == "explain":
            resp = client.post("/explain", json=payload)
        elif endpoint == "predict_raw":
            resp = client.post(
                "/predict",
                content=str(payload),
                headers={"content-type": "application/json"},
            )
        else:
            raise ValueError(f"Unknown endpoint test {endpoint}")

        assert resp.status_code == 422, f"Expected 422 for {label}, got {resp.status_code}: {resp.text}"


def test_outside_dev_range(client: TestClient, valid_payload: dict[str, object]) -> None:
    """A payload with value above dev max but below ceiling returns 200 and lists feature in outside_dev_range."""
    meta = client.get("/meta").json()
    dev_max = meta["feature_ranges"]["lag_rev_1"]["max"]
    ceiling = meta["feature_ranges"]["lag_rev_1"]["ceiling"]

    test_val = dev_max * 1.5
    assert dev_max < test_val < ceiling

    payload = dict(valid_payload)
    payload["lag_rev_1"] = test_val

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "lag_rev_1" in data["outside_dev_range"]
    assert data["predicted_revenue_brl"] >= 0.0


def test_openapi_snapshot() -> None:
    """Verify OpenAPI snapshot matches committed specification."""
    assert SNAPSHOT_PATH.exists(), f"Snapshot not found at {SNAPSHOT_PATH}"
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    actual = app.openapi()
    assert actual == expected, "OpenAPI schema has drifted from tests/snapshots/openapi.json"


def test_tamper_detection() -> None:
    """Startup and bundle loading raise ArtifactTampered on byte modifications."""
    # Case 1: Tampered final_model.joblib
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(MODELS_DIR, td, dirs_exist_ok=True)
        m_file = Path(td) / "final_model.joblib"
        raw = bytearray(m_file.read_bytes())
        raw[10] = (raw[10] + 1) % 256
        m_file.write_bytes(raw)

        with pytest.raises(ArtifactTampered):
            load_bundle(td)

    # Case 2: Tampered api_meta.json
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(MODELS_DIR, td, dirs_exist_ok=True)
        m_file = Path(td) / "api_meta.json"
        raw = bytearray(m_file.read_bytes())
        raw[10] = (raw[10] + 1) % 256
        m_file.write_bytes(raw)

        with pytest.raises(ArtifactTampered):
            load_bundle(td)

    # Case 3: Tampered explain_background.json
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(MODELS_DIR, td, dirs_exist_ok=True)
        m_file = Path(td) / "explain_background.json"
        raw = bytearray(m_file.read_bytes())
        raw[10] = (raw[10] + 1) % 256
        m_file.write_bytes(raw)

        with pytest.raises(ArtifactTampered):
            load_bundle(td)


def test_determinism(client: TestClient, valid_payload: dict[str, object]) -> None:
    """Same payload twice gives identical responses."""
    r1 = client.post("/predict", json=valid_payload)
    r2 = client.post("/predict", json=valid_payload)
    assert r1.status_code == 200
    assert r1.json() == r2.json()

    exp_payload = dict(valid_payload)
    exp_payload["top_k"] = 5
    e1 = client.post("/explain", json=exp_payload)
    e2 = client.post("/explain", json=exp_payload)
    assert e1.status_code == 200
    assert e1.json() == e2.json()


def test_static_api_forbidden_patterns() -> None:
    """Verify static absence of forbidden patterns in api/."""
    forbidden = [".fit(", "model_selection", "read_parquet", "data/"]
    hits: list[str] = []
    for py_file in (REPO_ROOT / "api").rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for pat in forbidden:
            if pat in content:
                hits.append(f"{py_file.name}: {pat}")
    assert len(hits) == 0, f"Found forbidden patterns in api/: {hits}"


def test_real_subprocess_http(valid_payload: dict[str, object]) -> None:
    """One real-subprocess HTTP test of uvicorn server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

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
        # Wait for server
        ready = False
        with httpx.Client(timeout=1.0) as client:
            for _ in range(50):
                try:
                    r = client.get(f"{url}/health")
                    if r.status_code == 200:
                        ready = True
                        break
                except (httpx.HTTPError, ConnectionError, OSError):
                    time.sleep(0.1)

        assert ready, "Real subprocess server failed to become ready"

        with httpx.Client(timeout=10.0) as client:
            r_health = client.get(f"{url}/health")
            assert r_health.status_code == 200
            assert r_health.json()["status"] == "ok"

            r_meta = client.get(f"{url}/meta")
            assert r_meta.status_code == 200
            assert "regions" in r_meta.json()

            r_pred = client.post(f"{url}/predict", json=valid_payload)
            assert r_pred.status_code == 200
            pred_data = r_pred.json()
            assert pred_data["predicted_revenue_brl"] >= 0.0

            exp_payload = dict(valid_payload)
            exp_payload["top_k"] = 5
            r_exp = client.post(f"{url}/explain", json=exp_payload)
            assert r_exp.status_code == 200
            exp_data = r_exp.json()
            assert len(exp_data["contributions"]) == 5
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
