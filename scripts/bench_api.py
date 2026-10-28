"""Benchmark script measuring latency of /predict and /explain endpoints."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from revpred.artifacts import load_bundle

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"


def find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(url: str, timeout_s: float = 10.0) -> None:
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


def run_benchmark(
    base_url: str,
    n: int = 500,
    warmup: int = 20,
) -> dict[str, dict[str, float]]:
    """Run sequential benchmark requests and return latency summary."""
    bundle = load_bundle(MODELS_DIR)
    predict_payload = dict(bundle.api_meta["example_payloads"][0])
    explain_payload = dict(predict_payload)
    explain_payload["top_k"] = 5

    url = base_url.rstrip("/")
    with httpx.Client(timeout=30.0) as client:
        # Warmup predict
        for _ in range(warmup):
            r = client.post(f"{url}/predict", json=predict_payload)
            if r.status_code != 200:
                raise RuntimeError(f"Warmup /predict failed: {r.status_code} {r.text}")

        # Timed predict
        latencies_predict: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = client.post(f"{url}/predict", json=predict_payload)
            t1 = time.perf_counter()
            if r.status_code != 200:
                raise RuntimeError(f"/predict failed: {r.status_code} {r.text}")
            latencies_predict.append((t1 - t0) * 1000.0)

        # Warmup explain
        for _ in range(warmup):
            r = client.post(f"{url}/explain", json=explain_payload)
            if r.status_code != 200:
                raise RuntimeError(f"Warmup /explain failed: {r.status_code} {r.text}")

        # Timed explain
        latencies_explain: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = client.post(f"{url}/explain", json=explain_payload)
            t1 = time.perf_counter()
            if r.status_code != 200:
                raise RuntimeError(f"/explain failed: {r.status_code} {r.text}")
            latencies_explain.append((t1 - t0) * 1000.0)

    pred_p50 = float(np.percentile(latencies_predict, 50))
    pred_p95 = float(np.percentile(latencies_predict, 95))
    pred_p99 = float(np.percentile(latencies_predict, 99))
    pred_mean = float(np.mean(latencies_predict))
    pred_min = float(np.min(latencies_predict))
    pred_max = float(np.max(latencies_predict))

    exp_p50 = float(np.percentile(latencies_explain, 50))
    exp_p95 = float(np.percentile(latencies_explain, 95))
    exp_p99 = float(np.percentile(latencies_explain, 99))
    exp_mean = float(np.mean(latencies_explain))
    exp_min = float(np.min(latencies_explain))
    exp_max = float(np.max(latencies_explain))

    print(f"=== LATENCY BENCHMARK (n={n}, warmup={warmup}) ===")
    print(f"/predict: p50={pred_p50:.2f}ms  p95={pred_p95:.2f}ms  p99={pred_p99:.2f}ms  mean={pred_mean:.2f}ms")
    print(f"/explain: p50={exp_p50:.2f}ms  p95={exp_p95:.2f}ms  p99={exp_p99:.2f}ms  mean={exp_mean:.2f}ms")

    results = {
        "n": n,
        "warmup": warmup,
        "predict": {
            "p50_ms": pred_p50,
            "p95_ms": pred_p95,
            "p99_ms": pred_p99,
            "mean_ms": pred_mean,
            "min_ms": pred_min,
            "max_ms": pred_max,
        },
        "explain": {
            "p50_ms": exp_p50,
            "p95_ms": exp_p95,
            "p99_ms": exp_p99,
            "mean_ms": exp_mean,
            "min_ms": exp_min,
            "max_ms": exp_max,
        },
    }

    out_file = REPORTS_DIR / "bench_api_local.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote benchmark results to {out_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FastAPI endpoints.")
    parser.add_argument("--url", type=str, default=None, help="Base URL of running service. If omitted, starts uvicorn.")
    parser.add_argument("--n", type=int, default=500, help="Number of benchmark requests.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup requests.")
    args = parser.parse_args()

    proc: subprocess.Popen | None = None
    url = args.url

    if url is None:
        port = find_free_port()
        cmd = [
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
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"http://127.0.0.1:{port}"

    try:
        wait_for_server(url, timeout_s=15.0)
        run_benchmark(url, n=args.n, warmup=args.warmup)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
