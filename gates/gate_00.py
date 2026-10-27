"""Gate 00: Scaffold, frozen gate machinery, AGENTS.md, and test suite."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

import gates.lib as gl


def build_checks() -> None:
    # 1. Tests collected and failed
    proc_pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no"],
        cwd=gl.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    passed_m = re.search(r"(\d+)\s+passed", proc_pytest.stdout)
    failed_m = re.search(r"(\d+)\s+failed", proc_pytest.stdout)
    n_passed = int(passed_m.group(1)) if passed_m else 0
    n_failed = int(failed_m.group(1)) if failed_m else 0
    tests_collected = n_passed + n_failed
    tests_failed = n_failed

    gl.check("tests_collected", tests_collected, ">=", "phase0.min_tests")
    gl.check("tests_failed", tests_failed, "==", "global.zero")

    # 2. Independent metric differences vs sklearn
    rng_metric = np.random.default_rng(int(gl.thr("global.seed")))
    max_diff = 0.0
    for _ in range(5):
        size = rng_metric.integers(50, 200)
        y = rng_metric.uniform(0.1, 1000.0, size=size)
        p = np.clip(y + rng_metric.normal(0, 10.0, size=size), 0.0, None)
        diff_rmsle = abs(gl.rmsle(y, p) - float(np.sqrt(mean_squared_log_error(y, p))))
        diff_mae = abs(gl.mae(y, p) - float(mean_absolute_error(y, p)))
        max_diff = max(max_diff, diff_rmsle, diff_mae)

    gl.check("independent_metric_max_abs_diff", max_diff, "<=", "phase0.independent_metric_abs_tol")

    # 3. Bootstrap null Type-I error rate
    n_trials = int(gl.thr("phase0.bootstrap_null_trials"))
    rng_null = np.random.default_rng(int(gl.thr("global.seed")))
    rejections = 0
    for _ in range(n_trials):
        t_seed = int(rng_null.integers(1, 1000000))
        t_rng = np.random.default_rng(t_seed)
        rows = []
        for fold in range(4):
            for c in range(30):
                for _ in range(2):
                    y_val = t_rng.uniform(10.0, 100.0)
                    base_pred = y_val + t_rng.normal(0, 5.0)
                    rows.append({
                        "cluster": f"c_{c}",
                        "fold": fold,
                        "y": y_val,
                        "pred_a": base_pred + t_rng.normal(0, 1e-4),
                        "pred_b": base_pred + t_rng.normal(0, 1e-4),
                        "tail_thr": 80.0,
                    })
        df_null = pd.DataFrame(rows)
        _, ci_lo, ci_hi, _, _ = gl.paired_cluster_bootstrap(
            df_null, "rmsle", "pred_a", "pred_b", n=500, seed=t_seed
        )
        if ci_lo > 0 or ci_hi < 0:
            rejections += 1
    bootstrap_null_type1 = rejections / n_trials
    gl.check("bootstrap_null_type1", bootstrap_null_type1, "<=", "phase0.bootstrap_null_type1_max")

    # 4. Bootstrap known effect detected
    rng_eff = np.random.default_rng(int(gl.thr("global.seed")))
    rows_eff = []
    n_folds = 6
    for fold in range(n_folds):
        for c in range(50):
            for _ in range(4):
                base_val = rng_eff.uniform(50.0, 200.0)
                log_y = np.log1p(base_val)
                eps = rng_eff.normal(0, 0.2)
                log_pred_a = log_y + eps
                log_pred_b = log_y + np.sqrt(1.10) * eps
                rows_eff.append({
                    "cluster": f"c_{c}",
                    "fold": fold,
                    "y": base_val,
                    "pred_a": np.expm1(log_pred_a),
                    "pred_b": np.expm1(log_pred_b),
                    "tail_thr": 150.0,
                })
    df_eff = pd.DataFrame(rows_eff)
    mean_diff, ci_lo, _, _, folds_fav = gl.paired_cluster_bootstrap(
        df_eff, "rmsle", a_col="pred_a", b_col="pred_b", n=1000, seed=42
    )
    known_detected = 1 if (mean_diff > 0 and ci_lo > 0 and folds_fav == n_folds) else 0
    gl.check("bootstrap_known_effect_detected", known_detected, "==", "global.bool_true")

    # 5. Thresholds SHA matches
    proc_sha = subprocess.run(
        ["sha256sum", "-c", "gates/thresholds.sha256"],
        cwd=gl.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sha_matches = 1 if proc_sha.returncode == 0 else 0
    gl.check("thresholds_sha_matches", sha_matches, "==", "global.bool_true")

    # 6. Pip check OK
    proc_pip = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=gl.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    pip_ok = 1 if (proc_pip.returncode == 0 and "No broken requirements found." in proc_pip.stdout) else 0
    gl.check("pip_check_ok", pip_ok, "==", "global.bool_true")

    # 7. verify_frozen self-test OK (three temp-repo cases)
    selftest_ok = 1
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        subprocess.run(["git", "init"], cwd=tpath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tpath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tpath, check=True, capture_output=True)
        (tpath / "gates").mkdir()
        (tpath / "gates" / "thresholds.yaml").write_text("version: 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tpath, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tpath, check=True, capture_output=True)
        subprocess.run(["git", "tag", "phase0-accepted"], cwd=tpath, check=True, capture_output=True)

        # Case 1: Untouched -> pass
        p1 = subprocess.run(
            [sys.executable, str(gl.REPO_ROOT / "scripts" / "verify_frozen.py")],
            cwd=tpath,
            capture_output=True,
            check=False,
        )
        if p1.returncode != 0:
            selftest_ok = 0

        # Case 2: Modified -> fail
        (tpath / "gates" / "thresholds.yaml").write_text("version: 2\n", encoding="utf-8")
        p2 = subprocess.run(
            [sys.executable, str(gl.REPO_ROOT / "scripts" / "verify_frozen.py")],
            cwd=tpath,
            capture_output=True,
            check=False,
        )
        if p2.returncode != 1:
            selftest_ok = 0

        # Case 3: Revert and add new file -> pass
        (tpath / "gates" / "thresholds.yaml").write_text("version: 1\n", encoding="utf-8")
        (tpath / "gates" / "gate_01.py").write_text("# new gate\n", encoding="utf-8")
        p3 = subprocess.run(
            [sys.executable, str(gl.REPO_ROOT / "scripts" / "verify_frozen.py")],
            cwd=tpath,
            capture_output=True,
            check=False,
        )
        if p3.returncode != 0:
            selftest_ok = 0

    gl.check("verify_frozen_selftest_ok", selftest_ok, "==", "global.bool_true")

    # 8. Ruff errors == 0
    proc_ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=gl.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    ruff_errors = 0 if proc_ruff.returncode == 0 else 1
    gl.check("ruff_errors", ruff_errors, "==", "global.zero")


def main() -> None:
    gl.run_gate("0", build_checks)


if __name__ == "__main__":
    main()
