"""Tests for gates.lib metrics, thresholds, bootstrap, and decision rules."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

import gates.lib as gl


def test_metrics_vs_sklearn():
    abs_tol = float(gl.thr("phase0.independent_metric_abs_tol"))
    rng = np.random.default_rng(gl.thr("global.seed"))

    for _ in range(5):
        size = rng.integers(50, 200)
        y = rng.uniform(0.1, 1000.0, size=size)
        p = np.clip(y + rng.normal(0, 10.0, size=size), 0.0, None)

        our_rmsle = gl.rmsle(y, p)
        sklearn_rmsle = float(np.sqrt(mean_squared_log_error(y, p)))
        assert abs(our_rmsle - sklearn_rmsle) <= abs_tol

        our_mae = gl.mae(y, p)
        sklearn_mae = float(mean_absolute_error(y, p))
        assert abs(our_mae - sklearn_mae) <= abs_tol


def test_wape_and_tail_metrics():
    y = np.array([10.0, 20.0, 30.0, 100.0])
    p = np.array([12.0, 18.0, 35.0, 90.0])

    # WAPE = sum(|p - y|) / sum(y) = (2 + 2 + 5 + 10) / 160 = 19 / 160
    assert abs(gl.wape(y, p) - (19.0 / 160.0)) < 1e-12

    # tail_mae and tail_bias with thr=50.0 (only 100.0 is >= 50)
    assert abs(gl.tail_mae(y, p, 50.0) - 10.0) < 1e-12
    assert abs(gl.tail_bias(y, p, 50.0) - (-10.0)) < 1e-12

    # tail_threshold on y_train
    q_val = gl.tail_threshold(y)
    assert q_val > 0


def test_check_raises_type_error_on_literal():
    # Calling check with a numeric threshold directly must raise TypeError
    with pytest.raises(TypeError):
        gl.check("bad_check", 5.0, "<=", 10.0)  # Literal float instead of key

    with pytest.raises(TypeError):
        gl.check("bad_check", 5.0, "<=", "global.seed", 10.0)  # Extra arg

    with pytest.raises(TypeError):
        gl.check("bad_check", 5.0, "<=", "global.seed", threshold=10.0)  # Kwarg


def test_thr_raises_on_unknown_key():
    with pytest.raises(KeyError):
        gl.thr("nonexistent.key.xyz")


def test_threshold_tamper_raised(tmp_path: Path):
    t_file = tmp_path / "thresholds.yaml"
    s_file = tmp_path / "thresholds.sha256"

    # Copy real thresholds
    t_file.write_text(gl.THRESHOLDS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    s_file.write_text(gl.SHA_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    # Untampered works
    val = gl.thr("global.seed", thresholds_path=t_file, sha_path=s_file)
    assert val == 42

    # Tamper with yaml
    t_file.write_text(t_file.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(gl.ThresholdTamper):
        gl.thr("global.seed", thresholds_path=t_file, sha_path=s_file)


def test_run_gate_subprocess_failure():
    code = """
import gates.lib as gl
def my_checks():
    gl.check('failing_check', 0, '==', 'global.bool_true')
gl.run_gate(99, my_checks)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=gl.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "=== GATE 99 BEGIN ===" in proc.stdout
    assert "FAIL" in proc.stdout
    assert "GATE 99 RESULT: FAIL" in proc.stdout
    assert "=== GATE 99 END ===" in proc.stdout


def test_bootstrap_known_effect():
    n_folds = 6
    n_clusters = 50
    rows = []
    rng = np.random.default_rng(gl.thr("global.seed"))

    for fold in range(n_folds):
        for c in range(n_clusters):
            for _ in range(4):
                base_val = rng.uniform(50.0, 200.0)
                y_val = base_val
                # a is close to y; b has 10% larger log error
                log_y = np.log1p(y_val)
                eps = rng.normal(0, 0.2)
                # MSE-log for a will be roughly eps^2, for b will be 1.1 * eps^2
                log_pred_a = log_y + eps
                log_pred_b = log_y + np.sqrt(1.10) * eps
                pred_a = np.expm1(log_pred_a)
                pred_b = np.expm1(log_pred_b)
                rows.append({
                    "cluster": f"c_{c}",
                    "fold": fold,
                    "y": y_val,
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "tail_thr": 150.0,
                })

    df = pd.DataFrame(rows)
    mean_diff, ci_lo, ci_hi, per_fold_diff, folds_favoring_a = gl.paired_cluster_bootstrap(
        df, "rmsle", a_col="pred_a", b_col="pred_b", n=1000, seed=42
    )

    assert mean_diff > 0
    assert ci_lo > 0  # CI excludes 0
    assert ci_hi > ci_lo
    assert len(per_fold_diff) == n_folds
    assert folds_favoring_a == n_folds


def test_bootstrap_null():
    n_trials = int(gl.thr("phase0.bootstrap_null_trials"))
    max_type1 = float(gl.thr("phase0.bootstrap_null_type1_max"))

    n_folds = 4
    n_clusters = 30
    rng = np.random.default_rng(gl.thr("global.seed"))

    rejections = 0
    for trial in range(n_trials):
        rows = []
        trial_seed = int(rng.integers(1, 1000000))
        trial_rng = np.random.default_rng(trial_seed)

        for fold in range(n_folds):
            for c in range(n_clusters):
                for _ in range(2):
                    y_val = trial_rng.uniform(10.0, 100.0)
                    base_pred = y_val + trial_rng.normal(0, 5.0)
                    # a and b are identical predictor plus independent tiny noise
                    pred_a = base_pred + trial_rng.normal(0, 1e-4)
                    pred_b = base_pred + trial_rng.normal(0, 1e-4)
                    rows.append({
                        "cluster": f"c_{c}",
                        "fold": fold,
                        "y": y_val,
                        "pred_a": pred_a,
                        "pred_b": pred_b,
                        "tail_thr": 80.0,
                    })

        df = pd.DataFrame(rows)
        _, ci_lo, ci_hi, _, _ = gl.paired_cluster_bootstrap(
            df, "rmsle", a_col="pred_a", b_col="pred_b", n=500, seed=trial_seed
        )

        # Null hypothesis is diff == 0. Rejection occurs if 0 is outside [ci_lo, ci_hi]
        if ci_lo > 0 or ci_hi < 0:
            rejections += 1

    type1_rate = rejections / n_trials
    assert type1_rate <= max_type1


def test_decide_among_equal_picks_cheapest():
    n_folds = 4
    n_clusters = 20
    rows = []
    rng = np.random.default_rng(42)

    for fold in range(n_folds):
        for c in range(n_clusters):
            y_val = rng.uniform(10.0, 100.0)
            pred = y_val + rng.normal(0, 2.0)
            rows.append({
                "cluster": f"c_{c}",
                "fold": fold,
                "y": y_val,
                "cheap": pred,
                "medium": pred,
                "costly": pred,
                "tail_thr": 80.0,
            })
    df = pd.DataFrame(rows)
    selected, evidence = gl.decide_among(df, ["cheap", "medium", "costly"], "rmsle")
    assert selected == "cheap"
    assert "best_candidate" in evidence


def test_decide_among_picks_costly_when_materially_better():
    n_folds = 6
    n_clusters = 40
    rows = []
    rng = np.random.default_rng(42)

    for fold in range(n_folds):
        for c in range(n_clusters):
            for _ in range(3):
                y_val = rng.uniform(10.0, 100.0)
                # costly is close to y; cheap is much worse
                pred_costly = y_val + rng.normal(0, 1.0)
                pred_cheap = y_val + rng.normal(0, 20.0)
                pred_med = y_val + rng.normal(0, 18.0)
                rows.append({
                    "cluster": f"c_{c}",
                    "fold": fold,
                    "y": y_val,
                    "cheap": pred_cheap,
                    "medium": pred_med,
                    "costly": pred_costly,
                    "tail_thr": 80.0,
                })
    df = pd.DataFrame(rows)
    selected, evidence = gl.decide_among(df, ["cheap", "medium", "costly"], "rmsle")
    assert selected == "costly"
    assert "best_candidate" in evidence
