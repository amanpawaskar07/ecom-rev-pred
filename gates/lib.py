"""Frozen gate machinery and statistical decision functions.
Independent of src/; uses numpy, pandas, yaml, and stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_PATH = REPO_ROOT / "gates" / "thresholds.yaml"
SHA_PATH = REPO_ROOT / "gates" / "thresholds.sha256"


class ThresholdTamper(Exception):
    """Raised when gates/thresholds.yaml does not match gates/thresholds.sha256."""


def thr(
    key: str,
    thresholds_path: str | Path | None = None,
    sha_path: str | Path | None = None,
) -> Any:
    """Perform dotted lookup in gates/thresholds.yaml after verifying sha256."""
    if not isinstance(key, str):
        raise TypeError(f"Threshold key must be a string, got {type(key).__name__}")

    t_path = Path(thresholds_path) if thresholds_path else THRESHOLDS_PATH
    s_path = Path(sha_path) if sha_path else SHA_PATH

    if not t_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {t_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"Sha256 file not found: {s_path}")

    # Read expected sha256
    expected_sha = s_path.read_text(encoding="utf-8").strip().split()[0].lower()
    actual_sha = hashlib.sha256(t_path.read_bytes()).hexdigest().lower()

    if actual_sha != expected_sha:
        raise ThresholdTamper(
            f"Checksum mismatch for {t_path}: expected {expected_sha}, got {actual_sha}"
        )

    content = yaml.safe_load(t_path.read_text(encoding="utf-8"))
    parts = key.split(".")
    curr = content
    for part in parts:
        if not isinstance(curr, dict) or part not in curr:
            raise KeyError(f"Threshold key '{key}' not found (missing '{part}')")
        curr = curr[part]

    return curr


_current_checks: list[dict[str, Any]] = []
_current_infos: list[dict[str, Any]] = []
_current_pendings: list[dict[str, Any]] = []


def check(name: str, value: Any, op: str, key: str, *args: Any, **kwargs: Any) -> bool:
    """Record a gate check. Literal thresholds are strictly prohibited."""
    if args or kwargs:
        raise TypeError(
            "check() does not accept literal thresholds or extra arguments. "
            "Signature is check(name, value, op, key)."
        )
    if not isinstance(key, str):
        raise TypeError(
            f"Threshold key must be a string key into thresholds.yaml, got {type(key).__name__}"
        )

    allowed_ops = {"<=", ">=", "<", ">", "=="}
    if op not in allowed_ops:
        raise ValueError(f"Unsupported op '{op}'. Must be one of {allowed_ops}")

    threshold = thr(key)

    if op == "<=":
        passed = bool(value <= threshold)
    elif op == ">=":
        passed = bool(value >= threshold)
    elif op == "<":
        passed = bool(value < threshold)
    elif op == ">":
        passed = bool(value > threshold)
    elif op == "==":
        passed = bool(value == threshold)
    else:
        passed = False

    record = {
        "name": str(name),
        "value": value,
        "op": op,
        "threshold": threshold,
        "key": key,
        "passed": passed,
    }
    _current_checks.append(record)
    return passed


def info(name: str, value: Any) -> None:
    """Record a non-gating informative output line."""
    _current_infos.append({"name": str(name), "value": value})


def pending(name: str, reason: str) -> None:
    """Record a human-pending check."""
    _current_pendings.append({"name": str(name), "reason": str(reason)})


def run_gate(n: int | str, build_checks: Callable[[], None]) -> None:
    """Run build_checks(), print gate results, write reports/gate_NN.json, exit accordingly."""
    _current_checks.clear()
    _current_infos.clear()
    _current_pendings.clear()

    gate_title = str(n)
    if isinstance(n, int):
        file_num = f"{n:02d}"
    else:
        file_num = f"{int(n):02d}" if str(n).isdigit() else str(n)

    # Check git status for committed code requirement
    proc = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            "gates",
            "tests",
            "config",
            "scripts",
            "api",
            "ui",
            "docker",
        ],
        cwd=REPO_ROOT,
        capture_output=True, check=False,
        text=True,
    )
    dirty_status = proc.stdout.strip()
    if dirty_status:
        check("git_status_clean", 0, "==", "global.bool_true")
    else:
        check("git_status_clean", 1, "==", "global.bool_true")

    build_checks()

    # Get thresholds sha and git sha (first 12 chars)
    thresholds_sha = hashlib.sha256(THRESHOLDS_PATH.read_bytes()).hexdigest()[:12]
    proc_git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, check=False, text=True
    )
    git_sha = proc_git.stdout.strip()[:12] if proc_git.returncode == 0 else "000000000000"

    has_fail = any(not c["passed"] for c in _current_checks)
    has_pending = len(_current_pendings) > 0

    if has_fail:
        result_status = "FAIL"
        exit_code = 1
    elif has_pending:
        result_status = "PASS-PENDING"
        exit_code = 3
    else:
        result_status = "PASS"
        exit_code = 0

    # Print output exactly according to spec
    print(f"=== GATE {gate_title} BEGIN ===")
    for c in _current_checks:
        status_str = "PASS" if c["passed"] else "FAIL"
        print(
            f"check {c['name']} | value={c['value']} | {c['op']} {c['threshold']} ({c['key']}) | {status_str}"
        )
    for inf in _current_infos:
        print(f"info {inf['name']} | value={inf['value']}")
    for p in _current_pendings:
        print(f"pending {p['name']} | reason={p['reason']} | PENDING-HUMAN")
    print(
        f"GATE {gate_title} RESULT: {result_status}  thresholds_sha={thresholds_sha} git_sha={git_sha}"
    )
    print(f"=== GATE {gate_title} END ===")

    # Write report
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"gate_{file_num}.json"
    report_data = {
        "gate": gate_title,
        "result": result_status,
        "thresholds_sha": thresholds_sha,
        "git_sha": git_sha,
        "checks": _current_checks,
        "infos": _current_infos,
        "pendings": _current_pendings,
    }
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    sys.exit(exit_code)


# ==============================================================================
# Metrics in plain numpy
# ==============================================================================


def rmsle(y: Any, p: Any) -> float:
    """Root Mean Squared Logarithmic Error (p clipped >= 0)."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.clip(np.asarray(p, dtype=float), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(p_arr) - np.log1p(y_arr)) ** 2)))


def mae(y: Any, p: Any) -> float:
    """Mean Absolute Error."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(p_arr - y_arr)))


def wape(y: Any, p: Any) -> float:
    """Weighted Absolute Percentage Error: sum|p - y| / sum y."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    denom = np.sum(y_arr)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(p_arr - y_arr)) / denom)


def tail_mae(y: Any, p: Any, thr_val: float) -> float:
    """MAE over rows with y >= thr_val."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    mask = y_arr >= thr_val
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(p_arr[mask] - y_arr[mask])))


def tail_bias(y: Any, p: Any, thr_val: float) -> float:
    """Mean bias (p - y) over rows with y >= thr_val."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    mask = y_arr >= thr_val
    if not np.any(mask):
        return 0.0
    return float(np.mean(p_arr[mask] - y_arr[mask]))


def tail_threshold(y_train: Any) -> float:
    """Quantile over positive train rows at metrics.tail_quantile."""
    y_arr = np.asarray(y_train, dtype=float)
    positive = y_arr[y_arr > 0]
    if len(positive) == 0:
        return 0.0
    q = float(thr("metrics.tail_quantile"))
    return float(np.quantile(positive, q))


# ==============================================================================
# Vectorized Paired Cluster Bootstrap and Decision Rules
# ==============================================================================


def paired_cluster_bootstrap(
    df: pd.DataFrame,
    metric: str | Callable[[Any, Any], float],
    a_col: str,
    b_col: str,
    n: int | None = None,
    seed: int | None = None,
) -> tuple[float, float, float, np.ndarray, int]:
    """Paired cluster bootstrap across folds.
    Returns: (mean_diff, ci_lo, ci_hi, per_fold_diff, folds_favoring_a)
    where diff = metric(b) - metric(a) (positive means a is better).
    """
    if n is None:
        n = int(thr("global.n_bootstrap"))
    if seed is None:
        seed = int(thr("global.seed"))

    metric_name = getattr(metric, "__name__", str(metric)).lower()

    clusters = np.sort(df["cluster"].unique())
    folds = np.sort(df["fold"].unique())
    n_clusters = len(clusters)
    n_folds = len(folds)

    cluster_idx_map = {c: i for i, c in enumerate(clusters)}
    fold_idx_map = {f: j for j, f in enumerate(folds)}

    c_idx = df["cluster"].map(cluster_idx_map).to_numpy(dtype=np.int64)
    f_idx = df["fold"].map(fold_idx_map).to_numpy(dtype=np.int64)
    flat_cf = c_idx * n_folds + f_idx
    min_length = n_clusters * n_folds

    y = df["y"].to_numpy(dtype=float)
    p_a = df[a_col].to_numpy(dtype=float)
    p_b = df[b_col].to_numpy(dtype=float)

    if metric_name == "rmsle":
        p_a_clip = np.clip(p_a, 0.0, None)
        p_b_clip = np.clip(p_b, 0.0, None)
        e_a = (np.log1p(p_a_clip) - np.log1p(y)) ** 2
        e_b = (np.log1p(p_b_clip) - np.log1p(y)) ** 2
        S_A = np.bincount(flat_cf, weights=e_a, minlength=min_length).reshape(n_clusters, n_folds)
        S_B = np.bincount(flat_cf, weights=e_b, minlength=min_length).reshape(n_clusters, n_folds)
        Counts = np.bincount(flat_cf, minlength=min_length).reshape(n_clusters, n_folds)
    elif metric_name == "mae":
        e_a = np.abs(p_a - y)
        e_b = np.abs(p_b - y)
        S_A = np.bincount(flat_cf, weights=e_a, minlength=min_length).reshape(n_clusters, n_folds)
        S_B = np.bincount(flat_cf, weights=e_b, minlength=min_length).reshape(n_clusters, n_folds)
        Counts = np.bincount(flat_cf, minlength=min_length).reshape(n_clusters, n_folds)
    elif metric_name == "wape":
        e_a = np.abs(p_a - y)
        e_b = np.abs(p_b - y)
        S_A = np.bincount(flat_cf, weights=e_a, minlength=min_length).reshape(n_clusters, n_folds)
        S_B = np.bincount(flat_cf, weights=e_b, minlength=min_length).reshape(n_clusters, n_folds)
        Counts = np.bincount(flat_cf, weights=y, minlength=min_length).reshape(n_clusters, n_folds)
    elif metric_name == "tail_mae":
        tail_thr = df["tail_thr"].to_numpy(dtype=float)
        tail_mask = y >= tail_thr
        e_a = np.where(tail_mask, np.abs(p_a - y), 0.0)
        e_b = np.where(tail_mask, np.abs(p_b - y), 0.0)
        S_A = np.bincount(flat_cf, weights=e_a, minlength=min_length).reshape(n_clusters, n_folds)
        S_B = np.bincount(flat_cf, weights=e_b, minlength=min_length).reshape(n_clusters, n_folds)
        Counts = np.bincount(flat_cf, weights=tail_mask.astype(float), minlength=min_length).reshape(
            n_clusters, n_folds
        )
    else:
        raise ValueError(f"Unsupported metric for cluster bootstrap: {metric_name}")

    # Point estimate on original sample (cluster weight = 1)
    tot_SA_orig = S_A.sum(axis=0)
    tot_SB_orig = S_B.sum(axis=0)
    tot_Counts_orig = Counts.sum(axis=0)

    if metric_name == "rmsle":
        m_a_f = np.sqrt(tot_SA_orig / np.maximum(tot_Counts_orig, 1e-12))
        m_b_f = np.sqrt(tot_SB_orig / np.maximum(tot_Counts_orig, 1e-12))
    else:
        m_a_f = tot_SA_orig / np.maximum(tot_Counts_orig, 1e-12)
        m_b_f = tot_SB_orig / np.maximum(tot_Counts_orig, 1e-12)

    per_fold_diff = m_b_f - m_a_f
    mean_diff = float(np.mean(per_fold_diff))
    folds_favoring_a = int(np.sum(per_fold_diff > 0))

    # Multinomial bootstrap weights
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_clusters, np.ones(n_clusters) / n_clusters, size=n)

    tot_SA_boot = weights @ S_A  # (n, n_folds)
    tot_SB_boot = weights @ S_B  # (n, n_folds)
    tot_Counts_boot = weights @ Counts  # (n, n_folds)

    if metric_name == "rmsle":
        m_a_boot = np.sqrt(tot_SA_boot / np.maximum(tot_Counts_boot, 1e-12))
        m_b_boot = np.sqrt(tot_SB_boot / np.maximum(tot_Counts_boot, 1e-12))
    else:
        m_a_boot = tot_SA_boot / np.maximum(tot_Counts_boot, 1e-12)
        m_b_boot = tot_SB_boot / np.maximum(tot_Counts_boot, 1e-12)

    diffs_boot = m_b_boot - m_a_boot  # (n, n_folds)
    boot_stats = np.mean(diffs_boot, axis=1)  # (n,)

    ci_alpha = float(thr("global.ci_alpha"))
    ci_lo = float(np.percentile(boot_stats, 100 * (ci_alpha / 2.0)))
    ci_hi = float(np.percentile(boot_stats, 100 * (1.0 - ci_alpha / 2.0)))

    return mean_diff, ci_lo, ci_hi, per_fold_diff, folds_favoring_a


def decide_pair(
    df: pd.DataFrame,
    advanced_col: str,
    rival_col: str,
    metric: str | Callable[[Any, Any], float],
) -> dict[str, Any]:
    """Decide between an advanced component and a rival.
    Advanced wins iff rival is MATERIALLY WORSE:
      relative gain >= min_effect_rel AND ci_lo > 0 AND folds_favoring_adv / n_folds >= folds_favoring_frac.
    """
    mean_diff, ci_lo, ci_hi, per_fold_diff, folds_favoring_adv = paired_cluster_bootstrap(
        df, metric, a_col=advanced_col, b_col=rival_col
    )
    n_folds = len(per_fold_diff)

    # Compute mean per-fold metric for both
    folds = np.sort(df["fold"].unique())
    metric_func = {
        "rmsle": rmsle,
        "mae": mae,
        "wape": wape,
        "tail_mae": lambda y, p: tail_mae(y, p, df["tail_thr"].iloc[0]),
    }.get(getattr(metric, "__name__", str(metric)).lower())

    if metric_func is None:
        raise ValueError(f"Unknown metric: {metric}")

    m_adv_list = []
    m_rival_list = []
    for f in folds:
        fold_df = df[df["fold"] == f]
        m_adv_list.append(metric_func(fold_df["y"], fold_df[advanced_col]))
        m_rival_list.append(metric_func(fold_df["y"], fold_df[rival_col]))

    m_adv = float(np.mean(m_adv_list))
    m_rival = float(np.mean(m_rival_list))

    relative_gain = (m_rival - m_adv) / m_rival if m_rival != 0 else 0.0

    min_effect_rel = float(thr("global.min_effect_rel"))
    folds_favoring_frac = float(thr("global.folds_favoring_frac"))

    advanced_wins = bool(
        (relative_gain >= min_effect_rel)
        and (ci_lo > 0)
        and ((folds_favoring_adv / n_folds) >= folds_favoring_frac)
    )
    winner = advanced_col if advanced_wins else rival_col

    return {
        "winner": winner,
        "advanced_wins": advanced_wins,
        "advanced_col": advanced_col,
        "rival_col": rival_col,
        "metric": getattr(metric, "__name__", str(metric)),
        "relative_gain": float(relative_gain),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "mean_diff": float(mean_diff),
        "folds_favoring_adv": int(folds_favoring_adv),
        "n_folds": int(n_folds),
        "folds_favoring_frac": float(folds_favoring_adv / n_folds),
        "m_rival": float(m_rival),
        "m_adv": float(m_adv),
        "per_fold_diff": per_fold_diff.tolist(),
    }


class DecisionResult:
    """Convenience container for decide_among return value."""

    def __init__(self, selected: str, evidence: dict[str, Any]):
        self.selected = selected
        self.evidence = evidence

    def __iter__(self):
        return iter((self.selected, self.evidence))

    def __getitem__(self, idx: int | str) -> Any:
        if idx == 0:
            return self.selected
        if idx == 1:
            return self.evidence
        if isinstance(idx, str) and idx in self.evidence:
            return self.evidence[idx]
        raise IndexError(f"Index out of range: {idx}")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.selected == other
        if isinstance(other, DecisionResult):
            return self.selected == other.selected and self.evidence == other.evidence
        if isinstance(other, tuple) and len(other) == 2:
            return (self.selected, self.evidence) == other
        return False

    def __repr__(self) -> str:
        return f"DecisionResult(selected={self.selected!r}, evidence={self.evidence!r})"


def decide_among(
    df: pd.DataFrame,
    candidates_in_cost_order: list[str],
    metric: str | Callable[[Any, Any], float],
) -> DecisionResult:
    """Pick cheapest candidate that is not materially worse than best."""
    metric_func = {
        "rmsle": rmsle,
        "mae": mae,
        "wape": wape,
        "tail_mae": lambda y, p: tail_mae(y, p, df["tail_thr"].iloc[0]),
    }.get(getattr(metric, "__name__", str(metric)).lower())

    if metric_func is None:
        raise ValueError(f"Unknown metric: {metric}")

    folds = np.sort(df["fold"].unique())
    mean_fold_metrics: dict[str, float] = {}

    for cand in candidates_in_cost_order:
        scores = []
        for f in folds:
            fold_df = df[df["fold"] == f]
            scores.append(metric_func(fold_df["y"], fold_df[cand]))
        mean_fold_metrics[cand] = float(np.mean(scores))

    best_candidate = min(candidates_in_cost_order, key=lambda c: mean_fold_metrics[c])

    evidence_dict: dict[str, Any] = {
        "best_candidate": best_candidate,
        "mean_fold_metrics": mean_fold_metrics,
        "pairwise_vs_best": {},
    }

    selected = None
    for cand in candidates_in_cost_order:
        if cand == best_candidate:
            evidence_dict["pairwise_vs_best"][cand] = {
                "is_best": True,
                "winner": cand,
                "advanced_wins": False,
            }
            if selected is None:
                selected = cand
        else:
            ev = decide_pair(df, advanced_col=best_candidate, rival_col=cand, metric=metric)
            evidence_dict["pairwise_vs_best"][cand] = ev
            # If best does NOT materially beat cand, then cand is NOT materially worse than best!
            if not ev["advanced_wins"] and selected is None:
                selected = cand

    if selected is None:
        selected = best_candidate

    evidence_dict["selected"] = selected
    return DecisionResult(selected, evidence_dict)
