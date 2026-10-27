"""Phase 3: Planted imbalance harness benchmarks and leak mutant validation.

Benchmarks:
  1. Learnable harness (smoter vs none, 5 repeats):
     - tail bias reduction relative to none
     - tail MAE gain relative to none
     - bulk RMSLE worsening relative to none
  2. Unlearnable harness (smoter vs none, 5 repeats):
     - tail MAE gain relative to none (null control)
  3. Leak mutant:
     - oversample whole dataset before random split
     - compute inflation ratio: tail_mae_correct / tail_mae_mutant

Writes:
  reports/harness_imbalance.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from gates import lib as gl
from revpred.config import get_thresholds
from revpred.harness import make_planted_imbalance
from revpred.models import make_xgb
from revpred.resample import SmoterResampler, TrainGuard
from revpred.targets import Log1pTargetTransform
from tests.mutants.mutants_phase3 import oversample_before_split_mutant_a

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


def run_harness_benchmarks() -> dict[str, Any]:
    """Run learnable, unlearnable, and leak mutant harness evaluations."""
    thr = get_thresholds()
    base_seed = int(thr["global"]["seed"])
    n_repeats = 5
    val_size = 0.20

    target_transform = Log1pTargetTransform()

    # 1. Learnable Harness
    learnable_results: list[dict[str, Any]] = []
    l_bias_red_list: list[float] = []
    l_mae_gain_list: list[float] = []
    l_rmsle_wor_list: list[float] = []

    for r in range(n_repeats):
        s_tr = base_seed + r
        s_val = base_seed + 1000 + r

        X, y, _ = make_planted_imbalance(seed=s_tr, learnable=True, n=6000, d=12)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_size, random_state=s_val)
        thr_tr = gl.tail_threshold(y_tr)

        # Transformed target
        y_tr_t = target_transform.transform(y_tr)

        # Candidate 0: none
        m0 = make_xgb().fit(X_tr, y_tr_t)
        p0 = target_transform.inverse(m0.predict(X_val))

        # Candidate 1: weights (sample_weight 3.0 for rare rows, else 1.0)
        w = np.where(y_tr >= thr_tr, 3.0, 1.0)
        m1 = make_xgb().fit(X_tr, y_tr_t, sample_weight=w)
        p1 = target_transform.inverse(m1.predict(X_val))

        # Candidate 2: smoter
        guard = TrainGuard(np.arange(len(X_tr)))
        res = SmoterResampler(guard=guard).fit_resample(
            X=X_tr,
            y=y_tr_t,
            train_idx=np.arange(len(X_tr)),
            seed=s_tr,
            tail_thr=thr_tr,
            y_raw=y_tr,
        )
        msm = make_xgb().fit(res.X, res.y)
        psm = target_transform.inverse(msm.predict(X_val))

        tb0 = gl.tail_bias(y_val, p0, thr_tr)
        tb1 = gl.tail_bias(y_val, p1, thr_tr)
        tbsm = gl.tail_bias(y_val, psm, thr_tr)

        tm0 = gl.tail_mae(y_val, p0, thr_tr)
        tm1 = gl.tail_mae(y_val, p1, thr_tr)
        tmsm = gl.tail_mae(y_val, psm, thr_tr)

        r0 = gl.rmsle(y_val, p0)
        r1 = gl.rmsle(y_val, p1)
        rsm = gl.rmsle(y_val, psm)

        # Metrics for smoter vs none
        bias_red_rel = (abs(tb0) - abs(tbsm)) / abs(tb0) if abs(tb0) > 0 else 0.0
        mae_gain_rel = (tm0 - tmsm) / tm0 if tm0 > 0 else 0.0
        rmsle_wor_rel = (rsm - r0) / r0 if r0 > 0 else 0.0

        l_bias_red_list.append(bias_red_rel)
        l_mae_gain_list.append(mae_gain_rel)
        l_rmsle_wor_list.append(rmsle_wor_rel)

        learnable_results.append({
            "repeat": r,
            "train_seed": s_tr,
            "val_seed": s_val,
            "tail_threshold": thr_tr,
            "tail_bias": {"none": tb0, "weights": tb1, "smoter": tbsm},
            "tail_mae": {"none": tm0, "weights": tm1, "smoter": tmsm},
            "rmsle": {"none": r0, "weights": r1, "smoter": rsm},
            "smoter_vs_none": {
                "tail_bias_reduction_rel": bias_red_rel,
                "tail_mae_gain_rel": mae_gain_rel,
                "bulk_rmsle_worsening_rel": rmsle_wor_rel,
            },
        })

    # 2. Unlearnable Harness
    unlearnable_results: list[dict[str, Any]] = []
    u_mae_gain_list: list[float] = []

    for r in range(n_repeats):
        s_tr = base_seed + r
        s_val = base_seed + 1000 + r

        Xu, yu, _ = make_planted_imbalance(seed=s_tr, learnable=False, n=6000, d=12)
        Xu_tr, Xu_val, yu_tr, yu_val = train_test_split(Xu, yu, test_size=val_size, random_state=s_val)
        thru_tr = gl.tail_threshold(yu_tr)

        yu_tr_t = target_transform.transform(yu_tr)

        mu0 = make_xgb().fit(Xu_tr, yu_tr_t)
        pu0 = target_transform.inverse(mu0.predict(Xu_val))

        wu = np.where(yu_tr >= thru_tr, 3.0, 1.0)
        mu1 = make_xgb().fit(Xu_tr, yu_tr_t, sample_weight=wu)
        pu1 = target_transform.inverse(mu1.predict(Xu_val))

        guard_u = TrainGuard(np.arange(len(Xu_tr)))
        resu = SmoterResampler(guard=guard_u).fit_resample(
            X=Xu_tr,
            y=yu_tr_t,
            train_idx=np.arange(len(Xu_tr)),
            seed=s_tr,
            tail_thr=thru_tr,
            y_raw=yu_tr,
        )
        musm = make_xgb().fit(resu.X, resu.y)
        pusm = target_transform.inverse(musm.predict(Xu_val))

        tmu0 = gl.tail_mae(yu_val, pu0, thru_tr)
        tmu1 = gl.tail_mae(yu_val, pu1, thru_tr)
        tmusm = gl.tail_mae(yu_val, pusm, thru_tr)

        u_mae_gain = (tmu0 - tmusm) / tmu0 if tmu0 > 0 else 0.0
        u_mae_gain_list.append(u_mae_gain)

        unlearnable_results.append({
            "repeat": r,
            "train_seed": s_tr,
            "val_seed": s_val,
            "tail_threshold": thru_tr,
            "tail_mae": {"none": tmu0, "weights": tmu1, "smoter": tmusm},
            "smoter_vs_none": {
                "tail_mae_gain_rel": u_mae_gain,
            },
        })

    # 3. Leak Mutant Evaluation
    # Compare correct procedure vs oversample-before-split mutant on learnable harness
    X_h, y_h, _ = make_planted_imbalance(seed=base_seed, learnable=True, n=6000, d=12)
    # Correct procedure
    X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(X_h, y_h, test_size=val_size, random_state=base_seed)
    thr_c = gl.tail_threshold(y_tr_c)
    y_tr_c_t = target_transform.transform(y_tr_c)
    res_c = SmoterResampler().fit_resample(
        X_tr_c, y_tr_c_t, train_idx=np.arange(len(X_tr_c)), seed=base_seed, tail_thr=thr_c, y_raw=y_tr_c
    )
    m_c = make_xgb().fit(res_c.X, res_c.y)
    p_c = target_transform.inverse(m_c.predict(X_val_c))
    tail_mae_correct = float(gl.tail_mae(y_val_c, p_c, thr_c))

    # Mutant procedure: oversample before split
    X_tr_m, X_val_m, y_tr_m, y_val_m, _ = oversample_before_split_mutant_a(
        X_h, y_h, test_size=val_size, seed=base_seed
    )
    y_tr_m_t = target_transform.transform(y_tr_m)
    m_m = make_xgb().fit(X_tr_m, y_tr_m_t)
    p_m = target_transform.inverse(m_m.predict(X_val_m))
    tail_mae_mutant = float(gl.tail_mae(y_val_m, p_m, thr_c))

    leak_mutant_inflation_ratio = float(tail_mae_correct / tail_mae_mutant) if tail_mae_mutant > 0 else 1.0

    report = {
        "learnable": {
            "repeats": learnable_results,
            "mean_tail_bias_reduction_rel": float(np.mean(l_bias_red_list)),
            "mean_tail_mae_gain_rel": float(np.mean(l_mae_gain_list)),
            "mean_bulk_rmsle_worsening_rel": float(np.mean(l_rmsle_wor_list)),
        },
        "unlearnable": {
            "repeats": unlearnable_results,
            "mean_tail_mae_gain_rel": float(np.mean(u_mae_gain_list)),
        },
        "leak_mutant": {
            "tail_mae_correct": tail_mae_correct,
            "tail_mae_mutant": tail_mae_mutant,
            "leak_mutant_inflation_ratio": leak_mutant_inflation_ratio,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "harness_imbalance.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    print("Running planted imbalance harness benchmarks and leak mutant...")
    report = run_harness_benchmarks()
    print("Harness benchmarks complete.")
    print(f"Learnable mean tail bias reduction: {report['learnable']['mean_tail_bias_reduction_rel']:.4f}")
    print(f"Learnable mean tail MAE gain: {report['learnable']['mean_tail_mae_gain_rel']:.4f}")
    print(f"Learnable mean bulk RMSLE worsening: {report['learnable']['mean_bulk_rmsle_worsening_rel']:.4f}")
    print(f"Unlearnable mean tail MAE gain: {report['unlearnable']['mean_tail_mae_gain_rel']:.4f}")
    print(f"Leak mutant inflation ratio: {report['leak_mutant']['leak_mutant_inflation_ratio']:.4f}")
    print("Results saved to reports/harness_imbalance.json")


if __name__ == "__main__":
    main()
