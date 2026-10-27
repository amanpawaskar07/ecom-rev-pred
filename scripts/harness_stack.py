"""Phase 4 linear ground-truth harness benchmark for temporal stacking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from revpred.config import get_project_config, get_thresholds
from revpred.cv import expanding_window_splits
from revpred.harness import make_linear_truth
from revpred.models import make_ridge, make_xgb
from revpred.stack import TemporalStacker

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


def run_linear_truth_harness(
    n_repeats: int = 5,
    min_train_months: int = 14,
    val_months: int = 1,
) -> dict[str, Any]:
    """Run linear-truth harness across 5 repeats for Ridge vs stack_xgb_meta."""
    thr = get_thresholds()
    cfg = get_project_config()
    base_seed = int(thr["global"]["seed"])
    meta_params = cfg.get("stack", {}).get("meta_params", {})
    min_inner_months = int(cfg.get("stack", {}).get("min_inner_train_months", 3))

    features = [f"x_{i}" for i in range(10)]
    repeat_results: list[dict[str, Any]] = []
    gains: list[float] = []
    mse_ridge_all: list[float] = []
    mse_stack_all: list[float] = []

    for r in range(n_repeats):
        seed = base_seed + r
        df = make_linear_truth(seed=seed, n=4000, d=10)
        splits = expanding_window_splits(
            df["month"],
            min_train_months=min_train_months,
            val_months=val_months,
        )

        cache: dict[tuple[int, tuple[str, ...]], Any] = {}
        r_mses: list[float] = []
        s_mses: list[float] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            tr = df.iloc[tr_idx]
            val = df.iloc[val_idx]

            # Ridge baseline on linear truth
            ridge = make_ridge().fit(tr[features], tr["revenue"])
            p_ridge = ridge.predict(val[features])
            mse_ridge = float(np.mean((val["revenue"] - p_ridge) ** 2))

            # Temporal stacker with XGB meta on linear truth (no target transform, no resampling)
            stacker = TemporalStacker(
                bases=["ridge", "rf"],
                meta=make_xgb(params=meta_params),
                min_inner_train_months=min_inner_months,
                features=features,
                target_transform=None,
                resampler=None,
                cache=cache,
            )
            stacker.fit(tr)
            p_stack = stacker.predict(val)
            mse_stack = float(np.mean((val["revenue"] - p_stack) ** 2))

            r_mses.append(mse_ridge)
            s_mses.append(mse_stack)

        mean_r = float(np.mean(r_mses))
        mean_s = float(np.mean(s_mses))
        rel_gain = float((mean_r - mean_s) / mean_r) if mean_r != 0 else 0.0

        gains.append(rel_gain)
        mse_ridge_all.append(mean_r)
        mse_stack_all.append(mean_s)

        repeat_results.append({
            "repeat": r,
            "seed": seed,
            "mse_ridge": mean_r,
            "mse_stack": mean_s,
            "stack_gain_rel": rel_gain,
        })
        print(f"Repeat {r + 1}/{n_repeats} (seed={seed}): Ridge MSE={mean_r:.4f}, Stack MSE={mean_s:.4f}, Gain={rel_gain:.4f}")

    overall_gain = float(np.mean(gains))
    report = {
        "n_repeats": n_repeats,
        "repeats": repeat_results,
        "mean_mse_ridge": float(np.mean(mse_ridge_all)),
        "mean_mse_stack": float(np.mean(mse_stack_all)),
        "linear_truth_stack_gain_rel": overall_gain,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "harness_stack.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out_file} with linear_truth_stack_gain_rel={overall_gain:.6f}")
    return report


def main() -> None:
    print("Running Phase 4 linear truth harness benchmarks...")
    run_linear_truth_harness(n_repeats=5)
    print("Linear truth harness benchmarks complete.")


if __name__ == "__main__":
    main()
