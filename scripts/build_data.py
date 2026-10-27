"""Reproducible build script: end-to-end data pipeline, panel, holdout seal, and baselines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revpred.baselines import run_all_baselines
from revpred.data import build_order_lines
from revpred.panel import evaluate_entity_ladder, split_and_save_panel


def build_data(
    out_dir: Path | str | None = None,
    raw_dir: Path | str | None = None,
) -> None:
    """Run data pipeline end-to-end and generate all processed artifacts and reports."""
    if out_dir is not None:
        target_root = Path(out_dir).resolve()
        processed_dir = target_root / "data" / "processed"
        holdout_dir = target_root / "data" / "holdout"
        reports_dir = target_root / "reports"
    else:
        processed_dir = REPO_ROOT / "data" / "processed"
        holdout_dir = REPO_ROOT / "data" / "holdout"
        reports_dir = REPO_ROOT / "reports"

    raw_path = Path(raw_dir).resolve() if raw_dir is not None else REPO_ROOT / "data" / "raw"

    print("Building data pipeline:")
    print(f"  raw_dir:       {raw_path}")
    print(f"  processed_dir: {processed_dir}")
    print(f"  holdout_dir:   {holdout_dir}")
    print(f"  reports_dir:   {reports_dir}")

    # 1. Join raw tables into order_lines.parquet and produce reports/data_profile.json
    print("Step 1/4: Building order_lines.parquet and data_profile.json...")
    order_lines, _profile_data = build_order_lines(
        raw_dir=raw_path,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
    )
    print(f"  order_lines rows: {len(order_lines)}")

    # 2. Evaluate entity ladder and write reports/entity_choice.json
    print("Step 2/4: Evaluating entity ladder and generating entity_choice.json...")
    chosen, _choice_data = evaluate_entity_ladder(
        order_lines=order_lines,
        reports_dir=reports_dir,
    )
    print(f"  chosen entity: {chosen}")

    # 3. Split and save panel_dev.parquet, panel_holdout.parquet, holdout_seal.json, panel_reconciliation.json
    print("Step 3/4: Splitting panel, sealing holdout, reconciling revenue...")
    panel_dev, panel_holdout, seal_data, _recon_summary = split_and_save_panel(
        order_lines=order_lines,
        entity=chosen,
        processed_dir=processed_dir,
        holdout_dir=holdout_dir,
        reports_dir=reports_dir,
    )
    print(f"  panel_dev rows: {len(panel_dev)}")
    print(f"  panel_holdout rows: {len(panel_holdout)}")
    print(f"  holdout sha256: {seal_data['sha256']}")

    # 4. Run baseline models on panel_dev, save OOFs and reports/baselines_cv.json
    print("Step 4/4: Running baselines CV and generating OOFs and baselines_cv.json...")
    run_all_baselines(
        panel_dev=panel_dev,
        reports_dir=reports_dir,
    )
    print("  baselines CV completed successfully.")
    print("Build complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild data/processed, data/holdout, and reports from raw CSVs."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Target root directory for output artifacts (default: repo root).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing raw data CSVs (default: data/raw).",
    )
    args = parser.parse_args()

    build_data(out_dir=args.out_dir, raw_dir=args.raw_dir)


if __name__ == "__main__":
    main()