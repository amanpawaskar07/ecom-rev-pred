"""Data ingestion, validation, join pipeline, and window discovery."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from revpred.config import get_project_config, get_thresholds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def manifest(
    raw_dir: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute sha256 and row counts for raw CSVs; verify row counts against project.yaml."""
    cfg = get_project_config()
    raw_path = Path(raw_dir) if raw_dir else REPO_ROOT / cfg["raw_dir"]
    out_path = Path(manifest_path) if manifest_path else raw_path / "MANIFEST.json"

    expected = cfg["expected_rowcounts"]
    manifest_data: dict[str, Any] = {}

    for key, filename in cfg["files"].items():
        file_path = raw_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required raw data file: {file_path}")

        # Compute sha256 and line count
        hasher = hashlib.sha256()
        row_count = 0
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)

        # Count data rows using stdlib csv
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is not None:
                for _ in reader:
                    row_count += 1

        exp_count = expected.get(key)
        if exp_count is not None and row_count != exp_count:
            raise ValueError(
                f"Row count mismatch for {filename}: expected {exp_count}, got {row_count}"
            )

        manifest_data[key] = {
            "filename": filename,
            "sha256": hasher.hexdigest(),
            "row_count": row_count,
            "expected_row_count": exp_count,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def compute_window(
    monthly_orders: dict[str, int] | pd.Series,
    min_order_frac_of_median: float,
    lag_depth: int,
    holdout_months: int,
) -> dict[str, Any]:
    """Determine the analysis window using the longest contiguous run of qualifying months."""
    if isinstance(monthly_orders, dict):
        s_orders = pd.Series(monthly_orders).sort_index()
    else:
        s_orders = monthly_orders.sort_index()

    median_orders = float(s_orders.median())
    min_threshold = min_order_frac_of_median * median_orders

    # Qualifying months
    qualifying_mask = s_orders >= min_threshold
    all_months = list(s_orders.index)

    # Find longest contiguous run of qualifying months
    longest_run: list[str] = []
    current_run: list[str] = []

    for m in all_months:
        if qualifying_mask[m]:
            current_run.append(m)
            if len(current_run) > len(longest_run):
                longest_run = list(current_run)
        else:
            current_run = []

    if not longest_run:
        raise ValueError("No qualifying months found for window")

    dropped_months = [m for m in all_months if m not in longest_run]
    modeled_months = longest_run[lag_depth:]
    if len(modeled_months) <= holdout_months:
        raise ValueError(
            f"Modeled months ({len(modeled_months)}) must exceed holdout months ({holdout_months})"
        )

    dev_modeled_months = modeled_months[:-holdout_months]
    holdout_modeled_months = modeled_months[-holdout_months:]

    return {
        "median_monthly_orders": median_orders,
        "min_order_threshold": min_threshold,
        "window_months": longest_run,
        "window_start": longest_run[0],
        "window_end": longest_run[-1],
        "dropped_months": dropped_months,
        "modeled_months": modeled_months,
        "dev_modeled_months": dev_modeled_months,
        "holdout_modeled_months": holdout_modeled_months,
    }


def build_order_lines(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join raw tables into order_lines.parquet and produce reports/data_profile.json."""
    cfg = get_project_config()
    thr = get_thresholds()

    raw_path = Path(raw_dir) if raw_dir else REPO_ROOT / cfg["raw_dir"]
    proc_path = Path(processed_dir) if processed_dir else REPO_ROOT / "data" / "processed"
    rep_path = Path(reports_dir) if reports_dir else REPO_ROOT / "reports"

    # 1. Ensure manifest is written
    manifest_dict = manifest(raw_dir=raw_path)

    # 2. Read raw files
    items = pd.read_csv(raw_path / cfg["files"]["items"])
    orders = pd.read_csv(raw_path / cfg["files"]["orders"])
    customers = pd.read_csv(raw_path / cfg["files"]["customers"])
    products = pd.read_csv(raw_path / cfg["files"]["products"])
    trans = pd.read_csv(raw_path / cfg["files"]["translation"], encoding="utf-8-sig")

    # Record raw status counts
    status_counts = orders["order_status"].value_counts().to_dict()

    # Drop exclude_statuses
    exclude_statuses = set(cfg["exclude_statuses"])
    non_excl_orders = orders[~orders["order_status"].isin(exclude_statuses)].copy()

    # Total items price over non-excluded orders
    items_in_non_excl = items[items["order_id"].isin(non_excl_orders["order_id"])].copy()

    # Month from orders
    month_col = cfg["month_source"].split(".")[1]
    non_excl_orders["month"] = pd.to_datetime(non_excl_orders[month_col]).dt.strftime("%Y-%m")

    # Region mapping
    state_to_region: dict[str, str] = {}
    for region_name, states in cfg["region_map"].items():
        for st in states:
            state_to_region[st] = region_name

    # Merge customers
    orders_cust = non_excl_orders.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
    )
    unmapped_mask = orders_cust["customer_state"].apply(lambda s: s not in state_to_region)
    unmapped_states_count = int(unmapped_mask.sum())
    orders_cust["region"] = orders_cust["customer_state"].map(state_to_region)

    # Merge items with orders_cust
    merged = items_in_non_excl.merge(
        orders_cust[["order_id", "month", "region", "customer_id"]],
        on="order_id",
        how="left",
    )

    # Merge products
    merged = merged.merge(
        products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
    )

    # Category translation: English, else Portuguese, else 'unknown'
    trans_map = dict(zip(trans["product_category_name"], trans["product_category_name_english"]))

    def resolve_category(cat_port: Any) -> str:
        if pd.isna(cat_port) or not cat_port:
            return "unknown"
        return str(trans_map.get(cat_port, cat_port))

    merged["category"] = merged["product_category_name"].apply(resolve_category)

    # Identify unmatched rows: failure to join order, customer, or product
    unmatched_mask = (
        merged["customer_id"].isna()
        | merged["region"].isna()
        | merged["product_id"].isna()
    )
    unmatched_items = int(unmatched_mask.sum())
    unmatched_revenue = float(merged.loc[unmatched_mask, "price"].sum())

    valid_lines = merged[~unmatched_mask].copy()

    # Final columns: order_id, order_item_id, product_id, category, region, month, price
    output_cols = ["order_id", "order_item_id", "product_id", "category", "region", "month", "price"]
    order_lines = valid_lines[output_cols].sort_values(["month", "order_id", "order_item_id"]).reset_index(drop=True)

    # Window computation based on non-excluded orders
    monthly_order_counts = non_excl_orders.groupby("month")["order_id"].count().to_dict()
    monthly_revenues = order_lines.groupby("month")["price"].sum().to_dict()

    window_info = compute_window(
        monthly_orders=monthly_order_counts,
        min_order_frac_of_median=float(thr["phase1"]["min_month_order_frac_of_median"]),
        lag_depth=int(cfg["lag_depth"]),
        holdout_months=int(thr["phase1"]["holdout_months"]),
    )

    total_valid_revenue = float(order_lines["price"].sum())
    total_revenue_pool = total_valid_revenue + unmatched_revenue
    unmatched_revenue_frac = (
        unmatched_revenue / total_revenue_pool if total_revenue_pool > 0 else 0.0
    )

    profile_data: dict[str, Any] = {
        "raw_row_counts": {k: v["row_count"] for k, v in manifest_dict.items()},
        "status_counts": status_counts,
        "unmatched_stats": {
            "unmatched_items": unmatched_items,
            "unmatched_revenue": unmatched_revenue,
            "unmatched_revenue_frac": unmatched_revenue_frac,
            "unmapped_states": unmapped_states_count,
        },
        "monthly_order_counts": monthly_order_counts,
        "monthly_revenues": monthly_revenues,
        "window": window_info,
        "order_lines_rows": len(order_lines),
        "total_valid_revenue": total_valid_revenue,
    }

    # Save order_lines.parquet
    proc_path.mkdir(parents=True, exist_ok=True)
    order_lines.to_parquet(proc_path / "order_lines.parquet", index=False)

    # Save reports/data_profile.json
    rep_path.mkdir(parents=True, exist_ok=True)
    with open(rep_path / "data_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    return order_lines, profile_data


def load_order_lines(parquet_path: Path | str | None = None) -> pd.DataFrame:
    """Load order_lines.parquet."""
    path = Path(parquet_path) if parquet_path else REPO_ROOT / "data" / "processed" / "order_lines.parquet"
    if not path.exists():
        raise FileNotFoundError(f"order_lines.parquet not found: {path}")
    return pd.read_parquet(path)