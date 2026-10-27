"""Gate 01: Data, Panel, Sealed Holdout, Baselines."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import yaml

import gates.lib as gl
from revpred.panel import FEATURES, build_panel, get_canonical_window
from tests.mutants.mutants_phase1 import (
    build_panel_mutant_a,
    build_panel_mutant_b,
    split_and_save_panel_mutant_c,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_checks() -> None:
    with open(REPO_ROOT / "config" / "project.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # =========================================================================
    # 1. Source Reconciliation with stdlib csv module (no pandas)
    # =========================================================================

    # Verify raw row counts match expected
    raw_rowcounts_match = 1
    for key, filename in cfg["files"].items():
        fp = REPO_ROOT / cfg["raw_dir"] / filename
        with open(fp, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            count = sum(1 for _ in reader) if header is not None else 0
        if count != cfg["expected_rowcounts"][key]:
            raw_rowcounts_match = 0

    gl.check("raw_rowcounts_match_expected", raw_rowcounts_match, "==", "global.bool_true")

    # Region mapping
    state_to_region: dict[str, str] = {}
    for region_name, states in cfg["region_map"].items():
        for st in states:
            state_to_region[st] = region_name

    # Customers mapping and unmapped states check
    unmapped_states = 0
    cust_map: dict[str, str] = {}
    with open(REPO_ROOT / cfg["raw_dir"] / cfg["files"]["customers"], "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            st = r["customer_state"]
            if st not in state_to_region:
                unmapped_states += 1
            cust_map[r["customer_id"]] = state_to_region.get(st, "unknown")

    gl.check("unmapped_states", unmapped_states, "<=", "phase1.max_unmapped_states")

    # Orders filtering
    exclude_statuses = set(cfg["exclude_statuses"])
    non_excl_orders: dict[str, tuple[str, str]] = {}
    with open(REPO_ROOT / cfg["raw_dir"] / cfg["files"]["orders"], "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["order_status"] not in exclude_statuses:
                m = r[cfg["month_source"].split(".")[1]][:7]
                non_excl_orders[r["order_id"]] = (m, r["customer_id"])

    # Translations
    trans_map: dict[str, str] = {}
    with open(REPO_ROOT / cfg["raw_dir"] / cfg["files"]["translation"], "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            trans_map[r["product_category_name"]] = r["product_category_name_english"]

    # Products mapping
    prod_map: dict[str, str] = {}
    with open(REPO_ROOT / cfg["raw_dir"] / cfg["files"]["products"], "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            port = r["product_category_name"]
            cat = trans_map.get(port, port) if port else "unknown"
            prod_map[r["product_id"]] = cat

    # Items aggregation
    raw_non_excl_items_sum = 0.0
    csv_order_lines_sum = 0.0
    unmatched_rev = 0.0
    by_month: dict[str, float] = {}
    by_region: dict[str, float] = {}
    by_category: dict[str, float] = {}

    with open(REPO_ROOT / cfg["raw_dir"] / cfg["files"]["items"], "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            oid = r["order_id"]
            if oid in non_excl_orders:
                p = float(r["price"])
                raw_non_excl_items_sum += p
                m, cid = non_excl_orders[oid]
                reg = cust_map.get(cid)
                pid = r["product_id"]
                cat = prod_map.get(pid)
                if reg is None or cat is None:
                    unmatched_rev += p
                else:
                    csv_order_lines_sum += p
                    by_month[m] = by_month.get(m, 0.0) + p
                    by_region[reg] = by_region.get(reg, 0.0) + p
                    by_category[cat] = by_category.get(cat, 0.0) + p

    recon_total_abs_diff = abs((csv_order_lines_sum + unmatched_rev) - raw_non_excl_items_sum)
    gl.check("recon_total_abs_diff", recon_total_abs_diff, "<=", "phase1.recon_abs_tol_brl")

    total_pool = csv_order_lines_sum + unmatched_rev
    unmatched_revenue_frac = unmatched_rev / total_pool if total_pool > 0 else 0.0
    gl.check("unmatched_revenue_frac", unmatched_revenue_frac, "<=", "phase1.max_unmatched_revenue_frac")

    # Compare independent CSV reconciliation against saved order_lines.parquet
    ol_table = pq.read_table(REPO_ROOT / "data" / "processed" / "order_lines.parquet")
    pq_prices = ol_table.column("price").to_pylist()
    pq_months = ol_table.column("month").to_pylist()
    pq_regions = ol_table.column("region").to_pylist()
    pq_cats = ol_table.column("category").to_pylist()

    pq_by_month: dict[str, float] = {}
    pq_by_region: dict[str, float] = {}
    pq_by_cat: dict[str, float] = {}
    for p, m, reg, cat in zip(pq_prices, pq_months, pq_regions, pq_cats):
        pq_by_month[m] = pq_by_month.get(m, 0.0) + p
        pq_by_region[reg] = pq_by_region.get(reg, 0.0) + p
        pq_by_cat[cat] = pq_by_cat.get(cat, 0.0) + p

    recon_max_month_abs_diff = max(abs(by_month[m] - pq_by_month[m]) for m in by_month)
    recon_max_region_abs_diff = max(abs(by_region[r] - pq_by_region[r]) for r in by_region)
    recon_max_category_abs_diff = max(abs(by_category[c] - pq_by_cat[c]) for c in by_category)

    gl.check("recon_max_month_abs_diff", recon_max_month_abs_diff, "<=", "phase1.recon_abs_tol_brl")
    gl.check("recon_max_region_abs_diff", recon_max_region_abs_diff, "<=", "phase1.recon_abs_tol_brl")
    gl.check("recon_max_category_abs_diff", recon_max_category_abs_diff, "<=", "phase1.recon_abs_tol_brl")

    # Panel reconciliation from reports/panel_reconciliation.json
    with open(REPO_ROOT / "reports" / "panel_reconciliation.json", encoding="utf-8") as f:
        panel_recon_data = json.load(f)
    panel_recon_max_abs_diff = float(panel_recon_data["max_abs_diff"])
    gl.check("panel_reconciliation_max_abs_diff", panel_recon_max_abs_diff, "<=", "phase1.recon_abs_tol_brl")

    # =========================================================================
    # 2. Structure Checks
    # =========================================================================

    with open(REPO_ROOT / "reports" / "data_profile.json", encoding="utf-8") as f:
        profile_data = json.load(f)
    window_months_list = profile_data["window"]["window_months"]
    window_months_count = len(window_months_list)
    gl.check("window_months", window_months_count, ">=", "phase1.min_window_months")

    holdout_months_count = len(profile_data["window"]["holdout_modeled_months"])
    gl.check("holdout_months", holdout_months_count, "==", "phase1.holdout_months")

    with open(REPO_ROOT / "reports" / "entity_choice.json", encoding="utf-8") as f:
        entity_choice = json.load(f)
    chosen_entity = entity_choice["chosen"]
    chosen_stats = entity_choice["candidates"][chosen_entity]

    series_count = int(chosen_stats["series"])
    panel_rows_count = int(chosen_stats["panel_rows"])
    cv_folds_count = int(chosen_stats["cv_folds"])

    gl.check("series", series_count, ">=", "phase1.min_series")
    gl.check("panel_rows", panel_rows_count, ">=", "phase1.min_panel_rows")
    gl.check("cv_folds", cv_folds_count, ">=", "phase1.min_cv_folds")

    # =========================================================================
    # 3. Leakage Checks
    # =========================================================================

    # Truncation invariance max abs diff across 3 cut months
    # Re-verify on order_lines
    ol_df = pq.read_table(REPO_ROOT / "data" / "processed" / "order_lines.parquet").to_pandas()
    window = get_canonical_window(ol_df)
    p_full = build_panel(ol_df, entity=chosen_entity, window=window)

    truncation_max_abs_diff = 0.0
    num_cols = [
        "revenue", "aux_lines", "aux_active", "cum_rev_6",
        "lag_rev_1", "lag_rev_2", "lag_rev_3",
        "lag_lines_1", "lag_lines_2", "lag_lines_3",
        "roll_mean_rev_3", "trailing_price_3",
        "active_months_before", "months_since_first_sale", "month_of_year",
    ]
    for cut_m in ["2017-07", "2017-11", "2018-04"]:
        ol_a = ol_df[ol_df["month"] < cut_m].copy()
        pa = build_panel(ol_a, entity=chosen_entity, window=window)
        sub_full = p_full[p_full["month"] < cut_m].sort_values("row_id").reset_index(drop=True)
        sub_a = pa[pa["month"] < cut_m].sort_values("row_id").reset_index(drop=True)
        d_a = float((sub_full[num_cols] - sub_a[num_cols]).abs().max().max())

        ol_b = ol_df.copy()
        ol_b.loc[ol_b["month"] >= cut_m, "price"] *= 100.0
        pb = build_panel(ol_b, entity=chosen_entity, window=window)
        sub_b = pb[pb["month"] < cut_m].sort_values("row_id").reset_index(drop=True)
        d_b = float((sub_full[num_cols] - sub_b[num_cols]).abs().max().max())

        truncation_max_abs_diff = max(truncation_max_abs_diff, d_a, d_b)

    gl.check("truncation_max_abs_diff", truncation_max_abs_diff, "<=", "phase1.truncation_max_abs_diff")

    # Mutants detection fraction
    mutants_caught = 0

    # Mutant A: lag leaks current revenue
    ma_panel = build_panel_mutant_a(ol_df, entity=chosen_entity, window=window)
    if (ma_panel["lag_rev_1"] == ma_panel["revenue"]).all():
        mutants_caught += 1

    # Mutant B: causal entry violated
    cut_test = "2017-07"
    ol_cut = ol_df[ol_df["month"] < cut_test].copy()
    mb_full = build_panel_mutant_b(ol_df, entity=chosen_entity, window=window)
    mb_cut = build_panel_mutant_b(ol_cut, entity=chosen_entity, window=window)
    s_full = set(mb_full[mb_full["month"] == "2017-06"]["series_id"])
    s_cut = set(mb_cut[mb_cut["month"] == "2017-06"]["series_id"])
    if s_full != s_cut:
        mutants_caught += 1

    # Mutant C: dev panel built from full data then sliced
    calls_mutant = []
    with patch("tests.mutants.mutants_phase1.build_panel", side_effect=build_panel) as mock_bp_c:
        split_and_save_panel_mutant_c(
            ol_df,
            entity=chosen_entity,
            processed_dir=REPO_ROOT / "reports" / "scratch" / "proc_m",
            holdout_dir=REPO_ROOT / "reports" / "scratch" / "hold_m",
            reports_dir=REPO_ROOT / "reports" / "scratch" / "rep_m",
        )
        for call in mock_bp_c.call_args_list:
            calls_mutant.append(str(call.args[0]["month"].max()))
    if "2018-05" not in calls_mutant:
        mutants_caught += 1

    mutants_caught_frac = mutants_caught / 3.0
    gl.check("mutants_caught_frac", mutants_caught_frac, ">=", "phase1.mutant_detection_min_frac")

    # Features constant: aux_* and target not in FEATURES
    bad_features = [f for f in FEATURES if f == "revenue" or f.startswith("aux_") or f == "cum_rev_6"]
    gl.check("features_contain_aux_or_target", len(bad_features), "==", "global.zero")

    # =========================================================================
    # 4. Holdout Checks
    # =========================================================================

    holdout_file = REPO_ROOT / "data" / "holdout" / "panel_holdout.parquet"
    with open(REPO_ROOT / "reports" / "holdout_seal.json", encoding="utf-8") as f:
        seal_data = json.load(f)

    h_sha = hashlib.sha256(holdout_file.read_bytes()).hexdigest()
    holdout_seal_ok = 1 if h_sha == seal_data["sha256"] else 0
    gl.check("holdout_seal_ok", holdout_seal_ok, "==", "global.bool_true")

    ledger_path = REPO_ROOT / "reports" / "holdout_ledger.jsonl"
    ledger_lines = (
        len(ledger_path.read_text(encoding="utf-8").strip().splitlines())
        if ledger_path.exists() and ledger_path.read_text(encoding="utf-8").strip()
        else 0
    )
    gl.check("holdout_ledger_lines", ledger_lines, "==", "global.zero")

    dev_table = pq.read_table(REPO_ROOT / "data" / "processed" / "panel_dev.parquet")
    dev_months_set = set(dev_table.column("month").to_pylist())
    holdout_months_set = set(seal_data["months"])
    holdout_in_dev_count = len(dev_months_set.intersection(holdout_months_set))
    gl.check("holdout_months_present_in_dev", holdout_in_dev_count, "==", "global.zero")

    # =========================================================================
    # 5. Baselines Checks
    # =========================================================================

    with open(REPO_ROOT / "reports" / "baselines_cv.json", encoding="utf-8") as f:
        baselines_data = json.load(f)

    ridge_rmsle = baselines_data["baseline_ridge"]["pooled"]["rmsle"]
    mean_rmsle = baselines_data["baseline_mean"]["pooled"]["rmsle"]
    ridge_beats_mean = 1 if ridge_rmsle < mean_rmsle else 0
    gl.check("ridge_beats_mean_dev_cv", ridge_beats_mean, "==", "global.bool_true")

    # Recompute metrics from saved OOF files
    max_recompute_rel_diff = 0.0
    for b in ["baseline_mean", "baseline_persistence", "baseline_ridge"]:
        oof_path = REPO_ROOT / "reports" / "oof" / f"{b}.parquet"
        oof_t = pq.read_table(oof_path)
        y_arr = oof_t.column("y").to_numpy()
        p_arr = oof_t.column("pred").to_numpy()

        r_rmsle = gl.rmsle(y_arr, p_arr)
        r_mae = gl.mae(y_arr, p_arr)
        r_wape = gl.wape(y_arr, p_arr)

        stored = baselines_data[b]["pooled"]
        diff_rmsle = abs(r_rmsle - stored["rmsle"]) / stored["rmsle"]
        diff_mae = abs(r_mae - stored["mae"]) / stored["mae"]
        diff_wape = abs(r_wape - stored["wape"]) / stored["wape"]

        max_recompute_rel_diff = max(max_recompute_rel_diff, diff_rmsle, diff_mae, diff_wape)

    gl.check("baseline_recompute_max_rel_diff", max_recompute_rel_diff, "<=", "global.repro_rel_tol")

    # =========================================================================
    # 6. Informative Output
    # =========================================================================

    gl.info("entity_chosen", chosen_entity)
    gl.info("series", series_count)
    gl.info("rows", panel_rows_count)
    gl.info("window", f"{window_months_list[0]}..{window_months_list[-1]}")

    for b in ["baseline_mean", "baseline_persistence", "baseline_ridge"]:
        m = baselines_data[b]["pooled"]
        prefix = b.replace("baseline_", "")
        gl.info(f"{prefix}_rmsle", round(m["rmsle"], 6))
        gl.info(f"{prefix}_mae", round(m["mae"], 6))
        gl.info(f"{prefix}_wape", round(m["wape"], 6))
        gl.info(f"{prefix}_tail_mae", round(m["tail_mae"], 6))


def main() -> None:
    gl.run_gate("1", build_checks)


if __name__ == "__main__":
    main()