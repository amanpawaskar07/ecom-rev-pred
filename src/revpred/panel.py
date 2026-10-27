"""Panel construction, causal entry, base lag features, entity ladder, and dev/holdout splitting."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from revpred.config import get_project_config, get_thresholds
from revpred.data import compute_window

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Export FEATURES constant (base + categorical)
FEATURES: tuple[str, ...] = tuple(
    list(get_project_config()["features"]["base"])
    + list(get_project_config()["features"]["categorical"])
)


def get_canonical_window(order_lines: pd.DataFrame | None = None) -> list[str]:
    """Retrieve canonical window from data_profile.json or compute from order_lines."""
    profile_path = REPO_ROOT / "reports" / "data_profile.json"
    if profile_path.exists():
        try:
            with open(profile_path, encoding="utf-8") as f:
                data = json.load(f)
            return list(data["window"]["window_months"])
        except (json.JSONDecodeError, KeyError, OSError):
            return compute_window(
                order_lines.groupby("month")["order_id"].nunique(),
                min_order_frac_of_median=float(get_thresholds()["phase1"]["min_month_order_frac_of_median"]),
                lag_depth=int(get_project_config()["lag_depth"]),
                holdout_months=int(get_thresholds()["phase1"]["holdout_months"]),
            )["window_months"]

    if order_lines is not None:
        cfg = get_project_config()
        thr = get_thresholds()
        window_info = compute_window(
            order_lines.groupby("month")["order_id"].nunique(),
            min_order_frac_of_median=float(thr["phase1"]["min_month_order_frac_of_median"]),
            lag_depth=int(cfg["lag_depth"]),
            holdout_months=int(thr["phase1"]["holdout_months"]),
        )
        return list(window_info["window_months"])

    raise ValueError("Cannot determine window without data_profile.json or order_lines")


def build_panel(
    order_lines: pd.DataFrame,
    entity: str,
    window: list[str] | None = None,
) -> pd.DataFrame:
    """Build entity x region monthly panel with causal entry and lag features."""
    cfg = get_project_config()
    thr = get_thresholds()

    if window is None:
        window = get_canonical_window(order_lines)

    lag_depth = int(cfg["lag_depth"])
    max_month_in_data = str(order_lines["month"].max()) if not order_lines.empty else window[-1]
    modeled_months = [m for m in window[lag_depth:] if m <= max_month_in_data]
    min_active = int(thr["phase1"]["min_active_months_before_entry"])

    # Filter to window months
    df_win = order_lines[order_lines["month"].isin(window)].copy()

    # Series monthly aggregation
    agg = (
        df_win.groupby([entity, "region", "month"])
        .agg(
            rev=("price", "sum"),
            lines=("order_item_id", "count"),
        )
        .reset_index()
    )

    # Entity to category mapping
    if entity == "category":
        prod_cat_map = {c: c for c in df_win["category"].unique()}
    else:
        prod_cat_map = df_win.groupby(entity)["category"].first().to_dict()

    # Full grid pivot for revenue and lines
    piv_r = (
        agg.pivot(index=[entity, "region"], columns="month", values="rev")
        .reindex(columns=window)
        .fillna(0.0)
    )
    piv_l = (
        agg.pivot(index=[entity, "region"], columns="month", values="lines")
        .reindex(columns=window)
        .fillna(0)
        .astype(int)
    )

    piv_active = (piv_r > 0).astype(int)
    piv_cum_active = piv_active.cumsum(axis=1)

    # First active month index per series
    first_active_idx: dict[tuple[Any, str], int] = {}
    arr_active = piv_active.to_numpy()
    series_keys = piv_r.index.tolist()
    for i, s in enumerate(series_keys):
        nz = np.nonzero(arr_active[i])[0]
        first_active_idx[s] = int(nz[0]) if len(nz) > 0 else 999

    rows: list[pd.DataFrame] = []
    for m in modeled_months:
        idx_in_win = window.index(m)

        # Causal entry rule: active count in [window_start, t-1] >= min_active
        active_before_col = piv_cum_active.iloc[:, idx_in_win - 1]
        qual_mask = active_before_col >= min_active
        qual_series = piv_r.index[qual_mask]

        if len(qual_series) == 0:
            continue

        month_int = int(m.split("-")[1])
        r_t = piv_r.loc[qual_series, m].to_numpy()
        l_t = piv_l.loc[qual_series, m].to_numpy()

        # Lag features
        r_1 = piv_r.loc[qual_series, window[idx_in_win - 1]].to_numpy()
        r_2 = piv_r.loc[qual_series, window[idx_in_win - 2]].to_numpy()
        r_3 = piv_r.loc[qual_series, window[idx_in_win - 3]].to_numpy()

        l_1 = piv_l.loc[qual_series, window[idx_in_win - 1]].to_numpy()
        l_2 = piv_l.loc[qual_series, window[idx_in_win - 2]].to_numpy()
        l_3 = piv_l.loc[qual_series, window[idx_in_win - 3]].to_numpy()

        roll_mean_rev_3 = (r_1 + r_2 + r_3) / 3.0
        sum_l_3 = l_1 + l_2 + l_3
        trailing_price_3 = np.where(
            sum_l_3 > 0, (r_1 + r_2 + r_3) / np.maximum(sum_l_3, 1), 0.0
        )

        # cum_rev_6 = sum R over t-6..t-1 (raw input, NOT a model feature)
        start_6 = max(0, idx_in_win - 6)
        r_6 = piv_r.loc[qual_series, window[start_6:idx_in_win]].sum(axis=1).to_numpy()
        act_before = active_before_col[qual_mask].to_numpy()

        ents = [s[0] for s in qual_series]
        regs = [s[1] for s in qual_series]
        cats = [prod_cat_map.get(e, "unknown") if entity != "category" else e for e in ents]
        s_ids = [f"{e}_{r}" for e, r in qual_series]
        r_ids = [f"{sid}_{m}" for sid in s_ids]
        since_first = [idx_in_win - first_active_idx[s] for s in qual_series]

        m_df = pd.DataFrame({
            "row_id": r_ids,
            "series_id": s_ids,
            "entity": ents,
            "category": cats,
            "region": regs,
            "month": m,
            "revenue": r_t,
            "aux_lines": l_t,
            "aux_active": (r_t > 0).astype(int),
            "cum_rev_6": r_6,
            "lag_rev_1": r_1,
            "lag_rev_2": r_2,
            "lag_rev_3": r_3,
            "lag_lines_1": l_1,
            "lag_lines_2": l_2,
            "lag_lines_3": l_3,
            "roll_mean_rev_3": roll_mean_rev_3,
            "trailing_price_3": trailing_price_3,
            "active_months_before": act_before,
            "months_since_first_sale": since_first,
            "month_of_year": month_int,
        })
        rows.append(m_df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "row_id",
                "series_id",
                "entity",
                "category",
                "region",
                "month",
                "revenue",
                "aux_lines",
                "aux_active",
                "cum_rev_6",
                "lag_rev_1",
                "lag_rev_2",
                "lag_rev_3",
                "lag_lines_1",
                "lag_lines_2",
                "lag_lines_3",
                "roll_mean_rev_3",
                "trailing_price_3",
                "active_months_before",
                "months_since_first_sale",
                "month_of_year",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def evaluate_entity_ladder(
    order_lines: pd.DataFrame,
    reports_dir: Path | str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Evaluate entity ladder against thresholds; pick first qualifying entity; write entity_choice.json."""
    cfg = get_project_config()
    thr = get_thresholds()
    rep_path = Path(reports_dir) if reports_dir else REPO_ROOT / "reports"

    window = get_canonical_window(order_lines)
    lag_depth = int(cfg["lag_depth"])
    modeled_months = window[lag_depth:]
    holdout_months = int(thr["phase1"]["holdout_months"])
    dev_modeled = modeled_months[:-holdout_months]
    cv_folds = len(dev_modeled) - int(cfg["cv"]["min_train_months"])

    req_series = int(thr["phase1"]["min_series"])
    req_rows = int(thr["phase1"]["min_panel_rows"])
    req_folds = int(thr["phase1"]["min_cv_folds"])

    candidates: dict[str, Any] = {}
    chosen: str | None = None

    for entity in cfg["entity_ladder"]:
        p = build_panel(order_lines, entity=entity, window=window)
        n_series = int(p["series_id"].nunique())
        n_rows = len(p)
        qualifies = (
            n_series >= req_series and n_rows >= req_rows and cv_folds >= req_folds
        )

        candidates[entity] = {
            "series": n_series,
            "panel_rows": n_rows,
            "modeled_months": len(modeled_months),
            "cv_folds": cv_folds,
            "qualifies": bool(qualifies),
        }

        if qualifies and chosen is None:
            chosen = entity

    if chosen is None:
        raise RuntimeError("No candidate in entity ladder qualifies! STATUS: STOP-HUMAN")

    choice_data = {
        "chosen": chosen,
        "candidates": candidates,
    }

    rep_path.mkdir(parents=True, exist_ok=True)
    with open(rep_path / "entity_choice.json", "w", encoding="utf-8") as f:
        json.dump(choice_data, f, indent=2)

    return chosen, choice_data


def split_and_save_panel(
    order_lines: pd.DataFrame,
    entity: str,
    processed_dir: Path | str | None = None,
    holdout_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Build dev and holdout panels with causal truncation, seal holdout, reconcile panel revenue."""
    cfg = get_project_config()
    thr = get_thresholds()

    proc_path = Path(processed_dir) if processed_dir else REPO_ROOT / "data" / "processed"
    hold_path = Path(holdout_dir) if holdout_dir else REPO_ROOT / "data" / "holdout"
    rep_path = Path(reports_dir) if reports_dir else REPO_ROOT / "reports"

    window = get_canonical_window(order_lines)
    lag_depth = int(cfg["lag_depth"])
    modeled_months = window[lag_depth:]
    holdout_m_count = int(thr["phase1"]["holdout_months"])
    dev_months = modeled_months[:-holdout_m_count]
    holdout_months = modeled_months[-holdout_m_count:]
    last_dev_month = dev_months[-1]

    # DEV panel: MUST be built from order_lines truncated to months <= last dev month!
    order_lines_dev = order_lines[order_lines["month"] <= last_dev_month].copy()
    panel_dev = build_panel(order_lines_dev, entity=entity, window=window)

    # HOLDOUT panel: extracted from full build
    panel_full = build_panel(order_lines, entity=entity, window=window)
    panel_holdout = panel_full[panel_full["month"].isin(holdout_months)].copy().reset_index(drop=True)

    # Save panel_dev.parquet
    proc_path.mkdir(parents=True, exist_ok=True)
    panel_dev.to_parquet(proc_path / "panel_dev.parquet", index=False)

    # Save panel_holdout.parquet
    hold_path.mkdir(parents=True, exist_ok=True)
    holdout_file = hold_path / "panel_holdout.parquet"
    panel_holdout.to_parquet(holdout_file, index=False)

    # Compute sha256 of holdout file
    h_sha = hashlib.sha256(holdout_file.read_bytes()).hexdigest()

    # Get git commit hash
    proc_git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit_sha = proc_git.stdout.strip() if proc_git.returncode == 0 else "unknown"

    seal_data = {
        "sha256": h_sha,
        "rows": len(panel_holdout),
        "months": holdout_months,
        "commit": commit_sha,
    }

    rep_path.mkdir(parents=True, exist_ok=True)
    with open(rep_path / "holdout_seal.json", "w", encoding="utf-8") as f:
        json.dump(seal_data, f, indent=2)

    # Panel reconciliation: for every modeled month: panel revenue + unentered revenue = order_lines revenue
    df_win = order_lines[order_lines["month"].isin(window)].copy()
    agg = (
        df_win.groupby([entity, "region", "month"])
        .agg(rev=("price", "sum"))
        .reset_index()
    )
    piv_r = (
        agg.pivot(index=[entity, "region"], columns="month", values="rev")
        .reindex(columns=window)
        .fillna(0.0)
    )
    piv_cum_active = (piv_r > 0).astype(int).cumsum(axis=1)

    min_active = int(thr["phase1"]["min_active_months_before_entry"])
    reconciliation: dict[str, Any] = {}
    max_abs_diff = 0.0

    for m in modeled_months:
        idx_in_win = window.index(m)
        active_before = piv_cum_active.iloc[:, idx_in_win - 1]
        entered_mask = active_before >= min_active
        unentered_mask = ~entered_mask

        panel_rev = float(piv_r.loc[entered_mask, m].sum())
        unentered_rev = float(piv_r.loc[unentered_mask, m].sum())
        ol_rev = float(order_lines[order_lines["month"] == m]["price"].sum())

        diff = abs((panel_rev + unentered_rev) - ol_rev)
        max_abs_diff = max(max_abs_diff, diff)

        reconciliation[m] = {
            "panel_revenue": panel_rev,
            "unentered_revenue": unentered_rev,
            "total_accounted": panel_rev + unentered_rev,
            "order_lines_revenue": ol_rev,
            "abs_diff": diff,
        }

    recon_summary = {
        "max_abs_diff": max_abs_diff,
        "by_month": reconciliation,
    }

    with open(rep_path / "panel_reconciliation.json", "w", encoding="utf-8") as f:
        json.dump(recon_summary, f, indent=2)

    return panel_dev, panel_holdout, seal_data, recon_summary