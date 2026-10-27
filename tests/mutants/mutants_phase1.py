"""Phase 1 mutants for leakage testing."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from revpred.config import get_project_config, get_thresholds
from revpred.panel import build_panel, get_canonical_window

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def build_panel_mutant_a(
    order_lines: pd.DataFrame,
    entity: str = "product_id",
    window: list[str] | None = None,
) -> pd.DataFrame:
    """Mutant (a): lag features leak the CURRENT month (e.g., lag_rev_1 = R(t))."""
    df = build_panel(order_lines, entity=entity, window=window)
    # Mutant leak: overwrite lag_rev_1 with current month revenue
    df["lag_rev_1"] = df["revenue"].copy()
    return df


def build_panel_mutant_b(
    order_lines: pd.DataFrame,
    entity: str = "product_id",
    window: list[str] | None = None,
) -> pd.DataFrame:
    """Mutant (b): entry computed from full-data active counts (future lookahead)."""
    cfg = get_project_config()
    thr = get_thresholds()

    if window is None:
        window = get_canonical_window(order_lines)

    lag_depth = int(cfg["lag_depth"])
    max_month_in_data = str(order_lines["month"].max()) if not order_lines.empty else window[-1]
    modeled_months = [m for m in window[lag_depth:] if m <= max_month_in_data]
    min_active = int(thr["phase1"]["min_active_months_before_entry"])

    df_win = order_lines[order_lines["month"].isin(window)].copy()
    agg = (
        df_win.groupby([entity, "region", "month"])
        .agg(
            rev=("price", "sum"),
            lines=("order_item_id", "count"),
        )
        .reset_index()
    )

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
    # LEAK: Count active months across ALL window months rather than strictly prior months!
    full_window_active_counts = piv_active.sum(axis=1)

    if entity == "category":
        prod_cat_map = {c: c for c in df_win["category"].unique()}
    else:
        prod_cat_map = df_win.groupby(entity)["category"].first().to_dict()

    first_active_idx: dict[tuple[Any, str], int] = {}
    arr_active = piv_active.to_numpy()
    series_keys = piv_r.index.tolist()
    for i, s in enumerate(series_keys):
        nz = np.nonzero(arr_active[i])[0]
        first_active_idx[s] = int(nz[0]) if len(nz) > 0 else 999

    rows: list[pd.DataFrame] = []
    for m in modeled_months:
        idx_in_win = window.index(m)

        # MUTANT LEAK: Use full window active counts instead of prior active counts
        qual_mask = full_window_active_counts >= min_active
        qual_series = piv_r.index[qual_mask]

        if len(qual_series) == 0:
            continue

        month_int = int(m.split("-")[1])
        r_t = piv_r.loc[qual_series, m].to_numpy()
        l_t = piv_l.loc[qual_series, m].to_numpy()

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

        start_6 = max(0, idx_in_win - 6)
        r_6 = piv_r.loc[qual_series, window[start_6:idx_in_win]].sum(axis=1).to_numpy()

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
            "active_months_before": full_window_active_counts[qual_mask].to_numpy(),
            "months_since_first_sale": since_first,
            "month_of_year": month_int,
        })
        rows.append(m_df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def split_and_save_panel_mutant_c(
    order_lines: pd.DataFrame,
    entity: str = "product_id",
    processed_dir: Path | str | None = None,
    holdout_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Mutant (c): dev panel built from FULL data then sliced (not truncated first)."""
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

    # LEAK: Build full panel first, then slice
    panel_full = build_panel(order_lines, entity=entity, window=window)
    panel_dev = panel_full[panel_full["month"].isin(dev_months)].copy().reset_index(drop=True)
    panel_holdout = panel_full[panel_full["month"].isin(holdout_months)].copy().reset_index(drop=True)

    proc_path.mkdir(parents=True, exist_ok=True)
    panel_dev.to_parquet(proc_path / "panel_dev.parquet", index=False)

    hold_path.mkdir(parents=True, exist_ok=True)
    holdout_file = hold_path / "panel_holdout.parquet"
    panel_holdout.to_parquet(holdout_file, index=False)

    h_sha = hashlib.sha256(holdout_file.read_bytes()).hexdigest()
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

    return panel_dev, panel_holdout, seal_data, {}