"""Expanding-window cross-validation splitter with strict temporal ordering."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def expanding_window_splits(
    months: Sequence[str] | pd.Series | np.ndarray,
    min_train_months: int,
    val_months: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window (train_idx, val_idx) splits where every train month < every val month."""
    if isinstance(months, pd.Series):
        s_months = months.reset_index(drop=True)
    else:
        s_months = pd.Series(months)

    unique_months = sorted(s_months.unique())
    n_unique = len(unique_months)

    if n_unique < min_train_months + val_months:
        return []

    month_to_row_indices: dict[str, np.ndarray] = {}
    for m in unique_months:
        month_to_row_indices[m] = np.where(s_months == m)[0]

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    n_folds = n_unique - min_train_months - val_months + 1

    for k in range(n_folds):
        train_month_list = unique_months[: min_train_months + k]
        val_month_list = unique_months[min_train_months + k : min_train_months + k + val_months]

        train_indices = (
            np.concatenate([month_to_row_indices[m] for m in train_month_list])
            if train_month_list
            else np.array([], dtype=int)
        )
        val_indices = (
            np.concatenate([month_to_row_indices[m] for m in val_month_list])
            if val_month_list
            else np.array([], dtype=int)
        )

        # Verification: every train month < every val month
        assert max(train_month_list) < min(val_month_list), "Temporal leakage in CV split!"
        assert len(set(train_indices).intersection(set(val_indices))) == 0, "Overlapping train and val sets!"

        splits.append((train_indices, val_indices))

    return splits