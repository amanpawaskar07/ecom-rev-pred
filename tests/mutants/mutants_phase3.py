"""Phase 3 mutants for leakage and causal testing."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from revpred.resample import SmoterResampler, _identify_column_types


def oversample_before_split_mutant_a(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[Any, Any, np.ndarray, np.ndarray, float]:
    """Mutant (a): oversample the whole dataset BEFORE a random train/val split (LEAK)."""
    # 1. LEAK: SMOTER oversamples all rows before split
    resampler = SmoterResampler()
    all_idx = np.arange(len(X))
    res = resampler.fit_resample(X=X, y=y, train_idx=all_idx, seed=seed)
    X_res = res.X
    y_res = res.y

    # 2. Split the resampled rows into train and "validation"
    X_tr_mut, X_val_mut, y_tr_mut, y_val_mut = train_test_split(
        X_res, y_res, test_size=test_size, random_state=seed
    )
    return X_tr_mut, X_val_mut, y_tr_mut, y_val_mut, res.tail_thr


class QuantileTargetTransformMutantB:
    """Mutant (b): QuantileTransformer fit on ALL rows (train + validation), ignoring fit guard (LEAK)."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.transformer_: QuantileTransformer | None = None
        self.zero_val_transformed_: float | None = None

    def fit_all(self, y_train: Any, y_val: Any) -> QuantileTargetTransformMutantB:
        """LEAK: Combines train and validation targets and fits on all rows."""
        y_tr = np.asarray(y_train, dtype=float).ravel()
        y_vl = np.asarray(y_val, dtype=float).ravel()
        y_all = np.concatenate([y_tr, y_vl])

        n_all = len(y_all)
        n_quantiles = min(1000, n_all)

        self.transformer_ = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=None,
            random_state=self.random_state,
        )
        self.transformer_.fit(y_all.reshape(-1, 1))
        z_trans = self.transformer_.transform(np.array([[0.0]]))[0, 0]
        self.zero_val_transformed_ = float(z_trans)
        return self

    def transform(self, y: Any) -> np.ndarray:
        if self.transformer_ is None:
            raise RuntimeError("Mutant must be fitted before transform.")
        y_arr = np.asarray(y, dtype=float).ravel()
        return self.transformer_.transform(np.maximum(y_arr, 0.0).reshape(-1, 1)).ravel()

    def inverse(self, y_trans: Any) -> np.ndarray:
        if self.transformer_ is None:
            raise RuntimeError("Mutant must be fitted before inverse.")
        p_arr = np.asarray(y_trans, dtype=float).ravel().reshape(-1, 1)
        raw_inv = self.transformer_.inverse_transform(p_arr).ravel()
        clipped = np.clip(raw_inv, 0.0, None)
        if self.zero_val_transformed_ is not None:
            clipped[p_arr.ravel() <= self.zero_val_transformed_] = 0.0
        return clipped


def smoter_all_neighbours_mutant_c(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: np.ndarray,
    train_idx: np.ndarray,
    seed: int = 42,
    tail_thr: float | None = None,
) -> tuple[Any, np.ndarray]:
    """Mutant (c): kNN neighbours drawn from ALL rows (train + validation) instead of train only (LEAK)."""
    is_df = isinstance(X_train, pd.DataFrame)
    if is_df:
        X_all = pd.concat([X_train, X_val], ignore_index=True)
        y_all = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
        num_cols, _, _ = _identify_column_types(X_train)
        X_num_all = X_all[num_cols].to_numpy(dtype=float)
        X_num_tr = X_train[num_cols].to_numpy(dtype=float)
    else:
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        X_num_all = X_all
        X_num_tr = X_train

    if tail_thr is None:
        pos = y_train[y_train > 0]
        tail_thr = float(np.quantile(pos, 0.90)) if len(pos) > 0 else 0.0

    # Rare rows in training
    rare_tr_mask = y_train >= tail_thr
    rare_tr_idx = np.where(rare_tr_mask)[0]

    # LEAK: Rare pool includes rare rows from validation as well!
    rare_all_mask = y_all >= tail_thr
    rare_all_idx = np.where(rare_all_mask)[0]

    scaler = StandardScaler()
    scaler.fit(X_num_tr)
    Z_all = scaler.transform(X_num_all)

    # Fit kNN on ALL rare rows (including validation)
    nn = NearestNeighbors(n_neighbors=6, algorithm="auto")
    nn.fit(Z_all[rare_all_idx])

    # Query neighbours for train rare rows from all rare rows
    Z_tr_rare = Z_all[rare_tr_idx]
    _, kn_indices = nn.kneighbors(Z_tr_rare)

    rng = np.random.default_rng(seed)
    n_orig = len(X_train)
    n_rare = len(rare_tr_idx)
    n_nonrare = n_orig - n_rare
    n_syn = max(0, int(np.round(0.30 * n_nonrare / 0.70)) - n_rare)

    X_syn_num = np.zeros((n_syn, X_num_tr.shape[1]), dtype=float)
    y_syn = np.zeros(n_syn, dtype=float)

    for i in range(n_syn):
        ia_tr = rng.integers(0, n_rare)
        # Random neighbour picked from the all-rows pool
        k_choice = rng.integers(1, 6)
        ib_all_pool = kn_indices[ia_tr, k_choice]

        orig_a = rare_tr_idx[ia_tr]
        orig_b_all = rare_all_idx[ib_all_pool]  # Might be in validation!

        lam = float(rng.uniform(0.0, 1.0))
        y_syn[i] = y_all[orig_a] + lam * (y_all[orig_b_all] - y_all[orig_a])
        X_syn_num[i] = X_num_all[orig_a] + lam * (X_num_all[orig_b_all] - X_num_all[orig_a])

    if is_df:
        syn_df = pd.DataFrame(X_syn_num, columns=num_cols)
        # Fill other columns from parent a
        for c in X_train.columns:
            if c not in num_cols:
                syn_df[c] = [X_train.iloc[rare_tr_idx[0]][c]] * n_syn
        X_res = pd.concat([X_train, syn_df[X_train.columns]], ignore_index=True)
    else:
        X_res = np.vstack([X_train, X_syn_num])

    y_res = np.concatenate([y_train, y_syn])
    return X_res, y_res
