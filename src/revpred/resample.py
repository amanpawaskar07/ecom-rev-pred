"""SMOTER-style continuous oversampling for imbalanced regression with guard objects.

Definitions:
- rare row: revenue >= tail_thr, tail_thr = quantile(train revenue > 0, metrics.tail_quantile)
  computed on training rows only.
- SMOTER algorithm:
  1. Numeric feature columns standardised with train stats.
  2. kNN (k=5) among RARE rows only.
  3. Each synthetic row = seed rare row a moved by lambda ~ U(0,1) towards random neighbour b:
     - numeric features interpolated: x_syn = x_a + lambda * (x_b - x_a)
     - y_syn = y_a + lambda * (y_b - y_a) on the transformed-target scale
     - categorical and binary features copied from a
  4. Oversample until rare rows are 30% of the resampled training set (+/- 1 row).
  5. Fewer than 6 rare rows in a fold -> return unchanged and record the skip.
  6. The resampler receives train_idx and asserts it is a subset of the fold's train indices.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from revpred.config import get_thresholds


class TrainGuard:
    """Guard object ensuring that resampling only operates on indices strictly within the fold."""

    def __init__(self, allowed_train_idx: Sequence[int] | set[int] | np.ndarray):
        if isinstance(allowed_train_idx, (set, frozenset)):
            self.allowed_set = set(allowed_train_idx)
        else:
            self.allowed_set = set(np.asarray(allowed_train_idx).tolist())

    def validate(self, train_idx: Sequence[int] | np.ndarray) -> None:
        """Assert train_idx is a subset of the allowed training indices."""
        idx_set = set(np.asarray(train_idx).tolist())
        disallowed = idx_set - self.allowed_set
        if disallowed:
            raise ValueError(
                f"TrainGuard violation: {len(disallowed)} indices outside allowed train indices (sample: {list(disallowed)[:5]})."
            )

    def __contains__(self, idx: int) -> bool:
        return idx in self.allowed_set


class ResampleResult(tuple):
    """Container for SMOTER output subclassing tuple (X_res, y_res, parent_ids, lambdas)."""

    def __new__(
        cls,
        X_res: Any,
        y_res: np.ndarray,
        parent_ids: list[tuple[Any, Any]],
        lambdas: np.ndarray,
        skipped: bool = False,
        n_syn: int = 0,
        tail_thr: float = 0.0,
    ) -> Self:
        return super().__new__(cls, (X_res, y_res, parent_ids, lambdas))

    def __init__(
        self,
        X_res: Any,
        y_res: np.ndarray,
        parent_ids: list[tuple[Any, Any]],
        lambdas: np.ndarray,
        skipped: bool = False,
        n_syn: int = 0,
        tail_thr: float = 0.0,
    ) -> None:
        self.X = X_res
        self.y = y_res
        self.parent_ids = parent_ids
        self.lambdas = lambdas
        self.skipped = skipped
        self.n_syn = n_syn
        self.tail_thr = tail_thr


def _identify_column_types(
    df: pd.DataFrame,
    numeric_cols: Sequence[str] | None = None,
    categorical_cols: Sequence[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Partition DataFrame columns into (numeric, binary, categorical)."""
    if numeric_cols is not None:
        num_cols = list(numeric_cols)
        cat_cols = list(categorical_cols) if categorical_cols is not None else [c for c in df.columns if c not in num_cols]
        bin_cols: list[str] = []
        for c in num_cols:
            vals = set(df[c].dropna().unique())
            if vals.issubset({0, 1}):
                bin_cols.append(c)
        num_cols = [c for c in num_cols if c not in bin_cols]
        return num_cols, bin_cols, cat_cols

    num_cols = []
    bin_cols = []
    cat_cols = []

    for col in df.columns:
        dt = df[col].dtype
        if dt == "object" or dt.name == "category" or dt == "string":
            cat_cols.append(col)
        else:
            vals = set(df[col].dropna().unique())
            if vals.issubset({0, 1}):
                bin_cols.append(col)
            else:
                num_cols.append(col)

    return num_cols, bin_cols, cat_cols


class SmoterResampler:
    """SMOTER oversampler for continuous targets."""

    def __init__(
        self,
        guard: TrainGuard | Sequence[int] | set[int] | None = None,
        k_neighbors: int = 5,
        target_rare_share: float = 0.30,
        tail_quantile: float | None = None,
        numeric_cols: Sequence[str] | None = None,
        categorical_cols: Sequence[str] | None = None,
    ) -> None:
        self.guard = guard if isinstance(guard, TrainGuard) or guard is None else TrainGuard(guard)
        self.k_neighbors = k_neighbors
        self.target_rare_share = target_rare_share
        if tail_quantile is None:
            thr = get_thresholds()
            self.tail_quantile = float(thr["metrics"]["tail_quantile"])
        else:
            self.tail_quantile = float(tail_quantile)
        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else None
        self.categorical_cols = list(categorical_cols) if categorical_cols is not None else None

    def fit_resample(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        train_idx: Sequence[int] | np.ndarray,
        seed: int,
        guard: TrainGuard | Sequence[int] | set[int] | None = None,
        tail_thr: float | None = None,
        y_raw: np.ndarray | pd.Series | None = None,
    ) -> ResampleResult:
        """Run SMOTER continuous oversampling on training fold.

        Parameters:
            X: Training features (DataFrame or 2D array).
            y: Training target on transformed scale.
            train_idx: Indices representing the training sample identity.
            seed: RNG seed.
            guard: Optional override of TrainGuard.
            tail_thr: Optional precomputed tail threshold on raw revenue scale.
            y_raw: Optional raw revenue targets (if y is transformed).
        """
        # 1. Guard validation
        active_guard = guard if isinstance(guard, TrainGuard) else (TrainGuard(guard) if guard is not None else self.guard)
        if active_guard is not None:
            active_guard.validate(train_idx)

        train_idx_arr = np.asarray(train_idx)
        if len(train_idx_arr) != len(X):
            raise ValueError(
                f"Length mismatch between train_idx ({len(train_idx_arr)}) and X ({len(X)})."
            )

        y_trans = np.asarray(y, dtype=float).ravel()
        if len(y_trans) != len(X):
            raise ValueError(f"Length mismatch between y ({len(y_trans)}) and X ({len(X)}).")

        # 2. Determine rare rows
        if y_raw is not None:
            raw_target = np.asarray(y_raw, dtype=float).ravel()
        else:
            raw_target = y_trans

        if tail_thr is None:
            pos_raw = raw_target[raw_target > 0]
            if len(pos_raw) == 0:
                calc_thr = 0.0
            else:
                calc_thr = float(np.quantile(pos_raw, self.tail_quantile))
        else:
            calc_thr = float(tail_thr)

        rare_mask = raw_target >= calc_thr
        rare_indices = np.where(rare_mask)[0]
        n_rare = len(rare_indices)

        # 3. Check minimum rare rows requirement
        if n_rare < (self.k_neighbors + 1):
            return ResampleResult(
                X_res=X,
                y_res=y_trans,
                parent_ids=[],
                lambdas=np.array([], dtype=float),
                skipped=True,
                n_syn=0,
                tail_thr=calc_thr,
            )

        # 4. Partition features
        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            num_cols, bin_cols, cat_cols = _identify_column_types(
                X, numeric_cols=self.numeric_cols, categorical_cols=self.categorical_cols
            )
            X_num = X[num_cols].to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            num_cols = list(range(X_arr.shape[1]))
            bin_cols = []
            cat_cols = []
            X_num = X_arr

        # 5. Standardise numeric feature columns with train stats
        means = np.mean(X_num, axis=0)
        stds = np.std(X_num, axis=0)
        stds = np.where(stds == 0.0, 1.0, stds)
        Z_num = (X_num - means) / stds

        # 6. kNN (k=5) among RARE rows only
        Z_rare = Z_num[rare_indices]
        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm="auto")
        nn.fit(Z_rare)
        _, kn_indices = nn.kneighbors(Z_rare)

        # 7. Calculate number of synthetic rows required
        n_orig = len(X)
        n_orig - n_rare
        # Oversample until rare rows are target_rare_share (30%) of resampled training set:
        # (n_rare + M) / (n_orig + M) = target_rare_share
        # (1 - target) * M = target * n_orig - n_rare
        denom = 1.0 - self.target_rare_share
        n_syn = int(np.round((self.target_rare_share * n_orig - n_rare) / denom))

        if n_syn <= 0:
            return ResampleResult(
                X_res=X,
                y_res=y_trans,
                parent_ids=[],
                lambdas=np.array([], dtype=float),
                skipped=False,
                n_syn=0,
                tail_thr=calc_thr,
            )

        # 8. Synthetic row generation
        rng = np.random.default_rng(seed)
        parent_ids: list[tuple[Any, Any]] = []
        lambdas: list[float] = []

        X_syn_num = np.zeros((n_syn, len(num_cols)), dtype=float)
        y_syn = np.zeros(n_syn, dtype=float)

        if is_df:
            X_syn_cat: dict[str, list[Any]] = {c: [] for c in cat_cols}
            X_syn_bin = np.zeros((n_syn, len(bin_cols)), dtype=X[bin_cols].dtypes.iloc[0] if bin_cols else int)

        for i in range(n_syn):
            idx_a_rare = rng.integers(0, n_rare)
            k_choice = rng.integers(1, self.k_neighbors + 1)
            idx_b_rare = kn_indices[idx_a_rare, k_choice]

            orig_a = rare_indices[idx_a_rare]
            orig_b = rare_indices[idx_b_rare]

            lam = float(rng.uniform(0.0, 1.0))
            parent_ids.append((train_idx_arr[orig_a], train_idx_arr[orig_b]))
            lambdas.append(lam)

            # Interpolate y on transformed scale
            y_syn[i] = y_trans[orig_a] + lam * (y_trans[orig_b] - y_trans[orig_a])

            # Interpolate numeric features
            X_syn_num[i] = X_num[orig_a] + lam * (X_num[orig_b] - X_num[orig_a])

            if is_df:
                # Binary features copied from a
                if bin_cols:
                    X_syn_bin[i] = X.iloc[orig_a][bin_cols].values
                # Categorical features copied from a
                for c in cat_cols:
                    X_syn_cat[c].append(X.iloc[orig_a][c])

        # 9. Combine original and synthetic rows (original rows come FIRST and unchanged)
        if is_df:
            syn_df_dict = {}
            for col_idx, col in enumerate(num_cols):
                syn_df_dict[col] = X_syn_num[:, col_idx]
            for col_idx, col in enumerate(bin_cols):
                syn_df_dict[col] = X_syn_bin[:, col_idx]
            for col in cat_cols:
                syn_df_dict[col] = X_syn_cat[col]

            syn_df = pd.DataFrame(syn_df_dict)[X.columns]
            X_res = pd.concat([X, syn_df], ignore_index=True)
        else:
            X_res = np.vstack([X, X_syn_num])

        y_res = np.concatenate([y_trans, y_syn])

        return ResampleResult(
            X_res=X_res,
            y_res=y_res,
            parent_ids=parent_ids,
            lambdas=np.asarray(lambdas, dtype=float),
            skipped=False,
            n_syn=n_syn,
            tail_thr=calc_thr,
        )


def fit_resample(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray | pd.Series,
    train_idx: Sequence[int] | np.ndarray,
    seed: int,
    guard: TrainGuard | Sequence[int] | set[int] | None = None,
    tail_thr: float | None = None,
    y_raw: np.ndarray | pd.Series | None = None,
    k_neighbors: int = 5,
    target_rare_share: float = 0.30,
) -> ResampleResult:
    """Convenience wrapper for SmoterResampler().fit_resample."""
    resampler = SmoterResampler(
        guard=guard,
        k_neighbors=k_neighbors,
        target_rare_share=target_rare_share,
    )
    return resampler.fit_resample(
        X=X,
        y=y,
        train_idx=train_idx,
        seed=seed,
        guard=guard,
        tail_thr=tail_thr,
        y_raw=y_raw,
    )
