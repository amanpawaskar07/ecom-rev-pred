"""Phase 6 mutants for SHAP verification and leakage checks."""
from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer

from revpred.explain import (
    check_background_disjoint,
    check_group_coverage,
    check_model_matches_served,
    groups_for,
)


def run_mutant_a_validation_as_background(
    val_row_ids: Sequence[Any],
) -> bool:
    """Mutant A: uses validation rows as background (violates background disjointness)."""
    # Background overlaps validation rows
    bg_row_ids = list(val_row_ids)[:50]
    return check_background_disjoint(val_row_ids, bg_row_ids)


def run_mutant_b_refit_model_mismatch(
    clean_model: Any,
    expected_coef_sha: str,
) -> bool:
    """Mutant B: uses an altered / refit model with different coefficients."""
    mutant_model = copy.deepcopy(clean_model)
    # Perturb coefficients
    mutant_model.best_estimator_.coef_ = mutant_model.best_estimator_.coef_ + 0.1
    return check_model_matches_served(mutant_model, expected_coef_sha)


def run_mutant_c_additivity_in_revenue_space(
    phi: np.ndarray,
    base_value: float,
    preds_log: np.ndarray,
    max_abs_err: float = 1e-3,
) -> bool:
    """Mutant C: checks additivity in nonlinear expm1 (revenue) space instead of log1p space."""
    preds_rev = np.clip(np.expm1(preds_log), 0.0, None)
    base_rev = float(np.clip(np.expm1(base_value), 0.0, None))
    phi_rev = np.expm1(phi)
    rev_err = float(np.max(np.abs(np.sum(phi_rev, axis=1) + base_rev - preds_rev)))
    return bool(rev_err <= max_abs_err)


def run_mutant_d_drop_one_hot_level(
    preprocessor: ColumnTransformer,
) -> bool:
    """Mutant D: group mapping drops one one-hot level column."""
    clean_groups = groups_for(preprocessor)
    # Drop the last column from group mapping
    mutant_groups = clean_groups[:-1]
    return check_group_coverage(preprocessor, mutant_groups)
