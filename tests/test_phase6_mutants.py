"""Tests verifying that Phase 6 mutants are rejected by verification checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

from revpred.explain import (
    check_additivity,
    check_background_disjoint,
    check_group_coverage,
    check_model_matches_served,
    groups_for,
)
from tests.mutants.mutants_phase6 import (
    run_mutant_a_validation_as_background,
    run_mutant_b_refit_model_mismatch,
    run_mutant_c_additivity_in_revenue_space,
    run_mutant_d_drop_one_hot_level,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
SHAP_DIR = REPO_ROOT / "reports" / "shap"


def test_mutant_a_validation_as_background_caught() -> None:
    """Mutant A (validation rows used as background) is caught by disjointness check."""
    row_ids = np.load(SHAP_DIR / "row_ids.npy", allow_pickle=True)
    # Clean check: training background is disjoint from validation rows
    clean_bg_ids = ["other_row_1", "other_row_2", "other_row_3"]
    assert check_background_disjoint(row_ids, clean_bg_ids)

    # Mutant A uses validation rows as background -> must be caught
    passed = run_mutant_a_validation_as_background(row_ids)
    assert not passed, "Mutant A was not caught by background disjointness check!"


def test_mutant_b_refit_model_mismatch_caught() -> None:
    """Mutant B (attributions from an unverified refit model) is caught by coefficient SHA check."""
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    expected_sha = hashlib.sha256(
        np.ascontiguousarray(model.best_estimator_.coef_).tobytes()
    ).hexdigest()

    # Clean check
    assert check_model_matches_served(model, expected_sha)

    # Mutant B modifies coefficients -> must be caught
    passed = run_mutant_b_refit_model_mismatch(model, expected_sha)
    assert not passed, "Mutant B was not caught by model coefficient SHA check!"


def test_mutant_c_additivity_in_revenue_space_caught() -> None:
    """Mutant C (additivity evaluated in expm1 revenue space) is caught by tolerance check."""
    phi = np.load(SHAP_DIR / "phi.npy")
    base = float(np.load(SHAP_DIR / "base.npy")[0])
    preds = np.load(SHAP_DIR / "preds.npy")

    # Clean check in log1p space passes
    assert check_additivity(phi, base, preds, max_abs_err=1e-3)

    # Mutant C checks additivity in revenue (expm1) space -> must fail
    passed = run_mutant_c_additivity_in_revenue_space(phi, base, preds, max_abs_err=1e-3)
    assert not passed, "Mutant C was not caught by revenue space additivity check!"


def test_mutant_d_drop_one_hot_level_caught() -> None:
    """Mutant D (group mapping drops one level of a one-hot feature) is caught by coverage check."""
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    clean_groups = groups_for(model.preprocessor_)

    # Clean check passes
    assert check_group_coverage(model.preprocessor_, clean_groups)

    # Mutant D drops one level -> must be caught
    passed = run_mutant_d_drop_one_hot_level(model.preprocessor_)
    assert not passed, "Mutant D was not caught by group coverage check!"
