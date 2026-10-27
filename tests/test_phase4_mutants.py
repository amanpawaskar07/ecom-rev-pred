"""Tests that Phase 4 mutants are caught by causality and leakage checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from revpred.stack import TemporalStacker
from tests.mutants.mutants_phase4 import (
    AllRowsFitStackerMutantB,
    InSampleMetaStackerMutantC,
    ShuffledKFoldStackerMutantA,
)


def _make_test_df(n_per_month: int = 50) -> pd.DataFrame:
    months = ["2017-01", "2017-02", "2017-03", "2017-04", "2017-05"]
    n = len(months) * n_per_month
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "row_id": np.arange(n),
        "month": np.repeat(months, n_per_month),
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "revenue": np.abs(rng.standard_normal(n)) * 100.0,
    })


def test_mutant_a_shuffled_kfold_caught() -> None:
    """Mutant A (shuffled KFold) produces causality violations."""
    df = _make_test_df()

    # Clean temporal stacker has 0 causality violations
    clean = TemporalStacker(min_inner_train_months=3, features=["f1", "f2"])
    clean.fit(df)
    clean_violations = int((clean.oof_log["fit_max_month"] >= clean.oof_log["month"]).sum())
    assert clean_violations == 0

    # Mutant A uses shuffled KFold, which produces causality violations
    mutant_a = ShuffledKFoldStackerMutantA(min_inner_train_months=3, features=["f1", "f2"])
    mutant_a.fit(df)
    mutant_violations = int((mutant_a.oof_log["fit_max_month"] >= mutant_a.oof_log["month"]).sum())
    assert mutant_violations > 0, "Mutant A was not caught by causality check!"


def test_mutant_b_all_rows_fit_caught() -> None:
    """Mutant B (bases fit on all rows) produces causality violations."""
    df = _make_test_df()

    mutant_b = AllRowsFitStackerMutantB(min_inner_train_months=3, features=["f1", "f2"])
    mutant_b.fit(df)
    mutant_violations = int((mutant_b.oof_log["fit_max_month"] >= mutant_b.oof_log["month"]).sum())
    assert mutant_violations > 0, "Mutant B was not caught by causality check!"


def test_mutant_c_in_sample_meta_caught() -> None:
    """Mutant C (meta fit on in-sample base predictions) produces causality violations."""
    df = _make_test_df()

    mutant_c = InSampleMetaStackerMutantC(min_inner_train_months=3, features=["f1", "f2"])
    mutant_c.fit(df)
    mutant_violations = int((mutant_c.oof_log["fit_max_month"] >= mutant_c.oof_log["month"]).sum())
    assert mutant_violations > 0, "Mutant C was not caught by causality check!"
