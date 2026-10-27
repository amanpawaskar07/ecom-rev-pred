"""Tests for holdout isolation, cryptographic seal verification, and audit ledger."""
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from revpred.holdout import HoldoutSealError, load_holdout

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_grep_load_holdout():
    """Verify load_holdout is referenced only in holdout.py and tests."""
    # Grep in src/
    proc = subprocess.run(
        ["grep", "-rn", "load_holdout", "src/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    # Every reference in src/ must be in src/revpred/holdout.py
    for line in lines:
        filepath = line.split(":")[0]
        assert filepath == "src/revpred/holdout.py", f"Unauthorized load_holdout reference: {line}"


def test_load_holdout_flow_with_temp_files(tmp_path):
    """Test load_holdout seal check and audit logging using isolated temp paths."""
    # Create fake holdout parquet
    holdout_file = tmp_path / "panel_holdout.parquet"
    df_fake = pd.DataFrame({"row_id": ["r1", "r2"], "revenue": [10.0, 20.0]})
    df_fake.to_parquet(holdout_file, index=False)
    actual_sha = hashlib.sha256(holdout_file.read_bytes()).hexdigest()

    # Create seal
    seal_file = tmp_path / "holdout_seal.json"
    seal_data = {
        "sha256": actual_sha,
        "rows": 2,
        "months": ["2018-06"],
        "commit": "testcommit",
    }
    seal_file.write_text(json.dumps(seal_data), encoding="utf-8")

    ledger_file = tmp_path / "holdout_ledger.jsonl"

    # Test successful load
    df_loaded = load_holdout(
        purpose="unit_test_verification",
        holdout_path=holdout_file,
        seal_path=seal_file,
        ledger_path=ledger_file,
    )
    assert len(df_loaded) == 2
    assert ledger_file.exists()

    ledger_lines = [json.loads(line) for line in ledger_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(ledger_lines) == 1
    assert ledger_lines[0]["purpose"] == "unit_test_verification"

    # Test seal mismatch detection
    seal_file.write_text(json.dumps({"sha256": "wrong_sha"}), encoding="utf-8")
    with pytest.raises(HoldoutSealError, match="sha256 mismatch"):
        load_holdout(
            purpose="tamper_test",
            holdout_path=holdout_file,
            seal_path=seal_file,
            ledger_path=ledger_file,
        )


def test_empty_purpose_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-empty purpose"):
        load_holdout("")