"""Sealed holdout loader with cryptographic verification and append-only audit ledger."""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOLDOUT_PARQUET = REPO_ROOT / "data" / "holdout" / "panel_holdout.parquet"
SEAL_JSON = REPO_ROOT / "reports" / "holdout_seal.json"
LEDGER_JSONL = REPO_ROOT / "reports" / "holdout_ledger.jsonl"


class HoldoutSealError(Exception):
    """Raised when holdout data does not match the cryptographic seal."""


def load_holdout(
    purpose: str,
    holdout_path: Path | str | None = None,
    seal_path: Path | str | None = None,
    ledger_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load sealed holdout dataset after cryptographic seal verification and audit logging."""
    if not purpose or not isinstance(purpose, str):
        raise ValueError("A non-empty purpose string is required to load the sealed holdout data.")

    h_path = Path(holdout_path) if holdout_path else HOLDOUT_PARQUET
    s_path = Path(seal_path) if seal_path else SEAL_JSON
    l_path = Path(ledger_path) if ledger_path else LEDGER_JSONL

    if not h_path.exists():
        raise FileNotFoundError(f"Holdout data file not found: {h_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"Holdout seal file not found: {s_path}")

    # Verify sha256
    with open(s_path, encoding="utf-8") as f:
        seal_data = json.load(f)

    expected_sha = seal_data["sha256"]
    actual_sha = hashlib.sha256(h_path.read_bytes()).hexdigest()

    if actual_sha != expected_sha:
        raise HoldoutSealError(
            f"Holdout sha256 mismatch! Expected {expected_sha}, got {actual_sha}"
        )

    # Caller file and git commit
    caller_frame = inspect.stack()[1]
    caller_file = str(Path(caller_frame.filename).resolve())

    proc_git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    git_sha = proc_git.stdout.strip() if proc_git.returncode == 0 else "unknown"

    ledger_entry = {
        "ts": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "caller_file": caller_file,
        "purpose": purpose,
    }

    l_path.parent.mkdir(parents=True, exist_ok=True)
    with open(l_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")

    return pd.read_parquet(h_path)