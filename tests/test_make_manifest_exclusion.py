"""Tests for reports manifest exclusions."""
from __future__ import annotations

from pathlib import Path

from scripts.make_manifest import is_excluded


def test_make_manifest_exclusion(tmp_path: Path) -> None:
    # Build a temporary reports-like tree
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ci_run.json").write_text("{}", encoding="utf-8")
    (reports_dir / "gate_10.json").write_text("{}", encoding="utf-8")
    (reports_dir / "MANIFEST.json").write_text("{}", encoding="utf-8")
    (reports_dir / "holdout_results.json").write_text("{}", encoding="utf-8")

    assert is_excluded("reports/ci_run.json") is True
    assert is_excluded("reports/gate_10.json") is True
    assert is_excluded("reports/MANIFEST.json") is True
    assert is_excluded("reports/holdout_results.json") is False

    # Also verify using relative paths from the temporary tree
    assert is_excluded(str((reports_dir / "ci_run.json").relative_to(tmp_path)).replace("\\", "/")) is True
    assert is_excluded(str((reports_dir / "gate_10.json").relative_to(tmp_path)).replace("\\", "/")) is True
    assert is_excluded(str((reports_dir / "MANIFEST.json").relative_to(tmp_path)).replace("\\", "/")) is True
    assert is_excluded(str((reports_dir / "holdout_results.json").relative_to(tmp_path)).replace("\\", "/")) is False
