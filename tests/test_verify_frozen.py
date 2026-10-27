"""Tests for scripts/verify_frozen.py self-test cases."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_frozen.py"


def run_verify(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_verify_frozen_no_tag(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    proc = run_verify(tmp_path)
    assert proc.returncode == 0
    assert "NOTICE" in proc.stdout


def test_verify_frozen_untouched(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True)

    # Create directories and files
    gates_dir = tmp_path / "gates"
    gates_dir.mkdir(parents=True)
    (gates_dir / "thresholds.yaml").write_text("version: 1\n", encoding="utf-8")

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "project.yaml").write_text("dataset: test\n", encoding="utf-8")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_dummy.py").write_text("# dummy test\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "phase0-accepted"], cwd=tmp_path, check=True)

    # Untouched -> must pass
    proc = run_verify(tmp_path)
    assert proc.returncode == 0
    assert "Verified" in proc.stdout


def test_verify_frozen_modifying_file(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True)

    gates_dir = tmp_path / "gates"
    gates_dir.mkdir(parents=True)
    t_file = gates_dir / "thresholds.yaml"
    t_file.write_text("version: 1\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "phase0-accepted"], cwd=tmp_path, check=True)

    # Modify frozen file
    t_file.write_text("version: 2\n", encoding="utf-8")
    proc = run_verify(tmp_path)
    assert proc.returncode == 1
    assert "Frozen file modified: gates/thresholds.yaml" in proc.stdout


def test_verify_frozen_adding_new_file(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True)

    gates_dir = tmp_path / "gates"
    gates_dir.mkdir(parents=True)
    (gates_dir / "thresholds.yaml").write_text("version: 1\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "phase0-accepted"], cwd=tmp_path, check=True)

    # Add a brand new file (new files are allowed)
    new_gate = gates_dir / "gate_01.py"
    new_gate.write_text("# new gate\n", encoding="utf-8")

    proc = run_verify(tmp_path)
    assert proc.returncode == 0
    assert "Verified" in proc.stdout
