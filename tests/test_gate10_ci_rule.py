"""Tests for Gate 10 CI run validation rule."""
from __future__ import annotations

import subprocess
from pathlib import Path

from gates.gate_10 import ci_run_status


def init_git_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test CI User"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test-ci@example.com"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "init.txt").write_text("initial commit", encoding="utf-8")
    subprocess.run(["git", "add", "init.txt"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo_dir, check=True, capture_output=True)


def get_head_sha(repo_dir: Path) -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
    return res.stdout.strip()


def test_sha_equals_head_passes(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    init_git_repo(repo_dir)
    head_sha = get_head_sha(repo_dir)
    ci_run = {"sha": head_sha, "conclusion": "success"}
    status, reason = ci_run_status(ci_run, head_sha, repo_dir)
    assert status == "PASS"
    assert "matches HEAD" in reason


def test_sha_parent_only_ci_run_changed_passes(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    init_git_repo(repo_dir)
    parent_sha = get_head_sha(repo_dir)

    # make commit modifying only allowed files: reports/ci_run.json, README.md, README.md.tmpl
    reports_dir = repo_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ci_run.json").write_text('{"sha": "abc", "conclusion": "success"}', encoding="utf-8")
    (repo_dir / "README.md").write_text("# Readme\nUpdated docs.", encoding="utf-8")
    (repo_dir / "README.md.tmpl").write_text("# Template\nUpdated tmpl.", encoding="utf-8")
    subprocess.run(["git", "add", "reports/ci_run.json", "README.md", "README.md.tmpl"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "doc updates and ci run"], cwd=repo_dir, check=True, capture_output=True)

    head_sha = get_head_sha(repo_dir)
    assert parent_sha != head_sha

    ci_run = {"sha": parent_sha, "conclusion": "success"}
    status, reason = ci_run_status(ci_run, head_sha, repo_dir)
    assert status == "PASS"
    assert "allowed files changed" in reason


def test_sha_parent_code_file_changed_fails(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    init_git_repo(repo_dir)
    parent_sha = get_head_sha(repo_dir)

    # commit code changes along with ci_run.json
    reports_dir = repo_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ci_run.json").write_text('{"sha": "abc", "conclusion": "success"}', encoding="utf-8")
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "module.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "reports/ci_run.json", "src/module.py"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "code and ci update"], cwd=repo_dir, check=True, capture_output=True)

    head_sha = get_head_sha(repo_dir)
    ci_run = {"sha": parent_sha, "conclusion": "success"}
    status, reason = ci_run_status(ci_run, head_sha, repo_dir)
    assert status == "FAIL"
    assert "src/module.py" in reason


def test_sha_not_ancestor_fails(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    init_git_repo(repo_dir)

    # create a branch, commit on branch
    subprocess.run(["git", "checkout", "-b", "other-branch"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "branch.txt").write_text("on branch", encoding="utf-8")
    subprocess.run(["git", "add", "branch.txt"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "commit on other branch"], cwd=repo_dir, check=True, capture_output=True)
    branch_sha = get_head_sha(repo_dir)

    # switch back to main/master, make different commit
    subprocess.run(["git", "checkout", "-"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "main.txt").write_text("on main", encoding="utf-8")
    subprocess.run(["git", "add", "main.txt"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "commit on main"], cwd=repo_dir, check=True, capture_output=True)
    head_sha = get_head_sha(repo_dir)

    ci_run = {"sha": branch_sha, "conclusion": "success"}
    status, reason = ci_run_status(ci_run, head_sha, repo_dir)
    assert status == "FAIL"
    assert "not an ancestor" in reason


def test_conclusion_failure_fails(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    init_git_repo(repo_dir)
    head_sha = get_head_sha(repo_dir)
    ci_run = {"sha": head_sha, "conclusion": "failure"}
    status, reason = ci_run_status(ci_run, head_sha, repo_dir)
    assert status == "FAIL"
    assert "failure" in reason


def test_missing_file_is_pending(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    init_git_repo(repo_dir)
    head_sha = get_head_sha(repo_dir)
    status, reason = ci_run_status(None, head_sha, repo_dir)
    assert status == "pending"
    assert "does not exist yet" in reason

