"""Gate 10: CI, README traceable numbers, protocol-integrity and holdout audit."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from gates import lib as gl
from scripts.check_readme_numbers import check_numbers
from scripts.harness_e2e import run_harness_e2e
from scripts.make_manifest import collect_files
from scripts.render_readme import PLACEHOLDER_REGEX, render

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"

ALLOWED_CI_DIFF_FILES = {
    "reports/ci_run.json",
    "README.md",
    "README.md.tmpl",
}


def ci_run_status(
    ci_run: dict[str, Any] | None,
    head_sha: str,
    repo_dir: Path | str,
) -> tuple[str, str]:
    """Evaluate CI run record against current git state.

    Returns:
        tuple[str, str]: (status, reason) where status is "pending", "PASS", or "FAIL".
    """
    if ci_run is None:
        return (
            "pending",
            "reports/ci_run.json does not exist yet (CI has not yet run on GitHub Actions)",
        )

    conclusion = ci_run.get("conclusion")
    if conclusion != "success":
        return ("FAIL", f"CI run conclusion is '{conclusion}', expected 'success'")

    ci_sha = ci_run.get("sha")
    if not ci_sha:
        return ("FAIL", "CI run record missing 'sha' field")

    if ci_sha == head_sha:
        return ("PASS", f"CI run SHA {ci_sha} matches HEAD exactly")

    # Check if ci_sha is an ancestor of head_sha
    proc_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ci_sha, head_sha],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc_ancestor.returncode != 0:
        return ("FAIL", f"CI run SHA {ci_sha} is not an ancestor of HEAD {head_sha}")

    # Inspect diff between ci_sha and head_sha
    proc_diff = subprocess.run(
        ["git", "diff", "--name-only", ci_sha, head_sha],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc_diff.returncode != 0:
        return ("FAIL", f"git diff failed between {ci_sha} and {head_sha}: {proc_diff.stderr.strip()}")

    changed_files = [
        line.strip().replace("\\", "/")
        for line in proc_diff.stdout.splitlines()
        if line.strip()
    ]

    offending = [f for f in changed_files if f not in ALLOWED_CI_DIFF_FILES]
    if offending:
        return (
            "FAIL",
            f"Offending files modified between CI run {ci_sha} and HEAD {head_sha}: {', '.join(sorted(offending))}",
        )

    return (
        "PASS",
        f"CI run SHA {ci_sha} is ancestor of HEAD with only allowed files changed: {', '.join(changed_files)}",
    )


def build_checks() -> None:
    # 1. README unsourced numbers check
    count_unsourced, _ = check_numbers()
    gl.check("readme_unsourced_numbers", count_unsourced, "<=", "phase10.readme_unsourced_numbers_max")

    # 2. README regenerates identically
    rendered, _ = render()
    readme_bytes = README_PATH.read_bytes() if README_PATH.exists() else b""
    ident = 1 if (rendered.encode("utf-8") == readme_bytes and len(readme_bytes) > 0) else 0
    gl.check("readme_regenerates_identically", ident, "==", "global.bool_true")

    # 3. Every README placeholder resolves
    leftover = PLACEHOLDER_REGEX.search(rendered)
    placeholders_resolved = 1 if (leftover is None and len(rendered) > 0) else 0
    gl.check("every_readme_placeholder_resolves", placeholders_resolved, "==", "global.bool_true")

    # 4. Reports manifest OK
    manifest_path = REPO_ROOT / "reports" / "MANIFEST.json"
    manifest_ok = 1
    if not manifest_path.exists():
        manifest_ok = 0
    else:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                m_data = json.load(f)
            m_files = m_data.get("files", m_data)
            for rel_path, expected_sha in m_files.items():
                full_path = REPO_ROOT / rel_path
                if not full_path.exists():
                    manifest_ok = 0
                    break
                actual_sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    manifest_ok = 0
                    break
            disk_files = collect_files()
            if set(disk_files.keys()) != set(m_files.keys()):
                manifest_ok = 0
        except (OSError, json.JSONDecodeError, KeyError):
            manifest_ok = 0
    gl.check("reports_manifest_ok", manifest_ok, "==", "global.bool_true")

    # 5. Clean clone CI exit code & SHA equality & excluded tests check
    with tempfile.TemporaryDirectory(prefix="gate10_clone_") as td:
        t_path = Path(td)
        subprocess.run(
            ["git", "clone", str(REPO_ROOT), str(t_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        proc_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        head_sha = proc_head.stdout.strip()

        proc_clone_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=t_path,
            capture_output=True,
            text=True,
            check=False,
        )
        clone_sha = proc_clone_head.stdout.strip()

        sha_equals = 1 if (clone_sha == head_sha and head_sha != "") else 0
        gl.check("clean_clone_sha_equals_head", sha_equals, "==", "global.bool_true")

        # Run make ci inside clone pointing to existing virtualenv
        venv_path = REPO_ROOT / ".venv"
        proc_ci = subprocess.run(
            ["make", "ci", f"VENV={venv_path}"],
            cwd=t_path,
            capture_output=True,
            text=True,
            check=False,
        )
        ci_exit_code = proc_ci.returncode
        gl.check("clean_clone_ci_exit_code", ci_exit_code, "==", "global.zero")

        # Verify excluded tests all fail due to missing data in the clean clone
        tests_file = REPO_ROOT / "ci" / "tests_needing_data.txt"
        excluded_tests = [
            line.strip()
            for line in tests_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        excluded_ok = 1 if len(excluded_tests) > 0 else 0
        sub_env = dict(os.environ, PYTHONPATH=f"{t_path / 'src'}:{t_path}")
        for t_id in excluded_tests:
            proc_t = subprocess.run(
                [str(venv_path / "bin" / "pytest"), "-q", "--tb=short", t_id],
                cwd=t_path,
                env=sub_env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc_t.returncode == 0:
                excluded_ok = 0
                break
            out = proc_t.stdout + proc_t.stderr
            if "FileNotFoundError" not in out and "data/" not in out and "data" not in out:
                excluded_ok = 0
                break
        gl.check("excluded_tests_all_need_data", excluded_ok, "==", "global.bool_true")

    # 6. Workflow has required jobs
    wf_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    wf_ok = 1
    if not wf_path.exists():
        wf_ok = 0
    else:
        try:
            with open(wf_path, encoding="utf-8") as f:
                wf_data = yaml.safe_load(f)
            jobs = set(wf_data.get("jobs", {}).keys())
            required_jobs = {"lint-test", "docker-smoke", "readme-check"}
            if not required_jobs.issubset(jobs):
                wf_ok = 0
        except (OSError, yaml.YAMLError, KeyError):
            wf_ok = 0
    gl.check("workflow_has_required_jobs", wf_ok, "==", "global.bool_true")

    # 7. Harness E2E ok
    e2e_ok = 1 if run_harness_e2e() else 0
    gl.check("harness_e2e_ok", e2e_ok, "==", "global.bool_true")

    # 8. README claim matches holdout results
    holdout_path = REPO_ROOT / "reports" / "holdout_results.json"
    claim_matches = 0
    if holdout_path.exists() and README_PATH.exists():
        with open(holdout_path, encoding="utf-8") as f:
            h_data = json.load(f)
        claim_text = h_data.get("claim_challenger_vs_ridge", "")
        readme_text = README_PATH.read_text(encoding="utf-8")
        if claim_text and claim_text in readme_text:
            claim_matches = 1
    gl.check("readme_claim_matches_holdout", claim_matches, "==", "global.bool_true")

    # =========================================================================
    # Protocol Integrity Checks
    # =========================================================================

    # 9. Ledger lines exact
    ledger_path = REPO_ROOT / "reports" / "holdout_ledger.jsonl"
    lines = (
        [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if ledger_path.exists()
        else []
    )
    ledger_exact = (
        1
        if len(lines) == 2
        and lines[0].get("purpose") == "final_eval"
        and lines[1].get("purpose") == "final_eval_recovery"
        else 0
    )
    gl.check("ledger_lines_exact", ledger_exact, "==", "global.bool_true")

    # 10. Holdout recovery touches
    recovery_touches = len(lines) - 1
    gl.check("holdout_recovery_touches", recovery_touches, "<=", "phase5.holdout_recovery_touches_max")

    # 11. Holdout artifacts unchanged since phase 5
    proc_art = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "phase5-accepted",
            "--",
            "reports/holdout_results.json",
            "reports/oof/holdout_*.parquet",
            "models/final_model.joblib",
            "models/feature_pipeline.joblib",
            "models/MANIFEST.json",
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    artifacts_unchanged = 1 if proc_art.returncode == 0 else 0
    gl.check("holdout_artifacts_unchanged_since_phase5", artifacts_unchanged, "==", "global.bool_true")

    # 12. First-touch reproduction recorded
    crash_file = REPO_ROOT / "reports" / "holdout_first_touch_crash.json"
    repro_ok = 0
    if crash_file.exists() and holdout_path.exists():
        with open(crash_file, encoding="utf-8") as f:
            c_data = json.load(f)
        with open(holdout_path, encoding="utf-8") as f:
            h_data = json.load(f)
        first_rmsle = c_data.get("pooled_rmsle", {}).get("final")
        recovery_rmsle = h_data.get("metrics", {}).get("pooled", {}).get("final", {}).get("rmsle")
        if first_rmsle is not None and recovery_rmsle is not None:
            repro_diff = abs(first_rmsle - recovery_rmsle)
            repro_tol = float(gl.thr("phase5.first_touch_rmsle_repro_abs"))
            if repro_diff <= repro_tol:
                repro_ok = 1
    gl.check("first_touch_reproduction_recorded", repro_ok, "==", "global.bool_true")

    # 13. Holdout seal intact
    seal_file = REPO_ROOT / "reports" / "holdout_seal.json"
    proc_seal = subprocess.run(
        ["git", "diff", "--quiet", "phase1-accepted", "--", "reports/holdout_seal.json"],
        cwd=REPO_ROOT,
        check=False,
    )
    seal_intact = 1 if (proc_seal.returncode == 0 and seal_file.exists()) else 0
    gl.check("holdout_seal_intact", seal_intact, "==", "global.bool_true")

    # 14. CI run recorded (pending until reports/ci_run.json exists)
    ci_run_file = REPO_ROOT / "reports" / "ci_run.json"
    ci_data = None
    if ci_run_file.exists():
        try:
            with open(ci_run_file, encoding="utf-8") as f:
                ci_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            ci_data = None

    proc_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    curr_head = proc_rev.stdout.strip()

    status, reason = ci_run_status(ci_data, curr_head, REPO_ROOT)
    if status == "pending":
        gl.pending("ci_run_recorded", reason)
    else:
        gl.check("ci_run_recorded", 1 if status == "PASS" else 0, "==", "global.bool_true")
        if reason:
            gl.info("ci_run_recorded_detail", reason)

    # Informational outputs
    gl.info("time_invariant_gates", ["gate-0", "gate-2", "gate-3", "gate-4", "gate-6", "gate-7", "gate-8", "gate-9"])
    gl.info("excluded_tests_count", len(excluded_tests))
    readme_size = README_PATH.stat().st_size if README_PATH.exists() else 0
    gl.info("readme_size_bytes", readme_size)
    gl.info("holdout_seal_check", "Verified reports/holdout_seal.json against phase1-accepted git tree without reading data/holdout/")


def main() -> None:
    gl.run_gate("10", build_checks)


if __name__ == "__main__":
    main()
