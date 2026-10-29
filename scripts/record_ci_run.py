"""Record a verified GitHub Actions CI run to reports/ci_run.json."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_RUN_PATH = REPO_ROOT / "reports" / "ci_run.json"
logger = logging.getLogger(__name__)


def record_ci_run(url: str, sha: str | None = None, conclusion: str | None = None) -> dict[str, str]:
    resolved_sha = sha
    resolved_conclusion = conclusion

    # If sha or conclusion not provided, try gh CLI
    if not (resolved_sha and resolved_conclusion) and shutil.which("gh"):
        try:
            proc = subprocess.run(
                ["gh", "run", "view", url, "--json", "headSha,conclusion"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                if not resolved_sha:
                    resolved_sha = data.get("headSha")
                if not resolved_conclusion:
                    resolved_conclusion = data.get("conclusion")
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            logger.debug("gh run view failed: %s", exc)

    if not resolved_sha:
        raise ValueError("Missing --sha and unable to retrieve from `gh run view`.")
    if not resolved_conclusion:
        raise ValueError("Missing --conclusion and unable to retrieve from `gh run view`.")

    record = {
        "url": url,
        "sha": resolved_sha,
        "conclusion": resolved_conclusion,
    }

    CI_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CI_RUN_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"Recorded CI run to {CI_RUN_PATH.relative_to(REPO_ROOT)}: {record}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a completed GitHub Actions CI run.")
    parser.add_argument("--run-url", required=True, help="GitHub Actions run URL.")
    parser.add_argument("--sha", default=None, help="Commit SHA corresponding to the run.")
    parser.add_argument(
        "--conclusion",
        default=None,
        choices=["success", "failure"],
        help="Run conclusion status.",
    )
    args = parser.parse_args()

    record_ci_run(url=args.run_url, sha=args.sha, conclusion=args.conclusion)


if __name__ == "__main__":
    main()
