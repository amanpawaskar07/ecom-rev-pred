"""Verify that frozen files from previously accepted phases have not been modified."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def parse_tag_rank(tag: str) -> tuple[int, str]:
    m = re.match(r"^phase(\d+)([a-z]?)-accepted$", tag)
    if not m:
        return (-1, "")
    return (int(m.group(1)), m.group(2))


def main() -> None:
    repo_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()

    proc = subprocess.run(
        ["git", "tag", "--list", "phase*-accepted"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    tags = [t.strip() for t in proc.stdout.splitlines() if t.strip()]

    valid_tags = [t for t in tags if parse_tag_rank(t)[0] >= 0]
    if not valid_tags:
        print("NOTICE: No phase*-accepted tag found; skipping frozen verification.")
        sys.exit(0)

    latest_tag = max(valid_tags, key=parse_tag_rank)

    # Get list of files that existed at latest_tag under gates, tests, config/project.yaml
    proc_files = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            latest_tag,
            "--",
            "gates",
            "tests",
            "config/project.yaml",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc_files.returncode != 0:
        print(f"Error querying git tree at {latest_tag}: {proc_files.stderr}")
        sys.exit(1)

    frozen_files = [f.strip() for f in proc_files.stdout.splitlines() if f.strip()]

    for rel_path in frozen_files:
        local_path = repo_root / rel_path
        if not local_path.exists():
            print(f"Frozen file deleted: {rel_path}")
            sys.exit(1)

        diff_proc = subprocess.run(
            ["git", "diff", "--quiet", latest_tag, "--", rel_path],
            cwd=repo_root,
            check=False,
        )
        if diff_proc.returncode != 0:
            print(f"Frozen file modified: {rel_path}")
            sys.exit(1)

    print(f"Verified {len(frozen_files)} frozen file(s) against {latest_tag}: OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
