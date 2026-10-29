"""Check that all numeric tokens in README.md come from placeholder expansions or allowlist."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from gates import lib as gl
from scripts.render_readme import EXPANDED_NUMBERS_PATH, render

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"


def get_expanded_tokens() -> set[str]:
    if EXPANDED_NUMBERS_PATH.exists():
        with open(EXPANDED_NUMBERS_PATH, encoding="utf-8") as f:
            tokens = set(json.load(f))
            return tokens
    # Fallback: compute directly from render
    _, tokens = render()
    return tokens


def check_numbers(content: str | None = None) -> tuple[int, list[str]]:
    if content is None:
        if not README_PATH.exists():
            raise FileNotFoundError(f"README.md not found at {README_PATH}")
        content = README_PATH.read_text(encoding="utf-8")

    # 1. Ignore fenced code blocks
    text_clean = re.sub(r"```[\s\S]*?```", " ", content)

    # 2. Ignore inline code spans
    text_clean = re.sub(r"`[^`\n]*`", " ", text_clean)

    # 3. Ignore Markdown links target URLs: [anchor](url) -> keep anchor, drop url
    text_clean = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text_clean)

    # 4. Ignore bare URLs
    text_clean = re.sub(r"https?://\S+", " ", text_clean)

    # 5. Extract all numeric tokens from prose
    # Matches integers, decimals, negative/positive numbers, percentages as distinct tokens
    raw_tokens = re.findall(
        r"(?:^|(?<=[^a-zA-Z0-9_]))[-+]?(?:\d*\.\d+|\d+)%?(?![a-zA-Z0-9_])",
        text_clean,
    )

    # Allowed tokens from thresholds.yaml
    allowlist = set(gl.thr("phase10.readme_numeric_allowlist"))
    expanded = get_expanded_tokens()

    unsourced: list[str] = []

    for tok in raw_tokens:
        tok_clean = tok.strip()
        if not tok_clean:
            continue

        # Check token itself, without percentage, or absolute value
        tok_no_pct = tok_clean.rstrip("%")
        tok_no_sign = tok_clean.lstrip("+-")
        tok_no_sign_pct = tok_no_pct.lstrip("+-")

        is_allowed = (
            tok_clean in allowlist
            or tok_clean in expanded
            or tok_no_pct in allowlist
            or tok_no_pct in expanded
            or tok_no_sign in allowlist
            or tok_no_sign in expanded
            or tok_no_sign_pct in allowlist
            or tok_no_sign_pct in expanded
        )

        if not is_allowed:
            unsourced.append(tok_clean)

    return len(unsourced), unsourced


def main() -> None:
    count, tokens = check_numbers()
    max_unsourced = int(gl.thr("phase10.readme_unsourced_numbers_max"))

    print(f"readme_unsourced_numbers={count}")
    if tokens:
        print(f"unsourced_tokens={tokens}")

    if count > max_unsourced:
        print(
            f"ERROR: Found {count} unsourced numbers in README.md (max allowed: {max_unsourced})",
            file=sys.stderr,
        )
        sys.exit(1)
    print("README numbers check PASSED")


if __name__ == "__main__":
    main()
