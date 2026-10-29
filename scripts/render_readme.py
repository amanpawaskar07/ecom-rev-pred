"""Render README.md from README.md.tmpl by expanding JSON report placeholders."""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TMPL_PATH = REPO_ROOT / "README.md.tmpl"
README_PATH = REPO_ROOT / "README.md"
SCRATCH_DIR = REPO_ROOT / "reports" / "scratch"
EXPANDED_NUMBERS_PATH = SCRATCH_DIR / "expanded_numbers.json"

PLACEHOLDER_REGEX = re.compile(r"\{\{([^#\}]+)#([^\|\}]+)(?:\|([^\|\}]+))?\}\}")


def get_nested(data: Any, path: str) -> Any:
    curr = data
    parts = path.split(".")
    for p in parts:
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
        elif isinstance(curr, (list, tuple)) and p.isdigit():
            curr = curr[int(p)]
        else:
            raise KeyError(f"Key '{p}' not found in structure (full path: '{path}')")
    return curr


def format_value(val: Any, fmt: str | None) -> str:
    if fmt is None or fmt == "str" or fmt == "":
        return str(val)
    if fmt == "f2":
        return f"{float(val):.2f}"
    if fmt == "f4":
        return f"{float(val):.4f}"
    if fmt == "pct1":
        return f"{float(val) * 100:.1f}%"
    if fmt == "int":
        return f"{round(float(val))}"
    raise ValueError(f"Unknown format specifier '{fmt}'")


def extract_numeric_tokens(text: str) -> set[str]:
    """Extract individual numeric tokens from a string."""
    tokens = set()
    for m in re.finditer(r"[-+]?(?:\d*\.\d+|\d+)%?", text):
        tok = m.group(0).strip()
        if tok:
            tokens.add(tok)
            base = tok.lstrip("+-")
            tokens.add(base)
            if tok.endswith("%"):
                tokens.add(tok[:-1])
                tokens.add(base[:-1])
    return tokens


def render() -> tuple[str, set[str]]:
    if not TMPL_PATH.exists():
        raise FileNotFoundError(f"Template not found: {TMPL_PATH}")

    content = TMPL_PATH.read_text(encoding="utf-8")
    expanded_tokens: set[str] = set()

    cache: dict[str, Any] = {}

    def replace_match(match: re.Match[str]) -> str:
        file_rel = match.group(1).strip()
        key_path = match.group(2).strip()
        fmt = match.group(3).strip() if match.group(3) else None

        file_full = REPO_ROOT / file_rel
        if not file_full.exists():
            raise FileNotFoundError(f"Placeholder referenced non-existent file: {file_full}")

        if file_rel not in cache:
            with open(file_full, encoding="utf-8") as f:
                cache[file_rel] = json.load(f)

        data = cache[file_rel]
        val = get_nested(data, key_path)
        formatted_str = format_value(val, fmt)

        # Collect numeric tokens
        expanded_tokens.update(extract_numeric_tokens(formatted_str))

        return formatted_str

    rendered_text = PLACEHOLDER_REGEX.sub(replace_match, content)

    # Check for any unexpanded placeholders
    leftover = PLACEHOLDER_REGEX.search(rendered_text)
    if leftover:
        raise ValueError(f"Unresolved placeholder found: {leftover.group(0)}")

    return rendered_text, expanded_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README.md from README.md.tmpl")
    parser.add_argument("--check", action="store_true", help="Check that README.md matches template output")
    args = parser.parse_args()

    rendered, expanded = render()

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPANDED_NUMBERS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(expanded), f, indent=2)

    if args.check:
        if not README_PATH.exists():
            print("README.md does not exist!", file=sys.stderr)
            sys.exit(1)
        current = README_PATH.read_text(encoding="utf-8")
        if current != rendered:
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile="README.md",
                tofile="rendered",
            )
            print("".join(diff), file=sys.stderr)
            print("ERROR: README.md does not match rendered README.md.tmpl!", file=sys.stderr)
            sys.exit(1)
        print("README.md is up to date.")
    else:
        README_PATH.write_text(rendered, encoding="utf-8")
        print(f"Rendered README.md with {len(expanded)} expanded numeric tokens.")


if __name__ == "__main__":
    main()
