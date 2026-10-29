"""Generate reports/MANIFEST.json with SHA256 hashes of all models/ and reports/ files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
MANIFEST_PATH = REPORTS_DIR / "MANIFEST.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def is_excluded(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if len(parts) >= 2 and parts[0] == "reports":
        # Exclude reports/scratch/
        if parts[1] == "scratch":
            return True
        # Exclude reports/logs/
        if parts[1] == "logs":
            return True
        # Exclude reports/MANIFEST.json
        if parts[1] == "MANIFEST.json":
            return True
        # Exclude reports/gate_*.json
        if parts[1].startswith("gate_") and parts[1].endswith(".json"):
            return True
    return False


def collect_files() -> dict[str, str]:
    files: dict[str, str] = {}

    # Scan models/
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                files[rel] = sha256_file(p)

    # Scan reports/
    if REPORTS_DIR.exists():
        for p in sorted(REPORTS_DIR.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                if not is_excluded(rel):
                    files[rel] = sha256_file(p)

    return dict(sorted(files.items()))


def make_manifest() -> dict[str, str]:
    files = collect_files()
    data = {"files": files}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(files)} files to {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return files


def main() -> None:
    make_manifest()


if __name__ == "__main__":
    main()
