"""Configuration loaders for project.yaml and thresholds.yaml."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_CONFIG_PATH = REPO_ROOT / "config" / "project.yaml"
THRESHOLDS_PATH = REPO_ROOT / "gates" / "thresholds.yaml"
SHA_PATH = REPO_ROOT / "gates" / "thresholds.sha256"


def _freeze(obj: Any) -> Any:
    """Recursively convert dicts to MappingProxyType and lists to tuples."""
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def load_project_config(config_path: Path | str | None = None) -> MappingProxyType:
    """Load and return an immutable view of config/project.yaml."""
    path = Path(config_path) if config_path else PROJECT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Project config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _freeze(data)


def load_thresholds(
    thresholds_path: Path | str | None = None,
    sha_path: Path | str | None = None,
) -> MappingProxyType:
    """Load and return an immutable view of gates/thresholds.yaml after checksum verification."""
    t_path = Path(thresholds_path) if thresholds_path else THRESHOLDS_PATH
    s_path = Path(sha_path) if sha_path else SHA_PATH

    if not t_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {t_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"Sha256 file not found: {s_path}")

    expected_sha = s_path.read_text(encoding="utf-8").strip().split()[0].lower()
    actual_sha = hashlib.sha256(t_path.read_bytes()).hexdigest().lower()

    if actual_sha != expected_sha:
        raise PermissionError(
            f"Checksum mismatch for {t_path}: expected {expected_sha}, got {actual_sha}"
        )

    with open(t_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _freeze(data)


_PROJECT_CONFIG: MappingProxyType | None = None
_THRESHOLDS: MappingProxyType | None = None


def get_project_config() -> MappingProxyType:
    """Return cached read-only project configuration."""
    global _PROJECT_CONFIG
    if _PROJECT_CONFIG is None:
        _PROJECT_CONFIG = load_project_config()
    return _PROJECT_CONFIG


def get_thresholds() -> MappingProxyType:
    """Return cached read-only thresholds configuration."""
    global _THRESHOLDS
    if _THRESHOLDS is None:
        _THRESHOLDS = load_thresholds()
    return _THRESHOLDS