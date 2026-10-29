"""Static tests for Dockerfile, docker-compose.yml, .dockerignore, and requirements-runtime.txt."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_pins_full_patch_tag() -> None:
    dockerfile_path = REPO_ROOT / "docker" / "Dockerfile"
    assert dockerfile_path.exists(), "docker/Dockerfile not found"

    content = dockerfile_path.read_text(encoding="utf-8")
    m = re.search(r"FROM\s+python:(3\.11\.\d+-slim)", content)
    assert m is not None, "Dockerfile does not pin a full python 3.11.x-slim tag"


def test_dockerfile_user_not_root_in_both_targets() -> None:
    dockerfile_path = REPO_ROOT / "docker" / "Dockerfile"
    lines = dockerfile_path.read_text(encoding="utf-8").splitlines()

    current_target: str | None = None
    target_users: dict[str, list[str]] = {"api": [], "ui": []}

    for line in lines:
        line_clean = line.strip()
        from_match = re.match(r"^FROM\s+\S+\s+AS\s+(\w+)", line_clean, re.IGNORECASE)
        if from_match:
            current_target = from_match.group(1).lower()

        if current_target in target_users:
            user_match = re.match(r"^USER\s+(\S+)", line_clean)
            if user_match:
                target_users[current_target].append(user_match.group(1))

    assert len(target_users["api"]) > 0, "No USER directive found in api target"
    assert len(target_users["ui"]) > 0, "No USER directive found in ui target"

    for target, users in target_users.items():
        for user in users:
            assert user.lower() not in ("root", "0"), f"Target {target} runs as root: {user}"


def test_dockerfile_copies_only_allowed_dirs() -> None:
    dockerfile_path = REPO_ROOT / "docker" / "Dockerfile"
    lines = dockerfile_path.read_text(encoding="utf-8").splitlines()

    allowed_sources = {
        "src/",
        "api/",
        "ui/",
        "models/",
        "requirements-runtime.txt",
    }

    for line in lines:
        line_clean = line.strip()
        copy_match = re.match(r"^COPY\s+(\S+)", line_clean)
        if copy_match:
            src = copy_match.group(1)
            assert src in allowed_sources or any(src.startswith(a) for a in allowed_sources), (
                f"Dockerfile copies disallowed source: {src}"
            )
            assert not src.startswith("data"), f"Dockerfile copies data: {src}"
            assert not src.startswith("tests"), f"Dockerfile copies tests: {src}"
            assert not src.startswith("gates"), f"Dockerfile copies gates: {src}"
            assert not src.startswith(".git"), f"Dockerfile copies .git: {src}"


def test_dockerignore_excludes_required_paths() -> None:
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore not found"

    lines = [
        line.strip()
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    required_entries = ["data/", "tests/", ".git", ".venv", "__pycache__", "*.csv", "*.parquet"]
    for req in required_entries:
        assert req in lines, f".dockerignore missing required entry: {req}"


def test_compose_healthchecks_and_dependency() -> None:
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml not found"

    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = data.get("services", {})

    assert "api" in services, "api service missing from compose"
    assert "ui" in services, "ui service missing from compose"

    assert "healthcheck" in services["api"], "api service missing healthcheck"
    assert "healthcheck" in services["ui"], "ui service missing healthcheck"

    depends_on = services["ui"].get("depends_on", {})
    assert isinstance(depends_on, dict), "ui depends_on must be a mapping specifying condition service_healthy"
    api_dep = depends_on.get("api", {})
    cond = api_dep.get("condition")
    assert cond == "service_healthy", f"ui depends_on api condition is {cond}, expected service_healthy"


def test_requirements_runtime_match_requirements_lock() -> None:
    runtime_path = REPO_ROOT / "requirements-runtime.txt"
    lock_path = REPO_ROOT / "requirements.lock"

    assert runtime_path.exists(), "requirements-runtime.txt not found"
    assert lock_path.exists(), "requirements.lock not found"

    lock_lines = {
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    runtime_lines = [
        line.strip()
        for line in runtime_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert len(runtime_lines) > 0, "requirements-runtime.txt is empty"

    for r_line in runtime_lines:
        assert r_line in lock_lines, (
            f"Package pin {r_line} in requirements-runtime.txt does not match requirements.lock"
        )
