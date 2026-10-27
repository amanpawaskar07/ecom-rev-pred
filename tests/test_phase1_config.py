"""Tests for src/revpred/config.py."""
from types import MappingProxyType

import pytest

from revpred.config import (
    get_project_config,
    get_thresholds,
    load_project_config,
    load_thresholds,
)


def test_load_project_config():
    cfg = load_project_config()
    assert isinstance(cfg, MappingProxyType)
    assert cfg["dataset"] == "olist"
    assert "orders" in cfg["files"]
    with pytest.raises(TypeError):
        cfg["new_key"] = "test"


def test_load_thresholds():
    thr = load_thresholds()
    assert isinstance(thr, MappingProxyType)
    assert "phase1" in thr
    assert thr["phase1"]["min_series"] == 150
    with pytest.raises(TypeError):
        thr["phase1"] = {}


def test_tampered_thresholds(tmp_path):
    tampered_yaml = tmp_path / "thresholds.yaml"
    sha_file = tmp_path / "thresholds.sha256"
    tampered_yaml.write_text("version: 999\n", encoding="utf-8")
    sha_file.write_text("0000000000000000000000000000000000000000000000000000000000000000  thresholds.yaml\n")
    with pytest.raises(PermissionError):
        load_thresholds(tampered_yaml, sha_file)


def test_get_cached_configs():
    c1 = get_project_config()
    c2 = get_project_config()
    assert c1 is c2
    t1 = get_thresholds()
    t2 = get_thresholds()
    assert t1 is t2