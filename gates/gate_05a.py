"""Gate 5A: Hyperparameter tuning verification, configuration freeze, and holdout protection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from gates import lib as gl

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OOF_DIR = REPORTS_DIR / "oof"


def build_checks() -> None:
    # =========================================================================
    # 1. Tuning budget verification
    # =========================================================================
    tuning_res_path = REPORTS_DIR / "tuning_results.json"
    with open(tuning_res_path, encoding="utf-8") as f:
        tuning_results = json.load(f)

    tuning_configs_run_ridge = len(tuning_results.get("ridge", {}).get("configs", []))
    tuning_configs_run_stack = len(tuning_results.get("stack", {}).get("configs", []))

    gl.check(
        "tuning_configs_run_ridge",
        tuning_configs_run_ridge,
        "==",
        "phase5.tuning_budget_configs",
    )
    gl.check(
        "tuning_configs_run_stack",
        tuning_configs_run_stack,
        "==",
        "phase5.tuning_budget_configs",
    )

    # =========================================================================
    # 2. Stability check: seed spread
    # =========================================================================
    final_cfg_path = REPORTS_DIR / "final_config.json"
    with open(final_cfg_path, encoding="utf-8") as f:
        final_cfg = json.load(f)

    seed_spread_rel = float(final_cfg.get("stability", {}).get("seed_spread_rel", float("inf")))
    gl.check("seed_spread_rel", seed_spread_rel, "<=", "phase5.seed_spread_max_rel")

    # =========================================================================
    # 3. Decision rule recompute from tuned OOF files
    # =========================================================================
    baseline_oof = pd.read_parquet(OOF_DIR / "baseline_ridge.parquet")
    tuned_ridge_oof = pd.read_parquet(OOF_DIR / "tuned_ridge.parquet")
    tuned_stack_oof = pd.read_parquet(OOF_DIR / "tuned_stack.parquet")

    df_eval = pd.DataFrame({
        "cluster": baseline_oof["cluster"],
        "fold": baseline_oof["fold"],
        "y": baseline_oof["y"],
        "tail_thr": baseline_oof["tail_thr"],
        "baseline_ridge": baseline_oof["pred"],
        "tuned_ridge": tuned_ridge_oof["pred"],
        "tuned_stack": tuned_stack_oof["pred"],
    })

    recomputed_decision = gl.decide_among(
        df_eval,
        candidates_in_cost_order=["baseline_ridge", "tuned_ridge", "tuned_stack"],
        metric="rmsle",
    )
    decision_recompute_matches = (
        1 if recomputed_decision.selected == final_cfg["final_model"] else 0
    )
    gl.check(
        "decision_recompute_matches",
        decision_recompute_matches,
        "==",
        "global.bool_true",
    )

    # =========================================================================
    # 4. Final configuration SHA & challenger freeze
    # =========================================================================
    sha_rec = final_cfg.get("sha256", "")
    final_config_sha_recorded = 1 if isinstance(sha_rec, str) and len(sha_rec) == 64 else 0
    gl.check(
        "final_config_sha_recorded",
        final_config_sha_recorded,
        "==",
        "global.bool_true",
    )

    challenger_frozen = 1 if final_cfg.get("challenger_frozen") == 1 else 0
    gl.check("challenger_frozen", challenger_frozen, "==", "global.bool_true")

    # =========================================================================
    # 5. Challenger specification completeness
    # =========================================================================
    c_hyp = final_cfg.get("challenger_hyperparameters", {})
    has_bases = "bases" in c_hyp and isinstance(c_hyp["bases"], list) and len(c_hyp["bases"]) >= 2
    has_features = (
        ("features" in c_hyp and isinstance(c_hyp["features"], list))
        or ("features" in final_cfg and isinstance(final_cfg["features"], list))
    )
    has_meta = "meta" in c_hyp or "meta_alpha" in c_hyp
    has_transform = (
        "target_transform" in c_hyp or "transform" in c_hyp or "target_transform" in final_cfg
    )
    has_resampler = "resampler" in c_hyp or "resampler" in final_cfg

    challenger_spec_complete = (
        1 if (has_bases and has_features and has_meta and has_transform and has_resampler) else 0
    )
    gl.check(
        "challenger_spec_complete",
        challenger_spec_complete,
        "==",
        "global.bool_true",
    )

    # =========================================================================
    # 6. Cryptographic checksums match disk
    # =========================================================================
    disk_hashes: dict[str, str] = {}
    for d in ["config", "src"]:
        dp = REPO_ROOT / d
        if not dp.exists():
            continue
        for p in sorted(dp.rglob("*")):
            if (
                p.is_file()
                and not p.name.endswith(".pyc")
                and "__pycache__" not in p.parts
                and not p.name.endswith(".gitkeep")
                and "revpred.egg-info" not in p.parts
            ):
                rel = str(p.relative_to(REPO_ROOT).as_posix())
                disk_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = "".join(f"{k}:{v}\n" for k, v in sorted(disk_hashes.items()))
    recomputed_sha256 = hashlib.sha256(manifest.encode("utf-8")).hexdigest()

    cfg_hashes = {k: v for k, v in disk_hashes.items() if k.startswith("config/")}
    recomputed_cfg_sha = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(cfg_hashes.items())).encode("utf-8")
    ).hexdigest()

    src_hashes = {k: v for k, v in disk_hashes.items() if k.startswith("src/")}
    recomputed_src_sha = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(src_hashes.items())).encode("utf-8")
    ).hexdigest()

    recorded_hashes = final_cfg.get("file_hashes", {})
    final_config_hashes_match_files = (
        1
        if (
            disk_hashes == recorded_hashes
            and recomputed_sha256 == final_cfg.get("sha256")
            and recomputed_cfg_sha == final_cfg.get("sha256_config")
            and recomputed_src_sha == final_cfg.get("sha256_src")
        )
        else 0
    )
    gl.check(
        "final_config_hashes_match_files",
        final_config_hashes_match_files,
        "==",
        "global.bool_true",
    )

    # =========================================================================
    # 7. Holdout protection: ledger and code references
    # =========================================================================
    ledger_path = REPORTS_DIR / "holdout_ledger.jsonl"
    holdout_ledger_lines = (
        len(ledger_path.read_text(encoding="utf-8").strip().splitlines())
        if ledger_path.exists() and ledger_path.read_text(encoding="utf-8").strip()
        else 0
    )
    gl.check("holdout_ledger_lines", holdout_ledger_lines, "==", "global.zero")

    allowed_holdout_files = {
        (REPO_ROOT / "src" / "revpred" / "holdout.py").resolve(),
        (REPO_ROOT / "scripts" / "final_eval.py").resolve(),
        Path(__file__).resolve(),
    }
    code_dirs = ["src", "scripts", "api", "ui", "gates"]
    outside_refs = 0
    for d in code_dirs:
        dp = REPO_ROOT / d
        if not dp.exists():
            continue
        for p in dp.rglob("*.py"):
            if p.resolve() in allowed_holdout_files:
                continue
            text = p.read_text(encoding="utf-8")
            if "load_holdout" in text:
                outside_refs += 1

    gl.check(
        "load_holdout_references_outside_allowed",
        outside_refs,
        "==",
        "global.zero",
    )

    # Informative output
    gl.info("final_model", final_cfg.get("final_model"))
    gl.info("challenger_model", final_cfg.get("challenger_model"))
    gl.info("seed_spread_rel", round(seed_spread_rel, 6))
    gl.info("sha256", final_cfg.get("sha256"))
    gl.info("sha256_config", final_cfg.get("sha256_config"))
    gl.info("sha256_src", final_cfg.get("sha256_src"))


def main() -> None:
    try:
        gl.run_gate("5A", build_checks)
    finally:
        src_rep = REPORTS_DIR / "gate_5A.json"
        if src_rep.exists():
            rep_text = src_rep.read_text(encoding="utf-8")
            (REPORTS_DIR / "gate_05a.json").write_text(rep_text, encoding="utf-8")
            (REPORTS_DIR / "gate_5a.json").write_text(rep_text, encoding="utf-8")


if __name__ == "__main__":
    main()
