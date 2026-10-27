# AGENTS.md — E-Commerce Revenue Predictor
Rules that survive between sessions. Read fully at the start of every session.

## 1. Scope (fixed; do not add components)
Product-level monthly revenue regression. Row = entity x region x month. Lag-only demand features.
Components, and ONLY these:
- engineered features: price_bucket, high_value_flag, cat_region_level
- imbalanced-regression handling: QuantileTransformer target, SMOTER-style oversampling
- temporal stacking: Ridge + RandomForest bases, XGBoost meta
- expanding-window CV, paired cluster bootstrap
- SHAP, FastAPI, Streamlit, Docker Compose, GitHub Actions CI
Data: raw Olist order-line CSVs only (fallback blocks exist; only the human triggers them).
No pre-aggregated revenue table. Synthetic data lives only in src/revpred/harness.py as a
known-truth test harness.
Allowed deps: pandas numpy scipy scikit-learn xgboost shap pyarrow pyyaml joblib fastapi
uvicorn pydantic httpx streamlit pytest ruff. Anything else -> STOP-HUMAN.

## 2. One phase per session
Do the active phase only. Never read, scaffold or plan a later phase. Never "quickly also" fix
an earlier phase; report it instead.

## 3. Evidence rules
- Every metric you state is pasted from a command you actually ran in this session.
- Forbidden in reports: "should", "likely passes", "approximately", "around", "successfully"
  without a pasted output right above it.
- Cannot run something (Docker missing, no network, data absent)? Write "NOT RUN", set STATUS
  STOP-HUMAN, explain. Never simulate output, never invent a file's contents.
- Gate output is pasted unedited, from the === GATE N BEGIN === line to the END line.

## 4. Frozen gates
Never modify: gates/thresholds.yaml, gates/thresholds.sha256, gates/lib.py, gate_NN.py of any
earlier phase, tests of any earlier phase, config/project.yaml. Add files only.
Thresholds are referenced by key via gates.lib.check(). Never write a numeric literal as a
threshold. If a threshold seems wrong, STOP-HUMAN and say why; the human edits it.
Gate failed: (a) print the failing check; (b) classify as BUG (fix src/) or TRUE NEGATIVE
(report, keep simpler option); (c) never touch thresholds, harness specs, or test
expectations; (d) after at most 3 fix attempts on the same failing check, STOP-HUMAN.

## 5. Leakage rules (each has a test; each test has a mutant in tests/mutants/)
L1 Fit any state (bucket edges, thresholds, scalers, quantile transform, resampler, imputers)
   on training-fold rows only. Validation rows only call transform/predict.
L2 Lag and rolling features use months strictly < the row's month. Row inclusion and series
   entry depend only on months strictly < the row's month.
L3 Resampling/oversampling happens only inside a training fold, after the split.
L4 Stacking OOF predictions come from base models fit on months strictly earlier than the
   predicted month. No shuffled K-fold anywhere.
L5 Columns prefixed aux_ and the target are never features.
L6 One-step-ahead: a row at month m may use realized revenue of months < m. No refit in holdout.

## 6. Decision rule (gates.lib.decide_pair / decide_among)
Advanced component vs cheaper rival, same folds, same fixed hyperparameters (before Phase 5),
same tuning budget (Phase 5). Advanced is default ONLY if the rival is materially worse:
relative gain >= global.min_effect_rel AND bootstrap CI lower bound > 0 AND
>= global.folds_favoring_frac of folds favor it. Otherwise the cheaper rival is default. The
advanced code may stay behind a config flag set to off. Negative results go into reports/
and the README exactly as measured. A negative result is a successful phase outcome.
Defaults are never chosen by eyeballing a table.
Every ablation saves OOF predictions to reports/oof/<variant>.parquet with columns:
row_id, series_id, month, fold, y, pred, tail_thr. The gate recomputes the decision from
these files with gates.lib, not with src/.

## 7. Holdout
Last holdout_months modeled months are sealed. Only revpred.holdout.load_holdout(purpose)
reads them; it appends a line to reports/holdout_ledger.jsonl. Used once, in Phase 5 Stage B,
by scripts/final_eval.py. Never print, plot, describe or sample holdout rows anywhere else.

## 8. Determinism
seed = thresholds global.seed everywhere. PYTHONHASHSEED=0. n_jobs=1 for XGBoost/RF in
anything that feeds a gate. A gate re-run on the same commit reproduces every number within
global.repro_rel_tol.

## 9. Git
Small commits, final commit "phase-N: <summary>". Never tag. Never force-push. Never rewrite
history. Never commit data/raw, data/processed, data/holdout.

## 10. FINAL REPORT format (print exactly, then STOP)
PHASE: N
STATUS: PASS | FAIL | STOP-HUMAN
COMMIT: <output of git rev-parse HEAD>
COMMANDS RUN: <exact commands in order>
GATE OUTPUT (verbatim, unedited): <block>
FILES: <output of git diff --stat HEAD~1>
NEGATIVE RESULTS: <list, or "none">
DEVIATIONS / NOT RUN: <list, or "none">
STOPPING NOW.

## 11. Layout
config/project.yaml  gates/{thresholds.yaml,lib.py,gate_NN.py}  src/revpred/{config,data,panel,cv,
evaluate,baselines,holdout,features,models,targets,resample,stack,explain,harness,artifacts}.py
api/app.py  ui/app.py  scripts/  tests/  tests/mutants/  reports/  reports/oof/  models/  docker/
