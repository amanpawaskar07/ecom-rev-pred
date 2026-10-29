# E-Commerce Monthly Revenue Predictor

An end-to-end, leakage-tested machine learning system predicting monthly product-level revenue across geographic regions for Brazilian e-commerce, evaluated against strict statistical gates and sealed holdout data.

## What this is

This system forecasts monthly sales revenue for individual products sold across Brazilian regions. In e-commerce inventory and supply-chain management, aggregate forecasts hide granular product stockouts, while unconstrained models overfit noisy long-tail items. This project solves that problem with a causal, expanding-window panel structure, pre-registered ablation gates, and an auditable evaluation protocol. The final production model is served through a FastAPI REST service, an interactive Streamlit user interface, and containerized Docker Compose deployment.

## Data

The raw data source consists of public order-line transactions from Olist. The forecasting unit is defined at the product level: each observation corresponds to an entity-region-month tuple.

- Grain selection: Candidate grains were evaluated between product identifier and product category.
- Chosen grain: `product_id`.
- Candidate scale: The product grain yielded 5259 distinct time series and 38044 monthly panel observations across 17 modeled months, whereas the category grain produced only 294 series and 3910 observations.
- Causal entry rule: A series enters the active panel strictly after three active historical months, preventing future survival bias.

## Method

- Panel formulation: Pure lag demand features constructed exclusively from past months strictly prior to the prediction month.
- Target transformation: Natural log revenue transformation `log1p(revenue)`.
- Temporal validation: Expanding-window time series cross-validation folds ensuring training data strictly precedes validation periods.
- Base architecture: Full Ridge regression with standardized continuous features and one-hot encoded geographic and category levels.
- Explanation engine: Closed-form exact linear SHAP attribution decomposed into interpretable original feature groupings.

## Results

Final performance evaluated on the sealed three-month holdout window encompassing 14632 observations:

- Final Model (Full Ridge): RMSLE 1.6113, MAE 38.20 BRL, WAPE 0.9744, Tail MAE 1060.05 BRL (evaluating products above the tail threshold of 619.01 BRL).
- Baseline Mean: RMSLE 1.9571, MAE 41.11 BRL, WAPE 1.0487, Tail MAE 1148.25 BRL.
- Baseline Persistence: RMSLE 2.2025, MAE 54.97 BRL, WAPE 1.4021, Tail MAE 732.68 BRL.
- Outperformance vs Mean: Relative error reduction of 17.6% with ninety-five percent cluster bootstrap confidence interval [16.3%, 19.0%].
- Challenger Stacker: RMSLE 1.6069, Tail MAE 948.11 BRL.
- Holdout Verdict: The challenger does not beat the baseline-equivalent Ridge on the sealed holdout.
- Tail MAE Comparison: The persistence baseline has the lowest tail MAE of the four models at 732.68 BRL, and Ridge (1060.05 BRL) is behind it and behind the stacker (948.11 BRL) on tail MAE.

## What did not work

In accordance with protocol rules, more complex options were tested against simpler rivals and rejected when failing pre-registered material effect thresholds:

- Stacking versus Ridge: The stacker is a Ridge-meta temporal stack over a FullRidgeRegressor and a Random Forest base (see reports/final_config.json). On synthetic linear-truth benchmarks, the stacker degraded relative performance with relative gain of -36.9% (negative means worse). On the sealed holdout, the challenger does not beat the Ridge baseline. Ridge was retained as the default served model.
- Target Transformation: QuantileTransformer target versus log1p: Quantile transformation failed to outperform the standard log1p transform, with relative gain of -25.2% (negative means worse) across validation folds (ninety-five percent confidence interval [-53.0%, -46.4%]), with 0 of 8 folds favoring the quantile transform.
- Imbalance Resampling: In cross-validation (`reports/ablation_resample.json`), comparing SMOTER against weighting (`decide_among_tail_mae.pairwise_vs_best.weights`), SMOTER showed a tail MAE relative gain of 3.3% (`decide_among_tail_mae.pairwise_vs_best.weights.relative_gain`), while bulk RMSLE changed from 1.9877 (`rmsle_by_candidate.weights`) to 2.1663 (`rmsle_by_candidate.smoter`), exceeding the bulk non-inferiority relative threshold of 2.0% (`bulk_noninferiority_rel_threshold`). The recorded decision in `reports/imbalance_decision.json` (`chosen_resampler`) selected `weights`. Additionally, on the synthetic benchmark (`reports/harness_imbalance.json`), comparing SMOTER against unweighted baseline, the mean tail MAE gain was 6.6% (`learnable.mean_tail_mae_gain_rel`) alongside a mean bulk RMSLE worsening of 2.0% (`learnable.mean_bulk_rmsle_worsening_rel`).
- Engineered Feature Controls: Three candidate features were ablated:
  - `price_bucket`: Relative gain 0.0%, lower bound below zero; dropped.
  - `high_value_flag`: Relative gain 0.0%, lower bound below zero; dropped.
  - `cat_region_level`: Relative gain 0.2%, below required threshold; dropped.

## Explanations

Interpretability is enforced via exact linear SHAP:

- Primary attribution driver: Feature `roll_mean_rev_3` is the largest attribution driver with mean absolute SHAP value 0.6695, followed by `lag_lines_1` (0.5313) and `trailing_price_3` (0.4763).
- Noise probe validation: In a probe run with added random noise columns, random probe features demonstrated minimal attribution, with maximum relative impact of 1.2% compared to the top feature, and zero noise features in the top group (exactly 0 noise features in top ranks).
- Method cross-check: Spearman rank correlation between SHAP attribution and permutation importance reached 0.9835, with top rank overlap of 3 features.

## API, UI and Docker usage

Run the application locally or via Docker Compose:

```bash
# Start FastAPI service locally
make serve-api
# or: PYTHONPATH=. .venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Start Streamlit UI locally
make serve-ui
# or: PYTHONPATH=. .venv/bin/python -m streamlit run ui/app.py --server.port 8501

# Run containerized services
make docker-up
# or: docker compose -p revpred-p9 up -d --build

# Stop containerized services
make docker-down

# Run docker smoke verification
bash scripts/docker_smoke.sh
```

FastAPI endpoints include `GET /health`, `GET /meta`, `POST /predict`, and `POST /explain`.

## Reproduce

All steps are reproducible under Python 3.11:

```bash
# Verify frozen gates and codebase integrity
make verify-frozen

# Execute test suite safe without data
make ci

# Run end-to-end synthetic harness
PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python scripts/harness_e2e.py

# Verify README template rendering and numeric consistency
make readme
```

## Honest limitations

- Single dataset, short development window, holdout months denser than development months.
- Most product-region-months have zero revenue.
- The caller supplies the lag features; there is no history store.
- Linear model: SHAP is exact but blind to interactions; collinear lag coefficients are unstable.
- Ridge trails behind the persistence baseline and the challenger stacker on tail MAE (at 1060.05 BRL versus 732.68 BRL and 948.11 BRL, respectively), though ahead of the mean baseline (1148.25 BRL).
- On the linear-truth harness the stack lost to ridge with relative gain of -36.9% (negative means worse); on the sealed holdout the stacker's point estimates are slightly lower on RMSLE and tail MAE but it does not beat the baseline-equivalent Ridge under the pre-registered material-gain rule, so it is not served.
- SMOTER harness margins were thin; feature tests are underpowered; the causality log is self-reported by the agent.
- In target ablation (QuantileTransformer target versus log1p), the relative gain point estimate (-25.2%) lies outside its reported ninety-five percent bootstrap confidence interval ([-53.0%, -46.4%]) in reports/ablation_target.json because the bootstrap confidence interval was computed on absolute metric difference while relative gain is scaled by rival metric.
- The `gates/lib.py` helper uses the first row's tail threshold in the tail MAE lambdas for confidence intervals and folds (a known bug, left in place because gates are frozen).
- The gate statistic is the mean of per-month relative gains, which differs slightly from the pooled gain; the months favoring threshold is `0.67` against `2/3`; bootstrap confidence interval versus point estimate oddity in Phase four info rows; the seed-spread check is vacuous for Ridge; Random Forest depth sits at the bottom edge of its search range.
- The first holdout touch crashed (KeyError) and a human-authorised recovery read followed; both are recorded in the ledger and the recovery reproduced the first result within the pre-registered tolerance.
- Gates one, five-a, and five are point-in-time and fail by design after the holdout touch.
- Docker images are not slimmed (large, API and UI share a base); read-only filesystem is on the API service only.
- Scripts need `PYTHONPATH=.`; CI skips tests that need local data; CI has not yet run on GitHub.

## Protocol integrity

- Holdout seal: The seal hash was recorded at the data phase and is verified against the tagged commit without reading the holdout.
- Audit ledger: The holdout was read twice, both reads are recorded in an append-only ledger committed to git (the first run crashed and a human-authorised recovery read reproduced it within the pre-registered tolerance).
- Frozen gates: Changes to gates, tests, and thresholds are detected against the tagged commits by verify-frozen.
