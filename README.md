# E-Commerce Monthly Revenue Predictor

An end-to-end, leak-free machine learning system predicting monthly product-level revenue across geographic regions for Brazilian e-commerce, evaluated against strict statistical gates and sealed holdout data.

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

## What did not work

In accordance with protocol rules, more complex options were tested against simpler rivals and rejected when failing pre-registered material effect thresholds:

- Stacking versus Ridge: A temporal stacking ensemble (Ridge and Random Forest bases combined with an XGBoost meta-regressor) was ablated against standalone Ridge. On synthetic linear-truth benchmarks, the stacker degraded relative performance by -36.9%. On the sealed holdout, the challenger does not beat the Ridge baseline. Ridge was retained as the default served model.
- Target Transformation: QuantileTransformer regression was ablated against standard `log1p` transformation. The QuantileTransformer degraded performance with relative loss of -25.2% across validation folds (ninety-five percent confidence interval [-53.0%, -46.4%]), failing across all folds.
- Imbalance Resampling: Synthetic minority oversampling for regression (SMOTER) was tested against standard weighting. SMOTER provided negligible benefit on tail error while increasing bulk variance, leading to its rejection.
- Engineered Feature Controls: Three candidate features were ablated:
  - `price_bucket`: Relative gain 0.0%, lower bound below zero; dropped.
  - `high_value_flag`: Relative gain 0.0%, lower bound below zero; dropped.
  - `cat_region_level`: Relative gain 0.2%, below required threshold; dropped.

## Explanations

Interpretability is enforced via exact linear SHAP:

- Primary attribution driver: Feature `roll_mean_rev_3` is the dominant signal with mean absolute SHAP value 0.5785.
- Noise probe validation: Random probe features demonstrated minimal attribution, with maximum relative impact of 1.2% compared to the top feature, and zero noise features in the top tier (exactly 0 noise features in top ranks).
- Method cross-check: Spearman rank correlation between SHAP attribution and permutation importance reached 0.9835, with top rank overlap of 3 features.

## API, UI and Docker usage

Run the application locally or via Docker Compose:

```bash
# Start FastAPI service locally
PYTHONPATH=. .venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Start Streamlit UI locally
PYTHONPATH=. .venv/bin/python -m streamlit run ui/app.py --server.port 8501

# Run containerized services
docker compose -p revpred-p9 up -d --build

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
- Ridge is worst on tail MAE.
- Stack lost to Ridge on the linear-truth harness (relative loss -36.9%) and on real data; the stacker is not served.
- SMOTER harness margins were thin; feature tests are underpowered; the causality log is self-reported by the agent.
- The `gates/lib.py` helper uses the first row's tail threshold in the tail MAE lambdas for confidence intervals and folds (a known bug, left in place because gates are frozen).
- The gate statistic is the mean of per-month relative gains, which differs slightly from the pooled gain; the months favoring threshold is `0.67` against `2/3`; bootstrap confidence interval versus point estimate oddity in Phase four info rows; the seed-spread check is vacuous for Ridge; Random Forest depth sits at the bottom edge of its search range.
- The first holdout touch crashed (KeyError) and a human-authorised recovery read followed; both are recorded in the ledger and the recovery reproduced the first result within the pre-registered tolerance.
- Gates one, five-a, and five are point-in-time and fail by design after the holdout touch.
- Docker images are not slimmed (large, API and UI share a base); read-only filesystem is on the API service only.
- Scripts need `PYTHONPATH=.`; CI skips tests that need local data; CI has not yet run on GitHub.

## Protocol integrity

- Holdout seal: Cryptographic `SHA256` checksum recorded prior to model selection, ensuring zero contamination.
- Audit ledger: Sealed holdout data accessed under pre-registered audit logging with immutable ledger records.
- Frozen gates: Automated gate verification scripts ensure thresholds and baseline behaviors cannot be retroactively modified.
