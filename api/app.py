"""FastAPI application for E-Commerce Revenue Predictor."""
from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Self

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from revpred.artifacts import Bundle, load_bundle

_api_meta_cache: dict[str, Any] | None = None


def get_cached_api_meta() -> dict[str, Any]:
    global _api_meta_cache
    if _api_meta_cache is not None:
        return _api_meta_cache
    meta_env_dir = os.environ.get("MODELS_DIR", "models")
    meta_file = Path(meta_env_dir) / "api_meta.json"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            _api_meta_cache = json.load(f)
            return _api_meta_cache
    return {}


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    lag_rev_1: float = Field(ge=0.0)
    lag_rev_2: float = Field(ge=0.0)
    lag_rev_3: float = Field(ge=0.0)
    lag_lines_1: float = Field(ge=0.0)
    lag_lines_2: float = Field(ge=0.0)
    lag_lines_3: float = Field(ge=0.0)
    roll_mean_rev_3: float = Field(ge=0.0)
    trailing_price_3: float = Field(ge=0.0)
    active_months_before: int = Field(ge=0)
    months_since_first_sale: int = Field(ge=0)
    month_of_year: int = Field(ge=1, le=12)
    region: str
    category: str

    @model_validator(mode="after")
    def validate_meta_rules(self) -> Self:
        meta = get_cached_api_meta()
        if not meta:
            return self

        allowed_regions = meta.get("regions", [])
        if allowed_regions and self.region not in allowed_regions:
            raise ValueError(f"Unknown region '{self.region}'. Must be one of {allowed_regions}")

        allowed_categories = meta.get("categories", [])
        if allowed_categories and self.category not in allowed_categories:
            raise ValueError(f"Unknown category '{self.category}'.")

        feature_ranges = meta.get("feature_ranges", {})
        for feat, limits in feature_ranges.items():
            if hasattr(self, feat):
                val = getattr(self, feat)
                ceiling = limits.get("ceiling")
                if ceiling is not None and val > ceiling:
                    raise ValueError(f"Feature '{feat}' value {val} exceeds 10x dev ceiling of {ceiling}.")

        return self


class ExplainRequest(PredictRequest):
    top_k: int = Field(default=5, ge=1, le=15)


class PredictResponse(BaseModel):
    predicted_revenue_brl: float
    prediction_log1p: float
    model_sha: str
    outside_dev_range: list[str]


class Contribution(BaseModel):
    feature: str
    value: Any
    shap: float


class ExplainResponse(BaseModel):
    base_value: float
    prediction_log1p: float
    contributions: list[Contribution]
    other_shap: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _api_meta_cache
    models_dir = os.environ.get("MODELS_DIR", "models")
    bundle: Bundle = load_bundle(models_dir)
    app.state.bundle = bundle
    _api_meta_cache = bundle.api_meta
    yield


app = FastAPI(
    title="E-Commerce Revenue Predictor API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    def sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        if isinstance(obj, Exception):
            return str(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    return JSONResponse(status_code=422, content={"detail": sanitize(exc.errors())})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    bundle: Bundle = request.app.state.bundle
    return {"status": "ok", "model_sha": bundle.model_sha}


@app.get("/meta")
def meta(request: Request) -> dict[str, Any]:
    bundle: Bundle = request.app.state.bundle
    return bundle.api_meta


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    bundle: Bundle = request.app.state.bundle
    payload_dict = payload.model_dump()

    pred_log_arr = bundle.predict_log(payload_dict)
    pred_log = float(pred_log_arr[0])

    pred_rev_arr = bundle.predict_revenue(payload_dict)
    pred_rev = float(pred_rev_arr[0])

    outside_dev: list[str] = []
    feature_ranges = bundle.api_meta.get("feature_ranges", {})
    for feat, limits in feature_ranges.items():
        if feat in payload_dict:
            val = payload_dict[feat]
            d_min = limits.get("min")
            d_max = limits.get("max")
            if (d_min is not None and val < d_min) or (d_max is not None and val > d_max):
                outside_dev.append(feat)

    return PredictResponse(
        predicted_revenue_brl=pred_rev,
        prediction_log1p=pred_log,
        model_sha=bundle.model_sha,
        outside_dev_range=outside_dev,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: ExplainRequest, request: Request) -> ExplainResponse:
    bundle: Bundle = request.app.state.bundle
    payload_dict = payload.model_dump()
    k = payload_dict.pop("top_k", 5)

    explain_res = bundle.explain(payload_dict, top_k=k)
    return ExplainResponse(**explain_res)
