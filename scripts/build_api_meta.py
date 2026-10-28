"""Build API metadata: feature ranges, regions, categories, example payloads, and API manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

import gates.lib as gl

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DEV_PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"


def main() -> None:
    seed = int(gl.thr("global.seed"))

    pipeline_path = MODELS_DIR / "feature_pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Missing feature pipeline at {pipeline_path}")
    pipeline = joblib.load(pipeline_path)

    numeric_features = list(pipeline["features"])
    categorical_levels = pipeline["categorical_levels"]
    regions = list(categorical_levels["region"])
    categories = list(categorical_levels["category"])

    if not DEV_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing dev panel at {DEV_PANEL_PATH}")
    df_dev = pd.read_parquet(DEV_PANEL_PATH)

    feature_ranges: dict[str, dict[str, float]] = {}
    for feat in numeric_features:
        f_min = float(df_dev[feat].min())
        f_max = float(df_dev[feat].max())
        ceiling = float(f_max * 10.0)
        feature_ranges[feat] = {
            "min": f_min,
            "max": f_max,
            "ceiling": ceiling,
        }

    # Sample 3 example payloads from DEV rows seeded with global seed
    sample_df = df_dev.sample(n=3, random_state=seed)
    int_features = {"active_months_before", "months_since_first_sale", "month_of_year"}

    example_payloads: list[dict[str, object]] = []
    for _, row in sample_df.iterrows():
        payload: dict[str, object] = {}
        for feat in numeric_features:
            val = row[feat]
            if feat in int_features:
                payload[feat] = int(val)
            else:
                payload[feat] = float(val)
        payload["region"] = str(row["region"])
        payload["category"] = str(row["category"])
        example_payloads.append(payload)

    api_meta = {
        "regions": regions,
        "categories": categories,
        "feature_ranges": feature_ranges,
        "example_payloads": example_payloads,
        "examples": example_payloads,
    }

    meta_path = MODELS_DIR / "api_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(api_meta, f, indent=2)
    print(f"Wrote API metadata to {meta_path}")

    meta_bytes = meta_path.read_bytes()
    meta_sha = hashlib.sha256(meta_bytes).hexdigest()

    manifest_path = MODELS_DIR / "API_MANIFEST.json"
    manifest_data = {
        "files": {
            "models/api_meta.json": meta_sha,
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Wrote API manifest to {manifest_path} (sha={meta_sha})")


if __name__ == "__main__":
    main()
