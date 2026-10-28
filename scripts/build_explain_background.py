"""Build explain background statistics and integrity manifest for Phase 6."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn

from revpred.explain import groups_for

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DEV_PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel_dev.parquet"


def build_background() -> None:
    print("Loading model and feature pipeline...")
    model_path = MODELS_DIR / "final_model.joblib"
    pipeline_path = MODELS_DIR / "feature_pipeline.joblib"

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    print(f"Reading dev panel from {DEV_PANEL_PATH}...")
    dev_df = pd.read_parquet(DEV_PANEL_PATH)
    n_rows = len(dev_df)
    print(f"Panel dev rows: {n_rows}")

    print("Computing transformed feature mean...")
    Z_dev = model.preprocessor_.transform(dev_df)
    z_mean = np.mean(Z_dev, axis=0)

    preds_log = model.predict(dev_df)
    base_value = float(np.mean(preds_log))

    group_map = groups_for(model.preprocessor_)
    feature_names = list(dict.fromkeys(group_map))

    background_data = {
        "n_rows": n_rows,
        "base_value": base_value,
        "feature_names": feature_names,
        "group_map": group_map,
        "z_mean": z_mean.tolist(),
        "intercept": float(model.best_estimator_.intercept_),
        "coef": [float(c) for c in model.best_estimator_.coef_],
        "categorical_levels": pipeline.get("categorical_levels", {}),
    }

    bg_path = MODELS_DIR / "explain_background.json"
    with open(bg_path, "w", encoding="utf-8") as f:
        json.dump(background_data, f, indent=2)
    print(f"Wrote {bg_path}")

    # Compute hashes
    def get_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest_data = {
        "files": {
            "models/explain_background.json": get_sha256(bg_path),
            "models/final_model.joblib": get_sha256(model_path),
            "models/feature_pipeline.joblib": get_sha256(pipeline_path),
        },
        "versions": {
            "python": sys.version,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
        },
        "created_at": datetime.now(UTC).isoformat(),
        "description": "Integrity manifest binding explain_background.json to final_model.joblib and feature_pipeline.joblib",
    }

    manifest_path = MODELS_DIR / "EXPLAIN_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    build_background()
