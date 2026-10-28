"""Artifact loading, verification, and inference bundle."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from revpred.explain import explain_row


class ArtifactTampered(Exception):
    """Raised when an artifact checksum does not match manifest or a manifest/artifact is missing."""


class Bundle:
    """Inference bundle wrapping served model, background stats, and API metadata."""

    def __init__(
        self,
        model: Any,
        explain_background: dict[str, Any],
        api_meta: dict[str, Any],
        model_sha: str,
    ) -> None:
        self.model = model
        self.explain_background = explain_background
        self.api_meta = api_meta
        self.model_sha = model_sha

    def predict_log(self, df: pd.DataFrame | pd.Series | dict[str, Any]) -> np.ndarray:
        """Predict log1p revenue for one or multiple rows."""
        if isinstance(df, dict):
            df_in = pd.DataFrame([df])
        elif isinstance(df, pd.Series):
            df_in = pd.DataFrame([df.to_dict()])
        else:
            df_in = df
        return np.asarray(self.model.predict(df_in), dtype=float)

    def predict_revenue(self, df: pd.DataFrame | pd.Series | dict[str, Any]) -> np.ndarray:
        """Predict revenue in BRL: clip(expm1(predict_log), 0)."""
        pred_log = self.predict_log(df)
        return np.clip(np.expm1(pred_log), 0.0, None)

    def explain(
        self,
        payload: pd.DataFrame | pd.Series | dict[str, Any],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Explain a single payload row using SHAP attributions."""
        if isinstance(payload, pd.DataFrame):
            payload_dict = payload.iloc[0].to_dict()
        elif isinstance(payload, pd.Series):
            payload_dict = payload.to_dict()
        else:
            payload_dict = dict(payload)

        res = explain_row(self.model, payload_dict, self.explain_background, top_k=top_k)

        base_value = float(res["base_value"])
        prediction_log1p = float(res["prediction_log"])
        top_features = res["top_features"]

        contributions: list[dict[str, Any]] = []
        for feat, shap_val in top_features:
            contributions.append({
                "feature": feat,
                "value": payload_dict.get(feat),
                "shap": float(shap_val),
            })

        top_feature_names = {c["feature"] for c in contributions}
        attributions = res["attributions"]
        other_shap = float(sum(val for f, val in attributions.items() if f not in top_feature_names))

        return {
            "base_value": base_value,
            "prediction_log1p": prediction_log1p,
            "contributions": contributions,
            "other_shap": other_shap,
        }


def load_bundle(models_dir: str | Path | None = None) -> Bundle:
    """Verify SHA256 of all files listed in manifests and return Bundle."""
    if models_dir is None:
        models_dir = os.environ.get("MODELS_DIR", "models")
    dir_path = Path(models_dir)

    manifest_names = ("MANIFEST.json", "EXPLAIN_MANIFEST.json", "API_MANIFEST.json")
    for m_name in manifest_names:
        m_file = dir_path / m_name
        if not m_file.exists():
            raise ArtifactTampered(f"Missing manifest file: {m_file}")

        try:
            with open(m_file, encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as exc:
            raise ArtifactTampered(f"Corrupt manifest file {m_file}: {exc}") from exc

        files_map = manifest_data.get("files", {})
        if not files_map:
            raise ArtifactTampered(f"No files listed in manifest {m_file}")

        for rel_path, expected_sha in files_map.items():
            target_path = dir_path / rel_path
            if not target_path.exists():
                target_path = dir_path / Path(rel_path).name
            if not target_path.exists():
                raise ArtifactTampered(f"Missing artifact file: {rel_path} in {dir_path}")

            actual_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise ArtifactTampered(
                    f"Checksum mismatch for {target_path}: expected {expected_sha}, got {actual_sha}"
                )

    model_path = dir_path / "final_model.joblib"
    if not model_path.exists():
        raise ArtifactTampered(f"Missing model file: {model_path}")
    model = joblib.load(model_path)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    bg_path = dir_path / "explain_background.json"
    if not bg_path.exists():
        raise ArtifactTampered(f"Missing background stats: {bg_path}")
    with open(bg_path, encoding="utf-8") as f:
        explain_background = json.load(f)

    meta_path = dir_path / "api_meta.json"
    if not meta_path.exists():
        raise ArtifactTampered(f"Missing api meta: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        api_meta = json.load(f)

    return Bundle(
        model=model,
        explain_background=explain_background,
        api_meta=api_meta,
        model_sha=model_sha,
    )
