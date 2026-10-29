"""Streamlit UI for E-Commerce Revenue Predictor."""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

FIELD_DISPLAY_ORDER = [
    "region",
    "category",
    "month_of_year",
    "lag_rev_1",
    "lag_rev_2",
    "lag_rev_3",
    "lag_lines_1",
    "lag_lines_2",
    "lag_lines_3",
    "roll_mean_rev_3",
    "trailing_price_3",
    "active_months_before",
    "months_since_first_sale",
]


def make_api_request(
    method: str,
    endpoint: str,
    json_data: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> tuple[Any | None, str | None]:
    """Execute an HTTP request to the API with error handling."""
    url = f"{API_URL}{endpoint}" if endpoint.startswith("/") else f"{API_URL}/{endpoint}"
    try:
        with httpx.Client(timeout=timeout) as client:
            if method.upper() == "GET":
                resp = client.get(url)
            elif method.upper() == "POST":
                resp = client.post(url, json=json_data)
            else:
                return None, f"Unsupported HTTP method: {method}"
    except httpx.TimeoutException:
        return None, f"API timeout connecting to {url}"
    except (httpx.ConnectError, httpx.NetworkError, ConnectionError, OSError) as e:
        return None, f"Could not connect to API at {url}: {e}"
    except httpx.RequestError as e:
        return None, f"HTTP request failed for {url}: {e}"
    except (ValueError, TypeError, KeyError, RuntimeError) as e:
        return None, f"Unexpected error communicating with API: {e}"

    if resp.status_code == 200:
        try:
            return resp.json(), None
        except (ValueError, json.JSONDecodeError) as e:
            return None, f"Failed to parse API response as JSON: {e}"

    if resp.status_code == 422:
        try:
            err_data = resp.json()
            detail = err_data.get("detail", [])
            if isinstance(detail, list):
                msgs = [d.get("msg", str(d)) if isinstance(d, dict) else str(d) for d in detail]
                return None, f"Validation error: {', '.join(msgs)}"
            return None, f"Validation error: {detail}"
        except (ValueError, json.JSONDecodeError):
            return None, f"Validation error: {resp.text}"

    return None, f"API error ({resp.status_code}): {resp.text}"


def main() -> None:
    st.title("E-Commerce Revenue Predictor")

    # Fetch OpenAPI schema and meta
    openapi, openapi_err = make_api_request("GET", "/openapi.json")
    if openapi_err:
        st.error(openapi_err)
        st.stop()

    meta, meta_err = make_api_request("GET", "/meta")
    if meta_err:
        st.error(meta_err)
        st.stop()

    props: dict[str, Any] = (
        openapi.get("components", {})
        .get("schemas", {})
        .get("PredictRequest", {})
        .get("properties", {})
    )

    ordered_fields = [f for f in FIELD_DISPLAY_ORDER if f in props] + [
        f for f in props if f not in FIELD_DISPLAY_ORDER
    ]

    regions: list[str] = meta.get("regions", [])
    categories: list[str] = meta.get("categories", [])
    feature_ranges: dict[str, Any] = meta.get("feature_ranges", {})
    examples: list[dict[str, Any]] = meta.get("examples") or meta.get("example_payloads", [])

    # Initialize session state with the first example or defaults
    first_example = examples[0] if examples else {}
    for field in ordered_fields:
        w_key = f"f_{field}"
        if w_key not in st.session_state:
            schema = props.get(field, {})
            p_type = schema.get("type", "number")
            if field in first_example:
                val = first_example[field]
                st.session_state[w_key] = int(val) if p_type == "integer" else val
            else:
                if p_type == "integer":
                    st.session_state[w_key] = 0
                elif p_type == "number":
                    st.session_state[w_key] = 0.0
                elif field == "region":
                    st.session_state[w_key] = regions[0] if regions else ""
                elif field == "category":
                    st.session_state[w_key] = categories[0] if categories else ""

    # Example selector and load button
    st.subheader("Example Payloads")
    ex_options = list(range(1, len(examples) + 1)) if examples else [1]
    st.selectbox(
        "Choose an example payload",
        options=ex_options,
        format_func=lambda x: f"Example {x}",
        key="sel_example",
    )

    def on_load_example() -> None:
        choice = st.session_state.get("sel_example", 1) - 1
        if 0 <= choice < len(examples):
            ex_data = examples[choice]
            for f_name, f_val in ex_data.items():
                target_key = f"f_{f_name}"
                f_schema = props.get(f_name, {})
                f_type = f_schema.get("type")
                if f_type == "integer":
                    st.session_state[target_key] = int(f_val)
                elif f_type == "number":
                    st.session_state[target_key] = float(f_val)
                else:
                    st.session_state[target_key] = str(f_val)

    st.button("Load example", key="btn_load_example", on_click=on_load_example)

    # Input Form
    st.subheader("Input Features")
    for field in ordered_fields:
        schema = props.get(field, {})
        w_key = f"f_{field}"
        title = schema.get("title", field.replace("_", " ").title())

        if field == "region":
            st.selectbox(
                label=title,
                options=regions,
                key=w_key,
            )
        elif field == "category":
            st.selectbox(
                label=title,
                options=categories,
                key=w_key,
            )
        else:
            p_type = schema.get("type", "number")
            limits = feature_ranges.get(field, {})
            min_hint = limits.get("min")
            max_hint = limits.get("max")
            hint_text = ""
            if min_hint is not None and max_hint is not None:
                hint_text = f"Dev range: [{min_hint}, {max_hint}]"

            if p_type == "integer":
                st.number_input(
                    label=title,
                    step=1,
                    key=w_key,
                    help=hint_text or None,
                )
            else:
                st.number_input(
                    label=title,
                    step=1.0,
                    format="%.2f",
                    key=w_key,
                    help=hint_text or None,
                )

    # Collect payload from current widget states
    payload: dict[str, Any] = {}
    for field in ordered_fields:
        schema = props.get(field, {})
        raw_val = st.session_state.get(f"f_{field}")
        p_type = schema.get("type", "number")
        if p_type == "integer":
            payload[field] = int(raw_val) if raw_val is not None else 0
        elif p_type == "number":
            payload[field] = float(raw_val) if raw_val is not None else 0.0
        else:
            payload[field] = str(raw_val) if raw_val is not None else ""

    col_pred, col_expl = st.columns(2)
    with col_pred:
        predict_clicked = st.button("Predict", key="btn_predict")
    with col_expl:
        explain_clicked = st.button("Explain", key="btn_explain")

    if predict_clicked:
        res, err = make_api_request("POST", "/predict", json_data=payload)
        if err:
            st.error(err)
        else:
            pred_brl = res.get("predicted_revenue_brl", 0.0)
            pred_log = res.get("prediction_log1p", 0.0)
            model_sha = res.get("model_sha", "")
            outside = res.get("outside_dev_range", [])

            formatted_brl = f"BRL {pred_brl:.2f}"
            st.metric("Predicted Revenue", formatted_brl)
            st.write(f"Predicted Revenue: {formatted_brl}")
            st.write(f"Model SHA: {model_sha[:8]}")
            st.write(f"Prediction (log1p): {pred_log:.4f}")

            if outside:
                feats_str = ", ".join(outside)
                st.warning(
                    f"Features outside development range: {feats_str}. "
                    "A linear model extrapolates without limit outside the training range."
                )

    if explain_clicked:
        explain_payload = dict(payload)
        explain_payload["top_k"] = 8
        res, err = make_api_request("POST", "/explain", json_data=explain_payload)
        if err:
            st.error(err)
        else:
            base_val = res.get("base_value", 0.0)
            pred_log = res.get("prediction_log1p", 0.0)
            contribs = res.get("contributions", [])
            other_shap = res.get("other_shap", 0.0)

            st.write(f"base value: {base_val:.4f}")
            st.write(f"prediction (log1p): {pred_log:.4f}")
            st.write(f"other features = {other_shap:.4f}")

            if contribs:
                df_contrib = pd.DataFrame(contribs)
                st.bar_chart(df_contrib, x="shap", y="feature", horizontal=True)

    with st.expander("About and limits"):
        st.write(
            "The caller supplies the lag features. "
            "The model is a linear model on the dev window of the Olist data. "
            "Many product-region-months have zero revenue. "
            "Predictions are not forecasts beyond that window. "
            "Explanations are exact but assume linear effects with no interactions."
        )


if __name__ == "__main__":
    main()
