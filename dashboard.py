"""Streamlit dashboard for the team 28 battery-life API and analysis artifacts."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from app.model import API_FEATURE_RANGES, FEATURES, PSRAM_BUDGET_BYTES, SRAM_BUDGET_BYTES
from train import clean_data


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "team_28_train.csv"
METRICS_PATH = ROOT / "model_comparison" / "metrics.csv"
PREDICTIONS_PATH = ROOT / "model_comparison" / "validation_predictions.csv"
CONFIG_PATH = ROOT / "model" / "config.json"
API_DEFAULT = os.getenv("BATTERY_API_URL", "http://127.0.0.1:8000")
PALETTE = ["#55d6be", "#8b7bff", "#ffbd69", "#ff6b8a", "#78a6ff"]

st.set_page_config(
    page_title="BatteryLife AI",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 75% 0%, #14233b 0, #0b1120 38%, #090e19 100%); }
    [data-testid="stSidebar"] { background: #0c1423; border-right: 1px solid #20304a; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
    .hero { padding: 1.5rem 1.7rem; border: 1px solid #263957; border-radius: 20px;
            background: linear-gradient(120deg, #152743, #111b2c 70%); margin-bottom: 1.2rem; }
    .hero h1 { margin: 0; color: #eef4ff; font-size: 2rem; }
    .hero p { color: #9fb0c9; margin: .45rem 0 0; }
    div[data-testid="stMetric"] { background: #121e30; border: 1px solid #263957;
            border-radius: 14px; padding: 1rem 1.1rem; }
    .section-note { color: #9fb0c9; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset():
    raw = pd.read_csv(DATA_PATH)
    cleaned, glitch_counts = clean_data(raw)
    return raw, cleaned, glitch_counts


@st.cache_data(show_spinner=False)
def load_experiment_results():
    if not METRICS_PATH.exists() or not PREDICTIONS_PATH.exists():
        return None, None
    return pd.read_csv(METRICS_PATH), pd.read_csv(PREDICTIONS_PATH)


def api_json(method: str, url: str, **kwargs):
    try:
        response = requests.request(method, url, timeout=15, **kwargs)
    except requests.RequestException as error:
        return None, f"Could not reach the API: {error}"
    try:
        body = response.json()
    except ValueError:
        return None, f"API returned non-JSON content with HTTP {response.status_code}: {response.text[:500]}"
    if not isinstance(body, dict):
        return None, f"API returned an unexpected JSON value with HTTP {response.status_code}"
    if not response.ok:
        return None, f"API returned HTTP {response.status_code}: {body}"
    return body, None


def bytes_label(byte_count: int) -> str:
    if byte_count < 1024:
        return f"{byte_count} B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f} KiB"
    return f"{byte_count / (1024 * 1024):.2f} MiB"


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def draw_prediction_page(api_url: str) -> None:
    hero("Battery Prediction", "Estimate remaining battery life from the latest sensor readings.")
    with st.form("prediction_form"):
        first, second, third = st.columns(3)
        battery = first.number_input(
            "Battery voltage (V)", min_value=API_FEATURE_RANGES["battery_v"][0],
            max_value=API_FEATURE_RANGES["battery_v"][1], value=3.86, step=0.01,
            help="Accepted range observed in the training dataset.",
        )
        temperature = second.number_input(
            "Temperature (°C)", min_value=API_FEATURE_RANGES["temp_c"][0],
            max_value=API_FEATURE_RANGES["temp_c"][1], value=28.0, step=0.5
        )
        rings = third.number_input(
            "Rings per day", min_value=API_FEATURE_RANGES["rings_per_day"][0],
            max_value=API_FEATURE_RANGES["rings_per_day"][1], value=5.0, step=1.0
        )
        fourth, fifth, _ = st.columns(3)
        rssi = fourth.number_input(
            "Wi-Fi RSSI (dBm)", min_value=API_FEATURE_RANGES["wifi_rssi_dbm"][0],
            max_value=API_FEATURE_RANGES["wifi_rssi_dbm"][1], value=-66.0, step=1.0,
            help="More negative readings represent weaker received signal strength.",
        )
        age = fifth.number_input(
            "Device age (months)", min_value=API_FEATURE_RANGES["age_months"][0],
            max_value=API_FEATURE_RANGES["age_months"][1], value=22.0, step=1.0
        )
        submitted = st.form_submit_button("🔋 Predict Battery Life", type="primary", use_container_width=True)

    if not submitted:
        st.info("Enter the current readings and select **Predict Battery Life**.")
        return

    payload = {
        "battery_v": battery,
        "temp_c": temperature,
        "rings_per_day": rings,
        "wifi_rssi_dbm": rssi,
        "age_months": age,
    }
    with st.spinner("Requesting a prediction from the FastAPI model…"):
        result, error = api_json("POST", f"{api_url}/predict", json=payload)
    if error:
        st.error(error)
        return

    days_left = float(result["days_left"])
    timings = result.get("timings_ms", {})
    metric, meter = st.columns([1, 1.35])
    metric.metric("Predicted battery life", f"{days_left:.2f} days")
    meter_plot = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(days_left, 365),
        number={"suffix": " days", "valueformat": ".1f"},
        title={"text": "Estimated life · 365-day visual scale"},
        gauge={
            "axis": {"range": [0, 365]},
            "bar": {"color": PALETTE[0]},
            "bgcolor": "#17243a",
            "bordercolor": "#263957",
            "steps": [
                {"range": [0, 30], "color": "#582a3c"},
                {"range": [30, 120], "color": "#51412c"},
                {"range": [120, 365], "color": "#173d3a"},
            ],
        },
    ))
    meter_plot.update_layout(template="plotly_dark", height=270, margin=dict(l=20, r=20, t=55, b=10))
    with meter:
        st.plotly_chart(meter_plot, use_container_width=True)
    st.caption("The gauge is a visual scale for predicted days, not a calibrated battery-charge percentage.")

    st.subheader("Sensor summary")
    st.dataframe(
        pd.DataFrame([{"Sensor": key, "Reading": value} for key, value in payload.items()]),
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Request timing")
    timing_columns = st.columns(5)
    for column, key in zip(timing_columns, ("parse", "normalise", "inference", "build", "server_total")):
        column.metric(key.replace("_", " ").title(), f"{timings.get(key, 0):.3f} ms")


def draw_model_page(api_url: str) -> None:
    hero("Model Dashboard", "Inspect the deployed network, its measured accuracy, and its parameter-derived footprint.")
    model_info, error = api_json("GET", f"{api_url}/model_info")
    if error:
        st.error(error)
        return
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    metrics, _ = load_experiment_results()
    mlp_metrics = metrics.loc[metrics["model"] == "Neural network"].iloc[0] if metrics is not None else None
    architecture = " → ".join(map(str, [len(FEATURES), *config["hidden"], 1]))
    columns = st.columns(4)
    columns[0].metric("Architecture", architecture)
    columns[1].metric("Parameters", f"{model_info['total_params']:,}")
    columns[2].metric("Weights · FP32", bytes_label(model_info["total_bytes_f32"]))
    columns[3].metric("Validation MAE", f"{config['val_mae']:.2f} days")
    columns = st.columns(4)
    columns[0].metric("INT8 estimate", bytes_label(model_info["total_bytes_int8"]))
    columns[1].metric("MACs / prediction", f"{model_info['total_macs']:,}")
    columns[2].metric("Training MAE", f"{config['train_mae']:.2f} days")
    inference = f"{mlp_metrics['inference_ms_per_sample']:.4f} ms" if mlp_metrics is not None else "not measured"
    columns[3].metric("Single-row prediction path", inference)
    st.caption("INT8 size is a parameter-count estimate. The deployed checkpoint and inference currently use FP32.")
    st.subheader("Per-layer accounting")
    st.dataframe(pd.DataFrame(model_info["layers"]), hide_index=True, use_container_width=True)
    layer_frame = pd.DataFrame(model_info["layers"])
    figure = px.bar(
        layer_frame,
        x="name",
        y="params",
        color="name",
        text="params",
        color_discrete_sequence=PALETTE,
        title="Parameters by linear layer",
    )
    figure.update_layout(template="plotly_dark", showlegend=False, xaxis_title="Layer", yaxis_title="Parameters")
    st.plotly_chart(figure, use_container_width=True)


def draw_comparison_page() -> None:
    hero("Model Comparison", "All candidates use the same cleaned data and fixed train/validation split.")
    metrics, predictions = load_experiment_results()
    if metrics is None:
        st.warning("Run `python evaluate_models.py` to generate comparison metrics.")
        return
    best = metrics.iloc[0]
    deployed = metrics.loc[metrics["model"] == "Neural network"].iloc[0]
    left, right = st.columns(2)
    left.metric("Best validation MAE", f"{best['validation_mae']:.2f} days", best["model"])
    right.metric("Deployed model", "Neural network", f"{deployed['validation_mae']:.2f} days validation MAE")
    st.caption("The deployed network is retained for its small, transparent edge footprint; the best tabular model is not silently substituted.")
    st.dataframe(metrics, hide_index=True, use_container_width=True)

    tab_mae, tab_rmse, tab_r2 = st.tabs(["MAE", "RMSE", "R²"])
    for tab, metric_name, title in (
        (tab_mae, "validation_mae", "Validation MAE (lower is better)"),
        (tab_rmse, "validation_rmse", "Validation RMSE (lower is better)"),
        (tab_r2, "validation_r2", "Validation R² (higher is better)"),
    ):
        with tab:
            figure = px.bar(
                metrics.sort_values(metric_name, ascending=(metric_name != "validation_r2")),
                x="model",
                y=metric_name,
                color="model",
                text_auto=".2f",
                color_discrete_sequence=PALETTE,
                title=title,
            )
            figure.update_layout(template="plotly_dark", showlegend=False, xaxis_title="")
            st.plotly_chart(figure, use_container_width=True)

    model_names = [name for name in predictions.columns if name not in ("row_index", "actual_days_left")]
    selected = st.selectbox("Actual vs predicted / residuals", model_names, index=model_names.index("Neural network"))
    actual = predictions["actual_days_left"]
    predicted = predictions[selected]
    scatter = px.scatter(
        x=actual, y=predicted, opacity=0.55,
        labels={"x": "Actual days left", "y": "Predicted days left"},
        title=f"{selected}: validation predictions",
        color_discrete_sequence=[PALETTE[0]],
    )
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    scatter.add_shape(type="line", x0=lower, y0=lower, x1=upper, y1=upper, line=dict(color="#ffbd69", dash="dash"))
    residuals = predicted - actual
    hist = px.histogram(
        x=residuals, nbins=35,
        labels={"x": "Prediction error (days)", "y": "Validation rows"},
        title=f"{selected}: residual distribution",
        color_discrete_sequence=[PALETTE[1]],
    )
    first, second = st.columns(2)
    first.plotly_chart(scatter, use_container_width=True)
    second.plotly_chart(hist, use_container_width=True)


def draw_dataset_page() -> None:
    hero("Dataset Analytics", "Exploration and cleaning summaries calculated from the supplied team-28 CSV.")
    raw, cleaned, glitch_counts = load_dataset()
    first, second, third, fourth = st.columns(4)
    first.metric("Raw rows", f"{len(raw):,}")
    second.metric("Usable rows", f"{len(cleaned):,}")
    third.metric("Removed as invalid", f"{glitch_counts['total_rows_removed']:,}")
    fourth.metric("Input features", str(len(FEATURES)))
    st.caption("Missing values are checked from the actual CSV; invalid sensor values are removed using physical limits, not sample extrema.")
    summary = pd.DataFrame({
        "column": raw.columns,
        "dtype": [str(raw[column].dtype) for column in raw.columns],
        "missing": [int(raw[column].isna().sum()) for column in raw.columns],
        "unique": [int(raw[column].nunique()) for column in raw.columns],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.subheader("Cleaning counts by rule")
    nonzero_counts = {key: count for key, count in glitch_counts.items() if count and key != "total_rows_removed"}
    st.dataframe(
        pd.DataFrame([{"rule": key, "rows flagged": value} for key, value in nonzero_counts.items()]),
        hide_index=True,
        use_container_width=True,
    )

    distributions, relationships = st.tabs(["Distributions", "Feature relationships"])
    with distributions:
        selected = st.selectbox("Column", raw.columns.tolist(), key="distribution_column")
        figure = px.histogram(
            raw, x=selected, nbins=40, marginal="box",
            color_discrete_sequence=[PALETTE[0]], title=f"Observed distribution: {selected}",
        )
        figure.update_layout(template="plotly_dark")
        st.plotly_chart(figure, use_container_width=True)
        corr = cleaned[FEATURES + ["days_left"]].corr(numeric_only=True)
        heatmap = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Tealrose",
            zmin=-1,
            zmax=1,
            title="Correlation matrix · cleaned rows",
        )
        heatmap.update_layout(template="plotly_dark")
        st.plotly_chart(heatmap, use_container_width=True)
    with relationships:
        selected = st.selectbox("Input feature", FEATURES, key="relationship_feature")
        figure = px.scatter(
            cleaned, x=selected, y="days_left", opacity=0.45,
            color_discrete_sequence=[PALETTE[2]], title=f"days_left vs {selected}",
        )
        figure.update_layout(template="plotly_dark")
        st.plotly_chart(figure, use_container_width=True)
        st.caption("These plots use actual cleaned observations; visible curvature suggests relationships a straight-line model cannot express.")


def draw_edge_page(api_url: str) -> None:
    hero("Edge AI Analysis", "Separate parameter arithmetic from actual quantized deployment measurements.")
    model_info, error = api_json("GET", f"{api_url}/model_info")
    if error:
        st.error(error)
        return
    parameters = model_info["total_params"]
    fp32 = model_info["total_bytes_f32"]
    int8 = model_info["total_bytes_int8"]
    columns = st.columns(4)
    columns[0].metric("Parameters", f"{parameters:,}")
    columns[1].metric("FP32 weights", bytes_label(fp32))
    columns[2].metric("INT8 estimate", bytes_label(int8))
    columns[3].metric("MACs", f"{model_info['total_macs']:,}")
    columns = st.columns(3)
    columns[0].metric("Fits 300 KiB SRAM · FP32", "Yes" if fp32 <= SRAM_BUDGET_BYTES else "No")
    columns[1].metric("Fits 300 KiB SRAM · INT8", "Yes" if int8 <= SRAM_BUDGET_BYTES else "No")
    columns[2].metric("Fits 8 MiB PSRAM · FP32", "Yes" if fp32 <= PSRAM_BUDGET_BYTES else "No")
    st.info(
        "The INT8 figure assumes one byte per weight and bias. It does not include quantization scales, "
        "runtime buffers, or prove that an INT8 model has been calibrated or benchmarked."
    )
    st.markdown(
        """
        **Feasibility:** this network's FP32 parameters require only a few KiB, so its weights are comfortably
        within the stated SRAM budget. A single 32-value FP32 hidden activation is 128 bytes (256 bytes if
        input and output buffers overlap). On-device execution can preserve readings locally and continue without
        Wi-Fi; a server permits easier model updates and centralized monitoring but adds network latency, service
        cost, data-transfer/privacy considerations, and a connectivity dependency. For this small model, local
        inference is technically feasible, subject to firmware and measured power/latency testing.
        """
    )
    st.caption("An ESP32-S3 clock-based estimate is not a measured device latency; see the report for the stated assumption and source.")


def predict_csv(api_url: str, frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["predicted_days_left"] = np.nan
    output["prediction_status"] = "pending"
    output["prediction_error"] = ""
    numeric_rows = []
    row_indices = []

    for index, row in frame.iterrows():
        values = {}
        errors = []
        for feature in FEATURES:
            if feature not in frame.columns:
                errors.append(f"missing column: {feature}")
                continue
            try:
                value = float(row[feature])
            except (TypeError, ValueError):
                errors.append(f"{feature} is not numeric")
                continue
            if not np.isfinite(value):
                errors.append(f"{feature} is missing or non-finite")
                continue
            values[feature] = value
        if errors:
            output.loc[index, "prediction_status"] = "invalid"
            output.loc[index, "prediction_error"] = "; ".join(errors)
        else:
            numeric_rows.append(values)
            row_indices.append(index)

    for start in range(0, len(numeric_rows), 10000):
        indices = row_indices[start:start + 10000]
        rows = numeric_rows[start:start + 10000]
        result, error = api_json("POST", f"{api_url}/predict_batch", json={"rows": rows})
        if error is None:
            for index, prediction in zip(indices, result["days_left"]):
                output.loc[index, "predicted_days_left"] = prediction
                output.loc[index, "prediction_status"] = "predicted"
            continue

        if "HTTP 422" not in error:
            for index in indices:
                output.loc[index, "prediction_status"] = "error"
                output.loc[index, "prediction_error"] = error
            continue

        for index, payload in zip(indices, rows):
            single, single_error = api_json("POST", f"{api_url}/predict", json=payload)
            if single_error:
                output.loc[index, "prediction_status"] = "invalid"
                output.loc[index, "prediction_error"] = single_error
            else:
                output.loc[index, "predicted_days_left"] = single["days_left"]
                output.loc[index, "prediction_status"] = "predicted"
    return output


def draw_batch_page(api_url: str) -> None:
    hero("Batch Prediction", "Upload a CSV with the five sensor fields; invalid rows are reported without blocking valid rows.")
    st.caption("Required columns: " + ", ".join(FEATURES))
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(uploaded)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not read the CSV: {error}")
        return
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        st.error("CSV is missing required sensor columns: " + ", ".join(missing))
        return
    st.write(f"Loaded **{len(frame):,}** rows.")
    if st.button("Predict valid rows", type="primary", disabled=frame.empty):
        with st.spinner("Validating and predicting uploaded rows…"):
            result = predict_csv(api_url, frame)
        st.dataframe(result, hide_index=True, use_container_width=True)
        predicted_count = int((result["prediction_status"] == "predicted").sum())
        invalid_count = len(result) - predicted_count
        first, second = st.columns(2)
        first.metric("Predicted", predicted_count)
        second.metric("Rejected rows", invalid_count)
        st.download_button(
            "Download predictions CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="battery_predictions.csv",
            mime="text/csv",
        )


with st.sidebar:
    st.markdown("## 🔋 BatteryLife AI")
    api_url = st.text_input("FastAPI base URL", value=API_DEFAULT).strip().rstrip("/")
    page = st.radio(
        "Navigate",
        [
            "Battery Prediction",
            "Model Dashboard",
            "Model Comparison",
            "Dataset Analytics",
            "Edge AI Analysis",
            "Batch Prediction",
        ],
    )
    health, health_error = api_json("GET", f"{api_url}/health")
    if health_error:
        st.error("API offline")
    else:
        st.success(f"API online · team {health['team']}")
    st.caption("Backend: FastAPI · Models: PyTorch / scikit-learn / XGBoost")

if page == "Battery Prediction":
    draw_prediction_page(api_url)
elif page == "Model Dashboard":
    draw_model_page(api_url)
elif page == "Model Comparison":
    draw_comparison_page()
elif page == "Dataset Analytics":
    draw_dataset_page()
elif page == "Edge AI Analysis":
    draw_edge_page(api_url)
else:
    draw_batch_page(api_url)
