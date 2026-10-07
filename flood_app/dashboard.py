from __future__ import annotations

from typing import Any

import matplotlib
import pandas as pd
import streamlit as st

from .config import DEFAULT_LOCATION, RISK_COLORS
from .data_layer import fetch_weather_snapshot, get_historical_rows, get_location_options
from .ml_pipeline import generate_future_risk, score_observation, train_and_evaluate

matplotlib.use("Agg")


def classify_risk(score: float) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def render_header() -> None:
    st.set_page_config(page_title="Flood Risk Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --bg-deep: #f4f9ff;
            --bg-mid: #dff2ff;
            --surface: rgba(255, 255, 255, 0.85);
            --panel: rgba(255, 255, 255, 0.92);
            --accent: #1f8ae0;
            --accent-strong: #145fb8;
            --text: #123a59;
            --muted: #456b87;
            --good: #29c77f;
            --warning: #f4b942;
            --danger: #e75b5b;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background: linear-gradient(135deg, var(--bg-deep) 0%, var(--bg-mid) 48%, #cfe9ff 100%) !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.72) !important;
            border-right: 1px solid rgba(40, 98, 150, 0.15);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stMetric, .stDataFrame, .stAlert, .stTabs [role="tablist"] {
            background: var(--surface);
            border: 1px solid rgba(31, 138, 224, 0.12);
            border-radius: 14px;
            box-shadow: 0 8px 22px rgba(18, 58, 89, 0.08);
        }
        .stMetric {
            padding: 0.85rem 1rem;
        }
        h1, h2, h3, h4 {
            color: var(--text) !important;
        }
        p, div, span, label {
            color: var(--text) !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 700;
        }
        .stSelectbox, .stTextInput, .stNumberInput {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI Flood Risk Prediction and Early Warning System")
    st.caption("Explainable AI-based flood decision support for weather, rainfall and water-level conditions. This does not replace official government alerts.")


def render_sidebar() -> str:
    st.sidebar.header("Location and controls")
    locations = get_location_options()
    selected = st.sidebar.selectbox("Select a location", locations, index=locations.index(DEFAULT_LOCATION) if DEFAULT_LOCATION in locations else 0)
    st.sidebar.caption("Data quality: live weather if configured; otherwise demo/fallback data is used and clearly marked.")
    return selected


def render_overview(snapshot: dict[str, Any], score: float, level: str, explanation: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Location", snapshot.get("location", "Unknown"))
    with col2:
        st.metric("Rainfall (1h)", f"{snapshot.get('rain_1h', 0.0):.1f} mm")
    with col3:
        st.metric("Water level", f"{snapshot.get('water_level', 0.0):.1f} m")
    with col4:
        st.metric("Risk score", f"{score:.1f}%")

    st.subheader("Current flood risk")
    risk_color = RISK_COLORS.get(level, "#2ecc71")
    st.markdown(
        f"<div style='padding: 1rem; border-radius: 10px; background-color: {risk_color}; color: white; font-size: 1.6rem; font-weight: 700; text-align: center;'>"
        f"{level} - {score:.1f}% Flood Risk</div>",
        unsafe_allow_html=True,
    )

    st.write("Main contributing factors:")
    for factor in explanation.get("factors", []):
        st.write(f"- {factor}")


def render_conditions(snapshot: dict[str, Any]) -> None:
    st.subheader("Current conditions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Temperature", f"{snapshot.get('temperature', 0.0):.1f} °C")
    with c2:
        st.metric("Humidity", f"{snapshot.get('humidity', 0.0):.1f} %")
    with c3:
        st.metric("Wind speed", f"{snapshot.get('wind_speed', 0.0):.1f} km/h")
    with c4:
        st.metric("Weather", snapshot.get("weather_condition", "Unknown"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("1h rainfall", f"{snapshot.get('rain_1h', 0.0):.1f} mm")
    with c2:
        st.metric("6h rainfall", f"{snapshot.get('rain_6h', 0.0):.1f} mm")
    with c3:
        st.metric("24h rainfall", f"{snapshot.get('rain_24h', 0.0):.1f} mm")
    with c4:
        st.metric("Rate of rise", f"{snapshot.get('rate_of_rise', 0.0):.2f} m/hr")

    st.caption(f"Data source: {snapshot.get('source', 'unknown')} | Resolution: {snapshot.get('resolution', 'unknown')}")


def render_future_risk(score: float) -> None:
    st.subheader("Forecasted risk timeline")
    forecast = generate_future_risk(score)
    timeline = pd.DataFrame({
        "Horizon": list(forecast.keys()),
        "Risk Score (%)": list(forecast.values()),
    })
    st.line_chart(timeline.set_index("Horizon"))


def render_historical_analysis(location: str) -> None:
    st.subheader("Historical rainfall and flood context")
    df = get_historical_rows(location)
    if df.empty:
        st.warning("No historical rows available for the selected location.")
        return
    chart_df = df[["rain_24h", "water_level", "flood_risk"]].copy()
    chart_df = chart_df.rename(columns={"rain_24h": "24h rainfall", "water_level": "Water level", "flood_risk": "Flood flag"})
    st.line_chart(chart_df)


def render_metrics() -> None:
    st.subheader("Model validation")
    historical = get_historical_rows()
    eval_result = train_and_evaluate(historical)
    metrics = eval_result["metrics"]
    st.json({
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    })


def render_alert(summary: str, score: float) -> None:
    level = classify_risk(score)
    if level in {"HIGH", "CRITICAL"}:
        st.warning(summary)
    else:
        st.info(summary)


def run_dashboard() -> None:
    render_header()
    location = render_sidebar()
    snapshot = fetch_weather_snapshot(location)

    model_result = train_and_evaluate(get_historical_rows())
    model = model_result["model"]
    score, level, explanation = score_observation({
        "rain_1h": snapshot.get("rain_1h", 0.0),
        "rain_3h": snapshot.get("rain_3h", 0.0),
        "rain_6h": snapshot.get("rain_6h", 0.0),
        "rain_12h": snapshot.get("rain_12h", 0.0),
        "rain_24h": snapshot.get("rain_24h", 0.0),
        "cumulative_rainfall": snapshot.get("cumulative_rainfall", 0.0),
        "temperature": snapshot.get("temperature", 0.0),
        "humidity": snapshot.get("humidity", 0.0),
        "wind_speed": snapshot.get("wind_speed", 0.0),
        "water_level": snapshot.get("water_level", 0.0),
        "previous_water_level": snapshot.get("previous_water_level", 0.0),
        "rate_of_rise": snapshot.get("rate_of_rise", 0.0),
        "warning_level": snapshot.get("warning_level", 0.0),
        "danger_level": snapshot.get("danger_level", 0.0),
        "elevation_m": snapshot.get("elevation_m", 0.0),
        "forecast_1h": snapshot.get("forecast_1h", snapshot.get("rain_1h", 0.0)),
        "forecast_3h": snapshot.get("forecast_3h", snapshot.get("rain_3h", 0.0)),
        "forecast_6h": snapshot.get("forecast_6h", snapshot.get("rain_6h", 0.0)),
        "lat": snapshot.get("lat", 0.0),
        "lon": snapshot.get("lon", 0.0),
    }, model)

    render_overview(snapshot, score, level, explanation)
    render_conditions(snapshot)
    render_future_risk(score)
    render_historical_analysis(location)
    render_metrics()

    alert_summary = (
        f"⚠️ Flood warning for {location}. Current risk is {level} with a score of {score:.1f}%. "
        f"The main drivers are: {', '.join(explanation.get('factors', []))}. "
        "This app is for decision support and does not replace official flood advisories."
    )
    render_alert(alert_summary, score)
