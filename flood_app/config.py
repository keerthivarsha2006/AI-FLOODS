from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DEMO_DATA_PATH = DATA_DIR / "demo_flood_data.csv"

DEFAULT_LOCATION = "Hyderabad"
RISK_THRESHOLDS = {
    "LOW": (0, 25),
    "MEDIUM": (25, 50),
    "HIGH": (50, 75),
    "CRITICAL": (75, 100),
}

RISK_COLORS = {
    "LOW": "#2ecc71",
    "MEDIUM": "#f1c40f",
    "HIGH": "#f39c12",
    "CRITICAL": "#e74c3c",
}

MODEL_FEATURES = [
    "rain_1h",
    "rain_3h",
    "rain_6h",
    "rain_12h",
    "rain_24h",
    "cumulative_rainfall",
    "temperature",
    "humidity",
    "wind_speed",
    "water_level",
    "previous_water_level",
    "rate_of_rise",
    "warning_level",
    "danger_level",
    "elevation_m",
    "forecast_1h",
    "forecast_3h",
    "forecast_6h",
    "lat",
    "lon",
]
