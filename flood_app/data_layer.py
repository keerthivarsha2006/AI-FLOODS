from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from .config import DATA_DIR, DEMO_DATA_PATH, DEFAULT_LOCATION

load_dotenv()


def _build_demo_dataset() -> pd.DataFrame:
    if not DEMO_DATA_PATH.exists():
        raise FileNotFoundError(f"Demo dataset not found at {DEMO_DATA_PATH}")
    df = pd.read_csv(DEMO_DATA_PATH)
    df["source"] = "demo"
    df["data_quality"] = "demo/fallback"
    return df


def fetch_weather_snapshot(location: str | None = None) -> dict[str, Any]:
    location = location or DEFAULT_LOCATION
    weather_api_enabled = os.getenv("WEATHER_API_ENABLED", "false").lower() == "true"
    if not weather_api_enabled:
        demo = _build_demo_dataset()
        row = demo[demo["location"].str.lower() == location.lower()].tail(1)
        if row.empty:
            row = demo.tail(1)
        record = row.iloc[0]
        return {
            "location": record["location"],
            "lat": float(record["lat"]),
            "lon": float(record["lon"]),
            "elevation_m": float(record["elevation_m"]),
            "temperature": float(record["temperature"]),
            "humidity": float(record["humidity"]),
            "wind_speed": float(record["wind_speed"]),
            "weather_condition": record["weather_condition"],
            "rain_1h": float(record["rain_1h"]),
            "rain_3h": float(record["rain_3h"]),
            "rain_6h": float(record["rain_6h"]),
            "rain_12h": float(record["rain_12h"]),
            "rain_24h": float(record["rain_24h"]),
            "cumulative_rainfall": float(record["cumulative_rainfall"]),
            "water_level": float(record["water_level"]),
            "previous_water_level": float(record["previous_water_level"]),
            "warning_level": float(record["warning_level"]),
            "danger_level": float(record["danger_level"]),
            "rate_of_rise": float(record["rate_of_rise"]),
            "forecast_1h": float(record["forecast_1h"]),
            "forecast_3h": float(record["forecast_3h"]),
            "forecast_6h": float(record["forecast_6h"]),
            "source": "demo",
            "data_quality": "demo/fallback",
            "resolution": "city-level demo data",
        }

    try:
        base_url = os.getenv("OPEN_METO_API_URL", "https://api.open-meteo.com/v1/forecast")
        geocoding_url = os.getenv("GEOCODING_API_URL", "https://geocoding-api.open-meteo.com/v1/search")
        geo = requests.get(geocoding_url, params={"name": location, "count": 1}, timeout=12)
        geo.raise_for_status()
        geo_json = geo.json()
        if not geo_json.get("results"):
            raise ValueError("No geocoding results found")
        result = geo_json["results"][0]
        lat = float(result["latitude"])
        lon = float(result["longitude"])
        weather = requests.get(
            base_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                "hourly": "precipitation",
                "timezone": "auto",
                "forecast_days": 2,
            },
            timeout=12,
        )
        weather.raise_for_status()
        payload = weather.json()
        current = payload.get("current", {})
        hourly = payload.get("hourly", {})
        precip = hourly.get("precipitation", [])
        total_precip = sum(float(v) for v in precip[:24] if v is not None)
        return {
            "location": location,
            "lat": lat,
            "lon": lon,
            "temperature": float(current.get("temperature_2m", 0.0)),
            "humidity": float(current.get("relative_humidity_2m", 0.0)),
            "wind_speed": float(current.get("wind_speed_10m", 0.0)),
            "weather_condition": "live weather",
            "rain_1h": float(precip[0]) if precip else 0.0,
            "rain_3h": float(sum(precip[:3])) if len(precip) >= 3 else float(sum(precip)),
            "rain_6h": float(sum(precip[:6])) if len(precip) >= 6 else float(sum(precip)),
            "rain_12h": float(sum(precip[:12])) if len(precip) >= 12 else float(sum(precip)),
            "rain_24h": float(sum(precip[:24])) if len(precip) >= 24 else float(sum(precip)),
            "cumulative_rainfall": total_precip,
            "water_level": 0.0,
            "previous_water_level": 0.0,
            "warning_level": 0.0,
            "danger_level": 0.0,
            "rate_of_rise": 0.0,
            "source": "live",
            "data_quality": "live",
            "resolution": "city-level weather data",
        }
    except Exception:
        demo = _build_demo_dataset()
        row = demo[demo["location"].str.lower() == location.lower()].tail(1)
        if row.empty:
            row = demo.tail(1)
        record = row.iloc[0]
        return {
            "location": record["location"],
            "lat": float(record["lat"]),
            "lon": float(record["lon"]),
            "elevation_m": float(record["elevation_m"]),
            "temperature": float(record["temperature"]),
            "humidity": float(record["humidity"]),
            "wind_speed": float(record["wind_speed"]),
            "weather_condition": record["weather_condition"],
            "rain_1h": float(record["rain_1h"]),
            "rain_3h": float(record["rain_3h"]),
            "rain_6h": float(record["rain_6h"]),
            "rain_12h": float(record["rain_12h"]),
            "rain_24h": float(record["rain_24h"]),
            "cumulative_rainfall": float(record["cumulative_rainfall"]),
            "water_level": float(record["water_level"]),
            "previous_water_level": float(record["previous_water_level"]),
            "warning_level": float(record["warning_level"]),
            "danger_level": float(record["danger_level"]),
            "rate_of_rise": float(record["rate_of_rise"]),
            "forecast_1h": float(record["forecast_1h"]),
            "forecast_3h": float(record["forecast_3h"]),
            "forecast_6h": float(record["forecast_6h"]),
            "source": "demo",
            "data_quality": "demo/fallback",
            "resolution": "city-level demo data",
        }


def get_location_options() -> list[str]:
    demo = _build_demo_dataset()
    return sorted(demo["location"].dropna().unique().tolist())


def get_historical_rows(location: str | None = None) -> pd.DataFrame:
    df = _build_demo_dataset()
    if location:
        df = df[df["location"].str.lower() == location.lower()].copy()
    return df.sort_values("location").reset_index(drop=True)
