import pandas as pd

from flood_app.data_layer import fetch_weather_snapshot, get_location_options
from flood_app.ml_pipeline import generate_future_risk, train_and_evaluate


def test_location_options_include_demo_locations():
    locations = get_location_options()
    assert "Hyderabad" in locations
    assert "Delhi" in locations


def test_demo_weather_snapshot_is_returned():
    snapshot = fetch_weather_snapshot("Hyderabad")
    assert snapshot["source"] == "demo"
    assert snapshot["location"] == "Hyderabad"
    assert snapshot["rain_1h"] >= 0
    assert snapshot["water_level"] > 0


def test_training_pipeline_produces_metrics():
    df = pd.read_csv("data/demo_flood_data.csv")
    result = train_and_evaluate(df)
    metrics = result["metrics"]
    assert metrics["accuracy"] >= 0.0
    assert metrics["f1"] >= 0.0
    assert metrics["recall"] >= 0.0


def test_future_risk_values_increase():
    future = generate_future_risk(40)
    assert future["now"] == 40
    assert future["+1h"] >= future["now"]
    assert future["+24h"] >= future["+12h"]
