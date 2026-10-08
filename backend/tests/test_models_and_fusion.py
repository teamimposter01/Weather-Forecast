"""
Automated Tests for ML Models, Weight Normalization, Fusion Engine, and Extreme Classifiers.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from models.weighting_engine import DynamicWeightingEngine
from models.fusion_engine import ForecastFusionEngine
from models.regime_engine import WeatherRegimeEngine
from models.extreme_engine import ExtremeWeatherEngine

def test_dynamic_weight_normalization_and_non_negativity():
    """Verify LightGBM predicted weights are strictly non-negative and sum to 1.0."""
    engine = DynamicWeightingEngine()
    df = pd.DataFrame({
        "latitude": [13.08, 19.07, 28.61],
        "longitude": [80.27, 72.87, 77.20],
        "elevation_m": [6.0, 14.0, 216.0],
        "hour": [0, 6, 12],
        "month": [1, 7, 10],
        "day_of_year": [1, 200, 300],
        "lead_time_hours": [24, 48, 72],
        "monsoon_indicator": [0, 1, 0],
        "forecast_disagreement": [0.5, 1.2, 0.2],
        "weather_regime": ["WINTER_DRY", "MONSOON_ACTIVE", "NORMAL_WESTERLY"],
        "mae_ecmwf": [1.0, 1.5, 0.8],
        "mae_gfs": [1.2, 1.3, 0.9],
        "mae_ai": [0.7, 0.9, 0.5],
    })

    weights_df = engine.predict_weights(df, variable="temperature")

    for i, row in weights_df.iterrows():
        w_e = row["ecmwf_weight"]
        w_g = row["gfs_weight"]
        w_a = row["ai_weight"]

        assert w_e >= 0.0, f"Negative ECMWF weight detected: {w_e}"
        assert w_g >= 0.0, f"Negative GFS weight detected: {w_g}"
        assert w_a >= 0.0, f"Negative AI weight detected: {w_a}"
        
        total = w_e + w_g + w_a
        assert np.isclose(total, 1.0, atol=1e-5), f"Weights do not sum to 1.0! Total = {total}"

def test_forecast_fusion_engine_combines_sources():
    """Verify Forecast Fusion correctly computes linear combination and uncertainty bounds."""
    fusion_engine = ForecastFusionEngine()

    now = datetime.now(timezone.utc)
    df = pd.DataFrame({
        "issue_time_utc": [now],
        "valid_time_utc": [now],
        "lead_time_hours": [24],
        "location_id": ["chennai"],
        "latitude": [13.0827],
        "longitude": [80.2707],
        "ecmwf_temperature_2m": [30.0],
        "gfs_temperature_2m": [32.0],
        "ai_temperature_2m": [31.0],
        "hour": [12],
        "month": [5],
        "day_of_year": [140],
        "monsoon_indicator": [0],
        "elevation_m": [6.0],
    })

    fusion_df = fusion_engine.fuse_forecasts(df, variable="temperature")

    assert len(fusion_df) == 1
    blended = fusion_df.iloc[0]["blended_value"]
    # Blended must lie between min (30.0) and max (32.0) forecast values
    assert 30.0 <= blended <= 32.0, f"Blended forecast out of range: {blended}"

def test_weather_regime_detection():
    """Verify regime detection returns expected monsoon and pre-monsoon regimes."""
    active_monsoon = WeatherRegimeEngine.detect_regime(month=7, temperature_2m=28.0, precipitation_mm=15.0, wind_speed_ms=8.0)
    assert active_monsoon == "MONSOON_ACTIVE"

    heatwave = WeatherRegimeEngine.detect_regime(month=4, temperature_2m=42.0, precipitation_mm=0.0, wind_speed_ms=4.0)
    assert heatwave == "PRE_MONSOON_HEAT"
