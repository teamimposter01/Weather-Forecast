"""
Weights API Routes.
Exposes context-aware dynamic model weights and GeoJSON grid weight map across India.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from fastapi import APIRouter, Query
from api.schemas import WeightMapResponse, GeoJSONFeature, WeightMapFeatureProperties
from config.locations import INDIA_GRID_BOUNDS
from models.weighting_engine import DynamicWeightingEngine
from models.regime_engine import WeatherRegimeEngine
from storage.database import storage

router = APIRouter(prefix="/weights", tags=["Dynamic Weights"])

@router.get("")
def get_weights_summary(variable: str = "temperature", location: Optional[str] = None):
    """Get current dynamic weights breakdown by location."""
    query = "SELECT location_id, variable, lead_time_hours, ecmwf_weight, gfs_weight, ai_weight, weather_regime FROM blended_forecasts WHERE variable = ?"
    params = [variable]
    if location:
        query += " AND location_id = ?"
        params.append(location.lower())

    df = storage.query(query, params)
    if df.empty:
        return []
    
    # Return grouped summary
    summary = []
    for (loc, lead), group in df.groupby(["location_id", "lead_time_hours"]):
        row = group.iloc[0]
        summary.append({
            "location_id": loc,
            "variable": variable,
            "lead_time_hours": int(lead),
            "weather_regime": row["weather_regime"],
            "ecmwf_weight": float(row["ecmwf_weight"]),
            "gfs_weight": float(row["gfs_weight"]),
            "ai_weight": float(row["ai_weight"]),
        })
    return summary

@router.get("/map", response_model=WeightMapResponse)
def get_weights_map(
    variable: str = Query("temperature", description="Variable (temperature, rainfall, wind_speed)"),
    lead_time: int = Query(24, description="Forecast lead time in hours"),
    season: str = Query("SOUTHWEST_MONSOON", description="Season context"),
):
    """
    Generate GeoJSON-compatible dynamic weight map over Indian geographic grid.
    Every map value comes directly from the actual LightGBM dynamic weighting engine.
    """
    engine = DynamicWeightingEngine()
    
    # Create spatial grid points across India
    min_lat, max_lat = INDIA_GRID_BOUNDS["min_lat"], INDIA_GRID_BOUNDS["max_lat"]
    min_lon, max_lon = INDIA_GRID_BOUNDS["min_lon"], INDIA_GRID_BOUNDS["max_lon"]
    step = INDIA_GRID_BOUNDS["step"]

    lats = np.arange(min_lat, max_lat + step, step)
    lons = np.arange(min_lon, max_lon + step, step)

    grid_records = []
    for lat in lats:
        for lon in lons:
            # Estimate elevation & month for grid point
            month = 7 if season == "SOUTHWEST_MONSOON" else (4 if season == "PRE_MONSOON" else 1)
            regime = WeatherRegimeEngine.detect_regime(month=month, temperature_2m=30.0, precipitation_mm=10.0 if season == "SOUTHWEST_MONSOON" else 0.0, wind_speed_ms=5.0)

            grid_records.append({
                "latitude": lat,
                "longitude": lon,
                "elevation_m": 150.0,
                "hour": 12,
                "month": month,
                "day_of_year": 200,
                "lead_time_hours": lead_time,
                "monsoon_indicator": 1 if season == "SOUTHWEST_MONSOON" else 0,
                "weather_regime": regime,
                "forecast_disagreement": 0.5,
                "mae_ecmwf": 1.2,
                "mae_gfs": 1.5,
                "mae_ai": 0.8,
            })

    grid_df = pd.DataFrame(grid_records)
    grid_df = WeatherRegimeEngine.apply_regime_detection(grid_df)

    weights_df = engine.predict_weights(grid_df, variable=variable)

    features = []
    for i, row in grid_df.iterrows():
        w_e = float(weights_df.iloc[i]["ecmwf_weight"])
        w_g = float(weights_df.iloc[i]["gfs_weight"])
        w_a = float(weights_df.iloc[i]["ai_weight"])

        dom = "AI XGBoost" if w_a >= max(w_e, w_g) else ("ECMWF" if w_e >= w_g else "GFS")

        feature = GeoJSONFeature(
            geometry={
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])],
            },
            properties=WeightMapFeatureProperties(
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                ecmwf_weight=round(w_e, 4),
                gfs_weight=round(w_g, 4),
                ai_weight=round(w_a, 4),
                dominant_model=dom,
                weather_regime=str(row["weather_regime"]),
                variable=variable,
                lead_time_hours=lead_time,
            )
        )
        features.append(feature)

    return WeightMapResponse(
        variable=variable,
        season=season,
        lead_time_hours=lead_time,
        features=features,
    )
