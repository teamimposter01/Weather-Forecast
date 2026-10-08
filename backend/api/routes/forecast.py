"""
Forecast API Routes.
Exposes blended multi-model forecasts with individual model outputs (ECMWF, GFS, AI),
context-aware dynamic weights, uncertainty estimates, weather regimes, extreme weather guidance, and data provenance.
Supports on-demand real NWP forecast fetching for arbitrary latitude/longitude coordinates globally.
"""
from datetime import datetime, timezone
import pandas as pd
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from api.schemas import ForecastItem, IndividualForecasts, ModelWeights, ExtremeGuidance, DataProvenance, DataIntegrityResponse
from config.locations import INDIAN_LOCATIONS, get_location_by_id
from pipeline.ingest import RealWeatherDataIngestor
from pipeline.preprocess import DataPreprocessor
from models.base_models import BaseWeatherModels
from models.regime_engine import WeatherRegimeEngine
from models.weighting_engine import DynamicWeightingEngine
from models.fusion_engine import ForecastFusionEngine
from models.extreme_engine import ExtremeWeatherEngine
from storage.database import storage

router = APIRouter(prefix="/forecast", tags=["Forecasts"])

def _format_forecast_row(row, extreme_dict=None, custom_name=None) -> ForecastItem:
    loc_id = str(row["location_id"])
    loc_name = custom_name if custom_name else (INDIAN_LOCATIONS.get(loc_id, {}).get("name", loc_id.title()))

    w_ecmwf = float(row["ecmwf_weight"])
    w_gfs = float(row["gfs_weight"])
    w_ai = float(row["ai_weight"])

    dominant_model = "AI XGBoost" if w_ai >= max(w_ecmwf, w_gfs) else ("ECMWF" if w_ecmwf >= w_gfs else "GFS")
    explanation = (
        f"{dominant_model} received the highest dynamic weight ({max(w_ecmwf, w_gfs, w_ai)*100:.1f}%) "
        f"for location {loc_name} ({row['variable']}) under weather regime {row['weather_regime']} "
        f"at lead time {row['lead_time_hours']}h due to superior historical verification skill and lower recent forecast disagreement."
    )

    ext = extreme_dict or {"heatwave": 0.0, "heavy_rain": 0.0, "high_wind": 0.0}

    now_utc = datetime.now(timezone.utc)
    issue_dt = pd.to_datetime(row["issue_time_utc"]).tz_localize(timezone.utc) if pd.to_datetime(row["issue_time_utc"]).tz is None else pd.to_datetime(row["issue_time_utc"])
    valid_dt = pd.to_datetime(row["valid_time_utc"]).tz_localize(timezone.utc) if pd.to_datetime(row["valid_time_utc"]).tz is None else pd.to_datetime(row["valid_time_utc"])
    
    data_age_sec = (now_utc - issue_dt).total_seconds()
    status_code = "LIVE" if data_age_sec < 86400 else "STALE"

    provenance = DataProvenance(
        data_source="open_meteo_live_forecast",
        last_updated_utc=now_utc,
        forecast_issue_time_utc=issue_dt,
        valid_time_utc=valid_dt,
        lead_time_hours=int(row["lead_time_hours"]),
        data_age_seconds=round(data_age_sec, 1),
        source_status=status_code,
    )

    return ForecastItem(
        location_id=loc_id,
        location_name=loc_name,
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        issue_time_utc=issue_dt,
        valid_time_utc=valid_dt,
        lead_time_hours=int(row["lead_time_hours"]),
        variable=str(row["variable"]),
        individual_forecasts=IndividualForecasts(
            ecmwf=float(row["ecmwf_value"]),
            gfs=float(row["gfs_value"]),
            ai=float(row["ai_value"]),
        ),
        weights=ModelWeights(
            ecmwf=w_ecmwf,
            gfs=w_gfs,
            ai=w_ai,
        ),
        blended_forecast=float(row["blended_value"]),
        uncertainty_std=float(row["uncertainty_std"]),
        weather_regime=str(row["weather_regime"]),
        extreme_guidance=ExtremeGuidance(
            heatwave_prob_pct=round(float(ext.get("heatwave", 0.0)) * 100.0, 1),
            heavy_rainfall_prob_pct=round(float(ext.get("heavy_rain", 0.0)) * 100.0, 1),
            high_wind_prob_pct=round(float(ext.get("high_wind", 0.0)) * 100.0, 1),
        ),
        provenance=provenance,
        explanation=explanation,
        model_version="v1.0.0",
    )

def _process_and_blend_single_location(loc_id: str, lat: float, lon: float):
    """Run preprocessing, base AI model inference, dynamic weighting, and fusion for single location."""
    ingestor = RealWeatherDataIngestor()
    forecast_df = ingestor.fetch_single_location_forecast(lat, lon, loc_id)
    if forecast_df is None or forecast_df.empty:
        return

    obs_df = storage.query("SELECT * FROM observations ORDER BY timestamp_utc")
    processor = DataPreprocessor()
    aligned_df = processor.create_aligned_dataset(obs_df, forecast_df, save_to_db=False)
    aligned_loc = aligned_df[aligned_df["location_id"] == loc_id].copy()
    if aligned_loc.empty:
        aligned_loc = aligned_df

    base_models = BaseWeatherModels()
    ai_preds = base_models.predict(aligned_loc)
    aligned_loc["ai_temperature_2m"] = ai_preds.get("temperature", aligned_loc["ecmwf_temperature_2m"])
    aligned_loc["ai_precipitation_mm"] = ai_preds.get("rainfall", aligned_loc["ecmwf_precipitation_mm"])
    aligned_loc["ai_wind_speed_ms"] = ai_preds.get("wind_speed", aligned_loc["ecmwf_wind_speed_ms"])

    aligned_loc = WeatherRegimeEngine.apply_regime_detection(aligned_loc)
    weighting_engine = DynamicWeightingEngine()
    fusion_engine = ForecastFusionEngine(weighting_engine=weighting_engine)

    b_temp = fusion_engine.fuse_forecasts(aligned_loc, variable="temperature")
    b_rain = fusion_engine.fuse_forecasts(aligned_loc, variable="rainfall")
    b_wind = fusion_engine.fuse_forecasts(aligned_loc, variable="wind_speed")

    all_blended = pd.concat([b_temp, b_rain, b_wind], ignore_index=True)
    try:
        storage.execute("DELETE FROM blended_forecasts WHERE location_id = ?", [loc_id])
    except Exception:
        pass
    storage.save_dataframe(all_blended, "blended_forecasts", mode="append")

    extreme_engine = ExtremeWeatherEngine()
    probs_df = extreme_engine.predict_extreme_probabilities(aligned_loc)
    try:
        storage.execute("DELETE FROM extreme_probabilities WHERE location_id = ?", [loc_id])
    except Exception:
        pass
    storage.save_dataframe(probs_df, "extreme_probabilities", mode="append")

def _get_forecasts(
    location_id: Optional[str] = None,
    variable: Optional[str] = None,
    lead_time: Optional[int] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
):
    # Handle arbitrary coordinates if passed
    if latitude is not None and longitude is not None:
        lat_clean = round(float(latitude), 3)
        lon_clean = round(float(longitude), 3)
        loc_key = f"loc_{abs(int(lat_clean * 1000))}_{abs(int(lon_clean * 1000))}"
        if location_id:
            loc_key = location_id.lower().strip()

        # Check if already present in DB for target variable
        var_filter = variable.lower() if variable else "temperature"
        check_df = storage.query(
            "SELECT COUNT(*) as cnt FROM blended_forecasts WHERE location_id = ? AND variable = ?",
            [loc_key, var_filter]
        )
        if check_df.empty or check_df.iloc[0]["cnt"] == 0:
            _process_and_blend_single_location(loc_key, lat_clean, lon_clean)
        location_id = loc_key

    query = "SELECT * FROM blended_forecasts WHERE 1=1"
    params = []
    if location_id:
        query += " AND location_id = ?"
        params.append(location_id.lower())
    if variable:
        query += " AND variable = ?"
        params.append(variable.lower())
    if lead_time:
        query += " AND lead_time_hours = ?"
        params.append(lead_time)

    query += " ORDER BY issue_time_utc DESC, valid_time_utc ASC, lead_time_hours ASC LIMIT 200"
    df = storage.query(query, params)
    
    if df.empty:
        return []

    ext_df = storage.query("SELECT * FROM extreme_probabilities")
    ext_lookup = {}
    if not ext_df.empty:
        for _, r in ext_df.iterrows():
            key = (str(r["location_id"]), int(r["lead_time_hours"]))
            ext_lookup[key] = {
                "heatwave": float(r["heatwave_prob"]),
                "heavy_rain": float(r["heavy_rainfall_prob"]),
                "high_wind": float(r["high_wind_prob"]),
            }

    results = []
    for _, row in df.iterrows():
        key = (str(row["location_id"]), int(row["lead_time_hours"]))
        ext_dict = ext_lookup.get(key, {"heatwave": 0.0, "heavy_rain": 0.0, "high_wind": 0.0})
        results.append(_format_forecast_row(row, ext_dict, custom_name=location_name))
    return results

@router.get("", response_model=List[ForecastItem])
def get_all_forecasts(
    location: Optional[str] = Query(None, description="Location identifier (e.g. chennai, mumbai, delhi)"),
    variable: Optional[str] = Query(None, description="Variable (temperature, rainfall, wind_speed)"),
    lead_time: Optional[int] = Query(None, description="Lead time in hours (24, 48, 72, etc.)"),
    latitude: Optional[float] = Query(None, description="Latitude coordinate"),
    longitude: Optional[float] = Query(None, description="Longitude coordinate"),
    location_name: Optional[str] = Query(None, description="Display name for custom location"),
):
    """Retrieve blended forecasts across all locations or arbitrary coordinates."""
    return _get_forecasts(location, variable, lead_time, latitude, longitude, location_name)

@router.get("/temperature", response_model=List[ForecastItem])
def get_temperature_forecasts(
    location: Optional[str] = None,
    lead_time: Optional[int] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
):
    """Retrieve temperature forecasts."""
    return _get_forecasts(location, "temperature", lead_time, latitude, longitude, location_name)

@router.get("/rainfall", response_model=List[ForecastItem])
def get_rainfall_forecasts(
    location: Optional[str] = None,
    lead_time: Optional[int] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
):
    """Retrieve rainfall forecasts."""
    return _get_forecasts(location, "rainfall", lead_time, latitude, longitude, location_name)

@router.get("/wind", response_model=List[ForecastItem])
def get_wind_forecasts(
    location: Optional[str] = None,
    lead_time: Optional[int] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
):
    """Retrieve wind speed forecasts."""
    return _get_forecasts(location, "wind_speed", lead_time, latitude, longitude, location_name)
