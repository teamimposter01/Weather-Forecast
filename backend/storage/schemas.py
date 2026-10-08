"""
Data validation schemas and Pydantic models for storage and pipeline data.
"""
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class WeatherRecord(BaseModel):
    timestamp_utc: datetime
    location_id: str
    latitude: float
    longitude: float
    temperature_2m: float                  # Celsius
    precipitation_mm: float                # mm/h or mm accumulation
    wind_speed_ms: float                   # m/s
    u_wind_ms: float                       # m/s
    v_wind_ms: float                       # m/s
    relative_humidity_pct: float            # %
    surface_pressure_hpa: float            # hPa
    data_source: str                       # 'observation', 'era5', 'ecmwf', 'gfs'

class ModelForecastRecord(BaseModel):
    issue_time_utc: datetime
    valid_time_utc: datetime
    lead_time_hours: int
    location_id: str
    latitude: float
    longitude: float
    model_name: str                        # 'ecmwf', 'gfs', 'ai'
    temperature_2m: float
    precipitation_mm: float
    wind_speed_ms: float
    u_wind_ms: Optional[float] = None
    v_wind_ms: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    surface_pressure_hpa: Optional[float] = None

class SkillRecord(BaseModel):
    location_id: str
    variable: str                          # 'temperature', 'rainfall', 'wind_speed'
    model_name: str
    lead_time_hours: int
    season: str
    mae: float
    rmse: float
    bias: float
    sample_count: int
    updated_at_utc: datetime

class DynamicWeightRecord(BaseModel):
    issue_time_utc: datetime
    valid_time_utc: datetime
    lead_time_hours: int
    location_id: str
    variable: str
    ecmwf_weight: float
    gfs_weight: float
    ai_weight: float
    weather_regime: str
    forecast_disagreement: float

class BlendedForecastRecord(BaseModel):
    issue_time_utc: datetime
    valid_time_utc: datetime
    lead_time_hours: int
    location_id: str
    latitude: float
    longitude: float
    variable: str
    ecmwf_value: float
    gfs_value: float
    ai_value: float
    blended_value: float
    ecmwf_weight: float
    gfs_weight: float
    ai_weight: float
    uncertainty_std: float
    weather_regime: str

class ExtremeWeatherRecord(BaseModel):
    issue_time_utc: datetime
    valid_time_utc: datetime
    lead_time_hours: int
    location_id: str
    heatwave_prob: float
    heavy_rainfall_prob: float
    high_wind_prob: float
