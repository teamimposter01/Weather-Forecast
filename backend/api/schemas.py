"""
API Request and Response Pydantic Schemas.
Includes complete Data Provenance and Data Integrity models.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    timestamp_utc: datetime
    version: str
    database_connected: bool
    data_source_status: str

class DataProvenance(BaseModel):
    data_source: str
    last_updated_utc: datetime
    forecast_issue_time_utc: datetime
    valid_time_utc: datetime
    lead_time_hours: int
    data_age_seconds: float
    source_status: str                      # 'LIVE', 'STALE', 'ERROR'

class ModelWeights(BaseModel):
    ecmwf: float
    gfs: float
    ai: float

class IndividualForecasts(BaseModel):
    ecmwf: float
    gfs: float
    ai: float

class ExtremeGuidance(BaseModel):
    heatwave_prob_pct: float
    heavy_rainfall_prob_pct: float
    high_wind_prob_pct: float

class ForecastItem(BaseModel):
    location_id: str
    location_name: str
    latitude: float
    longitude: float
    issue_time_utc: datetime
    valid_time_utc: datetime
    lead_time_hours: int
    variable: str
    individual_forecasts: IndividualForecasts
    weights: ModelWeights
    blended_forecast: float
    uncertainty_std: float
    weather_regime: str
    extreme_guidance: ExtremeGuidance
    provenance: DataProvenance
    explanation: str
    model_version: str

class WeightMapFeatureProperties(BaseModel):
    latitude: float
    longitude: float
    ecmwf_weight: float
    gfs_weight: float
    ai_weight: float
    dominant_model: str
    weather_regime: str
    variable: str
    lead_time_hours: int

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: WeightMapFeatureProperties

class WeightMapResponse(BaseModel):
    type: str = "FeatureCollection"
    variable: str
    season: str
    lead_time_hours: int
    features: List[GeoJSONFeature]

class ModelMetadataResponse(BaseModel):
    model_name: str
    variable: str
    version: str
    training_timestamp_utc: Optional[str] = None
    feature_schema: List[str]
    metrics: Dict[str, Any]

class SkillScoreItem(BaseModel):
    location_id: str
    variable: str
    model_name: str
    lead_time_hours: int
    season: str
    mae: float
    rmse: float
    bias: float
    sample_count: int

class BacktestResponse(BaseModel):
    variable: str
    sample_size: int
    models: Dict[str, Dict[str, float]]
    skill_improvement_pct: Dict[str, float]
    research_conclusion: str

class DataIntegrityResponse(BaseModel):
    timestamp_utc: datetime
    overall_status: str
    tables: Dict[str, Any]
