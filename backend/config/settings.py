"""
System Settings and Constants for Hybrid AI-NWP Weather Forecast Blending System.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = BASE_DIR / "models" / "saved"
STORAGE_DIR = BASE_DIR / "storage" / "db"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, MODELS_DIR, STORAGE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database / Storage Configuration
DB_PATH = STORAGE_DIR / "weather_fusion.duckdb"
PARQUET_STORE = PROCESSED_DATA_DIR / "aligned_weather.parquet"

# Forecast Models
FORECAST_MODELS = ["ecmwf", "gfs", "ai"]

# Lead Times (in hours)
DEFAULT_LEAD_TIMES = [6, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168]

# Weather Seasons for India
SEASONS = {
    "WINTER": [12, 1, 2],         # Dec - Feb
    "PRE_MONSOON": [3, 4, 5],     # Mar - May
    "SOUTHWEST_MONSOON": [6, 7, 8, 9], # Jun - Sep
    "POST_MONSOON": [10, 11],     # Oct - Nov
}

# Extreme Weather Thresholds (IMD standard guidelines adapted)
EXTREME_THRESHOLDS = {
    "heatwave_temp_c": 40.0,       # Absolute temperature threshold for heatwave
    "heatwave_anomaly_c": 4.5,     # Departure from normal
    "heavy_rainfall_mm": 64.5,     # IMD Heavy rainfall threshold (>= 64.5 mm/24h)
    "very_heavy_rainfall_mm": 115.6, # IMD Very Heavy rainfall threshold
    "high_wind_speed_ms": 15.0,    # Wind speed > 15 m/s (~54 km/h / 30 knots)
}

# Dynamic Weighting Constraints
WEIGHTING_CONFIG = {
    "min_weight": 0.0,
    "normalize_sum": 1.0,
    "rolling_window_days": 14,     # Rolling window for historical skill calculation
    "decay_factor": 0.95,          # Exponential decay factor for recent error weighting
}

# Feature definitions
WEATHER_VARIABLES = ["temperature", "rainfall", "wind_speed", "u_wind", "v_wind", "relative_humidity", "surface_pressure"]

# Environment Variables / API credentials
API_TITLE = "Hybrid AI-NWP Weather Forecast Blending System for India"
API_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
