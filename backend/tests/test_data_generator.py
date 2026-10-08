"""
Synthetic Test Weather Data Generator.
ISOLATED STRICTLY FOR UNIT TESTING AND OFFLINE CI BENCHMARKS.
NEVER USED IN PRODUCTION PIPELINES.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Tuple
from config.locations import INDIAN_LOCATIONS
from config.settings import DEFAULT_LEAD_TIMES

def generate_synthetic_test_dataset(days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic test weather dataset strictly for offline unit tests.
    """
    np.random.seed(42)
    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    time_stamps = [start_dt + timedelta(hours=i) for i in range(days * 24)]

    obs_list = []
    forecast_list = []

    for loc_id, loc in INDIAN_LOCATIONS.items():
        lat, lon = loc["latitude"], loc["longitude"]
        n_hours = len(time_stamps)

        temp_true = 25.0 + 5.0 * np.sin(np.linspace(0, 4*np.pi, n_hours)) + np.random.normal(0, 0.5, n_hours)
        rain_amount = np.where(np.random.uniform(0, 1, n_hours) < 0.1, np.random.exponential(5.0, n_hours), 0.0)
        wind_speed_true = np.clip(4.0 + np.random.normal(0, 1.0, n_hours), 0.5, 20.0)

        obs_df_loc = pd.DataFrame({
            "timestamp_utc": time_stamps,
            "location_id": loc_id,
            "latitude": lat,
            "longitude": lon,
            "temperature_2m": np.round(temp_true, 2),
            "precipitation_mm": np.round(rain_amount, 2),
            "wind_speed_ms": np.round(wind_speed_true, 2),
            "u_wind_ms": np.round(-wind_speed_true * 0.7, 2),
            "v_wind_ms": np.round(-wind_speed_true * 0.7, 2),
            "relative_humidity_pct": np.full(n_hours, 70.0),
            "surface_pressure_hpa": np.full(n_hours, 1013.0),
            "data_source": "synthetic_test_fixture",
        })
        obs_list.append(obs_df_loc)

        for i in range(0, n_hours - 48, 24):
            issue_ts = time_stamps[i]
            for lead in [24, 48]:
                target_idx = i + lead
                if target_idx >= n_hours:
                    continue
                valid_ts = time_stamps[target_idx]

                forecast_list.append({
                    "issue_time_utc": issue_ts,
                    "valid_time_utc": valid_ts,
                    "lead_time_hours": lead,
                    "location_id": loc_id,
                    "latitude": lat,
                    "longitude": lon,
                    "model_name": "ecmwf",
                    "temperature_2m": np.round(temp_true[target_idx] + 0.2, 2),
                    "precipitation_mm": np.round(rain_amount[target_idx], 2),
                    "wind_speed_ms": np.round(wind_speed_true[target_idx], 2),
                    "u_wind_ms": np.round(-wind_speed_true[target_idx] * 0.7, 2),
                    "v_wind_ms": np.round(-wind_speed_true[target_idx] * 0.7, 2),
                    "relative_humidity_pct": 70.0,
                    "surface_pressure_hpa": 1013.0,
                })
                forecast_list.append({
                    "issue_time_utc": issue_ts,
                    "valid_time_utc": valid_ts,
                    "lead_time_hours": lead,
                    "location_id": loc_id,
                    "latitude": lat,
                    "longitude": lon,
                    "model_name": "gfs",
                    "temperature_2m": np.round(temp_true[target_idx] - 0.3, 2),
                    "precipitation_mm": np.round(rain_amount[target_idx], 2),
                    "wind_speed_ms": np.round(wind_speed_true[target_idx], 2),
                    "u_wind_ms": np.round(-wind_speed_true[target_idx] * 0.7, 2),
                    "v_wind_ms": np.round(-wind_speed_true[target_idx] * 0.7, 2),
                    "relative_humidity_pct": 70.0,
                    "surface_pressure_hpa": 1013.0,
                })

    return pd.concat(obs_list, ignore_index=True), pd.DataFrame(forecast_list)
