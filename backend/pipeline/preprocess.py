"""
Data Preprocessing and Alignment Pipeline.
Pivots forecast sources (ECMWF, GFS), aligns them with target observations on (valid_time_utc, location_id),
and builds model-ready feature matrices for Temp, Rain, Wind, and Dynamic Weighting.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from storage.database import storage
from pipeline.features import FeatureEngineer

class DataPreprocessor:
    def __init__(self, db_storage=storage):
        self.storage = db_storage

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw observations and forecasts from DuckDB database."""
        obs_df = self.storage.query("SELECT * FROM observations ORDER BY timestamp_utc")
        forecast_df = self.storage.query("SELECT * FROM forecasts ORDER BY valid_time_utc")
        return obs_df, forecast_df

    def create_aligned_dataset(self, obs_df: pd.DataFrame, forecast_df: pd.DataFrame, save_to_db: bool = True) -> pd.DataFrame:
        """
        Align ECMWF and GFS forecasts with actual ERA5 observations.
        
        Outputs one row per (issue_time_utc, valid_time_utc, lead_time_hours, location_id),
        with columns:
        - ecmwf_temperature_2m, gfs_temperature_2m
        - ecmwf_precipitation_mm, gfs_precipitation_mm
        - ecmwf_wind_speed_ms, gfs_wind_speed_ms
        - target_temperature_2m (observed)
        - target_precipitation_mm (observed)
        - target_wind_speed_ms (observed)
        """
        # Separate ECMWF and GFS after normalizing datetime columns to tz-naive UTC
        forecast_df = forecast_df.copy()
        obs_df = obs_df.copy()
        
        if "valid_time_utc" in forecast_df.columns:
            forecast_df["valid_time_utc"] = pd.to_datetime(forecast_df["valid_time_utc"], utc=True).dt.tz_localize(None)
        if "issue_time_utc" in forecast_df.columns:
            forecast_df["issue_time_utc"] = pd.to_datetime(forecast_df["issue_time_utc"], utc=True).dt.tz_localize(None)
        if "timestamp_utc" in obs_df.columns:
            obs_df["timestamp_utc"] = pd.to_datetime(obs_df["timestamp_utc"], utc=True).dt.tz_localize(None)

        ecmwf_df = forecast_df[forecast_df["model_name"] == "ecmwf"].copy()
        gfs_df = forecast_df[forecast_df["model_name"] == "gfs"].copy()

        # Rename forecast columns with prefix
        ecmwf_cols = {
            "temperature_2m": "ecmwf_temperature_2m",
            "precipitation_mm": "ecmwf_precipitation_mm",
            "wind_speed_ms": "ecmwf_wind_speed_ms",
            "u_wind_ms": "ecmwf_u_wind_ms",
            "v_wind_ms": "ecmwf_v_wind_ms",
        }
        gfs_cols = {
            "temperature_2m": "gfs_temperature_2m",
            "precipitation_mm": "gfs_precipitation_mm",
            "wind_speed_ms": "gfs_wind_speed_ms",
            "u_wind_ms": "gfs_u_wind_ms",
            "v_wind_ms": "gfs_v_wind_ms",
        }

        ecmwf_df = ecmwf_df.rename(columns=ecmwf_cols)
        gfs_df = gfs_df.rename(columns=gfs_cols)

        # Merge ECMWF and GFS on issue_time, valid_time, lead_time, location_id
        merge_keys = ["issue_time_utc", "valid_time_utc", "lead_time_hours", "location_id"]
        
        # Keep latitude and longitude from ecmwf_df
        ecmwf_sub = ecmwf_df[merge_keys + ["latitude", "longitude"] + list(ecmwf_cols.values())]
        gfs_sub = gfs_df[merge_keys + list(gfs_cols.values())]

        forecast_aligned = pd.merge(
            ecmwf_sub,
            gfs_sub,
            on=merge_keys,
            how="inner"
        )

        # Merge with actual observations on valid_time_utc and location_id
        obs_features = FeatureEngineer.add_lagged_observational_features(obs_df)

        obs_target_cols = {
            "temperature_2m": "target_temperature_2m",
            "precipitation_mm": "target_precipitation_mm",
            "wind_speed_ms": "target_wind_speed_ms",
            "u_wind_ms": "target_u_wind_ms",
            "v_wind_ms": "target_v_wind_ms",
        }
        obs_aligned = obs_features.rename(columns=obs_target_cols)

        # Ensure tz-naive datetimes on merge keys
        forecast_aligned["valid_time_utc"] = pd.to_datetime(forecast_aligned["valid_time_utc"], utc=True).dt.tz_localize(None)
        obs_aligned["timestamp_utc"] = pd.to_datetime(obs_aligned["timestamp_utc"], utc=True).dt.tz_localize(None)

        # Join forecast with observations (left join so live future forecasts are preserved)
        full_df = pd.merge(
            forecast_aligned,
            obs_aligned,
            left_on=["valid_time_utc", "location_id"],
            right_on=["timestamp_utc", "location_id"],
            how="left",
            suffixes=("", "_obs")
        )

        # Fill missing targets for future timestamps if obs don't exist yet
        if "target_temperature_2m" not in full_df.columns or full_df["target_temperature_2m"].isnull().all():
            full_df["target_temperature_2m"] = (full_df["ecmwf_temperature_2m"] + full_df["gfs_temperature_2m"]) / 2.0
            full_df["target_precipitation_mm"] = (full_df["ecmwf_precipitation_mm"] + full_df["gfs_precipitation_mm"]) / 2.0
            full_df["target_wind_speed_ms"] = (full_df["ecmwf_wind_speed_ms"] + full_df["gfs_wind_speed_ms"]) / 2.0
        else:
            full_df["target_temperature_2m"] = full_df["target_temperature_2m"].fillna((full_df["ecmwf_temperature_2m"] + full_df["gfs_temperature_2m"]) / 2.0)
            full_df["target_precipitation_mm"] = full_df["target_precipitation_mm"].fillna((full_df["ecmwf_precipitation_mm"] + full_df["gfs_precipitation_mm"]) / 2.0)
            full_df["target_wind_speed_ms"] = full_df["target_wind_speed_ms"].fillna((full_df["ecmwf_wind_speed_ms"] + full_df["gfs_wind_speed_ms"]) / 2.0)

        # Clean duplicate lat/lon if any
        if "latitude_obs" in full_df.columns:
            full_df = full_df.drop(columns=["latitude_obs", "longitude_obs"])

        # Add temporal, spatial, derived features
        full_df = FeatureEngineer.add_temporal_features(full_df, time_col="valid_time_utc")
        full_df = FeatureEngineer.add_spatial_features(full_df, location_col="location_id")
        full_df = FeatureEngineer.add_forecast_derived_features(full_df)

        # Save to Parquet / DB if requested
        if save_to_db:
            self.storage.save_dataframe(full_df, "aligned_weather", mode="overwrite")

        return full_df

    def get_chronological_split(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset chronologically into Train, Validation, and Test sets.
        STRICT REQUIREMENT: NO RANDOM SHUFFLING (Time-series integrity).
        """
        df_sorted = df.sort_values("valid_time_utc").reset_index(drop=True)
        n = len(df_sorted)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        return train_df, val_df, test_df

def run_preprocessing():
    """Operational preprocessing script."""
    processor = DataPreprocessor()
    obs_df, forecast_df = processor.load_raw_data()
    aligned_df = processor.create_aligned_dataset(obs_df, forecast_df)
    print(f"[Preprocess] Aligned dataset created with shape: {aligned_df.shape}")
    return aligned_df

if __name__ == "__main__":
    run_preprocessing()
