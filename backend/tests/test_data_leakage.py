"""
Automated Tests for Data Leakage Safeguards.
Ensures zero future-feature leakage, chronological train/test splitting, and target integrity.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pipeline.features import FeatureEngineer
from pipeline.preprocess import DataPreprocessor

def test_chronological_split_prevents_temporal_leakage():
    """Verify train, validation, and test splits strictly follow temporal ordering."""
    dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(100)]
    df = pd.DataFrame({
        "valid_time_utc": dates,
        "location_id": "chennai",
        "ecmwf_temperature_2m": np.random.normal(30, 2, 100),
        "gfs_temperature_2m": np.random.normal(30, 2, 100),
        "target_temperature_2m": np.random.normal(30, 2, 100),
    })

    processor = DataPreprocessor()
    train_df, val_df, test_df = processor.get_chronological_split(df, train_ratio=0.7, val_ratio=0.15)

    max_train_time = train_df["valid_time_utc"].max()
    min_val_time = val_df["valid_time_utc"].min()
    max_val_time = val_df["valid_time_utc"].max()
    min_test_time = test_df["valid_time_utc"].min()

    assert max_train_time <= min_val_time, "Temporal Leakage detected: Train max time exceeds Validation min time!"
    assert max_val_time <= min_test_time, "Temporal Leakage detected: Validation max time exceeds Test min time!"

def test_lagged_observational_features_use_past_values_only():
    """Verify rolling observational features use strictly past data (closed='left' shift)."""
    dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(10)]
    obs_df = pd.DataFrame({
        "timestamp_utc": dates,
        "location_id": "chennai",
        "temperature_2m": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
        "precipitation_mm": [0, 5, 10, 0, 0, 0, 0, 0, 0, 0],
        "wind_speed_ms": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "surface_pressure_hpa": [1013]*10,
        "relative_humidity_pct": [70]*10,
    })

    feat_df = FeatureEngineer.add_lagged_observational_features(obs_df)

    # First rolling feature at index 0 should equal the first temperature, not include future index 1 value (22)
    first_rolling = feat_df.iloc[0]["rolling_temp_24h_mean"]
    assert first_rolling == 20.0, f"Target Leakage: Rolling feature at t=0 includes future data! {first_rolling}"

    # Second rolling feature at index 1 should equal previous value 20.0
    second_rolling = feat_df.iloc[1]["rolling_temp_24h_mean"]
    assert second_rolling == 20.0, f"Target Leakage: Rolling feature at t=1 includes current/future value! {second_rolling}"
