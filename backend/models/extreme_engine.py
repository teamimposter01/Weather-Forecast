"""
Extreme Weather Classifiers for Heatwave, Heavy Rainfall, and High Wind detection using XGBoost.
Outputs event probabilities rather than deterministic alarms.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import xgboost as xgb
from config.settings import MODELS_DIR, EXTREME_THRESHOLDS

EXTREME_FEATURES = [
    "latitude",
    "longitude",
    "elevation_m",
    "hour",
    "month",
    "day_of_year",
    "lead_time_hours",
    "monsoon_indicator",
    "ecmwf_temperature_2m",
    "gfs_temperature_2m",
    "ecmwf_precipitation_mm",
    "gfs_precipitation_mm",
    "ecmwf_wind_speed_ms",
    "gfs_wind_speed_ms",
    "temp_ensemble_mean",
    "rain_ensemble_mean",
    "wind_ensemble_mean",
]

class ExtremeWeatherEngine:
    def __init__(self, models_dir: str = str(MODELS_DIR)):
        self.models_dir = models_dir
        self.heatwave_clf = None
        self.heavy_rain_clf = None
        self.high_wind_clf = None

    def _prepare_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derive ground truth binary extreme targets from observed data."""
        t_target = df["target_temperature_2m"].values if "target_temperature_2m" in df.columns else df["ecmwf_temperature_2m"].values
        r_target = df["target_precipitation_mm"].values if "target_precipitation_mm" in df.columns else df["ecmwf_precipitation_mm"].values
        w_target = df["target_wind_speed_ms"].values if "target_wind_speed_ms" in df.columns else df["ecmwf_wind_speed_ms"].values

        heatwave = (t_target >= EXTREME_THRESHOLDS["heatwave_temp_c"]).astype(int)
        heavy_rain = (r_target >= EXTREME_THRESHOLDS["heavy_rainfall_mm"]).astype(int)
        high_wind = (w_target >= EXTREME_THRESHOLDS["high_wind_speed_ms"]).astype(int)

        return heatwave, heavy_rain, high_wind

    def train_extreme_classifiers(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train XGBoost binary classifiers for Heatwave, Heavy Rainfall, and High Wind."""
        print("[ExtremeEngine] Training Extreme Weather Classifiers...")

        X_train = train_df[EXTREME_FEATURES].fillna(0.0)
        hw_train, hr_train, hw_wind_train = self._prepare_targets(train_df)

        X_val = val_df[EXTREME_FEATURES].fillna(0.0)
        hw_val, hr_val, hw_wind_val = self._prepare_targets(val_df)

        # Heatwave model
        scale_hw = (len(hw_train) - sum(hw_train)) / max(sum(hw_train), 1)
        self.heatwave_clf = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5, scale_pos_weight=scale_hw, random_state=42
        )
        self.heatwave_clf.fit(X_train, hw_train, eval_set=[(X_val, hw_val)], verbose=False)

        # Heavy Rain model
        scale_hr = (len(hr_train) - sum(hr_train)) / max(sum(hr_train), 1)
        self.heavy_rain_clf = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5, scale_pos_weight=scale_hr, random_state=42
        )
        self.heavy_rain_clf.fit(X_train, hr_train, eval_set=[(X_val, hr_val)], verbose=False)

        # High Wind model
        scale_wind = (len(hw_wind_train) - sum(hw_wind_train)) / max(sum(hw_wind_train), 1)
        self.high_wind_clf = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5, scale_pos_weight=scale_wind, random_state=42
        )
        self.high_wind_clf.fit(X_train, hw_wind_train, eval_set=[(X_val, hw_wind_val)], verbose=False)

        # Save models
        joblib.dump(self.heatwave_clf, os.path.join(self.models_dir, "heatwave_clf.joblib"))
        joblib.dump(self.heavy_rain_clf, os.path.join(self.models_dir, "heavy_rain_clf.joblib"))
        joblib.dump(self.high_wind_clf, os.path.join(self.models_dir, "high_wind_clf.joblib"))

        print("[ExtremeEngine] Extreme Weather Classifiers successfully trained.")

    def load_saved_models(self):
        """Load trained extreme weather models."""
        hw_p = os.path.join(self.models_dir, "heatwave_clf.joblib")
        hr_p = os.path.join(self.models_dir, "heavy_rain_clf.joblib")
        wind_p = os.path.join(self.models_dir, "high_wind_clf.joblib")

        if os.path.exists(hw_p):
            self.heatwave_clf = joblib.load(hw_p)
        if os.path.exists(hr_p):
            self.heavy_rain_clf = joblib.load(hr_p)
        if os.path.exists(wind_p):
            self.high_wind_clf = joblib.load(wind_p)

    def predict_extreme_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict extreme weather event probabilities."""
        self.load_saved_models()
        X = df[EXTREME_FEATURES].fillna(0.0)
        n = len(X)

        if self.heatwave_clf is not None:
            hw_prob = self.heatwave_clf.predict_proba(X)[:, 1]
        else:
            hw_prob = np.zeros(n)

        if self.heavy_rain_clf is not None:
            hr_prob = self.heavy_rain_clf.predict_proba(X)[:, 1]
        else:
            hr_prob = np.zeros(n)

        if self.high_wind_clf is not None:
            wind_prob = self.high_wind_clf.predict_proba(X)[:, 1]
        else:
            wind_prob = np.zeros(n)

        return pd.DataFrame({
            "issue_time_utc": df["issue_time_utc"],
            "valid_time_utc": df["valid_time_utc"],
            "lead_time_hours": df["lead_time_hours"],
            "location_id": df["location_id"],
            "heatwave_prob": np.round(hw_prob, 3),
            "heavy_rainfall_prob": np.round(hr_prob, 3),
            "high_wind_prob": np.round(wind_prob, 3),
        })
