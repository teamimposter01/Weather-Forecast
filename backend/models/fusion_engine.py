"""
Forecast Fusion Engine.
Fuses ECMWF, GFS, and AI forecasts using context-aware LightGBM dynamic weights.
Calculates blended forecasts and uncertainty bounds.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from models.weighting_engine import DynamicWeightingEngine
from models.regime_engine import WeatherRegimeEngine
from storage.database import storage

class ForecastFusionEngine:
    def __init__(self, weighting_engine: DynamicWeightingEngine = None):
        self.weighting_engine = weighting_engine or DynamicWeightingEngine()

    def fuse_forecasts(
        self,
        df: pd.DataFrame,
        variable: str = "temperature",
        ai_predictions: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Perform dynamic fusion of ECMWF, GFS, and AI forecasts.
        """
        df = df.copy()
        n = len(df)

        # Identify source columns
        if variable == "temperature":
            ecmwf_col = "ecmwf_temperature_2m"
            gfs_col = "gfs_temperature_2m"
            ai_col = "ai_temperature_2m"
        elif variable == "rainfall":
            ecmwf_col = "ecmwf_precipitation_mm"
            gfs_col = "gfs_precipitation_mm"
            ai_col = "ai_precipitation_mm"
        else: # wind_speed
            ecmwf_col = "ecmwf_wind_speed_ms"
            gfs_col = "gfs_wind_speed_ms"
            ai_col = "ai_wind_speed_ms"

        # Populate AI prediction column if passed explicitly
        if ai_predictions is not None:
            df[ai_col] = ai_predictions
        elif ai_col not in df.columns:
            # Baseline mean fallback if AI prediction not computed yet
            df[ai_col] = (df[ecmwf_col] + df[gfs_col]) / 2.0

        # Detect weather regime if not already present
        if "weather_regime" not in df.columns:
            df = WeatherRegimeEngine.apply_regime_detection(df)

        # Predict dynamic context-aware weights
        weights_df = self.weighting_engine.predict_weights(df, variable=variable)
        
        w_ecmwf = weights_df["ecmwf_weight"].values
        w_gfs = weights_df["gfs_weight"].values
        w_ai = weights_df["ai_weight"].values

        val_ecmwf = df[ecmwf_col].values
        val_gfs = df[gfs_col].values
        val_ai = df[ai_col].values

        # Linear combination: Blended = w_ecmwf * ECMWF + w_gfs * GFS + w_ai * AI
        blended_values = w_ecmwf * val_ecmwf + w_gfs * val_gfs + w_ai * val_ai

        # Non-negativity clip for physical quantities (rainfall, wind)
        if variable in ["rainfall", "wind_speed"]:
            blended_values = np.maximum(0.0, blended_values)

        # Uncertainty estimation: weighted standard deviation + model disagreement
        disagreement_sq = (
            w_ecmwf * (val_ecmwf - blended_values) ** 2 +
            w_gfs * (val_gfs - blended_values) ** 2 +
            w_ai * (val_ai - blended_values) ** 2
        )
        uncertainty_std = np.sqrt(np.maximum(1e-4, disagreement_sq))

        result_df = pd.DataFrame({
            "issue_time_utc": df["issue_time_utc"],
            "valid_time_utc": df["valid_time_utc"],
            "lead_time_hours": df["lead_time_hours"],
            "location_id": df["location_id"],
            "latitude": df["latitude"],
            "longitude": df["longitude"],
            "variable": variable,
            "ecmwf_value": np.round(val_ecmwf, 2),
            "gfs_value": np.round(val_gfs, 2),
            "ai_value": np.round(val_ai, 2),
            "blended_value": np.round(blended_values, 2),
            "ecmwf_weight": np.round(w_ecmwf, 4),
            "gfs_weight": np.round(w_gfs, 4),
            "ai_weight": np.round(w_ai, 4),
            "uncertainty_std": np.round(uncertainty_std, 3),
            "weather_regime": df["weather_regime"],
        })

        return result_df
