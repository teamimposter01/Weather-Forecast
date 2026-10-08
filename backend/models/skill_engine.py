"""
Historical Skill Engine.
Calculates rolling MAE, RMSE, and Bias for individual models (ECMWF, GFS, AI)
broken down by location, variable, season, and lead time over historical verification windows.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional
from storage.database import storage

class HistoricalSkillEngine:
    def __init__(self, db_storage=storage):
        self.storage = db_storage

    def compute_skill_metrics(self, df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
        """
        Compute historical skill metrics (MAE, RMSE, Bias) for ECMWF, GFS, and AI forecast sources
        grouped by location_id, variable, model_name, lead_time_hours, and season.
        """
        df = df.copy()
        if df.empty:
            return pd.DataFrame()

        results = []

        variables = {
            "temperature": ("target_temperature_2m", ["ecmwf_temperature_2m", "gfs_temperature_2m", "ai_temperature_2m"]),
            "rainfall": ("target_precipitation_mm", ["ecmwf_precipitation_mm", "gfs_precipitation_mm", "ai_precipitation_mm"]),
            "wind_speed": ("target_wind_speed_ms", ["ecmwf_wind_speed_ms", "gfs_wind_speed_ms", "ai_wind_speed_ms"]),
        }

        for var_name, (target_col, model_cols) in variables.items():
            if target_col not in df.columns:
                continue

            for model_col in model_cols:
                if model_col not in df.columns:
                    continue
                model_name = model_col.split("_")[0]  # 'ecmwf', 'gfs', 'ai'

                # Group by location, season, lead_time
                groupby_cols = ["location_id", "season", "lead_time_hours"]
                for keys, group in df.groupby(groupby_cols):
                    loc_id, season, lead = keys
                    
                    y_true = group[target_col].values
                    y_pred = group[model_col].values
                    
                    # Remove NaNs
                    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                    if not np.any(valid_mask):
                        continue

                    yt, yp = y_true[valid_mask], y_pred[valid_mask]
                    count = len(yt)
                    if count == 0:
                        continue

                    errors = yp - yt
                    mae = float(np.mean(np.abs(errors)))
                    rmse = float(np.sqrt(np.mean(errors ** 2)))
                    bias = float(np.mean(errors))

                    results.append({
                        "location_id": loc_id,
                        "variable": var_name,
                        "model_name": model_name,
                        "lead_time_hours": int(lead),
                        "season": season,
                        "mae": round(mae, 4),
                        "rmse": round(rmse, 4),
                        "bias": round(bias, 4),
                        "sample_count": count,
                        "updated_at_utc": datetime.now(timezone.utc),
                    })

        skill_df = pd.DataFrame(results)
        if not skill_df.empty:
            self.storage.save_dataframe(skill_df, "historical_skill", mode="overwrite")

        return skill_df

    def get_model_skill(
        self, location_id: str, variable: str, season: str, lead_time: int
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve recent skill for all models for a specific context."""
        query = """
            SELECT model_name, mae, rmse, bias, sample_count
            FROM historical_skill
            WHERE location_id = ? AND variable = ? AND season = ? AND lead_time_hours = ?
        """
        df = self.storage.query(query, [location_id, variable, season, lead_time])
        
        skills = {}
        if df.empty:
            # Fallback default skills if no historical records exist yet
            for m in ["ecmwf", "gfs", "ai"]:
                skills[m] = {"mae": 1.5, "rmse": 2.0, "bias": 0.0, "sample_count": 100}
            return skills

        for _, row in df.iterrows():
            skills[row["model_name"]] = {
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "bias": float(row["bias"]),
                "sample_count": int(row["sample_count"]),
            }

        # Ensure all 3 models exist
        for m in ["ecmwf", "gfs", "ai"]:
            if m not in skills:
                skills[m] = {"mae": 1.5, "rmse": 2.0, "bias": 0.0, "sample_count": 50}

        return skills
