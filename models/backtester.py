"""
Out-of-sample Backtesting and Verification Framework.
Compares individual NWP models (ECMWF, GFS), AI models, Simple Equal Averaging,
and the Hybrid Dynamic Weighting System against actual observations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from models.sklearn_compat import mean_absolute_error, root_mean_squared_error

class Backtester:
    @staticmethod
    def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard metrics (MAE, RMSE, Bias)."""
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not np.any(valid):
            return {"mae": 0.0, "rmse": 0.0, "bias": 0.0}

        yt, yp = y_true[valid], y_pred[valid]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(root_mean_squared_error(yt, yp))
        bias = float(np.mean(yp - yt))
        return {"mae": round(mae, 4), "rmse": round(rmse, 4), "bias": round(bias, 4)}

    def run_backtest(
        self,
        test_df: pd.DataFrame,
        fusion_results: pd.DataFrame,
        variable: str = "temperature"
    ) -> Dict[str, Any]:
        """
        Run side-by-side out-of-sample backtest comparison across:
        - ECMWF
        - GFS
        - AI Base Model
        - Equal Weight Average
        - Hybrid Dynamic Blended System
        """
        if variable == "temperature":
            target_col = "target_temperature_2m"
            ecmwf_col = "ecmwf_temperature_2m"
            gfs_col = "gfs_temperature_2m"
            ai_col = "ai_temperature_2m"
        elif variable == "rainfall":
            target_col = "target_precipitation_mm"
            ecmwf_col = "ecmwf_precipitation_mm"
            gfs_col = "gfs_precipitation_mm"
            ai_col = "ai_precipitation_mm"
        else: # wind_speed
            target_col = "target_wind_speed_ms"
            ecmwf_col = "ecmwf_wind_speed_ms"
            gfs_col = "gfs_wind_speed_ms"
            ai_col = "ai_wind_speed_ms"

        y_true = test_df[target_col].values
        val_ecmwf = test_df[ecmwf_col].values
        val_gfs = test_df[gfs_col].values
        val_ai = test_df[ai_col].values if ai_col in test_df.columns else (val_ecmwf + val_gfs) / 2.0
        val_equal = (val_ecmwf + val_gfs + val_ai) / 3.0
        val_hybrid = fusion_results["blended_value"].values

        ecmwf_metrics = self.evaluate_model_performance(y_true, val_ecmwf)
        gfs_metrics = self.evaluate_model_performance(y_true, val_gfs)
        ai_metrics = self.evaluate_model_performance(y_true, val_ai)
        equal_metrics = self.evaluate_model_performance(y_true, val_equal)
        hybrid_metrics = self.evaluate_model_performance(y_true, val_hybrid)

        # Skill Score relative to ECMWF benchmark: 1 - (MAE_hybrid / MAE_ecmwf)
        mae_ecmwf = max(ecmwf_metrics["mae"], 1e-4)
        mae_equal = max(equal_metrics["mae"], 1e-4)
        
        skill_vs_ecmwf = round((1.0 - (hybrid_metrics["mae"] / mae_ecmwf)) * 100, 2)
        skill_vs_equal = round((1.0 - (hybrid_metrics["mae"] / mae_equal)) * 100, 2)

        results = {
            "variable": variable,
            "sample_size": len(test_df),
            "models": {
                "ECMWF": ecmwf_metrics,
                "GFS": gfs_metrics,
                "AI_Base": ai_metrics,
                "Equal_Weight_Average": equal_metrics,
                "Hybrid_Dynamic_Blended": hybrid_metrics,
            },
            "skill_improvement_pct": {
                "vs_ecmwf": skill_vs_ecmwf,
                "vs_equal_weight": skill_vs_equal,
            },
            "research_conclusion": (
                f"Hybrid Dynamic Blending improved forecast MAE by {skill_vs_ecmwf}% over ECMWF "
                f"and by {skill_vs_equal}% over simple equal averaging."
            )
        }
        return results
