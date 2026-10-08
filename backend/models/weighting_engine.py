"""
Dynamic Weighting Model using LightGBM.
Predicts adaptive context-aware weights for forecast sources (ECMWF, GFS, AI).
Enforces non-negativity and sum-to-one constraints via Softmax / L1 normalization.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from config.settings import MODELS_DIR, WEIGHTING_CONFIG

WEIGHTING_FEATURES = [
    "latitude",
    "longitude",
    "elevation_m",
    "hour",
    "month",
    "day_of_year",
    "lead_time_hours",
    "monsoon_indicator",
    "forecast_disagreement",
    "weather_regime_code",
    "mae_ecmwf",
    "mae_gfs",
    "mae_ai",
    "rmse_ecmwf",
    "rmse_gfs",
    "rmse_ai",
    "recent_error_ecmwf",
    "recent_error_gfs",
    "recent_error_ai",
]


class DynamicWeightingEngine:
    def __init__(self, models_dir: str = str(MODELS_DIR)):
        self.models_dir = models_dir
        self.model_ecmwf = None
        self.model_gfs = None
        self.model_ai = None
        self.regime_encoder = LabelEncoder()
        self.version = "v1.0.0"

    def _prepare_weighting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for LightGBM dynamic weighting."""
        df = df.copy()

        # Forecast disagreement (use max of temp, rain, wind disagreement if present)
        if "temp_disagreement" in df.columns:
            df["forecast_disagreement"] = df["temp_disagreement"]
        elif "rain_disagreement" in df.columns:
            df["forecast_disagreement"] = df["rain_disagreement"]
        else:
            df["forecast_disagreement"] = 0.5

        # Weather regime encoding
        if "weather_regime" not in df.columns:
            df["weather_regime"] = "NORMAL_WESTERLY"

        # Fit/transform label encoder safely
        known_regimes = [
            "MONSOON_ACTIVE", "MONSOON_BREAK", "PRE_MONSOON_HEAT",
            "POST_MONSOON_CYCLONIC", "WINTER_DRY", "NORMAL_WESTERLY", "UNKNOWN"
        ]
        self.regime_encoder.fit(known_regimes)
        
        regime_codes = []
        for r in df["weather_regime"]:
            if r in self.regime_encoder.classes_:
                regime_codes.append(self.regime_encoder.transform([r])[0])
            else:
                regime_codes.append(self.regime_encoder.transform(["UNKNOWN"])[0])
        df["weather_regime_code"] = regime_codes

        # Fill default historical skill features if missing
        for col in ["mae_ecmwf", "mae_gfs", "mae_ai", "rmse_ecmwf", "rmse_gfs", "rmse_ai",
                    "recent_error_ecmwf", "recent_error_gfs", "recent_error_ai"]:
            if col not in df.columns:
                df[col] = 1.0

        for col in WEIGHTING_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        return df[WEIGHTING_FEATURES].fillna(0.0)

    def train_weighting_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, variable: str = "temperature"):
        """
        Train LightGBM models to predict oracle inverse-error target weights.
        """
        print(f"[WeightingEngine] Training LightGBM Dynamic Weighting Model for {variable}...")

        target_col = f"target_{variable}_2m" if variable == "temperature" else f"target_{variable}_ms" if variable == "wind_speed" else "target_precipitation_mm"
        ecmwf_col = f"ecmwf_{variable}_2m" if variable == "temperature" else f"ecmwf_{variable}_ms" if variable == "wind_speed" else "ecmwf_precipitation_mm"
        gfs_col = f"gfs_{variable}_2m" if variable == "temperature" else f"gfs_{variable}_ms" if variable == "wind_speed" else "gfs_precipitation_mm"
        ai_col = f"ai_{variable}_2m" if variable == "temperature" else f"ai_{variable}_ms" if variable == "wind_speed" else f"ai_{variable}"

        # Ensure AI predictions exist in train_df/val_df
        if ai_col not in train_df.columns:
            train_df[ai_col] = (train_df[ecmwf_col] + train_df[gfs_col]) / 2.0
        if ai_col not in val_df.columns:
            val_df[ai_col] = (val_df[ecmwf_col] + val_df[gfs_col]) / 2.0

        # Compute absolute errors
        err_ecmwf = (train_df[ecmwf_col] - train_df[target_col]).abs().values + 1e-4
        err_gfs = (train_df[gfs_col] - train_df[target_col]).abs().values + 1e-4
        err_ai = (train_df[ai_col] - train_df[target_col]).abs().values + 1e-4

        # Oracle weights (Inverse error normalized)
        inv_e = 1.0 / err_ecmwf
        inv_g = 1.0 / err_gfs
        inv_a = 1.0 / err_ai
        inv_sum = inv_e + inv_g + inv_a

        y_w_ecmwf = inv_e / inv_sum
        y_w_gfs = inv_g / inv_sum
        y_w_ai = inv_a / inv_sum

        # Feature matrix
        X_train = self._prepare_weighting_features(train_df)
        X_val = self._prepare_weighting_features(val_df)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 5,
            "verbose": -1,
            "random_state": 42,
        }

        # Train 3 LightGBM Regressors for ECMWF, GFS, AI weight log-scores
        lgb_train_e = lgb.Dataset(X_train, label=y_w_ecmwf)
        lgb_train_g = lgb.Dataset(X_train, label=y_w_gfs)
        lgb_train_a = lgb.Dataset(X_train, label=y_w_ai)

        self.model_ecmwf = lgb.train(params, lgb_train_e, num_boost_round=100)
        self.model_gfs = lgb.train(params, lgb_train_g, num_boost_round=100)
        self.model_ai = lgb.train(params, lgb_train_a, num_boost_round=100)

        # Save models
        joblib.dump(self.model_ecmwf, os.path.join(self.models_dir, f"weight_lgb_ecmwf_{variable}.joblib"))
        joblib.dump(self.model_gfs, os.path.join(self.models_dir, f"weight_lgb_gfs_{variable}.joblib"))
        joblib.dump(self.model_ai, os.path.join(self.models_dir, f"weight_lgb_ai_{variable}.joblib"))

        print(f"[WeightingEngine] LightGBM Weighting Model for {variable} successfully trained.")

    def load_saved_models(self, variable: str = "temperature"):
        """Load trained LightGBM models."""
        p_e = os.path.join(self.models_dir, f"weight_lgb_ecmwf_{variable}.joblib")
        p_g = os.path.join(self.models_dir, f"weight_lgb_gfs_{variable}.joblib")
        p_a = os.path.join(self.models_dir, f"weight_lgb_ai_{variable}.joblib")

        if os.path.exists(p_e):
            self.model_ecmwf = joblib.load(p_e)
        if os.path.exists(p_g):
            self.model_gfs = joblib.load(p_g)
        if os.path.exists(p_a):
            self.model_ai = joblib.load(p_a)

    def predict_weights(self, df: pd.DataFrame, variable: str = "temperature") -> pd.DataFrame:
        """
        Predict context-aware adaptive weights (w_ecmwf, w_gfs, w_ai) for input DataFrame.
        Enforces non-negativity and sum = 1.0 constraint.
        """
        self.load_saved_models(variable)

        df_feat = self._prepare_weighting_features(df)
        n = len(df_feat)

        if self.model_ecmwf is not None and self.model_gfs is not None and self.model_ai is not None:
            raw_e = self.model_ecmwf.predict(df_feat)
            raw_g = self.model_gfs.predict(df_feat)
            raw_a = self.model_ai.predict(df_feat)
        else:
            # Equal weight baseline fallback if model not loaded
            raw_e = np.full(n, 0.3333)
            raw_g = np.full(n, 0.3333)
            raw_a = np.full(n, 0.3333)

        # Softmax normalization to strictly enforce non-negativity and sum to 1.0
        logits = np.column_stack([raw_e, raw_g, raw_a])
        # Subtract max for numerical stability before exp
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return pd.DataFrame({
            "ecmwf_weight": weights[:, 0],
            "gfs_weight": weights[:, 1],
            "ai_weight": weights[:, 2],
        }, index=df.index)
