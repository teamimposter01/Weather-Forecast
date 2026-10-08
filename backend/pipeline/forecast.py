"""
Pipeline Forecast Module.
Trains AI Base Models (Temp, Two-Stage Rain, Wind) on historical training data
and generates AI model predictions.
"""
import pandas as pd
from pipeline.preprocess import DataPreprocessor
from models.base_models import BaseWeatherModels

def run_forecast_pipeline():
    """Execute AI model training and prediction step."""
    print("[Pipeline] Starting AI Forecast Step...")
    processor = DataPreprocessor()
    obs_df, forecast_df = processor.load_raw_data()
    aligned_df = processor.create_aligned_dataset(obs_df, forecast_df)

    # Chronological train / validation split
    train_df, val_df, test_df = processor.get_chronological_split(aligned_df)

    # Initialize & train models
    base_models = BaseWeatherModels()
    temp_metrics = base_models.train_temperature_model(train_df, val_df)
    rain_metrics = base_models.train_rainfall_two_stage_model(train_df, val_df)
    wind_metrics = base_models.train_wind_model(train_df, val_df)

    # Generate predictions on full dataset
    ai_preds = base_models.predict(aligned_df)
    
    aligned_df["ai_temperature_2m"] = ai_preds.get("temperature", aligned_df["ecmwf_temperature_2m"])
    aligned_df["ai_precipitation_mm"] = ai_preds.get("rainfall", aligned_df["ecmwf_precipitation_mm"])
    aligned_df["ai_wind_speed_ms"] = ai_preds.get("wind_speed", aligned_df["ecmwf_wind_speed_ms"])

    print("[Pipeline] AI Forecast Step completed successfully.")
    return aligned_df

if __name__ == "__main__":
    run_forecast_pipeline()
