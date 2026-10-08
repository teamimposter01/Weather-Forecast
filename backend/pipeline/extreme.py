"""
Pipeline Extreme Weather Module.
Trains extreme weather classifiers and computes event probabilities (Heatwave, Heavy Rainfall, High Wind).
"""
import pandas as pd
from pipeline.preprocess import DataPreprocessor
from models.extreme_engine import ExtremeWeatherEngine
from storage.database import storage

def run_extreme_pipeline(df: pd.DataFrame = None):
    """Execute extreme weather probability generation step."""
    print("[Pipeline] Starting Extreme Weather Analysis Step...")

    processor = DataPreprocessor()
    if df is None:
        obs_df, forecast_df = processor.load_raw_data()
        df = processor.create_aligned_dataset(obs_df, forecast_df)

    train_df, val_df, test_df = processor.get_chronological_split(df)

    extreme_engine = ExtremeWeatherEngine()
    extreme_engine.train_extreme_classifiers(train_df, val_df)

    probs_df = extreme_engine.predict_extreme_probabilities(df)
    storage.save_dataframe(probs_df, "extreme_probabilities", mode="overwrite")

    print(f"[Pipeline] Extreme weather analysis completed ({len(probs_df)} records stored).")
    return probs_df

if __name__ == "__main__":
    run_extreme_pipeline()
