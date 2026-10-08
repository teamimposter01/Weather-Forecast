"""
Pipeline Blend Module.
Calculates historical model skill, detects weather regimes, predicts context-aware dynamic weights using LightGBM,
and fuses ECMWF, GFS, and AI forecasts into final blended forecasts.
"""
import pandas as pd
from pipeline.preprocess import DataPreprocessor
from pipeline.forecast import run_forecast_pipeline
from models.skill_engine import HistoricalSkillEngine
from models.regime_engine import WeatherRegimeEngine
from models.weighting_engine import DynamicWeightingEngine
from models.fusion_engine import ForecastFusionEngine
from storage.database import storage

def run_blend_pipeline(df: pd.DataFrame = None):
    """Execute dynamic forecast blending step."""
    print("[Pipeline] Starting Dynamic Forecast Blending Step...")

    if df is None:
        df = run_forecast_pipeline()

    processor = DataPreprocessor()
    train_df, val_df, test_df = processor.get_chronological_split(df)

    # 1. Historical Skill Engine
    skill_engine = HistoricalSkillEngine()
    skill_df = skill_engine.compute_skill_metrics(train_df)

    # 2. Weather Regime Engine
    df = WeatherRegimeEngine.apply_regime_detection(df)
    train_df = WeatherRegimeEngine.apply_regime_detection(train_df)
    val_df = WeatherRegimeEngine.apply_regime_detection(val_df)

    # 3. Dynamic Weighting Engine
    weighting_engine = DynamicWeightingEngine()
    for var in ["temperature", "rainfall", "wind_speed"]:
        weighting_engine.train_weighting_model(train_df, val_df, variable=var)

    # 4. Forecast Fusion Engine
    fusion_engine = ForecastFusionEngine(weighting_engine=weighting_engine)

    blended_temp = fusion_engine.fuse_forecasts(df, variable="temperature")
    blended_rain = fusion_engine.fuse_forecasts(df, variable="rainfall")
    blended_wind = fusion_engine.fuse_forecasts(df, variable="wind_speed")

    all_blended = pd.concat([blended_temp, blended_rain, blended_wind], ignore_index=True)
    storage.save_dataframe(all_blended, "blended_forecasts", mode="overwrite")

    print(f"[Pipeline] Dynamic Forecast Blending completed ({len(all_blended)} blended records stored).")
    return all_blended

if __name__ == "__main__":
    run_blend_pipeline()
