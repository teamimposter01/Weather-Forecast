"""
Backtest Verification API Route.
Runs out-of-sample backtest comparison across ECMWF, GFS, AI, Equal Weighting, and Hybrid Dynamic System.
"""
from typing import Dict, Any
from fastapi import APIRouter, Query
from api.schemas import BacktestResponse
from pipeline.preprocess import DataPreprocessor
from models.backtester import Backtester
from models.fusion_engine import ForecastFusionEngine
from models.weighting_engine import DynamicWeightingEngine
from storage.database import storage

router = APIRouter(prefix="/backtest", tags=["Backtesting"])

@router.get("", response_model=BacktestResponse)
def get_backtest_results(variable: str = Query("temperature", description="Variable (temperature, rainfall, wind_speed)")):
    """Run out-of-sample backtesting comparison and return scientific performance summary."""
    processor = DataPreprocessor()
    obs_df, forecast_df = processor.load_raw_data()
    aligned_df = processor.create_aligned_dataset(obs_df, forecast_df)

    # Get out-of-sample test split
    train_df, val_df, test_df = processor.get_chronological_split(aligned_df)

    # Run fusion on test split
    weighting_engine = DynamicWeightingEngine()
    weighting_engine.train_weighting_model(train_df, val_df, variable=variable)
    
    fusion_engine = ForecastFusionEngine(weighting_engine=weighting_engine)
    fusion_test_results = fusion_engine.fuse_forecasts(test_df, variable=variable)

    backtester = Backtester()
    results = backtester.run_backtest(test_df, fusion_test_results, variable=variable)

    return BacktestResponse(**results)
