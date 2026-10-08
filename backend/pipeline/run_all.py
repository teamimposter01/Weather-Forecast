"""
End-to-End Operational Pipeline Execution Script.
Executes Ingest -> Preprocess -> Forecast -> Blend -> Extreme -> Storage.
"""
import sys
import logging
from datetime import datetime, timezone

from pipeline.ingest import run_ingestion
from pipeline.preprocess import run_preprocessing
from pipeline.forecast import run_forecast_pipeline
from pipeline.blend import run_blend_pipeline
from pipeline.extreme import run_extreme_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_full_pipeline():
    """Run all steps of the weather forecasting and blending system."""
    logging.info("==================================================")
    logging.info("STARTING HYBRID AI-NWP WEATHER FORECAST BLENDING PIPELINE")
    logging.info("==================================================")

    start_time = datetime.now(timezone.utc)

    # Step 1: Ingestion
    logging.info("STEP 1: Data Ingestion (Real Open-Meteo NWP & ERA5 APIs)...")
    obs_df, forecast_df, provenance = run_ingestion()
    logging.info(f"Ingestion Provenance: {provenance}")

    # Step 2: Preprocessing & Alignment
    logging.info("STEP 2: Data Preprocessing & Alignment...")
    aligned_df = run_preprocessing()

    # Step 3: AI Base Models Forecasting
    logging.info("STEP 3: AI Models Training & Forecasting...")
    forecast_df = run_forecast_pipeline()

    # Step 4: Skill Engine, Weather Regime Detection & LightGBM Dynamic Blending
    logging.info("STEP 4: Skill Engine, Weather Regimes & Dynamic Fusion...")
    blended_df = run_blend_pipeline(forecast_df)

    # Step 5: Extreme Weather Probabilities
    logging.info("STEP 5: Extreme Weather Classification...")
    extreme_df = run_extreme_pipeline(forecast_df)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    logging.info("==================================================")
    logging.info(f"OPERATIONAL PIPELINE COMPLETED SUCCESSFULLY IN {duration:.2f} SECONDS")
    logging.info("==================================================")

if __name__ == "__main__":
    run_full_pipeline()
