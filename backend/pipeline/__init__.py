"""
Pipeline package initialization.
"""
from pipeline.ingest import run_ingestion
from pipeline.preprocess import run_preprocessing
from pipeline.forecast import run_forecast_pipeline
from pipeline.blend import run_blend_pipeline
from pipeline.extreme import run_extreme_pipeline
from pipeline.run_all import run_full_pipeline
