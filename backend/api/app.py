"""
FastAPI Backend Main Application Entrypoint.
Mounts modular API routers, handles CORS, and serves the Web Dashboard.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from config.settings import API_TITLE, API_VERSION, BASE_DIR
from api.routes import health, forecast, weights, models_route, skills, extremes, backtest, integrity
from dashboard.router import dashboard_router

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Hybrid AI-NWP Multi-Model Weather Forecast Blending System for India",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers under /api
app.include_router(health.router, prefix="/api")
app.include_router(forecast.router, prefix="/api")
app.include_router(weights.router, prefix="/api")
app.include_router(models_route.router, prefix="/api")
app.include_router(skills.router, prefix="/api")
app.include_router(extremes.router, prefix="/api")
app.include_router(backtest.router, prefix="/api")
app.include_router(integrity.router, prefix="/api")

# Include Dashboard Router
app.include_router(dashboard_router)

@app.on_event("startup")
def startup_event():
    """Ensure database tables and initial baseline forecast cache exist."""
    try:
        from storage.database import storage
        storage._init_db()
        check = storage.query("SELECT COUNT(*) as cnt FROM blended_forecasts")
        if check.empty or check.iloc[0]["cnt"] == 0:
            print("[Startup] Seeding initial forecast dataset...")
            from tests.test_data_generator import generate_synthetic_test_dataset
            from pipeline.preprocess import DataPreprocessor
            from models.fusion_engine import ForecastFusionEngine
            from models.extreme_engine import ExtremeWeatherEngine
            import pandas as pd
            obs, fc = generate_synthetic_test_dataset(days=3)
            p = DataPreprocessor()
            aligned = p.create_aligned_dataset(obs, fc, save_to_db=False)
            fe = ForecastFusionEngine()
            b_t = fe.fuse_forecasts(aligned, variable="temperature")
            b_r = fe.fuse_forecasts(aligned, variable="rainfall")
            b_w = fe.fuse_forecasts(aligned, variable="wind_speed")
            all_blended = pd.concat([b_t, b_r, b_w], ignore_index=True)
            storage.save_dataframe(all_blended, "blended_forecasts", mode="append")
            ee = ExtremeWeatherEngine()
            probs = ee.predict_extreme_probabilities(aligned)
            storage.save_dataframe(probs, "extreme_probabilities", mode="append")
            print("[Startup] Initial dataset successfully seeded.")
    except Exception as e:
        print(f"[Startup] Bootstrap notice: {e}")

# Mount Dashboard Static files if directory exists
static_dir = BASE_DIR / "dashboard" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    """Redirect to Dashboard interface."""
    return """
    <html>
        <head><title>Hybrid AI-NWP Weather Blending System</title></head>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
            <h1>Hybrid AI–NWP Weather Forecast Blending System for India</h1>
            <p><a href="/dashboard" style="font-size: 20px; font-weight: bold; color: #2563eb;">👉 Go to Interactive Dashboard</a></p>
            <p><a href="/docs" style="font-size: 16px; color: #4b5563;">Interactive API Documentation (Swagger Docs)</a></p>
        </body>
    </html>
    """
