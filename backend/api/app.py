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
