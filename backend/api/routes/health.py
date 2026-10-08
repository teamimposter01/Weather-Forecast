"""
Health Check API Route.
"""
from datetime import datetime, timezone
from fastapi import APIRouter
from api.schemas import HealthResponse
from config.settings import API_VERSION
from storage.database import storage

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_health():
    """System status and DB connectivity check."""
    db_connected = False
    try:
        res = storage.query("SELECT 1 AS check_val")
        if not res.empty and res.iloc[0]["check_val"] == 1:
            db_connected = True
    except Exception as e:
        print(f"[Health] DB Check error: {e}")

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        timestamp_utc=datetime.now(timezone.utc),
        version=API_VERSION,
        database_connected=db_connected,
        data_source_status="LIVE" if db_connected else "ERROR",
    )
