"""
Data Integrity API Route.
Exposes database audit report including record counts, duplicates, and freshness status.
"""
from datetime import datetime, timezone
from fastapi import APIRouter
from api.schemas import DataIntegrityResponse
from storage.database import storage

router = APIRouter(prefix="/integrity", tags=["Data Integrity"])

@router.get("", response_model=DataIntegrityResponse)
def get_data_integrity_report():
    """Audit database integrity, duplicate counts, and table health."""
    report = storage.check_data_integrity()
    
    # Calculate overall health
    degraded = any(info.get("status") != "HEALTHY" for info in report.values() if isinstance(info, dict))
    overall = "HEALTHY" if not degraded else "DEGRADED"

    return DataIntegrityResponse(
        timestamp_utc=datetime.now(timezone.utc),
        overall_status=overall,
        tables=report
    )
