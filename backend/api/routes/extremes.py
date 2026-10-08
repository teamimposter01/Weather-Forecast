"""
Extreme Weather API Route.
Exposes extreme weather event probabilities (Heatwave, Heavy Rainfall, High Wind).
"""
from typing import List, Optional
from fastapi import APIRouter
from storage.database import storage

router = APIRouter(prefix="/extremes", tags=["Extreme Weather"])

@router.get("")
def get_extreme_weather_guidance(location: Optional[str] = None):
    """Retrieve extreme weather probabilities across Indian locations."""
    query = "SELECT * FROM extreme_probabilities WHERE 1=1"
    params = []
    if location:
        query += " AND location_id = ?"
        params.append(location.lower())

    query += " ORDER BY valid_time_utc ASC, lead_time_hours ASC LIMIT 100"
    df = storage.query(query, params)

    if df.empty:
        return []

    results = []
    for _, r in df.iterrows():
        results.append({
            "location_id": str(r["location_id"]),
            "issue_time_utc": r["issue_time_utc"],
            "valid_time_utc": r["valid_time_utc"],
            "lead_time_hours": int(r["lead_time_hours"]),
            "heatwave_prob_pct": round(float(r["heatwave_prob"]) * 100.0, 1),
            "heavy_rainfall_prob_pct": round(float(r["heavy_rainfall_prob"]) * 100.0, 1),
            "high_wind_prob_pct": round(float(r["high_wind_prob"]) * 100.0, 1),
        })
    return results
