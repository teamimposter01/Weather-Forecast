"""
Historical Skill API Route.
Exposes historical verification skill metrics (MAE, RMSE, Bias) for ECMWF, GFS, and AI.
"""
from typing import List, Optional
from fastapi import APIRouter
from api.schemas import SkillScoreItem
from storage.database import storage

router = APIRouter(prefix="/skills", tags=["Historical Skill"])

@router.get("", response_model=List[SkillScoreItem])
def get_historical_skills(
    location: Optional[str] = None,
    variable: Optional[str] = None,
    season: Optional[str] = None,
):
    """Query historical forecast skill scores by location, variable, and season."""
    query = "SELECT * FROM historical_skill WHERE 1=1"
    params = []
    if location:
        query += " AND location_id = ?"
        params.append(location.lower())
    if variable:
        query += " AND variable = ?"
        params.append(variable.lower())
    if season:
        query += " AND season = ?"
        params.append(season.upper())

    query += " ORDER BY location_id, variable, lead_time_hours LIMIT 100"
    df = storage.query(query, params)

    if df.empty:
        return []

    results = []
    for _, r in df.iterrows():
        results.append(SkillScoreItem(
            location_id=str(r["location_id"]),
            variable=str(r["variable"]),
            model_name=str(r["model_name"]),
            lead_time_hours=int(r["lead_time_hours"]),
            season=str(r["season"]),
            mae=float(r["mae"]),
            rmse=float(r["rmse"]),
            bias=float(r["bias"]),
            sample_count=int(r["sample_count"]),
        ))
    return results
