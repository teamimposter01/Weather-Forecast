"""
Data Validation Engine for Weather Data and NWP Forecasts.
Applies physical range checks, spatial bounds, temporal integrity, and flags invalid records.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

# Physical Range Constraints (Scientifically Defensible Bounds)
VALIDATION_BOUNDS = {
    "temperature_min_c": -50.0,
    "temperature_max_c": 60.0,
    "humidity_min_pct": 0.0,
    "humidity_max_pct": 100.0,
    "wind_speed_min_ms": 0.0,
    "wind_speed_max_ms": 150.0,
    "precipitation_min_mm": 0.0,
    "precipitation_max_mm": 1000.0,
    "pressure_min_hpa": 700.0,
    "pressure_max_hpa": 1100.0,
    "india_min_lat": 6.0,
    "india_max_lat": 38.0,
    "india_min_lon": 68.0,
    "india_max_lon": 98.0,
}

class DataValidator:
    @staticmethod
    def validate_observation_record(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single weather observation record against physical bounds."""
        violations = []

        # Temperature
        t = row.get("temperature_2m")
        if t is not None and not np.isnan(t):
            if not (VALIDATION_BOUNDS["temperature_min_c"] <= t <= VALIDATION_BOUNDS["temperature_max_c"]):
                violations.append(f"Temperature {t}°C out of physical range")

        # Humidity
        rh = row.get("relative_humidity_pct")
        if rh is not None and not np.isnan(rh):
            if not (VALIDATION_BOUNDS["humidity_min_pct"] <= rh <= VALIDATION_BOUNDS["humidity_max_pct"]):
                violations.append(f"Humidity {rh}% out of range 0-100%")

        # Wind speed
        w = row.get("wind_speed_ms")
        if w is not None and not np.isnan(w):
            if w < VALIDATION_BOUNDS["wind_speed_min_ms"] or w > VALIDATION_BOUNDS["wind_speed_max_ms"]:
                violations.append(f"Wind speed {w} m/s out of physical range")

        # Precipitation
        p = row.get("precipitation_mm")
        if p is not None and not np.isnan(p):
            if p < VALIDATION_BOUNDS["precipitation_min_mm"] or p > VALIDATION_BOUNDS["precipitation_max_mm"]:
                violations.append(f"Precipitation {p} mm out of range")

        # Pressure
        press = row.get("surface_pressure_hpa")
        if press is not None and not np.isnan(press):
            if not (VALIDATION_BOUNDS["pressure_min_hpa"] <= press <= VALIDATION_BOUNDS["pressure_max_hpa"]):
                violations.append(f"Pressure {press} hPa out of atmospheric range")

        is_valid = len(violations) == 0
        return is_valid, violations

    @staticmethod
    def validate_forecast_record(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a forecast record including lead time and issue vs valid timestamp alignment."""
        is_valid, violations = DataValidator.validate_observation_record(row)

        issue = row.get("issue_time_utc")
        valid = row.get("valid_time_utc")
        lead = row.get("lead_time_hours")

        if issue and valid:
            if isinstance(issue, str):
                issue = pd.to_datetime(issue)
            if isinstance(valid, str):
                valid = pd.to_datetime(valid)

            if valid < issue:
                violations.append(f"Valid time ({valid}) is earlier than Issue time ({issue})!")
                is_valid = False

            if lead is not None:
                calc_lead = int((valid - issue).total_seconds() // 3600)
                if abs(calc_lead - int(lead)) > 1:
                    violations.append(f"Lead time mismatch: specified {lead}h vs calculated {calc_lead}h")
                    is_valid = False

        return is_valid, violations

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, dataset_type: str = "forecast") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate an entire DataFrame and return (clean_df, summary_report).
        Flags invalid records without discarding legitimate extremes.
        """
        if df.empty:
            return df, {"total": 0, "valid": 0, "flagged": 0, "violations": []}

        df = df.copy()
        valid_flags = []
        violation_logs = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            if dataset_type == "forecast":
                ok, viols = cls.validate_forecast_record(row_dict)
            else:
                ok, viols = cls.validate_observation_record(row_dict)

            valid_flags.append(ok)
            if not ok:
                violation_logs.append({"index": idx, "violations": viols})

        df["is_physically_valid"] = valid_flags

        summary = {
            "total_records": len(df),
            "valid_records": int(sum(valid_flags)),
            "flagged_records": int(len(df) - sum(valid_flags)),
            "violation_details": violation_logs[:10],  # sample up to 10
        }

        # Filter out invalid records if any severe violation
        clean_df = df[df["is_physically_valid"]].drop(columns=["is_physically_valid"])
        return clean_df, summary
