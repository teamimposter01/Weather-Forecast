"""
Weather Regime Engine for India.
Detects prevailing meteorological regimes (Monsoon Active, Monsoon Break, Pre-Monsoon Heatwave,
Post-Monsoon Cyclonic/Depression, Winter Dry, Normal Westerly) to inform dynamic weighting.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union

REGIME_NAMES = [
    "MONSOON_ACTIVE",
    "MONSOON_BREAK",
    "PRE_MONSOON_HEAT",
    "POST_MONSOON_CYCLONIC",
    "WINTER_DRY",
    "NORMAL_WESTERLY",
]

class WeatherRegimeEngine:
    @staticmethod
    def detect_regime(
        month: int,
        temperature_2m: float,
        precipitation_mm: float,
        wind_speed_ms: float,
        relative_humidity_pct: float = 65.0,
    ) -> str:
        """
        Detect weather regime based on physical meteorological indicators.
        """
        # Southwest Monsoon (June to September)
        if month in [6, 7, 8, 9]:
            if precipitation_mm > 5.0 or relative_humidity_pct > 75.0:
                return "MONSOON_ACTIVE"
            elif temperature_2m > 34.0 and precipitation_mm < 1.0:
                return "MONSOON_BREAK"
            else:
                return "MONSOON_ACTIVE"

        # Pre-Monsoon Summer (March to May)
        elif month in [3, 4, 5]:
            if temperature_2m >= 38.0:
                return "PRE_MONSOON_HEAT"
            elif wind_speed_ms > 12.0 and precipitation_mm > 15.0:
                return "POST_MONSOON_CYCLONIC"
            else:
                return "PRE_MONSOON_HEAT"

        # Post-Monsoon / Northeast Monsoon (October to November)
        elif month in [10, 11]:
            if wind_speed_ms > 12.0 or precipitation_mm > 20.0:
                return "POST_MONSOON_CYCLONIC"
            else:
                return "NORMAL_WESTERLY"

        # Winter (December to February)
        elif month in [12, 1, 2]:
            if temperature_2m < 22.0 and precipitation_mm < 2.0:
                return "WINTER_DRY"
            else:
                return "NORMAL_WESTERLY"

        return "NORMAL_WESTERLY"

    @classmethod
    def apply_regime_detection(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized regime detection for a DataFrame."""
        df = df.copy()
        regimes = []
        for _, row in df.iterrows():
            month = int(row.get("month", 6))
            temp = float(row.get("ecmwf_temperature_2m", row.get("temperature_2m", 25.0)))
            precip = float(row.get("ecmwf_precipitation_mm", row.get("precipitation_mm", 0.0)))
            wind = float(row.get("ecmwf_wind_speed_ms", row.get("wind_speed_ms", 3.0)))
            rh = float(row.get("relative_humidity_pct", 65.0))
            
            r = cls.detect_regime(month, temp, precip, wind, rh)
            regimes.append(r)

        df["weather_regime"] = regimes
        return df
