"""
Location configuration for Indian Weather Forecasting System.
"""
from typing import Dict, List, TypedDict

class LocationInfo(TypedDict):
    name: str
    latitude: float
    longitude: float
    elevation_m: float
    region: str
    state: str

INDIAN_LOCATIONS: Dict[str, LocationInfo] = {
    "chennai": {
        "name": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "elevation_m": 6.0,
        "region": "South Peninsular",
        "state": "Tamil Nadu",
    },
    "bengaluru": {
        "name": "Bengaluru",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "elevation_m": 920.0,
        "region": "South Peninsular",
        "state": "Karnataka",
    },
    "mumbai": {
        "name": "Mumbai",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "elevation_m": 14.0,
        "region": "West Coast",
        "state": "Maharashtra",
    },
    "delhi": {
        "name": "Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "elevation_m": 216.0,
        "region": "North India",
        "state": "Delhi",
    },
    "kolkata": {
        "name": "Kolkata",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "elevation_m": 9.0,
        "region": "East India",
        "state": "West Bengal",
    },
    "hyderabad": {
        "name": "Hyderabad",
        "latitude": 17.3850,
        "longitude": 78.4867,
        "elevation_m": 542.0,
        "region": "Central/Deccan",
        "state": "Telangana",
    },
}

# Regional mapping for spatial aggregation
REGIONS = {
    "South Peninsular": ["chennai", "bengaluru"],
    "West Coast": ["mumbai"],
    "North India": ["delhi"],
    "East India": ["kolkata"],
    "Central/Deccan": ["hyderabad"],
}

# Grid boundaries for India spatial dynamic weight mapping
INDIA_GRID_BOUNDS = {
    "min_lat": 8.0,
    "max_lat": 37.0,
    "min_lon": 68.0,
    "max_lon": 97.0,
    "step": 2.0,  # Grid resolution in degrees
}

def get_location_by_id(loc_id: str) -> LocationInfo:
    """Retrieve location details by identifier."""
    key = loc_id.lower().strip()
    if key not in INDIAN_LOCATIONS:
        raise ValueError(f"Unknown location identifier: {loc_id}. Valid locations: {list(INDIAN_LOCATIONS.keys())}")
    return INDIAN_LOCATIONS[key]

def list_locations() -> List[LocationInfo]:
    """Get list of all supported Indian locations."""
    return list(INDIAN_LOCATIONS.values())
