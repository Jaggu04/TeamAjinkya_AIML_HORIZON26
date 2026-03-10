"""
backend/models/parking_intelligence.py
MODULE 3 — Parking Intelligence for ALL 50 Mumbai parking lots.
Uses statistical model (works without sklearn if not installed).
"""

import numpy as np
from datetime import datetime
from backend.models.mumbai_routes import MUMBAI_PARKING, nearest_parking, ZONE_MULT

# Base occupancy curve by hour
HOURLY_OCC = {
    0:0.05,1:0.03,2:0.02,3:0.02,4:0.05,
    5:0.15,6:0.38,7:0.72,8:0.88,9:0.90,
   10:0.82,11:0.78,12:0.80,13:0.82,14:0.80,
   15:0.83,16:0.88,17:0.91,18:0.89,19:0.82,
   20:0.68,21:0.52,22:0.33,23:0.14,
}

# Zone parking demand multipliers
ZONE_PARK_MULT = {
    "BKC":             1.30,
    "South Mumbai":    1.20,
    "Central Mumbai":  1.15,
    "Western Suburbs": 1.00,
    "Eastern Suburbs": 0.95,
    "Thane":           0.90,
    "Navi Mumbai":     0.80,
    "Outskirts":       0.60,
}

# Special lot types
LOT_TYPE_MULT = {
    "airport":   0.70,  # large, rarely full
    "mall":      1.10,  # busy on weekends
    "hospital":  0.85,
    "station":   1.20,  # always busy
    "default":   1.00,
}

def _lot_type(lot_name: str) -> str:
    n = lot_name.lower()
    if "airport" in n:    return "airport"
    if "mall" in n:       return "mall"
    if "hospital" in n:   return "hospital"
    if "station" in n:    return "station"
    return "default"

def predict_lot_availability(lot: dict, hour: int, dow: int,
                              weather_code: int = 0,
                              has_event: bool = False) -> float:
    """
    Predict parking availability % for a specific lot.
    Returns 0–100 (higher = more parking available).
    """
    zone      = lot.get("zone", "Western Suburbs")
    ltype     = _lot_type(lot["name"])
    capacity  = lot.get("capacity", 300)

    base_occ   = HOURLY_OCC[hour]
    day_mult   = 1.30 if dow >= 5 else 1.0  # weekends busier for malls
    zone_mult  = ZONE_PARK_MULT.get(zone, 1.0)
    type_mult  = LOT_TYPE_MULT.get(ltype, 1.0)

    # Weekend adjustment — malls busier, offices less busy
    if dow >= 5:
        if ltype == "mall":    type_mult *= 1.30
        if "station" in ltype: type_mult *= 0.70
        zone_mult = ZONE_PARK_MULT.get(zone, 1.0) * 0.80  # overall less congested

    # Weather: rain reduces driving to malls/offices
    weather_adj = {0: 0.0, 1: -0.08, 2: -0.15, 3: -0.05}.get(weather_code, 0)

    # Event: increases demand near affected zone
    event_adj = 0.25 if has_event else 0.0

    # Capacity factor: large lots fill less percentage
    capacity_adj = -0.10 if capacity > 800 else (0.10 if capacity < 200 else 0.0)

    # Deterministic noise per lot
    seed  = hash(lot["id"] + str(hour) + str(dow)) % 1000
    noise = (seed % 100 - 50) / 500  # ±0.10

    occupancy = (base_occ * zone_mult * type_mult
                 + weather_adj + event_adj + capacity_adj + noise)
    occupancy = float(np.clip(occupancy, 0.0, 1.0))
    availability = round((1 - occupancy) * 100, 1)
    return float(np.clip(availability, 0, 100))


def predict_parking_near(dest_lat: float, dest_lng: float,
                          arrival_time: datetime,
                          weather_code: int = 0,
                          has_event: bool = False,
                          max_walk_km: float = 1.5) -> dict:
    """
    Find and rank all parking lots near destination.
    Returns primary recommendation + alternatives.
    """
    hour = arrival_time.hour
    dow  = arrival_time.weekday()

    nearby = nearest_parking(dest_lat, dest_lng, top_k=10, max_km=max_walk_km)

    if not nearby:
        # Expand search radius
        nearby = nearest_parking(dest_lat, dest_lng, top_k=5, max_km=5.0)

    results = []
    for dist_km, lot in nearby:
        avail      = predict_lot_availability(lot, hour, dow, weather_code, has_event)
        walk_min   = max(1, round((dist_km / 5.0) * 60))  # 5 km/h walking
        status     = "AVAILABLE" if avail >= 40 else "LIMITED" if avail >= 15 else "FULL"

        results.append({
            "lot_id":           lot["id"],
            "name":             lot["name"],
            "zone":             lot["zone"],
            "lat":              lot["lat"],
            "lng":              lot["lng"],
            "distance_km":      dist_km,
            "walk_minutes":     walk_min,
            "availability_pct": avail,
            "capacity":         lot["capacity"],
            "status":           status,
        })

    results.sort(key=lambda x: (-x["availability_pct"], x["walk_minutes"]))

    return {
        "dest_lat":      dest_lat,
        "dest_lng":      dest_lng,
        "arrival_time":  arrival_time.strftime("%I:%M %p"),
        "day":           ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow],
        "primary":       results[0]     if results else None,
        "alternatives":  results[1:4]   if len(results) > 1 else [],
        "all_nearby":    results,
        "total_found":   len(results),
        "search_radius": max_walk_km,
    }
