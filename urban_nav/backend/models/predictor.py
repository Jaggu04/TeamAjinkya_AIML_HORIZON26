"""
backend/models/predictor.py
Core prediction engine — calibrated for real Mumbai traffic ranges.
Returns realistic values for all 25 routes.
"""

import numpy as np
from datetime import datetime
from backend.models.mumbai_routes import MUMBAI_ROUTES, nearest_routes

# ── Mumbai hourly base congestion (0–1 scale, tuned to reality) ────
# These represent MEDIUM roads at average conditions
HOURLY_BASE = {
    0: 0.04, 1: 0.03, 2: 0.02, 3: 0.03, 4: 0.06,
    5: 0.14, 6: 0.38, 7: 0.62, 8: 0.78, 9: 0.72,
   10: 0.55, 11: 0.48, 12: 0.52, 13: 0.55, 14: 0.51,
   15: 0.58, 16: 0.68, 17: 0.80, 18: 0.82, 19: 0.74,
   20: 0.58, 21: 0.40, 22: 0.24, 23: 0.11,
}

# Day-of-week adjustment (additive, not multiplicative, to prevent blowup)
DAY_ADJUST = {0: +8, 1: +4, 2: 0, 3: 0, 4: +3, 5: -18, 6: -28}  # % points

# Weather additive adjustment (% points)
WEATHER_ADD = {0: 0, 1: +12, 2: +28, 3: +10}

# Zone additive adjustment (% points on top of base)
ZONE_ADD = {
    "Western Suburbs":  +6,
    "Eastern Suburbs":  +4,
    "Central Connector":+2,
    "Central Mumbai":   +9,
    "BKC":              +10,
    "South Mumbai":     +7,
    "Navi Mumbai":      -8,
    "Thane":            -3,
    "Outskirts":        -14,
}

# Road type scalar (applied to base BEFORE adding zone/day)
# Only scales the base traffic, keeping total in realistic range
TYPE_SCALE = {
    "expressway": 0.45,
    "highway":    0.75,
    "arterial":   1.00,
    "local":      0.95,
}

# Per-route personality offsets (realistic Mumbai knowledge, % points)
ROUTE_OFFSET = {
    "R001": +3,   # WEH always bad
    "R002": +2,   # SV Road
    "R003": +1,
    "R004": +2,
    "R005": -2,   # JVLR free
    "R006": +2,
    "R007": +3,   # LBS Marg congested
    "R008": +1,
    "R009": -1,
    "R010": -2,
    "R011": -10,  # Sea Link fast
    "R012": +5,   # Dadar worst
    "R013": +5,
    "R014": +6,   # BKC internal
    "R015": +2,
    "R016": -8,   # Eastern Freeway fast
    "R017": +2,
    "R018": +2,
    "R019": +3,
    "R020": +5,
    "R021": -7,   # Palm Beach fast
    "R022": -5,
    "R023": -3,
    "R024": -12,  # Panvel open
    "R025": -4,
}


def predict_congestion(route_id: str, hour: int, dow: int,
                       weather_code: int = 0, has_event: bool = False,
                       horizon_hours: int = 0) -> float:
    """
    Predict congestion % for a specific route at a specific time.
    Returns realistic values for all 25 Mumbai routes.
    """
    route = MUMBAI_ROUTES.get(route_id)
    if not route:
        return 50.0

    target_hour = (hour + horizon_hours) % 24
    zone        = route["zone"]
    rtype       = route["type"]

    # Step 1: base value for this hour (0–1)
    base = HOURLY_BASE[target_hour]

    # Step 2: apply road type scaling (stays 0–1 range mostly)
    scaled = base * TYPE_SCALE[rtype]

    # Step 3: convert to % and add adjustments
    pct = scaled * 100
    pct += ZONE_ADD.get(zone, 0)
    pct += DAY_ADJUST.get(dow, 0)
    pct += WEATHER_ADD.get(weather_code, 0)
    pct += ROUTE_OFFSET.get(route_id, 0)
    pct += 30 if has_event and zone in ["BKC", "South Mumbai", "Central Mumbai"] else 0

    # Step 4: future horizon — traffic regresses toward ~50% mean
    if horizon_hours > 0:
        future_base = HOURLY_BASE[target_hour] * TYPE_SCALE[rtype] * 100
        future_base += ZONE_ADD.get(zone, 0) + DAY_ADJUST.get(dow, 0)
        future_base += ROUTE_OFFSET.get(route_id, 0)
        pct = future_base

    # Step 5: small deterministic noise so routes aren't identical
    seed  = abs(hash(route_id + str(target_hour) + str(dow))) % 200
    noise = (seed - 100) / 25.0  # ±4 % points
    pct  += noise

    return round(float(np.clip(pct, 2.0, 97.0)), 1)


def cong_level(c: float) -> str:
    if c < 30:  return "LOW"
    if c < 55:  return "MODERATE"
    if c < 72:  return "HIGH"
    return "SEVERE"


def predict_route_full(route_id: str, hour: int, dow: int,
                       weather_code: int = 0, has_event: bool = False) -> dict:
    """Full prediction: current + 1h/3h/6h forecasts + hourly chart."""
    route  = MUMBAI_ROUTES.get(route_id, {})
    now_c  = predict_congestion(route_id, hour, dow, weather_code, has_event, 0)
    h1     = predict_congestion(route_id, hour, dow, weather_code, has_event, 1)
    h3     = predict_congestion(route_id, hour, dow, weather_code, has_event, 3)
    h6     = predict_congestion(route_id, hour, dow, weather_code, has_event, 6)

    hourly = []
    for offset in range(8):
        c = predict_congestion(route_id, hour, dow, weather_code, has_event, offset)
        t = datetime.now()
        from datetime import timedelta
        t = (datetime.now() + timedelta(hours=offset))
        hourly.append({
            "time":           t.strftime("%I:%M %p"),
            "hour_offset":    offset,
            "congestion_pct": c,
            "level":          cong_level(c),
        })

    base_speed = route.get("base_speed", 40)
    speed_now  = round(base_speed * (1 - now_c / 100 * 0.82), 1)
    speed_now  = max(3.0, speed_now)

    now_h = datetime.now().hour
    trend = "STABLE"
    if h1 > now_c + 5:   trend = "WORSENING"
    elif h1 < now_c - 5: trend = "IMPROVING"

    return {
        "route_id":     route_id,
        "route_name":   route.get("name", route_id),
        "zone":         route.get("zone", "Unknown"),
        "type":         route.get("type", "arterial"),
        "lat":          route.get("lat", 0),
        "lng":          route.get("lng", 0),
        "current": {
            "congestion_pct": now_c,
            "level":          cong_level(now_c),
            "speed_kmh":      speed_now,
        },
        "forecast_1h":  {"congestion_pct": h1, "level": cong_level(h1)},
        "forecast_3h":  {"congestion_pct": h3, "level": cong_level(h3)},
        "forecast_6h":  {"congestion_pct": h6, "level": cong_level(h6)},
        "hourly_chart": hourly,
        "trend":        trend,
    }


def predict_all_routes(hour: int, dow: int,
                       weather_code: int = 0, has_event: bool = False) -> list:
    """Predict congestion for all 25 Mumbai routes, sorted worst first."""
    results = []
    for rid, r in MUMBAI_ROUTES.items():
        c = predict_congestion(rid, hour, dow, weather_code, has_event)
        results.append({
            "route_id":       rid,
            "name":           r["name"],
            "zone":           r["zone"],
            "type":           r["type"],
            "lat":            r["lat"],
            "lng":            r["lng"],
            "congestion_pct": c,
            "level":          cong_level(c),
            "speed_kmh":      round(max(3, r["base_speed"] * (1 - c/100*0.82)), 1),
        })
    results.sort(key=lambda x: -x["congestion_pct"])
    return results


def predict_zone_summary(hour: int, dow: int, weather_code: int = 0) -> list:
    """Average congestion per Mumbai zone."""
    from collections import defaultdict
    zd = defaultdict(list)
    for rid, r in MUMBAI_ROUTES.items():
        c = predict_congestion(rid, hour, dow, weather_code)
        zd[r["zone"]].append(c)
    out = []
    for zone, vals in sorted(zd.items()):
        avg = round(float(np.mean(vals)), 1)
        out.append({
            "zone":           zone,
            "avg_congestion": avg,
            "level":          cong_level(avg),
            "route_count":    len(vals),
            "worst":          round(max(vals), 1),
            "best":           round(min(vals), 1),
        })
    out.sort(key=lambda x: -x["avg_congestion"])
    return out


def predict_by_coordinates(lat: float, lng: float, hour: int, dow: int,
                            weather_code: int = 0, has_event: bool = False) -> dict:
    """Predict traffic for any Mumbai coordinates via nearest routes."""
    nearby = nearest_routes(lat, lng, top_k=3)
    preds  = [predict_route_full(rid, hour, dow, weather_code, has_event)
              for rid, _ in nearby]
    avg = round(float(np.mean([p["current"]["congestion_pct"] for p in preds])), 1)
    return {
        "lat": lat, "lng": lng,
        "area_congestion": avg,
        "area_level":      cong_level(avg),
        "nearest_routes":  preds,
        "primary_route":   preds[0] if preds else None,
    }
