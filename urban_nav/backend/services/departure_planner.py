"""
backend/services/departure_planner.py
MODULE 2 — Arrive-By Departure Planner

Core contract:
    User says "I want to reach BKC by 10:00 AM from Andheri."
    System answers "Leave at 9:15 AM. Travel time: 40–45 min. Traffic: Moderate."

How it works:
    1.  Find the LATEST SAFE DEPARTURE — walk backwards from arrival_time in
        15-min steps, predicting congestion at each candidate departure HOUR.
        The first candidate whose ETA <= arrival_time is the latest you can leave.

    2.  Scan 2 hours back from that latest-safe point in 15-min steps.
        Each step uses congestion predicted at THAT departure hour specifically.

    3.  Score every window:
            Congestion at departure    45%
            Travel efficiency          35%
            Parking availability       20%
        +5  bonus  if ETA is 5–20 min before arrival (comfortable buffer)
        -3  penalty if ETA is 0–5 min before (cutting it close)
        0 / discard if ETA > arrival + 5 min (late)

    4.  Best-scored window = PRIMARY recommendation.
        Next 2 = alternatives.

    5.  Response root includes flat card fields that map directly to the UI:
            recommended_departure  → "9:15 AM"
            predicted_travel_time  → "40–45 min"
            traffic_level          → "Moderate"
            predicted_eta          → "9:58 AM"
            arrive_by_target       → "10:00 AM"
"""

import numpy as np
from datetime import datetime, timedelta
from backend.models.predictor import predict_congestion, cong_level
from backend.models.mumbai_routes import nearest_routes

W_CONGESTION      = 0.45
W_TRAVEL          = 0.35
W_PARKING         = 0.20
STEP_MINUTES      = 15
LATE_GRACE_MIN    = 5      # discard if ETA is more than 5 min late
EARLY_GRACE_MIN   = 30     # discard if ETA is more than 30 min early
BASE_SPEED_KMH    = 45.0


# ── Travel time estimate ───────────────────────────────────────────

def _travel_estimate(congestion_pct: float, distance_km: float) -> dict:
    adj_speed   = max(12.0, BASE_SPEED_KMH * (1 - congestion_pct / 100 * 0.55))
    normal_min  = max(1, round((distance_km / BASE_SPEED_KMH) * 60))
    traffic_min = max(1, round((distance_km / adj_speed) * 60))
    delay_min   = max(0, traffic_min - normal_min)
    lo = max(normal_min, round((traffic_min * 0.90) / 5) * 5)
    hi = round((traffic_min * 1.10) / 5) * 5
    return {
        "normal_minutes":  normal_min,
        "traffic_minutes": traffic_min,
        "delay_minutes":   delay_min,
        "travel_range":    f"{lo}–{hi} min" if lo != hi else f"{lo} min",
    }


# ── Scoring ────────────────────────────────────────────────────────

def _score(congestion, traffic_min, normal_min, parking_prob, minutes_early):
    # Discard: more than 5 min late OR more than 30 min early
    if minutes_early < -LATE_GRACE_MIN:
        return 0.0
    if minutes_early > EARLY_GRACE_MIN:
        return 0.0
    cong_s   = (1 - congestion / 100) * 100
    travel_s = max(0.0, (1 - (traffic_min / max(normal_min, 1) - 1) / 2.0)) * 100
    base     = W_CONGESTION * cong_s + W_TRAVEL * travel_s + W_PARKING * parking_prob * 100
    # Sweet spot: 5-15 min early (comfortable buffer)
    if 5 <= minutes_early <= 15:    base += 8.0
    # Acceptable: 15-30 min early
    elif 15 < minutes_early <= 30:  base += 2.0
    # Cutting it close: 0-5 min early
    elif 0 <= minutes_early < 5:    base -= 3.0
    return round(base, 1)


# ── Labels ─────────────────────────────────────────────────────────

def _traffic_label(c):
    if c < 30:  return "Light"
    if c < 55:  return "Moderate"
    if c < 72:  return "Heavy"
    return "Severe"

def _window_label(score, minutes_early):
    if minutes_early < -LATE_GRACE_MIN:  return "LATE"
    if minutes_early > EARLY_GRACE_MIN:  return "TOO EARLY"
    if minutes_early < 0:                return "RISKY"
    if score >= 78:  return "BEST"
    if score >= 62:  return "GOOD"
    if score >= 48:  return "OK"
    return "AVOID"


# ── Latest safe departure ──────────────────────────────────────────

def _latest_safe_departure(route_id, desired_arrival, distance_km,
                            weather_code, has_event):
    """
    Walk backwards from arrival in 15-min steps.
    Return the LATEST departure time whose ETA still lands within grace period.
    """
    for back_min in range(0, 180, STEP_MINUTES):
        candidate = desired_arrival - timedelta(minutes=back_min)
        cong      = predict_congestion(route_id, candidate.hour, candidate.weekday(),
                                       weather_code, has_event)
        travel    = _travel_estimate(cong, distance_km)
        eta       = candidate + timedelta(minutes=travel["traffic_minutes"])
        if eta <= desired_arrival + timedelta(minutes=LATE_GRACE_MIN):
            return candidate
    return desired_arrival - timedelta(minutes=30)


# ── Main planner ───────────────────────────────────────────────────

def plan_departure(origin_lat, origin_lng, dest_lat, dest_lng,
                   origin, destination, desired_arrival,
                   distance_km=15.0, parking_prob=0.70,
                   weather_code=0, has_event=False):
    """
    Arrive-by planner.  Returns a response whose root-level fields map
    directly to the UI card (recommended_departure, predicted_travel_time,
    traffic_level, predicted_eta, arrive_by_target).
    """
    nearby   = nearest_routes(origin_lat, origin_lng, top_k=1)
    route_id = nearby[0][0] if nearby else "R001"
    now      = datetime.now().replace(second=0, microsecond=0)

    # 1. Latest safe departure
    latest_safe = _latest_safe_departure(route_id, desired_arrival,
                                          distance_km, weather_code, has_event)

    # 2. Scan window anchored to arrival time.
    #    We want ETAs between (arrive_by - 30min) and (arrive_by + 5min).
    #    Since travel time is roughly fixed, work backwards:
    #    earliest departure that still lands ETA within 30 min of arrival.
    #    Use latest_safe travel time as proxy for scan width.
    rough_cong   = predict_congestion(route_id, desired_arrival.hour,
                                      desired_arrival.weekday(), weather_code, has_event)
    rough_travel = _travel_estimate(rough_cong, distance_km)
    rough_min    = rough_travel["traffic_minutes"]

    # scan_end   = latest departure to still arrive on time
    # scan_start = departure that would arrive exactly 30 min early
    #            = latest_safe - 30 min  (same travel time, just leave earlier)
    scan_end   = latest_safe
    scan_start = latest_safe - timedelta(minutes=EARLY_GRACE_MIN)

    if scan_start < now:
        scan_start = now
    if scan_end < now:
        scan_start = now
        scan_end   = now + timedelta(minutes=EARLY_GRACE_MIN)

    # 3. Evaluate each 15-min slot
    windows = []
    cursor  = scan_start
    while cursor <= scan_end + timedelta(seconds=30):
        cong   = predict_congestion(route_id, cursor.hour, cursor.weekday(),
                                    weather_code, has_event)
        travel = _travel_estimate(cong, distance_km)
        eta_dt = cursor + timedelta(minutes=travel["traffic_minutes"])
        m_early = (desired_arrival - eta_dt).total_seconds() / 60
        score   = _score(cong, travel["traffic_minutes"], travel["normal_minutes"],
                         parking_prob, m_early)

        windows.append({
            "depart_time":      cursor.strftime("%I:%M %p"),
            "depart_iso":       cursor.isoformat(),
            "eta":              eta_dt.strftime("%I:%M %p"),
            "eta_iso":          eta_dt.isoformat(),
            "minutes_early":    round(m_early),
            "congestion_pct":   cong,
            "congestion_level": cong_level(cong),
            "traffic_level":    _traffic_label(cong),
            "travel_minutes":   travel["traffic_minutes"],
            "travel_range":     travel["travel_range"],
            "delay_minutes":    travel["delay_minutes"],
            "normal_minutes":   travel["normal_minutes"],
            "score":            score,
            "confidence":       min(97, max(50, int(score))),
            "label":            _window_label(score, m_early),
            "route_used":       route_id,
        })
        cursor += timedelta(minutes=STEP_MINUTES)

    # 4. Filter & rank
    viable = sorted([w for w in windows if w["score"] > 0], key=lambda w: -w["score"])
    if not viable:
        viable = sorted(windows, key=lambda w: -w["score"])
    for i, w in enumerate(viable):
        w["rank"] = i + 1

    primary      = viable[0] if viable else None
    alternatives = viable[1:3]

    # 5. Build card + response
    card = _build_card(primary, desired_arrival, origin, destination) if primary else {}

    return {
        # Flat card fields — map directly to UI requirement output
        **card,

        # Structured fields
        "origin":                  origin,
        "destination":             destination,
        "arrive_by":               desired_arrival.strftime("%I:%M %p"),
        "arrive_by_iso":           desired_arrival.isoformat(),
        "latest_safe_departure":   latest_safe.strftime("%I:%M %p"),
        "distance_km":             distance_km,
        "route_used":              route_id,
        "generated_at":            now.isoformat(),
        "primary":                 primary,
        "alternatives":            alternatives,
        "recommendations":         ([primary] + alternatives) if primary else [],
        "all_windows":             viable,
        "best_score":              primary["score"] if primary else 0,
        "scan_info": {
            "scan_from":           scan_start.strftime("%I:%M %p"),
            "scan_to":             scan_end.strftime("%I:%M %p"),
            "windows_evaluated":   len(windows),
            "viable_windows":      len(viable),
            "latest_safe_depart":  latest_safe.strftime("%I:%M %p"),
        },
    }


def _build_card(primary, desired_arrival, origin, destination):
    """Flat fields that match the functional requirement UI card exactly."""
    m = primary["minutes_early"]
    arrival_note = (
        f"Arriving {m} min early" if m > 0
        else "Arriving exactly on time" if m == 0
        else f"Arriving {abs(m)} min late"
    )
    return {
        "recommended_departure": primary["depart_time"],
        "predicted_travel_time": primary["travel_range"],
        "traffic_level":         primary["traffic_level"],
        "predicted_eta":         primary["eta"],
        "arrive_by_target":      desired_arrival.strftime("%I:%M %p"),
        "arrival_note":          arrival_note,
        "confidence_pct":        primary["confidence"],
        "summary": (
            f"Leave {origin} at {primary['depart_time']}. "
            f"Expected travel time is {primary['travel_range']} with "
            f"{primary['traffic_level'].lower()} traffic "
            f"({primary['congestion_pct']}% congestion). "
            f"You will reach {destination} around {primary['eta']} — "
            f"{arrival_note.lower()} your {desired_arrival.strftime('%I:%M %p')} target."
        ),
    }