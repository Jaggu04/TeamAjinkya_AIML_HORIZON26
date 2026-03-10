"""
backend/api/main.py
FastAPI — Complete Mumbai Urban Navigation API
All 25 routes, 50 parking lots, coordinate-based predictions.
Run: uvicorn backend.api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional, List
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.models.predictor        import (predict_route_full, predict_all_routes,
                                              predict_zone_summary, predict_by_coordinates,
                                              predict_congestion, cong_level)
from backend.models.parking_intelligence import predict_parking_near
from backend.models.mumbai_routes    import MUMBAI_ROUTES, MUMBAI_PARKING, nearest_routes
from backend.models.personalization  import UserProfile, init_db, collaborative_recommendation
from backend.services.departure_planner import plan_departure
from backend.services.weather_service   import get_mumbai_weather, weather_congestion_adjustment

app = FastAPI(
    title="Urban Navigation AI — Full Mumbai",
    description="Covers all 25 routes and 50 parking lots across Mumbai. Coordinate-based predictions.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

init_db()
print("Urban Navigation API v2 ready — covering all of Mumbai")
print(f"Routes: {len(MUMBAI_ROUTES)} | Parking lots: {len(MUMBAI_PARKING)}")


# ── Pydantic models ────────────────────────────────────────────────
class DepartureRequest(BaseModel):
    origin:          str   = Field(..., example="Andheri Station, Mumbai")
    destination:     str   = Field(..., example="BKC, Mumbai")
    origin_lat:      float = Field(..., example=19.1197)
    origin_lng:      float = Field(..., example=72.8463)
    dest_lat:        float = Field(..., example=19.0631)
    dest_lng:        float = Field(..., example=72.8677)
    arrival_hour:    int   = Field(..., ge=0, le=23, example=9)
    arrival_minute:  int   = Field(0, ge=0, le=59)
    distance_km:     float = Field(15.0, example=15.0)
    has_event:       bool  = Field(False)
    user_id:         Optional[str] = None

class ParkingRequest(BaseModel):
    dest_lat:     float = Field(..., example=19.0631)
    dest_lng:     float = Field(..., example=72.8677)
    arrival_hour: int   = Field(..., ge=0, le=23, example=9)
    arrival_min:  int   = Field(0, ge=0, le=59)
    weather_code: int   = Field(0, ge=0, le=3)
    has_event:    bool  = Field(False)
    max_walk_km:  float = Field(1.5, gt=0, le=5.0)

class TripLogRequest(BaseModel):
    user_id:        str
    origin:         str
    destination:    str
    route_id:       str   = "R001"
    travel_minutes: int
    congestion_pct: float
    parking_lot_id: Optional[str] = None
    rating:         Optional[int] = Field(None, ge=1, le=5)

class CreateUserRequest(BaseModel):
    user_id:  str;  name: str
    home_lat: Optional[float] = None;  home_lng: Optional[float] = None
    work_lat: Optional[float] = None;  work_lng: Optional[float] = None


# ── Helpers ────────────────────────────────────────────────────────
def _now_ctx(weather_code_override: int = None):
    now     = datetime.now()
    weather = get_mumbai_weather()
    wcode   = weather_code_override if weather_code_override is not None else weather["weather_code"]
    return now, weather, wcode


# ══════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════
@app.get("/", tags=["Health"])
def root():
    return {"status":"online","version":"2.0","routes":len(MUMBAI_ROUTES),
            "parking_lots":len(MUMBAI_PARKING),"time":datetime.now().isoformat(),"docs":"/docs"}


# ══════════════════════════════════════════════════════════════════
# WEATHER
# ══════════════════════════════════════════════════════════════════
@app.get("/api/weather", tags=["Weather"], summary="Live Mumbai weather from OpenWeatherMap")
def get_weather():
    return get_mumbai_weather()


# ══════════════════════════════════════════════════════════════════
# TRAFFIC — By Route ID
# ══════════════════════════════════════════════════════════════════
@app.get("/api/forecast/{route_id}", tags=["Traffic Forecast"],
         summary="Full forecast for a specific route (R001–R025)")
def forecast_route(route_id: str):
    if route_id not in MUMBAI_ROUTES:
        raise HTTPException(404, f"Route {route_id} not found. Valid: R001–R025")
    now, weather, wcode = _now_ctx()
    result = predict_route_full(route_id, now.hour, now.weekday(), wcode)
    result["weather"] = weather
    result["generated_at"] = now.isoformat()
    return result


@app.get("/api/forecast/all-routes", tags=["Traffic Forecast"],
         summary="Current congestion for all 25 Mumbai routes")
def forecast_all():
    now, weather, wcode = _now_ctx()
    routes = predict_all_routes(now.hour, now.weekday(), wcode)
    return {"routes": routes, "total": len(routes),
            "weather": weather, "timestamp": now.isoformat()}


@app.get("/api/forecast/zones", tags=["Traffic Forecast"],
         summary="Congestion summary per Mumbai zone")
def forecast_zones():
    now, weather, wcode = _now_ctx()
    zones = predict_zone_summary(now.hour, now.weekday(), wcode)
    return {"zones": zones, "weather": weather, "timestamp": now.isoformat()}


# ══════════════════════════════════════════════════════════════════
# TRAFFIC — By Coordinates (ANY Mumbai location)
# ══════════════════════════════════════════════════════════════════
@app.get("/api/forecast/by-location", tags=["Traffic Forecast"],
         summary="Predict traffic for ANY Mumbai coordinates")
def forecast_by_location(
    lat: float = Query(..., example=19.0631, description="Latitude"),
    lng: float = Query(..., example=72.8677, description="Longitude"),
):
    """
    Pass any Mumbai lat/lng — get traffic predictions for nearby routes.
    Works for any point in Mumbai, not just predefined routes.
    """
    now, weather, wcode = _now_ctx()
    result = predict_by_coordinates(lat, lng, now.hour, now.weekday(), wcode)
    result["weather"]      = weather
    result["generated_at"] = now.isoformat()
    return result


@app.get("/api/forecast/area-comparison", tags=["Traffic Forecast"],
         summary="Compare traffic across multiple Mumbai areas right now")
def area_comparison():
    """Compare all Mumbai zones side by side."""
    now, weather, wcode = _now_ctx()
    areas = [
        {"name":"Andheri (Western)",    "lat":19.1197,"lng":72.8463},
        {"name":"BKC (Business)",       "lat":19.0631,"lng":72.8677},
        {"name":"Dadar (Central)",      "lat":19.0178,"lng":72.8478},
        {"name":"Churchgate (South)",   "lat":18.9322,"lng":72.8264},
        {"name":"Thane",                "lat":19.1890,"lng":72.9650},
        {"name":"Vashi (Navi Mumbai)",  "lat":19.0771,"lng":73.0108},
        {"name":"Ghatkopar (East)",     "lat":19.0861,"lng":72.9081},
        {"name":"Borivali (Far West)",  "lat":19.2288,"lng":72.8558},
    ]
    results = []
    for area in areas:
        pred = predict_by_coordinates(area["lat"], area["lng"],
                                       now.hour, now.weekday(), wcode)
        results.append({
            "area":           area["name"],
            "lat":            area["lat"],
            "lng":            area["lng"],
            "congestion_pct": pred["area_congestion"],
            "level":          pred["area_level"],
            "nearest_route":  pred["primary_route"]["route_name"] if pred["primary_route"] else "N/A",
        })
    results.sort(key=lambda x: -x["congestion_pct"])
    return {"areas": results, "weather": weather, "timestamp": now.isoformat()}


# ══════════════════════════════════════════════════════════════════
# DEPARTURE PLANNING
# ══════════════════════════════════════════════════════════════════
@app.post("/api/departure/plan", tags=["Departure Planning"],
          summary="Arrive-by planner — given a target arrival time, returns ideal departure")
def plan(req: DepartureRequest):
    """
    Core use-case: User says "I want to reach BKC by 10:00 AM from Andheri."

    Response root contains flat card fields ready for the UI:
      • recommended_departure  — "9:15 AM"
      • predicted_travel_time  — "40–45 min"
      • traffic_level          — "Moderate"
      • predicted_eta          — "9:58 AM"
      • arrive_by_target       — "10:00 AM"
      • summary                — plain-English sentence

    Also contains full ranked window list in `recommendations` / `all_windows`.
    """
    now, weather, wcode = _now_ctx()

    # Build the target arrival datetime
    arrival = now.replace(hour=req.arrival_hour, minute=req.arrival_minute,
                          second=0, microsecond=0)
    if arrival <= now:          # arrival time is in the past → push to tomorrow
        arrival += timedelta(days=1)

    # Live parking probability at destination at arrival time
    park         = predict_parking_near(req.dest_lat, req.dest_lng, arrival,
                                        wcode, int(req.has_event), 1.5)
    parking_prob = (park["primary"]["availability_pct"] / 100
                    if park.get("primary") else 0.70)

    # Run the arrive-by planner
    result = plan_departure(
        origin_lat=req.origin_lat,   origin_lng=req.origin_lng,
        dest_lat=req.dest_lat,       dest_lng=req.dest_lng,
        origin=req.origin,           destination=req.destination,
        desired_arrival=arrival,
        distance_km=req.distance_km,
        parking_prob=parking_prob,
        weather_code=wcode,
        has_event=req.has_event,
    )

    # Attach live context
    result["weather"]         = weather
    result["parking_preview"] = park.get("primary")
    weather_labels = {
        0: "No weather impact",
        1: "Light rain — adding ~12% congestion",
        2: "Heavy rain/storm — adding ~28% congestion",
        3: "Fog/mist — adding ~10% congestion",
    }
    result["weather_impact"] = weather_labels.get(wcode, "")

    # Personalization (optional)
    if req.user_id:
        try:
            u = UserProfile(req.user_id)
            result["personalization"] = {
                "insight":       u.get_personalized_insight(now.weekday()),
                "collaborative": collaborative_recommendation(req.user_id, now.weekday()),
            }
        except Exception:
            pass

    return result


# ══════════════════════════════════════════════════════════════════
# PARKING
# ══════════════════════════════════════════════════════════════════
@app.post("/api/parking/predict", tags=["Parking Intelligence"],
          summary="Predict parking availability near any Mumbai coordinates")
def parking_predict(req: ParkingRequest):
    """
    Pass destination coordinates → get availability predictions for nearby lots.
    Covers all 50 Mumbai parking lots.
    """
    arrival = datetime.now().replace(hour=req.arrival_hour, minute=req.arrival_min, second=0)
    return predict_parking_near(req.dest_lat, req.dest_lng, arrival,
                                 req.weather_code, req.has_event, req.max_walk_km)


@app.get("/api/parking/lots", tags=["Parking Intelligence"],
         summary="List all 50 Mumbai parking lots")
def parking_lots(zone: Optional[str] = None):
    lots = MUMBAI_PARKING
    if zone:
        lots = [l for l in lots if l["zone"].lower() == zone.lower()]
    return {"lots": lots, "total": len(lots),
            "zones": list(set(l["zone"] for l in MUMBAI_PARKING))}


@app.get("/api/parking/by-location", tags=["Parking Intelligence"],
         summary="Find parking near any Mumbai coordinates")
def parking_by_location(
    lat:          float = Query(..., example=19.0631),
    lng:          float = Query(..., example=72.8677),
    arrival_hour: int   = Query(9,   example=9),
    max_walk_km:  float = Query(1.5, example=1.5),
    has_event:    bool  = Query(False),
):
    _, weather, wcode = _now_ctx()
    arrival = datetime.now().replace(hour=arrival_hour, minute=0, second=0)
    return predict_parking_near(lat, lng, arrival, wcode, has_event, max_walk_km)


# ══════════════════════════════════════════════════════════════════
# PERSONALIZATION
# ══════════════════════════════════════════════════════════════════
@app.post("/api/users/create", tags=["Personalization"])
def create_user(req: CreateUserRequest):
    u = UserProfile(req.user_id)
    return {"status": "created", "user": u.create_user(req.name, req.home_lat, req.home_lng, req.work_lat, req.work_lng)}

@app.get("/api/users/{user_id}/profile", tags=["Personalization"])
def user_profile(user_id: str):
    u = UserProfile(user_id)
    user = u.get_user()
    if not user:
        raise HTTPException(404, f"User '{user_id}' not found")
    return {"user": user, "preferences": u.get_preferences(),
            "insight": u.get_personalized_insight(), "recent_trips": u.get_trip_history(5)}

@app.post("/api/users/log-trip", tags=["Personalization"])
def log_trip(req: TripLogRequest):
    u   = UserProfile(req.user_id)
    now = datetime.now()
    tid = u.log_trip(req.origin, req.destination, req.route_id,
        now - timedelta(minutes=req.travel_minutes), now,
        req.travel_minutes, req.congestion_pct, req.parking_lot_id, req.rating)
    return {"status": "logged", "trip_id": tid, "new_insight": u.get_personalized_insight()}

@app.get("/api/users/{user_id}/similar", tags=["Personalization"])
def similar_users(user_id: str):
    from backend.models.personalization import find_similar_users
    return {"similar_users": find_similar_users(user_id),
            "recommendation": collaborative_recommendation(user_id, datetime.now().weekday())}


# ══════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════
@app.get("/api/routes", tags=["Utility"], summary="List all 25 Mumbai routes")
def list_routes(zone: Optional[str] = None):
    routes = [{"route_id": rid, **info} for rid, info in MUMBAI_ROUTES.items()]
    if zone:
        routes = [r for r in routes if r["zone"].lower() == zone.lower()]
    zones = list(set(r["zone"] for r in routes))
    return {"routes": routes, "total": len(routes), "zones": zones}


@app.get("/api/nearest-routes", tags=["Utility"],
         summary="Find nearest routes to any Mumbai coordinates")
def get_nearest_routes(
    lat: float = Query(..., example=19.0631),
    lng: float = Query(..., example=72.8677),
    k:   int   = Query(3, ge=1, le=10),
):
    nearby = nearest_routes(lat, lng, top_k=k)
    now, weather, wcode = _now_ctx()
    return {
        "lat": lat, "lng": lng,
        "nearest": [
            {**info, "route_id": rid,
             "current_congestion": predict_congestion(rid, now.hour, now.weekday(), wcode),
             "level": cong_level(predict_congestion(rid, now.hour, now.weekday(), wcode))}
            for rid, info in nearby
        ]
    }