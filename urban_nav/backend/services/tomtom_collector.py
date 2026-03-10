"""
backend/services/tomtom_collector.py
Fetches real-time traffic data from TomTom API and stores it.
Run alongside training: python -m backend.services.tomtom_collector
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TOMTOM_API_KEY")

# ─── Mumbai route coordinates ─────────────────────────────────────
MUMBAI_ROUTES = {
    "R001": {"name": "Western Express Highway",  "lat": 19.1136, "lng": 72.8697},
    "R002": {"name": "Eastern Express Highway",  "lat": 19.0728, "lng": 72.8826},
    "R003": {"name": "Bandra-Worli Sea Link",    "lat": 19.0386, "lng": 72.8178},
    "R004": {"name": "LBS Marg Ghatkopar",       "lat": 19.0896, "lng": 72.9086},
    "R005": {"name": "SV Road Andheri",          "lat": 19.1197, "lng": 72.8464},
}

HISTORY_FILE = "data/processed/tomtom_live_history.csv"


def fetch_flow(route_id: str, lat: float, lng: float) -> dict | None:
    """Fetch flow segment data from TomTom for a single point."""
    url = (
        f"https://api.tomtom.com/traffic/services/4/"
        f"flowSegmentData/absolute/10/json"
        f"?point={lat},{lng}&key={API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        flow = resp.json()["flowSegmentData"]

        current_speed  = flow["currentSpeed"]
        freeflow_speed = flow["freeFlowSpeed"]
        confidence     = flow["confidence"]
        congestion     = max(0, min(100, round((1 - current_speed / freeflow_speed) * 100, 2)))

        now = datetime.now()
        return {
            "timestamp":      now.isoformat(),
            "route_id":       route_id,
            "hour":           now.hour,
            "day_of_week":    now.weekday(),
            "current_speed":  current_speed,
            "freeflow_speed": freeflow_speed,
            "congestion_pct": congestion,
            "confidence":     confidence,
        }
    except Exception as e:
        print(f"  ❌ TomTom error for {route_id}: {e}")
        return None


def fetch_incidents(lat: float, lng: float, radius: float = 0.05) -> dict:
    """Fetch traffic incidents near a point."""
    url = (
        f"https://api.tomtom.com/traffic/services/5/incidentDetails"
        f"?bbox={lng-radius},{lat-radius},{lng+radius},{lat+radius}"
        f"&key={API_KEY}"
    )
    try:
        resp      = requests.get(url, timeout=10)
        incidents = resp.json().get("incidents", [])
        serious   = [i for i in incidents
                     if i.get("properties", {}).get("magnitudeOfDelay", 0) >= 3]
        return {"total": len(incidents), "serious": len(serious), "has_major_jam": len(serious) > 0}
    except Exception:
        return {"total": 0, "serious": 0, "has_major_jam": False}


def collect_all_routes() -> list[dict]:
    """Collect live data for all Mumbai routes."""
    rows = []
    for route_id, info in MUMBAI_ROUTES.items():
        print(f"  📡 Fetching {info['name']}...")
        row = fetch_flow(route_id, info["lat"], info["lng"])
        if row:
            incidents        = fetch_incidents(info["lat"], info["lng"])
            row["has_event"] = int(incidents["has_major_jam"])
            rows.append(row)
            print(f"     ✅ Congestion: {row['congestion_pct']}% | Speed: {row['current_speed']} km/h")
        time.sleep(0.4)   # respect rate limit
    return rows


def append_to_history(rows: list[dict]):
    """Append new rows to the CSV history file."""
    df_new = pd.DataFrame(rows)
    try:
        df_old = pd.read_csv(HISTORY_FILE)
        df     = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df = df_new
    df.to_csv(HISTORY_FILE, index=False)
    print(f"  💾 Saved. Total records: {len(df)}")


def run_collector(interval_minutes: int = 30):
    """Main loop — collects data every N minutes."""
    import schedule
    print(f"🚦 TomTom Collector started — fetching every {interval_minutes} min")

    def job():
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collecting traffic data...")
        rows = collect_all_routes()
        if rows:
            append_to_history(rows)

    job()   # run immediately on start
    schedule.every(interval_minutes).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    run_collector(interval_minutes=30)
