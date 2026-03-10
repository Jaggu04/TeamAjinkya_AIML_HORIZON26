"""
data/generate_synthetic_data.py
Generates 60-day synthetic traffic dataset for ALL Mumbai zones.
Covers 25 routes across every major area of Mumbai.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Complete Mumbai Road Network (25 routes) ──────────────────────
ROUTES = {
    # Western Suburbs
    "R001": {"name": "Western Express Highway - Dahisar to Bandra",   "zone": "Western Suburbs",  "base_speed": 60, "type": "highway"},
    "R002": {"name": "SV Road - Dahisar to Bandra",                   "zone": "Western Suburbs",  "base_speed": 35, "type": "arterial"},
    "R003": {"name": "Link Road - Malad to Andheri",                  "zone": "Western Suburbs",  "base_speed": 30, "type": "arterial"},
    "R004": {"name": "Andheri-Kurla Road",                            "zone": "Western Suburbs",  "base_speed": 28, "type": "arterial"},
    "R005": {"name": "Jogeshwari-Vikhroli Link Road (JVLR)",          "zone": "Central Connector","base_speed": 45, "type": "highway"},
    # Eastern Suburbs
    "R006": {"name": "Eastern Express Highway - Thane to Sion",       "zone": "Eastern Suburbs",  "base_speed": 55, "type": "highway"},
    "R007": {"name": "LBS Marg - Thane to Kurla",                     "zone": "Eastern Suburbs",  "base_speed": 32, "type": "arterial"},
    "R008": {"name": "Ghodbunder Road - Thane",                       "zone": "Thane",             "base_speed": 40, "type": "arterial"},
    "R009": {"name": "Mulund-Airoli Road",                            "zone": "Eastern Suburbs",  "base_speed": 38, "type": "arterial"},
    "R010": {"name": "Sion-Panvel Highway",                           "zone": "Eastern Suburbs",  "base_speed": 50, "type": "highway"},
    # Central Mumbai
    "R011": {"name": "Bandra-Worli Sea Link",                         "zone": "Central Mumbai",   "base_speed": 80, "type": "expressway"},
    "R012": {"name": "Western Railway Parallel Road - Dadar",         "zone": "Central Mumbai",   "base_speed": 25, "type": "local"},
    "R013": {"name": "Dr Ambedkar Road - Dadar to Parel",             "zone": "Central Mumbai",   "base_speed": 22, "type": "local"},
    "R014": {"name": "BKC Internal Roads",                            "zone": "BKC",              "base_speed": 20, "type": "local"},
    "R015": {"name": "Kurla-Chembur Road",                            "zone": "Eastern Suburbs",  "base_speed": 26, "type": "local"},
    # South Mumbai
    "R016": {"name": "Eastern Freeway - Chembur to Fort",             "zone": "South Mumbai",     "base_speed": 70, "type": "expressway"},
    "R017": {"name": "Marine Drive (Netaji Subhash Chandra Bose Road)","zone": "South Mumbai",     "base_speed": 45, "type": "arterial"},
    "R018": {"name": "P D Mello Road - Masjid to Sion",              "zone": "South Mumbai",     "base_speed": 30, "type": "arterial"},
    "R019": {"name": "Peddar Road - Kemps Corner to Worli",           "zone": "South Mumbai",     "base_speed": 24, "type": "local"},
    "R020": {"name": "SG Barve Marg - Churchgate to CST",            "zone": "South Mumbai",     "base_speed": 18, "type": "local"},
    # Navi Mumbai & Outskirts
    "R021": {"name": "Palm Beach Road - Vashi to Belapur",            "zone": "Navi Mumbai",      "base_speed": 65, "type": "highway"},
    "R022": {"name": "Trans-Harbour Link Approach Road",              "zone": "Navi Mumbai",      "base_speed": 55, "type": "highway"},
    "R023": {"name": "Thane-Belapur Road",                            "zone": "Navi Mumbai",      "base_speed": 42, "type": "arterial"},
    "R024": {"name": "Khopoli Road - Panvel",                         "zone": "Outskirts",        "base_speed": 58, "type": "highway"},
    "R025": {"name": "Airoli Bridge - Thane to Navi Mumbai",          "zone": "Navi Mumbai",      "base_speed": 50, "type": "highway"},
}

# Zone-specific congestion multipliers (some zones are worse than others)
ZONE_MULT = {
    "Western Suburbs":  1.15,  # heavy residential + commercial traffic
    "Eastern Suburbs":  1.08,
    "Central Connector":1.05,
    "Central Mumbai":   1.20,  # worst congestion
    "BKC":              1.25,  # business district, peak office hours worst
    "South Mumbai":     1.10,
    "Navi Mumbai":      0.80,  # less congested
    "Thane":            0.90,
    "Outskirts":        0.60,
}

# Road type multipliers
TYPE_MULT = {
    "expressway": 0.50,  # less congestion on expressways
    "highway":    0.75,
    "arterial":   1.00,
    "local":      1.30,  # worst congestion on local roads
}

# Hour-of-day base congestion pattern (Mumbai-specific)
HOURLY_PATTERN = {
    0: 0.04, 1: 0.03, 2: 0.02, 3: 0.03, 4: 0.07,
    5: 0.18, 6: 0.45, 7: 0.78, 8: 0.92, 9: 0.88,
   10: 0.65, 11: 0.58, 12: 0.62, 13: 0.66, 14: 0.63,
   15: 0.68, 16: 0.82, 17: 0.94, 18: 0.97, 19: 0.90,
   20: 0.72, 21: 0.52, 22: 0.32, 23: 0.14,
}

DAY_MULT    = {0:1.12, 1:1.08, 2:1.02, 3:1.00, 4:1.06, 5:0.68, 6:0.48}
WEATHER_EFF = {"clear":0.00, "rain":0.22, "heavy_rain":0.42, "fog":0.18}
WEATHER_P   = [0.58, 0.27, 0.10, 0.05]
WEATHERS    = ["clear", "rain", "heavy_rain", "fog"]

# Mumbai-specific events
EVENTS = [
    {"name": "IPL Match - Wankhede",    "routes_affected": ["R011","R012","R017","R020"], "bump": 0.35},
    {"name": "Ganpati Festival",         "routes_affected": ["R016","R017","R018","R019","R020"], "bump": 0.45},
    {"name": "Republic Day Parade",      "routes_affected": ["R017","R020"], "bump": 0.30},
    {"name": "Concert - MMRDA Grounds",  "routes_affected": ["R014","R011"], "bump": 0.40},
    {"name": "Koli Festival - Versova",  "routes_affected": ["R002","R003"], "bump": 0.25},
]


def generate_dataset(days: int = 60) -> pd.DataFrame:
    rng     = np.random.default_rng(42)
    records = []
    start   = datetime.now() - timedelta(days=days)

    for day_offset in range(days):
        current_day  = start + timedelta(days=day_offset)
        dow          = current_day.weekday()
        weather      = rng.choice(WEATHERS, p=WEATHER_P)
        w_effect     = WEATHER_EFF[weather]
        # Random event (10% chance per day)
        has_event    = rng.random() < 0.10
        event_info   = rng.choice(EVENTS) if has_event else None
        event_hour   = rng.integers(17, 22) if has_event else -1

        for hour in range(24):
            for route_id, route_info in ROUTES.items():
                zone       = route_info["zone"]
                rtype      = route_info["type"]
                base_speed = route_info["base_speed"]

                base_cong  = HOURLY_PATTERN[hour]
                day_factor = DAY_MULT[dow]
                zone_mult  = ZONE_MULT[zone]
                type_mult  = TYPE_MULT[rtype]
                noise      = rng.normal(0, 0.04)

                # Event impact (only on affected routes)
                event_bump = 0.0
                if has_event and event_info:
                    if route_id in event_info["routes_affected"]:
                        if abs(hour - event_hour) <= 2:
                            event_bump = event_info["bump"]

                congestion = (base_cong * day_factor * zone_mult * type_mult
                              + w_effect + event_bump + noise)
                congestion = float(np.clip(congestion, 0.0, 1.0))

                curr_speed = base_speed * (1 - congestion * 0.85)

                records.append({
                    "timestamp":      (current_day + timedelta(hours=hour)).isoformat(),
                    "route_id":       route_id,
                    "route_name":     route_info["name"],
                    "zone":           zone,
                    "hour":           hour,
                    "day_of_week":    dow,
                    "weather":        weather,
                    "weather_code":   WEATHERS.index(weather),
                    "has_event":      int(has_event and event_bump > 0),
                    "congestion_pct": round(congestion * 100, 2),
                    "current_speed":  round(max(2.0, curr_speed), 2),
                    "freeflow_speed": base_speed,
                    "zone_mult":      zone_mult,
                    "type_mult":      type_mult,
                    "confidence":     round(float(rng.uniform(0.85, 0.99)), 2),
                })

    df = pd.DataFrame(records)
    os.makedirs("data/processed", exist_ok=True)
    path = "data/processed/mumbai_traffic_history.csv"
    df.to_csv(path, index=False)
    print(f"Generated {len(df):,} records | {days} days | {len(ROUTES)} routes | {len(df['zone'].unique())} zones")
    print(f"Saved to: {path}")
    return df


if __name__ == "__main__":
    df = generate_dataset(days=60)
    print("\nZone coverage:")
    for zone, grp in df.groupby("zone"):
        print(f"  {zone:<22}: {len(grp['route_id'].unique())} routes, avg congestion {grp['congestion_pct'].mean():.1f}%")
