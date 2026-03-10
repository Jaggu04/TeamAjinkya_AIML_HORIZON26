"""
data/generate_synthetic_data.py
Generates realistic synthetic Mumbai traffic dataset for LSTM training.
Run this first: python data/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ─── Mumbai Routes ────────────────────────────────────────────────
ROUTES = {
    "R001": {"name": "Western Express Highway (Andheri–Bandra)", "base_speed": 60},
    "R002": {"name": "Eastern Express Highway (Thane–Kurla)",    "base_speed": 55},
    "R003": {"name": "Bandra-Worli Sea Link",                    "base_speed": 70},
    "R004": {"name": "LBS Marg (Ghatkopar–Vikhroli)",           "base_speed": 45},
    "R005": {"name": "SV Road (Andheri–Bandra)",                 "base_speed": 40},
}

# ─── Congestion patterns (hour → multiplier, 1.0 = free flow) ────
HOURLY_PATTERN = {
    0: 0.05, 1: 0.03, 2: 0.03, 3: 0.04, 4: 0.08,
    5: 0.20, 6: 0.50, 7: 0.85, 8: 0.95, 9: 0.90,
   10: 0.65, 11: 0.55, 12: 0.60, 13: 0.65, 14: 0.60,
   15: 0.65, 16: 0.80, 17: 0.92, 18: 0.95, 19: 0.88,
   20: 0.70, 21: 0.50, 22: 0.30, 23: 0.15,
}

DAY_MULTIPLIER = {0: 1.10, 1: 1.05, 2: 1.00, 3: 1.00, 4: 1.05, 5: 0.70, 6: 0.50}
WEATHER_EFFECT = {"clear": 0.0, "rain": 0.25, "heavy_rain": 0.45, "fog": 0.20}
WEATHER_PROB   = [0.60,         0.25,          0.10,              0.05]
WEATHER_TYPES  = ["clear",      "rain",        "heavy_rain",      "fog"]


def generate_dataset(days: int = 60) -> pd.DataFrame:
    records = []
    start   = datetime.now() - timedelta(days=days)

    for day_offset in range(days):
        current_day = start + timedelta(days=day_offset)
        dow         = current_day.weekday()
        weather     = np.random.choice(WEATHER_TYPES, p=WEATHER_PROB)
        w_effect    = WEATHER_EFFECT[weather]
        # random event (cricket match, concert) ~10% of days
        has_event   = np.random.random() < 0.10
        event_hour  = np.random.randint(17, 22) if has_event else -1

        for hour in range(24):
            for route_id, route_info in ROUTES.items():
                base_cong   = HOURLY_PATTERN[hour]
                day_factor  = DAY_MULTIPLIER[dow]
                noise       = np.random.normal(0, 0.05)
                event_bump  = 0.30 if (has_event and abs(hour - event_hour) <= 1) else 0.0

                congestion  = base_cong * day_factor + w_effect + event_bump + noise
                congestion  = float(np.clip(congestion, 0.0, 1.0))

                free_speed  = route_info["base_speed"]
                curr_speed  = free_speed * (1 - congestion * 0.85)

                records.append({
                    "timestamp":       (current_day + timedelta(hours=hour)).isoformat(),
                    "route_id":        route_id,
                    "hour":            hour,
                    "day_of_week":     dow,
                    "weather":         weather,
                    "weather_code":    WEATHER_TYPES.index(weather),
                    "has_event":       int(has_event and abs(hour - event_hour) <= 2),
                    "congestion_pct":  round(congestion * 100, 2),
                    "current_speed":   round(curr_speed, 2),
                    "freeflow_speed":  free_speed,
                    "confidence":      round(np.random.uniform(0.85, 0.99), 2),
                })

    df = pd.DataFrame(records)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/mumbai_traffic_history.csv", index=False)
    print(f"✅ Generated {len(df):,} records across {days} days and {len(ROUTES)} routes.")
    return df


if __name__ == "__main__":
    df = generate_dataset(days=60)
    print(df.head(10))
    print("\nCongestion stats:")
    print(df["congestion_pct"].describe())
