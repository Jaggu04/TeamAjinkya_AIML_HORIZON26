"""
backend/models/personalization.py
MODULE 4 — Personalized Learning System

Stores user trip history and learns:
  - Frequent routes
  - Preferred departure times
  - Parking preferences
  - Congestion tolerance

Uses: SQLite + collaborative filtering (cosine similarity)
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

DB_PATH = "urban_nav.db"


# ══════════════════════════════════════════════════════════════════
# 1. Database Schema + Init
# ══════════════════════════════════════════════════════════════════

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id         TEXT PRIMARY KEY,
    name            TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    home_lat        REAL,
    home_lng        REAL,
    work_lat        REAL,
    work_lng        REAL,
    cong_tolerance  REAL DEFAULT 0.5,   -- 0=avoids any congestion, 1=doesn't mind
    walk_tolerance  INTEGER DEFAULT 10   -- max walking minutes from parking
);

CREATE TABLE IF NOT EXISTS trips (
    trip_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT,
    origin          TEXT,
    destination     TEXT,
    origin_lat      REAL,
    origin_lng      REAL,
    dest_lat        REAL,
    dest_lng        REAL,
    route_id        TEXT,
    departure_time  TEXT,
    arrival_time    TEXT,
    travel_minutes  INTEGER,
    congestion_pct  REAL,
    parking_lot_id  TEXT,
    rating          INTEGER,            -- 1-5 user satisfaction
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id         TEXT PRIMARY KEY,
    frequent_routes TEXT,               -- JSON list of (origin, dest) pairs
    preferred_hours TEXT,               -- JSON: {dow: [preferred_hours]}
    avg_congestion  REAL,
    avg_walk_min    REAL,
    updated_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript(SCHEMA)
    print("✅ Database initialized.")


# ══════════════════════════════════════════════════════════════════
# 2. User Profile Manager
# ══════════════════════════════════════════════════════════════════

class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        init_db()

    def create_user(self, name: str, home_lat: float = None, home_lng: float = None,
                    work_lat: float = None, work_lng: float = None) -> dict:
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO users (user_id, name, home_lat, home_lng, work_lat, work_lng) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self.user_id, name, home_lat, home_lng, work_lat, work_lng)
            )
        return self.get_user()

    def get_user(self) -> Optional[dict]:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id=?", (self.user_id,)).fetchone()
        return dict(row) if row else None

    def log_trip(self, origin: str, destination: str, route_id: str,
                 departure_time: datetime, arrival_time: datetime,
                 travel_minutes: int, congestion_pct: float,
                 parking_lot_id: str = None, rating: int = None,
                 origin_lat: float = None, origin_lng: float = None,
                 dest_lat: float = None, dest_lng: float = None) -> int:
        """Log a completed trip for learning."""
        with get_db() as conn:
            cur = conn.execute(
                """INSERT INTO trips
                   (user_id, origin, destination, origin_lat, origin_lng, dest_lat, dest_lng,
                    route_id, departure_time, arrival_time, travel_minutes,
                    congestion_pct, parking_lot_id, rating)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (self.user_id, origin, destination, origin_lat, origin_lng,
                 dest_lat, dest_lng, route_id,
                 departure_time.isoformat(), arrival_time.isoformat(),
                 travel_minutes, congestion_pct, parking_lot_id, rating)
            )
            trip_id = cur.lastrowid

        self._update_preferences()
        return trip_id

    def get_trip_history(self, limit: int = 50) -> list[dict]:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM trips WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (self.user_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def _update_preferences(self):
        """Recompute user preferences from trip history."""
        trips = self.get_trip_history(limit=100)
        if not trips:
            return

        # Frequent routes (top 5)
        route_counts = defaultdict(int)
        for t in trips:
            key = f"{t['origin']}→{t['destination']}"
            route_counts[key] += 1
        frequent = sorted(route_counts.items(), key=lambda x: -x[1])[:5]

        # Preferred departure hours per day of week
        preferred_hours: dict[int, list] = defaultdict(list)
        for t in trips:
            try:
                dt  = datetime.fromisoformat(t["departure_time"])
                dow = dt.weekday()
                preferred_hours[dow].append(dt.hour)
            except Exception:
                continue

        avg_hours = {
            str(dow): int(np.mean(hours)) if hours else 8
            for dow, hours in preferred_hours.items()
        }

        avg_cong = float(np.mean([t["congestion_pct"] for t in trips if t["congestion_pct"]]))

        with get_db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO user_preferences
                   (user_id, frequent_routes, preferred_hours, avg_congestion, updated_at)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (self.user_id,
                 json.dumps([r[0] for r in frequent]),
                 json.dumps(avg_hours),
                 round(avg_cong, 1))
            )

    def get_preferences(self) -> dict:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE user_id=?", (self.user_id,)
            ).fetchone()
        if not row:
            return {}
        prefs = dict(row)
        prefs["frequent_routes"] = json.loads(prefs.get("frequent_routes") or "[]")
        prefs["preferred_hours"] = json.loads(prefs.get("preferred_hours") or "{}")
        return prefs

    def get_personalized_insight(self, day_of_week: int = None) -> str:
        """Generate a human-readable personalization insight."""
        prefs = self.get_preferences()
        if not prefs:
            return "Complete a few trips and we'll personalize your experience!"

        dow = day_of_week if day_of_week is not None else datetime.now().weekday()
        DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_name  = DAY_NAMES[dow]

        pref_hour = prefs.get("preferred_hours", {}).get(str(dow))
        routes    = prefs.get("frequent_routes", [])

        parts = []
        if pref_hour is not None:
            parts.append(f"On {day_name}s, you usually leave around {pref_hour}:00.")
        if routes:
            top_route = routes[0]
            parts.append(f"Your most frequent route is {top_route}.")
        if prefs.get("avg_congestion"):
            avg = prefs["avg_congestion"]
            tol = "high" if avg > 70 else ("moderate" if avg > 40 else "low")
            parts.append(f"You typically travel during {tol}-congestion periods.")

        return " ".join(parts) if parts else "Keep travelling to get personalized insights!"


# ══════════════════════════════════════════════════════════════════
# 3. Collaborative Filtering
# ══════════════════════════════════════════════════════════════════

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom > 0 else 0.0


def build_user_vector(user_id: str) -> np.ndarray:
    """
    Build a feature vector for a user from their preferences:
    [avg_departure_hour, avg_congestion_tolerance, weekend_traveler, frequent_route_count]
    """
    profile = UserProfile(user_id)
    prefs   = profile.get_preferences()
    trips   = profile.get_trip_history()

    if not trips:
        return np.zeros(6)

    dep_hours    = []
    for t in trips:
        try:
            dep_hours.append(datetime.fromisoformat(t["departure_time"]).hour)
        except Exception:
            pass

    avg_dep_hour = float(np.mean(dep_hours)) if dep_hours else 8.0
    avg_cong     = prefs.get("avg_congestion", 50.0)
    n_routes     = len(prefs.get("frequent_routes", []))
    n_trips      = len(trips)
    weekend_pct  = sum(1 for t in trips
                       if datetime.fromisoformat(t.get("departure_time","2024-01-01")).weekday() >= 5
                  ) / max(len(trips), 1)
    ratings = [t["rating"] for t in trips if t.get("rating")]
    avg_rating   = float(np.mean(ratings)) if ratings else 3.0

    return np.array([avg_dep_hour, avg_cong / 100, n_routes / 10,
                     n_trips / 50, weekend_pct, avg_rating / 5])


def find_similar_users(target_user_id: str, top_k: int = 3) -> list[dict]:
    """Find users with similar travel behavior using cosine similarity."""
    with get_db() as conn:
        all_users = [r["user_id"] for r in
                     conn.execute("SELECT user_id FROM users").fetchall()]

    target_vec = build_user_vector(target_user_id)
    similarities = []

    for uid in all_users:
        if uid == target_user_id:
            continue
        vec  = build_user_vector(uid)
        sim  = cosine_similarity(target_vec, vec)
        similarities.append({"user_id": uid, "similarity": round(sim, 3)})

    similarities.sort(key=lambda x: -x["similarity"])
    return similarities[:top_k]


def collaborative_recommendation(target_user_id: str, day_of_week: int) -> dict:
    """
    Recommend departure hour based on what similar users prefer.
    """
    similar = find_similar_users(target_user_id)
    if not similar:
        return {"recommended_hour": 8, "source": "default", "message": "No similar users yet."}

    hours = []
    for s in similar:
        prefs = UserProfile(s["user_id"]).get_preferences()
        h     = prefs.get("preferred_hours", {}).get(str(day_of_week))
        if h is not None:
            hours.append((h, s["similarity"]))

    if not hours:
        return {"recommended_hour": 8, "source": "default"}

    # Weighted average by similarity score
    weighted = sum(h * w for h, w in hours) / sum(w for _, w in hours)
    rec_hour = round(weighted)

    DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    msg = (f"Users with similar patterns prefer leaving around "
           f"{rec_hour}:00 on {DAY_NAMES[day_of_week]}s.")

    return {
        "recommended_hour": rec_hour,
        "similar_users_count": len(similar),
        "source": "collaborative_filtering",
        "message": msg,
    }


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    # Create demo user
    u = UserProfile("user_001")
    u.create_user("Rahul Sharma", home_lat=19.1197, home_lng=72.8463,
                  work_lat=19.0631, work_lng=72.8677)

    # Log some fake trips
    now = datetime.now()
    for i in range(10):
        dep = now.replace(hour=8, minute=30) - timedelta(days=i)
        arr = dep + timedelta(minutes=35 + i % 10)
        u.log_trip("Andheri", "BKC", "R001", dep, arr,
                   travel_minutes=35+i%10, congestion_pct=75.0 + i*2)

    prefs   = u.get_preferences()
    insight = u.get_personalized_insight()
    print(f"\n📊 Preferences: {prefs}")
    print(f"💡 Insight: {insight}")
