#!/usr/bin/env python3
"""
setup_and_run.py  —  Run this FIRST when the hackathon starts.
Usage: python setup_and_run.py

Steps:
  1. Installs all dependencies
  2. Generates synthetic Mumbai traffic data (60 days)
  3. Trains LSTM model (if torch is available)
  4. Seeds database with demo users
  5. Starts the FastAPI backend on port 8000
"""

import os, sys, subprocess

G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[1m"; X = "\033[0m"

def step(n, msg):
    print(f"\n{B}{'='*55}\n  STEP {n} — {msg}\n{'='*55}{X}")

def ok(msg):   print(f"{G}  ✅ {msg}{X}")
def warn(msg): print(f"{Y}  ⚠️  {msg}{X}")
def err(msg):  print(f"{R}  ❌ {msg}{X}")

# ── Step 1: Install dependencies ──────────────────────────────────
step(1, "Installing dependencies")
result = subprocess.run([
    sys.executable, "-m", "pip", "install",
    "fastapi", "uvicorn[standard]", "torch", "numpy", "pandas",
    "scikit-learn", "requests", "geopy", "python-dotenv",
    "schedule", "pydantic", "--break-system-packages", "-q"
], capture_output=True, text=True)

if result.returncode != 0:
    # Try without --break-system-packages (Windows / standard venv)
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "fastapi", "uvicorn[standard]", "torch", "numpy", "pandas",
        "scikit-learn", "requests", "geopy", "python-dotenv",
        "schedule", "pydantic", "-q"
    ], capture_output=True, text=True)

if result.returncode == 0:
    ok("All dependencies installed")
else:
    warn("Some packages may have failed — continuing anyway")
    warn(result.stderr[:300] if result.stderr else "")

# ── Step 2: Generate synthetic data ───────────────────────────────
step(2, "Generating synthetic Mumbai traffic data (60 days × 25 routes)")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data.generate_synthetic_data import generate_dataset
    df = generate_dataset(days=60)
    ok(f"Generated {len(df):,} training records across 25 routes")
except Exception as e:
    warn(f"Data generation skipped: {e}")

# ── Step 3: Train LSTM (optional — system works without it) ───────
step(3, "Training LSTM Traffic Forecasting Model (~3-8 min)")
print(f"  {Y}Note: predictions work without this — it improves accuracy{X}")
try:
    from backend.models.lstm_traffic import train
    train()
    ok("LSTM model trained and saved to backend/models/saved/")
except ImportError as e:
    warn(f"LSTM training skipped — missing package: {e}")
    warn("System will use statistical predictor (still fully functional)")
except Exception as e:
    warn(f"LSTM training skipped: {e}")
    warn("System will use statistical predictor (still fully functional)")

# ── Step 4: Init database + demo users ────────────────────────────
step(4, "Initializing database and creating demo users")
try:
    from backend.models.personalization import init_db, UserProfile
    init_db()
    ok("SQLite database initialized (urban_nav.db)")

    import numpy as np
    from datetime import datetime, timedelta

    demo_users = [
        ("user_001", "Rahul Sharma",  19.1197, 72.8463, 19.0631, 72.8677),
        ("user_002", "Priya Mehta",   19.0728, 72.8826, 18.9322, 72.8264),
        ("user_003", "Amit Joshi",    19.1890, 72.9650, 19.1136, 72.8697),
    ]
    for uid, name, hlat, hlng, wlat, wlng in demo_users:
        u = UserProfile(uid)
        u.create_user(name, hlat, hlng, wlat, wlng)
        for i in range(10):
            dep = datetime.now().replace(hour=8, minute=30) - timedelta(days=i)
            arr = dep + timedelta(minutes=35 + i % 15)
            u.log_trip("Home", "Office", "R001", dep, arr,
                       travel_minutes=35 + i % 15,
                       congestion_pct=70 + np.random.randint(-15, 20))
    ok(f"Created {len(demo_users)} demo users with trip history")

except Exception as e:
    warn(f"Database setup issue: {e}")

# ── Step 5: Quick prediction test ─────────────────────────────────
step(5, "Testing prediction engine")
try:
    from backend.models.predictor import predict_all_routes, predict_zone_summary
    now_hour = __import__('datetime').datetime.now().hour
    routes = predict_all_routes(now_hour, __import__('datetime').datetime.now().weekday())
    zones  = predict_zone_summary(now_hour, __import__('datetime').datetime.now().weekday())
    ok(f"Prediction engine working — {len(routes)} routes active")
    print(f"\n  Current zone snapshot:")
    for z in zones[:4]:
        print(f"    {z['zone']:<22} {z['avg_congestion']:>3}%  {z['level']}")
    print(f"    ... and {len(zones)-4} more zones")
except Exception as e:
    warn(f"Prediction test failed: {e}")

# ── Step 6: Start API ─────────────────────────────────────────────
step(6, "Starting FastAPI Backend")
print(f"""
  {G}🚦 Urban Navigation AI is starting...{X}

  {B}API Server:{X}   http://localhost:8000
  {B}API Docs:{X}     http://localhost:8000/docs
  {B}Frontend:{X}     Open  frontend/index.html  in your browser

  {Y}Press Ctrl+C to stop the server{X}
""")

try:
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.api.main:app",
        "--reload", "--port", "8000", "--host", "0.0.0.0"
    ])
except KeyboardInterrupt:
    print(f"\n{Y}  Server stopped.{X}")