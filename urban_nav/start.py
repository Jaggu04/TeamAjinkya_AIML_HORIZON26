#!/usr/bin/env python3
"""
start.py  —  Single command to run the entire system
Usage: python start.py

What this does:
  1. Checks Python version
  2. Installs dependencies
  3. Generates synthetic training data
  4. Trains LSTM model (~5 min)
  5. Trains parking Random Forest model (~30 sec)
  6. Seeds database with demo users
  7. Starts FastAPI server on port 8000

After running, open:  frontend/index.html  in your browser
API docs at:          http://localhost:8000/docs
"""

import sys
import os
import subprocess

# ── Color output helpers ───────────────────────────────────────────
def green(s):  return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

def header(title):
    print(f"\n{bold('═'*55)}")
    print(f"  {bold(title)}")
    print(f"{bold('═'*55)}")

def ok(msg):   print(green(f"  ✅ {msg}"))
def warn(msg): print(yellow(f"  ⚠️  {msg}"))
def err(msg):  print(red(f"  ❌ {msg}"))
def info(msg): print(f"  ℹ️  {msg}")

# ── Step 0: Python version check ──────────────────────────────────
header("STEP 0 — Checking Python version")
v = sys.version_info
if v.major < 3 or (v.major == 3 and v.minor < 9):
    err(f"Python 3.9+ required. You have {v.major}.{v.minor}")
    err("Download from https://python.org")
    sys.exit(1)
ok(f"Python {v.major}.{v.minor}.{v.micro}")

# ── Step 1: Install dependencies ──────────────────────────────────
header("STEP 1 — Installing dependencies")
info("This may take 3-8 minutes on first run...")

packages = [
    "fastapi==0.111.0",
    "uvicorn[standard]==0.29.0",
    "torch",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scikit-learn==1.4.2",
    "requests==2.31.0",
    "geopy==2.4.1",
    "python-dotenv==1.0.1",
    "schedule==1.2.1",
    "pydantic==2.7.1",
]

result = subprocess.run(
    [sys.executable, "-m", "pip", "install"] + packages,
    capture_output=True, text=True
)
if result.returncode != 0:
    err("Dependency installation failed!")
    print(result.stderr)
    sys.exit(1)
ok("All dependencies installed")

# ── Step 2: Generate synthetic data ───────────────────────────────
header("STEP 2 — Generating synthetic Mumbai traffic data")
sys.path.insert(0, os.path.dirname(__file__))

try:
    from data.generate_synthetic_data import generate_dataset
    df = generate_dataset(days=60)
    ok(f"Generated {len(df):,} training records across 5 Mumbai routes")
except Exception as e:
    err(f"Data generation failed: {e}")
    sys.exit(1)

# ── Step 3: Train LSTM ─────────────────────────────────────────────
header("STEP 3 — Training LSTM Traffic Forecasting Model")
info("This takes approximately 3-8 minutes...")
info("You will see training progress below:\n")

try:
    from backend.models.lstm_traffic import train
    train()
    ok("LSTM model trained and saved to backend/models/saved/")
except Exception as e:
    warn(f"LSTM training error: {e}")
    warn("Continuing with rule-based forecasting fallback")

# ── Step 4: Train parking model ────────────────────────────────────
header("STEP 4 — Training Parking Intelligence Model")

try:
    from backend.models.parking_intelligence import train_parking_model
    train_parking_model()
    ok("Parking Random Forest model trained and saved")
except Exception as e:
    warn(f"Parking model error: {e}")
    warn("Continuing with default parking estimates")

# ── Step 5: Init database + seed demo users ────────────────────────
header("STEP 5 — Setting up database and demo users")

try:
    import numpy as np
    from datetime import datetime, timedelta
    from backend.models.personalization import init_db, UserProfile

    init_db()
    ok("SQLite database initialized (urban_nav.db)")

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
            arr = dep + timedelta(minutes=35 + i % 10)
            u.log_trip(
                origin="Home", destination="Office",
                route_id="R001", departure_time=dep, arrival_time=arr,
                travel_minutes=35 + i % 10,
                congestion_pct=70.0 + np.random.randint(-10, 25)
            )
    ok(f"Created {len(demo_users)} demo user profiles with trip history")

except Exception as e:
    warn(f"Database setup error: {e}")

# ── Step 6: Start the API ──────────────────────────────────────────
header("STEP 6 — Starting FastAPI Backend")

print(f"""
  {green('🚦 Urban Navigation AI is starting...')}

  {bold('API Server:')}   http://localhost:8000
  {bold('API Docs:')}     http://localhost:8000/docs
  {bold('Frontend:')}     Open  frontend/index.html  in your browser
                 OR run:  python -m http.server 3000 --directory frontend
                 Then go to:  http://localhost:3000

  {yellow('Press Ctrl+C to stop the server')}
""")

try:
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.api.main:app",
        "--reload",
        "--port", "8000",
        "--host", "0.0.0.0"
    ])
except KeyboardInterrupt:
    print(f"\n{yellow('  Server stopped.')}")
