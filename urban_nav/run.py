#!/usr/bin/env python3
import os, sys, subprocess


G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[1m"; X = "\033[0m"

def step(n, msg):
    print(f"\n{B}{'='*55}\n  STEP {n} — {msg}\n{'='*55}{X}")

def ok(msg):   print(f"{G}  ✅ {msg}{X}")
def warn(msg): print(f"{Y}  ⚠️  {msg}{X}")
def err(msg):  print(f"{R}  ❌ {msg}{X}")

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