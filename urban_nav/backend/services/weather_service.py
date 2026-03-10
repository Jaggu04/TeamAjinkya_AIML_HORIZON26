"""
backend/services/weather_service.py
Fetches real-time weather from OpenWeatherMap API.
Weather code is used as an input feature to the LSTM model.
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Weather condition → code your LSTM understands
WEATHER_CODE_MAP = {
    "Clear":        0,   # no impact on traffic
    "Clouds":       0,   # minimal impact
    "Drizzle":      1,   # +15% congestion
    "Rain":         1,   # +25% congestion
    "Thunderstorm": 2,   # +45% congestion
    "Snow":         2,   # rare in Mumbai but handled
    "Mist":         3,   # +20% congestion (visibility)
    "Fog":          3,
    "Haze":         3,
    "Smoke":        3,
}

WEATHER_LABELS = {0: "Clear", 1: "Rain", 2: "Heavy Rain/Storm", 3: "Fog/Mist"}


def get_mumbai_weather() -> dict:
    """
    Fetch current Mumbai weather.
    Returns weather_code (0-3) used as LSTM input feature.
    Falls back to code=0 (clear) if API unavailable.
    """
    if not API_KEY or API_KEY == "your_openweather_api_key_here":
        print("  ⚠️  OpenWeather API key not set — using default (clear weather)")
        return _default_weather()

    try:
        resp = requests.get(BASE_URL, params={
            "q":     "Mumbai,IN",
            "appid": API_KEY,
            "units": "metric",
        }, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        condition    = data["weather"][0]["main"]
        description  = data["weather"][0]["description"]
        temp         = data["main"]["temp"]
        humidity     = data["main"]["humidity"]
        wind_speed   = data["wind"]["speed"]
        rain_1h      = data.get("rain", {}).get("1h", 0.0)
        weather_code = WEATHER_CODE_MAP.get(condition, 0)

        # Override: heavy rain if rainfall > 5mm/h
        if rain_1h > 5.0:
            weather_code = 2

        result = {
            "condition":     condition,
            "description":   description,
            "weather_code":  weather_code,
            "weather_label": WEATHER_LABELS[weather_code],
            "temp_c":        round(temp, 1),
            "humidity_pct":  humidity,
            "wind_kmh":      round(wind_speed * 3.6, 1),
            "rain_1h_mm":    rain_1h,
            "timestamp":     datetime.now().isoformat(),
            "source":        "openweathermap",
        }

        print(f"  🌤️  Mumbai weather: {condition} ({description}), "
              f"{temp}°C, rain={rain_1h}mm → code={weather_code}")
        return result

    except Exception as e:
        print(f"  ❌ Weather API error: {e} — using default")
        return _default_weather()


def _default_weather() -> dict:
    return {
        "condition":     "Clear",
        "description":   "clear sky",
        "weather_code":  0,
        "weather_label": "Clear",
        "temp_c":        30.0,
        "humidity_pct":  65,
        "wind_kmh":      10.0,
        "rain_1h_mm":    0.0,
        "timestamp":     datetime.now().isoformat(),
        "source":        "default_fallback",
    }


def weather_congestion_adjustment(weather_code: int) -> float:
    """
    Returns percentage points to ADD to base congestion
    based on weather conditions. Used in departure planner.
    """
    return {0: 0.0, 1: 15.0, 2: 35.0, 3: 12.0}.get(weather_code, 0.0)


if __name__ == "__main__":
    import json
    weather = get_mumbai_weather()
    print(json.dumps(weather, indent=2))
    print(f"\nCongestion adjustment: +{weather_congestion_adjustment(weather['weather_code'])}%")
