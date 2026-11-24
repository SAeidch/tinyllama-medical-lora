"""
weather_server.py - MCP server #2: weather (latitude/longitude -> conditions + forecast)

Wraps the free Open-Meteo Forecast API (no API key needed for non-commercial use):
    https://api.open-meteo.com/v1/forecast

Speaks MCP over stdio; the agent (weather_agent.py) launches it as a subprocess.
Never print() to stdout in this file (stdout carries the MCP protocol).
"""

import sys

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("open-meteo-weather")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes used by Open-Meteo -> readable text
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def describe(code):
    """Turn a WMO weather code into text (falls back to 'Unknown')."""
    return WEATHER_CODES.get(code, "Unknown")


@mcp.tool()
async def get_weather(
    latitude: float,
    longitude: float,
    temperature_unit: str = "celsius",
    forecast_days: int = 3,
) -> dict:
    """Get current weather conditions and a daily forecast for a location.

    Args:
        latitude: Latitude in decimal degrees (from the geocode tool).
        longitude: Longitude in decimal degrees (from the geocode tool).
        temperature_unit: "celsius" (default) or "fahrenheit".
        forecast_days: Number of forecast days, 1 to 7 (default 3).

    Returns:
        A dict with "current" conditions (temperature, feels-like, humidity,
        precipitation, wind, description) and a "daily_forecast" list
        (date, description, max/min temperature, precipitation, rain chance).
    """
    unit = temperature_unit.lower()
    if unit not in ("celsius", "fahrenheit"):
        unit = "celsius"
    days = max(1, min(int(forecast_days), 7))

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ("temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"),
        "daily": ("weather_code,temperature_2m_max,temperature_2m_min,"
                  "precipitation_sum,precipitation_probability_max"),
        "temperature_unit": unit,
        "forecast_days": days,
        "timezone": "auto",
    }

    print(f"[weather_server] forecast for ({latitude}, {longitude})", file=sys.stderr)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(FORECAST_URL, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as err:
        return {"error": f"Weather request failed: {err}"}

    current = data.get("current", {})
    cu = data.get("current_units", {})
    daily = data.get("daily", {})
    du = data.get("daily_units", {})

    forecast = []
    for i, date in enumerate(daily.get("time", [])):
        forecast.append({
            "date": date,
            "description": describe(daily["weather_code"][i]),
            "temp_max": f'{daily["temperature_2m_max"][i]} {du.get("temperature_2m_max", "")}'.strip(),
            "temp_min": f'{daily["temperature_2m_min"][i]} {du.get("temperature_2m_min", "")}'.strip(),
            "precipitation_sum": f'{daily["precipitation_sum"][i]} {du.get("precipitation_sum", "")}'.strip(),
            "precipitation_probability_max": f'{daily["precipitation_probability_max"][i]} {du.get("precipitation_probability_max", "")}'.strip(),
        })

    return {
        "location": {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
        },
        "current": {
            "time": current.get("time"),
            "description": describe(current.get("weather_code")),
            "temperature": f'{current.get("temperature_2m")} {cu.get("temperature_2m", "")}'.strip(),
            "feels_like": f'{current.get("apparent_temperature")} {cu.get("apparent_temperature", "")}'.strip(),
            "humidity": f'{current.get("relative_humidity_2m")} {cu.get("relative_humidity_2m", "")}'.strip(),
            "precipitation": f'{current.get("precipitation")} {cu.get("precipitation", "")}'.strip(),
            "wind_speed": f'{current.get("wind_speed_10m")} {cu.get("wind_speed_10m", "")}'.strip(),
        },
        "daily_forecast": forecast,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
