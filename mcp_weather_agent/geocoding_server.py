"""
geocoding_server.py - MCP server #1: geocoding (city name -> latitude/longitude)

Wraps the free Open-Meteo Geocoding API (no API key needed):
    https://geocoding-api.open-meteo.com/v1/search

The server speaks MCP over stdio. The agent (weather_agent.py) starts it as a
subprocess, so you normally never run this file by hand.

IMPORTANT: with stdio transport, stdout is the MCP protocol channel. Never use
print() in this file (it would corrupt the protocol); log to stderr instead.
"""

import sys

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("open-meteo-geocoding")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


@mcp.tool()
async def geocode(city: str, country_code: str | None = None) -> dict:
    """Convert a city or place name into geographic coordinates.

    Args:
        city: Name of the city or place, e.g. "Paris" or "Boston".
        country_code: Optional ISO-3166 alpha-2 country code (e.g. "US", "FR")
            to narrow the search when the name is ambiguous.

    Returns:
        A dict with a "results" list (best match first). Each result has name,
        country, admin1 (state/region), latitude, longitude and timezone.
        If nothing is found, returns a dict with an "error" message.
    """
    params = {"name": city, "count": 5, "language": "en", "format": "json"}
    if country_code:
        params["countryCode"] = country_code.upper()

    print(f"[geocoding_server] looking up {city!r}", file=sys.stderr)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(GEOCODING_URL, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as err:
        return {"error": f"Geocoding request failed: {err}"}

    matches = data.get("results") or []
    if not matches:
        return {"error": f"No location found for '{city}'."}

    return {
        "query": city,
        "results": [
            {
                "name": m.get("name"),
                "country": m.get("country"),
                "country_code": m.get("country_code"),
                "admin1": m.get("admin1"),          # state / region
                "latitude": m.get("latitude"),
                "longitude": m.get("longitude"),
                "timezone": m.get("timezone"),
                "population": m.get("population"),
            }
            for m in matches
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
