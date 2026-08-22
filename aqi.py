"""
OpenAQ air quality data fetcher.

Fetches current PM2.5 readings for India from the OpenAQ v3 API by paging
through the global latest-values feed, filtering to fresh, valid, in-country
stations, and returning a clean pandas DataFrame. Raises AQIDataError on any
malformed or unexpected response so downstream code never receives silently
corrupt data.
"""

import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAQ_BASE = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2

# Bounding box covering India: (west, south, east, north).
INDIA_BOUNDS = {"west": 68.0, "south": 6.0, "east": 98.0, "north": 38.0}

# Readings older than this are considered stale and dropped.
MAX_AGE_DAYS = 2

# The latest-values feed is global and paginated (max 1000 per page). We page
# through it and keep only India points.
PAGE_SIZE = 1000
MAX_PAGES = 25


class AQIDataError(Exception):
    """Raised when the OpenAQ response is missing, malformed, or invalid."""


def _headers() -> dict:
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        raise AQIDataError(
            "OPENAQ_API_KEY not found. Set it in a .env file as "
            "OPENAQ_API_KEY=your_key_here"
        )
    return {"X-API-Key": key}


def get_air_quality() -> pd.DataFrame:
    """
    Fetch current PM2.5 station readings for India.

    Returns:
        A validated DataFrame with columns:
        latitude, longitude, pm25, datetime_utc.

    Raises:
        AQIDataError: If the requests fail or no valid India data is found.
    """
    url = f"{OPENAQ_BASE}/parameters/{PM25_PARAMETER_ID}/latest"
    rows = []

    for page in range(1, MAX_PAGES + 1):
        params = {"limit": PAGE_SIZE, "page": page}
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        except requests.RequestException as e:
            raise AQIDataError(f"Network error calling OpenAQ: {e}") from e

        # A 404 means we have paged past the last page -> stop cleanly.
        if resp.status_code == 404:
            break
        if resp.status_code != 200:
            raise AQIDataError(
                f"OpenAQ returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            results = resp.json()["results"]
        except (ValueError, KeyError) as e:
            raise AQIDataError(f"Unexpected OpenAQ response shape: {e}") from e

        if not results:
            break

        rows.extend(_clean(results))

    df = pd.DataFrame(rows)
    _validate(df)
    return df


def _clean(results: list) -> list:
    """Keep only fresh, valid, in-India readings from one page."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    b = INDIA_BOUNDS
    rows = []

    for r in results:
        coords = r.get("coordinates") or {}
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        value = r.get("value")
        dt = (r.get("datetime") or {}).get("utc")

        if lat is None or lon is None or value is None or dt is None:
            continue
        if value < 0:
            continue
        if not (b["west"] <= lon <= b["east"] and b["south"] <= lat <= b["north"]):
            continue

        reading_time = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        if reading_time < cutoff:
            continue

        rows.append(
            {"latitude": lat, "longitude": lon, "pm25": value, "datetime_utc": dt}
        )

    return rows


def _validate(df: pd.DataFrame) -> None:
    """Raise AQIDataError if the cleaned data is empty or malformed."""
    if df.empty:
        raise AQIDataError("No fresh, valid PM2.5 readings found for India.")

    required = ["latitude", "longitude", "pm25", "datetime_utc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AQIDataError(f"Missing expected columns: {missing}")


if __name__ == "__main__":
    try:
        aqi = get_air_quality()
        print(f"Retrieved {len(aqi)} current PM2.5 stations for India.")
        print(aqi.head())
    except AQIDataError as e:
        print(f"Error: {e}")
