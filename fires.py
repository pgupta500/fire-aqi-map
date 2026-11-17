"""
NASA FIRMS fire data fetcher.

Fetches active fire detections for a bounding box from the NASA FIRMS
Area API, validates the response, and returns a clean pandas DataFrame.
Raises FireDataError on any malformed or unexpected response so that
downstream code never receives silently corrupt data.
"""

import io
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# The satellite source is defined once here so it can be swapped in a single
# place. VIIRS_SNPP_NRT is being retired (Nov 2026), so the default is the
# successor feed VIIRS_NOAA20_NRT. Override with FIRMS_SOURCE in .env
# (e.g. VIIRS_NOAA21_NRT or MODIS_NRT) without touching code.
FIRMS_SOURCE = os.environ.get("FIRMS_SOURCE", "VIIRS_NOAA20_NRT")
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Bounding box covering India, formatted as "west,south,east,north".
INDIA_BBOX = "68,6,98,38"

# Columns the rest of the application depends on.
REQUIRED_COLUMNS = ["latitude", "longitude", "acq_date", "confidence", "frp"]

# Optional columns passed through when FIRMS provides them.
OPTIONAL_COLUMNS = ["acq_time"]


class FireDataError(Exception):
    """Raised when the FIRMS response is missing, malformed, or invalid."""


def _build_url(map_key: str, bbox: str, day_range: int) -> str:
    return f"{FIRMS_BASE}/{map_key}/{FIRMS_SOURCE}/{bbox}/{day_range}"


def get_fires(bbox: str = INDIA_BBOX, day_range: int = 5) -> pd.DataFrame:
    """
    Fetch active fire detections from NASA FIRMS for a bounding box.

    Args:
        bbox: Bounding box as "west,south,east,north". Defaults to India.
        day_range: Number of days back from today (1-5, per the API limit).

    Returns:
        A validated DataFrame of fire detections.

    Raises:
        FireDataError: If the request fails or the response is invalid.
    """
    if not 1 <= day_range <= 5:
        raise FireDataError(f"day_range must be 1-5, got {day_range}")

    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        raise FireDataError(
            "FIRMS_MAP_KEY not found. Set it in a .env file as "
            "FIRMS_MAP_KEY=your_key_here"
        )

    url = _build_url(map_key, bbox, day_range)

    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        raise FireDataError(f"Network error calling FIRMS: {e}") from e

    if resp.status_code != 200:
        raise FireDataError(
            f"FIRMS returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    # The API may return a plain-text error with a 200 status instead of CSV.
    # Verify the first line is a real header before parsing.
    body = resp.text
    if "latitude" not in body.split("\n", 1)[0].lower():
        raise FireDataError(
            f"Expected fire CSV but received: "
            f"{body.splitlines()[0] if body.strip() else '<empty>'}"
        )

    df = pd.read_csv(io.StringIO(body))
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    """Raise FireDataError if the DataFrame is empty or malformed."""
    if df.empty:
        raise FireDataError("FIRMS returned no fire detections for this query.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FireDataError(
            f"Missing expected columns: {missing}. Got: {list(df.columns)}"
        )

    bad_lat = ~df["latitude"].between(-90, 90)
    bad_lon = ~df["longitude"].between(-180, 180)
    if bad_lat.any() or bad_lon.any():
        raise FireDataError(
            f"Found {bad_lat.sum()} invalid latitudes and "
            f"{bad_lon.sum()} invalid longitudes."
        )


if __name__ == "__main__":
    try:
        fires = get_fires()
        print(f"Retrieved {len(fires)} fire detections for India (last 5 days).")
        print(fires[REQUIRED_COLUMNS].head())
    except FireDataError as e:
        print(f"Error: {e}")
