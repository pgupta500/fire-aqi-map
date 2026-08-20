"""
Phase 0: The fire data spine.

One job: given a region and a time window, return CLEAN, VALIDATED fire
detections from NASA FIRMS -- or raise loudly. Nothing downstream (the map,
the AQI join, the trends) can be trusted if this layer hands back garbage,
so this file is deliberately strict.
"""

import io
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env in this folder and puts values into os.environ

# --- Config -----------------------------------------------------------------
# The satellite feed is a VARIABLE, not hardcoded downstream, because
# VIIRS_SNPP_NRT retires Nov 2026 -> you'll swap to VIIRS_NOAA20_NRT in ONE
# place. This is the "abstract the source" decision paying off later.
FIRMS_SOURCE = "VIIRS_SNPP_NRT"
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Rough bounding box for India: "west,south,east,north"
INDIA_BBOX = "68,6,98,38"

# Columns FIRMS should return that we actually rely on downstream.
REQUIRED_COLUMNS = ["latitude", "longitude", "acq_date", "confidence", "frp"]


class FireDataError(Exception):
    """Raised when FIRMS returns something we refuse to trust."""


def _build_url(map_key: str, bbox: str, day_range: int) -> str:
    return f"{FIRMS_BASE}/{map_key}/{FIRMS_SOURCE}/{bbox}/{day_range}"


def get_fires(bbox: str = INDIA_BBOX, day_range: int = 1) -> pd.DataFrame:
    """
    Fetch active fire detections from NASA FIRMS for a bounding box.

    bbox:      "west,south,east,north" (defaults to India)
    day_range: days back from today, 1-10 (FIRMS Area API limit)

    Returns a validated DataFrame. RAISES FireDataError on anything wrong.
    """
    # ---- guard the inputs before we even call out ----
    if not 1 <= day_range <= 10:
        raise FireDataError(f"day_range must be 1-10, got {day_range}")

    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        raise FireDataError(
            "FIRMS_MAP_KEY not found. Put it in a .env file as "
            "FIRMS_MAP_KEY=your_key_here"
        )

    url = _build_url(map_key, bbox, day_range)

    # ---- fetch ----
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        raise FireDataError(f"Network error calling FIRMS: {e}") from e

    if resp.status_code != 200:
        raise FireDataError(
            f"FIRMS returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    # ---- THE TRAP (from the interviewer question) ----
    # FIRMS sometimes returns a plain-text error with a 200 status, e.g.
    # "Invalid MAP_KEY..." That is NOT csv. If we hand it straight to
    # read_csv we might get silent junk. So we sniff the body first: real
    # output starts with a header row containing "latitude".
    body = resp.text
    if "latitude" not in body.split("\n", 1)[0].lower():
        raise FireDataError(
            f"FIRMS did not return fire CSV. First line was: "
            f"{body.splitlines()[0] if body.strip() else '<empty>'}"
        )

    # ---- parse ----
    df = pd.read_csv(io.StringIO(body))

    # ---- validate ----
    _validate(df)

    return df


def _validate(df: pd.DataFrame) -> None:
    """Loudly reject anything malformed. Extend this -- it's the heart."""
    if df.empty:
        raise FireDataError("FIRMS returned zero fire rows for this query.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FireDataError(
            f"Missing expected columns: {missing}. Got: {list(df.columns)}"
        )

    # Coordinate sanity: lat in [-90, 90], lon in [-180, 180].
    bad_lat = ~df["latitude"].between(-90, 90)
    bad_lon = ~df["longitude"].between(-180, 180)
    if bad_lat.any() or bad_lon.any():
        raise FireDataError(
            f"Found {bad_lat.sum()} bad latitudes and {bad_lon.sum()} "
            f"bad longitudes -- data is corrupt."
        )


if __name__ == "__main__":
    # Quick manual test. Needs a real FIRMS_MAP_KEY in .env.
    try:
        fires = get_fires(day_range=1)
        print(f"OK: {len(fires)} fire detections for India in the last day.")
        print(fires[REQUIRED_COLUMNS].head())
    except FireDataError as e:
        print(f"FireDataError: {e}")
