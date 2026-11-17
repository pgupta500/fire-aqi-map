"""
Flask backend for the fire-aqi-map app.

Serves NASA FIRMS fire detections, OpenAQ PM2.5 readings, Open-Meteo wind,
and the wind-aware fire -> air quality attribution ("who is burning whose
air") as JSON for the map, plus the static map page itself.
"""

import os
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from aqi import AQIDataError, get_air_quality
from attribution import DEFAULT_HOURS, attribute, station_detail
from cache import TTLCache
from correlate import DEFAULT_RADIUS_KM, find_hotspots, spearman, station_fire_exposure, summarize
from fires import FIRMS_SOURCE, OPTIONAL_COLUMNS, REQUIRED_COLUMNS, FireDataError, get_fires
from wind import WindDataError, get_global_wind, get_wind

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder="static")
app.json.sort_keys = False
CORS(app)

# Upstream data changes slowly; cache it so page loads are fast and we stay
# well inside the free API quotas. Override with CACHE_TTL_SECONDS.
_cache = TTLCache(ttl_seconds=float(os.environ.get("CACHE_TTL_SECONDS", 1800)))
# Wind changes slowly and Open-Meteo's free tier counts every grid point as a
# call, so wind is cached longer (3 h regional, 6 h worldwide).
_wind_cache = TTLCache(ttl_seconds=float(os.environ.get("WIND_CACHE_TTL_SECONDS", 10800)))
_global_wind_cache = TTLCache(ttl_seconds=21600)

UPSTREAM_ERRORS = (FireDataError, AQIDataError, WindDataError)

# DEMO_MODE=1 serves synthetic data (no API keys needed) for development.
# The UI shows a clear "demo data" banner when this is on.
DEMO_MODE = os.environ.get("DEMO_MODE") == "1"


# --------------------------------------------------------------------------
# Query params

def _int_arg(name, default, lo, hi):
    try:
        v = int(request.args.get(name, default))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


def _days() -> int:
    """?days=1..5 (FIRMS limit), default 5."""
    return _int_arg("days", 5, 1, 5)


def _hours() -> int:
    """?hours=12..72 back-trajectory length, default 48."""
    return _int_arg("hours", DEFAULT_HOURS, 12, 72)


def _radius() -> float:
    """?radius_km=10..500, default 150."""
    try:
        r = float(request.args.get("radius_km", DEFAULT_RADIUS_KM))
    except ValueError:
        r = DEFAULT_RADIUS_KM
    return max(10.0, min(500.0, r))


# --------------------------------------------------------------------------
# Cached data

def _demo():
    from demo_data import demo_scene
    return _cache.get_or_set("demo", demo_scene)


def _fires(days: int):
    if DEMO_MODE:
        f = _demo()[1]
        cutoff = (np.datetime64("today") - np.timedelta64(days - 1, "D")).astype(str)
        return f[f["acq_date"] >= cutoff].reset_index(drop=True)
    return _cache.get_or_set(("fires", days), lambda: get_fires(day_range=days))


def _aqi():
    if DEMO_MODE:
        return _demo()[0]
    return _cache.get_or_set("aqi", get_air_quality)


def _wind():
    if DEMO_MODE:
        return _demo()[2]
    return _wind_cache.get_or_set("wind", get_wind)


def _global_wind():
    return _global_wind_cache.get_or_set("global", get_global_wind)


def _attribution(days: int, hours: int):
    def build():
        att = attribute(_aqi(), _fires(days), _wind(), hours=hours)
        # Distance-only exposure, kept for comparison with the wind model.
        near = station_fire_exposure(att.stations, _fires(days), radius_km=DEFAULT_RADIUS_KM)
        att.stations["fires_nearby"] = near["fires_nearby"].to_numpy()
        att.stations["fire_influence"] = near["fire_influence"].to_numpy()
        att.summary["spearman_nearby_vs_pm25"] = _round(
            spearman(near["fire_influence"], near["pm25"]))
        att.summary["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return att
    return _cache.get_or_set(("attribution", days, hours), build)


def _round(x, nd=3):
    return None if x is None else round(x, nd)


def _error(msg: str, status: int = 502):
    return jsonify({"error": msg}), status


def _records(df):
    """DataFrame -> JSON-safe records (NaN -> null)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


STATION_FIELDS = [
    "station_id", "name", "latitude", "longitude", "pm25", "category", "region", "datetime_utc",
    "location_id", "upwind_score", "smoke_affected", "top_source", "top_source_share",
    "local_share", "sources", "fires_nearby", "fire_influence", "trajectory_hours",
]


# --------------------------------------------------------------------------
# Routes

@app.route("/api/fires")
def fires_endpoint():
    """Return recent India fire detections as JSON for the map."""
    try:
        df = _fires(_days())
    except FireDataError as e:
        return _error(str(e))
    cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    return jsonify(_records(df[cols]))


@app.route("/api/aqi")
def aqi_endpoint():
    """Return current India PM2.5 station readings as JSON for the map."""
    try:
        df = _aqi()
    except AQIDataError as e:
        return _error(str(e))
    return jsonify(_records(df))


@app.route("/api/wind")
def wind_endpoint():
    """
    Wind for the hour nearest now (u/v in m/s) for drawing the animation:
    a fine South Asia grid plus a coarse worldwide grid. The worldwide grid is
    optional; if it fails the map just animates South Asia.
    """
    now = np.datetime64(datetime.now(timezone.utc).replace(tzinfo=None), "s")
    try:
        regional = _wind().snapshot(now)
    except WindDataError as e:
        return _error(str(e))
    try:
        if DEMO_MODE:
            from demo_data import demo_world_wind
            world = demo_world_wind(now.astype("datetime64[h]").astype("datetime64[s]")).snapshot(now)
        else:
            world = _global_wind().snapshot(now)
    except WindDataError:
        world = None
    return jsonify(dict(regional, regional=regional, world=world))


@app.route("/api/attribution")
def attribution_endpoint():
    """
    Wind-aware accountability analysis.

    Returns {summary, ledger, flows, stations}:
      ledger  - per source region: fires, smoke delivered/exported, receivers
      flows   - top cross-region smoke flows (source -> receiver)
      stations- every station with its upwind fire score and source mix
    """
    days, hours = _days(), _hours()
    try:
        att = _attribution(days, hours)
    except UPSTREAM_ERRORS as e:
        return _error(str(e))
    summary = dict(att.summary, days=days, hours=hours, fire_source=FIRMS_SOURCE,
                   demo=DEMO_MODE)
    stations = att.stations[[c for c in STATION_FIELDS if c in att.stations.columns]]
    return jsonify({
        "summary": summary,
        "ledger": att.ledger,
        "flows": att.flows,
        "stations": _records(stations),
    })


@app.route("/api/station/<int:station_id>")
def station_endpoint(station_id):
    """Back-trajectory path and the fires credited for one station."""
    days, hours = _days(), _hours()
    try:
        att = _attribution(days, hours)
    except UPSTREAM_ERRORS as e:
        return _error(str(e))
    try:
        detail = station_detail(att, station_id)
    except KeyError:
        return _error("Unknown station", 404)
    keep = set(STATION_FIELDS)
    detail["station"] = {k: v for k, v in detail["station"].items() if k in keep}
    return jsonify(detail)


@app.route("/api/analysis")
def analysis_endpoint():
    """
    Distance-only analysis (no wind): every station with nearby-fire metrics
    within radius_km, plus hotspots. Kept as a baseline for comparison.
    """
    days, radius = _days(), _radius()
    try:
        fires = _fires(days)
        stations = _aqi()
    except (FireDataError, AQIDataError) as e:
        return _error(str(e))

    exposure = station_fire_exposure(stations, fires, radius_km=radius)
    hotspots = find_hotspots(exposure)
    summary = summarize(exposure, n_fires=len(fires), radius_km=radius)
    summary["days"] = days
    summary["fire_source"] = FIRMS_SOURCE
    summary["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return jsonify({"summary": summary, "stations": _records(exposure),
                    "hotspots": _records(hotspots)})


@app.route("/api/regions")
def regions_endpoint():
    """State / province boundaries used for attribution (GeoJSON)."""
    resp = send_from_directory(os.path.join(BASE_DIR, "data"), "regions.geojson",
                               mimetype="application/geo+json")
    resp.cache_control.max_age = 86400
    return resp


@app.route("/api/health")
def health():
    """Liveness probe for the hosting platform. Does not call upstream APIs."""
    return jsonify({
        "status": "ok",
        "firms_key_configured": bool(os.environ.get("FIRMS_MAP_KEY")),
        "openaq_key_configured": bool(os.environ.get("OPENAQ_API_KEY")),
        "fire_source": FIRMS_SOURCE,
        "demo_mode": DEMO_MODE,
    })


@app.route("/")
def index():
    """Serve the map page."""
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0" if not debug else "127.0.0.1", port=port)
