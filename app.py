"""
Flask backend for the fire-aqi-map app.

Serves NASA FIRMS fire detections, OpenAQ PM2.5 readings, Open-Meteo wind,
and the wind-aware fire -> air quality attribution ("who is burning whose
air") as JSON for the map, plus the static map page itself.
"""

import hashlib
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from aqi import AQIDataError, get_air_quality
from attribution import DEFAULT_HOURS, attribute, station_detail
from cache import TTLCache
from correlate import DEFAULT_RADIUS_KM, find_hotspots, spearman, station_fire_exposure, summarize
from fires import FIRMS_SOURCE, OPTIONAL_COLUMNS, REQUIRED_COLUMNS, FireDataError, get_fires
from wind import WindDataError, get_wind, grid_spec, wind_from_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder="static")
app.json.sort_keys = False
CORS(app)

# Upstream data changes slowly; cache it so page loads are fast and we stay
# well inside the free API quotas. Override with CACHE_TTL_SECONDS.
_cache = TTLCache(ttl_seconds=float(os.environ.get("CACHE_TTL_SECONDS", 1800)))
# Wind changes slowly and Open-Meteo's free tier counts every grid point as a
# call (10,000/day per IP, and Render's free servers share IPs), so wind is
# cached for 6 h, and after a failure we don't ask again for 10 minutes.
_wind_cache = TTLCache(ttl_seconds=float(os.environ.get("WIND_CACHE_TTL_SECONDS", 21600)))
WIND_RETRY_AFTER_S = 600
_wind_failed = {"at": float("-inf"), "msg": ""}

# Wind supplied by visitors' browsers when the server is rate-limited, keyed
# by a hash of its content. Kept per key so one visitor can't change what
# others see. Bounded to a few entries.
_client_winds = {}
_client_winds_lock = threading.Lock()
CLIENT_WIND_TTL_S = 6 * 3600
CLIENT_WIND_MAX = 6

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

    def fetch():
        if time.monotonic() - _wind_failed["at"] < WIND_RETRY_AFTER_S:
            raise WindDataError(_wind_failed["msg"])
        try:
            return get_wind()
        except WindDataError as e:
            _wind_failed.update(at=time.monotonic(), msg=str(e))
            raise
    return _wind_cache.get_or_set("wind", fetch)


def _store_client_wind(w) -> str:
    key = hashlib.sha1(np.ascontiguousarray(w.u).tobytes() + np.ascontiguousarray(w.v).tobytes()
                       + str(w.times[0]).encode()).hexdigest()[:16]
    now = time.monotonic()
    with _client_winds_lock:
        for k in [k for k, (t, _) in _client_winds.items() if now - t > CLIENT_WIND_TTL_S]:
            del _client_winds[k]
        while len(_client_winds) >= CLIENT_WIND_MAX:
            del _client_winds[min(_client_winds, key=lambda k: _client_winds[k][0])]
        _client_winds[key] = (now, w)
    return key


def _client_wind(key):
    with _client_winds_lock:
        hit = _client_winds.get(key)
    if not hit or time.monotonic() - hit[0] > CLIENT_WIND_TTL_S:
        return None
    return hit[1]


def _attribution(days: int, hours: int, wind_key: str = None):
    def build():
        w = _wind() if wind_key is None else _client_wind(wind_key)
        if w is None:
            raise WindDataError("wind expired")
        att = attribute(_aqi(), _fires(days), w, hours=hours)
        # Distance-only exposure, kept for comparison with the wind model.
        near = station_fire_exposure(att.stations, _fires(days), radius_km=DEFAULT_RADIUS_KM)
        att.stations["fires_nearby"] = near["fires_nearby"].to_numpy()
        att.stations["fire_influence"] = near["fire_influence"].to_numpy()
        att.summary["spearman_nearby_vs_pm25"] = _round(
            spearman(near["fire_influence"], near["pm25"]))
        att.summary["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return att
    return _cache.get_or_set(("attribution", days, hours, wind_key), build)


def _round(x, nd=3):
    return None if x is None else round(x, nd)


def _error(msg: str, status: int = 502):
    return jsonify({"error": msg}), status


def _wind_unavailable(msg: str):
    """
    503 telling the page to fetch the wind itself from Open-Meteo (from the
    visitor's own IP) and POST it back; includes the exact grid to request.
    """
    return jsonify({"error": msg, "wind_unavailable": True, "wind_grid": grid_spec()}), 503


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
    """Return recent South Asia fire detections as JSON for the map."""
    try:
        df = _fires(_days())
    except FireDataError as e:
        return _error(str(e))
    cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    return jsonify(_records(df[cols]))


@app.route("/api/aqi")
def aqi_endpoint():
    """Return current South Asia PM2.5 station readings as JSON for the map."""
    try:
        df = _aqi()
    except AQIDataError as e:
        return _error(str(e))
    return jsonify(_records(df))


@app.route("/api/wind")
def wind_endpoint():
    """
    South Asia wind for the hour nearest now (u/v in m/s) for the animation.
    (The worldwide animation is fetched by the browser directly.) With
    ?wind_key=..., returns the browser-supplied wind instead.
    """
    now = np.datetime64(datetime.now(timezone.utc).replace(tzinfo=None), "s")
    key = request.args.get("wind_key")
    if key:
        w = _client_wind(key)
        if w is None:
            return _error("wind expired", 409)
        return jsonify(w.snapshot(now))
    try:
        return jsonify(_wind().snapshot(now))
    except WindDataError as e:
        return _wind_unavailable(str(e))


@app.route("/api/attribution", methods=["GET", "POST"])
def attribution_endpoint():
    """
    Wind-aware accountability analysis.

    GET uses the server's own wind. If that is unavailable (Open-Meteo rate
    limit), it returns 503 with the wind grid, and the page fetches the wind
    from Open-Meteo itself and POSTs {"wind": {lats, lons, t0, u, v}} here.

    Returns {summary, ledger, flows, stations, wind_key}:
      ledger  - per source region: fires, smoke delivered/exported, receivers
      flows   - top cross-region smoke flows (source -> receiver)
      stations- every station with its upwind fire score and source mix
    """
    days, hours = _days(), _hours()
    wind_key = None
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        try:
            wind_key = _store_client_wind(wind_from_client(body.get("wind") or {}))
        except WindDataError as e:
            return _error(str(e), 400)
    try:
        att = _attribution(days, hours, wind_key)
    except WindDataError as e:
        return _wind_unavailable(str(e))
    except UPSTREAM_ERRORS as e:
        return _error(str(e))
    summary = dict(att.summary, days=days, hours=hours, fire_source=FIRMS_SOURCE,
                   demo=DEMO_MODE, wind_from_browser=wind_key is not None)
    stations = att.stations[[c for c in STATION_FIELDS if c in att.stations.columns]]
    return jsonify({
        "summary": summary,
        "ledger": att.ledger,
        "flows": att.flows,
        "stations": _records(stations),
        "wind_key": wind_key,
    })


@app.route("/api/station/<int:station_id>")
def station_endpoint(station_id):
    """Back-trajectory path and the fires credited for one station."""
    days, hours = _days(), _hours()
    wind_key = request.args.get("wind_key") or None
    if wind_key and _client_wind(wind_key) is None:
        return _error("wind expired", 409)
    try:
        att = _attribution(days, hours, wind_key)
    except WindDataError as e:
        return _wind_unavailable(str(e))
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
