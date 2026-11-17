"""
Flask backend for the fire-aqi-map app.

Serves NASA FIRMS fire detections, OpenAQ air quality readings, and the
fire -> air quality accountability analysis as JSON for the frontend map,
and serves the static map page itself.
"""

import os
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from aqi import AQIDataError, get_air_quality
from cache import TTLCache
from correlate import (
    DEFAULT_RADIUS_KM,
    find_hotspots,
    station_fire_exposure,
    summarize,
)
from fires import FIRMS_SOURCE, FireDataError, get_fires

app = Flask(__name__, static_folder="static")
CORS(app)

# Upstream data changes slowly; cache it so page loads are fast and we stay
# well inside the free API quotas. Override with CACHE_TTL_SECONDS.
_cache = TTLCache(ttl_seconds=float(os.environ.get("CACHE_TTL_SECONDS", 1800)))

FIRE_COLUMNS = ["latitude", "longitude", "acq_date", "confidence", "frp"]


def _days() -> int:
    """?days=1..5 (FIRMS limit), default 5."""
    try:
        days = int(request.args.get("days", 5))
    except ValueError:
        days = 5
    return max(1, min(5, days))


def _radius() -> float:
    """?radius_km=10..500, default 150."""
    try:
        r = float(request.args.get("radius_km", DEFAULT_RADIUS_KM))
    except ValueError:
        r = DEFAULT_RADIUS_KM
    return max(10.0, min(500.0, r))


def _fires(days: int):
    return _cache.get_or_set(("fires", days), lambda: get_fires(day_range=days))


def _aqi():
    return _cache.get_or_set("aqi", get_air_quality)


def _error(msg: str, status: int = 502):
    return jsonify({"error": msg}), status


@app.route("/api/fires")
def fires_endpoint():
    """Return recent India fire detections as JSON for the map."""
    try:
        df = _fires(_days())
    except FireDataError as e:
        return _error(str(e))

    records = df[FIRE_COLUMNS]
    return jsonify(records.to_dict(orient="records"))


@app.route("/api/aqi")
def aqi_endpoint():
    """Return current India PM2.5 station readings as JSON for the map."""
    try:
        df = _aqi()
    except AQIDataError as e:
        return _error(str(e))

    return jsonify(df.to_dict(orient="records"))


@app.route("/api/analysis")
def analysis_endpoint():
    """
    Fire -> air quality accountability analysis.

    Returns {summary, stations, hotspots}: every station enriched with
    nearby-fire metrics, plus the worst stations where unhealthy air and
    heavy burning coincide.
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

    # NaN is not valid JSON; send null instead.
    def records(df):
        return df.astype(object).where(df.notna(), None).to_dict(orient="records")

    return jsonify(
        {"summary": summary, "stations": records(exposure), "hotspots": records(hotspots)}
    )


@app.route("/api/health")
def health():
    """Liveness probe for the hosting platform. Does not call upstream APIs."""
    return jsonify(
        {
            "status": "ok",
            "firms_key_configured": bool(os.environ.get("FIRMS_MAP_KEY")),
            "openaq_key_configured": bool(os.environ.get("OPENAQ_API_KEY")),
            "fire_source": FIRMS_SOURCE,
        }
    )


@app.route("/")
def index():
    """Serve the map page."""
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0" if not debug else "127.0.0.1", port=port)
