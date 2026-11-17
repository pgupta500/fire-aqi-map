"""
Tests for the fire -> air quality accountability analysis and the API.
Run: pytest
"""

import math

import pandas as pd
import pytest

import app as app_module
from cache import TTLCache
from correlate import (
    find_hotspots,
    haversine_km,
    naqi_category,
    spearman,
    station_fire_exposure,
    summarize,
)

# Two stations: one in Punjab surrounded by stubble fires, one in Chennai
# with clean air and no fires.
STATIONS = pd.DataFrame(
    {
        "latitude": [30.90, 13.08],
        "longitude": [75.85, 80.27],
        "pm25": [310.0, 25.0],
        "datetime_utc": ["2026-11-15T10:00:00Z"] * 2,
    }
)
FIRES = pd.DataFrame(
    {
        "latitude": [30.95, 30.80, 31.10, 30.70, 30.99, 31.00],
        "longitude": [75.80, 75.95, 75.70, 75.60, 76.00, 75.90],
        "acq_date": ["2026-11-15"] * 6,
        "confidence": ["h"] * 6,
        "frp": [12.0, 8.5, 20.1, 3.3, 15.0, 9.9],
    }
)


def test_haversine_known_distance():
    # Delhi -> Mumbai is ~1150 km great-circle.
    d = haversine_km(28.61, 77.21, 19.08, 72.88)
    assert 1100 < d < 1200
    assert haversine_km(10, 10, 10, 10) == 0


@pytest.mark.parametrize(
    "v,cat",
    [(0, "Good"), (30, "Good"), (31, "Satisfactory"), (90, "Moderate"),
     (100, "Poor"), (200, "Very Poor"), (400, "Severe")],
)
def test_naqi_category(v, cat):
    assert naqi_category(v) == cat


def test_exposure_counts_only_nearby_fires():
    exp = station_fire_exposure(STATIONS, FIRES, radius_km=100)
    punjab, chennai = exp.iloc[0], exp.iloc[1]
    assert punjab["fires_nearby"] == 6
    assert punjab["frp_nearby"] == pytest.approx(FIRES["frp"].sum(), abs=0.1)
    assert 0 < punjab["fire_influence"] < punjab["frp_nearby"]
    assert punjab["nearest_fire_km"] < 15
    assert chennai["fires_nearby"] == 0
    assert chennai["fire_influence"] == 0
    assert chennai["nearest_fire_km"] > 1500
    assert list(exp["category"]) == ["Severe", "Good"]


def test_exposure_with_no_fires():
    exp = station_fire_exposure(STATIONS, FIRES.iloc[0:0], radius_km=100)
    assert (exp["fires_nearby"] == 0).all()
    assert exp["nearest_fire_km"].isna().all()


def test_exposure_chunks_many_stations():
    many = pd.concat([STATIONS] * 100, ignore_index=True)  # > chunk size
    exp = station_fire_exposure(many, FIRES, radius_km=100)
    assert len(exp) == 200
    assert (exp["fires_nearby"].iloc[::2] == 6).all()


def test_invalid_radius():
    with pytest.raises(ValueError):
        station_fire_exposure(STATIONS, FIRES, radius_km=0)


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2], [1, 2]) is None


def test_hotspots_and_summary():
    exp = station_fire_exposure(STATIONS, FIRES, radius_km=100)
    hot = find_hotspots(exp)
    assert len(hot) == 1 and hot.iloc[0]["pm25"] == 310.0
    s = summarize(exp, n_fires=len(FIRES), radius_km=100)
    assert s["stations"] == 2 and s["fires"] == 6
    assert s["stations_with_fires_nearby"] == 1
    assert s["mean_pm25_near_fires"] == 310.0
    assert s["mean_pm25_no_fires"] == 25.0
    assert s["hotspots"] == 1


def test_ttl_cache():
    now = [0.0]
    c = TTLCache(10, clock=lambda: now[0])
    calls = []
    produce = lambda: calls.append(1) or len(calls)
    assert c.get_or_set("k", produce) == 1
    assert c.get_or_set("k", produce) == 1
    now[0] = 11
    assert c.get_or_set("k", produce) == 2


def test_ttl_cache_does_not_cache_errors():
    c = TTLCache(10)
    with pytest.raises(RuntimeError):
        c.get_or_set("k", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert c.get_or_set("k", lambda: 5) == 5


# ---- API -----------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    app_module._cache.clear()
    monkeypatch.setattr(app_module, "get_fires", lambda day_range=5: FIRES)
    monkeypatch.setattr(app_module, "get_air_quality", lambda: STATIONS)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_api_analysis(client):
    r = client.get("/api/analysis?days=3&radius_km=100")
    assert r.status_code == 200
    data = r.get_json()
    assert data["summary"]["days"] == 3
    assert data["summary"]["radius_km"] == 100
    assert len(data["stations"]) == 2
    assert len(data["hotspots"]) == 1
    # NaN never leaks into JSON
    for s in data["stations"]:
        assert s["nearest_fire_km"] is None or not math.isnan(s["nearest_fire_km"])


def test_api_clamps_params(client):
    data = client.get("/api/analysis?days=99&radius_km=abc").get_json()
    assert data["summary"]["days"] == 5
    assert data["summary"]["radius_km"] == 150


def test_api_fires_and_aqi(client):
    assert len(client.get("/api/fires").get_json()) == 6
    assert len(client.get("/api/aqi").get_json()) == 2


def test_api_upstream_error(client, monkeypatch):
    from fires import FireDataError

    def boom(day_range=5):
        raise FireDataError("FIRMS down")

    app_module._cache.clear()
    monkeypatch.setattr(app_module, "get_fires", boom)
    r = client.get("/api/analysis")
    assert r.status_code == 502
    assert r.get_json()["error"] == "FIRMS down"


def test_health_and_index(client):
    assert client.get("/api/health").get_json()["status"] == "ok"
    assert b"SmokeTrail" in client.get("/").data
