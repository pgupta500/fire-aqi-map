"""
Offline tests for the wind field, region lookup, back-trajectories and the
wind-aware attribution, plus the new API routes (using demo data).
Run: pytest
"""

import numpy as np
import pandas as pd
import pytest

from attribution import attribute, back_trajectories, cluster_fires, fire_times, station_detail
from regions import UNKNOWN, assign_regions
from wind import WindDataError, WindField, grid_axes, parse_response, speed_dir_to_uv

NOW = np.datetime64("2026-11-15T12:00:00", "s")


def uniform_wind(direction_deg, speed=5.0, hours_back=96):
    """A WindField with the same wind everywhere, blowing FROM direction_deg."""
    lats, lons = grid_axes()
    times = NOW - np.arange(hours_back, -12, -1).astype("timedelta64[h]")
    u0, v0 = speed_dir_to_uv(speed, direction_deg)
    shape = (len(times), len(lats), len(lons))
    return WindField(lats=lats, lons=lons, times=times.astype("datetime64[s]"),
                     u=np.full(shape, u0), v=np.full(shape, v0))


# ---- wind ------------------------------------------------------------------

def test_speed_dir_to_uv_convention():
    u, v = speed_dir_to_uv(10.0, 270.0)       # westerly: blows toward the east
    assert u == pytest.approx(10.0) and v == pytest.approx(0.0, abs=1e-9)
    u, v = speed_dir_to_uv(10.0, 0.0)         # northerly: blows toward the south
    assert v == pytest.approx(-10.0)


def test_sample_interpolates_and_masks_outside():
    w = uniform_wind(270)
    u, v = w.sample([28.6, 50.0], [77.2, 77.2], [10, 10])
    assert u[0] == pytest.approx(5.0)
    assert np.isnan(u[1])                     # outside the grid
    u, _ = w.sample(28.6, 77.2, -5)           # before the data starts
    assert np.isnan(u)


def _fake_payload(lats, lons, elevation=200.0):
    hourly = {
        "time": ["2026-11-15T00:00", "2026-11-15T01:00", "2026-11-15T02:00"],
        "wind_speed_10m": [2, 2, 2], "wind_direction_10m": [270, 270, 270],
        "wind_speed_925hPa": [6, 6, None], "wind_direction_925hPa": [270, 270, None],
    }
    return [{"elevation": elevation, "hourly": hourly} for _ in range(len(lats) * len(lons))]


def test_parse_response_blends_levels_and_handles_nulls():
    lats, lons = np.array([20.0, 22.0]), np.array([76.0, 78.0])
    w = parse_response(_fake_payload(lats, lons), lats, lons)
    assert w.u.shape == (3, 2, 2)
    assert w.u[0, 0, 0] == pytest.approx(4.0)   # mean of 2 and 6 m/s
    assert w.u[2, 0, 0] == pytest.approx(2.0)   # 925 hPa missing -> 10 m only


def test_parse_response_high_terrain_uses_10m_only():
    lats, lons = np.array([30.0, 32.0]), np.array([90.0, 92.0])
    w = parse_response(_fake_payload(lats, lons, elevation=4000), lats, lons)
    assert w.u[0, 0, 0] == pytest.approx(2.0)


def test_parse_response_rejects_wrong_count():
    with pytest.raises(WindDataError):
        parse_response([{}], np.array([1.0, 2.0]), np.array([1.0, 2.0]))


# ---- regions ---------------------------------------------------------------

def test_assign_regions_known_places():
    names, countries = assign_regions(
        [28.61, 30.90, 31.55, 13.08, 0.0],
        [77.21, 75.85, 74.34, 80.27, 0.0],
    )
    assert list(names[:4]) == ["Delhi", "Punjab", "Punjab (Pakistan)", "Tamil Nadu"]
    assert list(countries[:3]) == ["India", "India", "Pakistan"]
    assert names[4] == UNKNOWN


def test_assign_regions_rest_of_south_asia():
    names, countries = assign_regions(
        [23.81, 27.71, 6.93, 34.53, 27.47, 4.18, 35.69],
        [90.41, 85.32, 79.85, 69.17, 89.64, 73.51, 51.39],
    )
    assert list(names[:6]) == ["Dhaka (Bangladesh)", "Nepal", "Sri Lanka", "Afghanistan",
                               "Bhutan", "Maldives"]
    assert countries[6] != "India" and names[6] in {"Iran", UNKNOWN}


def test_wind_requests_are_chunked(monkeypatch):
    import wind
    calls = []

    class R:
        status_code = 200

        def __init__(self, n):
            self.n = n

        def json(self):
            hourly = {"time": ["2026-11-15T00:00", "2026-11-15T01:00"],
                      "wind_speed_10m": [1, 1], "wind_direction_10m": [90, 90]}
            return [{"elevation": 0, "hourly": hourly}] * self.n

    def fake_get(url, params, timeout):
        n = len(params["latitude"].split(","))
        calls.append(n)
        return R(n)

    monkeypatch.setattr(wind.requests, "get", fake_get)
    lats, lons = wind.grid_axes()
    w = wind.get_wind()
    assert sum(calls) == len(lats) * len(lons)
    assert max(calls) <= wind.MAX_LOCATIONS_PER_REQUEST and len(calls) > 1
    assert w.u.shape == (2, len(lats), len(lons))

    calls.clear()
    g = wind.get_global_wind()
    assert g.u.shape == (2, len(wind.GLOBAL_LATS), len(wind.GLOBAL_LONS))
    # The world grid alone must fit one request and Open-Meteo's ~600/minute limit.
    assert len(calls) == 1 and sum(calls) < 600


# ---- trajectories ------------------------------------------------------------

def test_back_trajectory_goes_upwind():
    # North-westerly wind (from 315 deg): air at Delhi came from the north-west.
    w = uniform_wind(315, speed=5.0)
    lat, lon = back_trajectories(w, np.array([28.6]), np.array([77.2]), np.array([NOW]), hours=24)
    assert lat[0, -1] > 28.6 + 1.0 and lon[0, -1] < 77.2 - 1.0
    # 5 m/s for 24 h is ~432 km
    from correlate import haversine_km
    d = haversine_km(28.6, 77.2, lat[0, -1], lon[0, -1])
    assert 400 < d < 460


def test_back_trajectory_stops_outside_grid():
    w = uniform_wind(180, speed=20.0)          # strong southerly -> back path goes south
    lat, _ = back_trajectories(w, np.array([6.0]), np.array([80.0]), np.array([NOW]), hours=24)
    assert np.isnan(lat[0, -1])


# ---- fires -----------------------------------------------------------------

def test_fire_times_and_clustering():
    fires = pd.DataFrame({
        "latitude": [30.91, 30.92, 30.93, 25.0],
        "longitude": [75.81, 75.82, 75.83, 80.0],
        "acq_date": ["2026-11-15"] * 4,
        "acq_time": [805, 810, 812, 2030],
        "confidence": ["n"] * 4,
        "frp": [10.0, 20.0, 5.0, 3.0],
    })
    t = fire_times(fires)
    assert str(t[0]) == "2026-11-15T08:05:00"
    c = cluster_fires(fires)
    assert len(c) == 2
    big = c.sort_values("frp").iloc[-1]
    assert big["count"] == 3 and big["frp"] == pytest.approx(35.0)
    assert big["region"] == "Punjab"


def test_fire_times_default_when_acq_time_missing():
    fires = pd.DataFrame({"acq_date": ["2026-11-15"]})
    assert str(fire_times(fires)[0]) == "2026-11-15T08:00:00"


# ---- attribution -------------------------------------------------------------

def _scene():
    """One Delhi monitor, fires upwind (Punjab) and downwind (east UP)."""
    stations = pd.DataFrame({
        "latitude": [28.61, 13.08], "longitude": [77.21, 80.27],
        "pm25": [320.0, 30.0],
        "datetime_utc": [str(NOW) + "Z"] * 2,
        "location_id": [1, 2],
    })
    fire_day = str(NOW.astype("datetime64[D]"))
    fires = pd.DataFrame({
        # upwind along a NW path ~100-250 km from Delhi, burned ~8-20 h before
        "latitude": [29.6, 30.2, 30.5, 27.5],
        "longitude": [76.1, 75.4, 75.0, 80.5],     # last one is DOWNWIND (south-east)
        "acq_date": [fire_day] * 4,
        "acq_time": [0, 0, 0, 0],                   # 12 h before NOW
        "confidence": ["n"] * 4,
        "frp": [40.0, 30.0, 25.0, 500.0],
    })
    return stations, fires


def test_attribution_credits_upwind_not_downwind():
    stations, fires = _scene()
    att = attribute(stations, fires, uniform_wind(315, speed=4.0), hours=48)
    delhi, chennai = att.stations.iloc[0], att.stations.iloc[1]
    assert delhi["region"] == "Delhi"
    assert delhi["upwind_score"] > 0
    assert chennai["upwind_score"] == 0
    # The huge downwind fire must get nothing.
    credited = att.clusters.loc[att.pairs["cluster"], "longitude"]
    assert (credited < 78).all()
    # Delhi's smoke is imported (from Haryana/Punjab), not local.
    assert delhi["local_share"] == 0
    srcs = {s["region"] for s in delhi["sources"]}
    assert srcs <= {"Punjab", "Haryana"}


def test_attribution_reverse_wind_credits_nothing():
    stations, fires = _scene()
    fires = fires.iloc[:3]                       # only the NW fires
    att = attribute(stations, fires, uniform_wind(135, speed=4.0), hours=48)
    assert att.stations["upwind_score"].sum() == 0


def test_ledger_flows_and_detail():
    stations, fires = _scene()
    att = attribute(stations, fires, uniform_wind(315, speed=4.0), hours=48)
    exporters = [r for r in att.ledger if r["smoke_exported"] > 0]
    assert exporters and exporters[0]["receivers"][0]["region"] == "Delhi"
    assert sum(r["share_of_all_exported"] for r in att.ledger) == pytest.approx(1.0, abs=0.01)
    assert att.flows and att.flows[0]["receiver"] == "Delhi"
    det = station_detail(att, 0)
    assert det["path"][0]["hours_ago"] == 0 and len(det["path"]) == 49
    assert det["fires"] and sum(f["share"] for f in det["fires"]) == pytest.approx(1.0, abs=0.01)
    with pytest.raises(KeyError):
        station_detail(att, 99)


def test_attribution_with_no_fires():
    stations, fires = _scene()
    att = attribute(stations, fires.iloc[0:0], uniform_wind(315), hours=24)
    assert (att.stations["upwind_score"] == 0).all()
    assert att.ledger == [] and att.flows == []


def test_fire_after_reading_is_ignored():
    stations, fires = _scene()
    fires = fires.iloc[:3].copy()
    fires["acq_date"] = str((NOW + np.timedelta64(1, "D")).astype("datetime64[D]"))
    att = attribute(stations, fires, uniform_wind(315, speed=4.0), hours=48)
    assert att.stations["upwind_score"].sum() == 0


# ---- API (demo mode) ---------------------------------------------------------

@pytest.fixture
def demo_client(monkeypatch):
    import app as app_module
    app_module._cache.clear()
    monkeypatch.setattr(app_module, "DEMO_MODE", True)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_api_attribution_demo(demo_client):
    r = demo_client.get("/api/attribution?days=5&hours=48")
    assert r.status_code == 200
    d = r.get_json()
    assert d["summary"]["demo"] is True
    assert d["ledger"][0]["region"] in {"Punjab", "Punjab (Pakistan)", "Haryana"}
    assert d["flows"] and d["stations"]
    sid = max(d["stations"], key=lambda s: s["upwind_score"])["station_id"]
    det = demo_client.get(f"/api/station/{sid}").get_json()
    assert det["path"] and det["fires"]
    assert demo_client.get("/api/station/99999").status_code == 404


def test_api_wind_and_regions(demo_client):
    w = demo_client.get("/api/wind").get_json()
    assert len(w["u"]) == len(w["lats"]) and len(w["u"][0]) == len(w["lons"])
    g = demo_client.get("/api/regions").get_json()
    assert g["type"] == "FeatureCollection" and len(g["features"]) > 40


# ---- server rate-limited: browser-supplied wind ------------------------------

def _client_payload(step=None):
    import wind
    lats, lons = wind.grid_axes()
    t0 = (np.datetime64("now", "h") - np.timedelta64(96, "h")).astype("datetime64[s]")
    nt = 120
    u0, v0 = speed_dir_to_uv(4.0, 315)
    return {"lats": lats.tolist(), "lons": lons.tolist(), "t0": str(t0),
            "u": np.full((nt, len(lats), len(lons)), round(float(u0), 2)).tolist(),
            "v": np.full((nt, len(lats), len(lons)), round(float(v0), 2)).tolist()}


@pytest.fixture
def limited_client(monkeypatch):
    """Real-data mode where Open-Meteo always rate-limits the server."""
    import app as app_module
    from wind import WindDataError as WDE
    stations, fires = _scene()
    now = str(np.datetime64("now", "h")) + ":00:00Z"
    stations["datetime_utc"] = now
    fires["acq_date"] = str(np.datetime64("now", "D"))
    app_module._cache.clear(); app_module._wind_cache.clear(); app_module._client_winds.clear()
    app_module._wind_failed.update(at=float("-inf"), msg="")
    calls = []

    def limited():
        calls.append(1)
        raise WDE("Open-Meteo returned HTTP 429: Daily API request limit exceeded")

    monkeypatch.setattr(app_module, "DEMO_MODE", False)
    monkeypatch.setattr(app_module, "get_wind", limited)
    monkeypatch.setattr(app_module, "get_air_quality", lambda: stations)
    monkeypatch.setattr(app_module, "get_fires", lambda day_range=5: fires)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), calls


def test_rate_limited_server_asks_browser_for_wind(limited_client):
    c, calls = limited_client
    r = c.get("/api/attribution")
    assert r.status_code == 503
    body = r.get_json()
    assert body["wind_unavailable"] is True
    assert len(body["wind_grid"]["lats"]) * len(body["wind_grid"]["lons"]) <= 400
    # Backoff: a second request doesn't hit Open-Meteo again.
    assert c.get("/api/attribution").status_code == 503
    assert len(calls) == 1


def test_browser_wind_is_used_and_keyed(limited_client):
    c, _ = limited_client
    r = c.post("/api/attribution?hours=48", json={"wind": _client_payload()})
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert d["summary"]["wind_from_browser"] is True and d["wind_key"]
    delhi = next(s for s in d["stations"] if s["region"] == "Delhi")
    assert delhi["upwind_score"] > 0
    det = c.get(f"/api/station/{delhi['station_id']}?hours=48&wind_key={d['wind_key']}")
    assert det.status_code == 200 and det.get_json()["path"]
    assert c.get(f"/api/wind?wind_key={d['wind_key']}").status_code == 200
    # Unknown keys are refused rather than silently using other wind.
    assert c.get("/api/station/0?wind_key=nope").status_code == 409


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(lats=p["lats"][:-1]),                       # wrong grid
    lambda p: p.update(t0="2001-01-01T00:00:00"),                  # stale
    lambda p: p["u"][0][0].__setitem__(0, 999.0),                  # impossible speed
    lambda p: p.update(u=p["u"][:5], v=p["v"][:5]),                # too few hours
    lambda p: p.pop("v"),                                          # missing field
])
def test_bad_browser_wind_is_rejected(limited_client, mutate):
    c, _ = limited_client
    p = _client_payload()
    mutate(p)
    assert c.post("/api/attribution", json={"wind": p}).status_code == 400
