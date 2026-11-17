"""
Open-Meteo wind field fetcher.

Downloads hourly wind for a regular lat/lon grid over South Asia in a single
multi-location request (free, no API key) and exposes it as a WindField that
can be sampled anywhere in space and time with bilinear/linear interpolation.

Transport wind = the average of the 10 m and 925 hPa (~750 m) winds, i.e. a
rough proxy for the mixed boundary layer where crop-fire smoke travels. Over
high terrain (where 925 hPa is underground) only the 10 m wind is used.
"""

import os
from dataclasses import dataclass

import numpy as np
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Grid covering South Asia with a margin so back-trajectories from border
# monitors (and smoke arriving from Iran/Central Asia/Myanmar) stay inside.
GRID_WEST, GRID_EAST = 52.5, 105.0
GRID_SOUTH, GRID_NORTH = -5.0, 45.0
GRID_STEP = float(os.environ.get("WIND_GRID_STEP", 2.5))

# Coarse worldwide grid, only used to draw the wind animation everywhere.
GLOBAL_STEP = 10.0
GLOBAL_LATS = np.arange(-80.0, 80.0 + 1e-9, GLOBAL_STEP)
GLOBAL_LONS = np.arange(-180.0, 180.0 - GLOBAL_STEP + 1e-9, GLOBAL_STEP)

# Open-Meteo counts each location as one call, so long coordinate lists are
# split into a few requests (also keeps URLs short).
MAX_LOCATIONS_PER_REQUEST = 300

# Days of past wind to fetch. Readings can be ~2 days old and trajectories run
# up to 48 h back, so 4 days is always enough.
PAST_DAYS = 4

# Above this surface elevation (m) the 925 hPa level is below ground.
HIGH_TERRAIN_M = 700.0

HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_925hPa",
    "wind_direction_925hPa",
]


class WindDataError(Exception):
    """Raised when the Open-Meteo response is missing or malformed."""


def grid_axes(step: float = GRID_STEP):
    lats = np.arange(GRID_SOUTH, GRID_NORTH + 1e-9, step)
    lons = np.arange(GRID_WEST, GRID_EAST + 1e-9, step)
    return lats, lons


def speed_dir_to_uv(speed, direction_deg):
    """Meteorological convention: direction is where the wind blows FROM."""
    rad = np.radians(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


@dataclass
class WindField:
    lats: np.ndarray     # (ny,) ascending
    lons: np.ndarray     # (nx,) ascending
    times: np.ndarray    # (nt,) datetime64[s] UTC, hourly, ascending
    u: np.ndarray        # (nt, ny, nx) eastward m/s
    v: np.ndarray        # (nt, ny, nx) northward m/s

    @property
    def t0(self):
        return self.times[0]

    def hours_since_start(self, t):
        """datetime64 (array) -> fractional hour index into self.times."""
        return (np.asarray(t, dtype="datetime64[s]") - self.t0).astype(float) / 3600.0

    def sample(self, lat, lon, hour_idx):
        """
        Interpolate (u, v) at points. All args broadcast to the same shape.
        Points outside the grid in space or time return NaN.
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        h = np.asarray(hour_idx, dtype=float)
        lat, lon, h = np.broadcast_arrays(lat, lon, h)

        ny, nx, nt = len(self.lats), len(self.lons), len(self.times)
        fy = (lat - self.lats[0]) / (self.lats[1] - self.lats[0])
        fx = (lon - self.lons[0]) / (self.lons[1] - self.lons[0])
        valid = (
            np.isfinite(fy) & np.isfinite(fx) & np.isfinite(h)
            & (fy >= 0) & (fy <= ny - 1) & (fx >= 0) & (fx <= nx - 1)
            & (h >= 0) & (h <= nt - 1)
        )
        fy = np.where(valid, fy, 0.0)
        fx = np.where(valid, fx, 0.0)
        h = np.where(valid, h, 0.0)

        y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        t0 = np.clip(np.floor(h).astype(int), 0, max(nt - 2, 0))
        wy, wx = fy - y0, fx - x0
        wt = h - t0 if nt > 1 else np.zeros_like(h)
        t1 = np.minimum(t0 + 1, nt - 1)

        def interp(field):
            def at(t):
                return (
                    field[t, y0, x0] * (1 - wy) * (1 - wx)
                    + field[t, y0, x0 + 1] * (1 - wy) * wx
                    + field[t, y0 + 1, x0] * wy * (1 - wx)
                    + field[t, y0 + 1, x0 + 1] * wy * wx
                )
            out = at(t0) * (1 - wt) + at(t1) * wt
            return np.where(valid, out, np.nan)

        return interp(self.u), interp(self.v)

    def nearest_hour_index(self, t):
        idx = int(round(float(self.hours_since_start(t))))
        return int(np.clip(idx, 0, len(self.times) - 1))

    def snapshot(self, t):
        """Grid of u/v for the hour nearest t (for drawing wind on the map)."""
        i = self.nearest_hour_index(t)
        return {
            "time": str(self.times[i]) + "Z",
            "lats": self.lats.round(3).tolist(),
            "lons": self.lons.round(3).tolist(),
            "u": np.round(self.u[i], 2).tolist(),
            "v": np.round(self.v[i], 2).tolist(),
        }


def _build_params(lats, lons):
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return {
        "latitude": ",".join(f"{x:g}" for x in LA.ravel()),
        "longitude": ",".join(f"{x:g}" for x in LO.ravel()),
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "past_days": PAST_DAYS,
        "forecast_days": 1,
        "timezone": "GMT",
    }


def _chunks(params):
    """Split one multi-location param set into several smaller requests."""
    la = params["latitude"].split(",")
    lo = params["longitude"].split(",")
    for i in range(0, len(la), MAX_LOCATIONS_PER_REQUEST):
        p = dict(params)
        p["latitude"] = ",".join(la[i:i + MAX_LOCATIONS_PER_REQUEST])
        p["longitude"] = ",".join(lo[i:i + MAX_LOCATIONS_PER_REQUEST])
        yield p


def _fetch(params):
    payload = []
    for p in _chunks(params):
        try:
            resp = requests.get(OPEN_METEO_URL, params=p, timeout=60)
        except requests.RequestException as e:
            raise WindDataError(f"Network error calling Open-Meteo: {e}") from e
        if resp.status_code != 200:
            raise WindDataError(f"Open-Meteo returned HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            part = resp.json()
        except ValueError as e:
            raise WindDataError("Open-Meteo did not return JSON") from e
        payload.extend(part if isinstance(part, list) else [part])
    return payload


def parse_response(payload, lats, lons) -> WindField:
    """Turn Open-Meteo's multi-location JSON list into a WindField."""
    ny, nx = len(lats), len(lons)
    if isinstance(payload, dict):
        if payload.get("error"):
            raise WindDataError(f"Open-Meteo error: {payload.get('reason')}")
        payload = [payload]
    if not isinstance(payload, list) or len(payload) != ny * nx:
        n = len(payload) if isinstance(payload, list) else "?"
        raise WindDataError(f"Expected {ny * nx} grid points from Open-Meteo, got {n}")

    try:
        times = np.array(payload[0]["hourly"]["time"], dtype="datetime64[s]")
        nt = len(times)
        u = np.full((nt, ny * nx), np.nan)
        v = np.full((nt, ny * nx), np.nan)
        for k, loc in enumerate(payload):
            hr = loc["hourly"]
            s10 = np.array(hr["wind_speed_10m"], dtype=float)
            d10 = np.array(hr["wind_direction_10m"], dtype=float)
            u10, v10 = speed_dir_to_uv(s10, d10)
            if "wind_speed_925hPa" in hr:
                s925 = np.array(hr["wind_speed_925hPa"], dtype=float)
                d925 = np.array(hr["wind_direction_925hPa"], dtype=float)
                u9, v9 = speed_dir_to_uv(s925, d925)
            else:
                u9 = v9 = np.full_like(u10, np.nan)
            if (loc.get("elevation") or 0) > HIGH_TERRAIN_M:
                uu, vv = u10, v10
            else:
                # Average the two levels; fall back to whichever exists.
                uu = np.nanmean(np.vstack([u10, u9]), axis=0)
                vv = np.nanmean(np.vstack([v10, v9]), axis=0)
            u[:, k], v[:, k] = uu, vv
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise WindDataError(f"Unexpected Open-Meteo response shape: {e}") from e

    # Fill any remaining gaps with 0 (calm) so trajectories don't break.
    u = np.nan_to_num(u).reshape(nt, ny, nx)
    v = np.nan_to_num(v).reshape(nt, ny, nx)
    if nt < 2:
        raise WindDataError("Open-Meteo returned fewer than 2 hourly steps")
    return WindField(lats=np.asarray(lats, float), lons=np.asarray(lons, float),
                     times=times, u=u, v=v)


def get_wind(step: float = GRID_STEP) -> WindField:
    """Fetch the hourly transport-wind field for South Asia."""
    lats, lons = grid_axes(step)
    return parse_response(_fetch(_build_params(lats, lons)), lats, lons)


def get_global_wind() -> WindField:
    """
    Fetch today's hourly 10 m wind on a coarse 10-degree worldwide grid.
    Only used to draw the wind animation outside South Asia.
    """
    params = _build_params(GLOBAL_LATS, GLOBAL_LONS)
    params.update(hourly="wind_speed_10m,wind_direction_10m", past_days=0)
    return parse_response(_fetch(params), GLOBAL_LATS, GLOBAL_LONS)


if __name__ == "__main__":
    w = get_wind()
    print(f"Wind grid {len(w.lats)}x{len(w.lons)}, {len(w.times)} hours "
          f"from {w.times[0]} to {w.times[-1]}")
