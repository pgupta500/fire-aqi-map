"""
Fire -> air quality accountability analysis.

For every PM2.5 station, measures how much fire activity happened nearby
(count, total intensity, a distance-weighted "fire influence" score, and the
nearest fire). Across all stations it reports a rank correlation between fire
influence and PM2.5, and flags "hotspots": stations with unhealthy air AND
heavy nearby burning.

Proximity is not proof of causation: this ignores wind, terrain and other
pollution sources (traffic, industry, dust). It points to where to look.
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

# Fires farther than this from a station are ignored.
DEFAULT_RADIUS_KM = 150.0

# Stations are processed in chunks to keep the distance matrix small
# (chunk x n_fires floats) even during peak stubble-burning season.
_CHUNK = 64

# India National AQI (NAQI) 24-hour PM2.5 breakpoints, in ug/m3.
NAQI_PM25 = [
    (30, "Good"),
    (60, "Satisfactory"),
    (90, "Moderate"),
    (120, "Poor"),
    (250, "Very Poor"),
    (float("inf"), "Severe"),
]

# A hotspot needs at least "Poor" air plus real nearby fire activity.
HOTSPOT_MIN_PM25 = 90.0
HOTSPOT_MIN_FIRES = 5


def naqi_category(pm25: float) -> str:
    """Map a PM2.5 value (ug/m3) to its India NAQI category name."""
    for upper, name in NAQI_PM25:
        if pm25 <= upper:
            return name
    return "Severe"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Broadcasts over numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def station_fire_exposure(
    stations: pd.DataFrame,
    fires: pd.DataFrame,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> pd.DataFrame:
    """
    Attach nearby-fire metrics to each station.

    Adds columns:
        fires_nearby      number of fire detections within radius_km
        frp_nearby        summed fire radiative power (MW) within radius_km
        fire_influence    sum of FRP * (1 - d / radius): closer, hotter fires
                          count more; 0 when nothing is in range
        nearest_fire_km   distance to the closest fire (NaN if no fires at all)
        category          NAQI category for the station's PM2.5
    """
    if radius_km <= 0:
        raise ValueError("radius_km must be positive")

    out = stations.copy().reset_index(drop=True)
    n = len(out)
    count = np.zeros(n, dtype=int)
    frp_sum = np.zeros(n)
    influence = np.zeros(n)
    nearest = np.full(n, np.nan)

    if n and fires is not None and len(fires):
        f_lat = fires["latitude"].to_numpy(dtype=float)
        f_lon = fires["longitude"].to_numpy(dtype=float)
        f_frp = np.nan_to_num(
            pd.to_numeric(fires["frp"], errors="coerce").to_numpy(dtype=float)
        ).clip(min=0)
        s_lat = out["latitude"].to_numpy(dtype=float)
        s_lon = out["longitude"].to_numpy(dtype=float)

        for start in range(0, n, _CHUNK):
            sl = slice(start, start + _CHUNK)
            d = haversine_km(
                s_lat[sl, None], s_lon[sl, None], f_lat[None, :], f_lon[None, :]
            )
            within = d <= radius_km
            weight = np.where(within, 1.0 - d / radius_km, 0.0)
            count[sl] = within.sum(axis=1)
            frp_sum[sl] = (within * f_frp).sum(axis=1)
            influence[sl] = (weight * f_frp).sum(axis=1)
            nearest[sl] = d.min(axis=1)

    out["fires_nearby"] = count
    out["frp_nearby"] = frp_sum.round(1)
    out["fire_influence"] = influence.round(1)
    out["nearest_fire_km"] = np.round(nearest, 1)
    out["category"] = [naqi_category(v) for v in out["pm25"]] if n else []
    return out


def spearman(x, y):
    """Spearman rank correlation; None if undefined (too few / constant)."""
    x = pd.Series(x, dtype=float).reset_index(drop=True)
    y = pd.Series(y, dtype=float).reset_index(drop=True)
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    rx, ry = x.rank(), y.rank()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def find_hotspots(exposure: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Stations with unhealthy PM2.5 and heavy nearby burning, worst first."""
    hot = exposure[
        (exposure["pm25"] >= HOTSPOT_MIN_PM25)
        & (exposure["fires_nearby"] >= HOTSPOT_MIN_FIRES)
    ]
    return hot.sort_values(
        ["fire_influence", "pm25"], ascending=False
    ).head(limit)


def summarize(
    exposure: pd.DataFrame, n_fires: int, radius_km: float = DEFAULT_RADIUS_KM
) -> dict:
    """Headline numbers for the map's summary panel."""
    near = exposure[exposure["fires_nearby"] > 0]
    far = exposure[exposure["fires_nearby"] == 0]

    def mean_or_none(s):
        return round(float(s.mean()), 1) if len(s) else None

    rho = spearman(exposure["fire_influence"], exposure["pm25"])
    return {
        "radius_km": radius_km,
        "fires": int(n_fires),
        "stations": int(len(exposure)),
        "stations_with_fires_nearby": int(len(near)),
        "mean_pm25_near_fires": mean_or_none(near["pm25"]),
        "mean_pm25_no_fires": mean_or_none(far["pm25"]),
        "spearman_fire_influence_vs_pm25": None if rho is None else round(rho, 3),
        "hotspots": int(len(find_hotspots(exposure, limit=len(exposure) or 1))),
        "note": (
            "Proximity-based association only; wind, terrain and non-fire "
            "pollution sources are not modelled."
        ),
    }
