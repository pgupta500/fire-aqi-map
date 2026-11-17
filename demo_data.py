"""
Synthetic demo data for offline development (DEMO_MODE=1).

Builds a plausible post-monsoon scene: north-westerly winds over the
Indo-Gangetic plain, heavy crop-residue burning in Punjab/Haryana (both
sides of the border), scattered fires elsewhere, and PM2.5 monitors at real
city locations. None of it is real data; the UI shows a DEMO banner.
"""

import numpy as np
import pandas as pd

from wind import WindField, grid_axes, speed_dir_to_uv

CITIES = [
    ("Delhi", 28.61, 77.21, 18), ("Gurugram", 28.46, 77.03, 4), ("Noida", 28.54, 77.39, 4),
    ("Ghaziabad", 28.67, 77.45, 3), ("Faridabad", 28.41, 77.32, 3), ("Ludhiana", 30.90, 75.85, 3),
    ("Amritsar", 31.63, 74.87, 2), ("Patiala", 30.34, 76.39, 2), ("Bathinda", 30.21, 74.95, 2),
    ("Chandigarh", 30.73, 76.78, 3), ("Panipat", 29.39, 76.97, 2), ("Hisar", 29.15, 75.72, 2),
    ("Karnal", 29.69, 76.99, 2), ("Jaipur", 26.91, 75.79, 4), ("Jodhpur", 26.24, 73.02, 2),
    ("Lucknow", 26.85, 80.95, 5), ("Kanpur", 26.45, 80.33, 4), ("Agra", 27.18, 78.01, 3),
    ("Meerut", 28.98, 77.71, 2), ("Varanasi", 25.32, 82.97, 3), ("Prayagraj", 25.44, 81.85, 2),
    ("Patna", 25.59, 85.14, 4), ("Muzaffarpur", 26.12, 85.39, 2), ("Kolkata", 22.57, 88.36, 6),
    ("Dehradun", 30.32, 78.03, 2), ("Bhopal", 23.26, 77.41, 3), ("Indore", 22.72, 75.86, 3),
    ("Gwalior", 26.22, 78.18, 2), ("Ahmedabad", 23.02, 72.57, 5), ("Surat", 21.17, 72.83, 3),
    ("Mumbai", 19.08, 72.88, 10), ("Pune", 18.52, 73.86, 5), ("Nagpur", 21.15, 79.09, 3),
    ("Raipur", 21.25, 81.63, 2), ("Bhubaneswar", 20.30, 85.82, 2), ("Hyderabad", 17.39, 78.49, 7),
    ("Visakhapatnam", 17.69, 83.22, 2), ("Bengaluru", 12.97, 77.59, 9), ("Chennai", 13.08, 80.27, 7),
    ("Kochi", 9.93, 76.27, 2), ("Thiruvananthapuram", 8.52, 76.94, 2), ("Coimbatore", 11.02, 76.96, 2),
    ("Guwahati", 26.14, 91.74, 3), ("Shillong", 25.58, 91.89, 1), ("Siliguri", 26.73, 88.40, 1),
    ("Ranchi", 23.34, 85.31, 2), ("Srinagar", 34.08, 74.80, 1), ("Jammu", 32.73, 74.86, 1),
]

FIRE_ZONES = [
    # (lat, lon, sd_lat, sd_lon, n, frp_scale)
    (30.6, 75.4, 0.45, 0.70, 2600, 9),   # Punjab (India)
    (29.7, 76.4, 0.35, 0.45, 1100, 8),   # Haryana
    (31.0, 73.6, 0.45, 0.55, 700, 8),    # Punjab (Pakistan)
    (27.5, 79.0, 0.80, 1.20, 300, 5),    # western UP
    (23.2, 78.0, 0.90, 1.40, 350, 6),    # Madhya Pradesh
    (24.0, 92.8, 0.80, 0.60, 250, 7),    # Manipur/Mizoram/Assam
    (19.5, 82.5, 1.00, 1.00, 120, 5),    # Odisha/Chhattisgarh
]


def demo_wind(now: np.datetime64) -> WindField:
    lats, lons = grid_axes()
    hours = np.arange(-24 * 4, 25)
    times = (now.astype("datetime64[h]") + hours.astype("timedelta64[h]")).astype("datetime64[s]")
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    # North of ~22N: north-westerly (from 300-320 deg); south: north-easterly.
    blend = 1 / (1 + np.exp(-(LA - 22) / 1.5))
    direction = blend * (310 + 10 * np.sin(LO / 6)) + (1 - blend) * 50
    speed = 2.5 + 2.5 * blend
    u = np.empty((len(times), *LA.shape))
    v = np.empty_like(u)
    for i, h in enumerate(hours):
        diurnal = 1 + 0.35 * np.sin(2 * np.pi * (h % 24 - 3) / 24)
        uu, vv = speed_dir_to_uv(speed * diurnal, direction + 8 * np.sin(h / 9))
        u[i], v[i] = uu, vv
    return WindField(lats=lats, lons=lons, times=times, u=u, v=v)


def demo_fires(now: np.datetime64, days: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    today = now.astype("datetime64[D]")
    for lat, lon, sla, slo, n, scale in FIRE_ZONES:
        n = int(n * days / 5)
        la = rng.normal(lat, sla, n)
        lo = rng.normal(lon, slo, n)
        day = today - rng.integers(0, days, n).astype("timedelta64[D]")
        # VIIRS overpasses ~08:00 UTC (day) and ~20:00 UTC (night, fewer fires)
        night = rng.random(n) < 0.2
        hhmm = np.where(night, 2000, 800) + rng.integers(-40, 40, n)
        frp = np.round(rng.gamma(1.6, scale, n), 1)
        for a, b, d, t, f in zip(la, lo, day, hhmm, frp):
            rows.append((a, b, str(d), int(t), "n", f))
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "acq_date", "acq_time",
                                     "confidence", "frp"])
    # Don't emit fires in the future.
    t = pd.to_datetime(df["acq_date"]) + pd.to_timedelta(df["acq_time"] // 100, "h")
    return df[t <= pd.Timestamp(now)].reset_index(drop=True)


def demo_stations(now: np.datetime64, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    stamp = str(now.astype("datetime64[h]") - np.timedelta64(1, "h")) + ":00:00Z"
    sid = 0
    for city, lat, lon, n in CITIES:
        for j in range(n):
            sid += 1
            rows.append({
                "name": f"{city} site {j + 1}" if n > 1 else city,
                "latitude": round(lat + rng.normal(0, 0.06), 4),
                "longitude": round(lon + rng.normal(0, 0.06), 4),
                "pm25": 0.0, "datetime_utc": stamp, "location_id": 900000 + sid,
            })
    return pd.DataFrame(rows)


def demo_scene(days: int = 5):
    """(stations, fires, wind) with PM2.5 made consistent with the smoke."""
    from attribution import attribute

    now = np.datetime64(pd.Timestamp.now("UTC").tz_localize(None).floor("h"), "s")
    wind = demo_wind(now)
    fires = demo_fires(now, days)
    st = demo_stations(now)
    att = attribute(st, fires, wind, hours=48)
    rng = np.random.default_rng(3)
    lat = st["latitude"].to_numpy()
    urban = 35 + 45 * np.exp(-((lat - 27.5) / 4.0) ** 2)   # IGP winter baseline
    smoke = 14 * np.sqrt(att.stations["upwind_score"].to_numpy())
    st["pm25"] = np.round(np.clip(urban + smoke + rng.normal(0, 12, len(st)), 8, 600), 1)
    return st, fires, wind
