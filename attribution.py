"""
Wind-aware fire -> air quality attribution ("who is burning whose air").

For every PM2.5 station we run a HYSPLIT-style *back-trajectory*: starting at
the station at the time of its reading, we step backwards hour by hour through
the Open-Meteo wind field to find where that air was over the previous
24-48 hours. Fires that burned close to that path, at about the time the air
passed over them, are credited with the smoke that reached the station.

    contribution(fire -> station) = FRP x max over path points k of
        exp(-d^2 / 2 sigma_k^2)       closeness to the path (corridor widens
                                      with time to reflect dispersion)
      x exp(-dt^2 / 2 * 6h^2)         fire was burning when the air passed
      x exp(-k / 36h)                 older smoke is more diluted

Contributions are then rolled up by the region the fire was in (Indian
state / Pakistani province / country) to build an accountability ledger:
which regions' fires delivered smoke to which other regions' monitors.

This is a simplified Lagrangian model (single transport level, no
chemistry, no deposition, no plume rise). It shows where smoke most likely
came from; it is not a regulatory source-apportionment study.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from correlate import HOTSPOT_MIN_PM25, naqi_category, spearman
from regions import UNKNOWN, assign_regions

EARTH_R_KM = 6371.0
KM_PER_DEG = 111.195

DEFAULT_HOURS = 48
CORRIDOR_BASE_KM = 25.0      # path half-width at the station
CORRIDOR_SPREAD_KM_H = 1.0   # widening per hour back in time
TIME_SIGMA_H = 6.0           # how long a detected fire is assumed to smoke
TIME_CUTOFF_H = 15.0
AGE_TAU_H = 36.0
# VIIRS overpasses India ~13:30 local (08:00 UTC); used if acq_time missing.
DEFAULT_ACQ_HHMM = 800

# A station counts as "receiving fire smoke" above this score (~ one 25 MW
# fire directly on the path at the right time).
SMOKE_MIN_SCORE = 25.0
# A region "impacts" a polluted station if it supplied >= this share.
IMPACT_MIN_SHARE = 0.20


# --------------------------------------------------------------------------
# Geometry helpers

def _xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    c = np.cos(lat)
    return np.column_stack([EARTH_R_KM * c * np.cos(lon),
                            EARTH_R_KM * c * np.sin(lon),
                            EARTH_R_KM * np.sin(lat)])


def _chord_to_arc(chord_km):
    return 2 * EARTH_R_KM * np.arcsin(np.clip(chord_km / (2 * EARTH_R_KM), 0, 1))


# --------------------------------------------------------------------------
# Back-trajectories

def back_trajectories(wind, lat0, lon0, t0, hours=DEFAULT_HOURS):
    """
    Integrate backwards in time from each start point using midpoint (RK2)
    steps of 1 hour.

    Returns lat, lon arrays of shape (n, hours + 1); column k is where the
    air was k hours before t0. Steps that leave the wind grid (in space or
    time) are NaN, and so is everything after them.
    """
    n = len(lat0)
    lat = np.full((n, hours + 1), np.nan)
    lon = np.full((n, hours + 1), np.nan)
    lat[:, 0], lon[:, 0] = lat0, lon0
    h0 = wind.hours_since_start(t0)
    dt = 3600.0

    def step(la, lo, u, v, scale):
        dlat = -v * dt * scale / (KM_PER_DEG * 1000.0)
        dlon = -u * dt * scale / (KM_PER_DEG * 1000.0 * np.cos(np.radians(la)))
        return la + dlat, lo + dlon

    for k in range(hours):
        la, lo = lat[:, k], lon[:, k]
        hk = h0 - k
        u1, v1 = wind.sample(la, lo, hk)
        mla, mlo = step(la, lo, u1, v1, 0.5)
        u2, v2 = wind.sample(mla, mlo, hk - 0.5)
        lat[:, k + 1], lon[:, k + 1] = step(la, lo, u2, v2, 1.0)
    # Once a step is NaN, keep it NaN.
    bad = np.cumsum(~np.isfinite(lat), axis=1) > 0
    lat[bad] = np.nan
    lon[bad] = np.nan
    return lat, lon


# --------------------------------------------------------------------------
# Fires

def fire_times(fires: pd.DataFrame) -> np.ndarray:
    """acq_date + acq_time (HHMM UTC) -> datetime64[s]."""
    if "acq_time" in fires.columns:
        hhmm = pd.to_numeric(fires["acq_time"], errors="coerce").fillna(DEFAULT_ACQ_HHMM)
    else:
        hhmm = pd.Series(DEFAULT_ACQ_HHMM, index=fires.index)
    hhmm = hhmm.astype(int)
    base = pd.to_datetime(fires["acq_date"], errors="coerce", utc=True)
    t = base + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m")
    return t.dt.tz_localize(None).to_numpy(dtype="datetime64[s]")


def cluster_fires(fires: pd.DataFrame, cell_deg: float = 0.1) -> pd.DataFrame:
    """
    Merge detections in the same ~10 km cell and the same hour. A satellite
    swath over Punjab can contain thousands of pixels; clustering keeps the
    attribution fast without changing the result meaningfully.
    """
    if fires is None or fires.empty:
        return pd.DataFrame(columns=["latitude", "longitude", "frp", "count", "time",
                                     "region", "country"])
    df = pd.DataFrame({
        "latitude": fires["latitude"].astype(float).to_numpy(),
        "longitude": fires["longitude"].astype(float).to_numpy(),
        "frp": pd.to_numeric(fires["frp"], errors="coerce").fillna(0).clip(lower=0).to_numpy(),
        "time": fire_times(fires),
    }).dropna(subset=["time"])
    df["cy"] = np.floor(df["latitude"] / cell_deg).astype(int)
    df["cx"] = np.floor(df["longitude"] / cell_deg).astype(int)
    df["hour"] = df["time"].dt.floor("h")
    # FRP-weighted position so the cluster sits on its hottest pixels.
    df["w"] = df["frp"] + 1e-3
    df["wlat"] = df["latitude"] * df["w"]
    df["wlon"] = df["longitude"] * df["w"]
    g = df.groupby(["cy", "cx", "hour"], sort=False).agg(
        wlat=("wlat", "sum"), wlon=("wlon", "sum"), w=("w", "sum"),
        frp=("frp", "sum"), count=("frp", "size"), time=("time", "mean"))
    out = pd.DataFrame({
        "latitude": (g["wlat"] / g["w"]).to_numpy(),
        "longitude": (g["wlon"] / g["w"]).to_numpy(),
        "frp": g["frp"].to_numpy(),
        "count": g["count"].to_numpy(),
        "time": g["time"].to_numpy(dtype="datetime64[s]"),
    })
    out["region"], out["country"] = assign_regions(out["latitude"], out["longitude"])
    return out


# --------------------------------------------------------------------------
# Attribution

@dataclass
class Attribution:
    stations: pd.DataFrame          # one row per station, enriched
    clusters: pd.DataFrame          # fire clusters
    pairs: pd.DataFrame             # station, cluster, contribution, k
    traj_lat: np.ndarray            # (n_stations, hours+1)
    traj_lon: np.ndarray
    traj_t0: np.ndarray             # datetime64 per station
    hours: int
    ledger: list = field(default_factory=list)
    flows: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _station_times(stations, wind):
    t = pd.to_datetime(stations["datetime_utc"], errors="coerce", utc=True)
    t = t.dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    last = wind.times[-1]
    nat = np.isnat(t)
    t = np.where(nat, last, t)
    return np.minimum(t, last).astype("datetime64[s]")


def attribute(stations: pd.DataFrame, fires: pd.DataFrame, wind,
              hours: int = DEFAULT_HOURS) -> Attribution:
    st = stations.copy().reset_index(drop=True)
    n = len(st)
    st["station_id"] = np.arange(n)
    st["region"], st["country"] = assign_regions(st["latitude"], st["longitude"])
    st["category"] = [naqi_category(v) for v in st["pm25"]]

    t0 = _station_times(st, wind)
    tlat, tlon = back_trajectories(wind, st["latitude"].to_numpy(float),
                                   st["longitude"].to_numpy(float), t0, hours)
    clusters = cluster_fires(fires)

    pairs = _pairs(tlat, tlon, t0, clusters, hours)

    # ---- per-station totals and source mix ------------------------------
    score = np.zeros(n)
    if len(pairs):
        tot = pairs.groupby("station")["contribution"].sum()
        score[tot.index.to_numpy()] = tot.to_numpy()
    st["upwind_score"] = score.round(1)
    st["trajectory_hours"] = np.isfinite(tlat).sum(axis=1) - 1

    by_src = pd.DataFrame(columns=["station", "source", "contribution"])
    if len(pairs):
        p = pairs.merge(clusters[["region"]], left_on="cluster", right_index=True)
        by_src = (p.groupby(["station", "region"])["contribution"].sum()
                  .reset_index().rename(columns={"region": "source"}))
    tops, top_src, top_share, local_share = [], [], [], []
    grouped = {k: g for k, g in by_src.groupby("station")} if len(by_src) else {}
    for i in range(n):
        g = grouped.get(i)
        if g is None or score[i] <= 0:
            tops.append([]); top_src.append(None); top_share.append(None); local_share.append(None)
            continue
        g = g.sort_values("contribution", ascending=False)
        shares = g["contribution"] / score[i]
        tops.append([{"region": r, "share": round(float(s), 3)}
                     for r, s in zip(g["source"].head(5), shares.head(5))])
        top_src.append(g["source"].iloc[0])
        top_share.append(round(float(shares.iloc[0]), 3))
        own = g.loc[g["source"] == st.at[i, "region"], "contribution"].sum()
        local_share.append(round(float(own / score[i]), 3))
    st["sources"] = tops
    st["top_source"] = top_src
    st["top_source_share"] = top_share
    st["local_share"] = local_share
    st["smoke_affected"] = st["upwind_score"] >= SMOKE_MIN_SCORE

    att = Attribution(stations=st, clusters=clusters, pairs=pairs,
                      traj_lat=tlat, traj_lon=tlon, traj_t0=t0, hours=hours)
    att.ledger = _ledger(st, clusters, by_src, pairs)
    att.flows = _flows(st, clusters, pairs)
    att.summary = _summary(st, clusters, att.ledger)
    return att


def _pairs(tlat, tlon, t0, clusters, hours):
    """All (station, cluster) contributions, maximised over path points."""
    cols = ["station", "cluster", "contribution", "k"]
    if clusters.empty:
        return pd.DataFrame(columns=cols)
    n, K = tlat.shape
    sidx, kidx = np.nonzero(np.isfinite(tlat))
    if len(sidx) == 0:
        return pd.DataFrame(columns=cols)
    plat, plon = tlat[sidx, kidx], tlon[sidx, kidx]

    rmax = CORRIDOR_BASE_KM + CORRIDOR_SPREAD_KM_H * hours
    path_tree = cKDTree(_xyz(plat, plon))
    fire_tree = cKDTree(_xyz(clusters["latitude"].to_numpy(), clusters["longitude"].to_numpy()))
    sdm = path_tree.sparse_distance_matrix(fire_tree, rmax, output_type="ndarray")
    if len(sdm) == 0:
        return pd.DataFrame(columns=cols)
    pi, ci = sdm["i"], sdm["j"]
    d = _chord_to_arc(sdm["v"])
    k = kidx[pi]
    radius = CORRIDOR_BASE_KM + CORRIDOR_SPREAD_KM_H * k
    keep = d <= radius
    pi, ci, d, k, radius = pi[keep], ci[keep], d[keep], k[keep], radius[keep]

    t_air = t0[sidx[pi]] - (k * 3600).astype("timedelta64[s]")
    t_fire = clusters["time"].to_numpy(dtype="datetime64[s]")[ci]
    dt_h = (t_air - t_fire).astype(float) / 3600.0
    # Fire must have been detected before the reading (1 h grace) and be
    # burning roughly when the air passed.
    t_station = t0[sidx[pi]]
    ok = (np.abs(dt_h) <= TIME_CUTOFF_H) & (t_fire <= t_station + np.timedelta64(3600, "s"))
    pi, ci, d, k, radius, dt_h = pi[ok], ci[ok], d[ok], k[ok], radius[ok], dt_h[ok]

    sigma = radius / 2.0
    w = (np.exp(-0.5 * (d / sigma) ** 2)
         * np.exp(-0.5 * (dt_h / TIME_SIGMA_H) ** 2)
         * np.exp(-k / AGE_TAU_H))
    contrib = clusters["frp"].to_numpy()[ci] * w
    df = pd.DataFrame({"station": sidx[pi], "cluster": ci, "contribution": contrib, "k": k})
    if df.empty:
        return pd.DataFrame(columns=cols)
    best = df.loc[df.groupby(["station", "cluster"])["contribution"].idxmax()]
    return best.reset_index(drop=True)


def _ledger(st, clusters, by_src, pairs):
    """Per source region: fires, smoke delivered, exported, who it hit."""
    if clusters.empty:
        return []
    fires_by_region = clusters.groupby("region").agg(fires=("count", "sum"), frp=("frp", "sum"))
    recv_region = st["region"].to_numpy()
    totals = st["upwind_score"].to_numpy()
    polluted = st["pm25"].to_numpy() >= HOTSPOT_MIN_PM25

    rows = []
    src_groups = {k: g for k, g in by_src.groupby("source")} if len(by_src) else {}
    for region, fr in fires_by_region.iterrows():
        g = src_groups.get(region)
        delivered = exported = 0.0
        impacted = 0
        receivers = []
        if g is not None:
            sids = g["station"].to_numpy()
            c = g["contribution"].to_numpy()
            delivered = float(c.sum())
            other = recv_region[sids] != region
            exported = float(c[other].sum())
            share = c / np.where(totals[sids] > 0, totals[sids], 1)
            impacted = int(((share >= IMPACT_MIN_SHARE) & polluted[sids]
                            & (totals[sids] >= SMOKE_MIN_SCORE)).sum())
            rec = pd.DataFrame({"region": recv_region[sids], "c": c})
            rec = rec[rec["region"] != region].groupby("region")["c"].sum()
            for rname, val in rec.sort_values(ascending=False).head(5).items():
                tot_r = totals[recv_region == rname].sum()
                receivers.append({
                    "region": rname,
                    "smoke": round(float(val), 1),
                    "share_of_receiver": round(float(val / tot_r), 3) if tot_r > 0 else None,
                })
        rows.append({
            "region": region,
            "fires": int(fr["fires"]),
            "frp": round(float(fr["frp"]), 1),
            "smoke_delivered": round(delivered, 1),
            "smoke_exported": round(exported, 1),
            "export_ratio": round(exported / delivered, 3) if delivered > 0 else None,
            "polluted_stations_impacted": impacted,
            "receivers": receivers,
        })
    total_exp = sum(r["smoke_exported"] for r in rows) or 1.0
    for r in rows:
        r["share_of_all_exported"] = round(r["smoke_exported"] / total_exp, 3)
    rows.sort(key=lambda r: (r["smoke_exported"], r["smoke_delivered"]), reverse=True)
    return rows


def _flows(st, clusters, pairs, limit=15):
    """Cross-border smoke flows source region -> receiving region."""
    if pairs.empty:
        return []
    p = pairs.merge(clusters[["region", "latitude", "longitude"]],
                    left_on="cluster", right_index=True)
    p["recv"] = st["region"].to_numpy()[p["station"].to_numpy()]
    p = p[(p["region"] != p["recv"]) & (p["region"] != UNKNOWN) & (p["recv"] != UNKNOWN)]
    if p.empty:
        return []
    p["wlat"] = p["latitude"] * p["contribution"]
    p["wlon"] = p["longitude"] * p["contribution"]
    g = p.groupby(["region", "recv"]).agg(smoke=("contribution", "sum"),
                                          wlat=("wlat", "sum"), wlon=("wlon", "sum"),
                                          stations=("station", "nunique"))
    g = g.sort_values("smoke", ascending=False).head(limit)
    recv_tot = st.groupby("region")["upwind_score"].sum()
    recv_pos = st.groupby("region")[["latitude", "longitude"]].mean()
    out = []
    for (src, recv), row in g.iterrows():
        out.append({
            "source": src, "receiver": recv,
            "smoke": round(float(row["smoke"]), 1),
            "share_of_receiver": round(float(row["smoke"] / recv_tot[recv]), 3)
            if recv_tot.get(recv, 0) > 0 else None,
            "stations": int(row["stations"]),
            "from": [round(row["wlat"] / row["smoke"], 3), round(row["wlon"] / row["smoke"], 3)],
            "to": [round(float(recv_pos.at[recv, "latitude"]), 3),
                   round(float(recv_pos.at[recv, "longitude"]), 3)],
        })
    return out


def _summary(st, clusters, ledger):
    polluted = st[st["pm25"] >= HOTSPOT_MIN_PM25]
    affected = st[st["smoke_affected"]]
    clean = st[~st["smoke_affected"]]

    def mean(s):
        return round(float(s.mean()), 1) if len(s) else None

    exported = [r for r in ledger if r["smoke_exported"] > 0]
    top = exported[0] if exported else None
    return {
        "stations": int(len(st)),
        "stations_with_trajectory": int((st["trajectory_hours"] > 0).sum()),
        "fire_clusters": int(len(clusters)),
        "smoke_affected_stations": int(len(affected)),
        "polluted_stations": int(len(polluted)),
        "polluted_and_smoke_affected": int(polluted["smoke_affected"].sum()) if len(polluted) else 0,
        "mean_pm25_smoke_affected": mean(affected["pm25"]),
        "mean_pm25_not_affected": mean(clean["pm25"]),
        "spearman_upwind_vs_pm25": _r(spearman(st["upwind_score"], st["pm25"])),
        "top_exporter": top["region"] if top else None,
        "top_exporter_share": top["share_of_all_exported"] if top else None,
    }


def _r(x):
    return None if x is None else round(x, 3)


# --------------------------------------------------------------------------
# Per-station detail for the map

def station_detail(att: Attribution, station_id: int, max_fires: int = 300) -> dict:
    st = att.stations
    if not 0 <= station_id < len(st):
        raise KeyError(station_id)
    row = st.iloc[station_id]
    la, lo = att.traj_lat[station_id], att.traj_lon[station_id]
    ok = np.isfinite(la)
    t0 = att.traj_t0[station_id]
    path = [{"lat": round(float(a), 4), "lon": round(float(b), 4), "hours_ago": int(k),
             "time": str(t0 - np.timedelta64(int(k) * 3600, "s")) + "Z"}
            for k, (a, b) in enumerate(zip(la, lo)) if ok[k]]

    fires = []
    if len(att.pairs):
        p = att.pairs[att.pairs["station"] == station_id].nlargest(max_fires, "contribution")
        c = att.clusters.loc[p["cluster"].to_numpy()]
        total = row["upwind_score"] or 1.0
        for (_, pr), (_, cr) in zip(p.iterrows(), c.iterrows()):
            fires.append({
                "lat": round(float(cr["latitude"]), 4), "lon": round(float(cr["longitude"]), 4),
                "frp": round(float(cr["frp"]), 1), "count": int(cr["count"]),
                "region": cr["region"], "time": str(cr["time"]) + "Z",
                "hours_before": int(pr["k"]),
                "share": round(float(pr["contribution"] / total), 4),
            })
    return {
        "station": _jsonable(row.to_dict()),
        "path": path,
        "fires": fires,
    }


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.bool_,)):
            v = bool(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = None if np.isnan(v) else float(v)
        elif isinstance(v, float) and np.isnan(v):
            v = None
        out[k] = v
    return out
