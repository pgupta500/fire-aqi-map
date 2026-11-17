"""
Point -> region lookup for South Asia (Indian states/UTs, Pakistani
provinces, Bangladeshi divisions, other South Asian countries, and
neighbouring countries) using the bundled data/regions.geojson
(Natural Earth 1:10m admin-1, public domain, simplified).
"""

import json
import os
from functools import lru_cache

import numpy as np
import shapely
from shapely.geometry import shape

REGIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "regions.geojson")
UNKNOWN = "Outside mapped area"

SOUTH_ASIA_COUNTRIES = {
    "India", "Pakistan", "Bangladesh", "Nepal", "Bhutan", "Sri Lanka", "Maldives", "Afghanistan",
}


@lru_cache(maxsize=1)
def _load():
    with open(REGIONS_PATH, encoding="utf-8") as f:
        gj = json.load(f)
    names, countries, geoms = [], [], []
    for feat in gj["features"]:
        names.append(feat["properties"]["name"])
        countries.append(feat["properties"]["country"])
        g = shape(feat["geometry"])
        shapely.prepare(g)
        geoms.append(g)
    return names, countries, geoms


def region_names():
    return list(_load()[0])


def assign_regions(lat, lon):
    """
    Vectorised lookup. Returns (region_names, countries) as numpy object
    arrays aligned with the input points. Points in no polygon get UNKNOWN.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    names, countries, geoms = _load()
    out_name = np.full(lat.shape, UNKNOWN, dtype=object)
    out_country = np.full(lat.shape, "", dtype=object)
    todo = np.ones(lat.shape, dtype=bool)
    for name, country, g in zip(names, countries, geoms):
        if not todo.any():
            break
        minx, miny, maxx, maxy = g.bounds
        cand = todo & (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        if not cand.any():
            continue
        idx = np.nonzero(cand)[0]
        inside = shapely.contains_xy(g, lon[idx], lat[idx])
        hit = idx[inside]
        out_name[hit] = name
        out_country[hit] = country
        todo[hit] = False

    # Coastal stations/fires can fall just outside simplified coastlines:
    # snap them to the nearest region within ~25 km.
    if todo.any():
        idx = np.nonzero(todo)[0]
        pts = shapely.points(lon[idx], lat[idx])
        best_d = np.full(len(idx), np.inf)
        best_k = np.full(len(idx), -1)
        for k, g in enumerate(geoms):
            d = shapely.distance(g, pts)
            better = d < best_d
            best_d[better] = d[better]
            best_k[better] = k
        close = best_d < 0.25
        for j in np.nonzero(close)[0]:
            out_name[idx[j]] = names[best_k[j]]
            out_country[idx[j]] = countries[best_k[j]]
    return out_name, out_country
