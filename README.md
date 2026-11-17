# SmokeTrail · fire-aqi-map

**Who's burning whose air?** A wind-aware accountability map for India's crop and forest fire smoke.

Every winter, stubble burning across Punjab and Haryana coincides with some of the worst air in the world hundreds of kilometres away. Seeing fires and bad air in the same region doesn't show who is responsible. SmokeTrail follows the **wind**: it traces every air monitor's air backwards in time, credits the fires that air actually passed over, and rolls the result up into a ledger of which states export smoke and which ones breathe it.

![stack](https://img.shields.io/badge/stack-Flask%20%7C%20Leaflet%20%7C%20NumPy%2FSciPy%20%7C%20Shapely-orange)

## What you see

- **Smoke exporters ledger.** Indian states and Pakistani provinces are ranked by how much of their fire smoke reached *other* regions' monitors, with where it went (for example "→ Delhi 41%" means 41% of Delhi's fire smoke came from that state).
- **Smoke flow arcs.** Animated arcs on the map, from source region to receiving region, sized by the amount of smoke.
- **Monitor deep-dive.** Click any monitor to draw the path its air took over the last 24–48 h. You see the credited fires sized by contribution, the source mix ("70% Punjab · 16% Haryana · 14% Punjab (Pakistan)"), and how much was imported versus local.
- **Live wind.** An animated wind field, the satellite fires, and PM2.5 monitors colored by India's NAQI bands.
- **Honest comparison.** The Method tab compares the wind-aware score with a distance-only baseline (fires within 150 km), so you can see whether following the wind explains PM2.5 better.

## How the attribution works

1. **Wind field.** Hourly wind from [Open-Meteo](https://open-meteo.com) (free, no key) on a 2° grid over South Asia. Transport wind is the mean of the 10 m and 925 hPa (~750 m) winds, a proxy for the boundary layer where smoke travels. Over high terrain only 10 m is used.
2. **Back-trajectories.** Starting at each monitor at the time of its reading, the model integrates backwards in 1 h midpoint (RK2) steps for 24 or 48 h. This is the same idea as NOAA's HYSPLIT, simplified to one level.
3. **Crediting fires.** Fire detections are clustered (about 10 km, 1 h) and matched to path points with a KD-tree:

   ```
   smoke(fire → monitor) = FRP
       × exp(−d² / 2σ²)          σ = (25 km + 1 km/h × hours back) / 2  (corridor widens with dispersion)
       × exp(−Δt² / 2·(6 h)²)    the fire was burning when the air passed
       × exp(−hours back / 36 h) older smoke is more diluted
   ```

   Each fire counts once per monitor, at its best-matching path point. Fires detected after the reading are ignored.
4. **Roll-up.** Fires and monitors are placed in a state, province or country (Natural Earth admin-1, bundled in `data/regions.geojson`). The ledger sums smoke delivered to monitors outside each source region.

> **Limits.** This is a simplified Lagrangian model: a single transport layer, a coarse wind grid, and no plume rise, chemistry or wet deposition. Satellites miss small or cloud-covered fires, and PM2.5 also comes from traffic, industry, dust and household burning. It shows where smoke most likely came from; it is not a regulatory source-apportionment study.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env    # then paste your keys
python app.py           # http://127.0.0.1:5000
```

No keys yet? `DEMO_MODE=1 python app.py` runs the whole UI on synthetic data, with a clear demo banner.

Free API keys:

- NASA FIRMS MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
- OpenAQ v3: https://explore.openaq.org/register
- Open-Meteo: no key needed

## API

| Endpoint | Description |
|---|---|
| `GET /api/attribution?days=1-5&hours=12-72` | `{summary, ledger, flows, stations}`: the wind-aware accountability analysis |
| `GET /api/station/<id>?days=&hours=` | One monitor's back-trajectory path and its credited fires |
| `GET /api/wind` | Transport-wind grid (u/v, m/s) for the current hour |
| `GET /api/regions` | State/province boundaries (GeoJSON) |
| `GET /api/fires?days=1-5` | FIRMS fire detections |
| `GET /api/aqi` | OpenAQ PM2.5 monitors (with station names) |
| `GET /api/analysis?days=&radius_km=` | Distance-only baseline (no wind) |
| `GET /api/health` | Liveness check and which keys are configured |

Upstream data and analysis results are cached in memory for `CACHE_TTL_SECONDS` (default 30 min). Concurrent requests share one computation, and failures are never cached. A full India run (about 1,400 monitors and 20k+ fires) takes about 1 s of CPU.

## Deploy (Render, free tier)

`render.yaml` is a ready-made Blueprint. On Render, choose **New → Blueprint**, pick this repo, and paste `FIRMS_MAP_KEY` and `OPENAQ_API_KEY`. Any host that runs a `Procfile` also works.

## Tests

```bash
pytest -q
```

The tests run offline with no keys. They cover the API validators, the wind parsing and interpolation conventions, region lookup, back-trajectory geometry (upwind, distance, leaving the grid), attribution correctness (an upwind fire is credited, a much bigger downwind fire is not, reversed wind credits nothing, future fires are ignored), the ledger and flows, the cache, and every route. GitHub Actions runs them on each push.

## Project layout

```
app.py              Flask routes, caching, demo mode
attribution.py      Back-trajectories, fire crediting, ledger, flows
wind.py             Open-Meteo wind grid + space/time interpolation
regions.py          Point → state/province lookup (Shapely)
fires.py            NASA FIRMS fetcher + validation
aqi.py              OpenAQ v3 fetcher (pages the global feed, keeps India)
correlate.py        Distance-only baseline, NAQI categories, Spearman
cache.py            Thread-safe TTL cache with per-key locking
demo_data.py        Synthetic scene for DEMO_MODE
data/regions.geojson  Natural Earth admin-1, simplified (public domain)
static/index.html   The map UI (Leaflet, vanilla JS)
test_*.py           Offline test suite
```

## Data notes

- NASA is retiring the `VIIRS_SNPP_NRT` fire feed in Nov 2026, so the default is now `VIIRS_NOAA20_NRT`. Override it with `FIRMS_SOURCE`.
- OpenAQ readings older than 2 days, readings outside India's bounding box, and negative values are dropped.
- NAQI PM2.5 bands (µg/m³): Good 0–30 · Satisfactory 31–60 · Moderate 61–90 · Poor 91–120 · Very Poor 121–250 · Severe 250+.

## Roadmap

- Multi-level / ensemble trajectories to show uncertainty
- District-level boundaries
- Historical season replay (FIRMS archive + OpenAQ history)
