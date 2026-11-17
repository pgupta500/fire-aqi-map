# fire-aqi-map

Fire and air quality accountability map for India.

Every autumn, crop-residue burning across Punjab and Haryana lines up with some of the worst air in the world downwind. This app puts the two datasets on one map and measures how closely they line up:

- **Fires**: NASA FIRMS VIIRS satellite detections from the last 1 to 5 days, colored by fire radiative power (FRP)
- **Air quality**: live PM2.5 readings from 1,300+ OpenAQ stations, colored by India's National AQI (NAQI) categories
- **Accountability analysis**: for every station, the app counts the fires within a chosen radius, adds up their intensity, and computes a distance-weighted *fire influence* score. It then ranks **hotspots**, meaning stations where the air is Poor or worse and there are at least 5 fires nearby.

![stack](https://img.shields.io/badge/stack-Flask%20%7C%20Leaflet%20%7C%20pandas-orange)

## How the analysis works

For each station *s* and each fire *f* within radius *R* (default 150 km):

```
fire_influence(s) = Σ FRP(f) × (1 − distance(s, f) / R)
```

Closer, hotter fires count more, and a fire right at the edge of the radius counts for almost nothing. The summary panel reports:

| Metric | Meaning |
|---|---|
| Stations with fires nearby | Stations with at least one detection inside *R* |
| Avg PM2.5 near fires vs. no fires | Mean PM2.5 of the two groups |
| Spearman ρ | Rank correlation between fire influence and PM2.5 across all stations |
| Hotspots | PM2.5 ≥ 90 µg/m³ (NAQI "Poor" or worse) **and** ≥ 5 fires within *R* |

> **Caveat:** this is spatial association, not causation. Wind direction, terrain, and non-fire sources (traffic, industry, dust) are not modeled. The map shows where to look; it does not assign blame on its own.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env    # then paste your keys
python app.py           # http://127.0.0.1:5000
```

Free API keys:

- NASA FIRMS MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/area/
- OpenAQ v3: https://explore.openaq.org/register

## API

| Endpoint | Description |
|---|---|
| `GET /api/fires?days=1-5` | Fire detections: latitude, longitude, acq_date, confidence, frp |
| `GET /api/aqi` | PM2.5 stations: latitude, longitude, pm25, datetime_utc |
| `GET /api/analysis?days=1-5&radius_km=10-500` | `{summary, stations, hotspots}`, with each station enriched with `fires_nearby`, `frp_nearby`, `fire_influence`, `nearest_fire_km`, `category` |
| `GET /api/health` | Liveness check and which keys are configured (never calls upstream) |

Upstream responses are cached in memory for `CACHE_TTL_SECONDS` (default 30 min), so page loads are fast and API quotas stay low. Failed upstream calls are never cached.

## Deploy (Render, free tier)

1. Push to GitHub.
2. On [render.com](https://render.com): **New → Blueprint**, then pick this repo. `render.yaml` configures everything.
3. Enter `FIRMS_MAP_KEY` and `OPENAQ_API_KEY` when prompted.

Any other host that runs a `Procfile` works too (Railway, Heroku, Fly):
`gunicorn app:app --bind 0.0.0.0:$PORT`.

## Tests

```bash
pytest -q
```

The tests run fully offline with no API keys. They cover the FIRMS/OpenAQ validators (including FIRMS's "plain-text error with HTTP 200" trap), the paging loop, the distance math, NAQI categories, the exposure/hotspot analysis, the cache, and every API route. GitHub Actions runs them on each push.

## Project layout

```
app.py            Flask routes, caching, parameter clamping
fires.py          NASA FIRMS fetcher + validation
aqi.py            OpenAQ v3 fetcher (pages the global feed, keeps India)
correlate.py      Fire -> PM2.5 exposure, Spearman correlation, hotspots
cache.py          Thread-safe in-memory TTL cache
static/index.html Leaflet map UI
test_*.py         Offline test suite
```

## Data notes

- `VIIRS_SNPP_NRT` is being retired by NASA in Nov 2026, so the default feed is now `VIIRS_NOAA20_NRT`. Switch with `FIRMS_SOURCE` (e.g. `VIIRS_NOAA21_NRT`, `MODIS_NRT`) without changing code.
- OpenAQ readings older than 2 days, readings outside India's bounding box, and negative values are dropped.
- NAQI PM2.5 bands (µg/m³): Good 0–30 · Satisfactory 31–60 · Moderate 61–90 · Poor 91–120 · Very Poor 121–250 · Severe 250+.

## Roadmap

- Wind-aware attribution (upwind fires only, using Open-Meteo wind fields)
- District/state aggregation with boundary GeoJSON
- Historical time slider (FIRMS archive + OpenAQ history)
