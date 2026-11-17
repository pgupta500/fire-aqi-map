"""
Offline tests for the OpenAQ cleaner/validator and the paging loop.
No API key or network needed: requests.get is replaced with a fake.
Run: pytest
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import aqi
from aqi import AQIDataError, _clean, _validate


def iso(delta_hours=0):
    t = datetime.now(timezone.utc) - timedelta(hours=delta_hours)
    return t.isoformat().replace("+00:00", "Z")


def reading(lat=28.6, lon=77.2, value=150.0, dt=None):
    return {
        "coordinates": {"latitude": lat, "longitude": lon},
        "value": value,
        "datetime": {"utc": dt or iso(1)},
    }


def test_clean_keeps_valid_india_reading():
    rows = _clean([reading()])
    assert len(rows) == 1 and rows[0]["pm25"] == 150.0


@pytest.mark.parametrize(
    "bad",
    [
        reading(value=-5),                       # negative sensor glitch
        reading(value=None),                     # missing value
        reading(value="NaN"),                    # non-numeric
        reading(lat=51.5, lon=-0.1),             # London, outside India
        reading(dt=iso(24 * 5)),                 # stale
        reading(dt="not-a-date"),                # unparsable timestamp
        {"value": 10, "datetime": {"utc": iso()}},  # no coordinates
    ],
)
def test_clean_drops_bad_rows(bad):
    assert _clean([bad]) == []


def test_validate_rejects_empty():
    with pytest.raises(AQIDataError):
        _validate(pd.DataFrame())


class FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_pages_until_404(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "test")
    pages = {1: [reading()], 2: [reading(lat=19.0, lon=72.8)]}

    def fake_get(url, headers, params, timeout):
        p = params["page"]
        return FakeResp(200, {"results": pages[p]}) if p in pages else FakeResp(404)

    monkeypatch.setattr(aqi.requests, "get", fake_get)
    df = aqi.get_air_quality()
    assert len(df) == 2


def test_http_error_raises(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "test")
    monkeypatch.setattr(aqi.requests, "get", lambda *a, **k: FakeResp(500, text="boom"))
    with pytest.raises(AQIDataError, match="HTTP 500"):
        aqi.get_air_quality()


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    with pytest.raises(AQIDataError, match="OPENAQ_API_KEY"):
        aqi.get_air_quality()


def test_duplicate_sensors_collapse_to_one_station(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "test")
    a = dict(reading(dt=iso(3)), locationsId=7)
    b = dict(reading(value=200.0, dt=iso(1)), locationsId=7)   # newer
    c = dict(reading(lat=19.0, lon=72.8), locationsId=8)

    def fake_get(url, headers, params, timeout):
        if url.endswith("/latest") and params["page"] == 1:
            return FakeResp(200, {"results": [a, b, c]})
        return FakeResp(404)

    monkeypatch.setattr(aqi.requests, "get", fake_get)
    df = aqi.get_air_quality()
    assert len(df) == 2
    assert df.loc[df["location_id"] == 7, "pm25"].item() == 200.0
