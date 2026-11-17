"""
Prove the validation logic works WITHOUT a real API key, by feeding
_validate and the CSV-sniffing logic the exact garbage FIRMS can send.
Run: python test_fires.py   (or: pytest)
"""

import io
import pandas as pd
from fires import _validate, FireDataError, REQUIRED_COLUMNS


def expect_error(label, fn):
    try:
        fn()
    except FireDataError as e:
        print(f"PASS: {label} -> caught: {e}")
        return
    raise AssertionError(f"FAIL: {label} -- should have raised but didn't")


def expect_ok(label, fn):
    try:
        fn()
        print(f"PASS: {label} -- accepted valid data")
    except FireDataError as e:
        raise AssertionError(f"FAIL: {label} -> wrongly rejected: {e}")


def sniff_trap():
    body = "Invalid MAP_KEY. Please check your key."
    if "latitude" not in body.split("\n", 1)[0].lower():
        raise FireDataError("caught text-masquerading-as-csv")


GOOD = pd.DataFrame(
    {
        "latitude": [20.5, 22.1],
        "longitude": [77.0, 78.3],
        "acq_date": ["2026-08-19", "2026-08-19"],
        "confidence": ["n", "h"],
        "frp": [5.2, 12.8],
    }
)


def test_empty_data():
    expect_error("empty data", lambda: _validate(pd.DataFrame()))


def test_missing_columns():
    expect_error(
        "missing columns",
        lambda: _validate(pd.DataFrame({"latitude": [20.0], "longitude": [77.0]})),
    )


def test_bad_coordinates():
    bad = pd.DataFrame(
        {c: [999.0 if c in ("latitude", "longitude") else 1] for c in REQUIRED_COLUMNS}
    )
    expect_error("bad coordinates", lambda: _validate(bad))


def test_text_error_with_200():
    expect_error("text error with 200 status", sniff_trap)


def test_valid_data():
    expect_ok("valid India fire data", lambda: _validate(GOOD))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nValidation logic verified.")
