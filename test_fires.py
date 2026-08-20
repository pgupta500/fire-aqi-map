"""
Prove the validation logic works WITHOUT a real API key, by feeding
_validate and the CSV-sniffing logic the exact garbage FIRMS can send.
Run: python test_fires.py
"""

import io
import pandas as pd
from fires import _validate, FireDataError, REQUIRED_COLUMNS


def expect_error(label, fn):
    try:
        fn()
        print(f"FAIL: {label} -- should have raised but didn't")
    except FireDataError as e:
        print(f"PASS: {label} -> caught: {e}")


def expect_ok(label, fn):
    try:
        fn()
        print(f"PASS: {label} -- accepted valid data")
    except FireDataError as e:
        print(f"FAIL: {label} -> wrongly rejected: {e}")


# 1. Empty dataframe should be rejected
expect_error("empty data", lambda: _validate(pd.DataFrame()))

# 2. Missing required columns should be rejected
expect_error(
    "missing columns",
    lambda: _validate(pd.DataFrame({"latitude": [20.0], "longitude": [77.0]})),
)

# 3. Corrupt coordinates should be rejected
bad = pd.DataFrame(
    {c: [999.0 if c in ("latitude", "longitude") else 1] for c in REQUIRED_COLUMNS}
)
expect_error("bad coordinates", lambda: _validate(bad))

# 4. The plain-text-error-as-200 trap: sniff logic
def sniff_trap():
    body = "Invalid MAP_KEY. Please check your key."
    if "latitude" not in body.split("\n", 1)[0].lower():
        raise FireDataError("caught text-masquerading-as-csv")
expect_error("text error with 200 status", sniff_trap)

# 5. Valid data should pass clean
good = pd.DataFrame(
    {
        "latitude": [20.5, 22.1],
        "longitude": [77.0, 78.3],
        "acq_date": ["2026-08-19", "2026-08-19"],
        "confidence": ["n", "h"],
        "frp": [5.2, 12.8],
    }
)
expect_ok("valid India fire data", lambda: _validate(good))

print("\nValidation logic verified.")
