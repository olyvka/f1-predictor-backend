"""
Тести чистих функцій app/fastf1_client.py, які не звертаються до мережі:
конвертація значень pandas/FastF1 у JSON-сумісний вигляд і мапа коротких
кодів сесій.
"""
import math

import numpy as np
import pandas as pd
import pytest

from app.fastf1_client import SESSION_NAME_TO_CODE, _clean, _clean_series, _clean_value


# ---------------------------------------------------------------------------
# _clean_value
# ---------------------------------------------------------------------------

def test_clean_value_nan_becomes_none():
    assert _clean_value(float("nan")) is None
    assert _clean_value(np.nan) is None
    assert _clean_value(pd.NaT) is None
    assert _clean_value(pd.NA) is None


def test_clean_value_timedelta_becomes_seconds():
    td = pd.Timedelta("0 days 00:01:27.452000")
    assert _clean_value(td) == pytest.approx(87.452)


def test_clean_value_timestamp_becomes_iso_string():
    ts = pd.Timestamp("2024-03-02T15:00:00Z")
    result = _clean_value(ts)
    assert isinstance(result, str)
    assert result.startswith("2024-03-02T15:00:00")


@pytest.mark.parametrize("value", [1, 1.5, "VER", True, 0])
def test_clean_value_passthrough_for_plain_values(value):
    assert _clean_value(value) == value


# ---------------------------------------------------------------------------
# _clean (DataFrame -> list[dict])
# ---------------------------------------------------------------------------

def test_clean_dataframe_replaces_nan_and_converts_timedelta():
    df = pd.DataFrame(
        {
            "Abbreviation": ["VER", "HAM"],
            "Points": [25.0, float("nan")],
            "LapTime": [pd.Timedelta(seconds=90.1), pd.NaT],
        }
    )

    records = _clean(df)

    assert records == [
        {"Abbreviation": "VER", "Points": 25.0, "LapTime": pytest.approx(90.1)},
        {"Abbreviation": "HAM", "Points": None, "LapTime": None},
    ]
    # результат обов'язково має серіалізуватись у JSON (allow_nan=False у FastAPI)
    import json

    json.dumps(records, allow_nan=False)


def test_clean_dataframe_empty():
    df = pd.DataFrame(columns=["Abbreviation", "Points"])
    assert _clean(df) == []


# ---------------------------------------------------------------------------
# _clean_series (одна pd.Series -> dict)
# ---------------------------------------------------------------------------

def test_clean_series():
    series = pd.Series({"Abbreviation": "VER", "Points": float("nan"), "GridPosition": 1})
    assert _clean_series(series) == {"Abbreviation": "VER", "Points": None, "GridPosition": 1}


# ---------------------------------------------------------------------------
# SESSION_NAME_TO_CODE
# ---------------------------------------------------------------------------

def test_session_name_to_code_mapping():
    assert SESSION_NAME_TO_CODE["Race"] == "R"
    assert SESSION_NAME_TO_CODE["Qualifying"] == "Q"
    assert SESSION_NAME_TO_CODE["Sprint"] == "S"
    # і Sprint Qualifying, і Sprint Shootout (різні ери регламенту спринтів)
    # мають зводитись до одного короткого коду "SQ"
    assert SESSION_NAME_TO_CODE["Sprint Qualifying"] == "SQ"
    assert SESSION_NAME_TO_CODE["Sprint Shootout"] == "SQ"
    assert all(
        code in {"FP1", "FP2", "FP3", "Q", "S", "SQ", "R"}
        for code in SESSION_NAME_TO_CODE.values()
    )
