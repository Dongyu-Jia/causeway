"""The single fee table (params/fees.json): every fund fee of the instrument layer comes from here (T14).

    fees.fee("TQQQ")                     # the current fee (annual, decimal)
    fees.fee("TQQQ", "2022-03-31")       # the fee in force on a date
    fees.fee_path("QLD", months)         # one fee per month (PeriodIndex or dates)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

FILE = Path(__file__).resolve().parent / "params" / "fees.json"


@lru_cache(maxsize=1)
def table() -> dict:
    return json.loads(FILE.read_text(encoding="utf-8"))


def _schedule(ticker: str) -> list[dict]:
    try:
        return table()["funds"][ticker]["schedule"]
    except KeyError:
        raise KeyError(f"no fee for {ticker!r} in params/fees.json") from None


def fee(ticker: str, date=None) -> float:
    """The annual net expense ratio of `ticker` (decimal) on `date` (default: the latest entry)."""
    sched = _schedule(ticker)
    if date is None:
        return float(sched[-1]["fee"])
    d = pd.Timestamp(date)
    out = float(sched[0]["fee"])
    for e in sched:
        if e["from"] is None or pd.Timestamp(e["from"]) <= d:
            out = float(e["fee"])
    return out


def fee_path(ticker: str, dates) -> np.ndarray:
    """The fee in force at each date (a PeriodIndex counts at its periods' starts)."""
    if isinstance(dates, pd.PeriodIndex):
        dates = dates.to_timestamp()
    return np.array([fee(ticker, d) for d in pd.DatetimeIndex(dates)])


def source(ticker: str) -> str:
    return str(_schedule(ticker)[-1].get("source", ""))
