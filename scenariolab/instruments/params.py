"""The instrument layer's frozen parameters (params/*.json, committed; refit only by fit.py, cut at train_end).

  structural.json  NDX volatility state, the IV model, the ruin tail, leveraged-ETF financing spreads, QQQI's
                   positions and distribution rate, TLT's yield mix, duration / convexity and long-spread models
  vol_joint.json   the US market's volatility equation and the shock correlation (monthly, quarterly)
  bridges.json     d_gz from the Baa - 10y change and the market (defect 2) and the history for diagnostics
  estimated.json   the ten estimated recipes: loadings, alpha (shrunk / raw / prior sd), RAW residuals by quarter,
                   the market vol at each residual date, payout models (defects 1 and 4 are applied at draw time)
  anchors.json     the start state at the core's anchor month (a fallback: anchors.current() rebuilds it from data)
  fees.json        the one fee table (fees.py)
  catalog.json     Chinese names, recipe cards, validation tables and chips for the UI
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

DIR = Path(__file__).resolve().parent / "params"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((DIR / f"{name}.json").read_text(encoding="utf-8"))


def load(name: str) -> dict:
    """A deep copy of params/<name>.json (callers may change it freely)."""
    return copy.deepcopy(_load(name))


def structural() -> dict:
    return _load("structural")


def vol_joint() -> dict:
    return _load("vol_joint")


def bridges() -> dict:
    return _load("bridges")


def estimated() -> dict:
    return _load("estimated")


def catalog() -> dict:
    return _load("catalog")


def packaged_anchors() -> dict:
    return _load("anchors")


def version() -> str:
    """One string naming every parameter file's version (instrument_meta.json)."""
    return "; ".join(f"{n}: {_load(n).get('version', '?')}" for n in
                     ("structural", "vol_joint", "bridges", "estimated", "fees", "anchors"))


def clear_cache() -> None:
    _load.cache_clear()
