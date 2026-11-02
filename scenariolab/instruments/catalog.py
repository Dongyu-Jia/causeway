"""The ticker catalog for GET /api/instruments (INSTRUMENTS_DESIGN §6.4, §7): Chinese names, kinds, recipe
cards (fees filled from the one fee table), parameters with defaults and ranges, validation tables, sources and
limits, chips, and from the built instrument_meta.json the start state, data vintages and diagnostics.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .. import config, jsonio
from . import TICKERS, fees
from . import params as prm


def entries(meta: dict | None = None) -> dict:
    """The catalog with every {fee} placeholder filled and, given the build's instrument_meta.json (default: the
    one in results/ if present), each ticker's start state, flags and the per-scenario headline statistics."""
    c = copy.deepcopy(prm.catalog())
    if meta is None:
        p = Path(config.RESULTS_DIR) / "instrument_meta.json"
        meta = jsonio.load(p) if p.exists() else {}
    for t, e in c["tickers"].items():
        try:
            f = fees.fee(t)
        except KeyError:
            f = None
        if f is not None:
            e["card"] = e["card"].replace("{fee}", f"{f * 100:.2f}%")
            e["fee"] = f
            e["fee_source"] = fees.source(t)
        if e.get("kind") == "estimated":
            for k, v in c["estimated_common"].items():
                e.setdefault(k, v)
        e["stored"] = t in TICKERS
        for v in e.get("validation") or []:
            _fill_validation(v, (meta or {}).get("validation"))
        if meta:
            e["start"] = (meta.get("start") or {}).get(t)
            e["flags"] = (meta.get("flags") or {}).get(t, [])
            e["stats"] = {sid: s.get(t) for sid, s in (meta.get("stats") or {}).items() if t in s}
    c["built"] = {k: meta.get(k) for k in ("built_at", "anchor_month", "start_source", "vintages", "params_version",
                                           "defaults", "alpha_variants", "file_bytes")} if meta else None
    c["diagnostics"] = meta.get("diagnostics") if meta else None
    return c


def _fill_validation(entry: dict, validation: dict | None) -> None:
    """Replace a validation entry's numbers with the build's own replay (instrument_meta.json["validation"]) at
    entry["src"] (a dotted path such as "TQQQ.windows.oos"), so the card never shows numbers the build did not
    reproduce; without a build the catalog's own numbers stay."""
    src = entry.get("src")
    if not src or not isinstance(validation, dict):
        return
    v = validation
    for k in str(src).split("."):
        v = v.get(k) if isinstance(v, dict) else None
    if not isinstance(v, dict):
        return
    for mine, theirs in (("actual", "ann_actual"), ("model", "ann_model"), ("rmse_monthly", "rmse_monthly"),
                         ("te_ann", "te_ann"), ("cover80", "cover80"), ("months", "months"), ("first", "first"),
                         ("last", "last")):
        if v.get(theirs) is not None and (mine in entry or mine in ("months", "first", "last")):
            entry[mine] = v[theirs]
    if isinstance(v.get("t3"), dict):
        entry["t3_pass"] = bool(v["t3"].get("pass"))
    entry["from_build"] = True


def name(ticker: str) -> str:
    return prm.catalog()["tickers"][ticker]["name"]
