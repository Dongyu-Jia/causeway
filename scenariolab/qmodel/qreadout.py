"""Extra readouts of a quarterly run for the scenario detail and the lab (SIM_DESIGN §4.4, §6.3, §10.3; §14 P8).

Added to /api/q/run results by api.execute, from the Rollout through rollout's public API:

  valuation_statements  per valuation statement of the run: its quarters, mode, size and, over those quarters, the
                        median summed CAPE-multiple leg, fundamentals leg and price (log), so the page can print
                        「估值假设：这一条相当于直接假设美股价格在这 N 个季度里 ±X%（其中基本面腿 +Y%）」
  y_fund                the fundamentals leg's quarterly fan (p10 / p50 / p90)
  attribution_by_h      rollout.attribution cumulated over the first 4 and 40 quarters (the 40-quarter focus, U-16)
                        when the run is longer: stated / completed / drift / valuation / residual medians per asset
  regimes               the regime classifier on the quarterly paths (quarter-end CPI y/y, unemployment and the IP
                        level, with the history before the origin first; factor_checks' quarterly params): per year
                        each label's share of path-quarters and the share of paths with a contraction in that year

No LLM, no download; nothing here changes a simulated number.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

QS = (10.0, 50.0, 90.0)
HORIZONS = (4, 40)
LABEL_IDS = ("expansion", "inflation", "deflation", "recession", "depression", "stagflation")


def _f(x, d: int = 6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _fan(a) -> dict:
    a = np.asarray(a, float)
    return {f"p{q:g}": [_f(v) for v in np.nanpercentile(a, q, axis=0)] for q in QS}


def valuation_statements(roll) -> list:
    """Each valuation statement's quarters with the median summed legs over them (log)."""
    out = []
    ax = roll.ax
    j = list(ax.assets).index("us_equity")
    for st in roll.valuation or ():
        qs = [int(q) for q in getattr(st, "quarters", ()) if 1 <= int(q) <= roll.T]
        if not qs:
            continue
        idx = np.array(qs) - 1
        mult = np.asarray(ax.y_mult, float)[:, idx].sum(axis=1)
        fund = np.asarray(ax.y_fund, float)[:, idx].sum(axis=1)
        price = np.asarray(ax.price, float)[:, idx, j].sum(axis=1)
        out.append({"quarters": qs, "n": len(qs), "mode": getattr(st, "mode", None),
                    "direction": getattr(st, "direction", None), "size": getattr(st, "size", None),
                    "mult_p50": _f(np.nanmedian(mult)), "fund_p50": _f(np.nanmedian(fund)),
                    "price_p50": _f(np.nanmedian(price))})
    return out


def attribution_by_horizon(roll, horizons=HORIZONS) -> dict:
    from . import rollout as R
    out = {}
    for k in horizons:
        if k < roll.H:
            out[str(k)] = R.attribution(dataclasses.replace(roll, H=k))["assets"]
    return out


def regime_readout(stack, roll) -> dict | None:
    """Per simulated year: each regime label's share of path-quarters and the share of paths with a contraction."""
    import pandas as pd

    from ..engine import regimes as RG
    from . import factor_checks as FC
    lv = roll.fx.levels
    if "cpi_yoy" not in lv or "unrate" not in lv or "d_ip" not in roll.fx.x:
        return None
    q0 = pd.Period(roll.labels[0], "Q")
    Mh = FC.quarter_end(stack.P).loc["1960Q2":q0 - 1]
    k = len(Mh)
    n = roll.n
    cpi = np.asarray(lv["cpi_yoy"], float)
    u = np.asarray(lv["unrate"], float)
    lip = math.log(float(Mh["lvl_ip"].iloc[-1])) + np.cumsum(np.asarray(roll.fx.x["d_ip"], float) / 100.0, axis=1)
    full = RG.classify(np.concatenate([np.broadcast_to(Mh["s_cpi_yoy"].to_numpy(float), (n, k)), cpi], 1),
                       np.concatenate([np.broadcast_to(Mh["lvl_unrate"].to_numpy(float), (n, k)), u], 1),
                       np.concatenate([np.broadcast_to(Mh["lvl_ip"].to_numpy(float), (n, k)), np.exp(lip)], 1),
                       params=FC.QP)
    lab = np.asarray(full["label"])[:, k:]
    con = np.asarray(full["contraction"], bool)[:, k:]
    T = lab.shape[1]
    years = []
    for y in range(T // 4):
        sl = slice(4 * y, 4 * y + 4)
        cell = lab[:, sl]
        known = cell >= 0
        tot = max(int(known.sum()), 1)
        mix = {LABEL_IDS[c]: _f((cell == c).sum() / tot, 4) for c in range(len(LABEL_IDS))}
        years.append({"year": y + 1, "mix": mix, "paths_with_contraction": _f(con[:, sl].any(axis=1).mean(), 4),
                      "quarters": [str(roll.labels[4 * y]), str(roll.labels[min(4 * y + 3, T - 1)])]})
    return {"labels": list(LABEL_IDS), "years": years,
            "text": "季末的 CPI 同比、失业率和工业产出水平按宏观体制的同一套规则贴标签（季度口径：失业率不再取 3 个月均值）"}


def extras(stack, roll) -> dict:
    out = {"valuation_statements": valuation_statements(roll), "y_fund": _fan(roll.ax.y_fund),
           "attribution_by_h": attribution_by_horizon(roll)}
    try:
        out["regimes"] = regime_readout(stack, roll)
    except Exception as err:  # noqa: BLE001 - a readout only
        out["regimes"] = {"error": f"{type(err).__name__}: {err}"}
    return out
