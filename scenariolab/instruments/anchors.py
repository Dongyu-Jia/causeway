"""The start state of every recipe at the core's anchor month (INSTRUMENTS_DESIGN §2.6, D1).

The start is always the core's anchor (the engine's paths start at the 2026-09-30 close), never today. compute()
builds it from data/raw/instruments with the complete-month rule; current() uses that when the data are there and
falls back to the packaged params/anchors.json for the same month otherwise (CI, a fresh clone).

  vol        NDX and US-market log RV of the anchor month, the month before and the quarter to date, the log
             drawdowns, VXN at the close (decimal)
  rates      10 / 20 / 30-year CMT yields and the effective fed funds at the close (decimal)
  prices     USD price per share at the anchor: QLD / TQQQ ProShares NAV; QQQ, QQQI, TLT, GLD and the estimated
             tickers from the latest 13F quarter-end price rolled to the anchor (13F's last quarter is 2026-06-30
             until the 2026Q3 data set appears in mid-November, then the actual quarter-end price is used)
  estimated  the payout state of the estimated tickers: last quarterly distribution, trailing 12 months, NAV
             (BDCs, BME); `estimated_start` marks a price rolled forward (chip start_est)

The roll of an estimated ticker over a quarter (2026Q3) uses the recipe's systematic return on the ACTUAL core
quarter (the residual is unknown, so +-1 quarter of residual) less the quarter's distribution; ARCC has its own
XBRL market price for 2026-09-30 and MO its declared 2026Q3 dividend (1.11).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import fees
from . import params as prm


def _q(month: str) -> pd.Period:
    return pd.Period(month, "M").asfreq("Q")


def vol_state(month: str, panel=None) -> dict:
    ndx = D.ndx_daily()
    ok = D.complete_months(ndx)
    m = pd.Period(month, "M")
    if m not in ok:
        raise D.DataMissing(f"{month} 的纳指日度数据不完整（完整月份规则）")
    x = np.log(ndx).diff().dropna()
    km = x.index.to_period("M")
    rv = (x ** 2).groupby(km).sum()
    q = _q(month)
    xq = x[(x.index.to_period("Q") == q) & (km <= m)]
    me = ndx.groupby(ndx.index.to_period("M")).last()
    me = me[me.index <= m]
    mk = D.us_market_daily()
    mk_ok = D.complete_months(mk)
    if m not in mk_ok:
        raise D.DataMissing(f"{month} 的美股日度数据（Ken French）不完整")
    rvm = (mk ** 2).groupby(mk.index.to_period("M")).sum()
    mkq = mk[(mk.index.to_period("Q") == q) & (mk.index.to_period("M") <= m)]
    from ..data.panel import load_panel
    P = load_panel() if panel is None else panel
    xp = np.log1p(P["p_us_equity"].astype(float))
    xp.index = pd.DatetimeIndex(xp.index).to_period("M")
    lp = xp.loc[pd.Period("1986-01", "M"):m].cumsum()
    vxn = D.vxn_daily()
    vx = vxn[vxn.index.to_period("M") == m]
    return {"lrv_ndx": float(np.log(rv[m])), "lrv_ndx_prev": float(np.log(rv[m - 1])),
            "lrv_ndx_q": float(np.log((xq ** 2).sum())),
            "lrv_mkt": float(np.log(rvm[m])), "lrv_mkt_prev": float(np.log(rvm[m - 1])),
            "lrv_mkt_q": float(np.log((mkq ** 2).sum())),
            "dd_ndx": float(np.log(me.iloc[-1] / me.cummax().iloc[-1])), "dd_mkt": float((lp - lp.cummax()).iloc[-1]),
            "vxn": float(vx.iloc[-1]) / 100.0, "ndx_close": float(me.iloc[-1]),
            "rv_ndx_ann_vol": float(np.sqrt(rv[m] * 12)), "rv_mkt_ann_vol": float(np.sqrt(rvm[m] * 12))}


def rates(month: str) -> dict:
    d = D.last_trading_day(month)
    out = {}
    for k, sid in (("y10", "DGS10"), ("y20", "DGS20"), ("y30", "DGS30"), ("dff", "DFF")):
        s = D.fred_daily(sid)
        out[k] = float(s.loc[:d].iloc[-1]) / 100.0
    return out


def _systematic_q(t: str, row: pd.Series) -> float:
    """The recipe's log return for one actual core quarter without the residual (systematic + shrunk alpha)."""
    r = prm.estimated()["tickers"][t]
    x = r["cash_passthrough"] * row["lr_cash"] + r["alpha_q"]
    for c, b in r["loadings"].items():
        if c.startswith("x_"):
            x += b * (row["lr_" + c[2:]] - row["lr_cash"])
        elif c == "d_gz":
            g = row.get("d_gz", np.nan)
            if not np.isfinite(g):           # no GZ yet for the quarter: the bridge (the market and the spread)
                br = prm.bridges()["d_gz"]["models"][prm.bridges()["d_gz"]["default"]]
                g = br["coef"][0] + br["coef"][1] * row["d_baa10"] + br["coef"][2] * row["lr_us_equity"]
            x += b * g
    return float(x)


def estimated_state(month: str, panel=None) -> dict:
    """{ticker: {price, D_last_q, D_ttm, yield_ttm, asof, estimated_start, basis[, nav, premium_log]}}."""
    m = pd.Period(month, "M")
    qend = m.asfreq("Q").to_timestamp(how="end").normalize()
    cq = D.core_quarterly(panel)
    out = {}
    bdc = D.derived("sec_bdc_periodic.csv")
    divs = D.derived("sec_dividends_quarterly.csv")
    for t in prm.estimated()["tickers"]:
        d = D.instr_quarterly(t)
        last = d["P"].last_valid_index()
        P0, Dq = float(d.loc[last, "P"]), float(d.loc[last, "D"]) if np.isfinite(d.loc[last, "D"]) else 0.0
        last4 = [float(v) if np.isfinite(v) else 0.0 for v in d["D"].loc[:last].tail(4)]
        st = {"price": P0, "D_last_q": Dq, "D_ttm": float(sum(last4)), "yield_ttm": float(sum(last4)) / P0,
              "asof": str(last.date()), "estimated_start": False, "basis": "13F quarter-end price (median over filers)"}
        if np.isfinite(d.loc[last].get("nav", np.nan)):
            st["nav"], st["nav_asof"] = float(d.loc[last, "nav"]), str(last.date())
        cur = last
        while cur < qend:                                     # roll quarter by quarter to the anchor
            nxt = (pd.Period(cur, "Q") + 1).to_timestamp(how="end").normalize()
            if nxt not in cq.index:
                break
            Dn = st["D_last_q"]
            if t == "MO":
                qlab = str(pd.Period(nxt, "Q"))
                hit = divs[(divs.ticker == "MO") & (divs.quarter == qlab)]
                if len(hit):
                    Dn = float(hit["per_share"].iloc[-1])
            x = _systematic_q(t, cq.loc[nxt])
            st["price"] = float(st["price"] * max(np.exp(x) - Dn / st["price"], 0.05))
            last4 = (last4 + [Dn])[-4:]
            st["D_ttm"], st["D_last_q"] = float(sum(last4)), Dn
            st["asof"], st["estimated_start"] = str(nxt.date()), True
            st["basis"] = (f"13F {last.date()} price rolled to {nxt.date()} by the recipe's systematic return on "
                           "the actual core quarter (residual unknown)")
            if t == "ARCC":
                hit = bdc[(bdc.ticker == "ARCC") & (bdc["freq"] == "Q")
                          & (pd.to_datetime(bdc.period_end_month).dt.to_period("Q") == pd.Period(nxt, "Q"))]
                if len(hit) and np.isfinite(hit["price_ps"].iloc[-1]):
                    st["price"], st["estimated_start"] = float(hit["price_ps"].iloc[-1]), False
                    st["basis"] = f"SEC XBRL market price {nxt.date()}"
                    if np.isfinite(hit["nav_ps"].iloc[-1]):
                        st["nav"], st["nav_asof"] = float(hit["nav_ps"].iloc[-1]), str(nxt.date())
            cur = nxt
        st["yield_ttm"] = st["D_ttm"] / st["price"]
        if "nav" in st:
            st["premium_log"] = float(np.log(st["price"] / st["nav"]))
        out[t] = st
    return out


def structural_prices(month: str, vs: dict, rt: dict) -> dict:
    """USD start prices of the structural tickers (display only: the paths are returns)."""
    d = D.last_trading_day(month)
    out = {}
    for t in ("QLD", "TQQQ"):
        nav = D.proshares_nav(t)
        out[t] = {"price": float(nav.loc[:d].iloc[-1]), "asof": str(nav.loc[:d].index[-1].date()),
                  "basis": "ProShares NAV"}
    p13 = D.prices_13f_table()
    ndx = D.ndx_daily()
    S = prm.structural()
    for t, scale in (("QQQ", "ndx"), ("QQQI", "ndx_cc"), ("TLT", "tlt"), ("GLD", "gold")):
        if p13 is None or t not in p13.columns:
            continue
        s = p13[t].dropna()
        q0 = s.index[-1]
        px = float(s.iloc[-1])
        if q0 < d:
            if scale in ("ndx", "ndx_cc"):
                r = float(ndx.loc[:d].iloc[-1] / ndx.loc[:q0].iloc[-1])
                if scale == "ndx_cc":           # QQQI: its beta to the index, less the NAV-rate distributions
                    months = (pd.Period(d, "M") - pd.Period(q0, "M")).n
                    r = ((1 + S["qqqi"]["beta_ndx_model"] * (r - 1))
                         * (1 - S["qqqi"]["distribution_rate_annual"] / 12) ** months)
                px *= r
            elif scale == "tlt":
                w = S["tlt"]["w20"]
                y20, y30 = D.fred_daily("DGS20"), D.fred_daily("DGS30")
                y0 = (w * y20.loc[:q0].iloc[-1] + (1 - w) * y30.loc[:q0].iloc[-1]) / 100
                y1 = (w * y20.loc[:d].iloc[-1] + (1 - w) * y30.loc[:d].iloc[-1]) / 100
                px *= float(1 - np.polyval(S["tlt"]["D_of_y"], y0) * (y1 - y0))
            elif scale == "gold":
                from ..data.panel import load_panel
                g = load_panel()["lvl_gold"].copy()
                g.index = pd.DatetimeIndex(g.index).to_period("M")
                px *= float(g.loc[pd.Period(d, "M")] / g.loc[pd.Period(q0, "M")]) * (1 - fees.fee("GLD") / 4)
        out[t] = {"price": float(px), "asof": str(d.date()) if q0 < d else str(q0.date()),
                  "from": str(q0.date()), "method": scale if q0 < d else None,
                  "basis": f"13F {q0.date()} rolled to {d.date()} by {ROLL_METHODS[scale]}"
                  if q0 < d else "13F quarter-end price"}
    return out


# how a structural ticker's 13F quarter-end price is carried to the anchor (structural_prices): model rolls, except QQQ
ROLL_METHODS = {"ndx": "the NDX price return", "ndx_cc": "the QQQI model (beta x the NDX return, less the NAV-rate "
                "distributions)", "tlt": "the duration model D(y) dy on the TLT yield (0.513 y20 + 0.487 y30)",
                "gold": "the ratio of the London monthly-average gold price, less the fee"}


def compute(month: str, panel=None) -> dict:
    """The full start state at the end of `month` from data/raw/instruments (raises DataMissing)."""
    vs = vol_state(month, panel)
    rt = rates(month)
    est = estimated_state(month, panel)
    prices = structural_prices(month, vs, rt)
    prices.update({t: {"price": s["price"], "asof": s["asof"], "basis": s["basis"]} for t, s in est.items()})
    return {"month": month, "date": str(D.last_trading_day(month).date()), "vol": vs, "rates": rt,
            "prices": prices, "estimated": est, "source": "data/raw/instruments (anchors.compute)",
            "vintages": D.vintages()}


def current(month: str, panel=None, prefer_data: bool = True) -> dict:
    """The anchor block for `month`: computed from data when possible, else the packaged one for that month."""
    if prefer_data:
        try:
            return compute(month, panel)
        except (D.DataMissing, KeyError, IndexError, ValueError, FileNotFoundError):
            pass
    pk = prm.packaged_anchors()
    if pk.get("month") == month:
        return {**pk, "source": "params/anchors.json (packaged)"}
    raise D.DataMissing(f"没有 {month} 的标的起点状态：请导入数据（instruments.import_raw）或更新 params/anchors.json")
