"""Structural recipes: QQQ, QLD, TQQQ, QQQI, TLT, GLD (and the GOLD alias), derived from the products' mechanics.

Nothing here is fitted on the ticker's own returns (QQQI's strikes and coverage are its disclosed N-PORT positions,
TLT's duration / convexity its holdings, the leveraged funds' spreads their disclosed swap terms, fees the table).

  QQQ     nasdaq asset - fee:                 total = r - fee / ppy, income = max(d - fee / ppy, 0)
  QLD     2x, TQQQ 3x, daily reset: log(1 + L(e^x - 1)) expanded to the fourth order in the day's log return x and
          summed over the period's days (the period's variance is the only modelled input):
              log(V_T / V_0) = L sum x - (L^2 - L)/2 RV + L(2L - 1)(L - 1)/6 RS3 + c4(L) RS4 + log(1 + carry)
              c4(L) = L/24 - 7L^2/24 + L^3/2 - L^4/4 (-1.08 for L = 2, -9.25 for L = 3)
              RS3 = E[sum x^3 | x, RV] = a1 RV^1.5 + a2 x RV;  RS4 = E[sum x^4 | RV] = k RV^2 / days (k fitted to 2014)
              carry = [L q - (L - 1)(b + s_eff) 365/360 - fee] / ppy,  b = fed funds, q = the NDX dividend yield
              (the swaps accrue (b + s) x days / 360: ACT/360)
          ruin: a day with the index <= -1/L wipes the fund out (GPD tail of EWMA-standardised NDX losses)
  QQQI    NDX + two written calls a month on ~63% of NAV (N-PORT): k = 1.019 / 1.034 x the month-start level,
              r = R - 1 + q T + sum_i c_i [p_i (1 + r_f T) - max(R - k_i, 0)] - fee T,
              p_i = BS(1, k_i, T, r_f, q, sigma_ATM), sigma_ATM = 0.855 x VXN (the volatility state)
          a quarterly core is split into three months by a Brownian bridge with month variance rho RV_Q / 3,
          rho = var(monthly log return) / mean(monthly RV) of the NDX, 1986-2014 (0.85: realized daily variance runs
          above the variance of monthly returns, so RV_Q / 3 made the bridged months about 18% too volatile);
          distributions 14.3% of NAV a year, paid monthly (or "premium_plus_dividend"; the total is the same)
  TLT     y_TLT = 0.513 y20 + 0.487 y30, the long spreads over the core's 10-year by error correction on the
          10y - fed funds slope; r = y/ppy - D(y) dy + C(y)/2 dy^2 - fee/ppy, D(y) = 20.96 - 95.2 y,
          C(y) = 514 - 3191 y (28 N-PORT reports 2019-2026); income = the coupon yield (0.857 x YTM, back to 1
          with a 7-year half-life); a quarterly core is stepped monthly on interpolated y10 / ff
  GLD     gold - 0.40%; GOLD = the core's gold asset itself (London monthly-AVERAGE price: chip gold_avg)

Each recipe returns an InstrumentPaths (layer.py) with additive log parts for the attribution chart.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genpareto, norm

from . import fees
from . import params as prm
from .core import CoreView
from .streams import stream
from .vol import VolState


def _paths(**kw):
    from .layer import InstrumentPaths
    return InstrumentPaths(**kw)


def level_path(start: float, total: np.ndarray, income: np.ndarray) -> np.ndarray:
    """USD price per share (ex-distribution): start x cumulative (1 + total - income); (n, T + 1)."""
    g = np.cumprod(np.maximum(1.0 + total - income, 0.0), axis=1)
    return float(start) * np.concatenate([np.ones((total.shape[0], 1)), g], axis=1)


def _log_parts(parts: dict, total: np.ndarray) -> dict:
    """Simple-return parts scaled to additive log parts: each times log(1 + total) / total."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(np.abs(total) > 1e-12, np.log1p(np.maximum(total, -0.999999)) / total, 1.0)
    return {k: v * s for k, v in parts.items()}


# ---------------------------------------------------------------- QQQ, GLD

def qqq(core: CoreView, vol: VolState, p: dict, start: float) -> object:
    f = float(p.get("fee", fees.fee("QQQ"))) / core.ppy
    r, d = core.asset_total("nasdaq"), core.asset_income("nasdaq")
    total = r - f
    income = np.maximum(d - f, 0.0)
    lr = np.log1p(total)
    parts = {"index": np.log1p(r - d), "dividend": lr - np.log1p(r - d) + f, "fee": np.full_like(r, -f)}
    return _paths(ticker="QQQ", freq=core.freq, total=total, income=income, level=level_path(start, total, income),
                  parts=parts, flags=("tax",))


def gld(core: CoreView, vol: VolState, p: dict, start: float) -> object:
    f = float(p.get("fee", fees.fee("GLD"))) / core.ppy
    g = core.asset_total("gold")
    total = g - f
    income = np.zeros_like(total)
    return _paths(ticker="GLD", freq=core.freq, total=total, income=income, level=level_path(start, total, income),
                  parts={"gold": np.log1p(g), "fee": np.log1p(total) - np.log1p(g)}, flags=("gold_avg", "tax"))


# ---------------------------------------------------------------- leveraged funds

def p_index_day_below(level_log: float, sigma_daily) -> np.ndarray:
    """P(one day's index log return <= level_log | that day's vol): GPD tail of EWMA-standardised NDX losses."""
    t = prm.structural()["tail"]["fit"]
    thr = -level_log / np.maximum(np.asarray(sigma_daily, dtype=float), 1e-6)
    p = t["p_u"] * genpareto.sf(np.maximum(thr - t["u"], 0.0), t["xi"], loc=0.0, scale=t["beta"])
    return np.where(thr <= t["u"], t["p_u"], p)


def ruin_probability(rv, L: float, freq: str = "M") -> np.ndarray:
    """P(at least one index day <= -1/L in the period), the daily vol of the period = sqrt(RV / days)."""
    days = prm.structural()["vol"][f"days_{freq}"]
    sd = np.sqrt(np.asarray(rv) / days)
    return 1.0 - (1.0 - p_index_day_below(np.log(1.0 - 1.0 / L), sd)) ** days


def c4(L: float) -> float:
    """The fourth-order coefficient of log(1 + L(e^x - 1)) in x."""
    return L / 24.0 - 7.0 * L * L / 24.0 + L ** 3 / 2.0 - L ** 4 / 4.0


def financing_accrual() -> float:
    """Annual accrual of the swaps' financing leg per unit of rate: ACT/360 accrues (b + s) x 365 / 360 a year."""
    basis = prm.structural()["letf"].get("financing_accrual", "ACT/360")
    return 365.0 / 360.0 if basis == "ACT/360" else 1.0


def letf_carry(L: float, q, b, s_eff, fee, ppy: int) -> np.ndarray:
    """Carry per period (decimal): [L q - (L - 1)(b + s_eff) x accrual - fee] / ppy, every rate annual decimal."""
    return (L * np.asarray(q) - (L - 1) * (np.asarray(b) + np.asarray(s_eff)) * financing_accrual()
            - np.asarray(fee)) / ppy


def letf_log_return(x, rv, rs3, L: float, carry, rs4=None) -> np.ndarray:
    """The daily-reset period formula, the fourth-order expansion of log(1 + L(e^x - 1)) summed over the days:
    L sum x - (L^2 - L)/2 RV + L(2L - 1)(L - 1)/6 RS3 + c4(L) RS4 + log(1 + carry) (rs4 None: third order)."""
    c2, c3 = (L * L - L) / 2.0, L * (2 * L - 1) * (L - 1) / 6.0
    out = L * np.asarray(x) - c2 * np.asarray(rv) + c3 * np.asarray(rs3) + np.log1p(carry)
    if rs4 is not None:
        out = out + c4(L) * np.asarray(rs4)
    return out


def expected_rs4(rv, freq: str) -> np.ndarray:
    """E[sum of the period's daily log returns^4 | RV] = k RV^2 / days (k: the 1986-2014 median ratio)."""
    V = prm.structural()["vol"]
    return float(V[f"rs4_{freq}"]["k"]) * np.asarray(rv) ** 2 / float(V[f"days_{freq}"])


def breakeven_total_return(L: float, sigma: float, financing: float, fee_l: float, fee_1: float) -> float:
    """The NDX arithmetic total return (mu + q) above which the L x fund grows faster than the 1x fund:
    mu + q - b - s > (L + 1) sigma^2 / 2 + (fee_L - fee_1) / (L - 1)  (financing = b + s)."""
    return financing + (L + 1) * sigma ** 2 / 2 + (fee_l - fee_1) / (L - 1)


def leveraged(core: CoreView, vol: VolState, p: dict, start: float, ticker: str) -> object:
    S = prm.structural()["letf"][ticker]
    L = float(S["L"])
    k = core.ppy
    x = core.x("nasdaq")
    q = core.div_yield("nasdaq")
    b = core.ff[:, :-1] / 100.0
    s_eff = p.get("s_eff", S["s_eff_default"])
    s_eff = float(S["s_eff"][s_eff]) if isinstance(s_eff, str) else float(s_eff)
    fee = float(p.get("fee", fees.fee(ticker)))
    rs3 = vol.rs3_ndx(x)
    rs4 = expected_rs4(vol.rv_ndx, core.freq)
    carry = letf_carry(L, q, b, s_eff, fee, k)
    logv = letf_log_return(x, vol.rv_ndx, rs3, L, carry, rs4)
    total = np.expm1(logv)
    alive = np.ones(total.shape, dtype=bool)
    lc = np.log1p(carry)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(carry) > 1e-15, lc / carry, 1.0)
    parts = {"index": L * x, "vol_drag": -(L * L - L) / 2.0 * vol.rv_ndx,
             "third_moment": L * (2 * L - 1) * (L - 1) / 6.0 * rs3, "fourth_moment": c4(L) * rs4,
             "dividend": w * L * q / k, "financing": -w * (L - 1) * (b + s_eff) * financing_accrual() / k,
             "fee": -w * fee / k}
    if p.get("ruin", True):
        hit = stream(core.set_id, core.key, "ruin", ticker).random(total.shape) < ruin_probability(vol.rv_ndx, L,
                                                                                                core.freq)
        dead = np.cumsum(hit, axis=1) > 0
        first = hit & (np.cumsum(hit, axis=1) == 1)
        total = np.where(first, -1.0, np.where(dead, 0.0, total))
        alive = ~dead
        for v in parts.values():
            v[dead] = np.nan
    income = np.zeros_like(total)
    flags = (("ruin_extrap",) if p.get("ruin", True) else ()) + ("rv_level", "tax") \
        + (("fixed_vol",) if vol.source == "fixed" else ())
    return _paths(ticker=ticker, freq=core.freq, total=total, income=income, level=level_path(start, total, income),
                  alive=alive if p.get("ruin", True) else None, parts=parts, flags=flags,
                  meta={"s_eff": s_eff, "fee": fee, "L": L, "ruin": bool(p.get("ruin", True))})


# ---------------------------------------------------------------- covered calls (QQQI)

def bs_call(S, K, T, r, q, sig):
    """Black-Scholes call (vectorised; S and K in the same units)."""
    S, K, T, r, q, sig = (np.asarray(v, dtype=float) for v in (S, K, T, r, q, sig))
    T = np.maximum(T, 1e-8)
    sig = np.maximum(sig, 1e-4)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def overlay_month(R, q, rf, sigma, strikes, fee, T: float = 1.0 / 12) -> tuple[np.ndarray, dict]:
    """One month of index + written calls: (simple return, its parts)."""
    index, div, fee_p = R - 1.0, q * T, -fee * T
    prem = np.zeros_like(index)
    cap = np.zeros_like(index)
    for k, c in strikes:
        p = bs_call(1.0, k, T, rf, q, sigma)
        prem = prem + c * p * (1 + rf * T)
        cap = cap - c * np.maximum(R - k, 0.0)
    tot = index + div + prem + cap + fee_p
    return tot, {"index": index, "premium": prem, "capped_upside": cap, "dividend": np.broadcast_to(div, tot.shape),
                 "fee": np.broadcast_to(fee_p, tot.shape)}


def bridge_var_ratio() -> float:
    """var(monthly NDX log return) / mean(monthly RV), 1986-2014 (structural.json vol.bridge_month_var_ratio)."""
    return float(prm.structural()["vol"]["bridge_month_var_ratio"]["ratio"])


def bridge_months(x_q: np.ndarray, rv_q: np.ndarray, z: np.ndarray, ratio: float | None = None) -> np.ndarray:
    """Split quarterly log returns into 3 months: N(0, ratio x RV_Q / 3) draws shifted so that they sum to the
    quarter (ratio default: bridge_var_ratio(), the monthly return variance per unit of realized variance).
    z: (n, Q, 3) standard normals. Returns (n, Q, 3) monthly log returns."""
    r = bridge_var_ratio() if ratio is None else float(ratio)
    w = z * np.sqrt(r * np.asarray(rv_q) / 3.0)[..., None]
    return w - w.mean(axis=2, keepdims=True) + np.asarray(x_q)[..., None] / 3.0


def qqqi(core: CoreView, vol: VolState, p: dict, start: float) -> object:
    S = prm.structural()["qqqi"]
    c = float(p.get("coverage", S["coverage_total"]))
    strikes = ((float(p.get("k1", S["k1"])), c / 2.0), (float(p.get("k2", S["k2"])), c / 2.0))
    fee = float(p.get("fee", fees.fee("QQQI")))
    rule = p.get("dist_rule", "nav_rate")
    rate = float(p.get("dist_rate", S["distribution_rate_annual"]))
    x = core.x("nasdaq")
    q = core.div_yield("nasdaq")
    rf = core.rf()
    sig = vol.sigma_atm
    if core.freq == "M":
        total, parts = overlay_month(np.exp(x), q, rf, sig, strikes, fee)
        prem_cash = parts["premium"] + parts["dividend"]
        income = np.full(total.shape, rate / 12.0) if rule == "nav_rate" else np.maximum(prem_cash, 0.0)
        parts = {k: np.array(v, dtype=float) for k, v in parts.items()}
    else:
        z = stream(core.set_id, core.key, "bridge", "QQQI").standard_normal(x.shape + (3,))
        xm = bridge_months(x, vol.rv_ndx, z)
        g = np.ones_like(x)
        cash = np.zeros_like(x)
        acc = {k: np.zeros_like(x) for k in ("index", "premium", "capped_upside", "dividend", "fee")}
        for j in range(3):
            tm, pm = overlay_month(np.exp(xm[..., j]), q, rf, sig, strikes, fee)
            cm = (rate / 12.0) if rule == "nav_rate" else np.maximum(pm["premium"] + pm["dividend"], 0.0)
            # cash paid in month j per unit of the quarter's start value (the NAV after the earlier months' payouts)
            cash = cash + cm * g
            for kk in acc:
                acc[kk] = acc[kk] + pm[kk] * g
            g = g * (1.0 + tm - cm)
        total = g + cash - 1.0
        income = cash
        parts = acc
    flags = ("qqqi_short", "tax") + (("bridge_month",) if core.freq == "Q" else ())
    lv = level_path(start, total, income)
    return _paths(ticker="QQQI", freq=core.freq, total=total, income=income, level=lv, dps=income * lv[:, :-1],
                  parts=_log_parts(parts, total), flags=flags,
                  meta={"coverage": c, "strikes": strikes, "fee": fee, "dist_rule": rule, "dist_rate": rate})


# ---------------------------------------------------------------- TLT

def long_yields(y10, ff, s20_0: float, s30_0: float, sub: int = 1, noise: bool = True, rng=None) -> dict:
    """20 / 30-year yields (decimal) from the 10-year and fed funds level paths (n, T + 1, decimal), stepping each
    period `sub` times on linearly interpolated y10 / ff. Returns sub-step paths (n, T * sub + 1)."""
    P = prm.structural()["tlt"]["spread_models"]
    y10, ff = np.asarray(y10, float), np.asarray(ff, float)
    n, T1 = y10.shape
    fr = np.arange(sub + 1) / sub
    Y = np.concatenate([y10[:, :1]] + [y10[:, t - 1:t] + fr[None, 1:] * (y10[:, t:t + 1] - y10[:, t - 1:t])
                                       for t in range(1, T1)], axis=1)
    F = np.concatenate([ff[:, :1]] + [ff[:, t - 1:t] + fr[None, 1:] * (ff[:, t:t + 1] - ff[:, t - 1:t])
                                      for t in range(1, T1)], axis=1)
    g = rng if rng is not None else np.random.default_rng(0)
    out = {}
    for key, s0 in (("s20", s20_0), ("s30", s30_0)):
        m = P[key]
        s = np.empty_like(Y)
        s[:, 0] = s0
        for t in range(1, Y.shape[1]):
            sl0, sl1 = Y[:, t - 1] - F[:, t - 1], Y[:, t] - F[:, t]
            ds = (m["a"] + m["b_dslope"] * (sl1 - sl0) + m["g_dy10"] * (Y[:, t] - Y[:, t - 1])
                  - m["kappa"] * (s[:, t - 1] - (m["c0"] + m["c1"] * sl0)))
            if noise:
                ds = ds + m["sigma"] * g.standard_normal(n)
            s[:, t] = s[:, t - 1] + ds
        out[key] = s
    w = prm.structural()["tlt"]["w20"]
    y20, y30 = Y + out["s20"], Y + out["s30"]
    return {"y10": Y, "y20": y20, "y30": y30, "y_tlt": w * y20 + (1 - w) * y30}


def tlt_returns(y: np.ndarray, ppy: int, fee: float, cy_ratio0: float, cy_halflife_years: float) -> dict:
    """Per step from the TLT yield path (n, S + 1): carry - D(y) dy + C(y)/2 dy^2 - fee, the coupon income, parts."""
    P = prm.structural()["tlt"]
    y0, y1 = y[:, :-1], y[:, 1:]
    dy = y1 - y0
    D, C = np.polyval(P["D_of_y"], y0), np.polyval(P["C_of_y"], y0)
    coupon, dur, conv, fee_p = y0 / ppy, -D * dy, 0.5 * C * dy * dy, np.full(y0.shape, -fee / ppy)
    total = coupon + dur + conv + fee_p
    t = np.arange(y0.shape[1]) / ppy
    ratio = 1.0 + (cy_ratio0 - 1.0) * 0.5 ** (t / cy_halflife_years)
    income = np.maximum(y0 * ratio[None, :] - fee, 0.0) / ppy
    return {"total": total, "income": income, "coupon": coupon, "duration": dur, "convexity": conv, "fee": fee_p,
            "D": D, "C": C}


def tlt(core: CoreView, vol: VolState, p: dict, start: float, rates: dict) -> object:
    S = prm.structural()["tlt"]
    fee = float(p.get("fee", fees.fee("TLT")))
    sub = 1 if core.freq == "M" else 3
    y10_0 = core.y10[:, 0] / 100.0
    s20_0 = np.asarray(p.get("s20_0", rates["y20"] - rates["y10"]), float)
    s30_0 = np.asarray(p.get("s30_0", rates["y30"] - rates["y10"]), float)
    LY = long_yields(core.y10 / 100.0, core.ff / 100.0, s20_0 + 0 * y10_0, s30_0 + 0 * y10_0, sub,
                     noise=p.get("noise", True), rng=stream(core.set_id, core.key, "spread", "TLT"))
    r = tlt_returns(LY["y_tlt"], 12, fee, float(p.get("cy_ratio0", S["current_yield_over_ytm_latest"])),
                    float(p.get("cy_halflife_years", S["cy_halflife_years"])))
    n = core.n
    if sub == 1:
        total, income = r["total"], r["income"]
        parts = {k: r[k] for k in ("coupon", "duration", "convexity", "fee")}
    else:
        T = core.T
        tm, im = r["total"].reshape(n, T, 3), r["income"].reshape(n, T, 3)
        g = np.ones((n, T))
        cash = np.zeros((n, T))
        acc = {k: np.zeros((n, T)) for k in ("coupon", "duration", "convexity", "fee")}
        for j in range(3):
            cash = cash + im[:, :, j] * g
            for k in acc:
                acc[k] = acc[k] + r[k].reshape(n, T, 3)[:, :, j] * g
            g = g * (1.0 + tm[:, :, j] - im[:, :, j])
        total, income = g + cash - 1.0, cash
        parts = acc
    lv = level_path(start, total, income)
    flags = ("tax",) + (("long_yield_range",) if (LY["y_tlt"].max() > S["yield_range_fitted"][1] + 0.03) else ())
    return _paths(ticker="TLT", freq=core.freq, total=total, income=income, level=lv, dps=income * lv[:, :-1],
                  parts=_log_parts(parts, total), flags=flags,
                  meta={"fee": fee, "y_tlt_end_p50": float(np.median(LY["y_tlt"][:, -1]))})
