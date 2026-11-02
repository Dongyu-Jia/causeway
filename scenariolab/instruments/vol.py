"""The joint volatility state (VolState, INSTRUMENTS_DESIGN §2.2): owned by the instrument layer, one per core view.

The leveraged funds' drag, QQQI's premiums and the estimated tickers' residual scale all need the variance inside a
period, which no core outputs (the quarterly SCHEMA lists VIX as E3, never an input). So the state is derived
from the core's own return paths, jointly for the NASDAQ-100 and the US market (fitted to 2014, tested 2015-2026):

    log RV_t = a + phi log RV_t-1 + b1 |x_t| + b2 min(x_t, 0) + b3 DD_t + b4 dspread_t + sigma eps_t,
    corr(eps_ndx, eps_mkt) = rho (0.786 monthly, 0.73 quarterly)

with x the period's log price return of the core's nasdaq / us_equity asset, DD the log drawdown of the cumulated
path (the anchor's drawdown carried in) and dspread the Baa - 10y change. The NDX equation is instr-structural's
rv_M / rv_Q, the market's instr-design's d03 / d02 (vol_joint.json). Implied vol for QQQI, at each period start:

    log VXN_t = c0 + c1 log sigma_RV,t + c2 log sigma_RV,t-1 + c3 DD_t + c4 dspread_t + sigma_u u,  sigma_ATM = 0.855 VXN

(period 0 uses the anchor's VXN). Sources (vol_source): "model" (default: eps drawn every period from the shared
stream, so every ticker sees the same state; noise=False gives the mean path E[RV] = exp(m + V/2)), "drawn" (the
monthly engine only: the drawn historical month's actual RV shifted by the model's return terms), "core" (a core
that outputs the NDX RV), "fixed" (a constant annual vol: sensitivity only, biased: TQQQ +4.6%/yr).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import params as prm
from .core import CoreView
from .streams import stream


@dataclass
class VolState:
    freq: str
    source: str
    rv_ndx: np.ndarray            # (n, T) NDX realized variance per period (sum of squared daily log returns)
    rv_mkt: np.ndarray            # (n, T) US market realized variance per period
    sigma_atm: np.ndarray         # (n, T) NDX ATM implied vol at each period start, annual
    dd_ndx: np.ndarray            # (n, T) log drawdowns at each period end
    dd_mkt: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def ppy(self) -> int:
        return 12 if self.freq == "M" else 4

    def rs3_ndx(self, x: np.ndarray) -> np.ndarray:
        """E[sum of cubed daily log returns | the period's log return, RV] (rs3 = a1 RV^1.5 + a2 x RV)."""
        p = prm.structural()["vol"][f"rs3_{self.freq}"]
        return p["a1"] * self.rv_ndx ** 1.5 + p["a2"] * x * self.rv_ndx

    def mkt_vol_quarterly(self) -> np.ndarray:
        """(n, Q) annualized US-market vol per quarter, sqrt(4 x the quarter's RV): the estimated recipes' residual
        scale (it matches instr-mapped's std x sqrt(252) definition: correlation 0.9994, ratio 0.999)."""
        if self.freq == "Q":
            return np.sqrt(4.0 * self.rv_mkt)
        n, T = self.rv_mkt.shape
        Q = T // 3
        return np.sqrt(4.0 * self.rv_mkt[:, :Q * 3].reshape(n, Q, 3).sum(axis=2))


def _anchor(anchor: dict | None) -> dict:
    if anchor is None:
        anchor = prm.packaged_anchors()
    return anchor["vol"] if "vol" in anchor else anchor


def _drawdown_path(x: np.ndarray, dd0: float) -> np.ndarray:
    """Log drawdown at every period end of the cumulated log returns, starting from the anchor's drawdown."""
    lp = np.cumsum(x, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.full((x.shape[0], 1), -dd0), lp], axis=1), axis=1)[:, 1:]
    return lp - peak


def model_paths(x_ndx, x_mkt, dspread, freq: str, anchor: dict | None = None, rng=None, noise: bool = True) -> dict:
    """The joint recursion. x_* (n, T) log price returns, dspread (n, T) pp. Returns lrv / rv / dd for both."""
    a = _anchor(anchor)
    S, V = prm.structural()["vol"], prm.vol_joint()
    key = "monthly" if freq == "M" else "quarterly"
    bn, sn = np.asarray(S[f"rv_{freq}"]["coef"]), float(S[f"rv_{freq}"]["sigma"])
    bm, sm = np.asarray(V[key]["mkt"]["coef"]), float(V[key]["mkt"]["sigma"])
    rho = float(V[key]["rho"])
    n, T = x_ndx.shape
    dd_n, dd_m = _drawdown_path(x_ndx, a["dd_ndx"]), _drawdown_path(x_mkt, a["dd_mkt"])
    lag_n = np.full(n, a["lrv_ndx"] if freq == "M" else a["lrv_ndx_q"])
    lag_m = np.full(n, a["lrv_mkt"] if freq == "M" else a["lrv_mkt_q"])
    lrv_n, lrv_m = np.empty((n, T)), np.empty((n, T))
    rv_n, rv_m = np.empty((n, T)), np.empty((n, T))
    var_n, var_m = np.zeros(n), np.zeros(n)
    g = rng if rng is not None else np.random.default_rng(0)
    for t in range(T):
        xn, xm = x_ndx[:, t], x_mkt[:, t]
        mn = (bn[0] + bn[1] * lag_n + bn[2] * np.abs(xn) + bn[3] * np.minimum(xn, 0.0) + bn[4] * dd_n[:, t]
              + bn[5] * dspread[:, t])
        mm = (bm[0] + bm[1] * lag_m + bm[2] * np.abs(xm) + bm[3] * np.minimum(xm, 0.0) + bm[4] * dd_m[:, t]
              + bm[5] * dspread[:, t])
        if noise:
            z1 = g.standard_normal(n)
            z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * g.standard_normal(n)
            lrv_n[:, t], lrv_m[:, t] = mn + sn * z1, mm + sm * z2
            rv_n[:, t], rv_m[:, t] = np.exp(lrv_n[:, t]), np.exp(lrv_m[:, t])
        else:
            var_n = sn * sn + bn[1] * bn[1] * var_n
            var_m = sm * sm + bm[1] * bm[1] * var_m
            lrv_n[:, t], lrv_m[:, t] = mn, mm
            rv_n[:, t], rv_m[:, t] = np.exp(mn + var_n / 2), np.exp(mm + var_m / 2)
        lag_n, lag_m = lrv_n[:, t], lrv_m[:, t]
    return {"lrv_ndx": lrv_n, "lrv_mkt": lrv_m, "rv_ndx": rv_n, "rv_mkt": rv_m, "dd_ndx": dd_n, "dd_mkt": dd_m}


def ndx_rv_path(x: np.ndarray, freq: str, init: dict, dspread: np.ndarray | None = None, noise: bool = False,
                rng=None) -> dict:
    """The NDX equation alone (instr-structural's vol_paths): the replays on history, which have no market leg.
    init: {"lrv": the log RV of the period before the start, "dd": the drawdown there}. Without dspread the
    equation fitted without it (rv_<f>_nospread). noise=False: the mean path, E[RV] = exp(m + V/2)."""
    S = prm.structural()["vol"]
    mp = S[f"rv_{freq}"] if dspread is not None else S[f"rv_{freq}_nospread"]
    b, sig = np.asarray(mp["coef"]), float(mp["sigma"])
    x = np.atleast_2d(np.asarray(x, float))
    n, T = x.shape
    dd = _drawdown_path(x, float(init.get("dd", 0.0)))
    lag = np.full(n, float(init["lrv"]))
    lrv, rv = np.empty((n, T)), np.empty((n, T))
    var = np.zeros(n)
    g = rng if rng is not None else np.random.default_rng(0)
    for t in range(T):
        feats = [lag, np.abs(x[:, t]), np.minimum(x[:, t], 0.0), dd[:, t]]
        if dspread is not None:
            feats.append(np.atleast_2d(dspread)[:, t])
        m = b[0] + sum(bk * f for bk, f in zip(b[1:], feats))
        if noise:
            lrv[:, t] = m + sig * g.standard_normal(n)
            rv[:, t] = np.exp(lrv[:, t])
        else:
            var = sig * sig + b[1] * b[1] * var
            lrv[:, t], rv[:, t] = m, np.exp(m + var / 2)
        lag = lrv[:, t]
    return {"rv": rv, "lrv": lrv, "dd": dd}


def implied_vol(rv_ndx, dd_ndx, dspread, freq: str, anchor: dict | None = None, rng=None,
                noise: bool = True) -> np.ndarray:
    """(n, T) ATM implied vol at each period start: 0.855 x the IV model's VXN on the state known at the end of
    the previous period (period 0: the anchor's VXN)."""
    a = _anchor(anchor)
    S = prm.structural()["vol"]
    iv, mult = S["iv_M"], float(S["iv_atm_multiplier_on_vxn"])
    bi = np.asarray(iv["coef"])
    ppy = 12.0 if freq == "M" else 4.0
    n, T = rv_ndx.shape
    lprev = a["lrv_ndx_prev"] if freq == "M" else a["lrv_ndx_q"]
    l0 = a["lrv_ndx"] if freq == "M" else a["lrv_ndx_q"]
    lv = np.concatenate([np.full((n, 1), 0.5 * np.log(np.exp(lprev) * ppy)),
                         np.full((n, 1), 0.5 * np.log(np.exp(l0) * ppy)), 0.5 * np.log(rv_ndx * ppy)], axis=1)
    g = rng if rng is not None else np.random.default_rng(0)
    sig = np.empty((n, T))
    sig[:, 0] = mult * float(a["vxn"])
    for t in range(1, T):
        mu = bi[0] + bi[1] * lv[:, t + 1] + bi[2] * lv[:, t] + bi[3] * dd_ndx[:, t - 1] + bi[4] * dspread[:, t - 1]
        mu = mu + (float(iv["sigma"]) * g.standard_normal(n) if noise else float(iv["sigma"]) ** 2 / 2)
        sig[:, t] = mult * np.exp(mu)
    return sig


def drawn_rv(drawn: np.ndarray, x_sim: np.ndarray, which: str = "ndx") -> np.ndarray:
    """Monthly engine only: the drawn historical month's actual RV, shifted by the RV model's return terms to the
    simulated month's own return (a drawn 2008-10 keeps its crisis volatility, sized to the simulated return)."""
    from . import data as D
    if which == "ndx":
        T = D.ndx_rv_table()
        b = dict(zip(prm.structural()["vol"]["rv_M_nospread"]["feats"], prm.structural()["vol"]["rv_M_nospread"]["coef"][1:]))
        ba, bneg = b["abs_r"], b["neg_r"]
    else:
        T = D.mkt_rv("M")
        x = D.us_market_daily()
        T = T.assign(x=x.groupby(x.index.to_period("M")).sum().reindex(T.index))
        c = prm.vol_joint()["monthly"]["mkt"]["coef"]
        ba, bneg = c[2], c[3]
    keys = np.asarray(T.index.astype(str))
    pos = {k: i for i, k in enumerate(keys)}
    dm = np.asarray(drawn).astype(str)
    ii = np.vectorize(lambda m: pos.get(m, -1))(dm)
    if (ii < 0).any():
        raise ValueError("drawn months outside the realized-variance table")
    xd = T["x"].to_numpy()[ii]
    lrv = np.log(T["rv"].to_numpy()[ii])
    shift = ba * (np.abs(x_sim) - np.abs(xd)) + bneg * (np.minimum(x_sim, 0) - np.minimum(xd, 0))
    return np.exp(lrv + shift)


def vol_state(core: CoreView, source: str = "model", anchor: dict | None = None, noise: bool = True,
              fixed_vol: float | None = None, fixed_vol_mkt: float | None = None) -> VolState:
    """The volatility state of a core view. The shocks come from the view's shared stream (set, key, "vol")."""
    x_ndx, x_mkt = core.x("nasdaq"), core.x("us_equity")
    dsp = core.dspread()
    rng = stream(core.set_id, core.key, "vol")
    if source == "fixed":
        if not fixed_vol:
            raise ValueError("vol_source 'fixed' needs fixed_vol (annual, e.g. 0.25)")
        fm = fixed_vol_mkt or 0.7 * fixed_vol
        rv_n = np.full(x_ndx.shape, fixed_vol ** 2 / core.ppy)
        rv_m = np.full(x_ndx.shape, fm ** 2 / core.ppy)
        a = _anchor(anchor)
        dd_n, dd_m = _drawdown_path(x_ndx, a["dd_ndx"]), _drawdown_path(x_mkt, a["dd_mkt"])
    else:
        mp = model_paths(x_ndx, x_mkt, dsp, core.freq, anchor, rng, noise)
        rv_n, rv_m, dd_n, dd_m = mp["rv_ndx"], mp["rv_mkt"], mp["dd_ndx"], mp["dd_mkt"]
        if source == "drawn":
            if core.freq != "M" or core.drawn is None:
                raise ValueError("vol_source 'drawn' needs the monthly engine's drawn months (CoreView.drawn)")
            rv_n = drawn_rv(core.drawn, x_ndx, "ndx")
            rv_m = drawn_rv(core.drawn, x_mkt, "mkt")
        elif source == "core":
            if core.rv_core is None:
                raise ValueError("vol_source 'core' needs CoreView.rv_core")
            rv_n = np.asarray(core.rv_core, float)
        elif source != "model":
            raise ValueError(f"unknown vol_source {source!r}")
    sig = implied_vol(rv_n, dd_n, dsp, core.freq, anchor, rng, noise)
    return VolState(freq=core.freq, source=source, rv_ndx=rv_n, rv_mkt=rv_m, sigma_atm=sig, dd_ndx=dd_n, dd_mkt=dd_m,
                    meta={"noise": noise, "fixed_vol": fixed_vol})
