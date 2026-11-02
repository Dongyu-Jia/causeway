"""run_layer: every recipe on one core view, sharing one volatility state (INSTRUMENTS_DESIGN §2.4).

    from scenariolab.instruments import layer
    core = CoreView.from_sim(sim, ctx.db, scenario_id="baseline")
    out = layer.run_layer(core)                         # all 16 tickers, the 2026-09 anchor, defaults
    out.paths["TQQQ"].total                             # (n, T) simple total returns
    tot, price, inc = out.year_factors()                # (n, Y, K): withdrawal.year_factors's convention

Conventions (the same as withdrawal.year_factors): income is per start-of-period value and the holding compounds
on its price return inside the period; a year's income is the year's cash per start-of-year value. A payout rule
only splits the total return into cash and price, it never changes the total (T6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ESTIMATED, LEVERAGED, STRUCTURAL, TICKERS
from . import anchors as anc
from . import estimated as est
from . import params as prm
from . import structural as st
from .core import CoreView
from .vol import VolState, vol_state


@dataclass
class InstrumentPaths:
    ticker: str
    freq: str
    total: np.ndarray                      # (n, T) simple total return per period (distributions reinvested)
    income: np.ndarray                     # (n, T) cash paid in the period / value at the start of the period
    level: np.ndarray                      # (n, T + 1) USD price per share, ex-distribution (start = start price)
    dps: np.ndarray | None = None          # cash per share: (n, Q) quarterly for the estimated, (n, T) QQQI / TLT
    alive: np.ndarray | None = None        # (n, T) leveraged funds: False from the wipe-out on
    parts: dict = field(default_factory=dict)   # additive LOG parts per period; sum = log(1 + total)
    flags: tuple = ()                      # caveat keys -> UI chips (catalog.json "chips")
    meta: dict = field(default_factory=dict)

    @property
    def price(self) -> np.ndarray:
        """Price-only return per period (total - income)."""
        return self.total - self.income


@dataclass
class InstrumentSet:
    core: CoreView
    paths: dict
    vol: VolState
    meta: dict

    @property
    def tickers(self) -> list[str]:
        return list(self.paths)

    def stacked(self, tickers=None) -> tuple[np.ndarray, np.ndarray]:
        """(total, income) (n, T, K) in the order of `tickers`."""
        ks = list(tickers or self.paths)
        return (np.stack([self.paths[k].total for k in ks], axis=2),
                np.stack([self.paths[k].income for k in ks], axis=2))

    def year_factors(self, tickers=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(total, price, income) (n, Y, K) yearly growth factors, as portfolio.withdrawal.year_factors."""
        from ..portfolio.withdrawal import year_factors
        total, income = self.stacked(tickers)
        f = year_factors(total, income, np.ones((total.shape[0], total.shape[1] + 1)),
                         months=12 if self.core.freq == "M" else 4)
        return f.total, f.price, f.income

    def diagnostics(self) -> dict:
        return diagnostics(self)


def _start_prices(anchor: dict) -> dict:
    return {t: float(v["price"]) for t, v in anchor["prices"].items()}


def run_layer(core: CoreView, tickers=TICKERS, params: dict | None = None, vol_source: str = "model",
              anchor: dict | None = None, alpha_mode: str = "shrunk", residual_mode: str = "fixed",
              vol: VolState | None = None, fixed_vol: float | None = None) -> InstrumentSet:
    """Every recipe in `tickers` on `core` with one shared VolState.

    params: per-ticker overrides, e.g. {"TQQQ": {"s_eff": "latest", "ruin": False}, "QQQI": {"coverage": 0.54,
            "dist_rule": "premium_plus_dividend"}, "MAIN": {"alpha_mode": "half"}}; the defaults are the design's.
    anchor: the start state (anchors.current(month)); default: the packaged block of the core's anchor month."""
    params = params or {}
    unknown = [t for t in tickers if t not in TICKERS]
    if unknown:
        raise KeyError(f"未知标的：{', '.join(unknown)}")
    if anchor is None:
        month = str(core.labels[0])[:7]
        anchor = anc.current(month, prefer_data=False)
    if vol is None:
        vol = vol_state(core, source=vol_source, anchor=anchor, fixed_vol=fixed_vol)
    prices = _start_prices(anchor)
    paths: dict = {}
    for t in tickers:
        p = params.get(t, {})
        if t == "QQQ":
            paths[t] = st.qqq(core, vol, p, prices.get(t, 1.0))
        elif t in LEVERAGED:
            paths[t] = st.leveraged(core, vol, p, prices.get(t, 1.0), t)
        elif t == "QQQI":
            paths[t] = st.qqqi(core, vol, p, prices.get(t, 1.0))
        elif t == "TLT":
            paths[t] = st.tlt(core, vol, p, prices.get(t, 1.0), anchor["rates"])
        elif t == "GLD":
            paths[t] = st.gld(core, vol, p, prices.get(t, 1.0))
    est_t = [t for t in tickers if t in ESTIMATED]
    if est_t:
        paths.update(est.simulate(core, vol.mkt_vol_quarterly(), est_t, params, anchor["estimated"],
                                  alpha_mode=alpha_mode, residual_mode=residual_mode))
    ordered = {t: paths[t] for t in tickers}
    meta = {"anchor_month": anchor.get("month"), "anchor_source": anchor.get("source"), "vol_source": vol_source,
            "alpha_mode": alpha_mode, "residual_mode": residual_mode, "params_version": prm.version(),
            "freq": core.freq, "n": core.n, "T": core.T, "set_id": core.set_id, "key": core.key}
    return InstrumentSet(core=core, paths=ordered, vol=vol, meta=meta)


# ---------------------------------------------------------------- diagnostics (T18: report, not a hard failure)

def _path_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (a * b).sum(axis=1) / den
    return float(np.nanmedian(c))


def credit_link_status(sims: dict, hist: float | None, tol: float | None = None) -> dict:
    """T18's verdict on one BDC / MLP ticker from its simulated quarterly correlation with US equity per scenario
    ({scenario id: sim}) against history: "bad" when history is more than `tol` (bridges.json, 0.15) above the
    simulation in any scenario (the credit_link chip), else the neutral credit_link_ok. Without a history or a
    simulated value the chip stays (bad)."""
    tol = float(prm.bridges()["credit_link_flag"]["corr_tolerance"] if tol is None else tol)
    vals = {k: float(v) for k, v in (sims or {}).items() if v is not None and np.isfinite(v)}
    if hist is None or not vals:
        return {"bad": True, "sim_lo": None, "sim_hi": None, "hist": hist, "n_over": None, "n": len(vals), "tol": tol}
    over = sorted(k for k, v in vals.items() if hist - v > tol)
    return {"bad": bool(over), "sim_lo": min(vals.values()), "sim_hi": max(vals.values()), "hist": float(hist),
            "n_over": len(over), "n": len(vals), "over": over, "tol": tol}


def resolve_credit_link(flags, status: dict | None) -> list:
    """A ticker's flags with credit_link kept when its status is bad and turned into credit_link_ok otherwise."""
    if status is None:
        return list(flags)
    return [("credit_link" if status["bad"] else "credit_link_ok") if f == "credit_link" else f for f in flags]


def diagnostics(s: InstrumentSet) -> dict:
    """Core-consistency checks per view: quarterly corr(d spread, US equity), corr(d_gz, US equity) and the
    BDC / MLP correlation with US equity, simulated (median over paths) against history; a deviation above the
    tolerance (0.15) is listed in `warnings` (credit_link_status turns the tickers' gaps into their chips)."""
    B = prm.bridges()
    q = s.core.quarterly()
    lr_eq = q.lr("us_equity")
    gz, src = q.gz_path()
    tol = B["credit_link_flag"]["corr_tolerance"]
    out = {"corr_dspread_us_equity": {"sim": _path_corr(q.dspread(), lr_eq),
                                      "hist": B["hist"]["corr_dspread_us_equity"]},
           "corr_dgz_us_equity": {"sim": _path_corr(gz, lr_eq), "hist": B["hist"]["corr_dgz_us_equity"],
                                  "gz_source": src},
           "ticker_corr_us_equity": {}, "warnings": []}
    for t, ip in s.paths.items():
        lt = np.log1p(np.maximum(ip.total, -0.999999))
        if ip.freq == "M":
            n, T = lt.shape
            lt = lt[:, :T // 3 * 3].reshape(n, T // 3, 3).sum(axis=2)
        h = B["hist_corr_with_us_equity"].get(t)
        out["ticker_corr_us_equity"][t] = {"sim": _path_corr(lt, lr_eq[:, :lt.shape[1]]), "hist": h}
    for k in ("corr_dspread_us_equity", "corr_dgz_us_equity"):
        if abs(out[k]["sim"] - out[k]["hist"]) > tol:
            out["warnings"].append(f"{k}: 模拟 {out[k]['sim']:.2f}，历史 {out[k]['hist']:.2f}")
    for t in B["credit_link_flag"]["tickers"]:
        c = out["ticker_corr_us_equity"].get(t)
        if c and c["hist"] is not None and abs(c["sim"] - c["hist"]) > tol:
            out["warnings"].append(f"{t} 与美股相关：模拟 {c['sim']:.2f}，历史 {c['hist']:.2f}")
    out["consistent"] = not out["warnings"]
    return out


# ---------------------------------------------------------------- the continuation (years 16..40)

def rows_for_chunk(ids: list[str], n_per_scenario: int, r0: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(row_group, row_index) of continuation rows r0 .. r0 + n - 1: row r continues scenario ids[r // N]'s path
    r % N (the portfolio service's one-path-per-row stacking), so each row keeps its alpha draw."""
    r = np.arange(r0, r0 + n)
    s = np.minimum(r // n_per_scenario, len(ids) - 1)
    return np.asarray(ids, dtype=object)[s].astype(str), (r - s * n_per_scenario).astype(np.int64)


def continuation_factors(sim, db, offset: int, row_group: np.ndarray, row_index: np.ndarray, chunk: int,
                         tickers=TICKERS, set_id: str = "builtin", anchor: dict | None = None,
                         alpha_modes=("shrunk", "half", "raw")) -> dict:
    """Years offset/12 + 1 .. of one continuation chunk (portfolio.service.build_continuation's simulate() of the
    whole horizon): the layer runs on every month (its own vol state, payout state and residual blocks, keyed
    ("continuation", chunk)) and the yearly factors after `offset` are returned. alpha draws stay keyed by the
    scenario row (row_group, row_index, set_id), so a path keeps its alpha across the seam (T19).

    Returns {"tickers", "total", "price", "income" (n, Y, K) float32, "alpha": {mode: (total, price, income)} for
    the estimated tickers' other alpha variants (ESTIMATED order)}."""
    from dataclasses import replace

    from ..portfolio.withdrawal import year_factors
    core = CoreView.from_sim(sim, db, scenario_id=f"chunk{chunk}", set_id="continuation")
    # shared streams: the continuation's own (vol, residual blocks); per-row alpha: the scenario rows' draws
    core = replace(core, row_group=np.asarray(row_group).astype(str), row_index=np.asarray(row_index, dtype=np.int64),
                   row_set_id=set_id)
    out = run_layer(core, tickers, anchor=anchor, alpha_mode=alpha_modes[0])
    total, income = out.stacked(tickers)
    ones = np.ones((core.n, core.T + 1))
    f = year_factors(total[:, offset:], income[:, offset:], ones[:, offset:])
    res = {"tickers": list(tickers), "total": f.total.astype(np.float32), "price": f.price.astype(np.float32),
           "income": f.income.astype(np.float32), "alpha": {}}
    est_t = [t for t in tickers if t in ESTIMATED]
    for mode in alpha_modes[1:]:
        if not est_t:
            break
        e = est.simulate(core, out.vol.mkt_vol_quarterly(), est_t, None, out_anchor(out, anchor), alpha_mode=mode)
        tt = np.stack([e[t].total for t in est_t], axis=2)[:, offset:]
        ii = np.stack([e[t].income for t in est_t], axis=2)[:, offset:]
        g = year_factors(tt, ii, ones[:, offset:])
        res["alpha"][mode] = (g.total.astype(np.float32), g.price.astype(np.float32), g.income.astype(np.float32))
    res["alpha_set_id"] = set_id
    return res


def out_anchor(out: InstrumentSet, anchor: dict | None) -> dict:
    if anchor is not None:
        return anchor["estimated"]
    return anc.current(str(out.core.labels[0])[:7], prefer_data=False)["estimated"]


__all__ = ["InstrumentPaths", "InstrumentSet", "run_layer", "diagnostics", "credit_link_status", "resolve_credit_link",
           "continuation_factors", "rows_for_chunk", "STRUCTURAL", "ESTIMATED", "TICKERS"]
