"""Scenario path simulator.

Pipeline for one scenario (or one backtest window):
  1. MonthDB holds the historical months the simulator may draw from, with
     their factor changes, states, eras and the asset-model residuals.
  2. Per phase, Entropy Pooling turns views into month weights (ep.apply_views).
  3. Weighted block bootstrap draws month indices (sampler.sample_phases).
  4. Factor changes of the drawn months are re-anchored to today's levels;
     rate changes are scaled by the ratio of today's level to the historical
     level, so a 1980 +3pp month does not become +3pp at a 4% starting point.
  5. Assets = today's transmission (fitted factor model) applied to the
     simulated factor changes + the drawn month's own residual (which carries
     the cross-asset co-movement the factors do not explain) + an optional
     capital-market-assumption drift adjustment + scenario overrides.
     Treasury, crude oil and cash are built structurally from the yield, oil
     and policy-rate paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from .factor_model import STRUCTURAL_ASSETS, AssetFactorModel, base_columns, features_from_paths
from .sampler import sample_phases

# Factor changes the simulator always tracks (beyond the model's own factors).
SIM_FACTORS = sorted(set(config.FACTOR_IDS) | {"cpi"})


def era_of(date: pd.Timestamp) -> str:
    for e in config.ERAS:
        start = pd.Timestamp(e["start"])
        end = pd.Timestamp(e["end"]) if e["end"] else pd.Timestamp("2100-01-01")
        if start <= date <= end:
            return e["id"]
    return "unknown"


class MonthDB:
    """Historical months available to the simulator (rows of a fixed window)."""

    def __init__(self, panel: pd.DataFrame, model: AssetFactorModel,
                 start: str = config.MODEL_START, end=None, extra_cols: list[str] | None = None):
        P = panel.loc[start:end]
        needed = [f"f_{f}" for f in SIM_FACTORS] + base_columns(model.spec)
        resid = model.residuals(panel).loc[P.index]
        ok = P[needed].notna().all(axis=1) & resid.notna().all(axis=1)
        # rate levels in the month *before* each drawn month (for level scaling)
        prev = panel[["lvl_ff", "lvl_y10", "s_spread"]].shift(1).loc[P.index]
        ok &= prev.notna().all(axis=1)
        P = P[ok]
        self.panel = panel
        self.model = model
        self.dates = P.index
        self.frame = P                                  # all columns, for EP views
        self.factors = P[[f"f_{f}" for f in SIM_FACTORS]].copy()
        self.factor_index = {f: i for i, f in enumerate(SIM_FACTORS)}
        self.resid = resid.loc[P.index, model.assets]
        self.prev_levels = prev.loc[P.index]
        self.eras = np.array([era_of(d) for d in P.index])
        self.extra_cols = extra_cols or []

    def __len__(self) -> int:
        return len(self.dates)

    def prior(self, half_life_years: float | None = None) -> np.ndarray:
        """Uniform prior, or exponentially decayed toward recent months."""
        T = len(self)
        if not half_life_years:
            return np.full(T, 1.0 / T)
        age = np.arange(T)[::-1] / 12.0
        w = 0.5 ** (age / half_life_years)
        return w / w.sum()

    def era_weights(self, q: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for e, w in zip(self.eras, q):
            out[e] = out.get(e, 0.0) + float(w)
        return out


@dataclass
class Anchor:
    """Levels at the simulation start (the last observed month)."""
    date: pd.Timestamp
    ff: float
    y10: float
    tb3: float
    spread: float
    oil: float
    cpi: float
    unrate: float
    usd: float
    ip: float
    labor_share: float
    cpi_log_hist: np.ndarray          # last 12 monthly log CPI changes (oldest first)
    base_init: np.ndarray             # last `lags` base factor rows for the model
    income_yield: dict[str, float]    # annual income yield per asset (decimal)


def make_anchor(panel: pd.DataFrame, model: AssetFactorModel, date=None) -> Anchor:
    P = panel.loc[:date] if date is not None else panel
    last = P.index.max()
    row = P.loc[last]
    lags = max(model.spec.lags, 1)
    base = P[base_columns(model.spec)].iloc[-lags:].to_numpy(float)
    base = np.nan_to_num(base)
    income = {}
    for a in config.ASSET_IDS:
        dy = P[f"d_{a}"].iloc[-12:].sum()
        income[a] = float(dy) if np.isfinite(dy) else 0.0
    return Anchor(
        date=last, ff=float(row["lvl_ff"]), y10=float(row["lvl_y10"]), tb3=float(row["lvl_tb3"]),
        spread=float(row["s_spread"]), oil=float(row["lvl_oil"]), cpi=float(row["lvl_cpi"]),
        unrate=float(row["lvl_unrate"]), usd=float(row["lvl_usd"]), ip=float(row["lvl_ip"]),
        labor_share=float(row["lvl_labor_share"]),
        cpi_log_hist=P["f_cpi"].iloc[-12:].to_numpy(float),
        base_init=base, income_yield=income,
    )


# ---------------------------------------------------------------- CMA anchoring

def capital_market_assumptions(db: MonthDB, anchor: Anchor, method: str = "valuation_adjusted",
                               reversion: float = 0.3, horizon_years: float = 15.0,
                               shrink_relative: float = 0.5) -> dict:
    """Annual drift adjustment per modelled asset (decimal, arithmetic).

    "historical": no adjustment, assets keep their sample-average drift.
    "valuation_adjusted": the market's price drift loses the historical
    re-rating tailwind (average change in log P/D) and gains a partial
    mean-reversion term from today's P/D toward the sample median. Each other
    equity asset gets that adjustment scaled by its beta to the market, and its
    historical drift relative to the market (e.g. Nasdaq's tech decades) is
    shrunk by `shrink_relative` toward zero, since past sector winners are not
    reliably future ones.
    """
    P = db.frame
    assets = db.model.assets
    hist = {a: float(P[f"p_{a}"].mean() * 12.0) for a in assets}
    out = {"method": method, "reversion": reversion, "horizon_years": horizon_years,
           "shrink_relative": shrink_relative, "historical_price_drift": hist,
           "adjustment": {a: 0.0 for a in assets}, "beta_to_market": {}, "relative_drift": {}}
    if method == "historical":
        return out
    full = db.panel
    pd_now = 100.0 / float(full.loc[:anchor.date, "s_div_yield"].dropna().iloc[-1])
    pd_series = 100.0 / P["s_div_yield"].dropna()
    pd_median = float(pd_series.median())
    rerating = float(P["f_valuation"].mean() * 12.0)
    reversion_term = float(reversion * np.log(pd_median / pd_now) / horizon_years)
    delta_mkt = -rerating + reversion_term
    mkt = P["p_us_equity"].to_numpy(float)
    for a in assets:
        if a in STRUCTURAL_ASSETS or a == "gold":
            continue
        y = P[f"p_{a}"].to_numpy(float)
        beta = float(np.cov(y, mkt)[0, 1] / np.var(mkt, ddof=1))
        rel = hist[a] - beta * hist["us_equity"]
        out["beta_to_market"][a] = beta
        out["relative_drift"][a] = rel
        out["adjustment"][a] = beta * delta_mkt - shrink_relative * rel
    out.update({"pd_now": pd_now, "pd_median": pd_median, "rerating_tailwind": rerating,
                "reversion_term": reversion_term, "market_adjustment": float(delta_mkt),
                "target_price_drift": {a: hist[a] + out["adjustment"][a] for a in assets}})
    return out


# ---------------------------------------------------------------- simulation

@dataclass
class SimOutput:
    total: np.ndarray                 # (n, T, A) total returns, decimal
    income: np.ndarray                # (n, T, A) income part of total
    macro: dict[str, np.ndarray]      # (n, T+1) level paths, index 0 = anchor
    idx: np.ndarray                   # (n, T) drawn month positions in the db
    assets: list[str] = field(default_factory=list)
    start: str = ""


def _scale(now: np.ndarray, hist_prev: np.ndarray, shift: float, lo: float, hi: float) -> np.ndarray:
    return np.clip((now + shift) / (hist_prev + shift), lo, hi)


# Block bootstrap keeps short-run dynamics but loses the long-run mean
# reversion of rates and unemployment, so those levels get a gentle monthly
# pull toward an anchor: policy rate -> r* + inflation (Taylor-like), 10-year
# -> r* + inflation + term premium, spread and unemployment -> fixed levels.
DEFAULT_REVERSION = {
    "kappa_ff": 0.02, "kappa_y10": 0.015, "kappa_spread": 0.03, "kappa_unrate": 0.015,
    "r_star": 0.75, "term_premium": 0.75, "spread_level": 2.0, "unrate_level": 4.8,
}


def simulate(db: MonthDB, anchor: Anchor, phase_weights: list[np.ndarray], phase_months: list[int],
             n_paths: int = 600, seed: int = 0, block: int = 6, cma: dict | None = None,
             overrides: list[dict] | None = None, scale_rates: bool = True,
             reversion: dict | None = None) -> SimOutput:
    """Simulate `n_paths` monthly paths through the given phases.

    overrides (one dict per phase, optional keys):
      factor_drift: {factor_id: annual change added to that factor}
      asset_drift:  {asset_id: annual price-return drift added (decimal)}
      resid_scale:  multiplier on drawn residuals (volatility of the unexplained part)
      r_star:       neutral real policy rate for this phase (e.g. negative under
                    financial repression)
    reversion: overrides for DEFAULT_REVERSION (pass kappa_*=0 to disable).
    """
    rev = {**DEFAULT_REVERSION, **(reversion or {})}
    model = db.model
    assets = list(config.ASSET_IDS)
    T = int(sum(phase_months))
    idx = sample_phases(phase_weights, phase_months, n_paths, block, seed)
    fi = db.factor_index
    X = db.factors.to_numpy(float)[idx]                      # (n, T, F)
    prev = db.prev_levels.to_numpy(float)[idx]              # (n, T, 3) ff, y10, spread

    phase_of = np.concatenate([np.full(m, k) for k, m in enumerate(phase_months)]) if T else np.zeros(0, int)
    overrides = list(overrides or [])
    overrides += [{} for _ in range(len(phase_months) - len(overrides))]
    for k, ov in enumerate(overrides):
        cols = phase_of == k
        for fid, annual in (ov.get("factor_drift") or {}).items():
            if fid in fi:
                X[:, cols, fi[fid]] += annual / 12.0

    # CPI YoY at every level date (index 0 = anchor month), from the anchor's
    # last 12 actual monthly changes followed by the simulated ones. Computed
    # first because the rate anchors follow inflation.
    log_cpi_ext = np.concatenate([np.broadcast_to(anchor.cpi_log_hist, (n_paths, 12)), X[:, :, fi["cpi"]]], axis=1)
    roll = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_cpi_ext, axis=1)], axis=1)
    yoy = 100.0 * (np.exp(roll[:, 12:] - roll[:, :-12]) - 1.0)   # (n, T+1)
    infl_anchor = np.clip(yoy, -2.0, 15.0)
    r_star = np.full(T, rev["r_star"], dtype=float)
    for k, ov in enumerate(overrides):
        if "r_star" in ov:
            r_star[phase_of == k] = float(ov["r_star"])

    # ---- rates and unemployment re-anchored with level scaling + reversion
    ff = np.empty((n_paths, T + 1)); ff[:, 0] = anchor.ff
    y10 = np.empty((n_paths, T + 1)); y10[:, 0] = anchor.y10
    sp = np.empty((n_paths, T + 1)); sp[:, 0] = anchor.spread
    ur = np.empty((n_paths, T + 1)); ur[:, 0] = anchor.unrate
    for t in range(T):
        dff, dy, dsp = X[:, t, fi["ff"]], X[:, t, fi["y10"]], X[:, t, fi["spread"]]
        if scale_rates:
            dff = dff * _scale(ff[:, t], prev[:, t, 0], 1.0, 0.3, 3.0)
            dy = dy * _scale(y10[:, t], prev[:, t, 1], 1.0, 0.3, 3.0)
            dsp = dsp * _scale(sp[:, t], prev[:, t, 2], 0.0, 0.5, 2.0)
        ff_target = r_star[t] + infl_anchor[:, t]
        y10_target = r_star[t] + infl_anchor[:, t] + rev["term_premium"]
        dff = dff + rev["kappa_ff"] * (ff_target - ff[:, t])
        dy = dy + rev["kappa_y10"] * (y10_target - y10[:, t])
        dsp = dsp + rev["kappa_spread"] * (rev["spread_level"] - sp[:, t])
        dur = X[:, t, fi["unrate"]] + rev["kappa_unrate"] * (rev["unrate_level"] - ur[:, t])
        ff[:, t + 1] = np.clip(ff[:, t] + dff, 0.0, 25.0)
        y10[:, t + 1] = np.clip(y10[:, t] + dy, 0.1, 20.0)
        sp[:, t + 1] = np.clip(sp[:, t] + dsp, 0.3, 10.0)
        ur[:, t + 1] = np.clip(ur[:, t] + dur, 2.0, 25.0)
    X[:, :, fi["ff"]] = np.diff(ff, axis=1)
    X[:, :, fi["y10"]] = np.diff(y10, axis=1)
    X[:, :, fi["spread"]] = np.diff(sp, axis=1)
    X[:, :, fi["unrate"]] = np.diff(ur, axis=1)

    # ---- macro level paths
    def level_path(start: float, changes: np.ndarray, log: bool = True) -> np.ndarray:
        cum = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(changes, axis=1)], axis=1)
        return start * np.exp(cum) if log else start + cum

    cpi = level_path(anchor.cpi, X[:, :, fi["cpi"]])
    macro = {
        "cpi": cpi, "ff": ff, "y10": y10, "spread": sp, "unrate": ur, "cpi_yoy": yoy,
        "oil": level_path(anchor.oil, X[:, :, fi["oil"]]),
        "ip": level_path(anchor.ip, X[:, :, fi["ip"]]),
        "usd": level_path(anchor.usd, X[:, :, fi["usd"]]),
        "labor_share": level_path(anchor.labor_share, X[:, :, fi["labor_share"]]),
    }

    # ---- factor-explained part of asset returns (today's transmission)
    spec = model.spec
    base_ids = [c[2:] if c.startswith("f_") else c for c in base_columns(spec)]
    base_paths = np.stack([X[:, :, fi[b]] if b in fi else db.frame[b].to_numpy(float)[idx]
                           for b in base_ids], axis=2)
    hi = (yoy[:, :-1] > 4.0).astype(float) if spec.regime == "inflation" else None
    feats = features_from_paths(spec, base_paths, anchor.base_init[-max(spec.lags, 1):] if spec.lags else anchor.base_init[:0], hi)
    sys = model.predict_array(feats.reshape(-1, feats.shape[2])).reshape(n_paths, T, len(model.assets))
    resid = db.resid.to_numpy(float)[idx]                    # (n, T, assets)

    adj = np.zeros(len(model.assets))
    if cma:
        adj = np.array([cma["adjustment"].get(a, 0.0) / 12.0 for a in model.assets])
    price = sys + adj[None, None, :]
    for k, ov in enumerate(overrides):
        cols = phase_of == k
        scale = ov.get("resid_scale")
        if scale:
            resid[:, cols, :] *= scale
        for aid, annual in (ov.get("asset_drift") or {}).items():
            if aid in model.assets:
                price[:, cols, model.assets.index(aid)] += annual / 12.0
    price = price + resid

    total = np.zeros((n_paths, T, len(assets)))
    income = np.zeros_like(total)
    for j, a in enumerate(assets):
        if a == "crude_oil":
            total[:, :, j] = np.expm1(X[:, :, fi["oil"]])
        elif a == "treasury_10y":
            c = y10[:, :-1] / 100.0
            yn = np.maximum(y10[:, 1:] / 100.0, 1e-6)
            disc = (1.0 + yn / 2.0) ** (-20)
            px = (c / 2.0) * (1.0 - disc) / (yn / 2.0) + disc
            income[:, :, j] = c / 12.0
            total[:, :, j] = px - 1.0 + income[:, :, j]
        elif a == "cash":
            basis = anchor.ff - anchor.tb3
            rate = np.maximum(ff[:, :-1] - basis, 0.0) / 1200.0
            income[:, :, j] = rate
            total[:, :, j] = rate
        else:
            m = model.assets.index(a)
            inc = anchor.income_yield.get(a, 0.0) / 12.0
            income[:, :, j] = inc
            total[:, :, j] = price[:, :, m] + inc
    total = np.maximum(total, -0.95)
    return SimOutput(total=total.astype(np.float32), income=income.astype(np.float32),
                     macro={k: v.astype(np.float32) for k, v in macro.items()},
                     idx=idx, assets=assets, start=str(anchor.date.date()))
