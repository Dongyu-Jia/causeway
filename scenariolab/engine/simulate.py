"""Scenario path simulator.

Pipeline for one scenario (or one backtest window):
  1. MonthDB holds the historical months the simulator may draw from, with
     their factor changes, states, eras and the asset-model residuals.
  2. Per phase, Entropy Pooling over block starts turns views into
     block-start probabilities (pooling.pool_phase).
  3. Weighted block bootstrap draws month indices (sampler.sample_phases).
  4. Factor changes of the drawn months are re-anchored to today's levels;
     rate changes are scaled by the ratio of today's level to the historical
     level, so a 1980 +3pp month does not become +3pp at a 4% starting point
     (capped, so near-zero-rate months are not blown up either). Rates,
     spreads, unemployment and the real oil price revert slowly toward
     anchors, which a block bootstrap would otherwise lose.
  5. Assets = today's transmission (fitted factor model) applied to the
     simulated factor changes + the drawn month's own residual (which carries
     the cross-asset co-movement the factors do not explain) + an optional
     capital-market-assumption drift adjustment + scenario overrides.
     Treasury, crude oil and cash are built structurally from the yield, oil
     and policy-rate paths.
  6. Optional valuation state (ValuationModel, off by default): the market's
     log P/D is tracked along every path (price log return minus the drawn
     month's dividend growth), its fair value moves with the simulated real
     yield, profit share and expected inflation, and equity prices get a pull
     -kappa * (log P/D - fair value), beta-scaled for sectors.
  7. Optional return attribution (simulate(attribution=True)): every part the
     engine adds up, averaged per path and phase (Attribution), without
     changing any simulated number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from .factor_model import STRUCTURAL_ASSETS, AssetFactorModel, base_columns, features_from_paths
from .pooling import uncond_frequency
from .sampler import sample_phases

# Factor changes the simulator always tracks (beyond the model's own factors).
SIM_FACTORS = sorted(set(config.FACTOR_IDS) | {"cpi"})
# Factors whose source may be missing (a cached file that is not there): their months are not
# required, and a missing value counts as no change.
OPTIONAL_FACTORS = {f["id"] for f in config.FACTORS if f.get("optional")}


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
        # optional factors, and factors a panel built before they existed lacks, are not required
        optional = {f for f in SIM_FACTORS if f in OPTIONAL_FACTORS or f"f_{f}" not in panel.columns}
        needed = [f"f_{f}" for f in SIM_FACTORS if f not in optional] + base_columns(model.spec)
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
        self.factors = P.reindex(columns=[f"f_{f}" for f in SIM_FACTORS]).copy()
        opt = [f"f_{f}" for f in SIM_FACTORS if f in optional]
        if opt and self.factors[opt].isna().any().any():
            self.factors[opt] = self.factors[opt].fillna(0.0)          # missing = no change
        self.factor_index = {f: i for i, f in enumerate(SIM_FACTORS)}
        self.resid = resid.loc[P.index, model.assets]
        self.prev_levels = prev.loc[P.index]
        self.eras = np.array([era_of(d) for d in P.index])
        self.extra_cols = extra_cols or []
        self.rare = np.ones(len(P))
        for r in config.RARE_MONTHS:
            self.rare[(P.index >= pd.Timestamp(r["start"])) & (P.index <= pd.Timestamp(r["end"]))] *= r["weight"]
        self._uncond = None
        # real oil (log oil / CPI): sample median, and the average monthly real drift an
        # unconditional path actually draws (block edges and rare months are under-drawn)
        log_real = np.log(P["lvl_oil"].to_numpy(float)) - np.log(P["lvl_cpi"].to_numpy(float))
        self.oil_log_real_median = float(np.nanmedian(log_real))
        real_oil = np.nan_to_num(P["f_oil"].to_numpy(float) - P["f_cpi"].to_numpy(float))
        self.oil_real_drift = float(self.uncond_frequency() @ real_oil)

    def __len__(self) -> int:
        return len(self.dates)

    def uncond_frequency(self) -> np.ndarray:
        """Each month's share of the draws of an unconditional 15-year path (the weights of every
        'historical average' the engine projects forward)."""
        if self._uncond is None:
            self._uncond = uncond_frequency(self.prior(), self.rare)
        return self._uncond

    def prior(self, half_life_years: float | None = None) -> np.ndarray:
        """Uniform prior (rare months down-weighted), optionally decayed toward recent months."""
        T = len(self)
        w = np.ones(T)
        if half_life_years:
            age = np.arange(T)[::-1] / 12.0
            w = 0.5 ** (age / half_life_years)
        w = w * self.rare
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
                               shrink_relative: float = 0.5, gold_reversion: float = 0.0) -> dict:
    """Annual drift adjustment per modelled asset (decimal, arithmetic).

    "historical": no adjustment, assets keep their sample-average drift.
    "valuation_state": "valuation_adjusted" without its constant reversion
    term, for the valuation-state mode (ValuationModel), whose state-dependent
    pull replaces it (reversion is forced to 0, so nothing is counted twice).
    "valuation_adjusted": the market's price drift loses the historical
    re-rating tailwind (average change in log P/D) and gains a partial
    mean-reversion term from today's P/D toward the sample median. Each other
    equity asset gets that adjustment scaled by its beta to the market, and its
    historical drift relative to the market (e.g. Nasdaq's tech decades) is
    shrunk by `shrink_relative` toward zero, since past sector winners are not
    reliably future ones. Gold has no cash flow, so its "valuation" is its
    real price: it loses the sample's real appreciation (mostly the one-off
    1970s repricing after Bretton Woods), so in the long run it keeps pace
    with inflation; `gold_reversion` > 0 would also pull today's real price
    toward the sample median (off by default: in the backtest windows that
    term made gold worse in 3 of 4).

    Every historical average is weighted the way an unconditional path draws
    the months (MonthDB.uncond_frequency), so the targets are what the engine
    delivers. The simulator already removes the oil factor's real drift from
    the oil path (simulate), so that part of an asset's relative drift is not
    shrunk a second time here.
    """
    if method not in ("historical", "valuation_adjusted", "valuation_state"):
        raise ValueError(f"unknown CMA method {method!r}")
    if method == "valuation_state":
        reversion = 0.0
    P = db.frame
    assets = db.model.assets
    w = db.uncond_frequency()
    col = lambda c: np.nan_to_num(P[c].to_numpy(float))                     # noqa: E731
    hist = {a: float(w @ col(f"p_{a}")) * 12.0 for a in assets}
    oil_drift = db.oil_real_drift * 12.0
    coef = db.model.coef
    oil_beta = {a: (float(sum(coef.loc[a, f] for f in coef.columns if f == "oil" or f.startswith("oil_l")))
                    if coef is not None else 0.0) for a in assets}
    out = {"method": method, "reversion": reversion, "horizon_years": horizon_years,
           "shrink_relative": shrink_relative, "gold_reversion": gold_reversion, "historical_price_drift": hist,
           "adjustment": {a: 0.0 for a in assets}, "beta_to_market": {}, "relative_drift": {}}
    if method == "historical":
        return out
    full = db.panel
    pd_now = 100.0 / float(full.loc[:anchor.date, "s_div_yield"].dropna().iloc[-1])
    pd_series = 100.0 / P["s_div_yield"].dropna()
    pd_median = float(pd_series.median())
    rerating = float(w @ col("f_valuation")) * 12.0
    reversion_term = float(reversion * np.log(pd_median / pd_now) / horizon_years)
    delta_mkt = -rerating + reversion_term
    mkt = col("p_us_equity")
    mkt_c = mkt - w @ mkt
    for a in assets:
        if a in STRUCTURAL_ASSETS or a == "gold":
            continue
        y = col(f"p_{a}")
        beta = float(w @ ((y - w @ y) * mkt_c) / (w @ (mkt_c * mkt_c)))
        rel = hist[a] - beta * hist["us_equity"] - (oil_beta[a] - beta * oil_beta["us_equity"]) * oil_drift
        out["beta_to_market"][a] = beta
        out["relative_drift"][a] = rel
        out["adjustment"][a] = beta * delta_mkt - shrink_relative * rel
    if "gold" in assets:
        g = np.log1p(col("p_gold")) - col("f_cpi")                                  # monthly real log change
        level = np.cumsum(g)                                                         # real log price index
        real_drift = float(w @ g) * 12.0
        gap = float(np.median(level) - level[-1])
        gold_term = float(gold_reversion * gap / horizon_years)
        out["adjustment"]["gold"] = -real_drift + oil_beta["gold"] * oil_drift + gold_term
        out["gold"] = {"real_drift_hist": real_drift, "real_gap_log": gap, "reversion_term": gold_term}
    out.update({"pd_now": pd_now, "pd_median": pd_median, "rerating_tailwind": rerating,
                "reversion_term": reversion_term, "market_adjustment": float(delta_mkt),
                "target_price_drift": {a: hist[a] + out["adjustment"][a] for a in assets}})
    return out


# ---------------------------------------------------------------- valuation state
#
# price = dividends x (P/D), so the multiple is not a same-month driver of the price (that would explain
# the price with itself): it is a state. Along every path log P/D_t+1 = log P/D_t + log(1 + p_t) - g_t,
# where p_t is the simulated market price return and g_t the drawn month's growth of trailing dividends.
# The fair value log P/D* = const + sum_k b_k x_k is estimated on the month database (the training months)
# and moves with the simulated levels x_k; equity prices get -kappa/12 * (log P/D - log P/D*) a month,
# scaled by each asset's beta to the market (as the CMA scales its market adjustment).

VALUATION_COLUMNS = {"real_y10": "s_real_y10", "profit_share": "s_profit_share_rt", "infl_expect": "s_infl_expect",
                     "y10": "s_y10"}
VALUATION_FEATURE_NAMES = {"const": "常数", "real_y10": "10年期实际收益率（%）", "profit_share": "企业税后利润/GDP（%）",
                           "infl_expect": "通胀预期代理（%）", "y10": "10年期名义收益率（%）"}
VALUATION_HAC_LAGS = 60          # Newey-West lags for the levels regression (monthly, persistent regressors)
ECM_MONTHS = 12                  # horizon of the in-sample error-correction regression ("ecm" kappa)
KAPPA_BOUNDS = (0.0, 1.0)


def hac_ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """OLS of y on [1, X] with Newey-West (Bartlett) t statistics: (beta, t, R^2, residuals)."""
    Z = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    e = y - Z @ beta
    inv = np.linalg.pinv(Z.T @ Z)
    u = Z * e[:, None]
    S = u.T @ u
    for lag in range(1, min(lags, len(y) - 1) + 1):
        G = u[lag:].T @ u[:-lag]
        S += (1.0 - lag / (lags + 1.0)) * (G + G.T)
    se = np.sqrt(np.maximum(np.diag(inv @ S @ inv), 0.0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    sst = float(((y - y.mean()) ** 2).sum())
    return beta, t, (1.0 - float(e @ e) / sst) if sst > 0 else float("nan"), e


def market_betas(db: MonthDB) -> dict[str, float]:
    """Beta of each model asset's price return to the market's, weighted like an unconditional path
    (the same estimate capital_market_assumptions scales its market adjustment by)."""
    P, w = db.frame, db.uncond_frequency()
    col = lambda c: np.nan_to_num(P[c].to_numpy(float))                     # noqa: E731
    mkt = col("p_us_equity")
    mkt_c = mkt - w @ mkt
    out = {}
    for a in db.model.assets:
        y = col(f"p_{a}")
        out[a] = float(w @ ((y - w @ y) * mkt_c) / (w @ (mkt_c * mkt_c)))
    out["us_equity"] = 1.0
    return out


@dataclass
class ValuationModel:
    """Fair value of the market's log P/D and the pull toward it, estimated on a MonthDB's months only."""
    features: tuple[str, ...]
    coef: dict[str, float]            # "const" and each feature: log P/D per unit of the feature
    tstat: dict[str, float]           # Newey-West t statistics
    r2: float
    sample: tuple[str, str]
    n_obs: int
    kappa: float                      # annual speed of the pull toward the fair value
    kappa_how: str                    # "fixed" or "ecm"
    ecm: dict                         # in-sample error-correction estimate (reported either way)
    log_pd0: float                    # log P/D at the anchor
    state0: dict[str, float]          # levels at the anchor (feature values, infl_expect, profit_share)
    infl_alpha: float                 # EWMA weight of the inflation-expectation recursion (per month)
    g_div: np.ndarray                 # (len(db),) monthly log growth of trailing dividends of each db month
    ps_change: np.ndarray             # (len(db),) monthly change of the published profit share
    beta: dict[str, float]            # model asset -> beta to the market (0: no pull, constant income)
    income: str = "constant"          # "constant" anchor income yield, or "path": yield_0 * (P/D_0 / P/D_t)^beta
    gap_sd: float = float("nan")      # in-sample sd of log P/D - fair value

    def fair_log_pd(self, levels: dict[str, np.ndarray]) -> np.ndarray:
        return self.coef["const"] + sum(self.coef[f] * np.asarray(levels[f], float) for f in self.features)

    def today(self) -> dict:
        fair = float(self.fair_log_pd(self.state0))
        return {"log_pd": self.log_pd0, "log_pd_fair": fair, "gap_log": self.log_pd0 - fair,
                "pd": float(np.exp(self.log_pd0)), "pd_fair": float(np.exp(fair)),
                "pull_annual": -self.kappa * (self.log_pd0 - fair)}

    def summary(self) -> dict:
        return {"features": list(self.features), "coef": self.coef, "tstat": self.tstat, "r2": self.r2,
                "sample": list(self.sample), "n_obs": self.n_obs, "kappa": self.kappa, "kappa_how": self.kappa_how,
                "half_life_years": (float(np.log(2.0) / self.kappa) if self.kappa > 0 else None),
                "ecm": self.ecm, "income": self.income, "gap_sd": self.gap_sd, "state0": self.state0,
                "beta": self.beta, "today": self.today()}


def _ecm(lpd: pd.Series, gap: pd.Series, months: int = ECM_MONTHS) -> dict:
    """Error correction on the training months: log P/D change over the next `months` on today's gap.

    Only pairs whose end month is itself a database month enter. The slope s maps to an annual
    pull kappa = -ln(1 + s) * 12 / months. The gap is an in-sample residual, so the estimate is
    biased toward faster reversion (the regression picks the fair value that makes it revert).
    """
    ahead = lpd.reindex(lpd.index + pd.DateOffset(months=months)).to_numpy(float)
    ok = np.isfinite(ahead) & np.isfinite(gap.to_numpy(float))
    if ok.sum() < 60:
        return {"slope": None, "t": None, "kappa": None, "n": int(ok.sum()), "months": months}
    y = (ahead - lpd.to_numpy(float))[ok]
    beta, t, _, _ = hac_ols(y, gap.to_numpy(float)[ok], lags=months)
    s = float(beta[1])
    kappa = float(np.clip(-np.log1p(max(s, -0.999)) * 12.0 / months, *KAPPA_BOUNDS))
    return {"slope": s, "t": float(t[1]), "kappa": kappa, "n": int(ok.sum()), "months": months}


def fit_valuation(db: MonthDB, anchor: Anchor, features=None, kappa=None, income: str | None = None,
                  hac_lags: int = VALUATION_HAC_LAGS) -> ValuationModel:
    """Fair-value regression, pull speed and dividend growth from the db's months; state at anchor.date.

    db.frame holds only the training months (a backtest's MonthDB ends at train_end), and the anchor
    levels are the last row of the panel up to anchor.date, so nothing after the anchor is used.
    kappa: an annual number, or "ecm" for the in-sample error-correction estimate (_ecm).
    """
    s = config.VALUATION
    features = tuple(features or s["features"])
    kappa = s["kappa"] if kappa is None else kappa
    income = income or s["income"]
    unknown = [f for f in features if f not in VALUATION_COLUMNS]
    if unknown or income not in ("constant", "path"):
        raise ValueError(f"unknown valuation features {unknown} or income {income!r}")
    P = db.frame
    lpd = np.log(100.0 / P["s_div_yield"])
    X = P[[VALUATION_COLUMNS[f] for f in features]]
    ok = lpd.notna() & X.notna().all(axis=1)
    beta, t, r2, e = hac_ols(lpd[ok].to_numpy(float), X[ok].to_numpy(float), hac_lags)
    coef = {"const": float(beta[0]), **{f: float(b) for f, b in zip(features, beta[1:])}}
    tstat = {"const": float(t[0]), **{f: float(v) for f, v in zip(features, t[1:])}}
    gap = pd.Series(e, index=lpd[ok].index)
    ecm = _ecm(lpd[ok], gap)
    if isinstance(kappa, str):
        if kappa != "ecm":
            raise ValueError(f"unknown kappa {kappa!r}")
        k, how = (ecm["kappa"] if ecm["kappa"] is not None else 0.0), "ecm"
    else:
        k, how = float(np.clip(float(kappa), *KAPPA_BOUNDS)), "fixed"
    row = db.panel.loc[:anchor.date].iloc[-1]
    state0 = {f: float(row[VALUATION_COLUMNS[f]]) for f in VALUATION_COLUMNS}
    hl = float(s["infl_half_life_months"])
    g_div = (np.log1p(P["p_us_equity"]) - P["f_valuation"]).to_numpy(float)
    ps_change = db.panel["s_profit_share_rt"].diff().reindex(db.dates).to_numpy(float)
    betas = market_betas(db)
    betas = {a: (0.0 if a in STRUCTURAL_ASSETS or a == "gold" else b) for a, b in betas.items()}
    return ValuationModel(features=features, coef=coef, tstat=tstat, r2=float(r2),
                          sample=(lpd[ok].index.min().strftime("%Y-%m"), lpd[ok].index.max().strftime("%Y-%m")),
                          n_obs=int(ok.sum()), kappa=k, kappa_how=how, ecm=ecm,
                          log_pd0=float(np.log(100.0 / row["s_div_yield"])), state0=state0,
                          infl_alpha=1.0 - 0.5 ** (1.0 / hl), g_div=np.nan_to_num(g_div),
                          ps_change=np.nan_to_num(ps_change), beta=betas, income=income, gap_sd=float(np.std(e)))


def valuation_settings(valuation: dict | None = None) -> dict | None:
    """The valuation-state settings to use, or None when the mode is off.

    None means the engine default (config.VALUATION, on only if its "enabled" is true); a dict is
    merged over config.VALUATION and must carry "enabled": True to turn the mode on.
    """
    s = {**config.VALUATION, **(valuation or {})}
    return s if s.get("enabled") else None


def valuation_setup(db: MonthDB, anchor: Anchor, settings: dict | None = None) -> tuple[dict, ValuationModel]:
    """(CMA, ValuationModel) of the valuation-state mode under `settings` (config.VALUATION by default)."""
    s = {**config.VALUATION, **(settings or {})}
    cma = capital_market_assumptions(db, anchor, method=s["cma"])
    return cma, fit_valuation(db, anchor, features=s["features"], kappa=s["kappa"], income=s["income"])


def _valuation_paths(vm: ValuationModel, model: AssetFactorModel, anchor: Anchor, price: np.ndarray,
                     idx: np.ndarray, X: np.ndarray, fi: dict, y10: np.ndarray,
                     ps_drift: np.ndarray | None = None) -> tuple[dict, dict]:
    """Add the pull to `price` (in place, month by month) and return (level paths, income per asset).

    The profit share moves with the drawn months' published changes plus `ps_drift` (pp per month, a
    phase's factor_drift on profit_share), so a scenario that states a profit-share drift moves the
    fair value with it, as a drift on CPI or the 10-year yield does through pie and y10.
    """
    n, T = idx.shape
    pi = 1200.0 * X[:, :, fi["cpi"]]
    pie = np.empty((n, T + 1))
    pie[:, 0] = vm.state0["infl_expect"]
    for t in range(T):
        pie[:, t + 1] = pie[:, t] + vm.infl_alpha * (pi[:, t] - pie[:, t])
    levels = {"infl_expect": pie, "real_y10": y10 - pie, "y10": y10}
    dps = vm.ps_change[idx]
    if ps_drift is not None:
        dps = dps + np.asarray(ps_drift, float)[None, :]
    ps = vm.state0["profit_share"] + np.concatenate([np.zeros((n, 1)), np.cumsum(dps, axis=1)], axis=1)
    levels["profit_share"] = ps
    fair = vm.fair_log_pd(levels)
    lpd = np.empty((n, T + 1))
    lpd[:, 0] = vm.log_pd0
    g = vm.g_div[idx]
    eq = model.assets.index("us_equity")
    pulled = [(m, a, vm.beta.get(a, 0.0)) for m, a in enumerate(model.assets) if vm.beta.get(a, 0.0) != 0.0]
    y0 = {a: anchor.income_yield.get(a, 0.0) / 12.0 for a in model.assets}
    income = {a: np.empty((n, T)) for _, a, _ in pulled} if vm.income == "path" else {}
    k_m = vm.kappa / 12.0
    for t in range(T):
        pull = -k_m * (lpd[:, t] - fair[:, t])
        for m, a, b in pulled:
            price[:, t, m] += b * pull
            if income:
                income[a][:, t] = y0[a] * np.exp(b * (vm.log_pd0 - lpd[:, t]))
        inc = income["us_equity"][:, t] if income else y0["us_equity"]
        total = np.maximum(price[:, t, eq] + inc, -0.95)            # the clip simulate() applies to totals
        lpd[:, t + 1] = lpd[:, t] + np.log1p(total - inc) - g[:, t]
    paths = {"pd": np.exp(lpd), "pd_fair": np.exp(fair), "infl_expect": pie, "real_y10": y10 - pie,
             "profit_share": ps}
    return paths, income


# ---------------------------------------------------------------- return attribution
#
# A simulated month's return is a sum the engine adds up itself, so the split is exact. For a model asset
#   price = sum_k beta_k x df_k   (the SIMULATED factor changes: after level scaling, reversion, the oil
#                                  pull or lock and the factor drifts, i.e. what the transmission sees)
#         + intercept + CMA adjustment / 12 + asset_drift / 12 + residual x resid_scale
#         (+ the valuation pull when that mode runs with kappa > 0)
#   total = price + income, floored at -95% (the floor's effect is its own term, in practice always 0).
# A mean is linear, so the per-path phase means of the parts add up to the phase mean of the total
# (arithmetic monthly returns; x 12 = an annual rate). The structural assets: the 10-year bond = coupon
# income + the repricing on the simulated yield change (term y10); cash = income; crude oil = the oil path
# (term oil). A transmission that is not a plain linear map of same-month factors (an MLP, lags, the
# inflation-regime interactions) is not split: its whole systematic part is the term "unattributed".

ATTRIBUTION_FIXED = ("intercept", "unattributed", "cma", "overrides", "valuation_pull", "residual", "income",
                     "floor")
STRUCTURAL_TERM = {"treasury_10y": "y10", "crude_oil": "oil"}       # the factor that builds a structural asset


def attribution_method(spec) -> tuple[str, str]:
    """("linear", "") when the transmission splits exactly into one term per feature, else
    ("unattributed", why) for a transmission whose systematic part stays one block."""
    if spec.kind != "linear":
        return "unattributed", "传导模型是非线性的（神经网络），因子贡献不能相加，系统性部分整块列为未归因"
    if spec.lags or spec.regime:
        return "unattributed", "传导模型含滞后项或通胀状态交互项，系统性部分整块列为未归因"
    return "linear", ""


@dataclass
class Attribution:
    """Each asset's mean monthly return per path and phase, split into additive terms (all x 12).

    values[..., k] are the terms (factor ids of the transmission, then ATTRIBUTION_FIXED ids and the
    structural factor ids "oil" / "y10"); their sum over k equals `total` (the phase mean of the floored
    total returns, float64, before SimOutput's float32 cast) up to floating-point rounding.
    """
    terms: list[str]
    factor_terms: list[str]           # the transmission's terms (their sum = the factor part of a model asset)
    values: np.ndarray                # (n, P, A, K) mean monthly contribution x 12
    total: np.ndarray                 # (n, P, A) mean monthly total return x 12
    phase_months: list[int]
    assets: list[str]
    kinds: dict[str, str]             # asset -> "model" | "treasury" | "cash" | "crude_oil"
    method: str                       # "linear" | "unattributed"
    reason: str = ""

    def get(self, term: str) -> np.ndarray:
        """(n, P, A) values of one term (zeros where it does not apply)."""
        return self.values[..., self.terms.index(term)]


def _phase_means(x: np.ndarray, phase_months: list[int]) -> np.ndarray:
    """(n, T, ...) -> (n, P, ...): the mean over each phase's months."""
    out, a = [], 0
    for m in phase_months:
        out.append(x[:, a:a + m].mean(axis=1) if m else np.zeros(x.shape[:1] + x.shape[2:]))
        a += m
    return np.stack(out, axis=1) if out else np.zeros(x.shape[:1] + (0,) + x.shape[2:])


def _attribute(model: AssetFactorModel, assets: list[str], phase_months: list[int], base_ids: list[str],
               base_paths: np.ndarray, sys: np.ndarray, adj: np.ndarray, overrides: list[dict],
               resid: np.ndarray, pull: np.ndarray | None, income: np.ndarray, raw_total: np.ndarray,
               total: np.ndarray) -> Attribution:
    """The exact split of simulate()'s returns (reads its arrays, changes none of them)."""
    method, reason = attribution_method(model.spec)
    factor_terms = list(base_ids) if method == "linear" else []
    terms = factor_terms + [t for t in STRUCTURAL_TERM.values() if t not in factor_terms]
    terms += [t for t in ATTRIBUTION_FIXED if (t != "intercept" or method == "linear")
              and (t != "unattributed" or method != "linear") and (t != "valuation_pull" or pull is not None)]
    ti = {t: k for k, t in enumerate(terms)}
    n, P, A = total.shape[0], len(phase_months), len(assets)
    V = np.zeros((n, P, A, len(terms)))
    pm = lambda x: _phase_means(x, phase_months) * 12.0          # noqa: E731
    # phase means of whole arrays at once: (n, P, assets) / (n, P, model assets) / (n, P, features)
    income_m, floor_m, price_m, resid_m = pm(income), pm(total - raw_total), pm(raw_total - income), pm(resid)
    sys_m = pm(sys) if method != "linear" else None
    pull_m = pm(pull) if pull is not None else None
    if method == "linear":
        coef = model.coef.loc[:, list(base_ids)].to_numpy(float)    # (model assets, features)
        mean_f = pm(base_paths)                                    # (n, P, features)
    kinds = {}
    for j, a in enumerate(assets):
        V[:, :, j, ti["income"]] = income_m[:, :, j]
        V[:, :, j, ti["floor"]] = floor_m[:, :, j]
        if a == "cash":
            kinds[a] = "cash"
            continue
        if a in STRUCTURAL_TERM:
            kinds[a] = "treasury" if a == "treasury_10y" else "crude_oil"
            V[:, :, j, ti[STRUCTURAL_TERM[a]]] = price_m[:, :, j]
            continue
        kinds[a] = "model"
        m = model.assets.index(a)
        if method == "linear":
            for k, f in enumerate(base_ids):
                V[:, :, j, ti[f]] = mean_f[:, :, k] * coef[m, k]
            V[:, :, j, ti["intercept"]] = float(model.intercept.iloc[m]) * 12.0
        else:
            V[:, :, j, ti["unattributed"]] = sys_m[:, :, m]
        V[:, :, j, ti["cma"]] = adj[m] * 12.0
        for p, ov in enumerate(overrides[:P]):
            V[:, p, j, ti["overrides"]] = sum(annual / 12.0 for aid, annual in (ov.get("asset_drift") or {}).items()
                                            if aid == a) * 12.0
        V[:, :, j, ti["residual"]] = resid_m[:, :, m]
        if pull_m is not None:
            V[:, :, j, ti["valuation_pull"]] = pull_m[:, :, m]
    return Attribution(terms=terms, factor_terms=factor_terms, values=V, total=pm(total),
                       phase_months=[int(m) for m in phase_months], assets=list(assets), kinds=kinds,
                       method=method, reason=reason)


# ---------------------------------------------------------------- simulation

@dataclass
class SimOutput:
    total: np.ndarray                 # (n, T, A) total returns, decimal
    income: np.ndarray                # (n, T, A) income part of total
    macro: dict[str, np.ndarray]      # (n, T+1) level paths, index 0 = anchor
    idx: np.ndarray                   # (n, T) drawn month positions in the db
    assets: list[str] = field(default_factory=list)
    start: str = ""
    attribution: Attribution | None = None    # simulate(attribution=True) only


def _scale(now: np.ndarray, hist_prev: np.ndarray, shift: float, lo: float, hi: float) -> np.ndarray:
    return np.clip((now + shift) / (hist_prev + shift), lo, hi)


# A drawn month's rate change is multiplied by clip((now + shift) / (then + shift), lo, hi).
# The floor keeps the 1980 swings small at today's lower rates; the cap stops
# months made from near-zero rates (2022's +4pp hikes from 0.1%) from being
# multiplied several times over. Credit spreads narrow in proportion to how
# wide they are, but widen from tight levels too (2007: 1.6 -> 6 in 2008), so
# a widening month keeps at least SPREAD_WIDENING_FLOOR of its size.
RATE_SCALING = {"ff": (1.5, 0.3, 1.5), "y10": (1.5, 0.3, 1.5), "spread": (1.0, 0.3, 1.5)}
SPREAD_WIDENING_FLOOR = 0.85
RATE_FACTORS = ("ff", "y10", "spread")
# Views or drifts on these say what oil does in a phase; such phases get no oil reversion.
OIL_VIEW_VARS = ("f_oil", "s_oil_yoy", "s_real_oil")


def with_oil_lock(phase_views: list[list[dict]], overrides: list[dict] | None) -> list[dict]:
    """Copies of the phase overrides with oil_locked set where the phase's views state an oil path."""
    out = [dict(ov or {}) for ov in (overrides or [])]
    out += [{} for _ in range(len(phase_views) - len(out))]
    for ov, views in zip(out, phase_views):
        if any(v.get("var") in OIL_VIEW_VARS for v in views or []):
            ov["oil_locked"] = True
    return out


def _spread_scale(level, prev, d):
    r = _scale(level, prev, *RATE_SCALING["spread"])
    return np.where(d > 0, np.maximum(r, SPREAD_WIDENING_FLOOR), r)


def spread_scaling_bias(db: MonthDB, grid: np.ndarray) -> np.ndarray:
    """Mean monthly change that level scaling adds to an unconditional spread path, per spread level.

    Scaling is asymmetric (widening keeps at least SPREAD_WIDENING_FLOOR, narrowing shrinks with
    the level), so without a correction the long-run spread would settle above spread_level.
    """
    w = db.uncond_frequency()
    d = np.nan_to_num(db.factors["f_spread"].to_numpy(float))
    prev = db.prev_levels["s_spread"].to_numpy(float)
    return np.array([float(w @ (d * _spread_scale(L, prev, d)) - w @ d) for L in grid])

# Block bootstrap keeps short-run dynamics but loses the long-run mean
# reversion of rates, unemployment and the real oil price, so those levels get
# a gentle monthly pull toward an anchor: policy rate -> r* + inflation
# (Taylor-like), 10-year -> r* + inflation + term premium, spread and
# unemployment -> fixed levels, real oil -> its sample median (the sample's
# average real oil drift, mostly the 1973-74 and 2000s repricings, is removed).
# oil_target_shift is a log shift of that real-oil target (0 = the 1973+ median):
# a scenario that states a structural change in oil demand or supply moves the
# long-run anchor instead of switching the reversion off for years (an oil
# factor_drift or oil view does that: see OIL_VIEW_VARS). Phases that state their
# own oil path are not pulled toward any target.
DEFAULT_REVERSION = {
    "kappa_ff": 0.02, "kappa_y10": 0.015, "kappa_spread": 0.03, "kappa_unrate": 0.015, "kappa_oil": 0.015,
    "r_star": 0.75, "term_premium": 0.75, "spread_level": 2.0, "unrate_level": 4.8, "oil_target_shift": 0.0,
}


def simulate(db: MonthDB, anchor: Anchor, phase_weights: list, phase_months: list[int],
             n_paths: int = 600, seed: int = 0, block: int = config.BLOCK_MONTHS, cma: dict | None = None,
             overrides: list[dict] | None = None, scale_rates: bool = True,
             reversion: dict | None = None, resid_mode: str = "coupled", resid_window: int | None = None,
             resid_coupling: float = 1.0, resid_seed: int | None = None,
             valuation: ValuationModel | None = None, attribution: bool = False) -> SimOutput:
    """Simulate `n_paths` monthly paths through the given phases.

    phase_weights: per phase, a sampler.BlockProbs (from pooling.pool_phase) or
    an array of month weights.
    overrides (one dict per phase, optional keys):
      factor_drift: {factor_id: annual change added to that factor; for ff, y10 and
                    spread it is added after level scaling, so it is delivered as asked}
      oil_locked:   the phase states its own oil path (with_oil_lock sets it from the
                    views; a factor_drift on oil implies it): no oil reversion there
      asset_drift:  {asset_id: annual price-return drift added (decimal)}
      resid_scale:  multiplier on drawn residuals (volatility of the unexplained part)
      r_star:       neutral real policy rate for this phase (e.g. negative under
                    financial repression)
    reversion: overrides for DEFAULT_REVERSION (pass kappa_*=0 to disable;
    oil_target_shift moves the real-oil anchor by that many log points).
    resid_mode / resid_window / resid_coupling / resid_seed: where the drawn
    months' residuals come from (residuals.draw; ablation only). The default
    "coupled" uses each drawn month's own residual, exactly as before.
    valuation: a ValuationModel (fit_valuation on this db) turns on the
    valuation state: the pull toward the fair value is added to equity price
    returns and macro gains "pd", "pd_fair", "real_y10", "infl_expect" and
    "profit_share" paths. Pass the matching CMA (valuation_setup), e.g.
    "valuation_state", whose market drift has no constant reversion. None (the
    default) leaves every number exactly as without it.
    attribution: also return SimOutput.attribution (Attribution: each asset's
    return per path and phase split into the parts above). Every simulated
    number is the same with or without it; off by default, so backtests, the
    forecast and the studies do not pay for it.
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
    rate_drift = {f: np.zeros(T) for f in RATE_FACTORS}
    ps_drift = np.zeros(T)                                   # profit-share drift, for the valuation state
    oil_locked = np.zeros(T, bool)
    for k, ov in enumerate(overrides):
        cols = phase_of == k
        for fid, annual in (ov.get("factor_drift") or {}).items():
            if fid in rate_drift:
                rate_drift[fid][cols] += annual / 12.0
            elif fid in fi:
                X[:, cols, fi[fid]] += annual / 12.0
            if fid == "profit_share":
                ps_drift[cols] += annual / 12.0
        if ov.get("oil_locked") or "oil" in (ov.get("factor_drift") or {}):
            oil_locked[cols] = True

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
    if scale_rates:
        sp_grid = np.linspace(0.3, 10.0, 98)
        sp_bias = spread_scaling_bias(db, sp_grid)
    for t in range(T):
        dff, dy, dsp = X[:, t, fi["ff"]], X[:, t, fi["y10"]], X[:, t, fi["spread"]]
        if scale_rates:
            dff = dff * _scale(ff[:, t], prev[:, t, 0], *RATE_SCALING["ff"])
            dy = dy * _scale(y10[:, t], prev[:, t, 1], *RATE_SCALING["y10"])
            dsp = dsp * _spread_scale(sp[:, t], prev[:, t, 2], dsp) - np.interp(sp[:, t], sp_grid, sp_bias)
        dff = dff + rate_drift["ff"][t]
        dy = dy + rate_drift["y10"][t]
        dsp = dsp + rate_drift["spread"][t]
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

    # ---- oil: no inherited real drift, real price pulled toward the sample median (shifted by
    # oil_target_shift), except in phases that state their own oil path (views or drift): there
    # the drawn months rule
    log_real = np.full(n_paths, np.log(anchor.oil))              # in anchor-month dollars
    target = db.oil_log_real_median + np.log(anchor.cpi) + float(rev["oil_target_shift"])
    for t in range(T):
        if not oil_locked[t]:
            X[:, t, fi["oil"]] += -db.oil_real_drift - rev["kappa_oil"] * (log_real - target)
        log_real = log_real + X[:, t, fi["oil"]] - X[:, t, fi["cpi"]]

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
    if resid_mode != "coupled":
        from . import residuals
        resid = residuals.draw(db, idx, phase_weights, phase_months, block, seed, mode=resid_mode,
                               window=resid_window, coupling=resid_coupling, resid_seed=resid_seed)

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
    income_path: dict = {}
    # the pull is added to `price` in place: keep a copy to measure it (only when it can be non-zero)
    pre_pull = price.copy() if attribution and valuation is not None and valuation.kappa != 0 else None
    if valuation is not None:
        vpaths, income_path = _valuation_paths(valuation, model, anchor, price, idx, X, fi, y10, ps_drift=ps_drift)
        macro.update(vpaths)

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
            inc = income_path[a] if a in income_path else anchor.income_yield.get(a, 0.0) / 12.0
            income[:, :, j] = inc
            total[:, :, j] = price[:, :, m] + inc
    floored = np.maximum(total, -0.95)
    att = None
    if attribution:
        att = _attribute(model, assets, list(phase_months), base_ids, base_paths, sys, adj, overrides, resid,
                         (price - pre_pull) if pre_pull is not None else None, income, total, floored)
    return SimOutput(total=floored.astype(np.float32), income=income.astype(np.float32),
                     macro={k: v.astype(np.float32) for k, v in macro.items()},
                     idx=idx, assets=assets, start=str(anchor.date.date()), attribution=att)
