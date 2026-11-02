"""The oracle-macro evaluation harness of the quarterly stack (SIM_DESIGN §3.9, §4, §6.6).

A port of the research round's sim_redesign/harness/simharness.py into the app.

Question it answers
-------------------
Scenario -> factor path -> asset path. Take the factor path as given (the realized one: "oracle macro", the stand-in
for a perfectly written scenario). A transmission model fitted only on data up to the train end is fed that path from
an origin forward. How close is its cumulative log PRICE return of each asset over 1, 4, 12 and 20 quarters to what
happened, and how wide is the band its own residual draw puts around it?

    from scenariolab.evaluation import harness as eh
    P  = eh.load_panel()                                                     # app panel + e_ shocks + v_ levels
    pf = eh.conditional_paths(eh.linear_model(eh.F), P, protocol="split")    # the factor round's F, monthly paths
    pq = eh.conditional_paths(eh.quarterly_model_fn(), P, protocol="split")  # the quarterly asset layer (L0)
    m  = eh.metrics(pd.concat([pf, pq]))                                     # asset x horizon x group (+ "mean10")
    eh.mean_table(m, "r2")

Plugging in a model (unchanged from the research harness)
---------------------------------------------------------
A model is a function model_fn(train: DataFrame, train_end: Timestamp) -> Fitted.
  * `train` is the monthly panel cut at train_end; nothing later exists for the fit.
  * The Fitted object provides assets and predict_path(vis, origin, months) -> DataFrame (months x assets) of predicted
    monthly LOG price returns; optionally simulate_paths(vis, origin, months, n_paths, seed) -> (n_paths, months, assets)
    or `resid` (training months x assets) for the default band, and predict_mult(vis, origin, months) -> the window's
    predicted change of the CAPE multiple (the placebo-corrected anatomy uses it).
  * `vis` (visible_panel) is the panel cut at the window's last month with every asset-derived column blanked from the
    origin on: realized macro of the window is visible (oracle), prices and valuations only before the origin.
  * A quarterly model puts each quarter's prediction in the quarter's last month (only window sums are scored):
    quarterly_model_fn() does this for the asset layer, reading the realized quarterly fields of the window's quarters
    from the quarterly panel (macro columns only: qmodel.assets.oracle_inputs) and the start-of-quarter states of the
    origin.

Protocols and metrics are the research harness's: "split" (train to 1994-12 / 2004-12 / 2014-12, windows inside the
following decade) and "rolling" (refit every January on data to the previous December, origins from 1985-01);
R² against the training-mean drift, its 90% moving-block bootstrap interval, corr, MAE, bias, hit rates, 80%
coverage, PIT; n_indep = span // window. Additions: CRPS of the model's band and its skill (crpss) against the
drift benchmark's distribution (the training mean plus h independent draws of demeaned training quarterly returns,
the research qmodel's benchmark), and paired_diff() for block-bootstrap intervals of a model difference (T-A).

Valuation anatomy (§6.6)
------------------------
corr(miss, Δmultiple)² is an identity (a constant forecast scores 0.83-1.00 on the harness), so the headline
statistics are placebo-corrected: (a) corr(predicted, realized Δmultiple) with n, (b) the share of the realized
multiple change the model predicts (the slope of realized on predicted, and R² against zero), (c) a placebo row (the
constant drift forecast) next to every model row. The research columns are kept in variance_anatomy() for export;
anatomy_headline() gives the table the UI shows.

Differences from the research file: the app's panel (the factor round's columns are native: f_spread is the
timing-fixed spread, f_gz, f_tp10_x_core4 and f_metals_oil are panel columns, B0's labor share is release dated), F
read from data/results/model_selection.json, Shiller's 2026 workbook from the project's raw cache (never downloaded),
no sys.path or bytecode handling, and the quarterly adapters, l0_reproduction() and the placebo-corrected anatomy.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..context import load_panel_with_shocks
from ..engine import residuals as _eres
from ..engine.factor_model import MODEL_ASSETS as _PROJECT_MODEL_ASSETS
from ..engine.factor_model import ModelSpec
from ..engine.pooling import BLOCK
from ..qmodel import config_q as cq

__version__ = "2.0"          # research harness 1.0, ported

# ---------------------------------------------------------------- constants
SCORE_ASSETS = list(cq.SCORE_ASSETS)
MODEL_ASSETS = list(_PROJECT_MODEL_ASSETS)          # 12: + crude_oil, treasury_10y (the joint training mask)
EQUITY_LIKE = list(cq.EQUITY_LIKE)
EXTRA_ASSETS = ["buseq"]                            # Ken French BusEq (tech), anatomy only (load_panel(extra=True))
HORIZONS_Q = (1, 4, 12, 20)
BENCH_START = config.MODEL_START                    # "1973-01"
SPLITS = {"1995-2004": ("1994-12", "1995-01", "2004-12"),
          "2005-2014": ("2004-12", "2005-01", "2014-12"),
          "2015-2026": ("2014-12", "2015-01", "2026-09")}
ROLL_FIRST, ROLL_REFIT = "1985-01", 12
ROLL_SPANS = {"1985-1994": ("1985-01", "1994-12"), "1995-2004": ("1995-01", "2004-12"),
              "2005-2014": ("2005-01", "2014-12"), "2015-2026": ("2015-01", "2026-09"),
              "1985-2026": ("1985-01", "2026-09")}
DECADES = {"1975-1984": ("1975-01", "1984-12"), **{k: v for k, v in ROLL_SPANS.items() if k != "1985-2026"}}
BAND = (0.10, 0.90)
N_PATHS, SEED = 500, 11

ASSET_PREFIXES = ("p_", "r_", "d_", "dy_", "v_")
ASSET_COLUMNS = {"lvl_gold", "lvl_nasdaq", "s_div_yield", "f_valuation", "s_cape", "f_cape", "s_wealth_income",
                 "f_wealth", "lvl_networth"}
KNOWN_LAG = {"v_sp_le": 3, "v_sp_lpe": 3, "v_sp_lcape": 3, "v_sp_le10r": 3, "v_sp_ld": 3, "v_sp_lpd": 3,
             "s_cape": 3, "s_profit_share_rt": 0, "s_fcf_share": 0}   # months beyond origin - 1


# ---------------------------------------------------------------- data
def _month_index(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx)).to_period("M").to_timestamp()


def load_shiller(path=None) -> pd.DataFrame:
    """Shiller's monthly S&P composite (price, dividend, earnings, cpi, gs10, cape) from the project's raw cache (the
    2026 workbook P0.3 imported; qmodel.sources_q). df.attrs["source"] names the file."""
    from ..qmodel import sources_q as sq
    df = sq.shiller_frame(path)
    df.attrs["source"] = df.attrs.get("path")
    return df


def add_extra_assets(P: pd.DataFrame) -> pd.DataFrame:
    """p_buseq / d_buseq / r_buseq: Ken French 12-industry BusEq, built like the panel's industries. Anatomy only."""
    from ..qmodel import sources_q as sq
    P = P.copy()
    tot = sq.french_table("12_Industry_Portfolios", "Average Value Weighted Returns -- Monthly")["BusEq"] / 100.0
    ex = sq.french_table("12_Industry_Portfolios_Wout_Div", "Average Value Weighted Returns -- Monthly")["BusEq"] / 100.0
    tot.index, ex.index = _month_index(tot.index), _month_index(ex.index)
    P["r_buseq"] = tot.reindex(P.index)
    P["d_buseq"] = (tot - ex).reindex(P.index)
    P["p_buseq"] = P["r_buseq"] - P["d_buseq"]
    return P


def _log_index(ret: pd.Series) -> pd.Series:
    """log price index from simple returns: cumulated log1p from the first return, 0 in the month before it."""
    lr = np.log1p(ret.astype(float))
    out = pd.Series(np.nan, index=ret.index)
    first = lr.first_valid_index()
    if first is None:
        return out
    out.loc[first:] = lr.loc[first:].fillna(0.0).cumsum()
    i = out.index.get_loc(first)
    if i > 0:
        out.iloc[i - 1] = 0.0
    return out


def add_valuation_columns(P: pd.DataFrame, shiller: pd.DataFrame | None = None) -> pd.DataFrame:
    """v_* log valuation levels (all masked from the origin on by visible_panel): per equity-like asset v_lp_a,
    v_ld12_a, v_lpd_a (v_lpd_us_equity = log(100 / s_div_yield)); Shiller's v_sp_lp, v_sp_le, v_sp_ld, v_sp_lpe,
    v_sp_lpd, v_sp_lcape, v_sp_lcpi and v_sp_le10r = v_sp_lp - v_sp_lcpi - v_sp_lcape."""
    P = P.copy()
    new = {}
    for a in EQUITY_LIKE + [x for x in EXTRA_ASSETS if f"p_{x}" in P.columns]:
        lp = _log_index(P[f"p_{a}"])
        px = np.exp(lp)
        d12 = (P[f"d_{a}"] * px.shift(1)).rolling(12).sum()
        new[f"v_lp_{a}"] = lp
        new[f"v_ld12_{a}"] = np.log(d12.where(d12 > 0))
        new[f"v_lpd_{a}"] = lp - new[f"v_ld12_{a}"]
    sh = load_shiller() if shiller is None else shiller
    src = sh.attrs.get("source")
    sh = sh.reindex(P.index)
    new["v_sp_lp"] = np.log(sh["price"])
    new["v_sp_le"] = np.log(sh["earnings"].where(sh["earnings"] > 0))
    new["v_sp_ld"] = np.log(sh["dividend"].where(sh["dividend"] > 0))
    new["v_sp_lcpi"] = np.log(sh["cpi"])
    new["v_sp_lcape"] = np.log(sh["cape"])
    new["v_sp_lpe"] = new["v_sp_lp"] - new["v_sp_le"]
    new["v_sp_lpd"] = new["v_sp_lp"] - new["v_sp_ld"]
    new["v_sp_le10r"] = new["v_sp_lp"] - new["v_sp_lcpi"] - new["v_sp_lcape"]
    out = pd.concat([P.drop(columns=[c for c in new if c in P.columns]), pd.DataFrame(new, index=P.index)], axis=1)
    out.attrs["shiller_source"] = src
    return out


def load_panel(valuation: bool = True, extra: bool = False) -> pd.DataFrame:
    """The app's monthly panel with the e_ structural shocks and rg_ regime columns (context.load_panel_with_shocks),
    month-start index; valuation: + the v_* levels; extra: + Ken French BusEq (anatomy only)."""
    P = load_panel_with_shocks().copy()
    P.index.freq = None
    if extra:
        P = add_extra_assets(P)
    if valuation:
        P = add_valuation_columns(P)
    return P


def asset_derived_columns(P: pd.DataFrame) -> list[str]:
    return [c for c in P.columns if c.startswith(ASSET_PREFIXES) or c in ASSET_COLUMNS]


def visible_panel(P: pd.DataFrame, origin, end, mask_cols: list[str] | None = None) -> pd.DataFrame:
    """What a model may see when predicting the window origin..end: the panel cut at `end`, with every asset-derived
    column blanked from `origin` on."""
    origin, end = pd.Timestamp(origin), pd.Timestamp(end)
    V = P.loc[:end].copy()
    cols = asset_derived_columns(P) if mask_cols is None else mask_cols
    V.loc[V.index >= origin, cols] = np.nan
    return V


def known_at(V: pd.DataFrame, col: str, origin, lag: int | None = None) -> float:
    """Value of `col` in month origin - 1 - lag (lag defaults to KNOWN_LAG.get(col, 0)); NaN if missing."""
    lag = KNOWN_LAG.get(col, 0) if lag is None else int(lag)
    t = pd.Timestamp(origin) - pd.DateOffset(months=1 + lag)
    try:
        return float(V.at[t, col])
    except KeyError:
        return float("nan")


def log_price_returns(P: pd.DataFrame, assets) -> pd.DataFrame:
    """Monthly log1p(p_asset)."""
    return np.log1p(P[[f"p_{a}" for a in assets]].astype(float)).set_axis(list(assets), axis=1)


def quarter_starts(start, end) -> pd.DatetimeIndex:
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    return idx[np.isin(idx.month, (1, 4, 7, 10))]


def add_months(t, k: int) -> pd.Timestamp:
    return pd.Timestamp(t) + pd.DateOffset(months=int(k))


def months_between(a, b) -> int:
    """Inclusive month count from a to b."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (b.year - a.year) * 12 + b.month - a.month + 1


# ---------------------------------------------------------------- the engine's residual draw
class _DrawDB:
    """Just what residuals.prior_draw reads from a MonthDB: len, prior(), rare (config.RARE_MONTHS weights)."""

    def __init__(self, dates):
        self.dates = pd.DatetimeIndex(dates)
        self.rare = np.ones(len(self.dates))
        for r in config.RARE_MONTHS:
            m = (self.dates >= pd.Timestamp(r["start"])) & (self.dates <= pd.Timestamp(r["end"]))
            self.rare[m] *= r["weight"]

    def __len__(self) -> int:
        return len(self.dates)

    def prior(self, half_life_years=None) -> np.ndarray:
        w = self.rare.copy()
        return w / w.sum()


def engine_residual_draw(dates, months: int, n_paths: int, seed: int) -> np.ndarray:
    """(n_paths, months) row positions into `dates` from the engine's unconditional block bootstrap
    (engine.residuals.prior_draw: 6-month blocks, prior block starts, rare months down-weighted)."""
    return _eres.prior_draw(_DrawDB(dates), [int(months)], int(n_paths), BLOCK, int(seed))


# ---------------------------------------------------------------- models
class Fitted:
    """Base class of what a model_fn returns. Subclasses implement predict_path; set `resid` for the default band."""
    name: str = "model"
    assets: list = []
    train_end = None
    resid: pd.DataFrame | None = None

    def predict_path(self, vis: pd.DataFrame, origin, months: int) -> pd.DataFrame:
        raise NotImplementedError

    def simulate_paths(self, vis: pd.DataFrame, origin, months: int, n_paths: int, seed: int):
        if self.resid is None:
            return None
        path = self.predict_path(vis, origin, months).reindex(columns=self.assets).to_numpy(float)
        R = self.resid.reindex(columns=self.assets)
        R = R[np.isfinite(R.to_numpy(float)).all(axis=1)]
        idx = engine_residual_draw(R.index, months, n_paths, seed)
        return path[None, :, :] + R.to_numpy(float)[idx]


@dataclass(frozen=True)
class LinearSpec:
    """Per-asset OLS of the monthly simple price return on same-month panel columns (with intercept).

    common: columns every equation uses; extra: {asset: extra columns}; start: {asset: first fit month}; assets:
    equations fitted. Training months = the joint mask (every feature of every equation and every asset's return)."""
    name: str
    common: tuple
    extra: dict = field(default_factory=dict)
    start: dict = field(default_factory=dict)
    assets: tuple = tuple(MODEL_ASSETS)

    def features(self, asset: str) -> list[str]:
        return list(self.common) + list(self.extra.get(asset, ()))

    def all_features(self) -> list[str]:
        out = list(self.common)
        for v in self.extra.values():
            out += [c for c in v if c not in out]
        return out

    def with_assets(self, assets) -> "LinearSpec":
        return LinearSpec(self.name, self.common, dict(self.extra), dict(self.start), tuple(assets))


def spec_from_model(spec: ModelSpec, name: str = "F") -> LinearSpec:
    """The harness LinearSpec of a project ModelSpec (linear, contemporaneous)."""
    if spec.kind != "linear" or spec.lags or spec.regime:
        raise ValueError(f"{spec.name}: only a linear contemporaneous spec maps onto the harness")
    common = tuple(f"f_{f}" for f in spec.factors) + tuple(spec.shock_cols)
    extra = {a: tuple(f"f_{f}" for f in ids if f not in spec.factors) for a, ids in spec.asset_factors}
    return LinearSpec(name, common, {a: v for a, v in extra.items() if v}, dict(spec.asset_start),
                      tuple(MODEL_ASSETS))


def _selected_f() -> LinearSpec:
    from ..qmodel.assets import l0_spec
    return spec_from_model(l0_spec(), "F")


CORE8 = ("f_oil", "f_ip", "f_cpi_goods", "f_cpi_services", "f_ff", "f_y10", "f_usd", "f_unrate")
B0 = LinearSpec("B0", tuple(f"f_{f}" for f in config.CORE_FACTOR_IDS))          # config's core factors
F = _selected_f()                                                                # model_selection.json


class LinearFitted(Fitted):
    def __init__(self, spec: LinearSpec, train: pd.DataFrame, train_end):
        self.spec, self.name = spec, spec.name
        self.assets = list(spec.assets)
        self.train_end = pd.Timestamp(train_end)
        feats = spec.all_features()
        T = train.loc[config.MODEL_START:self.train_end]
        X, Y = T[feats], T[[f"p_{a}" for a in self.assets]]
        ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
        rows = T.index[ok]
        if len(rows) < 3 * (len(feats) + 2):
            raise ValueError(f"{spec.name}: only {len(rows)} training months to {self.train_end:%Y-%m}")
        self.rows = rows
        self.coef, self.n_fit, self.c_log, self.sd = {}, {}, {}, {}
        R = pd.DataFrame(np.nan, index=rows, columns=self.assets)
        for a in self.assets:
            fa = spec.features(a)
            ra = rows[rows >= pd.Timestamp(spec.start.get(a, config.MODEL_START))]
            A = np.column_stack([np.ones(len(ra)), X.loc[ra, fa].to_numpy(float)])
            y = Y.loc[ra, f"p_{a}"].to_numpy(float)
            b = np.linalg.lstsq(A, y, rcond=None)[0]
            self.coef[a] = (fa, b)
            self.n_fit[a] = len(ra)
            ph = np.column_stack([np.ones(len(rows)), X.loc[rows, fa].to_numpy(float)]) @ b
            R[a] = Y.loc[rows, f"p_{a}"].to_numpy(float) - ph
            phr = A @ b
            self.c_log[a] = float(np.mean(np.log1p(y) - np.log1p(np.maximum(phr, -0.95))))
            self.sd[a] = float(np.std(y - phr))
        self.R_simple = R
        P_ = Y.loc[rows].set_axis(self.assets, axis=1)
        PH = P_ - R
        self.resid = np.log1p(P_) - (np.log1p(np.maximum(PH, -0.95)) + pd.Series(self.c_log))

    def predict_simple(self, vis: pd.DataFrame, origin, months: int) -> pd.DataFrame:
        t0 = pd.Timestamp(origin)
        rows = pd.date_range(t0, add_months(t0, months - 1), freq="MS")
        V = vis.reindex(rows)
        out = pd.DataFrame(np.nan, index=rows, columns=self.assets)
        for a in self.assets:
            fa, b = self.coef[a]
            x = V[fa].to_numpy(float)
            out[a] = b[0] + x @ b[1:]
        return out

    def predict_path(self, vis, origin, months):
        ph = self.predict_simple(vis, origin, months)
        return np.log1p(ph.clip(lower=-0.95)) + pd.Series(self.c_log)

    def simulate_paths(self, vis, origin, months, n_paths, seed):
        ph = self.predict_simple(vis, origin, months).to_numpy(float)          # (months, A)
        idx = engine_residual_draw(self.R_simple.index, months, n_paths, seed)  # engine's own draw
        e = self.R_simple.to_numpy(float)[idx]                                  # (n, months, A)
        return np.log1p(np.maximum(ph[None, :, :] + e, -0.95))


def linear_model(spec: LinearSpec):
    """model_fn for a LinearSpec."""
    def fn(train: pd.DataFrame, train_end) -> LinearFitted:
        return LinearFitted(spec, train, train_end)
    fn.__name__ = spec.name
    fn.model_name = spec.name
    return fn


def base_model():
    return linear_model(B0)


class DriftFitted(Fitted):
    """The benchmark as a model: training-mean monthly log price return (BENCH_START..train_end). R² = 0 by
    construction; its band is the engine block draw of the demeaned training log returns."""
    name = "drift"

    def __init__(self, train: pd.DataFrame, train_end, assets=None):
        self.assets = list(assets or SCORE_ASSETS)
        self.train_end = pd.Timestamp(train_end)
        lr = log_price_returns(train, self.assets).loc[BENCH_START:self.train_end].dropna()
        self.mu = lr.mean()
        self.resid = lr - self.mu

    def predict_path(self, vis, origin, months):
        idx = pd.date_range(pd.Timestamp(origin), periods=int(months), freq="MS")
        return pd.DataFrame(np.tile(self.mu.to_numpy(), (int(months), 1)), index=idx, columns=self.assets)


def drift_model(assets=None):
    def fn(train, train_end):
        return DriftFitted(train, train_end, assets)
    fn.model_name = "drift"
    return fn


# ---------------------------------------------------------------- quarterly models in the monthly harness
def _quarter_window(origin, months: int) -> tuple[pd.Period, int]:
    o = pd.Timestamp(origin)
    if o.month not in (1, 4, 7, 10) or int(months) % 3:
        raise ValueError("quarterly model: origins must be quarter starts and windows whole quarters")
    return pd.Period(o, "Q"), int(months) // 3


def _in_last_months(q: np.ndarray, origin, months: int, assets) -> np.ndarray:
    """(..., H, A) quarterly values -> (..., months, A) with each quarter in its last month, zeros elsewhere."""
    q = np.asarray(q, float)
    out = np.zeros(q.shape[:-2] + (int(months), len(assets)))
    out[..., 2::3, :] = q
    return out


class QuarterlyFitted(Fitted):
    """The quarterly asset layer (qmodel.assets.AssetLayer) as a harness model (as the research's
    qplugin._HarnessFitted): oracle macro = the realized quarterly fields of the window's quarters (macro columns of
    the quarterly panel only); the origin's start-of-quarter states; each quarter's log price return in the quarter's
    last month. predict_path is the systematic part (drift + factors), simulate_paths adds the layer's own residual
    draw (2-quarter blocks, volatility-scaled, optional parameter refits)."""

    def __init__(self, layer, Q: pd.DataFrame, cma: str = "historical", block: int | None = None,
                 params: bool | None = None, name: str = "L0q"):
        self.layer, self.Q, self.cma, self.block, self.params = layer, Q, cma, block, params
        self.name = name
        self.assets = list(cq.ASSETS)
        self.train_end = pd.Period(layer.train_end, "Q").end_time.to_period("M").to_timestamp()
        self.n_fit = {a: n for a, n in layer.n_fit_months.items()}

    def _inputs(self, origin, months):
        from ..qmodel.assets import oracle_inputs
        q0, H = _quarter_window(origin, months)
        qs = pd.period_range(q0, periods=H, freq="Q")
        leads = any(cq.TA_CANDIDATES[c]["kind"] == "lead" for c in self.layer.candidates)
        return q0, H, oracle_inputs(self.Q, qs, leads=leads)

    def _systematic(self, origin, months):
        """(q0, H, oracle inputs, start state, (H, 13) systematic log price returns). Candidate terms may read the
        states a path rolls (s_vol, s_pd): with candidates, the mean of the systematic part over simulated paths."""
        q0, H, X = self._inputs(origin, months)
        st = self.layer.start_state(self.Q, q0)
        if self.layer.candidates:
            sim = self.layer.simulate(X, st, n_paths=200, seed=0, cma=self.cma, params=False)
            mean = sim.systematic.mean(axis=0)
        else:
            mean = self.layer.mean(X, cma=self.cma).reindex(columns=self.assets).to_numpy(float)
        return q0, H, X, st, mean

    def predict_path(self, vis, origin, months):
        mean = self._systematic(origin, months)[-1]
        idx = pd.date_range(pd.Timestamp(origin), periods=int(months), freq="MS")
        return pd.DataFrame(_in_last_months(mean, origin, months, self.assets), index=idx, columns=self.assets)

    def predict_mult(self, vis, origin, months) -> float:
        """The window's predicted Δlog CAPE multiple: systematic us_equity price minus the y_fund equation's mean
        (the inferred mode, SIM_DESIGN §6.3), from the same oracle inputs."""
        from ..qmodel import valuation as val
        q0, H, X, st, mean = self._systematic(origin, months)
        price = mean[:, self.assets.index("us_equity")]
        reg = val.fund_regressors(X)
        lags = list(st["y_fund_lags"])
        fund = []
        for k in range(H):
            f = float(self.layer.valuation.fund.predict({c: v[k] for c, v in reg.items()}, lags[0]))
            fund.append(f)
            lags = [lags[1], f]
        return float(np.sum(price) - np.sum(fund))

    def simulate_paths(self, vis, origin, months, n_paths, seed):
        q0, H, X = self._inputs(origin, months)
        st = self.layer.start_state(self.Q, q0)
        sim = self.layer.simulate(X, st, n_paths=n_paths, seed=seed, cma=self.cma, block=self.block,
                                  params=self.params)
        return _in_last_months(sim.price, origin, months, self.assets)


class QuarterlyDrift(Fitted):
    """The quarterly benchmark: the training-mean quarterly log price return (quarterly panel, 1973Q1..train end),
    each quarter's value in its last month. Scored by the harness its R² is 0 (the drift-only check)."""
    name = "drift_q"

    def __init__(self, Q: pd.DataFrame, train_end, assets=None):
        self.assets = list(assets or SCORE_ASSETS)
        te = pd.Period(pd.Timestamp(train_end), "Q")
        self.train_end = pd.Timestamp(train_end)
        Y = Q.loc[pd.Period(cq.FIRST_QUARTER, "Q"):te, [f"y_{a}" for a in self.assets]]
        self.mu = Y.mean().set_axis(self.assets)

    def predict_path(self, vis, origin, months):
        q0, H = _quarter_window(origin, months)
        q = np.tile(self.mu.to_numpy(float), (H, 1))
        idx = pd.date_range(pd.Timestamp(origin), periods=int(months), freq="MS")
        return pd.DataFrame(_in_last_months(q, origin, months, self.assets), index=idx, columns=self.assets)


def quarterly_model_fn(Q: pd.DataFrame | None = None, *, cma: str = "historical", n_boot: int = cq.PARAM_BOOT,
                       candidates=cq.TA_DEFAULT, block: int | None = None, params: bool | None = None,
                       seed: int = 0, name: str | None = None):
    """model_fn for conditional_paths: the asset layer fitted at each train end (a quarter-end month) on the train
    panel and the quarterly panel up to that quarter. cma: "historical" (the walk-forward default) or "no_rerate";
    n_boot: parameter refits of F for the band (0: residuals only, the research L0's band); candidates: T-A
    candidates (off by default)."""
    from ..qmodel import panel as qpanel
    from ..qmodel.assets import AssetLayer
    Qq = qpanel.load_panel() if Q is None else Q
    label = name or ("L0q" if not candidates else "L0q+" + "+".join(candidates)) + ("" if cma == "historical"
                                                                                      else f"[{cma}]")

    def fn(train, train_end):
        te = pd.Timestamp(train_end)
        if te.month % 3:
            raise ValueError("the quarterly asset layer needs a quarter-end train end")
        layer = AssetLayer.fit(pd.Period(te, "Q"), panel=train, Q=Qq, n_boot=n_boot, candidates=candidates,
                               seed=seed)
        return QuarterlyFitted(layer, Qq, cma=cma, block=block, params=params, name=label)
    fn.model_name = label
    return fn


def quarterly_drift_fn(Q: pd.DataFrame | None = None, assets=None):
    from ..qmodel import panel as qpanel
    Qq = qpanel.load_panel() if Q is None else Q

    def fn(train, train_end):
        return QuarterlyDrift(Qq, train_end, assets)
    fn.model_name = "drift_q"
    return fn


# ---------------------------------------------------------------- conditional paths
def _seed(seed: int, origin: pd.Timestamp, hq: int) -> int:
    return int(seed) * 1_000_003 + origin.year * 100 + origin.month + 10_000 * int(hq)


def crps_samples(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample CRPS per column: E|X - y| - 0.5 E|X - X'| (draws (n, ...), y (...)); the sorted-sample formula."""
    n = draws.shape[0]
    xs = np.sort(draws, axis=0)
    t1 = np.mean(np.abs(xs - y[None]), axis=0)
    i = np.arange(1, n + 1).reshape((n,) + (1,) * (draws.ndim - 1))
    t2 = np.sum((2 * i - n - 1) * xs, axis=0) / (n * n)
    return t1 - t2


def _quarterly_demeaned(LR: pd.DataFrame, bench_start, train_end) -> np.ndarray:
    """Demeaned training quarterly log returns (calendar quarters with all three months, bench_start..train_end)."""
    W = LR.loc[bench_start:train_end]
    g = W.groupby(W.index.to_period("Q"))
    Qs = g.sum(min_count=3)[g.count().min(axis=1) == 3].dropna()
    return (Qs - Qs.mean()).to_numpy(float)


def bench_draws(R: np.ndarray, mean: np.ndarray, hq: int, n: int, seed: int) -> np.ndarray:
    """The drift benchmark's distribution of an hq-quarter window (the research qmodel's CRPS benchmark): the
    training-mean drift plus the sum of hq independent draws of demeaned training quarterly returns. (n, assets)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(R), size=(int(n), int(hq)))
    return np.asarray(mean, float)[None, :] + R[idx].sum(axis=1)


def conditional_paths(model_fn, panel: pd.DataFrame | None = None, protocol: str = "split",
                      horizons_q=HORIZONS_Q, assets=None, name: str | None = None, band: bool = True,
                      n_paths: int = N_PATHS, seed: int = SEED, splits: dict | None = None,
                      roll_first: str = ROLL_FIRST, roll_refit: int = ROLL_REFIT, roll_last=None,
                      inside: bool = True, bench_start: str = BENCH_START, crps: bool = True,
                      verbose: bool = False) -> pd.DataFrame:
    """Oracle-macro cumulative log price returns, model vs realized, per origin x horizon x asset.

    Returns a long DataFrame: model, protocol, group, train_end, origin, end, h_q, months, asset, y (realized sum of
    log1p(p) over the window), yhat, bench (months x training-mean log return), q10, q50, q90, pit (share of simulated
    window returns below y), crps and crps_bench (the model's band and the drift's, when band and crps), yhat_mult
    (the window's predicted Δlog CAPE multiple, for models with predict_mult; us_equity rows), n_fit."""
    P = load_panel() if panel is None else panel
    assets = list(assets or SCORE_ASSETS)
    name = name or getattr(model_fn, "model_name", getattr(model_fn, "__name__", "model"))
    mask_cols = asset_derived_columns(P)
    LR = log_price_returns(P, assets)
    last = P.index.max()
    jobs = []
    if protocol == "split":
        for g, (tr, t0, t1) in (splits or SPLITS).items():
            jobs.append((g, pd.Timestamp(tr), min(pd.Timestamp(t1), last), quarter_starts(t0, t1)))
    elif protocol == "rolling":
        origins = quarter_starts(roll_first, roll_last or last)
        first_end = add_months(roll_first, -1)
        by_end: dict = {}
        for o in origins:
            k = (months_between(first_end, add_months(o, -1)) - 1) // roll_refit
            by_end.setdefault(add_months(first_end, k * roll_refit), []).append(o)
        for tr, os_ in by_end.items():
            jobs.append(("rolling", tr, last, pd.DatetimeIndex(os_)))
    else:
        raise ValueError("protocol must be 'split' or 'rolling'")

    rows = []
    for g, tr, span_end, origins in jobs:
        train = P.loc[:tr].copy()
        fitted = model_fn(train, tr)
        mu = LR.loc[bench_start:tr].mean()
        R_bench = _quarterly_demeaned(LR, bench_start, tr) if (band and crps) else None
        n_fit = getattr(fitted, "n_fit", {})
        for o in origins:
            for hq in horizons_q:
                m = 3 * int(hq)
                end = add_months(o, m - 1)
                if end > last or (inside and end > span_end):
                    continue
                vis = visible_panel(P, o, end, mask_cols)
                path = fitted.predict_path(vis, o, m).reindex(columns=assets)
                yhat = path.sum(min_count=m) if len(path) == m else pd.Series(np.nan, index=assets)
                yhat[path.isna().any()] = np.nan
                win = LR.loc[o:end]
                y = win.sum(min_count=m)
                y[win.isna().any()] = np.nan
                q = np.full((3, len(assets)), np.nan)
                pit = np.full(len(assets), np.nan)
                cr = np.full(len(assets), np.nan)
                crb = np.full(len(assets), np.nan)
                yv = y.to_numpy(float)
                if band:
                    sims = fitted.simulate_paths(vis, o, m, n_paths, _seed(seed, o, hq))
                    if sims is not None:
                        fa = list(getattr(fitted, "assets", assets))
                        tot = np.asarray(sims, float).sum(axis=1)
                        tot = pd.DataFrame(tot, columns=fa).reindex(columns=assets).to_numpy(float)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            q = np.nanquantile(tot, [BAND[0], 0.5, BAND[1]], axis=0)
                        pit = np.where(np.isfinite(yv) & np.isfinite(q[1]), np.mean(tot < yv[None, :], axis=0), np.nan)
                        if R_bench is not None and len(R_bench):
                            ok = np.isfinite(yv) & np.isfinite(tot).all(axis=0)
                            cr = np.where(ok, crps_samples(np.nan_to_num(tot), np.nan_to_num(yv)), np.nan)
                            bs = bench_draws(R_bench, m * mu.to_numpy(float), hq, n_paths, _seed(seed + 1, o, hq))
                            crb = np.where(ok, crps_samples(bs, np.nan_to_num(yv)), np.nan)
                ym = float("nan")
                if hasattr(fitted, "predict_mult") and "us_equity" in assets:
                    ym = float(fitted.predict_mult(vis, o, m))
                for j, a in enumerate(assets):
                    rows.append((name, protocol, g, tr, o, end, int(hq), m, a, float(y[a]), float(yhat[a]),
                                 float(m * mu[a]), q[0, j], q[1, j], q[2, j], pit[j], cr[j], crb[j],
                                 ym if a == "us_equity" else float("nan"), n_fit.get(a, np.nan)))
        if verbose:
            print(f"  {name} {protocol} {g} train_end {tr:%Y-%m}: {len(origins)} origins", flush=True)
    cols = ["model", "protocol", "group", "train_end", "origin", "end", "h_q", "months", "asset", "y", "yhat",
            "bench", "q10", "q50", "q90", "pit", "crps", "crps_bench", "yhat_mult", "n_fit"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- metrics
def _r2(y, yh, b) -> float:
    sst = float(np.sum((y - b) ** 2))
    return 1.0 - float(np.sum((y - yh) ** 2)) / sst if sst > 0 else float("nan")


def _block_idx(n: int, L: int, n_boot: int, rng) -> np.ndarray:
    nb = int(math.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(n_boot, nb))
    return (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, -1)[:, :n]


def _stats(y, yh, b, q10, q90, pit, cr=None, crb=None) -> dict:
    n = len(y)
    e = y - yh
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = float(np.corrcoef(y, yh)[0, 1]) if n >= 3 and np.std(yh) > 0 and np.std(y) > 0 else float("nan")
    okb = np.isfinite(q10) & np.isfinite(q90)
    out = {"n": n, "r2": _r2(y, yh, b), "corr": corr, "mae": float(np.mean(np.abs(e))),
           "mae_bench": float(np.mean(np.abs(y - b))), "bias": float(np.mean(yh - y)),
           "hit": float(np.mean(np.sign(yh) == np.sign(y))),
           "hit_excess": float(np.mean(np.sign(yh - b) == np.sign(y - b))),
           "sd_y": float(np.std(y)), "sd_yhat": float(np.std(yh)),
           "cov80": float(np.mean((y[okb] >= q10[okb]) & (y[okb] <= q90[okb]))) if okb.any() else float("nan"),
           "below10": float(np.mean(y[okb] < q10[okb])) if okb.any() else float("nan"),
           "above90": float(np.mean(y[okb] > q90[okb])) if okb.any() else float("nan"),
           "pit_mad": float(np.nanmean(np.abs(pit - 0.5))) if np.isfinite(pit).any() else float("nan")}
    out["crps"] = out["crpss"] = float("nan")
    if cr is not None and crb is not None:
        okc = np.isfinite(cr) & np.isfinite(crb)
        if okc.any():
            out["crps"] = float(np.mean(cr[okc]))
            out["crpss"] = 1.0 - float(np.sum(cr[okc]) / np.sum(crb[okc])) if np.sum(crb[okc]) > 0 else float("nan")
    return out


def _group_rows(df: pd.DataFrame, protocol: str, spans: dict | None):
    """Yield (group label, span start, span end, rows) for one model x protocol."""
    if protocol == "split":
        sp = spans or SPLITS
        for g, sub in df.groupby("group"):
            t0, t1 = (sp[g][1], sp[g][2]) if g in sp and len(sp[g]) == 3 else (sub.origin.min(), sub.end.max())
            yield g, pd.Timestamp(t0), pd.Timestamp(t1), sub
    else:
        for g, (t0, t1) in (spans or ROLL_SPANS).items():
            t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
            yield g, t0, t1, df[(df.origin >= t0) & (df.end <= t1)]


def metrics(paths: pd.DataFrame, spans: dict | None = None, n_boot: int = 1000, seed: int = 0,
            score_assets=None) -> pd.DataFrame:
    """Per model x protocol x group x asset x horizon statistics (see the module docstring), plus asset 'mean10'
    (mean over score_assets, default SCORE_ASSETS, with a joint block-bootstrap interval for the mean R²)."""
    score_assets = list(score_assets or SCORE_ASSETS)
    rng = np.random.default_rng(seed)
    out = []
    for (model, protocol), df in paths.groupby(["model", "protocol"], sort=False):
        for g, t0, t1, sub in _group_rows(df, protocol, spans):
            span = months_between(t0, t1)
            for hq, sh in sub.groupby("h_q"):
                m = 3 * int(hq)
                n_indep = span // m
                L = max(int(hq), 4)
                per = {}
                for a, sa in sh.groupby("asset"):
                    sa = sa.sort_values("origin")
                    sa = sa[np.isfinite(sa.y) & np.isfinite(sa.yhat)]
                    if len(sa) < 3:
                        continue
                    y, yh, b = sa.y.to_numpy(), sa.yhat.to_numpy(), sa.bench.to_numpy()
                    cr = sa["crps"].to_numpy(float) if "crps" in sa else None
                    crb = sa["crps_bench"].to_numpy(float) if "crps_bench" in sa else None
                    st = _stats(y, yh, b, sa.q10.to_numpy(), sa.q90.to_numpy(), sa.pit.to_numpy(), cr, crb)
                    sub_no = sa.iloc[::int(hq)]
                    st["r2_nonoverlap"] = (_r2(sub_no.y.to_numpy(), sub_no.yhat.to_numpy(), sub_no.bench.to_numpy())
                                           if len(sub_no) >= 3 else float("nan"))
                    st["n_nonoverlap"] = len(sub_no)
                    lo = hi = float("nan")
                    Lb = min(L, len(sa) // 2)
                    if n_indep >= 3 and Lb >= 1:
                        ix = _block_idx(len(sa), Lb, n_boot, rng)
                        sse = np.sum((y - yh)[ix] ** 2, axis=1)
                        sst = np.sum((y - b)[ix] ** 2, axis=1)
                        r2b = 1 - sse / np.where(sst > 0, sst, np.nan)
                        lo, hi = np.nanpercentile(r2b, [5, 95])
                    st.update({"r2_lo90": float(lo), "r2_hi90": float(hi)})
                    per[a] = sa
                    out.append({"model": model, "protocol": protocol, "group": g, "asset": a, "h_q": int(hq),
                                "months": m, "n_indep": n_indep, "span_months": span, **st})
                have = [a for a in score_assets if a in per]
                if len(have) == len(score_assets):
                    piv = {k: pd.concat([per[a].set_index("origin")[k].rename(a) for a in have], axis=1)
                           for k in ("y", "yhat", "bench")}
                    common = piv["y"].dropna().index.intersection(piv["yhat"].dropna().index)
                    Y, YH, BB = (piv[k].loc[common].to_numpy() for k in ("y", "yhat", "bench"))
                    mean_rows = [r for r in out if r["model"] == model and r["protocol"] == protocol and
                                 r["group"] == g and r["h_q"] == int(hq) and r["asset"] in have]
                    agg = {k: float(np.mean([r[k] for r in mean_rows])) for k in
                           ("r2", "corr", "mae", "mae_bench", "bias", "hit", "hit_excess", "cov80", "below10",
                            "above90", "pit_mad", "r2_nonoverlap", "sd_y", "sd_yhat", "crps", "crpss")}
                    agg["r2_median"] = float(np.median([r["r2"] for r in mean_rows]))
                    lo = hi = float("nan")
                    Lb = min(L, len(common) // 2)
                    if n_indep >= 3 and Lb >= 1 and len(common) >= 3:
                        ix = _block_idx(len(common), Lb, n_boot, rng)
                        sse = np.sum((Y - YH)[ix] ** 2, axis=1)
                        sst = np.sum((Y - BB)[ix] ** 2, axis=1)
                        r2b = np.nanmean(1 - sse / np.where(sst > 0, sst, np.nan), axis=1)
                        lo, hi = np.nanpercentile(r2b, [5, 95])
                    out.append({"model": model, "protocol": protocol, "group": g, "asset": "mean10",
                                "h_q": int(hq), "months": m, "n_indep": n_indep, "span_months": span,
                                "n": len(common), "r2_lo90": float(lo), "r2_hi90": float(hi), **agg})
    return pd.DataFrame(out)


def mean_table(m: pd.DataFrame, value: str = "r2", asset: str = "mean10") -> pd.DataFrame:
    """value for one asset (default the 10-asset mean): rows model x protocol x group, columns horizon (quarters)."""
    return m[m.asset == asset].pivot_table(index=["model", "protocol", "group"], columns="h_q", values=value,
                                           aggfunc="first")


def paired_diff(paths: pd.DataFrame, model_a: str, model_b: str, *, protocol: str = "split", spans=None,
                horizons_q=None, n_boot: int = 2000, block: int = 4, seed: int = 0, score_assets=None) -> pd.DataFrame:
    """mean10 R² of model_b minus model_a on their common origins, per group x horizon, with a 90% moving-block
    bootstrap interval (blocks of `block` windows, the same resampled origins for both models and every asset):
    the T-A rule's "ΔR² against L0" (SIM_DESIGN §4.2)."""
    score_assets = list(score_assets or SCORE_ASSETS)
    rng = np.random.default_rng(seed)
    out = []
    A_all = paths[(paths.model == model_a) & (paths.protocol == protocol)]
    B_all = paths[(paths.model == model_b) & (paths.protocol == protocol)]
    B_groups = {g: sub for g, _, _, sub in _group_rows(B_all, protocol, spans)}
    for g, t0, t1, A in _group_rows(A_all, protocol, spans):
        B = B_groups.get(g)
        if B is None or A.empty or B.empty:
            continue
        for hq in sorted(set(A.h_q) & set(B.h_q)):
            if horizons_q is not None and hq not in horizons_q:
                continue
            pa = A[(A.h_q == hq) & A.asset.isin(score_assets)].pivot_table(index="origin", columns="asset",
                                                                            values=["y", "yhat", "bench"])
            pb = B[(B.h_q == hq) & B.asset.isin(score_assets)].pivot_table(index="origin", columns="asset",
                                                                            values=["yhat"])
            common = pa.dropna().index.intersection(pb.dropna().index)
            if len(common) < 3:
                continue
            Y, BB = pa["y"].loc[common, score_assets].to_numpy(), pa["bench"].loc[common, score_assets].to_numpy()
            YA, YB = pa["yhat"].loc[common, score_assets].to_numpy(), pb["yhat"].loc[common, score_assets].to_numpy()

            def r2m(ix):
                sst = np.sum((Y[ix] - BB[ix]) ** 2, axis=-2)
                return (np.nanmean(1 - np.sum((Y[ix] - YB[ix]) ** 2, axis=-2) / sst, axis=-1)
                        - np.nanmean(1 - np.sum((Y[ix] - YA[ix]) ** 2, axis=-2) / sst, axis=-1))
            d = float(r2m(np.arange(len(common))))
            L = min(int(block), len(common) // 2)
            ix = _block_idx(len(common), max(L, 1), n_boot, rng)
            db = r2m(ix)
            lo, hi = np.nanpercentile(db, [5, 95])
            out.append({"protocol": protocol, "group": g, "h_q": int(hq), "model_a": model_a, "model_b": model_b,
                        "n": len(common), "n_indep": months_between(t0, t1) // (3 * int(hq)), "diff_r2": d,
                        "lo90": float(lo), "hi90": float(hi)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- equity decomposition
MEASURES = {
    "pd": "Δlog P = Δlog D12 + Δlog(P/D12)  (own index: CRSP / French / Nareit; nasdaq uses BusEq's dividend yield)",
    "pe": "Δlog P(CRSP) = Δlog E(S&P, trailing 12m) + [Δlog P(CRSP) - Δlog E]  (multiple = CRSP price over S&P EPS)",
    "cape": "Δlog P(CRSP) = Δlog(CPI x E10 real, S&P) + [rest]  (the CAPE multiple: SIM_DESIGN §6.1)",
    "sp_pe": "Δlog P(S&P, monthly average) = Δlog E + Δlog(P/E)  (Shiller only)",
    "sp_cape": "Δlog P(S&P) = Δlog CPI + Δlog E10 real + Δlog CAPE  (Shiller only)",
}


def decomposition_levels(P: pd.DataFrame, asset: str, measure: str = "pd") -> pd.DataFrame:
    """Monthly log levels (lp, lf, lm) with lp = lf + lm, for decompose_windows."""
    if measure == "pd":
        lp, lf = P[f"v_lp_{asset}"], P[f"v_ld12_{asset}"]
    elif measure == "pe":
        lp, lf = P[f"v_lp_{asset}"], P["v_sp_le"]
    elif measure == "cape":
        lp, lf = P[f"v_lp_{asset}"], P["v_sp_lcpi"] + P["v_sp_le10r"]
    elif measure == "sp_pe":
        lp, lf = P["v_sp_lp"], P["v_sp_le"]
    elif measure == "sp_cape":
        lp, lf = P["v_sp_lp"], P["v_sp_lcpi"] + P["v_sp_le10r"]
    else:
        raise ValueError(f"unknown measure {measure!r}; one of {list(MEASURES)}")
    return pd.DataFrame({"lp": lp, "lf": lf, "lm": lp - lf})


def decompose_windows(P: pd.DataFrame, asset: str, origins, months: int, measure: str = "pd") -> pd.DataFrame:
    """Per origin: dp = Δlog P, dfund = Δlog fundamentals, dmult = Δlog multiple over origin..origin+months-1
    (changes from month origin-1 to the window's last month; dp = dfund + dmult exactly)."""
    L = decomposition_levels(P, asset, measure)
    rows = []
    for o in pd.DatetimeIndex(origins):
        t0, t1 = add_months(o, -1), add_months(o, months - 1)
        if t0 not in L.index or t1 not in L.index:
            continue
        a, b = L.loc[t0], L.loc[t1]
        rows.append({"origin": o, "end": t1, "dp": b.lp - a.lp, "dfund": b.lf - a.lf, "dmult": b.lm - a.lm})
    return pd.DataFrame(rows)


def _cov(a, b) -> float:
    return float(np.cov(a, b, ddof=1)[0, 1]) if len(a) >= 3 else float("nan")


def _constant(a: np.ndarray) -> bool:
    """True when a series has no variation beyond rounding (a constant forecast)."""
    a = np.asarray(a, float)
    return bool(np.ptp(a) <= 1e-12 * (1.0 + np.max(np.abs(a)))) if len(a) else True


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or _constant(a) or _constant(b):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def multiple_skill(pred, dm) -> dict:
    """The placebo-corrected statements of §6.6 for predicted vs realized multiple changes over windows: (a) corr with
    n, (b) the slope of realized on predicted (the share of the multiple change the prediction carries) and R² against
    a zero change. A constant prediction has no corr or slope (NaN), only its R² against zero."""
    pred, dm = np.asarray(pred, float), np.asarray(dm, float)
    flat = len(pred) < 3 or _constant(pred)
    s0 = float(np.sum(dm ** 2))
    return {"n": int(len(dm)), "corr_pred_dm": _corr(pred, dm),
            "slope_dm_on_pred": float("nan") if flat else _cov(dm, pred) / float(np.var(pred, ddof=1)),
            "r2_dm_vs_zero": 1.0 - float(np.sum((dm - pred) ** 2)) / s0 if s0 > 0 else float("nan")}


def variance_anatomy(P: pd.DataFrame, paths: pd.DataFrame | None = None, assets=None, measures=None,
                     spans: dict | None = None, horizons_q=HORIZONS_Q, protocol_pref=("split", "rolling"),
                     model: str | None = None) -> pd.DataFrame:
    """Share of realized window price-return variance from the multiple vs fundamentals, per asset x measure x decade
    x horizon (windows at quarter starts inside the decade; overlapping for h > 1), and, given model `paths`, what the
    model's oracle-macro prediction tracks.

    Columns as in the research harness (share_m, vr_m, r2_orc_m / _f, corr_yhat_*, err_share_m, err_r2_m = corr(miss,
    dmult)², ...; kept for export) plus the placebo-corrected statements (§6.6): for the model, pred_dm = its predicted
    multiple change (yhat_mult when the model provides one, else yhat - h·mu_f: the price forecast with fundamentals at
    their training drift), corr_pred_dm, slope_dm_on_pred and r2_dm_vs_zero (multiple_skill); for the placebo, a
    constant forecast (the training drift), placebo_r2_dm_vs_zero and placebo_err_r2_m = corr(dp, dm)², the value the
    old statistic gives any forecast that does not track prices (0.83-1.00 for the CAPE multiple)."""
    assets = list(assets or EQUITY_LIKE)
    spans = spans or DECADES
    rows = []
    for a in assets:
        ms = measures or (["pd", "pe", "cape", "sp_pe", "sp_cape"] if a == "us_equity" else ["pd"])
        for meas in ms:
            if meas == "pd" and f"v_lp_{a}" not in P.columns:
                continue
            Lv = decomposition_levels(P, a, meas).diff()
            for g, (s0, s1) in spans.items():
                tr = Lv.loc[config.MODEL_START:add_months(s0, -1)].dropna()
                mu = tr.mean() if len(tr) >= 60 else pd.Series(np.nan, index=Lv.columns)
                for hq in horizons_q:
                    m = 3 * int(hq)
                    os_ = [o for o in quarter_starts(s0, s1) if add_months(o, m - 1) <= pd.Timestamp(s1)]
                    D = decompose_windows(P, a, os_, m, meas).dropna()
                    if len(D) < 3:
                        continue
                    dp, df_, dm = D.dp.to_numpy(), D.dfund.to_numpy(), D.dmult.to_numpy()
                    vp = float(np.var(dp, ddof=1))
                    bench = m * float(mu["lp"])
                    rec = {"asset": a, "measure": meas, "group": g, "h_q": int(hq), "n": len(D),
                           "n_indep": months_between(s0, s1) // m, "first": D.origin.min(), "last_end": D.end.max(),
                           "mean_p": float(np.mean(dp)), "mean_f": float(np.mean(df_)), "mean_m": float(np.mean(dm)),
                           "sd_p": float(np.sqrt(vp)), "sd_f": float(np.std(df_, ddof=1)),
                           "sd_m": float(np.std(dm, ddof=1)), "share_m": _cov(dp, dm) / vp,
                           "share_f": _cov(dp, df_) / vp, "vr_m": float(np.var(dm, ddof=1)) / vp,
                           "vr_f": float(np.var(df_, ddof=1)) / vp,
                           "corr_fm": float(np.corrcoef(df_, dm)[0, 1]),
                           "mu_p": float(mu["lp"]), "mu_f": float(mu["lf"]), "mu_m": float(mu["lm"]),
                           "r2_orc_m": _r2(dp, m * float(mu["lf"]) + dm, bench),
                           "r2_orc_f": _r2(dp, df_ + m * float(mu["lm"]), bench),
                           "xs_rerate": float(np.mean(dm)) - m * float(mu["lm"])}
                    plc = multiple_skill(np.full(len(dm), m * float(mu["lm"])), dm)
                    rec.update({"placebo_n": plc["n"], "placebo_corr_pred_dm": plc["corr_pred_dm"],
                                "placebo_slope_dm_on_pred": plc["slope_dm_on_pred"],
                                "placebo_r2_dm_vs_zero": plc["r2_dm_vs_zero"],
                                "placebo_err_r2_m": _corr(dp - bench, dm) ** 2,
                                "placebo_err_share_m": _cov(dp, dm) / vp})
                    if paths is not None:
                        pa = paths[(paths.asset == a) & (paths.h_q == int(hq))]
                        if model is not None:
                            pa = pa[pa.model == model]
                        for prot in protocol_pref:
                            pp = pa[pa.protocol == prot]
                            if prot == "split":
                                pp = pp[pp.group == g]
                            cols = ["origin", "y", "yhat", "bench"] + (["yhat_mult"] if "yhat_mult" in pp else [])
                            J = D.merge(pp[cols], on="origin", how="inner")
                            J = J[np.isfinite(J.yhat)]
                            if len(J) >= 3:
                                yh, e = J.yhat.to_numpy(), J.dp.to_numpy() - J.yhat.to_numpy()
                                vy, ve = float(np.var(yh, ddof=1)), float(np.var(e, ddof=1))
                                if "yhat_mult" in J and meas == "cape" and np.isfinite(J.yhat_mult).all():
                                    pred, src = J.yhat_mult.to_numpy(float), "model"
                                else:
                                    pred, src = yh - m * float(mu["lf"]), "price_minus_fund_drift"
                                sk = multiple_skill(pred, J.dmult.to_numpy())
                                rec.update({
                                    "protocol": prot, "n_model": len(J),
                                    "r2": _r2(J.dp.to_numpy(), yh, J.bench.to_numpy()),
                                    "corr_yhat_p": float(np.corrcoef(yh, J.dp)[0, 1]),
                                    "corr_yhat_f": float(np.corrcoef(yh, J.dfund)[0, 1]),
                                    "corr_yhat_m": float(np.corrcoef(yh, J.dmult)[0, 1]),
                                    "beta_f": _cov(J.dfund, yh) / vy if vy > 0 else float("nan"),
                                    "beta_m": _cov(J.dmult, yh) / vy if vy > 0 else float("nan"),
                                    "err_share_m": _cov(e, J.dmult) / ve if ve > 0 else float("nan"),
                                    "err_r2_m": float(np.corrcoef(e, J.dmult)[0, 1] ** 2) if ve > 0 else float("nan"),
                                    "err_r2_f": float(np.corrcoef(e, J.dfund)[0, 1] ** 2) if ve > 0 else float("nan"),
                                    "mean_e": float(np.mean(e)),
                                    "max_abs_dp_minus_y": float(np.max(np.abs(J.dp - J.y))),
                                    "pred_dm_source": src, "corr_pred_dm": sk["corr_pred_dm"],
                                    "slope_dm_on_pred": sk["slope_dm_on_pred"], "r2_dm_vs_zero": sk["r2_dm_vs_zero"],
                                })
                                break
                    rows.append(rec)
    return pd.DataFrame(rows)


HEADLINE_COLUMNS = ["asset", "measure", "group", "h_q", "row", "n", "n_indep", "corr_pred_dm", "slope_dm_on_pred",
                    "r2_dm_vs_zero", "share_m"]


def anatomy_headline(anat: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """The anatomy as the UI shows it (§6.6, §10.1 D): per asset x measure x decade x horizon a model row (when the
    anatomy was computed with paths) and a placebo row (the constant drift forecast), with corr(predicted, realized
    Δmultiple), the slope of realized on predicted and R² against zero; never corr(miss, Δmultiple)²."""
    out = []
    for _, r in anat.iterrows():
        base = {"asset": r["asset"], "measure": r["measure"], "group": r["group"], "h_q": int(r["h_q"]),
                "n_indep": int(r["n_indep"]), "share_m": r["share_m"]}
        if "corr_pred_dm" in anat.columns and pd.notna(r.get("n_model")):
            out.append({**base, "row": model or "model", "n": int(r["n_model"]), "corr_pred_dm": r["corr_pred_dm"],
                        "slope_dm_on_pred": r["slope_dm_on_pred"], "r2_dm_vs_zero": r["r2_dm_vs_zero"]})
        out.append({**base, "row": "placebo", "n": int(r["placebo_n"]), "corr_pred_dm": r["placebo_corr_pred_dm"],
                    "slope_dm_on_pred": r["placebo_slope_dm_on_pred"], "r2_dm_vs_zero": r["placebo_r2_dm_vs_zero"]})
    return pd.DataFrame(out, columns=HEADLINE_COLUMNS)


# ---------------------------------------------------------------- the L0 reproduction (P2 exit test)
QSPLITS = {g: (str(pd.Period(tr, "Q")), str(pd.Period(t0, "Q")), "2026Q3" if g == "2015-2026" else
               str(pd.Period(t1, "Q"))) for g, (tr, t0, t1) in SPLITS.items()}


def l0_reproduction(panel: pd.DataFrame | None = None, Q: pd.DataFrame | None = None) -> dict:
    """The research round's L0 1Q scores (split protocol, mean R² over the 10 score assets against the training-mean
    drift; config_q.L0_CHECK_R2_1Q) against three evaluations of the same fitted F:

      monthly_path   Σ_m log1p(α + β'f_m) + c_log over the quarter's months: the research L0 (expected to reproduce)
      linear_monthly 3α + c + β'Σ_m f_m with F's own monthly tp10 switch (the quarterly form, minus the switch)
      quarterly      3α + c + β'F_q from the quarterly fields, tp10 switched at the quarter's start (the asset layer)

    and the identity check: linear_monthly computed from the monthly panel equals the quarterly form fed the monthly
    switch to 1e-12 (the unit conversion field / scale = the window sum). Returns per split the scores, the effects
    (switch = quarterly - linear_monthly; compounding = linear_monthly - monthly_path) and the largest quarter-level
    contributors to the gap."""
    from ..data import panel as mpanel
    from ..engine.factor_model import design_matrix
    from ..qmodel import panel as qpanel
    from ..qmodel.assets import RISK, AssetLayer, factor_inputs, oracle_inputs
    P = mpanel.load_panel() if panel is None else panel
    Q = qpanel.load_panel() if Q is None else Q
    out = {}
    for g, (tr, t0, t1) in QSPLITS.items():
        L = AssetLayer.fit(tr, panel=P, Q=Q, n_boot=0)
        qs = pd.period_range(t0, t1, freq="Q")
        X = oracle_inputs(Q, qs)
        y = Q.loc[qs, [f"y_{a}" for a in RISK]].set_axis(RISK, axis=1)
        bench = Q.loc[pd.Period(cq.FIRST_QUARTER, "Q"):pd.Period(tr, "Q"),
                      [f"y_{a}" for a in RISK]].mean().set_axis(RISK)
        months = pd.date_range(qs.min().start_time, qs.max().end_time.to_period("M").to_timestamp(), freq="MS")
        tpm = P.loc[months[0]:months[-1], "f_tp10_x_core4"]
        Fm = factor_inputs(X, L.features)               # the quarterly fields' factor sums ...
        Fm["tp10_x_core4"] = tpm.groupby(tpm.index.to_period("Q")).sum(min_count=3).reindex(qs).to_numpy(float)
        preds = {"monthly_path": L.monthly_path_mean(P, qs)[RISK],     # ... with F's own monthly switch
                 "linear_monthly": L.factor_mean(Fm, cma="historical", index=qs),
                 "quarterly": L.mean(X, cma="historical")[RISK]}
        # identity: the monthly panel's simple-return sum + the quarterly conversion = the quarterly fields' form
        Xm = design_matrix(P, L.spec).reindex(months)[L.features].to_numpy(float)
        ph = pd.DataFrame(Xm @ L.beta.loc[RISK, L.features].to_numpy(float).T, index=months, columns=RISK)
        lin = ph.groupby(ph.index.to_period("Q")).sum().reindex(qs) + L.kappa[RISK]
        ident = float((lin - preds["linear_monthly"]).abs().max().max())
        rec = {"train_end": tr, "test": [t0, t1], "n_quarters": int(len(qs)), "research": cq.L0_CHECK_R2_1Q[g],
               "identity_max_abs": ident}
        sst = ((y - bench) ** 2).sum()
        for k, pr in preds.items():
            r2 = 1.0 - ((y - pr) ** 2).sum() / sst
            rec[k] = float(r2.mean())
            rec[f"{k}_by_asset"] = {a: float(v) for a, v in r2.items()}
        rec["switch_effect"] = rec["quarterly"] - rec["linear_monthly"]
        rec["compounding_effect"] = rec["linear_monthly"] - rec["monthly_path"]
        rec["gap_vs_research"] = rec["quarterly"] - rec["research"]
        contrib = (((y - preds["quarterly"]) ** 2 - (y - preds["monthly_path"]) ** 2) / sst).sum(axis=1) / len(RISK)
        rec["largest_quarters"] = {str(q): -float(v) for q, v in contrib.sort_values(ascending=False).head(3).items()}
        out[g] = rec
    return out


# ---------------------------------------------------------------- self-check
def selfcheck(verbose: bool = True, quarterly: bool = True) -> dict:
    """Plumbing checks of the research harness, on the app's panel: B0 == AssetFactorModel, B0's residuals == MonthDB's,
    the engine draw identical, masking, v_lpd == log(100/s_div_yield), the drift model's R² = 0; and (quarterly) the
    quarterly drift model's R² = 0 through the quarterly adapter and the L0 unit-conversion identity."""
    from ..engine.factor_model import AssetFactorModel
    from ..engine.simulate import MonthDB
    P = load_panel(valuation=True)
    res = {}
    lf = LinearFitted(B0, P.loc[:"1994-12"], "1994-12")
    afm = AssetFactorModel(ModelSpec.from_dict({"factors": list(config.CORE_FACTOR_IDS)})).fit(P, end="1994-12")
    ph = lf.predict_simple(P, "1995-01", 120)
    res["b0_vs_assetfactormodel_max_abs"] = float((ph - afm.predict(P.loc["1995-01":"2004-12"])[lf.assets]).abs()
                                                  .max().max())
    res["b0_resid_vs_monthdb_max_abs"] = float((lf.R_simple - afm.residuals(P).loc[lf.rows, lf.assets]).abs()
                                               .max().max())
    db = MonthDB(P, afm, start=config.MODEL_START, end="1994-12")
    a = _eres.prior_draw(db, [60], 200, BLOCK, 123)
    b = engine_residual_draw(lf.R_simple.index, 60, 200, 123)
    res["engine_draw_identical"] = bool(len(db) == len(lf.R_simple) and np.array_equal(a, b))
    V = visible_panel(P, "2000-01", "2004-12")
    res["mask_ok"] = bool(V.loc["2000-01":, "p_us_equity"].isna().all() and V.loc["2000-01":, "f_oil"].notna().all()
                          and V.loc[:"1999-12", "p_us_equity"].notna().all() and V.index.max() == pd.Timestamp("2004-12"))
    lpd = np.log(100.0 / P["s_div_yield"])
    res["v_lpd_vs_div_yield_max_abs"] = float((P["v_lpd_us_equity"] - lpd).loc["1975":].abs().max())
    dm = metrics(conditional_paths(drift_model(), P, protocol="split", band=False), n_boot=10)
    res["drift_model_max_abs_r2"] = float(dm.r2.abs().max())
    res["shiller_source"] = P.attrs.get("shiller_source")
    if quarterly:
        from ..qmodel import panel as qpanel
        Q = qpanel.load_panel()
        dq = metrics(conditional_paths(quarterly_drift_fn(Q), P, protocol="split", horizons_q=(1, 4), band=False),
                     n_boot=10)
        res["drift_q_max_abs_r2"] = float(dq.r2.abs().max())
        rep = l0_reproduction(P, Q)
        res["l0_identity_max_abs"] = max(r["identity_max_abs"] for r in rep.values())
        res["l0_monthly_path_max_abs_dev"] = max(abs(r["monthly_path"] - r["research"]) for r in rep.values())
        res["l0_quarterly_dev"] = {g: r["quarterly"] - r["research"] for g, r in rep.items()}
    if verbose:
        for k, v in res.items():
            print(f"  {k}: {v}")
    return res


if __name__ == "__main__":
    selfcheck()
