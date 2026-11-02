"""The asset layer of the quarterly stack (SIM_DESIGN §4; Layer 3 资产路径：同期映射).

    r_{a,q} = κ_a + cma_a + Σ_k β_{a,k} F_{k,q} + ε_{a,q}        the 10 risk assets (L0)
    crude oil, the 10-year Treasury, cash                         identities on their own prices

L0 (the default, §4.1). α and β are the factor round's F (data/results/model_selection.json): a monthly OLS of
simple price returns on the monthly factor changes, refitted on the months up to the train end. F_{k,q} is the
quarter's sum of the monthly factor changes, obtained from the quarterly fields by unit conversion
(config_q.L0_FIELDS: d_oil/100 -> oil, d_ip/100 -> ip, d_cpi_goods/400 and d_cpi_serv/400 -> the CPI factors,
d_usd/100 -> usd, the pp fields unchanged, d_metals_oil/100 -> metals_oil in energy's equation only), and
tp10_x_core4 = d_tp10 × s_core4, switched at the quarter's start where F switches each month on the previous month's
core CPI. κ_a = 3 α_a + c_a puts the quarter in log units: c_a is the training-quarter mean of
y_{a,q} - 3 α_a - β_a'F_q (the simple-to-log conversion and the volatility drag), so the in-sample mean residual is 0.
The layer is linear in the fields, so the attribution is exact.

The research round's L0 summed log1p of the MONTHLY predictions (monthly_path_mean); a quarterly cell sees only the
quarter's sums, not how a factor's move is spread over the months, and its tp10 switch is set at the quarter's start.
evaluation.harness.l0_reproduction measures what each costs against the research scores (0.126 / 0.261 / 0.441 at
1Q, split): the monthly-path evaluation reproduces them; the quarterly form scores about 0.123 / 0.239 / 0.431, the
loss coming from the compounding of single extreme months (2008Q4, 2020Q1) and, in 2015-26, the switch (2021Q4).

Mechanical assets (identities, never regressions; scored apart from the 10 risk assets): crude = d_oil/100 (WTI
spot, no roll yield); the Treasury = a par bond with coupon s_y10 repriced at s_y10 + d_y10 (10-year maturity,
semiannual coupons, as data/panel.par_bond_return), income = coupon / 4; cash = the bill path (its three months earn
the bill rate of the month before, interpolated linearly from s_tb3 to the quarter-end rate). Cash's column holds
its total return, as the quarterly panel's y_cash does.

Income (simple, per quarter): equity-like assets k_a × exp(-s_pd_a)/4 (the own trailing P/D at the quarter's start;
k_a the training ratio of income to it); gold and crude 0; the Treasury its coupon; cash all of its return. The P/D
rolls with the price minus dividend growth: the valuation block's y_dg equation for us_equity, the training-mean
growth for the other dividend payers (their dividends are lumpy, §6.4).

Residuals (§4.5): the 13-asset quarterly residual vectors of the training quarters, with the y_fund and y_dg legs'
residuals of the same quarters, bootstrapped in blocks of config_q.RESID_BLOCK quarters (RESID_BLOCK_ARMS are the
sensitivity arms) and uncoupled from the factor innovations; each column is scaled by exp(γ (log s_vol - mean)),
γ fitted on the training quarters. Gold's residuals before its fit start (1982) come from that later fit, as in the
monthly engine. Mechanical assets have none. Parameter uncertainty: block-bootstrap refits of the monthly F
(config_q.PARAM_BOOT_BLOCK_MONTHS), one per path. Every path records the quarter label of each residual it drew.

Drift (§4.4): CMA "historical" keeps the fitted drift (walk-forward evaluation; backtests report it beside the other
arm); "no_rerate", the forward default (U-5, the default of mean / contributions / simulate), removes the
training-period re-rating of the CAPE multiple, ρ_a m̄ (ρ = 1 for us_equity, the other equities their beta to the
market), shrinks each equity's drift relative to the market by half (its oil-driven part excluded, since the factor
layer's anchors remove oil's real drift) and removes gold's real drift. No reversion term.

T-A candidates (§4.2): config_q.TA_CANDIDATES, all off (TA_DEFAULT). Switched on, which only the pre-registered test
T-A may do, a candidate adds a ridge on its terms fitted to the L0 training residuals (terms saturated at ±4 sd).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config, jsonio
from ..engine.factor_model import AssetFactorModel, ModelSpec, asset_feature_names, design_matrix
from . import config_q as cq
from . import valuation as val

ASSETS = list(cq.ASSETS)
RISK = list(cq.RISK_ASSETS)
MECH = list(cq.MECHANICAL_ASSETS)
LEGS = ["y_fund", "y_dg"]
POOL_COLUMNS = ASSETS + LEGS
DIVIDEND_ASSETS = list(cq.EQUITY_LIKE)
I = {a: i for i, a in enumerate(ASSETS)}
FIELD_OF = {fid: f for f, fid in cq.L0_FACTOR.items()}           # F factor id -> quarterly field
SCALE = {**{f: s for f, (_, s) in cq.L0_FIELDS.items()}, **{f: v[1] for f, v in cq.SCOPED_L0_FIELDS.items()}}
SELECTION_PATH = config.RESULTS_DIR / "model_selection.json"
VERSION = "2"                   # 2: the fix round (start state from published data, §3.2)


# ---------------------------------------------------------------- F (the factor round's selected model)

def l0_spec(path=None) -> ModelSpec:
    """F: the spec in data/results/model_selection.json, or config_q.L0_SPEC_FALLBACK where that file is missing."""
    p = SELECTION_PATH if path is None else path
    if p.exists():
        try:
            return ModelSpec.from_dict(jsonio.load(p)["spec"])
        except (KeyError, TypeError, ValueError):
            pass
    return ModelSpec.from_dict(cq.L0_SPEC_FALLBACK)


def check_spec(spec: ModelSpec) -> None:
    """The quarterly L0 needs a linear contemporaneous F on price returns whose factors all have quarterly fields."""
    if spec.kind != "linear" or spec.lags or spec.regime or spec.shock_cols or spec.target != "p":
        raise ValueError(f"季度 L0 只支持线性、同期、价格回报的 F（{spec.name}）")
    missing = [f for f in list(spec.factors) + [x for _, ids in spec.asset_factors for x in ids] if f not in FIELD_OF]
    if missing:
        raise ValueError(f"F 的因子 {missing} 没有对应的季度字段（config_q.L0_FACTOR）")


def _arr(v, shape=None) -> np.ndarray:
    a = np.asarray(v, float)
    return a if shape is None else np.broadcast_to(a, shape)


def _get(frame, col):
    if isinstance(frame, pd.DataFrame):
        return frame[col].to_numpy(float)
    return np.asarray(frame[col], float)


def factor_inputs(frame, features: list[str]) -> dict[str, np.ndarray]:
    """F's factor quarterly sums from quarterly fields (a DataFrame or a dict of equally shaped arrays): field/scale;
    tp10_x_core4 = d_tp10 × s_core4 (the start-of-quarter switch)."""
    out = {}
    for f in features:
        fld = FIELD_OF[f]
        x = _get(frame, fld)
        if f == "tp10_x_core4":
            out[f] = x * _get(frame, "s_core4")
        else:
            out[f] = x / SCALE[fld]
    return out


# ---------------------------------------------------------------- mechanical assets (identities)

def par_bond_price(coupon, y, years: float = 10.0) -> np.ndarray:
    """Price of a bond with annual coupon `coupon` (%), semiannual payments, priced at yield `y` (%), per 1 of face
    (data/panel.par_bond_return's formula)."""
    c = np.asarray(coupon, float) / 100.0
    yy = np.maximum(np.asarray(y, float) / 100.0, 1e-6)
    n = 2.0 * years
    disc = (1.0 + yy / 2.0) ** (-n)
    return (c / 2.0) * (1.0 - disc) / (yy / 2.0) + disc


def treasury_return(y10_start, d_y10) -> tuple[np.ndarray, np.ndarray]:
    """(log price return, simple income) of the 10-year par bond over a quarter: bought at s_y10, repriced at
    s_y10 + d_y10; income = coupon / 4."""
    y0 = np.asarray(y10_start, float)
    p = par_bond_price(y0, y0 + np.asarray(d_y10, float))
    return np.log(np.maximum(p, 1e-6)), y0 / 400.0


def cash_return(tb3_start, tb3_end) -> np.ndarray:
    """Log total return of 3-month bills over a quarter: its months earn the bill rate of the month before, i.e.
    tb3 at 0, 1/3 and 2/3 of the way from the quarter's start level to its end level (floored at 0)."""
    t0, t1 = np.asarray(tb3_start, float), np.asarray(tb3_end, float)
    return sum(np.log1p(np.maximum(t0 + (k / 3.0) * (t1 - t0), 0.0) / 1200.0) for k in range(3))


def bill_end(frame) -> np.ndarray:
    """The quarter-end bill rate: the frame's lvl_tb3 (oracle macro) or, without it, s_tb3 + d_ff (the bill moves
    one for one with the policy rate within the quarter), floored at 0."""
    if (isinstance(frame, pd.DataFrame) and "lvl_tb3" in frame.columns) or (not isinstance(frame, pd.DataFrame)
                                                                         and "lvl_tb3" in frame):
        t1 = _get(frame, "lvl_tb3")
        if np.isfinite(t1).all():
            return t1
    return np.maximum(_get(frame, "s_tb3") + _get(frame, "d_ff"), 0.0)


def mechanical_returns(frame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{asset: (log price return, simple income)} of crude, the Treasury and cash (cash: (total, exp(total) - 1))."""
    tre, tre_inc = treasury_return(_get(frame, "s_y10"), _get(frame, "d_y10"))
    cash = cash_return(_get(frame, "s_tb3"), bill_end(frame))
    oil = _get(frame, "d_oil") / 100.0
    return {"crude_oil": (oil, np.zeros_like(oil)), "treasury_10y": (tre, tre_inc), "cash": (cash, np.expm1(cash))}


# ---------------------------------------------------------------- residual pool

@dataclass
class ResidualPool:
    """Quarterly residual vectors of the training quarters (columns POOL_COLUMNS), standardized by the state
    volatility factor exp(γ (log s_vol - lv_mean)) and demeaned, for block-bootstrap draws."""
    labels: list
    columns: list
    raw: np.ndarray
    z: np.ndarray
    gamma: np.ndarray
    lv_mean: float
    block: int = cq.RESID_BLOCK

    @classmethod
    def fit(cls, E: pd.DataFrame, s_vol: pd.Series, block: int = cq.RESID_BLOCK,
            bounds=cq.VOL_GAMMA_BOUNDS) -> "ResidualPool":
        E = E.reindex(columns=POOL_COLUMNS)
        sv = s_vol.reindex(E.index)
        ok = np.isfinite(E.to_numpy(float)).all(axis=1) & np.isfinite(sv.to_numpy(float)) & (sv.to_numpy(float) > 0)
        E, sv = E[ok], sv[ok].to_numpy(float)
        if len(E) < 20:
            raise ValueError(f"残差池只有 {len(E)} 个季度")
        R = E.to_numpy(float)
        lv = np.log(sv)
        m = float(lv.mean())
        x = lv - m
        gamma = np.zeros(R.shape[1])
        for j in range(R.shape[1]):
            if np.allclose(R[:, j], 0.0):
                continue
            yj = np.log(np.abs(R[:, j]) + 1e-6)
            gamma[j] = float(np.clip(np.polyfit(x, yj, 1)[0], bounds[0], bounds[1]))
        Z = R / np.exp(gamma[None, :] * x[:, None])
        Z = Z - Z.mean(axis=0, keepdims=True)
        return cls(labels=[str(q) for q in E.index], columns=list(POOL_COLUMNS), raw=R, z=Z, gamma=gamma,
                   lv_mean=m, block=int(block))

    def __len__(self) -> int:
        return len(self.labels)

    def scale(self, s_vol) -> np.ndarray:
        """(..., k) volatility factors for state volatilities s_vol (...): exp(γ (log s_vol - lv_mean)); a missing or
        non-positive s_vol counts as the mean."""
        sv = np.asarray(s_vol, float)
        lv = np.where(np.isfinite(sv) & (sv > 0), np.log(np.where(sv > 0, sv, 1.0)), self.lv_mean)
        return np.exp((lv - self.lv_mean)[..., None] * self.gamma)

    def starts(self, block: int) -> np.ndarray:
        """Positions where `block` consecutive calendar quarters of the pool begin."""
        q = pd.PeriodIndex(self.labels, freq="Q")
        ordinal = np.asarray(q.asi8)
        n = len(q)
        ok = [i for i in range(n - block + 1) if ordinal[i + block - 1] - ordinal[i] == block - 1]
        if not ok:
            raise ValueError(f"残差池里没有连续 {block} 个季度")
        return np.asarray(ok, int)

    def draw_index(self, n_paths: int, H: int, rng: np.random.Generator, block: int | None = None) -> np.ndarray:
        """(n_paths, H) pool positions: blocks of `block` consecutive quarters with uniformly drawn starts."""
        B = int(block or self.block)
        st = self.starts(B)
        nb = int(math.ceil(H / B))
        s = st[rng.integers(0, len(st), size=(int(n_paths), nb))]
        return (s[:, :, None] + np.arange(B)[None, None, :]).reshape(int(n_paths), -1)[:, :H]

    def label_array(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(self.labels, dtype=object)[idx]

    def draws(self, idx: np.ndarray, s_vol) -> np.ndarray:
        """Residuals (..., k) for pool positions idx (...) at state volatilities s_vol (...)."""
        return self.z[idx] * self.scale(s_vol)

    def resid_beta(self) -> dict[str, float]:
        """The other equity-like assets' residual regression on the us_equity residual (the valuation shift)."""
        e = self.raw[:, I["us_equity"]]
        v = float(np.var(e))
        return {a: float(np.cov(self.raw[:, I[a]], e, bias=True)[0, 1] / v) for a in DIVIDEND_ASSETS}

    def summary(self) -> dict:
        return {"n": len(self), "first": self.labels[0], "last": self.labels[-1], "block": self.block,
                "gamma": dict(zip(self.columns, map(float, self.gamma))), "lv_mean": self.lv_mean,
                "sd": dict(zip(self.columns, map(float, self.raw.std(axis=0))))}


# ---------------------------------------------------------------- parameter uncertainty

@dataclass
class ParamDraws:
    """Block-bootstrap refits of the monthly F: alpha (B, 12 F assets), beta (B, 12, features)."""
    alpha: np.ndarray
    beta: np.ndarray
    block_months: int
    seed: int

    def __len__(self) -> int:
        return int(self.alpha.shape[0])


def bootstrap_f(X: np.ndarray, Y: np.ndarray, own: dict, cols: dict, n: int, block: int, seed: int) -> ParamDraws:
    """Refit every equation of F on 12-month blocks of the fit months drawn with replacement.

    X (T, features), Y (T, assets) on the fit months; cols[asset] = the feature positions of its equation;
    own[asset] = a row mask for an asset with a later first month (gold)."""
    rng = np.random.default_rng(seed)
    T, A = Y.shape
    nb = int(math.ceil(T / block))
    alpha = np.zeros((n, A))
    beta = np.zeros((n, A, X.shape[1]))
    for b in range(n):
        s = rng.integers(0, T - block + 1, size=nb)
        rows = (s[:, None] + np.arange(block)[None, :]).ravel()[:T]
        for j in range(A):
            r = rows if j not in own else rows[own[j][rows]]
            c = cols[j]
            Z = np.column_stack([np.ones(len(r)), X[np.ix_(r, c)]])
            coef = np.linalg.lstsq(Z, Y[r, j], rcond=None)[0]
            alpha[b, j] = coef[0]
            beta[b, j, c] = coef[1:]
    return ParamDraws(alpha=alpha, beta=beta, block_months=int(block), seed=int(seed))


# ---------------------------------------------------------------- T-A candidates (off by default)

def candidate_terms(cid: str, frame, asset: str, states: dict | None = None) -> dict[str, np.ndarray]:
    """The raw terms of candidate `cid` for one asset's equation (arrays shaped like the frame's columns). States
    that the path rolls (s_vol, s_pd) are taken from `states` when given."""
    spec = cq.TA_CANDIDATES[cid]
    st = states or {}

    def g(c):
        return np.asarray(st[c], float) if c in st else _get(frame, c)

    out = {}
    if spec["kind"] == "fields":
        names = list(spec["terms"]) + [f for f, scope in spec.get("scoped", {}).items() if asset in (scope or [])]
        for c in names:
            out[c] = np.nan_to_num(g(c))
    elif spec["kind"] == "states":
        out["s_vol"] = g("s_vol")
        out["s_zlb"] = g("s_zlb")
        out["s_core4"] = g("s_core4")
        out["y10_x_pd"] = g("d_y10") * (g("s_pd") - 3.6)          # the qmodel's centring
        out["oil_x_zlb"] = g("d_oil") * g("s_zlb")
    elif spec["kind"] == "lead":
        out["d_ip_lead"] = g("d_ip_lead")
        out["d_unrate_lead"] = g("d_unrate_lead")
    else:
        raise ValueError(f"候选 {cid}（{spec['zh']}）只作对照，没有实现为资产方程的附加项")
    return out


def _ridge_cv(Z: np.ndarray, y: np.ndarray, grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0), k: int = 5):
    """Ridge without intercept on centred data with blocked k-fold CV over lambda = g × n (qmodel qcore.ridge_cv)."""
    n, p = Z.shape
    edges = np.linspace(0, n, k + 1).astype(int)
    best, best_lam = np.inf, None
    for g in grid:
        lam = g * n
        sse = 0.0
        for i in range(k):
            te = np.arange(edges[i], edges[i + 1])
            tr = np.r_[0:edges[i], edges[i + 1]:n]
            zm, ym = Z[tr].mean(0), y[tr].mean()
            b = np.linalg.solve((Z[tr] - zm).T @ (Z[tr] - zm) + lam * np.eye(p), (Z[tr] - zm).T @ (y[tr] - ym))
            sse += float(np.sum((y[te] - ym - (Z[te] - zm) @ b) ** 2))
        if sse < best:
            best, best_lam = sse, lam
    zm, ym = Z.mean(0), y.mean()
    b = np.linalg.solve((Z - zm).T @ (Z - zm) + best_lam * np.eye(p), (Z - zm).T @ (y - ym))
    return float(ym - zm @ b), b, float(best_lam)


@dataclass
class CandidateFit:
    cid: str
    asset: str
    names: list
    mu: np.ndarray
    sd: np.ndarray
    intercept: float
    coef: np.ndarray
    lam: float

    def predict(self, terms: dict) -> np.ndarray:
        Zs = [np.clip((np.asarray(terms[c], float) - m) / s, -cq.TA_CLIP_SD, cq.TA_CLIP_SD)
              for c, m, s in zip(self.names, self.mu, self.sd)]
        out = self.intercept + sum(b * z for b, z in zip(self.coef, Zs))
        return np.asarray(out, float)


# ---------------------------------------------------------------- simulation output

@dataclass
class SimPaths:
    """Simulated quarterly paths of the asset layer (n paths × H quarters; assets in config_q.ASSETS order)."""
    quarters: list
    assets: list
    price: np.ndarray          # (n, H, 13) log price returns (cash: its total return)
    income: np.ndarray         # (n, H, 13) simple income
    total: np.ndarray          # (n, H, 13) log total returns
    systematic: np.ndarray     # (n, H, 13) drift + factors (+ candidates) before the residual and the shift
    resid: np.ndarray          # (n, H, 13)
    shift: np.ndarray          # (n, H, 13) the valuation statements' shift
    y_fund: np.ndarray         # (n, H)
    y_mult: np.ndarray         # (n, H) Δlog CAPE multiple = us_equity price - y_fund
    y_dg: np.ndarray           # (n, H)
    y_pd: np.ndarray           # (n, H)
    log_cape: np.ndarray       # (n, H + 1) implied log CAPE (start of each quarter, end of the last)
    log_pd: np.ndarray         # (n, H + 1) implied log P/D
    s_vol: np.ndarray          # (n, H) state volatility each quarter's residual was scaled with
    drawn: np.ndarray          # (n, H) quarter labels of the residual vectors drawn
    param_index: np.ndarray    # (n,) refit of F each path used (-1: the point fit)
    cma: str = "historical"
    statements: list = field(default_factory=list)

    def asset(self, a: str, what: str = "price") -> np.ndarray:
        return getattr(self, what)[:, :, I[a]]


# ---------------------------------------------------------------- the layer

@dataclass
class AssetLayer:
    """L0 + identities + residual model + income + CMA, fitted on the data up to `train_end` (a quarter)."""
    train_end: str
    spec: ModelSpec
    features: list
    f_assets: list                  # F's equations (config's MODEL_ASSETS: the 12 assets with a price return)
    alpha: pd.Series                # monthly intercepts
    beta: pd.DataFrame              # monthly betas (F assets × features; 0 outside an equation)
    c_log: pd.Series                # the harness's monthly simple-to-log constant (monthly-path L0)
    kappa: pd.Series                # quarterly log intercept of the risk assets (3 α + c)
    conv: pd.Series                 # c = kappa - 3 alpha
    calib_quarters: dict
    n_fit_months: dict
    pool: ResidualPool
    valuation: val.ValuationBlock
    cma_adj: dict                   # {arm: pd.Series over ASSETS (quarterly log)}
    cma_detail: dict
    income_k: pd.Series
    div_growth: pd.Series
    candidates: tuple = ()
    cand_fits: dict = field(default_factory=dict)
    params: ParamDraws | None = None
    fit_info: dict = field(default_factory=dict)

    # ------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, train_end, *, panel: pd.DataFrame | None = None, Q: pd.DataFrame | None = None,
            spec: ModelSpec | None = None, candidates=cq.TA_DEFAULT, n_boot: int = cq.PARAM_BOOT, seed: int = 0,
            block: int = cq.RESID_BLOCK) -> "AssetLayer":
        """Fit on the monthly panel to the last month of `train_end` and the quarterly panel to `train_end`.

        n_boot: block-bootstrap refits of F for the parameter uncertainty of simulated paths (0: point fit only);
        candidates: T-A candidates to add (off by default; only the pre-registered T-A may switch one on)."""
        from ..data import panel as monthly_panel
        from . import panel as qpanel
        te = pd.Period(str(train_end), "Q")
        te_m = te.end_time.to_period("M").to_timestamp()
        P = monthly_panel.load_panel() if panel is None else panel
        Q = qpanel.load_panel() if Q is None else Q
        spec = spec or l0_spec()
        check_spec(spec)
        for c in candidates:
            if c not in cq.TA_CANDIDATES:
                raise ValueError(f"未知的 T-A 候选 {c!r}")
            if cq.TA_CANDIDATES[c].get("available") is False:
                raise ValueError(f"候选 {c}（{cq.TA_CANDIDATES[c]['zh']}）只作对照，不能加入资产方程")
        m = AssetFactorModel(spec).fit(P, config.MODEL_START, te_m)
        features = list(m.features)
        f_assets = list(m.assets)
        alpha, beta = m.intercept.copy(), m.coef.copy()
        # ---- the fit months (AssetFactorModel's joint mask) and the harness's monthly log constant
        X_all = design_matrix(P, spec)
        W = P.loc[config.MODEL_START:te_m]
        X = X_all.loc[W.index]
        Y = W[[f"p_{a}" for a in f_assets]].set_axis(f_assets, axis=1)
        mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
        rows = X.index[mask]
        own_start = {a: pd.Timestamp(s) for a, s in m.asset_start_used.items() if s}
        pred = pd.DataFrame(X.loc[rows].to_numpy(float) @ beta.loc[f_assets, features].to_numpy(float).T
                            + alpha.loc[f_assets].to_numpy(float), index=rows, columns=f_assets)
        c_log, n_fit = {}, {}
        for a in f_assets:
            ra = rows[rows >= own_start[a]] if a in own_start else rows
            y = Y.loc[ra, a].to_numpy(float)
            c_log[a] = float(np.mean(np.log1p(y) - np.log1p(np.maximum(pred.loc[ra, a].to_numpy(float),
                                                                            cq.L0_LOG_FLOOR))))
            n_fit[a] = int(len(ra))
        # ---- quarterly log intercept: quarters whose three months are all fit months
        Qt = Q.loc[:te]
        Fq = factor_inputs(Qt, features)
        Fmat = np.column_stack([Fq[f] for f in features])
        months_in = pd.Series(1, index=rows).groupby(rows.to_period("Q")).count()
        kappa, conv, calib = {}, {}, {}
        for a in RISK:
            full = months_in[months_in == 3].index
            if a in own_start:
                full = full[full >= own_start[a].to_period("Q")]
            full = full[full.isin(Qt.index)]
            pos = Qt.index.get_indexer(full)
            s = 3.0 * alpha[a] + Fmat[pos] @ beta.loc[a, features].to_numpy(float)
            yq = Qt[f"y_{a}"].to_numpy(float)[pos]
            ok = np.isfinite(s) & np.isfinite(yq)
            conv[a] = float(np.mean(yq[ok] - s[ok]))
            kappa[a] = 3.0 * float(alpha[a]) + conv[a]
            calib[a] = [str(full[ok].min()), str(full[ok].max()), int(ok.sum())]
        kappa, conv = pd.Series(kappa), pd.Series(conv)
        # ---- the valuation block and the residual vectors (risk assets, mechanical 0, the two legs)
        vb = val.ValuationBlock.fit(Q, te)
        sysq = pd.DataFrame({a: kappa[a] + Fmat @ beta.loc[a, features].to_numpy(float) for a in RISK},
                            index=Qt.index)
        E = pd.DataFrame(index=Qt.index, columns=POOL_COLUMNS, dtype=float)
        for a in RISK:
            E[a] = Qt[f"y_{a}"] - sysq[a]
        for a in MECH:
            E[a] = 0.0
        E["y_fund"] = vb.fund.resid.reindex(Qt.index)
        E["y_dg"] = vb.dg.resid.reindex(Qt.index)
        E = E.loc[pd.Period(cq.FIRST_QUARTER, "Q"):]
        pool = ResidualPool.fit(E, Qt["s_vol"], block=block)
        layer = cls(train_end=str(te), spec=spec, features=features, f_assets=f_assets, alpha=alpha, beta=beta,
                    c_log=pd.Series(c_log), kappa=kappa, conv=conv, calib_quarters=calib, n_fit_months=n_fit,
                    pool=pool, valuation=vb, cma_adj={}, cma_detail={}, income_k=pd.Series(dtype=float),
                    div_growth=pd.Series(dtype=float), candidates=tuple(candidates))
        layer.cma_adj, layer.cma_detail = layer._cma(Qt)
        layer.income_k, layer.div_growth = layer._income(Qt)
        layer.fit_info = {"train_end_month": te_m.strftime("%Y-%m"),
                          "fit_window": [rows.min().strftime("%Y-%m"), rows.max().strftime("%Y-%m"), int(len(rows))],
                          "asset_start_used": dict(m.asset_start_used)}
        if candidates:
            layer.cand_fits = layer._fit_candidates(Qt, E)
        if n_boot:
            cols = {j: [features.index(f) for f in asset_feature_names(spec, a)] for j, a in enumerate(f_assets)}
            own = {j: np.asarray(rows >= own_start[a]) for j, a in enumerate(f_assets) if a in own_start}
            layer.params = bootstrap_f(X.loc[rows, features].to_numpy(float), Y.loc[rows].to_numpy(float), own,
                                       cols, int(n_boot), cq.PARAM_BOOT_BLOCK_MONTHS, int(seed))
        return layer

    def _cma(self, Qt: pd.DataFrame) -> tuple[dict, dict]:
        """The two CMA arms as quarterly log adjustments over ASSETS (SIM_DESIGN §4.4)."""
        T = Qt.loc[pd.Period(val.FIRST_LEG_QUARTER, "Q"):]
        m_bar = self.valuation.m_bar
        mkt = T["y_us_equity"]
        oil_real = (T["d_oil"] / 100.0 - T["d_cpi"] / 400.0).dropna()
        oil_drift = float(oil_real.mean())
        oil_beta = {a: float(self.beta.loc[a, "oil"]) if "oil" in self.beta.columns else 0.0 for a in RISK}
        hist = {a: float(T[f"y_{a}"].mean()) for a in RISK}
        rho, rel = {}, {}
        adj = pd.Series(0.0, index=ASSETS)
        adj["us_equity"] = -m_bar
        for a in DIVIDEND_ASSETS:
            if a == "us_equity":
                continue
            d = pd.concat([T[f"y_{a}"], mkt], axis=1).dropna()
            b = float(np.cov(d.iloc[:, 0], d.iloc[:, 1], bias=True)[0, 1] / np.var(d.iloc[:, 1]))
            rho[a] = b
            rel[a] = hist[a] - b * hist["us_equity"] - (oil_beta[a] - b * oil_beta["us_equity"]) * oil_drift
            adj[a] = -b * m_bar - cq.CMA_SHRINK_RELATIVE * rel[a]
        g = (T["y_gold"] - T["d_cpi"] / 400.0).dropna()
        gold_real = float(g.mean())
        adj["gold"] = -gold_real + oil_beta["gold"] * oil_drift
        detail = {"m_bar_q": m_bar, "m_bar_pct_per_year": 400.0 * m_bar, "beta_to_market": rho,
                  "relative_drift_q": rel, "shrink_relative": cq.CMA_SHRINK_RELATIVE, "oil_real_drift_q": oil_drift,
                  "gold_real_drift_q": gold_real, "historical_drift_q": hist,
                  "no_rerate_drift_q": {a: hist[a] + float(adj[a]) for a in RISK}}
        return {"historical": pd.Series(0.0, index=ASSETS), "no_rerate": adj}, detail

    def _income(self, Qt: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """k_a = Σ income / Σ exp(-s_pd_a)/4 over the training quarters; the non-market dividend payers' mean growth."""
        k, g = {}, {}
        for a in DIVIDEND_ASSETS:
            pdc = "s_pd" if a == "us_equity" else f"s_pd_{a}"
            d = pd.concat([Qt[f"inc_{a}"], np.exp(-Qt[pdc]) / 4.0], axis=1).dropna()
            k[a] = float(d.iloc[:, 0].sum() / d.iloc[:, 1].sum())
            if a != "us_equity":
                dg = (Qt[f"y_{a}"] - (Qt[pdc].shift(-1) - Qt[pdc])).dropna()
                g[a] = float(dg.mean())
        return pd.Series(k), pd.Series(g)

    def _fit_candidates(self, Qt: pd.DataFrame, E: pd.DataFrame) -> dict:
        out = {}
        frame = Qt.copy()
        frame["d_ip_lead"] = Qt["d_ip"].shift(-1)
        frame["d_unrate_lead"] = Qt["d_unrate"].shift(-1)
        for cid in self.candidates:
            first = cq.TA_CANDIDATES[cid].get("first", {})
            for a in RISK:
                terms = candidate_terms(cid, frame, a)
                for c, q0 in first.items():
                    if c in terms:
                        terms[c] = np.where(frame.index >= pd.Period(q0, "Q"), terms[c], 0.0)
                names = list(terms)
                M = np.column_stack([terms[c] for c in names])
                y = E[a].reindex(frame.index).to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(M).all(axis=1)
                mu, sd = M[ok].mean(0), M[ok].std(0)
                sd = np.where(sd > 1e-12, sd, 1.0)
                Z = np.clip((M[ok] - mu) / sd, -cq.TA_CLIP_SD, cq.TA_CLIP_SD)
                b0, b, lam = _ridge_cv(Z, y[ok])
                out[(cid, a)] = CandidateFit(cid, a, names, mu, sd, b0, b, lam)
        return out

    # ------------------------------------------------------------ coefficients
    def beta_fields(self) -> pd.DataFrame:
        """The risk assets' betas per quarterly field unit (β / scale; d_tp10's applies when s_core4 = 1): what a
        stated or completed field moves a quarter's log return by."""
        out = {}
        for f in self.features:
            fld = FIELD_OF[f]
            out[fld] = self.beta.loc[RISK, f] / (1.0 if f == "tp10_x_core4" else SCALE[fld])
        return pd.DataFrame(out)

    def cma(self, arm: str) -> pd.Series:
        if arm not in cq.CMA_ARMS:
            raise ValueError(f"未知的 CMA 档 {arm!r}（{cq.CMA_ARMS}）")
        return self.cma_adj[arm]

    # ------------------------------------------------------------ the systematic part
    def _risk_mean(self, F: dict, cma: str, draw: np.ndarray | None = None, shape=None) -> np.ndarray:
        """(..., 10) κ + cma + β'F for factor arrays F (each shaped `shape`); draw (n,) = each path's refit of F
        (F arrays shaped (n, H)), its quarterly intercept 3 α_draw + the point fit's conversion c."""
        adj = self.cma(cma)
        Fm = np.stack([np.broadcast_to(np.asarray(F[f], float), shape) for f in self.features], axis=-1)
        if draw is None or self.params is None:
            B = self.beta.loc[RISK, self.features].to_numpy(float)              # (10, k)
            return Fm @ B.T + (self.kappa[RISK] + adj[RISK]).to_numpy(float)
        cols = [self.f_assets.index(a) for a in RISK]
        Bd = self.params.beta[:, cols, :][np.asarray(draw, int)]                # (n, 10, k)
        ad = self.params.alpha[:, cols][np.asarray(draw, int)]                  # (n, 10)
        return (np.einsum("nhk,nak->nha", Fm, Bd) + 3.0 * ad[:, None, :]
                + (self.conv[RISK] + adj[RISK]).to_numpy(float))

    def factor_mean(self, F: dict, cma: str = cq.CMA_FORWARD, index=None) -> pd.DataFrame:
        """κ + cma + β'F of the 10 risk assets for given factor quarterly sums F {feature: array} (e.g. with F's own
        monthly tp10 switch summed over the quarter instead of the start-of-quarter one)."""
        shape = np.shape(next(iter(F.values())))
        return pd.DataFrame(self._risk_mean(F, cma, shape=shape), index=index, columns=RISK)

    def _cand_mean(self, frame, states: dict | None, shape) -> np.ndarray:
        out = np.zeros(tuple(shape) + (len(RISK),))
        for cid in self.candidates:
            for j, a in enumerate(RISK):
                cf = self.cand_fits[(cid, a)]
                terms = candidate_terms(cid, frame, a, states)
                out[..., j] += np.broadcast_to(cf.predict(terms), shape)
        return out

    def mean(self, frame, cma: str = cq.CMA_FORWARD, states: dict | None = None) -> pd.DataFrame | np.ndarray:
        """Systematic quarterly log price returns of the 13 assets (cash: total) for rows of quarterly inputs: the
        L0 fields, s_core4, and for the identities s_y10, d_y10, s_tb3, lvl_tb3 (or d_ff). A DataFrame in gives a
        DataFrame out; a dict of (n, H) arrays gives an (n, H, 13) array. cma defaults to the forward arm
        (no_rerate, U-5): walk-forward evaluation passes "historical", backtests report both."""
        F = factor_inputs(frame, self.features)
        shape = np.shape(next(iter(F.values())))
        out = np.zeros(tuple(shape) + (len(ASSETS),))
        out[..., [I[a] for a in RISK]] = self._risk_mean(F, cma, shape=shape) + self._cand_mean(frame, states, shape)
        for a, (p, _) in mechanical_returns(frame).items():
            out[..., I[a]] = p
        if isinstance(frame, pd.DataFrame):
            return pd.DataFrame(out, index=frame.index, columns=ASSETS)
        return out

    def contributions(self, frame: pd.DataFrame, stated=None, cma: str = cq.CMA_FORWARD) -> dict:
        """The attribution of the systematic part (SIM_DESIGN §4.4): per field β_k F_k (DataFrames rows × ASSETS),
        'drift' (κ + the CMA arm), 'stated' / 'completed' (the fields the QTable stated / the model completed: a set
        of field ids, or a boolean DataFrame rows × fields), and 'candidates'. Mechanical assets: crude to d_oil,
        the Treasury to d_y10, cash's carry at the start rate to drift and its bill change to d_ff. stated +
        completed + drift + candidates = mean()."""
        F = factor_inputs(frame, self.features)
        idx = frame.index
        adj = self.cma(cma)
        fields = {}
        for f in self.features:
            fld = FIELD_OF[f]
            d = fields.setdefault(fld, pd.DataFrame(0.0, index=idx, columns=ASSETS))
            d[RISK] = d[RISK].to_numpy() + np.outer(F[f], self.beta.loc[RISK, f].to_numpy(float))
        drift = pd.DataFrame(0.0, index=idx, columns=ASSETS)
        drift[RISK] = np.tile((self.kappa[RISK] + adj[RISK]).to_numpy(float), (len(idx), 1))
        mech = mechanical_returns(frame)
        for fld, a in (("d_oil", "crude_oil"), ("d_y10", "treasury_10y")):
            d = fields.setdefault(fld, pd.DataFrame(0.0, index=idx, columns=ASSETS))
            d[a] = mech[a][0]
        carry = cash_return(_get(frame, "s_tb3"), _get(frame, "s_tb3"))
        drift["cash"] = carry
        d = fields.setdefault("d_ff", pd.DataFrame(0.0, index=idx, columns=ASSETS))
        d["cash"] = mech["cash"][0] - carry
        cand = pd.DataFrame(0.0, index=idx, columns=ASSETS)
        if self.candidates:
            cand[RISK] = self._cand_mean(frame, None, (len(idx),))
        if stated is None:
            mask = pd.DataFrame(False, index=idx, columns=list(fields))
        elif isinstance(stated, pd.DataFrame):
            mask = stated.reindex(index=idx, columns=list(fields)).fillna(False).astype(bool)
        else:
            mask = pd.DataFrame({f: f in set(stated) for f in fields}, index=idx)
        st = sum(fields[f].mul(mask[f].astype(float), axis=0) for f in fields)
        co = sum(fields[f].mul(1.0 - mask[f].astype(float), axis=0) for f in fields)
        return {"fields": fields, "drift": drift, "stated": st, "completed": co, "candidates": cand}

    def monthly_path_mean(self, panel: pd.DataFrame, quarters) -> pd.DataFrame:
        """The research round's L0 (qmodel qeval.L0Model = simharness LinearFitted(F)): Σ over the quarter's months of
        log1p(max(α + β'f_m, -0.95)) + c_log, from the monthly panel's realized factor changes (oracle macro)."""
        qs = pd.PeriodIndex([pd.Period(str(q), "Q") for q in quarters], freq="Q")
        months = pd.date_range(qs.min().start_time, qs.max().end_time.to_period("M").to_timestamp(), freq="MS")
        X = design_matrix(panel, self.spec).reindex(months)[self.features].to_numpy(float)
        ph = X @ self.beta.loc[RISK, self.features].to_numpy(float).T + self.alpha[RISK].to_numpy(float)
        lm = pd.DataFrame(np.log1p(np.maximum(ph, cq.L0_LOG_FLOOR)) + self.c_log[RISK].to_numpy(float),
                          index=months, columns=RISK)
        g = lm.groupby(lm.index.to_period("Q"))
        out = g.sum(min_count=3)
        out[g.count() < 3] = np.nan
        return out.reindex(qs)

    # ------------------------------------------------------------ the start of a window
    def start_state(self, Q: pd.DataFrame, q0, information: str = "auto", prev: dict | None = None) -> dict:
        """The states the layer reads at the start of quarter q0, from data known then (§3.2): s_vol, the us_equity
        return of q0-1, s_cape, every dividend payer's log P/D, y_dg of q0-1, and the y_fund lags. y_fund of q0-2 is
        published at q0; that of q0-1 is not (Shiller E, lag 3) and is predicted from its equation on q0-1's d_cpi,
        d_ip, d_unrate and d_rgdp -- which are not published at q0 either (lag 1): information 'published' (the
        backtest default, factors.initial_state's rule) reads only published values, `prev` supplies the factor
        layer's imputed ones (the rollout passes them), and without either y_fund(q0-1) = y_fund(q0-2).
        'available' (the forward root, 'auto' at the panel's today row) reads every value the panel holds."""
        from . import factors as F
        q0 = pd.Period(str(q0), "Q")
        info = F._information(Q, q0, information)
        row = Q.loc[q0]
        st = {"s_vol": float(row["s_vol"]), "y_eq_l1": float(row["m1_us_equity"]), "s_cape": float(row["s_cape"]),
              "y_dg_l1": F._pub(Q, "y_dg", q0, 1, info)}
        for a in DIVIDEND_ASSETS:
            st[f"s_pd_{a}"] = float(row["s_pd"] if a == "us_equity" else row[f"s_pd_{a}"])
        f2 = F._pub(Q, "y_fund", q0, 2, info)
        f3 = F._pub(Q, "y_fund", q0, 3, info) if (q0 - 3) in Q.index else f2
        if not math.isfinite(f3):
            f3 = f2
        reg = {}
        for c in ("d_cpi", "d_ip", "d_unrate", "d_rgdp"):
            v = F._pub(Q, c, q0, 1, info)
            if not math.isfinite(v) and prev is not None and c in prev and math.isfinite(float(prev[c])):
                v = float(prev[c])
            reg[c] = np.array([v])
        f1 = float(self.valuation.fund.predict(val.fund_regressors(reg), f3)[0])
        if not math.isfinite(f1):                       # a macro input of q0-1 not published (and not imputed)
            f1 = f2
        st["y_fund_lags"] = [f2, f1]                    # values of q0-2 and q0-1
        return st

    # ------------------------------------------------------------ simulation
    def simulate(self, inputs, start: dict, n_paths: int = 1000, seed: int = 0, *, cma: str = cq.CMA_FORWARD,
                 block: int | None = None, params: bool | None = None, statements=(), quarters=None) -> SimPaths:
        """Simulate H quarters of the 13 assets from per-quarter inputs.

        inputs   a DataFrame with H rows (the same inputs on every path: the fields of mean() plus d_cpi, d_ip,
                 d_unrate, d_rgdp for the legs) or a dict of (n_paths, H) arrays (per-path completed factors)
        start    start_state(): s_vol, y_eq_l1, s_cape, s_pd_<asset>, y_dg_l1, y_fund_lags (scalars, or one value
                 per path: step mode)
        cma      "no_rerate" (the forward default, U-5) or "historical"; block: residual block (quarters, default
                 config_q.RESID_BLOCK); params: draw a refit of F per path (default: when the layer has refits);
                 statements: valuation.ValuationStatement list (c_q.valuation)
        Returns SimPaths. Determinism: the same seed gives the same paths."""
        rng = np.random.default_rng(seed)
        n = int(n_paths)
        if isinstance(inputs, pd.DataFrame):
            H = len(inputs)
            frame = {c: np.broadcast_to(inputs[c].to_numpy(float), (n, H)) for c in inputs.columns}
            labels = [str(q) for q in inputs.index] if quarters is None else list(quarters)
        else:
            frame = {c: np.broadcast_to(np.asarray(v, float), np.broadcast_shapes(np.shape(v), (n, np.shape(v)[-1])))
                     for c, v in inputs.items()}
            H = next(iter(frame.values())).shape[-1]
            labels = list(quarters) if quarters is not None else [f"q{k + 1}" for k in range(H)]
        use_params = (self.params is not None) if params is None else bool(params)
        if use_params and self.params is None:
            raise ValueError("这个资产层没有参数自助重估（fit 时 n_boot=0）")
        pidx = rng.integers(0, len(self.params), size=n) if use_params else np.full(n, -1)
        idx = self.pool.draw_index(n, H, rng, block)
        F = factor_inputs(frame, self.features)
        risk_cols = [I[a] for a in RISK]
        sysm = np.zeros((n, H, len(ASSETS)))
        sysm[:, :, risk_cols] = self._risk_mean(F, cma, pidx if use_params else None, shape=(n, H))
        mech = mechanical_returns(frame)
        for a, (p, _) in mech.items():
            sysm[:, :, I[a]] = p
        reg = val.fund_regressors(frame)
        price = np.zeros_like(sysm)
        resid = np.zeros_like(sysm)
        y_fund = np.zeros((n, H))
        y_dg = np.zeros((n, H))
        s_vol = np.zeros((n, H))

        def per_path(v):                       # a start state: one value for every path, or one per path
            return np.broadcast_to(np.asarray(v, float), (n,)).copy()

        sv = per_path(start["s_vol"])
        y_prev = per_path(start["y_eq_l1"])
        f_lags = [per_path(v) for v in start["y_fund_lags"]]               # q-2, q-1
        dg_prev = per_path(start["y_dg_l1"])
        cand_states = {"s_vol": sv, "s_pd": per_path(start["s_pd_us_equity"])}
        ip = self.pool.columns.index("y_fund"), self.pool.columns.index("y_dg")
        for k in range(H):
            s_vol[:, k] = sv
            if self.candidates:
                sub = {c: v[:, k] for c, v in frame.items()}
                sysm[:, k, risk_cols] += self._cand_mean(sub, cand_states, (n,))
            e = self.pool.draws(idx[:, k], sv)                         # (n, 15)
            resid[:, k, risk_cols] = e[:, risk_cols]
            price[:, k, :] = sysm[:, k, :] + resid[:, k, :]
            rk = {c: v[:, k] for c, v in reg.items()}
            y_fund[:, k] = self.valuation.fund.predict(rk, f_lags[0]) + e[:, ip[0]]
            y_dg[:, k] = self.valuation.dg.predict(rk, dg_prev) + e[:, ip[1]]
            f_lags = [f_lags[1], y_fund[:, k]]
            dg_prev = y_dg[:, k]
            y_eq = price[:, k, I["us_equity"]]
            sv = np.sqrt((y_eq ** 2 + y_prev ** 2) / 2.0)
            y_prev = y_eq
            cand_states = {"s_vol": sv, "s_pd": cand_states["s_pd"] + y_eq - y_dg[:, k]}
        shift = np.zeros_like(price)
        stmts = [s for s in statements if s.active]
        if stmts:
            price, shift = val.apply_statements(price, y_fund, stmts, bins=self.valuation.bins,
                                                resid_beta=self.pool.resid_beta(), rng=rng,
                                                log_cape0=start["s_cape"], m_star=self.valuation.m_star)
        y_us = price[:, :, I["us_equity"]]
        y_mult = y_us - y_fund
        y_pd = y_us - y_dg
        # ---- income: each dividend payer's own P/D at the start of every quarter
        income = np.zeros_like(price)
        for a in DIVIDEND_ASSETS:
            lpd = per_path(start[f"s_pd_{a}"])
            for k in range(H):
                income[:, k, I[a]] = self.income_k[a] * np.exp(-lpd) / 4.0
                g = y_dg[:, k] if a == "us_equity" else self.div_growth[a]
                lpd = lpd + price[:, k, I[a]] - g
        income[:, :, I["treasury_10y"]] = mech["treasury_10y"][1]
        income[:, :, I["cash"]] = np.expm1(price[:, :, I["cash"]])
        total = np.log(np.exp(price) + income)
        total[:, :, I["cash"]] = price[:, :, I["cash"]]
        log_cape, log_pd = val.implied_paths(start["s_cape"], start["s_pd_us_equity"], y_mult, y_pd)
        return SimPaths(quarters=labels, assets=list(ASSETS), price=price, income=income, total=total,
                        systematic=sysm, resid=resid, shift=shift, y_fund=y_fund, y_mult=y_mult, y_dg=y_dg,
                        y_pd=y_pd, log_cape=log_cape, log_pd=log_pd, s_vol=s_vol, drawn=self.pool.label_array(idx),
                        param_index=pidx, cma=cma, statements=list(stmts))

    def next_state(self, start: dict, sim: SimPaths) -> dict:
        """The start state of the quarter after a simulated window, per path (start_state's keys), so step mode can
        chain one-quarter simulate() calls: s_vol from the last two us_equity returns, s_cape and every log P/D rolled
        through the window, and the legs' own lags."""
        n, H = sim.price.shape[:2]

        def per_path(v):
            return np.broadcast_to(np.asarray(v, float), (n,)).copy()

        y_us = sim.price[:, :, I["us_equity"]]
        prev = y_us[:, -2] if H >= 2 else per_path(start["y_eq_l1"])
        lags = [sim.y_fund[:, -2], sim.y_fund[:, -1]] if H >= 2 else [per_path(start["y_fund_lags"][1]),
                                                                        sim.y_fund[:, -1]]
        st = {"s_vol": np.sqrt((y_us[:, -1] ** 2 + prev ** 2) / 2.0), "y_eq_l1": y_us[:, -1].copy(),
              "s_cape": sim.log_cape[:, -1].copy(), "y_dg_l1": sim.y_dg[:, -1].copy(), "y_fund_lags": lags}
        for a in DIVIDEND_ASSETS:
            g = sim.y_dg if a == "us_equity" else self.div_growth[a]
            st[f"s_pd_{a}"] = per_path(start[f"s_pd_{a}"]) + (sim.price[:, :, I[a]] - g).sum(axis=1)
        return st

    # ------------------------------------------------------------ export
    def summary(self) -> dict:
        """A JSON-ready description of the fit (qmodel.json's asset-layer section)."""
        return {"version": VERSION, "train_end": self.train_end, "spec": self.spec.to_dict(),
                "features": self.features, "fields": {f: FIELD_OF[f] for f in self.features},
                "alpha_monthly": self.alpha[RISK].to_dict(), "kappa_q": self.kappa.to_dict(),
                "conversion_q": self.conv.to_dict(), "c_log_monthly": self.c_log[RISK].to_dict(),
                "beta_fields": self.beta_fields().to_dict(orient="index"), "calibration_quarters": self.calib_quarters,
                "fit_months": self.n_fit_months, "pool": self.pool.summary(),
                "cma": {k: v.to_dict() for k, v in self.cma_adj.items()}, "cma_detail": self.cma_detail,
                "income_k": self.income_k.to_dict(), "div_growth_q": self.div_growth.to_dict(),
                "valuation": self.valuation.summary(), "candidates": list(self.candidates),
                "param_draws": len(self.params) if self.params is not None else 0, **self.fit_info}


def oracle_inputs(Q: pd.DataFrame, quarters, leads: bool = False) -> pd.DataFrame:
    """The realized macro inputs of the given quarters (the oracle, the "perfect translation"): the x and x_aux
    fields, the macro start states the layer reads (s_core4, s_zlb, s_y10, s_tb3, s_ff) and the quarter-end bill rate.
    No asset-derived column (y_, yt_, inc_, s_pd, s_cape, s_vol, momentum) is read. leads: + d_ip_lead and
    d_unrate_lead, the next quarter's realized values (candidate C3's path-mode terms; the last quarter's lead lies
    after the window)."""
    qs = pd.PeriodIndex([pd.Period(str(q), "Q") for q in quarters], freq="Q")
    xcols = [c for c in Q.columns if c.startswith(("d_", "x_")) and not c.endswith("__known")]
    cols = xcols + ["s_core4", "s_zlb", "s_y10", "s_tb3", "s_ff", "lvl_tb3"]
    out = Q.reindex(qs)[cols].copy()
    if "x_oil_cause" in out:
        out["x_oil_cause"] = out["x_oil_cause"].astype(float)
    if leads:
        out["d_ip_lead"] = Q["d_ip"].reindex(qs + 1).to_numpy(float)
        out["d_unrate_lead"] = Q["d_unrate"].reindex(qs + 1).to_numpy(float)
    return out
