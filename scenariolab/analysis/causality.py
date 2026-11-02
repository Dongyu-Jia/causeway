"""Causal identification: sign-restricted SVAR, distribution local projections, betas.

a) Structural VAR on y = [oil, ip, cpi, ff, spread] (monthly; oil/ip are
   100 x log changes, cpi is 1200 x log change = annualized monthly inflation
   in %, ff and spread are levels in %), 6 lags + constant, OLS. Four shocks are
   set-identified by sign restrictions (Rubio-Ramírez, Waggoner & Zha 2010:
   Haar-random rotations of the Cholesky factor, accepted when every
   restricted shock gets its own column, sign flips allowed). Restrictions
   apply to the reported responses: cumulative levels for oil/ip/cpi, levels
   for ff/spread. Bands combine parameter uncertainty (Normal-inverse-Wishart
   posterior draws under a flat prior) and rotation uncertainty. The point
   model is the Fry-Pagan median-target rotation at the OLS estimates; its
   shocks e_t = B^-1 u_t go to processed/structural_shocks.csv and drive the
   historical decomposition of 12-month CPI inflation.
   COVID months 2020-03..2020-06 (and the rows that use them as lags) are left
   out of estimation and blank in the shock file; in the decomposition their
   residuals form a separate "covid" bucket.
b) Quarterly local projections (Jordà 2005) of a labor-share shock on
   consumption, GDP, CPI, profits, real equity value, unemployment and the FF
   rate, 1960-1999 vs 2000-today, Newey-West 90% bands, scaled to a 1% decline.
   The shock is the labor-share change not explained by same-quarter GDP
   growth, inflation and the FF rate, nor by four lags of all four (labor share
   ordered last); the lags-only residual is reported as an alternative.
c) Standardized factor betas of the engine's asset model per period.
d) Curated causal chains whose edge evidence is filled from a)-c).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config, jsonio
from ..context import SHOCKS_PATH
from ..data import sources as src
from ..engine.factor_model import AssetFactorModel, ModelSpec, design_matrix

VARIABLES = [
    {"id": "oil", "name": "油价", "unit": "累计 %"},
    {"id": "ip", "name": "工业产出", "unit": "累计 %"},
    {"id": "cpi", "name": "CPI 价格水平", "unit": "累计 %"},
    {"id": "ff", "name": "联邦基金利率", "unit": "pp"},
    {"id": "spread", "name": "信用利差(BAA-10Y)", "unit": "pp"},
]
VAR_IDS = [v["id"] for v in VARIABLES]
# Reported response = cumsum(monthly response) * scale for flows, the response itself for levels.
CUM_SCALE = np.array([1.0, 1.0, 1.0 / 12.0, 0.0, 0.0])

SHOCKS = [
    {"id": "oil_supply", "name": "石油供给冲击",
     "restrictions": [("oil", 1, 0, 2), ("ip", -1, 0, 2), ("cpi", 1, 0, 2)]},
    {"id": "demand", "name": "总需求冲击",
     "restrictions": [("oil", 1, 0, 2), ("ip", 1, 0, 2), ("cpi", 1, 0, 2), ("ff", 1, 0, 2)]},
    {"id": "monetary", "name": "货币政策冲击（紧缩）",
     "restrictions": [("ff", 1, 0, 2), ("cpi", -1, 0, 5), ("ip", -1, 0, 5), ("spread", 1, 0, 2)]},
    {"id": "financial", "name": "金融/风险冲击",
     "restrictions": [("spread", 1, 0, 2), ("ip", -1, 0, 2), ("ff", -1, 0, 2)]},
]
SHOCK_IDS = [s["id"] for s in SHOCKS]
SHOCK_NAMES = {s["id"]: s["name"] for s in SHOCKS}

PERIODS = [
    {"id": "full", "name": "1973 年以来全样本", "start": "1973-01", "end": None},
    {"id": "oil_era", "name": "石油与通胀时代", "start": "1973-01", "end": "1985-12"},
    {"id": "great_moderation", "name": "大缓和", "start": "1986-01", "end": "2007-12"},
    {"id": "post_gfc", "name": "QE与疫情后", "start": "2008-01", "end": None},
]
PERIOD_LABEL = {"full": "1973 年以来", "oil_era": "1973-85", "great_moderation": "1986-2007", "post_gfc": "2008 后"}

LAGS = 6
HORIZON = 36
OUTLIERS = ("2020-03", "2020-06")
N_ACCEPT = 500                  # accepted draws per period for the bands
N_ACCEPT_MT = 1000              # rotation-only draws at the OLS estimates for the median target
ROTATIONS_PER_DRAW = 25         # rotations per posterior draw (about 1-2 accepted each)
MAX_ROTATIONS = 2_000_000

# distribution block (quarterly)
LP_PERIODS = [
    {"id": "p1960_1999", "name": "1960-1999", "start": "1960-01", "end": "1999-12"},
    {"id": "p2000_now", "name": "2000 年以来", "start": "2000-01", "end": None},
]
LP_HORIZON = 12
LP_LAGS = 4
LP_OUTLIERS = ("2020-01", "2020-12")
LP_VARS = [
    {"id": "consumption", "name": "实际消费", "unit": "累计 %"},
    {"id": "gdp", "name": "实际 GDP", "unit": "累计 %"},
    {"id": "cpi", "name": "CPI 价格水平", "unit": "累计 %"},
    {"id": "profit_share", "name": "企业税后利润/GDP", "unit": "pp"},
    {"id": "equity_real", "name": "美股实际价值（含股息）", "unit": "累计 %"},
    {"id": "unrate", "name": "失业率", "unit": "pp"},
    {"id": "ff", "name": "联邦基金利率", "unit": "pp"},
    {"id": "labor_share", "name": "劳动收入占比", "unit": "累计 %"},
]
LP_CONTROLS = ["labor_share", "gdp", "cpi", "ff"]       # their changes (ff: level) enter with 4 lags
# The labor share is nominal pay over nominal output, so with sticky pay it falls
# mechanically in any quarter when output or prices surprise upward. A lags-only
# residual therefore mostly measures the business cycle ("labor share falls ->
# boom"). The baseline orders the labor share after same-quarter GDP growth,
# inflation and the FF rate (recursive identification); the lags-only version is
# reported as an alternative.
LP_CONTEMPORANEOUS = ("gdp", "cpi", "ff")
Z90 = 1.6448536269514722


# ================================================================ reduced-form VAR

@dataclass
class VAR:
    coef: np.ndarray            # (1 + n*p, n): constant row, then lag-1 block, ..., lag-p block
    sigma: np.ndarray           # (n, n) residual covariance, dof-adjusted
    resid: np.ndarray           # (T, n) residuals of the estimation rows
    X: np.ndarray               # (T, 1 + n*p) regressors of the estimation rows
    dates: pd.DatetimeIndex     # estimation rows
    lags: int
    Y_all: pd.DataFrame         # every row with complete data and lags (incl. excluded ones)
    X_all: pd.DataFrame

    @property
    def n(self) -> int:
        return self.coef.shape[1]


def var_data(panel: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Monthly VAR variables in the units described in the module docstring."""
    D = pd.DataFrame({"oil": 100.0 * panel["f_oil"], "ip": 100.0 * panel["f_ip"],
                      "cpi": 1200.0 * panel["f_cpi"], "ff": panel["s_ff"], "spread": panel["s_spread"]})
    return D.loc[start:end]


def lagged(data: pd.DataFrame, p: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Y, X) for rows p.. of `data`; X = [const, y_{t-1}, ..., y_{t-p}] from inside the window only."""
    parts = [pd.Series(1.0, index=data.index, name="const")]
    for i in range(1, p + 1):
        parts.append(data.shift(i).add_suffix(f"_l{i}"))
    X = pd.concat(parts, axis=1).iloc[p:]
    Y = data.iloc[p:]
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    return Y[ok], X[ok]


def outlier_rows(index: pd.DatetimeIndex, p: int, outliers=OUTLIERS) -> np.ndarray:
    """Rows whose own month or any of their p lags falls inside the outlier window."""
    if not outliers:
        return np.zeros(len(index), bool)
    lo = pd.Timestamp(outliers[0])
    hi = pd.Timestamp(outliers[1]) + pd.DateOffset(months=p)
    return np.asarray((index >= lo) & (index <= hi))


def fit_var(data: pd.DataFrame, p: int = LAGS, outliers=OUTLIERS) -> VAR:
    Y_all, X_all = lagged(data, p)
    use = ~outlier_rows(Y_all.index, p, outliers)
    Y, X = Y_all.to_numpy(float)[use], X_all.to_numpy(float)[use]
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ coef
    sigma = resid.T @ resid / (len(Y) - X.shape[1])
    return VAR(coef=coef, sigma=sigma, resid=resid, X=X, dates=Y_all.index[use], lags=p,
               Y_all=Y_all, X_all=X_all)


def lag_matrices(coef: np.ndarray, p: int) -> np.ndarray:
    """(p, n, n) with y_t = c + sum_i A_i y_{t-i}."""
    n = coef.shape[1]
    return np.stack([coef[1 + i * n: 1 + (i + 1) * n].T for i in range(p)])


def ma_coefs(coef: np.ndarray, p: int, H: int) -> np.ndarray:
    """Reduced-form MA coefficients Psi_0..Psi_H, shape (H+1, n, n)."""
    A = lag_matrices(coef, p)
    n = coef.shape[1]
    psi = np.zeros((H + 1, n, n))
    psi[0] = np.eye(n)
    for h in range(1, H + 1):
        for i in range(1, min(h, p) + 1):
            psi[h] += A[i - 1] @ psi[h - i]
    return psi


def max_root(coef: np.ndarray, p: int) -> float:
    n = coef.shape[1]
    comp = np.zeros((n * p, n * p))
    comp[:n] = np.concatenate(list(lag_matrices(coef, p)), axis=1)
    comp[n:, :-n] = np.eye(n * (p - 1))
    return float(np.abs(np.linalg.eigvals(comp)).max())


def posterior_draw(var: VAR, rng: np.random.Generator, xtx_chol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(coef, sigma) from the Normal-inverse-Wishart posterior under a flat prior."""
    T, k = var.X.shape
    n = var.n
    S = var.resid.T @ var.resid
    Z = rng.standard_normal((T - k, n)) @ np.linalg.cholesky(np.linalg.inv(S)).T
    sigma = np.linalg.inv(Z.T @ Z)
    coef = var.coef + xtx_chol @ rng.standard_normal((k, n)) @ np.linalg.cholesky(sigma).T
    return coef, sigma


# ================================================================ sign restrictions

def to_reported(irf: np.ndarray, scale: np.ndarray = CUM_SCALE) -> np.ndarray:
    """(..., H+1, n, m) monthly responses -> cumulative levels for flows, levels for stocks."""
    cum = np.cumsum(irf, axis=-3) * scale[:, None]
    return np.where((scale > 0)[:, None], cum, irf)


def sign_masks(shocks: list[dict] = SHOCKS, var_ids: list[str] = VAR_IDS) -> np.ndarray:
    """(K, H_r, n) array of +1 / -1 / 0 (unrestricted) over horizons 0..H_r-1."""
    hr = 1 + max(r[3] for s in shocks for r in s["restrictions"])
    S = np.zeros((len(shocks), hr, len(var_ids)))
    for k, s in enumerate(shocks):
        for var, sign, h0, h1 in s["restrictions"]:
            S[k, h0:h1 + 1, var_ids.index(var)] = sign
    return S


def haar_rotations(rng: np.random.Generator, N: int, n: int) -> np.ndarray:
    """N Haar-distributed n x n orthogonal matrices (QR of Gaussian draws, diag(R) > 0)."""
    Q, R = np.linalg.qr(rng.standard_normal((N, n, n)))
    d = np.sign(np.diagonal(R, axis1=1, axis2=2))
    d[d == 0] = 1.0
    return Q * d[:, None, :]


def column_signs(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """For candidate responses R (N, H_r, n, n_cols): +1/-1 if a column (or its negative)
    satisfies shock k's restrictions, else 0. Returns (N, K, n_cols)."""
    active = (S != 0)[None, :, :, :, None]
    prod = S[None, :, :, :, None] * R[:, None, :, :, :]
    pos = np.where(active, prod > 0, True).all(axis=(2, 3))
    neg = np.where(active, prod < 0, True).all(axis=(2, 3))
    return pos.astype(np.int8) - neg.astype(np.int8)


def identify_batch(psi: np.ndarray, chol: np.ndarray, rng: np.random.Generator, N: int,
                   S: np.ndarray, perms: np.ndarray) -> list[np.ndarray]:
    """Draw N rotations; return the structural impact matrices (n, n) of the accepted ones.

    Columns 0..K-1 are the restricted shocks in order, the rest the unrestricted ones.
    """
    n = chol.shape[0]
    K, hr = S.shape[0], S.shape[1]
    B = chol[None] @ haar_rotations(rng, N, n)                       # (N, n, n)
    R = to_reported(np.einsum("hij,njk->nhik", psi[:hr], B))          # (N, hr, n, n)
    signs = column_signs(R, S)                                        # (N, K, n)
    picked = signs[:, np.arange(K)[None, :], perms]                   # (N, n_perm, K)
    valid = (picked != 0).all(axis=2)
    out = []
    for d in np.flatnonzero(valid.any(axis=1)):
        j = rng.choice(np.flatnonzero(valid[d]))
        cols = perms[j]
        rest = [c for c in range(n) if c not in cols]
        M = np.empty((n, n))
        M[:, :K] = B[d][:, cols] * picked[d, j][None, :]
        M[:, K:] = B[d][:, rest]
        out.append(M)
    return out


def sample_svar(var: VAR, rng: np.random.Generator, n_accept: int = N_ACCEPT, horizon: int = HORIZON,
                param_uncertainty: bool = True, rotations: int = ROTATIONS_PER_DRAW,
                max_rotations: int = MAX_ROTATIONS, shocks: list[dict] = SHOCKS) -> dict:
    """Accepted draws of the identified model.

    Returns irf (N, H+1, n, K) reported responses per 1-sd shock, impact (N, n, n),
    plus counters. With param_uncertainty=False only rotations are drawn (OLS
    reduced form), which the median-target model needs.
    """
    S = sign_masks(shocks)
    K = S.shape[0]
    perms = np.array(list(itertools.permutations(range(var.n), K)))
    xtx_chol = np.linalg.cholesky(np.linalg.inv(var.X.T @ var.X))
    irfs, impacts = [], []
    tried = param_draws = explosive = 0
    psi_ols = ma_coefs(var.coef, var.lags, horizon)
    chol_ols = np.linalg.cholesky(var.sigma)
    while len(irfs) < n_accept and tried < max_rotations:
        if param_uncertainty:
            coef, sigma = posterior_draw(var, rng, xtx_chol)
            param_draws += 1
            if max_root(coef, var.lags) >= 1.0:
                explosive += 1
                if explosive > 50 * max(param_draws - explosive, 1) and explosive > 2000:
                    break
                continue
            psi, chol = ma_coefs(coef, var.lags, horizon), np.linalg.cholesky(sigma)
        else:
            psi, chol = psi_ols, chol_ols
        tried += rotations
        for M in identify_batch(psi, chol, rng, rotations, S, perms):
            impacts.append(M)
            irfs.append(to_reported(psi @ M[None])[:, :, :K])
    irfs = np.array(irfs[:n_accept]) if irfs else np.zeros((0, horizon + 1, var.n, K))
    return {"irf": irfs, "impact": np.array(impacts[:n_accept]), "n_accepted": int(len(irfs)),
            "n_rotations": int(tried), "n_param_draws": int(param_draws), "n_explosive": int(explosive),
            "acceptance_rate": float(len(irfs) / tried) if tried else 0.0}


def median_target(irfs: np.ndarray) -> int:
    """Fry-Pagan: the draw whose standardized responses are closest to the pointwise median."""
    med = np.median(irfs, axis=0)
    sd = irfs.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return int((((irfs - med) / sd) ** 2).sum(axis=(1, 2, 3)).argmin())


def band_summary(irfs: np.ndarray, shock_ids=SHOCK_IDS, var_ids=VAR_IDS) -> dict:
    if len(irfs) == 0:
        return {}
    q = np.percentile(irfs, [16, 50, 84], axis=0)                    # (3, H+1, n, K)
    return {s: {v: {"p16": q[0, :, i, k], "p50": q[1, :, i, k], "p84": q[2, :, i, k]}
                for i, v in enumerate(var_ids)} for k, s in enumerate(shock_ids)}


def structural_shocks(var: VAR, impact: np.ndarray, standardize: bool = True) -> pd.DataFrame:
    """e_t = B^-1 u_t for every row with data and lags, columns in impact-column order.

    standardize=True scales to unit variance over the estimation rows and blanks
    the excluded (outlier) rows: a linear model reads the COVID collapse as
    10-20 sd "oil" or "monetary" shocks, which would dominate any regression on
    these series. standardize=False returns the raw shocks for every row.
    """
    U = var.Y_all.to_numpy(float) - var.X_all.to_numpy(float) @ var.coef
    E = pd.DataFrame(U @ np.linalg.inv(impact).T, index=var.Y_all.index)
    if standardize:
        est = np.isin(E.index, var.dates)
        E = (E / E[est].std(ddof=0)).where(pd.Series(est, index=E.index), axis=0)
    return E


def historical_decomposition(var: VAR, impact: np.ndarray, shocks: pd.DataFrame,
                             special: np.ndarray | None = None) -> dict:
    """y_t = baseline_t + sum_j contrib_j,t over the rows of `shocks` (consecutive months).

    baseline: constant + initial conditions propagated without shocks.
    contrib: (n + 1, T, n); bucket j < n is structural shock j, bucket n holds
    the whole residual of the `special` rows (e.g. the excluded COVID months),
    whose structural split is not trusted. `shocks` must be the raw B^-1 u_t.
    """
    gaps = shocks.index[1:] != shocks.index[:-1] + pd.DateOffset(months=1)
    if gaps.any():
        raise ValueError("historical decomposition needs consecutive months")
    A = lag_matrices(var.coef, var.lags)
    c = var.coef[0]
    p, n = var.lags, var.n
    E = shocks.to_numpy(float)
    T = len(E)
    special = np.zeros(T, bool) if special is None else np.asarray(special, bool)
    Y = var.Y_all.loc[shocks.index].to_numpy(float)
    first = var.Y_all.index.get_loc(shocks.index[0])
    init = var.X_all.iloc[first].to_numpy(float)[1:].reshape(p, n)    # y_{t-1}, ..., y_{t-p}
    base = np.zeros((T + p, n))
    base[:p] = init[::-1]
    contrib = np.zeros((n + 1, T + p, n))
    for t in range(T):
        base[p + t] = c + sum(A[i] @ base[p + t - 1 - i] for i in range(p))
        impulse = np.zeros((n + 1, n))
        if special[t]:
            impulse[n] = impact @ E[t]
        else:
            impulse[:n] = (impact * E[t][None, :]).T
        contrib[:, p + t] = sum(contrib[:, p + t - 1 - i] @ A[i].T for i in range(p)) + impulse
    return {"actual": Y, "baseline": base[p:], "contrib": contrib[:, p:]}


def svar_period(panel: pd.DataFrame, period: dict, rng: np.random.Generator, n_accept: int = N_ACCEPT,
                horizon: int = HORIZON) -> tuple[VAR, dict]:
    var = fit_var(var_data(panel, period["start"], period["end"]))
    draws = sample_svar(var, rng, n_accept=n_accept, horizon=horizon)
    return var, draws


# ================================================================ distribution block (quarterly LP)

def quarter_mean(s: pd.Series) -> pd.Series:
    """Quarterly average (index = quarter start); NaN for quarters with fewer than 3 months of data."""
    r = s.resample("QS")
    return r.mean().where(r.count() == 3)


def quarterly_data(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Quarterly series (index = quarter start). Log series are 100 x log levels."""
    f = lambda sid: src.fred(sid, refresh=refresh)  # noqa: E731
    qs = quarter_mean
    m = panel[["lvl_cpi", "lvl_ff", "lvl_unrate"]]
    real_ret = np.log1p(panel["r_us_equity"]) - panel["f_cpi"]
    eq = real_ret.resample("QS").sum(min_count=3)
    Q = pd.DataFrame({
        "labor_share": 100.0 * np.log(f("PRS85006173")),
        "gdp": 100.0 * np.log(f("GDPC1")),
        "consumption": 100.0 * np.log(f("PCECC96")),
        "profit_share": 100.0 * f("CP") / f("GDP"),
        "cpi": 100.0 * np.log(qs(m["lvl_cpi"])),
        "ff": qs(m["lvl_ff"]),
        "unrate": qs(m["lvl_unrate"]),
        "equity_real": 100.0 * eq.cumsum().where(eq.notna()),
    })
    return Q.sort_index()


def newey_west(y: np.ndarray, X: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (Bartlett, L lags) standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    g = X * (y - X @ beta)[:, None]
    S = g.T @ g
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        G = g[lag:].T @ g[:-lag]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.maximum(np.diag(V), 0.0))


def _changes(Q: pd.DataFrame, cols) -> pd.DataFrame:
    return pd.DataFrame({c: (Q[c] if c == "ff" else Q[c].diff()) for c in cols})


def lp_controls(Q: pd.DataFrame, lags: int = LP_LAGS) -> pd.DataFrame:
    base = _changes(Q, LP_CONTROLS)
    return pd.concat({f"{c}_l{i}": base[c].shift(i) for i in range(1, lags + 1) for c in LP_CONTROLS}, axis=1)


def outlier_quarters(index: pd.DatetimeIndex, outliers=LP_OUTLIERS) -> pd.Series:
    if not outliers:
        return pd.Series(False, index=index)
    return pd.Series((index >= pd.Timestamp(outliers[0])) & (index <= pd.Timestamp(outliers[1])), index=index)


def labor_share_shock(Q: pd.DataFrame, start, end, contemporaneous=LP_CONTEMPORANEOUS, lags: int = LP_LAGS,
                      outliers=LP_OUTLIERS) -> pd.Series:
    """Residual of 100 x dlog(labor share) on a constant, 4 lags of itself, real GDP growth,
    CPI inflation and the FF rate, and the same-quarter values of `contemporaneous`,
    estimated within [start, end]. Units: % of the labor-share index.

    Quarters in the outlier window, or whose changes or lags reach into it, get no shock.
    """
    parts = [Q["labor_share"].diff().rename("y"), lp_controls(Q, lags)]
    if contemporaneous:
        parts.append(_changes(Q, contemporaneous).add_suffix("_l0"))
    D = pd.concat(parts, axis=1).loc[start:end].dropna()
    bad = outlier_quarters(Q.index, outliers)
    touched = pd.concat([bad.shift(i, fill_value=False) for i in range(lags + 2)], axis=1).any(axis=1)
    D = D[~touched.reindex(D.index).fillna(False).to_numpy(bool)]
    X = np.column_stack([np.ones(len(D)), D.drop(columns="y").to_numpy(float)])
    beta = np.linalg.lstsq(X, D["y"].to_numpy(float), rcond=None)[0]
    return pd.Series(D["y"].to_numpy(float) - X @ beta, index=D.index, name="shock")


def local_projection(Q: pd.DataFrame, shock: pd.Series, var: str, horizon: int = LP_HORIZON,
                     lags: int = LP_LAGS, end=None, outliers=LP_OUTLIERS) -> dict:
    """Response of var_{t+h} - var_{t-1} to a 1-unit shock_t, h = 0..horizon.

    Controls: the shock-regression lags plus `lags` lags of the outcome's own
    change. Outcome windows [t-1, t+h] must end inside the period (`end`) and
    avoid the outlier window. Returns coef, se (Newey-West, h+1 lags), n per h.
    """
    level = Q[var]
    C = lp_controls(Q, lags)
    if var not in LP_CONTROLS:
        own = level.diff()
        C = pd.concat([C, pd.concat({f"own_l{i}": own.shift(i) for i in range(1, lags + 1)}, axis=1)], axis=1)
    bad = outlier_quarters(Q.index, outliers).astype(float)
    end_ts = pd.Timestamp(end) if end else Q.index.max()
    coef, se, nobs = [], [], []
    for h in range(horizon + 1):
        target = (level.shift(-h) - level.shift(1)).rename("y")
        window_bad = sum(bad.shift(-j, fill_value=0.0) for j in range(-1, h + 1)) > 0
        D = pd.concat([target, shock.rename("shock"), C], axis=1).loc[shock.index].dropna()
        D = D[(D.index <= end_ts - pd.DateOffset(months=3 * h))
              & ~window_bad.reindex(D.index).fillna(True).to_numpy(bool)]
        nobs.append(int(len(D)))
        if len(D) < C.shape[1] + 10:
            coef.append(np.nan)
            se.append(np.nan)
            continue
        X = np.column_stack([np.ones(len(D)), D.drop(columns="y").to_numpy(float)])
        b, s = newey_west(D["y"].to_numpy(float), X, h + 1)
        coef.append(float(b[1]))
        se.append(float(s[1]))
    return {"coef": np.array(coef), "se": np.array(se), "n": nobs}


def lp_responses(Q: pd.DataFrame, contemporaneous) -> tuple[dict, dict, dict, list[dict]]:
    """Responses to a 1% labor-share decline per LP period: coef, 90% band, se."""
    responses, n_obs, shock_sd, periods = {}, {}, {}, []
    for per in LP_PERIODS:
        end = per["end"] or Q.index.max().strftime("%Y-%m")
        shock = labor_share_shock(Q, per["start"], end, contemporaneous)
        shock_sd[per["id"]] = float(shock.std())
        out, nn = {}, {}
        for v in LP_VARS:
            r = local_projection(Q, shock, v["id"], end=end)
            coef = -r["coef"]
            out[v["id"]] = {"coef": coef, "lo": coef - Z90 * r["se"], "hi": coef + Z90 * r["se"], "se": r["se"]}
            nn[v["id"]] = r["n"]
        responses[per["id"]] = out
        n_obs[per["id"]] = nn
        periods.append({"id": per["id"], "name": per["name"], "start": per["start"], "end": end,
                        "shock_start": shock.index.min().strftime("%Y-%m"),
                        "shock_end": shock.index.max().strftime("%Y-%m")})
    return responses, n_obs, shock_sd, periods


def distribution_block(panel: pd.DataFrame, Q: pd.DataFrame | None = None) -> dict:
    Q = quarterly_data(panel) if Q is None else Q
    responses, n_obs, shock_sd, periods = lp_responses(Q, LP_CONTEMPORANEOUS)
    alt, _, alt_sd, _ = lp_responses(Q, ())
    return {"shock": "劳动收入占比下降 1%", "horizon_q": LP_HORIZON, "periods": periods,
            "variables": LP_VARS, "responses": responses, "n_obs": n_obs, "shock_sd_pct": shock_sd,
            "band": 0.9,
            "method": "Jordà 局部投影。冲击单位：劳动收入占比指数相对下降 1%（不是 1 个百分点）。"
                      "冲击 = 劳动收入占比季度变化中，不能被同季度实际 GDP 增长、CPI 通胀、"
                      "联邦基金利率以及它们和劳动收入占比自身过去 4 季解释的部分（递归识别：劳动收入占比排在最后，"
                      "只能滞后一季影响宏观）。Newey-West 标准误，90% 区间；2020 年疫情季度及跨越它的窗口剔除。",
            "alternative": {"id": "lags_only", "name": "只控制滞后项（劳动收入占比排在最前）",
                            "note": "同季度产出或物价意外上升会机械地压低劳动收入占比，所以这种冲击主要反映经济周期，"
                                    "不宜解读为分配冲击的因果效应。",
                            "shock_sd_pct": alt_sd,
                            "responses": {p: {v: {k: r[k] for k in ("coef", "lo", "hi")} for v, r in rv.items()}
                                          for p, rv in alt.items()}}}


# ================================================================ asset betas

def asset_betas(panel: pd.DataFrame, periods: list[dict] = PERIODS) -> dict:
    """Standardized betas coef * sd(factor) / sd(asset price return) of the default asset model.

    Always the base spec (one fixed set of core factors, every asset on the same equation and window), so the
    periods compare like with like; the scenarios run the walk-forward's selected model, whose term-premium term
    is zero in periods without high core inflation. `model` says which model this is (the page names it)."""
    spec = ModelSpec()
    last = panel.index.max().strftime("%Y-%m")
    out = {"periods": [{"id": p["id"], "name": p["name"], "start": p["start"], "end": p["end"] or last}
                       for p in periods],
           "model": {"describe": spec.describe(), "spec": spec.to_dict(), "base": True},
           "factors": list(spec.factors), "assets": [], "beta_std": {}, "tstat": {}, "r2": {}}
    X_all = design_matrix(panel, spec)
    for per in periods:
        m = AssetFactorModel(spec).fit(panel, per["start"], per["end"], honor_asset_start=False)
        P = panel.loc[m.train_start:m.train_end]
        X = X_all.loc[P.index]
        Y = P[m.target_columns()]
        ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
        sd_x = X[ok].std(ddof=0).to_numpy()
        sd_y = Y[ok].std(ddof=0).to_numpy()
        beta = m.coef.to_numpy() * sd_x[None, :] / sd_y[:, None]
        out["assets"] = m.assets
        out["beta_std"][per["id"]] = beta
        out["tstat"][per["id"]] = m.tstat.to_numpy()
        out["r2"][per["id"]] = dict(m.stats.r2)
    return out


# ================================================================ causal chains

SUB_PERIODS = ("oil_era", "great_moderation", "post_gfc")


def _irf_at(svar: dict, period: str, shock: str, var: str, h: int, pct: str = "p50") -> float:
    try:
        return float(svar["irf"][period][shock][var][pct][h])
    except (KeyError, IndexError, TypeError):
        return float("nan")


def _fmt(x: float, unit: str = "%", digits: int = 2) -> str:
    if not np.isfinite(x):
        return "n/a"
    if round(float(x), digits) == 0.0:                  # neither "+0.00%" nor "-0.00%": a zero has no sign
        return f"{0.0:.{digits}f}{unit}"
    return f"{x:+.{digits}f}{unit}"


def _band(mid: float, lo: float, hi: float, unit: str, level: str, digits: int = 2) -> str:
    return f"{_fmt(mid, unit, digits)}（{level} 区间 {_fmt(lo, unit, digits)}~{_fmt(hi, unit, digits)}）"


def _irf_text(svar: dict, shock: str, var: str, h: int, unit: str, digits: int = 2) -> str:
    return "；".join(
        f"{PERIOD_LABEL[p]} " + _band(_irf_at(svar, p, shock, var, h), _irf_at(svar, p, shock, var, h, "p16"),
                                      _irf_at(svar, p, shock, var, h, "p84"), unit, "68%", digits)
        for p in SUB_PERIODS)


def _irf_sign(svar: dict, shock: str, var: str, h: int) -> str:
    vals = [_irf_at(svar, p, shock, var, h) for p in SUB_PERIODS]
    return "+" if all(v > 0 for v in vals) else ("-" if all(v < 0 for v in vals) else "±")


def _restriction_note(shock: str, var: str) -> str:
    """Says whether the sign of a later-horizon response was partly imposed by the identification."""
    for v, sign, h0, h1 in next(s for s in SHOCKS if s["id"] == shock)["restrictions"]:
        if v == var:
            return f"（第 {h0}-{h1} 月{'上升' if sign > 0 else '下降'}是识别约束，之后由数据决定）"
    return "（未施加约束，由数据决定）"


def _norm_text(svar: dict, key: str, unit: str) -> str:
    return "；".join(f"{PERIOD_LABEL[p]} " + _band(*(svar["normalized"][p][key][q] for q in ("p50", "p16", "p84")),
                                                   unit, "68%") for p in SUB_PERIODS)


def _lp_at(dist: dict, period: str, var: str, h: int) -> tuple[float, float, float]:
    r = dist["responses"][period][var]
    return float(r["coef"][h]), float(r["lo"][h]), float(r["hi"][h])


def _lp_text(dist: dict, var: str, h: int, unit: str) -> str:
    return "；".join(f"{per['name']} " + _band(*_lp_at(dist, per["id"], var, h), unit, "90%")
                     for per in dist["periods"])


def _lp_sign(dist: dict, var: str, h: int) -> str:
    vals = [_lp_at(dist, per["id"], var, h)[0] for per in dist["periods"]]
    return "+" if all(v > 0 for v in vals) else ("-" if all(v < 0 for v in vals) else "±")


def _beta(betas: dict, period: str, asset: str, factor: str) -> float:
    return float(betas["beta_std"][period][betas["assets"].index(asset)][betas["factors"].index(factor)])


def _beta_text(betas: dict, asset: str, factor: str) -> str:
    return "；".join(f"{PERIOD_LABEL[p]} {_fmt(_beta(betas, p, asset, factor), '', 2)}" for p in SUB_PERIODS)


def _beta_sign(betas: dict, asset: str, factor: str) -> str:
    vals = [_beta(betas, p, asset, factor) for p in SUB_PERIODS]
    return "+" if all(v > 0 for v in vals) else ("-" if all(v < 0 for v in vals) else "±")


def _node(nid: str, label: str, group: str) -> dict:
    return {"id": nid, "label": label, "group": group}


def _edge(a: str, b: str, sign: str, label: str, evidence: str) -> dict:
    return {"from": a, "to": b, "sign": sign, "label": label, "evidence": evidence}


def build_chains(svar: dict, dist: dict, betas: dict, panel: pd.DataFrame) -> list[dict]:
    """Curated causal chains; every evidence string is computed in this module."""
    o, m, f = "oil_supply", "monetary", "financial"
    chains = [{
        "id": "oil_supply_chain", "title": "石油供给冲击 → 通胀 → 联储 → 增长与资产",
        "nodes": [_node("shock", "石油供给冲击", "shock"), _node("oil", "油价上涨", "price"),
                  _node("cpi", "CPI 上升", "price"), _node("fed", "联储利率", "policy"),
                  _node("output", "工业产出下降", "macro"), _node("energy", "能源股", "asset"),
                  _node("equity", "美股大盘", "asset")],
        "edges": [
            _edge("shock", "oil", "+", "当月", "1σ 冲击当月油价：" + _irf_text(svar, o, "oil", 0, "%", 1)),
            _edge("oil", "cpi", _irf_sign(svar, o, "cpi", 12), "0-12 月",
                  "1σ 石油供给冲击 12 个月后 CPI 累计：" + _irf_text(svar, o, "cpi", 12, "%")
                  + "。按冲击当月油价 +10% 折算：" + _norm_text(svar, "cpi12_per_oil10", "%")
                  + _restriction_note(o, "cpi")),
            _edge("cpi", "fed", _irf_sign(svar, o, "ff", 12), "3-12 月",
                  "12 个月后联邦基金利率：" + _irf_text(svar, o, "ff", 12, "pp") + _restriction_note(o, "ff")),
            _edge("oil", "output", _irf_sign(svar, o, "ip", 24), "0-24 月",
                  "24 个月后工业产出累计：" + _irf_text(svar, o, "ip", 24, "%")
                  + "。按油价 +10% 折算：" + _norm_text(svar, "ip24_per_oil10", "%") + _restriction_note(o, "ip")),
            _edge("oil", "energy", _beta_sign(betas, "energy", "oil"), "同月",
                  "能源股对油价月度变动的标准化 β：" + _beta_text(betas, "energy", "oil")),
            _edge("oil", "equity", _beta_sign(betas, "us_equity", "oil"), "同月",
                  "美股大盘对油价月度变动的标准化 β：" + _beta_text(betas, "us_equity", "oil")),
        ]}]
    chains.append({
        "id": "monetary_chain", "title": "货币紧缩 → 信用利差 → 产出 → 股市",
        "nodes": [_node("shock", "货币紧缩冲击", "shock"), _node("ff", "联邦基金利率上升", "policy"),
                  _node("spread", "信用利差走阔", "finance"), _node("output", "工业产出下降", "macro"),
                  _node("cpi", "通胀回落", "price"), _node("equity", "美股大盘", "asset")],
        "edges": [
            _edge("shock", "ff", "+", "当月", "1σ 紧缩冲击当月利率：" + _irf_text(svar, m, "ff", 0, "pp")
                  + "（2008 后受零利率下限约束，利率本身几乎不动）"),
            _edge("ff", "spread", _irf_sign(svar, m, "spread", 6), "0-6 月",
                  "6 个月后信用利差：" + _irf_text(svar, m, "spread", 6, "pp") + _restriction_note(m, "spread")),
            _edge("spread", "output", _irf_sign(svar, m, "ip", 24), "6-24 月",
                  "24 个月后工业产出累计：" + _irf_text(svar, m, "ip", 24, "%") + _restriction_note(m, "ip")),
            _edge("output", "cpi", _irf_sign(svar, m, "cpi", 36), "12-36 月",
                  "36 个月后 CPI 累计：" + _irf_text(svar, m, "cpi", 36, "%") + _restriction_note(m, "cpi")),
            _edge("spread", "equity", _beta_sign(betas, "us_equity", "spread"), "同月",
                  "美股大盘对信用利差月度变动的标准化 β：" + _beta_text(betas, "us_equity", "spread")
                  + "；对 10 年期收益率变动：" + _beta_text(betas, "us_equity", "y10")),
        ]})
    chains.append({
        "id": "labor_share_chain", "title": "劳动收入占比下降 → 消费 → 增长、利润与利率",
        "nodes": [_node("shock", "劳动收入占比下降 1%", "shock"), _node("profit", "企业利润/GDP", "distribution"),
                  _node("cons", "实际消费", "macro"), _node("cpi", "物价", "price"),
                  _node("gdp", "实际 GDP", "macro"), _node("unemp", "失业率", "macro"),
                  _node("ff", "联邦基金利率", "policy"), _node("equity", "美股实际价值", "asset")],
        "edges": [
            _edge("shock", "profit", _lp_sign(dist, "profit_share", 4), "0-4 季",
                  "4 季后利润/GDP：" + _lp_text(dist, "profit_share", 4, "pp")),
            _edge("shock", "cons", _lp_sign(dist, "consumption", 8), "0-8 季",
                  "8 季后实际消费累计：" + _lp_text(dist, "consumption", 8, "%")),
            _edge("shock", "cpi", _lp_sign(dist, "cpi", 8), "0-8 季",
                  "8 季后 CPI 累计：" + _lp_text(dist, "cpi", 8, "%")),
            _edge("cons", "gdp", _lp_sign(dist, "gdp", 8), "0-8 季",
                  "8 季后实际 GDP 累计：" + _lp_text(dist, "gdp", 8, "%")),
            _edge("gdp", "unemp", _lp_sign(dist, "unrate", 8), "4-12 季",
                  "8 季后失业率：" + _lp_text(dist, "unrate", 8, "pp")),
            _edge("cpi", "ff", _lp_sign(dist, "ff", 8), "4-12 季",
                  "8 季后联邦基金利率：" + _lp_text(dist, "ff", 8, "pp")),
            _edge("profit", "equity", _lp_sign(dist, "equity_real", 8), "0-12 季",
                  "8 季后美股实际价值累计：" + _lp_text(dist, "equity_real", 8, "%")),
        ]})
    chains.append({
        "id": "financial_chain", "title": "金融冲击 → 产出下滑 → 联储降息",
        "nodes": [_node("shock", "金融/风险冲击", "shock"), _node("spread", "信用利差走阔", "finance"),
                  _node("output", "工业产出下降", "macro"), _node("ff", "联储降息", "policy"),
                  _node("cpi", "通胀", "price")],
        "edges": [
            _edge("shock", "spread", "+", "当月", "1σ 金融冲击当月信用利差：" + _irf_text(svar, f, "spread", 0, "pp")),
            _edge("spread", "output", _irf_sign(svar, f, "ip", 12), "0-24 月",
                  "12 个月后工业产出累计：" + _irf_text(svar, f, "ip", 12, "%") + _restriction_note(f, "ip")),
            _edge("output", "ff", _irf_sign(svar, f, "ff", 12), "0-12 月",
                  "12 个月后联邦基金利率：" + _irf_text(svar, f, "ff", 12, "pp") + _restriction_note(f, "ff")),
            _edge("output", "cpi", _irf_sign(svar, f, "cpi", 24), "12-36 月",
                  "24 个月后 CPI 累计：" + _irf_text(svar, f, "cpi", 24, "%") + _restriction_note(f, "cpi")),
        ]})
    chains.append(ai_chain(dist, betas, panel))
    return chains


def ai_chain(dist: dict, betas: dict, panel: pd.DataFrame) -> dict:
    """Descriptive chain: the AI capex / profit-share boom has no identified shock yet."""
    P = panel.loc["1955-01":]
    last = P.index.max()
    cap, prof, ls, dy, wi, wc = (P[f"s_{c}"] for c in ("capex_share", "profit_share", "labor_share", "div_yield",
                                                      "wealth_income", "white_collar_share"))
    peak = cap.loc["1995-01":"2001-12"].idxmax()
    late = dist["periods"][-1]
    c8 = _lp_at(dist, late["id"], "consumption", 8)
    return {
        "id": "ai_capex_chain", "title": "AI 资本开支与利润占比繁荣（描述性，尚无可识别冲击）",
        "nodes": [_node("capex", "AI 资本开支", "capital"), _node("profit", "利润占比上升", "distribution"),
                  _node("labor", "劳动收入占比下降、入门白领受冲击", "distribution"),
                  _node("cons", "消费", "macro"), _node("valuation", "估值抬升（股息率走低）", "asset"),
                  _node("nasdaq", "科技股/纳斯达克", "asset")],
        "edges": [
            _edge("capex", "profit", "+", "数年",
                  f"非住宅投资/GDP {last:%Y-%m} 为 {cap.iloc[-1]:.1f}%（1955 年以来均值 {cap.mean():.1f}%，"
                  f"科技繁荣峰值 {cap.loc[peak]:.1f}%，{peak:%Y-%m}）；企业税后利润/GDP {prof.iloc[-1]:.1f}%"
                  f"（长期均值 {prof.mean():.1f}%）"),
            _edge("profit", "labor", "-", "同期",
                  f"劳动收入占比指数 {ls.iloc[-1]:.1f}（1955 年以来均值 {ls.mean():.1f}）；白领行业就业占比 "
                  f"{wc.iloc[-1]:.1f}%，12 个月变化 {wc.iloc[-1] - wc.iloc[-13]:+.2f}pp；劳动收入占比下降 1% 后 4 季"
                  "利润/GDP：" + _lp_text(dist, "profit_share", 4, "pp")),
            _edge("labor", "cons", _lp_sign(dist, "consumption", 8), "0-8 季",
                  f"{late['name']}劳动收入占比下降 1% 后 8 季实际消费 " + _band(*c8, "%", "90%")),
            _edge("profit", "valuation", "+", "数年",
                  f"美股股息率 {dy.iloc[-1]:.2f}%（1955 年以来均值 {dy.mean():.2f}%，最低 {dy.min():.2f}%）；"
                  f"家庭净值/可支配收入 {wi.iloc[-1]:.1f} 倍（均值 {wi.mean():.1f} 倍）"),
            _edge("valuation", "nasdaq", _beta_sign(betas, "nasdaq", "y10"), "同月",
                  "纳斯达克对 10 年期收益率变动的标准化 β：" + _beta_text(betas, "nasdaq", "y10")
                  + "；美股大盘：" + _beta_text(betas, "us_equity", "y10")),
        ]}


def normalized_responses(irfs: np.ndarray) -> dict:
    """Per-draw ratios that compare periods with different shock sizes (oil supply shock,
    responses per +10% oil price on impact)."""
    k = SHOCK_IDS.index("oil_supply")
    oil, cpi, ip = (VAR_IDS.index(v) for v in ("oil", "cpi", "ip"))
    oil0 = irfs[:, 0, oil, k]
    out = {}
    for key, num in (("cpi12_per_oil10", irfs[:, 12, cpi, k]), ("ip24_per_oil10", irfs[:, 24, ip, k])):
        r = 10.0 * num / oil0
        out[key] = {f"p{q}": float(np.percentile(r, q)) for q in (16, 50, 84)}
    return out


# ================================================================ run

def write_shocks(shocks: pd.DataFrame, start: str, end, path=SHOCKS_PATH) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="MS")
    out = shocks.reindex(idx)
    out.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, date_format="%Y-%m-%d", float_format="%.6f")
    return out


def run(ctx, n_accept: int = N_ACCEPT, seed: int = 0, write: bool = True, refresh: bool = False, **_) -> dict:
    """Everything in results/causality.json; also writes processed/structural_shocks.csv (write=True).

    refresh re-downloads the quarterly FRED series of the distribution block.
    """
    panel = ctx.panel
    rng = np.random.default_rng(seed)
    last = panel.index.max()
    irf, normalized, n_acc, counts, periods, fitted = {}, {}, {}, {}, [], {}
    for per in PERIODS:
        var, draws = svar_period(panel, per, rng, n_accept=n_accept)
        fitted[per["id"]] = var
        irf[per["id"]] = band_summary(draws["irf"])
        normalized[per["id"]] = normalized_responses(draws["irf"])
        n_acc[per["id"]] = draws["n_accepted"]
        counts[per["id"]] = {k: draws[k] for k in ("n_rotations", "n_param_draws", "n_explosive", "acceptance_rate")}
        counts[per["id"]]["max_root_ols"] = max_root(var.coef, var.lags)
        counts[per["id"]]["n_obs"] = int(len(var.dates))
        periods.append({"id": per["id"], "name": per["name"], "start": per["start"],
                        "end": per["end"] or last.strftime("%Y-%m")})

    full = fitted["full"]
    K = len(SHOCK_IDS)
    mt = sample_svar(full, rng, n_accept=max(n_accept, N_ACCEPT_MT), param_uncertainty=False)
    k = median_target(mt["irf"])
    impact = mt["impact"][k]
    E = structural_shocks(full, impact).iloc[:, :K]
    E.columns = [f"e_{s}" for s in SHOCK_IDS]
    if write:
        write_shocks(E, PERIODS[0]["start"], last)
    mt_irf = to_reported(ma_coefs(full.coef, full.lags, HORIZON) @ impact[None])[:, :, :K]

    # historical decomposition of 12-month CPI inflation (pp)
    raw = structural_shocks(full, impact, standardize=False)
    hd = historical_decomposition(full, impact, raw, special=~np.isin(raw.index, full.dates))
    ci = VAR_IDS.index("cpi")
    roll = lambda x: pd.Series(x / 12.0).rolling(12).sum().to_numpy()[11:]  # noqa: E731
    contrib = {s: roll(hd["contrib"][j, :, ci]) for j, s in enumerate(SHOCK_IDS)}
    contrib["other"] = roll(hd["contrib"][K:full.n, :, ci].sum(axis=0))
    contrib["covid"] = roll(hd["contrib"][full.n, :, ci])
    hdec = {"dates": [d.strftime("%Y-%m") for d in raw.index[11:]], "variable": "cpi",
            "unit": "12 个月 CPI 通胀（对数近似，%），各冲击贡献为百分点",
            "actual": roll(hd["actual"][:, ci]), "baseline": roll(hd["baseline"][:, ci]), "contrib": contrib,
            "names": {**SHOCK_NAMES, "other": "其他（未识别）冲击",
                      "covid": f"疫情月份残差（{OUTLIERS[0]} 起，未拆分）", "baseline": "初始条件与常数项"}}

    svar = {
        "variables": VARIABLES,
        "shocks": [{"id": s["id"], "name": s["name"], "restrictions": [
            {"var": v, "sign": "+" if sg > 0 else "-", "horizon": [h0, h1]} for v, sg, h0, h1 in s["restrictions"]],
            "text": restriction_text(s)} for s in SHOCKS],
        "lags": LAGS, "horizon": HORIZON, "periods": periods, "irf": irf, "n_accepted": n_acc,
        "draws": counts, "normalized": normalized,
        "median_target": {"period": "full", "draw": k, "n_rotation_draws": mt["n_accepted"],
                          "acceptance_rate": mt["acceptance_rate"],
                          "irf": {s: {v: mt_irf[:, i, j] for i, v in enumerate(VAR_IDS)}
                                  for j, s in enumerate(SHOCK_IDS)},
                          "impact": impact},
        "shock_corr": np.corrcoef(E.loc[full.dates].to_numpy(float), rowvar=False),
        "excluded": {"months": list(OUTLIERS),
                     "rows": [d.strftime("%Y-%m") for d in full.Y_all.index if d not in full.dates],
                     "note": "疫情月份及以其为滞后项的月份不参与估计；这些月份的结构冲击在 CSV 中留空，"
                             "历史分解中其残差单独记为“疫情月份残差”。"},
        "method": "符号约束 SVAR（Rubio-Ramírez–Waggoner–Zha 旋转算法）；6 阶滞后；"
                  "区间同时包含参数不确定性（正态-逆 Wishart 后验）与识别不确定性；"
                  "点模型为 Fry-Pagan 中位数目标。",
    }
    dist = distribution_block(panel, quarterly_data(panel, refresh=refresh))
    betas = asset_betas(panel)
    return {"svar": svar, "historical_decomposition": hdec, "distribution": dist,
            "asset_betas": betas, "chains": build_chains(svar, dist, betas, panel)}


def restriction_text(shock: dict) -> str:
    names = {v["id"]: v["name"] for v in VARIABLES}
    groups: dict[tuple[int, int], list[str]] = {}
    for v, sg, h0, h1 in shock["restrictions"]:
        groups.setdefault((h0, h1), []).append(names[v] + ("↑" if sg > 0 else "↓"))
    return "；".join(f"{'、'.join(items)}（冲击后第 {h0}-{h1} 月）" for (h0, h1), items in groups.items())


if __name__ == "__main__":
    from ..context import EngineContext

    jsonio.dump(run(EngineContext.load(selection=False)), config.RESULTS_DIR / "causality.json")
