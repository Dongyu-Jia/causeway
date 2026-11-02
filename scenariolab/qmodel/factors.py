"""The factor layer of the quarterly stack: one joint model of every x_q field (SIM_DESIGN §5, §14 P3).

Ports the research round's joint study (joint/jq.py, jq2-jq4, j11, j13: the quarterly BVAR, Waggoner-Zha
conditioning, the C' cycle hybrid) and the qmodel's x block (qcore.py: ridge satellites, missing-data EM with a
factor-analytic or shrinkage covariance) onto the project's quarterly panel (qmodel.panel).

The model
---------
Core: a BVAR(2) on ten quarterly variables (config_q.FACTOR_CORE), in the within-quarter recursive order of the
do-operator [E0] pc ps ip lu -> [E0p] rff -> [E1c] ro -> [E1] ry10 sp gz usd:

    pc   headline CPI inflation (annualized)            = d_cpi
    ps   services CPI inflation                         = d_cpi_serv
    ip   IP growth (annualized)                         = 4 d_ip
    lu   unemployment on a logit scale                  u = 2 + 28 / (1 + e^-lu): 2 < u < 30
    rff  the shadow real policy rate ff - pi_trend      observed ff = max(rff + pi_trend, 0.1)
    ro   100 log real oil                               d_oil = Δro + pc / 4
    ry10 real 10-year y10 - pi_trend (a shadow)        observed y10 = max(ry10 + pi_trend, 0); d_y10 = Δ observed
    sp   Baa - 10y (timing fixed), logit scale          spread = 0.1 + 14.9 / (1 + e^-sp): 0.1 < spread < 15
    gz   the GZ spread, logit scale                     gz = 0.1 + 19.9 / (1 + e^-gz): 0.1 < GZ < 20
    usd  100 log broad dollar
pi_trend is the EWMA of headline CPI inflation with a 36-month half-life (quarterly: pi_q = pi_{q-1} +
a (pc_q - pi_{q-1}), a = 1 - 0.5^(3/36)); goods inflation comes from the identity pc = w_g d_cpi_goods + w_s ps with
training weights (cpi_weights). The bounded scales (Bounded, config_q.FACTOR_BOUNDS; P4b) keep §5.4's floors (u > 2,
spreads > 0.1: T-F found the linear model's spreads below 0.2 in more than 0.5% of draws) and add ceilings well above
history; P3's log(level - lo) form (transform='log') made the levels lognormal and exploded at long horizons (§15 Q5's
alternative, decided in T-F F6). The observed 10-year has an effective lower bound (config_q.Y10_FLOOR) like the
policy rate's, with ry10 the shadow below it.

Estimation (§5.2): Minnesota + sum-of-coefficients dummy priors (Banbura, Giannone and Reichlin 2010) with lambda by
marginal likelihood on the training window; Student-t likelihood weights w_t = (nu + K)/(nu + m_t^2) by EM, nu in
{4, 8, 16, inf} by marginal likelihood (Chib's identity at the EM estimate; exact for nu = inf), so extreme quarters
get small weights by a rule, not by a calendar; the zero-bound quarters' shadow rate is missing data in the same EM
(§5.4); a steady-state prior on the unconditional means (Villani 2009) centred on the anchors (§5.5: the train-end
rule of engine/anchors.py in backtests, the §3.8 defaults forward), sd 1.0 pp in backtests and 0.5 pp forward,
applied to every posterior draw as the conditional posterior of the means given its dynamics; FACTOR_DRAWS posterior
draws with explosive draws redrawn (the rejection rate is recorded). The draw rule (stable, P4b) tests the system that
is simulated: the VAR and the VAR under the cycle module, which makes IP exogenous in statement-free quarters (35% of
the draws P3 accepted were explosive there); every root must lie below config_q.FACTOR_MAX_ROOT, so each mode at least
halves within the longest simulated horizon (40 years).

Satellites (§5.1): every other x field w, w_q = d + rho w_{q-1} + Gamma0 k_q + Gamma1 k_{q-1} + e_q (ridge, blocked
5-fold CV, the core's t-weights), where the same-quarter core terms are restricted by the recursive order (E0 fields
see E0 core only; policy fields also the policy rate; market prices and surveys all of it) and k_{q-1}, the core at
the end of the previous quarter, is the state. The satellites' residual covariance is factor-analytic (k in
{3, 5, 8}) or shrinkage, chosen by masked-entry error on the last training block, from a missing-data EM (late
series are masked before they start, never invented). The joint innovation of [core; satellites] is
[e_k; Gamma0 e_k + e_w]: one covariance shared with the core.

Simulation
----------
Every path draws whitened training residual vectors with equal probability (history's own frequency, U-12), recoloured
with its posterior draw (config_q.FACTOR_POOL_WINSOR_SD, off by default, would winsorize the whitened coordinates). Statements (Statement) become restrictions on the stacked path and are imposed by projecting
the stacked innovations (Waggoner and Zha 1999):  e* = e + V G' (G V G')^-1 (r - R mu - G e),  G = R M, V = I (x) Sigma.
Hard statements hold exactly on every path; soft ones (value + sd) are met within their sd (Andersson, Palmqvist and
Waggoner 2010: each path's target is drawn from N(value, sd^2) and imposed exactly); ranges by accept/reject while at
least 10% of paths pass, otherwise a Gibbs draw of the restricted coordinates. Statements on the bounded
variables (unemployment, the spreads) are linear when the previous level is pinned (the start, or a chain of stated
changes from it) and otherwise solved per path by Newton steps in the space the linear rows leave free; a
stated level outside the bounds is delivered just inside them; a stated
policy-rate change is met on the observed rate, with the floor handled by an active set (level and kink forms) per
path. The do-operator: a stated u_* is a
structural shock of its policy equation under the recursive order, so same-quarter E0 macro is not moved by it, E1 /
E2 respond on impact and E0 with the lags; a stated raw policy field is a reduced-form observation. The cycle module
(§5.6, C') draws IP growth from 2-quarter blocks of training IP growth in quarters that state nothing about growth
(soft, sd = 0.5 x the quarterly IP sd; the other variables co-move through the innovation covariance; applied
quarter by quarter, one-sided, so a statement-free quarter never reshapes the past). Saturation (§3.5): a stated
value's gap to its conditional mean saturates at +-4 training sd (d_oil and d_y10 pass through). Completion shrinkage
and the
regime covariance (§5.7) are T-C candidates (config_q.COMPLETION_W, REGIME_COV).

Path mode conditions on every statement at once (two-sided); step mode is H = 1. The rarity statistic is the
Mahalanobis size of the innovation adjustment the statements need, as a percentile among the simulator's own
bootstrap draws (§3.5; the chi-square against the t-weighted Sigma over-warned 13-fold). The bootstrap pool is centred,
so the bootstrap's mean path is the model's.

Nothing here downloads or calls an LLM; every fit uses quarters <= its train end only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, multigammaln, ndtr, ndtri
from scipy.stats import chi2, invwishart

from ..engine import anchors as eng_anchors
from . import card as qcard
from . import config_q as cq

VERSION = "2"                   # 2: the fix round (centred pool, published initial state, §3.5 saturation and rarity)
CORE = list(cq.FACTOR_CORE)
NC = len(CORE)
IDX = {v: i for i, v in enumerate(CORE)}
P_LAGS = cq.FACTOR_LAGS
CORE_CLASS = {"pc": "E0", "ps": "E0", "ip": "E0", "lu": "E0", "rff": "E0p", "ro": "E1c", "ry10": "E1", "sp": "E1",
              "gz": "E1", "usd": "E1"}
DELTA = {"pc": 0.5, "ps": 1.0, "ip": 0.0, "lu": 1.0, "rff": 1.0, "ro": 1.0, "ry10": 1.0, "sp": 1.0, "gz": 1.0,
         "usd": 1.0}                         # Minnesota prior mean of the own first lag (jq.DELTA)
RATE_VARS = ("pc", "ps", "ip")               # flows: the satellites' same-quarter block takes them as rates
ALPHA = 1.0 - 0.5 ** (3.0 / cq.EWMA_HALF_LIFE_MONTHS)
_LEVEL_EPS = 1e-4                            # a level target is kept this far inside its bounds
X_NEWTON_MAX = 40.0                          # Newton iterates on bounded scales stay within |x| <= this (logistic flat)


@dataclass(frozen=True)
class Bounded:
    """A core variable modelled on a bounded scale (§5.4, §15 Q5). kind 'logit' (the default since P4b): x = log((y -
    lo) / (hi - y)), y = lo + (hi - lo) / (1 + e^-x), so lo < y < hi; kind 'log' (P3's form, hi = inf): x = log(y - lo),
    y = lo + e^x. `value` maps a level to x (data outside (lo, hi): NaN; clip=True keeps a target _LEVEL_EPS inside)."""
    lo: float
    hi: float = math.inf
    kind: str = "logit"

    def level(self, x):
        x = np.asarray(x, float)
        if self.kind == "log":
            return self.lo + np.exp(x)
        return self.lo + (self.hi - self.lo) * expit(x)

    def dlevel(self, x):
        """dy / dx at x."""
        x = np.asarray(x, float)
        if self.kind == "log":
            return np.exp(x)
        s = expit(x)
        return (self.hi - self.lo) * s * (1.0 - s)

    def value(self, y, clip: bool = False):
        y = np.asarray(y, float)
        top = self.hi if self.kind == "logit" else math.inf
        if clip:
            y = np.clip(y, self.lo + _LEVEL_EPS, top - _LEVEL_EPS)
        ok = (y > self.lo) & (y < top)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log(y - self.lo) if self.kind == "log" else np.log(y - self.lo) - np.log(top - y)
        return np.where(ok, x, np.nan)

    def moments(self, m, s):
        """Mean and sd of level(X), X ~ N(m, s^2): closed form for 'log', Gauss-Hermite quadrature for 'logit'."""
        m = np.asarray(m, float)
        s = float(max(s, 0.0))
        if self.kind == "log":
            e = np.exp(m + 0.5 * s * s)
            return self.lo + e, e * math.sqrt(math.expm1(s * s))
        z, w = _GH
        lv = self.level(m[..., None] + s * z)
        mu = lv @ w
        return mu, np.sqrt(np.maximum((lv - mu[..., None]) ** 2 @ w, 0.0))


_z, _w = np.polynomial.hermite_e.hermegauss(40)
_GH = (_z, _w / _w.sum())


@dataclass(frozen=True)
class Transforms:
    """The core variables on a bounded scale and the observables measured on them."""
    vars: dict                              # core variable -> Bounded

    @property
    def obs(self) -> dict:
        """{observable: (core variable, "change" or "level")} for the transformed variables."""
        out = {}
        for v, (chg, lev) in (("lu", ("d_unrate", "unrate")), ("sp", ("d_spread", "spread")), ("gz", ("d_gz", "gz"))):
            if v in self.vars:
                out.update({chg: (v, "change"), lev: (v, "level")})
        return out

    def level(self, v: str, x):
        b = self.vars.get(v)
        return np.asarray(x, float) if b is None else b.level(x)

    def value(self, v: str, y, clip: bool = False):
        b = self.vars.get(v)
        return np.asarray(y, float) if b is None else b.value(y, clip=clip)

    def dlevel(self, v: str, x):
        b = self.vars.get(v)
        return np.ones_like(np.asarray(x, float)) if b is None else b.dlevel(x)

    def summary(self) -> dict:
        return {v: {"kind": b.kind, "lo": b.lo, "hi": None if not math.isfinite(b.hi) else b.hi}
                for v, b in self.vars.items()}


def transforms(kind: str | None = None, ceilings: dict | None = None) -> Transforms:
    """The core's bounded variables (config_q.FACTOR_BOUNDS): unemployment always (§5.4), the two spreads with
    FACTOR_SPREAD_LOG; kind 'logit' (config_q.FACTOR_TRANSFORM, the default) or 'log' (P3's log(level - lo));
    ceilings overrides the upper bounds (bound_ceilings)."""
    kind = cq.FACTOR_TRANSFORM if kind is None else kind
    if kind not in ("logit", "log"):
        raise ValueError(f"transform={kind!r}: 'logit' (default) or 'log'")
    use = ["lu"] + (["sp", "gz"] if cq.FACTOR_SPREAD_LOG else [])
    hi = {v: (ceilings or {}).get(v, cq.FACTOR_BOUNDS[v][1]) for v in use}
    return Transforms({v: Bounded(cq.FACTOR_BOUNDS[v][0], hi[v] if kind == "logit" else math.inf, kind) for v in use})


BOUND_LEVELS = {"lu": "lvl_unrate", "sp": "lvl_spread", "gz": "lvl_gz"}


def bound_ceilings(Q: pd.DataFrame, train_end, rule: str | None = None) -> dict | None:
    """The bounded scales' ceilings at a train end. rule 'fixed' (config_q.FACTOR_BOUNDS, the default: judgment values,
    see config_q) -> None; 'train_end' (the option the statistics review asked for): FACTOR_BOUND_MULT x each level's
    maximum over the quarters up to the train end, so a backtest's ceilings use no later data."""
    rule = cq.FACTOR_BOUNDS_RULE if rule is None else rule
    if rule == "fixed":
        return None
    if rule != "train_end":
        raise ValueError(f"bounds={rule!r}: 'fixed' (default) or 'train_end'")
    te = pd.Period(train_end, "Q")
    return {v: float(cq.FACTOR_BOUND_MULT * Q[col].loc[:te].max()) for v, col in BOUND_LEVELS.items()}


TRANSFORMS = transforms()                    # the default (config_q); a fitted model carries its own (FactorModel.tf)
# the x fields the core carries (the rest are satellites)
CORE_FIELDS = ("d_cpi_goods", "d_cpi_serv", "d_ip", "d_unrate", "d_ff", "d_oil", "d_y10", "d_spread", "d_gz", "d_usd")
LEVEL_OBS = ("ff", "ff_shadow", "y10", "y10_shadow", "unrate", "spread", "gz", "usd", "real_oil", "rff", "ry10", "pc",
             "cpi_yoy", "core_yoy")
SHOCKS = list(cq.POLICY_DO)
E1_CLASSES = ("E1c", "E1", "E2")
MACRO_CLASSES = ("E0", "E0p", "last")            # "last": d_bs, an E0p field ordered after everything
RANK = {"E0": 0, "E0p": 1, "E1c": 2, "E1": 2, "E2": 2, "last": 3}
CLIP_TEXT = "超出历史范围，按 ±4 个标准差截断"
# the panel columns the core variables and the start levels are built from (core_frame, initial_state) and the raw
# inputs that set their release lags (config_q.SERIES_LAG): a quarter-end value of quarter q - k is published by the
# start of q when its lag is at most 3 (k - 1) months, the rule of the panel's <x>_l<k>__known columns (§3.2)
CORE_COLUMN_SOURCES = {"d_cpi": ("panel:f_cpi",), "d_cpi_serv": ("panel:f_cpi_services",), "d_ip": ("panel:f_ip",),
                       "d_oil": ("panel:f_oil",), "d_core": ("panel:lvl_core",), "lvl_unrate": ("panel:lvl_unrate",),
                       "lvl_ff": ("panel:lvl_ff",), "lvl_infl_trend": ("panel:s_infl_expect",),
                       "lvl_oil_real": ("panel:lvl_oil", "panel:lvl_cpi"), "lvl_y10": ("panel:lvl_y10",),
                       "lvl_spread": ("panel:s_spread",), "lvl_gz": ("panel:s_gz",), "lvl_usd": ("panel:lvl_usd",),
                       "lvl_cpi": ("panel:lvl_cpi",)}
INFORMATION = ("auto", "published", "available")
# saturation (§3.5): the panel series of each level observable; a stated level h quarters ahead is measured against
# the training sd of the level's h-quarter change
LEVEL_SERIES = {"ff": "lvl_ff", "ff_shadow": "lvl_ff", "y10": "lvl_y10", "y10_shadow": "lvl_y10",
                "unrate": "lvl_unrate", "spread": "lvl_spread", "gz": "lvl_gz", "usd": "lvl_usd",
                "real_oil": "lvl_oil_real"}
RARITY_DRAWS = 2000                          # path-mode rarity: the reference distribution's draws (fixed seed)
RARITY_SEED = 31


class FactorError(ValueError):
    """A statement the factor layer cannot impose (Chinese message)."""


# ================================================================ card and data

_CARD: dict | None = None


def _card() -> dict:
    global _CARD
    if _CARD is None:
        _CARD = qcard.load_card()
    return _CARD


def field_classes() -> dict[str, str]:
    """{x field or shock: endogeneity class} from the model card (E0, E0p, E1c, E1, E2)."""
    card = _card()
    out = {r["id"]: r["endo_code"] for r in qcard.rows(card, "x")}
    out.update({r["id"]: "E0p" for r in qcard.rows(card, "u")})
    return out


def satellite_fields() -> list[str]:
    """Every x field of the card that the core does not carry, in card order (42)."""
    return [f for f in qcard.x_fields(_card()) if f not in CORE_FIELDS]


def zlb_mask(Q: pd.DataFrame) -> pd.Series:
    """Quarters whose end-of-quarter policy rate is below ZLB_FF: the shadow rate is missing there (§5.4)."""
    return Q["lvl_ff"] < cq.ZLB_FF


def core_frame(Q: pd.DataFrame, tf: Transforms | None = None) -> pd.DataFrame:
    """The core variables per quarter from the quarterly panel (levels at quarter ends, flows over the quarter);
    unemployment and the spreads on their bounded scale (tf, default TRANSFORMS); rff is NaN in zero-bound quarters
    (the shadow rate is imputed by the EM)."""
    tf = TRANSFORMS if tf is None else tf
    c = pd.DataFrame(index=Q.index)
    c["pc"] = Q["d_cpi"]
    c["ps"] = Q["d_cpi_serv"]
    c["ip"] = 4.0 * Q["d_ip"]
    c["lu"] = tf.value("lu", Q["lvl_unrate"].to_numpy(float))
    c["rff"] = (Q["lvl_ff"] - Q["lvl_infl_trend"]).where(~zlb_mask(Q))
    c["ro"] = Q["lvl_oil_real"]
    c["ry10"] = Q["lvl_y10"] - Q["lvl_infl_trend"]
    c["sp"] = tf.value("sp", Q["lvl_spread"].to_numpy(float))
    c["gz"] = tf.value("gz", Q["lvl_gz"].to_numpy(float))
    c["usd"] = Q["lvl_usd"]
    return c[CORE]


def cpi_weights(Q: pd.DataFrame, train_end) -> tuple[float, float]:
    """(w_g, w_s): headline = w_g goods + w_s services, OLS without intercept over the CPI_WEIGHT_QUARTERS to the
    train end (jq.cpi_weights)."""
    W = Q.loc[:pd.Period(train_end, "Q"), ["d_cpi_goods", "d_cpi_serv", "d_cpi"]].dropna().iloc[
        -cq.CPI_WEIGHT_QUARTERS:]
    b = np.linalg.lstsq(W[["d_cpi_goods", "d_cpi_serv"]].to_numpy(float), W["d_cpi"].to_numpy(float), rcond=None)[0]
    return float(b[0]), float(b[1])


# ================================================================ BVAR pieces (jq.py)

def lagmat(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """(Y_lhs (T-p, n), X (T-p, n p + 1)) with X = [y_{t-1}, ..., y_{t-p}, 1]."""
    T = Y.shape[0]
    X = np.column_stack([Y[p - lag:T - lag] for lag in range(1, p + 1)] + [np.ones((T - p, 1))])
    return Y[p:], X


def ar_sigma(Y: np.ndarray, p: int) -> np.ndarray:
    """Residual sd of univariate AR(p) regressions with a constant (the Minnesota scale)."""
    out = []
    for i in range(Y.shape[1]):
        yl, X = lagmat(Y[:, i:i + 1], p)
        ok = np.isfinite(yl).ravel() & np.isfinite(X).all(1)
        b = np.linalg.lstsq(X[ok], yl[ok], rcond=None)[0]
        e = yl[ok] - X[ok] @ b
        out.append(float(np.sqrt((e ** 2).sum() / max(len(e) - X.shape[1], 1))))
    return np.array(out)


def dummies(n: int, p: int, sigma: np.ndarray, delta: np.ndarray, mubar: np.ndarray, lam: float,
            tau: float | None, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Minnesota (lag decay 1/l), Sigma, constant and sum-of-coefficients dummy observations (jq.dummies)."""
    k = n * p + 1
    Y1 = np.zeros((n * p, n))
    Y1[:n, :] = np.diag(delta * sigma) / lam
    X1 = np.zeros((n * p, k))
    for lag in range(1, p + 1):
        X1[(lag - 1) * n:lag * n, (lag - 1) * n:lag * n] = np.diag(sigma) * lag / lam
    Yd, Xd = [Y1, np.diag(sigma), np.zeros((1, n))], [X1, np.zeros((n, k)), np.zeros((1, k))]
    Xd[2][0, -1] = eps
    if tau:
        Y4 = np.diag(delta * mubar) / tau
        X4 = np.zeros((n, k))
        for lag in range(p):
            X4[:, lag * n:(lag + 1) * n] = np.diag(delta * mubar) / tau
        keep = delta != 0
        Yd.append(Y4[keep])
        Xd.append(X4[keep])
    return np.vstack(Yd), np.vstack(Xd)


def _logc(XtX: np.ndarray, S: np.ndarray, T: float, k: int, n: int) -> float:
    return float(-0.5 * n * np.linalg.slogdet(XtX)[1] - 0.5 * (T - k) * np.linalg.slogdet(S)[1]
                 + multigammaln(0.5 * (T - k), n))


def log_iw(Sig: np.ndarray, S: np.ndarray, nu: float) -> float:
    n = S.shape[0]
    return float(0.5 * nu * np.linalg.slogdet(S)[1] - 0.5 * nu * n * math.log(2.0) - multigammaln(0.5 * nu, n)
                 - 0.5 * (nu + n + 1) * np.linalg.slogdet(Sig)[1] - 0.5 * np.trace(S @ np.linalg.inv(Sig)))


def log_mn(B: np.ndarray, M: np.ndarray, Sig: np.ndarray, XtX: np.ndarray) -> float:
    k, n = B.shape
    D = B - M
    return float(-0.5 * k * n * math.log(2 * math.pi) + 0.5 * n * np.linalg.slogdet(XtX)[1]
                 - 0.5 * k * np.linalg.slogdet(Sig)[1] - 0.5 * np.trace(np.linalg.solve(Sig, D.T @ XtX @ D)))


def log_t(E: np.ndarray, Sig: np.ndarray, nu: float) -> np.ndarray:
    """Row log densities of a K-variate t (scale Sig, nu dof; nu = inf: Gaussian)."""
    n = Sig.shape[0]
    L = np.linalg.cholesky(Sig)
    Z = np.linalg.solve(L, E.T).T
    m2 = (Z ** 2).sum(1)
    ld = 2.0 * np.log(np.diag(L)).sum()
    if not np.isfinite(nu):
        return -0.5 * n * math.log(2 * math.pi) - 0.5 * ld - 0.5 * m2
    return (gammaln(0.5 * (nu + n)) - gammaln(0.5 * nu) - 0.5 * n * math.log(nu * math.pi) - 0.5 * ld
            - 0.5 * (nu + n) * np.log1p(m2 / nu))


def companion(A: list[np.ndarray]) -> np.ndarray:
    n, p = A[0].shape[0], len(A)
    F = np.zeros((n * p, n * p))
    F[:n, :] = np.hstack(A)
    if p > 1:
        F[n:, :-n] = np.eye(n * (p - 1))
    return F


def max_root(A: list[np.ndarray]) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(companion(A)))))


def split_B(B: np.ndarray, n: int, p: int) -> tuple[list[np.ndarray], np.ndarray]:
    """B (k, n) rows [lag1 (n), ..., lagp (n), const] -> ([A_1, ..., A_p] (n, n) each, c (n,))."""
    return [B[(lag - 1) * n:lag * n].T.copy() for lag in range(1, p + 1)], B[n * p].copy()


# ================================================================ core estimation: EM with t weights and missing data

@dataclass
class CoreFit:
    """The core BVAR at one train end: posterior moments, the EM's weights and imputed shadow rate, the selection."""
    dates: pd.PeriodIndex                 # quarters of Y (training sample)
    Y: np.ndarray                         # (T, n) completed data (shadow rate imputed)
    missing: np.ndarray                   # (T, n) bool
    B: np.ndarray                         # (k, n) posterior mean (lag rows, const row: before the steady state)
    S1: np.ndarray
    X1X1: np.ndarray
    T1: float
    Sigma: np.ndarray                     # posterior mean of Sigma (the t scale)
    w: np.ndarray                         # (T - p,) Student-t weights of the LHS rows
    m2: np.ndarray                        # (T - p,) Mahalanobis norms at the end of the EM
    lam: float
    nu: float
    logml: dict                           # {nu: {"chib": ..., "lam": ..., "iters": ...}}
    sigma0: np.ndarray                    # Minnesota scales
    Vm: np.ndarray                        # posterior covariance of the missing entries
    resid: np.ndarray                     # (T - p, n) at the posterior mean
    iters: int = 0


def _box_qp(H: np.ndarray, g: np.ndarray, ub: np.ndarray, iters: int = 60) -> np.ndarray:
    """argmin 1/2 m'Hm + g'm subject to m <= ub (an active-set method; H positive definite)."""
    m = -np.linalg.solve(H, g)
    act = m > ub
    for _ in range(iters):
        x = np.where(act, ub, 0.0)
        fr = ~act
        if fr.any():
            x[fr] = np.linalg.solve(H[np.ix_(fr, fr)], -(g[fr] + H[np.ix_(fr, act)] @ ub[act]))
        grad = H @ x + g
        new = (act & (grad <= 0)) | (fr & (x > ub + 1e-12))
        if (new == act).all():
            return x
        act = new
    return np.minimum(x, ub)


def _em_once(Y: np.ndarray, p: int, lam_grid, nu: float, soc: bool, iters: int, tol: float,
             fixed: dict | None = None, upper: np.ndarray | None = None) -> dict:
    """EM for one nu: missing entries and t weights given (B, Sigma); lambda re-chosen each iteration by the
    closed-form marginal likelihood of the weighted, completed data. `fixed` = {B, Sigma, w}: only impute the missing
    entries at fixed parameters (one-sided shadow-rate filtering after a train end)."""
    T, n = Y.shape
    k = n * p + 1
    miss = ~np.isfinite(Y)
    mi = np.argwhere(miss)
    nm = len(mi)
    Yh = pd.DataFrame(Y).interpolate(limit_direction="both").to_numpy(float)
    if fixed is not None:
        sigma, mubar = None, None
    else:
        sigma = ar_sigma(Yh, p)
        mubar = Yh[:p].mean(0)
    delta = np.array([DELTA[v] for v in CORE[:n]]) if n == NC else np.ones(n)
    jmap = {(int(t), int(i)): j for j, (t, i) in enumerate(mi)}
    # positions in Z_r = [y_t, y_{t-1}, ..., y_{t-p}, 1] of missing entries, per LHS row r
    zpos: list[tuple[int, np.ndarray, np.ndarray]] = []
    if nm:
        for r in range(T - p):
            t = r + p
            cols, js = [], []
            for lag in range(0, p + 1):
                for i in range(n):
                    j = jmap.get((t - lag, i))
                    if j is not None:
                        cols.append(i if lag == 0 else n + (lag - 1) * n + i)
                        js.append(j)
            if cols:
                zpos.append((r, np.array(cols), np.array(js)))
    rows_m = np.array(sorted({t - p + lag for t, _ in mi for lag in range(0, p + 1) if 0 <= t - p + lag < T - p}),
                      dtype=int)
    rpos = {int(r): j for j, r in enumerate(rows_m)}
    dum = {}
    if fixed is None:
        for lam in lam_grid:
            Yd, Xd = dummies(n, p, sigma, delta, mubar, lam, cq.FACTOR_SOC_TAU * lam if soc else None)
            Bd = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
            Sd = (Yd - Xd @ Bd).T @ (Yd - Xd @ Bd)
            dum[lam] = (Xd.T @ Xd, Xd.T @ Yd, Yd.T @ Yd, Bd, Sd, Yd.shape[0],
                        _logc(Xd.T @ Xd, Sd, Yd.shape[0], k, n))
    w = np.ones(T - p) if fixed is None else np.where(np.isfinite(fixed["w"]), fixed["w"], 1.0)
    Vm = np.zeros((nm, nm))
    best = None
    it = 0
    for it in range(iters):
        if fixed is None:
            yl, X = lagmat(Yh, p)
            Z = np.column_stack([yl, X])
            G = Z.T @ (Z * w[:, None])
            for r, cols, js in zpos:
                G[np.ix_(cols, cols)] += w[r] * Vm[np.ix_(js, js)]
            YY, XY, XX = G[:n, :n], G[n:, :n], G[n:, n:]
            best = None
            for lam, (DXX, DXY, DYY, Bd, Sd, Td, c0) in dum.items():
                X1X1 = XX + DXX
                B = np.linalg.solve(X1X1, XY + DXY)
                S1 = YY + DYY - B.T @ X1X1 @ B
                S1 = 0.5 * (S1 + S1.T)
                T1 = (T - p) + Td
                ml = _logc(X1X1, S1, T1, k, n) - c0 - 0.5 * n * (T - p) * math.log(math.pi)
                if best is None or ml > best["ml"]:
                    best = dict(ml=ml, lam=lam, B=B, S1=S1, X1X1=X1X1, T1=T1, Bd=Bd, Sd=Sd, Td=Td, DXX=DXX)
            B, S1, T1 = best["B"], best["S1"], best["T1"]
            Sig = S1 / (T1 - k - n - 1)
        else:
            B, Sig = fixed["B"], fixed["Sigma"]
        A, _ = split_B(B, n, p)
        Si = np.linalg.inv(Sig)
        Ynew = Yh
        if nm:
            Y0 = Yh.copy()
            Y0[miss] = 0.0
            yl0, X0 = lagmat(Y0, p)
            a = (yl0 - X0 @ B)[rows_m]                         # residuals with the missing entries at 0
            Ce = np.zeros((len(rows_m), n, nm))                # d residual / d missing entry, involved rows only
            for j, (t, i) in enumerate(mi):
                if t >= p:
                    Ce[rpos[t - p], i, j] += 1.0
                for lag in range(1, p + 1):
                    r = t + lag
                    if p <= r < T:
                        Ce[rpos[r - p], :, j] -= A[lag - 1][:, i]
            CS = np.matmul(Si, Ce)                             # (R, n, nm)
            wr = w[rows_m]
            H = (Ce * wr[:, None, None]).reshape(-1, nm).T @ CS.reshape(-1, nm)
            g = (CS * wr[:, None, None]).reshape(-1, nm).T @ a.reshape(-1)
            Hs = 0.5 * (H + H.T) + 1e-12 * np.eye(nm)
            Vm = np.linalg.inv(Hs)
            Ynew = Yh.copy()
            if upper is None:
                Ynew[miss] = -Vm @ g
            else:                                      # censored: the shadow rate at or below the floor (JM 2021)
                ub = upper[miss]
                Ynew[miss] = _box_qp(Hs, g, np.where(np.isfinite(ub), ub, np.inf))
        yl, X = lagmat(Ynew, p)
        E = yl - X @ B
        m2 = np.einsum("ti,ij,tj->t", E, Si, E)
        if nm:
            m2[rows_m] += np.einsum("rim,rim->r", np.matmul(Ce, Vm), CS)
        wnew = np.ones(T - p) if not np.isfinite(nu) else (nu + n) / (nu + m2)
        if fixed is not None:                         # training rows keep their weights; later rows get the t rule
            wf = np.asarray(fixed["w"], float)
            wnew = np.where(np.isfinite(wf), wf, wnew)
        dy = float(np.nanmax(np.abs(Ynew - Yh))) if nm else 0.0
        dw = float(np.max(np.abs(wnew - w)))
        Yh, w = Ynew, wnew
        if it > 2 and dy < tol and dw < tol:
            break
    out = dict(Y=Yh, w=w, m2=m2, Vm=Vm, iters=it + 1, B=B, Sigma=Sig, miss=miss, mi=mi, sigma0=sigma)
    if fixed is None:
        out.update(best)
    return out


def _chib(r: dict, Y: np.ndarray, p: int, nu: float) -> float:
    """log marginal likelihood by Chib's identity at the EM estimate: log L_nu(theta) + log p(theta) -
    log p(theta | Y, w) (the weighted data's NIW posterior); exact for nu = inf without missing data."""
    n = Y.shape[1]
    k = n * p + 1
    B, S1, T1, X1X1 = r["B"], r["S1"], r["T1"], r["X1X1"]
    Sig_mode = S1 / (T1 - k + n + 1)
    yl, X = lagmat(r["Y"], p)
    loglik = float(log_t(yl - X @ B, Sig_mode, nu).sum())
    if len(r["mi"]):
        loglik += 0.5 * float(np.linalg.slogdet(2 * math.pi * r["Vm"])[1])
    lp = log_iw(Sig_mode, r["Sd"], r["Td"] - k) + log_mn(B, r["Bd"], Sig_mode, r["DXX"])
    lq = log_iw(Sig_mode, S1, T1 - k) + log_mn(B, B, Sig_mode, X1X1)
    return loglik + lp - lq


def fit_core(D: pd.DataFrame, p: int = P_LAGS, lam_grid=cq.FACTOR_LAMBDA_GRID, nu_grid=cq.FACTOR_NU_GRID,
             soc: bool = True, iters: int = cq.FACTOR_EM_ITERS, tol: float = cq.FACTOR_EM_TOL,
             upper: pd.DataFrame | None = None) -> CoreFit:
    """The core BVAR on D (quarters x CORE, NaN = missing): nu by Chib's marginal likelihood, lambda by the
    closed form inside the EM. upper: optional upper bounds of the missing entries (the censored shadow rate)."""
    Y = D.to_numpy(float)
    ub = None if upper is None else upper.reindex(index=D.index, columns=D.columns).to_numpy(float)
    n = Y.shape[1]
    runs, table = {}, {}
    for nu in nu_grid:
        r = _em_once(Y, p, lam_grid, nu, soc, iters, tol, upper=ub)
        r["chib"] = _chib(r, Y, p, nu)
        runs[nu] = r
        table["inf" if not np.isfinite(nu) else f"{nu:g}"] = {"chib": r["chib"], "lam": r["lam"],
                                                               "closed_form_weighted": r["ml"],
                                                               "iters": r["iters"]}
    nu = max(runs, key=lambda v: runs[v]["chib"])
    r = runs[nu]
    k = n * p + 1
    yl, X = lagmat(r["Y"], p)
    return CoreFit(dates=D.index, Y=r["Y"], missing=r["miss"], B=r["B"], S1=r["S1"], X1X1=r["X1X1"], T1=r["T1"],
                   Sigma=r["S1"] / (r["T1"] - k - n - 1), w=r["w"], m2=r["m2"], lam=r["lam"], nu=nu, logml=table,
                   sigma0=r["sigma0"], Vm=r["Vm"], resid=yl - X @ r["B"], iters=r["iters"])


# ================================================================ steady state (Villani 2009 prior on the means)

def steady_state(A: list[np.ndarray], Sig: np.ndarray, Y: np.ndarray, w: np.ndarray, mu_star: np.ndarray,
                 mu_sd: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(mu, c): the unconditional means' conditional posterior given the dynamics A and Sigma (a draw when `rng`,
    else its mean), with the prior mu ~ N(mu_star, diag(mu_sd^2)); c = (I - sum A) mu."""
    n = Sig.shape[0]
    p = len(A)
    T = Y.shape[0]
    yt = Y[p:] - sum(Y[p - lag:T - lag] @ A[lag - 1].T for lag in range(1, p + 1))
    D = np.eye(n) - sum(A)
    Si = np.linalg.inv(Sig)
    P0 = np.diag(1.0 / mu_sd ** 2)
    prec = P0 + float(w.sum()) * D.T @ Si @ D
    rhs = P0 @ mu_star + D.T @ Si @ (w[:, None] * yt).sum(0)
    prec = 0.5 * (prec + prec.T)
    ev, U = np.linalg.eigh(prec)
    ev = np.maximum(ev, 1e-12 * ev.max())
    mu = U @ ((U.T @ rhs) / ev)
    if rng is not None:                       # a draw from N(mu, prec^-1), stable when a direction is diffuse
        mu = mu + U @ (rng.standard_normal(n) / np.sqrt(ev))
    return mu, D @ mu


# ================================================================ satellites

def _std(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sw = w.sum()
    mu = (w[:, None] * X).sum(0) / sw
    sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0) / sw)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def _ridge(Z: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, pm: np.ndarray) -> tuple[float, np.ndarray]:
    sw = w.sum()
    zm = (w[:, None] * Z).sum(0) / sw
    ym = float((w * y).sum() / sw)
    Zc, yc = Z - zm, y - ym
    A = Zc.T @ (w[:, None] * Zc) + lam * np.diag(pm)
    b = np.linalg.solve(A + 1e-10 * np.eye(len(pm)), Zc.T @ (w * yc))
    return ym - zm @ b, b


def ridge_cv(Z: np.ndarray, y: np.ndarray, w: np.ndarray, pm: np.ndarray, grid=cq.SAT_RIDGE_GRID,
             folds: int = cq.SAT_FOLDS) -> tuple[float, np.ndarray, float]:
    """Weighted ridge with blocked k-fold CV over lambda = g x sum(w)."""
    n = len(y)
    edges = np.linspace(0, n, folds + 1).astype(int)
    best, best_lam = np.inf, None
    for g in grid:
        lam = g * float(w.sum())
        sse = 0.0
        for f in range(folds):
            te = np.arange(edges[f], edges[f + 1])
            tr = np.r_[0:edges[f], edges[f + 1]:n]
            a, b = _ridge(Z[tr], y[tr], w[tr], lam, pm)
            sse += float((w[te] * (y[te] - a - Z[te] @ b) ** 2).sum())
        if sse < best:
            best, best_lam = sse, lam
    a, b = _ridge(Z, y, w, best_lam, pm)
    return a, b, best_lam


def _cond_gauss(mu: np.ndarray, S: np.ndarray, obs: np.ndarray, val: np.ndarray):
    """Conditional mean (full vector, observed entries kept) and covariance of the unobserved entries."""
    mis = ~obs
    if not obs.any():
        return mu.copy(), S[np.ix_(mis, mis)]
    Soo = S[np.ix_(obs, obs)]
    Smo = S[np.ix_(mis, obs)]
    K = np.linalg.solve(Soo + 1e-10 * np.eye(int(obs.sum())), Smo.T).T
    m = mu.copy()
    m[mis] = mu[mis] + K @ (val[obs] - mu[obs])
    m[obs] = val[obs]
    C = S[np.ix_(mis, mis)] - K @ Smo.T
    return m, 0.5 * (C + C.T)


def _cond_linear(mu: np.ndarray, S: np.ndarray, obs: np.ndarray, val: np.ndarray, cons=()) -> np.ndarray:
    """The mean of x ~ N(mu, S) given x[obs] = val[obs] and the linear constraints c . x = h (all exact); the
    observed entries are kept as given."""
    n = len(mu)
    rows = [np.eye(n)[i] for i in np.flatnonzero(obs)] + [np.asarray(c, float) for c, _ in cons]
    if not rows:
        return mu.copy()
    Hm = np.vstack(rows)
    h = np.concatenate([np.asarray(val, float)[obs], np.array([float(b) for _, b in cons])])
    HS = Hm @ S
    A = HS @ Hm.T
    try:
        sol = np.linalg.solve(A + 1e-12 * np.eye(len(A)), h - Hm @ mu)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, h - Hm @ mu, rcond=None)[0]
    out = mu + HS.T @ sol
    out[obs] = np.asarray(val, float)[obs]
    return out


_COL_LAGS: dict | None = None


def _lag_of(col: str) -> int | None:
    """Release lag (months after the quarter's end) of a panel column: the core's inputs from config_q.SERIES_LAG,
    every other column from the panel meta's lag_months (None if unknown)."""
    global _COL_LAGS
    src = CORE_COLUMN_SOURCES.get(col)
    if src is not None:
        return max(int(cq.SERIES_LAG[s]) for s in src)
    if _COL_LAGS is None:
        try:
            from . import panel as qpanel
            _COL_LAGS = {c: int(m["lag_months"]) for c, m in qpanel.load_meta()["columns"].items()}
        except (OSError, KeyError, ValueError):
            _COL_LAGS = {}
    return _COL_LAGS.get(col)


def _information(Q: pd.DataFrame, q0, information: str) -> str:
    """'auto' -> 'available' at the panel's today row (start-of-quarter states only), else 'published'."""
    if information not in INFORMATION:
        raise ValueError(f"information={information!r}: one of {INFORMATION}")
    if information != "auto":
        return information
    today = q0 == Q.index[-1] and all(pd.isna(Q.at[q0, c]) for c in ("d_oil", "d_gz", "d_cpi") if c in Q.columns)
    return "available" if today else "published"


def _pub(Q: pd.DataFrame, col: str, q0, k: int, info: str) -> float:
    """The value of `col` in quarter q0 - k as known at the start of quarter q0 (NaN if not published then, or
    missing). 'published': the panel's <col>_l<k>__known column where it exists, else the column's release lag
    (published iff lag <= 3 (k - 1) months); 'available': the panel's value."""
    q = q0 - k
    if col not in Q.columns or q not in Q.index:
        return float("nan")
    if info == "published":
        kc = f"{col}_l{k}__known"
        if kc in Q.columns and q0 in Q.index:
            x = Q.at[q0, kc]
            return float(x) if pd.notna(x) else float("nan")
        lag = _lag_of(col)
        if lag is None or lag > 3 * (k - 1):
            return float("nan")
    x = Q.at[q, col]
    return float(x) if pd.notna(x) else float("nan")


def em_cov(E: np.ndarray, w: np.ndarray, kind: str = "fa", k: int = 5, alpha: float = 0.3,
           iters: int = cq.SAT_COV_ITERS, seed: int = 0) -> np.ndarray:
    """Covariance of zero-mean rows E (n, p), NaN = missing, rows weighted by w. kind 'fa': Lambda Lambda' + Psi
    (k factors); 'lw': (1 - alpha) S + alpha diag(S). Missing-data EM grouped by missingness pattern (qcore.em_cov)."""
    n, p = E.shape
    obs = np.isfinite(E)
    E0 = np.where(obs, E, 0.0)
    sw = float(w.sum())
    var = np.array([float(np.average(E[obs[:, j], j] ** 2, weights=w[obs[:, j]])) if obs[:, j].sum() > 2 else 1.0
                    for j in range(p)])
    var = np.where(var > 0, var, 1.0)
    pats: dict[bytes, list[int]] = {}
    for i in range(n):
        pats.setdefault(obs[i].tobytes(), []).append(i)
    groups = [(np.array(rows), obs[rows[0]]) for rows in pats.values() if obs[rows[0]].any()]
    if kind == "lw":
        S = np.diag(var)
        for _ in range(iters):
            acc = np.zeros((p, p))
            for rows, o in groups:
                mis = ~o
                if mis.any():
                    K = np.linalg.solve(S[np.ix_(o, o)] + 1e-10 * np.eye(int(o.sum())), S[np.ix_(mis, o)].T).T
                    C = S[np.ix_(mis, mis)] - K @ S[np.ix_(mis, o)].T
                else:
                    K, C = None, None
                M = E0[rows].copy()
                if mis.any():
                    M[:, mis] = M[:, o] @ K.T
                acc += M.T @ (w[rows, None] * M)
                if mis.any():
                    acc[np.ix_(mis, mis)] += float(w[rows].sum()) * C
            Sn = acc / sw
            S = (1.0 - alpha) * Sn + alpha * np.diag(np.diag(Sn))
        return 0.5 * (S + S.T)
    rng = np.random.default_rng(seed)
    L = rng.normal(scale=0.1, size=(p, k)) * np.sqrt(var)[:, None]
    Psi = 0.5 * var
    for _ in range(iters):
        Mf = np.zeros((n, k))
        Cg = []
        for rows, o in groups:
            Lo = L[o]
            Pinv = 1.0 / Psi[o]
            C = np.linalg.inv(np.eye(k) + (Lo * Pinv[:, None]).T @ Lo)
            Mf[rows] = (E0[np.ix_(rows, np.flatnonzero(o))] * Pinv) @ Lo @ C
            Cg.append((rows, o, C))
        for j in range(p):
            A = np.zeros((k, k))
            bvec = np.zeros(k)
            ssq = 0.0
            wsum = 0.0
            for rows, o, C in Cg:
                if not o[j]:
                    continue
                ww = w[rows]
                Mr = Mf[rows]
                A += Mr.T @ (ww[:, None] * Mr) + float(ww.sum()) * C
                bvec += (ww * E0[rows, j]) @ Mr
                wsum += float(ww.sum())
            if wsum <= 0:
                continue
            L[j] = np.linalg.solve(A + 1e-8 * np.eye(k), bvec)
            for rows, o, C in Cg:
                if not o[j]:
                    continue
                ww = w[rows]
                Mr = Mf[rows]
                e = E0[rows, j]
                quad = (Mr @ L[j]) ** 2 + float(L[j] @ C @ L[j])
                ssq += float((ww * (e ** 2 - 2 * e * (Mr @ L[j]) + quad)).sum())
            Psi[j] = max(ssq / wsum, 1e-4 * var[j])
    S = L @ L.T + np.diag(Psi)
    return 0.5 * (S + S.T)


def masked_mse(E: np.ndarray, S: np.ndarray, rng: np.random.Generator, rate: float = 0.5, reps: int = 4) -> float:
    """MSE (standardized units) of predicting randomly hidden observed entries of E from the rest (qcore)."""
    n, p = E.shape
    obs = np.isfinite(E)
    tot, cnt = 0.0, 0
    for _ in range(reps):
        for i in range(n):
            o = obs[i].copy()
            hide = o & (rng.random(p) < rate)
            if not hide.any():
                continue
            keep = o & ~hide
            m, _ = _cond_gauss(np.zeros(p), S, keep, np.where(keep, E[i], 0.0))
            tot += float(np.sum((m[hide] - E[i, hide]) ** 2))
            cnt += int(hide.sum())
    return tot / max(cnt, 1)


@dataclass
class Satellites:
    fields: list[str]
    classes: list[str]
    d: np.ndarray                         # (S,)
    rho: np.ndarray                       # (S,)
    G0: np.ndarray                        # (S, NC) same-quarter core loadings
    G1: np.ndarray                        # (S, NC) previous-quarter core loadings
    Sigma: np.ndarray                     # (S, S) residual covariance
    resid: np.ndarray                     # (T, S) residuals on the core's sample quarters (NaN where missing)
    sd: np.ndarray                        # (S,) residual sd
    mean: np.ndarray                      # (S,) training means of the fields (initial-condition fallback)
    info: dict = field(default_factory=dict)
    excluded: dict = field(default_factory=dict)


def fit_satellites(Q: pd.DataFrame, core: CoreFit, train_end: str, seed: int = 0) -> Satellites:
    """Ridge satellite equations and their residual covariance on the core's sample quarters (<= train end)."""
    classes = field_classes()
    dates = core.dates
    K = pd.DataFrame(core.Y, index=dates, columns=CORE)
    Kl = K.shift(1)
    wq = pd.Series(1.0, index=dates)
    wq.iloc[P_LAGS:] = core.w
    fields, cls, rows_d, rows_rho, G0s, G1s, resid, sds, means, lams = [], [], [], [], [], [], [], [], [], {}
    excluded = {}
    for f in satellite_fields():
        c = classes[f]
        same = [v for v in CORE if CORE_CLASS[v] == "E0"] if c == "E0" else (
            [v for v in CORE if CORE_CLASS[v] in ("E0", "E0p")] if c == "E0p" else list(CORE))
        y = Q[f].reindex(dates)
        ylag = Q[f].shift(1).reindex(dates)
        cols = {"own": ylag}
        for v in same:
            cols[f"s_{v}"] = K[v] if v in RATE_VARS else K[v] - Kl[v]
        for v in CORE:
            cols[f"l_{v}"] = Kl[v]
        Xf = pd.DataFrame(cols)
        ok = y.notna() & Xf.notna().all(axis=1)
        if int(ok.sum()) < cq.SAT_MIN_OBS:
            excluded[f] = int(ok.sum())
            continue
        X = Xf[ok].to_numpy(float)
        yy = y[ok].to_numpy(float)
        ww = wq[ok].to_numpy(float)
        mu, sd = _std(X, ww)
        Z = (X - mu) / sd
        pm = np.ones(Z.shape[1])
        pm[0] = cq.SAT_OWN_LAG_PENALTY
        a, b, lam = ridge_cv(Z, yy, ww, pm)
        coef = b / sd
        const = a - float(mu @ coef)
        names = list(Xf.columns)
        g0, g1 = np.zeros(NC), np.zeros(NC)
        for nm_, cf in zip(names, coef):
            if nm_ == "own":
                continue
            kind, v = nm_.split("_", 1)
            if kind == "s":
                g0[IDX[v]] += cf
                if v not in RATE_VARS:
                    g1[IDX[v]] -= cf
            else:
                g1[IDX[v]] += cf
        fit_all = const + Xf.to_numpy(float) @ coef
        e = (y - fit_all).where(ok)
        fields.append(f)
        cls.append(c)
        rows_d.append(const)
        rows_rho.append(float(coef[0]))
        G0s.append(g0)
        G1s.append(g1)
        resid.append(e.to_numpy(float))
        sds.append(float(np.sqrt(np.average(e[ok].to_numpy(float) ** 2, weights=ww))))
        means.append(float(Q[f].loc[:pd.Period(train_end, "Q")].mean()))
        lams[f] = lam / float(ww.sum())
    R = np.column_stack(resid)
    sdv = np.array(sds)
    Ez = R / sdv
    w_rows = wq.to_numpy(float)
    rng = np.random.default_rng(seed)
    nrow = len(Ez)
    cut = int(cq.SAT_COV_SELECT_SHARE * nrow)
    cands = [("fa", kk, None) for kk in cq.SAT_COV_K] + [("lw", None, al) for al in cq.SAT_COV_LW]
    scores = {}
    for kind, kk, al in cands:
        S = em_cov(Ez[:cut], w_rows[:cut], kind, kk or 5, al or 0.3, seed=seed)
        scores[f"{kind}{kk if kk else al}"] = masked_mse(Ez[cut:], S, rng)
    choice = min(scores, key=scores.get)
    kind = "fa" if choice.startswith("fa") else "lw"
    par = float(choice[2:])
    Sz = em_cov(Ez, w_rows, kind, int(par) if kind == "fa" else 5, par if kind == "lw" else 0.3, seed=seed)
    Sig = Sz * np.outer(sdv, sdv)
    return Satellites(fields=fields, classes=cls, d=np.array(rows_d), rho=np.array(rows_rho), G0=np.vstack(G0s),
                      G1=np.vstack(G1s), Sigma=0.5 * (Sig + Sig.T), resid=R, sd=sdv, mean=np.array(means),
                      info={"cov_choice": choice, "cov_scores": scores, "ridge_lambda": lams}, excluded=excluded)


# ================================================================ statements

@dataclass(frozen=True)
class Statement:
    """One statement on the simulated path (quarters are 1-based within the path).

    var    an x field (d_ff, d_gz, x_oil_cause, ...), a policy shock (u_ff, u_bs, u_def, u_tr, u_ptax: an
           intervention, the do-operator) or a level observable: ff (observed policy rate), ff_shadow, y10, unrate,
           spread, gz, usd, real_oil, rff, ry10, pc (headline CPI inflation, annualized), cpi_yoy, core_yoy
    agg    "each" (every listed quarter on its own), "sum" or "mean" over the listed quarters; levels and y/y are
           read at the quarter's end
    value  the stated value (hard without sd); sd > 0 makes it soft; lo / hi make it a range
    """
    var: str
    quarters: tuple
    value: float | None = None
    sd: float | None = None
    lo: float | None = None
    hi: float | None = None
    agg: str = "each"
    label: str = ""

    @property
    def is_range(self) -> bool:
        return self.value is None and (self.lo is not None or self.hi is not None)


@dataclass
class _Row:
    """A compiled restriction. kind 'y': coef . y + const (stacked path); 'e': coef . e (stacked innovations);
    'u': sum_j c_j exp(lu_{h_j}) + const (unemployment levels through lu)."""
    kind: str
    stmt: int
    coef: np.ndarray | None = None        # (H, n) for y / e rows
    const: float = 0.0
    exp_terms: tuple = ()                 # ((h, c), ...) for u rows
    target: np.ndarray | float | None = None   # per path (n,) or scalar; None for ranges
    lo: float | None = None
    hi: float | None = None
    e1: bool = False                      # the row states an E1 / E1c / E2 field
    shock: bool = False
    quarter: int = 0                      # 0-based quarter the row is about (its last quarter)
    ffchange: bool = False                # a stated policy-rate change imposed on the shadow rate (prev unpinned)


@dataclass
class InitState:
    """The start-of-path state: the stacked core + satellites of the last two quarters (oldest first) and the
    levels the observables are measured from."""
    origin: str
    z: np.ndarray                         # (2, n)
    pi0: float                            # pi_trend at the end of the previous quarter
    ff0: float                            # observed policy rate
    u0: float
    lvl_cpi0: float                       # 100 log CPI
    pc_hist: np.ndarray                   # headline CPI rates of the last 3 quarters (oldest first)
    core_hist: np.ndarray                 # core CPI rates (d_core) of the last 3 quarters
    imputed: list = field(default_factory=list)
    information: str = "available"        # "published" (release lags at the origin's start) or "available"


# ================================================================ the model

@dataclass
class FactorModel:
    train_end: str
    anchors_mode: str
    anchors: dict
    core: CoreFit
    sats: Satellites
    mu: np.ndarray                         # steady state of the core (posterior mean model)
    c: np.ndarray                          # the posterior-mean model's constant (from the steady state)
    A: list                                # [A_1, A_2] posterior mean
    Sigma: np.ndarray                      # posterior mean core Sigma
    draws: list                            # [(A_list, c, Sigma), ...]
    rejected: int
    wg: float
    ws: float
    mu_star: np.ndarray
    mu_sd: np.ndarray
    pool: np.ndarray                       # (T_pool, n) whitened joint residual vectors
    pool_labels: np.ndarray                # quarter label of each pool row
    ip_hist: np.ndarray                    # training quarterly IP growth (annualized) for the cycle module
    ip_contraction: np.ndarray             # 1 where the training quarter is a contraction (regime classifier)
    regime_sigma: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    seed: int = 0
    tf: Transforms = field(default_factory=lambda: TRANSFORMS)   # the bounded scales of lu, sp, gz
    draw_mu_ip: np.ndarray = field(default_factory=lambda: np.zeros(0))   # steady-state IP growth of each draw
    pool_C: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # covariance of the (centred) pool rows
    train_frame: pd.DataFrame | None = None   # training-quarter series of every statement observable (saturation)
    shock_sd: dict = field(default_factory=dict)  # training sd of each policy shock (policy.shock_sd)
    _sd_cache: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- the simulator's own innovation covariance
    def pool_sigma(self, draw: int | None = None) -> np.ndarray:
        """Covariance of the reduced-form innovations the bootstrap draws: L C_pool L' (L the draw's Cholesky factor
        of its t-weighted Sigma, C_pool the covariance of the whitened pool rows, about 1.2-3.9 on the core
        diagonal: the pool is the equal-weight residuals, Sigma the t-weighted fit)."""
        L = self.system(draw)["L"]
        C = self.pool_C if self.pool_C.size else np.eye(self.n)
        S = L @ C @ L.T
        return 0.5 * (S + S.T)

    # ---------------------------------------------------------------- the cycle module's level (P4b)
    def cycle_shift(self, draw: int | None = None) -> float:
        """What the cycle module adds to the training IP blocks: 0 for config_q.CYCLE_CENTER 'training_mean' (P3's
        form, the default); for the option 'steady_state' the posterior draw's (or the posterior mean's) steady-state
        IP growth minus the training mean, so the blocks supply the cycle's shape and the anchors its level."""
        if cq.CYCLE_CENTER != "steady_state" or not len(self.ip_hist):
            return 0.0
        mu = self.mu[IDX["ip"]] if (draw is None or draw < 0 or not len(self.draw_mu_ip)) else self.draw_mu_ip[draw]
        return float(mu - np.mean(self.ip_hist))

    # ---------------------------------------------------------------- dimensions and names
    @property
    def n(self) -> int:
        return NC + len(self.sats.fields)

    @property
    def names(self) -> list[str]:
        return CORE + list(self.sats.fields)

    @property
    def classes(self) -> list[str]:
        cls = [CORE_CLASS[v] for v in CORE] + list(self.sats.classes)
        return ["last" if nm == cq.POLICY_DO["u_bs"] else c for nm, c in zip(self.names, cls)]

    def do_order(self) -> np.ndarray:
        """Stacked positions in the do-operator's recursive order: E0 -> E0p -> E1c / E1 / E2 -> d_bs."""
        cls = self.classes
        return np.array(sorted(range(self.n), key=lambda i: (RANK[cls[i]], i)))

    # ---------------------------------------------------------------- the joint system of one parameter draw
    def system(self, draw: int | None = None, regime: str | None = None) -> dict:
        """C (n,), Phi = [Phi_1, Phi_2], Sigma_u, L_u (its Cholesky), P (structural impact matrix in stacked order:
        columns are the recursive structural shocks, P P' = Sigma_u) for the posterior mean (draw None) or a draw."""
        A, c, Sk = (self.A, self.c, self.Sigma) if draw is None else self.draws[draw]
        if regime and regime in self.regime_sigma:
            Sk_r, Sw_r, nr = self.regime_sigma[regime]
            n0 = cq.REGIME_N0
            Sk = (nr * Sk_r + n0 * Sk) / (nr + n0)
            Sw = (nr * Sw_r + n0 * self.sats.Sigma) / (nr + n0)
        else:
            Sw = self.sats.Sigma
        s = self.sats
        S = len(s.fields)
        n = NC + S
        G0, G1 = s.G0, s.G1
        C = np.concatenate([c, s.d + G0 @ c])
        P1 = np.zeros((n, n))
        P2 = np.zeros((n, n))
        P1[:NC, :NC] = A[0]
        P1[NC:, :NC] = G0 @ A[0] + G1
        P1[NC:, NC:] = np.diag(s.rho)
        P2[:NC, :NC] = A[1]
        P2[NC:, :NC] = G0 @ A[1]
        Su = np.zeros((n, n))
        Su[:NC, :NC] = Sk
        Su[:NC, NC:] = Sk @ G0.T
        Su[NC:, :NC] = G0 @ Sk
        Su[NC:, NC:] = G0 @ Sk @ G0.T + Sw
        Su = 0.5 * (Su + Su.T)
        Lk = np.linalg.cholesky(Sk)
        Lw = np.linalg.cholesky(Sw + 1e-12 * np.eye(S)) if S else np.zeros((0, 0))
        Lu = np.zeros((n, n))
        Lu[:NC, :NC] = Lk
        Lu[NC:, :NC] = G0 @ Lk
        Lu[NC:, NC:] = Lw
        order = self.do_order()
        Pd = np.linalg.cholesky(Su[np.ix_(order, order)])
        P = np.zeros((n, n))
        P[np.ix_(order, order)] = Pd
        return {"C": C, "Phi": [P1, P2], "Sigma": Su, "L": Lu, "P": P, "order": order}

    # ---------------------------------------------------------------- initial state at an origin
    def initial_state(self, Q: pd.DataFrame, origin, information: str = "auto") -> InitState:
        """The state at the start of quarter `origin` (the first simulated quarter): the stacked core + satellites of
        the two quarters before it and the levels the observables are measured from.

        information (§3.2: every value is known at that time, after publication lags):
          'published'  only what was published by the start of the origin quarter: the satellites' <x>_l1__known /
                       _l2__known columns, the core's inputs by their own release lags (CORE_COLUMN_SOURCES). At a
                       backtest origin the previous quarter's CPI, IP and unemployment (and with them pi_trend, the
                       real rates and real oil) and the NIPA / Z.1 / profit satellites are not published yet;
          'available'  every value the panel holds (the forward root: the last data quarter is today's vintage);
          'auto'       'available' when the origin is the panel's "today" row (start-of-quarter states only, the
                       realized fields NaN), else 'published'.
        An unpublished entry is imputed from the joint distribution given everything published (the "impute"
        capability): the conditional mean of the quarter's innovation given the published entries and the identities
        that tie a known nominal value to the unknown inflation (ff = rff + pi_trend, y10 = ry10 + pi_trend, the
        nominal oil change = Δro + pc / 4, pi_trend's EWMA). A zero-bound quarter's shadow rate comes from the EM in
        the training sample, else from the filter on published core data, else it is imputed."""
        q0 = pd.Period(origin, "Q")
        info = _information(Q, q0, information)
        sysm = self.system()
        n = self.n
        base = np.concatenate([self.mu, self.sats.mean])
        if q0 - 3 in Q.index:
            zm3 = self._row_at(Q, q0 - 3)
            zm4 = self._row_at(Q, q0 - 4) if q0 - 4 in Q.index else zm3
        else:
            zm3 = zm4 = None
        imputed: list = []
        z = np.empty((2, n))
        pis: dict = {}
        for k, (r, q) in ((2, (0, q0 - 2)), (1, (1, q0 - 1))):
            v, cons, pi_q, shadow_from = self._published_row(Q, q0, k, info, pis.get(k + 1), z, r)
            if shadow_from:
                imputed.append(f"rff@{q}")                # the shadow rate (EM / filter): never observed
            obs = np.isfinite(v)
            if obs.all() and not cons:
                z[r] = v
            else:
                if r == 0:
                    lag1, lag2 = (zm3, zm4) if zm3 is not None else (None, None)
                else:
                    lag1, lag2 = z[0], (zm3 if zm3 is not None else z[0])
                if lag1 is None:
                    m = base.copy()
                else:
                    m = sysm["C"] + sysm["Phi"][0] @ lag1 + sysm["Phi"][1] @ lag2
                z[r] = _cond_linear(m, sysm["Sigma"], obs, np.where(obs, v, 0.0), cons)
                imputed += [f"{self.names[i]}@{q}" for i in np.flatnonzero(~obs)]
            if not np.isfinite(pi_q):                     # pi_trend from its EWMA on the (imputed) headline rate
                prev = pis.get(k + 1, _pub(Q, "lvl_infl_trend", q0, k + 1, info))
                pi_q = prev + ALPHA * (z[r, IDX["pc"]] - prev)
            pis[k] = pi_q
        last_ff = _pub(Q, "lvl_ff", q0, 1, info)
        if not np.isfinite(last_ff):
            last_ff = max(float(z[1, IDX["rff"]] + pis[1]), cq.FF_FLOOR)
        u0 = _pub(Q, "lvl_unrate", q0, 1, info)
        if not np.isfinite(u0):
            u0 = float(self.tf.level("lu", z[1, IDX["lu"]]))
        lc = _pub(Q, "lvl_cpi", q0, 1, info)
        if not np.isfinite(lc):
            lc2 = _pub(Q, "lvl_cpi", q0, 2, info)
            lc = lc2 + z[1, IDX["pc"]] / 4.0
        pc_hist = np.array([_pub(Q, "d_cpi", q0, 3, info), _pub(Q, "d_cpi", q0, 2, info),
                            _pub(Q, "d_cpi", q0, 1, info)])
        pc_hist = np.where(np.isfinite(pc_hist), pc_hist, [np.nan, z[0, IDX["pc"]], z[1, IDX["pc"]]])
        core_hist = np.array([_pub(Q, "d_core", q0, 3, info), _pub(Q, "d_core", q0, 2, info),
                              _pub(Q, "d_core", q0, 1, info)])
        if "d_core" in self.sats.fields:
            j = NC + self.sats.fields.index("d_core")
            core_hist = np.where(np.isfinite(core_hist), core_hist, [np.nan, z[0, j], z[1, j]])
        return InitState(origin=str(q0), z=z, pi0=float(pis[1]), ff0=float(last_ff), u0=float(u0),
                         lvl_cpi0=float(lc), pc_hist=pc_hist, core_hist=core_hist, imputed=imputed,
                         information=info)

    def _published_row(self, Q: pd.DataFrame, q0, k: int, info: str, pi_prev, z: np.ndarray, r: int):
        """(values (n,) with NaN where not published, linear constraints, pi_trend if published, whether the shadow
        rate came from the EM or the filter) of quarter q0 - k as known at the start of q0; constraints:
        [(coef (n,), rhs)] from a known nominal value whose real counterpart needs the unknown inflation (the EWMA
        pi = pi_prev + ALPHA (pc - pi_prev))."""
        q = q0 - k
        shadow_from = False
        tf = self.tf
        g = lambda c: _pub(Q, c, q0, k, info)                                   # noqa: E731
        v = np.full(self.n, np.nan)
        cons: list = []
        pc, ps, ip = g("d_cpi"), g("d_cpi_serv"), g("d_ip")
        v[IDX["pc"]], v[IDX["ps"]], v[IDX["ip"]] = pc, ps, 4.0 * ip
        u = g("lvl_unrate")
        v[IDX["lu"]] = float(tf.value("lu", u)) if np.isfinite(u) else np.nan
        for name, col in (("sp", "lvl_spread"), ("gz", "lvl_gz")):
            x = g(col)
            v[IDX[name]] = float(tf.value(name, x)) if np.isfinite(x) else np.nan
        v[IDX["usd"]] = g("lvl_usd")
        pi = g("lvl_infl_trend")
        if pi_prev is None:
            pi_prev = _pub(Q, "lvl_infl_trend", q0, k + 1, info)
        ff, y10 = g("lvl_ff"), g("lvl_y10")
        e_pc = np.zeros(self.n)
        e_pc[IDX["pc"]] = 1.0
        zlb = np.isfinite(ff) and ff < cq.ZLB_FF
        if zlb:                                            # the shadow rate: EM, filter on published data, or impute
            core_known = np.isfinite(v[[IDX["pc"], IDX["ps"], IDX["ip"], IDX["lu"]]]).all() and np.isfinite(pi)
            shadow = pd.Series(self.core.Y[:, IDX["rff"]], index=self.core.dates)
            if q in shadow.index:
                v[IDX["rff"]] = float(shadow.loc[q])
                shadow_from = True
            elif core_known:
                v[IDX["rff"]] = self._filter_shadow(Q, q)
                shadow_from = True
        elif np.isfinite(ff) and np.isfinite(pi):
            v[IDX["rff"]] = ff - pi
        elif np.isfinite(ff) and np.isfinite(pi_prev):     # rff + ALPHA pc = ff - (1 - ALPHA) pi_prev
            c = e_pc * ALPHA
            c[IDX["rff"]] = 1.0
            cons.append((c, ff - (1.0 - ALPHA) * pi_prev))
        if np.isfinite(y10) and np.isfinite(pi):
            v[IDX["ry10"]] = y10 - pi
        elif np.isfinite(y10) and np.isfinite(pi_prev):
            c = e_pc * ALPHA
            c[IDX["ry10"]] = 1.0
            cons.append((c, y10 - (1.0 - ALPHA) * pi_prev))
        ro = g("lvl_oil_real")
        if np.isfinite(ro):
            v[IDX["ro"]] = ro
        else:
            doil = g("d_oil")
            ro_prev = z[r - 1, IDX["ro"]] if r == 1 else _pub(Q, "lvl_oil_real", q0, k + 1, info)
            if np.isfinite(doil) and np.isfinite(ro_prev):  # ro + pc / 4 = ro_prev + d_oil (nominal oil known)
                c = e_pc * 0.25
                c[IDX["ro"]] = 1.0
                cons.append((c, ro_prev + doil))
        for j, f in enumerate(self.sats.fields):
            v[NC + j] = g(f)
        # a constraint whose unknowns are all observed already is redundant (it holds in the data)
        cons = [(c, h) for c, h in cons if not np.isfinite(v[np.flatnonzero(c)]).all()]
        return v, cons, pi, shadow_from

    def _row_at(self, Q: pd.DataFrame, q) -> np.ndarray:
        K = core_frame(Q, self.tf)
        v = K.loc[q].to_numpy(float).copy() if q in K.index else np.full(NC, np.nan)
        if not np.isfinite(v[IDX["rff"]]):
            sh = pd.Series(self.core.Y[:, IDX["rff"]], index=self.core.dates)
            v[IDX["rff"]] = sh.get(q, self.mu[IDX["rff"]])
        w = np.array([Q[f].get(q, np.nan) for f in self.sats.fields], float)
        z = np.concatenate([v, w])
        base = np.concatenate([self.mu, self.sats.mean])
        return np.where(np.isfinite(z), z, base)

    def _filter_shadow(self, Q: pd.DataFrame, q) -> float:
        """The shadow rate of a zero-bound quarter after the train end, imputed at the fitted parameters from the
        core data up to quarter q only (one-sided)."""
        D = core_frame(Q, self.tf).loc[self.core.dates[0]:q]
        Y = D.to_numpy(float)
        T = len(Y)
        wf = np.full(T - P_LAGS, np.nan)
        m = min(len(self.core.w), T - P_LAGS)
        wf[:m] = self.core.w[:m]
        fixed = {"B": np.vstack([np.vstack([a.T for a in self.A]), self.c[None, :]]), "Sigma": self.Sigma, "w": wf}
        r = _em_once(Y, P_LAGS, None, self.core.nu, True, 30, 1e-8, fixed=fixed)
        return float(r["Y"][-1, IDX["rff"]])

    # ---------------------------------------------------------------- deterministic pieces
    def mean_path(self, init: InitState, H: int, sysm: dict) -> np.ndarray:
        """(H, n) zero-shock path from the initial state."""
        C, (P1, P2) = sysm["C"], sysm["Phi"]
        z1, z0 = init.z[1].copy(), init.z[0].copy()
        out = np.empty((H, self.n))
        for h in range(H):
            y = C + P1 @ z1 + P2 @ z0
            out[h] = y
            z0, z1 = z1, y
        return out

    @staticmethod
    def responses(sysm: dict, X: np.ndarray) -> np.ndarray:
        """Forward recursion of shock sequences: X (H, n, q) stacked innovation columns -> paths (H, n, q)."""
        P1, P2 = sysm["Phi"]
        H = X.shape[0]
        Y = np.empty_like(X)
        for h in range(H):
            acc = X[h].copy()
            if h >= 1:
                acc += P1 @ Y[h - 1]
            if h >= 2:
                acc += P2 @ Y[h - 2]
            Y[h] = acc
        return Y

    @staticmethod
    def adjoint(sysm: dict, R: np.ndarray) -> np.ndarray:
        """G = R M for rows R (q, H, n) on the stacked path: (q, H, n) rows on the stacked innovations."""
        P1, P2 = sysm["Phi"]
        q, H, n = R.shape
        lam = np.zeros((H + 2, n, q))
        for h in range(H - 1, -1, -1):
            lam[h] = R[:, h, :].T + P1.T @ lam[h + 1] + P2.T @ lam[h + 2]
        return np.transpose(lam[:H], (2, 0, 1))


def _hist_quarterly(P: pd.DataFrame, end) -> pd.DataFrame:
    """Quarter-end CPI y/y, unemployment and IP level from the monthly panel, 1960Q2 .. end (the regime classifier's
    quarterly inputs, as the joint study used them)."""
    end_m = pd.Period(end, "Q").end_time.to_period("M").to_timestamp()
    M = P.loc["1960-01":end_m, ["s_cpi_yoy", "lvl_unrate", "lvl_ip"]]
    M = M[M.index.month.isin((3, 6, 9, 12))]
    M.index = M.index.to_period("Q")
    return M


def regime_contractions(P: pd.DataFrame, end) -> pd.Series:
    """Ex-post contraction flag per quarter from the regime classifier on quarter-end series (u_smooth 1 quarter),
    using data up to `end` only (jq2.regime_labels)."""
    from ..engine import regimes as RG
    H = _hist_quarterly(P, end)
    r = RG.classify(H["s_cpi_yoy"].to_numpy(float), H["lvl_unrate"].to_numpy(float), H["lvl_ip"].to_numpy(float),
                    params={**RG.PARAMS, "u_smooth_months": 1})
    return pd.Series(np.asarray(r["contraction"], float), index=H.index)


_MICH: list = []


def _michigan_lr():
    """The Michigan 5-10-year median (monthly) for the pi_lr rule, read once from the cache; None without it (the
    rule then falls back to the trailing 40-quarter mean of core CPI y/y)."""
    if not _MICH:
        try:
            from . import sources_q
            _MICH.append(sources_q.umich_px5())
        except (FileNotFoundError, OSError, ValueError, KeyError):
            _MICH.append(None)
    return _MICH[0]


def resolve_anchors(Q: pd.DataFrame, P: pd.DataFrame | None, train_end: str, mode: str,
                    tf: Transforms | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """(anchors, mu_star, mu_sd) for the steady-state prior. mode 'forward': the §3.8 defaults (判断值); 'train_end':
    the §5.5 rule from data up to the train end (engine/anchors.py; the Michigan 5-10-year median from 1990Q2). The
    unemployment and spread anchors are set on their bounded scale (tf), the sd by the delta method."""
    tf = TRANSFORMS if tf is None else tf
    te = pd.Period(train_end, "Q")
    sd = cq.STEADY_STATE_SD["forward" if mode == "forward" else "train_end"]
    H = Q.loc[:te]
    ro_med = float(H["lvl_oil_real"].median())
    usd_med = float(H["lvl_usd"].median())
    if mode == "forward":
        a = dict(cq.FORWARD_ANCHORS)
        out = {"mode": "forward", "source": "§3.8 判断值", "pc": a["pc"], "ps": a["ps"], "r_star": a["r_star"],
               "term": a["term_premium"], "spread": a["spread"], "unrate": a["unrate"]}
    else:
        if P is None:
            from ..data import panel as monthly_panel
            P = monthly_panel.load_panel()
        mich = _michigan_lr()
        end_m = te.end_time.to_period("M").to_timestamp()
        an = eng_anchors.train_end_anchors(P, end_m, mich_lr=mich)
        gap = float((H["d_cpi_serv"] - H["d_cpi"]).dropna().iloc[-40:].mean())
        out = {"mode": "train_end", "source": "§5.5 训练期末可知的规则", "pc": an["pi_lr"],
               "ps": an["pi_lr"] + gap, "r_star": an["r_star"], "term": an["term"], "spread": an["spread"],
               "unrate": an["unrate"], "pi_lr_source": an["pi_lr_source"], "services_gap": gap}
    out.update({"real_oil": ro_med, "usd": usd_med, "sd": sd})
    mu_star = np.zeros(NC)
    mu_sd = np.full(NC, cq.STEADY_STATE_DIFFUSE_SD)
    u_star = out["unrate"]
    mu_star[IDX["pc"]], mu_sd[IDX["pc"]] = out["pc"], sd
    mu_star[IDX["ps"]], mu_sd[IDX["ps"]] = out["ps"], sd
    for v, level in (("lu", u_star), ("sp", out["spread"])):
        x = float(tf.value(v, level))
        mu_star[IDX[v]], mu_sd[IDX[v]] = x, sd / float(tf.dlevel(v, x))
    mu_star[IDX["rff"]], mu_sd[IDX["rff"]] = out["r_star"], sd
    mu_star[IDX["ro"]], mu_sd[IDX["ro"]] = ro_med, sd
    mu_star[IDX["ry10"]], mu_sd[IDX["ry10"]] = out["r_star"] + out["term"], sd
    mu_star[IDX["usd"]], mu_sd[IDX["usd"]] = usd_med, sd
    return out, mu_star, mu_sd


def cycle_A(A: list[np.ndarray], Sig: np.ndarray) -> list[np.ndarray]:
    """The core dynamics the cycle module simulates: IP growth set to its target each quarter and the other variables
    moved along their innovation regression on IP (kgain), i.e. y_t = M (C + A1 y_{t-1} + A2 y_{t-2} + e_t) + k target_t
    with M = I - k e_ip' (P4b). The IP row of M A is zero: IP is exogenous, so the error correction that runs through
    IP's own equation (higher unemployment -> an IP rebound) is gone, and the kgain move feeds it back with the
    opposite sign."""
    ip = IDX["ip"]
    k = Sig[:, ip] / Sig[ip, ip]
    M = np.eye(len(Sig)) - np.outer(k, np.eye(len(Sig))[ip])
    return [M @ a for a in A]


def roots(A: list[np.ndarray], Sig: np.ndarray) -> tuple[float, float]:
    """(max companion root of the VAR, of the system the cycle module simulates)."""
    return max_root(A), max_root(cycle_A(A, Sig))


def stable(A: list[np.ndarray], Sig: np.ndarray, limit: float = cq.FACTOR_MAX_ROOT) -> bool:
    """The draw rule (§5.2 'explosive draws are redrawn', P4b): both the VAR and the system the cycle module simulates
    in statement-free quarters have every root below `limit` (config_q.FACTOR_MAX_ROOT: every mode at least halves
    within the longest simulated horizon, FACTOR_HALF_LIFE_MAX_Q)."""
    r_var, r_cyc = roots(A, Sig)
    return bool(r_var < limit and r_cyc < limit)


def fit(Q: pd.DataFrame, train_end: str, anchors: str = "train_end", P: pd.DataFrame | None = None, seed: int = 0,
        n_draws: int = cq.FACTOR_DRAWS, nu_grid=cq.FACTOR_NU_GRID, lam_grid=cq.FACTOR_LAMBDA_GRID, soc: bool = True,
        satellites: bool = True, steady: bool = True, shadow: str = cq.FACTOR_SHADOW,
        transform: str | None = None, bounds: str | None = None) -> FactorModel:
    """Fit the factor layer on quarters FIRST_QUARTER .. train_end only. anchors 'train_end' (backtests, §5.5) or
    'forward' (scenarios, §3.8). shadow 'missing' (§5.4 default: zero-bound quarters' shadow rate is missing data in
    the EM) or 'censored' (the pre-registered alternative: imputed at or below the bound). transform 'logit' (default,
    config_q.FACTOR_TRANSFORM: unemployment and the spreads on bounded scales) or 'log' (P3's log(level - lo), the
    pre-registered alternative of §15 Q5)."""
    te = pd.Period(train_end, "Q")
    tf = transforms(transform, bound_ceilings(Q, te, bounds))
    D = core_frame(Q, tf).loc[:te]
    D = D.loc[D.drop(columns="rff").notna().all(axis=1).idxmax():]          # first complete quarter
    upper = None
    if shadow == "censored":                     # the pre-registered alternative: shadow ff <= ZLB_FF at the bound
        upper = pd.DataFrame(np.nan, index=D.index, columns=D.columns)
        upper["rff"] = (cq.ZLB_FF - Q["lvl_infl_trend"]).reindex(D.index).where(D["rff"].isna())
    elif shadow != "missing":
        raise ValueError(f"shadow={shadow!r}: 'missing' (default) or 'censored'")
    core = fit_core(D, nu_grid=nu_grid, lam_grid=lam_grid, soc=soc, upper=upper)
    anc, mu_star, mu_sd = resolve_anchors(Q, P, str(te), anchors, tf)
    if not steady:
        mu_sd = np.full(NC, 1e12)
    A, _ = split_B(core.B, NC, P_LAGS)
    w = core.w
    mu, c = steady_state(A, core.Sigma, core.Y, w, mu_star, mu_sd)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    k = NC * P_LAGS + 1
    Lx = np.linalg.cholesky(np.linalg.inv(core.X1X1))
    draws, rej, fallback, mu_ip = [], 0, 0, []
    while len(draws) < n_draws:
        ok = False
        for _ in range(cq.FACTOR_DRAW_TRIES):
            Sd = invwishart.rvs(df=core.T1 - k, scale=core.S1, random_state=rng)
            Bd = core.B + Lx @ rng.standard_normal((k, NC)) @ np.linalg.cholesky(Sd).T
            Ad, _ = split_B(Bd, NC, P_LAGS)
            if stable(Ad, Sd):
                ok = True
                break
            rej += 1
        if not ok:
            Ad, Sd = A, core.Sigma
            fallback += 1
        mud, cd = steady_state(Ad, Sd, core.Y, w, mu_star, mu_sd, rng)
        draws.append((Ad, cd, Sd))
        mu_ip.append(float(mud[IDX["ip"]]))
    sats = fit_satellites(Q, core, str(te), seed=seed) if satellites else _no_satellites()
    wg, ws = cpi_weights(Q, te)
    if P is None:
        from ..data import panel as monthly_panel
        P = monthly_panel.load_panel()
    contr = regime_contractions(P, te)
    ipq = (4.0 * Q["d_ip"]).loc[D.index[0]:te].dropna()
    model = FactorModel(train_end=str(te), anchors_mode=anchors, anchors=anc, core=core, sats=sats, mu=mu, c=c, A=A,
                        Sigma=core.Sigma, draws=draws, rejected=rej, wg=wg, ws=ws, mu_star=mu_star, mu_sd=mu_sd,
                        pool=np.zeros((0, 0)), pool_labels=np.array([]), ip_hist=ipq.to_numpy(float),
                        ip_contraction=contr.reindex(ipq.index).fillna(0.0).to_numpy(float), seed=seed, tf=tf,
                        draw_mu_ip=np.array(mu_ip))
    pool_info = _build_pool(model, seed)
    _regime_sigmas(model, Q)
    model.train_frame = train_frame(Q, te)
    model.shock_sd = _shock_sd(Q, str(te))
    model.info = {"version": VERSION, "train_end": str(te), "sample": [str(D.index[0]), str(te)],
                  "T": int(len(D)), "lambda": core.lam, "nu": core.nu, "logml": core.logml,
                  "rejected_draws": rej, "rejection_rate": rej / (rej + n_draws), "n_draws": n_draws,
                  "draw_fallbacks": fallback, "max_root_mean": max_root(A),
                  "cycle_root_mean": max_root(cycle_A(A, core.Sigma)), "root_limit": cq.FACTOR_MAX_ROOT,
                  "root_rule": "VAR 与周期模块下的系统：每个根 < 上限（每个模态在最长模拟期内至少衰减一半）",
                  "transform": tf.summary(), "cycle_center": cq.CYCLE_CENTER,
                  "cycle_shift_mean": model.cycle_shift(None),
                  "pool_winsor_sd": cq.FACTOR_POOL_WINSOR_SD, "pool_winsorized": pool_info,
                  "steady_state": dict(zip(CORE, np.round(mu, 4).tolist())),
                  "zlb_quarters": [str(q) for q, m in zip(D.index, core.missing[:, IDX["rff"]]) if m],
                  "cpi_weights": {"goods": wg, "services": ws}, "satellites": len(sats.fields),
                  "satellites_excluded": sats.excluded, "sat_cov": sats.info.get("cov_choice"),
                  "smallest_weights": _smallest_weights(core), "anchors": anc, "shadow": shadow}
    return model


def train_frame(Q: pd.DataFrame, train_end) -> pd.DataFrame:
    """The training quarters (FIRST_QUARTER .. train end) of every statement observable: the x fields and d_cpi, the
    level observables (LEVEL_SERIES), the real rates, CPI and core CPI y/y (the saturation's training sds, §3.5)."""
    T = Q.loc[:pd.Period(train_end, "Q")]
    cols = {c: T[c] for c in list(qcard.x_fields(_card())) + ["d_cpi"] if c in T.columns}
    for obs, col in LEVEL_SERIES.items():
        if col in T.columns:
            cols[obs] = T[col]
    if {"lvl_ff", "lvl_y10", "lvl_infl_trend"} <= set(T.columns):
        cols["rff"] = T["lvl_ff"] - T["lvl_infl_trend"]
        cols["ry10"] = T["lvl_y10"] - T["lvl_infl_trend"]
    if "lvl_cpi" in T.columns:
        cols["cpi_yoy"] = 100.0 * (np.exp((T["lvl_cpi"] - T["lvl_cpi"].shift(4)) / 100.0) - 1.0)
    if "d_core" in T.columns:
        cols["core_yoy"] = 100.0 * (np.exp(T["d_core"].rolling(4).sum() / 400.0) - 1.0)
    return pd.DataFrame(cols).astype(float)


def _shock_sd(Q: pd.DataFrame, train_end: str) -> dict:
    """Training sd of each policy shock (the residual sd of its reaction function, policy.shock_sd)."""
    try:
        from . import policy
        return {k: float(v) for k, v in policy.shock_sd(policy.fit(Q, train_end)).items() if np.isfinite(v)}
    except Exception:  # noqa: BLE001 - a shock without a fitted equation falls back to its model sd
        return {}


def statement_sd(model: "FactorModel", st: "Statement") -> float:
    """The training sd a stated value's gap to its conditional mean is measured in (§3.5: ±4 training sd): the field's
    sd over the training quarters (field_stats at the train end); a sum or mean over k quarters, the sd of its k-quarter
    sums or means; a level h quarters ahead (ff, unrate, y10, the spreads, CPI y/y, ...), the sd of the level's
    h-quarter change; a policy shock, its reaction function's residual sd. NaN if the training data have none."""
    var = "d_cpi" if st.var == "pc" else st.var
    qs = sorted(int(q) for q in st.quarters)
    key = (var, st.agg, tuple(qs))
    hit = model._sd_cache.get(key)
    if hit is not None:
        return hit
    k = len(qs)
    if var in SHOCKS:
        s = model.shock_sd.get(var, float("nan"))
        if not np.isfinite(s) and cq.POLICY_DO[var] in model.names:
            pos = model.names.index(cq.POLICY_DO[var])
            s = float(model.system()["P"][pos, pos])
        out = s * (math.sqrt(k) if st.agg == "sum" else 1.0 / math.sqrt(k) if st.agg == "mean" else 1.0)
    elif model.train_frame is None or var not in model.train_frame.columns:
        out = float("nan")
    else:
        x = model.train_frame[var]
        level = var in LEVEL_SERIES or var in ("rff", "ry10", "cpi_yoy", "core_yoy")
        if level:
            ch = [x.shift(-h) - x for h in qs]           # the level's change h quarters after a quarter's end
            s = sum(ch) / (k if st.agg != "sum" else 1.0) if st.agg in ("sum", "mean") else ch[-1]
        elif st.agg == "sum":
            s = x.rolling(k).sum()
        elif st.agg == "mean":
            s = x.rolling(k).mean()
        else:
            s = x
        s = s.dropna()
        out = float(s.std()) if len(s) > 2 else float("nan")
    model._sd_cache[key] = out
    return out


def _no_satellites() -> Satellites:
    return Satellites(fields=[], classes=[], d=np.zeros(0), rho=np.zeros(0), G0=np.zeros((0, NC)),
                      G1=np.zeros((0, NC)), Sigma=np.zeros((0, 0)), resid=np.zeros((0, 0)), sd=np.zeros(0),
                      mean=np.zeros(0))


def _smallest_weights(core: CoreFit, k: int = 6) -> list:
    dates = core.dates[P_LAGS:]
    order = np.argsort(core.w)[:k]
    return [[str(dates[i]), round(float(core.w[i]), 4), round(float(core.m2[i]), 1)] for i in order]


def _build_pool(model: FactorModel, seed: int) -> dict:
    """Whitened joint residual vectors of the training quarters (the bootstrap pool): the core's residuals at the
    posterior-mean model whitened by its Sigma; the satellites' (missing entries drawn from their conditional
    Gaussian once, by seed) whitened by theirs. The pool is centred (each coordinate's mean removed): the residuals at
    the posterior-mean model (the steady-state prior's intercept, shrinkage and t-weights) are not mean-zero, and
    (I - A1 - A2)^-1 turns a small mean into a long-run drift away from the anchors (forward fit: CPI +1.0, services
    CPI +1.2, real oil +11 log% before centring), so E[u] = 0 under every posterior draw and the bootstrap's mean path
    is the model's. With config_q.FACTOR_POOL_WINSOR_SD (an option, off by default) each coordinate is then winsorized
    at that multiple of its own sd in the pool: every quarter keeps its frequency (U-12) but not its size (2020Q2's
    unemployment coordinate is 11 of its sd). Returns what was cut and the mean removed."""
    core = model.core
    Y = core.Y
    A, c = model.A, model.c
    T = len(Y)
    E = Y[P_LAGS:] - c - sum(Y[P_LAGS - lag:T - lag] @ A[lag - 1].T for lag in range(1, P_LAGS + 1))
    Lk = np.linalg.cholesky(model.Sigma)
    Uk = np.linalg.solve(Lk, E.T).T
    S = len(model.sats.fields)
    if S:
        R = model.sats.resid[P_LAGS:]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
        Sw = model.sats.Sigma
        R = R.copy()
        for i in range(len(R)):
            o = np.isfinite(R[i])
            if o.all():
                continue
            m, C = _cond_gauss(np.zeros(S), Sw, o, np.where(o, R[i], 0.0))
            mis = ~o
            Lc = np.linalg.cholesky(C + 1e-12 * np.eye(int(mis.sum())))
            R[i, mis] = m[mis] + Lc @ rng.standard_normal(int(mis.sum()))
        Lw = np.linalg.cholesky(Sw + 1e-12 * np.eye(S))
        Uw = np.linalg.solve(Lw, R.T).T
        U = np.hstack([Uk, Uw])
    else:
        U = Uk
    labels = np.array([str(q) for q in core.dates[P_LAGS:]])
    mean = U.mean(axis=0)
    se = U.std(axis=0, ddof=1) / math.sqrt(max(len(U), 1))
    U = U - mean[None, :]                            # E[u] = 0: no drift from the bootstrap (centred before winsor)
    out = {"entries": 0, "core_entries": 0, "rows": [], "centred": True,
           "mean_removed_core": dict(zip(CORE, np.round(mean[:NC], 4).tolist())),
           "mean_removed_max_abs_t": float(np.max(np.abs(mean / np.where(se > 0, se, 1.0)))) if len(U) else 0.0}
    c = cq.FACTOR_POOL_WINSOR_SD
    if c:
        lim = c * U.std(axis=0)[None, :]
        cut = np.abs(U) > lim
        out.update({"entries": int(cut.sum()), "core_entries": int(cut[:, :NC].sum()),
                    "rows": sorted({str(labels[i]) for i in np.flatnonzero(cut[:, :NC].any(axis=1))})})
        U = np.clip(U, -lim, lim)                    # (the option clips a centred pool; a skewed tail's cut moves
        out["mean_after_winsor_max_abs"] = float(np.max(np.abs(U.mean(axis=0))))   # its mean a little: reported)
    model.pool = U
    model.pool_labels = labels
    model.pool_C = (U.T @ U) / max(len(U), 1)
    return out


def _regime_sigmas(model: FactorModel, Q: pd.DataFrame) -> None:
    """Regime-specific covariances (s_zlb, s_core4) of the core and satellite residuals, for the T-C candidate."""
    dates = model.core.dates[P_LAGS:]
    s_zlb = Q["s_zlb"].reindex(dates).to_numpy(float) > 0.5
    s_core4 = Q["s_core4"].reindex(dates).to_numpy(float) > 0.5
    regimes = {"zlb": s_zlb, "core4": s_core4 & ~s_zlb}
    Y = model.core.Y
    T = len(Y)
    E = Y[P_LAGS:] - model.c - sum(Y[P_LAGS - lag:T - lag] @ model.A[lag - 1].T for lag in range(1, P_LAGS + 1))
    w = model.core.w
    R = model.sats.resid[P_LAGS:] if len(model.sats.fields) else None
    for name, m in regimes.items():
        nr = int(m.sum())
        if nr < 8:
            continue
        ww = w[m]
        Sk = (E[m] * ww[:, None]).T @ E[m] / ww.sum()
        if R is not None:
            Rr = R[m]
            Sw = np.zeros((Rr.shape[1], Rr.shape[1]))
            for i in range(Rr.shape[1]):
                for j in range(i, Rr.shape[1]):
                    ok = np.isfinite(Rr[:, i]) & np.isfinite(Rr[:, j])
                    v = float((Rr[ok, i] * Rr[ok, j] * ww[ok]).sum() / max(ww[ok].sum(), 1e-12)) if ok.sum() > 3 \
                        else float(model.sats.Sigma[i, j])
                    Sw[i, j] = Sw[j, i] = v
            # keep it a covariance: shrink toward the pooled one until positive definite
            for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
                Sm = (1 - lam) * Sw + lam * model.sats.Sigma
                if np.linalg.eigvalsh(Sm).min() > 1e-10:
                    Sw = Sm
                    break
        else:
            Sw = np.zeros((0, 0))
        model.regime_sigma[name] = (0.5 * (Sk + Sk.T), 0.5 * (Sw + Sw.T), nr)


# ================================================================ statements -> restrictions

class _Expr:
    """An affine function of the stacked path y (H, n): coef . y + const."""

    __slots__ = ("coef", "const")

    def __init__(self, H: int, n: int):
        self.coef = np.zeros((H, n))
        self.const = 0.0

    def add(self, other: "_Expr", scale: float = 1.0) -> "_Expr":
        self.coef += scale * other.coef
        self.const += scale * other.const
        return self


class _Compiler:
    """Builds affine expressions of the model observables from the stacked path and the initial state."""

    def __init__(self, model: FactorModel, init: InitState, H: int):
        self.m, self.init, self.H, self.n = model, init, H, model.n
        self.sat = {f: NC + j for j, f in enumerate(model.sats.fields)}

    def var(self, pos: int, h: int) -> _Expr:
        e = _Expr(self.H, self.n)
        if h >= 0:
            e.coef[h, pos] = 1.0
        else:
            e.const = float(self.init.z[h + 2][pos])
        return e

    def core(self, v: str, h: int) -> _Expr:
        return self.var(IDX[v], h)

    def pi(self, h: int) -> _Expr:
        e = _Expr(self.H, self.n)
        if h < 0:
            e.const = self.init.pi0
            return e
        e.const = (1.0 - ALPHA) ** (h + 1) * self.init.pi0
        for j in range(h + 1):
            e.coef[j, IDX["pc"]] = ALPHA * (1.0 - ALPHA) ** (h - j)
        return e

    def shadow(self, h: int) -> _Expr:
        if h < 0:
            e = _Expr(self.H, self.n)
            e.const = self.init.ff0 if self.init.ff0 > cq.FF_FLOOR + 1e-9 else float(
                self.init.z[1][IDX["rff"]] + self.init.pi0)
            return e
        return self.core("rff", h).add(self.pi(h))

    def y10(self, h: int) -> _Expr:
        return self.core("ry10", h).add(self.pi(h))

    def field(self, f: str, h: int) -> _Expr:
        """Linear observables per quarter (d_unrate, d_ff, unrate, ff, y/y handled by the caller)."""
        if f in self.sat:
            return self.var(self.sat[f], h)
        if f == "d_cpi_goods":
            return _scaled(self.core("pc", h).add(self.core("ps", h), -self.m.ws), 1.0 / self.m.wg)
        if f in ("d_cpi_serv",):
            return self.core("ps", h)
        if f in ("pc", "d_cpi"):
            return self.core("pc", h)
        if f == "d_ip":
            return _scaled(self.core("ip", h), 0.25)
        if f == "d_oil":
            return self.core("ro", h).add(self.core("ro", h - 1), -1.0).add(self.core("pc", h), 0.25)
        if f == "d_y10":
            return self.y10(h).add(self.y10(h - 1), -1.0)
        if f in ("d_spread", "d_gz", "d_usd") and f not in self.m.tf.obs:
            v = {"d_spread": "sp", "d_gz": "gz", "d_usd": "usd"}[f]
            return self.core(v, h).add(self.core(v, h - 1), -1.0)
        if f == "d_ff":
            return self.shadow(h).add(self.shadow(h - 1), -1.0)
        if f == "ff_shadow":
            return self.shadow(h)
        if f in ("y10", "y10_shadow"):           # statements on the 10-year are imposed on its shadow (P4b floor)
            return self.y10(h)
        if f in ("spread", "gz", "usd", "real_oil", "rff", "ry10") and f not in self.m.tf.obs:
            return self.core({"spread": "sp", "real_oil": "ro"}.get(f, f), h)
        if f == "cpi_yoy":
            e = _Expr(self.H, self.n)
            for j in range(h - 3, h + 1):
                if j >= 0:
                    e.coef[j, IDX["pc"]] += 0.25
                else:
                    e.const += 0.25 * float(self.init.pc_hist[3 + j])
            return e
        if f == "core_yoy":
            if "d_core" not in self.sat:
                raise FactorError("核心通胀（d_core）在这个训练期没有估计，不能写核心 CPI 同比")
            e = _Expr(self.H, self.n)
            for j in range(h - 3, h + 1):
                if j >= 0:
                    e.coef[j, self.sat["d_core"]] += 0.25
                else:
                    e.const += 0.25 * float(self.init.core_hist[3 + j])
            return e
        raise FactorError(f"字段不存在：{f}")


def _scaled(e: _Expr, s: float) -> _Expr:
    e.coef *= s
    e.const *= s
    return e


def _known(model: FactorModel, var: str) -> bool:
    return (var in CORE_FIELDS or var in model.sats.fields or var in LEVEL_OBS or var in SHOCKS
            or var in ("d_cpi",))


def _yoy_target(v):
    """A y/y rate (%) as the sum of the four quarters' annualized rates / 4 (= 100 x the log change over the year)."""
    return 100.0 * np.log1p(np.asarray(v, float) / 100.0)


def compile_statements(model: FactorModel, init: InitState, H: int, stmts: list[Statement],
                       targets: list[np.ndarray | None], sysm: dict) -> list[_Row]:
    """Rows for the statements with their per-path targets (targets[i]: (n_paths,) observable targets of a value
    statement, already saturated and noise-drawn; None for ranges)."""
    cmp_ = _Compiler(model, init, H)
    classes = field_classes()
    tobs = model.tf.obs
    rows: list[_Row] = []
    order = sorted(range(len(stmts)), key=lambda i: (min(stmts[i].quarters), i))
    pinned: dict[str, dict] = {}                  # per bounded variable: {quarter: level known on every path}
    pinned_ff: dict[int, np.ndarray | float] = {-1: init.ff0}
    for i in order:
        st = stmts[i]
        qs = [int(q) - 1 for q in st.quarters]
        if not qs or min(qs) < 0 or max(qs) >= H:
            raise FactorError(f"{st.var}: 季度 {st.quarters} 超出模拟期（1..{H}）")
        if not _known(model, st.var):
            raise FactorError(f"字段不存在或这个训练期没有估计：{st.var}")
        t = targets[i]
        e1 = classes.get(st.var, "E0") in E1_CLASSES or st.var in ("y10", "y10_shadow", "spread", "gz", "usd",
                                                                   "real_oil", "ry10")
        # ---------------------------------------------------------- interventions: structural shocks
        if st.var in SHOCKS:
            pos = model.names.index(cq.POLICY_DO[st.var]) if cq.POLICY_DO[st.var] in model.names else None
            if pos is None:
                raise FactorError(f"{st.var}：这个训练期没有对应的政策方程")
            P = sysm["P"]
            Pinv_row = np.linalg.solve(P.T, np.eye(model.n)[:, pos])     # row pos of P^-1 (as a column)
            s = P[pos, pos] * Pinv_row
            for h in (qs if st.agg == "each" else [qs[-1]]):
                coef = np.zeros((H, model.n))
                if st.agg == "each":
                    coef[h] = s
                else:
                    for hh in qs:
                        coef[hh] = s / (len(qs) if st.agg == "mean" else 1.0)
                rows.append(_Row(kind="e", stmt=i, coef=coef, target=t, lo=st.lo, hi=st.hi, shock=True, quarter=h))
            continue
        # ---------------------------------------------------------- bounded variables (unemployment, spreads)
        if st.var in tobs:
            v, kind = tobs[st.var]
            pin = pinned.setdefault(v, {-1: _level0(model, init, v)})
            if st.agg == "each":
                for h in qs:
                    if kind == "level":
                        rows.append(_exp_level_row(cmp_, v, i, h, t, st, e1=e1))
                        if t is not None:
                            pin[h] = _clip_level(model, v, t)
                        continue
                    prev = pin.get(h - 1)
                    if prev is not None and (t is not None or np.ndim(prev) == 0):
                        tt = None if t is None else prev + t
                        rows.append(_exp_level_row(cmp_, v, i, h, tt, st, base=prev, e1=e1))
                        if tt is not None:
                            pin[h] = _clip_level(model, v, tt)
                    else:
                        rows.append(_urow(model, init, i, ((h, v, 1.0), (h - 1, v, -1.0)), 0.0, t, st, h, e1=e1))
                continue
            if kind == "level":                          # mean / sum of levels
                m_ = len(qs) if st.agg == "mean" else 1.0
                rows.append(_urow(model, init, i, tuple((h, v, 1.0 / m_) for h in qs), 0.0, t, st, qs[-1], e1=e1))
                continue
            a, b = min(qs), max(qs)                      # sum (or mean) of changes over a..b = (x_b - x_{a-1}) [/m]
            m_ = (b - a + 1) if st.agg == "mean" else 1.0
            prev = pin.get(a - 1)
            if prev is not None and np.ndim(prev) == 0 and st.agg == "sum":
                tt = None if t is None else prev + t
                rows.append(_exp_level_row(cmp_, v, i, b, tt, st, base=prev, e1=e1))
            else:
                rows.append(_urow(model, init, i, ((b, v, 1.0 / m_), (a - 1, v, -1.0 / m_)), 0.0, t, st, b,
                                  e1=e1))
            continue
        # ---------------------------------------------------------- the observed policy rate (floor at FF_FLOOR)
        if st.var in ("d_ff", "ff") and st.agg == "each":
            for h in qs:
                if st.var == "ff":
                    lvl = t
                else:
                    prev = pinned_ff.get(h - 1)
                    lvl = None if (prev is None or t is None) else prev + t
                if lvl is None:
                    ex = cmp_.shadow(h) if st.var == "ff" else cmp_.field("d_ff", h)
                    rows.append(_Row(kind="y", stmt=i, coef=ex.coef, const=ex.const, target=t, lo=st.lo, hi=st.hi,
                                     quarter=h, ffchange=st.var == "d_ff" and t is not None))
                    continue
                lvl = np.maximum(np.asarray(lvl, float), cq.FF_FLOOR)
                ex = cmp_.shadow(h)
                if np.all(lvl > cq.FF_FLOOR + 1e-9) or np.ndim(lvl) > 0:
                    rows.append(_Row(kind="y", stmt=i, coef=ex.coef, const=ex.const, target=lvl, quarter=h))
                else:                                    # at the floor: the shadow rate is at or below it
                    rows.append(_Row(kind="y", stmt=i, coef=ex.coef, const=ex.const, target=None, lo=None,
                                     hi=cq.FF_FLOOR, quarter=h))
                pinned_ff[h] = lvl
            continue
        # ---------------------------------------------------------- everything linear
        hs = qs if st.agg == "each" else [None]
        for h in hs:
            if h is None:
                ex = _Expr(H, model.n)
                for hh in qs:
                    ex.add(cmp_.field(st.var, hh), 1.0 / len(qs) if st.agg == "mean" else 1.0)
                hq = qs[-1]
            else:
                ex = cmp_.field(st.var, h)
                hq = h
            tt, lo, hi = t, st.lo, st.hi
            if st.var in ("cpi_yoy", "core_yoy"):
                f_ = (lambda v: None if v is None else _yoy_target(v))
                tt, lo, hi = f_(tt), f_(lo), f_(hi)
            rows.append(_Row(kind="y", stmt=i, coef=ex.coef, const=ex.const, target=tt, lo=lo, hi=hi, e1=e1,
                             quarter=hq))
    return rows


def _clip_level(model: FactorModel, v: str, x):
    """A stated level kept _LEVEL_EPS inside the variable's bounds (the level its row delivers)."""
    b = model.tf.vars[v]
    top = b.hi if b.kind == "logit" else math.inf
    return np.clip(np.asarray(x, float), b.lo + _LEVEL_EPS, top - _LEVEL_EPS)


def _level0(model: FactorModel, init: InitState, v: str) -> float:
    """The level of a bounded core variable at the end of the quarter before the path."""
    return float(model.tf.level(v, init.z[1][IDX[v]]))


def _exp_level_row(cmp_: _Compiler, v: str, i: int, h: int, level, st: Statement, base=None, e1: bool = False) -> _Row:
    """A row on the bounded core variable v at quarter h for a stated level (target: the level on v's scale, kept just
    inside the bounds); ranges transform the same way (relative to `base` for a stated change)."""
    ex = cmp_.core(v, h)
    tf = cmp_.m.tf

    def tr(x):
        if x is None:
            return None
        return tf.value(v, x, clip=True)

    lo = None if st.lo is None else (st.lo if base is None else base + st.lo)
    hi = None if st.hi is None else (st.hi if base is None else base + st.hi)
    return _Row(kind="y", stmt=i, coef=ex.coef, const=ex.const, target=tr(level), lo=tr(lo), hi=tr(hi), quarter=h,
                e1=e1)


def _urow(model: FactorModel, init: InitState, i: int, terms: tuple, const: float, target, st: Statement,
          quarter: int, e1: bool = False) -> _Row:
    """A row sum_j c_j level_v(x_{h_j}) + const on bounded core variables (terms (h, variable, c); level_v the
    variable's level, model.tf); terms before the path (h < 0) are known and folded into the constant."""
    keep = []
    for h, v, c in terms:
        if h >= 0:
            keep.append((h, IDX[v], c))
        else:
            const += c * float(model.tf.level(v, init.z[h + 2][IDX[v]]))
    return _Row(kind="u", stmt=i, exp_terms=tuple(keep), const=const, target=target, lo=st.lo, hi=st.hi,
                quarter=quarter, e1=e1)


# ================================================================ simulation

@dataclass
class SimResult:
    """Simulated factor paths (n_paths, H) in schema units, with levels, shocks, delivery and rarity."""
    x: dict                               # the 52 x fields + d_cpi (headline) in card units
    levels: dict                          # ff, ff_shadow, y10, unrate, spread, gz, usd, real_oil, pi_trend, cpi (100 log),
                                          # cpi_yoy, core_yoy, zlb, core4
    shocks: dict                          # u_ff, u_bs, u_def, u_tr, u_ptax: structural shocks in their field's units
                                          # (as imposed; levels["u_ff_at_floor"] marks quarters where the observed
                                          # policy rate sits at the floor, the measured u_ff = 0 convention)
    core: np.ndarray                      # (n_paths, H, NC) core variables
    sats: np.ndarray                      # (n_paths, H, S)
    innov: np.ndarray                     # (n_paths, H, n) reduced-form innovations (after conditioning)
    drawn: np.ndarray                     # (n_paths, H) the training quarter each residual vector came from
    draw_index: np.ndarray                # (n_paths,) posterior draw of each path (-1 = posterior mean)
    statements: list                      # the statements as imposed (one quarter each unless aggregated)
    delivery: list
    rarity: dict
    notes: list
    info: dict


def expand(statements) -> list[Statement]:
    """'each' statements over several quarters become one statement per quarter (each its own soft draw)."""
    out = []
    for st in statements:
        if st.agg == "each" and len(st.quarters) > 1:
            out += [Statement(st.var, (q,), st.value, st.sd, st.lo, st.hi, "each", st.label) for q in st.quarters]
        else:
            out.append(st)
    return out


def _quarter_flags(stmts: list[Statement], H: int) -> dict:
    classes = field_classes()
    growth = np.zeros(H, bool)
    e1 = np.zeros(H, bool)
    shock = np.zeros(H, bool)
    usd = False
    for st in stmts:
        for q in st.quarters:
            h = int(q) - 1
            if not 0 <= h < H:
                continue
            if st.var in cq.GROWTH_FIELDS:
                growth[h] = True
            if st.var in SHOCKS:
                shock[h] = True
            elif classes.get(st.var, "E0") in E1_CLASSES or st.var in ("y10", "y10_shadow", "spread", "gz", "usd",
                                                                          "real_oil", "ry10"):
                e1[h] = True
        if st.var in ("d_usd", "usd"):
            usd = True
    return {"growth": growth, "e1": e1, "shock": shock, "usd": usd}


def _metric(model: FactorModel, Su: np.ndarray, H: int, flags: dict, w: float) -> np.ndarray:
    """(H, n, n) projection metric: Sigma_u, with Cov(E1-class, macro) scaled by w in quarters that state no E1 /
    E1c / E2 field and no intervention (completion shrinkage, §3.5)."""
    V = np.broadcast_to(Su, (H, model.n, model.n)).copy()
    if w < 1.0:
        cls = np.array(model.classes)
        e1m, mac = np.isin(cls, E1_CLASSES), np.isin(cls, MACRO_CLASSES)
        Sw = Su.copy()
        Sw[np.ix_(e1m, mac)] *= w
        Sw[np.ix_(mac, e1m)] *= w
        for h in range(H):
            if not flags["e1"][h] and not flags["shock"][h]:
                V[h] = Sw
    return V


def _draw_whitened(model: FactorModel, rng: np.random.Generator, m: int, H: int, gaussian: bool, block: int):
    """(m, H, n) whitened innovation vectors and their source labels; drawn quarter by quarter, so a path's first
    quarters do not depend on H (step / path consistency)."""
    n = model.n
    if gaussian:
        U = np.empty((m, H, n))
        for h in range(H):
            U[:, h, :] = rng.standard_normal((m, n))
        return U, np.full((m, H), "gauss", dtype=object)
    Tp = len(model.pool)
    if block <= 1:
        idx = np.empty((m, H), dtype=int)
        for h in range(H):
            idx[:, h] = rng.integers(0, Tp, size=m)
    else:
        nb = int(math.ceil(H / block))
        starts = np.empty((m, nb), dtype=int)
        for b in range(nb):
            starts[:, b] = rng.integers(0, Tp - block + 1, size=m)
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(m, -1)[:, :H]
    return model.pool[idx], model.pool_labels[idx].astype(object)


def _tilt(share: np.ndarray, target: float) -> np.ndarray:
    """Equal weights tilted exponentially so the blocks' expected contraction share equals `target` (minimum relative
    entropy to the equal weights under that one moment)."""
    target = float(np.clip(target, share.min() + 1e-6, share.max() - 1e-6))
    lo, hi = -60.0, 60.0
    w = np.full(len(share), 1.0 / len(share))
    for _ in range(200):
        th = 0.5 * (lo + hi)
        w = np.exp(th * (share - share.mean()))
        w /= w.sum()
        if float(w @ share) < target:
            lo = th
        else:
            hi = th
    return w


def cycle_targets(model: FactorModel, rng_blocks: np.random.Generator, rng_noise: np.random.Generator, m: int,
                  H: int, odds: dict | None = None) -> np.ndarray:
    """(m, H) IP growth targets of the cycle module C' (§5.6) before its level shift (FactorModel.cycle_shift, added
    per posterior draw by the caller): 2-quarter blocks of the training quarters' IP growth
    (equal weights, or exponentially tilted to stated contraction odds in the stated quarters) plus N(0, sd^2) with
    sd = CYCLE_SD_FACTOR x the training quarterly IP sd (the soft statement's sd)."""
    ip = model.ip_hist
    blk = cq.CYCLE_BLOCK
    nstart = len(ip) - blk + 1
    share = np.array([model.ip_contraction[s:s + blk].mean() for s in range(nstart)])
    nb = int(math.ceil(H / blk))
    sd = cq.CYCLE_SD_FACTOR * float(np.std(ip, ddof=1))
    equal = np.full(nstart, 1.0 / nstart)
    tilted = _tilt(share, float(odds["contraction"])) if odds and odds.get("contraction") is not None else None
    qset = {int(q) - 1 for q in (odds or {}).get("quarters", range(1, H + 1))}
    out = np.empty((m, nb * blk))
    for b in range(nb):
        hq = b * blk
        s = rng_blocks.choice(nstart, size=m, p=tilted if (tilted is not None and hq in qset) else equal)
        out[:, hq:hq + blk] = ip[s[:, None] + np.arange(blk)[None, :]]
    noise = np.empty((m, H))
    for h in range(H):
        noise[:, h] = rng_noise.standard_normal(m)
    return out[:, :H] + sd * noise


def _run_paths(model: FactorModel, init: InitState, sysm: dict, U: np.ndarray, H: int, cyc: np.ndarray | None,
               active: np.ndarray, Lq: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Recursion from the whitened draws; in the cycle module's quarters the quarter's own innovation is moved so IP
    growth meets its target exactly, the other variables along their innovation regression on IP (one-sided)."""
    C, (P1, P2) = sysm["C"], sysm["Phi"]
    m, n = U.shape[0], model.n
    E = U @ sysm["L"].T if Lq is None else np.einsum("hab,mhb->mha", Lq, U)
    Su = sysm["Sigma"]
    ip = IDX["ip"]
    kgain = Su[:, ip] / Su[ip, ip]
    Y = np.empty((m, H, n))
    z1 = np.broadcast_to(init.z[1], (m, n)).copy()
    z0 = np.broadcast_to(init.z[0], (m, n)).copy()
    for h in range(H):
        pred = C + z1 @ P1.T + z0 @ P2.T
        if cyc is not None and active[h]:
            gap = cyc[:, h] - (pred[:, ip] + E[:, h, ip])
            E[:, h, :] += gap[:, None] * kgain[None, :]
        y = pred + E[:, h, :]
        Y[:, h, :] = y
        z0, z1 = z1, y
    return Y, E


def _row_values(rows: list[_Row], Y: np.ndarray, E: np.ndarray, tf: Transforms | None = None) -> np.ndarray:
    tf = TRANSFORMS if tf is None else tf
    out = np.empty((Y.shape[0], len(rows)))
    for j, r in enumerate(rows):
        if r.kind == "y":
            out[:, j] = np.einsum("hn,mhn->m", r.coef, Y) + r.const
        elif r.kind == "e":
            out[:, j] = np.einsum("hn,mhn->m", r.coef, E)
        else:
            v = np.full(Y.shape[0], r.const)
            for h, k, c in r.exp_terms:
                v = v + c * tf.level(CORE[k], Y[:, h, k])
            out[:, j] = v
    return out


def _lin_G(model: FactorModel, sysm: dict, rows: list[_Row], H: int) -> np.ndarray:
    """(q, H, n): the linear rows as rows on the stacked innovations (y rows through the adjoint recursion)."""
    G = np.zeros((len(rows), H, model.n))
    yi = [j for j, r in enumerate(rows) if r.kind == "y"]
    if yi:
        G[yi] = FactorModel.adjoint(sysm, np.stack([rows[j].coef for j in yi]))
    for j, r in enumerate(rows):
        if r.kind == "e":
            G[j] = r.coef
    return G


def _solve_psd(S: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        cf = np.linalg.cholesky(S)
        return np.linalg.solve(cf.T, np.linalg.solve(cf, B)), False
    except np.linalg.LinAlgError:
        return np.linalg.pinv(S) @ B, True


def project(model: FactorModel, sysm: dict, V: np.ndarray, H: int, eq: list[_Row], T_eq: np.ndarray,
            Y: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Impose the equality rows exactly on every path: the Waggoner-Zha projection of the stacked innovations in the
    metric V (y* = y + M V G' (G V G')^-1 (r - R y)); unemployment rows by Newton steps inside the space the linear rows
    leave free. T_eq: (m, q) per-path targets."""
    lin = [j for j, r in enumerate(eq) if r.kind in ("y", "e")]
    non = [j for j, r in enumerate(eq) if r.kind == "u"]
    info: dict = {"rows": len(eq)}
    Y1, E1 = Y, E
    if lin:
        rl = [eq[j] for j in lin]
        G = _lin_G(model, sysm, rl, H)
        VG = np.einsum("hab,qhb->haq", V, G)
        S = np.einsum("qhn,hnp->qp", G, VG)
        S = 0.5 * (S + S.T)
        Wl = FactorModel.responses(sysm, VG)
        d = T_eq[:, lin] - _row_values(rl, Y, E, model.tf)
        alpha, sing = _solve_psd(S, d.T)
        info["singular"] = sing
        Y1 = Y + np.einsum("hnq,qm->mhn", Wl, alpha)
        E1 = E + np.einsum("hnq,qm->mhn", VG, alpha)
    if non:
        ru = [eq[j] for j in non]
        ents = sorted({(h, k) for r in ru for h, k, _ in r.exp_terms})
        pos = {hk: e for e, hk in enumerate(ents)}
        Rsel = np.zeros((len(ents), H, model.n))
        for (h, k), e in pos.items():
            Rsel[e, h, k] = 1.0
        GE = FactorModel.adjoint(sysm, Rsel)
        VGE = np.einsum("hab,qhb->haq", V, GE)
        WE = FactorModel.responses(sysm, VGE)
        SEE = np.einsum("qhn,hnp->qp", GE, VGE)
        if lin:
            SLE = np.einsum("qhn,hnp->qp", G, VGE)
            K, _ = _solve_psd(S, SLE)
            ECE = SEE - SLE.T @ K
            W1 = WE - np.einsum("hnq,qe->hne", Wl, K)
            VG1 = VGE - np.einsum("hnq,qe->hne", VG, K)
        else:
            ECE, W1, VG1 = SEE, WE, VGE
        ECE = 0.5 * (ECE + ECE.T)
        ell1 = np.stack([Y1[:, h, k] for h, k in ents], axis=1)
        Cm = np.zeros((len(ru), len(ents)))
        const = np.array([r.const for r in ru])
        for k, r in enumerate(ru):
            for h, kk, c in r.exp_terms:
                Cm[k, pos[(h, kk)]] += c
        tgt = T_eq[:, non]
        bnd = [model.tf.vars[CORE[k]] for _, k in ents]

        def lev(x):                                                    # entries' levels and dlevel / dx
            Lx, Dx = np.empty_like(x), np.empty_like(x)
            for e, b in enumerate(bnd):
                Lx[:, e], Dx[:, e] = b.level(x[:, e]), b.dlevel(x[:, e])
            return Lx, Dx
        ell = ell1.copy()
        coefE = np.zeros_like(ell1)                                    # ell - ell1 = ECE coefE
        Lv, Dv = lev(ell)
        res = tgt - (Lv @ Cm.T + const)
        e_path = np.max(np.abs(res), axis=1)
        err, it, damped = float(e_path.max()), 0, 0
        for it in range(300):
            err = float(e_path.max())
            if err < 1e-12:
                break
            A = Cm[None, :, :] * Dv[:, None, :]                        # (m, nN, nE) Jacobian at the iterate
            rhs = res + np.einsum("mke,me->mk", A, ell - ell1)
            M = np.einsum("mke,ef,mlf->mkl", A, ECE, A)
            beta = np.linalg.solve(M + 1e-14 * np.eye(len(ru))[None], rhs[..., None])[..., 0]
            c_new = np.einsum("mke,mk->me", A, beta)
            # P3's undamped iteration (it converges, linearly where the adjustment is large); a step is halved only
            # where it would leave the region in which a logistic level still moves (|x| > X_NEWTON_MAX) or turn
            # non-finite, so the 'log' arm is P3's iteration exactly
            t = np.ones(len(ell))
            for _ in range(60):
                c_try = coefE + t[:, None] * (c_new - coefE)
                x_try = ell1 + c_try @ ECE
                L_try, D_try = lev(x_try)
                r_try = tgt - (L_try @ Cm.T + const)
                e_try = np.max(np.abs(r_try), axis=1)
                wild = (~np.isfinite(e_try) | (np.max(np.abs(x_try), axis=1) > X_NEWTON_MAX)) & (t > 1e-12)
                if not wild.any():
                    break
                damped += int(wild.sum())
                t = np.where(wild, 0.5 * t, t)
            coefE, ell, Lv, Dv, res, e_path = c_try, x_try, L_try, D_try, r_try, e_try
        err = float(e_path.max())
        info.update(newton_iters=it + 1, newton_err=err, newton_damped=damped)
        Y1 = Y1 + np.einsum("hne,me->mhn", W1, coefE)
        E1 = E1 + np.einsum("hne,me->mhn", VG1, coefE)
    return Y1, E1, info


def _gibbs_tmvn(mean: np.ndarray, cov: np.ndarray, lo: np.ndarray, hi: np.ndarray, m: int,
                rng: np.random.Generator, sweeps: int = cq.GIBBS_SWEEPS) -> np.ndarray:
    """(m, q) draws of N(mean, cov) truncated to [lo, hi]: Gibbs sweeps over the coordinates, one chain per path."""
    q = len(mean)
    start = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi),
                     np.where(np.isfinite(lo), np.maximum(mean, lo), np.where(np.isfinite(hi), np.minimum(mean, hi),
                                                                               mean)))
    x = np.broadcast_to(start, (m, q)).copy()
    P = np.linalg.inv(cov + 1e-12 * np.eye(q))
    for _ in range(sweeps):
        for j in range(q):
            sj = 1.0 / P[j, j]
            others = [k for k in range(q) if k != j]
            mj = mean[j] - sj * ((x[:, others] - mean[others]) @ P[j, others])
            sd = math.sqrt(sj)
            za, zb = (lo[j] - mj) / sd, (hi[j] - mj) / sd
            up = za > 0                                    # an interval in the upper tail: sample its mirror image,
            a = np.where(up, ndtr(-zb), ndtr(za))          # where the normal CDF is precise
            b = np.where(up, ndtr(-za), ndtr(zb))
            u = np.clip(a + (b - a) * rng.random(m), 1e-300, 1 - 1e-16)
            z = ndtri(u)
            x[:, j] = np.clip(np.where(up, mj - sd * z, mj + sd * z), lo[j], hi[j])
    return x


def outputs(model: FactorModel, init: InitState, Y: np.ndarray, E: np.ndarray, draw_index: np.ndarray, H: int,
            usd_hold: bool) -> tuple[dict, dict, dict]:
    """x fields (card units), levels and structural policy shocks of simulated core + satellite paths (the exact
    mapping of §5.1: d_ff = Δ observed ff, d_y10 = Δ observed y10 (= Δ(ry10 + pi_trend) above Y10_FLOOR), d_ip = ip / 4,
    d_oil = Δro + pc / 4, ...)."""
    m = Y.shape[0]
    g = {v: Y[:, :, IDX[v]] for v in CORE}
    pi = np.empty((m, H))
    prev = np.full(m, init.pi0)
    for h in range(H):
        prev = prev + ALPHA * (g["pc"][:, h] - prev)
        pi[:, h] = prev
    shadow = g["rff"] + pi
    ff = np.maximum(shadow, cq.FF_FLOOR)
    y10_sh = g["ry10"] + pi                          # the shadow 10-year; observed y10 = max(shadow, Y10_FLOOR) (P4b)
    y10 = np.maximum(y10_sh, cq.Y10_FLOOR)
    tf = model.tf
    u = tf.level("lu", g["lu"])
    z1 = init.z[1]
    lvl = {v: tf.level(v, g[v]) for v in ("sp", "gz")}
    lvl0 = {v: float(tf.level(v, z1[IDX[v]])) for v in ("sp", "gz")}
    if usd_hold:                                     # the cross-path mean over finite paths only: one non-finite path
        fin = np.where(np.isfinite(g["usd"]), g["usd"], np.nan)    # would otherwise make every path's dollar NaN
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(fin, axis=0, keepdims=True) if np.isfinite(fin).any() else np.zeros((1, g["usd"].shape[1]))
        usd = g["usd"] - (np.nan_to_num(mu, nan=z1[IDX["usd"]]) - z1[IDX["usd"]])
    else:
        usd = g["usd"]

    def lagged(a, first):
        return np.concatenate([np.full((m, 1), first), a[:, :-1]], axis=1)

    x = {"d_cpi": g["pc"], "d_cpi_serv": g["ps"], "d_cpi_goods": (g["pc"] - model.ws * g["ps"]) / model.wg,
         "d_ip": g["ip"] / 4.0, "d_unrate": u - lagged(u, init.u0), "d_ff": ff - lagged(ff, init.ff0),
         "d_y10": y10 - lagged(y10, max(float(z1[IDX["ry10"]] + init.pi0), cq.Y10_FLOOR)),
         "d_spread": lvl["sp"] - lagged(lvl["sp"], lvl0["sp"]), "d_gz": lvl["gz"] - lagged(lvl["gz"], lvl0["gz"]),
         "d_usd": usd - lagged(usd, z1[IDX["usd"]]),
         "d_oil": g["ro"] - lagged(g["ro"], z1[IDX["ro"]]) + g["pc"] / 4.0}
    for j, f in enumerate(model.sats.fields):
        x[f] = Y[:, :, NC + j]
    if "x_oil_cause" in x:
        x["x_oil_cause"] = np.clip(x["x_oil_cause"], 0.0, 1.0)
    pch = np.concatenate([np.broadcast_to(init.pc_hist, (m, 3)), g["pc"]], axis=1) / 400.0
    cs = np.concatenate([np.zeros((m, 1)), np.cumsum(pch, axis=1)], axis=1)
    lv = {"ff": ff, "ff_shadow": shadow, "y10": y10, "y10_shadow": y10_sh, "unrate": u, "spread": lvl["sp"],
          "gz": lvl["gz"], "usd": usd,
          "real_oil": g["ro"], "rff": g["rff"], "ry10": g["ry10"], "pi_trend": pi,
          "cpi": init.lvl_cpi0 + np.cumsum(g["pc"] / 4.0, axis=1), "cpi_yoy": 100.0 * (np.exp(cs[:, 4:] - cs[:, :-4]) - 1),
          "zlb": ff < cq.ZLB_FF}
    if "d_core" in x and np.all(np.isfinite(init.core_hist)):
        ch = np.concatenate([np.broadcast_to(init.core_hist, (m, 3)), x["d_core"]], axis=1) / 400.0
        cc = np.concatenate([np.zeros((m, 1)), np.cumsum(ch, axis=1)], axis=1)
        lv["core_yoy"] = 100.0 * (np.exp(cc[:, 4:] - cc[:, :-4]) - 1.0)
        lv["core4"] = lv["core_yoy"] > cq.CORE4
    shocks = {u_: np.zeros((m, H)) for u_ in SHOCKS}
    for d in np.unique(draw_index):
        rows = np.flatnonzero(draw_index == d)
        P = model.system(None if d < 0 else int(d))["P"]
        eps = np.linalg.solve(P, E[rows].reshape(-1, model.n).T).T.reshape(len(rows), H, model.n)
        for u_, var in cq.POLICY_DO.items():
            if var in model.names:
                k = model.names.index(var)
                shocks[u_][rows] = P[k, k] * eps[:, :, k]
    lv["u_ff_at_floor"] = (lagged(ff, init.ff0) < cq.ZLB_FF) & (ff < cq.ZLB_FF)    # the observed rate cannot move
    return x, lv, shocks


def evaluate(st: Statement, x: dict, levels: dict, shocks: dict) -> np.ndarray:
    """The statement's observable on simulated output: (n_paths,) for one quarter or an aggregate."""
    qs = [int(q) - 1 for q in st.quarters]
    src = shocks.get(st.var)
    if src is None:
        src = x.get("d_cpi") if st.var == "pc" else x.get(st.var, levels.get(st.var))
    if src is None:
        raise FactorError(f"字段不存在：{st.var}")
    vals = np.asarray(src, float)[:, qs]
    if st.agg == "sum":
        return vals.sum(axis=1)
    if st.agg == "mean":
        return vals.mean(axis=1)
    return vals[:, -1]


def saturation(model: FactorModel, init: InitState, H: int, stmts: list[Statement], seed: int = 0) -> tuple[list,
                                                                                                          list]:
    """§3.5 as written: the gap between each stated value and its conditional mean saturates at +- SATURATION_SD
    training sd (fixed a priori). The conditional mean is the statement's observable on the model's zero-innovation
    path from the state (posterior mean, no cycle module: analytic and deterministic, the same in step mode); the
    training sd is statement_sd (the field's sd over the training quarters, field_stats at the train end; sums, means
    and levels h quarters ahead by their own). d_oil and d_y10 (the mechanical own-price inputs) pass through.
    Returns (values used, notes). `seed` is not used (the rule has no Monte Carlo)."""
    del seed
    vals = [st.value for st in stmts]
    todo = [i for i, st in enumerate(stmts) if st.value is not None and st.var not in cq.MECHANICAL_FIELDS]
    if not todo:
        return vals, []
    Hp = max(max(int(q) for q in stmts[i].quarters) for i in todo)
    mp = model.mean_path(init, Hp, model.system())[None]
    x, lv, sh = outputs(model, init, mp, np.zeros_like(mp), np.full(1, -1), Hp, usd_hold=False)
    notes = []
    for i in todo:
        st = stmts[i]
        mu = float(evaluate(st, x, lv, sh)[0])
        sd = statement_sd(model, st)
        if not (np.isfinite(mu) and np.isfinite(sd) and sd > 0):
            continue
        lo, hi = mu - cq.SATURATION_SD * sd, mu + cq.SATURATION_SD * sd
        if not lo <= st.value <= hi:
            vals[i] = float(np.clip(st.value, lo, hi))
            notes.append({"statement": i, "var": st.var, "quarters": list(st.quarters), "stated": st.value,
                          "used": vals[i], "cond_mean": mu, "train_sd": sd, "rule": "±4 training sd (§3.5)",
                          "text": f"{st.var}：{CLIP_TEXT}"})
    return vals, notes


def _targets(stmts: list[Statement], vals: list, soft: dict, idx: np.ndarray) -> list:
    out = []
    for i, st in enumerate(stmts):
        if vals[i] is None:
            out.append(None)
        elif i in soft:
            out.append(vals[i] + st.sd * soft[i][idx])
        else:
            out.append(np.full(len(idx), float(vals[i])))
    return out


def _regime_path(model: FactorModel, init: InitState, H: int) -> list[str]:
    """The regime of each quarter for the regime covariance: the start state for quarter 1, then the posterior-mean
    model's zero-shock path (shared by every path)."""
    mp = model.mean_path(init, H, model.system())
    pi, ff_prev, out = init.pi0, init.ff0, []
    for h in range(H):
        out.append("zlb" if ff_prev < cq.ZLB_FF else "normal")
        pi = pi + ALPHA * (mp[h, IDX["pc"]] - pi)
        ff_prev = max(mp[h, IDX["rff"]] + pi, cq.FF_FLOOR)
    return out


def _batch(model: FactorModel, init: InitState, H: int, n: int, ss: np.random.SeedSequence, stmts, vals, soft_all,
           path_ids: np.ndarray, flags: dict, w: float, rc: bool, cycle: bool, cycle_odds, gaussian: bool,
           param_uncertainty: bool, block: int, extra_rows: list | None = None, extra_T: np.ndarray | None = None):
    """One batch of n paths: draws, the recursion with the cycle module, the projection. Returns Y, E, labels,
    draw index, the range rows (as compiled for the first group) and projection info."""
    r_shock, r_cb, r_cn = [np.random.default_rng(c) for c in ss.spawn(3)]
    draws = list(range(len(model.draws))) if (param_uncertainty and model.draws) else [None]
    U, lab = _draw_whitened(model, r_shock, n, H, gaussian, block)
    active = (~flags["growth"]) if cycle else np.zeros(H, bool)
    cyc = cycle_targets(model, r_cb, r_cn, n, H, cycle_odds) if active.any() else None
    regimes = _regime_path(model, init, H) if rc else None
    per = int(math.ceil(n / len(draws)))
    Y = np.empty((n, H, model.n))
    E = np.empty_like(Y)
    dix = np.full(n, -1)
    ranges, pinfo = None, {}
    for j, d in enumerate(draws):
        idx = np.arange(j * per, min((j + 1) * per, n))
        if len(idx) == 0:
            break
        sysm = model.system(d)
        V = _metric(model, sysm["Sigma"], H, flags, w)
        Lq = None
        if regimes is not None:
            Lq = np.empty((H, model.n, model.n))
            for h in range(H):
                sr = model.system(d, regime=regimes[h])
                Lq[h] = sr["L"]
                V[h] = _metric(model, sr["Sigma"], 1, {k: (v[h:h + 1] if isinstance(v, np.ndarray) else v)
                                                       for k, v in flags.items()}, w)[0]
        rows = compile_statements(model, init, H, stmts, _targets(stmts, vals, soft_all, path_ids[idx]), sysm) \
            if stmts else []
        cyc_g = None if cyc is None else cyc[idx] + model.cycle_shift(d)     # the draw's long-run IP growth (P4b)
        Yg, Eg = _run_paths(model, init, sysm, U[idx], H, cyc_g, active, Lq)
        eq = [r for r in rows if r.target is not None]
        T_eq = np.column_stack([np.broadcast_to(np.asarray(r.target, float), (len(idx),)) for r in eq]) if eq else \
            np.zeros((len(idx), 0))
        if extra_rows:
            eq = eq + extra_rows
            T_eq = np.hstack([T_eq, extra_T[idx]])
        if eq:
            Y0, E0 = Yg, Eg
            Yg, Eg, gi = project(model, sysm, V, H, eq, T_eq, Y0, E0)
            if any(r.ffchange for r in eq):
                Yg, Eg, gi["ff_floor_repair"] = _ff_repair(model, init, sysm, V, H, eq, T_eq, Y0, E0, Yg, Eg)
            # over the posterior draws' groups: the worst Newton error and iterations, the damped steps (P4b)
            for k_, f_ in (("newton_err", max), ("newton_iters", max), ("newton_damped", lambda a, b: a + b)):
                if k_ in gi and k_ in pinfo:
                    gi[k_] = f_(pinfo[k_], gi[k_])
            pinfo = gi
        Y[idx], E[idx], dix[idx] = Yg, Eg, -1 if d is None else d
        if ranges is None:
            ranges = [r for r in rows if r.target is None]
    return Y, E, lab, dix, ranges or [], pinfo


def _shadow_nominal(init: InitState, Y: np.ndarray) -> np.ndarray:
    """(m, H + 1) shadow nominal policy rate, column 0 = the start (the observed rate unless it sits at the floor)."""
    m, H = Y.shape[0], Y.shape[1]
    pi = np.full(m, init.pi0)
    out = np.empty((m, H + 1))
    out[:, 0] = init.ff0 if init.ff0 > cq.FF_FLOOR + 1e-9 else init.z[1][IDX["rff"]] + init.pi0
    for h in range(H):
        pi = pi + ALPHA * (Y[:, h, IDX["pc"]] - pi)
        out[:, h + 1] = Y[:, h, IDX["rff"]] + pi
    return out


def _ff_repair(model, init, sysm, V, H, eq, T_eq, Y0, E0, Y1, E1, rounds: int = 6):
    """A stated policy-rate change whose previous quarter is not stated is imposed on the shadow rate (observed =
    shadow above the floor). On a path that starts the quarter at the floor a hike means observed = floor + hike, so
    that path's row becomes the level row shadow_h = FF_FLOOR + v; if the re-projected path then starts the quarter
    above the floor, the statement holds only at the kink (shadow_{h-1} = FF_FLOOR and shadow_h = FF_FLOOR + v,
    two rows). Paths are re-projected from their unconditional draws (an active set, a few rounds). A cut from the
    floor cannot be delivered (reported)."""
    fr = [j for j, r in enumerate(eq) if r.ffchange]
    cmp_ = _Compiler(model, init, H)
    Y, E = Y1.copy(), E1.copy()
    form = np.zeros((len(Y), len(fr)), int)                # 0 change, 1 level, 2 kink
    infeasible = 0
    for _ in range(rounds):
        sh = _shadow_nominal(init, Y)
        new = form.copy()
        infeasible = 0
        for k, j in enumerate(fr):
            h = eq[j].quarter
            obs = np.maximum(sh[:, h + 1], cq.FF_FLOOR) - np.maximum(sh[:, h], cq.FF_FLOOR)
            miss = np.abs(obs - T_eq[:, j]) > 1e-9
            at_floor = sh[:, h] < cq.FF_FLOOR
            hike = T_eq[:, j] > 0
            new[:, k] = np.where(miss & at_floor & hike & (form[:, k] == 0), 1, new[:, k])
            new[:, k] = np.where(miss & ~at_floor & (form[:, k] == 1), 2, new[:, k])
            infeasible += int((miss & at_floor & ~hike).sum())
        if (new == form).all():
            break
        form = new
        for pat in {tuple(r) for r in form[(form > 0).any(axis=1)]}:
            rows_b = np.flatnonzero((form == np.array(pat)).all(axis=1))
            eq2 = list(eq)
            T2 = T_eq[rows_b].copy()
            extra, extra_t = [], []
            for k, j in enumerate(fr):
                if not pat[k]:
                    continue
                h = eq[j].quarter
                ex = cmp_.shadow(h)
                eq2[j] = _Row(kind="y", stmt=eq[j].stmt, coef=ex.coef, const=ex.const, target=0.0, quarter=h)
                T2[:, j] = cq.FF_FLOOR + T_eq[rows_b, j]
                if pat[k] == 2 and h >= 1:
                    ex0 = cmp_.shadow(h - 1)
                    extra.append(_Row(kind="y", stmt=eq[j].stmt, coef=ex0.coef, const=ex0.const, target=0.0,
                                      quarter=h - 1))
                    extra_t.append(np.full(len(rows_b), cq.FF_FLOOR))
            if extra:
                eq2 = eq2 + extra
                T2 = np.hstack([T2, np.column_stack(extra_t)])
            Yb, Eb, _ = project(model, sysm, V, H, eq2, T2, Y0[rows_b], E0[rows_b])
            Y[rows_b], E[rows_b] = Yb, Eb
    return Y, E, {"paths_level": int((form == 1).any(axis=1).sum()), "paths_kink": int((form == 2).any(axis=1).sum()),
                  "cut_at_floor_infeasible": infeasible}


def simulate(model: FactorModel, init: InitState, H: int, n_paths: int = 2000, seed: int = 0, statements=(),
             cycle: bool = True, cycle_odds: dict | None = None, gaussian: bool = False,
             param_uncertainty: bool = True, block: int = 1, completion_w: float | None = None,
             regime_cov: bool | None = None, usd_hold: bool | None = None, saturate: bool = True,
             _internal: bool = False) -> SimResult:
    """Path mode: H quarters from `init`, every statement imposed at once (two-sided); step mode is H = 1.

    statements: Statement objects (quarters 1..H). cycle: the C' module in quarters that state nothing about growth;
    cycle_odds = {"contraction": share, "quarters": [...]} tilts its blocks. gaussian: N(0, I) whitened innovations
    instead of the bootstrap. param_uncertainty: paths spread over the posterior draws (else the posterior mean).
    block: bootstrap block length (CYCLE_ALT_BLOCK with cycle=False is the pre-registered alternative to C').
    completion_w / regime_cov / usd_hold default to config_q. The same seed gives the same paths."""
    stmts = expand(statements)
    w = cq.COMPLETION_W if completion_w is None else float(completion_w)
    rc = cq.REGIME_COV if regime_cov is None else bool(regime_cov)
    flags = _quarter_flags(stmts, H)
    hold = (cq.USD_HOLD if usd_hold is None else bool(usd_hold)) and not flags["usd"]
    ss_main, ss_soft, ss_acc, ss_gibbs = np.random.SeedSequence(seed).spawn(4)
    notes: list = []
    vals = [st.value for st in stmts]
    if stmts and saturate:
        vals, clips = saturation(model, init, H, stmts, seed)
        notes += clips
    r_soft = np.random.default_rng(ss_soft)
    soft = {i: r_soft.standard_normal(n_paths) for i, st in enumerate(stmts) if st.value is not None and st.sd}
    pid = np.arange(n_paths)
    args = dict(flags=flags, w=w, rc=rc, cycle=cycle, cycle_odds=cycle_odds, gaussian=gaussian,
                param_uncertainty=param_uncertainty, block=block)
    Y, E, lab, dix, ranges, pinfo = _batch(model, init, H, n_paths, ss_main, stmts, vals, soft, pid, **args)
    info = {"n_paths": n_paths, "H": H, "seed": seed, "draws": int(len(np.unique(dix))),
            "cycle_quarters": int(((~flags["growth"]) if cycle else np.zeros(H, bool)).sum()), "completion_w": w,
            "regime_cov": rc, "usd_hold": hold, "gaussian": gaussian, "block": block,
            "projection": {k: v for k, v in pinfo.items() if not isinstance(v, np.ndarray)}}
    if ranges:
        Y, E, lab, dix, rinfo = _ranges(model, init, H, n_paths, stmts, vals, soft, ranges, Y, E, lab, dix, ss_acc,
                                        ss_gibbs, args)
        info["ranges"] = rinfo
    x, lv, sh = outputs(model, init, Y, E, dix, H, hold)
    res = SimResult(x=x, levels=lv, shocks=sh, core=Y[:, :, :NC], sats=Y[:, :, NC:], innov=E, drawn=lab,
                    draw_index=dix, statements=stmts, delivery=[], rarity={}, notes=notes, info=info)
    if not _internal and stmts:
        res.delivery = delivery(stmts, vals, res)
        res.rarity = rarity(model, init, H, stmts, vals)
        notes += [{"text": t} for t in res.rarity.get("warnings", [])]
    return res


def _ranges(model, init, H, n_paths, stmts, vals, soft, ranges, Y, E, lab, dix, ss_acc, ss_gibbs, args):
    """Ranges: accept/reject while at least TRUNC_MIN_ACCEPT of the paths pass (fresh batches until n_paths are
    accepted); otherwise a Gibbs draw of the restricted coordinates from their Gaussian conditional given the
    equality statements, imposed exactly on every path."""
    lo = np.array([-np.inf if r.lo is None else float(np.max(r.lo)) for r in ranges])
    hi = np.array([np.inf if r.hi is None else float(np.min(r.hi)) for r in ranges])

    def ok(Yb, Eb):
        v = _row_values(ranges, Yb, Eb, model.tf)
        return np.all((v >= lo - 1e-12) & (v <= hi + 1e-12), axis=1)

    keep = ok(Y, E)
    rate = float(keep.mean())
    info = {"acceptance": rate}
    if rate >= cq.TRUNC_MIN_ACCEPT:
        info["method"] = "accept_reject"
        parts = [(Y[keep], E[keep], lab[keep], dix[keep])]
        got, b = int(keep.sum()), 0
        kids = ss_acc.spawn(cq.TRUNC_MAX_BATCHES)
        while got < n_paths and b < cq.TRUNC_MAX_BATCHES:
            Yb, Eb, lb, db, _, _ = _batch(model, init, H, n_paths, kids[b], stmts, vals, soft, np.arange(n_paths),
                                          **args)
            kb = ok(Yb, Eb)
            parts.append((Yb[kb], Eb[kb], lb[kb], db[kb]))
            got += int(kb.sum())
            b += 1
        out = [np.concatenate([p[k] for p in parts])[:n_paths] for k in range(4)]
        info["batches"] = b
        if len(out[0]) < n_paths:
            info["short"] = int(n_paths - len(out[0]))
            reps = np.resize(np.arange(len(out[0])), n_paths)
            out = [o[reps] for o in out]
        return out[0], out[1], out[2], out[3], info
    info["method"] = "gibbs"
    if any(r.kind == "u" for r in ranges):
        raise FactorError("接受率低于 10% 的失业率非线性范围约束无法抽样，请改写为水平范围")
    sysm = model.system()
    V = _metric(model, sysm["Sigma"], H, args["flags"], args["w"])
    mu = model.mean_path(init, H, sysm)[None]
    rows0 = compile_statements(model, init, H, stmts, _targets(stmts, vals, {}, np.arange(1)), sysm)
    eq0 = [r for r in rows0 if r.target is not None]
    Gr = _lin_G(model, sysm, ranges, H)
    VGr = np.einsum("hab,qhb->haq", V, Gr)
    Srr = np.einsum("qhn,hnp->qp", Gr, VGr)
    mu_c = mu
    if eq0:
        T0 = np.column_stack([np.ravel(r.target)[:1] for r in eq0])
        mu_c, _, _ = project(model, sysm, V, H, eq0, T0, mu, np.zeros_like(mu))
        lin0 = [r for r in eq0 if r.kind != "u"]
        if lin0:
            Ge = _lin_G(model, sysm, lin0, H)
            VGe = np.einsum("hab,qhb->haq", V, Ge)
            See = np.einsum("qhn,hnp->qp", Ge, VGe)
            Sre = np.einsum("qhn,hnp->qp", Gr, VGe)
            Srr = Srr - Sre @ np.linalg.lstsq(See, Sre.T, rcond=None)[0]
    mean_r = _row_values(ranges, mu_c, np.zeros_like(mu_c), model.tf)[0]
    Tg = _gibbs_tmvn(mean_r, 0.5 * (Srr + Srr.T), lo, hi, n_paths, np.random.default_rng(ss_gibbs))
    eq_r = [_Row(kind=r.kind, stmt=r.stmt, coef=r.coef, const=r.const, exp_terms=r.exp_terms, target=0.0,
                 quarter=r.quarter) for r in ranges]
    Y2, E2, lab2, d2, _, _ = _batch(model, init, H, n_paths, np.random.SeedSequence(int(ss_acc.entropy) % (2 ** 32)),
                                    stmts, vals, soft, np.arange(n_paths), extra_rows=eq_r, extra_T=Tg, **args)
    info["acceptance_after"] = float(ok(Y2, E2).mean())
    return Y2, E2, lab2, d2, info


def delivery(stmts: list[Statement], vals: list, res: SimResult) -> list:
    """Per statement: hard -> max |delivered - used| over paths; soft -> the share within +-1 sd of the value used
    (68% when met within its sd); range -> the share inside."""
    out = []
    for i, st in enumerate(stmts):
        v = evaluate(st, res.x, res.levels, res.shocks)
        d = {"statement": i, "var": st.var, "quarters": list(st.quarters), "agg": st.agg, "label": st.label}
        if st.is_range:
            lo = -np.inf if st.lo is None else st.lo
            hi = np.inf if st.hi is None else st.hi
            d.update(kind="range", lo=st.lo, hi=st.hi,
                     share_inside=float(np.mean((v >= lo - 1e-9) & (v <= hi + 1e-9))))
        elif st.sd:
            d.update(kind="soft", value=st.value, used=vals[i], sd=st.sd, mean=float(np.mean(v)),
                     share_within_1sd=float(np.mean(np.abs(v - vals[i]) <= st.sd)))
        else:
            d.update(kind="hard", value=st.value, used=vals[i], max_abs_error=float(np.max(np.abs(v - vals[i]))))
        out.append(d)
    return out


def empirical_percentile(ref: np.ndarray, x) -> np.ndarray:
    """Mid-rank percentile of x among the reference values (ties count half)."""
    r = np.sort(np.asarray(ref, float))
    x = np.asarray(x, float)
    if not len(r):
        return np.full(x.shape, np.nan)
    return (np.searchsorted(r, x, "left") + np.searchsorted(r, x, "right")) / (2.0 * len(r))


def pool_innovations(model: FactorModel, H: int, sysm: dict | None = None, draws: int = RARITY_DRAWS,
                     seed: int = RARITY_SEED) -> np.ndarray:
    """(K, H, n) reduced-form innovations of the simulator's own bootstrap at the posterior mean: every pool row for
    H = 1, else `draws` stacked draws (independent rows per quarter, fixed seed): the rarity statistic's reference."""
    L = (sysm or model.system())["L"]
    if H == 1:
        return (model.pool @ L.T)[:, None, :]
    rng = np.random.default_rng([seed, H])
    idx = rng.integers(0, len(model.pool), size=(draws, H))
    return model.pool[idx] @ L.T


def rarity(model: FactorModel, init: InitState, H: int, stmts: list[Statement], vals: list | None = None) -> dict:
    """How unusual a statement set is (§3.5): m2 = d' S^-1 d for the stated values' gaps d to the model's mean path
    (S = G V G' + the soft variances, V the posterior-mean Sigma, linear rows) and, per quarter, the Mahalanobis norm
    of the mean innovation adjustment. The percentiles are those of the same statistics under the simulator's own
    innovations (pool_innovations: each bootstrap draw read as if its own values had been stated), not a chi-square:
    the chi-square assumes Gaussian innovations with covariance Sigma, while the pool's covariance is 1.2-3.9 times
    Sigma on the core diagonal and heavy-tailed, so it read 13% of the model's own draws as rarer than its 99th
    percentile. Above RARITY_WARN_PCT: "这个组合在历史上极少见". sd_per_constraint = sqrt(m2 / q) (the joint study's
    stagflation set needed 0.81)."""
    vals = [st.value for st in stmts] if vals is None else vals
    keep = [i for i, st in enumerate(stmts) if vals[i] is not None]
    if not keep:
        return {}
    sysm = model.system()
    sub = [stmts[i] for i in keep]
    rows = compile_statements(model, init, H, sub, [np.array([float(vals[i])]) for i in keep], sysm)
    rows = [r for r in rows if r.target is not None and r.kind in ("y", "e")]
    if not rows:
        return {}
    G = _lin_G(model, sysm, rows, H)
    VG = np.einsum("ab,qhb->haq", sysm["Sigma"], G)
    S = np.einsum("qhn,hnp->qp", G, VG)
    mu = model.mean_path(init, H, sysm)[None]
    d = np.array([float(np.ravel(r.target)[0]) for r in rows]) - _row_values(rows, mu, np.zeros_like(mu), model.tf)[0]
    sd2 = np.zeros(len(rows))
    for k, r in enumerate(rows):
        st = sub[r.stmt]
        if st.sd:
            scale = 1.0
            if st.var in model.tf.obs and r.kind == "y":
                v = model.tf.obs[st.var][0]
                scale = 1.0 / max(float(model.tf.dlevel(v, mu[0, r.quarter, IDX[v]])), 1e-3)
            sd2[k] = (st.sd * scale) ** 2
    q = len(rows)
    Ssoft = S + np.diag(sd2)
    m2 = float(d @ np.linalg.lstsq(Ssoft, d, rcond=None)[0])
    S_reg = S + 1e-12 * np.eye(q)
    dE = np.einsum("haq,q->ha", VG, np.linalg.lstsq(S_reg, d, rcond=None)[0])
    Si = np.linalg.inv(sysm["Sigma"])
    nq = np.zeros(H, int)
    for r in rows:
        nq[r.quarter] += 1
    # the reference: the same statistics for the simulator's own innovations
    Eu = pool_innovations(model, H, sysm)
    du = np.einsum("qhn,khn->kq", G, Eu)
    ref_m2 = np.einsum("kq,kq->k", du, np.linalg.lstsq(Ssoft, du.T, rcond=None)[0].T)
    au = np.linalg.lstsq(S_reg, du.T, rcond=None)[0]                     # (q, K)
    per_q, warnings = [], []
    for h in range(H):
        mh = float(dE[h] @ Si @ dE[h])
        if nq[h] or mh > 1e-9:
            dEu = VG[h] @ au                                              # (n, K)
            ref_h = np.einsum("nk,nk->k", dEu, Si @ dEu)
            pct = float(empirical_percentile(ref_h, mh))
            per_q.append({"q": h + 1, "m2": mh, "df": int(nq[h]), "percentile": pct, "sd": math.sqrt(mh),
                          "chi2_percentile": float(chi2.cdf(mh, max(int(nq[h]), 1)))})
            if pct > cq.RARITY_WARN_PCT and nq[h]:
                warnings.append(f"第 {h + 1} 季：这个组合在历史上极少见（需要约 {math.sqrt(mh):.1f} 个标准差的冲击）")
    pct = float(empirical_percentile(ref_m2, m2))
    if pct > cq.RARITY_WARN_PCT and not warnings:
        warnings.append(f"这个组合在历史上极少见（需要约 {math.sqrt(m2):.1f} 个标准差的冲击）")
    return {"m2": m2, "df": q, "percentile": pct, "chi2_percentile": float(chi2.cdf(m2, q)),
            "reference": {"kind": "pool_rows" if H == 1 else "pool_draws", "n": int(len(ref_m2))},
            "sd_per_constraint": math.sqrt(m2 / q), "per_quarter": per_q, "warnings": warnings}


def step(model: FactorModel, init: InitState, n_paths: int = 500, seed: int = 0, statements=(), **kw) -> SimResult:
    """Step mode (§3.7): one quarter's conditional draw given the state and the stated values (one-sided)."""
    return simulate(model, init, 1, n_paths=n_paths, seed=seed, statements=statements, **kw)


def conditional_moments(model: FactorModel, init: InitState, H: int, statements=(), completion_w: float = 1.0,
                        draw: int | None = None, regime: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Gaussian conditional mean (H, n) and covariance (H n, H n) of the stacked path given hard linear
    statements at fixed parameters without the cycle module: mu + W (r - R mu) and Cov - W S W'. regime: the regime
    covariance of every quarter (the T-C candidate; step mode knows the start state's regime)."""
    stmts = expand(statements)
    sysm = model.system(draw, regime=regime)
    V = _metric(model, sysm["Sigma"], H, _quarter_flags(stmts, H), completion_w)
    mu = model.mean_path(init, H, sysm)
    n = model.n
    eye = np.zeros((H, n, H * n))
    for h in range(H):
        eye[h, :, h * n:(h + 1) * n] = np.eye(n)
    M = FactorModel.responses(sysm, eye).reshape(H * n, H * n)
    Vb = np.zeros((H * n, H * n))
    for h in range(H):
        Vb[h * n:(h + 1) * n, h * n:(h + 1) * n] = V[h]
    Cov = M @ Vb @ M.T
    if not stmts:
        return mu, 0.5 * (Cov + Cov.T)
    rows = [r for r in compile_statements(model, init, H, stmts, [None if st.value is None else
                                                                  np.array([float(st.value)]) for st in stmts], sysm)
            if r.target is not None]
    if any(r.kind == "u" for r in rows):
        raise FactorError("解析条件矩只支持线性陈述")
    G = _lin_G(model, sysm, rows, H).reshape(len(rows), -1)
    off = np.array([r.const if r.kind == "y" else 0.0 for r in rows])
    Rm = np.array([r.coef.reshape(-1) if r.kind == "y" else np.zeros(H * n) for r in rows])
    t = np.array([float(np.ravel(r.target)[0]) for r in rows])
    v0 = np.array([float(Rm[k] @ mu.reshape(-1) + off[k]) if rows[k].kind == "y" else 0.0 for k in range(len(rows))])
    VG = Vb @ G.T
    S = G @ VG
    Wm = M @ VG @ np.linalg.inv(S)
    mean = mu.reshape(-1) + Wm @ (t - v0)
    cov = Cov - Wm @ S @ Wm.T
    return mean.reshape(H, n), 0.5 * (cov + cov.T)
