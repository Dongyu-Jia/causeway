"""Fitting the estimated recipes (instr-mapped's protocol): the refit of estimated.json and the out-of-sample check.

A recipe is a constrained style regression of a ticker's quarterly log total return (less pi x the cash return) on
the core's excess returns (style weights >= 0; funds sum to one, companies are capped) and factor paths, with the
intercept shrunk toward a prior (normal prior, sd 0.5% a quarter): the fund's fee (XLG, SCHD, BME), 0 (MO, BRK.B)
or the peer group's precision-weighted intercept (BDCs, MLPs). CSWC and MPLX use their group's pooled loadings.

Out-of-sample (no look-ahead: every fit uses quarters <= the origin): "expanding" refits at every origin from the
16th quarter and predicts the next h = 1, 4, 8 quarters' cumulative log total return from the REALISED core path
(the oracle a perfectly simulated core would give); "split" fits once on quarters <= 2019-12-31 and scores
2020Q1..2026Q2. R2 = 1 - SSE / SSE(bench), bench = the realised cash + h x the training mean excess return.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from . import data as D
from . import fees

TAU_ALPHA = 0.005
GROUP = {"MAIN": "BDC", "ARCC": "BDC", "CSWC": "BDC", "EPD": "MLP", "MPLX": "MLP"}
PEERS = {"BDC": ["MAIN", "ARCC", "CSWC"], "MLP": ["EPD", "MPLX"]}
POOLED = {"CSWC", "MPLX"}
PI = {"MAIN": 0.5, "ARCC": 0.5, "CSWC": 0.5}
EST_START = {"XLG": "2013-09-30", "MAIN": "2013-09-30", "ARCC": "2013-06-30", "CSWC": "2016-06-30",
             "MPLX": "2016-03-31", "EPD": "2013-06-30", "SCHD": "2013-09-30", "MO": "2013-06-30",
             "BME": "2013-09-30", "BRK.B": "2013-06-30"}
FEE_PRIOR = ("XLG", "SCHD", "BME")            # funds: alpha prior = -fee
ASSETS = ["us_equity", "nasdaq", "energy", "industrials", "staples", "healthcare", "utilities", "financials",
          "real_estate", "gold", "crude_oil", "treasury_10y", "cash"]
EQ_ALL = ["us_equity", "nasdaq", "energy", "industrials", "staples", "healthcare", "utilities", "financials",
          "real_estate"]
H = (1, 4, 8)
MIN_TRAIN = 16
SPLIT_END = pd.Timestamp("2019-12-31")
# the variant each final recipe uses (alpha prior, loadings, cash pass-through), instr-mapped s21
CHOSEN = {"XLG": ("zero", "own", "excess"), "SCHD": ("zero", "own", "excess"), "MO": ("zero", "own", "excess"),
          "BRK.B": ("zero", "own", "excess"), "BME": ("zero", "own", "excess"),
          "MAIN": ("group", "own", "half"), "ARCC": ("group", "own", "half"), "CSWC": ("group", "pooled", "half"),
          "EPD": ("group", "own", "excess"), "MPLX": ("group", "pooled", "excess")}


@dataclass
class Spec:
    style: list
    factors: list = field(default_factory=list)
    sum_to_one: bool = False
    max_sum: float | None = None
    free_style: bool = False
    alpha_prior: float = 0.0
    tau_alpha: float = TAU_ALPHA


def primary(t: str) -> Spec:
    a0 = -fees.fee(t) / 4.0 if t in FEE_PRIOR else 0.0
    if t == "XLG":
        return Spec(["us_equity", "nasdaq"], sum_to_one=True, alpha_prior=a0)
    if t == "SCHD":
        return Spec(EQ_ALL, sum_to_one=True, alpha_prior=a0)
    if t == "MO":
        return Spec(["staples", "utilities", "healthcare", "us_equity", "energy", "treasury_10y"], max_sum=1.5)
    if t == "BRK.B":
        return Spec(["us_equity", "financials", "industrials", "energy", "utilities", "staples", "nasdaq",
                     "treasury_10y"], max_sum=1.2)
    if t in ("MAIN", "ARCC", "CSWC"):
        return Spec(["financials", "us_equity"], factors=["d_gz"], max_sum=2.0)
    if t in ("EPD", "MPLX"):
        return Spec(["energy", "utilities", "us_equity"], factors=["d_gz"], max_sum=2.0)
    if t == "BME":
        return Spec(["healthcare", "us_equity"], max_sum=1.2, alpha_prior=a0)
    raise KeyError(t)


def market_spec() -> Spec:
    """The benchmark: a CAPM-type regression on the US market with a raw intercept."""
    return Spec(["us_equity"], free_style=True, tau_alpha=1.0)


def fit(y: np.ndarray, X: np.ndarray, spec: Spec) -> dict:
    """Constrained least squares (BVLS; sum-to-one / max-sum as a heavily weighted pseudo-observation) and the
    intercept shrunk toward spec.alpha_prior with prior sd spec.tau_alpha (the loadings are kept)."""
    n, k = X.shape
    ns = len(spec.style)
    A = np.column_stack([np.ones(n), X])
    lb, ub = np.full(k + 1, -np.inf), np.full(k + 1, np.inf)
    if not spec.free_style:
        lb[1:1 + ns] = 0.0
    yy = y.copy()
    big = 1e3
    if spec.sum_to_one and ns:
        row = np.zeros(k + 1)
        row[1:1 + ns] = big
        A = np.vstack([A, row])
        yy = np.append(yy, big)
    coef = lsq_linear(A, yy, bounds=(lb, ub), method="bvls").x
    if spec.max_sum is not None and ns and coef[1:1 + ns].sum() > spec.max_sum:
        row = np.zeros(k + 1)
        row[1:1 + ns] = big
        A2 = np.vstack([np.column_stack([np.ones(n), X]), row])
        coef = lsq_linear(A2, np.append(y, big * spec.max_sum), bounds=(lb, ub), method="bvls").x
    resid = y - coef[0] - X @ coef[1:]
    s2 = float(resid @ resid) / max(n - (k + 1), 1)
    se_a = np.sqrt(s2 / n)
    wh, wp = 1.0 / se_a ** 2, 1.0 / spec.tau_alpha ** 2
    a_post = (coef[0] * wh + spec.alpha_prior * wp) / (wh + wp)
    return {"alpha_hat": float(coef[0]), "alpha": float(a_post), "alpha_se": float(se_a), "beta": coef[1:],
            "sigma": float(np.sqrt(s2)), "resid": y - a_post - X @ coef[1:], "n": n,
            "r2": 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())}


def raw_fit(y, X, spec: Spec) -> dict:
    return fit(y, X, Spec(**{**spec.__dict__, "tau_alpha": 1e9}))


class Panel:
    """The ticker panels: the core's quarterly features joined to each ticker's quarters from its window."""

    def __init__(self, core: pd.DataFrame | None = None, ins: pd.DataFrame | None = None):
        self.core = D.core_quarterly() if core is None else core
        self.ins = D.instr_quarterly() if ins is None else ins
        F = pd.DataFrame(index=self.core.index)
        for a in ASSETS:
            F["x_" + a] = self.core["lr_" + a] - self.core["lr_cash"]
        for c in ("lr_cash", "d_gz", "rv_mkt", "tb3_end"):
            F[c] = self.core[c]
        self.F = F
        self.P = {}
        for t in CHOSEN:
            d = self.ins[self.ins["ticker"] == t].set_index("qend").sort_index()
            df = F.join(d[["lr", "lp", "y", "D", "P", "nav", "nii"]], how="inner")
            self.P[t] = df[df.index >= EST_START[t]].dropna(subset=["lr"])

    def arrays(self, t: str, pi: float | None = None, spec: Spec | None = None):
        sp = spec or primary(t)
        df = self.P[t]
        cols = ["x_" + a for a in sp.style] + sp.factors
        X = df[cols].to_numpy(float)
        ok = ~np.isnan(X).any(axis=1)
        df = df[ok]
        pi = PI.get(t, 1.0) if pi is None else pi
        return df.index, (df["lr"] - pi * df["lr_cash"]).to_numpy(), X[ok], df, cols


def fit_variant(Pn: Panel, t: str, origin, alpha_pol: str, load_pol: str, pi: float) -> dict:
    """One fit of ticker t on quarters <= origin: alpha policy raw / zero / group, own or pooled loadings."""
    sp = primary(t)
    idx, y, X, df, cols = Pn.arrays(t, pi)
    m = idx <= origin
    yt, Xt = y[m], X[m]
    if load_pol == "pooled" and t in GROUP:
        Ys, Xs = [], []
        for p in PEERS[GROUP[t]]:
            i2, y2, X2, _, _ = Pn.arrays(p, pi)
            mm = i2 <= origin
            if mm.sum() >= 8:
                Ys.append(y2[mm] - y2[mm].mean())
                Xs.append(X2[mm] - X2[mm].mean(axis=0))
        beta = raw_fit(np.concatenate(Ys), np.vstack(Xs), sp)["beta"]
        a_hat = float(np.mean(yt - Xt @ beta))
        r0 = yt - a_hat - Xt @ beta
        se = float(np.sqrt(r0 @ r0 / max(len(yt) - 1, 1))) / np.sqrt(len(yt))
        k = 1
    else:
        f = raw_fit(yt, Xt, sp)
        beta, a_hat, se = f["beta"], f["alpha_hat"], f["alpha_se"]
        k = X.shape[1] + 1
    prior, tau = sp.alpha_prior, sp.tau_alpha
    if alpha_pol == "raw":
        a, a_sd = a_hat, se
    else:
        if alpha_pol == "group" and t in GROUP:
            vals, ws = [], []
            for p in PEERS[GROUP[t]]:
                i2, y2, X2, _, _ = Pn.arrays(p, pi)
                mm = i2 <= origin
                if mm.sum() < 8:
                    continue
                fp = raw_fit(y2[mm], X2[mm], primary(p))
                vals.append(fp["alpha_hat"])
                ws.append(1.0 / fp["alpha_se"] ** 2)
            prior = float(np.average(vals, weights=ws))
        wh, wp = 1.0 / se ** 2, 1.0 / tau ** 2
        a = (a_hat * wh + prior * wp) / (wh + wp)
        a_sd = float(np.sqrt(1.0 / (wh + wp)))
    n = len(yt)
    return {"alpha": float(a), "alpha_sd": a_sd, "alpha_hat": float(a_hat), "alpha_se": float(se), "beta": beta,
            "cols": cols, "resid": yt - a - Xt @ beta, "infl": float(np.sqrt(n / max(n - k, 1))), "n": n, "k": k,
            "prior": float(prior)}


def oos(t: str, Pn: Panel | None = None, variant=None) -> dict:
    """Out-of-sample R2 (expanding and split, h = 1, 4, 8) of the chosen recipe and of the market benchmark."""
    Pn = Pn or Panel()
    ap, lp, cm = variant or CHOSEN[t]
    pi = {"excess": 1.0, "total": 0.0, "half": 0.5}[cm]
    idx, y_ex, X, df, cols = Pn.arrays(t, 1.0)
    lr, cash = df["lr"].to_numpy(), df["lr_cash"].to_numpy()
    _, _, X_m, _, _ = Pn.arrays(t, 1.0, market_spec())
    out = {}
    for proto in ("expanding", "split"):
        if proto == "expanding":
            origins = list(range(MIN_TRAIN - 1, len(idx) - 1))
        else:
            k0 = int(np.searchsorted(idx, SPLIT_END, side="right")) - 1
            origins = list(range(k0, len(idx) - 1))
        fs = fit_variant(Pn, t, idx[origins[0]], ap, lp, pi) if proto == "split" else None
        fm_s = raw_fit(y_ex[:origins[0] + 1], X_m[:origins[0] + 1], market_spec()) if proto == "split" else None
        rec = {h: [] for h in H}
        for i in origins:
            fv = fs if proto == "split" else fit_variant(Pn, t, idx[i], ap, lp, pi)
            fm = fm_s if proto == "split" else raw_fit(y_ex[:i + 1], X_m[:i + 1], market_spec())
            for h in H:
                if i + h >= len(idx):
                    continue
                sl = slice(i + 1, i + 1 + h)
                yhat = float(np.sum(pi * cash[sl] + fv["alpha"] + X[sl] @ fv["beta"]))
                yhat_m = float(np.sum(cash[sl] + fm["alpha"] + X_m[sl] @ fm["beta"]))
                bench = float(np.sum(cash[sl]) + h * y_ex[:i + 1].mean())
                rec[h].append((yhat, yhat_m, float(np.sum(lr[sl])), bench))
        for h in H:
            r = np.array(rec[h])
            if len(r) < 3:
                continue
            sseb = np.sum((r[:, 2] - r[:, 3]) ** 2)
            out[(proto, h)] = {"r2": 1 - np.sum((r[:, 2] - r[:, 0]) ** 2) / sseb,
                               "r2_market": 1 - np.sum((r[:, 2] - r[:, 1]) ** 2) / sseb,
                               "n": int(len(r)), "n_indep": int((len(r) + h - 1) // h),
                               "bias_per_year": float(np.mean(r[:, 2] - r[:, 0])) / h * 4}
    return out


def fit_recipes(Pn: Panel | None = None) -> dict:
    """The final recipes on the full samples (instr-mapped s16): {ticker: {loadings, alpha_hat_q, alpha_se_q,
    alpha_prior_q, alpha_q, alpha_sd_q, sigma_q, n, k, infl, resid {date: e}, rv_at {date: v}}} plus the group
    priors. Residuals are actual - (systematic + shrunk alpha), RAW (the draw demeans them over its pool)."""
    Pn = Pn or Panel()
    raw = {}
    for t in CHOSEN:
        idx, y, X, df, cols = Pn.arrays(t)
        raw[t] = {"idx": idx, "y": y, "X": X, "df": df, "cols": cols, "fit": raw_fit(y, X, primary(t))}
    pooled = {}
    for g, ps in PEERS.items():
        Ys = [raw[p]["y"] - raw[p]["y"].mean() for p in ps]
        Xs = [raw[p]["X"] - raw[p]["X"].mean(axis=0) for p in ps]
        pooled[g] = raw_fit(np.concatenate(Ys), np.vstack(Xs), primary(ps[0]))["beta"]
    prior_g = {g: float(np.average([raw[p]["fit"]["alpha_hat"] for p in ps],
                                   weights=[1 / raw[p]["fit"]["alpha_se"] ** 2 for p in ps])) for g, ps in PEERS.items()}
    out = {"group_alpha_prior_per_q": prior_g, "tickers": {}}
    for t, r in raw.items():
        y, X, df, cols = r["y"], r["X"], r["df"], r["cols"]
        if t in POOLED:
            beta = pooled[GROUP[t]]
            a_hat = float(np.mean(y - X @ beta))
            r0 = y - a_hat - X @ beta
            sig = float(np.sqrt(r0 @ r0 / (len(y) - 1)))
            se, k = sig / np.sqrt(len(y)), 1
        else:
            beta, a_hat, se, sig = r["fit"]["beta"], r["fit"]["alpha_hat"], r["fit"]["alpha_se"], r["fit"]["sigma"]
            k = X.shape[1] + 1
        prior = prior_g[GROUP[t]] if t in GROUP else primary(t).alpha_prior
        wh, wp = 1 / se ** 2, 1 / TAU_ALPHA ** 2
        a_post = (a_hat * wh + prior * wp) / (wh + wp)
        resid = y - a_post - X @ beta
        n = len(y)
        out["tickers"][t] = {"loadings": {c: float(b) for c, b in zip(cols, beta)}, "cash_passthrough": PI.get(t, 1.0),
                             "alpha_hat_q": float(a_hat), "alpha_se_q": float(se), "alpha_prior_q": float(prior),
                             "alpha_q": float(a_post), "alpha_sd_q": float(np.sqrt(1 / (wh + wp))),
                             "sigma_q": float(sig), "n": n, "k": k, "infl": float(np.sqrt(n / max(n - k, 1))),
                             "resid": {str(i.date()): float(v) for i, v in zip(df.index, resid)},
                             "rv_at": {str(i.date()): float(v) for i, v in zip(df.index, df["rv_mkt"])}}
    return out
