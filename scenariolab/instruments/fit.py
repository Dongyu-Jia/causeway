"""Explicit refits of the instrument layer's estimated parameters (INSTRUMENTS_DESIGN §5.5, T16).

Refitting never happens with a data update: it is this explicit step, and every fit is cut at its train_end, so
running it on newer data reproduces the parameters in params/ (T16 checks that) until someone moves train_end.

    python -m scenariolab.instruments.fit --check          # refit everything and compare with params/
    python -m scenariolab.instruments.fit --write          # rewrite the fitted blocks of params/ (review the diff)

What is fitted here (the rest of params/ are disclosed contract terms or calibrated assumptions, see each file):
  vol_ndx(freq)        NASDAQ-100 log RV state model (daily NDX, the panel's Baa-10y), 1986..2014, with the third /
                       fourth daily moments the leveraged formula needs (rs3, rs4) and, monthly, the variance of
                       monthly log returns per unit of RV (the quarterly Brownian bridge's month variance)
  iv_monthly()         month-end VXN on the RV state, 2001-02..2014-12
  vol_joint(freq)      the US market's RV equation on the same terms and the shock correlation with the NDX's
  d_gz_bridges()       quarterly GZ spread change on the Baa-10y change (and the market), 1990Q1..2014Q4
  tlt_spreads()        20y / 30y spreads over the 10-year: error correction on the 10y - fed funds slope, to 2014
  estimated(...)       the ten estimated recipes (estimated.fit_recipes)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import params as prm

TRAIN_END_M = "2014-12"


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients (constant first), residuals and the dof-corrected residual sd."""
    Z = np.column_stack([np.ones(len(y)), X]) if np.size(X) else np.ones((len(y), 1))
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    e = y - Z @ b
    return b, e, float(np.sqrt(e @ e / max(len(y) - Z.shape[1], 1)))


def _panel():
    from ..data.panel import load_panel
    return load_panel()


# ---------------------------------------------------------------- NASDAQ-100 RV state (instr-structural s06)

FEATS = ["lrv_lag", "abs_r", "neg_r", "dd", "dspread"]


def period_table(freq: str, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per period of the daily NDX: r (sum of daily log returns), rv, rs3, n, dd (log drawdown of the period
    closes), vxn_end, the panel's Baa-10y spread at the period end and its change, and the regressors."""
    ndx = D.ndx_daily()
    ok = D.complete_months(ndx)
    # daily returns on the NASDAQ trading calendar (NDX and Composite dates): a day the NDX is missing drops
    # both returns around it rather than merging them into one two-day return (as instr-structural fitted)
    cal = ndx.index.union(D.fred_daily("NASDAQCOM").index)
    x = np.log(ndx.reindex(cal)).diff().dropna()
    x = x[x.index.to_period("M").isin(ok)]
    key = x.index.to_period(freq)
    g = pd.DataFrame({"x": x, "x2": x ** 2, "x3": x ** 3, "x4": x ** 4, "key": key}).groupby("key")
    T = pd.DataFrame({"r": g["x"].sum(), "rv": g["x2"].sum(), "rs3": g["x3"].sum(), "rs4": g["x4"].sum(),
                      "n": g["x"].count()})
    px = ndx[ndx.index.to_period("M").isin(ok)]
    close = px.groupby(px.index.to_period(freq)).last()
    T["close"] = close
    T["dd"] = np.log(T["close"] / T["close"].cummax())
    vxn = D.vxn_daily()
    T["vxn_end"] = vxn.groupby(vxn.index.to_period(freq)).last() / 100.0
    P = _panel() if panel is None else panel
    sp = P["s_spread"].copy()
    sp.index = pd.DatetimeIndex(sp.index).to_period("M")
    if freq == "Q":
        sp = sp.groupby(sp.index.asfreq("Q")).last()
    T["spread"] = sp.reindex(T.index)
    T["dspread"] = T["spread"].diff()
    T["lrv"] = np.log(T["rv"])
    T["lrv_lag"] = T["lrv"].shift(1)
    T["abs_r"] = T["r"].abs()
    T["neg_r"] = np.minimum(T["r"], 0.0)
    return T[T.index >= pd.Period("1985-11", "M").asfreq(freq)]


def vol_ndx(freq: str = "M", train_end: str = TRAIN_END_M, panel=None) -> dict:
    """{rv_<f>, rv_<f>_nospread, rs3_<f>, days_<f>}: OLS of log RV on the lag, |r|, min(r, 0), DD (and dspread),
    on periods up to train_end (the first period has no lag)."""
    T = period_table(freq, panel)
    end = pd.Period(train_end, "M").asfreq(freq)
    tr = T.loc[:end]
    out = {}
    t1 = tr.dropna(subset=["lrv"] + FEATS).iloc[1:]
    b, e, s = ols(t1["lrv"].to_numpy(float), t1[FEATS].to_numpy(float))
    out[f"rv_{freq}"] = {"feats": FEATS, "coef": [float(v) for v in b], "sigma": s, "n": int(len(t1)),
                         "sample": [str(t1.index.min()), str(t1.index.max())]}
    f2 = FEATS[:4]
    t2 = tr.iloc[1:].dropna(subset=["lrv"] + f2)
    b2, _, s2 = ols(t2["lrv"].to_numpy(float), t2[f2].to_numpy(float))
    out[f"rv_{freq}_nospread"] = {"feats": f2, "coef": [float(v) for v in b2], "sigma": s2, "n": int(len(t2))}
    t3 = tr.iloc[1:].dropna(subset=["rs3", "rv", "r"])
    a = np.linalg.lstsq(np.column_stack([t3["rv"] ** 1.5, t3["r"] * t3["rv"]]), t3["rs3"].to_numpy(), rcond=None)[0]
    out[f"rs3_{freq}"] = {"form": "rs3 = a1*rv**1.5 + a2*x*rv", "a1": float(a[0]), "a2": float(a[1]), "n": int(len(t3))}
    t4 = tr.iloc[1:].dropna(subset=["rs4", "rv", "n"])
    k4 = (t4["rs4"] / (t4["rv"] ** 2 / t4["n"])).median()
    out[f"rs4_{freq}"] = {"form": "rs4 = k*rv**2/days", "k": float(k4), "n": int(len(t4)),
                          "note": "median over the training periods of sum(x^4) / (RV^2 / trading days); 3 for "
                                  "Gaussian days of constant vol"}
    out[f"days_{freq}"] = float(T["n"].median())
    if freq == "M":
        m = tr.loc[pd.Period("1986-01", "M"):].dropna(subset=["r", "rv"])
        out["bridge_month_var_ratio"] = {
            "ratio": float(((m["r"] - m["r"].mean()) ** 2).mean() / m["rv"].mean()), "n": int(len(m)),
            "sample": [str(m.index.min()), str(m.index.max())],
            "note": "var(monthly NDX log return) / mean(monthly RV): the month variance of the quarterly Brownian "
                    "bridge is this ratio x RV_Q / 3 (realized daily variance runs above the variance of monthly "
                    "returns, so RV_Q / 3 alone made the bridged months too volatile)"}
    return out


def iv_monthly(train_end: str = TRAIN_END_M, panel=None) -> dict:
    """log VXN (month end) on the log realized vol of the month and the one before, DD and dspread."""
    T = period_table("M", panel)
    T["lvol_rv"] = 0.5 * np.log(T["rv"] * 12)
    T["lvol_rv_lag"] = T["lvol_rv"].shift(1)
    T["lvxn"] = np.log(T["vxn_end"])
    f = ["lvol_rv", "lvol_rv_lag", "dd", "dspread"]
    tr = T.loc[:pd.Period(train_end, "M")].dropna(subset=["lvxn"] + f)
    b, _, s = ols(tr["lvxn"].to_numpy(float), tr[f].to_numpy(float))
    return {"feats": f, "coef": [float(v) for v in b], "sigma": s, "n": int(len(tr)),
            "sample": [str(tr.index.min()), str(tr.index.max())]}


# ---------------------------------------------------------------- the joint state: the US market's equation

def vol_joint(freq: str = "M", train_end: str = TRAIN_END_M, panel=None) -> dict:
    """The US market's log RV equation (Ken French daily) on the core's own variables, fitted like the NDX's, and
    the correlation of the two equations' residuals (instr-design d03 monthly, d02 quarterly).

    Monthly: x = the panel's log price returns (p_nasdaq, p_us_equity), dspread = the panel's f_spread, drawdowns
    of the cumulated x from 1986-01, sample 1986-02..train_end. Quarterly: x = the quarter's log price return,
    dspread = the change of the quarter-end Baa-10y, sample 1986..train_end."""
    P = _panel() if panel is None else panel
    end = pd.Period(train_end, "M")
    if freq == "M":
        Pm = P.copy()
        Pm.index = pd.DatetimeIndex(Pm.index).to_period("M")
        v = pd.DataFrame({"x_ndx": np.log1p(Pm["p_nasdaq"].astype(float)),
                          "x_mkt": np.log1p(Pm["p_us_equity"].astype(float)), "dsp": Pm["f_spread"].astype(float)})
        v["l_ndx"] = np.log(D.ndx_rv_table()["rv"].reindex(v.index))
        v["l_mkt"] = np.log(D.mkt_rv("M")["rv"].reindex(v.index))
        v = v[v.index >= pd.Period("1986-01", "M")]
    else:
        cq = D.core_quarterly(P)
        cq.index = pd.PeriodIndex(cq.index, freq="Q")
        rvq = D.ndx_rv_table()["rv"]
        rvq = rvq.groupby(rvq.index.asfreq("Q")).agg(["sum", "count"])
        rvq = rvq[rvq["count"] == 3]["sum"]
        v = pd.DataFrame({"x_ndx": cq["lp_nasdaq"], "x_mkt": cq["lp_us_equity"], "dsp": cq["d_baa10"],
                          "l_ndx": np.log(rvq.reindex(cq.index)), "l_mkt": np.log(cq["rv_mkt_ss"])})
        v = v[v.index >= pd.Period("1986Q1", "Q")]
        end = end.asfreq("Q")
    for a in ("ndx", "mkt"):
        v[f"lag_{a}"] = v[f"l_{a}"].shift(1)
        v[f"abs_{a}"] = v[f"x_{a}"].abs()
        v[f"neg_{a}"] = v[f"x_{a}"].clip(upper=0)
        lp = v[f"x_{a}"].fillna(0).cumsum()
        v[f"dd_{a}"] = lp - lp.cummax()
    v = v.dropna()
    tr = v[v.index <= end]
    res, out = {}, {"train": [str(tr.index.min()), str(tr.index.max())]}
    for a in ("ndx", "mkt"):
        xs = [f"lag_{a}", f"abs_{a}", f"neg_{a}", f"dd_{a}", "dsp"]
        b, e, _ = ols(tr[f"l_{a}"].to_numpy(), tr[xs].to_numpy())
        res[a] = e
        out[a] = {"feats": ["lrv_lag", "abs_r", "neg_r", "dd", "dspread"], "coef": [float(c) for c in b],
                  "sigma": float(np.std(e, ddof=6)), "n": int(len(tr))}
    out["shock_corr"] = float(np.corrcoef(res["ndx"], res["mkt"])[0, 1])
    return out


# ---------------------------------------------------------------- the GZ bridge (defect 2)

def d_gz_bridges(train=("1990-01-01", "2014-12-31"), panel=None) -> dict:
    """Quarterly d_gz on d(Baa-10y) [+ the US market's log total return (+ financials)], fitted on train, scored
    after it (instr-design p04). The default bridge is spread_eq; no bridge noise by default."""
    cq = D.core_quarterly(panel)
    h = cq[cq.index >= train[0]].dropna(subset=["d_gz", "d_baa10", "lr_us_equity", "lr_financials"])
    tr, te = h[h.index <= train[1]], h[h.index > train[1]]
    out = {}
    for name, xs in (("spread", ["d_baa10"]), ("spread_eq", ["d_baa10", "lr_us_equity"]),
                     ("spread_eq_fin", ["d_baa10", "lr_us_equity", "lr_financials"])):
        b, e, s = ols(tr["d_gz"].to_numpy(), tr[xs].to_numpy())
        pred = np.column_stack([np.ones(len(te)), te[xs].to_numpy()]) @ b
        et = te["d_gz"].to_numpy() - pred
        out[name] = {"x": xs, "coef": [float(v) for v in b], "sd_e_train": float(np.std(e, ddof=len(xs) + 1)),
                     "x_mean_train": {x: float(tr[x].mean()) for x in xs},
                     "rmse_test": float(np.sqrt((et ** 2).mean())),
                     "r2_test": float(1 - (et ** 2).sum() / ((te["d_gz"] - tr["d_gz"].mean()) ** 2).sum()),
                     "n_train": int(len(tr)), "n_test": int(len(te))}
    out["hist"] = {"corr_dspread_us_equity": float(h["d_baa10"].corr(h["lr_us_equity"])),
                   "corr_dgz_us_equity": float(h["d_gz"].corr(h["lr_us_equity"])),
                   "span": [str(h.index.min().date()), str(h.index.max().date())]}
    return out


# ---------------------------------------------------------------- TLT's long-yield model (instr-structural s10)

def monthly_long_yields(start: str = "1971-02") -> pd.DataFrame:
    """Month-end 10/20/30-year CMT yields (daily, gaps filled with the monthly GS20 / GS30 / GS10 means) and the
    monthly average fed funds, decimal, from `start` (instr-structural's daily table began with the NASDAQ
    Composite in 1971-02, so its spread fits start in 1971-03 / 1977-03)."""
    cal = D.ndx_daily().index.union(D.fred_daily("NASDAQCOM").index)       # the NASDAQ trading calendar
    cal = cal[cal >= pd.Timestamp(start)]

    def me(s: pd.Series) -> pd.Series:      # the value on the month's last trading day (carried <= 5 days)
        on_cal = s.reindex(s.index.union(cal)).ffill(limit=5).reindex(cal)
        return on_cal.dropna().resample("ME").last()
    M = pd.DataFrame({"y10": me(D.fred_daily("DGS10")), "y20": me(D.fred_daily("DGS20")),
                      "y30": me(D.fred_daily("DGS30"))})
    for c, sid in (("y20", "GS20"), ("y30", "GS30"), ("y10", "GS10")):
        g = D.fred_daily(sid).copy()
        g.index = g.index + pd.offsets.MonthEnd(0)
        M[c] = M[c].combine_first(g)
    ff = D.fred_daily("FEDFUNDS").copy()
    ff.index = ff.index + pd.offsets.MonthEnd(0)
    M["ff"] = ff.reindex(M.index)
    M = M / 100.0
    M["slope"] = M["y10"] - M["ff"]
    M["s20"] = M["y20"] - M["y10"]
    M["s30"] = M["y30"] - M["y10"]
    return M.loc["1962":]


def tlt_spreads(train_end: str = "2014-12-31") -> dict:
    """Two-step error correction of each long spread: level s = c0 + c1 slope (OLS of levels), then
    ds = a + b dslope + g dy10 - kappa (s_lag - c0 - c1 slope_lag) + e on months up to train_end."""
    M = monthly_long_yields()
    out = {}
    for col in ("s20", "s30"):
        d = pd.DataFrame({"ds": M[col].diff(), "dslope": M["slope"].diff(), "dy10": M["y10"].diff(),
                          "s_lag": M[col].shift(1), "slope_lag": M["slope"].shift(1)}).dropna()
        tr = d[d.index <= train_end]
        L = pd.DataFrame({"s": M[col], "slope": M["slope"]}).dropna()
        Ltr = L[L.index <= train_end]
        c1, c0 = np.polyfit(Ltr["slope"], Ltr["s"], 1)
        gap = (tr["s_lag"] - (c0 + c1 * tr["slope_lag"])).to_numpy()
        X = np.column_stack([np.ones(len(tr)), tr["dslope"], tr["dy10"], gap])
        b = np.linalg.lstsq(X, tr["ds"].to_numpy(), rcond=None)[0]
        e = tr["ds"].to_numpy() - X @ b
        out[col] = {"col": col, "c0": float(c0), "c1": float(c1), "a": float(b[0]), "b_dslope": float(b[1]),
                    "g_dy10": float(b[2]), "kappa": float(-b[3]), "sigma": float(np.std(e)), "n": int(len(tr)),
                    "sample": [str(tr.index.min().date()), str(tr.index.max().date())]}
    return out


# ---------------------------------------------------------------- compare / write

def _close(a, b, rtol: float, atol: float) -> bool:
    if isinstance(a, dict):
        return all(_close(a[k], b.get(k), rtol, atol) for k in a if k in b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, rtol, atol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))
    return a == b


def check(rtol: float = 1e-6, atol: float = 1e-8) -> dict:
    """Refit the fitted blocks and compare with params/: {block: True / False}."""
    S, V, B = prm.structural(), prm.vol_joint(), prm.bridges()
    out = {}
    for f in ("M", "Q"):
        r = vol_ndx(f)
        for k in (f"rv_{f}", f"rv_{f}_nospread", f"rs3_{f}", f"rs4_{f}"):
            out[f"vol.{k}"] = _close({kk: r[k][kk] for kk in ("coef", "sigma", "a1", "a2", "k") if kk in r[k]},
                                     S["vol"][k], rtol, atol)
        if f == "M":
            out["vol.bridge_month_var_ratio"] = _close({"ratio": r["bridge_month_var_ratio"]["ratio"]},
                                                       S["vol"]["bridge_month_var_ratio"], rtol, atol)
        j = vol_joint(f)
        key = "monthly" if f == "M" else "quarterly"
        out[f"vol_joint.{key}.mkt"] = _close({"coef": j["mkt"]["coef"], "sigma": j["mkt"]["sigma"]},
                                             V[key]["mkt"], rtol, atol)
        out[f"vol_joint.{key}.rho"] = _close(j["shock_corr"], V[key]["rho"], rtol, atol)
    out["vol.iv_M"] = _close({k: v for k, v in iv_monthly().items() if k in ("coef", "sigma")}, S["vol"]["iv_M"],
                             rtol, atol)
    br = d_gz_bridges()
    out["bridges.d_gz"] = all(_close({"coef": br[k]["coef"]}, B["d_gz"]["models"][k], rtol, atol)
                              for k in ("spread", "spread_eq", "spread_eq_fin"))
    ts = tlt_spreads()
    out["structural.tlt.spread_models"] = _close(ts, S["tlt"]["spread_models"], rtol, atol)
    from .estimated_fit import fit_recipes
    E, fr = prm.estimated(), fit_recipes()
    out["estimated.group_alpha_prior"] = _close(fr["group_alpha_prior_per_q"], E["group_alpha_prior_per_q"], 0, 1e-7)
    keys = ("loadings", "alpha_hat_q", "alpha_se_q", "alpha_prior_q", "alpha_q", "alpha_sd_q", "sigma_q", "infl",
            "resid", "rv_at")
    for t, r in fr["tickers"].items():
        # the core's quarterly table is stored with 8 significant digits in the research archive: 1e-7 absolute
        out[f"estimated.{t}"] = (_close({k: r[k] for k in keys}, E["tickers"][t], 0, 1e-7)
                                 and sorted(r["resid"]) == sorted(E["tickers"][t]["resid"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Refit the instrument layer's estimated parameters and compare")
    ap.add_argument("--check", action="store_true", help="refit and compare with params/ (the default)")
    ap.parse_args(argv)
    res = check()
    print(json.dumps(res, indent=1))
    return 0 if all(res.values()) else 1


PARAMS_DIR = Path(__file__).resolve().parent / "params"

if __name__ == "__main__":
    raise SystemExit(main())
