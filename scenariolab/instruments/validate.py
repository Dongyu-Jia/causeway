"""Historical replays of the recipes (INSTRUMENTS_DESIGN §3, T3 / T4 / T13), with what each one is fitted on.

  letf_replay        QLD 2006-07.., TQQQ 2010-03..2026-09 fed MONTHLY inputs only (NDX month-end returns, the panel's
                     fed funds, QQQ's dividend yield, the financing spreads; the RV state recursive from the month
                     before the start, no daily data), reported per window because the inputs are not all out of
                     sample: the RV model is fitted 1986-03..2014-12 and the spread before 2019 is the constant
                     train_2006_2018 that instr-structural (s07_letf) fitted on these funds' own NAV returns to
                     2018-12. So 2006..2018 is IN SAMPLE and only 2019-01..2026-09 (spreads from the N-PORT swap
                     terms disclosed before each month, the RV model 4+ years past its training end) is out of
                     sample. The full-window agreement (QLD 25.35 / 25.3, TQQQ 42.5 / 41.7 %/yr) nets out errors
                     of opposite sign; 2019-2026 fails T3 (QLD -1.6, TQQQ -4.6 %/yr). rv="actual" runs the same
                     formula on each month's actual daily RV, RS3 and RS4: within T3 in both windows, so the gap is
                     the RV state model's level (it over-predicts the NDX's variance by ~15% in 2006-2018 and
                     under-predicts it by ~15% in 2019-2026), not the leverage formula
  covered_call       QQQI 2024-02..2026-06 and QYLD 2019-08..2026-07 marked daily on 0.855 x VXN (third-Friday
                     rolls): design 22.57 / 22.28 and 8.60 / 8.19 %/yr. QQQI's coverage and strikes are the means /
                     medians of the 11 N-PORT reports spanning its scored months: IN SAMPLE
  tlt_replay         TLT 2019-09..2026-08 on actual 20 / 30-year yields (monthly RMSE 0.48%) and on the
                     simulation-feasible 10-year + spread model (1.24%; spread models fitted to 2014)
  qqq_tracking       QQQ 2019-07..2026-06: the nasdaq asset less the fee against N-PORT (TE 0.08%/yr)
  estimated_oos      the ten estimated recipes' out-of-sample R2 (estimated_fit.oos)
  estimated_replay   the actual core + actual residual dates + raw residuals reproduce every quarter exactly (T4)
  qld_year_end       QLD's 2006-12 / 2007-12 total return less 2 x NDX with the imputed distributions (T13)
Every replay keeps to complete months (data.complete_months). T3_TOLERANCE holds the design's T3 tolerances; a
window's "t3" says whether it meets them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import fees
from . import params as prm
from . import structural as st
from .vol import ndx_rv_path


def _panel_m() -> pd.DataFrame:
    from ..data.panel import load_panel
    P = load_panel().copy()
    P.index = pd.DatetimeIndex(P.index).to_period("M")
    return P


def _summary(actual: np.ndarray, model: np.ndarray, months: pd.PeriodIndex) -> dict:
    a, m = np.asarray(actual, float), np.asarray(model, float)
    ok = np.isfinite(a) & np.isfinite(m)
    a, m = a[ok], m[ok]
    e = a - m
    k = len(a)
    return {"months": int(k), "first": str(months[ok][0]), "last": str(months[ok][-1]),
            "ann_actual": float(np.prod(1 + a) ** (12 / k) - 1), "ann_model": float(np.prod(1 + m) ** (12 / k) - 1),
            "mean_err_ann": float(e.mean() * 12), "rmse_monthly": float(np.sqrt((e ** 2).mean())),
            "corr": float(np.corrcoef(a, m)[0, 1]), "te_ann": float(e.std() * np.sqrt(12))}


def _ndx_month_tables():
    ndx = D.ndx_daily()
    ok = D.complete_months(ndx)
    me = ndx.groupby(ndx.index.to_period("M")).last()
    me = me[me.index.isin(ok)]
    xm = np.log(me).diff().dropna()
    x = np.log(ndx).diff().dropna()
    rv = (x ** 2).groupby(x.index.to_period("M")).sum()
    dd = np.log(me / me.cummax())
    return xm, rv, dd


def s_eff_path(fund: str, months: pd.PeriodIndex) -> np.ndarray:
    """The financing spread in force each month: the last N-PORT report before it, the train constant before 2019."""
    sp = D.derived("letf_spreads.csv")
    s = sp[sp.fund == fund].copy()
    s["p"] = pd.PeriodIndex(pd.to_datetime(s["rep_date"]), freq="M")
    s = s.set_index("p")["s_eff_per_borrowed"].sort_index()
    train = prm.structural()["letf"][fund]["s_eff"]["train_2006_2018"]
    return np.array([float(s[s.index < m].iloc[-1]) if (s.index < m).any() else train for m in months])


# T3's tolerances in basis points (annualized difference "ann", monthly RMSE "rmse", tracking error "te")
T3_TOLERANCE_BP = {"QLD": {"ann": 50, "rmse": 60}, "TQQQ": {"ann": 100, "rmse": 150}, "QQQ": {"te": 20},
                   "QQQI": {"ann": 50}, "QYLD": {"ann": 60}, "TLT": {"rmse": 60}}
T3_TOLERANCE = {f: {k: v / 1e4 for k, v in d.items()} for f, d in T3_TOLERANCE_BP.items()}
LETF_IN_SAMPLE_END = "2018-12"          # the train_2006_2018 spread is fitted on the funds' NAV returns to here


def _t3(fund: str, s: dict) -> dict:
    tol = T3_TOLERANCE.get(fund, {})
    ok = True
    if "ann" in tol:
        ok &= abs(s["ann_model"] - s["ann_actual"]) <= tol["ann"]
    if "rmse" in tol:
        ok &= s["rmse_monthly"] <= tol["rmse"]
    if "te" in tol:
        ok &= s["te_ann"] <= tol["te"]
    return {"pass": bool(ok), **{f"{k}_max": v for k, v in tol.items()}}


def _daily_moments() -> pd.DataFrame:
    """Per complete month of the daily NDX: sum x, RV = sum x^2, RS3 = sum x^3, RS4 = sum x^4, trading days."""
    ndx = D.ndx_daily()
    ok = D.complete_months(ndx)
    cal = ndx.index.union(D.fred_daily("NASDAQCOM").index)
    x = np.log(ndx.reindex(cal)).diff().dropna()
    x = x[x.index.to_period("M").isin(ok)]
    g = pd.DataFrame({"x": x, "rv": x ** 2, "rs3": x ** 3, "rs4": x ** 4}).groupby(x.index.to_period("M"))
    return g.sum().assign(days=g["x"].count())


def letf_replay(fund: str, start: str, end: str = "2026-09", n_draws: int = 0, rv: str = "model") -> dict:
    """QLD / TQQQ against the funds' NAV total returns (instr-structural s15), the full window and per window
    ("in_sample" to LETF_IN_SAMPLE_END, "oos" after it), each with its T3 verdict. rv "model": monthly inputs only,
    the RV state recursive and E[RS3 | x, RV], E[RS4 | RV] (what a simulation can use); "actual": the month's own
    daily RV, RS3 and RS4 (the formula alone)."""
    S = prm.structural()["letf"][fund]
    L = float(S["L"])
    xm, rvt, dd = _ndx_month_tables()
    P = _panel_m()
    actual = D.letf_monthly(fund)
    months = pd.period_range(start, end, freq="M")
    months = months[months.isin(xm.index) & months.isin(actual.index)]
    x = xm.reindex(months).to_numpy()[None, :]
    prev = months[0] - 1
    init = {"lrv": float(np.log(rvt[prev])), "dd": float(dd[prev])}
    ff = (P["lvl_ff"].reindex(months).to_numpy() / 100)[None, :]
    q = D.ndx_div_yield(months)[None, :]
    fee = fees.fee_path(fund, months)[None, :]
    se = s_eff_path(fund, months)[None, :]
    carry = st.letf_carry(L, q, ff, se, fee, 12)
    V = ndx_rv_path(x, "M", init, noise=False)
    rs3 = prm.structural()["vol"]["rs3_M"]
    if rv == "actual":
        dm = _daily_moments().reindex(months)
        RV, R3, R4 = (dm[k].to_numpy()[None, :] for k in ("rv", "rs3", "rs4"))
        mean = np.expm1(st.letf_log_return(dm["x"].to_numpy()[None, :], RV, R3, L, carry, R4))[0]
    elif rv == "model":
        RV = V["rv"]
        r3 = rs3["a1"] * RV ** 1.5 + rs3["a2"] * x * RV
        mean = np.expm1(st.letf_log_return(x, RV, r3, L, carry, st.expected_rs4(RV, "M")))[0]
    else:
        raise ValueError(f"rv must be 'model' or 'actual' (got {rv!r})")
    a = actual.reindex(months).to_numpy()
    out = _summary(a, mean, months)
    out["rv_source"] = rv
    out["t3"] = _t3(fund, out)
    cut = pd.Period(LETF_IN_SAMPLE_END, "M")
    act_rv = _daily_moments()["rv"].reindex(months).to_numpy()
    out["windows"] = {}
    for name, m in (("in_sample", months <= cut), ("oos", months > cut)):
        if m.sum() < 12:
            continue
        w = _summary(a[m], mean[m], months[m])
        w["t3"] = _t3(fund, w)
        w["rv_ann_model"] = float(np.nanmean(V["rv"][0][m]) * 12)
        w["rv_ann_actual"] = float(np.nanmean(act_rv[m]) * 12)
        out["windows"][name] = w
    out["windows"]["in_sample"]["note"] = ("in sample: the RV model is fitted to 2014-12 and the spread before 2019 "
                                           "on these funds' NAV returns to 2018-12")
    if "oos" in out["windows"]:
        out["windows"]["oos"]["note"] = "out of sample: the spreads of the N-PORT reports before each month"
    if n_draws and rv == "model":
        X = np.repeat(x, n_draws, axis=0)
        Vn = ndx_rv_path(X, "M", init, noise=True, rng=np.random.default_rng(1))
        r3n = rs3["a1"] * Vn["rv"] ** 1.5 + rs3["a2"] * X * Vn["rv"]
        dr = np.expm1(st.letf_log_return(X, Vn["rv"], r3n, L, np.repeat(carry, n_draws, 0),
                                         st.expected_rs4(Vn["rv"], "M")))
        lo, hi = np.quantile(dr, [0.1, 0.9], axis=0)
        out["cover80"] = float(np.mean((a >= lo) & (a <= hi)))
    out["model_monthly"] = pd.Series(mean, index=months)
    return out


# ---------------------------------------------------------------- covered calls marked daily on VXN

def roll_dates(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Monthly option expirations: the third Friday, or the trading day before it when it is a holiday."""
    out = []
    for p in pd.period_range(idx.min(), idx.max(), freq="M"):
        d = pd.Timestamp(p.year, p.month, 1)
        f = d + pd.Timedelta(days=(4 - d.weekday()) % 7) + pd.Timedelta(days=14)
        prior = idx[idx <= f]
        if len(prior) and (f - prior[-1]).days <= 3:
            out.append(prior[-1])
    return out


def _daily_inputs() -> pd.DataFrame:
    ndx = D.ndx_daily()
    cal = ndx.index
    x = pd.DataFrame({"ndx": ndx})
    x["vxn"] = D.vxn_daily().reindex(cal)
    dtb3 = D.fred_daily("DTB3")
    x["dtb3"] = dtb3.reindex(dtb3.index.union(cal)).ffill(limit=5).reindex(cal).ffill() / 100.0
    dy = D.qqq_fy()
    lo = min(dy)
    fyear = np.where(cal.month >= 10, cal.year + 1, cal.year)
    x["q"] = [dy.get(int(f), dy[max(dy)] if f > max(dy) else dy[lo]) for f in fyear]
    x.loc[x.index < pd.Timestamp("2001-10-01"), "q"] = dy[lo]
    return x


def daily_marked(x: pd.DataFrame, sig_daily: pd.Series, strikes, fee: float, start: str, end: str) -> pd.Series:
    """Daily NAV of index + calls written at each roll on fractions c_i at k_i x the roll's index, marked with
    Black-Scholes at the day's vol, chained across cycles (instr-structural s09)."""
    xx = x.loc[start:end]
    rd = [d for d in roll_dates(x.index) if d >= xx.index[0] - pd.Timedelta(days=40)]
    nav = pd.Series(np.nan, index=xx.index)
    v = 1.0
    for t0, t1 in zip(rd[:-1], rd[1:]):
        seg = x.loc[t0:t1]
        if seg.index[-1] < xx.index[0] or seg.index[0] > xx.index[-1]:
            continue
        S0, r, q = seg["ndx"].iloc[0], seg["dtb3"].iloc[0], seg["q"].iloc[0]
        T0 = (t1 - t0).days / 365.0
        s0 = float(sig_daily.loc[t0])
        p0 = [float(st.bs_call(1.0, k, T0, r, q, s0)) for k, _ in strikes]
        tau = np.array([(t1 - d).days / 365.0 for d in seg.index])
        el = np.array([(d - t0).days / 365.0 for d in seg.index])
        Srel = seg["ndx"].to_numpy() / S0
        sg = sig_daily.reindex(seg.index).ffill().to_numpy()
        val = Srel + q * el - fee * el
        for (k, c), p in zip(strikes, p0):
            cv = np.where(tau > 1e-6, st.bs_call(Srel, k, tau, r, q, sg), np.maximum(Srel - k, 0))
            val = val + c * (p * (1 + r * el) - cv)
        path = v * val
        for d, pv in zip(seg.index[1:], path[1:]):
            if d in nav.index:
                nav.loc[d] = pv
        v = float(path[-1])
    return nav


def covered_call(fund: str = "QQQI") -> dict:
    """QQQI (two N-PORT strikes, 63% coverage, 0.68%) or QYLD (ATM, 100%, 0.60%) marked daily on 0.855 x VXN."""
    S = prm.structural()
    x = _daily_inputs()
    sig = (S["vol"]["iv_atm_multiplier_on_vxn"] * x["vxn"] / 100.0).ffill()
    if fund == "QQQI":
        q = S["qqqi"]
        c = q["coverage_total"]
        strikes, start = ((q["k1"], c / 2), (q["k2"], c / 2)), "2024-01-25"
    else:
        strikes, start = ((S["qyld_like"]["k"], S["qyld_like"]["coverage"]),), "2019-06-01"
    nav = daily_marked(x, sig, strikes, fees.fee(fund), start, str(x.index.max().date()))
    ok = D.complete_months(D.ndx_daily())
    mo = nav.dropna().groupby(nav.dropna().index.to_period("M")).last().pct_change().dropna()
    mo = mo[mo.index.isin(ok)]
    act = D.nport_returns(fund)
    months = mo.index[mo.index.isin(act.index)]
    out = _summary(act.reindex(months).to_numpy(), mo.reindex(months).to_numpy(), months)
    ndx_tr = (x["ndx"].groupby(x.index.to_period("M")).last().pct_change()
              + x["q"].groupby(x.index.to_period("M")).mean() / 12).reindex(months)
    out["beta_actual_on_ndx"] = float(np.polyfit(ndx_tr, act.reindex(months), 1)[0])
    out["beta_model_on_ndx"] = float(np.polyfit(ndx_tr, mo.reindex(months), 1)[0])
    out["t3"] = _t3(fund, out)
    if fund == "QQQI":
        out["in_sample"] = True
        out["note"] = ("in sample: the coverage (0.627) and the two strikes are the means / medians of the 11 N-PORT "
                       "reports (2024-02..2026-06) that span the scored months")
    return out


# ---------------------------------------------------------------- TLT

def tlt_replay() -> dict:
    """TLT against N-PORT: "struct" on the actual 20 / 30-year yields, "sim" on the 10-year + the spread model
    run from 2015-01 on the actual 10-year and fed funds (mean path)."""
    from .fit import monthly_long_yields
    T = prm.structural()["tlt"]
    M = monthly_long_yields()
    w = T["w20"]
    sims = {}
    for col in ("s20", "s30"):
        p = T["spread_models"][col]
        idx = M.index
        start_i = int(np.searchsorted(idx, pd.Timestamp("2015-01-31")))
        cur = float(M[col].iloc[start_i - 1])
        s = pd.Series(np.nan, index=idx)
        s.iloc[start_i - 1] = cur
        for i in range(start_i, len(M)):
            sl0, sl1 = M["slope"].iloc[i - 1], M["slope"].iloc[i]
            dy = M["y10"].iloc[i] - M["y10"].iloc[i - 1]
            if np.isfinite(sl0) and np.isfinite(sl1) and np.isfinite(dy):
                cur = cur + (p["a"] + p["b_dslope"] * (sl1 - sl0) + p["g_dy10"] * dy
                             - p["kappa"] * (cur - (p["c0"] + p["c1"] * sl0)))
            s.iloc[i] = cur
        sims[col] = s
    yT = w * M["y20"] + (1 - w) * M["y30"]
    yT = yT.where(M["y30"].notna(), M["y20"])
    yT_sim = M["y10"] + w * sims["s20"] + (1 - w) * sims["s30"]
    act = D.nport_returns("TLT")
    out = {}
    for tag, y in (("struct", yT), ("sim", yT_sim)):
        y0, y1 = y.shift(1), y
        dy = y1 - y0
        r = y0 / 12 - np.polyval(T["D_of_y"], y0) * dy + 0.5 * np.polyval(T["C_of_y"], y0) * dy ** 2 - fees.fee("TLT") / 12
        r.index = r.index.to_period("M")
        months = act.index[act.index.isin(r.dropna().index)]
        out[tag] = _summary(act.reindex(months).to_numpy(), r.reindex(months).to_numpy(), months)
        out[tag]["t3"] = _t3("TLT", out[tag]) if tag == "struct" else None     # T3 is on the actual yields
    return out


def qqq_tracking() -> dict:
    """QQQ = the panel's nasdaq asset less the fee in force, against its N-PORT returns."""
    P = _panel_m()
    act = D.nport_returns("QQQ")
    months = act.index[act.index.isin(P.index)]
    model = P["r_nasdaq"].reindex(months).to_numpy() - fees.fee_path("QQQ", months) / 12
    out = _summary(act.reindex(months).to_numpy(), model, months)
    out["t3"] = _t3("QQQ", out)
    return out


# ---------------------------------------------------------------- estimated

def estimated_oos(tickers=None) -> pd.DataFrame:
    from . import estimated_fit as EF
    Pn = EF.Panel()
    rows = []
    for t in tickers or list(EF.CHOSEN):
        res = EF.oos(t, Pn)
        for (proto, h), v in res.items():
            rows.append({"ticker": t, "protocol": proto, "h": h, **v})
    return pd.DataFrame(rows)


def estimated_replay(tickers=None) -> dict:
    """{ticker: max |simulated - actual| quarterly log total return} on 2016Q2..2026Q2 (the joint pool): the actual
    core quarters as the core, the actual quarter as each path's residual date, raw residuals (T4)."""
    from . import estimated as E
    from .core import CoreView
    cq = D.core_quarterly()
    pool = E.pool_dates()
    sel = cq.loc[pd.DatetimeIndex(pool)]
    Q = len(pool)
    ids = list(prm.estimated()["tickers"])
    total = np.stack([np.expm1(sel[f"lr_{a}"].to_numpy()) for a in D.CORE_ASSETS], axis=1)[None]
    income = np.stack([sel[f"inc_{a}"].to_numpy() for a in D.CORE_ASSETS], axis=1)[None]
    lvl = lambda c: np.concatenate([[cq[c].shift(1).loc[pd.Timestamp(pool[0])]], sel[c].to_numpy()])[None]  # noqa: E731
    core = CoreView(freq="Q", labels=[str(pool[0])[:7]] * (Q + 1), assets=list(D.CORE_ASSETS), total=total,
                    income=income, cpi=np.ones((1, Q + 1)), y10=lvl("y10_end"), ff=lvl("ff_end"),
                    spread=lvl("baa10_end"), d_gz=sel["d_gz"].to_numpy()[None], tb3=sel["tb3_end"].to_numpy()[None],
                    set_id="replay", key="history")
    res = E.simulate(core, None, tickers or ids, residual_mode="replay", replay_idx=np.arange(Q)[None],
                     alpha_uncertainty=False, c_lr_mode="frozen")
    out = {}
    for t, ip in res.items():
        a = D.instr_quarterly(t)["lr"].reindex(pd.DatetimeIndex(pool)).to_numpy()
        out[t] = float(np.nanmax(np.abs(np.log1p(ip.total[0]) - a)))
    return out


def qld_year_end() -> dict:
    """QLD's total return on the two imputed ex-date months against the daily-reset period formula on the month's
    actual daily NDX (2 x the index, the variance drag and third moment of its daily returns, the carry), T13:
    the residual is about -6% / -5% without the imputed distributions."""
    tr = D.proshares_total_return("QLD")
    ndx = D.ndx_daily()
    P = _panel_m()
    x_d = np.log(ndx).diff().dropna()
    out = {}
    for m in ("2006-12", "2007-12"):
        p = pd.Period(m, "M")
        a = tr[tr.index.to_period("M") <= p].iloc[-1] / tr[tr.index.to_period("M") < p].iloc[-1] - 1
        xd = x_d[x_d.index.to_period("M") == p]
        q = float(D.ndx_div_yield(pd.PeriodIndex([p]))[0])
        carry = st.letf_carry(2.0, q, float(P.loc[p, "lvl_ff"]) / 100,
                              prm.structural()["letf"]["QLD"]["s_eff"]["train_2006_2018"], fees.fee("QLD", p.to_timestamp()), 12)
        model = float(np.expm1(st.letf_log_return(xd.sum(), (xd ** 2).sum(), (xd ** 3).sum(), 2.0, carry, (xd ** 4).sum())))
        out[m] = {"qld_tr": float(a), "model": model, "residual": float(a - model),
                  "imputed_share_of_nav": float(D.imputed_distributions("QLD").loc[pd.Timestamp(m + "-20")]
                                                / D.proshares_nav("QLD").shift(1).loc[pd.Timestamp(m + "-20")])}
    return out


def summary() -> dict:
    """Every replay (instrument_meta.json["validation"]); data-dependent, run by the tests and on request. The
    leveraged funds carry their per-window results and the same replay on the actual RV ("actual_rv")."""
    out = {"QLD": letf_replay("QLD", "2006-07"), "TQQQ": letf_replay("TQQQ", "2010-03"),
           "QQQI": covered_call("QQQI"), "QYLD": covered_call("QYLD"), "TLT": tlt_replay(), "QQQ": qqq_tracking()}
    for k, start in (("QLD", "2006-07"), ("TQQQ", "2010-03")):
        out[k].pop("model_monthly", None)
        a = letf_replay(k, start, rv="actual")
        a.pop("model_monthly", None)
        out[k]["actual_rv"] = a
    return out
