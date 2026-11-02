"""T-F and T-C: the factor layer's acceptance statistics (SIM_DESIGN §5.8), written to data/results/qmodel.json.

T-F (posterior predictive checks): fits at config_q.TF_ENDS, each run TF_QUARTERS (52) quarters forward
unconditionally with TF_PATHS paths; every statistic is computed per simulated 13-year path and history's value must
lie within the simulated [p05, p95] (non-overlapping history windows where the statistic needs a window):

  F1  delivery of stated values: hard |error| < 1e-8 on every path; soft 68% +- 5% inside +-1 stated sd
  F2  CPI y/y > 4% share; 4-quarter autocorrelation of CPI y/y (Pearson, within the window, non-overlapping
      13-year windows); mean high-inflation spell
  F3  contractions per 15 years and mean length within [p05, p95]; share longer than 6 quarters <= 25%
  F4  bounce-back: 4-quarter IP growth on the drawdown from the 8-quarter peak
  F5  co-movement: Δff on Δu, Okun (Δu on IP growth), corr(d_gz, us_equity), corr(d_spread, us_equity) (the
      instruments' T18), stock-bond correlation by s_core4 (us_equity through the asset layer L0)
  F6  bounds: ff >= 0.1 always; u >= 2.5 always and < 3% in <= 5% of quarters; CPI y/y > 15% in <= 0.5%
  F7  fill-in skill with CPI, IP and oil stated (rolling 1990-2026, 4Q / 12Q): R² of the unstated factors against
      no change; >= the engine on unemployment and the Baa spread, >= -0.05 elsewhere, d_usd held at no change
  F8  80% coverage of 4-quarter factor changes, non-overlapping windows from 1985 (41): within [0.72, 0.88]
  F9  the do-operator: a stated -1 sd u_ff leaves same-quarter E0 exactly unchanged; signs (equities up, d_y2 and
      the dollar down on impact, unemployment's response peaking at 4-8 quarters); a stated raw cut with no
      recession stated imputes nothing
  F10 step / path consistency (§3.7)
T-C (completion): step-mode completion of the unstated x fields under three masks -- (a) a random 50% of x, (b) only
E0 and E0p stated, (c) only financial conditions stated -- with completion R² against the model's unconditional
(state-only) prediction, CRPS and 80% coverage; the completion weight w and the regime covariance are selected on the
rolling validation origins (yearly refits, origins 1995Q1-2014Q4) and scored once on the test (fit at TC_TEST_END,
2015Q1-2026Q3); gate: under (b) every E1 field's completion R² on the test >= -0.02.

Pass / fail is shown, nothing is gated (§14 P3); P5 freezes the protocol. No LLM, no download.

    python -m scenariolab.qmodel.factor_checks [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .. import jsonio
from ..engine import regimes as RG
from . import config_q as cq
from . import factors as F
from . import panel as qpanel
from . import policy

S = F.Statement
QP = {**RG.PARAMS, "u_smooth_months": 1}       # the classifier on quarter-end series (joint j02)


# ================================================================ history

def quarter_end(P: pd.DataFrame) -> pd.DataFrame:
    """Quarter-end CPI y/y, unemployment, IP level, fed funds and 10-year yield from the monthly panel."""
    M = P[P.index.month.isin((3, 6, 9, 12))][["s_cpi_yoy", "lvl_unrate", "lvl_ip", "lvl_ff", "lvl_y10"]].copy()
    M.index = M.index.to_period("Q")
    return M


def _runs(mask) -> list[tuple[int, int]]:
    m = np.asarray(mask, bool)
    e = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return list(zip(np.flatnonzero(e == 1).tolist(), (np.flatnonzero(e == -1) - 1).tolist()))


def _pearson_lag(w: np.ndarray, lag: int = 4) -> float:
    a, b = w[:-lag], w[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _slope(y, x) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    den = float((x * x).sum())
    return float((x * y).sum() / den) if den > 1e-12 else float("nan")


def _bounce(lip: np.ndarray, start: int) -> float:
    """Coefficient on the drawdown of y = 100 (lip_{t+4} - lip_t) on [1, dd_t, g1_t], t >= start (j08)."""
    rows = []
    for t in range(max(start, 8), len(lip) - 4):
        rows.append((100 * (lip[t + 4] - lip[t]), 100 * (lip[t] - lip[t - 7:t + 1].max()), 400 * (lip[t] - lip[t - 1])))
    if len(rows) < 8:
        return float("nan")
    R = np.array(rows)
    if np.std(R[:, 1]) < 1e-9:
        return float("nan")
    X = np.column_stack([np.ones(len(R)), R[:, 1], R[:, 2]])
    return float(np.linalg.lstsq(X, R[:, 0], rcond=None)[0][1])


def history(Q: pd.DataFrame, P: pd.DataFrame, end: str = "2026Q3") -> dict:
    """History values of F2-F6 on 1973 .. end (the bounce-back from 1960Q2), as the design and the critic define
    them; the 13-year statistics on non-overlapping windows."""
    M = quarter_end(P).loc[:end]
    cpi = M["s_cpi_yoy"].loc["1973Q1":]
    x = cpi.to_numpy(float)
    win = [x[a:a + cq.TF_QUARTERS] for a in range(0, len(x) - cq.TF_QUARTERS + 1, cq.TF_QUARTERS)]
    spells = [b - a + 1 for a, b in _runs(x > 4)]
    H = M.loc["1960Q2":]
    r = RG.classify(H["s_cpi_yoy"].to_numpy(float), H["lvl_unrate"].to_numpy(float), H["lvl_ip"].to_numpy(float),
                    params=QP)
    con = np.asarray(r["contraction"], bool)
    in73 = H.index >= pd.Period("1973Q1", "Q")
    eps = [(str(H.index[a]), b - a + 1) for a, b in _runs(con) if in73[a]]
    lip = np.log(H["lvl_ip"].to_numpy(float))
    ex = (H.index >= pd.Period("2019Q3", "Q")) & (H.index <= pd.Period("2021Q3", "Q"))
    rows = []
    for t in range(8, len(lip) - 4):
        if ex[t]:
            continue
        rows.append((100 * (lip[t + 4] - lip[t]), 100 * (lip[t] - lip[t - 7:t + 1].max()), 400 * (lip[t] - lip[t - 1])))
    R = np.array(rows)
    bdd = float(np.linalg.lstsq(np.column_stack([np.ones(len(R)), R[:, 1], R[:, 2]]), R[:, 0], rcond=None)[0][1])
    L = M.loc["1973Q1":, ["lvl_ff", "lvl_unrate"]].copy()
    L["ip"] = 4 * Q["d_ip"].reindex(L.index)
    L = L[~((L.index >= pd.Period("2020Q1", "Q")) & (L.index <= pd.Period("2021Q2", "Q")))]
    d = L.diff().dropna()
    lev = L.loc[d.index]
    Z = Q.loc["1985Q1":end]
    Z2 = Q.loc["1973Q1":end]
    sb = {}
    for k in (0, 1):
        s = Z2[Z2["s_core4"] == k]
        sb[str(k)] = float(s[["y_us_equity", "y_treasury_10y"]].corr().iloc[0, 1])
    return {"cpi_gt4": float((x > 4).mean()), "n_quarters": int(len(x)),
            "cpi_acf4_windows": [_pearson_lag(w - w.mean()) for w in win],
            "cpi_acf4_window_starts": [str(cpi.index[a]) for a in range(0, len(x) - cq.TF_QUARTERS + 1,
                                                                            cq.TF_QUARTERS)],
            "cpi_gt4_windows": [float((w > 4).mean()) for w in win],
            "high_infl_spell_q": float(np.mean(spells)), "spells": spells,
            "contractions": eps, "contractions_per_15y": len(eps) / (in73.sum() / 60.0),
            "contraction_mean_q": float(np.mean([e[1] for e in eps])),
            "contractions_gt6q": [int(sum(e[1] > 6 for e in eps)), len(eps)],
            "bounce_dd": bdd, "bounce_n": int(len(R)),
            "dff_on_du": _slope(d["lvl_ff"], d["lvl_unrate"]), "okun": _slope(d["lvl_unrate"], lev["ip"]),
            "comovement_n": int(len(d)),
            "corr_dgz_useq": float(Z[["d_gz", "y_us_equity"]].corr().iloc[0, 1]),
            "corr_dspread_useq": float(Z[["d_spread", "y_us_equity"]].corr().iloc[0, 1]), "corr_n": int(len(Z)),
            "stock_bond_by_core4": sb, "unrate_min": float(M["lvl_unrate"].loc["1973Q1":].min())}


# ================================================================ per-path statistics

def _asset_paths(al, res: F.SimResult, init: F.InitState, Q: pd.DataFrame, origin, n: int, seed: int):
    """us_equity and Treasury quarterly log price returns through the asset layer (L0 + identities + residuals)."""
    H = res.info["H"]
    q0 = pd.Period(origin, "Q")
    x = res.x
    core4_0 = float(Q.at[q0, "s_core4"]) if q0 in Q.index and np.isfinite(Q.at[q0, "s_core4"]) else 0.0
    c4 = res.levels.get("core4")
    s_core4 = np.concatenate([np.full((n, 1), core4_0), c4[:, :-1].astype(float)], axis=1) if c4 is not None \
        else np.full((n, H), core4_0)
    y10 = res.levels["y10"]
    y10_0 = float(init.z[1][F.IDX["ry10"]] + init.pi0)
    s_y10 = np.concatenate([np.full((n, 1), y10_0), y10[:, :-1]], axis=1)
    bill = float((Q["lvl_tb3"] - Q["lvl_ff"]).loc[:q0 - 1].mean())
    ff = res.levels["ff"]
    s_ff = np.concatenate([np.full((n, 1), init.ff0), ff[:, :-1]], axis=1)
    frame = {f: x[f] for f in ("d_oil", "d_ip", "d_cpi_goods", "d_cpi_serv", "d_ff", "d_y10", "d_spread", "d_usd",
                               "d_unrate", "d_gz", "d_tp10", "d_metals_oil", "d_cpi", "d_rgdp") if f in x}
    frame.update({"s_core4": s_core4, "s_y10": s_y10, "s_tb3": s_ff + bill, "lvl_tb3": ff + bill})
    with np.errstate(invalid="ignore", over="ignore"):
        sp = al.simulate(frame, al.start_state(Q, q0), n_paths=n, seed=seed, params=False)
    return sp.asset("us_equity"), sp.asset("treasury_10y")


def path_stats(res: F.SimResult, init: F.InitState, Q: pd.DataFrame, P: pd.DataFrame, origin,
               equity=None) -> dict:
    """Per-path statistics of F2-F6 on simulated paths (n_paths,) arrays, plus pooled bounds."""
    q0 = pd.Period(origin, "Q")
    n, H = res.levels["ff"].shape
    cpi = res.levels["cpi_yoy"]
    u = res.levels["unrate"]
    ff = res.levels["ff"]
    ip = res.core[:, :, F.IDX["ip"]]
    out = {}
    out["cpi_gt4"] = (cpi > 4).mean(axis=1)
    out["cpi_acf4"] = np.array([_pearson_lag(c - c.mean()) for c in cpi])
    sp = []
    for c in cpi:
        r = [b - a + 1 for a, b in _runs(c > 4)]
        sp.append(np.mean(r) if r else np.nan)
    out["high_infl_spell_q"] = np.array(sp)
    Mh = quarter_end(P).loc["1960Q2":q0 - 1]
    k = len(Mh)
    ip_lvl0 = float(Mh["lvl_ip"].iloc[-1])
    lip_sim = math.log(ip_lvl0) + np.cumsum(ip / 400.0, axis=1)
    full = RG.classify(np.concatenate([np.broadcast_to(Mh["s_cpi_yoy"].to_numpy(float), (n, k)), cpi], 1),
                       np.concatenate([np.broadcast_to(Mh["lvl_unrate"].to_numpy(float), (n, k)), u], 1),
                       np.concatenate([np.broadcast_to(Mh["lvl_ip"].to_numpy(float), (n, k)), np.exp(lip_sim)], 1),
                       params=QP)
    con = np.asarray(full["contraction"], bool)[:, k:]
    cnt, mlen, lens = np.zeros(n), np.full(n, np.nan), []
    for i in range(n):
        L = [b - a + 1 for a, b in _runs(con[i])]
        cnt[i] = len(L)
        if L:
            mlen[i] = float(np.mean(L))
            lens += L
    out["contractions_per_15y"] = cnt * 60.0 / H
    out["contraction_mean_q"] = mlen
    out["contraction_lengths"] = np.array(lens)
    out["contraction_share_of_quarters"] = float(con.mean())
    lh = np.log(Mh["lvl_ip"].to_numpy(float))[-8:]
    out["bounce_dd"] = np.array([_bounce(np.concatenate([lh, lip_sim[i]]), 8) for i in range(n)])
    du = res.x["d_unrate"]
    dff = res.x["d_ff"]
    out["dff_on_du"] = np.array([_slope(dff[i], du[i]) for i in range(n)])
    out["okun"] = np.array([_slope(du[i], ip[i]) for i in range(n)])
    if equity is not None:
        eq, tsy = equity
        out["corr_dgz_useq"] = np.array([np.corrcoef(res.x["d_gz"][i], eq[i])[0, 1] for i in range(n)])
        out["corr_dspread_useq"] = np.array([np.corrcoef(res.x["d_spread"][i], eq[i])[0, 1] for i in range(n)])
        c4 = res.levels.get("core4")
        if c4 is not None:
            q0c4 = float(Q.at[q0, "s_core4"]) if q0 in Q.index else 0.0
            st = np.concatenate([np.full((n, 1), q0c4 > 0.5), c4[:, :-1]], axis=1)
            for kk in (0, 1):
                vals = []
                for i in range(n):
                    m = st[i] == bool(kk)
                    vals.append(np.corrcoef(eq[i, m], tsy[i, m])[0, 1] if m.sum() >= 8 else np.nan)
                out[f"stock_bond_core4_{kk}"] = np.array(vals)
    out["bounds"] = {"ff_min": float(ff.min()), "u_min": float(u.min()), "u_below3": float((u < 3).mean()),
                     "cpi_gt15": float((cpi > 15).mean()), "spread_below_floor": float(
                         (res.levels["spread"] < cq.SPREAD_FLOOR).mean()),
                     "gz_below_floor": float((res.levels["gz"] < cq.SPREAD_FLOOR).mean())}
    return out


def _band(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"p05": None, "p50": None, "p95": None, "n": 0}
    p = np.percentile(v, [5, 50, 95])
    return {"p05": float(p[0]), "p50": float(p[1]), "p95": float(p[2]), "n": int(len(v))}


def _within(h, band) -> bool | None:
    if band["p05"] is None or h is None or not np.isfinite(h):
        return None
    return bool(band["p05"] <= h <= band["p95"])


def ppc(stats: dict, hist: dict) -> dict:
    """F2-F6 for one fit: the simulated per-path [p05, p95] against history."""
    out = {}
    b = _band(stats["cpi_gt4"])
    out["F2_cpi_gt4"] = {"history": hist["cpi_gt4"], **b, "pass": _within(hist["cpi_gt4"], b)}
    b = _band(stats["cpi_acf4"])
    inside = [_within(h, b) for h in hist["cpi_acf4_windows"]]
    out["F2_cpi_acf4"] = {"history_windows": hist["cpi_acf4_windows"], **b, "windows_inside": inside,
                          "pass": all(bool(x) for x in inside)}
    b = _band(stats["high_infl_spell_q"])
    out["F2_spell"] = {"history": hist["high_infl_spell_q"], **b, "pass": _within(hist["high_infl_spell_q"], b)}
    b = _band(stats["contractions_per_15y"])
    out["F3_per_15y"] = {"history": hist["contractions_per_15y"], **b,
                         "pass": _within(hist["contractions_per_15y"], b)}
    b = _band(stats["contraction_mean_q"])
    out["F3_mean_q"] = {"history": hist["contraction_mean_q"], **b, "pass": _within(hist["contraction_mean_q"], b)}
    L = stats["contraction_lengths"]
    share = float((L > 6).mean()) if len(L) else 0.0
    out["F3_gt6q"] = {"history": hist["contractions_gt6q"], "share": share, "n_episodes": int(len(L)),
                      "share_of_quarters": stats["contraction_share_of_quarters"], "pass": bool(share <= 0.25)}
    b = _band(stats["bounce_dd"])
    out["F4_bounce"] = {"history": hist["bounce_dd"], **b, "pass": _within(hist["bounce_dd"], b)}
    for key in ("dff_on_du", "okun", "corr_dgz_useq", "corr_dspread_useq"):
        if key in stats:
            b = _band(stats[key])
            out[f"F5_{key}"] = {"history": hist[key], **b, "pass": _within(hist[key], b)}
    for kk in (0, 1):
        key = f"stock_bond_core4_{kk}"
        if key in stats:
            b = _band(stats[key])
            hv = hist["stock_bond_by_core4"][str(kk)]
            out[f"F5_{key}"] = {"history": hv, **b, "pass": _within(hv, b)}
    bd = stats["bounds"]
    out["F6_bounds"] = {**bd, "pass": bool(bd["ff_min"] >= cq.FF_FLOOR - 1e-12 and bd["u_min"] >= 2.5
                                            and bd["u_below3"] <= 0.05 and bd["cpi_gt15"] <= 0.005),
                        "spread_floor_transform_due": bool(max(bd["spread_below_floor"], bd["gz_below_floor"])
                                                           > cq.SPREAD_FLOOR_SHARE)}
    return out


# ================================================================ F1, F9, F10 (at one fit)

def f1_statements(model: F.FactorModel, Q: pd.DataFrame, end: str) -> list:
    """A QTable-like statement set (Appendix B's example plus every statement kind): soft values and sizes, hard
    values (raw policy, y/y, a satellite, an unchained unemployment change), an intervention and ranges."""
    st = qpanel.field_stats(Q, ["d_gpr", "d_cpi_goods", "d_gz", "d_spread", "d_rgdp"], end)
    sd = st["sd"].to_dict()
    fits = policy.fit(Q, end)
    usd = policy.shock_sd(fits)["u_ff"]
    return [S("d_oil", (1,), 26.0, sd=8.0, label="油价一个季度内上涨约三成"),
            S("x_oil_cause", (1,), 1.0, label="供给"),
            S("d_gpr", (1,), 2.0 * sd["d_gpr"], sd=0.5 * sd["d_gpr"], label="地缘风险 大幅"),
            S("d_cpi_goods", (1,), 1.0 * sd["d_cpi_goods"], sd=0.5 * sd["d_cpi_goods"], label="商品价格 明显"),
            S("d_gz", (1,), 0.5 * sd["d_gz"], sd=0.25 * sd["d_gz"], label="fc 收紧"),
            S("d_spread", (1,), 0.5 * sd["d_spread"], sd=0.25 * sd["d_spread"], label="fc 收紧"),
            S("d_ff", (2,), 0.5, label="加息 50 个基点"),
            S("d_y10", (2,), 0.3, label="10 年期上升 30 个基点"),
            S("u_ff", (3,), -usd, label="意外降息 1 sd"),
            S("cpi_yoy", (4,), 3.5, label="CPI 同比 3.5%"),
            S("d_unrate", (1, 2, 3, 4), lo=-0.2, hi=0.3, label="就业基本稳定"),
            S("d_rgdp", (5,), -2.0, label="GDP 年化 -2%"),
            S("d_unrate", (6,), 0.3, label="失业率上升 0.3（前一季未锚定）")]


def f1(model: F.FactorModel, init: F.InitState, Q: pd.DataFrame, n: int, seed: int) -> dict:
    stmts = f1_statements(model, Q, model.train_end)
    res = F.simulate(model, init, 8, n_paths=n, seed=seed, statements=stmts)
    for d in res.delivery:
        if "max_abs_error" in d:
            d["max_abs_error_sci"] = f"{d['max_abs_error']:.2e}"
    hard = [d for d in res.delivery if d["kind"] == "hard"]
    soft = [d for d in res.delivery if d["kind"] == "soft"]
    rng = [d for d in res.delivery if d["kind"] == "range"]
    hard_err = max(d["max_abs_error"] for d in hard)
    return {"n_paths": n, "statements": len(stmts), "hard_max_abs_error": hard_err,
            "hard_max_abs_error_sci": f"{hard_err:.2e}",          # results JSON rounds floats to 6 decimals
            "soft_share_within_1sd": [d["share_within_1sd"] for d in soft],
            "range_share_inside": [d["share_inside"] for d in rng], "delivery": res.delivery,
            "clipped": [c for c in res.notes if "used" in c], "rarity": {k: v for k, v in res.rarity.items()
                                                                         if k != "per_quarter"},
            "ranges": res.info.get("ranges"),
            "pass": bool(hard_err < 1e-8 and all(abs(s - 0.68) <= 0.05 for s in [d["share_within_1sd"] for d in soft])
                         and all(s >= 1.0 - 1e-12 for s in [d["share_inside"] for d in rng]))}


E0_GROWTH = ["d_rgdp", "d_capex", "d_cons_goods", "d_cons_serv", "d_resinv", "d_exports", "d_itinv"]


def f9(model: F.FactorModel, init: F.InitState, Q: pd.DataFrame, al, n: int, seed: int) -> dict:
    """The do-operator checks at one fit (paired runs on the same seed)."""
    fits = policy.fit(Q, model.train_end)
    usd = policy.shock_sd(fits)["u_ff"]
    H = 20
    kw = dict(n_paths=n, seed=seed, cycle=False, param_uncertainty=False)
    base = F.simulate(model, init, H, **kw)
    ease = F.simulate(model, init, H, statements=[S("u_ff", (1,), -usd)], **kw)
    cls = F.field_classes()
    e0 = [f for f in base.x if cls.get(f) == "E0"] + ["d_cpi"]
    same_q = max(float(np.max(np.abs(ease.x[f][:, 0] - base.x[f][:, 0]))) for f in e0)
    # sign checks on the paired mean responses
    dy2 = float(np.mean(ease.x["d_y2"][:, 0] - base.x["d_y2"][:, 0])) if "d_y2" in base.x else float("nan")
    dusd_raw = float(np.mean(ease.core[:, 0, F.IDX["usd"]] - base.core[:, 0, F.IDX["usd"]]))
    du = np.mean(ease.levels["unrate"] - base.levels["unrate"], axis=0)
    peak = int(np.argmin(du)) + 1                     # the easing lowers unemployment: its largest fall
    beta = al.beta_fields().loc["us_equity"]
    deq = float(sum(beta.get(f, 0.0) * np.mean(ease.x[f][:, 0] - base.x[f][:, 0]) for f in beta.index
                    if f in base.x and f != "d_tp10"))
    signs = {"equity_impact": deq, "d_y2_impact": dy2, "usd_impact": dusd_raw,
             "unrate_response": [float(v) for v in du], "unrate_peak_q": peak}
    signs_ok = bool(deq > 0 and dy2 < 0 and dusd_raw < 0 and 4 <= peak <= 8)
    # a stated raw cut with no recession stated imputes nothing: d_ip and d_unrate fixed at the no-cut means
    mu = F.simulate(model, init, 4, n_paths=400, seed=seed, cycle=False, gaussian=True, param_uncertainty=False,
                    saturate=False)
    norec = [S("d_ip", (h,), float(np.mean(mu.x["d_ip"][:, h - 1]))) for h in range(1, 5)] + \
            [S("d_unrate", (h,), float(np.mean(mu.x["d_unrate"][:, h - 1]))) for h in range(1, 5)]
    fsd = qpanel.field_stats(Q, E0_GROWTH, model.train_end)["sd"].to_dict()
    kw4 = dict(n_paths=n, seed=seed, cycle=False, param_uncertainty=False, saturate=False)
    a = F.simulate(model, init, 4, statements=norec, **kw4)
    b = F.simulate(model, init, 4, statements=norec + [S("d_ff", (1,), -usd)], **kw4)
    c = F.simulate(model, init, 4, statements=[S("d_ff", (1,), -usd)], **kw4)
    imp = {f: float(np.mean(b.x[f][:, 0] - a.x[f][:, 0]) / fsd[f]) for f in E0_GROWTH if f in a.x}
    imp_alone = {f: float(np.mean(c.x[f][:, 0] - base.x[f][:, 0]) / fsd[f]) for f in E0_GROWTH if f in a.x}
    nothing = bool(max(abs(v) for v in imp.values()) < 0.1)
    return {"u_ff_sd": usd, "same_quarter_E0_max_abs_change": same_q, "same_quarter_E0_max_abs_change_sci":
            f"{same_q:.2e}", "same_quarter_E0_unchanged": bool(same_q < 1e-9),
            "signs": signs, "signs_as_literature": signs_ok,
            "raw_cut_no_recession_shift_sd": imp, "raw_cut_alone_shift_sd": imp_alone, "imputes_nothing": nothing,
            "pass": bool(same_q < 1e-9 and signs_ok and nothing)}


def f10(model: F.FactorModel, init: F.InitState, n: int, seed: int) -> dict:
    """Path mode with statements only in quarter 1 against step mode: identical draws on the same seed, and a
    two-sample KS test per field on independent seeds (Bonferroni over the fields)."""
    st = [S("d_ip", (1,), 0.5), S("d_unrate", (1,), 0.1), S("d_ff", (1,), -0.25), S("d_gz", (1,), 0.2, sd=0.1)]
    p1 = F.simulate(model, init, 8, n_paths=n, seed=seed, statements=st)
    s1 = F.step(model, init, n_paths=n, seed=seed, statements=st)
    fields = sorted(set(p1.x) & set(s1.x))
    same = max(float(np.max(np.abs(p1.x[f][:, 0] - s1.x[f][:, 0]))) for f in fields)
    s2 = F.step(model, init, n_paths=n, seed=seed + 1, statements=st)
    pv = {}
    for f in fields:
        a, b = p1.x[f][:, 0], s2.x[f][:, 0]
        if np.std(a) < 1e-12 and np.std(b) < 1e-12:
            continue
        pv[f] = float(ks_2samp(a, b).pvalue)
    mn = min(pv.values()) if pv else 1.0
    return {"n_draws": n, "same_seed_max_abs_diff": same, "same_seed_max_abs_diff_sci": f"{same:.2e}",
            "ks_min_p": mn, "ks_fields": len(pv),
            "ks_below_001": [f for f, p in pv.items() if p < 0.01],
            "pass": bool(same < 1e-10 and mn * max(len(pv), 1) > 0.01)}


# ================================================================ F7, F8 (rolling)

F7_FIELDS = ["ps", "ff", "y10", "sp", "gz", "ur", "usd"]
F8_FIELDS = ["d_oil", "d_ip", "d_cpi_goods", "d_cpi_serv", "d_ff", "d_y10", "d_spread", "d_usd", "d_unrate", "d_gz"]


def _window_changes(model: F.FactorModel, init: F.InitState, mean: np.ndarray) -> dict:
    H = mean.shape[0]
    pc = mean[:, F.IDX["pc"]]
    pi = init.pi0
    for h in range(H):
        pi = pi + F.ALPHA * (pc[h] - pi)
    z1 = init.z[1]
    ffH = max(mean[-1, F.IDX["rff"]] + pi, cq.FF_FLOOR)

    def lvl(v, x):
        return F.EXP_SHIFT[v] + math.exp(x) if v in F.EXP_SHIFT else x
    return {"ps": float(np.mean(mean[:, F.IDX["ps"]])), "ff": ffH - init.ff0,
            "y10": mean[-1, F.IDX["ry10"]] + pi - (z1[F.IDX["ry10"]] + init.pi0),
            "sp": lvl("sp", mean[-1, F.IDX["sp"]]) - lvl("sp", z1[F.IDX["sp"]]),
            "gz": lvl("gz", mean[-1, F.IDX["gz"]]) - lvl("gz", z1[F.IDX["gz"]]),
            "ur": lvl("lu", mean[-1, F.IDX["lu"]]) - init.u0, "usd": 0.0}


def _realized(Q: pd.DataFrame, q0, H: int) -> dict:
    W = Q.loc[q0:q0 + H - 1]
    last = q0 - 1
    return {"ps": float(W["d_cpi_serv"].mean()), "ff": float(W["lvl_ff"].iloc[-1] - Q.at[last, "lvl_ff"]),
            "y10": float(W["lvl_y10"].iloc[-1] - Q.at[last, "lvl_y10"]),
            "sp": float(W["lvl_spread"].iloc[-1] - Q.at[last, "lvl_spread"]),
            "gz": float(W["lvl_gz"].iloc[-1] - Q.at[last, "lvl_gz"]),
            "ur": float(W["lvl_unrate"].iloc[-1] - Q.at[last, "lvl_unrate"]),
            "usd": float(W["lvl_usd"].iloc[-1] - Q.at[last, "lvl_usd"])}


def rolling_fits(Q: pd.DataFrame, P: pd.DataFrame, first: str, last: str, seed: int) -> dict:
    """Core-only fits at every Q4 from `first` to `last` (train-end anchors): the January refits of the rolling
    protocol (data to the previous December)."""
    out = {}
    for y in range(pd.Period(first, "Q").year, pd.Period(last, "Q").year + 1):
        te = f"{y}Q4"
        out[te] = F.fit(Q, te, anchors="train_end", P=P, seed=seed, satellites=False)
    return out


def f7(Q: pd.DataFrame, fits: dict, horizons=(4, 12)) -> dict:
    """Fill-in skill with CPI, IP and oil stated (hard, each quarter): R² of the unstated factors' window changes
    against no change (services CPI: against its mean), rolling origins TF_ROLLING."""
    lo, hi = (pd.Period(q, "Q") for q in cq.TF_ROLLING)
    out = {}
    for H in horizons:
        pred, real, orig = {k: [] for k in F7_FIELDS}, {k: [] for k in F7_FIELDS}, []
        q = lo
        while q + H - 1 <= hi:
            m = fits[f"{q.year - 1}Q4"]
            init = m.initial_state(Q, q)
            W = Q.loc[q:q + H - 1]
            st = [S("pc", (h + 1,), float(W["d_cpi"].iloc[h])) for h in range(H)] + \
                 [S("d_ip", (h + 1,), float(W["d_ip"].iloc[h])) for h in range(H)] + \
                 [S("real_oil", (h + 1,), float(W["lvl_oil_real"].iloc[h])) for h in range(H)]
            mean, _ = F.conditional_moments(m, init, H, st)
            p = _window_changes(m, init, mean)
            r = _realized(Q, q, H)
            for k in F7_FIELDS:
                pred[k].append(p[k])
                real[k].append(r[k])
            orig.append(str(q))
            q += 1
        res = {}
        for k in F7_FIELDS:
            y, p = np.array(real[k]), np.array(pred[k])
            sst = ((y - y.mean()) ** 2).sum() if k == "ps" else (y ** 2).sum()
            res[k] = float(1.0 - ((y - p) ** 2).sum() / sst)
        eng = cq.TF_ENGINE_FILL[H]
        checks = {"ur": res["ur"] >= eng["ur"], "sp": res["sp"] >= eng["sp"]}
        for k in ("ff", "y10", "gz", "ps"):
            checks[k] = res[k] >= -0.05
        checks["usd_held"] = abs(res["usd"]) < 1e-12
        out[str(H)] = {"r2": res, "engine": eng, "checks": checks, "n_origins": len(orig),
                       "n_indep": len(orig) // H, "first": orig[0], "last": orig[-1], "pass": all(checks.values())}
    out["pass"] = all(out[str(H)]["pass"] for H in horizons)
    return out


def f8(Q: pd.DataFrame, fits: dict, n: int, seed: int) -> dict:
    """80% coverage of 4-quarter factor changes on non-overlapping windows from TF_COVERAGE_FROM (unconditional
    simulation from each January refit)."""
    q = pd.Period(cq.TF_COVERAGE_FROM, "Q")
    last = pd.Period(cq.TF_ROLLING[1], "Q")
    hits = {f: [] for f in F8_FIELDS}
    wins = []
    while q + 3 <= last:
        m = fits[f"{q.year - 1}Q4"]
        init = m.initial_state(Q, q)
        res = F.simulate(m, init, 4, n_paths=n, seed=seed + q.year)
        W = Q.loc[q:q + 3]
        for f in F8_FIELDS:
            sim = res.x[f].sum(axis=1)
            lo, hi = np.percentile(sim, [10, 90])
            y = float(W[f].sum())
            if np.isfinite(y):
                hits[f].append(bool(lo <= y <= hi))
        wins.append(str(q))
        q += 4
    per = {f: float(np.mean(v)) for f, v in hits.items()}
    pooled = float(np.mean([h for v in hits.values() for h in v]))
    return {"windows": len(wins), "first": wins[0], "last": wins[-1], "coverage": pooled, "per_field": per,
            "pass": bool(0.72 <= pooled <= 0.88)}


# ================================================================ T-C

def tc_masks(model: F.FactorModel, rng: np.random.Generator) -> dict:
    cls = F.field_classes()
    xs = [f for f in F.CORE_FIELDS] + list(model.sats.fields)
    card = F._card()
    fc = [r["id"] for r in card["rows"] if r.get("group") == "x" and r.get("fc_role") in ("required_one", "optional")]
    return {"b": [f for f in xs if cls.get(f) in ("E0", "E0p")], "c": [f for f in xs if f in fc], "all": xs}


def _crps(draws: np.ndarray, y: float) -> float:
    x = np.sort(draws)
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - (2.0 / (n * n)) * np.sum((2 * i - n - 1) * x) / 2.0)


def _step_draws(model: F.FactorModel, init: F.InitState, stmts, w: float, regime, rng, n: int) -> dict:
    mean, cov = F.conditional_moments(model, init, 1, stmts, completion_w=w, regime=regime)
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(cov)))
    Z = mean.reshape(1, -1) + rng.standard_normal((n, len(cov))) @ L.T
    Y = Z.reshape(n, 1, model.n)
    x, lv, _ = F.outputs(model, init, Y, np.zeros_like(Y), np.full(n, -1), 1, usd_hold=not any(
        s.var == "d_usd" for s in stmts))
    return {f: v[:, 0] for f, v in x.items()}


def _stated_values(Q: pd.DataFrame, q, fields: list[str]) -> list:
    out = []
    zlb = float(Q.at[q, "lvl_ff"]) < cq.ZLB_FF
    for f in fields:
        v = Q.at[q, f] if f in Q.columns else np.nan
        if not np.isfinite(v):
            continue
        if f == "d_ff" and (zlb or float(Q.at[q - 1, "lvl_ff"]) < cq.ZLB_FF):
            continue                                      # the shadow rate is not observed at the floor
        out.append(S(f, (1,), float(v)))
    return out


def tc_block(model: F.FactorModel, Q: pd.DataFrame, quarters, w_grid, regimes=(False, True), seed: int = 0,
             n: int = cq.TC_DRAWS) -> list[dict]:
    """Rows (origin, pattern, w, regime, field, y, mean, mean_uncond, crps, crps_uncond, covered) for step-mode
    completion at the given origins."""
    rows = []
    cls = F.field_classes()
    for q in quarters:
        q = pd.Period(q, "Q")
        init = model.initial_state(Q, q)
        rng = np.random.default_rng([seed, q.year, q.quarter])
        masks = tc_masks(model, rng)
        half = [f for f in masks["all"] if rng.random() < 0.5]
        pats = {"a": half, "b": masks["b"], "c": masks["c"]}
        zlb_now = bool(Q.at[q, "s_zlb"] > 0.5) if np.isfinite(Q.at[q, "s_zlb"]) else False
        core4 = bool(Q.at[q, "s_core4"] > 0.5) if np.isfinite(Q.at[q, "s_core4"]) else False
        pstm = {}
        for pat, fields in pats.items():
            stmts = _stated_values(Q, q, fields)
            vals, _ = F.saturation(model, init, 1, stmts, seed=seed)           # the model's +-4 sd saturation
            pstm[pat] = [F.Statement(s.var, s.quarters, v, agg=s.agg) for s, v in zip(stmts, vals)]
        for rc in regimes:
            reg = ("zlb" if zlb_now else ("core4" if core4 else None)) if rc else None
            unc = _step_draws(model, init, [], 1.0, reg, np.random.default_rng([seed, q.year, q.quarter, 1]), n)
            for pat in pats:
                stmts = pstm[pat]
                stated = {s.var for s in stmts}
                for w in (w_grid if pat == "b" else (1.0,)):
                    dr = _step_draws(model, init, stmts, w, reg, np.random.default_rng([seed, q.year, q.quarter, 2]),
                                     n)
                    for f in masks["all"]:
                        if f in stated or f not in dr or f not in Q.columns:
                            continue
                        y = Q.at[q, f]
                        if not np.isfinite(y):
                            continue
                        d, du = dr[f], unc[f]
                        lo, hi = np.percentile(d, [10, 90])
                        rows.append({"origin": str(q), "pattern": pat, "w": w, "regime_cov": rc, "zlb": zlb_now,
                                     "field": f, "cls": cls.get(f), "y": float(y), "mean": float(np.mean(d)),
                                     "mean_u": float(np.mean(du)), "crps": _crps(d, float(y)),
                                     "crps_u": _crps(du, float(y)), "cov": bool(lo <= y <= hi)})
    return rows


def _r2(g: pd.DataFrame) -> float:
    sse = float(((g["y"] - g["mean"]) ** 2).sum())
    sst = float(((g["y"] - g["mean_u"]) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def tc(Q: pd.DataFrame, P: pd.DataFrame, seed: int, n: int = cq.TC_DRAWS, quick: bool = False) -> dict:
    """T-C: selection of w and the regime covariance on the rolling validation origins TC_VALIDATION (yearly refits
    each January on data to the previous December), one score on the test (fit at TC_TEST_END, 2015Q1 ..
    TC_TEST_LAST)."""
    grid = cq.COMPLETION_W_GRID
    val_rows = []
    first, last = (pd.Period(q, "Q") for q in cq.TC_VALIDATION)
    for year in range(first.year, last.year + 1, 5 if quick else 1):
        end = f"{year - 1}Q4"                            # the January refit on data to the previous December
        m = F.fit(Q, end, anchors="train_end", P=P, seed=seed)
        qs = pd.period_range(f"{year}Q1", f"{year}Q4", freq="Q")
        val_rows += tc_block(m, Q, qs[::2] if quick else qs, grid, seed=seed, n=n)
    V = pd.DataFrame(val_rows)
    V["rel"] = V["crps"] / V["crps_u"].where(V["crps_u"] > 0)
    e1 = V["cls"].isin(F.E1_CLASSES)
    sel = V[(V["pattern"] == "b") & e1 & (~V["regime_cov"])].groupby("w")["rel"].mean()
    w_star = float(sel.idxmin())
    # regime covariance: E1 completion CRPS in ZLB quarters (pattern b at w*) and the loss elsewhere
    def rel(df):
        return float(df["rel"].mean()) if len(df) else float("nan")
    base_b = V[(V["pattern"] == "b") & (V["w"] == w_star)]
    z_off = rel(base_b[base_b["zlb"] & e1.loc[base_b.index] & ~base_b["regime_cov"]])
    z_on = rel(base_b[base_b["zlb"] & e1.loc[base_b.index] & base_b["regime_cov"]])
    others = V[((V["pattern"] != "b") | (V["w"] == w_star))]
    loss = {}
    for (pat, z), g in others.groupby(["pattern", "zlb"]):
        if pat == "b" and z:
            continue
        off = rel(g[~g["regime_cov"]])
        on = rel(g[g["regime_cov"]])
        loss[f"{pat}_{'zlb' if z else 'normal'}"] = on - off
    adopt = bool(np.isfinite(z_on) and z_on < z_off and all(v <= 0.01 for v in loss.values() if np.isfinite(v)))
    m = F.fit(Q, cq.TC_TEST_END, anchors="train_end", P=P, seed=seed)
    qs = pd.period_range(pd.Period(cq.TC_TEST_END, "Q") + 1, pd.Period(cq.TC_TEST_LAST, "Q"), freq="Q")
    if quick:
        qs = qs[::6]
    T = pd.DataFrame(tc_block(m, Q, qs, (w_star,), regimes=(adopt,), seed=seed, n=n))
    metrics = {}
    for pat, g in T.groupby("pattern"):
        per = {}
        for f, gf in g.groupby("field"):
            per[f] = {"cls": gf["cls"].iloc[0], "n": int(len(gf)), "r2_vs_uncond": _r2(gf),
                      "crps": float(gf["crps"].mean()), "crps_uncond": float(gf["crps_u"].mean()),
                      "cov80": float(gf["cov"].mean())}
        metrics[pat] = per
    gate_f = {f: v["r2_vs_uncond"] for f, v in metrics.get("b", {}).items() if v["cls"] == "E1"}
    gate = bool(all(v >= -0.02 for v in gate_f.values() if np.isfinite(v)))
    vsum = {f"w={w:g}": float(v) for w, v in sel.items()}
    return {"validation": "yearly refits (each Q4 to the previous year) scoring the next four quarters, "
                          f"{cq.TC_VALIDATION[0]}..{cq.TC_VALIDATION[1]}", "test_end": cq.TC_TEST_END,
            "n_validation_origins": int(V["origin"].nunique()), "n_test_origins": int(T["origin"].nunique()),
            "selection": {"criterion": "pattern (b), E1/E1c/E2 fields, mean CRPS relative to the unconditional",
                          "by_w": vsum, "completion_w": w_star,
                          "regime_cov": {"zlb_E1_rel_crps_off": z_off, "zlb_E1_rel_crps_on": z_on,
                                         "loss_elsewhere": loss, "adopted": adopt}},
            "test": metrics, "gate": {"rule": "pattern (b): every E1 field's completion R² >= -0.02",
                                      "E1_r2": gate_f, "pass": gate},
            "pass": gate}


# ================================================================ the run

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def tf_fit(Q, P, end: str, seed: int, n: int, al_cls=None, alt: bool = True) -> dict:
    """F1-F6, F9, F10 at one fit end (forward anchors at the last end, train-end anchors otherwise)."""
    t0 = time.time()
    mode = "forward" if end == cq.TF_ENDS[-1] else "train_end"
    m = F.fit(Q, end, anchors=mode, P=P, seed=seed)
    origin = str(pd.Period(end, "Q") + 1)
    init = m.initial_state(Q, origin)
    res = F.simulate(m, init, cq.TF_QUARTERS, n_paths=n, seed=seed)
    al = None
    eq = None
    if al_cls is not None:
        try:
            al = al_cls.fit(end, panel=P, Q=Q, n_boot=0, seed=seed)
            eq = _asset_paths(al, res, init, Q, origin, n, seed)
        except Exception as exc:                 # noqa: BLE001  (the asset layer is P2's: report, do not fail)
            eq = None
            al = None
            err = f"{type(exc).__name__}: {exc}"
        else:
            err = None
    else:
        err = "asset layer not available"
    hist = HISTORY_CACHE.setdefault("h", history(Q, P))
    stats = path_stats(res, init, Q, P, origin, eq)
    out = {"end": end, "anchors": mode, "origin": origin, "fit": _fit_summary(m), "ppc": ppc(stats, hist),
           "asset_layer": "L0 (qmodel.assets)" if al is not None else err}
    if alt:
        res8 = F.simulate(m, init, cq.TF_QUARTERS, n_paths=n, seed=seed, cycle=False, block=cq.CYCLE_ALT_BLOCK)
        eq8 = _asset_paths(al, res8, init, Q, origin, n, seed) if al is not None else None
        out["ppc_alt_innov8"] = ppc(path_stats(res8, init, Q, P, origin, eq8), hist)
    if end == cq.TF_ENDS[-1]:
        # the pre-registered shadow-rate alternative (§5.4: censored at the bound instead of missing)
        mc = F.fit(Q, end, anchors=mode, P=P, seed=seed, shadow="censored")
        ic = mc.initial_state(Q, origin)
        rc = F.simulate(mc, ic, cq.TF_QUARTERS, n_paths=n, seed=seed)
        eqc = _asset_paths(al, rc, ic, Q, origin, n, seed) if al is not None else None
        out["ppc_alt_shadow_censored"] = ppc(path_stats(rc, ic, Q, P, origin, eqc), hist)
        out["fit_alt_shadow_censored"] = _fit_summary(mc)
        out["F1"] = f1(m, init, Q, n, seed)
        if al is not None:
            out["F9"] = f9(m, init, Q, al, n, seed)
        out["F10"] = f10(m, init, 10000, seed)
    out["seconds"] = round(time.time() - t0, 1)
    return out


HISTORY_CACHE: dict = {}


def _fit_summary(m: F.FactorModel) -> dict:
    keep = ("sample", "T", "lambda", "nu", "logml", "rejected_draws", "rejection_rate", "max_root_mean",
            "steady_state", "zlb_quarters", "cpi_weights", "satellites", "satellites_excluded", "sat_cov",
            "smallest_weights", "anchors")
    out = {k: m.info.get(k) for k in keep}
    sh = pd.Series(m.core.Y[:, F.IDX["rff"]] + 0.0, index=m.core.dates)
    miss = m.core.missing[:, F.IDX["rff"]]
    out["shadow_rff_zlb"] = {str(q): float(v) for q, v in sh[miss].items()}
    Qp = qpanel.load_panel()
    nominal = sh[miss] + Qp["lvl_infl_trend"].reindex(sh.index[miss]).to_numpy(float)
    out["shadow_nominal_above_zlb"] = {str(q): float(v) for q, v in nominal[nominal > cq.ZLB_FF].items()}
    return out


def _summary_line(tf: dict) -> dict:
    """{check: pass} over every fit and the rolling checks."""
    lines = {}
    for end, r in tf["fits"].items():
        for k, v in r["ppc"].items():
            lines[f"{end}:{k}"] = v.get("pass")
        for k in ("F1", "F9", "F10"):
            if k in r:
                lines[f"{end}:{k}"] = r[k].get("pass")
    for k in ("F7", "F8"):
        if k in tf:
            lines[k] = tf[k].get("pass")
    return lines


def _cycle_choice(tf: dict) -> dict:
    """§5.6: the C' module against its pre-registered alternative (8-quarter innovation blocks, no module): the
    number of F2-F6 checks each passes per fit; the better posterior predictive fit wins (ties: C', the default)."""
    per, tot = {}, {"cprime": 0, "innov8": 0}
    for end, r in tf["fits"].items():
        if "ppc_alt_innov8" not in r:
            continue
        a = sum(bool(v.get("pass")) for v in r["ppc"].values())
        b = sum(bool(v.get("pass")) for v in r["ppc_alt_innov8"].values())
        per[end] = {"cprime": a, "innov8": b, "checks": len(r["ppc"])}
        tot["cprime"] += a
        tot["innov8"] += b
    return {"per_fit": per, "total": tot, "winner": "cprime" if tot["cprime"] >= tot["innov8"] else "innov8"}


def run(ctx=None, quick: bool = False, seed: int = cq.TF_SEED, write: bool = True) -> dict:
    """T-F and T-C; writes the factor-layer section of data/results/qmodel.json (other sections are kept)."""
    from ..data import panel as monthly_panel
    t0 = time.time()
    P = monthly_panel.load_panel()
    Q = qpanel.load_panel()
    try:
        from .assets import AssetLayer
    except ImportError:
        AssetLayer = None
    n = 400 if quick else cq.TF_PATHS
    ends = cq.TF_ENDS[-1:] if quick else cq.TF_ENDS
    tf = {"protocol": __doc__.split("T-C (completion)")[0].strip(), "history": history(Q, P), "fits": {}}
    for end in ends:
        tf["fits"][end] = tf_fit(Q, P, end, seed, n, AssetLayer, alt=not quick)
    roll = rolling_fits(Q, P, "1984Q4" if not quick else "2014Q4", "2025Q4", seed)
    if not quick:
        tf["F7"] = f7(Q, roll)
        tf["F8"] = f8(Q, roll, 1000, seed)
    tf["summary"] = _summary_line(tf)
    tf["cycle_choice"] = _cycle_choice(tf)
    out = {"version": F.VERSION, "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "quick": quick,
           "seed": seed, "T_F": tf, "T_C": tc(Q, P, seed, n=200 if quick else cq.TC_DRAWS, quick=quick),
           "policy": {end: policy.summary(policy.fit(Q, end)) for end in cq.TF_ENDS},
           "notes": ["只给出通过/未通过，不作为关卡（§14 P3）；冻结的预注册记录在 P5。",
                     "历史值按设计与批评的定义重算；13 年统计用不重叠窗口。"]}
    out["seconds"] = round(time.time() - t0, 1)
    out = _jsonable(out)
    if write:
        path = cq.QMODEL_RESULTS_PATH
        doc = jsonio.load(path) if path.exists() else {}
        doc["factor_layer"] = out
        jsonio.dump(doc, path)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="T-F and T-C of the factor layer (SIM_DESIGN §5.8)")
    ap.add_argument("--quick", action="store_true", help="one fit, fewer paths and origins (smoke run)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    out = run(quick=a.quick, write=not a.no_write)
    print(json.dumps(out["T_F"]["summary"], ensure_ascii=False, indent=1))
    print("T-C gate:", out["T_C"]["gate"]["pass"], "w*", out["T_C"]["selection"]["completion_w"],
          "regime cov adopted:", out["T_C"]["selection"]["regime_cov"]["adopted"], "seconds", out["seconds"])


if __name__ == "__main__":
    main()
