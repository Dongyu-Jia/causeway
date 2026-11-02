"""Macro regimes (宏观体制): what the labels mean in history and in the engine, for results/regimes.json.

Every month since 1948 gets one label from CPI y/y, the unemployment rate and industrial production
(engine/regimes.py, which also holds the conventions every consumer shares: the "1973" window = the month
database, geometric real returns, runs touching the window, the per-cell bootstrap generator). This module
measures what the labels say and what they do not:

  definitions   parameters, labels, flags, the rules in Chinese, the 3 x 3 grid, the four words the user
                asked about mapped to labels (numbers computed here), the conventions, interpolated months
  nowcast       today's real-time label and the distance to every threshold (regimes.nowcast)
  timeline      labels, flags, indicators, runs and the NBER bands of every month since 1948
  counts        months, shares and runs per label and flag (1948 and 1973 windows, ex post and real time)
  validation    the rule and its alternatives scored against NBER dates, the IP legs, one-parameter
                sensitivity, the out-of-sample grid (one recession held out at a time, split sample), the
                data-revision study and the 2022-24 near miss
  assets        same-month real returns per label and flag, episode-bootstrap intervals, the difference to
                the other months of the window, stock-bond correlations
  episodes      every contraction as a real equity path: the fall to the equity low, then the rebound
  factors       the factor changes and state levels that set each regime apart (z against the 1973 window)
  transitions   monthly persistence, durations, real-time label -> ex-post label 12 months on, and
                walk-forwards with ex-post targets computed on data through each origin
  forward       the year after a real-time label: windows t+2..t+13 (the label of month t is known during
                t+1; t+1..t+12 is kept as a sensitivity), placebo by circular shifts, HAC t, halves, the
                joint placebo and multiple-testing counts, and the recession-onset mechanism
  eras          the regime mix of every config.ERAS era (and 1948-54)
  transmission  the engine's betas in high- vs low-inflation months (descriptive: the model has no regime
                term, its walk-forward failed)
  engine        what a 12-month phase can draw (support), the engine's unconditional reference run against
                history, stock-bond co-movement, and simulated studies of regime views

Thin statistics (fewer than config.REGIMES["min_episodes"] episodes) are marked and get no placebo p.
Nothing here changes a model component: the labels describe history and simulated paths, and scenarios
draw on them through ordinary views.

    python -m scenariolab.analysis.regimes        # writes results/regimes.json
"""
from __future__ import annotations

import itertools
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats as sps

from .. import config, jsonio
from ..engine import factor_model as FM
from ..engine import regimes as R
from ..engine.pooling import BLOCK, pool_phase, view_support

CFG = config.REGIMES
B = int(CFG["bootstrap"]["replicates"])           # bootstrap replicates (whole episodes)
CI = tuple(CFG["bootstrap"]["ci"])                 # 90% percentile interval
THIN = R.THIN                                      # fewer episodes: thin (†), no placebo p
HAC_MONTHLY = int(CFG["hac_lags"]["monthly"])
HAC_FORWARD = int(CFG["hac_lags"]["forward"])
LAG = int(CFG["forward_lag_months"])
MIN_SHIFT = int(CFG["placebo_min_shift"])
LONGEST = int(CFG["longest_contraction_months"])
FORWARD = {"lag1": (1 + LAG, 12 + LAG), "base": (1, 12)}   # months ahead of the origin, inclusive
HALVES = {"1973": (("1973_1999", None, "1999-12"), ("2000_end", "2000-01", None)),
          "1948": (("1948_1972", None, "1972-12"), ("1973_end", "1973-01", None))}
WINDOWS = ("1973", "1948")
ASSETS = list(config.ASSET_IDS)
ASSETS_1948 = ["us_equity", "treasury_10y", "cash"]   # gold and crude oil were fixed or administered before 1973
ASSET_NAME = {a["id"]: a["name"] for a in config.ASSETS}
FACTOR_NAME = {f["id"]: f["name"] for f in config.FACTORS}
FACTOR_UNIT = {f["id"]: f["unit"] for f in config.FACTORS}
STATE_NAME = {s["id"]: s["name"] for s in config.STATES}
STATE_UNIT = {s["id"]: s["unit"] for s in config.STATES}
CORE = list(config.CORE_FACTOR_IDS)
EXTRA_FACTORS = ["cpi"]                            # headline monthly inflation next to the core factors
STATES = ["cpi_yoy", "core_yoy", "goods_yoy", "services_yoy", "infl_expect", "ip_yoy", "unrate", "ff", "real_ff",
          "y10", "real_y10", "term_spread", "spread", "oil_yoy", "real_oil", "usd_yoy", "real_wage_yoy",
          "labor_share", "profit_share", "div_yield", "sentiment"]
TABLE_STATES = ["cpi_yoy", "ip_yoy", "unrate", "ff", "real_wage_yoy", "div_yield"]   # the state table of the tab
EXPOST_FLAGS = ("contraction", "severe", "high_infl", "cpi_neg", "core_low")
RT_FLAGS = ("contraction_rt", "severe_rt", "high_infl", "cpi_neg")
FLAG_NAME = {d["id"]: d["name"] for d in R.flag_defs()}
CONTRACTION_LABELS = ("recession", "depression", "stagflation")
CPI_BUCKETS = (("<2", -np.inf, 2.0), ("2-3", 2.0, 3.0), ("3-4", 3.0, 4.0), ("4-6", 4.0, 6.0), (">6", 6.0, np.inf))
MIN_CORR_MONTHS = 9                                # correlations on fewer months are not reported
MIN_SIM_CORR_MONTHS = 31                           # simulated months per group for a reference correlation
# out-of-sample grid (the parameters were chosen against NBER dates; this scores the choice honestly)
GRID = {"ip_swing_pct": (2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
        "u_rise_pp": (0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25),
        "u_smooth_months": (1, 2, 3, 4, 6)}
SPLIT_AT = "1973-01"                               # split-sample halves of the NBER window
HELD_OUT_MARGIN = 12                               # months around a held-out recession kept out of the fit
# data-revision study (model-based; vintages cannot be downloaded): IP level error AR(1) with persistence
# REVISION_PHI scaled to a 12-month growth error, unemployment +-0.1 pp with probability 0.2 each, CPI y/y noise
REVISION_PATHS = 300
REVISION_PHI = 0.97
REVISION_CPI_SD = 0.07
REVISION_CASES = (("cpi_unrate", "只有 CPI 和失业率的修订噪声", 0.0, 1),
                  ("ip_mild", "再加工业产出修订（12 个月增速误差 0.5 个百分点）", 0.5, 2),
                  ("ip_large", "再加工业产出修订（12 个月增速误差 1.0 个百分点）", 1.0, 3))
REVISION_SPOTS = {"1959_steel_strike": ("1959-06", "1960-01", "1959 年钢铁罢工"),
                  "1990_91": ("1990-08", "1991-06", "1990–91 年"),
                  "2014_16": ("2014-11", "2016-06", "2014–16 年工业产出下滑"),
                  "2022_24": ("2022-04", "2025-06", "2022–24 年"),
                  "2001": ("2001-01", "2002-03", "2001 年")}
NEAR_MISS = {"peak_window": ("2021-06", "2022-12"), "low_until": "2025-12", "sahm_from": "2023-01"}
# simulated studies: one-phase specs through runner.run_spec (the engine part's specs and seeds)
STUDY_PATHS = 600
STUDY_SEED = 20261101
DURATION_SEED = 7
STUDY_MONTHS = (12, 48)
DURATION_MONTHS = (6, 12, 18, 24, 36, 48)
STUDIES = [   # id, name, views, target (a label id, or "contraction" for the flag)
    ("stagflation_08", "滞胀 0.8", [{"var": "rg_stagflation", "mean": 0.8}], "stagflation"),
    ("depression_max", "深度衰退（最多能抽到的占比）", [{"var": "rg_depression", "mean": 1.0}], "depression"),
    ("recession_max", "衰退（最多能抽到的占比）", [{"var": "rg_recession", "mean": 1.0}], "recession"),
    ("inflation_08", "通胀 0.8", [{"var": "rg_inflation", "mean": 0.8}], "inflation"),
    ("expansion_10", "扩张 1.0", [{"var": "rg_expansion", "mean": 1.0}], "expansion"),
    ("contraction_10", "收缩 1.0", [{"var": "rg_contraction", "mean": 1.0}], "contraction"),
    ("deflation_max", "物价负增长（最多能抽到的占比）", [{"var": "rg_deflation", "mean": 1.0}], "deflation"),
    ("cpi_falling", "CPI 每年 −1% + 收缩 0", [{"var": "f_cpi", "mean_annual": -0.01},
                                            {"var": "rg_contraction", "mean": 0.0}], "deflation"),
]
SUPPORT_IDS = {**{i: f"rg_{i}" for i in R.LABEL_IDS}, "contraction": "rg_contraction",
               "high_infl": "rg_high_infl", "cpi_neg": "rg_cpi_neg"}
CONFLICT = {"spec": "stagflation", "phase": 0, "regime": {"stagflation": 0.8}}
LEAVE_ONE_OUT_MIN_GAIN = 0.05
# raw FRED caches of the label inputs: a panel month the source lacks was interpolated by data/panel.py
RAW_INPUTS = {"cpi": ("CPIAUCSL", "lvl_cpi", "CPI", "CPI 指数"), "unrate": ("UNRATE", "lvl_unrate", "失业率", "失业率"),
              "ip": ("INDPRO", "lvl_ip", "工业产出", "工业产出指数"), "core": ("CPILFESL", "lvl_core", "核心 CPI", "核心 CPI 指数")}
KNOWN_GAPS = {"2025-10": "美国联邦政府停摆"}         # why a source month is missing, where it is known


# ============================================================================ helpers

def _ym(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def _span(idx, a: int, b: int) -> str:
    return f"{_ym(idx[a])}..{_ym(idx[b])}"


def _num(x):
    """A finite float or None."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _years(start: str, end: str) -> str:
    """'1973-12', '1975-05' -> '1973–75'; one year -> '1970'."""
    a, b = int(start[:4]), int(end[:4])
    if a == b:
        return str(a)
    return f"{a}–{b % 100:02d}" if a // 100 == b // 100 else f"{a}–{b}"


def _ci(samples) -> list | None:
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size < 10:
        return None
    lo, hi = np.percentile(s, CI)
    return [float(lo), float(hi)]


def _excludes(ci, at: float = 0.0) -> bool:
    return ci is not None and (ci[0] > at or ci[1] < at)


def _boot(rng: np.random.Generator, K: int) -> np.ndarray:
    """(B, K) multiplicities of K whole episodes drawn with replacement."""
    if K == 0:
        return np.zeros((B, 0))
    return rng.multinomial(K, np.full(K, 1.0 / K), size=B).astype(float)


def _nw(y: np.ndarray, Z: np.ndarray, lags: int):
    """OLS of y on Z (Z holds the constant) with Newey-West (Bartlett) covariance: (beta, cov, resid)."""
    inv = np.linalg.pinv(Z.T @ Z)
    beta = inv @ Z.T @ y
    e = y - Z @ beta
    g = Z * e[:, None]
    S = g.T @ g
    for lag in range(1, min(lags, len(y) - 1) + 1):
        G = g[lag:].T @ g[:-lag]
        S += (1.0 - lag / (lags + 1.0)) * (G + G.T)
    return beta, inv @ S @ inv, e


def _t_mean(d, lags: int = HAC_FORWARD) -> tuple[float, float]:
    """The mean of d and its Newey-West t."""
    d = np.asarray(d, float)
    b, V, _ = _nw(d, np.ones((len(d), 1)), lags)
    return float(b[0]), float(b[0] / np.sqrt(V[0, 0]))


def _codes(labels: pd.Series) -> np.ndarray:
    """Label ids -> codes (-1 = unknown)."""
    return np.array([R.CODE.get(x, -1) for x in labels], dtype=np.int64)


def _episode_list(idx, mask) -> list[str]:
    return [_span(idx, a, b) for a, b in R.runs(mask)]


def _panel_window(idx, start=None, end=None) -> np.ndarray:
    m = np.ones(len(idx), bool)
    if start is not None:
        m &= np.asarray(idx >= pd.Timestamp(start))
    if end is not None:
        m &= np.asarray(idx <= pd.Timestamp(end))
    return m


# ============================================================================ counts and timeline

def counts_section(lab: pd.DataFrame, masks: dict) -> dict:
    ex, rt = lab["label"].to_numpy(object), lab["label_rt"].to_numpy(object)
    out = {}
    for wid, win in masks.items():
        block = {}
        for ver, arr in (("expost", ex), ("realtime", rt)):
            m = win & (arr != "")
            n = int(m.sum())
            block[ver] = {"months": n, "labels": {i: {"months": int((m & (arr == i)).sum()),
                                                      "share": float((m & (arr == i)).sum() / n) if n else 0.0,
                                                      "runs": len(R.runs(m & (arr == i)))} for i in R.LABEL_IDS}}
        block["flags"] = {f: {"months": int(((lab[f] == 1).to_numpy() & win).sum()),
                              "runs": len(R.runs((lab[f] == 1).to_numpy() & win))}
                          for f in R.FLAG_IDS if f in lab}
        known = win & (ex != "")
        trend = lab["infl_trend_12m"].to_numpy(float)
        inf = known & (ex == "inflation") & np.isfinite(trend)
        # the trend needs CPI y/y 12 months earlier, so it starts a year after the window (1949-01 for "1948")
        has_trend = np.flatnonzero(known & np.isfinite(trend))
        block["inflation_falling"] = {"months_with_trend": int(inf.sum()), "falling": int((inf & (trend <= 0)).sum()),
                                      "first_month": _ym(lab.index[has_trend[0]]) if has_trend.size else None}
        out[wid] = block
    known48 = masks["1948"] & (ex != "")
    out["agreement_rt_expost"] = {wid: float((ex[win & (ex != "")] == rt[win & (ex != "")]).mean())
                                  for wid, win in masks.items()}
    conf = np.zeros((len(R.LABEL_IDS), len(R.LABEL_IDS)), int)
    for a, b in zip(ex[known48], rt[known48]):
        conf[R.CODE[a], R.CODE[b]] += 1
    out["confusion_1948"] = conf.tolist()
    first = int(np.flatnonzero(known48)[0])
    runs48 = _label_runs(ex, lab.index, first, int(np.flatnonzero(known48)[-1]))
    out["runs_expost_1948"] = {"runs": len(runs48), "le_2_months": sum(r["months"] <= 2 for r in runs48)}
    hi = config.REGIMES["params"]["high_inflation_pct"]
    c73 = lab["cpi_yoy"].to_numpy(float)[masks["1973"]]
    out["cpi_near_threshold_1973"] = {"threshold": hi, "within_0_10": int((np.abs(c73 - hi) <= 0.10).sum()),
                                      "within_0_25": int((np.abs(c73 - hi) <= 0.25).sum())}
    return out


def timeline_section(panel: pd.DataFrame, lab: pd.DataFrame, win: np.ndarray) -> dict:
    L = lab[win]
    series = ("cpi_yoy", "core_yoy", "infl_trend_12m", "unrate_3m", "sahm_gap", "ip_yoy", "ip_below_peak_rt",
              "ip_drawdown_24m", "u_rise_rt")
    prov = R.provisional_from(panel)
    return {
        "dates": [_ym(d) for d in L.index], "label": L["label"].tolist(), "label_rt": L["label_rt"].tolist(),
        "provisional_from": _ym(prov) if prov is not None else None,
        "flags": {f: L[f].astype(int).tolist() for f in (*R.FLAG_IDS, "infl_rising") if f in L},
        "series": {s: [None if not np.isfinite(v) else round(float(v), 3) for v in L[s].to_numpy(float)]
                   for s in series if s in L},
        "runs": R.episodes(L["label"]).to_dict("records"),
        "runs_rt": R.episodes(L["label_rt"]).to_dict("records"),
        "nber": [{"peak": pk, "trough": tr, "first_month": str(pd.Period(pk, "M") + 1)} for pk, tr in R.NBER_PEAK_TROUGH],
    }


def _label_runs(lab: np.ndarray, idx, first: int, last: int) -> list[dict]:
    """Runs of the label array over positions first..last (inclusive), censoring at both ends marked."""
    out, i = [], first
    while i <= last:
        j = i
        while j + 1 <= last and lab[j + 1] == lab[i]:
            j += 1
        out.append({"label": lab[i], "start": _ym(idx[i]), "end": _ym(idx[j]), "months": j - i + 1,
                    "left_censored": i == first, "right_censored": j == last})
        i = j + 1
    return out


# ============================================================================ validation against NBER dates

def nber_score(flag, index, lo: str | None = None, hi: str | None = None, before: int = 3, after: int = 12) -> dict:
    """Score any monthly 0/1 flag against the NBER months in [lo, hi], as regimes.validate_nber scores the
    contraction flags: each flagged run is assigned to the recession it overlaps most within
    [first recession month - before, trough + after]; unassigned runs are false alarms."""
    lo, hi = lo or CFG["nber_window"][0], hi or CFG["nber_window"][1]
    pidx = pd.DatetimeIndex(index).to_period("M")
    f = np.asarray(flag, bool)
    w = (pidx >= pd.Period(lo, "M")) & (pidx <= pd.Period(hi, "M"))
    rec = R.nber_months(pidx).astype(bool)
    P, Q = f[w], rec[w]
    tp, fp, fn, tn = int((P & Q).sum()), int((P & ~Q).sum()), int((~P & Q).sum()), int((~P & ~Q).sum())
    widx = pidx[w]
    rl = [(widx[a], widx[b]) for a, b in R.runs(P)]
    recs = [(pk, tr, pd.Period(pk, "M") + 1, pd.Period(tr, "M")) for pk, tr in R.NBER_PEAK_TROUGH
            if pd.Period(lo, "M") <= pd.Period(pk, "M") + 1 <= pd.Period(hi, "M")]
    assign: dict[int, list[int]] = {}
    for i, (a, b) in enumerate(rl):
        best, bestov = None, 0
        for j, (_, _, s0, e0) in enumerate(recs):
            ov = max(0, (min(b, e0 + after) - max(a, s0 - before)).n + 1)
            if ov > bestov:
                best, bestov = j, ov
        if best is not None:
            assign.setdefault(best, []).append(i)
    per = []
    for j, (pk, tr, s0, e0) in enumerate(recs):
        ids = assign.get(j, [])
        if not ids:
            per.append({"nber": f"{pk}→{tr}", "first_month": str(s0), "detected": False})
            continue
        a, b = min(rl[i][0] for i in ids), max(rl[i][1] for i in ids)
        per.append({"nber": f"{pk}→{tr}", "first_month": str(s0), "detected": bool(P[(widx >= s0) & (widx <= e0)].any()),
                    "start": str(a), "end": str(b), "start_err": (a - s0).n, "end_err": (b - e0).n})
    used = {i for v in assign.values() for i in v}
    se = [r["start_err"] for r in per if "start_err" in r]
    ee = [r["end_err"] for r in per if "end_err" in r]
    fa = [(a, b) for i, (a, b) in enumerate(rl) if i not in used]
    tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"balanced_accuracy": 0.5 * (tpr + tnr), "tpr": tpr, "tnr": tnr, "precision": tp / max(tp + fp, 1),
            "detected": sum(r["detected"] for r in per), "recessions": len(per),
            "mean_start_err": float(np.mean(se)) if se else None, "mean_end_err": float(np.mean(ee)) if ee else None,
            "mean_abs_start_err": float(np.mean(np.abs(se))) if se else None,
            "mean_abs_end_err": float(np.mean(np.abs(ee))) if ee else None,
            "false_alarms": [[str(a), str(b)] for a, b in fa], "fa_months": sum((b - a).n + 1 for a, b in fa),
            "months_after_window": int(f[pidx > pd.Period(hi, "M")].sum()), "per_recession": per}


def _zigzag(x: np.ndarray, h: float) -> list[tuple[int, int]]:
    """Alternating pivots of a series in absolute units (a peak is confirmed once x falls h below the running
    maximum, a trough once it rises h above the running minimum): [(position, +1 peak / -1 trough)]."""
    x = np.asarray(x, float)
    n = len(x)
    piv = []
    i0 = 0
    while i0 < n and np.isnan(x[i0]):
        i0 += 1
    trend, imax, imin = 0, i0, i0
    for i in range(i0 + 1, n):
        xi = x[i]
        if np.isnan(xi):
            continue
        if trend == 0:
            if xi > x[imax]:
                imax = i
            if xi < x[imin]:
                imin = i
            if x[imax] - xi >= h and imax > imin:
                piv.append((imax, 1))
                trend, imin = -1, i
            elif xi - x[imin] >= h and imin > imax:
                piv.append((imin, -1))
                trend, imax = 1, i
            elif x[imax] - xi >= h:
                piv.append((imax, 1))
                trend, imin = -1, i
            elif xi - x[imin] >= h:
                piv.append((imin, -1))
                trend, imax = 1, i
        elif trend == 1:
            if xi > x[imax]:
                imax = i
            elif x[imax] - xi >= h:
                piv.append((imax, 1))
                trend, imin = -1, i
        else:
            if xi < x[imin]:
                imin = i
            elif xi - x[imin] >= h:
                piv.append((imin, -1))
                trend, imax = 1, i
    return piv


def _down_legs(x: np.ndarray, h: float) -> np.ndarray:
    """Months after each zigzag peak through the next trough (an unfinished last leg runs to the end)."""
    piv = _zigzag(x, h)
    out = np.zeros(len(x), bool)
    for k, (p, kind) in enumerate(piv):
        if kind == 1:
            out[p + 1: (piv[k + 1][0] if k + 1 < len(piv) else len(x) - 1) + 1] = True
    return out


def alternative_rules(panel: pd.DataFrame, lab: pd.DataFrame) -> list[tuple]:
    """(id, name, kind, flag) of the chosen rule and the rules it was compared with (classifier part)."""
    ur = panel["lvl_unrate"].to_numpy(float)
    lip = 100.0 * np.log(panel["lvl_ip"].to_numpy(float))
    u3 = pd.Series(ur).rolling(3, min_periods=1).mean().to_numpy()
    u3_full = pd.Series(ur).rolling(3, min_periods=3).mean()
    gap = (u3_full - u3_full.shift(1).rolling(12, min_periods=1).min()).to_numpy()
    trend10 = pd.Series(ur).rolling(120, min_periods=60).mean().to_numpy()
    ip_yoy = panel["s_ip_yoy"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        sahm = gap >= 0.5
        proto = ((gap >= 0.3) & (ip_yoy < 0)) | (gap >= 0.5)
        slack = (u3 - trend10) >= 1.0
    return [
        ("chosen_expost", "选定规则（事后）", "expost", (lab["contraction"] == 1).to_numpy()),
        ("chosen_realtime", "选定规则（实时）", "realtime", (lab["contraction_rt"] == 1).to_numpy()),
        ("ip_only", "只看工业产出：3% 下跌段", "expost", _down_legs(lip, 3.0)),
        ("unemployment_only", "只看失业率：3 个月均值上行 1 个百分点以上的段", "expost", _down_legs(-u3, 1.0)),
        ("composite", "合成指数 100·ln(工业产出) − 3×失业率 的 6 点下跌段", "expost", _down_legs(lip - 3.0 * u3, 6.0)),
        ("sahm", "Sahm 规则：失业率缺口 ≥ 0.5（单边）", "realtime", sahm),
        ("prototype", "原型：(失业率缺口 ≥ 0.3 且工业产出同比 < 0) 或缺口 ≥ 0.5", "realtime", proto),
        ("slack", "失业率水平：3 个月均值比过去 10 年均值高 1 个百分点以上", "expost", slack),
    ]


def ip_legs(panel: pd.DataFrame, lab: pd.DataFrame) -> list[dict]:
    """Every confirmed IP down-leg with its decline, the unemployment rise at the trough and the NBER months."""
    p = CFG["params"]
    ip = panel["lvl_ip"].to_numpy(float)
    u3 = pd.Series(panel["lvl_unrate"].to_numpy(float)).rolling(int(p["u_smooth_months"]), min_periods=1).mean()
    fw = R.zigzag(ip, u3.to_numpy(), p["ip_swing_pct"])
    piv = np.flatnonzero(fw["pivot"][0])
    kinds = fw["pivot"][0, piv]
    rec = R.nber_months(panel.index)
    idx = panel.index
    out = []
    for k in range(len(piv)):
        if kinds[k] != 1:
            continue
        a = int(piv[k])
        b = int(piv[k + 1]) if k + 1 < len(piv) else len(idx) - 1
        rise = float(fw["rise"][0, b])
        out.append({"peak": _ym(idx[a]), "trough": _ym(idx[b]), "ip_decline": 100.0 * (1.0 - np.nanmin(ip[a: b + 1]) / ip[a]),
                    "u_rise": rise, "nber_months": int(rec[a + 1: b + 1].sum()),
                    "contraction": bool(rise >= p["u_rise_pp"] - R.EPS), "finished": k + 1 < len(piv)})
    return out


def _grid_flags(panel: pd.DataFrame) -> dict:
    cpi, ur, ip = (panel[c].to_numpy(float) for c in R.INPUTS.values())
    out = {}
    for g in itertools.product(*GRID.values()):
        r = R.classify(cpi, ur, ip, params=dict(zip(GRID, g)))
        out[g] = (r["contraction"].astype(bool), r["contraction_rt"].astype(bool))
    return out


def _ba(f: np.ndarray, rec: np.ndarray, m: np.ndarray) -> tuple[float, tuple]:
    P, Q = f[m], rec[m]
    tp, fn = int((P & Q).sum()), int((~P & Q).sum())
    tn, fp = int((~P & ~Q).sum()), int((P & ~Q).sum())
    return 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)), (tp, fn, tn, fp)


def _ba_from(c) -> float:
    tp, fn, tn, fp = c
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def out_of_sample(panel: pd.DataFrame, flags: dict) -> dict:
    """The parameter grid scored honestly: chosen by balanced accuracy with one recession window held out at a
    time (ties reported as a range) and scored on it; and selected on one half of the NBER window, scored on
    the other (critique part)."""
    idx = panel.index.to_period("M")
    rec = R.nber_months(idx).astype(bool)
    lo, hi = (pd.Period(x, "M") for x in CFG["nber_window"])
    inwin = np.asarray((idx >= lo) & (idx <= hi))
    pos = {p: i for i, p in enumerate(idx)}
    recs = [(pos[pd.Period(pk, "M") + 1], pos[pd.Period(tr, "M")]) for pk, tr in R.NBER_PEAK_TROUGH]
    T = len(idx)
    W = []
    for s0, e0 in recs:
        m = np.zeros(T, bool)
        m[max(s0 - HELD_OUT_MARGIN, 0): e0 + HELD_OUT_MARGIN + 1] = True
        W.append(m & inwin)
    p = CFG["params"]
    chosen = (p["ip_swing_pct"], p["u_rise_pp"], int(p["u_smooth_months"]))
    grid = list(flags)

    def held_out(f, k):
        m = W[k]
        s0, e0 = recs[k]
        _, cnt = _ba(f, rec, m)
        det = bool(f[s0: e0 + 1].any())
        on = np.flatnonzero(f & m)
        near = on[(on >= s0 - 3) & (on <= e0 + 12)]
        se, ee = (int(near.min() - s0), int(near.max() - e0)) if det and near.size else (None, None)
        return {"cnt": cnt, "detected": det, "start_err": se, "end_err": ee, "fp_months": int((f & ~rec & m).sum())}

    def key(r):
        return (r["detected"], -(abs(r["start_err"]) if r["start_err"] is not None else 99))

    agg = {"lo": np.zeros(4), "hi": np.zeros(4), "in": np.zeros(4)}
    rows = []
    for k, (s0, e0) in enumerate(recs):
        train = inwin & ~W[k]
        sc = {g: _ba(flags[g][0], rec, train)[0] for g in grid}
        best = max(sc.values())
        tied = [g for g in grid if sc[g] >= best - 1e-12]
        res = [held_out(flags[g][0], k) for g in tied]
        agg["lo"] += np.array(min(res, key=key)["cnt"], float)
        agg["hi"] += np.array(max(res, key=key)["cnt"], float)
        agg["in"] += np.array(held_out(flags[chosen][0], k)["cnt"], float)
        rows.append({"recession": f"{idx[s0]}..{idx[e0]}", "tied_settings": len(tied),
                     "missing": int(sum(not r["detected"] for r in res)), "chosen_tied": chosen in tied})
    halves = {"1948_1972": inwin & np.asarray(idx < pd.Period(SPLIT_AT, "M")),
              "1973_2020": inwin & np.asarray(idx >= pd.Period(SPLIT_AT, "M"))}
    split = {}
    for a, b in (("1948_1972", "1973_2020"), ("1973_2020", "1948_1972")):
        sc = {g: _ba(flags[g][0], rec, halves[a])[0] for g in grid}
        best = max(sc.values())
        tied = [g for g in grid if sc[g] >= best - 1e-12]
        test = [_ba(flags[g][0], rec, halves[b])[0] for g in tied]
        test_rt = [_ba(flags[g][1], rec, halves[b])[0] for g in tied]
        split[f"{a}_to_{b}"] = {"train_best_ba": best, "tied_settings": len(tied), "test_ba": [min(test), max(test)],
                                "test_ba_median": float(np.median(test)), "test_rt_ba": [min(test_rt), max(test_rt)],
                                "chosen_test_ba": _ba(flags[chosen][0], rec, halves[b])[0],
                                "chosen_tied": chosen in tied,
                                "share_tied_below_0_95": float(np.mean(np.array(test) < 0.95))}
    full = {g: _ba(flags[g][0], rec, inwin)[0] for g in grid}
    misses = [r for r in rows if r["missing"]]
    return {"loro": {"cv_ba_worst_tie": _ba_from(agg["lo"]), "cv_ba_best_tie": _ba_from(agg["hi"]),
                     "insample_ba_same_windows": _ba_from(agg["in"]),
                     "heldout_detected": {"all_tied": sum(r["missing"] == 0 for r in rows),
                                          "any_tied": sum(r["missing"] < r["tied_settings"] for r in rows),
                                          "recessions": len(rows)},
                     "misses": [{"recession": r["recession"], "tied_settings": r["tied_settings"],
                                 "missing": r["missing"]} for r in misses],
                     "per_recession": rows, "margin_months": HELD_OUT_MARGIN},
            "split": split, "grid_points_ba_ge_0955": int(sum(v >= 0.955 for v in full.values())),
            "grid_size": len(grid), "grid_best_full_ba": max(full.values()), "chosen_full_ba": full[chosen],
            "grid": {k: list(v) for k, v in GRID.items()}, "chosen": dict(zip(GRID, chosen))}


def sensitivity(panel: pd.DataFrame, flags: dict) -> list[dict]:
    """One parameter at a time around the chosen point (the grid's own flags)."""
    p = CFG["params"]
    chosen = {"ip_swing_pct": p["ip_swing_pct"], "u_rise_pp": p["u_rise_pp"], "u_smooth_months": int(p["u_smooth_months"])}
    out = []
    for param, values in GRID.items():
        for v in values:
            g = tuple((v if k == param else chosen[k]) for k in GRID)
            row = {"param": param, "value": v, "chosen": v == chosen[param]}
            for ver, f in (("expost", flags[g][0]), ("realtime", flags[g][1])):
                s = nber_score(f, panel.index)
                row[ver] = {"ba": s["balanced_accuracy"], "detected": s["detected"], "recessions": s["recessions"],
                            "fa_runs": len(s["false_alarms"]), "start_err": s["mean_start_err"],
                            "end_err": s["mean_end_err"], "start_err_abs": s["mean_abs_start_err"],
                            "end_err_abs": s["mean_abs_end_err"], "months_after_window": s["months_after_window"]}
            out.append(row)
    return out


def revisions(panel: pd.DataFrame, masks: dict) -> dict:
    """How often the labels survive revision-like noise in the inputs (model-based: today's revised series
    perturbed; IP gets a persistent level error, unemployment +-0.1 pp, CPI y/y a small noise)."""
    idx = panel.index
    cpi, ur, ip = (panel[c].to_numpy(float) for c in R.INPUTS.values())
    T = len(ip)
    base = R.classify(cpi, ur, ip)
    pidx = idx.to_period("M")
    lo, hi = (pd.Period(x, "M") for x in CFG["nber_window"])
    win = np.asarray((pidx >= lo) & (pidx <= hi))
    pos = {p: i for i, p in enumerate(pidx)}
    recs = [(pos[pd.Period(pk, "M") + 1], pos[pd.Period(tr, "M")]) for pk, tr in R.NBER_PEAK_TROUGH]
    near = np.zeros(T, bool)
    for s0, e0 in recs:
        near[max(s0 - 3, 0): e0 + 13] = True
    cn = np.concatenate([[0], np.cumsum(near)])

    def score(flag):
        rr = R.runs(flag)
        a = np.array([x for x, _ in rr], int)
        b = np.array([y for _, y in rr], int)
        det, lags = 0, []
        for s0, e0 in recs:
            det += int(flag[s0: e0 + 1].any())
            ov = (b >= s0 - 3) & (a <= e0 + 12) if a.size else np.zeros(0, bool)
            if ov.any():
                lags.append(int(a[ov].min() - s0))
        fa = [(x, y) for x, y in R.runs(flag & win) if cn[y + 1] - cn[x] == 0]
        return det, (float(np.mean(lags)) if lags else np.nan), len(fa)

    spots = {k: _panel_window(idx, a, b) for k, (a, b, _) in REVISION_SPOTS.items()}

    def q(x):
        return [float(np.percentile(x, p)) for p in (5, 50, 95)]

    cases = []
    for cid, name, s_ip, seed in REVISION_CASES:
        rng = np.random.default_rng(seed)
        n = REVISION_PATHS
        sig = (s_ip / 100.0) / np.sqrt(2 * (1 - REVISION_PHI ** 12)) if s_ip > 0 else 0.0
        e = np.zeros((n, T))
        if sig > 0:
            e[:, 0] = rng.normal(0, sig, n)
            innov = rng.normal(0, sig * np.sqrt(1 - REVISION_PHI ** 2), (n, T))
            for t in range(1, T):
                e[:, t] = REVISION_PHI * e[:, t - 1] + innov[:, t]
        ipn = ip[None, :] * np.exp(e)
        urn = ur[None, :] + rng.choice([-0.1, 0.0, 0.1], size=(n, T), p=[0.2, 0.6, 0.2])
        cpn = cpi[None, :] + rng.normal(0, REVISION_CPI_SD, (n, T))
        r = R.classify(cpn, urn, ipn)
        kept = {f"{ver}_kept_{wid}": q((r[key][:, w] == base[key][None, w]).mean(axis=1))
                for ver, key in (("rt", "label_rt"), ("ex", "label")) for wid, w in masks.items()}
        st_rt = [score(r["contraction_rt"][i].astype(bool)) for i in range(n)]
        st_ex = [score(r["contraction"][i].astype(bool)) for i in range(n)]
        det_rt = np.array([s[0] for s in st_rt])
        det_ex = np.array([s[0] for s in st_ex])
        cases.append({
            "id": cid, "name": name, "ip_growth_error_sd_pp": s_ip, "unrate_noise_pp": 0.1, "cpi_yoy_noise_sd": REVISION_CPI_SD,
            "seed": seed, **kept,
            "rt_detected": q(det_rt), "share_12_of_12_rt": float((det_rt == len(recs)).mean()),
            "rt_mean_start_lag": q(np.array([s[1] for s in st_rt])),
            "rt_false_alarm_runs": q(np.array([s[2] for s in st_rt])),
            "ex_detected": q(det_ex), "share_12_of_12_ex": float((det_ex == len(recs)).mean()),
            "ex_share_no_false_alarm": float((np.array([s[2] for s in st_ex]) == 0).mean()),
            "spots": {k: {"name": REVISION_SPOTS[k][2], "window": [REVISION_SPOTS[k][0], REVISION_SPOTS[k][1]],
                          "expost": float(r["contraction"][:, m].any(axis=1).mean()),
                          "realtime": float(r["contraction_rt"][:, m].any(axis=1).mean()),
                          "base_expost": bool(base["contraction"][m].any()),
                          "base_realtime": bool(base["contraction_rt"][m].any())} for k, m in spots.items()}})
    return {"cases": cases, "paths": REVISION_PATHS, "seed": [c[3] for c in REVISION_CASES], "phi": REVISION_PHI}


def near_miss(panel: pd.DataFrame, lab: pd.DataFrame) -> dict:
    """2022-24: the IP fall stopped short of the 3% line while the Sahm gap reached 0.5."""
    ipl = panel["lvl_ip"]
    pk = ipl.loc[NEAR_MISS["peak_window"][0]: NEAR_MISS["peak_window"][1]].idxmax()
    after = ipl.loc[pk: NEAR_MISS["low_until"]]
    dec = float(100.0 * (1.0 - after.min() / ipl.loc[pk]))
    gap = lab["sahm_gap"]
    sahm = gap.index[(gap >= 0.5 - R.EPS).to_numpy() & (gap.index >= pd.Timestamp(NEAR_MISS["sahm_from"]))]
    return {"ip_peak": _ym(pk), "max_decline_pct": dec, "trough": _ym(after.idxmin()),
            "swing_pct": CFG["params"]["ip_swing_pct"], "short_by_pp": CFG["params"]["ip_swing_pct"] - dec,
            "sahm_ge_05": [_ym(m) for m in sahm]}


def validation_section(panel: pd.DataFrame, lab: pd.DataFrame, masks: dict, timings: dict) -> dict:
    t0 = time.time()
    v = R.validate_nber(panel)
    alts = []
    for aid, name, kind, flag in alternative_rules(panel, lab):
        s = nber_score(flag, panel.index)
        alts.append({"id": aid, "name": name, "kind": kind, "ba": s["balanced_accuracy"], "tpr": s["tpr"], "tnr": s["tnr"],
                     "precision": s["precision"], "detected": s["detected"], "recessions": s["recessions"],
                     "start_err": s["mean_start_err"], "end_err": s["mean_end_err"],
                     "start_err_abs": s["mean_abs_start_err"], "end_err_abs": s["mean_abs_end_err"],
                     "fa_runs": len(s["false_alarms"]), "fa_months": s["fa_months"], "false_alarms": s["false_alarms"],
                     "months_after_window": s["months_after_window"]})
    legs = ip_legs(panel, lab)
    timings["validation_scores"] = time.time() - t0
    t0 = time.time()
    flags = _grid_flags(panel)
    oos = out_of_sample(panel, flags)
    sens = sensitivity(panel, flags)
    timings["validation_grid"] = time.time() - t0
    t0 = time.time()
    rev = revisions(panel, masks)
    timings["validation_revisions"] = time.time() - t0
    non_rec = [x["u_rise"] for x in legs if x["nber_months"] == 0]
    rec_legs = [x["u_rise"] for x in legs if x["nber_months"] > 0 and x["contraction"]]
    return {"window": v["window"], "expost": v["expost"], "realtime": v["realtime"],
            "months_flagged_after_window": v["months_flagged_after_window"], "alternatives": alts, "legs": legs,
            "legs_margin": {"largest_u_rise_without_recession": max(non_rec) if non_rec else None,
                            "smallest_u_rise_in_recession": min(rec_legs) if rec_legs else None,
                            "threshold": CFG["params"]["u_rise_pp"]},
            "sensitivity": sens, "out_of_sample": oos, "revisions": rev, "near_miss_2022_24": near_miss(panel, lab)}


# ============================================================================ same-month asset statistics

def _expost_groups(lab: pd.DataFrame) -> dict:
    """gid -> (kind, id, name, mask, domain); the 'rest' of a group is taken inside its domain (known months;
    for core_low, known months with core CPI)."""
    known = (lab["label"] != "").to_numpy()
    core_ok = known & np.isfinite(lab["core_yoy"].to_numpy(float)) if "core_yoy" in lab else np.zeros_like(known)
    g = {"all": ("all", "all", "全部月份", known, known)}
    for i in R.LABEL_IDS:
        g[f"label:{i}"] = ("label", i, R.NAME[i], (lab["label"] == i).to_numpy(), known)
    for f in EXPOST_FLAGS:
        if f in lab:
            dom = core_ok if f == "core_low" else known
            g[f"flag:{f}"] = ("flag", f, FLAG_NAME[f], (lab[f] == 1).to_numpy() & dom, dom)
    return g


def _is_contraction_group(kind: str, key: str) -> bool:
    return key in CONTRACTION_LABELS or (kind == "flag" and key in ("contraction", "severe"))


def _asset_sums(P: pd.DataFrame, a: str, rr) -> np.ndarray:
    """Per-episode sums: months, log, real log, return, squared return, excess over cash, positive months,
    positive real months, real return."""
    r = P[f"r_{a}"].to_numpy(float)
    lr = np.log1p(r)
    rlr = R.real_log_return(r, P["f_cpi"].to_numpy(float))
    rr_ = np.expm1(rlr)
    cols = np.column_stack([np.ones_like(r), lr, rlr, r, r * r, r - P["r_cash"].to_numpy(float),
                            (r > 0).astype(float), (rr_ > 0).astype(float), rr_])
    return np.array([cols[s: e + 1].sum(axis=0) for s, e in rr]) if rr else np.zeros((0, cols.shape[1]))


def _asset_stats(T: np.ndarray, is_cash: bool) -> dict[str, np.ndarray]:
    T = np.atleast_2d(T)
    n, lr, rlr, r, r2, ex, pos, rpos, rr_ = (T[:, i] for i in range(9))
    with np.errstate(invalid="ignore", divide="ignore"):
        vol = np.sqrt(np.maximum((r2 - r * r / n) / (n - 1), 0.0) * 12.0)
        return {"ann_nominal": np.expm1(12.0 * lr / n), "ann_real": np.expm1(12.0 * rlr / n), "vol": vol,
                "sharpe": np.full_like(n, np.nan) if is_cash else np.where(vol > 1e-10, 12.0 * ex / n / vol, np.nan),
                "share_positive": pos / n, "share_positive_real": rpos / n}


def _pair_sums(x: np.ndarray, y: np.ndarray, rr) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    cols = np.column_stack([ok.astype(float), xx, yy, xx * xx, yy * yy, xx * yy])
    return np.array([cols[s: e + 1].sum(axis=0) for s, e in rr])


def _corr_from(T: np.ndarray) -> np.ndarray:
    T = np.atleast_2d(T)
    n, sx, sy, sxx, syy, sxy = (T[:, i] for i in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        vx, vy = sxx - sx * sx / n, syy - sy * sy / n
        return np.where((vx > 0) & (vy > 0) & (n >= MIN_CORR_MONTHS), cov / np.sqrt(vx * vy), np.nan)


def _worst_window(P: pd.DataFrame, a: str, rr, k: int = 12, real: bool = False) -> dict | None:
    """Worst compounded return over min(k, episode length) consecutive months inside one episode."""
    r = P[f"r_{a}"].to_numpy(float)
    lr = R.real_log_return(r, P["f_cpi"].to_numpy(float)) if real else np.log1p(r)
    best = None
    for s, e in rr:
        w = min(k, e - s + 1)
        seg = lr[s: e + 1]
        if np.isnan(seg).any():
            continue
        cs = np.concatenate([[0.0], np.cumsum(seg)])
        sums = cs[w:] - cs[:-w]
        i = int(np.argmin(sums))
        if best is None or sums[i] < best[0]:
            best = (float(sums[i]), s + i, s + i + w - 1, w)
    if best is None:
        return None
    return {"value": float(np.expm1(best[0])), "start": _ym(P.index[best[1]]), "end": _ym(P.index[best[2]]),
            "months": best[3]}


def _corr_cell(P, wid, gid, gm, dom, win, xa, ya, cname) -> dict | None:
    x, y = P[f"r_{xa}"].to_numpy(float), P[f"r_{ya}"].to_numpy(float)
    valid = win & gm & np.isfinite(x) & np.isfinite(y)
    rr = R.runs(valid)
    if not rr:
        return None
    rng = R.rng_for("assets", wid, gid, cname, "corr")
    S = _pair_sums(x, y, rr)
    val = float(_corr_from(S.sum(axis=0))[0])
    bs = _corr_from(_boot(rng, len(rr)) @ S)
    rec = {"value": _num(val), "ci90": _ci(bs) if len(rr) >= 2 else None, "months": int(valid.sum()),
           "episodes": len(rr), "thin": len(rr) < THIN}
    if gid != "all":
        rrc = R.runs(win & ~gm & dom & np.isfinite(x) & np.isfinite(y))
        if rrc:
            Sc = _pair_sums(x, y, rrc)
            d = val - float(_corr_from(Sc.sum(axis=0))[0])
            ci = _ci(bs - _corr_from(_boot(rng, len(rrc)) @ Sc)) if len(rr) >= 2 else None
            rec["vs_rest"] = {"value": _num(d), "ci90": ci, "survives": bool(len(rr) >= THIN and _excludes(ci))}
    return rec


def _asset_cell(P, wid, gid, gm, dom, win, a) -> dict | None:
    r = P[f"r_{a}"].to_numpy(float)
    valid = win & gm & np.isfinite(r) & np.isfinite(P["f_cpi"].to_numpy(float)) & np.isfinite(P["r_cash"].to_numpy(float))
    rr = R.runs(valid)
    if not rr:
        return None
    rng = R.rng_for("assets", wid, gid, a, "returns")
    cash = a == "cash"
    S = _asset_sums(P, a, rr)
    point = {k: float(v[0]) for k, v in _asset_stats(S.sum(axis=0), cash).items()}
    bst = _asset_stats(_boot(rng, len(rr)) @ S, cash)
    two = len(rr) >= 2
    cell = {"months": int(valid.sum()), "episodes": len(rr), "thin": len(rr) < THIN,
            "first": _ym(P.index[rr[0][0]]), "last": _ym(P.index[rr[-1][1]]),
            "ann_real": {"value": _num(point["ann_real"]), "ci90": _ci(bst["ann_real"]) if two else None},
            "ann_nominal": {"value": _num(point["ann_nominal"]), "ci90": _ci(bst["ann_nominal"]) if two else None},
            "vol": _num(point["vol"]), "sharpe": None if cash else _num(point["sharpe"]),
            "share_positive": _num(point["share_positive"]), "share_positive_real": _num(point["share_positive_real"]),
            "worst_12m": _worst_window(P, a, rr), "worst_12m_real": _worst_window(P, a, rr, real=True)}
    if gid != "all":
        rrc = R.runs(win & ~gm & dom & np.isfinite(r))
        if rrc:
            Sc = _asset_sums(P, a, rrc)
            pc = _asset_stats(Sc.sum(axis=0), cash)
            bc = _asset_stats(_boot(rng, len(rrc)) @ Sc, cash)
            d = point["ann_real"] - float(pc["ann_real"][0])
            ci = _ci(bst["ann_real"] - bc["ann_real"]) if two else None
            cell["vs_rest"] = {"value": _num(d), "ci90": ci, "rest_value": _num(pc["ann_real"][0]),
                               "survives": bool(len(rr) >= THIN and _excludes(ci))}
    return cell


def assets_section(P: pd.DataFrame, lab: pd.DataFrame, masks: dict) -> dict:
    groups = _expost_groups(lab)
    known = (lab["label"] != "").to_numpy()
    eq_ok = np.isfinite(P["r_us_equity"].to_numpy(float))
    out = {"windows": {}}
    for wid in WINDOWS:
        win = masks[wid]
        assets = ASSETS if wid == "1973" else ASSETS_1948
        w = {"start": _ym(P.index[win][0]), "end": _ym(P.index[win][-1]), "months": int((win & known).sum()),
             "assets": assets, "groups": {}}
        for gid, (kind, key, name, gm, dom) in groups.items():
            g = {"kind": kind, "id": key, "name": name, "contraction_cell": _is_contraction_group(kind, key)}
            for cname, (xa, ya) in (("corr_stock_bond", ("us_equity", "treasury_10y")),
                                    ("corr_gold_equity", ("gold", "us_equity"))):
                if wid == "1948" and "gold" in (xa, ya):
                    continue
                cell = _corr_cell(P, wid, gid, gm, dom, win, xa, ya, cname)
                if cell is not None:
                    g[cname] = cell
            g["assets"] = {a: c for a in assets if (c := _asset_cell(P, wid, gid, gm, dom, win, a)) is not None}
            first = next(iter(g["assets"].values()), None)
            base = win & gm & eq_ok
            g["months"] = first["months"] if first else 0
            g["episodes"] = len(R.runs(base))
            g["episode_list"] = _episode_list(P.index, base)
            g["thin"] = g["episodes"] < THIN
            g["survivors"] = [a for a, c in g["assets"].items() if (c.get("vs_rest") or {}).get("survives")]
            w["groups"][gid] = g
        out["windows"][wid] = w
    out["inflation_by_era"] = _inflation_by_era(P, lab, masks)
    out["corr_by_half"] = _corr_by_half(P, lab, masks["1973"])
    out["corr_by_cpi_bucket"] = _corr_by_bucket(P, masks["1973"])
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= MIN_CORR_MONTHS else None


def _inflation_by_era(P, lab, masks) -> list[dict]:
    """Same-month real equity and Treasury returns in ex-post 通胀 months, block by block."""
    idx = P.index
    last = _ym(idx[masks["1973"]][-1])
    spans = [(None, "1982-12"), ("1983-01", "1999-12"), ("2000-01", "2019-12"), ("2020-01", None)]
    infl = (lab["label"] == "inflation").to_numpy() & masks["1973"]
    eq = R.real_log_return(P["r_us_equity"].to_numpy(float), P["f_cpi"].to_numpy(float))
    bd = R.real_log_return(P["r_treasury_10y"].to_numpy(float), P["f_cpi"].to_numpy(float))
    out = []
    for a, b in spans:
        m = infl & _panel_window(idx, a, b)
        start = a or _ym(idx[masks["1973"]][0])
        n = int(m.sum())
        out.append({"span": _years(start, b or last), "start": start, "end": b or last, "months": n,
                    "episodes": len(R.runs(m)), "thin": len(R.runs(m)) < THIN,
                    "equity_real": R.annualized(float(np.nansum(eq[m])), n) if n else None,
                    "bond_real": R.annualized(float(np.nansum(bd[m])), n) if n else None,
                    "corr_sb": _corr(P["r_us_equity"].to_numpy(float)[m], P["r_treasury_10y"].to_numpy(float)[m])})
    return out


def _corr_by_half(P, lab, w73) -> dict:
    idx = P.index
    eq, bd = P["r_us_equity"].to_numpy(float), P["r_treasury_10y"].to_numpy(float)
    ex = lab["label"].to_numpy(object)
    out = {}
    for hid, a, b in HALVES["1973"]:
        h = w73 & _panel_window(idx, a, b)
        out[hid] = {i: {"months": int((h & (ex == i)).sum()), "episodes": len(R.runs(h & (ex == i))),
                        "corr": _corr(eq[h & (ex == i)], bd[h & (ex == i)])} for i in R.LABEL_IDS}
    return out


def _bucket_of(x: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(x), -1, int)
    with np.errstate(invalid="ignore"):
        for k, (_, lo, hi) in enumerate(CPI_BUCKETS):
            out[(x >= lo) & (x < hi)] = k
    return out


def _corr_by_bucket(P, w73) -> dict:
    """Stock-bond correlation by the previous month's CPI y/y."""
    eq, bd = P["r_us_equity"].to_numpy(float), P["r_treasury_10y"].to_numpy(float)
    bk = _bucket_of(P["s_cpi_yoy"].shift(1).to_numpy(float))
    hist = {}
    for k, (name, _, _) in enumerate(CPI_BUCKETS):
        m = w73 & (bk == k)
        hist[name] = {"months": int(m.sum()), "corr": _corr(eq[m], bd[m])}
    return {"buckets": [b[0] for b in CPI_BUCKETS], "by": "上月 CPI 同比", "history": hist}


# ============================================================================ contraction episodes as equity paths

def episodes_section(P: pd.DataFrame, lab: pd.DataFrame, w73: np.ndarray) -> tuple[list[dict], dict]:
    idx = P.index
    con = (lab["contraction"] == 1).to_numpy()
    r_eq = P["r_us_equity"].to_numpy(float)
    f_cpi = P["f_cpi"].to_numpy(float)
    rl_eq = R.real_log_return(r_eq, f_cpi)
    rl_bd = R.real_log_return(P["r_treasury_10y"].to_numpy(float), f_cpi)
    out = []
    for s, e in R.runs(con):
        cum = np.concatenate([[0.0], np.cumsum(rl_eq[s: e + 1])])   # 0 = the end of the month before the start
        k = int(np.argmin(cum))
        cs = np.concatenate([[0.0], np.cumsum(np.log1p(r_eq[s: e + 1]))])
        w = min(12, e - s + 1)
        sums = cs[w:] - cs[:-w]
        j = int(np.argmin(sums))
        bond = rl_bd[s: e + 1]
        out.append({"label": lab["label"].iloc[s], "start": _ym(idx[s]), "end": _ym(idx[e]), "months": e - s + 1,
                    "ip_decline_pct": float(lab["episode_ip_decline"].iloc[s]), "u_rise_pp": float(lab["u_rise_rt"].iloc[e]),
                    "cpi_mean": float(lab["episode_cpi_mean"].iloc[s]),
                    "equity_real_total": float(np.expm1(cum[-1])), "equity_real_to_low": float(np.expm1(cum[k])),
                    "equity_low_month": _ym(idx[s + k - 1] if k > 0 else idx[s - 1]),
                    "equity_real_after_low": float(np.expm1(cum[-1] - cum[k])),
                    "months_after_low": int(e - (s + k - 1)) if k > 0 else e - s + 1,
                    "bond_real_total": float(np.expm1(bond.sum())) if np.isfinite(bond).all() else None,
                    "worst_window": {"start": _ym(idx[s + j]), "end": _ym(idx[s + j + w - 1]), "months": w,
                                     "value": float(np.expm1(sums[j]))},
                    "path": [float(v) for v in 100.0 * np.exp(cum)]})
    first73 = idx[w73][0]
    stag = [x for x in out if x["label"] == "stagflation" and pd.Timestamp(x["start"]) >= first73]
    after = sum(int(((idx > pd.Timestamp(x["equity_low_month"])) & (idx <= pd.Timestamp(x["end"]))).sum()) for x in stag)
    lead = [x["months_after_low"] for x in stag]
    summary = {"stagflation_after_low_1973": {"after": after, "of": sum(x["months"] for x in stag), "episodes": len(stag)},
               "stagflation_low_before_trough_1973": [min(lead), max(lead)] if lead else None}
    return out, summary


# ============================================================================ factor character

def factors_section(P: pd.DataFrame, lab: pd.DataFrame, w73: np.ndarray) -> dict:
    ref = {}
    for f in CORE + EXTRA_FACTORS:
        x = P.loc[w73, f"f_{f}"].dropna()
        ref[f"f_{f}"] = (float(x.mean()), float(x.std()))
    for s in STATES:
        x = P.loc[w73, f"s_{s}"].dropna()
        ref[f"s_{s}"] = (float(x.mean()), float(x.std()))
    out = {"window": [_ym(P.index[w73][0]), _ym(P.index[w73][-1])],
           "reference": {k: {"mean": m, "sd": s} for k, (m, s) in ref.items()},
           "factors_meta": {f: {"name": FACTOR_NAME.get(f, "CPI 总体通胀（月度）" if f == "cpi" else f),
                                "unit": FACTOR_UNIT.get(f, "Δlog"), "core": f in CORE} for f in CORE + EXTRA_FACTORS},
           "states_meta": {s: {"name": STATE_NAME.get(s, s), "unit": STATE_UNIT.get(s, "")} for s in STATES},
           "table_states": TABLE_STATES, "groups": {}}
    for gid, (kind, key, name, gm, _) in _expost_groups(lab).items():
        if gid == "all":
            continue
        base = w73 & gm
        rr = R.runs(base)
        K = len(rr)
        g = {"kind": kind, "id": key, "name": name, "months": int(base.sum()), "episodes": K, "thin": K < THIN,
             "episode_list": _episode_list(P.index, base), "factors": {}, "states": {}}
        for f in CORE + EXTRA_FACTORS:
            mu, sd = ref[f"f_{f}"]
            S = _mean_sums(P[f"f_{f}"].to_numpy(float), rr)
            tot = S.sum(axis=0)
            m = tot[1] / tot[0] if tot[0] > 0 else np.nan
            bm = _boot(R.rng_for("factors", "1973", gid, f"f_{f}", "z"), K) @ S
            with np.errstate(invalid="ignore", divide="ignore"):
                bz = (bm[:, 1] / bm[:, 0] - mu) / sd
            ci = _ci(bz) if K >= 2 else None
            g["factors"][f] = {"z": _num((m - mu) / sd), "ci90": ci, "sig": bool(K >= THIN and _excludes(ci)),
                               "mean_monthly": _num(m), "mean_annual": _num(12 * m), "months": int(tot[0])}
        for s in STATES:
            mu, sd = ref[f"s_{s}"]
            S = _mean_sums(P[f"s_{s}"].to_numpy(float), rr)
            tot = S.sum(axis=0)
            if tot[0] == 0:
                continue
            m = tot[1] / tot[0]
            bm = _boot(R.rng_for("factors", "1973", gid, f"s_{s}", "mean"), K) @ S
            with np.errstate(invalid="ignore", divide="ignore"):
                ci = _ci(bm[:, 1] / bm[:, 0]) if K >= 2 else None
            g["states"][s] = {"mean": float(m), "ci90": ci, "z": _num((m - mu) / sd),
                              "sig": bool(K >= THIN and _excludes(ci, mu)), "months": int(tot[0])}
        sigf = sorted([f for f in CORE if g["factors"][f]["sig"]], key=lambda f: -abs(g["factors"][f]["z"]))
        g["distinctive_factors"] = [{"factor": f, "name": FACTOR_NAME[f], "z": g["factors"][f]["z"]} for f in sigf[:4]]
        sigs = sorted([s for s in g["states"] if g["states"][s]["sig"]], key=lambda s: -abs(g["states"][s]["z"]))
        g["distinctive_states"] = [{"state": s, "name": STATE_NAME.get(s, s), "z": g["states"][s]["z"],
                                    "mean": g["states"][s]["mean"]} for s in sigs[:5]]
        out["groups"][gid] = g
    return out


def _mean_sums(x: np.ndarray, rr) -> np.ndarray:
    ok = np.isfinite(x)
    xx = np.where(ok, x, 0.0)
    return np.array([[ok[s: e + 1].sum(), xx[s: e + 1].sum()] for s, e in rr]) if rr else np.zeros((0, 2))


# ============================================================================ transitions

def _matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    M = np.zeros((len(R.LABEL_IDS), len(R.LABEL_IDS)), int)
    for x, y in zip(a, b):
        if x in R.CODE and y in R.CODE:
            M[R.CODE[x], R.CODE[y]] += 1
    return M


def _probs(M: np.ndarray) -> list:
    rows = M.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rows > 0, M / np.maximum(rows, 1), np.nan).tolist()


def vintages(panel: pd.DataFrame, t0: int, t1: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Ex-post label codes and contraction flags of months 0..t computed on data through t only, t0..t1.

    One classify() call on every history cut at t, right-aligned (leading months NaN): the rules are causal
    up to the last month, an unfinished last leg runs to the end of its row, and NaN months change no state,
    so each row equals classify() on the cut panel (tests/test_regimes_analysis.py checks it)."""
    cpi, ur, ip = (panel[c].to_numpy(float) for c in R.INPUTS.values())
    T = len(ip)
    ts = np.arange(t0, t1 + 1)

    def right(x):
        out = np.full((len(ts), T), np.nan)
        for i, t in enumerate(ts):
            out[i, T - 1 - t:] = x[: t + 1]
        return out

    r = R.classify(right(cpi), right(ur), right(ip))
    return {int(t): (r["label"][i, T - 1 - t:].astype(np.int64), r["contraction"][i, T - 1 - t:].astype(bool))
            for i, t in enumerate(ts)}


def _walk_forward(panel: pd.DataFrame, lab: pd.DataFrame, t0: int, vint: dict) -> dict:
    """At each origin t: P(next year's ex-post label | real-time label at t) from pairs (s, s+12) with s+12 <= t
    whose targets are ex-post labels computed on data through t (vintage t), scored against the base rate
    estimated the same way (add-0.5 smoothing); and P(any contraction month in t+1..t+12), also for origins
    whose real-time contraction flag is off (a new contraction)."""
    idx = panel.index
    k = len(R.LABEL_IDS)
    a = 0.5
    rt, ex = _codes(lab["label_rt"]), _codes(lab["label"])
    con_ex = (lab["contraction"] == 1).to_numpy()
    hi = (lab["high_infl"] == 1).to_numpy()
    con_rt = (lab["contraction_rt"] == 1).to_numpy()
    valid = ex >= 0
    cf = np.concatenate([[0], np.cumsum(con_ex)])
    tN = len(idx) - 1 - 12
    rows, rows_any = [], []
    for t in range(t0, tN + 1):
        vcode, vcon = vint[t]
        s_all = np.arange(0, t - 12 + 1)
        s_lab = s_all[valid[s_all] & (vcode[s_all + 12] >= 0)]
        Xs, Ys, Cs = rt[s_lab], vcode[s_lab + 12], vcon[s_lab + 12]
        M = np.zeros((k, k))
        np.add.at(M, (Xs, Ys), 1.0)
        p_cond = (M[rt[t]] + a) / (M[rt[t]].sum() + a * k)
        p_base = (M.sum(axis=0) + a) / (M.sum() + a * k)
        cnt_h = np.bincount(Ys[hi[s_lab] == hi[t]], minlength=k).astype(float)
        p_hi = (cnt_h + a) / (cnt_h.sum() + a * k)
        y = ex[t + 12]
        onehot = np.zeros(k)
        onehot[y] = 1.0
        sel, selh, selc = Xs == rt[t], hi[s_lab] == hi[t], con_rt[s_lab] == con_rt[t]
        rows.append({"t": t, "year": idx[t].year, "br_cond": float(((p_cond - onehot) ** 2).sum()),
                     "br_base": float(((p_base - onehot) ** 2).sum()), "br_hi": float(((p_hi - onehot) ** 2).sum()),
                     "ls_cond": np.log(p_cond[y]), "ls_base": np.log(p_base[y]), "ls_hi": np.log(p_hi[y]),
                     "yc": float(con_ex[t + 12]), "pb": (Cs.sum() + 0.5) / (len(s_lab) + 1.0),
                     "pl": (Cs[sel].sum() + 0.5) / (sel.sum() + 1.0), "ph": (Cs[selh].sum() + 0.5) / (selh.sum() + 1.0),
                     "pcr": (Cs[selc].sum() + 0.5) / (selc.sum() + 1.0)})
        # any contraction month in s+1..s+12 (vintage t) for the training pairs; the final labels for t
        s_any = s_all[valid[s_all]]
        cv = np.concatenate([[0], np.cumsum(vcon)])
        Y = ((cv[s_any + 13] - cv[s_any + 1]) > 0).astype(float)
        off = ~con_rt[s_any]
        Yo, s_off = Y[off], s_any[off]

        def cond(x, s, Yv):
            m = x[s] == x[t]
            return (Yv[m].sum() + 0.5) / (m.sum() + 1.0)

        rows_any.append({"t": t, "year": idx[t].year, "y": float(cf[t + 13] - cf[t + 1] > 0), "off": not con_rt[t],
                         "pb": (Y.sum() + 0.5) / (len(Y) + 1.0), "pl": cond(rt, s_any, Y), "ph": cond(hi, s_any, Y),
                         "pc": cond(con_rt, s_any, Y), "pbo": (Yo.sum() + 0.5) / (len(Yo) + 1.0),
                         "plo": cond(rt, s_off, Yo), "pho": cond(hi, s_off, Yo)})
    D, A = pd.DataFrame(rows), pd.DataFrame(rows_any)

    def exact(sub: pd.DataFrame) -> dict:
        res = {"origins": int(len(sub)), "first": _ym(idx[int(sub.t.iloc[0])]), "last": _ym(idx[int(sub.t.iloc[-1])])}
        for key, d in (("label_brier_gain", sub.br_base - sub.br_cond), ("label_logscore_gain", sub.ls_cond - sub.ls_base),
                       ("high_infl_brier_gain", sub.br_base - sub.br_hi),
                       ("label_vs_high_infl_brier_gain", sub.br_hi - sub.br_cond)):
            m, t = _t_mean(d.to_numpy())
            res[key] = {"mean": m, "t_hac": t}
        res["label_brier_skill"] = float(1 - sub.br_cond.mean() / sub.br_base.mean())
        res["high_infl_brier_skill"] = float(1 - sub.br_hi.mean() / sub.br_base.mean())
        yc = sub.yc.to_numpy()
        bb = ((sub.pb - yc) ** 2).mean()
        for key, col in (("contraction_given_rt_label", "pl"), ("contraction_given_high_infl", "ph"),
                         ("contraction_given_contraction_rt", "pcr")):
            bm = ((sub[col] - yc) ** 2).mean()
            _, t = _t_mean(((sub.pb - yc) ** 2 - (sub[col] - yc) ** 2).to_numpy())
            res[key] = {"brier_skill": float(1 - bm / bb), "t_hac": t}
        res["contraction_rate"] = float(yc.mean())
        return res

    def anyrow(sub: pd.DataFrame, base: str, conds: dict) -> dict:
        y = sub.y.to_numpy()
        bb = (sub[base] - y) ** 2
        res = {"origins": int(len(sub)), "rate": float(y.mean())}
        for key, col in conds.items():
            bm = (sub[col] - y) ** 2
            _, t = _t_mean((bb - bm).to_numpy())
            res[key] = {"brier_skill": float(1 - bm.mean() / bb.mean()), "t_hac": t}
        return res

    def halves(frame, fn, *args):
        return {"all": fn(frame, *args), "1973_1999": fn(frame[frame.year <= 1999], *args),
                "2000_end": fn(frame[frame.year >= 2000], *args)}

    off = A[A.off]
    return {"exactly_t12": halves(D, exact),
            "any_12m": halves(A, anyrow, "pb", {"rt_label": "pl", "high_infl": "ph", "contraction_rt": "pc"}),
            "new_contraction": halves(off, anyrow, "pbo", {"rt_label": "plo", "high_infl": "pho"}),
            "first_origin": _ym(idx[t0]), "last_origin": _ym(idx[tN])}


def transitions_section(panel: pd.DataFrame, lab: pd.DataFrame, masks: dict, timings: dict) -> dict:
    idx = panel.index
    ex, rt = lab["label"].to_numpy(object), lab["label_rt"].to_numpy(object)
    con = (lab["contraction"] == 1).to_numpy()
    valid = ex != ""
    first, last = int(np.flatnonzero(valid)[0]), len(idx) - 1
    all_runs = _label_runs(ex, idx, first, last)
    out = {}
    for wid in ("1948", "1973"):
        s0 = max(first, int(np.flatnonzero(masks[wid])[0]))
        start = _ym(idx[s0])
        pos = np.arange(s0, last)
        M1 = _matrix(ex[pos], ex[pos + 1])
        pos12 = np.arange(s0, last - 12 + 1)
        M12rt = _matrix(rt[pos12], ex[pos12 + 12])
        origins = np.zeros(len(idx), bool)
        origins[pos12] = True
        base12 = _matrix(ex[pos12], ex[pos12 + 12]).sum(axis=0)
        dur = {}
        for i in R.LABEL_IDS:
            ls = [r for r in all_runs if r["label"] == i and r["start"] >= start]
            comp = [r["months"] for r in ls if not (r["left_censored"] or r["right_censored"])]
            allm = [r["months"] for r in ls]
            pii = M1[R.CODE[i], R.CODE[i]] / max(M1[R.CODE[i]].sum(), 1)
            dur[i] = {"runs": len(ls), "complete_runs": len(comp), "mean": float(np.mean(comp)) if comp else None,
                      "median": float(np.median(comp)) if comp else None, "max": max(allm) if allm else None,
                      "min": min(allm) if allm else None, "markov_implied": float(1 / (1 - pii)) if pii < 1 else None,
                      "thin": len(ls) < THIN}
        row_ep = {i: len(R.runs(origins & (rt == i))) for i in R.LABEL_IDS}
        pc = {"by_label": {}, "by_flag": {}}
        for i in R.LABEL_IDS:
            m = pos12[rt[pos12] == i]
            mm = np.isin(np.arange(len(idx)), m)
            pc["by_label"][i] = {"origins": int(len(m)), "episodes": len(R.runs(mm)),
                                 "p": float(con[m + 12].mean()) if len(m) else None}
        for f in ("contraction_rt", "high_infl"):
            fl = (lab[f] == 1).to_numpy()
            pc["by_flag"][f] = {}
            for state, v in (("on", True), ("off", False)):
                m = pos12[fl[pos12] == v]
                pc["by_flag"][f][state] = {"origins": int(len(m)), "episodes": len(R.runs(np.isin(np.arange(len(idx)), m))),
                                           "p": float(con[m + 12].mean()) if len(m) else None}
        pc["all"] = {"origins": int(len(pos12)), "p": float(con[pos12 + 12].mean())}
        hits = pos12[(rt[pos12] == "inflation") & (ex[pos12 + 12] == "stagflation")]
        out[wid] = {"start": start, "end": _ym(idx[last]),
                    "monthly": {"counts": M1.tolist(), "probs": _probs(M1), "pairs": int(M1.sum())},
                    "durations": dur,
                    "rt_to_expost_12m": {"counts": M12rt.tolist(), "probs": _probs(M12rt), "pairs": int(M12rt.sum()),
                                         "row_episodes": row_ep, "thin_rows": [i for i in R.LABEL_IDS if row_ep[i] < THIN]},
                    "base_rate_12m": {i: float(base12[R.CODE[i]] / max(base12.sum(), 1)) for i in R.LABEL_IDS},
                    "p_contraction_12m": pc,
                    "inflation_to_stagflation_hits": {"hits": int(len(hits)),
                                                      "years": sorted({int(idx[t].year) for t in hits})}}
    t0 = time.time()
    first73 = int(np.flatnonzero(masks["1973"])[0])
    vint = vintages(panel, first73, len(idx) - 1 - 12)
    out["walk_forward"] = _walk_forward(panel, lab, first73, vint)
    timings["transitions_walk_forward"] = time.time() - t0
    return out


# ============================================================================ forward returns

def _forward_returns(P: pd.DataFrame, a: str, k0: int, k1: int) -> np.ndarray:
    """Compounded real total return over months t+k0..t+k1 of every origin t (NaN if incomplete)."""
    rlr = pd.Series(R.real_log_return(P[f"r_{a}"].to_numpy(float), P["f_cpi"].to_numpy(float)))
    w = k1 - k0 + 1
    return np.expm1(rlr.rolling(w, min_periods=w).sum().shift(-k1).to_numpy())


def _fwd_sums(v: np.ndarray, rr) -> np.ndarray:
    return np.array([[e - s + 1, v[s: e + 1].sum(), (v[s: e + 1] < 0).sum()] for s, e in rr])


def _shifted_masks(m: np.ndarray, min_shift: int = MIN_SHIFT) -> np.ndarray:
    """(n_shifts, N) circular shifts of a boolean mask by min_shift..N - min_shift positions."""
    N = len(m)
    ks = np.arange(min_shift, N - min_shift + 1)
    return m[(np.arange(N)[None, :] - ks[:, None]) % N]


def _rt_groups(lab: pd.DataFrame) -> dict:
    known = (lab["label_rt"] != "").to_numpy()
    g = {"all": ("all", "all", "全部起点", known)}
    for i in R.LABEL_IDS:
        g[f"label:{i}"] = ("label", i, R.NAME[i], (lab["label_rt"] == i).to_numpy())
    for f in RT_FLAGS:
        g[f"flag:{f}"] = ("flag", f, FLAG_NAME[f], (lab[f] == 1).to_numpy() & known)
    return g


def _forward_variant(P: pd.DataFrame, lab: pd.DataFrame, wid: str, win: np.ndarray, vid: str) -> dict:
    idx = P.index
    k0, k1 = FORWARD[vid]
    assets = ASSETS if wid == "1973" else ASSETS_1948
    known = (lab["label_rt"] != "").to_numpy()
    fwd = {a: _forward_returns(P, a, k0, k1) for a in assets}
    out = {"months_ahead": [k0, k1], "groups": {}}
    null_abs, act_abs, tests = [], [], []
    shifted: dict = {}
    for gid, (kind, key, name, gm) in _rt_groups(lab).items():
        gout = {"kind": kind, "id": key, "name": name, "assets": {}}
        for a in assets:
            v = fwd[a]
            orig = win & np.isfinite(v) & known
            sel, rest = orig & gm, orig & ~gm
            rr = R.runs(sel)
            if not rr:
                continue
            rng = R.rng_for("forward", wid, gid, a, vid)
            C = _boot(rng, len(rr))
            S = _fwd_sums(v, rr)
            tot = S.sum(axis=0)
            m = tot[1] / tot[0]
            bs = C @ S
            bm = bs[:, 1] / bs[:, 0]
            cell = {"origins": int(sel.sum()), "episodes": len(rr), "thin": len(rr) < THIN,
                    "first_origin": _ym(idx[np.flatnonzero(orig)[0]]), "last_origin": _ym(idx[np.flatnonzero(orig)[-1]]),
                    "mean": float(m), "ci90_mean": _ci(bm) if len(rr) >= 2 else None,
                    "median": float(np.median(v[sel])), "share_negative": float(tot[2] / tot[0])}
            rrc = R.runs(rest) if gid != "all" else []
            if rrc:
                Sc = _fwd_sums(v, rrc)
                totc = Sc.sum(axis=0)
                mall = (tot[1] + totc[1]) / (tot[0] + totc[0])
                bc = _boot(rng, len(rrc)) @ Sc
                ball = (bs[:, 1] + bc[:, 1]) / (bs[:, 0] + bc[:, 0])
                d = m - mall
                ci = _ci(bm - ball) if len(rr) >= 2 else None
                surv = bool(len(rr) >= THIN and _excludes(ci))
                oo = np.flatnonzero(orig)
                gmo = gm[oo]
                y = v[oo]
                sk = (gid, orig.tobytes())                 # the shifts depend on the origins, not the asset
                if sk not in shifted:
                    shifted[sk] = _shifted_masks(gmo)
                SM = shifted[sk]
                with np.errstate(invalid="ignore", divide="ignore"):
                    dn = (SM @ y) / SM.sum(axis=1) - y.mean()
                pp = float(np.mean(np.abs(dn) >= abs(d) - 1e-12))
                null_abs.append(np.abs(dn))
                act_abs.append(abs(d))
                b, V, _ = _nw(y, np.column_stack([np.ones(len(oo)), gmo.astype(float)]), HAC_FORWARD)
                hv = {}
                for hid, h0, h1 in HALVES[wid]:
                    hm = orig & _panel_window(idx, h0, h1)
                    gs, rs = hm & gm, hm & ~gm
                    hv[hid] = {"diff_vs_rest": float(v[gs].mean() - v[rs].mean()) if gs.sum() and rs.sum() else None,
                               "origins": int(gs.sum()), "episodes": len(R.runs(gs))}
                vals = [h["diff_vs_rest"] for h in hv.values()]
                same = bool(all(x is not None for x in vals) and np.sign(vals[0]) == np.sign(vals[1]) == np.sign(b[1]))
                cell.update({"uncond": float(mall), "diff": float(d), "ci90": ci, "survives": surv,
                             "placebo_p": None if len(rr) < THIN else pp,
                             "diff_vs_rest": float(b[1]), "t_hac": float(b[1] / np.sqrt(V[1, 1])), "halves": hv,
                             "same_sign_both_halves": same, "robust": bool(surv and pp < 0.10 and same)})
                tests.append((f"{gid}|{a}", pp, cell))
            gout["assets"][a] = cell
        out["groups"][gid] = gout
    if tests:
        # the joint placebo: how many (group, asset) differences beat their own placebo 90th percentile, against
        # the same count under every shift (the tests are correlated; the shared shifts carry that)
        n_sh = min(len(x) for x in null_abs)
        Dn = np.vstack([x[:n_sh] for x in null_abs])
        q90 = np.nanpercentile(Dn, 90, axis=1)
        actual = int(np.sum(np.asarray(act_abs) >= q90))
        null_counts = np.sum(Dn >= q90[:, None], axis=0)
        out["global_placebo"] = {"tests": len(tests), "actual": actual, "null_mean": float(null_counts.mean()),
                                 "null_p95": float(np.percentile(null_counts, 95)),
                                 "p": float(np.mean(null_counts >= actual)), "shifts": int(n_sh)}
        # Holm (5%) and Benjamini-Hochberg (10%) over the cells with enough episodes, p floored at the resolution
        eligible = sorted((max(pp, 1.0 / (n_sh + 1)), cid) for cid, pp, c in tests if c["episodes"] >= THIN)
        m = len(eligible)
        holm = 0
        for i, (p, _) in enumerate(eligible):
            if p > 0.05 / (m - i):
                break
            holm += 1
        bh = max([i for i, (p, _) in enumerate(eligible, start=1) if p <= 0.10 * i / m] or [0])
        out.update({"n_tests": len(tests), "eligible_tests": m, "expected_by_chance": 0.10 * m,
                    "n_survive": sum(c["survives"] for _, _, c in tests), "n_robust": sum(c["robust"] for _, _, c in tests),
                    "holm_discoveries": holm, "bh10_discoveries": [cid for _, cid in eligible[:bh]]})
    return out


def forward_section(P: pd.DataFrame, lab: pd.DataFrame, masks: dict) -> dict:
    out = {"lag_months": LAG, "windows": {}}
    for wid in WINDOWS:
        w = _forward_variant(P, lab, wid, masks[wid], "lag1")
        w["base"] = _forward_variant(P, lab, wid, masks[wid], "base")
        out["windows"][wid] = w
    out["mechanism"] = _mechanism(P, lab, masks["1973"])
    out["forecast_test"] = forecast_test_status(_ym(P.index.max()))
    return out


def forecast_test_status(data_end: str) -> dict:
    """Whether results/regime_forecast.json is this build's forecast test: it exists and was computed on the panel
    that ends at `data_end`. That step runs after this one, so the first build of a new data month reports it as
    not available (and states no verdict) until `--only regimes` runs again after it."""
    path = config.RESULTS_DIR / "regime_forecast.json"
    out = {"available": False, "file": path.name, "data_end": None, "adopted": None}
    try:
        rf = jsonio.load(path)
    except (OSError, ValueError):
        return out
    if not isinstance(rf, dict):
        return out
    out["data_end"] = rf.get("data_end")
    if out["data_end"] == data_end and isinstance((rf.get("verdict") or {}).get("adopted"), list):
        out["available"] = True
        out["adopted"] = list(rf["verdict"]["adopted"])
    return out


def _mechanism(P: pd.DataFrame, lab: pd.DataFrame, w73: np.ndarray) -> dict:
    """Real-time 通胀 origins split by whether an ex-post contraction month falls in t..t+12."""
    k0, k1 = FORWARD["lag1"]
    y = _forward_returns(P, "us_equity", k0, k1)
    rt = lab["label_rt"].to_numpy(object)
    con = (lab["contraction"] == 1).to_numpy()
    cs = np.concatenate([[0], np.cumsum(con)])
    T = len(con)
    pre = np.array([cs[min(t + 13, T)] - cs[t] > 0 for t in range(T)])
    orig = w73 & np.isfinite(y) & (rt != "")
    d = orig & (rt == "inflation")
    return {"origins": int(d.sum()), "within_12m_of_contraction": int((d & pre).sum()),
            "mean_within": float(y[d & pre].mean()), "mean_not": float(y[d & ~pre].mean()), "uncond": float(y[orig].mean()),
            "not_minus_uncond": float(y[d & ~pre].mean() - y[orig].mean()),
            "all_minus_uncond": float(y[d].mean() - y[orig].mean()), "all_origins": int(orig.sum()),
            "first_origin": _ym(P.index[np.flatnonzero(orig)[0]]), "last_origin": _ym(P.index[np.flatnonzero(orig)[-1]])}


# ============================================================================ eras

def eras_section(P: pd.DataFrame, lab: pd.DataFrame) -> list[dict]:
    idx = P.index
    ex, rt = lab["label"], lab["label_rt"]
    spans = [{"id": "pre_1955", "name": "1948–1954（早于历史时代划分，仅供参考）", "start": CFG["first_month"],
              "end": "1954-12"}]
    spans += [{"id": e["id"], "name": e["name"], "start": e["start"], "end": e["end"]} for e in config.ERAS]
    out = []
    for e in spans:
        m = _panel_window(idx, e["start"], e["end"]) & (ex != "").to_numpy()
        if not m.any():
            continue
        n = int(m.sum())
        ve, vr = ex[m].value_counts(), rt[m].value_counts()
        pos = np.flatnonzero(m)
        out.append({"id": e["id"], "name": e["name"], "start": e["start"], "end": e["end"] or _ym(idx[pos[-1]]), "months": n,
                    "dominant": max(R.LABEL_IDS, key=lambda i: ve.get(i, 0)),
                    "expost": {i: float(ve.get(i, 0) / n) for i in R.LABEL_IDS},
                    "realtime": {i: float(vr.get(i, 0) / n) for i in R.LABEL_IDS},
                    "flags": {f: float(lab.loc[m, f].mean()) for f in ("contraction", "severe", "high_infl", "cpi_neg",
                                                                        "contraction_rt")},
                    "runs_expost": [f"{r['start']}..{r['end']} {R.NAME[r['label']]}"
                                    for r in _label_runs(ex.to_numpy(object), idx, int(pos[0]), int(pos[-1]))]})
    return out


# ============================================================================ transmission by inflation state

# what the transmission's interaction term switches, in the model-use line (FACTORS_DESIGN §7.4)
TERM_CHANGE = {"tp10": "期限溢价变化"}


def transmission_switch(spec) -> dict | None:
    """The selected transmission's own high-inflation switch (FACTORS_DESIGN §7.3), for engine/regimes.switch_text:
    its first interaction factor (config.FACTORS "interaction": the term premium in high core inflation), else the
    whole-model split of ModelSpec(regime="inflation") (previous-month headline CPI y/y above
    factor_model.HIGH_INFLATION), else None (no inflation state in the model)."""
    info = {f["id"]: f for f in config.FACTORS}
    extras = [f for _, ids in (getattr(spec, "asset_factors", ()) or ()) for f in ids]
    for fid in (*spec.factors, *extras):
        rule = info.get(fid, {}).get("interaction")
        if rule:
            return {"kind": "interaction", "id": fid, **rule}
    if spec.regime == "inflation":
        return {"kind": "regime", "factor": None, "state": "cpi_yoy", "lag": 1, "above": FM.HIGH_INFLATION}
    return None


def transmission_section(ctx, lab: pd.DataFrame) -> dict:
    """The selected transmission's betas in high- vs low-inflation months (the engine's previous-month
    dummy) and in contraction months, by HAC Wald tests: a description; the model itself has no regime term.

    Each asset is tested on its own equation: its features (asset_feature_names: a dedicated extra enters
    only its own asset) and its own fit months (asset_train_start: gold may start later), so the
    full-sample check reproduces the engine's coefficients."""
    P = ctx.panel
    eng = ctx.model
    spec = eng.spec
    assets = ["us_equity", "nasdaq", "gold", "real_estate"]
    names = FM.feature_names(spec)
    X_all = FM.design_matrix(P, spec)
    rows = X_all.loc[eng.train_start: eng.train_end].index
    Yall = P.loc[rows, [f"{spec.target}_{a}" for a in eng.assets]]
    ok = X_all.loc[rows].notna().all(axis=1) & Yall.notna().all(axis=1)
    rows = rows[ok.to_numpy()]
    X = X_all.loc[rows].to_numpy(float)
    n = len(rows)
    D_hi = FM.design_matrix(P, replace(spec, regime="inflation")).loc[rows, "hi_infl"].to_numpy(float)
    con = (lab.loc[rows, "contraction"] == 1).to_numpy(float)
    own = {}
    for a in assets:
        nm_a = FM.asset_feature_names(spec, a)
        first = (eng.asset_train_start or {}).get(a)
        r_a = np.asarray(rows >= pd.Timestamp(first)) if first else np.ones(n, bool)
        cols = [names.index(f) for f in nm_a]
        own[a] = {"names": nm_a, "cols": cols, "rows": r_a, "sd": X[r_a][:, cols].std(axis=0),
                  "labor": np.array([f.split("_l")[0] == "labor_share" for f in nm_a])}
    check = 0.0
    for a in assets:
        o = own[a]
        Xa = X[o["rows"]][:, o["cols"]]
        b, _, _ = _nw(P.loc[rows[o["rows"]], f"{spec.target}_{a}"].to_numpy(float),
                      np.column_stack([np.ones(len(Xa)), Xa]), HAC_MONTHLY)
        check = max(check, float(np.max(np.abs(b[1:] - eng.coef.loc[a, o["names"]].to_numpy(float)))))

    def split(D, mask, sid, desc):
        m = mask.astype(bool)
        nm = int(m.sum())
        res = {"id": sid, "desc": desc, "months": nm, "months_high": int(D[m].sum()), "months_low": int(nm - D[m].sum()),
               "first": _ym(rows[m][0]), "last": _ym(rows[m][-1]), "assets": {}}
        n_sig = n_sig_x = n_tests = n_tests_x = 0
        for a in assets:
            o = own[a]
            ma = m & o["rows"]
            Xm, Dm = X[ma][:, o["cols"]], D[ma]
            k, nma = Xm.shape[1], int(ma.sum())
            if Dm.sum() < 3 * (k + 2) or (nma - Dm.sum()) < 3 * (k + 2):
                res["skipped"] = f"高、低两组里有一组的月份太少，估不出 {k + 1} 个系数"
                res["assets"] = {}
                return res
            Z = np.column_stack([np.ones(nma), Xm, Dm, Xm * Dm[:, None]])
            y = P.loc[rows, f"{spec.target}_{a}"].to_numpy(float)[ma]
            b, V, _ = _nw(y, Z, HAC_MONTHLY)
            bl, dd = b[1: k + 1], b[k + 2:]
            tt = dd / np.sqrt(np.diag(V)[k + 2:])
            w10 = float(dd @ np.linalg.pinv(V[k + 2:, k + 2:]) @ dd)
            sd_f = o["sd"]
            per = {f: {"beta_low": float(bl[j]), "beta_high": float(bl[j] + dd[j]), "diff": float(dd[j]),
                       "t_hac": float(tt[j]), "p": float(2 * sps.norm.sf(abs(tt[j]))),
                       "beta_low_1sd_pct": float(100 * bl[j] * sd_f[j]),
                       "beta_high_1sd_pct": float(100 * (bl[j] + dd[j]) * sd_f[j])} for j, f in enumerate(o["names"])}
            n_sig += int((np.abs(tt) >= 1.96).sum())
            n_sig_x += int((np.abs(tt[~o["labor"]]) >= 1.96).sum())
            n_tests += k
            n_tests_x += int((~o["labor"]).sum())
            res["assets"][a] = {"factors": per, "wald_slopes": {"chi2": w10, "df": k, "p": float(sps.chi2.sf(w10, k))},
                                "n_sig_5pct": int((np.abs(tt) >= 1.96).sum()), "months": nma,
                                "first": _ym(rows[ma][0])}
        res.update({"wald_p": {a: res["assets"][a]["wald_slopes"]["p"] for a in assets},
                    "n_sig_5pct": n_sig, "n_tests": n_tests, "expected_5pct": 0.05 * n_tests,
                    "n_sig_excl_labor_share": n_sig_x, "n_tests_excl_labor_share": n_tests_x,
                    "expected_5pct_excl_labor_share": 0.05 * n_tests_x})
        if "y10" in own["us_equity"]["names"]:
            f = res["assets"]["us_equity"]["factors"]["y10"]
            res["equity_y10"] = {"low": f["beta_low_1sd_pct"], "high": f["beta_high_1sd_pct"], "t": f["t_hac"]}
        return res

    yrs = rows.year
    hi = FM.HIGH_INFLATION
    # the headline previous-month dummy is the whole-model inflation split (ModelSpec(regime="inflation")): the
    # transmission's own state only when the selected model is that split (the walk-forward rejected it)
    whose = "传导模型的整体通胀状态交互" if spec.regime == "inflation" else "整体通胀状态交互，走步检验没有采用"
    splits = [split(D_hi, np.ones(n, bool), "inflation_prev_month", f"上月 CPI 同比 > {hi:g}%（{whose}），全样本"),
              split(D_hi, yrs <= 1999, "inflation_prev_month_1973_1999", "同上，只用 1999 年及以前的月份"),
              split(D_hi, yrs >= 2000, "inflation_prev_month_2000_end", "同上，只用 2000 年以后的月份"),
              split(con, np.ones(n, bool), "contraction_expost", "收缩月份（事后）对非收缩月份，全样本")]
    labor_in = any(o["labor"].any() for o in own.values())
    return {"fit_rows": [_ym(rows[0]), _ym(rows[-1])], "months": int(n), "assets": assets, "features": names,
            "asset_features": {a: own[a]["names"] for a in assets},
            "asset_fit_start": {a: _ym(rows[own[a]["rows"]][0]) for a in assets},
            "target": spec.target, "hac_lags": HAC_MONTHLY, "engine_coef_check_max_abs_diff": check,
            "labor_share_in_model": bool(labor_in),
            "labor_share_note": "劳动收入占比按发布月对齐，只在发布月变化" + ("，单列计数" if labor_in else "；它不在今天的传导模型里"),
            "splits": splits, "walk_forward_verdict": _diagnostics_verdict()}


def _diagnostics_verdict() -> dict | None:
    path = config.RESULTS_DIR / "diagnostics.json"
    if not path.exists():
        return None
    rec = next((r for r in jsonio.load(path).get("recommendations", []) if "通胀状态依赖" in r.get("title", "")), None)
    if rec is None:
        return None
    ev = rec.get("evidence") or {}
    return {"title": rec.get("title"), "detail": rec.get("detail"), "oos_r2_delta": ev.get("oos_delta"),
            "n_reject": ev.get("n_reject"), "changes": ev.get("changes") or []}


# ============================================================================ the engine side

def _support(db) -> dict:
    """What a 12-month phase with one view at 1.0 on each regime column can draw (pool_phase as run_spec calls
    it), and how many historical episodes that weight rests on."""
    prior = db.prior()
    out = {}
    for rid, col in SUPPORT_IDS.items():
        if col not in db.frame:
            continue
        x = np.nan_to_num(db.frame[col].to_numpy(float))
        rr = R.runs(x > 0.5)
        _, hi = view_support(db.frame, col, 12, BLOCK)
        pp = pool_phase(db.frame, [{"var": col, "mean": 1.0}], prior, 12, BLOCK, 1.0, rare=db.rare)
        w = pp.month_weights
        shares = np.array([w[a: b + 1].sum() for a, b in rr])
        pr = shares / shares.sum() if shares.sum() > 0 else shares
        pr = pr[pr > 0]
        eff = float(np.exp(-(pr * np.log(pr)).sum())) if pr.size else 0.0
        out[rid] = {"name": R.NAME.get(rid) or FLAG_NAME.get(rid, rid), "column": col, "months": int((x > 0.5).sum()),
                    "runs": len(rr), "support_hi": float(hi), "max_delivered": float(w @ x), "shrink": float(pp.shrink),
                    "reason": pp.shrink_reason, "ens": float(pp.ens),
                    "engine_episodes": None if pp.episodes is None else float(pp.episodes),
                    "regime_effective_episodes": eff,
                    "thin": bool(len(rr) < CFG["thin_runs"] or eff < CFG["thin_runs"] - 0.5),
                    "episodes": sorted([{"span": f"{_ym(db.dates[a])}..{_ym(db.dates[b])}", "weight": float(s)}
                                        for (a, b), s in zip(rr, shares)], key=lambda d: -d["weight"])}
    return out


def _run_lengths(rows: np.ndarray) -> tuple[dict, float]:
    """Mean run length per label and label changes per year over the rows of a code array."""
    lens = {k: [] for k in range(len(R.LABEL_IDS))}
    changes = months = 0
    for seq in np.atleast_2d(rows):
        e = np.flatnonzero(np.diff(seq) != 0)
        starts = np.concatenate([[0], e + 1])
        ends = np.concatenate([e, [len(seq) - 1]])
        changes += len(starts) - 1
        months += len(seq)
        for s, t in zip(starts, ends):
            if seq[s] >= 0:
                lens[int(seq[s])].append(t - s + 1)
    return ({R.LABEL_IDS[k]: (float(np.mean(v)) if v else None) for k, v in lens.items()},
            12.0 * changes / months if months else float("nan"))


def _contraction_runs(contr: np.ndarray, decline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L, D = [], []
    for i in range(contr.shape[0]):
        for s, e in R.runs(contr[i]):
            L.append(e - s + 1)
            D.append(float(np.nanmax(decline[i, s: e + 1])))
    return np.array(L, float), np.array(D, float)


def _pct(x: np.ndarray) -> dict:
    return {str(p): float(np.percentile(x, p)) for p in (10, 50, 90)}


def _group_corr(eq, bd, groups: np.ndarray, min_months: int) -> dict:
    out = {}
    for k, i in enumerate(R.LABEL_IDS):
        m = (groups == k) & np.isfinite(eq) & np.isfinite(bd)
        out[i] = {"months": int(m.sum()), "corr": float(np.corrcoef(eq[m], bd[m])[0, 1]) if m.sum() >= min_months else None}
    return out


def _bucket_corr(eq, bd, prev_cpi, min_months: int) -> dict:
    bk = _bucket_of(prev_cpi)
    out = {}
    for k, (name, _, _) in enumerate(CPI_BUCKETS):
        m = (bk == k) & np.isfinite(eq) & np.isfinite(bd)
        out[name] = {"months": int(m.sum()), "corr": float(np.corrcoef(eq[m], bd[m])[0, 1]) if m.sum() >= min_months else None}
    return out


def _spread(d: dict) -> float | None:
    v = [x["corr"] for x in d.values() if x["corr"] is not None]
    return float(v[-1] - v[0]) if len(v) >= 2 else None


def reference_section(ctx, lab: pd.DataFrame) -> tuple[dict, dict, dict, list, float]:
    """The engine's unconditional reference run (runner.reference_simulation) against the MonthDB months."""
    from ..scenarios import runner

    ref = runner.reference_simulation(ctx)
    sim, r = ref["sim"], ref["regimes"]
    s_lab, s_rt, s_con = r["label"][:, 1:], r["label_rt"][:, 1:], r["contraction"][:, 1:]
    drawn = ref["drawn"]
    windows = {}
    for w, (a, b) in {"year1": (0, 12), "years2_5": (12, 60), "years6_15": (60, 180), "whole": (0, 180),
                      "m25_180": (24, 180)}.items():
        windows[w] = {"simulated": R.shares(s_lab[:, a:b]), "simulated_rt": R.shares(s_rt[:, a:b]),
                      "drawn": R.shares(drawn[:, a:b]), "paths_with_contraction": float(s_con[:, a:b].any(axis=1).mean()),
                      "contraction_share": float(s_con[:, a:b].mean())}
    cpi = np.asarray(sim.macro["cpi_yoy"], float)
    later = cpi[:, 1:][:, 24:]
    runs_mean, changes = _run_lengths(s_lab[:, 24:])
    L, D = _contraction_runs(s_con, r["episode_ip_decline"][:, 1:])
    sev = CFG["params"]["severe_ip_decline_pct"]
    hi = CFG["params"]["high_inflation_pct"]
    reference = {"n_paths": ref["n_paths"], "seed": ref["seed"], "months": ref["months"], "windows": windows,
                 "anchor": R.sim_anchor(r), "cpi_gt4_share": float((later > hi).mean()),
                 "cpi_gt4_share_year1": float((cpi[:, 1:13] > hi).mean()), "cpi_pct": _pct(later),
                 "label_changes_per_year": changes, "mean_run_months": runs_mean,
                 "contraction_episodes": {"per_path": float(L.size / s_lab.shape[0]),
                                          "length_mean": float(L.mean()) if L.size else None,
                                          "share_longer_than_18m": float((L > LONGEST).mean()) if L.size else None,
                                          "share_severe": float((D >= sev - R.EPS).mean()) if D.size else None},
                 "window_months": {"m25_180": "第 25–180 个月（起点状态已经消退）"}}
    # history: the month database
    db = ctx.db
    codes = R.label_codes(db.frame)
    cpi_h = db.frame["s_cpi_yoy"].to_numpy(float)
    con_h = np.nan_to_num(db.frame["rg_contraction"].to_numpy(float)) > 0.5
    dec_h = lab["episode_ip_decline"].reindex(db.dates).to_numpy(float)
    hruns, hchanges = _run_lengths(codes[None, :])
    Lh, Dh = _contraction_runs(con_h[None, :], dec_h[None, :])
    base_rate = {}
    for m in (6, 12, 18, 24, 30, 36, 48, 60):
        starts = np.arange(0, len(con_h) - m + 1)
        cs = np.concatenate([[0], np.cumsum(con_h)])
        base_rate[str(m)] = float(((cs[starts + m] - cs[starts]) > 0).mean())
    history = {"months": int(len(db)), "first": _ym(db.dates[0]), "last": _ym(db.dates[-1]), "shares": R.shares(codes),
               "cpi_gt4_share": float((cpi_h > hi).mean()), "cpi_pct": _pct(cpi_h[np.isfinite(cpi_h)]),
               "label_changes_per_year": hchanges, "mean_run_months": hruns,
               "contraction_episodes": {"episodes": int(Lh.size), "per_15y": float(Lh.size * 180 / len(db)),
                                        "length_mean": float(Lh.mean()) if Lh.size else None,
                                        "longer_than_18m": int((Lh > LONGEST).sum()),
                                        "severe": int((Dh >= sev - R.EPS).sum())},
               "contraction_base_rate_by_window_months": base_rate}
    # co-movement
    eq_i, bd_i = sim.assets.index("us_equity"), sim.assets.index("treasury_10y")
    eq, bd = np.asarray(sim.total[:, :, eq_i], float), np.asarray(sim.total[:, :, bd_i], float)
    eh, bh = db.frame["r_us_equity"].to_numpy(float), db.frame["r_treasury_10y"].to_numpy(float)
    prev_h = ctx.panel["s_cpi_yoy"].shift(1).reindex(db.dates).to_numpy(float)
    como = {"history_by_label": _group_corr(eh, bh, codes, MIN_CORR_MONTHS),
            "reference_by_drawn_label": _group_corr(eq.ravel(), bd.ravel(), drawn.ravel(), MIN_SIM_CORR_MONTHS),
            "reference_by_sim_label": _group_corr(eq.ravel(), bd.ravel(), s_lab.ravel(), MIN_SIM_CORR_MONTHS),
            "history_by_cpi_bucket": _bucket_corr(eh, bh, prev_h, MIN_CORR_MONTHS),
            "reference_by_cpi_bucket": _bucket_corr(eq.ravel(), bd.ravel(), cpi[:, :-1].ravel(), MIN_SIM_CORR_MONTHS),
            "buckets": [b[0] for b in CPI_BUCKETS]}
    como["history_bucket_spread"] = _spread(como["history_by_cpi_bucket"])
    como["reference_bucket_spread"] = _spread(como["reference_by_cpi_bucket"])
    ct = np.zeros((len(R.LABEL_IDS), len(R.LABEL_IDS)), int)
    ok = (drawn >= 0) & (s_lab >= 0)
    np.add.at(ct, (drawn[ok].astype(int), s_lab[ok].astype(int)), 1)
    k_inf, k_exp = R.CODE["inflation"], R.CODE["expansion"]
    share = float(ct[k_exp, k_inf] / ct[:, k_inf].sum()) if ct[:, k_inf].sum() else None
    return reference, history, como, ct.tolist(), share


def _hist_worst_ip(panel: pd.DataFrame, months: int) -> float:
    ip = panel["lvl_ip"].to_numpy(float)
    return float(np.nanmin(100.0 * (ip[months:] / ip[:-months] - 1.0)))


def _study_run(ctx, views: list[dict], months: int, seed: int):
    """(detail, sim, regime arrays) of a one-phase spec through runner.run_spec."""
    from ..scenarios import runner

    spec = {"id": "regime_study", "name": "宏观体制研究", "prior_prob": 0.0,
            "phases": [{"name": "regime", "months": months, "views": views, "confidence": 1.0}]}
    detail, sim = runner.run_spec(ctx, spec, n_paths=STUDY_PATHS, seed=seed, source="test", attribution=False,
                                  regimes=False)
    used = {v["var"] for v in detail["phases"][0]["ep"]["views"]}
    missing = [v["var"] for v in views if v["var"] not in used]
    if missing:
        raise RuntimeError(f"the scenario schema dropped the views {missing}: rg_ columns must be allowed in views")
    r = R.classify_sim(sim.macro, ctx.panel, anchor=ctx.anchor.date)
    return detail, sim, r


def studies_section(ctx) -> dict:
    """What regime views deliver in simulated paths (drawn is not simulated), how expected contraction months
    compare with history, and a regime request that conflicts with a built-in phase's own views."""
    db = ctx.db
    codes = R.label_codes(db.frame)
    contr_db = np.nan_to_num(db.frame["rg_contraction"].to_numpy(float)) > 0.5
    rows = []
    for sid, name, views, target in STUDIES:
        for m in STUDY_MONTHS:
            detail, sim, r = _study_run(ctx, views, m, STUDY_SEED)
            ep = detail["phases"][0]["ep"]
            if target == "contraction":
                drawn = float(contr_db[sim.idx][:, :m].mean())
                simulated = float(r["contraction"][:, 1: m + 1].mean())
            else:
                k = R.CODE[target]
                drawn = float((codes[sim.idx][:, :m] == k).mean())
                simulated = float((r["label"][:, 1: m + 1] == k).mean())
            ip, ur = np.asarray(sim.macro["ip"], float), np.asarray(sim.macro["unrate"], float)
            d_ip = 100.0 * (ip[:, m] / ip[:, 0] - 1.0)
            rows.append({"case": sid, "name": name, "views": views, "target": target, "months": m,
                         "drawn_share": drawn, "simulated_share": simulated, "lambda": float(ep["shrink"]),
                         "ens": float(ep["ens"]), "ip_change_p50": float(np.median(d_ip)),
                         "unrate_end_p50": float(np.median(ur[:, m])),
                         "paths_ip_fall_beyond_hist": float((d_ip < _hist_worst_ip(ctx.panel, m)).mean()),
                         "simulated_mix": R.shares(r["label"][:, 1: m + 1])})
    duration = []
    for m in DURATION_MONTHS:
        detail, sim, r = _study_run(ctx, [{"var": "rg_contraction", "mean": 1.0}], m, DURATION_SEED)
        ip = np.asarray(sim.macro["ip"], float)
        d_ip = 100.0 * (ip[:, m] / ip[:, 0] - 1.0)
        worst = _hist_worst_ip(ctx.panel, m)
        duration.append({"share": 1.0, "months": m, "expected_contraction_months": float(m), "ip_change_p50": float(np.median(d_ip)),
                         "paths_ip_fall_beyond_hist": float((d_ip < worst).mean()), "hist_worst_ip_change": worst,
                         "simulated_contraction_share": float(r["contraction"][:, 1: m + 1].mean())})
    return {"n_paths": STUDY_PATHS, "seed": STUDY_SEED, "duration_seed": DURATION_SEED,
            "drawn_vs_simulated": rows, "duration": duration, "conflict_example": conflict_example(ctx),
            "longest_contraction_months": LONGEST}


def conflict_example(ctx) -> dict:
    """A regime request on top of a built-in phase's own views: pooled alone, together, and leave one out."""
    from ..scenarios import library
    from ..scenarios.schema import validate_spec

    db = ctx.db
    prior = db.prior()
    spec = next(s for s in library.resolved_specs(ctx) if s.get("id") == CONFLICT["spec"])
    clean, _ = validate_spec(spec, db.frame, config.HORIZON_MONTHS, model_spec=ctx.model.spec)
    ph = clean["phases"][CONFLICT["phase"]]
    rviews = [{"var": f"rg_{i}", "mean": float(s)} for i, s in CONFLICT["regime"].items()]
    others = [v for v in ph["views"] if v["var"] not in {r["var"] for r in rviews}]

    def pool(views):
        return pool_phase(db.frame, views, prior, ph["months"], BLOCK, 1.0, rare=db.rare)

    alone, full = pool(rviews), pool(rviews + others)
    x = np.nan_to_num(db.frame[rviews[0]["var"]].to_numpy(float))
    loo = []
    for v in others:
        lam = float(pool(rviews + [u for u in others if u is not v]).shrink)
        loo.append({"var": v["var"], "lambda_without": lam})
    loo.sort(key=lambda d: -d["lambda_without"])
    return {"spec": CONFLICT["spec"], "spec_name": clean["name"], "phase": CONFLICT["phase"], "phase_name": ph["name"],
            "months": ph["months"], "regime": CONFLICT["regime"], "lambda_alone": float(alone.shrink),
            "lambda": float(full.shrink), "reason": full.shrink_reason, "drawn_share": float(full.month_weights @ x),
            "drawn_share_alone": float(alone.month_weights @ x),
            "other_views": [{"var": v["var"], **{k: v.get(k) for k in ("mean", "mean_annual") if v.get(k) is not None}}
                            for v in others],
            "leave_one_out": loo,
            "culprits": [d["var"] for d in loo[:2] if d["lambda_without"] - full.shrink >= LEAVE_ONE_OUT_MIN_GAIN]}


def engine_section(ctx, lab: pd.DataFrame, timings: dict) -> dict:
    t0 = time.time()
    support = _support(ctx.db)
    timings["engine_support"] = time.time() - t0
    t0 = time.time()
    reference, history, como, ct, share = reference_section(ctx, lab)
    timings["engine_reference"] = time.time() - t0
    t0 = time.time()
    studies = studies_section(ctx)
    timings["engine_studies"] = time.time() - t0
    return {"support": support, "reference": reference, "history": history, "comovement": como,
            "drawn_vs_sim_crosstab": ct, "simulated_inflation_from_expansion_share": share, "studies": studies}


# ============================================================================ definitions and words

def interpolations(panel: pd.DataFrame) -> list[dict]:
    """Months of the label inputs the source does not have, filled by data/panel.py's interpolation (read
    from the cached FRED files; nothing is downloaded)."""
    found: dict[str, list] = {}
    for key, (sid, col, _, _) in RAW_INPUTS.items():
        path = config.RAW_DIR / "fred" / f"{sid}.csv"
        if col not in panel or not path.exists():
            continue
        df = pd.read_csv(path)
        s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(df.iloc[:, 0])).dropna()
        s.index = s.index.to_period("M").to_timestamp()
        lvl = panel[col]
        inside = (lvl.index > s.index.min()) & (lvl.index < s.index.max()) & lvl.notna().to_numpy()
        for m in lvl.index[inside & ~lvl.index.isin(s.index)]:
            found.setdefault(_ym(m), []).append((key, float(lvl.loc[m])))
    out = []
    for month, items in sorted(found.items()):
        names = [RAW_INPUTS[k][2] for k, _ in items]
        names = "、".join(names[:-1]) + "和" + names[-1] if len(names) > 1 else names[0]
        vals = "、".join(f"{RAW_INPUTS[k][3]} {v:.3f}" if k != "unrate" else f"失业率 {v:.2f}%" for k, v in items)
        cause = f"（{KNOWN_GAPS[month]}）" if month in KNOWN_GAPS else ""
        how = f"源数据缺少 {month} 的这些读数：{names}{cause}。面板用前后月份线性插值：{vals}。这个月的标签因此用到了下一个月的数据"
        if any(k == "unrate" for k, _ in items):
            how += "，这个月和之后两个月的失业率 3 个月均值也是"
        out.append({"month": month, "series": [k for k, _ in items], "values": {k: v for k, v in items}, "how": how + "。"})
    return out


def conventions(masks: dict, idx, switch: dict | None = None) -> dict:
    """The conventions every number of the page shares; `switch`: the selected transmission's high-inflation
    switch (transmission_switch), named next to the label's."""
    p = CFG["params"]
    w73, w48 = idx[masks["1973"]], idx[masks["1948"]]
    hl = CFG["hac_lags"]
    return {
        "window_1973": {"name": "“1973 年以来”", "text": f"引擎的历史月份数据库：{_ym(w73[0])} 至 {_ym(w73[-1])}，共 {len(w73)} 个月"
                                                       "（1973-01 缺少贸易加权美元指数）；13 类资产都有数据"},
        "window_1948": {"name": "“1948 年以来”", "text": f"{_ym(w48[0])} 至 {_ym(w48[-1])}，共 {len(w48)} 个月；资产只有美股、"
                                                       "10 年期美债（数据从 1953-05 起）和现金，黄金和原油在 1973 年以前是固定或管制价格，不使用"},
        "real_return": {"name": "实际回报", "text": "每月实际对数回报 = ln(1 + 名义总回报) − CPI 月度对数变化；年化 = exp(12 × 月均值) − 1"
                                               "（几何、连乘），12 个月远期回报 = exp(窗口内求和) − 1；不报告算术版本"},
        "forward_window": {"name": "远期窗口", "text": f"第 t 月的数据和标签在 t+1 月才公布，所以远期窗口是 t+{1 + LAG} 到 t+{12 + LAG} 月"
                                                   "（按发布时间）；t+1 到 t+12 月的版本作为敏感性另列（标为“按数据月份”）"},
        "episodes": {"name": "段数", "text": "同一标签（或标志）连续的月份算一段；只要有一个月落在窗口内就计入（被窗口起点截断的段算一次）；"
                                          "持续时间只用起点在窗口内、没有被样本两端截断的段"},
        "thin": {"name": "†（历史太薄）", "text": f"少于 {THIN} 段历史的格子、行或相关系数标 †，不给置换检验 p 值；少于 2 段时不给区间"},
        "intervals": {"name": "90% 区间", "text": f"整段重抽样的自助法：按段有放回地抽 {B} 次；每个格子用自己的随机数种子"
                                                "（由章节、窗口、分组、资产和统计量决定），结果与计算顺序无关"},
        "correlations": {"name": "相关系数", "text": f"月度总回报的皮尔逊相关；少于 {MIN_CORR_MONTHS} 个月不报告"},
        "hac": {"name": "HAC t 值", "text": f"Newey–West：月度回归 {hl['monthly']} 阶滞后，重叠的 12 个月窗口 {hl['forward']} 阶"},
        "placebo": {"name": "置换检验", "text": f"把实时标签序列循环平移至少 {MIN_SHIFT} 个月（保留标签的段结构和回报的自相关，"
                                             "确定性，不用随机数），看平移后的差异有多大"},
        "high_inflation_timing": {"name": "高通胀的时点", "text": R.switch_text(switch, p)},
        "names": {"name": "体制和标志", "text": "六种体制：" + "、".join(R.NAME[i] for i in R.LABEL_IDS) + "；标志："
                                           + "、".join(d["name"] for d in R.flag_defs())},
        "rounding": {"name": "取整", "text": "占比取整到 1%，回报保留 1 位小数（%），相关系数和 z 值保留 2 位"},
        "data_vintage": {"name": "数据版本", "text": "用今天的修订版数据：实时版只保证单边、按发布滞后，不是当年的首发值（单边、修订后数据）"},
    }


def _fmt_pct(x: float, signed: bool = True) -> str:
    return f"{100 * x:+.1f}%" if signed else f"{100 * x:.1f}%"


def _runs_text(spans: list[str]) -> str:
    return "、".join(spans)


def definitions_section(panel: pd.DataFrame, lab: pd.DataFrame, masks: dict, parts: dict,
                        switch: dict | None = None) -> dict:
    p = CFG["params"]
    hi, neg, sev, sw = p["high_inflation_pct"], p["cpi_neg_pct"], p["severe_ip_decline_pct"], p["ip_swing_pct"]
    idx = panel.index
    w48 = masks["1948"]
    ex = lab["label"].to_numpy(object)
    high = (lab["high_infl"] == 1).to_numpy() & w48
    con = (lab["contraction"] == 1).to_numpy()
    eps = parts["episodes"]
    counts = parts["counts"]
    # 通胀: high-inflation months by where they fall
    in_other = high & np.isin(ex, ("recession", "depression"))
    by_ep = []
    for s, e in R.runs(con & w48):
        k = int(in_other[s: e + 1].sum())
        if k:
            by_ep.append(f"{idx[s].year} 年 {k} 个月")
    infl_text = (f"CPI 同比高于 {hi:g}% 就是高通胀：1948 年以来 {int(high.sum())} 个月。没有收缩的 "
                 f"{int((high & (ex == 'inflation')).sum())} 个月叫通胀；滞胀期里有 {int((high & (ex == 'stagflation')).sum())} 个月")
    infl_text += (f"；另外 {int(in_other.sum())} 个月在平均通胀不高的收缩里（{'、'.join(by_ep)}）。" if in_other.any() else "。")
    # 通缩 -> 物价负增长
    neg_m = (lab["cpi_neg"] == 1).to_numpy() & w48
    defl = (ex == "deflation") & w48
    defl_runs = _episode_list(idx, defl)
    a48 = parts["assets"]["windows"]["1948"]["groups"].get("label:deflation", {}).get("assets", {}).get("us_equity")
    eq_txt = f"（这几段美股实际年化 {_fmt_pct(a48['ann_real']['value'])}）" if a48 and a48["ann_real"]["value"] is not None else ""
    defl_text = (f"数据里叫物价负增长：未收缩且 CPI 同比低于 {neg:g}。1948 年以来 CPI 同比为负的 {int(neg_m.sum())} 个月里，"
                 f"{int(defl.sum())} 个月没有收缩，分 {len(defl_runs)} 段：{_runs_text(defl_runs)}；它们是衰退之后的复苏期或油价暴跌后的"
                 f"基数效应，不是债务型通缩{eq_txt}。")
    # 滞胀
    stag = [x for x in eps if x["label"] == "stagflation" and pd.Timestamp(x["start"]) >= idx[w48][0]]
    w7779 = _panel_window(idx, "1977-01", "1979-12")
    cpi = lab["cpi_yoy"]
    slack = next((a for a in parts["validation"]["alternatives"] if a["id"] == "slack"), None)
    c77, c79 = (float(cpi.iloc[int(np.flatnonzero(idx == pd.Timestamp(m))[0])]) for m in ("1977-01", "1979-12"))
    stag_text = (f"收缩并且这段收缩期间 CPI 同比平均高于 {hi:g}%：{'、'.join(_years(x['start'], x['end']) for x in stag)}，"
                 f"共 {len(stag)} 段、{sum(x['months'] for x in stag)} 个月。1977–79 年 CPI 同比从 {c77:.1f}% 升到 {c79:.1f}%，"
                 f"但工业产出在增长，按规则是通胀（这 {int(w7779.sum())} 个月里 {int((w7779 & (ex == 'inflation')).sum())} 个月是通胀）。")
    if slack:
        stag_text += (f"用“失业率高于趋势 + 高通胀”的水平定义，和 NBER 衰退日期的吻合度（平衡准确率）只有 {slack['ba']:.2f}"
                      f"（选定规则 {parts['validation']['expost']['balanced_accuracy']:.2f}），所以没有采用。")
    # 萧条 -> 深度衰退
    sev_eps = [x for x in eps if x["ip_decline_pct"] >= sev - R.EPS]
    nxt = sorted([x for x in eps if x["ip_decline_pct"] < sev - R.EPS], key=lambda x: -x["ip_decline_pct"])[:2]
    dep_text = (f"按 GDP 口径，战后美国没有出现过萧条。数据里最深的 {len(sev_eps)} 次收缩——"
                + "和 ".join(f"{x['start']}..{x['end']}（工业产出 −{x['ip_decline_pct']:.1f}%）" for x in sev_eps)
                + f"——叫深度衰退（本轮工业产出从高点累计下跌 {sev:g}% 以上）；下一档是 "
                + "和 ".join(f"{_years(x['start'], x['end'])}（−{x['ip_decline_pct']:.1f}%"
                            + ("，按通胀算滞胀" if x["label"] == "stagflation" else "") + "）" for x in nxt) + "。")
    terms = [{"term": "通胀", "labels": ["inflation", "stagflation"], "text": infl_text},
             {"term": "通缩", "labels": ["deflation"], "text": defl_text},
             {"term": "滞胀", "labels": ["stagflation"], "text": stag_text},
             {"term": "萧条", "labels": ["depression"], "text": dep_text}]
    # names that changed
    fall = counts["1948"]["inflation_falling"]
    share = parts["engine"]["simulated_inflation_from_expansion_share"]
    sev_rt = _episode_list(idx, (lab["severe_rt"] == 1).to_numpy() & w48)
    defl73 = (ex == "deflation") & masks["1973"]
    up = (panel["f_cpi"].to_numpy(float) > 0) & defl73
    names_changed = [
        {"from": "扩张（金发姑娘）", "to": "扩张",
         "why": f"它包括所有未收缩、CPI 同比在 {neg:g}–{hi:g}% 之间的月份：既有繁荣，也有复苏缓慢、通胀偏低的年份，“金发姑娘”只是其中一部分"},
        {"from": "通胀（过热）", "to": "通胀",
         "why": f"{str(fall['first_month'] or '1948')[:4]} 年以来 {fall['months_with_trend']} 个通胀月份里有 {fall['falling']} 个月 "
                "CPI 同比低于 12 个月前（通胀在回落，常见于衰退之后）"
                + (f"；引擎无条件模拟里的通胀月份有 {share:.0%} 抽自历史上的扩张月份" if share is not None else "")},
        {"from": "通缩", "to": "物价负增长",
         "why": f"1948 年以来的 {len(defl_runs)} 段都是衰退之后的复苏或油价暴跌后的基数效应"
                + (f"，美股实际年化 {_fmt_pct(a48['ann_real']['value'])}" if a48 and a48["ann_real"]["value"] is not None else "")
                + f"；1973 年以来的 {int(defl73.sum())} 个月里有 {int(up.sum())} 个月 CPI 环比在涨，不是债务型通缩"},
        {"from": "萧条", "to": "深度衰退",
         "why": f"战后最深的 {len(sev_eps)} 次收缩（工业产出跌 " + " 和 ".join(f"{x['ip_decline_pct']:.1f}%" for x in sev_eps)
                + "）按 GDP 口径都不是萧条；实时版要到低谷附近才亮（" + "、".join(s.split("..")[0] for s in sev_rt) + "）"},
    ]
    grid = {"rows": ["未收缩", "收缩（未达深度）", "深度收缩"], "cols": [f"CPI < {neg:g}", f"{neg:g}–{hi:g}%", f"> {hi:g}%"],
            "cells": [["deflation", "expansion", "inflation"], ["recession", "recession", "stagflation"],
                      ["depression", "depression", "stagflation"]],
            "note": "收缩期的通胀按整段平均判断，一段收缩只有一个结论"}
    return {"params": dict(p), "labels": [{k: d.get(k) for k in ("id", "code", "name", "term", "color", "desc")} for d in R.LABELS],
            "flags": R.flag_defs(), "precedence": list(R.PRECEDENCE), "rule_text": R.rule_text(switch=switch),
            "terms": terms, "names_changed": names_changed, "grid": grid, "conventions": conventions(masks, idx, switch),
            "interpolations": interpolations(panel), "swing_pct": sw}


# ============================================================================ the file

def model_use_section(parts: dict, switch: dict | None = None) -> dict:
    verdict = parts["transmission"].get("walk_forward_verdict") or {}
    deltas = verdict.get("oos_r2_delta") or []
    dtxt = "、".join(f"{d['period']} {d['delta']:+.3f}" for d in deltas if isinstance(d, dict) and "delta" in d)
    split = parts["transmission"]["splits"][0]
    if "add_tp10_x_core4" in (verdict.get("changes") or []):
        # the walk-forward kept one inflation-state term (the term premium in high core inflation, FACTORS_DESIGN §7.4)
        dd = " / ".join(f"{d['delta']:+.3f}".replace("-", "−") for d in deltas if isinstance(d, dict) and "delta" in d)
        sw = switch if switch and switch.get("kind") == "interaction" else {"factor": "tp10", "state": "core_yoy",
                                                                             "lag": 1, "above": FM.HIGH_INFLATION}
        rule = f"{R.switch_state(sw)}高于 {float(sw['above']):g}% 时，{TERM_CHANGE.get(sw.get('factor'), '这一项')}单独进入"
        trans_reason = ("传导模型不按通胀状态拆分全部系数" + (f"（整体交互的样本外 R² 变化 {dd}）" if dd else "")
                        + f"；走步检验只保留了一项：{rule}（见诊断页“缺失因子审计”）。")
    else:
        trans_reason = "传导模型不加通胀状态交互项：诊断的走步检验里样本外 R² 变化" + (f" {dtxt}" if dtxt else "不稳定") + "，不显著也不稳定。"
    if "n_sig_5pct" in split:
        trans_reason += (f"样本内按上月高通胀拆分，{split['n_tests']} 个斜率差异里 {split['n_sig_5pct']} 个 |t| ≥ 1.96"
                         f"（偶然约 {split['expected_5pct']:.1f} 个）")
        if parts["transmission"].get("labor_share_in_model", True):
            trans_reason += (f"，去掉劳动收入占比（只在发布月变化）后是 {split['n_tests_excl_labor_share']} 个里 "
                             f"{split['n_sig_excl_labor_share']} 个（偶然约 {split['expected_5pct_excl_labor_share']:.1f} 个）")
        trans_reason += "。"
    # the forecast test is its own opt-in step that runs after this one: its verdict is stated only when its file
    # was computed on this build's data (forward.forecast_test)
    ft = parts["forward"].get("forecast_test") or {}
    fc_reason = "一年期预测按现行设定运行，不使用宏观体制；"
    if ft.get("available") and not ft.get("adopted"):
        fc_reason += "体制特征的样本外检验（可选构建步骤）里没有一类做法通过采用条件。"
    elif ft.get("available"):
        fc_reason += f"体制特征的样本外检验里有 {len(ft['adopted'])} 类做法通过了采用条件，还没有并入一年期预测。"
    else:
        fc_reason += "体制特征的样本外检验是可选构建步骤，这次构建没有与当前数据对应的检验结果。"
    return {"transmission": {"used": False, "reason": trans_reason, "oos_r2_delta": deltas or None,
                             "tested": bool(deltas)},
            "forecast": {"used": False, "reason": fc_reason, "test_file": "regime_forecast.json",
                         "tested": bool(ft.get("available")), "adopted": ft.get("adopted")},
            "scenarios": {"used": True, "reason": "情景的每个阶段可以要求多抽某种体制的历史月份（熵池化里的普通均值视图）；"
                                                  "每个情景的结果都附有模拟路径的体制读数，并和引擎的无条件抽样、历史基准对照。"}}


def model_use_note(use: dict) -> str:
    """The last note: what the out-of-sample tests behind 'regimes stay out of the models' say, stating a test's
    verdict only when this build has it (the transmission walk-forward from diagnostics, the forecast test from
    a regime_forecast.json of the same data month)."""
    tr, fc = use["transmission"], use["forecast"]
    if tr.get("tested") and fc.get("tested") and not fc.get("adopted"):
        return "体制不进入传导模型和一年期预测：两者都做过样本外检验，没有通过。"
    parts = ["传导模型的样本外检验没有通过" if tr.get("tested") else "传导模型这次构建没有走步检验结果"]
    if fc.get("tested") and not fc.get("adopted"):
        parts.append("一年期预测的样本外检验没有通过")
    elif fc.get("tested"):
        parts.append("一年期预测的样本外检验里有做法通过了采用条件，还没有并入预测")
    else:
        parts.append("一年期预测的样本外检验是可选构建步骤，这次构建没有与当前数据对应的检验结果")
    return f"体制不进入传导模型和一年期预测：{'；'.join(parts)}。"


def run(ctx, **_) -> dict:
    t_all = time.time()
    panel = ctx.panel
    if not R.has_inputs(panel):
        raise ValueError("the panel lacks the label inputs (s_cpi_yoy, lvl_unrate, lvl_ip)")
    timings: dict[str, float] = {}
    lab = R.classify_panel(panel)
    masks = {wid: R.window_mask(panel.index, wid, ctx.db.dates) for wid in WINDOWS}
    parts: dict = {}

    def timed(name, fn):
        t0 = time.time()
        parts[name] = fn()
        timings[name] = time.time() - t0

    timed("nowcast", lambda: R.nowcast(panel))
    timed("timeline", lambda: timeline_section(panel, lab, masks["1948"]))
    timed("counts", lambda: counts_section(lab, masks))
    timed("validation", lambda: validation_section(panel, lab, masks, timings))
    timed("assets", lambda: assets_section(panel, lab, masks))
    timed("episodes", lambda: episodes_section(panel, lab, masks["1973"]))
    timed("factors", lambda: factors_section(panel, lab, masks["1973"]))
    timed("transitions", lambda: transitions_section(panel, lab, masks, timings))
    timed("forward", lambda: forward_section(panel, lab, masks))
    timed("eras", lambda: eras_section(panel, lab))
    timed("transmission", lambda: transmission_section(ctx, lab))
    timed("engine", lambda: engine_section(ctx, lab, timings))
    episodes, ep_summary = parts.pop("episodes")
    parts["episodes"] = episodes
    switch = transmission_switch(ctx.model.spec)
    timed("definitions", lambda: definitions_section(panel, lab, masks, parts, switch))
    db = ctx.db
    use = model_use_section(parts, switch)
    timings["total"] = time.time() - t_all
    return {
        "version": 1, "data_end": _ym(panel.index.max()),
        "monthdb": {"first": _ym(db.dates[0]), "last": _ym(db.dates[-1]), "months": int(len(db))},
        "definitions": parts["definitions"], "nowcast": parts["nowcast"], "timeline": parts["timeline"],
        "counts": parts["counts"], "validation": parts["validation"], "assets": parts["assets"],
        "episodes": episodes, "episodes_summary": ep_summary, "factors": parts["factors"],
        "transitions": parts["transitions"], "forward": parts["forward"], "eras": parts["eras"],
        "transmission": parts["transmission"], "engine": parts["engine"], "model_use": use,
        "settings": {"bootstrap": B, "seed": int(CFG["bootstrap"]["seed"]), "ci": list(CI), "min_episodes": THIN,
                     "hac_lags": dict(CFG["hac_lags"]), "placebo_min_shift": MIN_SHIFT, "forward_lag_months": LAG,
                     "seconds": timings},
        "units": ("回报是小数（0.05 = 5%），年化 = exp(12 × 月均实际对数回报) − 1；远期回报是 12 个月的复利实际回报；"
                  "占比是 0–1；z 值 = (体制均值 − 1973 年以来均值) / 1973 年以来标准差；CPI、失业率、工业产出的跌幅和读数是百分数"),
        "notes": [
            "标签只用 CPI 同比、失业率和工业产出三条序列，历史月份和模拟路径用同一套规则；NBER 衰退日期只用来给规则打分，不进入任何标签。",
            "实时版只用截至当月的数据（单边、修订后数据），不是当年的首发值；事后版会把收缩追溯到真正的拐点。",
            "CPI、工业产出和失业率本身就是判定规则的一部分，它们在各体制里的均值部分是定义出来的；其余因子与定义无关。",
            "收缩期里股市通常先跌后涨，同月年化回报只适合在资产之间比较，不代表持有体验，请看逐段路径。",
            "† 表示少于 " + str(THIN) + " 段历史：不给置换检验 p 值，少于 2 段不给区间。",
            model_use_note(use),
        ],
    }


if __name__ == "__main__":
    from ..context import EngineContext

    t = time.time()
    jsonio.dump(run(EngineContext.load(selection=True)), config.RESULTS_DIR / "regimes.json")
    print(f"wrote {config.RESULTS_DIR / 'regimes.json'} in {time.time() - t:.1f}s")
