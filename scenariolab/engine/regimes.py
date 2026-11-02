"""Macro regimes: 扩张 / 通胀 / 物价负增长 / 衰退 / 深度衰退 / 滞胀, for history and simulated paths alike.

Every rule uses only the three series the simulator also produces (SimOutput.macro "cpi_yoy", "unrate",
"ip"), so historical months and simulated paths are labelled by one function. Core CPI, which the
simulator does not produce, feeds one history-only flag (core_low) and never a label. Parameters, ids,
names and the conventions every consumer shares live in config.REGIMES.

Growth axis: the direction of activity, not the level of slack
    1. IP legs (a zigzag on the industrial production level). A peak is confirmed once IP falls 3% below
       the highest level since the last confirmed trough; a trough is confirmed once IP rises 3% above the
       lowest level since that peak. A down-leg runs from the peak (exclusive) to the trough (inclusive);
       an unresolved start takes the more recent extreme as the pivot.
    2. Unemployment confirmation. Inside the leg, the 3-month mean unemployment rate must rise at least
       0.5 pp above its lowest value in the leg, counted from the last month it sat at that low.
    3. Contraction months: from the month after the later of (IP peak, that unemployment low) through the
       IP trough. One such run is a contraction episode.
    4. Deep (flag `severe`): the leg's IP decline from its peak to its lowest level is at least 15%.
Inflation axis
    high_infl   CPI y/y > 4% in the same month (factor_model.HIGH_INFLATION has the same threshold but
                reads the previous month)
    inflationary contraction   CPI y/y averaged over the episode's contraction months is above 4%: each
                contraction gets one verdict instead of flipping with the monthly reading (2008-09 stays
                one 深度衰退 episode)
    cpi_neg     CPI y/y < 0 in the month
    core_low    core CPI y/y < 1% (history only)
Labels (first match wins)
    滞胀 inflationary contraction (however deep) > 深度衰退 deep contraction > 衰退 other contraction
    > 物价负增长 CPI y/y < 0 > 通胀 high inflation > 扩张 everything else. A month without its own CPI y/y,
    unemployment or IP reading is unknown (code -1, id "", every rg_ column NaN).

Ex post and real time
    The ex-post label uses the whole path: once the IP decline and the unemployment rise confirm a
    contraction it is dated back to its start, and depth and inflation are judged on the whole episode.
    The real-time label of month t is the same rule on data through t, read at t; month t's data are
    published during t+1, so it is known from t+1. At the end of a path an unconfirmed down-leg counts as
    continuing until a 3% rebound confirms the trough, so the last month of any path has the same label
    in both versions.

Data caveat: the panel holds today's revised series, so "real time" means one-sided and publication-lagged,
not first-release vintages (IP revisions in particular can move a 3% swing).

rg_ columns (regime_columns): the ex-post one-hot labels rg_<label id>, the ex-post flags rg_contraction,
rg_severe, rg_high_infl, rg_cpi_neg, and the real-time rg_rt_<label id>, rg_rt_contraction, rg_rt_severe.
context.load_panel_with_shocks appends them on load (they are never stored: an ex-post label depends on
later data), and a MonthDB that ends early replaces the ex-post ones with asof_columns(panel, end).

Imports numpy, pandas and config only (no simulator, scenario or analysis module).
"""
from __future__ import annotations

import hashlib
import weakref
import zlib

import numpy as np
import pandas as pd

from .. import config

PARAMS = config.REGIMES["params"]
EPS = 1e-9
THIN = int(config.REGIMES["min_episodes"])      # a statistic resting on fewer episodes is thin (†)

# Labels in display order; `code` is the integer the label arrays carry (-1 = unknown).
LABELS = [{**d, "code": k} for k, d in enumerate(config.REGIMES["labels"])]
LABEL_IDS = [d["id"] for d in LABELS]
CODE = {d["id"]: d["code"] for d in LABELS}
ID_OF = {**{d["code"]: d["id"] for d in LABELS}, -1: ""}
NAME = {**{d["id"]: d["name"] for d in LABELS}, "": ""}
PRECEDENCE = ["stagflation", "depression", "recession", "deflation", "inflation", "expansion"]
FLAG_IDS = ("contraction", "severe", "high_infl", "cpi_neg", "core_low", "contraction_rt", "severe_rt")
_IDS = np.array(LABEL_IDS + [""], dtype=object)          # _IDS[code]: code -1 reads the last entry, ""

# panel columns of the inputs (core CPI is optional and history only)
INPUTS = {"cpi_yoy": "s_cpi_yoy", "unrate": "lvl_unrate", "ip": "lvl_ip"}
CORE_COLUMN = "s_core_yoy"

EXPOST_COLUMNS = [f"rg_{i}" for i in LABEL_IDS] + ["rg_contraction", "rg_severe", "rg_high_infl", "rg_cpi_neg"]
REALTIME_COLUMNS = [f"rg_rt_{i}" for i in LABEL_IDS] + ["rg_rt_contraction", "rg_rt_severe"]
COLUMNS = EXPOST_COLUMNS + REALTIME_COLUMNS
_FLAG_COLUMNS = (("rg_contraction", "contraction"), ("rg_severe", "severe"), ("rg_high_infl", "high_infl"),
                 ("rg_cpi_neg", "cpi_neg"), ("rg_rt_contraction", "contraction_rt"), ("rg_rt_severe", "severe_rt"))

# the arrays of classify() that hold labels, flags and episodes (the others are continuous indicators)
LABEL_KEYS = ("label", "label_rt", "contraction", "severe", "high_infl", "cpi_neg", "contraction_rt", "severe_rt",
              "infl_rising", "episode", "episode_ip_decline", "episode_cpi_mean", "episode_cpi_mean_rt")
INDICATORS = ("cpi_yoy", "core_yoy", "infl_trend_12m", "infl_trend_6m", "unrate", "unrate_3m", "sahm_gap", "misery",
              "ip_yoy", "ip_drawdown_24m", "episode_ip_decline", "episode_cpi_mean", "episode_cpi_mean_rt",
              "ip_below_peak_rt", "ip_leg_decline_rt", "u_rise_rt")
READINGS = ("cpi_yoy", "core_yoy", "infl_trend_12m", "infl_trend_6m", "unrate", "unrate_3m", "sahm_gap", "misery",
            "ip_yoy", "ip_drawdown_24m", "ip_below_peak_rt", "u_rise_rt")


def _params(params: dict | None = None) -> dict:
    return {**config.REGIMES["params"], **(params or {})}


def has_inputs(panel: pd.DataFrame) -> bool:
    """Whether a panel carries the three series the labels are computed from."""
    return all(c in panel.columns for c in INPUTS.values())


def rule_text(params: dict | None = None) -> dict[str, str]:
    """The rules in Chinese for the UI, built from the parameters the code uses."""
    p = _params(params)
    sw, ur, um, sev = p["ip_swing_pct"], p["u_rise_pp"], int(p["u_smooth_months"]), p["severe_ip_decline_pct"]
    hi, neg, core = p["high_inflation_pct"], p["cpi_neg_pct"], p["core_low_pct"]
    return {
        "growth": (f"收缩：工业产出从高点回落 {sw:g}% 以上（高点要等跌够 {sw:g}% 才确认，低点要等从低位反弹 {sw:g}% 才确认），"
                   f"并且这段下跌里失业率（{um} 个月均值）比段内低点上升至少 {ur:g} 个百分点。"
                   "收缩期从工业产出高点和失业率低点中较晚的那个月的下一个月算起，到工业产出低点为止。"),
        "severity": (f"深度收缩：这一轮工业产出从高点到最低点累计下跌 {sev:g}% 以上（事后版按整段下跌判断，"
                     "实时版按到当月为止的跌幅）。"),
        "inflation": (f"高通胀：当月 CPI 同比高于 {hi:g}%。一段收缩算不算滞胀，看这段收缩期间 CPI 同比的平均值是否高于 "
                      f"{hi:g}%（整段只给一个判断，不随单月读数来回跳）。CPI 同比为负：当月 CPI 同比低于 {neg:g}%。"
                      f"核心 CPI 偏低（仅历史，模拟路径没有核心 CPI）：核心 CPI 同比低于 {core:g}%。"),
        "precedence": (f"按顺序取第一个成立的：滞胀（收缩，且收缩期间 CPI 同比平均高于 {hi:g}%，不论深浅）→ 深度衰退（深度收缩）"
                       f"→ 衰退（其余收缩）→ 物价负增长（未收缩，当月 CPI 同比低于 {neg:g}%）→ 通胀（未收缩，当月 CPI 同比高于 "
                       f"{hi:g}%）→ 扩张（其余月份）。"),
        "realtime": ("实时版：第 t 月的标签只用到第 t 月为止的数据，就是把同一套规则用在截至 t 月的数据上、读最后一个月："
                     f"收缩要等工业产出跌够 {sw:g}% 且失业率升够 {ur:g} 个百分点才亮，要等工业产出从低点反弹 {sw:g}% 才熄；"
                     "深度和收缩期间的平均通胀都按到当月为止计算。事后版会把收缩的起点往前追溯到真正的拐点，"
                     "所以同一个月的两个版本可能不同。"),
        "data": ("数据是今天的修订版，不是当年的首发值：实时版只保证单边、按发布滞后，不保证与当年看到的一模一样。"
                 "单月缺失的读数用前后月份插值，这些月份的标签用到了下一个月的数据。"),
        "timing": (f"时点：第 t 月的数据在 t+1 月公布，所以第 t 月的实时标签 t+1 月才知道。高通胀的标签和标志看当月的 "
                   f"CPI 同比，传导模型里的高通胀状态看上个月的 CPI 同比：门槛都是 {hi:g}%，时点差一个月。"),
    }


def flag_defs(params: dict | None = None) -> list[dict]:
    """The flags (FLAG_IDS order) with their Chinese names and rules; `rt`: one-sided (real time)."""
    p = _params(params)
    sw, ur, um, sev = p["ip_swing_pct"], p["u_rise_pp"], int(p["u_smooth_months"]), p["severe_ip_decline_pct"]
    hi, neg, core = p["high_inflation_pct"], p["cpi_neg_pct"], p["core_low_pct"]
    rows = [
        ("contraction", "收缩（事后）",
         f"工业产出从高点回落 {sw:g}% 以上、失业率（{um} 个月均值）升 ≥ {ur:g} 个百分点的下跌段，从拐点追溯到工业产出低点",
         False, False),
        ("severe", "深度收缩（事后）", f"本轮工业产出从高点到最低点累计下跌 ≥ {sev:g}%", False, False),
        ("high_infl", "高通胀", f"当月 CPI 同比 > {hi:g}%", False, True),
        ("cpi_neg", "CPI 同比为负", f"当月 CPI 同比 < {neg:g}%", False, True),
        ("core_low", "核心 CPI 偏低", f"核心 CPI 同比 < {core:g}%（模拟路径没有核心 CPI）", True, True),
        ("contraction_rt", "收缩（实时）",
         f"截至当月：工业产出已从本轮高点回落 {sw:g}% 以上、还没从低点反弹 {sw:g}%，且失业率升够 {ur:g} 个百分点",
         False, True),
        ("severe_rt", "深度收缩（实时）", f"截至当月：本轮工业产出已累计下跌 ≥ {sev:g}%", False, True),
    ]
    return [{"id": i, "name": n, "desc": d, "history_only": h, "rt": rt} for i, n, d, h, rt in rows]


# ----------------------------------------------------------------------------- array helpers

def _as2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("expected an array shaped (T,) or (n, T)")
    return a


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if k < a.shape[1]:
        out[:, k:] = a[:, : a.shape[1] - k]
    return out


def _rolling(a: np.ndarray, window: int, how: str) -> np.ndarray:
    """Trailing rolling statistic along time (axis 1), NaN-aware, partial windows at the start."""
    df = pd.DataFrame(a.T).rolling(window, min_periods=1)
    return getattr(df, how)().to_numpy().T


def _span_mean(cs: np.ndarray, cn: np.ndarray, i: int, a: int, b: int) -> float:
    """Mean over months a..b (inclusive) of path i from NaN-aware cumulative sums (cs, cn start with 0)."""
    k = cn[i, b + 1] - cn[i, a]
    return float((cs[i, b + 1] - cs[i, a]) / k) if k > 0 else np.nan


# ----------------------------------------------------------------------------- growth axis

def _forward(ip: np.ndarray, u3: np.ndarray, swing_pct: float) -> dict[str, np.ndarray]:
    """One causal pass over (n, T) arrays: the IP zigzag and the unemployment tracker.

    The unemployment window always starts at the current IP peak (the running maximum while no down-leg
    is confirmed, the confirmed peak inside a down-leg), so its state at month t is exactly what the rule
    can know at t.
    """
    n, T = ip.shape
    s = swing_pct / 100.0
    trend = np.zeros(n, np.int8)                     # 0 unknown, +1 up-leg, -1 confirmed down-leg
    x0 = ip[:, 0]
    vmax = x0.copy()
    imax = np.zeros(n, np.int64)
    vmin = x0.copy()
    imin = np.zeros(n, np.int64)
    vpeak = np.full(n, np.nan)
    ipeak = np.zeros(n, np.int64)
    umin = u3[:, 0].copy()
    ulast = np.zeros(n, np.int64)
    uafter = u3[:, 0].copy()

    pivot = np.zeros((n, T), np.int8)                # +1 confirmed peak, -1 confirmed trough (at the pivot month)
    down = np.zeros((n, T), bool)                    # inside a confirmed down-leg as of month t
    peak_idx = np.zeros((n, T), np.int64)            # the current peak (candidate or confirmed) as of month t
    rise = np.zeros((n, T))                          # u3 rise above its last low since the current peak
    lastmin = np.zeros((n, T), np.int64)             # month of that low
    below_peak = np.full((n, T), np.nan)             # % below the current peak
    leg_decline = np.full((n, T), np.nan)            # % decline from the confirmed peak to the lowest level so far
    rows = np.arange(n)

    for t in range(T):
        x = ip[:, t]
        ok = ~np.isnan(x)
        reset = np.zeros(n, bool)
        if t > 0:
            seek_max = ok & (trend >= 0)
            newmax = seek_max & ~(x <= vmax)          # also true while vmax is still NaN
            vmax[newmax] = x[newmax]
            imax[newmax] = t
            reset |= newmax
            seek_min = ok & (trend <= 0)
            newmin = seek_min & ~(x >= vmin)
            vmin[newmin] = x[newmin]
            imin[newmin] = t
            pk = ok & (trend >= 0) & (x <= vmax * (1.0 - s))
            tr = ok & (trend <= 0) & (x >= vmin * (1.0 + s))
            unknown = trend == 0
            pk &= ~(unknown & tr & (imin > imax))     # an unresolved start: the more recent extreme is the pivot
            tr &= ~pk
            if pk.any():
                pivot[rows[pk], imax[pk]] = 1
                trend[pk] = -1
                vpeak[pk] = vmax[pk]
                ipeak[pk] = imax[pk]
                vmin[pk] = x[pk]
                imin[pk] = t
            if tr.any():
                pivot[rows[tr], imin[tr]] = -1
                trend[tr] = 1
                vmax[tr] = x[tr]
                imax[tr] = t
                reset |= tr
        # unemployment tracker over [current peak .. t]
        u = u3[:, t]
        uok = ~np.isnan(u)
        start = reset & uok
        low = uok & ~reset & (np.isnan(umin) | (u <= umin + EPS))
        up = uok & ~reset & ~low
        fresh = start | low
        umin[fresh] = u[fresh]
        ulast[fresh] = t
        uafter[fresh] = u[fresh]
        uafter[up] = np.maximum(uafter[up], u[up])
        gone = reset & ~uok                           # a new peak in a month without unemployment data
        umin[gone] = np.nan
        uafter[gone] = np.nan
        ulast[gone] = t

        dn = trend == -1
        down[:, t] = dn
        peak_idx[:, t] = np.where(dn, ipeak, imax)
        rise[:, t] = np.where(np.isnan(umin), 0.0, uafter - umin)
        lastmin[:, t] = ulast
        peak_level = np.where(dn, vpeak, vmax)
        below_peak[:, t] = 100.0 * (1.0 - x / peak_level)
        leg_decline[:, t] = np.where(dn, 100.0 * (1.0 - vmin / vpeak), np.nan)
    return {"pivot": pivot, "down": down, "peak_idx": peak_idx, "rise": rise, "lastmin": lastmin,
            "below_peak": below_peak, "leg_decline": leg_decline}


def zigzag(ip, u3, swing_pct: float) -> dict[str, np.ndarray]:
    """The causal IP zigzag and unemployment tracker that classify() runs, for (T,) or (n, T) levels of IP and
    of the smoothed unemployment rate (keys pivot, down, peak_idx, rise, lastmin, below_peak, leg_decline)."""
    return _forward(_as2d(ip), _as2d(u3), swing_pct)


def _expost(ip: np.ndarray, cs: np.ndarray, cn: np.ndarray, fw: dict, p: dict,
            first_needed: int = 0) -> dict[str, np.ndarray]:
    """Two-sided contraction episodes from the confirmed legs (and an unfinished last leg): months, depth
    (the whole leg's IP decline) and the episode's average CPI y/y.

    Legs that end before `first_needed` (the prepended history of simulated paths) are only counted, so the
    episode numbers are those of a full pass while the months nobody reads are not filled in."""
    n, T = ip.shape
    contraction = np.zeros((n, T), bool)
    decline = np.full((n, T), np.nan)
    cpi_mean = np.full((n, T), np.nan)
    episode = np.full((n, T), -1, np.int32)
    u_rise = p["u_rise_pp"] - EPS
    for i in range(n):
        pos = np.flatnonzero(fw["pivot"][i])
        kind = fw["pivot"][i, pos]
        ep = 0
        for k in range(len(pos)):
            if kind[k] != 1:
                continue
            peak = int(pos[k])
            trough = int(pos[k + 1]) if k + 1 < len(pos) else T - 1   # unfinished leg: runs to the end
            if fw["rise"][i, trough] < u_rise:
                continue
            if trough >= first_needed:
                s0 = max(peak, int(fw["lastmin"][i, trough]))
                sl = slice(s0 + 1, trough + 1)
                contraction[i, sl] = True
                decline[i, sl] = 100.0 * (1.0 - np.nanmin(ip[i, peak: trough + 1]) / ip[i, peak])
                cpi_mean[i, sl] = _span_mean(cs, cn, i, s0 + 1, trough)
                episode[i, sl] = ep
            ep += 1
    severe = contraction & (decline >= p["severe_ip_decline_pct"] - EPS)
    return {"contraction": contraction, "severe": severe, "ip_decline": decline, "cpi_mean": cpi_mean,
            "episode": episode}


def _realtime_cpi_mean(cs: np.ndarray, cn: np.ndarray, fw: dict, contraction_rt: np.ndarray) -> np.ndarray:
    """Average CPI y/y of the contraction as the rule dates it with data through t: months s0+1..t."""
    n, T = contraction_rt.shape
    out = np.full((n, T), np.nan)
    ii, tt = np.nonzero(contraction_rt)
    if ii.size:
        s0 = np.maximum(fw["peak_idx"][ii, tt], fw["lastmin"][ii, tt])
        k = cn[ii, tt + 1] - cn[ii, s0 + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ii, tt] = np.where(k > 0, (cs[ii, tt + 1] - cs[ii, s0 + 1]) / k, np.nan)
    return out


def _labels(contraction, severe, infl_episode, high, neg, unknown) -> np.ndarray:
    """Label codes: later assignments win, so the order below is the precedence reversed."""
    lab = np.full(contraction.shape, CODE["expansion"], np.int8)
    lab[high & ~contraction] = CODE["inflation"]
    lab[neg & ~contraction] = CODE["deflation"]
    lab[contraction] = CODE["recession"]
    lab[contraction & severe] = CODE["depression"]
    lab[contraction & infl_episode] = CODE["stagflation"]
    lab[unknown] = -1
    return lab


# ----------------------------------------------------------------------------- classifier

def classify(cpi_yoy, unrate, ip, core_yoy=None, n_hist: int = 0, params: dict | None = None) -> dict:
    """Label the months of one or many paths.

    cpi_yoy (%), unrate (%), ip (level): arrays shaped (T,) or (n, T); for simulated paths put the observed
    history first (the first `n_hist` columns, identical across paths), so the zigzag and the rolling
    windows are defined from the first simulated month. core_yoy (%) is optional (history only).
    Returns arrays for columns n_hist.. (shape (n, T - n_hist), or 1-D for 1-D input):
      label, label_rt        int8 codes (LABELS; -1 = unknown: the month lacks its own CPI y/y,
                             unemployment or IP reading)
      flags (bool)           contraction, severe, high_infl, cpi_neg, contraction_rt, severe_rt, infl_rising
                             (CPI y/y above its level 12 months earlier) and, with core_yoy, core_low
      episode measures       episode (int, -1 outside), episode_ip_decline, episode_cpi_mean (ex post),
                             episode_cpi_mean_rt (real time, the episode so far)
      indicators (one-sided) cpi_yoy, core_yoy, infl_trend_12m, infl_trend_6m, unrate, unrate_3m, sahm_gap,
                             misery, ip_yoy, ip_drawdown_24m, in_ip_down_leg_rt, ip_below_peak_rt,
                             ip_leg_decline_rt, u_rise_rt
    Flags are not masked in unknown months here (classify_panel and regime_columns mask them).
    """
    p = _params(params)
    one_d = np.asarray(ip).ndim == 1
    cpi, ur, lvl = _as2d(cpi_yoy), _as2d(unrate), _as2d(ip)
    n = max(cpi.shape[0], ur.shape[0], lvl.shape[0])
    cpi, ur, lvl = (np.broadcast_to(a, (n, a.shape[1])).astype(float) for a in (cpi, ur, lvl))
    if not (cpi.shape == ur.shape == lvl.shape):
        raise ValueError("cpi_yoy, unrate and ip must have the same shape")
    T = lvl.shape[1]
    if T == 0:                       # no months: every key with its dtype and no columns (one unknown month dropped)
        nan = np.full((n, 1), np.nan)
        r = classify(nan, nan, nan, None if core_yoy is None else nan, params=params)
        return {k: (v[0, :0] if one_d else v[:, :0]) for k, v in r.items()}
    u3 = _rolling(ur, int(p["u_smooth_months"]), "mean")
    fw = _forward(lvl, u3, p["ip_swing_pct"])
    zero = np.zeros((n, 1))
    cs = np.concatenate([zero, np.cumsum(np.nan_to_num(cpi), axis=1)], axis=1)
    cn = np.concatenate([zero, np.cumsum(~np.isnan(cpi), axis=1)], axis=1)
    ex = _expost(lvl, cs, cn, fw, p, first_needed=n_hist)

    high = cpi > p["high_inflation_pct"]
    neg = cpi < p["cpi_neg_pct"]
    # each month's own readings decide whether it is known (the 3-month unemployment mean has partial
    # windows, so it would label a month that has no unemployment print of its own)
    unknown = np.isnan(cpi) | np.isnan(ur) | np.isnan(lvl)
    contraction_rt = fw["down"] & (fw["rise"] >= p["u_rise_pp"] - EPS)
    severe_rt = contraction_rt & (fw["leg_decline"] >= p["severe_ip_decline_pct"] - EPS)
    cpi_mean_rt = _realtime_cpi_mean(cs, cn, fw, contraction_rt)
    with np.errstate(invalid="ignore"):
        infl_ep = ex["contraction"] & (ex["cpi_mean"] > p["high_inflation_pct"])
        infl_ep_rt = contraction_rt & (cpi_mean_rt > p["high_inflation_pct"])
    trend12 = cpi - _shift(cpi, 12)
    with np.errstate(invalid="ignore"):
        rising = trend12 > 0

    out = {
        "label": _labels(ex["contraction"], ex["severe"], infl_ep, high, neg, unknown),
        "label_rt": _labels(contraction_rt, severe_rt, infl_ep_rt, high, neg, unknown),
        "contraction": ex["contraction"], "severe": ex["severe"],
        "high_infl": high, "cpi_neg": neg,
        "contraction_rt": contraction_rt, "severe_rt": severe_rt,
        "episode": ex["episode"],
        "episode_ip_decline": ex["ip_decline"],           # ex post: the whole leg's IP decline, %
        "episode_cpi_mean": ex["cpi_mean"],               # ex post: CPI y/y averaged over the episode, %
        "episode_cpi_mean_rt": cpi_mean_rt,               # real time: the same, over the episode so far
        # continuous indicators (all one-sided)
        "cpi_yoy": cpi,
        "infl_trend_12m": trend12,
        "infl_trend_6m": cpi - _shift(cpi, 6),
        "infl_rising": rising,
        "unrate": ur, "unrate_3m": u3,
        "sahm_gap": u3 - _rolling(_shift(u3, 1), 12, "min"),
        "misery": cpi + ur,
        "ip_yoy": 100.0 * (lvl / _shift(lvl, 12) - 1.0),
        "ip_drawdown_24m": 100.0 * (lvl / _rolling(lvl, 24, "max") - 1.0),
        # the rule's own real-time state
        "in_ip_down_leg_rt": fw["down"],
        "ip_below_peak_rt": fw["below_peak"],             # % below the current IP peak
        "ip_leg_decline_rt": fw["leg_decline"],           # % from the confirmed peak to the low so far
        "u_rise_rt": fw["rise"],                          # pp above the unemployment low since the peak
    }
    if core_yoy is not None:
        core = np.broadcast_to(_as2d(core_yoy), (n, T)).astype(float)
        out["core_yoy"] = core
        out["core_low"] = core < p["core_low_pct"]
    # copies, so a caller that keeps the result does not keep the prepended history's months alive
    res = {k: (np.ascontiguousarray(v[:, n_hist:]) if n_hist else v) for k, v in out.items()}
    if one_d:
        res = {k: v[0] for k, v in res.items()}
    return res


def classify_panel(panel: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """One row per panel month: labels (ex post, real time; ids, "" = unknown), flags (0/1, 0 in unknown
    months; core_low 0 where core CPI is missing) and the indicators. Reads s_cpi_yoy, lvl_unrate, lvl_ip
    and, for core_low, s_core_yoy."""
    core = panel[CORE_COLUMN].to_numpy(float) if CORE_COLUMN in panel.columns else None
    r = classify(*(panel[c].to_numpy(float) for c in INPUTS.values()), core_yoy=core, params=params)
    unknown = r["label"] < 0
    df = pd.DataFrame(index=panel.index)
    df["label"] = _IDS[r["label"]]
    df["label_rt"] = _IDS[r["label_rt"]]
    for k in FLAG_IDS:
        if k in r:
            v = np.asarray(r[k]).astype(int)
            df[k] = np.where(np.isnan(r["core_yoy"]) if k == "core_low" else unknown, 0, v)
    for k in INDICATORS:
        if k in r:
            df[k] = np.asarray(r[k], float)
    df["in_ip_down_leg_rt"] = np.asarray(r["in_ip_down_leg_rt"]).astype(int)
    df["infl_rising"] = np.asarray(r["infl_rising"]).astype(int)
    return df


def classify_sim(macro: dict, panel: pd.DataFrame, anchor=None, params: dict | None = None) -> dict:
    """Label simulated paths: SimOutput.macro arrays (n, T+1), index 0 = the anchor month.

    The panel's months before the anchor (default: the panel's last month) are prepended, so column 0 is
    the anchor month's own label given its history. Returns classify()'s arrays, shaped (n, T+1)."""
    P = panel.loc[:anchor] if anchor is not None else panel
    hist = P.iloc[:-1]
    n = np.asarray(macro["ip"]).shape[0]

    def ext(key, col):
        h = hist[col].to_numpy(float)
        return np.concatenate([np.broadcast_to(h, (n, h.size)), np.asarray(macro[key], float)], axis=1)

    return classify(*(ext(k, c) for k, c in INPUTS.items()), n_hist=len(hist), params=params)


def sim_anchor(r: dict) -> dict:
    """The anchor month (column 0) of classify_sim output: its real-time label, the same on every path
    because it uses only the history, and the share of paths per ex-post label, which can depend on the
    path while the anchor sits inside an unconfirmed down-leg."""
    rt = np.unique(r["label_rt"][:, 0])
    if rt.size != 1:
        raise ValueError(f"the anchor's real-time label differs between paths: {rt.tolist()}")
    ex = np.asarray(r["label"][:, 0])
    return {"label_rt": ID_OF[int(rt[0])],
            "label_expost_mix": {ID_OF[int(c)]: float((ex == c).mean()) for c in np.unique(ex)}}


def label_codes(frame: pd.DataFrame, realtime: bool = False) -> np.ndarray:
    """Label codes (-1 = unknown) of each row from the one-hot rg_ columns (rg_rt_ with `realtime`); with
    SimOutput.idx they give the label of the historical month each simulated month was drawn from."""
    pre = "rg_rt_" if realtime else "rg_"
    codes = np.full(len(frame), -1, np.int8)
    for k, i in enumerate(LABEL_IDS):
        codes[np.nan_to_num(frame[f"{pre}{i}"].to_numpy(float)) > 0.5] = k
    return codes


# ----------------------------------------------------------------------------- rg_ columns

def regime_columns(panel: pd.DataFrame, end=None) -> pd.DataFrame:
    """The rg_ columns (COLUMNS) for every panel month, computed from panel.loc[:end] only: 0/1 floats,
    NaN after `end` and where the label is unknown (before 1948-01, or a month without its own readings)."""
    P = panel.loc[:end] if end is not None else panel
    lab = classify_panel(P)
    known = (lab["label"] != "").to_numpy()
    ex, rt = lab["label"].to_numpy(object), lab["label_rt"].to_numpy(object)
    data = {f"rg_{i}": ex == i for i in LABEL_IDS}
    data.update({f"rg_rt_{i}": rt == i for i in LABEL_IDS})
    data.update({col: lab[f].to_numpy() for col, f in _FLAG_COLUMNS})
    out = pd.DataFrame({c: np.where(known, np.asarray(data[c], float), np.nan) for c in COLUMNS}, index=P.index)
    return out.reindex(panel.index)


# as-of columns per panel object: id(panel) -> (weakref to it, {(last month, inputs digest, params): frame});
# at most _ASOF_MAX windows per panel (the oldest goes first), so a long-running server stays bounded
_ASOF: dict[int, tuple[weakref.ref, dict]] = {}
_ASOF_MAX = 256


def _forget(key: int, ref: weakref.ref) -> None:
    hit = _ASOF.get(key)
    if hit is not None and hit[0] is ref:
        del _ASOF[key]


def asof_columns(panel: pd.DataFrame, end) -> pd.DataFrame:
    """The ex-post rg_ columns (EXPOST_COLUMNS) as data through `end` define them, on the rows of
    panel.loc[:end]: what a window ending at `end` may see (a window ending 2008-06 must not see 2008-01..06
    labelled 深度衰退). The rg_rt_ columns need no as-of version: they are one-sided by construction.

    Cached per panel object (its id checked against a weakref; the entry goes with the panel), per last
    month, per digest of the input series up to it and per parameters, so repeated MonthDBs of one origin
    reuse one classification and the same object comes back."""
    P = panel.loc[:end]
    if not len(P):
        return pd.DataFrame(columns=EXPOST_COLUMNS, index=P.index, dtype=float)
    cols = [c for c in (*INPUTS.values(), CORE_COLUMN) if c in P.columns]
    digest = hashlib.blake2b(np.ascontiguousarray(P[cols].to_numpy(float)).tobytes(), digest_size=16).hexdigest()
    key = (P.index[-1], digest, tuple(sorted(config.REGIMES["params"].items())))
    pid = id(panel)
    hit = _ASOF.get(pid)
    if hit is None or hit[0]() is not panel:
        ref = weakref.ref(panel, lambda r, k=pid: _forget(k, r))
        hit = _ASOF[pid] = (ref, {})
    store = hit[1]
    if key not in store:
        while len(store) >= _ASOF_MAX:
            del store[next(iter(store))]
        store[key] = regime_columns(P)[EXPOST_COLUMNS]
    return store[key]


# ----------------------------------------------------------------------------- nowcast and provisional months

def _ym(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def provisional_from(panel: pd.DataFrame, params: dict | None = None):
    """The first month whose ex-post label later data can still change (a panel index value), or None
    when every month is final.

    Outside a confirmed down-leg, the months up to the current IP reference peak (the running maximum since
    the last confirmed trough) are final: a later 3% fall dates a contraction from the month after
    max(peak, unemployment low). Inside an unfinished confirmed down-leg the leg's months stay open. An empty
    panel has no months to change: None."""
    if not len(panel):
        return None
    p = _params(params)
    ur = _as2d(panel[INPUTS["unrate"]].to_numpy(float))
    ip = _as2d(panel[INPUTS["ip"]].to_numpy(float))
    fw = _forward(ip, _rolling(ur, int(p["u_smooth_months"]), "mean"), p["ip_swing_pct"])
    first = int(fw["peak_idx"][0, -1]) + 1
    return None if first > len(panel) - 1 else panel.index[first]


def nowcast(panel: pd.DataFrame, params: dict | None = None) -> dict:
    """Today's regime: the last complete month's real-time label, its readings and the distance to every
    threshold, plus the series missing after it, the provisional months and the last 24 months.

    "Complete" means the month has its own CPI y/y, unemployment and IP readings. Later months (a
    release-order gap or a missing print at the panel's end) are listed in missing_after =
    {series: [months]}; a trailing gap never makes the label unknown or raises. ValueError only when no
    month is complete (an empty panel included)."""
    p = _params(params)
    sw, ur_th, um = p["ip_swing_pct"], p["u_rise_pp"], int(p["u_smooth_months"])
    hi, neg, sev, core = p["high_inflation_pct"], p["cpi_neg_pct"], p["severe_ip_decline_pct"], p["core_low_pct"]
    if not len(panel):
        raise ValueError("没有同时有 CPI 同比、失业率和工业产出读数的月份")
    df = classify_panel(panel, params)
    known = (df["label_rt"] != "").to_numpy()
    if not known.any():
        raise ValueError("没有同时有 CPI 同比、失业率和工业产出读数的月份")
    pos = int(np.flatnonzero(known)[-1])
    row = df.iloc[pos]
    month = df.index[pos]
    missing = {}
    for key, col in INPUTS.items():
        later = panel[col].iloc[pos + 1:]
        miss = [_ym(m) for m in later.index[later.isna().to_numpy()]]
        if miss:
            missing[key] = miss
    down = bool(row["in_ip_down_leg_rt"])
    below = float(row["ip_below_peak_rt"])
    rise = float(row["u_rise_rt"])
    depth = float(row["ip_leg_decline_rt"]) if down else below
    cpi = float(row["cpi_yoy"])
    checks = [
        {"id": "high_infl", "name": "高通胀", "rule": f"当月 CPI 同比 > {hi:g}%", "reading": cpi, "threshold": hi,
         "distance": cpi - hi, "on": bool(row["high_infl"])},
        {"id": "cpi_neg", "name": "CPI 同比为负", "rule": f"当月 CPI 同比 < {neg:g}%", "reading": cpi, "threshold": neg,
         "distance": cpi - neg, "on": bool(row["cpi_neg"])},
        {"id": "ip_leg", "name": "工业产出下跌段",
         "rule": (f"下跌段已确认；要等工业产出从低点反弹 {sw:g}% 才结束" if down else f"工业产出比本轮高点低 ≥ {sw:g}%"),
         "reading": below, "threshold": sw, "distance": None if down else below - sw, "on": down},
        {"id": "u_rise", "name": "失业率确认",
         "rule": f"失业率 {um} 个月均值比本轮工业产出高点以来的低点高 ≥ {ur_th:g} 个百分点", "reading": rise,
         "threshold": ur_th, "distance": rise - ur_th, "on": rise >= ur_th - EPS},
        {"id": "severe", "name": "深度收缩", "rule": f"本轮工业产出从高点累计下跌 ≥ {sev:g}%", "reading": depth,
         "threshold": sev, "distance": depth - sev, "on": bool(row["severe_rt"])},
    ]
    if bool(row["contraction_rt"]):
        m = float(row["episode_cpi_mean_rt"])
        checks.append({"id": "infl_episode", "name": "收缩期间平均通胀",
                       "rule": f"这段收缩到目前为止 CPI 同比平均 > {hi:g}% 则为滞胀", "reading": m, "threshold": hi,
                       "distance": m - hi, "on": m > hi})
    if "core_low" in df.columns and np.isfinite(row.get("core_yoy", np.nan)):
        c = float(row["core_yoy"])
        checks.append({"id": "core_low", "name": "核心 CPI 偏低", "rule": f"核心 CPI 同比 < {core:g}%（仅历史）",
                       "reading": c, "threshold": core, "distance": c - core, "on": bool(row["core_low"])})
    readings = {k: float(row[k]) for k in READINGS if k in df.columns and pd.notna(row[k])}
    done = df.iloc[: pos + 1]
    prov = provisional_from(panel.iloc[: pos + 1], params)
    hot = done.index[(done["high_infl"] == 1).to_numpy()]
    last_hot = ({"month": _ym(hot[-1]), "cpi_yoy": float(done.loc[hot[-1], "cpi_yoy"])} if len(hot) else None)
    recent = [{"month": _ym(t), "label": r["label"], "label_rt": r["label_rt"], "cpi_yoy": float(r["cpi_yoy"]),
               "ip_below_peak_rt": float(r["ip_below_peak_rt"]), "u_rise_rt": float(r["u_rise_rt"])}
              for t, r in done.iloc[-24:].iterrows()]
    note = "实时标签只用截至该月的数据（单边、修订后数据）；之后的数据确认拐点后，事后标签可能改写最近几个月。"
    lo, hi_w = config.REGIMES["nber_window"]
    if len(done) and done.index[0] <= pd.Timestamp(lo) and done.index[-1] >= pd.Timestamp(hi_w):
        v = validate_nber(panel.iloc[: pos + 1], params=params)["realtime"]
        if v["mean_start_err"] is not None:
            note += (f"{lo[:4]}–{hi_w[:4]} 年 {v['recessions']} 次 NBER 衰退中，实时版的收缩信号平均比 NBER 起点晚 "
                     f"{v['mean_start_err']:.1f} 个月亮、比 NBER 低谷晚 {v['mean_end_err']:.1f} 个月熄。")
    return {"month": _ym(month), "known_from": _ym(month + pd.offsets.MonthBegin(1)),
            "label": row["label_rt"], "name": NAME[row["label_rt"]], "label_expost": row["label"],
            "readings": readings, "checks": checks, "missing_after": missing,
            "provisional_from": _ym(prov) if prov is not None else None, "last_high_infl": last_hot,
            "recent": recent, "note": note}


# ----------------------------------------------------------------------------- episodes and validation

def _month_strings(index) -> np.ndarray:
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("M").astype(str).to_numpy()
    if isinstance(index, pd.DatetimeIndex):
        return index.strftime("%Y-%m").to_numpy()
    return np.asarray(index.astype(str))


def episodes(labels: pd.Series, months: pd.Series | None = None) -> pd.DataFrame:
    """Every run of consecutive months with the same label id ("" = unknown is skipped): label, name,
    start, end, months. Months default to the series index as YYYY-MM."""
    lab = labels.to_numpy(object)
    mon = months.to_numpy() if months is not None else _month_strings(labels.index)
    rows, i, n = [], 0, len(lab)
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        if lab[i] != "":
            rows.append({"label": lab[i], "name": NAME[lab[i]], "start": str(mon[i]), "end": str(mon[j]),
                         "months": j - i + 1})
        i = j + 1
    return pd.DataFrame(rows, columns=["label", "name", "start", "end", "months"])


# NBER business-cycle reference dates (peak, trough). For validation only: they are announced 6-21 months
# late and never enter a label, a real-time signal or a forecast.
NBER_PEAK_TROUGH = [("1948-11", "1949-10"), ("1953-07", "1954-05"), ("1957-08", "1958-04"), ("1960-04", "1961-02"),
                    ("1969-12", "1970-11"), ("1973-11", "1975-03"), ("1980-01", "1980-07"), ("1981-07", "1982-11"),
                    ("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04")]


def _month_index(index) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("M")
    return pd.DatetimeIndex(index).to_period("M")


def nber_months(index) -> np.ndarray:
    """1 in NBER recession months (the month after each peak through the trough), else 0."""
    idx = _month_index(index)
    out = np.zeros(len(idx), int)
    for pk, tr in NBER_PEAK_TROUGH:
        out[(idx > pd.Period(pk, "M")) & (idx <= pd.Period(tr, "M"))] = 1
    return out


def validate_nber(panel: pd.DataFrame, lo: str | None = None, hi: str | None = None, before: int = 3,
                  after: int = 12, params: dict | None = None) -> dict:
    """Score the ex-post and the real-time contraction flags against the NBER months in [lo, hi]
    (default config.REGIMES["nber_window"]): {"window", "expost", "realtime", "months_flagged_after_window"}.

    Each run of flagged months is assigned to the recession it overlaps most within [first recession month
    - before, trough + after]; runs assigned to none are false alarms. Errors are in months (flag minus NBER;
    for the real-time flag they are detection lags)."""
    lo, hi = lo or config.REGIMES["nber_window"][0], hi or config.REGIMES["nber_window"][1]
    df = classify_panel(panel, params)
    idx = _month_index(panel.index)
    w = (idx >= pd.Period(lo, "M")) & (idx <= pd.Period(hi, "M"))
    rec = nber_months(idx).astype(bool)
    out = {"window": [lo, hi]}
    recs = [(pk, tr, pd.Period(pk, "M") + 1, pd.Period(tr, "M")) for pk, tr in NBER_PEAK_TROUGH
            if pd.Period(lo, "M") <= pd.Period(pk, "M") + 1 <= pd.Period(hi, "M")]
    for name, key in (("expost", "contraction"), ("realtime", "contraction_rt")):
        f = df[key].to_numpy().astype(bool)
        P, R = f[w], rec[w]
        tp, fp = int((P & R).sum()), int((P & ~R).sum())
        fn, tn = int((~P & R).sum()), int((~P & ~R).sum())
        widx = idx[w]
        rl = [(widx[a], widx[b]) for a, b in runs(P)]
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
            per.append({"nber": f"{pk}→{tr}", "first_month": str(s0),
                        "detected": bool(P[(widx >= s0) & (widx <= e0)].any()),
                        "start": str(a), "end": str(b), "start_err": (a - s0).n, "end_err": (b - e0).n})
        used = {i for v in assign.values() for i in v}
        se = [r["start_err"] for r in per if "start_err" in r]
        ee = [r["end_err"] for r in per if "end_err" in r]
        tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        out[name] = {"balanced_accuracy": 0.5 * (tpr + tnr), "tpr": tpr, "tnr": tnr, "precision": tp / max(tp + fp, 1),
                     "detected": sum(r["detected"] for r in per), "recessions": len(per),
                     "mean_start_err": float(np.mean(se)) if se else None,
                     "mean_end_err": float(np.mean(ee)) if ee else None,
                     "mean_abs_start_err": float(np.mean(np.abs(se))) if se else None,
                     "mean_abs_end_err": float(np.mean(np.abs(ee))) if ee else None,
                     "false_alarms": [[str(a), str(b)] for i, (a, b) in enumerate(rl) if i not in used],
                     "per_recession": per}
    later = idx > pd.Period(hi, "M")
    out["months_flagged_after_window"] = {name: int(df[key].to_numpy()[later].sum())
                                          for name, key in (("expost", "contraction"), ("realtime", "contraction_rt"))}
    return out


# ----------------------------------------------------------------------------- conventions (every consumer)

def runs(mask) -> list[tuple[int, int]]:
    """(start, end) positions, inclusive, of the True runs of a boolean array (NaN counts as False)."""
    m = np.nan_to_num(np.asarray(mask, float)) > 0.5
    e = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return list(zip(np.flatnonzero(e == 1).tolist(), (np.flatnonzero(e == -1) - 1).tolist()))


def real_log_return(r, f_cpi):
    """Monthly real log return log(1 + r) - f_cpi (f_cpi = Δlog CPI). Sums over months chain-link."""
    return np.log1p(np.asarray(r, float)) - np.asarray(f_cpi, float)


def annualized(log_sum: float, months: int) -> float:
    """Geometric annual rate expm1(12 · mean) from a sum of `months` monthly log returns (NaN for 0 months);
    a 12-month window's return is expm1(its sum)."""
    return float(np.expm1(12.0 * log_sum / months)) if months else float("nan")


def window_mask(index, window: str, db_dates=None) -> np.ndarray:
    """Boolean mask of `index` for a statistics window: "1973" = the MonthDB months (db_dates' first..last
    month; 1973-01 lacks the trade-weighted dollar), "1948" = config.REGIMES["first_month"] through the
    last MonthDB month (the last index month without db_dates)."""
    idx = pd.DatetimeIndex(index)
    if window == "1973":
        if db_dates is None or not len(db_dates):
            raise ValueError("the 1973 window is the MonthDB months: pass db_dates")
        lo, last = pd.Timestamp(db_dates[0]), pd.Timestamp(db_dates[-1])
    elif window == "1948":
        lo = pd.Timestamp(config.REGIMES["first_month"])
        last = pd.Timestamp(db_dates[-1]) if db_dates is not None and len(db_dates) else idx.max()
    else:
        raise ValueError(f"unknown window {window!r} (\"1973\" or \"1948\")")
    return np.asarray((idx >= lo) & (idx <= last))


def rng_for(*keys) -> np.random.Generator:
    """The bootstrap generator of one cell, seeded by the cell's own keys (section, window, group, asset,
    stat), so a value does not depend on the order the cells are computed in."""
    tag = "|".join(str(k) for k in keys)
    return np.random.default_rng([int(config.REGIMES["bootstrap"]["seed"]), zlib.crc32(tag.encode())])


def shares(codes) -> dict[str, float]:
    """{label id: share} of the known codes (>= 0) of an array of label codes; sums to 1 (all 0 if none)."""
    c = np.asarray(codes).ravel()
    c = c[c >= 0]
    n = c.size
    counts = np.bincount(c.astype(np.int64), minlength=len(LABEL_IDS)) if n else np.zeros(len(LABEL_IDS))
    return {i: (float(counts[k] / n) if n else 0.0) for k, i in enumerate(LABEL_IDS)}
