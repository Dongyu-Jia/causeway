"""Macro regimes in scenarios: the `regime` phase key, its checks, the regime readout of every simulated
scenario, the library's regime fields and the translator's prompt section.

A phase can ask for historical months of a regime (engine/regimes.py labels every month):

    regime: {滞胀: 0.7}                                      YAML (ids or the Chinese aliases)
    regime: [{"id": "stagflation", "share": 0.7, "why": …}]  LLM form ([] when unused)

validate_spec turns each entry into a plain view {var: rg_<id>, mean: share} placed first (it replaces a
view on the same column, so re-validating a clean spec changes nothing) and keeps regime: {id: share} in the
clean phase, so the detail shows what was asked. The six labels are mutually exclusive (shares summing to
more than 1 are rescaled); contraction and high_infl are flags that combine with them. Pooling is unchanged:
these are ordinary views on the ex-post rg_ columns, clipped to what a phase can average like any view.
Plain views on rg_rt_ columns are refused (real-time labels serve the nowcast and the forecast test only).

Asking for a regime is not getting it, so every request is checked (warnings W1-W11, DESIGN §3.3):
statically in validate_spec (W1-W7) and against the month database after the phases pool (W8-W11, from
runner.run_spec and translator.preview). Plain views on ex-post rg_ columns get the same checks. Every number in a warning comes from the frame or the config at
validation time; how much a request delivers in simulated paths is measured in regimes.json (engine
studies), never typed into a warning.

The readout (detail["regimes"]) labels each simulated path with the classifier (the panel history
prepended), next to the historical months it was drawn from and the engine's own unconditional reference
run over the same months, with notes generated from the numbers (N1-N4). It only reads SimOutput arrays,
so no simulated number depends on it.

Does not import runner (runner imports this module; the reference run is passed in).
"""
from __future__ import annotations

import json
import math
import weakref
from fractions import Fraction

import numpy as np
import pandas as pd

from .. import config
from ..engine import regimes
from ..engine.pooling import BLOCK, pool_phase, start_values, view_support
from ..engine.sampler import phase_block

# ----------------------------------------------------------------------------- vocabulary

LABEL_IDS = list(regimes.LABEL_IDS)
FLAG_NAMES = {"contraction": "收缩", "high_infl": "高通胀"}
IDS = (*LABEL_IDS, *FLAG_NAMES)                          # what a phase's regime key may ask for
NAMES = {**{i: regimes.NAME[i] for i in LABEL_IDS}, **FLAG_NAMES}
ALIASES = {"扩张": "expansion", "金发姑娘": "expansion", "软着陆": "expansion", "通胀": "inflation", "过热": "inflation",
           "通缩": "deflation", "物价负增长": "deflation", "衰退": "recession", "萧条": "depression",
           "深度衰退": "depression", "滞胀": "stagflation", "收缩": "contraction", "高通胀": "high_infl"}
# ex-post rg_ columns a view may target -> the id the checks use: the eight ids, plus rg_severe (the
# support checks of 深度衰退, on its own column) and rg_cpi_neg (W5 and the support checks)
VIEW_IDS = {**{f"rg_{i}": i for i in IDS}, "rg_severe": "severe", "rg_cpi_neg": "cpi_neg"}
COLUMN_NAMES = {**NAMES, "severe": "深度收缩", "cpi_neg": "CPI 同比为负"}
SUPPORT_IDS = (*IDS, "cpi_neg")                          # the published support table (DESIGN §3.2)
INFLATION_IDS = ("inflation", "stagflation", "high_infl")              # W6
CONTRACTION_IDS = ("contraction", "recession", "depression", "stagflation")   # W7
DEFLATION_IDS = ("deflation", "cpi_neg")                                 # W5

SUPPORT_MONTHS = 12          # the support table is what a 12-month phase can deliver at target 1
DELIVERY_TOL = 0.005         # W8: a request above the deliverable share by more than this (display precision)
CONFLICT_SD = 1.5            # W10: a same-phase view this many block-sd away from the regime's own drawn mean
CONFLICT_GAIN = 0.05         # W10b: a view whose removal lifts lambda by at least this is named
SHRINK_TOL = 1e-6
CONTRACTION_SHARE_MIN = 0.3  # W11: phases drawing at least this share of contraction months add up
NOTE_REQUEST_GAP = 0.10      # N1: requested - drawn
NOTE_CATCH_UP_GAP = 0.20     # N2: drawn - simulated
NOTE_TERM_SHARE = 0.5        # N3: the labels a name's term means hold less than this share of the phase
CPI_DRIFT_FROM_MONTH = 25    # N4: the reference's CPI y/y from this simulated month on (past today's level)
# words in scenario and phase names and the labels they mean (N3); a name's own terms apply to its phase,
# otherwise the scenario name's terms apply to every phase with views
NAME_TERMS = (("滞胀", ("stagflation",)), ("通缩", ("deflation",)), ("萧条", ("depression",)),
              ("衰退", ("recession", "depression", "stagflation")), ("通胀", ("inflation", "stagflation")),
              ("过热", ("inflation", "stagflation")))
WINDOWS = {"year1": (0, 12), "years2_5": (12, 60), "years6_15": (60, 180), "whole": (0, None)}
VERSION = 1


def _longest() -> int:
    return int(config.REGIMES["longest_contraction_months"])


def _thin_runs() -> int:
    return int(config.REGIMES["thin_runs"])


def _first_year() -> str:
    return str(config.REGIMES["first_month"])[:4]


def _db_year(frame: pd.DataFrame) -> str:
    return f"{frame.index[0]:%Y}" if len(frame) else ""


def _num(x) -> float | None:
    if isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def regime_id(raw) -> str | None:
    """An id or Chinese alias -> the id (None when unknown)."""
    s = str(raw if raw is not None else "").strip()
    if s in ALIASES:
        return ALIASES[s]
    s = s.lower()
    return s if s in IDS else None


def available(ctx) -> bool:
    """Whether a context can carry the readout: the panel has the classifier's inputs and the month
    database the ex-post rg_ columns."""
    return regimes.has_inputs(ctx.panel) and all(c in ctx.db.frame.columns for c in regimes.EXPOST_COLUMNS)


# ----------------------------------------------------------------------------- spec syntax (validate_spec)

def unknown_text(k: int, raw) -> str:
    """W1 (also used by translator.repair)."""
    return f"阶段 {k}：未知的宏观体制 {raw}，已忽略（可用：{'、'.join(NAMES[i] for i in IDS)}）"


def why_text(rid: str, share: float) -> str:
    return f"宏观体制：{NAMES[rid]}占抽中月份的 {share:.0%}"


def normalize_regime(raw, k: int) -> tuple[dict[str, float], list[str], list[dict]]:
    """A phase's regime key -> ({id: share}, warnings W1-W3, clipped entries); k is the phase number.

    Accepts the dict form {id or alias: share} and the list form [{"id", "share", "why"}]. Shares are
    clipped to 0..1 (reported in `clipped`); the first entry of an id wins. When the six labels, which
    are mutually exclusive, sum to more than 1 they are rescaled to 1.
    """
    if raw is None or raw == [] or raw == {}:
        return {}, [], []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(d.get("id"), d.get("share")) if isinstance(d, dict) else (d, None) for d in raw]
    else:
        items = [(raw, None)]
    out: dict[str, float] = {}
    warns: list[str] = []
    clipped: list[dict] = []
    for key, val in items:
        rid = regime_id(key)
        if rid is None:
            warns.append(unknown_text(k, key))
            continue
        share = _num(val)
        if share is None:
            warns.append(f"阶段 {k}：宏观体制 {NAMES[rid]} 的占比不是数值，已忽略")
            continue
        if rid in out:
            continue
        used = min(max(share, 0.0), 1.0)
        if used != share:
            clipped.append({"phase": k, "var": f"regime.{rid}", "requested": share, "used": used})
        out[rid] = used
    total = sum(s for r, s in out.items() if r in LABEL_IDS)
    if total > 1.0 + 1e-9:
        for r in out:
            if r in LABEL_IDS:
                out[r] = out[r] / total
        warns.append(f"阶段 {k}：六种体制互斥，占比之和 {total:.2f} 超过 1，已按比例缩到合计 1")
    return out, warns, clipped


def expand_regime(reg: dict[str, float], views):
    """The views a phase pools on: the regime key as plain views on rg_ columns first, replacing views on
    the same columns (so a clean spec re-validates to the same views). Without a key, `views` as given."""
    if not reg:
        return views
    new = [{"var": f"rg_{r}", "mean": s, "why": why_text(r, s)} for r, s in reg.items()]
    cols = {v["var"] for v in new}
    kept = [v for v in (views if isinstance(views, list) else [])
            if not (isinstance(v, dict) and str(v.get("var", "")) in cols)]
    return new + kept


def realtime_view_text(k: int, var: str) -> str:
    """W4."""
    return (f"阶段 {k}：{var} 是实时标签（只用于现况判断和一年期预测检验），情景视图请用事后标签 rg_<体制>，"
            "已忽略")


def phase_requests(phase: dict, k: int, clipped=()) -> dict[str, float]:
    """{id: share asked for} of a clean phase: its regime key, plus plain mean views on ex-post rg_ columns
    at the value asked before the support clip (from validate_spec's `clipped`; k is the phase number)."""
    asked = {c.get("var"): c.get("requested") for c in clipped or ()
             if c.get("phase") == k and c.get("key") == "mean"}
    out: dict[str, float] = {}
    for v in phase.get("views") or []:
        rid = VIEW_IDS.get(v.get("var"))
        val = _num(asked.get(v.get("var"), v.get("mean")))
        if rid is not None and val is not None:
            out[rid] = min(max(val, 0.0), 1.0)
    for rid, s in (phase.get("regime") or {}).items():
        if rid in IDS and _num(s) is not None:
            out[rid] = float(s)
    return out


def _runs_text(frame: pd.DataFrame, col: str) -> tuple[int, list[str]]:
    x = np.nan_to_num(frame[col].to_numpy(float)) > 0.5
    d = frame.index
    return int(x.sum()), [f"{d[a]:%Y-%m}..{d[b]:%Y-%m}" for a, b in regimes.runs(x)]


def static_checks(phase: dict, k: int, frame: pd.DataFrame, requests: dict[str, float]) -> list[str]:
    """W5-W7 of a clean phase (validate_spec; the frame is the MonthDB frame). k is the phase number."""
    out = []
    m = int(phase["months"])
    asked = {r: s for r, s in requests.items() if s > 0}
    if any(r in asked for r in DEFLATION_IDS) and "rg_deflation" in frame.columns and "f_cpi" in frame.columns:
        n, runs = _runs_text(frame, "rg_deflation")
        lo = view_support(frame, "f_cpi", m, BLOCK)[0] * 12.0
        out.append(f"阶段 {k}：{_db_year(frame)} 年以来物价负增长的 {n} 个月（{'、'.join(runs) or '无'}）是油价暴跌后的"
                   "基数效应，当月物价其实在涨；多抽这些月份几乎不会让模拟的 CPI 下跌（实测见“宏观体制”页“引擎里的体制”）。"
                   f"要表达物价下跌，请对 f_cpi 设 mean_annual（历史支撑下限约 {lo:.3f}/年），加 收缩 0，"
                   "并给 f_unrate 一个视图")
    if m < 12:
        for r in INFLATION_IDS:
            if r in asked:
                out.append(f"阶段 {k} 只有 {m} 个月：模拟的 CPI 同比约要 12 个月才追上抽中月份的通胀，"
                           f"这一阶段里{NAMES[r]}的模拟占比会明显低于抽中占比")
    longest = _longest()
    for r in CONTRACTION_IDS:
        if r in asked and asked[r] * m > longest:
            out.append(f"阶段 {k}：{NAMES[r]} {asked[r]:.0%} × {m} 个月 ≈ {asked[r] * m:.0f} 个收缩月份，超过 "
                       f"{_first_year()} 年以来最长的一次收缩（{longest} 个月）。收缩区块会首尾相接，收缩月份越多，"
                       "工业产出跌幅超出历史上最深一次的路径就越多（实测见“宏观体制”页）。"
                       f"建议拆成不超过 {longest} 个月的收缩阶段，再接一个复苏阶段")
    return out


# ----------------------------------------------------------------------------- support (cached per MonthDB)

_SUPPORT: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def support_table(db) -> dict[str, dict]:
    """What a 12-month phase can deliver per ex-post rg_ column of a month database (cached per database).

    {id: {id, name, column, months, runs, episodes ["YYYY-MM..YYYY-MM"], support_hi (the clip bound),
    max_delivered (drawn share at target 1, after the clip and the engine's thinness shrink), shrink, reason,
    ens, engine_episodes (the engine's 12-month bins of all drawn months), regime_effective_episodes (how
    many of the regime's own runs the weight rests on, exp of the entropy of the runs' weight shares),
    weights [{episode, share}] (descending)}} for the ids of VIEW_IDS the frame has.
    """
    hit = _SUPPORT.get(db)
    if hit is None:
        hit = _SUPPORT[db] = _support(db)
    return hit


def _support(db) -> dict[str, dict]:
    F = db.frame
    prior = db.prior()
    out = {}
    for col, rid in VIEW_IDS.items():
        if col not in F.columns:
            continue
        x = np.nan_to_num(F[col].to_numpy(float))
        rr = regimes.runs(x > 0.5)
        _, hi = view_support(F, col, SUPPORT_MONTHS, BLOCK)
        pp = pool_phase(F, [{"var": col, "mean": 1.0}], prior, SUPPORT_MONTHS, BLOCK, 1.0, rare=db.rare)
        w = pp.month_weights
        shares = np.array([w[a:b + 1].sum() for a, b in rr], float)
        p = shares / shares.sum() if shares.sum() > 0 else shares
        p = p[p > 0]
        names = [f"{F.index[a]:%Y-%m}..{F.index[b]:%Y-%m}" for a, b in rr]
        order = np.argsort(-shares, kind="stable")
        out[rid] = {"id": rid, "name": COLUMN_NAMES[rid], "column": col, "months": int((x > 0.5).sum()),
                    "runs": len(rr), "episodes": names, "support_hi": float(hi), "max_delivered": float(w @ x),
                    "shrink": float(pp.shrink), "reason": pp.shrink_reason, "ens": float(pp.ens),
                    "engine_episodes": None if pp.episodes is None else float(pp.episodes),
                    "regime_effective_episodes": float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0,
                    "weights": [{"episode": names[i], "share": float(shares[i])} for i in order]}
    return out


def support_rows(db) -> dict[str, dict]:
    """The published support table (DESIGN §3.2: the eight ids and rg_cpi_neg)."""
    sup = support_table(db)
    return {r: sup[r] for r in SUPPORT_IDS if r in sup}


# ----------------------------------------------------------------------------- database checks (W8-W11)

# Pools of the database checks, per month database (weakly) and per (months, views): translator.preview and
# runner.run_spec check the same clean spec, and a phase whose views conflict with its regime needs a pool per
# pair and per left-out view (a second or more each), so the second check reuses the first one's pools.
_POOLS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_POOLS_MAX = 64


def _check_pool(db, views: list[dict], months: int):
    """pool_phase of `views` over a `months`-month phase of `db` at full confidence (cached)."""
    store = _POOLS.get(db)
    if store is None:
        store = _POOLS[db] = {}
    key = (int(months), json.dumps(views, sort_keys=True, ensure_ascii=False, default=str))
    hit = store.get(key)
    if hit is None:
        while len(store) >= _POOLS_MAX:
            del store[next(iter(store))]
        hit = store[key] = pool_phase(db.frame, views, db.prior(), months, BLOCK, 1.0, rare=db.rare)
    return hit


def _target(v: dict) -> tuple[float, float] | None:
    """(monthly target, display unit) of a mean-type view, None for other kinds."""
    if _num(v.get("mean")) is not None:
        return float(v["mean"]), 1.0
    if _num(v.get("mean_annual")) is not None:
        return float(v["mean_annual"]) / 12.0, 12.0
    return None


def regime_checks(db, phase: dict, k: int, pool=None, clipped=()) -> list[str]:
    """W8-W10b of one clean phase against the month database; `pool` is the phase's own pool (its lambda
    is the one of all its views together), k the phase number."""
    asked = {r: s for r, s in phase_requests(phase, k, clipped).items() if s > 0}
    if not asked:
        return []
    F = db.frame
    sup = support_table(db)
    year = _db_year(F)
    out = []
    for rid, share in asked.items():
        s = sup.get(rid)
        if s is None:
            continue
        if share > s["max_delivered"] + DELIVERY_TOL:
            out.append(f"阶段 {k}：{COLUMN_NAMES[rid]} 最多只能占抽中月份的 {s['max_delivered']:.0%}（要求 {share:.0%}）")
        if s["runs"] < _thin_runs() or s["regime_effective_episodes"] < _thin_runs() - 0.5:
            out.append(f"阶段 {k}：{COLUMN_NAMES[rid]} 在 {year} 年以来只有 {s['runs']} 段历史"
                       f"（{'、'.join(s['episodes'])}），按抽样权重约等于 {s['regime_effective_episodes']:.1f} 段；"
                       "引擎按全部抽中月份做的稀薄检查看不出这一点，模拟只会重复这几段的月份")
    views = phase.get("views") or []
    rviews = [v for v in views if v.get("var") in VIEW_IDS]
    others = [v for v in views if v.get("var") not in VIEW_IDS]
    if not rviews or not others:
        return out
    months = int(phase["months"])

    def pool_of(vs):
        return _check_pool(db, vs, months)

    base = pool_of(rviews)
    full = pool if pool is not None else pool_of(views)
    B = phase_block(BLOCK, months, len(F))
    for v in others:
        tgt = _target(v)
        if tgt is None:
            continue
        x = F[v["var"]].to_numpy(float)
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        reg_mean = float(base.month_weights @ x)
        sv = start_values(x, B, months)
        sd = float(np.sqrt(((sv - sv.mean()) ** 2).mean()))
        z = abs(tgt[0] - reg_mean) / sd if sd > 0 else 0.0
        pair = pool_of(rviews + [v])
        if z > CONFLICT_SD or pair.shrink < 1.0 - SHRINK_TOL:
            out.append(f"阶段 {k}：{v['var']} 的目标 {tgt[0] * tgt[1]:.3g} 与所选体制的历史月份"
                       f"（平均 {reg_mean * tgt[1]:.3g}）相差 {z:.1f} 个标准差")
    if full.shrink < base.shrink - SHRINK_TOL:
        gains = []
        for v in others:
            rest = [u for u in views if u is not v]
            gains.append((pool_of(rest).shrink - full.shrink, v["var"]))
        gains.sort(key=lambda g: -g[0])
        culprits = [f"{var}（去掉后 λ 升到 {full.shrink + g:.2f}）" for g, var in gains[:2] if g >= CONFLICT_GAIN]
        out.append(f"阶段 {k}：所选体制单独可以满足（λ={base.shrink:.2f}），和其他视图一起只能按 λ={full.shrink:.2f} 满足"
                   + (f"；冲突最大的是 {'、'.join(culprits)}" if culprits else ""))
    return out


def phase_contraction_check(db, pools: list, phases: list[dict]) -> list[str]:
    """W11, any spec: expected contraction months (drawn rg_contraction share x months) summed over
    consecutive phases that draw at least 30% contraction months, against the longest contraction since
    1948."""
    if "rg_contraction" not in db.frame.columns:
        return []
    x = np.nan_to_num(db.frame["rg_contraction"].to_numpy(float))
    longest = _longest()
    out, run, names = [], 0.0, []
    for pool, ph in [*zip(pools, phases), (None, None)]:
        share = float(pool.month_weights @ x) if pool is not None else 0.0
        if pool is not None and share >= CONTRACTION_SHARE_MIN:
            run += share * int(ph["months"])
            names.append(str(ph["name"]))
            continue
        if run > longest:
            out.append(f"阶段「{'」「'.join(names)}」合计约 {run:.0f} 个收缩月份（抽中月份里收缩占比 × 月数），超过 "
                       f"{_first_year()} 年以来最长的一次收缩（{longest} 个月）：区块首尾相接会造出比历史上任何一次都"
                       "更深更长的收缩")
        run, names = 0.0, []
    return out


def spec_checks(db, spec: dict, pools: list, clipped=()) -> list[str]:
    """W8-W11 of a validated spec whose phases pooled to `pools` (runner.run_spec, translator.preview)."""
    if not any(c in db.frame.columns for c in VIEW_IDS):
        return []
    out = []
    for k, (ph, pool) in enumerate(zip(spec["phases"], pools), start=1):
        out += regime_checks(db, ph, k, pool=pool, clipped=clipped)
    return out + phase_contraction_check(db, pools, spec["phases"])


# ----------------------------------------------------------------------------- readout (detail["regimes"])

def _shares(codes: np.ndarray) -> dict[str, float]:
    return regimes.shares(codes)


def _dominant(mix: dict[str, float]) -> str:
    return max(LABEL_IDS, key=lambda i: mix.get(i, 0.0))


def _any(contr: np.ndarray) -> float:
    return float(contr.any(axis=1).mean()) if contr.size else 0.0


def history_base_rate(frame: pd.DataFrame, months: int) -> float | None:
    """Share of the month database's windows of `months` consecutive months that hold an (ex-post)
    contraction month; None when the database is shorter than the window."""
    if "rg_contraction" not in frame.columns or months < 1 or months > len(frame):
        return None
    x = np.nan_to_num(frame["rg_contraction"].to_numpy(float)) > 0.5
    cs = np.concatenate([[0], np.cumsum(x)])
    return float(np.mean(cs[months:] - cs[:-months] > 0))


def episode_stats(contr: np.ndarray, decline: np.ndarray) -> dict:
    """Contraction runs of simulated paths (months 1..T): runs per path, mean length, the share longer than
    the longest since 1948 and the share whose IP decline reaches the deep-contraction threshold."""
    lengths, severe = [], []
    sev = float(regimes.PARAMS["severe_ip_decline_pct"])
    for i in range(contr.shape[0]):
        for s, e in regimes.runs(contr[i]):
            lengths.append(e - s + 1)
            severe.append(bool(np.nanmax(decline[i, s:e + 1]) >= sev))
    n = len(lengths)
    return {"per_path": float(n / contr.shape[0]) if contr.shape[0] else 0.0,
            "length_mean": float(np.mean(lengths)) if n else None,
            "share_longer_than_18m": float(np.mean(np.array(lengths) > _longest())) if n else None,
            "share_severe": float(np.mean(severe)) if n else None}


def history_episodes(frame: pd.DataFrame, horizon: int = config.HORIZON_MONTHS) -> dict:
    """The month database's contraction episodes on the same scale: per `horizon` months, mean length,
    how many were longer than the longest since 1948 and how many were deep."""
    x = np.nan_to_num(frame["rg_contraction"].to_numpy(float)) > 0.5
    sev = np.nan_to_num(frame["rg_severe"].to_numpy(float)) > 0.5 if "rg_severe" in frame.columns else x & False
    rr = regimes.runs(x)
    lengths = [e - s + 1 for s, e in rr]
    return {"episodes": len(rr), "per_15y": float(len(rr) * horizon / len(frame)) if len(frame) else None,
            "length_mean": float(np.mean(lengths)) if rr else None,
            "longer_than_18m": int(sum(n > _longest() for n in lengths)),
            "severe": int(sum(bool(sev[s:e + 1].any()) for s, e in rr))}


def _window_slices(T: int) -> dict[str, slice]:
    out = {}
    for name, (a, b) in WINDOWS.items():
        b = T if b is None else min(b, T)
        if a < b:
            out[name] = slice(a, b)
    return out


def readout(ctx, sim, spec: dict, reference: dict, clipped=()) -> dict:
    """detail["regimes"] of a simulated scenario (see DESIGN §3.4 for the shape).

    simulated: the ex-post label of each simulated month (regimes.classify_sim, the panel history before
    the anchor prepended); simulated_rt: the real-time label; drawn: the ex-post label of the historical
    month each simulated month was drawn from (SimOutput.idx); reference: the same months of the engine's
    unconditional run (runner.reference_simulation). Every share dict sums to 1.
    """
    frame = ctx.db.frame
    r = regimes.classify_sim(sim.macro, ctx.panel, anchor=ctx.anchor.date)
    lab, lab_rt = r["label"][:, 1:], r["label_rt"][:, 1:]
    contr, high = r["contraction"][:, 1:], r["high_infl"][:, 1:]
    drawn = regimes.label_codes(frame)[sim.idx]
    flags_drawn = {f: np.nan_to_num(frame[f"rg_{f}"].to_numpy(float))[sim.idx] for f in FLAG_NAMES}
    flags_sim = {"contraction": contr, "high_infl": high}
    ref = reference["regimes"]
    ref_lab, ref_contr = ref["label"][:, 1:], ref["contraction"][:, 1:]
    T = lab.shape[1]

    def ref_block(sl: slice) -> dict:
        return {"simulated": _shares(ref_lab[:, sl]), "paths_with_contraction": _any(ref_contr[:, sl])}

    phases, start = [], 0
    for k, ph in enumerate(spec["phases"], start=1):
        m = int(ph["months"])
        sl = slice(start, min(start + m, T))
        sim_mix = _shares(lab[:, sl])
        requests = {rid: s for rid, s in phase_requests(ph, k, clipped).items() if rid in IDS}
        phases.append({"name": ph["name"], "start_month": start, "months": m,
                       "requested": requests or None,
                       "simulated": sim_mix, "simulated_rt": _shares(lab_rt[:, sl]), "drawn": _shares(drawn[:, sl]),
                       "dominant": _dominant(sim_mix),
                       "paths_with_contraction": _any(contr[:, sl]), "contraction_share": float(contr[:, sl].mean()),
                       "reference": ref_block(sl), "history_base_rate": history_base_rate(frame, m)})
        start += m
    windows = {}
    for name, sl in _window_slices(T).items():
        windows[name] = {"simulated": _shares(lab[:, sl]), "simulated_rt": _shares(lab_rt[:, sl]),
                         "drawn": _shares(drawn[:, sl]), "paths_with_contraction": _any(contr[:, sl]),
                         "contraction_share": float(contr[:, sl].mean()), "reference": ref_block(sl)}
    years = [{"year": y + 1, "simulated": _shares(lab[:, 12 * y:12 * y + 12]),
              "paths_with_contraction": _any(contr[:, 12 * y:12 * y + 12])} for y in range(math.ceil(T / 12))]
    episodes = {**episode_stats(contr, r["episode_ip_decline"][:, 1:]),
                "reference": episode_stats(ref_contr, ref["episode_ip_decline"][:, 1:]),
                "history": history_episodes(frame)}
    out = {"version": VERSION,
           "labels": [{"id": d["id"], "name": d["name"], "color": d["color"]} for d in regimes.LABELS],
           "anchor": {"month": f"{pd.Timestamp(ctx.anchor.date):%Y-%m}", **regimes.sim_anchor(r)},
           "phases": phases, "windows": windows, "years": years, "episodes": episodes}

    def phase_share(table: dict, rid: str, sl: slice, row: dict, key: str) -> float:
        return row[key][rid] if rid in LABEL_IDS else float(table[rid][:, sl].mean())

    n1, n2, n3 = [], [], []
    scenario_terms = [t for t in NAME_TERMS if t[0] in str(spec.get("name") or "")]
    for ph, row in zip(spec["phases"], phases):
        sl = slice(row["start_month"], min(row["start_month"] + row["months"], T))
        for rid, req in (row["requested"] or {}).items():
            d = phase_share(flags_drawn, rid, sl, row, "drawn")
            s = phase_share(flags_sim, rid, sl, row, "simulated")
            if req - d > NOTE_REQUEST_GAP:
                n1.append(f"阶段「{row['name']}」要求{NAMES[rid]}占抽中月份 {req:.0%}，实际抽中 {d:.0%}"
                          "（受历史支撑或同阶段其他视图限制）")
            if d - s > NOTE_CATCH_UP_GAP:
                n2.append(f"阶段「{row['name']}」抽中月份里{NAMES[rid]}占 {d:.0%}，模拟路径里只有 {s:.0%}："
                          "模拟的 CPI 同比和工业产出从今天的水平累加，要过一段时间才追上抽中月份")
        if not ph.get("views"):
            continue                                    # a phase without views claims nothing
        terms, where = [t for t in NAME_TERMS if t[0] in str(ph["name"])], "名称"
        if not terms:
            terms, where = scenario_terms, "情景名称"
        seen = set()
        for term, ids in terms:
            if ids in seen:
                continue
            seen.add(ids)
            share = sum(row["simulated"][i] for i in ids)
            if share < NOTE_TERM_SHARE:
                n3.append(f"阶段「{row['name']}」：{where}里有“{term}”，但模拟路径的这一阶段里"
                          f"{'、'.join(NAMES[i] for i in ids)}合计只占 {share:.0%}")
    cpi = np.asarray(reference["sim"].macro["cpi_yoy"], float)
    tail = cpi[:, CPI_DRIFT_FROM_MONTH:] if cpi.shape[1] > CPI_DRIFT_FROM_MONTH else cpi[:, 1:]
    ref_hi = float((tail > regimes.PARAMS["high_inflation_pct"]).mean())
    hist_hi = float(np.nanmean(frame["rg_high_infl"].to_numpy(float)))
    n4 = [f"引擎无条件抽样里 CPI 同比高于 {regimes.PARAMS['high_inflation_pct']:g}% 的月份占 {ref_hi:.0%}"
          f"（{_db_year(frame)} 年以来的历史为 {hist_hi:.0%}），所以模拟路径里“通胀”的占比系统性偏高；"
          "比较时请看同一时段的“引擎无条件”列"]
    out["notes"] = n1 + n2 + n3 + n4
    return out


# ----------------------------------------------------------------------------- library index

def index_fields(r: dict) -> dict:
    """What each scenarios.json entry carries (the cross-scenario card needs no detail file)."""
    return {"year1": r["windows"]["year1"]["simulated"], "whole": r["windows"]["whole"]["simulated"],
            "p_contraction_year1": r["windows"]["year1"]["paths_with_contraction"],
            "dominant_by_phase": [p["dominant"] for p in r["phases"]]}


def library_mix(entries: list[dict]) -> dict:
    """Probability-weighted year-1 mix and P(contraction in year 1) over index entries (weights = prob)."""
    rows = [e for e in entries if e.get("regimes")]
    w = np.array([max(float(e.get("prob") or 0.0), 0.0) for e in rows])
    w = w / w.sum() if w.sum() > 0 else np.full(len(rows), 1.0 / max(len(rows), 1))
    return {"year1_weighted": {i: float(sum(wk * e["regimes"]["year1"][i] for wk, e in zip(w, rows)))
                               for i in LABEL_IDS},
            "p_contraction_year1_weighted": float(sum(wk * e["regimes"]["p_contraction_year1"]
                                                      for wk, e in zip(w, rows)))}


def reference_summary(reference: dict) -> dict:
    """Year 1 of the engine's unconditional reference run, for the index."""
    ref = reference["regimes"]
    return {"year1": _shares(ref["label"][:, 1:13]), "p_contraction_year1": _any(ref["contraction"][:, 1:13]),
            "n_paths": int(reference["n_paths"]), "seed": int(reference["seed"])}


def history_summary(frame: pd.DataFrame) -> dict:
    """The month database's ex-post label shares and its 12-month contraction base rate, for the index."""
    return {"shares_1973": _shares(regimes.label_codes(frame)), "p_contraction_12m_window": history_base_rate(frame, 12)}


# ----------------------------------------------------------------------------- translator

TRANSLATOR_RULE = """## 宏观体制（regime 字段）

引擎用工业产出、失业率和 CPI 同比三条序列给每个月贴标签，历史月份和模拟路径用同一套规则：收缩 = 工业产出从高点回落 ≥{swing}% 且失业率（{smooth} 个月均值）比段内低点升 ≥{rise} 个百分点；深度收缩 = 本轮工业产出累计跌 ≥{severe}%；高通胀 = 当月 CPI 同比 > {high}%。标签按顺序取第一个成立的：滞胀（收缩且收缩期间平均 CPI 同比 > {high}%）、深度衰退（深度收缩）、衰退（其余收缩）、物价负增长（未收缩且 CPI 同比 < {neg}）、通胀（未收缩且高通胀）、扩张。
regime 字段让一个阶段多抽某种体制的历史月份：[{{"id": ..., "share": 0–1, "why": ...}}]，share 是该阶段被抽中月份里这种体制的占比；六个标签互斥（占比之和 ≤ 1），contraction（收缩）和 high_infl（高通胀）是可以和标签叠加的标志；不用时写 []。
{year} 年以来的历史支撑（{support_months} 个月阶段最多能抽到的占比；历史段数）：{support_lines}。
1. 优先用 contraction、high_infl 加 s_cpi_yoy 等连续视图；衰退（{year} 年以来只有 {recession_runs}）和深度衰退（按抽样权重约 {depression_eff} 段：{depression_span} 年）单独作为目标时只会重复这一段历史。
2. 「滞胀」只在叙事写明产出下降、失业上升而通胀仍高时用 stagflation（≤ 0.8），阶段不超过 {longest} 个月。1977–79 年那种「高通胀 + 增长放缓」但没有收缩，按引擎定义是通胀：用 inflation 或 s_cpi_yoy，并在 narrative 写明。
3. 「过热」= inflation 0.6–0.8，配合 s_ip_yoy ≥ 2.5 或 f_unrate 为负。
4. 「衰退」优先写 contraction 0.6–0.8 加 s_cpi_yoy ≤ 3。
5. 「萧条」在引擎里叫深度衰退：depression 最多约 {depression_max}，阶段不超过 12–{longest} 个月，并在 narrative 写明它实际只有 {depression_span} 年一段历史支撑{depression_rare}。
6. 「通缩」若指物价下跌：f_cpi 的 mean_annual 设在历史下限附近（约 {f_cpi_lo}）并加 contraction 0，阶段 24 个月以上，同时给 f_unrate 设视图{deflation_recipe}；不要用 deflation 标签——{year} 年以来 CPI 同比为负的月份是基数效应，抽中它们不会让模拟的物价下跌。若指「低通胀」，用 s_cpi_yoy 1–2%。
7. 「扩张 / 软着陆 / 金发姑娘」= expansion ≤ 0.8，配合 s_cpi_yoy 2–3。
8. 收缩类（contraction、stagflation、recession、depression）的 share × 月数不要超过 {longest}。
9. 通胀类的模拟同比约 12 个月才追上抽中月份{stag_catch_up}。
10. 同一阶段的其他视图要和所选体制在历史上同时出现：stagflation 配工业产出同比为正、实际利率很负、CPI 7% 和实际工资 −1% 会互相冲突{conflict}。"""

# the engine studies (regimes.json engine.studies) whose numbers the rule quotes, matched on their views
STAG_STUDY = {"views": [{"var": "rg_stagflation", "mean": 0.8}], "months": 12}
DEFLATION_STUDY = {"views": [{"var": "f_cpi", "mean_annual": -0.01}, {"var": "rg_contraction", "mean": 0.0}],
                   "months": 48}


def _view_key(views) -> frozenset | None:
    out = set()
    for v in views or []:
        if not isinstance(v, dict):
            return None
        for key, val in v.items():
            if key not in ("var", "why") and _num(val) is not None:
                out.add((str(v.get("var")), key, round(float(val), 6)))
    return frozenset(out)


def study_row(studies: dict | None, case: dict) -> dict | None:
    """The engine.studies.drawn_vs_simulated row of `case` (same views and months), or None."""
    want = _view_key(case["views"])
    for row in ((studies or {}).get("drawn_vs_simulated") or []):
        if isinstance(row, dict) and row.get("months") == case["months"] and _view_key(row.get("views")) == want:
            return row
    return None


def _span(run: str) -> str:
    """"2008-01..2009-06" -> "2008–09"; "2001-01..2001-12" -> "2001"."""
    a, b = run.split("..")
    ya, yb = int(a[:4]), int(b[:4])
    if ya == yb:
        return str(ya)
    return f"{ya}–{yb % 100:02d}" if ya // 100 == yb // 100 else f"{ya}–{yb}"


def _spans(runs: list[str]) -> str:
    return "、".join(dict.fromkeys(_span(x) for x in runs))


def _overlaps(run: str, start: str, end: str) -> bool:
    a, b = run.split("..")
    return a <= str(end)[:7] and b >= str(start)[:7]


def translator_fields(db, studies: dict | None = None) -> dict[str, str]:
    """The {…} fields of TRANSLATOR_RULE: definitions from config.REGIMES, history from the support table,
    and the study numbers from regimes.json engine.studies (a clause whose study is missing keeps its rule
    and states no number)."""
    p = regimes.PARAMS
    sup = support_table(db)
    F = db.frame
    longest = _longest()
    lines = []
    for rid in IDS:
        s = sup.get(rid)
        if s is None:
            continue
        thin = f"（{_spans(s['episodes'])}）" if s["runs"] < _thin_runs() and s["episodes"] else ""
        lines.append(f"{NAMES[rid]}（{rid}）{s['max_delivered']:.0%}，{s['runs']} 段{thin}")
    rec = sup.get("recession") or {"runs": 0, "episodes": []}
    recession_runs = (f"{_spans(rec['episodes'])} 年一段" if rec["runs"] == 1
                      else f"{rec['runs']} 段（{_spans(rec['episodes'])} 年）" if rec["runs"] else "0 段")
    dep = sup.get("depression") or {"weights": [], "episodes": [], "regime_effective_episodes": 0.0,
                                     "max_delivered": 0.0}
    top = dep["weights"][0]["episode"] if dep["weights"] else None
    # why the other deep contractions barely count: a rare-month period (config.RARE_MONTHS) covers them
    rare = ""
    for rm in config.RARE_MONTHS:
        if float(rm["weight"]) < 1 and any(_overlaps(e, rm["start"], rm["end"]) for e in dep["episodes"] if e != top):
            frac = Fraction(float(rm["weight"])).limit_denominator(20)
            rare = f"（{str(rm['start'])[:4]} 年{rm['name']}月份的抽样权重是 {frac.numerator}/{frac.denominator}）"
            break
    stag = study_row(studies, STAG_STUDY)
    defl = study_row(studies, DEFLATION_STUDY)
    conflict = _num(((studies or {}).get("conflict_example") or {}).get("lambda"))
    stag_share, defl_unrate = (_num((stag or {}).get("simulated_share")), _num((defl or {}).get("unrate_end_p50")))
    return {
        "swing": f"{p['ip_swing_pct']:g}", "smooth": f"{int(p['u_smooth_months'])}", "rise": f"{p['u_rise_pp']:g}",
        "severe": f"{p['severe_ip_decline_pct']:g}", "high": f"{p['high_inflation_pct']:g}",
        "neg": f"{p['cpi_neg_pct']:g}", "year": _db_year(F), "support_months": str(SUPPORT_MONTHS),
        "support_lines": "；".join(lines), "recession_runs": recession_runs,
        "depression_eff": f"{dep['regime_effective_episodes']:.1f}",
        "depression_span": _span(top) if top else "", "depression_max": f"{dep['max_delivered']:.2f}",
        "depression_rare": rare, "longest": str(longest),
        "f_cpi_lo": f"{view_support(F, 'f_cpi', SUPPORT_MONTHS, BLOCK)[0] * 12.0:.3f}" if "f_cpi" in F.columns else "—",
        "deflation_recipe": (f"（只加这两项时 {DEFLATION_STUDY['months']} 个月后失业率中位数是 {defl_unrate:.1f}%）"
                             if defl_unrate is not None else ""),
        "stag_catch_up": (f"：{STAG_STUDY['months']} 个月阶段里 stagflation {STAG_STUDY['views'][0]['mean']:g} 只有 "
                          f"{stag_share:.0%} 的模拟月份是滞胀" if stag_share is not None
                          else "：阶段越短，模拟路径里这类体制的占比比抽中月份里的占比低得越多"),
        "conflict": f"（λ {conflict:.2f}）" if conflict is not None else "",
    }


def translator_rule(db, studies: dict | None = None) -> str:
    """The prompt section 「## 宏观体制（regime 字段）」, deterministic for a build."""
    return TRANSLATOR_RULE.format(**translator_fields(db, studies))
