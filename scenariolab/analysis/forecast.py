"""One-year-ahead forecasts: the rolling track record and the live forecast from today.

Several years ahead nothing can be forecast without a theme: the scenario
library and the conditional backtest windows supply one (a narrative turned
into factor directions). What can be forecast without one is the next 12
months, and that is what is scored here. At every year-end origin from
FIRST_ORIGIN on (plus the latest month with a full year after it), the
engine is refit on months up to the origin only and forecasts the 12-month
total return of every asset:

  uncond   a random historical year: one 12-month block drawn under the prior
  analog   the years that followed historical starting points whose state
           (KEY_STATES, standardized on the training months) resembled the
           state at the origin: kernel weights on block starts, bandwidth set
           so the weights rest on a given number of starting months. That
           number is itself chosen walk-forward: the ENS_GRID value with the
           best CRPS on origins up to SPLIT, then checked on the later ones
  naive    benchmark: a normal distribution with the training mean and sd of
           12-month returns

Draws go through the full engine (re-anchoring, rate scaling, reversion,
today's transmission, residuals, CMA). Scores per asset: 90% coverage, PIT,
CRPS (a proper score for the whole distribution, lower is better), error of
the median, the hit rate of the sign of the excess return over cash, and the
cross-sectional rank correlation of predicted vs realized returns (IC).

Years containing a config.EXOGENOUS_SHOCKS event are flagged and every score
is also reported without them. Each large miss is attributed: exogenous
shock, macro surprise (the realized 12-month change of a key macro variable
fell outside the forecast's 90% range), or asset-pricing gap (the macro came
in as expected but the asset did not: valuation or earnings moves that the
factors do not carry).

The live forecast applies uncond and analog at the last observed month and
adds the probability-weighted mixture of the scenario library's first 12
months (results/scenario_paths.npz).

Reconciliation. The mixture's first year is more bearish than the scored
analog forecast because the shock scenarios start their shocks in month one.
`reconcile` splits the gap in means and in P(loss) exactly by scenario
(probability x the scenario's first-year value minus the analog value), and
re-runs each built-in from its detail spec (bit-identical to the npz, checked)
to get the 12-month macro changes of the same paths. The calibrated
probabilities are entropy pooling over the 13 scenarios with the model
posterior as the prior and, as views, the analog method's live mean 12-month
change of each macro variable the analog method forecasts better than both the
unconditional draw and the naive normal of past 12-month changes (`macro_skill`:
CRPS skill > 0 over all origins and in both halves, paired t >= SKILL_MIN_T,
against each). The naive benchmark matters: both engine methods draw the
historical months' CPI changes without re-anchoring them, so their CPI y/y
forecasts share a large upward bias, and beating the unconditional draw alone
says nothing about the forecast's quality. Without such a variable nothing is
calibrated.

Ledger. The build's own run records its live forecast in data/ledger/ (one
file per origin, written atomically and never replaced) and scores every
entry whose 12 months the panel now covers: the true out-of-sample record. It
only appends in time order (no origin older than one recorded) and only while
the forecast is real time (recorded at most LEDGER_MAX_DELAY_MONTHS after its
origin month); a scenario mixture from paths of another start month is never
shown, reconciled or recorded.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import config, jsonio
from ..context import EngineContext
from ..engine.ep import effective_number, entropy_pool, kl_divergence
from ..engine.factor_model import AssetFactorModel, ModelSpec
from ..engine.pooling import SHRINK_STEPS, start_prior
from ..engine.sampler import BlockProbs
from ..engine.simulate import (MonthDB, capital_market_assumptions, make_anchor, simulate, valuation_settings,
                               valuation_setup)
from ..portfolio.metrics import weighted_percentile

HORIZON = 12
FIRST_ORIGIN = "1984-12"
ORIGIN_MONTH = 12
KEY_STATES = ("cpi_yoy", "unrate", "real_ff", "term_spread", "spread", "div_yield", "ip_yoy", "oil_yoy")
ANALOG_ENS = 72.0          # default when no selection runs (e.g. a custom set of origins)
ENS_GRID = (36, 72, 144, 288, 576)
SPLIT = "2004-12"          # the analog bandwidth is chosen on origins up to here and checked after
DISPLAY_ENS = 72.0         # narrow kernel used only to list the closest historical starting points
N_PATHS = 1000
SEED = 23
PCTS = (5, 25, 50, 75, 95)
ASSETS = list(config.ASSET_IDS)
NAMES = {a["id"]: a["name"] for a in config.ASSETS}
RISK_ASSETS = [a for a in ASSETS if a not in ("treasury_10y", "crude_oil", "cash")]
RANKED_ASSETS = [a for a in ASSETS if a != "cash"]
# realized 12-month macro change: (panel column, "diff" in its units or "log" change)
MACRO = {"unrate": ("lvl_unrate", "diff"), "ff": ("lvl_ff", "diff"), "cpi_yoy": ("s_cpi_yoy", "diff"),
         "spread": ("s_spread", "diff"), "oil": ("lvl_oil", "log")}
MACRO_CN = {"unrate": "失业率", "ff": "政策利率", "cpi_yoy": "CPI 同比", "spread": "信用利差", "oil": "油价"}
METHOD_CN = {"analog": "相似起点", "uncond": "无条件", "naive": "历史均值", "scenarios": "情景加权",
             "scenarios_calibrated": "情景加权（校准）"}
# A macro change is forecastable (and may become a calibration view) when the analog CRPS beats both
# benchmarks (SKILL_REFS: the unconditional draw and the naive normal) over all origins and in both halves,
# with a paired t of at least this against each (macro_skill).
SKILL_MIN_T = 1.0
# The live forecast ledger: one file per origin, append-only (data/ is git-ignored).
LEDGER_DIR = config.DATA_DIR / "ledger"
LEDGER_SCHEMA = 1
LEDGER_TAUS = (np.arange(100) + 0.5) / 100.0      # stored quantile levels: CRPS and PIT are scored on them
LEDGER_METHODS = ("analog", "uncond", "scenarios", "scenarios_calibrated", "naive")
# A build records its forecast only while it is real time: at most this many months after the origin month
# (the panel's last month is usually 1-2 months old when the build runs).
LEDGER_MAX_DELAY_MONTHS = 3
METHODS = {
    "uncond": {"name": "无条件（随机历史年份）", "description": "从训练期所有历史月份里随机抽一段连续 12 个月，经引擎重新锚定到起点水平"},
    "analog": {"name": "相似起点（类比法）", "description": "按起点宏观状态（通胀、失业、实际利率、期限利差、信用利差、估值、增长、油价）"
               "与当时的相似度给历史起点加权，抽它们之后的 12 个月，经引擎重新锚定；加权的集中程度由滚动验证选出"},
    "naive": {"name": "朴素基准（历史均值）", "description": "训练期 12 个月收益的均值和标准差构成的正态分布，不看任何宏观信息"},
    "scenarios": {"name": "情景加权（仅今天）", "description": "情景库每个情景前 12 个月的路径，按情景后验概率加权"},
}
NOTES = [
    "一年期是能在没有情景主题的情况下认真打分的最长期限；多年期路径必须先给出情景主题和因子方向（见“未来情景”和回测窗口）。",
    "每个起点只用当时及以前的数据：重新估计传导模型、历史月份库、起点水平和估值调整；被抽到的 12 个月都在起点之前。",
    "外生冲击（战争、疫情、恐怖袭击、石油禁运）发生的年份被单独标出，成绩同时给出“全部年份”和“剔除外生冲击年份”两版。",
    "CRPS 衡量整个预测分布与实际值的距离（越小越好），同时惩罚偏差和过宽/过窄的区间；技能分 = 1 − CRPS/基准 CRPS。",
    "现金的 12 个月收益在起点基本已知（当时利率），所以不参与方向命中率和排序相关。",
]


# ---------------------------------------------------------------- helpers

def shock_names(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Curated exogenous shocks overlapping [start, end]."""
    return [s["name"] for s in config.EXOGENOUS_SHOCKS
            if pd.Timestamp(s["start"]) <= end and pd.Timestamp(s["end"]) >= start]


def forecast_origins(panel: pd.DataFrame, first: str = FIRST_ORIGIN, month: int = ORIGIN_MONTH) -> list[pd.Timestamp]:
    """Year-end origins with a full realized year after them, plus the latest such month."""
    last = panel.index.max()
    full = [d for d in panel.loc[first:].index if d + pd.DateOffset(months=HORIZON) <= last]
    out = [d for d in full if d.month == month]
    if full and full[-1] not in out:
        out.append(full[-1])
    return out


def pit(samples: np.ndarray, y: float) -> float:
    return float(((samples < y).sum() + 0.5 * (samples == y).sum()) / samples.size)


def crps_samples(samples: np.ndarray, y: float) -> float:
    """CRPS of an empirical distribution: E|X - y| - E|X - X'| / 2."""
    x = np.sort(samples)
    n = x.size
    spread = 2.0 * ((2 * np.arange(n) - n + 1) * x).sum() / (n * n)
    return float(np.abs(x - y).mean() - 0.5 * spread)


def crps_normal(mu: float, sd: float, y: float) -> float:
    z = (y - mu) / sd
    return float(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))


def _bands(samples: np.ndarray, weights=None) -> dict:
    q = weighted_percentile(samples, PCTS, weights)
    out = {f"p{p}": float(v) for p, v in zip(PCTS, q)}
    w = None if weights is None else np.asarray(weights, float)
    out["mean"] = float(np.average(samples, weights=w))
    out["prob_loss"] = float(np.average(samples < 0, weights=w))
    return out


def _macro_bands(samples: np.ndarray) -> dict:
    """5% / 50% / 95% and the mean of a simulated 12-month macro change."""
    s = np.asarray(samples, float)
    return {"p5": float(np.percentile(s, 5)), "p50": float(np.median(s)), "p95": float(np.percentile(s, 95)),
            "mean": float(s.mean())}


def state_at(panel: pd.DataFrame, date) -> np.ndarray:
    cols = [f"s_{k}" for k in KEY_STATES]
    return panel.loc[:date, cols].ffill().iloc[-1].to_numpy(float)


def analog_weights(db: MonthDB, state: np.ndarray, block: int = HORIZON, ens: float = ANALOG_ENS) -> tuple[np.ndarray, dict]:
    """Kernel weights over block starts s by how close the state at s - 1 (the origin) is to `state`.

    Distances are mean squared z-scores over KEY_STATES (standardized on the
    database months); the Gaussian bandwidth is set by bisection so the
    weights rest on about `ens` starting months. Start 0 has no origin.
    """
    X = db.frame[[f"s_{k}" for k in KEY_STATES]].to_numpy(float)
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.nan_to_num((X - mu) / sd)
    z_now = np.nan_to_num((state - mu) / sd)
    n_starts = len(db) - block + 1
    d2 = np.full(n_starts, np.inf)
    d2[1:] = ((Z[: n_starts - 1] - z_now) ** 2).mean(axis=1)          # origin of start s is month s - 1
    base = start_prior(db.prior(), block, db.rare)
    base[0] = 0.0
    finite = np.isfinite(d2)
    raw = d2.copy()                                                  # mean squared z distance
    d2 = np.where(finite, d2 - d2[finite].min(), np.inf)             # shifted: avoids underflow at small bandwidths

    def weights(h: float) -> np.ndarray:
        w = base * np.exp(-d2 / (2.0 * h * h))
        return w / w.sum()

    lo, hi = np.log(1e-3), np.log(1e3)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if effective_number(weights(np.exp(mid))) < ens:
            lo = mid
        else:
            hi = mid
    h = float(np.exp(hi))
    w = weights(h)
    order = np.argsort(-w)
    top = [{"date": db.dates[s - 1].strftime("%Y-%m"), "weight": float(w[s]), "distance": float(np.sqrt(raw[s]))}
           for s in order[:8] if s > 0 and w[s] > 0]
    z_today = {k: float(z) for k, z in zip(KEY_STATES, (state - mu) / sd)}
    return w, {"bandwidth": h, "ens": effective_number(w), "top": top, "z_today": z_today}


def simulate_year(db: MonthDB, anchor, cma: dict, probs: np.ndarray, n_paths: int, seed: int,
                  valuation=None) -> dict:
    """12-month total returns (n, A), inflation and macro changes from one engine run
    (`valuation`: an engine/simulate.ValuationModel for the valuation-state mode, None = off)."""
    sim = simulate(db, anchor, [BlockProbs(probs, HORIZON)], [HORIZON], n_paths=n_paths, seed=seed,
                   block=HORIZON, cma=cma, valuation=valuation, **config.RESIDUALS)
    growth = np.prod(1.0 + sim.total.astype(float), axis=1) - 1.0
    macro = {k: _macro_change(sim.macro[k], kind) for k, (_, kind) in MACRO.items()}
    infl = sim.macro["cpi"][:, HORIZON].astype(float) / sim.macro["cpi"][:, 0].astype(float) - 1.0
    return {"growth": growth, "inflation": infl, "macro": macro, "assets": sim.assets}


def realized_year(panel: pd.DataFrame, t: pd.Timestamp) -> dict:
    months = panel.loc[t + pd.DateOffset(months=1): t + pd.DateOffset(months=HORIZON)]
    r = months[[f"r_{a}" for a in ASSETS]]
    growth = {a: (float(np.prod(1.0 + r[f"r_{a}"])) - 1.0 if r[f"r_{a}"].notna().all() else None) for a in ASSETS}
    end = months.index[-1]
    macro = {}
    for k, (col, kind) in MACRO.items():
        a, b = panel.at[t, col], panel.at[end, col]
        macro[k] = float(np.log(b / a)) if kind == "log" else float(b - a)
    infl = float(panel.at[end, "lvl_cpi"] / panel.at[t, "lvl_cpi"] - 1.0)
    return {"growth": growth, "macro": macro, "inflation": infl, "end": end}


def naive_forecast(panel: pd.DataFrame, t: pd.Timestamp) -> dict[str, tuple[float, float]]:
    """(mean, sd) of overlapping 12-month returns over the training months, per asset."""
    out = {}
    hist = panel.loc[config.MODEL_START:t, [f"r_{a}" for a in ASSETS]]
    for a in ASSETS:
        lr = np.log1p(hist[f"r_{a}"].dropna())
        g = np.expm1(lr.rolling(HORIZON).sum().dropna())
        out[a] = (float(g.mean()), float(g.std())) if len(g) > 24 else (float("nan"), float("nan"))
    return out


def naive_macro(panel: pd.DataFrame, t: pd.Timestamp) -> dict[str, tuple[float, float]]:
    """(mean, sd) of the 12-month changes of every MACRO variable over the training months up to t: the
    naive benchmark of a macro forecast (about "no change", with the historical spread), as `naive` is for
    returns. The engine's own unconditional draw is no such benchmark: both engine methods draw the
    historical months' CPI changes without re-anchoring them, so they share a large inflation bias."""
    out = {}
    for k, (col, kind) in MACRO.items():
        v = panel.loc[config.MODEL_START:t, col].astype(float)
        ch = (np.log(v / v.shift(HORIZON)) if kind == "log" else v - v.shift(HORIZON)).dropna()
        out[k] = (float(ch.mean()), float(ch.std())) if len(ch) > 24 else (float("nan"), float("nan"))
    return out


# ---------------------------------------------------------------- one origin

def forecast_origin(panel: pd.DataFrame, spec: ModelSpec, t: pd.Timestamp, n_paths: int = N_PATHS,
                    seed: int = SEED, cma_method: str = "valuation_adjusted", analog_ens: float = ANALOG_ENS,
                    valuation: dict | None = None) -> dict:
    """Both engine forecasts made at origin t with data up to t only, and what happened.

    `valuation` (engine/simulate.valuation_settings; None = config.VALUATION): with the valuation
    state on, its CMA and pull (fair value and kappa estimated on the months up to t) replace
    `cma_method`.
    """
    model = AssetFactorModel(spec).fit(panel, config.MODEL_START, t)
    db = MonthDB(panel, model, start=config.MODEL_START, end=t)
    anchor = make_anchor(panel, model, date=t)
    vset, vm = valuation_settings(valuation), None
    if vset:
        cma, vm = valuation_setup(db, anchor, vset)
    else:
        cma = capital_market_assumptions(db, anchor, method=cma_method)
    prior = start_prior(db.prior(), HORIZON, db.rare)
    w, info = analog_weights(db, state_at(panel, t), ens=analog_ens)
    runs = {"uncond": simulate_year(db, anchor, cma, prior, n_paths, seed, valuation=vm),
            "analog": simulate_year(db, anchor, cma, w, n_paths, seed + 1, valuation=vm)}
    real = realized_year(panel, t)
    naive = naive_forecast(panel, t)
    flags = shock_names(t + pd.DateOffset(months=1), real["end"])
    assets = {}
    for j, a in enumerate(ASSETS):
        y = real["growth"][a]
        entry = {"realized": y}
        for key, run in runs.items():
            s = run["growth"][:, j]
            b = _bands(s)
            if y is not None:
                b.update({"pit": pit(s, y), "crps": crps_samples(s, y)})
            entry[key] = b
        mu, sd = naive[a]
        if np.isfinite(mu) and sd > 0:
            entry["naive"] = {"p5": mu - 1.645 * sd, "p50": mu, "p95": mu + 1.645 * sd, "mean": mu}
            if y is not None:
                entry["naive"].update({"pit": float(norm.cdf((y - mu) / sd)), "crps": crps_normal(mu, sd, y)})
        assets[a] = entry
    macro = {k: {"realized": real["macro"][k],
                 **{key: {**_macro_bands(run["macro"][k]), "pit": pit(run["macro"][k], real["macro"][k]),
                          "crps": crps_samples(run["macro"][k], real["macro"][k])}
                    for key, run in runs.items()}}
             for k in MACRO}
    for k, (mu, sd) in naive_macro(panel, t).items():
        y = real["macro"][k]
        if np.isfinite(mu) and sd > 0 and np.isfinite(y):
            macro[k]["naive"] = {"p5": mu - 1.645 * sd, "p50": mu, "p95": mu + 1.645 * sd, "mean": mu,
                                 "pit": float(norm.cdf((y - mu) / sd)), "crps": crps_normal(mu, sd, y)}
    out = {"date": t.strftime("%Y-%m"), "end": real["end"].strftime("%Y-%m"), "shocks": flags,
           "inflation": real["inflation"], "analogs": info["top"][:6], "bandwidth": info["bandwidth"],
           "assets": assets, "macro": macro}
    if vm is not None:
        out["valuation"] = vm.today()
    return out


# ---------------------------------------------------------------- scoring

def _score(rows: list[dict], method: str, assets: list[str]) -> dict:
    out = {}
    for a in assets:
        recs = [r["assets"][a] for r in rows if r["assets"][a]["realized"] is not None and method in r["assets"][a]
                and "pit" in r["assets"][a][method]]
        if not recs:
            continue
        f = [x[method] for x in recs]
        y = np.array([x["realized"] for x in recs])
        out[a] = {"n": len(recs),
                  "coverage90": float(np.mean([(v["p5"] <= yy <= v["p95"]) for v, yy in zip(f, y)])),
                  "pit_mad": float(np.mean([abs(v["pit"] - 0.5) for v in f])),
                  "crps": float(np.mean([v["crps"] for v in f])),
                  "mae_p50": float(np.mean([abs(v["p50"] - yy) for v, yy in zip(f, y)])),
                  "bias_p50": float(np.mean([v["p50"] - yy for v, yy in zip(f, y)]))}
    return out


def _hit_rate(rows: list[dict], method: str) -> dict:
    """Sign of the excess return over cash: predicted (median) vs realized, risk assets pooled."""
    hits, ups = [], []
    for r in rows:
        cash = r["assets"]["cash"]
        for a in RISK_ASSETS:
            e = r["assets"][a]
            if e["realized"] is None or method not in e or cash["realized"] is None:
                continue
            pred = e[method]["p50"] - cash.get(method, cash.get("uncond"))["p50"]
            real = e["realized"] - cash["realized"]
            hits.append(np.sign(pred) == np.sign(real))
            ups.append(real > 0)
    return {"hit_rate": float(np.mean(hits)) if hits else None, "base_rate_up": float(np.mean(ups)) if ups else None,
            "n": len(hits)}


def _rank_ic(rows: list[dict], method: str) -> dict:
    """Mean Spearman correlation across assets of predicted median vs realized return, per origin."""
    ics = []
    for r in rows:
        pairs = [(r["assets"][a][method]["p50"], r["assets"][a]["realized"]) for a in RANKED_ASSETS
                 if r["assets"][a]["realized"] is not None and method in r["assets"][a]]
        if len(pairs) >= 5:
            p, y = np.array(pairs).T
            ics.append(float(pd.Series(p).rank().corr(pd.Series(y).rank())))
    if not ics:
        return {"mean": None, "t": None, "n": 0}
    m, s = float(np.mean(ics)), float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan")
    return {"mean": m, "t": float(m / s * np.sqrt(len(ics))) if s and np.isfinite(s) and s > 0 else None,
            "n": len(ics), "by_origin": ics}


def _avg(scores: dict, assets: list[str], key: str) -> float | None:
    vals = [scores[a][key] for a in assets if a in scores]
    return float(np.mean(vals)) if vals else None


def scores(rows: list[dict]) -> dict:
    out = {}
    for method in ("uncond", "analog", "naive"):
        per = _score(rows, method, ASSETS)
        out[method] = {"assets": per,
                       "risk_avg": {k: _avg(per, RISK_ASSETS, k) for k in ("coverage90", "pit_mad", "crps", "mae_p50")},
                       "all_avg": {k: _avg(per, RANKED_ASSETS, k) for k in ("coverage90", "pit_mad", "crps", "mae_p50")},
                       "direction": _hit_rate(rows, method), "rank_ic": _rank_ic(rows, method)}
    skill = {}
    for a, b in (("analog", "uncond"), ("analog", "naive"), ("uncond", "naive")):
        num, den = out[a]["risk_avg"]["crps"], out[b]["risk_avg"]["crps"]
        skill[f"{a}_vs_{b}"] = (1.0 - num / den) if num is not None and den else None
    out["crps_skill_risk"] = skill
    return out


def attribute_misses(rows: list[dict], method: str = "analog", top: int = 15) -> list[dict]:
    """Largest misses (realized outside the 90% band), each tagged shock / macro surprise / pricing gap."""
    misses = []
    for r in rows:
        for a in RANKED_ASSETS:
            e = r["assets"][a]
            if e["realized"] is None or method not in e or "pit" not in e[method]:
                continue
            f = e[method]
            if f["p5"] <= e["realized"] <= f["p95"]:
                continue
            z = float(norm.ppf(np.clip(f["pit"], 0.001, 0.999)))
            surprises = [k for k, m in r["macro"].items() if not 0.05 <= m[method]["pit"] <= 0.95]
            if r["shocks"]:
                cause, why = "shock", "外生冲击：" + "、".join(r["shocks"])
            elif surprises:
                cause = "macro"
                why = "宏观意外：" + "；".join(
                    f"{MACRO_CN[k]}实际 {_fmt_change(k, r['macro'][k]['realized'])}，预测中位数 "
                    f"{_fmt_change(k, r['macro'][k][method]['p50'])}" for k in surprises)
            else:
                cause, why = "pricing", "宏观大体符合预测，资产本身偏离：估值或盈利的变化，因子里没有"
            misses.append({"origin": r["date"], "asset": a, "realized": e["realized"], "p5": f["p5"], "p50": f["p50"],
                           "p95": f["p95"], "z": z, "cause": cause, "why": why})
    misses.sort(key=lambda m: -abs(m["z"]))
    return misses[:top]


def _fmt_change(k: str, v: float) -> str:
    if MACRO[k][1] == "log":
        return f"{np.expm1(v):+.0%}"
    return f"{v:+.2f}pp"                 # unrate, ff, cpi_yoy and spread changes are all percentage points


# ---------------------------------------------------------------- macro skill (what the calibration may use)

SKILL_REFS = ("uncond", "naive")          # the benchmarks a forecastable macro change must beat
SKILL_REF_CN = {"uncond": "无条件抽样", "naive": "历史均值基准"}


def _skill_part(rows: list[dict], k: str, ref: str = "uncond") -> dict:
    """Mean CRPS of the analog forecast of macro change k and of the benchmark `ref`, the skill and the paired t."""
    if len(rows) < 2:
        return {"n": len(rows), "crps_analog": None, f"crps_{ref}": None, "skill": None, "t": None}
    a = np.array([r["macro"][k]["analog"]["crps"] for r in rows], float)
    u = np.array([r["macro"][k][ref]["crps"] for r in rows], float)
    d = u - a
    sd = float(d.std(ddof=1))
    return {"n": len(rows), "crps_analog": float(a.mean()), f"crps_{ref}": float(u.mean()),
            "skill": float(1.0 - a.mean() / u.mean()) if u.mean() > 0 else None,
            "t": float(d.mean() / sd * np.sqrt(len(d))) if sd > 0 else None}


def _beats(parts: dict, min_t: float) -> bool:
    a, v, t = parts.get("all") or {}, parts.get("val") or {}, parts.get("test") or {}
    return bool(all(x.get("skill") is not None and x["skill"] > 0 for x in (a, v, t))
                and a.get("t") is not None and a["t"] >= min_t)


def macro_skill(rows: list[dict], split: str = SPLIT, min_t: float = SKILL_MIN_T) -> dict:
    """CRPS skill of the analog method's 12-month macro forecasts against two benchmarks: the engine's
    unconditional draw (does the analog weighting add anything?) and the naive normal of the training
    months' 12-month changes (is the forecast any good at all? the two engine methods share biases, e.g.
    both draw the historical months' inflation without re-anchoring it).

    A macro change counts as forecastable when, against each benchmark, the analog CRPS is lower over all
    origins and in both halves (origins up to `split` and after) and the paired t statistic of the
    per-origin CRPS gain is at least `min_t`. Only those may become calibration views. The comparison with
    the unconditional draw stays in all / val / test / ex_shock; the naive one is under `naive`.
    """
    cut = pd.Timestamp(split)
    out = {}
    for k in MACRO:
        entry = {"name": MACRO_CN[k], "unit": "log" if MACRO[k][1] == "log" else "pp", "coverage90": {}, "beats": {}}
        for ref in SKILL_REFS:
            recs = [r for r in rows if all("crps" in (r["macro"][k].get(m) or {}) for m in ("analog", ref))]
            sel = {"all": recs, "val": [r for r in recs if pd.Timestamp(r["date"]) <= cut],
                   "test": [r for r in recs if pd.Timestamp(r["date"]) > cut],
                   "ex_shock": [r for r in recs if not r["shocks"]]}
            parts = {tag: _skill_part(s, k, ref) for tag, s in sel.items()}
            if ref == "uncond":
                entry.update(parts)
            else:
                entry[ref] = parts
            entry["beats"][ref] = _beats(parts, min_t)
            for m in ("analog", ref):
                if m not in entry["coverage90"]:
                    entry["coverage90"][m] = (float(np.mean([r["macro"][k][m]["p5"] <= r["macro"][k]["realized"]
                                                             <= r["macro"][k][m]["p95"] for r in recs])) if recs else None)
        entry["skilled"] = all(entry["beats"].values())
        out[k] = entry
    rule = (f"相似起点法对一个宏观变量 12 个月变化的预测，要同时胜过无条件抽样和历史均值基准（训练期 12 个月变化的均值与标准差构成的正态分布）："
            f"对每一个基准，都要在全部 {len(rows)} 个起点上、并且在 {split[:4]} 年及以前和以后两半 CRPS 都更低，"
            f"逐起点 CRPS 改善的配对 t 值不低于 {min_t:g}，才算有预测能力")
    return {"rule": rule, "split": split, "min_t": min_t, "n_origins": len(rows), "refs": list(SKILL_REFS),
            "vars": out, "skilled": [k for k in MACRO if out[k]["skilled"]]}


def _skill_verdicts(skill: dict) -> str:
    """Why each macro change did or did not pass, in one sentence: 'CPI 同比胜过无条件抽样（+1.3%），但比历史均值基准差 45%（…）'."""
    out = []
    for k, v in (skill.get("vars") or {}).items():
        u, n = v.get("all") or {}, (v.get("naive") or {}).get("all") or {}
        beats = v.get("beats") or {}

        def part(ref, s):
            x = s.get("skill")
            if x is None:
                return f"没有{SKILL_REF_CN[ref]}的成绩"
            word = "胜过" if beats.get(ref) else ("略好于" if x > 0 else "不如")
            return f"{word}{SKILL_REF_CN[ref]}（CRPS 技能 {x:+.1%}，t = {s['t']:.1f}）" if s.get("t") is not None \
                else f"{word}{SKILL_REF_CN[ref]}（CRPS 技能 {x:+.1%}）"
        out.append(f"{v.get('name', k)}{part('uncond', u)}、{part('naive', n)}")
    return "；".join(out)


# ---------------------------------------------------------------- live forecast: the scenario library's first year

def _macro_change(path: np.ndarray, kind: str, h: int = HORIZON) -> np.ndarray:
    p = np.asarray(path, float)
    return np.log(p[:, h] / p[:, 0]) if kind == "log" else p[:, h] - p[:, 0]


def year_one_paths(results_dir=None) -> dict | None:
    """The first 12 months of every built-in scenario (results/scenario_paths.npz), or None if not built:
    the monthly totals (float32, for the reproduction check), 12-month total returns (S, N, A), inflation
    (S, N), the probabilities stored with the paths (normalized), ids, assets and paths per scenario."""
    path = Path(results_dir or config.RESULTS_DIR) / "scenario_paths.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        total = np.array(z["total"][:, :, :HORIZON, :])
        cpi = z["cpi"][:, :, [0, HORIZON]].astype(float)
        probs = z["probs"].astype(float)
        ids, assets = [str(x) for x in z["ids"]], [str(x) for x in z["assets"]]
        start = str(z["labels"][0]) if "labels" in z.files and len(z["labels"]) else None
    growth = np.prod(1.0 + total.astype(float), axis=2) - 1.0                  # (S, N, A)
    return {"total": total, "growth": growth, "infl": cpi[:, :, 1] / cpi[:, :, 0] - 1.0,
            "probs": probs / probs.sum(), "probs_raw": probs, "ids": ids, "assets": assets,
            "n_paths": int(growth.shape[1]), "start": start}


def _scenario_vector(year_one: dict, probs: dict | None) -> np.ndarray:
    if probs is None:
        return year_one["probs"]
    p = np.array([max(float(probs.get(i, 0.0) or 0.0), 0.0) for i in year_one["ids"]])
    return p / p.sum() if p.sum() > 0 else year_one["probs"]


def mixture_bands(year_one: dict, probs: dict | None = None, assets: list[str] | None = None) -> dict:
    """Bands of the probability-weighted mixture of the scenarios' first 12 months per asset
    (`probs` {id: p}; None = the probabilities stored with the paths)."""
    g, infl = year_one["growth"], year_one["infl"]
    N = g.shape[1]
    w = np.repeat(_scenario_vector(year_one, probs) / N, N)
    out = {}
    for j, a in enumerate(year_one["assets"]):
        if assets is not None and a not in assets:
            continue
        b = _bands(g[:, :, j].ravel(), w)
        b["real_p50"] = float(weighted_percentile(((1 + g[:, :, j]) / (1 + infl) - 1).ravel(), 50, w))
        out[a] = b
    return out


def scenario_mixture(results_dir=None, year_one: dict | None = None) -> dict | None:
    """Probability-weighted 12-month returns from the scenario library's paths (None if not built)."""
    y1 = year_one or year_one_paths(results_dir)
    if y1 is None:
        return None
    g, ids = y1["growth"], y1["ids"]
    return {"assets": mixture_bands(y1),
            "per_scenario": {a: {sid: float(np.median(g[s, :, j])) for s, sid in enumerate(ids)}
                             for j, a in enumerate(y1["assets"])},
            "probs": dict(zip(ids, y1["probs_raw"].tolist()))}


def scenario_year_one_macro(ctx: EngineContext, year_one: dict, results_dir=None) -> dict:
    """12-month macro changes along the scenarios' own paths.

    scenario_paths.npz keeps returns and CPI only, so each built-in is re-run from its detail spec
    (results/scenarios/<id>.json, the validated spec of the same build) with the library's seed and
    path count; that reproduces the npz exactly, which `reproduced` checks on the first 12 months.
    Returns {"macro": {id: {var: samples}}, "reproduced": {id: bool}, "missing": [id]}.
    """
    from ..scenarios import library, runner
    rdir = Path(results_dir or config.RESULTS_DIR)
    out, reproduced, missing = {}, {}, []
    for s, sid in enumerate(year_one["ids"]):
        try:
            spec = jsonio.load(rdir / "scenarios" / f"{sid}.json")["spec"]
            _, sim = runner.run_spec(ctx, spec, n_paths=year_one["n_paths"], seed=library.seed_for(sid))
        except (OSError, KeyError, TypeError, ValueError):
            missing.append(sid)
            continue
        reproduced[sid] = bool(np.array_equal(sim.total[:, :HORIZON], year_one["total"][s]))
        out[sid] = {k: _macro_change(sim.macro[k], kind) for k, (_, kind) in MACRO.items()}
    return {"macro": out, "reproduced": reproduced, "missing": missing}


def _reachable(p: np.ndarray, A: np.ndarray, base: np.ndarray, b: np.ndarray, floor: float = 1e-3) -> float:
    """Largest lambda in [0, 1] for which some probabilities q >= floor x p (summing to 1) give
    A q = base + lambda (b - base): a small linear programme. The floor keeps the views strictly inside
    what the scenarios span, where entropy pooling has a solution."""
    from scipy.optimize import linprog
    K, S = A.shape
    c = np.zeros(S + 1)
    c[-1] = -1.0                                                   # maximize lambda
    A_eq = np.zeros((K + 1, S + 1))
    A_eq[:K, :S], A_eq[:K, -1], A_eq[K, :S] = A, -(b - base), 1.0
    res = linprog(c, A_eq=A_eq, b_eq=np.concatenate([base, [1.0]]),
                  bounds=[(floor * x, 1.0) for x in p] + [(0.0, 1.0)], method="highs")
    if res.status != 0:
        return 0.0
    lam = float(np.clip(res.x[-1], 0.0, 1.0))
    return 1.0 if lam > 1.0 - 1e-9 else lam


def calibrate_probs(prior: dict, means: dict, targets: dict, variables) -> dict:
    """Entropy pooling over scenarios: the probabilities q closest to `prior` in relative entropy whose
    probability-weighted means of `variables` equal `targets`.

    prior {id: p}; means {id: {var: mean 12-month change}}; targets {var: value}. A view set the scenario
    means cannot meet is shrunk toward what the prior already implies (target_used = prior mean +
    lambda (target - prior mean), the largest reachable lambda in [0, 1]), as the phase pooling does.
    """
    ids = [i for i in prior if i in means]
    p = np.array([max(float(prior[i]), 0.0) for i in ids])
    p = p / p.sum()
    use = [v for v in variables if v in targets and np.isfinite(targets[v])
           and all(np.isfinite(means[i].get(v, np.nan)) for i in ids)]
    A = np.array([[float(means[i][v]) for i in ids] for v in use]).reshape(len(use), len(ids))
    b = np.array([float(targets[v]) for v in use])
    base = A @ p

    def solve(lam: float):
        if not use:
            return p.copy(), True, 0.0
        return entropy_pool(p, A, base + lam * (b - base))

    # how far toward the targets some mix of scenarios can go (an LP), so the entropy pooling is only
    # ever asked for reachable means: an unreachable view would run the solver to its iteration limit
    lam = _reachable(p, A, base, b) if use else 1.0
    q, ok, viol = solve(lam)
    if not ok:                                       # numerically at the edge: bisect below it
        lo, hi, best = 0.0, lam, (p.copy(), True, 0.0)
        for _ in range(SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            res = solve(mid)
            if res[1]:
                lo, best = mid, res
            else:
                hi = mid
        lam, (q, ok, viol) = lo, best
    q = q / q.sum()
    return {"ids": ids, "vars": use, "prior": dict(zip(ids, p.tolist())), "probs": dict(zip(ids, q.tolist())),
            "kl": max(kl_divergence(q, p), 0.0), "ens": effective_number(q), "ens_prior": effective_number(p),
            "converged": bool(ok), "max_violation": float(viol), "shrink": float(lam),
            "views": [{"var": v, "target": float(b[j]), "target_used": float(base[j] + lam * (b[j] - base[j])),
                       "prior_mean": float(base[j]), "posterior_mean": float(A[j] @ q)} for j, v in enumerate(use)]}


def _fmt_macro_mean(k: str, v: float) -> str:
    return f"{v:+.0%}" if MACRO[k][1] == "log" else f"{v:+.2f} pp"


def _names(results_dir=None) -> tuple[dict, dict]:
    """({id: name}, {id: {asset: 15-year median CAGR}}) from scenarios.json (empty when missing)."""
    try:
        index = jsonio.load(Path(results_dir or config.RESULTS_DIR) / "scenarios.json")
    except (OSError, ValueError):
        return {}, {}
    names, cagr = {}, {}
    for s in index.get("scenarios") or []:
        names[s["id"]] = s.get("name") or s["id"]
        cagr[s["id"]] = {a: (x or {}).get("cagr_p50") for a, x in (s.get("assets") or {}).items()}
    return names, cagr


def reconcile(live_assets: dict, runs: dict, year_one: dict, y1_macro: dict, skill: dict | None,
              results_dir=None) -> tuple[dict, dict]:
    """Why the probability-weighted first year differs from the scored analog forecast, and the calibrated
    scenario probabilities. Returns (reconciliation, calibration).

    The gap in means and in P(loss) is split exactly by scenario: p_s x (scenario s's first-year value -
    the analog value) sums to (mixture - analog). The calibration is entropy pooling over the 13
    scenarios with the model posterior as the prior and, as views, the analog method's live mean
    12-month changes of the macro variables that `skill` finds forecastable; with none, nothing is
    calibrated and `calibration.available` is false.
    """
    ids, g, N = year_one["ids"], year_one["growth"], year_one["n_paths"]
    names, cagr15 = _names(results_dir)
    prior = dict(zip(ids, year_one["probs"].tolist()))
    have = [sid for sid in ids if sid in y1_macro["macro"]]
    means = {sid: {k: float(np.mean(y1_macro["macro"][sid][k])) for k in MACRO} for sid in have}
    targets = {k: float(np.mean(runs["analog"]["macro"][k])) for k in MACRO}
    skilled = list((skill or {}).get("skilled") or [])
    p_vec = year_one["probs"]
    mix_macro = {k: (float(sum(prior[s] * means[s][k] for s in have) / sum(prior[s] for s in have)) if have else None)
                 for k in MACRO}

    # ---- calibration (only on forecastable macro changes)
    reason = None
    if not skill:
        reason = "没有一年期滚动预测的宏观成绩，无法判断哪些宏观变量可以用来校准"
    elif len(have) < len(ids):
        reason = f"缺少情景详情文件（{'、'.join(s for s in ids if s not in have)}），无法得到这些情景第一年的宏观路径"
    elif not skilled:
        reason = (f"相似起点法在 {skill['n_origins']} 个起点上没有一个宏观变量同时胜过无条件抽样和历史均值基准，所以不做校准："
                  "校准只能对齐有预测能力的变量，否则只是把一种没有被验证过的宏观判断强加给情景概率")
    cal = calibrate_probs(prior, means, targets, skilled) if reason is None else None
    q_vec = _scenario_vector(year_one, cal["probs"]) if cal else None
    cal_bands = mixture_bands(year_one, cal["probs"]) if cal else None

    # ---- the gap per asset, split by scenario
    summary, contrib, contrib_loss = {}, {}, {}
    mix_bands = mixture_bands(year_one)
    for j, a in enumerate(year_one["assets"]):
        an = (live_assets.get(a) or {}).get("analog")
        if not an:
            continue
        mu = g[:, :, j].mean(axis=1)
        pl = (g[:, :, j] < 0).mean(axis=1)
        contrib[a] = {sid: float(p_vec[s] * (mu[s] - an["mean"])) for s, sid in enumerate(ids)}
        contrib_loss[a] = {sid: float(p_vec[s] * (pl[s] - an["prob_loss"])) for s, sid in enumerate(ids)}
        mb = mix_bands[a]
        row = {"analog": {k: an[k] for k in ("p50", "mean", "prob_loss")},
               "scenarios": {k: mb[k] for k in ("p50", "mean", "prob_loss")},
               "gap_p50": mb["p50"] - an["p50"], "gap_mean": mb["mean"] - an["mean"],
               "gap_prob_loss": mb["prob_loss"] - an["prob_loss"], "calibrated": None}
        if cal_bands:
            cb = cal_bands[a]
            row["calibrated"] = {k: cb[k] for k in ("p50", "mean", "prob_loss")}
            row["calibrated_gap_p50"] = cb["p50"] - an["p50"]
        summary[a] = row

    per_scenario = []
    for s, sid in enumerate(ids):
        y1 = {a: {"p50": float(np.median(g[s, :, j])), "mean": float(g[s, :, j].mean()),
                  "prob_loss": float((g[s, :, j] < 0).mean())} for j, a in enumerate(year_one["assets"])}
        c15 = cagr15.get(sid, {})
        per_scenario.append({
            "id": sid, "name": names.get(sid, sid), "prob": prior[sid],
            "prob_calibrated": cal["probs"].get(sid) if cal else None,
            "year1": y1, "cagr15_p50": c15, "macro": means.get(sid),
            "front_load": {a: (y1[a]["p50"] - c15[a]) if isinstance(c15.get(a), (int, float)) else None for a in y1}})

    macro = {k: {"name": MACRO_CN[k], "unit": "log" if MACRO[k][1] == "log" else "pp",
                 "analog_mean": targets[k], "uncond_mean": float(np.mean(runs["uncond"]["macro"][k])),
                 "scenarios_mean": mix_macro[k],
                 "calibrated_mean": (float(sum(cal["probs"][s] * means[s][k] for s in have)) if cal else None),
                 "skilled": k in skilled,
                 "skill": ((skill or {}).get("vars") or {}).get(k, {}).get("all", {}).get("skill"),
                 "t": ((skill or {}).get("vars") or {}).get(k, {}).get("all", {}).get("t"),
                 "skill_naive": ((skill or {}).get("vars") or {}).get(k, {}).get("naive", {}).get("all", {}).get("skill"),
                 "t_naive": ((skill or {}).get("vars") or {}).get(k, {}).get("naive", {}).get("all", {}).get("t")}
             for k in MACRO}

    # ---- what each macro view alone (and all five) would do: a what-if, never offered as probabilities
    sensitivity = []
    if len(have) == len(ids):
        for vs in [[k] for k in MACRO] + [list(MACRO)]:
            c = calibrate_probs(prior, means, targets, vs)
            ub = mixture_bands(year_one, c["probs"], assets=["us_equity"]).get("us_equity", {})
            sensitivity.append({"vars": vs, "skilled": all(v in skilled for v in vs), "kl": c["kl"], "ens": c["ens"],
                                "converged": c["converged"], "shrink": c["shrink"],
                                "us_equity": {k: ub.get(k) for k in ("p50", "mean", "prob_loss")}})

    reproduced = y1_macro.get("reproduced") or {}
    rec = {"assets": list(summary), "summary": summary, "contrib_mean": contrib, "contrib_prob_loss": contrib_loss,
           "scenarios": per_scenario, "macro": macro, "sensitivity": sensitivity,
           "n_paths": N, "paths_reproduced": bool(reproduced) and all(reproduced.values()) and not y1_macro.get("missing"),
           "not_reproduced": [s for s, ok in reproduced.items() if not ok], "missing": y1_macro.get("missing") or []}
    calibration = {"available": cal is not None, "reason": reason, "rule": (skill or {}).get("rule"),
                   "verdicts": _skill_verdicts(skill) if skill else None,
                   "candidates": list(MACRO), "skilled": skilled, "prior": prior}
    if cal:
        shifts = sorted(({"id": sid, "name": names.get(sid, sid), "prior": prior[sid], "calibrated": cal["probs"][sid],
                          "shift": cal["probs"][sid] - prior[sid]} for sid in ids), key=lambda r: -abs(r["shift"]))
        calibration.update({k: cal[k] for k in ("probs", "kl", "ens", "ens_prior", "converged", "max_violation", "shrink")})
        calibration["views"] = [{**v, "name": MACRO_CN[v["var"]]} for v in cal["views"]]
        calibration["shifts"] = shifts
    calibration["text"] = _calibration_line(calibration, skill, len(ids))
    rec["text"] = _reconcile_text(rec, calibration, skill, q_vec is not None)
    return rec, calibration


def _sp(name: str) -> str:
    """A name that follows Chinese text: a leading Latin word gets a space before it ("对齐 CPI 同比")."""
    return f" {name}" if name[:1].isascii() else name


def _calibration_line(cal: dict, skill: dict | None, n_scen: int) -> str:
    """One line for the probability editor: what the calibrated set is, with its KL."""
    if not cal["available"]:
        return f"未校准：{cal['reason']}。"
    views = "、".join(f"{v['name']}变化（{_fmt_macro_mean(v['var'], v['target_used'])}）" for v in cal["views"])
    shrink = f"，视图按 λ = {cal['shrink']:.2f} 收缩" if cal["shrink"] < 0.999 else ""
    return (f"以模型后验为先验做熵池化，让 {n_scen} 个情景第一年的概率加权均值等于相似起点法的预测：{views}{shrink}；"
            f"只用相似起点法在 {(skill or {}).get('n_origins', '—')} 个起点上有预测能力的宏观变量。"
            f"KL = {cal['kl']:.3f}，有效情景数 {cal['ens_prior']:.1f} → {cal['ens']:.1f}。")


def _reconcile_text(rec: dict, cal: dict, skill: dict | None, calibrated: bool) -> list[str]:
    S = rec["summary"]
    if "us_equity" not in S:
        return []
    eq = S["us_equity"]
    an, mx = eq["analog"], eq["scenarios"]
    out = [f"同一个起点、同一套引擎，情景加权的第一年比相似起点法悲观：美股大盘中位数 {mx['p50']:+.1%} 对 {an['p50']:+.1%}，"
           f"均值 {mx['mean']:+.1%} 对 {an['mean']:+.1%}，P(亏损) {mx['prob_loss']:.0%} 对 {an['prob_loss']:.0%}。"]
    if "nasdaq" in S:
        nq = S["nasdaq"]
        out[0] += f"纳指100 中位数 {nq['scenarios']['p50']:+.1%} 对 {nq['analog']['p50']:+.1%}。"
    rows = rec["scenarios"]
    um = rec["macro"]["unrate"]
    shock = [r for r in rows if r["macro"] and r["macro"]["unrate"] - um["analog_mean"] >= 0.25
             and r["year1"]["us_equity"]["p50"] < an["p50"]]
    shock.sort(key=lambda r: r["year1"]["us_equity"]["p50"])
    if shock:
        parts = []
        for r in shock:
            c = r["cagr15_p50"].get("us_equity")
            parts.append(f"{r['name']}（美股第一年中位数 {r['year1']['us_equity']['p50']:+.0%}"
                         + (f"，15 年年化 {c:+.1%}" if isinstance(c, (int, float)) else "") + "）")
        p_shock = sum(r["prob"] for r in shock)
        avg_u = sum(r["prob"] * r["macro"]["unrate"] for r in shock) / p_shock
        out.append(f"差距来自冲击类情景把冲击放在第一年：{'、'.join(parts)}。这 {len(shock)} 个情景合计概率 {p_shock:.0%}，"
                   f"第一年失业率平均上升 {avg_u:.2f} pp（按概率加权），相似起点法只有 {um['analog_mean']:+.2f} pp；"
                   f"13 个情景加权后第一年失业率平均变化 {um['scenarios_mean']:+.2f} pp。")
    c = rec["contrib_mean"]["us_equity"]
    names = {r["id"]: r["name"] for r in rows}
    neg = sorted((v, s) for s, v in c.items() if v < 0)[:3]
    pos = sorted(((v, s) for s, v in c.items() if v > 0), reverse=True)[:2]
    out.append(f"按情景拆解美股的均值差距 {eq['gap_mean']:+.1%}（概率 × 该情景第一年均值与相似起点法均值之差，各项合计正好等于差距）："
               + ("拉低最多的是" + _sp("、".join(f"{names[s]} {v * 100:+.1f} pp" for v, s in neg)) if neg else "")
               + ("；抬高最多的是" + _sp("、".join(f"{names[s]} {v * 100:+.1f} pp" for v, s in pos)) if pos else "") + "。")
    if calibrated:
        views = cal["views"]
        sk = (skill or {}).get("vars") or {}

        def skill_txt(k: str) -> str:
            u, n = sk[k]["all"], (sk[k].get("naive") or {}).get("all") or {}
            txt = f"对无条件抽样 CRPS 技能 {u['skill']:+.1%}，t = {u['t']:.1f}"
            if n.get("skill") is not None and n.get("t") is not None:
                txt += f"；对历史均值基准 {n['skill']:+.1%}，t = {n['t']:.1f}"
            return txt
        why = "、".join(f"{v['name']}（{skill_txt(v['var'])}）" for v in views)
        others = "、".join(MACRO_CN[k] for k in MACRO if k not in cal["skilled"])
        ce = eq["calibrated"]
        gap0, gap1 = abs(mx["p50"] - an["p50"]), abs(ce["p50"] - an["p50"])
        shifts = cal.get("shifts") or []
        down = "、".join(f"{s['name']}（{s['shift'] * 100:+.1f} pp）" for s in sorted(shifts, key=lambda s: s["shift"])[:3]
                        if s["shift"] < -0.001)
        up = "、".join(f"{s['name']}（{s['shift'] * 100:+.1f} pp）" for s in sorted(shifts, key=lambda s: -s["shift"])[:3]
                      if s["shift"] > 0.001)
        moved = (f"概率主要从{_sp(down)}移到{_sp(up)}" if down and up else "")
        views_txt = "、".join(f"{v['name']}变化 {_fmt_macro_mean(v['var'], v['prior_mean'])} → "
                              f"{_fmt_macro_mean(v['var'], v['posterior_mean'])}" for v in views)
        verdict = ("差距缩小了。" if gap1 < gap0 - 0.002 else
                   ("差距基本不变。" if gap1 <= gap0 + 0.002 else "差距反而略微扩大：这个变量拉向的正是第一年最差的那几个情景。"))
        out.append(f"校准只能用相似起点法有预测能力的宏观变量：{why}；{others}没有。"
                   f"只对齐{_sp(views_txt)}（KL {cal['kl']:.3f}，有效情景数 {cal['ens_prior']:.1f} → {cal['ens']:.1f}）"
                   + (f"，{moved}" if moved else "")
                   + f"；校准后美股中位数 {ce['p50']:+.1%}、P(亏损) {ce['prob_loss']:.0%}，" + verdict)
    elif cal.get("reason"):
        out.append(f"未校准：{cal['reason']}（逐个变量的成绩见下方「校准」）。")
    sens = {tuple(x["vars"]): x for x in rec["sensitivity"]}
    one = sens.get(("unrate",))
    full = sens.get(tuple(MACRO))
    if one and full and one["us_equity"]["p50"] is not None:
        u_ok = "unrate" in ((skill or {}).get("skilled") or [])
        out.append(f"作为对照（不作为可选的概率）：只把失业率对齐到相似起点法，美股中位数就到 {one['us_equity']['p50']:+.1%}"
                   f"（KL {one['kl']:.2f}）；5 个宏观变量全部对齐到 {full['us_equity']['p50']:+.1%}（KL {full['kl']:.2f}）。"
                   "差距主要来自情景库对第一年宏观（尤其失业率）的设定；"
                   + ("相似起点法对失业率有预测能力，校准已经包含它。" if u_ok else
                      "相似起点法对失业率的预测没有通过检验，一年期打分因此不足以支持据此下调衰退类情景的概率。"))
    if not rec["paths_reproduced"]:
        out.append("注意：重跑情景得到的路径与 scenario_paths.npz 不完全一致（情景结果与当前引擎不是同一次构建），"
                   "请运行 python -m scenariolab.build --only scenarios forecast。")
    return out


def live_forecast(ctx: EngineContext, n_paths: int = N_PATHS, seed: int = SEED, analog_ens: float = ANALOG_ENS,
                  valuation: dict | None = None, skill: dict | None = None, results_dir=None,
                  ledger: dict | None = None) -> dict:
    """The forecast from today: uncond and analog, the scenario library's first year, its reconciliation with
    the analog forecast and the calibrated scenario probabilities (`skill` = macro_skill of the track record).
    `ledger`, when a dict, receives what a ledger entry needs (quantiles of every method)."""
    db, panel = ctx.db, ctx.panel
    state = state_at(panel, ctx.anchor.date)
    w, info_sel = analog_weights(db, state, ens=analog_ens)
    _, info = analog_weights(db, state, ens=DISPLAY_ENS)
    vset, cma, vm = valuation_settings(valuation), ctx.cma, None
    if vset:
        cma, vm = valuation_setup(db, ctx.anchor, vset)
    runs = {"uncond": simulate_year(db, ctx.anchor, cma, start_prior(db.prior(), HORIZON, db.rare), n_paths, seed,
                                    valuation=vm),
            "analog": simulate_year(db, ctx.anchor, cma, w, n_paths, seed + 1, valuation=vm)}
    year_one = year_one_paths(results_dir)
    origin = ctx.anchor.date.strftime("%Y-%m")
    stale = None
    if year_one is not None and year_one.get("start") not in (None, origin):
        # paths of another start month (e.g. the scenarios step failed after a data update): their first year is
        # not the next 12 months, so they are neither shown as today's, reconciled, calibrated nor recorded
        stale = (f"情景路径（scenario_paths.npz）从 {year_one['start']} 出发，不是这次的起点 {origin}："
                 "情景结果与这次的数据不是同一次构建，请运行 python -m scenariolab.build --only scenarios forecast")
        print(f"[forecast] scenario mixture skipped: paths start {year_one['start']}, origin {origin}", file=sys.stderr)
        year_one = None
    mix = scenario_mixture(year_one=year_one) if year_one else None
    assets = {}
    for j, a in enumerate(ASSETS):
        entry = {}
        for key, run in runs.items():
            b = _bands(run["growth"][:, j])
            b["real_p50"] = float(np.median((1 + run["growth"][:, j]) / (1 + run["inflation"]) - 1))
            entry[key] = b
        if mix and a in mix["assets"]:
            entry["scenarios"] = mix["assets"][a]
        assets[a] = entry
    # what happened after the closest historical starting points
    analogs = []
    for item in info["top"][:6]:
        t = pd.Timestamp(item["date"])
        if t + pd.DateOffset(months=HORIZON) <= panel.index.max():
            real = realized_year(panel, t)
            analogs.append({**item, "next12": {a: real["growth"][a] for a in ("us_equity", "nasdaq", "treasury_10y",
                                                                               "gold", "energy", "real_estate")},
                            "macro": real["macro"]})
    macro = {k: {key: _macro_bands(run["macro"][k]) for key, run in runs.items()} for k in MACRO}
    out = {"origin": origin, "state": dict(zip(KEY_STATES, map(float, state))),
           "z_today": info["z_today"], "analog_ens": info_sel["ens"], "bandwidth": info_sel["bandwidth"],
           "display_ens": info["ens"],
           "analogs": analogs, "assets": assets, "macro": macro,
           "scenario_medians": mix["per_scenario"] if mix else None,
           "scenario_probs": mix["probs"] if mix else None,
           "scenario_probs_calibrated": None, "calibration": None, "reconciliation": None}
    if stale:
        out["calibration"] = {"available": False, "reason": stale, "text": f"未校准：{stale}。"}
    cal_probs = None
    if year_one is not None:
        try:
            y1_macro = scenario_year_one_macro(ctx, year_one, results_dir)
            rec, cal = reconcile(assets, runs, year_one, y1_macro, skill, results_dir)
        except Exception as err:  # noqa: BLE001 - an add-on: the forecast itself must still be written
            print(f"[forecast] reconciliation skipped: {err!r}", file=sys.stderr)
            rec, cal = None, {"available": False, "reason": f"对账失败：{err}", "text": f"未校准：对账失败（{err}）。"}
        out["reconciliation"], out["calibration"] = rec, cal
        if cal["available"]:
            cal_probs = cal["probs"]
            out["scenario_probs_calibrated"] = cal_probs
            for a, b in mixture_bands(year_one, cal_probs).items():
                if a in assets:
                    assets[a]["scenarios_calibrated"] = b
    if ledger is not None:
        ledger.update(_ledger_quantiles(panel, ctx.anchor.date, runs, year_one, cal_probs))
    return out


# ---------------------------------------------------------------- the live forecast ledger (data/ledger)

def _quantiles(samples: np.ndarray, weights=None) -> list[float]:
    return [float(v) for v in weighted_percentile(np.asarray(samples, float), LEDGER_TAUS * 100.0, weights)]


def _ledger_quantiles(panel: pd.DataFrame, origin: pd.Timestamp, runs: dict, year_one: dict | None,
                      cal_probs: dict | None) -> dict:
    """Per method and asset: the bands and LEDGER_TAUS quantiles a ledger entry stores (naive: mu, sd)."""
    methods = {}
    for key, run in runs.items():
        methods[key] = {a: {**_bands(run["growth"][:, j]), "q": _quantiles(run["growth"][:, j])}
                        for j, a in enumerate(ASSETS)}
    if year_one is not None:
        N = year_one["n_paths"]
        for key, probs in (("scenarios", None), ("scenarios_calibrated", cal_probs)):
            if key == "scenarios_calibrated" and probs is None:
                continue
            w = np.repeat(_scenario_vector(year_one, probs) / N, N)
            methods[key] = {a: {**_bands(year_one["growth"][:, :, j].ravel(), w),
                                "q": _quantiles(year_one["growth"][:, :, j].ravel(), w)}
                            for j, a in enumerate(year_one["assets"]) if a in ASSETS}
    naive = naive_forecast(panel, origin)
    methods["naive"] = {a: {"mu": mu, "sd": sd} for a, (mu, sd) in naive.items() if np.isfinite(mu) and sd > 0}
    macro = {key: {k: {**_macro_bands(run["macro"][k]), "q": _quantiles(run["macro"][k])} for k in MACRO}
             for key, run in runs.items()}
    return {"methods": methods, "macro": macro}


def ledger_entry(live: dict, quantiles: dict, ctx: EngineContext, n_paths: int = N_PATHS, seed: int = SEED) -> dict:
    """What a build records about its live forecast: per method and asset the bands and quantiles."""
    from .. import __version__
    origin = pd.Timestamp(f"{live['origin']}-01")
    return {
        "schema": LEDGER_SCHEMA, "origin": live["origin"],
        "matures": (origin + pd.DateOffset(months=HORIZON)).strftime("%Y-%m"), "horizon_months": HORIZON,
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "version": __version__,
        "model": ctx.spec.describe(), "analog_ens": live["analog_ens"], "n_paths": n_paths, "seed": seed,
        "taus": "q[i] = 第 (i + 0.5)% 分位，i = 0..99", "units": "12 个月总回报（小数）；宏观为 12 个月变化（pp，油价为对数变化）",
        "methods": {m: {"name": METHOD_CN.get(m, m), "assets": v} for m, v in quantiles["methods"].items()},
        "macro": quantiles["macro"], "state": live["state"],
        "scenario_probs": live.get("scenario_probs"), "scenario_probs_calibrated": live.get("scenario_probs_calibrated"),
    }


def _months_between(a: str, b: str) -> int:
    """Whole months from "YYYY-MM" a to b (b later = positive)."""
    ya, ma = (int(x) for x in a[:7].split("-"))
    yb, mb = (int(x) for x in b[:7].split("-"))
    return (yb - ya) * 12 + (mb - ma)


def ledger_refusal(entry: dict, existing: list[dict], max_delay: int = LEDGER_MAX_DELAY_MONTHS) -> str | None:
    """Why `entry` must not join the ledger (None = it may), so the ledger stays a real-time record:
    no origin older than one already recorded (a backfill made after later data were known), and no
    forecast recorded more than `max_delay` months after its origin month (a stale panel: the months it
    forecasts were already partly known when it was made)."""
    origin = entry["origin"]
    later = [e["origin"] for e in existing if e.get("origin", "") > origin]
    if later:
        return (f"起点 {origin} 早于账本里已有的起点 {max(later)}：账本只按时间顺序追加，"
                "不补记过去的起点（补记时后来的数据已经知道了）")
    created = str(entry.get("created") or "")
    if re.match(r"^\d{4}-\d{2}", created) and _months_between(origin, created) > max_delay:
        return (f"这次构建的数据只到 {origin}，而记录时间是 {created[:7]}：晚于起点 {max_delay} 个月以上的预测"
                "不算实时预测，没有记账（先更新数据：python -m scenariolab.build --refresh）")
    return None


def append_ledger(entry: dict, ledger_dir=LEDGER_DIR) -> bool:
    """Record `entry` as <ledger_dir>/forecast_<origin>.json unless that origin is already recorded.

    The file is written under a unique temporary name and hard-linked into place, so it appears complete
    or not at all, and an existing entry is never replaced (of two builds racing on one origin, one wins).
    Returns whether the entry was added.
    """
    d = Path(ledger_dir)
    d.mkdir(parents=True, exist_ok=True)
    final = d / f"forecast_{entry['origin']}.json"
    if final.exists():
        return False
    tmp = d / f".forecast_{entry['origin']}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(jsonio.dumps(entry))
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, final)
        except FileExistsError:
            return False
        return True
    finally:
        tmp.unlink(missing_ok=True)


ORIGIN_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def load_ledger(ledger_dir=LEDGER_DIR) -> tuple[list[dict], list[str]]:
    """Every readable ledger entry (sorted by origin, one per origin) and the files that could not be read.

    A copy of a record under another name (e.g. "forecast_2026-09 copy.json") would count one forecast twice:
    per origin only the file named forecast_<origin>.json is used (else the earliest recorded), and the
    others are reported."""
    by_origin, errors = {}, []
    d = Path(ledger_dir)
    for p in sorted(d.glob("forecast_*.json")) if d.exists() else []:
        try:
            e = jsonio.load(p)
        except (OSError, ValueError) as err:
            errors.append(f"{p.name}：{err}")
            continue
        if (not isinstance(e, dict) or not isinstance(e.get("origin"), str) or not ORIGIN_RE.match(e["origin"])
                or not isinstance(e.get("methods"), dict)):
            errors.append(f"{p.name}：不是预测记录")
            continue
        by_origin.setdefault(e["origin"], []).append((p.name != f"forecast_{e['origin']}.json", str(e.get("created") or ""),
                                                      p.name, e))
    entries = []
    for origin, cands in by_origin.items():
        cands.sort(key=lambda c: c[:3])
        entries.append(cands[0][3])
        errors += [f"{c[2]}：与 {cands[0][2]} 是同一个起点 {origin} 的重复记录，未计入" for c in cands[1:]]
    entries.sort(key=lambda e: e["origin"])
    return entries, errors


def realized_growth(panel: pd.DataFrame, origin: pd.Timestamp, months: int = HORIZON) -> dict | None:
    """Realized total return of every asset over the `months` after `origin` (None while not observed)."""
    end = origin + pd.DateOffset(months=months)
    if end > panel.index.max():
        return None
    block = panel.loc[origin + pd.DateOffset(months=1):end]
    if len(block) != months:
        return None
    out = {}
    for a in ASSETS:
        col = f"r_{a}"
        r = block[col] if col in block else None
        out[a] = float(np.prod(1.0 + r)) - 1.0 if r is not None and r.notna().all() else None
    return out


def _score_one(f: dict, y: float, naive: bool = False) -> dict:
    if naive:
        mu, sd = f["mu"], f["sd"]
        return {"covered": bool(mu - 1.645 * sd <= y <= mu + 1.645 * sd), "pit": float(norm.cdf((y - mu) / sd)),
                "crps": crps_normal(mu, sd, y), "err_p50": mu - y}
    q = np.asarray(f["q"], float)
    return {"covered": bool(f["p5"] <= y <= f["p95"]), "pit": float(np.interp(y, q, LEDGER_TAUS, left=0.0, right=1.0)),
            "crps": crps_samples(q, y), "err_p50": f["p50"] - y}


def score_ledger(entries: list[dict], panel: pd.DataFrame) -> dict:
    """Score every entry whose 12 months the panel now covers: per method and asset coverage of the 90%
    band, PIT, CRPS (of the stored quantiles) and the error of the median, as the rolling track record."""
    rows, matured = [], []
    for e in entries:
        t = pd.Timestamp(f"{e['origin']}-01")
        h = int(e.get("horizon_months", HORIZON))
        methods = e.get("methods") or {}
        row = {"origin": e["origin"], "created": e.get("created"), "matures": e.get("matures"), "status": "pending",
               "version": e.get("version"), "model": e.get("model"), "analog_ens": e.get("analog_ens"),
               "methods": {m: {a: {k: v[k] for k in ("p5", "p25", "p50", "p75", "p95", "mean", "prob_loss") if k in v}
                               for a, v in (x.get("assets") or {}).items()}
                           for m, x in methods.items() if m != "naive"},
               "realized": None, "scores": None}
        real = realized_growth(panel, t, h)
        if real is not None:
            sc = {}
            for m, x in methods.items():
                per = {}
                for a, f in (x.get("assets") or {}).items():
                    y = real.get(a)
                    if y is None:
                        continue
                    try:
                        per[a] = _score_one(f, y, naive=m == "naive")
                    except (KeyError, TypeError, ValueError):
                        continue
                sc[m] = per
            row.update({"status": "matured", "realized": real, "scores": sc})
            matured.append(row)
        rows.append(row)
    pending = [r for r in rows if r["status"] == "pending"]
    return {"entries": rows, "matured": len(matured), "pending": len(pending),
            "first_maturity": min((r["matures"] for r in pending if r["matures"]), default=None),
            "scores": _ledger_scores(matured) if matured else None}


def _ledger_scores(rows: list[dict]) -> dict:
    out = {}
    methods = sorted({m for r in rows for m in (r["scores"] or {})}, key=lambda m: list(METHOD_CN).index(m)
                     if m in METHOD_CN else 99)
    for m in methods:
        per = {}
        for a in ASSETS:
            s = [r["scores"][m][a] for r in rows if a in (r["scores"].get(m) or {})]
            if not s:
                continue
            per[a] = {"n": len(s), "coverage90": float(np.mean([x["covered"] for x in s])),
                      "pit_mad": float(np.mean([abs(x["pit"] - 0.5) for x in s])),
                      "crps": float(np.mean([x["crps"] for x in s])),
                      "mae_p50": float(np.mean([abs(x["err_p50"]) for x in s])),
                      "bias_p50": float(np.mean([x["err_p50"] for x in s]))}
        out[m] = {"assets": per, "risk_avg": {k: _avg(per, RISK_ASSETS, k) for k in ("coverage90", "pit_mad", "crps", "mae_p50")},
                  "n_entries": sum(1 for r in rows if r["scores"].get(m))}
    for a_, b_ in (("analog", "uncond"), ("analog", "naive"), ("scenarios", "analog")):
        num = (out.get(a_) or {}).get("risk_avg", {}).get("crps")
        den = (out.get(b_) or {}).get("risk_avg", {}).get("crps")
        out.setdefault("crps_skill_risk", {})[f"{a_}_vs_{b_}"] = (1.0 - num / den) if num is not None and den else None
    return out


def ledger_report(entries: list[dict], errors: list[str], panel: pd.DataFrame, appended: bool | None,
                  ledger_dir=LEDGER_DIR) -> dict:
    sc = score_ledger(entries, panel)
    d = Path(ledger_dir)
    try:
        shown = str(d.relative_to(config.ROOT))
    except ValueError:
        shown = str(d)
    n = len(entries)
    if not n:
        text = "账本里还没有记录：每次构建一年期预测时，会把当时的预测原样存进账本。"
    elif not sc["matured"]:
        first = sc["entries"][0]
        text = (f"账本里有 {n} 条实时预测（起点 {entries[0]['origin']}" + (f" 至 {entries[-1]['origin']}" if n > 1 else "")
                + f"），都还没有到期：最早一条（起点 {first['origin']}）要等数据更新到 {first['matures']} 才能评分。")
    else:
        an = (sc["scores"] or {}).get("analog", {}).get("risk_avg", {})
        un = (sc["scores"] or {}).get("uncond", {}).get("risk_avg", {})
        cov = "—" if an.get("coverage90") is None else f"{an['coverage90']:.0%}"
        text = (f"账本里有 {n} 条实时预测，其中 {sc['matured']} 条已经到期并评分：相似起点法风险资产 90% 区间覆盖率 {cov}，"
                f"CRPS {an.get('crps', float('nan')):.3f}（无条件 {un.get('crps', float('nan')):.3f}）。"
                + (f"下一条在数据更新到 {sc['first_maturity']} 后到期。" if sc["first_maturity"] else ""))
    return {"dir": shown, "n_entries": n, **sc, "appended": appended, "errors": errors, "text": text,
            "methods": {m: METHOD_CN.get(m, m) for m in LEDGER_METHODS}}


# ---------------------------------------------------------------- report

def summary_text(sc_all: dict, sc_ex: dict, rows: list[dict], misses: list[dict]) -> str:
    def pct(x):
        return "—" if x is None else f"{x:.0%}"

    an, un = sc_all["analog"], sc_all["uncond"]
    n_shock = sum(1 for r in rows if r["shocks"])
    causes = {c: sum(1 for m in misses if m["cause"] == c) for c in ("shock", "macro", "pricing")}
    skill = sc_all["crps_skill_risk"]
    ic = an["rank_ic"]
    d = an["direction"]
    txt = [
        f"{len(rows)} 个起点（{rows[0]['date']} 至 {rows[-1]['date']}），每次只用当时的数据预测下一年。",
        f"风险资产（股票、REITs、黄金）的 90% 区间覆盖率：相似起点法 {pct(an['risk_avg']['coverage90'])}，"
        f"无条件 {pct(un['risk_avg']['coverage90'])}（理想值 90%）；剔除 {n_shock} 个外生冲击年份后相似起点法为 "
        f"{pct(sc_ex['analog']['risk_avg']['coverage90'])}。",
        f"按 CRPS（整个分布的准确度）：相似起点法{_versus(skill['analog_vs_uncond'], '无条件')}，"
        f"{_versus(skill['analog_vs_naive'], '只用历史均值的朴素基准')}。",
        f"超额收益方向命中率 {pct(d['hit_rate'])}（同期上涨年份占比 {pct(d['base_rate_up'])}，"
        "只猜“涨”就能达到这个比例）；跨资产排序相关 IC 平均 "
        + ("—" if ic["mean"] is None else f"{ic['mean']:.2f}")
        + (f"（t = {ic['t']:.1f}）。" if ic.get("t") is not None else "。"),
        f"偏离最大的 {len(misses)} 次中：外生冲击 {causes['shock']} 次，宏观意外（失业率、政策利率、CPI 同比、信用利差或油价的"
        f"实际变化落在预测的 90% 区间之外）{causes['macro']} 次，宏观符合预期但资产偏离（估值或盈利）{causes['pricing']} 次。"
        "后两类才是模型需要改进的地方。",
    ]
    return "".join(txt)


def _better(s: float | None) -> str:
    """A CRPS skill as 好 x% / 差 x% / 差不多 (the phrase that follows a period: 检验期好 3%)."""
    if s is None:
        return "—"
    return f"好 {s:.0%}" if s > 0.005 else (f"差 {-s:.0%}" if s < -0.005 else "差不多")


def _versus(s: float | None, ref: str) -> str:
    """A CRPS skill against `ref`: 比{ref}好 x% / 比{ref}差 x% / 与{ref}差不多 (never 比…差不多)."""
    if s is None:
        return f"与{ref}相比 —"
    return f"比{ref}{_better(s)}" if abs(s) > 0.005 else f"与{ref}差不多"


def select_analog_ens(panel: pd.DataFrame, spec: ModelSpec, dates: list[pd.Timestamp], n_paths: int, seed: int,
                      grid=ENS_GRID, split: str = SPLIT, valuation: dict | None = None) -> tuple[float, list[dict], dict]:
    """Walk-forward choice of the analog bandwidth: best risk-asset CRPS on origins up to `split`.

    Returns (chosen ENS, the forecast rows made with it, the selection report). The
    unconditional forecasts use the same seeds in every trial, so they are identical.
    """
    cut = pd.Timestamp(split)
    trials, table = {}, []
    for ens in grid:
        rows = [forecast_origin(panel, spec, t, n_paths, seed + i, analog_ens=ens, valuation=valuation)
                for i, t in enumerate(dates)]
        val = scores([r for r in rows if pd.Timestamp(r["date"]) <= cut])
        test = scores([r for r in rows if pd.Timestamp(r["date"]) > cut])
        trials[ens] = rows
        table.append({"ens": float(ens),
                      "val_crps_analog": val["analog"]["risk_avg"]["crps"], "val_crps_uncond": val["uncond"]["risk_avg"]["crps"],
                      "val_crps_naive": val["naive"]["risk_avg"]["crps"], "val_coverage": val["analog"]["risk_avg"]["coverage90"],
                      "test_crps_analog": test["analog"]["risk_avg"]["crps"], "test_crps_uncond": test["uncond"]["risk_avg"]["crps"],
                      "test_crps_naive": test["naive"]["risk_avg"]["crps"], "test_coverage": test["analog"]["risk_avg"]["coverage90"],
                      "test_rank_ic": test["analog"]["rank_ic"]["mean"]})
    best = min(table, key=lambda r: r["val_crps_analog"])
    gain_val = 1.0 - best["val_crps_analog"] / best["val_crps_uncond"]
    gain_test = 1.0 - best["test_crps_analog"] / best["test_crps_uncond"]
    text = (f"相似起点的加权集中度在 {split} 及以前的起点上选（CRPS 最低），之后的起点只用来检验："
            f"选中有效起点数 {best['ens']:.0f}，验证期 CRPS {_versus(gain_val, '无条件')}，检验期{_better(gain_test)}。"
            f"加权越集中（有效起点越少），区间越窄、越容易被现实打穿：有效起点 {table[0]['ens']:.0f} 时检验期覆盖率只有 "
            f"{table[0]['test_coverage']:.0%}。结论：一年期里，“今天像哪段历史”只能轻微调整预测，强行押注少数相似年份反而更差。")
    report = {"grid": table, "chosen": best["ens"], "split": split, "rule": "验证期（起点 ≤ split）风险资产平均 CRPS 最低",
              "gain_val": gain_val, "gain_test": gain_test, "text": text}
    return best["ens"], trials[best["ens"]], report


def run(ctx: EngineContext, n_paths: int = N_PATHS, seed: int = SEED, origins: list | None = None,
        valuation: dict | None = None, ledger_dir=LEDGER_DIR, results_dir=None) -> dict:
    """forecast.json. The build's own run (default origins, paths, seed and engine) also records its live
    forecast in the ledger (`ledger_dir`; None = neither record nor score), so studies that call this with
    other settings never add entries; every run scores the entries that have matured."""
    panel = ctx.panel
    dates = [pd.Timestamp(t) for t in (origins or forecast_origins(panel))]
    cut = pd.Timestamp(SPLIT)
    if any(d <= cut for d in dates) and any(d > cut for d in dates):
        ens, rows, selection = select_analog_ens(panel, ctx.spec, dates, n_paths, seed, valuation=valuation)
    else:
        ens, selection = ANALOG_ENS, None
        rows = [forecast_origin(panel, ctx.spec, t, n_paths, seed + i, analog_ens=ens, valuation=valuation)
                for i, t in enumerate(dates)]
    calm = [r for r in rows if not r["shocks"]]
    sc_all, sc_ex = scores(rows), scores(calm)
    misses = attribute_misses(rows)
    skill = macro_skill(rows)
    quant: dict = {}
    live = live_forecast(ctx, n_paths, seed, analog_ens=ens, valuation=valuation, skill=skill,
                         results_dir=results_dir, ledger=quant)
    ledger = None
    if ledger_dir is not None:
        # only the build's own forecast is recorded: default origins, paths, seed, engine and scenario paths
        official = (origins is None and valuation is None and n_paths == N_PATHS and seed == SEED
                    and results_dir is None)
        appended, errors, skipped = None, [], None
        if official:
            try:
                entry = ledger_entry(live, quant, ctx, n_paths, seed)
                skipped = ledger_refusal(entry, load_ledger(ledger_dir)[0])
                appended = False if skipped else append_ledger(entry, ledger_dir)
            except OSError as err:
                errors.append(f"无法写入预测账本：{err}")
        entries, read_errors = load_ledger(ledger_dir)
        ledger = ledger_report(entries, errors + read_errors, panel, appended, ledger_dir)
        ledger["skipped"] = skipped
    return {
        "horizon_months": HORIZON, "n_paths": n_paths, "key_states": list(KEY_STATES), "analog_ens": ens,
        "methods": METHODS, "assets": ASSETS, "asset_names": NAMES, "risk_assets": RISK_ASSETS,
        "shocks": config.EXOGENOUS_SHOCKS, "notes": NOTES, "selection": selection,
        "origins": rows, "scores": {"all": sc_all, "ex_shock": sc_ex}, "n_shock_years": len(rows) - len(calm),
        "misses": misses, "text": summary_text(sc_all, sc_ex, rows, misses), "macro_skill": skill,
        "live": live, "ledger": ledger,
    }


def main() -> None:
    ctx = EngineContext.load(selection=True)
    jsonio.dump(run(ctx), config.RESULTS_DIR / "forecast.json")


if __name__ == "__main__":
    main()
