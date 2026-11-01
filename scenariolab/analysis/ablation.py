"""Ablation: why knowing the realized macro path made the backtest worse, and what helps.

backtest.json compares, per holdout window, the engine fed the realized 24-month
macro means as views (conditional) with the engine without views
(unconditional). In three of four windows the conditional median CAGR of the
risk assets is further from the actual than the unconditional one. This module
takes that apart and tests engine-level fixes with walk-forward discipline:

  H1 residual contamination   simulate(resid_mode=coupled | decoupled | detrended |
                              trend_decoupled | partial | recentered), engine/residuals.py
  H2 phase granularity        24- vs 12-month phases of realized views
  H3 view set                 the 8 backtest variables vs level-only (s_) views
  H4 CMA                      historical vs valuation_adjusted drift
  H5 missing transmission     how much of the oracle-vs-actual gap the realized
                              valuation change (endogenous, diagnostic only),
                              profit share and capex share account for
  H6 thin view support        MIN_ENS 30 vs 60 / 120 effective months

Every window metric is the mean over `seed` and `extra_seeds` (paired: every
variant sees the same seeds, so the factor draws are common random numbers
and only what the variant changes differs). The baseline is the engine before
the adoption (coupled residuals); the variant that matches config.RESIDUALS
(partial_50 since its adoption) reproduces backtest.json exactly at `seed`
alone (settings.engine_reproduces_backtest).

Selection (selection.rule): only engine variants (H1, H4, H6) can be adopted;
H2 and H3 change how the backtest encodes the realized macro, which the
forward scenarios set in their own specs. A candidate must keep the
validation coverage, must not narrow the unconditional 15-year fan from today
of any risk asset by more than FAN_DROP (the windows are 7-12 years and cannot
see that, but drawdowns and ruin probabilities can), and must improve the validation MAE (tech_bubble, gfc_qe) by more than
MIN_GAIN and two Monte Carlo standard errors. The best such candidate is then
confirmed once on the test windows (covid_ai, post_covid; they overlap): it is
adopted only if it does not hurt the test MAE or the test coverage. There is
no second pick after the test.

    python -m scenariolab.analysis.ablation [--paths 400] [--scen-dir DIR] [--no-scenarios]

writes data/results/ablation.json (and nothing else under data/results); each
variant's run of the 13 future scenarios goes to --scen-dir, or by default to a
temporary directory that is removed afterwards (each run writes ~85 MB of paths).
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, jsonio
from ..context import EngineContext
from ..engine import residuals, summary
from ..engine.factor_model import AssetFactorModel, ModelSpec
from ..engine.pooling import BLOCK, MIN_ENS, MIN_EPISODES, pool_phase
from ..engine.simulate import capital_market_assumptions, simulate
from . import backtest as bt

VALIDATION = ("tech_bubble", "gfc_qe")
TEST = ("covid_ai", "post_covid")
RISK = bt.RISK_ASSETS
NAMES = bt.NAMES
WINDOW_NAMES = {w["id"]: w["name"] for w in bt.WINDOWS}
EXTRA_SEEDS = (12, 13, 14, 15)         # besides `seed`; metrics are seed means, the rule uses their spread
COVERAGE_DROP = 0.05                   # coverage may fall at most this far below the baseline
MIN_GAIN = 0.002                       # smallest validation MAE gain (decimal CAGR) worth adopting
FAN_DROP = 0.05                        # no risk asset's unconditional 15-year fan may narrow more (relative)
FAN_PATHS, FAN_SEED = 4000, 3
SCEN_PATHS = 600                       # as the build runs the library
HEADLINE_KEYS = ("us_equity_cagr_p50", "us_equity_real_cagr_p50", "nasdaq_cagr_p50", "gold_cagr_p50",
                 "treasury_10y_cagr_p50", "us_equity_max_dd_p50", "cpi_avg")
H5_SETS = {"f_valuation": ("valuation",), "f_profit_share": ("profit_share",), "f_capex_share": ("capex_share",),
           "f_profit_share+f_capex_share": ("profit_share", "capex_share"),
           "all": ("valuation", "profit_share", "capex_share")}

VIEW_SETS = {
    "full": bt.VIEW_VARS,
    "levels2": ("s_cpi_yoy", "s_ip_yoy"),
    "levels8": ("s_cpi_yoy", "s_ip_yoy", "s_unrate", "s_ff", "s_y10", "s_spread", "s_oil_yoy", "s_usd_yoy"),
}
BASE = {"resid_mode": "coupled", "resid_window": None, "resid_coupling": 1.0, "phase_months": bt.PHASE_MONTHS,
        "views": "full", "cma": "historical", "min_ens": MIN_ENS}
ENGINE_HYPOTHESES = ("H1", "H4", "H6")       # what the forward scenarios would change if adopted


def _v(vid, hyp, name, description, params, adoptable=True):
    return {"id": vid, "hypothesis": hyp, "name": name, "description": description, "params": params,
            "adoptable": adoptable and hyp in ENGINE_HYPOTHESES}


RECENTER = "残差仍跟着抽中的月份（冲击、资产间结构和时间位置都不变），只把每个阶段残差的条件均值（视图抽中的月份 − 先验抽中的月份）"
VARIANTS = [
    _v("baseline", "H1", "基准引擎（残差全耦合，采用前）",
       "抽中的历史月份带着它自己的残差（因子解释不了的部分）；24 个月阶段、8 个宏观视图、历史漂移、最少 30 个有效月份。"
       "这是采用部分耦合之前的引擎；现在的回测页用的是引擎现行的残差设定（与「部分耦合（50% 混合）」一行在主种子上逐位相同）。",
       {}, adoptable=False),
    _v("decoupled", "H1", "残差解耦",
       "因子变化照常按视图抽月份，残差改由独立的、按先验（无视图）权重的 6 个月区块自助法抽取："
       "保留资产间的相关结构和区块内的持续性，去掉“宏观视图 → 残差”的绑定。", {"resid_mode": "decoupled"}),
    _v("detrended_24", "H1", "残差去趋势（24 个月）",
       "残差减去以该月为中心的 24 个月滚动均值（只用训练期月份估计），再加回先验均值：去掉泡沫、崩盘这类跨年的时代漂移，"
       "保留单月冲击；代价是长期波动变小。", {"resid_mode": "detrended", "resid_window": 24}),
    _v("detrended_36", "H1", "残差去趋势（36 个月）", "同上，滚动窗口 36 个月。",
       {"resid_mode": "detrended", "resid_window": 36}),
    _v("detrended_60", "H1", "残差去趋势（60 个月）", "同上，滚动窗口 60 个月。",
       {"resid_mode": "detrended", "resid_window": 60}),
    _v("detrended_120", "H1", "残差去趋势（120 个月）", "同上，滚动窗口 120 个月（窗口越长越接近基准引擎）。",
       {"resid_mode": "detrended", "resid_window": 120}),
    _v("trend_decoupled_36", "H1", "慢变残差解耦（36 个月）",
       "残差拆成快变部分（减去 36 个月滚动均值）和慢变部分（滚动均值）：快变部分仍跟着抽中的月份，慢变部分改由独立的先验区块"
       "抽取。两部分的波动都保留，只去掉时代漂移与宏观视图的绑定。", {"resid_mode": "trend_decoupled", "resid_window": 36}),
    _v("partial_50", "H1", "部分耦合（50% 混合）",
       "每个 6 个月区块以 50% 的概率用抽中月份自己的残差，否则用独立先验抽取的残差（混合分布，单月残差的分布不变）。",
       {"resid_mode": "partial", "resid_coupling": 0.5}),
    _v("recentered", "H1", "残差漂移归零", RECENTER + "全部去掉。无条件模拟与基准引擎逐位相同。",
       {"resid_mode": "recentered", "resid_coupling": 0.0}),
    _v("recentered_25", "H1", "残差漂移保留 25%", RECENTER + "收缩到 25%。",
       {"resid_mode": "recentered", "resid_coupling": 0.25}),
    _v("recentered_50", "H1", "残差漂移保留 50%", RECENTER + "收缩到 50%。",
       {"resid_mode": "recentered", "resid_coupling": 0.5}),
    _v("recentered_75", "H1", "残差漂移保留 75%", RECENTER + "收缩到 75%。",
       {"resid_mode": "recentered", "resid_coupling": 0.75}),
    _v("phase_12", "H2", "12 个月阶段",
       "回测视图按每 12 个月（而不是 24 个月）的实际宏观均值设定，阶段内的宏观路径刻画得更细。", {"phase_months": 12}),
    _v("phase_12_recentered", "H2", "12 个月阶段 + 残差漂移归零",
       "诊断：12 个月阶段在去掉残差漂移后是否有帮助（与“残差漂移归零”比较）。",
       {"phase_months": 12, "resid_mode": "recentered", "resid_coupling": 0.0}),
    _v("views_levels2", "H3", "只用水平视图（2 个）",
       "只保留 s_ 水平视图（CPI 同比、工业产出同比），去掉 6 个 f_ 月度变化视图。", {"views": "levels2"}),
    _v("views_levels8", "H3", "只用水平视图（8 个）",
       "把 f_ 变化视图换成对应的水平视图：CPI 同比、工业产出同比、失业率、联邦基金利率、10 年期收益率、信用利差、"
       "油价同比、美元同比。", {"views": "levels8"}),
    _v("views_levels8_recentered", "H3", "水平视图（8 个）+ 残差漂移归零",
       "诊断：水平视图的改善是否来自更好的宏观条件，还是来自它挑中的历史片段的残差漂移（与“残差漂移归零”比较）。",
       {"views": "levels8", "resid_mode": "recentered", "resid_coupling": 0.0}),
    _v("cma_va", "H4", "估值调整 CMA",
       "条件与无条件扇形图都用估值调整的资本市场假设（去掉历史估值抬升、向 P/D 中位数回归；黄金去掉历史"
       "实际涨幅），即未来情景现在使用的设定。", {"cma": "valuation_adjusted"}),
    _v("min_ens_60", "H6", "最少 60 个有效月份",
       "视图组合至少要由 60 个有效月份（约 10 个区块）支撑，否则向先验收缩（现行 30；“至少 6 段不同的 12 个月历史片段”"
       "这一条不变）。", {"min_ens": 60.0}),
    _v("min_ens_120", "H6", "最少 120 个有效月份", "同上，下限 120 个有效月份。", {"min_ens": 120.0}),
]

HYPOTHESES = {
    "H1": ("残差污染", "抽中月份自带的残差（泡沫、崩盘、重估这类时代漂移）与宏观视图无关，却随视图一起被抽中。"
           "比较残差耦合、解耦、去趋势（24/36/60/120 个月）、慢变解耦、50% 混合和残差漂移收缩（0/25/50/75%）；"
           "每个变体的无条件残差均值与基准引擎相同，不引入新的漂移。"),
    "H2": ("阶段粒度", "24 个月阶段把阶段内的宏观路径平均掉了。比较 12 个月阶段（耦合残差和去掉残差漂移两种）。"),
    "H3": ("视图集合", "f_ 月度变化视图现在能被精确交付，但它们让资产更准吗？比较 8 个变量与只用 s_ 水平视图（2 个、8 个），"
           "并在去掉残差漂移后再比一次。"),
    "H4": ("资本市场假设", "条件扇形图用历史漂移还是估值调整漂移（未来情景现在用的）。"),
    "H5": ("缺失的传导", "2015 年以来 oracle（训练期因子模型 × 实际因子）远低于实际。用训练期数据给因子模型加上候选因子，"
           "再喂入测试期的实际值，看差距能被估值变化（内生，只作诊断）、利润占比、资本开支占比解释多少。"),
    "H6": ("视图支撑太薄", "有的阶段只靠少数几段历史满足视图，少数历史片段的残差主导结果。比较最少 60 / 120 个有效月份。"),
}


# ---------------------------------------------------------------- windows

@dataclass
class Window:
    w: dict
    months: pd.DatetimeIndex
    years: float
    model: AssetFactorModel
    db: object
    anchor: object
    actual: np.ndarray                  # (T+1, A) realized total-return index
    oracle: np.ndarray                  # (T+1, A) engine transmission fed the actual path
    sys_actual: dict                    # annual mean factor-explained price return on the actual path
    cmas: dict
    weights: dict = field(default_factory=dict)   # (phase_months, views, min_ens) -> (plan, cond, ep)


def setup_window(panel: pd.DataFrame, spec: ModelSpec, w: dict) -> Window:
    """Fit on training months only (backtest.fit_window) and fix the actual and oracle paths."""
    train_end = pd.Timestamp(w["train_end"])
    months = panel.loc[w["test"][0]:w["test"][1] or panel.index.max()].index
    model, db, anchor = bt.fit_window(panel, spec, train_end)
    actual = bt.index_path(panel.loc[months, [f"r_{a}" for a in bt.ASSETS]].to_numpy())
    oracle = bt.oracle_paths(panel, model, anchor, months)
    pred = bt.predict(model, panel).loc[months]
    sys_actual = {a: float(pred[a].mean() * 12.0) for a in RISK}
    cmas = {m: capital_market_assumptions(db, anchor, method=m) for m in ("historical", "valuation_adjusted")}
    return Window(w, months, len(months) / 12.0, model, db, anchor, actual, oracle, sys_actual, cmas)


def window_weights(panel: pd.DataFrame, win: Window, phase_months: int, views: str, min_ens: float):
    """(plan, backtest.view_weights output) for one view protocol, cached per window."""
    key = (int(phase_months), views, float(min_ens))
    if key not in win.weights:
        plan = bt.phase_plan(len(win.months), int(phase_months))
        real = bt.realized_views(panel, win.months, plan, view_vars=VIEW_SETS[views])
        win.weights[key] = (plan, bt.view_weights(win.db, real, plan, min_ens=float(min_ens)))
    return win.weights[key]


def sim_kwargs(params: dict) -> dict:
    """simulate() keyword arguments of a variant's residual draw ({} for the default)."""
    p = {**BASE, **params}
    if p["resid_mode"] == "coupled":
        return {}
    return {"resid_mode": p["resid_mode"], "resid_window": p["resid_window"], "resid_coupling": p["resid_coupling"]}


def evaluate(panel: pd.DataFrame, win: Window, params: dict, n_paths: int, seed: int) -> dict:
    """Window metrics of one variant and seed (definitions as backtest.window_summary, risk assets)."""
    p = {**BASE, **params}
    plan, vw = window_weights(panel, win, p["phase_months"], p["views"], p["min_ens"])
    cond_w, ep = vw["cond"], vw["ep"]
    db, cma, kw = win.db, win.cmas[p["cma"]], sim_kwargs(p)
    sims = {"cond": simulate(db, win.anchor, cond_w, plan, n_paths=n_paths, seed=seed, block=BLOCK, cma=cma,
                             overrides=vw["overrides"], **kw),
            "uncond": simulate(db, win.anchor, vw["prior"], plan, n_paths=n_paths, seed=seed,
                               block=BLOCK, cma=cma, overrides=vw["overrides_prior"], **kw)}
    W = {k: summary.wealth_paths(s.total) for k, s in sims.items()}
    drawn = residuals.draw(db, sims["cond"].idx, cond_w, plan, BLOCK, seed, mode=p["resid_mode"],
                           window=p["resid_window"], coupling=p["resid_coupling"])
    years, A, O = win.years, win.actual, win.oracle
    per, parts = {}, {k: [] for k in ("scenario", "engine", "transmission")}
    for a in RISK:
        j, m = bt.ASSETS.index(a), db.model.assets.index(a)
        p50 = {k: float(np.median(v[:, -1, j])) for k, v in W.items()}
        lo, hi = np.percentile(W["cond"][:, :, j], [5, 95], axis=0)
        dec = bt.decompose(A[-1, j], O[-1, j], p50["cond"], p50["uncond"], years)
        for k in parts:
            parts[k].append(abs(dec[f"{k}_annual"]))
        c = sims["cond"]
        price = c.total[:, :, j].astype(float) - c.income[:, :, j].astype(float)
        sys_cond = float((price - drawn[:, :, m]).mean() * 12.0 - cma["adjustment"].get(a, 0.0))
        per[a] = {"actual": float(bt.cagr(A[-1, j], years)), "oracle": float(bt.cagr(O[-1, j], years)),
                  "cond_p50": float(bt.cagr(p50["cond"], years)), "uncond_p50": float(bt.cagr(p50["uncond"], years)),
                  "coverage_cond": bt.coverage(A[:, j], lo, hi), "pit_cond": bt.pit(A[-1, j], W["cond"][:, -1, j]),
                  "resid_drift_cond": float(drawn[:, :, m].mean() * 12.0),
                  "sys_cond": sys_cond, "sys_actual": win.sys_actual[a]}
    mae = lambda key: bt._avg(abs(per[a][key] - per[a]["actual"]) for a in RISK)       # noqa: E731
    return {
        "cagr_mae_cond_risk": mae("cond_p50"), "cagr_mae_uncond_risk": mae("uncond_p50"),
        "coverage_cond_risk": bt._avg(per[a]["coverage_cond"] for a in RISK),
        "pit_tail_share": float(np.mean([(per[a]["pit_cond"] < 0.05) or (per[a]["pit_cond"] > 0.95) for a in RISK])),
        "engine_abs_annual": bt._avg(parts["engine"]), "transmission_abs_annual": bt._avg(parts["transmission"]),
        "scenario_abs_annual": bt._avg(parts["scenario"]),
        "cagr_mae_oracle_risk": mae("oracle"),
        "resid_drift_abs_annual": bt._avg(abs(per[a]["resid_drift_cond"]) for a in RISK),
        "resid_drift_mean_annual": bt._avg(per[a]["resid_drift_cond"] for a in RISK),
        "sys_gap_abs_annual": bt._avg(abs(per[a]["sys_cond"] - per[a]["sys_actual"]) for a in RISK),
        "min_shrink": float(min(r["shrink"] for r in ep)), "ens_min": float(min(r["ens"] for r in ep)),
        "n_phases": len(plan), "assets": per,
        "thin_phase_years": sorted({int(m["date"][:4]) for r in ep if r["ens"] < 2 * MIN_ENS for m in r["top_months"]}),
    }


def _mean_dicts(items: list[dict]) -> dict:
    """Element-wise mean of equally shaped nested dicts of numbers (ints kept from the first)."""
    first = items[0]
    out = {}
    for k, v in first.items():
        if isinstance(v, dict):
            out[k] = _mean_dicts([it[k] for it in items])
        elif isinstance(v, (list, tuple, str)):
            out[k] = v
        elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            out[k] = v
        else:
            out[k] = float(np.mean([it[k] for it in items]))
    return out


# ---------------------------------------------------------------- forward checks

def forward_fan(ctx: EngineContext, params: dict, n_paths: int = FAN_PATHS, seed: int = FAN_SEED) -> dict:
    """Unconditional 15-year distribution from today (prior weights, the scenarios' CMA) under a variant."""
    T = config.HORIZON_MONTHS
    prior = pool_phase(ctx.db.frame, [], ctx.db.prior(), T, BLOCK, rare=ctx.db.rare).blocks
    s = simulate(ctx.db, ctx.anchor, [prior], [T], n_paths=n_paths, seed=seed, cma=ctx.cma,
                 **sim_kwargs(params))
    W = summary.wealth_paths(s.total)
    years = T / 12.0
    out = {}
    for a in RISK:
        j = config.ASSET_IDS.index(a)
        c = W[:, -1, j] ** (1.0 / years) - 1.0
        out[a] = {"p5": float(np.percentile(c, 5)), "p50": float(np.percentile(c, 50)),
                  "p95": float(np.percentile(c, 95)),
                  "max_dd_p50": float(np.median(summary.max_drawdown(W[:, :, j])))}
    return out


def fan_ratio(fan: dict, base: dict) -> dict[str, float]:
    """Per risk asset, the 15-year p5-p95 CAGR width relative to the baseline's (a portfolio can hold any one)."""
    return {a: float((fan[a]["p95"] - fan[a]["p5"]) / (base[a]["p95"] - base[a]["p5"])) for a in RISK}


@contextlib.contextmanager
def scenario_engine(params: dict):
    """Run the scenario runner with a variant's residual draw and MIN_ENS (runner.py itself is untouched)."""
    from ..scenarios import runner
    p = {**BASE, **params}
    saved = (runner.RESIDUALS, runner.pool_phase)
    runner.RESIDUALS = sim_kwargs(p)                 # the runner passes these to simulate()
    if float(p["min_ens"]) != float(MIN_ENS):
        runner.pool_phase = functools.partial(saved[1], min_ens=float(p["min_ens"]))
    try:
        yield
    finally:
        runner.RESIDUALS, runner.pool_phase = saved


def _headlines(index: dict) -> dict:
    return {s["id"]: {**{k: s["headline"].get(k) for k in HEADLINE_KEYS}, "prob": s["prob"], "name": s.get("name", s["id"])}
            for s in index["scenarios"]}


def _prob_weighted(h: dict) -> dict:
    return {k: float(sum(v["prob"] * v[k] for v in h.values() if v[k] is not None)) for k in HEADLINE_KEYS}


def scenario_effects(ctx: EngineContext, variants: list[dict], out_dir: Path, n_paths: int = SCEN_PATHS) -> dict:
    """Headline numbers of the 13 future scenarios under each engine variant (written to out_dir only)."""
    from ..scenarios import library
    out_dir = Path(out_dir).resolve()
    results = config.RESULTS_DIR.resolve()
    if out_dir == results or results in out_dir.parents:
        raise ValueError("scenario ablation runs must not write into data/results")
    runs = [("baseline", {}, ctx)]
    for v in variants:
        if v["id"] != "baseline" and v["hypothesis"] in ("H1", "H6"):
            runs.append((v["id"], v["params"], ctx))
    hist = replace(ctx, cma=capital_market_assumptions(ctx.db, ctx.anchor, method="historical"))
    runs.append(("cma_historical", {}, hist))           # H4: what the backtest evidence would imply for the scenarios
    heads = {}
    for vid, params, c in runs:
        with scenario_engine(params):
            heads[vid] = _headlines(library.run(c, n_paths=n_paths, out_dir=out_dir / vid))
    base = heads["baseline"]
    # built-ins that state an equity de-rating as a bounded asset_drift (the residual of the analog months no
    # longer carries a whole bust once only half of it is coupled)
    busts = [s["id"] for s in library.load_specs()
             if any(float(((ph.get("overrides") or {}).get("asset_drift") or {}).get("us_equity", 0.0)) < 0.0
                    for ph in s.get("phases") or [])]
    out = {"n_paths": n_paths, "keys": list(HEADLINE_KEYS), "baseline": base,
           "units": "cpi_avg 为百分数（3.4 = 3.4%），其余为小数；delta = 变体 − 基准引擎；prob 为情景概率",
           "baseline_prob_weighted": _prob_weighted(base), "variants": {}, "asset_drift_busts": busts}
    for vid, h in heads.items():
        if vid == "baseline":
            continue
        delta = {sid: {k: (h[sid][k] - base[sid][k]) if h[sid][k] is not None and base[sid][k] is not None else None
                       for k in HEADLINE_KEYS} for sid in base}
        pw, pb = _prob_weighted(h), out["baseline_prob_weighted"]
        big = max(base, key=lambda sid: abs(delta[sid]["us_equity_cagr_p50"]))
        out["variants"][vid] = {"headline": h, "delta": delta, "prob_weighted": pw,
                                "prob_weighted_delta": {k: pw[k] - pb[k] for k in HEADLINE_KEYS},
                                "max_abs_us_equity_delta": abs(delta[big]["us_equity_cagr_p50"]), "max_scenario": big}
    # the run of the engine's own variant (config.RESIDUALS; the coupled baseline is the engine before the
    # adoption) must reproduce the built results
    eng = engine_variant_id(variants) if variants else "baseline"
    out["engine_variant"] = eng
    try:
        built = _headlines(jsonio.load(config.RESULTS_DIR / "scenarios.json"))
        ref = heads.get(eng, base)
        out["engine_matches_results"] = all(
            abs(built[s]["us_equity_cagr_p50"] - ref[s]["us_equity_cagr_p50"]) < 1e-5 for s in ref)
    except (OSError, KeyError, ValueError):
        out["engine_matches_results"] = None
    return out


# ---------------------------------------------------------------- H5 transmission gap

def transmission_gap(panel: pd.DataFrame, spec: ModelSpec, win: Window) -> dict:
    """Per risk asset: actual vs oracle CAGR, the annual log gap (positive = the factors miss upside), and how
    much of it a factor model with candidate factors (fitted on the training months) closes when fed the
    test period's actual values: (log oracle_with - log oracle) / years."""
    train_end = pd.Timestamp(win.w["train_end"])
    oracles = {}
    for key, extra in H5_SETS.items():
        s = ModelSpec.from_dict({**spec.to_dict(), "factors": list(spec.factors) + [f for f in extra if f not in spec.factors],
                                 "name": key})
        model = AssetFactorModel(s).fit(panel, config.MODEL_START, train_end)
        oracles[key] = bt.oracle_paths(panel, model, win.anchor, win.months)
    out = {}
    for a in RISK:
        j = bt.ASSETS.index(a)
        gap = float(np.log(win.actual[-1, j] / win.oracle[-1, j]) / win.years)
        expl = {k: float(np.log(o[-1, j] / win.oracle[-1, j]) / win.years) for k, o in oracles.items()}
        expl["unexplained"] = gap - expl["all"]
        out[a] = {"actual": float(bt.cagr(win.actual[-1, j], win.years)),
                  "oracle": float(bt.cagr(win.oracle[-1, j], win.years)), "gap_annual": gap, "explained_by": expl}
    return out


def gap_context(panel: pd.DataFrame, win: Window) -> dict:
    """Realized market re-rating and distribution shifts over a window (for the text)."""
    P, train = panel.loc[win.months], panel.loc[config.MODEL_START:win.w["train_end"]]
    first, last = win.months[0] - pd.DateOffset(months=1), win.months[-1]

    def change(col):
        a, b = panel[col].get(first, np.nan), panel[col].get(last, np.nan)
        return float(b - a) if np.isfinite(a) and np.isfinite(b) else None
    return {"valuation_annual": float(P["f_valuation"].mean() * 12.0),
            "valuation_annual_train": float(train["f_valuation"].mean() * 12.0),
            "pd_start": float(100.0 / panel["s_div_yield"].get(first)), "pd_end": float(100.0 / panel["s_div_yield"].get(last)),
            "profit_share_change_pp": change("s_profit_share"), "capex_share_change_pp": change("s_capex_share"),
            "labor_share_change_log": (float(np.log(panel["lvl_labor_share"].get(last) / panel["lvl_labor_share"].get(first)))
                                       if np.isfinite(panel["lvl_labor_share"].get(first, np.nan)) else None)}


# ---------------------------------------------------------------- run

def _window_entry(per_seed: list[dict]) -> dict:
    m = _mean_dicts(per_seed)
    keep = ("cagr_mae_cond_risk", "cagr_mae_uncond_risk", "coverage_cond_risk", "pit_tail_share", "engine_abs_annual",
            "transmission_abs_annual", "scenario_abs_annual", "cagr_mae_oracle_risk", "resid_drift_abs_annual",
            "resid_drift_mean_annual", "sys_gap_abs_annual", "min_shrink", "ens_min", "n_phases", "thin_phase_years")
    out = {k: m[k] for k in keep}
    out["cagr_mae_cond_risk_main_seed"] = per_seed[0]["cagr_mae_cond_risk"]
    out["assets"] = {a: {k: m["assets"][a][k] for k in ("actual", "oracle", "cond_p50", "uncond_p50",
                                                        "resid_drift_cond", "sys_cond", "sys_actual")} for a in RISK}
    return out


def _split(per_seed: dict, key: str, wins: tuple) -> np.ndarray:
    """Per seed, the mean of `key` over the windows `wins`."""
    return np.array([np.mean([s[w][key] for w in wins]) for s in per_seed])


def run(ctx: EngineContext, n_paths: int = bt.N_PATHS, seed: int = bt.SEED, extra_seeds=EXTRA_SEEDS,
        scenarios: bool = True, scen_dir=None, scen_paths: int = SCEN_PATHS, variants: list[dict] | None = None,
        windows: list[dict] | None = None) -> dict:
    panel, spec = ctx.panel, ctx.spec
    seeds = [int(seed), *[int(s) for s in extra_seeds if int(s) != int(seed)]]
    wdefs = windows or bt.WINDOWS
    wins = {w["id"]: setup_window(panel, spec, w) for w in wdefs}
    val = tuple(w for w in VALIDATION if w in wins)
    tst = tuple(w for w in TEST if w in wins)
    vlist = variants or VARIANTS
    raw = {}                                          # variant -> [ {window: metrics} per seed ]
    for v in vlist:
        raw[v["id"]] = [{wid: evaluate(panel, win, v["params"], n_paths, s) for wid, win in wins.items()} for s in seeds]

    base_fan = forward_fan(ctx, {})
    out_variants = []
    for v in vlist:
        per_seed = raw[v["id"]]
        entry = {"id": v["id"], "name": v["name"], "description": v["description"], "hypothesis": v["hypothesis"],
                 "params": {**BASE, **v["params"]}, "adoptable": v["adoptable"],
                 "windows": {wid: _window_entry([s[wid] for s in per_seed]) for wid in wins}}
        for tag, ws in (("validation", val), ("test", tst)):
            mae = _split(per_seed, "cagr_mae_cond_risk", ws)
            diff = mae - _split(raw["baseline"], "cagr_mae_cond_risk", ws)
            se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
            entry[f"{tag}_mae"] = float(mae.mean())
            entry[f"{tag}_coverage"] = float(_split(per_seed, "coverage_cond_risk", ws).mean())
            entry[f"{tag}_tail_share"] = float(_split(per_seed, "pit_tail_share", ws).mean())
            entry[f"{tag}_gain"] = float(-diff.mean())          # > 0 = better than the baseline
            entry[f"{tag}_gain_se"] = se
        if v["hypothesis"] in ENGINE_HYPOTHESES:
            fan = base_fan if not sim_kwargs(v["params"]) else forward_fan(ctx, v["params"])
            ratio = fan_ratio(fan, base_fan)
            worst = min(ratio, key=ratio.get)
            entry["forward_fan"] = {"us_equity": fan["us_equity"], "nasdaq": fan["nasdaq"], "width_ratio": ratio,
                                    "width_ratio_min": ratio[worst], "worst_asset": worst,
                                    "p5_change": {a: fan[a]["p5"] - base_fan[a]["p5"] for a in RISK}}
        out_variants.append(entry)

    by_id = {e["id"]: e for e in out_variants}
    selection = select(out_variants)
    scen = None
    if scenarios:
        if scen_dir:                                  # keep every variant's scenario files for inspection
            scen = scenario_effects(ctx, vlist, Path(scen_dir), n_paths=scen_paths)
        else:                                         # each run writes ~85 MB of paths: discard them afterwards
            with tempfile.TemporaryDirectory(prefix="msl_ablation_scen_") as tmp:
                scen = scenario_effects(ctx, vlist, Path(tmp), n_paths=scen_paths)
        for e in out_variants:
            s = scen["variants"].get(e["id"])
            if s:
                e["scenarios"] = {"prob_weighted_delta": s["prob_weighted_delta"],
                                  "max_abs_us_equity_delta": s["max_abs_us_equity_delta"], "max_scenario": s["max_scenario"]}

    tg = {"windows": {wid: transmission_gap(panel, spec, win) for wid, win in wins.items()},
          "context": {wid: gap_context(panel, win) for wid, win in wins.items()}}
    tg["text"] = _gap_text(tg)
    hyps = hypotheses(by_id, tg, scen, val, tst)
    base = by_id["baseline"]
    engine_id = engine_variant_id(out_variants)
    reproduces = _matches_backtest(raw[engine_id][0]) if engine_id in raw else None
    selection["engine_variant"] = engine_id
    if engine_id != "baseline":
        name = by_id[engine_id]["name"] if engine_id in by_id else engine_id
        selection["reason"] += (f"引擎现在的默认设置就是「{name}」（{engine_id}），回测、情景和一年期预测都用它。"
                                if selection.get("adopt") and selection.get("chosen") == engine_id else
                                f"注意：引擎现在的默认设置是「{name}」（{engine_id}），与本次的选择不一致，需要复核。")
    return {
        "question": "为什么在回测里“事先知道真实宏观路径”反而让资产预测更差？问题出在引擎（抽样与重加权）还是传导"
                    "（宏观因子本身解释不了）？哪些引擎改动在走步检验下真的有帮助？",
        "text": _summary_text(by_id, selection, tg, scen, val, tst, hyps),
        "windows": [{"id": w["id"], "name": w["name"], "test": [wins[w["id"]].months[0].strftime("%Y-%m"),
                                                                 wins[w["id"]].months[-1].strftime("%Y-%m")],
                     "train_end": w["train_end"], "role": "validation" if w["id"] in VALIDATION else "test"}
                    for w in wdefs],
        "hypotheses": hyps, "variants": out_variants, "selection": selection, "transmission_gap": tg,
        "scenario_effects": scen,
        "settings": {"n_paths": n_paths, "seeds": seeds, "coverage_drop": COVERAGE_DROP, "min_gain": MIN_GAIN,
                     "fan_drop": FAN_DROP, "fan_paths": FAN_PATHS, "engine_variant": engine_id,
                     "engine_reproduces_backtest": reproduces,
                     "baseline_validation_mae": base["validation_mae"], "baseline_test_mae": base["test_mae"]},
        "units": {"cagr": "年化收益率（小数，0.05 = 5%）", "mae": "风险资产（股票、REITs、黄金）条件中位数年化收益与实际之差的"
                  "绝对值平均", "coverage": "实际累计指数落在条件模拟 5%–95% 区间内的月份占比",
                  "pit_tail_share": "实际终值落在条件模拟 5% 以下或 95% 以上的风险资产占比（校准良好时约 0.1）",
                  "decomposition": "对数终值误差的年化绝对值平均（风险资产）：scenario = 知道宏观让中位数移动多少，engine = "
                  "条件中位数 − oracle，transmission = oracle − 实际",
                  "resid_drift": "条件模拟抽中月份残差的年化均值（因子解释不了、随抽中月份带进来的漂移）",
                  "seeds": "每个数字是多个随机种子的平均；*_gain 为相对基准引擎的改善（正 = 更好），*_gain_se 为种子间的标准误",
                  "gap_annual": "传导缺口 = (ln 实际终值 − ln oracle 终值)/年数，正数 = 因子模型漏掉的上涨",
                  "explained_by": "训练期加入候选因子后，用测试期实际值喂入，oracle 每年上移多少（与 gap_annual 同单位）"},
    }


def engine_variant_id(variants: list[dict]) -> str:
    """The variant whose residual settings are the engine's (config.RESIDUALS); backtest.json was built with it."""
    want = {**BASE, **config.RESIDUALS}
    for v in variants:
        p = {**BASE, **v.get("params", {})}
        if all(p.get(k) == want.get(k) for k in ("resid_mode", "resid_coupling", "resid_window", "phase_months",
                                                    "views", "cma", "min_ens")):
            return v["id"]
    return "baseline"


def _matches_backtest(main_seed: dict) -> bool | None:
    try:
        ref = {r["id"]: r for r in jsonio.load(config.RESULTS_DIR / "backtest.json")["summary"]["windows"]}
    except (OSError, KeyError, ValueError):
        return None
    keys = ("cagr_mae_cond_risk", "cagr_mae_uncond_risk", "coverage_cond_risk", "engine_abs_annual",
            "transmission_abs_annual", "scenario_abs_annual")
    return all(abs(main_seed[w][k] - ref[w][k]) < 2e-6 for w in main_seed if w in ref for k in keys)


# ---------------------------------------------------------------- selection

def select(variants: list[dict]) -> dict:
    by_id = {v["id"]: v for v in variants}
    base = by_id["baseline"]
    floor_val = base["validation_coverage"] - COVERAGE_DROP
    floor_test = base["test_coverage"] - COVERAGE_DROP
    checks, eligible = {}, []
    for v in variants:
        if not v["adoptable"]:
            continue
        need = max(MIN_GAIN, 2.0 * v["validation_gain_se"])
        c = {"coverage": v["validation_coverage"] >= floor_val,
             "forward_fan": v.get("forward_fan", {}).get("width_ratio_min", 1.0) >= 1.0 - FAN_DROP,
             "gain": v["validation_gain"] > need}
        checks[v["id"]] = c
        v["eligible"] = all(c.values())
        if v["eligible"]:
            eligible.append(v)
    rule = (f"只有引擎层面的变体（H1 残差、H4 资本市场假设、H6 最少有效月份）可以采用；H2/H3 改变的是回测怎样表达“已知的宏观路径”，"
            f"未来情景的阶段和视图由各自的情景规格决定，只给结论不采用。候选必须同时满足：① 验证窗口（科技繁荣与泡沫破裂、"
            f"金融危机与QE）的条件覆盖率不低于基准引擎 − {COVERAGE_DROP:.2f}；② 从今天出发、无视图的 15 年分布里，每一个风险"
            f"资产年化收益的 5%–95% 宽度都收窄不超过 {FAN_DROP:.0%}——回测窗口只有 7–12 年，看不到长期离散度，而组合可能"
            f"集中在任何一个资产上，回撤和破产概率取决于它；"
            f"③ 验证窗口条件风险资产年化 MAE 的改善大于 {MIN_GAIN:.1%}，且大于种子间标准误的 2 倍。满足的候选中取验证 MAE "
            f"最低者；然后只在测试窗口（疫情、再通胀与AI，疫情以来；两者重叠）确认一次：测试 MAE 不比基准引擎差（容差为 2 倍"
            f"标准误）、测试覆盖率不低于基准引擎 − {COVERAGE_DROP:.2f}，才采用；测试不通过就不采用，不再换第二名。")
    if not eligible:
        return {"rule": rule, "chosen": "baseline", "adopt": False, "checks": checks, "eligible": [],
                "reason": "没有引擎变体同时满足覆盖率、长期分布和稳健改善三个条件，保留基准引擎。"}
    pick = min(eligible, key=lambda v: v["validation_mae"])
    test_ok = pick["test_mae"] <= base["test_mae"] + 2.0 * pick["test_gain_se"]
    cov_ok = pick["test_coverage"] >= floor_test
    adopt = bool(test_ok and cov_ok)
    others = sorted((v for v in eligible if v is not pick), key=lambda v: v["validation_mae"])
    txt = (f"验证窗口选出「{pick['name']}」（{pick['id']}）：验证 MAE {pick['validation_mae']:.2%}，基准引擎 "
           f"{base['validation_mae']:.2%}（改善 {pick['validation_gain']:.2%} ± {pick['validation_gain_se']:.2%}）；"
           f"验证覆盖率 {pick['validation_coverage']:.0%}（基准 {base['validation_coverage']:.0%}）。测试窗口：MAE "
           f"{pick['test_mae']:.2%}，基准 {base['test_mae']:.2%}（改善 {pick['test_gain']:.2%} ± {pick['test_gain_se']:.2%}），"
           f"覆盖率 {pick['test_coverage']:.0%}（基准 {base['test_coverage']:.0%}）。")
    txt += ("测试确认通过，建议采用。" if adopt else
            "测试没有确认（" + ("MAE 变差" if not test_ok else "覆盖率下降过多") + "），不采用。")
    if others:
        txt += "其他满足条件的候选（按验证 MAE）：" + "、".join(
            f"「{v['name']}」（验证 {v['validation_mae']:.2%}，测试 {v['test_mae']:.2%}）" for v in others[:4]) + "。"
    blocked = [v for v in variants if v["adoptable"] and checks.get(v["id"]) and not checks[v["id"]]["forward_fan"]
               and checks[v["id"]]["gain"]]
    if blocked:
        txt += "改善足够但被长期分布条件排除的：" + "、".join(
            f"「{v['name']}」（验证 {v['validation_mae']:.2%}、测试 {v['test_mae']:.2%}，但{NAMES[v['forward_fan']['worst_asset']]}"
            f"的 15 年分布宽度只有基准的 {v['forward_fan']['width_ratio_min']:.0%}，5% 分位年化收益抬高 "
            f"{v['forward_fan']['p5_change'][v['forward_fan']['worst_asset']]:+.1%}，会让回撤和破产概率显得更乐观）"
            for v in blocked) + "。"
    return {"rule": rule, "chosen": pick["id"], "adopt": adopt, "checks": checks,
            "eligible": [v["id"] for v in eligible], "reason": txt}


# ---------------------------------------------------------------- verdicts and text

def _mean_over(entry: dict, key: str, wins: tuple) -> float:
    return bt._avg(entry["windows"][w][key] for w in wins if w in entry["windows"])


def _verdict(val_gain: float, val_se: float, test_gain: float, test_se: float) -> str:
    """supported = robustly better on validation and not worse on test; rejected = no robust gain on either;
    inconclusive = better on one set of windows and not on the other."""
    val_better, test_better = val_gain > max(MIN_GAIN, 2 * val_se), test_gain > max(MIN_GAIN, 2 * test_se)
    if val_better and test_gain > -2 * test_se:
        return "supported"
    if not val_better and not test_better:
        return "rejected"
    return "inconclusive"


def _gains(e: dict) -> str:
    return (f"验证 {e['validation_mae']:.2%}（{e['validation_gain']:+.2%}）、测试 {e['test_mae']:.2%}"
            f"（{e['test_gain']:+.2%}）")


def drift_slope(by_id: dict, tg: dict, wins: tuple) -> float | None:
    """Slope (through 0) of the realized beyond-factor drift on the coupled draw's residual drift, risk assets.

    1 = the drift the coupled draw brings in is worth keeping in full, 0 = it is noise; it is the
    least-squares shrinkage factor for the conditional residual drift.
    """
    x, y = [], []
    for w in wins:
        if w not in tg["windows"]:
            continue
        for a in RISK:
            x.append(by_id["baseline"]["windows"][w]["assets"][a]["resid_drift_cond"])
            y.append(tg["windows"][w][a]["gap_annual"])
    x, y = np.asarray(x), np.asarray(y)
    return float(x @ y / (x @ x)) if x.size and x @ x > 0 else None


def _windows_worse(e: dict, base: dict) -> int:
    """Number of windows where the variant's conditional MAE is above the baseline's."""
    return sum(e["windows"][w]["cagr_mae_cond_risk"] > base["windows"][w]["cagr_mae_cond_risk"] for w in e["windows"])


def _both_better(e: dict) -> bool:
    return e["validation_gain"] > 0 and e["test_gain"] > 0


def hypotheses(by_id: dict, tg: dict, scen: dict | None, val: tuple, tst: tuple) -> list[dict]:
    base = by_id["baseline"]
    n_win = len(base["windows"])
    out = []

    def add(hid, verdict, result, evidence, numbers, test=None):
        name, default_test = HYPOTHESES[hid]
        out.append({"id": hid, "name": name, "test": test or default_test, "result": result, "verdict": verdict,
                    "evidence": evidence, "numbers": numbers})

    # H1 residual contamination
    g = "gfc_qe" if "gfc_qe" in base["windows"] else val[0]
    gname = WINDOW_NAMES.get(g, g)
    bw = base["windows"][g]
    worst = sorted(RISK, key=lambda a: bw["assets"][a]["resid_drift_cond"])[:2]
    h1 = [e for e in by_id.values() if e["hypothesis"] == "H1" and e["id"] != "baseline"]
    full = by_id.get("recentered") or by_id.get("decoupled")
    half = [by_id[k] for k in ("partial_50", "recentered_50") if k in by_id]
    det = [by_id[k] for k in ("detrended_24", "detrended_36", "detrended_60", "detrended_120") if k in by_id]
    slope_v, slope_t = drift_slope(by_id, tg, val), drift_slope(by_id, tg, tst)
    sel = [e for e in h1 if e.get("eligible")]
    pick = min(sel, key=lambda e: e["validation_mae"]) if sel else None
    if pick:
        v1 = _verdict(pick["validation_gain"], pick["validation_gain_se"], pick["test_gain"], pick["test_gain_se"])
    else:
        v1 = "inconclusive" if any(_both_better(e) for e in h1) else "rejected"
    ev = [f"基准引擎在「{gname}」窗口里，因子部分交付得很准：条件模拟的因子解释收益与实际路径的因子解释收益平均只差 "
          f"{bw['sys_gap_abs_annual']:.1%}/年；但抽中月份的残差平均带来每年 {bw['resid_drift_mean_annual']:+.1%} 的漂移"
          f"（{NAMES[worst[0]]} {bw['assets'][worst[0]]['resid_drift_cond']:+.1%}、{NAMES[worst[1]]} "
          f"{bw['assets'][worst[1]]['resid_drift_cond']:+.1%}），而无条件抽样时这一项接近 0。"]
    if full:
        dg = full["windows"][g]["cagr_mae_cond_risk"] - bw["cagr_mae_cond_risk"]
        ev.append(f"把这项漂移全部去掉（{full['name']}）：{_gains(full)}；「{gname}」窗口的误差变化 {dg:+.2%}"
                  + ("——危机月份的残差确实带有信息（金融股在 2008 年的跌幅远超因子所能解释），全部去掉会丢掉它。" if dg > 0
                     else "。"))
    if half:
        ev.append("保留一半：" + "；".join(f"{e['name']} {_gains(e)}" for e in half)
                  + ("，验证和测试都更好。" if all(_both_better(e) for e in half) else "。"))
    if slope_v is not None:
        ev.append(f"直接检验：验证窗口里，实际的“因子外漂移”（ln 实际 − ln oracle，年化）对耦合抽样带来的残差漂移的回归斜率为 "
                  f"{slope_v:.2f}（1 = 全部是信息，0 = 全部是噪声，它就是最小二乘意义下应保留的比例）；测试窗口为 "
                  f"{slope_t:.2f}" + ("（2015 年后的估值重估让实际缺口整体为正，而抽中的残差漂移为负）。" if slope_t is not None
                                       and slope_t < 0 else "。"))
    if det:
        ev.append("去趋势：" + "；".join(f"{e['params']['resid_window']} 个月 {_gains(e)}" for e in det) +
                  "。去趋势会削掉跨年的泡沫与崩盘，长期分布随之变窄（见 selection 的长期分布条件）。")
    if pick:
        r1 = (f"残差漂移是引擎误差的主要来源，但其中只有一部分是污染。按规则选中的残差变体「{pick['name']}」：验证 MAE "
              f"{base['validation_mae']:.2%} → {pick['validation_mae']:.2%}，测试 {base['test_mae']:.2%} → {pick['test_mae']:.2%}")
        if full and full["validation_gain"] < 0:
            r1 += f"；全部去掉漂移反而让验证变差 {-full['validation_gain']:.2%}"
        r1 += "。"
    else:
        r1 = "残差漂移是引擎误差的重要来源，但没有残差变体同时满足覆盖率、长期分布和稳健改善的条件。"
    add("H1", v1, r1, "".join(ev),
        {"resid_drift_mean_annual": {w: base["windows"][w]["resid_drift_mean_annual"] for w in base["windows"]},
         "resid_drift_abs_annual": {w: base["windows"][w]["resid_drift_abs_annual"] for w in base["windows"]},
         "sys_gap_abs_annual": {w: base["windows"][w]["sys_gap_abs_annual"] for w in base["windows"]},
         "drift_slope_validation": slope_v, "drift_slope_test": slope_t, "chosen": pick["id"] if pick else None})

    # H2 phase granularity
    p12, p12r, rc = by_id.get("phase_12"), by_id.get("phase_12_recentered"), by_id.get("recentered")
    if p12:
        v2 = _verdict(p12["validation_gain"], p12["validation_gain_se"], p12["test_gain"], p12["test_gain_se"])
        ev = (f"12 个月阶段（耦合残差）：{_gains(p12)}。12 个月的视图更极端，抽样更集中在少数片段，残差漂移"
              f"{'更大' if _mean_over(p12, 'resid_drift_abs_annual', tst) > _mean_over(base, 'resid_drift_abs_annual', tst) else '没有更小'}"
              f"（测试窗口平均每年 {_mean_over(p12, 'resid_drift_abs_annual', tst):.1%}，基准 "
              f"{_mean_over(base, 'resid_drift_abs_annual', tst):.1%}）。")
        if p12r and rc:
            better_v, better_t = p12r["validation_mae"] < rc["validation_mae"], p12r["test_mae"] < rc["test_mae"]
            ev += (f"去掉残差漂移后再比：12 个月 {p12r['validation_mae']:.2%}/{p12r['test_mae']:.2%}，24 个月 "
                   f"{rc['validation_mae']:.2%}/{rc['test_mae']:.2%}（验证/测试），"
                   + ("12 个月两边都更好。" if better_v and better_t else "12 个月两边都更差。" if not (better_v or better_t)
                      else f"12 个月在{'验证' if better_v else '测试'}窗口更好、{'测试' if better_v else '验证'}窗口更差，"
                           f"没有稳定的结论。"))
        r2 = {"supported": "12 个月阶段在验证和测试上都更好", "rejected": "更细的阶段没有帮助",
              "inconclusive": "12 个月阶段的效果不稳定"}[v2]
        add("H2", v2, f"{r2}：12 个月阶段验证 {p12['validation_gain']:+.2%}、测试 {p12['test_gain']:+.2%}（相对基准引擎的"
            f"改善，正 = 更好）。", ev,
            {"phase_12": [p12["validation_gain"], p12["test_gain"]],
             "phase_12_recentered_vs_recentered": ([rc["validation_mae"] - p12r["validation_mae"],
                                                    rc["test_mae"] - p12r["test_mae"]] if p12r and rc else None)})

    # H3 view set
    l2, l8, l8r = by_id.get("views_levels2"), by_id.get("views_levels8"), by_id.get("views_levels8_recentered")
    if l2 and l8:
        flip = None
        if l8r and rc:
            dv, dt = rc["validation_mae"] - l8r["validation_mae"], rc["test_mae"] - l8r["test_mae"]
            flip = (dv > 0) != (dt > 0)
        c = "covid_ai" if "covid_ai" in l8["windows"] else tst[0]
        cov_drop = l8["test_coverage"] < base["test_coverage"] - COVERAGE_DROP
        ev = (f"只用 2 个水平视图：{_gains(l2)}；8 个水平视图：{_gains(l8)}，测试覆盖率 {l8['test_coverage']:.0%}"
              f"（基准 {base['test_coverage']:.0%}）、尾部占比 {l8['test_tail_share']:.0%}（基准 {base['test_tail_share']:.0%}）。"
              f"水平视图挑中的历史片段更窄（例如 2015–2021 的零利率只能对上 2009–2014），「{WINDOW_NAMES.get(c, c)}」窗口里"
              f"抽中残差的漂移每年 {l8['windows'][c]['resid_drift_abs_annual']:.1%}（现行 "
              f"{base['windows'][c]['resid_drift_abs_annual']:.1%}）。")
        if l8r and rc:
            ev += (f"去掉残差漂移后，8 个水平视图对 8 个原视图：验证 {l8r['validation_mae']:.2%} 对 {rc['validation_mae']:.2%}、"
                   f"测试 {l8r['test_mae']:.2%} 对 {rc['test_mae']:.2%}" + ("，验证和测试方向相反：水平视图的好处主要来自它"
                   "挑中的片段的残差漂移碰巧有利，而不是更好的宏观条件。" if flip else "。"))
        if l2["test_gain"] > 2 * l2["test_gain_se"] and l2["validation_gain"] < l2["test_gain"]:
            ev += ("只用 2 个水平视图在测试窗口好很多，是因为它不告诉引擎 2022 年以来的加息和收益率上行：因子模型认为这会"
                   "压低股价，而实际股市没有跟随（见 H5），所以“知道得少”反而更准，这不是可以推广的改进。")
        v3 = "inconclusive" if (flip or cov_drop) else \
            _verdict(l8["validation_gain"], l8["validation_gain_se"], l8["test_gain"], l8["test_gain_se"])
        r3 = ("f_ 变化视图被精确交付本身不是问题：水平视图在部分窗口更准，但改善伴随更大的残差漂移，"
              + ("去掉漂移后方向不稳定" if flip else "去掉漂移后仍然成立") + ("，并降低了测试窗口的校准度。" if cov_drop else "。"))
        add("H3", v3, r3, ev,
            {"views_levels2": [l2["validation_gain"], l2["test_gain"]], "views_levels8": [l8["validation_gain"], l8["test_gain"]],
             "views_levels8_recentered_vs_recentered": ([rc["validation_mae"] - l8r["validation_mae"],
                                                         rc["test_mae"] - l8r["test_mae"]] if l8r and rc else None)})

    # H4 CMA
    va = by_id.get("cma_va")
    if va:
        nw = _windows_worse(va, base)
        v4 = _verdict(va["validation_gain"], va["validation_gain_se"], va["test_gain"], va["test_gain_se"])
        pw = scen["variants"].get("cma_historical") if scen else None
        ev = (f"估值调整 CMA：{_gains(va)}，{n_win} 个窗口中 {nw} 个更差。各窗口起点的市价/股息与终点："
              f"{_ctx_line(tg)}；CMA 预期估值向训练期中位数回归，而此后估值大多继续抬升，所以它偏悲观。这只说明 1995 年以来的"
              f"这几个起点（同一个长期降息、估值抬升的大趋势，窗口还互相重叠）没有出现估值回归，不能证明未来不会回归。")
        if pw:
            d = pw["prob_weighted_delta"]
            ev += (f"未来情景现在用的正是估值调整 CMA；若改用历史漂移，概率加权的美股中位数年化收益变化 "
                   f"{d['us_equity_cagr_p50']:+.1%}、纳指 {d['nasdaq_cagr_p50']:+.1%}、黄金 {d['gold_cagr_p50']:+.1%}，"
                   f"远大于任何残差变体，属于假设选择而不是引擎修复。")
        r4 = {"rejected": "估值调整 CMA 在回测中更差", "supported": "估值调整 CMA 在回测中更好",
              "inconclusive": "估值调整 CMA 的效果不稳定"}[v4]
        add("H4", v4, f"{r4}：验证 {va['validation_gain']:+.2%}、测试 {va['test_gain']:+.2%}（相对历史漂移的改善，"
            f"正 = 更好）。", ev,
            {"cma_va": [va["validation_gain"], va["test_gain"]], "windows_worse": nw,
             "scenario_prob_weighted_delta_if_historical": pw["prob_weighted_delta"] if pw else None})

    # H5 missing transmission
    w5 = "covid_ai" if "covid_ai" in tg["windows"] else (tst[0] if tst else None)
    if w5:
        T5 = tg["windows"][w5]
        eq, nd, au = T5["us_equity"], T5["nasdaq"], T5["gold"]

        def share(e, k):
            return e["explained_by"][k] / e["gap_annual"] if abs(e["gap_annual"]) > 1e-9 else float("nan")
        c5 = tg["context"][w5]
        sv, spc = share(eq, "f_valuation"), share(eq, "f_profit_share+f_capex_share")
        ev = (f"「{WINDOW_NAMES.get(w5, w5)}」（市价/股息 {c5['pd_start']:.0f} → {c5['pd_end']:.0f} 倍，估值每年抬升 "
              f"{c5['valuation_annual']:.1%}，训练期平均 {c5['valuation_annual_train']:.1%}；利润占 GDP "
              f"{c5['profit_share_change_pp']:+.1f} 个百分点，资本开支占 GDP {c5['capex_share_change_pp']:+.1f} 个百分点，劳动收入"
              f"占比 {c5['labor_share_change_log']:+.1%}）：美股实际年化 {eq['actual']:.1%}、oracle {eq['oracle']:.1%}，缺口每年 "
              f"{eq['gap_annual']:.1%}，其中估值变化解释 {eq['explained_by']['f_valuation']:.1%}（{sv:.0%}），利润占比 + 资本开支"
              f"占比解释 {eq['explained_by']['f_profit_share+f_capex_share']:.1%}（{spc:.0%}）；纳指缺口 {nd['gap_annual']:.1%}，"
              f"估值解释 {nd['explained_by']['f_valuation']:.1%}，三者合计后剩 {nd['explained_by']['unexplained']:+.1%}"
              f"（AI 利润集中在少数公司，宏观总量看不到）；黄金缺口 {au['gap_annual']:.1%}，三个候选合计只解释 "
              f"{au['explained_by']['all']:+.1%}（央行购金与地缘风险不在面板里）。必需消费、医疗等防御板块的缺口为负："
              f"它们跑输了因子模型，估值因子反而把它们也抬高，所以“全部资产平均”的解释比例没有意义，要看单个资产。")
        ok = sv >= 0.5 and abs(spc) < 0.25
        add("H5", "supported" if ok else "inconclusive",
            ("传导缺口确实是“少了一条渠道”，但少的是估值重估（风险偏好、AI 叙事）：美股缺口的 "
             f"{sv:.0%} 来自估值变化，它本身就是资产价格的一部分（内生），不能当作可以事先给定的宏观因子；利润占比、资本开支"
             f"占比这些外生候选只解释 {spc:.0%}。这部分不是引擎或调参能修的，在情景里应作为显式、有上限的估值假设"
             f"（asset_drift 或 CMA）来写。") if ok else
            (f"估值变化解释美股缺口的 {sv:.0%}，利润与资本开支占比解释 {spc:.0%}，不足以确定缺失的是哪条渠道。"),
            ev + "方法：在训练期给因子模型加入候选因子，再用测试期的实际值喂入，看 oracle 每年上移多少；估值因子是市场 P/D "
            "的变化，几乎就是股价本身，只用来诊断，不能用于预测。",
            {"window": w5, "us_equity_gap": eq["gap_annual"], "us_equity_valuation_share": sv,
             "us_equity_profit_capex_share": spc, "nasdaq_gap": nd["gap_annual"], "gold_gap": au["gap_annual"]})

    # H6 thin support
    e6 = [by_id[k] for k in ("min_ens_60", "min_ens_120") if k in by_id]
    if e6:
        b6 = max(e6, key=lambda e: e["validation_gain"])
        v6 = _verdict(b6["validation_gain"], b6["validation_gain_se"], b6["test_gain"], b6["test_gain_se"])
        ev = "；".join(f"{e['name']} {_gains(e)}" for e in e6) + \
             "。提高支撑下限会把视图更多地收缩回先验：抽样更分散，但丢掉的宏观信息也更多；残差漂移可以直接处理（H1）。"
        r6 = {"supported": "提高最少有效月份有帮助", "rejected": "提高最少有效月份没有帮助",
              "inconclusive": "提高最少有效月份的效果不稳定"}[v6]
        thin = sorted(w["ens_min"] for w in base["windows"].values() if w["ens_min"] < 2 * MIN_ENS)
        test6 = (f"回测里支撑最薄的阶段只有约 {thin[0]:.0f}–{thin[-1]:.0f} 个有效月份（引擎要求至少 {MIN_ENS:.0f} 个有效月份、"
                 f"{MIN_EPISODES:.0f} 段不同的 12 个月历史片段，不够时视图向先验收缩），少数历史片段的残差主导结果。"
                 "比较最少 60 / 120 个有效月份。") if thin else None
        add("H6", v6, f"{r6}：" + "、".join(f"「{e['name']}」验证 {e['validation_gain']:+.2%}、测试 {e['test_gain']:+.2%}"
                                            for e in e6) + "（正 = 更好）。", ev,
            {e["id"]: [e["validation_gain"], e["test_gain"]] for e in e6}, test=test6)
    return out


def _ctx_line(tg: dict) -> str:
    return "、".join(f"{WINDOW_NAMES.get(w, w)} {c['pd_start']:.0f} → {c['pd_end']:.0f} 倍" for w, c in tg["context"].items())


def _gap_text(tg: dict) -> str:
    parts = []
    for w, rows in tg["windows"].items():
        eq = rows["us_equity"]
        parts.append(f"{WINDOW_NAMES.get(w, w)}：美股缺口 {eq['gap_annual']:+.1%}/年，估值解释 {eq['explained_by']['f_valuation']:+.1%}，"
                     f"利润与资本开支占比 {eq['explained_by']['f_profit_share+f_capex_share']:+.1%}")
    return ("传导缺口 = 实际与 oracle（训练期因子模型 × 测试期实际因子）之差，正数表示因子漏掉了上涨。explained_by 是给训练期"
            "因子模型加入候选因子、再喂入测试期实际值后 oracle 每年上移多少；all 是三个一起加，unexplained 是剩下的。"
            "f_valuation（市场 P/D 变化）是内生的，几乎就是股价本身，只用来诊断。" + "；".join(parts) + "。")


def _summary_text(by_id: dict, selection: dict, tg: dict, scen: dict | None, val: tuple, tst: tuple,
                  hyps: list[dict] | None = None) -> str:
    base = by_id["baseline"]
    unc_v, unc_t = _mean_over(base, "cagr_mae_uncond_risk", val), _mean_over(base, "cagr_mae_uncond_risk", tst)
    g = "gfc_qe" if "gfc_qe" in base["windows"] else val[0]
    bw = base["windows"][g]
    txt = [f"采用前的基准引擎（残差全耦合）：知道真实宏观路径后，风险资产年化收益中位数的平均误差在验证窗口为 {base['validation_mae']:.2%}"
           f"（不知道宏观 {unc_v:.2%}），测试窗口为 {base['test_mae']:.2%}（不知道 {unc_t:.2%}）。原因有两个。"
           f"一是引擎：视图把抽样集中到少数历史片段，这些月份自带的残差漂移跟着进来——「{WINDOW_NAMES.get(g, g)}」窗口里"
           f"因子部分与实际路径只差 {bw['sys_gap_abs_annual']:.1%}/年，抽中残差却平均带来 {bw['resid_drift_mean_annual']:+.1%}/年"
           + (f"（支撑最薄的阶段主要抽 {'、'.join(str(y) for y in bw['thin_phase_years'][:5])} 年的月份）"
              if bw.get("thin_phase_years") else "") + "。"]
    w5 = "covid_ai" if "covid_ai" in tg["windows"] else None
    if w5:
        eq = tg["windows"][w5]["us_equity"]
        txt.append(f"二是传导：2015 年以来 oracle 本身就远低于实际（美股 {eq['oracle']:.1%} 对 {eq['actual']:.1%}），缺口每年 "
                   f"{eq['gap_annual']:.1%} 中估值重估占 {eq['explained_by']['f_valuation']:.1%}，利润与资本开支占比只占 "
                   f"{eq['explained_by']['f_profit_share+f_capex_share']:.1%}；知道 2022 年以来的加息让条件中位数向 oracle "
                   f"靠拢，而 oracle 离实际更远，所以“知道”反而更差。")
    pick = by_id.get(selection["chosen"])
    if selection["adopt"] and pick:
        pu_v, pu_t = _mean_over(pick, "cagr_mae_uncond_risk", val), _mean_over(pick, "cagr_mae_uncond_risk", tst)
        still = (pick["validation_mae"] > pu_v) or (pick["test_mae"] > pu_t)
        cov_kept = (pick["validation_coverage"] >= base["validation_coverage"]
                    and pick["test_coverage"] >= base["test_coverage"])
        width = (pick.get("forward_fan") or {}).get("width_ratio_min")
        txt.append(f"走步检验选中并确认「{pick['name']}」：验证 {base['validation_mae']:.2%} → {pick['validation_mae']:.2%}，"
                   f"测试 {base['test_mae']:.2%} → {pick['test_mae']:.2%}，"
                   + ("覆盖率不降" if cov_kept else f"覆盖率下降不超过 {COVERAGE_DROP:.2f}")
                   + (f"，15 年无条件分布的宽度不低于基准的 {width:.1%}。" if width is not None else "。")
                   + (f"它缩小但没有消除“知道反而更差”：采用后不知道宏观的误差为验证 {pu_v:.2%}、测试 {pu_t:.2%}，"
                      f"剩下的主要是传导问题，不是调参能修的。" if still else
                      f"采用后知道宏观的误差在验证和测试窗口都低于不知道宏观（{pu_v:.2%}、{pu_t:.2%}）。"))
        s = (scen or {}).get("variants", {}).get(pick["id"])
        if s:
            heads = scen.get("baseline") or {}
            big = s["max_scenario"]
            big_delta = (s.get("delta") or {}).get(big, {}).get("us_equity_cagr_p50")
            busts = [heads.get(x, {}).get("name", x) for x in scen.get("asset_drift_busts") or []]
            txt.append(f"对 {len(heads)} 个未来情景：概率加权的美股中位数年化收益变化 {s['prob_weighted_delta']['us_equity_cagr_p50']:+.2%}，"
                       f"变化最大的是「{heads.get(big, {}).get('name', big)}」（"
                       + (f"{big_delta:+.1%}" if big_delta is not None else f"{s['max_abs_us_equity_delta']:.1%}") + "）"
                       + (("——类比月份的残差采用后只带进一半，所以内置情景把估值压缩写成显式、有上限的 asset_drift（"
                           + "、".join(busts) + "）。") if pick["hypothesis"] == "H1" and busts else
                          "——这些情景的一部分跌幅原本来自类比月份的残差，采用后应改用显式、有上限的 asset_drift 表达。"
                          if pick["hypothesis"] == "H1" else "。"))
    else:
        txt.append("走步检验没有找到稳健、可采用的引擎改动：" + selection["reason"])
    det, det120 = by_id.get("detrended_60"), by_id.get("detrended_120")
    if det and det.get("forward_fan") and not det.get("eligible", True) and \
            not selection.get("checks", {}).get("detrended_60", {}).get("forward_fan", True):
        txt.append(f"残差去趋势（60 个月）的验证误差最低（{det['validation_mae']:.2%}），但它把"
                   f"{NAMES[det['forward_fan']['worst_asset']]}的 15 年分布收窄到基准的 {det['forward_fan']['width_ratio_min']:.0%}，"
                   f"让回撤和破产概率显得更乐观" + ("，而且窗口换成 120 个月测试误差就变差" if det120 and det120["test_gain"] < 0
                                                   else "") + "，所以不采用。")
    if hyps:
        fix = {"H1": "处理残差漂移", "H2": "更细的 12 个月阶段", "H3": "只用水平视图", "H4": "估值调整 CMA",
               "H5": "补一个外生因子", "H6": "提高最少有效月份"}
        rej = [f"{fix[h['id']]}（{h['id']}）" for h in hyps if h["verdict"] == "rejected" and h["id"] in fix]
        inc = [f"{fix[h['id']]}（{h['id']}）" for h in hyps if h["verdict"] == "inconclusive" and h["id"] in fix]
        if rej:
            txt.append("没有帮助的改动：" + "、".join(rej) + "。")
        if inc:
            txt.append("没有定论：" + "、".join(inc) + "。")
    return "".join(txt)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Engine ablation over the backtest windows -> data/results/ablation.json")
    ap.add_argument("--paths", type=int, default=bt.N_PATHS, help="simulated paths per window and variant")
    ap.add_argument("--seed", type=int, default=bt.SEED)
    ap.add_argument("--extra-seeds", type=int, nargs="*", default=list(EXTRA_SEEDS))
    ap.add_argument("--scen-dir", default=None, help="keep the variants' scenario runs here (default: a temp dir, removed afterwards)")
    ap.add_argument("--scen-paths", type=int, default=SCEN_PATHS)
    ap.add_argument("--no-scenarios", action="store_true", help="skip the future-scenario runs")
    args = ap.parse_args(argv)
    ctx = EngineContext.load(selection=True)
    out = run(ctx, n_paths=args.paths, seed=args.seed, extra_seeds=tuple(args.extra_seeds),
              scenarios=not args.no_scenarios, scen_dir=args.scen_dir, scen_paths=args.scen_paths)
    jsonio.dump(out, config.RESULTS_DIR / "ablation.json")


if __name__ == "__main__":
    main()
