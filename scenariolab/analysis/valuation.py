"""Valuation and earnings channel: should the P/E, ROIC or free cash flow be macro factors?

The question (Chinese UI): treat the S&P 500 / QQQ P/E, ROIC or free cash flow as macro factors, or add
more factors? Answered with evidence in three parts.

1. Factors. A price is earnings x multiple, so a multiple's change in the same month is (almost) the
   price change itself; f_valuation (P/D) and f_cape (Shiller CAPE) are endogenous and only scored for
   reference. The exogenous new series (real 10-year yield, NIPA profit growth and profits minus
   capex, both dated at publication) are candidates of the walk-forward factor selection
   (walkforward.py, exactly like the existing ones); factor_candidates reports what they add.
2. Valuation as a state (engine/simulate.py ValuationModel): equity price = dividend growth + change in
   log P/D, with log P/D tracked along every path from today's level, a fair value
   ln P/D* = a + b real_y10 + c profit_share + d infl_expect estimated on the training months (Newey-West
   t statistics) and a pull -kappa (ln P/D - ln P/D*), beta-scaled for sectors. The variants (pull
   speed, drift it is added to, path-dependent dividend income, the fair-value formula) are chosen with
   the ablation's discipline: validation windows (tech_bubble, gfc_qe), one confirmation on the test
   windows (covid_ai, post_covid), seeds 11-15, coverage may fall at most 0.05, no risk asset's
   unconditional 15-year p5-p95 width may narrow more than 5%, and the one-year forecast (42 origins,
   CRPS and rank IC split at 2004) must not get worse. The current engine (config.RESIDUALS; CMA
   historical for the backtest fans, valuation_adjusted for scenarios and the one-year forecast) is the
   baseline.
3. Scenario effects: the 13 future scenarios' probability-weighted headlines under each mode (run into a
   scratch directory, never into data/results).

    python -m scenariolab.analysis.valuation [--paths 400] [--scen-dir DIR] [--no-scenarios] [--no-forecast]

writes data/results/valuation.json and nothing else under data/results.
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, jsonio
from ..context import EngineContext
from ..engine import summary
from ..engine.factor_model import ModelSpec
from ..engine.pooling import BLOCK, MIN_ENS, pool_phase
from ..engine.simulate import (ECM_MONTHS, KAPPA_BOUNDS, VALUATION_COLUMNS, VALUATION_FEATURE_NAMES, VALUATION_HAC_LAGS,
                               ValuationModel, capital_market_assumptions, fit_valuation, hac_ols, simulate)
from . import ablation as ab
from . import backtest as bt
from . import forecast as fc
from . import walkforward as wf

RESULT_PATH = config.RESULTS_DIR / "valuation.json"
VALIDATION, TEST = ab.VALIDATION, ab.TEST
RISK = bt.RISK_ASSETS
NAMES = bt.NAMES
WINDOW_NAMES = {w["id"]: w["name"] for w in bt.WINDOWS}
SEEDS = (bt.SEED, *ab.EXTRA_SEEDS)
COVERAGE_DROP, MIN_GAIN, FAN_DROP = ab.COVERAGE_DROP, ab.MIN_GAIN, ab.FAN_DROP
FAN_PATHS, FAN_SEED = ab.FAN_PATHS, ab.FAN_SEED
FORECAST_TOL = 0.01            # the one-year forecast's risk-asset CRPS may be at most 1% worse (each split)
FORECAST_IC_TOL = 0.05         # and its mean rank IC at most 0.05 lower (each split)
FORECAST_PATHS = fc.N_PATHS
SCEN_PATHS = ab.SCEN_PATHS
DATA_START = "1955-01"
HEADLINE_KEYS = ab.HEADLINE_KEYS
QUESTION = ("要不要把标普500或纳指（QQQ）的市盈率、ROIC、自由现金流也当成一个宏观因子？还是需要再加一些宏观因子？"
            "估值和盈利应该怎样进入模型？")

VS = {"enabled": True, "features": ("real_y10", "profit_share", "infl_expect"), "cma": "historical",
      "income": "constant"}


def _variant(vid, name, description, cma, settings=None, adoptable=True, forecast_cma=None):
    return {"id": vid, "name": name, "description": description, "cma": cma,
            "settings": ({**VS, **settings} if settings is not None else None), "adoptable": adoptable,
            "forecast_cma": forecast_cma or cma}


VARIANTS = [
    _variant("baseline", "现行引擎（不加估值拉力）",
             "回测扇形图用训练期的历史漂移，未来情景和一年期预测用估值调整 CMA；没有估值拉力："
             "未来情景只按 κ=0 记录每条路径的 P/D（回报不变），回测和一年期预测不记录。"
             "残差为引擎现行的 50% 部分耦合。", "historical", adoptable=False, forecast_cma="valuation_adjusted"),
    _variant("cma_historical", "参照：全部用历史漂移",
             "回测、一年期预测和未来情景都用训练期的历史漂移，不做任何估值假设（H4 的回测证据支持的设定）。",
             "historical", adoptable=False),
    _variant("cma_va", "参照：全部用估值调整 CMA",
             "去掉历史估值抬升、P/D 向样本中位数做常数回归（15 年回归 30%），即未来情景现在用的设定。",
             "valuation_adjusted", adoptable=False),
    _variant("vs_k05", "估值状态（κ=0.05，历史漂移）",
             "在历史漂移上加估值拉力 −κ·(ln P/D − ln P/D*)，公允值随模拟的实际利率、利润占比和通胀预期变动；κ=0.05/年"
             "（半衰期约 14 年）。", "historical", {"kappa": 0.05}),
    _variant("vs_k10", "估值状态（κ=0.10，历史漂移）", "同上，κ=0.10/年（半衰期约 7 年）。", "historical", {"kappa": 0.10}),
    _variant("vs_k20", "估值状态（κ=0.20，历史漂移）", "同上，κ=0.20/年（半衰期约 3.5 年）。", "historical", {"kappa": 0.20}),
    _variant("vs_ecm", "估值状态（κ 由训练期误差修正回归估计）",
             "κ 不事先给定：在训练期内，用“今天的估值缺口”回归“之后 12 个月 ln P/D 的变化”，斜率换算成年化 κ。",
             "historical", {"kappa": "ecm"}),
    _variant("vs_tw_k10", "估值状态（κ=0.10，去掉历史估值抬升）",
             "漂移改用估值状态模式的 CMA：去掉训练期的估值抬升（按 β 缩放）、行业相对漂移收缩 50%、黄金只跟通胀，"
             "但没有常数回归项——回归完全由状态拉力承担，不重复计算。", "valuation_state", {"kappa": 0.10, "cma": "valuation_state"}),
    _variant("vs_tw_ecm", "估值状态（κ 估计，去掉历史估值抬升）", "同上，κ 由训练期误差修正回归估计。",
             "valuation_state", {"kappa": "ecm", "cma": "valuation_state"}),
    _variant("vs_k10_income", "估值状态（κ=0.10）+ 股息率随估值路径变化",
             "股票的股息收入不再固定为起点股息率，而是起点股息率 × (P/D₀ / P/Dₜ)^β：估值下跌时股息率相应上升。",
             "historical", {"kappa": 0.10, "income": "path"}),
    _variant("vs_k00_income", "只让股息率随估值路径变化（κ=0）",
             "不加拉力，只检验“股息率 = 1/(P/D 路径)”这一记账方式本身是否改善回测。", "historical",
             {"kappa": 0.0, "income": "path"}),
    _variant("vs_k10_theory", "估值状态（κ=0.10，公允值不含通胀）",
             "公允值只用实际利率和利润占比（理论上的设定）：ln P/D* = a + b·实际利率 + c·利润占比。", "historical",
             {"kappa": 0.10, "features": ("real_y10", "profit_share")}),
]

FACTOR_CANDIDATES = ("real_y10", "profit_growth", "fcf_share", "profit_share", "capex_share", "cape", "valuation")

# Evidence on the fair value's economics (regression_evidence, persistence).
THEORY_SIGN = {"real_y10": -1, "y10": -1, "profit_share": 1}   # higher real yield -> lower multiple; profits -> higher
STABILITY_SPLIT = bt.WINDOWS[0]["train_end"]                    # non-overlapping halves split where the backtest starts
ROLLING_YEARS, ROLLING_STEP = 20, 5                             # rolling sub-samples of the fair-value regression
REAL_TIME_FIRST_YEARS = 12          # the first real-time fair value is fitted on this many years of months
REAL_TIME_MIN_PAIRS = 60            # past (gap, next-year return) pairs before the first out-of-sample prediction
ROIC_DEPRECIATION = 0.10            # perpetual-inventory depreciation of the ROIC proxy (annual)
CN = {"real_y10": "实际利率", "profit_share": "利润占比", "infl_expect": "通胀预期", "y10": "名义收益率"}


# ---------------------------------------------------------------- helpers

def _ym(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def _avg(values) -> float:
    return bt._avg(values)


def _settings_key(settings: dict | None) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in (settings or {}).items()))


def window_engine(win: ab.Window, variant: dict, cache: dict) -> tuple[dict, ValuationModel | None]:
    """(CMA, ValuationModel) of a variant in one window, estimated on its training months only."""
    key = (win.w["id"], variant["cma"], _settings_key(variant["settings"]))
    if key not in cache:
        cma = win.cmas.get(variant["cma"]) or capital_market_assumptions(win.db, win.anchor, method=variant["cma"])
        vm = None
        if variant["settings"]:
            s = variant["settings"]
            vm = fit_valuation(win.db, win.anchor, features=s["features"], kappa=s["kappa"], income=s["income"])
        cache[key] = (cma, vm)
    return cache[key]


def evaluate(panel: pd.DataFrame, win: ab.Window, variant: dict, n_paths: int, seed: int, cache: dict) -> dict:
    """Window metrics of one variant and seed: conditional and unconditional fans (definitions as
    backtest.window_summary and ablation.evaluate, risk assets) plus the P/D the paths end at."""
    plan, vw = ab.window_weights(panel, win, bt.PHASE_MONTHS, "full", MIN_ENS)
    cma, vm = window_engine(win, variant, cache)
    sims = {"cond": simulate(win.db, win.anchor, vw["cond"], plan, n_paths=n_paths, seed=seed, block=BLOCK, cma=cma,
                             overrides=vw["overrides"], valuation=vm, **config.RESIDUALS),
            "uncond": simulate(win.db, win.anchor, vw["prior"], plan, n_paths=n_paths, seed=seed, block=BLOCK, cma=cma,
                               overrides=vw["overrides_prior"], valuation=vm, **config.RESIDUALS)}
    W = {k: summary.wealth_paths(s.total) for k, s in sims.items()}
    years, A, O = win.years, win.actual, win.oracle
    per, parts = {}, {k: [] for k in ("scenario", "engine", "transmission")}
    for a in RISK:
        j = bt.ASSETS.index(a)
        p50 = {k: float(np.median(v[:, -1, j])) for k, v in W.items()}
        lo, hi = np.percentile(W["cond"][:, :, j], [5, 95], axis=0)
        dec = bt.decompose(A[-1, j], O[-1, j], p50["cond"], p50["uncond"], years)
        for k in parts:
            parts[k].append(abs(dec[f"{k}_annual"]))
        per[a] = {"actual": float(bt.cagr(A[-1, j], years)), "cond_p50": float(bt.cagr(p50["cond"], years)),
                  "uncond_p50": float(bt.cagr(p50["uncond"], years)), "coverage_cond": bt.coverage(A[:, j], lo, hi)}
    mae = lambda key: _avg(abs(per[a][key] - per[a]["actual"]) for a in RISK)       # noqa: E731
    out = {"cagr_mae_cond_risk": mae("cond_p50"), "cagr_mae_uncond_risk": mae("uncond_p50"),
           "coverage_cond_risk": _avg(per[a]["coverage_cond"] for a in RISK),
           "engine_abs_annual": _avg(parts["engine"]), "transmission_abs_annual": _avg(parts["transmission"]),
           "scenario_abs_annual": _avg(parts["scenario"]),
           "us_equity": {k: per["us_equity"][k] for k in ("actual", "cond_p50", "uncond_p50")}}
    pd_end = float(100.0 / panel["s_div_yield"].loc[win.months[-1]])
    out["pd"] = {"actual_end": pd_end, "start": float(100.0 / panel["s_div_yield"].loc[win.anchor.date])}
    if vm is not None:
        out["pd"].update({"cond_end_p50": float(np.median(sims["cond"].macro["pd"][:, -1])),
                          "uncond_end_p50": float(np.median(sims["uncond"].macro["pd"][:, -1])),
                          "fair_start": vm.today()["pd_fair"], "gap_start": vm.today()["gap_log"], "kappa": vm.kappa})
    return out


def _split(per_seed: list[dict], key: str, wins: tuple) -> np.ndarray:
    return np.array([np.mean([s[w][key] for w in wins]) for s in per_seed])


def forward_fan(ctx: EngineContext, variant: dict, n_paths: int = FAN_PATHS, seed: int = FAN_SEED) -> dict:
    """Unconditional 15-year CAGR distribution from today under a variant's scenario engine (the baseline
    is the current scenario engine: CMA valuation_adjusted, no valuation state)."""
    T = config.HORIZON_MONTHS
    prior = pool_phase(ctx.db.frame, [], ctx.db.prior(), T, BLOCK, rare=ctx.db.rare).blocks
    cma, vm = scenario_engine_parts(ctx, variant)
    s = simulate(ctx.db, ctx.anchor, [prior], [T], n_paths=n_paths, seed=seed, cma=cma, valuation=vm,
                 **config.RESIDUALS)
    W = summary.wealth_paths(s.total)
    years = T / 12.0
    out = {}
    for a in RISK:
        j = config.ASSET_IDS.index(a)
        c = W[:, -1, j] ** (1.0 / years) - 1.0
        out[a] = {"p5": float(np.percentile(c, 5)), "p50": float(np.percentile(c, 50)),
                  "p95": float(np.percentile(c, 95))}
    if vm is not None:
        out["pd_end_p50"] = float(np.median(s.macro["pd"][:, -1]))
    return out


def track_settings(cma: str) -> dict:
    """Valuation state tracked with kappa 0 and constant income: every return is exactly the engine's
    without it, and the P/D path shows the multiple that engine implies."""
    return {**VS, "cma": cma, "kappa": 0.0, "income": "constant"}


def implied_pd(ctx: EngineContext, n_paths: int = FAN_PATHS, seed: int = FAN_SEED) -> dict:
    """The market P/D after 15 unconditional years under the current engines (tracked, returns unchanged),
    and how much of the 15-year log price return's dispersion is the multiple's drift."""
    T = config.HORIZON_MONTHS
    prior = pool_phase(ctx.db.frame, [], ctx.db.prior(), T, BLOCK, rare=ctx.db.rare).blocks
    vm = fit_valuation(ctx.db, ctx.anchor, kappa=0.0, income="constant")
    j = config.ASSET_IDS.index("us_equity")
    out = {}
    for key, cma in (("valuation_adjusted", ctx.cma),
                     ("historical", capital_market_assumptions(ctx.db, ctx.anchor, method="historical"))):
        s = simulate(ctx.db, ctx.anchor, [prior], [T], n_paths=n_paths, seed=seed, cma=cma, valuation=vm,
                     **config.RESIDUALS)
        lpd = np.log(s.macro["pd"].astype(float))
        monthly = np.log1p((s.total[:, :, j] - s.income[:, :, j]).astype(float))
        lp = monthly.sum(axis=1)
        dval = lpd[:, -1] - lpd[:, 0]
        q = np.percentile(np.exp(lpd[:, -1]), [5, 50, 95])
        out[key] = {"pd_end": {"p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2])},
                    "pd_fair_end_p50": float(np.median(s.macro["pd_fair"][:, -1])),
                    "log_price_sd": float(lp.std()), "valuation_part_sd": float(dval.std()),
                    "dividend_part_sd": float((lp - dval).std()),
                    "valuation_dividend_corr": float(np.corrcoef(dval, lp - dval)[0, 1]),
                    "variance_ratio": float(lp.var() / (T * monthly.var())),
                    "valuation_change_mean": float(dval.mean()),
                    "pd_p50_by_year": [float(np.median(np.exp(lpd[:, 12 * y]))) for y in range(T // 12 + 1)]}
    return out


def scenario_engine_parts(ctx: EngineContext, variant: dict) -> tuple[dict, ValuationModel | None]:
    """(CMA, ValuationModel) a variant's scenarios would run with from today."""
    if variant["id"] == "baseline":
        return ctx.cma, None
    cma = capital_market_assumptions(ctx.db, ctx.anchor, method=variant["cma"])
    s = variant["settings"]
    vm = fit_valuation(ctx.db, ctx.anchor, features=s["features"], kappa=s["kappa"], income=s["income"]) if s else None
    return cma, vm


# ---------------------------------------------------------------- one-year forecast

def forecast_rows(panel: pd.DataFrame, spec: ModelSpec, dates: list, variant: dict, ens: float,
                  n_paths: int = FORECAST_PATHS, seed: int = fc.SEED) -> list[dict]:
    """The 42 rolling one-year forecasts of a variant (seeds and analog bandwidth as forecast.json)."""
    settings = variant["settings"]
    return [fc.forecast_origin(panel, spec, t, n_paths, seed + i, cma_method=variant["forecast_cma"],
                               analog_ens=ens, valuation=settings if settings else {"enabled": False})
            for i, t in enumerate(dates)]


def forecast_scores(rows: list[dict]) -> dict:
    cut = pd.Timestamp(fc.SPLIT)
    out = {}
    for tag, sel in (("val", [r for r in rows if pd.Timestamp(r["date"]) <= cut]),
                     ("test", [r for r in rows if pd.Timestamp(r["date"]) > cut]), ("all", rows)):
        sc = fc.scores(sel)
        out[tag] = {"n": len(sel), "crps_analog": sc["analog"]["risk_avg"]["crps"],
                    "crps_uncond": sc["uncond"]["risk_avg"]["crps"],
                    "coverage_analog": sc["analog"]["risk_avg"]["coverage90"],
                    "rank_ic_analog": sc["analog"]["rank_ic"]["mean"], "rank_ic_uncond": sc["uncond"]["rank_ic"]["mean"],
                    "us_equity_mae_p50_analog": sc["analog"]["assets"]["us_equity"]["mae_p50"],
                    "us_equity_bias_p50_analog": sc["analog"]["assets"]["us_equity"]["bias_p50"]}
    return out


def forecast_skill(score: dict, base: dict) -> dict:
    """CRPS skill (1 - CRPS / baseline CRPS, analog method, risk assets) and rank-IC change, per split."""
    out = {}
    for tag in ("val", "test", "all"):
        s, b = score[tag], base[tag]
        out[tag] = {"crps_skill": 1.0 - s["crps_analog"] / b["crps_analog"],
                    "rank_ic_delta": s["rank_ic_analog"] - b["rank_ic_analog"]}
    return out


# ---------------------------------------------------------------- selection

def pull_off_text() -> str:
    """What the engine keeps when no variant is adopted: no pull toward the fair value, the state still tracked."""
    tracked = bool(config.VALUATION.get("track"))
    return "估值拉力默认关闭" + ("（估值状态照常按 κ=0 逐路径记录，不改变回报）" if tracked else "") + "。"


def select(variants: list[dict]) -> dict:
    """The ablation's rule (validation pick, one test confirmation, coverage and 15-year fan guards) plus
    the one-year forecast as a second criterion."""
    by_id = {v["id"]: v for v in variants}
    base = by_id["baseline"]
    floor_val, floor_test = base["validation_coverage"] - COVERAGE_DROP, base["test_coverage"] - COVERAGE_DROP
    checks, eligible = {}, []
    for v in variants:
        if not v["adoptable"]:
            v["eligible"] = False
            continue
        need = max(MIN_GAIN, 2.0 * v["validation_gain_se"])
        c = {"coverage": v["validation_coverage"] >= floor_val,
             "forward_fan": (v["fan_width_ratio_min"] if v["fan_width_ratio_min"] is not None else 1.0) >= 1.0 - FAN_DROP,
             "gain": v["validation_gain"] > need}
        f = v.get("forecast_skill")
        if f is not None:
            c["forecast_val"] = f["val"]["crps_skill"] >= -FORECAST_TOL and f["val"]["rank_ic_delta"] >= -FORECAST_IC_TOL
        checks[v["id"]] = c
        v["eligible"] = all(c.values())
        if v["eligible"]:
            eligible.append(v)
    rule = (f"只有估值状态的变体可以采用（参照行只作比较）。候选必须同时满足：① 验证窗口（科技繁荣与泡沫破裂、金融危机与QE）"
            f"的条件覆盖率不低于现行引擎 − {COVERAGE_DROP:.2f}；② 从今天出发、无视图的 15 年分布里，每一个风险资产年化收益的"
            f" 5%–95% 宽度收窄不超过 {FAN_DROP:.0%}；③ 验证窗口条件风险资产年化 MAE 的改善大于 {MIN_GAIN:.1%}，且大于 "
            f"{len(SEEDS)} 个随机种子间标准误的 2 倍；④ 第二标准：一年期滚动预测在 {fc.SPLIT[:4]} 年及以前的起点上，风险资产 CRPS "
            f"不比现行引擎差 {FORECAST_TOL:.0%} 以上、排序相关 IC 不低 {FORECAST_IC_TOL:.2f} 以上。满足的候选中取验证 MAE 最低者，"
            f"然后只在测试窗口（疫情、再通胀与AI，疫情以来）和 {fc.SPLIT[:4]} 年以后的预测起点上确认一次：测试 MAE 不比现行引擎差"
            f"（容差 2 倍标准误）、测试覆盖率不低于现行引擎 − {COVERAGE_DROP:.2f}、之后起点的 CRPS 与 IC 同样不变差，才采用；"
            f"不通过就不采用，不再换第二名。")
    if not eligible:
        worse = [v for v in variants if v["adoptable"] and v["validation_gain"] < 0]
        best = max((v for v in variants if v["adoptable"]), key=lambda v: v["validation_gain"])
        reason = (f"没有估值状态变体通过：{len(worse)}/{sum(v['adoptable'] for v in variants)} 个在验证窗口上比现行引擎更差；"
                  f"最接近的是「{best['name']}」，验证 MAE {best['validation_mae']:.2%}（现行 {base['validation_mae']:.2%}，"
                  f"{-best['validation_gain']:+.2%}），测试 {best['test_mae']:.2%}（现行 {base['test_mae']:.2%}）。保留现行引擎，"
                  + pull_off_text())
        return {"rule": rule, "chosen": "baseline", "adopt": False, "checks": checks, "eligible": [], "reason": reason,
                "best_candidate": best["id"]}
    pick = min(eligible, key=lambda v: v["validation_mae"])
    test_ok = pick["test_mae"] <= base["test_mae"] + 2.0 * pick["test_gain_se"]
    cov_ok = pick["test_coverage"] >= floor_test
    f = pick.get("forecast_skill")
    fc_ok = f is None or (f["test"]["crps_skill"] >= -FORECAST_TOL and f["test"]["rank_ic_delta"] >= -FORECAST_IC_TOL)
    adopt = bool(test_ok and cov_ok and fc_ok)
    reason = (f"验证窗口选出「{pick['name']}」：验证 MAE {pick['validation_mae']:.2%}（现行 {base['validation_mae']:.2%}），"
              f"测试 MAE {pick['test_mae']:.2%}（现行 {base['test_mae']:.2%}），测试覆盖率 {pick['test_coverage']:.0%}"
              f"（现行 {base['test_coverage']:.0%}）。" + ("测试确认通过，采用。" if adopt else "测试没有确认（"
              + "、".join(x for x, ok in (("MAE 变差", test_ok), ("覆盖率下降过多", cov_ok), ("一年期预测变差", fc_ok)) if not ok)
              + "），不采用。"))
    return {"rule": rule, "chosen": pick["id"], "adopt": adopt, "checks": checks,
            "eligible": [v["id"] for v in eligible], "reason": reason}


# ---------------------------------------------------------------- factor candidates

def factor_candidates(panel: pd.DataFrame) -> dict:
    """Walk-forward evidence for the factor candidates (walkforward.run_walkforward, as diagnostics runs it).

    validation_delta: mean out-of-sample R^2 change when the factor is added alone to the base model,
    over the two decades that play the validation role (1995-2004 fitted on 1973-1994, 2005-2014 on
    1973-2004); test_delta: 2015-2026 fitted on 1973-2014. Positive = better than the base.
    """
    res = wf.run_walkforward(panel, ModelSpec())
    its = res["iterations"]
    abl = [{r["candidate"]: r for r in it["test_ablation"]} for it in its]
    accepted = {c for it in its for c in it["selected"]["changes"]}
    leakage = {f: wf.leakage_evidence(panel, f"f_{f}", (config.MODEL_START, _ym(panel.index.max())))
               for f in FACTOR_CANDIDATES if f"f_{f}" in panel}
    rows = []
    for f in FACTOR_CANDIDATES:
        cid = f"add_{f}"
        if not all(cid in a for a in abl):
            continue
        periods = [{"period": f"{its[0]['validate'][0][:4]}–{its[0]['validate'][1][:4]}", "delta": abl[0][cid]["val_delta"]}]
        periods += [{"period": f"{it['test'][0][:4]}–{it['test'][1][:4]}", "delta": a[cid]["test_delta"]}
                    for it, a in zip(its, abl)]
        endogenous = f in wf.ENDOGENOUS
        robust = all(p["delta"] > wf.MIN_GAIN for p in periods)
        spanned = None
        if not endogenous:
            from .diagnostics import spanned_r2
            rows_tr = panel.loc[config.MODEL_START:its[-1]["train"][1]].index
            spanned = spanned_r2(panel, ModelSpec(), f"f_{f}", rows_tr)
        if endogenous:
            verdict = "内生，只作对照"
            why = (f"{wf.ENDOGENOUS[f]}。单独加入后样本外 R² 大幅上升（"
                   + "、".join(f"{p['period']} {p['delta']:+.3f}" for p in periods)
                   + f"），正是因为它就是答案本身（与当月美股价格收益相关 {leakage[f]['same_month']:.2f}）。")
        elif robust:
            verdict = "稳健：值得加入"
            why = "每个样本外十年都提升样本外 R²。"
        else:
            verdict = "不加入"
            why = ("单独加入后样本外 R² 的变化：" + "、".join(f"{p['period']} {p['delta']:+.3f}" for p in periods)
                   + "，没有在每个十年都稳定提升" + (f"；它的月度变化 {_share(spanned)} 已能由现有 10 个因子线性解释"
                                                    if spanned is not None and np.isfinite(spanned) and spanned > 0.5 else "")
                   + "。")
        rows.append({"id": f, "name": wf.FACTOR_NAMES.get(f, f), "new": f in ("real_y10", "profit_growth", "fcf_share", "cape"),
                     "endogenous": endogenous, "release_dated": f in wf.RELEASE_DATED,
                     "validation_delta": float(np.mean([p["delta"] for p in periods[:2]])),
                     "test_delta": float(periods[-1]["delta"]), "periods": periods, "spanned_r2": spanned,
                     "accepted_in_walkforward": cid in accepted, "corr_same_month_equity": leakage.get(f, {}).get("same_month"),
                     "verdict": verdict, "why": why})
    fin = res["final"]
    return {"candidates": rows, "final": {"describe": fin["describe"], "changes": fin["changes"], "is_base": fin["is_base"],
                                          "reason": fin["reason"]},
            "iterations": [{"iter": it["iter"], "selected": it["selected"]["changes"],
                            "test_delta": it["test_result"]["delta"], "improved": it["test_result"]["improved"]}
                           for it in its],
            "min_gain": wf.MIN_GAIN}


# ---------------------------------------------------------------- regression evidence

def _fit_report(vm: ValuationModel) -> dict:
    return {"coefficients": {k: {"beta": vm.coef[k], "t": vm.tstat[k], "name": VALUATION_FEATURE_NAMES[k]}
                             for k in ("const", *vm.features)},
            "r2": vm.r2, "sample": list(vm.sample), "n_obs": vm.n_obs, "gap_sd": vm.gap_sd, "ecm": vm.ecm}


def _formula(features) -> str:
    sym = {"real_y10": "实际收益率", "profit_share": "利润占比", "infl_expect": "通胀预期", "y10": "名义收益率"}
    letters = "bcde"
    return "ln(P/D)* = a + " + " + ".join(f"{letters[i]}·{sym[f]}" for i, f in enumerate(features))


def _fit_rows(P: pd.DataFrame, features, start=None, end=None, lags: int = VALUATION_HAC_LAGS) -> dict | None:
    """The fair-value regression on the rows of P between start and end (None if too few months)."""
    lpd = np.log(100.0 / P["s_div_yield"]).loc[start:end]
    X = P[[VALUATION_COLUMNS[f] for f in features]].loc[start:end]
    ok = lpd.notna() & X.notna().all(axis=1)
    if ok.sum() < 120:
        return None
    b, t, r2, _ = hac_ols(lpd[ok].to_numpy(float), X[ok].to_numpy(float), lags)
    return {"start": _ym(lpd[ok].index.min()), "end": _ym(lpd[ok].index.max()), "n_obs": int(ok.sum()), "r2": float(r2),
            "coefficients": {k: {"beta": float(bb), "t": float(tt)} for k, bb, tt in zip(("const", *features), b, t)}}


def stability(P: pd.DataFrame, features, split: str = STABILITY_SPLIT, years: int = ROLLING_YEARS,
              step: int = ROLLING_STEP) -> dict:
    """The fair-value regression on non-overlapping sub-samples (before / after `split`) and on rolling
    `years`-year windows; per feature, how many rolling windows give a significant (|t| >= 2) positive or
    negative coefficient. The expanding training windows (by_window) all contain 1973-1994, so they hide
    how much the relation moves later."""
    split_ts = pd.Timestamp(split)
    halves = [h for h in (_fit_rows(P, features, None, split_ts),
                          _fit_rows(P, features, split_ts + pd.DateOffset(months=1), None)) if h is not None]
    first, last = P.index.min().year, P.index.max().year
    rolling = []
    for y0 in range(first, last - years + 2, step):
        r = _fit_rows(P, features, f"{y0}-01", f"{y0 + years - 1}-12")
        if r is not None:
            rolling.append(r)
    counts = {f: {"n": len(rolling),
                  "sig_pos": sum(r["coefficients"][f]["t"] >= 2.0 for r in rolling),
                  "sig_neg": sum(r["coefficients"][f]["t"] <= -2.0 for r in rolling)} for f in features}
    flips = [f for f in features if len(halves) == 2
             and np.sign(halves[0]["coefficients"][f]["beta"]) != np.sign(halves[1]["coefficients"][f]["beta"])
             and max(abs(h["coefficients"][f]["t"]) for h in halves) >= 2.0]
    return {"split": split, "halves": halves, "rolling_years": years, "rolling_step": step, "rolling": rolling,
            "rolling_counts": counts, "sign_flips": flips}


def theory_check(coef: dict, stab: dict | None = None) -> dict:
    """Coefficient signs against theory (THEORY_SIGN), with the fair value's response to +1 unit of each."""
    out = {}
    for f, c in coef.items():
        if f not in THEORY_SIGN:
            continue
        ok = np.sign(c["beta"]) == THEORY_SIGN[f]
        rc = (stab or {}).get("rolling_counts", {}).get(f)
        wrong = (rc["sig_pos"] if THEORY_SIGN[f] < 0 else rc["sig_neg"]) if rc else None
        out[f] = {"expected": "+" if THEORY_SIGN[f] > 0 else "-", "beta": c["beta"], "t": c["t"], "consistent": bool(ok),
                  "significant_wrong": bool((not ok) and abs(c["t"]) >= 2.0),
                  "fair_pd_change_per_unit": float(np.expm1(c["beta"])),
                  "rolling_significant_wrong": wrong, "rolling_n": rc["n"] if rc else None}
    return out


def real_time_gap(P: pd.DataFrame, features, first_years: int = REAL_TIME_FIRST_YEARS) -> pd.Series:
    """Gap of log P/D to a fair value refitted every December on the months up to then: a month's gap uses
    the fit of the December before it and the month's own levels, i.e. only what was known at the time
    (the in-sample residual of fit_valuation uses the whole sample's coefficients)."""
    lpd = np.log(100.0 / P["s_div_yield"])
    X = P[[VALUATION_COLUMNS[f] for f in features]]
    ok = lpd.notna() & X.notna().all(axis=1)
    lpd, X = lpd[ok], X[ok]
    if lpd.empty:
        return pd.Series(dtype=float)
    first_end = lpd.index[0] + pd.DateOffset(years=first_years) - pd.DateOffset(months=1)
    fits = {}
    for end in lpd.index[(lpd.index.month == 12) & (lpd.index >= first_end)]:
        Z = np.column_stack([np.ones(int((lpd.index <= end).sum())), X.loc[:end].to_numpy(float)])
        fits[end.year] = np.linalg.lstsq(Z, lpd.loc[:end].to_numpy(float), rcond=None)[0]
    gap = {d: float(lpd[d] - (b[0] + X.loc[d].to_numpy(float) @ b[1:]))
           for d in lpd.index if (b := fits.get(d.year - 1)) is not None}
    return pd.Series(gap, dtype=float)


def _half_life(rho: float, per_year: int) -> float | None:
    return float(np.log(0.5) / (per_year * np.log(rho))) if 0.0 < rho < 1.0 else None


def persistence(P: pd.DataFrame, features, gap_in_sample: pd.Series | None = None,
                horizon: int = config.HORIZON_MONTHS) -> dict:
    """How fast the multiple and the gap revert, from the db months only.

    AR(1) of log P/D (monthly and December-to-December) and of the in-sample gap; the error-correction
    slope on the *real-time* gap (real_time_gap; the in-sample gap is fitted to revert and overstates the
    speed); the recursive out-of-sample R^2 of the real-time gap for the next 12 months' log price return
    against the expanding historical mean (Campbell-Thompson); and the variance ratio of `horizon`-month
    log price returns (overlapping windows) to `horizon` x the monthly variance (1 = random walk).
    """
    lpd = np.log(100.0 / P["s_div_yield"]).dropna()
    out: dict = {}

    def ar1(s: pd.Series) -> float:
        return float(np.polyfit(s.to_numpy(float)[:-1], s.to_numpy(float)[1:], 1)[0])

    rho = ar1(lpd)
    dec = lpd[lpd.index.month == 12]
    rho_a = ar1(dec) if len(dec) > 10 else float("nan")
    out.update({"log_pd_ar1_monthly": rho, "log_pd_half_life_years": _half_life(rho, 12),
                "log_pd_ar1_annual": rho_a, "log_pd_half_life_years_annual": _half_life(rho_a, 1), "n_years": len(dec),
                "sample": [_ym(lpd.index.min()), _ym(lpd.index.max())]})
    if gap_in_sample is not None and len(gap_in_sample) > 24:
        rg = ar1(gap_in_sample)
        out.update({"gap_ar1_monthly": rg, "gap_half_life_years": _half_life(rg, 12)})
    gap = real_time_gap(P, features)
    ecm = {"slope": None, "t": None, "kappa": None, "half_life_years": None, "n": 0, "first": None}
    oos = {"r2": None, "n": 0}
    if len(gap) > REAL_TIME_MIN_PAIRS:
        ahead = lpd.reindex(gap.index + pd.DateOffset(months=ECM_MONTHS)).to_numpy(float)
        y = ahead - lpd.reindex(gap.index).to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() >= REAL_TIME_MIN_PAIRS:
            b, t, _, _ = hac_ols(y[ok], gap.to_numpy(float)[ok], lags=ECM_MONTHS)
            k = float(np.clip(-np.log1p(max(float(b[1]), -0.999)) * 12.0 / ECM_MONTHS, *KAPPA_BOUNDS))
            ecm = {"slope": float(b[1]), "t": float(t[1]), "kappa": k, "half_life_years": _half_life(np.exp(-k), 1),
                   "n": int(ok.sum()), "first": _ym(gap.index.min())}
        # next-12-month log price return, known 12 months later
        lp = np.log1p(P["p_us_equity"].astype(float)).cumsum()
        fwd = (lp.shift(-ECM_MONTHS) - lp).reindex(gap.index)
        err_m, err_h = [], []
        for d in gap.index:
            if not np.isfinite(fwd[d]):
                continue
            past = gap.index[(gap.index + pd.DateOffset(months=ECM_MONTHS)) <= d]
            yp, xp = fwd.reindex(past).to_numpy(float), gap.reindex(past).to_numpy(float)
            m = np.isfinite(yp)
            if m.sum() < REAL_TIME_MIN_PAIRS:
                continue
            slope, icpt = np.polyfit(xp[m], yp[m], 1)
            err_m.append((fwd[d] - (icpt + slope * gap[d])) ** 2)
            err_h.append((fwd[d] - yp[m].mean()) ** 2)
        if err_m:
            oos = {"r2": float(1.0 - np.sum(err_m) / np.sum(err_h)), "n": len(err_m)}
    out["ecm_real_time"] = ecm
    out["oos_r2_price_12m"] = oos
    x = np.log1p(P["p_us_equity"].astype(float)).dropna()
    if len(x) > 2 * horizon:
        c = x.cumsum()
        long = (c.shift(-horizon) - c).dropna()
        out["variance_ratio"] = {"months": horizon, "ratio": float(long.var() / (horizon * x.var())),
                                 "independent_windows": float(len(x) / horizon)}
    return out


def roic_proxy(panel: pd.DataFrame, depreciation: float = ROIC_DEPRECIATION) -> pd.Series:
    """After-tax profits / a perpetual-inventory nonresidential capital stock (percent), from the panel's
    monthly profit and investment levels (both $bn at annual rates, interpolated from quarters): a
    descriptive proxy only, it is not release-dated."""
    inv = panel["lvl_pnfi"].dropna()
    d = 1.0 - (1.0 - depreciation) ** (1.0 / 12.0)
    k = float(inv.iloc[:12].mean()) / depreciation                      # steady state at the first year
    cap = []
    for v in inv.to_numpy(float):
        k = (1.0 - d) * k + v / 12.0
        cap.append(k)
    return 100.0 * panel["lvl_profits"].reindex(inv.index) / pd.Series(cap, index=inv.index)


def roic_evidence(ctx: EngineContext, features) -> dict | None:
    """How closely the ROIC proxy tracks the profit share, and the fair-value coefficient if it replaces it."""
    if not {"lvl_pnfi", "lvl_profits", "s_profit_share"} <= set(ctx.panel.columns) or "profit_share" not in features:
        return None
    P = ctx.db.frame
    roic = roic_proxy(ctx.panel).reindex(P.index)
    both = pd.concat([roic, P["s_profit_share"]], axis=1).dropna()
    if len(both) < 120:
        return None
    X = pd.concat([P[VALUATION_COLUMNS[f]] if f != "profit_share" else roic for f in features], axis=1)
    lpd = np.log(100.0 / P["s_div_yield"])
    ok = lpd.notna() & X.notna().all(axis=1)
    b, t, r2, _ = hac_ols(lpd[ok].to_numpy(float), X[ok].to_numpy(float), VALUATION_HAC_LAGS)
    j = 1 + list(features).index("profit_share")
    return {"depreciation": ROIC_DEPRECIATION, "corr_with_profit_share": float(both.corr().iloc[0, 1]),
            "sample": [_ym(both.index.min()), _ym(both.index.max())], "coef_in_fair_value": float(b[j]),
            "t_in_fair_value": float(t[j]), "r2": float(r2), "level_today": float(roic.dropna().iloc[-1])}


def regression_evidence(ctx: EngineContext, vm: ValuationModel) -> dict:
    """Today's fair-value regression (all months 1973-today), its alternatives, its stability on
    non-overlapping and rolling sub-samples and by backtest window, and the theory sign check."""
    P = ctx.db.frame
    alt = []
    for feats in (("real_y10", "profit_share"), ("y10", "profit_share"), ("real_y10",), ("infl_expect",)):
        m = fit_valuation(ctx.db, ctx.anchor, features=feats, kappa=0.0)
        alt.append({"features": list(feats), "formula": _formula(feats), **_fit_report(m),
                    "pd_fair_today": m.today()["pd_fair"], "gap_log_today": m.today()["gap_log"]})
    cape = np.log(P["s_cape"])
    X = P[[VALUATION_COLUMNS[f] for f in vm.features]]
    ok = cape.notna() & X.notna().all(axis=1)
    cape_fit = None
    if ok.sum() > 120:
        b, t, r2, _ = hac_ols(cape[ok].to_numpy(float), X[ok].to_numpy(float), VALUATION_HAC_LAGS)
        cape_fit = {"dependent": "ln CAPE", "coefficients": {k: {"beta": float(bb), "t": float(tt)}
                                                             for k, bb, tt in zip(("const", *vm.features), b, t)},
                    "r2": float(r2)}
    by_window = []
    for w in bt.WINDOWS:
        win = ab.setup_window(ctx.panel, ctx.spec, w)
        m = fit_valuation(win.db, win.anchor, features=vm.features, kappa="ecm")
        end = ctx.panel.loc[win.months[-1]]
        fair_end = float(np.exp(m.fair_log_pd({f: float(end[VALUATION_COLUMNS[f]]) for f in m.features})))
        by_window.append({"window": w["id"], "name": w["name"], "train_end": w["train_end"], **_fit_report(m),
                          "pd_start": m.today()["pd"], "pd_fair_start": m.today()["pd_fair"],
                          "gap_start": m.today()["gap_log"], "kappa_ecm": m.ecm["kappa"],
                          "pd_end_actual": float(100.0 / end["s_div_yield"]),
                          "pd_fair_end_realized": fair_end,       # the training model at the realized end states
                          "gap_end_realized": float(np.log(100.0 / end["s_div_yield"]) - np.log(fair_end))})
    stab = stability(P, vm.features)
    rep = _fit_report(vm)
    return {"formula": _formula(vm.features), **rep, "hac_lags": VALUATION_HAC_LAGS, "alternatives": alt,
            "cape": cape_fit, "by_window": by_window, "stability": stab,
            "theory_check": theory_check({k: v for k, v in rep["coefficients"].items() if k != "const"}, stab)}


def today_block(ctx: EngineContext, vm: ValuationModel, kappa: float, reg: dict | None = None) -> dict:
    """Today's P/D against its fair value, the pull it implies, what moves the fair value and (with `reg`,
    regression_evidence) the gap under the alternative fair-value formulas."""
    P = ctx.db.frame
    means = {f: float(P[VALUATION_COLUMNS[f]].mean()) for f in vm.features}
    fair_mean = vm.coef["const"] + sum(vm.coef[f] * means[f] for f in vm.features)
    drivers = {f: float(vm.coef[f] * (vm.state0[f] - means[f])) for f in vm.features}
    t = vm.today()
    gap = t["gap_log"]
    hist_pd = 100.0 / P["s_div_yield"]
    cape = ctx.panel["s_cape"].dropna()
    alt = {a["formula"]: a["gap_log_today"] for a in (reg or {}).get("alternatives", []) if "gap_log_today" in a}
    gaps = [gap, *alt.values()]
    years = config.HORIZON_MONTHS / 12.0
    # the gap's own place in its history (the regression months, today's coefficients): the P/D percentile
    # alone overstates how dear today is relative to the fair value
    hist_gap = in_sample_gap(ctx.db, vm)
    gap_pct = float((hist_gap < gap).mean()) if len(hist_gap) else None
    gap_max = ({"value": float(hist_gap.max()), "date": _ym(hist_gap.idxmax())} if len(hist_gap) else None)
    return {"date": ctx.anchor.date.strftime("%Y-%m"), "pd": t["pd"], "pd_fair": t["pd_fair"], "gap_log": gap,
            "gap_percentile": gap_pct, "gap_max": gap_max,
            "gap_log_alternatives": alt, "gap_log_range": [float(min(gaps)), float(max(gaps))],
            "kappa": kappa, "implied_annual_drag": -kappa * gap, "horizon_years": years,
            "implied_annual_drag_15y": -gap * (1.0 - np.exp(-years * kappa)) / years,
            "drivers": drivers, "driver_names": {f: VALUATION_FEATURE_NAMES[f] for f in vm.features},
            "levels": {f: vm.state0[f] for f in vm.features}, "sample_means": means,
            "pd_fair_at_sample_means": float(np.exp(fair_mean)),
            "pd_percentile": float((hist_pd < t["pd"]).mean()), "pd_median": float(hist_pd.median()),
            "cape": float(cape.iloc[-1]) if len(cape) else None,
            "cape_percentile": float((cape.loc[config.MODEL_START:] < cape.iloc[-1]).mean()) if len(cape) else None,
            "profit_share": vm.state0["profit_share"], "real_y10": vm.state0["real_y10"],
            "infl_expect": vm.state0["infl_expect"]}


def data_block(ctx: EngineContext, vm: ValuationModel, start: str = DATA_START) -> dict:
    P = ctx.panel.loc[start:]
    X = {f: P[VALUATION_COLUMNS[f]].to_numpy(float) for f in vm.features}
    fair = np.exp(vm.fair_log_pd(X))

    def col(x, nd=3):
        return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(x, float)]
    return {"dates": [d.strftime("%Y-%m") for d in P.index], "pd": col(100.0 / P["s_div_yield"], 2),
            "pd_fair": col(fair, 2), "real_yield": col(P["s_real_y10"]), "profit_share": col(P["s_profit_share_rt"]),
            "profit_growth_yoy": col(P["s_profit_growth_yoy"], 2), "pe_proxy": col(P["s_cape"], 2),
            "fcf_proxy": col(P["s_fcf_share"]), "infl_expect": col(P["s_infl_expect"]),
            "eras": [{"id": e["id"], "name": e["name"], "start": e["start"], "end": e["end"]} for e in config.ERAS],
            "units": {"pd": "倍（市价/过去 12 个月股息）", "pd_fair": f"倍（今天的公允值回归，{vm.sample[0]} 以前为回推）",
                      "real_yield": "%", "profit_share": "% of GDP", "profit_growth_yoy": "%", "pe_proxy": "倍（席勒 CAPE）",
                      "fcf_proxy": "% of GDP（税后利润 − 非住宅投资，未加回折旧）", "infl_expect": "%"}}


# ---------------------------------------------------------------- scenarios

@contextlib.contextmanager
def scenario_mode(settings: dict | None):
    """Run the scenario runner with a valuation setting (runner.py reads runner.VALUATION)."""
    from ..scenarios import runner
    saved = runner.VALUATION
    runner.VALUATION = dict(settings) if settings else {"enabled": False}
    try:
        yield
    finally:
        runner.VALUATION = saved


def _headlines(index: dict, out_dir: Path) -> dict:
    out = {}
    for s in index["scenarios"]:
        h = {**{k: s["headline"].get(k) for k in HEADLINE_KEYS}, "prob": s["prob"]}
        try:
            d = jsonio.load(out_dir / "scenarios" / f"{s['id']}.json")
            pdp = d["macro"].get("pd", {}).get("fan", {}).get("p50")
            h["pd_end_p50"] = float(pdp[-1]) if pdp else None
        except (OSError, KeyError, ValueError):
            h["pd_end_p50"] = None
        out[s["id"]] = h
    return out


def _prob_weighted(h: dict, keys=HEADLINE_KEYS) -> dict:
    return {k: float(sum(v["prob"] * v[k] for v in h.values() if v.get(k) is not None)) for k in keys}


def scenario_effects(ctx: EngineContext, variants: list[dict], scen_dir: Path, n_paths: int = SCEN_PATHS) -> dict:
    """Headlines of the 13 future scenarios under each mode; every run goes to its own subdirectory of
    `scen_dir`, which is removed after its headlines are read."""
    from ..scenarios import library
    scen_dir = Path(scen_dir).resolve()
    results = config.RESULTS_DIR.resolve()
    if scen_dir == results or results in scen_dir.parents:
        raise ValueError("valuation scenario runs must not write into data/results")
    heads = {}
    for v in variants:
        c = ctx
        if v["id"] != "baseline":
            c = replace(ctx, cma=capital_market_assumptions(ctx.db, ctx.anchor, method=v["cma"]))
        # modes without a valuation state are run with it tracked at kappa 0 (returns unchanged) for their P/D path
        settings = v["settings"] or track_settings("valuation_adjusted" if v["id"] == "baseline" else v["cma"])
        out = scen_dir / v["id"]
        with scenario_mode(settings):
            index = library.run(c, n_paths=n_paths, out_dir=out)
        heads[v["id"]] = _headlines(index, out)
        shutil.rmtree(out, ignore_errors=True)
    base = heads["baseline"]
    pb = _prob_weighted(base)
    res = {"n_paths": n_paths, "keys": list(HEADLINE_KEYS), "baseline": base, "baseline_prob_weighted": pb,
           "units": "cpi_avg 为百分数（3.4 = 3.4%），pd_end_p50 为 15 年后市价/股息的中位数（倍），其余为小数；"
                    "delta = 该模式 − 现行引擎；prob 为情景概率",
           "modes": {}}
    for vid, h in heads.items():
        if vid == "baseline":
            continue
        pw = _prob_weighted(h)
        delta = {sid: {k: (h[sid][k] - base[sid][k]) if h[sid][k] is not None and base[sid][k] is not None else None
                       for k in HEADLINE_KEYS} for sid in base}
        big = max(base, key=lambda sid: abs(delta[sid]["us_equity_cagr_p50"]))
        pd_end = [v["pd_end_p50"] for v in h.values() if v.get("pd_end_p50") is not None]
        res["modes"][vid] = {"prob_weighted": pw, "prob_weighted_delta": {k: pw[k] - pb[k] for k in HEADLINE_KEYS},
                             "per_scenario": {sid: {"us_equity_cagr_p50": h[sid]["us_equity_cagr_p50"],
                                                    "us_equity_delta": delta[sid]["us_equity_cagr_p50"],
                                                    "nasdaq_delta": delta[sid]["nasdaq_cagr_p50"],
                                                    "pd_end_p50": h[sid].get("pd_end_p50")} for sid in base},
                             "max_abs_us_equity_delta": abs(delta[big]["us_equity_cagr_p50"]), "max_scenario": big,
                             "pd_end_p50_prob_weighted": (float(sum(v["prob"] * v["pd_end_p50"] for v in h.values()
                                                                    if v.get("pd_end_p50") is not None)) if pd_end else None)}
    try:
        built = jsonio.load(config.RESULTS_DIR / "scenarios.json")
        bh = {s["id"]: s["headline"]["us_equity_cagr_p50"] for s in built["scenarios"]}
        res["baseline_matches_results"] = all(abs(bh[s] - base[s]["us_equity_cagr_p50"]) < 1e-5 for s in base if s in bh)
    except (OSError, KeyError, ValueError):
        res["baseline_matches_results"] = None
    return res


# ---------------------------------------------------------------- text

def _pct(x, nd=1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}%}"


def _share(x: float) -> str:
    """An R^2 share as diagnostics shows it (a decimal kept near 1)."""
    from .diagnostics import share_text
    return share_text(float(x))


LITERATURE = {"source": "Cochrane (2008, Review of Financial Studies 21(4))，美国 1926–2004 年度数据", "dp_ar1_annual": 0.941}


def _coef_list(coefs: dict, feats, nd: int = 3) -> str:
    return "、".join(f"{CN.get(k, k)} {coefs[k]['beta']:+.{nd}f}（t={coefs[k]['t']:.1f}）" for k in feats)


def summary_text(reg: dict, today: dict, sel: dict, by_id: dict, facs: dict, scen: dict | None,
                 ipd: dict | None = None, paragraphs: bool = False, kap: dict | None = None):
    """The page's summary (Chinese), as one string or (paragraphs=True) its paragraphs. Every number and
    every directional word (higher / lower, significant, against theory, adopted) comes from the inputs."""
    c = reg["coefficients"]
    base = by_id["baseline"]
    cand = [v for v in by_id.values() if v["adoptable"]]
    n_worse = sum(v["validation_gain"] < 0 for v in cand)
    n_worse_t = sum(v["test_gain"] < 0 for v in cand)
    new = [r for r in facs["candidates"] if r["new"] and not r["endogenous"]]
    robust_new = [r for r in new if r["verdict"].startswith("稳健")]
    endo = [r for r in facs["candidates"] if r["endogenous"]]
    feats = [k for k in c if k != "const"]
    sig = [k for k in feats if abs(c[k]["t"]) >= 2.0]
    insig = [k for k in feats if abs(c[k]["t"]) < 2.0]
    th = reg.get("theory_check") or {}
    wrong = [k for k in feats if k in th and not th[k]["consistent"]]
    adopted = bool(sel.get("adopt"))
    since = reg["sample"][0][:4]
    years = today.get("horizon_years", config.HORIZON_MONTHS / 12.0)
    low_corr = [r for r in new if r["release_dated"] and r["corr_same_month_equity"] is not None]
    txt = [
        "结论一：市盈率不能当成同月的宏观因子。股价 = 盈利 × 市盈率，市盈率这个月的变化几乎就是这个月的股价变化——"
        + "；".join(f"{r['name']}与当月美股价格收益相关 {r['corr_same_month_equity']:.2f}" for r in endo
                   if r["corr_same_month_equity"] is not None)
        + "，走步检验里加入它们样本外 R² 大涨，只是因为它们就是答案本身，所以只作对照、不参与选择。"
        "纳指（QQQ）的市盈率没有免费的历史数据，这里不编造。",
        ("结论二：不需要再加宏观因子。" if not robust_new else "结论二：有新因子值得加入。")
        + "新增的外生候选单独加入基准模型时样本外 R² 的变化："
        + "；".join(f"{r['name']} " + "、".join(f"{p['period']} {p['delta']:+.3f}" for p in r["periods"]) for r in new)
        + f"（每个十年都要 > {facs['min_gain']} 才算稳定）；走步选择的最终模型仍是 {facs['final']['describe']}。"
        + ("按发布月对齐（不前视）后，" + "、".join(f"{r['name']}与当月美股价格收益的相关只有 {r['corr_same_month_equity']:+.2f}"
                                             for r in low_corr) + "；" if low_corr else "")
        + "".join(f"{r['name']}的月度变化有 {_share(r['spanned_r2'])} 已能由现有因子线性解释。" for r in new
                  if r["spanned_r2"] is not None and np.isfinite(r["spanned_r2"]) and r["spanned_r2"] > 0.5),
    ]
    # ---- conclusion three: the fair value, its signs against theory and its stability
    p3 = (f"结论三：估值应当是一个“状态”，而不是驱动因子。今天的公允值回归（{reg['sample'][0]} 至 {reg['sample'][1]}，"
          f"Newey-West t 值）：{reg['formula']}，" + _coef_list(c, feats) + f"，R² {reg['r2']:.2f}。"
          + (f"显著的只有{'、'.join(CN.get(k, k) for k in sig)}" if sig else "没有一个系数显著")
          + (f"，{'、'.join(CN.get(k, k) for k in insig)}的系数不显著" if insig else "")
          + ("：在全样本里，决定美股 P/D 高低的主要是通胀（高通胀时期估值低，即所谓“货币幻觉”）"
             if "infl_expect" in sig and c["infl_expect"]["beta"] < 0 else "") + "。")
    for k in wrong:
        x = th[k]
        p3 += (f"{CN.get(k, k)}的系数{'为正' if x['beta'] > 0 else '为负'}，与理论（{CN.get(k, k)}越高估值越"
               f"{'低' if x['expected'] == '-' else '高'}）相反{'且显著' if x['significant_wrong'] else ''}：按这个回归，"
               f"{CN.get(k, k)}上升 1 个百分点，公允 P/D 反而{'上升' if x['fair_pd_change_per_unit'] > 0 else '下降'} "
               f"{abs(x['fair_pd_change_per_unit']):.1%}"
               + (f"；{x['rolling_n']} 个滚动窗口中有 {x['rolling_significant_wrong']} 个显著地与理论相反"
                  if x.get("rolling_n") and x.get("rolling_significant_wrong") else "") + "。")
    theory_feats = [k for k in feats if k in th]
    if theory_feats and all(k in insig or k in wrong for k in theory_feats):
        p3 += (f"“{'、'.join(CN.get(k, k) + ('高' if th[k]['expected'] == '+' else '低') for k in theory_feats)}"
               f"就该有更高的估值”在 {since} 年以来的美国数据里得不到支持。")
    stab = reg.get("stability") or {}
    halves = stab.get("halves") or []
    if len(halves) == 2:
        h0, h1 = halves
        flips = stab.get("sign_flips") or []          # sign changes with |t| >= 2 in at least one half
        weak = [k for k in feats if k not in flips and k in h0["coefficients"] and k in h1["coefficients"]
                and np.sign(h0["coefficients"][k]["beta"]) != np.sign(h1["coefficients"][k]["beta"])]
        r2_drop = min(h0["r2"], h1["r2"]) < 0.5 * max(h0["r2"], h1["r2"])
        p3 += ("分段看" + ("并不稳定" if flips or r2_drop else "大体稳定") + f"：{h0['start']} 至 {h0['end']} "
               + _coef_list(h0["coefficients"], feats) + f"，R² {h0['r2']:.2f}；{h1['start']} 至 {h1['end']} "
               + _coef_list(h1["coefficients"], feats) + f"，R² {h1['r2']:.2f}"
               + (f"——{'、'.join(CN.get(k, k) for k in flips)}的系数在两段里符号相反（至少一段显著）" if flips else "")
               + ((f"；{'、'.join(CN.get(k, k) for k in weak)}的符号也变了，但两段都不显著" if flips else
                   f"——{'、'.join(CN.get(k, k) for k in weak)}的符号变了，但两段都不显著") if weak else "") + "。")
        rc = stab.get("rolling_counts") or {}
        if rc and any(v["sig_pos"] and v["sig_neg"] for v in rc.values()):
            both = [k for k in feats if rc.get(k, {}).get("sig_pos") and rc.get(k, {}).get("sig_neg")]
            p3 += (f"{stab['rolling_years']} 年滚动窗口里，{'、'.join(CN.get(k, k) for k in both)}的系数既有显著为正、"
                   "也有显著为负的时期：没有一个能跨时期使用的公允值公式。")
    txt.append(p3)
    # ---- today
    dmax = max(today["drivers"], key=lambda k: abs(today["drivers"][k]))
    rel = "高于" if today["pd_fair"] > today["pd_fair_at_sample_means"] else "低于"
    alt = today.get("gap_log_alternatives") or {}
    lo, hi = today.get("gap_log_range", [today["gap_log"], today["gap_log"]])
    gp, gm = today.get("gap_percentile"), today.get("gap_max")
    txt.append(
        f"今天（{today['date']}）P/D {today['pd']:.0f} 倍，高于 {since} 年以来 {today['pd_percentile']:.0%} 的月份"
        + (f"（席勒 CAPE {today['cape']:.1f}，高于 {today['cape_percentile']:.0%} 的月份）" if today.get("cape") is not None else "")
        + f"；公允值 {today['pd_fair']:.0f} 倍，对数缺口 {today['gap_log']:+.2f}"
        + (f"（约贵 {np.expm1(today['gap_log']):.0%}）" if today["gap_log"] > 0 else f"（约便宜 {-np.expm1(today['gap_log']):.0%}）")
        + (f"，缺口本身高于 {gp:.0%} 的月份" + (f"（最高 {gm['value']:+.2f}，{gm['date']}）" if gm else "")
           if gp is not None else "") + "。"
        f"公允值{rel}各变量取样本均值时的 {today['pd_fair_at_sample_means']:.0f} 倍，"
        + "、".join(f"{CN.get(k, k)} {v:+.2f}" for k, v in sorted(today["drivers"].items(), key=lambda kv: -abs(kv[1])))
        + f"（对数点）；其中{CN.get(dmax, dmax)}的贡献最大"
        + ("，但它的系数并不显著，这部分最不可靠" if dmax in insig else "")
        + ("，而且符号与理论相反" if dmax in wrong else "") + "。"
        + (f"换用另外 {len(alt)} 种公允值公式，今天的缺口在 {lo:+.2f} 到 {hi:+.2f} 之间。" if alt else "")
        + f"如果按参考拉力 κ={today['kappa']:.2f}/年（{'采用的设定' if adopted else '回测不支持，见下'}），第一年会拖累股价约 "
        f"{_pct(-today['implied_annual_drag'])}，{years:.0f} 年平均每年约 {_pct(-today['implied_annual_drag_15y'])}。")
    pers = (kap or {}).get("persistence") or {}
    if pers:
        e = pers.get("ecm_real_time") or {}
        ecm_in = (kap or {}).get("ecm_full_sample") or {}
        hl_a = pers.get("log_pd_half_life_years_annual")
        line = "拉力有多快："
        if hl_a:
            line += (f"ln P/D 的年度自相关 {pers['log_pd_ar1_annual']:.3f}（{pers['sample'][0][:4]}–{pers['sample'][1][:4]}，"
                     f"半衰期约 {hl_a:.0f} 年；文献中美国股息率的年度自相关约 {LITERATURE['dp_ar1_annual']}，半衰期约 "
                     f"{np.log(0.5) / np.log(LITERATURE['dp_ar1_annual']):.0f} 年，{LITERATURE['source']}）"
                     + ("，估值的回归是十年量级的。" if hl_a >= 7 else "。"))
        if ecm_in.get("kappa") is not None:
            line += (f"误差修正回归用样本内缺口估出 κ≈{ecm_in['kappa']:.2f}/年（半衰期约 {np.log(2.0) / ecm_in['kappa']:.0f} 年），"
                     "但这个缺口是按全样本拟合出来的，天然会回归，所以偏快")
        if e.get("kappa") is not None:
            line += (f"；只用当时数据（每年 12 月重估公允值）的实时缺口，斜率 {e['slope']:+.3f}（t={e['t']:.1f}），折合 "
                     f"κ≈{e['kappa']:.2f}/年" + (f"（半衰期约 {e['half_life_years']:.0f} 年）" if e.get("half_life_years") else "")
                     + ("，并不显著" if abs(e["t"]) < 2.0 else ""))
        o = pers.get("oos_r2_price_12m") or {}
        if o.get("r2") is not None:
            line += (f"。实时缺口对下一年市场价格收益的样本外 R² 为 {o['r2']:+.2f}（{o['n']} 个月，相对历史均值）"
                     + ("：它并不能预测下一年的收益" if o["r2"] < 0 else "：有一点预测力"))
        txt.append(line + "。")
    best = by_id.get(sel.get("best_candidate") or "")
    head = (f"回测检验（验证窗口选、测试窗口确认一次，{len(SEEDS)} 个随机种子）：{len(cand)} 个估值状态变体中 {n_worse} 个在验证窗口上、"
            f"{n_worse_t} 个在测试窗口上比现行引擎更差（现行引擎验证 MAE {base['validation_mae']:.2%}、测试 {base['test_mae']:.2%}）。")
    if not adopted and not sel.get("eligible") and best:     # the reason would repeat the count: state the rest
        txt.append(head + f"没有一个通过；最接近的是「{best['name']}」，验证 MAE {best['validation_mae']:.2%}"
                   f"（{-best['validation_gain']:+.2%}），测试 {best['test_mae']:.2%}。保留现行引擎，" + pull_off_text())
    else:
        txt.append(head + sel["reason"])
    w = {r["window"]: r for r in reg["by_window"]}
    if not adopted and w:
        def side(r):
            return "偏便宜" if r["gap_start"] < 0 else "偏贵"
        lines = [f"{r['train_end'][:4]} 年底 P/D {r['pd_start']:.0f}、训练期公允值 {r['pd_fair_start']:.0f}（{side(r)}），"
                 f"窗口结束时 P/D {r['pd_end_actual']:.0f}、按当时实际状态算的公允值 {r['pd_fair_end_realized']:.0f}"
                 for r in w.values()]
        n_wider = sum(abs(r["gap_end_realized"]) > abs(r["gap_start"]) for r in w.values())
        first_test = min(str(int(r["train_end"][:4]) + 1) for r in w.values())
        txt.append("估值拉力在回测里的表现：" + "；".join(lines) + f"。{len(w)} 个窗口中有 {n_wider} 个在结束时偏离公允值比开始时更远"
                   + (f"：{first_test} 年以来估值并没有按训练期估计的速度回到公允值，拉力在这些窗口里主要是在加误差。"
                      if n_worse * 2 >= len(cand) and n_wider * 2 >= len(w) else "。"))
    if ipd:
        a, h = ipd["valuation_adjusted"], ipd["historical"]
        vr_e = h.get("variance_ratio")
        vr_h = (pers.get("variance_ratio") or {}) if pers else {}
        pull_on = bool(config.VALUATION.get("enabled"))
        tracked = bool(config.VALUATION.get("track")) and not pull_on
        line = (("反过来看不带拉力的引擎：" if pull_on else
                 "反过来看现行引擎：它不把估值拉向公允值" + ("（κ=0，只记录每条路径的 P/D，回报不变）" if tracked else "") + "，")
                + f"无条件模拟 {years:.0f} 年后美股 P/D 的中位数是 {a['pd_end']['p50']:.0f} 倍"
                f"（5%–95%：{a['pd_end']['p5']:.0f}–{a['pd_end']['p95']:.0f}；未来情景用的估值调整 CMA），用历史漂移时是 "
                f"{h['pd_end']['p50']:.0f} 倍（{h['pd_end']['p5']:.0f}–{h['pd_end']['p95']:.0f}）："
                + ("估值像随机游走一样漂移" if vr_e is None or vr_e > 0.85 else "估值的漂移很大")
                + f"——{years:.0f} 年对数价格收益的标准差 {h['log_price_sd']:.2f}，其中估值变化部分 {h['valuation_part_sd']:.2f}、"
                f"股息增长部分 {h['dividend_part_sd']:.2f}"
                + (f"（两者相关 {h['valuation_dividend_corr']:+.2f}：模拟的股价与股息增长几乎不挂钩，股息多涨的路径上 P/D 相应下降，"
                   "所以估值部分比总和还大）" if h["valuation_part_sd"] > h["log_price_sd"] and h.get("valuation_dividend_corr", 0) < 0 else "")
                + "。")
        if vr_e is not None and vr_h.get("ratio") is not None:
            line += (f"现行引擎 {years:.0f} 年对数价格收益的方差是“月度方差 × {vr_h['months']}”的 {vr_e:.0%}（随机游走 = 100%），"
                     f"历史上（{since} 年以来的重叠窗口，只有约 {vr_h['independent_windows']:.0f} 个独立窗口，误差很大）是 "
                     f"{vr_h['ratio']:.0%}。")
        line += (("这不一定是错（回测并不支持更快的估值回归）" if not adopted else "")
                 + (("；" if not adopted else "") + "每个情景隐含的 P/D 路径都记录在情景详情里（宏观路径的 P/D 图和 15 年后的 P/D 中位数），"
                    "各估值模式下的对比见本页第 8 节。" if tracked or pull_on else
                    ("，" if not adopted else "") + "但每个情景隐含的 P/D 路径应当摆出来（见本页第 8 节）。"))
        txt.append(line)
    k10 = by_id.get("vs_k10")
    if k10 and k10.get("fan_width_ratio_min") is not None and k10["fan_width_ratio_min"] < 1.0:
        checks = sel.get("checks") or {}
        all_fail_gain = bool(checks) and all(not ch.get("gain", False) for ch in checks.values())
        txt.append(f"估值拉力还会让长期分布变窄（κ={k10['settings']['kappa']:.2f} 时{NAMES[k10['forward_fan']['worst_asset']]}的 "
                   f"{years:.0f} 年分布宽度为现行的 {k10['fan_width_ratio_min']:.0%}），让回撤和破产概率显得更确定"
                   + (f"。“宽度收窄不得超过 {FAN_DROP:.0%}”这条护栏会挡住任何估值回归（回归本身就意味着长期方差变小），"
                      "这里的结论不靠它：所有变体在验证 MAE 上已经不达标。" if all_fail_gain else "。"))
    if scen and scen.get("modes"):
        m = scen["modes"]
        pb = scen["baseline_prob_weighted"]

        def d(vid, key="us_equity_cagr_p50"):
            return m[vid]["prob_weighted_delta"][key]
        pd_base = sum(v["prob"] * v["pd_end_p50"] for v in scen["baseline"].values() if v.get("pd_end_p50"))
        line = (f"对 {len(scen['baseline'])} 个未来情景（现行设定：估值调整 CMA）：概率加权的美股中位数年化收益 "
                f"{pb['us_equity_cagr_p50']:.1%}、纳指 {pb['nasdaq_cagr_p50']:.1%}，{years:.0f} 年后 P/D 中位数（概率加权）约 "
                f"{pd_base:.0f} 倍。")
        if "vs_tw_k10" in m:
            line += (f"把其中的常数估值回归换成状态拉力（κ={by_id['vs_tw_k10']['settings']['kappa']:.2f}，其余假设不变，"
                     f"「{by_id['vs_tw_k10']['name']}」）：美股 {d('vs_tw_k10'):+.1%}、纳指 {d('vs_tw_k10', 'nasdaq_cagr_p50'):+.1%}，"
                     f"{years:.0f} 年后 P/D 约 {m['vs_tw_k10']['pd_end_p50_prob_weighted']:.0f} 倍"
                     + ("——状态拉力比现在的常数回归更悲观。" if d("vs_tw_k10") < -0.002 else
                        "——与现在的常数回归相差不大。" if d("vs_tw_k10") <= 0.002 else "——状态拉力比现在的常数回归更乐观。"))
        if "cma_historical" in m and "vs_k10" in m:
            pd_h = m["cma_historical"]["pd_end_p50_prob_weighted"]
            line += (f"完全不做估值假设（全部用历史漂移）：美股 {d('cma_historical'):+.1%}、P/D {'涨' if pd_h > today['pd'] else '降'}到约 "
                     f"{pd_h:.0f} 倍；在历史漂移上加 κ={by_id['vs_k10']['settings']['kappa']:.2f} 的拉力：美股 "
                     f"{d('vs_k10'):+.1%}（这两种模式还去掉了“黄金只跟通胀”的假设，黄金 {d('cma_historical', 'gold_cagr_p50'):+.1%}）。")
        txt.append(line)
    fk = [v for v in cand if v.get("forecast_skill")]
    if fk:
        lo_v = min(fk, key=lambda v: v["forecast_skill"]["all"]["crps_skill"])
        mild = [v for v in fk if v["settings"] and isinstance(v["settings"]["kappa"], (int, float))
                and 0.0 < v["settings"]["kappa"] <= 0.1]
        ks = sorted({float(v["settings"]["kappa"]) for v in mild})
        rng = (min(v["forecast_skill"]["all"]["crps_skill"] for v in mild),
               max(v["forecast_skill"]["all"]["crps_skill"] for v in mild)) if mild else None
        ic_t = [v["forecast_skill"]["test"]["rank_ic_delta"] for v in fk]
        n_orig = (fk[0].get("forecast") or {}).get("all", {}).get("n")
        txt.append("一年期预测（第二标准" + (f"，{n_orig} 个起点" if n_orig else "") + "；CRPS 技能 = 1 − CRPS/现行引擎的 CRPS，"
                   "正 = 更好）：" + (f"κ={'、'.join(f'{k:.2f}' for k in ks)} 的变体 CRPS 技能在 {rng[0]:+.1%} 到 {rng[1]:+.1%} 之间，"
                                     if rng else "")
                   + f"最差的「{lo_v['name']}」{lo_v['forecast_skill']['all']['crps_skill']:+.1%}；{fc.SPLIT[:4]} 年以后起点上的排序 IC "
                   f"变化 {min(ic_t):+.2f} 到 {max(ic_t):+.2f}。"
                   + ("没有一个变体让下一年的 CRPS 改善 1% 以上，一年期预测保持现状。"
                      if max(v["forecast_skill"]["all"]["crps_skill"] for v in fk) < 0.01
                      else "部分变体改善了下一年的预测，" + ("并已采用。" if adopted else "但第一标准（多年期回测）没有通过。")))
    txt.append("所以：一年期预测照常只预测下一年；多年期情景需要给出主题和因子方向，估值假设应在情景里显式写出（例如对 "
               "s_div_yield 的视图或有上限的 asset_drift），本页给出今天的估值缺口作为参考。")
    return txt if paragraphs else "".join(txt)


def _shiller_end() -> str | None:
    """Last month of Shiller's own CAPE in the cached file (read only: this never downloads it)."""
    try:
        from ..data import sources as src
        sh = src.shiller(refresh=False, download=False)
    except Exception:  # noqa: BLE001 - optional source
        return None
    if sh is None or sh["cape"].dropna().empty:
        return None
    return _ym(sh["cape"].last_valid_index())


def in_sample_gap(db, vm: ValuationModel) -> pd.Series:
    P = db.frame
    lpd = np.log(100.0 / P["s_div_yield"])
    fair = vm.fair_log_pd({f: P[VALUATION_COLUMNS[f]].to_numpy(float) for f in vm.features})
    return (lpd - fair).dropna()


def notes(ctx: EngineContext, vm: ValuationModel, reg: dict, roic: dict | None = None,
          shiller_end: str | None = None) -> list[str]:
    """The page's notes, every number computed from the data, the settings or the regression evidence."""
    P = ctx.db.frame
    s = config.VALUATION
    c = reg["coefficients"]
    feats = [k for k in c if k != "const"]
    out = []
    both = P.loc[:shiller_end, ["s_div_yield", "s_cape"]].dropna()
    line = "P/D = CRSP 全市场价格指数 / 过去 12 个月股息（Ken French 数据）"
    if len(both) > 24:
        corr = float(np.corrcoef(np.log(100.0 / both["s_div_yield"]), np.log(both["s_cape"]))[0, 1])
        line += f"；它与席勒 CAPE 在 {_ym(both.index.min())} 至 {_ym(both.index.max())} 的对数相关为 {corr:.2f}"
    cf = reg.get("cape")
    if cf:
        line += ("，把公允值回归的因变量换成 ln CAPE，系数为 " + "、".join(f"{CN.get(k, k)} {cf['coefficients'][k]['beta']:+.3f}"
                                                         for k in feats)
                 + "（P/D 版 " + "、".join(f"{c[k]['beta']:+.3f}" for k in feats) + f"），R² {cf['r2']:.2f}")
    out.append(line + "；P/D 每个月都有、不需要发布时滞，且与引擎里的股息收入一致，所以引擎用 P/D 作为估值状态。")
    out.append(f"通胀预期代理是过去 CPI 通胀（年化的月度变化）的指数加权均值（半衰期 {s['infl_half_life_months']} 个月），"
               "实际收益率 = 10 年期收益率 − 通胀预期代理；模拟路径上用同样的递推从模拟的 CPI 计算，所以历史与情景里的实际利率定义一致。")
    line = (f"利润占比（企业税后利润/GDP）、利润同比、利润减资本开支都按发布月对齐（季度第一个月 + {s['release_lag_months']} 个月"
            "才算已知），没有前视；用的是今天的修订版数据。")
    if roic:
        line += (f"ROIC 代理（利润 / 永续盘存法估算的非住宅资本存量，折旧率 {roic['depreciation']:.0%}）与利润占比（同为季度插值口径）"
                 f"在 {roic['sample'][0]} 至 {roic['sample'][1]} 的相关为 {roic['corr_with_profit_share']:.2f}；把公允值回归里的利润占比"
                 f"换成它，系数 {roic['coef_in_fair_value']:+.3f}（t={roic['t_in_fair_value']:.1f}）"
                 + ("，同样不显著" if abs(roic["t_in_fair_value"]) < 2.0 else "") + "，所以用利润占比代表盈利能力。")
    out.append(line)
    fcf = P["s_fcf_share"].dropna() if "s_fcf_share" in P else pd.Series(dtype=float)
    out.append("自由现金流代理 = 税后利润 − 非住宅固定投资（占 GDP），没有加回折旧"
               + (f"，{_ym(fcf.index.min())} 以来 {float((fcf < 0).mean()):.0%} 的月份为负" if len(fcf) else "")
               + "，所以看它的变化：资本开支上升会把它压低。")
    if shiller_end:
        out.append(f"席勒 CAPE 在他的数据结束（{shiller_end}）之后，盈利按 NIPA 税后利润（按发布月）增长、价格按 CRSP 全市场价格指数"
                   "外推，只作展示与对照。")
    bw = reg.get("by_window") or []
    clipped = [w["train_end"] for w in bw if w.get("kappa_ecm") is not None and w["kappa_ecm"] >= KAPPA_BOUNDS[1]]
    line = (f"公允值回归是水平对水平的回归，变量都很持久：t 值用 Newey-West（{reg.get('hac_lags', VALUATION_HAC_LAGS)} 个月）修正，"
            "仍可能偏高；误差修正回归估计的 κ 用的是样本内缺口，会偏向更快的回归"
            + (f"（训练期截至 {'、'.join(clipped)} 时估出 κ ≥ {KAPPA_BOUNDS[1]:g}/年，被截断为 {KAPPA_BOUNDS[1]:g}）" if clipped else "")
            + "。")
    ps = [(w["train_end"], w["coefficients"]["profit_share"]) for w in bw if "profit_share" in w.get("coefficients", {})]
    if ps:
        line += ("各训练期（都从 " + reg["sample"][0][:4] + " 年开始）里利润占比的系数："
                 + "、".join(f"截至 {te} {x['beta']:+.3f}（t={x['t']:.1f}）" for te, x in ps) + "。")
    out.append(line)
    cma = ctx.cma if ctx.cma.get("method") == "valuation_adjusted" else capital_market_assumptions(
        ctx.db, ctx.anchor, method="valuation_adjusted")
    tail = float(cma.get("rerating_tailwind", 0.0))
    fair = vm.fair_log_pd({f: P[VALUATION_COLUMNS[f]].to_numpy(float) for f in vm.features})
    w = ctx.db.uncond_frequency()
    dfair = float(w[1:] @ np.nan_to_num(np.diff(fair))) * 12.0
    k_ref = float(s["kappa"]) if isinstance(s["kappa"], (int, float)) and s["kappa"] > 0 else 0.1
    hist_ref = next((v["name"] for v in VARIANTS if v["id"] == "cma_historical"), "参照：全部用历史漂移")
    out.append(f"历史漂移里含训练期平均估值抬升 {tail:+.1%}/年，其中公允值本身在样本内的上升只占 {dfair:+.1%}/年；"
               f"在历史漂移上加拉力，P/D 会停在公允值之上约“抬升 / κ”（κ={k_ref:.2f} 时约 {tail / k_ref:+.2f} 对数点），"
               f"而不是公允值本身。估值状态模式的漂移去掉了这部分抬升，但同时带上了行业相对漂移收缩 "
               f"{cma.get('shrink_relative', 0.0):.0%} 和“黄金只跟通胀”。")
    out.append("每个回测窗口和每个预测起点的公允值、κ 和起点状态都只用当时以前的数据重新估计；估值拉力按各资产对大盘的 β 缩放"
               "（与 CMA 相同），黄金、国债、原油、现金不受拉力影响。")
    out.append("一年期预测的对照沿用一年期预测页的设定（相似起点有效数、随机种子）；变体换掉的是估值模式和它所加的漂移"
               f"（历史漂移或估值状态模式的漂移；现行一年期预测用估值调整 CMA），「{hist_ref}」一行给出只换漂移的效果。")
    return out


# ---------------------------------------------------------------- run

def run(ctx: EngineContext, n_paths: int = bt.N_PATHS, seeds=SEEDS, variants: list[dict] | None = None,
        windows: list[dict] | None = None, scenarios: bool = True, scen_dir=None, scen_paths: int = SCEN_PATHS,
        forecast: bool = True, forecast_paths: int = FORECAST_PATHS, verbose: bool = False) -> dict:
    t0 = time.time()
    log = (lambda msg: print(f"[valuation] {msg} ({time.time() - t0:.0f}s)", flush=True)) if verbose else (lambda msg: None)
    panel, spec = ctx.panel, ctx.spec
    vlist = variants or VARIANTS
    wdefs = windows or bt.WINDOWS
    wins = {w["id"]: ab.setup_window(panel, spec, w) for w in wdefs}
    val = tuple(w for w in VALIDATION if w in wins)
    tst = tuple(w for w in TEST if w in wins)
    seeds = [int(s) for s in seeds]
    cache: dict = {}
    raw, same_as = {}, {}
    for v in vlist:
        twin = next((u["id"] for u in vlist if u["id"] in raw and u["cma"] == v["cma"] and u["settings"] == v["settings"]),
                    None)
        if twin:                                         # same backtest engine (e.g. cma_historical = baseline)
            raw[v["id"]], same_as[v["id"]] = raw[twin], twin
            continue
        raw[v["id"]] = [{wid: evaluate(panel, win, v, n_paths, s, cache) for wid, win in wins.items()} for s in seeds]
        log(f"backtest {v['id']}")

    base_fan = forward_fan(ctx, vlist[0])
    dates = fc.forecast_origins(panel)
    try:
        ens = float(jsonio.load(config.RESULTS_DIR / "forecast.json")["analog_ens"])
    except (OSError, KeyError, ValueError, TypeError):
        ens = fc.ANALOG_ENS
    f_scores, f_base, f_repro = {}, None, None
    if forecast:
        rows_base = forecast_rows(panel, spec, dates, vlist[0], ens, forecast_paths)
        f_base = forecast_scores(rows_base)
        f_scores["baseline"] = f_base
        try:
            ref = jsonio.load(config.RESULTS_DIR / "forecast.json")["origins"]
            f_repro = (len(ref) == len(rows_base) and forecast_paths == fc.N_PATHS
                       and all(abs(a["assets"]["us_equity"]["analog"]["p50"] - b["assets"]["us_equity"]["analog"]["p50"]) < 1e-5
                               for a, b in zip(rows_base, ref)))
        except (OSError, KeyError, ValueError):
            f_repro = None
        log("forecast baseline")

    out_variants = []
    for v in vlist:
        per_seed = raw[v["id"]]
        entry = {"id": v["id"], "name": v["name"], "description": v["description"], "adoptable": v["adoptable"],
                 "cma": v["cma"], "settings": v["settings"], "same_backtest_as": same_as.get(v["id"]),
                 "windows": {}}
        for wid in wins:
            m = [s[wid] for s in per_seed]
            entry["windows"][wid] = {
                **{k: float(np.mean([x[k] for x in m])) for k in ("cagr_mae_cond_risk", "cagr_mae_uncond_risk",
                                                                   "coverage_cond_risk", "transmission_abs_annual",
                                                                   "engine_abs_annual", "scenario_abs_annual")},
                "cagr_mae_cond_risk_main_seed": m[0]["cagr_mae_cond_risk"],
                "us_equity": {k: float(np.mean([x["us_equity"][k] for x in m])) for k in ("actual", "cond_p50", "uncond_p50")},
                "pd": {k: (float(np.mean([x["pd"][k] for x in m])) if all(k in x["pd"] for x in m) else None)
                       for k in m[0]["pd"]}}
        for tag, ws in (("validation", val), ("test", tst)):
            mae = _split(per_seed, "cagr_mae_cond_risk", ws)
            diff = mae - _split(raw["baseline"], "cagr_mae_cond_risk", ws)
            entry[f"{tag}_mae"] = float(mae.mean())
            entry[f"{tag}_mae_uncond"] = float(_split(per_seed, "cagr_mae_uncond_risk", ws).mean())
            entry[f"{tag}_coverage"] = float(_split(per_seed, "coverage_cond_risk", ws).mean())
            entry[f"{tag}_gain"] = float(-diff.mean())
            entry[f"{tag}_gain_se"] = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        fan = base_fan if v["id"] == "baseline" else forward_fan(ctx, v)
        ratio = {a: float((fan[a]["p95"] - fan[a]["p5"]) / (base_fan[a]["p95"] - base_fan[a]["p5"])) for a in RISK}
        worst = min(ratio, key=ratio.get)
        entry["forward_fan"] = {"us_equity": fan["us_equity"], "nasdaq": fan["nasdaq"], "width_ratio": ratio,
                                "worst_asset": worst, "pd_end_p50": fan.get("pd_end_p50"),
                                "p50_change": {a: fan[a]["p50"] - base_fan[a]["p50"] for a in RISK}}
        entry["fan_width_ratio_min"] = ratio[worst]
        entry["forecast_crps_skill"], entry["forecast_skill"], entry["forecast"] = None, None, None
        if forecast:
            if v["id"] == "baseline" or (not v["settings"] and v["forecast_cma"] == vlist[0]["forecast_cma"]):
                sc = f_base                                  # the same one-year engine as the baseline
            elif v["adoptable"] or v["id"] in ("cma_historical", "cma_va"):
                sc = forecast_scores(forecast_rows(panel, spec, dates, v, ens, forecast_paths))
                log(f"forecast {v['id']}")
            else:
                sc = None
            if sc is not None:
                f_scores[v["id"]] = sc
                sk = forecast_skill(sc, f_base)
                entry["forecast"], entry["forecast_skill"] = sc, sk
                entry["forecast_crps_skill"] = sk["all"]["crps_skill"]
        out_variants.append(entry)
    by_id = {e["id"]: e for e in out_variants}
    selection = select(out_variants)
    log("selection")

    facs = factor_candidates(panel)
    log("factor candidates")
    ref_settings = {**VS, "kappa": config.VALUATION["kappa"]}
    vm_today = fit_valuation(ctx.db, ctx.anchor, features=ref_settings["features"], kappa=ref_settings["kappa"])
    chosen = by_id.get(selection["chosen"])
    kappa_today = vm_today.kappa
    if chosen and chosen.get("settings"):
        vm_c = fit_valuation(ctx.db, ctx.anchor, features=chosen["settings"]["features"], kappa=chosen["settings"]["kappa"])
        kappa_today = vm_c.kappa
    reg = regression_evidence(ctx, vm_today)
    today = today_block(ctx, vm_today, kappa_today, reg)
    pers = persistence(ctx.db.frame, vm_today.features, gap_in_sample=in_sample_gap(ctx.db, vm_today))
    roic = roic_evidence(ctx, vm_today.features)
    ipd = implied_pd(ctx)

    scen = None
    if scenarios:
        modes = [v for v in vlist if v["id"] in ("baseline", "cma_historical") or v["adoptable"]]
        if scen_dir:
            scen = scenario_effects(ctx, modes, Path(scen_dir), n_paths=scen_paths)
        else:
            with tempfile.TemporaryDirectory(prefix="msl_valuation_scen_") as tmp:
                scen = scenario_effects(ctx, modes, Path(tmp), n_paths=scen_paths)
        for e in out_variants:
            s = scen["modes"].get(e["id"])
            if s:
                e["scenarios"] = {"prob_weighted_delta": s["prob_weighted_delta"],
                                  "max_abs_us_equity_delta": s["max_abs_us_equity_delta"],
                                  "max_scenario": s["max_scenario"], "pd_end_p50_prob_weighted": s["pd_end_p50_prob_weighted"]}
        log("scenarios")

    base_main = raw["baseline"][0]
    reproduces = ab._matches_backtest({w: {k: base_main[w][k] for k in ("cagr_mae_cond_risk", "cagr_mae_uncond_risk",
                                                                         "coverage_cond_risk", "engine_abs_annual",
                                                                         "transmission_abs_annual", "scenario_abs_annual")}
                                       for w in base_main})
    ecm_full = vm_today.ecm
    grid = list(dict.fromkeys(v["settings"]["kappa"] for v in vlist if v["adoptable"] and v["settings"]
                              and v["settings"]["kappa"] != 0.0))
    grid_txt = ", ".join("训练期误差修正回归估计" if k == "ecm" else f"{k:.2f}" for k in grid)
    ert = pers.get("ecm_real_time") or {}
    kappa_block = {"value": kappa_today, "half_life_years": (float(np.log(2.0) / kappa_today) if kappa_today > 0 else None),
                   "how_chosen": (f"κ 的候选 {{{grid_txt}}} 与其他设定一起在验证窗口上比较；"
                                  + (f"选中并采用「{chosen['name']}」。" if selection["adopt"] and chosen else
                                     f"没有变体通过，引擎不使用估值拉力；这里的数字用参考设定 κ={config.VALUATION['kappa']:.2f}/年"
                                     f"（半衰期约 {np.log(2.0) / config.VALUATION['kappa']:.0f} 年）只为说明今天的缺口意味着什么。")
                                  + (f"全样本误差修正回归：{ECM_MONTHS} 个月后 ln P/D 的变化对今天缺口的斜率 {ecm_full['slope']:+.3f}"
                                     f"（t={ecm_full['t']:.1f}），折合 κ≈{ecm_full['kappa']:.2f}/年；它用的是样本内缺口，偏快。"
                                     if ecm_full.get("slope") is not None else "")
                                  + (f"只用当时数据的实时缺口：斜率 {ert['slope']:+.3f}（t={ert['t']:.1f}），κ≈{ert['kappa']:.2f}/年。"
                                     if ert.get("slope") is not None else "")),
                   "grid": grid, "ecm_full_sample": ecm_full,
                   "ecm_by_window": {r["window"]: r["kappa_ecm"] for r in reg["by_window"]},
                   "persistence": pers, "literature": LITERATURE}
    result = {
        "question": QUESTION,
        "text": summary_text(reg, today, selection, by_id, facs, scen, ipd, kap=kappa_block),
        "text_paragraphs": summary_text(reg, today, selection, by_id, facs, scen, ipd, paragraphs=True, kap=kappa_block),
        "data": data_block(ctx, vm_today),
        "regression": reg,
        "kappa": kappa_block,
        "today": today,
        "implied_pd": ipd,
        "roic_proxy": roic,
        "variants": out_variants,
        "selection": selection,
        "factor_candidates": facs["candidates"],
        "factor_walkforward": {k: facs[k] for k in ("final", "iterations", "min_gain")},
        "scenario_effects": scen,
        "settings": {"n_paths": n_paths, "seeds": seeds, "validation": list(val), "test": list(tst),
                     "coverage_drop": COVERAGE_DROP, "min_gain": MIN_GAIN, "fan_drop": FAN_DROP, "fan_paths": FAN_PATHS,
                     "forecast_tol": FORECAST_TOL, "forecast_ic_tol": FORECAST_IC_TOL, "forecast_paths": forecast_paths,
                     "forecast_split": fc.SPLIT, "analog_ens": ens, "n_origins": len(dates),
                     "residuals": config.RESIDUALS, "engine_valuation": config.VALUATION,
                     "baseline_reproduces_backtest": reproduces, "baseline_reproduces_forecast": f_repro,
                     "seconds": round(time.time() - t0, 1)},
        "units": {"mae": "风险资产（股票、REITs、黄金）条件中位数年化收益与实际之差的绝对值平均（小数）",
                  "coverage": "实际累计指数落在条件模拟 5%–95% 区间内的月份占比",
                  "gain": "相对现行引擎的改善（正 = 更好），*_gain_se 为种子间标准误",
                  "fan_width_ratio_min": f"从今天出发、无视图的 {config.HORIZON_MONTHS // 12} 年年化收益 5%–95% 宽度 / 现行引擎的宽度，取风险资产中最小的",
                  "forecast_crps_skill": "一年期预测（相似起点法）风险资产 CRPS 相对现行引擎的技能 = 1 − CRPS/现行 CRPS（正 = 更好）",
                  "gap_log": "ln(P/D) − ln(公允 P/D)，正数 = 比公允值贵",
                  "implied_annual_drag": "−κ × 缺口：拉力在第一年对股价的年化影响（小数）",
                  "drivers": f"各变量相对 {reg['sample'][0]} 以来样本均值、对公允 ln(P/D) 的贡献（对数点）",
                  "variance_ratio": "N 个月对数价格收益的方差 / (N × 月度方差)，1 = 随机游走，< 1 = 长期有回归",
                  "theory_check": "系数符号与理论（实际利率越高估值越低、利润占比越高估值越高）比较；"
                                  "fair_pd_change_per_unit = 该变量 +1 个百分点时公允 P/D 的变化",
                  "oos_r2_price_12m": "实时缺口预测下一年对数价格收益的样本外 R²（相对当时的历史均值；负 = 不如历史均值）",
                  "delta": "R² 变化（样本外，正 = 比基准模型好）"},
        "notes": notes(ctx, vm_today, reg, roic, _shiller_end()),
    }
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Valuation and earnings channel -> data/results/valuation.json")
    ap.add_argument("--paths", type=int, default=bt.N_PATHS, help="simulated paths per window, variant and seed")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--scen-dir", default=None, help="scratch directory for the scenario runs (removed per mode)")
    ap.add_argument("--scen-paths", type=int, default=SCEN_PATHS)
    ap.add_argument("--no-scenarios", action="store_true")
    ap.add_argument("--no-forecast", action="store_true")
    ap.add_argument("--forecast-paths", type=int, default=FORECAST_PATHS)
    args = ap.parse_args(argv)
    ctx = EngineContext.load(selection=True)
    out = run(ctx, n_paths=args.paths, seeds=tuple(args.seeds), scenarios=not args.no_scenarios, scen_dir=args.scen_dir,
              scen_paths=args.scen_paths, forecast=not args.no_forecast, forecast_paths=args.forecast_paths, verbose=True)
    jsonio.dump(out, RESULT_PATH)


if __name__ == "__main__":
    main()
