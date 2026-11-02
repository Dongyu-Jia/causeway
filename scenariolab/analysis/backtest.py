"""Backtests: simulated vs actual.

transmission
    For every asset, the cumulative actual total-return index against the
    factor model fed with the ACTUAL monthly factor changes plus the actual
    income. `fitted` uses the model estimated on the whole sample
    (in-sample); `fitted_oos` refits every 60 months on the months before and
    predicts the next 60 (expanding window). r2 / r2_oos are monthly R^2 of
    price returns with SST around the training mean (the full-sample mean in
    sample, the expanding training mean out of sample); r2_in_sample_oos_months
    is the in-sample model on the out-of-sample months, for a like-for-like gap.
windows
    Holdout periods. The model, the MonthDB, the anchor and the long-run
    reversion levels see only months up to `train_end`: the simulator's r*,
    term premium, spread and unemployment anchors follow the pre-registered
    train-end rule of engine/anchors.py (SIM_DESIGN §5.5), not
    DEFAULT_REVERSION, today's judgement values, which forward scenarios keep.
    Over the test months each asset gets
      actual       realized total-return index
      oracle       the engine's transmission fed the actual test path: training
                   factor model x actual factors + actual income; treasury,
                   crude oil and cash through the simulator's structural rules
      fan_cond     engine simulation whose 24-month phase views equal the
                   realized test-period means of eight variables (VIEW_VARS),
                   weighted exactly as the scenario runner does: five macro
                   ones and three market prices (the 10-year yield, the Baa
                   spread, the dollar), so it answers "had we known the macro
                   path and the financial conditions" (宏观 + 金融条件)
      fan_cond_macro  the same with the five macro views only (MACRO_VIEW_VARS:
                   CPI y/y, IP y/y, the policy rate, oil, unemployment; 只有宏观);
                   what fan_cond gains over it is what the financial
                   conditions, market prices themselves, carry
      fan_uncond   engine simulation with prior weights (no views)
    The fans use CMA "historical" (training drift only); fan_cond and
    fan_uncond are re-run with the "valuation_adjusted" CMA. Phase weights come
    from the runner's own rule (pooling.pool_phase): views are solved on 6-month
    block means, and a view set the training months cannot meet is shrunk
    toward the prior until at least MIN_ENS effective months support it
    (lambda is reported).

Model-implied paths (fitted, fitted_oos, oracle) are conditional-median paths:
compounded systematic return + actual income, less the volatility drag of the
part the model does not explain (sigma^2 / 2 per month, sigma from the
training fit). A noise-free path otherwise compounds 1-3%/yr faster than any
realized or simulated equity path, which would bias every comparison.

Error decomposition of the log terminal value (positive = predicted too high):
    total        = log(uncond p50) - log(actual)
    scenario     = log(uncond p50) - log(cond p50)   what knowing the macro buys
    engine       = log(cond p50)   - log(oracle)     sampling, reweighting, re-anchoring
    transmission = log(oracle)     - log(actual)     what the factor model misses

Custom windows (`custom_window`, `run_custom`; served by POST /api/backtest/run) are the same
run_window on any training end from 15 years after MODEL_START and any test length of 12..180
months that the panel covers, so a new period can be tried without a rebuild. Every window
(built-in or custom) lists the config.EXOGENOUS_SHOCKS its test months overlap.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import weakref
from collections import OrderedDict

import numpy as np
import pandas as pd

from .. import config, jsonio
from ..context import EngineContext
from ..engine import anchors as long_run
from ..engine import summary
from ..engine.ep import weighted_corr
from ..engine.factor_model import STRUCTURAL_ASSETS, AssetFactorModel, ModelSpec
from ..engine.pooling import BLOCK, MIN_ENS, MIN_EPISODES, PhasePool, pool_phase
from ..engine.simulate import (DEFAULT_REVERSION, Anchor, MonthDB, capital_market_assumptions, make_anchor, simulate,
                               valuation_settings, valuation_setup, with_oil_lock)

ASSETS = list(config.ASSET_IDS)
NAMES = {a["id"]: a["name"] for a in config.ASSETS}
RISK_ASSETS = [a for a in ASSETS if a not in STRUCTURAL_ASSETS]      # equities, REITs, gold
CORR_ASSETS = [a for a in ASSETS if a != "cash"]                       # cash "returns" are a slow rate
# fan_cond's views: five macro variables and three market prices (the 10-year yield, the Baa spread, the dollar)
VIEW_VARS = ("s_cpi_yoy", "s_ip_yoy", "f_ff", "f_y10", "f_spread", "f_oil", "f_unrate", "f_usd")
# fan_cond_macro's views: the macro ones only (SIM_DESIGN §10.5, P0.2). fan_cond is partly a market-price oracle
# (critic): what it gains over this arm is what the financial conditions carry.
MACRO_VIEW_VARS = ("s_cpi_yoy", "s_ip_yoy", "f_ff", "f_oil", "f_unrate")
ARM_NAMES = {"cond": "宏观 + 金融条件（含市场价格）", "cond_macro": "只有宏观"}
MACRO = {"cpi_yoy": "s_cpi_yoy", "ff": "lvl_ff", "y10": "lvl_y10", "unrate": "lvl_unrate"}
VARIANTS = {                      # key -> (weights, CMA method)
    "cond": ("cond", "historical"), "uncond": ("prior", "historical"), "cond_macro": ("cond_macro", "historical"),
    "cond_va": ("cond", "valuation_adjusted"), "uncond_va": ("prior", "valuation_adjusted"),
}
# the reversion keys the train-end anchors set (engine/anchors.REVERSION_KEYS), for the reports
ANCHOR_KEYS = tuple(long_run.REVERSION_KEYS.values())
PHASE_MONTHS = 24
N_PATHS = 400
SEED = 11
REFIT_EVERY = 60
MIN_TRAIN_MONTHS = 120
# Display names of the view variables in window texts: s_ = a state level, f_ = a monthly factor change.
VIEW_NAMES = {**{f"s_{x['id']}": x["name"] for x in config.STATES}, **{f"f_{x['id']}": x["name"] for x in config.FACTORS}}
PARTS_CN = {"scenario": "情景部分（不知道宏观）", "engine": "引擎部分（抽样与重加权）",
            "transmission": "传导部分（因子解释不了的）"}

WINDOWS = [
    {"id": "tech_bubble", "name": "科技繁荣与泡沫破裂", "train_end": "1994-12", "test": ("1995-01", "2004-12")},
    {"id": "gfc_qe", "name": "金融危机与QE", "train_end": "2004-12", "test": ("2005-01", "2014-12")},
    {"id": "covid_ai", "name": "疫情、再通胀与AI", "train_end": "2014-12", "test": ("2015-01", None)},
    {"id": "post_covid", "name": "疫情以来", "train_end": "2019-12", "test": ("2020-01", None)},
]
# custom windows (POST /api/backtest/run): bounds of the request
CUSTOM_ID, CUSTOM_NAME = "custom", "自选窗口"
CUSTOM_PATHS = 200                 # default paths per fan: a 15-year window then runs in about 20 s (the entropy
#                                    pooling of the two conditional arms is nearly all of it)
CUSTOM_PATHS_RANGE = (100, 400)
CUSTOM_MONTHS_RANGE = (12, 180)
MIN_TRAIN_MONTHS_CUSTOM = 180      # at least 15 years of training months from config.MODEL_START

UNITS = {
    "index": "累计总回报指数，起点 = 1",
    "cagr": "年化收益率（小数，0.05 = 5%）",
    "coverage": "实际值落在模拟 5%–95% 分位区间内的月份占比",
    "pit": "模拟终值低于实际终值的路径占比（0.5 = 实际正好在中位数）",
    "pit_mad": "|PIT − 0.5| 的平均值；校准良好时约 0.25，越大说明实际终值越常落在模拟分布的尾部",
    "decomposition": "对数终值误差（自然对数点；标“年化”的为每年），正数 = 预测偏高",
    "r2": "月度价格收益的 R² = 1 − SSE/SST。样本内 SST 以全样本均值为基准；样本外 R² 和只看传导（oracle）的 R² "
          "以当时可用的训练期均值为基准（Campbell–Thompson），R² > 0 表示用实际因子变化好过只用历史均值",
    "macro": "CPI 同比、联邦基金利率、10 年期收益率、失业率为百分数（4.5 = 4.5%）",
    "shrink": f"视图在训练期无法同时满足、或支撑太薄（有效月份不到 {MIN_ENS:.0f} 个，或不同的 12 个月历史片段不到 "
              f"{MIN_EPISODES:.0f} 段）时，所有视图按 λ 向先验收缩："
              "使用的视图 = 先验均值 + λ ×（实际均值 − 先验均值），λ = 1 表示原样使用",
    "view_gap_sd": "视图的偏离，以训练期该变量 6 个月均值的标准差为单位：模拟偏离 = 模拟路径（经利率缩放、均值回归后）"
                   "该阶段的平均变化 − 真实均值；熵池化偏离 = 熵池化后抽中月份的期望均值 − 真实均值"
                   "（只来自截断和收缩）；抽中偏离 = 实际抽中月份的均值 − 真实均值（另含蒙特卡洛误差）",
    "anchors": "长期锚：中性实际利率、期限利差（10 年期 − 联邦基金利率）、信用利差、失业率，百分数（利差为百分点）",
}


def _view_list(view_vars) -> str:
    return "、".join(VIEW_NAMES.get(v, v) for v in view_vars)


NOTES = [
    "模型路径（样本内拟合、样本外拟合、只看传导的 oracle）是给定实际因子路径时的中位数路径：系统性收益 + 实际收入，"
    "再扣掉模型解释不了的残差波动造成的复利拖累（每月 σ²/2，σ 取训练期残差）。",
    "窗口回测中，模型、历史月份库和起点都只用训练期数据；条件扇形图的视图是测试期每 24 个月的实际均值，分两档："
    f"「{ARM_NAMES['cond']}」用 {len(VIEW_VARS)} 个变量（其中{_view_list(v for v in VIEW_VARS if v not in MACRO_VIEW_VARS)}"
    f"是市场价格），「{ARM_NAMES['cond_macro']}」只用{_view_list(MACRO_VIEW_VARS)}。"
    "回答的是“如果事先知道这些路径，引擎能不能算对资产”，不是预测它们本身；两档之差是金融条件（市场价格本身）承载的部分。",
    "利率、信用利差和失业率的长期锚按训练期末可知的规则估计（预先登记，不调参）：中性实际利率 = 过去 40 个季度"
    "（联邦基金利率 − 核心 CPI 同比）的均值，期限利差 = 过去 40 个季度（10 年期收益率 − 联邦基金利率）的均值，信用利差 = "
    "过去 40 个季度 Baa − 10 年期利差的均值，失业率 = 过去 80 个季度的均值；前瞻情景仍用今天的判断值（中性实际利率 "
    f"{DEFAULT_REVERSION['r_star']:g}、期限利差 {DEFAULT_REVERSION['term_premium']:g}、信用利差 "
    f"{DEFAULT_REVERSION['spread_level']:g}、失业率 {DEFAULT_REVERSION['unrate_level']:g}）。每个窗口用的数值见窗口的长期锚。",
    "三种扇形图的长期回报假设（CMA）都用训练期的历史漂移；估值调整版另算，结果见“估值调整 CMA 有没有帮助”。",
    "国债、原油、现金在模拟里由收益率、油价、政策利率路径结构化生成，oracle 用同样的规则；"
    "“传导”图里这三类的拟合线是线性因子回归，仅作参考（原油价格收益对油价对数变化是凸的，线性回归的累计线会偏低）。",
    "条件扇形图的阶段权重与情景运行完全相同：视图按 6 个月区块的均值求解熵池化，抽样直接按区块起点的后验概率抽，"
    f"所以抽中月份的均值在期望上等于视图；同一阶段的视图在训练期无法同时满足、或只有不到 {MIN_ENS:.0f} 个有效月份或 "
    f"{MIN_EPISODES:.0f} 段不同的历史片段支撑时，所有视图按 λ 向先验收缩（见阶段表中的 λ）。",
    "相关性比较不含现金（月度现金收益只是缓慢变化的利率，相关系数没有意义）。",
]


def level_view_lag_note(windows: list[dict]) -> str | None:
    """Largest phase-end CPI YoY gap between actual and the conditional median, as a note.

    A view on the phase mean of a trailing 12-month level (CPI YoY) draws months whose YoY was
    already there, so the simulated YoY path trails the actual one by part of a phase.
    """
    best = None
    for w in windows:
        m = w["macro"]["cpi_yoy"]
        for k in np.cumsum(w["phases"]):
            gap = m["actual"][k] - m["cond"]["p50"][k]
            if np.isfinite(gap) and (best is None or abs(gap) > abs(best[0])):
                best = (gap, w["dates"][k], m["actual"][k], m["cond"]["p50"][k])
    if best is None or abs(best[0]) < 1.0:
        return None
    return (f"CPI 同比这类“过去 12 个月”的水平视图，会挑出当时同比已经在目标附近的历史月份，模拟的同比路径因此会比实际滞后："
            f"差距最大的阶段末是 {best[1]}，实际 CPI 同比 {best[2]:.1f}%，条件模拟中位数 {best[3]:.1f}%。")


def _ym(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def _labels(base, months: pd.DatetimeIndex) -> list[str]:
    return [_ym(base)] + [_ym(m) for m in months]


def _avg(values) -> float:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def r2_score(y, yhat, benchmark=None) -> float:
    """1 - SSE/SST over months where both are finite.

    SST is taken around `benchmark` (scalar or per-month array). Out of sample this is the
    training-sample mean, i.e. the no-factor forecast available at the time (Campbell-Thompson
    R^2); with None it is the sample's own mean, which in sample is the same thing.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    b = np.broadcast_to(np.asarray(benchmark if benchmark is not None else np.nan, float), y.shape)
    m = np.isfinite(y) & np.isfinite(yhat) & (np.isfinite(b) if benchmark is not None else True)
    if m.sum() < 3:
        return float("nan")
    ref = b[m] if benchmark is not None else y[m].mean()
    sst = ((y[m] - ref) ** 2).sum()
    return float(1.0 - ((y[m] - yhat[m]) ** 2).sum() / sst) if sst > 0 else float("nan")


def index_path(total: np.ndarray) -> np.ndarray:
    """(T, A) simple returns -> (T+1, A) growth of 1, index 0 = 1."""
    total = np.asarray(total, float).reshape(len(total), -1)
    return np.vstack([np.ones((1, total.shape[1])), np.cumprod(1.0 + total, axis=0)])


def median_path(price: np.ndarray, income: np.ndarray, resid_var) -> np.ndarray:
    """Conditional-median growth of 1 for (T, A) model price returns + income, less sigma^2/2 drag."""
    step = np.log1p(np.maximum(price + income, -0.95)) - 0.5 * np.asarray(resid_var, float)
    return np.exp(np.vstack([np.zeros((1, step.shape[1])), np.cumsum(step, axis=0)]))


def coverage(actual, lo, hi) -> float:
    """Share of months 1..T with lo <= actual <= hi (index 0 is the common start)."""
    a, lo, hi = (np.asarray(x, float)[1:] for x in (actual, lo, hi))
    return float(np.mean((a >= lo) & (a <= hi)))


def pit(value: float, draws) -> float:
    """Probability integral transform: share of simulated draws below the actual value."""
    d = np.asarray(draws, float)
    return float((np.sum(d < value) + 0.5 * np.sum(d == value)) / d.size)


def cagr(terminal, years: float):
    return np.asarray(terminal, float) ** (1.0 / years) - 1.0


def decompose(actual: float, oracle: float, cond_p50: float, uncond_p50: float, years: float) -> dict:
    """Split the log terminal error of the unconditional median into scenario + engine + transmission."""
    la, lo, lc, lu = np.log([actual, oracle, cond_p50, uncond_p50])
    parts = {"total": lu - la, "scenario": lu - lc, "engine": lc - lo, "transmission": lo - la}
    out = {k: float(v) for k, v in parts.items()}
    out.update({f"{k}_annual": float(v / years) for k, v in parts.items()})
    return out


def corr_mae(a, b) -> float:
    """Mean absolute difference over the off-diagonal (upper-triangle) elements."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    iu = np.triu_indices(a.shape[0], 1)
    return float(np.nanmean(np.abs(a[iu] - b[iu])))


def _rounded(m: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 3) for v in row] for row in m]


def predict(model: AssetFactorModel, panel: pd.DataFrame) -> pd.DataFrame:
    """Model price-return predictions; blank structural-shock months (the SVAR leaves out
    the COVID collapse) count as "no shock" so the paths stay defined."""
    cols = [c for c in model.spec.shock_cols if c in panel]
    if cols:
        panel = panel.assign(**{c: panel[c].fillna(0.0) for c in cols})
    return model.predict(panel)


# ---------------------------------------------------------------- transmission

def expanding_predictions(panel: pd.DataFrame, spec: ModelSpec, months: pd.DatetimeIndex,
                          refit_every: int = REFIT_EVERY, min_train: int = MIN_TRAIN_MONTHS
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Out-of-sample price-return predictions, refitting every `refit_every` months on data up to t-1.

    Returns (predictions, residual variance of the fit each month used, the training-sample mean
    price return each month used as the no-factor R^2 benchmark, refit months).
    Months before the first fit that has enough data stay NaN.
    """
    assets = AssetFactorModel(spec).assets
    pred = pd.DataFrame(np.nan, index=months, columns=assets)
    var, bench = pred.copy(), pred.copy()
    refits = []
    for i in range(min_train, len(months), refit_every):
        block = months[i:i + refit_every]
        end = months[i] - pd.DateOffset(months=1)
        try:
            model = AssetFactorModel(spec).fit(panel, config.MODEL_START, end)
        except ValueError:
            continue
        pred.loc[block] = predict(model, panel).loc[block, assets].to_numpy()
        var.loc[block] = np.array([model.stats.resid_sd[a] ** 2 for a in assets])
        bench.loc[block] = panel.loc[config.MODEL_START:end, [f"p_{a}" for a in assets]].mean().to_numpy()
        refits.append(_ym(months[i]))
    return pred, var, bench, refits


def transmission(panel: pd.DataFrame, model: AssetFactorModel) -> dict:
    """Actual vs factor-fitted cumulative indices (in-sample and expanding-window out of sample)."""
    pred_in = predict(model, panel).loc[config.MODEL_START:]
    tot = panel.loc[config.MODEL_START:, [f"r_{a}" for a in ASSETS]]
    months = pred_in.index[pred_in.notna().all(axis=1) & tot.notna().all(axis=1)]
    base = months[0] - pd.DateOffset(months=1)
    pred_oos, var_oos, bench_oos, refits = expanding_predictions(panel, model.spec, months)
    oos_ok = pred_oos.notna().all(axis=1).to_numpy()
    i0 = int(np.argmax(oos_ok)) if oos_ok.any() else None
    series, r2, r2_oos, r2_same = {}, {}, {}, {}
    for a in ASSETS:
        actual = index_path(panel.loc[months, f"r_{a}"].to_numpy())[:, 0]
        if a not in model.assets:                 # cash: income only, nothing to explain
            series[a] = {"actual": actual, "fitted": actual, "fitted_oos": actual}
            r2[a] = r2_oos[a] = r2_same[a] = None
            continue
        inc = panel.loc[months, [f"d_{a}"]].to_numpy()
        price = panel.loc[months, f"p_{a}"].to_numpy()
        fitted = median_path(pred_in.loc[months, [a]].to_numpy(), inc, model.stats.resid_sd[a] ** 2)[:, 0]
        oos = np.full(len(months) + 1, np.nan)
        if i0 is not None:
            g = median_path(pred_oos[[a]].to_numpy()[i0:], inc[i0:], var_oos[[a]].to_numpy()[i0:])[:, 0]
            oos[i0:] = actual[i0] * g
        series[a] = {"actual": actual, "fitted": fitted, "fitted_oos": oos}
        r2[a] = r2_score(price, pred_in.loc[months, a])
        r2_same[a] = r2_score(price[oos_ok], pred_in.loc[months, a].to_numpy()[oos_ok], price.mean())
        r2_oos[a] = r2_score(price[oos_ok], pred_oos[a].to_numpy()[oos_ok], bench_oos[a].to_numpy()[oos_ok])
    risk = [a for a in RISK_ASSETS if a in model.assets]
    return {
        "dates": _labels(base, months), "start": _ym(months[0]), "end": _ym(months[-1]),
        "oos_start": _ym(months[i0]) if i0 is not None else None,
        "refit_every": REFIT_EVERY, "refits": refits, "basis": "conditional_median",
        "r2_benchmark": "training_mean", "structural": list(STRUCTURAL_ASSETS),
        "series": series, "r2": r2, "r2_oos": r2_oos, "r2_in_sample_oos_months": r2_same,
        "r2_risk_avg": _avg(r2[a] for a in risk), "r2_oos_risk_avg": _avg(r2_oos[a] for a in risk),
        "r2_in_sample_oos_months_risk_avg": _avg(r2_same[a] for a in risk),
    }


# ---------------------------------------------------------------- holdout windows

def phase_plan(n_months: int, phase_months: int = PHASE_MONTHS) -> list[int]:
    plan = [phase_months] * (n_months // phase_months)
    if n_months % phase_months:
        plan.append(n_months % phase_months)
    return plan


def realized_views(panel: pd.DataFrame, months: pd.DatetimeIndex, plan: list[int],
                   view_vars=VIEW_VARS) -> list[list[dict]]:
    """Per-phase EP views: the realized means of `view_vars` over that phase's test months."""
    out, start = [], 0
    for m in plan:
        block = panel.loc[months[start:start + m], list(view_vars)].mean()
        out.append([{"var": v, "mean": float(block[v])} for v in view_vars if np.isfinite(block[v])])
        start += m
    return out


def fit_window(panel: pd.DataFrame, spec: ModelSpec, train_end) -> tuple[AssetFactorModel, MonthDB, Anchor]:
    """Model, month database and anchor estimated only on months up to `train_end`."""
    model = AssetFactorModel(spec).fit(panel, config.MODEL_START, train_end)
    db = MonthDB(panel, model, start=config.MODEL_START, end=train_end)
    return model, db, make_anchor(panel, model, date=train_end)


def _ep_report(k: int, pool: PhasePool, db: MonthDB, realized: list[dict]) -> dict:
    """`realized` are the phase's true macro means (the oracle views before clipping or shrinking)."""
    res, w = pool.ep, pool.month_weights
    eras = sorted(db.era_weights(w).items(), key=lambda kv: -kv[1])
    top = np.argsort(-w)[:3]
    return {
        "phase": k + 1, "ens": pool.ens, "ens_ratio": pool.ens / len(db), "kl": res.kl,
        "converged": res.converged, "max_violation": res.max_violation, "block": pool.blocks.block,
        "shrink": pool.shrink, "shrink_reason": pool.shrink_reason, "ens_requested": pool.ens_requested,
        "episodes": pool.episodes, "clipped": [v["var"] for v in res.views if v.get("clipped")],
        "views": [{"var": v["var"], "realized": r["mean"], "target": v.get("target"),
                   "target_used": v.get("target_used"), "prior_mean": v["prior_mean"], "prior_sd": v["prior_sd"],
                   "posterior_mean": v["posterior_mean"]}
                  for v, r in zip(res.views, realized)],
        "era_weights": dict(eras[:4]),
        "top_months": [{"date": db.dates[i].strftime("%Y-%m"), "weight": float(w[i]),
                        "weight_ratio": float(w[i] / pool.prior_month_weights[i])} for i in top],
    }


def view_weights(db: MonthDB, views: list[list[dict]], plan: list[int], min_ens: float = MIN_ENS,
                 unconditional: bool = True) -> dict:
    """Per-phase block-start weights exactly as the scenario runner sets them, with their reports,
    plus the unconditional ones (no views; `unconditional=False` leaves them out) and the overrides the
    runner would add (oil lock)."""
    prior, rare = db.prior(), getattr(db, "rare", None)
    out = {"cond": [], "ep": [], "overrides": with_oil_lock(views, None)}
    if unconditional:
        out.update(prior=[], overrides_prior=[{} for _ in plan])
    for k, (vs, m) in enumerate(zip(views, plan)):
        pool = pool_phase(db.frame, vs, prior, m, BLOCK, 1.0, min_ens, rare=rare)
        out["cond"].append(pool.blocks)
        out["ep"].append(_ep_report(k, pool, db, vs))
        if unconditional:
            out["prior"].append(pool_phase(db.frame, [], prior, m, BLOCK, rare=rare).blocks)
    return out


# what a view variable becomes in the simulated macro paths: (macro key, "diff" | "logdiff" | "level")
SIM_VIEW = {"f_ff": ("ff", "diff"), "f_y10": ("y10", "diff"), "f_spread": ("spread", "diff"),
            "f_unrate": ("unrate", "diff"), "f_oil": ("oil", "logdiff"), "f_usd": ("usd", "logdiff"),
            "s_cpi_yoy": ("cpi_yoy", "level")}


def simulated_view_mean(sim, var: str, start: int, m: int) -> float | None:
    """Phase mean of a view variable along the simulated paths (after re-anchoring), in the view's units."""
    if var not in SIM_VIEW or SIM_VIEW[var][0] not in sim.macro:
        return None
    key, kind = SIM_VIEW[var]
    path = sim.macro[key].astype(float)
    if kind == "level":
        return float(path[:, start + 1:start + m + 1].mean())
    seg = np.log(path) if kind == "logdiff" else path
    return float(np.diff(seg, axis=1)[:, start:start + m].mean())


def add_view_delivery(reports: list[dict], db: MonthDB, sim, plan: list[int]) -> dict:
    """Add, per phase and view, the mean over the months actually drawn and along the simulated paths.

    Gaps are against the realized macro mean in units of the sd of 6-month means: `ep_gap_sd`
    is what the pooled weights deliver in expectation (off only through clipping or shrinking),
    `drawn_gap_sd` what the bootstrap drew, and `sim_gap_sd` what the simulated macro the assets
    see did after level scaling and mean reversion. Returns the mean |gap| of the last two.
    """
    drawn, simulated, start = [], [], 0
    for rep, m in zip(reports, plan):
        block = sim.idx[:, start:start + m]
        for v in rep["views"]:
            x = db.frame[v["var"]].to_numpy(float)
            sd = v["prior_sd"] if v["prior_sd"] > 0 else float("nan")
            v["drawn_mean"] = float(np.nanmean(x[block]))
            v["ep_gap_sd"] = (v["posterior_mean"] - v["realized"]) / sd
            v["drawn_gap_sd"] = (v["drawn_mean"] - v["realized"]) / sd
            drawn.append(abs(v["drawn_gap_sd"]))
            sm = simulated_view_mean(sim, v["var"], start, m)
            v["simulated_mean"] = sm
            v["sim_gap_sd"] = None if sm is None else (sm - v["realized"]) / sd
            if sm is not None:
                simulated.append(abs(v["sim_gap_sd"]))
        rep["max_abs_drawn_gap_sd"] = max((abs(v["drawn_gap_sd"]) for v in rep["views"]), default=0.0)
        start += m
    return {"drawn": _avg(drawn), "simulated": _avg(simulated)}


def oracle_paths(panel: pd.DataFrame, model: AssetFactorModel, anchor: Anchor,
                 months: pd.DatetimeIndex) -> np.ndarray:
    """(T+1, A) the engine's transmission fed the actual test path.

    Factor-model assets: training model x actual factors + actual income (conditional median).
    Treasury and crude oil: the simulator's par-bond and oil-price rules reproduce the actual
    returns exactly. Cash: the simulator's rule, prior-month policy rate less the anchor's ff - tb3.
    """
    out = index_path(panel.loc[months, [f"r_{a}" for a in ASSETS]].to_numpy())
    pred = predict(model, panel).loc[months]
    for j, a in enumerate(ASSETS):
        if a == "cash":
            ff_prev = panel["lvl_ff"].reindex([anchor.date, *months[:-1]]).to_numpy(float)
            out[:, j] = index_path(np.maximum(ff_prev - (anchor.ff - anchor.tb3), 0.0) / 1200.0)[:, 0]
        elif a not in STRUCTURAL_ASSETS:
            out[:, j] = median_path(pred[[a]].to_numpy(), panel.loc[months, [f"d_{a}"]].to_numpy(),
                                    model.stats.resid_sd[a] ** 2)[:, 0]
    return out


def window_shocks(start, end) -> list[dict]:
    """config.EXOGENOUS_SHOCKS that overlap the months start..end (inclusive)."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    return [dict(s) for s in config.EXOGENOUS_SHOCKS
            if pd.Timestamp(s["start"]) <= t1 and pd.Timestamp(s["end"]) >= t0]


def window_setup(panel: pd.DataFrame, spec: ModelSpec, w: dict, macro_vars=None) -> dict:
    """What every run of a window shares whatever its seed and path count: the test months, the phase plan,
    the model, month database and anchor fitted on the training months, the long-run anchors knowable at the
    train end (engine/anchors.py) and the reversion they set, and the pooled phase weights of both conditional
    arms (entropy pooling with its shrink search: nearly all of a window's run time). `macro_vars` are the
    views of the macro-only arm (MACRO_VIEW_VARS)."""
    train_end = pd.Timestamp(w["train_end"])
    months = panel.loc[w["test"][0]:w["test"][1] or panel.index.max()].index
    plan = phase_plan(len(months))
    model, db, anchor = fit_window(panel, spec, train_end)
    lr = long_run.train_end_anchors(panel, train_end)
    macro_vars = tuple(MACRO_VIEW_VARS if macro_vars is None else macro_vars)
    return {"train_end": train_end, "months": months, "plan": plan, "model": model, "db": db, "anchor": anchor,
            "anchors": lr, "reversion": long_run.reversion(lr),
            "vw": view_weights(db, realized_views(panel, months, plan), plan),
            "macro_vars": macro_vars,
            "vw_macro": view_weights(db, realized_views(panel, months, plan, macro_vars), plan, unconditional=False)}


def anchors_report(s: dict, rev: dict, given: bool) -> dict:
    """win["anchors"]: the train-end anchors, the reversion the window simulated with and today's defaults."""
    effective = {**DEFAULT_REVERSION, **rev}
    return {"rule": "given" if given else "train_end", **{k: v for k, v in s["anchors"].items() if k != "rules"},
            "reversion": {k: float(effective[k]) for k in ANCHOR_KEYS},
            "default": {k: float(DEFAULT_REVERSION[k]) for k in ANCHOR_KEYS}}


def run_window(panel: pd.DataFrame, spec: ModelSpec, w: dict, n_paths: int = N_PATHS, seed: int = SEED,
               valuation: dict | None = None, setup: dict | None = None, reversion: dict | None = None) -> dict:
    """One holdout window. `valuation` (engine/simulate.valuation_settings; None = config.VALUATION) turns
    the valuation state on for the main fans (cond, cond_macro, uncond): its CMA and pull, estimated on the
    training months, replace CMA historical there; the *_va fans stay as they are. `setup` (window_setup of
    the same panel, spec and window) skips the fitting and pooling. `reversion`: None simulates with the
    anchors knowable at the train end (setup["reversion"], SIM_DESIGN §5.5); a dict is passed to simulate()
    as it is ({} = DEFAULT_REVERSION, the fixed anchors every window used before)."""
    s = setup or window_setup(panel, spec, w)
    train_end, months, plan = s["train_end"], s["months"], s["plan"]
    model, db, anchor = s["model"], s["db"], s["anchor"]
    rev = dict(s["reversion"] if reversion is None else reversion)
    vw = {**s["vw"], "ep": copy.deepcopy(s["vw"]["ep"])}          # each run adds its own delivery stats
    vwm = {**s["vw_macro"], "ep": copy.deepcopy(s["vw_macro"]["ep"])}
    T = len(months)
    years = T / 12.0
    weights = {"cond": vw["cond"], "prior": vw["prior"], "cond_macro": vwm["cond"]}
    ovs = {"cond": vw["overrides"], "prior": vw["overrides_prior"], "cond_macro": vwm["overrides"]}
    cmas = {m: capital_market_assumptions(db, anchor, method=m) for m in ("historical", "valuation_adjusted")}
    vset = valuation_settings(valuation)
    main = {"cma": cmas["historical"], "valuation": None}
    if vset:
        cma_v, vm = valuation_setup(db, anchor, vset)
        main = {"cma": cma_v, "valuation": vm}
    sims = {key: simulate(db, anchor, weights[wk], plan, n_paths=n_paths, seed=seed, block=BLOCK,
                          overrides=ovs[wk], reversion=rev, **(main if cm == "historical" else {"cma": cmas[cm]}),
                          **config.RESIDUALS)
            for key, (wk, cm) in VARIANTS.items()}
    W = {k: summary.wealth_paths(s.total) for k, s in sims.items()}
    gap_cond = add_view_delivery(vw["ep"], db, sims["cond"], plan)
    gap_macro = add_view_delivery(vwm["ep"], db, sims["cond_macro"], plan)
    actual = index_path(panel.loc[months, [f"r_{a}" for a in ASSETS]].to_numpy())
    oracle = oracle_paths(panel, model, anchor, months)
    pred = predict(model, panel).loc[months]
    train_mean = panel.loc[config.MODEL_START:train_end, [f"p_{a}" for a in ASSETS]].mean()
    labels = _labels(train_end, months)

    assets = {}
    for j, a in enumerate(ASSETS):
        band = {k: np.percentile(v[:, :, j], [5, 95], axis=0) for k, v in W.items()}
        p50 = {k: float(np.median(v[:, -1, j])) for k, v in W.items()}
        fan_va = summary.fan(W["cond_va"][:, :, j])
        entry = {
            "dates": labels, "actual": actual[:, j], "oracle": oracle[:, j],
            "fan_cond": summary.fan(W["cond"][:, :, j]), "fan_uncond": summary.fan(W["uncond"][:, :, j]),
            "fan_cond_macro": summary.fan(W["cond_macro"][:, :, j]),
            "fan_cond_va": {p: fan_va[p] for p in ("p5", "p50", "p95")},
            "cagr_actual": float(cagr(actual[-1, j], years)), "cagr_oracle": float(cagr(oracle[-1, j], years)),
            "r2_oracle": (r2_score(panel.loc[months, f"p_{a}"], pred[a], train_mean[f"p_{a}"])
                          if a in RISK_ASSETS else None),
        }
        for k in VARIANTS:
            entry[f"coverage_{k}"] = coverage(actual[:, j], band[k][0], band[k][1])
            entry[f"pit_terminal_{k}"] = pit(actual[-1, j], W[k][:, -1, j])
            entry[f"cagr_pred_{k}_p50"] = float(cagr(p50[k], years))
        for suffix, (c, u) in {"": ("cond", "uncond"), "_macro": ("cond_macro", "uncond"),
                               "_va": ("cond_va", "uncond_va")}.items():
            entry[f"decomposition{suffix}"] = decompose(actual[-1, j], oracle[-1, j], p50[c], p50[u], years)
        assets[a] = entry

    cols = [ASSETS.index(a) for a in CORR_ASSETS]
    realized = np.corrcoef(panel.loc[months, [f"r_{a}" for a in CORR_ASSETS]].to_numpy(float), rowvar=False)
    posterior = np.corrcoef(sims["cond"].total[:, :, cols].reshape(-1, len(cols)).astype(float), rowvar=False)
    prior = weighted_corr(db.frame[[f"r_{a}" for a in CORR_ASSETS]].to_numpy(float), db.prior())
    corr = {"assets": CORR_ASSETS, "realized": _rounded(realized), "posterior": _rounded(posterior),
            "prior": _rounded(prior), "mae_posterior": corr_mae(posterior, realized),
            "mae_prior": corr_mae(prior, realized)}

    macro = {}
    for key, col in MACRO.items():
        macro[key] = {"actual": panel[col].reindex([train_end, *months]).to_numpy(float),
                      "cond": summary.fan(sims["cond"].macro[key].astype(float)),
                      "cond_macro_p50": np.median(sims["cond_macro"].macro[key].astype(float), axis=0),
                      "uncond_p50": np.median(sims["uncond"].macro[key].astype(float), axis=0)}
    if main["valuation"] is not None:                    # the market P/D the valuation state tracks
        macro["pd"] = {"actual": (100.0 / panel["s_div_yield"]).reindex([train_end, *months]).to_numpy(float),
                       "cond": summary.fan(sims["cond"].macro["pd"].astype(float)),
                       "cond_macro_p50": np.median(sims["cond_macro"].macro["pd"].astype(float), axis=0),
                       "uncond_p50": np.median(sims["uncond"].macro["pd"].astype(float), axis=0)}

    win = {
        "id": w["id"], "name": w["name"], "train": [config.MODEL_START, _ym(train_end)],
        "test": [_ym(months[0]), _ym(months[-1])], "phase_months": PHASE_MONTHS, "phases": plan,
        "n_paths": n_paths, "dates": labels, "assets": assets, "corr": corr, "macro": macro,
        "arms": {"cond": {"name": ARM_NAMES["cond"], "view_vars": list(VIEW_VARS)},
                 "cond_macro": {"name": ARM_NAMES["cond_macro"], "view_vars": list(s["macro_vars"])}},
        "ep": vw["ep"], "ep_macro": vwm["ep"],
        "view_gap_sd": {"cond": gap_cond["drawn"], "cond_simulated": gap_cond["simulated"],
                        "cond_macro": gap_macro["drawn"], "cond_macro_simulated": gap_macro["simulated"]},
        "anchors": anchors_report(s, rev, reversion is not None),
        "model_r2_train": {a: model.stats.r2[a] for a in model.assets},
        "shocks": window_shocks(months[0], months[-1]),
    }
    if main["valuation"] is not None:
        win["valuation"] = {k: v for k, v in main["valuation"].summary().items() if k != "beta"}
    win["summary"] = window_summary(win, years)
    return win


def window_summary(win: dict, years: float) -> dict:
    A = win["assets"]
    risk = RISK_ASSETS
    va_assets = [a for a in risk if a != "gold"]           # equities and REITs; gold reported separately

    def mean_of(key: str, assets: list[str]) -> float:
        return _avg(A[a][key] for a in assets)

    def cagr_mae(key: str, assets: list[str]) -> float:
        return _avg(abs(A[a][key] - A[a]["cagr_actual"]) for a in assets)

    def cond_err(dec: str, a: str) -> float:          # |log(cond p50) - log(actual)| per year
        d = A[a][dec]
        return abs(d["engine"] + d["transmission"]) / years

    parts = ("total", "scenario", "engine", "transmission")
    dec_abs = {p: _avg(abs(A[a]["decomposition"][f"{p}_annual"]) for a in risk) for p in parts}
    dec_mean = {p: _avg(A[a]["decomposition"][f"{p}_annual"] for a in risk) for p in parts}
    dec_macro_abs = {p: _avg(abs(A[a]["decomposition_macro"][f"{p}_annual"]) for a in risk) for p in parts}
    va = {
        "assets": va_assets,
        "coverage_cond_historical": mean_of("coverage_cond", va_assets),
        "coverage_cond_va": mean_of("coverage_cond_va", va_assets),
        "coverage_uncond_historical": mean_of("coverage_uncond", va_assets),
        "coverage_uncond_va": mean_of("coverage_uncond_va", va_assets),
        "terminal_err_cond_historical": _avg(cond_err("decomposition", a) for a in va_assets),
        "terminal_err_cond_va": _avg(cond_err("decomposition_va", a) for a in va_assets),
        "terminal_err_uncond_historical": _avg(abs(A[a]["decomposition"]["total_annual"]) for a in va_assets),
        "terminal_err_uncond_va": _avg(abs(A[a]["decomposition_va"]["total_annual"]) for a in va_assets),
        "cagr_mae_cond_historical": cagr_mae("cagr_pred_cond_p50", va_assets),
        "cagr_mae_cond_va": cagr_mae("cagr_pred_cond_va_p50", va_assets),
        "cagr_mae_uncond_historical": cagr_mae("cagr_pred_uncond_p50", va_assets),
        "cagr_mae_uncond_va": cagr_mae("cagr_pred_uncond_va_p50", va_assets),
        "gold_terminal_err_cond_historical": cond_err("decomposition", "gold"),
        "gold_terminal_err_cond_va": cond_err("decomposition_va", "gold"),
    }
    def shrunk_of(ep: list[dict]) -> list[dict]:
        return [{"phase": p["phase"], "shrink": p["shrink"], "reason": p["shrink_reason"],
                 "ens_requested": p["ens_requested"], "ens": p["ens"]} for p in ep if p["shrink"] < 1.0]

    worst = max(risk, key=lambda a: cond_err("decomposition", a))
    ep_gap = _avg(abs(v["ep_gap_sd"]) for p in win["ep"] for v in p["views"])
    s = {
        "years": years, "n_phases": len(win["phases"]),
        "view_gap_sd_ep": ep_gap, "view_gap_sd_drawn": win["view_gap_sd"]["cond"],
        "view_gap_sd_simulated": win["view_gap_sd"]["cond_simulated"],
        "coverage_cond_avg": mean_of("coverage_cond", ASSETS), "coverage_uncond_avg": mean_of("coverage_uncond", ASSETS),
        "coverage_cond_risk": mean_of("coverage_cond", risk), "coverage_uncond_risk": mean_of("coverage_uncond", risk),
        "pit_cond_mad": _avg(abs(A[a]["pit_terminal_cond"] - 0.5) for a in ASSETS),
        "pit_uncond_mad": _avg(abs(A[a]["pit_terminal_uncond"] - 0.5) for a in ASSETS),
        "cagr_mae_cond": cagr_mae("cagr_pred_cond_p50", ASSETS), "cagr_mae_uncond": cagr_mae("cagr_pred_uncond_p50", ASSETS),
        "cagr_mae_cond_risk": cagr_mae("cagr_pred_cond_p50", risk),
        "cagr_mae_uncond_risk": cagr_mae("cagr_pred_uncond_p50", risk),
        "cagr_mae_oracle_risk": cagr_mae("cagr_oracle", risk),
        "decomposition_mean_abs_annual": dec_abs, "decomposition_mean_annual": dec_mean,
        "corr_mae_posterior": win["corr"]["mae_posterior"], "corr_mae_prior": win["corr"]["mae_prior"],
        "ens_min": min(p["ens"] for p in win["ep"]),
        "clipped_views": sorted({v for p in win["ep"] for v in p["clipped"]}),
        "worst_asset": worst, "valuation_adjusted": va, "shrunk_phases": shrunk_of(win["ep"]),
        # the macro-only arm (fan_cond_macro): the same statistics without the three market-price views
        "coverage_cond_macro_avg": mean_of("coverage_cond_macro", ASSETS),
        "coverage_cond_macro_risk": mean_of("coverage_cond_macro", risk),
        "pit_cond_macro_mad": _avg(abs(A[a]["pit_terminal_cond_macro"] - 0.5) for a in ASSETS),
        "cagr_mae_cond_macro": cagr_mae("cagr_pred_cond_macro_p50", ASSETS),
        "cagr_mae_cond_macro_risk": cagr_mae("cagr_pred_cond_macro_p50", risk),
        "decomposition_macro_mean_abs_annual": dec_macro_abs,
        "view_gap_sd_ep_macro": _avg(abs(v["ep_gap_sd"]) for p in win["ep_macro"] for v in p["views"]),
        "view_gap_sd_drawn_macro": win["view_gap_sd"]["cond_macro"],
        "view_gap_sd_simulated_macro": win["view_gap_sd"]["cond_macro_simulated"],
        "ens_min_macro": min(p["ens"] for p in win["ep_macro"]),
        "clipped_views_macro": sorted({v for p in win["ep_macro"] for v in p["clipped"]}),
        "shrunk_phases_macro": shrunk_of(win["ep_macro"]),
    }
    s["interpretation"] = _window_text(win, s, worst)
    return s


def main_source(dec_abs: dict) -> str:
    """Larger of the two parts of the conditional error. The scenario part is not an error
    source: it is how far knowing the macro moved the median, and it can overshoot."""
    return max(("engine", "transmission"), key=lambda p: dec_abs[p])


def _gain_text(gain: float, what: str) -> str:
    return (f"{what}让误差减少 {gain:.1%}/年" if gain > 0.002 else
            (f"{what}几乎没有改善误差" if gain > -0.002 else f"{what}反而让误差增加 {-gain:.1%}/年"))


def _shrunk_text(phases: list[dict], n_views: int, what: str) -> str:
    ph = "、".join(f"第 {p['phase']} 阶段（λ={p['shrink']:.2f}）" for p in phases)
    return (f"{ph}的 {n_views} 个真实均值在训练期找不到足够多同时满足的历史区块，"
            f"视图按 λ 向训练期均值收缩，这些阶段只部分“知道”{what}。")


def _window_text(win: dict, s: dict, worst: str) -> str:
    A = win["assets"]
    t0, t1 = win["test"]
    full, macro_only = win["arms"]["cond"]["view_vars"], win["arms"]["cond_macro"]["view_vars"]
    market = [v for v in full if v not in macro_only]
    gain = s["cagr_mae_uncond_risk"] - s["cagr_mae_cond_risk"]
    gain_fc = s["cagr_mae_cond_macro_risk"] - s["cagr_mae_cond_risk"]
    dec = s["decomposition_mean_abs_annual"]
    main = main_source(dec)
    wd = A[worst]["decomposition"]
    w_part = max(("engine", "transmission"), key=lambda p: abs(wd[p]))
    txt = [
        f"测试期 {t0} 至 {t1}（{s['years']:.1f} 年，{s['n_phases']} 个阶段）。",
        f"已知每 24 个月的真实宏观与金融条件均值时（{len(full)} 个视图变量"
        + (f"，其中{_view_list(market)}是市场价格" if market else "") + "），"
        f"风险资产（股票、REITs、黄金）年化收益中位数的平均绝对误差为 {s['cagr_mae_cond_risk']:.1%}；"
        f"只知道宏观（{len(macro_only)} 个）时为 {s['cagr_mae_cond_macro_risk']:.1%}；都不知道时为 "
        f"{s['cagr_mae_uncond_risk']:.1%}。" + _gain_text(gain, "知道宏观与金融条件") + "；与只知道宏观相比，"
        + _gain_text(gain_fc, "再加上金融条件") + "。",
        f"实际路径落在 90% 区间内的月份占比：宏观与金融条件 {s['coverage_cond_risk']:.0%}，只有宏观 "
        f"{s['coverage_cond_macro_risk']:.0%}，无条件 {s['coverage_uncond_risk']:.0%}。",
        f"对数终值误差拆分（年化，风险资产绝对值平均）：不知道宏观时 {dec['total']:.1%}；知道宏观与金融条件让中位数平均移动 "
        f"{dec['scenario']:.1%}（情景部分）；知道之后剩下的误差里，引擎部分 {dec['engine']:.1%}、"
        f"传导部分 {dec['transmission']:.1%}，主要来自{PARTS_CN[main]}。",
        f"偏差最大的是{NAMES[worst]}：实际年化 {A[worst]['cagr_actual']:.1%}，条件中位数 "
        f"{A[worst]['cagr_pred_cond_p50']:.1%}，主要来自{PARTS_CN[w_part]}（{wd[w_part + '_annual']:+.1%}/年）。",
        f"被抽中月份的 {len(full)} 个视图变量与真实均值平均相差 {s['view_gap_sd_drawn']:.2f} 个标准差"
        f"（6 个月均值的标准差；截断和收缩造成 {s['view_gap_sd_ep']:.2f}，其余是抽样误差）；"
        f"按今天的水平缩放、加上均值回归之后，模拟路径相差 {s['view_gap_sd_simulated']:.2f}。这些都计入引擎误差。",
    ]
    if s["shrunk_phases"]:
        txt.append(_shrunk_text(s["shrunk_phases"], len(full), "宏观与金融条件"))
    if s["shrunk_phases_macro"]:
        txt.append("只有宏观的一档：" + _shrunk_text(s["shrunk_phases_macro"], len(macro_only), "宏观"))
    lr = win.get("anchors") or {}
    rev, dflt = lr.get("reversion") or {}, lr.get("default") or {}
    if rev and dflt:
        txt.append(("长期锚按训练期末可知的规则" if lr.get("rule") == "train_end" else "长期锚由调用方给定")
                   + f"：中性实际利率 {rev['r_star']:.2f}、期限利差 {rev['term_premium']:.2f}、信用利差 "
                   f"{rev['spread_level']:.2f}、失业率 {rev['unrate_level']:.2f}（前瞻情景用 {dflt['r_star']:g}、"
                   f"{dflt['term_premium']:g}、{dflt['spread_level']:g}、{dflt['unrate_level']:g}）。")
    va = s["valuation_adjusted"]
    h, v = va["terminal_err_cond_historical"], va["terminal_err_cond_va"]
    gh, gv = va["gold_terminal_err_cond_historical"], va["gold_terminal_err_cond_va"]
    txt.append(f"估值调整版 CMA 把股票与 REITs 的条件终值误差从每年 {h:.1%} 变为 {v:.1%}"
               + ("，有帮助" if v < h - 0.001 else ("，没有明显差别" if v <= h + 0.001 else "，反而更差"))
               + (f"；黄金基本不变（{gh:.1%} → {gv:.1%}）。" if abs(gv - gh) <= 0.001 else
                  f"；黄金从 {gh:.1%} 变为 {gv:.1%}。"))
    mp, mpr = s["corr_mae_posterior"], s["corr_mae_prior"]
    txt.append(f"相关矩阵：条件模拟与实际的平均绝对差 {mp:.2f}，训练期历史相关与实际 {mpr:.2f}"
               + ("，情景重加权让相关结构更接近实际。" if mp < mpr - 0.01 else "，情景重加权没有让相关结构明显更接近实际。"))
    if s["clipped_views"]:
        txt.append(f"注意：{'、'.join(VIEW_NAMES.get(v, v) for v in s['clipped_views'])}的实际均值超出训练期历史 2%–98% 分位，视图被截断。")
    return "".join(txt)


def overall_summary(windows: list[dict], trans: dict) -> dict:
    rows = []
    pits, pits_macro = [], []
    for w in windows:
        s = w["summary"]
        d = s["decomposition_mean_abs_annual"]
        dm = s["decomposition_macro_mean_abs_annual"]
        rows.append({"id": w["id"], "name": w["name"], "test": w["test"],
                     "coverage_cond_risk": s["coverage_cond_risk"], "coverage_uncond_risk": s["coverage_uncond_risk"],
                     "cagr_mae_cond_risk": s["cagr_mae_cond_risk"], "cagr_mae_uncond_risk": s["cagr_mae_uncond_risk"],
                     "scenario_abs_annual": d["scenario"], "engine_abs_annual": d["engine"],
                     "transmission_abs_annual": d["transmission"],
                     "terminal_err_cond_historical": s["valuation_adjusted"]["terminal_err_cond_historical"],
                     "terminal_err_cond_va": s["valuation_adjusted"]["terminal_err_cond_va"],
                     "view_gap_sd_drawn": s["view_gap_sd_drawn"], "view_gap_sd_simulated": s["view_gap_sd_simulated"],
                     "coverage_cond_macro_risk": s["coverage_cond_macro_risk"],
                     "cagr_mae_cond_macro_risk": s["cagr_mae_cond_macro_risk"],
                     "scenario_abs_annual_macro": dm["scenario"], "engine_abs_annual_macro": dm["engine"],
                     "view_gap_sd_drawn_macro": s["view_gap_sd_drawn_macro"]})
        pits += [w["assets"][a]["pit_terminal_cond"] for a in RISK_ASSETS]
        pits_macro += [w["assets"][a]["pit_terminal_cond_macro"] for a in RISK_ASSETS]
    agg = {k: _avg(r[k] for r in rows) for k in rows[0] if k not in ("id", "name", "test")}
    tail = float(np.mean([(p < 0.05) or (p > 0.95) for p in pits]))
    tail_macro = float(np.mean([(p < 0.05) or (p > 0.95) for p in pits_macro]))
    text = (f"宏观因子在样本内解释风险资产月度价格波动的 {trans['r2_risk_avg']:.0%}；在 {trans['oos_start']} 以后的同一批月份上，"
            f"样本内模型解释 {trans['r2_in_sample_oos_months_risk_avg']:.0%}，扩展窗口的样本外模型只有 "
            f"{trans['r2_oos_risk_avg']:.0%}（以训练期均值为基准）：模拟 K 线的大部分波动来自重抽样的真实历史残差，"
            f"因子决定的是条件分布的位置。"
            f"{len(rows)} 个回测窗口平均：已知宏观与金融条件时风险资产年化误差 {agg['cagr_mae_cond_risk']:.1%}"
            f"（只知道宏观 {agg['cagr_mae_cond_macro_risk']:.1%}，都不知道 {agg['cagr_mae_uncond_risk']:.1%}），"
            f"90% 区间覆盖率 {agg['coverage_cond_risk']:.0%}（只知道宏观 {agg['coverage_cond_macro_risk']:.0%}，"
            f"无条件 {agg['coverage_uncond_risk']:.0%}），"
            f"{len(pits)} 个终值中落在 5%/95% 尾部之外的比例 {tail:.0%}（只知道宏观 {tail_macro:.0%}；校准良好时约 10%；"
            f"同一窗口内各资产高度相关，这个比例的抽样误差很大）。"
            f"知道宏观与金融条件后剩下的误差中传导部分平均 {agg['transmission_abs_annual']:.1%}/年，引擎部分 "
            f"{agg['engine_abs_annual']:.1%}/年；引擎部分包括抽中月份的视图变量平均偏离真实值 "
            f"{agg['view_gap_sd_drawn']:.2f} 个标准差，缩放和均值回归之后模拟路径偏离 {agg['view_gap_sd_simulated']:.2f}。"
            "各窗口的长期锚（中性利率、期限利差、信用利差、失业率）按训练期末可知的规则估计。")
    return {"windows": rows, **agg, "pit_tail_share_cond": tail, "pit_tail_share_cond_macro": tail_macro,
            "n_pit": len(pits), "text": text}


# ---------------------------------------------------------------- custom windows (POST /api/backtest/run)

MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")


def _integer(value, name: str, lo: int, hi: int, what: str) -> int:
    """An integer request parameter in lo..hi (ints, integral floats or digit strings); ValueError in Chinese."""
    msg = f"{name} 必须是 {lo} 到 {hi} 之间的整数（{what}）"
    if isinstance(value, bool) or value is None:
        raise ValueError(msg)
    if isinstance(value, str):
        value = value.strip()
        if not re.fullmatch(r"[+-]?\d+", value):
            raise ValueError(f"{msg}，收到「{value}」")
        value = int(value)
    if isinstance(value, float):
        if not np.isfinite(value) or value != int(value):
            raise ValueError(f"{msg}，收到 {value:g}")
        value = int(value)
    if not isinstance(value, (int, np.integer)):
        raise ValueError(msg)
    if not lo <= int(value) <= hi:
        raise ValueError(f"{msg}，收到 {int(value)}")
    return int(value)


def custom_window(panel: pd.DataFrame, train_end, months) -> dict:
    """The window of a custom backtest: train on the months up to `train_end`, test the next `months`.

    Raises ValueError with a Chinese message when train_end is not a YYYY-MM month inside the
    panel, leaves fewer than 15 years of training months after config.MODEL_START, or the test
    would run past the last panel month.
    """
    last, first = panel.index.max(), panel.index.min()
    lo_m, hi_m = CUSTOM_MONTHS_RANGE
    m = MONTH_RE.match(str(train_end if train_end is not None else "").strip())
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise ValueError(f"train_end「{train_end}」不是合法的月份：格式应为 YYYY-MM，例如 1999-12")
    ym = (int(m.group(1)), int(m.group(2)))
    # compared as numbers first: a year pandas cannot format (e.g. 0000) must get this message, not an error
    if ym < (first.year, first.month) or ym > (last.year, last.month):
        raise ValueError(f"train_end {m.group(1)}-{m.group(2)} 不在数据范围内（{_ym(first)} 至 {_ym(last)}）")
    end = pd.Timestamp(year=ym[0], month=ym[1], day=1)
    start = pd.Timestamp(config.MODEL_START)
    earliest = start + pd.DateOffset(months=MIN_TRAIN_MONTHS_CUSTOM - 1)
    if end < earliest:
        n = (end.year - start.year) * 12 + end.month - start.month + 1
        raise ValueError(f"训练期至少需要 15 年：模型从 {config.MODEL_START} 开始估计，train_end {_ym(end)} 只有 "
                         f"{max(n, 0)} 个月，最早可选 {_ym(earliest)}")
    months = _integer(months, "months", lo_m, hi_m, "测试期月数")
    test_end = end + pd.DateOffset(months=months)
    if test_end > last:
        room = (last.year - end.year) * 12 + last.month - end.month
        tip = (f"从 {_ym(end)} 起最多还有 {room} 个月" if room >= lo_m
               else f"测试期至少 {lo_m} 个月，train_end 最晚为 {_ym(last - pd.DateOffset(months=lo_m))}")
        raise ValueError(f"测试期会到 {_ym(test_end)}，超出数据的最后一个月 {_ym(last)}：{tip}")
    test = panel.loc[end + pd.DateOffset(months=1):test_end].index
    if len(test) != months:
        raise ValueError(f"数据在 {_ym(end)} 之后不是逐月连续的，无法组成 {months} 个月的测试期")
    span = f"{months // 12} 年" if months % 12 == 0 else f"{months} 个月"
    return {"id": CUSTOM_ID, "name": CUSTOM_NAME, "label": f"{CUSTOM_NAME}：{_ym(end)} 之后 {span}",
            "train_end": _ym(end), "test": (_ym(test[0]), _ym(test[-1])), "months": months}


def custom_request(panel: pd.DataFrame, body: dict) -> dict:
    """Validated keyword arguments of run_custom from a request body (ValueError in Chinese)."""
    allowed = ("train_end", "months", "n_paths", "seed")
    unknown = [str(k) for k in body if k not in allowed]
    if unknown:
        raise ValueError(f"不认识的参数：{'、'.join(unknown)}（可用参数：{'、'.join(allowed)}）")
    missing = [k for k in ("train_end", "months") if body.get(k) in (None, "")]
    if missing:
        raise ValueError(f"缺少参数：{'、'.join(missing)}（train_end = 训练期最后一个月 YYYY-MM，months = 测试期月数）")
    w = custom_window(panel, body["train_end"], body["months"])
    n_paths = _integer(body.get("n_paths", CUSTOM_PATHS), "n_paths", *CUSTOM_PATHS_RANGE, "每组模拟的路径数")
    seed = _integer(body.get("seed", SEED), "seed", 0, 2**31 - 1, "随机种子")
    return {"train_end": w["train_end"], "months": w["months"], "n_paths": n_paths, "seed": seed}


# Fitted and pooled windows of recent custom runs, so another seed or path count for the same window skips the
# entropy pooling: (train_end, months, spec) -> (weak reference to the panel, window_setup). Runs of one window
# that arrive together (several seeds at once) wait for a single setup instead of each pooling it again.
SETUP_CACHE_MAX = 8
_setups: OrderedDict = OrderedDict()
_setups_lock = threading.Lock()
_setup_key_locks: dict = {}


def _cached_setup(panel: pd.DataFrame, spec: ModelSpec, w: dict) -> dict:
    key = (w["train_end"], w["months"], json.dumps(spec.to_dict(), sort_keys=True, default=str))
    with _setups_lock:
        key_lock = _setup_key_locks.setdefault(key, threading.Lock())
    with key_lock:
        with _setups_lock:
            hit = _setups.get(key)
            if hit is not None and hit[0]() is panel:
                _setups.move_to_end(key)
                return hit[1]
        setup = window_setup(panel, spec, w)
        with _setups_lock:
            _setups[key] = (weakref.ref(panel), setup)
            _setups.move_to_end(key)
            while len(_setups) > SETUP_CACHE_MAX:
                old, _ = _setups.popitem(last=False)
                _setup_key_locks.pop(old, None)
    return setup


def run_custom(panel: pd.DataFrame, spec: ModelSpec, train_end: str, months: int, n_paths: int = CUSTOM_PATHS,
               seed: int = SEED) -> dict:
    """One custom holdout window: the same run_window as the built-in windows (fan_cond and fan_cond_macro
    with the realized paths as oracle views, fan_uncond, coverage, PIT, CAGR, decomposition, posterior
    correlations, the train-end anchors), plus `custom` (the request) and `label`. The fitted and pooled
    window is reused across seeds and path counts."""
    w = custom_window(panel, train_end, months)
    win = run_window(panel, spec, w, n_paths=n_paths, seed=seed, setup=_cached_setup(panel, spec, w))
    win["label"] = w["label"]
    win["custom"] = {"train_end": w["train_end"], "months": w["months"], "n_paths": n_paths, "seed": seed,
                     "param": f"{w['train_end']}:{w['months']}"}
    return win


def run(ctx: EngineContext, n_paths: int = N_PATHS, seed: int = SEED, windows: list[dict] | None = None,
        valuation: dict | None = None) -> dict:
    panel = ctx.panel
    trans = transmission(panel, ctx.model)
    wins = [run_window(panel, ctx.spec, w, n_paths=n_paths, seed=seed, valuation=valuation) for w in (windows or WINDOWS)]
    notes = list(NOTES)
    lag = level_view_lag_note(wins)
    if lag:
        notes.append(lag)
    if ctx.spec.shock_cols:
        notes.append("所选模型含结构冲击序列（e_*），它们由全样本识别，窗口回测对这些列存在轻微前视。")
    return {
        "assets": ASSETS, "asset_names": NAMES, "risk_assets": RISK_ASSETS,
        "spec": ctx.spec.to_dict(), "describe": ctx.spec.describe(), "n_paths": n_paths, "seed": seed,
        "view_vars": list(VIEW_VARS), "view_vars_macro": list(MACRO_VIEW_VARS), "arm_names": ARM_NAMES,
        "anchor_rules": dict(long_run.RULES), "default_reversion": {k: DEFAULT_REVERSION[k] for k in ANCHOR_KEYS},
        "units": UNITS, "notes": notes, "shocks": config.EXOGENOUS_SHOCKS,
        "custom": {"paths": CUSTOM_PATHS, "paths_range": list(CUSTOM_PATHS_RANGE),
                   "months_range": list(CUSTOM_MONTHS_RANGE), "min_train_months": MIN_TRAIN_MONTHS_CUSTOM,
                   "earliest_train_end": _ym(pd.Timestamp(config.MODEL_START)
                                             + pd.DateOffset(months=MIN_TRAIN_MONTHS_CUSTOM - 1)),
                   "last_month": _ym(panel.index.max())},
        "transmission": trans, "windows": wins, "summary": overall_summary(wins, trans),
    }


def main() -> None:
    ctx = EngineContext.load(selection=True)
    jsonio.dump(run(ctx), config.RESULTS_DIR / "backtest.json")


if __name__ == "__main__":
    main()
