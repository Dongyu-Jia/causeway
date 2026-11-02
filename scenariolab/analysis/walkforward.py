"""Walk-forward improvement loop: diagnose -> fix -> test on untouched months -> repeat.

Score: mean out-of-sample monthly R^2 of PRICE returns over the non-structural
assets (equities, REITs, gold). Every month is predicted from its ACTUAL factor
changes by a model fitted only on the training months, so the score measures
transmission quality, not forecasting. R^2 = 1 - SSE/SST with SST around the
training-period mean (Campbell-Thompson): a positive score means the factors
beat the no-factor historical-mean forecast available at the time.

Stage A, each iteration
  (a) diagnoses its starting spec on train / validate (an injected `diagnose`
      callable, see diagnostics.py),
  (b) runs greedy forward selection on the VALIDATION score: every candidate
      is tried on top of the current spec and the best is accepted if it adds
      more than MIN_GAIN, at most MAX_ACCEPT changes,
  (c) refits the base, starting and selected specs on train + validate and
      scores them on the untouched TEST months.
The gate is incremental: an iteration's new changes are kept only if its
selection beats the spec the iteration STARTED from on its test months
(iteration 1 starts from the base). The next iteration starts from the kept
spec, and the same gate gives the common spec: the last iteration's selection
if it passed, otherwise its start (REQUIRE_TEST_GAIN). Candidates built from
asset prices (market valuation P/D, Shiller CAPE, household net worth) are
scored for reference but never accepted, and so are the small neural networks
while MLP_ELIGIBLE is False: the betas, the attribution and the eras
decomposition need a linear transmission.

Stage B (dedicated_pass) then tries the pre-registered changes to single
assets' equations on top of the common spec: an extra regressor or a later
first month. In every iteration an option has to raise that asset's R^2 by
more than MIN_GAIN_ASSET on the validation months (the best option is chosen
there) and again on the test months; the final spec takes the changes that
pass in the last iteration. `test_ablation` scores every candidate alone on top
of the base over the test months, after selection, to show which fixes are
robust, and `common_robustness` does the same for the tuning candidates on top
of the common spec; neither feeds back into any choice.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from ..engine.factor_model import (HIGH_INFLATION, MODEL_ASSETS, SHORT_NAMES, STRUCTURAL_ASSETS, AssetFactorModel,
                                   ModelSpec)
from .backtest import r2_score

SCORE_ASSETS = [a for a in MODEL_ASSETS if a not in STRUCTURAL_ASSETS]
MIN_GAIN = 0.003
MIN_GAIN_ASSET = 0.003         # stage B: gain in the one asset's R^2, on validation and again on test
MAX_ACCEPT = 3
MIN_COVERAGE = 0.99            # share of evaluation months a spec must be able to predict
REQUIRE_TEST_GAIN = True       # an iteration's changes must beat its start on the untouched test
# "incremental": the selection must beat the iteration's start (the base in iteration 1, then the spec kept so
# far); "base": the gate before the missing-factor round, against the base (kept to report that protocol)
GATE = "incremental"
MLP_ELIGIBLE = False           # the neural networks are scored for reference only (one line to revert)
SHOCK_COLS = ("e_oil_supply", "e_demand", "e_monetary", "e_financial")
CANDIDATE_FACTORS = ("valuation", "sentiment", "profit_share", "capex_share", "wealth", "white_collar", "wage",
                     "real_y10", "profit_growth", "fcf_share", "cape")
# quarterly series dated at publication (config.RELEASE_LAGS; no look-ahead): they change in release months only
RELEASE_DATED = {"profit_share", "capex_share", "wealth", "labor_share", "profit_growth", "fcf_share"}
QUESTIONS = {
    "missing_factor": "是不是少了一个因子",
    "missing_mechanism": "是不是缺一个传导机制",
    "tuning": "是不是要调参数",
    "neural_net": "是不是要训练一个小型神经网络",
}
ENDOGENOUS = {
    "valuation": "f_valuation 由美股价格本身构造（市场 P/D 的对数变化 ≈ 当月美股价格收益 − 股息增长），"
                 "放进传导模型等于用资产价格解释资产价格",
    "wealth": "家庭净值包含股票市值，按发布月对齐后，它在发布月的变化主要就是它所描述那个季度的股市涨跌："
              "由资产价格构成，情景里也无法独立设定",
    "cape": "席勒 CAPE 的变化 = 指数价格变化 − 10 年平均实际盈利的变化，几乎就是当月股价本身（价格 = 盈利 × 市盈率），"
            "放进传导模型等于用股价解释股价",
}
# months from a quarter's first month to its release, for the endogenous release-dated series (leakage evidence)
RELEASE_QUARTER_LAG = {"wealth": config.RELEASE_LAGS["TNWBSHNO"]}
ITERATIONS = [
    {"iter": 1, "train": ("1973-01", "1994-12"), "validate": ("1995-01", "2004-12"), "test": ("2005-01", "2014-12")},
    {"iter": 2, "train": ("1973-01", "2004-12"), "validate": ("2005-01", "2014-12"), "test": ("2015-01", None)},
]
SPEC_KEYS = ("lags", "regime", "ridge", "window_years", "kind", "hidden")
FACTOR_NAMES = {f["id"]: f["name"] for f in config.FACTORS}
# the core-inflation switch of the term-premium term (config.FACTORS; the same 4.0 as HIGH_INFLATION, a test checks it)
TP_THRESHOLD = next((f["interaction"]["above"] for f in config.FACTORS if f.get("interaction")), HIGH_INFLATION)
MLP_NOTE = "传导模型必须是线性的：界面显示每个因子的 β，情景回报按因子拆分；神经网络只作对照（见缺失因子审计）。"
# Pre-registered by the missing-factor round (FACTORS_DESIGN §2.1): (id, group, hypothesis, change, eligible).
# The report-only ones are the audit's comparisons.
ROUND_CANDIDATES = (
    ("add_gz", "missing_factor", "少了「GZ 信用利差」这个因子", {"add_factors": ["gz"]}, True),
    ("add_tp10_x_core4", "missing_mechanism", f"高通胀时期（上月核心 CPI 同比 > {TP_THRESHOLD:g}%）期限溢价冲击单独传导",
     {"add_factors": ["tp10_x_core4"]}, True),
    ("drop_labor_share", "missing_factor", "劳动收入占比（按发布月）不该是核心因子", {"drop_factors": ["labor_share"]}, True),
    ("add_tp10_x_infl4", "missing_mechanism", f"（对照）期限溢价 × 上月 CPI 同比 > {HIGH_INFLATION:g}%",
     {"add_factors": ["tp10_x_infl4"]}, False),
    ("add_tp10", "missing_factor", "（对照）少了「期限溢价」", {"add_factors": ["tp10"]}, False),
    ("add_ebp", "missing_factor", "（对照）少了「超额债券溢价」", {"add_factors": ["ebp"]}, False),
)
# Stage B, pre-registered: changes to one asset's equation (asset_factors: extra regressors; asset_start: that
# equation's first month). A candidate with `options` picks one of them on validation.
DEDICATED_CANDIDATES = [
    {"id": "energy_metals_oil", "asset": "energy", "eligible": True,
     "hypothesis": "能源股还受基础金属相对油价驱动：金属价格标出油价变化里的需求部分",
     "change": {"asset_factors": {"energy": ["metals_oil"]}},
     "note": "世界银行基础金属价格指数除以原油均价（都是月均值）的对数变化，只进能源股的方程。"},
    {"id": "gold_window", "asset": "gold", "eligible": True,
     "hypothesis": "黄金方程只用 1980 年代以后的月份估计：1973–81 年是布雷顿森林体系之后的重新定价",
     "options": [{"asset_start": {"gold": "1982-01"}}, {"asset_start": {"gold": "1985-01"}}],
     "note": "1973 年金价才开始自由浮动；1973–81 年有官方售金和 1979–80 年的暴涨，用这些月份估计的固定系数会扭曲之后的关系。"
             "两个起始月份预先登记，在验证期选择。",
     "caveat": "这一改动是在同一批十年上发现的，走步检验对它不是独立确认。"},
    {"id": "industrials_usd_em", "asset": "industrials", "eligible": False,
     "hypothesis": "工业股还受美元对新兴市场货币的汇率驱动",
     "change": {"asset_factors": {"industrials": ["usd_em"]}},
     "note": "只作对照：1995–2004 年没有增益；这个指数 2006 年从月均值换成月末值，残差里的关联只在 2005 年以后出现，"
             "需要口径一致的月末新兴市场美元指数。"},
    {"id": "gold_tips10_avg", "asset": "gold", "eligible": False,
     "hypothesis": "黄金还受实际利率（10 年期 TIPS 收益率，月均）驱动",
     "change": {"asset_factors": {"gold": ["tips10_avg"]}},
     "note": "只作对照：TIPS 收益率 2003 年才有数据，不能作为 1973 年以来每个月都要有的传导因子；它也部分由市场决定。"},
    {"id": "gold_mortgage_spread", "asset": "gold", "eligible": False,
     "hypothesis": "黄金还受房贷利差驱动",
     "change": {"asset_factors": {"gold": ["mort_spread"]}},
     "note": "只作对照：系数为负、没有机制；它的大部分变化能由核心因子解释（给定 10 年期收益率后的房贷利率变化）。"},
]
# names of the stage-B regressors that are not config factors (report-only candidates)
DEDICATED_NAMES = {"usd_em": "美元对新兴市场货币变动", "tips10_avg": "10 年期 TIPS 收益率变动（月均）",
                   "mort_spread": "房贷利差变动"}
ROBUSTNESS_IDS = ("ridge_0.01", "ridge_0.05")    # tried on the common spec after selection, for the report


def _ym(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def same_spec(a: ModelSpec, b: ModelSpec) -> bool:
    da, db = a.to_dict(), b.to_dict()
    da.pop("name"), db.pop("name")
    return da == db


@dataclass(frozen=True)
class Candidate:
    id: str
    group: str                 # key of QUESTIONS
    hypothesis: str            # Chinese label shown in the UI
    change: dict = field(default_factory=dict)
    eligible: bool = True
    note: str = ""

    def apply(self, spec: ModelSpec) -> ModelSpec:
        d = spec.to_dict()
        factors = [f for f in d["factors"] if f not in self.change.get("drop_factors", ())]
        factors += [f for f in self.change.get("add_factors", ()) if f not in factors]
        shocks = list(d["shock_cols"]) + [c for c in self.change.get("shock_cols", ()) if c not in d["shock_cols"]]
        d.update({k: v for k, v in self.change.items() if k in SPEC_KEYS})
        d["factors"], d["shock_cols"] = factors, shocks
        return ModelSpec.from_dict(d)

    def is_noop(self, spec: ModelSpec) -> bool:
        """True when applying the change cannot alter the fitted model."""
        if spec.kind == "mlp" and "ridge" in self.change:
            return True                           # the ridge penalty only enters the linear fit
        return same_spec(self.apply(spec), spec)

    def to_dict(self) -> dict:
        return {"id": self.id, "group": self.group, "question": QUESTIONS[self.group],
                "hypothesis": self.hypothesis, "change": self.change, "eligible": self.eligible, "note": self.note}


def leakage_evidence(panel: pd.DataFrame, col: str, train: tuple, quarter_lag: int | None = None) -> dict:
    """Correlation of a candidate series with the market's price return this month, next month and last month.

    With `quarter_lag` (a release-dated quarterly series, e.g. household net worth), also with the return over
    the quarter its release describes (months t - quarter_lag .. t - quarter_lag + 2), over its release months.
    """
    P = panel.loc[train[0]:train[1]]
    x, y = P[col], P["p_us_equity"]
    out = {"same_month": float(x.corr(y)), "next_month": float(x.corr(y.shift(-1))),
           "prev_month": float(x.corr(y.shift(1)))}
    if quarter_lag:
        q = sum(panel["p_us_equity"].shift(k) for k in range(quarter_lag - 2, quarter_lag + 1)).loc[train[0]:train[1]]
        rel = x.notna() & (x != 0)
        out["quarter"] = float(x[rel].corr(q[rel]))
    return out


def _endogenous_note(f: str, ev: dict) -> str:
    if "quarter" in ev:
        lag = RELEASE_QUARTER_LAG[f]
        return (f"{ENDOGENOUS[f]}（训练期的发布月与 {lag - 2}–{lag} 个月前那个季度的美股价格收益之和相关 "
                f"{ev['quarter']:.2f}，与当月 {ev['same_month']:.2f}），只作对照，不参与选择。")
    return (f"{ENDOGENOUS[f]}（训练期与美股当月价格收益相关 {ev['same_month']:.2f}，"
            f"与下月 {ev['next_month']:.2f}），只作对照，不参与选择。")


def _covered(panel: pd.DataFrame, col: str, train: tuple) -> bool:
    return col in panel and panel.loc[train[0]:train[1], col].notna().sum() >= 120


def _round_note(panel: pd.DataFrame, cid: str, train: tuple) -> str:
    """UI note of a pre-registered candidate of the missing-factor round, with this panel's numbers."""
    if cid == "add_gz":
        ev = leakage_evidence(panel, "f_gz", train)
        return ("美联储的 GZ 信用利差（公司债相对同期限国债的平均利差，由二级市场价格算出）。它本身也是债券价格，"
                f"训练期与美股当月价格收益相关 {ev['same_month']:.2f}；美联储每月都可能修订整段历史。")
    if cid == "add_tp10_x_core4":
        n = ""
        if "s_core_yoy" in panel:
            hi = panel["s_core_yoy"].shift(1).loc[train[0]:train[1]] > TP_THRESHOLD
            n = f"（训练期有 {int(hi.sum())} 个月）"
        return (f"纽约联储 ACM 10 年期限溢价的月度变化，只在上月核心 CPI 同比高于 {TP_THRESHOLD:g}% 的月份进入{n}，"
                "其余月份为 0；ACM 是在全样本上估计的模型值。")
    if cid == "drop_labor_share":
        return "季度数据按发布月对齐后，劳动收入占比只在发布月变化；去掉后它仍是情景变量和模拟里的一条路径。"
    if cid == "add_tp10_x_infl4":
        return "与「通胀状态交互」同一个开关（看上月 CPI 同比，不是核心 CPI）的期限溢价项，预先登记为对照，不参与选择。"
    if cid == "add_tp10":
        return "期限溢价本身的变化（不分通胀状态），研究轮已否定，只作对照，不参与选择。"
    if cid == "add_ebp":
        return "超额债券溢价（GZ 利差扣除预期违约风险后的部分，反映风险偏好），研究轮已否定，只作对照，不参与选择。"
    return ""


def build_candidates(panel: pd.DataFrame, train: tuple) -> list[Candidate]:
    out = []
    for f in CANDIDATE_FACTORS:
        col = f"f_{f}"
        if not _covered(panel, col, train):
            continue
        note = ""
        if f in ENDOGENOUS:
            note = _endogenous_note(f, leakage_evidence(panel, col, train, RELEASE_QUARTER_LAG.get(f)))
        elif f in RELEASE_DATED:
            note = "季度数据按通常的发布滞后对齐，只在发布月变化（个别推迟发布的季度会早 1–2 个月出现）。"
        elif f == "real_y10":
            note = (f"10 年期收益率减去过去通胀的指数加权均值（半衰期 {config.VALUATION['infl_half_life_months']} 个月）；"
                    "与名义收益率变化高度重合。")
        out.append(Candidate(f"add_{f}", "missing_factor", f"少了「{FACTOR_NAMES[f]}」这个因子",
                             {"add_factors": [f]}, f not in ENDOGENOUS, note))
    out += [
        Candidate("lags1", "missing_mechanism", "资产对因子的反应有 1 个月滞后", {"lags": 1}),
        Candidate("lags3", "missing_mechanism", "资产对因子的反应延续 1–3 个月", {"lags": 3}),
        Candidate("regime_inflation", "missing_mechanism", "高通胀（上月 CPI 同比 > 4%）和低通胀时期的传导系数不同",
                  {"regime": "inflation"}),
    ]
    shocks = [c for c in SHOCK_COLS if c in panel]
    if len(shocks) == len(SHOCK_COLS) and panel.loc[train[0]:train[1], shocks].notna().all(axis=1).sum() >= 120:
        note = "结构冲击由因果模块在全样本上识别，验证期和测试期对这些列存在轻微前视。"
        out += [
            Candidate("shocks_add", "missing_mechanism", "结构冲击（石油供给、总需求、货币、金融）提供原始因子之外的信息",
                      {"shock_cols": list(SHOCK_COLS)}, True, note),
            Candidate("shocks_replace_oil", "missing_mechanism", "区分供给型和需求型油价变化：用结构冲击替代原始油价",
                      {"shock_cols": list(SHOCK_COLS), "drop_factors": ["oil"]}, True, note),
        ]
    nn_note = "" if MLP_ELIGIBLE else MLP_NOTE
    out += [
        Candidate("ridge_0.01", "tuning", "系数估计噪声大，需要轻度收缩（岭回归 λ=0.01）", {"ridge": 0.01}),
        Candidate("ridge_0.05", "tuning", "系数估计噪声大，需要较强收缩（岭回归 λ=0.05）", {"ridge": 0.05}),
        Candidate("window_15", "tuning", "传导系数随时代变化：只用最近 15 年估计", {"window_years": 15}),
        Candidate("window_25", "tuning", "传导系数随时代变化：只用最近 25 年估计", {"window_years": 25}),
        Candidate("mlp16", "neural_net", "传导是非线性的：小型神经网络（16 个隐藏单元）", {"kind": "mlp", "hidden": 16},
                  MLP_ELIGIBLE, nn_note),
        Candidate("mlp32", "neural_net", "传导是非线性的：小型神经网络（32 个隐藏单元）", {"kind": "mlp", "hidden": 32},
                  MLP_ELIGIBLE, nn_note),
    ]
    for cid, group, hypothesis, change, eligible in ROUND_CANDIDATES:
        if all(_covered(panel, f"f_{f}", train) for f in change.get("add_factors", ())):
            out.append(Candidate(cid, group, hypothesis, change, eligible, _round_note(panel, cid, train)))
    return out


def evaluate(panel: pd.DataFrame, spec: ModelSpec, train: tuple, test: tuple,
             assets: list[str] = SCORE_ASSETS, months: pd.DatetimeIndex | None = None) -> dict:
    """Fit on `train`, then score the price returns of `test` predicted from the actual factors.

    `months` restricts the scoring to those months, so specs can be compared on a common sample
    (structural shocks are blank in the COVID months). `_index` lists the months scored.
    The R^2 benchmark is each asset's mean price return over the whole `train` period, the same
    for every spec (a recent-window spec, or an equation with its own later first month, is not
    scored against its own shorter mean).
    """
    model = AssetFactorModel(spec).fit(panel, train[0], train[1])
    pred = model.predict(panel).loc[test[0]:test[1], assets]
    act = panel.loc[test[0]:test[1], [f"p_{a}" for a in assets]].set_axis(assets, axis=1)
    bench = panel.loc[train[0]:train[1], [f"p_{a}" for a in assets]].mean().set_axis(assets)
    ok = pred.notna().all(axis=1) & act.notna().all(axis=1)
    coverage = float(ok.mean())
    if months is not None:
        ok &= pred.index.isin(months)
    per = {a: r2_score(act.loc[ok, a], pred.loc[ok, a], bench[a]) for a in assets}
    return {"score": float(np.mean(list(per.values()))), "per_asset": per, "coverage": coverage,
            "n_months": int(ok.sum()), "train_r2": float(np.mean([model.stats.r2[a] for a in assets])),
            "fit_sample": [_ym(model.train_start), _ym(model.train_end)], "_index": pred.index[ok]}


def compare(panel: pd.DataFrame, specs: dict[str, ModelSpec], train: tuple, test: tuple) -> dict[str, dict]:
    """Evaluate several specs on the months all of them can predict."""
    res = {k: evaluate(panel, s, train, test) for k, s in specs.items()}
    common = res[next(iter(res))]["_index"]
    for r in res.values():
        common = common.intersection(r["_index"])
    for k, s in specs.items():
        if len(res[k]["_index"]) > len(common):
            res[k] = evaluate(panel, s, train, test, months=common)
    return res


def _trial(rnd: int, c: Candidate, spec: ModelSpec, res: dict, base_score: float, seconds: float) -> dict:
    return {"round": rnd, "candidate": c.id, "group": c.group, "hypothesis": c.hypothesis, "eligible": c.eligible,
            "describe": spec.describe(), "val_score": res["score"], "delta": res["score"] - base_score,
            "train_r2": res["train_r2"], "per_asset": res["per_asset"], "coverage": res["coverage"],
            "accepted": False, "seconds": seconds}


def greedy_select(panel: pd.DataFrame, start: ModelSpec, candidates: list[Candidate], train: tuple,
                  validate: tuple, max_accept: int = MAX_ACCEPT, min_gain: float = MIN_GAIN) -> dict:
    """Forward selection on the validation score; ineligible candidates are scored in round 1 only."""
    current, cur = start, evaluate(panel, start, train, validate)
    accepted, trials = [], []
    path = [{"round": 0, "change": None, "describe": start.describe(), "val_score": cur["score"]}]
    for rnd in range(1, max_accept + 1):
        best = None
        for c in candidates:
            if c.id in accepted or (not c.eligible and rnd > 1) or c.is_noop(current):
                continue
            spec = c.apply(current)
            t0 = time.perf_counter()
            try:
                res = evaluate(panel, spec, train, validate)
            except ValueError as err:                # too few months for this many features
                trials.append({"round": rnd, "candidate": c.id, "error": str(err), "accepted": False})
                continue
            trial = _trial(rnd, c, spec, res, cur["score"], time.perf_counter() - t0)
            trials.append(trial)
            if c.eligible and res["coverage"] >= MIN_COVERAGE and (best is None or res["score"] > best[1]["score"]):
                best = (c, res, spec, trial)
        if best is None or best[1]["score"] - cur["score"] <= min_gain:
            break
        c, res, spec, trial = best
        trial["accepted"] = True
        accepted.append(c.id)
        current, cur = spec, res
        path.append({"round": rnd, "change": c.id, "describe": spec.describe(), "val_score": res["score"]})
    return {"spec": current, "accepted": accepted, "result": cur, "trials": trials, "path": path}


def _period(p: tuple, panel: pd.DataFrame) -> tuple[str, str]:
    return p[0], p[1] or _ym(panel.index.max())


def _test_text(sel: dict, test: dict, period: tuple, cand: dict, start_is_base: bool, k: int,
               start_label: str, vs_start: bool = True) -> str:
    """The test paragraph of iteration k: the selection against the base, and from a carried-over start on
    against that start (the incremental gate; `vs_start=False` is the gate against the base)."""
    s, b = test["selected"]["score"], test["base"]["score"]
    st = b if start_is_base else test["start"]["score"]
    d = s - st if vs_start else s - b
    where = f"在未参与选择的测试期 {period[0]} 至 {period[1]}"
    if not sel["accepted"]:
        head = f"验证期没有任何候选让平均 R² 提高超过 {MIN_GAIN}，保留起始模型；它{where}的平均 R² 为 {s:.3f}"
        return head + ("。" if start_is_base else f"，基准模型 {b:.3f}（{s - b:+.3f}）。")
    names = "、".join(cand[c].hypothesis for c in sel["accepted"])
    head = f"验证期选中：{names}（平均 R² {sel['path'][0]['val_score']:.3f} → {sel['result']['score']:.3f}）。{where}："
    if start_is_base or not vs_start:
        head += f"基准模型 {b:.3f}，选中模型 {s:.3f}，"
        if d > MIN_GAIN:
            return head + f"提升 {d:.3f}，改进经受住了样本外检验。"
        if d > -MIN_GAIN:
            return head + f"差别 {d:+.3f}，改进没有延续到测试期，但也没有变差。"
        return head + f"下降 {-d:.3f}，验证期的改进是过拟合或时代特有的，没有延续到测试期。"
    head += f"起始模型 {st:.3f}，选中模型 {s:.3f}，"
    if d > MIN_GAIN:
        return head + f"比起始模型提升 {d:.3f}（比基准 {s - b:+.3f}）：第 {k} 轮的改动经受住了样本外检验。"
    if d > -MIN_GAIN:
        return head + f"与起始模型相差 {d:+.3f}，没有超过 {MIN_GAIN} 的门槛：第 {k} 轮的改动不保留，保留{start_label}。"
    return head + f"比起始模型下降 {-d:.3f}：第 {k} 轮的改动没有延续到测试期，保留{start_label}。"


def run_iteration(panel: pd.DataFrame, it: dict, start: ModelSpec, base: ModelSpec,
                  candidates: list[Candidate], diagnose: Callable | None = None,
                  start_label: str = "起始模型", gate: str = GATE) -> dict:
    train, validate, test = it["train"], it["validate"], _period(it["test"], panel)
    findings = diagnose(panel, start, train, validate, candidates) if diagnose else []
    base_val = evaluate(panel, base, train, validate)
    sel = greedy_select(panel, start, candidates, train, validate)
    trainval = (train[0], validate[1])
    start_is_base = same_spec(start, base)
    specs = {"base": base, "selected": sel["spec"], **({} if start_is_base else {"start": start})}
    test_res = compare(panel, specs, trainval, test)
    base_all = evaluate(panel, base, trainval, test)
    delta = test_res["selected"]["score"] - test_res["base"]["score"]
    delta_vs_start = test_res["selected"]["score"] - test_res["base" if start_is_base else "start"]["score"]
    if gate == "incremental":
        # what this iteration added must beat the spec it started from on the untouched test (iteration 1
        # starts from the base, so there both deltas are the same); a selection with no new change keeps the
        # start without a test of its own
        improved = not same_spec(sel["spec"], start) and delta_vs_start > MIN_GAIN
    else:
        # against the base: a spec carried over from the last iteration with no new change still counts if it
        # beats the base on the untouched test
        improved = not same_spec(sel["spec"], base) and delta > MIN_GAIN
    cand = {c.id: c for c in candidates}
    ablation = []
    for c in candidates:
        if c.is_noop(base):
            continue
        try:
            v = compare(panel, {"base": base, "c": c.apply(base)}, train, validate)
            r = compare(panel, {"base": base, "c": c.apply(base)}, trainval, test)
        except ValueError:
            continue
        ablation.append({"candidate": c.id, "group": c.group, "eligible": c.eligible,
                         "val_delta": v["c"]["score"] - v["base"]["score"], "test_score": r["c"]["score"],
                         "test_delta": r["c"]["score"] - r["base"]["score"], "test_months": r["c"]["n_months"]})
    return {
        "iter": it["iter"], "train": list(train), "validate": list(validate), "test": list(test),
        "start": {"spec": start.to_dict(), "describe": start.describe(), "is_base": start_is_base,
                  "label": "基准模型" if start_is_base else start_label},
        "baseline": {"spec": base.to_dict(), "describe": base.describe(), "val_score": base_val["score"],
                     "val_per_asset": base_val["per_asset"], "train_r2": base_val["train_r2"]},
        "findings": findings,
        "trials": sel["trials"],
        "selected": {"spec": sel["spec"].to_dict(), "describe": sel["spec"].describe(), "changes": sel["accepted"],
                     "val_score": sel["result"]["score"], "val_per_asset": sel["result"]["per_asset"],
                     "path": sel["path"]},
        "test_result": {
            **{k: {"score": r["score"], "per_asset": r["per_asset"]} for k, r in test_res.items()},
            "n_months": test_res["base"]["n_months"],
            "base_all_months": {"score": base_all["score"], "per_asset": base_all["per_asset"],
                                "n_months": base_all["n_months"]},
            "delta": delta, "delta_vs_start": delta_vs_start, "improved": improved, "gate": gate,
            "text": _test_text(sel, test_res, test, cand, start_is_base, it["iter"], start_label, gate == "incremental"),
        },
        "test_ablation": sorted(ablation, key=lambda r: -r["test_delta"]),
    }


# ---------------------------------------------------------------- stage B: one asset's equation

def with_change(spec: ModelSpec, change: dict) -> ModelSpec:
    """`spec` with a dedicated change merged in: extra regressors of single assets (asset_factors) and / or
    their equations' own first months (asset_start)."""
    d = spec.to_dict()
    extra = {a: list(ids) for a, ids in d["asset_factors"].items()}
    for a, ids in change.get("asset_factors", {}).items():
        extra[a] = extra.get(a, []) + [f for f in ids if f not in extra.get(a, [])]
    return ModelSpec.from_dict({**d, "asset_factors": extra,
                                "asset_start": {**d["asset_start"], **change.get("asset_start", {})}})


def change_columns(change: dict) -> list[str]:
    return [f"f_{f}" for ids in change.get("asset_factors", {}).values() for f in ids]


def change_text(change: dict) -> str:
    """Chinese label of a dedicated change, e.g. 「能源股 +基础金属相对油价变动」, 「黄金从 1982 年起估计」."""
    parts = []
    for a, ids in change.get("asset_factors", {}).items():
        parts.append(f"{SHORT_NAMES.get(a, a)} +" + "、".join(DEDICATED_NAMES.get(f, FACTOR_NAMES.get(f, f)) for f in ids))
    for a, month in change.get("asset_start", {}).items():
        year, mon = str(month).split("-")
        parts.append(f"{SHORT_NAMES.get(a, a)}从 {year} 年{'' if mon == '01' else f' {int(mon)} 月'}起估计")
    return "，".join(parts)


def _option_key(change: dict) -> str:
    return "+".join([*(f for ids in change.get("asset_factors", {}).values() for f in ids),
                     *change.get("asset_start", {}).values()])


def _asset_gain(panel: pd.DataFrame, spec: ModelSpec, other: ModelSpec, asset: str, fit: tuple, score: tuple) -> dict:
    """The asset's R^2 with and without the change, on the months both specs can predict."""
    r = compare(panel, {"spec": spec, "option": other}, fit, score)
    before, after = r["spec"]["per_asset"][asset], r["option"]["per_asset"][asset]
    return {"r2_before": before, "r2_after": after, "gain": after - before,
            "score_gain": r["option"]["score"] - r["spec"]["score"], "n_months": r["option"]["n_months"],
            "coverage": r["option"]["coverage"]}


def _dedicated_rows(panel: pd.DataFrame, spec: ModelSpec, cand: dict, train: tuple, validate: tuple,
                    test: tuple) -> list[dict]:
    """One row per option of a stage-B candidate in one iteration; the best eligible option on validation is
    `chosen` when its gain is above MIN_GAIN_ASSET, and `accepted` when its test gain is above it too."""
    asset, eligible = cand["asset"], bool(cand.get("eligible", True))
    rows = []
    for o in cand.get("options") or [cand["change"]]:
        row = {"id": cand["id"], "asset": asset, "option": _option_key(o), "label": change_text(o), "change": o,
               "eligible": eligible, "evaluated": False, "reason": None, "val": None, "test": None,
               "val_gain": None, "test_gain": None, "chosen": False, "accepted": False}
        missing = [c for c in change_columns(o) if c not in panel]
        if asset not in SCORE_ASSETS:
            row["reason"] = "这个资产不在打分的资产里"
        elif missing:
            row["reason"] = "面板里没有这一序列，走步检验没有重算（研究轮的结果见缺失因子审计）"
        else:
            other = with_change(spec, o)
            try:
                v = _asset_gain(panel, spec, other, asset, train, validate)
                t = _asset_gain(panel, spec, other, asset, (train[0], validate[1]), test)
            except ValueError as err:              # an MLP common spec, or too few months for the equation
                row["reason"] = ("单个资产的方程只适用于线性传导模型" if spec.kind == "mlp" else
                                 "样本月份不足" if "not enough months" in str(err) else str(err))
            else:
                row.update(evaluated=True, val=v, test=t, val_gain=v["gain"], test_gain=t["gain"])
        rows.append(row)
    ok = [r for r in rows if r["evaluated"] and r["eligible"] and r["val"]["coverage"] >= MIN_COVERAGE]
    if ok:
        best = max(ok, key=lambda r: r["val_gain"])          # ties keep the first (pre-registered) option
        if best["val_gain"] > MIN_GAIN_ASSET:
            best["chosen"] = True
            best["accepted"] = best["test_gain"] > MIN_GAIN_ASSET
    return rows


def _gains(r: dict) -> str:
    return f"验证期 {r['val_gain']:+.3f}、测试期 {r['test_gain']:+.3f}"


def _stage_b_text(spec: ModelSpec, cands: list[dict], last: dict | None) -> str:
    """One paragraph on stage B's last iteration (the one that decides), with the caveats of what it adopted."""
    if not last:
        return ""
    head = (f"共同模型（{spec.describe()}）定下之后，逐个尝试只改单个资产方程的改动，按这个资产自己的样本外 R² 判断"
            f"（基准仍是整个训练期的平均收益）：验证期提升超过 {MIN_GAIN_ASSET} 的方案再到测试期检验，最后一轮"
            f"（验证 {last['validate'][0][:4]}–{last['validate'][1][:4]}、测试 {last['test'][0][:4]}–{last['test'][1][:4]}）"
            "两次都通过才进入最终模型。")
    parts, caveats = [], []
    for c in cands:
        rows = [r for r in last["rows"] if r["id"] == c["id"]]
        if not rows:
            continue
        if not rows[0]["eligible"]:
            done = [r for r in rows if r["evaluated"]]
            parts.append(f"{rows[0]['label']}：只作对照" + (f"（{_gains(done[0])}）" if done else "，面板里没有这一序列"))
            continue
        main = next((r for r in rows if r["chosen"]), None)
        if main is None:
            best = max((r for r in rows if r["evaluated"]), key=lambda r: r["val_gain"], default=None)
            parts.append(f"{rows[0]['label']}：没有通过" + (f"（{_gains(best)}）" if best else "，没有评估"))
            continue
        txt = f"{main['label']}：{_gains(main)}，{'采用' if main['accepted'] else '测试期没有通过'}"
        others = [r for r in rows if r is not main and r["evaluated"]]
        if others:
            txt += "（" + "；".join(f"{r['label']}：{_gains(r)}，没选" for r in others) + "）"
        parts.append(txt)
        if main["accepted"] and c.get("caveat"):
            caveats.append(f"{SHORT_NAMES.get(c['asset'], c['asset'])}：{c['caveat']}")
    return head + "；".join(parts) + ("。" if parts else "") + "".join(caveats)


def dedicated_pass(panel: pd.DataFrame, spec: ModelSpec, its: list[dict], candidates: list[dict] | None = None) -> dict:
    """Stage B: the pre-registered changes to single assets' equations, on top of the common spec.

    For each iteration: fit `spec` and `spec` + each option on train and score the ASSET's R^2 on validate
    (the benchmark stays the asset's mean price return over the whole training window, so a later first month
    is not scored against its own shorter mean); an eligible candidate's best option is chosen when it gains
    more than MIN_GAIN_ASSET there, and kept when, refitted on train + validate, it gains more than
    MIN_GAIN_ASSET on the test months as well. Every option is scored on the test months for the report.
    A change is in the final spec when it passes in the LAST iteration; the earlier ones are reported. The
    test decades are the ones stage A used, so they took part in the choice.
    """
    cands = DEDICATED_CANDIDATES if candidates is None else candidates
    out = []
    for it in its:
        train, validate, test = it["train"], it["validate"], _period(it["test"], panel)
        rows = [r for c in cands for r in _dedicated_rows(panel, spec, c, train, validate, test)]
        out.append({"iter": it["iter"], "train": list(train), "validate": list(validate), "test": list(test),
                    "rows": rows})
    last = out[-1] if out else None
    accepted = [r for r in (last["rows"] if last else []) if r["accepted"]]
    change: dict = {"asset_factors": {}, "asset_start": {}}
    for r in accepted:
        for a, ids in r["change"].get("asset_factors", {}).items():
            change["asset_factors"].setdefault(a, []).extend(ids)
        change["asset_start"].update(r["change"].get("asset_start", {}))
    return {"candidates": [{**c, "eligible": bool(c.get("eligible", True))} for c in cands],
            "common": {"spec": spec.to_dict(), "describe": spec.describe()},
            "min_gain_asset": MIN_GAIN_ASSET, "min_coverage": MIN_COVERAGE, "r2_benchmark": "training_mean",
            "final_rule": "last_iteration", "test_used_for_choice": True,
            "iterations": out, "accepted": [r["id"] for r in accepted],
            "accepted_options": {r["id"]: r["option"] for r in accepted}, "change": change,
            "text": _stage_b_text(spec, cands, last)}


# ---------------------------------------------------------------- reports on the common spec

def oos_windows(its: list[dict], panel: pd.DataFrame) -> list[tuple[str, tuple, tuple]]:
    """(label, fit window, scored window) of each out-of-sample decade: the first validation, then every test
    (iteration k's test is iteration k+1's validation, fitted on the same months)."""
    first = its[0]
    out = [(f"{first['validate'][0][:4]}–{first['validate'][1][:4]}", tuple(first["train"]), tuple(first["validate"]))]
    for it in its:
        test = _period(it["test"], panel)
        out.append((f"{test[0][:4]}–{test[1][:4]}", (it["train"][0], it["validate"][1]), test))
    return out


def robustness(panel: pd.DataFrame, spec: ModelSpec, candidates: list[Candidate], its: list[dict]) -> dict:
    """Each candidate alone on top of `spec` (the common spec), per out-of-sample decade: the change of the
    score on the months both can predict. Reported after every choice is made; never feeds back."""
    rows = []
    for c in candidates:
        if c.is_noop(spec):
            continue
        other, periods = c.apply(spec), []
        try:
            for label, fit, score in oos_windows(its, panel):
                r = compare(panel, {"spec": spec, "c": other}, fit, score)
                periods.append({"period": label, "delta": r["c"]["score"] - r["spec"]["score"],
                                "n_months": r["c"]["n_months"]})
        except ValueError:
            continue
        rows.append({"candidate": c.id, "group": c.group, "periods": periods})
    return {"spec": spec.to_dict(), "describe": spec.describe(), "rows": rows}


def _final_reason(last: dict, kept: str, start_label: str) -> str:
    tr = last["test_result"]
    n, sel = tr["n_months"], tr["selected"]["score"]
    if kept == "selected" and not tr["improved"]:
        return "按协议采用最后一轮选中的模型，测试期只用于报告。"
    if kept == "selected":
        vs_base = last["start"]["is_base"] or tr["gate"] != "incremental"
        ref = "基准" if vs_base else "本轮起始模型"
        ref_score = tr["base"]["score"] if vs_base else tr["start"]["score"]
        vs = (f"（同一批 {n} 个月上 {sel:.3f} 对 {ref_score:.3f}"
              + ("" if vs_base else f"，比基准 {tr['delta']:+.3f}") + "）")
        return f"最后一轮选中的模型在未参与选择的测试期胜过{ref}{vs}，采用它。"
    if kept == "base":
        return (f"最后一轮选中的模型（{last['selected']['describe']}）在未参与选择的测试期没有胜过基准"
                f"（同一批 {n} 个月上 {sel:.3f} 对 {tr['base']['score']:.3f}），所以保留基准模型。")
    return (f"第 {last['iter']} 轮选中的模型（{last['selected']['describe']}）在未参与选择的测试期没有胜过本轮起始模型"
            f"（同一批 {n} 个月上 {sel:.3f} 对 {tr['start']['score']:.3f}，比起始模型 {tr['delta_vs_start']:+.3f}；"
            f"比基准 {tr['delta']:+.3f}），所以保留起始模型，即{start_label}（{last['start']['describe']}）。")


def run_walkforward(panel: pd.DataFrame, base: ModelSpec | None = None, iterations: list[dict] | None = None,
                    diagnose: Callable | None = None, dedicated: list[dict] | None = None, gate: str = GATE,
                    mlp_eligible: bool | None = None) -> dict:
    """Stage A (the iterations), then stage B on the common spec; `dedicated=[]` turns stage B off.

    `gate` and `mlp_eligible` (None = MLP_ELIGIBLE) rerun the protocol the audit compares this one with:
    the gate against the base, or the neural networks eligible.
    """
    base = base or ModelSpec()
    its = iterations or ITERATIONS
    mlp = MLP_ELIGIBLE if mlp_eligible is None else bool(mlp_eligible)
    start, changes, label = base, [], "基准模型"
    out, cands = [], []
    for it in its:
        cands = build_candidates(panel, it["train"])
        if mlp != MLP_ELIGIBLE:
            cands = [replace(c, eligible=mlp, note="" if mlp else MLP_NOTE) if c.group == "neural_net" else c
                     for c in cands]
        res = run_iteration(panel, it, start, base, cands, diagnose, label, gate)
        selected = ModelSpec.from_dict(res["selected"]["spec"])
        chain = changes + res["selected"]["changes"]
        res["selected"]["chain"] = chain
        res["start"]["chain"] = changes
        if res["test_result"]["improved"]:
            start, changes, label = selected, chain, f"第 {it['iter']} 轮的模型"
            res["carry_over"] = ("选中模型在测试期胜过基准，下一轮从它出发。"
                                 if res["start"]["is_base"] or gate != "incremental"
                                 else "选中模型在测试期胜过本轮起始模型，下一轮从它出发。")
        elif res["start"]["is_base"] or gate != "incremental":
            start, changes, label = base, [], "基准模型"
            res["carry_over"] = "选中模型没有在测试期胜过基准，下一轮重新从基准模型出发。"
        else:
            res["carry_over"] = f"选中模型没有在测试期胜过本轮起始模型，下一轮仍从{label}出发。"
        out.append(res)
    last = out[-1]
    tr = last["test_result"]
    literal = {"spec": last["selected"]["spec"], "describe": last["selected"]["describe"],
               "changes": last["selected"]["chain"], "val_score": last["selected"]["val_score"],
               "test_score": tr["selected"]["score"], "delta": tr["delta"], "delta_vs_start": tr["delta_vs_start"],
               "improved": tr["improved"]}
    if tr["improved"] or not REQUIRE_TEST_GAIN:
        common, common_changes, kept = ModelSpec.from_dict(last["selected"]["spec"]), last["selected"]["chain"], "selected"
    else:                    # the spec the loop carries on with: the last start, or the base under the old gate
        common, common_changes = start, changes
        kept = "start" if gate == "incremental" and not last["start"]["is_base"] else "base"
    reason = _final_reason(last, kept, label)

    stage_b = dedicated_pass(panel, common, its, dedicated)
    changes = common_changes + stage_b["accepted"]
    final = with_change(common, stage_b["change"]) if stage_b["accepted"] else common
    final = ModelSpec.from_dict({**final.to_dict(), "name": "+".join(["base", *changes])})
    if stage_b["accepted"]:
        reason += ("专门改动阶段（只改单个资产的方程）加入"
                   + "、".join(r["label"] for r in stage_b["iterations"][-1]["rows"] if r["accepted"])
                   + f"：它们在最后一轮的验证期和测试期都让各自资产的 R² 提高超过 {MIN_GAIN_ASSET}。")
    elif any(c.get("eligible", True) for c in stage_b["candidates"]):
        reason += "专门改动阶段（只改单个资产的方程）没有一项在最后一轮的验证期和测试期都通过。"
    if REQUIRE_TEST_GAIN:
        reason += ("测试期因此参与了这一次“采用 / 不采用”的决定，之后没有再留出的数据；"
                   + ("基准模型是事先固定的，所以它的测试期 R² 仍是无偏的样本外估计。" if not changes else
                      "选中模型的测试期 R² 是在通过这道关卡的条件下观察到的，会略偏乐观。"))
    train, validate = tuple(last["train"]), tuple(last["validate"])
    trainval = (train[0], validate[1])
    val = evaluate(panel, final, train, validate)["score"]
    own = evaluate(panel, final, trainval, tuple(last["test"]))            # every month the final spec can predict
    tst = own["score"]
    cand = {c.id: c for c in cands}
    return {
        "candidates": [c.to_dict() for c in cands], "score_assets": SCORE_ASSETS,
        "min_gain": MIN_GAIN, "max_accept": MAX_ACCEPT, "require_test_gain": REQUIRE_TEST_GAIN,
        "gate": gate, "mlp_eligible": mlp, "min_gain_asset": MIN_GAIN_ASSET,
        "r2_benchmark": "training_mean",
        "iterations": out,
        "stage_b": stage_b,
        "common_robustness": robustness(panel, common, [cand[i] for i in ROBUSTNESS_IDS if i in cand], its),
        "final": {"spec": final.to_dict(), "describe": final.describe(), "changes": changes, "is_base": not changes,
                  "test_used_for_choice": REQUIRE_TEST_GAIN,
                  "val_score": val, "test_score": tst, "test_months": own["n_months"],
                  "test_per_asset": own["per_asset"], "base_val_score": last["baseline"]["val_score"],
                  "base_test_score": tst if not changes else evaluate(panel, base, trainval, tuple(last["test"]))["score"],
                  "reason": reason,
                  "common": {"spec": common.to_dict(), "describe": common.describe(), "changes": common_changes,
                             "kept": kept},
                  "dedicated": stage_b["accepted"],
                  "last_selected": literal,
                  "selected_on": {"train": last["train"], "validate": last["validate"], "test": last["test"]}},
    }
