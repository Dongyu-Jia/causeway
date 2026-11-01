"""Walk-forward improvement loop: diagnose -> fix -> test on untouched months -> repeat.

Score: mean out-of-sample monthly R^2 of PRICE returns over the non-structural
assets (equities, REITs, gold). Every month is predicted from its ACTUAL factor
changes by a model fitted only on the training months, so the score measures
transmission quality, not forecasting. R^2 = 1 - SSE/SST with SST around the
training-period mean (Campbell-Thompson): a positive score means the factors
beat the no-factor historical-mean forecast available at the time.

Each iteration
  (a) diagnoses its starting spec on train / validate (an injected `diagnose`
      callable, see diagnostics.py),
  (b) runs greedy forward selection on the VALIDATION score: every candidate
      is tried on top of the current spec and the best is accepted if it adds
      more than MIN_GAIN, at most MAX_ACCEPT changes,
  (c) refits the base, starting and selected specs on train + validate and
      scores them on the untouched TEST months.
The next iteration starts from the selected spec only if it beat the base on
its test months, and the same gate decides the final spec: the last
iteration's selection if it beat the base on the untouched test, otherwise the
base (REQUIRE_TEST_GAIN). Candidates built from asset prices (market
valuation P/D, Shiller CAPE) or that leak later quarters into the month
(household wealth) are scored for reference but never accepted. `test_ablation` scores every
candidate alone on top of the base over the test months, after selection, to
show which fixes are robust; it never feeds back into any choice.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from ..engine.factor_model import MODEL_ASSETS, STRUCTURAL_ASSETS, AssetFactorModel, ModelSpec
from .backtest import r2_score

SCORE_ASSETS = [a for a in MODEL_ASSETS if a not in STRUCTURAL_ASSETS]
MIN_GAIN = 0.003
MAX_ACCEPT = 3
MIN_COVERAGE = 0.99            # share of evaluation months a spec must be able to predict
REQUIRE_TEST_GAIN = True       # final spec must beat the base on the last untouched test
SHOCK_COLS = ("e_oil_supply", "e_demand", "e_monetary", "e_financial")
CANDIDATE_FACTORS = ("valuation", "sentiment", "profit_share", "capex_share", "wealth", "white_collar", "wage",
                     "real_y10", "profit_growth", "fcf_share", "cape")
QUARTERLY = {"profit_share", "capex_share", "wealth", "labor_share"}     # interpolated from quarterly data
RELEASE_DATED = {"profit_growth", "fcf_share"}                           # quarterly, dated at publication (no look-ahead)
QUESTIONS = {
    "missing_factor": "是不是少了一个因子",
    "missing_mechanism": "是不是缺一个传导机制",
    "tuning": "是不是要调参数",
    "neural_net": "是不是要训练一个小型神经网络",
}
ENDOGENOUS = {
    "valuation": "f_valuation 由美股价格本身构造（市场 P/D 的对数变化 ≈ 当月美股价格收益 − 股息增长），"
                 "放进传导模型等于用资产价格解释资产价格",
    "wealth": "家庭净值包含股票市值，且季度数据放在季中月份线性插值，会把下一季度的股市涨跌提前带进当月",
    "cape": "席勒 CAPE 的变化 = 指数价格变化 − 10 年平均实际盈利的变化，几乎就是当月股价本身（价格 = 盈利 × 市盈率），"
            "放进传导模型等于用股价解释股价",
}
ITERATIONS = [
    {"iter": 1, "train": ("1973-01", "1994-12"), "validate": ("1995-01", "2004-12"), "test": ("2005-01", "2014-12")},
    {"iter": 2, "train": ("1973-01", "2004-12"), "validate": ("2005-01", "2014-12"), "test": ("2015-01", None)},
]
SPEC_KEYS = ("lags", "regime", "ridge", "window_years", "kind", "hidden")
FACTOR_NAMES = {f["id"]: f["name"] for f in config.FACTORS}


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


def leakage_evidence(panel: pd.DataFrame, col: str, train: tuple) -> dict:
    """Correlation of a candidate series with the market's price return this month, next month and last month."""
    P = panel.loc[train[0]:train[1]]
    x, y = P[col], P["p_us_equity"]
    return {"same_month": float(x.corr(y)), "next_month": float(x.corr(y.shift(-1))),
            "prev_month": float(x.corr(y.shift(1)))}


def build_candidates(panel: pd.DataFrame, train: tuple) -> list[Candidate]:
    out = []
    for f in CANDIDATE_FACTORS:
        col = f"f_{f}"
        if col not in panel or panel.loc[train[0]:train[1], col].notna().sum() < 120:
            continue
        ev = leakage_evidence(panel, col, train)
        note = ""
        if f in ENDOGENOUS:
            note = (f"{ENDOGENOUS[f]}（训练期与美股当月价格收益相关 {ev['same_month']:.2f}，"
                    f"与下月 {ev['next_month']:.2f}），只作对照，不参与选择。")
        elif f in QUARTERLY:
            note = "季度数据插值成月度，含 1–2 个月的前视信息，样本外结果会略偏乐观。"
        elif f in RELEASE_DATED:
            note = "季度数据按发布月对齐（没有前视），只在发布月变化。"
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
    out += [
        Candidate("ridge_0.01", "tuning", "系数估计噪声大，需要轻度收缩（岭回归 λ=0.01）", {"ridge": 0.01}),
        Candidate("ridge_0.05", "tuning", "系数估计噪声大，需要较强收缩（岭回归 λ=0.05）", {"ridge": 0.05}),
        Candidate("window_15", "tuning", "传导系数随时代变化：只用最近 15 年估计", {"window_years": 15}),
        Candidate("window_25", "tuning", "传导系数随时代变化：只用最近 25 年估计", {"window_years": 25}),
        Candidate("mlp16", "neural_net", "传导是非线性的：小型神经网络（16 个隐藏单元）", {"kind": "mlp", "hidden": 16}),
        Candidate("mlp32", "neural_net", "传导是非线性的：小型神经网络（32 个隐藏单元）", {"kind": "mlp", "hidden": 32}),
    ]
    return out


def evaluate(panel: pd.DataFrame, spec: ModelSpec, train: tuple, test: tuple,
             assets: list[str] = SCORE_ASSETS, months: pd.DatetimeIndex | None = None) -> dict:
    """Fit on `train`, then score the price returns of `test` predicted from the actual factors.

    `months` restricts the scoring to those months, so specs can be compared on a common sample
    (structural shocks are blank in the COVID months). `_index` lists the months scored.
    The R^2 benchmark is each asset's mean price return over the whole `train` period, the same
    for every spec (a recent-window spec is not scored against its own shorter mean).
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


def _test_text(sel: dict, test: dict, period: tuple, cand: dict, start_is_base: bool) -> str:
    d = test["selected"]["score"] - test["base"]["score"]
    if not sel["accepted"]:
        head = (f"验证期没有任何候选让平均 R² 提高超过 {MIN_GAIN}，保留起始模型；"
                f"它在未参与选择的测试期 {period[0]} 至 {period[1]} 的平均 R² 为 {test['selected']['score']:.3f}")
        if start_is_base:
            return head + "。"
        head += f"，基准模型 {test['base']['score']:.3f}，"
    else:
        names = "、".join(cand[c].hypothesis for c in sel["accepted"])
        head = (f"验证期选中：{names}（平均 R² {sel['path'][0]['val_score']:.3f} → {sel['result']['score']:.3f}）。"
                f"在未参与选择的测试期 {period[0]} 至 {period[1]}：基准模型 {test['base']['score']:.3f}，"
                f"选中模型 {test['selected']['score']:.3f}，")
    if d > MIN_GAIN:
        return head + f"提升 {d:.3f}，改进经受住了样本外检验。"
    if d > -MIN_GAIN:
        return head + f"差别 {d:+.3f}，改进没有延续到测试期，但也没有变差。"
    return head + f"下降 {-d:.3f}，验证期的改进是过拟合或时代特有的，没有延续到测试期。"


def run_iteration(panel: pd.DataFrame, it: dict, start: ModelSpec, base: ModelSpec,
                  candidates: list[Candidate], diagnose: Callable | None = None) -> dict:
    train, validate, test = it["train"], it["validate"], _period(it["test"], panel)
    findings = diagnose(panel, start, train, validate, candidates) if diagnose else []
    base_val = evaluate(panel, base, train, validate)
    sel = greedy_select(panel, start, candidates, train, validate)
    trainval = (train[0], validate[1])
    specs = {"base": base, "selected": sel["spec"], **({} if same_spec(start, base) else {"start": start})}
    test_res = compare(panel, specs, trainval, test)
    base_all = evaluate(panel, base, trainval, test)
    delta = test_res["selected"]["score"] - test_res["base"]["score"]
    # the gate is about the selected spec, not this round: a spec carried over from the last
    # iteration with no new change still counts if it beats the base on the untouched test
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
        "start": {"spec": start.to_dict(), "describe": start.describe(), "is_base": same_spec(start, base)},
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
            "delta": delta, "improved": improved,
            "text": _test_text(sel, test_res, test, cand, same_spec(start, base)),
        },
        "test_ablation": sorted(ablation, key=lambda r: -r["test_delta"]),
    }


def run_walkforward(panel: pd.DataFrame, base: ModelSpec | None = None, iterations: list[dict] | None = None,
                    diagnose: Callable | None = None) -> dict:
    base = base or ModelSpec()
    start, changes = base, []
    out, cands = [], []
    for it in iterations or ITERATIONS:
        cands = build_candidates(panel, it["train"])
        res = run_iteration(panel, it, start, base, cands, diagnose)
        selected = ModelSpec.from_dict(res["selected"]["spec"])
        chain = changes + res["selected"]["changes"]
        res["selected"]["chain"] = chain
        if res["test_result"]["improved"]:
            start, changes = selected, chain
            res["carry_over"] = "选中模型在测试期胜过基准，下一轮从它出发。"
        else:
            start, changes = base, []
            res["carry_over"] = "选中模型没有在测试期胜过基准，下一轮重新从基准模型出发。"
        out.append(res)
    last = out[-1]
    passed = last["test_result"]["improved"] or not REQUIRE_TEST_GAIN
    literal = {"spec": last["selected"]["spec"], "describe": last["selected"]["describe"],
               "changes": last["selected"]["chain"], "val_score": last["selected"]["val_score"],
               "test_score": last["test_result"]["selected"]["score"]}
    trainval = (last["train"][0], last["validate"][1])
    vs = (f"（同一批 {last['test_result']['n_months']} 个月上 {literal['test_score']:.3f} 对 "
          f"{last['test_result']['base']['score']:.3f}）")
    if passed:
        changes = last["selected"]["chain"]
        final = ModelSpec.from_dict({**last["selected"]["spec"], "name": "+".join(["base", *changes])})
        val = literal["val_score"]
        reason = (f"最后一轮选中的模型在未参与选择的测试期胜过基准{vs}，采用它。" if last["test_result"]["improved"]
                  else "按协议采用最后一轮选中的模型，测试期只用于报告。")
    else:
        changes, final = [], base
        val = last["baseline"]["val_score"]
        reason = f"最后一轮选中的模型（{literal['describe']}）在未参与选择的测试期没有胜过基准{vs}，所以保留基准模型。"
    if REQUIRE_TEST_GAIN:
        reason += ("测试期因此参与了这一次“采用 / 不采用”的决定，之后没有再留出的数据；"
                   + ("基准模型是事先固定的，所以它的测试期 R² 仍是无偏的样本外估计。" if not changes else
                      "选中模型的测试期 R² 是在通过这道关卡的条件下观察到的，会略偏乐观。"))
    own = evaluate(panel, final, trainval, tuple(last["test"]))            # every month the final spec can predict
    tst = own["score"]
    return {
        "candidates": [c.to_dict() for c in cands], "score_assets": SCORE_ASSETS,
        "min_gain": MIN_GAIN, "max_accept": MAX_ACCEPT, "require_test_gain": REQUIRE_TEST_GAIN,
        "r2_benchmark": "training_mean",
        "iterations": out,
        "final": {"spec": final.to_dict(), "describe": final.describe(), "changes": changes, "is_base": not changes,
                  "test_used_for_choice": REQUIRE_TEST_GAIN,
                  "val_score": val, "test_score": tst, "test_months": own["n_months"],
                  "test_per_asset": own["per_asset"], "base_val_score": last["baseline"]["val_score"],
                  "base_test_score": tst if not changes else evaluate(panel, base, trainval, tuple(last["test"]))["score"],
                  "reason": reason,
                  "last_selected": literal,
                  "selected_on": {"train": last["train"], "validate": last["validate"], "test": last["test"]}},
    }
