"""The built-in scenarios as QTables (SIM_DESIGN §7.4, §14 P6): scenariolab/scenarios/qtables/<id>.json.

    python tools/qtable_from_yaml.py              # convert the 13 YAML specs and write the reviewed QTables
    python tools/qtable_from_yaml.py --check      # exit 1 if a committed QTable differs from a fresh conversion
    python tools/qtable_from_yaml.py --delivery   # run each QTable on the quarterly stack: delivery per view
    python tools/qtable_from_yaml.py --only oil_war,financial_crisis

The conversion is deterministic (it needs the quarterly panel: the training sd of each field is the unit of the
statements' sd and of the stance bins):

* every phase view becomes a soft statement on the matching quarterly field, over the phase's quarters. A YAML view
  is a phase average, so it becomes a c_q.constraints "mean" over those quarters (the policy rate's level: "level",
  quarter by quarter, because the observed rate's floor is per quarter), in the field's units (VIEWS:
  f_oil mean_annual a -> d_oil 25 a log%/q; f_ff / f_unrate / f_y10 / f_tp10 / f_sentiment a -> a / 4 per quarter;
  f_ip / f_usd a -> 25 a; s_cpi_yoy, s_unrate, s_ff m -> the level's mean m; s_goods_yoy, s_services_yoy,
  s_productivity_yoy m -> the annualized rate's mean m; s_ip_yoy m -> d_ip m / 4 (a y/y is about four quarters'
  growth); s_real_ff and s_real_wage_yoy m -> the nominal level m + the phase's CPI y/y view, when the phase has one);
* the view confidence becomes the sd: (0.25 + 0.75 (1 - confidence)) x the field's training sd (confidence 1 gives
  the 小 bin, the finest size the QTable knows);
* spread views (f_spread, s_spread) become the phase's fc stances (stance_plan): k quarters of one stance at the
  phase start (收紧 / 宽松 +-1, 明显收紧 / 明显宽松 +-2 training sd of the quarterly spread change), then 中性,
  with the stance and k chosen so the phase's mean change (f_spread) or mean level (s_spread, from the level the
  previous phases' stances reach) comes as close as the stance grid allows; the remaining gap is recorded as the
  conversion's discretization. An e_oil_supply view becomes the phase's x_oil_cause = supply; regime keys become
  c_q.regime requests;
* what the QTable cannot say is dropped and recorded: vol_mult-only views, views on variables with no QTable field
  (capex and profit shares, dividend yield, sentiment levels, white-collar wages, real disposable income, the IT
  share), and the monthly engine's overrides (asset and factor drifts, per-phase r_star: anchors live in c_q);
* every quote is the view's own `why` (or a clause of the narrative), cut to 120 characters, so it is a substring of
  the scenario's text (summary, narrative, phase names and the whys) and names the variable where the rule asks.

A human review then adds the financial-conditions stance where the YAML has none (REVIEW: each decision with its
reason and quote, recorded in the QTable's envelope), and the validator must accept every QTable as written.

--delivery checks the converted QTables on the quarterly stack: each converted view's delivered phase mean against
the YAML's target, within the monthly engine's own delivery tolerance (how far its simulated median ends from the
same target, from data/results/scenarios/<id>.json; never below the 小 bin, 0.25 training sd), reported per view in
data/results/qtable_delivery.json. Nothing here calls an LLM or downloads anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenariolab import config  # noqa: E402
from scenariolab.qmodel import config_q as cq  # noqa: E402
from scenariolab.qmodel import qtable as QT  # noqa: E402

SPECS = ROOT / "scenariolab" / "scenarios" / "specs"
OUT = QT.QTABLES_DIR
DELIVERY_PATH = config.RESULTS_DIR / "qtable_delivery.json"
GENERATED_BY = "tools/qtable_from_yaml.py"
DIGITS = 4
# monthly view -> (QTable field, key, factor): the quarterly value = factor x the view's number
VIEWS = {
    ("f_oil", "mean_annual"): ("d_oil", 25.0), ("f_ip", "mean_annual"): ("d_ip", 25.0),
    ("f_usd", "mean_annual"): ("d_usd", 25.0), ("f_ff", "mean_annual"): ("d_ff", 0.25),
    ("f_unrate", "mean_annual"): ("d_unrate", 0.25), ("f_y10", "mean_annual"): ("d_y10", 0.25),
    ("f_tp10", "mean_annual"): ("d_tp10", 0.25), ("f_sentiment", "mean_annual"): ("d_sent", 0.25),
    ("s_cpi_yoy", "mean"): ("s_cpi_yoy", 1.0), ("s_unrate", "mean"): ("s_unrate", 1.0),
    ("s_ff", "mean"): ("s_ff", 1.0), ("s_goods_yoy", "mean"): ("d_cpi_goods", 1.0),
    ("s_services_yoy", "mean"): ("d_cpi_serv", 1.0), ("s_productivity_yoy", "mean"): ("d_lprod", 1.0),
    ("s_ip_yoy", "mean"): ("d_ip", 0.25),
}
DERIVED = {("s_real_ff", "mean"): ("s_ff", "名义政策利率 = 实际政策利率 + 同阶段的 CPI 同比视图"),
           ("s_real_wage_yoy", "mean"): ("d_wage", "名义工资增长 = 实际工资同比 + 同阶段的 CPI 同比视图")}
SPREAD_VIEWS = ("f_spread", "s_spread")
DROP = {     # reader text (the forecast tab lists these records): no code ids
    "f_capex_share": "资本开支占 GDP 在季度情景表里没有对应字段（季度模型只有实际投资增长，不是占比）",
    "f_profit_share": "利润占 GDP 在季度情景表里没有对应字段（季度模型里它是期初状态，不是输入）",
    "s_div_yield": "股息率视图是月度引擎挑选高估值月份的手段：季度情景表不写估值数字，估值由宏观与信用隐含",
    "s_sentiment": "消费者信心的水平在季度情景表里没有对应字段（只有它的变动）",
    "s_wc_real_wage_yoy": "白领实际时薪在季度情景表里没有对应字段",
    "s_real_dpi_yoy": "实际可支配收入在季度情景表里没有对应字段",
    "s_it_invest_share": "IT 投资占 GDP 在季度模型里是视图，不是输入",
}
VOL_NOTE = "波动约束（按历史波动的倍数挑月份）没有季度对应：不作输入"
OVERRIDE_NOTE = {"asset_drift": "资产价格漂移不是季度情景表的输入（资产价格与估值倍数不是输入）",
                 "factor_drift": "因子漂移是月度引擎的覆盖项：季度情景表只写文字陈述的字段",
                 "r_star": "阶段的中性利率覆盖不是季度情景表的输入（长期锚要带引文写进情景表）",
                 "term_premium": "阶段的期限溢价覆盖不是季度情景表的输入", "resid_scale": "残差缩放不是季度情景表的输入"}
REGIME_ZH_TO_ID = QT.REGIME_ALIASES
# The human review (§7.4): where the YAML has no spread view, the financial-conditions stance a reader of the
# scenario's text would write, with the quote it rests on; and the decisions to leave a stance unstated.
REVIEW = {
    "agi_boom": [{"op": "keep", "why": "叙事明说信用来自被视图加权的历史月份，没有陈述金融条件：保留未说明（模型补全，W-FC）"}],
    "ai_bust": [{"op": "keep", "why": "前两个阶段的信用利差视图已换成金融条件；回归常态阶段没有陈述"}],
    "baseline": [{"op": "keep", "why": "前两个阶段的信用利差视图已换成金融条件；回归常态阶段没有陈述"}],
    "deglobalization": [{"op": "keep", "why": "叙事没有说信用利差或金融条件：保留未说明"}],
    "dollar_decline": [{"op": "keep", "why": "叙事只说期限溢价上升（已作为 d_tp10 的陈述），没有说信用利差：保留未说明"}],
    "energy_transition": [{"op": "keep", "why": "叙事没有说信用利差或金融条件：保留未说明"}],
    "financial_crisis": [{"op": "keep", "why": "三个阶段的信用利差视图已换成金融条件（第一阶段按利差的年度升幅，第二、三阶段按水平）"}],
    "fiscal_dominance": [{"op": "keep", "why": "“金融抑制”说的是利率被压低，不是信用利差：金融条件保留未说明"}],
    "japanification": [{"op": "keep", "why": "叙事没有说信用利差或金融条件：保留未说明"}],
    "k_shaped": [{"op": "keep", "why": "叙事没有说信用利差或金融条件：保留未说明"}],
    "oil_war": [{"op": "fc", "phase": 2, "quote": "信用利差从 1.5 升到 1.9 左右", "change_pp": 0.4,
                 "why": "叙事写明衰退期信用利差从 1.5 升到 1.9 左右：一年 +0.4 个百分点，按同一换算约每季 +0.2 sd，取最近的档"}],
    "reacceleration": [{"op": "keep", "why": "第一阶段的信用利差视图已换成金融条件；高利率常态阶段没有陈述"}],
    "stagflation": [{"op": "keep", "why": "叙事没有说信用利差或金融条件：保留未说明"}],
}


def _r(x: float) -> float:
    return float(round(float(x), DIGITS))


def source_text(spec: dict) -> str:
    """What a built-in scenario's quotes must come from: its summary, narrative, phase names and the views' whys."""
    parts = [str(spec.get("summary") or ""), str(spec.get("narrative") or "")]
    for ph in spec.get("phases") or []:
        parts.append(str(ph.get("name") or ""))
        for v in ph.get("views") or []:
            if v.get("why"):
                parts.append(str(v["why"]))
    return "\n".join(p for p in parts if p)


def _quote_for(text: str, why: str | None, words=None) -> str | None:
    """The view's why (cut to 120 characters) when it names the variable; else the first clause of the text that
    does; None when nothing in the text names it."""
    if why:
        q = why.strip()[:cq.QUOTE_MAX]
        if words is None or QT.names(q, words):
            return q
    if words is None:
        return None
    for clause in re.findall(r"[^。；，\n]+", text):
        if QT.names(clause, words):
            return clause.strip()[:cq.QUOTE_MAX]
    return None


def stance_plan(how: str, target: float, s0: float, n: int, b: float) -> tuple[list, float, float]:
    """The fc stances of a phase from a spread view: k quarters of one stance at the phase start (收紧 / 宽松 = +-1 sd
    of the quarterly spread change, 明显 = +-2 sd: config_q.FC_STANCE), then 中性, with the stance and k chosen so the
    phase's mean change ("flow": per quarter) or mean level ("level", from the start level s0) is as close as the
    stance grid allows. ([(stance, quarters)], the implied value, the end level)."""
    best = (abs(s0 - target) if how == "level" else abs(target), 0, 0.0, 0)      # all 中性
    for mult in (1.0, 2.0):
        for sign in (1.0, -1.0):
            c = sign * mult * b
            for k in range(1, n + 1):
                val = c * k / n if how == "flow" else s0 + c * (k * (k + 1) / 2.0 + k * (n - k)) / n
                cand = (abs(val - target), mult, c, k)
                if cand[0] < best[0] - 1e-12:
                    best = cand
    err, mult, c, k = best
    name = {(1.0, True): "收紧", (1.0, False): "宽松", (2.0, True): "明显收紧", (2.0, False): "明显宽松"}
    plan = ([(name[(mult, c > 0)], k)] if k else []) + ([("中性", n - k)] if n - k else [])
    implied = c * k / n if how == "flow" else s0 + c * (k * (k + 1) / 2.0 + k * (n - k)) / n
    return plan, (implied if k else (0.0 if how == "flow" else s0)), s0 + c * k


def _expand(plan: list) -> list:
    return [(st_, i) for st_, c_ in plan for i in range(c_)]


def convert(spec: dict, stats: dict, today: dict) -> dict:
    """(the envelope of) one YAML spec as a QTable: {id, name, qtable, text, conversion, review, source}."""
    from scenariolab.qmodel import card as qcard
    card = qcard.load_card()
    kw = {r["id"]: r.get("keywords") for r in qcard.rows(card, "x") if r.get("permission") == "stated_only"}
    text = source_text(spec)
    H = sum(int(ph["months"]) for ph in spec["phases"]) // 3
    if H > cq.QTABLE_MAX_QUARTERS:
        raise SystemExit(f"{spec['id']}: {H} quarters > {cq.QTABLE_MAX_QUARTERS}")
    rows = [{"q": q, "x": {}, "fc": {"stance": "未说明"}} for q in range(1, H + 1)]
    constraints, regime, conversion, notes = [], [], [], []
    start, spread0 = 0, today.get("s_spread")
    for k, ph in enumerate(spec["phases"], start=1):
        n = int(ph["months"]) // 3
        qs = list(range(start + 1, start + n + 1))
        start += n
        conf = float(ph.get("confidence") if ph.get("confidence") is not None else 1.0)
        sd_factor = 0.25 + 0.75 * (1.0 - conf)
        views = ph.get("views") or []
        cpi = next((v.get("mean") for v in views if v.get("var") == "s_cpi_yoy"), None)
        rec = lambda v, **kv: conversion.append({"phase": k, "phase_name": ph["name"], "quarters": qs,  # noqa: E731
                                                 "var": v["var"], "target": {t: v[t] for t in v if t not in
                                                                             ("var", "why")}, **kv})
        spread_view = None                       # ("flow", per quarter, view) or ("level", phase mean, view)
        for v in views:
            var = v["var"]
            key = next((t for t in ("mean_annual", "mean") if v.get(t) is not None), None)
            if var in SPREAD_VIEWS and key is not None:
                if var == "f_spread":           # the flow wins over a level view of the same phase
                    spread_view = ("flow", float(v[key]) * (0.25 if key == "mean_annual" else 3.0), v)
                elif spread_view is None:
                    spread_view = ("level", float(v[key]), v)
                rec(v, status="stance", to="rows.fc", note="信用利差视图换成金融条件")
                continue
            if var == "e_oil_supply" and key is not None:
                if float(v[key]) > 0:
                    q_ = _quote_for(text, v.get("why"))
                    for q in qs:
                        rows[q - 1]["x"]["x_oil_cause"] = {"value": "supply", "src": "stated_qual", "quote": q_}
                    rec(v, status="converted", to="rows.x.x_oil_cause", field="x_oil_cause", value="supply")
                else:
                    rec(v, status="dropped", note="油价的需求冲击方向：季度情景表只按原因（供给 / 需求 / 混合）写")
                continue
            if var in DROP:
                rec(v, status="dropped", note=DROP[var])
                continue
            if key is None:
                rec(v, status="dropped", note=VOL_NOTE if "vol_mult" in v else "没有均值目标：不作输入")
                continue
            if (var, key) in DERIVED:
                field, how = DERIVED[(var, key)]
                if cpi is None:
                    rec(v, status="dropped", note=f"{how}：这个阶段没有 CPI 同比视图，换算不了")
                    continue
                value = float(v[key]) + float(cpi)
                note = how
            elif (var, key) in VIEWS:
                field, factor = VIEWS[(var, key)]
                value = float(v[key]) * factor
                note = "工业产出同比 ≈ 四个季度的增长之和：按每季 m / 4 写" if var == "s_ip_yoy" else None
            else:
                rec(v, status="dropped", note=f"{var} 没有对应的季度字段")
                continue
            unit = (stats.get(field) or {}).get("sd")
            if unit is None:
                rec(v, status="dropped", note=f"{field} 没有训练期标准差")
                continue
            quote = _quote_for(text, v.get("why"), kw.get(field))
            if quote is None:
                rec(v, status="dropped", note=f"文字没有提到 {field} 的关键词：市场价格和调查值不能由翻译器决定")
                continue
            # a phase average; the policy rate's level is stated quarter by quarter (its floor is per quarter, so a
            # mean of the observed rate is not linear: §5.4)
            kind = "level" if field == "s_ff" else "mean"
            item = {"kind": kind, "field": field, "quarters": qs, "value": _r(value), "sd": _r(sd_factor * unit),
                    "quote": quote}
            constraints.append(item)
            rec(v, status="converted", to="c_q.constraints", field=field, value=item["value"], sd=item["sd"],
                **({"note": note} if note else {}))
        if spread_view is not None:
            how, target, v = spread_view
            s0 = spread0 if spread0 is not None else 2.0
            plan, implied, end = stance_plan(how, target, s0, n, stats["d_spread"]["sd"])
            quote = _quote_for(text, v.get("why"), QT.FC_WORDS + tuple(kw.get("d_spread") or ()))
            for (stance, m_), q in zip(_expand(plan), qs):
                rows[q - 1]["fc"] = {"stance": stance, "quote": quote}
            spread0 = end
            conversion.append({"phase": k, "phase_name": ph["name"], "quarters": qs, "var": "fc", "status": "converted",
                               "to": "rows.fc", "from": v["var"], "how": how, "target": _r(target), "start": _r(s0),
                               "plan": [{"stance": st_, "quarters": c_} for st_, c_ in plan], "implied": _r(implied),
                               "discretization": _r(abs(implied - target))})
        for zh, share in (ph.get("regime") or {}).items():
            rid = REGIME_ZH_TO_ID.get(str(zh), zh)
            words = (zh, QT.REGIME_ZH.get(rid, zh))
            quote = _quote_for(text, None, words) or ph["name"][:cq.QUOTE_MAX]
            regime.append({"quarters": qs, "label": rid, "share": float(share), "quote": quote})
            conversion.append({"phase": k, "phase_name": ph["name"], "quarters": qs, "var": f"regime:{zh}",
                               "status": "converted", "to": "c_q.regime", "label": rid, "share": float(share)})
        for key_, val in (ph.get("overrides") or {}).items():
            conversion.append({"phase": k, "phase_name": ph["name"], "quarters": qs, "var": f"overrides.{key_}",
                               "target": val, "status": "dropped",
                               "note": OVERRIDE_NOTE.get(key_, "月度引擎的覆盖项不是 QTable 输入")})
    review = []
    for edit in REVIEW.get(spec["id"], []):
        review.append(dict(edit))
        if edit["op"] == "fc":
            ph_start = sum(int(p["months"]) // 3 for p in spec["phases"][:edit["phase"] - 1])
            n = int(spec["phases"][edit["phase"] - 1]["months"]) // 3
            plan, implied, _ = stance_plan("flow", edit["change_pp"] / n, 0.0, n, stats["d_spread"]["sd"])
            review[-1].update(plan=[{"stance": st_, "quarters": c_} for st_, c_ in plan], implied=_r(implied))
            for (stance, _m), q in zip(_expand(plan), range(ph_start + 1, ph_start + n + 1)):
                if rows[q - 1]["fc"]["stance"] == "未说明":
                    rows[q - 1]["fc"] = {"stance": stance, "quote": edit["quote"]}
    c_q = {}
    if constraints:
        c_q["constraints"] = constraints
    if regime:
        c_q["regime"] = regime
    notes.append(f"由 {spec['id']}.yaml 换算（{GENERATED_BY}），经人工复核；YAML 的视图是阶段平均，写成 c_q 的 mean 约束")
    qt = {"schema": QT.SCHEMA_ID, "model_card": cq.CARD_ID, "n_quarters": H, "rows": rows, "c_q": c_q,
          "events": [], "notes": notes}
    return {"id": spec["id"], "name": spec.get("name"), "prior_prob": spec.get("prior_prob"), "qtable": qt,
            "text": text, "conversion": conversion, "review": review}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(only: list[str] | None = None) -> dict[str, dict]:
    from scenariolab.qmodel import panel as qpanel
    from scenariolab.qmodel import rollout as R
    Q = qpanel.load_panel()
    te = R.last_data_quarter(Q)
    stats = QT.stats_table(Q, te)
    today = {c: float(Q[c].iloc[-1]) for c in ("s_spread", "s_cpi_yoy", "s_ff") if c in Q.columns}
    out = {}
    for path in sorted(SPECS.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if only and spec["id"] not in only:
            continue
        env = convert(spec, stats, today)
        env["source"] = {"yaml": f"scenariolab/scenarios/specs/{path.name}", "yaml_sha256": _sha(path),
                         "train_end": te, "today": {k: _r(v) for k, v in today.items()}}
        env["generated_by"] = GENERATED_BY
        rep = QT.validate(env["qtable"], env["text"], mode="production", source="builtin", stats=stats)
        if not rep["ok"]:
            raise SystemExit(f"{spec['id']}: the reviewed QTable does not validate: {QT.first_problem(rep)}")
        env["qtable"] = rep["qtable"]
        out[spec["id"]] = env
    return out


def dump(env: dict) -> str:
    return json.dumps(env, ensure_ascii=False, indent=1) + "\n"


# ---------------------------------------------------------------- delivery on the quarterly stack

E_VARS = {"s_cpi_yoy": ("cpi_yoy", "level"), "s_unrate": ("unrate", "level"), "s_ff": ("ff", "level"),
          "d_ff": ("ff", "change"), "d_unrate": ("unrate", "change"), "d_y10": ("y10", "change"),
          "d_oil": ("oil", "logchange"), "d_spread": ("spread", "change"), "s_spread": ("spread", "level")}


def _engine_value(detail: dict | None, field: str, start: int, months: int) -> float | None:
    """The monthly engine's simulated median over a phase, in the QTable field's units (None if not simulated)."""
    import numpy as np
    if detail is None or field not in E_VARS:
        return None
    var, how = E_VARS[field]
    fan = ((detail.get("macro") or {}).get(var) or {}).get("fan") or {}
    p50 = fan.get("p50")
    if not p50 or len(p50) <= start + months:
        return None
    a = np.asarray(p50, float)
    nq = months / 3.0
    if how == "level":
        return float(a[start + 1:start + months + 1].mean())
    if how == "change":
        return float((a[start + months] - a[start]) / nq)
    return float(100.0 * (math.log(a[start + months]) - math.log(a[start])) / nq)


def delivery(envs: dict[str, dict], n_paths: int = 1000, seed: int = 0) -> dict:
    """Per scenario and converted view: what the quarterly stack delivers over the view's phase (the median over
    paths of each path's phase mean, on the paths the factor layer keeps finite and inside its tail bounds) against
    the YAML's target, and the tolerance: the monthly engine's own gap on the same target, never below 0.25 training
    sd. Two documented exceptions hold a view to what the QTable could state instead of the YAML's number: a statement
    the factor layer saturated at +-4 sd of its conditional distribution (§3.5) is held to its clipped value, and a
    spread view (an fc stance plan, whose grid is 1 sd per quarter) to the plan's implied value; both are reported."""
    import numpy as np

    from scenariolab.qmodel import rollout as R
    stack = R.get_stack()
    stats = QT.stats_table(stack.Q, stack.train_end, shocks=True)
    report = {"label": cq.TRIAL_LABEL, "n_paths": n_paths, "seed": seed, "stack": stack.version,
              "statistic": "median over paths of the phase mean", "scenarios": {}}
    for sid, env in envs.items():
        read = QT.constraints(env["qtable"], stack=stack, narrative=env["text"], source="builtin")
        with np.errstate(all="ignore"):         # the factor layer's diverging paths (counted below) overflow
            roll = R.run_path(stack, read["statements"], read["valuation"], H=read["H"], n_paths=n_paths, seed=seed,
                              cycle_odds=read["cycle_odds"], states=False)
        try:
            detail = json.loads((config.RESULTS_DIR / "scenarios" / f"{sid}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            detail = None
        lv = roll.fx.levels
        last = max(c["quarters"][-1] for c in env["conversion"] if c["status"] != "dropped") if env["conversion"] else 1
        # the factor layer's own paths only: the asset layer is not what a view states (and one non-finite factor
        # path can make the dollar hold's cross-path mean, and so every path's assets, non-finite)
        with np.errstate(invalid="ignore"):
            core = np.stack([roll.fx.x[f][:, :last] for f in ("d_ip", "d_unrate", "d_spread", "d_gz", "d_ff",
                                                              "d_cpi_goods", "d_cpi_serv", "d_oil", "d_y10")])
            keep = np.isfinite(core).all(axis=(0, 2)) & (np.nan_to_num(lv["gz"][:, :last], nan=1e9) <= R.TAIL_GZ).all(
                axis=1) & (np.nan_to_num(lv["unrate"][:, :last], nan=1e9) <= R.TAIL_U).all(axis=1)
        clips = {(n.get("var"), tuple(n.get("quarters") or ())): n for n in roll.fx.notes
                 if isinstance(n, dict) and n.get("used") is not None}
        plans = {c["phase"]: c for c in env["conversion"] if c["var"] == "fc"}

        def phase_median(a: np.ndarray, idx: list) -> float:
            return float(np.median(a[keep][:, idx].mean(axis=1)))

        rows = []
        for c in env["conversion"]:
            if c["status"] == "dropped" or c["var"].startswith("regime:") or c["var"] == "fc" \
                    or c.get("field") == "x_oil_cause":
                continue
            qs = c["quarters"]
            idx = [q - 1 for q in qs]
            start, months = 3 * (qs[0] - 1), 3 * len(qs)
            clip, expected, why = None, None, None
            if c["var"] in SPREAD_VIEWS:
                key = "mean_annual" if "mean_annual" in c["target"] else "mean"
                plan = plans.get(c["phase"])
                if c["var"] == "f_spread":
                    field, target = "d_spread", c["target"][key] * (0.25 if key == "mean_annual" else 3.0)
                    got = phase_median(roll.fx.x["d_spread"], idx)
                else:
                    field, target = "s_spread", float(c["target"][key])
                    got = phase_median(lv["spread"], idx)
                if plan is not None and plan.get("from") == c["var"]:
                    expected, why = plan["implied"], "fc 立场的网格（每季 1 sd）：按立场计划的隐含值核对"
                unit = stats["d_spread"]["sd"]
            else:
                field, target = c["field"], float(c["value"])
                var = QT.LEVEL_VARS.get(field, field)
                got = phase_median(lv[var] if field in QT.LEVEL_VARS else roll.fx.x[field], idx)
                unit = stats[field]["sd"]
                clip = clips.get((var, tuple(qs))) or clips.get(("ff_shadow", tuple(qs)))
                if clip is not None:
                    expected, why = float(clip["used"]), clip.get("text")
            eng = _engine_value(detail, field, start, months)
            e_gap = None if eng is None else abs(eng - target)
            tol = max(e_gap or 0.0, 0.25 * unit)
            gap = abs(got - target)
            ok = gap <= tol or (expected is not None and abs(got - expected) <= 0.25 * unit)
            row = {"phase": c["phase"], "phase_name": c["phase_name"], "quarters": [qs[0], qs[-1]],
                   "view": c["var"], "field": field, "target": _r(target), "quarterly": _r(got),
                   "engine": None if eng is None else _r(eng), "gap": _r(gap),
                   "engine_gap": None if e_gap is None else _r(e_gap), "tolerance": _r(tol), "ok": bool(ok)}
            if expected is not None:
                row.update(held_to=_r(expected), gap_held=_r(abs(got - expected)), reason=why,
                           saturated=clip is not None)
            rows.append(row)
        report["scenarios"][sid] = {"views": rows, "ok": all(r["ok"] for r in rows),
                                    "excluded_paths": int((~keep).sum()),
                                    "dropped": [{"phase": c["phase"], "view": c["var"], "note": c.get("note")}
                                                for c in env["conversion"] if c["status"] == "dropped"]}
    report["ok"] = all(s["ok"] for s in report["scenarios"].values())
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="13 个内置情景 → 季度情景表（QTable/1）")
    ap.add_argument("--check", action="store_true", help="与已提交的文件比较（不写）")
    ap.add_argument("--delivery", action="store_true", help="在季度模型上检查每个视图的交付")
    ap.add_argument("--only", default="", help="逗号分隔的情景编号")
    ap.add_argument("--paths", type=int, default=1000)
    a = ap.parse_args(argv)
    only = [s for s in a.only.split(",") if s] or None
    envs = build(only)
    if a.check:
        bad = []
        for sid, env in envs.items():
            path = OUT / f"{sid}.json"
            if not path.exists() or path.read_text(encoding="utf-8") != dump(env):
                bad.append(sid)
        if bad:
            print(f"与重新换算的结果不同：{', '.join(bad)}（运行 python {GENERATED_BY}）")
            return 1
        print(f"{len(envs)} 个季度情景表与 YAML 一致")
        return 0
    if a.delivery:
        rep = delivery(envs, n_paths=a.paths)
        DELIVERY_PATH.parent.mkdir(parents=True, exist_ok=True)
        DELIVERY_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        for sid, s in rep["scenarios"].items():
            fails = [r for r in s["views"] if not r["ok"]]
            print(f"{sid}: {len(s['views'])} 个视图，{'全部交付' if not fails else f'{len(fails)} 个超出容差'}")
            for r in fails:
                print(f"   阶段 {r['phase']} {r['view']} → {r['field']}: 目标 {r['target']}，季度模型 {r['quarterly']}，"
                      f"差 {r['gap']} > 容差 {r['tolerance']}")
            for r in s["views"]:
                if r.get("held_to") is not None and r["gap"] > r["tolerance"]:
                    print(f"   阶段 {r['phase']} {r['view']} → {r['field']}: 目标 {r['target']}，按 {r['held_to']} 核对"
                          f"（{r['reason']}），交付 {r['quarterly']}")
            if s["excluded_paths"]:
                print(f"   {s['excluded_paths']} 条路径在情景季度里发散（非有限值或超出尾部界限：因子层），已剔除")
        print(f"→ {DELIVERY_PATH}")
        return 0 if rep["ok"] else 1
    OUT.mkdir(parents=True, exist_ok=True)
    for sid, env in envs.items():
        (OUT / f"{sid}.json").write_text(dump(env), encoding="utf-8")
    print(f"写了 {len(envs)} 个季度情景表 → {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
