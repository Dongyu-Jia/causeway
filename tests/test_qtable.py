"""The scenario layer (SIM_DESIGN §7, Appendix B; §14 P6): qmodel/qtable.py, llm/qtranslator.py, the mock's QTable
answers, tools/qtable_from_yaml.py and the 13 reviewed built-in QTables.

P6 exit tests: the JSON schema validates the examples; every refusal rule has a fixture; mock end to end: narrative
-> QTable -> run -> report; malformed JSON, refusal, timeout and usage-limit fixtures map to repair / fallback /
pause; YAML -> QTable delivery; the test-mode prompt scans clean.

The rules, the prompt, the mock and the fixtures run anywhere (synthetic training statistics stand in for the
quarterly panel). Running a QTable needs the quarterly panel (python -m scenariolab.qmodel.panel) and skips without
it; the YAML delivery test also needs the built monthly scenarios (data/results/scenarios/). Every LLM call here is
the mock: nothing spends plan usage.
"""
import copy
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path

import pytest
import yaml

from scenariolab import config
from scenariolab.llm import jsonfix
from scenariolab.llm import mock as llm_mock
from scenariolab.llm import qtranslator as QX
from scenariolab.qmodel import card as qcard
from scenariolab.qmodel import config_q as cq
from scenariolab.qmodel import qtable as QT

needs_qpanel = pytest.mark.skipif(not (cq.PANEL_Q_PATH.exists() and config.PANEL_PATH.exists()),
                                  reason="quarterly panel not built (python -m scenariolab.qmodel.panel)")
needs_results = pytest.mark.skipif(not (config.RESULTS_DIR / "scenarios" / "oil_war.json").exists(),
                                   reason="built monthly scenarios missing (python -m scenariolab.build)")
SPECS = Path(QT.__file__).resolve().parent.parent / "scenarios" / "specs"
TOOL = Path(QT.__file__).resolve().parent.parent.parent / "tools" / "qtable_from_yaml.py"

# synthetic training statistics (sd of the quarterly value, its historical range): the rules need no panel
STATS = {"d_oil": {"sd": 20.0, "lo": -100.0, "hi": 80.0}, "d_ff": {"sd": 1.2, "lo": -7.7, "hi": 8.0},
         "d_gz": {"sd": 0.45, "lo": -2.1, "hi": 2.9}, "d_spread": {"sd": 0.43, "lo": -1.5, "hi": 2.3},
         "d_unrate": {"sd": 0.6, "lo": -3.2, "hi": 6.6}, "d_ip": {"sd": 1.7, "lo": -16.0, "hi": 6.0},
         "d_cpi_goods": {"sd": 5.4, "lo": -30.0, "hi": 40.0}, "d_gpr": {"sd": 22.0, "lo": -90.0, "hi": 110.0},
         "d_rgdp": {"sd": 3.0, "lo": -30.0, "hi": 35.0}, "d_lprod": {"sd": 3.0, "lo": -5.0, "hi": 9.0},
         "d_tp10": {"sd": 0.42, "lo": -1.5, "hi": 1.6}, "s_unrate": {"sd": 0.6, "lo": 3.4, "hi": 13.0},
         "u_ff": {"sd": 1.07, "lo": None, "hi": None}, "u_bs": {"sd": 0.5, "lo": None, "hi": None}}

NARRATIVE = ("产油国卷入战争，供应中断，油价一个季度内上涨约三成，地缘政治风险急剧上升，商品价格明显上涨，信用利差走阔。"
             "随后央行加息 50 个基点。之后在政治压力下央行意外降息，金融条件大体平稳。就业基本稳定。"
             "中性利率维持在目前的估计。央行宣布扩表购买国债。实际 GDP 增长 2%，生产率增长 1.5%，失业率下降 0.2 个百分点。"
             "赤字扩大，转移支付增加。估值泡沫明显膨胀。央行按部就班地降息。经济陷入衰退。")

# Appendix B's example (each quote is in NARRATIVE)
APPENDIX_B = {"schema": "qtable/1", "model_card": "card/1 (schema-revise-2 3.0 + SIM_DESIGN)", "n_quarters": 4,
              "rows": [
                  {"q": 1, "x": {"d_oil": {"value": 26, "sd": 8, "src": "stated", "quote": "油价一个季度内上涨约三成"},
                                 "x_oil_cause": {"value": "supply", "src": "stated_qual", "quote": "产油国卷入战争，供应中断"},
                                 "d_gpr": {"sign": "+", "size": "大幅", "src": "stated_qual", "quote": "地缘政治风险急剧上升"},
                                 "d_cpi_goods": {"sign": "+", "size": "明显", "src": "stated_qual", "quote": "商品价格明显上涨"}},
                   "fc": {"stance": "收紧", "quote": "信用利差走阔"}},
                  {"q": 2, "x": {"d_ff": {"value": 0.5, "src": "stated", "quote": "央行加息 50 个基点"}},
                   "fc": {"stance": "未说明"}},
                  {"q": 3, "x": {}, "u": {"u_ff": {"sign": "-", "size": "明显", "src": "stated_qual",
                                                    "quote": "政治压力下央行意外降息"}},
                   "fc": {"stance": "中性", "quote": "金融条件大体平稳"}},
                  {"q": 4, "x": {}, "fc": {"stance": "未说明"}}],
              "c_q": {"anchors": {"r_star": {"value": 0.75, "quote": "中性利率维持在目前的估计"}},
                      "constraints": [{"kind": "range", "field": "d_unrate", "quarters": [1, 2, 3, 4], "min": -0.2,
                                       "max": 0.3, "quote": "就业基本稳定"}]},
              "events": [{"text": "产油国卷入战争", "quarters": [1, 2], "templates": ["G02"],
                          "channels": ["d_oil", "x_oil_cause", "d_gpr", "d_epu"], "generic_footprint": True,
                          "how_decomposed": "能源生产国卷入的战争 → 石油供给冲击、地缘风险上升、商品通胀"}]}


def _qt(rows=None, c_q=None, events=None, notes=None, n=2):
    rows = rows or [{"q": q, "x": {}, "fc": {"stance": "未说明"}} for q in range(1, n + 1)]
    out = {"schema": "qtable/1", "n_quarters": n, "rows": rows, "c_q": c_q or {}, "events": events or []}
    if notes is not None:
        out["notes"] = notes
    return out


def _row(q=1, x=None, u=None, fc=None):
    r = {"q": q, "x": x or {}, "fc": fc or {"stance": "未说明"}}
    if u:
        r["u"] = u
    return r


def _codes(report, level):
    return [it["code"] for it in report[{"refused": "refused", "warning": "warnings", "error": "errors"}[level]]]


# ================================================================ the JSON schema validates the examples

def test_the_card_generated_schema_validates_the_examples():
    schema = qcard.qtable_json_schema()
    examples = [APPENDIX_B, QT.minimal(6)] + [ex["qtable"] for ex in QX.EXAMPLES]
    examples += [QT.builtin(sid)["qtable"] for sid in QT.builtin_ids()]
    assert len(QT.builtin_ids()) == 13
    for ex in examples:
        assert jsonfix.validate(ex, schema) == [], ex.get("notes")
    # and it is the model's input schema: the x enum is the card's x_q set, the shocks are the card's
    assert qcard.qtable_fields(schema)["x"] == set(qcard.x_fields(qcard.load_card()))
    bad = copy.deepcopy(APPENDIX_B)
    bad["rows"][0]["x"]["d_gpr"]["size"] = "巨大"
    bad["rows"][1]["fc"]["stance"] = "很紧"
    bad["rows"][0]["x"]["d_oil"]["sd"] = 0
    errs = {e.kind for e in jsonfix.validate(bad, schema)}
    assert {"enum", "minimum"} <= errs                                 # exclusiveMinimum: sd > 0


def test_appendix_b_validates_against_its_narrative():
    rep = QT.validate(APPENDIX_B, NARRATIVE, stats=STATS)
    assert rep["ok"] and rep["readable"] and rep["n_quarters"] == 4
    assert QT.schema_errors(rep["qtable"]) == []
    assert rep["counts"]["stated"] == 2 and rep["counts"]["stated_qual"] == 4
    assert [r["q"] for r in rep["qtable"]["rows"]] == [1, 2, 3, 4]
    fc = [m for m in rep["missing"] if m["code"] == "W-FC"][0]
    assert fc["quarters"] == [2, 4] and fc["evidence"] and fc["evidence"] in NARRATIVE
    assert "W-ANCHOR" in _codes(rep, "warning")                        # anchors are recorded, not yet applied
    assert rep["anchors"]["r_star"] == {"value": 0.75, "from_q": 1, "quote": "中性利率维持在目前的估计"}


# ================================================================ every refusal rule has a fixture

def _x(q=1, **entries):
    return _qt([_row(q, x=entries)] + [_row(k) for k in range(1, 3) if k != q])


def _rows_qt(*rows, c_q=None, events=None, notes=None):
    rows = list(rows)
    have = {r["q"] for r in rows}
    n = max(have)
    return _qt(rows + [_row(k) for k in range(1, n + 1) if k not in have], c_q=c_q, events=events, notes=notes, n=n)


OIL = {"value": 26, "src": "stated", "quote": "油价一个季度内上涨约三成"}
RULES = [
    # (id, level, code, qtable, kwargs)
    ("not an object", "error", "R-SHAPE", "rows", {}),
    ("no rows", "error", "R-SHAPE", {"rows": []}, {}),
    ("too long", "error", "R-SHAPE", {**_qt(), "n_quarters": 61}, {}),
    ("bad row q", "refused", "R-SHAPE", _qt([_row(7)], n=2), {}),
    ("unknown field", "refused", "R-FIELD", _x(d_nope={"value": 1, "src": "stated", "quote": "就业基本稳定"}), {}),
    ("shock in x", "refused", "R-FIELD", _x(u_ff={"value": 1, "src": "stated", "quote": "就业基本稳定"}), {}),
    ("unknown shock", "refused", "R-FIELD",
     _rows_qt(_row(1, u={"u_nope": {"sign": "-", "size": "明显", "src": "stated_qual", "quote": "意外降息"}})), {}),
    ("unknown anchor", "refused", "R-FIELD", _qt(c_q={"anchors": {"r_nope": {"value": 1, "quote": "中性利率维持在目前的估计"}}}), {}),
    ("valuation leg", "refused", "R-E3", _x(y_mult={"value": 0.1, "src": "stated", "quote": "估值泡沫明显膨胀"}), {}),
    ("asset return", "refused", "R-E3", _x(us_equity={"value": -10, "src": "stated", "quote": "估值泡沫明显膨胀"}), {}),
    ("start state", "refused", "R-STATE", _x(s_unrate={"value": 5, "src": "stated", "quote": "就业基本稳定"}), {}),
    ("two forms", "refused", "R-FORM", _x(d_oil={**OIL, "sign": "+", "size": "大幅"}), {}),
    ("no form", "refused", "R-FORM", _x(d_oil={"src": "stated", "quote": "油价一个季度内上涨约三成"}), {}),
    ("sd not positive", "refused", "R-FORM", _x(d_oil={**OIL, "sd": 0}), {}),
    ("min above max", "refused", "R-FORM",
     _x(d_unrate={"min": 0.5, "max": 0.1, "src": "stated", "quote": "就业基本稳定"}), {}),
    ("oil cause 0/1", "refused", "R-FORM",
     _x(x_oil_cause={"value": 1, "src": "stated", "quote": "产油国卷入战争，供应中断"}), {}),
    ("unknown entry key", "refused", "R-FORM", _x(d_oil={**OIL, "confidence": 0.9}), {}),
    ("yoy on IP", "refused", "R-FORM",
     _qt(c_q={"constraints": [{"kind": "yoy", "field": "d_ip", "quarters": [1], "value": 2, "quote": "经济陷入衰退"}]}), {}),
    ("bad stance", "refused", "R-FORM", _rows_qt(_row(1, fc={"stance": "很紧", "quote": "信用利差走阔"})), {}),
    ("size outside the bins", "refused", "R-SIZE",
     _x(d_gz={"sign": "+", "size": "巨大", "src": "stated_qual", "quote": "信用利差走阔"}), {}),
    ("unknown src", "refused", "R-SRC", _x(d_oil={**OIL, "src": "inferred"}), {}),
    ("words written as a number", "refused", "R-SRC",
     _x(d_gz={"value": 0.4, "src": "stated_qual", "quote": "信用利差走阔"}), {}),
    ("no quote", "refused", "R-QUOTE", _x(d_oil={"value": 26, "src": "stated"}), {}),
    ("quote not in the narrative", "refused", "R-QUOTE",
     _x(d_oil={"value": 26, "src": "stated", "quote": "油价翻了一倍"}), {}),
    ("E1 without its name", "refused", "R-E1",
     _x(d_gz={"sign": "+", "size": "明显", "src": "stated_qual", "quote": "央行加息 50 个基点"}), {}),
    ("survey without its name", "refused", "R-E1",
     _x(d_sent={"sign": "-", "size": "明显", "src": "stated_qual", "quote": "经济陷入衰退"}), {}),
    ("stance without credit", "refused", "R-E1", _rows_qt(_row(1, fc={"stance": "收紧", "quote": "就业基本稳定"})), {}),
    ("beyond the range, test", "refused", "R-RANGE", _x(d_oil={**OIL, "value": 500}), {"mode": "test"}),
    ("beyond the range, production", "warning", "W-RANGE", _x(d_oil={**OIL, "value": 500}), {}),
    ("valuation in test mode", "refused", "R-VAL",
     _qt(c_q={"valuation": [{"quarters": [1, 2], "mode": "hold", "quote": "估值泡沫明显膨胀"}]}), {"mode": "test"}),
    ("valuation in production", "warning", "W-VAL",
     _qt(c_q={"valuation": [{"quarters": [1, 2], "mode": "rerate", "direction": "+", "size": "明显",
                             "quote": "估值泡沫明显膨胀"}]}), {}),
    ("year token in a quote", "refused", "R-YEAR",
     _x(d_oil={"value": 26, "src": "stated", "quote": "油价在 2008 年上涨约三成"}),
     {"mode": "test", "narrative": "油价在 2008 年上涨约三成。"}),
    ("year token in an event", "refused", "R-YEAR",
     _qt(events=[{"text": "1990 年的战争", "quarters": [1], "channels": ["d_oil"], "generic_footprint": True,
                  "how_decomposed": "战争 → 油价"}]), {"mode": "test"}),
    ("month count in a note", "refused", "R-YEAR", _qt(notes=["样本第 640 个月起"]), {"mode": "test"}),
    ("QE without its paths", "warning", "W-QE",
     _rows_qt(_row(1, u={"u_bs": {"sign": "+", "size": "明显", "src": "stated_qual", "quote": "央行宣布扩表购买国债"}})), {}),
    ("output identity", "warning", "W-IDENT",
     _x(d_rgdp={"value": 2, "src": "stated", "quote": "实际 GDP 增长 2%"},
        d_lprod={"value": 1.5, "src": "stated", "quote": "生产率增长 1.5%"},
        d_unrate={"value": -0.2, "src": "stated", "quote": "失业率下降 0.2 个百分点"}), {}),
    ("deficit identity", "warning", "W-IDENT",
     _x(d_deficit={"sign": "+", "size": "明显", "src": "stated_qual", "quote": "赤字扩大"},
        d_transfers={"sign": "+", "size": "明显", "src": "stated_qual", "quote": "转移支付增加"}), {}),
    ("shock without an intervention", "warning", "W-U",
     _rows_qt(_row(1, u={"u_ff": {"sign": "-", "size": "明显", "src": "stated_qual", "quote": "央行按部就班地降息"}})), {}),
    ("credit stance left to the model", "warning", "W-FC", _qt(), {}),
    ("policy left to the model", "warning", "W-R1", _qt(), {}),
    ("growth left to the model", "warning", "W-R2", _qt(), {}),
    ("inflation left to the model", "warning", "W-R3", _qt(), {}),
    ("oil left to the model", "warning", "W-R4", _qt(), {}),
    ("oil without its cause", "warning", "W-OIL", _x(d_oil=OIL), {}),
    ("anchors recorded", "warning", "W-ANCHOR", _qt(c_q={"anchors": {"r_star": {"value": 0.75, "quote": "中性利率维持在目前的估计"}}}), {}),
    ("no narrative", "warning", "W-QUOTE", _x(d_oil=OIL), {"narrative": None}),
    ("event without a footprint", "warning", "W-EVENT",
     _qt(events=[{"text": "一家公司财务造假", "quarters": [1], "channels": [], "generic_footprint": False,
                  "how_decomposed": "没有通用渠道"}]), {}),
    ("regime alias", "warning", "W-REGIME",
     _qt(c_q={"regime": [{"quarters": [1, 2], "label": "衰退", "share": 1.0, "quote": "经济陷入衰退"}]}), {}),
]


@pytest.mark.parametrize("name, level, code, qt, kw", RULES, ids=[r[0] for r in RULES])
def test_every_refusal_rule_has_a_fixture(name, level, code, qt, kw):
    kw = dict(kw)
    narrative = kw.pop("narrative", NARRATIVE)
    rep = QT.validate(qt, narrative, stats=STATS, **kw)
    assert code in _codes(rep, level), (name, rep["errors"], rep["refused"], [w["code"] for w in rep["warnings"]])
    item = next(it for it in rep[{"refused": "refused", "warning": "warnings", "error": "errors"}[level]]
                if it["code"] == code)
    assert re.search(r"[一-鿿]", item["text"])                    # Chinese messages
    if level == "error":
        assert not rep["readable"] and rep["qtable"] is None
    else:
        assert rep["readable"] and QT.schema_errors(rep["qtable"]) == []
        assert rep["ok"] is (level != "refused" and not rep["refused"])
    if level == "refused" and code not in ("R-SHAPE",):
        with pytest.raises(QT.QTableError):
            QT.check(qt, narrative, stats=STATS, **kw)


def test_the_messages_of_section_7_3():
    rep = QT.validate(_x(d_nope={"value": 1, "src": "stated", "quote": "就业基本稳定"},
                         d_oil={"value": 26, "src": "stated", "quote": "油价翻了一倍"}), NARRATIVE, stats=STATS)
    texts = [it["text"] for it in rep["refused"]]
    assert "字段不存在：d_nope（没有可测的对应变量，未作输入）" in texts
    assert "引文在叙事里找不到：这个值没有文字依据" in texts
    e1 = QT.validate(_x(d_gz={"sign": "+", "size": "明显", "src": "stated_qual", "quote": "央行加息 50 个基点"}),
                     NARRATIVE, stats=STATS)["refused"][0]["text"]
    assert e1.startswith("文字没有说的市场价格和调查值不能由翻译器决定")
    e3 = QT.validate(_x(y_mult={"value": 0.1, "src": "stated", "quote": "估值泡沫明显膨胀"}), NARRATIVE,
                     stats=STATS)["refused"][0]["text"]
    assert e3.startswith("资产价格与估值倍数不是输入")
    fc = [w for w in QT.validate(_qt(), NARRATIVE, stats=STATS)["warnings"] if w["code"] == "W-FC"][0]
    assert QT.MSG["fc"] in fc["text"] and fc["quarters"] == [1, 2]
    clip = QT.validate(_x(d_oil={**OIL, "value": 500}), NARRATIVE, stats=STATS)
    assert clip["clipped"][0]["used"] == pytest.approx(80 + 0.25 * 180)          # the range widened to 1.5x
    assert clip["qtable"]["rows"][0]["x"]["d_oil"]["value"] == pytest.approx(125.0)


def test_refused_entries_leave_the_rest_of_the_table():
    qt = copy.deepcopy(APPENDIX_B)
    qt["rows"][0]["x"]["d_gz"] = {"sign": "+", "size": "巨大", "src": "stated_qual", "quote": "信用利差走阔"}
    qt["rows"][1]["x"]["d_sent"] = {"sign": "-", "size": "明显", "src": "stated_qual", "quote": "央行加息 50 个基点"}
    rep = QT.validate(qt, NARRATIVE, stats=STATS)
    assert not rep["ok"] and rep["readable"]
    assert sorted(_codes(rep, "refused")) == ["R-E1", "R-SIZE"]
    clean = rep["qtable"]
    assert "d_gz" not in clean["rows"][0]["x"] and "d_oil" in clean["rows"][0]["x"]
    assert "d_sent" not in clean["rows"][1]["x"] and clean["rows"][1]["x"]["d_ff"]["value"] == 0.5
    assert QT.validate(clean, NARRATIVE, stats=STATS)["ok"]                      # the clean form is clean


def test_user_cells_and_legacy_tables():
    qt = _x(d_gz={"value": 0.5, "src": "stated", "quote": "我的假设"})
    assert QT.validate(qt, None, source="user", stats=STATS)["ok"]                # the author states it: no E1 rule
    assert "R-E1" in _codes(QT.validate(qt, None, stats=STATS), "refused")
    legacy = _x(d_oil={"value": 26, "src": "stated"})
    rep = QT.validate(legacy, None, source="legacy", stats=STATS)
    assert rep["ok"] and any("研究轮代译" in w["text"] for w in rep["warnings"])


def test_scan_blind_finds_years_and_month_counts():
    assert QT.scan_blind({"a": ["1990 年", {"b": "第 640 个月"}], "c": "随后 6 个月"}) == [
        ("$.a[0]", "1990"), ("$.a[1].b", "640 个月")]
    assert QT.scan_blind("T0+18m") == [("$", "T0+")]


def test_constraints_need_the_panel_or_a_stack():
    with pytest.raises(QT.QTableError):
        QT.constraints(APPENDIX_B)


# ================================================================ the monthly engine's phase spec (§7.4)

def test_to_phase_spec_converts_units_merges_phases_and_drops_what_the_engine_lacks():
    rows = [_row(q, x={"d_ff": {"value": 0.6, "src": "stated", "quote": "央行加息 50 个基点"},
                       "d_cpi_goods": {"value": 6.0, "src": "stated", "quote": "商品价格明显上涨"},
                       "d_ip": {"value": -3.0, "src": "stated", "quote": "经济陷入衰退"},
                       "d_oil": {"value": 30.0, "src": "stated", "quote": "油价一个季度内上涨约三成"},
                       "x_oil_cause": {"value": "supply", "src": "stated_qual", "quote": "产油国卷入战争，供应中断"},
                       "d_gpr": {"sign": "+", "size": "大幅", "src": "stated_qual", "quote": "地缘政治风险急剧上升"}},
                 fc={"stance": "收紧", "quote": "信用利差走阔"}) for q in (1, 2)]
    rows.append(_row(3, x={"d_unrate": {"value": 0.3, "src": "stated", "quote": "就业基本稳定"}}))
    qt = _qt(rows, c_q={"constraints": [{"kind": "mean", "field": "s_unrate", "quarters": [3, 4], "value": 5.0,
                                         "quote": "就业基本稳定"}]}, n=4)
    spec, rep = QT.to_phase_spec(qt, stats=STATS)
    ph = spec["phases"]
    assert [p["months"] for p in ph] == [6, 6, 168] and ph[-1]["name"] == "回归常态" and not ph[-1]["views"]
    v = {x["var"]: x for x in ph[0]["views"]}
    assert v["f_ff"]["mean"] == pytest.approx(0.6 / 3) and v["f_cpi_goods"]["mean"] == pytest.approx(6.0 / 1200)
    assert v["f_ip"]["mean"] == pytest.approx(-3.0 / 300) and v["f_oil"]["mean"] == pytest.approx(30.0 / 300)
    assert v["f_spread"]["mean"] == pytest.approx(0.43 / 3) and v["f_gz"]["mean"] == pytest.approx(0.45 / 3)
    # quarter 3 alone joins quarter 4: its unemployment change keeps its cumulative size over the 2-quarter phase
    v2 = {x["var"]: x for x in ph[1]["views"]}
    assert v2["f_unrate"]["mean"] == pytest.approx(0.3 / 3 / 2) and v2["s_unrate"]["mean"] == pytest.approx(5.0)
    dropped = {d["field"]: d for d in rep["dropped"]}
    assert set(dropped) >= {"x_oil_cause", "d_gpr"} and dropped["d_gpr"]["text"] == QT.MONTHLY_NOTE
    assert spec["source"] == "qtable" and all(p["confidence"] == 1.0 for p in ph)


def test_to_phase_spec_respects_the_engines_phase_limit():
    rows = [_row(q, x={"d_ff": {"value": 0.25 * (q % 2 or -1), "src": "stated", "quote": "央行加息 50 个基点"}}
                 if q % 3 else {"d_unrate": {"value": 0.1, "src": "stated", "quote": "就业基本稳定"}})
            for q in range(1, 41)]
    spec, rep = QT.to_phase_spec(_qt(rows, n=40), stats=STATS)
    assert len(spec["phases"]) <= QT.MAX_PHASES and sum(p["months"] for p in spec["phases"]) == 180
    assert all(p["months"] >= 6 for p in spec["phases"])


# ================================================================ the prompt, production and test mode

def test_the_test_mode_prompt_scans_clean():
    text = QX.build_system_prompt("test")
    assert QX.prompt_scan(text) == []
    assert "footprint_quarters" not in text and "measured" not in text
    tmpl = QX.template_section("test")
    assert not re.search(r"[0-9.]+\s*~?\s*[+\-]?[0-9.]*\s*sd", tmpl)              # signs only, never sizes
    assert QX.MODE_LINE["test"] in text and QX.MODE_LINE["production"] not in text
    for line in QX.state_rows("test"):                  # percentile bands, never a level (levels date an episode)
        assert not re.search(r"[0-9]\.[0-9]", line) and ("分位" in line or line.startswith("（")), line
    assert QX.user_message("覆盖期约 6 个季度。经济温和放缓。", 6, mode="test") == \
        "请把下面的叙事翻译成季度情景表（QTable/1）。\n\n<narrative>\n覆盖期约 6 个季度。经济温和放缓。\n</narrative>"


def test_the_production_prompt_follows_section_7_2():
    text = QX.build_system_prompt("production")
    order = [text.index(h) for h in ("# 角色", "# 字段", "# 定性幅度", "# 通用模板", "# 规则", "# 季初状态",
                                     "# 示例", QX.SCHEMA_HEADER)]
    assert order == sorted(order)
    assert "G01" in text and "G35" in text and "sd" in QX.template_section("production")
    assert all(f"- {k} = " in text for k in cq.SIZE_BINS)
    schema = llm_mock.schema_from_system(text)
    assert llm_mock.call_kind(schema, "x") == "qtable"
    assert set(schema["properties"]["rows"]["items"]["properties"]["x"]["propertyNames"]["enum"]) == \
        set(qcard.x_fields(qcard.load_card()))
    for ex in QX.EXAMPLES:                       # synthetic few-shots: no years, valid against their own narratives
        assert not QT.scan_blind(ex) and QT.validate(ex["qtable"], ex["narrative"], stats=STATS)["ok"]


# ================================================================ the mock and the fixtures: repair / fallback / pause

OIL_NARRATIVE = "霍尔木兹海峡被封锁，原油供应中断，油价大涨约五成；信用利差急剧走阔，失业率上升，央行紧急降息。"


def test_mock_translation_is_a_valid_qtable_quoting_the_narrative(mock_llm):
    res = QX.translate(OIL_NARRATIVE, n_quarters=6, stats=STATS)
    assert res["status"] == "ok" and res["backend"] == "mock" and res["calls"] == 1 and not res["cached"]
    qt, rep = res["qtable"], res["validation"]
    assert rep["ok"] and qt["n_quarters"] == 6 and QT.schema_errors(qt) == []
    q1 = qt["rows"][0]
    assert q1["x"]["d_oil"]["sign"] == "+" and q1["x"]["x_oil_cause"]["value"] == "supply"
    assert q1["fc"]["stance"] == "明显收紧" and q1["u"]["u_ff"]["sign"] == "-"     # 紧急降息 is an intervention
    for e in list(q1["x"].values()) + list(q1["u"].values()) + [q1["fc"]]:
        assert e["quote"] in OIL_NARRATIVE
    assert mock_llm.calls[-1]["kind"] == "qtable"
    again = QX.translate(OIL_NARRATIVE, n_quarters=6, stats=STATS)
    assert again["calls"] == 1 and not again["cached"]                            # mock answers are never cached


def test_mock_test_mode_answers_the_minimal_qtable_unless_a_replay_is_registered():
    narr = "覆盖期约 6 个季度。油价大涨，央行加息。"
    res = QX.translate(narr, mode="test", stats=STATS)
    assert res["qtable"]["n_quarters"] == 6 and all(not r["x"] for r in res["qtable"]["rows"])
    assert all(r["fc"]["stance"] == "未说明" for r in res["qtable"]["rows"])
    recorded = copy.deepcopy(APPENDIX_B)
    key = llm_mock.register_qtable(NARRATIVE, recorded)
    assert key == llm_mock.qtable_key(NARRATIVE.replace("，", "， "))            # whitespace does not matter
    res = QX.translate(NARRATIVE, mode="test", stats=STATS)
    assert res["status"] == "ok" and res["qtable"]["rows"][0]["x"]["d_oil"]["value"] == 26


@pytest.mark.parametrize("fixture, status, kind", [
    ("malformed_json", "repaired", None), ("fenced", "ok", None), ("trailing_comma", "ok", None),
    ("refusal", "fallback", "refusal"), ("refusal_text", "fallback", "refusal"), ("timeout", "fallback", "timeout"),
    ("overloaded", "fallback", "overloaded"), ("max_tokens", "fallback", "max_tokens"),
    ("empty", "fallback", "empty")])
def test_fixtures_map_to_repair_or_fallback(mock_llm, fixture, status, kind):
    mock_llm.push(fixture, match="qtable")
    res = QX.translate(OIL_NARRATIVE, n_quarters=5, stats=STATS)
    assert res["status"] == status, res["error"]
    if status == "repaired":
        assert res["calls"] == 2 and res["repair"]["problems"] and res["repair"]["ok"]
        assert res["validation"]["ok"] and res["qtable"]["rows"][0]["x"]
    if status == "fallback":
        assert res["error"]["kind"] == kind and res["error"]["action"] == "fallback"
        qt = res["qtable"]
        assert qt["n_quarters"] == 5 and all(not r["x"] for r in qt["rows"])     # everything left to the model
        assert any("翻译没有得到可用的情景表" in n for n in qt["notes"])
        assert res["validation"]["ok"] and "W-FC" in _codes(res["validation"], "warning")


@pytest.mark.parametrize("fixture, kind", [("usage_limit", "limit"), ("usage_limit_text", "limit"),
                                           ("weekly_limit", "limit"), ("login", "login"),
                                           ("cli_missing", "cli_missing"), ("model_missing", "model")])
def test_usage_limit_login_and_cli_failures_pause(mock_llm, fixture, kind):
    mock_llm.push(fixture, match="qtable")
    with pytest.raises(QX.QTablePause) as err:
        QX.translate(OIL_NARRATIVE, stats=STATS)
    assert err.value.kind == kind and re.search(r"[一-鿿]", str(err.value))
    if fixture == "usage_limit":
        assert err.value.reset_at and err.value.reset_at > time.time()


def test_a_form_problem_triggers_one_repair_call(mock_llm):
    def bad_answer(kind, system, user, answer):
        answer["rows"][0]["x"]["d_nope"] = {"value": 1, "src": "stated", "quote": "失业率上升"}
        return json.dumps(answer, ensure_ascii=False)
    mock_llm.push(bad_answer, match="qtable")
    res = QX.translate(OIL_NARRATIVE, stats=STATS)
    assert res["calls"] == 2 and res["repair"]["problems"] and "字段不存在" in res["repair"]["problems"][0]
    assert [c["kind"] for c in mock_llm.calls][-2:] == ["qtable", "repair"]
    assert "d_nope" not in res["qtable"]["rows"][0]["x"] and res["qtable"]["rows"][0]["x"]["d_oil"]


def test_estimate_makes_no_call(mock_llm):
    est = QX.estimate(OIL_NARRATIVE)
    assert est["backend"] == "mock" and est["calls"] == 0 and not est["needs_confirmation"]
    assert est["input_tokens"] > 5000 and est["output_tokens"] == [3000, 8000]
    assert not mock_llm.calls
    real = QX.estimate(OIL_NARRATIVE, backend="claude_code")
    assert real["calls"] == 1 and real["max_calls"] == 2 and real["needs_confirmation"]


def test_the_api_backend_is_not_offered_for_qtables(monkeypatch):
    from scenariolab.llm import translator as tr
    with pytest.raises(tr.TranslatorError, match="不走 Anthropic API"):
        QX.translate(OIL_NARRATIVE, backend="api")


# ================================================================ the API: validate, translate, built-ins

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from scenariolab.server import app
    return TestClient(app, raise_server_exceptions=False, base_url="http://127.0.0.1")


def test_api_validate_estimate_and_translate(client, mock_llm):
    r = client.post("/api/q/validate", json={"qtable": APPENDIX_B, "narrative": NARRATIVE})
    assert r.status_code == 200 and r.json()["ok"] and r.json()["qtable"]["n_quarters"] == 4
    r = client.post("/api/q/validate", json={"qtable": APPENDIX_B, "narrative": "别的叙事", "mode": "test"})
    assert r.status_code == 200 and not r.json()["ok"] and "R-QUOTE" in _codes(r.json(), "refused")
    assert client.post("/api/q/validate", json={"qtable": APPENDIX_B, "source": "nope"}).status_code == 400
    est = client.post("/api/q/translate/estimate", json={"narrative": OIL_NARRATIVE}).json()
    assert est["backend"] == "mock" and est["calls"] == 0
    r = client.post("/api/q/translate", json={"narrative": OIL_NARRATIVE, "n_quarters": 4})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok" and r.json()["qtable"]["n_quarters"] == 4 and r.json()["validation"]["ok"]
    assert client.post("/api/q/translate", json={"narrative": "  "}).status_code == 400
    mock_llm.push("usage_limit", match="qtable")
    r = client.post("/api/q/translate", json={"narrative": OIL_NARRATIVE})
    assert r.status_code == 503 and r.json()["pause"] == "limit" and r.json()["reset_at"]


def test_api_translate_needs_this_machine_and_a_confirmation_for_a_real_backend(client, monkeypatch, mock_llm):
    from fastapi.testclient import TestClient

    from scenariolab.server import app
    other = TestClient(app, raise_server_exceptions=False)               # Host: testserver
    assert other.post("/api/q/translate", json={"narrative": OIL_NARRATIVE}).status_code == 403
    r = client.post("/api/q/translate", json={"narrative": OIL_NARRATIVE},
                    headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    monkeypatch.setenv("MSL_LLM_BACKEND", "claude_code")
    r = client.post("/api/q/translate", json={"narrative": OIL_NARRATIVE})
    assert r.status_code == 409 and r.json()["estimate"]["needs_confirmation"] and "confirm" in r.json()["detail"]
    assert not mock_llm.calls                                             # nothing was called


def test_api_lists_the_built_in_qtables(client):
    r = client.get("/api/q/scenarios")
    assert r.status_code == 200
    rows = r.json()["scenarios"]
    assert len(rows) == 13 and all(s["ok"] for s in rows) and {s["id"] for s in rows} >= {"oil_war", "baseline"}
    one = client.get("/api/q/scenarios/oil_war").json()
    assert one["qtable"]["n_quarters"] == 60 and one["validation"]["ok"] and one["review"]
    assert client.get("/api/q/scenarios/nope").status_code == 404
    assert client.get("/api/q/scenarios/../x").status_code in (400, 404)


# ================================================================ the built-in scenarios as QTables (§7.4)

def _tool():
    spec = importlib.util.spec_from_file_location("qtable_from_yaml", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_the_built_in_qtables_are_reviewed_conversions_of_the_yaml_specs():
    for sid in QT.builtin_ids():
        env = QT.builtin(sid)
        path = SPECS / f"{sid}.yaml"
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        # not stale: regenerate with python tools/qtable_from_yaml.py when a YAML spec changes
        assert env["source"]["yaml_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), sid
        rep = QT.validate(env["qtable"], env["text"], mode="production", source="builtin")
        assert rep["ok"], (sid, QT.first_problem(rep))
        assert env["qtable"]["n_quarters"] == sum(p["months"] for p in spec["phases"]) // 3 == 60
        # every view of every phase is accounted for (converted, a stance, or dropped with a reason)
        views = [(k, v["var"]) for k, ph in enumerate(spec["phases"], 1) for v in ph.get("views") or []]
        recorded = [(c["phase"], c["var"]) for c in env["conversion"] if c["var"] in {v for _, v in views}]
        assert sorted(recorded) == sorted(views), sid
        assert all(c.get("note") for c in env["conversion"] if c["status"] == "dropped")
        assert env["review"] and all(r.get("why") for r in env["review"])          # the human review is recorded
        for q in env["qtable"]["rows"]:
            assert q["fc"]["stance"] in cq.FC_STANCES


@needs_qpanel
def test_qtable_from_yaml_reproduces_the_committed_files():
    tool = _tool()
    envs = tool.build()
    assert len(envs) == 13
    for sid, env in envs.items():
        assert (QT.QTABLES_DIR / f"{sid}.json").read_text(encoding="utf-8") == tool.dump(env), sid


@needs_qpanel
@needs_results
def test_yaml_views_are_delivered_on_the_quarterly_stack():
    """The converted QTables deliver the YAML's view means within the monthly engine's own delivery tolerance
    (reported per view); the exceptions are documented in the report: statements the factor layer saturates at
    +-4 sd, and spread views held to their stance plan."""
    tool = _tool()
    envs = {sid: QT.builtin(sid) for sid in QT.builtin_ids()}
    rep = tool.delivery(envs, n_paths=400, seed=1)
    bad = [(sid, r) for sid, s in rep["scenarios"].items() for r in s["views"] if not r["ok"]]
    assert not bad, bad
    n_views = sum(len(s["views"]) for s in rep["scenarios"].values())
    assert n_views >= 90
    held = [r for s in rep["scenarios"].values() for r in s["views"] if r.get("held_to") is not None
            and r["gap"] > r["tolerance"]]
    assert all(r.get("reason") for r in held)                    # every miss of the YAML number has its cause
    assert all(s["excluded_paths"] < 0.25 * 400 for s in rep["scenarios"].values())


# ================================================================ mock end to end: narrative -> QTable -> run -> report

@needs_qpanel
def test_mock_end_to_end_narrative_qtable_run_report(client):
    from scenariolab.qmodel import rollout as R
    R.get_stack()
    tr = client.post("/api/q/translate", json={"narrative": OIL_NARRATIVE, "n_quarters": 4}).json()
    assert tr["status"] == "ok" and tr["validation"]["ok"]
    r = client.post("/api/q/run", json={"qtable": tr["qtable"], "narrative": OIL_NARRATIVE, "n_paths": 300,
                                        "seed": 2})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    for _ in range(600):
        s = client.get(f"/api/q/run/{jid}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done", s
    res = s["result"]
    assert res["label"] == cq.TRIAL_LABEL and len(res["quarters"]) == 4
    v = res["validation"]
    assert v["ok"] and {m["code"] for m in v["missing"]} >= {"W-R3"}
    codes = {n["code"] for n in res["qtable_notes"]}
    assert "W-FC" in codes and "W-R3" in codes
    assert res["factors"]["d_oil"]["p50"][0] > 0 and res["levels"]["spread"]["p50"][0] > res["levels"]["spread"]["p10"][0]
    # the quotes are checked against the narrative: a QTable that does not quote it is refused
    bad = client.post("/api/q/run", json={"qtable": tr["qtable"], "narrative": "别的叙事", "n_paths": 300})
    assert bad.status_code == 400 and "引文在叙事里找不到" in bad.json()["detail"]


@needs_qpanel
def test_a_built_in_scenario_runs_by_id(client):
    r = client.post("/api/q/run", json={"scenario": "oil_war", "n_paths": 200, "seed": 1, "mode": "step"})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    for _ in range(600):
        s = client.get(f"/api/q/run/{jid}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done" and len(s["result"]["quarters"]) == 1
    assert client.post("/api/q/run", json={"scenario": "nope"}).status_code == 404


@needs_qpanel
def test_constraints_read_the_clean_qtable_into_statements():
    from scenariolab.qmodel import rollout as R
    stack = R.get_stack()
    read = QT.constraints(APPENDIX_B, stack=stack, narrative=NARRATIVE)
    st = {(s.var, s.quarters[0]): s for s in read["statements"]}
    sd = QT.stats_table(stack.Q, stack.train_end)
    assert st[("d_gpr", 1)].value == pytest.approx(2 * sd["d_gpr"]["sd"])          # 大幅 = 2 training sd
    assert st[("d_gpr", 1)].sd == pytest.approx(0.25 * sd["d_gpr"]["sd"])
    assert st[("d_gz", 1)].value == pytest.approx(sd["d_gz"]["sd"])               # 收紧 = +1 sd on d_gz, d_spread
    assert st[("d_spread", 3)].value == 0.0                                        # 中性
    assert st[("x_oil_cause", 1)].value == 1.0 and st[("d_ff", 2)].sd is None
    assert read["anchors"]["r_star"]["value"] == 0.75 and any(n["code"] == "W-ANCHOR" for n in read["notes"])
    regime = _qt(c_q={"regime": [{"quarters": [1, 2], "label": "stagflation", "share": 0.8, "quote": "经济陷入衰退"}]})
    read = QT.constraints(regime, stack=stack, narrative=NARRATIVE)
    assert read["cycle_odds"] == {"contraction": 0.8, "quarters": [1, 2]}
    infl = [s for s in read["statements"] if s.var == "cpi_yoy"][0]
    assert infl.lo == QT.HIGH_INFL and infl.agg == "mean"
    mean_ff = _qt(c_q={"constraints": [{"kind": "mean", "field": "s_ff", "quarters": [1, 2], "value": 3.0,
                                        "quote": "央行加息 50 个基点"}]})
    assert [s.var for s in QT.constraints(mean_ff, stack=stack, narrative=NARRATIVE)["statements"]] == ["ff_shadow"]
    bad = copy.deepcopy(APPENDIX_B)
    bad["rows"][0]["x"]["d_nope"] = {"value": 1, "src": "stated", "quote": "就业基本稳定"}
    with pytest.raises(QT.QTableError, match="字段不存在"):
        QT.constraints(bad, stack=stack, narrative=NARRATIVE)                    # strict: the API's reading
    loose = QT.constraints(bad, stack=stack, narrative=NARRATIVE, strict=False)  # the clean table runs
    assert loose["report"]["refused"] and all(s.var != "d_nope" for s in loose["statements"])
    with pytest.raises(QT.QTableError):
        QT.constraints({"rows": []}, stack=stack, strict=False)


@needs_qpanel
def test_to_phase_spec_passes_the_monthly_validator():
    from scenariolab.data import panel as monthly_panel
    from scenariolab.qmodel import panel as qpanel
    from scenariolab.qmodel import rollout as R
    from scenariolab.scenarios.schema import validate_spec
    Q = qpanel.load_panel()
    frame = monthly_panel.load_panel()
    for qt in (APPENDIX_B, QT.builtin("financial_crisis")["qtable"]):
        spec, rep = QT.to_phase_spec(qt, Q=Q, train_end=R.last_data_quarter(Q))
        clean, report = validate_spec(spec, frame)
        assert not report["errors"], report["errors"]
        assert sum(p["months"] for p in clean["phases"]) == config.HORIZON_MONTHS


# ================================================================ review fixes: numbers need numbers, N3, ranges, words

def test_a_number_marked_stated_or_derived_needs_a_number_in_its_quote():
    """§3.4 R-SRC: stated = a number in the text, derived = arithmetic on stated numbers. A value or range whose quote
    carries no numeral (digits, 百分之…, 三成, 五十个基点, 一半) is refused; a sign + size from the same words passes."""
    nar = "信用利差明显走阔，失业上升，央行降息。油价上涨约三成，失业率上升 0.4 个百分点，信用利差扩大一半。"
    bad = _qt(rows=[_row(1, x={"d_gz": {"value": 2.4, "src": "stated", "quote": "信用利差明显走阔"},
                               "d_unrate": {"value": 1.7, "src": "derived", "quote": "失业上升"}})], n=1)
    rep = QT.validate(bad, nar, mode="test", stats=STATS)
    assert _codes(rep, "refused") == ["R-SRC", "R-SRC"] and rep["qtable"]["rows"][0]["x"] == {}
    assert all("引文里没有数字" in r["text"] for r in rep["refused"])
    good = _qt(rows=[_row(1, x={"d_oil": {"value": 26, "src": "stated", "quote": "油价上涨约三成"},
                                "d_unrate": {"value": 0.4, "src": "stated", "quote": "失业率上升 0.4 个百分点"},
                                "d_gz": {"sign": "+", "size": "明显", "src": "stated_qual", "quote": "信用利差明显走阔"},
                                "d_spread": {"value": 0.4, "src": "derived", "quote": "信用利差扩大一半"}})], n=1)
    rep = QT.validate(good, nar, mode="test", stats=STATS)
    assert rep["ok"], rep["refused"]
    short = _qt(rows=[_row(1, x={"d_gz": {"sign": "+", "size": "明显", "src": "stated_qual", "quote": "利"}})], n=1)
    assert _codes(QT.validate(short, nar, stats=STATS), "refused") == ["R-QUOTE"]
    # the rule is the translator's: the lab user's own cells and the reviewed built-ins keep their numbers
    assert QT.validate(bad, nar, source="user", stats=STATS)["ok"]


def test_n3_takes_market_prices_and_surveys_only_as_size_bins():
    """§8.4: in N3 (numbers=False), E1 / E1c / E2 entries are size bins with quotes: a translator's number is refused
    (R-N3); the research round's stand-in numbers are re-expressed as the nearest bin (W-N3, recorded)."""
    nar = "信用利差大约走阔 2 个百分点，失业率上升 0.4 个百分点。"
    qt = _qt(rows=[_row(1, x={"d_gz": {"value": 0.9, "src": "stated", "quote": "信用利差大约走阔 2 个百分点"},
                              "d_unrate": {"value": 0.4, "src": "stated", "quote": "失业率上升 0.4 个百分点"}})], n=1)
    assert QT.validate(qt, nar, mode="test", stats=STATS)["ok"]                    # N1 / N2: numbers allowed
    rep = QT.validate(qt, nar, mode="test", stats=STATS, numbers=False)
    assert _codes(rep, "refused") == ["R-N3"] and set(rep["qtable"]["rows"][0]["x"]) == {"d_unrate"}
    rep = QT.validate(qt, None, mode="test", source="legacy", stats=STATS, numbers=False)
    e = rep["qtable"]["rows"][0]["x"]["d_gz"]
    assert rep["ok"] and e == {"sign": "+", "size": "大幅", "src": "stated_qual", "quote": "信用利差大约走阔 2 个百分点"}
    assert rep["binned"] == [{"q": 1, "field": "d_gz", "asked": 0.9, "sign": "+", "size": "大幅"}]
    assert "W-N3" in _codes(rep, "warning")
    cons = _qt(c_q={"constraints": [{"kind": "sum", "field": "d_gz", "quarters": [1, 2], "value": 1.0,
                                     "quote": "信用利差大约走阔 2 个百分点"}]})
    assert _codes(QT.validate(cons, nar, mode="test", stats=STATS, numbers=False), "refused") == ["R-N3"]


def test_policy_shocks_and_yoy_constraints_are_range_checked():
    """§7.3 range rule: a u_* shock beyond ±4 shock sd (the factor layer's saturation) and a y/y constraint beyond
    1.5 x its level's range are refused in test mode and clipped in production, so both engines see the same value."""
    stats = {**STATS, "s_cpi_yoy": {"sd": 0.5, "lo": -2.0, "hi": 14.8}}
    nar = "央行紧急降息 40 个百分点。CPI 同比达到 40%。"
    qt = _qt(rows=[_row(1, u={"u_ff": {"value": -40, "src": "stated", "quote": "央行紧急降息 40 个百分点"}}), _row(2)],
             c_q={"constraints": [{"kind": "yoy", "field": "s_cpi_yoy", "quarters": [1, 2], "value": 40,
                                   "quote": "CPI 同比达到 40%"}]})
    rep = QT.validate(qt, nar, mode="test", stats=stats)
    assert _codes(rep, "refused") == ["R-RANGE", "R-RANGE"]
    assert "u" not in rep["qtable"]["rows"][0] and "constraints" not in rep["qtable"]["c_q"]
    rep = QT.validate(qt, nar, mode="production", stats=stats)
    assert rep["ok"] and rep["qtable"]["rows"][0]["u"]["u_ff"]["value"] == pytest.approx(-4 * 1.07)
    assert rep["qtable"]["c_q"]["constraints"][0]["value"] == pytest.approx(6.4 + 1.5 * 8.4)
    assert _codes(rep, "warning").count("W-RANGE") == 2


def test_an_unstated_stance_quote_and_valuation_numbers_are_checked():
    """A 未说明 stance's quote goes through the quote rules (a year token in test mode is refused, the stance stays
    未说明 without it); a number inside a valuation statement is refused; unknown c_q keys are reported."""
    nar = "信用利差明显走阔。"
    qt = _qt(rows=[_row(1, fc={"stance": "未说明", "quote": "这像 2008 年的雷曼时刻"})], n=1)
    rep = QT.validate(qt, nar, mode="test", stats=STATS)
    assert _codes(rep, "refused") == ["R-QUOTE"] and rep["qtable"]["rows"][0]["fc"] == {"stance": "未说明"}
    assert QT.scan_blind(rep["qtable"]) == []
    val = _qt(n=1, c_q={"valuation": [{"mode": "rerate", "direction": "-", "size": "明显", "quarters": [1], "value": -0.3,
                                       "quote": "信用利差明显走阔"}]})
    rep = QT.validate(val, nar, mode="production", stats=STATS)
    assert _codes(rep, "refused") == ["R-VAL"] and "不写数字" in rep["refused"][0]["text"]
    extra = _qt(n=2, c_q={"regime": [{"label": "recession", "share": 0.6, "quarters": [1, 2], "quote": "信用利差明显走阔",
                                      "why": "x"}]})
    rep = QT.validate(extra, nar, stats=STATS)
    assert rep["ok"] and "W-FORM" in _codes(rep, "warning") and "why" not in rep["qtable"]["c_q"]["regime"][0]


def test_reader_messages_name_fields_in_chinese():
    """Validation messages and their 'where' name a known field by its Chinese name, never by its code id."""
    nar = "油价上涨约三成。央行宣布扩表。实际 GDP 增长 2%，生产率增长 1.5%，失业率下降 0.2 个百分点。"
    qt = _qt(rows=[_row(1, x={"d_oil": {"value": 26, "src": "stated", "quote": "油价上涨约三成"},
                              "d_rgdp": {"value": 2, "src": "stated", "quote": "实际 GDP 增长 2%"},
                              "d_lprod": {"value": 1.5, "src": "stated", "quote": "生产率增长 1.5%"},
                              "d_unrate": {"value": -0.2, "src": "stated", "quote": "失业率下降 0.2 个百分点"}},
                   u={"u_bs": {"sign": "+", "size": "明显", "src": "stated_qual", "quote": "央行宣布扩表"}})], n=1)
    rep = QT.validate(qt, nar, stats=STATS)
    code_id = re.compile(r"(?<![A-Za-z0-9])(?:[dsuxa]_[a-z0-9_]+)")
    texts = [f"{w['where']} {w['text']}" for w in rep["warnings"] + rep["refused"]]
    assert {"W-OIL", "W-QE", "W-IDENT"} <= set(_codes(rep, "warning"))
    assert texts and not [t for t in texts if code_id.search(t)], texts
