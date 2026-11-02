"""P8a (SIM_DESIGN §10.1, §10.2, §14 P8): the tabs 情景条件检验 (evaluation) and 变量卡片 (card).

Static checks of the two web modules (routing, deep links, no text through innerHTML, the wording rules), the
evaluation view on fixture records shaped like §14 P5 describes (尚无记录, registered, results appended in the record or
beside it, a failed and a passed switch gate, an unreadable file), the factor layer's computed checks, the card view
(readable texts, today's values, evidence), the research ablation's check values (§2 E1) and the endpoints.
Data-dependent checks skip cleanly without data/; nothing here calls an LLM or downloads anything."""
import hashlib
import json
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scenariolab import config
from scenariolab.evaluation import evalview as EV
from scenariolab.qmodel import cardview as CV
from scenariolab.qmodel import config_q as cq
from scenariolab.server import app

client = TestClient(app, raise_server_exceptions=False)
ROOT = config.ROOT
WEB = ROOT / "web" / "js"
TABS = WEB / "tabs"
MINE = ["evaluation.js", "card.js"]
PANEL = cq.PANEL_Q_PATH
needs_panel = pytest.mark.skipif(not PANEL.exists(), reason="quarterly panel not built (data/ is not in CI)")
needs_qmodel = pytest.mark.skipif(not cq.QMODEL_RESULTS_PATH.exists(), reason="qmodel.json not built")
TN_FILE = ROOT / "scenariolab" / "evaluation" / "narrative" / "preregistered" / "2026-11-02_TN.json"
ALLOWED_IDS = {"x_q", "s_q", "c_q"}                 # the spec's notation for the three inputs, not code ids
CODE_ID = re.compile(r"(?<![A-Za-z0-9_])[a-z]{1,2}_[a-z0-9_]+")


def src(name: str) -> str:
    return (TABS / name).read_text(encoding="utf-8")


# ================================================================ routing and deep links

def test_the_two_tabs_are_routed_in_their_groups():
    """情景条件检验 is the first tab of 验证, 变量卡片 sits in 说明 after 方法与数据; every earlier tab keeps its id and
    order, so existing links still work; the modules are served and revalidated."""
    app_js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "{ group: '验证' },\n  { id: 'evaluation', name: '情景条件检验' },\n  { id: 'forecast'," in app_js
    assert "{ group: '说明' },\n  { id: 'method', name: '方法与数据' },\n  { id: 'card', name: '变量卡片' },\n];" in app_js
    ids = re.findall(r"\{ id: '([a-z_]+)', name: '", app_js)
    assert ids == ["overview", "eras", "regimes", "causality", "evaluation", "forecast", "backtest", "diagnostics",
                   "valuation", "scenarios", "lab", "instruments", "agents", "portfolio", "withdrawal", "method", "card"]
    for name in MINE:
        r = client.get(f"/js/tabs/{name}")
        assert r.status_code == 200 and r.headers.get("cache-control") == "no-cache", name
        assert "export async function render(root, ctx)" in r.text, name


def test_the_new_deep_links():
    """#evaluation?<section> for the seven sections A-G, #evaluation?run=<run id> for a narrative run, #card?<field>;
    the lab's cell editor links its field to the card, and its QTable headers carry the card's tooltip."""
    ev = src("evaluation.js")
    for key in ("conclusion", "oracle", "narrative", "valuation", "engine", "ledger", "records"):
        assert f"['{key}', " in ev, key
    assert "const runParam = /^run=([A-Za-z0-9_]{1,80})$/.exec(param || '');" in ev
    assert "return key && sections[key] ? sections[key] : null;" in ev
    card = src("card.js")
    assert "const want = param && ID_RE.test(param) ? param : null;" in card
    assert "export function fieldLabel(id, text, fallback = null)" in card and "export function attachCardTip(el, id, fallback = null)" in card
    qlab = src("_qlab.js")
    assert "import { fieldLabel } from './card.js';" in qlab
    assert "const head = (f) => (cardRows[f] ? fieldLabel(f, zh(f), cardRows[f]) : h('span', {}, zh(f)));" in qlab
    assert "h('a', { href: `#card?${encodeURIComponent(f)}` }, '变量卡片 →')" in qlab
    # the diagnostics page links the evaluation tab once it is in the nav (P8b's hook)
    assert "document.querySelector('#nav a[href=\"#evaluation\"]')" in src("diagnostics.js")


# ================================================================ no text reaches innerHTML

def test_the_new_modules_set_text_never_markup():
    for name in MINE:
        text = "\n".join(line for line in src(name).splitlines() if not line.lstrip().startswith("//"))
        assert "innerHTML" not in text and "outerHTML" not in text and "insertAdjacentHTML" not in text, name
        assert not re.search(r"[{,]\s*html\s*:", text), name
        assert "document.write" not in text, name
    ev = src("evaluation.js")
    # raw records and the critic's lines are text in a <pre>; small multiples and sparklines are SVG nodes
    assert "h('pre', { class: 'code ev-wrap' }, JSON.stringify(full.record, null, 2))" in ev
    assert "document.createElementNS(SVG, tag)" in ev and "document.createElementNS(SVG, tag)" in src("card.js")
    assert "for (const c of kids) if (c) el.append(typeof c === 'string' ? document.createTextNode(c) : c);" in ev


# ================================================================ wording rules (§10.6, §8.6)

ANECDOTE_RE = re.compile(r"12[–-]20 个季度|(?<![前第\d])(?:12|20) 个季度|\b(?:12|20)Q\b")


def test_the_wording_rules():
    """「预测」 only for the probabilities (Layer 1) and the Fed's own forecast; 12-20Q figures carry the anecdote
    label; 估值假设 carries 「相当于直接假设股价」; the claims' sentences are never written by the page (they come from
    the report, which builds them from the pre-registered record)."""
    allowed = ("预测能力", "预测只在情景概率里", "美联储当时的预测", "美联储预测", "不作预测量", "不是预测量")
    files = {name: src(name) for name in MINE}
    for py in ("scenariolab/evaluation/evalview.py", "scenariolab/qmodel/cardview.py"):
        files[py] = (ROOT / py).read_text(encoding="utf-8")
    for name, text in files.items():
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith(("//", "#")):
                continue
            if "预测" in line:
                rest = line
                for a in allowed:
                    rest = rest.replace(a, "")
                assert "预测" not in rest, (name, i + 1, line.strip())
            if name in MINE and ANECDOTE_RE.search(line):        # the backend marks those rows (anecdote: true)
                window = " ".join(lines[max(0, i - 1):i + 3])
                assert "只能当个案" in window or "ANECDOTE" in window, (name, i + 1, line.strip())
            if "估值假设" in line:
                assert "相当于直接假设股价" in " ".join(lines[i:i + 3]), (name, i + 1)
    ev = src("evaluation.js")
    assert "export const ANECDOTE = '每个十年只有 2–3 个独立窗口，只能当个案';" in ev
    for claim in ("叙事带来了改进", "金融条件承载了结果", "季度模型比月度引擎好"):
        assert claim not in ev, claim
    a = ev.split("function sectionA(ev, rep) {")[1].split("\n}\n")[0]
    assert "rep.headline" in a and "claims" not in a
    # a real run's estimate needs the tick before confirm: true is sent
    assert "tick.addEventListener('change', () => { go.disabled = !tick.checked; });" in ev
    assert "api.post('evaluation/narrative/run', { estimate_id: e.id, confirm: true })" in ev
    assert "'用真实大模型重跑叙事检验…'" in ev


# ================================================================ the evaluation view on fixtures

def _ta(results=True, nested=False):
    rec = {"id": "TA", "test": "T-A", "version": 1, "date": "2026-11-03", "title": "资产层候选的预注册检验（T-A）",
           "frozen": "在任何一次运行之前写定。",
           "protocol": {"selection": {"origins": ["1990Q1", "2014Q4"], "text": "每年 1 月重新拟合"},
                        "test": {"train_end": "2014Q4", "span": ["2015Q1", "2026Q3"]}},
           "candidates": [{"id": "C1", "text": "F 之外的一级字段"}, {"id": "NN", "text": "神经网络", "report_only": True}],
           "thresholds": {"adopt_dr2_1q_lo90": 0.0}, "seeds": {"bootstrap": 1}, "hashes": {"data": "a" * 64, "card": "b" * 64, "code": "c" * 40}}
    if not results:
        return rec
    if nested:
        test = {"L0": {"1": {"r2": 0.44, "lo90": 0.3, "hi90": 0.55, "crpss": 0.1, "cov80": 0.8, "n_indep": 47},
                       "12": {"r2": 0.1, "lo90": -0.2, "hi90": 0.3, "n_indep": 3}},
                "NN": {"1Q": {"r2": 0.18, "lo90": 0.05, "hi90": 0.3, "n_indep": 47}}}
        rec["results"] = [{"date": "2026-11-03", "asset_layer": "L0", "adopted": [], "test": test}]
    else:
        rec["results"] = [{"date": "2026-11-03T14:00:00", "status": "done", "asset_layer": "L0", "adopted": [], "passed": False,
                           "rows": [{"model": "L0", "horizon": "1Q", "r2": 0.44, "r2_lo90": 0.3, "r2_hi90": 0.55, "crpss": 0.1, "cov80": 0.8, "n_indep": 47},
                                    {"model": "L0", "horizon": "20Q", "r2": -0.1, "r2_lo90": -0.4, "r2_hi90": 0.2, "n_indep": 2},
                                    {"model": "NN", "horizon": "1Q", "r2": 0.18, "r2_lo90": 0.05, "r2_hi90": 0.3, "n_indep": 47}],
                           "conditions": [{"id": "dr2_1q", "text": "ΔR²（1 个季度）的 90% 下界 > 0", "diff": 0.004, "lo90": -0.02, "hi90": 0.03, "passed": False}]}]
    return rec


def _ts(passed: bool):
    conds = [{"id": "TF", "text": "T-F 通过", "passed": passed},
             {"id": "narrative", "text": "叙事臂：季度模型 − 月度引擎的上界 < +0.03", "diff": 0.021, "lo90": 0.004, "hi90": 0.041,
              "bound": 0.03, "passed": passed}]
    return {"id": "TS", "date": "2026-11-05", "title": "切换关（T-S）", "results": [{"date": "2026-11-06", "conditions": conds}]}


def _ts_evaluated(rid: str):
    """A T-S record with a passed result in the switch's own format (P9, evaluation/switch.py): every condition the
    protocol requires, each passed, made against this record's protocol and today's quarterly-stack version."""
    from scenariolab.evaluation import switch as SW
    rec = {"id": "TS", "date": "2026-11-05", "title": "切换关（T-S）", "thresholds": {"oracle_hi90_below": 0.02}}
    proto = SW.DEFAULT_PROTOCOL
    conds = [{"id": "TF", "group": "tf", "text": "因子引擎检验 T-F 全部通过", "evaluated": True, "passed": True}]
    for cid in SW.required_ids(proto)[1:]:
        if cid == "PIT":
            conds.append({"id": cid, "group": "calibration", "text": "校准", "evaluated": True, "passed": True,
                          "q": 0.26, "e": 0.25, "bound": 0.02})
        else:
            grp = "oracle" if cid.startswith("O:") else "narrative"
            b = 0.02 if grp == "oracle" else 0.03
            conds.append({"id": cid, "group": grp, "text": cid, "evaluated": True, "passed": True, "diff": -0.01,
                          "lo90": -0.03, "hi90": 0.005, "bound": b, "n": 14})
    res = {"schema": SW.SCHEMA, "status": "passed", "passed": True, "evaluated_at": "2026-11-06T12:00:00Z",
           "required": SW.required_ids(proto), "conditions": conds, "checks": [],
           "record": {"id": rid, "protocol_sha256": SW.protocol_digest(rec)},
           "model": {"semantic": SW.current_model_version()}}
    return {**rec, "results": [res]}


@pytest.fixture
def fx(monkeypatch, tmp_path):
    """The view pointed at empty fixture folders (records, results) and a run list without real runs."""
    pre, res = tmp_path / "prereg", tmp_path / "results"
    pre.mkdir()
    res.mkdir()
    monkeypatch.setattr(EV, "PREREG_DIRS", [pre])
    monkeypatch.setattr(EV, "RESULTS_DIR", res)
    monkeypatch.setattr(EV, "narrative_runs", lambda: {"headline_run": None, "headline_kind": None, "n_real": 0, "n_mock": 0, "n_runs": 0})

    def write(name, obj):
        p = pre / name
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        return p
    return {"pre": pre, "res": res, "write": write}


def _parts(s):
    return {p["key"]: p["text"] for p in s["conclusion"]["parts"]}


def test_no_records_shows_the_empty_state(fx):
    s = EV.summary()
    assert s["schema"] == "evaluation-view/1" and s["records"] == []
    assert {t: v["status"] for t, v in s["tests"].items()} == {"TA": "missing", "TC": "missing", "TS": "missing", "TN": "missing"}
    assert s["asset_layer"] is None and s["conditional"] is None and s["anatomy"] is None and s["factor_layer"] is None
    assert s["switch_gate"]["status"] == "not_evaluated" and s["switch_gate"]["text"] == EV.SWITCH_DEFAULT
    p = _parts(s)
    assert "T-A 尚无记录" in p["asset_layer"] and "因子引擎检验尚无结果" in p["factor_layer"]
    assert "真实大模型的叙事检验还没有按盲测规范运行过" in p["narrative_missing"]
    assert s["ablation"] and s["ablation"]["schema"] == "oracle-ablation/1"          # the committed research card
    assert len(s["contamination_notes"]) >= 3


def test_records_registered_then_done(fx):
    fx["write"]("2026-11-03_TA.json", _ta(results=False))
    fx["write"]("2026-11-03_TC.json", {"id": "TC", "date": "2026-11-03"})
    s = EV.summary()
    assert s["tests"]["TA"]["status"] == "registered" and s["tests"]["TC"]["status"] == "registered"
    assert s["asset_layer"] is None and "已预注册（2026-11-03），还没有结果" in _parts(s)["asset_layer"]
    rec = next(r for r in s["records"] if r["test"] == "TA")
    raw = (fx["pre"] / "2026-11-03_TA.json").read_bytes()
    assert rec["sha256"] == hashlib.sha256(raw).hexdigest() and rec["file"].endswith("2026-11-03_TA.json")
    assert rec["selection"].startswith("选择：1990Q1–2014Q4") and "检验：2015Q1–2026Q3；训练到 2014Q4" in rec["selection"]
    assert [c["report_only"] for c in rec["candidates"]] == [False, True]
    assert rec["hashes"]["code"] == "c" * 40 and rec["n_results"] == 0 and rec["last_result"] is None
    # the run appends its results: the section-B table is normalized (Chinese model names, report-only, anecdotes)
    fx["write"]("2026-11-03_TA.json", _ta())
    s = EV.summary()
    al = s["asset_layer"]
    assert s["tests"]["TA"]["status"] == "done" and al["asset_layer"] == "L0" and al["adopted"] == []
    rows = {(r["model"], r["horizon"]): r for r in al["rows"]}
    assert rows[("L0", "1Q")]["r2"] == 0.44 and rows[("L0", "1Q")]["lo90"] == 0.3 and rows[("L0", "1Q")]["n_indep"] == 47
    assert rows[("L0", "20Q")]["anecdote"] and not rows[("L0", "1Q")]["anecdote"]
    assert rows[("NN", "1Q")]["report_only"] and "只作对照" in rows[("NN", "1Q")]["model_zh"]
    assert al["conditions"][0]["passed"] is False and al["conditions"][0]["detail"] == "+0.004 [-0.020, +0.030]"
    assert "没有候选通过" in _parts(s)["asset_layer"]


def test_a_nested_result_table_and_results_beside_the_record(fx):
    fx["write"]("2026-11-03_TA.json", _ta(nested=True))
    al = EV.summary()["asset_layer"]
    assert [(r["model"], r["horizon"]) for r in al["rows"]] == [("L0", "1Q"), ("L0", "12Q"), ("NN", "1Q")]
    # results appended to a sibling file (append-only JSON lines; a torn line is skipped)
    fx["write"]("2026-11-04_TC.json", {"id": "TC", "date": "2026-11-04"})
    (fx["pre"] / "2026-11-04_TC.results.jsonl").write_text(
        json.dumps({"date": "2026-11-04", "passed": True, "text": "关卡通过"}) + "\n{torn\n", encoding="utf-8")
    s = EV.summary()
    tc = next(r for r in s["records"] if r["test"] == "TC")
    assert tc["n_results"] == 1 and tc["status"] == "done" and tc["last_result"]["passed"] is True
    assert all(not r["id"].endswith(".results") for r in s["records"])


def test_the_switch_gate_names_the_failed_conditions(fx):
    fx["write"]("2026-11-05_TS.json", _ts(passed=False))
    g = EV.summary()["switch_gate"]
    assert g["status"] == "failed" and g["passed"] is False and g["source"] == "2026-11-05_TS"
    assert g["text"].startswith("切换关 T-S 没有通过，未满足的条件：「T-F 通过」；「叙事臂：")
    assert "（+0.021 [+0.004, +0.041]，界线 +0.030）" in g["text"] and "默认引擎仍是月度引擎" in g["text"]
    # P9: a passed result counts only in the switch's own format; a hand-written one shows but switches nothing
    fx["write"]("2026-11-06_TS.json", _ts(passed=True))
    g = EV.summary()["switch_gate"]
    assert g["status"] == "invalid" and g["passed"] is False and "不据此切换" in g["text"]
    fx["write"]("2026-11-07_TS.json", _ts_evaluated("2026-11-07_TS"))       # a newer record wins
    g = EV.summary()["switch_gate"]
    assert g["status"] == "passed" and g["passed"] and "通过（2026-11-06）" in g["text"]


def test_the_switch_gate_reads_only_evaluations(fx):
    """P9: forecast_mix.json's copy of the gate is derived, never a source; a hand-written switch_gate.json that says
    passed is refused (status invalid: nothing switches); one that says failed reads as failed."""
    (fx["res"] / "forecast_mix.json").write_text(json.dumps({"switch_gate": {"status": "passed", "passed": True,
                                                                               "text": "通过了"}}), encoding="utf-8")
    g = EV.summary()["switch_gate"]
    assert g["status"] == "not_evaluated" and g["text"] == EV.SWITCH_DEFAULT and g["reasons"]
    (fx["res"] / "switch_gate.json").write_text(json.dumps({"passed": True, "text": "通过了"}), encoding="utf-8")
    g = EV.summary()["switch_gate"]
    assert g["status"] == "invalid" and g["passed"] is False and g["source"] == "switch_gate.json"
    (fx["res"] / "switch_gate.json").write_text(json.dumps({"passed": False, "text": "没通过"}), encoding="utf-8")
    g = EV.summary()["switch_gate"]
    assert g["status"] == "failed" and g["text"] == "没通过"


def test_unreadable_and_unknown_records(fx):
    fx["write"]("2026-11-04_TA.json", "{not json")
    s = EV.summary()
    r = s["records"][0]
    assert r["status"] == "unreadable" and r["test"] == "TA" and s["tests"]["TA"]["status"] == "unreadable"
    assert EV.record("2026-11-04_TA")["record"] is None
    with pytest.raises(KeyError):
        EV.record("2026-11-09_nope")
    with pytest.raises(KeyError):
        EV.record("../evil")
    # newest first within a test
    fx["write"]("2026-11-08_TA.json", {**_ta(results=False), "date": "2026-11-08"})
    assert [x["id"] for x in EV.records() if x["test"] == "TA"] == ["2026-11-08_TA", "2026-11-04_TA"]
    assert EV.summary()["tests"]["TA"]["status"] == "registered"                # the newest record decides


def test_the_evaluation_step_tables_pass_through(fx):
    ev = {"conditional": {"text": "合并", "rows": [{"arm": "OB", "horizon": "4Q", "r2": 0.2, "r2_lo90": 0.1, "r2_hi90": 0.3, "n_indep": 41}]},
          "anatomy": [{"group": "1985-2026", "h_q": 4, "row": "placebo", "n": 160, "corr_pred_dm": None}]}
    (fx["res"] / "evaluation.json").write_text(json.dumps(ev), encoding="utf-8")
    s = EV.summary()
    assert s["conditional"]["rows"][0]["model_zh"].startswith("O-B") and s["conditional"]["text"] == "合并"
    assert s["anatomy"]["rows"][0]["row"] == "placebo"


def test_the_headline_run_is_real_or_research_never_mock(monkeypatch):
    from scenariolab.evaluation.narrative import report as RP
    monkeypatch.setattr(RP, "runs", lambda: [{"id": "n_mock", "kind": "mock", "status": "done", "has_report": True}])
    assert EV.narrative_runs()["headline_run"] is None
    monkeypatch.setattr(RP, "runs", lambda: [{"id": "research_2026_11_01", "kind": "research", "status": "done"},
                                             {"id": "n_mock", "kind": "mock", "status": "done", "has_report": True},
                                             {"id": "n_real", "kind": "real", "status": "done", "has_report": True}])
    n = EV.narrative_runs()
    assert n["headline_run"] == "n_real" and n["headline_kind"] == "real" and n["n_mock"] == 1


def _qmodel_fixture():
    ppc = {"F2_cpi_gt4": {"history": 0.33, "p05": 0.06, "p50": 0.41, "p95": 0.83, "pass": True},
           "F6_bounds": {"ff_min": 0.1, "u_min": 2.0, "u_below3": 0.05, "cpi_gt15": 0.07, "pass": False}}
    fits = {"2014Q4": {"anchors": "train_end", "origin": "2015Q1", "ppc": ppc},
            "2026Q3": {"anchors": "forward", "origin": "2026Q4", "ppc": ppc,
                       "F1": {"n_paths": 10, "hard_max_abs_error_sci": "1e-14", "pass": True,
                              "delivery": [{"var": "d_gz", "kind": "soft", "label": "fc 收紧", "share_within_1sd": 0.7}]}}}
    return {"factor_layer": {"created": "2026-11-02", "notes": ["只给出通过/未通过（§14 P3）；冻结的记录在 P5。", "历史值用不重叠窗口。"],
                             "T_F": {"fits": fits, "F7": {"4": {"r2": {"gz": 0.37}, "checks": {"gz": True}, "pass": True},
                                                          "12": {"r2": {"gz": 0.5}, "pass": False}},
                                     "summary": {"2026Q3:F6_bounds": False, "2026Q3:F2_cpi_gt4": True, "F7": False}},
                             "T_C": {"n_validation_origins": 80, "gate": {"E1_r2": {"d_tp10": -0.021, "d_gz": 0.01}, "pass": False},
                                     "selection": {"completion_w": 0.25, "regime_cov": {"adopted": False}},
                                     "test": {"b": {"d_gz": {"cls": "E1", "n": 47, "r2_vs_uncond": 0.01, "crps": 0.2, "crps_uncond": 0.21, "cov80": 0.6}}}}}}


def test_the_factor_layer_summary_is_reader_text(fx):
    (fx["res"] / "qmodel.json").write_text(json.dumps(_qmodel_fixture(), ensure_ascii=False), encoding="utf-8")
    s = EV.summary()
    fl = s["factor_layer"]
    assert [r["id"] for r in fl["tf_rows"]] == ["F2_cpi_gt4", "F6_bounds"]
    assert fl["tf_rows"][0]["zh"] == "CPI 同比 > 4% 的季度占比" and fl["tf_rows"][1]["passed"] is False
    assert fl["notes"] == ["历史值用不重叠窗口。"]                                  # no phase or section ids
    assert fl["f1"]["delivery"][0]["label"] == "金融条件 收紧" and fl["f1"]["delivery"][0]["field"] == "GZ 信用利差变动"
    assert fl["f7"]["12"]["anecdote"] and not fl["f7"]["4"]["anecdote"]
    assert fl["tc"]["validation"].startswith("验证：每年第 4 季") and "（80 个起点）" in fl["tc"]["validation"]
    assert fl["summary"] == {"total": 3, "passed": 1, "failed": ["2026Q3:F6_bounds", "F7"]}
    text = _parts(s)["factor_layer"]
    assert "1/3 项通过；没有通过的：边界、补全技能" in text and "10 年期限溢价变动（ACM）的补全 R² 低于 −0.02，关卡未通过" in text
    for p in s["conclusion"]["parts"]:                                              # nothing claims an improvement
        assert "叙事带来了改进" not in p["text"] and "季度模型比月度引擎好" not in p["text"]
    assert not re.search(r"§|(?<![A-Za-z])P\d", json.dumps(fl, ensure_ascii=False))


@needs_qmodel
def test_the_factor_layer_reads_the_built_file():
    fl = EV.factor_layer()
    assert fl and [f["end"] for f in fl["fits"]] == list(cq.TF_ENDS)
    assert all(r["zh"] and r["fits"] for r in fl["tf_rows"])
    assert fl["tc"]["gate_rule"].startswith("情形 b")
    assert not re.search(r"§|(?<![A-Za-z])P\d", json.dumps(fl, ensure_ascii=False))


# ================================================================ the research ablation (§2 E1)

def test_the_research_ablation_reproduces_the_spec():
    """The committed oracle_ablation.json (tools/import_oracle_ablation.py) holds §2 E1's pooled rolling mean10 R² at
    1/4/12/20 quarters within 0.0005, with 167 / 41 / 13 / 8 independent windows and the critic's paired differences."""
    d = json.loads(EV.ABLATION_PATH.read_text(encoding="utf-8"))
    want = {"F": (0.226, 0.223, -0.007, -0.030), "F_noGZ": (0.128, 0.118, -0.119, -0.133),
            "F_nocredit": (-0.003, -0.016, -0.258, -0.285), "MACRO5": (-0.081, -0.145, -0.373, -0.421),
            "GZonly": (0.165, 0.149, 0.110, 0.108)}
    got = {m["id"]: m for m in d["models"]}
    for mid, vals in want.items():
        for h, v in zip((1, 4, 12, 20), vals):
            c = got[mid]["cells"][str(h)]
            assert abs(c["r2"] - v) <= 0.0005, (mid, h)
            assert c["lo90"] < c["r2"] < c["hi90"], (mid, h)
    assert [got["F"]["cells"][str(h)]["n_indep"] for h in (1, 4, 12, 20)] == [167, 41, 13, 8]
    assert all(c["ok"] for c in d["checks"]) and len(d["checks"]) == 20
    pair = {(p["b"], p["h"]): p for p in d["paired"]}
    assert pair[("F_nocredit", 1)]["diff"] == 0.229 and pair[("F_nocredit", 1)]["lo90"] == 0.12
    assert len(d["sources"]) == 4 and all(len(v) == 64 for v in d["sources"].values())
    assert "只能当个案" in " ".join(d["notes"]) and not CODE_ID.search(d["text"])


@pytest.mark.skipif(not os.environ.get("MSL_RESEARCH_DIR"), reason="needs the research folder (MSL_RESEARCH_DIR)")
def test_the_research_ablation_is_rebuilt_from_the_archive():
    import importlib.util
    spec = importlib.util.spec_from_file_location("import_oracle_ablation", ROOT / "tools" / "import_oracle_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--from", os.environ["MSL_RESEARCH_DIR"], "--check"]) == 0


# ================================================================ the card view

def _reader_texts(row):
    keys = ("zh", "slot", "lead", "timing", "endo", "endo_short", "route", "route_short", "translator",
            "translator_short", "fc", "unit", "lag", "note", "definition", "family", "group_zh")
    out = [str(row.get(k) or "") for k in keys] + [e["text"] for e in row["evidence"]]
    if row.get("today"):
        out += [row["today"]["text"], row["today"]["when"]]
    return out


def test_readable_replaces_card_ids():
    names = {"d_tp10": "10 年期限溢价变动（ACM）", "d_y2": "2 年期收益率变动", "u_bs": "资产负债表冲击"}
    assert CV.readable("资产效应只经由 d_tp10、d_y2 的干预路径", names) == "资产效应只经由「10 年期限溢价变动（ACM）」、「2 年期收益率变动」的干预路径"
    assert CV.readable("x_q（政策，经冲击 u_bs）", names) == "x_q（政策，经冲击「资产负债表冲击」）"
    assert CV.readable("（FWT D1 修正）；MAIN 拼接版；与 qdata 的约定", names) == "（因子轮 D1 修正）；原应用拼接版；与研究轮的数据脚本的约定"
    assert CV.readable("情景估值只能写 c_q.valuation 的定性模式", names) == "情景估值只能写情景表的估值陈述的定性模式"


def test_the_card_view_without_data(monkeypatch):
    """card.json is committed: the view works without the panel (no series, the anchors' defaults), every reader text
    is free of code ids, the filters cover the 101 rows, and the evidence links point at existing pages."""
    monkeypatch.setattr(CV, "_panel", lambda: (None, {}))
    monkeypatch.setattr(cq, "QMODEL_RESULTS_PATH", Path("/nonexistent/qmodel.json"))
    v = CV._build()
    assert v["schema"] == "card-view/1" and v["n_rows"] == 101 and v["today"] is None
    assert [(g["id"], g["n"]) for g in v["groups"]] == [("x", 52), ("u", 5), ("s", 22), ("c", 5), ("y", 5), ("v", 12)]
    for key, opts in v["filters"].items():
        assert sum(o["n"] for o in opts) == 101, key
    left = {(r["id"], m) for r in v["rows"] for t in _reader_texts(r) for m in CODE_ID.findall(t) if m not in ALLOWED_IDS}
    assert not left, sorted(left)[:10]
    rows = {r["id"]: r for r in v["rows"]}
    assert all(r["series"] is None for r in v["rows"])
    assert rows["a_spread"]["today"]["text"] == "2.00 pp" and rows["a_ustar"]["today"]["text"] == "4.80%"
    assert rows["d_gz"]["route_kind"] == "L0" and rows["d_gz"]["fc_role"] == "required_one"
    assert {e["href"] for e in rows["d_gz"]["evidence"]} >= {"#diagnostics", "#evaluation?oracle"}
    assert rows["d_igrea"]["route_kind"] == "candidate" and rows["d_igrea"]["evidence"][0]["href"] == "#evaluation?records"
    assert rows["s_cape"]["evidence"][0]["href"] == "#valuation?quarterly"
    assert rows["u_ff"]["route_kind"] == "shock" and rows["d_sent"]["permission"] == "stated_only" and rows["d_sent"]["keywords"]
    pages = set(re.findall(r"\{ id: '([a-z_]+)'", (WEB / "app.js").read_text(encoding="utf-8")))
    for r in v["rows"]:
        for e in r["evidence"]:
            assert e["href"].lstrip("#").split("?")[0] in pages, (r["id"], e["href"])


@needs_panel
def test_the_card_view_reads_today_from_the_panel():
    v = CV._build()
    rows = {r["id"]: r for r in v["rows"]}
    cape = rows["s_cape"]["today"]
    assert cape["text"] == "CAPE 41.2 倍" and 0.9 < cape["pct"] <= 1.0 and cape["kind"] == "start"
    gz = rows["d_gz"]
    assert gz["series"]["n"] >= 200 and gz["series"]["first"] == "1973Q2" and gz["today"]["kind"] == "realized"
    assert len(gz["series"]["values"]) >= gz["series"]["n"] and gz["today"]["quarter"] <= v["today"]["last_data_quarter"]
    assert rows["x_oil_cause"]["today"]["pct"] is None and rows["x_oil_cause"]["today"]["text"] in ("需求推动", "供给与需求混合", "供给冲击")
    assert rows["y_assets"]["series"]["label"] == "美股的季度对数价格回报"
    assert any(e["kind"] == "tc" for e in gz["evidence"]) or not cq.QMODEL_RESULTS_PATH.exists()


# ================================================================ endpoints

def test_the_endpoints():
    r = client.get("/api/evaluation")                       # not taken by /api/{name}
    assert r.status_code == 200 and r.json()["schema"] == "evaluation-view/1"
    tn = next((x for x in r.json()["records"] if x["id"] == "2026-11-02_TN"), None)
    assert tn and tn["sha256"] == hashlib.sha256(TN_FILE.read_bytes()).hexdigest() and tn["test"] == "TN"
    r = client.get("/api/evaluation/records/2026-11-02_TN")
    assert r.status_code == 200 and r.json()["record"]["id"] == "TN"
    r = client.get("/api/evaluation/records/2026-01-01_nope")
    assert r.status_code == 404 and "找不到预注册记录" in r.json()["detail"]
    r = client.get("/api/q/card/view")
    assert r.status_code == 200 and r.json()["n_rows"] == 101
    assert client.get("/api/evaluation/narrative").status_code == 200        # P7's router is still reachable
