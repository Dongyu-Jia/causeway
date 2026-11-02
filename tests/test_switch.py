"""P9 (SIM_DESIGN §4.6, §14 P9): the switch gate T-S and the switch every default reads.

The evaluator runs on fixtures only: a frozen T-S record shaped like §14 P5 describes, a computed T-F (qmodel.json) and a
narrative-test run directory with both engines' score rows (what P7's scorer writes). Nothing here fits a model,
simulates, downloads or calls an LLM, and nothing reads P5's or P7's real records: the real evaluation runs after the
merge (`python -m scenariolab.evaluation.switch evaluate --run latest --write --append`).

Covered: a passing fixture switches every default; each condition failing on its own (the oracle arms, calibration,
the narrative arms, T-F) switches nothing and names its numbers; missing inputs fail closed; the record's protocol in
its three shapes; stale, inconsistent and hand-written results; the append-only result file; the endpoint, the CLI,
the build hook and the pages that render the reasons."""
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scenariolab import config, jsonio
from scenariolab.analysis import scenario_mix as SM
from scenariolab.evaluation import evalview as EV
from scenariolab.evaluation import switch as SW
from scenariolab.evaluation.narrative import common as NC
from scenariolab.portfolio import service
from scenariolab.server import app

client = TestClient(app, raise_server_exceptions=False)
TABS = config.ROOT / "web" / "js" / "tabs"
EQ9 = list(NC.EQUITY9)
CLEAN = [f"X{i:02d}" for i in range(1, 15)]              # 14 clean episodes, as §8.2
WITH_NARRATIVES = CLEAN                                   # v2 narratives for every clean episode (§8.2)
V1_NARRATIVES = CLEAN[:10]                                # the v1 narratives (research round) cover 10 of them
SELECTED = "base+add_gz+add_tp10_x_core4+drop_labor_share+energy_metals_oil+gold_window"
ARMS = ["U", "OA", "OB", "OF", "N1", "N2", "N3"]
TF_KEYS = [f"{e}:{k}" for e in ("1994Q4", "2004Q4", "2014Q4", "2026Q3")
           for k in ("F2_cpi_gt4", "F2_cpi_acf4", "F3_per_15y", "F5_okun", "F6_bounds")] + ["F7", "F8"]
TS_RECORD = {"id": "TS", "test": "T-S", "version": 1, "date": "2026-11-05", "title": "切换关（T-S）的预注册记录",
             "frozen": "在评估之前写定。", "protocol": {"text": "§4.6 的四个条件，干净集"},
             "thresholds": {"oracle_hi90_below": 0.02, "narrative_hi90_below": 0.03, "pit_margin": 0.02},
             "seeds": {"bootstrap": 20261102}, "hashes": {"card": "c" * 64}}


def src(name: str) -> str:
    return (TABS / name).read_text(encoding="utf-8")


# ================================================================ fixtures

def _rows(delta=None, pit_q=0.25, pit_e=0.25, engines=("Q", "E"), drop=None, noise=0.002, narrated=None):
    """Score rows for 14 clean episodes: E's |error| is a base level, Q's is that plus delta[(arm, horizon)] plus a small
    per-episode term (so the bootstrap interval has width); |PIT − 0.5| is pit_q / pit_e on every cell."""
    delta = delta or {}
    out = []
    for i, code in enumerate(CLEAN):
        eps = noise * ((i % 3) - 1)
        for arm in ARMS:
            if arm.startswith("N") and code not in (WITH_NARRATIVES if narrated is None else narrated):
                continue
            for eng in engines:
                if drop and (code, arm, eng) in drop:
                    continue
                for hz, q in (("4Q", 4), ("end", 6)):
                    for j, a in enumerate(EQ9):
                        base = 0.10 + 0.01 * i + 0.001 * j
                        ae = base + (delta.get((arm, hz), 0.0) + eps if eng == "Q" else 0.0)
                        dev = pit_q if eng == "Q" else pit_e
                        pit = 0.5 + (dev if (i + j) % 2 else -dev)
                        out.append({"code": code, "set": "clean", "bust": i < 5, "engine": eng, "arm": arm,
                                    "variant": f"{eng}:{arm}", "cma": "historical", "asset": a, "horizon": hz,
                                    "quarters": q, "y": 0.0, "q10": -0.1, "q50": ae, "q90": 0.1, "mean": ae,
                                    "err": ae, "abs_err": ae, "width": 0.2, "crps": ae, "degenerate": False,
                                    "cov80": 1.0, "pit": pit})
    return pd.DataFrame(out)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fixture folders for the record, the results, the narrative runs and the registry; returns helpers."""
    pre, res, runs = tmp_path / "prereg", tmp_path / "results", tmp_path / "runs"
    for d in (pre, res, runs):
        d.mkdir()
    monkeypatch.setattr(EV, "PREREG_DIRS", [pre])
    monkeypatch.setattr(EV, "RESULTS_DIR", res)
    monkeypatch.setattr(SW, "RESULTS_DIR", res)
    monkeypatch.setattr(NC, "RUNS_DIR", runs)
    monkeypatch.setattr(service, "RESULTS_DIR", res)
    reg = {"clean": list(CLEAN),
           "narratives": {c: ({"v2": ["n1", "n2", "n3"]} | ({"v1": ["n1", "n2", "n3"]} if c in V1_NARRATIVES else {}))
                          for c in CLEAN},
           "standin": {c: {"v1": ["n1", "n2", "n3"]} for c in V1_NARRATIVES}}
    monkeypatch.setattr(SW, "_registry", lambda: reg)
    SW.clear_cache()
    service.clear_cache()
    jsonio.dump({"spec": {"name": SELECTED}}, res / "model_selection.json")

    def record(obj=None, name="2026-11-05_TS.json"):
        p = pre / name
        p.write_text(json.dumps(TS_RECORD if obj is None else obj, ensure_ascii=False), encoding="utf-8")
        return p

    def tf(failed=()):
        summ = {k: k not in failed for k in TF_KEYS}
        jsonio.dump({"factor_layer": {"created": "2026-11-02 04:35", "quick": False, "T_F": {"summary": summ}}},
                    res / "qmodel.json")

    def run(rid="n_20261105_100000_abcd", rows=None, kind="real", status="done", model=SELECTED, stack=None,
            sources="llm", engines=("Q", "E"), created="2026-11-05T10:00:00", narratives=None):
        version = narratives or ("v1" if kind == "mock" else "v2")
        narrated = V1_NARRATIVES if version == "v1" else WITH_NARRATIVES
        d = runs / rid
        (d / "translations").mkdir(parents=True)
        R = _rows(narrated=narrated) if rows is None else rows
        R.to_csv(d / "score_rows.csv", index=False)
        sem = stack or f"{SW.current_model_version()}/2000Q4/train_end/s0/b200/abc"
        eps = {c: {"stack": sem, "engine_E": {"model": model}} for c in CLEAN}
        (d / "episodes.json").write_text(json.dumps(eps), encoding="utf-8")
        (d / "stats.json").write_text(json.dumps({"settings": {"engines": list(engines)}}), encoding="utf-8")
        (d / "run.json").write_text(json.dumps({"id": rid, "status": status, "kind": kind,
                                                "backend": "mock" if kind == "mock" else "claude_code",
                                                "created": created, "settings": {"narratives": version,
                                                                                 "engines": list(engines)}}),
                                    encoding="utf-8")
        for c in narrated:
            for arm in ("N1", "N2", "N3"):
                rec = {"code": c, "arm": arm, "backend": "mock", "qtable": {"schema": "qtable/1"}}
                rec.update({"standin": {"status": "standin", "source_kind": "legacy"},
                            "llm": {"status": "ok", "backend": "claude_code"},
                            "fallback": {"status": "fallback"},
                            "mock": {"status": "ok"}}[sources])
                (d / "translations" / f"{c}__{arm}.json").write_text(json.dumps(rec), encoding="utf-8")
        return rid
    return {"pre": pre, "res": res, "runs": runs, "record": record, "tf": tf, "run": run}


def _passing(env, **kw):
    env["record"]()
    env["tf"]()
    return env["run"](**kw)


def _by_id(res):
    return {c["id"]: c for c in res["conditions"] + res["checks"]}


# ================================================================ the gate

def test_a_passing_fixture_passes_every_condition(env):
    rid = _passing(env)
    res = SW.evaluate(rid)
    assert res["schema"] == SW.SCHEMA and res["passed"] is True and res["status"] == "passed"
    assert res["incomplete"] is False and res["reasons"] == []
    c = _by_id(res)
    assert set(SW.required_ids(SW.DEFAULT_PROTOCOL)) <= set(c)
    assert len([k for k in c if k.startswith("O:")]) == 6 and len([k for k in c if k.startswith("N:")]) == 6
    ob = c["O:OB:end"]
    assert ob["evaluated"] and ob["passed"] and ob["n"] == 14 and ob["hi90"] < 0.02 and ob["bound"] == 0.02
    assert c["N:N2:4Q"]["n"] == 14 and c["N:N2:4Q"]["bound"] == 0.03          # every clean episode (v2)
    assert c["PIT"]["q"] == pytest.approx(0.25) and c["PIT"]["e"] == pytest.approx(0.25) and c["PIT"]["passed"]
    assert c["TF"]["passed"] and c["IN:model"]["passed"] and c["IN:sources"]["passed"]
    assert res["protocol_source"] == "record" and res["record"]["id"] == "2026-11-05_TS"
    assert res["model"]["semantic"] == SW.current_model_version() and res["translations"] == {"llm": 42}
    assert res["text"].startswith("切换关 T-S 通过（") and "默认引擎换成季度模型" in res["text"]
    assert "预测" not in json.dumps(res, ensure_ascii=False)


def test_the_comparisons_are_the_narrative_tests_own_statistics(env):
    """Each paired condition is narrative/stats.paired on the stored rows (episodes as the unit, 2000 draws), so the
    numbers equal what P7's stats.json would carry for Q − E."""
    from scenariolab.evaluation.narrative import stats as ST
    R = _rows(delta={("OB", "end"): 0.015})
    rid = _passing(env, rows=R)
    res = SW.evaluate(rid)
    p = ST.paired(R, ("Q:OB", "historical"), ("E:OB", "historical"), "end", "equity9", codes=CLEAN)
    c = _by_id(res)["O:OB:end"]
    assert (c["diff"], c["lo90"], c["hi90"], c["n"]) == (p["diff"], p["lo90"], p["hi90"], p["n_episodes"])
    noise = np.mean([0.002 * ((i % 3) - 1) for i in range(len(CLEAN))])     # the fixture's per-episode term
    assert c["diff"] == pytest.approx(0.015 + noise, abs=1e-6)


@pytest.mark.parametrize("case, cid, words", [
    ("oracle", "O:OB:end", "已知宏观与金融市场（O-B）·窗口末"),
    ("narrative", "N:N2:4Q", "叙事 N2（加金融条件）·4 个季度"),
    ("pit", "PIT", "校准"),
    ("tf", "TF", "因子引擎检验 T-F"),
])
def test_each_failing_condition_switches_nothing_and_names_its_numbers(env, case, cid, words):
    env["record"]()
    env["tf"](failed=("2014Q4:F6_bounds", "F7") if case == "tf" else ())
    rows = {"oracle": _rows(delta={("OB", "end"): 0.031}), "narrative": _rows(delta={("N2", "4Q"): 0.045}),
            "pit": _rows(pit_q=0.30, pit_e=0.25), "tf": _rows()}[case]
    rid = env["run"](rows=rows)
    res = SW.evaluate(rid)
    c = _by_id(res)
    assert res["passed"] is False and res["status"] == "failed" and c[cid]["passed"] is False
    assert [k for k, v in c.items() if v.get("evaluated") and v.get("passed") is False] == [cid]
    line = next(r for r in res["reasons"] if words in r)
    assert line.startswith("未满足：")
    if case == "oracle":
        assert "+0.031" in line and "不低于 +0.020" in line and "14 个片段" in line
    if case == "narrative":
        assert "+0.045" in line and "不低于 +0.030" in line and "14 个片段" in line
    if case == "pit":
        assert "季度模型 0.300，月度引擎 0.250" in line and "超过 +0.020" in line
    if case == "tf":
        assert "2 / 22 项没有通过" in line and "边界（2014Q4）" in line and "补全技能" in line
    assert "默认引擎仍是月度引擎" in res["text"] and words in res["text"]
    # written and read back: nothing switches, the reasons travel with the state
    SW.write_result(res, results_dir=env["res"])
    st = SW.state(env["res"])
    assert st["status"] == "failed" and st["quarterly"] is False and st["defaults"] == {k: "monthly" for k in SW.CONSUMERS}
    assert any(words in r for r in st["reasons"]) and st["label"] == "试运行"
    assert service.resolve_set(None)["id"] == "monthly"
    assert SM.switch_gate(env["res"])["passed"] is False
    g = EV.summary()["switch_gate"]
    assert g["status"] == "failed" and any(words in r for r in g["reasons"])
    body = client.get("/api/evaluation/switch").json()
    assert body["status"] == "failed" and body["quarterly"] is False and any(words in r for r in body["reasons"])


def test_missing_inputs_fail_closed(env):
    env["record"]()
    env["tf"]()
    # one engine only
    res = SW.evaluate(env["run"]("n_q_only", rows=_rows(engines=("Q",)), engines=("Q",)))
    c = _by_id(res)
    assert not res["passed"] and res["incomplete"] and c["IN:engines"]["passed"] is False
    assert c["O:OA:4Q"]["evaluated"] is False and "没有两个引擎都有的片段" in c["O:OA:4Q"]["detail"]
    # E skipped one clean episode on O-B: that comparison does not cover the clean set
    drop = {("X07", "OB", "E")}
    res = SW.evaluate(env["run"]("n_gap", rows=_rows(drop=drop)))
    c = _by_id(res)
    assert not res["passed"] and c["O:OB:4Q"]["evaluated"] is False and c["O:OB:4Q"]["missing"] == ["X07"]
    assert c["O:OA:4Q"]["passed"] is True
    # the engine arm ran the research round's base model
    res = SW.evaluate(env["run"]("n_base", model="base"))
    assert not res["passed"] and _by_id(res)["IN:model"]["passed"] is False
    # narrative tables that are the minimal fallback, or mock-generated: not translations
    for k in ("fallback", "mock"):
        res = SW.evaluate(env["run"](f"n_{k}", sources=k))
        cc = _by_id(res)["IN:sources"]
        assert not res["passed"] and cc["passed"] is False and "不能用的" in cc["detail"]
    # a real run's LLM translations are accepted
    assert SW.evaluate(env["run"]("n_llm", kind="real", sources="llm"))["passed"] is True
    # a real run whose narratives cover 10 of the 14 clean episodes: the narrative comparisons do not cover the set
    res = SW.evaluate(env["run"]("n_v1", kind="real", sources="llm", narratives="v1"))
    cn = _by_id(res)["N:N2:4Q"]
    assert not res["passed"] and cn["evaluated"] is False and cn["missing"] == CLEAN[10:]
    # two quarterly-stack versions in one run
    res = SW.evaluate(env["run"]("n_mixed"))
    eps = json.loads((env["runs"] / "n_mixed" / "episodes.json").read_text())
    eps["X03"]["stack"] = "qmodel-r9-f1-a1/2000Q4"
    (env["runs"] / "n_mixed" / "episodes.json").write_text(json.dumps(eps))
    assert _by_id(SW.evaluate("n_mixed"))["IN:stack"]["passed"] is False
    # an unfinished run
    res = SW.evaluate(env["run"]("n_running", status="scoring"))
    assert _by_id(res)["IN:run"]["passed"] is False and not res["passed"]


def test_a_mock_build_is_a_pipeline_check_that_cannot_pass(env):
    """The token-free build (mock backend + the research round's stand-ins, non-blind, no quotes, 10 of 14 clean
    episodes) never certifies the narrative arms: they are not evaluated, T-S fails closed, and the result is not
    appended to the frozen record's results; the oracle arms are still computed."""
    env["record"]()
    env["tf"]()
    rid = env["run"]("n_mock", kind="mock", sources="standin")
    assert SW.latest_run() == rid
    res = SW.evaluate(rid)
    c = _by_id(res)
    assert res["passed"] is False and res["incomplete"] is True
    for k in ("N:N1:4Q", "N:N2:end", "N:N3:4Q"):
        assert c[k]["evaluated"] is False and c[k]["detail"] == SW.MOCK_RUN_TEXT
    assert c["O:OB:end"]["passed"] is True and c["IN:run"]["passed"] is True
    assert any(SW.MOCK_RUN_TEXT in r for r in res["reasons"])
    with pytest.raises(SW.SwitchError, match="流程检验"):
        SW.write_result(res, results_dir=env["res"], append=True)
    SW.write_result(res, results_dir=env["res"])                    # written for the page: it shows why
    assert SW.state(env["res"])["quarterly"] is False
    # a real run is preferred as "latest" over a newer mock run
    real = env["run"]("n_real", created="2026-11-04T10:00:00")
    assert SW.latest_run() == real


def test_a_record_that_accepts_stand_ins_and_partial_coverage_says_so(env):
    """Stand-ins count only when the frozen record lists them (narrative_sources), and partial coverage only when it
    says narrative_coverage "with_narratives": the condition then names how many clean episodes it covers."""
    env["record"]({**TS_RECORD, "protocol": {"narrative_sources": ["llm", "standin"],
                                             "narrative_coverage": "with_narratives"}})
    env["tf"]()
    res = SW.evaluate(env["run"]("n_mock", kind="mock", sources="standin"))
    c = _by_id(res)
    assert res["passed"] is True and c["N:N2:4Q"]["n"] == 10
    assert "覆盖 10 / 14 个干净片段" in c["N:N2:4Q"]["detail"]
    assert any("只覆盖 10 / 14" in n for n in res["notes"])


def test_no_record_means_no_decision(env):
    env["tf"]()
    rid = env["run"]()
    res = SW.evaluate(rid)
    assert res["passed"] is False and _by_id(res)["IN:record"]["evaluated"] is False
    assert all(c["passed"] for c in res["conditions"])             # every number passes, yet nothing can switch
    with pytest.raises(SW.SwitchError, match="没有冻结的 T-S 预注册记录"):
        SW.write_result(res, results_dir=env["res"])
    assert not (env["res"] / SW.GATE_FILE).exists()


def test_latest_picks_the_newest_run_with_both_engines(env):
    _passing(env, rid="n_20261105_090000_aaaa", created="2026-11-05T09:00:00")
    env["run"]("n_20261105_110000_bbbb", rows=_rows(engines=("Q",)), engines=("Q",), created="2026-11-05T11:00:00")
    assert SW.latest_run() == "n_20261105_090000_aaaa"
    assert SW.evaluate("latest")["run"]["id"] == "n_20261105_090000_aaaa"
    with pytest.raises(SW.SwitchError):
        SW.evaluate("../etc")


# ================================================================ the record's protocol

def test_the_protocol_comes_from_the_record_in_its_usual_shapes():
    proto, info = SW.protocol({"data": {"thresholds": {"oracle_margin": 0.015, "narrative": {"hi90_below": 0.025}}}})
    assert proto["oracle_hi90_below"] == 0.015 and proto["narrative_hi90_below"] == 0.025 and proto["pit_margin"] == 0.02
    assert info["source"] == "record" and "pit_margin" in info["defaults"] and not info["problems"]
    assert any("与设计默认的" in n for n in info["notes"])
    # the T-N record's way: comparisons listed as conditions
    conds = [{"a": f"Q:{a}", "b": f"E:{a}", "assets": "equity9", "set": "clean", "horizon": h, "hi90_below": m}
             for a, m in (("OA", 0.02), ("OB", 0.02), ("OF", 0.02), ("N1", 0.03), ("N2", 0.03), ("N3", 0.03))
             for h in ("4Q", "end")] + [{"id": "PIT", "text": "calibration", "pit_margin": 0.02}, {"id": "TF"}]
    proto, info = SW.protocol({"data": {"conditions": conds}})
    assert proto["oracle_arms"] == ["OA", "OB", "OF"] and proto["narrative_arms"] == ["N1", "N2", "N3"]
    assert proto["horizons"] == ["4Q", "end"] and not info["problems"] and info["source"] == "record"
    # words instead of numbers keep the spec's numbers, with a note; nothing at all: all defaults
    proto, info = SW.protocol({"data": {"protocol": {"oracle": "Q − E 的上界低于 +0.02"}}})
    assert proto["oracle_hi90_below"] == 0.02 and any("文字说明" in n for n in info["notes"]) and not info["problems"]
    proto, info = SW.protocol({"data": {"id": "TS"}})
    assert proto == {**SW.DEFAULT_PROTOCOL} and info["source"] == "spec"
    # descriptions in words where a token is expected: the token inside, or the default with a note
    proto, info = SW.protocol({"data": {"protocol": {"set": "clean（干净集 14 个片段）", "assets": "股票类 9 个资产",
                                                      "cma": "historical"}}})
    assert (proto["set"], proto["assets"], proto["cma"]) == ("clean", "equity9", "historical") and not info["problems"]
    # what the reader cannot place is a problem (the evaluation then cannot pass)
    for bad in ({"thresholds": {"oracle_margin": [1]}}, {"thresholds": 3},
                {"conditions": [{"a": "Q:OB", "b": "E:OB", "horizon": "4Q", "hi90_below": 0.02, "assets": "all13"}]},
                {"conditions": [{"a": "Q:OB", "b": "E:OB", "horizon": "4Q", "hi90_below": 0.02},
                                {"a": "Q:OA", "b": "E:OA", "horizon": "4Q", "hi90_below": 0.01}]},
                {"conditions": [{"what": "something else"}]}, {"thresholds": {"assets": "all13"}},
                {"protocol": {"cma": "monthly"}}, {"thresholds": {"narrative_sources": ["guess"]}}):
        proto, info = SW.protocol({"data": bad})
        assert info["problems"], bad
        assert proto["assets"] == "equity9" and proto["cma"] in ("historical", "no_rerate"), bad   # never crashes


def test_an_unreadable_protocol_cannot_pass(env):
    env["record"]({**TS_RECORD, "thresholds": {"oracle_margin": "zero point two"}, "gate": 5})
    env["tf"]()
    res = SW.evaluate(env["run"]())
    assert res["passed"] is False and _by_id(res)["IN:protocol"]["passed"] is False


def test_the_record_s_thresholds_are_applied(env):
    env["record"]({**TS_RECORD, "thresholds": {"oracle_hi90_below": 0.003}})
    env["tf"]()
    res = SW.evaluate(env["run"](rows=_rows(delta={("OA", "4Q"): 0.004})))
    c = _by_id(res)["O:OA:4Q"]
    assert c["bound"] == 0.003 and c["hi90"] > 0.003 and c["passed"] is False and not res["passed"]
    assert _by_id(res)["O:OA:end"]["passed"] is True and _by_id(res)["O:OA:end"]["bound"] == 0.003


# ================================================================ the state every default reads

def test_a_passed_evaluation_switches_every_default(env):
    rid = _passing(env)
    res = SW.evaluate(rid)
    out = SW.write_result(res, results_dir=env["res"])
    assert out["gate_file"].endswith(SW.GATE_FILE)
    st = SW.state(env["res"])
    assert st["status"] == "passed" and st["quarterly"] is True and st["label"] == "默认"
    assert st["defaults"] == {k: "quarterly" for k in SW.CONSUMERS} and st["default_text"] == "默认引擎：季度模型（切换关 T-S 已通过）"
    # the services follow once the quarterly set exists; until then the monthly set, with a note
    sinfo = service.resolve_set(None)
    assert sinfo["id"] == "monthly" and "季度情景集还没有生成" in sinfo["note"]
    (env["res"] / service.QUARTERLY_FILE).write_bytes(b"")
    jsonio.dump({"set": {"engine": "quarterly", "model": "qmodel-r1-f0-a1"}}, env["res"] / service.QUARTERLY_INDEX)
    sinfo = service.resolve_set(None)                  # a set built on another model version than T-S certified
    assert sinfo["id"] == "monthly" and sinfo["note"] == service.STALE_QSET_NOTE
    jsonio.dump({"set": {"engine": "quarterly", "model": SW.current_model_version()}},
                env["res"] / service.QUARTERLY_INDEX)
    assert service.resolve_set(None)["id"] == "quarterly"
    assert SM.switch_gate(env["res"])["passed"] is True
    assert EV.summary()["switch_gate"]["status"] == "passed"


def test_stale_and_inconsistent_results_switch_nothing(env, monkeypatch):
    rid = _passing(env)
    res = SW.evaluate(rid)
    SW.write_result(res, results_dir=env["res"])
    assert SW.state(env["res"])["quarterly"] is True
    # the model's version changed after the evaluation (a code change: a new process, so no cached state)
    monkeypatch.setattr(SW, "current_model_version", lambda: "qmodel-r2-f1-a1")
    SW.clear_cache()
    st = SW.state(env["res"])
    assert st["status"] == "stale" and st["quarterly"] is False and "版本变了" in st["reasons"][0]
    monkeypatch.undo()
    monkeypatch.setattr(EV, "PREREG_DIRS", [env["pre"]])
    monkeypatch.setattr(EV, "RESULTS_DIR", env["res"])
    monkeypatch.setattr(SW, "RESULTS_DIR", env["res"])
    SW.clear_cache()
    assert SW.state(env["res"])["quarterly"] is True
    # results appended inside the record leave its protocol digest unchanged; an edit to the protocol does not
    env["record"]({**TS_RECORD, "results": [{"note": "appended"}]})
    assert SW.state(env["res"])["quarterly"] is True
    env["record"]({**TS_RECORD, "thresholds": {"oracle_hi90_below": 0.05}})
    st = SW.state(env["res"])
    assert st["status"] == "stale" and "预注册记录不见了或改过" in st["reasons"][0]
    env["record"]()
    # a hand-edited result: says passed but one bound is not met
    bad = json.loads(json.dumps(res))
    next(c for c in bad["conditions"] if c["id"] == "O:OF:end")["hi90"] = 0.05
    jsonio.dump(bad, env["res"] / SW.GATE_FILE)
    st = SW.state(env["res"])
    assert st["status"] == "invalid" and st["quarterly"] is False and any("O:OF:end" in r for r in st["reasons"])
    # a hand-written hook, or a result that lost a required condition
    jsonio.dump({"passed": True, "text": "通过了"}, env["res"] / SW.GATE_FILE)
    assert SW.state(env["res"])["status"] == "invalid"
    short = json.loads(json.dumps(res))
    short["conditions"] = [c for c in short["conditions"] if c["id"] != "PIT"]
    jsonio.dump(short, env["res"] / SW.GATE_FILE)
    st = SW.state(env["res"])
    assert st["status"] == "invalid" and "PIT" in st["reasons"][0]


def test_results_are_appended_beside_the_record_never_into_it(env):
    p = env["record"]()
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    env["tf"]()
    res = SW.evaluate(env["run"]())
    out = SW.write_result(res, results_dir=env["res"], append=True)
    side = env["pre"] / "2026-11-05_TS.results.jsonl"
    assert out["appended"] == str(side) and hashlib.sha256(p.read_bytes()).hexdigest() == before
    SW.write_result(SW.evaluate("latest"), results_dir=env["res"], append=True)
    lines = side.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(x)["schema"] == SW.SCHEMA for x in lines)
    # the evaluation tab's record list sees the appended results, and the state reads the newest one
    rec = next(r for r in EV.records() if r["test"] == "TS")
    assert rec["n_results"] == 2 and rec["status"] == "done"
    (env["res"] / SW.GATE_FILE).unlink()
    SW.clear_cache()
    st = SW.state(env["res"])
    assert st["status"] == "passed" and st["source"]["kind"] == "record"
    # a record changed after the evaluation: nothing is appended
    env["record"]({**TS_RECORD, "title": "改过"})
    with pytest.raises(SW.SwitchError, match="改过"):
        SW.write_result(res, results_dir=env["res"], append=True)


def test_nothing_evaluated_lists_what_is_missing(env):
    st = SW.state(env["res"])
    assert st["status"] == "not_evaluated" and st["quarterly"] is False and st["text"] == SW.not_evaluated_text()
    assert any("预注册记录（冻结的协议与门槛）还不在" in r for r in st["reasons"]) and any("T-F" in r for r in st["reasons"])
    assert not any(k in json.dumps(st, ensure_ascii=False) for k in ("SIM_DESIGN", "P5", "preregistered/"))
    assert any("还没有同时用季度模型和月度引擎评分" in r for r in st["reasons"])
    assert st["commands"] == SW.COMMANDS and "--run latest --write --append" in st["commands"][-1]
    env["record"]()
    env["tf"](failed=("F7",))
    SW.clear_cache()
    st = SW.state(env["res"])
    assert any("已预注册（2026-11-05）" in r for r in st["reasons"]) and any("补全技能" in r for r in st["reasons"])


def test_a_records_status_entries_are_not_evaluations(env):
    """P5 appends a 'status' entry (and notes) to the T-S record while T-S is not evaluable: they are snapshots, not
    results, so the gate stays not_evaluated instead of reading them as an invalid result."""
    rec = json.loads(json.dumps(TS_RECORD))
    rec["results"] = [{"n": 1, "kind": "status", "result": {"evaluable": False, "switch": False}},
                      {"n": 2, "kind": "note", "result": {"text": "a disclosure"}}]
    env["record"](rec)
    env["tf"]()
    SW.clear_cache()
    st = SW.state(env["res"])
    assert st["status"] == "not_evaluated" and st["quarterly"] is False


# ================================================================ endpoint, CLI, build hook

def test_the_endpoint_and_the_sets_list_show_the_switch(env):
    body = client.get("/api/evaluation/switch").json()
    assert body["schema"] == SW.STATE_SCHEMA and body["status"] == "not_evaluated" and body["reasons"]
    sets = client.get("/api/portfolio/sets").json()
    assert sets["default"] == "monthly" and sets["switch"]["status"] == "not_evaluated"
    assert [s["id"] for s in sets["sets"]][:2] == ["monthly", "quarterly"]
    q = sets["sets"][1]
    assert q["available"] is False and "python -m scenariolab.portfolio.qsets build" in q["issue"] and q["trial"] is True


def test_the_cli(env, capsys):
    assert SW.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "切换关 T-S 还没有评估" in out and "默认引擎：月度引擎" in out
    assert SW.main(["inputs"]) == 0 and "evaluate --run latest --write --append" in capsys.readouterr().out
    env["tf"]()
    rid = env["run"]()
    assert SW.main(["evaluate", "--run", rid]) == 1                 # no record: not passed
    assert SW.main(["evaluate", "--run", rid, "--write"]) == 2       # and nothing to write against
    assert not (env["res"] / SW.GATE_FILE).exists()
    env["record"]()
    assert SW.main(["evaluate", "--run", rid]) == 0
    assert "只是评估，没有写入" in capsys.readouterr().out and not (env["res"] / SW.GATE_FILE).exists()
    assert SW.main(["evaluate", "--run", rid, "--write"]) == 0 and (env["res"] / SW.GATE_FILE).exists()


def test_the_build_rebuilds_the_quarterly_set_only_after_the_switch(monkeypatch, tmp_path):
    from scenariolab import build
    calls = []
    monkeypatch.setattr(build, "_run", lambda module, ctx, **kw: calls.append((module, kw)) or {})
    monkeypatch.setattr(SW, "state", lambda results_dir=None: {"quarterly": False})
    build.build_portfolio("ctx", tmp_path)
    assert [m for m, _ in calls] == ["portfolio.service"]
    calls.clear()
    monkeypatch.setattr(SW, "state", lambda results_dir=None: {"quarterly": True})
    build.build_portfolio("ctx", tmp_path)
    assert [m for m, _ in calls] == ["portfolio.service", "portfolio.qsets"] and calls[1][1] == {"results": tmp_path}


# ================================================================ the pages render the reasons

def test_the_pages_show_the_default_and_the_reasons():
    """Where a default is shown, the page renders the switch's reasons (_switch.js: from the state, through h(), never
    innerHTML): the evaluation tab, the forecast tab, the scenarios tab and both portfolio pages."""
    sw = src("_switch.js")
    code = "\n".join(line for line in sw.splitlines() if not line.lstrip().startswith("//"))
    assert "export function switchNotice(st, { where = '', compact = false } = {})" in sw
    assert "h('ul', {}, reasons.map(r => h('li', {}, r)))" in sw
    assert "api.get('evaluation/switch')" in sw and "innerHTML" not in code
    assert "switchNotice(sw, { where: '未来一年页' })" in src("forecast.js")
    assert "const defEngine = switched && engines.includes('quarterly') ? 'quarterly'" in src("forecast.js")
    assert "switchNotice(sw, { where: '13 个内置情景' })" in src("scenarios.js")
    assert "sw.replaceChildren(switchNotice(rep.switch, { where: '组合与提取的默认情景集', compact: true }));" in src("_weights.js")
    for name in ("portfolio.js", "withdrawal.js"):
        text = src(name)
        assert "setPicker({ want: q.kv.set || null" in text and "if (shared.set) b.scenario_set = shared.set;" in text
    ev = src("evaluation.js")
    assert "reasons.map(r => h('li', {}, r))" in ev and "cmds.join('\\n')" in ev
    det = src("_scenario_detail.js")
    assert "const qFirst = quarterlyDefault(lastSwitch())" in det


def test_the_quarterly_set_s_rebalancing_and_drawdown_words():
    pf = src("portfolio.js")
    assert "{ value: 'quarterly', label: '每季再平衡'" in pf
    assert "disabled: q3 && r.value === 'monthly' ? true : null, title: q3 && r.value === 'monthly' ? NO_MONTHLY : r.help" in pf
    assert "'季度收益的标准差 × √4，各路径平均'" in pf and "'最大回撤（中位数，季末口径）'" in pf
    # existing deep links keep working: #portfolio?preset=income&rebalance=annual&scenario=ai_bust
    assert "const reb = q.kv.rebalance || take(t => REBALANCE.some(r => r.value === t));" in pf
    for word in ("预测",):
        for name in ("_switch.js",):
            assert word not in re.sub(r"//.*", "", src(name)), name
