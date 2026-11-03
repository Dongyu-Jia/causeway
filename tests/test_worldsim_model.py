"""WorldSim's data model: schema and world validation, value coercion, periods, the synthesizer's next state checked
against the schema, and the committed default world (which must load and validate without data/)."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from worldsim import config
from worldsim import model as M

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = {"fields": [
    {"id": "rate", "label": "Rate", "group": "Policy", "type": "percent", "unit": "%", "min": 0, "max": 20},
    {"id": "jobs", "label": "Jobs", "group": "Labour", "type": "integer", "unit": "k"},
    {"id": "mood", "label": "Mood", "group": "Politics", "type": "enum", "options": ["calm", "tense", "crisis"]},
    {"id": "open", "label": "Open", "group": "Politics", "type": "boolean"},
    {"id": "note", "label": "Note", "group": "Politics", "type": "text"},
]}
CURRENT = {"step": 0, "period": "2026Q4", "values": {"rate": 3.88, "jobs": 50, "mood": "tense", "open": True, "note": "x"},
           "text": {"narrative": "Start.", "key_questions": [{"question": "Will rates rise?", "status": "open", "answer": ""}],
                    "attributes": [{"name": "Stance", "value": "hawkish"}]}}


def world(**kw):
    w = {"id": "w1", "name": "Test", "language": "English", "schema": copy.deepcopy(SCHEMA),
         "actors": [{"id": "a", "name": "A", "action_space": [{"id": "hold", "label": "Hold"}]}],
         "initial_state": copy.deepcopy(CURRENT)}
    w.update(kw)
    return w


# ------------------------------------------------------------------------------------------------ schema and world

def test_a_valid_world_normalizes_without_problems():
    w, problems = M.normalize_world(world())
    assert problems == []
    assert [f["id"] for f in w["schema"]["fields"]] == ["rate", "jobs", "mood", "open", "note"]
    assert w["actors"][0]["allow_free_action"] is True and w["actors"][0]["visible_fields"] is None
    assert w["period_step"] == {"unit": "quarter", "count": 1} and w["initial_state"]["values"]["rate"] == 3.88


@pytest.mark.parametrize("bad, needle", [
    ({"id": "Bad Id", "label": "x", "type": "number"}, "id must be"),
    ({"id": "x", "type": "color"}, "type must be one of"),
    ({"id": "x", "type": "enum", "options": []}, "at least one option"),
    ({"id": "x", "type": "number", "min": 5, "max": 1}, "min is greater than max"),
    ({"id": "x", "type": "number", "min": "abc"}, "min must be a number"),
])
def test_bad_fields_are_reported(bad, needle):
    _, problems = M.normalize_schema({"fields": [bad]})
    assert any(needle in p for p in problems), problems


def test_duplicate_ids_and_bad_actors_are_reported():
    _, problems = M.normalize_schema({"fields": [{"id": "a", "type": "number"}, {"id": "a", "type": "text"}]})
    assert any("used twice" in p for p in problems)
    actors = [{"id": "x", "name": "X", "action_space": [{"id": "free", "label": "F"}]},
              {"id": "x", "name": "Y", "action_space": [{"id": "go", "label": "Go"}, {"id": "go", "label": "Go"}]},
              {"id": "z", "name": "Z", "action_space": [], "allow_free_action": False}]
    _, problems = M.normalize_actors(actors)
    text = " ".join(problems)
    assert "reserved for free actions" in text and "actor id x is used twice" in text
    assert "action id go is used twice" in text and "needs at least one action" in text


def test_initial_values_are_typed_and_unreadable_ones_reported():
    state, problems = M.normalize_state(SCHEMA, {"values": {"rate": "3.5%", "jobs": 12.0, "mood": "Tense",
                                                            "open": "yes", "note": 7, "ghost": 1}})
    assert problems == []
    assert state["values"] == {"rate": 3.5, "jobs": 12, "mood": "tense", "open": True, "note": "7"}
    _, problems = M.normalize_state(SCHEMA, {"values": {"rate": "high", "mood": "chaos"}})
    assert len(problems) == 2


def test_visible_fields_drop_unknown_ids():
    w, problems = M.normalize_world(world(actors=[{"id": "a", "name": "A", "visible_fields": ["rate", "nope"],
                                                   "action_space": [{"id": "hold", "label": "Hold"}]}]))
    assert problems == [] and w["actors"][0]["visible_fields"] == ["rate"]


@pytest.mark.parametrize("period, unit, count, nxt", [
    ("2026Q4", "quarter", 1, "2027Q1"), ("2026Q2", "quarter", 3, "2027Q1"), ("2026Q4", "year", 1, "2027Q4"),
    ("2026-11", "month", 1, "2026-12"), ("2026-12", "month", 2, "2027-02"), ("2026", "year", 1, "2027"),
    ("Turn 3", "quarter", 1, "Turn 4"), ("start", "quarter", 1, "start +1")])
def test_periods_advance(period, unit, count, nxt):
    assert M.advance_period(period, unit, count) == nxt


def test_blank_world_is_valid():
    w = M.blank_world("abc", "Blank")
    assert M.normalize_world(w)[1] == [] and w["name"] == "Blank"


# ------------------------------------------------------------------------------------------------ the next state

def answer(**values):
    vals = {"rate": {"value": 4.13, "reason": "the Fed hiked 25bp"}, "jobs": {"value": 50, "reason": ""},
            "mood": {"value": "tense", "reason": ""}, "open": {"value": True, "reason": ""},
            "note": {"value": "x", "reason": ""}}
    vals.update(values)
    return {"summary": "A hike.", "values": vals, "narrative": "The Fed hiked.",
            "key_questions": [{"question": "Will rates rise?", "status": "answered", "answer": "Yes, by 25bp"}],
            "attributes": [{"name": "Stance", "value": "more hawkish"}]}


def test_a_clean_answer_becomes_the_next_state_with_changes_and_reasons():
    state, flags, changes = M.validate_next_state(SCHEMA, CURRENT, answer(), period="2027Q1", step=1)
    assert flags == [] and state["step"] == 1 and state["period"] == "2027Q1"
    assert state["values"]["rate"] == 4.13 and state["reasons"]["rate"] == "the Fed hiked 25bp"
    assert changes == [{"field": "rate", "label": "Rate", "from": 3.88, "to": 4.13, "reason": "the Fed hiked 25bp"}]
    assert state["text"]["key_questions"][0]["status"] == "answered"
    assert M.text_changes(CURRENT["text"], state["text"])["answered_questions"] == ["Will rates rise?"]


def test_missing_null_and_mistyped_values_are_carried_forward_and_flagged():
    out = answer(jobs={"value": None, "reason": ""}, mood={"value": "panic", "reason": "?"},
                 open={"value": "maybe", "reason": ""})
    del out["values"]["note"]
    out["values"]["invented"] = {"value": 1, "reason": "made up"}
    state, flags, _ = M.validate_next_state(SCHEMA, CURRENT, out, period="2027Q1", step=1)
    by = {(f["field"], f["code"]) for f in flags}
    assert {("jobs", "null"), ("mood", "type"), ("open", "type"), ("note", "missing"), ("invented", "unknown_field")} <= by
    assert state["values"]["jobs"] == 50 and state["values"]["mood"] == "tense" and state["values"]["note"] == "x"
    assert "invented" not in state["values"]


def test_out_of_range_values_are_kept_but_flagged_and_a_change_needs_a_reason():
    out = answer(rate={"value": 25.0, "reason": ""}, jobs={"value": "61", "reason": ""})
    state, flags, changes = M.validate_next_state(SCHEMA, CURRENT, out, period="2027Q1", step=1)
    codes = {(f["field"], f["code"]) for f in flags}
    assert ("rate", "range") in codes and ("rate", "no_reason") in codes and ("jobs", "coerced") in codes
    assert state["values"]["rate"] == 25.0 and state["values"]["jobs"] == 61
    assert {c["field"] for c in changes} == {"rate", "jobs"}


def test_an_answer_without_values_or_text_keeps_the_current_state():
    state, flags, changes = M.validate_next_state(SCHEMA, CURRENT, {"summary": ""}, period="2027Q1", step=1)
    assert state["values"] == CURRENT["values"] and changes == []
    assert state["text"]["narrative"] == "Start." and state["text"]["attributes"] == CURRENT["text"]["attributes"]
    codes = {f["code"] for f in flags}
    assert {"values", "narrative", "key_questions", "attributes"} <= codes


def test_plain_values_without_reason_objects_are_accepted():
    out = answer()
    out["values"] = {k: v["value"] for k, v in out["values"].items()}
    state, flags, _ = M.validate_next_state(SCHEMA, CURRENT, out, period="2027Q1", step=1)
    assert state["values"]["rate"] == 4.13 and [f["code"] for f in flags] == ["no_reason"]


# ------------------------------------------------------------------------------------------------ the default world

def default_world():
    return json.loads(config.DEFAULT_WORLD_FILE.read_text(encoding="utf-8"))


def test_the_default_world_loads_and_validates_without_data():
    w = default_world()
    norm, problems = M.normalize_world(w)
    assert problems == []
    assert norm["name"] == "美国与世界 · 2026Q4" and norm["language"] == "中文"
    assert norm["initial_state"]["period"] == "2026Q4"
    fields = norm["schema"]["fields"]
    assert len(fields) >= 45
    vals = norm["initial_state"]["values"]
    assert all(vals[f["id"]] is not None for f in fields), "every field has a value today"
    quoted = {"tips10": 2.93, "debt_gdp": 97.37, "consumer_sentiment": 48.1, "initial_claims": 200.5,
              "labor_share": 93.4, "profit_share": 13.14, "dividend_yield": 1.03, "it_invest_gdp": 5.05}
    for k, v in quoted.items():
        assert vals[k] == pytest.approx(v, abs=0.06), k
    for f in fields:
        if f["type"] in M.NUMERIC_TYPES:
            assert f["min"] is not None and f["max"] is not None and f["min"] <= vals[f["id"]] <= f["max"], f["id"]
            assert f["hist"], f["id"]
    qualitative = {f["id"]: f for f in fields if f["type"] in ("enum", "text")}
    for fid in ("fed_stance", "ieepa_status", "oil_disruption", "house_control", "china_truce", "opec_policy"):
        assert fid in qualitative
    assert qualitative["ieepa_status"]["tag"] == "未确认"
    text = norm["initial_state"]["text"]
    assert 6 <= len(text["key_questions"]) <= 10 and len(text["attributes"]) >= 4 and len(text["narrative"]) > 400


def test_the_default_world_has_all_fourteen_actors_with_sound_action_spaces():
    w = default_world()
    names = [a["name"] for a in w["actors"]]
    assert names == ["美联储", "总统与白宫", "国会", "财政部", "最高法院与法院", "选民", "中国", "欧佩克+", "冲突方",
                     "超大型科技公司 CEO", "消费者/居民", "华尔街投资者", "银行/放贷机构", "非科技企业"]
    for a in w["actors"]:
        assert a["enabled"] and a["allow_free_action"] and a["role"], a["id"]
        assert 3 <= len(a["cares_about"]) <= 6, a["id"]
        assert 3 <= len(a["action_space"]) <= 7, a["id"]
        assert "现状" in a["action_space"][0]["description"], f"{a['id']}: the status quo comes first"
        assert a["private_info"], a["id"]
    synth = w["synthesizer"]["instructions"]
    assert "联合" in synth and "资产价格" in synth


def test_worldsim_does_not_import_causeway():
    """The package is self-contained: importing the server must not pull in scenariolab or pandas."""
    code = ("import sys, worldsim.server, worldsim.engine; "
            "bad = [m for m in ('scenariolab', 'pandas', 'numpy') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "panel_monthly.csv").exists(), reason="needs Causeway's local data/")
def test_the_committed_default_world_matches_a_fresh_build():
    r = subprocess.run([sys.executable, "tools/build_worldsim_default.py", "--check"], cwd=ROOT, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
