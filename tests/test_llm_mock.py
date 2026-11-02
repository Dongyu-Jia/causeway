"""MockClaude (scenariolab/llm/mock.py): deterministic, schema-valid answers derived from the prompt, and the
fixtures for edge cases. No model, no process; the translator end-to-end test needs the panel."""
import json
import threading
import time

import pytest

from agent_llm_cases import china, ctx_deep, ctx_root, fed, hazards, president, synth_request, tariff_pair
from scenariolab import config
from scenariolab.llm import agent_prompts as P
from scenariolab.llm import backend
from scenariolab.llm import claude_code as cc
from scenariolab.llm import jsonfix as jf
from scenariolab.llm import mock
from scenariolab.llm import translator as tr
from scenariolab.scenarios.schema import SPEC_JSON_SCHEMA

MODEL = "claude-opus-5-5"
needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")


def ask(system, user, **kw):
    text, meta = mock.run(system, user, MODEL, **kw)
    return json.loads(text), meta


def actor_call(actor, ctx=None, follower=None):
    schema = P.actor_output_schema(actor, follower)
    return schema, P.actor_system(schema, ctx), P.actor_user(ctx or ctx_root(), actor, follower=follower)


def test_the_conftest_makes_the_mock_the_default_backend():
    assert backend.backend_name() == "mock" and tr.backend_name() == "mock"
    st = tr.status()
    assert st["backend"] == "mock" and st["available"] and "不消耗" in st["label"]
    assert backend.status()["backend"] == "mock"
    with pytest.raises(FileNotFoundError, match="tests never start Claude Code"):
        cc._popen(["claude", "-p"])


def test_actor_answers_choose_the_prior_mode_and_validate():
    schema, system, user = actor_call(fed())
    ans, meta = ask(system, user)
    assert jf.validate(ans, schema) == []
    probs = {o["id"]: o["prob"] for o in ans["options"]}
    assert probs == {"hold": 0.94, "hike_25": 0.03, "cut_50": 0.03}         # modal option + status quo + runner-up
    assert meta["backend"] == "mock" and meta["mock"]["kind"] == "actor" and meta["usage"]["input_tokens"] > 0
    assert mock.run(system, user, MODEL)[0] == mock.run(system, user, MODEL)[0]          # deterministic


def test_prior_and_draw_modes(monkeypatch):
    schema, system, user = actor_call(fed())
    monkeypatch.setenv("MSL_LLM_MOCK_MODE", "prior")
    ans, _ = ask(system, user)
    assert [o["id"] for o in ans["options"]] == ["hold", "hike_25", "cut_50", "hike_50"]
    assert sum(o["prob"] for o in ans["options"]) == pytest.approx(1.0) and ans["vs_base_rate"] is None
    assert ans["options"][0]["prob"] == pytest.approx(0.48 / 0.88, abs=0.02)      # two decimals, summing to 1
    monkeypatch.setenv("MSL_LLM_MOCK_MODE", "draw")
    chosen = set()
    for seed in range(40):
        monkeypatch.setenv("MSL_LLM_MOCK_SEED", str(seed))
        a, _ = ask(system, user)
        assert jf.validate(a, schema) == [] and a["options"][0]["prob"] == 0.94
        chosen.add(a["options"][0]["id"])
        monkeypatch.setenv("MSL_LLM_MOCK_SEED", str(seed))
        assert ask(system, user)[0] == a                                    # a seeded draw repeats
    assert "hold" in chosen and len(chosen) >= 3                            # draws follow the prior, not the mode


def test_actor_without_base_rates_chooses_its_status_quo_and_magnitudes_take_the_default():
    schema, system, user = actor_call(china())
    ans, _ = ask(system, user)
    assert ans["options"][0]["id"] == "no_response" and jf.validate(ans, schema) == []
    a = fed(options=[{"id": "cut_75plus", "label": "紧急降息", "base_rate": 0.9,
                      "magnitude": {"default": -1.0, "lo": -1.75, "hi": -0.75}},
                     {"id": "hold", "label": "不变", "base_rate": 0.1, "status_quo": True}])
    schema, system, user = actor_call(a)
    ans, _ = ask(system, user)
    assert ans["options"][0] == {**ans["options"][0], "id": "cut_75plus", "magnitude": -1.0}


def test_a_follower_answers_with_the_prior_rows():
    pair = tariff_pair()
    schema, system, user = actor_call(china(), follower=pair)
    ans, _ = ask(system, user)
    assert jf.validate(ans, schema) == []
    assert all(o["prob"] is None for o in ans["options"])
    rows = {r["leader_option"]: {p["option"]: p["prob"] for p in r["follower_probs"]} for r in ans["reaction"]["rows"]}
    assert rows["tariff_up_small"] == {"no_response": 0.4, "retaliate_tariffs": 0.5, "negotiate_truce": 0.1}
    assert rows["tariff_down"] == {"no_response": 0.2, "negotiate_truce": 0.8} and ans["reaction"]["leader"] == "president"


def test_merged_node_answer_covers_every_actor_and_the_reaction_pairs():
    actors, pair = [fed(), president(), china()], tariff_pair()
    schema = P.node_output_schema(actors, [pair], hazards())
    ans, _ = ask(P.actor_system(schema), P.node_user(ctx_deep(), actors, [pair], hazards()))
    assert jf.validate(ans, schema) == []
    assert set(ans["actors"]) == {"fed", "president", "china"} and ans["hazard_tilts"] == []
    assert ans["reactions"][0]["leader"] == "president" and len(ans["reactions"][0]["rows"]) == 3
    assert all("private_note" not in b for b in ans["actors"].values())


def test_synthesizer_answers_with_partitions_at_the_models_own_masses():
    req = synth_request()
    schema = P.synth_output_schema(req)
    ans, _ = ask(P.synth_system(schema), P.synth_user(req))
    assert jf.validate(ans, schema) == []
    k1, k2 = ans["clusters"]
    assert (k1["cluster_id"], k1["axis"], k2["axis"]) == ("k1", "d_unrate", "d_gz")
    assert [b["prob"] for b in k1["branches"]] == [0.16, 0.68, 0.16]
    assert [b["kind"] for b in k1["branches"]] == ["partition", "central", "partition"]
    assert all(b["shifts"] == {} and b["kind"] != "mechanism" for c in ans["clusters"] for b in c["branches"])
    assert [(b["region"]["lo_z"], b["region"]["hi_z"]) for b in k2["branches"]] == [(None, -1), (-1, 1), (1, None)]
    assert [b["prob"] for b in k2["branches"]] == [0.12, 0.70, 0.18]       # the 1% panic region joined 收紧
    req = synth_request(clusters=[{"id": "k1", "actions": [], "written_fields": ["d_unrate", "d_core", "x_supply_chain",
                                                                                 "d_gz", "d_cpbill"]}])
    schema = P.synth_output_schema(req)
    ans, _ = ask(P.synth_system(schema), P.synth_user(req))
    assert ans["clusters"][0]["status"] == "decline" and jf.validate(ans, schema) == []   # no axis left


def test_root_story_and_probe_answers():
    levers = [{"id": "d_tariff", "zh": "有效关税率变动"}]
    schema = P.root_output_schema(levers)
    narrative = "我认为美联储一定会降息。总统宣布对全部进口商品加征关税。"
    ans, _ = ask(P.root_system(schema), P.root_user(ctx_root(), narrative, levers))
    assert jf.validate(ans, schema) == [] and ans["user_views"] == ["我认为美联储一定会降息。"]
    assert ans["forced"] == [] and ans["h0"]["summary"] == narrative
    for schema, system, user in ((P.story_output_schema(), P.story_system(P.story_output_schema()),
                                  P.story_user(["2026Q4：美联储按兵不动"])),
                                 (P.leak_probe_schema(), P.probe_system(P.leak_probe_schema()), P.probe_user("T0"))):
        assert jf.validate(ask(system, user)[0], schema) == []


def test_translator_answers_with_a_spec_in_the_schemas_shape():
    system = "情景翻译。\n\n" + tr.schema_section()
    user = tr._user_message("中东战争升级，油价急涨，美联储被迫加息，经济滑向衰退。", 180)
    spec, meta = ask(system, user)
    assert jf.validate(spec, SPEC_JSON_SCHEMA) == [] and meta["mock"]["kind"] == "translator"
    views = {v["var"] for v in spec["phases"][0]["views"]}
    assert {"f_oil", "s_cpi_yoy", "f_unrate", "f_ip", "f_ff"} <= views
    assert sum(p["months"] for p in spec["phases"]) == 180 and spec["id"].startswith("mock_")
    plain, _ = ask(system, tr._user_message("一个平静的世界。", 24))
    assert plain["phases"][0]["views"] == [] and [p["months"] for p in plain["phases"]] == [24]


@needs_panel
def test_translate_runs_end_to_end_on_the_mock_and_is_not_cached():
    from scenariolab.context import EngineContext
    ctx = EngineContext.load(selection=False)
    out = tr.translate("中东战争升级，霍尔木兹海峡航运受阻半年，油价急涨，随后经济滑向滞胀。", ctx)
    assert out["backend"] == "mock" and out["validation"]["ok"] and out["cached"] is False
    assert out["spec"]["id"].startswith("llm_mock_") and any(p["views"] for p in out["spec"]["phases"])
    assert not list(tr.CACHE_DIR.glob("*.json"))
    assert tr.request_info("mock", MODEL)["model_call"] is False


# ---------------------------------------------------------------- fixtures

@pytest.mark.parametrize("name, check", [
    ("malformed_json", lambda t: jf.extract_json(t)[0] is None),
    ("fenced", lambda t: t.startswith("好的") and jf.extract_json(t)[0] is not None),
    ("trailing_comma", lambda t: ", }" in t and jf.extract_json(t)[0] is not None),
    ("fullwidth", lambda t: "“status”：" in t and jf.extract_json(t)[0]["status"] == "ok"),
    ("refusal_text", lambda t: t.startswith("I can't help")),
    ("percent_probs", lambda t: json.loads(t)["options"][0]["prob"] == 94),
    ("bad_sum", lambda t: json.loads(t)["options"][0]["prob"] == 0.10),
    ("price_sentence", lambda t: json.loads(t)["options"][0]["rationale"] == mock.PRICE_SENTENCE),
    ("out_of_power", lambda t: json.loads(t)["options"][0]["writes"] == {"d_ff": -0.5}),
    ("decline", lambda t: json.loads(t)["status"] == "decline"),
    ("decline_actor", lambda t: json.loads(t)["status"] == "decline" and json.loads(t)["options"] == []),
])
def test_text_and_answer_fixtures(mock_llm, name, check):
    _, system, user = actor_call(fed())
    mock_llm.push(name)
    text, meta = mock.run(system, user, MODEL)
    assert check(text) and meta["mock"]["fixture"] == name
    assert mock.run(system, user, MODEL)[1]["mock"]["fixture"] is None          # consumed (times=1)


@pytest.mark.parametrize("name, kind", [
    ("refusal", "refusal"), ("timeout", "timeout"), ("usage_limit", "limit"), ("usage_limit_text", "limit"),
    ("weekly_limit", "limit"), ("login", "login"), ("rate_limit", "rate"), ("overloaded", "overloaded"),
    ("server_error", "server"), ("model_missing", "model"), ("max_tokens", "max_tokens"), ("context", "context"),
    ("empty", "empty"), ("cli_missing", "cli_missing"), ("crash", "unparseable"),
])
def test_error_fixtures_raise_the_backends_own_errors(mock_llm, name, kind):
    _, system, user = actor_call(fed())
    mock_llm.push(name)
    with pytest.raises(cc.ClaudeCodeError) as err:
        mock.run(system, user, MODEL, 42)
    assert err.value.kind == kind
    if name == "usage_limit":
        assert err.value.reset_at and "预计" in str(err.value) and "用量上限" in str(err.value)
    if name == "usage_limit_text":
        assert "3pm (America/Los_Angeles) 重置" in str(err.value)
        import datetime as dt
        from zoneinfo import ZoneInfo
        at = dt.datetime.fromtimestamp(err.value.reset_at, ZoneInfo("America/Los_Angeles"))
        assert (at.hour, at.minute) == (15, 0) and 0 < err.value.reset_at - dt.datetime.now().timestamp() <= 86400
    if name == "timeout":
        assert "42 秒内没有完成" in str(err.value)


def test_fixture_matching_and_persistence(mock_llm, monkeypatch):
    _, system, user = actor_call(fed())
    req = synth_request()
    s_schema = P.synth_output_schema(req)
    s_system, s_user = P.synth_system(s_schema), P.synth_user(req)
    mock_llm.push("timeout", match="synth")
    assert mock.run(system, user, MODEL)[1]["mock"]["fixture"] is None         # an actor call: not matched
    with pytest.raises(cc.ClaudeCodeError):
        mock.run(s_system, s_user, MODEL)
    mock_llm.push("empty", match="actor:美联储", times=2)
    for _ in range(2):
        with pytest.raises(cc.ClaudeCodeError, match="结果为空"):
            mock.run(system, user, MODEL)
    mock.run(system, user, MODEL)
    mock_llm.push(lambda kind, s, u, answer: json.dumps({**answer, "vs_base_rate": f"custom {kind}"}))
    assert json.loads(mock.run(system, user, MODEL)[0])["vs_base_rate"] == "custom actor"
    monkeypatch.setenv("MSL_LLM_MOCK_FIXTURE", "overloaded@synth,refusal@nothing-matches")
    for _ in range(2):
        with pytest.raises(cc.ClaudeCodeError, match="繁忙"):
            mock.run(s_system, s_user, MODEL)
    assert mock.run(system, user, MODEL)[1]["mock"]["fixture"] is None
    with pytest.raises(ValueError, match="unknown mock fixture"):
        mock_llm.push("no-such-fixture")
    assert [c["kind"] for c in mock.calls()][:2] == ["actor", "synth"]


def test_a_repair_call_renormalises_or_rebuilds_from_the_schema():
    schema, _, user = actor_call(fed())
    bad = json.dumps({"schema_version": "agents-actor-2", "actor": "fed", "status": "ok",
                      "options": [{"id": "hold", "prob": 60, "rationale": "x"}, {"id": "cut_25", "prob": 20,
                                                                                   "rationale": "y"}]})
    ans, meta = ask(P.repair_system(schema), P.repair_user(["概率之和偏离 1"], bad))
    assert meta["mock"]["kind"] == "repair" and [o["prob"] for o in ans["options"]] == [0.75, 0.25]
    ans, _ = ask(P.repair_system(schema), P.repair_user(["不是合法的 JSON"], bad[:30]))
    assert jf.validate(ans, schema) == [] and ans["options"][0]["id"] == "hold"   # the schema lists the status quo first


def test_cassettes_are_read_first(tmp_path, monkeypatch):
    _, system, user = actor_call(fed())
    key = cc.request_key(MODEL, system, user, None, False)
    (tmp_path / f"{key}.json").write_text(json.dumps({"text": '{"recorded": true}'}), encoding="utf-8")
    monkeypatch.setenv("MSL_LLM_MOCK_CASSETTES", str(tmp_path))
    assert mock.run(system, user, MODEL)[0] == '{"recorded": true}'
    (tmp_path / f"{key}.json").write_text(json.dumps({"error": {"kind": "limit", "message": "录制的用量上限"}}),
                                          encoding="utf-8")
    with pytest.raises(cc.ClaudeCodeError, match="录制的用量上限") as err:
        mock.run(system, user, MODEL)
    assert err.value.kind == "limit"
    assert json.loads(mock.run(system, user + " ", MODEL)[0])["actor"] == "fed"    # no cassette: derived


def test_delay_hang_and_cancel(mock_llm, monkeypatch):
    _, system, user = actor_call(fed())
    monkeypatch.setenv("MSL_LLM_MOCK_DELAY_S", "0.05")
    t0 = time.perf_counter()
    mock.run(system, user, MODEL)
    assert time.perf_counter() - t0 >= 0.04
    with pytest.raises(cc.ClaudeCodeError) as err:
        mock.run(system, user, MODEL, 0.01)                         # the delay is longer than the timeout
    assert err.value.kind == "timeout"
    monkeypatch.delenv("MSL_LLM_MOCK_DELAY_S")
    cancel = threading.Event()
    mock_llm.push("hang")
    threading.Timer(0.1, cancel.set).start()
    t0 = time.perf_counter()
    with pytest.raises(cc.ClaudeCodeError) as err:
        mock.run(system, user, MODEL, 20, cancel=cancel)
    assert err.value.kind == "cancelled" and time.perf_counter() - t0 < 5
    with pytest.raises(cc.ClaudeCodeError) as err:                      # already cancelled: nothing runs
        mock.run(system, user, MODEL, cancel=cancel)
    assert err.value.kind == "cancelled"
    mock_llm.push("hang")
    with pytest.raises(cc.ClaudeCodeError) as err:
        mock.run(system, user, MODEL, 0.05)
    assert err.value.kind == "timeout"


def test_backend_run_dispatches_and_rejects_unknown_backends():
    _, system, user = actor_call(fed())
    text, meta = backend.run(system, user, MODEL, 30, effort="medium", safe_mode=True)
    assert meta["backend"] == "mock" and meta["effort"] == "medium" and meta["safe_mode"] is True
    with pytest.raises(cc.ClaudeCodeError, match="不认识") as err:
        backend.run(system, user, MODEL, backend="openai")
    assert err.value.kind == "cli_missing"
    assert backend.capabilities()["cli_version"] == mock.MOCK_VERSION
    assert backend.normalize(" Claude-Code ") == "claude_code" and backend.normalize("") == "claude_code"
