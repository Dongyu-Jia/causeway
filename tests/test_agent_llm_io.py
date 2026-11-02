"""The agent tree's LLM call layer (scenariolab/agents/llm_io.py) on the mock backend: every call type end to end,
the cache, the budget, retries, the format repair, the rephrase, pauses, the circuit breaker, cancellation and the
concurrency cap. No model, no process, no data."""
import json
import threading
import time

import pytest

from agent_llm_cases import china, ctx_deep, ctx_root, fed, hazards, president, synth_request, tariff_pair
from scenariolab.agents import llm_io as io
from scenariolab.llm import backend
from scenariolab.llm import claude_code as cc
from scenariolab.llm import jsonfix as jf
from scenariolab.llm import mock
from scenariolab.llm import agent_prompts as P

SLEEPS: list = []


def llm(**kw):
    cache = kw.pop("cache", None)
    on_event = kw.pop("on_event", None)
    cancel = kw.pop("cancel", None)
    kill = kw.pop("kill", None)
    ledger = kw.pop("ledger", None)
    failure_log = kw.pop("failure_log", None)
    return io.AgentLLM(io.Settings(**kw), cache=cache, on_event=on_event, cancel=cancel, failure_log=failure_log,
                       sleep=SLEEPS.append, kill=kill, ledger=ledger)


@pytest.fixture(autouse=True)
def no_sleeps():
    SLEEPS.clear()
    yield


def ids(answer):
    return {o["id"]: o["p_llm"] for o in answer.options}


# ---------------------------------------------------------------- every call type

def test_actor_node_synth_root_story_and_probe_on_the_mock(tmp_path):
    a = llm()
    r = a.actor(ctx_root(), fed())
    assert ids(r.answer) == {"hold": 0.94, "hike_25": 0.03, "cut_50": 0.03} and r.flags == []
    assert r.calls[0].kind == "actor" and r.calls[0].effort == "medium" and r.calls[0].backend == "mock"
    f = a.actor(ctx_root(), china(), follower=tariff_pair())
    assert f.answer.reaction["tariff_up_small"]["retaliate_tariffs"] == 0.5
    n = a.node(ctx_deep(), [fed(), president(), china()], reactions=[tariff_pair()], hazards=hazards())
    assert set(n.actors) == {"fed", "president", "china"} and not any(x.fallback for x in n.actors.values())
    assert n.reactions["president"]["china"]["tariff_down"] == {"no_response": 0.2, "negotiate_truce": 0.8}
    assert n.calls[0].effort == "medium" and n.hazards == {}
    s = a.synth(synth_request())
    assert {k: c.status for k, c in s.clusters.items()} == {"k1": "ok", "k2": "ok"}
    assert s.clusters["k1"].axis == "d_unrate" and [b["p_llm"] for b in s.clusters["k1"].branches] == [0.16, 0.68,
                                                                                                         0.16]
    root = a.root(ctx_root(), "我认为美联储一定会降息。总统宣布加征关税。", settable=[{"id": "d_tariff"}])
    assert root.ok and root.user_views == ["我认为美联储一定会降息。"] and root.forced == []
    story = a.story(["2026Q4：美联储按兵不动", "2027Q1：模型中枢"])
    assert story.ok and story.payload["headline"] == "模拟叙事"
    probe = a.leak_probe("T0：核心通胀 2.6%")
    assert probe.ok and probe.payload["quarter_guess"] == 1
    assert a.stats["calls_new"] == 7 and a.stats["failed"] == 0 and a.stats["usage"]["input_tokens"] > 0
    json.dumps(n.to_dict(), ensure_ascii=False)                      # results serialise for node detail files
    json.dumps(r.to_dict(), ensure_ascii=False)
    json.dumps(s.to_dict(), ensure_ascii=False)


def test_private_items_reach_only_the_owners_separate_call(monkeypatch):
    seen = []
    real = backend.run
    monkeypatch.setattr(backend, "run", lambda system, user, *a, **k: seen.append(user) or real(system, user, *a, **k))
    a = llm()
    a.actor(ctx_root(), fed())
    a.node(ctx_deep(), [fed(), president(), china()])
    assert fed().private in seen[0] and fed().private not in seen[1]


# ---------------------------------------------------------------- cache and determinism

def test_a_rerun_on_the_cache_makes_no_calls_and_gives_the_same_answers(tmp_path):
    cache = io.FileCache(tmp_path / "cache")
    first = llm(cache=cache)
    r1 = first.actor(ctx_root(), fed())
    n1 = first.node(ctx_deep(), [fed(), china()])
    again = llm(cache=cache)
    r2 = again.actor(ctx_root(), fed())
    n2 = again.node(ctx_deep(), [fed(), china()])
    assert again.stats["calls_new"] == 0 and again.stats["cached"] == 2
    assert r2.calls[0].cached and r1.answer.to_dict() == r2.answer.to_dict()
    assert {k: v.to_dict() for k, v in n1.actors.items()} == {k: v.to_dict() for k, v in n2.actors.items()}
    key = r1.calls[0].key
    path = cache.path(key)
    assert json.loads(path.read_text(encoding="utf-8"))["kind"] == "actor"
    path.write_text("{torn", encoding="utf-8")                        # a torn entry is a miss, never an error
    third = llm(cache=cache)
    third.actor(ctx_root(), fed())
    assert third.stats["calls_new"] == 1


def test_the_cache_key_names_everything_that_shapes_an_answer():
    a = llm()
    base = dict(effort="high", sample=0, repair=0)
    k = a.key("actor", "S", "U", **base)
    assert k == a.key("actor", "S", "U", **base)
    assert len({k, a.key("actor", "S", "U2", **base), a.key("actor", "S2", "U", **base),
                a.key("actor", "S", "U", effort="medium", sample=0, repair=0),
                a.key("actor", "S", "U", effort="high", sample=1, repair=0),
                a.key("actor", "S", "U", effort="high", sample=0, repair=1),
                llm(versions={"registry": "x"}).key("actor", "S", "U", **base),
                llm(model="claude-sonnet-5-5").key("actor", "S", "U", **base),
                llm(backend="claude_code").key("actor", "S", "U", **base)}) == 9   # mock answers never pose as real


# ---------------------------------------------------------------- budget, retries, failures

def test_the_budget_turns_actors_prior_only_and_the_run_goes_on():
    a = llm(max_calls=1)
    assert not a.actor(ctx_root(), fed()).answer.fallback
    r = a.actor(ctx_root(), president())
    assert r.answer.fallback and r.answer.prior_only and "budget" in r.flags
    assert r.calls[0].error_kind == "budget" and a.stats["budget_hits"] == 1 and a.stats["calls_new"] == 1
    s = a.synth(synth_request())
    assert "budget" in s.flags and all(c.status == "default" for c in s.clusters.values())


def test_transient_failures_retry_twice_with_backoff(mock_llm):
    mock_llm.push("timeout", match="actor", times=2)
    r = llm().actor(ctx_root(), fed())
    assert not r.answer.fallback and r.calls[0].attempts == 3 and r.calls[0].retried == ["timeout", "timeout"]
    assert SLEEPS == [30.0, 120.0] and r.calls[0].error_kind is None
    SLEEPS.clear()
    mock_llm.push("overloaded", match="actor", times=3)
    r = llm().actor(ctx_root(), fed())
    assert r.answer.fallback and r.calls[0].error_kind == "overloaded" and r.calls[0].attempts == 3
    assert "服务繁忙" in r.answer.reason


def test_empty_or_unparseable_cli_output_retries_once(mock_llm):
    mock_llm.push("crash", match="actor")
    r = llm().actor(ctx_root(), fed())
    assert not r.answer.fallback and r.calls[0].attempts == 2 and SLEEPS == [30.0]
    mock_llm.push("empty", match="actor", times=2)
    r = llm().actor(ctx_root(), fed())
    assert r.answer.fallback and r.calls[0].attempts == 2


@pytest.mark.parametrize("fixture", ["max_tokens", "context"])
def test_max_tokens_and_context_go_prior_only_without_retry(mock_llm, fixture):
    mock_llm.push(fixture, match="actor")
    r = llm().actor(ctx_root(), fed())
    assert r.answer.prior_only and r.calls[0].attempts == 1 and r.calls[0].error_kind == fixture


def test_an_unusable_answer_gets_one_format_repair(mock_llm):
    mock_llm.push("malformed_json", match="actor")
    r = llm().actor(ctx_root(), fed())
    assert "repaired" in r.flags and not r.answer.fallback
    assert [c.purpose for c in r.calls] == ["call", "repair"] and r.calls[1].effort == "medium"
    assert r.calls[0].key != r.calls[1].key and "经过一次格式修复" in r.report
    mock_llm.push("bad_sum", match="actor")
    r = llm().actor(ctx_root(), fed())
    assert "repaired" in r.flags and sum(ids(r.answer).values()) == pytest.approx(1.0)
    mock_llm.push("malformed_json", match="actor")
    mock_llm.push("malformed_json", match="repair")
    r = llm().actor(ctx_root(), fed())
    assert r.answer.fallback and "fallback" in r.flags and any("修复后仍不合格" in x for x in r.report)


def test_deterministic_fixes_need_no_extra_call(mock_llm):
    for fixture in ("fenced", "trailing_comma", "fullwidth", "percent_probs", "price_sentence", "out_of_power"):
        mock_llm.push(fixture, match="actor")
        a = llm()
        r = a.actor(ctx_root(), fed())
        assert len(r.calls) == 1 and not r.answer.fallback, fixture
        assert sum(ids(r.answer).values()) == pytest.approx(1.0)
        assert mock.PRICE_SENTENCE not in json.dumps(r.answer.options, ensure_ascii=False)   # the report quotes it


def test_refusals_and_declines_get_one_rephrase_then_the_fallback(mock_llm):
    for fixture in ("refusal", "refusal_text", "decline", "decline_actor"):
        mock_llm.push(fixture, match="actor")
        r = llm().actor(ctx_root(), fed())
        assert "rephrased" in r.flags and not r.answer.fallback, fixture
        assert [c.purpose for c in r.calls] == ["call", "rephrase"]
    seen = []
    mock_llm.push(lambda kind, s, u, ans: seen.append(u) or "I can't help with that.", match="actor", times=2)
    r = llm().actor(ctx_root(), fed())
    assert r.answer.fallback and r.answer.declined and r.answer.prior_only
    assert seen[1].startswith(P.REPHRASE) and not seen[0].startswith(P.REPHRASE)
    assert llm(rephrase=False).actor(ctx_root(), fed()).answer.fallback is False


@pytest.mark.parametrize("fixture, kind, reason", [("usage_limit", "limit", "usage_limit"),
                                                   ("usage_limit_text", "limit", "usage_limit"),
                                                   ("login", "login", "auth"), ("cli_missing", "cli_missing", "cli_missing"),
                                                   ("model_missing", "model", "model")])
def test_pause_classes_raise_llmpause(mock_llm, fixture, kind, reason):
    mock_llm.push(fixture, match="node")
    with pytest.raises(io.LLMPause) as err:
        llm().node(ctx_deep(), [fed(), china()])
    assert err.value.kind == kind and err.value.reason == reason and err.value.record.kind == "node"
    if fixture == "usage_limit":
        assert err.value.reset_at and io.llm_status()["limited"]
        assert "用量上限" in io.llm_status()["limited_reason"]


def test_the_circuit_breaker_pauses_after_three_failures_of_one_class(mock_llm):
    mock_llm.push("max_tokens", match="actor", times=None)
    a = llm()
    a.actor(ctx_root(), fed())
    a.actor(ctx_root(), china())
    with pytest.raises(io.LLMPause) as err:
        a.actor(ctx_root(), president())
    assert err.value.kind == "breaker" and "连续 3 次" in str(err.value)
    mock_llm.reset()
    mock_llm.push("max_tokens", match="actor", times=2)
    b = llm()
    b.actor(ctx_root(), fed())
    b.actor(ctx_root(), china())
    b.actor(ctx_root(), president())                                  # a success resets the streak
    mock_llm.push("max_tokens", match="actor", times=2)
    b.actor(ctx_root(), fed(zh="美联储2"))
    b.actor(ctx_root(), china(zh="中国2"))


def test_cancel_before_and_during_a_call(mock_llm, monkeypatch, tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(io.LLMPause) as err:
        llm(cancel=cancel).actor(ctx_root(), fed())
    assert err.value.kind == "cancelled" and err.value.reason == "cancelled"
    # force: the kill event stops a call in flight
    cancel, kill = threading.Event(), threading.Event()
    mock_llm.push("hang", match="actor")
    threading.Timer(0.1, lambda: (cancel.set(), kill.set())).start()
    t0 = time.perf_counter()
    with pytest.raises(io.LLMPause) as err:
        llm(cancel=cancel, kill=kill).actor(ctx_root(), fed())
    assert err.value.kind == "cancelled" and time.perf_counter() - t0 < 5
    # graceful (review E10): a call in flight finishes and is cached; the next one does not start
    monkeypatch.setenv("MSL_LLM_MOCK_DELAY_S", "0.4")
    cancel = threading.Event()
    a = llm(cancel=cancel, kill=threading.Event(), cache=io.FileCache(tmp_path / "graceful"))
    threading.Timer(0.1, cancel.set).start()
    r = a.actor(ctx_root(), fed())
    assert r.calls[0].ok and not r.answer.fallback and a.stats["calls_new"] == 1
    with pytest.raises(io.LLMPause):
        a.actor(ctx_root(), president())
    monkeypatch.delenv("MSL_LLM_MOCK_DELAY_S")
    again = llm(cache=io.FileCache(tmp_path / "graceful")).actor(ctx_root(), fed())
    assert again.calls[0].cached                                        # the finished call was kept


def test_at_most_concurrency_calls_run_at_once(monkeypatch):
    monkeypatch.setenv("MSL_LLM_MOCK_DELAY_S", "0.15")
    state = {"now": 0, "max": 0}
    lock = threading.Lock()
    real = backend.run

    def counting(*a, **k):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        try:
            return real(*a, **k)
        finally:
            with lock:
                state["now"] -= 1

    monkeypatch.setattr(backend, "run", counting)
    a = llm(concurrency=2)
    threads = [threading.Thread(target=a.actor, args=(ctx_root(), fed(zh=f"美联储{k}"))) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert state["max"] == 2 and a.stats["calls_new"] == 6


def test_events_and_the_first_failure_log(mock_llm, tmp_path, monkeypatch):
    events = []
    log = tmp_path / "cli_failures.jsonl"
    secret = "sk-ant-oat01-" + "Z9" * 30
    monkeypatch.setenv(cc.TOKEN_VAR, secret)

    def failing(kind, s, u, ans):
        raise cc.ClaudeCodeError("服务繁忙", "overloaded", detail=f"overloaded {secret}")

    mock_llm.push(failing, match="actor", times=None)
    a = llm(on_event=lambda event, **p: events.append((event, p)), failure_log=log)
    a.actor(ctx_root(), fed())
    a.actor(ctx_root(), china())
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1 and lines[0]["kind"] == "overloaded" and secret not in log.read_text(encoding="utf-8")
    kinds = [e for e, _ in events]
    assert kinds.count("call_started") == 6 and kinds.count("call_done") == 6
    assert events[1][1]["error"] == "overloaded" and events[1][1]["call"] == "actor"


# ---------------------------------------------------------------- the merged node

def test_merged_node_failures_per_actor(mock_llm):
    mock_llm.push("decline_actor", match="node")
    n = llm().node(ctx_deep(), [fed(), president(), china()])
    assert [c.purpose for c in n.calls] == ["call", "rephrase"] and n.calls[1].kind == "actor"
    assert not n.actors["fed"].fallback
    mock_llm.push("missing_actor", match="node")
    n = llm().node(ctx_deep(), [fed(), president(), china()])
    assert n.actors["fed"].fallback and "缺少该行为者" in n.actors["fed"].reason and len(n.calls) == 1
    mock_llm.push("bad_sum", match="node")
    n = llm().node(ctx_deep(), [fed(), president(), china()])
    assert [c.purpose for c in n.calls] == ["call", "repair"] and not n.actors["fed"].fallback
    mock_llm.push("decline", match="node", times=2)
    n = llm().node(ctx_deep(), [fed(), china()])
    assert all(x.fallback and x.declined for x in n.actors.values()) and "rephrased" in n.flags
    mock_llm.push("timeout", match="node", times=3)
    n = llm().node(ctx_deep(), [fed(), china()])
    assert all(x.prior_only for x in n.actors.values()) and "fallback" in n.flags


def test_synthesizer_failures_fall_back_to_the_default_partition(mock_llm):
    mock_llm.push("malformed_json", match="synth")
    mock_llm.push("malformed_json", match="repair")
    s = llm().synth(synth_request())
    assert all(c.status == "default" for c in s.clusters.values()) and "synth_default" in s.flags
    mock_llm.push("decline_actor", match="synth")                     # the first cluster declines
    s = llm().synth(synth_request())
    assert s.clusters["k1"].status == "default" and s.clusters["k1"].declined and s.clusters["k2"].status == "ok"
    mock_llm.push("decline", match="synth")
    s = llm().synth(synth_request())
    assert all(c.status == "default" for c in s.clusters.values())
    mock_llm.push("malformed_json", match="synth")
    s = llm().synth(synth_request())
    assert "repaired" in s.flags and all(c.status == "ok" for c in s.clusters.values())


def test_root_translation_failures(mock_llm):
    mock_llm.push("max_tokens", match="root")
    out = llm().root(ctx_root(), "加征关税", settable=[{"id": "d_tariff"}])
    assert not out.ok and out.error == "输出达到长度上限"
    out = llm().root(ctx_root(), "长" * 25000)
    assert out.ok and "已截断到 20000 字" in out.report[0]


# ---------------------------------------------------------------- catalogue adapter, settings, status

MENU = {"actor": "fed", "zh": "美联储（FOMC）", "active": True,
        "main": {"domain": "rate", "options": [
            {"id": "cut_25", "zh": "降息 25bp", "status_quo": False, "param": None,
             "writes": [{"role": "policy_rate_chg", "field": "d_ff", "kind": "point", "dq": 0, "a": -0.25, "b": 0.0}],
             "signal": {"text": "降息合适", "direction": -1, "horizon_q": 1}},
            {"id": "hold", "zh": "按兵不动", "status_quo": True, "param": None,
             "writes": [{"role": "policy_rate_chg", "field": "d_ff", "kind": "point", "dq": 0, "a": 0.0, "b": 0.0}]},
            {"id": "emergency_cut", "zh": "紧急降息", "status_quo": False, "note": "仅在金融事故后开放",
             "param": {"name": "size", "zh": "幅度", "unit": "pp", "lo": -1.75, "hi": -0.75, "default": -1.0, "step": 0.25},
             "writes": [{"role": "policy_rate_chg", "field": "d_ff", "kind": "point", "dq": 0, "a": None, "b": 0.0}]},
            {"id": "as_committed", "zh": "如期执行（降息）", "status_quo": False, "param": None, "commitment": True}],
            "required": ["as_committed"]}}


def test_actor_spec_from_the_catalogues_menu_and_prior():
    spec = io.actor_spec_from_menu(MENU, {"p": {"cut_25": 0.3, "hold": 0.6, "emergency_cut": 0.1}, "source": "有序 probit",
                                          "kind": "history"}, instruments=["d_ff"], packet="账户")
    assert spec.option_ids == ["hold", "cut_25", "emergency_cut", "as_committed"]
    cut = spec.option("emergency_cut")
    assert cut.magnitude.lo == -1.75 and cut.writes == "d_ff = 幅度（pp）" and cut.point == {"d_ff": -1.0}
    assert spec.option("cut_25").writes == "d_ff = -0.25" and spec.option("cut_25").signal == "降息合适"
    assert spec.option("as_committed").required and spec.option("hold").base_rate == 0.6
    r = llm().actor(ctx_root(), spec)
    assert ids(r.answer)["hold"] == 0.94 and r.answer.missing_required == ["as_committed"]
    uniform = io.actor_spec_from_menu(MENU, {"p": {"hold": 0.25}, "kind": "uniform"})
    assert not uniform.has_base_rates


def test_the_catalogue_menu_renders_when_available():
    actors = pytest.importorskip("scenariolab.agents.actors")
    if not hasattr(actors, "menu"):
        pytest.skip("no catalogue menu yet")
    try:
        menu = actors.menu("fed")
    except Exception as err:                       # the catalogue's own preconditions (data files) are not ours
        pytest.skip(f"catalogue menu unavailable: {err}")
    if not (menu or {}).get("main"):
        pytest.skip("fed inactive")
    spec = io.actor_spec_from_menu(menu)
    schema = P.actor_output_schema(spec)
    text, _ = mock.run(P.actor_system(schema), P.actor_user(ctx_root(), spec), "claude-opus-5-5")
    assert jf.validate(json.loads(text), schema) == []


def test_settings_coercion_and_status():
    s = io.Settings.coerce({"effort": {"node": "low"}, "timeouts": {"node": 30}, "max_calls": 5, "unknown": 1})
    assert s.effort["node"] == "low" and s.effort["actor_root"] == "medium" and s.timeouts["node"] == 30.0
    assert set(io.DEFAULT_EFFORT.values()) == {"medium"}               # opus 5.5 at medium for every agent-tree call
    assert s.max_calls == 5 and io.Settings.coerce(None).max_calls == 60
    st = io.llm_status()
    assert st["available"] and st["backend"] == "mock" and st["mock"] and st["cli_version"] == mock.MOCK_VERSION
    assert set(st) >= {"available", "backend", "cli_version", "model", "reason", "concurrency", "latency", "limited"}
    assert io.action_for("timeout") == "retry" and io.action_for("limit") == "pause"
    assert io.action_for("refusal") == "prior_only" and io.action_for("unparseable") == "retry_once"
    assert io.action_for("cancelled") == "cancel"


def test_timeouts_never_exceed_msl_llm_timeout(monkeypatch):
    monkeypatch.setenv("MSL_LLM_TIMEOUT", "45")
    a = llm()
    assert a._timeout("node") == 45.0 and a._timeout("probe") == 45.0
    monkeypatch.setenv("MSL_LLM_TIMEOUT", "600")
    assert a._timeout("node") == 300.0 and a._timeout("repair") == 90.0


def test_real_claude_code_calls_need_the_users_confirmation(fake_cli, tmp_path):
    """With the claude_code backend nothing starts until the run is confirmed; cached answers replay without it."""
    cache = io.FileCache(tmp_path / "cache")
    unconfirmed = io.AgentLLM(io.Settings(backend="claude_code"), cache=cache, sleep=SLEEPS.append)
    with pytest.raises(io.LLMPause) as err:
        unconfirmed.actor(ctx_root(), fed())
    assert err.value.kind == "unconfirmed" and err.value.reason == "unconfirmed" and "确认" in str(err.value)
    assert fake_cli.print_calls == []                                         # nothing was started
    schema = P.actor_output_schema(fed())
    fake_cli.result, _ = mock.run(P.actor_system(schema, ctx_root()), P.actor_user(ctx_root(), fed()), "m")
    confirmed = io.AgentLLM(io.Settings(backend="claude_code", confirmed=True), cache=cache, sleep=SLEEPS.append)
    r = confirmed.actor(ctx_root(), fed())
    assert len(fake_cli.print_calls) == 1 and not r.answer.fallback and r.calls[0].backend == "claude_code"
    args = fake_cli.print_calls[0][0]
    assert args[-3:] == ["--effort", "medium", "--safe-mode"]                 # the fake CLI's help lists both
    replay = io.AgentLLM(io.Settings(backend="claude_code"), cache=cache, sleep=SLEEPS.append)
    assert replay.actor(ctx_root(), fed()).calls[0].cached and len(fake_cli.print_calls) == 1


def test_an_identical_call_that_waited_for_a_slot_reuses_the_cached_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("MSL_LLM_MOCK_DELAY_S", "0.2")
    a = llm(concurrency=1, cache=io.FileCache(tmp_path / "cache"))
    results = []
    threads = [threading.Thread(target=lambda: results.append(a.actor(ctx_root(), fed()))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert a.stats["calls_new"] == 1 and a.stats["cached"] == 1
    assert sorted(r.calls[0].cached for r in results) == [False, True]
    assert results[0].answer.to_dict() == results[1].answer.to_dict()
