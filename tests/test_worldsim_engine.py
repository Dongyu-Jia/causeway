"""One WorldSim iteration end to end on the mock backend: actors in parallel, the synthesizer, validation and
persistence; the failure paths (an actor that fails, bad JSON and its repair, a synthesizer failure that leaves the
state unchanged and its retry); rewind, fork and the cost estimate."""
import json
import threading
import time

import pytest
from worldsim_fakes import small_world

from worldsim import model as M
from worldsim import settings as S
from worldsim.engine import Engine, RunManager
from worldsim.llm import CallRequest, Completion, LLMError, MockBackend, extract_json
from worldsim.store import Conflict, NotFound, Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "ws")


def engine_for(store, backend):
    return Engine(store, lambda: S.load(store.root), backend_factory=lambda s: backend)


def test_an_iteration_runs_actors_then_the_synthesizer_and_saves_the_next_step(store):
    w = store.create_world(small_world())
    backend = MockBackend(delay_s=0)
    result = engine_for(store, backend).run(w["id"])
    assert result["status"] == "done" and result["step"] == 1 and result["period"] == "2027Q1"
    assert store.step_numbers(w["id"]) == [0, 1]
    state = store.load_step(w["id"], 1)
    assert state["period"] == "2027Q1" and set(state["values"]) == {"rate", "tension", "phase"}
    assert state["text"]["narrative"].startswith("[mock] 2027Q1")
    it = store.load_iteration(w["id"], 1)
    assert [r["status"] for r in it["actors"]] == ["ok", "ok", "ok"]
    assert it["synthesizer"]["status"] == "ok" and it["usage"]["calls"] == 4 and it["cost"] == 0.0
    # every call is saved with its prompt and raw response
    assert len(it["calls"]) == 4 and all(c["messages"] and c["response_text"] for c in it["calls"])
    purposes = [c.purpose for c in backend.calls]
    assert purposes[-1] == "synth" and purposes.count("actor") == 3
    # each actor sees only its own private information; the synthesizer sees every actor's output
    for c in backend.calls:
        if c.purpose == "actor":
            me = c.context["actor"]["id"]
            text = json.dumps(c.messages, ensure_ascii=False)
            assert f"SECRET-{me[1:]}" in text and sum(f"SECRET-{i}" in text for i in range(3)) == 1
    synth_text = backend.calls[-1].messages[1]["content"]
    assert all(f"Actor {i}" in synth_text for i in range(3)) and "SECRET" not in synth_text


def test_three_iterations_build_a_history_that_later_prompts_see(store):
    w = store.create_world(small_world())
    backend = MockBackend(delay_s=0)
    eng = engine_for(store, backend)
    for _ in range(3):
        assert eng.run(w["id"])["status"] == "done"
    assert [store.load_step(w["id"], n)["period"] for n in range(4)] == ["2026Q4", "2027Q1", "2027Q2", "2027Q3"]
    last_actor_prompt = [c for c in backend.calls if c.purpose == "actor"][-1].messages[1]["content"]
    assert "Step 2 (2027Q2)" in last_actor_prompt and "Actions that led to it" in last_actor_prompt


def test_actors_are_called_in_parallel_up_to_max_parallel(store):
    w = store.create_world(small_world(n_actors=6))
    active, peak, lock = [0], [0], threading.Lock()

    class Slow(MockBackend):
        def complete(self, req):
            if req.purpose == "actor":
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.15)
                with lock:
                    active[0] -= 1
            return super().complete(req)

    S.save({**S.DEFAULTS, "max_parallel": 3}, store.root)
    t0 = time.monotonic()
    assert engine_for(store, Slow(delay_s=0)).run(w["id"])["status"] == "done"
    assert peak[0] == 3 and time.monotonic() - t0 < 6 * 0.15


def test_a_failing_actor_is_retried_once_then_marked_no_response_and_the_synthesizer_is_told(store):
    w = store.create_world(small_world())
    backend = MockBackend(delay_s=0, script={"actor:a0": ["error", "ok"], "actor:a1": ["error", "timeout"]})
    assert engine_for(store, backend).run(w["id"])["status"] == "done"
    it = store.load_iteration(w["id"], 1)
    by = {r["actor_id"]: r for r in it["actors"]}
    assert by["a0"]["status"] == "ok" and by["a0"]["attempts"] == 2
    assert by["a1"]["status"] == "no_response" and by["a1"]["attempts"] == 2 and "timeout" in by["a1"]["error"]
    assert by["a2"]["status"] == "ok" and by["a2"]["attempts"] == 1
    synth_prompt = backend.calls[-1].messages[1]["content"]
    assert "Actor 1 [a1]: NO RESPONSE" in synth_prompt
    assert "No response: Actor 1" in store.load_step(w["id"], 1)["text"]["narrative"]


def test_bad_json_gets_one_repair_call(store):
    w = store.create_world(small_world(n_actors=1))
    backend = MockBackend(delay_s=0, script={"actor:a0": ["bad_json"], "synth": ["bad_json"]})
    assert engine_for(store, backend).run(w["id"])["status"] == "done"
    it = store.load_iteration(w["id"], 1)
    assert [(c["who"], c["repair"]) for c in it["calls"]] == [("a0", False), ("a0", True), ("synthesizer", False),
                                                                ("synthesizer", True)]
    assert it["calls"][0]["error"].startswith("unusable answer") and it["calls"][1]["error"] is None
    repair = [c for c in backend.calls if c.repair][0]
    assert repair.messages[-2]["role"] == "assistant" and "could not be used" in repair.messages[-1]["content"]


def test_an_actor_with_bad_json_twice_falls_back_to_a_retry(store):
    w = store.create_world(small_world(n_actors=1))
    backend = MockBackend(delay_s=0, script={"actor:a0": ["bad_json", "ok"], "actor:a0:repair": ["bad_json"]})
    assert engine_for(store, backend).run(w["id"])["status"] == "done"
    r = store.load_iteration(w["id"], 1)["actors"][0]
    assert r["status"] == "ok" and r["attempts"] == 2


def test_an_action_outside_the_space_is_repaired_or_becomes_a_free_action(store):
    wd = small_world(n_actors=2)
    wd["actors"][0]["allow_free_action"] = False
    w = store.create_world(wd)
    backend = MockBackend(delay_s=0, script={"actor:a0": ["invalid"], "actor:a1": ["invalid"]})
    assert engine_for(store, backend).run(w["id"])["status"] == "done"
    by = {r["actor_id"]: r for r in store.load_iteration(w["id"], 1)["actors"]}
    assert by["a0"]["status"] == "ok" and by["a0"]["output"]["action_id"] in ("hold", "push")     # after the repair
    assert by["a1"]["output"]["action_id"] == "free" and "no_such_action" in by["a1"]["output"]["action_detail"]
    assert by["a1"]["notes"]


def test_a_synthesizer_failure_leaves_the_state_unchanged_and_a_retry_completes_it(store):
    w = store.create_world(small_world())
    backend = MockBackend(delay_s=0, script={"synth": ["bad_json", "ok"], "synth:repair": ["bad_json"]})
    eng = engine_for(store, backend)
    result = eng.run(w["id"])
    assert result["status"] == "failed" and "bad answer after one repair" in result["error"]
    assert store.step_numbers(w["id"]) == [0]
    pending = store.load_pending(w["id"])
    assert pending["from_step"] == 0 and len(pending["actors"]) == 3 and len(pending["calls"]) == 5
    n_actor_calls = sum(c.purpose == "actor" for c in backend.calls)
    result = eng.retry(w["id"])
    assert result["status"] == "done" and store.step_numbers(w["id"]) == [0, 1]
    assert sum(c.purpose == "actor" for c in backend.calls) == n_actor_calls      # the actors were not called again
    assert store.load_pending(w["id"]) is None
    it = store.load_iteration(w["id"], 1)
    assert len(it["calls"]) == 6 and it["usage"]["calls"] == 6                    # the failed attempt's calls are kept


def test_a_synthesizer_error_is_shown_and_a_retry_without_pending_is_refused(store):
    w = store.create_world(small_world())
    eng = engine_for(store, MockBackend(delay_s=0, script={"synth": ["error"]}))
    result = eng.run(w["id"])
    assert result["status"] == "failed" and "injected failure" in result["error"]
    store.clear_pending(w["id"])
    with pytest.raises(NotFound):
        eng.retry(w["id"])


def test_a_stale_pending_attempt_is_refused(store):
    w = store.create_world(small_world())
    eng = engine_for(store, MockBackend(delay_s=0, script={"synth": ["error"]}))
    eng.run(w["id"])
    pending = store.load_pending(w["id"])
    store.save_pending(w["id"], {**pending, "from_step": 7})
    with pytest.raises(Conflict):
        eng.retry(w["id"])


def test_rewind_and_fork(store):
    w = store.create_world(small_world())
    eng = engine_for(store, MockBackend(delay_s=0))
    for _ in range(3):
        eng.run(w["id"])
    fork = store.fork(w["id"], 2, "Fork")
    assert fork["name"] == "Fork" and store.step_numbers(fork["id"]) == [0, 1, 2]
    assert store.load_step(fork["id"], 2) == store.load_step(w["id"], 2)
    assert store.rewind(w["id"], 1) == 2 and store.step_numbers(w["id"]) == [0, 1]
    assert eng.run(w["id"])["step"] == 2
    with pytest.raises(NotFound):
        store.rewind(w["id"], 9)


def test_the_run_manager_runs_in_the_background_and_refuses_a_second_run(store):
    w = store.create_world(small_world())
    rm = RunManager(engine_for(store, MockBackend(delay_s=0.2)))
    snap = rm.start(w["id"])
    assert snap["status"] == "running" and [a["status"] for a in snap["actors"]] == ["queued"] * 3
    with pytest.raises(Exception, match="already running"):
        rm.start(w["id"])
    final = rm.wait(w["id"], 30)
    assert final["status"] == "done" and final["step"] == 1
    assert all(a["status"] == "done" and a["action_label"] for a in final["actors"])
    assert final["synthesizer"]["status"] == "done"


def test_the_estimate_prices_every_call_with_the_model_list(store):
    w = store.create_world(small_world())
    est = engine_for(store, MockBackend(delay_s=0)).estimate(w["id"])
    assert est["calls"] == 4 and est["priced"] and est["cost"] > 0
    assert {r["model"] for r in est["rows"]} == {"google/gemini-4-argon"}
    assert est["prompt_tokens"] > 0 and est["next_period"] == "2027Q1"


def test_model_overrides_reach_the_requests(store):
    wd = small_world(n_actors=2)
    wd["actors"][1]["model"] = "openai/gpt-x"
    wd["synthesizer"]["model"] = "anthropic/claude-y"
    wd["model"] = {"model": "google/world-model", "temperature": 0.2, "reasoning_effort": "low"}
    w = store.create_world(wd)
    backend = MockBackend(delay_s=0)
    engine_for(store, backend).run(w["id"])
    models = {(c.purpose, c.context.get("actor", {}).get("id")): c for c in backend.calls}
    assert models[("actor", "a0")].model == "google/world-model" and models[("actor", "a1")].model == "openai/gpt-x"
    assert models[("synth", None)].model == "anthropic/claude-y"
    assert models[("actor", "a0")].temperature == 0.2 and models[("actor", "a0")].reasoning_effort == "low"


def test_visible_fields_limit_what_an_actor_sees(store):
    wd = small_world(n_actors=1)
    wd["actors"][0]["visible_fields"] = ["rate"]
    w = store.create_world(wd)
    backend = MockBackend(delay_s=0)
    engine_for(store, backend).run(w["id"])
    prompt = backend.calls[0].messages[1]["content"]
    assert "Rate [rate]" in prompt and "Tension [tension]" not in prompt


def test_an_unexpected_backend_exception_in_one_actor_does_not_stop_the_iteration(store):
    w = store.create_world(small_world())

    class Buggy(MockBackend):
        def complete(self, req):
            if req.purpose == "actor" and req.context["actor"]["id"] == "a2":
                raise KeyError("boom")
            return super().complete(req)
    assert engine_for(store, Buggy(delay_s=0)).run(w["id"])["status"] == "done"
    by = {r["actor_id"]: r for r in store.load_iteration(w["id"], 1)["actors"]}
    assert by["a2"]["status"] == "no_response" and "internal error" in by["a2"]["error"]


def test_a_non_retryable_error_is_not_retried(store):
    w = store.create_world(small_world(n_actors=1))

    class NoKey(MockBackend):
        def complete(self, req):
            self.calls.append(req)
            raise LLMError("no OpenRouter key", "auth", retryable=False)
    backend = NoKey(delay_s=0)
    result = engine_for(store, backend).run(w["id"])
    assert result["status"] == "failed" and len(backend.calls) == 2      # one actor call, one synthesizer call


@pytest.mark.parametrize("text, ok", [
    ('{"a": 1}', True), ('```json\n{"a": 1,}\n```', True), ('Here: {"a": True, "b": None} thanks', True),
    ('{“a”: 1}', True), ('{"a": "x，y"}', True), ('{"a": 1', False), ('[1, 2]', False), ('', False)])
def test_extract_json(text, ok):
    obj, err = extract_json(text)
    assert (obj is not None) == ok, err
    if ok:
        assert "a" in obj
    if text == '{"a": "x，y"}':
        assert obj["a"] == "x，y"                     # punctuation inside strings is left alone


def test_completion_and_request_types_are_plain():
    req = CallRequest(purpose="actor", model="m", messages=[], schema_name="s", json_schema={})
    assert req.repair is False and Completion("t", {}, None, "m", "json_object").meta == {}
    assert M.FREE_ACTION == "free"
