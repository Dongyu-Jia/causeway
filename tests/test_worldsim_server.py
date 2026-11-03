"""The WorldSim server: world CRUD, schema / initial state / actors editing, runs with the mock backend, rewind and
fork, export / import, settings and the key (stored 0600, masked, never returned or saved elsewhere), the origin
guard, and the static UI. TestClient only; the OpenRouter transport is a fake."""
import json
import os
import stat

import pytest
from fastapi.testclient import TestClient
from worldsim_fakes import TEST_KEY, FakeTransport, small_world

from worldsim import config
from worldsim.server import create_app

BASE = "http://127.0.0.1:8940"


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path / "ws", transport=FakeTransport())


@pytest.fixture
def client(app):
    return TestClient(app, base_url=BASE)


def mock_backend(client):
    assert client.put("/api/settings", json={"backend": "mock"}).status_code == 200


def run(client, app, wid, mode="run"):
    r = client.post(f"/api/worlds/{wid}/{mode}", json={})
    assert r.status_code == 200, r.text
    app.state.runs.wait(wid, 60)
    return client.get(f"/api/worlds/{wid}/run").json()["run"]


def new_world(client, **over):
    w = small_world()
    w.update(over)
    r = client.post("/api/worlds/import", json=w)
    assert r.status_code == 200, r.text
    return r.json()["world"]


# ------------------------------------------------------------------------------------------------ worlds

def test_the_default_world_is_preloaded_on_first_start(client):
    worlds = client.get("/api/worlds").json()
    assert [w["id"] for w in worlds["worlds"]] == ["us_2026q4"] and worlds["template"]
    w = worlds["worlds"][0]
    assert w["name"] == "美国与世界 · 2026Q4" and w["period"] == "2026Q4" and w["actors"] == 14 and w["fields"] >= 45
    full = client.get("/api/worlds/us_2026q4").json()
    assert full["steps"] == [0] and full["pending"] is None and full["busy"] is False


def test_world_crud(client):
    r = client.post("/api/worlds", json={"mode": "blank", "name": "Blank one"})
    wid = r.json()["world"]["id"]
    assert r.status_code == 200 and r.json()["world"]["name"] == "Blank one"
    r = client.post("/api/worlds", json={"mode": "template", "name": "Copy of the template"})
    assert r.status_code == 200 and len(r.json()["world"]["actors"]) == 14
    r = client.put(f"/api/worlds/{wid}", json={"name": "Renamed", "language": "中文",
                                                "period_step": {"unit": "month", "count": 2}, "history_window": 5})
    assert r.status_code == 200 and r.json()["world"]["period_step"] == {"unit": "month", "count": 2}
    bad = client.put(f"/api/worlds/{wid}", json={"period_step": {"unit": "fortnight"}})
    assert bad.status_code == 400 and bad.json()["detail"]["problems"]
    dup = client.post(f"/api/worlds/{wid}/duplicate", json={"name": "Dup"}).json()["world"]
    assert dup["name"] == "Dup" and dup["id"] != wid
    assert client.request("DELETE", f"/api/worlds/{wid}", json={}).status_code == 400          # no confirmation
    assert client.request("DELETE", f"/api/worlds/{wid}", json={"confirm": True}).status_code == 200
    assert client.get(f"/api/worlds/{wid}").status_code == 404
    assert client.get("/api/worlds/..%2F..%2Fetc").status_code == 404
    names = {w["name"] for w in client.get("/api/worlds").json()["worlds"]}
    assert names == {"美国与世界 · 2026Q4", "Copy of the template", "Dup"}


def test_schema_initial_state_actors_and_synthesizer_are_edited_and_validated(client):
    w = new_world(client)
    wid = w["id"]
    fields = w["schema"]["fields"] + [{"id": "gdp", "label": "GDP", "group": "Growth", "type": "number", "min": -10, "max": 10}]
    r = client.put(f"/api/worlds/{wid}/schema", json={"fields": fields, "initial_values": {"gdp": 2.5, "rate": 4}})
    assert r.status_code == 200 and r.json()["world"]["initial_state"]["values"]["gdp"] == 2.5
    bad = client.put(f"/api/worlds/{wid}/schema", json={"fields": [{"id": "Bad!", "type": "number"}]})
    assert bad.status_code == 400 and "id must be" in " ".join(bad.json()["detail"]["problems"])
    r = client.put(f"/api/worlds/{wid}/initial", json={"period": "2027Q1", "values": {"rate": "5%"},
                                                         "text": {"narrative": "New start", "key_questions": [], "attributes": []}})
    assert r.status_code == 200
    init = r.json()["world"]["initial_state"]
    assert init["period"] == "2027Q1" and init["values"]["rate"] == 5.0 and init["text"]["narrative"] == "New start"
    assert client.get(f"/api/worlds/{wid}/steps/0").json()["state"]["period"] == "2027Q1"     # step 0 follows
    bad = client.put(f"/api/worlds/{wid}/initial", json={"values": {"phase": "nonsense"}})
    assert bad.status_code == 400
    actors = w["actors"] + [{"id": "a0", "name": "dup id", "action_space": [{"id": "x", "label": "X"}]}]
    bad = client.put(f"/api/worlds/{wid}/actors", json={"actors": actors})
    assert bad.status_code == 400 and "used twice" in " ".join(bad.json()["detail"]["problems"])
    actors[-1]["id"] = "a9"
    r = client.put(f"/api/worlds/{wid}/actors", json={"actors": actors})
    assert r.status_code == 200 and len(r.json()["world"]["actors"]) == 4
    r = client.put(f"/api/worlds/{wid}/synthesizer", json={"instructions": "Be bold.", "model": "acme/x"})
    assert r.json()["world"]["synthesizer"] == {"instructions": "Be bold.", "model": "acme/x"}


# ------------------------------------------------------------------------------------------------ running

def test_a_run_through_the_api_saves_a_step_viewable_with_its_iteration_and_calls(client, app):
    mock_backend(client)
    wid = new_world(client)["id"]
    est = client.get(f"/api/worlds/{wid}/estimate").json()
    assert est["backend"] == "mock" and est["calls"] == 4
    final = run(client, app, wid)
    assert final["status"] == "done" and final["step"] == 1 and final["period"] == "2027Q1"
    assert [s["step"] for s in client.get(f"/api/worlds/{wid}/steps").json()["steps"]] == [0, 1]
    view = client.get(f"/api/worlds/{wid}/steps/1").json()
    assert view["previous"]["step"] == 0 and view["latest"] == 1
    it = view["iteration"]
    assert len(it["actors"]) == 3 and all("messages" not in c and "response_text" not in c for c in it["calls"])
    calls = client.get(f"/api/worlds/{wid}/iterations/1/calls").json()["calls"]
    assert len(calls) == 4 and all(c["messages"] and c["response_text"] for c in calls)


def test_without_a_key_the_openrouter_backend_refuses_to_run(client):
    wid = new_world(client)["id"]
    r = client.post(f"/api/worlds/{wid}/run", json={})
    assert r.status_code == 400 and "No OpenRouter key" in r.json()["detail"]


def test_a_world_without_enabled_actors_cannot_run(client):
    mock_backend(client)
    w = small_world()
    for a in w["actors"]:
        a["enabled"] = False
    wid = new_world(client, actors=w["actors"])["id"]
    assert client.post(f"/api/worlds/{wid}/run", json={}).status_code == 400


def test_a_synthesizer_failure_is_shown_and_retried_through_the_api(tmp_path):
    from worldsim.llm import MockBackend
    backend = MockBackend(delay_s=0, script={"synth": ["error", "ok"]})
    app = create_app(tmp_path / "ws", transport=FakeTransport(), backend_factory=lambda s: backend)
    client = TestClient(app, base_url=BASE)
    mock_backend(client)
    wid = new_world(client)["id"]
    final = run(client, app, wid)
    assert final["status"] == "failed" and "injected failure" in final["error"]
    info = client.get(f"/api/worlds/{wid}").json()
    assert info["steps"] == [0] and "injected failure" in info["pending"]["error"] and "calls" not in info["pending"]
    assert len(client.get(f"/api/worlds/{wid}/pending/calls").json()["calls"]) == 4
    final = run(client, app, wid, mode="retry")
    assert final["status"] == "done" and client.get(f"/api/worlds/{wid}").json()["steps"] == [0, 1]
    assert client.post(f"/api/worlds/{wid}/retry", json={}).status_code == 404               # nothing left to retry


def test_the_mock_script_env_drives_the_failure_path_in_the_real_app(tmp_path, monkeypatch):
    from worldsim import llm
    monkeypatch.setattr(llm, "_SCRIPTED", {})
    monkeypatch.setenv("WORLDSIM_LLM", "mock")
    monkeypatch.setenv("WORLDSIM_MOCK_SCRIPT", '{"synth": ["error"]}')
    app = create_app(tmp_path / "ws", transport=FakeTransport())
    client = TestClient(app, base_url=BASE)
    assert client.get("/api/settings").json()["effective_backend"] == "mock"
    wid = new_world(client)["id"]
    assert run(client, app, wid)["status"] == "failed"
    assert run(client, app, wid, mode="retry")["status"] == "done"


def test_after_a_step_the_initial_state_is_read_only_but_the_schema_can_grow(client, app):
    mock_backend(client)
    w = new_world(client)
    wid = w["id"]
    run(client, app, wid)
    r = client.put(f"/api/worlds/{wid}/initial", json={"period": "2030Q1"})
    assert r.status_code == 409 and "rewind" in r.json()["detail"]
    fields = w["schema"]["fields"] + [{"id": "extra", "label": "Extra", "type": "number"}]
    r = client.put(f"/api/worlds/{wid}/schema", json={"fields": fields, "initial_values": {"rate": 99}})
    assert r.status_code == 409                                                               # would change step 0
    r = client.put(f"/api/worlds/{wid}/schema", json={"fields": fields})
    assert r.status_code == 200
    run(client, app, wid)
    st = client.get(f"/api/worlds/{wid}/steps/2").json()["state"]
    assert st["values"]["extra"] == 0.0 and "extra" in st["reasons"]                         # the mock filled it


def test_rewind_and_fork_through_the_api(client, app):
    mock_backend(client)
    wid = new_world(client)["id"]
    for _ in range(3):
        run(client, app, wid)
    r = client.post(f"/api/worlds/{wid}/fork", json={"step": 2, "name": "Branch"})
    fork = r.json()["world"]
    assert r.status_code == 200 and fork["name"] == "Branch"
    assert [s["step"] for s in client.get(f"/api/worlds/{fork['id']}/steps").json()["steps"]] == [0, 1, 2]
    assert client.post(f"/api/worlds/{wid}/rewind", json={"step": 1}).status_code == 400      # needs confirm
    r = client.post(f"/api/worlds/{wid}/rewind", json={"step": 1, "confirm": True})
    assert r.json() == {"ok": True, "dropped": 2, "step": 1}
    assert client.post(f"/api/worlds/{wid}/rewind", json={"step": 7, "confirm": True}).status_code == 404
    assert client.post(f"/api/worlds/{wid}/fork", json={"step": "x"}).status_code == 400


def test_export_and_import_round_trip_with_history(client, app):
    mock_backend(client)
    wid = new_world(client)["id"]
    run(client, app, wid)
    r = client.get(f"/api/worlds/{wid}/export")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    data = r.json()
    assert len(data["history"]["steps"]) == 2 and len(data["history"]["iterations"]) == 1
    imported = client.post("/api/worlds/import", json=data).json()["world"]
    assert imported["id"] != wid
    assert client.get(f"/api/worlds/{imported['id']}/steps/1").json()["state"]["period"] == "2027Q1"
    bad = client.post("/api/worlds/import", json={"name": "x", "schema": {"fields": [{"id": "1bad"}]}})
    assert bad.status_code == 400


# ------------------------------------------------------------------------------------------------ settings and the key

def test_the_key_is_stored_0600_masked_and_never_returned(client, tmp_path):
    r = client.put("/api/settings", json={"api_key": TEST_KEY, "temperature": 0.3})
    assert r.status_code == 200 and TEST_KEY not in r.text
    pub = client.get("/api/settings").json()
    assert pub["has_key"] and pub["key_masked"] == "sk-or-…cdef" and pub["temperature"] == 0.3
    path = tmp_path / "ws" / "settings.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600 and json.loads(path.read_text())["api_key"] == TEST_KEY
    assert client.put("/api/settings", json={"api_key": ""}).json()["has_key"]                  # a blank box keeps it
    bad = client.put("/api/settings", json={"temperature": 9})
    assert bad.status_code == 400 and client.get("/api/settings").json()["temperature"] == 0.3
    assert client.put("/api/settings", json={"clear_key": True}).json()["has_key"] is False
    assert client.post("/api/settings/test_key", json={}).json() == {"ok": False, "message": "No OpenRouter key is set."}


def test_the_test_key_button_calls_get_key(tmp_path):
    fake = FakeTransport()
    client = TestClient(create_app(tmp_path / "ws", transport=fake), base_url=BASE)
    client.put("/api/settings", json={"api_key": TEST_KEY})
    r = client.post("/api/settings/test_key", json={})
    assert r.json()["ok"] and r.json()["info"]["limit_remaining"] == 7.5 and TEST_KEY not in r.text
    assert fake.requests[-1]["url"] == "https://openrouter.ai/api/v1/key"


def test_an_openrouter_run_leaves_the_key_only_in_settings(tmp_path, caplog):
    caplog.set_level("DEBUG")
    fake = FakeTransport()
    app = create_app(tmp_path / "ws", transport=fake)
    client = TestClient(app, base_url=BASE)
    client.put("/api/settings", json={"api_key": TEST_KEY, "backend": "openrouter"})
    wid = new_world(client)["id"]
    final = run(client, app, wid)
    assert final["status"] == "done" and len(fake.chats) == 4 and final["cost"] == pytest.approx(0.0123 * 4)
    assert fake.chats[0]["body"]["model"] == "google/gemini-4-argon"
    for path in (tmp_path / "ws").rglob("*"):
        if path.is_file():
            assert (TEST_KEY in path.read_text(encoding="utf-8", errors="replace")) == (path.name == "settings.json"), path
    for url in ("/api/settings", "/api/worlds", f"/api/worlds/{wid}", f"/api/worlds/{wid}/steps/1",
                f"/api/worlds/{wid}/iterations/1/calls", f"/api/worlds/{wid}/export", f"/api/worlds/{wid}/run", "/api/health"):
        assert TEST_KEY not in client.get(url).text, url
    assert TEST_KEY not in caplog.text


# ------------------------------------------------------------------------------------------------ the guard and the UI

@pytest.mark.parametrize("headers, ok", [
    ({}, True),
    ({"Origin": BASE}, True),
    ({"Origin": "http://localhost:8940", "Host": "localhost:8940"}, True),
    ({"Origin": "https://evil.example"}, False),
    ({"Origin": "null"}, False),
    ({"Sec-Fetch-Site": "cross-site"}, False),
    ({"Sec-Fetch-Site": "same-site"}, False),
    ({"Sec-Fetch-Site": "same-origin", "Origin": BASE}, True),
    ({"Host": "evil.example"}, False),
])
def test_the_origin_guard(client, headers, ok):
    r = client.put("/api/settings", json={"temperature": 0.5}, headers=headers)
    assert (r.status_code == 200) == ok, (r.status_code, r.text)
    if not ok:
        assert r.status_code == 403


def test_changes_need_a_json_body(client):
    for kw in ({"content": "temperature=1", "headers": {"Content-Type": "application/x-www-form-urlencoded"}},
               {"content": '{"temperature": 1}', "headers": {"Content-Type": "text/plain"}}):
        r = client.put("/api/settings", **kw)
        assert r.status_code == 403 and "JSON body" in r.json()["detail"]
    r = client.request("DELETE", "/api/worlds/us_2026q4")
    assert r.status_code == 403 and client.get("/api/worlds/us_2026q4").status_code == 200
    r = client.put("/api/settings", content="[1, 2]", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_reads_from_another_host_name_are_refused(client):
    assert client.get("/api/worlds", headers={"Host": "attacker.example"}).status_code == 403
    assert client.get("/", headers={"Host": "attacker.example"}).status_code == 403
    assert client.get("/api/nope").status_code == 404


def test_the_ui_is_served_with_a_strict_content_security_policy(client):
    r = client.get("/")
    assert r.status_code == 200 and '<script type="module" src="js/app.js">' in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"] and r.headers["x-content-type-options"] == "nosniff"
    for path in ("/js/app.js", "/js/api.js", "/js/ui.js", "/css/app.css") + tuple(
            f"/js/pages/{p}.js" for p in ("sim", "actors", "schema", "synth", "settings", "worlds")):
        assert client.get(path).status_code == 200, path


def test_the_ui_never_renders_server_text_as_html():
    web = config.WEB_DIR / "js"
    for path in list(web.glob("*.js")) + list((web / "pages").glob("*.js")):
        src = path.read_text(encoding="utf-8")
        assert "innerHTML" not in src.replace("'innerHTML'", "") and "insertAdjacentHTML" not in src, path.name
        assert "outerHTML" not in src and "document.write" not in src, path.name


def test_the_models_endpoint_works_offline(client):
    data = client.get("/api/models").json()
    assert data["source"] == "builtin" and data["models"][0]["id"] == "google/gemini-4-argon"
    assert data["default_model"] == "google/gemini-4-argon"
