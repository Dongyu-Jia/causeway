"""The end-to-end narrative test (SIM_DESIGN §8; §14 P7): scenariolab/evaluation/narrative/.

P7 exit tests:
  * the prompt builder cannot import the key (the import graph, a fresh interpreter, and building every prompt and
    running a mock translation with key.json made unreadable);
  * the mock run passes end to end (seal -> estimate -> run -> score -> audit -> report; data-dependent);
  * statistics recomputed from the research scorer's stored rows (score_rows.csv) equal its published pooled numbers
    on the original 11 episodes (equity9 MAE within 0.002) and the critic's subsets (within the printed precision);
  * /run refuses without a fresh estimate id and confirm: true;
  * pytest runs the mock backend, and the runner refuses any other backend under pytest.
Tests that need the local quarterly panel (data/, git-ignored) skip cleanly without it; the rest run anywhere. No
test spends plan usage or downloads anything.
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scenariolab.evaluation.narrative import common as C
from scenariolab.evaluation.narrative import episodes as EP
from scenariolab.evaluation.narrative import key as K
from scenariolab.evaluation.narrative import prompts as PR
from scenariolab.evaluation.narrative import research as RS
from scenariolab.evaluation.narrative import runner as RN
from scenariolab.evaluation.narrative import stats as ST
from scenariolab.llm import mock as llm_mock
from scenariolab.llm import translator as tr
from scenariolab.qmodel import config_q as cq
from scenariolab.qmodel import qtable as QT

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "scenariolab" / "evaluation" / "narrative"
HAVE_PANEL = cq.PANEL_Q_PATH.exists() and (ROOT / "data" / "results" / "model_selection.json").exists()
needs_panel = pytest.mark.skipif(not HAVE_PANEL, reason="needs the local quarterly panel (data/ is not in CI)")
BLIND_MODULES = ("prompts", "runner", "episodes", "common", "stats")
NARRATIVE = ("# 情景叙事\n\n覆盖期约 2 个季度。\n\n**第 1 个季度。** 油价上涨约三成，央行加息 50 个基点。"
             "信用利差明显走阔，失业率上升 0.3 个百分点。\n\n**第 2 个季度。** 通胀回落，央行按兵不动。\n")


@pytest.fixture(autouse=True)
def tmp_dirs(monkeypatch, tmp_path):
    """Every test writes its registry and runs under tmp_path, never into data/."""
    monkeypatch.setattr(C, "EVAL_DIR", tmp_path / "evaluation")
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path / "narrative")
    EP.clear_cache()
    K._KEY.clear()
    yield
    EP.clear_cache()
    K._KEY.clear()


def write_registry(codes: dict, narratives: dict) -> None:
    """A hand-made registry (no panel needed): {code: entry}, {(code, version, n): text}."""
    C.eval_dir().mkdir(parents=True, exist_ok=True)
    reg = {"schema": C.REGISTRY_SCHEMA, "seed": 1, "sealed_at": "test", "panel": {}, "key_sha256": "0" * 64,
           "codes": codes, "missing": []}
    C.registry_path().write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
    for (code, version, n), text in narratives.items():
        p = C.narratives_dir() / code / version / f"{n}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    EP.clear_cache()


def tiny_registry():
    rows = ["- 政策利率（联邦基金利率）：偏高（历史 60–80 分位）", "- 失业率：中等（历史 40–60 分位）"]
    codes = {"X01": {"kind": "episode", "set": "clean", "n_quarters": 2, "narratives": {"v1": ["n1", "n2", "n3"]},
                     "standin": {}, "pair": "X02", "state_rows": rows},
             "X02": {"kind": "episode", "set": "clean", "n_quarters": 2, "narratives": {"v1": ["n1"]}, "standin": {},
                     "pair": "X01", "state_rows": rows}}
    texts = {("X01", "v1", n): NARRATIVE for n in ("n1", "n2", "n3")}
    texts[("X02", "v1", "n1")] = NARRATIVE.replace("三成", "两成")
    write_registry(codes, texts)


# ================================================================ exit test: the prompt builder cannot import the key

def _package_imports(path: Path) -> set[str]:
    """Modules of the narrative package a module imports (relative `from . import x` and absolute ones)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module is None:
                out |= {a.name for a in node.names}
            elif node.level == 1 and node.module:
                out.add(node.module.split(".")[0])
            elif node.module and node.module.startswith("scenariolab.evaluation.narrative"):
                rest = node.module[len("scenariolab.evaluation.narrative"):].strip(".")
                out |= {rest.split(".")[0]} if rest else {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("scenariolab.evaluation.narrative."):
                    out.add(a.name.split(".")[3])
    return out


def test_the_prompt_builder_and_the_runner_cannot_reach_the_key_through_any_import():
    seen, todo = set(), list(BLIND_MODULES)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo += [x for x in _package_imports(PKG / f"{m}.py") if (PKG / f"{x}.py").exists()]
    assert "key" not in seen, f"a blind module reaches key.py: {sorted(seen)}"
    assert not seen & {"score", "audit", "synth", "jobs", "api", "report", "research", "engines", "sep"}, sorted(seen)
    for m in seen:
        src = (PKG / f"{m}.py").read_text(encoding="utf-8")
        assert "key.json" not in src and "load_key" not in src, m


def test_a_fresh_interpreter_importing_the_prompt_builder_and_the_runner_never_loads_the_key():
    code = ("import sys; import scenariolab.evaluation.narrative.prompts, scenariolab.evaluation.narrative.runner; "
            "print(any(m.endswith('narrative.key') or m.endswith('narrative.score') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                         env={"MSL_LLM_BACKEND": "mock", "PATH": "/usr/bin:/bin", "HOME": str(ROOT)})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "False"


def test_building_every_prompt_and_a_mock_run_never_open_the_key(monkeypatch):
    tiny_registry()
    C.eval_dir().joinpath("key.json").write_text('{"sealed": "never read by the runner"}', encoding="utf-8")
    real_open, real_read = builtins.open, Path.read_text

    def guarded_open(file, *a, **k):
        assert not str(file).endswith("key.json"), "the blind path opened key.json"
        return real_open(file, *a, **k)

    def guarded_read(self, *a, **k):
        assert self.name != "key.json", "the blind path read key.json"
        return real_read(self, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    for code in EP.codes():
        text = PR.translation_system(code)
        assert PR.scan_prompt(text) == [], PR.scan_prompt(text)
        assert "偏高（历史 60–80 分位）" in text and "模式：叙事检验（盲测）" in text
    est = RN.estimate({"episodes": ["X01", "X02"], "narratives": "v1", "arms": ["N1", "N2", "N3", "NW"],
                       "probe": True})
    rid = RN.create_run(RN.fresh_estimate(est["id"]))
    assert RN.translate_run(rid) == "translated"
    tr_ = RN.translations(rid)
    assert set(tr_) == {"X01__N1", "X01__N2", "X01__N3", "X01__NW", "X02__N1", "X02__NW"}
    assert tr_["X02__NW"]["source"]["code"] == "X01" and tr_["X02__NW"]["prompt_code"] == "X02"
    assert all(t["status"] == "ok" for t in tr_.values())
    assert all(p["answer"]["known"] is False for p in RN.probes(rid).values())


def test_a_usage_limit_pauses_the_run_and_a_resume_finishes_it(mock_llm):
    tiny_registry()
    est = RN.estimate({"episodes": ["X01"], "narratives": "v1", "arms": ["N1", "N2"], "probe": False})
    rid = RN.create_run(RN.fresh_estimate(est["id"]))
    mock_llm.push("usage_limit", match="qtable")
    assert RN.translate_run(rid) == "paused"
    run = RN.read_run(rid)
    assert run["pause"]["reason"] == "limit" and run["pause"]["reset_at"] and run["progress"]["done"] == 0
    assert RN.translate_run(rid) == "translated"                       # resumes by run id, skipping what is done
    assert set(RN.translations(rid)) == {"X01__N1", "X01__N2"} and RN.read_run(rid)["pause"] is None


# ================================================================ exit test: pytest runs the mock backend

def test_pytest_selects_the_mock_backend_and_the_runner_refuses_any_other():
    assert tr.backend_name() == "mock" and RN.backend_name() == "mock"
    tiny_registry()
    est = RN.estimate({"episodes": ["X01"], "narratives": "v1", "arms": ["N1"], "probe": False})
    assert est["backend"] == "mock" and est["calls"]["new"] == 0 and not est["real_calls"]
    s = {**RN.normalize({"episodes": ["X01"], "narratives": "v1", "arms": ["N1"], "probe": False}),
         "backend": "claude_code"}
    call = RN.plan({**s, "backend": "mock"})["calls"][0]
    with pytest.raises(RN.RunnerError, match="测试里只能用模拟后端"):
        RN.run_one("n_test", call, s)
    with pytest.raises(RN.RunnerError, match="服务器配置的后端是 mock"):
        RN.normalize({"episodes": ["X01"], "backend": "claude_code"})
    with pytest.raises(RN.RunnerError, match="不走按 token 计费"):
        RN.normalize({"episodes": ["X01"], "backend": "api"})


# ================================================================ exit test: /run refuses without a fresh estimate and confirm

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from scenariolab.evaluation.narrative import jobs as JB
    from scenariolab.server import app
    started = []
    monkeypatch.setattr(JB, "start", lambda rid, **kw: started.append(rid) or "queued")
    c = TestClient(app, raise_server_exceptions=False, base_url="http://127.0.0.1")
    c.started = started
    return c


def test_run_refuses_without_a_fresh_estimate_id_and_confirm_true(client):
    tiny_registry()
    url = "/api/evaluation/narrative"
    r = client.post(f"{url}/estimate", json={"episodes": "X01", "narratives": "v1", "arms": ["N1"], "probe": False})
    assert r.status_code == 200, r.text
    eid = r.json()["id"]
    assert r.json()["needs_confirmation"] is True
    r = client.post(f"{url}/run", json={"confirm": True})
    assert r.status_code == 400 and "estimate_id" in r.json()["detail"]
    r = client.post(f"{url}/run", json={"estimate_id": "nothex", "confirm": True})
    assert r.status_code == 400
    r = client.post(f"{url}/run", json={"estimate_id": "0123456789ab", "confirm": True})
    assert r.status_code == 400 and "没有找到估算" in r.json()["detail"]
    for body in ({"estimate_id": eid}, {"estimate_id": eid, "confirm": "true"}, {"estimate_id": eid, "confirm": 1}):
        r = client.post(f"{url}/run", json=body)
        assert r.status_code == 409 and r.json()["needs_confirm"] is True and r.json()["estimate"]["id"] == eid
    path = C.estimates_dir() / f"{eid}.json"
    est = json.loads(path.read_text(encoding="utf-8"))
    est["created"] = time.time() - C.ESTIMATE_FRESH_S - 5
    path.write_text(json.dumps(est), encoding="utf-8")
    r = client.post(f"{url}/run", json={"estimate_id": eid, "confirm": True})
    assert r.status_code == 409 and r.json()["stale"] is True and "过期" in r.json()["detail"]
    assert client.started == []
    eid = client.post(f"{url}/estimate", json={"episodes": "X01", "narratives": "v1", "arms": ["N1"],
                                               "probe": False}).json()["id"]
    (C.narratives_dir() / "X01" / "v1" / "n1.md").write_text(NARRATIVE + "（改过）", encoding="utf-8")
    r = client.post(f"{url}/run", json={"estimate_id": eid, "confirm": True})
    assert r.status_code == 409 and r.json()["stale"] is True and "变了" in r.json()["detail"]
    eid = client.post(f"{url}/estimate", json={"episodes": "X01", "narratives": "v1", "arms": ["N1"],
                                               "probe": False}).json()["id"]
    r = client.post(f"{url}/run", json={"estimate_id": eid, "confirm": True})
    assert r.status_code == 200 and client.started == [r.json()["run_id"]]
    assert RN.read_run(r.json()["run_id"])["estimate_id"] == eid


def test_narrative_posts_answer_only_this_machines_own_json_requests(client):
    from fastapi.testclient import TestClient

    from scenariolab.server import app
    tiny_registry()
    url = "/api/evaluation/narrative"
    body = {"episodes": "X01", "narratives": "v1", "arms": ["N1"], "probe": False}
    assert TestClient(app, raise_server_exceptions=False).post(f"{url}/estimate", json=body).status_code == 403
    assert client.post(f"{url}/estimate", json=body, headers={"origin": "https://evil.example"}).status_code == 403
    assert client.post(f"{url}/estimate", content=json.dumps(body),
                       headers={"content-type": "text/plain"}).status_code in (400, 403)
    eid = client.post(f"{url}/estimate", json=body).json()["id"]
    r = TestClient(app, raise_server_exceptions=False).post(f"{url}/run", json={"estimate_id": eid, "confirm": True})
    assert r.status_code == 403 and client.started == []


def test_the_api_lists_the_research_run_and_reports_an_unsealed_registry(client):
    r = client.get("/api/evaluation/narrative")
    assert r.status_code == 200
    j = r.json()
    assert j["registry"]["sealed"] is False and "seal" in j["registry"]["hint"]
    assert j["runs"][0]["id"] == RS.RESEARCH_ID and j["sep"]["available"] in (True, False)
    assert client.post("/api/evaluation/narrative/estimate", json={}).status_code == 503
    assert client.get("/api/evaluation/narrative/runs/n_nope").status_code == 404
    assert client.get("/api/evaluation/narrative/runs/..%2Fx").status_code in (400, 404)


# ================================================================ exit test: the research round's statistics, recomputed

def test_research_archive_reproduces_the_scorers_pooled_numbers_and_the_critics_subsets():
    chk = RS.checks(RS.recompute(intervals=False))
    assert chk["ok"], chk["failed"][:5]
    scorer = [i for i in chk["items"] if i["what"] == "scorer.mae" and i["key"][2] == "equity9"]
    assert len(scorer) == 91 and max(i["dev"] for i in scorer) <= 0.002
    assert max(i["dev"] for i in chk["items"] if i["what"].startswith("scorer")) < 1e-6
    critic = [i for i in chk["items"] if i["what"] == "critic.point"]
    assert len(critic) == 76 and all(i["ok"] for i in critic)
    P = RS.scorer_pooled().set_index(["variant", "horizon", "assets"])
    assert abs(P.at[("E-n2", "end", "equity9"), "mae"] - 0.246) < 0.002          # the scoreboard's E-n2 end
    assert abs(P.at[("Q-n1", "end", "equity9"), "mae"] - 0.306) < 0.002
    R = RS.load_rows()
    sub = {"stat": "mse_skill", "horizon": "end", "a": "E-oracle", "b": "E-uncond"}
    assert abs(RS.point(R, {**sub, "excluded": ["E06", "E03"]}) - (-0.27)) <= 0.005           # critic 4: -0.27
    assert abs(RS.point(R, {**sub, "excluded": ["E06", "E03", "E05", "E12"]}) - 0.27) <= 0.005  # +0.27 (n 8)
    ch = RS.critic_c17(R)
    assert (ch["tests"], ch["favourable"], ch["unfavourable"]) == (30, 1, 3)                   # 1 of 30, 3 against


def test_research_critic_ports_reproduce_the_printed_intervals():
    chk = RS.checks()
    assert chk["ok"], chk["failed"][:5]
    assert chk["max_interval_dev"] <= 0.01 + 1e-9          # the critic printed its intervals to 2-3 decimals


def test_research_report_carries_its_labels_corrections_and_checks():
    from scenariolab.evaluation.narrative import report as RP
    rep = RP.research_report()
    assert rep["schema"] == "narrative-report/1" and rep["kind"] == "research"
    assert rep["label"] == "研究轮（代译，不盲，2026-11-01）"
    assert "识别出年代的回放（不是盲测）" in rep["labels"] and "研究轮代译（无引文）" in rep["labels"]
    assert rep["checks"]["ok"] and rep["blindness"]["recognition"]["rate"] == 1.0
    assert any(c["id"] == "E03" for c in rep["corrections"])
    end = {r["variant"]: r for r in rep["scoreboard"]["rows"] if r["horizon"] == "end"}
    assert abs(end["E0:N2"]["mae"] - 0.2462) < 0.0005 and end["E0:N2"]["n_episodes"] == 11
    assert {r["n_episodes"] for r in rep["scoreboard_clean"]["rows"]} <= {6, 10}
    assert rep["chance"]["tests"] == 30 and "不是盲测" in rep["headline"]


# ================================================================ the stand-in import

STANDIN = {"code": "E99", "narrative": "n1", "timing": {"n_quarters": 2}, "constraints": [{"field": "d_oil"}],
           "c_q": {"anticip": [{"field": "d_ff"}]}, "events": [{}],
           "rows": [{"q": "q1", "x": {"d_ff": 0.5, "d_oil": 25.5, "x_oil_cause": 0.0, "d_gz": None, "d_sent": -3.0},
                     "sign": {"d_capex": "+"}, "src": {"d_ff": "interp", "d_oil": "stated", "x_oil_cause": "stated",
                                                       "d_sent": "inferred"}},
                    {"q": "q2", "x": {"d_ff": 0.0, "x_oil_cause": 1.0}, "sign": {}, "src": {"d_ff": "stated"}}]}


def test_convert_standin_fixes_the_oil_cause_and_keeps_only_numeric_values():
    out = RS.convert_standin(STANDIN)
    qt, rec = out["qtable"], out["conversion"]
    r1, r2 = qt["rows"]
    assert r1["x"]["x_oil_cause"]["value"] == "supply" and r2["x"]["x_oil_cause"]["value"] == "demand"
    assert r1["x"]["d_ff"] == {"value": 0.5, "src": "derived", "quote": ""}          # interp -> derived
    assert r1["x"]["d_oil"]["src"] == "stated" and "d_gz" not in r1["x"] and "d_capex" not in r1["x"]
    assert rec["inferred"] == 1 and rec["dropped"]["sign_only"] == 1 and rec["dropped"]["constraints"] == 1
    assert all(r["fc"]["stance"] == "未说明" for r in qt["rows"]) and RS.STANDIN_NOTE in qt["notes"]
    rep = QT.validate(qt, "任意的叙事", mode="test", source="legacy", stats={})
    assert rep["ok"] and rep["readable"] and any(w["code"] == "W-QUOTE" for w in rep["warnings"])


def test_the_committed_archive_holds_12_episodes_and_36_legacy_tables():
    arch = RS.load_archive()
    assert sorted(arch["episodes"]) == sorted(RS.CODES)
    n = 0
    for code, ep in arch["episodes"].items():
        assert set(ep["narratives"]) == {"n1", "n2", "n3"} and set(ep["standin"]) == {"n1", "n2", "n3"}
        for nn, rec in ep["standin"].items():
            rep = QT.validate(rec["qtable"], ep["narratives"][nn], mode="test", source="legacy", stats={})
            assert rep["readable"] and not rep["errors"], (code, nn, rep["errors"][:2])
            n += 1
    assert n == 36
    fixes = sum(sum(ep["standin"][nn]["conversion"]["oil_cause"].values()) for ep in arch["episodes"].values()
                for nn in ("n1", "n2", "n3"))
    assert fixes == 65 + 49 + 23                                   # every 0 / 1 / 0.5 code became the enum


@pytest.mark.skipif(not os.environ.get("MSL_RESEARCH_DIR"), reason="set MSL_RESEARCH_DIR to the sim_redesign folder")
def test_import_standin_check_reproduces_the_committed_files_from_the_research_folder():
    src = Path(os.environ["MSL_RESEARCH_DIR"])
    if not (src / "scorer" / "out" / "score_rows.csv").exists():
        pytest.skip("MSL_RESEARCH_DIR is not the research round's sim_redesign folder")
    assert RS.import_standin(src, check=True)["ok"]


# ================================================================ registry pieces (no data needed)

def test_codes_are_a_fixed_seeded_permutation_and_pairs_follow_the_rule():
    a, b = K.assign_codes(), K.assign_codes()
    assert a == b and sorted(a.values()) == [f"X{i:02d}" for i in range(1, 19)] and set(a) == set(K.EPISODES)
    assert a != {s: f"X{i + 1:02d}" for i, s in enumerate(sorted(K.EPISODES))}
    eps = {"X01": {"set": "clean", "n_quarters": 6, "origin": "1990Q1"},
           "X02": {"set": "clean", "n_quarters": 6, "origin": "1992Q1"},       # too close to X01
           "X03": {"set": "clean", "n_quarters": 8, "origin": "2000Q1"},
           "X04": {"set": "clean", "n_quarters": 4, "origin": "2010Q1"},
           "X05": {"set": "short", "n_quarters": 6, "origin": "1975Q1"}}
    p = K.pairs(eps)
    assert p["X01"] == "X03" and p["X02"] == "X03" and p["X03"] == "X01" and p["X04"] == "X01" and "X05" not in p


def test_window_stats_drawdown_and_training_quarters():
    idx = pd.period_range("1973Q1", periods=8, freq="Q")
    Q = pd.DataFrame({"y_us_equity": [0.1, 0.1, -0.3, 0.1, 0.05, 0.0, 0.0, 0.0], "y_nasdaq": 0.0, "d_gz": 0.1},
                     index=idx)
    st = K.window_stats(Q, "1973Q3", "1973Q4")
    assert st["n_quarters"] == 2 and st["train_quarters"] == 2 and st["origin"] == "1973Q3"
    assert st["train_end"] == "1973Q2" and abs(st["useq"] + 0.2) < 1e-12 and abs(st["dd"] + 0.3) < 1e-12


def test_probe_prompt_mock_answer_and_reader():
    sysm, user = PR.probe_system(), PR.probe_user(NARRATIVE)
    schema = llm_mock.schema_from_system(sysm)
    assert llm_mock.call_kind(schema, user) == "nprobe"
    text, _ = llm_mock.run(sysm, user, "m")
    ans = PR.read_probe(text)
    assert ans["valid"] and ans["known"] is False and ans["period"] == "unknown"
    assert PR.read_probe("not json")["known"] is False
    got = PR.read_probe(json.dumps({"schema_version": C.PROBE_SCHEMA_VERSION, "known": True, "period": "x",
                                    "start_year": 2007, "end_year": 2009, "confidence": 0.9, "clues": ["a"]}))
    from scenariolab.evaluation.narrative import audit as AU
    assert AU.recognised(got, "2007Q3", "2009Q2") and not AU.recognised(got, "1990Q3", "1991Q4")
    assert AU.recognised({**got, "start_year": 2010, "end_year": None}, "2007Q3", "2009Q2")


def test_plan_puts_n3_first_lists_missing_narratives_and_replays_stand_ins(monkeypatch):
    tiny_registry()
    s = RN.normalize({"episodes": ["X01", "X02"], "narratives": "v1", "arms": "N1,N2,N3,wrong", "probe": True})
    p = RN.plan(s)
    x01 = [c for c in p["calls"] if c["code"] == "X01"]
    assert x01[0]["arm"] == "N3" and [c["kind"] for c in x01][-2:] == ["probe", "probe"]
    assert {c["arm"] for c in x01 if c["kind"] == "translate"} == {"N1", "N2", "N3", "NW"}
    nw = next(c for c in x01 if c["arm"] == "NW")
    assert nw["source"]["code"] == "X02" and nw["prompt_code"] == "X01"
    assert {(m["code"], m["arm"]) for m in p["missing"]} == {("X02", "N2"), ("X02", "N3")}
    v2 = RN.plan(RN.normalize({"episodes": ["X01"], "narratives": "v2", "arms": "N1", "probe": False}))
    assert v2["calls"] == [] and v2["missing"][0]["text"] == C.LABELS["no_v2"]


def test_stats_cell_crps_and_paired_are_the_research_definitions():
    rng = np.random.default_rng(0)
    d = rng.normal(0.0, 1.0, 4000)
    c = ST.cell(d, 0.0)
    assert abs(c["q50"]) < 0.05 and c["cov80"] == 1.0 and abs(c["pit"] - 0.5) < 0.03
    assert abs(c["crps"] - (np.sqrt(2) - 1) / np.sqrt(np.pi)) < 0.02          # CRPS of N(0,1) at its median
    deg = ST.cell(np.zeros(10), 0.1)
    assert deg["degenerate"] and np.isnan(deg["cov80"]) and abs(deg["crps"] - 0.1) < 1e-12
    rows = []
    for e, (ea, eb) in {"A": (0.1, 0.2), "B": (0.3, 0.3), "C": (0.0, 0.4), "D": (0.2, 0.1)}.items():
        for v, err in (("Q:N2", ea), ("Q:U", eb)):
            rows.append({"code": e, "variant": v, "cma": "historical", "asset": "us_equity", "horizon": "end",
                         "err": err, "abs_err": abs(err)})
    R = pd.DataFrame(rows)
    p = ST.paired(R, ("Q:N2", "historical"), ("Q:U", "historical"), "end", "us_equity")
    assert p["n_episodes"] == 4 and abs(p["diff"] - (-0.1 + 0.0 - 0.4 + 0.1) / 4) < 1e-12
    assert p["lo90"] <= p["diff"] <= p["hi90"] and p["mde"] > 0
    assert p == ST.paired(R, ("Q:N2", "historical"), ("Q:U", "historical"), "end", "us_equity")   # seeded by name
    assert abs(p["mse_skill"] - (1 - (0.01 + 0.09 + 0 + 0.04) / (0.04 + 0.09 + 0.16 + 0.01))) < 1e-6   # 6 decimals


def test_synthetic_writer_reference_translation_passes_the_translators_own_rules():
    from scenariolab.evaluation.narrative import synth as SY
    H = 3
    path = {"d_ff": [0.5, 0.0, -0.25], "d_unrate": [0.2, 0.0, -0.1], "d_ip": [-1.2, 0.3, 0.0],
            "d_oil": [20.0, -5.0, 0.2], "cpi_yoy": [3.1, 3.4, 2.9], "d_gz": [0.4, -0.2, 0.0],
            "d_spread": [0.3, 0.0, -0.1], "d_y10": [-0.2, 0.1, 0.0], "d_usd": [2.0, -1.0, 0.0]}
    out = SY.write(path, {"s_ff": 2.0, "s_unrate": 5.0, "s_cpi_yoy": 3.0}, n_quarters=H)
    for n in ("n1", "n2"):
        assert "覆盖期约 3 个季度" in out[n] and not PR.scan_narrative(out[n], "synth", n)["years"]
        rep = QT.validate(out["reference"][n], out[n], mode="test", source="translator", stats={})
        assert rep["ok"], (n, rep["refused"][:3])
    x2 = out["reference"]["n2"]["rows"][0]["x"]
    assert x2["d_gz"]["value"] == 0.4 and abs(x2["d_oil"]["value"] - 100 * np.log(1.22)) < 1e-9
    assert out["reference"]["n1"]["c_q"]["constraints"][1]["value"] == 3.4


def test_sep_conversion_by_code_and_the_gated_download(tmp_path):
    from scenariolab.evaluation.narrative import sep as SEP
    with pytest.raises(SEP.SepError, match="需要你的同意"):
        SEP.fetch(approved=False)
    plan = SEP.plan()
    assert plan["approved"] is False and plan["host"] == "api.stlouisfed.org"
    assert {r["series_id"] for r in plan["requests"]} >= {"UNRATEMD", "GDPC1MD", "PCECTPIMD", "JCXFEMD", "FEDTARMD"}
    assert not SEP.available(tmp_path)

    def obs(vals):     # {(realtime_start, year): value}
        return pd.DataFrame([{"date": pd.Timestamp(f"{y}-01-01"), "value": v, "realtime_start": pd.Timestamp(rs),
                              "realtime_end": pd.Timestamp("2262-04-11")} for (rs, y), v in vals.items()])
    data = {"UNRATEMD": obs({("2012-12-12", 2013): 7.5}), "GDPC1MD": obs({("2012-12-12", 2013): 2.5}),
            "PCECTPIMD": obs({("2012-12-12", 2013): 1.6}), "JCXFEMD": obs({("2012-12-12", 2013): 1.7}),
            "FEDTARMD": obs({("2012-12-12", 2013): 0.25})}
    idx = pd.period_range("2010Q1", "2014Q1", freq="Q")
    Q = pd.DataFrame({"s_unrate": 7.8, "s_cpi_yoy": 1.9, "s_core_yoy": 1.9, "s_ff": 0.15}, index=idx)
    wins = SEP.windows(data, "2013Q4")
    assert wins == [{"code": "SEP2013", "year": 2013, "release": "2012-12-12", "origin": "2013Q1",
                     "train_end": "2012Q4", "n_quarters": 4}]
    qt, rec = SEP.to_qtable(wins[0], data, Q, fc0=True)
    assert rec["used"]["unrate"]["value"] == 7.5 and "pce" not in rec["used"]      # no PCE index: no CPI gap
    rep = QT.validate(qt, None, mode="test", source="user", stats={})
    assert rep["ok"], rep["refused"][:3]
    cons = qt["c_q"]["constraints"]
    assert [c["value"] for c in cons if c["field"] == "s_unrate"][-1] == 7.5
    assert all(r["fc"]["stance"] == "中性" for r in qt["rows"]) and qt["rows"][0]["x"]["d_rgdp"]["value"] == 2.5


# ================================================================ data-dependent: sealing and the mock run end to end

@needs_panel
def test_seal_reproduces_the_spec_episode_table():
    out = K.seal(synth=False)
    assert out["sealed"] and out["codes"] == 18 and out["checks_failed"] == [], out["checks_failed"]
    key = K.load_key()
    clean = [e for e in key["episodes"].values() if e["set"] == "clean"]
    assert len(clean) == 14 and sum(e["n_quarters"] for e in clean) == 119
    assert sum(e["bust"] for e in clean) == 5                                       # §8.1: 5 busts in the clean set
    reg = EP.load_registry()
    assert all(PR.scan_prompt("\n".join(e["state_rows"])) == [] for e in reg["codes"].values())
    assert all(c is None or reg["codes"][c]["set"] == "clean" for c in (e["pair"] for e in reg["codes"].values()))
    with_v1 = [c for c, e in reg["codes"].items() if "v1" in e["narratives"]]
    assert len(with_v1) == 12 and len(reg["missing"]) == 18 + 6
    assert K.seal()["sealed"] is False                                              # sealed once
    C.eval_dir().joinpath("key.json").write_text(C.eval_dir().joinpath("key.json").read_text() + " ")
    K._KEY.clear()
    with pytest.raises(K.KeyError_, match="哈希不一致"):
        K.load_key()


@needs_panel
def test_mock_run_passes_end_to_end():
    from scenariolab.evaluation.narrative import jobs as JB
    from scenariolab.evaluation.narrative import report as RP
    K.seal(synth=True, n_boot=0)
    codes = K.resolve("E11") + ["S01"]
    est = RN.estimate({"episodes": codes, "narratives": "v1", "arms": ["N1", "N2", "N3", "NW", "NS"], "probe": True})
    assert est["backend"] == "mock" and est["calls"]["standin"] == 4 and est["calls"]["new"] == 0
    rid = RN.create_run(RN.fresh_estimate(est["id"]))
    status = JB.start(rid, background=False, score_kwargs={"n_paths": 200, "n_boot": 0, "cmas": ("historical",),
                                                           "e_paths": 60, "e_arms": ("U", "OB", "N2")})
    run = RN.read_run(rid)
    assert status == "done", run.get("error")
    tr_ = RN.translations(rid)
    assert {t["status"] for k, t in tr_.items() if k.startswith(codes[0])} == {"standin"}
    assert tr_["S01__NS"]["status"] == "ok" and tr_["S01__NS"]["validation"]["ok"]     # the synthetic reference replay
    rep = json.loads((RN.run_dir(rid) / "report.json").read_text(encoding="utf-8"))
    assert rep == RP.load(rid) and rep["schema"] == "narrative-report/1" and rep["kind"] == "mock"
    assert rep["labels"][0] == C.LABELS["mock"] and C.LABELS["legacy"] in rep["labels"]
    assert [c["id"] for c in rep["claims"]] == ["C1", "C2", "C3", "C4"]
    assert all(not c.get("passed") for c in rep["claims"] if not c.get("always"))
    variants = {r["variant"] for r in rep["scoreboard"]["rows"]}
    assert {"Q:U", "Q:OA", "Q:OAp", "Q:OB", "Q:OF", "Q:N1", "Q:N2", "Q:N3", "Q:NW"} <= variants
    assert "Q:V" not in variants and "Q:V" in {r["variant"] for r in rep["groups"]["us_equity"]}
    ep = next(e for e in rep["episodes"] if e["code"] == codes[0])
    assert ep["window"]["spec_code"] == "E11" and ep["window"]["start"] == "1990Q3"
    assert len(ep["series"]["Q:N2"]["us_equity"]["p50"]) == 6
    syn = next(e for e in rep["episodes"] if e["code"] == "S01")
    assert "Q:NS" in syn["series"] and syn["window"]["set"] == "synth"
    bl = rep["blindness"]
    assert bl["prompt_clean"] and bl["recognition"]["n"] == 2 and bl["recognition"]["rate"] == 0.0
    assert bl["contamination"]["overall"]["n"] > 0 and bl["translations"]["status"]["standin"] == 4
    assert bl["delivery"]["overall"]["n"] > 0 and bl["delivery"]["by_arm"]["NS"]["median_gap_sd"] < 0.5
    R = pd.read_csv(RN.run_dir(rid) / "score_rows.csv")
    real = R[(R["variant"] == "Q:U") & (R["asset"] == "us_equity") & (R["horizon"] == "end") & (R["code"] == codes[0])]
    assert abs(float(real["y"].iloc[0]) - 0.168) < 0.0005                            # the window's realized useq
    v = R[(R["variant"] == "Q:V") & (R["code"] == codes[0]) & (R["horizon"] == "end")]
    of = R[(R["variant"] == "Q:OF") & (R["code"] == codes[0]) & (R["horizon"] == "end") & (R["asset"] == "us_equity")]
    assert float(v["width"].iloc[0]) < 0.5 * float(of["width"].iloc[0])             # the multiple is given: a tight band
    assert run["settings"]["engines"] == ["Q", "E"]                                 # both engines by default
    e_vars = {r["variant"] for r in rep["scoreboard"]["rows"] if r["engine"] == "E"}
    assert e_vars == {"E:U", "E:OB", "E:N2"}                                        # synthetic: Q only
    assert rep["ts_inputs"]["oracle"]["OB|end"]["diff"] is not None and rep["ts_inputs"]["pit_mad"]["E"] is not None
    assert "E:N2" in ep["series"] and ep["engine_E"]["arms"]["N2"]["phases"] >= 1
    m = json.loads((RN.run_dir(rid) / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((RN.run_dir(rid) / "plan.json").read_text(encoding="utf-8"))
    assert set(m["calls"]) == {c["id"] for c in plan["calls"]}
    assert not (RN.run_dir(rid) / "sent").exists()                                  # the mock sends nothing


@needs_panel
def test_engine_arm_runs_the_monthly_engine_on_the_identical_table():
    from scenariolab.evaluation.narrative import engines as EN
    from scenariolab.qmodel import panel as qpanel
    Q = qpanel.load_panel()
    arch = RS.load_archive()
    qt = arch["episodes"]["E11"]["standin"]["n2"]["qtable"]
    clean = QT.validate(qt, None, mode="test", source="legacy")["qtable"]
    st = EN.setup("1990Q2")
    spec, rep = EN.phase_spec(clean, Q, "1990Q2", 6)
    assert sum(p["months"] for p in spec["phases"]) == 18 and rep["phases"]
    paths, info = EN.run(st, spec, 6, 60, 1)
    assert paths.shape == (60, 6, 13) and np.isfinite(paths).all()
    with pytest.raises(EN.EngineSkip):
        EN.setup("1973Q3")


@needs_panel
@pytest.mark.parametrize("train_end, origin", [("2007Q2", "2007Q3"), ("1990Q2", "1990Q3")])
def test_check_context_nothing_after_t0_reaches_an_episode(train_end, origin):
    """§8.4 check_context (truncation test): the scorer fits each episode's stack from the full panels; fitting it from
    the panels cut at the origin (start-of-quarter columns only on the origin row) gives identical no-narrative paths
    for a fixed seed, factors and assets."""
    from scenariolab.evaluation.narrative import score as SC
    res = SC.check_context(train_end, origin, H=4, n_paths=120, seed=3)
    assert res["ok"], res["max_abs_diff"]
    assert set(res["max_abs_diff"]) >= {"price", "d_ff", "d_unrate", "d_gz"}


# ================================================================ review fixes: resume, every byte sent, identical tables

def test_resume_needs_an_estimate_of_the_runs_own_remaining_calls(client):
    """/runs/{rid}/resume refuses an estimate made for another plan (a 1-call estimate cannot confirm a bigger run);
    the estimate of the run's own remaining calls ({run_id}) is accepted, and /run refuses a resume estimate."""
    tiny_registry()
    url = "/api/evaluation/narrative"
    big = client.post(f"{url}/estimate", json={"episodes": "X01", "narratives": "v1", "arms": ["N1", "N2", "N3"],
                                               "probe": True}).json()
    rid = RN.create_run(RN.fresh_estimate(big["id"]))
    RN.write_run(rid, status="paused")
    small = client.post(f"{url}/estimate", json={"episodes": "X02", "narratives": "v1", "arms": ["N1"],
                                                 "probe": False}).json()
    assert small["calls"]["total"] < big["calls"]["total"] and small["plan_digest"] != big["plan_digest"]
    r = client.post(f"{url}/runs/{rid}/resume", json={"estimate_id": small["id"], "confirm": True})
    assert r.status_code == 409 and r.json()["stale"] is True and rid in r.json()["detail"]
    r = client.post(f"{url}/runs/{rid}/resume", json={"estimate_id": big["id"], "confirm": True})
    assert r.status_code == 409                       # the run's original estimate is not a resume estimate either
    assert client.started == [] and RN.read_run(rid).get("resumed_with") is None
    own = client.post(f"{url}/estimate", json={"run_id": rid}).json()
    assert own["run_id"] == rid and own["remaining"] == big["calls"]["total"] and own["done"] == 0
    r = client.post(f"{url}/run", json={"estimate_id": own["id"], "confirm": True})
    assert r.status_code == 409 and "继续运行" in r.json()["detail"]
    r = client.post(f"{url}/runs/{rid}/resume", json={"estimate_id": own["id"], "confirm": True})
    assert r.status_code == 200 and client.started == [rid] and RN.read_run(rid)["resumed_with"] == own["id"]


def test_the_cli_prices_a_resume_from_the_runs_remaining_calls(monkeypatch, capsys):
    """The CLI never trusts the supplied estimate's real_calls: resume prices the run's own remaining calls now and
    asks (or needs the flag) whenever one of them is not cached; an estimate of another plan is refused."""
    from scenariolab.evaluation.narrative import __main__ as M
    from scenariolab.evaluation.narrative import jobs as JB
    tiny_registry()
    est = RN.estimate({"episodes": ["X01"], "narratives": "v1", "arms": ["N1"], "probe": False})
    rid = RN.create_run(est)
    RN.write_run(rid, status="paused")
    other = RN.estimate({"episodes": ["X02"], "narratives": "v1", "arms": ["N1"], "probe": False})
    started = []
    monkeypatch.setattr(JB, "start", lambda r, **kw: started.append(r) or "done")
    assert M.main(["resume", rid, "--estimate", other["id"]]) == 2 and started == []
    assert "不是运行" in capsys.readouterr().err
    # a real backend with an uncached remaining call: no TTY, no flag -> refused, even with an all-cached estimate
    live = {**RN.estimate_resume(rid), "backend": "claude_code", "real_calls": True}
    monkeypatch.setattr(RN, "estimate_resume", lambda r, **kw: live)
    monkeypatch.setattr(M.sys.stdin, "isatty", lambda: False, raising=False)
    assert M.main(["resume", rid]) == 1 and started == []
    assert M.main(["resume", rid, "--i-understand-this-uses-my-plan"]) == 0 and started == [rid]


def test_a_format_repair_is_recorded_under_sent_and_in_the_manifest(monkeypatch):
    """Every byte sent to a real backend is kept: the format repair's user message too (fake backend: no CLI)."""
    from scenariolab.llm import backend as LB
    from scenariolab.llm import claude_code
    from scenariolab.llm import qtranslator as QX
    tiny_registry()
    sent = []

    def fake_run(system, user, model, timeout, effort=None, cancel=None):
        sent.append(user)
        return ("{ not json", {"usage": {}}) if len(sent) == 1 else (json.dumps(QT.minimal(2)), {"usage": {}})

    class Slot:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(claude_code, "run", fake_run)
    monkeypatch.setattr(QX.tr, "cache_store", lambda *a, **k: None)
    monkeypatch.setattr(QX.tr, "cache_load", lambda *a, **k: None)
    monkeypatch.setattr(LB, "cli_slot", lambda **k: Slot())
    monkeypatch.setattr(QX.tr, "status", lambda: {"available": True})
    monkeypatch.setattr(QX.claude_code, "status", lambda m: {"available": True})
    monkeypatch.setattr(QX.tr, "backend_name", lambda: "claude_code")
    est = RN.estimate({"episodes": ["X01"], "narratives": "v1", "arms": ["N1"], "probe": False}, save=False)
    rid = RN.create_run({**est, "id": "0" * 12})
    call = RN._call("translate", "X01", "N1", ("X01", "v1", "n1"), "X01")
    call.update(standin=False, reference=False)
    rec = RN._translate(rid, call, {"backend": "claude_code", "model": "m", "refresh": False}, None)
    assert rec["status"] == "repaired" and len(sent) == 2
    files = {p.stem for p in (RN.run_dir(rid) / "sent").glob("*.txt")}
    assert rec["repair_user_sha256"] == PR.sha256(sent[1]) and rec["repair_user_sha256"] in files
    assert PR.sha256(sent[0]) in files and rec["repair"].get("user") is None     # the text lives under sent/ only


def test_the_engine_arms_convert_exactly_the_table_the_quarterly_arm_ran():
    """§4.6 #4 / §7.4: E converts the clean table Q ran at the episode's train end (the same R-RANGE refusals and
    shock range), without the anchors Q records but does not apply; nothing Q refused reaches E's views."""
    from scenariolab.evaluation.narrative import score as SC
    stats = {"d_unrate": {"sd": 0.6, "lo": -1.0, "hi": 1.5}, "d_ff": {"sd": 1.2, "lo": -7.7, "hi": 8.0},
             "u_ff": {"sd": 1.0, "lo": None, "hi": None}}
    nar = "失业率下降 4.6 个百分点，央行降息 50 个基点，央行紧急降息 9 个百分点。中性利率 2.5%。"
    qt = {"schema": "qtable/1", "n_quarters": 2, "rows": [
        {"q": 1, "x": {"d_unrate": {"value": -4.6, "src": "stated", "quote": "失业率下降 4.6 个百分点"},
                       "d_ff": {"value": -0.5, "src": "stated", "quote": "央行降息 50 个基点"}},
         "u": {"u_ff": {"value": -9.0, "src": "stated", "quote": "央行紧急降息 9 个百分点"}}, "fc": {"stance": "未说明"}},
        {"q": 2, "x": {}, "fc": {"stance": "未说明"}}],
        "c_q": {"anchors": {"r_star": {"value": 2.5, "quote": "中性利率 2.5%"}}}}
    rep = QT.validate(qt, nar, mode="test", source="translator", stats=stats)
    codes = sorted(r["code"] for r in rep["refused"])
    assert codes == ["R-RANGE", "R-RANGE"]                      # d_unrate beyond 1.5x the range; u_ff beyond ±4 sd
    clean = rep["qtable"]
    table, notes = SC.e_table(clean)
    assert "anchors" not in table["c_q"] and notes == [SC.ANCHORS_NOTE]
    spec, srep = QT.to_phase_spec(table, stats=stats, validated=True, mode="test", horizon_months=6)
    views = {v["var"] for ph in spec["phases"] for v in ph["views"]}
    assert views == {"f_ff"} and not any(ph.get("overrides") for ph in spec["phases"])
    assert not any(v.get("mean") == -9.0 / 3.0 for ph in spec["phases"] for v in ph["views"])


def test_the_blind_prompt_has_no_episode_specific_content():
    """§7.2 / §8.4: the test-mode prompt's decomposition example is synthetic, its templates drop the single-instance
    one and every duration label and era word; the scan covers event words."""
    from scenariolab.llm import qtranslator as QX
    tiny_registry()
    text = PR.translation_system("X01")
    assert QX.DECOMPOSE_EXAMPLE["test"] in text and QX.DECOMPOSE_EXAMPLE["production"] not in text
    for w in PR.EVENT_WORDS + PR.ERA_NAMES:
        assert w not in text, w
    assert "G01 " not in text and "（1–2 季）" not in text
    assert PR.scan_prompt(text) == []
    assert PR.scan_prompt(text + "疫情封锁") == ["事件词「疫情」", "事件词「封锁」"]
    assert QX.DECOMPOSE_EXAMPLE["production"] in QX.build_system_prompt("production")
