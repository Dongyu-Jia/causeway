"""API smoke tests against built results (skipped until `python -m scenariolab.build` has run)."""
import pytest
from fastapi.testclient import TestClient

from scenariolab import config, server
from scenariolab.server import app

client = TestClient(app, raise_server_exceptions=False)
BUILT = (config.RESULTS_DIR / "meta.json").exists()
needs_build = pytest.mark.skipif(not BUILT, reason="results not built")


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "宏观情景实验室" in r.text


def test_unknown_result_404():
    assert client.get("/api/nope").status_code == 404
    assert client.get("/api/scenarios/BAD-ID!").status_code in (400, 404)


def test_written_text_has_no_negative_zero():
    from scenariolab import jsonio
    out = jsonio.clean({"a": "合计只解释 -0.0%，均值 −0.00，λ=-0", "b": ["1990-08 至 2020-03", "3.6e-08", "-0.05", "-0.50%"]})
    assert out["a"] == "合计只解释 0.0%，均值 0.00，λ=0"
    assert out["b"] == ["1990-08 至 2020-03", "3.6e-08", "-0.05", "-0.50%"]


def test_ui_modules_are_revalidated():
    """Browsers must not keep stale tab modules or styles after an edit."""
    for path in ("/js/app.js", "/js/charts.js", "/js/tabs/overview.js", "/css/app.css", "/"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers.get("cache-control") == "no-cache", path


def test_scenario_errors_are_explained_in_chinese():
    r = client.get("/api/scenarios/Bad-Id")
    assert r.status_code == 400 and "不合法" in r.json()["detail"]
    r = client.get("/api/scenarios/no_such_scenario_xyz")
    assert r.status_code == 404 and "没有找到情景" in r.json()["detail"]
    r = client.delete("/api/scenarios/custom/no_such_scenario_xyz")
    assert r.status_code == 404 and "没有找到" in r.json()["detail"]


def test_bad_requests_are_explained_in_chinese():
    """Malformed bodies, unknown or missing parameters, unknown paths: one Chinese line, no English stack."""
    r = client.post("/api/portfolio/evaluate", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and "JSON" in r.json()["detail"] and "请求体" in r.json()["detail"]
    r = client.post("/api/portfolio/evaluate", json=[1, 2])
    assert r.status_code == 400 and "JSON 对象" in r.json()["detail"]
    r = client.post("/api/portfolio/evaluate", json={"weights": {"us_equity": 1}, "bogus": 1})
    assert r.status_code == 400 and "不认识的参数：bogus" in r.json()["detail"] and "weights" in r.json()["detail"]
    r = client.post("/api/withdrawal/evaluate", json={"rate": 0.04})
    assert r.status_code == 400 and r.json()["detail"] == "缺少参数：weights"
    r = client.get("/js/tabs/no_such_tab.js")
    assert r.status_code == 404 and "找不到这个地址" in r.json()["detail"]
    r = client.post("/api/meta")
    assert r.status_code == 405 and "不支持" in r.json()["detail"]
    # our own Chinese details pass through unchanged
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["detail"] == "没有名为「nope」的结果"


def test_missing_results_name_the_command_that_builds_them(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RESULTS", tmp_path)
    hints = {"ablation": "--only ablation", "valuation": "--only valuation",
             "forecast": "--only forecast", "eras": "python -m scenariolab.build"}
    for name, hint in hints.items():
        r = client.get(f"/api/{name}")
        assert r.status_code == 404, name
        assert hint in r.json()["detail"] and f"{name}.json" in r.json()["detail"], r.json()
    # a built-in scenario whose file is missing points at the build, an unknown one does not
    r = client.get("/api/scenarios/baseline")
    assert r.status_code == 404 and "--only scenarios" in r.json()["detail"]


def test_custom_ids_cannot_shadow_builtins(monkeypatch):
    monkeypatch.setattr(server, "builtin_ids", lambda: {"baseline", "custom_baseline"})
    assert server._custom_id({"id": "baseline"})["id"] == "custom2_baseline"
    assert server._custom_id({"id": "Oil War!"})["id"] == "oil_war"
    assert server._custom_id({"id": "my_own"}) == {"id": "my_own"}
    monkeypatch.setattr(server, "builtin_ids", lambda: {"oil_war"})
    assert server._custom_id({"name": "Oil War"})["id"] == "custom_oil_war"
    # nothing usable for an id (a Chinese name, no id): a unique id, never the shared fallback "scenario"
    a, b = server._custom_id({"name": "我的情景"})["id"], server._custom_id({"name": "我的情景"})["id"]
    assert a.startswith("custom_") and b.startswith("custom_") and a != b
    assert server._custom_id({"id": "scenario"})["id"].startswith("custom_")


@needs_build
@pytest.mark.parametrize("name", ["meta", "eras", "causality", "backtest", "diagnostics", "scenarios"])
def test_results_served(name):
    r = client.get(f"/api/{name}")
    assert r.status_code == 200, r.text[:200]
    assert isinstance(r.json(), dict)


@needs_build
def test_scenario_detail_and_portfolio_roundtrip():
    index = client.get("/api/scenarios").json()
    sid = index["scenarios"][0]["id"]
    detail = client.get(f"/api/scenarios/{sid}").json()
    assert len(detail["labels"]) == config.HORIZON_MONTHS + 1
    weights = {"us_equity": 0.6, "treasury_10y": 0.4}
    p = client.post("/api/portfolio/evaluate", json={"weights": weights})
    assert p.status_code == 200, p.text[:300]
    assert p.json()["per_scenario"]
    w = client.post("/api/withdrawal/evaluate", json={"weights": weights, "rate": 0.04})
    assert w.status_code == 200, w.text[:300]
    body = w.json()
    assert set(body["strategies"]) >= {"pro_rata", "div_then_pro_rata", "div_then_winners"}


@needs_build
def test_custom_scenario_run_and_delete():
    spec = {"id": "test_custom_api", "name": "测试情景", "phases": [
        {"name": "高通胀", "months": 24, "views": [{"var": "s_cpi_yoy", "mean": 6.0}]}]}
    r = client.post("/api/scenarios/custom/run", json={"spec": spec, "n_paths": 100})
    assert r.status_code == 200, r.text[:300]
    assert r.json()["id"] == "test_custom_api"
    listed = [s for s in client.get("/api/scenarios/custom").json()["scenarios"] if s["id"] == "test_custom_api"]
    assert listed and {"id", "name", "summary", "source", "created", "prior_prob", "prob"} <= set(listed[0])
    assert client.delete("/api/scenarios/custom/test_custom_api").status_code == 200


@needs_build
def test_custom_run_with_a_builtin_id_is_renamed(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "CUSTOM", tmp_path / "custom")       # keep data/results/custom untouched
    builtin = client.get("/api/scenarios").json()["scenarios"][0]["id"]
    spec = {"id": builtin, "name": "撞名的自定义情景", "phases": [
        {"name": "高通胀", "months": 24, "views": [{"var": "s_cpi_yoy", "mean": 6.0}]}]}
    r = client.post("/api/scenarios/custom/run", json={"spec": spec, "n_paths": 100})
    assert r.status_code == 200, r.text[:300]
    new = r.json()["id"]
    assert new == f"custom_{builtin}"[:40] and r.json()["spec"]["id"] == new
    assert (tmp_path / "custom" / f"{new}.json").exists()
    assert client.get(f"/api/scenarios/{builtin}").json()["source"] == "builtin"     # the built-in still wins
    assert client.get(f"/api/scenarios/{new}").json()["name"] == "撞名的自定义情景"


@needs_build
def test_custom_run_refuses_a_spec_with_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "CUSTOM", tmp_path / "custom")
    r = client.post("/api/scenarios/custom/run", json={"spec": {"name": "坏规格", "phases": "不是列表"}, "n_paths": 100})
    assert r.status_code == 400 and "phases 必须是列表" in r.json()["detail"]
    assert not (tmp_path / "custom").exists()                         # nothing was simulated or saved


def test_llm_status_shape():
    r = client.get("/api/llm/status")
    if r.status_code == 500:  # translator module not built yet
        pytest.skip("translator unavailable")
    body = r.json()
    assert "available" in body and "model" in body


needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")


@needs_panel
@pytest.mark.parametrize("body, needle", [
    ({}, "缺少参数：train_end、months"),
    ({"train_end": "1999-13", "months": 60}, "不是合法的月份"),
    ({"train_end": "1980-12", "months": 60}, "训练期至少需要 15 年"),
    ({"train_end": "2999-01", "months": 12}, "不在数据范围内"),
    ({"train_end": "0000-01", "months": 36}, "不在数据范围内"),          # was a 500 (pandas cannot format year 0)
    ({"train_end": "2025-12", "months": 120}, "超出数据的最后一个月"),
    ({"train_end": "2007-06", "months": 6}, "months 必须是 12 到 180 之间的整数"),
    ({"train_end": "2007-06", "months": 36, "n_paths": 1000}, "n_paths 必须是 100 到 400"),
    ({"train_end": "2007-06", "months": 36, "seed": "x"}, "seed 必须是"),
    ({"train_end": "2007-06", "months": 36, "speed": 1}, "不认识的参数：speed"),
])
def test_custom_backtest_rejects_bad_windows_in_chinese(body, needle):
    r = client.post("/api/backtest/run", json=body)
    assert r.status_code == 400, r.text[:200]
    assert needle in r.json()["detail"]


@needs_panel
def test_custom_backtest_window_shape_cache_and_seed(monkeypatch):
    monkeypatch.setattr(server, "_backtest_cache", server.OrderedDict())       # start cold, leave no state behind
    body = {"train_end": "2019-12", "months": 24, "n_paths": 100, "seed": 5}
    r = client.post("/api/backtest/run", json=body)
    assert r.status_code == 200, r.text[:300]
    win = r.json()
    assert {"id", "name", "train", "test", "phases", "dates", "assets", "corr", "macro", "ep", "summary", "shocks",
            "custom", "label", "elapsed_s"} <= set(win)
    assert win["name"] == "自选窗口" and win["test"] == ["2020-01", "2021-12"] and win["n_paths"] == 100
    assert {"fan_cond", "fan_uncond", "coverage_cond", "pit_terminal_cond", "decomposition"} <= set(win["assets"]["us_equity"])
    assert win["summary"]["interpretation"] and win["corr"]["posterior"]
    again = client.post("/api/backtest/run", json={**body, "months": "24"})          # same key: served from the cache
    assert again.content == r.content
    server._backtest_cache.clear()                                                    # recomputed: same seed, same numbers
    redo = client.post("/api/backtest/run", json=body).json()
    assert redo["assets"]["us_equity"]["fan_cond"] == win["assets"]["us_equity"]["fan_cond"]
    other = client.post("/api/backtest/run", json={**body, "seed": 6}).json()
    assert other["assets"]["us_equity"]["fan_cond"]["p50"] != win["assets"]["us_equity"]["fan_cond"]["p50"]
    assert other["assets"]["us_equity"]["cagr_actual"] == win["assets"]["us_equity"]["cagr_actual"]


@pytest.mark.parametrize("horizon, needle", [("abc", "收到「abc」"), (180.5, "收到「180.5」"), (6, "收到 6"), (601, "收到 601"),
                                             (True, "horizon_months 必须是"), ([180], "horizon_months 必须是")])
def test_llm_translate_rejects_a_bad_horizon_with_a_chinese_400(horizon, needle):
    """Checked before the credentials, so the answer is the same with or without an API key."""
    if client.get("/api/llm/status").status_code == 500:
        pytest.skip("translator unavailable")
    r = client.post("/api/llm/translate", json={"narrative": "油价翻倍、美联储被迫加息", "horizon_months": horizon})
    assert r.status_code == 400, r.text[:200]
    assert "horizon_months 必须是 12 到 600 之间的整数" in r.json()["detail"] and needle in r.json()["detail"]
    r = client.post("/api/llm/translate", json={"narrative": "长" * 20001})
    assert r.status_code == 400 and "叙事过长" in r.json()["detail"]
