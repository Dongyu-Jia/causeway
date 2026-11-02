"""P9 (SIM_DESIGN §11.3; AGENT_DESIGN §14): quarterly scenario sets for 组合评估 and 提取 (step_months = 3).

A monthly fixture set and the same paths summed to quarter-ends: quarterly rebalancing on the quarterly set equals
rebalancing every third month on the monthly one; volatility is annualized by √4; yearly withdrawals are unchanged
(pro rata exactly); the quarterly set's own continuation; a monthly custom scenario inside a quarterly set; the agent
tree's export layout loads the same way; the default set follows the switch; a monthly request is exactly what it was.
The existing portfolio and withdrawal tests run unchanged on the monthly sets (tests/test_portfolio.py,
tests/test_withdrawal.py). Only the last test needs data (the real quarterly stack, a few paths)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scenariolab import config, jsonio
from scenariolab.evaluation import switch as SW
from scenariolab.portfolio import metrics, qsets, service
from scenariolab.portfolio import withdrawal as wd
from scenariolab.server import app

client = TestClient(app, raise_server_exceptions=False)
ASSETS = list(config.ASSET_IDS)
N, YEARS = 40, 10
T = 12 * YEARS
IDS = ("calm", "storm", "boom")
QMODEL = "qmodel-r0-f0-a0"                    # the model version the fixture set was built on (and T-S certified)
W = {"us_equity": 0.6, "treasury_10y": 0.4}
PANEL_Q = config.ROOT / "data" / "processed" / "panel_quarterly.csv"
needs_panel = pytest.mark.skipif(not PANEL_Q.exists(), reason="quarterly panel not built (data/ is not in CI)")


def _monthly(seed=0, n=N, ids=IDS, drift=(0.006, -0.004, 0.012)):
    rng = np.random.default_rng(seed)
    S, A = len(ids), len(ASSETS)
    total = np.empty((S, n, T, A), dtype=np.float32)
    for s in range(S):
        total[s] = rng.normal(drift[s], 0.035, size=(n, T, A))
    total[..., ASSETS.index("cash")] = 0.003
    income = np.full_like(total, 0.002)
    income[..., ASSETS.index("cash")] = 0.003
    income[..., ASSETS.index("gold")] = 0.0
    cpi = 100.0 * np.cumprod(np.concatenate([np.ones((S, n, 1)), 1.0 + rng.normal(0.0025, 0.002, (S, n, T))], axis=2),
                             axis=2)
    labels = [f"{2026 + (8 + k) // 12}-{(8 + k) % 12 + 1:02d}" for k in range(T + 1)]
    return total, income, cpi.astype(np.float32), labels


def _quarterly(total, income, cpi):
    """The monthly paths summed to quarter-ends in the stack's convention: total compounds, 1 + total - income is the
    quarter's price growth, the CPI at the quarter ends."""
    S, n, Tm, A = total.shape
    r = total.astype(np.float64).reshape(S, n, Tm // 3, 3, A)
    inc = income.astype(np.float64).reshape(S, n, Tm // 3, 3, A)
    tq = np.prod(1.0 + r, axis=3) - 1.0
    g = np.prod(1.0 + r - inc, axis=3)
    return tq.astype(np.float32), (1.0 + tq - g).astype(np.float32), cpi[:, :, ::3]


def _cont(seed=5, rows=len(IDS) * N, years=30):
    """A quarterly continuation's yearly factors (what qsets writes as cont_*)."""
    rng = np.random.default_rng(seed)
    q = 4 * years
    tot = rng.normal(0.012, 0.05, (rows, q, len(ASSETS)))
    tot[:, :, ASSETS.index("cash")] = 0.009
    inc = np.full_like(tot, 0.006)
    inc[:, :, ASSETS.index("cash")] = 0.009
    cpi = np.cumprod(np.concatenate([np.ones((rows, 1)), 1.0 + rng.normal(0.007, 0.004, (rows, q))], axis=1), axis=1)
    return wd.year_factors(tot, inc, cpi, months=4)


def _write_sets(root, cont=True, cont_start=None):
    root.mkdir(parents=True, exist_ok=True)
    total, income, cpi, labels = _monthly()
    np.savez_compressed(root / service.PATHS_FILE, total=total, income=income, cpi=cpi, ids=np.array(IDS),
                        assets=np.array(ASSETS), labels=np.array(labels))
    jsonio.dump({"scenarios": [{"id": "calm", "name": "平稳", "prob": 0.5}, {"id": "storm", "name": "风暴", "prob": 0.3},
                               {"id": "boom", "name": "繁荣", "prob": 0.2}]}, root / service.INDEX_FILE)
    tq, iq, cq = _quarterly(total, income, cpi)
    qlabels = labels[::3]
    arrays = {"total": tq, "income": iq, "cpi": cq.astype(np.float64), "ids": np.array(IDS), "assets": np.array(ASSETS),
              "labels": np.array(qlabels), "probs": np.array([0.5, 0.3, 0.2]), "step_months": np.array(3)}
    if cont:
        cf = _cont()
        clabels = [service._shift_label(qlabels[-1], 12 * k) for k in range(cf.years + 1)]
        arrays.update(cont_total_y=cf.total.astype(np.float32), cont_price_y=cf.price.astype(np.float32),
                      cont_income_y=cf.income.astype(np.float32), cont_cpi_y=cf.cpi, cont_labels=np.array(clabels),
                      cont_start=np.array(cont_start or qlabels[0]), cont_offset=np.array(4 * YEARS),
                      cont_info=np.array(json.dumps({"method": "quarterly_stack", "text": "同一个模型不加陈述接着往下推"})))
    np.savez_compressed(root / service.QUARTERLY_FILE, **arrays)
    jsonio.dump({"start": qlabels[0], "horizon_months": T, "n_paths": N, "step_months": 3,
                 "scenarios": [{"id": "calm", "name": "平稳", "prob": 0.5, "prior_prob": 0.5},
                               {"id": "storm", "name": "风暴", "prob": 0.3, "prior_prob": 0.3},
                               {"id": "boom", "name": "繁荣", "prob": 0.2, "prior_prob": 0.2}],
                 "set": {"engine": "quarterly", "step_months": 3, "text": "测试用季度情景集", "n_paths": N,
                         "model": QMODEL}},
                root / service.QUARTERLY_INDEX)
    return total, income, cpi


@pytest.fixture
def sets(tmp_path, monkeypatch):
    root = tmp_path / "results"
    monkeypatch.setattr(service, "RESULTS_DIR", root)
    monkeypatch.setattr(service, "AGENT_SETS_DIR", tmp_path / "agents_sets")
    monkeypatch.setattr(SW, "state", lambda results_dir=None: {"status": "not_evaluated", "quarterly": False,
                                                               "passed": False, "text": "还没有评估",
                                                               "default_text": "默认引擎：月度引擎", "reasons": []})
    service.clear_cache()
    arrays = _write_sets(root)
    yield {"root": root, "arrays": arrays}
    service.clear_cache()


# ================================================================ metrics on steps of a quarter

def test_quarterly_rebalancing_is_rebalancing_every_third_month(sets):
    total, income, cpi = sets["arrays"]
    tq, _, _ = _quarterly(total, income, cpi)
    w = np.zeros(len(ASSETS))
    w[ASSETS.index("us_equity")], w[ASSETS.index("gold")], w[ASSETS.index("treasury_10y")] = 0.5, 0.2, 0.3
    m, q = total[0], tq[0]
    for rule in ("quarterly", "annual", "none"):
        _, mode_m, k_m = service.rebalance_rule(rule, 1)
        _, mode_q, k_q = service.rebalance_rule(rule, 3)
        wm = metrics.portfolio_wealth(m, w, mode_m, **({"period": k_m} if mode_m == "annual" else {}))
        wq = metrics.portfolio_wealth(q, w, mode_q, **({"period": k_q} if mode_q == "annual" else {}))
        assert np.allclose(wq, wm[:, ::3], rtol=2e-6), rule
    assert service.rebalance_rule("quarterly", 3) == ("quarterly", "monthly", 1)      # every step
    assert service.rebalance_rule("annual", 3) == ("annual", "annual", 4)
    assert service.rebalance_rule(None, 3)[0] == "quarterly" and service.rebalance_rule(None, 1)[0] == "monthly"
    assert service.rebalance_rule("monthly", 1) == ("monthly", "monthly", 1)
    with pytest.raises(ValueError, match="不能每月再平衡"):
        service.rebalance_rule("monthly", 3)
    with pytest.raises(ValueError):
        service.rebalance_rule("weekly", 3)


def test_volatility_sharpe_and_sortino_are_annualized_by_four_steps():
    r = np.array([[0.04, -0.02, 0.04, -0.02, 0.04, -0.02, 0.04, -0.02]])          # two years of quarters
    wealth = np.concatenate([[[1.0]], np.cumprod(1 + r, axis=1)], axis=1)
    m = metrics.path_metrics(wealth, np.ones_like(wealth), 0.0, periods=4)
    assert m["vol"][0] == pytest.approx(0.03 * 2)                                 # std 0.03 × √4
    assert m["sharpe"][0] == pytest.approx(0.01 * 4 / 0.06)
    assert m["sortino"][0] == pytest.approx(0.04 / np.sqrt(0.0002 * 4))
    assert m["cagr"][0] == pytest.approx(wealth[0, -1] ** 0.5 - 1)                 # 8 steps = 2 years


# ================================================================ the service on a quarterly set

def test_the_portfolio_on_the_quarterly_set(sets):
    q = service.evaluate_portfolio(W, scenario_set="quarterly")
    mq = service.evaluate_portfolio(W, scenario_set="monthly", rebalance="quarterly")
    assert q["step_months"] == 3 and q["horizon_months"] == T and q["rebalance"] == "quarterly"
    assert q["years"] == [str(2026 + k) for k in range(YEARS + 1)] and q["dates"][:2] == ["2026-09", "2027-09"]
    assert q["dates"] == mq["dates"] and mq["step_months"] == 1
    for a, b in zip(q["per_scenario"], mq["per_scenario"]):
        # the same wealth at every quarter end: terminal wealth and CAGRs agree, the fans at the year ends too
        for k in ("cagr_p50", "real_cagr_p50", "cagr_p5", "cvar5_terminal", "prob_loss_real"):
            assert a[k] == pytest.approx(b[k], abs=2e-5), k
        assert np.allclose(a["fan"]["p50"], b["fan"]["p50"], atol=2e-5)
        assert a["max_dd_p50"] >= b["max_dd_p50"] - 1e-9                          # quarter-ends miss the deepest month
    # the volatility is the quarterly returns' sd × √4 (per path, then the mean)
    data = service.load_scenarios(scenario_set="quarterly")
    _, mode, _ = service.rebalance_rule("quarterly", 3)
    wvec = service.parse_weights(W, data.assets)
    blk = data.parts[0][0]
    wealth = metrics.portfolio_wealth(blk.total[:N], wvec, mode)
    ret = wealth[:, 1:] / wealth[:, :-1] - 1
    assert q["per_scenario"][0]["vol"] == pytest.approx(float(np.mean(ret.std(axis=1) * 2)), rel=1e-6)
    ss = q["scenario_set"]
    assert ss["id"] == "quarterly" and ss["steps_per_year"] == 4 and ss["trial"] is True and ss["label"] == "试运行"
    assert any("季末口径" in n for n in q["notes"]) and any("√4" in n for n in q["notes"])
    assert any("跑输现金的季度" in n for n in q["notes"])
    with pytest.raises(ValueError, match="不能每月再平衡"):
        service.evaluate_portfolio(W, scenario_set="quarterly", rebalance="monthly")
    with pytest.raises(ValueError, match="季度情景集暂不支持标的"):
        service.evaluate_portfolio({"QQQ": 0.5, "us_equity": 0.5}, scenario_set="quarterly")


def test_yearly_withdrawals_are_unchanged_on_a_quarterly_set(sets):
    total, income, cpi = sets["arrays"]
    tq, iq, cq = _quarterly(total, income, cpi)
    S, n = total.shape[:2]
    fm = wd.year_factors(total.reshape(S * n, T, -1), income.reshape(S * n, T, -1), cpi.reshape(S * n, T + 1))
    fq = wd.year_factors(tq.reshape(S * n, T // 3, -1), iq.reshape(S * n, T // 3, -1), cq.reshape(S * n, T // 3 + 1),
                         months=4)
    assert fq.years == fm.years == YEARS
    assert np.allclose(fq.total, fm.total, rtol=2e-6) and np.allclose(fq.price, fm.price, rtol=2e-6)
    assert np.allclose(fq.cpi, fm.cpi, rtol=1e-6)
    # the income: summing monthly paths to quarters counts what a quarter's dividends earn inside the quarter as income
    # (second order: about 1% of the income itself in the median, under 10% at worst on these volatile fixtures); the
    # quarterly stack's own paths have no such term (its income is paid on the quarter's start value). It moves only
    # the dividend-first rules' cash pot, never pro rata.
    rel = np.abs(fq.income - fm.income) / np.maximum(np.abs(fm.income), 1e-9)
    assert float(np.median(rel)) < 0.01 and float(rel.max()) < 0.1
    kw = dict(rate=0.045, horizon_years=YEARS)
    q = service.evaluate_withdrawal(W, scenario_set="quarterly", **kw)
    m = service.evaluate_withdrawal(W, scenario_set="monthly", **kw)
    assert q["step_months"] == 3 and q["scenario_years"] == m["scenario_years"] == YEARS
    pq, pm = q["strategies"]["pro_rata"], m["strategies"]["pro_rata"]
    assert pq["aggregate"]["ruin_prob_weighted"] == pytest.approx(pm["aggregate"]["ruin_prob_weighted"])
    assert pq["aggregate"]["terminal_real_p50_weighted"] == pytest.approx(pm["aggregate"]["terminal_real_p50_weighted"],
                                                                         rel=1e-5)
    assert q["safe_rate"]["pro_rata"]["weighted"] == pytest.approx(m["safe_rate"]["pro_rata"]["weighted"], abs=1e-4)
    for s in ("div_then_pro_rata", "div_then_winners"):
        a, b = q["strategies"][s]["aggregate"], m["strategies"][s]["aggregate"]
        assert abs(a["ruin_prob_weighted"] - b["ruin_prob_weighted"]) <= 0.02
        assert a["terminal_real_p50_weighted"] == pytest.approx(b["terminal_real_p50_weighted"], rel=5e-3)
    assert any("提取仍是每年年末一次" in n for n in q["notes"]) and any("市价股息比" in n for n in q["notes"])
    assert q["scenario_set"]["id"] == "quarterly"


def test_the_quarterly_set_s_own_continuation(sets):
    r = service.evaluate_withdrawal(W, scenario_set="quarterly", horizon_years=20, strategy="pro_rata")
    c = r["continuation"]
    assert r["horizon_years"] == 20 and r["scenario_years"] == YEARS and c["label"] == service.continuation_label_q(YEARS)
    assert c["method"] == "quarterly_stack" and c["first_year"] == YEARS + 1 and c["years"] == 10
    assert any("同一个模型不加陈述接着往下推" in n for n in r["notes"])
    # the first ten years are the same withdrawal runs as a ten-year horizon
    ten = service.evaluate_withdrawal(W, scenario_set="quarterly", horizon_years=YEARS, strategy="pro_rata")
    assert r["scenario_horizon"]["strategies"]["pro_rata"]["aggregate"]["ruin_prob_weighted"] == pytest.approx(
        ten["strategies"]["pro_rata"]["aggregate"]["ruin_prob_weighted"])
    # a set without continuation arrays, or with one built for another start: horizons past its paths wait
    _write_sets(sets["root"], cont=False)
    service.clear_cache()
    with pytest.raises(service.PathsNotReady, match="没有续推数组"):
        service.evaluate_withdrawal(W, scenario_set="quarterly", horizon_years=20)
    _write_sets(sets["root"], cont_start="2020-09")
    service.clear_cache()
    with pytest.raises(service.PathsNotReady, match="2020-09"):
        service.evaluate_withdrawal(W, scenario_set="quarterly", horizon_years=20)
    r = client.post("/api/withdrawal/evaluate", json={"weights": W, "scenario_set": "quarterly", "horizon_years": 20})
    assert r.status_code == 503 and "续推" in r.json()["detail"]


def test_a_monthly_custom_scenario_in_a_quarterly_set(sets):
    root = sets["root"]
    (root / "custom").mkdir()
    total, income, cpi, labels = _monthly(seed=9, ids=("c1",), drift=(-0.01,))
    np.savez_compressed(root / "custom" / "c1.npz", total=total, income=income, cpi=cpi, ids=np.array(["c1"]),
                        assets=np.array(ASSETS), labels=np.array(labels))
    jsonio.dump({"id": "c1", "name": "自定义崩盘", "prior_prob": 0.1}, root / "custom" / "c1.json")
    q = service.evaluate_withdrawal(W, scenario_set="quarterly", include_custom=["c1"], horizon_years=YEARS)
    m = service.evaluate_withdrawal(W, scenario_set="monthly", include_custom=["c1"], horizon_years=YEARS)
    for s in wd.STRATEGIES:                     # its yearly factors are its own monthly ones: identical figures
        a = q["strategies"][s]["per_scenario"][-1]
        b = m["strategies"][s]["per_scenario"][-1]
        assert a["id"] == b["id"] == "c1" and a["ruin_prob"] == b["ruin_prob"]
        assert a["terminal_real_p50"] == pytest.approx(b["terminal_real_p50"])
    assert any("来自月度引擎：按季末汇总" in n for n in q["notes"])
    p = service.evaluate_portfolio(W, scenario_set="quarterly", include_custom=["c1"])
    pm = service.evaluate_portfolio(W, scenario_set="monthly", include_custom=["c1"], rebalance="quarterly")
    assert p["per_scenario"][-1]["cagr_p50"] == pytest.approx(pm["per_scenario"][-1]["cagr_p50"], abs=2e-5)


def test_the_agent_tree_export_layout_loads_as_a_set(sets, tmp_path):
    """agents/tree.export_arrays writes the scenario_paths.npz layout with step_months = 3 (AGENT_DESIGN §14): such a
    file under data/agents/sets/ is a scenario set (P11 enables the export; the services already accept it)."""
    from scenariolab.agents import tree as TR
    rng = np.random.default_rng(3)
    leaves = [SimpleNamespace(key=f"leaf{i}", p_path=p, cloud=SimpleNamespace(
        n=N, path=rng.normal(0.01, 0.05, (N, 4 * YEARS, len(ASSETS))), cpi=np.full(N, 0.25)))
        for i, p in enumerate((0.6, 0.4))]
    arr = TR.export_arrays(leaves, ASSETS, first_label="2026-09")
    assert int(arr["step_months"]) == 3 and TR.EXPORT_ENABLED is False
    base = tmp_path / "agents_sets"
    base.mkdir()
    np.savez_compressed(base / "r1.npz", **arr)
    jsonio.dump({"scenarios": [{"id": "leaf0", "name": "分支一", "prob": 0.6}, {"id": "leaf1", "name": "分支二", "prob": 0.4}],
                 "set": {"engine": "quarterly", "name": "智能体推演 r1", "step_months": 3}}, base / "r1.json")
    r = service.evaluate_portfolio(W, scenario_set="agent:r1")
    assert r["step_months"] == 3 and [e["id"] for e in r["per_scenario"]] == ["leaf0", "leaf1"]
    assert [e["source"] for e in r["per_scenario"]] == ["agent", "agent"] and r["scenario_set"]["name"] == "智能体推演 r1"
    assert [e["prob"] for e in r["per_scenario"]] == pytest.approx([0.6, 0.4])
    assert any(s["id"] == "agent:r1" and s["available"] for s in service.sets_report()["sets"])
    for bad in ("agent:../x", "agent:", "weekly", 3):
        with pytest.raises(ValueError):
            service.evaluate_portfolio(W, scenario_set=bad)
    with pytest.raises(ValueError, match="不存在"):
        service.evaluate_portfolio(W, scenario_set="agent:nope")


def test_the_default_set_follows_the_switch(sets, monkeypatch):
    r = service.evaluate_portfolio(W)
    assert r["scenario_set"]["id"] == "monthly" and r["scenario_set"]["chosen_by_default"] and r["step_months"] == 1
    monkeypatch.setattr(SW, "state", lambda results_dir=None: {"status": "passed", "quarterly": True, "passed": True,
                                                               "text": "通过", "default_text": "默认引擎：季度模型", "model": {"semantic": QMODEL},
                                                               "reasons": []})
    r = service.evaluate_portfolio(W)
    ss = r["scenario_set"]
    assert ss["id"] == "quarterly" and ss["chosen_by_default"] and ss["trial"] is False and ss["label"] == "默认"
    assert r["step_months"] == 3 and r["rebalance"] == "quarterly"
    assert service.evaluate_withdrawal(W, horizon_years=YEARS)["step_months"] == 3
    rep = service.sets_report()
    assert rep["default"] == "quarterly" and next(s for s in rep["sets"] if s["id"] == "quarterly")["is_default"]
    # a set built on another model version than the one T-S certified is a trial set, not the default
    idx = jsonio.load(sets["root"] / service.QUARTERLY_INDEX)
    idx["set"]["model"] = "qmodel-r0-f9-a0"
    jsonio.dump(idx, sets["root"] / service.QUARTERLY_INDEX)
    service.clear_cache()
    sinfo = service.resolve_set(None)
    assert sinfo["id"] == "monthly" and sinfo["note"] == service.STALE_QSET_NOTE
    rep = service.sets_report()
    q = next(s for s in rep["sets"] if s["id"] == "quarterly")
    assert rep["default"] == "monthly" and q["is_default"] is False and q["trial"] is True and q["issue"]
    assert next(s for s in rep["sets"] if s["id"] == "monthly")["is_default"] is True
    ss = service.evaluate_portfolio(W, scenario_set="quarterly")["scenario_set"]
    assert ss["label"] == "试运行" and ss["is_default"] is False and ss["trial"] is True
    idx["set"]["model"] = QMODEL
    jsonio.dump(idx, sets["root"] / service.QUARTERLY_INDEX)
    service.clear_cache()
    # an explicit monthly request stays monthly; a missing quarterly set falls back to monthly with a note
    assert service.evaluate_portfolio(W, scenario_set="monthly")["step_months"] == 1
    (sets["root"] / service.QUARTERLY_FILE).unlink()
    service.clear_cache()
    r = service.evaluate_portfolio(W)
    assert r["scenario_set"]["id"] == "monthly" and "季度情景集还没有生成" in r["notes"][0]


def test_a_monthly_request_is_exactly_what_it_was(sets):
    """The default (the switch not passed) and an explicit monthly set give the monthly engine's numbers, computed the
    same way as before sets existed (portfolio_report on load_scenarios() with the old default rebalancing)."""
    a = service.evaluate_portfolio(W)
    b = service.evaluate_portfolio(W, scenario_set="monthly")
    c = service.portfolio_report(service.load_scenarios(), W, None, "monthly")
    for x in (a, b, c):
        x.pop("scenario_set")
    assert a == b == c and a["rebalance"] == "monthly"
    wa = service.evaluate_withdrawal(W, horizon_years=YEARS)
    wb = service.withdrawal_report(service.load_scenarios(), W, horizon_years=YEARS)
    wa.pop("scenario_set")
    wb.pop("scenario_set")
    assert wa == wb


def test_the_endpoints_take_the_set(sets):
    r = client.post("/api/portfolio/evaluate", json={"weights": W, "scenario_set": "quarterly", "rebalance": "annual"})
    assert r.status_code == 200 and r.json()["step_months"] == 3 and r.json()["rebalance"] == "annual"
    r = client.post("/api/portfolio/evaluate", json={"weights": W, "scenario_set": "quarterly", "rebalance": "monthly"})
    assert r.status_code == 400 and "不能每月再平衡" in r.json()["detail"]
    r = client.post("/api/portfolio/evaluate", json={"weights": W, "scenario_set": "weekly"})
    assert r.status_code == 400 and "未知情景集" in r.json()["detail"]
    r = client.post("/api/withdrawal/evaluate", json={"weights": W, "scenario_set": "quarterly", "horizon_years": 10})
    assert r.status_code == 200 and r.json()["scenario_set"]["id"] == "quarterly"
    rep = client.get("/api/portfolio/sets").json()
    q = next(s for s in rep["sets"] if s["id"] == "quarterly")
    assert rep["default"] == "monthly" and q["available"] and q["step_months"] == 3 and q["trial"]
    assert [s["id"] for s in q["scenarios"]] == list(IDS) and rep["rebalance"]["quarterly"] == ["quarterly", "annual", "none"]


# ================================================================ the builder (portfolio/qsets.py)

def test_flagged_paths_are_replaced_and_too_many_leave_a_scenario_out():
    rng = np.random.default_rng(0)
    n, H, K, A = 20, 8, 12, 3
    tl = rng.normal(0, 0.05, (n, H + K, A))
    inc = np.full_like(tl, 0.01)
    dcpi = np.full((n, H + K), 2.0)
    bad = np.zeros((n, H + K), bool)
    bad[0, 2] = bad[1, 5] = True                       # inside the scenario quarters: whole path replaced
    bad[2, H + 3] = True                               # only in the continuation: the head stays
    t2, i2, c2, q = qsets.clean_paths(tl.copy(), inc.copy(), dcpi.copy(), bad, np.random.default_rng(1), H)
    assert q["head_flagged"] == pytest.approx(0.1) and q["cont_flagged"] == pytest.approx(0.05) and not q["excluded"]
    clean = [r for r in range(n) if r not in (0, 1)]
    for r in (0, 1):
        assert any(np.array_equal(t2[r], tl[c]) for c in clean)
    assert np.array_equal(t2[2, :H], tl[2, :H]) and not np.array_equal(t2[2, H:], tl[2, H:])
    assert np.array_equal(t2[3], tl[3])
    many = np.zeros_like(bad)
    many[:11, 0] = True
    assert qsets.clean_paths(tl, inc, dcpi, many, np.random.default_rng(1), H)[3]["excluded"] is True


def test_quarter_end_labels():
    assert qsets.quarter_end_labels("2026Q4", 4) == ["2026-09", "2026-12", "2027-03", "2027-06", "2027-09"]


def test_the_headline_numbers_use_the_monthly_sets_units():
    n, H = 5, 60
    tot = np.full((n, H + 4, len(ASSETS)), 0.02)
    dcpi = np.full((n, H + 4), 2.4)                                   # annualized percent
    cpi = np.concatenate([np.ones((n, 1)), np.exp(np.cumsum(dcpi / 400.0, axis=1))], axis=1)
    hl = qsets.headline(tot, cpi, dcpi, np.full(n, 3.1), ASSETS, H)
    assert hl["us_equity_cagr_p50"] == pytest.approx(1.02 ** 4 - 1, abs=1e-6)
    assert hl["us_equity_real_cagr_p50"] == pytest.approx(1.02 ** 4 / np.exp(0.024) - 1, abs=1e-6)
    assert hl["cpi_avg"] == pytest.approx(2.4) and hl["ff_end_p50"] == pytest.approx(3.1)


@needs_panel
def test_the_builder_on_the_quarterly_stack(tmp_path, monkeypatch):
    """Two built-ins at 30 paths on the real stack: the file loads as the quarterly set (60 quarters, step 3, its own
    continuation to year 40 or a recorded reason), probabilities are the hand-set priors normalised."""
    monkeypatch.setattr(service, "RESULTS_DIR", tmp_path)
    service.clear_cache()
    out = qsets.build(n_paths=30, ids=["baseline", "stagflation"], results=tmp_path)
    assert out["shape"][1:] == [30, 60, 13] and (tmp_path / service.QUARTERLY_INDEX).exists()
    idx = jsonio.load(tmp_path / service.QUARTERLY_INDEX)
    kept = idx["scenarios"]
    assert sum(s["prob"] for s in kept) == pytest.approx(1.0)
    pri = {s["id"]: s["prior_prob"] for s in kept}
    assert all(s["prob"] == pytest.approx(pri[s["id"]] / sum(pri.values()), abs=1e-6) for s in kept)
    assert idx["set"]["step_months"] == 3 and idx["set"]["model"].startswith("qmodel-") and idx["set"]["quality"]
    assert all(np.isfinite(s["headline"]["us_equity_cagr_p50"]) for s in kept)
    data = service.load_scenarios(scenario_set="quarterly", results_dir=tmp_path)
    assert data.step_months == 3 and data.months == 60 and data.years == 15 and data.labels[0] == "2026-09"
    assert np.isfinite(data.factors.total).all() and np.isfinite(data.parts[0][0].total).all()
    if out["continuation"]:
        assert data.max_years == 40
    r = service.evaluate_portfolio(W, scenario_set="quarterly")
    assert r["step_months"] == 3 and r["scenario_set"]["trial"] is True
    service.clear_cache()
