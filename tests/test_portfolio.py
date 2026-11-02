import json
import os
import time

import numpy as np
import pytest

from scenariolab import config, jsonio
from scenariolab.portfolio import metrics, service

ASSETS = list(config.ASSET_IDS)


# ---------------------------------------------------------------- metrics, hand-computed

def test_rebalancing_modes_hand_computed():
    # asset 0: +100% in month 0, -25% in month 1, +100% in month 12; asset 1 flat
    total = np.zeros((1, 24, 2))
    total[0, 0, 0], total[0, 1, 0], total[0, 12, 0] = 1.0, -0.25, 1.0
    w = np.array([0.5, 0.5])
    monthly = metrics.portfolio_wealth(total, w, "monthly")[0]
    annual = metrics.portfolio_wealth(total, w, "annual")[0]
    none = metrics.portfolio_wealth(total, w, "none")[0]
    # monthly: 1 * 1.5 * (1 - 0.125) = 1.3125, then * 1.5
    assert np.allclose(monthly[[1, 2, 12, 13, 24]], [1.5, 1.3125, 1.3125, 1.96875, 1.96875])
    # annual: drift inside year 1 (0.75 + 0.5 = 1.25), rebalance to 0.625 / 0.625, then 1.25 + 0.625
    assert np.allclose(annual[[1, 2, 12, 13, 24]], [1.5, 1.25, 1.25, 1.875, 1.875])
    # buy-and-hold: 0.75 -> 1.5 in asset 0 plus 0.5 in asset 1
    assert np.allclose(none[[1, 2, 12, 13, 24]], [1.5, 1.25, 1.25, 2.0, 2.0])
    assert monthly[0] == annual[0] == none[0] == 1.0
    with pytest.raises(ValueError):
        metrics.portfolio_wealth(total, w, "weekly")


def test_rebalancing_modes_agree_for_a_single_asset_and_float32():
    rng = np.random.default_rng(0)
    total = rng.normal(0.005, 0.04, size=(20, 30, 3)).astype(np.float32)
    w = np.array([0.0, 1.0, 0.0])
    ref = np.cumprod(1.0 + total[:, :, 1].astype(float), axis=1)
    for mode in metrics.REBALANCE_MODES:
        assert np.allclose(metrics.portfolio_wealth(total, w, mode)[:, 1:], ref, rtol=1e-6)


def test_drawdown_sharpe_sortino_hand_computed():
    wealth = np.array([[1.0, 1.2, 0.9, 1.5, 0.75, 1.0]])
    assert np.allclose(metrics.drawdowns(wealth)[0], [0, 0, -0.25, 0, -0.5, 1.0 / 1.5 - 1])
    assert metrics.max_drawdown(wealth)[0] == pytest.approx(-0.5)
    r = np.array([[0.02, -0.01, 0.03, 0.00]])
    rf = np.full_like(r, 0.005)
    # excess mean 0.005/month = 6%/yr; vol = sqrt(2.5e-4 * 12); downside = sqrt(0.625e-4 * 12)
    assert metrics.sharpe(r, rf)[0] == pytest.approx(0.06 / np.sqrt(2.5e-4 * 12))
    assert metrics.sortino(r, rf)[0] == pytest.approx(0.06 / np.sqrt(0.625e-4 * 12))
    flat = np.zeros((1, 4))
    assert np.isnan(metrics.sharpe(flat, 0.0)[0]) and np.isnan(metrics.sortino(flat + 0.01, 0.0)[0])
    assert metrics.cagr(np.array([2.0]), 10)[0] == pytest.approx(2 ** 0.1 - 1)


def test_path_metrics_and_summary_on_known_paths():
    wealth = np.array([np.linspace(1.0, 1.21, 25), np.linspace(1.0, 0.81, 25)])   # 2 years
    cpi = np.tile(np.linspace(100.0, 110.0, 25), (2, 1))
    m = metrics.path_metrics(wealth, cpi, 0.0)
    assert np.allclose(m["cagr"], [0.1, -0.1])
    assert np.allclose(m["terminal_real"], [1.21 / 1.1, 0.81 / 1.1])
    s = metrics.summarize(m)
    assert s["prob_loss_real"] == 0.5                     # 1.21/1.1 > 1 > 0.81/1.1
    assert s["max_dd_p50"] == pytest.approx(np.percentile([0.0, 0.81 - 1.0], 50, method="hazen"))
    assert s["max_dd_p95"] <= s["max_dd_p50"]             # the bad tail is deeper
    assert s["cvar5_terminal"] == pytest.approx(0.81)


# ---------------------------------------------------------------- weighted statistics

def test_weighted_percentile_equal_weights_is_hazen():
    rng = np.random.default_rng(1)
    x = rng.normal(size=777)
    pcts = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    assert np.allclose(metrics.weighted_percentile(x, pcts), np.percentile(x, pcts, method="hazen"))
    assert metrics.weighted_percentile(x, 50) == pytest.approx(np.percentile(x, 50, method="hazen"))
    # renormalization: any positive scale of the weights gives the same answer
    w = rng.uniform(0.1, 2.0, size=x.size)
    assert np.allclose(metrics.weighted_percentile(x, pcts, w), metrics.weighted_percentile(x, pcts, 37.5 * w))


def test_weighted_percentile_drops_zero_weights_and_nans():
    assert metrics.weighted_percentile([1.0, 2.0, 3.0, 100.0], 50, [1, 1, 1, 0]) == pytest.approx(2.0)
    assert metrics.weighted_percentile([1.0, np.nan, 3.0], 50) == pytest.approx(2.0)
    assert np.isnan(metrics.weighted_percentile([np.nan], 50))
    # values [0, 1] with weights 0.9 / 0.1: positions 0.45 / 0.95, so p50 = 0.1
    assert metrics.weighted_percentile([0.0, 1.0], 50, [0.9, 0.1]) == pytest.approx(0.1)


def test_mixture_weights_mix_by_probability_not_path_count():
    rng = np.random.default_rng(2)
    a = rng.uniform(0.0, 1.0, 10000)           # scenario 0, 10000 paths
    b = rng.uniform(10.0, 11.0, 2000)          # scenario 1, 2000 paths
    groups = np.r_[np.zeros(a.size, int), np.ones(b.size, int)]
    w = metrics.mixture_weights(groups, [0.8, 0.2])
    assert w.sum() == pytest.approx(1.0)
    assert w[groups == 0].sum() == pytest.approx(0.8) and w[groups == 1].sum() == pytest.approx(0.2)
    x = np.r_[a, b]
    # mixture quantiles: median at 0.5 / 0.8 = 0.625 inside scenario 0; p90 at 10.5 inside scenario 1
    assert metrics.weighted_percentile(x, 50, w) == pytest.approx(0.625, abs=0.01)
    assert metrics.weighted_percentile(x, 90, w) == pytest.approx(10.5, abs=0.03)
    # unnormalized probabilities are renormalized; a zero-probability set drops out
    assert np.allclose(metrics.mixture_weights(groups, [4, 1]), w)
    w0 = metrics.mixture_weights(groups, [1, 0])
    assert metrics.weighted_percentile(x, 99, w0) < 1.0
    with pytest.raises(ValueError):
        metrics.mixture_weights(groups, [0, 0])
    with pytest.raises(ValueError):
        metrics.normalize_probs([0.5, -0.1])
    assert np.allclose(metrics.normalize_probs([1e308, 1e308, 0.0]), [0.5, 0.5, 0.0])     # no overflow
    with pytest.raises(ValueError):
        metrics.normalize_probs([])


def test_weighted_cvar_and_mean():
    x = np.arange(1.0, 101.0)
    assert metrics.weighted_cvar(x, 0.05) == pytest.approx(3.0)            # mean of 1..5
    # 0.02 of mass at 1, 0.02 at 2, and 0.01 of the 0.96 at 3 fill the 5% tail
    assert metrics.weighted_cvar([1.0, 2.0, 3.0], 0.05, [0.02, 0.02, 0.96]) == pytest.approx(1.8)
    assert metrics.weighted_mean([1.0, 3.0], [3, 1]) == pytest.approx(1.5)
    fan = metrics.fan(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert set(fan) == {"p5", "p25", "p50", "p75", "p95"} and fan["p50"] == [2.0, 3.0]


# ---------------------------------------------------------------- service on a synthetic results dir

S, N, T = 3, 60, 120            # 10-year paths: the shortest withdrawal horizon needs no continuation
CONT_YEARS = 30                 # the synthetic continuation reaches the 40-year maximum


def _write_continuation(root, seed=21, n=N, years=CONT_YEARS, offset=T, start="2026-09", drift=0.004):
    """A synthetic unconditional continuation in the format build_continuation writes."""
    rng = np.random.default_rng(seed)
    Tc, A = 12 * years, len(ASSETS)
    total = rng.normal(drift, 0.03, size=(n, Tc, A)).astype(np.float32)
    total[..., ASSETS.index("cash")] = 0.003
    income = np.full_like(total, 0.002)
    income[..., ASSETS.index("cash")] = 0.003
    income[..., ASSETS.index("gold")] = 0.0
    cpi = 250.0 * np.exp(np.cumsum(np.c_[np.zeros(n), rng.normal(0.0025, 0.002, (n, Tc))], axis=1))
    labels = [service._shift_label(start, offset + k) for k in range(Tc + 1)]
    return service.write_continuation(root / service.CONTINUATION_FILE, total, income, cpi, ASSETS, labels,
                                      start=start, offset=offset, info={"method": "synthetic", "seed": seed})


def _write_paths(root, seed=0, n=N, ids=("calm", "storm", "boom"), drift=(0.006, -0.004, 0.012), custom=None,
                 inflation=None, start_month=8):
    """inflation: monthly CPI growth per scenario (default 0.3%); start_month: 0-based month of 2026."""
    rng = np.random.default_rng(seed)
    S_ = len(ids)
    A = len(ASSETS)
    total = np.empty((S_, n, T, A), dtype=np.float32)
    for s in range(S_):
        total[s] = rng.normal(drift[s], 0.03, size=(n, T, A))
    total[..., ASSETS.index("cash")] = 0.003
    income = np.full_like(total, 0.002)
    income[..., ASSETS.index("cash")] = 0.003
    income[..., ASSETS.index("gold")] = 0.0
    g = np.asarray(inflation if inflation is not None else [0.003] * S_, dtype=float)
    cpi = 100.0 * (1.0 + g[:, None, None]) ** np.arange(T + 1)[None, None, :] * np.ones((1, n, 1))
    labels = np.array([f"{2026 + (start_month + k) // 12}-{(start_month + k) % 12 + 1:02d}" for k in range(T + 1)])
    target = root if custom is None else root / "custom"
    target.mkdir(parents=True, exist_ok=True)
    name = "scenario_paths.npz" if custom is None else f"{custom}.npz"
    np.savez_compressed(target / name, total=total, income=income, cpi=cpi.astype(np.float32),
                        ids=np.array(ids), assets=np.array(ASSETS), labels=labels)
    return total


@pytest.fixture()
def results(tmp_path, monkeypatch):
    _write_paths(tmp_path)
    jsonio.dump({"scenarios": [{"id": "calm", "name": "平稳", "prob": 0.5},
                               {"id": "storm", "name": "风暴", "prob": 0.3},
                               {"id": "boom", "name": "繁荣", "prob": 0.2}]}, tmp_path / "scenarios.json")
    _write_paths(tmp_path, seed=9, n=30, ids=("c1",), drift=(-0.01,), custom="c1")
    jsonio.dump({"id": "c1", "name": "自定义崩盘", "prior_prob": 0.1}, tmp_path / "custom" / "c1.json")
    _write_continuation(tmp_path)
    monkeypatch.setattr(service, "RESULTS_DIR", tmp_path)
    service.clear_cache()
    yield tmp_path
    service.clear_cache()


def test_evaluate_portfolio_contract_shape(results):
    r = service.evaluate_portfolio({"us_equity": 60, "treasury_10y": 40})
    assert r["years"] == [str(2026 + k) for k in range(11)] and r["dates"][0] == "2026-09"
    assert r["horizon_months"] == T                         # the portfolio page keeps the scenario length
    assert r["weights"] == {"us_equity": 0.6, "treasury_10y": 0.4}
    keys = {"id", "name", "prob", "cagr_p5", "cagr_p50", "cagr_p95", "real_cagr_p50", "vol", "sharpe",
            "sortino", "max_dd_p50", "max_dd_p95", "cvar5_terminal", "prob_loss_real", "fan", "fan_real"}
    for e in r["per_scenario"]:
        assert keys <= set(e)
        assert len(e["fan"]["p50"]) == len(r["years"]) and e["fan"]["p50"][0] == 1.0
        assert e["cagr_p5"] <= e["cagr_p50"] <= e["cagr_p95"]
        assert e["max_dd_p95"] <= e["max_dd_p50"] <= 0
        assert e["fan_real"]["p50"][-1] < e["fan"]["p50"][-1]           # CPI rises 0.3%/month
    assert [e["prob"] for e in r["per_scenario"]] == pytest.approx([0.5, 0.3, 0.2])
    agg = r["aggregate"]
    assert set(agg) == {"weighted", "worst", "best", "equal_weight"}
    real = {e["id"]: e["real_cagr_p50"] for e in r["per_scenario"]}
    assert agg["worst"]["id"] == min(real, key=real.get) == "storm"
    assert agg["best"]["id"] == max(real, key=real.get) == "boom"
    medians = [e["cagr_p50"] for e in r["per_scenario"]]
    assert min(medians) <= agg["weighted"]["cagr_p50"] <= max(medians)
    assert jsonio.dumps(r)


@pytest.fixture()
def two_regimes(tmp_path, monkeypatch):
    # "hot" grows fastest in nominal terms but has 1%/month inflation; no scenarios.json -> equal probs
    _write_paths(tmp_path, ids=("calm", "hot"), drift=(0.006, 0.009), inflation=(0.002, 0.01))
    monkeypatch.setattr(service, "RESULTS_DIR", tmp_path)
    service.clear_cache()
    yield tmp_path
    service.clear_cache()


def test_worst_and_best_are_ranked_in_real_terms(two_regimes):
    r = service.evaluate_portfolio({"us_equity": 1})
    nominal = {e["id"]: e["cagr_p50"] for e in r["per_scenario"]}
    real = {e["id"]: e["real_cagr_p50"] for e in r["per_scenario"]}
    assert max(nominal, key=nominal.get) == "hot" and min(real, key=real.get) == "hot"
    assert r["aggregate"]["best"]["id"] == "calm" and r["aggregate"]["worst"]["id"] == "hot"
    assert [e["prob"] for e in r["per_scenario"]] == pytest.approx([0.5, 0.5])


def test_weighted_aggregate_with_one_scenario_equals_that_scenario(results):
    r = service.evaluate_portfolio({"us_equity": 1}, scenario_probs={"calm": 1, "storm": 0, "boom": 0})
    calm = r["per_scenario"][0]
    agg = r["aggregate"]["weighted"]
    for k in ("cagr_p5", "cagr_p50", "real_cagr_p50", "vol", "sharpe", "max_dd_p95", "cvar5_terminal"):
        assert agg[k] == pytest.approx(calm[k])
    for band in ("p5", "p50", "p95"):
        assert agg["fan"][band] == pytest.approx(calm["fan"][band])
    # worst / best only consider scenarios that can happen
    assert r["aggregate"]["worst"]["id"] == r["aggregate"]["best"]["id"] == "calm"


def test_probability_overrides_are_partial_and_renormalized(results):
    data = service.load_scenarios()
    assert np.allclose(service.scenario_probabilities(data, {"calm": 2}), np.array([2, 0.3, 0.2]) / 2.5)
    assert np.allclose(service.scenario_probabilities(data, None), [0.5, 0.3, 0.2])
    for bad in ({"nope": 1}, {"calm": -1}, {"calm": 0, "storm": 0, "boom": 0}, {"calm": "x"}):
        with pytest.raises(ValueError):
            service.scenario_probabilities(data, bad)


def test_custom_scenarios_mix_by_probability(results):
    r = service.evaluate_portfolio({"us_equity": 1}, include_custom=["c1", "c1"])
    ids = [e["id"] for e in r["per_scenario"]]
    assert ids == ["calm", "storm", "boom", "c1"]
    c1 = r["per_scenario"][-1]
    assert c1["source"] == "custom" and c1["n_paths"] == 30 and c1["name"] == "自定义崩盘"
    assert c1["prob"] == pytest.approx(0.1 / 1.1)
    data = service.load_scenarios(["c1"])
    w = metrics.mixture_weights(data.groups, service.scenario_probabilities(data))
    assert w[data.groups == 3].sum() == pytest.approx(0.1 / 1.1)      # 30 paths, still its full prob
    for bad in (["../c1"], ["missing"]):
        with pytest.raises(ValueError):
            service.evaluate_portfolio({"us_equity": 1}, include_custom=bad)


def test_input_validation(results):
    bad_weights = [{}, {"bitcoin": 1}, {"us_equity": -0.1, "cash": 1}, {"us_equity": 0, "cash": 0},
                   {"us_equity": "lots"}, {"us_equity": float("nan")}, [("us_equity", 1)]]
    for w in bad_weights:
        with pytest.raises(ValueError):
            service.evaluate_portfolio(w)
    with pytest.raises(ValueError):
        service.evaluate_portfolio({"us_equity": 1}, rebalance="weekly")
    for kwargs in ({"rate": -0.01}, {"rate": 2}, {"initial": 0}, {"horizon_years": 0}, {"strategy": "yolo"},
                   {"inflation_indexed": "maybe"}, {"inflation_indexed": 2}):
        with pytest.raises(ValueError):
            service.evaluate_withdrawal({"us_equity": 1}, **kwargs)
    # horizons: whole years in 10..40, rejected with a Chinese message before any path is read
    (results / service.CONTINUATION_FILE).unlink()
    service.clear_cache()
    for bad in (0, 9, 41, 30.5, -30, "thirty", True, None, float("inf")):
        with pytest.raises(ValueError) as err:
            service.evaluate_withdrawal({"us_equity": 1}, horizon_years=bad)
        assert not isinstance(err.value, service.PathsNotReady) and "期限" in str(err.value)
    assert "10 到 40" in str(pytest.raises(ValueError, service.parse_horizon, 41).value)
    assert service.parse_horizon(30.0) == 30 and service.parse_horizon("40") == 40


def test_huge_inputs_normalize_and_flags_are_strict(results):
    # a finite but overflowing sum must not silently zero the weights or the probabilities
    assert service.evaluate_portfolio({"us_equity": 1e308, "cash": 1e308})["weights"] == {"us_equity": 0.5, "cash": 0.5}
    data = service.load_scenarios()
    assert np.allclose(service.scenario_probabilities(data, {"calm": 1e308, "storm": 1e308, "boom": 0}), [0.5, 0.5, 0])
    kw = {"strategy": "pro_rata", "horizon_years": 10}
    for value, expected in (("false", False), ("true", True), (0, False), (None, True), (False, False)):
        assert service.evaluate_withdrawal({"us_equity": 1}, inflation_indexed=value, **kw)["inflation_indexed"] is expected


def test_withdrawal_ties_are_broken_by_real_terminal_wealth(results):
    s = service.evaluate_withdrawal({"us_equity": 1}, rate=0.0, strategy="pro_rata",
                                    horizon_years=10)["strategies"]["pro_rata"]
    agg = s["aggregate"]
    term = {e["id"]: e["terminal_real_p50"] for e in s["per_scenario"]}
    assert agg["ruin_prob_worst"] == agg["ruin_prob_best"] == 0.0
    assert agg["best_id"] == max(term, key=term.get) == "boom"
    assert agg["worst_id"] == min(term, key=term.get) == "storm"


def test_custom_scenario_notes(results):
    _write_paths(results, seed=3, n=20, ids=("old",), drift=(0.0,), custom="old", start_month=2)   # starts 2026-03
    r = service.evaluate_portfolio({"us_equity": 1}, include_custom=["c1", "old"])
    assert any("自定义崩盘" in n and "10.0%" in n for n in r["notes"])          # c1's prior_prob 0.1
    assert any("old" in n and "2026-03" in n and "2026-09" in n for n in r["notes"])
    assert not any("2026-03" in n for n in service.evaluate_portfolio({"us_equity": 1}, include_custom=["c1"])["notes"])
    w = service.evaluate_withdrawal({"us_equity": 1}, include_custom=["old"], strategy="pro_rata")
    assert any("2026-03" in n for n in w["notes"])


def test_missing_paths_raise_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "RESULTS_DIR", tmp_path)
    with pytest.raises(service.PathsNotReady):
        service.evaluate_portfolio({"us_equity": 1})
    assert service.warm(tmp_path) is False


def test_cache_reloads_when_the_file_changes(results):
    a = service.evaluate_portfolio({"us_equity": 1})["per_scenario"][0]["cagr_p50"]
    assert service.load_scenarios() is service.load_scenarios()          # cached
    path = results / "scenario_paths.npz"
    _write_paths(results, seed=5, drift=(0.02, 0.0, 0.0))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    b = service.evaluate_portfolio({"us_equity": 1})["per_scenario"][0]["cagr_p50"]
    assert b > a + 0.05


def test_evaluate_withdrawal_contract_shape(results):
    for horizon in (10, 30):
        r = service.evaluate_withdrawal({"us_equity": 0.6, "treasury_10y": 0.4}, initial=1000, rate=0.05,
                                        horizon_years=horizon, include_custom=["c1"])
        assert r["horizon_years"] == horizon and len(r["years"]) == horizon + 1 and len(r["dates"]) == horizon + 1
        assert r["scenario_years"] == 10 and r["horizon_bounds"] == [10, 40]
        assert r["dates"][10] == "2036-09" and r["dates"][-1] == f"{2026 + horizon}-09"
        assert set(r["strategies"]) == {"pro_rata", "div_then_pro_rata", "div_then_winners"}
        for name, s in r["strategies"].items():
            assert len(s["per_scenario"]) == 4
            for e in s["per_scenario"]:
                assert {"id", "name", "prob", "ruin_prob", "survival", "terminal_real_p5", "terminal_real_p50",
                        "terminal_real_p95", "terminal_nominal_p50", "depletion_year_median",
                        "wealth_fan_real"} <= set(e)
                assert len(e["survival"]) == horizon + 1 and len(e["wealth_fan_real"]["p50"]) == horizon + 1
                assert e["survival"][0] == 1.0 and np.all(np.diff(e["survival"]) <= 0)
                assert e["ruin_prob"] == pytest.approx(1 - e["survival"][-1])
                assert e["wealth_fan_real"]["p50"][0] == 1000.0
            agg = s["aggregate"]
            assert agg["ruin_prob_best"] <= agg["ruin_prob_weighted"] <= agg["ruin_prob_worst"]
            probs = np.array([e["prob"] for e in s["per_scenario"]])
            surv = np.array([e["survival"] for e in s["per_scenario"]])
            assert np.allclose(agg["survival_weighted"], probs @ surv)
            safe = r["safe_rate"][name]
            assert set(safe["per_scenario"]) == {"calm", "storm", "boom", "c1"}
            assert 0.0 <= safe["weighted"] <= 0.2
            curve = r["ruin_curve"][name]["weighted"]
            assert curve[0] == 0.0 and np.all(np.diff(curve) >= 0)
        assert jsonio.dumps(r)
        if horizon == 10:                                   # all inside the scenario paths
            assert r["scenario_horizon"] is None and r["continuation"] is None
            assert not any("续抽" in n for n in r["notes"])
            continue
        c = r["continuation"]
        assert c["label"] == "第 11 年起没有情景主题：按历史月份无条件续抽" and c["first_year"] == 11
        assert c["years"] == 20 and c["start_date"] == "2036-09" and c["n_paths"] == N and c["shared"] is True
        assert any(c["label"] in n for n in r["notes"])
        part = r["scenario_horizon"]
        assert part["years"] == 10 and part["date"] == "2036-09"
        for name, s in r["strategies"].items():
            ten = part["strategies"][name]
            assert set(ten["per_scenario"]) == {"calm", "storm", "boom", "c1"}
            for e in s["per_scenario"]:
                # the 10-year figures are the same runs cut at year 10
                assert ten["per_scenario"][e["id"]]["ruin_prob"] == pytest.approx(1 - e["survival"][10])
                assert ten["per_scenario"][e["id"]]["ruin_prob"] <= e["ruin_prob"]
            assert ten["aggregate"]["ruin_prob_weighted"] == pytest.approx(1 - s["aggregate"]["survival_weighted"][10])
            assert part["safe_rate"][name]["weighted"] >= r["safe_rate"][name]["weighted"]
            assert np.all(np.array(part["ruin_curve"][name]["weighted"]) <= np.array(r["ruin_curve"][name]["weighted"]))
    one = service.evaluate_withdrawal({"us_equity": 1}, strategy="div_then_winners", rate=0.0)
    assert list(one["strategies"]) == ["div_then_winners"] and one["horizon_years"] == service.DEFAULT_HORIZON
    assert one["strategies"]["div_then_winners"]["aggregate"]["ruin_prob_weighted"] == 0.0


def _report_json(r: dict) -> str:
    return jsonio.dumps({k: v for k, v in r.items() if k != "notes"})


def test_continuation_leaves_the_scenario_years_bit_identical(results):
    weights = {"us_equity": 0.5, "treasury_10y": 0.3, "gold": 0.2}
    data = service.load_scenarios(["c1"])
    assert data.continuation is not None and data.max_years == 40
    f40 = data.factors_for(40)
    for k in ("total", "price", "income", "order"):
        assert np.array_equal(getattr(f40, k)[:, :10], getattr(data.factors, k))
    assert np.array_equal(f40.cpi[:, :11], data.factors.cpi)
    # the continuation is appended by path number: row k of every scenario goes on with path k mod N
    cont = data.continuation.factors
    rows = data.continuation_rows()
    assert rows.tolist() == list(range(N)) * 3 + list(range(30))
    assert np.array_equal(f40.total[:, 10:], cont.total[rows])
    assert np.allclose(f40.cpi[:, 11:] / f40.cpi[:, 10:11], cont.cpi[rows, 1:], rtol=1e-13)
    # a 10-year horizon is bit-identical with and without the continuation file, and so is year 10 of a 30-year run
    with_cont = service.evaluate_withdrawal(weights, include_custom=["c1"], horizon_years=10)
    long = service.evaluate_withdrawal(weights, include_custom=["c1"], horizon_years=30)
    portfolio_with = service.evaluate_portfolio(weights, include_custom=["c1"])
    (results / service.CONTINUATION_FILE).unlink()
    service.clear_cache()
    without = service.evaluate_withdrawal(weights, include_custom=["c1"], horizon_years=10)
    assert _report_json(with_cont) == _report_json(without)
    for s in service.wd.STRATEGIES:
        for a, b in zip(long["strategies"][s]["per_scenario"], without["strategies"][s]["per_scenario"]):
            assert a["survival"][:11] == b["survival"]
            assert [x[:11] for x in a["wealth_fan_real"].values()] == list(b["wealth_fan_real"].values())
            assert long["scenario_horizon"]["strategies"][s]["per_scenario"][a["id"]]["terminal_real_p50"] == b["terminal_real_p50"]
        assert long["scenario_horizon"]["safe_rate"][s] == without["safe_rate"][s]
        assert long["scenario_horizon"]["ruin_curve"][s] == without["ruin_curve"][s]
    # the portfolio page is untouched by the continuation
    assert _report_json(portfolio_with) == _report_json(service.evaluate_portfolio(weights, include_custom=["c1"]))


def test_missing_or_stale_continuation_is_a_clear_503(results):
    path = results / service.CONTINUATION_FILE
    path.unlink()
    service.clear_cache()
    assert service.evaluate_withdrawal({"us_equity": 1}, horizon_years=10)["horizon_years"] == 10
    with pytest.raises(service.PathsNotReady) as err:
        service.evaluate_withdrawal({"us_equity": 1}, horizon_years=30)
    assert service.CONTINUATION_FILE in str(err.value) and "--only portfolio" in str(err.value)
    for kwargs, why in (({"start": "2026-10"}, "2026-10"), ({"offset": 180}, "180"), ({"years": 20}, "30 年")):
        _write_continuation(results, **{"years": CONT_YEARS, **kwargs})
        service.clear_cache()
        horizon = 40 if "years" in kwargs else 30
        with pytest.raises(service.PathsNotReady) as err:
            service.evaluate_withdrawal({"us_equity": 1}, horizon_years=horizon)
        assert why in str(err.value) and "--only portfolio" in str(err.value)
    path.write_bytes(b"not a zip")                          # half-written: horizons within the paths still work
    service.clear_cache()
    assert service.evaluate_withdrawal({"us_equity": 1}, horizon_years=10)["horizon_years"] == 10
    with pytest.raises(service.PathsNotReady):
        service.evaluate_withdrawal({"us_equity": 1}, horizon_years=11)


def test_one_continuation_path_per_row_and_the_old_shared_format_is_refused(results):
    weights = {"us_equity": 0.6, "treasury_10y": 0.4}
    # as many continuation paths as built-in rows (what the build writes): every built-in row has its own
    _write_continuation(results, n=S * N)
    service.clear_cache()
    data = service.load_scenarios(["c1"])
    rows = data.continuation_rows()
    assert rows.tolist() == list(range(S * N)) + list(range(30))       # the custom rows start over
    assert data.continuation_shared() == ["自定义崩盘"]
    f = data.factors_for(30)
    assert np.array_equal(f.total[:, 10:], data.continuation.factors.total[rows, :20])
    builtin = service.evaluate_withdrawal(weights, horizon_years=30)
    assert builtin["continuation"]["shared"] is False and builtin["continuation"]["n_paths"] == S * N
    assert builtin["continuation"]["rows"] == S * N and builtin["continuation"]["shared_scenarios"] == []
    assert any("独立的续抽路径" in n for n in builtin["notes"]) and not any("循环" in n for n in builtin["notes"])
    mixed = service.evaluate_withdrawal(weights, horizon_years=30, include_custom=["c1"])
    assert mixed["continuation"]["shared"] is True and mixed["continuation"]["shared_scenarios"] == ["自定义崩盘"]
    assert any("循环" in n and "自定义崩盘" in n for n in mixed["notes"])
    # the scenario years are untouched by which continuation follows them
    assert builtin["scenario_horizon"]["safe_rate"] == service.evaluate_withdrawal(weights, horizon_years=10)["safe_rate"]
    # the earlier monthly format (one set of paths shared by every scenario): rebuild, until then a 503
    path = results / service.CONTINUATION_FILE
    rng = np.random.default_rng(3)
    np.savez_compressed(path, total=rng.normal(0.004, 0.03, (N, 360, len(ASSETS))).astype(np.float32),
                        income=np.full((N, 360, len(ASSETS)), 0.002, np.float32),
                        cpi=np.cumprod(np.full((N, 361), 1.002), axis=1), assets=np.array(ASSETS),
                        labels=np.array([service._shift_label("2036-09", k) for k in range(361)]),
                        start=np.array("2026-09"), offset=np.array(T), info=np.array("{}"))
    service.clear_cache()
    assert service.evaluate_withdrawal(weights, horizon_years=10)["horizon_years"] == 10
    with pytest.raises(service.PathsNotReady) as err:
        service.evaluate_withdrawal(weights, horizon_years=30)
    assert "旧格式" in str(err.value) and "--only portfolio" in str(err.value)


def test_horizon_message_shows_the_value_received():
    assert service.parse_horizon(30) == 30 and service.parse_horizon("40") == 40 and service.parse_horizon(10.0) == 10
    for value, shown in ((41, "收到 41）"), (25.5, "收到 25.5）"), (30.0000001, "收到 30.0000001）"), (-3, "收到 -3）")):
        with pytest.raises(ValueError) as err:
            service.parse_horizon(value)
        assert shown in str(err.value) and "10 到 40" in str(err.value)


def test_ruin_monotone_in_rate_and_horizon_and_safe_rate_falls(results):
    weights = {"us_equity": 0.6, "treasury_10y": 0.4}
    horizons = (10, 15, 30, 40)
    rates = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    runs = {(H, r): service.evaluate_withdrawal(weights, rate=r, horizon_years=H) for H in horizons for r in rates}
    for s in service.wd.STRATEGIES:
        ruin = np.array([[runs[H, r]["strategies"][s]["aggregate"]["ruin_prob_weighted"] for r in rates] for H in horizons])
        assert np.all(np.diff(ruin, axis=1) >= 0) and np.all(np.diff(ruin, axis=0) >= 0)
        assert ruin[-1, -1] > ruin[0, 0]
        safe = [runs[H, 0.04]["safe_rate"][s]["weighted"] for H in horizons]
        assert all(a >= b for a, b in zip(safe, safe[1:])) and safe[-1] < safe[0]
        for sid in ("calm", "storm", "boom"):
            per = [runs[H, 0.04]["safe_rate"][s]["per_scenario"][sid] for H in horizons]
            assert all(a >= b for a, b in zip(per, per[1:]))


def test_api_accepts_30_and_40_and_rejects_41_and_0_in_chinese(results):
    from fastapi.testclient import TestClient

    from scenariolab.server import app
    client = TestClient(app, raise_server_exceptions=False)
    body = {"weights": {"us_equity": 0.6, "treasury_10y": 0.4}, "rate": 0.04, "strategy": "pro_rata"}
    for horizon in (30, 40):
        r = client.post("/api/withdrawal/evaluate", json={**body, "horizon_years": horizon})
        assert r.status_code == 200, r.text[:300]
        assert r.json()["horizon_years"] == horizon and len(r.json()["dates"]) == horizon + 1
    for horizon in (41, 0, 9, 25.5, "abc"):
        r = client.post("/api/withdrawal/evaluate", json={**body, "horizon_years": horizon})
        assert r.status_code == 400, r.text[:300]
        detail = r.json()["detail"]
        assert "期限" in detail and ("10 到 40" in detail or "必须是数字" in detail)
    (results / service.CONTINUATION_FILE).unlink()
    service.clear_cache()
    r = client.post("/api/withdrawal/evaluate", json={**body, "horizon_years": 30})
    assert r.status_code == 503 and "--only portfolio" in r.json()["detail"]


def test_run_writes_default_file(results):
    out = service.run(None)
    assert (results / service.OUTPUT_FILE).exists()
    assert len(out["presets"]) == len(service.PRESETS)
    assert out["portfolio"]["weights"] == service.DEFAULT_WEIGHTS
    # no local absolute path in the result file; a non-official path set is flagged
    assert out["source"]["paths_file"] == "scenario_paths.npz" and out["source"]["official_paths"] is False
    assert out["source"]["continuation_file"] == service.CONTINUATION_FILE
    assert out["withdrawal"]["horizon_years"] == service.DEFAULT_HORIZON
    assert out["continuation"]["label"].startswith("第 11 年起") and out["continuation"]["max_years"] == 40
    for p in out["presets"]:
        assert p["withdrawal_horizon_years"] == 10 and set(p["withdrawal_by_horizon"]) == {"10", "30"}
        assert p["ruin_prob_weighted"] == p["withdrawal_by_horizon"]["10"]["ruin_prob_weighted"]
        for s, rate in p["withdrawal_by_horizon"]["30"]["safe_rate_weighted"].items():
            assert rate <= p["safe_rate_weighted"][s]


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
def test_build_continuation_is_the_unconditional_engine_run_deterministic_by_seed(tmp_path):
    from scenariolab.context import EngineContext
    from scenariolab.engine.pooling import BLOCK, pool_phase
    from scenariolab.engine.simulate import capital_market_assumptions, simulate

    ctx = EngineContext.load(selection=True)
    a = service.build_continuation(ctx, offset=180, n_paths=40, seed=5)
    b = service.build_continuation(ctx, offset=180, n_paths=40, seed=5)
    c = service.build_continuation(ctx, offset=180, n_paths=40, seed=6)
    for k in ("total", "income", "cpi"):
        assert np.array_equal(a[k], b[k]) and not np.array_equal(a[k], c[k])
    assert a["total"].shape == (40, 300, len(ASSETS)) and a["cpi"].shape == (40, 301)
    assert a["info"]["cma"] == "valuation_state" and a["start"] == ctx.anchor.date.strftime("%Y-%m")
    # months 181..480 of one view-free 40-year simulate() from today's anchor, valuation_state drift
    pool = pool_phase(ctx.db.frame, [], ctx.db.prior(), 480, BLOCK, rare=ctx.db.rare)
    sim = simulate(ctx.db, ctx.anchor, [pool.blocks], [480], n_paths=40, seed=5, block=BLOCK,
                   cma=capital_market_assumptions(ctx.db, ctx.anchor, method="valuation_state"), **config.RESIDUALS)
    assert np.array_equal(a["total"], sim.total[:, 180:]) and np.array_equal(a["income"], sim.income[:, 180:])
    # the 15-year reversion term is the only part of the scenarios' drift that is dropped
    assert a["info"]["market_adjustment"] == pytest.approx(ctx.cma["market_adjustment"] - ctx.cma["reversion_term"])
    path = service.write_continuation(tmp_path / service.CONTINUATION_FILE, **a)
    loaded = service._load_continuation(path)
    assert loaded.offset == 180 and loaded.years == 25 and loaded.n_paths == 40 and loaded.info == json.loads(jsonio.dumps(a["info"]))
    # the file keeps the yearly factors (float32) of exactly these months
    ref = service.wd.year_factors(a["total"], a["income"], a["cpi"])
    for k in ("total", "price", "income"):
        assert np.array_equal(getattr(loaded.factors, k), getattr(ref, k).astype(np.float32).astype(np.float64)), k
    assert np.array_equal(loaded.factors.cpi, ref.cpi) and loaded.labels == a["labels"][::12]


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
def test_build_continuation_runs_in_seeded_chunks_one_path_per_row(tmp_path):
    from scenariolab.context import EngineContext
    from scenariolab.engine.pooling import BLOCK, pool_phase
    from scenariolab.engine.simulate import capital_market_assumptions, simulate

    ctx = EngineContext.load(selection=True)
    a = service.build_continuation(ctx, offset=180, n_paths=50, seed=5, chunk=20)       # runs of 20, 20, 10
    b = service.build_continuation(ctx, offset=180, n_paths=50, seed=5, chunk=20)
    assert a["total"].shape == (50, 300, len(ASSETS)) and a["cpi"].shape == (50, 301)
    assert all(np.array_equal(a[k], b[k]) for k in ("total", "income", "cpi"))
    seeds = [service.continuation_chunk_seed(5, c) for c in range(3)]
    assert seeds[0] == 5 and len(set(seeds)) == 3 and a["info"]["chunk_seeds"] == seeds
    assert a["info"]["n_paths"] == 50 and a["info"]["chunk_paths"] == 20
    pool = pool_phase(ctx.db.frame, [], ctx.db.prior(), 480, BLOCK, rare=ctx.db.rare)
    cma = capital_market_assumptions(ctx.db, ctx.anchor, method="valuation_state")
    for c, (lo, n) in enumerate(((0, 20), (20, 20), (40, 10))):
        sim = simulate(ctx.db, ctx.anchor, [pool.blocks], [480], n_paths=n, seed=seeds[c], block=BLOCK, cma=cma,
                       **config.RESIDUALS)
        assert np.array_equal(a["total"][lo:lo + n], sim.total[:, 180:]), c
        assert np.array_equal(a["cpi"][lo:lo + n], sim.macro["cpi"][:, 180:].astype(np.float64)), c
    # chunks are independent draws, not copies of each other
    assert not np.array_equal(a["total"][:10], a["total"][20:30])
    # the build's memory-light path (yearly factors per run) gives the same factors, info and file
    y = service.build_continuation(ctx, offset=180, n_paths=50, seed=5, chunk=20, yearly=True)
    ref = service.wd.year_factors(a["total"], a["income"], a["cpi"])
    assert "total" not in y and y["info"] == a["info"] and y["labels"] == a["labels"]
    for k in ("total", "price", "income", "cpi", "order"):
        assert np.array_equal(getattr(y["factors"], k), getattr(ref, k)), k
    p1 = service.write_continuation(tmp_path / "monthly.npz", **a)
    p2 = service.write_continuation(tmp_path / "yearly.npz", **y)
    with np.load(p1) as z1, np.load(p2) as z2:
        assert sorted(z1.files) == sorted(z2.files) and all(np.array_equal(z1[k], z2[k]) for k in z1.files)
    assert service._continuation_from(y).factors.total.shape == (50, 25, len(ASSETS))


@pytest.mark.skipif(not (config.RESULTS_DIR / "scenario_paths.npz").exists(), reason="scenario paths not built")
def test_real_paths_latency():
    service.clear_cache()
    service.load_scenarios()                                            # load once (cached afterwards)
    weights = {"us_equity": 0.5, "treasury_10y": 0.3, "gold": 0.1, "cash": 0.1}
    t = time.time()
    p = service.evaluate_portfolio(weights, rebalance="annual")
    t_portfolio = time.time() - t
    t = time.time()
    w = service.evaluate_withdrawal(weights, horizon_years=15)
    t_withdrawal = time.time() - t
    assert t_portfolio < 2.0 and t_withdrawal < 2.0
    assert -0.2 < p["aggregate"]["weighted"]["cagr_p50"] < 0.2
    assert 0.0 <= w["strategies"]["pro_rata"]["aggregate"]["ruin_prob_weighted"] <= 1.0
    if service.load_scenarios().continuation is None:
        pytest.skip("continuation not built (python -m scenariolab.build --only portfolio)")
    t = time.time()                              # first 30-year request: the continuation is chained on, and
    service.evaluate_withdrawal(weights, horizon_years=30)      # the critical rates at 30 and 15 years computed
    assert time.time() - t < 8.0
    t = time.time()
    service.evaluate_withdrawal(weights, rate=0.035, horizon_years=30)      # another rate: cached critical rates
    assert time.time() - t < 2.0


needs_continuation = pytest.mark.skipif(
    not (config.RESULTS_DIR / service.CONTINUATION_FILE).exists() or not (config.RESULTS_DIR / "scenario_paths.npz").exists(),
    reason="scenario paths or their continuation not built")


@needs_continuation
def test_real_paths_first_180_months_bit_identical():
    service.clear_cache()
    data = service.load_scenarios()
    assert data.continuation is not None, data.continuation_issue
    assert data.months == config.HORIZON_MONTHS and data.max_years == service.HORIZON_BOUNDS[1]
    with np.load(config.RESULTS_DIR / "scenario_paths.npz") as z:
        S, N_, T_, A = z["total"].shape
        raw = service.wd.year_factors(z["total"].reshape(S * N_, T_, A), z["income"].reshape(S * N_, T_, A),
                                      z["cpi"].reshape(S * N_, T_ + 1).astype(np.float64))
    f = data.factors_for(40)
    for k in ("total", "price", "income", "cpi", "order"):
        assert np.array_equal(getattr(f, k)[:, :15 + (k == "cpi")], getattr(raw, k)), k
    # one continuation path per built-in path: row r goes on with path r, none is shared
    assert data.continuation.n_paths == S * N_ and np.array_equal(data.continuation_rows(), np.arange(S * N_))
    assert data.continuation_shared() == []
    assert data.continuation.offset == T_ and data.continuation.start == data.labels[0]


@needs_continuation
def test_real_paths_ruin_monotone_and_30_year_safe_rate_below_15():
    service.clear_cache()
    weights = {"us_equity": 0.6, "treasury_10y": 0.4}
    rates = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    horizons = (15, 30, 40)
    runs = {(H, r): service.evaluate_withdrawal(weights, rate=r, horizon_years=H) for H in horizons for r in rates}
    for s in service.wd.STRATEGIES:
        ruin = np.array([[runs[H, r]["strategies"][s]["aggregate"]["ruin_prob_weighted"] for r in rates] for H in horizons])
        assert np.all(np.diff(ruin, axis=1) >= 0) and np.all(np.diff(ruin, axis=0) >= 0), (s, ruin)
        safe = {H: runs[H, 0.04]["safe_rate"][s]["weighted"] for H in horizons}
        assert safe[40] <= safe[30] <= safe[15] and safe[30] < safe[15], (s, safe)
        r30 = runs[30, 0.04]
        part = r30["scenario_horizon"]
        assert part["years"] == 15 and part["safe_rate"][s] == runs[15, 0.04]["safe_rate"][s]
        assert part["strategies"][s]["aggregate"]["ruin_prob_weighted"] == pytest.approx(
            runs[15, 0.04]["strategies"][s]["aggregate"]["ruin_prob_weighted"], abs=1e-12)
        for sid, rate in r30["safe_rate"][s]["per_scenario"].items():
            assert rate <= part["safe_rate"][s]["per_scenario"][sid]
    assert runs[30, 0.04]["continuation"]["label"] == "第 16 年起没有情景主题：按历史月份无条件续抽"


# ---------------------------------------------------------------- tickers (INSTRUMENTS_DESIGN §6.1, §6.4, §7.3, T17)

from scenariolab.instruments import ESTIMATED, TICKERS  # noqa: E402
from scenariolab.portfolio import tickers as tk  # noqa: E402

K = len(TICKERS)
PAYERS = {"MAIN": 0.007, "QQQI": 0.012, "TLT": 0.004, "SCHD": 0.003}       # monthly cash per start-of-month value


def _write_tickers(root, seed=1, ids=("calm", "storm", "boom"), n=N, custom=None, fingerprint=True, alpha_shift=-0.02):
    """Synthetic ticker paths in instrument_paths.npz's layout for the paths _write_paths wrote (same ids and rows):
    random totals, the payers' cash, a TQQQ wipe-out on path 0, and alpha variants whose yearly factors are the
    default ones shifted by alpha_shift (half) and -alpha_shift (raw) a year."""
    rng = np.random.default_rng(seed)
    S_ = len(ids)
    total = rng.normal(0.006, 0.05, size=(S_, n, T, K)).astype(np.float32)
    income = np.zeros_like(total)
    for t, x in PAYERS.items():
        income[..., TICKERS.index(t)] = x
    j = TICKERS.index("TQQQ")
    total[:, 0, 5, j], total[:, 0, 6:, j] = -1.0, 0.0
    alive = np.ones((S_, n, T, 2), dtype=bool)
    alive[:, 0, 5:, 1] = False
    est = [TICKERS.index(t) for t in ESTIMATED]
    f = service.wd.year_factors(total.reshape(-1, T, K), income.reshape(-1, T, K), np.ones((S_ * n, T + 1)))
    arrays = {}
    for mode, shift in (("half", alpha_shift), ("raw", -alpha_shift)):
        g = np.exp(shift)
        for k, a in (("total", f.total * g), ("price", f.price * g), ("income", f.income)):
            arrays[f"alpha_{mode}_{k}"] = a[:, :, est].reshape(S_, n, T // 12, len(est)).astype(np.float32)
    target = root if custom is None else root / "custom"
    name = tk.PATHS_FILE if custom is None else f"{custom}{tk.CUSTOM_SUFFIX}"
    with np.load(target / ("scenario_paths.npz" if custom is None else f"{custom}.npz")) as z:
        fp = tk.paths_fingerprint([str(x) for x in z["ids"]], np.asarray(z["total"])[:, :2])
        labels = z["labels"]
    if custom is not None and fingerprint:
        arrays["paths_fingerprint"] = np.array(fp)
    np.savez_compressed(target / name, tickers=np.array(TICKERS), ids=np.array(ids), probs=np.full(S_, 1 / S_),
                        labels=labels, start=np.linspace(10.0, 100.0, K), total=total, income=income, alive=alive,
                        estimated=np.array(ESTIMATED), leveraged=np.array(["QLD", "TQQQ"]), **arrays)
    if custom is None:
        meta = {"scenario_ids": list(ids), "n_paths": n, "months": T, "anchor_month": "2026-09",
                "built_at": "2026-11-02 00:00:00", "flags": {"MAIN": ["estimated", "credit_link", "tax"], "QQQ": ["tax"],
                                                             "TQQQ": ["ruin_extrap", "tax"]},
                "start": {t: {"price": float(p), "asof": "2026-09-30"} for t, p in zip(TICKERS, np.linspace(10, 100, K))}}
        if fingerprint:
            meta["paths_fingerprint"] = fp
        jsonio.dump(meta, root / tk.META_FILE)
    return total, income


def _write_ticker_continuation(root, seed=31, fingerprint=None):
    """A ticker continuation (tickers.write_continuation's format) for the asset continuation _write_continuation
    wrote: one row per continuation path, its asset fingerprint unless `fingerprint` is given."""
    cont = service._load_continuation(root / service.CONTINUATION_FILE)
    R, Yc = cont.n_paths, cont.years
    rng = np.random.default_rng(seed)
    tot = np.exp(rng.normal(0.06, 0.15, size=(R, Yc, K)))
    inc = np.zeros_like(tot)
    for t, x in PAYERS.items():
        inc[..., TICKERS.index(t)] = 12 * x
    est = [TICKERS.index(t) for t in ESTIMATED]
    alpha = {m: (tot[:, :, est] * np.exp(s), (tot - inc)[:, :, est] * np.exp(s), inc[:, :, est])
             for m, s in (("half", -0.02), ("raw", 0.02))}
    info = {"asset_fingerprint": fingerprint or cont.fingerprint(), "rows": R, "years": Yc}
    tk.write_continuation(root / tk.CONTINUATION_FILE, TICKERS, ESTIMATED,
                          {"total": tot, "price": tot - inc, "income": inc}, alpha, cont.factors.cpi, cont.labels,
                          cont.start, cont.offset, info)
    return tot, inc


@pytest.fixture()
def with_tickers(results):
    _write_tickers(results)
    _write_ticker_continuation(results)
    service.clear_cache()
    yield results
    service.clear_cache()


INCOME_MIX = {"us_equity": 0.4, "MAIN": 0.3, "tlt": 0.2, "GOLD": 0.1}          # tlt: case-insensitive; GOLD: the asset


def test_weights_parse_tickers_aliases_and_unknown_ids():
    assets = list(ASSETS)
    wa, held = service.parse_book_weights({"us_equity": 2, "brk.b": 1, "GOLD": 1, "Main": 0}, assets)
    assert held == {"BRK.B": 1.0, "MAIN": 0.0} and wa[assets.index("gold")] == 1.0 and wa.sum() == 3.0
    for bad, needle in (({"bitcoin": 1}, "未知资产或标的：bitcoin"), ({"QQQ": -1}, "标的 QQQ 的权重不能为负"),
                        ({"QQQ": "x"}, "必须是数字"), ([], "非空对象")):
        with pytest.raises(ValueError) as err:
            service.parse_book_weights(bad, assets)
        assert needle in str(err.value)
    with pytest.raises(ValueError):
        service.parse_weights({"QQQ": 1}, assets)            # the asset-class parser refuses a held ticker


def test_asset_only_results_ignore_the_ticker_files(results):
    """Without a held ticker nothing of the instrument layer is read: the same report with or without its files,
    and a ticker at weight 0 or the GOLD alias change nothing."""
    weights = {"us_equity": 0.5, "treasury_10y": 0.3, "gold": 0.2}
    before = (_report_json(service.evaluate_portfolio(weights)),
              _report_json(service.evaluate_withdrawal(weights, horizon_years=30, include_custom=["c1"])))
    _write_tickers(results)
    _write_ticker_continuation(results)
    service.clear_cache()
    alias = {"us_equity": 0.5, "treasury_10y": 0.3, "GOLD": 0.2, "QQQ": 0}
    assert _report_json(service.evaluate_portfolio(weights)) == before[0]
    assert _report_json(service.evaluate_portfolio(alias)) == before[0]
    assert _report_json(service.evaluate_withdrawal(alias, horizon_years=30, include_custom=["c1"])) == before[1]
    assert "instruments" not in service.evaluate_portfolio(weights)


def _manual_book(results, cols_assets, cols_tickers, s):
    """Scenario s's monthly total and income in the given columns, straight from the files."""
    with np.load(results / "scenario_paths.npz") as z, np.load(results / tk.PATHS_FILE) as zt:
        tot = np.concatenate([z["total"][s][:, :, [ASSETS.index(a) for a in cols_assets]],
                              zt["total"][s][:, :, [TICKERS.index(t) for t in cols_tickers]]], axis=2)
        inc = np.concatenate([z["income"][s][:, :, [ASSETS.index(a) for a in cols_assets]],
                              zt["income"][s][:, :, [TICKERS.index(t) for t in cols_tickers]]], axis=2)
        cpi = z["cpi"][s].astype(np.float64)
    return tot, inc, cpi


def test_portfolio_with_tickers_equals_the_hand_computed_mix(with_tickers):
    r = service.evaluate_portfolio(INCOME_MIX, rebalance="annual")
    assert r["weights"] == pytest.approx({"us_equity": 0.4, "gold": 0.1, "TLT": 0.2, "MAIN": 0.3})
    tot, _, cpi = _manual_book(with_tickers, ["us_equity", "gold"], ["TLT", "MAIN"], 1)
    w = np.array([0.4, 0.1, 0.2, 0.3])
    with np.load(with_tickers / "scenario_paths.npz") as z:
        rf = z["total"][1][:, :, ASSETS.index("cash")].astype(np.float64)
    m = metrics.path_metrics(metrics.portfolio_wealth(tot, w, "annual"), cpi, rf)
    storm = r["per_scenario"][1]
    assert storm["id"] == "storm"
    for k, v in metrics.summarize(m).items():
        assert storm[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k
    ins = r["instruments"]
    assert ins["tickers"] == ["TLT", "MAIN"] and ins["names"]["MAIN"].startswith("Main Street")
    assert ins["chips"]["credit_link"].startswith("与股市的联动") and ins["alpha_mode"] == "shrunk"
    assert any("Main Street" in n for n in r["notes"]) and any("与股市的联动低于历史" in n for n in r["notes"])
    # a leveraged fund wiped out on path 0 of every scenario
    lev = service.evaluate_portfolio({"TQQQ": 1})
    assert lev["instruments"]["wipeout_prob_scenario_end"]["TQQQ"] == pytest.approx(1 / N)
    assert lev["aggregate"]["weighted"]["max_dd_p95"] >= -1.0 and any("归零" in n for n in lev["notes"])


def test_withdrawal_uses_each_tickers_own_distributions(with_tickers):
    """The dividend-first pot is each ticker's own cash: the hand-assembled year factors give the same results."""
    weights = {"cash": 0.2, "MAIN": 0.5, "QQQI": 0.3}
    r = service.evaluate_withdrawal(weights, horizon_years=10, rate=0.05)
    pw = metrics.mixture_weights(service.load_scenarios().groups, [0.5, 0.3, 0.2])
    parts = [_manual_book(with_tickers, ["cash"], ["QQQI", "MAIN"], s) for s in range(3)]
    f = service.wd.year_factors(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                                np.concatenate([p[2] for p in parts]))
    w = np.array([0.2, 0.3, 0.5])
    for s in service.wd.STRATEGIES:
        res = service.wd.simulate(f, w, 1e6, 0.05, s, True, 10)
        assert r["strategies"][s]["aggregate"]["ruin_prob_weighted"] == pytest.approx(
            service.wd.summarize(res, f.cpi, pw)["ruin_prob"], abs=1e-12)
        crit = service.wd.critical_rates(f, w, s, True, 10)
        assert r["safe_rate"][s]["weighted"] == service.wd.safe_rate(crit, pw)
    # year 1's pot: MAIN pays 0.7% and QQQI 1.2% a month on their value
    share = r["cash_flow"]["pot_share_y1"]["columns"]
    need1 = 0.05 * f.cpi[:, 1]
    assert share["MAIN"] == pytest.approx(metrics.weighted_percentile(0.5 * f.income[:, 0, 2] / need1, 50, pw))
    assert share["QQQI"] > share["MAIN"] * 0.3 / 0.5 and "QQQI" in r["cash_flow"]["cut_pressure"]
    flow = r["cash_flow"]["strategies"]["div_then_pro_rata"]
    assert len(flow["coverage_fan"]["p50"]) == 10 and 0.0 <= flow["prob_pot_below_half"] <= 1.0
    assert set(flow["prob_pot_below_half_per_scenario"]) == {"calm", "storm", "boom"}
    # a ticker that pays nothing leaves the pot to the others: pro_rata is unchanged by the payout split
    assert any("粘性分配" in n for n in r["notes"]) and r["instruments"]["tickers"] == ["QQQI", "MAIN"]


def test_alpha_variants_change_only_the_estimated_tickers(with_tickers):
    for weights, moves in (({"MAIN": 0.5, "QQQ": 0.5}, True), ({"QQQ": 0.6, "TLT": 0.4}, False)):
        out = {m: service.evaluate_withdrawal(weights, horizon_years=30, rate=0.05, strategy="pro_rata", alpha_mode=m)
               for m in tk.ALPHA_MODES}
        ruin = {m: o["strategies"]["pro_rata"]["aggregate"]["ruin_prob_weighted"] for m, o in out.items()}
        safe = {m: o["safe_rate"]["pro_rata"]["weighted"] for m, o in out.items()}
        if moves:                                   # half: 2% a year less growth, raw: 2% more
            assert safe["half"] < safe["shrunk"] < safe["raw"] and ruin["half"] >= ruin["shrunk"] >= ruin["raw"]
            assert out["half"]["instruments"]["alpha_mode"] == "half"
        else:
            assert len({_report_json({k: v for k, v in o.items() if k != "instruments"}) for o in out.values()}) == 1
    with pytest.raises(ValueError) as err:
        service.evaluate_withdrawal({"MAIN": 1}, alpha_mode="double")
    assert "alpha_mode" in str(err.value)


def test_ticker_years_past_the_scenarios_come_from_the_ticker_continuation(with_tickers):
    data = service.load_scenarios(["c1"]) if (with_tickers / "custom" / f"c1{tk.CUSTOM_SUFFIX}").exists() else None
    assert data is None                             # c1 has no ticker file yet
    data = service.load_scenarios()
    book = service._book(data, {"us_equity": 0.5, "MAIN": 0.5})
    f40 = service._book_factors(data, book, 40)
    ts = data.tickers()
    head = ts.year_columns(["MAIN"])
    tail = ts.cont_columns(["MAIN"])
    rows = data.continuation_rows()
    assert np.array_equal(f40.total[:, :10, 1], head[0][:, :, 0])
    assert np.array_equal(f40.total[:, 10:, 1], tail[0][rows, :, 0])
    assert np.array_equal(f40.total[:, :, 0], data.factors_for(40).total[:, :, ASSETS.index("us_equity")])
    assert np.array_equal(f40.cpi, data.factors_for(40).cpi)
    assert set(np.unique(f40.order)) == {0, 1}            # the sale order of div_then_winners covers the ticker
    r = service.evaluate_withdrawal({"us_equity": 0.5, "MAIN": 0.5}, horizon_years=40, strategy="pro_rata")
    assert r["continuation"]["instruments"]["rows"] == N and any("估计型标的的 α 按路径固定" in n for n in r["notes"])
    # the 10-year figures are the same with or without the ticker continuation
    ten = _report_json(service.evaluate_withdrawal({"us_equity": 0.5, "MAIN": 0.5}, horizon_years=10))
    (with_tickers / tk.CONTINUATION_FILE).unlink()
    service.clear_cache()
    assert _report_json(service.evaluate_withdrawal({"us_equity": 0.5, "MAIN": 0.5}, horizon_years=10)) == ten
    with pytest.raises(service.PathsNotReady) as err:
        service.evaluate_withdrawal({"us_equity": 0.5, "MAIN": 0.5}, horizon_years=30)
    assert tk.CONTINUATION_FILE in str(err.value) and "--only instruments" in str(err.value)


def test_stale_or_missing_ticker_files_are_refused_with_the_build_command(results):
    weights = {"us_equity": 0.5, "QQQ": 0.5}
    with pytest.raises(service.PathsNotReady) as err:
        service.evaluate_portfolio(weights)
    assert "--only instruments" in str(err.value)
    _write_tickers(results)
    service.clear_cache()
    assert service.evaluate_portfolio(weights)["weights"] == {"us_equity": 0.5, "QQQ": 0.5}
    # the scenario paths are rebuilt (other numbers): the ticker paths no longer belong to them
    _write_paths(results, seed=11)
    service.clear_cache()
    with pytest.raises(service.PathsNotReady) as err:
        service.evaluate_portfolio(weights)
    assert "旧的情景路径" in str(err.value) and "--only instruments" in str(err.value)
    assert service.evaluate_portfolio({"us_equity": 1})["weights"] == {"us_equity": 1.0}     # assets still work
    # an asset continuation rebuilt from another engine state: the ticker continuation is refused, not mixed
    _write_tickers(results)
    _write_ticker_continuation(results, fingerprint="0" * 64)
    service.clear_cache()
    assert service.evaluate_withdrawal(weights, horizon_years=10)["horizon_years"] == 10
    with pytest.raises(service.PathsNotReady) as err:
        service.evaluate_withdrawal(weights, horizon_years=30)
    assert "另一组资产续抽路径" in str(err.value)


def test_custom_scenarios_need_their_own_ticker_file(with_tickers):
    weights = {"us_equity": 0.5, "MAIN": 0.5}
    with pytest.raises(ValueError) as err:
        service.evaluate_portfolio(weights, include_custom=["c1"])
    assert not isinstance(err.value, service.PathsNotReady) and "上线前保存" in str(err.value)
    assert "自定义崩盘" in str(err.value)
    _write_tickers(with_tickers, seed=4, ids=("c1",), n=30, custom="c1")
    service.clear_cache()
    r = service.evaluate_portfolio(weights, include_custom=["c1"])
    assert [e["id"] for e in r["per_scenario"]] == ["calm", "storm", "boom", "c1"]
    w = service.evaluate_withdrawal(weights, include_custom=["c1"], horizon_years=30, strategy="div_then_winners")
    assert "自定义崩盘" in w["continuation"]["shared_scenarios"] and w["instruments"]["tickers"] == ["MAIN"]
    # a custom ticker file run on other paths than the custom scenario's own is refused
    _write_paths(with_tickers, seed=12, n=30, ids=("c1",), drift=(0.0,), custom="c1")
    service.clear_cache()
    with pytest.raises(ValueError) as err:
        service.evaluate_portfolio(weights, include_custom=["c1"])
    assert "不一致" in str(err.value)


def test_api_tickers_in_portfolio_and_withdrawal_requests(with_tickers):
    from fastapi.testclient import TestClient

    from scenariolab.server import app
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/api/portfolio/evaluate", json={"weights": {"QQQ": 60, "TLT": 40}})
    assert r.status_code == 200, r.text[:300]
    assert r.json()["weights"] == {"QQQ": 0.6, "TLT": 0.4} and r.json()["instruments"]["tickers"] == ["QQQ", "TLT"]
    r = client.post("/api/withdrawal/evaluate", json={"weights": {"MAIN": 1}, "horizon_years": 30, "alpha_mode": "half",
                                                      "strategy": "div_then_pro_rata"})
    assert r.status_code == 200, r.text[:300]
    assert r.json()["instruments"]["alpha_mode"] == "half" and r.json()["cash_flow"]["pot_share_y1"]["columns"]["MAIN"] > 0
    for body, needle in (({"weights": {"XYZ": 1}}, "未知资产或标的：XYZ"),
                         ({"weights": {"MAIN": 1}, "alpha_mode": "x"}, "alpha_mode")):
        r = client.post("/api/withdrawal/evaluate", json=body)
        assert r.status_code == 400 and needle in r.json()["detail"], r.text[:300]
    (with_tickers / tk.PATHS_FILE).unlink()
    service.clear_cache()
    r = client.post("/api/portfolio/evaluate", json={"weights": {"QQQ": 1}})
    assert r.status_code == 503 and "--only instruments" in r.json()["detail"]
    assert client.post("/api/portfolio/evaluate", json={"weights": {"us_equity": 1}}).status_code == 200


def test_api_instruments_catalog_without_and_with_the_files(results, monkeypatch):
    from fastapi.testclient import TestClient

    from scenariolab.server import app
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/instruments")
    assert r.status_code == 200, r.text[:300]
    c = r.json()
    assert set(TICKERS) | {"GOLD"} <= set(c["tickers"]) and c["tickers"]["TQQQ"]["name"].startswith("三倍纳指")
    assert c["status"]["available"] is False and "--only instruments" in c["status"]["issue"]
    assert c["order"][:4] == ["QQQ", "QLD", "TQQQ", "QQQI"] and c["aliases"] == {"GOLD": "gold"}
    assert [m["id"] for m in c["alpha_modes"]] == list(tk.ALPHA_MODES) and c["presets"][0]["id"] == "income_tickers"
    assert "{fee}" not in c["tickers"]["TQQQ"]["card"] and c["tickers"]["TQQQ"]["fee"] > 0
    _write_tickers(results)
    service.clear_cache()
    c = client.get("/api/instruments").json()
    assert c["status"]["available"] is True and c["tickers"]["MAIN"]["flags"] == ["estimated", "credit_link", "tax"]
    # the result file itself, and its build step in the 404 hint
    import scenariolab.server as server
    monkeypatch.setattr(server, "RESULTS", results)
    assert client.get("/api/instrument_meta").json()["scenario_ids"] == ["calm", "storm", "boom"]
    assert server.BUILD_STEP["instrument_meta"] == "instruments"


def test_example_ticker_preset_is_evaluated_into_portfolio_default(with_tickers):
    out = service.run(None)
    assert out["instrument_presets_issue"] is None and len(out["instrument_presets"]) == len(service.INSTRUMENT_PRESETS)
    p = out["instrument_presets"][0]
    assert p["id"] == "income_tickers" and set(p["withdrawal_by_horizon"]) == {"10", "30"}
    assert sum(p["weights"].values()) == pytest.approx(1.0) and p["cash_flow_by_horizon"]["30"]["cash_y1_over_withdrawal_p50"] > 0
    assert [q["id"] for q in out["presets"]] == [q["id"] for q in service.PRESETS]       # the asset presets stay
    (with_tickers / tk.PATHS_FILE).unlink()
    service.clear_cache()
    out = service.run(None)
    assert out["instrument_presets"] is None and "--only instruments" in out["instrument_presets_issue"]


# ---------------------------------------------------------------- the build: macro paths, the instruments step

def test_scenario_macro_is_saved_from_the_runs_in_hand(tmp_path):
    """library.save_macro writes what instruments.build reads (fresh: no re-simulation) from the SimOutputs only."""
    from types import SimpleNamespace

    import pandas as pd

    from scenariolab.instruments import build as ib
    from scenariolab.scenarios import library
    rng = np.random.default_rng(5)
    dates = pd.date_range("1973-01-01", periods=600, freq="MS")
    sims = [SimpleNamespace(total=rng.normal(0, 0.03, (6, 24, len(ASSETS))).astype(np.float32),
                            macro={k: rng.normal(4, 1, (6, 25)).astype(np.float32) for k in ("y10", "ff", "spread")},
                            idx=rng.integers(0, 600, (6, 24))) for _ in range(2)]
    ids = ["a", "b"]
    np.savez_compressed(tmp_path / "scenario_paths.npz", total=np.stack([s.total for s in sims]).astype(np.float32),
                        ids=np.array(ids))
    assert library.save_macro(tmp_path, sims, ids, SimpleNamespace(dates=dates)) == tmp_path / library.MACRO_FILE
    with np.load(tmp_path / "scenario_paths.npz") as z:
        got = ib.scenario_macro(None, tmp_path, {k: z[k] for k in z.files})       # ctx=None: must not simulate
    assert got["source"] == library.MACRO_FILE
    assert np.array_equal(got["y10"][1], sims[1].macro["y10"]) and got["drawn"][0][2, 3] == dates[sims[0].idx[2, 3]].strftime("%Y-%m")
    # a failure is reported, not raised: the scenario paths are already written
    assert library.save_macro(tmp_path, [SimpleNamespace(total=None)], ["x"], None) is None


def test_the_instruments_step_runs_the_layer_then_the_continuation(monkeypatch, tmp_path, capsys):
    from scenariolab import build
    calls = []
    monkeypatch.setattr(build, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(build.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(build.config, "PANEL_PATH", tmp_path / "panel.csv")
    (tmp_path / "panel.csv").write_text("date\n", encoding="utf-8")
    monkeypatch.setattr(build.EngineContext, "load", classmethod(lambda cls, **kw: "ctx"))
    monkeypatch.setattr(build, "_run", lambda module, ctx, **kw: calls.append(module) or {})
    assert build.main(["--only", "instruments"]) == 0
    assert calls == ["instruments.build", "portfolio.tickers"]
    calls.clear()
    assert build.main(["--only", "scenarios", "instruments", "portfolio"]) == 0
    assert calls == ["scenarios.library", "instruments.build", "portfolio.tickers", "portfolio.service"]
    (tmp_path / tk.PATHS_FILE).write_bytes(b"")
    capsys.readouterr()
    assert build.main(["--only", "scenarios"]) == 0
    assert "--only instruments" in capsys.readouterr().out


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
def test_build_continuation_hook_sees_every_run_and_changes_nothing():
    from scenariolab.context import EngineContext

    ctx = EngineContext.load(selection=True)
    seen = []
    a = service.build_continuation(ctx, offset=180, n_paths=50, seed=5, chunk=20, yearly=True)
    b = service.build_continuation(ctx, offset=180, n_paths=50, seed=5, chunk=20, yearly=True,
                                   hook=lambda sim, r0, c: seen.append((r0, c, sim.total.shape)) or c)
    assert seen == [(0, 0, (20, 480, len(ASSETS))), (20, 1, (20, 480, len(ASSETS))), (40, 2, (10, 480, len(ASSETS)))]
    assert b["hook"] == [0, 1, 2] and "hook" not in a and a["info"] == b["info"]
    for k in ("total", "price", "income", "cpi", "order"):
        assert np.array_equal(getattr(a["factors"], k), getattr(b["factors"], k)), k


# ---------------------------------------------------------------- the project endpoint and custom scenarios (the layer)

def _db():
    from types import SimpleNamespace

    import pandas as pd
    return SimpleNamespace(dates=pd.date_range("1973-01-01", periods=600, freq="MS"))


def _macro(rng, S_, n):
    y10 = 5.29 + np.concatenate([np.zeros((S_, n, 1)), np.cumsum(rng.normal(0, 0.08, (S_, n, T)), axis=2)], axis=2)
    ff = np.clip(3.75 + 0.8 * (y10 - 5.29), 0.0, None)
    spread = np.clip(1.47 + np.concatenate([np.zeros((S_, n, 1)), np.cumsum(rng.normal(0, 0.04, (S_, n, T)), axis=2)],
                                           axis=2), 0.3, None)
    return {"y10": y10.astype(np.float32), "ff": ff.astype(np.float32), "spread": spread.astype(np.float32)}


def _write_macro(root, seed=2, custom=None):
    """scenario_macro.npz (or custom/<id>.macro.npz) for the synthetic paths, in library.save_macro's layout."""
    target = root if custom is None else root / "custom"
    with np.load(target / ("scenario_paths.npz" if custom is None else f"{custom}.npz")) as z:
        ids = [str(x) for x in z["ids"]]
        S_, n = z["total"].shape[:2]
        fp = tk.paths_fingerprint(ids, np.asarray(z["total"])[:, :2])
    rng = np.random.default_rng(seed)
    dates = np.asarray(_db().dates.strftime("%Y-%m")).astype("U7")
    name = tk.MACRO_FILE if custom is None else f"{custom}{tk.CUSTOM_MACRO_SUFFIX}"
    np.savez_compressed(target / name, ids=np.array(ids), **_macro(rng, S_, n),
                        drawn_idx=rng.integers(0, 600, (S_, n, T)).astype(np.int16), db_dates=dates,
                        paths_fingerprint=np.array(fp))


def test_project_api_on_synthetic_paths(with_tickers):
    """POST /api/instruments/project runs the real recipes on the stored scenario paths (T17, T21)."""
    from fastapi.testclient import TestClient

    from scenariolab.server import app
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/api/instruments/project", json={"ticker": "TQQQ"})
    assert r.status_code == 503 and tk.MACRO_FILE in r.json()["detail"] and "--only instruments" in r.json()["detail"]
    _write_macro(with_tickers)
    t0 = time.time()
    r = client.post("/api/instruments/project", json={"ticker": "tqqq"})
    assert r.status_code == 200, r.text[:300]
    assert time.time() - t0 < 10.0
    p = r.json()
    assert p["ticker"] == "TQQQ" and p["kind"] == "structural" and len(p["labels"]) == T + 1
    assert p["quarter_index"][:3] == [0, 3, 6] and [e["id"] for e in p["per_scenario"]] == ["calm", "storm", "boom"]
    agg = p["aggregate"]["weighted"]
    for key in ("cagr10_p10", "cagr10_p50", "cagr10_p90", "max_dd_p50", "prob_loss15", "wipeout_prob"):
        assert key in agg, key
    assert agg["cagr10_p10"] <= agg["cagr10_p50"] <= agg["cagr10_p90"] and agg["income_y1_p50"] == 0.0
    fans = agg["fans"]
    assert len(fans["total"]["p50"]) == T + 1 and fans["total"]["p50"][0] == 1.0 and len(fans["dps"]["p50"]) == T // 12
    assert set(p["per_scenario"][0]["attribution"]["median_path"]) == {"index", "vol_drag", "third_moment",
                                                                       "fourth_moment", "dividend", "financing", "fee"}
    assert p["part_names"]["vol_drag"] == "波动损耗" and set(p["corr"]["with"]) >= {"us_equity", "nasdaq", "QQQ", "MAIN"}
    assert p["recomputed"] is False and p["params"]["s_eff"] == "mean_2019_2026" and "ruin_extrap" in p["flags"]
    again = client.post("/api/instruments/project", json={"ticker": "TQQQ"}).json()
    assert again["cached"] is True and again["aggregate"] == p["aggregate"]
    dear = client.post("/api/instruments/project", json={"ticker": "TQQQ", "params": {"s_eff": "latest", "ruin": False}}).json()
    assert dear["recomputed"] is True and dear["aggregate"]["weighted"]["wipeout_prob"] is None    # not drawn
    assert dear["ruin"] is False and "ruin_extrap" not in dear["flags"] and any("爆仓抽签已关闭" in n for n in dear["notes"])
    assert dear["aggregate"]["weighted"]["cagr10_p50"] < agg["cagr10_p50"] + 0.01
    # an estimated ticker pays cash, by its own payout rule; the alpha variant is a parameter
    m = client.post("/api/instruments/project", json={"ticker": "MAIN", "alpha_mode": "half"}).json()
    assert m["alpha_mode"] == "half" and m["params"]["alpha_mode"] == "half"
    assert m["aggregate"]["weighted"]["income_y1_p50"] > 0.05 and "alpha" in m["aggregate"]["weighted"]["attribution"]["mean"]
    # the credit-link chip follows this request's own correlations (T18); a typed start price is the reader's
    assert m["credit_link"] is not None and ("credit_link" in m["flags"]) == m["credit_link"]["bad"]
    assert ("credit_link_ok" in m["flags"]) == (not m["credit_link"]["bad"])
    u = client.post("/api/instruments/project", json={"ticker": "MAIN", "params": {"price": 40}}).json()
    assert u["start"]["basis"] == "user" and u["start"]["user_price"] == 40 and u["start"]["price"] == 40
    assert "start_est" not in u["flags"] and "start_est" in m["flags"] and isinstance(u["start"]["build"], dict)
    v = client.post("/api/instruments/project", json={"ticker": "QQQ", "vol_source": "fixed", "fixed_vol": 0.3}).json()
    assert v["vol_applies"] is False and v["recomputed"] is False and any("不影响这个标的" in n for n in v["notes"])
    g = client.post("/api/instruments/project", json={"ticker": "GOLD", "params": {"price": 4000}}).json()
    assert g["start"]["price"] == 4000.0 and "gold_avg" in g["flags"]
    for body, needle in (({"ticker": "XYZ"}, "未知标的"), ({"ticker": "TQQQ", "params": {"bogus": 1}}, "没有参数"),
                         ({"ticker": "TQQQ", "params": {"s_eff": 0.5}}, "之间"),
                         ({"ticker": "TQQQ", "params": {"s_eff": "soon"}}, "之一"),
                         ({"ticker": "TQQQ", "vol_source": "fixed"}, "fixed_vol"),
                         ({"ticker": "TQQQ", "vol_source": "fixed", "fixed_vol": 3}, "固定波动"),
                         ({"ticker": "TQQQ", "fixed_vol": 0.2}, "fixed_vol"),
                         ({"ticker": "MAIN", "alpha_mode": "x"}, "alpha_mode"),
                         ({"ticker": "TQQQ", "speed": 1}, "不认识的参数"), ({}, "缺少参数：ticker")):
        r = client.post("/api/instruments/project", json=body)
        assert r.status_code == 400 and needle in r.json()["detail"], (body, r.text[:200])
    fixed = client.post("/api/instruments/project", json={"ticker": "QLD", "vol_source": "fixed", "fixed_vol": 0.25})
    assert fixed.status_code == 200 and "fixed_vol" in fixed.json()["flags"]
    drawn = client.post("/api/instruments/project", json={"ticker": "QQQI", "vol_source": "drawn"})
    assert drawn.status_code in (200, 400), drawn.text[:200]          # needs the drawn months' RV table (data/)


def test_custom_scenario_ticker_files_are_written_and_recomputed_alike(with_tickers):
    """server.custom_run's tickers.write_custom: the layer on a custom run, saved like instrument_paths.npz (S = 1,
    alpha variants included) with its macro paths; the service mixes it in and the project endpoint recomputes the
    same numbers from the saved arrays."""
    from types import SimpleNamespace
    rng = np.random.default_rng(8)
    n = 40
    total = rng.normal(0.005, 0.04, (n, T, len(ASSETS))).astype(np.float32)
    income = np.full_like(total, 0.002)
    total[..., ASSETS.index("cash")] = income[..., ASSETS.index("cash")] = 0.003
    macro = {k: v[0] for k, v in _macro(rng, 1, n).items()}
    macro["cpi"] = (100.0 * 1.002 ** np.arange(T + 1) * np.ones((n, 1))).astype(np.float32)
    sim = SimpleNamespace(total=total, income=income, macro=macro, idx=rng.integers(0, 600, (n, T)),
                          assets=list(ASSETS), start="2026-09-30")
    labels = [service._shift_label("2026-09", k) for k in range(T + 1)]
    custom = with_tickers / "custom"
    np.savez_compressed(custom / "c2.npz", total=total[None], income=income[None], cpi=macro["cpi"][None],
                        ids=np.array(["c2"]), assets=np.array(ASSETS), labels=np.array(labels))
    jsonio.dump({"id": "c2", "name": "自定义标的", "prior_prob": 0.1}, custom / "c2.json")
    info = tk.write_custom(sim, _db(), "c2", custom, labels=labels)
    assert info["saved"] and info["files"] == ["c2.instr.npz", "c2.macro.npz"]
    blk = tk.load_block(custom / "c2.instr.npz")
    assert blk.ids == ["c2"] and blk.n_paths == n and set(blk.alpha) == {"half", "raw"} and blk.fingerprint
    service.clear_cache()
    r = service.evaluate_withdrawal({"MAIN": 0.5, "QQQ": 0.5}, include_custom=["c2"], horizon_years=10,
                                    alpha_mode="raw", strategy="pro_rata")
    assert [e["id"] for e in r["strategies"]["pro_rata"]["per_scenario"]][-1] == "c2"
    _write_macro(with_tickers)
    p = tk.evaluate_project("MAIN", include_custom=["c2"])
    j = blk.col("MAIN")
    stored = np.cumprod(1.0 + blk.total[:, :, j].astype(np.float64), axis=1)[:, -1] ** (1 / 10) - 1
    c2 = [e for e in p["per_scenario"] if e["id"] == "c2"][0]
    assert c2["cagr10_p50"] == pytest.approx(float(np.percentile(stored, 50, method="hazen")), abs=1e-6)
    assert c2["source"] == "custom" and c2["n_paths"] == n


needs_tickers = pytest.mark.skipif(
    not all((config.RESULTS_DIR / f).exists() for f in ("scenario_paths.npz", service.CONTINUATION_FILE, tk.PATHS_FILE,
                                                         tk.CONTINUATION_FILE, tk.META_FILE)),
    reason="instruments not built (python -m scenariolab.build --only instruments)")


@needs_tickers
def test_real_ticker_files_belong_to_the_built_paths_and_stay_fast():
    """The built ticker files compose with the built scenario paths and their continuation (fingerprints), a row's
    ticker years follow its scenario years, and a 12-ticker request stays within the T21 budget."""
    service.clear_cache()
    data = service.load_scenarios()
    ts = data.tickers()
    assert ts.continuation is not None, ts.continuation_issue
    assert ts.continuation.info["asset_fingerprint"] == data.continuation.fingerprint()
    assert ts.tickers == list(TICKERS) and ts.factors.total.shape[:2] == (data.factors.total.shape[0], data.years)
    for k in ("total", "price", "income"):
        assert np.isfinite(getattr(ts.continuation.factors, k)).all() and np.isfinite(getattr(ts.factors, k)).all()
    weights = service.INSTRUMENT_PRESETS[0]["weights"]
    t = time.time()
    p = service.evaluate_portfolio(weights)
    assert time.time() - t < 3.0 and len(p["instruments"]["tickers"]) == 12
    t = time.time()
    w = service.evaluate_withdrawal(weights, horizon_years=30)
    assert time.time() - t < 15.0
    t = time.time()
    service.evaluate_withdrawal(weights, horizon_years=30, rate=0.035)      # cached critical rates
    assert time.time() - t < 3.0
    agg = {s: v["aggregate"]["ruin_prob_weighted"] for s, v in w["strategies"].items()}
    assert agg["pro_rata"] <= agg["div_then_pro_rata"] + 0.05 and 0 < agg["pro_rata"] < 0.5
    assert 1.2 < w["cash_flow"]["pot_share_y1"]["total"] < 2.5           # ~1.7x the 4% withdrawal (design 1.76x)
    half = service.evaluate_withdrawal(weights, horizon_years=30, alpha_mode="half", strategy="pro_rata")
    assert half["safe_rate"]["pro_rata"]["weighted"] < w["safe_rate"]["pro_rata"]["weighted"]
