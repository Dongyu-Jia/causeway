from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.analysis import backtest as bt
from scenariolab.engine.factor_model import ModelSpec

needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")


def test_coverage_and_pit_on_toy_fan():
    actual = np.array([1.0, 1.0, 1.2, 0.8, 1.05])
    lo, hi = np.full(5, 0.9), np.full(5, 1.1)
    assert bt.coverage(actual, lo, hi) == 0.5            # months 1..4: in, out, out, in
    assert bt.coverage(np.full(5, 1.0), lo, hi) == 1.0
    draws = np.array([0.5, 1.0, 1.5, 2.0])
    assert bt.pit(1.0, draws) == pytest.approx(1.5 / 4)   # one below, one tie
    assert bt.pit(10.0, draws) == 1.0 and bt.pit(0.1, draws) == 0.0


def test_error_decomposition_sums_to_total():
    rng = np.random.default_rng(0)
    for _ in range(20):
        actual, oracle, cond, uncond = rng.uniform(0.3, 4.0, size=4)
        d = bt.decompose(actual, oracle, cond, uncond, years=7.5)
        assert d["scenario"] + d["engine"] + d["transmission"] == pytest.approx(d["total"], abs=1e-12)
        assert d["total"] == pytest.approx(np.log(uncond / actual))
        assert d["total_annual"] == pytest.approx(d["total"] / 7.5)


def test_median_path_matches_monte_carlo_median():
    rng = np.random.default_rng(1)
    T, sigma = 120, 0.045
    sys = rng.normal(0.006, 0.02, size=(T, 1))
    income = np.full((T, 1), 0.002)
    noise = rng.normal(0.0, sigma, size=(20000, T))
    terminal = np.prod(1.0 + sys[:, 0] + income[:, 0] + noise, axis=1)
    analytic = bt.median_path(sys, income, sigma ** 2)[-1, 0]
    assert analytic == pytest.approx(np.median(terminal), rel=0.03)
    no_drag = np.prod(1.0 + sys[:, 0] + income[:, 0])
    assert no_drag > 1.1 * np.median(terminal)          # why the drag term is needed


def test_index_path_and_r2():
    path = bt.index_path(np.array([0.1, -0.5, 1.0]))
    assert np.allclose(path[:, 0], [1.0, 1.1, 0.55, 1.1])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert bt.r2_score(y, y) == 1.0
    assert bt.r2_score(y, np.full(4, y.mean())) == 0.0
    assert bt.r2_score(y, np.array([1.0, np.nan, 3.0, 4.0])) == 1.0


def test_out_of_sample_r2_is_against_the_training_mean():
    rng = np.random.default_rng(6)
    y = rng.normal(0.01, 0.04, size=200)                  # test months drift above the training mean
    train_mean = 0.0
    assert bt.r2_score(y, np.full(200, train_mean), train_mean) == pytest.approx(0.0)
    # forecasting the test period's own mean beats the historical mean, so it scores > 0 ...
    assert bt.r2_score(y, np.full(200, y.mean()), train_mean) > 0.0
    # ... while against the test mean (the old definition) it would score exactly 0
    assert bt.r2_score(y, np.full(200, y.mean())) == pytest.approx(0.0)
    bench = np.full(200, train_mean)
    bench[:5] = np.nan                                    # months without a benchmark are skipped
    yhat = y + rng.normal(0, 0.01, size=200)
    expected = 1 - ((y[5:] - yhat[5:]) ** 2).sum() / ((y[5:] - train_mean) ** 2).sum()
    assert bt.r2_score(y, yhat, bench) == pytest.approx(expected)


def test_main_source_never_blames_the_scenario_part():
    # knowing the macro moved the median 7%/yr (and overshot); the conditional error is engine + transmission
    dec = {"total": 0.03, "scenario": 0.07, "engine": 0.05, "transmission": 0.02}
    assert bt.main_source(dec) == "engine"
    assert bt.main_source({**dec, "transmission": 0.06}) == "transmission"


class _ToyDB:
    def __init__(self, frame: pd.DataFrame):
        self.frame, self.dates = frame, frame.index

    def __len__(self) -> int:
        return len(self.frame)

    def prior(self) -> np.ndarray:
        return np.full(len(self.frame), 1.0 / len(self.frame))

    def era_weights(self, q) -> dict:
        return {"all": float(np.sum(q))}


def test_shrunk_phase_report_keeps_the_realized_value_and_lambda():
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    idx = pd.date_range("1980-01-01", periods=300, freq="MS")
    db = _ToyDB(pd.DataFrame({"a": x, "b": x + 0.05 * rng.normal(size=300)}, index=idx))
    views = [[{"var": "a", "mean": 1.0}, {"var": "b", "mean": -1.0}]]    # jointly impossible
    rep = bt.view_weights(db, views, [24])["ep"][0]
    assert rep["shrink"] < 1.0 and rep["shrink_reason"] == "infeasible"
    assert [v["realized"] for v in rep["views"]] == [1.0, -1.0]
    assert abs(rep["views"][0]["target_used"]) < abs(rep["views"][0]["target"])   # what EP was actually asked for


def test_view_delivery_measures_block_smoothing():
    idx = pd.date_range("1980-01-01", periods=120, freq="MS")
    x = np.zeros(120)
    x[50] = 1.0                                                        # one spike month
    db = _ToyDB(pd.DataFrame({"x": x}, index=idx))
    q = np.zeros(120)
    q[50] = 1.0                                                        # EP puts all weight on it
    from scenariolab.engine.sampler import sample_phases
    draws = sample_phases([q], [24], n_paths=200, block=6, seed=0)
    rep = [{"views": [{"var": "x", "realized": 1.0, "posterior_mean": 1.0, "prior_sd": float(x.std())}]}]
    sim = SimpleNamespace(idx=draws, macro={})                         # month weights: legacy sampler path
    gap = bt.add_view_delivery(rep, db, sim, [24])
    v = rep[0]["views"][0]
    assert v["ep_gap_sd"] == pytest.approx(0.0)
    assert v["drawn_mean"] == pytest.approx(1.0 / 6.0)                 # each 6-month block holds the spike once
    assert gap["drawn"] == pytest.approx(abs(1.0 / 6.0 - 1.0) / x.std())
    assert v["simulated_mean"] is None                                 # x has no simulated macro path


def test_phase_plan_and_realized_views():
    assert bt.phase_plan(141) == [24] * 5 + [21]
    assert bt.phase_plan(48) == [24, 24]
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    panel = pd.DataFrame({v: np.arange(30, dtype=float) for v in bt.VIEW_VARS}, index=idx)
    views = bt.realized_views(panel, idx, [24, 6])
    assert views[0][0] == {"var": bt.VIEW_VARS[0], "mean": pytest.approx(11.5)}
    assert views[1][0]["mean"] == pytest.approx(26.5)


def _perturbed_after(panel: pd.DataFrame, date: str, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    P = panel.copy()
    rows = P.index > pd.Timestamp(date)
    cols = [c for c in P.columns if c.startswith(("r_", "p_", "d_", "f_", "s_", "lvl_", "e_", "dy_"))]
    noise = rng.normal(1.0, 0.5, size=(int(rows.sum()), len(cols)))
    P.loc[rows, cols] = P.loc[rows, cols].to_numpy() * noise + 0.01
    return P


@needs_panel
def test_window_engine_uses_only_training_months():
    from scenariolab.context import load_panel_with_shocks
    from scenariolab.engine.simulate import simulate
    panel = load_panel_with_shocks()
    train_end = "1994-12"
    other = _perturbed_after(panel, train_end)
    spec = ModelSpec()
    m1, db1, a1 = bt.fit_window(panel, spec, train_end)
    m2, db2, a2 = bt.fit_window(other, spec, train_end)
    assert np.allclose(m1.coef.to_numpy(), m2.coef.to_numpy())
    assert np.allclose(db1.resid.to_numpy(), db2.resid.to_numpy())
    assert db1.dates.max() == pd.Timestamp(train_end)
    assert (a1.ff, a1.y10, a1.cpi) == (a2.ff, a2.y10, a2.cpi) and a1.income_yield == a2.income_yield
    views = [[{"var": "s_cpi_yoy", "mean": 3.0}, {"var": "f_oil", "mean": 0.01}]] * 2
    w1, w2 = bt.view_weights(db1, views, [24, 24]), bt.view_weights(db2, views, [24, 24])
    assert all(np.allclose(p.probs, q.probs) for p, q in zip(w1["cond"], w2["cond"]))
    s1 = simulate(db1, a1, w1["cond"], [24, 24], n_paths=50, seed=4)
    s2 = simulate(db2, a2, w2["cond"], [24, 24], n_paths=50, seed=4)
    assert np.allclose(s1.total, s2.total)


def test_expanding_predictions_do_not_see_the_future():
    rng = np.random.default_rng(5)
    n = 360                                              # 1980-01 .. 2009-12: refits 1990, 95, 2000, 05
    idx = pd.date_range("1980-01-01", periods=n, freq="MS")
    P = pd.DataFrame(index=idx)
    for f in config.CORE_FACTOR_IDS:
        P[f"f_{f}"] = rng.normal(scale=0.01, size=n)
    for a in config.ASSET_IDS:
        P[f"p_{a}"] = 0.004 + 0.8 * P["f_oil"] - 1.5 * P["f_y10"] + rng.normal(scale=0.01, size=n)
    months = P.index
    pred, _, bench, refits = bt.expanding_predictions(P, ModelSpec(), months, refit_every=60, min_train=120)
    assert refits[0] == "1990-01" and pred.loc[:"1989-12"].isna().all().all()
    # the R^2 benchmark of each block is the mean of the months before it, not of the block itself
    assert bench.loc["1995-01-01", "us_equity"] == pytest.approx(P.loc[:"1994-12", "p_us_equity"].mean())
    assert bench.loc["1990-01-01", "us_equity"] == bench.loc["1994-12-01", "us_equity"]
    t = pd.Timestamp("2001-06-01")
    Q = P.copy()
    Q.loc[Q.index >= t, [c for c in Q.columns if c.startswith("p_")]] *= -3.0
    Q.loc[Q.index > t, [c for c in Q.columns if c.startswith("f_")]] += 0.05
    pred_q, _, bench_q, _ = bt.expanding_predictions(Q, ModelSpec(), months, refit_every=60, min_train=120)
    assert np.allclose(pred.loc[:t].dropna(), pred_q.loc[:t].dropna())
    assert np.allclose(bench.loc[:t].dropna(), bench_q.loc[:t].dropna())
    assert not np.allclose(pred.loc["2005-01":], pred_q.loc["2005-01":])


@needs_panel
def test_oracle_follows_the_simulators_structural_rules():
    from scenariolab.context import load_panel_with_shocks
    panel = load_panel_with_shocks()
    model, db, anchor = bt.fit_window(panel, ModelSpec(), "2004-12")
    months = panel.loc["2005-01":"2014-12"].index
    oracle = bt.oracle_paths(panel, model, anchor, months)
    actual = bt.index_path(panel.loc[months, [f"r_{a}" for a in bt.ASSETS]].to_numpy())
    for a in ("treasury_10y", "crude_oil"):
        j = bt.ASSETS.index(a)
        assert np.allclose(oracle[:, j], actual[:, j])
    j = bt.ASSETS.index("cash")
    gap = np.log(oracle[-1, j] / actual[-1, j]) / 10.0
    assert abs(gap) < 0.01                              # ff-tb3 basis rule, within 1%/yr


# ---------------------------------------------------------------- custom windows (POST /api/backtest/run)

def _month_panel(start: str = "1947-01", end: str = "2026-09") -> pd.DataFrame:
    idx = pd.date_range(f"{start}-01", f"{end}-01", freq="MS")
    return pd.DataFrame({"x": np.zeros(len(idx))}, index=idx)


def test_custom_window_trains_up_to_train_end_and_tests_the_next_months():
    panel = _month_panel()
    w = bt.custom_window(panel, "1999-12", 60)
    assert w["id"] == bt.CUSTOM_ID and w["name"] == "自选窗口"
    assert w["train_end"] == "1999-12" and w["test"] == ("2000-01", "2004-12") and w["months"] == 60
    assert w["label"] == "自选窗口：1999-12 之后 5 年"
    assert bt.custom_window(panel, " 2007-06 ", "36")["test"] == ("2007-07", "2010-06")      # strings are read
    assert bt.custom_window(panel, "2019-12", 30.0)["label"].endswith("30 个月")
    # the bounds: 15 training years from MODEL_START, a test that ends by the last panel month
    assert bt.custom_window(panel, "1987-12", 180)["test"] == ("1988-01", "2002-12")
    assert bt.custom_window(panel, "2025-09", 12)["test"] == ("2025-10", "2026-09")


@pytest.mark.parametrize("train_end, months, needle", [
    ("1999-13", 60, "不是合法的月份"), ("abc", 60, "格式应为 YYYY-MM"), (None, 60, "不是合法的月份"),
    ("2030-01", 12, "不在数据范围内"), ("1940-01", 12, "不在数据范围内"),
    ("0000-01", 12, "不在数据范围内"), ("9999-12", 12, "不在数据范围内"),       # years pandas cannot format
    ("1987-11", 12, "训练期至少需要 15 年"), ("1980-12", 60, "最早可选 1987-12"),
    ("2024-12", 60, "超出数据的最后一个月 2026-09"), ("2026-01", 12, "train_end 最晚为 2025-09"),
    ("1999-12", 6, "months 必须是 12 到 180 之间的整数"), ("1999-12", 181, "收到 181"),
    ("1999-12", 60.5, "收到 60.5"), ("1999-12", True, "months 必须是"), ("1999-12", "五年", "收到「五年」"),
])
def test_custom_window_rejects_bad_requests_in_chinese(train_end, months, needle):
    with pytest.raises(ValueError, match=needle):
        bt.custom_window(_month_panel(), train_end, months)


def test_custom_request_checks_every_parameter():
    panel = _month_panel()
    req = bt.custom_request(panel, {"train_end": "2007-06", "months": 36})
    assert req == {"train_end": "2007-06", "months": 36, "n_paths": bt.CUSTOM_PATHS, "seed": bt.SEED}
    assert bt.custom_request(panel, {"train_end": "2007-06", "months": 36, "n_paths": "400", "seed": 3})["n_paths"] == 400
    for body, needle in [({"months": 36}, "缺少参数：train_end"), ({"train_end": "2007-06"}, "缺少参数：months"),
                         ({"train_end": "2007-06", "months": 36, "bogus": 1}, "不认识的参数：bogus"),
                         ({"train_end": "2007-06", "months": 36, "n_paths": 50}, "n_paths 必须是 100 到 400"),
                         ({"train_end": "2007-06", "months": 36, "n_paths": 1e9}, "n_paths 必须是"),
                         ({"train_end": "2007-06", "months": 36, "seed": -1}, "seed 必须是"),
                         ({"train_end": "2007-06", "months": 36, "seed": 1.5}, "seed 必须是")]:
        with pytest.raises(ValueError, match=needle):
            bt.custom_request(panel, body)


def test_window_shocks_lists_the_overlapping_exogenous_shocks():
    names = [s["name"] for s in bt.window_shocks("2019-12", "2021-12")]
    assert names == ["新冠疫情与全面封锁"]
    assert bt.window_shocks("2010-01", "2010-12") == []
    assert len(bt.window_shocks("1973-01", "2026-09")) == len(config.EXOGENOUS_SHOCKS)


@needs_panel
def test_custom_window_has_the_builtin_shape_and_is_deterministic_by_seed():
    from scenariolab.context import load_panel_with_shocks
    panel = load_panel_with_shocks()
    a = bt.run_custom(panel, ModelSpec(), "2019-12", 24, n_paths=100, seed=5)
    b = bt.run_custom(panel, ModelSpec(), "2019-12", 24, n_paths=100, seed=5)
    c = bt.run_custom(panel, ModelSpec(), "2019-12", 24, n_paths=100, seed=6)
    builtin = bt.run_window(panel, ModelSpec(), {"id": "x", "name": "x", "train_end": "2019-12", "test": ("2020-01", "2021-12")},
                            n_paths=100, seed=5)
    assert set(builtin) <= set(a) and {"custom", "label", "shocks"} <= set(a)
    assert a["train"] == ["1973-01", "2019-12"] and a["test"] == ["2020-01", "2021-12"] and a["n_paths"] == 100
    assert a["custom"] == {"train_end": "2019-12", "months": 24, "n_paths": 100, "seed": 5, "param": "2019-12:24"}
    assert [s["name"] for s in a["shocks"]] == ["新冠疫情与全面封锁"]
    eq = a["assets"]["us_equity"]
    for key in ("fan_cond", "fan_uncond", "coverage_cond", "coverage_uncond", "pit_terminal_cond", "cagr_actual",
                "cagr_pred_cond_p50", "decomposition"):
        assert key in eq, key
    assert len(eq["fan_cond"]["p50"]) == 25 and {"posterior", "realized", "prior"} <= set(a["corr"])
    assert np.allclose(eq["fan_cond"]["p50"], builtin["assets"]["us_equity"]["fan_cond"]["p50"])     # run_window itself
    assert np.allclose(eq["fan_cond"]["p50"], b["assets"]["us_equity"]["fan_cond"]["p50"])
    assert not np.allclose(eq["fan_cond"]["p50"], c["assets"]["us_equity"]["fan_cond"]["p50"])
    assert eq["cagr_actual"] == c["assets"]["us_equity"]["cagr_actual"]                                # only the draws moved


@needs_panel
def test_custom_window_sees_no_month_after_train_end_but_the_oracle_views():
    """Perturbing the data after the test window changes nothing; perturbing every month after train_end leaves
    the model, the month database, the anchor, the CMA and the unconditional fans as they were (only the
    documented oracle views, the realized test path and the scores move)."""
    from scenariolab import jsonio
    from scenariolab.context import load_panel_with_shocks
    from scenariolab.engine.simulate import capital_market_assumptions
    panel, spec = load_panel_with_shocks(), ModelSpec()
    te, months = "2019-12", 24
    base = bt.run_custom(panel, spec, te, months, n_paths=100, seed=5)
    later = bt.run_custom(_perturbed_after(panel, "2021-12"), spec, te, months, n_paths=100, seed=5)
    assert jsonio.dumps(later) == jsonio.dumps(base)                       # nothing after the test window is read
    other = _perturbed_after(panel, te)
    w = bt.custom_window(panel, te, months)
    s1, s2 = bt.window_setup(panel, spec, w), bt.window_setup(other, spec, w)
    assert s1["model"].coef.equals(s2["model"].coef) and s1["db"].frame.equals(s2["db"].frame)
    assert s1["db"].resid.equals(s2["db"].resid) and s1["db"].dates.max() == pd.Timestamp(te)
    assert jsonio.dumps(vars(s1["anchor"])) == jsonio.dumps(vars(s2["anchor"]))
    for method in ("historical", "valuation_adjusted"):
        assert (jsonio.dumps(capital_market_assumptions(s1["db"], s1["anchor"], method=method))
                == jsonio.dumps(capital_market_assumptions(s2["db"], s2["anchor"], method=method)))
    moved = bt.run_custom(other, spec, te, months, n_paths=100, seed=5)
    for a in bt.ASSETS:
        assert moved["assets"][a]["fan_uncond"] == base["assets"][a]["fan_uncond"]
    assert moved["assets"]["us_equity"]["fan_cond"] != base["assets"]["us_equity"]["fan_cond"]      # the oracle views
    assert not np.allclose(moved["assets"]["us_equity"]["actual"], base["assets"]["us_equity"]["actual"])


# ---------------------------------------------------------------- P0.1 train-end anchors, P0.2 the macro-only arm

def test_the_two_conditional_arms_are_the_design_lists():
    """SIM_DESIGN §10.5: 宏观 + 金融条件 = the current VIEW_VARS; 只有宏观 = the five macro variables."""
    assert bt.VIEW_VARS == ("s_cpi_yoy", "s_ip_yoy", "f_ff", "f_y10", "f_spread", "f_oil", "f_unrate", "f_usd")
    assert bt.MACRO_VIEW_VARS == ("s_cpi_yoy", "s_ip_yoy", "f_ff", "f_oil", "f_unrate")
    assert set(bt.MACRO_VIEW_VARS) < set(bt.VIEW_VARS)
    assert bt.ARM_NAMES == {"cond": "宏观 + 金融条件（含市场价格）", "cond_macro": "只有宏观"}
    assert bt.VARIANTS["cond_macro"] == ("cond_macro", "historical")


SHORT = {"id": "x", "name": "x", "train_end": "2019-12", "test": ("2020-01", "2021-12")}


@needs_panel
def test_windows_simulate_with_the_anchors_knowable_at_the_train_end(monkeypatch):
    from scenariolab.context import load_panel_with_shocks
    from scenariolab.engine import anchors
    from scenariolab.engine.simulate import DEFAULT_REVERSION
    panel, spec = load_panel_with_shocks(), ModelSpec()
    s = bt.window_setup(panel, spec, SHORT)
    assert s["reversion"] == anchors.window_reversion(panel, "2019-12") != {k: DEFAULT_REVERSION[k] for k in s["reversion"]}
    seen = []
    real = bt.simulate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("reversion"))
        return real(*args, **kwargs)

    monkeypatch.setattr(bt, "simulate", spy)
    win = bt.run_window(panel, spec, SHORT, n_paths=100, seed=5, setup=s)
    assert len(seen) == len(bt.VARIANTS) and all(r == s["reversion"] for r in seen)       # every fan, both CMAs
    a = win["anchors"]
    assert a["rule"] == "train_end" and a["end"] == "2019-12"
    assert a["reversion"] == s["reversion"] and a["default"] == {k: DEFAULT_REVERSION[k] for k in bt.ANCHOR_KEYS}
    assert "长期锚按训练期末可知的规则" in win["summary"]["interpretation"]
    seen.clear()
    old = bt.run_window(panel, spec, SHORT, n_paths=100, seed=5, setup=s, reversion={})   # the fixed anchors of before
    assert seen == [{}] * len(bt.VARIANTS) and old["anchors"]["rule"] == "given"
    assert old["anchors"]["reversion"] == old["anchors"]["default"]
    assert old["assets"]["us_equity"]["fan_cond"] != win["assets"]["us_equity"]["fan_cond"]
    assert np.array_equal(old["assets"]["us_equity"]["actual"], win["assets"]["us_equity"]["actual"])
    # the anchors read no month after the train end
    other = _perturbed_after(panel, "2019-12")
    assert bt.window_setup(other, spec, SHORT)["anchors"] == s["anchors"]


@needs_panel
def test_the_macro_arm_reproduces_fan_cond_when_its_views_are_the_same():
    """P0.2 exit test: the new arm is the old fan_cond's machinery; with VIEW_VARS as its views it is fan_cond."""
    from scenariolab import jsonio
    from scenariolab.context import load_panel_with_shocks
    panel, spec = load_panel_with_shocks(), ModelSpec()
    same = jsonio.clean(bt.run_window(panel, spec, SHORT, n_paths=100, seed=5,
                                      setup=bt.window_setup(panel, spec, SHORT, macro_vars=bt.VIEW_VARS)))
    for a in bt.ASSETS:
        e = same["assets"][a]
        assert e["fan_cond_macro"] == e["fan_cond"], a
        assert (e["coverage_cond_macro"], e["pit_terminal_cond_macro"], e["cagr_pred_cond_macro_p50"]) == (
            e["coverage_cond"], e["pit_terminal_cond"], e["cagr_pred_cond_p50"])
        assert e["decomposition_macro"] == e["decomposition"]
    s = same["summary"]
    assert s["cagr_mae_cond_macro_risk"] == s["cagr_mae_cond_risk"] and s["coverage_cond_macro_risk"] == s["coverage_cond_risk"]
    assert same["ep_macro"] == same["ep"] and same["arms"]["cond_macro"]["view_vars"] == list(bt.VIEW_VARS)
    # by default the arm has the five macro views only, and its fan differs
    win = jsonio.clean(bt.run_window(panel, spec, SHORT, n_paths=100, seed=5))
    assert [v["var"] for v in win["ep_macro"][0]["views"]] == list(bt.MACRO_VIEW_VARS)
    assert [v["var"] for v in win["ep"][0]["views"]] == list(bt.VIEW_VARS)
    assert win["assets"]["us_equity"]["fan_cond_macro"] != win["assets"]["us_equity"]["fan_cond"]
    assert win["assets"]["us_equity"]["fan_cond"] == same["assets"]["us_equity"]["fan_cond"]      # fan_cond unchanged
    assert win["arms"]["cond"]["name"] == "宏观 + 金融条件（含市场价格）" and win["arms"]["cond_macro"]["name"] == "只有宏观"
    text = win["summary"]["interpretation"]
    assert "只知道宏观" in text and "金融条件" in text
    for key in ("cagr_mae_cond_macro_risk", "coverage_cond_macro_risk", "pit_cond_macro_mad", "ens_min_macro"):
        assert np.isfinite(win["summary"][key]), key
