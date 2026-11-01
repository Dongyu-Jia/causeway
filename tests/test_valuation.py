"""Valuation state: no look-ahead, the P/D accounting identity, default behaviour, reproducibility, selection."""
import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.analysis import backtest as bt
from scenariolab.analysis import valuation as va
from scenariolab.data.panel import inflation_expectation, release_dated
from scenariolab.engine.factor_model import ModelSpec
from scenariolab.engine.simulate import (capital_market_assumptions, fit_valuation, hac_ols, simulate,
                                         valuation_settings)

needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
NEW_COLS = ("s_infl_expect", "s_real_y10", "s_profit_share_rt", "s_profit_growth_yoy", "s_fcf_share", "s_cape",
            "f_real_y10", "f_profit_growth", "f_fcf_share", "f_cape")


@pytest.fixture(scope="module")
def panel():
    from scenariolab.context import load_panel_with_shocks
    return load_panel_with_shocks()


def _window(panel, train_end="2004-12"):
    return bt.fit_window(panel, ModelSpec(), train_end)


def _perturbed_after(panel: pd.DataFrame, date: str, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    P = panel.copy()
    rows = P.index > pd.Timestamp(date)
    cols = [c for c in P.columns if c.startswith(("r_", "p_", "d_", "f_", "s_", "lvl_", "e_", "dy_"))]
    P.loc[rows, cols] = P.loc[rows, cols].to_numpy() * rng.normal(1.0, 0.5, size=(int(rows.sum()), len(cols))) + 0.01
    return P


def _prior(db, months):
    from scenariolab.engine.pooling import BLOCK, pool_phase
    return [pool_phase(db.frame, [], db.prior(), m, BLOCK, rare=db.rare).blocks for m in months]


# ---------------------------------------------------------------- data (no look-ahead)

def test_release_dated_uses_only_published_quarters():
    q = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2000-01-01", periods=4, freq="QS"))
    idx = pd.date_range("2000-01-01", "2001-12-01", freq="MS")
    s = release_dated(q, idx, lag_months=5)
    assert s.loc[:"2000-05"].isna().all()                        # Q1 (Jan) is first known in June
    assert s.loc["2000-06-01"] == 1.0 and s.loc["2000-08-01"] == 1.0 and s.loc["2000-09-01"] == 2.0
    assert s.loc["2001-03-01"] == 4.0 and s.loc["2001-12-01"] == 4.0   # held until the next release
    # a later quarter changing cannot move any earlier month
    q2 = q.copy()
    q2.iloc[-1] = 99.0
    s2 = release_dated(q2, idx, lag_months=5)
    assert s.loc[:"2001-02"].equals(s2.loc[:"2001-02"])


def test_inflation_expectation_is_the_simulators_recursion():
    rng = np.random.default_rng(0)
    cpi = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.003, 0.002, 240))),
                    index=pd.date_range("1990-01-01", periods=240, freq="MS"))
    e = inflation_expectation(cpi, half_life=36)
    alpha = 1.0 - 0.5 ** (1.0 / 36)
    pi = 1200.0 * np.log(cpi).diff().to_numpy()
    rec = np.empty(240)
    rec[1] = pi[1]
    for t in range(2, 240):
        rec[t] = rec[t - 1] + alpha * (pi[t] - rec[t - 1])
    assert np.isnan(e.iloc[0]) and np.allclose(e.to_numpy()[1:], rec[1:])
    # only the past enters
    cpi2 = cpi.copy()
    cpi2.iloc[200:] *= 1.5
    assert np.allclose(inflation_expectation(cpi2).iloc[:200], inflation_expectation(cpi).iloc[:200], equal_nan=True)


def test_hac_ols_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 2))
    y = 1.0 + 0.5 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(scale=0.1, size=600)
    b, t, r2, e = hac_ols(y, X, lags=12)
    assert np.allclose(b, [1.0, 0.5, -0.2], atol=0.02) and r2 > 0.9 and abs(t[1]) > 20 and abs(e.mean()) < 1e-10


@needs_panel
def test_new_panel_columns_are_complete_and_release_dated(panel):
    P = panel.loc[config.MODEL_START:]
    assert all(c in P for c in NEW_COLS)
    assert P[[c for c in NEW_COLS if c != "s_cape" and c != "f_cape"]].notna().all().all()
    # profit growth only moves in release months (quarter start + 5: March, June, September, December)
    moves = P.index[P["f_profit_growth"].abs() > 0]
    assert set(moves.month) <= {3, 6, 9, 12}
    assert (P["s_real_y10"] - (P["lvl_y10"] - P["s_infl_expect"])).abs().max() < 1e-9


# ---------------------------------------------------------------- engine: no look-ahead

@needs_panel
def test_fair_value_kappa_and_simulation_use_only_training_months(panel):
    train_end = "2004-12"
    other = _perturbed_after(panel, train_end)
    m1, db1, a1 = _window(panel, train_end)
    m2, db2, a2 = _window(other, train_end)
    v1 = fit_valuation(db1, a1, kappa="ecm", income="path")
    v2 = fit_valuation(db2, a2, kappa="ecm", income="path")
    assert v1.coef == v2.coef and v1.tstat == v2.tstat and v1.kappa == v2.kappa and v1.ecm == v2.ecm
    assert v1.state0 == v2.state0 and v1.log_pd0 == v2.log_pd0 and v1.sample[1] == train_end
    assert np.array_equal(v1.g_div, v2.g_div) and np.array_equal(v1.ps_change, v2.ps_change) and v1.beta == v2.beta
    c1 = capital_market_assumptions(db1, a1, method="valuation_state")
    c2 = capital_market_assumptions(db2, a2, method="valuation_state")
    w1, w2 = _prior(db1, [24, 24]), _prior(db2, [24, 24])
    s1 = simulate(db1, a1, w1, [24, 24], n_paths=60, seed=4, cma=c1, valuation=v1, **config.RESIDUALS)
    s2 = simulate(db2, a2, w2, [24, 24], n_paths=60, seed=4, cma=c2, valuation=v2, **config.RESIDUALS)
    assert np.array_equal(s1.total, s2.total) and np.array_equal(s1.macro["pd"], s2.macro["pd"])
    assert np.array_equal(s1.macro["pd_fair"], s2.macro["pd_fair"])


@needs_panel
def test_backtest_and_forecast_with_the_valuation_state_use_only_the_past(panel):
    from scenariolab.analysis import forecast as fc
    t = pd.Timestamp("1999-12-01")
    other = _perturbed_after(panel, "1999-12")
    vs = {"enabled": True, "kappa": 0.1, "cma": "valuation_state"}
    a = fc.forecast_origin(panel, ModelSpec(), t, n_paths=100, seed=5, valuation=vs)
    b = fc.forecast_origin(other, ModelSpec(), t, n_paths=100, seed=5, valuation=vs)
    assert a["valuation"] == b["valuation"]
    for asset in ("us_equity", "nasdaq", "gold"):
        assert a["assets"][asset]["analog"]["p50"] == pytest.approx(b["assets"][asset]["analog"]["p50"], abs=1e-12)


# ---------------------------------------------------------------- accounting identity

@needs_panel
@pytest.mark.parametrize("income", ["constant", "path"])
def test_price_log_return_is_dividend_growth_plus_change_in_log_pd(panel, income):
    m, db, a = _window(panel, "2014-12")
    vm = fit_valuation(db, a, kappa=0.15, income=income)
    cma = capital_market_assumptions(db, a, method="valuation_state")
    plan = [24, 36]
    s = simulate(db, a, _prior(db, plan), plan, n_paths=200, seed=7, cma=cma, valuation=vm, **config.RESIDUALS)
    j = config.ASSET_IDS.index("us_equity")
    price = s.total[:, :, j].astype(np.float64) - s.income[:, :, j].astype(np.float64)
    lpd = np.log(s.macro["pd"].astype(np.float64))
    assert np.allclose(lpd[:, 0], vm.log_pd0, atol=1e-6)
    lhs = np.log1p(price)
    rhs = vm.g_div[s.idx] + np.diff(lpd, axis=1)
    assert np.abs(lhs - rhs).max() < 2e-5                       # float32 outputs; exact in float64 inside
    # with path income the dividend yield is the anchor's scaled by P/D_0 / P/D_t (beta 1 for the market)
    if income == "path":
        y = s.income[:, :, j].astype(np.float64) * 12.0
        expect = a.income_yield["us_equity"] * np.exp(vm.log_pd0 - lpd[:, :-1])
        assert np.allclose(y, expect, rtol=1e-5)


@needs_panel
def test_pull_lowers_equities_when_expensive_and_leaves_other_assets_alone(panel):
    m, db, a = _window(panel, "1999-12")                         # P/D near its 2000 peak: far above fair value
    cma = capital_market_assumptions(db, a, method="historical")
    plan = [60]
    w = _prior(db, plan)
    v0 = fit_valuation(db, a, kappa=0.0)
    v1 = fit_valuation(db, a, kappa=0.3)
    assert v1.today()["gap_log"] > 0.3
    s0 = simulate(db, a, w, plan, n_paths=300, seed=2, cma=cma, valuation=v0, **config.RESIDUALS)
    s1 = simulate(db, a, w, plan, n_paths=300, seed=2, cma=cma, valuation=v1, **config.RESIDUALS)
    d = (s1.total - s0.total).astype(float).mean(axis=(0, 1)) * 12.0
    ix = config.ASSET_IDS.index
    assert d[ix("us_equity")] < -0.03
    assert d[ix("utilities")] > d[ix("us_equity")]                # beta-scaled (utilities' beta < 1)
    for asset in ("gold", "treasury_10y", "crude_oil", "cash"):
        assert d[ix(asset)] == 0.0
    assert np.median(s1.macro["pd"][:, -1]) < np.median(s0.macro["pd"][:, -1])


# ---------------------------------------------------------------- default behaviour

@needs_panel
def test_mode_off_and_kappa_zero_are_bit_identical(panel):
    m, db, a = _window(panel, "2014-12")
    cma = capital_market_assumptions(db, a, method="historical")
    plan = [24, 24]
    w = _prior(db, plan)
    base = simulate(db, a, w, plan, n_paths=80, seed=9, cma=cma, **config.RESIDUALS)
    off = simulate(db, a, w, plan, n_paths=80, seed=9, cma=cma, valuation=None, **config.RESIDUALS)
    track = simulate(db, a, w, plan, n_paths=80, seed=9, cma=cma, valuation=fit_valuation(db, a, kappa=0.0),
                     **config.RESIDUALS)
    for s in (off, track):
        assert np.array_equal(base.total, s.total) and np.array_equal(base.income, s.income)
        assert all(np.array_equal(base.macro[k], s.macro[k]) for k in base.macro)
    assert "pd" not in base.macro and "pd" in track.macro
    assert valuation_settings(None) is None and valuation_settings({"enabled": False}) is None
    assert not config.VALUATION["enabled"]                       # the engine default (analysis/valuation.py)


@needs_panel
def test_new_columns_do_not_change_the_default_simulation(panel):
    """A panel without the new columns (as built before them) simulates exactly the same paths."""
    old = panel.drop(columns=list(NEW_COLS))
    out = []
    for P in (panel, old):
        m, db, a = _window(P, "2019-12")
        cma = capital_market_assumptions(db, a, method="valuation_adjusted")
        out.append(simulate(db, a, _prior(db, [36]), [36], n_paths=60, seed=1, cma=cma, **config.RESIDUALS))
    assert np.array_equal(out[0].total, out[1].total)


@needs_panel
def test_runner_default_tracks_pd_without_changing_returns():
    from scenariolab.context import EngineContext
    from scenariolab.scenarios import runner
    ctx = EngineContext.load(selection=False)
    spec = {"name": "测试", "phases": [{"name": "常态", "months": 36, "views": [{"var": "s_cpi_yoy", "mean": 3.0}]}]}
    d0, s0 = runner.run_spec(ctx, spec, n_paths=60, seed=3)        # default: pull off, P/D tracked at kappa 0
    saved = runner.VALUATION
    try:
        runner.VALUATION = {"enabled": False}
        d_off, s_off = runner.run_spec(ctx, spec, n_paths=60, seed=3)
        runner.VALUATION = va.track_settings("valuation_adjusted")
        d1, s1 = runner.run_spec(ctx, spec, n_paths=60, seed=3)
        runner.VALUATION = {"enabled": True, "kappa": 0.1, "cma": "valuation_state"}
        d2, s2 = runner.run_spec(ctx, spec, n_paths=60, seed=3)
    finally:
        runner.VALUATION = saved
    assert "pd" not in d_off["macro"] and "pd" not in s_off.macro
    assert np.array_equal(s0.total, s_off.total) and np.array_equal(s0.total, s1.total)   # same paths, plus the P/D
    assert "pd" in d0["macro"] and "pd_fair" not in d0["macro"]   # no fair value is in use, so none is shown
    assert len(d0["macro"]["pd"]["fan"]["p50"]) == config.HORIZON_MONTHS + 1
    assert d0["macro"]["pd"]["fan"]["p50"] == d1["macro"]["pd"]["fan"]["p50"]
    assert runner.headline(d0)["pd_end_p50"] == d0["macro"]["pd"]["fan"]["p50"][-1]
    assert {"pd", "pd_fair"} <= set(d2["macro"]) and not np.array_equal(s0.total, s2.total)


@needs_panel
def test_contexts_sharing_a_month_database_keep_their_own_cma():
    """Regression (runner.valuation_for): the kappa-0 state is cached per MonthDB, the CMA never is. A context
    that shares the db but carries another CMA (the ablation's historical-CMA scenario run) gets its own drift;
    the cache once handed it the first context's CMA, so both runs returned the same paths."""
    from dataclasses import replace

    from scenariolab.context import EngineContext
    from scenariolab.scenarios import runner
    ctx = EngineContext.load(selection=False)
    hist = replace(ctx, cma=capital_market_assumptions(ctx.db, ctx.anchor, method="historical"))
    assert hist.db is ctx.db and ctx.cma["method"] == "valuation_adjusted" and hist.cma["method"] == "historical"
    spec = {"name": "测试", "phases": [{"name": "常态", "months": 36, "views": [{"var": "s_cpi_yoy", "mean": 3.0}]}]}
    _, s0 = runner.run_spec(ctx, spec, n_paths=60, seed=3)        # caches the state for this db first
    _, s1 = runner.run_spec(hist, spec, n_paths=60, seed=3)
    (c0, v0), (c1, v1) = runner.valuation_for(ctx), runner.valuation_for(hist)
    assert c0 is ctx.cma and c1 is hist.cma and v0 is v1           # own CMA, shared kappa-0 state
    assert not np.array_equal(s0.total, s1.total)
    # the drift is the only difference: every month of us_equity differs by the gap in CMA adjustments / 12
    i = list(s0.assets).index("us_equity")
    gap = s0.total[..., i].astype(float) - s1.total[..., i].astype(float)
    adj0, adj1 = ctx.cma["adjustment"]["us_equity"], (hist.cma.get("adjustment") or {}).get("us_equity", 0.0)
    assert abs(adj0 - adj1) > 1e-3
    assert np.allclose(gap, (adj0 - adj1) / 12.0, atol=1e-5)


# ---------------------------------------------------------------- reproducibility

@needs_panel
def test_valuation_simulation_is_reproducible(panel):
    m, db, a = _window(panel, "2014-12")
    vm = fit_valuation(db, a, kappa="ecm", income="path")
    cma = capital_market_assumptions(db, a, method="valuation_state")
    plan = [24, 24]
    w = _prior(db, plan)
    run = lambda seed: simulate(db, a, w, plan, n_paths=50, seed=seed, cma=cma, valuation=vm,   # noqa: E731
                                **config.RESIDUALS)
    x, y, z = run(5), run(5), run(6)
    assert np.array_equal(x.total, y.total) and np.array_equal(x.macro["pd"], y.macro["pd"])
    assert not np.array_equal(x.total, z.total)
    again = fit_valuation(db, a, kappa="ecm", income="path")
    assert again.coef == vm.coef and again.kappa == vm.kappa


# ---------------------------------------------------------------- selection rule

def _entry(vid, adoptable=True, val=0.04, test=0.05, cov_v=0.8, cov_t=0.95, se=0.001, fan=1.0, fval=0.0, ftest=0.0,
           ic_v=0.0, ic_t=0.0):
    return {"id": vid, "name": vid, "adoptable": adoptable, "validation_mae": val, "test_mae": test,
            "validation_coverage": cov_v, "test_coverage": cov_t, "validation_gain_se": se, "test_gain_se": se,
            "validation_gain": 0.04 - val, "test_gain": 0.05 - test, "fan_width_ratio_min": fan,
            "forecast_skill": {"val": {"crps_skill": fval, "rank_ic_delta": ic_v},
                               "test": {"crps_skill": ftest, "rank_ic_delta": ic_t}}}


def test_selection_rule():
    base = _entry("baseline", adoptable=False)
    # all candidates worse: keep the engine
    sel = va.select([base, _entry("a", val=0.045), _entry("b", val=0.041)])
    assert sel["chosen"] == "baseline" and not sel["adopt"] and sel["best_candidate"] == "b"
    # the best validation candidate is picked among eligible ones; guards exclude the others
    sel = va.select([base, _entry("narrow", val=0.030, fan=0.80), _entry("cov", val=0.031, cov_v=0.70),
                     _entry("fc", val=0.032, fval=-0.05), _entry("ok", val=0.034), _entry("small", val=0.0395)])
    assert sel["chosen"] == "ok" and sel["adopt"] and sel["eligible"] == ["ok"]
    assert not sel["checks"]["narrow"]["forward_fan"] and not sel["checks"]["cov"]["coverage"]
    assert not sel["checks"]["fc"]["forecast_val"] and not sel["checks"]["small"]["gain"]
    # one test confirmation, no second pick
    sel = va.select([base, _entry("best", val=0.030, test=0.06), _entry("second", val=0.032, test=0.04)])
    assert sel["chosen"] == "best" and not sel["adopt"]
    sel = va.select([base, _entry("best", val=0.030, ftest=-0.05)])
    assert sel["chosen"] == "best" and not sel["adopt"]
    # a noisy gain (within 2 standard errors) is not enough
    sel = va.select([base, _entry("noisy", val=0.037, se=0.002)])
    assert sel["chosen"] == "baseline"


# ---------------------------------------------------------------- review fixes (economics, honesty, generated texts)

def test_theory_check_flags_coefficients_against_theory():
    th = va.theory_check({"real_y10": {"beta": 0.05, "t": 2.5}, "profit_share": {"beta": 0.03, "t": 1.0},
                          "infl_expect": {"beta": -0.2, "t": -6.0}})
    assert set(th) == {"real_y10", "profit_share"}                  # inflation has no theory sign here
    assert not th["real_y10"]["consistent"] and th["real_y10"]["significant_wrong"]
    assert th["real_y10"]["fair_pd_change_per_unit"] == pytest.approx(np.expm1(0.05))
    assert th["profit_share"]["consistent"] and not th["profit_share"]["significant_wrong"]


def _synthetic_levels(flip: bool) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    idx = pd.date_range("1973-01-01", "2026-09-01", freq="MS")
    n = len(idx)
    walk = lambda s: np.cumsum(rng.normal(0, s, n))                    # noqa: E731
    infl = 4 + walk(0.15)
    ry, ps = 2 + walk(0.1), 8 + walk(0.05)
    d = np.where((idx > pd.Timestamp(va.STABILITY_SPLIT)) & flip, 0.2, -0.2)
    lpd = 4.0 + d * (infl - 4) + 0.03 * ps + rng.normal(0, 0.02, n)
    return pd.DataFrame({"s_div_yield": 100 / np.exp(lpd), "s_real_y10": ry, "s_profit_share_rt": ps,
                         "s_infl_expect": infl, "p_us_equity": rng.normal(0.005, 0.04, n)}, index=idx)


def test_stability_reports_non_overlapping_halves_and_sign_flips():
    feats = ("real_y10", "profit_share", "infl_expect")
    flip = va.stability(_synthetic_levels(True), feats)
    same = va.stability(_synthetic_levels(False), feats)
    assert [h["end"] for h in flip["halves"]][0] == va.STABILITY_SPLIT
    assert flip["halves"][1]["start"] > va.STABILITY_SPLIT
    assert "infl_expect" in flip["sign_flips"] and "infl_expect" not in same["sign_flips"]
    assert all(r["n_obs"] == va.ROLLING_YEARS * 12 for r in flip["rolling"])
    c = flip["halves"][1]["coefficients"]["infl_expect"]
    assert c["beta"] == pytest.approx(0.2, abs=0.02)


def test_real_time_gap_and_persistence_use_only_the_past():
    feats = ("real_y10", "profit_share", "infl_expect")
    P = _synthetic_levels(False)
    Q = P.copy()
    late = Q.index > pd.Timestamp("2004-12-01")
    Q.loc[late] = Q.loc[late].to_numpy() * np.random.default_rng(1).uniform(0.5, 1.5, Q.loc[late].shape)
    g1, g2 = va.real_time_gap(P, feats), va.real_time_gap(Q, feats)
    early = g1.index <= pd.Timestamp("2004-12-01")
    # the first fit is the first December with REAL_TIME_FIRST_YEARS of months; its gaps start the month after
    enough = P.index.min() + pd.DateOffset(years=va.REAL_TIME_FIRST_YEARS) - pd.DateOffset(months=1)
    first_dec = P.index[(P.index.month == 12) & (P.index >= enough)][0]
    assert g1.index.min() == first_dec + pd.DateOffset(months=1)
    assert np.array_equal(g1[early].to_numpy(), g2[g2.index <= pd.Timestamp("2004-12-01")].to_numpy())
    assert not np.allclose(g1[~early].to_numpy(), g2[g2.index > pd.Timestamp("2004-12-01")].to_numpy())
    pers = va.persistence(P, feats)
    assert pers["ecm_real_time"]["n"] > 0 and pers["oos_r2_price_12m"]["n"] > 0
    assert 0 < pers["log_pd_ar1_monthly"] < 1 and pers["variance_ratio"]["months"] == config.HORIZON_MONTHS


def _text_inputs(adopt: bool, largest: str):
    coef = {"const": {"beta": 4.0, "t": 10.0}, "real_y10": {"beta": 0.05, "t": 1.7},
            "profit_share": {"beta": 0.04, "t": 1.2}, "infl_expect": {"beta": -0.2, "t": -6.0}}
    reg = {"coefficients": coef, "sample": ["1973-02", "2026-09"], "formula": "f", "r2": 0.7, "by_window": [],
           "theory_check": va.theory_check({k: v for k, v in coef.items() if k != "const"})}
    drivers = {"real_y10": 0.01, "profit_share": 0.02, "infl_expect": 0.03}
    drivers[largest] = 0.5
    today = {"date": "2026-09", "pd": 97.0, "pd_fair": 40.0, "pd_fair_at_sample_means": 45.0, "gap_log": 0.9,
             "gap_log_alternatives": {"a": 0.7, "b": 1.1}, "gap_log_range": [0.7, 1.1], "pd_percentile": 0.99,
             "cape": 40.0, "cape_percentile": 0.98, "kappa": 0.1, "implied_annual_drag": -0.09,
             "implied_annual_drag_15y": -0.04, "horizon_years": 15.0, "drivers": drivers}
    fs = {"val": {"crps_skill": 0.0, "rank_ic_delta": 0.0}, "test": {"crps_skill": 0.0, "rank_ic_delta": 0.0},
          "all": {"crps_skill": 0.0, "rank_ic_delta": 0.0}}
    var = {"id": "vs_k10", "name": "k10", "adoptable": True, "validation_gain": 0.01, "test_gain": 0.01,
           "validation_mae": 0.03, "test_mae": 0.05, "settings": {"kappa": 0.1}, "fan_width_ratio_min": 1.0,
           "forecast_skill": fs, "forecast": {"all": {"n": 7}}}
    base = {**var, "id": "baseline", "adoptable": False, "settings": None}
    sel = {"adopt": adopt, "reason": "理由。", "checks": {"vs_k10": {"gain": adopt}}}
    facs = {"candidates": [], "min_gain": 0.003, "final": {"describe": "10 个因子"}}
    h = {"us_equity_cagr_p50": 0.06, "nasdaq_cagr_p50": 0.07, "gold_cagr_p50": 0.04, "pd_end_p50": 80.0, "prob": 0.5}
    scen = {"baseline": {"a": h, "b": h}, "baseline_prob_weighted": h, "modes": {"other": {}}}
    return reg, today, sel, {"baseline": base, "vs_k10": var}, facs, scen


@pytest.mark.parametrize("adopt,largest,name", [(False, "real_y10", "实际利率"), (True, "infl_expect", "通胀预期")])
def test_summary_text_is_generated_from_its_inputs(adopt, largest, name):
    reg, today, sel, by_id, facs, scen = _text_inputs(adopt, largest)
    txt = va.summary_text(reg, today, sel, by_id, facs, scen)
    assert f"其中{name}的贡献最大" in txt
    assert ("回测不支持" in txt) != adopt and ("采用的设定" in txt) == adopt
    assert "公允值低于各变量取样本均值时的 45 倍" in txt                 # 40 < 45: the direction is computed
    assert "对 2 个未来情景" in txt and "13 个" not in txt
    assert "实际利率的系数为正，与理论" in txt and "公允 P/D 反而上升 5.1%" in txt
    assert "7 个起点" in txt and "42 个起点" not in txt
    assert "缺口在 +0.70 到 +1.10 之间" in txt


@needs_panel
def test_notes_are_computed_from_the_data():
    from scenariolab.context import EngineContext
    ctx = EngineContext.load(selection=False)
    vm = fit_valuation(ctx.db, ctx.anchor, kappa=0.1)
    reg = va.regression_evidence(ctx, vm)
    roic = va.roic_evidence(ctx, vm.features)
    out = va.notes(ctx, vm, reg, roic, "2023-09")
    text = "".join(out)
    assert f"相关为 {roic['corr_with_profit_share']:.2f}" in text
    assert f"半衰期 {config.VALUATION['infl_half_life_months']} 个月" in text
    assert f"+ {config.VALUATION['release_lag_months']} 个月" in text
    P = ctx.db.frame.loc[:"2023-09", ["s_div_yield", "s_cape"]].dropna()
    corr = np.corrcoef(np.log(100 / P["s_div_yield"]), np.log(P["s_cape"]))[0, 1]
    assert f"对数相关为 {corr:.2f}" in text
    assert reg["theory_check"]["real_y10"]["consistent"] == (reg["coefficients"]["real_y10"]["beta"] < 0)
    assert len(reg["stability"]["halves"]) == 2


@needs_panel
def test_profit_share_drift_moves_the_valuation_state(panel):
    m, db, a = _window(panel, "2014-12")
    cma = capital_market_assumptions(db, a, method="historical")
    plan = [36]
    w = _prior(db, plan)
    vm = fit_valuation(db, a, kappa=0.0)
    s0 = simulate(db, a, w, plan, n_paths=50, seed=2, cma=cma, valuation=vm, **config.RESIDUALS)
    s1 = simulate(db, a, w, plan, n_paths=50, seed=2, cma=cma, valuation=vm,
                  overrides=[{"factor_drift": {"profit_share": 1.2}}], **config.RESIDUALS)
    dps = s1.macro["profit_share"].astype(float) - s0.macro["profit_share"].astype(float)
    assert np.allclose(dps[:, -1], 1.2 * 3.0, atol=1e-4) and np.allclose(dps[:, 0], 0.0)
    dfair = np.log(s1.macro["pd_fair"].astype(float)) - np.log(s0.macro["pd_fair"].astype(float))
    assert np.allclose(dfair[:, -1], vm.coef["profit_share"] * 3.6, atol=1e-4)
    assert np.array_equal(s0.total, s1.total)                     # kappa 0: tracked only, returns unchanged
