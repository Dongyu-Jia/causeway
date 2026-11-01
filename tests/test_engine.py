import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.data.panel import par_bond_return
from scenariolab.engine.ep import apply_views, entropy_pool, weighted_corr
from scenariolab.engine.factor_model import AssetFactorModel, ModelSpec, design_matrix, features_from_paths
from scenariolab.engine.sampler import block_start_probs, sample_phases


def test_entropy_pool_satisfies_views_and_stays_close():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    p = np.full(x.size, 1 / x.size)
    q, ok, viol = entropy_pool(p, A_eq=x[None, :], b_eq=np.array([0.5]))
    assert ok and viol < 1e-6
    assert abs(q @ x - 0.5) < 1e-6
    assert q.min() >= 0 and abs(q.sum() - 1) < 1e-12
    # no views -> prior
    q0, ok0, _ = entropy_pool(p)
    assert ok0 and np.allclose(q0, p)


def test_entropy_pool_inequality_inactive_and_active():
    x = np.linspace(-1, 1, 501)
    p = np.full(x.size, 1 / x.size)
    q, ok, _ = entropy_pool(p, A_ineq=-x[None, :], b_ineq=np.array([-0.3]))   # E[x] >= 0.3
    assert ok and q @ x >= 0.3 - 1e-6
    q2, ok2, _ = entropy_pool(p, A_ineq=x[None, :], b_ineq=np.array([0.5]))   # E[x] <= 0.5, already true
    assert ok2 and np.allclose(q2, p, atol=1e-8)


def test_apply_views_reports_and_clips():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000)})
    res = apply_views(frame, [{"var": "a", "shift_sd": 1.0}, {"var": "b", "mean": 99.0}])
    a_view, b_view = res.views
    assert abs(a_view["posterior_mean"] - (a_view["prior_mean"] + a_view["prior_sd"])) < 1e-4
    assert b_view["clipped"] and b_view["target"] < 99
    assert 0 < res.ens <= 1000 and res.kl >= 0


def test_weighted_corr_matches_numpy_for_uniform_weights():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 3))
    X[:, 1] += 0.5 * X[:, 0]
    c = weighted_corr(X, np.ones(500))
    assert np.allclose(c, np.corrcoef(X, rowvar=False), atol=1e-10)


def test_block_sampler_respects_weights():
    w = np.zeros(100)
    w[40:50] = 1.0
    probs = block_start_probs(w / w.sum(), 5)
    assert probs.size == 96 and abs(probs.sum() - 1) < 1e-12
    idx = sample_phases([w / w.sum()], [60], n_paths=50, block=5, seed=3)
    assert idx.shape == (50, 60)
    assert idx.min() >= 36 and idx.max() <= 53      # blocks overlap the weighted region


def test_par_bond_return_zero_when_yield_unchanged():
    y = pd.Series([4.0, 4.0, 5.0])
    price, income = par_bond_return(y.shift(1), y)
    assert abs(price.iloc[1]) < 1e-12
    assert price.iloc[2] < -0.07 and price.iloc[2] > -0.09     # ~duration 7.9
    assert abs(income.iloc[1] - 0.04 / 12) < 1e-12


def _synthetic_panel(n=400, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    P = pd.DataFrame(index=idx)
    for f in config.CORE_FACTOR_IDS:
        P[f"f_{f}"] = rng.normal(scale=0.01, size=n)
    P["s_cpi_yoy"] = 3 + rng.normal(size=n)
    for a in config.ASSET_IDS:
        P[f"p_{a}"] = 0.005 + 0.8 * P["f_oil"] - 1.5 * P["f_y10"] + rng.normal(scale=0.01, size=n)
    return P


def test_linear_model_recovers_betas():
    P = _synthetic_panel()
    m = AssetFactorModel(ModelSpec()).fit(P, "1990-01", None)
    assert abs(m.coef.loc["us_equity", "oil"] - 0.8) < 0.15
    assert abs(m.coef.loc["us_equity", "y10"] + 1.5) < 0.15
    resid = m.residuals(P)["us_equity"].dropna()
    assert abs(resid.mean()) < 1e-3


def test_features_from_paths_match_design_matrix_with_lags():
    P = _synthetic_panel(60)
    spec = ModelSpec(lags=2)
    X = design_matrix(P, spec).iloc[2:].to_numpy()
    base = P[[f"f_{f}" for f in spec.factors]].to_numpy()
    feats = features_from_paths(spec, base[None, 2:, :], base[:2], None)[0]
    assert np.allclose(feats, X)


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
def test_simulation_on_real_panel_shapes_and_sanity():
    from scenariolab.context import EngineContext
    from scenariolab.engine.simulate import simulate
    ctx = EngineContext.load(selection=False)
    out = simulate(ctx.db, ctx.anchor, [ctx.db.prior()], [120], n_paths=200, seed=1, cma=ctx.cma)
    assert out.total.shape == (200, 120, len(config.ASSET_IDS))
    assert np.isfinite(out.total).all()
    assert (out.macro["ff"] >= 0).all() and (out.macro["y10"] > 0).all()
    cash = out.total[:, :, config.ASSET_IDS.index("cash")]
    assert cash.mean() * 12 < 0.12


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
def test_runner_end_to_end_small():
    from scenariolab.context import EngineContext
    from scenariolab.scenarios.runner import run_spec
    ctx = EngineContext.load(selection=False)
    spec = {"name": "测试", "phases": [{"name": "滞胀", "months": 24,
            "views": [{"var": "s_cpi_yoy", "mean": 7.0}, {"var": "s_cpi_yoy", "mean": 99}]}]}
    detail, sim = run_spec(ctx, spec, n_paths=100, seed=2)
    assert detail["spec"]["phases"][-1]["months"] == config.HORIZON_MONTHS - 24
    assert detail["validation"]["clipped"]
    assert len(detail["labels"]) == config.HORIZON_MONTHS + 1
    c = detail["assets"]["us_equity"]["candles_monthly"]
    assert len(c) == config.HORIZON_MONTHS and len(detail["assets"]["us_equity"]["candles_yearly"]) >= 15
    assert detail["phases"][0]["ep"]["era_weights"]


def test_pooled_blocks_deliver_the_view_to_the_drawn_months():
    from scenariolab.engine.pooling import month_frequency, pool_phase, start_values
    from scenariolab.engine.sampler import sample_phase
    rng = np.random.default_rng(4)
    T = 400
    x = rng.standard_t(3, size=T) * 0.02                       # fat-tailed monthly flow
    x[200] = 0.5                                               # one spike month, like 2020-04
    frame = pd.DataFrame({"x": x}, index=pd.date_range("1980-01-01", periods=T, freq="MS"))
    prior = np.full(T, 1 / T)
    for months in (24, 20, 6, 4):
        target = float(np.percentile(start_values(x, min(6, months), months), 80))
        pool = pool_phase(frame, [{"var": "x", "mean": target}], prior, months, block=6, min_ens=10)
        assert pool.shrink == 1.0 and pool.blocks.block == min(6, months)
        w = pool.month_weights
        assert w.sum() == pytest.approx(1.0) and w @ x == pytest.approx(target, abs=1e-9)
        assert np.allclose(w, month_frequency(pool.blocks.probs, pool.blocks.block, months, T))
        idx = sample_phase(pool.blocks, months, 40000, 6, np.random.default_rng(2))
        assert abs(x[idx].mean() - target) < 4 * x.std() / np.sqrt(idx.size / 6)


def test_pool_phase_shrinks_jointly_impossible_views():
    from scenariolab.engine.pooling import pool_phase
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    frame = pd.DataFrame({"a": x, "b": x + 0.05 * rng.normal(size=400)})
    prior = np.full(400, 1 / 400)
    pool = pool_phase(frame, [{"var": "a", "mean": 1.0}, {"var": "b", "mean": -1.0}], prior, 12, min_ens=30)
    assert 0.0 <= pool.shrink < 1.0 and pool.shrink_reason == "infeasible"
    assert pool.ep.converged and pool.ens >= 30
    easy = pool_phase(frame, [{"var": "a", "mean": 0.1}], prior, 12, min_ens=30)
    assert easy.shrink == 1.0 and easy.month_weights @ x == pytest.approx(0.1, abs=1e-6)
    thin = pool_phase(frame, [{"var": "a", "mean": 0.6}], prior, 12, min_ens=300)   # feasible but too narrow
    assert thin.shrink < 1.0 and thin.shrink_reason == "thin" and thin.ens >= 300 - 1e-6


def test_apply_views_shrink_moves_every_kind_of_view_toward_the_prior():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"a": rng.normal(size=800)})
    full = apply_views(frame, [{"var": "a", "mean": 0.5, "vol_mult": 1.5}])
    half = apply_views(frame, [{"var": "a", "mean": 0.5, "vol_mult": 1.5}], shrink=0.5)
    none = apply_views(frame, [{"var": "a", "mean": 0.5, "min": 0.4}], shrink=0.0)
    prior_mean = full.views[0]["prior_mean"]
    assert half.views[0]["posterior_mean"] == pytest.approx(prior_mean + 0.5 * (0.5 - prior_mean), abs=1e-6)
    assert half.views[0]["posterior_sd"] < full.views[0]["posterior_sd"]
    assert np.allclose(none.q, np.full(800, 1 / 800), atol=1e-9)


def test_rare_months_are_down_weighted_in_the_prior():
    from scenariolab.context import EngineContext
    if not config.PANEL_PATH.exists():
        pytest.skip("panel not built")
    db = EngineContext.load(selection=False).db
    p = db.prior()
    lock = (db.dates >= "2020-03-01") & (db.dates <= "2020-08-01")
    assert lock.sum() == 6 and p[lock].max() == pytest.approx(0.25 * p[~lock].max())


# ---------------------------------------------------------------- engine review fixes (real panel)

@pytest.fixture(scope="module")
def real_ctx():
    if not config.PANEL_PATH.exists():
        pytest.skip("panel not built")
    from scenariolab.context import EngineContext
    return EngineContext.load(selection=False)


def _one_phase(ctx, views, months, n=3000, seed=5, overrides=None, reversion=None):
    from scenariolab.engine.pooling import pool_phase
    from scenariolab.engine.simulate import simulate, with_oil_lock
    pool = pool_phase(ctx.db.frame, views, ctx.db.prior(), months, rare=ctx.db.rare)
    ovs = with_oil_lock([views], overrides)
    return pool, simulate(ctx.db, ctx.anchor, [pool.blocks], [months], n_paths=n, seed=seed, cma=ctx.cma,
                          overrides=ovs, reversion=reversion, **config.RESIDUALS)


def test_oil_views_are_delivered_and_not_reverted(real_ctx):
    pool, sim = _one_phase(real_ctx, [{"var": "f_oil", "mean_annual": 0.15}], 36)
    dlog = np.diff(np.log(sim.macro["oil"].astype(float)), axis=1).mean() * 12.0
    assert dlog == pytest.approx(0.15, abs=0.02)                       # the view, not the view minus reversion
    _, free = _one_phase(real_ctx, [], 36)
    assert np.diff(np.log(free.macro["oil"].astype(float)), axis=1).mean() * 12.0 < 0.0   # today's high oil reverts


def test_oil_target_shift_moves_the_long_run_real_oil_anchor(real_ctx):
    from scenariolab.engine.simulate import DEFAULT_REVERSION
    shift, k = -0.35, DEFAULT_REVERSION["kappa_oil"]
    assert DEFAULT_REVERSION["oil_target_shift"] == 0.0
    _, base = _one_phase(real_ctx, [], 180, n=200, seed=7)
    _, low = _one_phase(real_ctx, [], 180, n=200, seed=7, reversion={"oil_target_shift": shift})
    gap = np.log(low.macro["oil"].astype(float)) - np.log(base.macro["oil"].astype(float))
    t = np.arange(gap.shape[1])
    # same draws, so the only difference is the pull toward the shifted target: shift * (1 - (1 - kappa)^t)
    assert np.allclose(gap, shift * (1.0 - (1.0 - k) ** t)[None, :], atol=1e-4)
    # a phase that states its own oil path is not pulled toward any target, shifted or not
    views = [{"var": "f_oil", "mean_annual": 0.1}]
    _, a = _one_phase(real_ctx, views, 36, n=100, seed=7)
    _, b = _one_phase(real_ctx, views, 36, n=100, seed=7, reversion={"oil_target_shift": shift})
    assert np.array_equal(a.macro["oil"], b.macro["oil"])


def test_oil_target_shift_is_bounded_by_the_spec_checker(real_ctx):
    from scenariolab.scenarios.schema import REVERSION_BOUNDS, validate_spec
    lo, hi = REVERSION_BOUNDS["oil_target_shift"]
    assert lo < 0.0 < hi
    spec = {"name": "测试", "phases": [{"name": "一", "months": 12, "views": []}]}
    for asked, used in ((-2.0, lo), (1.0, hi), (-0.35, -0.35)):
        clean, report = validate_spec({**spec, "reversion": {"oil_target_shift": asked}}, real_ctx.db.frame)
        assert clean["reversion"]["oil_target_shift"] == pytest.approx(used)
        # a clipped reversion value is reported like every other bound; an in-bounds one is not
        assert [c["var"] for c in report["clipped"]] == (["reversion.oil_target_shift"] if asked != used else [])
    clean, report = validate_spec({**spec, "reversion": {"no_such_key": 1.0}}, real_ctx.db.frame)
    assert clean["reversion"] == {} and any("no_such_key" in w for w in report["warnings"])


def test_rare_months_are_drawn_at_their_weight(real_ctx):
    from scenariolab.engine.pooling import uncond_frequency
    db = real_ctx.db
    f = uncond_frequency(db.prior(), db.rare)
    flat = uncond_frequency(np.full(len(db), 1.0 / len(db)))
    apr = int(np.flatnonzero(db.dates == pd.Timestamp("2020-04-01"))[0])
    assert f[apr] / flat[apr] == pytest.approx(0.25, abs=0.02)


def test_one_episode_view_sets_are_shrunk(real_ctx):
    # 2021-23 style: high inflation with very low unemployment only occurred in one stretch since 1973
    pool, _ = _one_phase(real_ctx, [{"var": "s_cpi_yoy", "mean": 8.0}, {"var": "s_unrate", "mean": 3.6}], 48, n=50)
    assert pool.shrink < 1.0 and pool.episodes >= 6.0 - 1e-6


def test_confidence_does_not_bypass_the_support_guard(real_ctx):
    from scenariolab.engine.pooling import pool_phase
    views = [{"var": "s_cpi_yoy", "mean": 8.0}, {"var": "s_unrate", "mean": 3.7}]
    full = pool_phase(real_ctx.db.frame, views, real_ctx.db.prior(), 48, rare=real_ctx.db.rare)
    half = pool_phase(real_ctx.db.frame, views, real_ctx.db.prior(), 48, confidence=0.5, rare=real_ctx.db.rare)
    assert half.shrink <= full.shrink + 1e-9


def test_vol_mult_scales_monthly_volatility(real_ctx):
    from scenariolab.engine.pooling import pool_phase
    db = real_ctx.db
    x = db.frame["f_y10"].to_numpy(float)
    for k in (0.5, 2.0):
        pool = pool_phase(db.frame, [{"var": "f_y10", "vol_mult": k}], db.prior(), 36, rare=db.rare)
        base = pool.prior_month_weights
        w = pool.month_weights
        var_p = base @ x ** 2 - (base @ x) ** 2
        var_q = w @ x ** 2 - (base @ x) ** 2               # around the prior mean, as the view states
        assert var_q / var_p == pytest.approx(k * k, rel=0.03)


def test_rate_drift_is_delivered_as_asked(real_ctx):
    rev = {"kappa_ff": 0.0}
    _, a = _one_phase(real_ctx, [], 24, n=500, reversion=rev)
    _, b = _one_phase(real_ctx, [], 24, n=500, reversion=rev, overrides=[{"factor_drift": {"ff": 1.2}}])
    first = (b.macro["ff"][:, 1] - a.macro["ff"][:, 1]).astype(float)   # same draws, same level at t=0
    off_floor = (a.macro["ff"][:, 1] > 0.0) & (b.macro["ff"][:, 1] > 0.0)
    assert np.allclose(first[off_floor], 0.1, atol=1e-5)               # +1.2pp a year = +0.1 a month, not level-scaled


def test_long_run_spread_settles_at_its_target(real_ctx):
    _, sim = _one_phase(real_ctx, [], 180, n=2000)
    late = sim.macro["spread"][:, -60:].astype(float)
    assert float(late.mean()) == pytest.approx(2.0, abs=0.15)          # the reversion target is a mean
    assert float(np.median(late)) < float(late.mean())                 # spreads are right-skewed


def test_cma_targets_are_what_an_unconditional_path_delivers(real_ctx):
    _, sim = _one_phase(real_ctx, [], 180, n=3000)
    j = config.ASSET_IDS.index("gold")
    price = sim.total[:, :, j].astype(float) - sim.income[:, :, j].astype(float)
    log_real = (np.log1p(price) - np.diff(np.log(sim.macro["cpi"].astype(float)), axis=1)).mean() * 12.0
    assert abs(log_real) < 0.01                                       # gold keeps pace with inflation (gold_reversion 0)


def test_attribution_is_opt_in_and_changes_no_simulated_number(real_ctx):
    from scenariolab.engine.simulate import simulate
    kw = dict(n_paths=60, seed=2, cma=real_ctx.cma, overrides=[{"asset_drift": {"nasdaq": -0.04}}], **config.RESIDUALS)
    a = simulate(real_ctx.db, real_ctx.anchor, [real_ctx.db.prior(), real_ctx.db.prior()], [18, 42], **kw)
    b = simulate(real_ctx.db, real_ctx.anchor, [real_ctx.db.prior(), real_ctx.db.prior()], [18, 42], attribution=True, **kw)
    assert a.attribution is None and b.attribution is not None
    assert np.array_equal(a.total, b.total) and np.array_equal(a.income, b.income) and np.array_equal(a.idx, b.idx)
    assert all(np.array_equal(a.macro[k], b.macro[k]) for k in a.macro)
    att = b.attribution
    assert att.values.shape == (60, 2, len(config.ASSET_IDS), len(att.terms))
    assert np.abs(att.values.sum(axis=-1) - att.total).max() < 1e-12
    nq = config.ASSET_IDS.index("nasdaq")
    assert np.allclose(att.get("overrides")[:, :, nq], [-0.04, 0.0])
