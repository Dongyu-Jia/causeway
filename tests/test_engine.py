import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.data.panel import par_bond_return
from scenariolab.engine.ep import apply_views, effective_number, entropy_pool, weighted_corr
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
