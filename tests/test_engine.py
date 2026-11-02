import json

import numpy as np
import pandas as pd
import pytest

from scenariolab import config, jsonio
from scenariolab.data.panel import par_bond_return
from scenariolab.engine.ep import apply_views, entropy_pool, weighted_corr
from scenariolab.engine.factor_model import (MIN_ASSET_WINDOW, MODEL_ASSETS, AssetFactorModel, ModelSpec,
                                             asset_columns, asset_feature_names, base_columns, design_matrix,
                                             feature_names, features_from_paths)
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


# ---------------------------------------------------------------- per-asset equations (FACTORS_DESIGN §3.2-3.3)

# the expected final model of the missing-factor round: 11 common factors, energy + metals vs oil, gold from 1982
F_SPEC = ModelSpec(factors=("oil", "ip", "cpi_goods", "cpi_services", "ff", "y10", "spread", "usd", "unrate", "gz",
                            "tp10_x_core4"),
                   asset_factors={"energy": ["metals_oil"]}, asset_start={"gold": "1982-01"},
                   name="base+add_gz+add_tp10_x_core4+drop_labor_share+energy_metals_oil+gold_window")
EQ_SPEC = ModelSpec(asset_factors={"energy": ["x"]}, asset_start={"gold": "2000-01"}, name="eq")


def _equations_panel(n=420, seed=6):
    """_synthetic_panel (1990-01 on) plus a factor x that moves energy only and a gold oil beta that changes in 2000."""
    P = _synthetic_panel(n, seed)
    P["f_x"] = np.random.default_rng(seed + 1).normal(scale=0.01, size=n)
    P["p_energy"] += 0.6 * P["f_x"]
    early = P.index < pd.Timestamp("2000-01-01")
    P.loc[early, "p_gold"] += 2.0 * P.loc[early, "f_oil"]
    return P


def test_model_spec_round_trip_with_asset_factors_and_starts():
    assert F_SPEC.asset_factors == (("energy", ("metals_oil",)),) and F_SPEC.asset_start == (("gold", "1982-01"),)
    d = F_SPEC.to_dict()
    assert d["asset_factors"] == {"energy": ["metals_oil"]} and d["asset_start"] == {"gold": "1982-01"}
    assert list(d)[-3:] == ["asset_factors", "asset_start", "name"]
    back = ModelSpec.from_dict(json.loads(json.dumps(d)))
    assert back == F_SPEC and hash(back) == hash(F_SPEC)
    # lists of pairs, repeated or unsorted ids, empty entries and full dates normalise to the same spec
    loose = ModelSpec.from_dict({**d, "asset_factors": [["energy", ["metals_oil", "metals_oil"]], ["gold", []]],
                                 "asset_start": [["gold", "1982-01-01"]]})
    assert loose == F_SPEC
    two = ModelSpec(asset_factors={"industrials": "usd_em", "energy": ["x", "metals_oil"]})
    assert two.asset_factors == (("energy", ("metals_oil", "x")), ("industrials", ("usd_em",)))
    # a model_selection.json written before the two keys existed loads unchanged
    old = {"factors": list(config.CORE_FACTOR_IDS), "lags": 0, "regime": None, "kind": "linear", "ridge": 0.0,
           "window_years": None, "target": "p", "shock_cols": [], "hidden": 16, "name": "base"}
    assert ModelSpec.from_dict(old) == ModelSpec() and ModelSpec.from_dict(old).asset_factors == ()
    assert ModelSpec().to_dict() == {**old, "asset_factors": {}, "asset_start": {}}
    assert ModelSpec().describe() == "10 个因子"
    assert F_SPEC.describe() == "11 个因子、能源股 +1 个专门因子、黄金从 1982 年起估计"
    assert two.describe() == "10 个因子、2 个资产有专门因子"
    assert (ModelSpec(ridge=0.05, asset_start={"gold": "1985-07"}).describe()
            == "10 个因子、岭回归 λ=0.05、黄金从 1985 年 7 月起估计")
    # an extra that is a common factor already adds nothing
    assert ModelSpec(asset_factors={"energy": ["oil"]}).describe() == "10 个因子"
    assert base_columns(ModelSpec(asset_factors={"energy": ["oil"]})) == base_columns(ModelSpec())


def test_each_asset_equation_is_its_own_ols():
    P = _equations_panel()
    m = AssetFactorModel(EQ_SPEC).fit(P, "1990-01", None)
    feats = feature_names(EQ_SPEC)
    assert m.features == feats and feats[-1] == "x" and base_columns(EQ_SPEC)[-1] == "f_x"
    assert asset_columns(EQ_SPEC, "energy") == base_columns(EQ_SPEC)
    assert asset_columns(EQ_SPEC, "gold") == [f"f_{f}" for f in config.CORE_FACTOR_IDS]
    X = design_matrix(P, EQ_SPEC)
    assert list(X.columns) == feats
    for a in m.assets:
        own = asset_feature_names(EQ_SPEC, a)
        rows = P.index >= pd.Timestamp("2000-01-01" if a == "gold" else "1990-01-01")
        Z = np.column_stack([np.ones(rows.sum()), X.loc[rows, own].to_numpy()])
        y = P.loc[rows, f"p_{a}"].to_numpy()
        b = np.linalg.lstsq(Z, y, rcond=None)[0]
        e = y - Z @ b
        se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * (e @ e) / (len(y) - Z.shape[1]))
        assert np.abs(m.coef.loc[a, own].to_numpy() - b[1:]).max() < 1e-12, a
        assert abs(m.intercept[a] - b[0]) < 1e-12, a
        assert np.allclose(m.tstat.loc[a, own].to_numpy(), (b / se)[1:], rtol=1e-9, atol=0), a
        outside = [f for f in feats if f not in own]
        assert (m.coef.loc[a, outside] == 0.0).all() and m.tstat.loc[a, outside].isna().all()
        assert m.stats.n_obs[a] == rows.sum()
        assert m.stats.r2[a] == pytest.approx(1.0 - (e @ e) / ((y - y.mean()) ** 2).sum(), abs=1e-12)
        assert m.stats.resid_sd[a] == pytest.approx(e.std(), abs=1e-12)
    assert asset_feature_names(EQ_SPEC, "energy") == feats and "x" not in asset_feature_names(EQ_SPEC, "us_equity")
    # the start is honoured: gold's own months and its 2000+ oil beta (0.8), not the blend with the 1990s' 2.8
    assert m.asset_start_used == {"gold": "2000-01"} and m.train_start == pd.Timestamp("1990-01-01")
    assert m.asset_train_start["gold"] == "2000-01" and m.asset_train_start["us_equity"] == "1990-01"
    assert abs(m.coef.loc["gold", "oil"] - 0.8) < 0.15
    # predictions are x @ coef.T + intercept over the union; a feature outside an equation contributes 0
    pred = m.predict(P)
    assert np.abs(pred.to_numpy() - (X.to_numpy() @ m.coef.to_numpy().T + m.intercept.to_numpy())).max() < 1e-14
    contrib = m.contributions(P)
    assert (contrib["us_equity"]["x"] == 0.0).all() and (contrib["energy"]["x"] != 0.0).all()
    for a in ("energy", "gold"):
        assert np.allclose(contrib[a].sum(axis=1) + m.intercept[a], pred[a], rtol=0, atol=1e-14)
    s = json.loads(jsonio.dumps(m.summary()))
    assert s["asset_factors"] == {"energy": ["x"]} and s["asset_start"] == {"gold": "2000-01"}
    assert s["asset_train_start"]["gold"] == "2000-01" and s["asset_start_used"] == {"gold": "2000-01"}
    assert s["coef"]["us_equity"]["x"] == 0.0 and s["tstat"]["us_equity"]["x"] is None


def test_asset_start_falls_back_on_short_windows_and_the_mlp_refuses_equations():
    P = _equations_panel()
    short = AssetFactorModel(EQ_SPEC).fit(P, "1990-01", "2009-11")        # gold's own window: 119 months
    assert short.asset_start_used == {"gold": None} and short.asset_train_start["gold"] == "1990-01"
    assert short.stats.n_obs["gold"] == short.stats.n_obs["us_equity"]
    enough = AssetFactorModel(EQ_SPEC).fit(P, "1990-01", "2009-12")       # exactly MIN_ASSET_WINDOW months
    assert MIN_ASSET_WINDOW == 120 and enough.asset_start_used == {"gold": "2000-01"}
    assert enough.stats.n_obs["gold"] == 120
    off = AssetFactorModel(EQ_SPEC).fit(P, "1990-01", None, honor_asset_start=False)
    plain = AssetFactorModel(ModelSpec(asset_factors={"energy": ["x"]})).fit(P, "1990-01", None)
    assert off.asset_start_used == {"gold": None}
    assert np.array_equal(off.coef.to_numpy(), plain.coef.to_numpy())
    assert np.array_equal(off.intercept.to_numpy(), plain.intercept.to_numpy())
    for bad in (ModelSpec(kind="mlp", asset_factors={"energy": ["x"]}),
                ModelSpec(kind="mlp", asset_start={"gold": "2000-01"})):
        with pytest.raises(ValueError, match="linear"):
            AssetFactorModel(bad).fit(P, "1990-01", None)


def test_an_asset_subset_without_the_extras_owner_keeps_the_extra_out():
    """A model of some assets only, none of which owns the spec's extra: the extra stays out of every equation
    (each equation is still its own OLS on the common features, over the same months)."""
    P = _equations_panel()
    spec = ModelSpec(asset_factors={"energy": ["x"]})
    sub = [a for a in MODEL_ASSETS if a != "energy"]
    m = AssetFactorModel(spec, assets=sub).fit(P, "1990-01", None)
    assert (m.coef["x"] == 0.0).all() and m.tstat["x"].isna().all()
    plain = AssetFactorModel(ModelSpec(), assets=sub).fit(P, "1990-01", None)
    common = [f for f in m.features if f != "x"]
    assert np.abs(m.coef[common].to_numpy() - plain.coef[common].to_numpy()).max() < 1e-12
    assert np.abs(m.intercept.to_numpy() - plain.intercept.to_numpy()).max() < 1e-12


def _previous_design(panel: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """design_matrix before per-asset equations, verbatim."""
    base = panel[[f"f_{f}" for f in spec.factors] + list(spec.shock_cols)].copy()
    base.columns = [c[2:] if c.startswith("f_") else c for c in base.columns]
    parts = [base] + [base.shift(lag).add_suffix(f"_l{lag}") for lag in range(1, spec.lags + 1)]
    if spec.regime == "inflation":
        dummy = (panel["s_cpi_yoy"].shift(1) > 4.0).astype(float)
        parts += [base.mul(dummy, axis=0).add_suffix("_hi"), dummy.rename("hi_infl").to_frame()]
    return pd.concat(parts, axis=1)


def _previous_fit(panel: pd.DataFrame, spec: ModelSpec, start=config.MODEL_START, end=None):
    """The transmission fit before per-asset equations, verbatim: one OLS / ridge of every model asset on the
    joint design. (coef (assets, k), intercept, tstat (assets, k), x_mean, x_sd)"""
    P = panel.loc[start:end]
    X = _previous_design(panel, spec).loc[P.index]
    Y = P[[f"{spec.target}_{a}" for a in MODEL_ASSETS]]
    mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
    x, y = X[mask].to_numpy(float), Y[mask].to_numpy(float)
    x_mean, x_sd = x.mean(axis=0), x.std(axis=0)
    x_sd[x_sd < 1e-12] = 1.0
    z = (x - x_mean) / x_sd
    n, k = z.shape
    ym = y.mean(axis=0)
    A = z.T @ z + spec.ridge * n * np.eye(k)
    beta_z = np.linalg.solve(A, z.T @ (y - ym))
    coef = beta_z / x_sd[:, None]
    intercept = ym - x_mean @ coef
    resid = (y - ym) - z @ beta_z
    sigma2 = (resid ** 2).sum(axis=0) / max(n - k - 1, 1)
    inv = np.linalg.inv(A)
    se = np.sqrt(np.outer(np.diag(inv @ (z.T @ z) @ inv), sigma2))
    return coef.T, intercept, (beta_z / se).T, x_mean, x_sd


def _previous_features(spec: ModelSpec, base_paths, base_init, hi_infl):
    """features_from_paths before per-asset equations, verbatim."""
    n, T, k = base_paths.shape
    parts = [base_paths]
    if spec.lags:
        ext = np.concatenate([np.broadcast_to(base_init[None, :, :], (n, spec.lags, k)), base_paths], axis=1)
        parts += [ext[:, spec.lags - lag: spec.lags - lag + T, :] for lag in range(1, spec.lags + 1)]
    if spec.regime == "inflation":
        parts += [base_paths * hi_infl[:, :, None], hi_infl[:, :, None]]
    return np.concatenate(parts, axis=2)


def _assert_previous_path(model: AssetFactorModel, panel: pd.DataFrame, start=config.MODEL_START, end=None):
    pd.testing.assert_frame_equal(design_matrix(panel, model.spec), _previous_design(panel, model.spec),
                                  check_exact=True)
    assert model.features == list(_previous_design(panel.iloc[:2], model.spec).columns)
    coef, intercept, tstat, x_mean, x_sd = _previous_fit(panel, model.spec, start, end)
    assert np.array_equal(model.coef.to_numpy(), coef) and np.array_equal(model.intercept.to_numpy(), intercept)
    assert np.array_equal(model.tstat.to_numpy(), tstat, equal_nan=True)
    assert np.array_equal(model.x_mean, x_mean) and np.array_equal(model.x_sd, x_sd)


def test_specs_without_equations_take_the_previous_code_path_bit_for_bit():
    """The spec without extras simulates bit-identically to the previous code path: the fit, the design and the
    simulated features are the pre-change computations exactly (simulate() reads nothing else of the model)."""
    P = _synthetic_panel(400)
    rng = np.random.default_rng(9)
    for spec in (ModelSpec(), ModelSpec(lags=2, regime="inflation", ridge=0.05, name="lr"),
                 ModelSpec(asset_factors={"energy": ["oil"]}, asset_start={"gold": "1960-01"}, name="no-op")):
        m = AssetFactorModel(spec).fit(P, "1990-01", None)
        _assert_previous_path(m, P, "1990-01", None)
        assert m.asset_train_start == {a: m.train_start.strftime("%Y-%m") for a in m.assets}
        k = len(base_columns(spec))
        paths, init = rng.normal(size=(5, 30, k)), rng.normal(size=(max(spec.lags, 1), k))
        hi = (rng.random((5, 30)) > 0.5).astype(float)
        assert np.array_equal(features_from_paths(spec, paths, init[-spec.lags:] if spec.lags else init[:0], hi),
                              _previous_features(spec, paths, init[-spec.lags:] if spec.lags else init[:0], hi))


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


# ---------------------------------------------------------------- per-asset equations on the real panel

def _final_spec_or_skip(panel: pd.DataFrame) -> ModelSpec:
    missing = [c for c in base_columns(F_SPEC) if c not in panel.columns]
    if missing:
        pytest.skip(f"the panel has no {', '.join(missing)} (built before the missing-factor round)")
    return F_SPEC


def test_base_model_on_the_real_panel_is_the_previous_fit(real_ctx):
    _assert_previous_path(real_ctx.model, real_ctx.panel)
    _assert_previous_path(AssetFactorModel(ModelSpec()).fit(real_ctx.panel, config.MODEL_START, "2004-12"),
                          real_ctx.panel, config.MODEL_START, "2004-12")


def test_no_op_equations_simulate_bit_identically(real_ctx):
    from scenariolab.engine.simulate import MonthDB, make_anchor, simulate
    noop = ModelSpec(asset_factors={"energy": ["oil"]}, asset_start={"gold": "1960-01"}, name="no-op")
    model = AssetFactorModel(noop).fit(real_ctx.panel)
    db = MonthDB(real_ctx.panel, model)
    kw = dict(n_paths=100, seed=3, cma=real_ctx.cma, overrides=[{"factor_drift": {"ip": -0.02, "spread": 0.4}}, {}],
              attribution=True, **config.RESIDUALS)
    a = simulate(real_ctx.db, real_ctx.anchor, [real_ctx.db.prior()] * 2, [24, 36], **kw)
    b = simulate(db, make_anchor(real_ctx.panel, model), [db.prior()] * 2, [24, 36], **kw)
    assert np.array_equal(a.total, b.total) and np.array_equal(a.idx, b.idx)
    assert all(np.array_equal(a.macro[k], b.macro[k]) for k in a.macro)
    assert np.array_equal(a.attribution.values, b.attribution.values)


def test_gold_start_falls_back_before_its_window_fills(real_ctx):
    spec = ModelSpec(asset_start={"gold": "1982-01"}, name="gold_window")      # the rule needs no new column
    early = AssetFactorModel(spec).fit(real_ctx.panel, config.MODEL_START, "1984-12")
    assert early.asset_start_used["gold"] is None
    assert early.asset_train_start["gold"] == early.asset_train_start["us_equity"]
    late = AssetFactorModel(spec).fit(real_ctx.panel, config.MODEL_START, "1994-12")
    assert late.asset_start_used["gold"] == "1982-01" and late.asset_train_start["gold"] == "1982-01"
    assert late.stats.n_obs["gold"] == 156 < late.stats.n_obs["us_equity"]
    # the full-sample fit draws the same months; gold's 1973-81 residuals come from its 1982+ equation
    from scenariolab.engine.simulate import MonthDB
    full = AssetFactorModel(spec).fit(real_ctx.panel)
    db = MonthDB(real_ctx.panel, full)
    assert db.dates.equals(real_ctx.db.dates)
    X = design_matrix(real_ctx.panel, spec).loc[db.dates]
    gold = real_ctx.panel.loc[db.dates, "p_gold"] - (X @ full.coef.loc["gold"] + full.intercept["gold"])
    assert np.allclose(db.resid["gold"], gold, rtol=0, atol=1e-12)
    assert db.dates[0] < pd.Timestamp("1982-01-01") and np.isfinite(db.resid.loc[:"1981-12", "gold"]).all()
    # one asset's own start moves no other equation (to rounding: that group is standardized on its own)
    assert np.allclose(db.resid["us_equity"], real_ctx.db.resid["us_equity"], rtol=0, atol=1e-12)


def test_final_model_runs_through_the_forecast_origins_and_the_causality_eras(real_ctx):
    spec = _final_spec_or_skip(real_ctx.panel)
    from scenariolab.analysis import causality, forecast
    panel = real_ctx.panel
    origins = forecast.forecast_origins(panel)
    if panel.index.max() == pd.Timestamp("2026-09-01"):
        assert len(origins) == 42
    for t in origins:
        eng = forecast.origin_engine(panel, spec, t)
        # gold's 1982 start needs 120 months of its own: from the 1991-12 origin on
        assert eng["model"].asset_start_used["gold"] == (None if t < pd.Timestamp("1991-12-01") else "1982-01"), t
        assert np.isfinite(eng["model"].coef.to_numpy()).all() and len(eng["db"]) > 0
    for per in causality.PERIODS:
        m = AssetFactorModel(spec).fit(panel, per["start"], per["end"], honor_asset_start=False)
        assert m.asset_start_used == {"gold": None} and np.isfinite(m.coef.to_numpy()).all()


def test_final_model_draws_the_new_inputs_from_the_drawn_months(real_ctx, monkeypatch):
    spec = _final_spec_or_skip(real_ctx.panel)
    from scenariolab.engine import simulate as S
    model = AssetFactorModel(spec).fit(real_ctx.panel)
    db = S.MonthDB(real_ctx.panel, model)
    assert db.dates.equals(real_ctx.db.dates)                         # the new inputs restrict no month
    if real_ctx.panel.index.max() == pd.Timestamp("2026-09-01"):
        assert len(db) == 644 and db.dates[0] == pd.Timestamp("1973-02-01")
    seen = []

    def spy(*args):
        seen.append(args[1].copy())
        return features_from_paths(*args)

    monkeypatch.setattr(S, "features_from_paths", spy)
    sim = S.simulate(db, S.make_anchor(real_ctx.panel, model), [db.prior()] * 2, [24, 36], n_paths=200, seed=4,
                     overrides=[{"factor_drift": {"ip": -0.02, "spread": 0.4}}, {}])
    cols = base_columns(spec)
    for c in ("f_gz", "f_tp10_x_core4", "f_metals_oil"):
        assert np.array_equal(seen[0][:, :, cols.index(c)], db.frame[c].to_numpy(float)[sim.idx]), c
    # gold's residuals before 1982 come from its 1982+ equation; its months are still drawn
    resid = model.residuals(real_ctx.panel)["gold"]
    assert np.array_equal(db.resid["gold"].to_numpy(), resid.loc[db.dates].to_numpy())
    assert db.dates[0] < pd.Timestamp("1982-01-01") and model.asset_train_start["gold"] == "1982-01"
