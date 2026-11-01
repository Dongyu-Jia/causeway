"""Return attribution (传导归因): simulate(attribution=True) and runner.attribution_summary.

The split must leave every simulated number unchanged, add up exactly to the mean arithmetic return
(and, with the reconciliation term, to the median CAGR the page shows), put a pure oil shock in the oil
term alone, and build the structural assets from their own paths. eras[].attribution is tested in
test_eras.py.
"""
import numpy as np
import pytest

from scenariolab import config

pytestmark = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")

# a 12-month oil shock, then a recession phase with every kind of override (padded to 180 months)
OIL_SHOCK = {"id": "attr_oil", "name": "测试：油价冲击", "phases": [
    {"name": "冲击", "months": 12, "views": [{"var": "f_oil", "mean_annual": 0.6}]},
    {"name": "衰退", "months": 24, "views": [{"var": "f_unrate", "mean_annual": 0.6, "vol_mult": 1.0}],
     "overrides": {"asset_drift": {"us_equity": -0.05, "energy": 0.03}, "factor_drift": {"ip": -0.01},
                   "resid_scale": 1.3}},
]}


@pytest.fixture(scope="module")
def ctx():
    from scenariolab.context import EngineContext
    return EngineContext.load(selection=True)


def _plan(ctx, spec):
    """What run_spec hands simulate(): weights, phase months and keyword arguments."""
    from scenariolab.engine.simulate import with_oil_lock
    from scenariolab.scenarios import runner
    from scenariolab.scenarios.schema import validate_spec
    clean, report = validate_spec(spec, ctx.db.frame, config.HORIZON_MONTHS)
    weights, _ = runner.phase_weights(ctx, clean, report["warnings"])
    ovs = with_oil_lock([p["views"] for p in clean["phases"]], [p.get("overrides", {}) for p in clean["phases"]])
    cma, vm = runner.valuation_for(ctx)
    kw = dict(block=runner.BLOCK, cma=cma, overrides=ovs, reversion=clean.get("reversion") or None, valuation=vm,
              **runner.RESIDUALS)
    return weights, [p["months"] for p in clean["phases"]], kw


def _phase_mean(x, months):
    """(n, T, ...) -> (n, P, ...) phase means x 12."""
    starts = np.concatenate([[0], np.cumsum(months)[:-1]])
    return np.stack([x[:, s:s + m].astype(float).mean(axis=1) for s, m in zip(starts, months)], axis=1) * 12.0


@pytest.fixture(scope="module")
def pair(ctx):
    from scenariolab.engine.simulate import simulate
    weights, months, kw = _plan(ctx, OIL_SHOCK)
    plain = simulate(ctx.db, ctx.anchor, weights, months, n_paths=300, seed=3, **kw)
    attr = simulate(ctx.db, ctx.anchor, weights, months, n_paths=300, seed=3, attribution=True, **kw)
    return plain, attr, months


def test_attribution_leaves_every_simulated_number_bit_identical(pair):
    plain, attr, _ = pair
    assert plain.attribution is None and attr.attribution is not None
    assert np.array_equal(plain.total, attr.total) and np.array_equal(plain.income, attr.income)
    assert np.array_equal(plain.idx, attr.idx) and plain.macro.keys() == attr.macro.keys()
    for k in plain.macro:
        assert np.array_equal(plain.macro[k], attr.macro[k]), k


def test_run_spec_with_and_without_attribution_is_bit_identical(ctx):
    from scenariolab.scenarios.library import load_specs, resolve_calibration, seed_for
    from scenariolab.scenarios.runner import run_spec
    spec, _ = resolve_calibration(next(s for s in load_specs() if s["id"] == "oil_war"))
    a, sa = run_spec(ctx, spec, n_paths=200, seed=seed_for("oil_war"), attribution=False)
    b, sb = run_spec(ctx, spec, n_paths=200, seed=seed_for("oil_war"), attribution=True)
    assert np.array_equal(sa.total, sb.total) and np.array_equal(sa.income, sb.income)
    assert all(np.array_equal(sa.macro[k], sb.macro[k]) for k in sa.macro)
    assert "attribution" not in a["assets"]["energy"] and "attribution" in b["assets"]["energy"]
    for asset, x in a["assets"].items():
        assert x["stats"] == b["assets"][asset]["stats"] and x["fan"] == b["assets"][asset]["fan"]


def test_terms_add_up_to_the_mean_arithmetic_return(ctx, pair):
    _, sim, months = pair
    att = sim.attribution
    n, P, A = sim.total.shape[0], len(months), len(sim.assets)
    assert att.values.shape == (n, P, A, len(att.terms)) and att.phase_months == months
    assert np.abs(att.values.sum(axis=-1) - att.total).max() < 1e-12              # exact, float64
    assert np.abs(att.values.sum(axis=-1) - _phase_mean(sim.total, months)).max() < 1e-6   # vs the float32 totals
    assert att.method == "linear" and att.factor_terms == list(ctx.model.features)
    # the overrides term is exactly the phase's asset_drift; CMA and intercept are the engine's constants
    eq, en = sim.assets.index("us_equity"), sim.assets.index("energy")
    assert np.allclose(att.get("overrides")[:, :, eq], [0.0, -0.05, 0.0])
    assert np.allclose(att.get("overrides")[:, :, en], [0.0, 0.03, 0.0])
    for j, a in enumerate(sim.assets):
        if att.kinds[a] == "model":
            assert np.allclose(att.get("cma")[:, :, j], ctx.cma["adjustment"][a])
            assert np.allclose(att.get("intercept")[:, :, j], 12.0 * ctx.model.intercept[a])
            assert np.all(att.get("floor")[:, :, j] == 0.0)


def test_runner_summary_reconciles_to_the_median_cagr(ctx):
    from scenariolab.scenarios.runner import run_spec
    detail, _ = run_spec(ctx, OIL_SHOCK, n_paths=300, seed=4)
    months = [p["months"] for p in detail["phases"]]
    for a, x in detail["assets"].items():
        att = x["attribution"]
        assert att["total"]["cagr_p50"] == x["stats"]["cagr_p50"]                  # the number the page shows
        assert [p["months"] for p in att["phases"]] == months and att["total"]["months"] == sum(months)
        for blk in att["phases"] + [att["total"]]:
            assert sum(blk["terms"].values()) == pytest.approx(blk["arithmetic"], abs=1e-12)
            assert blk["arithmetic"] + blk["reconciliation"] == pytest.approx(blk["cagr_p50"], abs=1e-12)
            if att["kind"] == "model":
                fs = sum(blk["terms"][f] for f in ctx.model.features)
                assert blk["factor_sum"] == pytest.approx(fs, abs=1e-12)
                assert blk["factor_band"][0] <= blk["factor_band"][1]
        # the whole path is the months-weighted average of the phases
        for t, v in att["total"]["terms"].items():
            assert v == pytest.approx(sum(p["terms"][t] * p["months"] for p in att["phases"]) / sum(months), abs=1e-12)
    assert detail["assets"]["us_equity"]["attribution"]["phases"][1]["terms"]["overrides"] == pytest.approx(-0.05)
    gold = detail["assets"]["gold"]["attribution"]["total"]["terms"]
    assert "overrides" not in gold and "income" not in gold                      # no asset_drift, no income
    assert set(detail["assets"]["cash"]["attribution"]["total"]["terms"]) == {"income"}


def test_pure_oil_shock_is_attributed_to_oil(ctx):
    """Same draws, plus an oil drift (an oil shock beyond the drawn months): every asset's return changes
    by its oil term alone, and crude oil's whole change sits in its oil term."""
    from scenariolab.engine.simulate import simulate
    spec = {"name": "测试", "phases": [{"name": "一", "months": 24, "views": [{"var": "s_cpi_yoy", "mean": 4.0}]}]}
    weights, months, kw = _plan(ctx, spec)
    base = simulate(ctx.db, ctx.anchor, weights, months, n_paths=400, seed=9, attribution=True, **kw)
    shock_ov = [{**kw["overrides"][0], "factor_drift": {"oil": 0.5}}] + kw["overrides"][1:]
    shock = simulate(ctx.db, ctx.anchor, weights, months, n_paths=400, seed=9, attribution=True,
                     **{**kw, "overrides": shock_ov})
    a, b = base.attribution, shock.attribution
    assert np.array_equal(base.idx, shock.idx)
    for t in a.terms:
        if t != "oil":
            assert np.array_equal(b.get(t), a.get(t)), t                         # nothing else moves
    d_oil = b.get("oil") - a.get("oil")
    assert np.allclose(b.total - a.total, d_oil, atol=1e-12)                       # the whole change is oil
    crude, en, eq = (base.assets.index(x) for x in ("crude_oil", "energy", "us_equity"))
    assert (d_oil[:, 0, crude] > 0).all() and d_oil[:, 0, crude].mean() > 0.3
    beta_en, beta_eq = (float(ctx.model.coef.loc[x, "oil"]) for x in ("energy", "us_equity"))
    assert beta_en > 0 and (d_oil[:, 0, en] > 0).all()
    assert np.allclose(d_oil[:, 0, eq] / d_oil[:, 0, en], beta_eq / beta_en)       # beta x the same oil change


def test_an_oil_view_lands_in_the_oil_term(pair):
    _, sim, _ = pair
    att = sim.attribution
    en, crude = sim.assets.index("energy"), sim.assets.index("crude_oil")
    shock = {f: float(att.get(f)[:, 0, en].mean()) for f in att.factor_terms}
    assert max(shock, key=lambda f: abs(shock[f])) == "oil" and shock["oil"] > 0.05
    assert att.get("oil")[:, 0, crude].mean() > 0.4                                # the +0.6 log view, in simple terms


def test_factor_terms_use_the_simulated_factor_changes(ctx, pair):
    """beta x the factor changes the transmission saw, recovered from outside from the macro level paths (after
    level scaling, reversion, the oil pull and the drifts), not the drawn months' raw changes."""
    _, sim, months = pair
    att = sim.attribution
    M = {k: v.astype(float) for k, v in sim.macro.items()}
    simulated = {f: np.diff(M[f], axis=1) for f in ("ff", "y10", "spread", "unrate")}
    simulated.update({f: np.diff(np.log(M[f]), axis=1) for f in ("oil", "ip", "usd", "labor_share")})
    moved = set()
    for a in ("us_equity", "energy"):
        j = sim.assets.index(a)
        for f, d in simulated.items():
            beta = float(ctx.model.coef.loc[a, f])
            assert np.allclose(att.get(f)[:, :, j], beta * _phase_mean(d, months), atol=2e-6), (a, f)
            raw = ctx.db.factors[f"f_{f}"].to_numpy(float)[sim.idx]
            if not np.allclose(att.get(f)[:, :, j], beta * _phase_mean(raw, months), atol=1e-4):
                moved.add(f)
    # rates and unemployment are re-anchored, oil is pulled outside its view phase, ip carries phase 2's drift;
    # the dollar and the labour share are the drawn months' own changes
    assert moved == {"ff", "y10", "spread", "unrate", "oil", "ip"}


def test_structural_assets_are_built_from_their_own_paths(pair):
    _, sim, months = pair
    att = sim.attribution
    total, income = _phase_mean(sim.total, months), _phase_mean(sim.income, months)
    tr, cash, oil = (sim.assets.index(a) for a in ("treasury_10y", "cash", "crude_oil"))
    assert (att.kinds["treasury_10y"], att.kinds["cash"], att.kinds["crude_oil"]) == ("treasury", "cash", "crude_oil")
    # treasury = coupon income + the repricing of a par bond on the simulated yield change
    assert np.allclose(att.get("income")[:, :, tr], income[:, :, tr], atol=1e-6)
    assert np.allclose(att.get("y10")[:, :, tr], total[:, :, tr] - income[:, :, tr], atol=1e-6)
    w = np.asarray(months, float) / sum(months)
    reprice = att.get("y10")[:, :, tr] @ w                        # whole path, per path
    rising = sim.macro["y10"][:, -1].astype(float) > sim.macro["y10"][:, 0].astype(float)
    assert reprice[rising].mean() < 0 < reprice[~rising].mean()   # yields up -> capital loss, down -> gain
    # cash = interest only; crude oil = the oil path's simple monthly change
    assert np.allclose(att.get("income")[:, :, cash], total[:, :, cash], atol=1e-6)
    lo = np.log(sim.macro["oil"].astype(float))
    assert np.allclose(att.get("oil")[:, :, oil], _phase_mean(np.expm1(np.diff(lo, axis=1)), months), atol=1e-5)
    for j in (tr, cash, oil):                                     # no model term reaches a structural asset
        for t in att.terms:
            if t not in ("y10", "oil", "income", "floor"):
                assert np.all(att.get(t)[:, :, j] == 0.0), (sim.assets[j], t)


def test_lagged_transmission_falls_back_to_an_unattributed_block(ctx):
    from scenariolab.engine.factor_model import ModelSpec
    from scenariolab.engine.simulate import MonthDB, make_anchor, simulate
    model = ctx.fit(ModelSpec(lags=1, name="lag1"))
    db = MonthDB(ctx.panel, model)
    sim = simulate(db, make_anchor(ctx.panel, model), [db.prior(), db.prior()], [24, 12], n_paths=100, seed=1,
                   attribution=True)
    att = sim.attribution
    assert att.method == "unattributed" and att.factor_terms == [] and "滞后" in att.reason
    assert "unattributed" in att.terms and "intercept" not in att.terms
    assert np.abs(att.values.sum(axis=-1) - att.total).max() < 1e-12
    assert np.abs(att.get("unattributed")[:, :, sim.assets.index("us_equity")]).max() > 0


def test_valuation_pull_gets_its_own_term_and_changes_nothing(ctx):
    """With the valuation-state mode on (kappa > 0, as the valuation study runs it) the pull toward the fair
    value is a term of its own; the split still adds up and the returns are the same as without it."""
    from scenariolab.engine.simulate import simulate, valuation_setup
    cma, vm = valuation_setup(ctx.db, ctx.anchor, {"enabled": True, "kappa": 0.1, "cma": "valuation_state"})
    kw = dict(n_paths=120, seed=5, cma=cma, valuation=vm, **config.RESIDUALS)
    a = simulate(ctx.db, ctx.anchor, [ctx.db.prior(), ctx.db.prior()], [36, 144], **kw)
    b = simulate(ctx.db, ctx.anchor, [ctx.db.prior(), ctx.db.prior()], [36, 144], attribution=True, **kw)
    assert np.array_equal(a.total, b.total) and all(np.array_equal(a.macro[k], b.macro[k]) for k in a.macro)
    att = b.attribution
    assert "valuation_pull" in att.terms
    assert np.abs(att.values.sum(axis=-1) - att.total).max() < 1e-12
    eq, gold = b.assets.index("us_equity"), b.assets.index("gold")
    pull = att.get("valuation_pull")
    assert np.abs(pull[:, :, eq]).max() > 0 and np.all(pull[:, :, gold] == 0.0)     # gold has no P/D pull
    assert pull[:, 0, eq].mean() < 0                                               # today's P/D is above its fair value
