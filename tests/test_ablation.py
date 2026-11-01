from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scenariolab import config, jsonio
from scenariolab.analysis import ablation as ab
from scenariolab.analysis import backtest as bt
from scenariolab.engine import residuals
from scenariolab.engine.sampler import BlockProbs, block_start_probs, sample_phases

needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
MODES = [("decoupled", {}), ("detrended", {"window": 24}), ("trend_decoupled", {"window": 36}),
         ("partial", {"coupling": 0.5}), ("recentered", {"coupling": 0.0}), ("recentered", {"coupling": 0.5})]


class _ToyDB:
    """Residuals with slow era drifts (24-month regimes) plus monthly noise, a uniform prior with rare months."""

    def __init__(self, T: int = 240, A: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dates = pd.date_range("1980-01-01", periods=T, freq="MS")
        drift = np.repeat(rng.normal(0.0, 0.01, size=(T // 24 + 1, A)), 24, axis=0)[:T]
        self.resid = pd.DataFrame(rng.normal(0.0, 0.04, size=(T, A)) + drift, index=self.dates,
                                  columns=[f"a{i}" for i in range(A)])
        self.model = SimpleNamespace(assets=list(self.resid.columns))

    def __len__(self) -> int:
        return len(self.resid)

    def prior(self) -> np.ndarray:
        w = np.ones(len(self))
        w[100:106] = 0.25
        return w / w.sum()


def _expected(db, weights, plan, block=6) -> np.ndarray:
    """Expected residual mean over all simulated months under the sampler's month frequencies."""
    freqs = residuals.phase_frequencies(weights, plan, block, len(db))
    R = db.resid.to_numpy(float)
    return sum(m * (f @ R) for f, m in zip(freqs, plan)) / sum(plan)


def _tilted(db, lo: int, hi: int, block: int = 6) -> BlockProbs:
    """Block-start weights concentrated on months lo..hi (a view picking one episode)."""
    w = np.full(len(db), 1e-4)
    w[lo:hi] = 1.0
    return BlockProbs(block_start_probs(w / w.sum(), block), block)


# ---------------------------------------------------------------- residual variants (toy data)

@pytest.mark.parametrize("mode,kw", MODES)
def test_residual_variants_shape_and_finite(mode, kw):
    db, plan = _ToyDB(), [24, 21]
    weights = [_tilted(db, 40, 70), db.prior()]
    idx = sample_phases(weights, plan, 50, 6, seed=1)
    out = residuals.draw(db, idx, weights, plan, 6, seed=1, mode=mode, **kw)
    assert out.shape == (50, sum(plan), 3) and np.isfinite(out).all()
    again = residuals.draw(db, idx, weights, plan, 6, seed=1, mode=mode, **kw)
    assert np.array_equal(out, again)                                  # deterministic given the seed


def test_coupled_draw_is_the_drawn_months_own_residual():
    db, plan = _ToyDB(), [24, 24]
    idx = sample_phases([db.prior()] * 2, plan, 30, 6, seed=2)
    out = residuals.draw(db, idx, [db.prior()] * 2, plan, 6, seed=2, mode="coupled", window=12, coupling=0.1)
    assert np.array_equal(out, db.resid.to_numpy(float)[idx])
    with pytest.raises(ValueError):
        residuals.draw(db, idx, [db.prior()] * 2, plan, 6, seed=2, mode="nope")


def test_detrended_parts_keep_the_prior_mean_exactly():
    db = _ToyDB()
    R = db.resid.to_numpy(float)
    f = residuals.prior_frequency(db.prior(), 6)
    assert f.sum() == pytest.approx(1.0)
    for window in (24, 36, 60):
        fast, slow = residuals.components(db, window, 6)
        assert np.allclose(fast + slow, R)
        assert np.allclose(f @ slow, 0.0, atol=1e-15)                  # the slow part adds no drift
        assert np.allclose(f @ fast, f @ R, atol=1e-15)                # detrended = same unconditional mean
        assert fast.std(axis=0).mean() < R.std(axis=0).mean()          # and removes the era drifts


def test_recentered_is_bit_identical_without_views_and_exact_with_them():
    db, plan = _ToyDB(), [24, 24, 12]
    prior = [db.prior()] * 3
    idx = sample_phases(prior, plan, 40, 6, seed=3)
    for c in (0.0, 0.5):
        assert np.array_equal(residuals.draw(db, idx, prior, plan, 6, 3, mode="recentered", coupling=c),
                              db.resid.to_numpy(float)[idx])
    weights = [_tilted(db, 30, 60), _tilted(db, 150, 190), db.prior()]
    drift = residuals.conditional_drift(db, weights, plan, 6)
    assert np.allclose(drift[2], 0.0) and np.abs(drift[0]).max() > 1e-3
    # expected drawn mean = coupling * conditional + (1 - coupling) * prior, phase by phase
    R = db.resid.to_numpy(float)
    q = residuals.phase_frequencies(weights, plan, 6, len(db))
    p = residuals.phase_frequencies(prior, plan, 6, len(db))
    for c in (0.0, 0.25, 0.5):
        shifted = R[None, :, :] - (1 - c) * drift[:, None, :]            # what each phase's months become
        for k in range(3):
            assert np.allclose(q[k] @ shifted[k], c * (q[k] @ R) + (1 - c) * (p[k] @ R), atol=1e-14)


@pytest.mark.parametrize("mode,kw", [m for m in MODES if m[0] in ("decoupled", "partial", "trend_decoupled")])
def test_stochastic_variants_keep_the_unconditional_mean(mode, kw):
    db, plan = _ToyDB(), [24, 24]
    prior = [db.prior()] * 2
    idx = sample_phases(prior, plan, 20000, 6, seed=4)
    out = residuals.draw(db, idx, prior, plan, 6, seed=4, mode=mode, **kw)
    target = _expected(db, prior, plan)
    se = db.resid.to_numpy(float).std(axis=0) / np.sqrt(out.shape[0] * sum(plan) / 6)
    assert np.all(np.abs(out.reshape(-1, 3).mean(axis=0) - target) < 5 * se)


def test_variants_remove_or_shrink_the_conditional_drift():
    db, plan = _ToyDB(seed=5), [24, 24]
    weights = [_tilted(db, 24, 48), _tilted(db, 24, 48)]                # one 24-month regime with its own drift
    idx = sample_phases(weights, plan, 20000, 6, seed=5)
    prior_mean, cond_mean = _expected(db, [db.prior()] * 2, plan), _expected(db, weights, plan)
    assert np.abs(cond_mean - prior_mean).max() > 0.005

    def drawn(mode, **kw):
        return residuals.draw(db, idx, weights, plan, 6, seed=5, mode=mode, **kw).reshape(-1, 3).mean(axis=0)
    tol = 6 * 0.04 / np.sqrt(20000 * 8)
    assert np.allclose(drawn("coupled"), cond_mean, atol=tol)
    assert np.allclose(drawn("decoupled"), prior_mean, atol=tol)
    assert np.allclose(drawn("partial", coupling=0.5), 0.5 * (cond_mean + prior_mean), atol=tol)
    assert np.allclose(drawn("recentered", coupling=0.5), 0.5 * (cond_mean + prior_mean), atol=tol)
    assert np.allclose(drawn("partial", coupling=1.0), drawn("coupled"))
    assert np.array_equal(residuals.draw(db, idx, weights, plan, 6, 5, mode="partial", coupling=0.0),
                          residuals.draw(db, idx, weights, plan, 6, 5, mode="decoupled"))
    # detrending removes most of a 24-month regime's drift
    assert np.abs(drawn("detrended", window=24) - prior_mean).max() < 0.5 * np.abs(cond_mean - prior_mean).max()


def test_block_ids_follow_the_sampler_blocks():
    ids = residuals.block_ids([24, 9, 6], [6, 6, 6])
    assert list(ids[:7]) == [0] * 6 + [1] and ids[23] == 3 and ids[24] == 4 and ids[30] == 5 and ids[33] == 6
    assert len(ids) == 39 and ids[-1] == 6


def test_rolling_mean_uses_only_the_frames_own_months():
    db = _ToyDB(T=120)
    full = residuals.rolling_mean(db.resid, 24)
    cut = residuals.rolling_mean(db.resid.iloc[:80], 24)
    assert np.allclose(full.iloc[:68].to_numpy(), cut.iloc[:68].to_numpy())   # window fully inside: same
    assert not np.allclose(full.iloc[79].to_numpy(), cut.iloc[79].to_numpy())  # at the cut: no later months used
    assert np.allclose(cut.iloc[79].to_numpy(), db.resid.iloc[67:80].mean().to_numpy())


# ---------------------------------------------------------------- real engine

@pytest.fixture(scope="module")
def window():
    if not config.PANEL_PATH.exists():
        pytest.skip("panel not built")
    from scenariolab.context import load_panel_with_shocks
    from scenariolab.engine.factor_model import ModelSpec
    panel = load_panel_with_shocks()
    model, db, anchor = bt.fit_window(panel, ModelSpec(), "2004-12")
    months = panel.loc["2005-01":"2014-12"].index
    plan = bt.phase_plan(len(months))
    vw = bt.view_weights(db, bt.realized_views(panel, months, plan), plan)
    return panel, db, anchor, plan, vw["cond"]


@needs_panel
def test_default_parameters_reproduce_the_coupled_engine(window):
    from scenariolab.engine.simulate import simulate
    _, db, anchor, plan, cond = window
    a = simulate(db, anchor, cond, plan, n_paths=60, seed=11)
    b = simulate(db, anchor, cond, plan, n_paths=60, seed=11, resid_mode="coupled", resid_window=24,
                 resid_coupling=0.2, resid_seed=99)
    assert np.array_equal(a.total, b.total) and np.array_equal(a.idx, b.idx)
    assert all(np.array_equal(a.macro[k], b.macro[k]) for k in a.macro)
    # the residual inside the default simulation is exactly the drawn month's own residual
    c = simulate(db, anchor, cond, plan, n_paths=60, seed=11, overrides=[{"resid_scale": 2.0}] * len(plan))
    model_j = [config.ASSET_IDS.index(x) for x in db.model.assets if x not in ("treasury_10y", "crude_oil")]
    implied = (c.total - a.total).astype(float)[:, :, model_j]
    own = db.resid.to_numpy(float)[a.idx][:, :, [db.model.assets.index(config.ASSET_IDS[j]) for j in model_j]]
    ok = (a.total[:, :, model_j] > -0.95) & (c.total[:, :, model_j] > -0.95)
    assert np.allclose(implied[ok], own[ok], atol=1e-5)
    # and the macro paths do not depend on the residual mode
    d = simulate(db, anchor, cond, plan, n_paths=60, seed=11, resid_mode="decoupled")
    assert np.array_equal(a.idx, d.idx) and np.array_equal(a.macro["ff"], d.macro["ff"])
    assert not np.array_equal(a.total, d.total)


@needs_panel
def test_unconditional_simulation_is_unchanged_by_recentering(window):
    from scenariolab.engine.simulate import simulate
    _, db, anchor, plan, _ = window
    prior = [db.prior()] * len(plan)
    a = simulate(db, anchor, prior, plan, n_paths=80, seed=3)
    b = simulate(db, anchor, prior, plan, n_paths=80, seed=3, resid_mode="recentered", resid_coupling=0.0)
    assert np.array_equal(a.total, b.total)
    R, f = db.resid.to_numpy(float), residuals.prior_frequency(db.prior(), 6)
    for w in (24, 60):
        fast, slow = residuals.components(db, w, 6)
        assert np.allclose(f @ fast, f @ R, atol=1e-14) and np.allclose(f @ slow, 0.0, atol=1e-14)


def _perturbed_after(panel: pd.DataFrame, date: str, seed: int = 3) -> pd.DataFrame:
    """Every month after `date` scrambled (as in test_backtest): a training-window engine must not notice."""
    rng = np.random.default_rng(seed)
    P = panel.copy()
    rows = P.index > pd.Timestamp(date)
    cols = [c for c in P.columns if c.startswith(("r_", "p_", "d_", "f_", "s_", "lvl_", "e_", "dy_"))]
    P.loc[rows, cols] = P.loc[rows, cols].to_numpy() * rng.normal(1.0, 0.5, size=(int(rows.sum()), len(cols))) + 0.01
    return P


@needs_panel
def test_residual_variants_do_not_look_ahead(window):
    from scenariolab.engine.factor_model import ModelSpec
    from scenariolab.engine.simulate import simulate
    panel = window[0]
    views = [[{"var": "s_cpi_yoy", "mean": 3.0}, {"var": "f_oil", "mean": 0.01}]] * 2     # fixed views: only the
    runs = []                                                                              # residual draw is tested
    for P in (panel, _perturbed_after(panel, "2004-12")):
        _, db, anchor = bt.fit_window(P, ModelSpec(), "2004-12")
        cond = bt.view_weights(db, views, [24, 24])["cond"]
        sims = [simulate(db, anchor, cond, [24, 24], n_paths=40, seed=7, resid_mode=m,
                         resid_window=kw.get("window"), resid_coupling=kw.get("coupling", 1.0)) for m, kw in MODES]
        runs.append((residuals.components(db, 60, 6), residuals.conditional_drift(db, cond, [24, 24], 6), sims))
    (c1, d1, s1), (c2, d2, s2) = runs
    assert np.allclose(c1[0], c2[0]) and np.allclose(c1[1], c2[1]) and np.allclose(d1, d2)
    assert all(np.allclose(a.total, b.total) for a, b in zip(s1, s2))


@needs_panel
@pytest.mark.skipif(not (config.RESULTS_DIR / "backtest.json").exists(), reason="backtest not built")
def test_ablation_engine_variant_reproduces_the_backtest():
    from scenariolab.context import EngineContext
    ctx = EngineContext.load(selection=True)
    w = next(x for x in bt.WINDOWS if x["id"] == "post_covid")      # the window without shrunk phases is fast
    m = ab.evaluate(ctx.panel, ab.setup_window(ctx.panel, ctx.spec, w), dict(config.RESIDUALS), bt.N_PATHS, bt.SEED)
    ref = next(r for r in jsonio.load(config.RESULTS_DIR / "backtest.json")["summary"]["windows"] if r["id"] == "post_covid")
    for k in ("cagr_mae_cond_risk", "cagr_mae_uncond_risk", "coverage_cond_risk", "engine_abs_annual",
              "transmission_abs_annual", "scenario_abs_annual"):
        assert m[k] == pytest.approx(ref[k], abs=2e-6), k


def test_backtest_view_helpers_keep_their_defaults():
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    panel = pd.DataFrame({v: np.arange(30, dtype=float) for v in set(bt.VIEW_VARS) | set(ab.VIEW_SETS["levels8"])},
                         index=idx)
    assert bt.realized_views(panel, idx, [24, 6]) == bt.realized_views(panel, idx, [24, 6], view_vars=bt.VIEW_VARS)
    lv = bt.realized_views(panel, idx, [24, 6], view_vars=ab.VIEW_SETS["levels2"])
    assert [v["var"] for v in lv[0]] == ["s_cpi_yoy", "s_ip_yoy"]


# ---------------------------------------------------------------- selection and wiring

def _entry(vid, val, test, *, hyp="H1", val_cov=0.75, test_cov=0.97, se=0.0005, fan=1.0, adoptable=True):
    return {"id": vid, "name": vid, "hypothesis": hyp, "adoptable": adoptable, "validation_mae": val, "test_mae": test,
            "validation_coverage": val_cov, "test_coverage": test_cov, "validation_gain": 0.04 - val,
            "test_gain": 0.06 - test, "validation_gain_se": se, "test_gain_se": se,
            "forward_fan": {"width_ratio_min": fan, "worst_asset": "us_equity", "p5_change": {"us_equity": 0.006}}}


def test_selection_rule_is_validation_first_and_confirms_once():
    base = _entry("baseline", 0.04, 0.06, adoptable=False)
    narrow = _entry("narrow", 0.030, 0.050, fan=0.90)                  # best on validation, but narrows the fan
    good = _entry("good", 0.035, 0.055)
    lucky = _entry("lucky", 0.041, 0.030)                              # great on test only: never chosen
    proto = _entry("proto", 0.020, 0.040, hyp="H3", adoptable=False)   # protocol variants are not adoptable
    s = ab.select([base, narrow, good, lucky, proto])
    assert s["chosen"] == "good" and s["adopt"] and s["eligible"] == ["good"]
    assert not s["checks"]["narrow"]["forward_fan"] and "narrow" in s["reason"]
    hurt = _entry("good", 0.035, 0.070)                                # validation pick fails the test: no second pick
    s2 = ab.select([base, narrow, hurt, _entry("second", 0.037, 0.050)])
    assert s2["chosen"] == "good" and not s2["adopt"]
    s3 = ab.select([base, _entry("noisy", 0.039, 0.05, se=0.002)])      # gain within 2 standard errors
    assert s3["chosen"] == "baseline" and not s3["adopt"]


def test_scenario_engine_patches_and_restores_the_runner():
    from scenariolab.scenarios import runner
    resid, pool = runner.RESIDUALS, runner.pool_phase
    assert resid == config.RESIDUALS                                   # the runner ships the engine's setting
    with ab.scenario_engine({"resid_mode": "recentered", "resid_coupling": 0.5, "min_ens": 60.0}):
        assert runner.RESIDUALS == {"resid_mode": "recentered", "resid_window": None, "resid_coupling": 0.5}
        assert runner.pool_phase.keywords == {"min_ens": 60.0}
    assert runner.RESIDUALS is resid and runner.pool_phase is pool
    with ab.scenario_engine({}):                                       # the reference engine: fully coupled
        assert runner.RESIDUALS == {} and runner.pool_phase is pool
    assert runner.RESIDUALS is resid


def test_scenario_runs_never_write_into_results():
    with pytest.raises(ValueError):
        ab.scenario_effects(None, [], config.RESULTS_DIR / "x")
    with pytest.raises(ValueError):
        ab.scenario_effects(None, [], config.RESULTS_DIR)


def test_variant_list_is_well_formed():
    ids = [v["id"] for v in ab.VARIANTS]
    assert len(ids) == len(set(ids)) and ids[0] == "baseline"
    for v in ab.VARIANTS:
        p = {**ab.BASE, **v["params"]}
        assert p["resid_mode"] in residuals.MODES and p["views"] in ab.VIEW_SETS and v["hypothesis"] in ab.HYPOTHESES
        assert v["adoptable"] == (v["hypothesis"] in ab.ENGINE_HYPOTHESES and v["id"] != "baseline")


@pytest.mark.skipif(not (config.RESULTS_DIR / "ablation.json").exists(), reason="ablation not built")
def test_ablation_json_contract():
    d = jsonio.load(config.RESULTS_DIR / "ablation.json")
    assert {"question", "text", "windows", "hypotheses", "variants", "selection", "transmission_gap"} <= set(d)
    assert isinstance(d["question"], str) and isinstance(d["text"], str) and d["text"]
    roles = {w["id"]: w["role"] for w in d["windows"]}
    assert roles == {"tech_bubble": "validation", "gfc_qe": "validation", "covid_ai": "test", "post_covid": "test"}
    assert all(len(w["test"]) == 2 for w in d["windows"])
    for h in d["hypotheses"]:
        assert {"id", "name", "test", "result", "verdict", "evidence"} <= set(h)
        assert h["verdict"] in ("supported", "rejected", "inconclusive")
    keys = {"cagr_mae_cond_risk", "cagr_mae_uncond_risk", "coverage_cond_risk", "pit_tail_share", "engine_abs_annual",
            "transmission_abs_annual", "scenario_abs_annual"}
    for v in d["variants"]:
        assert {"id", "name", "description", "params", "windows", "validation_mae", "test_mae",
                "validation_coverage", "test_coverage"} <= set(v)
        assert set(v["windows"]) == set(roles) and all(keys <= set(m) for m in v["windows"].values())
        assert 0 <= v["validation_mae"] < 0.5 and 0 <= v["test_coverage"] <= 1
    sel = d["selection"]
    assert {"rule", "chosen", "adopt", "reason"} <= set(sel) and isinstance(sel["adopt"], bool)
    assert sel["chosen"] in {v["id"] for v in d["variants"]}
    for wid, rows in d["transmission_gap"]["windows"].items():
        assert wid in roles
        for a, e in rows.items():
            assert a in bt.RISK_ASSETS and {"actual", "oracle", "gap_annual", "explained_by"} <= set(e)
