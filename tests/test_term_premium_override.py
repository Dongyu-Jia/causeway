"""P0.6: a per-phase term_premium override in the monthly engine, so a stated policy path (r_star) no longer forces
the 10-year's reversion target (SIM_DESIGN §14, §7.3)."""
import numpy as np
import pytest

from scenariolab import config
from scenariolab.analysis import backtest as bt
from scenariolab.engine.factor_model import ModelSpec
from scenariolab.engine.simulate import DEFAULT_REVERSION, simulate
from scenariolab.scenarios.schema import SPEC_JSON_SCHEMA, TERM_PREMIUM_BOUNDS, validate_spec

needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
PLAN = [24, 24]


@pytest.fixture(scope="module")
def window():
    if not config.PANEL_PATH.exists():
        pytest.skip("panel not built")
    from scenariolab.context import load_panel_with_shocks
    from scenariolab.engine.pooling import BLOCK, pool_phase
    panel = load_panel_with_shocks()
    model, db, anchor = bt.fit_window(panel, ModelSpec(), "2004-12")
    prior = [pool_phase(db.frame, [], db.prior(), m, BLOCK, rare=db.rare).blocks for m in PLAN]
    return db, anchor, prior


def _run(window, overrides, seed=4):
    db, anchor, prior = window
    return simulate(db, anchor, prior, PLAN, n_paths=80, seed=seed, overrides=overrides)


@needs_panel
def test_a_phase_term_premium_moves_only_that_phases_10_year_target(window):
    base = _run(window, [{}, {}])
    same = _run(window, [{}, {"term_premium": DEFAULT_REVERSION["term_premium"]}])
    assert np.array_equal(base.total, same.total) and np.array_equal(base.macro["y10"], same.macro["y10"])
    steep = _run(window, [{}, {"term_premium": 2.5}])
    assert np.array_equal(base.macro["y10"][:, :25], steep.macro["y10"][:, :25])         # phase 1 untouched
    gap = steep.macro["y10"][:, -1].astype(float) - base.macro["y10"][:, -1]
    assert (gap > 0).mean() > 0.95 and gap.mean() > 0.2                                  # pulled up in phase 2
    assert np.array_equal(base.macro["ff"], steep.macro["ff"])                           # the policy rate is not


@needs_panel
def test_a_stated_policy_path_need_not_force_the_10_year_target(window):
    """r_star moves both reversion targets; with term_premium = default + (default r* - r_star) the 10-year target,
    and so the whole 10-year path, is the baseline's while the policy rate follows the stated path."""
    base = _run(window, [{}, {}])
    r_star = -1.0
    policy = _run(window, [{"r_star": r_star}, {"r_star": r_star}])
    keep = DEFAULT_REVERSION["term_premium"] + DEFAULT_REVERSION["r_star"] - r_star
    both = _run(window, [{"r_star": r_star, "term_premium": keep}, {"r_star": r_star, "term_premium": keep}])
    assert (policy.macro["ff"][:, -1] < base.macro["ff"][:, -1]).mean() > 0.95
    assert np.array_equal(both.macro["ff"], policy.macro["ff"])                          # the policy path holds
    assert (policy.macro["y10"][:, -1] < base.macro["y10"][:, -1]).mean() > 0.95          # r_star alone drags y10
    assert np.allclose(both.macro["y10"], base.macro["y10"], rtol=0, atol=2e-6)          # ... the override does not
    j = config.ASSET_IDS.index("treasury_10y")
    assert np.allclose(both.total[:, :, j], base.total[:, :, j], rtol=0, atol=2e-6)


def test_validate_spec_keeps_and_bounds_a_phase_term_premium():
    import pandas as pd
    idx = pd.date_range("1990-01-01", periods=240, freq="MS")
    frame = pd.DataFrame({"f_oil": np.random.default_rng(0).normal(0, 0.05, 240)}, index=idx)
    spec = {"name": "期限溢价", "phases": [
        {"name": "一", "months": 60, "views": [], "overrides": {"r_star": -1.0, "term_premium": 2.5}},
        {"name": "二", "months": 60, "views": [], "overrides": {"term_premium": 9.0}},
        {"name": "三", "months": 60, "views": [], "overrides": {"term_premium": None}}]}
    clean, report = validate_spec(spec, frame, 180)
    ov = [p["overrides"] for p in clean["phases"]]
    assert ov[0] == {"r_star": -1.0, "term_premium": 2.5} and ov[1] == {"term_premium": TERM_PREMIUM_BOUNDS[1]}
    assert ov[2] == {}
    assert {"phase": 2, "var": "term_premium", "requested": 9.0, "used": 3.0} in report["clipped"]
    # the LLM's output schema does not offer it (the translator keeps its four override keys)
    phase_ov = SPEC_JSON_SCHEMA["properties"]["phases"]["items"]["properties"]["overrides"]
    assert "term_premium" not in phase_ov["properties"]
