import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.analysis import forecast as fc
from scenariolab.analysis import regime_forecast as rf
from scenariolab.engine.factor_model import ModelSpec

needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")


@pytest.fixture(scope="module")
def panel():
    from scenariolab.context import load_panel_with_shocks
    return load_panel_with_shocks()


# ---------------------------------------------------------------- the look-ahead guard

def test_check_frame_refuses_the_future_and_ex_post_columns():
    idx = pd.date_range("2000-01-01", "2001-12-01", freq="MS")
    frame = pd.DataFrame({"label_rt": 0, "sahm_gap": 0.1, "rg_rt_contraction": 0.0}, index=idx)
    t = pd.Timestamp("2001-12-01")
    rf.check_frame(frame, t)                                          # real-time columns up to the origin: fine
    with pytest.raises(rf.LookAheadError, match="past the origin"):
        rf.check_frame(frame, pd.Timestamp("2001-06-01"))
    for col in ("label", "contraction", "episode_cpi_mean", "nber_recession", "rg_contraction", "rg_stagflation"):
        with pytest.raises(rf.LookAheadError, match="ex-post"):
            rf.check_frame(frame.assign(**{col: 0}), t)


@needs_panel
def test_realtime_regime_equals_the_whole_panel_run_truncated(panel):
    whole = rf.realtime_regime(panel, panel.index[-1])
    for t in ("1990-12", "2008-06", "2009-03", "2020-03", "2024-12"):
        t = pd.Timestamp(f"{t}-01")
        cut = rf.realtime_regime(panel, t)
        assert cut.index.max() == t and rf._same_frame(cut, whole.loc[:t])
    with pytest.raises(rf.LookAheadError):
        rf.realtime_regime(panel, pd.Timestamp("1990-12-15"))       # not a panel month
    assert set(rf.EXPOST_KEYS).isdisjoint(whole.columns)


@needs_panel
def test_a_scrambled_future_moves_no_weight_at_any_origin(panel):
    """Every row after the origin replaced (IP -40%, unemployment +5 pp, CPI y/y 12%, the rest shuffled and
    sign-flipped): every method's block-start weights at the origin stay the same at all origins, while the
    ex-post oracle's do move somewhere (the test has power)."""
    spec = ModelSpec()
    expost = rf.expost_regime(panel)
    moved = 0
    for i, t in enumerate(fc.forecast_origins(panel)):
        other = rf.scramble(panel, t, 100 + i)
        W1, _ = rf.weights_at(rf.Origin(panel, spec, t), rf.realtime_regime(panel, t), expost)
        W2, _ = rf.weights_at(rf.Origin(other, spec, t), rf.realtime_regime(other, t), rf.expost_regime(other))
        for k in W1:
            if "oracle" not in k:
                assert np.array_equal(W1[k], W2[k]), (t, k)
        moved += any(not np.array_equal(W1[k], W2[k]) for k in ("c_oracle", "cc_oracle"))
    assert moved > 0


@needs_panel
def test_the_current_forecast_is_the_forecast_modules_analog(panel):
    """With the same seed at the origin (common random numbers), the walk-forward's plain kernel at a share is
    the one-year forecast's analog at that share, draw for draw."""
    t = pd.Timestamp("2008-12-01")
    rows, diag = rf.walk_forward(panel, ModelSpec(), [t], seed_bases=(5,), n_paths=200)
    row = rows[5][0]
    ref = fc.forecast_origin(panel, ModelSpec(), t, n_paths=200, seed=5, analog_share=0.5)
    for a in fc.ASSETS:
        assert row["assets"][a]["an_2"] == {k: v for k, v in ref["assets"][a]["analog"].items() if k in rf.KEEP}
        assert row["assets"][a]["uncond"] == {k: v for k, v in ref["assets"][a]["uncond"].items() if k in rf.KEEP}
    assert diag[0]["flat"]["an_2"] is ref["flat"] is False
    assert {k for k in row["assets"]["us_equity"] if k != "realized"} >= {"a1_2", "a2_32", "b25_4", "bc0_8", "c", "cc",
                                                                         "c_oracle", "naive"}


# ---------------------------------------------------------------- the adoption rule

def _seeds(val, test, ic=0.0, cov=0.0):
    return [{"val": v, "test": t, "ic_delta": ic, "cov_delta": cov} for v, t in zip(val, test)]


def test_adoption_needs_both_halves_in_every_seed_set():
    better = [-0.001] * 5
    ok = rf.adoption(_seeds(better, better))
    assert ok["passes"] and ok["better_both_halves_seeds"] == 5
    four = rf.adoption(_seeds(better, [-0.001] * 4 + [0.0002]))     # better in 4 of 5 seed sets: fails
    assert not four["passes"] and four["better_both_halves_seeds"] == 4
    assert not four["checks"]["both_halves_every_seed"] and four["checks"]["rank_ic"] and four["checks"]["coverage"]
    assert not rf.adoption(_seeds(better, better, ic=-0.06))["passes"]          # rank IC down more than 0.05
    assert rf.adoption(_seeds(better, better, ic=-0.05))["passes"]
    assert not rf.adoption(_seeds(better, better, cov=-0.03))["passes"]         # coverage down more than 0.02
    assert not rf.adoption([])["passes"]


def test_thin_pool_guard_shrinks_toward_the_fallback():
    T = 240
    n = T - fc.HORIZON + 1
    flat = np.full(n, 1.0 / n)
    thin = np.zeros(n)
    thin[10:14] = 0.25                                              # four neighbouring starts: one episode
    q, kappa = rf.guard(thin, flat, T)
    assert 0.0 <= kappa < 1.0 and q.sum() == pytest.approx(1.0)
    m, e = rf.thinness(q, T)
    assert m >= rf.MIN_ENS and e >= rf.MIN_EPISODES
    q, kappa = rf.guard(flat, thin, T)                              # rich enough: kept as is
    assert kappa == 1.0 and np.array_equal(q, flat)
    q, kappa = rf.guard(np.zeros(n), flat, T)                       # empty pool: the fallback
    assert kappa == 0.0 and np.array_equal(q, flat)


def test_settings_and_their_names():
    assert rf.settings_of("a1") == ["a1_32", "a1_16", "a1_8", "a1_4", "a1_2"]
    assert len(rf.settings_of("b")) == 15 and rf.settings_of("cc") == ["cc"]
    assert rf.setting_text("b50_2") == "λ 0.5，可用起点的 1/2" and rf.setting_text("a2_8") == "可用起点的 1/8"
    assert rf.setting_text("b0_4") == "λ 0，可用起点的 1/4"


def test_a_test_half_gain_is_stated_with_the_forecast_modules_claim_rule():
    """Relative CRPS changes and paired t: negative = better. A gain is claimed only beyond CLAIM_SKILL with
    |t| >= CLAIM_T; otherwise the text says it is within the noise (the cc family today: -2.83%, t -1.29)."""
    assert rf.gain_note(-0.0283, -1.29) == "（t = -1.3，在随机误差之内）"
    assert rf.gain_note(-0.03, -2.4) == "（t = -2.4）"
    assert rf.gain_note(-0.003, -2.4) == "（t = -2.4，在随机误差之内）"           # below CLAIM_SKILL
    assert rf.gain_note(-0.03, None) == "（在随机误差之内）"
    fam = {"id": "cc", "name": "只从同一实时体制的起点抽（3 组）", "passes": False, "better_both_halves_seeds": 0,
           "risk_crps_rel": {"val": {"mean": 0.0408}, "test": {"mean": -0.0283}}, "t": {"val": 1.31, "test": -1.29},
           "fragility_seed23": {"best_origins": ["2007-12", "2021-12"], "test_without_best_rel": -0.0089}}
    text, verdict = rf.report_text([fam], {"name": "现行一年期预测"}, {}, 42, 5, fc.SPLIT)
    assert "检验期 -2.83%（t = -1.3，在随机误差之内）、验证期 +4.08%" in text and verdict.startswith("未采用")
