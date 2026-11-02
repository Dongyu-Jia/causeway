"""engine/anchors.py: long-run anchors knowable at a train end (SIM_DESIGN §5.5, P0.1).

The rule's check values on the real panel, the sampling and release-lag convention on a synthetic panel, the
truncation property (a panel cut at the train end gives the same anchors), the Michigan switch of pi_lr, and that
forward runs keep DEFAULT_REVERSION.
"""
import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.engine import anchors
from scenariolab.engine.simulate import DEFAULT_REVERSION

needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")

# SIM_DESIGN §5.5 (design/d02 and d05): r* / term / spread / unemployment (80Q), two decimals
CHECK = {"1994-12": (2.19, 1.67, 2.01, 7.05), "2004-12": (1.79, 1.24, 2.26, 5.73),
         "2014-12": (-0.35, 1.73, 2.71, 6.01), "2026-06": (-0.74, 0.46, 2.04, 5.77)}
CORE_40Q = {"1994-12": 4.11, "2004-12": 2.31, "2014-12": 1.93, "2026-06": 3.12}      # d02 core_10y (pi_lr rule)


@pytest.fixture(scope="module")
def panel():
    from scenariolab.data.panel import load_panel
    return load_panel()


def _synthetic(n_months: int = 480, start: str = "1980-01-01") -> pd.DataFrame:
    """Every input equals a code of its own month (column k: k * 10000 + months since start), so a mean says
    exactly which months were read."""
    idx = pd.date_range(start, periods=n_months, freq="MS")
    t = np.arange(n_months, dtype=float)
    return pd.DataFrame({"s_ff": 1e4 + t, "s_y10": 3e4 + t, "s_spread": 4e4 + t, "s_core_yoy": 2e4 + t,
                         "s_unrate": 5e4 + t}, index=idx)


def _mean_code(end_pos: int, n: int, lag: int) -> float:
    return float(np.mean([end_pos - 3 * k - lag for k in range(n)]))


def test_samples_are_quarters_ending_at_the_train_end_with_release_lags():
    P = _synthetic()
    end = P.index[300]
    a = anchors.train_end_anchors(P, end)
    q, u = anchors.QUARTERS, anchors.UNRATE_QUARTERS
    # r* = ff (lag 0) - core (lag 1): the codes differ by 1e4 and one month
    assert a["r_star"] == pytest.approx((1e4 + _mean_code(300, q, 0)) - (2e4 + _mean_code(300, q, 1)))
    assert a["term"] == pytest.approx(2e4)                                  # y10 and ff: same months
    assert a["spread"] == pytest.approx(4e4 + _mean_code(300, q, 0))
    assert a["unrate"] == pytest.approx(5e4 + _mean_code(300, u, 1))
    assert a["pi_lr"] == pytest.approx(2e4 + _mean_code(300, q, 1)) and a["pi_lr_source"] == "core_cpi_40q"
    assert a["end"] == end.strftime("%Y-%m")
    assert a["from"]["unrate"] == P.index[300 - 3 * (u - 1) - 1].strftime("%Y-%m")
    assert a["from"]["r_star"] == P.index[300 - 3 * (q - 1) - 1].strftime("%Y-%m")
    months = anchors.sample_months(end, 3, lag=1)
    assert months == [P.index[299], P.index[296], P.index[293]]


def test_only_rows_up_to_the_train_end_are_read():
    P = _synthetic()
    rng = np.random.default_rng(0)
    for pos in (250, 301, 333, 479):
        end = P.index[pos]
        later = P.copy()
        later.loc[later.index > end] = rng.normal(size=later.loc[later.index > end].shape) * 1e6
        assert anchors.train_end_anchors(later, end) == anchors.train_end_anchors(P, end)
        assert anchors.train_end_anchors(P.loc[:end], end) == anchors.train_end_anchors(P, end)


def test_too_little_history_raises_in_chinese():
    P = _synthetic(300)
    with pytest.raises(ValueError, match="s_unrate 有缺失"):
        anchors.train_end_anchors(P, P.index[200])                      # 80 quarters need 240 months
    with pytest.raises(ValueError, match="面板缺少 s_spread"):
        anchors.train_end_anchors(P.drop(columns="s_spread"), P.index[-1])


def test_pi_lr_switches_to_the_michigan_median_from_1990q2():
    P = _synthetic(600, "1970-01-01")
    mich = pd.Series(np.nan, index=P.index)
    mich.loc["1990-04-01":] = 2.5 + np.arange(len(mich.loc["1990-04-01":])) / 1000.0
    early = anchors.train_end_anchors(P.assign(s_mich5y=mich.fillna(9.9)), "1990-03")
    assert early["pi_lr_source"] == "core_cpi_40q"                       # before 1990Q2: the core-CPI rule
    late = anchors.train_end_anchors(P.assign(s_mich5y=mich), "1994-12")
    assert late["pi_lr_source"] == "michigan_5_10y" and late["pi_lr"] == float(mich.loc["1994-12-01"])
    gap = mich.copy()
    gap.loc["1994-09-01":] = np.nan                                         # the latest survey is 1994-08
    assert anchors.train_end_anchors(P.assign(s_mich5y=gap), "1994-12")["pi_lr_source"] == "core_cpi_40q"   # 4 months
    assert anchors.train_end_anchors(P.assign(s_mich5y=gap), "1994-10")["pi_lr"] == float(mich.loc["1994-08-01"])  # 2
    passed = anchors.train_end_anchors(P, "2000-06", mich_lr=mich)          # an explicit series, no panel column
    assert passed["pi_lr"] == float(mich.loc["2000-06-01"])
    # pi_lr never changes the other anchors
    base = anchors.train_end_anchors(P, "1994-12")
    assert all(late[k] == base[k] for k in ("r_star", "term", "spread", "unrate"))


def test_reversion_maps_the_anchors_onto_the_engines_keys():
    a = {"r_star": -0.5, "term": 1.2, "spread": 2.3, "unrate": 6.1, "pi_lr": 2.0}
    rev = anchors.reversion(a)
    assert rev == {"r_star": -0.5, "term_premium": 1.2, "spread_level": 2.3, "unrate_level": 6.1}
    assert set(rev) <= set(DEFAULT_REVERSION)
    P = _synthetic()
    assert anchors.window_reversion(P, P.index[400]) == anchors.reversion(anchors.train_end_anchors(P, P.index[400]))


def test_default_reversion_is_still_the_forward_judgement():
    """Forward scenarios keep today's judgement values (SIM_DESIGN §3.8); only backtests use the train-end rule."""
    assert {k: DEFAULT_REVERSION[k] for k in ("r_star", "term_premium", "spread_level", "unrate_level")} == {
        "r_star": 0.75, "term_premium": 0.75, "spread_level": 2.0, "unrate_level": 4.8}


@needs_panel
def test_check_values_of_the_rule(panel):
    for end, (r_star, term, spread, unrate) in CHECK.items():
        a = anchors.train_end_anchors(panel, end)
        got = (a["r_star"], a["term"], a["spread"], a["unrate"])
        assert np.allclose(got, (r_star, term, spread, unrate), atol=0.005 + 1e-9), (end, got)
        assert a["pi_lr_source"] == "core_cpi_40q" and a["pi_lr"] == pytest.approx(CORE_40Q[end], abs=0.005 + 1e-9)


@needs_panel
def test_truncation_on_the_real_panel(panel):
    """Building at train end t equals the full build's value at t, for December and other months."""
    for end in ("1987-12", "1994-12", "2001-07", "2008-09", "2019-12", "2026-06"):
        assert anchors.train_end_anchors(panel.loc[:end], end) == anchors.train_end_anchors(panel, end), end


@needs_panel
def test_forward_scenarios_simulate_with_the_default_reversion(monkeypatch):
    """The scenario runner passes no reversion for a spec without one (simulate falls back to DEFAULT_REVERSION),
    while a backtest window passes the train-end anchors."""
    from scenariolab.analysis import backtest as bt
    from scenariolab.context import EngineContext
    from scenariolab.scenarios import runner
    ctx = EngineContext.load(selection=False)
    seen = []
    real = runner.simulate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("reversion"))
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "simulate", spy)
    spec = {"name": "无视图", "phases": [{"name": "常态", "months": config.HORIZON_MONTHS, "views": []}]}
    runner.run_spec(ctx, spec, n_paths=50, seed=1, attribution=False, regimes=False)
    assert seen == [None]
    w = next(x for x in bt.WINDOWS if x["id"] == "tech_bubble")
    rev = anchors.window_reversion(ctx.panel, w["train_end"])
    assert rev != {k: DEFAULT_REVERSION[k] for k in rev}
    assert rev["r_star"] == pytest.approx(CHECK["1994-12"][0], abs=0.005)
