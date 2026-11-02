"""The quarterly stack's asset layer, valuation block and evaluation harness (SIM_DESIGN §4, §6, §14 P2).

Exit tests of P2:
  * L0 reproduces the research round's L0 1Q scores (0.126 / 0.261 / 0.441) to 1e-3: the monthly-path evaluation of
    the fitted F does; the quarterly form (what a quarterly cell can compute) is measured against them, with its gap
    split into the start-of-quarter tp10 switch and the intra-quarter compounding of monthly predictions;
  * a drift-only model scores R² = 0 to 1e-12 (monthly and quarterly);
  * the research harness's selfcheck() values are reproduced on the app's panel;
  * the placebo row is present in the valuation anatomy;
  * the T-A candidate flags are off by default.
The synthetic tests run anywhere; the data tests need the monthly and quarterly panels (and Shiller's workbook for
the harness's valuation columns) and skip without them. Nothing downloads and nothing calls an LLM.
"""
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.data.panel import par_bond_return
from scenariolab.evaluation import harness as eh
from scenariolab.qmodel import assets as qa
from scenariolab.qmodel import config_q as cq
from scenariolab.qmodel import valuation as val

SHILLER = config.RAW_DIR / "shiller" / "ie_data.xls"
needs_panels = pytest.mark.skipif(not (config.PANEL_PATH.exists() and cq.PANEL_Q_PATH.exists()),
                                  reason="monthly or quarterly panel not built")
needs_shiller = pytest.mark.skipif(not (config.PANEL_PATH.exists() and cq.PANEL_Q_PATH.exists() and SHILLER.exists()),
                                   reason="panels or Shiller's workbook missing (data/raw/shiller/ie_data.xls)")


@pytest.fixture(scope="module")
def P():
    from scenariolab.data import panel as mpanel
    return mpanel.load_panel()


@pytest.fixture(scope="module")
def Q():
    from scenariolab.qmodel import panel as qpanel
    return qpanel.load_panel()


@pytest.fixture(scope="module")
def layer_today(P, Q):
    return qa.AssetLayer.fit("2026Q3", panel=P, Q=Q, n_boot=50, seed=4)


# ================================================================ synthetic (no data needed)

def test_ta_candidate_flags_are_off_by_default():
    assert cq.TA_DEFAULT == ()
    assert inspect.signature(qa.AssetLayer.fit).parameters["candidates"].default == ()
    assert inspect.signature(eh.quarterly_model_fn).parameters["candidates"].default == ()
    assert set(cq.TA_CANDIDATES) == {"C1", "C2", "C3", "C4", "NN", "NN-res"}
    for cid, c in cq.TA_CANDIDATES.items():
        assert c["mode"] in ("quarter", "path", "report_only"), cid
        if c["mode"] == "report_only":
            assert c.get("available") is False
            with pytest.raises(ValueError, match="只作对照"):
                qa.candidate_terms(cid, {}, "us_equity")
    # the C1 scopes are the schema's tier-1s scopes; d_tips10 only from 2003Q2
    assert cq.TA_CANDIDATES["C1"]["scoped"]["d_sent"] == ["financials"]
    assert cq.TA_CANDIDATES["C1"]["first"] == {"d_tips10": "2003Q2"}


def test_factor_inputs_are_field_over_scale_and_the_start_switch():
    feats = ["oil", "ip", "cpi_goods", "cpi_services", "ff", "y10", "spread", "usd", "unrate", "gz", "tp10_x_core4",
             "metals_oil"]
    frame = {"d_oil": np.array([10.0, -20.0]), "d_ip": np.array([1.0, 2.0]), "d_cpi_goods": np.array([4.0, 8.0]),
             "d_cpi_serv": np.array([2.0, 4.0]), "d_ff": np.array([0.5, -1.0]), "d_y10": np.array([0.2, 0.1]),
             "d_spread": np.array([0.3, 0.0]), "d_usd": np.array([5.0, -5.0]), "d_unrate": np.array([0.1, 1.0]),
             "d_gz": np.array([0.4, 2.0]), "d_tp10": np.array([0.3, 0.6]), "s_core4": np.array([0.0, 1.0]),
             "d_metals_oil": np.array([7.0, 3.0])}
    F = qa.factor_inputs(frame, feats)
    assert np.allclose(F["oil"], [0.10, -0.20]) and np.allclose(F["ip"], [0.01, 0.02])
    assert np.allclose(F["cpi_goods"], [0.01, 0.02]) and np.allclose(F["cpi_services"], [0.005, 0.01])
    assert np.allclose(F["usd"], [0.05, -0.05]) and np.allclose(F["gz"], frame["d_gz"])
    assert np.allclose(F["tp10_x_core4"], [0.0, 0.6])                   # switched on at the quarter's start
    assert np.allclose(F["metals_oil"], [0.07, 0.03])
    qa.check_spec(qa.l0_spec())
    with pytest.raises(ValueError):
        qa.check_spec(qa.ModelSpec.from_dict({**cq.L0_SPEC_FALLBACK, "lags": 1}))


def test_mechanical_identities():
    y0 = np.array([2.0, 4.5, 8.0])
    p, inc = qa.treasury_return(y0, np.zeros(3))
    assert np.allclose(p, 0.0, atol=1e-12) and np.allclose(inc, y0 / 400.0)          # a par bond stays at par
    p2, _ = qa.treasury_return(y0, np.array([1.0, -0.5, 0.25]))
    ref, _ = par_bond_return(pd.Series(y0), pd.Series(y0 + np.array([1.0, -0.5, 0.25])))
    assert np.allclose(p2, np.log1p(ref.to_numpy()), atol=1e-14)                     # data/panel's repricing formula
    assert np.allclose(qa.cash_return(3.0, 3.0), 3.0 * np.log1p(3.0 / 1200.0))
    assert qa.cash_return(0.0, 3.0) == pytest.approx(np.log1p(1.0 / 1200) + np.log1p(2.0 / 1200), abs=1e-15)
    frame = {"d_oil": np.array([26.0]), "s_y10": np.array([4.0]), "d_y10": np.array([0.0]),
             "s_tb3": np.array([2.0]), "d_ff": np.array([-3.0])}
    mech = qa.mechanical_returns(frame)
    assert mech["crude_oil"][0][0] == pytest.approx(0.26) and mech["crude_oil"][1][0] == 0.0
    assert qa.bill_end(frame)[0] == 0.0                                  # floored at the zero bound
    assert mech["cash"][1][0] == pytest.approx(math.expm1(mech["cash"][0][0]))


def _synthetic_pool(n=60, gap_at=30, seed=0):
    rng = np.random.default_rng(seed)
    qs = list(pd.period_range("1980Q1", periods=gap_at, freq="Q")) + \
        list(pd.period_range("1995Q1", periods=n - gap_at, freq="Q"))
    idx = pd.PeriodIndex(qs, freq="Q")
    sv = pd.Series(np.exp(rng.normal(-2.3, 0.4, n)), index=idx)
    E = pd.DataFrame(rng.normal(0, 0.05, (n, len(qa.POOL_COLUMNS))), index=idx, columns=qa.POOL_COLUMNS)
    E[qa.MECH] = 0.0
    E["us_equity"] *= np.exp(0.6 * (np.log(sv) - np.log(sv).mean()))            # volatility-dependent residual
    return qa.ResidualPool.fit(E, sv, block=2), idx


def test_residual_pool_draws_consecutive_blocks_with_labels():
    pool, idx = _synthetic_pool()
    assert np.allclose(pool.z.mean(axis=0), 0.0, atol=1e-12)
    assert np.all((pool.gamma >= cq.VOL_GAMMA_BOUNDS[0]) & (pool.gamma <= cq.VOL_GAMMA_BOUNDS[1]))
    assert np.all(pool.gamma[[qa.I[a] for a in qa.MECH]] == 0.0)
    assert pool.gamma[qa.I["us_equity"]] > 0.2                           # the planted volatility elasticity
    assert 29 not in pool.starts(2)                                      # no block across the 1987 -> 1995 gap
    rng = np.random.default_rng(1)
    ix = pool.draw_index(500, 7, rng)
    assert ix.shape == (500, 7)
    lab = pd.PeriodIndex(pool.label_array(ix).ravel(), freq="Q").asi8.reshape(ix.shape)
    assert np.all(lab[:, 1::2][:, :3] - lab[:, 0::2][:, :3] == 1)         # each 2-quarter block is consecutive
    for B in cq.RESID_BLOCK_ARMS:
        assert pool.draw_index(10, 9, rng, block=B).shape == (10, 9)
    d = pool.draws(ix[:, 0], np.full(500, math.exp(pool.lv_mean)))
    assert np.allclose(d, pool.z[ix[:, 0]])                              # at the mean volatility: unscaled
    hi = pool.draws(ix[:, 0], np.full(500, 3 * math.exp(pool.lv_mean)))
    assert np.all(np.abs(hi[:, qa.I["us_equity"]]) >= np.abs(d[:, qa.I["us_equity"]]) - 1e-15)


def test_valuation_statements_parse_and_refuse():
    ok = val.parse_statements([{"quarters": [1, 2, 3, 4], "mode": "rerate", "direction": "-", "size": "明显",
                                "quote": "估值压缩"}, {"quarters": [5], "mode": "hold", "quote": "x"},
                               {"quarters": [6, 7], "mode": "inferred", "reversion_half_life_years": 20, "quote": "y"}],
                              8)
    assert [s.mode for s in ok] == ["rerate", "hold", "inferred"] and ok[0].direction == -1 and ok[2].active
    bad = [([{"quarters": [1], "mode": "pull", "quote": ""}], "模式"),
           ([{"quarters": [1], "mode": "rerate", "size": "明显", "quote": ""}], "方向"),
           ([{"quarters": [1], "mode": "rerate", "direction": "+", "size": "很大", "quote": ""}], "幅度"),
           ([{"quarters": [9], "mode": "hold", "quote": ""}], "超出范围"),
           ([{"quarters": [1], "mode": "inferred", "reversion_half_life_years": 10, "quote": ""}], "至少 14 年"),
           ([{"quarters": [1], "mode": "hold", "reversion_half_life_years": 20, "quote": ""}], "inferred"),
           ([{"quarters": [1, 2], "mode": "hold", "quote": ""},
             {"quarters": [2], "mode": "rerate", "direction": "+", "size": "温和", "quote": ""}], "重叠")]
    for entries, msg in bad:
        with pytest.raises(ValueError, match=msg):
            val.parse_statements(entries, 8)
    with pytest.raises(ValueError, match="测试模式"):
        val.parse_statements([{"quarters": [1], "mode": "hold", "quote": ""}], 4, mode="test")
    assert val.parse_statements([{"quarters": [1], "mode": "inferred", "quote": ""}], 4, mode="test")[0].active is False


def test_valuation_statements_are_delivered_within_their_sd():
    rng = np.random.default_rng(0)
    n, H = 20000, 6
    price = rng.normal(0.02, 0.08, (n, H, len(qa.ASSETS)))
    y_fund = rng.normal(0.015, 0.01, (n, H))
    bins = {"温和": 0.12, "明显": 0.20, "极端": 0.34}
    stm = val.parse_statements([{"quarters": [1, 2, 3, 4], "mode": "rerate", "direction": "+", "size": "明显",
                                 "quote": ""}, {"quarters": [5, 6], "mode": "hold", "quote": ""}], H)
    beta = {"nasdaq": 1.4, "energy": 0.6}
    P2, shift = val.apply_statements(price, y_fund, stm, bins=lambda span: bins, resid_beta=beta, rng=rng)
    ym = P2[:, :, qa.I["us_equity"]] - y_fund
    s1, s2 = ym[:, :4].sum(1), ym[:, 4:].sum(1)
    sd1, sd2 = 0.25 * 0.20, 0.25 * 0.12
    assert s1.mean() == pytest.approx(0.20, abs=0.002) and s1.std() == pytest.approx(sd1, rel=0.03)
    assert np.mean(np.abs(s1 - 0.20) <= sd1) == pytest.approx(0.68, abs=0.05)       # soft: 68% ± 5% within 1 sd
    assert s2.mean() == pytest.approx(0.0, abs=0.002) and np.mean(np.abs(s2) <= sd2) == pytest.approx(0.68, abs=0.05)
    assert np.allclose(shift[:, :, qa.I["nasdaq"]], 1.4 * shift[:, :, qa.I["us_equity"]])   # inherit the re-rating
    assert np.allclose(shift[:, :, qa.I["gold"]], 0.0) and np.allclose(shift[:, :, qa.I["treasury_10y"]], 0.0)
    assert np.allclose(P2 - price, shift)
    # an explicit reversion pulls an expensive market down, never by default
    rev = val.parse_statements([{"quarters": list(range(1, 7)), "mode": "inferred", "reversion_half_life_years": 14,
                                 "quote": ""}], H)
    P3, sh3 = val.apply_statements(price, y_fund, rev, bins=lambda s: bins, resid_beta=beta, rng=rng,
                                   log_cape0=math.log(41.2), m_star=math.log(20.0))
    k = val.reversion_kappa(14)
    assert k == pytest.approx(math.log(2) / 56)
    assert sh3[:, 0, qa.I["us_equity"]] == pytest.approx(-k * (math.log(41.2) - math.log(20.0)))
    assert np.all(sh3[:, :, qa.I["us_equity"]] < 0)
    P4, sh4 = val.apply_statements(price, y_fund, val.parse_statements([], H), bins=lambda s: bins, resid_beta=beta,
                                   rng=rng)
    assert np.array_equal(P4, price) and not sh4.any()


def test_size_bins_and_implied_paths():
    y = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.2, -0.05])
    b = val.size_bins_from(y, 2)
    sums = np.abs(np.convolve(y, np.ones(2), "valid"))
    assert b["温和"] == pytest.approx(np.quantile(sums, 0.5)) and b["极端"] == pytest.approx(np.quantile(sums, 0.95))
    with pytest.raises(ValueError):
        val.size_bins_from(y, 9)
    cape, pdr = val.implied_paths(math.log(40.0), math.log(100.0), np.array([[0.1, -0.2]]), np.array([[0.0, 0.05]]))
    assert cape.shape == (1, 3) and cape[0, -1] == pytest.approx(math.log(40.0) - 0.1)
    assert pdr[0, -1] == pytest.approx(math.log(100.0) + 0.05)
    txt = val.statement_text(4, math.log(0.8), math.log(1.03))
    assert txt == "估值假设：这一条相当于直接假设美股价格在这 4 个季度里 -20%（其中基本面腿 +3%）"


def test_harness_metrics_scoring_helpers():
    # CRPS of a degenerate ensemble is the absolute error
    assert eh.crps_samples(np.full((50, 1), 0.3), np.array([0.1]))[0] == pytest.approx(0.2)
    # a constant (placebo) prediction of the multiple has no corr and no slope, only R² against zero
    dm = np.array([0.1, -0.2, 0.05, 0.3])
    sk = eh.multiple_skill(np.full(4, 0.01), dm)
    assert math.isnan(sk["corr_pred_dm"]) and math.isnan(sk["slope_dm_on_pred"])
    assert sk["r2_dm_vs_zero"] == pytest.approx(1 - np.sum((dm - 0.01) ** 2) / np.sum(dm ** 2))
    perfect = eh.multiple_skill(dm, dm)
    assert perfect["corr_pred_dm"] == pytest.approx(1.0) and perfect["slope_dm_on_pred"] == pytest.approx(1.0)
    # metrics: a forecast equal to the benchmark scores R² = 0; two identical models differ by 0
    rng = np.random.default_rng(0)
    origins = pd.date_range("2005-01-01", periods=40, freq="QS")
    rows = []
    for name in ("a", "b"):
        for o in origins:
            for a in eh.SCORE_ASSETS:
                y = float(rng.normal(0.02, 0.08))
                rows.append({"model": name, "protocol": "split", "group": "2005-2014", "train_end": "2004-12",
                             "origin": o, "end": o + pd.DateOffset(months=2), "h_q": 1, "months": 3, "asset": a,
                             "y": y, "yhat": 0.02, "bench": 0.02, "q10": -0.1, "q90": 0.14, "pit": 0.5, "crps": 0.04,
                             "crps_bench": 0.05, "yhat_mult": np.nan, "n_fit": 100})
    paths = pd.DataFrame(rows)
    paths.loc[paths.model == "b", "y"] = paths.loc[paths.model == "a", "y"].to_numpy()
    m = eh.metrics(paths, n_boot=50)
    assert m.r2.abs().max() < 1e-12
    assert m[m.asset == "mean10"].crpss.iloc[0] == pytest.approx(0.2)
    d = eh.paired_diff(paths, "a", "b", n_boot=50)
    assert len(d) == 1 and d.diff_r2.iloc[0] == pytest.approx(0.0, abs=1e-12)
    R = rng.normal(0, 0.05, (80, 3))
    bd = eh.bench_draws(R - R.mean(0), np.array([0.01, 0.02, 0.03]), 4, 2000, 3)
    assert bd.shape == (2000, 3) and np.allclose(bd.mean(0), [0.01, 0.02, 0.03], atol=0.01)


# ================================================================ data (the exit tests)

@needs_panels
def test_l0_reproduces_the_research_l0_1q_scores(P, Q):
    """Exit test: the fitted F scored on its monthly paths reproduces 0.126 / 0.261 / 0.441 to 1e-3. The quarterly
    form is the same F on the quarter's sums; its gap is exactly the tp10 switch plus the intra-quarter compounding
    (measured: about -0.002 / -0.022 / -0.010, mostly 2008Q4, 2020Q1 and the 2021Q4 switch)."""
    rep = eh.l0_reproduction(P, Q)
    assert set(rep) == set(cq.L0_CHECK_R2_1Q)
    for g, r in rep.items():
        assert abs(r["monthly_path"] - cq.L0_CHECK_R2_1Q[g]) <= 1e-3, g
        assert r["identity_max_abs"] <= 1e-12, g                          # field / scale = the window sum
        assert r["gap_vs_research"] == pytest.approx(r["switch_effect"] + r["compounding_effect"]
                                                     + r["monthly_path"] - r["research"], abs=1e-12)
        assert r["compounding_effect"] <= 0.0, g                          # the quarterly sum loses the convexity
        assert -0.03 < r["gap_vs_research"] < 0.0, g
    assert rep["1995-2004"]["switch_effect"] == 0.0 and rep["2005-2014"]["switch_effect"] == 0.0
    assert rep["2015-2026"]["switch_effect"] < 0.0                       # core CPI crossed 4% inside 2021Q4
    assert "2008Q4" in rep["2005-2014"]["largest_quarters"]


@needs_shiller
def test_the_harness_reproduces_the_research_l0_and_scores_drift_at_zero(Q):
    """Exit tests: the ported conditional_paths + metrics give the research L0's 1Q scores for F, and a drift-only
    model (monthly or quarterly) scores R² = 0 to 1e-12 at 1Q and 4Q."""
    Ph = eh.load_panel(valuation=False)
    pf = eh.conditional_paths(eh.linear_model(eh.F), Ph, protocol="split", horizons_q=(1,), band=False)
    pdm = eh.conditional_paths(eh.drift_model(), Ph, protocol="split", horizons_q=(1, 4), band=False)
    pdq = eh.conditional_paths(eh.quarterly_drift_fn(Q), Ph, protocol="split", horizons_q=(1, 4), band=False)
    m = eh.metrics(pd.concat([pf, pdm, pdq]), n_boot=20)
    r2 = eh.mean_table(m, "r2")
    for g, ref in cq.L0_CHECK_R2_1Q.items():
        assert abs(r2.loc[("F", "split", g), 1] - ref) <= 1e-3
    assert m[m.model.isin(["drift", "drift_q"])].r2.abs().max() <= 1e-12


@needs_shiller
def test_selfcheck_values_are_reproduced():
    """Exit test: simharness.selfcheck() on the app's panel (research: b0 4.7e-16, resid 1.8e-15, draw identical,
    mask ok, v_lpd 1.8e-15, drift 6.7e-16, the 2026 Shiller workbook)."""
    res = eh.selfcheck(verbose=False, quarterly=False)
    assert res["b0_vs_assetfactormodel_max_abs"] <= 1e-12
    assert res["b0_resid_vs_monthdb_max_abs"] <= 1e-12
    assert res["engine_draw_identical"] is True
    assert res["mask_ok"] is True
    assert res["v_lpd_vs_div_yield_max_abs"] <= 1e-12
    assert res["drift_model_max_abs_r2"] <= 1e-12
    assert str(res["shiller_source"]).endswith("data/raw/shiller/ie_data.xls")


@needs_shiller
def test_the_valuation_anatomy_carries_a_placebo_row(Q):
    """Exit test: every anatomy cell has a placebo (constant drift forecast) row next to the model's; the placebo
    has no corr or slope, and the retired corr(miss, Δmultiple)² gives it 0.83-1.00 on the CAPE multiple (the
    identity the critic found), at least the model's in every cell."""
    Ph = eh.load_panel(valuation=True)
    pq = eh.conditional_paths(eh.quarterly_model_fn(Q, n_boot=0), Ph, protocol="split", horizons_q=(1, 4),
                              band=False)
    assert pq[pq.asset == "us_equity"].yhat_mult.notna().all()
    A = eh.variance_anatomy(Ph, pq, assets=["us_equity"], measures=["cape", "pd"], horizons_q=(1, 4))
    H = eh.anatomy_headline(A, "L0q")
    assert list(H.columns) == eh.HEADLINE_COLUMNS and "err_r2_m" not in H.columns
    cells = H.groupby(["measure", "group", "h_q"])["row"].apply(set)
    assert all("placebo" in s for s in cells)
    scored = H[H.group.isin(["1995-2004", "2005-2014", "2015-2026"])]
    assert set(scored.row) == {"L0q", "placebo"}
    plc = H[H.row == "placebo"]
    assert plc.corr_pred_dm.isna().all() and plc.slope_dm_on_pred.isna().all()
    cape = A[(A.measure == "cape") & A.placebo_err_r2_m.notna()]
    assert cape.placebo_err_r2_m.between(0.83, 1.0).all()
    model = A[A.err_r2_m.notna()]
    assert (model.placebo_err_r2_m >= model.err_r2_m).all()
    assert (A[A.measure == "cape"].dropna(subset=["n_model"]).pred_dm_source == "model").all()


@needs_panels
def test_candidates_switch_on_only_when_asked(P, Q):
    base = qa.AssetLayer.fit("2014Q4", panel=P, Q=Q, n_boot=0)
    assert base.candidates == () and base.cand_fits == {}
    c1 = qa.AssetLayer.fit("2014Q4", panel=P, Q=Q, n_boot=0, candidates=("C1",))
    assert set(c1.cand_fits) == {("C1", a) for a in qa.RISK}
    X = qa.oracle_inputs(Q, pd.period_range("2015Q1", "2016Q4", freq="Q"))
    assert not np.allclose(c1.mean(X)[qa.RISK], base.mean(X)[qa.RISK])
    assert "d_sent" in c1.cand_fits[("C1", "financials")].names and "d_sent" not in c1.cand_fits[("C1", "gold")].names
    for cid in ("C4", "NN", "NN-res", "C9"):
        with pytest.raises(ValueError):
            qa.AssetLayer.fit("2014Q4", panel=P, Q=Q, n_boot=0, candidates=(cid,))
    # the state and lead candidates run through the harness adapter (path-rolled states, next-quarter leads)
    Ph = P.copy()
    split = {"2015-2026": ("2014-12", "2015-01", "2016-12")}
    for cid in ("C2", "C3"):
        pc = eh.conditional_paths(eh.quarterly_model_fn(Q, n_boot=0, candidates=(cid,)), Ph, protocol="split",
                                  horizons_q=(1,), band=False, splits=split)
        assert pc.model.iloc[0] == f"L0q+{cid}" and pc.yhat.notna().all() and len(pc) == 8 * 10


@needs_panels
def test_the_quarterly_adapter_reads_no_asset_data_from_the_origin_on(P, Q):
    """Oracle discipline: realized asset returns from the origin on, and the asset-derived states after it, never
    reach the adapter's prediction, band or predicted multiple."""
    fitted = eh.quarterly_model_fn(Q, n_boot=0)(P.loc[:"2014-12"], pd.Timestamp("2014-12"))
    q0 = pd.Period("2016Q1", "Q")
    Qp = Q.copy()
    rng = np.random.default_rng(0)
    realized = [c for c in Q.columns if c.startswith(("y_", "yt_", "inc_"))]
    states = [c for c in Q.columns if c.startswith(("s_pd", "s_cape", "s_vol", "m1_", "m4_", "s_pe", "s_ecyc",
                                                    "s_fcfy", "s_pb_", "s_w_"))]
    Qp.loc[Qp.index >= q0, realized] += rng.normal(0, 0.1, (int((Qp.index >= q0).sum()), len(realized)))
    Qp.loc[Qp.index > q0, states] += 1.0
    other = eh.QuarterlyFitted(fitted.layer, Qp, name="L0q")
    o = q0.start_time
    for m in (3, 12):
        pd.testing.assert_frame_equal(fitted.predict_path(None, o, m), other.predict_path(None, o, m))
        assert np.array_equal(fitted.simulate_paths(None, o, m, 50, 1), other.simulate_paths(None, o, m, 50, 1))
        assert fitted.predict_mult(None, o, m) == other.predict_mult(None, o, m)


@needs_panels
def test_the_layer_reads_only_data_up_to_its_train_end(P, Q):
    te = pd.Period("2004Q4", "Q")
    full = qa.AssetLayer.fit(te, panel=P, Q=Q, n_boot=0)
    cut = qa.AssetLayer.fit(te, panel=P.loc[:"2004-12"], Q=Q.loc[:te], n_boot=0)
    for name in ("alpha", "kappa", "conv", "c_log", "income_k", "div_growth"):
        pd.testing.assert_series_equal(getattr(full, name), getattr(cut, name), check_exact=False, rtol=0, atol=1e-13)
    pd.testing.assert_frame_equal(full.beta, cut.beta, check_exact=False, rtol=0, atol=1e-13)
    assert np.allclose(full.pool.raw, cut.pool.raw, atol=1e-13) and full.pool.labels == cut.pool.labels
    assert full.pool.labels[-1] == "2004Q4"
    for arm in cq.CMA_ARMS:
        pd.testing.assert_series_equal(full.cma(arm), cut.cma(arm), check_exact=False, rtol=0, atol=1e-13)
    assert full.valuation.summary() == cut.valuation.summary()


@needs_panels
def test_mechanical_identities_match_history(P, Q):
    L = qa.AssetLayer.fit("2026Q3", panel=P, Q=Q, n_boot=0)
    qs = pd.period_range("1973Q2", "2026Q3", freq="Q")
    m = L.mean(qa.oracle_inputs(Q, qs))
    y = Q.loc[qs]
    assert (m["crude_oil"] - y["y_crude_oil"]).abs().max() <= 1e-12
    e = (m["treasury_10y"] - y["y_treasury_10y"]).dropna()
    assert e.abs().max() < 0.003 and 1 - (e ** 2).sum() / ((y["y_treasury_10y"] - y["y_treasury_10y"].mean()) ** 2).sum() > 0.999
    assert (m["cash"] - y["y_cash"]).abs().max() < 0.002
    assert (L.pool.raw[:, [qa.I[a] for a in qa.MECH]] == 0).all()        # identities carry no residual


@needs_panels
def test_cma_arms_and_the_rerating_drift(P, Q):
    for te, pct in cq.MULT_DRIFT_CHECK.items():
        assert val.mult_drift_pct(Q, te) == pytest.approx(pct, abs=0.006)
    L = qa.AssetLayer.fit("2026Q3", panel=P, Q=Q, n_boot=0)
    assert (L.cma("historical") == 0).all()
    nr = L.cma("no_rerate")
    assert nr["us_equity"] == pytest.approx(-L.valuation.m_bar, abs=1e-15)           # rho = 1 for us_equity
    d = L.cma_detail
    for a in ("nasdaq", "energy", "financials"):
        assert nr[a] == pytest.approx(-d["beta_to_market"][a] * d["m_bar_q"] - 0.5 * d["relative_drift_q"][a])
    assert nr["gold"] < 0 and (nr[qa.MECH] == 0).all()
    assert cq.CMA_FORWARD == "no_rerate"
    with pytest.raises(ValueError):
        L.cma("valuation_adjusted")


@needs_panels
def test_simulated_paths_obey_the_identities(layer_today, Q):
    L = layer_today
    st = L.start_state(Q, "2026Q4")
    assert st["s_cape"] == pytest.approx(math.log(41.2), abs=0.002)
    X = qa.oracle_inputs(Q, pd.period_range("2024Q4", "2026Q3", freq="Q"))
    a = L.simulate(X, st, n_paths=3000, seed=7, cma="no_rerate")
    b = L.simulate(X, st, n_paths=3000, seed=7, cma="no_rerate")
    assert np.array_equal(a.price, b.price) and np.array_equal(a.drawn, b.drawn)          # determinism by seed
    assert a.price.shape == (3000, 8, 13) and a.drawn.shape == (3000, 8) and (a.param_index >= 0).all()
    assert set(a.drawn.ravel()) <= set(L.pool.labels)
    assert np.allclose(a.total, np.where(np.arange(13) == qa.I["cash"], a.price, np.log(np.exp(a.price) + a.income)))
    assert np.allclose(a.y_mult, a.price[:, :, qa.I["us_equity"]] - a.y_fund)               # V-PRICE
    assert np.allclose(a.y_pd, a.price[:, :, qa.I["us_equity"]] - a.y_dg)
    assert np.allclose(a.log_cape[:, 1:] - a.log_cape[:, :-1], a.y_mult)
    assert (a.income[:, :, [qa.I[x] for x in qa.DIVIDEND_ASSETS]] > 0).all()
    assert (a.income[:, :, qa.I["gold"]] == 0).all() and (a.income[:, :, qa.I["crude_oil"]] == 0).all()
    assert np.allclose(a.price[:, :, qa.I["crude_oil"]], X["d_oil"].to_numpy() / 100.0)
    assert np.abs(a.resid[:, :, [qa.I[x] for x in qa.RISK]].mean(axis=(0, 1))).max() < 0.006
    one = L.simulate(X.iloc[:1], {**st, "s_vol": np.full(3000, st["s_vol"])}, n_paths=3000, seed=7)
    assert one.price.shape == (3000, 1, 13)                               # step mode with per-path states
    nxt = L.next_state(st, one)                                           # chaining one-quarter steps
    assert np.allclose(nxt["s_pd_us_equity"], one.log_pd[:, -1]) and np.allclose(nxt["s_cape"], one.log_cape[:, -1])
    assert np.allclose(nxt["y_fund_lags"][0], st["y_fund_lags"][1]) and np.allclose(nxt["y_fund_lags"][1], one.y_fund[:, 0])
    two = L.simulate(X.iloc[1:2], nxt, n_paths=3000, seed=8)
    assert np.allclose(two.income[:, 0, qa.I["us_equity"]], L.income_k["us_equity"] * np.exp(-nxt["s_pd_us_equity"]) / 4)
    full = L.next_state(st, a)
    assert np.allclose(full["s_pd_us_equity"], a.log_pd[:, -1])
    assert np.allclose(full["s_vol"], np.sqrt((a.price[:, -1, 0] ** 2 + a.price[:, -2, 0] ** 2) / 2))
    stm = val.parse_statements([{"quarters": [1, 2, 3, 4], "mode": "rerate", "direction": "-", "size": "明显",
                                 "quote": ""}], 8)
    c = L.simulate(X, st, n_paths=4000, seed=9, cma="no_rerate", statements=stm)
    s = c.y_mult[:, :4].sum(axis=1)
    target = L.valuation.bins(4)["明显"]
    assert s.mean() == pytest.approx(-target, abs=0.005)
    assert np.mean(np.abs(s + target) <= 0.25 * target) == pytest.approx(0.68, abs=0.05)
    assert c.shift[:, :, qa.I["nasdaq"]].sum(1).mean() < 0 and np.allclose(c.shift[:, 4:, :], 0.0)


@needs_panels
def test_attribution_adds_up_to_the_systematic_part(P, Q):
    L = qa.AssetLayer.fit("2014Q4", panel=P, Q=Q, n_boot=0)
    X = qa.oracle_inputs(Q, pd.period_range("2015Q1", "2019Q4", freq="Q"))
    stated = {"d_ff", "d_unrate", "d_cpi_goods", "d_cpi_serv", "d_ip", "d_oil"}
    for cma in cq.CMA_ARMS:
        parts = L.contributions(X, stated=stated, cma=cma)
        total = parts["stated"] + parts["completed"] + parts["drift"] + parts["candidates"]
        assert (total - L.mean(X, cma=cma)).abs().max().max() <= 1e-12
        assert (sum(parts["fields"].values()) + parts["drift"] - L.mean(X, cma=cma)).abs().max().max() <= 1e-12
    parts = L.contributions(X, stated=stated)
    assert (parts["stated"]["us_equity"] - sum(parts["fields"][f]["us_equity"] for f in stated)).abs().max() < 1e-12
    bf = L.beta_fields()
    assert bf.loc["us_equity", "d_gz"] == pytest.approx(L.beta.loc["us_equity", "gz"])
    assert bf.loc["us_equity", "d_oil"] == pytest.approx(L.beta.loc["us_equity", "oil"] / 100.0)


@needs_shiller
def test_valuation_block_check_values(Q):
    for span, ref in cq.SIZE_BIN_CHECK.items():
        b = val.size_bins(Q, cq.SIZE_BIN_CHECK_END, span)
        assert [b[s] for s in val.SIZES] == pytest.approx(list(ref), abs=0.006)
    assert val.leg_frame(Q)["gap"].abs().max() <= 1e-10
    res = val.research_equations(Q, "2026Q3")
    assert res["y_fund"]["coef"]["dcpi"] == pytest.approx(0.97, abs=0.006)
    assert res["y_fund"]["t"]["dcpi"] == pytest.approx(8.9, abs=0.05)
    assert res["y_fund"]["r2"] == pytest.approx(0.56, abs=0.006)
    assert res["y_dg"]["coef"]["dcpi"] == pytest.approx(0.44, abs=0.006)
    assert res["y_dg"]["coef"]["g_lag"] == pytest.approx(0.30, abs=0.006)
    assert res["y_dg"]["r2"] == pytest.approx(0.13, abs=0.006)
    K = val.shiller_known()
    assert (K["log_cape"].reindex(Q.index) - Q["s_cape"]).abs().max() <= 1e-12
    rows = {r["id"]: r for r in val.state_table(Q)}
    assert rows["cape"]["value"] == pytest.approx(41.2, abs=0.05)
    assert rows["cape"]["percentile"] == pytest.approx(97.8, abs=0.1) and rows["cape"]["history_from"] == "1948Q1"
    assert rows["ecy"]["value"] == pytest.approx(0.79, abs=0.01)
    assert rows["pd"]["value"] == pytest.approx(97.4, abs=0.1)
    assert rows["fcfy"]["value"] == pytest.approx(1.96, abs=0.01) and "SPY/QQQ" in rows["fcfy"]["caveat"]
    assert rows["pb_buseq"]["value"] == pytest.approx(12.2, abs=0.1)
    assert rows["pb_buseq"]["reference"]["2000Q1 季末"] == pytest.approx(15.1, abs=0.1)
    assert "估计值" in rows["cape"]["caveat"]


@needs_panels
def test_parameter_refits(layer_today):
    L = layer_today
    assert len(L.params) == 50 and L.params.block_months == cq.PARAM_BOOT_BLOCK_MONTHS
    cols = [L.f_assets.index(a) for a in qa.RISK]
    j = L.features.index("metals_oil")
    assert np.all(L.params.beta[:, [L.f_assets.index(a) for a in qa.RISK if a != "energy"], j] == 0.0)
    b, sd = L.params.beta[:, cols, :].mean(0), L.params.beta[:, cols, :].std(0)
    point = L.beta.loc[qa.RISK, L.features].to_numpy()
    ok = sd > 0
    assert np.all(np.abs(b - point)[ok] < 1.5 * sd[ok])
    assert inspect.signature(qa.AssetLayer.fit).parameters["n_boot"].default == cq.PARAM_BOOT == 200
