"""The recipes against history and the stored build (INSTRUMENTS_DESIGN §8 T1, T3, T4, T11-T13, T16, T21): these
need data/raw/instruments (python -m scenariolab.instruments.import_raw) and the panel, and skip without them (a
fresh clone has no data/). No network, no LLM. The tolerances are the design's; the numbers are its tables'."""
import time

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.instruments import ESTIMATED, TICKERS
from scenariolab.instruments import data as D
from scenariolab.instruments import params as prm

HAVE_DATA = (config.RAW_DIR / "instruments" / "SOURCES.json").exists() and config.PANEL_PATH.exists()
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="instrument data not imported (data/raw/instruments)")
HAVE_PATHS = (config.RESULTS_DIR / "scenario_paths.npz").exists()


@pytest.fixture(scope="module")
def V():
    from scenariolab.instruments import validate
    return validate


# ---------------------------------------------------------------- structural replays (T3)

def test_leveraged_replay_from_monthly_inputs(V):
    q = V.letf_replay("QLD", "2006-07")
    t = V.letf_replay("TQQQ", "2010-03")
    assert (q["first"], q["last"], q["months"]) == ("2006-07", "2026-09", 243)
    assert abs(q["ann_model"] - q["ann_actual"]) <= 0.005 and q["rmse_monthly"] <= 0.006 and q["t3"]["pass"]
    assert abs(t["ann_model"] - t["ann_actual"]) <= 0.010 and t["rmse_monthly"] <= 0.015 and t["t3"]["pass"]
    # the funds' actual returns are the design's: QLD 25.35, TQQQ 42.47 %/yr (the model: fourth order, ACT/360)
    assert q["ann_actual"] == pytest.approx(0.2535, abs=1e-4) and q["ann_model"] == pytest.approx(0.2527, abs=5e-4)
    assert t["ann_actual"] == pytest.approx(0.4247, abs=1e-4) and t["ann_model"] == pytest.approx(0.4168, abs=1e-3)


def test_leveraged_replay_out_of_sample_window_is_reported_and_fails_t3(V):
    """Review finding 3: 2006..2018 is in sample (the RV model fitted to 2014, the pre-2019 spread fitted on these
    NAV returns), so only 2019-01..2026-09 is out of sample, and there the monthly-input replay misses T3 (QLD
    -1.6, TQQQ -4.6 %/yr: the full window nets out errors of opposite sign). The same formula on the months'
    actual RV / RS3 / RS4 meets T3 in both windows: the gap is the RV state model's level, which over-predicts the
    variance before 2019 and under-predicts it after."""
    for fund, start, oos_actual, oos_model in (("QLD", "2006-07", 0.3710, 0.3865), ("TQQQ", "2010-03", 0.4490, 0.4954)):
        r = V.letf_replay(fund, start)
        w = r["windows"]
        assert (w["in_sample"]["first"], w["in_sample"]["last"]) == (start, "2018-12")
        assert (w["oos"]["first"], w["oos"]["last"], w["oos"]["months"]) == ("2019-01", "2026-09", 93)
        assert w["oos"]["ann_actual"] == pytest.approx(oos_actual, abs=5e-4)
        assert w["oos"]["ann_model"] == pytest.approx(oos_model, abs=1e-3)
        assert not w["oos"]["t3"]["pass"] and not w["in_sample"]["t3"]["pass"]
        assert w["in_sample"]["ann_model"] < w["in_sample"]["ann_actual"] and w["oos"]["ann_model"] > w["oos"]["ann_actual"]
        assert w["in_sample"]["rv_ann_model"] > w["in_sample"]["rv_ann_actual"]
        assert w["oos"]["rv_ann_model"] < w["oos"]["rv_ann_actual"]
        a = V.letf_replay(fund, start, rv="actual")
        for k in ("in_sample", "oos"):
            assert a["windows"][k]["t3"]["pass"], (fund, k, a["windows"][k])
    s = V.summary()
    assert s["TQQQ"]["windows"]["oos"]["t3"]["pass"] is False and s["TQQQ"]["actual_rv"]["windows"]["oos"]["t3"]["pass"]
    assert s["QQQI"]["in_sample"] and "N-PORT" in s["QQQI"]["note"]


def test_catalog_validation_comes_from_the_build():
    """The recipe card shows the build's own replay (instrument_meta.json validation), labelled in / out of sample."""
    from scenariolab.instruments import catalog
    val = {"TQQQ": {"ann_actual": 0.4, "ann_model": 0.41, "rmse_monthly": 0.01, "t3": {"pass": True},
                    "windows": {"oos": {"ann_actual": 0.449, "ann_model": 0.495, "rmse_monthly": 0.0186,
                                        "t3": {"pass": False}, "months": 93, "first": "2019-01", "last": "2026-09"}}}}
    e = catalog.entries({"validation": val, "start": {}, "flags": {}, "stats": {}})["tickers"]["TQQQ"]["validation"]
    oos = next(v for v in e if v.get("src") == "TQQQ.windows.oos")
    assert oos["model"] == 0.495 and oos["t3_pass"] is False and oos["sample"] == "oos" and oos["from_build"]
    assert {v["sample"] for v in e} == {"full", "in", "oos"}


def test_covered_calls_marked_on_vxn(V):
    q = V.covered_call("QQQI")
    assert (q["first"], q["last"], q["months"]) == ("2024-02", "2026-06", 29)
    assert abs(q["ann_model"] - q["ann_actual"]) <= 0.005
    assert q["ann_actual"] == pytest.approx(0.2257, abs=1e-4) and q["ann_model"] == pytest.approx(0.2228, abs=5e-4)
    assert q["beta_model_on_ndx"] == pytest.approx(0.688, abs=0.005)
    y = V.covered_call("QYLD")
    assert abs(y["ann_model"] - y["ann_actual"]) <= 0.006 and y["months"] == 84


def test_tlt_and_qqq(V):
    t = V.tlt_replay()
    assert t["struct"]["rmse_monthly"] <= 0.006 and t["struct"]["rmse_monthly"] == pytest.approx(0.0048, abs=2e-4)
    assert t["sim"]["rmse_monthly"] == pytest.approx(0.0124, abs=2e-4)
    q = V.qqq_tracking()
    assert q["te_ann"] <= 0.002 and q["months"] == 84


def test_qld_year_end_distributions_are_imputed(V):
    """T13 / defect 3: with the imputed 2006-12 and 2007-12 distributions (6.07% / 5.13% of NAV) QLD's month tracks
    the daily-reset formula to < 0.5% (about -6% / -5% without them)."""
    r = V.qld_year_end()
    for m, share in (("2006-12", 0.0607), ("2007-12", 0.0513)):
        assert abs(r[m]["residual"]) < 0.005, m
        assert r[m]["imputed_share_of_nav"] == pytest.approx(share, abs=5e-4)


# ---------------------------------------------------------------- estimated recipes

# the design's out-of-sample R2 (expanding h = 1/4/8; CSWC: the 2019 split), INSTRUMENTS_DESIGN §3.0 / MAPPED §3
DESIGN_OOS = {"XLG": (0.95, 0.94, 0.92), "SCHD": (0.87, 0.90, 0.80), "MO": (0.43, 0.51, 0.44),
              "BRK.B": (0.34, 0.52, 0.42), "MAIN": (0.70, 0.61, 0.63), "ARCC": (0.81, 0.68, 0.69),
              "CSWC": (0.64, 0.56, 0.71), "EPD": (0.83, 0.78, 0.76), "MPLX": (0.80, 0.75, 0.73),
              "BME": (0.86, 0.87, 0.88)}


def test_estimated_out_of_sample_r2(V):
    df = V.estimated_oos()
    better = 0
    for t, ref in DESIGN_OOS.items():
        proto = "split" if t == "CSWC" else "expanding"
        g = df[(df.ticker == t) & (df.protocol == proto)].set_index("h")
        for h, r in zip((1, 4, 8), ref):
            assert g.loc[h, "r2"] == pytest.approx(r, abs=0.01), (t, h)
        better += int((g["r2"] > g["r2_market"] + 0.02).sum() >= 2)
    assert better >= 9                                      # 9 of 10 clearly beat the market-only benchmark


def test_estimated_replay_is_exact(V):
    """T4: the actual core, the actual residual dates and the raw residuals reproduce every quarter."""
    err = V.estimated_replay()
    assert set(err) == set(ESTIMATED)
    assert max(err.values()) < 1e-6


def test_refits_reproduce_the_frozen_parameters():
    """T16: every fitted block refit on data cut at its train_end equals params/."""
    from scenariolab.instruments import fit
    res = fit.check()
    assert all(res.values()), [k for k, v in res.items() if not v]


def test_13f_prices_rebuild_from_the_extracts():
    p = D.prices_13f()
    ref = D.derived("p13f_adj.csv")
    ref.index = pd.to_datetime(ref.pop("qend"))
    for t in ("XLG", "SCHD", "MO", "BRK.B", "MAIN", "ARCC", "CSWC", "EPD", "MPLX", "BME"):
        k = p[t].dropna().index.intersection(ref[t].dropna().index)
        assert len(k) >= 40 and np.allclose(p.loc[k, t], ref.loc[k, t], rtol=1e-5), t


# ---------------------------------------------------------------- data rules and the start state

def test_complete_month_rule_on_the_vintage_mismatch():
    """T12: ProShares has 2026-10-30, FRED's NASDAQ-100 stops at 10-29: October is incomplete together."""
    nav, ndx = D.proshares_nav("QLD"), D.ndx_daily()
    oct_ = pd.Period("2026-10", "M")
    assert oct_ in D.complete_months(nav) and oct_ not in D.complete_months(nav, ndx)
    assert oct_ not in D.letf_monthly("QLD").index
    r1029 = nav.loc["2026-10-29"] / nav.loc["2026-09-30"] - 1
    r1030 = nav.loc["2026-10-30"] / nav.loc["2026-09-30"] - 1
    assert r1029 == pytest.approx(-0.0722, abs=1e-4) and r1030 == pytest.approx(-0.1367, abs=1e-4)
    assert pd.Period("2026-10", "M") not in D.ndx_rv_table().index


def test_start_state_from_the_data_matches_the_packaged_block():
    from scenariolab.instruments import anchors
    a = anchors.compute("2026-09")
    pk = prm.packaged_anchors()
    for t in TICKERS:
        assert a["prices"][t]["price"] == pytest.approx(pk["prices"][t]["price"], rel=1e-9), t
    for k in ("lrv_ndx", "lrv_ndx_prev", "lrv_mkt", "dd_mkt", "vxn"):
        assert a["vol"][k] == pytest.approx(pk["vol"][k], abs=1e-12), k
    # the design's table 1.1: 13F 2026-06-30 rolled one quarter; ARCC's XBRL price; MO's declared dividend
    for t, px in {"XLG": 61.93, "SCHD": 30.69, "MO": 70.79, "BRK.B": 480.66, "MAIN": 50.85, "ARCC": 19.10,
                  "CSWC": 23.19, "EPD": 38.96, "MPLX": 60.48, "BME": 44.02}.items():
        assert a["prices"][t]["price"] == pytest.approx(px, abs=0.006), t
    assert a["vol"]["lrv_ndx"] == pytest.approx(-6.197, abs=1e-3) and a["vol"]["lrv_mkt"] == pytest.approx(-6.955, abs=1e-3)
    assert a["vol"]["vxn"] == pytest.approx(0.2246) and a["prices"]["QLD"]["price"] == pytest.approx(95.6425)


# ---------------------------------------------------------------- the engine: T1, T11, T21, continuation

@pytest.fixture(scope="module")
def ctx():
    from scenariolab.context import EngineContext
    return EngineContext.load(selection=True)


@pytest.mark.skipif(not HAVE_PATHS, reason="scenario paths not built")
def test_resimulated_macro_paths_leave_the_stored_paths_unchanged(ctx):
    """T1: the build's own seeds give the stored scenario paths bit for bit (the macro paths come along)."""
    from scenariolab.instruments import build
    z = np.load(config.RESULTS_DIR / "scenario_paths.npz", allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    k = ids.index("baseline")
    sub = {"total": z["total"][k:k + 1], "income": z["income"][k:k + 1]}
    out = build._resimulate(ctx, ["baseline"], z["total"].shape[1], sub)
    assert out["max_abs_diff"]["baseline"] == 0.0
    assert out["y10"].shape == (1, z["total"].shape[1], z["total"].shape[2] + 1)


def _unconditional(ctx, n: int, years: int, seed: int):
    from scenariolab.engine.pooling import BLOCK, pool_phase
    from scenariolab.engine.simulate import capital_market_assumptions, simulate
    from scenariolab.portfolio.service import CONTINUATION_CMA
    db, an = ctx.db, ctx.anchor
    T = years * 12
    pool = pool_phase(db.frame, [], db.prior(), T, BLOCK, rare=db.rare)
    cma = capital_market_assumptions(db, an, method=CONTINUATION_CMA)
    return simulate(db, an, [pool.blocks], [T], n_paths=n, seed=seed, block=BLOCK, cma=cma, **config.RESIDUALS)


def test_vol_state_on_view_free_engine_paths(ctx):
    """T11: 1000 view-free paths x 40 years: the simulated RV medians within +-15% of the 1986-2026 history."""
    from scenariolab.instruments.core import CoreView
    from scenariolab.instruments.vol import vol_state
    sim = _unconditional(ctx, 1000, 40, 7)
    v = vol_state(CoreView.from_sim(sim, ctx.db, scenario_id="t11", set_id="test"))
    h_ndx = D.ndx_rv_table()["rv"].loc[pd.Period("1986-01", "M"):].median()
    h_mkt = D.mkt_rv("M")["rv"].loc[pd.Period("1986-01", "M"):].median()
    assert np.median(v.rv_ndx) == pytest.approx(h_ndx, rel=0.15)
    assert np.median(v.rv_mkt) == pytest.approx(h_mkt, rel=0.15)
    assert np.corrcoef(np.log(v.rv_ndx[:, -1]), np.log(v.rv_mkt[:, -1]))[0, 1] > 0.5


def test_layer_speed_and_the_continuation_hook(ctx):
    """T21 (one scenario's 600 x 180 well under the 30 s budget for 7,800) and continuation_factors: years
    16..40 for every ticker plus the alpha variants, each row keeping its scenario alpha (T19)."""
    from scenariolab.instruments import layer
    from scenariolab.instruments.core import CoreView
    sim = _unconditional(ctx, 600, 15, 3)
    core = CoreView.from_sim(sim, ctx.db, scenario_id="baseline")
    t0 = time.time()
    out = layer.run_layer(core)
    assert time.time() - t0 < 3.0
    cont = _unconditional(ctx, 40, 40, 5)
    g, i = layer.rows_for_chunk(["baseline", "ai_bust"], 30, 0, 40)
    res = layer.continuation_factors(cont, ctx.db, 180, g, i, chunk=0)
    assert res["total"].shape == (40, 25, len(TICKERS)) and np.isfinite(res["total"]).all()
    assert set(res["alpha"]) == {"half", "raw"} and res["alpha"]["half"][0].shape == (40, 25, len(ESTIMATED))
    assert (res["income"] >= 0).all()
    a_scen = out.paths["MAIN"].parts["alpha"][:30, 0]
    c_core = CoreView.from_sim(cont, ctx.db, scenario_id="chunk0", set_id="continuation")
    from dataclasses import replace
    c_core = replace(c_core, row_group=g, row_index=i, row_set_id="builtin")
    a_cont = layer.run_layer(c_core, ["MAIN"]).paths["MAIN"].parts["alpha"][:30, 0]
    assert np.allclose(a_scen, a_cont)


@pytest.mark.skipif(not (config.RESULTS_DIR / "scenario_macro.npz").exists(), reason="instruments not built")
def test_engine_paths_seen_quarterly_keep_qqqi_and_the_leveraged_funds_within_t3():
    """Review finding 4 on the engine's baseline paths: the monthly core and its quarterly aggregate (each with
    its own volatility model) give QQQI and QLD / TQQQ (ruin off) the same mean 15-year log return within T3."""
    from scenariolab.instruments import anchors, build, layer
    from scenariolab.instruments.core import CoreView
    z = build.load_scenarios()
    ids, labels = [str(x) for x in z["ids"]], [str(x) for x in z["labels"]]
    s = ids.index("baseline")
    cm = CoreView.from_arrays("M", z["total"][s], z["income"][s], {k: z[k][s] for k in ("cpi", "y10", "ff", "spread")},
                              labels, [str(a) for a in z["assets"]], scenario_id="baseline")
    anchor = anchors.current(labels[0][:7], prefer_data=True)
    p = {t: {"ruin": False} for t in ("QLD", "TQQQ")}
    om = layer.run_layer(cm, ["QQQI", "QLD", "TQQQ"], params=p, anchor=anchor)
    oq = layer.run_layer(cm.quarterly(), ["QQQI", "QLD", "TQQQ"], params=p, anchor=anchor)
    for t, tol in (("QQQI", 0.005), ("QLD", 0.005), ("TQQQ", 0.010)):
        a = np.log1p(om.paths[t].total).sum(axis=1).mean() / 15
        b = np.log1p(oq.paths[t].total).sum(axis=1).mean() / 15
        assert abs(a - b) < tol, (t, a, b)


@pytest.mark.skipif(not (config.RESULTS_DIR / "instrument_paths.npz").exists(), reason="instruments not built")
def test_the_built_file():
    from scenariolab.instruments import build
    z = np.load(config.RESULTS_DIR / "scenario_paths.npz", allow_pickle=False)
    r = build.read_paths(tickers=["QQQ", "TQQQ", "MAIN"], scenarios=["baseline"])
    S, N, T = 1, z["total"].shape[1], z["total"].shape[2]
    assert r["total"].shape == (S, N, T, 3) and r["level"].shape == (S, N, T + 1, 3)
    k = [str(x) for x in z["ids"]].index("baseline")
    j = [str(a) for a in z["assets"]].index("nasdaq")
    from scenariolab.instruments import fees
    assert np.allclose(r["total"][0, :, :, 0], z["total"][k][:, :, j] - fees.fee("QQQ") / 12, atol=1e-6)
    with np.load(config.RESULTS_DIR / "instrument_paths.npz", allow_pickle=False) as f:
        assert [str(x) for x in f["ids"]] == [str(x) for x in z["ids"]]
        assert f["alpha_half_total"].shape == (len(z["ids"]), N, T // 12, len(ESTIMATED))


@pytest.mark.skipif(not (config.RESULTS_DIR / "instrument_paths.npz").exists()
                    or not (config.RESULTS_DIR / "scenario_macro.npz").exists(), reason="instruments not built")
def test_project_recomputes_the_build_and_changes_only_what_is_asked():
    """/api/instruments/project: the default parameters give the stored numbers; another financing spread changes
    TQQQ and nothing else (T19), within the API's time budget (T21)."""
    from scenariolab.instruments import build
    data = build.load_scenarios()
    t0 = time.time()
    base = build.project(["TQQQ", "QQQI", "MAIN"], data=data, scenario_ids=["baseline", "ai_bust"])
    assert time.time() - t0 < 3.0
    stored = build.read_paths(tickers=["TQQQ", "QQQI", "MAIN"], scenarios=["baseline", "ai_bust"])
    for s, sid in enumerate(stored["ids"]):
        for k, t in enumerate(stored["tickers"]):
            assert np.allclose(base[sid].paths[t].total, stored["total"][s, :, :, k], atol=1e-6), (sid, t)
    alt = build.project(["TQQQ", "QQQI"], params={"TQQQ": {"s_eff": "latest"}}, data=data, scenario_ids=["baseline"])
    assert np.array_equal(alt["baseline"].paths["QQQI"].total, base["baseline"].paths["QQQI"].total)
    assert (alt["baseline"].paths["TQQQ"].total <= base["baseline"].paths["TQQQ"].total + 1e-12).all()
