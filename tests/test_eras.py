from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scenariolab import config, jsonio
from scenariolab.analysis import eras
from scenariolab.engine.ep import effective_number


def test_soft_pool_hits_feasible_targets_and_relaxes_to_prior():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 3))
    Z = X.T
    b = np.array([0.4, -0.3, 0.2])
    q = eras.soft_pool(Z, b, 1e-6)
    assert np.allclose(Z @ q, b, atol=1e-3)
    q_loose = eras.soft_pool(Z, b, 1e4)
    assert np.allclose(q_loose, 1 / 800, rtol=1e-2)


def test_match_today_keeps_ens_floor_when_target_is_outside_history():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 2))
    target = np.array([6.0, 0.0])                        # 6 sd beyond anything in the sample
    q, tau = eras.match_today(X, target, ens_min=40)
    assert np.isfinite(q).all() and abs(q.sum() - 1) < 1e-9
    assert effective_number(q) >= 40 * 0.99
    m = q @ X
    assert 0.5 < m[0] < 6.0 and abs(m[1]) < 0.3           # pulled toward the view, gap remains


def _toy_panel(n=240, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    oil = rng.normal(scale=0.08, size=n)
    lag = lambda k: np.r_[np.zeros(k), oil[:-k]]  # noqa: E731
    cpi = 0.002 + 0.02 * oil + 0.01 * lag(1) + 0.005 * lag(3) + rng.normal(scale=0.0005, size=n)
    return pd.DataFrame({"f_oil": oil, "f_cpi": cpi}, index=idx)


def test_oil_cpi_passthrough_recovers_lag_sum():
    P = _toy_panel()
    pt, se = eras.oil_cpi_passthrough(P, "1991-01", None)
    assert abs(pt - 0.035) < 3 * se + 1e-3 and se < 0.003


def test_oil_cpi_passthrough_skips_administered_oil_prices():
    P = _toy_panel()
    step = np.zeros(len(P))
    step[[30, 70, 110, 150, 190]] = 0.08                 # a posted price that moves 5 times
    P["f_oil"] = step
    P["f_cpi"] += 0.5 * pd.Series(step, index=P.index)
    assert eras.oil_price_moves(P, "1991-01", None) == 5
    pt, se = eras.oil_cpi_passthrough(P, "1991-01", None)
    assert np.isnan(pt) and np.isnan(se)


def test_similarity_database_stops_before_ongoing_era():
    idx = pd.date_range("2000-01-01", "2026-09-01", freq="MS")
    P = pd.DataFrame(index=idx)
    ongoing = [{"id": "old", "start": "2000-01", "end": "2023-12"}, {"id": "now", "start": "2024-01", "end": None}]
    end, excluded = eras.similarity_end(P, ongoing)
    assert end == pd.Timestamp("2023-12-01") and excluded == ["now"]
    young = [{"id": "old", "start": "2000-01", "end": "2026-03"}, {"id": "now", "start": "2026-04", "end": None}]
    end, excluded = eras.similarity_end(P, young)
    assert end == pd.Timestamp("2024-09-01") and excluded == ["now"]   # the 24-month gap binds


def test_asset_block_cagr_real_and_drawdown():
    idx = pd.date_range("2000-01-01", periods=25, freq="MS")
    r = np.full(25, 0.01)
    r[12] = -0.5
    P = pd.DataFrame({"lvl_cpi": 100 * 1.002 ** np.arange(25)}, index=idx)
    for a in config.ASSET_IDS:
        P[f"r_{a}"] = r if a == "us_equity" else np.nan
        P[f"d_{a}"] = 0.001
    out = eras.asset_block(P, idx[1], idx[-1])
    assert list(out) == ["us_equity"]
    st = out["us_equity"]
    growth = 1.01 ** 23 * 0.5
    assert abs(st["cagr"] - (growth ** (12 / 24) - 1)) < 1e-12
    assert abs(st["real_cagr"] - ((growth / 1.002 ** 24) ** (12 / 24) - 1)) < 1e-12
    assert abs(st["max_dd"] + 0.5) < 1e-12
    assert abs(st["avg_income"] - 0.012) < 1e-12


@pytest.fixture(scope="module")
def real():
    if not config.PANEL_PATH.exists():
        pytest.skip("panel not built")
    from scenariolab.data.panel import load_panel

    return eras.run(SimpleNamespace(panel=load_panel()))


def test_eras_run_returns_all_eras_with_sane_fields(real):
    recs = {r["id"]: r for r in real["eras"]}
    assert list(recs) == [e["id"] for e in config.ERAS]
    for r in recs.values():
        assert r["n_months"] > 24 and len(r["signature"]) == eras.SIGNATURE_SIZE
        assert r["narrative"].startswith(r["name"])
        assert {"cpi_yoy", "unrate", "spread"} <= set(r["states"])
        assert "us_equity" in r["assets"] and len(r["corr"]["assets"]) == len(r["corr"]["matrix"])
    gi, qe = recs["great_inflation"], recs["stagnation_qe"]
    assert 6.0 < gi["states"]["cpi_yoy"]["mean"] < 10.0
    assert gi["assets"]["us_equity"]["real_cagr"] < 0.02
    assert -0.6 < qe["assets"]["us_equity"]["max_dd"] < -0.35
    assert gi["stock_bond_corr"] > 0.1 and qe["stock_bond_corr"] < -0.1
    assert 0 < gi["oil_cpi_passthrough"] < 0.1
    ga = recs["golden_age"]                                           # posted WTI price: 6 moves
    assert ga["oil_price_moves"] < eras.MIN_OIL_MOVES and np.isnan(ga["oil_cpi_passthrough"])
    assert "管制挂牌价" in ga["narrative"]
    assert "nasdaq" not in recs["golden_age"]["assets"]               # starts 1971
    assert recs["golden_age"]["phillips_measure"] == "core_yoy"
    assert recs["ai_capex"]["ongoing"] and recs["ai_capex"]["end"] == real["reference"]["end"]


def test_fingerprint_timeline_and_today_similarity(real):
    fp = real["fingerprint"]
    assert np.asarray(fp["z"], dtype=float).shape == (len(config.ERAS), len(eras.FINGERPRINT_STATES))
    gi = fp["rows"].index("great_inflation")
    assert fp["z"][gi][fp["cols"].index("cpi_yoy")] > 1.0
    assert fp["values"][fp["rows"].index("golden_age")][fp["cols"].index("real_wage_yoy")] is None
    ts = real["today_similarity"]
    assert abs(sum(ts["weights"].values()) - 1) < 1e-6
    assert ts["ens"] >= eras.ENS_MIN * 0.99
    assert ts["database"][1] < ts["window"][0]
    ongoing = [e for e in config.ERAS if e["end"] is None]
    assert ts["excluded_eras"] == [e["id"] for e in ongoing]
    for e in ongoing:                                                 # today is inside it: no self-matching
        assert e["id"] not in ts["weights"] and ts["database"][1] < e["start"]
    assert len(ts["views"]) == len(eras.SIMILARITY_STATES) and ts["residual"]
    assert ts["interpretation"]
    tl = real["timeline"]
    assert tl["dates"][0] == "1955-01" and len(tl["series"]["cpi_yoy"]) == len(tl["dates"])
    jsonio.dumps(real)


def test_era_attribution_splits_the_mean_price_return_exactly():
    from scenariolab.engine.factor_model import AssetFactorModel, ModelSpec, design_matrix
    rng = np.random.default_rng(5)
    n = 240
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    P = pd.DataFrame({"f_oil": rng.normal(0.004, 0.08, n), "f_ip": rng.normal(0.002, 0.01, n)}, index=idx)
    P["p_us_equity"] = 0.006 + 0.3 * P["f_oil"] - 1.2 * P["f_ip"] + rng.normal(0, 0.03, n)
    P["d_us_equity"] = 0.0015
    P["r_us_equity"] = P["p_us_equity"] + P["d_us_equity"]
    P["p_cash"], P["d_cash"], P["r_cash"] = 0.0, 0.003, 0.003
    spec = ModelSpec(factors=("oil", "ip"), name="toy")
    model = AssetFactorModel(spec, assets=["us_equity"]).fit(P, "1990-01", None)
    out = eras.era_attribution(P, model, design_matrix(P, spec), model.predict(P), idx[60], idx[179])
    W = P.loc[idx[60]:idx[179]]
    r = out["assets"]["us_equity"]
    assert out["n_months"] == 120 and out["coverage"] == 1.0 and out["method"] == "linear"
    assert r["terms"]["oil"] == pytest.approx(12 * model.coef.loc["us_equity", "oil"] * W["f_oil"].mean())
    assert r["terms"]["intercept"] == pytest.approx(12 * model.intercept["us_equity"])
    resid = W["p_us_equity"] - model.predict(P).loc[W.index, "us_equity"]
    assert r["terms"]["residual"] == pytest.approx(12 * resid.mean())
    assert sum(r["terms"].values()) == pytest.approx(12 * W["r_us_equity"].mean(), abs=1e-12)
    assert r["price_mean"] == pytest.approx(r["total_mean"] - r["terms"]["income"], abs=1e-12)
    assert r["factor_sum"] == pytest.approx(r["terms"]["oil"] + r["terms"]["ip"], abs=1e-15)
    assert r["cagr"] == pytest.approx(r["total_mean"] + r["reconciliation"], abs=1e-12)
    ob = r["own_beta"]                                     # the era's own OLS: its residual has mean 0
    assert ob["factor_sum"] + ob["intercept"] == pytest.approx(r["price_mean"], abs=1e-10) and 0 < ob["r2"] < 1
    assert out["assets"]["cash"]["terms"] == pytest.approx({"income": 0.036})
    # most months without a factor: no attribution, and the reason names the factor
    Q = P.copy()
    Q.loc[idx[:150], "f_ip"] = np.nan
    none = eras.era_attribution(Q, model, design_matrix(Q, spec), model.predict(Q), idx[0], idx[199])
    assert none["assets"] == {} and none["missing"] == ["ip"] and "工业产出" in none["reason"]


def test_era_attribution_on_the_real_panel(real):
    info = real["attribution_info"]
    assert info["method"] == "linear" and info["factor_terms"] and info["model"]["train_start"] >= config.MODEL_START
    recs = {r["id"]: r for r in real["eras"]}
    ga = recs["golden_age"]["attribution"]                 # no trade-weighted dollar before 1973
    assert ga["assets"] == {} and "usd" in ga["missing"] and "1973" in ga["reason"]
    for rec in recs.values():
        for x in rec["attribution"]["assets"].values():
            assert sum(x["terms"].values()) == pytest.approx(x["total_mean"], abs=1e-9)
            assert x["total_mean"] - x["terms"]["income"] == pytest.approx(x["price_mean"], abs=1e-9)
            assert x["cagr"] == pytest.approx(x["total_mean"] + x["reconciliation"], abs=1e-12)
            if x["kind"] == "model":
                assert x["factor_sum"] == pytest.approx(sum(x["terms"][f] for f in info["factor_terms"]), abs=1e-12)
    gi = recs["great_inflation"]["attribution"]
    assert gi["coverage"] > 0.99 and gi["months"][0] == "1973-02"
    assert gi["assets"]["us_equity"]["own_beta"] is not None
    assert recs["ai_capex"]["attribution"]["assets"]["us_equity"]["own_beta"] is None          # 33 months: too short
    qe = recs["stagnation_qe"]                              # a complete era reconciles to the asset table's CAGR
    assert qe["attribution"]["assets"]["us_equity"]["cagr"] == pytest.approx(qe["assets"]["us_equity"]["cagr"], abs=1e-12)
    en = gi["assets"]["energy"]["terms"]                    # the oil-shock decade: oil is energy's largest factor term
    assert max(info["factor_terms"], key=lambda f: abs(en[f])) == "oil" and en["oil"] > 0
    jsonio.dumps(real["attribution_info"])


def test_era_attribution_covers_its_own_months(real):
    """大通胀 starts in 1973-01, which has no trade-weighted dollar: its attribution and CAGR cover 1973-02 on, so
    they differ from the asset table's 120-month CAGR (the eras page says so); complete eras match the table."""
    from scenariolab.data.panel import load_panel
    panel = load_panel()
    recs = {r["id"]: r for r in real["eras"]}
    gi = recs["great_inflation"]
    assert gi["attribution"]["n_months"] == gi["n_months"] - 1 and gi["attribution"]["months"] == ["1973-02", gi["end"]]
    r = panel.loc["1973-02":gi["end"], "r_us_equity"].to_numpy(float)
    us = gi["attribution"]["assets"]["us_equity"]
    assert us["n_months"] == len(r) == 119
    assert us["total_mean"] == pytest.approx(12 * r.mean(), abs=1e-12)
    assert us["cagr"] == pytest.approx(np.prod(1 + r) ** (12 / len(r)) - 1, abs=1e-12)
    assert abs(us["cagr"] - gi["assets"]["us_equity"]["cagr"]) > 1e-3             # the table counts 1973-01 too
    for rec in recs.values():
        for a, x in rec["attribution"]["assets"].items():
            if x["n_months"] == rec["n_months"] and a in rec["assets"]:
                assert x["cagr"] == pytest.approx(rec["assets"][a]["cagr"], abs=1e-12), (rec["id"], a)
