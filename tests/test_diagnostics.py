import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.analysis import diagnostics as dg
from scenariolab.analysis import walkforward as wf
from scenariolab.engine.factor_model import MODEL_ASSETS, ModelSpec

TRAIN, VALIDATE, TEST = ("1980-01", "1999-12"), ("2000-01", "2009-12"), ("2010-01", "2019-12")


def synthetic_panel(beta_sent=0.01, flip_in_test=False, n=480, seed=0) -> pd.DataFrame:
    """Core factors plus candidates; asset price returns load on oil, y10 and (truly) sentiment."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1980-01-01", periods=n, freq="MS")
    P = pd.DataFrame(index=idx)
    for f in config.FACTOR_IDS:
        P[f"f_{f}"] = rng.normal(scale=0.01, size=n)
    P["f_sentiment"] = rng.normal(size=n)
    P["s_cpi_yoy"] = 3.0 + rng.normal(size=n)
    beta = np.full(n, beta_sent)
    if flip_in_test:
        beta[idx >= pd.Timestamp(TEST[0])] = -beta_sent
    for a in MODEL_ASSETS:
        P[f"p_{a}"] = (0.005 + 0.8 * P["f_oil"] - 1.5 * P["f_y10"] + beta * P["f_sentiment"]
                       + rng.normal(scale=0.01, size=n))
    P["f_valuation"] = P["p_us_equity"] + rng.normal(scale=0.001, size=n)    # built from the price itself
    return P


def test_ljung_box_white_noise_vs_ar1():
    rng = np.random.default_rng(1)
    e = rng.normal(size=600)
    ar = np.zeros(600)
    for t in range(1, 600):
        ar[t] = 0.5 * ar[t - 1] + e[t]
    assert dg.ljung_box(e)[1] > 0.01
    assert dg.ljung_box(ar)[1] < 1e-6
    assert dg.acf(ar, 1)[0] == pytest.approx(0.5, abs=0.1)


def test_chow_nested_f_and_reset_on_known_cases():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 1))
    noise = rng.normal(scale=0.5, size=(400, 1))
    y_break = np.where(np.arange(400)[:, None] < 200, 1 + 2 * x, 1 - 2 * x) + noise
    y_same = 1 + 2 * x + noise
    assert dg.chow(y_break[:200], x[:200], y_break[200:], x[200:])[1][0] < 1e-6
    assert dg.chow(y_same[:200], x[:200], y_same[200:], x[200:])[1][0] > 0.01
    z = rng.normal(size=(400, 1))
    y = 1 + 2 * x + 0.8 * z + noise
    assert dg.nested_f(y, x, np.column_stack([x, z]))[1][0] < 1e-6
    assert dg.nested_f(y_same, x, np.column_stack([x, z]))[1][0] > 0.01
    assert dg.reset_test(1 + x ** 2 + noise, x)[1][0] < 1e-6
    assert dg.reset_test(y_same, x)[1][0] > 0.01


def test_candidate_changes_and_noops():
    c = {k.id: k for k in wf.build_candidates(synthetic_panel(), TRAIN)}
    base = ModelSpec()
    assert c["add_sentiment"].apply(base).factors[-1] == "sentiment"
    assert c["lags3"].apply(base).lags == 3
    assert c["mlp32"].apply(base).kind == "mlp" and c["mlp32"].apply(base).hidden == 32
    assert c["ridge_0.05"].is_noop(ModelSpec(kind="mlp"))        # ridge only enters the linear fit
    assert c["lags1"].is_noop(ModelSpec(lags=1)) and not c["lags3"].is_noop(ModelSpec(lags=1))
    assert not c["add_valuation"].eligible and c["add_sentiment"].eligible
    assert "shocks_add" not in c                                  # no e_* columns in this panel


def test_evaluate_fits_on_training_rows_only():
    P = synthetic_panel()
    r1 = wf.evaluate(P, ModelSpec(), TRAIN, VALIDATE)
    Q = P.copy()
    after = Q.index > pd.Timestamp(VALIDATE[1])
    Q.loc[after, [c for c in Q.columns if c.startswith("p_")]] *= -5.0
    assert wf.evaluate(Q, ModelSpec(), TRAIN, VALIDATE)["score"] == pytest.approx(r1["score"])
    R = P.copy()
    in_val = (R.index >= pd.Timestamp(VALIDATE[0])) & (R.index <= pd.Timestamp(VALIDATE[1]))
    R.loc[in_val, "p_us_equity"] += 0.05
    r3 = wf.evaluate(R, ModelSpec(), TRAIN, VALIDATE)
    assert r3["train_r2"] == pytest.approx(r1["train_r2"])        # same fit ...
    assert r3["per_asset"]["us_equity"] != pytest.approx(r1["per_asset"]["us_equity"])   # ... new score


SUBSET = {"add_valuation", "add_sentiment", "add_wage", "add_white_collar", "lags1", "ridge_0.05", "window_15"}


def test_greedy_selects_the_factor_that_truly_matters():
    P = synthetic_panel()
    cands = [c for c in wf.build_candidates(P, TRAIN) if c.id in SUBSET]
    sel = wf.greedy_select(P, ModelSpec(), cands, TRAIN, VALIDATE)
    assert sel["accepted"][0] == "add_sentiment"
    assert sel["result"]["score"] > sel["path"][0]["val_score"] + 0.1
    assert [p["change"] for p in sel["path"]][:2] == [None, "add_sentiment"]


def test_greedy_never_accepts_an_endogenous_candidate():
    P = synthetic_panel(beta_sent=0.0)              # nothing is missing; only the price-built series "helps"
    cands = [c for c in wf.build_candidates(P, TRAIN) if c.id in SUBSET]
    sel = wf.greedy_select(P, ModelSpec(), cands, TRAIN, VALIDATE)
    first = [t for t in sel["trials"] if t["round"] == 1]
    best = max(first, key=lambda t: t["delta"])
    assert best["candidate"] == "add_valuation" and best["delta"] > 0.02 and not best["eligible"]
    assert "add_valuation" not in sel["accepted"]
    assert all(t["round"] == 1 for t in sel["trials"] if t["candidate"] == "add_valuation")


def test_diagnose_flags_the_missing_factor():
    P = synthetic_panel()
    cands = wf.build_candidates(P, TRAIN)
    findings = {f["id"]: f for f in dg.diagnose(P, ModelSpec(), TRAIN, VALIDATE, cands)}
    assert set(findings) == {"overfit", "missing_factor", "missing_lags", "regime", "parameter_break",
                             "nonlinearity", "drift_bias"}
    mf = findings["missing_factor"]
    assert mf["flagged"] and "add_sentiment" in mf["suggests"]
    assert not findings["parameter_break"]["flagged"] and not findings["missing_lags"]["flagged"]


def test_final_spec_must_survive_the_untouched_test():
    its = [{"iter": 1, "train": TRAIN, "validate": VALIDATE, "test": TEST}]
    held = wf.run_walkforward(synthetic_panel(), iterations=its)
    assert held["final"]["changes"][:1] == ["add_sentiment"] and not held["final"]["is_base"]
    flipped = wf.run_walkforward(synthetic_panel(flip_in_test=True), iterations=its)
    it = flipped["iterations"][0]
    assert "add_sentiment" in it["selected"]["changes"] and not it["test_result"]["improved"]
    assert flipped["final"]["is_base"] and flipped["final"]["spec"]["factors"] == list(config.CORE_FACTOR_IDS)
    assert flipped["final"]["last_selected"]["changes"] == it["selected"]["changes"]


def test_evaluate_scores_against_the_training_mean():
    P = synthetic_panel()
    r = wf.evaluate(P, ModelSpec(), TRAIN, VALIDATE)
    from scenariolab.engine.factor_model import AssetFactorModel
    m = AssetFactorModel(ModelSpec()).fit(P, *TRAIN)
    y = P.loc[VALIDATE[0]:VALIDATE[1], "p_us_equity"]
    yhat = m.predict(P).loc[VALIDATE[0]:VALIDATE[1], "us_equity"]
    bench = P.loc[TRAIN[0]:TRAIN[1], "p_us_equity"].mean()
    expected = 1 - ((y - yhat) ** 2).sum() / ((y - bench) ** 2).sum()
    assert r["per_asset"]["us_equity"] == pytest.approx(expected)
    # a 15-year-window spec is scored against the same full-training-period mean
    w = wf.evaluate(P, ModelSpec(window_years=15), TRAIN, VALIDATE)
    yhat_w = AssetFactorModel(ModelSpec(window_years=15)).fit(P, *TRAIN).predict(P).loc[VALIDATE[0]:VALIDATE[1], "us_equity"]
    assert w["per_asset"]["us_equity"] == pytest.approx(1 - ((y - yhat_w) ** 2).sum() / ((y - bench) ** 2).sum())


def test_carried_over_spec_counts_when_it_beats_the_base(monkeypatch):
    """Iteration 2 starts from iteration 1's pick and accepts nothing new: the pick still has to be
    judged on iteration 2's test, not silently dropped because this round made no change."""
    its = [{"iter": 1, "train": ("1980-01", "1994-12"), "validate": ("1995-01", "2004-12"), "test": ("2005-01", "2009-12")},
           {"iter": 2, "train": ("1980-01", "2004-12"), "validate": ("2005-01", "2009-12"), "test": ("2010-01", "2019-12")}]
    keep = {"add_sentiment", "add_wage", "add_white_collar"}
    build = wf.build_candidates
    monkeypatch.setattr(wf, "build_candidates", lambda panel, train: [c for c in build(panel, train) if c.id in keep])
    res = wf.run_walkforward(synthetic_panel(), iterations=its)
    it2 = res["iterations"][1]
    assert not it2["start"]["is_base"] and it2["selected"]["changes"] == []
    assert it2["test_result"]["delta"] > 0.1 and it2["test_result"]["improved"]
    assert res["final"]["changes"] == ["add_sentiment"] and not res["final"]["is_base"]
    assert res["final"]["test_used_for_choice"] == wf.REQUIRE_TEST_GAIN


def test_rolling_betas_track_a_known_break():
    P = synthetic_panel(beta_sent=0.0)
    late = P.index >= pd.Timestamp("2000-01-01")
    for a in MODEL_ASSETS:
        P.loc[late, f"p_{a}"] += 3.0 * P.loc[late, "f_y10"]       # y10 beta -1.5 -> +1.5 after 2000
    rb = dg.rolling_betas(P, ModelSpec(), window=120)
    sd = rb["factor_sd"]["y10"]
    beta = dict(zip(rb["years"], rb["assets"]["us_equity"]["beta"]["y10"]))
    assert beta["1995"] / sd == pytest.approx(-1.5, abs=0.3)
    assert beta["2015"] / sd == pytest.approx(1.5, abs=0.3)
