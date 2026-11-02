import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.analysis import diagnostics as dg
from scenariolab.analysis import walkforward as wf
from scenariolab.engine.factor_model import MODEL_ASSETS, AssetFactorModel, ModelSpec

TRAIN, VALIDATE, TEST = ("1980-01", "1999-12"), ("2000-01", "2009-12"), ("2010-01", "2019-12")
needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")


def synthetic_panel(beta_sent=0.01, flip_in_test=False, n=480, seed=0, start="1980-01") -> pd.DataFrame:
    """Core factors plus candidates; asset price returns load on oil, y10 and (truly) sentiment."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{start}-01", periods=n, freq="MS")
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


# the eligibility FACTORS_DESIGN §2.1 pre-registers: endogenous series, the neural networks (report-only while
# MLP_ELIGIBLE is False) and the audit's three comparisons are scored but never accepted
INELIGIBLE = {"add_valuation", "add_wealth", "add_cape", "mlp16", "mlp32", "add_tp10_x_infl4", "add_tp10", "add_ebp"}


def _with_shocks(P: pd.DataFrame, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    for col in wf.SHOCK_COLS:
        P[col] = rng.normal(size=len(P))
    return P


def test_the_round_registers_28_candidates_and_the_networks_are_report_only(monkeypatch):
    """Test 16: today's 22 candidates plus the six of the missing-factor round, with §2.1's eligibility."""
    cands = wf.build_candidates(_with_shocks(synthetic_panel()), TRAIN)
    by = {c.id: c for c in cands}
    assert len(cands) == 28 and len(by) == 28
    assert wf.MLP_ELIGIBLE is False
    assert {c.id for c in cands if not c.eligible} == INELIGIBLE
    for cid in ("mlp16", "mlp32"):
        assert by[cid].note == wf.MLP_NOTE and "线性" in by[cid].note
    round_ids = [c.id for c in cands][-6:]
    assert round_ids == ["add_gz", "add_tp10_x_core4", "drop_labor_share", "add_tp10_x_infl4", "add_tp10", "add_ebp"]
    assert by["add_gz"].group == "missing_factor" and by["add_tp10_x_core4"].group == "missing_mechanism"
    assert by["add_tp10_x_core4"].change == {"add_factors": ["tp10_x_core4"]}
    assert f"> {wf.TP_THRESHOLD:g}%" in by["add_tp10_x_core4"].hypothesis and wf.TP_THRESHOLD == 4.0
    # drop_labor_share removes the factor from a spec that has it and is a no-op on one that does not
    base = ModelSpec()
    assert "labor_share" in base.factors and not by["drop_labor_share"].is_noop(base)
    dropped = by["drop_labor_share"].apply(base)
    assert "labor_share" not in dropped.factors and by["drop_labor_share"].is_noop(dropped)
    # ... and is never a "missing factor" column of the residual tests
    assert all(cid != "drop_labor_share" for cid, _ in dg._candidate_columns(cands).values())
    # a candidate whose column the panel lacks is not built; reverting D1 is one line
    assert "add_gz" not in {c.id for c in wf.build_candidates(synthetic_panel().drop(columns=["f_gz"]), TRAIN)}
    # the audit's other protocol: the networks eligible for one run, without touching the constant
    build = wf.build_candidates
    monkeypatch.setattr(wf, "build_candidates", lambda panel, train: [c for c in build(panel, train) if c.id == "mlp16"])
    its = [{"iter": 1, "train": TRAIN, "validate": VALIDATE, "test": TEST}]
    run = wf.run_walkforward(synthetic_panel(), iterations=its, dedicated=[], mlp_eligible=True)
    assert run["mlp_eligible"] is True and run["candidates"][0]["eligible"] and wf.MLP_ELIGIBLE is False
    monkeypatch.setattr(wf, "MLP_ELIGIBLE", True)
    again = {c.id: c for c in build(synthetic_panel(), TRAIN)}
    assert again["mlp32"].eligible and again["mlp16"].eligible and again["mlp32"].note == ""


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
    # a report-only comparison is not labelled endogenous
    rows = {r["candidate"]: r for r in mf["table"]}
    assert rows["add_valuation"]["endogenous"] and not rows["add_tp10"]["endogenous"] and not rows["add_tp10"]["eligible"]


def test_final_spec_must_survive_the_untouched_test():
    its = [{"iter": 1, "train": TRAIN, "validate": VALIDATE, "test": TEST}]
    held = wf.run_walkforward(synthetic_panel(), iterations=its, dedicated=[])
    assert held["final"]["changes"][:1] == ["add_sentiment"] and not held["final"]["is_base"]
    flipped = wf.run_walkforward(synthetic_panel(flip_in_test=True), iterations=its, dedicated=[])
    it = flipped["iterations"][0]
    assert "add_sentiment" in it["selected"]["changes"] and not it["test_result"]["improved"]
    assert flipped["final"]["is_base"] and flipped["final"]["spec"]["factors"] == list(config.CORE_FACTOR_IDS)
    assert flipped["final"]["last_selected"]["changes"] == it["selected"]["changes"]
    assert flipped["final"]["common"]["kept"] == "base" and flipped["stage_b"]["accepted"] == []


def test_evaluate_scores_against_the_training_mean():
    P = synthetic_panel()
    r = wf.evaluate(P, ModelSpec(), TRAIN, VALIDATE)
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


TWO_ITERS = [{"iter": 1, "train": ("1980-01", "1994-12"), "validate": ("1995-01", "2004-12"), "test": ("2005-01", "2009-12")},
             {"iter": 2, "train": ("1980-01", "2004-12"), "validate": ("2005-01", "2009-12"), "test": ("2010-01", "2019-12")}]


def test_carried_over_spec_is_kept_when_the_next_iteration_adds_nothing(monkeypatch):
    """Iteration 2 starts from iteration 1's pick and accepts nothing new: under the incremental gate there is
    nothing to adopt, and the pick it started from is the common spec (not silently replaced by the base)."""
    keep = {"add_sentiment", "add_wage", "add_white_collar"}
    build = wf.build_candidates
    monkeypatch.setattr(wf, "build_candidates", lambda panel, train: [c for c in build(panel, train) if c.id in keep])
    res = wf.run_walkforward(synthetic_panel(), iterations=TWO_ITERS, dedicated=[])
    it2 = res["iterations"][1]
    assert not it2["start"]["is_base"] and it2["selected"]["changes"] == []
    tr = it2["test_result"]
    assert tr["delta"] > 0.1 and tr["delta_vs_start"] == 0.0 and not tr["improved"]
    assert res["final"]["changes"] == ["add_sentiment"] and not res["final"]["is_base"]
    assert res["final"]["common"]["kept"] == "start"
    assert res["final"]["test_used_for_choice"] == wf.REQUIRE_TEST_GAIN


def gate_panel(flip_wage: bool) -> pd.DataFrame:
    """synthetic_panel plus a second true driver, wage, that keeps its beta to 2009 and (optionally) flips after."""
    P = synthetic_panel()
    rng = np.random.default_rng(11)
    P["f_wage"] = rng.normal(size=len(P))
    beta = np.where((P.index >= pd.Timestamp("2010-01-01")) & flip_wage, -0.005, 0.005)
    for a in MODEL_ASSETS:
        P[f"p_{a}"] += beta * P["f_wage"]
    return P


def _one_per_iteration(monkeypatch):
    """Iteration 1 may only add sentiment, iteration 2 only wage."""
    build = wf.build_candidates
    allowed = {"1994-12": {"add_sentiment"}, "2004-12": {"add_wage"}}
    monkeypatch.setattr(wf, "build_candidates",
                        lambda panel, train: [c for c in build(panel, train) if c.id in allowed[train[1]]])


def test_incremental_gate_keeps_the_start_when_the_new_changes_lose(monkeypatch):
    """Test 17: iteration 2 adds wage, which fits its validation decade but flips in its test decade. The
    selection still beats the base (sentiment carries it), so the old gate would adopt it; the incremental
    gate compares it with iteration 2's start and keeps the start."""
    _one_per_iteration(monkeypatch)
    res = wf.run_walkforward(gate_panel(flip_wage=True), iterations=TWO_ITERS, dedicated=[])
    it1, it2 = res["iterations"]
    # iteration 1 behaves as before: its start is the base, so both deltas are the same test against the base
    t1 = it1["test_result"]
    assert it1["start"]["is_base"] and "start" not in t1 and t1["delta_vs_start"] == t1["delta"] > wf.MIN_GAIN
    assert t1["improved"] and it1["selected"]["changes"] == ["add_sentiment"]
    # iteration 2: the selection beats the base but loses to its start
    t2 = it2["test_result"]
    assert it2["selected"]["changes"] == ["add_wage"] and not it2["start"]["is_base"]
    assert t2["delta"] > wf.MIN_GAIN                              # what the old gate looked at
    assert t2["delta_vs_start"] == pytest.approx(t2["selected"]["score"] - t2["start"]["score"])
    assert t2["delta_vs_start"] < -wf.MIN_GAIN and not t2["improved"] and t2["gate"] == "incremental"
    assert "保留第 1 轮的模型" in t2["text"] and "起始模型" in it2["carry_over"]
    fin = res["final"]
    assert fin["changes"] == ["add_sentiment"] and fin["common"]["kept"] == "start"
    assert fin["last_selected"]["changes"] == ["add_sentiment", "add_wage"]
    assert "没有胜过本轮起始模型" in fin["reason"] and "比基准" in fin["reason"]
    assert fin["test_score"] == pytest.approx(
        wf.evaluate(gate_panel(True), ModelSpec.from_dict(fin["spec"]), ("1980-01", "2009-12"), ("2010-01", "2019-12"))["score"])
    # the gate before this round (against the base), kept to report that protocol, adopts the losing changes
    old = wf.run_walkforward(gate_panel(flip_wage=True), iterations=TWO_ITERS, dedicated=[], gate="base")
    assert old["gate"] == "base" and old["iterations"][1]["test_result"]["improved"]
    assert old["final"]["changes"] == ["add_sentiment", "add_wage"]


def test_incremental_gate_adopts_a_selection_that_beats_its_start(monkeypatch):
    _one_per_iteration(monkeypatch)
    res = wf.run_walkforward(gate_panel(flip_wage=False), iterations=TWO_ITERS, dedicated=[])
    t2 = res["iterations"][1]["test_result"]
    assert t2["delta_vs_start"] > wf.MIN_GAIN and t2["improved"]
    assert res["final"]["changes"] == ["add_sentiment", "add_wage"] and res["final"]["common"]["kept"] == "selected"
    assert "胜过本轮起始模型" in res["final"]["reason"]


# ---------------------------------------------------------------- stage B

STAGE_B_ITERS = [{"iter": 1, "train": ("1970-01", "1994-12"), "validate": ("1995-01", "2004-12"), "test": ("2005-01", "2009-12")},
                 {"iter": 2, "train": ("1970-01", "2004-12"), "validate": ("2005-01", "2009-12"), "test": ("2010-01", "2019-12")}]
STAGE_B_CANDIDATES = [
    {"id": "energy_x", "asset": "energy", "eligible": True, "hypothesis": "能源股的专门因子",
     "change": {"asset_factors": {"energy": ["metals_oil"]}}},
    # listed first, but the later start fits the history better: the choice is made on validation
    {"id": "gold_window", "asset": "gold", "eligible": True, "hypothesis": "黄金的起始月份",
     "options": [{"asset_start": {"gold": "1976-01"}}, {"asset_start": {"gold": "1982-01"}}]},
    {"id": "industrials_wage", "asset": "industrials", "eligible": True, "hypothesis": "工业股的专门因子",
     "change": {"asset_factors": {"industrials": ["wage"]}}},
    {"id": "staples_noise", "asset": "staples", "eligible": True, "hypothesis": "必需消费股的噪声",
     "change": {"asset_factors": {"staples": ["sentiment"]}}},
    {"id": "gold_missing", "asset": "gold", "eligible": False, "hypothesis": "面板里没有的序列",
     "change": {"asset_factors": {"gold": ["not_in_panel"]}}},
]


def stage_b_panel() -> pd.DataFrame:
    """Energy loads on its own regressor; gold follows another relation (and mean) before 1982; industrials load
    on wage until 2009 and the opposite way after; sentiment drives nothing."""
    P = synthetic_panel(beta_sent=0.0, n=600, seed=3, start="1970-01")
    rng = np.random.default_rng(103)
    P["f_metals_oil"] = rng.normal(size=len(P))
    P["p_energy"] += 0.006 * P["f_metals_oil"]
    early = P.index < pd.Timestamp("1982-01-01")
    P.loc[early, "p_gold"] += 0.02 + 3.0 * P.loc[early, "f_oil"]
    P["f_wage"] = rng.normal(size=len(P))
    P["p_industrials"] += np.where(P.index >= pd.Timestamp("2010-01-01"), -0.006, 0.006) * P["f_wage"]
    return P


def test_stage_b_chooses_on_validation_gates_per_asset_and_uses_the_last_iteration(monkeypatch):
    """Test 18 on synthetic data."""
    P = stage_b_panel()
    monkeypatch.setattr(wf, "build_candidates", lambda panel, train: [])          # stage A keeps the base
    res = wf.run_walkforward(P, iterations=STAGE_B_ITERS, dedicated=STAGE_B_CANDIDATES)
    sb = res["stage_b"]
    rows = [{(r["id"], r["option"]): r for r in it["rows"]} for it in sb["iterations"]]
    for k, it in enumerate(rows):
        # the option is chosen on the asset's validation R^2 (not by its place in the list) and gated on the test
        early, late = it[("gold_window", "1976-01")], it[("gold_window", "1982-01")]
        assert late["val_gain"] > early["val_gain"] and late["chosen"] and not early["chosen"]
        assert late["accepted"] and not early["accepted"] and late["test_gain"] > wf.MIN_GAIN_ASSET
        assert it[("energy_x", "metals_oil")]["accepted"]
        noise = it[("staples_noise", "sentiment")]
        assert noise["evaluated"] and noise["val_gain"] <= wf.MIN_GAIN_ASSET and not noise["chosen"]
        missing = it[("gold_missing", "not_in_panel")]
        assert not missing["evaluated"] and missing["reason"] and not missing["accepted"]
    # industrials' regressor passes iteration 1 but fails iteration 2's test: the final follows the last iteration
    assert rows[0][("industrials_wage", "wage")]["accepted"]
    last_ind = rows[1][("industrials_wage", "wage")]
    assert last_ind["chosen"] and last_ind["test_gain"] < -wf.MIN_GAIN_ASSET and not last_ind["accepted"]
    assert sb["accepted"] == ["energy_x", "gold_window"] and sb["accepted_options"]["gold_window"] == "1982-01"
    fin = res["final"]
    assert fin["spec"]["asset_factors"] == {"energy": ["metals_oil"]} and fin["spec"]["asset_start"] == {"gold": "1982-01"}
    assert fin["changes"] == ["energy_x", "gold_window"] and fin["dedicated"] == ["energy_x", "gold_window"]
    assert fin["common"]["kept"] == "base" and fin["spec"]["name"] == "base+energy_x+gold_window"
    assert sb["test_used_for_choice"] is True and sb["final_rule"] == "last_iteration"
    # the asset's R^2 is scored against its mean price return over the WHOLE training window
    train, val = STAGE_B_ITERS[0]["train"], STAGE_B_ITERS[0]["validate"]
    m = AssetFactorModel(wf.with_change(ModelSpec(), {"asset_start": {"gold": "1982-01"}})).fit(P, *train)
    assert m.asset_start_used["gold"] == "1982-01"
    y, yhat = P.loc[val[0]:val[1], "p_gold"], m.predict(P).loc[val[0]:val[1], "gold"]
    sse = ((y - yhat) ** 2).sum()
    whole = 1 - sse / ((y - P.loc[train[0]:train[1], "p_gold"].mean()) ** 2).sum()
    own = 1 - sse / ((y - P.loc["1982-01":train[1], "p_gold"].mean()) ** 2).sum()
    assert rows[0][("gold_window", "1982-01")]["val"]["r2_after"] == pytest.approx(whole)
    assert abs(whole - own) > 1e-3                                 # the two benchmarks differ on this panel
    # every evaluated option is scored on the test months too, and the summary names what was adopted
    assert all(r["test_gain"] is not None for r in rows[1].values() if r["evaluated"])
    assert "黄金从 1982 年起估计" in sb["text"] and "能源股" in sb["text"]


def test_stage_b_needs_a_linear_common_spec():
    P = stage_b_panel()
    rows = wf.dedicated_pass(P, ModelSpec(kind="mlp"), STAGE_B_ITERS[:1], STAGE_B_CANDIDATES[:1])["iterations"][0]["rows"]
    assert not rows[0]["evaluated"] and not rows[0]["accepted"] and "线性" in rows[0]["reason"]


def test_residual_analysis_uses_each_assets_own_equation():
    """The final spec's per-asset parts reach the full-sample diagnostics: gold's tests start at its own first
    month, energy's design holds its extra regressor, and an asset without either is unchanged."""
    P = stage_b_panel()
    spec = ModelSpec.from_dict({"asset_factors": {"energy": ["metals_oil"]}, "asset_start": {"gold": "1982-01"}})
    cands = wf.build_candidates(P, ("1970-01", "2019-12"))
    ra = dg.residual_analysis(P, spec, cands)
    plain = dg.residual_analysis(P, ModelSpec(), cands)
    assert ra["assets"]["gold"]["fit_start"] == "1982-01" and plain["assets"]["gold"]["fit_start"] == config.MODEL_START
    assert ra["assets"]["energy"]["own_factors"] == ["metals_oil"] and ra["asset_start_used"] == {"gold": "1982-01"}
    for a in ("us_equity", "industrials"):
        assert ra["assets"][a]["chow_p"] == pytest.approx(plain["assets"][a]["chow_p"], rel=1e-9)
        assert ra["assets"][a]["ljung_box_q"] == pytest.approx(plain["assets"][a]["ljung_box_q"], rel=1e-9)
    # gold's Chow test on 1982+ only no longer sees the pre-1982 relation that the plain fit mixes in
    assert ra["assets"]["gold"]["chow_p"] > plain["assets"]["gold"]["chow_p"]
    rb = ra["rolling_betas"]
    rb_plain = dg.rolling_betas(P, ModelSpec())
    assert rb["assets"]["us_equity"]["beta"]["oil"] == pytest.approx(rb_plain["assets"]["us_equity"]["beta"]["oil"])
    assert rb["assets"]["energy"]["beta"]["oil"] != pytest.approx(rb_plain["assets"]["energy"]["beta"]["oil"])


def test_recommendations_answer_the_four_questions_and_keep_the_regime_contract(monkeypatch):
    """The regime layer finds the inflation-state recommendation by its title and reads its per-decade deltas."""
    P = _with_shocks(synthetic_panel())
    res = wf.run_walkforward(P, iterations=TWO_ITERS, diagnose=dg.diagnose, dedicated=[])
    resid = dg.residual_analysis(P, ModelSpec.from_dict(res["final"]["spec"]),
                                 wf.build_candidates(P, ("1980-01", "2019-12")))
    recs = dg.recommendations(res, resid, ["信用利差的计时修正"])
    assert {r["group"] for r in recs} >= set(wf.QUESTIONS)
    reg = [r for r in recs if "通胀状态依赖" in r["title"]]
    assert len(reg) == 1 and isinstance(reg[0]["evidence"]["oos_delta"], list)
    assert [d["period"] for d in reg[0]["evidence"]["oos_delta"]] == ["1995–2004", "2005–2009", "2010–2019"]
    nn = next(r for r in recs if r["group"] == "neural_net")
    assert nn["title"].startswith("只作对照") and wf.MLP_NOTE in nn["detail"]
    ridge = next(r for r in recs if "岭回归" in r["title"])
    assert set(ridge["evidence"]["oos_delta"]) == {"ridge_0.01", "ridge_0.05"}
    assert len(recs) <= dg.MAX_RECS


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


# ---------------------------------------------------------------- the build's panel

@needs_panel
def test_walkforward_on_the_panel_gives_the_expected_selection():
    """Test 19 (FACTORS_DESIGN §2.2): iteration 1 adopts GZ, the term-premium term and drops labor share;
    iteration 2's changes lose to that start on the untouched test; stage B adds the energy regressor and gold's
    1982 start. Tolerances of §10: 1e-4 on the paths, 0.002 on iteration 2 (same decision), 0.001 on the final."""
    from scenariolab.context import load_panel_with_shocks

    P = load_panel_with_shocks()
    if not {"f_gz", "f_tp10_x_core4", "f_metals_oil"} <= set(P.columns):
        pytest.skip("panel built before the missing-factor round")
    res = wf.run_walkforward(P, ModelSpec())
    assert len(res["candidates"]) == 28
    assert {c["id"] for c in res["candidates"] if not c["eligible"]} == INELIGIBLE
    it1, it2 = res["iterations"]
    assert it1["selected"]["changes"] == ["add_gz", "add_tp10_x_core4", "drop_labor_share"]
    assert [p["val_score"] for p in it1["selected"]["path"]] == pytest.approx([-0.0114, 0.0372, 0.0625, 0.0754], abs=1e-4)
    assert it1["test_result"]["base"]["score"] == pytest.approx(0.0162, abs=1e-4)
    assert it1["test_result"]["selected"]["score"] == pytest.approx(0.1679, abs=1e-4)
    assert it1["test_result"]["improved"]
    # iteration 2's selection holds the structural shocks, so this value moves with data/processed/structural_shocks.csv:
    # the design's -0.0088 used the shocks identified on the old spread (sha1 0b0124639c02); the causality step of the
    # full build re-identifies them on the fixed spread (seed 0; e_financial corr 0.70 with the old ones), which gives
    # -0.0114 with the same selection and the same decision. Either way the start is kept, which is what matters.
    import hashlib
    shocks = config.PROCESSED_DIR / "structural_shocks.csv"
    old = shocks.exists() and hashlib.sha1(shocks.read_bytes()).hexdigest().startswith("0b0124639c02")
    path2 = [0.1679, 0.1867, 0.1951, 0.2016] if old else [0.1679, 0.1867, 0.2045, 0.2112]
    assert it2["selected"]["changes"] == ["add_real_y10", "shocks_replace_oil", "add_profit_growth"]
    assert [p["val_score"] for p in it2["selected"]["path"]] == pytest.approx(path2, abs=1e-4)
    assert not it2["test_result"]["improved"]
    assert it2["test_result"]["delta_vs_start"] == pytest.approx(-0.0088 if old else -0.0114, abs=0.002)
    assert it2["test_result"]["delta"] == pytest.approx(0.0711 if old else 0.0685, abs=0.002)
    sb = res["stage_b"]
    assert sb["accepted"] == ["energy_metals_oil", "gold_window"] and sb["accepted_options"]["gold_window"] == "1982-01"
    fin = res["final"]
    assert fin["spec"]["factors"] == ["oil", "ip", "cpi_goods", "cpi_services", "ff", "y10", "spread", "usd", "unrate",
                                      "gz", "tp10_x_core4"]
    assert fin["spec"]["asset_factors"] == {"energy": ["metals_oil"]} and fin["spec"]["asset_start"] == {"gold": "1982-01"}
    assert fin["describe"] == "11 个因子、能源股 +1 个专门因子、黄金从 1982 年起估计"
    assert fin["test_score"] == pytest.approx(0.2678, abs=0.001) and fin["test_months"] == 141
