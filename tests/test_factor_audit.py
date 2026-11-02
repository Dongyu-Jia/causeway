"""The missing-factor audit: the ported test statistics, the replication of the research archive, the residual
anatomy and out-of-universe check values (FACTORS_DESIGN §8 tests 20-24) and the factor_audit.json contract.

The statistics run on synthetic data; the replication and the check values need the built panel (needs_panel),
the out-of-universe test also the cached French industry file. Nothing here touches the network."""
import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from scenariolab import config, jsonio
from scenariolab.analysis import factor_audit as A
from scenariolab.analysis import factor_tests as T
from scenariolab.engine.factor_model import MODEL_ASSETS, ModelSpec

CORE9 = ("oil", "ip", "cpi_goods", "cpi_services", "ff", "y10", "spread", "usd", "unrate")
S_SPEC = ModelSpec(factors=CORE9 + ("gz", "tp10_x_core4"), name="S")
F_SPEC = ModelSpec(factors=CORE9 + ("gz", "tp10_x_core4"), asset_factors={"energy": ["metals_oil"]},
                   asset_start={"gold": "1982-01"}, name="F")
NEEDED = ["x_spread_mixed", "x_labor_share_interp", "f_gz", "f_tp10_x_core4", "f_metals_oil", "s_mort_spread"]


def _panel_or_none():
    if not config.PANEL_PATH.exists():
        return None
    from scenariolab.data.panel import load_panel
    P = load_panel()
    return P if all(c in P.columns for c in NEEDED) else None


@pytest.fixture(scope="module")
def panel():
    P = _panel_or_none()
    if P is None:
        pytest.skip("panel not built with the missing-factor columns")
    return T.month_start(P)


@pytest.fixture(scope="module")
def archive():
    return A.load_archive()


@pytest.fixture(scope="module")
def replication(panel, archive):
    return A.replicate(panel, archive, placebo_n=20)


# ---------------------------------------------------------------- synthetic data for the statistics

def synthetic_panel(seed=3, start="1970-01", end="2026-09", beta=0.6):
    """A panel with the core factor columns and every model asset's price return; `cand` drives us_equity."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="MS")
    n = len(idx)
    P = pd.DataFrame(index=idx)
    for f in config.CORE_FACTOR_IDS:
        P[f"f_{f}"] = rng.standard_normal(n)
    cand = rng.standard_normal(n)
    for j, a in enumerate(MODEL_ASSETS):
        y = 0.01 * rng.standard_normal(n) + 0.002 * P["f_oil"].to_numpy() * (j % 3)
        if a == "us_equity":
            y = y + 0.01 * beta * cand
        P[f"p_{a}"] = y
    return P, pd.Series(cand, index=idx, name="cand")


def brute_bh(p, q=0.10):
    m = len(p)
    order = np.argsort(p)
    qv = np.empty(m)
    for r, i in enumerate(order):                       # q_(r) = min over k >= r of p_(k) m / k
        qv[i] = min(min(p[order[k]] * m / (k + 1) for k in range(r, m)), 1.0)
    return qv, qv <= q


# ---------------------------------------------------------------- test 21: bh, placebo, the transmission core

def test_bh_equals_brute_force():
    rng = np.random.default_rng(11)
    for m in (1, 7, 50, 400):
        p = np.concatenate([rng.uniform(0, 1, m - m // 5), rng.uniform(0, 0.002, m // 5)])
        out = T.bh(p)
        qv, rej = brute_bh(p)
        assert np.allclose(out["q_value"].to_numpy(), qv, rtol=0, atol=1e-15)
        assert (out["reject"].to_numpy() == rej).all()
    s = pd.Series([0.01, np.nan, 0.04, 0.5], index=list("abcd"))
    out = T.bh(s)
    assert np.isnan(out.loc["b", "q_value"]) and not out.loc["b", "reject"]
    qv, _ = brute_bh(np.array([0.01, 0.04, 0.5]))
    assert np.allclose(out.loc[["a", "c", "d"], "q_value"], qv)


def test_circular_placebo_is_deterministic_per_candidate():
    P, cand = synthetic_panel()
    bench = T.Bench(P)
    a = T.circular_placebo(bench, "cand", cand, ["us_equity"], n=30)
    b = T.circular_placebo(T.Bench(P), "cand", cand, ["us_equity"], n=30)
    assert a == b
    assert np.array_equal(T.placebo_shifts("gz", 640), T.placebo_shifts("gz", 640))
    assert not np.array_equal(T.placebo_shifts("gz", 640), T.placebo_shifts("tp10_x_core4", 640))
    s = T.placebo_shifts("gz", 640)
    assert len(s) == T.PLACEBO_N and s.min() >= 24 and s.max() < 640 - 24
    # a real driver of us_equity beats every shifted copy of itself: the smallest possible p, 1 / (n + 1)
    assert a["us_equity"]["n"] == 30 and a["us_equity"]["p"] == pytest.approx(1 / 31)
    assert a["us_equity"]["z_obs"] > a["us_equity"]["q95"]


def test_same_sample_transmission_equals_brute_force_ols():
    P, cand = synthetic_panel()
    late = cand.where(cand.index >= "1980-01")              # a late starter: the base is refitted on its months
    bench = T.Bench(P)
    rows, pooled = bench.transmission({"late": late}, decades=["T2"])
    r = rows[rows.asset == "us_equity"].iloc[0]
    tr = (P.index >= "1980-01") & (P.index <= "2004-12")
    te = (P.index >= "2005-01") & (P.index <= "2014-12")
    X = P[[f"f_{f}" for f in config.CORE_FACTOR_IDS]].to_numpy()
    y = P["p_us_equity"].to_numpy()

    def fit_pred(Xm):
        A1 = np.column_stack([np.ones(tr.sum()), Xm[tr]])
        b = np.linalg.lstsq(A1, y[tr], rcond=None)[0]
        return np.column_stack([np.ones(te.sum()), Xm[te]]) @ b
    bench_mean = P.loc["1973-01":"2004-12", "p_us_equity"].mean()
    sst = ((y[te] - bench_mean) ** 2).sum()
    r2_base = 1 - ((y[te] - fit_pred(X)) ** 2).sum() / sst
    r2_cand = 1 - ((y[te] - fit_pred(np.column_stack([X, late.to_numpy()]))) ** 2).sum() / sst
    assert r["n_train"] == tr.sum() and r["train_start"] == "1980-01" and r["n_test"] == te.sum()
    assert r["delta_ss"] == pytest.approx(r2_cand - r2_base, abs=1e-12)
    assert r["delta"] != pytest.approx(r["delta_ss"], abs=1e-6)          # the full-window base differs
    assert r["dm_t"] > 2 and r["cw_t"] >= r["dm_t"]
    z, k = T.stouffer([1.0, 2.0, 3.0, np.nan], [263, 383, 50, 400])
    assert k == 2 and z == pytest.approx(3 / math.sqrt(2))


# ---------------------------------------------------------------- test 20: replication of the archive

def test_replication_reproduces_the_archive(replication, archive):
    checks = replication["checks"]
    assert checks["count_mismatches"] == []
    assert checks["required"] == ["spread_fixed", "gz", "tp10_x_core4", "metals_oil"]
    for cid in checks["required"]:
        c = replication["candidates"][cid]
        assert c["available"], cid
        assert c["diff"]["max_delta"] <= 1e-6, (cid, c["diff"])
        assert c["diff"]["max_z"] <= 1e-4 and c["diff"]["max_t"] <= 1e-4, (cid, c["diff"])
        # the panel's series are the verifier's full-precision rebuild: its z agree to rounding error
        assert c["diff"]["max_z_vs_rebuilt"] <= 1e-6, (cid, c["diff"])
        arc = archive["battery"][c["archive_key"]]
        for a in c["assets"]:
            for d in ("T1", "T2", "T3"):
                got, want = c["battery"]["base"][a][d], arc["base"][a][d]
                assert got["n_train"] == want["n_train"]                       # counts exactly
                assert abs(got["delta_ss"] - want["delta_ss"]) <= 1e-6
        assert [c["battery"]["base"][c["assets"][0]][d]["n_test"] for d in ("T1", "T2", "T3")] == [120, 120, 141]
    assert checks["b0_max_diff"] <= 1e-6
    assert replication["core4_active_months"] == archive["rederived"]["regime_active_months_core4"]
    assert checks["predictive_max_diff"] <= 1e-6
    assert {r["id"] for r in replication["predictive"]} == {"re_loans", "labor_share_rt", "philly_capex"}
    for r in replication["predictive"]:
        assert r["n"] == r["n_archive"], r["id"]
    assert checks["ok"]


def test_the_research_base_is_rebuilt_from_the_audit_columns(panel):
    P0 = A.research_panel(panel)
    assert (P0["f_spread"].dropna() == panel["x_spread_mixed"].diff().dropna()).all()
    # the spread fix changes 1962-85 only; the labor share now changes in release months only
    diff = (panel["s_spread"] - P0["s_spread"]).abs()
    assert diff.loc["1986-01":].max() < 1e-12 and diff.loc["1973-01":"1985-12"].max() > 0.1
    months = panel["f_labor_share"].loc["1990-01":"1999-12"].fillna(0) != 0
    assert set(months[months].index.month) == {2, 5, 8, 11}


# ---------------------------------------------------------------- tests 22 and 24: current-mode check values

def test_residual_anatomy_of_the_final_model(panel):
    an, R = A.residual_anatomy(panel, F_SPEC)
    assert an["months"] == 501 and an["first"] == "1985-01"
    assert an["pc1_share"] == pytest.approx(0.514, abs=0.002)
    assert an["pd_r2_us_equity"] == pytest.approx(0.581, abs=0.002)
    base, _ = A.residual_anatomy(A.research_panel(panel), ModelSpec(factors=A.B0_FACTORS))
    assert base["pc1_share"] == pytest.approx(0.556, abs=0.002) and base["pd_r2_us_equity"] == pytest.approx(0.777, abs=0.002)


def test_final_model_scores_and_leave_one_out(panel):
    res = A._decade_scores(panel, {"F": F_SPEC})
    assert [res["F"][d]["score"] for d in ("T1", "T2", "T3")] == pytest.approx([0.0839, 0.1858, 0.2678], abs=1e-4)
    assert res["F"]["T3"]["n_months"] == 141
    loo = A._loo(panel, S_SPEC)                  # in the common spec S (FACTORS_DESIGN §0, d04)
    assert loo["gz"] == pytest.approx([0.0669, 0.1376, 0.0269], abs=1e-4)
    assert loo["tp10_x_core4"] == pytest.approx([0.0247, 0.0479, 0.0158], abs=1e-4)
    assert loo["labor_share"] == pytest.approx([0.0129, -0.0029, -0.0040], abs=1e-4)
    ded = A._loo(panel, F_SPEC)
    assert ded["metals_oil"] == pytest.approx([0.0033, 0.0078, 0.0150], abs=1e-4)
    assert ded["gold_window"] == pytest.approx([0.0815, 0.1711, 0.1081], abs=1e-4)


@pytest.mark.skipif(not (config.RAW_DIR / "french" / "12_Industry_Portfolios_CSV.zip").exists(),
                    reason="French industry file not cached")
def test_out_of_universe_final_common_set(panel):
    oou = A.out_of_universe(panel, A.research_panel(panel), S_SPEC)
    assert oou["n"] == 6 and oou["n_better"] == [6, 6, 6]
    assert oou["delta_vs_base"] == pytest.approx([0.2031, 0.3011, 0.1194], abs=1e-4)


# ---------------------------------------------------------------- test 23: the JSON contract

TOP = {"version", "data_version", "archive", "summary", "coverage", "data_fixes", "changes", "plausible", "families",
       "per_asset", "bh", "refutation", "forecast_states", "upper_bound", "residuals", "model_vs_factor",
       "out_of_universe", "protocol", "concepts", "untested", "no_source", "issues", "notes"}
ARCHIVE_SECTIONS = ("bh", "upper_bound", "model_vs_factor")


@pytest.fixture(scope="module")
def audit(panel):
    from scenariolab.context import EngineContext
    return json.loads(jsonio.dumps(A.run(EngineContext.load(selection=True, spec=F_SPEC))))


def _strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def check_contract(out: dict):
    assert TOP <= set(out), TOP - set(out)
    assert out["version"] == 1
    assert {"panel_sha1", "model_selection_sha1", "data_end"} <= set(out["data_version"])
    assert out["archive"]["id"] == "research_2026_11_01" and out["archive"]["panel_sha1"] == "3869fdde3437"
    assert out["archive"]["source"] == "archive" and "研究轮存档" in out["archive"]["tag"]
    for sec in ARCHIVE_SECTIONS:
        assert out[sec]["source"] == "archive", sec
    assert all(r["source"] == "archive" for r in out["families"] + out["plausible"] + out["concepts"])
    assert out["residuals"]["source"] == "recomputed"
    assert {r["source"] for r in out["refutation"]} <= {"archive", "recomputed"}
    s = out["summary"]
    assert (s["n_transmission"], s["n_predictive"], s["families"]) == (476, 369, 6)
    assert s["transmission"] == {"CONFIRMED": 4, "PLAUSIBLE": 7, "REFUTED": 465}
    assert s["predictive"] == {"PLAUSIBLE": 21, "REFUTED": 348}
    assert s["changes"] == {"data_fixes": 3, "data_fixes_changing_model": 2, "common": 3, "dedicated": 2}
    # the summary agrees with its sections
    kinds = [c["kind"] for c in out["changes"]]
    assert s["changes"]["data_fixes"] == kinds.count("data_fix") and s["changes"]["dedicated"] == kinds.count("dedicated")
    fams = out["families"]
    assert sum(f["tested_n"]["transmission"] for f in fams) == s["n_transmission"]
    assert sum(f["tested_n"]["predictive"] for f in fams) == s["n_predictive"]
    for v in ("CONFIRMED", "PLAUSIBLE", "REFUTED"):
        assert sum(f["verdicts"]["transmission"][v] for f in fams) == s["transmission"][v]
    for k in ("B0", "B1", "final"):
        assert len(s["score"][k]) == 3
    assert len(out["per_asset"]) == 10 and all(len(r["r2"]["final"]) == 3 for r in out["per_asset"])
    assert s["unexplained_share_2015_2026"] == pytest.approx(np.mean([r["unexplained"] for r in out["per_asset"]]), abs=1e-5)
    assert out["bh"]["transmission_pairs"]["n"] == 3960 and out["bh"]["transmission_pairs"]["rejections"] == 17
    assert len(out["bh"]["transmission_pairs"]["rows"]) == 17
    assert (out["bh"]["prediction"]["n"], out["bh"]["prediction"]["cw"], out["bh"]["prediction"]["dm"],
            out["bh"]["prediction"]["calibrated"]) == (5634, 17, 0, 1)
    assert out["bh"]["raw_dm_would_add"]["n"] == 45 and out["bh"]["raw_dm_would_add"]["gold"] == 28
    assert len(out["concepts"]) == 42
    assert out["concept_counts"] == {"earns": 10, "partial": 10, "null": 12, "untested": 6, "none": 4}
    for c in out["changes"]:
        assert {"id", "name", "kind", "definition", "alignment", "coverage", "walkforward", "replication",
                "refutation", "loo_in_final", "verdict", "caveats"} <= set(c), c["id"]
        assert c["kind"] in ("data_fix", "common", "dedicated") and isinstance(c["caveats"], list)
    for f in out["data_fixes"]:
        assert {"id", "name", "before", "after", "score_delta", "per_asset", "evidence"} <= set(f)
    for row in out["coverage"]:
        assert {"col", "name", "role", "first", "last", "share_since_1973_02", "full_history", "alignment",
                "lag_months", "revised", "copyright"} <= set(row)
    assert {"replicated", "plausible", "engine"} <= set(out["forecast_states"])
    assert {"archive_U4", "final_common"} <= set(out["out_of_universe"])
    assert out["protocol"]["gate"] == "incremental" and out["protocol"]["source"] in ("archive", "recomputed")
    assert [r["id"] for r in out["protocol"]["archive_alternatives"]] == ["spec", "old_gate", "mlp_eligible"]
    for text in _strings(out):                                   # no key ever, a redacted URL at most
        assert "api_key=" not in text.replace("api_key=REDACTED", ""), text[:120]


def test_factor_audit_json_contract(audit):
    check_contract(audit)
    assert audit["model"]["is_expected"] and audit["issues"] == []
    assert audit["summary"]["replication_ok"]
    assert audit["summary"]["score"]["final"] == pytest.approx([0.0839, 0.1858, 0.2678], abs=1e-4)
    assert audit["summary"]["unexplained_share_2015_2026"] == pytest.approx(0.71, abs=0.01)
    ch = {c["id"]: c for c in audit["changes"]}
    assert [c["id"] for c in audit["changes"]] == ["spread_timing", "release_dating", "sentiment_hold", "add_gz",
                                                   "add_tp10_x_core4", "drop_labor_share", "energy_metals_oil",
                                                   "gold_window"]
    assert ch["spread_timing"]["table"]["values"] == pytest.approx([0.0156, 0.0894, 0.0380], abs=1e-4)
    assert ch["release_dating"]["table"]["values"] == pytest.approx([-0.0022, 0.0306, 0.0057], abs=1e-4)
    assert ch["add_gz"]["table"]["values"] == pytest.approx([0.0487, 0.1074, 0.0292], abs=1e-4)
    assert ch["add_tp10_x_core4"]["table"]["values"] == pytest.approx([0.0081, 0.0213, 0.0135], abs=1e-4)
    assert ch["drop_labor_share"]["table"]["values"] == pytest.approx([0.0119, -0.0073, 0.0030], abs=1e-4)
    assert all(c["status"] == "adopted" for c in audit["changes"] if c["kind"] != "data_fix")
    assert "研究轮存档" not in "".join(ch["gold_window"]["caveats"]) and "同一批十年" in ch["gold_window"]["caveats"][0]
    today = audit["today"]
    assert today["tp_term_active_months_monthdb"] == 231 and today["monthdb_months"] == 644
    assert today["tp_term_on"] is False and today["tp_term_last_on"] == "2023-12"
    # d07's readings: percentiles over the MonthDB's months, 1973-02 on (exact for the 2026-09 panel)
    if today["month"] == "2026-09":
        assert today["gz"]["pct_since_1973"] == pytest.approx(0.085404, abs=1e-6)
        assert today["tp10"]["pct_since_1973"] == pytest.approx(0.335404, abs=1e-6)
    # §6.2 I: each asset's honest range over the six common-penalty fits and the hindsight beta-drift ceiling
    ub = audit["upper_bound"]
    hr = {r["asset"]: r["honest_range_T3"] for r in ub["per_asset"] if r["asset"] != "mean"}
    assert hr["us_equity"] == pytest.approx([0.339, 0.403], abs=1e-3) and hr["gold"] == pytest.approx([-0.169, -0.062], abs=1e-3)
    assert ub["beta_drift"]["by_decade_range"] == pytest.approx([0.06, 0.29], abs=0.005)
    # the replication's differences survive the 6-decimal rounding of results files as text
    sci = audit["replication"]["checks"]["sci"]
    assert 0 < float(sci["required_max_delta_diff"]) <= 1e-6 and float(sci["required_max_z_diff"]) <= 1e-4
    # caveats that quote archived numbers are marked (aligned with the caveats); a dedicated change's
    # leave-one-out is its own asset's R^2, its 10-asset effect a tenth of it
    assert ch["add_gz"]["caveat_sources"][0] == "archive" and len(ch["add_gz"]["caveat_sources"]) == len(ch["add_gz"]["caveats"])
    gw = ch["gold_window"]
    assert gw["loo_in_final"] == pytest.approx(gw["dedicated"]["gain"], abs=1e-6)
    assert gw["dedicated"]["score_gain"] == pytest.approx([g / 10 for g in gw["dedicated"]["gain"]], abs=1e-6)
    assert audit["residuals"]["fallback"] == {"gold": [1985, 1991]}


@pytest.mark.skipif(not (config.RESULTS_DIR / "factor_audit.json").exists(), reason="factor_audit not built")
def test_the_built_factor_audit_keeps_the_contract():
    check_contract(jsonio.load(config.RESULTS_DIR / "factor_audit.json"))


def test_the_archive_is_frozen_package_data(archive):
    assert archive["id"] == "research_2026_11_01" and archive["panel_sha1"] == "3869fdde3437"
    assert archive["tag"] == "研究轮存档（2026-11-01，面板 3869fdde3437）"
    root = config.ROOT if hasattr(config, "ROOT") else A.ARCHIVE_PATH.parents[3]
    for doc in archive["docs"]:
        assert (root / doc).exists(), doc
    text = A.ARCHIVE_PATH.read_text(encoding="utf-8")
    assert not re.search(r"api_key=(?!REDACTED)", text)
    assert "/Users/" not in text and "/private/tmp" not in text


def test_an_old_gate_row_read_off_the_incremental_run_is_the_old_gate_run(panel):
    """§2.3 protocols: with two iterations the gate changes only the last decision, so the audit reads the
    old gate's row off the incremental run of the same candidates; it equals a run with the old gate."""
    from scenariolab.analysis import walkforward as wf
    from scenariolab.context import load_panel_with_shocks
    P = load_panel_with_shocks()
    inc = wf.run_walkforward(P, ModelSpec(), dedicated=[], gate="incremental")
    old = wf.run_walkforward(P, ModelSpec(), dedicated=[], gate="base")
    derived = A._protocol_row("old_gate", "x", inc, "base", derived=True)
    direct = A._protocol_row("old_gate", "x", old, "base")
    assert derived == direct
    assert derived["kept"] == "selected" and derived["improved"]          # the old gate adopts iteration 2's changes
    spec_row = A._protocol_row("spec", "x", inc, "incremental")
    assert spec_row["kept"] == "start" and not spec_row["improved"] and spec_row["t3_vs_start"] < 0
