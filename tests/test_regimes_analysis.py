"""The regimes build step (analysis/regimes.py -> results/regimes.json): the contract of run(), the check
values of the same-month asset table and the forward-return table, and the shortcuts the step takes (the
right-aligned vintage classification, the NBER scorer) against the slow versions they replace."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.analysis import regimes as A
from scenariolab.engine import regimes as R


def _panel_month() -> str | None:
    try:
        return str(json.loads(config.PANEL_META_PATH.read_text(encoding="utf-8"))["end"])[:7]
    except (OSError, ValueError, KeyError, TypeError):
        return None


needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
# the check values were computed on the panel that ends 2026-09 (as in tests/test_regimes.py)
CHECK_MONTH = "2026-09"
check_values = pytest.mark.skipif(_panel_month() != CHECK_MONTH,
                                  reason=f"check values are for the panel ending {CHECK_MONTH}")

TOP_KEYS = {"version", "data_end", "monthdb", "definitions", "nowcast", "timeline", "counts", "validation", "assets",
            "episodes", "factors", "transitions", "forward", "eras", "transmission", "engine", "model_use", "units",
            "notes"}


@pytest.fixture(scope="module")
def ctx():
    from scenariolab.context import EngineContext
    return EngineContext.load(selection=True)


@pytest.fixture(scope="module")
def result(ctx):
    """run() with the slow studies cut down (a 4-point grid, 20 revision histories, two 12-month regime studies
    and one duration run, no conflict example: its seven infeasible pools take 10-20 s); every other section is
    computed in full. The full-size studies are checked on the built file below."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(A, "GRID", {"ip_swing_pct": (3.0, 4.0), "u_rise_pp": (0.5, 1.0), "u_smooth_months": (3,)})
        mp.setattr(A, "REVISION_PATHS", 20)
        mp.setattr(A, "STUDIES", [s for s in A.STUDIES if s[0] in ("stagflation_08", "contraction_10")])
        mp.setattr(A, "STUDY_MONTHS", (12,))
        mp.setattr(A, "DURATION_MONTHS", (12,))
        mp.setattr(A, "conflict_example", lambda ctx: {"spec": "stagflation", "lambda": None})
        return A.run(ctx)


def _share_dicts(obj, path=""):
    """Every {label id: share} dict anywhere in the result (keys exactly the six labels, values in [0, 1])."""
    if isinstance(obj, dict):
        vals = list(obj.values())
        if set(obj) == set(R.LABEL_IDS) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) \
                and all(0.0 <= v <= 1.0 for v in vals):
            yield path, obj
        for k, v in obj.items():
            yield from _share_dicts(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _share_dicts(v, f"{path}[{i}]")


# ---------------------------------------------------------------- 12. the run() contract

@needs_panel
def test_run_has_every_section_and_the_shapes_the_ui_reads(result, ctx):
    assert TOP_KEYS <= set(result) and result["version"] == 1
    assert result["data_end"] == ctx.panel.index.max().strftime("%Y-%m")
    assert result["monthdb"] == {"first": ctx.db.dates[0].strftime("%Y-%m"), "last": ctx.db.dates[-1].strftime("%Y-%m"),
                                 "months": len(ctx.db)}
    tl = result["timeline"]
    n = len(tl["dates"])
    assert tl["dates"][0] == config.REGIMES["first_month"] and tl["dates"][-1] == result["data_end"]
    assert len(tl["label"]) == len(tl["label_rt"]) == n
    assert all(len(v) == n for v in (*tl["flags"].values(), *tl["series"].values()))
    assert set(tl["flags"]) >= set(R.FLAG_IDS) and {"cpi_yoy", "ip_below_peak_rt", "unrate_3m"} <= set(tl["series"])
    assert len(tl["nber"]) == len(R.NBER_PEAK_TROUGH)
    d = result["definitions"]
    assert [x["id"] for x in d["labels"]] == R.LABEL_IDS and d["precedence"] == R.PRECEDENCE
    assert {x["term"] for x in d["terms"]} == {"通胀", "通缩", "滞胀", "萧条"}
    assert np.array(d["grid"]["cells"]).shape == (3, 3)
    assert set(d["rule_text"]) == {"growth", "severity", "inflation", "precedence", "realtime", "data", "timing"}
    for matrix in (result["counts"]["confusion_1948"], result["engine"]["drawn_vs_sim_crosstab"],
                   result["transitions"]["1973"]["monthly"]["counts"], result["transitions"]["1973"]["rt_to_expost_12m"]["counts"]):
        assert np.array(matrix).shape == (6, 6)
    assert set(result["assets"]["windows"]) == {"1973", "1948"}
    groups = result["assets"]["windows"]["1973"]["groups"]
    assert {f"label:{i}" for i in R.LABEL_IDS} | {"all", "flag:contraction", "flag:high_infl", "flag:cpi_neg"} <= set(groups)
    assert set(groups["all"]["assets"]) == set(config.ASSET_IDS)
    assert set(result["assets"]["windows"]["1948"]["groups"]["all"]["assets"]) == set(A.ASSETS_1948)
    fw = result["forward"]
    assert fw["lag_months"] == 1 and fw["windows"]["1973"]["months_ahead"] == [2, 13]
    assert fw["windows"]["1973"]["base"]["months_ahead"] == [1, 12]
    assert {"exactly_t12", "any_12m", "new_contraction"} <= set(result["transitions"]["walk_forward"])
    eng = result["engine"]
    assert set(eng["reference"]["windows"]) == {"year1", "years2_5", "years6_15", "whole", "m25_180"}
    assert set(eng["support"]) >= set(R.LABEL_IDS) | {"contraction", "high_infl"}
    for ep in result["episodes"]:
        assert len(ep["path"]) == ep["months"] + 1 and ep["path"][0] == pytest.approx(100.0)
    assert result["model_use"]["transmission"]["used"] is False and result["model_use"]["forecast"]["used"] is False
    assert result["model_use"]["scenarios"]["used"] is True
    json.dumps(result, allow_nan=True)                      # plain JSON types throughout


@needs_panel
def test_every_share_dict_sums_to_one(result):
    found = list(_share_dicts(result))
    assert len(found) >= 30, len(found)
    for path, d in found:
        assert sum(d.values()) == pytest.approx(1.0, abs=1e-9), path
    for w in ("1948", "1973"):
        for ver in ("expost", "realtime"):
            labels = result["counts"][w][ver]["labels"]
            assert sum(v["share"] for v in labels.values()) == pytest.approx(1.0)
            assert sum(v["months"] for v in labels.values()) == result["counts"][w][ver]["months"]
        for row in result["transitions"][w]["rt_to_expost_12m"]["probs"]:
            if all(isinstance(p, float) and math.isfinite(p) for p in row):
                assert sum(row) == pytest.approx(1.0)


def test_the_forecast_test_counts_only_for_the_same_data_month(tmp_path, monkeypatch):
    """forward.forecast_test and the last note state the forecast test's verdict only when regime_forecast.json
    was computed on this build's data month (that step runs after regimes)."""
    from scenariolab import jsonio
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    assert A.forecast_test_status("2026-09") == {"available": False, "file": "regime_forecast.json",
                                                 "data_end": None, "adopted": None}
    jsonio.dump({"verdict": {"adopted": []}}, tmp_path / "regime_forecast.json")            # older file: no month
    assert A.forecast_test_status("2026-09")["available"] is False
    jsonio.dump({"data_end": "2026-08", "verdict": {"adopted": []}}, tmp_path / "regime_forecast.json")
    assert A.forecast_test_status("2026-09") == {"available": False, "file": "regime_forecast.json",
                                                 "data_end": "2026-08", "adopted": None}
    jsonio.dump({"data_end": "2026-09", "verdict": {"adopted": []}}, tmp_path / "regime_forecast.json")
    st = A.forecast_test_status("2026-09")
    assert st["available"] is True and st["adopted"] == []
    tr = {"tested": True}
    assert A.model_use_note({"transmission": tr, "forecast": {"tested": True, "adopted": []}}) \
        == "体制不进入传导模型和一年期预测：两者都做过样本外检验，没有通过。"
    stale = A.model_use_note({"transmission": tr, "forecast": {"tested": False, "adopted": None}})
    assert "传导模型的样本外检验没有通过" in stale and "一年期预测的样本外检验没有通过" not in stale
    assert "没有与当前数据对应的检验结果" in stale


@needs_panel
def test_no_code_ids_in_the_chinese_texts(result):
    """UI prose names things in Chinese: no snake_case ids, file names or function names."""
    d = result["definitions"]
    texts = [x["text"] for x in d["terms"]] + [x["why"] for x in d["names_changed"]] + list(d["rule_text"].values())
    texts += [c["text"] for c in d["conventions"].values()] + [x["how"] for x in d["interpolations"]]
    texts += [v["reason"] for v in result["model_use"].values()] + result["notes"] + [result["units"]]
    texts += [s["desc"] for s in result["transmission"]["splits"]] + [e["name"] for e in result["eras"]]
    for t in texts:
        assert not any(tok in t for tok in ("_", ".json", ".py", "()")), t


@needs_panel
def test_thin_cells_get_no_placebo_p_and_survival_needs_enough_episodes(result):
    for w in ("1973", "1948"):
        blk = result["forward"]["windows"][w]
        for variant in (blk, blk["base"]):
            for g in variant["groups"].values():
                for c in g["assets"].values():
                    assert c["thin"] == (c["episodes"] < R.THIN)
                    if c["thin"]:
                        assert c.get("placebo_p") is None and not c.get("survives") and not c.get("robust")
        for g in result["assets"]["windows"][w]["groups"].values():
            for c in g["assets"].values():
                if c["episodes"] < 2:
                    assert c["ann_real"]["ci90"] is None
                if (c.get("vs_rest") or {}).get("survives"):
                    assert c["episodes"] >= R.THIN


# ---------------------------------------------------------------- 12. check values (DESIGN §4.1)

# same-month real returns, 1973 window: (annual real return, difference to the other months); geometric
ASSET_POINTS = {
    ("all", "us_equity"): (0.07015, None), ("all", "nasdaq"): (0.09952, None), ("all", "treasury_10y"): (0.02113, None),
    ("all", "gold"): (0.04054, None), ("all", "crude_oil"): (0.02646, None), ("all", "real_estate"): (0.06676, None),
    ("all", "energy"): (0.06935, None), ("all", "cash"): (0.00549, None),
    ("label:expansion", "us_equity"): (0.13249, 0.15487), ("label:expansion", "nasdaq"): (0.18251, 0.2036),
    ("label:expansion", "treasury_10y"): (0.03167, 0.02714), ("label:expansion", "real_estate"): (0.1127, 0.11538),
    ("label:expansion", "energy"): (0.09994, 0.0777),
    ("label:inflation", "us_equity"): (-0.02346, -0.12732), ("label:inflation", "nasdaq"): (-0.02219, -0.1663),
    ("label:inflation", "treasury_10y"): (-0.04064, -0.0836), ("label:inflation", "crude_oil"): (0.08464, 0.07718),
    ("label:inflation", "real_estate"): (0.01049, -0.07605), ("label:inflation", "energy"): (0.10531, 0.04788),
    ("label:deflation", "us_equity"): (0.22784, 0.15954), ("label:deflation", "crude_oil"): (0.34946, 0.32653),
    ("label:recession", "us_equity"): (-0.12603, -0.2003), ("label:recession", "nasdaq"): (-0.33577, -0.44586),
    ("label:depression", "us_equity"): (-0.22607, -0.30738), ("label:depression", "crude_oil"): (-0.50367, -0.55432),
    ("label:stagflation", "us_equity"): (0.07747, 0.00787), ("label:stagflation", "treasury_10y"): (0.12922, 0.11577),
    ("label:stagflation", "energy"): (-0.10306, -0.18662), ("label:stagflation", "cash"): (0.02114, 0.01682),
    ("flag:contraction", "us_equity"): (-0.043, -0.12951), ("flag:contraction", "treasury_10y"): (0.10262, 0.09208),
    ("flag:contraction", "energy"): (-0.154, -0.25791), ("flag:high_infl", "us_equity"): (-0.01843, -0.1356),
    ("flag:high_infl", "nasdaq"): (-0.01732, -0.18007), ("flag:cpi_neg", "us_equity"): (0.29821, 0.23265),
    ("flag:cpi_neg", "real_estate"): (0.49946, 0.44074),
}
# 90% episode-bootstrap bounds (annual real return, difference to the rest) of the cells the tab prints
ASSET_CI = {
    ("label:expansion", "us_equity"): ([0.09752, 0.17617], [0.09889, 0.24149]),
    ("label:expansion", "treasury_10y"): ([0.00146, 0.05195], [-0.03493, 0.06227]),
    ("label:inflation", "us_equity"): ([-0.09121, 0.01667], [-0.21666, -0.07854]),
    ("label:inflation", "treasury_10y"): ([-0.08058, 0.01513], [-0.13804, -0.02002]),
    ("label:stagflation", "us_equity"): ([-0.04382, 0.33474], [-0.1052, 0.26512]),
    ("label:stagflation", "treasury_10y"): ([-0.02591, 0.32836], [-0.04072, 0.32849]),
    ("label:stagflation", "energy"): ([-0.15141, -0.04906], [-0.25318, -0.1205]),
    ("flag:contraction", "us_equity"): ([-0.14816, 0.11614], [-0.23432, 0.03299]),
    ("flag:contraction", "treasury_10y"): ([0.00527, 0.23856], [-0.00876, 0.23221]),
}
# the year after a real-time label (t+2..t+13), US equities, 1973 window:
# (origins, episodes, mean, diff vs all origins, placebo p, HAC t, 90% interval of the diff)
FORWARD = {
    "label:expansion": (382, 14, 0.11272, 0.02414, 0.0908, 1.80, [0.00313, 0.04351]),
    "label:inflation": (178, 13, 0.02157, -0.06702, 0.0, -2.61, [-0.12438, -0.02067]),
    "label:stagflation": (42, 5, 0.18823, 0.09965, 0.0291, 1.77, [-0.00325, 0.17136]),
    "flag:high_infl": (213, 11, 0.05196, -0.03663, 0.0993, -1.47, [-0.08025, 0.00322]),
    "flag:contraction_rt": (65, 7, 0.13597, 0.04738, 0.3219, 0.75, [-0.0816, 0.14795]),
    "label:recession": (15, 2, -0.05509, -0.14367, None, -1.13, None),
    "label:depression": (8, 2, 0.21981, 0.13122, None, 2.20, None),
    "label:deflation": (6, 2, 0.02679, -0.0618, None, -1.12, None),
}


@needs_panel
@check_values
def test_same_month_asset_table(result):
    groups = result["assets"]["windows"]["1973"]["groups"]
    assert groups["all"]["months"] == 644 and groups["label:expansion"]["episodes"] == 15
    for (gid, a), (real, vs_rest) in ASSET_POINTS.items():
        cell = groups[gid]["assets"][a]
        assert cell["ann_real"]["value"] == pytest.approx(real, abs=5e-4), (gid, a)
        if vs_rest is not None:
            assert cell["vs_rest"]["value"] == pytest.approx(vs_rest, abs=5e-4), (gid, a)
    for (gid, a), (ci, ci_rest) in ASSET_CI.items():
        cell = groups[gid]["assets"][a]
        assert cell["ann_real"]["ci90"] == pytest.approx(ci, abs=0.015), (gid, a)
        assert cell["vs_rest"]["ci90"] == pytest.approx(ci_rest, abs=0.015), (gid, a)
    assert groups["label:stagflation"]["survivors"] == ["energy"]
    assert set(groups["flag:contraction"]["survivors"]) == {"energy", "industrials"}
    w48 = result["assets"]["windows"]["1948"]["groups"]
    assert w48["label:deflation"]["assets"]["us_equity"]["ann_real"]["value"] == pytest.approx(0.31014, abs=5e-4)
    sb = groups["label:stagflation"]["corr_stock_bond"]
    assert sb["value"] == pytest.approx(0.2988, abs=5e-4) and sb["ci90"] == pytest.approx([0.0872, 0.5232], abs=0.03)
    assert groups["label:deflation"]["corr_stock_bond"]["value"] is None          # 8 months: not reported


@needs_panel
@check_values
def test_forward_table_placebo_and_hac(result):
    blk = result["forward"]["windows"]["1973"]
    assert blk["groups"]["all"]["assets"]["us_equity"]["mean"] == pytest.approx(0.08859, abs=5e-4)
    for gid, (origins, episodes, mean, diff, p, t, ci) in FORWARD.items():
        c = blk["groups"][gid]["assets"]["us_equity"]
        assert (c["origins"], c["episodes"]) == (origins, episodes), gid
        assert c["mean"] == pytest.approx(mean, abs=5e-4) and c["diff"] == pytest.approx(diff, abs=5e-4), gid
        assert c["t_hac"] == pytest.approx(t, abs=0.005 + 1e-9), gid
        if p is None:
            assert c["placebo_p"] is None, gid
        else:
            assert c["placebo_p"] == pytest.approx(p, abs=1e-4), gid
            assert c["ci90"] == pytest.approx(ci, abs=0.015), gid
    assert blk["global_placebo"]["tests"] == 130 and blk["global_placebo"]["actual"] == 22
    assert blk["global_placebo"]["p"] == pytest.approx(0.1695, abs=1e-4)
    assert blk["eligible_tests"] == 65 and blk["holm_discoveries"] == 0
    assert blk["bh10_discoveries"] == ["flag:high_infl|industrials", "label:inflation|industrials",
                                       "label:inflation|us_equity", "label:stagflation|staples"]
    assert abs(blk["n_survive"] - 16) <= 2 and abs(blk["n_robust"] - 13) <= 2          # Monte Carlo: +-2
    m = result["forward"]["mechanism"]
    assert (m["origins"], m["within_12m_of_contraction"], m["all_origins"]) == (178, 71, 631)
    assert m["mean_within"] == pytest.approx(-0.024, abs=5e-4) and m["not_minus_uncond"] == pytest.approx(-0.0368, abs=5e-4)


@needs_panel
@check_values
def test_counts_episodes_transitions_and_walk_forward(result):
    c = result["counts"]
    assert c["1948"]["expost"]["labels"]["inflation"] == {"months": 214, "share": pytest.approx(0.2265, abs=5e-5), "runs": 18}
    assert c["1973"]["realtime"]["labels"]["stagflation"]["months"] == 42
    assert c["agreement_rt_expost"] == {"1948": pytest.approx(0.8942, abs=5e-5), "1973": pytest.approx(0.9053, abs=5e-5)}
    assert c["1948"]["inflation_falling"] == {"months_with_trend": 207, "falling": 59, "first_month": "1949-01"}
    why = next(x["why"] for x in result["definitions"]["names_changed"] if x["to"] == "通胀")
    assert why.startswith("1949 年以来 207 个通胀月份里有 59 个月")
    eps = result["episodes"]
    assert len(eps) == 12 and eps[5]["start"] == "1973-12" and eps[5]["label"] == "stagflation"
    assert round(100 * eps[5]["equity_real_to_low"], 1) == -39.1 and round(100 * eps[5]["equity_real_after_low"], 1) == 43.2
    assert result["episodes_summary"]["stagflation_after_low_1973"] == {"after": 22, "of": 45, "episodes": 4}
    t73 = result["transitions"]["1973"]
    assert t73["p_contraction_12m"]["by_label"]["inflation"]["p"] == pytest.approx(0.2584, abs=5e-5)
    assert t73["inflation_to_stagflation_hits"]["hits"] == 40
    wf = result["transitions"]["walk_forward"]
    assert wf["exactly_t12"]["all"]["origins"] == 632
    assert wf["exactly_t12"]["all"]["label_brier_skill"] == pytest.approx(0.1547, abs=5e-5)
    assert wf["any_12m"]["all"]["rt_label"]["brier_skill"] == pytest.approx(0.103, abs=5e-4)
    assert wf["new_contraction"]["all"]["origins"] == 567
    assert wf["new_contraction"]["2000_end"]["rt_label"]["brier_skill"] == pytest.approx(-0.026, abs=5e-4)
    assert wf["new_contraction"]["all"]["rt_label"]["t_hac"] == pytest.approx(0.60, abs=0.005)


@needs_panel
@check_values
def test_validation_engine_and_definitions(result):
    v = result["validation"]
    assert v["expost"]["balanced_accuracy"] == pytest.approx(0.960, abs=5e-4) and v["expost"]["false_alarms"] == []
    assert v["realtime"]["false_alarms"] == [["1959-11", "1959-11"]]
    alt = {a["id"]: a for a in v["alternatives"]}
    assert alt["ip_only"]["ba"] == pytest.approx(0.922, abs=5e-4) and alt["ip_only"]["fa_runs"] == 5
    assert alt["sahm"]["months_after_window"] == 6 and alt["prototype"]["months_after_window"] == 13
    assert alt["slack"]["ba"] == pytest.approx(0.587, abs=5e-4) and alt["slack"]["detected"] == 8
    assert len(v["legs"]) == 18 and v["legs_margin"]["largest_u_rise_without_recession"] == pytest.approx(0.40)
    assert v["near_miss_2022_24"]["ip_peak"] == "2022-04" and v["near_miss_2022_24"]["sahm_ge_05"] == ["2024-07", "2024-08", "2024-09"]
    e = result["engine"]
    assert e["support"]["recession"]["max_delivered"] == pytest.approx(0.3733, abs=5e-5)
    assert e["support"]["depression"]["reason"] == "thin" and e["support"]["depression"]["thin"]
    ref = e["reference"]
    assert (ref["n_paths"], ref["seed"]) == (2000, 4242)
    assert ref["windows"]["year1"]["paths_with_contraction"] == pytest.approx(0.1915, abs=5e-5)
    assert ref["cpi_gt4_share"] == pytest.approx(0.4077, abs=5e-5) and ref["anchor"] == {
        "label_rt": "expansion", "label_expost_mix": {"expansion": 1.0}}
    assert e["history"]["contraction_episodes"]["episodes"] == 7
    assert e["simulated_inflation_from_expansion_share"] == pytest.approx(60778 / 121923)
    s = e["studies"]["drawn_vs_simulated"][0]
    assert (s["case"], s["months"]) == ("stagflation_08", 12)
    assert s["drawn_share"] == pytest.approx(0.7943, abs=5e-5) and s["simulated_share"] == pytest.approx(0.5969, abs=5e-5)
    d = e["studies"]["duration"][0]
    assert d["paths_ip_fall_beyond_hist"] == pytest.approx(0.06) and d["ip_change_p50"] == pytest.approx(-8.853, abs=5e-4)
    terms = {t["term"]: t["text"] for t in result["definitions"]["terms"]}
    assert "280 个月" in terms["通胀"] and "214 个月叫通胀" in terms["通胀"] and "+31.0%" in terms["通缩"]
    assert "共 5 段、56 个月" in terms["滞胀"] and "−17.2%" in terms["萧条"] and "−18.8%" in terms["萧条"]
    interp = result["definitions"]["interpolations"]
    if interp:                                               # read from the cached FRED files (absent in a fresh clone)
        assert interp[0]["month"] == "2025-10" and {"cpi", "unrate"} <= set(interp[0]["series"])


# ---------------------------------------------------------------- the shortcuts against the slow versions

@needs_panel
def test_vintages_equal_the_classifier_on_the_cut_panel(ctx):
    panel = ctx.panel
    cpi, ur, ip = (panel[c].to_numpy(float) for c in R.INPUTS.values())
    T = len(ip)
    t0 = int(np.flatnonzero(panel.index >= pd.Timestamp("1973-02"))[0])
    vint = A.vintages(panel, t0, T - 13)
    rng = np.random.default_rng(5)
    picks = sorted({t0, T - 13, *rng.integers(t0, T - 12, 25).tolist(),
                    *(int(np.flatnonzero(panel.index == pd.Timestamp(m))[0]) for m in ("2008-06", "2009-03", "2020-03"))})
    for t in picks:
        r = R.classify(cpi[: t + 1], ur[: t + 1], ip[: t + 1])
        assert np.array_equal(vint[t][0], r["label"]) and np.array_equal(vint[t][1], r["contraction"]), panel.index[t]


@needs_panel
def test_the_nber_scorer_agrees_with_the_classifier_validation(ctx):
    lab = R.classify_panel(ctx.panel)
    v = R.validate_nber(ctx.panel)
    for ver, col in (("expost", "contraction"), ("realtime", "contraction_rt")):
        s = A.nber_score((lab[col] == 1).to_numpy(), ctx.panel.index)
        for k in ("balanced_accuracy", "tpr", "tnr", "precision", "detected", "recessions", "mean_start_err",
                  "mean_end_err", "false_alarms", "per_recession"):
            assert s[k] == v[ver][k], (ver, k)


def test_zigzag_down_legs_and_the_years_label():
    x = np.array([0, 1, 2, 5, 4, 2, 1, 3, 4, 6, 3, 2.5, np.nan, 2.0])
    assert A._zigzag(x, 2.5) == [(0, -1), (3, 1), (6, -1), (9, 1)]
    legs = A._down_legs(x, 2.5)
    assert legs.tolist() == [False] * 4 + [True] * 3 + [False] * 3 + [True] * 4    # the last leg runs to the end
    assert A._years("1973-12", "1975-05") == "1973–75" and A._years("1970-01", "1970-11") == "1970"
    assert A._years("1999-06", "2001-02") == "1999–2001"


def test_the_bootstrap_draws_whole_episodes_and_needs_ten_finite_replicates():
    rng = np.random.default_rng(1)
    C = A._boot(rng, 5)
    assert C.shape == (A.B, 5) and (C.sum(axis=1) == 5).all()
    assert A._boot(rng, 0).shape == (A.B, 0)
    assert A._ci([np.nan] * 20 + [1.0] * 9) is None and A._ci(np.arange(100.0)) == pytest.approx([4.95, 94.05])


# ---------------------------------------------------------------- the built file (the slow studies at full size)

def _built() -> dict | None:
    path = config.RESULTS_DIR / "regimes.json"
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if d.get("data_end") == CHECK_MONTH and d.get("version") == 1 else None


@pytest.mark.skipif(_built() is None, reason=f"results/regimes.json for {CHECK_MONTH} not built")
def test_the_built_file_holds_the_out_of_sample_grid_revisions_and_studies():
    d = _built()
    oos = d["validation"]["out_of_sample"]
    assert oos["grid_size"] == 270 and oos["grid_points_ba_ge_0955"] == 88
    loro = oos["loro"]
    assert loro["cv_ba_worst_tie"] == pytest.approx(0.9089, abs=5e-5) and loro["cv_ba_best_tie"] == pytest.approx(0.9331, abs=5e-5)
    assert loro["insample_ba_same_windows"] == pytest.approx(0.9412, abs=5e-5)
    assert loro["misses"] == [{"recession": "1990-08..1991-03", "tied_settings": 25, "missing": 9}]
    split = oos["split"]["1948_1972_to_1973_2020"]["test_ba"]
    assert split == pytest.approx([0.9255, 0.966], abs=5e-5)
    rev = {c["id"]: c for c in d["validation"]["revisions"]["cases"]}
    assert d["validation"]["revisions"]["paths"] == 300
    # the critique's own draws are reproduced (same seeds): within its +-3 pp tolerance in any case
    assert [rev[k]["rt_kept_1948"][1] for k in ("cpi_unrate", "ip_mild", "ip_large")] == pytest.approx([0.992, 0.982, 0.967], abs=0.03)
    assert rev["ip_large"]["spots"]["2022_24"]["realtime"] == pytest.approx(0.48, abs=0.03)
    rows = {(r["case"], r["months"]): r for r in d["engine"]["studies"]["drawn_vs_simulated"]}
    assert len(rows) == 16
    assert rows[("cpi_falling", 48)]["simulated_share"] == pytest.approx(0.6198, abs=5e-5)
    assert rows[("cpi_falling", 48)]["unrate_end_p50"] == pytest.approx(2.28, abs=0.005)
    assert rows[("depression_max", 12)]["ip_change_p50"] == pytest.approx(-9.80, abs=0.005)
    dur = {r["months"]: r["paths_ip_fall_beyond_hist"] for r in d["engine"]["studies"]["duration"]}
    assert dur == pytest.approx({6: 0.0, 12: 0.06, 18: 1 / 6, 24: 0.46, 36: 0.9467, 48: 0.9933}, abs=5e-4)
    cf = d["engine"]["studies"]["conflict_example"]
    assert cf["lambda"] == pytest.approx(0.65, abs=0.005) and cf["lambda_alone"] == 1.0
    assert cf["drawn_share"] == pytest.approx(0.545, abs=5e-4) and cf["culprits"] == ["s_real_ff", "s_ip_yoy"]
    loo = {x["var"]: x["lambda_without"] for x in cf["leave_one_out"]}
    assert loo["s_real_ff"] == pytest.approx(0.97, abs=0.005) and loo["s_ip_yoy"] == pytest.approx(0.88, abs=0.005)
