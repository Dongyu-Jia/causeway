import json

import numpy as np
import pytest

from scenariolab import config
from scenariolab.scenarios import calibrate, library, probabilities
from scenariolab.scenarios.schema import validate_spec

REQUIRED_IDS = {"baseline", "agi_boom", "ai_bust", "stagflation", "fiscal_dominance", "deglobalization",
                "oil_war", "dollar_decline", "japanification", "energy_transition", "financial_crisis", "k_shaped"}
needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")


@pytest.fixture(scope="module")
def ctx():
    from scenariolab.context import EngineContext
    return EngineContext.load(selection=False)


# ---------------------------------------------------------------- specs

def test_specs_load_with_required_ids_and_fields():
    specs = library.load_specs()
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids)) and REQUIRED_IDS <= set(ids) and len(ids) >= 12
    for s in specs:
        assert s["name"] and s["summary"] and s["tags"] and s["analogs"]
        assert 4 <= s["narrative"].count("。") <= 8, s["id"]
        assert 5 <= len(s["causal_chain"]) <= 10, s["id"]
        assert 2 <= len(s["phases"]) <= 4, s["id"]
        assert sum(p["months"] for p in s["phases"]) == config.HORIZON_MONTHS, s["id"]
        for p in s["phases"]:
            assert all(not v["var"].startswith(("r_", "p_", "d_")) for v in p.get("views") or [])
    total = sum(s["prior_prob"] for s in specs)
    assert 0.95 <= total <= 1.05


@needs_panel
def test_drift_on_a_factor_outside_the_transmission_is_dropped_with_a_reason(ctx):
    spec = {"name": "测试", "phases": [{"name": "一", "months": 24, "views": [{"var": "s_cape", "mean": 30.0}],
                                       "overrides": {"factor_drift": {"white_collar": -0.01, "labor_share": -0.01,
                                                                      "no_such": 0.1}}}]}
    clean, report = validate_spec(spec, ctx.db.frame)
    assert clean["phases"][0]["overrides"]["factor_drift"] == {"labor_share": -0.01}
    text = "\n".join(report["warnings"])
    assert "white_collar 不在今天的传导模型里" in text and "未知因子漂移 no_such" in text
    assert "s_cape 部分由资产价格本身构成（内生）" in text          # a level view on an endogenous state


@needs_panel
def test_a_calibration_block_outside_the_library_warns_that_it_is_ignored(ctx):
    """Only the built-in library resolves `calibration`; a custom or LLM spec that carries one loses it."""
    raw = next(s for s in library.load_specs() if s["id"] == "k_shaped")
    assert raw.get("calibration")
    clean, report = validate_spec(raw, ctx.db.frame)               # as the lab or the API would receive it
    assert "calibration" not in clean and report["ok"]
    hits = [w for w in report["warnings"] if "calibration" in w]
    assert len(hits) == 1 and "已忽略" in hits[0] and "内置情景库" in hits[0]
    resolved, _ = library.resolve_calibration(raw, library.calibration_base(ctx))   # the library path: no warning
    assert not any("calibration" in w for w in validate_spec(resolved, ctx.db.frame)[1]["warnings"])
    for empty in (None, [], {}):
        assert not any("calibration" in w for w in validate_spec({**resolved, "calibration": empty}, ctx.db.frame)[1]["warnings"])


@needs_panel
def test_every_spec_validates_without_errors_or_clipping(ctx):
    base = library.calibration_base(ctx)
    for raw in library.load_specs():
        spec, _ = library.resolve_calibration(raw, base)
        clean, report = validate_spec(spec, ctx.db.frame)
        assert report["ok"] and not report["errors"], (raw["id"], report["errors"])
        assert not report["clipped"], (raw["id"], report["clipped"])
        assert not report["warnings"], (raw["id"], report["warnings"])
        assert len(clean["phases"]) == len(raw["phases"])


@needs_panel
def test_views_hold_for_the_months_the_sampler_draws(ctx):
    """Every mean view must hold for the months the block sampler draws, not just under EP.

    Pooling single months lets a view on a fat-tailed monthly change be met through a few spike
    months (1980 rate swings, 2020-04 unemployment) whose neighbours inside the same 6-month
    block reverse it. Pooling block starts makes the expected drawn mean equal the target.
    """
    from scenariolab.scenarios.runner import phase_pools
    base = library.calibration_base(ctx)
    F = ctx.db.frame
    for raw in library.load_specs():
        clean, _ = validate_spec(library.resolve_calibration(raw, base)[0], F)
        for ph, pool in zip(clean["phases"], phase_pools(ctx, clean, [])):
            for v in (pool.ep.views if pool.ep else []):
                if v.get("kind") != "mean":
                    continue
                x = F[v["var"]].to_numpy(float)
                x = np.where(np.isfinite(x), x, np.nanmean(x))
                drawn = float(pool.month_weights @ x)
                c = ph["confidence"]                  # < 1 blends the posterior back toward the prior
                expected = c * v["target_used"] + (1.0 - c) * v["prior_mean"]
                scale = max(abs(v["target"] - v["prior_mean"]), v["prior_sd"])
                assert abs(drawn - expected) < 1e-3 * scale, (raw["id"], ph["name"], v["var"])
                if pool.shrink == 1.0:
                    assert v["target_used"] == pytest.approx(v["target"]), (raw["id"], ph["name"], v["var"])


@needs_panel
def test_schema_examples_match_llm_schema_and_round_trip(ctx):
    from scenariolab.scenarios.schema import SPEC_JSON_SCHEMA
    phase_keys = SPEC_JSON_SCHEMA["properties"]["phases"]["items"]["required"]
    view_keys = SPEC_JSON_SCHEMA["properties"]["phases"]["items"]["properties"]["views"]["items"]["required"]
    resolved = {s["id"]: s for s in library.resolved_specs(ctx)}
    for ex in library.schema_examples(ctx):
        assert set(ex) == set(SPEC_JSON_SCHEMA["required"])
        for ph in ex["phases"]:
            assert set(ph) == set(phase_keys)
            assert set(ph["overrides"]) == {"factor_drift", "asset_drift", "resid_scale", "r_star"}
            assert all(set(v) == set(view_keys) for v in ph["views"])
        a, _ = validate_spec(ex, ctx.db.frame)
        b, _ = validate_spec(resolved[ex["id"]], ctx.db.frame)
        assert [(p["views"], p["overrides"]) for p in a["phases"]] == [(p["views"], p["overrides"]) for p in b["phases"]]


def test_calibration_merges_views_and_drifts():
    raw = next(s for s in library.load_specs() if s["id"] == "k_shaped")
    spec, cals = library.resolve_calibration(raw, {"ip_yoy": 1.5, "services_yoy": 3.0})
    assert "calibration" not in spec and len(cals) == 2
    ph = spec["phases"][cals[0]["phase"]]
    names = [v["var"] for v in ph["views"]]
    assert {"s_ip_yoy", "s_services_yoy", "f_unrate"} <= set(names) and len(names) == len(set(names))
    assert ph["overrides"]["factor_drift"]["unrate"] > 0 > ph["overrides"]["factor_drift"]["labor_share"]


# ---------------------------------------------------------------- calculator

def test_zero_shock_gives_zero_impact():
    views, expl = calibrate.distribution_views({}, {"ip_yoy": 2.0, "services_yoy": 3.0})
    imp = expl["impact"]
    for key in ("consumption_change", "real_gdp_change", "labor_income_change", "labor_share_log_change",
                "unemployment_change_pp"):
        assert abs(imp[key]) < 1e-12, key
    targets = {v["var"]: v.get("mean", v.get("mean_annual")) for v in views}
    assert targets == {"s_ip_yoy": 2.0, "s_services_yoy": 3.0, "f_unrate": 0.0, "f_labor_share": 0.0}
    assert all(abs(x) < 1e-12 for x in expl["overrides"]["factor_drift"].values())


def test_top10_shock_scales_with_its_consumption_share():
    shock = {"labor_income_shock": {"top10": -0.10}}
    a = calibrate.distribution_impact({**shock, "consumption_share": {"top10": 0.30}})
    b = calibrate.distribution_impact({**shock, "consumption_share": {"top10": 0.60}})
    assert a["consumption_change"] < 0
    assert abs(b["consumption_change"] / a["consumption_change"] - 2.0) < 1e-9
    top = next(g for g in a["groups"] if g["id"] == "top10")
    assert abs(top["contribution_labor"] - 0.30 * 0.2 * -0.10 * 0.57) < 1e-12
    # doubling the shock doubles the impact; the wealth channel is linear in the equity change
    c = calibrate.distribution_impact({"labor_income_shock": {"top10": -0.20}, "consumption_share": {"top10": 0.30}})
    assert abs(c["consumption_change"] / a["consumption_change"] - 2.0) < 1e-9
    w = calibrate.distribution_impact({"equity_wealth_change": 0.10})
    assert abs(w["consumption_change_wealth"] - 0.03 * 0.10 * 2.7) < 1e-12
    assert abs(sum(g["contribution_wealth"] for g in w["groups"]) - w["consumption_change_wealth"]) < 1e-12


def test_calculator_channels_and_units():
    views, expl = calibrate.distribution_views({"labor_income_shock": {"mid40": -0.05}, "months": 48})
    imp = expl["impact"]
    assert imp["unemployment_direct_pp"] == pytest.approx(100 * 0.4 * 0.5 * 0.05)
    assert imp["real_gdp_change"] == pytest.approx(0.68 * imp["consumption_change"])
    drift = expl["overrides"]["factor_drift"]
    assert drift["unrate"] == pytest.approx(imp["unemployment_direct_pp"] / 4.0, abs=1e-4)
    assert drift["labor_share"] == pytest.approx(imp["labor_share_log_change"] / 4.0, abs=1e-5)
    assert imp["labor_share_log_change"] < 0 and expl["text"]
    # the Okun part becomes the drawn months' unemployment trend (pp per year), the direct part a drift
    okun = next(v for v in views if v["var"] == "f_unrate")
    assert okun["mean_annual"] == pytest.approx(imp["unemployment_okun_pp"] / 4.0, abs=1e-4) and okun["mean_annual"] > 0
    assert 0.3 <= okun["vol_mult"] < 1.0
    # the labour-share shift is the drift alone: the drawn months carry no labour-share trend of their own
    share = next(v for v in views if v["var"] == "f_labor_share")
    assert share["mean_annual"] == 0.0 and "vol_mult" not in share


# ---------------------------------------------------------------- probabilities

def test_posterior_is_positive_normalized_and_tempered():
    today = {"a": 1.0, "b": 2.0}
    scales = {"a": 1.0, "b": 1.0}
    items = [{"id": "near", "prior": 0.2, "means": {"a": 1.0, "b": 2.0}},
             {"id": "far", "prior": 0.6, "means": {"a": 3.0, "b": 0.0}},
             {"id": "mid", "prior": 0.4, "means": {"a": 2.0, "b": 2.0}}]
    out = {o["id"]: o for o in probabilities.posterior(items, today, scales, gamma=0.5)}
    post = np.array([o["posterior"] for o in out.values()])
    assert (post > 0).all() and abs(post.sum() - 1) < 1e-12
    assert abs(sum(o["prior"] for o in out.values()) - 1) < 1e-12
    assert out["near"]["likelihood"] == pytest.approx(1.0)
    assert out["far"]["likelihood"] == pytest.approx(np.exp(-0.5 * 4.0))
    ratio = (out["far"]["posterior"] / out["near"]["posterior"]) / (out["far"]["prior"] / out["near"]["prior"])
    assert ratio == pytest.approx(np.exp(-0.5 * 4.0) ** 0.5)
    flat = probabilities.posterior(items, today, scales, gamma=0.0)
    assert [o["posterior"] for o in flat] == pytest.approx([o["prior"] for o in flat])


@needs_panel
def test_probabilities_on_real_specs(ctx):
    specs = [validate_spec(library.resolve_calibration(s, library.calibration_base(ctx))[0], ctx.db.frame)[0]
             for s in library.load_specs()]
    res = probabilities.run(ctx, specs)
    post = np.array([o["posterior"] for o in res["scenarios"]])
    assert (post > 0).all() and abs(post.sum() - 1) < 1e-9
    assert set(res["today"]) == set(probabilities.KEY_STATES) and res["method"]
    by = {o["id"]: o for o in res["scenarios"]}
    assert by["baseline"]["likelihood"] > by["financial_crisis"]["likelihood"]


# ---------------------------------------------------------------- library run

@needs_panel
def test_library_run_writes_files_with_contract_shapes(ctx, tmp_path):
    ids = ["baseline", "k_shaped", "oil_war"]
    index = library.run(ctx, n_paths=40, out_dir=tmp_path, ids=ids)
    T, A = config.HORIZON_MONTHS, len(config.ASSET_IDS)
    assert {s["id"] for s in index["scenarios"]} == set(ids)
    assert index["horizon_months"] == T and index["n_paths"] == 40 and index["probability_method"]
    probs = [s["prob"] for s in index["scenarios"]]
    assert probs == sorted(probs, reverse=True) and abs(sum(probs) - 1) < 1e-9
    for s in index["scenarios"]:
        assert {"us_equity_cagr_p50", "us_equity_real_cagr_p50", "cpi_avg", "ff_end_p50"} <= set(s["headline"])
        assert s["ens_min"] > 0 and s["source"] == "builtin"
    on_disk = json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8"))
    assert on_disk["start"] == index["start"] and len(on_disk["scenarios"]) == 3
    # every index entry carries each asset's stats (the UI's cross-scenario comparison reads only the index),
    # identical to the detail file's stats; the older keys stay
    for s in on_disk["scenarios"]:
        assert {"id", "name", "summary", "tags", "source", "prior_prob", "likelihood", "prob", "headline",
                "ens_min", "analogs", "warnings", "calibrated", "assets"} <= set(s)
        assert list(s["assets"]) == config.ASSET_IDS
        d = json.loads((tmp_path / "scenarios" / f"{s['id']}.json").read_text(encoding="utf-8"))
        for a, st in s["assets"].items():
            assert set(st) == set(library.INDEX_ASSET_STATS)
            assert all(isinstance(st[k], float) for k in library.INDEX_ASSET_STATS), (s["id"], a)
            assert st == {k: d["assets"][a]["stats"][k] for k in library.INDEX_ASSET_STATS}
        assert s["assets"]["us_equity"]["cagr_p50"] == pytest.approx(s["headline"]["us_equity_cagr_p50"])
        eq = s["assets"]["us_equity"]
        assert eq["cagr_p5"] <= eq["cagr_p50"] <= eq["cagr_p95"] and eq["max_dd_p50"] <= 0
    detail = json.loads((tmp_path / "scenarios" / "k_shaped.json").read_text(encoding="utf-8"))
    assert detail["prob"] == pytest.approx(detail["probability"]["posterior"])
    cal = detail["calibration"]
    assert isinstance(cal, list) and cal[0]["engine_check"]["us_equity_total_return_p50"] is not None
    check = cal[0]["engine_check"]
    assert check["us_equity_price_return_p50"] < check["us_equity_total_return_p50"]   # dividends are not wealth
    assert len(detail["labels"]) == T + 1
    z = np.load(tmp_path / "scenario_paths.npz")
    S = len(ids)
    assert z["total"].shape == (S, 40, T, A) and z["total"].dtype == np.float32
    assert z["income"].shape == (S, 40, T, A) and z["cpi"].shape == (S, 40, T + 1)
    assert list(z["ids"]) == [s["id"] for s in index["scenarios"]]
    assert list(z["assets"]) == config.ASSET_IDS and len(z["labels"]) == T + 1
    assert z["probs"] == pytest.approx(probs)
    assert np.isfinite(z["total"]).all()


@needs_panel
def test_library_run_is_reproducible(ctx, tmp_path):
    a = library.run(ctx, n_paths=30, out_dir=tmp_path / "a", ids=["stagflation"])
    b = library.run(ctx, n_paths=30, out_dir=tmp_path / "b", ids=["stagflation"])
    assert a["scenarios"][0]["headline"] == b["scenarios"][0]["headline"]
    za, zb = np.load(tmp_path / "a" / "scenario_paths.npz"), np.load(tmp_path / "b" / "scenario_paths.npz")
    assert np.array_equal(za["total"], zb["total"]) and np.array_equal(za["cpi"], zb["cpi"])
    # a different base seed gives different paths
    c = library.run(ctx, n_paths=30, out_dir=tmp_path / "c", ids=["stagflation"], base_seed=1)
    zc = np.load(tmp_path / "c" / "scenario_paths.npz")
    assert not np.array_equal(za["total"], zc["total"]) and c["n_paths"] == 30
