import json

import numpy as np
import pytest

from scenariolab import config, jsonio
from scenariolab.engine import regimes as R
from scenariolab.scenarios import calibrate, library, probabilities
from scenariolab.scenarios import regime_views as rv
from scenariolab.scenarios.schema import validate_spec

REQUIRED_IDS = {"baseline", "agi_boom", "ai_bust", "stagflation", "fiscal_dominance", "deglobalization",
                "oil_war", "dollar_decline", "japanification", "energy_transition", "financial_crisis", "k_shaped"}
needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")


def _panel_month() -> str | None:
    try:
        return str(json.loads(config.PANEL_META_PATH.read_text(encoding="utf-8"))["end"])[:7]
    except (OSError, ValueError, KeyError, TypeError):
        return None


# exact check values (DESIGN §3.3-3.5) were computed on the panel that ends 2026-09
CHECK_MONTH = "2026-09"
check_values = pytest.mark.skipif(_panel_month() != CHECK_MONTH,
                                  reason=f"check values are for the panel ending {CHECK_MONTH}")


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


# ---------------------------------------------------------------- macro regimes: the regime key (14)

W1_TAIL = "已忽略（可用：扩张、通胀、物价负增长、衰退、深度衰退、滞胀、收缩、高通胀）"


def test_regime_key_forms_aliases_and_rescaling():
    as_dict, w1, c1 = rv.normalize_regime({"滞胀": 0.7, "过热": 0.2}, 1)
    as_list, w2, c2 = rv.normalize_regime([{"id": "stagflation", "share": 0.7, "why": "叙事"},
                                           {"id": "inflation", "share": 0.2, "why": "叙事"}], 1)
    assert as_dict == as_list == {"stagflation": 0.7, "inflation": 0.2} and not (w1 or w2 or c1 or c2)
    aliases = {"过热": "inflation", "萧条": "depression", "通缩": "deflation", "深度衰退": "depression",
               "物价负增长": "deflation", "金发姑娘": "expansion", "软着陆": "expansion", "收缩": "contraction",
               "高通胀": "high_infl", "Stagflation": "stagflation"}
    for word, rid in aliases.items():
        assert rv.normalize_regime({word: 0.3}, 1)[0] == {rid: 0.3}, word
    out, warns, clipped = rv.normalize_regime({"滞胀": 0.7, "通胀": 0.6, "高通胀": 1.0, "收缩": 1.5,
                                               "bogus": 0.1, "衰退": "abc", "stagflation": 0.1}, 3)
    assert out == pytest.approx({"stagflation": 0.7 / 1.3, "inflation": 0.6 / 1.3, "high_infl": 1.0,
                                 "contraction": 1.0})          # flags stay out of the sum; the first entry wins
    assert warns == [f"阶段 3：未知的宏观体制 bogus，{W1_TAIL}", "阶段 3：宏观体制 衰退 的占比不是数值，已忽略",
                     "阶段 3：六种体制互斥，占比之和 1.30 超过 1，已按比例缩到合计 1"]
    assert clipped == [{"phase": 3, "var": "regime.contraction", "requested": 1.5, "used": 1.0}]
    assert rv.normalize_regime(None, 1) == rv.normalize_regime([], 1) == ({}, [], [])
    views = [{"var": "rg_stagflation", "mean": 0.2}, {"var": "f_oil", "mean_annual": 0.1}]
    assert rv.expand_regime({}, views) is views
    assert rv.expand_regime({"stagflation": 0.7}, views) == [
        {"var": "rg_stagflation", "mean": 0.7, "why": "宏观体制：滞胀占抽中月份的 70%"}, views[1]]


@needs_panel
def test_regime_key_becomes_views_and_revalidates_to_the_same_spec(ctx):
    from scenariolab.engine.pooling import view_support
    F = ctx.db.frame
    phase = {"name": "滞胀期", "months": 12, "regime": {"滞胀": 0.7},
             "views": [{"var": "rg_stagflation", "mean": 0.2}, {"var": "f_oil", "mean_annual": 0.15},
                       {"var": "rg_rt_stagflation", "mean": 0.5}]}
    clean, report = validate_spec({"name": "测试", "phases": [phase]}, F)
    ph = clean["phases"][0]
    assert ph["regime"] == {"stagflation": 0.7} and list(ph) == ["name", "months", "regime", "views", "overrides",
                                                                  "confidence"]
    assert ph["views"] == [{"var": "rg_stagflation", "mean": 0.7, "why": "宏观体制：滞胀占抽中月份的 70%"},
                           {"var": "f_oil", "mean_annual": 0.15}]
    assert ("阶段 1：rg_rt_stagflation 是实时标签（只用于现况判断和一年期预测检验），情景视图请用事后标签 rg_<体制>，已忽略"
            in report["warnings"])
    listed, _ = validate_spec({"name": "测试", "phases": [{**phase, "regime": [
        {"id": "stagflation", "share": 0.7, "why": "叙事"}]}]}, F)
    assert listed["phases"] == clean["phases"]
    again, report2 = validate_spec(clean, F)                     # idempotent: no duplicated views, no warnings
    assert again["phases"] == clean["phases"] and not report2["warnings"] and not report2["clipped"]
    # pooled like any view: clipped to what a 12-month phase can average, with the standard warning
    hi = view_support(F, "rg_recession", 12)[1]
    c2, r2 = validate_spec({"name": "测试", "phases": [{"name": "衰退", "months": 12, "regime": {"衰退": 1.0}}]}, F)
    assert c2["phases"][0]["regime"] == {"recession": 1.0} and c2["phases"][0]["views"][0]["mean"] == pytest.approx(hi)
    assert f"阶段 1：rg_recession 的 mean=1 超出历史可支撑范围，截断为 {hi:.4g}" in r2["warnings"]
    # the requests the checks read: the key, plus plain views at the value asked before the clip
    plain, r3 = validate_spec({"name": "测试", "phases": [{"name": "p", "months": 12,
                                                          "views": [{"var": "rg_depression", "mean": 0.9}]}]}, F)
    assert rv.phase_requests(plain["phases"][0], 1, r3["clipped"]) == {"depression": 0.9}
    assert "regime" not in plain["phases"][0]


# ---------------------------------------------------------------- the checks W5-W11 (15)

def _one(name: str, months: int, **phase) -> dict:
    return {"id": "regime_test", "name": name, "prior_prob": 0.01, "phases": [{"name": name, "months": months, **phase}]}


@pytest.fixture(scope="module")
def regime_examples(ctx):
    """The warnings run_spec would report (validate_spec, then the checks of the pooled phases) for the
    DESIGN §3.3 examples."""
    from scenariolab.scenarios.runner import phase_pools
    stag = next(s for s in library.resolved_specs(ctx) if s["id"] == "stagflation")
    stag["phases"][0]["regime"] = {"stagflation": 0.8}
    cases = {
        "deflation": _one("通缩", 36, regime={"通缩": 0.8}),
        "short_inflation": _one("通胀冲击", 6, regime=[{"id": "inflation", "share": 0.8, "why": "叙事"}]),
        "depression": _one("萧条", 36, regime={"萧条": 0.9}),
        "conflict": _one("滞胀", 12, regime={"滞胀": 0.8}, views=[{"var": "f_ip", "mean_annual": 0.04}]),
        "spiral": stag,
        "plain_contraction": {"id": "t", "name": "两段收缩", "prior_prob": 0.01, "phases": [
            {"name": "下行", "months": 12, "views": [{"var": "rg_contraction", "mean": 0.9}]},
            {"name": "再下行", "months": 12, "views": [{"var": "rg_contraction", "mean": 0.9}]}]},
        "feasible": _one("滞胀", 12, regime={"滞胀": 0.8}, views=[{"var": "f_oil", "mean_annual": 0.15}]),
    }
    out = {}
    for key, spec in cases.items():
        clean, report = validate_spec(spec, ctx.db.frame)
        pools = phase_pools(ctx, clean, report["warnings"])
        out[key] = report["warnings"] + rv.spec_checks(ctx.db, clean, pools, report["clipped"])
    return out


def _has(warnings: list[str], *parts: str) -> bool:
    return any(all(p in w for p in parts) for w in warnings)


@needs_panel
def test_each_regime_warning_fires_on_its_example(ctx, regime_examples):
    F = ctx.db.frame
    year = f"{F.index[0]:%Y}"
    n = int((F["rg_deflation"] > 0.5).sum())
    ex = regime_examples
    assert _has(ex["deflation"], f"阶段 1：{year} 年以来物价负增长的 {n} 个月（", "）是油价暴跌后的基数效应，当月物价其实在涨；"
                "多抽这些月份几乎不会让模拟的 CPI 下跌（实测见“宏观体制”页“引擎里的体制”）。要表达物价下跌，请对 f_cpi 设 "
                "mean_annual（历史支撑下限约 ", "/年），加 收缩 0，并给 f_unrate 一个视图")                          # W5
    assert ("阶段 1 只有 6 个月：模拟的 CPI 同比约要 12 个月才追上抽中月份的通胀，这一阶段里通胀的模拟占比会明显低于抽中占比"
            in ex["short_inflation"])                                                                          # W6
    assert ("阶段 1：深度衰退 90% × 36 个月 ≈ 32 个收缩月份，超过 1948 年以来最长的一次收缩（18 个月）。收缩区块会首尾相接，"
            "收缩月份越多，工业产出跌幅超出历史上最深一次的路径就越多（实测见“宏观体制”页）。建议拆成不超过 18 个月的收缩阶段，"
            "再接一个复苏阶段" in ex["depression"])                                                             # W7
    assert _has(ex["depression"], "阶段 1：深度衰退 最多只能占抽中月份的 ", "（要求 90%）")                        # W8
    assert _has(ex["depression"], f"阶段 1：深度衰退 在 {year} 年以来只有 ", " 段历史（", "），按抽样权重约等于 ",
                " 段；引擎按全部抽中月份做的稀薄检查看不出这一点，模拟只会重复这几段的月份")                      # W9
    assert _has(ex["conflict"], "阶段 1：f_ip 的目标 0.04 与所选体制的历史月份（平均 ", "）相差 ", " 个标准差")     # W10
    assert _has(ex["spiral"], "阶段 1：所选体制单独可以满足（λ=", "），和其他视图一起只能按 λ=", " 满足；冲突最大的是 ")  # W10b
    for key in ("depression", "spiral"):                                                                 # W11
        assert _has(ex[key], "合计约 ", " 个收缩月份（抽中月份里收缩占比 × 月数），超过 1948 年以来最长的一次收缩（18 个月）："
                    "区块首尾相接会造出比历史上任何一次都更深更长的收缩")
    # plain views on ex-post rg_ columns get the same checks; W11 needs no regime key
    assert _has(ex["plain_contraction"], "阶段「下行」「再下行」合计约 ")
    assert not _has(ex["plain_contraction"], "× 12 个月")                    # 0.9 x 12 is no W7
    # a feasible request warns about nothing but the padding to the horizon
    assert ex["feasible"] == ["阶段总月数 12 少于 180，已补一个无视图的回归常态阶段"]


@needs_panel
@check_values
def test_regime_warnings_check_values(regime_examples):
    ex = regime_examples
    assert ("阶段 1：1973 年以来物价负增长的 8 个月（2009-07..2009-10、2015-01..2015-04）是油价暴跌后的基数效应，当月物价其实"
            "在涨；多抽这些月份几乎不会让模拟的 CPI 下跌（实测见“宏观体制”页“引擎里的体制”）。要表达物价下跌，请对 f_cpi 设 "
            "mean_annual（历史支撑下限约 -0.008/年），加 收缩 0，并给 f_unrate 一个视图" in ex["deflation"])
    assert "阶段 1：物价负增长 最多只能占抽中月份的 33%（要求 80%）" in ex["deflation"]
    assert "阶段 1：深度衰退 最多只能占抽中月份的 76%（要求 90%）" in ex["depression"]
    assert ("阶段 1：深度衰退 在 1973 年以来只有 2 段历史（2008-01..2009-06、2020-03..2020-04），按抽样权重约等于 1.0 段；"
            "引擎按全部抽中月份做的稀薄检查看不出这一点，模拟只会重复这几段的月份" in ex["depression"])
    assert ("阶段「萧条」合计约 28 个收缩月份（抽中月份里收缩占比 × 月数），超过 1948 年以来最长的一次收缩（18 个月）："
            "区块首尾相接会造出比历史上任何一次都更深更长的收缩" in ex["depression"])
    assert ("阶段 1：所选体制单独可以满足（λ=1.00），和其他视图一起只能按 λ=0.65 满足；冲突最大的是 s_real_ff（去掉后 λ "
            "升到 0.97）、s_ip_yoy（去掉后 λ 升到 0.88）" in ex["spiral"])
    assert _has(ex["spiral"], "阶段「工资-物价螺旋」「沃尔克式紧缩」合计约 38 个收缩月份")


@needs_panel
def test_no_regime_check_fires_on_the_built_ins(ctx):
    """The 13 built-ins have no regime key: validation warns about nothing, the database checks find nothing,
    so their only warnings stay the engine's own shrink warnings."""
    from scenariolab.scenarios.runner import phase_pools
    base = library.calibration_base(ctx)
    for raw in library.load_specs():
        clean, report = validate_spec(library.resolve_calibration(raw, base)[0], ctx.db.frame)
        assert not report["warnings"] and all("regime" not in p for p in clean["phases"]), raw["id"]
        warnings: list[str] = []
        pools = phase_pools(ctx, clean, warnings)
        assert all("λ=" in w for w in warnings), raw["id"]
        assert rv.spec_checks(ctx.db, clean, pools, report["clipped"]) == [], raw["id"]


# ---------------------------------------------------------------- the readout (16) and the library index (17)

@pytest.fixture(scope="module")
def library_build(ctx, tmp_path_factory):
    """The whole library at the build's paths and seeds, in a temporary directory, as the UI reads it."""
    out = tmp_path_factory.mktemp("library")
    library.run(ctx, n_paths=600, out_dir=out)
    index = json.loads((out / "scenarios.json").read_text(encoding="utf-8"))
    details = {s["id"]: json.loads((out / "scenarios" / f"{s['id']}.json").read_text(encoding="utf-8"))
               for s in index["scenarios"]}
    return index, details


def _mixes(reg: dict):
    for p in reg["phases"]:
        yield from (p["simulated"], p["simulated_rt"], p["drawn"], p["reference"]["simulated"])
    for w in reg["windows"].values():
        yield from (w["simulated"], w["simulated_rt"], w["drawn"], w["reference"]["simulated"])
    for y in reg["years"]:
        yield y["simulated"]


@needs_panel
def test_regime_readout_shape_and_shares(library_build):
    index, details = library_build
    for sid, d in details.items():
        reg = d["regimes"]
        assert reg["version"] == 1 and [x["id"] for x in reg["labels"]] == R.LABEL_IDS
        assert set(reg["windows"]) == {"year1", "years2_5", "years6_15", "whole"}
        assert [y["year"] for y in reg["years"]] == list(range(1, config.HORIZON_MONTHS // 12 + 1))
        assert [p["name"] for p in reg["phases"]] == [p["name"] for p in d["spec"]["phases"]]
        assert [p["months"] for p in reg["phases"]] == [p["months"] for p in d["spec"]["phases"]]
        for mix in _mixes(reg):
            assert list(mix) == R.LABEL_IDS and abs(sum(mix.values()) - 1.0) < 1e-5, sid
        for p in reg["phases"]:
            assert p["requested"] is None and p["dominant"] == max(R.LABEL_IDS, key=lambda i: p["simulated"][i])
            assert 0 <= p["contraction_share"] <= p["paths_with_contraction"] <= 1
            assert 0 < p["history_base_rate"] <= 1
        assert reg["anchor"]["month"] == index["start"] and reg["anchor"]["label_rt"] in R.LABEL_IDS
        assert reg["notes"][-1].startswith("引擎无条件抽样里 CPI 同比高于 4% 的月份占 "), sid     # N4, always
        ep = reg["episodes"]
        assert ep["history"]["episodes"] >= 1 and ep["per_path"] > 0 and ep["reference"]["per_path"] > 0
    # the engine's reference run is one run per process: the same blocks in every scenario
    a, b = details["baseline"]["regimes"], details["financial_crisis"]["regimes"]
    assert a["episodes"]["reference"] == b["episodes"]["reference"]
    assert all(a["windows"][w]["reference"] == b["windows"][w]["reference"] for w in a["windows"])
    assert a["phases"][0]["months"] == b["phases"][0]["months"] == 12
    assert a["phases"][0]["reference"] == b["phases"][0]["reference"] == a["windows"]["year1"]["reference"]


@needs_panel
def test_regime_readout_reads_the_simulated_arrays(ctx, library_build):
    from scenariolab.scenarios import runner
    _, details = library_build
    spec = next(s for s in library.resolved_specs(ctx) if s["id"] == "stagflation")
    detail, sim = runner.run_spec(ctx, spec, n_paths=600, seed=library.seed_for("stagflation"), attribution=False)
    r = R.classify_sim(sim.macro, ctx.panel, anchor=ctx.anchor.date)
    assert np.unique(r["label_rt"][:, 0]).size == 1                         # the anchor uses only the history
    assert detail["regimes"]["anchor"]["label_rt"] == R.ID_OF[int(r["label_rt"][0, 0])]
    drawn = R.label_codes(ctx.db.frame)[sim.idx]
    for p in detail["regimes"]["phases"]:
        sl = slice(p["start_month"], p["start_month"] + p["months"])
        assert p["drawn"] == R.shares(drawn[:, sl]) and p["simulated"] == R.shares(r["label"][:, 1:][:, sl])
    # the build's detail file holds the same readout (JSON rounds to 6 decimals)
    assert details["stagflation"]["regimes"] == json.loads(jsonio.dumps(detail["regimes"]))
    # the switch adds or drops the readout and nothing else
    off, sim_off = runner.run_spec(ctx, spec, n_paths=600, seed=library.seed_for("stagflation"), attribution=False,
                                   regimes=False)
    assert "regimes" not in off and np.array_equal(sim_off.total, sim.total) and np.array_equal(sim_off.idx, sim.idx)
    assert jsonio.dumps({k: v for k, v in detail.items() if k != "regimes"}) == jsonio.dumps(off)


@needs_panel
def test_regime_readout_notes(library_build):
    _, details = library_build
    notes = {sid: [n for n in d["regimes"]["notes"] if "里有“" in n] for sid, d in details.items()}
    assert {sid for sid, n in notes.items() if n} == {"stagflation", "japanification", "oil_war", "energy_transition"}
    assert [n.split("」")[0] for n in notes["stagflation"]] == ["阶段「工资-物价螺旋", "阶段「沃尔克式紧缩"]
    assert [n.split("」")[0] for n in notes["japanification"]] == ["阶段「去杠杆与降息", "阶段「长期停滞"]
    assert all(not d["regimes"]["notes"][:-1] for sid, d in details.items() if sid in ("baseline", "financial_crisis"))


@needs_panel
@check_values
def test_regime_readout_check_values(library_build):
    _, details = library_build
    notes = [n for d in details.values() for n in d["regimes"]["notes"] if "里有“" in n]
    assert sorted(notes) == sorted([
        "阶段「工资-物价螺旋」：情景名称里有“滞胀”，但模拟路径的这一阶段里滞胀合计只占 26%",
        "阶段「沃尔克式紧缩」：情景名称里有“滞胀”，但模拟路径的这一阶段里滞胀合计只占 44%",
        "阶段「去杠杆与降息」：情景名称里有“通缩”，但模拟路径的这一阶段里物价负增长合计只占 3%",
        "阶段「长期停滞」：情景名称里有“通缩”，但模拟路径的这一阶段里物价负增长合计只占 4%",
        "阶段「衰退与油价回落」：名称里有“衰退”，但模拟路径的这一阶段里衰退、深度衰退、滞胀合计只占 41%",
        "阶段「绿色通胀」：名称里有“通胀”，但模拟路径的这一阶段里通胀、滞胀合计只占 41%"])
    reg = details["baseline"]["regimes"]
    assert reg["notes"] == ["引擎无条件抽样里 CPI 同比高于 4% 的月份占 41%（1973 年以来的历史为 33%），所以模拟路径里“通胀”"
                            "的占比系统性偏高；比较时请看同一时段的“引擎无条件”列"]
    assert reg["anchor"] == {"month": "2026-09", "label_rt": "expansion", "label_expost_mix": {"expansion": 1.0}}
    one, two = reg["phases"][:2]                                             # DESIGN §3.4, baseline
    assert [round(100 * one["simulated"][i], 1) for i in ("expansion", "inflation", "recession", "depression",
                                                           "stagflation")] == [67.8, 26.4, 5.3, 0.1, 0.4]
    assert (round(one["paths_with_contraction"], 3), round(one["reference"]["paths_with_contraction"], 4),
            round(one["history_base_rate"], 4)) == (0.098, 0.1915, 0.2417)
    assert [round(100 * two["simulated"][i], 1) for i in ("expansion", "inflation", "recession", "depression",
                                                           "stagflation")] == [74.4, 11.3, 11.5, 1.0, 1.4]
    assert (round(two["paths_with_contraction"], 3), round(two["reference"]["paths_with_contraction"], 3),
            round(two["history_base_rate"], 4)) == (0.367, 0.469, 0.4516)
    ep = reg["episodes"]
    assert (round(ep["reference"]["per_path"], 3), round(ep["reference"]["length_mean"], 2),
            round(ep["reference"]["share_longer_than_18m"], 4), round(ep["reference"]["share_severe"], 4)) == (
        2.116, 11.92, 0.1729, 0.0687)
    assert ep["history"] == {"episodes": 7, "per_15y": pytest.approx(7 * 180 / 644), "length_mean": 11.0,
                             "longer_than_18m": 0, "severe": 2}


@needs_panel
def test_library_index_regime_fields(library_build):
    index, details = library_build
    for s in index["scenarios"]:
        reg, d = s["regimes"], details[s["id"]]["regimes"]
        assert set(reg) == {"year1", "whole", "p_contraction_year1", "dominant_by_phase"}
        assert reg["year1"] == d["windows"]["year1"]["simulated"] and reg["whole"] == d["windows"]["whole"]["simulated"]
        assert reg["p_contraction_year1"] == d["windows"]["year1"]["paths_with_contraction"]
        assert reg["dominant_by_phase"] == [p["dominant"] for p in d["phases"]]
    lib = index["regimes"]
    assert set(lib) == {"year1_weighted", "p_contraction_year1_weighted", "reference", "history"}
    w = np.array([s["prob"] for s in index["scenarios"]])
    assert abs(w.sum() - 1.0) < 1e-9                                         # the library weights
    assert abs(sum(lib["year1_weighted"].values()) - 1.0) < 1e-5            # JSON rounds to 6 decimals
    for i in R.LABEL_IDS:
        assert lib["year1_weighted"][i] == pytest.approx(sum(wk * s["regimes"]["year1"][i]
                                                             for wk, s in zip(w, index["scenarios"])), abs=1e-5)
    ref = lib["reference"]
    assert (ref["n_paths"], ref["seed"]) == (config.REGIMES["reference"]["n_paths"], config.REGIMES["reference"]["seed"])
    assert ref["year1"] == details["baseline"]["regimes"]["windows"]["year1"]["reference"]["simulated"]
    assert abs(sum(lib["history"]["shares_1973"].values()) - 1.0) < 1e-5


@needs_panel
@check_values
def test_library_index_regime_check_values(library_build):
    """DESIGN §3.5: probability-weighted year 1 (simulated, real time, drawn) and P(contraction in year 1)."""
    index, details = library_build
    lib = index["regimes"]
    want = {"expansion": 0.508, "inflation": 0.323, "deflation": 0.002, "recession": 0.090, "depression": 0.023,
            "stagflation": 0.054}
    assert lib["year1_weighted"] == pytest.approx(want, abs=0.002)
    assert lib["p_contraction_year1_weighted"] == pytest.approx(0.257, abs=0.002)
    w = {s["id"]: s["prob"] for s in index["scenarios"]}

    def weighted(key: str) -> dict:
        return {i: sum(w[sid] * d["regimes"]["windows"]["year1"][key][i] for sid, d in details.items())
                for i in R.LABEL_IDS}

    assert weighted("simulated_rt") == pytest.approx({"expansion": 0.557, "inflation": 0.356, "deflation": 0.001,
                                                      "recession": 0.047, "depression": 0.001, "stagflation": 0.040},
                                                     abs=0.002)
    assert weighted("drawn") == pytest.approx({"expansion": 0.614, "inflation": 0.212, "deflation": 0.012,
                                               "recession": 0.049, "depression": 0.057, "stagflation": 0.056},
                                              abs=0.002)
    assert lib["reference"]["year1"] == pytest.approx({"expansion": 0.5441, "inflation": 0.3502, "deflation": 0.0019,
                                                       "recession": 0.0345, "depression": 0.0083,
                                                       "stagflation": 0.061}, abs=5e-5)
    assert lib["reference"]["p_contraction_year1"] == pytest.approx(0.1915)
    assert lib["history"]["shares_1973"] == pytest.approx({"expansion": 396 / 644, "inflation": 163 / 644,
                                                           "deflation": 8 / 644, "recession": 12 / 644,
                                                           "depression": 20 / 644, "stagflation": 45 / 644},
                                                          abs=1e-6)
    assert lib["history"]["p_contraction_12m_window"] == pytest.approx(0.2417, abs=5e-5)
    p_year1 = {s["id"]: round(s["regimes"]["p_contraction_year1"], 2) for s in index["scenarios"]}
    assert p_year1 == {"baseline": 0.10, "k_shaped": 0.17, "oil_war": 0.50, "ai_bust": 0.45, "fiscal_dominance": 0.26,
                       "reacceleration": 0.01, "stagflation": 0.43, "deglobalization": 0.40, "agi_boom": 0.00,
                       "energy_transition": 0.16, "dollar_decline": 0.15, "financial_crisis": 0.79,
                       "japanification": 0.32}
