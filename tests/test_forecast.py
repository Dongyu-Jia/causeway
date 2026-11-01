import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.analysis import forecast as fc
from scenariolab.engine.factor_model import ModelSpec

needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")


def test_crps_of_samples_matches_the_normal_closed_form():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 0.15, size=200000)
    for y in (-0.3, 0.05, 0.4):
        assert fc.crps_samples(x, y) == pytest.approx(fc.crps_normal(0.05, 0.15, y), rel=0.02)


def test_pit_and_shock_flags():
    s = np.arange(100, dtype=float)
    assert fc.pit(s, 49.5) == 0.5 and fc.pit(s, -1) == 0.0 and fc.pit(s, 1e9) == 1.0
    assert fc.shock_names(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-01")) == ["新冠疫情与全面封锁"]
    assert fc.shock_names(pd.Timestamp("2015-01-01"), pd.Timestamp("2015-12-01")) == []


@needs_panel
def test_origins_have_a_full_realized_year():
    from scenariolab.context import load_panel_with_shocks
    panel = load_panel_with_shocks()
    dates = fc.forecast_origins(panel)
    assert dates[0] == pd.Timestamp(fc.FIRST_ORIGIN)
    assert all(d + pd.DateOffset(months=12) <= panel.index.max() for d in dates)
    assert all(d.month == 12 for d in dates[:-1])


@needs_panel
def test_forecast_uses_only_data_up_to_the_origin():
    from scenariolab.context import load_panel_with_shocks
    panel = load_panel_with_shocks()
    t = pd.Timestamp("2004-12-01")
    other = panel.copy()
    rows = other.index > t
    cols = [c for c in other.columns if c.startswith(("r_", "p_", "d_", "f_", "s_", "lvl_", "e_", "dy_"))]
    rng = np.random.default_rng(3)
    other.loc[rows, cols] = other.loc[rows, cols].to_numpy() * rng.normal(1.0, 0.3, size=(int(rows.sum()), len(cols)))
    a = fc.forecast_origin(panel, ModelSpec(), t, n_paths=200, seed=5)
    b = fc.forecast_origin(other, ModelSpec(), t, n_paths=200, seed=5)
    for asset in fc.ASSETS:
        for m in ("uncond", "analog"):
            assert a["assets"][asset][m]["p50"] == pytest.approx(b["assets"][asset][m]["p50"], abs=1e-12)
    assert a["analogs"] == b["analogs"]
    assert a["assets"]["us_equity"]["realized"] != b["assets"]["us_equity"]["realized"]   # only the outcome moved


@needs_panel
def test_analog_weights_rest_on_the_requested_number_of_starts():
    from scenariolab.context import EngineContext
    from scenariolab.engine.ep import effective_number
    ctx = EngineContext.load(selection=False)
    w, info = fc.analog_weights(ctx.db, fc.state_at(ctx.panel, ctx.anchor.date))
    assert w[0] == 0.0 and w.sum() == pytest.approx(1.0)
    assert effective_number(w) == pytest.approx(fc.ANALOG_ENS, rel=0.01)
    assert info["top"] and all(pd.Timestamp(x["date"]) < ctx.anchor.date - pd.DateOffset(months=11) for x in info["top"])


# ---------------------------------------------------------------- the live forecast ledger

def _band_entry(mu: float, sd: float) -> dict:
    """A method's per-asset record as the ledger stores it, for a normal forecast."""
    from scipy.stats import norm
    q = norm.ppf(fc.LEDGER_TAUS, mu, sd)
    p = {f"p{k}": float(norm.ppf(k / 100, mu, sd)) for k in (5, 25, 50, 75, 95)}
    return {**p, "mean": mu, "prob_loss": float(norm.cdf(0.0, mu, sd)), "q": q.tolist()}


def _entry(origin: str, mu: float = 0.10, sd: float = 0.15) -> dict:
    t = pd.Timestamp(f"{origin}-01")
    return {"schema": fc.LEDGER_SCHEMA, "origin": origin, "horizon_months": 12,
            "matures": (t + pd.DateOffset(months=12)).strftime("%Y-%m"), "created": "2026-01-01T00:00:00Z",
            "methods": {"analog": {"assets": {a: _band_entry(mu, sd) for a in fc.ASSETS}},
                        "uncond": {"assets": {a: _band_entry(0.0, 0.30) for a in fc.ASSETS}},
                        "naive": {"assets": {a: {"mu": mu, "sd": sd} for a in fc.ASSETS}}}}


def test_ledger_is_append_only_and_never_duplicates_an_origin(tmp_path):
    first = _entry("2026-09")
    assert fc.append_ledger(first, tmp_path) is True
    path = tmp_path / "forecast_2026-09.json"
    before = path.read_bytes()
    # a rebuild on the same panel (same origin) adds nothing and leaves the first record as it was
    assert fc.append_ledger({**_entry("2026-09", mu=0.5), "created": "2026-02-02T00:00:00Z"}, tmp_path) is False
    assert path.read_bytes() == before
    assert fc.append_ledger(_entry("2026-10"), tmp_path) is True
    assert sorted(p.name for p in tmp_path.iterdir()) == ["forecast_2026-09.json", "forecast_2026-10.json"]  # no temp files
    entries, errors = fc.load_ledger(tmp_path)
    assert [e["origin"] for e in entries] == ["2026-09", "2026-10"] and errors == []
    assert entries[0]["created"] == "2026-01-01T00:00:00Z" and entries[0]["methods"]["analog"]["assets"]["gold"]["mean"] == 0.1
    (tmp_path / "forecast_2027-01.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "forecast_bad.json").write_text('{"origin": "next year", "methods": {}}', encoding="utf-8")
    entries, errors = fc.load_ledger(tmp_path)
    assert len(entries) == 2 and len(errors) == 2 and "forecast_2027-01.json" in errors[0] and "不是预测记录" in errors[1]


def test_a_matured_ledger_entry_is_scored_and_a_young_one_waits():
    from scipy.stats import norm
    idx = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
    panel = pd.DataFrame({f"r_{a}": np.full(len(idx), 0.01) for a in fc.ASSETS}, index=idx)
    y = 1.01 ** 12 - 1.0                                         # every asset: +12.7% over any 12 months
    out = fc.score_ledger([_entry("2024-06"), _entry("2025-06")], panel)
    assert out["matured"] == 1 and out["pending"] == 1 and out["first_maturity"] == "2026-06"
    done, young = out["entries"]
    assert done["status"] == "matured" and young["status"] == "pending" and young["scores"] is None
    assert done["realized"]["us_equity"] == pytest.approx(y)
    s = done["scores"]["analog"]["us_equity"]
    assert s["covered"] is True
    assert s["pit"] == pytest.approx(norm.cdf(y, 0.10, 0.15), abs=0.01)
    assert s["crps"] == pytest.approx(fc.crps_normal(0.10, 0.15, y), rel=0.02)     # quantile CRPS ~ the closed form
    assert done["scores"]["naive"]["us_equity"]["crps"] == pytest.approx(fc.crps_normal(0.10, 0.15, y))
    agg = out["scores"]
    assert agg["analog"]["risk_avg"]["coverage90"] == 1.0 and agg["analog"]["n_entries"] == 1
    assert agg["crps_skill_risk"]["analog_vs_uncond"] > 0          # the sharp, centred forecast beats the wide one
    assert done["methods"]["analog"]["us_equity"]["p50"] == pytest.approx(0.10) and "q" not in done["methods"]["analog"]["us_equity"]


def test_ledger_only_appends_in_time_and_in_real_time():
    """No backfill (an origin older than one recorded) and no stale forecast (recorded months after its origin)."""
    recorded = [_entry("2026-09"), _entry("2026-12")]
    late = {**_entry("2026-10"), "created": "2026-11-02T00:00:00Z"}
    assert "早于账本里已有的起点 2026-12" in fc.ledger_refusal(late, recorded)
    assert fc.ledger_refusal({**_entry("2027-01"), "created": "2027-02-15T00:00:00Z"}, recorded) is None
    assert fc.ledger_refusal({**_entry("2026-12"), "created": "2027-01-15T00:00:00Z"}, recorded) is None  # same origin: append_ledger says no
    stale = fc.ledger_refusal({**_entry("2027-01"), "created": "2027-06-01T00:00:00Z"}, recorded)
    assert stale and "不算实时预测" in stale and "2027-06" in stale
    assert fc.ledger_refusal({**_entry("2027-01"), "created": "2027-04-30T23:59:59Z"}, recorded) is None   # 3 months: still real time


def test_a_copied_ledger_file_is_counted_once(tmp_path):
    fc.append_ledger(_entry("2026-09"), tmp_path)
    (tmp_path / "forecast_2026-09 copy.json").write_bytes((tmp_path / "forecast_2026-09.json").read_bytes())
    entries, errors = fc.load_ledger(tmp_path)
    assert [e["origin"] for e in entries] == ["2026-09"]
    assert len(errors) == 1 and "forecast_2026-09 copy.json" in errors[0] and "重复记录" in errors[0]


@needs_panel
def test_scenario_paths_of_another_start_month_are_not_used(tmp_path):
    """After a data update whose scenarios step failed, the old paths' first year is not the next 12 months:
    no scenario mixture, no reconciliation, no calibration and nothing for the ledger."""
    from scenariolab.context import EngineContext
    ctx = EngineContext.load(selection=True)
    S, N, T, A = 2, 20, 24, len(fc.ASSETS)
    start = (ctx.anchor.date - pd.DateOffset(months=3)).strftime("%Y-%m")
    np.savez(tmp_path / "scenario_paths.npz", total=np.full((S, N, T, A), 0.005, np.float32),
             income=np.zeros((S, N, T, A), np.float32), cpi=np.full((S, N, T + 1), 100.0, np.float32),
             ids=np.array(["a", "b"]), assets=np.array(fc.ASSETS), probs=np.array([0.5, 0.5]),
             labels=np.array([(pd.Timestamp(f"{start}-01") + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(T + 1)]))
    assert fc.year_one_paths(tmp_path)["start"] == start
    quant: dict = {}
    live = fc.live_forecast(ctx, n_paths=60, results_dir=tmp_path, ledger=quant)
    assert live["scenario_probs"] is None and live["reconciliation"] is None
    assert live["calibration"]["available"] is False and start in live["calibration"]["reason"]
    assert all("scenarios" not in live["assets"][a] for a in fc.ASSETS)
    assert set(quant["methods"]) == {"analog", "uncond", "naive"}


def test_ledger_report_says_when_the_first_entry_matures(tmp_path):
    idx = pd.date_range("2025-01-01", "2026-09-01", freq="MS")
    panel = pd.DataFrame({f"r_{a}": np.zeros(len(idx)) for a in fc.ASSETS}, index=idx)
    fc.append_ledger(_entry("2026-09"), tmp_path)
    entries, errors = fc.load_ledger(tmp_path)
    rep = fc.ledger_report(entries, errors, panel, appended=True, ledger_dir=tmp_path)
    assert rep["n_entries"] == 1 and rep["matured"] == 0 and rep["scores"] is None and rep["first_maturity"] == "2027-09"
    assert "2027-09" in rep["text"] and rep["appended"] is True


# ---------------------------------------------------------------- calibrated scenario probabilities

def test_calibrated_probabilities_sum_to_one_meet_the_views_and_stay_close():
    prior = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    means = {"a": {"unrate": 0.1, "cpi_yoy": -0.4}, "b": {"unrate": 0.8, "cpi_yoy": 1.2},
             "c": {"unrate": 1.5, "cpi_yoy": -2.0}, "d": {"unrate": -0.2, "cpi_yoy": 0.5}}
    targets = {"unrate": 0.3, "cpi_yoy": -0.1}
    out = fc.calibrate_probs(prior, means, targets, ["unrate", "cpi_yoy"])
    q = out["probs"]
    assert sum(q.values()) == pytest.approx(1.0) and all(v > 0 for v in q.values())
    for k, target in targets.items():
        assert sum(q[s] * means[s][k] for s in q) == pytest.approx(target, abs=1e-4)
    assert out["converged"] and out["shrink"] == 1.0 and out["kl"] >= 0.0 and out["ens"] <= 4.0
    assert [v["var"] for v in out["views"]] == ["unrate", "cpi_yoy"]
    # views the prior already meets cost nothing; no views return the prior itself
    same = fc.calibrate_probs(prior, means, {k: sum(prior[s] * means[s][k] for s in prior) for k in targets}, ["unrate"])
    assert same["kl"] == pytest.approx(0.0, abs=1e-9)
    none = fc.calibrate_probs(prior, means, targets, [])
    assert none["probs"] == pytest.approx(prior) and none["kl"] == 0.0
    # a target no mix of scenarios can reach is shrunk toward the prior instead of failing
    far = fc.calibrate_probs(prior, means, {"unrate": 5.0}, ["unrate"])
    assert far["shrink"] < 1.0 and far["converged"] and sum(far["probs"].values()) == pytest.approx(1.0)


def _macro_rows(analog_better_early: bool, analog_better_late: bool, naive: float = 1.2) -> list[dict]:
    rows = []
    for i, year in enumerate(range(1985, 2025)):
        early = year <= 2004
        better = analog_better_early if early else analog_better_late
        jitter = 0.01 * ((i * 7) % 5 - 2)
        a, u = (0.9 + jitter, 1.0) if better else (1.1 + jitter, 1.0)
        macro = {k: {"realized": 0.0, "analog": {"p5": -1, "p50": 0, "p95": 1, "crps": a if k == "cpi_yoy" else 1.0},
                     "uncond": {"p5": -1, "p50": 0, "p95": 1, "crps": u},
                     "naive": {"p5": -1, "p50": 0, "p95": 1, "crps": naive + jitter / 2}} for k in fc.MACRO}
        rows.append({"date": f"{year}-12", "shocks": [], "macro": macro})
    return rows


def test_macro_skill_needs_both_halves():
    good = fc.macro_skill(_macro_rows(True, True))
    assert good["skilled"] == ["cpi_yoy"] and good["vars"]["cpi_yoy"]["all"]["skill"] > 0
    assert good["vars"]["cpi_yoy"]["all"]["t"] > 1 and good["vars"]["unrate"]["skilled"] is False   # a tie is no skill
    assert good["vars"]["cpi_yoy"]["beats"] == {"uncond": True, "naive": True}
    assert fc.macro_skill(_macro_rows(True, False))["skilled"] == []                # worse after 2004: not forecastable


def test_macro_skill_must_also_beat_the_naive_benchmark():
    """The engine's unconditional draw shares the analog's biases (both draw the historical months' inflation
    without re-anchoring it): beating it alone is no skill when a naive forecast of the change does far better."""
    out = fc.macro_skill(_macro_rows(True, True, naive=0.5))
    cpi = out["vars"]["cpi_yoy"]
    assert cpi["beats"] == {"uncond": True, "naive": False} and cpi["naive"]["all"]["skill"] < 0
    assert out["skilled"] == [] and "历史均值基准" in out["rule"] and "无条件抽样" in out["rule"]
    txt = fc._skill_verdicts(out)
    assert "CPI 同比胜过无条件抽样" in txt and "不如历史均值基准" in txt
    no_naive = [{**r, "macro": {k: {m: v for m, v in x.items() if m != "naive"} for k, x in r["macro"].items()}}
                for r in _macro_rows(True, True)]
    assert fc.macro_skill(no_naive)["skilled"] == []                             # no benchmark scored: not shown to beat it


@needs_panel
def test_naive_macro_benchmark_uses_only_months_up_to_the_origin():
    from scenariolab.context import load_panel_with_shocks
    panel = load_panel_with_shocks()
    t = pd.Timestamp("1999-12-01")
    a = fc.naive_macro(panel, t)
    b = fc.naive_macro(_scaled_after(panel, t), t)
    assert a == b and all(np.isfinite(mu) and sd > 0 for mu, sd in a.values())
    ch = (panel.loc[config.MODEL_START:t, "s_cpi_yoy"] - panel.loc[config.MODEL_START:t, "s_cpi_yoy"].shift(12)).dropna()
    assert a["cpi_yoy"] == pytest.approx((ch.mean(), ch.std()))


def _scaled_after(panel: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    other = panel.copy()
    cols = [c for c in other.columns if c.startswith(("s_", "lvl_"))]
    other.loc[other.index > t, cols] = other.loc[other.index > t, cols].to_numpy() * 1.7
    return other


def _toy_reconcile(skilled: list[str]):
    rng = np.random.default_rng(4)
    ids, A, N = ["calm", "bust", "boom"], list(fc.ASSETS), 400
    shift = {"calm": 0.06, "bust": -0.15, "boom": 0.12}
    growth = np.stack([rng.normal(shift[s], 0.15, size=(N, len(A))) for s in ids])
    year_one = {"ids": ids, "assets": A, "n_paths": N, "growth": growth, "infl": np.full((3, N), 0.03),
                "probs": np.array([0.5, 0.3, 0.2]), "total": None}
    unrate = {"calm": 0.0, "bust": 1.5, "boom": -0.5}
    y1_macro = {"macro": {s: {k: rng.normal(unrate[s] if k == "unrate" else 0.0, 0.3, N) for k in fc.MACRO} for s in ids},
                "reproduced": {s: True for s in ids}, "missing": []}
    an = rng.normal(0.08, 0.15, size=(1000, len(A)))
    runs = {"analog": {"growth": an, "macro": {k: rng.normal(0.1 if k == "unrate" else 0.0, 0.5, 1000) for k in fc.MACRO}},
            "uncond": {"growth": an, "macro": {k: rng.normal(0.0, 0.6, 1000) for k in fc.MACRO}}}
    live_assets = {a: {"analog": fc._bands(an[:, j])} for j, a in enumerate(A)}
    skill = {"skilled": skilled, "rule": "测试规则", "n_origins": 42,
             "vars": {k: {"all": {"skill": 0.02 if k in skilled else -0.01, "t": 1.5}} for k in fc.MACRO}}
    return fc.reconcile(live_assets, runs, year_one, y1_macro, skill, results_dir="/nonexistent")


def test_reconciliation_splits_the_gap_by_scenario_and_calibrates_only_on_skill():
    rec, cal = _toy_reconcile(["unrate"])
    eq = rec["summary"]["us_equity"]
    assert sum(rec["contrib_mean"]["us_equity"].values()) == pytest.approx(eq["gap_mean"])        # an exact split
    assert sum(rec["contrib_prob_loss"]["us_equity"].values()) == pytest.approx(eq["gap_prob_loss"])
    assert min(rec["contrib_mean"]["us_equity"], key=rec["contrib_mean"]["us_equity"].get) == "bust"
    assert cal["available"] and cal["skilled"] == ["unrate"] and cal["kl"] >= 0
    assert sum(cal["probs"].values()) == pytest.approx(1.0)
    v = cal["views"][0]
    assert v["var"] == "unrate" and v["posterior_mean"] == pytest.approx(v["target"], abs=1e-4)
    assert cal["probs"]["bust"] < 0.3                                     # the analog's milder unemployment tilts away from it
    assert eq["calibrated"] is not None and rec["text"] and cal["text"].startswith("以模型后验为先验")
    rec0, cal0 = _toy_reconcile([])
    assert cal0["available"] is False and "不做校准" in cal0["reason"] and "probs" not in cal0
    assert rec0["summary"]["us_equity"]["calibrated"] is None and rec0["sensitivity"]
