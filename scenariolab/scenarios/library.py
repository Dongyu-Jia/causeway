"""Built-in scenario library: the most likely futures from today.

Every spec in specs/*.yaml (the same schema the LLM translator emits) is run
through the real engine with its own seed. A spec may carry a top-level

    calibration: {kind: distribution, phase: <0-based phase index>, params: {...}}

(or a list of them): the income-distribution calculator turns the params into
views that replace same-variable views of that phase and factor drifts that
are added to the phase's own drifts; the explanations (one per calibrated
phase, with an engine_check against the simulation) are attached to the
detail JSON as the list detail["calibration"]. Probabilities come from
probabilities.run. Outputs: results/scenarios/<id>.json, results/scenarios.json
and results/scenario_paths.npz (arrays ordered like the index, by probability).
Each index entry carries the headline numbers and, under "assets", every asset's
cagr_p5 / cagr_p50 / cagr_p95 / real_cagr_p50 / max_dd_p50, so the UI can compare
scenarios without loading every detail file; under "regimes", the year-1 and
whole-horizon regime mix of its paths (regime_views.index_fields). The index's
"regimes" holds the probability-weighted year-1 mix, the engine's unconditional
reference run and the history of the month database.
"""
from __future__ import annotations

import copy
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from .. import config, jsonio
from ..context import EngineContext
from . import calibrate, probabilities, regime_views
from .runner import headline, reference_simulation, run_spec
from .schema import MAX_VIEWS_PER_PHASE, VIEW_SCHEMA

SPEC_DIR = Path(__file__).resolve().parent / "specs"
BASE_WINDOW = 12          # months averaged for the calculator's no-shock base
# Per-asset stats copied into the index, so the cross-scenario comparison needs no detail files.
INDEX_ASSET_STATS = ("cagr_p5", "cagr_p50", "cagr_p95", "real_cagr_p50", "max_dd_p50")
VIEW_KEYS = [k for k in VIEW_SCHEMA["properties"] if k not in ("var", "why")]


def load_specs(spec_dir: Path | None = None) -> list[dict]:
    """Raw built-in specs as written in YAML (sorted by file name)."""
    out = []
    for path in sorted(Path(spec_dir or SPEC_DIR).glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(spec, dict):
            spec.setdefault("id", path.stem)
            out.append(spec)
    return out


def calibration_base(ctx: EngineContext, window: int = BASE_WINDOW) -> dict:
    """Today's recent levels the calculator measures shocks from (percent)."""
    tail = ctx.panel[["s_ip_yoy", "s_services_yoy"]].dropna().iloc[-window:]
    return {"ip_yoy": float(tail["s_ip_yoy"].mean()), "services_yoy": float(tail["s_services_yoy"].mean())}


def resolve_calibration(spec: dict, base: dict | None = None, model_spec=None) -> tuple[dict, list[dict]]:
    """Spec with calibration views merged into their phases, plus the explanations.

    model_spec: the active transmission; when the labour share is not in it, each explanation carries
    the calculator's labour-share caveat (calibrate.LABOR_SHARE_CAVEAT)."""
    s = copy.deepcopy(spec)
    cal = s.pop("calibration", None)
    cals = cal if isinstance(cal, list) else [cal] if cal else []
    explanations = []
    phases = s.get("phases") or []
    for c in cals:
        if not isinstance(c, dict) or c.get("kind", "distribution") != "distribution":
            continue
        k = int(c.get("phase", 0))
        if not 0 <= k < len(phases):
            continue
        ph = phases[k]
        params = {**(c.get("params") or {}), "months": int(ph.get("months", 12))}
        views, expl = calibrate.distribution_views(params, base, model_spec=model_spec)
        new_vars = {v["var"] for v in views}
        kept = [v for v in ph.get("views") or [] if v.get("var") not in new_vars]
        ph["views"] = (kept + views)[:MAX_VIEWS_PER_PHASE]
        ov = dict(ph.get("overrides") or {})
        drift = dict(ov.get("factor_drift") or {})
        for fid, val in expl["overrides"]["factor_drift"].items():
            drift[fid] = round(float(drift.get(fid, 0.0)) + val, 6)
        ov["factor_drift"] = drift
        ph["overrides"] = ov
        start = sum(int(p.get("months", 0)) for p in phases[:k])
        expl.update({"phase": k, "phase_name": ph.get("name", f"阶段{k + 1}"), "start_month": start,
                     "months": int(ph.get("months", 12))})
        explanations.append(expl)
    return s, explanations


def resolved_specs(ctx: EngineContext | None = None, spec_dir: Path | None = None) -> list[dict]:
    """Built-in specs with calibrations applied (what the engine actually runs)."""
    base = calibration_base(ctx) if ctx is not None else None
    model_spec = ctx.model.spec if ctx is not None else None
    return [resolve_calibration(s, base, model_spec)[0] for s in load_specs(spec_dir)]


def schema_example(spec: dict) -> dict:
    """A resolved spec in the exact shape of schema.SPEC_JSON_SCHEMA (LLM few-shot example).

    Views carry every key (null when unused) and overrides use the list form;
    top-level `reversion` is not part of the LLM schema and is dropped.
    """
    phases = []
    for ph in spec.get("phases") or []:
        ov = ph.get("overrides") or {}
        views = [{"var": v["var"], **{k: v.get(k) for k in VIEW_KEYS}, "why": v.get("why", "")}
                 for v in ph.get("views") or []]
        phases.append({
            "name": ph.get("name", ""), "months": int(ph.get("months", 12)),
            "regime": regime_list(ph.get("regime")), "views": views,
            "overrides": {
                "factor_drift": [{"factor": k, "annual": float(x), "why": "超出历史支撑的部分，见叙述"}
                                 for k, x in (ov.get("factor_drift") or {}).items()],
                "asset_drift": [{"asset": k, "annual": float(x), "why": "超出历史支撑的部分，见叙述"}
                                for k, x in (ov.get("asset_drift") or {}).items()],
                "resid_scale": ov.get("resid_scale"), "r_star": ov.get("r_star")},
            "confidence": float(ph.get("confidence", 1.0)),
        })
    keys = ("id", "name", "summary", "narrative", "tags", "prior_prob", "analogs", "causal_chain")
    return {**{k: copy.deepcopy(spec.get(k)) for k in keys}, "phases": phases}


def regime_list(reg) -> list[dict]:
    """A phase's regime key in the LLM list form [{id, share, why}] (ids and aliases as written)."""
    if isinstance(reg, list):
        return [{"id": d.get("id"), "share": d.get("share"), "why": d.get("why") or ""} for d in reg
                if isinstance(d, dict)]
    out = []
    for key, share in (reg or {}).items():
        rid = regime_views.regime_id(key)
        try:
            why = regime_views.why_text(rid, float(share)) if rid else ""
        except (TypeError, ValueError):
            why = ""
        out.append({"id": rid or key, "share": share, "why": why})
    return out


def schema_examples(ctx: EngineContext | None = None, ids: list[str] | None = None) -> list[dict]:
    """Built-ins (calibrations applied) as LLM-schema instances, optionally filtered by id."""
    return [schema_example(s) for s in resolved_specs(ctx) if not ids or s.get("id") in ids]


def seed_for(sid: str, base_seed: int = 0) -> int:
    return (zlib.crc32(sid.encode("utf-8")) + base_seed) % 2**31


def _engine_check(expl: dict, sim) -> dict:
    """Compare the calculator's assumptions with what the engine simulated over that phase.

    The simulated changes include the historical background of the drawn months,
    so they are a consistency check, not a test of the calculator. Equity wealth
    is compared with the price return (paid-out dividends are not wealth).
    """
    a, m = expl["start_month"], expl["months"]
    eq = sim.assets.index("us_equity")
    total = sim.total[:, a:a + m, eq].astype(float)
    growth = np.prod(1.0 + total, axis=1) - 1.0
    price = np.prod(1.0 + total - sim.income[:, a:a + m, eq].astype(float), axis=1) - 1.0
    du = (sim.macro["unrate"][:, a + m] - sim.macro["unrate"][:, a]).astype(float)
    ls = np.log(sim.macro["labor_share"][:, a + m] / sim.macro["labor_share"][:, a]).astype(float)
    imp = expl["impact"]
    return {"equity_wealth_change_assumed": imp["params"]["equity_wealth_change"],
            "us_equity_price_return_p50": float(np.median(price)),
            "us_equity_total_return_p50": float(np.median(growth)),
            "unemployment_change_assumed_pp": imp["unemployment_change_pp"],
            "unemployment_change_p50_pp": float(np.median(du)),
            "unemployment_change_mean_pp": float(du.mean()),
            "labor_share_log_change_assumed": imp["labor_share_log_change"],
            "labor_share_log_change_p50": float(np.median(ls))}


def _anchor_dict(ctx: EngineContext) -> dict:
    d = asdict(ctx.anchor)
    d["date"] = ctx.anchor.date.strftime("%Y-%m")
    d.pop("cpi_log_hist", None)
    d.pop("base_init", None)
    return d


def _asset_stats(detail: dict) -> dict:
    """{asset: {cagr_p5, cagr_p50, cagr_p95, real_cagr_p50, max_dd_p50}} for the index."""
    out = {}
    for a, x in (detail.get("assets") or {}).items():
        st = (x or {}).get("stats") or {}
        out[a] = {k: st.get(k) for k in INDEX_ASSET_STATS}
    return out


def _headline(detail: dict) -> dict:
    h = headline(detail)
    h.pop("ens_min", None)
    a = detail["assets"]
    y10 = detail["macro"].get("y10", {}).get("fan", {}).get("p50", [])
    ur = detail["macro"].get("unrate", {}).get("fan", {}).get("p50", [])
    h.update({
        "us_equity_max_dd_p50": a["us_equity"]["stats"]["max_dd_p50"],
        "nasdaq_cagr_p50": a["nasdaq"]["stats"]["cagr_p50"],
        "gold_cagr_p50": a["gold"]["stats"]["cagr_p50"],
        "treasury_10y_cagr_p50": a["treasury_10y"]["stats"]["cagr_p50"],
        "cash_real_cagr_p50": a["cash"]["stats"]["real_cagr_p50"],
        "y10_end_p50": y10[-1] if y10 else None,
        "unrate_max_p50": max(ur) if ur else None,
    })
    return h


def run(ctx: EngineContext, n_paths: int = 600, out_dir: Path | None = None,
        spec_dir: Path | None = None, ids: list[str] | None = None, base_seed: int = 0,
        regimes: bool | None = None) -> dict:
    """Run every built-in scenario, write the result files and return the index.

    `ids` runs a subset and writes a subset index and npz (probabilities
    renormalized over it), so pass an `out_dir` other than results/ with it.
    `regimes` is run_spec's switch for the regime readout (None = runner.REGIME_READOUT);
    a study that needs only the returns passes False.
    """
    out_dir = Path(out_dir or config.RESULTS_DIR)
    base = calibration_base(ctx)
    details, sims = [], []
    for raw in load_specs(spec_dir):
        if ids and raw.get("id") not in ids:
            continue
        spec, cals = resolve_calibration(raw, base, ctx.model.spec)
        detail, sim = run_spec(ctx, spec, n_paths=n_paths, seed=seed_for(str(spec.get("id")), base_seed),
                               source="builtin", regimes=regimes)
        if cals:
            for c in cals:
                c["engine_check"] = {**_engine_check(c, sim), **calibrate.transmission_check(ctx.model.spec)}
            detail["calibration"] = cals
        details.append(detail)
        sims.append(sim)

    probs = probabilities.run(ctx, [d["spec"] for d in details])
    by_id = {p["id"]: p for p in probs["scenarios"]}
    for d in details:
        p = by_id[d["id"]]
        d["probability"] = {"prior": p["prior"], "likelihood": p["likelihood"], "posterior": p["posterior"],
                            "distance": p["distance"], "z": p["z"], "phase_means": p["phase_means"],
                            "today": probs["today"], "method": probs["method"]}
        d["prob"] = p["posterior"]
    order = sorted(range(len(details)), key=lambda i: -details[i]["prob"])
    details = [details[i] for i in order]
    sims = [sims[i] for i in order]

    scen_dir = out_dir / "scenarios"
    for d in details:
        jsonio.dump(d, scen_dir / f"{d['id']}.json")
    index = {
        "start": ctx.anchor.date.strftime("%Y-%m"), "horizon_months": config.HORIZON_MONTHS,
        "n_paths": n_paths, "anchor": _anchor_dict(ctx), "cma": ctx.cma,
        "probability_method": probs["method"],
        "probability": {k: probs[k] for k in ("gamma", "window_months", "states", "today", "scales", "date")},
        "scenarios": [{
            "id": d["id"], "name": d["name"], "summary": d["summary"], "tags": d["tags"],
            "source": "builtin", "prior_prob": d["probability"]["prior"],
            "likelihood": d["probability"]["likelihood"], "prob": d["prob"],
            "headline": _headline(d), "ens_min": min(p["ep"]["ens"] for p in d["phases"]),
            "analogs": d.get("analogs", []), "warnings": len(d["validation"]["warnings"]),
            "calibrated": "calibration" in d, "assets": _asset_stats(d),
            **({"regimes": regime_views.index_fields(d["regimes"])} if "regimes" in d else {}),
        } for d in details],
    }
    if any("regimes" in s for s in index["scenarios"]):
        index["regimes"] = {**regime_views.library_mix(index["scenarios"]),
                            "reference": regime_views.reference_summary(reference_simulation(ctx)),
                            "history": regime_views.history_summary(ctx.db.frame)}
    jsonio.dump(index, out_dir / "scenarios.json")
    if sims:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "scenario_paths.tmp.npz"
        np.savez_compressed(
            tmp,
            total=np.stack([s.total for s in sims]).astype(np.float32),
            income=np.stack([s.income for s in sims]).astype(np.float32),
            cpi=np.stack([s.macro["cpi"] for s in sims]).astype(np.float32),
            ids=np.array([d["id"] for d in details]), assets=np.array(sims[0].assets),
            labels=np.array(details[0]["labels"]), probs=np.array([d["prob"] for d in details]),
        )
        tmp.replace(out_dir / "scenario_paths.npz")
    return index
