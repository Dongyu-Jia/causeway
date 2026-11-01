"""Run one scenario spec through the real engine.

spec -> validate_spec -> per-phase entropy pooling over block starts ->
simulate -> chart-ready detail JSON (fans, monthly and yearly K-lines, macro
paths, posterior correlations, the evidence behind the weights, each asset's
return attribution per phase, and the macro regimes of the simulated paths
next to the engine's own unconditional reference run). Used for the built-in
library, LLM-translated scenarios and user-edited specs alike.
"""
from __future__ import annotations

import threading
import weakref

import numpy as np
import pandas as pd

from .. import config
from ..context import EngineContext
from ..engine import regimes, summary
from ..engine.ep import weighted_corr
from ..engine.pooling import BLOCK, MIN_ENS, MIN_EPISODES, PhasePool, pool_phase
from ..engine.sampler import BlockProbs
from ..engine.simulate import (MonthDB, SimOutput, ValuationModel, fit_valuation, simulate, valuation_settings,
                               valuation_setup, with_oil_lock)
from . import regime_views
from .schema import validate_spec

MACRO_VARS = ("cpi_yoy", "ff", "y10", "unrate", "oil", "spread")
VALUATION_MACRO = ("pd", "pd_fair")     # fanned when the simulation pulls toward the fair value
TRACKED_MACRO = ("pd",)                 # fanned when the state is only tracked (kappa 0: no fair value in use)
RESIDUALS = dict(config.RESIDUALS)      # simulate()'s residual draw (swapped by analysis.ablation)
VALUATION = dict(config.VALUATION)      # valuation-state mode (swapped by analysis.valuation); on if "enabled"
# run_spec adds assets[a].attribution (a study that only needs the returns may switch it off; it changes no
# simulated number either way and adds about 30 ms to a 600-path scenario run)
ATTRIBUTION = True
# run_spec adds detail["regimes"] (regime_views.readout: the macro regimes of the paths, about 0.25 s per 600
# paths plus one cached reference run). It only reads the simulated arrays; studies that need only the
# returns pass regimes=False.
REGIME_READOUT = True
ATTRIBUTION_UNITS = "年化：每月算术回报的均值 ×12（小数），各项相加 = arithmetic；cagr_p50 = 各路径几何年化总回报的中位数"
ATTRIBUTION_BAND = (10, 90)             # percentiles over paths of the factor part
_VALUATION_CACHE: "weakref.WeakKeyDictionary[MonthDB, dict]" = weakref.WeakKeyDictionary()


def valuation_for(ctx: EngineContext) -> tuple[dict, ValuationModel | None]:
    """(CMA, ValuationModel or None) the runner simulates with (cached per month database and settings).

    Mode on: the mode's CMA and the fair value estimated on ctx.db. Mode off with "track": ctx.cma and
    the state at kappa 0, which leaves every return as without it and records the P/D the paths imply;
    a panel without the valuation columns then simply runs untracked. Otherwise (ctx.cma, None).
    """
    s = valuation_settings(VALUATION)
    if not s and not VALUATION.get("track"):
        return ctx.cma, None
    key = (ctx.anchor.date, tuple(sorted((k, str(v)) for k, v in (s or {"track": True}).items())))
    cache = _VALUATION_CACHE.setdefault(ctx.db, {})
    if key not in cache:
        if s:
            cache[key] = valuation_setup(ctx.db, ctx.anchor, s)
        else:
            # only the kappa-0 state is cached: it depends on the month database alone, while the CMA is
            # the caller's (two contexts can share a db with different CMAs, e.g. the ablation's
            # historical-CMA scenario run)
            try:
                cache[key] = None, fit_valuation(ctx.db, ctx.anchor, kappa=0.0, income="constant")
            except (KeyError, ValueError, np.linalg.LinAlgError):
                cache[key] = None, None
    cma, vm = cache[key]
    return (cma if s else ctx.cma), vm


def _top_months(dates: pd.DatetimeIndex, q: np.ndarray, p: np.ndarray, k: int = 10) -> list[dict]:
    ratio = np.divide(q, p, out=np.zeros_like(q), where=p > 0)
    order = np.argsort(-q)[:k]
    return [{"date": dates[i].strftime("%Y-%m"), "weight_ratio": float(ratio[i]), "weight": float(q[i])}
            for i in order]


def ep_report(db: MonthDB, pool: PhasePool) -> dict:
    """What a phase's weights rest on: effective months, era mix, top months, per-view targets.

    Views are reported on the pooling frame (block means), so prior_sd is the
    spread of what a phase can average, and posterior_mean is the expected
    mean of the drawn months.
    """
    res = pool.ep
    rep = {"ens": pool.ens, "ens_ratio": pool.ens / len(db), "block": pool.blocks.block, "episodes": pool.episodes,
           "shrink": pool.shrink, "shrink_reason": pool.shrink_reason, "ens_requested": pool.ens_requested,
           "kl": res.kl if res else 0.0, "converged": res.converged if res else True,
           "max_violation": res.max_violation if res else 0.0, "views": res.views if res else []}
    rep["era_weights"] = db.era_weights(pool.month_weights)
    rep["top_months"] = _top_months(db.dates, pool.month_weights, pool.prior_month_weights)
    return rep


def phase_pools(ctx: EngineContext, spec: dict, warnings: list[str]) -> list[PhasePool]:
    """Per phase of a validated spec: block-start weights whose drawn months honour the views."""
    db = ctx.db
    prior = db.prior()
    pools = []
    for k, ph in enumerate(spec["phases"]):
        pool = pool_phase(db.frame, ph["views"], prior, ph["months"], BLOCK, ph.get("confidence", 1.0), rare=db.rare)
        if pool.shrink < 1.0:
            why = (f"在历史上找不到能同时满足的 {pool.blocks.block} 个月区块组合" if pool.shrink_reason == "infeasible"
                   else f"在历史上的支撑太薄（约 {pool.ens_requested:.0f} 个有效月份；要求至少 {MIN_ENS:.0f} 个有效月份、"
                   f"{MIN_EPISODES:.0f} 段不同的 12 个月历史片段）")
            warnings.append(f"阶段 {k + 1}「{ph['name']}」的视图组合{why}，已把所有视图按 λ={pool.shrink:.2f} "
                            f"向历史均值收缩（λ=1 为原样，0 为不加视图），以免模拟只重复少数几段历史")
        pools.append(pool)
    return pools


def phase_weights(ctx: EngineContext, spec: dict, warnings: list[str]) -> tuple[list[BlockProbs], list[dict]]:
    """(what simulate() draws from, per-phase reports) for a validated spec."""
    pools = phase_pools(ctx, spec, warnings)
    return [p.blocks for p in pools], _phase_reports(ctx, spec, pools)


def _phase_reports(ctx: EngineContext, spec: dict, pools: list[PhasePool]) -> list[dict]:
    reports, start = [], 0
    for ph, pool in zip(spec["phases"], pools):
        reports.append({"name": ph["name"], "start_month": start, "months": ph["months"],
                        "views": ph["views"], "overrides": ph.get("overrides", {}),
                        "confidence": ph.get("confidence", 1.0), "ep": ep_report(ctx.db, pool)})
        start += ph["months"]
    return reports


def yearly_candles(monthly: list[list]) -> list[list]:
    """Aggregate [label, open, close, low, high] monthly candles into calendar years."""
    out: list[list] = []
    for row in monthly:
        year = row[0][:4]
        if out and out[-1][0] == year:
            cur = out[-1]
            cur[2] = row[2]
            cur[3] = min(cur[3], row[3])
            cur[4] = max(cur[4], row[4])
        else:
            out.append([year, row[1], row[2], row[3], row[4]])
    return out


def summarize(ctx: EngineContext, sim: SimOutput, valuation_macro: tuple[str, ...] = VALUATION_MACRO) -> dict:
    """Chart-ready summaries of a simulation (shared with backtests); `valuation_macro` are the valuation
    paths fanned when the simulation has them."""
    T = sim.total.shape[1]
    labels = summary.month_labels(sim.start, T)
    W = summary.wealth_paths(sim.total)                                  # (n, T+1, A)
    deflator = sim.macro["cpi"] / sim.macro["cpi"][:, :1]
    W_real = W / deflator[:, :, None]
    eq = sim.assets.index("us_equity")
    k = summary.representative_path(W[:, :, eq])
    years = T / 12.0
    assets = {}
    for j, a in enumerate(sim.assets):
        r = sim.total[:, :, j].astype(float)
        vol_m = float(r.std())
        monthly = summary.candles(r[k], vol_m, labels, seed=17 + j)
        real_cagr = W_real[:, -1, j] ** (1.0 / years) - 1.0
        cagr = W[:, -1, j] ** (1.0 / years) - 1.0
        assets[a] = {
            "fan": summary.fan(W[:, :, j]),
            "fan_real": summary.fan(W_real[:, :, j]),
            "candles_monthly": monthly,
            "candles_yearly": yearly_candles(monthly),
            "stats": {
                "cagr_p5": float(np.percentile(cagr, 5)), "cagr_p50": float(np.percentile(cagr, 50)),
                "cagr_p95": float(np.percentile(cagr, 95)), "cagr_mean": float(cagr.mean()),
                "real_cagr_p5": float(np.percentile(real_cagr, 5)),
                "real_cagr_p50": float(np.percentile(real_cagr, 50)),
                "real_cagr_p95": float(np.percentile(real_cagr, 95)),
                "vol": float(r.std(axis=1).mean() * np.sqrt(12.0)),
                "max_dd_p50": float(np.percentile(summary.max_drawdown(W[:, :, j]), 50)),
                "income_yield": float(sim.income[:, :, j].mean() * 12.0),
            },
        }
    macro = {}
    for m in MACRO_VARS + tuple(valuation_macro):
        if m in sim.macro:
            macro[m] = {"fan": summary.fan(sim.macro[m].astype(float))}
    eras = ctx.db.eras[sim.idx.ravel()]
    vals, counts = np.unique(eras, return_counts=True)
    era_comp = {str(v): float(c / eras.size) for v, c in zip(vals, counts)}
    return {"labels": labels, "years": sorted({l[:4] for l in labels}), "assets": assets, "macro": macro,
            "corr": summary.correlation(sim.total, sim.assets), "era_composition": era_comp,
            "representative_path": k}


def _attr_terms(att, kind: str, j: int) -> list[str]:
    """The terms that apply to one asset, in display order. Scenario overrides, income and the floor are
    listed only when they move this asset at all (no asset_drift, no income source, no floored month)."""
    if kind == "model":
        keep = list(att.factor_terms) + [t for t in ("intercept", "unattributed", "cma", "overrides",
                                                     "valuation_pull", "residual", "income") if t in att.terms]
    elif kind == "cash":
        keep = ["income"]
    else:
        keep = ["y10" if kind == "treasury" else "oil", "income"]
    keep.append("floor")
    optional = {"overrides", "income", "floor"} - ({"income"} if kind == "cash" else set())
    return [t for t in keep if t not in optional or np.any(att.get(t)[:, :, j] != 0.0)]


def _attr_block(vals: np.ndarray, total: np.ndarray, terms: list[str], ti: dict, factor: list[str],
                cagr: np.ndarray | float) -> dict:
    """Means over paths of one asset's terms (vals: (n, K) per path), with the factor part's band and the
    reconciliation from the arithmetic sum to the median CAGR."""
    out = {"terms": {t: float(vals[:, ti[t]].mean()) for t in terms}}
    if factor:
        f = vals[:, [ti[t] for t in factor]].sum(axis=1)
        lo, hi = np.percentile(f, ATTRIBUTION_BAND)
        out.update({"factor_sum": float(f.mean()), "factor_band": [float(lo), float(hi)]})
    arith = float(total.mean())
    med = float(np.percentile(cagr, 50)) if isinstance(cagr, np.ndarray) else float(cagr)
    out.update({"arithmetic": arith, "cagr_p50": med, "reconciliation": med - arith})
    return out


def attribution_summary(sim: SimOutput, phases: list[dict], assets: dict) -> dict[str, dict]:
    """{asset: assets[a].attribution} from sim.attribution: per phase and over the whole path, the mean over
    paths of every term (annualized arithmetic contributions that add up to `arithmetic`), the 10-90% band
    of the factor part, and `reconciliation` = median CAGR - arithmetic (compounding, volatility drag and
    median vs mean), so the parts add up to the median CAGR the page shows (`total.cagr_p50` is the
    stats' cagr_p50)."""
    att = sim.attribution
    if att is None:
        return {}
    months = np.asarray(att.phase_months, float)
    T = float(months.sum())
    w = months / T if T else months
    W = summary.wealth_paths(sim.total)                                  # as summarize() builds it
    starts = np.concatenate([[0], np.cumsum(att.phase_months)[:-1]]).astype(int)
    ti = {t: k for k, t in enumerate(att.terms)}
    out = {}
    for j, a in enumerate(att.assets):
        kind = att.kinds[a]
        terms = _attr_terms(att, kind, j)
        factor = (list(att.factor_terms) if kind == "model" else
                  [] if kind == "cash" else ["y10" if kind == "treasury" else "oil"])
        rows = []
        for p, (ph, s0) in enumerate(zip(phases, starts)):
            m = int(att.phase_months[p])
            growth = W[:, s0 + m, j] / W[:, s0, j]
            block = _attr_block(att.values[:, p, j, :], att.total[:, p, j], terms, ti, factor,
                                growth ** (12.0 / m) - 1.0)
            rows.append({"name": ph.get("name", f"阶段{p + 1}"), "start_month": int(s0), "months": m, **block})
        whole = (att.values[:, :, j, :] * w[None, :, None]).sum(axis=1)
        total = _attr_block(whole, (att.total[:, :, j] * w[None, :]).sum(axis=1), terms, ti, factor,
                            assets[a]["stats"]["cagr_p50"])
        method = "structural" if kind != "model" else att.method
        out[a] = {"kind": kind, "method": method, "units": ATTRIBUTION_UNITS, "phases": rows,
                  "total": {"months": int(T), **total}}
        if method == "unattributed" and att.reason:
            out[a]["reason"] = att.reason
    return out


def prior_correlation(ctx: EngineContext) -> dict:
    cols = [f"r_{a}" for a in config.ASSET_IDS]
    X = ctx.db.frame[cols].to_numpy(float)
    c = weighted_corr(X, ctx.db.prior())
    return {"assets": list(config.ASSET_IDS), "matrix": [[round(float(v), 3) for v in row] for row in c]}


def run_spec(ctx: EngineContext, spec: dict, n_paths: int = 600, seed: int = 0,
             source: str = "builtin", attribution: bool | None = None,
             regimes: bool | None = None) -> tuple[dict, SimOutput]:
    """(detail, SimOutput) of one spec; attribution (default ATTRIBUTION) adds assets[a].attribution, regimes
    (default REGIME_READOUT) adds detail["regimes"]. Neither changes a simulated number.

    The regime checks of the pooled phases (regime_views.spec_checks: what the month database can deliver
    for a regime request, and runs of contraction-heavy phases) are appended to validation.warnings.
    """
    attribution = ATTRIBUTION if attribution is None else bool(attribution)
    readout = REGIME_READOUT if regimes is None else bool(regimes)
    clean, report = validate_spec(spec, ctx.db.frame, config.HORIZON_MONTHS)
    pools = phase_pools(ctx, clean, report["warnings"])
    report["warnings"].extend(regime_views.spec_checks(ctx.db, clean, pools, report["clipped"]))
    weights, phase_reports = [p.blocks for p in pools], _phase_reports(ctx, clean, pools)
    overrides = with_oil_lock([p["views"] for p in clean["phases"]], [p.get("overrides", {}) for p in clean["phases"]])
    cma, vm = valuation_for(ctx)
    sim = simulate(ctx.db, ctx.anchor, weights, [p["months"] for p in clean["phases"]],
                   n_paths=n_paths, seed=seed, block=BLOCK, cma=cma, overrides=overrides,
                   reversion=clean.get("reversion") or None, valuation=vm, attribution=attribution, **RESIDUALS)
    nodes, seen = [], set()
    for e in clean["causal_chain"]:
        for n in (e["from"], e["to"]):
            if n not in seen:
                seen.add(n)
                nodes.append({"id": n, "label": n})
    detail = {
        "id": clean["id"], "name": clean["name"], "summary": clean["summary"],
        "narrative": clean["narrative"], "tags": clean["tags"], "source": source,
        "prior_prob": clean["prior_prob"], "analogs": clean["analogs"],
        "spec": clean, "validation": report,
        "causal_chain": {"nodes": nodes, "edges": clean["causal_chain"]},
        "phases": phase_reports,
        "start": sim.start, "n_paths": n_paths,
        "corr_prior": prior_correlation(ctx),
    }
    detail.update(summarize(ctx, sim, TRACKED_MACRO if vm is not None and vm.kappa == 0 else VALUATION_MACRO))
    for a, att in attribution_summary(sim, phase_reports, detail["assets"]).items():
        detail["assets"][a]["attribution"] = att
    if readout and regime_views.available(ctx):
        detail["regimes"] = regime_views.readout(ctx, sim, clean, reference_simulation(ctx), report["clipped"])
    return detail, sim


# the engine's unconditional reference run, per month database (reference_simulation); at most
# _REFERENCE_KEEP settings per database, the least recently used going first (each run holds about 72 MB)
_REFERENCE_CACHE: "weakref.WeakKeyDictionary[MonthDB, dict]" = weakref.WeakKeyDictionary()
_REFERENCE_LOCK = threading.Lock()
_REFERENCE_KEEP = 2


def reference_simulation(ctx: EngineContext) -> dict:
    """The engine's own unconditional run: the spec {phases: [{name: 回归常态, months: HORIZON_MONTHS,
    views: []}]} simulated exactly as run_spec simulates it, with config.REGIMES["reference"] paths and
    seed, and without a detail, a readout or attribution (which change no simulated number).

    Scenario regime mixes are compared with it rather than with history, since the engine's own CPI drift
    and contraction dynamics are in both. Cached through a weakref to ctx.db and keyed by the anchor date,
    the CMA, the paths and seed and this module's RESIDUALS and VALUATION, so a study that swaps them gets
    its own reference; repeated calls return the same dict (one run per process, about 1 s). A database
    keeps the references of its _REFERENCE_KEEP most recently used settings.

    Returns a dict to treat as read-only (its arrays are not writeable):
      sim      SimOutput, (n_paths, months) returns and (n_paths, months + 1) macro paths
      regimes  the label, flag and episode arrays (regimes.LABEL_KEYS) of regimes.classify_sim of the
               paths with the panel history before the anchor: (n_paths, months + 1), column 0 = the
               anchor month (the continuous indicators are left out; they follow from sim.macro)
      drawn    (n_paths, months) ex-post label codes of the historical month each simulated month was drawn
               from (regimes.label_codes of ctx.db.frame at sim.idx; -1 where unknown)
      n_paths, seed, months, spec (the validated spec)
    """
    ref = config.REGIMES["reference"]
    n_paths, seed = int(ref["n_paths"]), int(ref["seed"])
    cma = ctx.cma or {}
    key = (ctx.anchor.date, n_paths, seed, config.HORIZON_MONTHS,
           tuple(sorted((k, str(v)) for k, v in RESIDUALS.items())),
           tuple(sorted((k, str(v)) for k, v in VALUATION.items())),
           cma.get("method"), tuple(sorted((cma.get("adjustment") or {}).items())))
    with _REFERENCE_LOCK:
        cache = _REFERENCE_CACHE.setdefault(ctx.db, {})
        if key in cache:
            cache[key] = cache.pop(key)                 # most recently used last
            return cache[key]
        spec = {"id": "reference", "name": "引擎无条件抽样", "prior_prob": 0.0,
                "phases": [{"name": "回归常态", "months": config.HORIZON_MONTHS, "views": []}]}
        clean, report = validate_spec(spec, ctx.db.frame, config.HORIZON_MONTHS)
        weights, _ = phase_weights(ctx, clean, report["warnings"])
        months = [p["months"] for p in clean["phases"]]
        overrides = with_oil_lock([p["views"] for p in clean["phases"]],
                                  [p.get("overrides", {}) for p in clean["phases"]])
        sim_cma, vm = valuation_for(ctx)
        sim = simulate(ctx.db, ctx.anchor, weights, months, n_paths=n_paths, seed=seed, block=BLOCK, cma=sim_cma,
                       overrides=overrides, reversion=clean.get("reversion") or None, valuation=vm,
                       attribution=False, **RESIDUALS)
        full = regimes.classify_sim(sim.macro, ctx.panel, anchor=ctx.anchor.date)
        r = {k: full[k] for k in regimes.LABEL_KEYS}
        frame = ctx.db.frame
        codes = (regimes.label_codes(frame) if all(f"rg_{i}" in frame.columns for i in regimes.LABEL_IDS)
                 else np.full(len(frame), -1, np.int8))
        drawn = codes[sim.idx]
        for a in (sim.total, sim.income, sim.idx, drawn, *sim.macro.values(), *r.values()):
            a.flags.writeable = False
        out = {"sim": sim, "regimes": r, "drawn": drawn, "n_paths": n_paths, "seed": seed,
               "months": int(sum(months)), "spec": clean}
        while len(cache) >= _REFERENCE_KEEP:
            del cache[next(iter(cache))]
        cache[key] = out
        return out


def headline(detail: dict) -> dict:
    eq = detail["assets"]["us_equity"]["stats"]
    cpi = detail["macro"].get("cpi_yoy", {}).get("fan", {}).get("p50", [])
    ff = detail["macro"].get("ff", {}).get("fan", {}).get("p50", [])
    pdp = detail["macro"].get("pd", {}).get("fan", {}).get("p50", [])
    return {"us_equity_cagr_p50": eq["cagr_p50"], "us_equity_real_cagr_p50": eq["real_cagr_p50"],
            "cpi_avg": float(np.mean(cpi[1:])) if cpi else None,
            "ff_end_p50": ff[-1] if ff else None, "pd_end_p50": pdp[-1] if pdp else None,
            "ens_min": min(p["ep"]["ens"] for p in detail["phases"])}
