"""Run one scenario spec through the real engine.

spec -> validate_spec -> per-phase entropy pooling -> simulate -> chart-ready
detail JSON (fans, monthly and yearly K-lines, macro paths, posterior
correlations, the evidence behind the weights). Used for the built-in library,
LLM-translated scenarios and user-edited specs alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..context import EngineContext
from ..engine import summary
from ..engine.ep import apply_views, weighted_corr
from ..engine.simulate import SimOutput, simulate
from .schema import validate_spec

MIN_ENS = 20.0          # below this many effective months a view set is too extreme
MACRO_VARS = ("cpi_yoy", "ff", "y10", "unrate", "oil", "spread")


def _top_months(dates: pd.DatetimeIndex, q: np.ndarray, p: np.ndarray, k: int = 10) -> list[dict]:
    ratio = q / p
    order = np.argsort(-q)[:k]
    return [{"date": dates[i].strftime("%Y-%m"), "weight_ratio": float(ratio[i]), "weight": float(q[i])}
            for i in order]


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


def phase_weights(ctx: EngineContext, spec: dict, warnings: list[str]) -> tuple[list[np.ndarray], list[dict]]:
    db = ctx.db
    prior = db.prior()
    weights, reports = [], []
    start = 0
    for k, ph in enumerate(spec["phases"]):
        if ph["views"]:
            res = apply_views(db.frame, ph["views"], p=prior, confidence=ph.get("confidence", 1.0))
            if res.ens < MIN_ENS and ph.get("confidence", 1.0) > 0.7:
                warnings.append(f"阶段 {k + 1}「{ph['name']}」的视图组合在历史上只有约 {res.ens:.0f} 个有效月份支撑，"
                                f"已把置信度降到 0.7 以免过拟合少数月份")
                res = apply_views(db.frame, ph["views"], p=prior, confidence=0.7)
            q = res.q
            ep = {"ens": res.ens, "ens_ratio": res.ens_ratio, "kl": res.kl, "converged": res.converged,
                  "max_violation": res.max_violation, "views": res.views}
        else:
            q = prior
            ep = {"ens": float(len(prior)), "ens_ratio": 1.0, "kl": 0.0, "converged": True,
                  "max_violation": 0.0, "views": []}
        ep["era_weights"] = db.era_weights(q)
        ep["top_months"] = _top_months(db.dates, q, prior)
        weights.append(q)
        reports.append({"name": ph["name"], "start_month": start, "months": ph["months"],
                        "views": ph["views"], "overrides": ph.get("overrides", {}),
                        "confidence": ph.get("confidence", 1.0), "ep": ep})
        start += ph["months"]
    return weights, reports


def summarize(ctx: EngineContext, sim: SimOutput) -> dict:
    """Chart-ready summaries of a simulation (shared with backtests)."""
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
    for m in MACRO_VARS:
        if m in sim.macro:
            macro[m] = {"fan": summary.fan(sim.macro[m].astype(float))}
    eras = ctx.db.eras[sim.idx.ravel()]
    vals, counts = np.unique(eras, return_counts=True)
    era_comp = {str(v): float(c / eras.size) for v, c in zip(vals, counts)}
    return {"labels": labels, "years": sorted({l[:4] for l in labels}), "assets": assets, "macro": macro,
            "corr": summary.correlation(sim.total, sim.assets), "era_composition": era_comp,
            "representative_path": k}


def prior_correlation(ctx: EngineContext) -> dict:
    cols = [f"r_{a}" for a in config.ASSET_IDS]
    X = ctx.db.frame[cols].to_numpy(float)
    c = weighted_corr(X, ctx.db.prior())
    return {"assets": list(config.ASSET_IDS), "matrix": [[round(float(v), 3) for v in row] for row in c]}


def run_spec(ctx: EngineContext, spec: dict, n_paths: int = 600, seed: int = 0,
             source: str = "builtin") -> tuple[dict, SimOutput]:
    clean, report = validate_spec(spec, ctx.db.frame, config.HORIZON_MONTHS)
    weights, phase_reports = phase_weights(ctx, clean, report["warnings"])
    sim = simulate(ctx.db, ctx.anchor, weights, [p["months"] for p in clean["phases"]],
                   n_paths=n_paths, seed=seed, cma=ctx.cma,
                   overrides=[p.get("overrides", {}) for p in clean["phases"]],
                   reversion=clean.get("reversion") or None)
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
    detail.update(summarize(ctx, sim))
    return detail, sim


def headline(detail: dict) -> dict:
    eq = detail["assets"]["us_equity"]["stats"]
    cpi = detail["macro"].get("cpi_yoy", {}).get("fan", {}).get("p50", [])
    ff = detail["macro"].get("ff", {}).get("fan", {}).get("p50", [])
    return {"us_equity_cagr_p50": eq["cagr_p50"], "us_equity_real_cagr_p50": eq["real_cagr_p50"],
            "cpi_avg": float(np.mean(cpi[1:])) if cpi else None,
            "ff_end_p50": ff[-1] if ff else None,
            "ens_min": min(p["ep"]["ens"] for p in detail["phases"])}
