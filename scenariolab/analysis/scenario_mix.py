"""未来一年：情景条件模拟 (SIM_DESIGN §9; §14 P8): the built-in scenarios' next four quarters, mixed by probability.

This is not a forecast. The only predictive layer is the scenario probabilities (Layer 1); each scenario's four
quarters are a simulation given its hypothesis. The forecast tab mixes them with the probabilities the reader sets.

Engines (one block each in the output):
  monthly    the monthly engine's scenario paths (scenario_paths.npz, scenario_macro.npz, scenarios/<id>.json and
             scenarios.json of the scenarios build step), aggregated to quarter-ends (months 3, 6, 9, 12). It is the
             default until the switch gate T-S passes (§4.6, U-1).
  quarterly  the quarterly stack (试运行): each reviewed built-in QTable (scenarios/qtables/) in path mode over all its
             quarters (two-sided conditioning: later statements shape the first four), through rollout's public API.

Per engine and scenario:
  assets  each asset's cumulative total return (simple, decimal) at the four quarter-ends as a quantile grid at GRID
          percentiles (0 and 100 = the path minimum and maximum), its mean and P(loss); the page inverts the mixture
          CDF sum_i w_i F_i (piecewise linear in the grids), so any probabilities give exact mixture quantiles of the
          interpolated distributions. mix_quantile() is the same rule in Python (the ledger and the tests use it).
  macro   policy rate, unemployment, CPI y/y and Baa - 10y at the quarter-ends and the cumulative oil change, as
          p10 / p50 / p90 (monthly engine: p5 / p50 / p95 of its stored fans)
  rates   the three base-rate indicators of §7.5 on one year: a contraction within the four quarters (the regime
          classifier; the monthly engine's own monthly labels, the quarterly stack's labels on quarter-end series),
          CPI y/y above 4% at the fourth quarter-end, the policy rate up at least 1 pp by then
  finite  the share of paths whose four quarters are finite: a scenario without finite paths is left out of the
          mixture (named in the notes, its probability spread over the others in proportion)

Also: what each scenario's QTable states for the four quarters (qmodel.qview.cells: badges 文字数字 / 文字定性 / 推算 /
模型补全 / 锚 / 干预). That is what the quarterly stack runs; the monthly engine runs the scenario library's own phase
specs (their views and overrides, which the page reads from the scenario detail), of which the QTable is a conversion
that drops some records (scenarios/qtables/<id>.json conversion[]). Also the unconditional quarterly band of each engine
(无情景基准: the quarterly stack without statements; the monthly engine's reference run), the analog method's 12-month
band where it differs from the unconditional one, the history base rates (quarter-end series since 1973), and the
forward ledger (前瞻账本): every build appends the mixture's quarterly quantiles at the default probabilities to
data/results/ledger/forecast_ledger.jsonl (append only, one line per origin / engine / model / card / probability
version, never rewritten) and scores the entries whose quarters have arrived (coverage of the 80% band, PIT, CRPS
from the quantile grid). Probabilities: the hand-set prior_prob of each scenario, normalised (§7.5, U-9); the
engine's posterior tilt by today's state is offered as an alternative, off by default.

    python -m scenariolab.analysis.scenario_mix [--paths N] [--seed S] [--no-ledger]

Writes data/results/forecast_mix.json. No LLM call, no download.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .. import config, jsonio

VERSION = 1
QUARTERS = 4
GRID = (0.0, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0,
        70.0, 75.0, 80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5, 100.0)
MIN_FINITE = 50                   # fewer finite paths than this: no grid (the scenario is left out of the mixture)
OUTPUT = "forecast_mix.json"
LEDGER_DIR = "ledger"
LEDGER_FILE = "forecast_ledger.jsonl"
DEFAULT_PATHS = 1000
DEFAULT_SEED = 20261104
ANALOG_DIFF = 0.02                # the analog band "differs" when a 12-month p5 / p50 / p95 differs by 2 pp or more
LEDGER_PROBS = (10.0, 50.0, 90.0)
BAND = (10.0, 90.0)               # the 80% band that the ledger's coverage scores
ENGINE_ZH = {"monthly": "月度引擎（季末汇总）", "quarterly": "季度模型（试运行）"}
LEDGER_NOTE = "需要很多年才能分辨中等程度的偏差：一季只有一个实现值，一年只有四个。"


# ================================================================ quantile grids and mixtures

def quantile_grid(x) -> list | None:
    """GRID percentiles of the finite values of x (None when fewer than MIN_FINITE are finite)."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size < MIN_FINITE:
        return None
    return [round(float(q), 5) for q in np.percentile(v, GRID)]


def grid_cdf(grid, x: float) -> float:
    """F(x) of the distribution whose quantiles at GRID are `grid`, linear between grid points (0 below, 1 above)."""
    g = np.asarray(grid, float)
    p = np.asarray(GRID, float) / 100.0
    if x < g[0]:
        return 0.0
    if x >= g[-1]:
        return 1.0
    k = int(np.searchsorted(g, x, side="right"))          # g[k-1] <= x < g[k]
    lo, hi = g[k - 1], g[k]
    if hi <= lo:
        return float(p[k])
    return float(p[k - 1] + (p[k] - p[k - 1]) * (x - lo) / (hi - lo))


def mix_cdf(grids, weights, x: float) -> float:
    w = np.asarray(weights, float)
    return float(sum(wi * grid_cdf(g, x) for g, wi in zip(grids, w)) / w.sum())


def mix_quantile(grids, weights, prob: float) -> float | None:
    """The prob-quantile (0..1) of the mixture sum_i w_i F_i, exactly: the mixture CDF is linear between the sorted
    union of the grids' knots, so a binary search over the knots finds the segment where it reaches prob and the
    quantile is interpolated inside it (a bisection on the values would not converge when a divergent path puts a
    grid's maximum at 1e40). The page's mixQuantile (web/js/tabs/forecast.js) is the same rule."""
    pairs = [(g, w) for g, w in zip(grids, weights) if g is not None and w > 0]
    if not pairs:
        return None
    gs, ws = [g for g, _ in pairs], [w for _, w in pairs]
    knots = np.unique(np.concatenate([np.asarray(g, float) for g in gs]))
    if knots.size == 1:
        return float(knots[0])
    lo, hi = 0, knots.size - 1                       # smallest j with F(knots[j]) >= prob
    while lo < hi:
        mid = (lo + hi) // 2
        if mix_cdf(gs, ws, float(knots[mid])) >= prob:
            hi = mid
        else:
            lo = mid + 1
    if lo == 0:
        return float(knots[0])
    x0, x1 = float(knots[lo - 1]), float(knots[lo])
    f0, f1 = mix_cdf(gs, ws, x0), mix_cdf(gs, ws, x1)
    return float(x0 + (prob - f0) * (x1 - x0) / (f1 - f0)) if f1 > f0 else x1


def weights_for(probs: dict, ids: list, available: dict) -> dict:
    """Probabilities of the scenarios in `ids` that have a result (available[id] True), renormalised."""
    w = {i: max(float(probs.get(i) or 0.0), 0.0) for i in ids if available.get(i)}
    s = sum(w.values())
    return {i: (v / s if s > 0 else 0.0) for i, v in w.items()}


def crps_grid(grid, y: float) -> float:
    """CRPS of the quantile-grid distribution at y: CRPS = 2 * integral over tau of the pinball loss of the
    tau-quantile (the quantile decomposition, Gneiting & Ranjan 2011), by the trapezoid rule on the GRID levels
    (0 and 100 included: the path minimum and maximum)."""
    taus = np.asarray(GRID, float) / 100.0
    q = np.asarray(grid, float)
    loss = np.where(y >= q, taus * (y - q), (1.0 - taus) * (q - y))
    return float(2.0 * np.sum(0.5 * (loss[1:] + loss[:-1]) * np.diff(taus)))


# ================================================================ helpers

def _f(x, d: int = 6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _band(x, lo: float = 10.0, hi: float = 90.0) -> dict | None:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size < MIN_FINITE:
        return None
    a, m, b = np.percentile(v, [lo, 50.0, hi])
    return {f"p{lo:g}": _f(a), "p50": _f(m), f"p{hi:g}": _f(b)}


def _share(mask) -> float | None:
    m = np.asarray(mask)
    return _f(m.mean(), 4) if m.size else None


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _asset_block(cum: np.ndarray, assets: list) -> tuple[dict, float]:
    """cum (n, 4, A) cumulative simple total returns -> per asset {grid: [4 x grid], mean, prob_loss}, finite share."""
    finite = np.isfinite(cum).all(axis=(1, 2))
    out = {}
    for j, a in enumerate(assets):
        grids, means, loss = [], [], []
        for k in range(cum.shape[1]):
            col = cum[:, k, j]
            col = col[np.isfinite(col)]
            g = quantile_grid(col)
            grids.append(g)
            means.append(_f(col.mean()) if col.size >= MIN_FINITE else None)
            loss.append(_f((col < 0).mean(), 4) if col.size >= MIN_FINITE else None)
        out[a] = {"grid": grids, "mean": means, "prob_loss": loss}
    return out, float(finite.mean()) if finite.size else 0.0


# ================================================================ the monthly engine (scenario paths at quarter-ends)

def monthly_engine(results: Path) -> dict:
    """Per built-in scenario: the four quarter-ends of the monthly engine's stored paths (the scenarios step)."""
    z = np.load(results / "scenario_paths.npz", allow_pickle=False)
    ids = [str(i) for i in z["ids"]]
    assets = [str(a) for a in z["assets"]]
    total, cpi = z["total"], z["cpi"]
    months = [3 * k for k in range(1, QUARTERS + 1)]
    try:
        mz = np.load(results / "scenario_macro.npz", allow_pickle=False)
        ff = mz["ff"] if [str(i) for i in mz["ids"]] == ids else None
    except (OSError, KeyError, ValueError):
        ff = None
    index = jsonio.load(results / "scenarios.json")
    by_id = {s["id"]: s for s in index.get("scenarios", [])}
    out = {}
    for s, sid in enumerate(ids):
        w = np.cumprod(1.0 + total[s, :, :months[-1], :].astype(np.float64), axis=1)
        cum = w[:, [m - 1 for m in months], :] - 1.0
        block, finite = _asset_block(cum, assets)
        c = cpi[s].astype(np.float64)
        cpi_q4 = 100.0 * (c[:, months[-1]] / c[:, 0] - 1.0)
        rates = {"cpi_gt4_q4": _share(cpi_q4 > 4.0)}
        if ff is not None:
            rates["ff_up1_q4"] = _share(ff[s, :, months[-1]] - ff[s, :, 0] >= 1.0)
        reg = (by_id.get(sid) or {}).get("regimes") or {}
        if isinstance(reg.get("p_contraction_year1"), (int, float)):
            rates["contraction_4q"] = _f(reg["p_contraction_year1"], 4)
        macro = {}
        try:
            det = jsonio.load(results / "scenarios" / f"{sid}.json")
            fans = det.get("macro") or {}
            for key, mid in (("ff", "ff"), ("unrate", "unrate"), ("cpi_yoy", "cpi_yoy"), ("spread", "spread")):
                fan = (fans.get(mid) or {}).get("fan") or {}
                if fan.get("p50"):
                    macro[key] = {p: [_f(fan[p][m]) for m in months] for p in ("p5", "p50", "p95") if fan.get(p)}
            oil = (fans.get("oil") or {}).get("fan") or {}
            if oil.get("p50") and oil["p50"][0]:
                o0 = float(oil["p50"][0])
                macro["oil"] = {p: [_f(oil[p][m] / o0 - 1.0) for m in months] for p in ("p5", "p50", "p95")
                                if oil.get(p)}
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            pass
        out[sid] = {"assets": block, "finite": _f(finite, 4), "rates": rates, "macro": macro,
                    "n_paths": int(total.shape[1]), "band": "p5_p95"}
    return {"assets": assets, "scenarios": out, "labels": [str(x) for x in z["labels"][:months[-1] + 1]],
            "fingerprint": _sha({"ids": ids, "first": total[:, :2, :12, :].astype(float).round(6).tolist()})}


def monthly_reference(ctx, assets: list, results: Path | None = None) -> dict | None:
    """The monthly engine's unconditional reference run (scenarios/runner.reference_simulation) at quarter-ends."""
    if ctx is None:
        return None
    from ..scenarios import runner
    ref = runner.reference_simulation(ctx)
    sim = ref["sim"]
    order = [list(sim.assets).index(a) for a in assets]
    months = [3 * k for k in range(1, QUARTERS + 1)]
    w = np.cumprod(1.0 + np.asarray(sim.total, float)[:, :months[-1], :][:, :, order], axis=1)
    cum = w[:, [m - 1 for m in months], :] - 1.0
    rates = {}
    macro = getattr(sim, "macro", None) or {}
    if "cpi" in macro:
        c = np.asarray(macro["cpi"], float)
        rates["cpi_gt4_q4"] = _share(100.0 * (c[:, months[-1]] / c[:, 0] - 1.0) > 4.0)
    if "ff" in macro:
        f = np.asarray(macro["ff"], float)
        rates["ff_up1_q4"] = _share(f[:, months[-1]] - f[:, 0] >= 1.0)
    try:                                   # the regime layer's readout of the same reference run (scenarios.json)
        ref = (jsonio.load(Path(results or config.RESULTS_DIR) / "scenarios.json").get("regimes") or {}).get("reference") or {}
        if isinstance(ref.get("p_contraction_year1"), (int, float)):
            rates["contraction_4q"] = _f(ref["p_contraction_year1"], 4)
    except (OSError, ValueError):
        pass
    return {"assets": {a: {f"p{p:g}": [_f(np.percentile(cum[:, k, j], p)) for k in range(QUARTERS)]
                           for p in (10.0, 50.0, 90.0)} for j, a in enumerate(assets)},
            "rates": rates, "n_paths": int(cum.shape[0]), "name": "引擎无条件抽样（不加任何情景视图）"}


# ================================================================ the quarterly stack (试运行)

def _contraction_4q(stack, roll) -> np.ndarray:
    """Per path: a contraction in the first four quarters (the regime classifier on quarter-end CPI y/y,
    unemployment and the IP level, with the history before the origin first; factor_checks' quarterly params)."""
    from ..engine import regimes as RG
    from ..qmodel import factor_checks as FC
    import pandas as pd
    q0 = pd.Period(roll.labels[0], "Q")
    Mh = FC.quarter_end(stack.P).loc["1960Q2":q0 - 1]
    k = len(Mh)
    n = roll.n
    cpi = np.asarray(roll.fx.levels["cpi_yoy"], float)
    u = np.asarray(roll.fx.levels["unrate"], float)
    lip = math.log(float(Mh["lvl_ip"].iloc[-1])) + np.cumsum(np.asarray(roll.fx.x["d_ip"], float) / 100.0, axis=1)
    full = RG.classify(np.concatenate([np.broadcast_to(Mh["s_cpi_yoy"].to_numpy(float), (n, k)), cpi], 1),
                       np.concatenate([np.broadcast_to(Mh["lvl_unrate"].to_numpy(float), (n, k)), u], 1),
                       np.concatenate([np.broadcast_to(Mh["lvl_ip"].to_numpy(float), (n, k)), np.exp(lip)], 1),
                       params=FC.QP)
    con = np.asarray(full["contraction"], bool)[:, k:k + QUARTERS]
    return con.any(axis=1)


def _quarterly_block(stack, roll, assets: list) -> dict:
    """The first four quarters of a quarterly-stack run. Paths the stack itself flags as diverging in those quarters
    (rollout.tail_flags: GZ > 15, unemployment > 20% or a quarterly equity log return beyond ±100%, "不可信") are
    left out of the asset grids and means, and their share is recorded (tail_4q)."""
    from ..qmodel import rollout as R
    tot = np.asarray(roll.ax.total, float)[:, :QUARTERS, :]
    lv0 = roll.fx.levels
    tail = R.tail_flags(np.asarray(lv0["gz"], float)[:, :QUARTERS], np.asarray(lv0["unrate"], float)[:, :QUARTERS],
                        tot[:, :, list(R.ASSETS).index("us_equity")]).any(axis=1)
    cum = np.expm1(np.cumsum(tot, axis=1))
    finite_all = np.isfinite(cum).all(axis=(1, 2))
    block, _ = _asset_block(cum[~tail], assets)
    finite = float((finite_all & ~tail).mean()) if finite_all.size else 0.0
    lv = roll.fx.levels
    macro = {}
    for key in ("ff", "unrate", "cpi_yoy", "spread", "y10", "gz"):
        if key in lv:
            arr = np.asarray(lv[key], float)[:, :QUARTERS]
            macro[key] = {p: [_f(np.nanpercentile(arr[:, k], q)) for k in range(QUARTERS)]
                          for p, q in (("p10", 10.0), ("p50", 50.0), ("p90", 90.0))}
    oil = np.expm1(np.cumsum(np.asarray(roll.fx.x["d_oil"], float)[:, :QUARTERS], axis=1) / 100.0)
    macro["oil"] = {p: [_f(np.nanpercentile(oil[:, k], q)) for k in range(QUARTERS)]
                    for p, q in (("p10", 10.0), ("p50", 50.0), ("p90", 90.0))}
    ff = np.asarray(lv["ff"], float)
    ff0 = np.asarray(roll.states["s_ff"], float)[:, 0]
    rates = {"cpi_gt4_q4": _share(np.asarray(lv["cpi_yoy"], float)[:, QUARTERS - 1] > 4.0),
             "ff_up1_q4": _share(ff[:, QUARTERS - 1] - ff0 >= 1.0)}
    try:
        rates["contraction_4q"] = _share(_contraction_4q(stack, roll))
    except Exception as err:  # noqa: BLE001 - one indicator; the rest of the block stands
        rates["contraction_error"] = f"{type(err).__name__}: {err}"
    return {"assets": block, "finite": _f(finite, 4), "tail_4q": _f(tail.mean(), 4), "rates": rates, "macro": macro,
            "n_paths": int(roll.n), "band": "p10_p90"}


def quarterly_engine(n_paths: int = DEFAULT_PATHS, seed: int = DEFAULT_SEED) -> dict:
    """Per built-in QTable: the quarterly stack in path mode over all its quarters, the first four summarised; and
    the stack without statements (the unconditional band)."""
    from ..qmodel import qtable as QT
    from ..qmodel import rollout as R
    stack = R.get_stack()
    assets = list(R.ASSETS)
    out, notes = {}, []
    for sid in QT.builtin_ids():
        env = QT.builtin(sid)
        try:
            read = R.from_qtable(env["qtable"], stack, "production", narrative=env["text"], source="builtin")
            roll = R.run_path(stack, read["statements"], read["valuation"], H=read["H"], n_paths=n_paths, seed=seed,
                              cycle_odds=read.get("cycle_odds"), mode="production")
        except Exception as err:  # noqa: BLE001 - one scenario fails, the mixture says so and goes on
            out[sid] = {"error": f"{type(err).__name__}: {err}", "finite": 0.0}
            continue
        blk = _quarterly_block(stack, roll, assets)
        blk["tail_share"] = _f((roll.info or {}).get("tail_share"), 4)
        blk["scenario_quarters"] = int(read["H"])
        out[sid] = blk
    unc = R.run_path(stack, [], [], H=QUARTERS, n_paths=n_paths, seed=seed)
    ub = _quarterly_block(stack, unc, assets)
    reference = {"assets": {a: {k: [None if g is None else _f(np.interp(k_p, GRID, g)) for g in v["grid"]]
                                for k, k_p in (("p10", 10.0), ("p50", 50.0), ("p90", 90.0))}
                            for a, v in ub["assets"].items()},
                 "macro": ub["macro"], "rates": ub["rates"], "n_paths": ub["n_paths"],
                 "name": "季度模型不加任何陈述（无条件）"}
    return {"assets": assets, "scenarios": out, "reference": reference, "version": stack.version,
            "origin": str(stack.origin), "train_end": str(stack.train_end), "labels": list(unc.labels),
            "notes": notes, "stack": stack}


# ================================================================ history base rates (quarter-end series)

def history_rates(P, end: str | None = None) -> dict:
    """Shares of origins since 1973Q1 with, in their next four quarters, a contraction (the quarterly classifier),
    CPI y/y > 4% at the fourth quarter-end and the policy rate up at least 1 pp by then."""
    from ..engine import regimes as RG
    from ..qmodel import factor_checks as FC
    M = FC.quarter_end(P)
    if end:
        M = M.loc[:end]
    r = RG.classify(M["s_cpi_yoy"].to_numpy(float), M["lvl_unrate"].to_numpy(float), M["lvl_ip"].to_numpy(float),
                    params=FC.QP)
    con = np.asarray(r["contraction"], bool)
    idx = M.index
    cpi, ff = M["s_cpi_yoy"].to_numpy(float), M["lvl_ff"].to_numpy(float)
    rows = [t for t in range(1, len(idx) - QUARTERS + 1) if str(idx[t]) >= "1973Q1"]
    c4 = [bool(con[t:t + QUARTERS].any()) for t in rows]
    i4 = [bool(cpi[t + QUARTERS - 1] > 4.0) for t in rows if np.isfinite(cpi[t + QUARTERS - 1])]
    f4 = [bool(ff[t + QUARTERS - 1] - ff[t - 1] >= 1.0) for t in rows
          if np.isfinite(ff[t + QUARTERS - 1]) and np.isfinite(ff[t - 1])]
    return {"contraction_4q": _f(np.mean(c4), 4), "cpi_gt4_q4": _f(np.mean(i4), 4), "ff_up1_q4": _f(np.mean(f4), 4),
            "n_origins": len(rows), "first": str(idx[rows[0]]) if rows else None,
            "last": str(idx[rows[-1]]) if rows else None,
            "note": "历史：1973 年以来每个季度作为起点、之后 4 个季度的季末序列；重叠窗口，独立的一年只有约 "
                    f"{len(rows) // QUARTERS} 个"}


# ================================================================ the analog method (forecast.json), probabilities

def analog_block(results: Path) -> dict | None:
    """The analog method's 12-month band against the unconditional one (forecast.json live), per asset, with
    whether it differs (ANALOG_DIFF); the regime round's paired CRPS check (regime_forecast.json) when present."""
    try:
        fc = jsonio.load(results / "forecast.json")
    except (OSError, ValueError):
        return None
    live = fc.get("live") or {}
    rows = {}
    for a, x in (live.get("assets") or {}).items():
        an, un = x.get("analog") or {}, x.get("uncond") or {}
        if not an or not un:
            continue
        gap = max(abs(float(an.get(p, 0)) - float(un.get(p, 0))) for p in ("p5", "p50", "p95"))
        rows[a] = {"analog": {p: _f(an.get(p)) for p in ("p5", "p50", "p95")},
                   "uncond": {p: _f(un.get(p)) for p in ("p5", "p50", "p95")}, "max_gap": _f(gap, 4),
                   "differs": bool(gap >= ANALOG_DIFF)}
    out = {"origin": live.get("origin"), "flat": live.get("flat"), "assets": rows,
           "differs": sorted(a for a, r in rows.items() if r["differs"])}
    try:
        rf = jsonio.load(results / "regime_forecast.json")
        aa = ((rf.get("baseline") or {}).get("aa_noise") or {})
        ts = [v.get("all", {}).get("t") for v in aa.values() if isinstance(v, dict)]
        ts = [t for t in ts if isinstance(t, (int, float))]
        if ts:
            out["paired_t"] = [round(min(ts), 2), round(max(ts), 2)]
            out["n_seeds"] = len(ts)
    except (OSError, ValueError):
        pass
    return out


def probabilities(results: Path, ids: list) -> dict:
    """The hand-set prior_prob of each built-in (QTable envelopes, the YAML's value), normalised: the default (§7.5,
    U-9). The monthly engine's posterior (scenarios.json prob: prior x likelihood of today's state) as an
    alternative the page offers, off by default."""
    from ..qmodel import qtable as QT
    prior = {}
    for sid in ids:
        try:
            prior[sid] = float(QT.builtin(sid).get("prior_prob") or 0.0)
        except Exception:  # noqa: BLE001 - a scenario without an envelope keeps 0
            prior[sid] = 0.0
    s = sum(prior.values())
    prior = {k: (v / s if s > 0 else 1.0 / len(ids)) for k, v in prior.items()}
    post = {}
    try:
        for x in jsonio.load(results / "scenarios.json").get("scenarios", []):
            if x.get("id") in prior and isinstance(x.get("prob"), (int, float)):
                post[x["id"]] = float(x["prob"])
    except (OSError, ValueError):
        pass
    sp = sum(post.values())
    post = {k: v / sp for k, v in post.items()} if sp > 0 else {}
    return {"default": "prior", "sets": {
        "prior": {"name": "手设概率（默认）", "probs": {k: round(v, 6) for k, v in prior.items()},
                  "text": "每个内置情景的主观先验，归一到合计 1（默认不按今天的状态倾斜）"},
        **({"posterior": {"name": "按今天状态倾斜（默认关闭）", "probs": {k: round(v, 6) for k, v in post.items()},
                          "text": "月度引擎的后验：先验 × 与今天状态的相似度；领先宏观状态还没有通过预注册的预测能力检验，所以默认不用"}}
           if post else {})},
        "version": _sha(prior)[:12]}


# ================================================================ the forward ledger

def _ledger_path(results: Path) -> Path:
    return results / LEDGER_DIR / LEDGER_FILE


def read_ledger(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue                              # a torn line is skipped, never rewritten
    return out


def ledger_entry(mix: dict, engine: str) -> dict | None:
    """The mixture's quarterly quantiles at the default probabilities: one ledger line for one engine."""
    eng = mix["engines"].get(engine)
    if not eng or not eng.get("scenarios"):
        return None
    probs = mix["probabilities"]["sets"][mix["probabilities"]["default"]]["probs"]
    ids = [s["id"] for s in mix["scenarios"]]
    assets = {}
    for a in mix["assets"]:
        per_q = []
        for k in range(QUARTERS):
            grids, ws = [], []
            for sid in ids:
                blk = (eng["scenarios"].get(sid) or {}).get("assets", {}).get(a)
                g = blk["grid"][k] if blk else None
                if g is not None and probs.get(sid, 0) > 0:
                    grids.append(g)
                    ws.append(probs[sid])
            if not grids:
                per_q.append(None)
                continue
            grid = [_f(mix_quantile(grids, ws, p / 100.0)) for p in GRID]
            per_q.append(grid)
        assets[a] = per_q
    key = {"origin": mix["origin"], "engine": engine, "model": eng.get("version"), "card": mix["versions"]["card"],
           "probabilities": mix["probabilities"]["version"]}
    return {"schema": "forecast_ledger/1", "created": mix["created"], **key, "key": _sha(key)[:16],
            "quarters": mix["quarters"], "grid_percentiles": list(GRID), "assets": assets,
            "probs": probs, "default_engine": mix["default_engine"]}


def append_ledger(path: Path, entries: list) -> dict:
    """Append the entries whose key is not in the ledger yet (append only: an existing line is never rewritten)."""
    have = {e.get("key") for e in read_ledger(path)}
    new = [e for e in entries if e and e.get("key") not in have]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for e in new:
                fh.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    return {"appended": len(new), "skipped": len(entries) - len(new)}


def realized_cum(Q, quarters: list, asset: str) -> list:
    """Cumulative simple total return of `asset` over the first k of `quarters` (k = 1..4), from the quarterly
    panel's log total returns (yt_<asset>); None from the first quarter without data."""
    col = f"yt_{asset}"
    out, acc = [], 0.0
    for q in quarters:
        v = Q[col].get(q) if col in Q.columns else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float("nan")
        if not math.isfinite(v):
            out.extend([None] * (len(quarters) - len(out)))
            break
        acc += v
        out.append(math.expm1(acc))
    return out


def score_ledger(entries: list, Q) -> dict:
    """Score the ledger lines whose quarters have arrived: per engine, coverage of the 80% band, PIT and CRPS
    (quantile grid) of the cumulative total return at each matured quarter-end, pooled over the risk assets."""
    from ..qmodel import config_q as cq
    risk = list(cq.RISK_ASSETS)
    Qi = Q.copy()
    Qi.index = [str(i) for i in Qi.index]
    scored, by_engine = [], {}
    for e in entries:
        rec = {"origin": e.get("origin"), "engine": e.get("engine"), "created": e.get("created"),
               "model": e.get("model"), "matured": 0}
        horizons = []
        for a in risk:
            grids = (e.get("assets") or {}).get(a) or []
            real = realized_cum(Qi, e.get("quarters") or [], a)
            for k, (g, y) in enumerate(zip(grids, real)):
                if g is None or y is None:
                    continue
                lo = float(np.interp(BAND[0], GRID, g))
                hi = float(np.interp(BAND[1], GRID, g))
                horizons.append({"k": k + 1, "asset": a, "covered": bool(lo <= y <= hi), "pit": grid_cdf(g, y),
                                 "crps": crps_grid(g, y)})
        if horizons:
            rec["matured"] = max(h["k"] for h in horizons)
            rec["coverage80"] = _f(np.mean([h["covered"] for h in horizons]), 4)
            rec["pit_mad"] = _f(np.mean([abs(h["pit"] - 0.5) for h in horizons]), 4)
            rec["crps"] = _f(np.mean([h["crps"] for h in horizons]), 6)
            b = by_engine.setdefault(e.get("engine"), [])
            b.extend(horizons)
        scored.append(rec)
    summary = {eng: {"n": len(h), "coverage80": _f(np.mean([x["covered"] for x in h]), 4),
                     "pit_mad": _f(np.mean([abs(x["pit"] - 0.5) for x in h]), 4),
                     "crps": _f(np.mean([x["crps"] for x in h]), 6)} for eng, h in by_engine.items() if h}
    return {"entries": scored, "summary": summary}


# ================================================================ assemble

def switch_gate(results: Path) -> dict:
    """The switch gate T-S (§4.6) as the forecast tab needs it: evaluation/switch.py state() (P9), the same switch the
    services and the other tabs read; passed -> the quarterly stack is the default. The page reads the live state too
    (GET /api/evaluation/switch), so a later evaluation does not wait for the next build."""
    from ..evaluation import switch
    st = switch.state(results)
    return {"status": st["status"], "passed": bool(st["passed"]), "text": st["text"],
            "reasons": list(st.get("reasons") or []), "default_text": st.get("default_text")}


def run(ctx=None, *, n_paths: int = DEFAULT_PATHS, seed: int = DEFAULT_SEED, results: Path | None = None,
        ledger: bool = True, write: bool = True) -> dict:
    """Build forecast_mix.json (both engines) and append the forward ledger. ctx: the engine context (the monthly
    engine's reference run); without it the monthly block has no unconditional band."""
    from ..qmodel import card as qcard
    from ..qmodel import config_q as cq
    from ..qmodel import qtable as QT
    from ..qmodel import qview
    results = Path(results or config.RESULTS_DIR)
    names = {}
    ids = QT.builtin_ids()
    for sid in ids:
        env = QT.builtin(sid)
        names[sid] = env.get("name") or sid
    notes = []
    engines = {}
    try:
        mon = monthly_engine(results)
        mon_ids = [i for i in ids if i in mon["scenarios"]]
        missing = [names[i] for i in ids if i not in mon["scenarios"]]
        if missing:
            notes.append(f"月度引擎的情景路径里没有：{'、'.join(missing)}（重新运行 scenarios 构建步骤）")
        try:
            mref = monthly_reference(ctx, mon["assets"], results)
        except Exception as err:  # noqa: BLE001 - the band is a reference line only
            mref = None
            notes.append(f"月度引擎的无条件参照没有算出：{type(err).__name__}")
        engines["monthly"] = {"name": ENGINE_ZH["monthly"], "trial": False,
                              "scenarios": {i: mon["scenarios"][i] for i in mon_ids}, "reference": mref,
                              "version": f"monthly/{mon['fingerprint'][:12]}", "labels": mon["labels"],
                              "text": "月度引擎的情景路径（与「未来情景」同一批）按季末（第 3、6、9、12 个月）汇总"}
    except (OSError, KeyError, ValueError) as err:
        notes.append(f"月度引擎的情景路径读不了（{type(err).__name__}）：先运行 python -m scenariolab.build --only scenarios")
    qeng = quarterly_engine(n_paths=n_paths, seed=seed)
    stack = qeng.pop("stack")
    bad = [i for i, b in qeng["scenarios"].items() if not b.get("assets") or (b.get("finite") or 0) < 0.5]
    qnotes = []
    for i in bad:
        b = qeng["scenarios"][i]
        why = ("运行出错" if b.get("error") else
               f"只有 {round(100 * (b.get('finite') or 0))}% 的路径算出了有限的结果")
        qnotes.append(f"「{names.get(i, i)}」的资产路径这次没有算出来（{why}：季度模型的已知问题，待修正）。"
                      "季度模型的情景加权暂不含它，它的概率按比例分给其余情景。")
    for i, b in qeng["scenarios"].items():
        if i not in bad and (b.get("tail_4q") or 0) >= 0.001:
            qnotes.append(f"「{names.get(i, i)}」有 {100 * b['tail_4q']:.1f}% 的路径在前四季就发散（GZ 信用利差 > 15、失业率 > 20% "
                          "或美股单季对数回报超过 ±100%，模型自己标为不可信）：已从它的区间和均值里去掉。")
    engines["quarterly"] = {"name": ENGINE_ZH["quarterly"], "trial": True, "label": cq.TRIAL_LABEL,
                            "scenarios": qeng["scenarios"], "reference": qeng["reference"],
                            "version": qeng["version"], "labels": qeng["labels"], "excluded": bad, "notes": qnotes,
                            "text": "季度模型（试运行）：每个内置情景的 QTable 在路径模式下运行全部季度（后面季度的陈述也约束前四季），取前四季"}
    if "monthly" not in engines:
        notes.append("月度引擎的结果没有生成：页面只能显示季度模型，它在切换关 T-S 通过之前只是试运行")
    origin = qeng["origin"]
    quarters = list(qeng["labels"])
    gate = switch_gate(results)
    if gate["passed"]:
        engines["quarterly"].update(trial=False, name="季度模型", label="默认",
                                    text=engines["quarterly"]["text"].replace("季度模型（试运行）", "季度模型", 1))
    stmts = {sid: qview.cells(QT.builtin(sid)["qtable"], QUARTERS) for sid in ids}
    probs = probabilities(results, ids)
    order = sorted(ids, key=lambda i: -probs["sets"]["prior"]["probs"].get(i, 0.0))
    card_bytes = Path(qcard.__file__).with_name("card.json").read_bytes() if hasattr(qcard, "__file__") else b""
    mix = {
        "version": VERSION, "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "zh": "未来一年：情景条件模拟（不是预测：预测只在情景概率里）",
        "origin": origin, "quarters": quarters, "train_end": qeng["train_end"],
        "assets": list(qeng["assets"]), "asset_names": {a["id"]: a["name"] for a in config.ASSETS},
        "grid": list(GRID), "switch_gate": gate,
        # the build-time copy only: the pages read the live switch (a stale or failed result switches nothing), and
        # a missing monthly block is a notice, never a promotion of the quarterly stack
        "default_engine": "quarterly" if gate["passed"] else ("monthly" if "monthly" in engines else None),
        "engines": engines,
        "scenarios": [{"id": i, "name": names[i]} for i in order],
        "probabilities": probs,
        "statements": stmts, "channels": qview.channels(),
        "history_rates": history_rates(stack.P, end=str(stack.train_end)),
        "rate_names": {"contraction_4q": "未来 4 个季度内出现收缩", "cpi_gt4_q4": "第 4 季末 CPI 同比高于 4%",
                       "ff_up1_q4": "第 4 季末政策利率比今天高 1 个百分点以上"},
        "sep": {"available": False, "text": "美联储当时的预测（SEP）要从美联储的实时数据库下载（下载须经你同意），这里暂时没有"},
        "analog": analog_block(results),
        "versions": {"card": hashlib.sha256(card_bytes).hexdigest()[:12], "qmodel": qeng["version"],
                     "monthly": engines.get("monthly", {}).get("version"), "probabilities": probs["version"]},
        "notes": notes,
    }
    led_path = _ledger_path(results)
    if ledger:
        entries = [ledger_entry(mix, e) for e in engines]
        mix["ledger_write"] = append_ledger(led_path, [e for e in entries if e])
    entries = read_ledger(led_path)
    mix["ledger"] = {"path": f"data/results/{LEDGER_DIR}/{LEDGER_FILE}", "n_entries": len(entries),
                     "recent": [{"origin": e.get("origin"), "engine": e.get("engine"), "created": e.get("created"),
                                 "model": e.get("model"), "probabilities": e.get("probabilities"),
                                 "quarters": e.get("quarters"),
                                 "us_equity": [None if g is None else
                                               {"p10": _f(np.interp(10.0, GRID, g)), "p50": _f(np.interp(50.0, GRID, g)),
                                                "p90": _f(np.interp(90.0, GRID, g))}
                                               for g in (e.get("assets") or {}).get("us_equity") or []]}
                                for e in entries[-8:]][::-1],
                     **score_ledger(entries, stack.Q), "text": LEDGER_NOTE}
    if write:
        jsonio.dump(mix, results / OUTPUT)
    return mix


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="未来一年：情景条件模拟 (forecast_mix.json)")
    ap.add_argument("--paths", type=int, default=DEFAULT_PATHS, help="quarterly-stack paths per scenario")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-ledger", action="store_true", help="do not append to the forward ledger")
    ap.add_argument("--no-context", action="store_true", help="skip the monthly engine's reference run")
    args = ap.parse_args(argv)
    ctx = None
    if not args.no_context:
        from ..context import EngineContext
        ctx = EngineContext.load(selection=True)
    mix = run(ctx, n_paths=args.paths, seed=args.seed, ledger=not args.no_ledger)
    print(f"[scenario_mix] {OUTPUT}: {len(mix['scenarios'])} scenarios, engines {list(mix['engines'])}, "
          f"ledger {mix.get('ledger_write')}; notes: {len(mix['notes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
