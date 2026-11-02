"""The instruments build step: every ticker on every built-in scenario path (INSTRUMENTS_DESIGN §2.8, §6.1).

    python -m scenariolab.instruments.build          # standalone (the build's step calls run(ctx))
    # in scenariolab/build.py:  step("instruments", lambda: _run("instruments.build", ctx))   (after scenarios)

Reads results/scenario_paths.npz and the macro paths the layer needs (y10, ff, Baa - 10y, the drawn months):
results/scenario_macro.npz when library.run wrote it (core.macro_arrays), else the scenarios are simulated again
with the build's own seeds (library.seed_for(id, 0), the stored path count) and every total / income must equal the
npz bit for bit (T1: the instrument layer never changes a stored number). Writes, next to the paths:

  instrument_paths.npz   tickers (K,), ids (S,), probs (S,), labels (T + 1,), start (K,) USD start prices,
                         total / income (S, N, T, K) float32 monthly, alive (S, N, T, 2) bool (QLD, TQQQ), and for
                         the estimated tickers' other alpha variants the yearly factors
                         alpha_<mode>_{total,price,income} (S, N, Y, 10) float32 (np.savez_compressed). USD price
                         levels and cash per share are not stored: read_paths() derives them from total, income and
                         start (level_t = start x prod(1 + total - income), cash_t = income_t x level_t-1)
  instrument_meta.json   parameter versions, the start state and its source, data vintages, per-scenario
                         diagnostics (T18), headline statistics, flags, file sizes, timings
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import config, jsonio
from . import ESTIMATED, LEVERAGED, TICKERS
from . import anchors as anc
from . import params as prm
from .core import CoreView
from .layer import credit_link_status, resolve_credit_link, run_layer

PATHS_FILE = "instrument_paths.npz"
META_FILE = "instrument_meta.json"
MACRO_FILE = "scenario_macro.npz"
SCENARIO_PATHS = "scenario_paths.npz"
ALPHA_VARIANTS = ("half", "raw")
BUILD_CMD = "python -m scenariolab.build --only instruments"


def _resimulate(ctx, ids: list[str], n_paths: int, z) -> dict:
    """Macro paths and drawn months of the stored scenarios, simulated again with the build's seeds; each total /
    income is checked against the npz (bit for bit) before its macro paths are used."""
    from ..scenarios import library
    from ..scenarios.runner import run_spec
    base = library.calibration_base(ctx)
    specs = {s.get("id"): s for s in library.load_specs()}
    out = {"y10": [], "ff": [], "spread": [], "drawn": [], "max_abs_diff": {}}
    months = np.asarray(ctx.db.dates.strftime("%Y-%m")).astype("U7")
    for k, sid in enumerate(ids):
        if sid not in specs:
            raise KeyError(f"scenario {sid!r} of {SCENARIO_PATHS} is not a built-in spec any more: rebuild scenarios")
        spec, _ = library.resolve_calibration(specs[sid], base, ctx.model.spec)
        _, sim = run_spec(ctx, spec, n_paths=n_paths, seed=library.seed_for(sid, 0), source="builtin",
                          attribution=False, regimes=False)
        d = max(float(np.abs(sim.total - z["total"][k]).max()), float(np.abs(sim.income - z["income"][k]).max()))
        out["max_abs_diff"][sid] = d
        if d != 0.0:
            raise RuntimeError(f"re-simulated {sid} differs from {SCENARIO_PATHS} (max |diff| {d:.2e}): the stored "
                               "paths come from another engine state; rebuild scenarios first")
        for key in ("y10", "ff", "spread"):
            out[key].append(np.asarray(sim.macro[key], np.float32))
        out["drawn"].append(months[np.asarray(sim.idx)])
    return {k: (np.stack(v) if isinstance(v, list) else v) for k, v in out.items()}


def paths_fingerprint(z) -> str:
    """A cheap fingerprint of scenario_paths.npz's numbers (ids + the first two paths' returns): a macro file
    made for other paths is never used."""
    import hashlib
    h = hashlib.sha256()
    h.update("|".join(str(x) for x in z["ids"]).encode())
    h.update(np.ascontiguousarray(np.asarray(z["total"])[:, :2]).tobytes())
    return h.hexdigest()


def scenario_macro(ctx, root: Path, z, write: bool = True) -> dict:
    """y10 / ff / spread level paths and the drawn months of every stored scenario path: from scenario_macro.npz
    (library.run's, written with core.macro_arrays, or this step's own cache) when it belongs to these paths
    (same ids and shapes, and the same fingerprint or a file newer than scenario_paths.npz), else simulated again
    with the build's seeds (checked bit for bit) and cached there."""
    ids = [str(x) for x in z["ids"]]
    S, N, T, _ = z["total"].shape
    p = root / MACRO_FILE
    fp = paths_fingerprint(z)
    if p.exists():
        try:
            with np.load(p, allow_pickle=False) as m:
                same = ([str(x) for x in m["ids"]] == ids if "ids" in m.files else False) and \
                    m["y10"].shape == (S, N, T + 1)
                fresh = (str(m["paths_fingerprint"]) == fp if "paths_fingerprint" in m.files
                         else p.stat().st_mtime >= (root / SCENARIO_PATHS).stat().st_mtime)
                if same and fresh:
                    dates = np.asarray(m["db_dates"]).astype(str) if "db_dates" in m.files else None
                    drawn = dates[np.asarray(m["drawn_idx"])] if dates is not None and "drawn_idx" in m.files else None
                    return {"y10": np.asarray(m["y10"]), "ff": np.asarray(m["ff"]),
                            "spread": np.asarray(m["spread"]), "drawn": drawn, "source": MACRO_FILE}
        except (OSError, KeyError, ValueError):      # unreadable or of another layout: simulate again
            pass
    t = time.time()
    out = _resimulate(ctx, ids, N, z)
    out["source"] = f"re-simulated with the build seeds ({time.time() - t:.1f}s), identical to {SCENARIO_PATHS}"
    if write:
        months = np.asarray(ctx.db.dates.strftime("%Y-%m")).astype("U7")
        pos = {mm: i for i, mm in enumerate(months)}
        idx = np.vectorize(pos.__getitem__)(out["drawn"]).astype(np.int16)
        tmp = root / "scenario_macro.tmp.npz"
        np.savez_compressed(tmp, ids=np.array(ids), y10=out["y10"], ff=out["ff"], spread=out["spread"],
                            drawn_idx=idx, db_dates=months, paths_fingerprint=np.array(fp))
        tmp.replace(p)
    return out


def income_y1(ip) -> np.ndarray:
    """(n,) the first year's cash per share over the start price (the project endpoint's definition: the year's
    cash income_m x level_m summed, divided by level_0)."""
    m = 12 if ip.freq == "M" else 4
    lv = np.asarray(ip.level, dtype=np.float64)
    cash = np.asarray(ip.income, dtype=np.float64)[:, :m] * lv[:, :m]
    return cash.sum(axis=1) / np.maximum(lv[:, 0], 1e-12)


def _cagr(total: np.ndarray, years: int) -> np.ndarray:
    m = years * 12
    if total.shape[1] < m:
        return np.full(total.shape[0], np.nan)
    return np.exp(np.log1p(np.maximum(total[:, :m], -0.999999)).sum(axis=1) / years) - 1.0


def run(ctx, results_dir: Path | None = None, write: bool = True, tickers=TICKERS) -> dict:
    """Run the layer on every built-in scenario and write instrument_paths.npz / instrument_meta.json."""
    t0 = time.time()
    root = Path(results_dir or config.RESULTS_DIR)
    with np.load(root / SCENARIO_PATHS, allow_pickle=False) as f:      # each z[k] would decompress again
        z = {k: np.asarray(f[k]) for k in f.files}
    z["files"] = list(z)
    ids = [str(x) for x in z["ids"]]
    labels = [str(x) for x in z["labels"]]
    probs = np.asarray(z["probs"], float) if "probs" in z["files"] else np.full(len(ids), 1.0 / len(ids))
    assets = [str(a) for a in z["assets"]]
    S, N, T, _ = z["total"].shape
    macro = scenario_macro(ctx, root, z)
    t_macro = time.time() - t0
    anchor = anc.current(labels[0][:7], prefer_data=True)
    tickers = list(tickers)
    K, Y = len(tickers), T // 12
    est_cols = [t for t in tickers if t in ESTIMATED]
    total = np.empty((S, N, T, K), np.float32)
    income = np.empty((S, N, T, K), np.float32)
    alive = np.ones((S, N, T, 2), bool)
    var = {m: {k: np.empty((S, N, Y, len(est_cols)), np.float32) for k in ("total", "price", "income")}
           for m in ALPHA_VARIANTS}
    diag, stats, flags = {}, {}, {}
    for s, sid in enumerate(ids):
        core = CoreView.from_arrays("M", z["total"][s], z["income"][s],
                                    {"cpi": z["cpi"][s], "y10": macro["y10"][s], "ff": macro["ff"][s],
                                     "spread": macro["spread"][s]}, labels, assets, scenario_id=sid,
                                    set_id="builtin", drawn=None if macro.get("drawn") is None else macro["drawn"][s])
        out = run_layer(core, tickers, anchor=anchor)
        for k, t in enumerate(tickers):
            ip = out.paths[t]
            total[s, :, :, k], income[s, :, :, k] = ip.total, ip.income
            if t in LEVERAGED and ip.alive is not None:
                alive[s, :, :, LEVERAGED.index(t)] = ip.alive
            flags[t] = list(ip.flags)
        if est_cols:
            from .estimated import simulate as est_sim
            from ..portfolio.withdrawal import year_factors
            for mode in ALPHA_VARIANTS:
                e = est_sim(core, out.vol.mkt_vol_quarterly(), est_cols, None, anchor["estimated"], alpha_mode=mode)
                f = year_factors(np.stack([e[t].total for t in est_cols], axis=2),
                                 np.stack([e[t].income for t in est_cols], axis=2), np.ones((N, T + 1)))
                for kk in ("total", "price", "income"):
                    var[mode][kk][s] = getattr(f, kk)
        diag[sid] = out.diagnostics()
        stats[sid] = {t: {"cagr10_p10": float(np.nanpercentile(_cagr(out.paths[t].total, 10), 10)),
                          "cagr10_p50": float(np.nanmedian(_cagr(out.paths[t].total, 10))),
                          "cagr10_p90": float(np.nanpercentile(_cagr(out.paths[t].total, 10), 90)),
                          "income_y1_p50": float(np.median(income_y1(out.paths[t])))}
                      for t in tickers}
    t_layer = time.time() - t0 - t_macro
    credit = {}
    for t in prm.bridges()["credit_link_flag"]["tickers"]:
        if t in flags and "credit_link" in flags[t]:
            credit[t] = credit_link_status({sid: (diag[sid]["ticker_corr_us_equity"].get(t) or {}).get("sim") for sid in ids},
                                           prm.bridges()["hist_corr_with_us_equity"].get(t))
            flags[t] = resolve_credit_link(flags[t], credit[t])
    start = np.array([float(anchor["prices"].get(t, {}).get("price", np.nan)) for t in tickers])
    arrays = {"tickers": np.array(tickers), "ids": np.array(ids), "probs": probs, "labels": np.array(labels),
              "start": start, "total": total, "income": income, "alive": alive,
              "leveraged": np.array(LEVERAGED), "estimated": np.array(est_cols)}
    for mode in ALPHA_VARIANTS:
        for kk in ("total", "price", "income"):
            arrays[f"alpha_{mode}_{kk}"] = var[mode][kk]
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "tickers": tickers, "scenario_ids": ids,
        "n_paths": N, "months": T, "anchor_month": labels[0], "params_version": prm.version(),
        "start": {t: anchor["prices"].get(t) for t in tickers}, "start_source": anchor.get("source"),
        "estimated_state": anchor.get("estimated"), "vol_anchor": anchor.get("vol"), "rates_anchor": anchor.get("rates"),
        "vintages": anchor.get("vintages"), "macro_source": macro.get("source"),
        "macro_check_max_abs_diff": macro.get("max_abs_diff"), "flags": flags, "diagnostics": diag,
        "stats": stats, "alpha_variants": list(ALPHA_VARIANTS), "alpha_default": "shrunk", "credit_link": credit,
        "defaults": {"vol_source": "model", "residual_mode": "fixed", "d_gz": prm.bridges()["d_gz"]["default"],
                     "d_gz_formula": "as fitted (D15 default), the same for every row and view",
                     "payout_cut_multiple": (prm.estimated().get("payout_cut") or {}).get("multiple"),
                     "letf_s_eff": {t: prm.structural()["letf"][t]["s_eff_default"] for t in LEVERAGED},
                     "qqqi_dist_rate": prm.structural()["qqqi"]["distribution_rate_annual"]},
        "seconds": {"macro": round(t_macro, 2), "layer": round(t_layer, 2)},
    }
    try:                                            # the replays on history (data/raw/instruments; ~3 s)
        from . import validate
        meta["validation"] = validate.summary()
    except (FileNotFoundError, KeyError, ValueError, IndexError) as err:
        meta["validation"] = {"unavailable": type(err).__name__}
    if write:
        tmp = root / "instrument_paths.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(root / PATHS_FILE)
        meta["file_bytes"] = (root / PATHS_FILE).stat().st_size
        meta["seconds"]["total"] = round(time.time() - t0, 2)
        jsonio.dump(meta, root / META_FILE)
    return meta


def read_paths(path: Path | None = None, tickers=None, scenarios=None) -> dict:
    """instrument_paths.npz as the service needs it, for the chosen tickers / scenario ids (default: all):
    {"tickers", "ids", "probs", "labels", "start", "total", "income" (S, N, T, K) float32, "level" (S, N, T + 1, K)
    USD per share, "cash" (S, N, T, K) USD per share paid in each month, "alive" (S, N, T, 2)}."""
    p = Path(path or config.RESULTS_DIR / PATHS_FILE)
    with np.load(p, allow_pickle=False) as z:
        all_t = [str(x) for x in z["tickers"]]
        all_s = [str(x) for x in z["ids"]]
        kt = [all_t.index(t) for t in (tickers or all_t)]
        ks = [all_s.index(s) for s in (scenarios or all_s)]
        total = np.asarray(z["total"][ks][..., kt], np.float32)
        income = np.asarray(z["income"][ks][..., kt], np.float32)
        start = np.asarray(z["start"])[kt]
        out = {"tickers": [all_t[k] for k in kt], "ids": [all_s[k] for k in ks],
               "probs": np.asarray(z["probs"])[ks], "labels": [str(x) for x in z["labels"]], "start": start,
               "total": total, "income": income, "alive": np.asarray(z["alive"])[ks]}
    g = np.cumprod(np.maximum(1.0 + total.astype(np.float64) - income, 0.0), axis=2)
    level = np.concatenate([np.ones(g.shape[:2] + (1,) + g.shape[3:]), g], axis=2) * start
    out["level"] = level
    out["cash"] = income * level[:, :, :-1]
    return out


CUSTOM_SUFFIX = ".instr.npz"
CONTINUATION_FILE = "instrument_continuation.npz"


def load_scenarios(results_dir: Path | None = None) -> dict:
    """scenario_paths.npz + scenario_macro.npz in memory (what project() needs; cache it by the files' mtimes)."""
    root = Path(results_dir or config.RESULTS_DIR)
    with np.load(root / SCENARIO_PATHS, allow_pickle=False) as f:
        z = {k: np.asarray(f[k]) for k in f.files}
    z["files"] = list(z)
    with np.load(root / MACRO_FILE, allow_pickle=False) as m:
        if [str(x) for x in m["ids"]] != [str(x) for x in z["ids"]]:
            raise FileNotFoundError(f"{MACRO_FILE} 与 {SCENARIO_PATHS} 不一致：请运行 {BUILD_CMD}")
        if "paths_fingerprint" in m.files and str(m["paths_fingerprint"]) != paths_fingerprint(z):
            raise FileNotFoundError(f"{MACRO_FILE} 已过期：请运行 {BUILD_CMD}")
        z.update({k: np.asarray(m[k]) for k in ("y10", "ff", "spread")})
    return z


def project(tickers, params: dict | None = None, scenario_ids=None, data: dict | None = None,
            vol_source: str = "model", alpha_mode: str = "shrunk", fixed_vol: float | None = None) -> dict:
    """The recipes recomputed on the stored scenario paths with other parameters (/api/instruments/project):
    {scenario id: InstrumentSet}. Streams are the build's, so with the default parameters the numbers equal
    instrument_paths.npz; a structural ticker takes well under a second for the 13 scenarios."""
    z = data if data is not None else load_scenarios()
    ids = [str(x) for x in z["ids"]]
    labels, assets = [str(x) for x in z["labels"]], [str(a) for a in z["assets"]]
    anchor = anc.current(labels[0][:7], prefer_data=True)
    out = {}
    for s, sid in enumerate(ids):
        if scenario_ids and sid not in scenario_ids:
            continue
        core = CoreView.from_arrays("M", z["total"][s], z["income"][s],
                                    {"cpi": z["cpi"][s], "y10": z["y10"][s], "ff": z["ff"][s], "spread": z["spread"][s]},
                                    labels, assets, scenario_id=sid, set_id="builtin")
        out[sid] = run_layer(core, list(tickers), params=params, vol_source=vol_source, anchor=anchor,
                             alpha_mode=alpha_mode, fixed_vol=fixed_vol)
    return out


def write_custom(sim, db, custom_id: str, out_dir: Path | None = None, anchor: dict | None = None,
                 tickers=TICKERS) -> Path:
    """results/custom/<id>.instr.npz for a custom scenario (server.custom_run, after its own npz): the layer on its
    SimOutput, streams keyed ("custom:<id>", id); the same layout as instrument_paths.npz with S = 1."""
    out_dir = Path(out_dir or config.RESULTS_DIR / "custom")
    core = CoreView.from_sim(sim, db, scenario_id=custom_id, set_id=f"custom:{custom_id}")
    out = run_layer(core, tickers, anchor=anchor)
    total, income = out.stacked(tickers)
    alive = np.stack([out.paths[t].alive if out.paths[t].alive is not None else np.ones(total.shape[:2], bool)
                      for t in LEVERAGED if t in out.paths], axis=2)
    a = anchor or anc.current(str(core.labels[0])[:7], prefer_data=False)
    p = out_dir / f"{custom_id}{CUSTOM_SUFFIX}"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez_compressed(tmp, tickers=np.array(list(tickers)), ids=np.array([custom_id]), probs=np.array([1.0]),
                        labels=np.array(core.labels), total=total[None].astype(np.float32),
                        income=income[None].astype(np.float32), alive=alive[None],
                        start=np.array([float(a["prices"].get(t, {}).get("price", np.nan)) for t in tickers]))
    tmp.replace(p)
    return p


def run_continuation(ctx, results_dir: Path | None = None, write: bool = True, tickers=TICKERS) -> dict:
    """Years 16..40 of every ticker: portfolio.service.build_continuation's chunks simulated again with its seeds
    (one 480-month unconditional path per stored scenario path), the layer on each chunk (continuation_factors),
    and results/instrument_continuation.npz: total_y / price_y / income_y (S x N, 25, K) float32, the alpha
    variants' alpha_<mode>_{total,price,income}_y (S x N, 25, 10), cpi_y, tickers. T2: the asset factors of each
    chunk are compared with continuation_paths.npz when it is there (max |diff| in the meta, expected 0)."""
    from ..engine.pooling import BLOCK, pool_phase
    from ..engine.simulate import capital_market_assumptions, fit_valuation, simulate
    from ..portfolio import service as svc
    from ..portfolio.withdrawal import year_factors
    from .layer import continuation_factors, rows_for_chunk
    t0 = time.time()
    root = Path(results_dir or config.RESULTS_DIR)
    with np.load(root / SCENARIO_PATHS, allow_pickle=False) as f:
        ids, (S, N, T, _) = [str(x) for x in f["ids"]], f["total"].shape
    ref = None
    if (root / svc.CONTINUATION_FILE).exists():
        with np.load(root / svc.CONTINUATION_FILE, allow_pickle=False) as f:
            ref = np.asarray(f["total_y"])
    Tc = svc.HORIZON_BOUNDS[1] * 12
    db, an = ctx.db, ctx.anchor
    pool = pool_phase(db.frame, [], db.prior(), Tc, BLOCK, rare=db.rare)
    cma = capital_market_assumptions(db, an, method=svc.CONTINUATION_CMA)
    try:
        vm = fit_valuation(db, an, kappa=0.0, income="constant")
    except (KeyError, ValueError, np.linalg.LinAlgError):
        vm = None
    anchor = anc.current(str(an.date.strftime("%Y-%m")), prefer_data=True)
    rows, chunk = S * N, svc.CONTINUATION_CHUNK
    Yc = (Tc - T) // 12
    K, E = len(tickers), [t for t in tickers if t in ESTIMATED]
    out = {k: np.empty((rows, Yc, K), np.float32) for k in ("total", "price", "income")}
    alt = {m: {k: np.empty((rows, Yc, len(E)), np.float32) for k in ("total", "price", "income")}
           for m in ALPHA_VARIANTS}
    cpi_y = np.empty((rows, Yc + 1))
    check = []
    for c, r0 in enumerate(range(0, rows, chunk)):
        n = min(chunk, rows - r0)
        sim = simulate(db, an, [pool.blocks], [Tc], n_paths=n, seed=svc.continuation_chunk_seed(svc.CONTINUATION_SEED, c),
                       block=BLOCK, cma=cma, valuation=vm, **config.RESIDUALS)
        if ref is not None:
            f = year_factors(sim.total[:, T:], sim.income[:, T:], np.asarray(sim.macro["cpi"][:, T:], np.float64))
            check.append(float(np.abs(f.total.astype(np.float32) - ref[r0:r0 + n]).max()))
        g, i = rows_for_chunk(ids, N, r0, n)
        res = continuation_factors(sim, db, T, g, i, chunk=c, tickers=tickers, anchor=anchor,
                                   alpha_modes=("shrunk",) + ALPHA_VARIANTS)
        for k, j in (("total", 0), ("price", 1), ("income", 2)):
            out[k][r0:r0 + n] = res[k]
            for m in ALPHA_VARIANTS:
                alt[m][k][r0:r0 + n] = res["alpha"][m][j]
        cpi = np.asarray(sim.macro["cpi"][:, T:], np.float64)
        cpi_y[r0:r0 + n] = cpi[:, 0::12][:, :Yc + 1] / cpi[:, :1]
    meta = {"rows": rows, "years": Yc, "tickers": list(tickers), "chunks": len(check) or None,
            "asset_factors_max_abs_diff_vs_continuation_paths": max(check) if check else None,
            "seconds": round(time.time() - t0, 2)}
    if write:
        arrays = {"tickers": np.array(list(tickers)), "estimated": np.array(E), "cpi_y": cpi_y,
                  **{f"{k}_y": v for k, v in out.items()},
                  **{f"alpha_{m}_{k}_y": v for m in ALPHA_VARIANTS for k, v in alt[m].items()}}
        tmp = root / "instrument_continuation.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(root / CONTINUATION_FILE)
        meta["file_bytes"] = (root / CONTINUATION_FILE).stat().st_size
    return meta


def main(argv=None) -> int:
    import argparse

    from ..context import EngineContext
    from ..env import load_dotenv
    ap = argparse.ArgumentParser(description="Build the instrument paths (and optionally the continuation)")
    ap.add_argument("--continuation", action="store_true", help="also write instrument_continuation.npz")
    args = ap.parse_args(argv)
    load_dotenv()
    ctx = EngineContext.load(selection=True)
    meta = run(ctx)
    print(json.dumps({"file_bytes": meta.get("file_bytes"), "seconds": meta["seconds"],
                      "macro_source": meta["macro_source"], "start_source": meta["start_source"]}, indent=1))
    if args.continuation:
        print(json.dumps(run_continuation(ctx), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
