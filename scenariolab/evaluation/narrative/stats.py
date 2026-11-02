"""Scores and statistics of the narrative test (SIM_DESIGN §8.6), shared by the scorer and the research import.

Per cell (episode × variant × CMA arm × asset × horizon), from the simulated draws of the cumulative log price
return (cash: its total return) and the realized value y: the median error err = q50 − y, |err|, the 80% band
(q10, q90) and whether it holds y, the PIT (share of draws below y), the sample CRPS and the band width. A
degenerate band (q90 − q10 < 1e-6) has no coverage or PIT. These are the research scorer's definitions
(sim_redesign/scorer/s05_score.py `cell`), so the research run's stored rows and a new run's rows are comparable.

Statistics treat the episode as the independent unit: a set's value for an episode is the mean over its assets; a
pooled number is the mean over episodes, with a 90% interval from BOOT_DRAWS bootstrap draws over episodes; a paired
difference is the mean over episodes of the per-episode difference on the cells both variants have. Each statistic
draws from its own generator, seeded from BOOT_SEED and the statistic's name, so a number does not depend on what
else was computed. The minimum detectable effect of a paired comparison is (z0.95 + z0.80) × sd(d) / √n (a one-sided
5% test with 80% power), printed beside every claim (§8.1).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from . import common as C

Z_MDE = 1.6448536269514722 + 0.8416212335729143      # z_{0.95} + z_{0.80}
CELL_KEYS = ("y", "q10", "q50", "q90", "mean", "err", "abs_err", "width", "crps", "cov80", "pit", "degenerate")


def crps(draws: np.ndarray, y: float) -> float:
    """The sample CRPS of the draws at y (the research scorer's estimator)."""
    x = np.sort(np.asarray(draws, float)[np.isfinite(draws)])
    n = x.size
    if n == 0:
        return float("nan")
    a = float(np.mean(np.abs(x - y)))
    i = np.arange(1, n + 1)
    b = float(np.sum((2 * i - n - 1) * x) / (n * n))
    return a - b


def cell(draws: np.ndarray, y: float) -> dict:
    """One scored cell (CELL_KEYS), or {} when there are no finite draws or no realized value."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0 or y is None or not np.isfinite(y):
        return {}
    q10, q50, q90 = (float(v) for v in np.percentile(d, [10, 50, 90]))
    degenerate = (q90 - q10) < 1e-6
    out = {"y": float(y), "q10": q10, "q50": q50, "q90": q90, "mean": float(d.mean()), "err": q50 - float(y),
           "abs_err": abs(q50 - float(y)), "width": q90 - q10,
           "crps": crps(d, float(y)) if not degenerate else abs(q50 - float(y)), "degenerate": bool(degenerate)}
    if degenerate:
        out.update(cov80=float("nan"), pit=float("nan"))
    else:
        out.update(cov80=float(q10 <= y <= q90), pit=float((np.sum(d < y) + 0.5 * np.sum(d == y)) / d.size))
    return out


def horizon_quarters(H: int) -> dict[str, int]:
    """The scored horizons inside an H-quarter window: 1Q, 2Q, 4Q, 8Q where they fit, and the window end."""
    out = {h: int(h[:-1]) for h in C.HORIZONS if h != "end" and int(h[:-1]) <= H}
    out["end"] = int(H)
    return out


def score_draws(paths: np.ndarray, realized: np.ndarray, H: int) -> list[dict]:
    """Rows (asset, horizon, quarters, CELL_KEYS) of one variant: paths (n, ≥H, 13) and realized (H, 13) quarterly
    log returns in config_q.ASSETS order, scored as cumulative sums."""
    cum = np.cumsum(np.asarray(paths, float)[:, :H, :], axis=1)
    ycum = np.cumsum(np.asarray(realized, float)[:H, :], axis=0)
    rows = []
    for hname, q in horizon_quarters(H).items():
        for j, a in enumerate(C.ASSETS):
            c = cell(cum[:, q - 1, j], float(ycum[q - 1, j]))
            if c:
                rows.append({"asset": a, "horizon": hname, "quarters": q, **c})
    return rows


# ---------------------------------------------------------------- bootstrap over episodes

def rng_for(*parts) -> np.random.Generator:
    """A generator seeded from BOOT_SEED and the statistic's name (reproducible whatever else was computed)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return np.random.default_rng([C.BOOT_SEED, int(h[:12], 16)])


def boot_mean(values, name: str, n_boot: int = C.BOOT_DRAWS) -> tuple[float, float, float]:
    """(mean, 5th, 95th percentile of the bootstrap mean) over the finite values; no interval below 3 values."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return (float("nan"),) * 3
    pt = float(v.mean())
    if v.size < 3:
        return pt, float("nan"), float("nan")
    idx = rng_for("boot_mean", name).integers(0, v.size, (n_boot, v.size))
    m = v[idx].mean(axis=1)
    return pt, float(np.percentile(m, 5)), float(np.percentile(m, 95))


def mde(diffs) -> float:
    """Minimum detectable effect of a paired comparison: (z0.95 + z0.80) × sd(d) / √n; nan below 3 episodes."""
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    if d.size < 3:
        return float("nan")
    return float(Z_MDE * d.std(ddof=1) / math.sqrt(d.size))


def _f(x, nd: int = 6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def pooled(rows: pd.DataFrame, codes=None, sets=None, by=("variant", "cma", "horizon")) -> list[dict]:
    """Per (by) × asset set: n_episodes, n_cells, mae [lo, hi], rmse, bias, cov80 [lo, hi], pit_mad, crps, the
    width and the cross-episode corr(median, realized) (§8.6) of the set's mean."""
    sets = sets or C.ASSET_SETS
    sub = rows if codes is None else rows[rows["code"].isin(list(codes))]
    out = []
    if sub.empty:
        return out
    for keys, g in sub.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for sname, al in sets.items():
            gg = g[g["asset"].isin(al)]
            if gg.empty:
                continue
            per = gg.groupby("code")
            mae = per["abs_err"].mean()
            name = "|".join(str(k) for k in keys) + f"|{sname}"
            m, lo, hi = boot_mean(mae.to_numpy(float), f"mae|{name}")
            cov = per["cov80"].mean()
            c, clo, chi = boot_mean(cov.to_numpy(float), f"cov|{name}")
            med, yv = per["q50"].mean(), per["y"].mean()
            corr = float(np.corrcoef(med, yv)[0, 1]) if len(med) >= 3 and med.std() > 0 and yv.std() > 0 else None
            out.append({**dict(zip(by, keys)), "assets": sname, "n_episodes": int(len(mae)), "n_cells": int(len(gg)),
                        "mae": _f(m), "mae_lo90": _f(lo), "mae_hi90": _f(hi),
                        "rmse": _f(math.sqrt(float(per["err"].apply(lambda s: (s ** 2).mean()).mean()))),
                        "bias": _f(per["err"].mean().mean()), "cov80": _f(c), "cov80_lo90": _f(clo),
                        "cov80_hi90": _f(chi), "pit_mad": _f(per["pit"].apply(lambda s: (s - 0.5).abs().mean()).mean()),
                        "crps": _f(per["crps"].mean().mean()), "width": _f(per["width"].mean().mean()),
                        "corr_median_realized": _f(corr, 4)})
    return out


def paired(rows: pd.DataFrame, a: tuple, b: tuple, horizon: str, assets: str = "equity9", codes=None,
           value: str = "abs_err") -> dict:
    """Variant a − variant b (each (variant, cma)) at a horizon over an asset set: the mean over episodes of the
    per-episode difference of `value` on the cells both have, its 90% interval, the MSE skill of a against b
    (1 − Σ err_a² / Σ err_b²) with its interval, the share of episodes where a wins, sd, MDE. {} when no episode
    has both."""
    al = C.ASSET_SETS[assets] if isinstance(assets, str) else list(assets)
    sub = rows[(rows["horizon"] == horizon) & rows["asset"].isin(al)]
    if codes is not None:
        sub = sub[sub["code"].isin(list(codes))]
    A = sub[(sub["variant"] == a[0]) & (sub["cma"] == a[1])].set_index(["code", "asset"])
    B = sub[(sub["variant"] == b[0]) & (sub["cma"] == b[1])].set_index(["code", "asset"])
    ix = A.index.intersection(B.index)
    if len(ix) == 0:
        return {}
    A, B = A.loc[ix], B.loc[ix]
    eps = sorted(set(ix.get_level_values(0)))
    d = np.array([float((A.loc[e, value] - B.loc[e, value]).mean()) for e in eps])
    sa = np.array([float((A.loc[e, "err"] ** 2).sum()) for e in eps])
    sb = np.array([float((B.loc[e, "err"] ** 2).sum()) for e in eps])
    name = f"paired|{a}|{b}|{horizon}|{assets}|{value}|{','.join(eps)}"
    m, lo, hi = boot_mean(d, name)
    skill = float(1 - sa.sum() / sb.sum()) if sb.sum() > 0 else float("nan")
    s_lo = s_hi = float("nan")
    if len(eps) >= 3:
        idx = rng_for("skill", name).integers(0, len(eps), (C.BOOT_DRAWS, len(eps)))
        den = sb[idx].sum(axis=1)
        bs = np.where(den > 0, 1 - sa[idx].sum(axis=1) / np.where(den > 0, den, 1.0), np.nan)
        s_lo, s_hi = float(np.nanpercentile(bs, 5)), float(np.nanpercentile(bs, 95))
    return {"a": a[0], "a_cma": a[1], "b": b[0], "b_cma": b[1], "horizon": horizon, "assets": assets,
            "value": value, "n_episodes": len(eps), "n_cells": int(len(ix)), "episodes": eps,
            "diff": _f(m), "lo90": _f(lo), "hi90": _f(hi), "sd": _f(float(np.std(d, ddof=1)) if len(d) > 1 else None),
            "mde": _f(mde(d)), "mse_skill": _f(skill), "skill_lo90": _f(s_lo), "skill_hi90": _f(s_hi),
            "a_wins_share": _f(float(np.mean(d < 0))), "per_episode": {e: _f(x) for e, x in zip(eps, d)},
            "favourable": bool(np.isfinite(hi) and hi < 0), "unfavourable": bool(np.isfinite(lo) and lo > 0)}


def chance(pairs: list[dict], level: float = 0.90) -> dict:
    """How many paired intervals exclude 0 in either direction, against what chance gives at this level."""
    tests = [p for p in pairs if p and p.get("lo90") is not None]
    side = (1.0 - level) / 2.0
    return {"tests": len(tests), "favourable": sum(1 for p in tests if p["favourable"]),
            "unfavourable": sum(1 for p in tests if p["unfavourable"]),
            "expected_each_side": round(side * len(tests), 2), "level": level}
