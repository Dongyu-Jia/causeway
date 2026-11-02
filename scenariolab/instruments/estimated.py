"""Estimated recipes: XLG, SCHD, MO, BRK.B, MAIN, ARCC, CSWC, EPD, MPLX, BME (quarterly, 13F prices from 2013).

Per quarter (log total returns, estimated.json):

    lr = pi lr_cash + alpha + sum_j w_j (lr_j - lr_cash) + sum_k b_k f_k + eps

  lr_j    the core's asset classes (style weights w >= 0; funds sum to 1)
  f_k     d_gz only (BDCs and MLPs): the core's own, or the bridge (defect 2, core.gz_path)
  pi      0.5 for the BDCs (floating-rate carry, half passed through), 1 elsewhere
  alpha   shrunk intercept (BDC group mean +6.3%/yr, MLP +2.25%, single companies to 0, funds to -fee); per row
          one draw of its parameter uncertainty, the same in the scenario years and the continuation; variants
          "shrunk" (default), "half", "raw" differ only in this scalar (D10-D12)
  eps     the ticker's own residual, drawn by quarter jointly for all ten tickers (the same historical quarter on
          a path keeps the BDC and MLP blocks' correlation), by a stationary block bootstrap (mean block 2) of the
          joint pool 2016Q2..2026Q2 (or the core's own residual quarter where it lies in the pool: pairing, D16),
          scaled by clip(v / rv_d, 0.5, 2.5) x sqrt(n / (n - k)), v = the volatility state's US-market vol

Defect 1 (the residual draw was not mean zero): the stored residuals are actual - (systematic + shrunk alpha), so
their mean is alpha_hat - alpha and drawing them adds the shrinkage back; and the vol scaling is correlated with the
residual. Fixed at draw time: (a) every ticker's residuals are demeaned over the pool actually drawn from; (b) the
scaled draw has its conditional mean given v removed, m(v) = mean over the pool of e_d infl clip(v / rv_d), computed
exactly for every simulated v. The historical replay (actual core, actual dates, raw residuals) still reproduces
the actual returns exactly (T4). residual_mode "asis" keeps the old draw for comparison only (D9). The run reports
the drawn residuals' measured mean overall and in 5 buckets of v (T5's check, with its sampling error) in meta.

Payouts (distributions.py): the sticky rule's c_LR is the row's own (its scenario's systematic mean + its alpha
draw), and a payout that would put the yield on the smoothed price above estimated.json payout_cut x the normal
yield is cut back to the normal yield (an explicit cut, so a holding never pays its whole value out).

The credit-link chip: a BDC / MLP recipe on a core without its own d_gz (the monthly engine: the bridge) carries
the flag credit_link; whoever has the views' diagnostics (instruments.build, the project endpoint) keeps it only for
the tickers whose simulated correlation with US equity is more than the tolerance below history and turns it into
the neutral credit_link_ok for the others (layer.credit_link_status, T18).

On a monthly core (D14): the systematic part is computed month by month (linear in log returns, so the quarter's sum
equals the quarterly recipe), the quarterly residual is split into three months by a Brownian bridge (each month
N(0, s(v)^2 / 3) shifted to sum to the quarter's draw), and the payout rule runs on quarter-end prices with cash
booked by the payment calendar (MAIN, BME monthly; the rest at the quarter's end): chip bridge_month.
"""
from __future__ import annotations

import numpy as np

from . import distributions as dist
from . import params as prm
from .core import CoreView
from .streams import row_normals, stream

TICKERS = ("XLG", "SCHD", "MO", "BRK.B", "MAIN", "ARCC", "CSWC", "EPD", "MPLX", "BME")
BDC_MLP = ("MAIN", "ARCC", "CSWC", "EPD", "MPLX")


def recipe(t: str) -> dict:
    return prm.estimated()["tickers"][t]


def pool_dates(tickers=TICKERS) -> np.ndarray:
    """The joint residual pool: the quarters every estimated ticker has a residual for (2016Q2..2026Q2)."""
    sets = [set(recipe(t)["resid"]) for t in tickers]
    return np.array(sorted(set.intersection(*sets)))


QUARTER_END_DAY = {3: 31, 6: 30, 9: 30, 12: 31}


def qend(month: str) -> str:
    """'YYYY-MM' (or a date) -> the quarter-end date 'YYYY-MM-DD' of its quarter."""
    y, m = int(month[:4]), int(month[5:7])
    qm = ((m - 1) // 3 + 1) * 3
    return f"{y:04d}-{qm:02d}-{QUARTER_END_DAY[qm]:02d}"


def systematic(core: CoreView, r: dict, gz: np.ndarray) -> dict:
    """Per-period systematic log return parts of a recipe on the core (no alpha, no residual)."""
    cash = core.lr("cash")
    parts = {"cash": r["cash_passthrough"] * cash}
    for c, b in r["loadings"].items():
        if abs(b) < 1e-12:
            continue
        if c.startswith("x_"):
            parts[c[2:]] = b * (core.lr(c[2:]) - cash)
        elif c == "d_gz":
            parts["credit_spread"] = b * gz
        else:
            raise KeyError(f"recipe factor {c!r} has no core path")
    return parts


def draw_dates(n: int, Q: int, P: int, rng: np.random.Generator, p_new: float = 0.5,
               paired: np.ndarray | None = None) -> np.ndarray:
    """(n, Q) pool positions: stationary block bootstrap (continue the block with prob 1 - p_new, wrapping), with
    `paired` (n, Q) pool positions (or -1) of the core's own residual quarters taking precedence."""
    idx = np.empty((n, Q), dtype=np.int64)
    idx[:, 0] = rng.integers(0, P, n)
    for q in range(1, Q):
        new = rng.random(n) < p_new
        idx[:, q] = np.where(new, rng.integers(0, P, n), (idx[:, q - 1] + 1) % P)
    if paired is not None:
        idx = np.where(paired >= 0, paired, idx)
    return idx


def scaled_residuals(e: np.ndarray, rv: np.ndarray, infl: float, v: np.ndarray, clip=(0.5, 2.5)) -> np.ndarray:
    """(n, Q, P): every pool residual scaled to every simulated vol v (n, Q)."""
    s = np.clip(np.asarray(v)[..., None] / rv[None, None, :], clip[0], clip[1])
    return e[None, None, :] * infl * s


def residual_draw(t: str, pool: np.ndarray, idx: np.ndarray, v: np.ndarray | None, mode: str = "fixed") -> dict:
    """The residual of ticker t at pool positions idx (n, Q) for vol v (n, Q).

    mode "fixed": demeaned over the pool, vol-scaled, recentered on its conditional mean m(v) (defect 1);
    "asis": raw, vol-scaled (the research plug-in, for comparison); "replay": raw, unscaled, uninflated;
    "none": no residual (a path without shocks).
    Returns {"eps" (n, Q), "scale" (n, Q): the residual sd given v (for the monthly split)}."""
    r = recipe(t)
    clip = tuple(prm.estimated()["clip"])
    e = np.array([r["resid"][d] for d in pool])
    rv = np.array([r["rv_at"][d] for d in pool])
    if mode == "none":
        return {"eps": np.zeros(idx.shape), "scale": np.zeros(idx.shape)}
    if mode == "replay":
        return {"eps": e[idx], "scale": np.full(idx.shape, float(np.std(e)))}
    if mode == "fixed":
        e = e - e.mean()
    if v is None:
        sc = np.broadcast_to((e * r["infl"])[None, None, :], idx.shape + (len(pool),))
    else:
        sc = scaled_residuals(e, rv, r["infl"], v, clip)
    pick = np.take_along_axis(sc, idx[..., None], axis=2)[..., 0]
    m = sc.mean(axis=2)
    sd = np.sqrt(np.maximum(sc.var(axis=2), 0.0))
    return {"eps": pick - m if mode == "fixed" else pick, "scale": sd, "cond_mean": m}


def _paired_positions(core: CoreView, pool: np.ndarray, Q: int) -> np.ndarray | None:
    if core.resid_drawn is None:
        return None
    rd = np.asarray(core.resid_drawn)
    first = rd[:, 0:Q * 3:3] if core.freq == "M" else rd[:, :Q]
    pos = {d: i for i, d in enumerate(pool)}
    return np.vectorize(lambda m: pos.get(qend(str(m)), -1))(first).astype(np.int64)


def simulate(core: CoreView, mkt_vol_q: np.ndarray | None, tickers=TICKERS, params: dict | None = None,
             anchor_est: dict | None = None, alpha_mode: str = "shrunk", residual_mode: str = "fixed",
             replay_idx: np.ndarray | None = None, alpha_uncertainty: bool = True,
             c_lr_mode: str = "running") -> dict:
    """The estimated tickers on a core view (monthly or quarterly). mkt_vol_q (n, Q): the annualized US-market vol
    of every quarter (VolState.mkt_vol_quarterly()). Returns {ticker: InstrumentPaths}."""
    from .layer import InstrumentPaths
    params = params or {}
    E = prm.estimated()
    anchor_est = anchor_est or prm.packaged_anchors()["estimated"]
    n = core.n
    monthly = core.freq == "M"
    if monthly and core.T % 3:
        raise ValueError("a monthly core needs whole quarters (T divisible by 3)")
    cq = core.quarterly()
    Q = cq.T
    pool = pool_dates()
    if replay_idx is not None:
        idx = np.asarray(replay_idx, dtype=np.int64)
    else:
        idx = draw_dates(n, Q, len(pool), stream(core.set_id, core.key, "resid_dates"), E["block_p_new"],
                         _paired_positions(core, pool, Q))
    gz, gz_src = core.gz_path()
    tb3_q = cq.tb3_path()
    cut_multiple = (E.get("payout_cut") or {}).get("multiple")
    groups, index = core.groups(), core.indices()
    out = {}
    for t in tickers:
        r = recipe(t)
        p = params.get(t, {})
        mode = p.get("alpha_mode", alpha_mode)
        a0 = float(r["alpha_variants_q"][mode])
        a_row = np.full(n, a0)
        if alpha_uncertainty and residual_mode != "replay":
            a_row = a_row + r["alpha_sd_q"] * row_normals(core.row_namespace(), groups, index, "alpha", t)
        sp = systematic(core, r, gz)
        x = sum(sp.values())
        per = 3.0 if monthly else 1.0
        alpha_p = np.repeat((a_row / per)[:, None], core.T, axis=1)
        rd = residual_draw(t, pool, idx, None if residual_mode == "replay" else mkt_vol_q,
                           "replay" if residual_mode == "replay" else residual_mode)
        eps_q = rd["eps"]
        if monthly:
            z = stream(core.set_id, core.key, "bridge", t).standard_normal((n, Q, 3))
            w = z * (rd["scale"] / np.sqrt(3.0))[..., None]
            eps = (w - w.mean(axis=2, keepdims=True) + eps_q[..., None] / 3.0).reshape(n, Q * 3)
        else:
            eps = eps_q
        lr = x + alpha_p + eps
        lr_q = lr.reshape(n, Q, 3).sum(axis=2) if monthly else lr
        st = dict(anchor_est[t])
        if "price" in p:
            st["price"] = float(p["price"])
        d = {"cut_multiple": cut_multiple, **r["dist"], **(p.get("dist") or {})}   # overrides, e.g. {"gamma": -0.15}
        mlr = lr.reshape(n, Q, 3) if monthly else None
        if d["type"] == "sticky_ec":
            if c_lr_mode == "running":
                c_lr = dist.c_lr_running(x, a_row, d["normal_yield"], core.ppy, groups)
            else:
                c_lr = np.full(n, float(d["c_lr"]))
            res = dist.sticky_quarterly(lr_q, d, st, tb3_q, c_lr, monthly_lr=mlr, pay_per_year=r["pay_per_year"])
        elif d["type"] == "price_prop":
            yq = d["k"] * (d["weights"]["us_equity"] * cq.asset_income("us_equity")
                           + d["weights"]["nasdaq"] * cq.asset_income("nasdaq"))
            res = dist.price_prop(lr_q, yq, st["price"], monthly_lr=mlr)
            c_lr = None
        else:
            res = dist.no_payout(lr_q, st["price"], monthly_lr=mlr)
            c_lr = None
        if monthly:
            total, income, level = np.expm1(lr), res["income_m"], res["price_m"]
        else:
            total = np.expm1(lr)
            income = dist.safe_div(res["dps_q"], res["price_q"][:, :-1])
            level = res["price_q"]
        parts = {**sp, "alpha": alpha_p, "residual": eps}
        meta = {"alpha_mode": mode, "alpha_q": a0, "residual_mode": residual_mode, "gz": gz_src,
                "c_lr": None if c_lr is None else float(np.mean(c_lr))}
        if residual_mode in ("fixed", "asis") and mkt_vol_q is not None:
            meta.update(residual_check(eps_q, mkt_vol_q))
        if "cut_q" in res:
            meta["cut_multiple"] = d.get("cut_multiple")
            meta["cut_share_paths"] = float(res["cut_q"].any(axis=1).mean())
            meta["floor_share_paths"] = float(res["floor_q"].any(axis=1).mean())
        flags = ["estimated", "no_2008", "tax"]
        if st.get("estimated_start"):
            flags.append("start_est")
        if monthly:
            flags.append("bridge_month")
        if t in BDC_MLP and gz_src != "core":
            flags.append("credit_link")
        if d["type"] == "sticky_ec" and d.get("cut_multiple") is not None:
            flags.append("payout_cut")
        out[t] = InstrumentPaths(ticker=t, freq=core.freq, total=total, income=income, level=level,
                                 dps=res["dps_q"], parts=parts, flags=tuple(flags), meta=meta)
    return out


def residual_check(eps_q: np.ndarray, v: np.ndarray, buckets: int = 5) -> dict:
    """T5 on the residuals actually drawn: their mean and their mean in `buckets` quantile buckets of the simulated
    vol v, annualized (x 4), with the sampling error of a bucket's mean (one standard error, annualized)."""
    e, vv = np.asarray(eps_q, float).ravel(), np.asarray(v, float).ravel()
    if not e.size:
        return {}
    edges = np.quantile(vv, np.linspace(0.0, 1.0, buckets + 1))
    k = np.clip(np.searchsorted(edges, vv, side="right") - 1, 0, buckets - 1)
    means = np.array([e[k == b].mean() if (k == b).any() else 0.0 for b in range(buckets)])
    se = float(e.std() / np.sqrt(max(e.size / buckets, 1.0)))
    return {"resid_mean_ann": float(4.0 * e.mean()), "resid_bucket_mean_max_ann": float(4.0 * np.abs(means).max()),
            "resid_bucket_se_ann": 4.0 * se, "resid_buckets": int(buckets)}
