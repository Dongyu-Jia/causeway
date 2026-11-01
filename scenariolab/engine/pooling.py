"""Phase weights: entropy pooling over block starts.

The simulator draws blocks of B consecutive historical months (sampler.py). A
view says what the drawn months should average over a phase, so it is imposed
on block starts rather than on single months: each row of the pooling frame
is one feasible block start s, valued at the mean the sampler draws from it
(the B-month mean of x[s:s+B], blended with the shorter last block when the
phase is not a multiple of B). The sampler then draws starts straight from
the posterior, so the expected mean of the drawn months equals each view
target exactly, up to Monte Carlo noise. Pooling single months instead lets a
view on a fat-tailed flow (a 1980 rate swing, the April 2020 unemployment
jump) be met by a few spike months whose neighbours inside the same block
undo it. A view on a variable is judged only where it is observed: blocks
that would draw a month where it is missing are left out of that phase.

Month weights (each month's expected share of the draws) are derived for
reporting: era mix, top months, the effective number of months.

A view set must rest on enough history: at least MIN_ENS effective months and
MIN_EPISODES effective 12-month episodes (overlapping starts inflate the
month count, so one 2-3 year stretch alone can reach 30 months). Both are
checked on the pooled solution before any confidence blending, so a lower
confidence cannot make the paths more repetitive. When no mix of historical
blocks meets the views, or only a thin one, every view is shrunk toward the
prior by the largest feasible lambda in [0, 1], and lambda is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from .ep import EPResult, apply_views, blend, effective_number, support
from .sampler import BlockProbs, block_start_probs, phase_block

BLOCK = config.BLOCK_MONTHS
MIN_ENS = 30.0          # fewest effective months a phase may rest on
MIN_EPISODES = 6.0      # fewest effective 12-month episodes (calendar bins of block starts)
EPISODE_MONTHS = 12
SHRINK_STEPS = 14


@dataclass
class PhasePool:
    blocks: BlockProbs                 # what the sampler draws from
    month_weights: np.ndarray          # expected share of each month among the drawn months
    prior_month_weights: np.ndarray    # the same without views
    ep: EPResult | None                # pooling over block starts (None without views)
    shrink: float                      # 1.0 = views as requested, < 1 = shrunk toward the prior
    ens: float                         # effective number of months, exp(H(month_weights))
    shrink_reason: str | None = None   # "infeasible" (no mix of blocks meets the views) or "thin"
    ens_requested: float | None = None # effective months of the views as requested (feasible but thin)
    episodes: float | None = None      # effective 12-month episodes of the pooled solution (before blending)


def start_values(x: np.ndarray, block: int, n_months: int, propagate_nan: bool = False) -> np.ndarray:
    """Expected mean of the months drawn from each block start s = 0..T-block.

    A phase of n months draws n // block full blocks plus one block cut to
    the remaining n % block months, all from the same start distribution.
    Missing values count as the column mean, or, with propagate_nan, make the
    start's value NaN whenever a month it would draw is missing.
    """
    x = np.asarray(x, float)
    nan = np.isnan(x)
    if nan.any():
        x = np.where(nan, np.nanmean(x), x)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    s = np.arange(x.size - block + 1)
    full = (csum[s + block] - csum[s]) / block
    n_full, rest = divmod(n_months, block)
    out = full
    if rest:
        part = (csum[s + rest] - csum[s]) / rest
        out = (n_full * block * full + rest * part) / n_months
    if propagate_nan and nan.any():
        cnan = np.concatenate([[0], np.cumsum(nan)])
        out = np.where(cnan[s + block] - cnan[s] > 0, np.nan, out)
    return out


def start_frame(frame: pd.DataFrame, cols, block: int, n_months: int, propagate_nan: bool = False) -> pd.DataFrame:
    """Pooling frame: one row per block start (indexed by its first month)."""
    index = frame.index[: len(frame) - block + 1]
    return pd.DataFrame({c: start_values(frame[c].to_numpy(float), block, n_months, propagate_nan) for c in cols},
                        index=index)


def start_prior(prior: np.ndarray, block: int, rare: np.ndarray | None = None) -> np.ndarray:
    """Block-start probabilities under the month prior.

    A rare month's factor (config.RARE_MONTHS) applies in full to every block
    that would draw it; averaging month weights over a block would instead
    dilute it (a 0.25 month inside a block of ordinary months).
    """
    prior = np.asarray(prior, float)
    if rare is None or np.all(np.asarray(rare) == 1.0):
        return block_start_probs(prior, block)
    rare = np.asarray(rare, float)
    base = block_start_probs(prior / rare, block)
    S = base.size
    rmin = np.minimum.reduce([rare[i:i + S] for i in range(min(block, rare.size))])
    p = base * rmin
    return p / p.sum()


def month_frequency(start_probs: np.ndarray, block: int, n_months: int, T: int) -> np.ndarray:
    """Expected share of each of the T months among a phase's draws from `start_probs`."""
    q = np.asarray(start_probs, float)
    csum = np.concatenate([[0.0], np.cumsum(q)])
    t = np.arange(T)

    def cover(length: int) -> np.ndarray:      # P(a block of `length` months covers month t)
        hi = np.minimum(t, q.size - 1) + 1
        lo = np.maximum(t - length + 1, 0)
        return np.where(hi > lo, csum[hi] - csum[np.minimum(lo, hi)], 0.0)

    n_full, rest = divmod(n_months, block)
    return (n_full * cover(block) + (cover(rest) if rest else 0.0)) / n_months


def uncond_frequency(prior: np.ndarray, rare: np.ndarray | None = None, block: int = BLOCK,
                     n_months: int = config.HORIZON_MONTHS) -> np.ndarray:
    """How often an unconditional path draws each month (block edges and rare months under-drawn)."""
    T = len(prior)
    B = phase_block(block, n_months, T)
    return month_frequency(start_prior(prior, B, rare), B, n_months, T)


def episode_ens(start_probs: np.ndarray, width: int = EPISODE_MONTHS) -> float:
    """Effective number of distinct `width`-month episodes the block starts come from."""
    q = np.asarray(start_probs, float)
    return effective_number(np.add.reduceat(q, np.arange(0, q.size, width)))


def pool_phase(frame: pd.DataFrame, views: list[dict], prior: np.ndarray, n_months: int,
               block: int = BLOCK, confidence: float = 1.0, min_ens: float = MIN_ENS,
               rare: np.ndarray | None = None, min_episodes: float = MIN_EPISODES) -> PhasePool:
    """Block-start weights for one phase whose drawn months honour `views` on average."""
    T = len(frame)
    B = phase_block(block, n_months, T)
    sp = start_prior(prior, B, rare)
    prior_w = month_frequency(sp, B, n_months, T)
    if not views:
        return PhasePool(BlockProbs(sp, B), prior_w, prior_w, None, 1.0, effective_number(prior_w))
    cols = sorted({v["var"] for v in views})
    sf = start_frame(frame, cols, B, n_months, propagate_nan=True)
    for v in views:
        if "vol_mult" in v:                           # second moment of the drawn months (ep.apply_views)
            x = frame[v["var"]].to_numpy(float)
            sf[f"{v['var']}__sq"] = start_values(x * x, B, n_months, propagate_nan=True)
    ok = np.isfinite(sf.to_numpy(float)).all(axis=1)
    if not ok.any():
        return PhasePool(BlockProbs(sp, B), prior_w, prior_w, None, 0.0, effective_number(prior_w), "infeasible")
    sf_ok, p_ok = sf[ok], sp[ok] / sp[ok].sum()

    def full_q(q_ok: np.ndarray) -> np.ndarray:
        q = np.zeros(sp.size)
        q[ok] = q_ok
        return q

    def solve(lam: float):
        raw = apply_views(sf_ok, views, p=p_ok, shrink=lam)
        q_raw = full_q(raw.q)
        ens_raw = effective_number(month_frequency(q_raw, B, n_months, T))
        res = blend(raw, sf_ok, p_ok, confidence)
        w = month_frequency(full_q(res.q), B, n_months, T)
        return res, w, ens_raw, episode_ens(q_raw)

    def good(cand) -> bool:
        return cand[0].converged and cand[2] >= min_ens and cand[3] >= min_episodes

    first = solve(1.0)
    if good(first):
        res, w, _, epi = first
        return PhasePool(BlockProbs(full_q(res.q), B), w, prior_w, res, 1.0, effective_number(w), episodes=epi)
    reason = "thin" if first[0].converged else "infeasible"
    lo, hi, best = 0.0, 1.0, solve(0.0)
    for _ in range(SHRINK_STEPS):
        mid = 0.5 * (lo + hi)
        cand = solve(mid)
        if good(cand):
            lo, best = mid, cand
        else:
            hi = mid
    res, w, _, epi = best
    return PhasePool(BlockProbs(full_q(res.q), B), w, prior_w, res, lo, effective_number(w), reason,
                     first[2] if reason == "thin" else None, epi)


def view_support(frame: pd.DataFrame, col: str, n_months: int, block: int = BLOCK) -> tuple[float, float]:
    """2nd-98th percentile of what a phase of `n_months` can average for `col` (pool_phase clips to it)."""
    B = phase_block(block, n_months, len(frame))
    v = start_values(frame[col].to_numpy(float), B, n_months, propagate_nan=True)
    return support(v[np.isfinite(v)])
