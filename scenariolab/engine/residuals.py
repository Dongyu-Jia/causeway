"""Residual draws: where a simulated month's unexplained part comes from.

Every simulated month adds to its factor-explained return the factor-model
residual of a historical month. The residual carries what the macro factors do
not explain: cross-asset co-movement, fat tails, and also slow, era-specific
drifts (the late-1990s tech bubble, the 2001-02 bust, re-ratings).

modes (simulate(resid_mode=...)):

coupled          the residual of the same month whose factor changes were drawn
                 (the engine default). Under views the drawn months are tilted
                 toward some episodes, and those episodes' residual drifts come
                 along whether or not they have anything to do with the views.
decoupled        residual vectors from an independent block bootstrap under the
                 prior, with the sampler's own block structure: the cross-asset
                 structure and within-block persistence are kept, the
                 macro -> residual alignment is removed.
detrended        the drawn month's residual minus a centered rolling mean of the
                 residuals (`window` months, estimated on the month database only,
                 i.e. the training window in a backtest), plus the constant that
                 restores the prior mean. Removes era drifts but also their share
                 of long-horizon variance.
trend_decoupled  the fast part (residual minus rolling mean) stays coupled, the
                 slow part (the rolling mean) comes from the independent prior
                 draw: variance of both parts kept, only the slow alignment removed.
partial          per sampler block, the coupled residual with probability
                 `coupling`, otherwise the decoupled one (a mixture: the marginal
                 distribution of a month's residual is unchanged).
recentered       the coupled residuals, shifted per phase so that their expected
                 mean moves from what the phase's weights draw toward what the
                 prior draws: shift = (1 - coupling) * (E_q[resid] - E_prior[resid]).
                 Shocks, cross-asset structure and their timing stay coupled; only
                 the conditional drift is shrunk (coupling = 0 removes it).

Every mode has the same expected residual under unconditional (prior)
sampling as the coupled draw, so none adds a drift of its own (recentered
leaves an unconditional simulation bit-identical).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pooling import month_frequency, start_prior
from .sampler import BlockProbs, block_start_probs, phase_block, sample_phases

MODES = ("coupled", "decoupled", "detrended", "trend_decoupled", "partial", "recentered")
DEFAULT_WINDOW = 36
SEED_OFFSET = 7919          # the independent residual bootstrap's seed = seed + SEED_OFFSET


def rolling_mean(resid: pd.DataFrame, window: int) -> pd.DataFrame:
    """Centered `window`-month rolling mean of each column, over calendar months of the frame itself.

    Only the frame's own months enter (the training window in a backtest), so nothing after its
    last month is used; near either end the window is truncated (at least window // 2 months).
    """
    if window < 2:
        raise ValueError("window must be at least 2 months")
    idx = resid.index
    full = pd.date_range(idx.min(), idx.max(), freq="MS")
    r = resid.reindex(full)
    m = r.rolling(window, center=True, min_periods=max(window // 2, 1)).mean()
    return m.reindex(idx)


def prior_frequency(prior: np.ndarray, block: int, rare: np.ndarray | None = None) -> np.ndarray:
    """Expected share of each month among unconditional draws (full blocks from prior weights)."""
    T = len(prior)
    B = phase_block(block, block, T)
    return month_frequency(start_prior(prior, B, rare), B, B, T)


def prior_blocks(db, phase_months: list[int], block: int) -> list[BlockProbs]:
    """The unconditional block-start probabilities of every phase (rare months per block, as pooling does)."""
    T, rare = len(db), getattr(db, "rare", None)
    out = []
    for m in phase_months:
        B = phase_block(block, m, T)
        out.append(BlockProbs(start_prior(db.prior(), B, rare), B))
    return out


def components(db, window: int | None = None, block: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """(fast, slow) residual parts of the month database, (T, assets) each.

    slow = centered rolling mean minus its prior-frequency mean; fast = resid - slow. So
    fast + slow = resid, and under unconditional sampling the slow part has mean zero.
    """
    R = db.resid
    M = rolling_mean(R, window or DEFAULT_WINDOW).to_numpy(float)
    M = np.where(np.isfinite(M), M, 0.0)
    f = prior_frequency(db.prior(), block, getattr(db, "rare", None))
    slow = M - f @ M
    return R.to_numpy(float) - slow, slow


def phase_blocks(phase_weights: list, phase_months: list[int], block: int, T: int) -> list[int]:
    """Block length the sampler uses in each phase (BlockProbs carry their own)."""
    return [w.block if isinstance(w, BlockProbs) else phase_block(block, m, T)
            for w, m in zip(phase_weights, phase_months)]


def phase_frequencies(phase_weights: list, phase_months: list[int], block: int, T: int) -> list[np.ndarray]:
    """Per phase, the expected share of each month among that phase's draws (as the sampler draws)."""
    out = []
    for w, m in zip(phase_weights, phase_months):
        if isinstance(w, BlockProbs):                  # pooling and start_prior already normalize
            out.append(month_frequency(np.asarray(w.probs, float), w.block, m, T))
        else:
            B = phase_block(block, m, T)
            out.append(month_frequency(block_start_probs(w, B), B, m, T))
    return out


def conditional_drift(db, phase_weights: list, phase_months: list[int], block: int) -> np.ndarray:
    """(phases, model assets) expected drawn residual mean of each phase minus the prior's (monthly units)."""
    R, T = db.resid.to_numpy(float), len(db)
    cond = phase_frequencies(phase_weights, phase_months, block, T)
    prior = phase_frequencies(prior_blocks(db, phase_months, block), phase_months, block, T)
    return np.array([(q - p) @ R for q, p in zip(cond, prior)]).reshape(len(phase_months), R.shape[1])


def block_ids(phase_months: list[int], blocks: list[int]) -> np.ndarray:
    """Global block number of every simulated month (blocks restart at each phase)."""
    out, nxt = [], 0
    for m, B in zip(phase_months, blocks):
        k = np.arange(m) // max(B, 1)
        out.append(nxt + k)
        nxt += int(k.max()) + 1 if m else 0
    return np.concatenate(out) if out else np.zeros(0, int)


def prior_draw(db, phase_months: list[int], n_paths: int, block: int, seed: int) -> np.ndarray:
    """(n_paths, T) month positions from an independent block bootstrap under the prior."""
    return sample_phases(prior_blocks(db, phase_months, block), phase_months, n_paths, block, seed)


def draw(db, idx: np.ndarray, phase_weights: list, phase_months: list[int], block: int, seed: int,
         mode: str = "coupled", window: int | None = None, coupling: float = 1.0,
         resid_seed: int | None = None) -> np.ndarray:
    """(n_paths, T, model assets) residuals for the simulated months `idx` (sampler positions in db)."""
    if mode not in MODES:
        raise ValueError(f"unknown resid_mode {mode!r}; expected one of {MODES}")
    R = db.resid.to_numpy(float)
    if mode == "coupled":
        return R[idx]
    n_paths = idx.shape[0]
    rseed = seed + SEED_OFFSET if resid_seed is None else int(resid_seed)
    if mode == "detrended":
        fast, _ = components(db, window, block)
        return fast[idx]
    if mode == "recentered":
        shift = (1.0 - float(np.clip(coupling, 0.0, 1.0))) * conditional_drift(db, phase_weights, phase_months, block)
        per_month = np.repeat(shift, phase_months, axis=0) if len(phase_months) else shift
        return R[idx] - per_month[None, :, :]
    other = prior_draw(db, phase_months, n_paths, block, rseed)
    if mode == "decoupled":
        return R[other]
    if mode == "trend_decoupled":
        fast, slow = components(db, window, block)
        return fast[idx] + slow[other]
    # partial: per (path, sampler block) Bernoulli(coupling) choice between coupled and decoupled
    pi = float(np.clip(coupling, 0.0, 1.0))
    ids = block_ids(phase_months, phase_blocks(phase_weights, phase_months, block, len(db)))
    rng = np.random.default_rng([rseed, 1])
    keep = rng.random((n_paths, int(ids.max()) + 1 if ids.size else 0)) < pi
    mask = keep[:, ids]                                         # (n, T)
    return np.where(mask[:, :, None], R[idx], R[other])
