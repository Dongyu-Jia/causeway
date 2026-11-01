"""Weighted block bootstrap over historical months.

Blocks of consecutive months keep short-run persistence (inflation inertia,
volatility clustering). A block's chance of being drawn is proportional to the
average posterior weight of the months inside it, so blocks that "look like"
the scenario are drawn more often.
"""
from __future__ import annotations

import numpy as np


def block_start_probs(weights: np.ndarray, block: int) -> np.ndarray:
    """Probability of starting a block at each feasible month."""
    w = np.asarray(weights, float)
    T = w.size
    block = max(1, min(block, T))
    csum = np.concatenate([[0.0], np.cumsum(w)])
    avg = (csum[block:] - csum[:-block]) / block      # length T - block + 1
    avg = np.maximum(avg, 0.0)
    total = avg.sum()
    if total <= 0:
        return np.full(avg.size, 1.0 / avg.size)
    return avg / total


def sample_phase(weights: np.ndarray, n_months: int, n_paths: int, block: int,
                 rng: np.random.Generator) -> np.ndarray:
    """(n_paths, n_months) array of month indices for one scenario phase."""
    if n_months <= 0:
        return np.zeros((n_paths, 0), dtype=np.int64)
    block = max(1, min(block, n_months, len(weights)))
    probs = block_start_probs(weights, block)
    n_blocks = int(np.ceil(n_months / block))
    starts = rng.choice(probs.size, size=(n_paths, n_blocks), p=probs)
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
    return idx[:, :n_months]


def sample_phases(phase_weights: list[np.ndarray], phase_months: list[int], n_paths: int,
                  block: int, seed: int = 0) -> np.ndarray:
    """Concatenate phases into (n_paths, sum(phase_months)) month indices."""
    rng = np.random.default_rng(seed)
    parts = [sample_phase(w, m, n_paths, block, rng) for w, m in zip(phase_weights, phase_months)]
    return np.concatenate(parts, axis=1) if parts else np.zeros((n_paths, 0), dtype=np.int64)
