"""Weighted block bootstrap over historical months.

Blocks of consecutive months keep short-run persistence (inflation inertia,
volatility clustering). A phase is drawn from block-start probabilities: either
given directly (BlockProbs, from pooling.pool_phase, which solves the views on
block means so the drawn months honour them), or derived from month weights,
in which case a block's chance is proportional to the average weight of the
months inside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockProbs:
    """Block-start probabilities for one phase: probs[s] = P(a block starts at month s)."""
    probs: np.ndarray            # length T - block + 1
    block: int


def phase_block(block: int, n_months: int, T: int) -> int:
    """Block length used for a phase of `n_months` drawn from T months."""
    return max(1, min(block, n_months, T))


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


def sample_phase(weights, n_months: int, n_paths: int, block: int,
                 rng: np.random.Generator) -> np.ndarray:
    """(n_paths, n_months) array of month indices for one scenario phase.

    `weights` is a BlockProbs or an array of month weights.
    """
    if n_months <= 0:
        return np.zeros((n_paths, 0), dtype=np.int64)
    if isinstance(weights, BlockProbs):
        B = weights.block
        probs = np.asarray(weights.probs, float)
        probs = probs / probs.sum()
    else:
        B = phase_block(block, n_months, len(weights))
        probs = block_start_probs(weights, B)
    n_blocks = int(np.ceil(n_months / B))
    starts = rng.choice(probs.size, size=(n_paths, n_blocks), p=probs)
    offsets = np.arange(B)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
    return idx[:, :n_months]


def sample_phases(phase_weights: list, phase_months: list[int], n_paths: int,
                  block: int, seed: int = 0) -> np.ndarray:
    """Concatenate phases into (n_paths, sum(phase_months)) month indices."""
    rng = np.random.default_rng(seed)
    parts = [sample_phase(w, m, n_paths, block, rng) for w, m in zip(phase_weights, phase_months)]
    return np.concatenate(parts, axis=1) if parts else np.zeros((n_paths, 0), dtype=np.int64)
