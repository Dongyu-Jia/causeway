"""Random streams of the instrument layer (INSTRUMENTS_DESIGN §2.7).

A stream is keyed by (set id, scenario id or node key, purpose[, ticker]) through crc32, so
  * shared streams (one per scenario: the volatility shocks, the bridge noise, the joint residual dates) keep the
    leveraged funds, QQQI and the estimated tickers consistent with each other;
  * every ticker has its own streams (ruin draws, QQQI's month split, TLT's spread noise, the estimated monthly
    split): changing one ticker's parameters never changes another ticker's draws (T19);
  * draws fixed per ROW (alpha) are keyed by the row's (group, path index), so a path keeps its alpha in the
    scenario years and in the unconditional continuation that follows it.
"""
from __future__ import annotations

import zlib

import numpy as np


def stream(*keys) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(str(k).encode("utf-8")) for k in keys])


def row_normals(set_id: str, groups: np.ndarray, index: np.ndarray, *purpose) -> np.ndarray:
    """One standard normal per row, fixed by (set_id, the row's group, purpose...) and the row's path index."""
    groups = np.asarray(groups).astype(str)
    index = np.asarray(index, dtype=np.int64)
    out = np.empty(len(index))
    for g in np.unique(groups):
        m = groups == g
        k = index[m]
        z = stream(set_id, g, *purpose).standard_normal(int(k.max()) + 1)
        out[m] = z[k]
    return out
