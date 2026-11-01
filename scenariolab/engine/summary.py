"""Turn simulated paths into chart-ready summaries (fans, candles, correlations)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PCTS = (5, 25, 50, 75, 95)


def month_labels(start: str, n_steps: int) -> list[str]:
    """Labels for level dates 0..n_steps (index 0 = the anchor month)."""
    d0 = pd.Timestamp(start)
    return [str((d0 + pd.DateOffset(months=k)).date())[:7] for k in range(n_steps + 1)]


def wealth_paths(total: np.ndarray) -> np.ndarray:
    """(n, T, A) returns -> (n, T+1, A) growth of 1, index 0 = 1."""
    n, T, A = total.shape
    g = np.cumprod(1.0 + total.astype(float), axis=1)
    return np.concatenate([np.ones((n, 1, A)), g], axis=1)


def fan(values: np.ndarray, pcts=PCTS) -> dict[str, list[float]]:
    """Percentile bands over paths for a (n, T+1) array."""
    q = np.percentile(values, pcts, axis=0)
    return {f"p{p}": [round(float(v), 5) for v in row] for p, row in zip(pcts, q)}


def max_drawdown(w: np.ndarray) -> np.ndarray:
    """Max drawdown per path for (n, T+1) wealth."""
    peak = np.maximum.accumulate(w, axis=1)
    return (w / peak - 1.0).min(axis=1)


def representative_path(w: np.ndarray) -> int:
    """Path whose terminal value and max drawdown ranks are both closest to the median."""
    term = w[:, -1]
    dd = max_drawdown(w)
    rt = term.argsort().argsort() / max(len(term) - 1, 1)
    rd = dd.argsort().argsort() / max(len(dd) - 1, 1)
    score = (rt - 0.5) ** 2 + (rd - 0.5) ** 2
    return int(score.argmin())


def candles(monthly_returns: np.ndarray, monthly_vol: float, labels: list[str],
            seed: int = 0, steps: int = 21, start_price: float = 100.0) -> list[list]:
    """Monthly OHLC for one path, with intramonth highs/lows from a Brownian bridge.

    The month's open and close are exact; the bridge only shapes the wicks,
    using the asset's typical monthly volatility.
    """
    rng = np.random.default_rng(seed)
    out = []
    price = start_price
    sigma = monthly_vol / np.sqrt(steps)
    for k, r in enumerate(monthly_returns):
        r = float(max(r, -0.95))
        target = np.log1p(r)
        w = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, sigma, steps))])
        tgrid = np.linspace(0.0, 1.0, steps + 1)
        bridge = w - tgrid * w[-1] + tgrid * target
        path = price * np.exp(bridge)
        o, c = float(price), float(price * np.exp(target))
        out.append([labels[k + 1], round(o, 3), round(c, 3), round(float(path.min()), 3), round(float(path.max()), 3)])
        price = c
    return out  # [label, open, close, low, high] (ECharts candlestick order)


def correlation(total: np.ndarray, assets: list[str]) -> dict:
    """Correlation of monthly returns pooled across paths and months."""
    flat = total.reshape(-1, total.shape[2]).astype(float)
    c = np.corrcoef(flat, rowvar=False)
    return {"assets": assets, "matrix": [[round(float(v), 3) for v in row] for row in c]}


def annualized_stats(total: np.ndarray, assets: list[str]) -> dict:
    """Per-asset CAGR percentiles and average annualized volatility."""
    n, T, A = total.shape
    years = T / 12.0
    w = wealth_paths(total)
    out = {}
    for j, a in enumerate(assets):
        cagr = w[:, -1, j] ** (1.0 / years) - 1.0
        vol = total[:, :, j].std(axis=1) * np.sqrt(12.0)
        out[a] = {
            "cagr_p5": float(np.percentile(cagr, 5)), "cagr_p50": float(np.percentile(cagr, 50)),
            "cagr_p95": float(np.percentile(cagr, 95)), "cagr_mean": float(cagr.mean()),
            "vol": float(vol.mean()), "max_dd_p50": float(np.percentile(max_drawdown(w[:, :, j]), 50)),
        }
    return out
