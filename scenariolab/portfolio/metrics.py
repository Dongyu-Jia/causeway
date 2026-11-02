"""Portfolio statistics on simulated scenario paths.

Pure numpy, no I/O. Conventions: monthly decimal simple returns (0.01 = 1%),
wealth paths are (n, T+1) with index 0 = the starting value, 12 months per
year, percentiles in 0..100. Weighted statistics take per-path probability
weights of any positive scale (they are renormalized); a mixture of scenario
path sets is expressed through mixture_weights().
"""
from __future__ import annotations

import numpy as np

PCTS = (5, 25, 50, 75, 95)
MONTHS_PER_YEAR = 12
REBALANCE_MODES = ("monthly", "annual", "none")


# ---------------------------------------------------------------- wealth

def portfolio_wealth(total: np.ndarray, weights: np.ndarray, rebalance: str = "monthly",
                     period: int = MONTHS_PER_YEAR) -> np.ndarray:
    """(n, T, A) asset total returns -> (n, T+1) wealth of 1 invested at t=0.

    monthly: back to the target weights every month; annual: every `period`
    months (month 12, 24, ...), buy-and-hold in between; none: buy-and-hold
    over the whole horizon. Total returns compound, i.e. income is reinvested
    in the asset that paid it.
    """
    if rebalance not in REBALANCE_MODES:
        raise ValueError(f"rebalance 必须是 {'/'.join(REBALANCE_MODES)} 之一")
    total = np.asarray(total)
    n, T, A = total.shape
    w = np.asarray(weights, dtype=np.float64)
    out = np.empty((n, T + 1))
    out[:, 0] = 1.0
    if rebalance == "monthly":
        dtype = total.dtype if total.dtype in (np.float32, np.float64) else np.float64
        rp = (total.reshape(-1, A) @ w.astype(dtype)).reshape(n, T)
        out[:, 1:] = np.cumprod(1.0 + rp.astype(np.float64), axis=1)
        return out
    h = np.tile(w, (n, 1))                      # holdings per unit of initial wealth
    for s in range(0, T, period):
        g = np.cumprod(1.0 + total[:, s:s + period, :].astype(np.float64), axis=1)
        vals = np.einsum("na,nma->nm", h, g)
        out[:, s + 1:s + 1 + g.shape[1]] = vals
        h = vals[:, -1:] * w if rebalance == "annual" else h * g[:, -1, :]
    return out


def period_returns(wealth: np.ndarray) -> np.ndarray:
    """(n, T+1) wealth -> (n, T) simple returns (NaN after a path's wealth reaches 0, e.g. a portfolio of one
    leveraged fund that is wiped out; the per-path statistics then leave that path out)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return wealth[:, 1:] / wealth[:, :-1] - 1.0


def drawdowns(wealth: np.ndarray) -> np.ndarray:
    """(n, T+1) wealth -> drawdown from the running peak (<= 0, decimal)."""
    return wealth / np.maximum.accumulate(wealth, axis=1) - 1.0


def max_drawdown(wealth: np.ndarray) -> np.ndarray:
    """Worst peak-to-trough loss per path (negative decimal, -0.5 = halved)."""
    return drawdowns(wealth).min(axis=1)


def cagr(multiple: np.ndarray, years: float) -> np.ndarray:
    """Compound annual growth rate of a wealth multiple over `years`."""
    return np.maximum(np.asarray(multiple, dtype=float), 1e-12) ** (1.0 / years) - 1.0


def annualized_vol(returns: np.ndarray, periods: int = MONTHS_PER_YEAR) -> np.ndarray:
    """Per-path standard deviation of period returns, annualized by sqrt(periods)."""
    return returns.std(axis=1) * np.sqrt(periods)


def sharpe(returns: np.ndarray, rf: np.ndarray | float = 0.0, periods: int = MONTHS_PER_YEAR) -> np.ndarray:
    """Per-path Sharpe ratio: annualized mean excess return over `rf` / annualized vol.

    `rf` is the cash return per period ((n, T) or scalar); the denominator is
    the volatility of the portfolio's own returns (the reported "vol"). NaN
    where that volatility is ~0.
    """
    excess = (returns - rf).mean(axis=1) * periods
    vol = annualized_vol(returns, periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vol > 1e-10, excess / vol, np.nan)


def sortino(returns: np.ndarray, rf: np.ndarray | float = 0.0, periods: int = MONTHS_PER_YEAR) -> np.ndarray:
    """Per-path Sortino ratio: annualized mean excess return / annualized downside deviation.

    Downside deviation = sqrt(mean over all periods of min(r - rf, 0)^2), i.e.
    the target is the cash return. NaN for paths that never fall below it.
    """
    ex = returns - rf
    down = np.sqrt((np.minimum(ex, 0.0) ** 2).mean(axis=1) * periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(down > 1e-10, ex.mean(axis=1) * periods / down, np.nan)


# ---------------------------------------------------------------- weighted statistics

def clean_sample(values, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a sample and its weights, dropping NaNs and non-positive weights."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.ones(v.size) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != v.size:
        raise ValueError("weights and values differ in length")
    keep = np.isfinite(v) & (w > 0)
    return v[keep], w[keep]


def weighted_percentile(values, pcts=PCTS, weights=None):
    """Percentiles (0..100) of a 1-d sample under probability weights.

    The i-th smallest value sits at plotting position (S_i - w_i / 2) / W
    (S_i = cumulative weight, W = total weight) with linear interpolation in
    between and clamping outside, so equal weights reproduce
    np.percentile(..., method="hazen"). NaNs and zero-weight values are
    dropped; an empty sample gives NaN. Scalar `pcts` -> float.
    """
    q = np.asarray(pcts, dtype=float)
    v, w = clean_sample(values, weights)
    if v.size == 0:
        return float("nan") if q.ndim == 0 else np.full(q.shape, np.nan)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    out = np.interp(q / 100.0, (cw - 0.5 * w) / cw[-1], v)
    return float(out) if q.ndim == 0 else out


def weighted_mean(values, weights=None) -> float:
    v, w = clean_sample(values, weights)
    return float((v * w).sum() / w.sum()) if v.size else float("nan")


def weighted_cvar(values, alpha: float = 0.05, weights=None) -> float:
    """Expected value of the worst `alpha` probability mass (lower tail) of a sample."""
    v, w = clean_sample(values, weights)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order] / w.sum()
    take = np.clip(alpha - (np.cumsum(w) - w), 0.0, w)      # tail mass taken from each value
    return float((take * v).sum() / take.sum())


def fan(values: np.ndarray, weights=None, pcts=PCTS, decimals: int = 6) -> dict[str, list[float]]:
    """Weighted percentile bands over paths (rows) for every column of an (n, K) array."""
    q = np.column_stack([weighted_percentile(values[:, k], pcts, weights) for k in range(values.shape[1])])
    return {f"p{p}": [round(float(x), decimals) for x in row] for p, row in zip(pcts, q)}


def normalize_probs(probs) -> np.ndarray:
    """Non-negative probabilities renormalized to sum to 1 (any finite scale, no overflow)."""
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("情景概率必须是非负数")
    if not p.size or p.max() <= 0:
        raise ValueError("情景概率不能全为 0")
    with np.errstate(over="ignore"):
        if not np.isfinite(p.sum()):
            p = p / p.max()
    return p / p.sum()


def mixture_weights(groups: np.ndarray, probs) -> np.ndarray:
    """Per-path weights of a probability mixture of path sets.

    groups: (n,) index of the set (scenario) each path belongs to; probs: (S,)
    set probabilities (renormalized). A set's probability is shared equally by
    its paths, so sets with different path counts mix by probability, not by
    path count. Weights sum to 1.
    """
    groups = np.asarray(groups, dtype=np.int64)
    p = normalize_probs(probs)
    counts = np.bincount(groups, minlength=p.size).astype(float)
    w = p[groups] / counts[groups]
    return w / w.sum()


# ---------------------------------------------------------------- summaries

def path_metrics(wealth: np.ndarray, deflator: np.ndarray, rf: np.ndarray | float = 0.0,
                 periods: int = MONTHS_PER_YEAR) -> dict[str, np.ndarray]:
    """Per-path statistics of (n, T+1) nominal wealth.

    deflator: (n, T+1) price index (any base; real = nominal / (index_t / index_0));
    rf: cash return per period, (n, T) or scalar. Returns arrays of shape (n,):
    terminal / terminal_real wealth multiples, cagr / real_cagr, vol (annualized),
    sharpe, sortino, max_dd (negative decimal).
    """
    years = (wealth.shape[1] - 1) / periods
    r = period_returns(wealth)
    terminal = wealth[:, -1] / wealth[:, 0]
    terminal_real = terminal / (deflator[:, -1] / deflator[:, 0])
    return {"terminal": terminal, "terminal_real": terminal_real,
            "cagr": cagr(terminal, years), "real_cagr": cagr(terminal_real, years),
            "vol": annualized_vol(r, periods), "sharpe": sharpe(r, rf, periods),
            "sortino": sortino(r, rf, periods), "max_dd": max_drawdown(wealth)}


def summarize(m: dict[str, np.ndarray], weights=None, alpha: float = 0.05) -> dict[str, float]:
    """Distribution summary of path_metrics() output under optional path weights.

    vol is the mean of per-path vols; sharpe / sortino are medians of per-path
    ratios; max_dd_p95 is the bad tail (the drawdown only 5% of paths exceed,
    i.e. the 5th percentile of the negative values); cvar5_terminal is the
    mean terminal wealth multiple of the worst 5% of paths; prob_loss_real is
    the probability that real terminal wealth ends below the start.
    """
    def pct(key: str, p: float) -> float:
        return weighted_percentile(m[key], p, weights)

    return {
        "cagr_p5": pct("cagr", 5), "cagr_p50": pct("cagr", 50), "cagr_p95": pct("cagr", 95),
        "real_cagr_p5": pct("real_cagr", 5), "real_cagr_p50": pct("real_cagr", 50),
        "real_cagr_p95": pct("real_cagr", 95),
        "vol": weighted_mean(m["vol"], weights),
        "sharpe": pct("sharpe", 50), "sortino": pct("sortino", 50),
        "max_dd_p50": pct("max_dd", 50), "max_dd_p95": pct("max_dd", 5),
        "terminal_p50": pct("terminal", 50), "terminal_real_p50": pct("terminal_real", 50),
        "cvar5_terminal": weighted_cvar(m["terminal"], alpha, weights),
        "cvar5_terminal_real": weighted_cvar(m["terminal_real"], alpha, weights),
        "prob_loss_real": weighted_mean(m["terminal_real"] < 1.0, weights),
    }
