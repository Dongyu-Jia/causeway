"""Entropy Pooling (Meucci 2008).

Re-weight historical months so that a set of views holds on average, while the
new weights q stay as close as possible to the prior p in relative entropy:

    min_q  sum_t q_t log(q_t / p_t)
    s.t.   sum_t q_t f_k(t)  = b_k      (equality views)
           sum_t q_t g_j(t) <= h_j      (inequality views)
           sum_t q_t = 1

The solution has the form q_t ∝ p_t exp(-lambda·f(t) - mu·g(t)), so only the
few multipliers are solved for (a smooth convex dual). Every simulated month is
still a real historical month; views only change how often each one is drawn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass
class EPResult:
    q: np.ndarray
    converged: bool
    max_violation: float          # largest |E_q[f] - b| in standardized units
    ens: float                    # effective number of scenarios exp(H(q))
    ens_ratio: float              # ens / number of months
    kl: float                     # KL(q || p)
    views: list[dict] = field(default_factory=list)  # per-view report


def effective_number(q: np.ndarray) -> float:
    q = q[q > 0]
    return float(np.exp(-(q * np.log(q)).sum()))


def kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
    m = q > 0
    return float((q[m] * np.log(q[m] / p[m])).sum())


def entropy_pool(p: np.ndarray,
                 A_eq: np.ndarray | None = None, b_eq: np.ndarray | None = None,
                 A_ineq: np.ndarray | None = None, b_ineq: np.ndarray | None = None,
                 max_iter: int = 1000) -> tuple[np.ndarray, bool, float]:
    """Solve the entropy-pooling problem. A_* are (k, T); returns (q, ok, viol)."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    T = p.size
    A_eq = np.zeros((0, T)) if A_eq is None else np.atleast_2d(np.asarray(A_eq, float))
    b_eq = np.zeros(0) if b_eq is None else np.atleast_1d(np.asarray(b_eq, float))
    A_in = np.zeros((0, T)) if A_ineq is None else np.atleast_2d(np.asarray(A_ineq, float))
    b_in = np.zeros(0) if b_ineq is None else np.atleast_1d(np.asarray(b_ineq, float))
    k_eq, k_in = A_eq.shape[0], A_in.shape[0]
    if k_eq + k_in == 0:
        return p.copy(), True, 0.0

    # Standardize each constraint row by its prior mean / sd for conditioning.
    def standardize(A, b):
        if A.shape[0] == 0:
            return A, b
        mu = A @ p
        sd = np.sqrt(np.maximum(((A - mu[:, None]) ** 2) @ p, 1e-18))
        return (A - mu[:, None]) / sd[:, None], (b - mu) / sd

    F, bf = standardize(A_eq, b_eq)
    G, hg = standardize(A_in, b_in)
    M = np.vstack([F, G])
    rhs = np.concatenate([bf, hg])
    logp = np.log(np.maximum(p, 1e-300))

    def dual(theta):
        z = logp - M.T @ theta
        lse = logsumexp(z)
        q = np.exp(z - lse)
        val = lse + theta @ rhs
        grad = rhs - M @ q
        return val, grad

    bounds = [(None, None)] * k_eq + [(0.0, None)] * k_in
    res = minimize(dual, np.zeros(k_eq + k_in), jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": max_iter, "gtol": 1e-10, "ftol": 1e-14})
    theta = res.x
    z = logp - M.T @ theta
    q = np.exp(z - logsumexp(z))
    viol_eq = np.abs(F @ q - bf).max() if k_eq else 0.0
    viol_in = np.maximum(G @ q - hg, 0.0).max() if k_in else 0.0
    viol = float(max(viol_eq, viol_in))
    return q, bool(viol < 1e-4), viol


def apply_views(frame: pd.DataFrame, views: list[dict], p: np.ndarray | None = None,
                confidence: float = 1.0) -> EPResult:
    """Translate view dicts into constraints on `frame` columns and solve.

    View forms (values in the column's native monthly units unless noted):
      {"var": col, "mean": x}            E_q[col] = x
      {"var": col, "mean_annual": x}     E_q[col] = x / 12  (for monthly changes)
      {"var": col, "shift_sd": k}        E_q[col] = prior mean + k * prior sd
      {"var": col, "min": x} / {"max": x}  inequality on the mean (also *_annual)
      {"var": col, "vol_mult": k}        E_q[(col - m)^2] = (k * prior sd)^2,
                                         m = the view mean if given, else prior mean
    Targets outside the historical range are clipped to the 2nd/98th percentile
    of the column (reported per view), since no reweighting can reach them.
    `confidence` < 1 blends the posterior back toward the prior (opinion pooling).
    """
    T = len(frame)
    p = np.full(T, 1.0 / T) if p is None else np.asarray(p, float) / np.sum(p)
    A_eq, b_eq, A_in, b_in, report = [], [], [], [], []
    means: dict[str, float] = {}
    for v in views:
        col = v["var"]
        x = frame[col].to_numpy(dtype=float)
        if np.isnan(x).any():
            x = np.where(np.isnan(x), np.nanmean(x), x)
        prior_mean = float(x @ p)
        prior_sd = float(np.sqrt(((x - prior_mean) ** 2) @ p))
        lo, hi = np.percentile(x, [2, 98])
        entry = {"var": col, "prior_mean": prior_mean, "prior_sd": prior_sd, "clipped": False}

        target = None
        if "mean" in v:
            target = float(v["mean"])
        elif "mean_annual" in v:
            target = float(v["mean_annual"]) / 12.0
        elif "shift_sd" in v:
            target = prior_mean + float(v["shift_sd"]) * prior_sd
        if target is not None:
            if target < lo or target > hi:
                entry["clipped"] = True
                entry["requested"] = target
                target = float(np.clip(target, lo, hi))
            A_eq.append(x)
            b_eq.append(target)
            means[col] = target
            entry.update({"kind": "mean", "target": target})

        for key, sign in (("min", -1.0), ("max", 1.0), ("min_annual", -1.0), ("max_annual", 1.0)):
            if key in v:
                val = float(v[key]) / (12.0 if key.endswith("_annual") else 1.0)
                val = float(np.clip(val, lo, hi))
                A_in.append(sign * x)
                b_in.append(sign * val)
                entry.update({"kind": key.split("_")[0], "target": val})

        if "vol_mult" in v:
            m = means.get(col, prior_mean)
            target_var = (float(v["vol_mult"]) * prior_sd) ** 2
            A_eq.append((x - m) ** 2)
            b_eq.append(target_var)
            entry.update({"vol_mult": float(v["vol_mult"])})
        report.append(entry)

    q, ok, viol = entropy_pool(p,
                               np.array(A_eq) if A_eq else None, np.array(b_eq) if b_eq else None,
                               np.array(A_in) if A_in else None, np.array(b_in) if b_in else None)
    if confidence < 1.0:
        q = confidence * q + (1.0 - confidence) * p
    for entry in report:
        x = frame[entry["var"]].to_numpy(dtype=float)
        x = np.where(np.isnan(x), np.nanmean(x), x)
        entry["posterior_mean"] = float(x @ q)
        entry["posterior_sd"] = float(np.sqrt(((x - x @ q) ** 2) @ q))
    ens = effective_number(q)
    return EPResult(q=q, converged=ok, max_violation=viol, ens=ens, ens_ratio=ens / T,
                    kl=kl_divergence(q, p), views=report)


def weighted_corr(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Correlation matrix of the columns of X under probability weights w."""
    w = np.asarray(w, float) / np.sum(w)
    mu = w @ X
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    return cov / np.outer(sd, sd)
