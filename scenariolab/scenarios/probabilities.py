"""Scenario probabilities: subjective prior x likelihood(today)^gamma.

A transparent heuristic, not a forecasting model. For each scenario the
first phase's entropy-pooling weights describe the regime the scenario starts
in; the likelihood asks how close today's recent state (the average of the
last `window` months) is to that regime's posterior means, measured in units
of each state's historical standard deviation:

    z_k = (today_k - E_q[s_k]) / sd_k
    L   = exp(-0.5 * mean_k z_k^2)
    posterior ∝ prior * L^gamma

Averaging z^2 over the states (instead of summing) treats the correlated
states as one piece of evidence, and gamma < 1 tempers it further, so the data
can tilt the subjective priors but not overrule them.
"""
from __future__ import annotations

import numpy as np

from ..context import EngineContext
from .runner import phase_pools

KEY_STATES = ("cpi_yoy", "ip_yoy", "unrate", "ff", "spread", "oil_yoy", "y10")
GAMMA = 0.5
WINDOW = 6

METHOD = (
    "情景概率 = 主观先验 × 似然^γ，再归一化（γ={gamma}）。似然比较“今天”（最近 {window} 个月平均）的 "
    "{n} 个状态（CPI 同比、工业产出同比、失业率、联邦基金利率、信用利差、油价同比、10 年期收益率）"
    "与每个情景第一阶段会抽到的历史月份的均值，以各状态的历史标准差为单位计算距离 z，"
    "L = exp(−½·平均 z²)。这是一个透明的启发式：它只衡量今天的状态与情景起点的吻合程度"
    "（以急转折开局的情景会被适度压低），不是预测模型；γ<1 让数据只能调整、不能推翻主观先验。"
)


def today_state(ctx: EngineContext, window: int = WINDOW, states=KEY_STATES) -> dict[str, float]:
    """Average of the last `window` months of each state (percent units)."""
    tail = ctx.panel[[f"s_{s}" for s in states]].dropna(how="all").iloc[-window:]
    return {s: float(tail[f"s_{s}"].mean()) for s in states}


def state_scales(ctx: EngineContext, states=KEY_STATES) -> dict[str, float]:
    """Historical standard deviation of each state over the simulator's months."""
    return {s: float(ctx.db.frame[f"s_{s}"].std()) for s in states}


def phase_means(ctx: EngineContext, spec: dict, phase: int = 0, states=KEY_STATES) -> dict[str, float]:
    """Means of the key states over the months a phase draws (validated spec)."""
    q = phase_pools(ctx, {"phases": [spec["phases"][phase]]}, [])[0].month_weights
    out = {}
    for s in states:
        x = ctx.db.frame[f"s_{s}"].to_numpy(float)
        x = np.where(np.isnan(x), np.nanmean(x), x)
        out[s] = float(x @ q)
    return out


def posterior(items: list[dict], today: dict, scales: dict, gamma: float = GAMMA) -> list[dict]:
    """items: [{"id", "prior", "means"}] -> per-scenario prior/likelihood/posterior (all normalized).

    States that are missing or non-finite today or in a scenario are left out of its distance.
    """
    priors = np.array([max(float(it["prior"]), 0.0) for it in items])
    priors = priors / priors.sum() if priors.sum() > 0 else np.full(len(items), 1.0 / max(len(items), 1))
    out = []
    for it, prior in zip(items, priors):
        z = {s: (today[s] - it["means"][s]) / scales[s] for s in today
             if scales.get(s, 0) > 0 and np.isfinite(today[s]) and np.isfinite(it["means"].get(s, np.nan))}
        d2 = float(np.mean([v * v for v in z.values()])) if z else 0.0
        out.append({"id": it["id"], "prior": float(prior), "likelihood": float(np.exp(-0.5 * d2)),
                    "distance": float(np.sqrt(d2)), "z": z, "phase_means": it["means"]})
    w = np.array([o["prior"] * o["likelihood"] ** gamma for o in out])
    w = w / w.sum() if w.sum() > 0 else priors
    for o, p in zip(out, w):
        o["posterior"] = float(p)
    return out


def run(ctx: EngineContext, specs: list[dict], gamma: float = GAMMA, window: int = WINDOW) -> dict:
    """Probabilities for validated specs (e.g. detail["spec"] from run_spec)."""
    today = today_state(ctx, window)
    scales = state_scales(ctx)
    items = [{"id": s["id"], "prior": s.get("prior_prob", 0.0), "means": phase_means(ctx, s)} for s in specs]
    return {"method": METHOD.format(gamma=gamma, window=window, n=len(KEY_STATES)),
            "gamma": gamma, "window_months": window, "states": list(KEY_STATES),
            "today": today, "scales": scales,
            "date": ctx.panel.index.max().strftime("%Y-%m"),
            "scenarios": posterior(items, today, scales, gamma)}
