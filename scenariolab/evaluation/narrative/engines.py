"""The monthly engine as the narrative test's engine arm E (SIM_DESIGN §8.3, §4.6, §7.4).

E is today's monthly engine running the selected model F (data/results/model_selection.json; the research round ran
B0), fitted on months up to the episode's train end (analysis/backtest.fit_window: the model, the month database
and the anchor), with the long-run anchors knowable at the train end (engine/anchors.py, §5.5: the backtests' rule)
and CMA "historical" (training drift), as the monthly backtest windows run. It reads the SAME QTable the quarterly
stack reads, converted by qtable.to_phase_spec (each stated field becomes a monthly view, a credit stance f_spread and
f_gz views, u_ff an f_ff view; fields without a monthly column are dropped with 「月度引擎没有这个字段」), which
removes the research round's confound of separate spec and table translations. Oracle arms are QTables of realized
values, converted the same way. The structural shock columns (e_*) are not inputs (§8.4).

The engine's monthly paths are summed to the window's calendar quarters: log price returns (log1p of the simple price
return, floored at -95% as the research scorer did), cash its total return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ... import config

MIN_TRAIN_MONTHS = 120                         # backtest.MIN_TRAIN_MONTHS: fewer and the engine is not run


class EngineSkip(RuntimeError):
    """The engine cannot run this episode (Chinese reason)."""


_PANEL: dict = {}


def engine_panel() -> pd.DataFrame:
    """The monthly panel with the structural shocks dropped (no e_* inputs, §8.4)."""
    if "p" not in _PANEL:
        from ...context import load_panel_with_shocks
        p = load_panel_with_shocks()
        _PANEL["p"] = p.drop(columns=[c for c in p.columns if c.startswith("e_")])
    return _PANEL["p"]


def train_end_month(train_end_q: str) -> pd.Timestamp:
    return pd.Period(train_end_q, "Q").end_time.to_period("M").to_timestamp()


def setup(train_end_q: str) -> dict:
    """The engine fitted to the train end: {"ctx", "reversion", "anchors", "T0", "spec_name"}."""
    from ...analysis import backtest as bt
    from ...context import EngineContext, selected_spec
    from ...data.panel import load_meta
    from ...engine import anchors as long_run
    from ...engine.factor_model import ModelSpec
    from ...engine.simulate import capital_market_assumptions
    panel = engine_panel()
    T0 = train_end_month(train_end_q)
    n_train = int(((panel.index >= pd.Timestamp(config.MODEL_START)) & (panel.index <= T0)).sum())
    if n_train < MIN_TRAIN_MONTHS:
        raise EngineSkip(f"训练期只有 {n_train} 个月（月度引擎至少要 {MIN_TRAIN_MONTHS} 个月）：没有运行月度引擎")
    spec = selected_spec() or ModelSpec()
    if any(c.startswith("e_") for c in getattr(spec, "shock_cols", []) or []):
        spec = ModelSpec.from_dict({**spec.to_dict(), "shock_cols": []})
    model, db, anchor = bt.fit_window(panel, spec, T0)
    notes = []
    try:
        lr = long_run.train_end_anchors(panel, T0)
        rev = long_run.reversion(lr)
    except ValueError as err:
        lr, rev = None, {}
        notes.append(f"训练期末可知的长期锚算不出来（{err}）：用默认长期锚")
    cma = capital_market_assumptions(db, anchor, method="historical")
    ctx = EngineContext(panel=panel.loc[:T0], meta=load_meta(), spec=spec, model=model, db=db, anchor=anchor, cma=cma)
    return {"ctx": ctx, "reversion": rev, "anchors": lr, "T0": T0, "spec_name": getattr(spec, "name", ""),
            "notes": notes, "n_train_months": n_train}


def phase_spec(qt: dict, Q: pd.DataFrame, train_end_q: str, H: int, *, stats: dict | None = None) -> tuple[dict, dict]:
    """The monthly phase spec of a clean QTable for an H-quarter window (no tail phase)."""
    from ...qmodel import qtable as QT
    stats = stats if stats is not None else QT.stats_table(Q, train_end_q, shocks=True)
    return QT.to_phase_spec(qt, Q=Q, train_end=train_end_q, stats=stats, validated=True, mode="test",
                            horizon_months=3 * int(H))


def run(st: dict, spec: dict, H: int, n_paths: int, seed: int) -> tuple[np.ndarray, dict]:
    """(paths (n, H, 13) quarterly log price returns in config.ASSET_IDS order, report) of one phase spec."""
    from ...engine.pooling import BLOCK
    from ...engine.simulate import simulate, with_oil_lock
    from ...scenarios import runner as srun
    from ...scenarios.schema import validate_spec
    ctx = st["ctx"]
    clean, report = validate_spec(spec, ctx.db.frame, 3 * int(H), model_spec=ctx.model.spec)
    pools = srun.phase_pools(ctx, clean, report["warnings"])
    weights = [p.blocks for p in pools]
    overrides = with_oil_lock([p["views"] for p in clean["phases"]], [p.get("overrides", {}) for p in clean["phases"]])
    months = [p["months"] for p in clean["phases"]]
    sim = simulate(ctx.db, ctx.anchor, weights, months, n_paths=int(n_paths), seed=int(seed), block=BLOCK,
                   cma=ctx.cma, overrides=overrides, reversion=st["reversion"] or None, valuation=None,
                   **srun.RESIDUALS)
    T = 3 * int(H)
    tot = sim.total[:, :T, :].astype(float)
    inc = sim.income[:, :T, :].astype(float)
    m = np.log1p(np.maximum(tot - inc, -0.95))
    j = list(sim.assets).index("cash")
    m[:, :, j] = np.log1p(tot[:, :, j])
    order = [list(sim.assets).index(a) for a in config.ASSET_IDS]
    q = m[:, :, order].reshape(m.shape[0], int(H), 3, len(order)).sum(axis=2)
    shrink = [float(p.shrink) for p in pools]
    return q, {"phases": [{"name": p["name"], "months": p["months"], "views": [v["var"] for v in p["views"]]}
                          for p in clean["phases"]],
               "shrink": shrink, "warnings": list(report["warnings"])[:20], "clipped": report.get("clipped", [])[:20]}
