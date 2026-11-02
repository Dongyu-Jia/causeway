"""CoreView: what a core simulation hands the instrument layer (INSTRUMENTS_DESIGN §2.1), and the bridges.

Array conventions (the engine's SimOutput): per-period arrays (n, T), level paths (n, T + 1) with index 0 = the
anchor; returns decimal (simple), rates and spreads percent; freq "M" or "Q"; rows in scenario order.

    core = CoreView.from_sim(sim, db, scenario_id="baseline")            # the monthly engine
    core = CoreView.from_arrays("M", total, income, macro, labels, ...)   # scenario_paths.npz + scenario_macro.npz
    core = CoreView.from_quarterly(draws)                                 # the quarterly model (SCHEMA fields)
    q = core.quarterly()                                                  # M -> Q aggregation

Derived by the layer itself: x(a) = log(1 + total_a - income_a) (log price return), lr(a) = log(1 + total_a),
the NDX dividend yield income_nasdaq x ppy, the cash rate income_cash x ppy, the period-end T-bill (tb3_path),
and the GZ spread change (gz_path): the core's own d_gz when it has one (the quarterly model's tier-1 field), else
the bridge (defect 2): d_gz = a + b d(Baa - 10y) + c x the US market's log total return, fitted 1990..2014 on
quarters, run month by month on a monthly core (it is linear, so the months sum to the quarter's bridge exactly).

The bridge is the design's D15 default as fitted (0.019 + 1.003 d(Baa - 10y) - 0.771 r_eq per quarter), the same
map for every row of every view: a scenario with lower equity returns gets more GZ widening, and a path's d_gz never
depends on the other rows of its view. (An earlier version centred the equity term on each view's own mean per
scenario to drop the bridge's level drift on cores whose equity return is below the 1990-2014 mean; that also
removed every scenario-level equity stress from d_gz and made a row's d_gz depend on the rest of its view, so it
was withdrawn; the drift is the bridge's, about +0.8pp of GZ over 15 years on the baseline, and is disclosed.)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .. import config
from . import params as prm
from .streams import stream

PPY = {"M": 12, "Q": 4}


@dataclass(frozen=True)
class CoreView:
    freq: str                                   # "M" | "Q"
    labels: list                                # T + 1 period-end labels ('YYYY-MM'), labels[0] = the anchor
    assets: list                                # asset ids of the last axis of total / income
    total: np.ndarray                           # (n, T, A) simple total returns
    income: np.ndarray                          # (n, T, A) income per start-of-period value
    cpi: np.ndarray                             # (n, T + 1) CPI level
    y10: np.ndarray                             # (n, T + 1) 10-year yield, percent
    ff: np.ndarray                              # (n, T + 1) fed funds, percent
    spread: np.ndarray                          # (n, T + 1) Baa - 10y, pp
    d_gz: np.ndarray | None = None              # (n, T) GZ change, pp per period (a core that models it)
    tb3: np.ndarray | None = None               # (n, T) 3-month bill at the period end, percent
    d_tp10: np.ndarray | None = None            # (n, T) term-premium change (accepted; not used yet, §10)
    rv_core: np.ndarray | None = None           # (n, T) NDX realized variance from the core (vol_source "core")
    drawn: np.ndarray | None = None             # (n, T) 'YYYY-MM' historical month drawn (vol_source "drawn")
    resid_drawn: np.ndarray | None = None       # (n, T) 'YYYY-MM' the core residual's own month (pairing, D16)
    row_group: np.ndarray | None = None         # (n,) group of each row (scenario id / node key)
    row_index: np.ndarray | None = None         # (n,) path index of each row within its group
    set_id: str = "builtin"                     # stream namespace: "builtin", "custom:<id>", "continuation", ...
    key: str = ""                               # this view's key for the shared streams (scenario id, chunk)
    row_set_id: str = ""                        # namespace of the per-row draws (alpha); "" = set_id. The
    #                                             continuation passes its scenario set's, so rows keep their alpha
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- shape
    @property
    def n(self) -> int:
        return int(self.total.shape[0])

    @property
    def T(self) -> int:
        return int(self.total.shape[1])

    @property
    def ppy(self) -> int:
        return PPY[self.freq]

    def j(self, asset: str) -> int:
        return self.assets.index(asset)

    def asset_total(self, a: str) -> np.ndarray:
        return self.total[:, :, self.j(a)]

    def asset_income(self, a: str) -> np.ndarray:
        return self.income[:, :, self.j(a)]

    def x(self, a: str) -> np.ndarray:
        """Log price return of asset a per period."""
        return np.log1p(self.asset_total(a) - self.asset_income(a))

    def lr(self, a: str) -> np.ndarray:
        """Log total return of asset a per period."""
        return np.log1p(self.asset_total(a))

    def row_namespace(self) -> str:
        return self.row_set_id or self.set_id

    def groups(self) -> np.ndarray:
        return (np.asarray(self.row_group).astype(str) if self.row_group is not None
                else np.full(self.n, self.key or "core"))

    def indices(self) -> np.ndarray:
        return np.asarray(self.row_index, dtype=np.int64) if self.row_index is not None else np.arange(self.n)

    def dspread(self) -> np.ndarray:
        return np.diff(self.spread, axis=1)

    def tb3_path(self) -> np.ndarray:
        """3-month bill at each period end (percent): the core's own, else the cash asset's rate (the engine's cash
        is max(ff - basis, 0) with basis = anchor ff - anchor tb3)."""
        if self.tb3 is not None:
            return np.asarray(self.tb3, float)
        return self.asset_income("cash") * self.ppy * 100.0

    def rf(self) -> np.ndarray:
        """Cash rate per period, annual decimal (>= 0)."""
        return np.maximum(self.asset_income("cash") * self.ppy, 0.0)

    def div_yield(self, a: str = "nasdaq") -> np.ndarray:
        return self.asset_income(a) * self.ppy

    def gz_path(self, model: str | None = None, noise: bool | None = None) -> tuple[np.ndarray, str]:
        """(d_gz per period in pp, where it came from: "core" or "bridge:<model>").

        The bridge is the fitted formula as it stands (bridges.json, the design's D15 default), applied to every
        row alike: the market and spread terms move d_gz path by path and scenario by scenario."""
        if self.d_gz is not None and model is None:
            return np.asarray(self.d_gz, float), "core"
        B = prm.bridges()["d_gz"]
        name = model or B["default"]
        mdl = B["models"][name]
        q_per = 4.0 / self.ppy                               # quarters per period
        X = {"d_baa10": self.dspread(), "lr_us_equity": self.lr("us_equity"), "lr_financials": self.lr("financials")}
        g = mdl["coef"][0] * q_per + sum(b * X[xk] for b, xk in zip(mdl["coef"][1:], mdl["x"]))
        if B["noise"] if noise is None else noise:
            g = g + mdl["sd_e_train"] * np.sqrt(q_per) * stream(self.set_id, self.key, "gz").standard_normal(g.shape)
        return g, f"bridge:{name}"

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_sim(cls, sim, db=None, scenario_id: str = "", set_id: str = "builtin", row_offset: int = 0,
                 labels: list | None = None, resid_idx: np.ndarray | None = None) -> "CoreView":
        """A monthly engine run (engine.simulate.SimOutput); db (MonthDB) gives the drawn months."""
        from ..engine.summary import month_labels
        total = np.asarray(sim.total, dtype=np.float64)
        n, T, _ = total.shape
        m = sim.macro
        drawn = resid = None
        if db is not None:
            months = np.asarray(db.dates.strftime("%Y-%m")).astype("U7")
            if getattr(sim, "idx", None) is not None:
                drawn = months[np.asarray(sim.idx)].reshape(n, T)
            if resid_idx is not None:
                resid = months[np.asarray(resid_idx)].reshape(n, T)
        return cls(freq="M", labels=list(labels or month_labels(sim.start, T)), assets=list(sim.assets), total=total,
                   income=np.asarray(sim.income, dtype=np.float64), cpi=np.asarray(m["cpi"], float),
                   y10=np.asarray(m["y10"], float), ff=np.asarray(m["ff"], float), spread=np.asarray(m["spread"], float),
                   drawn=drawn, resid_drawn=resid, row_group=np.full(n, scenario_id or "core"),
                   row_index=np.arange(n) + int(row_offset), set_id=set_id, key=scenario_id or "core")

    @classmethod
    def from_arrays(cls, freq: str, total, income, macro: dict, labels, assets=None, scenario_id: str = "",
                    set_id: str = "builtin", row_offset: int = 0, drawn=None, resid_drawn=None,
                    **extra) -> "CoreView":
        """Arrays as the npz files store them: total / income (n, T, A), macro {cpi, y10, ff, spread} (n, T + 1)."""
        total = np.asarray(total, dtype=np.float64)
        n = total.shape[0]
        return cls(freq=freq, labels=list(labels), assets=list(assets or config.ASSET_IDS), total=total,
                   income=np.asarray(income, dtype=np.float64), cpi=np.asarray(macro["cpi"], float),
                   y10=np.asarray(macro["y10"], float), ff=np.asarray(macro["ff"], float),
                   spread=np.asarray(macro["spread"], float), drawn=drawn, resid_drawn=resid_drawn,
                   row_group=np.full(n, scenario_id or "core"), row_index=np.arange(n) + int(row_offset),
                   set_id=set_id, key=scenario_id or "core", **extra)

    @classmethod
    def from_quarterly(cls, draws: dict, labels=None, scenario_id: str = "", set_id: str = "quarterly",
                       row_offset: int = 0, income_yield: dict | None = None) -> "CoreView":
        """The quarterly model's draws in its SCHEMA fields (n paths, Q quarters):
          y_assets (n, Q, 13) quarterly LOG total returns in config.ASSET_IDS order (or y_<asset> (n, Q) each)
          inc_assets (n, Q, 13) income returns (optional: else the engine's convention, `income_yield` = the
                     anchor's annual yields (ctx.anchor.income_yield) / 4, the 10-year's coupon and the cash rate)
          s_ff, s_y10, s_spread (n,) or scalars: the start levels; d_ff, d_y10, d_spread (n, Q) quarterly changes
          d_gz (n, Q) the GZ change (tier 1), d_tp10 (n, Q) optional, cpi (n, Q + 1) or infl_q (n, Q) log changes
          tb3 (n, Q) optional (else from the cash asset), rv_ndx (n, Q) optional (vol_source "core")."""
        ids = list(config.ASSET_IDS)
        if "y_assets" in draws:
            y = np.asarray(draws["y_assets"], float)
        else:
            y = np.stack([np.asarray(draws[f"y_{a}"], float) for a in ids], axis=2)
        n, Q, _ = y.shape
        total = np.expm1(y)

        def level(start, d):
            s = np.broadcast_to(np.asarray(start, float), (n,)).reshape(n, 1)
            return np.concatenate([s, s + np.cumsum(np.asarray(d, float), axis=1)], axis=1)
        y10, ff = level(draws["s_y10"], draws["d_y10"]), level(draws["s_ff"], draws["d_ff"])
        if "inc_assets" in draws:
            income = np.asarray(draws["inc_assets"], float)
        else:
            income = np.zeros_like(total)
            for k, a in enumerate(ids):
                income[:, :, k] = float((income_yield or {}).get(a, 0.0)) / 4.0
            income[:, :, ids.index("treasury_10y")] = y10[:, :-1] / 400.0
            income[:, :, ids.index("cash")] = np.maximum(ff[:, :-1], 0.0) / 400.0

        if "cpi" in draws:
            cpi = np.asarray(draws["cpi"], float)
        else:
            cpi = np.concatenate([np.ones((n, 1)), np.exp(np.cumsum(np.asarray(draws["infl_q"], float), axis=1))],
                                 axis=1)
        return cls(freq="Q", labels=list(labels or [str(k) for k in range(Q + 1)]), assets=ids, total=total,
                   income=income, cpi=cpi, y10=y10, ff=ff, spread=level(draws["s_spread"], draws["d_spread"]),
                   d_gz=None if draws.get("d_gz") is None else np.asarray(draws["d_gz"], float),
                   d_tp10=draws.get("d_tp10"), tb3=draws.get("tb3"), rv_core=draws.get("rv_ndx"),
                   row_group=np.full(n, scenario_id or "core"), row_index=np.arange(n) + int(row_offset),
                   set_id=set_id, key=scenario_id or "core")

    # ---------------------------------------------------------------- aggregation
    def quarterly(self) -> "CoreView":
        """M -> Q: total = product of (1 + monthly total) - 1, income = the sum of the three months' income,
        levels at quarter ends, d_gz / rv_core summed, tb3 at the quarter's last month (whole quarters only)."""
        if self.freq == "Q":
            return self
        Q = self.T // 3
        n = self.n
        lt = np.log1p(self.total[:, :Q * 3]).reshape(n, Q, 3, -1).sum(axis=2)
        inc = self.income[:, :Q * 3].reshape(n, Q, 3, -1).sum(axis=2)
        lv = lambda a: None if a is None else np.asarray(a)[:, 0:Q * 3 + 1:3]           # noqa: E731
        sm = lambda a: None if a is None else np.asarray(a)[:, :Q * 3].reshape(n, Q, 3).sum(axis=2)   # noqa: E731
        return replace(self, freq="Q", labels=list(self.labels[0:Q * 3 + 1:3]), total=np.expm1(lt), income=inc,
                       cpi=lv(self.cpi), y10=lv(self.y10), ff=lv(self.ff), spread=lv(self.spread),
                       d_gz=sm(self.d_gz), tb3=None if self.tb3 is None else np.asarray(self.tb3)[:, 2:Q * 3:3],
                       d_tp10=sm(self.d_tp10), rv_core=sm(self.rv_core), drawn=None,
                       resid_drawn=None if self.resid_drawn is None else np.asarray(self.resid_drawn)[:, 0:Q * 3:3])

    def rows(self, a: int, b: int) -> "CoreView":
        cut = lambda v: None if v is None else np.asarray(v)[a:b]          # noqa: E731
        return replace(self, total=self.total[a:b], income=self.income[a:b], cpi=self.cpi[a:b], y10=self.y10[a:b],
                       ff=self.ff[a:b], spread=self.spread[a:b], d_gz=cut(self.d_gz), tb3=cut(self.tb3),
                       d_tp10=cut(self.d_tp10), rv_core=cut(self.rv_core), drawn=cut(self.drawn),
                       resid_drawn=cut(self.resid_drawn), row_group=cut(self.row_group), row_index=cut(self.row_index))


def macro_arrays(sims: list, ids: list[str], db=None) -> dict:
    """What library.run can write as results/scenario_macro.npz in its existing loop (no new simulation), in the
    order of scenario_paths.npz (ids): y10, ff, spread (S, N, T + 1) float32, the drawn month positions (S, N, T)
    int16 (db rows; db_dates maps them) and the fingerprint of the stored paths (instruments.build reads it):

        np.savez_compressed(out_dir / "scenario_macro.npz", **macro_arrays(sims, ids, ctx.db))"""
    from .build import paths_fingerprint
    out = {k: np.stack([np.asarray(s.macro[k], np.float32) for s in sims]) for k in ("y10", "ff", "spread")}
    out["drawn_idx"] = np.stack([np.asarray(s.idx, np.int16) for s in sims])
    out["ids"] = np.array([str(i) for i in ids])
    out["paths_fingerprint"] = np.array(paths_fingerprint(
        {"ids": out["ids"], "total": np.stack([np.asarray(s.total, np.float32) for s in sims])}))
    if db is not None:
        out["db_dates"] = np.asarray(db.dates.strftime("%Y-%m")).astype("U7")
    return out
