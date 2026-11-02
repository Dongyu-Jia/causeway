"""The quarterly model as the agent tree sees it (AGENT_DESIGN §6.1, §16): a ModelAdapter interface and its stub.

The tree never imports the asset model directly. Everything it needs from "the history-estimated quarterly model"
goes through one object with the contract below. :func:`load_adapter` picks it (phase P14):

    MSL_AGENT_MODEL set        that adapter: "stub" or "<module>:<factory>" (validated once per adapter object)
    else, the quarterly data   the history-estimated quarterly stack, scenariolab.qmodel.adapter:make_adapter
    exist (data/processed/       (试运行 until the switch gate T-S passes), when its panels exist and it fits
    panel_quarterly.csv and
    panel_monthly.csv)
    else                       :class:`GaussianStub`, labelled 桩模型（不是资产模型）, with the reason in
                               :func:`model_info` (CI and fresh clones have no data/)

The test suite pins the stub (tests/conftest.py sets MSL_AGENT_MODEL=stub); tests that need the real model opt in
and skip without data. Every run that the stub prices carries 桩模型（不是资产模型）; no run is exported yet
(tree.EXPORT_DISABLED_WHY).

What a real adapter must provide (the engine checks it with :func:`validate_adapter`)
=====================================================================================

Attributes

    version: str                 enters every cache key and run.json.versions.model_adapter
    stub: bool                   True only for stand-ins (labels the run, disables the hand-off)
    capabilities: frozenset      subset of {"structural", "rarity", "impute"}; without "structural" the engine uses
                                 the two-step protocol of §6.3 (lag-protected E0 pinned at the masked draw A0)
    x_fields: tuple[str, ...]    schema x ids the adapter completes (every one may be pinned or left free)
    assets: tuple[str, ...]      the 13 asset ids in config.ASSET_IDS order

Methods (all vectorised over N particles; seeded by the rng passed in; never reads a clock or a global RNG)

    field_info() -> {field id: {zh, unit, endo, cls, synth, axis, channel_class, lag_discipline, key,
                                hist: {p01, p05, p50, p95, p99, min, max, sd_q, sd_q_1985_2019}}}
        Metadata for every x field (and the s fields it keeps). `hist.sd_q_1985_2019` is the distance and
        support scale; the engine falls back to `sd_q`.
    root_state(n) -> State
        The start-of-quarter state at the origin (start of 2026Q4) for n particles (copies or an imputed draw).
    complete(state, given, rng, *, structural=None) -> Draws
        One joint draw per particle of every x field and every asset return for the coming quarter, conditional on
        the start-of-quarter state and on `given` (per-particle point constraints; NaN = free, integrated under the
        model's own conditional distribution: this is what makes the prior-only trees reproduce the model). Pinned
        values must come back unchanged in Draws.x, except where field_info documents a bound (the quarterly
        stack: a policy rate pinned below its floor `level_floor` 0.1 is delivered at the floor, unemployment
        below 2.05 and spreads below 0.11 at the bound; Draws.notes says how many particles). How a pin is read is
        the adapter's `field_info()[f]["given_mode"]`: "do" (an intervention through the field's structural shock:
        same-quarter E0 macro unmoved, markets respond on impact; the quarterly stack's d_ff and d_bs), "do_last"
        (set after the quarter's draw, nothing responds in the quarter; d_tariff, d_lf) or "reduced" (an
        observation: the rest is drawn from the conditional distribution). `structural` (capability "structural"
        only): {field: values} further interventions (do for fields with a policy shock: d_deficit, d_transfers,
        d_perstax; do_last for other satellites). The same rng consumed in the same order gives the same draw
        (common random numbers): a particle nothing pins gives back its draw; a pin, even at the particle's own
        value, can re-shape it (the quarterly stack's ±4 sd saturation and its dollar hold), so the engine pins only
        what an action wrote.
    update_state(state, draws) -> State
        The next start-of-quarter state: the state roll, the valuation identities, the previous-quarter x values
        (as "<x id>_l1", plus "d_unrate_l2"), last quarter's asset LOG TOTAL returns ("<asset>_l1") and
        "nasdaq_4q" (the sum of the last four quarters' Nasdaq log total returns).
    rarity(state, given) -> np.ndarray            (optional, capability "rarity"; SCHEMA rule 6)

State is a dict of numpy arrays whose first axis is the particle (shape (N,) or (N, k)). The engine only takes,
concatenates and copies along axis 0 (resampling, pooling, reassignment) and reads these keys, as particle
medians, for priors, gates, packets and node distances (REQUIRED_STATE):

    s_ff s_unrate s_core_yoy s_spread s_term s_infexp1 s_rstar s_sahm s_cape
    d_ff_l1 d_unrate_l1 d_unrate_l2 d_spread_l1 d_itinv_l1 d_gz_l1 us_equity_l1 nasdaq_4q real_oil_d1 real_oil_d2

Keys that start with "_" are private to the adapter (windows, caches of its own); the engine keeps one key of its own
in the state, "ff_target" (the policy-rate target at the quarter's end, see below), which adapters carry along like
any other key. Conventions of the quarterly stack's root (2026Q4 start): s_unrate (4.2), s_spread, s_gz and s_y10
are 2026Q3 quarter-end values; s_ff is the schema's "上季末月的联邦基金利率（月均）", the monthly average of the
quarter's last month (September 2026: 3.75, half of the mid-September hike), and d_ff the change between two such
averages, so a meeting in a quarter's last month splits across two quarters (tree.Engine.fed_units maps the Fed's
target moves into these units); CPI y/y, the Sahm gap, Michigan expectations and CAPE are the published
start-of-quarter values; s_rstar is held at HLW's 1.009.

Draws: x {field id: (N,)} for every x field (keys starting with "_" are the adapter's private columns, possibly 2-D,
which update_state reads; the engine carries them along axis 0 and never reads them); y {asset id: (N,)} quarterly
LOG total returns and y["mult"] (the US-equity valuation leg: Δlog CAPE multiple, price / (CPI × 10-year average
real EPS); R-A2); inc {asset id: (N,)} quarterly income (simple return); cpi_q (N,) the quarter's headline CPI log
change; drawn (N,) optional labels of the historical quarter each particle's asset residual was drawn from (R-A4);
notes: optional [{field?, code?, particles, text}] about saturation, floors, bounds and tails. Units are the
schema's (pp, %年化, log% …); x_oil_cause is a float: 0 = demand, 0.5 = mixed, 1 = supply (R-A1; the qdata
convention). take / concat carry drawn and notes.

The quarterly stack's adapter (scenariolab/qmodel/adapter.py, QAdapter) maps onto this one-to-one: rollout.Stepper
.draw is complete(), Stepper.roll is update_state(); the do-operator shocks u_ff, u_bs, u_def, u_tr, u_ptax are
the "structural" capability. Load any other adapter with MSL_AGENT_MODEL=<module>:<factory> (the factory returns
the adapter object).

What an intervention reaches is the model's business, and the engine measures it rather than assuming it
(tree.transmission): on the quarterly stack only the ten core variables (CPI, services CPI, IP, unemployment, the
shadow rate, real oil, the real 10-year yield, the Baa and GZ spreads, the dollar) carry lags into each other; the
42 satellites (GDP, consumption, investment, the fiscal fields, d_bs, d_tariff, d_lf, d_itinv, …) follow the core but
never feed back, and the satellite shocks u_bs, u_tr, u_def, u_ptax move only later-ordered financial and survey
satellites on impact. So a do on a satellite lever moves no macro field and no asset at any horizon; the engine
then treats that lever's channel as absent (the synthesizer's gate G1), keeps AGENT_DESIGN §6.3's response table
(transfers → goods consumption) and writes QE's intervention paths (SIM_DESIGN §5.3, template G08).

Robustness (P14): an adapter's long-horizon tails can leave history far behind (until the quarterly round's P4b,
26% of 80-quarter step chains reach GZ > 15 or unemployment > 20%). :func:`tail_mask` flags particles whose draw is
not finite or whose quarter ends beyond TAIL_BOUNDS: the quarterly stack's documented bounds (rollout.tail_flags:
GZ, unemployment, US equity) and, from the P14 review, every asset's quarterly log return, the policy rate, the
10-year yield (the Treasury pricing formula is not valid below 0) and core and headline CPI y/y outside history; the
engine drops and resamples them within the node and records the share per quarter and cumulated along each path
(tree.Engine.sanitize).

Identity (P14 review): :func:`identity` is the adapter's version with the quarterly stack's file-time digest replaced
by a hash of the data's bytes; it keys the tree's root, every LLM cache entry and the version checks of a resume, so
a byte-identical copy or rebuild of data/ keeps the tree and its cache.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

STUB_PATH = Path(__file__).resolve().parent / "stub_model.json"
REQUIRED_STATE = ("s_ff", "s_unrate", "s_core_yoy", "s_spread", "s_term", "s_infexp1", "s_rstar", "s_sahm", "s_cape",
                  "d_ff_l1", "d_unrate_l1", "d_unrate_l2", "d_spread_l1", "d_itinv_l1", "d_gz_l1", "us_equity_l1",
                  "nasdaq_4q", "real_oil_d1", "real_oil_d2")
CAPABILITIES = frozenset({"structural", "rarity", "impute"})
STUB_LABEL = "桩模型（不是资产模型）"
QUARTERLY_SPEC = "scenariolab.qmodel.adapter:make_adapter"
QUARTERLY_LABEL = "季度模型（按历史估计，试运行）"
SATURATE_SD = 4.0          # conditioning saturates pinned inputs at ±4 training sd (qmodel README, Saturation)
# Far outside history: a particle whose quarter ends beyond these, or whose draw is not finite, is a tail particle
# (dropped and resampled in the node). The first three are the quarterly stack's rollout.tail_flags (GZ's quarterly
# maximum is about 8, unemployment's 13); the rest come from the P14 review of a B 2 × 80 build, where 11.8% of the
# particles at 2046Q3 had a 10-year yield ≤ 0 (the Treasury price formula keeps a negative coupon there: −69% a
# quarter), 8.6% a policy rate above 12% (up to 22.5%) and core CPI y/y up to 21.5%:
#   returns        |quarterly log total return| of every asset (crude oil 1.5: −1.1 in 2020Q1 happened)
#   levels         quarter-end levels (min, max): the policy rate above 20 (history's maximum 19.1, 1981), the 10-year
#                  yield below 0 (history 0.52), core CPI y/y outside −2…15 (history 0.6–13.6) and headline CPI y/y
#                  outside −4…17 (history −2.0–14.8); y/y levels are read on the rolled state
TAIL_BOUNDS = {"s_gz": 15.0, "s_unrate": 20.0, "us_equity": 1.0, "returns": 1.0, "returns_asset": {"crude_oil": 1.5},
               "levels": {"s_ff": (None, 20.0), "s_y10": (0.0, None), "s_core_yoy": (-2.0, 15.0),
                          "s_cpi_yoy": (-4.0, 17.0)}}
TAIL_RULE = ("季度模型的长期尾部：季末 GZ 信用利差 > 15、失业率 > 20%、政策利率 > 20%、10 年期收益率 < 0、核心 CPI 同比"
             "不在 −2%…15%、CPI 同比不在 −4%…17%、任一资产单季对数回报超过 ±100%（原油 ±150%）或出现非有限数的粒子"
             "在节点内丢弃，并从同一动作簇的其余粒子中重抽补足（比例记在节点摘要与日志里）")
FLOAT32_MAX = float(np.finfo(np.float32).max)

State = dict


# =====================================================================================================================
# Containers
# =====================================================================================================================

@dataclass
class Given:
    """Per-particle point constraints: values[i, j] pins field ids[j] for particle i; NaN leaves it free."""
    ids: tuple
    values: np.ndarray

    @classmethod
    def empty(cls, n: int) -> Given:
        return cls((), np.zeros((int(n), 0)))

    @classmethod
    def from_dict(cls, pins: Mapping[str, np.ndarray], n: int) -> Given:
        ids = tuple(sorted(pins))
        vals = np.full((int(n), len(ids)), np.nan)
        for j, k in enumerate(ids):
            vals[:, j] = np.broadcast_to(np.asarray(pins[k], float), (int(n),))
        return cls(ids, vals)

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    def to_dict(self) -> dict[str, np.ndarray]:
        return {k: self.values[:, j].copy() for j, k in enumerate(self.ids)}

    def take(self, idx) -> Given:
        return Given(self.ids, self.values[np.asarray(idx, int)].copy())


@dataclass
class Draws:
    """One joint draw per particle of the quarter: x fields, asset log returns (+ "mult"), income, CPI; optionally
    the residual quarter each particle drew (`drawn`, R-A4) and the adapter's notes (saturation, floors, tails)."""
    x: dict
    y: dict
    inc: dict = field(default_factory=dict)
    cpi_q: np.ndarray | None = None
    drawn: np.ndarray | None = None
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        for d in (self.x, self.y):
            for v in d.values():
                return int(len(v))
        return 0

    def take(self, idx) -> Draws:
        idx = np.asarray(idx, int)
        return Draws({k: v[idx].copy() for k, v in self.x.items()}, {k: v[idx].copy() for k, v in self.y.items()},
                     {k: v[idx].copy() for k, v in self.inc.items()},
                     None if self.cpi_q is None else self.cpi_q[idx].copy(),
                     None if self.drawn is None else np.asarray(self.drawn)[idx].copy(),
                     [dict(nt) for nt in (self.notes or [])])

    @staticmethod
    def concat(parts: list[Draws]) -> Draws:
        p0 = parts[0]
        cat = lambda key: {k: np.concatenate([getattr(p, key)[k] for p in parts]) for k in getattr(p0, key)}  # noqa: E731
        cpi = None if p0.cpi_q is None else np.concatenate([p.cpi_q for p in parts])
        drawn = None
        if any(getattr(p, "drawn", None) is not None for p in parts):
            drawn = np.concatenate([np.asarray(p.drawn, dtype=object) if getattr(p, "drawn", None) is not None
                                    else np.full(p.n, "", dtype=object) for p in parts])
        notes = [dict(nt) for p in parts for nt in (getattr(p, "notes", None) or [])]
        return Draws(cat("x"), cat("y"), cat("inc"), cpi, drawn, notes)


def as_draws(d) -> Draws:
    """Any adapter's draw as this module's Draws (drawn and notes kept when the object has them)."""
    if isinstance(d, Draws):
        return d
    return Draws(dict(d.x), dict(d.y), dict(getattr(d, "inc", None) or {}), getattr(d, "cpi_q", None),
                 getattr(d, "drawn", None), list(getattr(d, "notes", None) or []))


def take_state(state: Mapping[str, np.ndarray], idx) -> dict:
    idx = np.asarray(idx, int)
    return {k: np.asarray(v)[idx].copy() for k, v in state.items()}


def concat_states(parts: list[Mapping[str, np.ndarray]]) -> dict:
    return {k: np.concatenate([np.asarray(p[k]) for p in parts], axis=0) for k in parts[0]}


def state_nbytes(state: Mapping[str, np.ndarray]) -> int:
    return int(sum(np.asarray(v).nbytes for v in state.values()))


def _rows_finite(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind not in "fc":
        return np.ones(a.shape[0] if a.ndim else 1, bool)
    ok = np.isfinite(a)
    return ok.reshape(a.shape[0], -1).all(axis=1) if a.ndim > 1 else ok


LEVEL_CHANGE = {"s_ff": "d_ff", "s_y10": "d_y10", "s_gz": "d_gz", "s_unrate": "d_unrate"}


def tail_reasons(state: Mapping[str, np.ndarray], draws, bounds: Mapping | None = None,
                 end_state: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """{reason: (N,) bool} of :func:`tail_mask`: "nonfinite", "s_gz", "s_unrate", "returns" (any asset beyond its
    bound) and one entry per level bound. Quarter-end levels are read on `end_state` (the rolled state) when given,
    else as start-of-quarter level + the quarter's change where the draw has it (y/y levels need end_state)."""
    b = dict(TAIL_BOUNDS if bounds is None else bounds)
    n = len(np.asarray(state["s_ff"]))
    out: dict = {}
    bad = np.zeros(n, bool)
    for part in (draws.x, draws.y, getattr(draws, "inc", None) or {}):
        for k, v in part.items():
            if str(k).startswith("_"):
                continue
            bad |= ~_rows_finite(v)
    if getattr(draws, "cpi_q", None) is not None:
        bad |= ~_rows_finite(draws.cpi_q)
    if end_state is not None:
        bad |= state_tail_mask(end_state)
    out["nonfinite"] = bad

    def level(key):
        if end_state is not None and key in end_state and np.ndim(end_state[key]) == 1:
            return np.asarray(end_state[key], float)
        ch = LEVEL_CHANGE.get(key)
        if key in state and ch is not None and ch in draws.x:
            return np.asarray(state[key], float) + np.asarray(draws.x[ch], float)
        return None
    with np.errstate(invalid="ignore"):
        for key in ("s_gz", "s_unrate"):
            v = level(key) if key in b else None
            if v is not None:
                out[key] = ~(v <= float(b[key]))
        ret = np.zeros(n, bool)
        if "us_equity" in b and "us_equity" in draws.y:
            ret |= ~(np.abs(np.asarray(draws.y["us_equity"], float)) <= float(b["us_equity"]))
        if b.get("returns") is not None:
            per = dict(b.get("returns_asset") or {})
            for a, v in draws.y.items():
                if a == "mult" or str(a).startswith("_"):
                    continue
                ret |= ~(np.abs(np.asarray(v, float)) <= float(per.get(a, b["returns"])))
        out["returns"] = ret
        for key, (lo, hi) in (b.get("levels") or {}).items():
            v = level(key)
            if v is None:
                continue
            m = np.zeros(n, bool)
            if lo is not None:
                m |= ~(v >= float(lo))
            if hi is not None:
                m |= ~(v <= float(hi))
            out[key] = m
    return out


def tail_mask(state: Mapping[str, np.ndarray], draws, bounds: Mapping | None = None,
              end_state: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    """(N,) True for a tail particle: any public x, y, income or CPI value of its draw (or of the rolled state
    `end_state`) not finite, or the quarter ending beyond `bounds` (TAIL_BOUNDS: GZ and unemployment levels, every
    asset's |quarterly log return|, the policy rate, the 10-year yield and core / headline CPI y/y). Private columns
    ("_" keys) may hold NaN by design and are not read."""
    n = len(np.asarray(state["s_ff"]))
    bad = np.zeros(n, bool)
    for v in tail_reasons(state, draws, bounds, end_state).values():
        bad |= v
    return bad


def state_tail_mask(state: Mapping[str, np.ndarray]) -> np.ndarray:
    """(N,) True where a public state value is not finite (a cloud that must never be saved or expanded as is)."""
    n = len(np.asarray(state["s_ff"]))
    bad = np.zeros(n, bool)
    for k, v in state.items():
        if not str(k).startswith("_"):
            bad |= ~_rows_finite(v)
    return bad


def to_float32(v: np.ndarray) -> np.ndarray:
    """A float64 array as float32 without overflow warnings: finite values beyond the float32 range are clipped to
    it (only tail particles can reach them, and those are resampled before a cloud is saved); NaN and ±inf stay."""
    a = np.asarray(v)
    if a.dtype != np.float64:
        return a
    if a.size and np.nanmax(np.abs(np.where(np.isfinite(a), a, 0.0))) > FLOAT32_MAX:
        a = np.where(np.isfinite(a), np.clip(a, -FLOAT32_MAX, FLOAT32_MAX), a)
    return a.astype(np.float32)


@runtime_checkable
class ModelAdapter(Protocol):
    version: str
    stub: bool
    capabilities: frozenset
    x_fields: tuple
    assets: tuple

    def field_info(self) -> dict: ...

    def root_state(self, n: int) -> dict: ...

    def complete(self, state: dict, given: Given, rng: np.random.Generator, *, structural: dict | None = None) -> Draws:
        ...

    def update_state(self, state: dict, draws: Draws) -> dict: ...


# =====================================================================================================================
# The Gaussian stub (AGENT_DESIGN §16: GaussianStub)
# =====================================================================================================================

class GaussianStub:
    """x_i,t = c_i + rho_i x_i,t-1 + Σ b_ij s_j,t + e_i;  y_a,t = c_a + b_a s_cape,t + e_a;  (e_x, e_y) ~ N(0, Σ).

    Fitted on 1985Q1–2019Q4 (stub_model.json; agent_impl/engine/build_stub.py). Conditional completion is exact
    Gaussian conditioning on the pinned subset, with each pinned input's deviation saturated at ±4 sd: association,
    not cause (CONSEQUENCES §4.0), which is why the engine pins the lag-protected E0 set from A0 under the two-step
    protocol. y["mult"] is the quarter's change in log CAPE (the stub's y_mult was fitted as s_cape(t+1) − s_cape(t)),
    so it already carries R-A2's Δlog CAPE multiple. Its income is a constant quarterly yield per asset, its policy
    rate floor is 0 and it has no x_oil_cause. Not an asset model."""

    level_floor = {"d_ff": 0.0}

    stub = True
    capabilities = frozenset()

    def __init__(self, params: Mapping | None = None, path: Path | None = None):
        p = dict(params) if params is not None else json.loads((path or STUB_PATH).read_text(encoding="utf-8"))
        self.params = p
        self.version = str(p.get("version", "gaussian-stub"))
        self.x_fields = tuple(p["x_fields"])
        self.y_fields = tuple(p["y_fields"])
        self.assets = tuple(p["assets"])
        self.order = self.x_fields + self.y_fields
        self.index = {f: i for i, f in enumerate(self.order)}
        self.cov = np.asarray(p["cov"], float)
        self.eqs = p["equations"]
        self._chol = np.linalg.cholesky(self.cov + 1e-10 * np.eye(len(self.order)))
        self._sd_diag = np.sqrt(np.diag(self.cov))
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.income = {a: float(p.get("income_q", {}).get(a, 0.0)) for a in self.assets}
        self.c = np.array([self.eqs[f]["c"] for f in self.order])
        self.rho = np.array([self.eqs[f].get("rho", 0.0) for f in self.order])
        self.fb = [(self.index[f], s, float(b)) for f in self.order for s, b in self.eqs[f].get("b", {}).items()]

    # ---------------------------------------------------------------- metadata
    def field_info(self) -> dict:
        return json.loads(json.dumps(self.params.get("fields", {})))

    def sd(self, fid: str) -> float:
        h = ((self.params.get("fields") or {}).get(fid) or {}).get("hist") or {}
        v = h.get("sd_q_1985_2019") or h.get("sd_q")
        if v:
            return float(v)
        if fid in self.index:
            return float(math.sqrt(self.cov[self.index[fid], self.index[fid]]))
        return 1.0

    # ---------------------------------------------------------------- states
    def root_state(self, n: int) -> dict:
        n = int(n)
        root, lags = self.params["root"], self.params["lags"]
        s = {k: np.full(n, float(v)) for k, v in root.items()}
        for f in self.x_fields:
            s[f"{f}_l1"] = np.full(n, float(lags.get(f"{f}_l1", self.eqs[f]["mean"])))
        s["d_unrate_l2"] = np.full(n, float(lags.get("d_unrate_l2", 0.0)))
        for a in self.assets:
            s[f"{a}_l1"] = np.full(n, float(lags.get(f"{a}_l1", 0.0)))
        nas = list(lags.get("nasdaq_win") or [0.0] * 4)[-4:]
        s["_nasdaq_win"] = np.tile(np.array(nas, float), (n, 1))
        s["nasdaq_4q"] = s["_nasdaq_win"].sum(axis=1)
        core = list(lags.get("core_win") or [root["s_core_yoy"]] * 4)[-4:]
        s["_core_win"] = np.tile(np.array(core, float), (n, 1))
        un = list(lags.get("unrate_win") or [root["s_unrate"]] * 4)[-4:]
        s["_unrate_win"] = np.tile(np.array(un, float), (n, 1))
        s["s_sahm"] = s["_unrate_win"][:, -1] - s["_unrate_win"][:, :-1].min(axis=1)
        oil = list(lags.get("oil_real_win") or [root["s_oil_real"]] * 3)[-3:]
        s["_oil_win"] = np.tile(np.array(oil, float), (n, 1))
        s["real_oil_d1"] = s["s_oil_real"] - s["_oil_win"][:, -2]
        s["real_oil_d2"] = s["s_oil_real"] - s["_oil_win"][:, -3]
        return s

    def mean(self, state: Mapping[str, np.ndarray]) -> np.ndarray:
        """(N, D) conditional means given the start-of-quarter state."""
        n = len(state["s_ff"])
        mu = np.tile(self.c, (n, 1))
        for i, f in enumerate(self.x_fields):
            lag = state.get(f"{f}_l1")
            if lag is not None and self.rho[i]:
                mu[:, i] += self.rho[i] * np.asarray(lag, float)
        for i, s, b in self.fb:
            v = state.get(s)
            if v is not None:
                mu[:, i] += b * np.asarray(v, float)
        return mu

    # ---------------------------------------------------------------- conditional completion
    def _decomp(self, pinned: tuple):
        key = pinned
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        D = len(self.order)
        P = np.array(pinned, int)
        F = np.array([i for i in range(D) if i not in set(pinned)], int)
        S = self.cov
        if len(P) == 0:
            out = (P, F, None, self._chol)
        else:
            Spp = S[np.ix_(P, P)] + 1e-9 * np.eye(len(P))
            Sfp = S[np.ix_(F, P)]
            K = np.linalg.solve(Spp, Sfp.T).T                     # (|F|, |P|)
            C = S[np.ix_(F, F)] - K @ Sfp.T
            C = 0.5 * (C + C.T)
            w, V = np.linalg.eigh(C)
            L = V * np.sqrt(np.maximum(w, 0.0))
            out = (P, F, K, L)
        with self._lock:
            self._cache[key] = out
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
        return out

    def complete(self, state: Mapping[str, np.ndarray], given: Given | None, rng: np.random.Generator, *,
                 structural: dict | None = None) -> Draws:
        n = len(state["s_ff"])
        mu = self.mean(state)
        D = len(self.order)
        Z = np.full((n, D), np.nan)
        extra = {}
        if given is not None and len(given.ids):
            for j, fid in enumerate(given.ids):
                if fid in self.index:
                    Z[:, self.index[fid]] = given.values[:, j]
                else:
                    extra[fid] = given.values[:, j].copy()
        mask = ~np.isnan(Z)
        if mask.any():
            _, inv = np.unique(np.packbits(mask, axis=1), axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            groups = [np.flatnonzero(inv == g) for g in range(int(inv.max()) + 1)]
        else:
            groups = [np.arange(n)]
        eps = rng.standard_normal((n, D))
        # Matheron's rule: the unconditional draw z (the same for the same stream), then z_F + K (pin − z_P). Exact
        # Gaussian conditioning, and with common random numbers a pin at a particle's own draw returns that draw
        out = mu + eps @ self._chol.T
        for rows in groups:
            pinned = tuple(int(j) for j in np.flatnonzero(mask[rows[0]]))
            P, F, K, _ = self._decomp(pinned)
            if K is None:
                continue
            dev = Z[np.ix_(rows, P)] - mu[np.ix_(rows, P)]
            lim = SATURATE_SD * self._sd_diag[P]
            dev = np.clip(dev, -lim, lim)            # standardized inputs saturate at ±4 sd (as the qmodel does)
            gap = (mu[np.ix_(rows, P)] + dev) - out[np.ix_(rows, P)]
            out[np.ix_(rows, F)] = out[np.ix_(rows, F)] + gap @ K.T
            out[np.ix_(rows, P)] = Z[np.ix_(rows, P)]
        # the zero lower bound on free policy-rate draws (pinned values were floored by the engine)
        j = self.index.get("d_ff")
        if j is not None:
            free = ~mask[:, j]
            floor = -np.asarray(state["s_ff"], float)
            out[free, j] = np.maximum(out[free, j], floor[free])
        x = {f: out[:, self.index[f]].copy() for f in self.x_fields}
        x.update(extra)
        y = {f[2:]: out[:, self.index[f]].copy() for f in self.y_fields}
        inc = {a: np.full(n, self.income[a]) for a in self.assets}
        cpi = (0.37 * x["d_cpi_goods"] + 0.63 * x["d_cpi_serv"]) / 400.0
        return Draws(x, y, inc, cpi)

    # ---------------------------------------------------------------- the state roll
    def update_state(self, state: Mapping[str, np.ndarray], draws: Draws) -> dict:
        """The stub's roll. Valuation (R-A2): s_cape += y["mult"] is exact (mult is Δlog CAPE); s_pd += y["mult"]
        assumes dividends grow with CPI × the 10-year average real EPS (a constant payout on smoothed earnings),
        so Δlog(P/D) = Δlog CAPE; the stub has no dividend model of its own."""
        s = {k: np.array(v, float, copy=True) for k, v in state.items()}
        x, y = draws.x, draws.y
        s["s_ff"] = np.maximum(s["s_ff"] + x["d_ff"], 0.0)
        s["s_unrate"] = np.clip(s["s_unrate"] + x["d_unrate"], 2.0, 20.0)
        s["_core_win"] = np.concatenate([s["_core_win"][:, 1:], x["d_core"][:, None]], axis=1)
        s["s_core_yoy"] = s["_core_win"].mean(axis=1)
        s["s_term"] = s["s_term"] + x["d_y10"] - x["d_ff"]
        s["s_y10"] = s["s_ff"] + s["s_term"]
        s["s_spread"] = np.maximum(s["s_spread"] + x["d_spread"], 0.3)
        s["s_gz"] = np.maximum(s["s_gz"] + x["d_gz"], 0.2)
        s["s_infexp1"] = np.clip(s["s_infexp1"] + x["d_infexp1"], -2.0, 15.0)
        s["s_infexp_lr"] = np.clip(s["s_infexp_lr"] + x["d_infexp_lr"], 0.0, 10.0)
        s["s_real_y10"] = s["s_real_y10"] + x["d_y10"] - x["d_infexp_lr"]
        s["s_cape"] = s["s_cape"] + y["mult"]
        s["s_pd"] = s["s_pd"] + y["mult"]
        s["s_tariff"] = np.maximum(s["s_tariff"] + x["d_tariff"], 0.0)
        pi_q = (0.37 * x["d_cpi_goods"] + 0.63 * x["d_cpi_serv"]) / 400.0
        s["_oil_win"] = np.concatenate([s["_oil_win"][:, 1:], s["s_oil_real"][:, None]], axis=1)
        s["s_oil_real"] = s["s_oil_real"] + x["d_oil"] / 100.0 - pi_q
        s["real_oil_d1"] = s["s_oil_real"] - s["_oil_win"][:, -1]
        s["real_oil_d2"] = s["s_oil_real"] - s["_oil_win"][:, -2]
        s["_unrate_win"] = np.concatenate([s["_unrate_win"][:, 1:], s["s_unrate"][:, None]], axis=1)
        s["s_sahm"] = s["_unrate_win"][:, -1] - s["_unrate_win"][:, :-1].min(axis=1)
        s["d_unrate_l2"] = s["d_unrate_l1"].copy()
        for f in self.x_fields:
            s[f"{f}_l1"] = np.asarray(x[f], float).copy()
        for a in self.assets:
            s[f"{a}_l1"] = np.asarray(y[a], float).copy()
        s["_nasdaq_win"] = np.concatenate([s["_nasdaq_win"][:, 1:], y["nasdaq"][:, None]], axis=1)
        s["nasdaq_4q"] = s["_nasdaq_win"].sum(axis=1)
        return s


# =====================================================================================================================
# Loading and validation
# =====================================================================================================================

_STUB: GaussianStub | None = None
_STUB_LOCK = threading.Lock()


def stub() -> GaussianStub:
    global _STUB
    with _STUB_LOCK:
        if _STUB is None:
            _STUB = GaussianStub()
        return _STUB


STUB_NAMES = ("stub", "gaussian", "gaussian_stub")
_VALIDATED: dict = {}                 # id(adapter) -> version, adapters that passed validate_adapter once
_FALLBACK = {"why": "", "key": None}  # why the default fell back to the stub (and the data state it was decided for)
_LOAD_LOCK = threading.Lock()


def quarterly_data_ready() -> tuple[bool, str]:
    """(ready, why not): the quarterly stack's panels exist locally (data/ is git-ignored: CI and fresh clones
    have none) and its package imports."""
    try:
        from .. import config
        from ..qmodel import config_q as cq
    except ImportError as err:                       # pragma: no cover - the package ships with the repository
        return False, f"季度模型包不可用：{err}"
    q = Path(cq.PANEL_Q_PATH)
    m = Path(config.PROCESSED_DIR) / "panel_monthly.csv"
    missing = [p.name for p in (q, m) if not p.exists()]
    if missing:
        return False, "缺少季度模型的数据：" + "、".join(missing) + "（data/processed/，运行 python -m scenariolab.build）"
    return True, ""


def _data_key() -> tuple:
    try:
        from .. import config
        from ..qmodel import config_q as cq
        return tuple((str(p), p.stat().st_mtime_ns) for p in (Path(cq.PANEL_Q_PATH),
                                                               Path(config.PROCESSED_DIR) / "panel_monthly.csv")
                     if p.exists())
    except (ImportError, OSError):                   # pragma: no cover
        return ()


def _from_spec(name: str):
    if ":" not in name:
        raise ValueError(f"MSL_AGENT_MODEL 必须是 stub 或 <模块>:<工厂函数>，收到「{name}」")
    mod, fn = name.split(":", 1)
    adapter = getattr(importlib.import_module(mod), fn)()
    ver = getattr(adapter, "version", None)
    if _VALIDATED.get(id(adapter)) != ver:
        problems = validate_adapter(adapter)
        if problems:
            raise ValueError("季度模型适配器不符合接口：" + "；".join(problems))
        _VALIDATED[id(adapter)] = ver
    return adapter


def default_adapter():
    """No MSL_AGENT_MODEL: the quarterly stack when its data exist and it fits, else the stub (the reason is kept
    for model_info). A failure is remembered for the data state it happened in, so a request does not refit."""
    key = _data_key()
    with _LOAD_LOCK:
        if _FALLBACK["key"] is not None and _FALLBACK["key"] == key:
            return stub()
    ok, why = quarterly_data_ready()
    if not ok:
        with _LOAD_LOCK:
            _FALLBACK.update(why=why, key=key)
        return stub()
    try:
        adapter = _from_spec(QUARTERLY_SPEC)
    except Exception as err:  # noqa: BLE001 - any failure of the optional model falls back to the labelled stub
        with _LOAD_LOCK:
            _FALLBACK.update(why=f"季度模型无法加载（{type(err).__name__}: {str(err)[:160]}）", key=key)
        return stub()
    with _LOAD_LOCK:
        _FALLBACK.update(why="", key=None)
    return adapter


def load_adapter(spec: str | None = None):
    """The adapter named by `spec` or MSL_AGENT_MODEL ("stub", or "<module>:<factory>"); without either, the
    default (:func:`default_adapter`: the quarterly stack when its data exist, else the stub). An explicit spec that
    cannot be loaded raises ValueError in Chinese (never a fallback: the user asked for that model)."""
    name = (spec or os.environ.get("MSL_AGENT_MODEL", "") or "").strip()
    if name in STUB_NAMES:
        return stub()
    if name in ("", "auto", "default"):
        return default_adapter()
    try:
        return _from_spec(name)
    except ValueError:
        raise
    except Exception as err:  # noqa: BLE001 - the explicit model failed: a Chinese message instead of a traceback
        hint = ""
        if name == QUARTERLY_SPEC:
            ok, why = quarterly_data_ready()
            hint = f"：{why}" if not ok else ""
        raise ValueError(f"MSL_AGENT_MODEL={name} 无法加载（{type(err).__name__}: {str(err)[:200]}）{hint}") from err


def explicit_stub() -> bool:
    """MSL_AGENT_MODEL names the stub: the user (or the test suite) asked for it, so it is not a fallback."""
    return os.environ.get("MSL_AGENT_MODEL", "").strip() in STUB_NAMES


def adapter_kind(adapter) -> str:
    """stub | quarterly | custom."""
    if adapter is None or getattr(adapter, "stub", True):
        return "stub"
    return "quarterly" if str(getattr(adapter, "version", "")).startswith("qadapter") else "custom"


def fit_text(adapter) -> str:
    """A plain description of the fit for readers (never the version string): 训练截至 2026Q3 · 起点 2026Q4."""
    st = getattr(adapter, "stack", None)
    te, org = getattr(st, "train_end", None), getattr(st, "origin", None)
    if te is None:
        import re
        qs = re.findall(r"(?<![0-9])(\d{4}Q[1-4])(?![0-9])", str(getattr(adapter, "version", "")))
        te = qs[0] if qs else None
    if te is None:
        return ""
    return f"训练截至 {te}" + (f" · 起点 {org}" if org else "")


def model_info(adapter=None) -> dict:
    """What priced a run, for run.json, the snapshot manifest, the CLI and the UI: {kind, stub, version, identity,
    label, fit, capabilities, trial, why_stub, fallback, structural}. fallback: the stub priced it because the
    default could not load the quarterly model (not because MSL_AGENT_MODEL=stub asked for it). The label disappears
    (stub False) with a real model; readers see label and fit, never the version."""
    adapter = adapter if adapter is not None else load_adapter()
    kind = adapter_kind(adapter)
    caps = sorted(getattr(adapter, "capabilities", ()) or ())
    label = STUB_LABEL if kind == "stub" else (QUARTERLY_LABEL if kind == "quarterly" else "外部模型适配器")
    why = ""
    fallback = False
    if kind == "stub":
        name = os.environ.get("MSL_AGENT_MODEL", "").strip()
        if name in STUB_NAMES:
            why = "MSL_AGENT_MODEL=stub（测试与演示固定用桩模型）"
        else:
            why = _FALLBACK.get("why") or ""
            fallback = name in ("", "auto", "default")
    return {"kind": kind, "stub": kind == "stub", "version": str(getattr(adapter, "version", "")),
            "identity": identity(adapter), "label": label, "fit": fit_text(adapter), "capabilities": caps,
            "structural": "structural" in caps, "trial": kind == "quarterly", "why_stub": why, "fallback": fallback}


# ---------------------------------------------------------------- identity (content, not file times)
_SHA_CACHE: dict = {}


def _file_sha(path) -> str:
    """sha256 of a file's bytes (12 hex), cached per (path, mtime, size); "missing" when it does not exist."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return "missing"
    key = (str(p), st.st_mtime_ns, st.st_size)
    hit = _SHA_CACHE.get(key)
    if hit is None:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        hit = h.hexdigest()[:12]
        _SHA_CACHE[key] = hit
    return hit


def _quarterly_content_digest() -> str | None:
    """rollout._panel_digest with the file times replaced by the files' bytes: the panel meta's input and card
    digests, the quarterly panel and the asset-layer selection file."""
    try:
        from ..qmodel import assets as AS
        from ..qmodel import config_q as cq
        from ..qmodel import panel as qpanel
        meta = qpanel.load_meta()["info"]
        stamp = (f"{meta.get('inputs_digest', '')}|{meta.get('card_digest', '')}|{_file_sha(cq.PANEL_Q_PATH)}"
                 f"|{_file_sha(AS.SELECTION_PATH)}")
    except (ImportError, OSError, KeyError, ValueError):
        return None
    return "c" + hashlib.sha256(stamp.encode("utf-8")).hexdigest()[:11]


def identity(adapter) -> str:
    """The adapter's version as a content identity. The quarterly stack's version embeds rollout._panel_digest(),
    which hashes the panel file's modification time, so a byte-identical copy or rebuild of data/ would change every
    random stream and LLM cache key; here that component is replaced by a hash of the bytes (_quarterly_content_
    digest). Other adapters: their `content_version` attribute when they have one, else their version."""
    v = str(getattr(adapter, "version", "") or "")
    cv = getattr(adapter, "content_version", None)
    if cv:
        return str(cv)
    if not v.startswith("qadapter"):
        return v
    stack_v = str(getattr(getattr(adapter, "stack", None), "version", "") or "")
    digest = stack_v.rsplit("/", 1)[-1] if "/" in stack_v else ""
    parts = v.split("/")
    if not digest or digest not in parts:
        return v
    c = _quarterly_content_digest()
    return v if c is None else "/".join(c if p == digest else p for p in parts)


def same_version(built: str | None, adapter) -> bool:
    """A folder priced by version `built` and this adapter are the same fit: equal identities (or the raw
    version, for folders recorded before identities)."""
    b = str(built or "")
    return bool(b) and b in (identity(adapter), str(getattr(adapter, "version", "")))


def short_version(v: str | None) -> str:
    """A version for messages: the stub's id, or the quarterly stack's train end and data digest."""
    s = str(v or "")
    if s.startswith("qadapter"):
        p = s.split("/")
        return "/".join(x for x in (p[2:3] + p[6:7]) if x) or s[:40]
    return s[:40]


def same_model_kind(version_a: str | None, version_b: str | None) -> bool:
    """Two adapter versions of the same family (state layouts compatible): both the stub, or both the quarterly
    stack (any fit), or exactly equal."""
    a, b = str(version_a or ""), str(version_b or "")
    if a == b:
        return True
    fam = lambda v: "quarterly" if v.startswith("qadapter") else ("stub" if v.startswith("gaussian-stub") else v)  # noqa: E731
    return fam(a) == fam(b)


def adapter_for_version(built: str | None, *, is_stub=None, what: str = "推演"):
    """The adapter that can extend or resume a run or snapshot priced by adapter version `built`: the stub for a stub
    build; the quarterly stack for a quarterly build (whatever MSL_AGENT_MODEL says, when its data exist: a refit
    of the stack keeps the state layout, so the saved clouds replay); else the default adapter of the same family.
    ValueError in Chinese when no compatible adapter can be loaded."""
    b = str(built or "")
    if is_stub or b.startswith("gaussian-stub"):
        return stub()
    if b.startswith("qadapter"):
        ok, why = quarterly_data_ready()
        if not ok:
            raise ValueError(f"这个{what}由季度模型定价，但{why}：不能即时生成或继续")
        return _from_spec(QUARTERLY_SPEC)
    ad = load_adapter()
    if b and not same_model_kind(b, ad.version):
        raise ValueError(f"这个{what}由「{b[:48]}」定价，现在加载的是{model_info(ad)['label']}：粒子云的状态不兼容，"
                         "不能即时生成或继续")
    return ad


def resume_adapter(built: str | None, *, is_stub=None, what: str = "推演", allow_change: bool = False):
    """The adapter that continues a run or snapshot build priced by version `built`: :func:`adapter_for_version`
    (whatever MSL_AGENT_MODEL or the server's default says), then the exact-fit check. A different fit of the same
    family (a refit after the data changed) would mix two fits in one tree: refused in Chinese unless allow_change
    (the caller then records the switch). ValueError when no compatible adapter loads."""
    ad = adapter_for_version(built, is_stub=is_stub, what=what)
    if built and not same_version(built, ad) and not allow_change:
        raise ValueError(f"这个{what}由 {short_version(built)} 定价，现在能加载的是 {short_version(identity(ad))}"
                         "（数据变了，模型重新拟合过）：继续会在同一棵树里混用两次拟合，已拒绝。请换一个 id 重新构建，"
                         "或明确允许换用新的拟合（命令行 --allow-model-change）")
    return ad


def field_floor(info: Mapping, fid: str, default: float = 0.0) -> float:
    """The level floor field_info documents for a change field (the quarterly stack: d_ff 0.1), else `default`."""
    v = (info.get(fid) or {}).get("level_floor")
    try:
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def validate_adapter(adapter, n: int = 64, seed: int = 0) -> list[str]:
    """Chinese problems with an adapter, empty when it satisfies the contract of this module's docstring."""
    problems: list[str] = []
    for attr in ("version", "stub", "capabilities", "x_fields", "assets"):
        if not hasattr(adapter, attr):
            problems.append(f"缺少属性 {attr}")
    for meth in ("field_info", "root_state", "complete", "update_state"):
        if not callable(getattr(adapter, meth, None)):
            problems.append(f"缺少方法 {meth}()")
    if problems:
        return problems
    unknown = set(adapter.capabilities) - CAPABILITIES
    if unknown:
        problems.append(f"未知的能力 {sorted(unknown)}")
    try:
        s0 = adapter.root_state(n)
        miss = [k for k in REQUIRED_STATE if k not in s0]
        if miss:
            problems.append("起点状态缺少：" + "、".join(miss))
        bad = [k for k, v in s0.items() if np.asarray(v).shape[:1] != (n,)]
        if bad:
            problems.append("状态数组的第一维不是粒子数：" + "、".join(bad[:5]))
        rng = np.random.default_rng(seed)
        pin_f = adapter.x_fields[0]
        g = Given.from_dict({pin_f: np.full(n, 0.123)}, n)
        d = adapter.complete(s0, g, rng)
        if not np.allclose(d.x.get(pin_f, np.nan), 0.123):
            problems.append(f"固定的字段 {pin_f} 没有原样返回")
        miss_x = [f for f in adapter.x_fields if f not in d.x]
        if miss_x:
            problems.append("补全结果缺少 x 字段：" + "、".join(miss_x[:5]))
        miss_y = [a for a in adapter.assets if a not in d.y]
        if miss_y:
            problems.append("补全结果缺少资产回报：" + "、".join(miss_y[:5]))
        d2 = adapter.complete(s0, g, np.random.default_rng(seed))
        if not all(np.array_equal(d.y[a], d2.y[a]) for a in adapter.assets if a in d.y and a in d2.y):
            problems.append("同一种子的两次补全结果不同（必须可复现）")
        s1 = adapter.update_state(s0, d)
        miss1 = [k for k in REQUIRED_STATE if k not in s1]
        if miss1:
            problems.append("滚动后的状态缺少：" + "、".join(miss1))
    except Exception as err:  # noqa: BLE001 - reported, not raised
        problems.append(f"调用失败：{type(err).__name__}: {err}")
    return problems


def medians(state: Mapping[str, np.ndarray], keys: Iterable[str] | None = None) -> dict[str, float]:
    out = {}
    for k in (keys if keys is not None else state):
        v = state.get(k)
        if v is None:
            continue
        arr = np.asarray(v, float)
        if arr.ndim != 1 or not len(arr):
            continue
        out[k] = float(np.median(arr))
    return out
