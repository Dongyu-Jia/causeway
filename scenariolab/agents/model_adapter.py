"""The quarterly model as the agent tree sees it (AGENT_DESIGN §6.1, §16): a ModelAdapter interface and its stub.

The tree never imports the asset model directly. Everything it needs from "the history-estimated quarterly model"
goes through one object with the contract below; until the real model exists, :class:`GaussianStub` stands in
(every run that uses it carries 桩模型（不是资产模型） and cannot be exported).

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
        values must come back unchanged in Draws.x. `structural` (capability "structural" only): {field: values}
        instruments whose gap to the model's own expectation is attributed to their structural shock.
    update_state(state, draws) -> State
        The next start-of-quarter state: the state roll, the valuation identities, the previous-quarter x values
        (as "<x id>_l1", plus "d_unrate_l2"), last quarter's asset returns ("<asset>_l1") and "nasdaq_4q".
    rarity(state, given) -> np.ndarray            (optional, capability "rarity"; SCHEMA rule 6)

State is a dict of numpy arrays whose first axis is the particle (shape (N,) or (N, k)). The engine only takes,
concatenates and copies along axis 0 (resampling, pooling, reassignment) and reads these keys, as particle
medians, for priors, gates, packets and node distances (REQUIRED_STATE):

    s_ff s_unrate s_core_yoy s_spread s_term s_infexp1 s_rstar s_sahm s_cape
    d_ff_l1 d_unrate_l1 d_unrate_l2 d_spread_l1 d_itinv_l1 d_gz_l1 us_equity_l1 nasdaq_4q real_oil_d1 real_oil_d2

Keys that start with "_" are private to the adapter (windows, caches of its own).

Draws: x {field id: (N,)} for every x field; y {asset id: (N,)} quarterly LOG total returns and y["mult"] (the
US-equity valuation change, Δlog P/Ê); inc {asset id: (N,)} quarterly income (simple return); cpi_q (N,) the
quarter's CPI log change. Units are the schema's (pp, %年化, log% …) as in schema.json.

The qmodel plug-in (sim_redesign/qmodel/qplugin.py) maps onto this one-to-one: its per-path state DataFrame is the
State (as columns), GStruct.sample_step(state, x_given, rng) is complete(), and the macro/asset state roll of
predict() is update_state(); the do-operator shocks u_ff, u_bs, u_def, u_tr, u_ptax are the "structural" capability.
Load a real adapter with MSL_AGENT_MODEL=<module>:<factory> (the factory returns the adapter object).
"""
from __future__ import annotations

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
SATURATE_SD = 4.0          # conditioning saturates pinned inputs at ±4 training sd (qmodel README, Saturation)

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
    """One joint draw per particle of the quarter: x fields, asset log returns (+ "mult"), income, CPI."""
    x: dict
    y: dict
    inc: dict = field(default_factory=dict)
    cpi_q: np.ndarray | None = None

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
                     None if self.cpi_q is None else self.cpi_q[idx].copy())

    @staticmethod
    def concat(parts: list[Draws]) -> Draws:
        p0 = parts[0]
        cat = lambda key: {k: np.concatenate([getattr(p, key)[k] for p in parts]) for k in getattr(p0, key)}  # noqa: E731
        cpi = None if p0.cpi_q is None else np.concatenate([p.cpi_q for p in parts])
        return Draws(cat("x"), cat("y"), cat("inc"), cpi)


def take_state(state: Mapping[str, np.ndarray], idx) -> dict:
    idx = np.asarray(idx, int)
    return {k: np.asarray(v)[idx].copy() for k, v in state.items()}


def concat_states(parts: list[Mapping[str, np.ndarray]]) -> dict:
    return {k: np.concatenate([np.asarray(p[k]) for p in parts], axis=0) for k in parts[0]}


def state_nbytes(state: Mapping[str, np.ndarray]) -> int:
    return int(sum(np.asarray(v).nbytes for v in state.values()))


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
    protocol. Not an asset model."""

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
        out = np.empty((n, D))
        mask = ~np.isnan(Z)
        if mask.any():
            _, inv = np.unique(np.packbits(mask, axis=1), axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            groups = [np.flatnonzero(inv == g) for g in range(int(inv.max()) + 1)]
        else:
            groups = [np.arange(n)]
        eps = rng.standard_normal((n, D))
        for rows in groups:
            pinned = tuple(int(j) for j in np.flatnonzero(mask[rows[0]]))
            P, F, K, L = self._decomp(pinned)
            if K is None:
                out[rows] = mu[rows] + eps[rows] @ L.T
                continue
            dev = Z[np.ix_(rows, P)] - mu[np.ix_(rows, P)]
            lim = SATURATE_SD * self._sd_diag[P]
            dev = np.clip(dev, -lim, lim)            # standardized inputs saturate at ±4 sd (as the qmodel does)
            out[np.ix_(rows, F)] = mu[np.ix_(rows, F)] + dev @ K.T + eps[np.ix_(rows, F)] @ L.T
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


def load_adapter(spec: str | None = None):
    """The adapter named by `spec` or MSL_AGENT_MODEL: "stub" (default) or "<module>:<factory>"."""
    name = (spec or os.environ.get("MSL_AGENT_MODEL", "") or "stub").strip()
    if name in ("stub", "gaussian", "gaussian_stub"):
        return stub()
    if ":" not in name:
        raise ValueError(f"MSL_AGENT_MODEL 必须是 stub 或 <模块>:<工厂函数>，收到「{name}」")
    mod, fn = name.split(":", 1)
    adapter = getattr(importlib.import_module(mod), fn)()
    problems = validate_adapter(adapter)
    if problems:
        raise ValueError("季度模型适配器不符合接口：" + "；".join(problems))
    return adapter


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
