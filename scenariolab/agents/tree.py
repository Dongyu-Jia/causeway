"""The agent tree engine (AGENT_DESIGN §6, §8, §9, §11): the quarterly cell, beam search with Heitsch–Römisch
reduction, probability pooling, the two zero-call reference trees, checkpoints and resume.

One quarter for one node (§6.2):

    [0] packets (accounts.own_block, the news digest, the signal board) and the shared state table
    [1] calendar, pinned facts, due commitments, the account gates -> menus (actors.menu) and priors (priors.prior)
    [2] A0 = model.complete(state, given = user pins only): every actor lever and event channel masked
          -> instrument priors (bins of A0, when the model is real), shift bases, nature's tail masses
    [3] actors decide (AgentLLM: separate calls at the root, one merged call below) -> pooled with the priors
          (p = c·p_LLM + (1 − c)·p_prior) -> follower rows -> nature tilts -> joint profiles of the branching
          actors -> K action clusters by forward selection in payload space
    [4] B_k = model.complete(state, given = levers pinned per particle (Fed slopes, the floor), shifts = A0 + Δ,
          the oil / GPR event channels, lag-protected E0 from A0 under the two-step protocol)
    [5] the synthesizer: one call for the clusters that pass the gate; the rest get the default partition
    [6] children: partitions regroup B_k's particles; mechanisms re-complete B_k + Δ; nature events branch
    [7] update_state per particle, accounts.step per particle (books move with their particles), identities
    [8] memory line, news digest, realised signals and credibility, summaries
    [9] level reduction: recombination -> forward selection to B (Kantorovich) -> mass and particles reassigned

Everything except the LLM's text is a deterministic function of (settings, the adapter's version, the catalogue,
the cache): seeds are sha256(run seed, node lineage key, purpose), nodes are processed in key order, and calls are
consumed in key order, so concurrency never changes the tree. A rerun on the cache makes no new call; a resume
continues exactly from the last level checkpoint.

The three trees of every run (§9.4): 智能体树 (tree_kind "agent", the run's settings), 历史基率树 ("base_rate":
c = 0, shift levers with their decision-record priors, the synthesizer's default partition) and 模型树 ("model":
c = 0, instruments as bins of A0, shift levers at the status quo, nature at its tail masses; its leaf mixture is the
model's own distribution).
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import dataclasses
import gzip
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config
from . import accounts as AC
from . import actors as A
from . import model_adapter as MA
from . import priors as P
from . import synthesizer as SY

ENGINE_VERSION = "agent-tree-2 (2026-11-02; per-particle books, discrete split, follower marginals)"
LEDGER_NAME = "calls_paid.tsv"              # the run's paid-calls ledger (cache key, level), append-only
AGENTS_DIR = config.DATA_DIR / "agents"
RUNS_DIR = AGENTS_DIR / "runs"
SNAPSHOTS_DIR = AGENTS_DIR / "snapshots"
CACHE_DIR = AGENTS_DIR / "cache"
SETS_DIR = AGENTS_DIR / "sets"
FAILURE_LOG = AGENTS_DIR / "cli_failures.jsonl"
ROOT_QUARTER = "2026Q3"                     # the quarter that ended at the root (data to 2026-09)
FIRST_QUARTER = "2026Q4"
KEY_STATE = (("s_ff", 2.82), ("s_unrate", 1.51), ("s_core_yoy", 1.08), ("s_spread", 0.59), ("s_term", 1.27),
             ("s_cape", 0.32))             # node distances: start-of-quarter states scaled by their sd (1973–2026)
CUM_SCALE = 0.08                            # cumulative US-equity log return in node distances (one quarter's sd)
BAND_FIELDS = ("d_ff", "d_unrate", "d_rgdp", "d_core", "d_cpi_goods", "d_gz", "d_spread", "d_y10", "d_oil",
               "d_metals_oil", "d_tariff", "d_itinv", "d_capex", "d_cons_serv", "d_sent", "d_infexp1", "d_gpr",
               "d_usd", "d_transfers", "d_deficit")
STATE_ROWS = (("s_ff", "政策利率（联邦基金）", "%"), ("s_unrate", "失业率", "%"), ("s_core_yoy", "核心 CPI 同比", "%"),
              ("s_infexp1", "1 年通胀预期（密歇根）", "%"), ("s_spread", "Baa − 10 年利差", "pp"),
              ("s_term", "期限利差（10 年 − 政策利率）", "pp"), ("s_y10", "10 年期收益率", "%"),
              ("s_real_y10", "10 年实际收益率", "%"), ("s_cape", "log 席勒 CAPE", "log"),
              ("s_tariff", "有效关税率", "%"), ("s_sahm", "萨姆规则缺口", "pp"))
LAST_ROWS = (("d_rgdp", "上季实际 GDP（年化）", "%"), ("d_core", "上季核心通胀（年化）", "%"),
             ("d_unrate", "上季失业率变动", "pp"), ("d_gz", "上季 GZ 利差变动", "pp"), ("d_oil", "上季油价", "log%"))
DIGEST_ACTION_MAX = 4
BAND_ZH = {"d_rgdp": ("实际 GDP 增速（年化）", "%"), "d_core": ("核心通胀（年化）", "%"),
           "d_unrate": ("失业率变动", " 个百分点")}       # the memory line and the digest name fields, not ids
ACCOUNT_HEADLINE = (("fiscal.limit_headroom", "联邦债务上限余量"), ("fed.reserves", "准备金"),
                    ("bigtech.fcf", "五家云公司自由现金流"), ("households.saving_rate", "家庭储蓄率"),
                    ("fiscal.debt_gdp", "公众持有债务/GDP"))
MEMORY_MAX = 1500
MAX_PROFILES = 512
BRANCH_MODAL = 0.85                          # an actor whose modal option has ≥ 0.85 does not branch (v1 rule)
MAX_BRANCHING = 3
SPLIT_BUDGET = 1200                          # candidates per level the discrete split aims at (B · K · M_max · cells)
SPLIT_MAX_CELLS = 24                         # cells per consequence branch at most (small beams)
SPLIT_CELL_MIN_PARTICLES = 5                 # a cell with fewer particles joins its nearest larger cell
SPLIT_CELL_MIN_SHARE = 0.01                  # … or with less than 1% of the branch's particles
SPLIT_WEIGHT = {"political": 3.0, "main": 1.0, "attribute": 0.5}   # which discrete states are split first (判断值):
                                             # elections and other political changes, then actors' main decisions
COMMITMENT_KEEP = 0.7                        # prior that a commitment due this quarter is executed (判断值)
FED_FLOOR_FIELD = "s_ff"


# =====================================================================================================================
# Settings
# =====================================================================================================================

@dataclass
class TreeSettings:
    first_quarter: str = FIRST_QUARTER
    depth: int = 8                           # quarters with actors
    llm_depth: int | None = None             # quarters with LLM actors (later quarters prior-only); None = depth
    B: int = 4                               # nodes kept per level
    K: int = 3                               # action clusters per node
    M_max: int = 3                           # consequence branches per cluster
    p_min: float = 0.05
    eps_merge: float = 0.25
    fact_penalty: float = 3.0
    call_mode: str = "hybrid"                # hybrid (separate at the root) | merged | separate
    root_samples: int = 1
    actors: list | None = None               # None = every catalogue actor
    c: dict = field(default_factory=dict)    # per-actor LLM pool weight overrides
    c_scale: float | None = None             # multiplies every c (0 -> the base-rate tree's pooling)
    synth: dict = field(default_factory=lambda: {"mode": "gated", "depth": 4, "c_s": SY.C_S, "c_m": SY.C_M,
                                                 "axes": dict(SY.DEFAULT_AXES)})
    stress_mode: bool = False
    prior_source: str = "auto"               # auto (A0 bins with a real model, actor tables with the stub) | a0 | table
                                             # | model (the model tree's rules on any tree kind)
    tree_kind: str = "agent"                 # agent | base_rate | model
    n_particles: int = 500
    seed: int = 7
    max_calls: int | None = 60               # soft cap: past it, nodes go prior-only (§11.2)
    hard_cap: int | None = None              # G9: reaching it pauses the run ("budget"); jobs set 1.2 × the estimate
    concurrency: int = 3
    model: str | None = None                 # None = MSL_LLM_MODEL / claude-opus-5-5
    deep_model: str | None = None            # optional faster model for levels >= deep_from (None: same model)
    deep_from: int = 8
    split_max: int | None = None             # cells per consequence branch (discrete split); None = auto (2–12)
    effort: dict = field(default_factory=dict)
    safe_mode: bool = True
    keep_paths: str = "auto"                 # full (carry quarterly paths for the hand-off) | none | auto
    prior_trees: bool = True
    overrides: list = field(default_factory=list)       # [{node, actor, option, prob}] (interventions)
    external_priors: dict = field(default_factory=dict)  # {actor: {option: p}} market odds (c = 0 for that actor)
    confirmed: bool = False                  # the user confirmed the estimate (real CLI only); not in the hash
    name: str = ""

    HASH_EXCLUDE = ("concurrency", "max_calls", "hard_cap", "confirmed", "name", "prior_trees")

    @classmethod
    def coerce(cls, s) -> TreeSettings:
        if s is None:
            return cls()
        if isinstance(s, TreeSettings):
            return s
        d = dict(s)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        out = cls(**known)
        syn = {"mode": "gated", "depth": 4, "c_s": SY.C_S, "c_m": SY.C_M, "axes": dict(SY.DEFAULT_AXES)}
        syn.update(dict(d.get("synth") or {}))
        syn["axes"] = {**SY.DEFAULT_AXES, **dict(syn.get("axes") or {})}
        out.synth = syn
        return out

    def to_dict(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def hash(self) -> str:
        d = self.to_dict()
        for k in self.HASH_EXCLUDE:
            d.pop(k, None)
        return hashlib.sha256(canonical(d).encode("utf-8")).hexdigest()[:16]

    def problems(self) -> list[str]:
        out = []
        if not 1 <= int(self.depth) <= 80:
            out.append("depth 必须在 1–80 季之间")
        if not 1 <= int(self.B) <= 30:
            out.append("B（每层保留的节点）必须在 1–30 之间")
        if not 1 <= int(self.K) <= 4:
            out.append("K（每个节点的动作簇）必须在 1–4 之间")
        if not 1 <= int(self.M_max) <= 4:
            out.append("M_max（每个动作簇的后果分支）必须在 1–4 之间")
        if not 50 <= int(self.n_particles) <= 5000:
            out.append("n_particles 必须在 50–5000 之间")
        if self.call_mode not in ("hybrid", "merged", "separate"):
            out.append("call_mode 只能是 hybrid / merged / separate")
        if self.tree_kind not in ("agent", "base_rate", "model"):
            out.append("tree_kind 只能是 agent / base_rate / model")
        if self.prior_source not in ("auto", "a0", "table", "model"):
            out.append("prior_source 只能是 auto / a0 / table / model")
        if (self.synth or {}).get("mode") not in ("off", "gated", "every"):
            out.append("synth.mode 只能是 off / gated / every")
        if self.keep_paths not in ("auto", "full", "none"):
            out.append("keep_paths 只能是 auto / full / none")
        try:
            A.quarter_add(self.first_quarter, 0)
        except ValueError as err:
            out.append(str(err))
        return out

    @property
    def paths(self) -> bool:
        return self.keep_paths == "full" or (self.keep_paths == "auto" and int(self.depth) <= 20)

    @property
    def llm_levels(self) -> int:
        return int(self.depth if self.llm_depth is None else min(int(self.llm_depth), int(self.depth)))


def split_max_of(st) -> int:
    """Cells per consequence branch of the discrete split: the setting, or a level budget of about 1,200 candidates
    (B · K · M_max · cells), between 2 and SPLIT_MAX_CELLS."""
    if st.split_max is not None:
        return max(1, int(st.split_max))
    per = max(1, int(st.B) * int(st.K) * int(st.M_max))
    return int(min(SPLIT_MAX_CELLS, max(2, SPLIT_BUDGET // per)))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def rnd(x, nd: int = 4):
    """Rounded for records (significant digits for large values)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    if not math.isfinite(v):
        return None
    if v == 0:
        return 0.0
    mag = int(math.floor(math.log10(abs(v))))
    return round(v, max(nd - 1 - mag, 0)) if abs(v) >= 1 else round(v, nd + 2)


# =====================================================================================================================
# Particle clouds
# =====================================================================================================================

@dataclass
class Cloud:
    """Equal-weight particles of one node: the model state, the books, last quarter's draws, the path so far."""
    state: dict
    acc: AC.Accounts | None
    x: dict = field(default_factory=dict)
    y: dict = field(default_factory=dict)
    cum: dict = field(default_factory=dict)            # cumulative log total return per asset since the root
    cpi: np.ndarray | None = None                      # cumulative CPI log change since the root
    path: np.ndarray | None = None                     # (N, T, A) float32 quarterly simple total returns
    inc: dict = field(default_factory=dict)            # last quarter's income (simple) per asset

    @property
    def n(self) -> int:
        return int(len(self.state["s_ff"]))

    def take(self, idx) -> Cloud:
        idx = np.asarray(idx, int)
        return Cloud(MA.take_state(self.state, idx), None if self.acc is None else self.acc.take(idx),
                     {k: v[idx].copy() for k, v in self.x.items()}, {k: v[idx].copy() for k, v in self.y.items()},
                     {k: v[idx].copy() for k, v in self.cum.items()}, None if self.cpi is None else self.cpi[idx].copy(),
                     None if self.path is None else self.path[idx].copy(), {k: v[idx].copy() for k, v in self.inc.items()})

    @staticmethod
    def concat(parts: list[Cloud]) -> Cloud:
        p0 = parts[0]
        cat = lambda key: {k: np.concatenate([getattr(p, key)[k] for p in parts]) for k in getattr(p0, key)}  # noqa: E731
        acc = AC.Accounts.concat([p.acc for p in parts]) if p0.acc is not None else None
        return Cloud(MA.concat_states([p.state for p in parts]), acc, cat("x"), cat("y"), cat("cum"),
                     None if p0.cpi is None else np.concatenate([p.cpi for p in parts]),
                     None if p0.path is None else np.concatenate([p.path for p in parts]), cat("inc"))

    def nbytes(self) -> int:
        b = MA.state_nbytes(self.state)
        if self.acc is not None:
            b += MA.state_nbytes(self.acc.s) + sum(v.nbytes for v in self.acc.flows.values() if isinstance(v, np.ndarray))
        for d in (self.x, self.y, self.cum, self.inc):
            b += sum(np.asarray(v).nbytes for v in d.values())
        if self.cpi is not None:
            b += self.cpi.nbytes
        if self.path is not None:
            b += self.path.nbytes
        return int(b)


def acc_arrays(acc: AC.Accounts, prefix: str) -> dict:
    out = {f"{prefix}s:{k}": v for k, v in acc.s.items()}
    out.update({f"{prefix}f:{k}": v for k, v in acc.flows.items() if isinstance(v, np.ndarray)})
    return out


def acc_meta(acc: AC.Accounts) -> dict:
    return {"n": acc.n, "carried": acc.carried, "params": dataclasses.asdict(acc.p), "quarter": acc.quarter,
            "q": acc.q, "fields": acc._fields, "as_of": acc.as_of, "resid": acc.resid}


def acc_restore(meta: Mapping, arrays: Mapping, prefix: str) -> AC.Accounts:
    pd_ = dict(meta["params"])
    if isinstance(pd_.get("private_nl_p01_p99"), list):
        pd_["private_nl_p01_p99"] = tuple(pd_["private_nl_p01_p99"])
    params = AC.Params(**{k: v for k, v in pd_.items() if k in AC.Params.__dataclass_fields__})
    s = {k[len(prefix) + 2:]: np.array(v) for k, v in arrays.items() if k.startswith(prefix + "s:")}
    acc = AC.Accounts(int(meta["n"]), s, dict(meta["carried"]), params, quarter=meta["quarter"], q=int(meta["q"]),
                      as_of=meta.get("as_of", ""))
    acc._fields = dict(meta["fields"])
    acc.flows = {k[len(prefix) + 2:]: np.array(v) for k, v in arrays.items() if k.startswith(prefix + "f:")}
    acc.resid = dict(meta.get("resid") or {})
    return acc


def cloud_arrays(cloud: Cloud, prefix: str) -> tuple[dict, dict]:
    arr = {f"{prefix}m:{k}": v for k, v in cloud.state.items()}
    for key in ("x", "y", "cum", "inc"):
        arr.update({f"{prefix}{key}:{k}": v for k, v in getattr(cloud, key).items()})
    if cloud.cpi is not None:
        arr[f"{prefix}cpi"] = cloud.cpi
    if cloud.path is not None:
        arr[f"{prefix}path"] = cloud.path
    meta = {}
    if cloud.acc is not None:
        arr.update(acc_arrays(cloud.acc, prefix + "a"))
        meta = acc_meta(cloud.acc)
    return arr, meta


def cloud_restore(arrays: Mapping, meta: Mapping, prefix: str) -> Cloud:
    def grab(key):
        pre = f"{prefix}{key}:"
        return {k[len(pre):]: np.array(v) for k, v in arrays.items() if k.startswith(pre)}
    acc = acc_restore(meta, arrays, prefix + "a") if meta else None
    return Cloud(grab("m"), acc, grab("x"), grab("y"), grab("cum"),
                 np.array(arrays[f"{prefix}cpi"]) if f"{prefix}cpi" in arrays else None,
                 np.array(arrays[f"{prefix}path"]) if f"{prefix}path" in arrays else None, grab("inc"))


# =====================================================================================================================
# Small numerics
# =====================================================================================================================

def seed_for(seed: int, key: str, purpose: str) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}|{key}|{purpose}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def allocate(masses, n: int) -> np.ndarray:
    """Largest-remainder allocation of n particles to members by mass (ties broken by order)."""
    m = np.asarray(masses, float)
    m = np.where(m > 0, m, 0.0)
    if m.sum() <= 0:
        m = np.ones(len(m))
    q = m / m.sum() * n
    base = np.floor(q).astype(int)
    rest = n - int(base.sum())
    if rest > 0:
        base[np.argsort(-(q - base), kind="stable")[:rest]] += 1
    return base


def systematic(w, n: int, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(w, float)
    w = w / w.sum()
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), u, side="right"), len(w) - 1)


def resample_rows(rows: np.ndarray, n: int, rng: np.random.Generator, weights=None) -> np.ndarray:
    """n indices drawn from `rows` (equal weights or `weights`), systematic; identity when len(rows) == n and equal."""
    rows = np.asarray(rows, int)
    if len(rows) == 0:
        raise ValueError("没有粒子可以重采样")
    if weights is None and len(rows) == n:
        return rows.copy()
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, float)
    return rows[systematic(w, n, rng)]


def forward_select(p: np.ndarray, D: np.ndarray, keep: int, forced: Iterable[int] = ()) -> tuple[list, np.ndarray]:
    """Heitsch & Römisch (2003) fast forward selection: keep `keep` points minimising Σ_i p_i min_{j∈J} D_ij; every
    other point is assigned to its nearest kept point (its mass is reassigned, not pruned)."""
    n = len(p)
    if n <= keep:
        return list(range(n)), np.arange(n)
    p = np.asarray(p, float)
    J = list(dict.fromkeys(int(i) for i in forced))
    best = D[:, J].min(axis=1) if J else np.full(n, np.inf)
    while len(J) < keep:
        Js = set(J)
        cand = np.array([u for u in range(n) if u not in Js], int)
        cost = (p[:, None] * np.minimum(best[:, None], D[:, cand])).sum(axis=0)
        u = int(cand[int(np.argmin(cost))])
        J.append(u)
        best = np.minimum(best, D[:, u])
    J.sort()
    Ja = np.array(J, int)
    a = Ja[np.argmin(D[:, Ja], axis=1)]
    a[Ja] = Ja
    return J, a


def recombine(p: np.ndarray, D: np.ndarray, same: np.ndarray, eps: float) -> np.ndarray:
    """Near-duplicates (D < eps and identical standing facts) join the heaviest such node; deterministic order."""
    order = np.lexsort((np.arange(len(p)), -p))
    a = np.arange(len(p))
    heads: list = []
    for i in order:
        if heads:
            H = np.array(heads, int)
            ok = np.flatnonzero((D[i, H] < eps) & same[i, H])
            if len(ok):
                a[i] = int(H[ok[0]])
                continue
        heads.append(int(i))
    return a


def wquantiles(values: np.ndarray, weights: np.ndarray, qs=(0.05, 0.5, 0.95)) -> list[float]:
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    cw = cw / cw[-1]
    return [float(v[o][min(np.searchsorted(cw, q), len(v) - 1)]) for q in qs]


# =====================================================================================================================
# Journal and store
# =====================================================================================================================

class Journal:
    """Append-only events (journal.jsonl, SSE source): {seq, t, type, ...}. Thread-safe."""

    def __init__(self, path: Path | None = None, listener: Callable | None = None):
        self.path = path
        self.listener = listener
        self.lock = threading.Lock()
        self.events: list = []
        self.seq = 0
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                self.events.append(ev)
                self.seq = max(self.seq, int(ev.get("seq", 0)))

    def emit(self, event_type: str, **payload) -> dict:
        with self.lock:
            self.seq += 1
            ev = {"seq": self.seq, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": event_type, **payload}
            self.events.append(ev)
            if len(self.events) > 20000:
                self.events = self.events[-10000:]
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        if self.listener is not None:
            try:
                self.listener(ev)
            except Exception:      # noqa: BLE001 - a listener must not break the run
                pass
        return ev

    def since(self, seq: int) -> list:
        with self.lock:
            return [e for e in self.events if int(e.get("seq", 0)) > int(seq)]


def write_json(path: Path, obj, *, gz: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    if gz:
        data = gzip.compress(data, compresslevel=6, mtime=0)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def read_json(path: Path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return json.loads(data.decode("utf-8"))


class Store:
    """One run's folder (also a snapshot's frozen core): run.json (or manifest.json), index.json, levels/<d>.nodes.json.gz
    (records, pruned ones included), levels/<d>.detail.json.gz (details of expanded nodes), journal.jsonl,
    checkpoint/ (the frontier's clouds and the engine state), leaves/<key>.npz (leaf clouds), frontier/<key>.npz (every
    kept node's cloud after the reduction, so a pruned child replays one step from its parent)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def level_path(self, d: int, what: str = "nodes") -> Path:
        return self.root / "levels" / f"{int(d):03d}.{what}.json.gz"

    def write_level(self, d: int, records: list, details: Mapping) -> None:
        write_json(self.level_path(d, "nodes"), {"level": d, "nodes": records}, gz=True)
        write_json(self.level_path(d, "detail"), {"level": d, "details": dict(details)}, gz=True)

    def read_level(self, d: int, what: str = "nodes"):
        p = self.level_path(d, what)
        return read_json(p) if p.exists() else None

    def levels(self) -> list[int]:
        out = []
        for p in sorted((self.root / "levels").glob("*.nodes.json.gz")):
            try:
                out.append(int(p.name.split(".")[0]))
            except ValueError:
                continue
        return out

    def save_checkpoint(self, level: int, frontier: list, engine_state: dict) -> None:
        ck = self.root / "checkpoint"
        ck.mkdir(parents=True, exist_ok=True)
        arrays, metas = {}, {}
        for i, nd in enumerate(frontier):
            a, m = cloud_arrays(nd.cloud, f"n{i}|")
            arrays.update(a)
            metas[nd.key] = m
        tmp = ck / f".frontier.{os.getpid()}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, ck / "frontier.npz")
        write_json(ck / "state.json", {"level": level, "frontier": [nd.record() for nd in frontier],
                                       "acc_meta": metas, "engine": engine_state})

    def load_checkpoint(self) -> tuple[int, list, dict] | None:
        ck = self.root / "checkpoint"
        if not (ck / "state.json").exists() or not (ck / "frontier.npz").exists():
            return None
        st = read_json(ck / "state.json")
        with np.load(ck / "frontier.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        nodes = []
        for i, rec in enumerate(st["frontier"]):
            nd = Node.from_record(rec)
            nd.cloud = cloud_restore(arrays, st["acc_meta"].get(nd.key) or {}, f"n{i}|")
            nodes.append(nd)
        return int(st["level"]), nodes, st.get("engine") or {}

    def save_leaf(self, nd, *, dtype=np.float32) -> None:
        """The leaf's particle cloud (float32; last quarter's x draws and income are not kept: the state has the
        lagged x values) for the hand-off, the leaf mixture and deeper extension."""
        cl = nd.cloud
        lean = Cloud(cl.state, cl.acc, {}, cl.y, cl.cum, cl.cpi, cl.path, {})
        a, meta = cloud_arrays(lean, "full|")
        arr = {k: (v.astype(dtype) if v.dtype == np.float64 else v) for k, v in a.items()}
        d = self.root / "leaves"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{nd.key}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, d / f"{nd.key}.npz")
        write_json(d / f"{nd.key}.json", {"acc_meta": meta, "record": nd.record()})

    def save_frontier(self, nd) -> None:
        """A kept node's cloud as the next level expands it: pooled with the nodes it absorbed and resampled to N,
        which a replay from the root cannot rebuild. float32 like the leaves; the path and last quarter's draws and
        income are not kept (an expansion reads the child's own)."""
        cl = nd.cloud
        a, meta = cloud_arrays(Cloud(cl.state, cl.acc, {}, cl.y, cl.cum, cl.cpi, None, {}), "full|")
        arr = {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in a.items()}
        d = self.root / "frontier"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{nd.key}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, d / f"{nd.key}.npz")
        write_json(d / f"{nd.key}.json", {"acc_meta": meta})

    def load_frontier_cloud(self, key: str) -> Cloud | None:
        p, j = self.root / "frontier" / f"{key}.npz", self.root / "frontier" / f"{key}.json"
        if not p.exists() or not j.exists():
            return None
        meta = read_json(j)
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: (z[k].astype(np.float64) if z[k].dtype == np.float32 else z[k]) for k in z.files}
        return cloud_restore(arrays, meta.get("acc_meta") or {}, "full|")

    def load_leaf_cloud(self, key: str) -> Cloud | None:
        p, j = self.root / "leaves" / f"{key}.npz", self.root / "leaves" / f"{key}.json"
        if not p.exists() or not j.exists():
            return None
        meta = read_json(j)
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: (z[k].astype(np.float64) if z[k].dtype == np.float32 and k.startswith("full|") else z[k])
                      for k in z.files}
        return cloud_restore(arrays, meta.get("acc_meta") or {}, "full|")


# =====================================================================================================================
# Nodes
# =====================================================================================================================

@dataclass
class Node:
    key: str
    parent: str | None
    level: int
    quarter: str                             # the quarter that ended at this node
    p_path: float = 1.0
    p_cond: float = 1.0                      # P(this child | parent) = P(cluster) · P(branch | cluster)
    own_mass: float = 1.0                    # p_path before reassignment
    cluster: dict | None = None
    branch: dict | None = None
    facts: dict = field(default_factory=dict)
    political: dict = field(default_factory=dict)
    commitments: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    cred: dict = field(default_factory=dict)
    memory_line: str = ""
    memory: str = ""
    digest: list = field(default_factory=list)          # news items [kind, text] for the next quarter's prompt
    status: str = "pending"                  # pending | expanded | leaf | pruned | recombined
    absorbed: list = field(default_factory=list)
    absorbed_into: str | None = None
    reason: str = ""
    flags: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    lineage: list = field(default_factory=list)        # [(parent key, cluster id, branch id)] from the root
    stress: bool = False
    cloud: Cloud | None = None

    RECORD = ("key", "parent", "level", "quarter", "p_path", "p_cond", "own_mass", "cluster", "branch", "facts",
              "political", "commitments", "signals", "cred", "memory_line", "memory", "digest", "status", "absorbed",
              "absorbed_into", "reason", "flags", "summary", "children", "lineage", "stress")

    def record(self) -> dict:
        d = {k: getattr(self, k) for k in self.RECORD}
        for k in ("p_path", "p_cond", "own_mass"):
            d[k] = float(d[k])
        if self.status == "pruned":
            # compact: the full memory is the parent's plus memory_line (rebuild_memory); no inputs or attribution
            d["memory"] = ""
            d["summary"] = {k: v for k, v in (d["summary"] or {}).items() if k not in ("inputs", "attribution")}
        return json.loads(json.dumps(d, ensure_ascii=False, default=_jsonable))

    @classmethod
    def from_record(cls, rec: Mapping) -> Node:
        return cls(**{k: copy.deepcopy(rec.get(k)) for k in cls.RECORD if k in rec})

    def skeleton(self) -> dict:
        s = self.summary or {}
        return {"key": self.key, "parent": self.parent, "level": self.level, "quarter": self.quarter,
                "status": self.status, "p_path": rnd(self.p_path, 6), "p_cond": rnd(self.p_cond, 6),
                "own_mass": rnd(self.own_mass, 6), "label": (self.cluster or {}).get("label", "起点"),
                "branch": {k: (self.branch or {}).get(k) for k in ("kind", "label", "axis")} if self.branch else None,
                "flags": self.flags, "absorbed_into": self.absorbed_into,
                "headline": s.get("headline"), "n_children": len(self.children)}


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


ACCOUNT_FACT_ZH = {"arrears": "联邦出现逾期未付（技术性违约）", "debt_limit_binding": "债务上限约束（非常规措施中）",
                   "xdate_passed": "越过 X 日期（逾期未付）"}


def live_facts(facts: Mapping) -> list[str]:
    """Standing facts as Chinese text (metadata keys such as *_expiry_q and synthesizer facts are not listed)."""
    out = []
    for f, v in sorted(facts.items()):
        if not v or f.startswith("synth:") or f.endswith("_expiry_q"):
            continue
        out.append(A.FACTS.get(f, {}).get("zh") or ACCOUNT_FACT_ZH.get(f, f))
    return out


def rebuild_memory(parent_memory: str, line: str, facts: Mapping) -> str:
    """The forward memory of a node from its parent's memory and its own line (the code template of §6.7)."""
    lines = [x for x in (parent_memory or "").split("\n") if x and not x.startswith("现行：")] + ([line] if line else [])
    live = live_facts(facts)
    tail = [f"现行：{'；'.join(live)}"] if live else []
    while len("\n".join(lines + tail)) > MEMORY_MAX and len(lines) > 1:
        lines = lines[1:]
    return "\n".join(lines + tail)


def child_key(parent: str, cluster_head: str, branch_id: str) -> str:
    return hashlib.sha256(f"{parent}|{cluster_head}|{branch_id}".encode("utf-8")).hexdigest()[:16]


# =====================================================================================================================
# The engine: one quarter for one node
# =====================================================================================================================

class Engine:
    """Stateless across nodes except caches; one per run (tree kind)."""

    def __init__(self, settings: TreeSettings, *, adapter=None, llm=None, deep_llm=None, root_accounts=None,
                 institutions: list | None = None, journal: Journal | None = None, priors_tables=None):
        self.st = TreeSettings.coerce(settings)
        self.model = adapter or MA.stub()
        self.llm = llm
        self.deep_llm = deep_llm
        self.journal = journal or Journal()
        self.root_acc = root_accounts
        self.items = institutions if institutions is not None else A.load_institutions()
        self.inst = A.institution_values(self.items)
        self.meta = self.model.field_info()
        self.sd = {f: self._sd(f) for f in self.model.x_fields}
        fields = set(self.model.x_fields) | {k for k in self.meta if k.startswith("s_")}
        self.resolver = A.FieldResolver(fields=fields, info=self.meta)
        self.support = {f: (self.meta.get(f) or {}).get("hist") or {} for f in self.model.x_fields}
        self.tables = priors_tables if priors_tables is not None else P.load_priors()
        self.kind = self.st.tree_kind
        prior_src = self.st.prior_source
        if prior_src in ("auto", "model"):
            prior_src = "table" if getattr(self.model, "stub", True) else "a0"
        # the model tree's prior rules (instruments as bins of A0, other levers masked at the status quo, nature at its
        # tail masses): tree_kind "model", or prior_source "model" on any tree (with c = 0 it is the model tree)
        self.neutral = self.kind == "model" or self.st.prior_source == "model"
        self.prior_src = "a0" if self.neutral else prior_src
        self.actor_ids = [a for a in A.STAGE_ORDER if self.st.actors is None or a in self.st.actors]
        self.assets = tuple(self.model.assets)
        self._payload_cache: dict = {}
        self._lock = threading.Lock()

    def _sd(self, f: str) -> float:
        h = (self.meta.get(f) or {}).get("hist") or {}
        v = h.get("sd_q_1985_2019") or h.get("sd_q")
        if v:
            return float(v)
        try:
            return float(self.model.sd(f))
        except Exception:  # noqa: BLE001
            return 1.0

    # ---------------------------------------------------------------- the root
    def root(self) -> Node:
        n = int(self.st.n_particles)
        state = self.model.root_state(n)
        acc = None
        if self.root_acc is not None:
            acc = AC.Accounts.from_root(self.root_acc, n, institutions=self.inst, resolver=self.resolver,
                                        quarter=ROOT_QUARTER)
        cum = {a: np.zeros(n) for a in self.assets}
        path = np.zeros((n, 0, len(self.assets)), np.float32) if self.st.paths else None
        cloud = Cloud(state, acc, {}, {}, cum, np.zeros(n), path, {})
        rs = A.root_state(self.items)
        facts = {k: v for k, v in rs["facts"].items()}
        key = hashlib.sha256(f"root|{self.st.first_quarter}|{self.model.version}|{canonical(facts)}".encode()).hexdigest()[:16]
        nd = Node(key=key, parent=None, level=0, quarter=A.quarter_add(self.st.first_quarter, -1), facts=facts,
                  political=dict(rs["political"]), cloud=cloud, memory="", lineage=[])
        if self.root_acc is not None:
            nd.digest = [list(t) for t in AC.root_digest_items(self.root_acc, self.inst)]
        else:
            nd.flags.append("accounts_missing")
        nd.summary = self.summarize(nd, None)
        return nd

    # ---------------------------------------------------------------- context for menus and priors
    def covariates(self, nd: Node) -> dict:
        memo = getattr(nd.cloud, "_cov", None)
        if memo is not None:
            return dict(memo)
        out = self._covariates(nd)
        nd.cloud._cov = dict(out)
        return out

    def _covariates(self, nd: Node) -> dict:
        cl = nd.cloud
        st = MA.medians(cl.state, [k for k, v in cl.state.items() if not k.startswith("_") and np.ndim(v) == 1])
        if cl.acc is not None:
            st["spr_inventory_mb"] = float(np.median(cl.acc.s["spr"]))
        if "s_spread" in st:
            h = (self.meta.get("s_spread") or {}).get("hist") or {}
            if h.get("p50") is not None and h.get("sd_q"):
                st["s_spread_pctile"] = float(50 + 50 * math.erf((st["s_spread"] - h["p50"]) / (h["sd_q"] * math.sqrt(2))))
        return st

    def node_ctx(self, nd: Node, q: str, *, gates: dict | None = None, leaders: dict | None = None) -> A.NodeContext:
        due = []
        for cm in nd.commitments:
            try:
                due_in = A.quarters_between(q, cm["due_quarter"])
            except (KeyError, ValueError):
                continue
            due.append({**cm, "due_in": due_in})
        return A.NodeContext(quarter=q, state=self.covariates(nd), facts=dict(nd.facts), political=dict(nd.political),
                             institutions=self.inst, gates=gates or {}, commitments=due,
                             stress_mode=bool(self.st.stress_mode), resolver=self.resolver,
                             leader_choices=dict(leaders or {}))

    # ---------------------------------------------------------------- payloads
    def payload(self, aid: str, oid: str, magnitude=None, custom: dict | None = None,
                commitments: list | None = None) -> dict:
        if oid == "as_committed":
            writes = [dict(cm["write"]) for cm in (commitments or []) if cm.get("actor") == aid]
            return {"actor": aid, "option": oid, "zh": "如期执行", "writes": writes, "events": {"oil_z": [], "gpr": []},
                    "facts": [], "clears": [], "political": {}, "account": {}, "flags": [], "protocol": None}
        if oid == "custom" and custom is not None:
            rules = A.custom_option_rules(aid, self.resolver)
            writes = []
            for f, v in sorted((custom.get("writes") or {}).items()):
                kind = "point" if f in rules["kinds"]["point"] else "shift"
                writes.append(A._w(f, "custom", kind, float(v), 0.0, 0, f"actor:{aid}", note="自定义选项"))
            facts = [{"id": f, "value": True, "exit": A.FACTS.get(f, {}).get("exit"), "zh": A.FACTS.get(f, {}).get("zh", f)}
                     for f in custom.get("facts") or []]
            return {"actor": aid, "option": "custom", "zh": custom.get("label") or "自定义", "writes": writes,
                    "events": {"oil_z": [], "gpr": []}, "facts": facts, "clears": [], "political": {}, "account": {},
                    "flags": ["custom"], "protocol": "thin"}
        key = (aid, oid, None if magnitude is None else round(float(magnitude), 6))
        with self._lock:
            hit = self._payload_cache.get(key)
        if hit is None:
            hit = A.option_payload(aid, oid, magnitude, resolver=self.resolver, support=self.support)
            with self._lock:
                self._payload_cache[key] = hit
        return hit

    def features(self, payload: dict) -> dict:
        """Standardised expected writes + facts + political keys + commitments: the option's position in payload space
        (§8.1: each differing fact counts fact_penalty sd, as in the level reduction)."""
        f: dict = {}
        for w in payload.get("writes", []):
            if int(w.get("dq", 0)) > 0 or w["kind"] == "label":
                continue
            fid = w["field"]
            f[fid] = f.get(fid, 0.0) + float(w.get("a") or 0.0) / self.sd.get(fid, 1.0)
        for e in payload.get("events", {}).get("oil_z", []):
            f["oil_z"] = f.get("oil_z", 0.0) + float(e["a"])
        for e in payload.get("events", {}).get("gpr", []):
            f["gpr_sd"] = f.get("gpr_sd", 0.0) + float(e["a_sd"])
        fp = float(self.st.fact_penalty)
        for fa in payload.get("facts", []):
            f[f"fact:{fa['id']}"] = fp
        for k in payload.get("political", {}):
            f[f"pol:{k}"] = fp
        if any(int(w.get("dq", 0)) > 0 for w in payload.get("writes", [])):
            f["commit"] = fp
        return f

    # ---------------------------------------------------------------- decisions
    def _llm_for(self, level: int):
        if self.kind != "agent" or level >= self.st.llm_levels:
            return None
        if self.deep_llm is not None and level >= int(self.st.deep_from):
            return self.deep_llm
        return self.llm

    def c_of(self, aid: str) -> float:
        if self.kind != "agent":
            return 0.0
        if aid in (self.st.external_priors or {}):
            return 0.0
        c = float((self.st.c or {}).get(aid, A.ACTORS[aid].pool_c if aid in A.ACTORS else A.NATURE_POOL_C))
        if aid == "voter" and self.inst.get("election_2026_odds_house_flip") not in (None, ""):
            c = 0.0
        if self.st.c_scale is not None:
            c *= float(self.st.c_scale)
        return float(min(max(c, 0.0), 1.0))

    def a0_bins(self, aid: str, ids: list, a0x: Mapping) -> dict | None:
        """The model's own prior: the share of A0 particles in each option's bin (merged to cover the support)."""
        bins = [(o, lo, hi) for o, lo, hi in A.instrument_bins(aid) if o in ids]
        if not bins:
            return None
        roles = {w.role for o in A.ACTORS[aid].main.options for w in o.writes if w.kind == "point"}
        fid = next((self.resolver.field(r) for r in sorted(roles) if self.resolver.field(r)), None)
        if fid is None or fid not in a0x:
            return None
        bins = sorted(bins, key=lambda t: t[1])
        edges = [(-math.inf if i == 0 else lo, math.inf if i == len(bins) - 1 else hi)
                 for i, (o, lo, hi) in enumerate(bins)]
        for i in range(1, len(edges)):            # contiguous: each bin starts where the previous one ends
            edges[i] = (edges[i - 1][1], edges[i][1])
        v = np.asarray(a0x[fid])
        out = {"p": {}, "edges": {}, "field": fid, "source": "a0_bins", "label": "模型自身 A0 分箱", "kind": "model",
               "values": v}
        for (o, _, _), (lo, hi) in zip(bins, edges):
            out["p"][o] = float(np.mean((v > lo) & (v <= hi)))
            out["edges"][o] = (lo, hi)
        for o in ids:
            out["p"].setdefault(o, 0.0)
        return out

    def decide(self, nd: Node, q: str, a0, gates: dict, calls: list, report: list) -> dict:
        """Per active actor: menu, prior, LLM answer, pooled probabilities (and follower conditionals)."""
        ctx = self.node_ctx(nd, q, gates=gates)
        acts = [a for a in A.active_actors(ctx) if a["llm"] and a["active"] and a["id"] in self.actor_ids]
        out: dict = {"actors": {}, "order": [], "hazards": {}, "flags": [], "packets": {}}
        menus = {}
        for a in acts:
            m = A.menu(a["id"], ctx)
            if m.get("main") and m["main"].get("options"):
                menus[a["id"]] = m
        for aid, m in menus.items():
            out["actors"][aid] = self._actor_block(aid, m, ctx, a0, nd)
            out["order"].append(aid)
        self._followers(out, ctx)
        hz = SY.hazard_masses(a0.x)
        table = {r.get("id"): r for r in P.hazard_table(self.tables)} if hasattr(P, "hazard_table") else {}
        for h, info in hz.items():
            out["hazards"][h] = {**info, "zh": A.HAZARDS.get(h, {}).get("zh", h), "p_llm": None, "p": info["m"],
                                 "table_rate": (table.get(h) or {}).get("p_q")}
        llm = self._llm_for(nd.level)
        if llm is not None and out["actors"] and (any(ab["c"] > 0 for ab in out["actors"].values())
                                                  or (self.st.c_scale is None or self.st.c_scale > 0)
                                                  and nd.level > 0 and out["hazards"]):
            self._ask(nd, q, out, llm, calls, report)
        self._pool(out)
        self._overrides(nd, out)
        self._follower_marginals(out)
        return out

    def _actor_block(self, aid: str, m: dict, ctx, a0, nd: Node) -> dict:
        main = m["main"]
        ids = [o["id"] for o in main["options"]]
        ext = (self.st.external_priors or {}).get(aid)
        pr = None
        if ext:
            pr = {"p": P.normalize({k: float(v) for k, v in ext.items()}, ids, 0.0), "source": "market", "label": "用户输入的市场赔率",
                  "kind": "market"}
        if pr is None and self.prior_src == "a0":
            pr = self.a0_bins(aid, ids, a0.x)
        if pr is None:
            pr = P.prior(aid, ids, ctx, tables=self.tables)
        sq = next((o["id"] for o in main["options"] if o.get("status_quo")), ids[0])
        model_tree_sq = self.neutral and pr.get("source") != "a0_bins"
        if model_tree_sq:                                    # the model tree: shift levers at the status quo
            pr = {"p": {o: (1.0 if o == sq else 0.0) for o in ids}, "source": "status_quo", "label": "现状（模型树）",
                  "kind": "model"}
        keep = list(dict.fromkeys([sq] + list(main.get("required") or [])))
        p = dict(pr["p"])
        if self._keeps_commitment(ids, pr):
            p = self._keep_rule(p)
            pr = {**pr, "p": p, "label": (pr.get("label") or "") + f"；到期承诺如期执行 {COMMITMENT_KEEP:.0%}（判断值）"}
        live = ids if (self.neutral or pr.get("source") == "a0_bins") else \
            A.trim_menu_by_prior(ids, p, cover=0.90, max_options=int(main.get("max_options") or 4), keep=keep)
        if pr.get("source") == "a0_bins" and len(live) > 4:
            live = A.trim_menu_by_prior(ids, p, cover=0.995, max_options=7, keep=keep)
        merged = self._merge_trimmed(aid, ids, live, p, pr)
        opts = []
        for o in main["options"]:
            if o["id"] not in live:
                continue
            mag = (o.get("param") or {}).get("default")
            opts.append({"id": o["id"], "label": o.get("zh") or o["id"], "status_quo": bool(o.get("status_quo")),
                         "p_prior": float(merged.get(o["id"], 0.0)), "p_llm": None, "p_used": None, "magnitude": mag,
                         "signal": None, "rationale": "", "source": "menu", "param": o.get("param"),
                         "bin": (pr.get("edges") or {}).get(o["id"]), "summary": o})
        attrs = []
        for d in m.get("attributes") or []:
            dids = [o["id"] for o in d["options"]]
            if not dids:
                continue
            if self.neutral:
                dsq = next((o["id"] for o in d["options"] if o.get("status_quo")), dids[0])
                dp = {o: (1.0 if o == dsq else 0.0) for o in dids}
            else:
                dp = P.prior(aid, dids, ctx, domain=d["domain"], tables=self.tables)["p"]
            attrs.append({"domain": d["domain"], "zh": d["zh"],
                          "options": [{"id": o["id"], "label": o.get("zh") or o["id"], "p": float(dp.get(o["id"], 0.0))}
                                      for o in d["options"]]})
        return {"id": aid, "zh": m.get("zh") or A.ACTORS[aid].zh, "stage": A.ACTORS[aid].stage, "domain": main["domain"],
                "menu": m, "options": opts, "prior": {k: pr.get(k) for k in ("source", "label", "kind")},
                "c": self.c_of(aid), "attributes": attrs, "fallback": False, "flags": [], "follower_of": None,
                "cond": None, "cond_prior": None, "answer": None, "required": list(main.get("required") or [])}

    def _keeps_commitment(self, ids, pr: Mapping) -> bool:
        return "as_committed" in ids and not self.neutral and pr.get("source") != "market"

    @staticmethod
    def _keep_rule(p: Mapping) -> dict:
        """A commitment due this quarter is kept unless the actor changes course (判断值 COMMITMENT_KEEP): 如期执行
        gets COMMITMENT_KEEP, the other options share the rest in proportion to their prior."""
        rest = sum(v for k, v in p.items() if k != "as_committed")
        return {k: (COMMITMENT_KEEP if k == "as_committed" else (1 - COMMITMENT_KEEP) * v / rest if rest > 0 else 0.0)
                for k, v in p.items()}

    def _merge_trimmed(self, aid: str, ids: list, live: list, p: dict, pr: dict) -> dict:
        """Trimmed options' prior mass joins the nearest kept option by standardised write distance (mass kept)."""
        out = {o: float(p.get(o, 0.0)) for o in live}
        feats = {o: self.features(self.payload(aid, o)) for o in ids}
        for o in ids:
            if o in live or p.get(o, 0.0) <= 0:
                continue
            fo = feats[o]

            def dist(k, fo=fo):
                fk = feats[k]
                keys = set(fo) | set(fk)
                return math.sqrt(sum((fo.get(x, 0.0) - fk.get(x, 0.0)) ** 2 for x in keys))
            near = min(live, key=lambda k: (dist(k), live.index(k)))
            out[near] += float(p[o])
        tot = sum(out.values())
        if tot <= 0:
            return {o: 1.0 / len(live) for o in live}
        if pr.get("edges"):
            # A0 bins: the kept bins widen over the trimmed ones, and each kept option's prior is the share of A0
            # particles in its widened bin (so masses and particles agree exactly: neutrality)
            edges = pr["edges"]
            kept = sorted(live, key=lambda o: edges[o][0])
            new = {}
            for i, o in enumerate(kept):
                lo = -math.inf if i == 0 else new[kept[i - 1]][1]
                hi = math.inf if i == len(kept) - 1 else edges[kept[i + 1]][0]
                new[o] = (lo, hi)
            pr["edges"] = new
            v = np.asarray(pr.get("values"), float)
            if v.size:
                return {o: float(np.mean((v > new[o][0]) & (v <= new[o][1]))) for o in live}
        return {o: v / tot for o, v in out.items()}

    def _followers(self, out: dict, ctx) -> None:
        """Reaction rows P(follower option | leader option) for the catalogue's follower pairs. Each row goes through
        the same steps as the follower's unconditional prior (the history prior given the leader's choice, the
        commitment keep rule, trimmed options' mass to the nearest kept option), so a row and the marginal shown
        for the follower (§9.3, _follower_marginals) are one model."""
        acts = out["actors"]
        for rr in A.REACTIONS:
            if rr["kind"] not in ("rows", "tilt"):
                continue
            fa, fd = rr["follower"]
            la, ld = rr["leader"]
            if fa not in acts or la not in acts or acts[fa]["domain"] != fd or acts[la]["domain"] != ld:
                continue
            if acts[fa]["follower_of"]:
                continue
            fb = acts[fa]
            live = [o["id"] for o in fb["options"]]
            all_ids = [o["id"] for o in (fb["menu"].get("main") or {}).get("options") or []] or live
            cond = {}
            for lo in [o["id"] for o in acts[la]["options"]]:
                if self.neutral or fb["prior"].get("source") in ("market", "a0_bins", "status_quo"):
                    cond[lo] = {o["id"]: o["p_prior"] for o in fb["options"]}
                    continue
                c2 = A.NodeContext(**{**ctx.__dict__, "leader_choices": {la: lo}})
                pr = P.prior(fa, all_ids, c2, tables=self.tables)
                pp = dict(pr["p"])
                if self._keeps_commitment(all_ids, pr):
                    pp = self._keep_rule(pp)
                row = self._merge_trimmed(fa, all_ids, live, pp, {})
                cond[lo] = {k: float(row.get(k, 0.0)) for k in live}
            fb["follower_of"] = la
            fb["cond_prior"] = cond
            fb["reaction_source"] = rr["source"]

    def _follower_marginals(self, out: dict) -> None:
        """A follower's shown probabilities are the marginals of its rows under the leader's adopted probabilities,
        p(o) = Σ_l p_used(l) · p(o | l), for 采用概率, 大模型判断 and 历史基率 alike (the direct unconditional answers
        are kept as *_uncond). The tree realizes exactly these: profiles and particle slots use the rows (and the
        leader is allocated from the joint when only the follower branches)."""
        acts = out["actors"]
        for aid in out["order"]:
            ab = acts[aid]
            la = ab.get("follower_of")
            if not ab.get("cond") or la not in acts:
                continue
            pl = {o["id"]: float(o["p_used"]) for o in acts[la]["options"]}
            tot_l = sum(pl.values()) or 1.0
            pl = {k: v / tot_l for k, v in pl.items()}
            cond, cond_p, cond_l = ab["cond"], ab.get("cond_prior") or {}, ab.get("cond_llm") or {}
            for o in ab["options"]:
                o["p_used_uncond"], o["p_llm_uncond"], o["p_prior_uncond"] = o["p_used"], o["p_llm"], o["p_prior"]
            for o in ab["options"]:
                oid = o["id"]
                o["p_used"] = float(sum(w * float((cond.get(lo) or {}).get(oid, 0.0)) for lo, w in pl.items()))
                o["p_prior"] = float(sum(w * float((cond_p.get(lo) or {}).get(oid, 0.0)) for lo, w in pl.items()))
                if o["p_llm_uncond"] is not None:
                    o["p_llm"] = float(sum(w * float((cond_l.get(lo) or {}).get(oid, o["p_llm_uncond"]))
                                           for lo, w in pl.items()))
            for key in ("p_used", "p_prior"):
                t = sum(float(o[key] or 0.0) for o in ab["options"])
                if t > 0:
                    for o in ab["options"]:
                        o[key] = float(o[key] or 0.0) / t
            ab["marginal_of_rows"] = True

    # ---------------------------------------------------------------- the LLM round
    def _specs(self, nd: Node, out: dict, ctx_prompt, *, separate: bool) -> dict:
        from .llm_io import actor_spec_from_menu
        specs = {}
        for aid in out["order"]:
            ab = out["actors"][aid]
            live = {o["id"] for o in ab["options"]}
            menu = copy.deepcopy(ab["menu"])
            menu["main"]["options"] = [o for o in menu["main"]["options"] if o["id"] in live]
            prior = {"p": {o["id"]: o["p_prior"] for o in ab["options"]}, "source": ab["prior"].get("label") or "",
                     "kind": ab["prior"].get("kind")}
            rules = A.custom_option_rules(aid, self.resolver)
            levers = [{"id": f, **(self.meta.get(f) or {})} for f in rules["writable_fields"]]
            own = ""
            if nd.cloud.acc is not None or self.root_acc is not None:
                try:
                    own = AC.own_block(aid, self.root_acc, nd.cloud.acc if nd.level > 0 else None,
                                       institutions=self.inst)
                except (KeyError, ValueError, TypeError, ZeroDivisionError) as err:
                    own = f"（账户块不可用：{type(err).__name__}）"
            due = [f"{cm.get('zh') or cm.get('option')}（{cm.get('due_quarter')} 到期）" for cm in nd.commitments
                   if cm.get("actor") == aid]
            role = A.SIGNAL_INSTRUMENT_ROLE.get(aid)
            inst = self.resolver.field(role) if role else None
            specs[aid] = actor_spec_from_menu(menu, prior, levers=levers if rules["allowed"] else [],
                                              facts=rules["facts"] if rules["allowed"] else [],
                                              custom=bool(rules["allowed"]), custom_cap=float(rules["cap"] or 0.15),
                                              instruments=[inst] if inst else [], packet=own, private="",
                                              commitments=due)
            out["packets"][aid] = AC.packet(aid, own, AC.news_digest([tuple(t) for t in nd.digest]),
                                            self.board_text(nd, aid))
        return specs

    def _reaction_specs(self, out: dict) -> list:
        from .llm_io import ReactionSpec
        rs = []
        for aid in out["order"]:
            ab = out["actors"][aid]
            if not ab.get("follower_of"):
                continue
            la = ab["follower_of"]
            rs.append(ReactionSpec(leader=la, follower=aid, leader_options=[o["id"] for o in out["actors"][la]["options"]],
                                   follower_options=[o["id"] for o in ab["options"]], prior=ab["cond_prior"],
                                   leader_zh=out["actors"][la]["zh"], follower_zh=ab["zh"],
                                   why=ab.get("reaction_source") or ""))
        return rs

    def _ask(self, nd: Node, q: str, out: dict, llm, calls: list, report: list) -> None:
        from .llm_io import HazardSpec
        ctx_p = self.prompt_ctx(nd, q, out)
        separate = self.st.call_mode == "separate" or (self.st.call_mode == "hybrid" and nd.level == 0)
        specs = self._specs(nd, out, ctx_p, separate=separate)
        reactions = self._reaction_specs(out)
        if separate:
            by_f = {r.follower: r for r in reactions}
            n_s = max(1, int(self.st.root_samples)) if nd.level == 0 else 1
            jobs = [(aid, k) for aid in out["order"] for k in range(n_s)]

            def one(job):
                aid, k = job
                return llm.actor(ctx_p, specs[aid], follower=by_f.get(aid), sample=k, root=nd.level == 0)
            # the actors' separate calls are independent (R4: simultaneous moves): k at a time, consumed in order
            workers = max(1, min(int(self.st.concurrency), len(jobs)))
            if workers > 1:
                with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                    results = list(ex.map(one, jobs))
            else:
                results = [one(j) for j in jobs]
            for aid in out["order"]:
                mine = [r for (a, _), r in zip(jobs, results) if a == aid]
                for res in mine:
                    calls += [c.to_dict() for c in res.calls]
                    report += res.report
                    out["actors"][aid]["flags"] += list(res.flags)
                out["actors"][aid]["answer"] = _average_answers([r.answer for r in mine])
                last = mine[-1].answer if mine else None
                if aid in by_f and getattr(last, "reaction", None):
                    out["actors"][aid]["cond_llm"] = last.reaction
        else:
            hazards = [HazardSpec(id=h, zh=v["zh"], mass=v["m"], table_rate=v.get("table_rate"))
                       for h, v in sorted(out["hazards"].items()) if nd.level > 0 and v["m"] > 0]
            res = llm.node(ctx_p, [specs[a] for a in out["order"]], reactions=reactions, hazards=hazards)
            calls += [c.to_dict() for c in res.calls]
            report += res.report
            for aid in out["order"]:
                out["actors"][aid]["answer"] = res.actors.get(aid)
            for leader, by in (res.reactions or {}).items():
                for follower, rows in by.items():
                    if follower in out["actors"]:
                        out["actors"][follower]["cond_llm"] = rows
            for h, p in (res.hazards or {}).items():
                if h in out["hazards"]:
                    out["hazards"][h]["p_llm"] = float(p)
            out["flags"] += res.flags
        if any("budget" in (out["actors"][a]["flags"] or []) for a in out["order"]) or "budget" in out["flags"]:
            out["flags"].append("budget")

    # ---------------------------------------------------------------- pooling (§9.3)
    def _pool(self, out: dict) -> None:
        for aid in out["order"]:
            ab = out["actors"][aid]
            ans = ab.get("answer")
            c = ab["c"]
            p_llm = {}
            if ans is not None and not getattr(ans, "prior_only", False) and not getattr(ans, "fallback", False):
                for o in ans.options:
                    if o.get("p_llm") is not None:
                        p_llm[o["id"]] = float(o["p_llm"])
                ab["answered"] = True
            else:
                if ans is not None:
                    ab["fallback"] = True
                c = 0.0
            ab["c_used"] = c
            ids = {o["id"] for o in ab["options"]}
            if ans is not None:
                for o in ans.options:
                    if o.get("custom") and o["id"] == "custom" and o.get("p_llm") is not None and c > 0:
                        ab["options"].append({"id": "custom", "label": o.get("label") or "自定义", "status_quo": False,
                                              "p_prior": 0.0, "p_llm": float(o["p_llm"]), "p_used": None,
                                              "magnitude": None, "signal": o.get("signal"),
                                              "rationale": o.get("rationale") or "", "source": "custom",
                                              "custom": {"writes": o.get("writes") or {}, "facts": o.get("facts") or [],
                                                         "label": o.get("label")}})
                        ids.add("custom")
                    elif o["id"] in ids:
                        tgt = next(x for x in ab["options"] if x["id"] == o["id"])
                        if o.get("magnitude") is not None and tgt.get("param"):
                            tgt["magnitude"] = float(o["magnitude"])
                        tgt["signal"] = o.get("signal")
                        tgt["rationale"] = o.get("rationale") or ""
            if ab.get("answered") and not p_llm and ab.get("cond_llm") and ab.get("follower_of") in out["actors"]:
                lead = out["actors"][ab["follower_of"]]
                for lo in lead["options"]:
                    row = ab["cond_llm"].get(lo["id"]) or {}
                    for oid, pv in row.items():
                        p_llm[oid] = p_llm.get(oid, 0.0) + float(lo["p_used"]) * float(pv)
            for o in ab["options"]:
                o["p_llm"] = p_llm.get(o["id"], 0.0 if ab.get("answered") else None)
                pl = o["p_llm"] if o["p_llm"] is not None else o["p_prior"]
                o["p_used"] = c * pl + (1.0 - c) * o["p_prior"]
            tot = sum(o["p_used"] for o in ab["options"])
            for o in ab["options"]:
                o["p_used"] = o["p_used"] / tot if tot > 0 else 1.0 / len(ab["options"])
            if ab.get("cond_prior"):
                cl = ab.get("cond_llm") or {}
                cond = {}
                for lo, row in ab["cond_prior"].items():
                    lrow = cl.get(lo) or {}
                    r = {}
                    for o in ab["options"]:
                        pp = float(row.get(o["id"], 0.0))
                        pl = float(lrow.get(o["id"], o["p_llm"] if o["p_llm"] is not None else pp))
                        r[o["id"]] = c * pl + (1.0 - c) * pp
                    t = sum(r.values())
                    cond[lo] = {k: (v / t if t > 0 else 1.0 / len(r)) for k, v in r.items()}
                ab["cond"] = cond
        c_nat = A.NATURE_POOL_C * (1.0 if self.st.c_scale is None else float(self.st.c_scale))
        for h, v in out["hazards"].items():
            if v.get("p_llm") is not None and self.kind == "agent" and c_nat > 0:
                lo, hi = v["m"] * A.HAZARD_TILT_BOUNDS[0], min(1.0, v["m"] * A.HAZARD_TILT_BOUNDS[1])
                pl = float(np.clip(v["p_llm"], lo, hi))
                v["p"] = c_nat * pl + (1.0 - c_nat) * v["m"]
            else:
                v["p"] = v["m"]

    def _overrides(self, nd: Node, out: dict) -> None:
        for ov in self.st.overrides or []:
            if ov.get("node") != nd.key or ov.get("actor") not in out["actors"]:
                continue
            ab = out["actors"][ov["actor"]]
            prob = float(ov.get("prob", 1.0))
            if ov.get("option") not in {o["id"] for o in ab["options"]}:
                continue
            rest = sum(o["p_used"] for o in ab["options"] if o["id"] != ov["option"])
            for o in ab["options"]:
                o["p_used"] = prob if o["id"] == ov["option"] else ((1 - prob) * o["p_used"] / rest if rest > 0 else 0.0)
            ab["cond"] = None
            ab["flags"].append("intervened")
            out["flags"].append("intervened")

    # ---------------------------------------------------------------- joint profiles and clusters (§8.1)
    def clusters(self, nd: Node, dec: dict) -> list[dict]:
        acts = dec["actors"]
        impact = {}
        for aid in dec["order"]:
            ab = acts[aid]
            ps = np.array([o["p_used"] for o in ab["options"]])
            if len(ps) < 2 or ps.max() >= BRANCH_MODAL:
                continue
            feats = [self.features(self._opt_payload(aid, o, nd)) for o in ab["options"]]
            keys = sorted({k for f in feats for k in f})
            if not keys:
                continue
            W = np.array([[f.get(k, 0.0) for k in keys] for f in feats])
            wbar = ps @ W
            impact[aid] = float(np.sum(ps * np.sqrt(((W - wbar) ** 2).sum(axis=1))))
        branching = [a for a, v in sorted(impact.items(), key=lambda t: (-t[1], A.STAGE_ORDER.index(t[0]))) if v > 0]
        branching = sorted(branching[:MAX_BRANCHING], key=lambda a: A.STAGE_ORDER.index(a))
        for aid in dec["order"]:
            acts[aid]["branching"] = aid in branching
        profiles = [({}, 1.0)]
        for aid in branching:
            ab = acts[aid]
            new = []
            for choice, p in profiles:
                la = ab.get("follower_of")
                if ab.get("cond") and la in choice:
                    row = ab["cond"].get(choice[la]) or {o["id"]: o["p_used"] for o in ab["options"]}
                else:
                    row = {o["id"]: o["p_used"] for o in ab["options"]}
                for o in ab["options"]:
                    q = float(row.get(o["id"], 0.0))
                    if q > 0:
                        new.append(({**choice, aid: o["id"]}, p * q))
            new.sort(key=lambda t: -t[1])
            profiles = new[:MAX_PROFILES]
        tot = sum(p for _, p in profiles) or 1.0
        profiles = [(c, p / tot) for c, p in profiles]
        K = int(self.st.K)
        if self.neutral:
            K = max(K, len(profiles)) if len(profiles) <= 8 else K
        if len(profiles) <= 1:
            heads = [(profiles[0][0], [0])]
        else:
            feats = []
            for choice, _ in profiles:
                f: dict = {}
                for aid, oid in choice.items():
                    for k, v in self.features(self._opt_payload(aid, self._opt(acts[aid], oid), nd)).items():
                        f[f"{aid}:{k}"] = f.get(f"{aid}:{k}", 0.0) + v
                feats.append(f)
            keys = sorted({k for f in feats for k in f})
            X = np.array([[f.get(k, 0.0) for k in keys] for f in feats])
            D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
            pv = np.array([p for _, p in profiles])
            elig = [i for i in range(len(pv)) if pv[i] >= self.st.p_min] or [int(np.argmax(pv))]
            keep = min(K, len(elig))
            sub = np.array(elig)
            J_sub, _ = forward_select(pv[sub], D[np.ix_(sub, sub)], keep, forced=[int(np.argmax(pv[sub]))])
            J = [int(sub[j]) for j in J_sub]
            assign = np.array([J[int(np.argmin(D[i, J]))] for i in range(len(pv))])
            heads = [(profiles[j][0], [i for i in range(len(pv)) if assign[i] == j]) for j in J]
        out = []
        for head, members in heads:
            mem = [{"options": profiles[i][0], "p": float(profiles[i][1])} for i in members]
            out.append({"head": head, "members": mem, "p": float(sum(m["p"] for m in mem)),
                        "head_key": canonical(head)})
        out.sort(key=lambda c: c["head_key"])
        for i, c in enumerate(out, start=1):
            c["id"] = f"c{i}"
            c["label"] = self.cluster_label(c["head"], acts)
        return out

    @staticmethod
    def _opt(ab: dict, oid: str) -> dict:
        return next(o for o in ab["options"] if o["id"] == oid)

    def _opt_payload(self, aid: str, o: dict, nd: Node) -> dict:
        q = A.quarter_add(nd.quarter, 1)
        due = [cm for cm in nd.commitments if cm.get("due_quarter") == q]      # 如期执行 writes only what is due
        return self.payload(aid, o["id"], o.get("magnitude"), custom=o.get("custom"), commitments=due)

    def cluster_label(self, head: Mapping, acts: Mapping) -> str:
        if not head:
            return "各方按先验（未分叉）"
        parts = []
        for aid, oid in head.items():
            ab = acts.get(aid) or {}
            lab = next((o["label"] for o in ab.get("options", []) if o["id"] == oid), oid)
            parts.append(f"{ab.get('zh', aid)}：{lab}")
        return "；".join(parts)

    # ---------------------------------------------------------------- allocation and B_k (§6.3, §6.4)
    def allocate_cluster(self, nd: Node, cl: dict, dec: dict, a0, rng: np.random.Generator) -> dict:
        """Particle slots of a cluster: each slot gets a member profile (by mass) and an option for every other
        actor (pooled probabilities; followers conditional on their leader's slot option) and for every attribute
        domain (its prior). Model tree: bin actors take the A0 particles of their bin (no random allocation)."""
        acts = dec["actors"]
        n = nd.cloud.n
        rows = np.arange(n)
        member = np.repeat(np.arange(len(cl["members"])), allocate([m["p"] for m in cl["members"]], n))
        rng.shuffle(member)
        assign: dict = {}
        bin_actors = [aid for aid in dec["order"] if self.neutral and any(o.get("bin") for o in acts[aid]["options"])]
        if bin_actors:
            # the model tree: a particle's bin option is the bin its own A0 draw falls in
            keep = np.ones(n, bool)
            for aid in bin_actors:
                ab = acts[aid]
                fid = (self.a0_bins(aid, [o["id"] for o in ab["options"]], a0.x) or {}).get("field")
                v = np.asarray(a0.x[fid])
                code = np.full(n, -1)
                for k, o in enumerate(ab["options"]):
                    lo, hi = o["bin"]
                    code[(v > lo) & (v <= hi)] = k
                assign[aid] = code
                if ab.get("branching"):
                    ok = np.zeros(n, bool)
                    for m in cl["members"]:
                        ok |= code == [o["id"] for o in ab["options"]].index(m["options"][aid])
                    keep &= ok
            rows = np.flatnonzero(keep)
            member = np.zeros(len(rows), int)
            for i, m in enumerate(cl["members"]):
                sel = np.ones(len(rows), bool)
                for aid, oid in m["options"].items():
                    if aid in bin_actors:
                        sel &= assign[aid][rows] == [o["id"] for o in acts[aid]["options"]].index(oid)
                member[sel] = i
            assign = {k: v[rows] for k, v in assign.items()}
        m_n = len(rows)
        for aid in dec["order"]:                    # branching actors: the slot's member profile
            ab = acts[aid]
            if aid in assign or not ab.get("branching"):
                continue
            ids = [o["id"] for o in ab["options"]]
            assign[aid] = np.array([ids.index(cl["members"][member[i]]["options"][aid]) for i in range(m_n)], int)
        for aid in dec["order"]:                    # the others, in stage order, from the joint (§9.3):
            ab = acts[aid]                          # p(x | leader, followers) ∝ p(x | leader) · Π_f p(f | x)
            if aid in assign:
                continue
            ids = [o["id"] for o in ab["options"]]
            la = ab.get("follower_of") if ab.get("cond") else None
            la = la if la in assign else None
            fols = [f for f in dec["order"] if f in assign and acts[f].get("follower_of") == aid and acts[f].get("cond")]
            if la is None and not fols:
                cnt = allocate([o["p_used"] for o in ab["options"]], m_n)
                code = np.repeat(np.arange(len(ids)), cnt)
                rng.shuffle(code)
                assign[aid] = code
                continue
            keys = np.stack(([assign[la]] if la else []) + [assign[f] for f in fols], axis=1)
            code = np.zeros(m_n, int)
            uk, inv = np.unique(keys, axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            lids = [o["id"] for o in acts[la]["options"]] if la else []
            base = {o["id"]: float(o["p_used"]) for o in ab["options"]}
            for g, kv in enumerate(uk):
                sel = np.flatnonzero(inv == g)
                if la:
                    w = dict((ab["cond"] or {}).get(lids[int(kv[0])]) or base)
                else:
                    w = dict(base)
                for j, f in enumerate(fols):
                    fb = acts[f]
                    fo = fb["options"][int(kv[j + (1 if la else 0)])]["id"]
                    for o in ids:
                        w[o] = float(w.get(o, 0.0)) * float(((fb["cond"] or {}).get(o) or {}).get(fo, 0.0))
                ws = [max(float(w.get(o, 0.0)), 0.0) for o in ids]
                if sum(ws) <= 0:                     # an impossible joint (rounding): the actor's own marginal
                    ws = [base[o] for o in ids]
                cnt = allocate(ws, len(sel))
                lab = np.repeat(np.arange(len(ids)), cnt)
                rng.shuffle(lab)
                code[sel] = lab
            assign[aid] = code
        attr: dict = {}
        for aid in dec["order"]:
            for d in acts[aid]["attributes"]:
                cnt = allocate([o["p"] for o in d["options"]], m_n)
                code = np.repeat(np.arange(len(d["options"])), cnt)
                rng.shuffle(code)
                attr[(aid, d["domain"])] = code
        return {"rows": rows, "assign": assign, "attr": attr, "member": member}

    def build_bk(self, nd: Node, cl: dict, dec: dict, a0, alloc: dict, rng: np.random.Generator) -> dict:
        """B_k: levers pinned per particle under the two-step protocol, the rest completed by the model."""
        acts = dec["actors"]
        rows = alloc["rows"]
        m = len(rows)
        state = MA.take_state(nd.cloud.state, rows)
        a0x = {k: np.asarray(v)[rows] for k, v in a0.x.items()}
        pins: dict = {}
        shifts: dict = {}
        oil_a, oil_b2 = np.zeros(m), np.zeros(m)
        gpr_a, gpr_b2 = np.zeros(m), np.zeros(m)
        protocols: set = set()
        written: dict = {}
        flags: list = []
        point_writes = {}
        for aid in dec["order"]:
            ab = acts[aid]
            code = alloc["assign"][aid]
            for k, o in enumerate(ab["options"]):
                sel = np.flatnonzero(code == k)
                if not len(sel):
                    continue
                if self.neutral and (o.get("bin") or o.get("status_quo")):
                    continue                              # the model tree: A0 draws (bins) and masked levers
                pay = self._opt_payload(aid, o, nd)
                if pay.get("protocol"):
                    protocols.add(pay["protocol"])
                flags += pay.get("flags") or []
                for w in pay.get("writes", []):
                    if int(w.get("dq", 0)) > 0 or w["kind"] == "label":
                        continue
                    fid = w["field"]
                    if fid not in self.model.x_fields:
                        continue
                    if w["kind"] == "bin":                # A0 particles inside the option's own bin, resampled
                        lo = -math.inf if w.get("lo") is None else float(w["lo"])
                        hi = math.inf if w.get("hi") is None else float(w["hi"])
                        base = np.asarray(a0x[fid], float)
                        pool = base[(base > lo) & (base <= hi)]
                        arr = pins.setdefault(fid, base.copy())
                        arr[sel] = rng.choice(pool, len(sel)) if len(pool) else np.clip(base[sel], lo, hi)
                        written.setdefault(fid, []).append(float(np.mean(arr[sel])))
                        point_writes.setdefault(fid, set()).add(aid)
                        continue
                    written.setdefault(fid, []).append(float(w.get("a") or 0.0))
                    if w["kind"] == "point":
                        arr = pins.setdefault(fid, np.array(a0x[fid], float))
                        arr[sel] = float(w["a"])
                        point_writes.setdefault(fid, set()).add(aid)
                    elif w["kind"] == "normal":
                        arr = pins.setdefault(fid, np.array(a0x[fid], float))
                        arr[sel] = float(w["a"]) + float(w.get("b") or 0.0) * rng.standard_normal(len(sel))
                    else:
                        arr = shifts.setdefault(fid, np.zeros(m))
                        arr[sel] += float(w.get("a") or 0.0) + float(w.get("b") or 0.0) * rng.standard_normal(len(sel))
                for e in pay.get("events", {}).get("oil_z", []):
                    oil_a[sel] += float(e["a"])
                    oil_b2[sel] += float(e["b"]) ** 2
                for e in pay.get("events", {}).get("gpr", []):
                    gpr_a[sel] += float(e["a_sd"])
                    gpr_b2[sel] += float(e["b_sd"]) ** 2
        # Fed within-branch slopes (§6.6) on its point write, then the floor at the lower bound
        f_ff = self.resolver.field("policy_rate_chg")
        if f_ff in pins and "fed" in point_writes.get(f_ff, set()):
            corr = np.zeros(m)
            sl = P.FED_SLOPES
            zs = {"us_equity_logret": state.get("us_equity_l1"), "d_unrate": state.get("d_unrate_l1"),
                  "core_gap": None if state.get("s_core_yoy") is None else state["s_core_yoy"] - 2.0,
                  "d_ff_lag": state.get("d_ff_l1"), "ff_level": state.get(FED_FLOOR_FIELD)}
            for k, v in zs.items():
                if v is None or k not in sl:
                    continue
                v = np.asarray(v, float)
                corr += float(sl[k]["b"]) * (v - float(np.median(v)))
            pins[f_ff] = pins[f_ff] + np.clip(corr, -sl["clip_pp"], sl["clip_pp"])
            pins[f_ff] = np.maximum(pins[f_ff], -np.asarray(state[FED_FLOOR_FIELD], float))
        # the event channels (§4.4): oil-supply composite and GPR contributions, as shifts on A0
        if np.any(oil_a != 0):
            beta = A.OIL_COMPOSITE["beta_metals_oil_per_z"]
            fid = self.resolver.field("oil_supply_signature")
            if fid in self.model.x_fields:
                shifts[fid] = shifts.get(fid, np.zeros(m)) + beta * oil_a + abs(beta) * np.sqrt(oil_b2) * rng.standard_normal(m)
                written.setdefault(fid, []).append(float(np.mean(beta * oil_a)))
        if np.any(gpr_a != 0):
            fid = self.resolver.field("gpr_chg")
            if fid in self.model.x_fields:
                sd = self.sd.get(fid, 22.15)
                shifts[fid] = shifts.get(fid, np.zeros(m)) + gpr_a * sd + np.sqrt(gpr_b2) * sd * rng.standard_normal(m)
                written.setdefault(fid, []).append(float(np.mean(gpr_a * sd)))
        for fid, d in shifts.items():
            base = pins.get(fid, a0x[fid])
            v = np.asarray(base, float) + d
            h = self.support.get(fid) or {}
            if h.get("min") is not None and h.get("max") is not None:
                v = np.clip(v, float(h["min"]), float(h["max"]))
            pins[fid] = v
        # response tables (§6.3): transfers -> goods consumption, R = +2.1 per pp of personal income
        f_tr, f_cg = self.resolver.field("transfers_chg"), "d_cons_goods"
        if f_tr in shifts and f_cg in self.model.x_fields and f_cg not in pins:
            pins[f_cg] = np.asarray(a0x[f_cg], float) + 2.1 * shifts[f_tr]
        # the two-step protocol: lag-protected E0 at A0 for lagged / thin levers (unless the model is structural)
        structural = "structural" in getattr(self.model, "capabilities", ())
        if protocols & {"lagged", "thin"} and not structural:
            for fid in SY.LAG_PROTECTED:
                if fid in self.model.x_fields and fid not in pins:
                    pins[fid] = np.asarray(a0x[fid], float)
        # nature: the hazard event fields stay at their A0 draws unless an action wrote them
        for h, (fid, _, _) in SY.HAZARD_REGION.items():
            if fid in self.model.x_fields and fid not in pins and fid != "d_gz":
                pins[fid] = np.asarray(a0x[fid], float)
        if self.neutral and not pins.keys() - {f for _, (f, _, _) in SY.HAZARD_REGION.items()}:
            draws = a0.take(rows)
            given = MA.Given.from_dict({}, m)
        else:
            given = MA.Given.from_dict(pins, m)
            draws = self.model.complete(state, given, rng)
        wr = {f: float(np.mean(v)) for f, v in written.items()}
        return {"rows": rows, "draws": draws, "given": given, "written": sorted(wr), "write_means": wr,
                "protocols": sorted(protocols), "flags": sorted(set(flags)), "state": state}

    # ---------------------------------------------------------------- the cell
    def expand(self, nd: Node) -> tuple[list[Node], dict]:
        """One quarter for one node: (children with clouds, the node's detail)."""
        q = A.quarter_add(nd.quarter, 1)
        cl0 = nd.cloud
        if cl0.acc is not None and "summary" not in cl0.acc.__dict__:
            memo = cl0.acc.summary()
            cl0.acc.summary = lambda memo=memo: memo        # the node's books do not change while it expands
        calls: list = []
        report: list = []
        gates = AC.gates(cl0.acc) if cl0.acc is not None else {}
        rng_a0 = seed_for(self.st.seed, nd.key, "a0")
        a0 = self.model.complete(cl0.state, MA.Given.empty(cl0.n), rng_a0)
        dec = self.decide(nd, q, a0, gates, calls, report)
        clusters = self.clusters(nd, dec)
        nature_branch = {}
        llm = self._llm_for(nd.level)
        synth_on = self._synth_on(nd)
        bks = {}
        kept_clusters = []
        for cl in clusters:
            rng = seed_for(self.st.seed, nd.key, f"alloc|{cl['head_key']}")
            alloc = self.allocate_cluster(nd, cl, dec, a0, rng)
            if self.neutral and len(alloc["rows"]) != cl0.n:
                # bins of A0 partition the particles: a cluster's mass is its particles' share (joint, not the
                # product of the actors' marginal bin shares)
                cnt = np.bincount(alloc["member"], minlength=len(cl["members"])) if len(alloc["rows"]) else \
                    np.zeros(len(cl["members"]))
                for mm, c in zip(cl["members"], cnt):
                    mm["p"] = float(c) / cl0.n
                cl["p"] = float(len(alloc["rows"])) / cl0.n
                if not len(alloc["rows"]):
                    continue
            kept_clusters.append(cl)
            bk = self.build_bk(nd, cl, dec, a0, alloc, seed_for(self.st.seed, nd.key, f"bk|{cl['head_key']}"))
            bk["alloc"] = alloc
            bk = self.apply_nature(nd, cl, bk, dec, nature_branch)
            bk["split"] = self.discrete_split(nd, dec, bk["alloc"])
            bks[cl["id"]] = bk
        clusters = kept_clusters
        synth_answers, synth_calls = {}, []
        gated = {}
        for cl in clusters:
            gated[cl["id"]] = self.gate(nd, cl, bks[cl["id"]], dec, nature_branch.get(cl["id"]))
        to_synth = [cl for cl in clusters if synth_on and llm is not None and gated[cl["id"]]
                    and (float(self.st.synth.get("c_s", SY.C_S)) > 0 or float(self.st.synth.get("c_m", SY.C_M)) > 0)]
        if to_synth:
            req = self.synth_request(nd, q, to_synth, bks, dec)
            res = llm.synth(req)
            synth_calls = [c.to_dict() for c in res.calls]
            calls += synth_calls
            report += res.report
            synth_answers = res.clusters
        children: list[Node] = []
        consequences = []
        for cl in clusters:
            bk = bks[cl["id"]]
            ans = synth_answers.get(cl["id"])
            branches, rep = self.branches(nd, cl, bk, ans)
            report += rep
            status = "called" if ans is not None and ans.status == "ok" else (
                "failed" if ans is not None else ("gated_off" if not gated[cl["id"]] or not synth_on else "no_llm"))
            consequences.append({"cluster": cl["id"], "synth_status": status, "gates": gated[cl["id"]],
                                 "branches": [_branch_view(b) for b in branches], "report": rep})
            for b in branches:
                children += self.realize(nd, q, cl, bk, b, a0, dec)
        tot = sum(c.p_cond for c in children) or 1.0
        for c in children:
            c.p_cond = c.p_cond / tot
            c.p_path = nd.p_path * c.p_cond
            c.own_mass = c.p_path
        nd.status = "expanded"
        if "budget" in dec["flags"] or any(dec["actors"][a].get("fallback") for a in dec["order"]):
            nd.flags = sorted(set(nd.flags) | {"prior_only"})
        if any(c["synth_status"] == "failed" for c in consequences):
            nd.flags = sorted(set(nd.flags) | {"synth_default"})
        if "intervened" in dec["flags"]:
            nd.flags = sorted(set(nd.flags) | {"intervened"})
        detail = self.detail(nd, q, dec, clusters, bks, consequences, calls, report, a0)
        return children, detail

    def _synth_on(self, nd: Node) -> bool:
        sy = self.st.synth or {}
        mode = sy.get("mode", "gated")
        if mode == "off" or self.kind != "agent":
            return False
        if mode == "every":
            return True
        return nd.level == 0 or nd.level < int(sy.get("depth", 4))

    def apply_nature(self, nd: Node, cl: dict, bk: dict, dec: dict, nature_branch: dict) -> dict:
        """Nature partitions B_k's own tails (§6.5): a particle mixture below 5% (stress mode off), else a branch."""
        x = bk["draws"].x
        m = len(bk["rows"])
        w = np.ones(m)
        best = None
        for h, v in sorted(dec["hazards"].items()):
            if v["m"] <= 0:
                continue
            mask = SY.hazard_mask(x, h)
            mk = float(np.mean(mask))
            if mk <= 0 or mk >= 1:
                continue
            ph = float(np.clip(v["p"] / max(v["m"], 1e-12) * mk, 0.0, 0.95))
            if (ph >= A.HAZARD_BRANCH_MIN or self.st.stress_mode) and not self.neutral:   # agent and base-rate trees alike
                if best is None or ph > best[1]:
                    best = (h, ph, mask)
                continue
            if abs(ph - mk) > 1e-12:
                w = w * np.where(mask, ph / mk, (1 - ph) / (1 - mk))
        if not np.allclose(w, 1.0):
            rng = seed_for(self.st.seed, nd.key, f"nature|{cl['head_key']}")
            idx = resample_rows(np.arange(m), m, rng, w)
            bk = {**bk, "rows": bk["rows"][idx], "draws": bk["draws"].take(idx), "given": bk["given"].take(idx),
                  "state": MA.take_state(bk["state"], idx),
                  "alloc": {**bk["alloc"], "assign": {k: v[idx] for k, v in bk["alloc"]["assign"].items()},
                            "attr": {k: v[idx] for k, v in bk["alloc"]["attr"].items()}}}
        if best is not None:
            h, ph, mask = best
            nature_branch[cl["id"]] = {"hazard": h, "p": ph, "mask": mask}
            bk["nature"] = {"hazard": h, "p": ph}
        return bk

    def gate(self, nd: Node, cl: dict, bk: dict, dec: dict, nature: dict | None) -> list[str]:
        support = [f for f in bk["flags"] if "罕见" in f or "超出" in f]
        thin = []
        for fid in bk["written"]:
            if (self.meta.get(fid) or {}).get("channel_class") in ("thin", "absent"):
                thin.append(fid)
        for f in ("export_controls_level", "fed_removal_attempt", "fed_removal_allowed", "fed_pressure_public"):
            if any(fa["id"] == f for aid in dec["order"] for o in dec["actors"][aid]["options"]
                   for fa in self._opt_payload(aid, o, nd).get("facts", []) if o["p_used"] >= 0.5):
                thin.append(f)
        chi, k = SY.lever_rarity({f: v for f, v in bk["write_means"].items()}, self.sd)
        regime = SY.regime_lines(self.covariates(nd), nd.facts, nd.stress)
        facts = [f for f, v in nd.facts.items() if v and f in A.FACTS and A.FACTS[f].get("exit") is not None]
        g = SY.gates(root=nd.level == 0, support_flags=support, thin_levers=thin, rarity=chi if k else None,
                     rarity_k=k, regime=regime, facts=facts, nature_branched=nature is not None)
        if (self.st.synth or {}).get("mode") == "every":
            g = g or ["G5"]
        return g

    def synth_request(self, nd: Node, q: str, clusters: list, bks: dict, dec: dict):
        from .llm_io import ClusterSpec, SynthRequest
        ctx_p = self.prompt_ctx(nd, q, dec)
        specs = []
        all_axes = set()
        for cl in clusters:
            bk = bks[cl["id"]]
            x = bk["draws"].x
            axes = self.axes_for(bk, dec)
            all_axes |= set(axes)
            summ = []
            for fid in sorted({f for _, f in axes} | {"d_unrate", "d_rgdp", "d_core", "d_cons_serv", "d_sent",
                                                      "d_infexp1", "d_gz"}):
                if fid not in x:
                    continue
                v = np.asarray(x[fid])
                summ.append({"field": fid, "zh": (self.meta.get(fid) or {}).get("zh", fid),
                             "unit": (self.meta.get(fid) or {}).get("unit", ""), "median": rnd(np.median(v), 3),
                             "p10": rnd(np.percentile(v, 10), 3), "p90": rnd(np.percentile(v, 90), 3),
                             "sd": rnd(np.std(v), 3)})
            ms = SY.cluster_masses(x, axes, panic=self.panic_ok(nd, bk))
            if "mult" in {f for _, f in axes}:
                ms["mult"] = SY.masses(bk["draws"].y["mult"], SY.regions_for("equity_leg"))
            for v in ms.values():
                for r in v:
                    r["mass"] = rnd(r["mass"], 3)
            acts = []
            for aid in dec["order"]:
                ab = dec["actors"][aid]
                code = bk["alloc"]["assign"][aid]
                k = int(np.bincount(code, minlength=len(ab["options"])).argmax()) if len(code) else 0
                o = ab["options"][k]
                acts.append({"actor": aid, "actor_zh": ab["zh"], "option": o["id"], "label": o["label"],
                             "writes": _writes_text(self._opt_payload(aid, o, nd))})
            stress = float(np.mean(SY.stress_flags(x, bk["draws"].y)))
            regime = SY.regime_lines(self.covariates(nd), nd.facts, nd.stress)
            facts = [A.FACTS.get(f, {}).get("zh", f) for f, v in nd.facts.items() if v]
            specs.append(ClusterSpec(id=cl["id"], actions=acts, written_fields=bk["written"],
                                     support=[f for f in bk["flags"] if "罕见" in f or "超出" in f][:4], summary=summ,
                                     masses=ms, stress_share=rnd(stress, 3), regime=regime, facts=facts[:4]))
        fields = SY.field_table(self.meta, sorted(all_axes, key=lambda t: t[0]))
        sector = AC.sector_summary(nd.cloud.acc, self.root_acc) if nd.cloud.acc is not None else ""
        events = [f"{v['zh']}（本季概率 {v['p']:.2f}）" for h, v in sorted(dec["hazards"].items())
                  if any((bks[c["id"]].get("nature") or {}).get("hazard") == h for c in clusters)]
        return SynthRequest(ctx=ctx_p, clusters=specs, fields=fields, channels=SY.channel_table(),
                            anchors=SY.DIRECTIONS, signatures=SY.SIGNATURES, sector_summary=sector,
                            nature_events=events)

    def panic_ok(self, nd: Node, bk: dict) -> bool:
        return bool(self.st.stress_mode or nd.facts.get("financial_accident") or
                    (bk.get("nature") or {}).get("hazard") == "financial_accident")

    def axes_for(self, bk: dict, dec: dict) -> list[tuple[str, str]]:
        bt = "bigtech" in dec["actors"] and any("d_itinv" == f for f in bk["written"])
        avail = list(bk["draws"].x) + (["mult"] if "mult" in bk["draws"].y else [])
        return SY.axis_list(avail, (self.st.synth or {}).get("axes"), written=bk["written"], bigtech_wrote_itinv=bt)

    def branches(self, nd: Node, cl: dict, bk: dict, ans) -> tuple[list, list]:
        """The cluster's consequence branches with P(branch | cluster): the synthesizer's, else the default."""
        x, y = bk["draws"].x, bk["draws"].y
        nature = bk.get("nature")
        rep: list = []
        out: list = []
        p_rest = 1.0
        if nature is not None:
            out.append({"id": f"nature_{nature['hazard']}", "kind": "nature_event", "hazard": nature["hazard"],
                        "label": A.HAZARDS.get(nature["hazard"], {}).get("zh", nature["hazard"]), "p": nature["p"],
                        "m": nature["p"], "p_llm": None, "memory": ""})
            p_rest = 1.0 - nature["p"]
            mask = ~SY.hazard_mask(x, nature["hazard"])
            sub = np.flatnonzero(mask)
            xs = {k: np.asarray(v)[sub] for k, v in x.items()}
            ys = {k: np.asarray(v)[sub] for k, v in y.items()}
        else:
            xs, ys = x, y
        if ans is not None and getattr(ans, "status", "") == "ok" and ans.branches:
            sy = self.st.synth or {}
            causes = [f"{a}.{o}" for a, o in cl["head"].items()]
            classes = [(self.meta.get(f) or {}).get("channel_class") or "moderate" for f in bk["written"]]
            if any("罕见" in f or "超出" in f for f in bk["flags"]):
                classes.append("thin")
            if any(nd.facts.get(f) for f in ("export_controls_level", "fed_removal_allowed", "fed_removal_attempt")):
                classes.append("absent")
            regime = bool(SY.regime_lines(self.covariates(nd), nd.facts, nd.stress)) or bool(
                [f for f, v in nd.facts.items() if v and f in A.FACTS and A.FACTS[f].get("exit") is not None])
            br, r2 = SY.consequences(ans, xs, written=bk["written"], causes=causes, classes=classes, regime=regime,
                                     axes=self.axes_for(bk, {"actors": {}}), c_s=float(sy.get("c_s", SY.C_S)),
                                     c_m=float(sy.get("c_m", SY.C_M)), stress_mode=bool(self.st.stress_mode),
                                     panic=self.panic_ok(nd, bk), m_max=int(self.st.M_max))
            rep += [f"{cl['id']}：{line}" for line in r2]
        else:
            br = SY.default_partition(xs, ys, nd.p_path * cl["p"] * p_rest)
        for b in br:
            b["p"] = float(b["p"]) * p_rest
            out.append(b)
        return out, rep

    # ---------------------------------------------------------------- children
    # ---------------------------------------------------------------- discrete state (§8.1; review fix)
    def _commitments_after(self, aid: str, o: Mapping, nd: Node, q: str, pay: Mapping) -> list:
        later = [cm for cm in nd.commitments if cm.get("actor") == aid and cm.get("due_quarter")
                 and A.quarters_between(q, cm["due_quarter"]) > 0]
        if o["id"] == "as_committed":
            out = later
        else:
            out = [cm for cm in later if not cm.get("cancellable", True)]
            out = out + (A.commitments_of(pay, q) if o["id"] != "custom" else [])
        return sorted([str(cm.get("option")), str(cm.get("due_quarter")), str((cm.get("write") or {}).get("field")),
                       rnd((cm.get("write") or {}).get("a"), 4)] for cm in out)

    def discrete_effect(self, aid: str, o: Mapping, nd: Node, q: str, *, main: bool = True) -> dict:
        """What an option changes in the node's discrete state: facts set, facts cleared, political state and (main
        domains) the actor's commitments afterwards. Options with equal effects form one discrete class."""
        pay = self._opt_payload(aid, o, nd) if main else self.payload(aid, o["id"])
        eff = {"facts": sorted([f["id"], str(f.get("value", True))] for f in pay.get("facts") or []),
               "clears": sorted(pay.get("clears") or []),
               "political": {k: str(v) for k, v in sorted((pay.get("political") or {}).items())}}
        if main:
            eff["commitments"] = self._commitments_after(aid, o, nd, q, pay)
        return eff

    @staticmethod
    def _effect_distance(a: Mapping, b: Mapping) -> int:
        d = len({tuple(x) for x in a["facts"]} ^ {tuple(x) for x in b["facts"]})
        d += len(set(a["clears"]) ^ set(b["clears"]))
        d += sum(1 for k in set(a["political"]) | set(b["political"]) if a["political"].get(k) != b["political"].get(k))
        d += int(a.get("commitments") != b.get("commitments"))
        return d

    def split_max(self) -> int:
        return split_max_of(self.st)

    def discrete_split(self, nd: Node, dec: dict, alloc: Mapping) -> dict:
        """The discrete columns of a cluster's particles (every actor's main domain and attribute domains: the class
        of each particle's option) and the columns its consequence branches are split by. Non-branching actors'
        options are allocated per particle, so one child's particles carry several discrete states; a node can
        carry one. Columns are split in order of the mass a single state would misstate (classes below p_min fold
        into the nearest kept class), while the product of kept classes stays within split_max; the rest keep the
        child's modal class, and that mass is reported (summary.discrete)."""
        q = A.quarter_add(nd.quarter, 1)
        acts = dec["actors"]
        cols = []
        for aid in dec["order"]:
            ab = acts[aid]
            code = (alloc.get("assign") or {}).get(aid)
            if code is None:
                continue
            doms = [(None, ab["options"], code, True)]
            for dd in ab["attributes"]:
                k = (aid, dd["domain"])
                if k in (alloc.get("attr") or {}):
                    doms.append((dd["domain"], dd["options"], alloc["attr"][k], False))
            for dom, options, codes, main in doms:
                effs = [self.discrete_effect(aid, o, nd, q, main=main) for o in options]
                keys = [canonical(e) for e in effs]
                ids: dict = {}
                opt_class = np.array([ids.setdefault(k, len(ids)) for k in keys], int)
                eff_of = [effs[keys.index(k)] for k in ids]
                codes = np.asarray(codes, int)
                cls = opt_class[codes] if len(codes) else np.zeros(0, int)
                share = np.bincount(cls, minlength=len(ids)).astype(float) / max(len(cls), 1)
                rep = []
                for c in range(len(ids)):
                    members = np.flatnonzero(opt_class == c)
                    cnt = np.array([np.sum(codes == k) for k in members])
                    rep.append(int(members[int(np.argmax(cnt))]) if len(members) else 0)
                kept = sorted({int(np.argmax(share))} | {c for c in range(len(ids)) if share[c] >= float(self.st.p_min)})
                fold = np.arange(len(ids))
                for c in range(len(ids)):
                    if c not in kept:
                        fold[c] = min(kept, key=lambda k, c=c: (self._effect_distance(eff_of[c], eff_of[k]), -share[k], k))
                gain = float(sum(share[c] for c in kept) - share.max())
                weight = SPLIT_WEIGHT["political"] if main and any(e["political"] for e in eff_of) else \
                    SPLIT_WEIGHT["main" if main else "attribute"]
                cols.append({"actor": aid, "domain": dom, "main": main, "opt_class": opt_class, "eff": eff_of,
                             "weight": weight,
                             "share": share, "rep": rep, "kept": kept, "fold": fold, "gain": gain,
                             "options": [o["id"] for o in options], "labels": [o.get("label") or o["id"] for o in options]})
        budget = self.split_max()
        order = sorted(range(len(cols)), key=lambda i: (-cols[i]["gain"] * cols[i]["weight"],
                                                         A.STAGE_ORDER.index(cols[i]["actor"])
                                                         if cols[i]["actor"] in A.STAGE_ORDER else 99, i))
        chosen, prod = [], 1
        for i in order:
            k = len(cols[i]["kept"])
            if k < 2 or cols[i]["gain"] < float(self.st.p_min) or prod * k > budget:
                continue
            chosen.append(i)
            prod *= k
        return {"cols": cols, "chosen": sorted(chosen), "budget": budget}

    def _cells(self, split: Mapping, assign: Mapping, attr: Mapping, n: int) -> list:
        """[(particle indices, {column: class})] of one branch's particles by the chosen columns' folded classes."""
        chosen = list(split.get("chosen") or [])
        if not chosen:
            return [(np.arange(n), {})]
        codes = []
        for i in chosen:
            col = split["cols"][i]
            raw = assign[col["actor"]] if col["main"] else attr[(col["actor"], col["domain"])]
            codes.append(col["fold"][col["opt_class"][np.asarray(raw, int)]])
        X = np.stack(codes, axis=1)
        uk, inv = np.unique(X, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        cnt = np.bincount(inv, minlength=len(uk))
        # cells too small to stand alone join the nearest larger cell (fewest differing columns, weighted)
        floor = max(SPLIT_CELL_MIN_PARTICLES, int(math.ceil(SPLIT_CELL_MIN_SHARE * n)))
        big = [g for g in range(len(uk)) if cnt[g] >= floor] or [int(np.argmax(cnt))]
        w = np.array([split["cols"][i]["weight"] for i in chosen], float)
        into = np.arange(len(uk))
        for g in range(len(uk)):
            if g not in big:
                into[g] = min(big, key=lambda b, g=g: (float(np.sum(w * (uk[g] != uk[b]))), -int(cnt[b]), b))
        out = []
        for g in big:
            rows = np.flatnonzero(into[inv] == g)
            out.append((rows, {i: int(c) for i, c in zip(chosen, uk[g])}))
        return out

    # ---------------------------------------------------------------- children
    def realize(self, nd: Node, q: str, cl: dict, bk: dict, b: dict, a0, dec: dict) -> list[Node]:
        """The children of one consequence branch: its particles of B_k (their natural count: reduction resamples
        the survivors to N), split into cells by discrete state when the cluster's split says so (§8.1)."""
        draws, rows_all = bk["draws"], bk["rows"]
        key0 = child_key(nd.key, cl["head_key"], b["id"])
        rng = seed_for(self.st.seed, key0, "realize")
        m = len(rows_all)
        if b["kind"] == "nature_event":
            sel = np.flatnonzero(SY.hazard_mask(draws.x, b["hazard"]))
        else:
            base = np.arange(m)
            if bk.get("nature") is not None:
                base = np.flatnonzero(~SY.hazard_mask(draws.x, bk["nature"]["hazard"]))
            sub_x = {k: np.asarray(v)[base] for k, v in draws.x.items()}
            sub_y = {k: np.asarray(v)[base] for k, v in draws.y.items()}
            sel = base[SY.branch_rows(b, sub_x, sub_y)] if b["kind"] != "mechanism" else base
        if len(sel) == 0 or b["p"] <= 0:
            return []
        idx = np.asarray(sel, int)
        d = draws.take(idx)
        if b["kind"] == "mechanism":
            pins = bk["given"].take(idx).to_dict()
            pins.update(SY.mechanism_pins(b, d.x, rng))
            d = self.model.complete(MA.take_state(bk["state"], idx), MA.Given.from_dict(pins, len(idx)), rng)
        src = rows_all[idx]
        assign = {k: v[idx] for k, v in bk["alloc"]["assign"].items()}
        attr = {k: v[idx] for k, v in bk["alloc"]["attr"].items()}
        split = bk.get("split") or {"cols": [], "chosen": []}
        cells = self._cells(split, assign, attr, len(idx))
        out = []
        for rows, cell in cells:
            if len(rows) == 0:
                continue
            sig = {f"{split['cols'][i]['actor']}:{split['cols'][i]['domain'] or 'main'}":
                   split["cols"][i]["options"][split["cols"][i]["rep"][c]] for i, c in cell.items()}
            key = key0 if len(cells) == 1 else child_key(nd.key, cl["head_key"], f"{b['id']}|{canonical(sig)}")
            a_sub = {k: v[rows] for k, v in assign.items()}
            t_sub = {k: v[rows] for k, v in attr.items()}
            cloud = self.advance(nd, d.take(rows), src[rows], a_sub, t_sub, dec)
            share = float(len(rows)) / float(len(idx))
            ch = Node(key=key, parent=nd.key, level=nd.level + 1, quarter=q,
                      p_cond=float(cl["p"]) * float(b["p"]) * share, cloud=cloud,
                      lineage=nd.lineage + [[nd.key, cl["id"], b["id"]] + ([canonical(sig)] if len(cells) > 1 else [])])
            self.child_state(nd, ch, q, cl, b, dec, a_sub, t_sub, split=split, cell=cell, cell_share=share)
            ch.summary = self.summarize(ch, nd, a0=a0, bk=bk, branch=b)
            ch.summary["discrete"] = ch.cluster.pop("_discrete", None)
            out.append(ch)
        return out

    def account_columns(self, nd: Node, dec: dict, assign: dict, attr: dict) -> list:
        """(option code per particle, [account actions per option]) for every actor domain whose options carry
        account actions, in the order actors.account_actions merges a profile (main domain, then attributes)."""
        cols = []
        for aid in dec["order"]:
            ab = dec["actors"][aid]
            acc_of = [self._opt_payload(aid, o, nd).get("account") or {} for o in ab["options"]]
            if any(acc_of) and aid in assign:
                cols.append((np.asarray(assign[aid], int), acc_of))
            for dd in ab["attributes"]:
                k = (aid, dd["domain"])
                if k not in attr:
                    continue
                acc_d = [self.payload(aid, o["id"]).get("account") or {} for o in dd["options"]]
                if any(acc_d):
                    cols.append((np.asarray(attr[k], int), acc_d))
        return cols

    def advance(self, nd: Node, d: MA.Draws, src: np.ndarray, assign: dict, attr: dict, dec: dict) -> Cloud:
        """The next start-of-quarter cloud: the model's state roll and the books, per particle. Every particle's
        books are stepped with its own options' account actions (accounts.per_particle_actions), in one step."""
        parent = nd.cloud
        n = len(src)
        state = self.model.update_state(MA.take_state(parent.state, src), d)
        acc = None
        if parent.acc is not None:
            acc = parent.acc.take(src)
            acc.step(d.x, {a: d.y[a] for a in AC.ACCOUNT_ASSETS if a in d.y},
                     AC.per_particle_actions(self.account_columns(nd, dec, assign, attr), n))
            acc.check()
            acc.quarter = A.quarter_add(nd.quarter, 1)
        cum = {a: parent.cum[a][src] + d.y[a] for a in self.assets}
        cpi = (parent.cpi[src] if parent.cpi is not None else np.zeros(n)) + (d.cpi_q if d.cpi_q is not None else 0.0)
        path = None
        if parent.path is not None:
            q_ret = np.stack([np.expm1(d.y[a]) for a in self.assets], axis=1).astype(np.float32)
            path = np.concatenate([parent.path[src], q_ret[:, None, :]], axis=1)
        return Cloud(state, acc, dict(d.x), {a: d.y[a] for a in d.y}, cum, cpi, path, dict(d.inc))

    def _applied(self, col: Mapping, codes: np.ndarray, cell: Mapping, ci: int) -> tuple[int | None, int, float]:
        """(class applied to the child or None, realised option index, its share in the child) for one column:
        the cell's class when the column is split, else the child's modal class (attributes: only with ≥ 50%)."""
        codes = np.asarray(codes, int)
        cls = col["opt_class"][codes] if len(codes) else np.zeros(0, int)
        if ci in cell:
            c = int(cell[ci])
        else:
            cnt = np.bincount(cls, minlength=len(col["eff"])) if len(cls) else np.zeros(len(col["eff"]))
            c = int(np.argmax(cnt))
            if not col["main"] and cnt[c] / max(len(cls), 1) < 0.5:
                c = None
        if c is None:
            k = int(np.bincount(codes, minlength=len(col["options"])).argmax()) if len(codes) else 0
            return None, k, float(np.mean(codes == k)) if len(codes) else 0.0
        mine = codes[cls == c] if len(cls) else codes
        if len(mine):
            k = int(np.bincount(mine, minlength=len(col["options"])).argmax())
        else:
            k = int(col["rep"][c])
        return c, k, float(np.mean(codes == k)) if len(codes) else 0.0

    def child_state(self, nd: Node, ch: Node, q: str, cl: dict, b: dict, dec: dict, assign: dict, attr: dict, *,
                    split: Mapping | None = None, cell: Mapping | None = None, cell_share: float = 1.0) -> None:
        """Facts, political state, commitments, realised signals, credibility, memory and the next digest. Discrete
        state per column (actor main domain, attribute domain): the cell's class where the branch was split, the
        child's modal class elsewhere; what that misstates is recorded in ch.cluster["_discrete"]."""
        rng = seed_for(self.st.seed, ch.key, "facts")
        facts = {}
        bell_on = "belligerents" in dec["actors"]
        for f, v in sorted(nd.facts.items()):
            ex = A.FACTS.get(f, {}).get("exit")
            if f == "oil_disruption_ongoing" and bell_on:
                ex = None
            if ex is not None and not A.FACTS.get(f, {}).get("src") and rng.random() < float(ex):
                continue
            facts[f] = v
        for f in [k for k in facts if k.endswith("_expiry_q")]:      # a dated fact ends after its expiry quarter
            try:
                over = A.quarters_between(str(facts[f]), q) > 0
            except ValueError:
                over = False
            if over:
                facts.pop(f, None)
                facts.pop(f[:-len("_expiry_q")], None)
        political = dict(nd.political)
        commitments = [cm for cm in nd.commitments if cm.get("due_quarter") and A.quarters_between(q, cm["due_quarter"]) > 0]
        split = split or self.discrete_split(nd, dec, {"assign": assign, "attr": attr})
        cell = dict(cell or {})
        cols = {(c["actor"], c["domain"]): (i, c) for i, c in enumerate(split["cols"])}
        modal = {}
        signals = []
        labels = []
        misstated = []
        split_labels = []
        for aid in dec["order"]:
            ab = dec["actors"][aid]
            code = np.asarray(assign[aid], int)
            ci, col = cols.get((aid, None), (None, None))
            if col is not None:
                c, k, sh = self._applied(col, code, cell, ci)
                own = col["opt_class"][code] if len(code) else np.zeros(0, int)
                off = float(np.mean(own != c)) if len(own) else 0.0
                if off > 1e-9:
                    misstated.append({"actor": aid, "domain": None, "share": rnd(off, 4)})
                if ci in cell:
                    split_labels.append(f"{ab['zh']}：{ab['options'][k]['label']}")
            else:
                cnt = np.bincount(code, minlength=len(ab["options"]))
                k = int(cnt.argmax())
                sh = float(cnt[k]) / max(len(code), 1)
            o = ab["options"][k]
            share = sh
            modal[aid] = o["id"]
            pay = self._opt_payload(aid, o, nd)
            for f in pay.get("clears") or []:
                facts.pop(f, None)
            for fa in pay.get("facts") or []:
                facts[fa["id"]] = fa.get("value", True)
            political.update(pay.get("political") or {})
            if o["id"] == "as_committed":
                pass                       # the due commitments were executed (and dropped above); later ones stay
            else:
                commitments = [cm for cm in commitments if cm.get("actor") != aid or not cm.get("cancellable", True)]
                commitments += A.commitments_of(pay, q) if o["id"] not in ("custom",) else []
            for dd in ab["attributes"]:
                kk = (aid, dd["domain"])
                if kk not in attr:
                    continue
                ci, col = cols.get(kk, (None, None))
                if col is None:
                    continue
                codes_d = np.asarray(attr[kk], int)
                c, k2, _ = self._applied(col, codes_d, cell, ci)
                own = col["opt_class"][codes_d] if len(codes_d) else np.zeros(0, int)
                eff_none = c is None
                off = float(np.mean([(eff_none and any(col["eff"][x].values())) or (not eff_none and x != c)
                                     for x in own])) if len(own) else 0.0
                if off > 1e-9 and any(any(e.values()) for e in col["eff"]):
                    misstated.append({"actor": aid, "domain": dd["domain"], "share": rnd(off, 4)})
                if c is None:
                    continue
                oo = dd["options"][k2]
                if ci in cell:
                    split_labels.append(f"{ab['zh']}·{dd['zh']}：{oo['label']}")
                elif oo["p"] >= 1:
                    continue
                pa = self.payload(aid, oo["id"])
                for fa in pa.get("facts") or []:
                    facts[fa["id"]] = fa.get("value", True)
                for f in pa.get("clears") or []:
                    facts.pop(f, None)
                political.update(pa.get("political") or {})
            sig = o.get("signal") if isinstance(o.get("signal"), dict) and o.get("signal", {}).get("text") else None
            if sig is None and o["id"] not in ("as_committed", "custom"):
                try:
                    sig = A.default_signal(aid, o["id"], resolver=self.resolver)
                except KeyError:
                    sig = None
            if sig:
                sig = {**A.validate_signal(sig)[0], "actor": aid, "quarter": q, "option": o["id"]}
                signals.append(sig)
            labels.append(f"{ab['zh']}：{o['label']}" + ("" if share > 0.99 else f"（{share:.0%}）"))
        if b["kind"] == "nature_event":
            fa = A.hazard_payload(b["hazard"])["facts"][0]
            facts[fa["id"]] = True
        for f in b.get("facts") or []:
            if isinstance(f, dict) and f.get("text"):
                facts[f"synth:{f['text'][:20]}"] = True
        if ch.cloud.acc is not None:
            for f, v in ch.cloud.acc.facts(0.5).items():
                if v:
                    facts[f] = True
                else:
                    facts.pop(f, None)
        cred = {k: dict(v) for k, v in (nd.cred or {}).items()}
        med = {k: float(np.median(v)) for k, v in ch.cloud.x.items()}
        med_y = {k: float(np.median(v)) for k, v in ch.cloud.y.items()}
        for s in nd.signals:
            due = A.quarter_add(s.get("quarter", q), int(s.get("horizon_q", 1)) - 1) if s.get("quarter") else q
            if due != q:
                continue
            inst = s.get("instrument")
            if inst not in med:
                continue
            fired = _condition_fired(s.get("condition"), med, med_y, facts)
            kept, excused = AC.settle_signal(s, int(np.sign(round(med[inst], 6))), fired)
            c = AC.Credibility(**cred.get(s["actor"], {}))
            c.update(kept, excused)
            cred[s["actor"]] = {"a": c.a, "b": c.b, "kept": c.kept, "total": c.total}
        ch.facts, ch.political, ch.commitments, ch.signals, ch.cred = facts, political, commitments, signals, cred
        ch.cluster = {"id": cl["id"], "label": cl["label"] + (f"｜{'；'.join(split_labels)}" if split_labels else ""),
                      "p": rnd(cl["p"], 6), "head": cl["head"],
                      "members": [{"options": m["options"], "p": rnd(m["p"], 6)} for m in cl["members"]],
                      "realized": modal}
        if split_labels:
            ch.cluster["split"] = split_labels
        worst = max((x["share"] for x in misstated), default=0.0)
        ch.cluster["_discrete"] = {"split": split_labels, "cell_share": rnd(cell_share, 6), "misstated": misstated,
                                   "max_misstated": rnd(worst, 4), "budget": split.get("budget")}
        ch.branch = _branch_view(b)
        ch.stress = bool(np.mean(SY.stress_flags(ch.cloud.x, ch.cloud.y)) >= 0.5)
        band = "，".join(f"{BAND_ZH[k][0]} {med[k]:+.2g}{BAND_ZH[k][1]}" for k in BAND_ZH if k in med)   # read by actors
        sentence = (b.get("memory") or "").strip()
        line = (f"{q}（本支 {ch_p(cl, b):.0%}）：{'；'.join(labels[:5])}。{b.get('label', '')}"
                + (f"：{sentence}" if sentence else "") + (f"。季末中位数：{band}" if band else ""))
        ch.memory_line = line[:300]
        ch.memory = rebuild_memory(nd.memory, ch.memory_line, facts)
        live = live_facts(facts)
        items = [("action", x) for x in labels[:DIGEST_ACTION_MAX]]
        if ch.cloud.acc is not None:
            items += AC.account_events(nd.cloud.acc.fact_shares() if nd.cloud.acc is not None else None, ch.cloud.acc)
        if b["kind"] == "nature_event":
            items.insert(0, ("event", A.HAZARDS.get(b["hazard"], {}).get("zh", b["hazard"])))
        if sentence:
            items.append(("consequence", sentence))
        if band:
            items.append(("data", f"{q} 季末中位数：{band}"))
        items.append(("fact", "现行：" + "；".join(live[:4]) if live else "无持续事实"))
        ch.digest = [list(t) for t in AC.digest_guard(items)[0]]

    # ---------------------------------------------------------------- summaries and details
    def summarize(self, nd: Node, parent: Node | None, *, a0=None, bk=None, branch=None) -> dict:
        cl = nd.cloud
        rep = {k: rnd(v, 4) for k, v in MA.medians(cl.state, [k for k, _ in KEY_STATE] + ["s_y10", "s_gz",
                                                                                            "s_infexp1"]).items()}
        rep["cum_us_equity"] = rnd(float(np.median(cl.cum.get("us_equity", np.zeros(1)))), 4)
        bf = [f for f in BAND_FIELDS if f in cl.x]
        bands = {}
        if bf:
            Q = np.percentile(np.stack([cl.x[f] for f in bf], axis=1), [10, 50, 90], axis=0)
            bands = {f: [rnd(Q[0, j], 4), rnd(Q[1, j], 4), rnd(Q[2, j], 4)] for j, f in enumerate(bf)}
        assets = {}
        have = [a for a in self.assets if a in cl.y]
        if have:
            Qy = np.percentile(np.stack([cl.y[a] for a in have], axis=1), [5, 50, 95], axis=0)
            Qc = np.percentile(np.stack([cl.cum[a] for a in have], axis=1), [5, 50, 95], axis=0)
            for j, a in enumerate(have):
                assets[a] = {"q": [rnd(Qy[i, j], 4) for i in range(3)], "cum": [rnd(Qc[i, j], 4) for i in range(3)]}
        for a in self.assets:
            if a not in assets and a in cl.cum:
                assets[a] = {"q": None, "cum": [0.0, 0.0, 0.0]}
        accts = []
        if cl.acc is not None:
            vals = cl.acc._values()
            keys = [k for k, _ in ACCOUNT_HEADLINE if k in vals]
            if keys:
                M = np.stack([np.broadcast_to(np.asarray(vals[k], float), (cl.n,)) for k in keys], axis=1)
                Qa = np.nanpercentile(M, [10, 50, 90], axis=0)
                zh = dict(ACCOUNT_HEADLINE)
                accts = [{"line": k, "zh": zh[k], "median": rnd(Qa[1, j], 4), "p10": rnd(Qa[0, j], 4),
                          "p90": rnd(Qa[2, j], 4)} for j, k in enumerate(keys)]
        ident = None
        if cl.acc is not None and cl.acc.resid:
            ident = {"ok": bool(cl.acc.ok()), "max": rnd(max(cl.acc.resid.values()), 3)}
        out = {"rep": rep, "bands": bands, "assets": assets, "accounts": accts, "identities": ident,
               "cpi_cum": rnd(float(np.median(cl.cpi)), 4) if cl.cpi is not None else None,
               "headline": {"us_equity_cum_p50": (assets.get("us_equity") or {}).get("cum", [None] * 3)[1],
                            "s_ff": rep.get("s_ff"), "s_unrate": rep.get("s_unrate"), "s_core_yoy": rep.get("s_core_yoy")}}
        if a0 is not None and bk is not None and branch is not None:
            out["attribution"] = SY.attribution(a0.y, bk["draws"].y, cl.y, branch.get("kind"), branch.get("axis"),
                                                ("us_equity", "nasdaq", "treasury_10y", "gold", "crude_oil"))
            out["inputs"] = self.inputs(cl, bk, branch)
        return out

    def inputs(self, cl: Cloud, bk: dict, branch: dict) -> dict:
        written = set(bk["written"])
        pinned = set(bk["given"].ids)
        fs = [f for f in self.model.x_fields if f in cl.x]
        Q = np.percentile(np.stack([cl.x[f] for f in fs], axis=1), [10, 50, 90], axis=0)
        out = {}
        for j, f in enumerate(fs):
            if f in written:
                src = "actor"
            elif branch.get("kind") == "mechanism" and f in (branch.get("shifts") or {}):
                src = "synth:mechanism"
            elif branch.get("axis") == f:
                src = "synth:partition"
            elif f in pinned:
                src = "a0_pinned"
            else:
                src = "model"
            out[f] = [rnd(Q[0, j], 4), rnd(Q[1, j], 4), rnd(Q[2, j], 4), src]
        return out

    def detail(self, nd: Node, q: str, dec: dict, clusters: list, bks: dict, consequences: list, calls: list,
               report: list, a0) -> dict:
        nxt = []
        for aid in dec["order"]:
            ab = dec["actors"][aid]
            nxt.append({"actor": aid, "zh": ab["zh"], "domain": ab["domain"], "c": ab["c"], "c_used": ab.get("c_used"),
                        "prior": ab["prior"], "branching": ab.get("branching", False), "fallback": ab.get("fallback"),
                        "follower_of": ab.get("follower_of"), "flags": ab.get("flags"),
                        "required": ab.get("required"),
                        "marginal_of_rows": bool(ab.get("marginal_of_rows")),
                        "options": [{"id": o["id"], "label": o["label"], "p_prior": rnd(o["p_prior"], 4),
                                     "p_llm": rnd(o["p_llm"], 4), "p_used": rnd(o["p_used"], 4),
                                     **({"p_used_uncond": rnd(o.get("p_used_uncond"), 4),
                                         "p_llm_uncond": rnd(o.get("p_llm_uncond"), 4),
                                         "p_prior_uncond": rnd(o.get("p_prior_uncond"), 4)}
                                        if ab.get("marginal_of_rows") else {}),
                                     "magnitude": o.get("magnitude"), "rationale": o.get("rationale"),
                                     "signal": o.get("signal"), "source": o.get("source"),
                                     "status_quo": bool(o.get("status_quo")),
                                     "bin": list(o["bin"]) if o.get("bin") else None, "custom": o.get("custom"),
                                     "param": o.get("param"),
                                     "writes": _writes_text(self._opt_payload(aid, o, nd))} for o in ab["options"]],
                        "cond": ab.get("cond"), "cond_prior": ab.get("cond_prior"),
                        "attributes": [{"domain": d["domain"], "zh": d["zh"],
                                        "options": [{"id": o["id"], "label": o["label"], "p": rnd(o["p"], 4)}
                                                    for o in d["options"]]} for d in ab["attributes"]]})
        cls = []
        for cl in clusters:
            bk = bks[cl["id"]]
            x = bk["draws"].x
            cls.append({"id": cl["id"], "label": cl["label"], "p": rnd(cl["p"], 6), "head": cl["head"],
                        "members": [{"options": m["options"], "p": rnd(m["p"], 6)} for m in cl["members"]],
                        "written": bk["written"], "protocols": bk["protocols"], "flags": bk["flags"],
                        "n_particles": int(len(bk["rows"])), "nature": bk.get("nature"),
                        "split": [{"actor": bk["split"]["cols"][i]["actor"], "domain": bk["split"]["cols"][i]["domain"],
                                   "kept": [bk["split"]["cols"][i]["options"][bk["split"]["cols"][i]["rep"][c]]
                                            for c in bk["split"]["cols"][i]["kept"]]}
                                  for i in (bk.get("split") or {}).get("chosen", [])],
                        "pre_run": {f: [rnd(np.percentile(x[f], 10), 4), rnd(np.median(x[f]), 4),
                                        rnd(np.percentile(x[f], 90), 4)] for f in BAND_FIELDS if f in x}})
        acc_detail = None
        if nd.cloud.acc is not None:
            s = nd.cloud.acc.summary()
            acc_detail = {"lines": {k: [rnd(v["p10"], 4), rnd(v["median"], 4), rnd(v["p90"], 4)] for k, v in s.items()},
                          "identities": {k: rnd(v, 3) for k, v in (nd.cloud.acc.resid or {}).items()},
                          "gates": _gate_view(AC.gates(nd.cloud.acc))}
        return {"key": nd.key, "quarter_next": q, "level": nd.level, "agent_view": "median+bands",
                "next": nxt, "clusters": cls, "consequences": consequences,
                "hazards": {h: {k: (rnd(v, 5) if isinstance(v, float) else v) for k, v in hv.items()}
                            for h, hv in dec["hazards"].items()},
                "packets": {k: {kk: v[kk] for kk in ("text", "chars", "tokens_est", "hash")}
                            for k, v in dec["packets"].items()},
                "accounts": acc_detail, "calls": calls, "validation_report": report[:200], "flags": nd.flags,
                "a0": {f: [rnd(np.percentile(a0.x[f], 10), 4), rnd(np.median(a0.x[f]), 4),
                           rnd(np.percentile(a0.x[f], 90), 4)] for f in BAND_FIELDS if f in a0.x}}

    # ---------------------------------------------------------------- prompt context (llm_io.NodeContext)
    def board_rows(self, nd: Node, viewer: str | None = None) -> list:
        rows = []
        for s in nd.signals:
            if viewer and s.get("actor") == viewer:
                continue
            c = AC.Credibility(**(nd.cred or {}).get(s["actor"], {}))
            rows.append({"actor": s["actor"], "actor_zh": AC.ZH.get(s["actor"], s["actor"]), "text": s.get("text", ""),
                         "instrument": s.get("instrument"), "direction": int(s.get("direction", 0)),
                         "horizon_q": int(s.get("horizon_q", 1)), "condition": s.get("condition"),
                         "commit": bool(s.get("commit")), "credibility": round(c.mean, 2), "kept": c.kept,
                         "total": c.total})
        return rows

    def board_text(self, nd: Node, viewer: str) -> str:
        sigs = [AC.Signal(actor=s["actor"], text=s.get("text", ""), instrument=s.get("instrument"),
                          direction=int(s.get("direction", 0)), horizon_q=int(s.get("horizon_q", 1)),
                          condition=s.get("condition"), commit=bool(s.get("commit"))) for s in nd.signals]
        cred = {k: AC.Credibility(**v) for k, v in (nd.cred or {}).items()}
        return AC.signal_board(sigs, cred, viewer=viewer)

    def state_table(self, nd: Node) -> str:
        cl = nd.cloud
        root = nd.level == 0
        rows = ["# 一、共享状态表（" + ("起点：真实观测，数据截至 2026-09" if root else
                                     f"{nd.quarter} 季末：本分支中位数（10%–90%）") + "）",
                "| 字段 | 名称 | 数值 | 单位 |", "|---|---|---|---|"]
        for fid, zh, unit in STATE_ROWS:
            v = cl.state.get(fid)
            if v is None:
                continue
            v = np.asarray(v, float)
            if root or np.ptp(v) < 1e-9:
                val = AC.sig3(float(np.median(v)))
            else:
                val = f"{AC.sig3(float(np.median(v)))}（{AC.sig3(float(np.percentile(v, 10)))}–{AC.sig3(float(np.percentile(v, 90)))}）"
            rows.append(f"| {fid} | {zh} | {val} | {unit} |")
        for fid, zh, unit in LAST_ROWS:
            src = cl.x.get(fid) if cl.x else None
            if src is None:
                src = cl.state.get(f"{fid}_l1")
            if src is None:
                continue
            v = np.asarray(src, float)
            val = AC.sig3(float(np.median(v))) if (root or np.ptp(v) < 1e-9) else \
                f"{AC.sig3(float(np.median(v)))}（{AC.sig3(float(np.percentile(v, 10)))}–{AC.sig3(float(np.percentile(v, 90)))}）"
            rows.append(f"| {fid} | {zh} | {val} | {unit} |")
        cov = self.covariates(nd)
        rows.append(AC.regime_line(cov, nd.facts, A.quarter_add(nd.quarter, 1), cl.acc))
        return "\n".join(rows)

    def prompt_ctx(self, nd: Node, q: str, dec: dict):
        from .llm_io import NodeContext as PCtx
        facts = []
        for f, v in sorted(nd.facts.items()):
            if not v or f.startswith("synth:") or f.endswith("_expiry_q"):
                continue
            meta = A.FACTS.get(f, {})
            zh = meta.get("zh") or ACCOUNT_FACT_ZH.get(f, f)
            exp = nd.facts.get(f"{f}_expiry_q")
            if exp:
                zh += f"（{exp} 到期）"
            facts.append({"text": zh, "exit_hazard": meta.get("exit")} if meta.get("exit") is not None else zh)
        inst = []
        for it in self.items:
            if it.get("visibility", "public") != "public":
                continue
            status = {"confirmed": "已确认", "unconfirmed": "未确认", "default": "默认"}.get(it.get("status"), "")
            inst.append({"zh": it.get("zh"), "value": it.get("value"), "status": status})
        cal = []
        e = A.ELECTIONS.get(q)
        if e:
            cal.append(e["zh"])
        if q in A.FOMC_PINNED:
            cal.append(f"{A.FOMC_PINNED[q]['zh']}（已锁定）")
        cal.append("美联储本季两次例会；超大型科技公司财报：" + A.EARNINGS)
        commits = [f"{AC.ZH.get(cm.get('actor'), cm.get('actor'))}：{cm.get('zh') or cm.get('option')}（{cm.get('due_quarter')}）"
                   for cm in nd.commitments]
        return PCtx(quarter_next=q, depth=nd.level + 1, quarter_done=None if nd.level == 0 else nd.quarter,
                    state=self.state_table(nd), news=[t[1] for t in nd.digest][:6], board=self.board_rows(nd),
                    memory=nd.memory, facts=facts[:8], institutions=inst[:14], calendar=cal, commitments=commits,
                    as_of="2026-11-01", data_end="2026-09", order_seed=nd.key)


def ch_p(cl: dict, b: dict) -> float:
    return float(cl["p"]) * float(b["p"])


def _branch_view(b: dict) -> dict:
    keep = ("id", "kind", "label", "axis", "region", "p", "m", "q", "p_llm", "hazard", "rule", "memory", "mechanism",
            "cause", "anchors", "unintended", "beyond_history", "facts", "vs_prior")
    out = {k: b.get(k) for k in keep if k in b}
    if b.get("shifts"):
        out["shifts"] = {f: {k: rnd(v, 4) if isinstance(v, float) else v for k, v in s.items()}
                         for f, s in b["shifts"].items()}
    for k in ("p", "m", "q", "p_llm"):
        if isinstance(out.get(k), float):
            out[k] = rnd(out[k], 6)
    return out


def _writes_text(pay: dict) -> str:
    parts = []
    for w in pay.get("writes", []):
        if w["kind"] == "label":
            continue
        a = w.get("a")
        txt = f"{w['field']} {'+=' if w['kind'] == 'shift' else '='} {AC.sig3(a) if a is not None else '幅度'}"
        if int(w.get("dq") or 0) > 0:
            txt += f"（{w['dq']} 季后）"
        parts.append(txt)
    for e in pay.get("events", {}).get("oil_z", []):
        parts.append(f"石油供给合成量 {e['a']:+.2f}z")
    for e in pay.get("events", {}).get("gpr", []):
        parts.append(f"GPR {e['a_sd']:+.2f}sd")
    for f in pay.get("facts", []):
        parts.append(f"事实：{f.get('zh') or f['id']}")
    return "；".join(parts) or "—"


def _gate_view(g: dict) -> dict:
    return json.loads(json.dumps(g, default=_jsonable))


def _condition_fired(cond, med: Mapping, med_y: Mapping, facts: Mapping) -> bool:
    if not cond:
        return False
    if cond == "inflation_up":
        return med.get("d_core", 0) > 4.0
    if cond == "inflation_down":
        return med.get("d_core", 9) < 1.5
    if cond == "unemp_up_0.3":
        return med.get("d_unrate", 0) >= 0.3
    if cond == "equity_down_10":
        return med_y.get("us_equity", 0) <= math.log(0.9)
    if cond == "spread_up_0.5":
        return med.get("d_spread", 0) >= 0.5
    if cond == "oil_up_20":
        return med.get("d_oil", 0) >= 18.2
    return bool(facts.get({"escalation": "escalation", "ceasefire": "ceasefire",
                           "debt_limit_binding": "debt_limit_binding", "export_controls_up": "export_controls_cn",
                           "tariff_up_partner": "cn_retaliation", "ruling_against": "tariff_authority"}.get(cond, cond)))


def _average_answers(samples: list):
    """root_samples > 1: options matched by id, probabilities averaged (missing = 0)."""
    samples = [s for s in samples if s is not None]
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    base = copy.deepcopy(samples[0])
    acc: dict = {}
    for s in samples:
        for o in s.options:
            if o.get("p_llm") is not None:
                acc[o["id"]] = acc.get(o["id"], 0.0) + float(o["p_llm"]) / len(samples)
    for o in base.options:
        if o["id"] in acc:
            o["p_llm"] = acc[o["id"]]
    return base


# =====================================================================================================================
# Level reduction (§8.3)
# =====================================================================================================================

def node_vector(nd: Node) -> np.ndarray:
    st = nd.cloud.state if nd.cloud is not None else {}
    v = []
    for k, sd in KEY_STATE:
        val = st.get(k)
        v.append(float(np.median(val)) / sd if val is not None else (nd.summary.get("rep", {}).get(k) or 0.0) / sd)
    cum = nd.cloud.cum.get("us_equity") if nd.cloud is not None else None
    v.append((float(np.median(cum)) if cum is not None else 0.0) / CUM_SCALE)
    return np.array(v)


def fact_key(nd: Node) -> str:
    """The node's discrete state for reduction (+fact_penalty sd per differing element, never recombined): standing
    facts, the political state and pending commitments."""
    keys = [k for k, v in nd.facts.items() if v and not k.startswith("synth:")]
    keys += [f"pol:{k}={v}" for k, v in sorted((nd.political or {}).items()) if v not in (None, "", 0, 0.0, False)]
    keys += [f"commit:{cm.get('actor')}:{cm.get('option')}" for cm in nd.commitments or []]   # one per plan
    return canonical(sorted(set(keys)))


def discrete_stats(cands: list[Node]) -> dict:
    """Per level: candidates created by the discrete split, and the mass-weighted share of particles whose own
    discrete state (in the worst column) differs from their node's (the approximation that remains, §8.1)."""
    tot = sum(c.p_path for c in cands) or 1.0
    split = sum(1 for c in cands if ((c.summary or {}).get("discrete") or {}).get("split"))
    mis = sum(c.p_path * float(((c.summary or {}).get("discrete") or {}).get("max_misstated") or 0.0) for c in cands)
    return {"discrete_split_candidates": int(split), "discrete_misstated": rnd(mis / tot, 4)}


def reduce_level(cands: list[Node], B: int, *, eps: float, fact_penalty: float, n: int, seed: int,
                 keep_all: bool = False) -> tuple[list[Node], list[Node], dict]:
    """Recombination, then forward selection to B nodes minimising the Kantorovich distance; every dropped node's
    mass and particles move to its nearest survivor (pooled by mass and resampled to n). Returns (kept, pruned,
    stats); pruned nodes keep their own mass and the reason."""
    cands = sorted(cands, key=lambda c: c.key)
    if not cands:
        return [], [], {"candidates": 0, "kept": 0, "own_mass": 0.0, "reassigned": 0.0, "transport_error": 0.0,
                        "mass": 0.0}
    p = np.array([c.p_path for c in cands])
    X = np.array([node_vector(c) for c in cands])
    facts_sets = [json.loads(fact_key(c)) for c in cands]
    vocab = {k: i for i, k in enumerate(sorted({k for fs in facts_sets for k in fs}))}
    Mf = np.zeros((len(cands), max(len(vocab), 1)), np.float64)
    for i, fs in enumerate(facts_sets):
        for k in fs:
            Mf[i, vocab[k]] = 1.0
    F = Mf @ (1.0 - Mf).T + (1.0 - Mf) @ Mf.T          # |facts_i Δ facts_j|
    same = F < 0.5
    sq = (X ** 2).sum(axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)) + fact_penalty * F
    np.fill_diagonal(D, 0.0)
    a1 = recombine(p, D, same, eps)
    heads = sorted(set(int(h) for h in a1))
    pos = {h: j for j, h in enumerate(heads)}
    ph = np.bincount([pos[int(h)] for h in a1], weights=p, minlength=len(heads))
    keep_n = len(heads) if keep_all else min(int(B), len(heads))
    J_h, a2 = forward_select(ph, D[np.ix_(heads, heads)], keep_n)
    kept_idx = [heads[j] for j in J_h]
    final = np.array([heads[a2[pos[int(a1[i])]]] for i in range(len(cands))])
    members: dict = {}
    for i, f in enumerate(final):
        members.setdefault(int(f), []).append(i)
    kept, pruned = [], []
    for j in kept_idx:
        grp = members.get(j, [j])
        nd = cands[j]
        nd.own_mass = float(p[j])
        if len(grp) > 1 or (nd.cloud is not None and nd.cloud.n != n):
            clouds = [cands[i].cloud for i in grp]
            w = np.concatenate([np.full(c.n, p[i] / c.n) for c, i in zip(clouds, grp)])
            big = Cloud.concat(clouds) if len(clouds) > 1 else clouds[0]
            idx = systematic(w, n, seed_for(seed, nd.key, "pool"))
            nd.cloud = big.take(idx)
        if len(grp) > 1:
            nd.absorbed = [{"key": cands[i].key, "p": rnd(p[i], 6), "label": (cands[i].cluster or {}).get("label"),
                            "branch": (cands[i].branch or {}).get("label"), "distance": rnd(D[i, j], 3),
                            "how": "recombined" if a1[i] == j else "reassigned"} for i in grp if i != j]
        nd.p_path = float(p[grp].sum())
        kept.append(nd)
    for i, c in enumerate(cands):
        if final[i] == i:
            continue
        surv = cands[int(final[i])]
        how = "recombined" if int(a1[i]) != i else "reassigned"
        c.status = "pruned"
        c.absorbed_into = surv.key
        c.own_mass = float(p[i])
        c.reason = (f"与 {surv.key[:8]} 重合（距离 {D[i, int(final[i])]:.2f} 个标准差，事实相同），并入" if how == "recombined"
                    else f"前向选择未保留：质量与粒子并入最近的保留节点 {surv.key[:8]}（距离 {D[i, int(final[i])]:.2f}）")
        c.flags = sorted(set(c.flags) | {how})
        c.cloud = None
        pruned.append(c)
    te = float(np.sum(p * D[np.arange(len(cands)), final]))
    stats = {"candidates": len(cands), "kept": len(kept), "own_mass": float(p[kept_idx].sum()),
             "reassigned": float(1.0 - p[kept_idx].sum() / p.sum()) if p.sum() > 0 else 0.0,
             "transport_error": te, "mass": float(p.sum()), "recombined": int(len(cands) - len(heads))}
    return kept, pruned, stats


# =====================================================================================================================
# The run
# =====================================================================================================================

class Paused(Exception):
    def __init__(self, reason: str, message: str = "", reset_at: float | None = None, level: int | None = None):
        super().__init__(message or reason)
        self.reason, self.message, self.reset_at, self.level = reason, message or reason, reset_at, level


def make_llm(settings: TreeSettings, *, cancel: threading.Event | None, journal: Journal | None,
             cache_dir: Path | None = None, model: str | None = None, max_calls: int | None = None, versions=None,
             kill: threading.Event | None = None, ledger: Path | None = None, hard_cap: int | None = None):
    from .llm_io import AgentLLM, FileCache
    from .llm_io import Settings as LSettings
    ls = LSettings(model=model or settings.model, max_calls=max_calls if max_calls is not None else settings.max_calls,
                   hard_cap=hard_cap if hard_cap is not None else settings.hard_cap,
                   concurrency=int(settings.concurrency), safe_mode=bool(settings.safe_mode),
                   confirmed=bool(settings.confirmed), versions=dict(versions or {}))
    if settings.effort:
        ls.effort.update(settings.effort)
    emit = (lambda ev, **kw: journal.emit(ev, **kw)) if journal is not None else None
    return AgentLLM(ls, cache=FileCache(cache_dir or CACHE_DIR), cancel=cancel, on_event=emit, kill=kill, ledger=ledger,
                    failure_log=FAILURE_LOG if cache_dir is None else Path(cache_dir) / "cli_failures.jsonl")


def versions(adapter=None) -> dict:
    adapter = adapter or MA.stub()
    from ..llm import agent_prompts
    return {"engine": ENGINE_VERSION, "catalogue": A.CATALOGUE_VERSION, "accounts": AC.ACCOUNTS_VERSION,
            "priors": P.PRIORS_VERSION, "model_adapter": adapter.version, "prompts": agent_prompts.PROMPT_VERSION}


class TreeRun:
    """Breadth-first beam over levels. The result is a pure function of (settings, adapter version, catalogue,
    institutions, cache contents); a checkpoint after every level makes --resume continue exactly."""

    def __init__(self, settings, store: Store, *, adapter=None, llm=None, deep_llm=None, root_accounts=None,
                 institutions=None, journal: Journal | None = None, cancel: threading.Event | None = None,
                 cache_dir: Path | None = None, keep_all_last: bool = False, kill: threading.Event | None = None):
        self.st = TreeSettings.coerce(settings)
        self.store = store
        self.adapter = adapter or MA.stub()
        self.cancel = cancel or threading.Event()        # graceful: no new call starts
        self.kill = kill or threading.Event()            # force: calls in flight are killed
        self.journal = journal or Journal(store.root / "journal.jsonl")
        self.cache_dir = cache_dir
        self.versions = versions(self.adapter)
        if llm is None and self.st.tree_kind == "agent":
            llm = make_llm(self.st, cancel=self.cancel, journal=self.journal, cache_dir=cache_dir,
                           versions=self.versions, kill=self.kill, ledger=store.root / LEDGER_NAME)
            if self.st.deep_model and self.st.deep_model != (self.st.model or llm.model):
                deep_llm = make_llm(self.st, cancel=self.cancel, journal=self.journal, cache_dir=cache_dir,
                                    model=self.st.deep_model, versions=self.versions, kill=self.kill,
                                    ledger=store.root / f"deep_{LEDGER_NAME}")
        self.llm, self.deep_llm = llm, deep_llm
        if root_accounts is None:
            root_accounts = AC.load_root()
        self.engine = Engine(self.st, adapter=self.adapter, llm=llm, deep_llm=deep_llm, root_accounts=root_accounts,
                             institutions=institutions, journal=self.journal)
        self.level_stats: list = []
        self.keep_all_last = keep_all_last
        self.leaves: list[Node] = []
        self.index: dict = {}

    # ---------------------------------------------------------------- budget bookkeeping
    def engine_state(self) -> dict:
        out = {"level_stats": self.level_stats, "index": self.index}
        if self.llm is not None:
            out["llm_stats"] = self.llm.stats
        if self.deep_llm is not None:
            out["deep_llm_stats"] = self.deep_llm.stats
        return out

    def restore(self, es: Mapping) -> None:
        self.level_stats = list(es.get("level_stats") or [])
        self.index = dict(es.get("index") or {})
        if self.llm is not None and es.get("llm_stats"):
            self.llm.stats.update(es["llm_stats"])
        if self.deep_llm is not None and es.get("deep_llm_stats"):
            self.deep_llm.stats.update(es["deep_llm_stats"])

    def calls_new(self) -> int:
        """The budget counter (new calls as an uninterrupted run would have counted them)."""
        return int(sum((x.stats.get("calls_new", 0) if x is not None else 0) for x in (self.llm, self.deep_llm)))

    def calls_paid(self) -> int:
        """Distinct calls this run paid for (the ledger; exact across pauses and resumes)."""
        return int(sum((len(getattr(x, "paid", ()) or ()) if x is not None else 0) for x in (self.llm, self.deep_llm)))

    def _parallel_ok(self, frontier: list) -> bool:
        caps = [c for c in (self.st.max_calls, self.st.hard_cap) if c is not None]
        if self.llm is None or not caps:
            return True
        left = int(min(caps)) - self.calls_new()
        return left >= 8 * len(frontier)            # upper bound per node: actor rounds, repairs, rephrases, synth

    # ---------------------------------------------------------------- the loop
    def run(self, *, resume: bool = True, on_level: Callable | None = None) -> dict:
        ck = self.store.load_checkpoint() if resume else None
        if ck is not None:
            level0, frontier, es = ck
            self.restore(es)
            for x in (self.llm, self.deep_llm):
                if x is not None and hasattr(x, "restore_consumed"):
                    x.restore_consumed(level0)
            self.journal.emit("run_resumed", level=level0, frontier=len(frontier), calls_new=self.calls_new(),
                              calls_paid=self.calls_paid())
        else:
            root = self.engine.root()
            frontier, level0 = [root], 0
            self.index[root.key] = [0, "pending", None]
            self.store.save_frontier(root)
            self.journal.emit("run_started", settings=self.st.hash(), depth=self.st.depth, B=self.st.B,
                              tree_kind=self.st.tree_kind, stub=bool(getattr(self.adapter, "stub", True)))
        pruned_buf = self._read_pruned(level0)
        for d in range(level0, int(self.st.depth)):
            if self.cancel.is_set():
                raise Paused("cancelled", "推演已取消", level=d)
            q = A.quarter_add(self.st.first_quarter, d)
            for x in (self.llm, self.deep_llm):
                if x is not None:
                    x.level = d
            self.journal.emit("level_started", level=d, quarter=q, nodes=len(frontier))
            t0 = time.monotonic()
            results = self._expand_level(frontier, d)
            cands, details = [], {}
            for nd, (children, detail) in zip(frontier, results):
                cands += children
                details[nd.key] = detail
                self.journal.emit("node_done", level=d, node=nd.key, children=len(children), flags=nd.flags)
            last = d == int(self.st.depth) - 1
            disc = discrete_stats(cands)
            kept, pruned, stats = reduce_level(cands, int(self.st.B), eps=float(self.st.eps_merge),
                                               fact_penalty=float(self.st.fact_penalty), n=int(self.st.n_particles),
                                               seed=int(self.st.seed), keep_all=last and self.keep_all_last)
            stats.update(disc)
            by_key = {c.key: c for c in kept + pruned}
            for nd in frontier:
                nd.children = []
                for ch_key in [c.key for c in cands if c.parent == nd.key]:
                    c = by_key[ch_key]
                    nd.children.append({"key": c.key, "status": "expanded" if c.status != "pruned" and not last else
                                        ("leaf" if c.status != "pruned" else "pruned"),
                                        "p_cond": rnd(c.p_cond, 6), "own_mass": rnd(c.own_mass, 6),
                                        "label": (c.cluster or {}).get("label"), "branch": (c.branch or {}).get("label"),
                                        "kind": (c.branch or {}).get("kind"), "reason": c.reason,
                                        "absorbed_into": c.absorbed_into})
            records = [nd.record() for nd in frontier] + pruned_buf
            self.store.write_level(d, records, details)
            for nd in frontier:
                self.index[nd.key] = [d, nd.status, nd.parent]
            for c in pruned_buf:
                self.index[c["key"]] = [d, "pruned", c.get("parent")]
            pruned_buf = [c.record() for c in pruned]
            stats.update(level=d + 1, quarter=q, seconds=round(time.monotonic() - t0, 2), calls_new=self.calls_new(),
                         calls_paid=self.calls_paid())
            self.level_stats = [s for s in self.level_stats if s.get("level") != d + 1] + [stats]
            for c in kept:
                c.status = "leaf" if last else "pending"
                if not last:
                    self.store.save_frontier(c)
            frontier = kept
            self._write_pruned(d + 1, pruned_buf)
            self.store.save_checkpoint(d + 1, frontier, self.engine_state())
            self.journal.emit("level_done", **{k: (rnd(v, 6) if isinstance(v, float) else v) for k, v in stats.items()})
            if on_level is not None:
                on_level(d + 1, stats)
        d_last = int(self.st.depth)
        for nd in frontier:
            nd.status = "leaf"
            self.index[nd.key] = [d_last, "leaf", nd.parent]
            self.store.save_leaf(nd)
        self.store.write_level(d_last, [nd.record() for nd in frontier] + pruned_buf, {})
        for c in pruned_buf:
            self.index[c["key"]] = [d_last, "pruned", c.get("parent")]
        write_json(self.store.root / "index.json", self.index)
        self.leaves = frontier
        th = self.tree_hash()
        self.journal.emit("run_done", calls_new=self.calls_new(), calls_paid=self.calls_paid(), tree_hash=th,
                          leaves=len(frontier))
        return {"tree_hash": th, "leaves": [nd.key for nd in frontier], "levels": self.level_stats,
                "calls_new": self.calls_new(), "calls_paid": self.calls_paid(),
                "llm_stats": self.llm.stats if self.llm is not None else None}

    def _read_pruned(self, level: int) -> list:
        p = self.store.root / "checkpoint" / f"pruned_{level:03d}.json"
        return read_json(p) if p.exists() else []

    def _write_pruned(self, level: int, recs: list) -> None:
        write_json(self.store.root / "checkpoint" / f"pruned_{level:03d}.json", recs)

    def _expand_level(self, frontier: list, d: int) -> list:
        order = sorted(range(len(frontier)), key=lambda i: frontier[i].key)
        out: list = [None] * len(frontier)

        def one(i):
            return self.engine.expand(frontier[i])
        try:
            from .llm_io import LLMPause
        except ImportError:   # pragma: no cover
            LLMPause = Exception          # noqa: N806
        try:
            if self._parallel_ok(frontier) and len(frontier) > 1 and int(self.st.concurrency) > 1:
                ex = cf.ThreadPoolExecutor(max_workers=max(1, int(self.st.concurrency)))
                try:
                    futs = {i: ex.submit(one, i) for i in order}
                    for i in order:
                        try:
                            out[i] = futs[i].result()
                        except LLMPause as err:
                            # the first pause stops the level: no queued node starts, no running node starts a call
                            for x in (self.llm, self.deep_llm):
                                if x is not None and hasattr(x, "halt"):
                                    x.halt(err)
                            ex.shutdown(wait=True, cancel_futures=True)
                            raise
                finally:
                    ex.shutdown(wait=True, cancel_futures=True)
            else:
                for i in order:
                    out[i] = one(i)
        except LLMPause as err:
            raise Paused(err.reason, str(err), getattr(err, "reset_at", None), level=d) from err
        return out

    def tree_hash(self) -> str:
        h = hashlib.sha256()
        for d in self.store.levels():
            lv = self.store.read_level(d) or {}
            for r in sorted(lv.get("nodes", []), key=lambda r: r["key"]):
                h.update(canonical({k: r.get(k) for k in ("key", "parent", "p_path", "p_cond", "status", "summary",
                                                          "absorbed_into", "facts")}).encode("utf-8"))
        return h.hexdigest()[:16]


# =====================================================================================================================
# The hand-off (§14): a quarterly scenario set in the scenario_paths.npz layout (disabled until the portfolio and
# withdrawal services accept quarterly sets; stub runs are never exported)
# =====================================================================================================================

EXPORT_ENABLED = False
EXPORT_DISABLED_WHY = "组合评估与提取服务尚不支持季度情景集（step_months = 3）：交给组合评估暂时禁用"


def export_arrays(leaves: list, assets: Iterable[str], first_label: str = "2026-09") -> dict:
    """{total, income, cpi, probs, ids, assets, labels, step_months} from leaf clouds that carried their quarterly
    paths (keep_paths "full"): S leaves × N particles × T quarters × A assets of simple total returns; probabilities
    are the leaves' reassigned masses (they sum to 1)."""
    assets = list(assets)
    leaves = [lf for lf in leaves if lf.cloud is not None and lf.cloud.path is not None]
    if not leaves:
        raise ValueError("叶节点没有保存季度路径（keep_paths 不是 full）：无法生成情景集")
    S_, N = len(leaves), min(lf.cloud.n for lf in leaves)
    T_ = min(lf.cloud.path.shape[1] for lf in leaves)
    total = np.stack([lf.cloud.path[:N, :T_, :] for lf in leaves]).astype(np.float32)
    income = np.zeros_like(total)
    cpi = np.ones((S_, N, T_ + 1))
    for i, lf in enumerate(leaves):
        if lf.cloud.cpi is not None:
            cpi[i, :, -1] = np.exp(lf.cloud.cpi[:N])
    p = np.array([lf.p_path for lf in leaves], float)
    y, m = int(first_label[:4]), int(first_label[5:7])
    labels = []
    for t in range(T_ + 1):
        k = y * 12 + (m - 1) + 3 * t
        labels.append(f"{k // 12}-{k % 12 + 1:02d}")
    return {"total": total, "income": income, "cpi": cpi, "probs": p / p.sum(), "ids": np.array([lf.key for lf in leaves]),
            "assets": np.array(assets), "labels": np.array(labels), "step_months": np.array(3)}


# =====================================================================================================================
# The estimate (§11.3; design/d02_cost.py generalised)
# =====================================================================================================================

MEMORY_OVERHEAD = 1.9               # measured peak RSS / cloud bytes (B 30 × 6 quarters, N 500, mock + stub)
CAND_PARTICLES_PER_CLUSTER = 1.3    # candidates' particles per action cluster, in units of N (natural counts)
MEMORY_BASE_BYTES = 170 * 2**20     # the process with numpy, pandas, scipy and the engine imported
TOK = {"actor_system": 9000, "actor_node": 3500, "actor_out_merged": 3500, "actor_out_sep": 900,
       "packet_merged": 1750, "packet_sep": 500, "synth_system": 6000, "synth_node": 2500, "synth_per_cluster_in": 900,
       "synth_sector_block": 300, "synth_per_branch_out": 250, "synth_out_fixed": 500}
LATENCY_S = (20.0, 40.0, 60.0)        # per call (assumed until measured; ui/UI.md §3.3)
DEEP_LATENCY_FACTOR = 0.6             # a faster model at deep levels (assumption, unmeasured)


SYNTH_GATE_PASS = 1.0                 # share of expanded nodes whose synthesizer gate passes (1.0: an upper bound;
                                      # the mock passes about 0.65 below the root)


def estimate(settings, *, latency: tuple | None = None, root_actors: int | None = None, cached_levels: int = 0) -> dict:
    """Upper-bound plan: expanded nodes n_0 = 1, n_d = min(B, ceil(n_{d−1} · K · M)); calls per level (separate root
    actor calls in stage waves, one merged call below, the gated synthesizer in the first synth_depth levels with
    every gate passing), rounds, minutes and tokens; the first `cached_levels` levels are cache hits (calls_cached).
    At c_scale 0 no actor call is made (the synthesizer still is, unless it is off). Repairs and rephrases are not
    counted (the hard cap, 1.2 × this estimate, leaves room for them). Plan usage has no published token figure:
    calls and tokens only."""
    st = TreeSettings.coerce(settings)
    lat = tuple(latency or LATENCY_S)
    if len(lat) == 2:
        lat = (lat[0], (lat[0] + lat[1]) / 2, lat[1])
    ctx = A.NodeContext.of(None)
    stages: dict = {}
    for a in A.active_actors(ctx):
        if a["llm"] and a["active"] and (st.actors is None or a["id"] in st.actors):
            stages.setdefault(a["stage"], []).append(a["id"])
    n_root = root_actors if root_actors is not None else sum(len(v) for v in stages.values())
    M = 2.5 if (st.synth or {}).get("mode", "gated") != "off" else 1.25
    levels = [1]
    for _ in range(1, int(st.depth)):
        levels.append(int(min(st.B, math.ceil(levels[-1] * st.K * M))))
    k = max(1, int(st.concurrency))
    sy = st.synth or {}
    per = []
    tot = {"actor": 0, "synth": 0, "rounds": 0, "tin": 0, "tout": 0, "deep_calls": 0, "seconds": [0.0, 0.0, 0.0],
           "cached": 0}
    llm = st.tree_kind == "agent"
    actors_ask = st.c_scale is None or float(st.c_scale) > 0
    synth_ask = float(sy.get("c_s", SY.C_S)) > 0 or float(sy.get("c_m", SY.C_M)) > 0
    for d, n in enumerate(levels):
        if not llm or d >= st.llm_levels:
            per.append({"level": d, "quarter": A.quarter_add(st.first_quarter, d), "nodes": n, "actor_calls": 0,
                        "synth_calls": 0, "rounds": 0, "mode": "prior_only"})
            continue
        sep = st.call_mode == "separate" or (st.call_mode == "hybrid" and d == 0)
        samples = max(1, int(st.root_samples)) if d == 0 else 1
        if not actors_ask:
            c, r, tin, tout = 0, 0, 0, 0
        elif sep:
            c = n_root * n * samples
            r = sum(math.ceil(len(v) * n * samples / k) for v in stages.values()) if root_actors is None else math.ceil(c / k)
            tin = c * (TOK["actor_system"] + TOK["actor_node"] + TOK["packet_sep"])
            tout = c * TOK["actor_out_sep"]
        else:
            c = n
            r = math.ceil(n / k)
            tin = c * (TOK["actor_system"] + TOK["actor_node"] + TOK["packet_merged"])
            tout = c * TOK["actor_out_merged"]
        s = 0
        mode = sy.get("mode", "gated")
        if not synth_ask:
            s = 0
        elif mode == "every":
            s = n
        elif mode == "gated" and (d == 0 or d < int(sy.get("depth", 4))):
            s = n if d == 0 else math.ceil(n * SYNTH_GATE_PASS)
        rs = math.ceil(s / k) if s else 0
        tin += s * (TOK["synth_system"] + TOK["synth_node"] + st.K * TOK["synth_per_cluster_in"] + TOK["synth_sector_block"])
        tout += s * (st.K * M * TOK["synth_per_branch_out"] + TOK["synth_out_fixed"])
        deep = bool(st.deep_model) and d >= int(st.deep_from)
        f = DEEP_LATENCY_FACTOR if deep else 1.0
        for i in range(3):
            tot["seconds"][i] += (r + rs) * lat[i] * f
        cached = d < cached_levels
        per.append({"level": d, "quarter": A.quarter_add(st.first_quarter, d), "nodes": n, "actor_calls": c,
                    "synth_calls": s, "rounds": r + rs, "mode": "separate" if sep else "merged",
                    "model": st.deep_model if deep else (st.model or "默认（claude-opus-5-5）"), "cached": cached})
        if cached:
            tot["cached"] += c + s
            continue
        tot["actor"] += c
        tot["synth"] += s
        tot["rounds"] += r + rs
        tot["tin"] += tin
        tot["tout"] += tout
        if deep:
            tot["deep_calls"] += c + s
    calls = tot["actor"] + tot["synth"]
    cap = st.max_calls
    hard = st.hard_cap if st.hard_cap is not None else max(HARD_CAP_MIN, int(math.ceil(HARD_CAP_FACTOR * calls)))
    return {"levels": levels, "expanded_nodes": int(sum(levels)), "per_level": per, "actor_calls": tot["actor"],
            "synth_calls": tot["synth"], "calls_new": calls, "calls_cached": tot["cached"], "calls_cap": cap,
            "hard_cap": hard, "over_cap": cap is not None and calls > cap, "rounds": tot["rounds"], "concurrency": k,
            "minutes": [round(x / 60, 1) for x in tot["seconds"]], "hours": [round(x / 3600, 2) for x in tot["seconds"]],
            "latency_s": list(lat), "tokens": {"input": int(tot["tin"]), "output": int(tot["tout"])},
            "model": st.model or "claude-opus-5-5", "effort": "medium",
            "deep_model": st.deep_model, "deep_from": st.deep_from, "deep_calls": tot["deep_calls"],
            "actors_ask": actors_ask,
            "basis": "上限估计：后果合成器按每个节点都通过门控计；每次调用 20/40/60 秒与 token 大小都是假设（未测量）；"
                     "格式修复与改述重问不计入，硬上限（预计的 1.2 倍）为它们留了余量",
            "usage_note": "推演通过本机 Claude Code 的非交互模式（claude -p）调用大模型，计入你的订阅用量；达到用量上限时推演会暂停，"
                          "已完成的调用保存在本地缓存里，额度恢复后继续不会重复计入。"}


HARD_CAP_FACTOR = 1.2                 # G9 (§10.4): a hard stop at 1.2 × the estimate
HARD_CAP_MIN = 10


def hard_cap_for(settings, *, spent: int = 0, cached_levels: int = 0) -> int:
    """G9: the new calls a run may make before it pauses with reason "budget": `spent` (calls already paid for)
    plus 1.2 × the estimate of the levels not done yet."""
    est = estimate(settings, cached_levels=int(cached_levels))
    return int(spent) + max(HARD_CAP_MIN, int(math.ceil(HARD_CAP_FACTOR * est["calls_new"])))


def ledger_size(root: Path) -> int:
    """Distinct calls a run folder paid for (its ledger)."""
    n = 0
    for name in (LEDGER_NAME, f"deep_{LEDGER_NAME}"):
        p = Path(root) / name
        if p.exists():
            try:
                n += len({ln.split("\t")[0] for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()})
            except OSError:
                pass
    return n


def memory_estimate(settings, *, adapter=None, root_accounts=None, n_probe: int = 64) -> dict:
    """Bytes per particle of a cloud (measured on a probe cloud) and the peak resident memory of a beam run:
    the frontier (B) plus the level's candidates (B · K · M_max), each with N particles."""
    st = TreeSettings.coerce(settings)
    adapter = adapter or MA.stub()
    eng = Engine(dataclasses.replace(st, n_particles=n_probe), adapter=adapter, llm=None,
                 root_accounts=root_accounts if root_accounts is not None else AC.load_root())
    root = eng.root()
    per_particle = root.cloud.nbytes() / n_probe
    x_y = (len(adapter.x_fields) + 2 * len(adapter.assets) + 2) * 8
    path_per_q = len(adapter.assets) * 4 if st.paths else 0
    per_particle_last = per_particle + x_y + path_per_q * int(st.depth)
    cands = int(st.B) * int(st.K) * int(st.M_max) * int(eng.split_max())
    # children keep their natural particle count (a cluster's N particles split among its branches and cells;
    # mechanism branches re-complete all of them): about 1.3 N per cluster, plus the frontier's N per node
    particles = (int(st.B) * int(st.K) * CAND_PARTICLES_PER_CLUSTER + int(st.B)) * int(st.n_particles)
    clouds = particles * per_particle_last
    peak = MEMORY_BASE_BYTES + MEMORY_OVERHEAD * clouds
    return {"bytes_per_particle": int(per_particle_last), "candidates_max": cands, "cloud_bytes": int(clouds),
            "peak_bytes": int(peak), "peak_mb": round(peak / 2**20, 1), "paths": st.paths,
            "basis": "候选云的粒子（每个动作簇约 1.3 N）× 每粒子字节 × 1.9（中间结果与 Python 开销）+ 170 MB（进程与依赖）；"
                     "系数按 B 30 × 6 季、500 粒子的实测峰值校准"}
