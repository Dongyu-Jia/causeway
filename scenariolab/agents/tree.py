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

On the real quarterly model (phase P14; capability "structural"):

* conditioning: no two-step protocol. The Fed's d_ff and d_bs pins are interventions ("do", the adapter's
  given_modes), d_tariff and d_lf are set last ("do_last"); Congress's fiscal levers go through structural= (do on
  d_deficit, d_transfers, d_perstax by their shocks, do_last on d_corptax and d_defense) and so does the hyperscaler
  fallback on d_itinv (a thin lever, do_last); a commitment due later is routed like the option that made it (the
  protocol travels with its write); the oil-supply composite and GPR contributions stay reduced-form observations
  (the schema's own supply signature carries their identification) and label x_oil_cause = 1 (supply) when
  |Σz| ≥ 0.5; the hazard event fields an action did not write are held at A0 by do_last; attribute options (the
  Fed's balance sheet, immigration, the strategic reserve) write like main options;
* transmission (P14 review): what an intervention reaches is measured, not assumed (transmission(): a probe with
  common random numbers over the quarter of the action and the two after it). On the quarterly stack the satellites
  never feed back into the macro core, so every satellite lever (the balance sheet, Congress's fiscal tools, tariffs,
  immigration, the hyperscaler fallback) is inert: its channel class becomes "absent" (the synthesizer's gate G1 and
  its mechanisms), transfers keep AGENT_DESIGN §6.3's response table (goods consumption +2.1 per pp, an observation
  the model completes the quarter around) and QE writes SIM_DESIGN §5.3's intervention paths (template G08: d_tp10,
  d_y2 set last, d_gz observed);
* common random numbers: B_k is completed with A0's own random stream (same particles, same seed), so a particle
  nothing pins gives back its A0 draw and the clusters differ only by their actions. A pin, even at a particle's
  own A0 value, is not neutral in the quarterly stack (its ±4 sd saturation and dollar hold re-shape the draw), so
  nothing is pinned that an action did not write;
* instruments: the Fed's options are target moves (both meetings of the quarter) and the model's d_ff is the change
  of the last month's average rate, so options are pinned in the model's units (fed_units: the carry-over of last
  quarter's late meeting plus the share of this quarter's moves the last month sees) and the end-of-quarter target
  is a node state ("ff_target"); the floor is the model's (field_info level_floor 0.1, the stub's 0); A0 bins
  (of the moves the A0 draws imply) split adjacent options at the midpoint of their value sets (the catalogue bins
  when every option is on the menu); a pinned first meeting renormalizes the A0 prior over the compatible options;
  a ranged Fed option's default size is the median of A0 inside its bin (≥ 30 particles); the President's tariff
  prior is the decision-history table (the model's d_tariff drifts; it is not a decision rule); prior-only bin actors
  (c = 0 on an A0-bin prior) leave their instrument to the model's own reaction function (no pin), each particle
  labelled with the bin of its A0 draw;
* robustness: tail particles (MA.tail_mask: not finite, GZ > 15, unemployment > 20%, the policy rate > 20%, the
  10-year yield < 0, CPI y/y outside history, any asset's |quarterly log return| > 1) are dropped and resampled
  within their cluster; the share is in the node summary (per quarter and cumulated along the path), the detail,
  the level stats and the journal (tail_particles), and no float32 cast overflows. The adapter's notes (floors,
  bounds, ±4 sd saturation) are aggregated per cluster into the detail, the level stats and the journal.
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import dataclasses
import gzip
import hashlib
import json
import math
import re
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
from . import channels as CH
from . import model_adapter as MA
from . import overlay as OV
from . import priors as P
from . import synthesizer as SY

ENGINE_VERSION = ("agent-tree-6 (2026-11-02; the channels review: inflation increments for the overlay and the core-goods "
                  "pass-through, channels in the agent tree only, the X-date on a live standoff, the ladder as levels, "
                  "activity decay, deferred attribution, version drift refused; agent-tree-5: template channels P1-P15, "
                  "the price-puzzle overlay, G1 on channel coverage; P14: measured transmission)")
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
STATE_ROWS = (("s_ff", "政策利率（联邦基金，季末月的月均：模型口径）", "%"), ("s_unrate", "失业率", "%"), ("s_core_yoy", "核心 CPI 同比", "%"),
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


def key_scales(meta: Mapping | None = None) -> tuple:
    """((state key, sd), …) of the node distances: the model's 1985–2019 quarterly sd of each start-of-quarter state
    (field_info hist.sd_q_1985_2019, else sd_q), else KEY_STATE's constants (the stub's values)."""
    out = []
    for k, sd0 in KEY_STATE:
        h = ((meta or {}).get(k) or {}).get("hist") or {}
        v = h.get("sd_q_1985_2019") or h.get("sd_q")
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = None
        out.append((k, v if v is not None and math.isfinite(v) and v > 0 else sd0))
    return tuple(out)


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

# ---------------------------------------------------------------- the Fed in the model's units (P14 review)
# The schema's s_ff is the monthly average of the quarter's last month and d_ff the change between two such averages
# (panel lvl_ff; the stub was fitted on the same definition), while the Fed's options are the quarter's total target
# move (both meetings). Each quarter's first meeting falls in its first or second month (fully in the last month's
# average); the second falls in the last month, which sees only the days after the decision took effect. Share of
# the last month after the second meeting (2026 FOMC dates: Mar 17–18, Jun 16–17, Sep 15–16, Dec 8–9; later years are
# assumed to keep this pattern, 判断值):
FOMC_LAST_MONTH_SHARE = {1: 13 / 31, 2: 13 / 30, 3: 14 / 30, 4: 22 / 31}
# the effective rate after the September 2026 hike, held at the pinned October meeting (FRED DFF 3.88% on 2026-10-29,
# IORB 3.90%, target range 3.75–4.00%; actors.INSTITUTION_DEFAULTS fomc_2026_10): the root's end-of-quarter target
ROOT_FF_TARGET = 3.88
FF_TARGET_KEY = "ff_target"                  # the engine's own state key (adapters carry it like any other key)
DEFICIT_ACTION_KEY = "_agent_deficit_action"  # a private draw column: actions' discretionary deficit change (books)
FF_TARGET_STEP = 0.25

# ---------------------------------------------------------------- transmission (P14 review)
# Fields whose response to an intervention says whether a lever reaches the economy or asset prices (the probe's
# quarter and the two after it: a VAR with two lags would show any feedback by then)
TRANSMISSION_FIELDS = ("d_rgdp", "d_unrate", "d_ip", "d_core", "d_cpi_goods", "d_cpi_serv", "d_cons_goods",
                       "d_cons_serv", "d_capex", "d_wage", "d_y10", "d_gz", "d_spread", "d_usd", "d_oil")
TRANSMISSION_ASSETS = ("us_equity", "nasdaq", "treasury_10y", "gold", "crude_oil")
TRANSMISSION_QUARTERS = 3
TRANSMISSION_PARTICLES = 64
TRANSMISSION_TOL = 0.01                      # a +1 sd intervention that moves no response field by 1% of its sd
                                             # (quarterly stack: the fiscal shocks reach ≤ 1.2e-4 sd, d_bs 0)
_TRANSMISSION: dict = {}                     # identity -> probe result (one probe per adapter per process)
_TRANSMISSION_LOCK = threading.Lock()
# AGENT_DESIGN §6.3's response table, used on models without the "structural" capability and, on structural models,
# for a lever whose intervention the model does not transmit: transfers -> goods consumption, +2.1% annualized per pp
# of personal income (the nondurables estimate, se 0.18; W_C01 A5; consistent with the 2008 payments, Parker et al.
# 2013); tax and purchase changes are small in the quarter (Romer–Romer 2010; Ramey 2019): no entry
RESPONSE_TABLE = {"transfers_chg": (("d_cons_goods", 2.1),)}      # lever role -> (field, R per unit)
# SIM_DESIGN §5.3 and template G08: u_bs alone moves no asset (26 quarters, 4 programs), so on a model whose d_bs does
# not transmit, QE comes with its intervention paths, in each field's quarterly sd (1985–2019): d_tp10 −1 sd (G08),
# d_gz −1…−3 sd (G08; −2 ± 0.5), d_y2 down (G08 gives no size: −0.5 ± 0.25 sd, 判断值). Satellites (d_tp10, d_y2) are
# set last (structural=): the asset layer prices them and nothing is read into them; the GZ spread is in the
# quarterly stack's core, so it is an observation the model completes the quarter around (financial conditions)
QE_PATHS = {"bs_qe": (("d_tp10", -1.0, 0.3), ("d_y2", -0.5, 0.25), ("d_gz", -2.0, 0.5))}
QE_PATHS_NOTE = "量化宽松按 2008–14 年的足迹写干预路径：期限溢价 −1 sd、2 年期收益率 −0.5 sd（判断值）、GZ 利差 −2 sd"
# a ranged option's default size from the A0 bin: only for decision instruments (the Fed's reaction function), and
# only with enough particles in the bin (P14 review: 4–8 particles made tariff sizes jump from 1.9 to 5.4)
A0_DECISION_ACTORS = ("fed",)
A0_MAGNITUDE_MIN = 30


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
    channels: str = "on"                     # template channels (channels.py): on | off (P14's response table and QE paths)
    puzzle_overlay: str = "auto"             # the price-puzzle overlay: auto (while the probe finds the puzzle) | off
    puzzle_literature_arm: str = "off"       # the literature's disinflation on top of the overlay (a user decision)
    big_hike_gate: str = "fixed"             # the Fed's ≥75bp gate: fixed (the channels round) | p14 (as P14 built it)

    HASH_EXCLUDE = ("concurrency", "max_calls", "hard_cap", "confirmed", "name", "prior_trees")
    # settings a folder saved before a field existed read as that field's behaviour then (TreeSettings.saved)
    SAVED_DEFAULTS = {"channels": "off", "puzzle_overlay": "off", "puzzle_literature_arm": "off", "big_hike_gate": "p14"}

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

    @classmethod
    def saved(cls, s) -> TreeSettings:
        """Settings read back from a folder (run.json, a snapshot manifest): a field the folder's engine did not have
        yet takes the behaviour that engine had (SAVED_DEFAULTS: a P14 folder has no template channels), so resuming
        or extending it never mixes two protocols in one tree."""
        if s is None or isinstance(s, TreeSettings):
            return cls.coerce(s)
        d = dict(s)
        for k, v in cls.SAVED_DEFAULTS.items():
            d.setdefault(k, v)
        return cls.coerce(d)

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
        if self.channels not in ("on", "off"):
            out.append("channels 只能是 on / off")
        if self.puzzle_overlay not in ("auto", "off"):
            out.append("puzzle_overlay 只能是 auto / off")
        if self.puzzle_literature_arm not in ("on", "off"):
            out.append("puzzle_literature_arm 只能是 on / off")
        if self.big_hike_gate not in ("fixed", "p14"):
            out.append("big_hike_gate 只能是 fixed / p14")
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


def clip_tail(state: Mapping, d, bounds: Mapping) -> MA.Draws:
    """The all-tail fallback of Engine.sanitize (no healthy particle to resample from): non-finite public values
    become 0 (no change this quarter); the GZ and unemployment levels and |US-equity log return| are clipped to the
    bounds."""
    def fix(v):
        a = np.asarray(v, float)
        return np.where(np.isfinite(a), a, 0.0)
    x = {k: (v if str(k).startswith("_") else fix(v)) for k, v in d.x.items()}
    y = {k: fix(v) for k, v in d.y.items()}
    inc = {k: fix(v) for k, v in (getattr(d, "inc", None) or {}).items()}
    cpi = None if getattr(d, "cpi_q", None) is None else fix(d.cpi_q)
    if "d_gz" in x and "s_gz" in state and "s_gz" in bounds:
        x["d_gz"] = np.minimum(x["d_gz"], float(bounds["s_gz"]) - np.asarray(state["s_gz"], float))
    if "d_unrate" in x and "s_unrate" in state and "s_unrate" in bounds:
        x["d_unrate"] = np.minimum(x["d_unrate"], float(bounds["s_unrate"]) - np.asarray(state["s_unrate"], float))
    if "us_equity" in y and "us_equity" in bounds:
        y["us_equity"] = np.clip(y["us_equity"], -float(bounds["us_equity"]), float(bounds["us_equity"]))
    return MA.Draws(x, y, inc, cpi, getattr(d, "drawn", None), list(getattr(d, "notes", None) or []))


def merge_notes(notes: Iterable[Mapping]) -> list[dict]:
    """The adapter's and the engine's notes of one cluster summed by (field, code, text): [{field, code, particles,
    text}], largest first (floors, bounds, the ±4 sd saturation, QE paths, cuts the floor delivered short)."""
    acc: dict = {}
    for nt in notes or []:
        if not isinstance(nt, Mapping):
            continue
        key = (str(nt.get("field") or ""), str(nt.get("code") or ""), str(nt.get("text") or "")[:160])
        row = acc.setdefault(key, {"field": nt.get("field"), "code": nt.get("code"), "particles": 0, "text": key[2]})
        row["particles"] += int(nt.get("particles") or nt.get("paths") or 0)
        if nt.get("fields"):
            row["fields"] = list(nt["fields"])[:8]
    return sorted(acc.values(), key=lambda r: (-r["particles"], str(r["code"] or ""), str(r["field"] or "")))


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
        arr = {k: (MA.to_float32(v) if (v.dtype == np.float64 and dtype == np.float32) else
                   (v.astype(dtype) if v.dtype == np.float64 else v)) for k, v in a.items()}
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
        arr = {k: MA.to_float32(v) for k, v in a.items()}
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
        self.model = adapter if adapter is not None else MA.load_adapter()
        self.llm = llm
        self.deep_llm = deep_llm
        self.journal = journal or Journal()
        self.root_acc = root_accounts
        self.items = institutions if institutions is not None else A.load_institutions()
        self.inst = A.institution_values(self.items)
        self.meta = OV.merge(self.model.field_info())          # the overlay's tree keys on the model's metadata
        self.sd = {f: self._sd(f) for f in self.model.x_fields}
        self.structural = "structural" in (getattr(self.model, "capabilities", ()) or ())
        self.ff_floor = MA.field_floor(self.meta, "d_ff",
                                       float((getattr(self.model, "level_floor", None) or {}).get("d_ff", 0.0)))
        self.given_modes = {f: (self.meta.get(f) or {}).get("given_mode") for f in self.model.x_fields
                            if (self.meta.get(f) or {}).get("given_mode")}
        self.tail_bounds = dict(getattr(self.model, "tail_bounds", None) or MA.TAIL_BOUNDS)
        self.key_scales = key_scales(self.meta)
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
        self.identity = MA.identity(self.model)
        # what the model's interventions reach (structural models only: without the capability the two-step protocol
        # and the response table of §6.3 apply as before); inert levers' channel is absent for the synthesizer
        self.transmission = transmission(self.model, self) if self.structural else {}
        self.inert = sorted(f for f, r in (self.transmission.get("levers") or {}).items() if r.get("inert"))
        if self.inert:
            self.meta = {k: dict(v) for k, v in self.meta.items()}
            for f in self.inert:
                row = self.meta.setdefault(f, {})
                row["channel_class_model"] = row.get("channel_class")
                row["channel_class"] = "absent"
                row["inert"] = True
        # template channels (channels.py): per particle in B_k, in the agent tree only (P20). Never in the model tree
        # (its leaf mixture is the model's own distribution, so no channel fires there, the X-date and a pinned
        # pandemic included) nor in the base-rate tree: it draws history's actions at history's frequencies, and the
        # model's residuals already contain their average effects (with channels the review measured GZ 3.02 against
        # 1.66 at quarter 20: the same actions counted twice). An agent tree at c = 0 pools exactly the base rates (it
        # is the base-rate tree, test_c0_agent_tree_equals_the_base_rate_tree), so it has none either
        base_pool = self.st.c_scale is not None and float(self.st.c_scale) == 0.0
        self.channels_on = self.st.channels == "on" and not self.neutral and self.kind == "agent" and not base_pool
        self.defer_attribution = False          # TreeRun: the channel attribution only for kept children
        self.layer = CH.Layer(self.sd, self.model.x_fields, facts_exit={f: v.get("exit") for f, v in A.FACTS.items()}) \
            if self.channels_on else None
        self._overlay: bool | None = None

    # ---------------------------------------------------------------- routing of an action's writes
    def route(self, fid: str, kind: str, protocol: str | None) -> str:
        """Where a write goes: "structural" (an intervention by structural=: do by its shock, else do_last; shift
        levers whose protocol is an intervention, on structural models) or "given" (instrument points through the
        adapter's given_modes, and every reduced-form observation)."""
        if (self.structural and kind == "shift" and protocol in self.STRUCTURAL_PROTOCOLS
                and protocol != "signature"):
            return "structural"
        return "given"

    def _sd(self, f: str) -> float:
        h = (self.meta.get(f) or {}).get("hist") or {}
        v = h.get("sd_q_1985_2019") or h.get("sd_q")
        if v:
            return float(v)
        try:
            return float(self.model.sd(f))
        except Exception:  # noqa: BLE001
            return 1.0

    # ---------------------------------------------------------------- the price-puzzle overlay (channels.py)
    @property
    def overlay_on(self) -> bool:
        """The interim neutral overlay runs while the probe still finds the price puzzle (transmission's ff_irf), on a
        structural model with channels on and puzzle_overlay "auto"; the kernel is measured once per adapter."""
        if self._overlay is None:
            ok = (self.channels_on and self.structural and self.st.puzzle_overlay == "auto"
                  and bool(((self.transmission or {}).get("ff_irf") or {}).get("puzzle")))
            if ok:
                ker = CH.overlay_kernel(self.model, self.identity)
                ok = ker is not None
                if ok:
                    self.layer.kernel = {f: np.asarray(v, float) for f, v in ker.items()}
            self._overlay = bool(ok)
        return self._overlay

    # ---------------------------------------------------------------- the root
    def root(self) -> Node:
        n = int(self.st.n_particles)
        state = self.model.root_state(n)
        state[FF_TARGET_KEY] = np.full(n, self.root_ff_target(state))
        if self.channels_on:
            state.update(self.layer.empty_state(n))
        acc = None
        if self.root_acc is not None:
            acc = AC.Accounts.from_root(self.root_acc, n, institutions=self.inst, resolver=self.resolver,
                                        quarter=ROOT_QUARTER)
        cum = {a: np.zeros(n) for a in self.assets}
        path = np.zeros((n, 0, len(self.assets)), np.float32) if self.st.paths else None
        cloud = Cloud(state, acc, {}, {}, cum, np.zeros(n), path, {})
        rs = A.root_state(self.items)
        facts = {k: v for k, v in rs["facts"].items()}
        key = hashlib.sha256(f"root|{self.st.first_quarter}|{self.identity}|{canonical(facts)}".encode()).hexdigest()[:16]
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
        st = MA.medians(cl.state, [k for k, v in cl.state.items() if not k.startswith("_") and np.ndim(v) == 1
                                   and not (self.meta.get(k) or {}).get("display_only")])
        st["ff_floor"] = float(self.ff_floor)                # the model's policy-rate floor (menus: room to cut)
        st["ff_target"] = float(np.median(self.ff_target(cl.state)))     # the end-of-quarter target (room to cut)
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
                             leader_choices=dict(leaders or {}), pinned_meeting=not self.neutral,
                             big_hike_gate=self.st.big_hike_gate)

    # ---------------------------------------------------------------- the Fed in the model's units (P14 review)
    def root_ff_target(self, state) -> float:
        """The root's end-of-quarter policy-rate target: ROOT_FF_TARGET (the effective rate after the mid-September
        hike) when the model's root s_ff is the September average of that path (within 0.5), else the model's own
        s_ff (another origin: no carry-over assumed)."""
        s = float(np.median(np.asarray(state[FED_FLOOR_FIELD], float)))
        return float(ROOT_FF_TARGET) if abs(ROOT_FF_TARGET - s) <= 0.5 else s

    def ff_target(self, state) -> np.ndarray:
        """Per particle: the policy-rate target at the end of the quarter the state starts after (FF_TARGET_KEY;
        a cloud saved before it existed: the model's s_ff, no carry-over)."""
        v = state.get(FF_TARGET_KEY)
        if v is None:
            return np.asarray(state[FED_FLOOR_FIELD], float).copy()
        return np.asarray(v, float)

    def fed_pin(self, q: str) -> dict | None:
        """The pinned first meeting of quarter q ({"move", "option"}), or None (actors.FOMC_PINNED)."""
        pin = A.FOMC_PINNED.get(q)
        if not pin:
            return None
        opt = str(self.inst.get(pin["id"]) or "hold")
        mv = A.PINNED_MEETING_MOVE.get(opt)
        return None if mv is None else {"move": float(mv), "option": opt, "meeting": pin["id"]}

    def fed_units(self, state, q: str, pinned: Mapping | None = None) -> dict:
        """The map between a quarter's total target move m (both meetings: the Fed's options) and the model's d_ff
        (the change of the last month's average rate): d_ff = carry + a + b·m per particle. carry = the end-of-
        quarter target minus s_ff (the part of last quarter's late meeting the model's level has not seen yet);
        f = the share of the last month after the second meeting (FOMC_LAST_MONTH_SHARE). A free quarter splits the
        move between the meetings evenly in expectation (a = 0, b = (1 + f)/2); with a pinned first meeting m1 the
        rest is the second meeting's (a = m1·(1 − f), b = f)."""
        f = float(FOMC_LAST_MONTH_SHARE[int(str(q)[-1])])
        carry = self.ff_target(state) - np.asarray(state[FED_FLOOR_FIELD], float)
        if pinned and pinned.get("move") is not None:
            m1 = float(pinned["move"])
            return {"carry": carry, "a": m1 * (1.0 - f), "b": f, "f": f, "pinned": m1}
        return {"carry": carry, "a": 0.0, "b": 0.5 * (1.0 + f), "f": f, "pinned": None}

    @staticmethod
    def implied_moves(units: Mapping, d_ff) -> np.ndarray:
        """The target moves the model's d_ff draws imply (the inverse of model_units)."""
        return (np.asarray(d_ff, float) - units["carry"] - units["a"]) / units["b"]

    @staticmethod
    def model_units(units: Mapping, move) -> np.ndarray:
        return units["carry"] + units["a"] + units["b"] * np.asarray(move, float)

    def instrument_view(self, nd: Node, q: str, aid: str, x: Mapping) -> Mapping:
        """An actor's instrument as its options read it: the Fed's d_ff draws as the target moves they imply (A0
        bins, bin labels); other instruments as they are."""
        if aid != "fed":
            return x
        f_ff = self.resolver.field("policy_rate_chg")
        if f_ff not in x:
            return x
        units = self.fed_units(nd.cloud.state, q, self.fed_pin(q))
        return {**x, f_ff: self.implied_moves(units, x[f_ff])}

    def _instrument_field(self, aid: str) -> str | None:
        roles = {w.role for o in A.ACTORS[aid].main.options for w in o.writes if w.kind == "point"}
        return next((self.resolver.field(r) for r in sorted(roles) if self.resolver.field(r)), None)

    # ---------------------------------------------------------------- payloads
    def payload(self, aid: str, oid: str, magnitude=None, custom: dict | None = None,
                commitments: list | None = None) -> dict:
        if oid == "as_committed":
            writes = [dict(cm["write"], option_of=cm.get("option")) for cm in (commitments or []) if cm.get("actor") == aid]
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

    @staticmethod
    def bin_edges(aid: str, kept: list) -> dict:
        """{option: (lo, hi]} covering the real line for the kept instrument options: adjacent options split at the
        midpoint of the nearest ends of their value sets (A.option_value_range), i.e. each A0 draw is labelled with
        the kept option whose decision is nearest. With every option kept these are the catalogue bins."""
        rows = []
        for o, lo, hi in A.instrument_bins(aid):
            if o not in kept:
                continue
            vr = A.option_value_range(aid, o) or (lo, hi)
            rows.append((o, float(vr[0]), float(vr[1]), lo))
        rows.sort(key=lambda t: (t[1], t[3]))
        out = {}
        for i, (o, vlo, vhi, _) in enumerate(rows):
            lo = -math.inf if i == 0 else 0.5 * (rows[i - 1][2] + vlo)
            hi = math.inf if i == len(rows) - 1 else 0.5 * (vhi + rows[i + 1][1])
            out[o] = (lo, hi)
        return out

    def a0_bins(self, aid: str, ids: list, a0x: Mapping, ok: np.ndarray | None = None,
                pinned: Mapping | None = None) -> dict | None:
        """The model's own prior: the share of A0 particles in each option's bin (bins cover the real line; tail
        particles, `ok` False, are not counted). A pinned first meeting (2026Q4) renormalizes over the compatible
        regular options: the A0 mass of the catalogue bins the pin rules out is dropped, as the probit prior does
        (priors.fed_pinned_renormalize), because the model cannot condition on the pinned decision."""
        bins = [(o, lo, hi) for o, lo, hi in A.instrument_bins(aid) if o in ids]
        if not bins:
            return None
        roles = {w.role for o in A.ACTORS[aid].main.options for w in o.writes if w.kind == "point"}
        fid = next((self.resolver.field(r) for r in sorted(roles) if self.resolver.field(r)), None)
        if fid is None or fid not in a0x:
            return None
        v = np.asarray(a0x[fid], float)
        if ok is not None and np.any(ok):
            v = v[np.asarray(ok, bool)]
        v = v[np.isfinite(v)]
        if not len(v):
            return None
        out = {"p": {}, "edges": {}, "field": fid, "source": "a0_bins", "label": "模型自身 A0 分箱", "kind": "model",
               "values": v}
        if pinned and aid == "fed":
            cat = self.bin_edges(aid, [o for o, _, _ in A.instrument_bins(aid)])      # the seven catalogue bins
            first = float(pinned.get("move", 0.0))
            okv = {round(first + s, 2) for s in (-0.25, 0.0, 0.25)}
            regular = [o for o in ids if o in A.FED_OPTION_VALUE and round(A.FED_OPTION_VALUE[o], 2) in okv]
            raw = {o: float(np.mean((v > cat[o][0]) & (v <= cat[o][1]))) for o in regular if o in cat}
            tot = sum(raw.values())
            if tot <= 0:
                return None
            out["p"] = {o: raw.get(o, 0.0) / tot for o in ids}
            out["edges"] = {o: cat[o] for o in regular if o in cat}
            out["pinned"] = dict(pinned)
            out["renormalized_mass"] = float(tot)
            out["label"] = (f"模型自身 A0 分箱：10 月会议锁定为 {pinned.get('option')}，在相容选项上重新归一"
                            f"（A0 中 {1 - tot:.0%} 的质量与锁定的决定不相容，已去掉）")
            return out
        edges = self.bin_edges(aid, [o for o, _, _ in bins])
        for o, (lo, hi) in edges.items():
            out["p"][o] = float(np.mean((v > lo) & (v <= hi)))
            out["edges"][o] = (lo, hi)
        for o in ids:
            out["p"].setdefault(o, 0.0)
        return out

    def a0_bin_magnitude(self, aid: str, o: Mapping, pr: Mapping) -> float | None:
        """A ranged option's default size under an A0-bin prior: the median of the A0 draws inside its bin, clipped
        to the option's range and step (the model's own typical size for that decision), else None."""
        par = o.get("param")
        edges = (pr.get("edges") or {}).get(o["id"])
        vals = pr.get("values")
        if not par or edges is None or vals is None:
            return None
        v = np.asarray(vals, float)
        sel = v[(v > edges[0]) & (v <= edges[1])]
        if len(sel) < A0_MAGNITUDE_MIN:              # a handful of particles is not a typical size (P14 review)
            return None
        lo, hi = float(min(par["lo"], par["hi"])), float(max(par["lo"], par["hi"]))
        m = float(np.clip(np.median(sel), lo, hi))
        if par.get("step"):
            m = float(np.clip(round(m / float(par["step"])) * float(par["step"]), lo, hi))
        return m

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
        ok = getattr(a0, "ok", None)
        hz = SY.hazard_masses({k: np.asarray(v)[ok] for k, v in a0.x.items() if not str(k).startswith("_")}
                              if ok is not None and np.any(ok) and not np.all(ok) else a0.x)
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
        if pr is None and self.prior_src == "a0" and (self.neutral or aid in A0_DECISION_ACTORS):
            # the model's own prior for decision instruments (the Fed's reaction function); the model tree reads
            # every instrument from A0 (exact neutrality). Not the President's tariff: the model's d_tariff drifts
            # with import composition and refunds (0.92 on tariff_down at the root), it is not a decision rule
            pinned = None if self.neutral else main.get("pinned_first")
            pr = self.a0_bins(aid, ids, self.instrument_view(nd, ctx.quarter, aid, a0.x), ok=getattr(a0, "ok", None),
                              pinned=pinned)
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
            mag_src = "default"
            if pr.get("source") == "a0_bins" and o.get("param") and aid in A0_DECISION_ACTORS:
                m_a0 = self.a0_bin_magnitude(aid, o, pr)
                if m_a0 is not None:
                    mag, mag_src = m_a0, "a0_bin_median"
            opts.append({"id": o["id"], "label": o.get("zh") or o["id"], "status_quo": bool(o.get("status_quo")),
                         "p_prior": float(merged.get(o["id"], 0.0)), "p_llm": None, "p_used": None, "magnitude": mag,
                         "magnitude_source": mag_src, "signal": None, "rationale": "", "source": "menu",
                         "param": o.get("param"), "bin": (pr.get("edges") or {}).get(o["id"]), "summary": o})
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
                "menu": m, "options": opts, "prior": {k: pr.get(k) for k in ("source", "label", "kind", "pinned")},
                "c": self.c_of(aid), "attributes": attrs, "fallback": False, "flags": [], "follower_of": None,
                "cond": None, "cond_prior": None, "answer": None, "required": list(main.get("required") or [])}

    def is_bin_actor(self, ab: Mapping) -> bool:
        """The actor's instrument is left to the model (no pin) and each particle carries the option whose bin its A0
        draw falls in: the model tree's instruments, and a prior-only actor (c used 0, no intervention) whose prior
        is the A0 bins without a pinned renormalization and whose every option is a bin (the base-rate tree's and a
        c = 0 agent tree's instruments: the model's own reaction function, AGENT_DESIGN §4.3 payload kind "bin";
        with common random numbers a particle nothing else pins keeps its very A0 draw)."""
        opts = ab.get("options") or []
        if not opts or not any(o.get("bin") for o in opts):
            return False
        if self.neutral:
            return True
        pr = ab.get("prior") or {}
        return (pr.get("source") == "a0_bins" and not pr.get("pinned") and float(ab.get("c_used") or 0.0) == 0.0
                and "intervened" not in (ab.get("flags") or []) and all(o.get("bin") for o in opts)
                and not any(o.get("source") == "custom" for o in opts))

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
        if pr.get("edges") and not pr.get("pinned"):
            # A0 bins: the kept bins cover the trimmed ones (each A0 draw goes to the kept option whose decision is
            # nearest, bin_edges), and each kept option's prior is the share of A0 particles in its bin (so masses
            # and particles agree exactly: neutrality)
            kept = [o for o in live if o in pr["edges"]]
            if len(kept) == len(live):
                new = self.bin_edges(aid, kept)
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
            for o in menu["main"]["options"]:          # an A0-bin prior's size: the model's median in the bin
                ao = next((x for x in ab["options"] if x["id"] == o["id"]), {})
                if ao.get("magnitude_source") == "a0_bin_median" and o.get("param"):
                    o["param"] = {**o["param"], "default": ao["magnitude"]}
                    o["note"] = ((o.get("note") or "") + f"（默认幅度 {ao['magnitude']:g} = 模型 A0 在该档内的中位数）").strip()
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
                                            self.board_text(nd, aid),
                                            dashboard_ref=f"（共享状态表见第一节；{self.target_text(nd)}）")
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
        if not profiles:                            # every branching option at zero mass: the prior's modal path
            profiles = [({}, 1.0)]
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
            # two heads with identical feature vectors (options that write the same fields) tie at distance 0 and
            # every profile goes to the first: the other head is empty, and its profile is merged into the first
            heads = [h for h in heads if h[1]]
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
        bin_actors = [aid for aid in dec["order"] if self.is_bin_actor(acts[aid])]
        by_bin = None
        if bin_actors:
            # a particle's bin option is the bin its own A0 draw falls in (the model tree's instruments; prior-only
            # instrument actors on an A0-bin prior): the instrument keeps the model's own draw, exactly
            keep = np.ones(n, bool)
            br_bin = [aid for aid in bin_actors if acts[aid].get("branching")]
            q = A.quarter_add(nd.quarter, 1)
            for aid in bin_actors:
                ab = acts[aid]
                view = self.instrument_view(nd, q, aid, a0.x)       # the Fed: the target moves its draws imply
                fid = (self.a0_bins(aid, [o["id"] for o in ab["options"]], view) or {}).get("field")
                v = np.asarray(view[fid], float)
                code = np.full(n, -1)
                for k, o in enumerate(ab["options"]):
                    lo, hi = o["bin"]
                    code[(v > lo) & (v <= hi)] = k
                if np.any(code < 0):              # a non-finite A0 draw (a tail particle): its nearest-mass bin
                    code[code < 0] = int(np.argmax([o["p_used"] for o in ab["options"]]))
                assign[aid] = code
            sig_of = lambda m: tuple([o["id"] for o in acts[a]["options"]].index(m["options"][a]) for a in br_bin)  # noqa: E731
            sigs = [sig_of(m) for m in cl["members"]]
            if br_bin:
                X = np.stack([assign[a] for a in br_bin], axis=1)
                keep = np.zeros(n, bool)
                for s in set(sigs):
                    keep |= np.all(X == np.array(s), axis=1)
            rows = np.flatnonzero(keep)
            member = np.zeros(len(rows), int)
            groups: dict = {}
            for i, s in enumerate(sigs):
                groups.setdefault(s, []).append(i)
            share = {}
            for s, mids in sorted(groups.items()):
                sel = np.arange(len(rows)) if not br_bin else \
                    np.flatnonzero(np.all(np.stack([assign[a][rows] for a in br_bin], axis=1) == np.array(s), axis=1))
                share[s] = float(len(sel)) / max(n, 1)
                if len(mids) == 1:
                    member[sel] = mids[0]
                    continue
                lab = np.repeat(np.array(mids), allocate([cl["members"][i]["p"] for i in mids], len(sel)))
                rng.shuffle(lab)
                member[sel] = lab
            # members' masses with the bin actors' joint A0 share in place of the product of their marginals
            marg = []
            for m in cl["members"]:
                pb = 1.0
                for a in br_bin:
                    pb *= float(next((o["p_used"] for o in acts[a]["options"] if o["id"] == m["options"][a]), 0.0))
                marg.append(pb)
            by_bin = {"actors": list(bin_actors), "branching": br_bin, "share": {str(list(s)): v for s, v in share.items()},
                      "p": [float(m["p"]) * (share[sigs[i]] / marg[i] if marg[i] > 0 else 0.0) if br_bin else float(m["p"])
                            for i, m in enumerate(cl["members"])]}
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
        return {"rows": rows, "assign": assign, "attr": attr, "member": member, "by_bin": by_bin}

    STRUCTURAL_PROTOCOLS = ("response_table", "thin")      # shift levers that are interventions (structural=)

    def _write_protocol(self, w: Mapping, pay: Mapping) -> str | None:
        """A write's conditioning protocol: its own (a commitment carries the option's that made it), else the
        payload's, else the catalogue's for the field (commitments saved before protocols travelled with them)."""
        return w.get("protocol") or pay.get("protocol") or (self._levers().get(w.get("field")) or {}).get("protocol")

    def _levers(self) -> dict:
        lv = self.__dict__.get("_lever_fields")
        if lv is None:
            lv = lever_fields(self.resolver)
            self.__dict__["_lever_fields"] = lv
        return lv

    def _intervenes(self, fid: str) -> bool:
        """Whether the model takes `fid` through structural= (a satellite it can set last, or a field with a policy
        shock): probed once per engine on two root particles; a model that refuses it gets the pin as an
        observation (the quarterly stack's core variables, e.g. the GZ spread)."""
        memo = self.__dict__.setdefault("_intervenes_memo", {})
        if fid not in memo:
            ok = bool(self.structural) and fid in self.model.x_fields
            if ok:
                try:
                    s = self.model.root_state(2)
                    self.model.complete(s, MA.Given.empty(2), np.random.default_rng(0), structural={fid: np.zeros(2)})
                except Exception:  # noqa: BLE001 - the model cannot take it as an intervention
                    ok = False
            memo[fid] = ok
        return memo[fid]

    def build_bk(self, nd: Node, cl: dict, dec: dict, a0, alloc: dict, rng: np.random.Generator) -> dict:
        """B_k: levers pinned per particle (only the particles an action wrote; the rest are free), then the model
        completes everything else, with A0's own random stream (common random numbers). Main options and attribute
        options (the Fed's balance sheet, immigration, the strategic reserve) write alike; a commitment due this
        quarter is routed by the protocol of the option that made it.

        Structural adapters (the quarterly stack): instrument points through `given` (the adapter's given_modes:
        d_ff and d_bs "do", d_tariff "do_last"); shift levers whose protocol is an intervention (Congress's fiscal
        components, the response_table protocol; thin levers such as the hyperscaler fallback on d_itinv and
        immigration on d_lf) through `structural=` (do by their policy shocks where the field has one, else
        do_last); the oil-supply composite and GPR contributions as reduced-form observations in `given`, with
        x_oil_cause = 1 (supply) where |Σz| ≥ 0.5; the hazard event fields an action did not write held at A0 by
        do_last. The Fed's target moves are pinned in the model's units (fed_units). Other adapters: everything in
        `given` and the two-step protocol of §6.3 (lag-protected E0 pinned at A0 for lagged and thin levers).

        Template channels (settings channels "on", channels.py): after the levers, each particle's due queue entries
        and this quarter's statements (shifts on its A0 draws in `given`, holds at A0, labels, displays set last on
        satellites), the room rule, the caps and the joint rarity; a d_ip statement is an increment on the IP the
        quarter's other pins give (two completions); the particle's next channel state rides in the draw's private
        columns. With channels "off" (P14): the response table of §6.3 (transfers -> goods consumption) and QE's G08
        intervention paths stand in for the levers the model does not transmit."""
        acts = dec["actors"]
        rows = alloc["rows"]
        n_all = nd.cloud.n
        m = len(rows)
        q = A.quarter_add(nd.quarter, 1)
        state = MA.take_state(nd.cloud.state, rows)
        a0x = {k: np.asarray(v)[rows] for k, v in a0.x.items() if not str(k).startswith("_")}
        nan = lambda: np.full(m, np.nan)                       # noqa: E731
        pins: dict = {}                    # given: reduced-form observations, or the adapter's given_modes
        spins: dict = {}                   # structural=: interventions (structural adapters only)
        shifts: dict = {}                  # field -> (Δ per particle, written mask, protocols)
        ev = {"oil_a": np.zeros(m), "oil_b2": np.zeros(m), "gpr_a": np.zeros(m), "gpr_b2": np.zeros(m)}
        protocols: set = set()
        written: dict = {}
        flags: list = []
        notes: list = []
        point_writes: dict = {}
        own_a0: set = set()                # instruments of prior-only bin actors: left to the model (not written)
        bin_actors = set((alloc.get("by_bin") or {}).get("actors") or [])
        qe_mask = np.zeros(m, bool)
        choices: list = []                 # what each particle chose, with its realized writes (template channels)
        chan = self.channels_on

        def apply(aid: str, o: Mapping, pay: Mapping, sel: np.ndarray, main: bool) -> None:
            if pay.get("protocol"):
                protocols.add(pay["protocol"])
            flags.extend(pay.get("flags") or [])
            realized = []
            for w in pay.get("writes", []):
                if int(w.get("dq", 0)) > 0 or w["kind"] == "label":
                    continue
                fid = w["field"]
                if fid not in self.model.x_fields:
                    continue
                prot = self._write_protocol(w, pay)
                if prot:
                    protocols.add(prot)
                if main and aid in bin_actors and o.get("bin") and w["kind"] in ("point", "bin", "normal"):
                    # a prior-only bin actor follows the model's own reaction function: its instrument is not
                    # pinned (a pin, even at the particle's own A0 value, is not neutral in the quarterly stack:
                    # its ±4 sd saturation and dollar hold re-shape the draw); each particle carries the option
                    # whose bin its A0 draw falls in, and with common random numbers keeps that very draw
                    own_a0.add(fid)
                    continue
                if w["kind"] == "bin":                # A0 particles inside the option's own bin, resampled
                    lo = -math.inf if w.get("lo") is None else float(w["lo"])
                    hi = math.inf if w.get("hi") is None else float(w["hi"])
                    base = np.asarray(a0x[fid], float)
                    pool = base[(base > lo) & (base <= hi) & np.isfinite(base)]
                    arr = pins.setdefault(fid, nan())
                    arr[sel] = rng.choice(pool, len(sel)) if len(pool) else np.clip(base[sel], lo, hi)
                    written.setdefault(fid, []).append(float(np.mean(arr[sel])))
                    point_writes.setdefault(fid, set()).add(aid)
                    realized.append({"field": fid, "kind": "bin", "value": arr[sel].copy(), "a": None,
                                     "option": w.get("option_of")})
                    continue
                written.setdefault(fid, []).append(float(w.get("a") or 0.0))
                if w["kind"] == "point":
                    arr = pins.setdefault(fid, nan())
                    arr[sel] = float(w["a"])
                    point_writes.setdefault(fid, set()).add(aid)
                    val = np.full(len(sel), float(w["a"]))
                elif w["kind"] == "normal":
                    arr = pins.setdefault(fid, nan())
                    arr[sel] = float(w["a"]) + float(w.get("b") or 0.0) * rng.standard_normal(len(sel))
                    point_writes.setdefault(fid, set()).add(aid)
                    val = arr[sel].copy()
                else:
                    d, mk, prs = shifts.setdefault(fid, (np.zeros(m), np.zeros(m, bool), set()))
                    val = float(w.get("a") or 0.0) + float(w.get("b") or 0.0) * rng.standard_normal(len(sel))
                    d[sel] += val
                    mk[sel] = True
                    prs.add(prot)
                realized.append({"field": fid, "kind": w["kind"], "value": np.asarray(val, float),
                                 "a": w.get("a"), "option": w.get("option_of")})
            for e in pay.get("events", {}).get("oil_z", []):
                ev["oil_a"][sel] += float(e["a"])
                ev["oil_b2"][sel] += float(e["b"]) ** 2
            for e in pay.get("events", {}).get("gpr", []):
                ev["gpr_a"][sel] += float(e["a_sd"])
                ev["gpr_b2"][sel] += float(e["b_sd"]) ** 2
            if chan:
                choices.append(CH.Choice(actor=aid, option=str(pay.get("option") or o.get("id")), rows=np.asarray(sel, int),
                                         magnitude=pay.get("magnitude"), writes=realized,
                                         facts=[f["id"] for f in pay.get("facts") or []],
                                         clears=list(pay.get("clears") or [])))

        for aid in dec["order"]:
            ab = acts[aid]
            code = alloc["assign"][aid]
            for k, o in enumerate(ab["options"]):
                sel = np.flatnonzero(code == k)
                if not len(sel):
                    continue
                if self.neutral and (o.get("bin") or o.get("status_quo")):
                    continue                              # the model tree: A0 draws (bins) and masked levers
                apply(aid, o, self._opt_payload(aid, o, nd), sel, True)
            for dd in ab["attributes"]:                   # attribute options write like main options
                codes = (alloc.get("attr") or {}).get((aid, dd["domain"]))
                if codes is None:
                    continue
                for k, oo in enumerate(dd["options"]):
                    sel = np.flatnonzero(np.asarray(codes) == k)
                    if not len(sel):
                        continue
                    pay = self.payload(aid, oo["id"])
                    if not (pay.get("writes") or pay["events"]["oil_z"] or pay["events"]["gpr"]):
                        if chan and (pay.get("facts") or pay.get("clears")) and not oo.get("status_quo"):
                            choices.append(CH.Choice(actor=aid, option=oo["id"], rows=np.asarray(sel, int),
                                                     magnitude=pay.get("magnitude"),
                                                     facts=[f["id"] for f in pay.get("facts") or []],
                                                     clears=list(pay.get("clears") or [])))
                        continue
                    apply(aid, oo, pay, sel, False)
                    if oo["id"] in QE_PATHS:
                        qe_mask[sel] = True
        # the Fed's decision (a quarter's target move): within-branch slopes (§6.6), the floor on the target, then
        # the model's units (fed_units: carry-over + the share of the moves the last month's average sees) and the
        # model's floor on its own level (the stub's 0, the stack's 0.1)
        f_ff = self.resolver.field("policy_rate_chg")
        surprise = None
        if f_ff in pins and "fed" in point_writes.get(f_ff, set()):
            mv = pins[f_ff]
            on = np.isfinite(mv)
            corr = np.zeros(m)
            sl = P.FED_SLOPES
            zs = {"us_equity_logret": state.get("us_equity_l1"), "d_unrate": state.get("d_unrate_l1"),
                  "core_gap": None if state.get("s_core_yoy") is None else state["s_core_yoy"] - 2.0,
                  "d_ff_lag": state.get("d_ff_l1"), "ff_level": state.get(FED_FLOOR_FIELD)}
            for key, v in zs.items():
                if v is None or key not in sl:
                    continue
                v = np.asarray(v, float)
                corr += float(sl[key]["b"]) * (v - float(np.median(v)))
            want = mv + np.clip(corr, -sl["clip_pp"], sl["clip_pp"])
            target = self.ff_target(state)
            move = np.maximum(want, self.ff_floor - target)            # the target cannot go below the floor
            units = self.fed_units(state, q, self.fed_pin(q))
            d_model = self.model_units(units, move)
            lo = self.ff_floor - np.asarray(state[FED_FLOOR_FIELD], float)
            d_model = np.maximum(d_model, lo)
            short = on & ((move - want > 1e-9) | (self.model_units(units, move) - lo < -1e-9))
            pins[f_ff] = np.where(on, d_model, np.nan)
            if short.any():
                flags.append("floor_short")
                notes.append({"field": f_ff, "code": "floor_engine", "particles": int(short.sum()),
                              "text": f"降息受模型的政策利率下限 {self.ff_floor:g}% 限制：交付的降幅小于选项（{int(short.sum())} 个粒子）"})
            # the Fed surprise each pinned particle adds (the price-puzzle overlay's trigger): pinned minus own A0
            if f_ff in a0x:
                surprise = np.where(on, pins[f_ff] - np.asarray(a0x[f_ff], float), 0.0)
        elif f_ff in pins:
            lo = self.ff_floor - np.asarray(state[FED_FLOOR_FIELD], float)
            ok = np.isfinite(pins[f_ff])
            pins[f_ff] = np.where(ok, np.maximum(pins[f_ff], lo), np.nan)
        # the event channels (§4.4): oil-supply composite and GPR contributions, as shifts on A0
        oil_a, oil_b2, gpr_a, gpr_b2 = ev["oil_a"], ev["oil_b2"], ev["gpr_a"], ev["gpr_b2"]
        oil_on = oil_a != 0
        oil_z = None
        if np.any(oil_on):
            beta = A.OIL_COMPOSITE["beta_metals_oil_per_z"]
            fid = self.resolver.field("oil_supply_signature")
            oil_z = np.where(oil_on, oil_a, 0.0)
            if fid in self.model.x_fields:
                d, mk, prs = shifts.setdefault(fid, (np.zeros(m), np.zeros(m, bool), set()))
                inc = np.where(oil_on, beta * oil_a + abs(beta) * np.sqrt(oil_b2) * rng.standard_normal(m), 0.0)
                d += inc
                mk |= oil_on
                prs.add("signature")
                written.setdefault(fid, []).append(float(np.mean(beta * oil_a)))
                oil_z = np.where(oil_on, inc / beta, 0.0)          # each particle's composite draw, in z
            lab = self.resolver.field("oil_cause_label")          # R-A1: 1 = supply (0 demand, 0.5 mixed)
            sup = np.abs(oil_a) >= A.OIL_COMPOSITE["cause_label_min_abs_z"]
            if lab in self.model.x_fields and np.any(sup):
                arr = pins.setdefault(lab, nan())
                arr[sup] = OV.OIL_CAUSE["supply"]
                written.setdefault(lab, []).append(OV.OIL_CAUSE["supply"])
        gpr_on = gpr_a != 0
        if np.any(gpr_on):
            fid = self.resolver.field("gpr_chg")
            if fid in self.model.x_fields:
                sd = self.sd.get(fid, 22.15)
                d, mk, prs = shifts.setdefault(fid, (np.zeros(m), np.zeros(m, bool), set()))
                d += np.where(gpr_on, gpr_a * sd + np.sqrt(gpr_b2) * sd * rng.standard_normal(m), 0.0)
                mk |= gpr_on
                # with template channels the geopolitical effect is the templates' (GEO, the blockade): the index is
                # set last on its satellite (do_last, nothing responds), so the exercise is display-only as mapped and
                # an escalation's activity effect is not counted twice; P14: a reduced-form observation as before
                prs.add("gpr_last" if (chan and self.structural) else "signature")
                written.setdefault(fid, []).append(float(np.mean(gpr_a * sd)))
        actor_points = {f for f in point_writes}
        for fid, (d, mk, prs) in shifts.items():
            base = pins.get(fid)
            base = np.where(np.isfinite(base), base, a0x[fid]) if base is not None else np.asarray(a0x[fid], float)
            v = np.asarray(base, float) + d
            h = self.support.get(fid) or {}
            if h.get("min") is not None and h.get("max") is not None:
                v = np.clip(v, float(h["min"]), float(h["max"]))
            v = np.where(mk, v, np.nan)
            if self.structural and ((prs & set(self.STRUCTURAL_PROTOCOLS) and "signature" not in prs)
                                    or prs == {"gpr_last"}):
                spins[fid] = v                                    # an intervention (do by its shock, else do_last)
            else:
                pins[fid] = np.where(mk, v, pins[fid]) if fid in pins else v
        response = []
        if not chan:
            # response tables (§6.3): transfers -> goods consumption, R = +2.1 per pp of personal income; on the
            # two-step protocol, and on a structural model whose transfer intervention reaches nothing
            # (transmission()), as an observation the model completes the quarter around
            for role, table in RESPONSE_TABLE.items():
                f_lev = self.resolver.field(role)
                if f_lev not in shifts or (self.structural and f_lev not in self.inert):
                    continue
                d, mk, _ = shifts[f_lev]
                for f_resp, R in table:
                    if f_resp in self.model.x_fields and f_resp not in pins and f_resp not in spins and mk.any():
                        pins[f_resp] = np.where(mk, np.asarray(a0x[f_resp], float) + R * d, np.nan)
                        written.setdefault(f_resp, []).append(float(np.mean(R * d[mk])))
                        response.append({"lever": f_lev, "field": f_resp, "R": R, "particles": int(mk.sum())})
            # QE's intervention paths (SIM_DESIGN §5.3, template G08) where the model's balance sheet reaches nothing
            f_bs = self.resolver.field("fed_bs_chg")
            if qe_mask.any() and f_bs in self.inert:
                for fid, a_sd, b_sd in QE_PATHS["bs_qe"]:
                    if fid not in self.model.x_fields:
                        continue
                    sd = self.sd.get(fid, 1.0)
                    v = np.asarray(a0x[fid], float) + (a_sd + b_sd * rng.standard_normal(m)) * sd
                    v = np.where(qe_mask, v, np.nan)
                    tgt = spins if self._intervenes(fid) else pins
                    tgt[fid] = np.where(qe_mask, v, tgt[fid]) if fid in tgt else v
                    written.setdefault(fid, []).append(float(a_sd * sd))
                notes.append({"code": "qe_paths", "particles": int(qe_mask.sum()), "text": QE_PATHS_NOTE})
        # the two-step protocol: lag-protected E0 at A0 for lagged / thin levers (unless the model is structural)
        if protocols & {"lagged", "thin"} and not self.structural:
            for fid in SY.LAG_PROTECTED:
                if fid in self.model.x_fields and fid not in pins:
                    pins[fid] = np.asarray(a0x[fid], float)
        hazard_fields = {f for _, (f, _, _) in SY.HAZARD_REGION.items()}
        lever_pins = {k: v.copy() for k, v in pins.items()}
        lever_spins = {k: v.copy() for k, v in spins.items()}
        # template channels: this quarter's statements and the due queue entries, per particle
        res = None
        if chan:
            xdate = None
            acc = nd.cloud.acc
            if acc is not None and "tga" in acc.s:
                xdate = np.asarray(acc.s["tga"], float)[rows] < float(acc.p.tga_floor)
            ch_rng = seed_for(self.st.seed, nd.key, f"channels|{cl.get('head_key', '')}")
            res = self.layer.step(state, a0x, choices, ch_rng, node_facts=nd.facts, oil_z=oil_z, xdate=xdate,
                                  surprise=surprise if self.overlay_on else None, overlay=self.overlay_on,
                                  literature=self.st.puzzle_literature_arm == "on")
            notes += res.notes
            # the cluster is beyond the templates' support (G1, the synthesizer's mechanisms) only when at least
            # half its particles are (判断值, the accounts' fact threshold); a minority stays in the notes
            if res.beyond is not None and float(np.mean(res.beyond)) >= self.BEYOND_FLAG_SHARE:
                flags.append(f"模板通道{CH.BEYOND_ZH}（疫情、封锁或越过 X 日期）")
        scale = np.ones(m)
        merged = self._merge_channels(res, lever_pins, lever_spins, a0x, actor_points, scale) if res is not None else None
        if merged is not None:
            pins, spins = merged["pins"], merged["spins"]
            # the joint rarity of the cluster's pins (P10): above the 99th percentile of the model's own draws, the
            # channel deltas (not the actors' levers) are scaled back together; beyond-history templates are exempt
            scale, rare = self._rarity_scale(nd, rows, state, res, lever_pins, lever_spins, a0x, actor_points, m)
            if rare:
                merged = self._merge_channels(res, lever_pins, lever_spins, a0x, actor_points, scale)
                pins, spins = merged["pins"], merged["spins"]
                notes.append({"code": "channel_rarity", "particles": int(rare),
                              "text": "模板通道与动作合起来的联合罕见度高于模型自身抽样的 99%：通道增量按比例缩回"})
        self._hold_hazards(pins, spins, a0x, hazard_fields, m)
        pins = {k: v for k, v in pins.items() if np.any(np.isfinite(v))}
        spins = {k: v for k, v in spins.items() if np.any(np.isfinite(v))}
        crn = False
        inc = self._increments(merged, res)
        given_p1 = MA.Given.from_dict(pins, m)
        pre = None
        if not (pins.keys() | spins.keys()) - hazard_fields and (self.neutral or self.structural) and not inc:
            # nothing but the hazard fields held at A0: B_k is A0 itself (exact; the hazard holds are A0's values)
            draws = MA.as_draws(a0.take(rows))
            given = MA.Given.from_dict({}, m)
            sgiven = MA.Given.from_dict({}, m)
            given_p1 = given
            crn = True
        else:
            sgiven = MA.Given.from_dict(spins, m)
            crn_box = [False]

            def complete(p):
                d, crn_box[0] = self.complete_crn(nd, rows, state, MA.Given.from_dict(p, m), sgiven, a0)
                return d
            # the passes (P7, P16): pass 1 without the increments, then IP's increments, then the inflation pass
            draws, final, pre = self._passes(complete, pins, inc, m)
            pins = final
            given = MA.Given.from_dict(pins, m)
            crn = crn_box[0]
        if merged is not None:
            self._post_displays(draws, merged["post"])
            for k, v in res.next.items():
                draws.x[CH.NEXT_PREFIX + k] = v
        f_def = self.resolver.field("deficit_chg")
        if f_def in shifts:                           # the books' fiscal baseline moves with discretionary actions
            d_def, mk_def, _ = shifts[f_def]
            draws.x[DEFICIT_ACTION_KEY] = np.where(mk_def, d_def, 0.0)
        # attribution (§7.7): what the channels add to the actions through the model (and the overlay's part); with
        # defer_attribution (the beam's levels) computed only for the clusters whose children survive the reduction
        chan_attr = None
        attr_job = None
        if res is not None and res.any():
            attr_job = self._attribution_job(nd, rows, a0, draws, pre, inc, lever_pins, lever_spins, a0x,
                                             hazard_fields, m, spins=spins)
            if not self.defer_attribution:
                chan_attr = attr_job()
                attr_job = None
        notes = merge_notes(list(getattr(draws, "notes", None) or []) + notes)
        # prior-only bin actors: the share of particles whose completed instrument left the bin they are labelled
        # with (other actors' reduced-form pins re-draw it), reported rather than relabelled (the cluster is its bins)
        drift = {}
        for aid in sorted(bin_actors & set(dec["order"])):
            ab = acts[aid]
            fid = self._instrument_field(aid)
            if fid is None or fid not in own_a0 or fid not in draws.x:
                continue
            v = np.asarray(draws.x[fid], float)
            if aid == "fed":
                v = self.implied_moves(self.fed_units(state, q, self.fed_pin(q)), v)
            code = np.asarray(alloc["assign"][aid], int)
            inb = np.zeros(m, bool)
            for k, o in enumerate(ab["options"]):
                if o.get("bin"):
                    lo, hi = o["bin"]
                    inb |= (code == k) & (v > lo) & (v <= hi)
            drift[aid] = rnd(float(np.mean(~inb)), 4) if m else 0.0
        wr = {f: float(np.mean(v)) for f, v in written.items()}
        lever_written = sorted(wr)
        ch_fields = sorted(res.fields & set(self.model.x_fields)) if res is not None else []
        out = {"rows": rows, "draws": draws, "given": given, "given_p1": given_p1, "inc": inc, "structural": sgiven,
               "written": sorted(set(lever_written) | set(ch_fields)), "lever_written": lever_written,
               "write_means": wr, "protocols": sorted(p for p in protocols if p), "flags": sorted(set(flags)),
               "state": state, "crn": crn, "n_parent": n_all, "own_a0": sorted(own_a0), "notes": notes,
               "bin_drift": drift, "response": response, "qe_particles": int(qe_mask.sum()) if not chan else 0}
        if res is not None:
            out.update(channels=self._channel_rows(res), channel_fields=ch_fields,
                       template_levers=sorted(res.covered), channel_signs=dict(res.signs),
                       post=merged["post"] if merged is not None else {}, channel_attr=chan_attr,
                       overlay_particles=int(res.overlay.sum()) if res.overlay is not None else 0,
                       channel_scaled=int(np.sum(scale < 1.0)))
            if attr_job is not None:
                out["_attr_job"] = attr_job
        return out

    # ---------------------------------------------------------------- the completions of B_k (P7, P16)
    def _increments(self, merged, res) -> dict:
        """The quarter's increments (not pinned in pass 1): {"ip": (Δ, mask)} of the d_ip statements (P7) and
        {"infl": {field: (Δ, mask)}} on the three inflation flows: the core-goods pass-through on d_core and the
        price-puzzle overlay's h·e on all three (P16). Empty without channels."""
        out: dict = {}
        if merged is None:
            return out
        if merged.get("ip") is not None:
            out["ip"] = merged["ip"]
        infl: dict = {}
        m = res.m
        if merged.get("core") is not None:
            infl["d_core"] = (np.asarray(merged["core"][0], float), np.asarray(merged["core"][1], bool))
            out["core_only"] = (np.where(infl["d_core"][1], infl["d_core"][0], 0.0), infl["d_core"][1].copy())
        for f, (d, mk) in (merged.get("overlay") or {}).items():
            d0, m0 = infl.get(f, (np.zeros(m), np.zeros(m, bool)))
            infl[f] = (d0 + np.where(mk, np.asarray(d, float), 0.0), m0 | np.asarray(mk, bool))
        infl = {f: v for f, v in infl.items() if f in self.model.x_fields and np.asarray(v[1]).any()}
        if infl:
            out["infl"] = infl
        return out

    @staticmethod
    def _take_inc(inc: Mapping, idx) -> dict:
        """The increments (Engine._increments) of the particles `idx`."""
        out: dict = {}
        if inc.get("ip") is not None:
            out["ip"] = (np.asarray(inc["ip"][0])[idx], np.asarray(inc["ip"][1])[idx])
        if inc.get("infl"):
            out["infl"] = {f: (np.asarray(d)[idx], np.asarray(mk)[idx]) for f, (d, mk) in inc["infl"].items()}
        return out

    def _passes(self, complete: Callable, pins: Mapping, inc: Mapping, m: int):
        """(draws, final pins, pre): the quarter's completions. Pass 1: the pins without the increments. Pass 2
        (P7, when there are IP increments): d_ip pinned at pass 1's IP + Δ. Inflation pass (P16, when there are
        inflation increments): for the particles with any, services, goods and core CPI pinned at the previous
        pass's values plus their increments (a d_core increment alone would impute services through the
        observation; holding the other two at their values keeps it on core). `pre` is (draws, pins) before the
        inflation pass (the overlay's attribution reuses it); `complete(pins) -> Draws`."""
        nan = np.full(m, np.nan)
        p = {k: np.asarray(v, float).copy() for k, v in pins.items()}
        d = complete(p)
        ip = (inc or {}).get("ip")
        if ip is not None and "d_ip" in d.x:
            dd, mk = ip
            p["d_ip"] = np.where(mk, np.asarray(d.x["d_ip"], float) + dd, p.get("d_ip", nan))
            d = complete(p)
        pre = (d, {k: v.copy() for k, v in p.items()})
        infl = (inc or {}).get("infl") or {}
        if infl:
            on = np.zeros(m, bool)
            for _, mk in infl.values():
                on |= np.asarray(mk, bool)
            for f in CH.INFLATION_FIELDS:
                if f not in d.x:
                    continue
                dd, _ = infl.get(f, (np.zeros(m), None))
                p[f] = np.where(on, np.asarray(d.x[f], float) + np.asarray(dd, float), p.get(f, nan))
            d = complete(p)
        return d, p, pre

    # ---------------------------------------------------------------- template channels in B_k
    def _merge_channels(self, res, pins: Mapping, spins: Mapping, a0x: Mapping, actor_points: set, scale) -> dict:
        """The cluster's lever pins with the channels' statements merged in (copies): shifts on each particle's
        A0 draw (or on an actor's own shift of the field) in `given`, scaled per particle by `scale` (the joint
        rarity rule); holds at A0 where nothing else wrote the field; labels where no actor labelled it; displays
        set last through `structural=` on satellites the model takes as interventions (other adapters: after the
        completion, `post`); the growth-field rule (a write on unemployment or GDP comes with an IP statement or an
        IP hold at A0); and the increments (returned, not pinned): d_ip's (P7), d_core's core-goods pass-through
        and the price-puzzle overlay's on the three inflation flows (P16; the overlay is never scaled by the joint
        rarity rule: it is not a channel of an action). An actor's instrument point always wins."""
        m = res.m
        nan = np.full(m, np.nan)
        P2 = {k: np.asarray(v, float).copy() for k, v in pins.items()}
        S2 = {k: np.asarray(v, float).copy() for k, v in spins.items()}
        post: dict = {}
        sc = np.asarray(scale, float)
        written = set()
        g = np.zeros(m, bool)                                  # particles a channel wrote a growth field for
        growth = ("d_unrate", "d_rgdp")
        for f, (d, mk) in res.shift.items():
            if f not in self.model.x_fields:
                continue
            d = np.asarray(d, float) * sc
            mk = mk.copy()
            if f in actor_points and f in P2:
                mk &= ~np.isfinite(P2[f])                      # an actor's point wins
            if f in S2:
                mk &= ~np.isfinite(S2[f])                      # an actor's intervention on the field wins
            if not mk.any():
                continue
            base = P2.get(f)
            base = np.where(np.isfinite(base), base, a0x[f]) if base is not None else np.asarray(a0x[f], float)
            P2[f] = np.where(mk, base + d, P2.get(f, nan))
            written.add(f)
            if f in growth:
                g |= mk
        ip = None
        if res.ip is not None and res.ip[1].any() and "d_ip" in self.model.x_fields:
            ip = (np.asarray(res.ip[0], float) * sc, res.ip[1].copy())
            if "d_ip" in P2:
                ip[1][np.isfinite(P2["d_ip"])] = False
            if not ip[1].any():
                ip = None
        core = None
        rc = getattr(res, "core", None)
        if rc is not None and rc[1].any() and "d_core" in self.model.x_fields:
            core = (np.asarray(rc[0], float) * sc, rc[1].copy())
            if "d_core" in P2:
                core[1][np.isfinite(P2["d_core"])] = False
            if not core[1].any():
                core = None
        for f, mk in res.hold.items():
            if f not in self.model.x_fields:
                continue
            free = mk & ~np.isfinite(P2.get(f, nan)) & ~np.isfinite(S2.get(f, nan))
            if f == "d_ip" and ip is not None:
                free &= ~ip[1]                                 # a hold yields to any write
            if free.any():
                P2[f] = np.where(free, np.asarray(a0x[f], float), P2.get(f, nan))
                written.add(f)
                if f in growth:
                    g |= free
        for f, vals in res.label.items():
            if f not in self.model.x_fields:
                continue
            free = np.isfinite(vals) & ~np.isfinite(P2.get(f, nan))
            if free.any():
                P2[f] = np.where(free, vals, P2.get(f, nan))
                written.add(f)
        for f, (d, mk) in res.display.items():
            if f not in self.model.x_fields:
                continue
            d = np.asarray(d, float)
            mk = mk & ~np.isfinite(P2.get(f, nan))             # an observation of the field wins over a display
            if not mk.any():
                continue
            if self.structural:
                if not self._intervenes(f):
                    continue
                base = S2.get(f)
                base = np.where(np.isfinite(base), base, a0x[f]) if base is not None else np.asarray(a0x[f], float)
                S2[f] = np.where(mk, base + d, S2.get(f, nan))
            else:
                post[f] = (np.asarray(a0x[f], float) + d, mk)
            written.add(f)
            if f in growth:
                g |= mk
        # the growth-field rule (P6): a channel write on unemployment or GDP, in any mode, holds IP at A0 unless
        # IP is stated (the quarterly stack's cycle module otherwise redraws IP around it); a particle with an IP
        # increment then gets it on A0 (pass 1 holds IP at A0, P7)
        if g.any() and "d_ip" in self.model.x_fields:
            need = g & ~np.isfinite(P2.get("d_ip", nan))
            if need.any():
                P2["d_ip"] = np.where(need, np.asarray(a0x["d_ip"], float), P2.get("d_ip", nan))
        return {"pins": P2, "spins": S2, "post": post, "ip": ip, "core": core, "written": written,
                "overlay": dict(getattr(res, "overlay_inc", None) or {})}

    def _hold_hazards(self, pins: dict, spins: dict, a0x: Mapping, hazard_fields: set, m: int) -> None:
        """Nature: the hazard event fields stay at their A0 draws unless an action wrote them (structural: do_last, so
        holding them moves nothing else in the quarter)."""
        for fid in sorted(hazard_fields - {"d_gz"}):
            if fid not in self.model.x_fields:
                continue
            free = ~np.isfinite(pins[fid]) if fid in pins else np.ones(m, bool)
            hold = np.where(free, np.asarray(a0x[fid], float), np.nan)
            if self.structural:
                spins[fid] = np.where(free, hold, spins[fid]) if fid in spins else hold
            else:
                pins[fid] = np.where(free, hold, pins[fid]) if fid in pins else hold

    @staticmethod
    def _post_displays(draws, post: Mapping) -> None:
        for f, (vals, mk) in (post or {}).items():
            if f in draws.x:
                draws.x[f] = np.where(mk, vals, np.asarray(draws.x[f], float))

    RARITY_SCREEN_SD = 1.0

    def _rarity_scale(self, nd: Node, rows, state, res, lever_pins, lever_spins, a0x, actor_points, m):
        """(per-particle scale of the channel deltas, particles scaled): the adapter's rarity percentile of the pins
        among its own draws (capability "rarity"); a particle above protocol rarity_pctl that is not beyond history
        steps its channels back (0.75, 0.5, 0.25, 0) until it is not. Other adapters: the χ² of the standardized
        channel shifts against the empirical p99 by number of fields (synthesizer.chi2_limits). P10 scales back what
        the channels add, so a particle that is as rare without them (its actions' own pins: a +2pp tariff, a big
        hike) keeps its channels. Only particles the channels moved by at least RARITY_SCREEN_SD jointly (the root
        sum of squares of their deltas in sd, 判断值) are tested, and each step re-tests only those still rare: the
        stack's rarity is per particle, its cost per distinct pin pattern, and a channel displacement under one sd
        cannot by itself carry a draw past the model's 99th percentile."""
        pr = self.layer.pr
        cut = float(pr.get("rarity_pctl", 0.99))
        steps = [float(x) for x in pr.get("rarity_steps") or (0.75, 0.5, 0.25, 0.0)]
        scale = np.ones(m)
        z2 = np.zeros(m)
        incs = [("d_ip", res.ip)] if res.ip is not None else []
        incs += [("d_core", res.core)] if getattr(res, "core", None) is not None else []
        for f, (d, mk) in list(res.shift.items()) + incs:
            z2 += np.where(mk, (np.nan_to_num(np.asarray(d, float)) / max(self.sd.get(f, 1.0), 1e-9)) ** 2, 0.0)
        exempt = res.beyond if res.beyond is not None else np.zeros(m, bool)
        cand = (z2 >= self.RARITY_SCREEN_SD ** 2) & ~exempt
        if not cand.any():
            return scale, 0
        rarity = getattr(self.model, "rarity_detail", None) if "rarity" in (getattr(self.model, "capabilities", ())
                                                                            or ()) else None

        def over(sc, idx):
            """The particles among idx (positions in the cluster) whose pins are rare at channel scale sc."""
            out = np.zeros(m, bool)
            idx = np.asarray(idx, int)
            if not len(idx):
                return out
            if rarity is not None:
                mg = self._merge_channels(res, lever_pins, lever_spins, a0x, actor_points, sc)
                p2 = dict(mg["pins"])
                for f_i, inc_i in (("d_ip", mg["ip"]), ("d_core", mg.get("core"))):
                    if inc_i is not None:              # the increments on A0 (an approximation of the passes)
                        p2[f_i] = np.where(inc_i[1], np.asarray(a0x[f_i], float) + inc_i[0],
                                           p2.get(f_i, np.full(m, np.nan)))
                whole = len(idx) == m
                pick = (lambda v: np.asarray(v, float)) if whole else (lambda v: np.asarray(v, float)[idx])
                g = {k: pick(v) for k, v in p2.items()}
                sd = {k: pick(v) for k, v in mg["spins"].items()}
                g = MA.Given.from_dict({k: v for k, v in g.items() if np.any(np.isfinite(v))}, len(idx))
                sd = {k: v for k, v in sd.items() if np.any(np.isfinite(v))}
                try:
                    det = rarity(state if whole else MA.take_state(state, idx), g, structural=sd or None)
                except Exception:  # noqa: BLE001 - no rarity: nothing is scaled
                    return out
                pct = np.asarray(det.get("percentile"), float)
                out[idx] = np.isfinite(pct) & (pct > cut)
                return out
            z2 = np.zeros(m)
            k = np.zeros(m)
            for f, (d, mk) in res.shift.items():
                z2 += np.where(mk, (np.asarray(d, float) * sc / max(self.sd.get(f, 1.0), 1e-9)) ** 2, 0.0)
                k += mk
            lim = np.array([SY.chi2_limits(int(max(kk, 1)))[1] for kk in k])
            out[idx] = ((k > 0) & (z2 > lim))[idx]
            return out
        bad = over(scale, np.flatnonzero(cand))
        if bad.any():
            bad &= ~over(np.zeros(m), np.flatnonzero(bad))
        for s in steps:
            if not bad.any():
                break
            scale = np.where(bad, s, scale)
            bad = over(scale, np.flatnonzero(bad & (scale > 0)))
        return scale, int(np.sum(scale < 1.0))

    ATTR_ASSETS = ("us_equity", "nasdaq", "treasury_10y", "gold", "crude_oil")

    def _attribution_job(self, nd, rows, a0, draws, pre, inc, lever_pins, lever_spins, a0x, hazard_fields, m,
                         spins=None):
        """The channel attribution of one cluster as a deferred computation (memoized): per asset, the median of B_k
        minus B_k completed with the actors' levers only (the 通道 component of §7.7) and, while the overlay runs,
        its own part (B_k minus B_k without the overlay's increments: 价格之谜修正; the passes before the inflation
        pass are B_k's own, so it costs one completion with core-goods increments and none without). Medians over
        the same particles, before the tail rule. With Engine.defer_attribution the beam runs it only for clusters
        with a child that survives the reduction (finish_attribution). `pre`: (draws, pins) before the inflation
        pass."""
        assets = [a for a in self.ATTR_ASSETS if a in draws.y]
        med = {a: float(np.median(draws.y[a])) for a in assets}
        infl = dict((inc or {}).get("infl") or {})
        ov = bool(self.overlay_on and pre is not None and infl)
        pre_med = pre_x = pre_pins = core = None
        if ov:
            pre_d, pre_pins = pre
            pre_med = {a: float(np.median(pre_d.y[a])) for a in assets}
            pre_x = {f: np.asarray(pre_d.x[f], float) for f in CH.INFLATION_FIELDS if f in pre_d.x}
            core = (inc.get("core_only") if inc else None)
        box: dict = {}

        def job() -> dict:
            if "out" in box:
                return box["out"]
            state = MA.take_state(nd.cloud.state, rows)

            def complete(pins, sp):
                p2 = {k: v for k, v in pins.items() if np.any(np.isfinite(v))}
                s2 = {k: v for k, v in sp.items() if np.any(np.isfinite(v))}
                self._hold_hazards(p2, s2, a0x, hazard_fields, m)
                d, _ = self.complete_crn(nd, rows, state, MA.Given.from_dict(p2, m), MA.Given.from_dict(s2, m), a0)
                return d
            try:
                d0 = complete(lever_pins, lever_spins)
            except Exception:  # noqa: BLE001 - attribution is information
                box["out"] = {}
                return box["out"]
            out = {"channels": {a: rnd(med[a] - float(np.median(d0.y[a])), 5) for a in assets}}
            if ov:
                try:
                    if core is None:
                        d1_med = pre_med                    # no core-goods increments: B_k without the overlay
                    else:
                        dc, on = core
                        p1 = {k: np.asarray(v, float).copy() for k, v in pre_pins.items()}
                        for f in CH.INFLATION_FIELDS:
                            if f in pre_x:
                                add = np.asarray(dc, float) if f == "d_core" else 0.0
                                p1[f] = np.where(on, pre_x[f] + add, p1.get(f, np.full(m, np.nan)))
                        d1 = complete(p1, spins or {})
                        d1_med = {a: float(np.median(d1.y[a])) for a in assets}
                    out["overlay"] = {a: rnd(med[a] - d1_med[a], 5) for a in assets}
                except Exception:  # noqa: BLE001
                    pass
            box["out"] = out
            return out
        return job

    def finish_attribution(self, nodes: Iterable) -> None:
        """Run the deferred channel attributions of `nodes` (the reduction's survivors) and write them into their
        summaries' attribution rows (通道（模板与文献） and 价格之谜修正). Pruned children keep the actions' total
        (their summary says the split was not computed)."""
        for nd in nodes:
            job = nd.__dict__.pop("_attr_job", None)
            if job is None:
                continue
            ca = job() or {}
            self._attr_rows((nd.summary or {}).get("attribution"), ca)
            if (nd.summary or {}).get("attribution") is not None:
                nd.summary.pop("attribution_split", None)

    @staticmethod
    def _attr_rows(attribution: Mapping | None, ca: Mapping) -> None:
        for a, row in (attribution or {}).items():
            # the template channels' part of the actions (B_k minus B_k without channels), and of that the
            # price-puzzle overlay's own part (§7.7 rows 通道（模板与文献） and 价格之谜修正)
            c = (ca.get("channels") or {}).get(a)
            if c is None:
                continue
            o = float((ca.get("overlay") or {}).get(a) or 0.0)
            row["channels"] = round(float(c) - o, 5)
            row["overlay"] = round(o, 5)
            row["actions"] = round(float(row["actions"]) - float(c), 5)

    MODE_RANK = {"shift": 0, "label": 1, "hold": 2, "display": 3}

    def _channel_rows(self, res, min_share: float = 0.005) -> list:
        """The cluster's applied statements for the detail and the UI: source (option, lever, oil, fact, event,
        queue, overlay), field, mode, the mean shift in the field's sd over the particles it applied to, the share of
        the cluster's particles, the quarter offsets, the templates; shifts first, then by size × share."""
        out = []
        for r in res.rows:
            if float(r.get("share") or 0.0) < min_share:
                continue
            out.append({k: (rnd(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        out.sort(key=lambda r: (r["when"] != "now", self.MODE_RANK.get(r["mode"], 9),
                                -abs(float(r.get("mean_sd") or 0.0)) * float(r.get("share") or 0.0), r["source"], r["field"]))
        return out

    def complete_crn(self, nd: Node, rows: np.ndarray, state: dict, given: MA.Given, sgiven: MA.Given, a0):
        """model.complete for the particles `rows` of the node with A0's own random stream (common random numbers):
        the same seed as A0 over the node's full particle set (pins NaN outside `rows`), then the rows taken, so
        each particle draws exactly the innovations it drew in A0. Returns (draws, crn used)."""
        n_all = nd.cloud.n
        rng = seed_for(self.st.seed, nd.key, "a0")
        kw = {"structural": sgiven.to_dict()} if (self.structural and len(sgiven.ids)) else {}
        if len(rows) == n_all and np.array_equal(rows, np.arange(n_all)):
            return MA.as_draws(self.model.complete(state, given, rng, **kw)), True
        def expand(g: MA.Given) -> MA.Given:
            vals = np.full((n_all, len(g.ids)), np.nan)
            vals[rows] = g.values
            return MA.Given(g.ids, vals)
        g_all = expand(given)
        kw_all = {"structural": expand(sgiven).to_dict()} if kw else {}
        d = self.model.complete(nd.cloud.state, g_all, rng, **kw_all)
        return MA.as_draws(d).take(rows), True

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
        a0 = self.a0(nd)
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
            by_bin = alloc.get("by_bin")
            if by_bin and by_bin["branching"]:
                # bins of A0 partition the particles: a member's mass carries the bin actors' joint A0 share (not the
                # product of their marginal bin shares); the model tree's clusters are exactly their particles' share
                for mm, pm in zip(cl["members"], by_bin["p"]):
                    mm["p"] = float(pm)
                cl["p"] = float(sum(by_bin["p"]))
                if not len(alloc["rows"]):
                    continue
            kept_clusters.append(cl)
            bk = self.build_bk(nd, cl, dec, a0, alloc, seed_for(self.st.seed, nd.key, f"bk|{cl['head_key']}"))
            bk["alloc"] = alloc
            bk = self.sanitize(nd, q, cl, bk)
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

    def a0(self, nd: Node):
        """A0 = model.complete with every lever masked (§6.2 step 2), with `ok`: False for its tail particles
        (MA.tail_mask), which the priors, nature's tail masses and the shown A0 bands do not count."""
        a0 = MA.as_draws(self.model.complete(nd.cloud.state, MA.Given.empty(nd.cloud.n),
                                             seed_for(self.st.seed, nd.key, "a0")))
        a0.ok = ~MA.tail_mask(nd.cloud.state, a0, self.tail_bounds, end_state=self.end_state(nd.cloud.state, a0))
        return a0

    @staticmethod
    def take_bk(bk: dict, idx) -> dict:
        """bk with every per-particle entry re-indexed by idx: rows, the draw, given, structural (the interventions),
        the state and the slots (alloc assign / attr / member). Everything that resamples a cluster's particles
        (sanitize, nature's mixture) goes through here, so pins and slots stay with the particle they belong to."""
        idx = np.asarray(idx, int)
        out = dict(bk)
        out["rows"] = np.asarray(bk["rows"])[idx]
        out["draws"] = bk["draws"].take(idx)
        out["given"] = bk["given"].take(idx)
        if bk.get("given_p1") is not None:
            out["given_p1"] = bk["given_p1"].take(idx)
        if bk.get("inc"):
            out["inc"] = Engine._take_inc(bk["inc"], idx)
        if bk.get("post"):
            out["post"] = {f: (np.asarray(v)[idx], np.asarray(mk)[idx]) for f, (v, mk) in bk["post"].items()}
        if bk.get("structural") is not None:
            out["structural"] = bk["structural"].take(idx)
        out["state"] = MA.take_state(bk["state"], idx)
        al = bk.get("alloc")
        if al is not None:
            m0 = len(bk["rows"])
            out["alloc"] = {**al, "assign": {a: np.asarray(v)[idx] for a, v in al["assign"].items()},
                            "attr": {a: np.asarray(v)[idx] for a, v in al["attr"].items()}}
            if al.get("member") is not None and len(al["member"]) == m0:
                out["alloc"]["member"] = np.asarray(al["member"])[idx]
        return out

    def end_state(self, state: Mapping, d) -> dict | None:
        """The rolled state the tail rule's level bounds read (CPI y/y, the policy rate, the 10-year yield); None if
        the adapter cannot roll this draw (the bounds then read start-of-quarter level + change)."""
        try:
            with np.errstate(all="ignore"):
                return self.model.update_state(MA.take_state(state, np.arange(len(state[FED_FLOOR_FIELD]))), d)
        except Exception:  # noqa: BLE001 - a draw with non-finite values may not roll; the draw-level checks remain
            return None

    def sanitize(self, nd: Node, q: str, cl: dict, bk: dict, *, rows_key: str = "") -> dict:
        """Tail particles of a cluster's draw (MA.tail_mask on the draw and its rolled state: not finite, or beyond
        the bounds) are dropped and replaced by particles resampled from the cluster's other ones (state, draw, pins
        and slots together, take_bk), so books, clouds and charts only see finite draws inside the bounds. The
        share goes to bk["tail"] (with the reasons; node summaries and details), the journal (tail_particles) and
        the level stats. If no particle is left, the draw is clipped to the bounds instead and the cluster is
        flagged tail_all."""
        d = bk["draws"]
        reasons = MA.tail_reasons(bk["state"], d, self.tail_bounds, end_state=self.end_state(bk["state"], d))
        bad = np.zeros(len(bk["rows"]), bool)
        for v in reasons.values():
            bad |= v
        m = len(bad)
        k = int(bad.sum())
        info = {"n": k, "of": m, "share": (k / m) if m else 0.0}
        if k:
            info["why"] = {r: int(v.sum()) for r, v in reasons.items() if v.any()}
        bk = dict(bk)
        bk["tail"] = info
        if k == 0:
            return bk
        good = np.flatnonzero(~bad)
        self.journal.emit("tail_particles", node=nd.key, level=nd.level, quarter=q, cluster=cl.get("id"),
                          n=k, of=m, share=rnd(k / m, 4), why=info["why"], rule=MA.TAIL_RULE)
        if not len(good):
            info["all"] = True
            bk["draws"] = clip_tail(bk["state"], d, self.tail_bounds)
            bk["flags"] = sorted(set(bk.get("flags") or []) | {"tail_all"})
            return bk
        rng = seed_for(self.st.seed, nd.key, f"tail|{cl.get('head_key', '')}{rows_key}")
        idx = np.arange(m)
        idx[bad] = good[systematic(np.ones(len(good)), k, rng)]
        out = self.take_bk(bk, idx)
        out["tail"] = info
        return out

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
            bk = self.take_bk(bk, idx)                 # every per-particle entry, the interventions included
        if best is not None:
            h, ph, mask = best
            nature_branch[cl["id"]] = {"hazard": h, "p": ph, "mask": mask}
            bk["nature"] = {"hazard": h, "p": ph}
        return bk

    THIN_FACTS = ("export_controls_level", "fed_removal_attempt", "fed_removal_allowed", "fed_pressure_public")
    BEYOND_FLAG_SHARE = 0.5

    @staticmethod
    def support_flags(bk: Mapping) -> list[str]:
        """The cluster's support flags that open G1: a lever outside the model's 1–99% range (罕见) or beyond its
        history (超出). A template lever's 罕见 is not one: its channel is the template's, whose support is the
        literature's episodes, not the satellite's own history (the judge's rule: mechanisms from a template lever
        remain only beyond the template's support); a value clipped at the history's extreme still is."""
        cov = set(bk.get("template_levers") or ())
        out = []
        for f in bk.get("flags") or []:
            if "罕见" not in f and "超出" not in f:
                continue
            fid = f.split(" = ", 1)[0] if " = " in f else None
            if fid in cov and "罕见" in f and "超出" not in f:
                continue
            out.append(f)
        return out

    def fact_covered(self, fact: str) -> bool:
        """A standing fact whose consequence a template channel states (with channels on): its channel exists, so
        G1 does not open for it and mechanisms caused by it weigh as "template"."""
        return bool(self.channels_on and fact in CH.covered_facts())

    def lever_class(self, fid: str, bk: Mapping) -> str:
        """The cause class of a lever write in a cluster: the overlay's (or "absent" for a lever the model does not
        transmit), "template" where a template channel covers every write of it in the cluster (G1 then does not
        open for its inertness)."""
        c = (self.meta.get(fid) or {}).get("channel_class") or "moderate"
        if c in ("thin", "absent") and fid in set(bk.get("template_levers") or ()):
            return "template"
        return c

    def gate(self, nd: Node, cl: dict, bk: dict, dec: dict, nature: dict | None) -> list[str]:
        support = self.support_flags(bk)
        thin = []
        for fid in bk.get("lever_written", bk["written"]):
            if self.lever_class(fid, bk) in ("thin", "absent"):
                thin.append(fid)
        for f in self.THIN_FACTS:
            if self.fact_covered(f):
                continue
            if any(fa["id"] == f for aid in dec["order"] for o in dec["actors"][aid]["options"]
                   for fa in self._opt_payload(aid, o, nd).get("facts", []) if o["p_used"] >= 0.5):
                thin.append(f)
        chi, k = SY.lever_rarity({f: v for f, v in bk["write_means"].items()
                                  if not (self.meta.get(f) or {}).get("enum")}, self.sd)     # labels are not levers
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
            specs.append(ClusterSpec(id=cl["id"], actions=acts, written_fields=bk.get("lever_written", bk["written"]),
                                     support=self.support_flags(bk)[:4], summary=summ,
                                     masses=ms, stress_share=rnd(stress, 3), regime=regime, facts=facts[:4],
                                     channels=[{"text": t} for t in self.channel_lines(bk)],
                                     channel_fields=list(bk.get("channel_fields") or [])))
        fields = SY.field_table(self.meta, sorted(all_axes, key=lambda t: t[0]))
        sector = AC.sector_summary(nd.cloud.acc, self.root_acc) if nd.cloud.acc is not None else ""
        events = [f"{v['zh']}（本季概率 {v['p']:.2f}）" for h, v in sorted(dec["hazards"].items())
                  if any((bks[c["id"]].get("nature") or {}).get("hazard") == h for c in clusters)]
        return SynthRequest(ctx=ctx_p, clusters=specs, fields=fields, channels=SY.channel_table(),
                            anchors=SY.DIRECTIONS, signatures=SY.SIGNATURES, sector_summary=sector,
                            nature_events=events)

    MODE_ZH = {"shift": "移动", "hold": "按住（当季不随其他字段补全）", "display": "展示（本季最后设定，不传导）", "label": "标签"}

    def option_channel(self, aid: str, oid: str) -> dict:
        """How an option reaches the fields the model reads, for the 动作 tab: kind "model" (the model transmits
        it: the Fed's rate), "template" (a template channel, with its ids, names and literature basis), "display"
        (a template decided it has no macro footprint: shown, not transmitted), "commitment", "custom" or "none"."""
        key = f"{aid}.{oid}"
        hit = self.__dict__.setdefault("_opt_channel", {}).get(key)
        if hit is not None:
            return hit
        names = CH.template_names()
        if oid == "as_committed":
            out = {"kind": "commitment", "zh": "如期执行：到期的写入按做出承诺的那个选项的通道传导（被取消的阶段不再写）"}
        elif oid == "custom":
            out = {"kind": "custom", "zh": "自定义选项：只有它所写杠杆自身的通道（关税按每个百分点），没有新闻冲击"}
        elif not self.channels_on:
            roles = set()
            try:
                roles = {w.role for w in A.ACTORS[aid].find(oid)[1].writes}
            except KeyError:
                pass
            fids = {self.resolver.field(r) for r in roles} - {None}
            if fids and not (fids & set(self.inert)):
                out = {"kind": "model", "zh": "模型自身：经季度模型传导"}
            elif fids:
                out = {"kind": "none", "zh": "卫星变量，模型不传导：只有后果合成器的机制（前几季）"}
            else:
                out = {"kind": "none", "zh": "不写字段：后果只经由合成器"}
        else:
            e = CH.option(key) or {}
            tm = [{"id": t, "zh": names.get(t, t)} for t in e.get("templates") or []]
            if e.get("status") == "statements":
                bases, lines = [], []
                for st_ in CH.statements_of(e):
                    b = str(st_.get("basis") or "").strip()
                    if b and b not in bases:
                        bases.append(b)
                    z = self.basis_zh(st_)
                    if z and z not in lines:
                        lines.append(z)
                out = {"kind": "display" if e.get("display_only") else "template", "zh": self.no_ids(e.get("zh") or ""),
                       "templates": tm, "basis": bases[:4], "basis_zh": lines,
                       "reading_zh": self.reading_zh(e.get("review_reading")),
                       "beyond_history": bool(e.get("beyond_history"))}
            elif e:
                out = {"kind": e.get("kind") or "none", "zh": self.no_ids(e.get("zh") or ""), "templates": tm}
            else:
                out = {"kind": "none", "zh": "没有通道条目"}
        self.__dict__["_opt_channel"][key] = out
        return out

    _ID_RE = re.compile(r"\b[GSPMFX]\d\d\b")
    _FIELD_RE = re.compile(r"\b[dsxu]_[a-z0-9_]+\b")
    _REF_RE = re.compile(r"(?:[A-Z][A-Za-z'’]+(?:[-–][A-Z][A-Za-z'’]+)*(?: et al\.)?|[A-Z]{2,6})(?: Economics)? (?:19|20)\d\d")
    MODE_WORD = {"shift": "移动", "hold": "按住", "display": "只展示", "label": "标签"}

    def no_ids(self, text: str) -> str:
        """A Chinese text without code ids: template ids become their names, field ids their Chinese names."""
        names = CH._NAMES if CH._NAMES is not None else CH.template_names()
        text = self._ID_RE.sub(lambda mt: f"“{names.get(mt.group(0), '模板')}”", str(text or ""))
        return self._FIELD_RE.sub(lambda mt: (self.meta.get(mt.group(0)) or {}).get("zh") or "字段", text)

    def basis_zh(self, st_: Mapping) -> str:
        """One statement in Chinese for the 动作 tab (no code ids): the field, how and how much (sd per unit of its
        trigger), when, and the literature it cites (author-year), the template footprint or 判断值."""
        f = str(st_.get("field") or "")
        fz = (self.meta.get(f) or {}).get("zh") or "字段"
        mode = str(st_.get("mode") or "")
        qs = sorted(int(q) for q in st_.get("q") or [0])
        if st_.get("cond") == "while_fact":
            when = "事实持续期间每季"
        elif qs == [0]:
            when = "本季"
        elif qs == list(range(qs[0], qs[-1] + 1)):
            when = f"本季至之后第 {qs[-1]} 季" if qs[0] == 0 else (f"之后第 {qs[0]}–{qs[-1]} 季" if len(qs) > 1
                                                                   else f"之后第 {qs[0]} 季")
        else:
            when = "之后第 " + "、".join(str(q) for q in qs) + " 季"
        size = ""
        if mode in ("shift", "display") and st_.get("size"):
            per = "（每单位触发量）" if st_.get("scale") == "lin" else ""
            size = f" {float(st_['size']):+.3g} 个标准差{per}"
        tail = f"，之后 {int(st_['decay'])} 季回吐" if st_.get("decay") else ""
        b = str(st_.get("basis") or "")
        refs = list(dict.fromkeys(self._REF_RE.findall(b)))
        if re.search(r"\d{4}Q\d", b):
            refs.append("历次季度足迹")
        if "判断值" in b:
            refs.append("判断值")
        if not refs:
            refs = ["模板足迹"] if self._ID_RE.search(b) else (["签名按住"] if mode == "hold" else ["判断值"])
        return f"{fz}：{self.MODE_WORD.get(mode, mode)}{size}（{when}{tail}）。依据：" + "；".join(refs[:4])

    @staticmethod
    def reading_zh(rd: Mapping | None) -> list[str]:
        """The stack's reading of an entry at its default intensity (channel_probe, stored in the table by the review
        round), in Chinese: impact, and core inflation with the Fed on the model's rule and with the rate on the
        path it would have taken without the action (the model's own rule carries the price puzzle's lags)."""
        if not rd or not rd.get("rule"):
            return []
        a, b = rd["rule"], rd.get("held") or {}
        f = lambda v, d=2: "—" if v is None else f"{float(v):+.{d}f}"  # noqa: E731
        out = [f"季度模型读数（默认强度）：当季美股 {f(a.get('eq_impact_pct'))}%、10 年期美债 {f(a.get('tsy_impact_pct'))}%；"
               f"第 4 季工业产出水平 {f(a.get('ip_q4_pct'))}%、CPI 同比 {f(a.get('cpi_h4'))}、核心 {f(a.get('core_h4'))} 个百分点"]
        if b:
            out.append(f"第 20 季核心 CPI 同比：美联储按模型规则反应 {f(a.get('core_h20'))}，利率沿不采取这一动作时的路径 "
                       f"{f(b.get('core_h20'))} 个百分点（差别是模型规则带着价格之谜的滞后效应）")
        return out

    def source_zh(self, src: str) -> str:
        """A channel row's source in words: the option, the lever, the oil composite, a standing fact, an event,
        the queue of earlier quarters' consequences or the price-puzzle overlay."""
        if src == "queue":
            return "此前动作排队到本季的后果"
        if src == "overlay":
            return "价格之谜修正"
        if src == "oil":
            return "石油供给合成量"
        if src.startswith("lever:"):
            return {"d_tariff": "关税杠杆（每个百分点）"}.get(src[6:], "杠杆")
        if src.startswith("event:"):
            ev = (CH.table().get("events") or {}).get(src[6:]) or {}
            return {"xdate_passed": "越过 X 日期", "pandemic": "严重疫情"}.get(src[6:], ev.get("key") or src[6:])
        if src.startswith("fact:"):
            return f"持续事实：{A.FACTS.get(src[5:], {}).get('zh', src[5:])}"
        aid, _, oid = src.partition(".")
        try:
            return f"{A.ACTORS[aid].zh}·{A.ACTORS[aid].find(oid)[1].zh}"
        except KeyError:
            return src

    def channel_lines(self, bk: Mapping, limit: int = 12) -> list[str]:
        """The cluster's template channels as short lines (the synthesizer's 模板通道 section): per field, mode and
        timing, the cluster-mean shift in the field's sd (over all its particles), the share it applied to and its
        two largest sources; this quarter's shifts first."""
        groups: dict = {}
        for r in bk.get("channels") or []:
            if r.get("source") == "overlay":            # an increment on inflation, not a channel-written field
                continue
            k = (r.get("field"), r.get("mode"), r.get("when"))
            g = groups.setdefault(k, {"net": 0.0, "share": 0.0, "src": [], "templates": []})
            sh = float(r.get("share") or 0.0)
            if isinstance(r.get("mean_sd"), (int, float)):
                g["net"] += float(r["mean_sd"]) * sh
            g["share"] = max(g["share"], sh)
            g["src"].append((abs(float(r.get("mean_sd") or 0.0)) * sh + sh * 1e-3, self.source_zh(str(r.get("source")))))
            g["templates"] += [t for t in r.get("templates") or [] if t not in g["templates"]]
        order = sorted(groups.items(), key=lambda kv: (kv[0][2] != "now", self.MODE_RANK.get(kv[0][1], 9),
                                                         -abs(kv[1]["net"]), str(kv[0][0])))
        out = []
        for (f, mode, when), g in order[:limit]:
            amount = f" 平均 {g['net']:+.2f} sd" if mode in ("shift", "display") else ""
            src = "、".join(dict.fromkeys(x for _, x in sorted(g["src"], reverse=True)[:2]))
            tm = "、".join(g["templates"][:3])
            out.append(f"{f} {self.MODE_ZH.get(mode, mode)}{amount}（{'本季' if when == 'now' else '之后几季（已排队）'}；"
                       f"至多 {g['share']:.0%} 的粒子；{src}" + (f"；模板 {tm}" if tm else "") + "）")
        return out

    def panic_ok(self, nd: Node, bk: dict) -> bool:
        return bool(self.st.stress_mode or nd.facts.get("financial_accident") or
                    (bk.get("nature") or {}).get("hazard") == "financial_accident")

    def axes_for(self, bk: dict, dec: dict) -> list[tuple[str, str]]:
        """Partition axes: every axis but the actors' own lever writes (channel-shifted axes stay: a partition is a
        region of B_k's own distribution, which a mean shift does not change)."""
        lw = bk.get("lever_written", bk["written"])
        bt = "bigtech" in dec["actors"] and any("d_itinv" == f for f in lw)
        avail = [k for k in bk["draws"].x if not str(k).startswith("_")] + (["mult"] if "mult" in bk["draws"].y else [])
        return SY.axis_list(avail, (self.st.synth or {}).get("axes"), written=lw, bigtech_wrote_itinv=bt)

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
            classes = [self.lever_class(f, bk) for f in bk.get("lever_written", bk["written"])]
            if self.support_flags(bk):
                classes.append("thin")
            for f in ("export_controls_level", "fed_removal_allowed", "fed_removal_attempt"):
                if nd.facts.get(f):
                    classes.append("template" if self.fact_covered(f) else "absent")
            regime = bool(SY.regime_lines(self.covariates(nd), nd.facts, nd.stress)) or bool(
                [f for f, v in nd.facts.items() if v and f in A.FACTS and A.FACTS[f].get("exit") is not None])
            br, r2 = SY.consequences(ans, xs, written=bk["written"], causes=causes, classes=classes, regime=regime,
                                     axes=self.axes_for(bk, {"actors": {}}), c_s=float(sy.get("c_s", SY.C_S)),
                                     c_m=float(sy.get("c_m", SY.C_M)), stress_mode=bool(self.st.stress_mode),
                                     panic=self.panic_ok(nd, bk), m_max=int(self.st.M_max),
                                     signs=bk.get("channel_signs"))
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
        mech_tail = None
        if b["kind"] == "mechanism":
            # the mechanism on B_k's first-pass pins, then the same increments as B_k (P7, P16): IP's and the
            # inflation pass (the core-goods pass-through and the price-puzzle overlay) on top of the mechanism
            pins = (bk.get("given_p1") or bk["given"]).take(idx).to_dict()
            mp = SY.mechanism_pins(b, d.x, rng)
            pins.update(mp)
            inc = self._take_inc(bk.get("inc") or {}, idx)
            if set(mp) & {"d_unrate", "d_rgdp"} and "d_ip" not in mp and "d_ip" in d.x:
                # the growth-field rule (P6): a mechanism on unemployment or GDP keeps IP at B_k's value (its IP
                # increment included; the quarterly stack's cycle module would otherwise switch off and redraw it)
                ip_now = pins.get("d_ip")
                keep = np.asarray(d.x["d_ip"], float)
                pins["d_ip"] = keep if ip_now is None else np.where(np.isfinite(ip_now), ip_now, keep)
                inc.pop("ip", None)
            if "d_ip" in mp:
                inc.pop("ip", None)
            sg = bk.get("structural")
            sgiven = sg.take(idx) if sg is not None else MA.Given.from_dict({}, len(idx))
            st_idx = MA.take_state(bk["state"], idx)
            src_rows = rows_all[idx]
            unique = len(np.unique(src_rows)) == len(src_rows)
            kw = {"structural": sgiven.to_dict()} if (self.structural and len(sgiven.ids)) else {}

            def complete(p, unique=unique, src_rows=src_rows, st_idx=st_idx, sgiven=sgiven, kw=kw):
                g = MA.Given.from_dict(p, len(idx))
                if unique:                                         # common random numbers with A0 and B_k
                    return self.complete_crn(nd, src_rows, st_idx, g, sgiven, a0)[0]
                return MA.as_draws(self.model.complete(st_idx, g, seed_for(self.st.seed, key0, "mech"), **kw))
            d, _, _ = self._passes(complete, pins, inc, len(idx))
            for k in [k for k in draws.x if k == DEFICIT_ACTION_KEY or str(k).startswith(CH.NEXT_PREFIX)]:
                d.x[k] = np.asarray(draws.x[k])[idx]           # the engine's private columns ride with the particle
            if bk.get("post"):
                self._post_displays(d, {f: (np.asarray(v)[idx], np.asarray(mk)[idx]) for f, (v, mk) in bk["post"].items()})
            bad = MA.tail_mask(st_idx, d, self.tail_bounds, end_state=self.end_state(st_idx, d))
            mech_tail = {"n": int(bad.sum()), "of": int(len(bad)), "share": float(bad.mean()) if len(bad) else 0.0}
            if bad.any():
                good = np.flatnonzero(~bad)
                self.journal.emit("tail_particles", node=nd.key, level=nd.level, quarter=q, cluster=cl.get("id"),
                                  branch=b.get("id"), n=int(bad.sum()), of=int(len(bad)),
                                  share=rnd(float(bad.mean()), 4), rule=MA.TAIL_RULE)
                if len(good):
                    loc = np.arange(len(idx))
                    loc[bad] = good[systematic(np.ones(len(good)), int(bad.sum()),
                                               seed_for(self.st.seed, key0, "tail"))]
                    idx, d = idx[loc], d.take(loc)
                else:
                    d = clip_tail(st_idx, d, self.tail_bounds)
                    mech_tail["all"] = True
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
            if bk.get("_attr_job") is not None:          # deferred: run for the reduction's survivors only
                ch.__dict__["_attr_job"] = bk["_attr_job"]
            tail = dict(bk.get("tail") or {"n": 0, "of": int(len(rows_all)), "share": 0.0})
            if mech_tail is not None:
                tail["mechanism"] = mech_tail
            if ch.cloud.__dict__.get("_tail_state"):
                tail["state"] = ch.cloud.__dict__.pop("_tail_state")
            # cumulated along the path: the chance that a path of this node was resampled at some quarter so far,
            # 1 − Π(1 − s) over its quarters (the per-quarter shares understate the truncation over 80 quarters)
            s_q = 1.0 - (1.0 - float(tail.get("share") or 0.0)) * (1.0 - float((mech_tail or {}).get("share") or 0.0))
            prev = float(((nd.summary or {}).get("tail") or {}).get("cum") or 0.0)
            tail["cum"] = 1.0 - (1.0 - prev) * (1.0 - s_q)
            ch.summary["tail"] = {k: (rnd(v, 4) if isinstance(v, float) else v) for k, v in tail.items()}
            if bk.get("notes"):
                ch.summary["notes"] = [{"kind": note_kind(nt), **{k: nt.get(k) for k in ("field", "particles")
                                                                    if nt.get(k) is not None}} for nt in bk["notes"][:6]]
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
        st_src = MA.take_state(parent.state, src)
        state = self.model.update_state(st_src, d)
        f_ff = self.resolver.field("policy_rate_chg")
        if f_ff in d.x:                       # the target after the quarter: the moves the delivered d_ff implies
            q = A.quarter_add(nd.quarter, 1)
            mv = self.implied_moves(self.fed_units(st_src, q, self.fed_pin(q)), d.x[f_ff])
            tgt = self.ff_target(st_src) + np.where(np.isfinite(mv), mv, 0.0)
            state[FF_TARGET_KEY] = np.maximum(tgt, float(self.ff_floor))
        if self.channels_on:
            for k in CH.STATE_KEYS:
                col = d.x.get(CH.NEXT_PREFIX + k)
                if col is not None:
                    state[k] = np.asarray(col, np.float32).copy()
                elif k in st_src:                              # a draw without the layer (a probe): one quarter older
                    state[k] = np.asarray(st_src[k], np.float32).copy()
        bad = MA.state_tail_mask(state)
        tail_state = None
        if bad.any() and not bad.all():
            # a safety net behind sanitize: a finite draw inside the bounds rolled into a non-finite state (not seen
            # in the measured builds). Those particles are replaced by resampled healthy ones of the same child.
            good = np.flatnonzero(~bad)
            loc = np.arange(n)
            loc[bad] = good[systematic(np.ones(len(good)), int(bad.sum()),
                                       seed_for(self.st.seed, nd.key, f"tail_state|{int(src[0])}|{n}"))]
            tail_state = {"n": int(bad.sum()), "share": float(bad.mean())}
            state, d, src = MA.take_state(state, loc), d.take(loc), src[loc]
            assign = {k: v[loc] for k, v in assign.items()}
            attr = {k: v[loc] for k, v in attr.items()}
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
            q_ret = MA.to_float32(np.stack([np.expm1(d.y[a]) for a in self.assets], axis=1))
            path = np.concatenate([parent.path[src], q_ret[:, None, :]], axis=1)
        out = Cloud(state, acc, dict(d.x), {a: d.y[a] for a in d.y}, cum, cpi, path, dict(d.inc))
        if tail_state is not None:
            out._tail_state = tail_state
        return out

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
        med = {k: float(np.median(v)) for k, v in ch.cloud.x.items() if not str(k).startswith("_") and np.ndim(v) == 1}
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
                                                                                            "s_infexp1",
                                                                                            FF_TARGET_KEY]).items()}
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
                                                self.ATTR_ASSETS)
            self._attr_rows(out["attribution"], bk.get("channel_attr") or {})
            out["inputs"] = self.inputs(cl, bk, branch)
        return out

    def inputs(self, cl: Cloud, bk: dict, branch: dict) -> dict:
        written = set(bk.get("lever_written", bk["written"]))
        chan = set(bk.get("channel_fields") or [])
        pinned = set(bk["given"].ids) | set(getattr(bk.get("structural"), "ids", ()) or ())
        fs = [f for f in self.model.x_fields if f in cl.x]
        Q = np.percentile(np.stack([cl.x[f] for f in fs], axis=1), [10, 50, 90], axis=0)
        out = {}
        for j, f in enumerate(fs):
            if f in written:
                src = "actor"
            elif f in chan:
                src = "channel"
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
                                     "writes": _writes_text(self._opt_payload(aid, o, nd)),
                                     "channel": self.option_channel(aid, o["id"])} for o in ab["options"]],
                        "cond": ab.get("cond"), "cond_prior": ab.get("cond_prior"),
                        "attributes": [{"domain": d["domain"], "zh": d["zh"],
                                        "options": [{"id": o["id"], "label": o["label"], "p": rnd(o["p"], 4),
                                                     "channel": self.option_channel(aid, o["id"])}
                                                    for o in d["options"]]} for d in ab["attributes"]]})
        cls = []
        names = CH.template_names()
        for cl in clusters:
            bk = bks[cl["id"]]
            x = bk["draws"].x
            cls.append({"id": cl["id"], "label": cl["label"], "p": rnd(cl["p"], 6), "head": cl["head"],
                        "members": [{"options": m["options"], "p": rnd(m["p"], 6)} for m in cl["members"]],
                        "written": bk["written"], "protocols": bk["protocols"], "flags": bk["flags"],
                        "n_particles": int(len(bk["rows"])), "nature": bk.get("nature"),
                        "tail": {k: (rnd(v, 4) if isinstance(v, float) else v) for k, v in (bk.get("tail") or {}).items()},
                        "crn": bool(bk.get("crn")), "notes": bk.get("notes") or [], "bin_drift": bk.get("bin_drift") or {},
                        "response": bk.get("response") or [], "qe_particles": int(bk.get("qe_particles") or 0),
                        "channels": [{**r, "source_zh": self.source_zh(str(r.get("source"))),
                                      "templates_zh": [names.get(t, "模板") for t in r.get("templates") or []]}
                                     for r in bk.get("channels") or []],
                        "channel_fields": bk.get("channel_fields") or [],
                        "template_levers": bk.get("template_levers") or [],
                        "overlay_particles": int(bk.get("overlay_particles") or 0),
                        "channel_scaled": int(bk.get("channel_scaled") or 0),
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
                "a0": a0_bands(a0),
                "model": {"kind": MA.adapter_kind(self.model), "structural": self.structural,
                          "ff_floor": self.ff_floor, "given_modes": dict(self.given_modes), "inert": list(self.inert),
                          "ff_units": {k: (rnd(v, 4) if isinstance(v, float) else v) for k, v in
                                       self.fed_units(nd.cloud.state, q, self.fed_pin(q)).items() if k != "carry"},
                          "ff_carry": rnd(float(np.median(self.fed_units(nd.cloud.state, q)["carry"])), 4)}}

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
        rows.append(self.target_text(nd) + "（s_ff 是季末月的月均；本季的动作是两次会议合计的目标变动）")
        cov = self.covariates(nd)
        rows.append(AC.regime_line(cov, nd.facts, A.quarter_add(nd.quarter, 1), cl.acc))
        return "\n".join(rows)

    def target_text(self, nd: Node) -> str:
        """The policy-rate target at the node's quarter end, as the packets and the state table show it: the target
        range (25bp steps) and the effective rate (the branch median)."""
        tgt = float(np.median(self.ff_target(nd.cloud.state)))
        lo = max(0.0, math.floor((tgt - 0.07) / FF_TARGET_STEP) * FF_TARGET_STEP)
        return f"政策利率目标（季末）：区间约 {lo:.2f}–{lo + FF_TARGET_STEP:.2f}%，有效利率 {AC.sig3(tgt)}%"

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


def a0_bands(a0) -> dict:
    """10/50/90% of A0's band fields over its healthy particles (a0.ok; all when none is healthy)."""
    ok = getattr(a0, "ok", None)
    if ok is None or not np.any(ok):
        ok = np.ones(a0.n, bool)
    out = {}
    for f in BAND_FIELDS:
        if f not in a0.x:
            continue
        v = np.asarray(a0.x[f], float)[ok]
        v = v[np.isfinite(v)]
        if len(v):
            out[f] = [rnd(np.percentile(v, 10), 4), rnd(np.median(v), 4), rnd(np.percentile(v, 90), 4)]
    return out


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

def node_vector(nd: Node, scales: Iterable | None = None) -> np.ndarray:
    st = nd.cloud.state if nd.cloud is not None else {}
    v = []
    for k, sd in (scales or KEY_STATE):
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


def tail_stats(cands: list[Node]) -> dict:
    """Per level: the mass-weighted share of tail particles the level's draws dropped and resampled (Engine.sanitize;
    the cluster's share, inherited by each child), the largest share of any candidate, and the cumulated share
    1 − Π(1 − s) along the candidates' paths (mass-weighted mean and maximum)."""
    tot = sum(c.p_path for c in cands) or 1.0
    tails = [((c.summary or {}).get("tail") or {}) for c in cands]
    shares = [float(t.get("share") or 0.0) for t in tails]
    cums = [float(t.get("cum") or 0.0) for t in tails]
    mean = sum(c.p_path * s_ for c, s_ in zip(cands, shares)) / tot
    cum = sum(c.p_path * s_ for c, s_ in zip(cands, cums)) / tot
    return {"tail_share": rnd(mean, 6), "tail_max": rnd(max(shares, default=0.0), 4), "tail_cum": rnd(cum, 6),
            "tail_cum_max": rnd(max(cums, default=0.0), 4)}


def ident_stats(cands: list[Node]) -> dict:
    """Per level: the books' accounting identities (I1–I10) the candidates failed: their mass-weighted share, their
    count and the largest residual (accounts.Accounts.check)."""
    tot = sum(c.p_path for c in cands) or 1.0
    ids = [((c.summary or {}).get("identities") or {}) for c in cands]
    bad = [(c, i) for c, i in zip(cands, ids) if i and i.get("ok") is False]
    return {"ident_fail": rnd(sum(c.p_path for c, _ in bad) / tot, 6), "ident_fail_n": len(bad),
            "ident_max": rnd(max((float(i.get("max") or 0.0) for i in ids if i), default=0.0), 3)}


def note_kind(nt: Mapping) -> str:
    """floor | bound | saturation | tail | qe_paths | floor_short | other: what an adapter or engine note says."""
    code, text = str(nt.get("code") or ""), str(nt.get("text") or "")
    if code == "floor_engine":
        return "floor_short"
    if code in ("tail", "qe_paths"):
        return code
    if "下限" in text:
        return "floor"
    if "下界" in text:
        return "bound"
    if "±4" in text or "截断" in text:
        return "saturation"
    return "other"


def notes_stats(cands: list[Node]) -> dict:
    """Per level: particles each kind of note covered (note_kind), summed over the level's distinct clusters."""
    seen, acc = set(), {}
    for c in cands:
        k = (c.parent, (c.cluster or {}).get("id"))
        if k in seen:
            continue
        seen.add(k)
        for nt in (c.summary or {}).get("notes") or []:
            kind = nt.get("kind") or note_kind(nt)
            acc[kind] = acc.get(kind, 0) + int(nt.get("particles") or 0)
    return {"model_notes": dict(sorted(acc.items()))} if acc else {}


def discrete_stats(cands: list[Node]) -> dict:
    """Per level: candidates created by the discrete split, and the mass-weighted share of particles whose own
    discrete state (in the worst column) differs from their node's (the approximation that remains, §8.1)."""
    tot = sum(c.p_path for c in cands) or 1.0
    split = sum(1 for c in cands if ((c.summary or {}).get("discrete") or {}).get("split"))
    mis = sum(c.p_path * float(((c.summary or {}).get("discrete") or {}).get("max_misstated") or 0.0) for c in cands)
    return {"discrete_split_candidates": int(split), "discrete_misstated": rnd(mis / tot, 4)}


def reduce_level(cands: list[Node], B: int, *, eps: float, fact_penalty: float, n: int, seed: int,
                 keep_all: bool = False, scales: Iterable | None = None) -> tuple[list[Node], list[Node], dict]:
    """Recombination, then forward selection to B nodes minimising the Kantorovich distance; every dropped node's
    mass and particles move to its nearest survivor (pooled by mass and resampled to n). Returns (kept, pruned,
    stats); pruned nodes keep their own mass and the reason. `scales`: the node-distance sds (key_scales)."""
    cands = sorted(cands, key=lambda c: c.key)
    if not cands:
        return [], [], {"candidates": 0, "kept": 0, "own_mass": 0.0, "reassigned": 0.0, "transport_error": 0.0,
                        "mass": 0.0}
    p = np.array([c.p_path for c in cands])
    X = np.array([node_vector(c, scales) for c in cands])
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
    """The versions that key every LLM cache entry and a folder's identity. model_adapter is the adapter's content
    identity (MA.identity: the quarterly stack's file-time digest replaced by a hash of the data's bytes);
    model_adapter_build is the version string the adapter reports (information only)."""
    adapter = adapter if adapter is not None else MA.load_adapter()
    from ..llm import agent_prompts
    return {"engine": ENGINE_VERSION, "catalogue": A.CATALOGUE_VERSION, "accounts": AC.ACCOUNTS_VERSION,
            "priors": P.PRIORS_VERSION, "model_adapter": MA.identity(adapter), "prompts": agent_prompts.PROMPT_VERSION,
            "channels": CH.version()}


# the versions that decide a tree's behaviour (the pricing model's identity is folder_adapter's): a folder built
# under other ones is not continued by this code (resume, extend), and an on-demand branch under it is flagged
BEHAVIOUR_VERSIONS = ("engine", "catalogue", "accounts", "priors", "prompts", "channels")
VERSION_ZH = {"engine": "推演引擎", "catalogue": "行为者目录", "accounts": "账户", "priors": "历史基率", "prompts": "提示词",
              "channels": "模板通道表"}


def version_drift(saved: Mapping | None, adapter=None, *, current: Mapping | None = None) -> list[str]:
    """The behaviour versions a folder recorded (its manifest or run.json "versions") that differ from this code's,
    in BEHAVIOUR_VERSIONS order; a version the folder did not record yet is not compared (the engine's is)."""
    saved = dict(saved or {})
    cur = dict(current or versions(adapter))
    return [k for k in BEHAVIOUR_VERSIONS if saved.get(k) and saved.get(k) != cur.get(k)]


def drift_zh(keys: Iterable[str]) -> str:
    return "、".join(VERSION_ZH.get(k, k) for k in keys)


def drift_refusal(keys: Iterable[str], what: str = "快照") -> str:
    return (f"这个{what}由另一版代码构建（{drift_zh(keys)}已改变）：继续构建会在同一棵树里混用两套规则，已拒绝；"
            f"请用当前版本重新构建一个{what}")


def lever_fields(resolver) -> dict[str, dict]:
    """{field: {"protocol", "kind", "actors", "options"}} of every catalogue write that is an action on the model
    (main and attribute options, commitments included; the hyperscaler fallback on d_itinv), resolved on `resolver`.
    Signature writes (the oil composite, GPR) are observations and are not listed."""
    out: dict = {}
    for aid, a in A.ACTORS.items():
        for d in a.domains:
            for o in d.options:
                if o.protocol in (None, "signature"):
                    continue
                for w in o.writes:
                    fid = resolver.field(w.role)
                    kind = w.kind
                    if fid is None and w.role == "hyperscaler_capex_g":
                        fid, kind = resolver.field("it_invest_g"), "shift"
                    if fid is None:
                        continue
                    row = out.setdefault(fid, {"protocol": o.protocol, "kind": kind, "actors": [], "options": []})
                    if aid not in row["actors"]:
                        row["actors"].append(aid)
                    row["options"].append(o.id)
    return out


def transmission(adapter, eng: Engine) -> dict:
    """What the model's interventions reach (structural models): for every lever field (lever_fields), the largest
    change, in each response field's sd, that a +1 sd intervention routed as the engine routes it (Engine.route)
    makes against a placebo pinned at the same particles' own draws in the same mode, with common random numbers, in
    TRANSMISSION_FIELDS and the main asset returns over the quarter and the next TRANSMISSION_QUARTERS − 1. "inert"
    below TRANSMISSION_TOL. Also the policy rate's own response (ff_irf: +50bp "do" against the placebo, 12
    quarters): core CPI y/y over quarters 4–8 and unemployment at quarter 12; a positive core response is the price
    puzzle the honesty card discloses. Cached per adapter identity; a failing probe is recorded, never raised."""
    key = (MA.identity(adapter), tuple(sorted(eng.given_modes.items())))
    with _TRANSMISSION_LOCK:
        hit = _TRANSMISSION.get(key)
    if hit is not None:
        return hit
    levers = lever_fields(eng.resolver)
    n = TRANSMISSION_PARTICLES
    out: dict = {"levers": {}, "fields": list(TRANSMISSION_FIELDS), "quarters": TRANSMISSION_QUARTERS, "particles": n,
                 "basis": "同一组随机数下，把杠杆在各粒子自己的季前抽样值上 +1 sd，与固定在原值的对照比较：动作当季与之后 "
                          f"{TRANSMISSION_QUARTERS - 1} 季，{len(TRANSMISSION_FIELDS)} 个宏观与金融字段和主要资产回报的最大变化"}
    try:
        s0 = adapter.root_state(n)
    except Exception as err:  # noqa: BLE001 - recorded
        out["error"] = f"{type(err).__name__}: {err}"
        return out
    empty = MA.Given.empty(n)

    def draw(state, given, structural, h):
        rng = np.random.default_rng(91_000 + h)
        kw = {"structural": structural} if structural else {}
        return MA.as_draws(adapter.complete(state, given, rng, **kw))

    def vec(d):
        cols = [np.asarray(d.x[f], float) / max(eng.sd.get(f, 1.0), 1e-9) for f in TRANSMISSION_FIELDS if f in d.x]
        cols += [np.asarray(d.y[a], float) / CUM_SCALE for a in TRANSMISSION_ASSETS if a in d.y]
        return np.stack(cols, axis=1) if cols else np.zeros((n, 0))

    def path(f, v, route, H):
        g = MA.Given.from_dict({f: v}, n) if route == "given" else empty
        sg = {f: v} if route == "structural" else None
        d = draw(s0, g, sg, 0)
        s, vs, states = s0, [vec(d)], []
        for h in range(1, H):
            s = adapter.update_state(s, d)
            states.append(s)
            d = draw(s, empty, None, h)
            vs.append(vec(d))
        states.append(adapter.update_state(s, d))
        return vs, states
    try:
        base = draw(s0, empty, None, 0)
    except Exception as err:  # noqa: BLE001 - recorded
        out["error"] = f"{type(err).__name__}: {err}"
        return out
    for fid, row in sorted(levers.items()):
        if fid not in adapter.x_fields or fid not in base.x:
            continue
        route = eng.route(fid, row["kind"], row["protocol"])
        v0 = np.asarray(base.x[fid], float)
        try:
            p0, _ = path(fid, v0, route, TRANSMISSION_QUARTERS)
            p1, _ = path(fid, v0 + eng.sd.get(fid, 1.0), route, TRANSMISSION_QUARTERS)
        except Exception as err:  # noqa: BLE001 - recorded
            out["levers"][fid] = {"route": route, "error": f"{type(err).__name__}: {str(err)[:120]}", "inert": False,
                                  **{k: row[k] for k in ("protocol", "actors")}}
            continue
        z = max((float(np.max(np.abs(a - b))) if a.size else 0.0) for a, b in zip(p1, p0))
        out["levers"][fid] = {"route": route, "max_z": rnd(z, 4), "inert": bool(z < TRANSMISSION_TOL),
                              "protocol": row["protocol"], "actors": row["actors"],
                              "mode": eng.given_modes.get(fid, "reduced") if route == "given" else "structural"}
    f_ff = eng.resolver.field("policy_rate_chg")
    if f_ff in base.x:
        try:
            v0 = np.asarray(base.x[f_ff], float)
            _, s_p = path(f_ff, v0, "given", 12)
            _, s_h = path(f_ff, v0 + 0.5, "given", 12)
            dif = lambda k, h: float(np.mean(np.asarray(s_h[h][k], float) - np.asarray(s_p[h][k], float)))  # noqa: E731
            core = float(np.mean([dif("s_core_yoy", h) for h in range(3, 8)])) if "s_core_yoy" in s_p[0] else None
            un = dif("s_unrate", 11) if "s_unrate" in s_p[0] else None
            out["ff_irf"] = {"hike": 0.5, "core_yoy_q4_8": rnd(core, 4), "unrate_q12": rnd(un, 4),
                             "puzzle": bool(core is not None and core > 0.02)}
        except Exception as err:  # noqa: BLE001 - recorded
            out["ff_irf"] = {"error": f"{type(err).__name__}: {str(err)[:120]}"}
    out["inert"] = sorted(f for f, r in out["levers"].items() if r.get("inert"))
    with _TRANSMISSION_LOCK:
        _TRANSMISSION[key] = out
    return out


CHANNEL_PROBE_PARTICLES = 400
CHANNEL_PROBE_QUARTERS = 12
_CHANNEL_PROBE: dict = {}
_CHANNEL_PROBE_LOCK = threading.Lock()


def channel_probe(adapter, eng: Engine, key: str, *, n: int | None = None, H: int | None = None,
                  seed: int = 91_000, fed: str = "rule", overlay_history: float = 0.0) -> dict:
    """One entry of the channel table ("actor.option", or "event:xdate_passed" / "event:pandemic") at its default
    intensity on the root, through the engine's own layer, merge and passes (Engine._merge_channels, _passes: holds,
    displays, the growth-field rule, IP's and the inflation increments), against a placebo with common random
    numbers, in the agent tree's configuration: commitments executed when due, standing facts living per particle
    with their exit hazards. fed "rule": the Fed pinned at its own A0 draw in the action quarter and free after (the
    model's reaction function: its systematic moves carry the price puzzle's lags, which the overlay does not
    neutralize); "own": pinned at its own A0 draw every quarter (no same-quarter response, no surprise: A0 still
    carries the rule's reaction to the state); "held": the treated arm's rate pinned at the placebo arm's realized
    path every quarter (no systematic response to the action at all; the agent tree's overlay sees that deviation from
    the rule as a surprise and neutralizes its inflation effect, which this reading does not apply). overlay_history: a Fed surprise of that many pp one quarter before the
    action, in both arms, so the overlay runs from the action quarter (a tiny one isolates the overlay's mechanics).
    Returns the stack reading the judge's mapping reports (impact %, IP level %, y/y and unemployment differences in
    pp, the largest response in sd over the first three quarters) and the per-quarter paths of the states. Cached per
    (identity, the channel table's version, the engine's channel settings, key, n, H, seed, fed, overlay_history)."""
    n = int(n or CHANNEL_PROBE_PARTICLES)
    H = int(H or CHANNEL_PROBE_QUARTERS)
    ck = (MA.identity(adapter), CH.version(), bool(eng.channels_on), bool(eng.layer is not None and eng.overlay_on),
          eng.st.puzzle_literature_arm, id(eng.layer.options) if eng.layer is not None else None,
          key, n, H, seed, fed, float(overlay_history))
    with _CHANNEL_PROBE_LOCK:
        if ck in _CHANNEL_PROBE:
            return _CHANNEL_PROBE[ck]
    if eng.layer is None:
        return {"error": "模板通道关闭"}
    empty = MA.Given.empty(n)
    ev_key = key.split(":", 1)[1] if key.startswith("event:") else None
    aid, _, oid = key.partition(".")
    pay = None if ev_key else A.option_payload(aid, oid, None, resolver=eng.resolver, support=eng.support)
    pre = (CH.option(key) or {}).get("cond_fact")          # e.g. the court's reversal needs a live removal attempt
    pres = [pre] if pre else []
    if ev_key:                                             # the X-date escalates a live debt-limit standoff
        pres += list(((CH.table().get("events") or {}).get(ev_key) or {}).get("requires_live") or [])

    ff_path: dict = {}

    def run(on: bool):
        s = adapter.root_state(n)
        s[FF_TARGET_KEY] = np.full(n, eng.root_ff_target(s))
        s.update(eng.layer.empty_state(n))
        for pf in pres:                                     # the condition holds in both arms (common random numbers)
            if pf in eng.layer.facts:
                s[CH.FACT_KEY][:, eng.layer.facts.index(pf)] = 1.0
        if overlay_history:
            s[CH.FFS_KEY][:, 0] = float(overlay_history)
        cum = {a: np.zeros(n) for a in TRANSMISSION_ASSETS}
        out = []
        for h in range(H):
            a0 = MA.as_draws(adapter.complete(s, empty, np.random.default_rng(seed + h)))
            a0x = {k: np.asarray(v, float) for k, v in a0.x.items() if not str(k).startswith("_")}
            pins: dict = {}
            spins: dict = {}
            if (h == 0 or fed == "own") and "d_ff" in a0x:
                pins["d_ff"] = a0x["d_ff"].copy()
            if fed == "held" and on and h in ff_path:
                pins["d_ff"] = ff_path[h].copy()                    # the placebo's own rate path
            choices, oil_z, points = [], None, {"d_ff"}
            wr = np.random.default_rng(seed * 7 + h)
            if on and pay is not None:
                realized = []
                for w in pay["writes"]:
                    if int(w.get("dq", 0)) != h or w["kind"] in ("label", "bin") or w["field"] not in adapter.x_fields:
                        continue
                    fid = w["field"]
                    if w["kind"] == "point":
                        val = np.full(n, float(w["a"]))
                        pins[fid] = val
                        points.add(fid)
                    elif w["kind"] == "normal":
                        val = float(w["a"]) + float(w.get("b") or 0.0) * wr.standard_normal(n)
                        pins[fid] = val
                        points.add(fid)
                    else:
                        val = float(w["a"]) + float(w.get("b") or 0.0) * wr.standard_normal(n)
                        prot = eng._write_protocol(w, pay)
                        if eng.structural and prot in Engine.STRUCTURAL_PROTOCOLS:
                            spins[fid] = a0x[fid] + val
                        else:
                            pins[fid] = a0x[fid] + val
                    realized.append({"field": fid, "kind": w["kind"], "value": val, "a": w.get("a"), "option": oid})
                if h == 0 or realized:
                    choices.append(CH.Choice(actor=aid, option=oid if h == 0 else "as_committed", rows=np.arange(n),
                                             magnitude=pay.get("magnitude"), writes=realized,
                                             facts=[f["id"] for f in pay.get("facts") or []] if h == 0 else [],
                                             clears=list(pay.get("clears") or []) if h == 0 else []))
                if h == 0 and pay["events"]["oil_z"]:
                    za = sum(float(e["a"]) for e in pay["events"]["oil_z"])
                    zb = math.sqrt(sum(float(e["b"]) ** 2 for e in pay["events"]["oil_z"]))
                    oil_z = za + zb * wr.standard_normal(n)
                    beta = A.OIL_COMPOSITE["beta_metals_oil_per_z"]
                    if "d_metals_oil" in a0x:
                        pins["d_metals_oil"] = a0x["d_metals_oil"] + beta * oil_z
                    if "x_oil_cause" in a0x and abs(za) >= A.OIL_COMPOSITE["cause_label_min_abs_z"]:
                        pins["x_oil_cause"] = np.full(n, OV.OIL_CAUSE["supply"])
                if h == 0 and pay["events"]["gpr"] and "d_gpr" in a0x:
                    g = sum(float(e["a_sd"]) for e in pay["events"]["gpr"])
                    pins["d_gpr"] = a0x["d_gpr"] + g * eng.sd.get("d_gpr", 22.15)
            facts = {"pandemic": True} if (on and ev_key == "pandemic") else {}
            xdate = np.full(n, bool(on and ev_key == "xdate_passed" and h == 0))
            res = eng.layer.step(s, a0x, choices, np.random.default_rng(seed * 13 + h), node_facts=facts, oil_z=oil_z,
                                 xdate=xdate, surprise=np.zeros(n), overlay=eng.overlay_on,
                                 literature=eng.st.puzzle_literature_arm == "on")
            mg = eng._merge_channels(res, pins, spins, a0x, points, np.ones(n))
            P2, S2 = mg["pins"], mg["spins"]
            eng._hold_hazards(P2, S2, a0x, {f for _, (f, _, _) in SY.HAZARD_REGION.items()}, n)
            P2 = {k: v for k, v in P2.items() if np.any(np.isfinite(v))}
            S2 = {k: v for k, v in S2.items() if np.any(np.isfinite(v))}
            kw = {"structural": S2} if (eng.structural and S2) else {}

            def complete(p, s=s, h=h, kw=kw):
                return MA.as_draws(adapter.complete(s, MA.Given.from_dict(p, n), np.random.default_rng(seed + h), **kw))
            d, _, _ = eng._passes(complete, P2, eng._increments(mg, res), n)
            eng._post_displays(d, mg["post"])
            if not on and "d_ff" in d.x:
                ff_path[h] = np.asarray(d.x["d_ff"], float).copy()       # the placebo's realized rate path
            for a in cum:
                if a in d.y:
                    cum[a] = cum[a] + np.asarray(d.y[a], float)
            s = adapter.update_state(s, d)
            s.update({k: np.asarray(v, np.float32) for k, v in res.next.items()})
            out.append({"x": {f: np.asarray(d.x[f], float) for f in set(TRANSMISSION_FIELDS) | {"d_ip", "d_ff"}
                              if f in d.x},
                        "y": {a: np.asarray(d.y[a], float) for a in TRANSMISSION_ASSETS if a in d.y},
                        "s": {k: np.asarray(s[k], float) for k in ("s_core_yoy", "s_cpi_yoy", "s_unrate", "s_ff")
                              if k in s}, "cum": {a: v.copy() for a, v in cum.items()}})
        return out
    try:
        r0, r1 = run(False), run(True)
    except Exception as err:  # noqa: BLE001 - recorded, never raised
        out = {"error": f"{type(err).__name__}: {str(err)[:160]}"}
    else:
        mean = lambda v: float(np.nanmean(v))  # noqa: E731
        z = 0.0
        for h in range(min(3, H)):
            for f in TRANSMISSION_FIELDS:
                if f in r1[h]["x"]:
                    z = max(z, abs(mean(r1[h]["x"][f] - r0[h]["x"][f])) / max(eng.sd.get(f, 1.0), 1e-9))
            for a in TRANSMISSION_ASSETS:
                if a in r1[h]["y"]:
                    z = max(z, abs(mean(r1[h]["y"][a] - r0[h]["y"][a])) / CUM_SCALE)
        ipl = np.cumsum([mean(r1[h]["x"]["d_ip"] - r0[h]["x"]["d_ip"]) for h in range(H)]) if "d_ip" in r1[0]["x"] else None

        def dyn(h):
            return {k: rnd(mean(r1[h - 1]["s"][k] - r0[h - 1]["s"][k]), 4) for k in r1[h - 1]["s"]}
        out = {"key": key, "particles": n, "quarters": H, "max_abs_3q_sd": rnd(z, 4), "transmits": bool(z >= TRANSMISSION_TOL),
               "impact_pct": {a: rnd(100 * mean(r1[0]["y"][a] - r0[0]["y"][a]), 4) for a in r1[0]["y"]},
               "impact_sd": {f: rnd(mean(r1[0]["x"][f] - r0[0]["x"][f]) / max(eng.sd.get(f, 1.0), 1e-9), 4)
                             for f in r1[0]["x"]},
               "ip_level_pct": ({f"q{h}": rnd(float(ipl[h - 1]), 4) for h in (4, 8) if h <= H} if ipl is not None else {}),
               "h4": dyn(min(4, H)), "h8": dyn(min(8, H)), "h12": dyn(min(12, H)),
               "us_equity_cum_pct_h8": rnd(100 * mean(r1[min(8, H) - 1]["cum"]["us_equity"] - r0[min(8, H) - 1]["cum"]["us_equity"]), 4)
               if "us_equity" in r1[0]["cum"] else None,
               # quarters 1-8, the effect on each response field in sd and on cumulated US equity in %
               "path_sd": {f: [rnd(mean(r1[h]["x"][f] - r0[h]["x"][f]) / max(eng.sd.get(f, 1.0), 1e-9), 4)
                               for h in range(min(8, H))] for f in TRANSMISSION_FIELDS if f in r1[0]["x"]},
               "us_equity_cum_pct": ([rnd(100 * mean(r1[h]["cum"]["us_equity"] - r0[h]["cum"]["us_equity"]), 4)
                                      for h in range(min(8, H))] if "us_equity" in r1[0]["cum"] else []),
               # every quarter: the states' differences (pp) and the IP level (%)
               "s_path": {k: [rnd(mean(r1[h]["s"][k] - r0[h]["s"][k]), 4) for h in range(H)] for k in r1[0]["s"]},
               "ip_level_path": [rnd(float(v), 4) for v in ipl] if ipl is not None else [],
               "fed": fed, "overlay_history": float(overlay_history)}
    with _CHANNEL_PROBE_LOCK:
        _CHANNEL_PROBE[ck] = out
    return out


def transmission_zh(tr: Mapping | None, meta: Mapping | None = None, *, channels: Mapping | None = None,
                    root: Mapping | None = None) -> list[str]:
    """Plain Chinese lines for the honesty card and the CLI (no code ids): which levers the pricing model does not
    transmit and what stands in for them (with template channels: the channels; P14: the response table and QE's
    paths), the policy rate's price puzzle when the probe found one and what the engine does about it, and the
    model's own hawkish prior for the first quarter with the states behind it (`root`, root_prior)."""
    tr = dict(tr or {})
    ch = dict(channels or {})
    if not tr.get("levers"):
        return []
    zh = lambda f: ((meta or {}).get(f) or {}).get("zh") or f  # noqa: E731
    lines = []
    inert = [f for f in tr.get("inert") or [] if f != "d_ff"]
    if inert and ch.get("on"):
        lines.append("这个季度模型里，" + "、".join(zh(f) for f in inert) + "是卫星变量：模型按它们的政策冲击或在季末写入，"
                     "但它们不反馈到宏观核心与资产价格（当季和之后都一样）。所以这些动作的效果改由模板通道写入：代码按历史"
                     "模板与文献，把每个动作分解成模型能读的通用通道（商品与服务通胀、工业产出、失业率、10 年期收益率、"
                     "信用利差、美元、油价），逐季、逐粒子写入，之后各季同样有效；后果合成器不再重复写这些字段。美联储的"
                     "利率决定由模型自身传导。")
        cov = ch.get("coverage") or {}
        if cov:
            dormant = (f"（另有停用的参与者的 {cov['dormant']} 个选项保留了通道，不会出现）" if cov.get("dormant") else "")
            lines.append(f"目录里 {cov.get('template', 0)} 个动作有模板通道（其中 {cov.get('display', 0)} 个历史上没有宏观或"
                         f"金融足迹：只展示，诚实地不传导；{cov.get('beyond', 0)} 个超出可测历史，不受上限约束并标出）；"
                         f"美联储的 {cov.get('model', 0)} 个利率选项由模型自身传导；现状与没有通用足迹的 "
                         f"{cov.get('none', 0)} 个选项不写后果，每个都写明了理由{dormant}。另有越过 X 日期（只在国会把"
                         "债务上限对峙拖过期限时）与严重疫情两个事件模板。")
        lines.append("模板通道只用于智能体树。历史基率树按历史频率抽取动作，模型的残差本身已包含这些动作的平均效果，"
                     "再加通道就重复计算了（复核时测得：基率树加通道后第 20 季 GZ 利差 3.02，不加 1.66），所以基率树不写"
                     "通道，与模型树的差别只在动作的标签与概率。")
    elif inert:
        lines.append("这个季度模型里，" + "、".join(zh(f) for f in inert) + "是卫星变量：模型按它们的政策冲击或在季末写入，"
                     "但它们不反馈到宏观核心与资产价格（当季和之后都一样）。所以它们的效果改由后果合成器的机制写（前几季、"
                     "通道记为缺失），转移支付按响应表写入商品消费（每个百分点 +2.1%，模型围绕它补全当季），"
                     "量化宽松按 2008–14 年的足迹写入期限溢价、2 年期收益率与 GZ 利差的干预路径。")
    ff = tr.get("ff_irf") or {}
    if ff.get("puzzle"):
        un = ff.get("unrate_q12")
        head = (f"价格之谜：在这个模型里，加息 50 个基点（干预）后第 4–8 季核心 CPI 同比平均反而高 "
                f"{float(ff.get('core_yoy_q4_8') or 0.0):.2f} 个百分点"
                + (f"（第 12 季失业率 {float(un):+.2f} 个百分点）" if un is not None else ""))
        if ch.get("overlay"):
            lines.append(head + "。在模型重新识别货币政策冲击之前，推演用一个中性修正抵消它：意外的加息或降息对通胀的作用"
                         "设为零（不是文献里的去通胀效应），对失业率与资产价格的作用照常；修正叠加在当季其他动作的通胀效果"
                         "之上，不抹掉它们。代价是：通胀因别的原因偏高时，一个按通胀加息的美联储会过度收紧——失业率上升，"
                         "通胀并不因此下降；同样，一个被施压而降息的美联储，降息本身也不推高通胀。修正只认美联储决定相对"
                         "模型自身抽样的意外，同季其他动作改变的通胀让模型隐含的那部分意外它看不到。")
            xr = ((CH.table().get("events") or {}).get("xdate_passed") or {}).get("review_reading") or {}
            ex = ""
            if (xr.get("rule") or {}).get("core_h20") is not None and (xr.get("held") or {}).get("core_h20") is not None:
                ex = (f"（例如越过 X 日期后第 20 季的核心通胀：按规则 {float(xr['rule']['core_h20']):+.2f}，利率不变 "
                      f"{float(xr['held']['core_h20']):+.2f} 个百分点）")
            lines.append("还有一点：模型自己的利率方程（它按通胀与失业作出的系统性反应）带着同样方向错误的滞后效应，修正不"
                         "处理这一部分。信用冲击之后，模型的降息会让通胀在之后几年进一步走低；通胀冲击之后，加息会让通胀"
                         "更高。所以智能体树里的通胀大体沿着模型规则的路径走，基本不随智能体美联储的选择而变。每个动作的"
                         "“文献与足迹”里有两种读数：美联储按模型规则反应，和利率沿不采取这一动作时的路径" + ex + "。")
        else:
            lines.append(head + "：利率决定对通胀的方向与文献相反，读美联储分支的通胀结果时要注意。")
    rp = dict(root or {})
    if rp.get("d_ff_median") is not None:
        drv = []
        if rp.get("core_yoy") is not None:
            drv.append(f"核心 CPI 同比 {float(rp['core_yoy']):.2f}%")
        if rp.get("cpi_yoy") is not None:
            drv.append(f"CPI 同比 {float(rp['cpi_yoy']):.2f}%")
        if rp.get("unrate") is not None:
            drv.append(f"失业率 {float(rp['unrate']):.1f}%")
        if rp.get("oil_pct") is not None:
            drv.append(f"实际油价处于历史 {float(rp['oil_pct']):.0f} 分位")
        same = []
        if rp.get("goods_median") is not None:
            same.append(f"商品通胀 {float(rp['goods_median']):.1f}%")
        if rp.get("serv_median") is not None:
            same.append(f"服务通胀 {float(rp['serv_median']):.1f}%")
        mich = (f"起点的 1 年通胀预期（密歇根）{float(rp['infexp1']):.1f}% 偏高，但它在模型里没有锚定、也不进利率方程，"
                "只影响大幅加息选项的门槛（门槛已修正：这一项另要求核心通胀至少 3.5%）。" if rp.get("infexp1") is not None else "")
        carry = float(rp.get("carry") or 0.0)
        carry_txt = ""
        if abs(carry) >= 0.005:
            carry_txt = (f"（其中约 {carry:+.2f} 是上一次{'加息' if carry > 0 else '降息'}在季度平均里尚未体现的部分）")
        pin = rp.get("first_meeting") or {}
        if pin.get("option"):
            move = {"hold": "按兵不动", "cut_25": "降息 25 个基点", "cut_50": "降息 50 个基点",
                    "hike_25": "加息 25 个基点"}.get(str(pin["option"]), str(pin["option"]))
            pin_txt = (f"本季第一次议息会议的结果设为{move}" + ("（已确认）" if pin.get("confirmed") else "（默认值，未确认）")
                       + "，美联储的历史先验只在相容的选项上重新归一。")
        else:
            pin_txt = "本季没有锁定的议息会议结果。"
        lines.append(f"季度模型自己给第一季的美联储分布偏鹰：按模型口径（季末月的月均），政策利率变动的中位数约 "
                     f"{float(rp['d_ff_median']):+.2f} 个百分点，{float(rp.get('share_25', 0)):.0%} 的粒子至少加息 25 个基点，"
                     f"{float(rp.get('share_75', 0)):.0%} 至少 75 个基点{carry_txt}。模型的利率方程按同季的通胀与失业、期初的"
                     "核心通胀、失业率和利率给出这一季的利率；起点上它读到的是" + "、".join(drv)
                     + ("，以及季前分布里本季的" + "、".join(same) + "（年化中位数）" if same else "")
                     + "。" + mich + pin_txt)
    return lines


ROOT_PRIOR_PARTICLES = 1000
_ROOT_PRIOR: dict = {}


def root_prior(adapter, eng: Engine) -> dict:
    """The model's own Fed draw at the root, for the honesty card (the hawkish prior and the states behind it): A0's
    d_ff in the model's units (median, shares ≥ +25bp and ≥ +75bp), the carry of September's hike, the root's core
    and headline CPI y/y, unemployment, Michigan 1-year expectations and real oil's historical percentile, and A0's
    same-quarter goods and services inflation (medians). Cached per adapter identity."""
    key = (MA.identity(adapter), canonical(eng.inst), eng.st.first_quarter)
    hit = _ROOT_PRIOR.get(key)
    if hit is not None:
        return hit
    out: dict = {}
    try:
        n = ROOT_PRIOR_PARTICLES
        s = adapter.root_state(n)
        s[FF_TARGET_KEY] = np.full(n, eng.root_ff_target(s))
        d = MA.as_draws(adapter.complete(s, MA.Given.empty(n), np.random.default_rng(1)))
        dff = np.asarray(d.x["d_ff"], float)
        med = lambda k: float(np.median(np.asarray(s[k], float))) if k in s else None  # noqa: E731
        out = {"d_ff_median": rnd(float(np.median(dff)), 4), "share_25": rnd(float(np.mean(dff >= 0.25 - 1e-9)), 4),
               "share_75": rnd(float(np.mean(dff >= 0.75 - 1e-9)), 4),
               "carry": rnd(float(np.median(eng.ff_target(s) - np.asarray(s[FED_FLOOR_FIELD], float))), 4),
               "core_yoy": rnd(med("s_core_yoy"), 4), "cpi_yoy": rnd(med("s_cpi_yoy"), 4),
               "unrate": rnd(med("s_unrate"), 4), "infexp1": rnd(med("s_infexp1"), 4),
               "goods_median": rnd(float(np.median(d.x["d_cpi_goods"])), 4) if "d_cpi_goods" in d.x else None,
               "serv_median": rnd(float(np.median(d.x["d_cpi_serv"])), 4) if "d_cpi_serv" in d.x else None}
        pin = eng.fed_pin(eng.st.first_quarter)
        if pin:
            item = next((i for i in eng.items if i.get("id") == pin.get("meeting")), {}) or {}
            out["first_meeting"] = {"option": pin["option"], "confirmed": str(item.get("status") or "") == "confirmed"}
        h = (eng.meta.get("s_oil_real") or {}).get("hist") or {}
        if "s_oil_real" in s and h.get("p50") is not None:
            v = med("s_oil_real")
            qs = [(p, h.get(f"p{p:02d}")) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99) if h.get(f"p{p:02d}") is not None]
            if qs and v is not None:
                xs, ys = [q[1] for q in qs], [q[0] for q in qs]
                out["oil_pct"] = rnd(float(np.interp(v, xs, ys)), 3)
    except Exception as err:  # noqa: BLE001 - information for readers, never fatal
        out = {"error": f"{type(err).__name__}: {str(err)[:120]}"}
    _ROOT_PRIOR[key] = out
    return out


def coverage_actors(eng: Engine) -> list[str]:
    """The actors whose options can fire on the engine's model: the Treasury needs a bill-share field (schema 3.0
    has none), every other actor is in the catalogue's menus."""
    return [a for a, x in A.ACTORS.items()
            if not (x.active == "bill_share_field" and eng.resolver.field("bill_share_chg") is None)]


def channel_coverage(actors: Iterable[str] | None = None) -> dict:
    """Counts of the channel table's option entries by kind (the honesty card's coverage line), over the catalogue's
    active actors (an actor A.ACTORS marks off, the Treasury in schema 3.0, is counted as "dormant")."""
    out = {"template": 0, "display": 0, "beyond": 0, "model": 0, "none": 0, "dormant": 0}
    live = set(actors) if actors is not None else {a for a, x in A.ACTORS.items() if x.active != "bill_share_field"}
    dormant = set(A.ACTORS) - live                           # nature is code, not a catalogue actor: always counted
    for e in (CH.table().get("options") or {}).values():
        if e.get("actor") in dormant:
            out["dormant"] += 1
            continue
        if e.get("status") == "statements":
            out["template"] += 1
            out["display"] += int(bool(e.get("display_only")))
            out["beyond"] += int(bool(e.get("beyond_history")))
        elif e.get("kind") == "model":
            out["model"] += 1
        else:
            out["none"] += 1
    return out


class TreeRun:
    """Breadth-first beam over levels. The result is a pure function of (settings, adapter version, catalogue,
    institutions, cache contents); a checkpoint after every level makes --resume continue exactly."""

    def __init__(self, settings, store: Store, *, adapter=None, llm=None, deep_llm=None, root_accounts=None,
                 institutions=None, journal: Journal | None = None, cancel: threading.Event | None = None,
                 cache_dir: Path | None = None, keep_all_last: bool = False, kill: threading.Event | None = None):
        self.st = TreeSettings.coerce(settings)
        self.store = store
        self.adapter = adapter if adapter is not None else MA.load_adapter()
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
        self.engine.defer_attribution = True           # the channel attribution only for the reduction's survivors
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
                              calls_paid=self.calls_paid(), model=MA.model_info(self.adapter))
        else:
            root = self.engine.root()
            frontier, level0 = [root], 0
            self.index[root.key] = [0, "pending", None]
            self.store.save_frontier(root)
            self.journal.emit("run_started", settings=self.st.hash(), depth=self.st.depth, B=self.st.B,
                              tree_kind=self.st.tree_kind, stub=bool(getattr(self.adapter, "stub", True)),
                              model=MA.model_info(self.adapter))
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
                notes: dict = {}                  # the adapter's and the engine's notes, by kind (floors, saturation …)
                for c in (detail or {}).get("clusters") or []:
                    for nt in c.get("notes") or []:
                        k = note_kind(nt)
                        notes[k] = notes.get(k, 0) + int(nt.get("particles") or 0)
                self.journal.emit("node_done", level=d, node=nd.key, children=len(children), flags=nd.flags,
                                  **({"notes": notes} if notes else {}))
            last = d == int(self.st.depth) - 1
            disc = discrete_stats(cands)
            tails = tail_stats(cands)
            tails.update(ident_stats(cands))
            tails.update(notes_stats(cands))
            kept, pruned, stats = reduce_level(cands, int(self.st.B), eps=float(self.st.eps_merge),
                                               fact_penalty=float(self.st.fact_penalty), n=int(self.st.n_particles),
                                               seed=int(self.st.seed), keep_all=last and self.keep_all_last,
                                               scales=self.engine.key_scales)
            self._finish_attribution(kept)
            for c in pruned:
                c.__dict__.pop("_attr_job", None)
            stats.update(disc)
            stats.update(tails)
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

    def _finish_attribution(self, kept: list) -> None:
        """The deferred channel attributions of the level's survivors (Engine.finish_attribution), in parallel when
        the level's expansion is (each job is one or two model completions on its parent's particles)."""
        todo = [c for c in kept if c.__dict__.get("_attr_job") is not None]
        if not todo:
            return
        if len(todo) > 1 and int(self.st.concurrency) > 1:
            with cf.ThreadPoolExecutor(max_workers=max(1, int(self.st.concurrency))) as ex:
                list(ex.map(lambda c: self.engine.finish_attribution([c]), sorted(todo, key=lambda c: c.key)))
        else:
            self.engine.finish_attribution(sorted(todo, key=lambda c: c.key))

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
EXPORT_DISABLED_WHY = "组合评估与提取服务还不能读取季度情景集：交给组合评估暂时禁用"


def export_status(model: Mapping | None = None, keep_paths: str | None = None) -> dict:
    """Whether a run or snapshot can be handed to portfolio and withdrawal (AGENT_DESIGN §14 D14, §16). Never for a
    stub run (§16). For a run the real quarterly model priced: only once the portfolio and withdrawal services take
    quarterly scenario sets (step_months = 3, phase P11, not built), so it stays disabled; when that lands, such an
    export must carry the stack's 试运行 label (T-S has not passed) and needs keep_paths "full" (a snapshot keeps no
    quarterly path history). {enabled, reasons, why}."""
    m = dict(model or {})
    reasons = [EXPORT_DISABLED_WHY]
    if m.get("stub", True):
        reasons.append("桩模型的推演不能导出（资产价格只用于检查流程）")
    elif m.get("trial"):
        reasons.append("季度模型仍在试运行（切换门未通过）：服务支持后，导出也要带这个标注")
    if keep_paths == "none":
        reasons.append("没有保存逐季路径：导出需要重新运行一次并保存逐季路径")
    return {"enabled": bool(EXPORT_ENABLED and not m.get("stub", True)), "reasons": reasons, "why": "；".join(reasons)}


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

CAND_PARTICLES_PER_CLUSTER = 1.3    # candidates' particles per action cluster, in units of N (natural counts)
# peak RSS = the process with the pricing model loaded + a fixed part (records, journal, buffers) + MEMORY_OVERHEAD ×
# the clouds' bytes. Measured 2026-11-02 (mock, N 500): the process 96 MB with the stub, 180–186 MB with the quarterly
# stack and its panels; peaks 323 MB (B 2 × 80) and 388 MB (B 6 × 10) with the stack, 273 MB (B 2 × 80) with the stub
MEMORY_BASE_BYTES = {"stub": 96 * 2**20, "quarterly": 180 * 2**20, "custom": 180 * 2**20}
MEMORY_FIXED_BYTES = 120 * 2**20
MEMORY_OVERHEAD = 1.6
# engine and model time per expanded node (no LLM; N 500, this machine, 2026-11-02): the stub 0.25 s, the stack 0.36 s
COMPUTE_S_PER_NODE = {"stub": 0.25, "quarterly": 0.36, "custom": 0.36}
# template channels on (agent-tree-6): the merge, the joint rarity test, the passes (pass 1, IP's increments, the
# inflation pass) and the attribution for the reduction's survivors only; B 2 × 80 on the stack, mock LLM, N 500:
# 251 s against 75 s with channels off (2026-11-02; agent-tree-5 with the attribution in every cluster: 272 s)
COMPUTE_CHANNELS_FACTOR = 3.3
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


PROBE_QUARTERS = 2                  # the probe cloud is advanced this many quarters by the pricing model's own draws
PROBE_MAX_PARTICLES = 1000
DEPTH_BYTES_FACTOR = 1.10           # compressed bytes per particle deep in a tree / after PROBE_QUARTERS (measured
                                    # 2026-11-02 on B 2 × 80 mock builds: quarterly stack 1,440 / 1,335, stub 965 / 858)
# by level (P14 review, frontier clouds of B 2/3/4/8/15/30 mock builds on the stack): the root's identical particles
# 0.09 of the probe, level 1 0.71, level 2 1.0 (the probe's own depth), then rising to DEPTH_BYTES_FACTOR by level 6;
# a wide level keeps about parents · K / kept of N distinct particles per node, and its compressed bytes scale with
# that fraction to the power 0.6 (B 8 / 15 / 30 level 1: 0.56 / 0.37 / 0.61 measured, 0.55 / 0.38 / 0.60 modelled)
DEPTH_LEVEL_FACTOR = (0.09, 0.71, 1.0)
DEPTH_RAMP_LEVELS = 4
WIDTH_BYTES_EXPONENT = 0.6


def level_bytes_factor(d: int, parents: int, K: int, kept: int) -> float:
    """Compressed cloud bytes per particle at level d relative to the probe cloud (memory_estimate's probe)."""
    d = int(d)
    if d < len(DEPTH_LEVEL_FACTOR):
        g = DEPTH_LEVEL_FACTOR[d]
    else:
        g = 1.0 + (DEPTH_BYTES_FACTOR - 1.0) * min(1.0, (d - len(DEPTH_LEVEL_FACTOR) + 1) / DEPTH_RAMP_LEVELS)
    h = 1.0 if d == 0 else min(1.0, max(int(parents), 1) * int(K) / max(int(kept), 1)) ** WIDTH_BYTES_EXPONENT
    return float(g * h)


def probe_cloud(settings, *, adapter=None, root_accounts=None, n_probe: int | None = None) -> tuple:
    """(engine, node) whose cloud the pricing model advanced PROBE_QUARTERS quarters from the root with its own
    draws (A0, no actors) and the books stepped with them: a cloud as dispersed as an early tree node."""
    st = TreeSettings.coerce(settings)
    adapter = adapter if adapter is not None else MA.load_adapter()
    n = int(n_probe or min(int(st.n_particles), PROBE_MAX_PARTICLES))
    eng = Engine(dataclasses.replace(st, n_particles=n, keep_paths="none"), adapter=adapter, llm=None,
                 root_accounts=root_accounts if root_accounts is not None else AC.load_root())
    nd = eng.root()
    empty = {"order": [], "actors": {}}
    for _ in range(PROBE_QUARTERS):
        d = eng.a0(nd)
        cl = eng.advance(nd, d, np.arange(n), {}, {}, empty)
        nd = Node(key=hashlib.sha256(f"probe|{nd.key}".encode()).hexdigest()[:16], parent=nd.key, level=nd.level + 1,
                  quarter=A.quarter_add(nd.quarter, 1), cloud=cl)
    return eng, nd


def _npz_bytes(arrays: Mapping) -> int:
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: MA.to_float32(v) for k, v in arrays.items()})
    return int(buf.tell())


def memory_estimate(settings, *, adapter=None, root_accounts=None, n_probe: int | None = None) -> dict:
    """Bytes per particle measured on a probe cloud (probe_cloud) and the peak resident memory of a beam run: the
    frontier (B) plus the level's candidates (about 1.3 N per action cluster), each particle at its deepest size,
    times MEMORY_OVERHEAD, plus the process with the pricing model loaded. Also what the store writes per particle
    (float32, compressed: the frontier and leaf clouds), for size_estimate."""
    st = TreeSettings.coerce(settings)
    adapter = adapter if adapter is not None else MA.load_adapter()
    eng, nd = probe_cloud(st, adapter=adapter, root_accounts=root_accounts, n_probe=n_probe)
    n = nd.cloud.n
    cl = nd.cloud
    per_particle = (cl.nbytes() - sum(np.asarray(v).nbytes for d in (cl.x, cl.inc) for v in d.values())) / n
    x_y = (len(adapter.x_fields) + 2 * len(adapter.assets) + 2) * 8
    path_per_q = len(adapter.assets) * 4 if st.paths else 0
    per_particle_last = per_particle + x_y + path_per_q * int(st.depth)
    frontier_arrays, frontier_meta = cloud_arrays(Cloud(cl.state, cl.acc, {}, cl.y, cl.cum, cl.cpi, None, {}), "full|")
    probe_bpp = _npz_bytes(frontier_arrays) / n
    frontier_bpp = DEPTH_BYTES_FACTOR * probe_bpp
    leaf_bpp = frontier_bpp + (path_per_q * int(st.depth) if st.paths else 0)
    frontier_json = len(json.dumps({"acc_meta": frontier_meta}, ensure_ascii=False, default=_jsonable))
    cands = int(st.B) * int(st.K) * int(st.M_max) * int(eng.split_max())
    # children keep their natural particle count (a cluster's N particles split among its branches and cells;
    # mechanism branches re-complete all of them): about 1.3 N per cluster, plus the frontier's N per node
    particles = (int(st.B) * int(st.K) * CAND_PARTICLES_PER_CLUSTER + int(st.B)) * int(st.n_particles)
    clouds = particles * per_particle_last
    kind = MA.adapter_kind(adapter)
    base = MEMORY_BASE_BYTES.get(kind, MEMORY_BASE_BYTES["custom"]) + MEMORY_FIXED_BYTES
    peak = base + MEMORY_OVERHEAD * clouds
    return {"bytes_per_particle": int(per_particle_last), "candidates_max": cands, "cloud_bytes": int(clouds),
            "peak_bytes": int(peak), "peak_mb": round(peak / 2**20, 1), "paths": st.paths, "model_kind": kind,
            "frontier_bytes_per_particle": int(round(frontier_bpp)), "leaf_bytes_per_particle": int(round(leaf_bpp)),
            "probe_bytes_per_particle": int(round(probe_bpp)),
            "frontier_json_bytes": int(frontier_json), "probe": {"particles": n, "quarters": PROBE_QUARTERS},
            "transmission": _estimate_transmission(eng),
            "compute_s_per_node": COMPUTE_S_PER_NODE.get(kind, COMPUTE_S_PER_NODE["custom"]) * int(st.n_particles) / 500.0
            * (COMPUTE_CHANNELS_FACTOR if (eng.channels_on and eng.structural) else 1.0),
            "basis": f"候选云的粒子（每个动作簇约 1.3 N）× 每粒子字节 × {MEMORY_OVERHEAD:g}（中间结果与 Python 开销）"
                     f"+ {base / 2**20:.0f} MB（进程、定价模型与固定开销）；每粒子字节在定价模型推进 {PROBE_QUARTERS} 季的探针云上实测，"
                     "系数按 B 2 × 80 与 B 6 × 10、500 粒子的实测峰值校准"}


def _estimate_transmission(eng: Engine) -> dict:
    """The estimate's transmission block: the manifest's (snapshot.engine_transmission), for the CLI's 传导 lines."""
    from .snapshot import engine_transmission
    return engine_transmission(eng)
