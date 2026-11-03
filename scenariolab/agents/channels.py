"""Template channels (the policy-channels round, AGENT_DESIGN §6.3–§6.4, SIM_DESIGN §7.2): what an action does to the
fields the quarterly model reads, when the model itself does not transmit it.

On the quarterly stack the 42 satellites never feed back into the ten core variables, so every satellite lever (the
balance sheet, Congress's fiscal tools, tariffs, immigration, the hyperscaler fallback) moves no macro field and no
asset (tree.transmission). SIM_DESIGN §7.2 decomposes every event into generic channels with at least two historical
instances (schema.json's templates G01–G35). This module is that decomposition as data (channels.json: the judge's
mapping of the round, every catalogue option either with statements or with a documented reason) and one protocol that
applies it per particle in B_k (Engine.build_bk):

* a **statement** pins a field for the particles that took the action: a *shift* x = A0 + ΣΔ (Δ ~ N(size, soft²) per
  particle, in the field's 1985–2019 quarterly sd; channels and actors add), a *hold* at the particle's own A0 draw
  (yields to any write), a *display* set last on a satellite (nothing responds) or a *label*. Shifts on the
  increment fields (INC_FIELDS: industrial production, P7, and core CPI, the core-goods pass-through) are instead
  increments on the value the quarter's other pins give (later completions, Engine._passes);
* **triggers**: a realized lever write (per unit, tariffs by sign; commitments fire when executed), the oil-supply
  composite Σz, a chosen option at its onset (sizes at the default intensity x0, scaled by the realized magnitude), a
  standing fact while it lives (per particle, at most 8 quarters per onset), conditions on the start-of-quarter state
  or on a live fact, the accounts' X-date (once per particle: the first quarter its TGA is below the floor while
  its debt-limit standoff is still live, i.e. Congress let the deadline pass) and a pinned pandemic (at each onset);
  the Fed-independence ladder writes only the increment over the highest rung already live for the particle, and
  its expectations drift follows the highest live rung while it lives;
* **persistence**: later quarters' statements wait in a per-particle queue (`_ch_queue`, a private state key the
  resampling and the roll carry); later choices never cancel it; demand-side activity statements carry a `decay`
  (the same pulse given back over the next quarters, so a demand shock is not a permanent level of output);
* **rules**: room to the historical extremes for GZ, the Baa spread and unemployment; |ΣΔ| ≤ 3 sd per field-quarter
  except beyond-history templates (pandemic, blockade, X-date: flagged 超出可测历史); policy uncertainty ≤ 3 units a
  quarter; news statements of the k-th consecutive repeat within a news group (an actor's moves in one direction)
  scaled by 1/k; channels never write the policy rate, prices, valuations, anchors or states.

The price-puzzle overlay (the judge's interim decision until the model role re-identifies u_ff) is a channel of the
same protocol: each particle's Fed surprise e (pinned d_ff minus its own A0 d_ff) is followed by increments
h_k · e on services CPI, goods CPI and core CPI for 20 quarters, on top of what the quarter's other pins give (the
last completion pins the three flows at their values plus the increments), with a kernel deconvolved once per
adapter so the surprise's net effect on those three flows is zero (`overlay_kernel`). Because it is an increment, it
never erases another action's inflation effect. Everything here is code: no LLM, no model call except the kernel's.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TABLE_PATH = Path(__file__).resolve().parent / "channels.json"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "qmodel" / "schema.json"
MODES = ("shift", "hold", "display", "label")
KINDS = ("flow", "news")
SCALES = ("lin", "step", "fixed")
# private state keys (the engine's, carried along axis 0 by take / concat / the adapters' rolls) and their draw columns
QUEUE_KEY, FACT_KEY, FACTX_KEY, STREAK_KEY, FFS_KEY, BEYOND_KEY = (
    "_ch_queue", "_ch_fact", "_ch_factx", "_ch_streak", "_ch_ffs", "_ch_beyond")
STATE_KEYS = (QUEUE_KEY, FACT_KEY, FACTX_KEY, STREAK_KEY, FFS_KEY, BEYOND_KEY)
NEXT_PREFIX = "_chn|"                       # draws.x[NEXT_PREFIX + key]: the particle's next channel state
OVERLAY_FIELDS = ("d_cpi_serv", "d_cpi_goods", "d_core")
INFLATION_FIELDS = OVERLAY_FIELDS           # the inflation pass pins these three flows (Engine._passes)
INC_FIELDS = ("d_ip", "d_core")             # shifts on these are increments on the quarter's other pins
OVERLAY_K = 20
OVERLAY_ATTR = "价格之谜修正"
CHANNEL_ATTR = "通道（模板与文献）"
BEYOND_ZH = "超出可测历史"

_TABLE: dict | None = None
_TABLE_HASH = ""
_TABLE_LOCK = threading.Lock()


def table() -> dict:
    """channels.json (cached)."""
    global _TABLE, _TABLE_HASH
    with _TABLE_LOCK:
        if _TABLE is None:
            raw = TABLE_PATH.read_bytes()
            _TABLE = json.loads(raw.decode("utf-8"))
            _TABLE_HASH = hashlib.sha256(raw).hexdigest()[:12]
        return _TABLE


def version() -> str:
    """The table's version string and a hash of its bytes: an edit that forgets to bump the string still changes
    the version a folder records (resume and on-demand expansion compare it)."""
    t = table()
    return f"{t.get('version') or ''} sha256:{_TABLE_HASH}"


def option(key: str) -> dict | None:
    """The table entry of "actor.option" (None for as_committed, custom or an unknown option)."""
    return (table().get("options") or {}).get(key)


_COVERED: set | None = None


def covered_facts() -> set:
    """Facts set by options with template statements (their consequence has a channel), cached."""
    global _COVERED
    if _COVERED is None:
        from . import actors as A
        out = set()
        for e in (table().get("options") or {}).values():
            if e.get("status") != "statements":
                continue
            out |= {f for f in (e.get("onset_fact"), e.get("flow_fact")) if f}
            a = A.ACTORS.get(e.get("actor"))
            if a is not None:
                try:
                    out |= {f for f, _ in a.find(e["option"])[1].facts}
                except KeyError:
                    pass
        _COVERED = out
    return set(_COVERED)


_NAMES: dict | None = None


def template_names() -> dict[str, str]:
    """{template id: Chinese name} from the quarterly schema (G01–G35 and the channel map's ids), read once."""
    global _NAMES
    if _NAMES is not None:
        return dict(_NAMES)
    try:
        s = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {t["id"]: t.get("zh") or t["id"] for t in s.get("templates") or [] if isinstance(t, Mapping) and t.get("id")}
    for k, v in (s.get("channel_map") or {}).items():
        out.setdefault(k, (v or {}).get("name") or k)
    _NAMES = out
    return dict(out)


def statements_of(entry: Mapping) -> list[dict]:
    """Every statement of an entry: its own, its per-write tables', its lever's (both signs)."""
    out = list(entry.get("statements") or [])
    for ws in (entry.get("writes") or {}).values():
        out += list(ws)
    lv = (table().get("levers") or {}).get(entry.get("lever") or "")
    if lv:
        out += list(lv.get("up") or []) + list(lv.get("down") or [])
    return out


def problems(fields: Iterable[str] | None = None, catalogue: Mapping | None = None) -> list[str]:
    """Chinese load-time problems: template ids that resolve to no schema template or channel-map id; unknown modes,
    kinds or scales; displays outside the protocol's display fields; growth fields without IP; queue offsets beyond
    the queue; fields the model does not have (`fields`); catalogue options without an entry (`catalogue`: {actor:
    [option ids]})."""
    t = table()
    names = template_names()
    out: list[str] = []
    pr = t.get("protocol") or {}
    disp_ok = set(pr.get("display_fields") or ())
    never = set(pr.get("never") or ())
    h = int(pr.get("queue_h", 8))
    fields = set(fields) if fields is not None else None

    def check(where, stmts, templates):
        for tid in templates or []:
            if tid not in names:
                out.append(f"{where}：模板 {tid} 在季度模型的模板表与通道表里都没有")
        growth = {}
        for s in stmts:
            f, mode = s.get("field"), s.get("mode")
            if mode not in MODES:
                out.append(f"{where}：{f} 的方式 {mode} 无效")
            if s.get("kind") not in KINDS:
                out.append(f"{where}：{f} 的类别 {s.get('kind')} 无效")
            if s.get("scale") not in SCALES:
                out.append(f"{where}：{f} 的缩放 {s.get('scale')} 无效")
            if f in never:
                out.append(f"{where}：通道不能写 {f}")
            if mode == "display" and f not in disp_ok:
                out.append(f"{where}：{f} 不在展示字段表里（展示只用于卫星变量）")
            if fields is not None and f not in fields:
                out.append(f"{where}：模型没有字段 {f}")
            if s.get("cond") != "while_fact" and max(s.get("q") or [0]) >= h:
                out.append(f"{where}：{f} 的季度偏移超出队列长度 {h}")
            if s.get("decay"):
                if mode != "shift":
                    out.append(f"{where}：{f} 的回吐（decay）只用于移动")
                elif max(s.get("q") or [0]) + int(s["decay"]) > h:
                    out.append(f"{where}：{f} 的回吐超出队列长度 {h}")
            for q in s.get("q") or []:
                if f in ("d_unrate", "d_rgdp"):
                    growth.setdefault(q, set()).add(f)
                if f == "d_ip":
                    growth.setdefault(q, set()).add("d_ip")
        for q, fs in growth.items():
            if fs & {"d_unrate", "d_rgdp"} and "d_ip" not in fs:
                out.append(f"{where}：第 {q} 季写了 {sorted(fs)} 却没有写或按住工业产出（增长字段规则）")
    for k, lv in (t.get("levers") or {}).items():
        check(f"杠杆 {k}", list(lv.get("up") or []) + list(lv.get("down") or []), lv.get("templates"))
    check("石油合成量", (t.get("oil") or {}).get("per_z") or [], (t.get("oil") or {}).get("templates"))
    for k, ev in (t.get("events") or {}).items():
        check(f"事件 {k}", ev.get("statements") or [], ev.get("templates"))
    facts = t.get("facts") or {}
    for f in (pr.get("ladder") or {}).get("facts") or {}:
        if f not in facts:
            out.append(f"独立性阶梯：事实 {f} 不在通道的事实表里")
    for k, ev in (t.get("events") or {}).items():
        for f in ev.get("requires_live") or []:
            if f not in facts:
                out.append(f"事件 {k}：前提事实 {f} 不在通道的事实表里")
    for k, e in (t.get("options") or {}).items():
        if e.get("status") == "statements":
            check(k, statements_of(e), e.get("templates"))
            for f in [e.get("onset_fact"), e.get("flow_fact"), e.get("cond_fact")] + list(e.get("ends") or []):
                if f and f not in facts:
                    out.append(f"{k}：事实 {f} 不在通道的事实表里")
        elif e.get("status") == "no_statement":
            if not (e.get("why") or e.get("zh")):
                out.append(f"{k}：没有写后果，也没有写理由")
            for tid in e.get("templates") or []:
                if tid not in names:
                    out.append(f"{k}：模板 {tid} 在季度模型的模板表与通道表里都没有")
        else:
            out.append(f"{k}：状态 {e.get('status')} 无效")
    for aid, ids in (catalogue or {}).items():
        for oid in ids:
            if f"{aid}.{oid}" not in (t.get("options") or {}):
                out.append(f"目录选项 {aid}.{oid} 没有通道条目（要么写后果，要么写理由）")
    return out


# =====================================================================================================================
# The per-particle layer
# =====================================================================================================================

@dataclass
class Choice:
    """One actor option on a subset of a cluster's particles (rows: indices into the cluster)."""
    actor: str
    option: str                             # catalogue option id, "as_committed" or "custom"
    rows: np.ndarray
    magnitude: float | None = None
    writes: list = field(default_factory=list)       # [{field, kind, value: (len(rows),), a, option}]
    facts: list = field(default_factory=list)        # fact ids the option sets
    clears: list = field(default_factory=list)


@dataclass
class Result:
    """One cluster-quarter of statements, in field units, per particle (m particles)."""
    m: int
    shift: dict = field(default_factory=dict)        # field -> (Δ, mask): current-quarter shifts (not INC_FIELDS)
    ip: tuple | None = None                          # (Δ, mask) of d_ip: increments on pass-1 IP (two-pass rule)
    core: tuple | None = None                        # (Δ, mask) of d_core: the core-goods pass-through increments
    hold: dict = field(default_factory=dict)         # field -> mask
    label: dict = field(default_factory=dict)        # field -> values (NaN = none)
    display: dict = field(default_factory=dict)      # field -> (Δ, mask)
    beyond: np.ndarray | None = None                 # particles with a beyond-history statement this quarter
    next: dict = field(default_factory=dict)         # the particles' next channel state
    rows: list = field(default_factory=list)         # summary rows (bk["channels"])
    fields: set = field(default_factory=set)         # fields any statement touched this quarter
    signs: dict = field(default_factory=dict)        # field -> sign of the cluster's mean channel Δ (incl. queued)
    covered: set = field(default_factory=set)        # lever fields whose writes have a channel entry
    notes: list = field(default_factory=list)
    active: np.ndarray | None = None                 # particles with any statement (current or queued)
    overlay: np.ndarray | None = None                # particles the price-puzzle overlay moved
    overlay_inc: dict = field(default_factory=dict)  # field -> (Δ, mask): the overlay's increments (inflation pass)

    def any(self) -> bool:
        return bool(self.active is not None and self.active.any())


class Layer:
    """The channel table applied to one cluster's particles per quarter (see the module docstring).

    sd: the engine's field sds (1985–2019 quarterly); fields: the model's x fields; facts_exit: the catalogue's exit
    hazards (actors.FACTS); kernel: the price-puzzle overlay's {field: (K + 1,) per pp} or None (off)."""

    def __init__(self, sd: Mapping[str, float], fields: Iterable[str], *, facts_exit: Mapping[str, float | None],
                 kernel: Mapping | None = None, tab: Mapping | None = None):
        self.t = dict(tab or table())
        self.pr = dict(self.t.get("protocol") or {})
        self.sd = dict(sd)
        self.fields = set(fields)
        self.kernel = {f: np.asarray(v, float) for f, v in (kernel or {}).items()} or None
        self.H = int(self.pr.get("queue_h", 8))
        self.facts = sorted(self.t.get("facts") or {})
        self.fact_src = dict(self.t.get("facts") or {})
        self.fact_exit = {f: facts_exit.get(f) for f in self.facts}
        self.fact_max = int(self.pr.get("fact_max_q", 8))
        self.room = {f: tuple(v) for f, v in (self.pr.get("room") or {}).items()}
        self.cap = float(self.pr.get("cap_sd", 3.0))
        self.tpu_cap = float(self.pr.get("tpu_cap", 3.0))
        self.soft0 = float(self.pr.get("soft_default", 0.5))
        lo, hi = self.pr.get("write_ratio_clip") or (0.0, 3.0)
        self.ratio = (float(lo), float(hi))
        self.options = dict(self.t.get("options") or {})
        self.levers = dict(self.t.get("levers") or {})
        self.oil = dict(self.t.get("oil") or {})
        self.events = dict(self.t.get("events") or {})
        # standing facts' flow options (fact -> option key) and the options with news statements (habituation)
        self.flow_of = {e["flow_fact"]: k for k, e in self.options.items() if e.get("flow_fact")}
        # news habituation by group (an actor's moves in one direction: alternating a small and a large escalation
        # is still a repeat); an option without a group is its own
        news_opts = sorted(k for k, e in self.options.items() if e.get("status") == "statements"
                           and any(s.get("kind") == "news" for s in statements_of(e)))
        self.news_group = {k: str(self.options[k].get("news_group") or k) for k in news_opts}
        self.news = sorted(set(self.news_group.values()))
        gix = {g: i for i, g in enumerate(self.news)}
        self.news_ix = {k: gix[g] for k, g in self.news_group.items()}
        # the Fed-independence ladder: fact -> level (a share of the full template); onsets write the increment
        # over the highest rung already live, the drift follows the highest live rung
        self.ladder = {str(f): float(v) for f, v in ((self.pr.get("ladder") or {}).get("facts") or {}).items()}
        self.slots = self._slots()
        self.slot_ix = {s: i for i, s in enumerate(self.slots)}

    # ---------------------------------------------------------------- state
    def _slots(self) -> list[tuple[str, str]]:
        """(field, mode) of every statement that can fall due after its trigger's quarter (the queue's columns)."""
        out = set()
        srcs = [s for e in self.options.values() for s in statements_of(e) if s.get("cond") != "while_fact"]
        srcs += list(self.oil.get("per_z") or [])
        srcs += [s for ev in self.events.values() for s in ev.get("statements") or []]
        for s in srcs:
            if max(s.get("q") or [0]) > 0 or s.get("decay"):
                out.add((s["field"], s["mode"]))
        return sorted(out)

    def empty_state(self, n: int) -> dict:
        n = int(n)
        return {QUEUE_KEY: np.zeros((n, self.H, len(self.slots)), np.float32),
                FACT_KEY: np.zeros((n, len(self.facts)), np.float32),
                FACTX_KEY: np.zeros((n, len(self.facts)), np.float32),
                STREAK_KEY: np.zeros((n, len(self.news)), np.float32),
                FFS_KEY: np.zeros((n, OVERLAY_K), np.float32),
                BEYOND_KEY: np.zeros(n, np.float32)}

    def read(self, state: Mapping, m: int) -> dict:
        """The channel state of m particles (zeros where a cloud saved before channels has none, or the layout
        changed)."""
        z = self.empty_state(m)
        out = {}
        for k, v in z.items():
            a = state.get(k)
            out[k] = np.asarray(a, np.float64).copy() if a is not None and np.shape(a) == v.shape else v.astype(np.float64)
        return out

    # ---------------------------------------------------------------- one quarter
    def step(self, state: Mapping, a0x: Mapping, choices: list[Choice], rng: np.random.Generator, *,
             node_facts: Mapping | None = None, oil_z: np.ndarray | None = None, xdate: np.ndarray | None = None,
             surprise: np.ndarray | None = None, overlay: bool = False, literature: bool = False) -> Result:
        m = len(np.asarray(next(iter(a0x.values()))))
        st = self.read(state, m)
        res = Result(m=m)
        cur = _Acc(m)
        Q = st[QUEUE_KEY]
        Qn = np.zeros_like(Q)
        Qn[:, :-1] = Q[:, 1:]
        age, factx, streak = st[FACT_KEY], st[FACTX_KEY], st[STREAK_KEY]
        beyond_left = st[BEYOND_KEY]
        res.beyond = beyond_left > 0
        summary: dict = {}
        sig: dict = {}
        active = np.zeros(m, bool)
        node_facts = dict(node_facts or {})

        def emit(src: str, s: Mapping, rows: np.ndarray, factor, *, shift_q: int = 0, beyond: bool = False,
                 templates=(), scale_news=None, tpu_scale=None):
            """One statement for `rows` with per-particle factor (scalar or (len(rows),)): offset 0 into this
            quarter, later offsets into the queue."""
            rows = np.asarray(rows, int)
            if not len(rows):
                return
            f, mode = s["field"], s["mode"]
            if f not in self.fields:
                return
            fac = np.broadcast_to(np.asarray(factor, float), (len(rows),)).copy()
            if scale_news is not None and s.get("kind") == "news":
                fac = fac * scale_news
            if tpu_scale is not None and s.get("tpu"):
                fac = fac * tpu_scale
            sd = float(self.sd.get(f, 1.0))
            if mode in ("shift", "display"):
                size = float(s.get("size") or 0.0) * sd
                soft = s.get("soft")
                soft = abs(float(soft)) if soft is not None else self.soft0 * abs(float(s.get("size") or 0.0))
                d = size * fac + soft * sd * np.abs(fac) * rng.standard_normal(len(rows))
                if mode == "display" and s.get("cap") is not None:
                    c = float(s["cap"]) * sd
                    d = np.clip(d, -c, c)
                if mode == "shift" and s.get("cap") is not None and not beyond:
                    c = float(s["cap"]) * sd
                    d = np.clip(d, -c, c)
            else:
                d = None
            for q in s.get("q") or [0]:
                qq = int(q) + int(shift_q)
                if qq == 0:
                    cur.add(mode, f, rows, d, s.get("value"))
                elif qq < self.H + 1:
                    j = self.slot_ix.get((f, mode))
                    if j is None:
                        continue
                    if mode in ("shift", "display"):
                        Qn[rows, qq - 1, j] += d
                    elif mode == "hold":
                        Qn[rows, qq - 1, j] = 1.0
                    else:
                        Qn[rows, qq - 1, j] = float(s.get("value") or 0.0) + 1.0
                key = (src, f, mode, "now" if qq == 0 else "later")
                row = summary.setdefault(key, {"source": src, "field": f, "mode": mode, "when": key[3], "sum": 0.0,
                                               "n": 0, "q": set(), "templates": list(templates), "beyond": beyond})
                row["q"].add(qq)
                if d is not None and qq == int(min(s.get("q") or [0])) + int(shift_q):
                    row["sum"] += float(np.sum(d)) / max(sd, 1e-12)
                    row["n"] += len(rows)
                elif d is None and qq == int(min(s.get("q") or [0])) + int(shift_q):
                    row["n"] += len(rows)
                if d is not None and mode == "shift":
                    sig[f] = sig.get(f, 0.0) + float(np.sum(d))
            if s.get("decay") and mode == "shift" and d is not None:
                # the pulse given back: −(Δ × its quarters) / n over the n quarters after its last one (demand-side
                # activity: a multiplier or a credit shock fades; Ramey 2019, Gilchrist-Zakrajšek 2012)
                n_dec = int(s["decay"])
                j = self.slot_ix.get((f, mode))
                back = -d * len(s.get("q") or [0]) / max(n_dec, 1)
                last = int(max(s.get("q") or [0])) + int(shift_q)
                for k in range(1, n_dec + 1):
                    qq = last + k
                    if j is not None and 0 < qq < self.H + 1:
                        Qn[rows, qq - 1, j] += back
                key = (src, f, mode, "decay")
                row = summary.setdefault(key, {"source": src, "field": f, "mode": mode, "when": "later", "sum": 0.0,
                                               "n": 0, "q": set(), "templates": list(templates), "beyond": beyond,
                                               "decay": True})
                row["q"] |= set(range(last + 1, min(last + n_dec, self.H) + 1))
                row["sum"] += float(np.sum(back)) / max(sd, 1e-12)
                row["n"] += len(rows)
            active[rows] = True
            if beyond:
                beyond_left[rows] = np.maximum(beyond_left[rows], float(max(s.get("q") or [0]) + shift_q + 1))
                res.beyond[rows] = True

        # 1. due queue entries
        if Q.shape[1] and len(self.slots):
            due = Q[:, 0, :]
            for j, (f, mode) in enumerate(self.slots):
                col = due[:, j]
                on = np.flatnonzero(col != 0)
                if not len(on):
                    continue
                if mode in ("shift", "display"):
                    cur.add(mode, f, on, col[on], None)
                elif mode == "hold":
                    cur.add(mode, f, on, None, None)
                else:
                    cur.add(mode, f, on, None, col[on] - 1.0)
                active[on] = True
                key = ("queue", f, mode, "now")
                row = summary.setdefault(key, {"source": "queue", "field": f, "mode": mode, "when": "now", "sum": 0.0,
                                               "n": 0, "q": {0}, "templates": [], "beyond": False})
                if mode in ("shift", "display"):
                    row["sum"] += float(np.sum(col[on])) / max(float(self.sd.get(f, 1.0)), 1e-12)
                row["n"] += len(on)
        fx = {f: i for i, f in enumerate(self.facts)}
        onset_now = np.zeros((m, len(self.facts)), bool)
        ended = np.zeros((m, len(self.facts)), bool)
        new_x = factx.copy()
        # 2. node and account events (beyond history)
        for evk, ev in self.events.items():
            j = fx.get(evk)
            if j is None:
                continue
            src = self.fact_src.get(evk)
            if src == "node":
                live = np.full(m, bool(node_facts.get(evk)))
                age[~live, j] = 0.0
            elif src == "accounts":
                # once per particle: the first quarter its account falls below the cash floor (the judge's "第一次");
                # the age is never reset, so a TGA hovering at the floor does not repeat the sovereign shock. Only
                # while the facts it escalates are live for the particle (the debt-limit standoff: Congress kept it
                # through the deadline); a TGA below the floor without one is bookkeeping, not a payments failure
                if xdate is None:
                    continue
                live = np.asarray(xdate, bool).copy()
                for rf in ev.get("requires_live") or []:
                    jr = fx.get(rf)
                    live &= (age[:, jr] >= 1) if jr is not None else False
            else:
                continue
            rows = np.flatnonzero(live & (age[:, j] == 0))
            onset_now[rows, j] = True
            for s in ev.get("statements") or []:
                emit(f"event:{evk}", s, rows, 1.0, beyond=bool(ev.get("beyond_history")), templates=ev.get("templates"))
        # 3. the actors' choices
        tpu_tot = np.zeros(m)
        plan = []
        for ch in choices:
            key = f"{ch.actor}.{ch.option}"
            ent = self.options.get(key) if ch.option not in ("as_committed", "custom") else None
            rows = np.asarray(ch.rows, int)
            if ent is not None and ent.get("status") == "statements":
                r = self.news_ix.get(key)
                k_old = streak[rows, r] if r is not None else np.zeros(len(rows))
                scale_news = 1.0 / (k_old + 1.0)
                ok = np.ones(len(rows), bool)
                cf = ent.get("cond_fact")
                if cf:
                    ok &= age[rows, fx[cf]] >= 1
                of = ent.get("onset_fact")
                onset_ok = ok.copy()
                if of:
                    onset_ok &= age[rows, fx[of]] == 0
                if ent.get("tpu") is not None:
                    tpu_tot[rows[onset_ok]] += float(ent["tpu"]) * scale_news[onset_ok]
                plan.append((ch, ent, rows, scale_news, ok, onset_ok))
            else:
                plan.append((ch, ent, rows, np.ones(len(rows)), np.ones(len(rows), bool), np.ones(len(rows), bool)))
        tpu_scale = np.where(np.abs(tpu_tot) > self.tpu_cap, self.tpu_cap / np.maximum(np.abs(tpu_tot), 1e-12), 1.0)
        streak_new = np.zeros_like(streak)
        # the ladder: the highest rung live at the start of the quarter, raised as this quarter's onsets climb it
        lad_now = np.zeros(m)
        for lf, lv in self.ladder.items():
            if lf in fx:
                lad_now = np.where(age[:, fx[lf]] >= 1, np.maximum(lad_now, lv), lad_now)
        lvl = {f: np.asarray(state.get(k), float) for f, (k, *_) in self.room.items() if state.get(k) is not None}
        for ch, ent, rows, scale_news, ok, onset_ok in plan:
            key = f"{ch.actor}.{ch.option}"
            if ent is not None and ent.get("status") == "statements":
                r = self.news_ix.get(key)
                if r is not None:
                    streak_new[rows, r] = streak[rows, r] + 1.0
                tmpl = ent.get("templates") or []
                beyond = bool(ent.get("beyond_history"))
                fac = self._x_factor(ent, ch, len(rows))
                lad = np.ones(len(rows))
                rung = self.ladder.get(str(ent.get("onset_fact") or ""))
                if rung:
                    # only the increment over the highest rung already live (attempt after a nominee adds 0.25 of
                    # the full template, not 0.75): the rungs are levels of one institutional shock, not new shocks
                    lad = np.clip((rung - lad_now[rows]) / rung, 0.0, 1.0)
                    lad = np.where(onset_ok, lad, 0.0)
                    lad_now[rows[onset_ok]] = np.maximum(lad_now[rows[onset_ok]], rung)
                for s in ent.get("statements") or []:
                    if s.get("cond") == "while_fact":
                        continue
                    sel = onset_ok.copy()
                    if rung:
                        sel &= lad > 1e-9
                    if s.get("cond") and s["cond"] != "while_fact":
                        sel &= _cond(s["cond"], state, rows)
                    f_s = fac if s.get("scale") == "lin" else 1.0
                    if rung:
                        f_s = f_s * lad
                    f_s = f_s[sel] if np.ndim(f_s) else f_s
                    emit(key, s, rows[sel], f_s, beyond=beyond, templates=tmpl, scale_news=scale_news[sel],
                         tpu_scale=tpu_scale[rows[sel]])
                # standing-fact onset and the onset quarter's flows (q 0)
                ff = ent.get("flow_fact") or ent.get("onset_fact")
                if ff and ff in fx:
                    j = fx[ff]
                    live = age[rows, j] >= 1
                    first = ok & ~live
                    onset_now[rows[first], j] = True
                    if ent.get("flow_fact"):
                        xf = np.broadcast_to(np.asarray(fac if np.ndim(fac) else np.full(len(rows), float(fac)), float),
                                             (len(rows),))
                        new_x[rows[ok], j] = xf[ok]                       # the latest magnitude
                        for s in ent.get("statements") or []:
                            if s.get("cond") == "while_fact" and 0 in (s.get("q") or []):
                                emit(f"fact:{ff}", s, rows[first], xf[first] if s.get("scale") == "lin" else 1.0,
                                     templates=tmpl)
                for e in ent.get("ends") or []:
                    if e in fx:
                        ended[rows[ok], fx[e]] = True
            for c in ch.clears or []:
                if c in fx:
                    ended[rows, fx[c]] = True
            # the realized writes: the lever's channel (tariff points, by sign) and the option's per-write tables
            for w in ch.writes or []:
                fid = w.get("field")
                vals = np.asarray(w.get("value"), float)
                if vals.shape != (len(rows),):
                    vals = np.broadcast_to(vals, (len(rows),))
                made_by = w.get("option") or ch.option
                ment = self.options.get(f"{ch.actor}.{made_by}") or {}
                lv = self.levers.get(fid)
                rn = self.news_ix.get(f"{ch.actor}.{made_by}")
                s_news = (1.0 / (streak[rows, rn] + 1.0)) if (rn is not None and ch.option == made_by) else np.ones(len(rows))
                if lv is not None and w.get("kind") in (lv.get("kinds") or ["point"]) and (
                        ch.option in ("custom", "as_committed") or ment.get("lever") == fid):
                    res.covered.add(fid)
                    up, dn = vals > 0, vals < 0
                    for side, sel in (("up", up), ("down", dn)):
                        if not sel.any():
                            continue
                        for s in lv.get(side) or []:
                            f_s = vals[sel] if s.get("scale") == "lin" else 1.0
                            emit(f"lever:{fid}", s, rows[sel], f_s, templates=lv.get("templates"), scale_news=s_news[sel])
                tab = (ment.get("writes") or {}).get(fid)
                if tab and ch.option != "custom":
                    res.covered.add(fid)
                    a = float(w.get("a") if w.get("a") is not None else np.mean(vals))
                    v = vals
                    if a != 0:
                        lo, hi = self.ratio
                        v = a * np.clip(vals / a, lo, hi)               # never against the option's sign, ≤ 3× its mean
                    for s in tab:
                        f_s = v if s.get("scale") in ("lin",) else 1.0
                        if s["mode"] == "hold":
                            f_s = 1.0
                        emit(f"{ch.actor}.{made_by}", s, rows, f_s, templates=ment.get("templates"),
                             scale_news=s_news)
                elif ment.get("status") in ("statements", "no_statement") and ch.option != "custom":
                    res.covered.add(fid)                              # a documented decision (e.g. the status quo)
        # 4. standing facts' flows in later quarters (per particle, while the fact lives)
        for ff, okey in self.flow_of.items():
            j = fx.get(ff)
            ent = self.options.get(okey)
            if j is None or ent is None:
                continue
            for s in ent.get("statements") or []:
                if s.get("cond") != "while_fact":
                    continue
                qs = set(int(q) for q in s.get("q") or [])
                live = (age[:, j] >= 1) & ~ended[:, j]
                if ff in self.ladder:
                    # the drift of the highest live rung only (a nominee's drift does not add to the court's)
                    top = np.zeros(m)
                    for lf, lv in self.ladder.items():
                        if lf in fx:
                            top = np.where((age[:, fx[lf]] >= 1) & ~ended[:, fx[lf]], np.maximum(top, lv), top)
                    live &= np.abs(top - self.ladder[ff]) < 1e-9
                if "while_max" in s and s["while_max"] is None:
                    # while the fact lives, without the protocol's cap (its exit hazard ends it)
                    lo = min([q for q in qs if q >= 1] or [1])
                    rows = np.flatnonzero(live & (age[:, j] >= lo))
                else:
                    rows = np.flatnonzero(live & np.isin(age[:, j], [q for q in qs if q >= 1])
                                          & (age[:, j] <= self.fact_max))
                if 0 in qs:                                            # every live quarter (onset counted above)
                    rows = np.flatnonzero(live & (age[:, j] < self.fact_max))
                if not len(rows):
                    continue
                f_s = factx[rows, j] if s.get("scale") == "lin" else 1.0
                s0 = dict(s, q=[0])
                emit(f"fact:{ff}", s0, rows, f_s, templates=ent.get("templates"))
        # 5. the oil-supply composite (per particle z)
        if oil_z is not None:
            z = np.asarray(oil_z, float)
            rows = np.flatnonzero(np.isfinite(z) & (z != 0))
            if len(rows):
                for s in self.oil.get("per_z") or []:
                    emit("oil", dict(s, soft=0.0), rows, z[rows], templates=self.oil.get("templates"))
        # 6. the price-puzzle overlay: h_k · e_(t−k) on the three inflation flows (field units)
        ffs = st[FFS_KEY]
        if overlay and self.kernel is not None:
            K = min(OVERLAY_K, ffs.shape[1])
            moved = np.zeros(m, bool)
            for f in OVERLAY_FIELDS:
                if f not in self.fields or f not in self.kernel:
                    continue
                h = np.asarray(self.kernel[f], float)[1:K + 1].copy()
                if literature:
                    h[2:] += -0.15                                     # CEE 1999 / Coibion 2012 (判断值; user option)
                d = ffs[:, :K] @ h
                on = np.flatnonzero(np.abs(d) > 1e-9)
                if len(on):
                    mk = np.abs(d) > 1e-9
                    res.overlay_inc[f] = (np.where(mk, d, 0.0), mk)
                    moved[on] = True
                    key = ("overlay", f, "shift", "now")
                    row = summary.setdefault(key, {"source": "overlay", "field": f, "mode": "shift", "when": "now",
                                                   "sum": 0.0, "n": 0, "q": {0}, "templates": [], "beyond": False})
                    row["sum"] += float(np.sum(d[on])) / max(float(self.sd.get(f, 1.0)), 1e-12)
                    row["n"] += len(on)
            res.overlay = moved
            active |= moved
        e = np.zeros(m) if surprise is None else np.where(np.isfinite(surprise), np.asarray(surprise, float), 0.0)
        ffs_n = np.zeros_like(ffs)
        if overlay and self.kernel is not None:
            ffs_n[:, 1:] = ffs[:, :-1]
            ffs_n[:, 0] = e
        # 7. room, caps (beyond-history particles exempt)
        for f, (d, mk) in list(cur.shift.items()):
            if f in self.room and f in lvl:
                _, p01, p50, p99 = self.room[f]
                L = lvl[f]
                fall = np.clip((L - p01) / max(p50 - p01, 1e-9), 0.0, 1.0)
                rise = np.clip((p99 - L) / max(p99 - p50, 1e-9), 0.0, 1.0)
                d = np.where(d < 0, d * fall, d * rise)
            sd = float(self.sd.get(f, 1.0))
            c = self.cap * sd
            over = mk & (np.abs(d) > c) & ~res.beyond
            if over.any():
                res.notes.append({"field": f, "code": "channel_cap", "particles": int(over.sum()),
                                  "text": f"{f}：模板通道合计超过 {self.cap:g} 个标准差，已截到上限"})
                d = np.where(over, np.clip(d, -c, c), d)
            cur.shift[f] = (d, mk)
        if res.beyond.any():
            res.notes.append({"code": "beyond_history", "particles": int(res.beyond.sum()),
                              "text": f"{BEYOND_ZH}的模板（疫情、封锁或越过 X 日期）：不受 3 个标准差上限约束"})
        # 8. the next state: ages (onsets, ends, per-particle survival), streaks, the queue, the overlay history
        nage = age.copy()
        for f, j in fx.items():
            src = self.fact_src.get(f)
            live = age[:, j] >= 1
            if src == "option":
                ex = self.fact_exit.get(f)
                surv = np.ones(m, bool) if ex is None else (rng.random(m) >= float(ex))
                nage[:, j] = np.where(live & surv, age[:, j] + 1.0, 0.0)
                nage[onset_now[:, j], j] = 1.0
                nage[ended[:, j], j] = 0.0
            else:                                    # node: live this quarter → one quarter older; accounts: once
                was = live | onset_now[:, j]         # passed, the X-date stays passed for the particle
                nage[:, j] = np.where(was, np.maximum(age[:, j], 0.0) + 1.0, 0.0)
        new_x[nage == 0] = 0.0
        res.next = {QUEUE_KEY: Qn.astype(np.float32), FACT_KEY: nage.astype(np.float32),
                    FACTX_KEY: new_x.astype(np.float32), STREAK_KEY: streak_new.astype(np.float32),
                    FFS_KEY: ffs_n.astype(np.float32),
                    BEYOND_KEY: np.maximum(beyond_left - 1.0, 0.0).astype(np.float32)}
        # 9. the result
        res.shift = {f: v for f, v in cur.shift.items() if f not in INC_FIELDS}
        res.ip = cur.shift.get("d_ip")
        res.core = cur.shift.get("d_core")
        res.hold, res.label, res.display = cur.hold, cur.label, cur.disp
        res.fields = set(cur.shift) | set(cur.hold) | set(cur.label) | set(cur.disp)
        res.active = active | (np.abs(Qn).sum(axis=(1, 2)) > 0 if Qn.size else active)
        res.signs = {f: float(np.sign(v)) for f, v in sig.items() if abs(v) > 0}
        for f, j in self.slot_ix.items():
            fld, mode = f
            if mode == "shift" and Qn.size:
                tot = float(Qn[:, :, j].sum())
                if abs(tot) > 0:
                    res.signs.setdefault(fld, float(np.sign(tot)))
        res.rows = [{"source": r["source"], "field": r["field"], "mode": r["mode"], "when": r["when"],
                     "mean_sd": (r["sum"] / r["n"]) if (r["n"] and r["mode"] in ("shift", "display")) else None,
                     "share": r["n"] / max(m, 1), "q": sorted(r["q"]), "templates": r["templates"], "beyond": r["beyond"],
                     **({"decay": True} if r.get("decay") else {})}
                    for r in summary.values()]
        res.rows.sort(key=lambda r: (r["when"] != "now", r["source"], r["field"], r["mode"]))
        return res

    def _x_factor(self, ent: Mapping, ch: Choice, n: int):
        """x/x0 for an option's "lin" statements: the realized magnitude (the option's) or write (per particle),
        clipped to the option's range; 1 without an intensity."""
        x = ent.get("x") or {}
        x0 = x.get("x0")
        if not x0:
            return 1.0
        rng_ = x.get("range") or [None, None]
        lo = min(v for v in rng_ if v is not None) if any(v is not None for v in rng_) else -math.inf
        hi = max(v for v in rng_ if v is not None) if any(v is not None for v in rng_) else math.inf
        src = str(x.get("source") or "magnitude")
        if src.startswith("write:"):
            fid = src.split(":", 1)[1]
            w = next((w for w in ch.writes or [] if w.get("field") == fid), None)
            if w is None:
                return 1.0
            v = np.clip(np.asarray(w.get("value"), float), lo, hi)
            return v / float(x0)
        mag = ch.magnitude
        if mag is None:
            return 1.0
        return float(np.clip(float(mag), lo, hi)) / float(x0)


class _Acc:
    """This quarter's statements, in field units: shifts and displays add, holds and labels mark."""

    def __init__(self, m: int):
        self.m = m
        self.shift: dict = {}
        self.disp: dict = {}
        self.hold: dict = {}
        self.label: dict = {}

    def add(self, mode: str, f: str, rows: np.ndarray, d, value) -> None:
        rows = np.asarray(rows, int)
        if mode in ("shift", "display"):
            tgt = self.shift if mode == "shift" else self.disp
            dd, mk = tgt.setdefault(f, (np.zeros(self.m), np.zeros(self.m, bool)))
            np.add.at(dd, rows, np.asarray(d, float))
            mk[rows] = True
        elif mode == "hold":
            mk = self.hold.setdefault(f, np.zeros(self.m, bool))
            mk[rows] = True
        else:
            arr = self.label.setdefault(f, np.full(self.m, np.nan))
            arr[rows] = np.asarray(value, float) if np.ndim(value) else float(value or 0.0)


def _cond(cond: str, state: Mapping, rows: np.ndarray) -> np.ndarray:
    """A start-of-quarter condition per particle (e.g. "s_unrate<4.5")."""
    for op in ("<=", ">=", "<", ">"):
        if op in cond:
            k, v = cond.split(op, 1)
            a = state.get(k.strip())
            if a is None:
                return np.zeros(len(rows), bool)
            x = np.asarray(a, float)[rows]
            v = float(v)
            return {"<=": x <= v, ">=": x >= v, "<": x < v, ">": x > v}[op]
    return np.ones(len(rows), bool)


# =====================================================================================================================
# The price-puzzle overlay's kernel
# =====================================================================================================================

_KERNEL: dict = {}
_KERNEL_LOCK = threading.Lock()


def overlay_kernel(adapter, identity: str, *, n: int = 2000, K: int = OVERLAY_K, seed: int = 11,
                   given_cls=None) -> dict | None:
    """{field: (K + 1,)} per 1 pp of a Fed surprise (field units), deconvolved by a lockstep pass with common random
    numbers: a +1 pp "do" surprise at k = 0, then at each k the correction that sets the three flows' net response
    (the surprise's propagation plus the earlier corrections' own) to zero (the judge's J07, design a's a13). Cached
    per adapter identity; None when the adapter cannot do it."""
    key = (identity, int(n), int(K), int(seed))
    with _KERNEL_LOCK:
        if key in _KERNEL:
            return _KERNEL[key]
    from . import model_adapter as MA
    G = given_cls or MA.Given
    try:
        sp, ss = adapter.root_state(n), adapter.root_state(n)
        fields = [f for f in OVERLAY_FIELDS if f in adapter.x_fields]
        ker = {f: [0.0] for f in fields}
        for k in range(K + 1):
            def r(k=k):
                return np.random.default_rng(seed * 1000 + k)
            a_p = MA.as_draws(adapter.complete(sp, G.empty(n), r()))
            a_s = MA.as_draws(adapter.complete(ss, G.empty(n), r()))
            if k == 0:
                d_s = MA.as_draws(adapter.complete(ss, G.from_dict({"d_ff": np.asarray(a_s.x["d_ff"]) + 1.0}, n), r()))
            else:
                pins = {}
                for f in fields:
                    p = float(np.mean(np.asarray(a_s.x[f], float) - np.asarray(a_p.x[f], float)))
                    ker[f].append(-p)
                    pins[f] = np.asarray(a_s.x[f], float) - p
                d_s = MA.as_draws(adapter.complete(ss, G.from_dict(pins, n), r()))
            sp, ss = adapter.update_state(sp, a_p), adapter.update_state(ss, d_s)
        out = {f: [round(float(v), 6) for v in vals] for f, vals in ker.items()}
    except Exception:  # noqa: BLE001 - the overlay then stays off; the honesty card says so
        out = None
    with _KERNEL_LOCK:
        _KERNEL[key] = out
    return out
