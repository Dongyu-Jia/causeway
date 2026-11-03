"""Snapshot mode (USER_DECISIONS 2026-11-02): a beam search run once with the real local Claude Code, frozen, and
explored without running the model again; a branch the snapshot did not expand is generated on demand by the LLM and
saved in an overlay next to the frozen core.

    python -m scenariolab.agents.snapshot --beam 2 --quarters 80 --dry-run       # the estimate; spends nothing
    python -m scenariolab.agents.snapshot --beam 2 --quarters 80 --yes           # build (real CLI: needs --yes)
    python -m scenariolab.agents.snapshot --beam 2 --quarters 80 --resume --yes  # continue a paused build

The first real snapshot is beam 2 × 80 quarters (USER_DECISIONS 2026-11-02: reviewed before scaling up); beam 15 or
30 is an explicit --beam. Every call uses claude-opus-5-5 at effort medium. A build stops at its hard cap (1.2 × the
estimate, reason "budget") and continues only with a new confirmation.

Layout of data/agents/snapshots/<id>/ (read-only once status is "done"):

    manifest.json                schema / prompt / model versions, data_end, settings, call counts, sizes, created
    index.json                   node key -> [level, status, parent]
    levels/<d>.nodes.json.gz     every node of level d: expanded, leaf or pruned (with its own mass and the reason)
    levels/<d>.detail.json.gz    node details of the level's expanded nodes (decisions, clusters, consequences, …)
    leaves/<key>.npz / .json     the final leaves' full particle clouds (the portfolio hand-off; extension)
    frontier/<key>.npz / .json   every kept node's cloud after the reduction (float32, no path): what the next level
                                 expanded, so a pruned child replays one quarter from its parent
    journal.jsonl                the build's events
    checkpoint/                  the frontier after the last finished level (deleted when the build is done)
    overlay/                     on-demand expansions (nodes/<key>.json, clouds/<key>.npz, index.json, journal)

Every node lists all of its children, each "expanded" (in the snapshot), "leaf" or "pruned" (its mass reassigned to
the nearest kept node, with the reason). Requesting an unexpanded node rebuilds its particle cloud by replaying one
quarter from its parent's saved cloud (frontier/, or an overlay cloud) with the recorded action cluster and
consequence branch, then runs the cell for one quarter (or a small beam for several); the frozen core is never
written. The parent's cloud is the one the build expanded: pooled with the nodes it absorbed and resampled to N.
Saved clouds are float32, so in rare cases a particle on a partition boundary or a cell at its minimum size comes out
differently; the replay then takes the cell with the same discrete state (else the branch's largest) and says so
(replay_exact false, labelled 近似重放). A snapshot built before frontier/ existed replays from the root along the
node's own path only.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import re
import shutil
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import accounts as AC
from . import actors as A
from . import model_adapter as MA
from . import tree as T

SNAPSHOT_VERSION = "agent-snapshot-1 (2026-11-02)"
ID_RE = re.compile(r"^[a-z0-9_\-]{1,60}$")
# gzip-compressed JSON bytes per record (measured on reduced builds, agent_impl/engine/calib.py, and re-measured
# 2026-11-02 on B 2 × 80 mock builds with both models: about 1,450 bytes per record in the level files)
BYTES_KEPT_RECORD = 2_600
BYTES_PRUNED_RECORD = 1_400
SPLIT_CELLS_TYPICAL = 9.5             # cells per consequence branch at N 500 with a budget ≥ 19 (B 2 × 80 mock builds:
                                      # 9.6 with the quarterly stack, 8.5 with the stub; 11 on the earlier stub builds)
BYTES_DETAIL = 10_800
# per particle, what the store writes (float32, compressed): measured on the pricing model's own probe cloud
# (tree.memory_estimate) when the estimate has one; else these constants, the stub's values deep in a B 2 × 80 tree
# (the quarterly stack's clouds are about 1,440 bytes per particle: more state, less compressible)
LEAF_BYTES_PER_PARTICLE = 970
LEAF_JSON_BYTES = 48_000              # a leaf's record and books metadata 80 quarters deep (B 2 × 80: 47–50 kB)
FRONTIER_BYTES_PER_PARTICLE = 965
FRONTIER_JSON_BYTES = 11_000
EXPAND_BEAM = 4                       # beam of an on-demand extension deeper than one quarter
EXPAND_MAX_QUARTERS = 8
EXPAND_MAX_CALLS = 12                 # per extended quarter (one merged call + the synthesizer + repairs)


def snapshots_dir() -> Path:
    return T.SNAPSHOTS_DIR


def _backend_name() -> str:
    from ..llm import backend
    return backend.backend_name()


def snapshot_root(sid: str, base: Path | None = None) -> Path:
    if not ID_RE.match(str(sid)):
        raise ValueError(f"快照 id「{sid}」不合法：只能包含小写字母、数字、下划线和连字符（最多 60 个字符）")
    return Path(base or snapshots_dir()) / sid


DEFAULT_BEAM = 2                      # the first real snapshot: beam 2 × 80 quarters (the user scales up after review)
DEFAULT_QUARTERS = 80


def default_settings(beam: int = DEFAULT_BEAM, quarters: int = DEFAULT_QUARTERS, **kw) -> T.TreeSettings:
    """Snapshot defaults: no quarterly path history in memory (the hand-off keeps the leaves' clouds), no prior trees,
    no soft call cap (the hard cap, 1.2 × the estimate, pauses the build instead of falling back to base rates), the
    synthesizer gated in the first four quarters."""
    base = dict(depth=int(quarters), B=int(beam), keep_paths="none", prior_trees=False, max_calls=None)
    base.update({k: v for k, v in kw.items() if v is not None})
    return T.TreeSettings.coerce(base)


# =====================================================================================================================
# Estimates: calls, time, tokens (the cost model), memory and size
# =====================================================================================================================

def size_estimate(settings, probe: dict | None = None) -> dict:
    """Snapshot bytes from the measured per-record sizes: expanded nodes (record + detail), pruned candidates (a
    compact record) and the leaves' particle clouds. Candidates per node: K × M_max while the synthesizer runs (its
    partitions branch), about K × 1.3 later (the default partition splits only in stress states)."""
    st = T.TreeSettings.coerce(settings)
    est = T.estimate(st)
    levels = est["levels"]
    synth_depth = int((st.synth or {}).get("depth", 4)) if (st.synth or {}).get("mode", "gated") != "off" else 0
    # the discrete split: each consequence branch splits into about min(budget / 2, 11) cells at N 500, fewer with
    # fewer particles (a cell needs ≥ 5 particles); measured on B 2/4/8 builds with the mock (2026-11-02)
    cells = min(T.split_max_of(st) / 2.0, SPLIT_CELLS_TYPICAL) * min(1.0, (int(st.n_particles) / 500.0) ** 0.6)
    cells = max(1.0, cells)
    probe = probe or {}
    f_bpp = float(probe.get("frontier_bytes_per_particle") or FRONTIER_BYTES_PER_PARTICLE)
    p_bpp = float(probe.get("probe_bytes_per_particle") or f_bpp / T.DEPTH_BYTES_FACTOR)
    l_bpp = float(probe.get("leaf_bytes_per_particle") or LEAF_BYTES_PER_PARTICLE)
    f_json = float(probe.get("frontier_json_bytes") or FRONTIER_JSON_BYTES)
    cands = [n * int(st.K) * (int(st.M_max) if d < synth_depth else 1.3) * cells for d, n in enumerate(levels)]
    kept_next = levels[1:] + [min(int(st.B), int(cands[-1]))]
    pruned = int(sum(max(c - k, 0) for c, k in zip(cands, kept_next)))
    kept = int(sum(levels))
    leaves = kept_next[-1]
    # the kept nodes' clouds level by level (P14 review: one deep factor for every level overstated wide, shallow
    # trees by up to 46%): the root's identical particles compress to almost nothing, a level-1 cloud carries one
    # quarter of dispersion, and a wide level keeps fewer distinct particles per node (T.level_bytes_factor)
    frontier = sum(n * (int(st.n_particles) * p_bpp * T.level_bytes_factor(d, levels[d - 1] if d else 1, int(st.K), n)
                        + f_json) for d, n in enumerate(levels))
    b = kept * (BYTES_KEPT_RECORD + BYTES_DETAIL) + pruned * BYTES_PRUNED_RECORD + \
        leaves * (int(st.n_particles) * l_bpp + LEAF_JSON_BYTES) + frontier
    return {"expanded_nodes": kept, "pruned_nodes": pruned, "leaves": leaves, "bytes": int(b), "cells": round(cells, 1),
            "mb": round(b / 2**20, 1), "frontier_mb": round(frontier / 2**20, 1),
            "frontier_bytes_per_particle": int(round(f_bpp)), "probe_bytes_per_particle": int(round(p_bpp)),
            "measured": bool(probe.get("frontier_bytes_per_particle")),
            "basis": "按缩小规模构建实测的每条记录字节数外推（gzip 后；叶节点粒子 float32；离散状态拆分让每个后果分支"
                     "再分成几个子节点）；另含每个保留节点的粒子云（float32），点开被剪分支时从父节点精确重放一个季度"}


def full_estimate(settings, *, latency=None, cached_levels: int = 0, adapter=None) -> dict:
    """Calls, time and tokens (tree.estimate), memory (tree.memory_estimate on a probe cloud the pricing model
    advanced tree.PROBE_QUARTERS quarters) and size (size_estimate with that probe's bytes per particle);
    `pricing_model` says which model priced it (the default adapter unless one is given)."""
    st = T.TreeSettings.coerce(settings)
    adapter = adapter if adapter is not None else MA.load_adapter()
    out = T.estimate(st, latency=latency, cached_levels=cached_levels)
    out["pricing_model"] = MA.model_info(adapter)
    try:
        out["memory"] = T.memory_estimate(st, adapter=adapter)
    except (KeyError, OSError, ValueError) as err:
        out["memory"] = {"error": str(err)}
    out["size"] = size_estimate(st, probe=out["memory"] if "error" not in out["memory"] else None)
    out["transmission"] = (out["memory"] or {}).pop("transmission", None)
    return out


def engine_transmission(eng: T.Engine) -> dict:
    """What the pricing model's interventions reach, for the manifest, run.json and the UI's honesty card: the
    probe (T.transmission), the inert levers, the template channels (on, the price-puzzle overlay and its first
    kernel terms, the table's coverage), the model's own Fed prior at the root and the lines the card shows."""
    tr = dict(eng.transmission or {})
    ch = {"on": bool(eng.channels_on), "version": T.CH.version() if eng.channels_on else None}
    if eng.channels_on:
        ch["overlay"] = bool(eng.overlay_on)
        ch["coverage"] = T.channel_coverage(T.coverage_actors(eng))
        if ch["overlay"] and eng.layer.kernel is not None:
            ch["kernel_k1_4"] = {f: [T.rnd(float(x), 4) for x in v[1:5]] for f, v in eng.layer.kernel.items()}
        ch["literature_arm"] = eng.st.puzzle_literature_arm == "on"
    root = T.root_prior(eng.model, eng) if eng.structural else {}
    return {"levers": tr.get("levers") or {}, "inert": list(eng.inert), "ff_irf": tr.get("ff_irf"),
            "basis": tr.get("basis"), "channels": ch, "root_prior": root,
            "zh": T.transmission_zh(tr, eng.meta, channels=ch, root=root)}


def format_estimate(est: dict, st: T.TreeSettings) -> str:
    mem = est.get("memory") or {}
    size = est.get("size") or {}
    pm = est.get("pricing_model") or {}
    priced = (f"{pm.get('label')}（{pm.get('fit') or ''}；{MA.short_version(pm.get('identity') or pm.get('version'))}）"
              if not pm.get("stub", True)
              else f"{pm.get('label') or MA.STUB_LABEL}" + (f"（{pm.get('why_stub')}）" if pm.get("why_stub") else ""))
    lines = [f"快照估计：B {st.B} × {st.depth} 季，每节点 {st.n_particles} 个粒子，并发 {est['concurrency']}",
             f"  定价模型：{priced}",
             f"  扩展节点 {est['expanded_nodes']}；大模型调用 {est['calls_new']}（行为者 {est['actor_calls']}，后果合成器 {est['synth_calls']}；"
             f"上限 {st.max_calls if st.max_calls is not None else '无'}）",
             f"  轮次 {est['rounds']}；耗时约 {est['hours'][0]}–{est['hours'][2]} 小时（每次调用 {est['latency_s'][0]:.0f}–"
             f"{est['latency_s'][2]:.0f} 秒的假设）",
             f"  token：输入约 {est['tokens']['input'] / 1e6:.1f}M，输出约 {est['tokens']['output'] / 1e6:.2f}M（假设的大小，未测量）",
             f"  内存峰值约 {mem.get('peak_mb', '?')} MB（候选 {mem.get('candidates_max', '?')} 个云 × 每粒子 "
             f"{mem.get('bytes_per_particle', '?')} 字节）；本机计算约 "
             f"{round(est['expanded_nodes'] * float(mem.get('compute_s_per_node') or 0.3) / 60.0, 1)} 分钟（不含大模型）",
             f"  快照大小约 {size.get('mb', '?')} MB（扩展 {size.get('expanded_nodes')}、剪枝 {size.get('pruned_nodes')}、叶 "
             f"{size.get('leaves')}；保留节点的粒子云约 {size.get('frontier_mb', '?')} MB）"]
    if est.get("deep_model"):
        lines.append(f"  第 {int(st.deep_from) + 1} 季起使用 {est['deep_model']}（{est['deep_calls']} 次调用；你明确指定的）")
    lines.append(f"  模型 {est.get('model') or 'claude-opus-5-5'}，力度 medium；硬上限 {est.get('hard_cap')} 次新调用"
                 "（预计的 1.2 倍：达到后暂停，确认后继续）")
    for ln in ((est.get("transmission") or {}).get("zh") or []):
        lines.append("  传导：" + ln)
    lines.append("  " + est["usage_note"])
    return "\n".join(lines)


# =====================================================================================================================
# Building
# =====================================================================================================================

class BuildPaused(Exception):
    def __init__(self, reason: str, message: str, reset_at: float | None):
        super().__init__(message)
        self.reason, self.message, self.reset_at = reason, message, reset_at


def write_manifest(root: Path, man: dict) -> None:
    T.write_json(root / "manifest.json", man)


def read_manifest(root: Path) -> dict | None:
    p = Path(root) / "manifest.json"
    return T.read_json(p) if p.exists() else None


def real_backend() -> bool:
    from ..llm import backend
    return backend.backend_name() == "claude_code"


def refuse_fallback_stub(adapter, what: str = "快照") -> None:
    """Real Claude Code calls priced by a stub the default fell back to (no quarterly data, or the model failed to
    load) are refused: SIM_DESIGN R-A3 wants the real adapter, and the LLM's decisions depend on model-priced
    packets. The stub is allowed only when MSL_AGENT_MODEL=stub says so."""
    if real_backend() and getattr(adapter, "stub", True) and not MA.explicit_stub():
        why = MA.model_info(adapter).get("why_stub") or ""
        raise ValueError(f"定价模型回退到了桩模型{('（' + why + '）') if why else ''}：用本机 Claude Code 的{what}必须由季度模型定价。"
                         "请先准备季度数据（python -m scenariolab.build），并设置 "
                         f"MSL_AGENT_MODEL={MA.QUARTERLY_SPEC}；只想检查流程时明确设置 MSL_AGENT_MODEL=stub")


def folder_adapter(man: dict | None, *, adapter=None, allow_model_change: bool = False, what: str = "快照"):
    """The adapter that continues a folder (its manifest or run.json): the one it was priced with
    (MA.resume_adapter), whatever MSL_AGENT_MODEL or the server's default says; a given adapter must be that very
    fit. A different fit is refused (ValueError in Chinese) unless allow_model_change."""
    built = ((man or {}).get("versions") or {}).get("model_adapter")
    if adapter is None:
        return MA.resume_adapter(built, is_stub=(man or {}).get("stub"), what=what, allow_change=allow_model_change)
    if built and not MA.same_version(built, adapter) and not allow_model_change:
        raise ValueError(f"这个{what}由 {MA.short_version(built)} 定价，传入的模型是 {MA.short_version(MA.identity(adapter))}："
                         "继续会在同一棵树里混用两次拟合，已拒绝")
    return adapter


def build(sid: str, settings, *, resume: bool = False, base: Path | None = None, adapter=None,
          cache_dir: Path | None = None, listener=None, cancel: threading.Event | None = None,
          root_accounts=None, institutions=None, kill: threading.Event | None = None,
          budget_confirmed: bool = False, allow_model_change: bool = False) -> dict:
    """Build (or continue) the snapshot `sid`. Raises BuildPaused on a pause (usage limit, login, budget, cancel, …):
    the checkpoint keeps every finished level and the cache keeps every finished call, so build(resume=True)
    continues exactly. A resumed build is priced by the adapter the folder was priced with (folder_adapter: never
    the environment's default; another fit only with allow_model_change, recorded in pricing_history and the
    journal). A resume after a "budget" pause raises the hard cap only with budget_confirmed (the user saw and
    confirmed the remaining estimate). Any other error is recorded in the manifest (status "failed")."""
    root = snapshot_root(sid, base)
    st = T.TreeSettings.coerce(settings)
    probs = st.problems()
    if probs:
        raise ValueError("快照设置有错误：" + "；".join(probs))
    man = read_manifest(root)
    provisional = bool(man and man.get("provisional"))
    if man is not None and not provisional:
        if man.get("status") == "done":
            raise ValueError(f"快照「{sid}」已经完成并冻结；换一个 id 重新构建")
        if not resume:
            raise ValueError(f"快照「{sid}」已存在（{man.get('status')}）：用 --resume 继续，或换一个 id")
        if T.TreeSettings.saved(man["settings"]).hash() != st.hash():
            raise ValueError(f"快照「{sid}」的设置与这次不同：继续构建必须用相同的设置（{man.get('settings_hash')}）")
    root.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if man is None or provisional:        # the manifest first: a reader sees "building" before the slow estimate
        man = {"id": sid, "version": SNAPSHOT_VERSION, "created": (man or {}).get("created") or now,
               "settings": st.to_dict(), "settings_hash": st.hash(), "status": "building", "levels_done": 0,
               "spent_calls": 0, "pause": None, "provisional": True}
        write_manifest(root, man)
    resuming = not man.get("provisional")
    try:
        if resuming:
            adapter = folder_adapter(man, adapter=adapter, allow_model_change=allow_model_change)
            drift = T.version_drift(man.get("versions"), adapter)
            if drift:                                     # another engine, catalogue, channel table, prompts …
                raise ValueError(T.drift_refusal(drift, "快照"))
        else:
            adapter = adapter or MA.load_adapter()
        refuse_fallback_stub(adapter)
    except ValueError as err:
        if man.get("provisional"):                    # nothing was built: the provisional manifest goes away
            shutil.rmtree(root, ignore_errors=True)
        else:
            m = read_manifest(root) or man
            m.update(status="paused" if m.get("status") in ("building", "paused") else m.get("status"),
                     error=str(err), updated=now)
            write_manifest(root, m)
        raise
    done = int(man.get("levels_done") or 0)
    if st.hard_cap is None:
        st = dataclasses.replace(st, hard_cap=T.hard_cap_for(st))
    prev_pause = man.get("pause") or man.get("resume_pause") or {}
    if prev_pause.get("reason") == "budget" and budget_confirmed:
        st = dataclasses.replace(st, hard_cap=T.hard_cap_for(st, spent=T.ledger_size(root), cached_levels=done))
    info = MA.model_info(adapter)
    if man.get("provisional"):
        man.update(estimate=full_estimate(st, adapter=adapter), versions=T.versions(adapter),
                   stub=bool(getattr(adapter, "stub", True)), pricing_model=info,
                   export=T.export_status(info, st.keep_paths), data_end="2026-09", provisional=False,
                   backend=_backend_name(),
                   pricing_history=[{"from_level": 0, "identity": info["identity"], "label": info["label"]}])
    else:
        hist = list(man.get("pricing_history") or [{"from_level": 0, "identity": (man.get("versions") or {}).get(
            "model_adapter"), "label": (man.get("pricing_model") or {}).get("label")}])
        if hist[-1].get("identity") != info["identity"] and not MA.same_version(hist[-1].get("identity"), adapter):
            hist.append({"from_level": done, "identity": info["identity"], "label": info["label"]})
        man["pricing_history"] = hist
    man.update(status="building", updated=now, pause=None, resume_pause=None, error=None, settings=st.to_dict(),
               hard_cap=st.hard_cap, model=st.model or "claude-opus-5-5", effort="medium")
    write_manifest(root, man)
    journal = T.Journal(root / "journal.jsonl", listener=listener)
    if len(man.get("pricing_history") or []) > 1 and man["pricing_history"][-1].get("from_level") == done:
        journal.emit("model_changed", level=done, model=info, history=man["pricing_history"])
    store = T.Store(root)
    if institutions is None and man.get("institutions") is not None:
        institutions = man["institutions"]                 # a resumed build keeps the pinned facts it started with
    run = T.TreeRun(st, store, adapter=adapter, journal=journal, cancel=cancel, kill=kill, cache_dir=cache_dir,
                    root_accounts=root_accounts, institutions=institutions)
    if man.get("institutions") is None:
        man["institutions"] = run.engine.items
        man["accounts_root_hash"] = (run.engine.root_acc or {}).get("hash")
    man["transmission"] = engine_transmission(run.engine)
    write_manifest(root, man)
    t0 = time.monotonic()

    def on_level(level, stats):
        m = read_manifest(root) or man
        m.update(updated=time.strftime("%Y-%m-%dT%H:%M:%S"), levels_done=level, levels=run.level_stats,
                 calls=run.llm.stats if run.llm is not None else None, spent_calls=run.calls_paid())
        write_manifest(root, m)
    try:
        res = run.run(resume=True, on_level=on_level)
    except T.Paused as err:
        m = read_manifest(root) or man
        m.update(status="paused", pause={"reason": err.reason, "message": err.message, "reset_at": err.reset_at,
                                         "level": err.level, "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                 updated=time.strftime("%Y-%m-%dT%H:%M:%S"), spent_calls=T.ledger_size(root),
                 calls=run.llm.stats if run.llm is not None else m.get("calls"))
        write_manifest(root, m)
        journal.emit("run_paused", reason=err.reason, message=err.message, reset_at=err.reset_at, level=err.level)
        raise BuildPaused(err.reason, err.message, err.reset_at) from err
    except Exception as err:
        m = read_manifest(root) or man
        m.update(status="failed", error=f"{type(err).__name__}: {err}", updated=time.strftime("%Y-%m-%dT%H:%M:%S"),
                 spent_calls=T.ledger_size(root))
        write_manifest(root, m)
        journal.emit("run_failed", error=f"{type(err).__name__}: {err}")
        raise
    ck = root / "checkpoint"
    if ck.exists():
        shutil.rmtree(ck, ignore_errors=True)
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    m = read_manifest(root) or man
    m.update(status="done", finished=time.strftime("%Y-%m-%dT%H:%M:%S"), tree_hash=res["tree_hash"],
             leaves=res["leaves"], levels=res["levels"], calls=res["llm_stats"], seconds=round(time.monotonic() - t0, 1),
             size_bytes=size, size_mb=round(size / 2**20, 2), pause=None, spent_calls=T.ledger_size(root),
             model=(run.llm.model if run.llm is not None else None),
             deep_model=(run.deep_llm.model if run.deep_llm is not None else None))
    write_manifest(root, m)
    return m


# =====================================================================================================================
# Reading (instant, read-only) and on-demand expansion (overlay)
# =====================================================================================================================

class Explorer:
    """A read-only view of a frozen core (snapshot or run folder) plus its overlay of on-demand expansions."""

    def __init__(self, root: Path, *, adapter=None, cache_dir: Path | None = None, kind: str = "snapshot"):
        self.root = Path(root)
        self.kind = kind
        self.store = T.Store(self.root)
        self.adapter = adapter
        self.cache_dir = cache_dir
        self._levels: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._index = None
        self._clouds: OrderedDict = OrderedDict()
        self.jobs: dict = {}
        self._job_lock = threading.Lock()
        self._expand_lock = threading.Lock()          # one on-demand expansion at a time per folder
        self._shares = None

    def invalidate(self) -> None:
        """Drop cached level files, the index and clouds (a run folder changed: a level finished, an extension)."""
        with self._lock:
            self._levels.clear()
            self._index = None
            self._clouds.clear()
            self._shares = None

    def action_shares(self) -> list:
        """各季动作分布: per expanded level, each actor's pooled option probabilities averaged over the level's expanded
        nodes by path probability (computed before reduction, so exact under the tree model; cached)."""
        with self._lock:
            if getattr(self, "_shares", None) is not None:
                return self._shares
        out = []
        for d in self.store.levels():
            det = self.level(d, "detail")
            lv = self.level(d)
            if not det or not lv:
                continue
            by_key = lv.get("by_key") or {}
            acc: dict = {}
            tot = 0.0
            quarter = None
            for key, dt in (det.get("details") or {}).items():
                p = float((by_key.get(key) or {}).get("p_path") or 0.0)
                tot += p
                quarter = dt.get("quarter_next") or quarter
                for a in dt.get("next") or []:
                    row = acc.setdefault(a["actor"], {})
                    for o in a.get("options") or []:
                        row[o["id"]] = row.get(o["id"], 0.0) + p * float(o.get("p_used") or 0.0)
            if tot <= 0:
                continue
            out.append({"level": d, "quarter": quarter,
                        "actors": {a: {o: T.rnd(v / tot, 4) for o, v in row.items()} for a, row in acc.items()}})
        with self._lock:
            self._shares = out
        return out

    # ---------------------------------------------------------------- core
    def manifest(self) -> dict:
        m = read_manifest(self.root)
        if m is None:
            p = self.root / "run.json"
            m = T.read_json(p) if p.exists() else {}
        return m

    def settings(self) -> T.TreeSettings:
        return T.TreeSettings.saved(self.manifest().get("settings") or {})

    def index(self) -> dict:
        with self._lock:
            if self._index is None:
                p = self.root / "index.json"
                self._index = T.read_json(p) if p.exists() else {}
                if not self._index:
                    for d in self.store.levels():
                        for r in (self.store.read_level(d) or {}).get("nodes", []):
                            self._index[r["key"]] = [d, r["status"], r.get("parent")]
            return self._index

    def level(self, d: int, what: str = "nodes") -> dict | None:
        key = (int(d), what)
        with self._lock:
            if key in self._levels:
                self._levels.move_to_end(key)
                return self._levels[key]
        lv = self.store.read_level(int(d), what)
        if lv is None:
            return None
        if what == "nodes":
            lv = {**lv, "by_key": {r["key"]: r for r in lv["nodes"]}}
        with self._lock:
            self._levels[key] = lv
            while len(self._levels) > 24:
                self._levels.popitem(last=False)
        return lv

    def core_record(self, key: str) -> dict | None:
        ent = self.index().get(key)
        if ent is None:
            return None
        lv = self.level(int(ent[0]))
        return copy.deepcopy((lv or {}).get("by_key", {}).get(key))

    def core_detail(self, key: str) -> dict | None:
        ent = self.index().get(key)
        if ent is None:
            return None
        lv = self.level(int(ent[0]), "detail")
        return copy.deepcopy(((lv or {}).get("details") or {}).get(key))

    # ---------------------------------------------------------------- overlay
    @property
    def overlay(self) -> Path:
        return self.root / "overlay"

    def overlay_index(self) -> dict:
        p = self.overlay / "index.json"
        return T.read_json(p) if p.exists() else {}

    def overlay_entry(self, key: str) -> dict | None:
        p = self.overlay / "nodes" / f"{key}.json"
        return T.read_json(p) if p.exists() else None

    def record(self, key: str) -> dict | None:
        rec = self.core_record(key)
        if rec is not None:
            rec["source"] = "snapshot"
        else:
            ent = self.overlay_entry(key)
            if ent is None:
                return None
            rec = dict(ent["record"])
            rec["source"] = "overlay"
        ov = self.overlay_entry(key)
        if ov is not None and ov.get("detail") is not None:
            rec["overlay_children"] = ov["record"].get("children") or []
            rec["overlay"] = {"expanded_at": ov.get("expanded_at"), "replayed": ov.get("replayed"),
                              "replay_exact": ov.get("replay_exact"), "quarters": ov.get("quarters"),
                              "model_changed": bool(ov.get("model_changed")),
                              "versions_changed": ov.get("versions_changed"),
                              "model_fit": ov.get("model_fit") or MA.short_version(ov.get("model_version"))}
        for ch in rec.get("children") or []:
            if ch.get("status") == "pruned" and self.overlay_entry(ch["key"]) is not None:
                ch["overlay"] = "expanded"
        return rec

    def detail(self, key: str) -> dict | None:
        d = self.core_detail(key)
        if d is not None:
            return d
        ov = self.overlay_entry(key)
        return None if ov is None else ov.get("detail")

    def path(self, key: str) -> list[str]:
        out, seen = [], set()
        cur = key
        while cur and cur not in seen:
            seen.add(cur)
            out.append(cur)
            rec = self.core_record(cur)
            if rec is None:
                ov = self.overlay_entry(cur)
                rec = (ov or {}).get("record")
            cur = (rec or {}).get("parent")
        return list(reversed(out))

    def best_leaf(self, key: str) -> str:
        """The most probable kept descendant of `key` at the deepest level its subtree reaches (core and overlay):
        the comic plays root → key → that leaf, so a path never stops early at a beam-search dead end."""
        idx = self.index()
        parent = {k: v[2] for k, v in idx.items()}
        level = {k: int(v[0]) for k, v in idx.items()}
        kept = {k for k, v in idx.items() if v[1] != "pruned"}
        for k, v in self.overlay_index().items():
            parent.setdefault(k, v.get("parent"))
            level.setdefault(k, int(v.get("level") or 0))
            kept.add(k)
        memo: dict = {key: True}

        def under(k) -> bool:
            chain = []
            cur = k
            while cur is not None and cur not in memo:
                chain.append(cur)
                cur = parent.get(cur)
            res = memo.get(cur, False) if cur is not None else False
            for x in chain:
                memo[x] = res
            return res
        by_level: dict = {}
        for k in kept:
            if k != key and level.get(k, -1) > level.get(key, 0):
                by_level.setdefault(level[k], []).append(k)
        for lv in sorted(by_level, reverse=True):
            mine = [k for k in by_level[lv] if under(k)]
            if not mine:
                continue

            def mass(k):
                r = self.record(k) or {}
                return float(r.get("p_path_own") if r.get("source") == "overlay" and r.get("p_path_own") is not None
                             else r.get("p_path") or 0.0)
            return max(sorted(mine), key=mass)
        return key

    def summary(self) -> dict:
        m = self.manifest()
        root_key = next((k for k, v in self.index().items() if v[0] == 0 and v[1] != "pruned"), None)
        lv = {}
        for d in self.store.levels():
            nodes = (self.level(d) or {}).get("nodes", [])
            lv[d] = {"nodes": len(nodes), "expanded": sum(1 for r in nodes if r["status"] in ("expanded", "leaf")),
                     "pruned": sum(1 for r in nodes if r["status"] == "pruned"),
                     "quarter": nodes[0]["quarter"] if nodes else None,
                     "mass_kept": T.rnd(sum(r["p_path"] for r in nodes if r["status"] != "pruned"), 6)}
        return {"manifest": m, "root": root_key, "levels": lv, "overlay": self.overlay_index(),
                "read_only": m.get("status") == "done"}

    # ---------------------------------------------------------------- clouds: saved leaves, overlay, replay
    def institutions(self):
        """The pinned facts the build started with (so a later edit of institutions.json cannot change the root)."""
        return self.manifest().get("institutions")

    def model(self):
        """The adapter this folder was priced with (MA.adapter_for_version): the stub for a stub build, the quarterly
        stack for a quarterly build whatever MSL_AGENT_MODEL says (a refit of the stack keeps the state layout, so
        the saved clouds replay; the overlay entry records model_changed), else the default adapter of the same
        family. A different family cannot extend the folder: ValueError in Chinese."""
        if self.adapter is not None:
            return self.adapter
        m = self.manifest()
        self.adapter = MA.adapter_for_version((m.get("versions") or {}).get("model_adapter"), is_stub=m.get("stub"),
                                              what="快照" if self.kind == "snapshot" else "推演")
        return self.adapter

    def model_changed(self) -> bool:
        """The adapter that prices on-demand branches is another fit than the one the frozen core was priced with
        (same family: a refit after the data changed). Equal content identities are the same fit."""
        built = (self.manifest().get("versions") or {}).get("model_adapter")
        try:
            return bool(built) and not MA.same_version(built, self.model())
        except ValueError:
            return True

    def _engine(self, llm=None, deep_llm=None, tree_kind=None):
        st = self.settings()
        if tree_kind:
            st = dataclasses.replace(st, tree_kind=tree_kind)
        return T.Engine(st, adapter=self.model(), llm=llm, deep_llm=deep_llm,
                        root_accounts=AC.load_root(), institutions=self.institutions())

    def _saved_cloud(self, key: str) -> T.Cloud | None:
        with self._lock:
            if key in self._clouds:
                self._clouds.move_to_end(key)
                return self._clouds[key]
        cl = self.store.load_leaf_cloud(key)
        if cl is None:
            cl = self.store.load_frontier_cloud(key)
        if cl is None:
            p = self.overlay / "clouds" / f"{key}.npz"
            j = self.overlay / "clouds" / f"{key}.json"
            if p.exists() and j.exists():
                with np.load(p, allow_pickle=False) as z:
                    arrays = {k: (z[k].astype(np.float64) if z[k].dtype == np.float32 else z[k]) for k in z.files}
                cl = T.cloud_restore(arrays, (T.read_json(j) or {}).get("acc_meta") or {}, "c|")
        if cl is not None:
            self._remember(key, cl)
        return cl

    def _remember(self, key: str, cl: T.Cloud) -> None:
        with self._lock:
            self._clouds[key] = cl
            while len(self._clouds) > 16:
                self._clouds.popitem(last=False)

    def node_with_cloud(self, key: str, eng: T.Engine) -> tuple[T.Node, bool]:
        """(node with its particle cloud, replayed?): a saved cloud when one exists, else the lineage replayed."""
        rec = self.record(key)
        if rec is None:
            raise KeyError(f"找不到节点「{key}」")
        nd = T.Node.from_record(rec)
        if not nd.memory and nd.parent:
            prec = self.record(nd.parent) or {}
            nd.memory = T.rebuild_memory(prec.get("memory") or "", nd.memory_line, nd.facts)
        cl = self._saved_cloud(key)
        if cl is not None:
            nd.cloud = cl
            return nd, False
        chain = self.path(key)
        start, cloud = None, None
        for i in range(len(chain) - 1, -1, -1):
            c = self._saved_cloud(chain[i])
            if c is not None:
                start, cloud = i, c
                break
        drift = T.version_drift(self.manifest().get("versions"), eng.model)
        if drift and start != len(chain) - 1:
            raise ValueError(f"这个{'快照' if self.kind == 'snapshot' else '推演'}由另一版代码构建（{T.drift_zh(drift)}已改变）："
                             "节点的粒子云无法按路径重放，只有保存了粒子云的末端可以继续生成；请用当前版本重新构建")
        if cloud is None:
            root = eng.root()
            if root.key != chain[0]:
                raise ValueError("快照的起点与当前代码不一致（起点状态、制度现状或模型版本已变）：无法重放")
            start, cloud = 0, root.cloud
        cur = T.Node.from_record(self.record(chain[start]))
        cur.cloud = cloud
        exact = True
        for k in chain[start + 1:]:
            child_rec = self.record(k)
            cur = self._replay_step(eng, cur, child_rec)
            exact = exact and cur.replay_exact
        nd.cloud, nd.replay_exact = cur.cloud, exact
        self._remember(key, nd.cloud)
        return nd, True

    def _replay_step(self, eng: T.Engine, parent: T.Node, child_rec: dict) -> T.Node:
        det = self.detail(parent.key)
        if det is None:
            raise ValueError(f"节点 {parent.key} 没有保存决策细节：无法重放")
        cl_id = (child_rec.get("cluster") or {}).get("id")
        br_id = (child_rec.get("branch") or {}).get("id")
        dec = dec_from_detail(det)
        cl = next(c for c in det["clusters"] if c["id"] == cl_id)
        cluster = {"id": cl["id"], "label": cl["label"], "p": float(cl["p"]), "head": cl["head"],
                   "members": [{"options": m["options"], "p": float(m["p"])} for m in cl["members"]],
                   "head_key": T.canonical(cl["head"])}
        cons = next(c for c in det["consequences"] if c["cluster"] == cl_id)
        branch = next(b for b in cons["branches"] if b["id"] == br_id)
        branch = dict(branch)
        q = A.quarter_add(parent.quarter, 1)
        a0 = eng.a0(parent)
        alloc = eng.allocate_cluster(parent, cluster, dec, a0, T.seed_for(eng.st.seed, parent.key,
                                                                          f"alloc|{cluster['head_key']}"))
        bk = eng.build_bk(parent, cluster, dec, a0, alloc, T.seed_for(eng.st.seed, parent.key,
                                                                      f"bk|{cluster['head_key']}"))
        bk["alloc"] = alloc
        bk = eng.sanitize(parent, q, cluster, bk)
        bk = eng.apply_nature(parent, cluster, bk, dec, {})
        if cl.get("nature") and bk.get("nature") is None:
            bk["nature"] = dict(cl["nature"])
        bk["split"] = eng.discrete_split(parent, dec, bk["alloc"])
        chs = eng.realize(parent, q, cluster, bk, branch, a0, dec)
        ch = next((c for c in chs if c.key == child_rec.get("key")), None)
        if ch is None and chs:
            # float32 rounding moved a particle across a boundary: the cell with the same discrete state, else the
            # branch's largest; it keeps the recorded key so the overlay hangs under the node the reader clicked
            sig = (child_rec.get("lineage") or [[]])[-1][3:4]
            ch = next((c for c in chs if c.lineage[-1][3:4] == sig), None) or max(chs, key=lambda c: c.cloud.n)
            ch.key, ch.lineage = child_rec["key"], list(child_rec.get("lineage") or ch.lineage)
            approx = True
        else:
            approx = False
        if ch is None:
            raise ValueError(f"重放失败：节点 {child_rec.get('key')} 的分支在重放的粒子里为空")
        out = T.Node.from_record(child_rec)
        n = int(eng.st.n_particles)
        cloud = ch.cloud
        if cloud.n != n:                       # a child keeps its natural particle count; the replay continues with N
            cloud = cloud.take(T.systematic(np.ones(cloud.n), n, T.seed_for(eng.st.seed, ch.key, "replay")))
        out.cloud = cloud
        out.replay_exact = not approx
        return out

    # ---------------------------------------------------------------- on-demand expansion
    def expandable(self, key: str) -> tuple[bool, str]:
        rec = self.record(key)
        if rec is None:
            return False, f"找不到节点「{key}」"
        if rec.get("overlay_children"):
            return False, "该节点已在覆盖层中展开"
        if rec.get("status") == "expanded" and rec.get("source") == "snapshot":
            return False, "该节点已在快照中展开"
        return True, ""

    def expand_estimate(self, key: str, quarters: int = 1) -> dict:
        """Calls of an on-demand expansion: one merged call per expanded node, the synthesizer where the node's level
        is within the synthesizer's depth (upper bound), a beam of EXPAND_BEAM beyond the first quarter; the cap is
        EXPAND_MAX_CALLS per quarter."""
        quarters = int(min(max(int(quarters), 1), EXPAND_MAX_QUARTERS))
        rec = self.record(key) or {}
        st = self.settings()
        level = int(rec.get("level") or 0)
        sy = st.synth or {}
        nodes, calls, n = [], 0, 1
        for qi in range(quarters):
            nodes.append(n)
            lv = level + qi
            synth = sy.get("mode", "gated") == "every" or (sy.get("mode", "gated") == "gated" and lv < int(sy.get("depth", 4)))
            calls += n * (2 if synth else 1)
            n = min(EXPAND_BEAM, n * int(st.K) * int(st.M_max))
        return {"node": key, "quarters": quarters, "nodes": nodes, "calls_new": int(calls),
                "max_calls": EXPAND_MAX_CALLS * quarters, "model": st.model or "claude-opus-5-5", "effort": "medium",
                "level": level}

    def expand(self, key: str, *, quarters: int = 1, llm_factory=None, confirmed: bool = False,
               journal_listener=None, cancel: threading.Event | None = None, max_calls: int | None = None) -> dict:
        """Run the cell for an unexpanded node (pruned, a leaf, or an overlay leaf) for `quarters` quarters (a beam of
        EXPAND_BEAM beyond the first) and save the result in the overlay. Returns the overlay entry of `key`."""
        quarters = int(min(max(int(quarters), 1), EXPAND_MAX_QUARTERS))
        ok, why = self.expandable(key)
        if not ok:
            ent = self.overlay_entry(key)
            if ent is not None:
                return ent
            raise ValueError(why)
        st = self.settings()
        st = dataclasses.replace(st, confirmed=bool(confirmed), max_calls=max_calls or EXPAND_MAX_CALLS * quarters,
                                 prior_trees=False)
        journal = T.Journal(self.overlay / "journal.jsonl", listener=journal_listener)
        cancel = cancel or threading.Event()
        llm = deep = None
        model = self.model()
        versions = T.versions(model)
        if st.tree_kind == "agent":
            llm = (llm_factory or T.make_llm)(st, cancel=cancel, journal=journal, cache_dir=self.cache_dir,
                                              versions=versions)
            if st.deep_model:
                deep = (llm_factory or T.make_llm)(st, cancel=cancel, journal=journal, cache_dir=self.cache_dir,
                                                   model=st.deep_model, versions=versions)
        eng = T.Engine(st, adapter=model, llm=llm, deep_llm=deep,
                       root_accounts=AC.load_root(), journal=journal, institutions=self.institutions())
        nd, replayed = self.node_with_cloud(key, eng)
        model_changed = self.model_changed()
        # a saved cloud continued under other behaviour versions: allowed (an explicit on-demand branch), recorded on
        # every overlay entry it writes and in the journal, and shown on the node
        drift = T.version_drift(self.manifest().get("versions"), model)
        if drift:
            journal.emit("versions_changed", node=key, parts=drift)
        nd.status = "pending"
        journal.emit("expand_started", node=key, quarters=quarters, replayed=replayed)
        try:
            frontier = [nd]
            entries, saved = [], []
            for qi in range(quarters):
                results = []
                for f in sorted(frontier, key=lambda x: x.key):
                    children, det = eng.expand(f)
                    results.append((f, children, det))
                cands = [c for _, ch, _ in results for c in ch]
                last = qi == quarters - 1
                kept, pruned, stats = T.reduce_level(cands, EXPAND_BEAM, eps=float(st.eps_merge),
                                                     fact_penalty=float(st.fact_penalty), n=int(st.n_particles),
                                                     seed=int(st.seed), keep_all=quarters == 1,
                                                     scales=eng.key_scales)
                keep_keys = {c.key for c in kept}
                for f, children, det in results:
                    f.children = [{"key": c.key, "status": ("leaf" if last else "expanded") if c.key in keep_keys
                                   else "pruned", "p_cond": T.rnd(c.p_cond, 6), "own_mass": T.rnd(c.own_mass, 6),
                                   "label": (c.cluster or {}).get("label"), "branch": (c.branch or {}).get("label"),
                                   "kind": (c.branch or {}).get("kind"), "reason": c.reason,
                                   "absorbed_into": c.absorbed_into} for c in children]
                    f.status = "expanded"
                    entries.append((f, det, children))
                for c in pruned:
                    entries.append((c, None, []))
                for c in kept:
                    c.status = "leaf" if last else "pending"
                    saved.append(c)
                frontier = kept
            for c in frontier:
                entries.append((c, None, []))
        except Exception as err:
            from .llm_io import LLMPause
            journal.emit("expand_failed", node=key, error=str(err))
            if isinstance(err, LLMPause):
                raise BuildPaused(err.reason, str(err), getattr(err, "reset_at", None)) from err
            raise
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        idx = self.overlay_index()
        for node, det, _ in entries:
            ent_path = self.overlay / "nodes" / f"{node.key}.json"
            prev = T.read_json(ent_path) if ent_path.exists() else None
            if prev is not None and prev.get("detail") is not None and det is None:
                continue
            rec = node.record()
            rec["p_path_own"] = rec["p_path"]
            T.write_json(ent_path, {"record": rec, "detail": det, "expanded_at": now if det is not None else None,
                                    "model_version": MA.identity(model), "model_changed": model_changed,
                                    "versions_changed": T.drift_zh(drift) if drift else None,
                                    "model_fit": MA.fit_text(model),
                                    "replayed": replayed if node.key == key else None,
                                    "replay_exact": getattr(node, "replay_exact", None) if node.key == key else None,
                                    "quarters": quarters if node.key == key else None, "anchor": key})
            idx[node.key] = {"parent": node.parent, "level": node.level, "status": node.status,
                             "expanded": det is not None, "anchor": key}
        for c in saved:                        # every kept node: what the next step expanded, or a final leaf
            if c.cloud is not None:
                arr, meta = T.cloud_arrays(c.cloud, "c|")
                arr = {k: MA.to_float32(v) for k, v in arr.items()}
                d = self.overlay / "clouds"
                d.mkdir(parents=True, exist_ok=True)
                tmp = d / f".{c.key}.{os.getpid()}.tmp.npz"
                np.savez_compressed(tmp, **arr)
                os.replace(tmp, d / f"{c.key}.npz")
                T.write_json(d / f"{c.key}.json", {"acc_meta": meta})
        T.write_json(self.overlay / "index.json", idx)
        journal.emit("expand_done", node=key, nodes=len(entries),
                     calls_new=(llm.stats["calls_new"] if llm is not None else 0))
        return self.overlay_entry(key)

    # ---------------------------------------------------------------- background expansion jobs
    def submit(self, key: str, *, quarters: int = 1, llm_factory=None, confirmed: bool = False) -> dict:
        jid = f"{key}-q{int(quarters)}"
        with self._job_lock:
            job = self.jobs.get(jid)
            if job is not None and job["status"] in ("queued", "running"):
                return dict(job)
            job = {"id": jid, "node": key, "quarters": int(quarters), "status": "queued", "error": None,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self.jobs[jid] = job

        def work():
            with self._expand_lock:
                job["status"] = "running"
                try:
                    self.expand(key, quarters=quarters, llm_factory=llm_factory, confirmed=confirmed)
                    job["status"] = "done"
                except BuildPaused as err:
                    job.update(status="paused", error=err.message, reason=err.reason, reset_at=err.reset_at)
                except Exception as err:  # noqa: BLE001 - reported to the UI in Chinese
                    job.update(status="failed", error=f"{type(err).__name__}: {err}")
                job["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        threading.Thread(target=work, daemon=True, name=f"expand-{key[:8]}").start()
        return dict(job)

    def job(self, jid: str) -> dict | None:
        with self._job_lock:
            j = self.jobs.get(jid)
            return None if j is None else dict(j)

    # ---------------------------------------------------------------- the leaf mixture
    def branches(self, depth: int | None = None, assets=("us_equity", "nasdaq", "treasury_10y", "gold")) -> dict:
        levels = self.store.levels()
        if not levels:
            return {"leaves": []}
        d = levels[-1] if depth is None else min(int(depth), levels[-1])
        nodes = [r for r in (self.level(d) or {}).get("nodes", []) if r["status"] != "pruned"]
        leaves = []
        mix_vals = {a: [] for a in assets}
        mix_w = []
        for r in sorted(nodes, key=lambda r: -r["p_path"]):
            leaves.append({"key": r["key"], "p": T.rnd(r["p_path"], 6), "own_mass": T.rnd(r["own_mass"], 6),
                           "label": (r.get("cluster") or {}).get("label"), "branch": (r.get("branch") or {}).get("label"),
                           "assets": {a: (r.get("summary") or {}).get("assets", {}).get(a) for a in assets},
                           "headline": (r.get("summary") or {}).get("headline")})
            cl = self.store.load_leaf_cloud(r["key"]) if d == levels[-1] else None
            if cl is not None:
                for a in assets:
                    mix_vals[a].append(cl.cum[a])
                mix_w.append(np.full(cl.n, r["p_path"] / cl.n))
        mixture = {}
        if mix_w:
            w = np.concatenate(mix_w)
            for a in assets:
                v = np.concatenate(mix_vals[a])
                mixture[a] = [T.rnd(x, 4) for x in T.wquantiles(v, w, (0.05, 0.25, 0.5, 0.75, 0.95))]
        return {"depth": d, "quarter": nodes[0]["quarter"] if nodes else None, "leaves": leaves,
                "mixture_cum": mixture, "mass": T.rnd(sum(r["p_path"] for r in nodes), 6),
                "action_shares": self.action_shares()}


def dec_from_detail(det: dict) -> dict:
    """The decisions a replay needs (pooled options, attributes, followers, hazards) from a node detail."""
    actors, order = {}, []
    for a in det.get("next") or []:
        opts = []
        for o in a["options"]:
            oo = {"id": o["id"], "label": o["label"], "p_used": float(o["p_used"] or 0.0), "p_prior": o.get("p_prior"),
                  "magnitude": o.get("magnitude"), "signal": o.get("signal"), "status_quo": bool(o.get("status_quo")),
                  "bin": tuple(o["bin"]) if o.get("bin") else None, "custom": o.get("custom"),
                  "param": o.get("param"), "source": o.get("source")}
            opts.append(oo)
        actors[a["actor"]] = {"id": a["actor"], "zh": a["zh"], "domain": a.get("domain"), "options": opts,
                              "cond_prior": a.get("cond_prior"), "marginal_of_rows": a.get("marginal_of_rows"),
                              "attributes": [{"domain": d["domain"], "zh": d["zh"],
                                              "options": [{"id": o["id"], "label": o["label"], "p": float(o["p"] or 0)}
                                                          for o in d["options"]]} for d in a.get("attributes") or []],
                              "branching": bool(a.get("branching")), "follower_of": a.get("follower_of"),
                              "cond": a.get("cond"), "flags": a.get("flags") or [], "c": a.get("c", 0.0),
                              "c_used": a.get("c_used", a.get("c", 0.0)), "prior": dict(a.get("prior") or {})}
        order.append(a["actor"])
    hz = {h: {"m": float(v.get("m") or 0.0), "p": float(v.get("p") or 0.0), "zh": v.get("zh", h)}
          for h, v in (det.get("hazards") or {}).items()}
    return {"actors": actors, "order": order, "hazards": hz, "flags": [], "packets": {}}


def list_snapshots(base: Path | None = None) -> list[dict]:
    root = Path(base or snapshots_dir())
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        m = read_manifest(d) if d.is_dir() else None
        if m:
            out.append({k: m.get(k) for k in ("id", "status", "created", "finished", "settings_hash", "stub",
                                              "size_mb", "levels_done", "tree_hash", "pause")} |
                       {"B": (m.get("settings") or {}).get("B"), "depth": (m.get("settings") or {}).get("depth"),
                        "keep_paths": (m.get("settings") or {}).get("keep_paths"),
                        "model_kind": (m.get("pricing_model") or {}).get("kind") or ("stub" if m.get("stub") else None),
                        "model_label": (m.get("pricing_model") or {}).get("label"),
                        "model_fit": (m.get("pricing_model") or {}).get("fit"),
                        "pricing_model": {k: (m.get("pricing_model") or {}).get(k) for k in ("kind", "stub", "label",
                                                                                              "fit", "trial")}
                        if m.get("pricing_model") else None})
    return out


# =====================================================================================================================
# CLI
# =====================================================================================================================

def _wait_until(reset_at: float | None, fallback_s: float) -> None:
    target = (reset_at + 60.0) if reset_at else (time.time() + fallback_s)
    while True:
        left = target - time.time()
        if left <= 0:
            return
        time.sleep(min(left, 60.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scenariolab.agents.snapshot",
                                 description="构建智能体推演的快照（beam search，冻结后供界面读取）")
    ap.add_argument("--beam", type=int, default=DEFAULT_BEAM,
                    help=f"每层保留的节点数 B（默认 {DEFAULT_BEAM}：第一个真实快照先用 2，检查后再用 15 或 30）")
    ap.add_argument("--quarters", type=int, default=DEFAULT_QUARTERS, help="推演的季度数（默认 80 = 20 年）")
    ap.add_argument("--id", default=None, help="快照 id（默认 b<beam>_q<quarters>）")
    ap.add_argument("--resume", action="store_true", help="从上一个检查点继续")
    ap.add_argument("--dry-run", action="store_true", help="只打印估计，不调用大模型")
    ap.add_argument("--yes", action="store_true", help="确认开始（使用本机 Claude Code 时必须给出）")
    ap.add_argument("--particles", type=int, default=500)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-calls", type=int, default=None,
                    help="软上限：超过后节点只用历史基率（默认不设；快照靠硬上限暂停而不是悄悄回退）")
    ap.add_argument("--hard-cap", type=int, default=None,
                    help="硬上限：新调用达到它时暂停（默认预计调用的 1.2 倍）；继续时按剩余部分重新估计")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--model", default=None, help="模型（默认 MSL_LLM_MODEL 或 claude-opus-5-5）")
    ap.add_argument("--deep-model", default=None,
                    help="可选：第 deep-from 季之后换用的模型（默认不换：每一步都是 claude-opus-5-5、力度 medium）")
    ap.add_argument("--deep-from", type=int, default=8, help="从第几层开始用 --deep-model（默认 8 = 前 8 季用主模型）")
    ap.add_argument("--synth-depth", type=int, default=4)
    ap.add_argument("--channels", choices=("on", "off"), default="on",
                    help="模板通道（默认 on；off = P14 的转移支付响应表与 QE 路径）")
    ap.add_argument("--puzzle-overlay", choices=("auto", "off"), default="auto",
                    help="价格之谜的中性修正（默认 auto：传导探针仍发现价格之谜时启用）")
    ap.add_argument("--puzzle-literature-arm", choices=("on", "off"), default="off",
                    help="在中性修正之上再加文献的去通胀效应（默认 off：用户决定）")
    ap.add_argument("--no-wait", action="store_true", help="达到用量上限时不等待，直接退出（之后用 --resume 继续）")
    ap.add_argument("--allow-model-change", action="store_true",
                    help="继续构建时允许换用新的季度模型拟合（默认拒绝：一棵树只用一次拟合；切换会记在 manifest 与日志里）")
    args = ap.parse_args(argv)
    st = default_settings(args.beam, args.quarters, n_particles=args.particles, K=args.K, seed=args.seed,
                          max_calls=args.max_calls, concurrency=args.concurrency, model=args.model,
                          deep_model=args.deep_model, deep_from=args.deep_from, hard_cap=args.hard_cap,
                          synth={"mode": "gated", "depth": args.synth_depth}, channels=args.channels,
                          puzzle_overlay=args.puzzle_overlay, puzzle_literature_arm=args.puzzle_literature_arm)
    sid = args.id or f"b{args.beam}_q{args.quarters}"
    probs = st.problems()
    if probs:
        print("设置有错误：" + "；".join(probs), file=sys.stderr)
        return 2
    adapter = None
    done0 = 0
    try:
        man0 = read_manifest(snapshot_root(sid)) if args.resume else None
        if man0 is not None and not man0.get("provisional"):
            # a resumed build keeps the settings it started with and is priced by the model it was priced with
            st = T.TreeSettings.saved(man0.get("settings") or st.to_dict())
            adapter = folder_adapter(man0, allow_model_change=args.allow_model_change)
            done0 = int(man0.get("levels_done") or 0)
        else:
            adapter = MA.load_adapter()
        est = full_estimate(st, adapter=adapter, cached_levels=done0)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 2
    print(format_estimate(est, st))
    if args.dry_run:
        return 0
    from ..llm import backend
    real = backend.backend_name() == "claude_code"
    try:
        refuse_fallback_stub(adapter)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 2
    if real and not args.yes:
        if not sys.stdin.isatty():
            print("使用本机 Claude Code 构建快照需要确认：加上 --yes。", file=sys.stderr)
            return 3
        ans = input("这会调用本机 Claude Code 并计入你的订阅用量。确认开始？(y/N) ").strip().lower()
        if ans not in ("y", "yes"):
            print("已取消。")
            return 1
    st = dataclasses.replace(st, confirmed=True)
    t0 = time.monotonic()

    def listener(ev):
        if ev.get("type") == "level_done":
            done = int(ev.get("level") or 0)
            el = time.monotonic() - t0
            eta = el / max(done, 1) * (st.depth - done)
            print(f"[{time.strftime('%H:%M:%S')}] 第 {done}/{st.depth} 层：候选 {ev.get('candidates')}，保留 {ev.get('kept')}，"
                  f"重新分配质量 {ev.get('reassigned')}，新调用累计 {ev.get('calls_new')}，预计还需 {eta / 3600:.1f} 小时",
                  flush=True)
        elif ev.get("type") == "run_paused":
            print(f"[{time.strftime('%H:%M:%S')}] 暂停：{ev.get('message')}", flush=True)
    resume = bool(args.resume)
    while True:
        try:
            man = build(sid, st, resume=resume, listener=listener, budget_confirmed=True, adapter=adapter,
                        allow_model_change=args.allow_model_change)
            print(f"快照完成：{snapshot_root(sid)}（{man.get('size_mb')} MB，树哈希 {man.get('tree_hash')}；"
                  f"定价：{(man.get('pricing_model') or {}).get('label') or '—'}）")
            return 0
        except BuildPaused as err:
            if err.reason == "usage_limit" and not args.no_wait and err.reset_at:
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(err.reset_at))
                print(f"达到用量上限：等到 {when} 再继续（Ctrl-C 退出，之后用 --resume 继续）", flush=True)
                try:
                    _wait_until(err.reset_at, 3600.0)
                except KeyboardInterrupt:
                    return 75
                resume = True
                continue
            if err.reason == "usage_limit":
                print("达到用量上限，但 Claude Code 没有给出恢复时间：不自动重试。额度恢复后用 --resume 继续。", file=sys.stderr)
            elif err.reason == "budget":
                print(f"{err.message} 用 --resume --yes 确认后继续。", file=sys.stderr)
            else:
                print(f"构建暂停（{err.reason}）：{err.message}。之后用 --resume 继续。", file=sys.stderr)
            return 75
        except ValueError as err:
            print(str(err), file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("已中断；之后用 --resume 继续。", file=sys.stderr)
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
