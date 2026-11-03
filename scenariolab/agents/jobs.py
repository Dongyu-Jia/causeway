"""Background jobs of the agent tree (AGENT_DESIGN §11.2, §11.5, §12.1): one active run at a time, progress through
the journal, cancel, resume from the last level checkpoint, saved runs under data/agents/runs/<run_id>/.

    run.json                 id, name, kind, status (queued | running | paused | done | failed | cancelled),
                             pause {reason, message, reset_at}, settings, settings_hash, versions, stub, estimate,
                             levels, stats, tree_hash, leaves, error, parent_run, created, updated
    journal.jsonl, index.json, levels/, leaves/, checkpoint/       the Store layout of tree.py
    prior_trees/<base_rate|model>/                                 the two zero-call reference trees
    overlay/                                                       on-demand expansions (snapshot.Explorer)

Real Claude Code runs only from an explicit, confirmed user action: with the claude_code backend a start, resume or
extend needs confirm=True (the user saw the estimate). Every run and snapshot build has a hard cap (G9: 1.2 × the
estimate): reaching it pauses with reason "budget", and continuing needs a new confirmation (the cap is raised to the
calls already paid plus 1.2 × the estimate of what is left). A usage-limit pause records the reset time; the manager
resumes on its own after that time only when the CLI said when (no retry loop without a reset time). Cancel is
graceful (calls in flight finish and are cached) unless force kills them. A server restart turns runs and snapshot
builds left "running" / "building" into "paused" (server_restart).
"""
from __future__ import annotations

import dataclasses
import shutil
import threading
import time
import uuid
from pathlib import Path

from ..llm import backend
from . import model_adapter as MA
from . import snapshot as S
from . import tree as T

RUN_ID_PREFIX = "r_"
TERMINAL = ("done", "failed", "cancelled")


class Conflict(Exception):
    def __init__(self, active: str):
        super().__init__(f"已有推演正在运行（{active}）：同一时间只能运行一个，所有调用共用同一份订阅额度")
        self.active = active


class NeedsConfirm(Exception):
    def __init__(self, estimate: dict):
        super().__init__("这次推演会调用本机 Claude Code（计入你的订阅额度）：请先查看预计用量，再带上 confirm: true 重新提交")
        self.estimate = estimate


def real_backend() -> bool:
    return backend.backend_name() == "claude_code"


def new_run_id() -> str:
    return f"{RUN_ID_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


class JobManager:
    def __init__(self, base: Path | None = None, *, auto_resume: bool = True):
        self.base = Path(base or T.AGENTS_DIR)
        self.runs_dir = self.base / "runs"
        self.snaps_dir = self.base / "snapshots"
        self.cache_dir = self.base / "cache"
        self.lock = threading.RLock()
        self.active: dict | None = None
        self.auto_resume = auto_resume
        self._timers: dict = {}
        self._explorers: dict = {}
        self.recover()

    # ---------------------------------------------------------------- run.json
    def run_dir(self, rid: str) -> Path:
        if not rid or "/" in rid or ".." in rid or not rid.replace("_", "").replace("-", "").isalnum():
            raise KeyError(f"推演 id「{rid}」不合法")
        return self.runs_dir / rid

    def read(self, rid: str) -> dict:
        p = self.run_dir(rid) / "run.json"
        if not p.exists():
            raise KeyError(f"没有找到推演「{rid}」（可能已被删除）")
        return T.read_json(p)

    def write(self, rid: str, **upd) -> dict:
        with self.lock:
            p = self.run_dir(rid) / "run.json"
            cur = T.read_json(p) if p.exists() else {}
            cur.update(upd, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
            T.write_json(p, cur)
            return cur

    def recover(self) -> None:
        """Runs left running or queued, and snapshot builds left building, by a previous server process become paused
        (server_restart)."""
        if self.snaps_dir.exists():
            for d in self.snaps_dir.iterdir():
                m = S.read_manifest(d) if d.is_dir() else None
                if m and m.get("status") in ("building", "queued", "cancelling"):
                    m.update(status="paused", pause={"reason": "server_restart", "message": "服务器重启：快照构建已暂停，可以继续",
                                                     "reset_at": None, "level": m.get("levels_done")},
                             updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    S.write_manifest(d, m)
        if not self.runs_dir.exists():
            return
        for d in self.runs_dir.iterdir():
            p = d / "run.json"
            if not p.exists():
                continue
            try:
                r = T.read_json(p)
            except (OSError, ValueError):
                continue
            if r.get("status") in ("running", "queued", "cancelling"):
                r.update(status="paused", pause={"reason": "server_restart", "message": "服务器重启：推演已暂停，可以继续",
                                                 "reset_at": None})
                T.write_json(p, r)

    # ---------------------------------------------------------------- start, resume, extend
    def start(self, settings, *, name: str | None = None, confirm: bool = False, parent_run: str | None = None) -> dict:
        st = T.TreeSettings.coerce(settings)
        probs = st.problems()
        if probs:
            raise ValueError("推演设置有错误：" + "；".join(probs))
        if st.tree_kind == "agent" and real_backend() and not confirm:
            raise NeedsConfirm(T.estimate(st))
        st = dataclasses.replace(st, confirmed=bool(confirm) or not real_backend(),
                                 hard_cap=st.hard_cap if st.hard_cap is not None else T.hard_cap_for(st))
        adapter = MA.load_adapter()                    # ValueError (400): an explicit model that cannot load
        if st.tree_kind == "agent":
            S.refuse_fallback_stub(adapter, "推演")     # real calls are never priced by a fallback stub
        with self.lock:
            if self.active is not None:
                raise Conflict(self.active["id"])
            rid = new_run_id()
            d = self.run_dir(rid)
            d.mkdir(parents=True, exist_ok=True)
            info = MA.model_info(adapter)
            meta = {"id": rid, "name": name or st.name or "", "kind": "run", "status": "queued",
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "settings": st.to_dict(),
                    "settings_hash": st.hash(), "versions": T.versions(adapter), "stub": bool(adapter.stub),
                    "pricing_model": info, "export": T.export_status(info, "full" if st.paths else "none"),
                    "backend": backend.backend_name(),
                    "estimate": T.estimate(st), "parent_run": parent_run, "pause": None, "error": None}
            T.write_json(d / "run.json", meta)
            self._launch(rid, st, resume=False)
            return {"run_id": rid, "status": "queued"}

    def resume(self, rid: str, *, confirm: bool = False, depth: int | None = None) -> dict:
        r = self.read(rid)
        if r.get("status") == "done" and depth is None:
            return {"run_id": rid, "status": "done", "calls_remaining": 0}
        if r.get("status") in ("running", "queued", "cancelling"):
            raise Conflict(rid)
        st = T.TreeSettings.saved(r["settings"])
        done = int(r.get("levels_done") or 0)
        if depth is not None:
            if not st.depth < int(depth) <= 80:
                raise ValueError(f"加深后的深度必须大于 {st.depth} 且不超过 80 季")
            st = dataclasses.replace(st, depth=int(depth))
        est = T.estimate(st, cached_levels=done)
        adapter = self._adapter_for(r)                 # ValueError (400): the run's fit cannot be loaded or changed
        drift = T.version_drift(r.get("versions"), adapter)
        if drift:                                      # ValueError (400): another engine, catalogue, channel table …
            raise ValueError(T.drift_refusal(drift, "推演"))
        if st.tree_kind == "agent":
            S.refuse_fallback_stub(adapter, "推演")
        if st.tree_kind == "agent" and real_backend() and not confirm:
            raise NeedsConfirm(est)
        spent = T.ledger_size(self.run_dir(rid))
        cap = st.hard_cap
        if cap is None or depth is not None or (r.get("pause") or {}).get("reason") == "budget":
            cap = T.hard_cap_for(st, spent=spent, cached_levels=done)       # the confirmed remainder
        st = dataclasses.replace(st, confirmed=bool(confirm) or not real_backend(), hard_cap=cap)
        with self.lock:
            if self.active is not None:
                raise Conflict(self.active["id"])
            self.write(rid, status="queued", settings=st.to_dict(), settings_hash=st.hash(), pause=None, error=None,
                       estimate=est)
            self._launch(rid, st, resume=True)
        return {"run_id": rid, "status": "running"}

    def extend(self, rid: str, quarters: int, *, confirm: bool = False) -> dict:
        r = self.read(rid)
        st = T.TreeSettings.saved(r["settings"])
        return self.resume(rid, confirm=confirm, depth=int(st.depth) + int(quarters))

    def _launch(self, rid: str, st: T.TreeSettings, *, resume: bool) -> None:
        cancel, kill = threading.Event(), threading.Event()
        self.active = {"id": rid, "cancel": cancel, "kill": kill, "started": time.time()}
        th = threading.Thread(target=self._worker, args=(rid, st, cancel, resume, kill), daemon=True,
                              name=f"agents-{rid}")
        self.active["thread"] = th
        th.start()

    def _worker(self, rid: str, st: T.TreeSettings, cancel: threading.Event, resume: bool,
                kill: threading.Event | None = None) -> None:
        d = self.run_dir(rid)
        self.write(rid, status="running")
        journal = T.Journal(d / "journal.jsonl")
        try:
            r0 = self.read(rid)
            items = r0.get("institutions")
            adapter = self._adapter_for(r0)
            run = T.TreeRun(st, T.Store(d), journal=journal, cancel=cancel, kill=kill, cache_dir=self.cache_dir,
                            institutions=items, adapter=adapter)
            if items is None:
                self.write(rid, institutions=run.engine.items,
                           accounts_root_hash=(run.engine.root_acc or {}).get("hash"))
            self.write(rid, transmission=S.engine_transmission(run.engine))

            def on_level(level, stats):
                self.write(rid, levels=run.level_stats, levels_done=level, calls_paid=run.calls_paid(),
                           stats=run.llm.stats if run.llm is not None else None)
                self.forget("run", rid)
            res = run.run(resume=True, on_level=on_level)
            self.write(rid, status="running", tree_hash=res["tree_hash"], leaves=res["leaves"], levels=res["levels"],
                       stats=res["llm_stats"], levels_done=int(st.depth), calls_paid=res.get("calls_paid"))
            if st.prior_trees and st.tree_kind == "agent":
                trees = {}
                for kind in ("base_rate", "model"):
                    if cancel.is_set():
                        break
                    sub = d / "prior_trees" / kind
                    st2 = dataclasses.replace(st, tree_kind=kind, prior_trees=False)
                    r2 = T.TreeRun(st2, T.Store(sub), journal=T.Journal(sub / "journal.jsonl"), cancel=cancel,
                                   cache_dir=self.cache_dir, institutions=run.engine.items,
                                   adapter=adapter).run(resume=True)
                    trees[kind] = {"tree_hash": r2["tree_hash"], "leaves": r2["leaves"]}
                journal.emit("prior_trees_done", trees=list(trees))
                self.write(rid, prior_trees=trees)
            self.write(rid, status="done", finished=time.strftime("%Y-%m-%dT%H:%M:%S"), pause=None)
        except T.Paused as err:
            status = "cancelled" if err.reason == "cancelled" else "paused"
            self.write(rid, status=status, pause={"reason": err.reason, "message": err.message,
                                                  "reset_at": err.reset_at, "level": err.level},
                       calls_paid=T.ledger_size(d))
            journal.emit("run_paused", reason=err.reason, message=err.message, reset_at=err.reset_at, level=err.level)
            if err.reason == "usage_limit" and self.auto_resume and err.reset_at:
                self._schedule_resume(rid, err.reset_at)
        except Exception as err:  # noqa: BLE001 - recorded in Chinese for the UI
            self.write(rid, status="failed", error=f"{type(err).__name__}: {err}")
            journal.emit("run_failed", error=f"{type(err).__name__}: {err}")
        finally:
            self.forget("run", rid)
            with self.lock:
                if self.active is not None and self.active.get("id") == rid:
                    self.active = None

    @staticmethod
    def _adapter_for(r: dict):
        """The adapter a run was started with (run.json versions.model_adapter), whatever MSL_AGENT_MODEL or the
        server's default says now; the exact fit, else ValueError in Chinese (a refit would mix two fits)."""
        return S.folder_adapter(r, what="推演")

    def _schedule_resume(self, rid: str, reset_at: float | None) -> None:
        """Resume once after the reset the CLI announced (the user confirmed the run); never without a reset time."""
        if not reset_at:
            return
        delay = max(60.0, reset_at - time.time() + 60.0)

        def go():
            self._timers.pop(rid, None)
            try:
                self.resume(rid, confirm=True)
            except (Conflict, KeyError, ValueError):
                pass
        t = threading.Timer(delay, go)
        t.daemon = True
        old = self._timers.pop(rid, None)
        if old is not None:
            old.cancel()
        self._timers[rid] = t
        t.start()
        self.write(rid, auto_resume_at=time.time() + delay)

    # ---------------------------------------------------------------- control and queries
    def cancel(self, rid: str, *, force: bool = False) -> dict:
        """Graceful (no new call starts; calls in flight finish and are cached) or forced (they are killed)."""
        t = self._timers.pop(rid, None)
        if t is not None:
            t.cancel()
        with self.lock:
            if self.active is not None and self.active["id"] == rid:
                self.active["cancel"].set()
                if force and self.active.get("kill") is not None:
                    self.active["kill"].set()
                self.write(rid, status="cancelling", cancel_force=bool(force))
                return {"run_id": rid, "status": "cancelling"}
        r = self.read(rid)
        if r.get("status") in ("paused", "queued"):
            self.write(rid, status="cancelled", auto_resume_at=None)
            return {"run_id": rid, "status": "cancelled"}
        return {"run_id": rid, "status": r.get("status")}

    def delete(self, rid: str) -> dict:
        with self.lock:
            if self.active is not None and self.active["id"] == rid:
                raise Conflict(rid)
        d = self.run_dir(rid)
        if not d.exists():
            raise KeyError(f"没有找到推演「{rid}」（可能已被删除）")
        shutil.rmtree(d)
        with self.lock:
            self._explorers.pop(("run", rid), None)
        return {"deleted": rid}

    def rename(self, rid: str, name: str) -> dict:
        self.read(rid)
        return self.write(rid, name=str(name or "")[:80])

    def list(self, status: str | None = None, q: str | None = None) -> list[dict]:
        out = []
        if not self.runs_dir.exists():
            return out
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            p = d / "run.json"
            if not p.exists():
                continue
            try:
                r = T.read_json(p)
            except (OSError, ValueError):
                continue
            if status == "active" and r.get("status") not in ("running", "queued", "cancelling"):
                continue
            if status and status != "active" and r.get("status") != status:
                continue
            if q and q not in (r.get("name") or "") and q not in r.get("id", ""):
                continue
            s = r.get("settings") or {}
            out.append({"id": r.get("id"), "name": r.get("name"), "status": r.get("status"), "created": r.get("created"),
                        "updated": r.get("updated"), "depth": s.get("depth"), "B": s.get("B"),
                        "tree_kind": s.get("tree_kind"),
                        "calls_new": r.get("calls_paid") if r.get("calls_paid") is not None else (r.get("stats") or {}).get("calls_new"),
                        "calls_cached": (r.get("stats") or {}).get("cached"), "levels_done": r.get("levels_done"),
                        "stub": r.get("stub"), "parent_run": r.get("parent_run"), "settings_hash": r.get("settings_hash"),
                        "pause": r.get("pause"), "keep_paths": s.get("keep_paths"),
                        "pricing_model": {k: (r.get("pricing_model") or {}).get(k) for k in ("kind", "stub", "label",
                                                                                              "fit", "trial")}
                        if r.get("pricing_model") else None})
        return out

    def get(self, rid: str) -> dict:
        r = self.read(rid)
        ex = self.explorer("run", rid)
        summ = ex.summary()
        return {**r, "levels_view": summ["levels"], "root": summ["root"], "overlay": summ["overlay"],
                "active": self.active is not None and self.active.get("id") == rid}

    def progress(self, rid: str, since: int = 0) -> dict:
        r = self.read(rid)
        j = T.Journal(self.run_dir(rid) / "journal.jsonl")
        return {"status": r.get("status"), "events": j.since(int(since)), "pause": r.get("pause"), "seq": j.seq}

    def cached_levels(self, settings, adapter=None) -> int:
        """Levels of `settings`' tree already in the cache: a saved run or snapshot with the same settings at any depth
        is a prefix of this tree (selection at a level depends only on earlier levels), so its finished levels are
        exact cache hits (§11.4) — but only when it was priced by the same model (every cache key carries the
        versions: engine, catalogue, accounts, priors, prompts and the model's identity) and called through the
        same LLM backend. A folder that recorded no backend does not count (the estimate errs on the safe side)."""
        st = T.TreeSettings.coerce(settings)
        want = dataclasses.replace(st, depth=1).hash()
        adapter = adapter if adapter is not None else MA.load_adapter()
        vers = T.versions(adapter)
        be = backend.backend_name()
        best = 0
        folders = []
        if self.runs_dir.exists():
            folders += [(d / "run.json") for d in self.runs_dir.iterdir()]
        if self.snaps_dir.exists():
            folders += [(d / "manifest.json") for d in self.snaps_dir.iterdir()]
        for p in folders:
            try:
                r = T.read_json(p) if p.exists() else None
            except (OSError, ValueError):
                continue
            if not r or not r.get("settings"):
                continue
            try:
                other = T.TreeSettings.saved(r["settings"])
            except (TypeError, ValueError):
                continue
            if dataclasses.replace(other, depth=1).hash() != want:
                continue
            if r.get("versions") != vers or r.get("backend") != be:
                continue
            best = max(best, min(int(r.get("levels_done") or 0), int(st.depth)))
        return best

    def explorer(self, kind: str, xid: str) -> S.Explorer:
        key = (kind, xid)
        with self.lock:
            ex = self._explorers.get(key)
            if ex is None:
                root = self.run_dir(xid) if kind == "run" else S.snapshot_root(xid, self.snaps_dir)
                if not root.exists():
                    raise KeyError(f"没有找到{'推演' if kind == 'run' else '快照'}「{xid}」")
                ex = S.Explorer(root, cache_dir=self.cache_dir, kind=kind)
                self._explorers[key] = ex
            return ex

    def forget(self, kind: str, xid: str) -> None:
        """The folder changed: cached levels and indexes are dropped (the explorer's expansion jobs are kept)."""
        with self.lock:
            ex = self._explorers.get((kind, xid))
        if ex is not None:
            ex.invalidate()

    # ---------------------------------------------------------------- snapshots started from the server
    def snapshot_settings(self, sid: str, settings, *, resume: bool) -> T.TreeSettings:
        """A new build's settings from the request; a resumed build's from its own manifest (the request's beam and
        depth are ignored, so a build with other options resumes exactly)."""
        root = S.snapshot_root(sid, self.snaps_dir)             # ValueError on a bad id
        man = S.read_manifest(root)
        if resume:
            if man is None:
                raise KeyError(f"没有找到快照「{sid}」：不能继续构建")
            if man.get("status") == "done":
                raise ValueError(f"快照「{sid}」已经完成并冻结；换一个 id 重新构建")
            st = T.TreeSettings.saved(man.get("settings") or {})
        else:
            if man is not None:
                raise ValueError(f"快照「{sid}」已存在（{man.get('status')}）：继续构建请用“继续构建”，或换一个 id")
            raw = dict(settings or {})
            try:
                beam, depth = int(raw.get("B", S.DEFAULT_BEAM)), int(raw.get("depth", S.DEFAULT_QUARTERS))
            except (TypeError, ValueError) as err:
                raise ValueError("B 和 depth 必须是整数") from err
            st = S.default_settings(beam, depth, **{k: v for k, v in raw.items()
                                                   if k in T.TreeSettings.__dataclass_fields__ and k not in ("B", "depth")})
        probs = st.problems()
        if probs:
            raise ValueError("快照设置有错误：" + "；".join(probs))
        return st

    def start_snapshot(self, sid: str, settings, *, confirm: bool = False, resume: bool = False) -> dict:
        """Validate now (400 / 404 / 409 before anything runs), then build in a thread; the worker's outcome (done,
        paused, failed with the error) is in the manifest."""
        st = self.snapshot_settings(sid, settings, resume=resume)
        root = S.snapshot_root(sid, self.snaps_dir)
        man = S.read_manifest(root)
        done = int((man or {}).get("levels_done") or 0)
        # the pricing model before anything runs (400 in Chinese): a resumed build is priced by its own fit, and
        # real calls are never priced by a fallback stub
        adapter = S.folder_adapter(man) if (resume and man and not man.get("provisional")) else MA.load_adapter()
        S.refuse_fallback_stub(adapter)
        if real_backend() and not confirm:
            raise NeedsConfirm(S.full_estimate(dataclasses.replace(st)) if not resume else
                               T.estimate(st, cached_levels=done))
        with self.lock:
            if self.active is not None:
                raise Conflict(self.active["id"])
            cancel, kill = threading.Event(), threading.Event()
            self.active = {"id": f"snapshot:{sid}", "cancel": cancel, "kill": kill, "started": time.time()}
        if not resume:                     # the manifest exists before the slow estimate (the UI polls it at once)
            root.mkdir(parents=True, exist_ok=True)
            S.write_manifest(root, {"id": sid, "version": S.SNAPSHOT_VERSION, "status": "building", "levels_done": 0,
                                    "settings": st.to_dict(), "settings_hash": st.hash(),
                                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "spent_calls": 0, "pause": None,
                                    "provisional": True})
        else:
            # the previous pause stays readable for build() (resume_pause): a "budget" pause the user confirmed
            # raises the hard cap there (it was cleared here before, so a budget pause could never be resumed)
            S.write_manifest(root, {**(man or {}), "status": "building", "pause": None,
                                    "resume_pause": (man or {}).get("pause") or (man or {}).get("resume_pause"),
                                    "error": None, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})

        def work():
            try:
                S.build(sid, dataclasses.replace(st, confirmed=True), resume=resume, base=self.snaps_dir,
                        cache_dir=self.cache_dir, cancel=cancel, kill=kill, budget_confirmed=True,
                        adapter=adapter)
            except S.BuildPaused as err:
                if err.reason == "usage_limit" and self.auto_resume and err.reset_at:
                    delay = max(60.0, err.reset_at - time.time() + 60.0)
                    key = f"snapshot:{sid}"

                    def go():
                        self._timers.pop(key, None)
                        try:
                            self.start_snapshot(sid, None, confirm=True, resume=True)
                        except (Conflict, KeyError, ValueError):
                            pass
                    t = threading.Timer(delay, go)
                    t.daemon = True
                    self._timers[key] = t
                    t.start()
                    m = S.read_manifest(root) or {}
                    S.write_manifest(root, {**m, "auto_resume_at": time.time() + delay})
            except Exception as err:   # noqa: BLE001 - recorded in the manifest for the UI
                m = S.read_manifest(root) or {"id": sid}
                m.update(status="failed", error=f"{type(err).__name__}: {err}",
                         updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
                S.write_manifest(root, m)
            finally:
                with self.lock:
                    if self.active is not None and self.active.get("id") == f"snapshot:{sid}":
                        self.active = None
                self.forget("snapshot", sid)
        threading.Thread(target=work, daemon=True, name=f"snapshot-{sid}").start()
        return {"snapshot_id": sid, "status": "building"}

    def cancel_snapshot(self, sid: str, *, force: bool = False) -> dict:
        """Stop a build (graceful unless force) or a pending auto-resume; the checkpoint keeps every finished level."""
        root = S.snapshot_root(sid, self.snaps_dir)
        man = S.read_manifest(root)
        if man is None:
            raise KeyError(f"没有找到快照「{sid}」")
        t = self._timers.pop(f"snapshot:{sid}", None)
        if t is not None:
            t.cancel()
        with self.lock:
            if self.active is not None and self.active.get("id") == f"snapshot:{sid}":
                self.active["cancel"].set()
                if force:
                    self.active["kill"].set()
                S.write_manifest(root, {**man, "status": "cancelling", "cancel_force": bool(force)})
                return {"snapshot_id": sid, "status": "cancelling"}
        if man.get("status") in ("paused", "building", "failed"):
            S.write_manifest(root, {**man, "status": "paused", "auto_resume_at": None,
                                    "pause": {**(man.get("pause") or {}), "reason": (man.get("pause") or {}).get("reason")
                                              or "cancelled", "message": "已取消：可以继续构建"}})
            return {"snapshot_id": sid, "status": "paused"}
        return {"snapshot_id": sid, "status": man.get("status")}


_MANAGER: JobManager | None = None
_MANAGER_LOCK = threading.Lock()


def manager() -> JobManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER


def reset_manager(base: Path | None = None, *, auto_resume: bool = True) -> JobManager:
    """A fresh manager (tests point it at a temporary folder)."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = JobManager(base, auto_resume=auto_resume)
        return _MANAGER
