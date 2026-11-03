"""One iteration = one click.

(a) every enabled actor is called in parallel (at most `max_parallel` at a time) with the world, its role, cares,
    action space, private info, the current state (its visible fields) and the recent history;
(b) the synthesizer is called with the current state, the schema, every actor's output and its instructions;
(c) its answer is validated against the schema (model.validate_next_state: types, enums, ranges; missing values
    carried forward and flagged; nothing invented);
(d) the next state and the iteration record (every call's prompt, raw response, usage and cost) are saved.

Failures: an actor call that fails is retried once, then the actor is marked "no response" and the synthesizer is
told. Bad JSON gets one repair call, then the failure path. A synthesizer failure leaves the state unchanged: the
actors' outputs are kept in pending.json and retry() calls the synthesizer again with them.

RunManager runs iterations in a background thread per world and keeps the live progress the UI polls.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import model as M
from . import prompts as P
from .llm import CallRequest, LLMError, est_tokens, extract_json, make_backend
from .openrouter import ModelCatalog
from .settings import effective_backend, redact
from .store import Conflict, NotFound, Store, now_iso

ITERATION_FORMAT = "worldsim-iteration/1"
PENDING_FORMAT = "worldsim-pending/1"
REASONING_ALLOWANCE = {"none": 0, "minimal": 300, "low": 1000, "medium": 2500, "high": 6000, "xhigh": 10000,
                       "max": 16000}
ACTOR_OUT_TOKENS = 600
SYNTH_OUT_BASE, SYNTH_OUT_PER_FIELD = 1500, 70


class Busy(RuntimeError):
    pass


class Progress:
    """The live state of one run, read by the UI while the run's thread updates it."""

    def __init__(self, world_id: str, mode: str, actors: list[dict], step_from: int, period: str):
        self._lock = threading.Lock()
        self.data: dict[str, Any] = {
            "run_id": uuid.uuid4().hex[:12], "world_id": world_id, "mode": mode, "status": "running",
            "phase": "actors" if mode == "run" else "synthesizer", "started": now_iso(), "finished": None,
            "step_from": step_from, "step": None, "period": period, "error": None, "cost": 0.0,
            "actors": [{"id": a["id"], "name": a["name"], "status": "queued", "attempts": 0, "error": None,
                        "action_label": None} for a in actors],
            "synthesizer": {"status": "waiting", "error": None}}

    def actor(self, aid: str, **kw) -> None:
        with self._lock:
            for a in self.data["actors"]:
                if a["id"] == aid:
                    a.update(kw)

    def synth(self, **kw) -> None:
        with self._lock:
            self.data["synthesizer"].update(kw)

    def set(self, **kw) -> None:
        with self._lock:
            self.data.update(kw)

    def add_cost(self, cost: float | None) -> None:
        if cost:
            with self._lock:
                self.data["cost"] = round(self.data["cost"] + cost, 8)

    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self.data)


def _zh(world: dict) -> bool:
    lang = (world.get("language") or "").lower()
    return "中" in lang or "chinese" in lang or lang.startswith("zh")


def validate_actor_output(world: dict, actor: dict, obj: Any) -> tuple[dict | None, str | None, list[str]]:
    """(clean output, error, notes). An error means the answer is unusable (the caller repairs or retries)."""
    if not isinstance(obj, dict):
        return None, "the answer is not a JSON object", []
    notes: list[str] = []
    aid = obj.get("action_id")
    if not isinstance(aid, str) or not aid.strip():
        return None, "action_id is missing", []
    aid = aid.strip()
    actions = {a["id"]: a for a in actor.get("action_space") or []}
    allow_free = actor.get("allow_free_action") or not actions
    detail = M._text(obj.get("action_detail"), 4000)
    if aid in actions:
        label = actions[aid]["label"]
    elif aid == M.FREE_ACTION and allow_free:
        label = "自由行动" if _zh(world) else "Free action"
    elif allow_free:
        notes.append(f"unknown action id {aid!r} was treated as a free action")
        detail = f"[{aid}] {detail}".strip()
        aid, label = M.FREE_ACTION, ("自由行动" if _zh(world) else "Free action")
    else:
        return None, f"action_id {aid!r} is not in the action space ({', '.join(actions)})", []
    cons = obj.get("expected_consequences")
    if isinstance(cons, str):
        cons = [cons]
    cons = [M._text(c, 600) for c in (cons if isinstance(cons, list) else []) if M._text(c, 600)][:8]
    return ({"action_id": aid, "action_label": label, "action_detail": detail,
             "rationale": M._text(obj.get("rationale"), 4000),
             "public_statement": M._text(obj.get("public_statement"), 2000),
             "expected_consequences": cons}, None, notes)


def validate_synth_output(obj: Any) -> tuple[dict | None, str | None]:
    if not isinstance(obj, dict):
        return None, "the answer is not a JSON object"
    if not isinstance(obj.get("values"), dict):
        return None, "the answer has no 'values' object with a value for every field"
    return obj, None


def _sum_usage(calls: list[dict]) -> tuple[dict, float, bool]:
    tot = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "calls": len(calls)}
    cost, complete = 0.0, True
    for c in calls:
        u = c.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            if isinstance(u.get(k), (int, float)):
                tot[k] += int(u[k])
        if c.get("error") and not u:
            continue                                  # a call that failed before an answer: no usage, no cost
        if isinstance(c.get("cost"), (int, float)):
            cost += float(c["cost"])
        else:
            complete = False
    return tot, round(cost, 8), complete


class Engine:
    def __init__(self, store: Store, settings_loader: Callable[[], dict], catalog: ModelCatalog | None = None,
                 backend_factory: Callable[[dict], Any] | None = None):
        self.store = store
        self.settings_loader = settings_loader
        self.catalog = catalog or ModelCatalog(store.root)
        self.backend_factory = backend_factory or (lambda s: make_backend(s, catalog=self.catalog))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def model_for(world: dict, settings: dict, override: str = "") -> str:
        return override or (world.get("model") or {}).get("model") or settings.get("default_model")

    @staticmethod
    def params(world: dict, settings: dict) -> dict:
        wm = world.get("model") or {}
        temp = wm.get("temperature")
        return {"temperature": settings.get("temperature") if temp is None else temp,
                "reasoning_effort": wm.get("reasoning_effort") or settings.get("reasoning_effort") or "",
                "timeout_s": settings.get("timeout_s") or 240, "max_tokens": settings.get("max_tokens")}

    def history(self, wid: str, world: dict, upto: int) -> list[dict]:
        """The last `history_window` steps up to `upto`: period, summary, narrative and the actions that led to it."""
        window = int(world.get("history_window", 3) or 0)
        if window <= 0:
            return []
        out = []
        for n in range(max(0, upto - window + 1), upto + 1):
            try:
                st = self.store.load_step(wid, n)
            except NotFound:
                continue
            entry = {"step": n, "period": st.get("period"), "summary": st.get("summary") or "",
                     "narrative": (st.get("text") or {}).get("narrative") or "", "actions": []}
            if n > 0:
                try:
                    it = self.store.load_iteration(wid, n)
                    for r in it.get("actors") or []:
                        o = r.get("output") or {}
                        entry["actions"].append({"actor_id": r.get("actor_id"), "name": r.get("name"),
                                                 "status": r.get("status"), "action_label": o.get("action_label"),
                                                 "action_detail": o.get("action_detail"),
                                                 "public_statement": o.get("public_statement")})
                except NotFound:
                    pass
            out.append(entry)
        return out

    def _call(self, backend, req: CallRequest, calls: list[dict], lock: threading.Lock, who: str, attempt: int,
              progress: Progress | None, secrets: list[str]) -> tuple[Any, str | None]:
        """One request (+ one repair on bad JSON) -> (parsed object | None, error | None). Every call is logged."""
        validator = req.context.get("_validator")
        text, err = "", None
        for repair in (False, True):
            r = req if not repair else CallRequest(**{**req.__dict__, "messages": P.repair_messages(req.messages, text, err),
                                                      "repair": True})
            entry = {"who": who, "purpose": req.purpose, "attempt": attempt, "repair": repair, "model": req.model,
                     "started": now_iso(), "messages": r.messages, "response_format": None, "response_text": None,
                     "usage": None, "cost": None, "cost_source": None, "error": None, "duration_s": None}
            t0 = time.monotonic()
            try:
                comp = backend.complete(r)
            except LLMError as e:
                entry.update(error=redact(str(e), secrets), duration_s=round(time.monotonic() - t0, 3))
                with lock:
                    calls.append(entry)
                if repair:
                    return None, f"the repair call failed: {entry['error']}"
                raise
            entry.update(response_format=comp.response_format, response_text=comp.text, usage=comp.usage,
                         cost=comp.cost, cost_source=comp.meta.get("cost_source"), served_model=comp.model,
                         duration_s=round(time.monotonic() - t0, 3))
            if progress:
                progress.add_cost(comp.cost)
            text = comp.text
            obj, err = extract_json(comp.text)
            if obj is not None and validator:
                obj, err = validator(obj)
            if err:
                entry["error"] = f"unusable answer: {err}"
            with lock:
                calls.append(entry)
            if not err:
                return obj, None
            if not repair and progress and req.purpose == "synth":
                progress.synth(status="repairing")
        return None, f"bad answer after one repair: {err}"

    # ------------------------------------------------------------------ actors
    def _run_actor(self, backend, world: dict, settings: dict, actor: dict, state: dict, history: list[dict],
                   next_period: str, calls: list[dict], lock: threading.Lock, progress: Progress | None) -> dict:
        model = self.model_for(world, settings, actor.get("model") or "")
        params = self.params(world, settings)
        notes_box: list[str] = []

        def validator(obj):
            clean, err, notes = validate_actor_output(world, actor, obj)
            notes_box[:] = notes
            return clean, err

        ctx = {"world": {"id": world["id"], "name": world["name"], "language": world["language"]},
               "actor": actor, "state": state, "_validator": validator}
        req = CallRequest(purpose="actor", model=model, schema_name=P.ACTOR_SCHEMA_NAME,
                          messages=P.actor_messages(world, actor, state, history, next_period),
                          json_schema=P.actor_schema(actor), context=ctx, **params)
        secrets = [settings.get("api_key") or ""]
        error = None
        attempts = 0
        for attempt in (1, 2):
            attempts = attempt
            if progress:
                progress.actor(actor["id"], status="running" if attempt == 1 else "retrying", attempts=attempt)
            try:
                out, error = self._call(backend, req, calls, lock, actor["id"], attempt, progress, secrets)
            except LLMError as e:
                out, error = None, redact(str(e), secrets)
                if not e.retryable:
                    break
            if out is not None:
                if progress:
                    progress.actor(actor["id"], status="done", action_label=out["action_label"], error=None)
                return {"actor_id": actor["id"], "name": actor["name"], "status": "ok", "model": model,
                        "attempts": attempt, "output": out, "notes": list(notes_box), "error": None}
        if progress:
            progress.actor(actor["id"], status="failed", error=error)
        return {"actor_id": actor["id"], "name": actor["name"], "status": "no_response", "model": model,
                "attempts": attempts, "output": None, "notes": [], "error": error}

    def run_actors(self, backend, world: dict, settings: dict, state: dict, history: list[dict], next_period: str,
                   calls: list[dict], progress: Progress | None) -> list[dict]:
        actors = [a for a in world["actors"] if a.get("enabled")]
        lock = threading.Lock()
        workers = max(1, min(int(settings.get("max_parallel") or 6), len(actors) or 1))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="worldsim-actor") as ex:
            futures = [ex.submit(self._run_actor, backend, world, settings, a, state, history, next_period, calls,
                                 lock, progress) for a in actors]
            results = []
            for a, f in zip(actors, futures):
                try:
                    results.append(f.result())
                except Exception as e:  # noqa: BLE001 - one actor's bug must not stop the iteration
                    results.append({"actor_id": a["id"], "name": a["name"], "status": "no_response", "attempts": 1,
                                    "output": None, "notes": [], "model": None,
                                    "error": redact(f"internal error: {type(e).__name__}: {e}", [settings.get("api_key") or ""])})
        return results

    # ------------------------------------------------------------------ synthesizer
    def run_synth(self, backend, world: dict, settings: dict, state: dict, results: list[dict], history: list[dict],
                  next_period: str, calls: list[dict], progress: Progress | None) -> tuple[dict | None, str | None, str]:
        model = self.model_for(world, settings, (world.get("synthesizer") or {}).get("model") or "")
        params = self.params(world, settings)
        ctx = {"world": {"id": world["id"], "name": world["name"], "language": world["language"]},
               "schema": world["schema"], "state": state, "actor_results": results, "next_period": next_period,
               "_validator": validate_synth_output}
        req = CallRequest(purpose="synth", model=model, schema_name=P.SYNTH_SCHEMA_NAME,
                          messages=P.synth_messages(world, state, results, history, next_period),
                          json_schema=P.synth_schema(world["schema"]), context=ctx, **params)
        if progress:
            progress.set(phase="synthesizer")
            progress.synth(status="running", error=None)
        secrets = [settings.get("api_key") or ""]
        try:
            out, err = self._call(backend, req, calls, threading.Lock(), "synthesizer", 1, progress, secrets)
        except LLMError as e:
            out, err = None, redact(str(e), secrets)
        if progress:
            progress.synth(status="done" if out is not None else "failed", error=err)
        return out, err, model

    # ------------------------------------------------------------------ an iteration
    def _prepare(self, wid: str) -> tuple[dict, dict, dict, str, list[dict]]:
        world = self.store.load_world(wid)
        settings = self.settings_loader()
        state = self.store.latest_step(wid)
        unit, count = world["period_step"]["unit"], world["period_step"]["count"]
        next_period = M.advance_period(state.get("period"), unit, count)
        history = self.history(wid, world, int(state["step"]))
        return world, settings, state, next_period, history

    def run(self, wid: str, progress: Progress | None = None) -> dict:
        world, settings, state, next_period, history = self._prepare(wid)
        backend = self.backend_factory(settings)
        started = now_iso()
        calls: list[dict] = []
        self.store.clear_pending(wid)
        results = self.run_actors(backend, world, settings, state, history, next_period, calls, progress)
        return self._finish(wid, world, settings, backend, state, results, history, next_period, calls, started,
                            progress)

    def retry(self, wid: str, progress: Progress | None = None) -> dict:
        pending = self.store.load_pending(wid)
        if not pending:
            raise NotFound("there is no failed synthesizer attempt to retry")
        world, settings, state, next_period, history = self._prepare(wid)
        if pending.get("from_step") != state["step"]:
            self.store.clear_pending(wid)
            raise Conflict("the failed attempt no longer matches the current step: run a new iteration")
        backend = self.backend_factory(settings)
        calls = list(pending.get("calls") or [])
        return self._finish(wid, world, settings, backend, state, pending.get("actors") or [], history, next_period,
                            calls, pending.get("started") or now_iso(), progress)

    def _finish(self, wid, world, settings, backend, state, results, history, next_period, calls, started, progress):
        out, err, synth_model = self.run_synth(backend, world, settings, state, results, history, next_period, calls,
                                               progress)
        usage, cost, complete = _sum_usage(calls)
        if out is None:
            pending = {"format": PENDING_FORMAT, "world_id": wid, "from_step": state["step"], "period": next_period,
                       "started": started, "failed_at": now_iso(), "error": err, "actors": results, "calls": calls,
                       "synthesizer_model": synth_model, "usage": usage, "cost": cost, "cost_complete": complete,
                       "backend": getattr(backend, "name", "?")}
            self.store.save_pending(wid, pending)
            return {"status": "failed", "error": err, "step": state["step"], "cost": cost, "usage": usage}
        n = int(state["step"]) + 1
        nxt, flags, changes = M.validate_next_state(world["schema"], state, out, period=next_period, step=n)
        nxt["created"] = now_iso()
        record = {"format": ITERATION_FORMAT, "world_id": wid, "step": n, "from_step": state["step"],
                  "period_from": state.get("period"), "period": next_period, "started": started, "finished": now_iso(),
                  "backend": getattr(backend, "name", "?"), "actors": results,
                  "synthesizer": {"status": "ok", "model": synth_model, "output": out},
                  "flags": flags, "changes": changes,
                  "text_changes": M.text_changes(state.get("text") or {}, nxt["text"]),
                  "usage": usage, "cost": cost, "cost_complete": complete, "calls": calls}
        nxt["usage"], nxt["cost"], nxt["cost_complete"] = usage, cost, complete
        self.store.append(wid, nxt, record)
        return {"status": "done", "step": n, "period": next_period, "cost": cost, "usage": usage,
                "flags": len(flags), "changes": len(changes)}

    # ------------------------------------------------------------------ estimate
    def _allowance(self, effort: str, info: dict | None) -> int:
        if effort:
            return REASONING_ALLOWANCE.get(effort, 2500)
        if info and info.get("reasoning") and (info.get("reasoning_mandatory") or info.get("default_effort")):
            return REASONING_ALLOWANCE.get(info.get("default_effort") or "medium", 2500)
        return 0

    def estimate(self, wid: str) -> dict:
        """A rough cost of the next iteration: the real prompts' token counts (estimated from their length) and an
        assumed answer length (plus a reasoning allowance), priced with the model list. Mock runs cost nothing."""
        world, settings, state, next_period, history = self._prepare(wid)
        params = self.params(world, settings)
        rows: list[dict] = []
        actors = [a for a in world["actors"] if a.get("enabled")]
        placeholder = []
        for a in actors:
            model = self.model_for(world, settings, a.get("model") or "")
            msgs = P.actor_messages(world, a, state, history, next_period)
            info = self.catalog.lookup(model)
            rows.append({"who": a["name"], "model": model, "prompt_tokens": est_tokens("".join(m["content"] for m in msgs)),
                         "completion_tokens": ACTOR_OUT_TOKENS + self._allowance(params["reasoning_effort"], info),
                         "info": info})
            placeholder.append({"actor_id": a["id"], "name": a["name"], "status": "ok",
                                "output": {"action_id": "x", "action_label": "x", "action_detail": "x" * 400,
                                           "rationale": "x" * 600, "public_statement": "x" * 200,
                                           "expected_consequences": ["x" * 120] * 2}})
        smodel = self.model_for(world, settings, (world.get("synthesizer") or {}).get("model") or "")
        smsgs = P.synth_messages(world, state, placeholder, history, next_period)
        sinfo = self.catalog.lookup(smodel)
        nfields = len(world["schema"]["fields"])
        rows.append({"who": "synthesizer", "model": smodel,
                     "prompt_tokens": est_tokens("".join(m["content"] for m in smsgs)),
                     "completion_tokens": SYNTH_OUT_BASE + SYNTH_OUT_PER_FIELD * nfields
                     + self._allowance(params["reasoning_effort"], sinfo), "info": sinfo})
        total, priced = 0.0, True
        for r in rows:
            info = r.pop("info")
            if info and info.get("prompt_price") is not None and info.get("completion_price") is not None:
                r["cost"] = r["prompt_tokens"] * info["prompt_price"] + r["completion_tokens"] * info["completion_price"]
                total += r["cost"]
            else:
                r["cost"] = None
                priced = False
        return {"calls": len(rows), "rows": rows, "next_period": next_period,
                "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                "completion_tokens": sum(r["completion_tokens"] for r in rows),
                "cost": round(total, 6) if rows else 0.0, "priced": priced,
                "backend": effective_backend(settings),
                "assumptions": (f"prompt tokens estimated from the prompts' length; answers assumed ~{ACTOR_OUT_TOKENS} "
                                f"tokens per actor and ~{SYNTH_OUT_BASE} + {SYNTH_OUT_PER_FIELD}/field for the "
                                "synthesizer, plus a reasoning allowance for reasoning models; repairs and retries "
                                "not included")}


class RunManager:
    """At most one iteration per world at a time, each in its own thread; progress kept in memory."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self._lock = threading.Lock()
        self._runs: dict[str, Progress] = {}
        self._threads: dict[str, threading.Thread] = {}

    def busy(self, wid: str) -> bool:
        with self._lock:
            t = self._threads.get(wid)
            return bool(t and t.is_alive())

    def start(self, wid: str, mode: str = "run") -> dict:
        world = self.engine.store.load_world(wid)
        with self._lock:
            t = self._threads.get(wid)
            if t and t.is_alive():
                raise Busy("an iteration is already running for this world")
            state = self.engine.store.latest_step(wid)
            if mode == "retry":
                pending = self.engine.store.load_pending(wid)
                if not pending:
                    raise NotFound("there is no failed synthesizer attempt to retry")
                actors = [{"id": r["actor_id"], "name": r["name"]} for r in pending.get("actors") or []]
            else:
                actors = [a for a in world["actors"] if a.get("enabled")]
            period = M.advance_period(state.get("period"), world["period_step"]["unit"], world["period_step"]["count"])
            progress = Progress(wid, mode, actors, int(state["step"]), period)
            if mode == "retry":
                pend = self.engine.store.load_pending(wid) or {}
                for r in pend.get("actors") or []:
                    progress.actor(r["actor_id"], status="done" if r.get("status") == "ok" else "failed",
                                   action_label=(r.get("output") or {}).get("action_label"), error=r.get("error"),
                                   attempts=r.get("attempts", 0))
            self._runs[wid] = progress
            thread = threading.Thread(target=self._work, args=(wid, mode, progress), daemon=True,
                                      name=f"worldsim-run-{wid}")
            self._threads[wid] = thread
            thread.start()
        return progress.snapshot()

    def _work(self, wid: str, mode: str, progress: Progress) -> None:
        try:
            fn = self.engine.retry if mode == "retry" else self.engine.run
            result = fn(wid, progress)
            if result["status"] == "done":
                progress.set(status="done", phase="done", step=result["step"], finished=now_iso(), cost=result["cost"])
            else:
                progress.set(status="failed", phase="synthesizer", error=result.get("error"), finished=now_iso(),
                             cost=result.get("cost", 0.0))
        except Exception as e:  # noqa: BLE001 - shown in the UI, never raised into the server
            secrets = [self.engine.settings_loader().get("api_key") or ""]
            progress.set(status="failed", error=redact(f"{type(e).__name__}: {e}", secrets), finished=now_iso())

    def status(self, wid: str) -> dict | None:
        with self._lock:
            p = self._runs.get(wid)
        return p.snapshot() if p else None

    def wait(self, wid: str, timeout: float = 30) -> dict | None:
        with self._lock:
            t = self._threads.get(wid)
        if t:
            t.join(timeout)
        return self.status(wid)

    def forget(self, wid: str) -> None:
        with self._lock:
            self._runs.pop(wid, None)
