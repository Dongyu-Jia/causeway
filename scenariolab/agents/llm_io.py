"""The agent tree's LLM call layer (AGENT_DESIGN §10–§11; the spec's cli.py): one object per run, AgentLLM, that
the engine asks for actor decisions, merged node decisions, consequences, the opening translation, leaf stories and
leak probes. It builds the prompt (scenariolab/llm/agent_prompts.py), calls the backend (MSL_LLM_BACKEND:
claude_code, or mock in every test), parses and repairs the answer (scenariolab/llm/jsonfix.py), applies the
guardrails (scenariolab/llm/agent_guard.py) and falls back when the model cannot help. It never decides a
probability itself: pooling, joint profiles, reduction and the synthesizer's probability calculus are the engine's.

Per call (§10.4, §11.5):

    cache hit (content-addressed key, §11.4) ............................ no call
    budget (max_calls, the soft cap) reached ............................ prior-only fallback, flag "budget"
    hard cap (G9: 1.2 × the estimate) reached ........................... LLMPause("budget"): the run pauses
    timeout, 429, 529, 5xx .............................................. retry twice (30 s, 120 s)
    empty, unparseable or failed CLI output ............................. retry once, then fallback
    answer not parseable / structurally incomplete / probabilities off ... one P5 format repair, then fallback
    refusal (stop_reason, plain-text refusal, a declining actor) ........ one rephrase, then fallback (prior-only)
    max_tokens, context ................................................. fallback (prior-only)
    usage limit, login, token, CLI missing, model not allowed ........... LLMPause (the run pauses; resumable)
    cancel (graceful: no new call starts; calls in flight finish and are cached) ... LLMPause("cancelled")
    cancel with force (the kill event: the CLI's process group is killed) ........... LLMPause("cancelled")
    another node of the level paused (halt) ............................. LLMPause of the same kind, no new call
    the same failure class 3 times in a row ............................. LLMPause("breaker")

Calls run at most `concurrency` at once per run (a semaphore; the engine runs a level's calls in threads) and at most
MSL_LLM_CONCURRENCY at once in the whole process (backend.cli_slot, shared with on-demand expansions, stories and the
translator), identical requests share one call (claude_code's in-flight lock), and every returned text is cached, so
a rerun or a resume makes no new calls. Prompts never contain a run id, node id or timestamp.

The paid-calls ledger (ledger=<path>): the cache keys this run paid for, one per line, append-only. A resumed run
counts its own earlier calls, now cache hits, against its budget again in the order it meets them, so a pause and
resume consumes the budget exactly as an uninterrupted run; calls_paid (the run's spend) is the ledger's size.

Real Claude Code runs only from an explicit, confirmed user action: with the claude_code backend a new call needs
Settings(confirmed=True), which the engine sets only when the user confirmed the run's estimate; without it the
first new call raises LLMPause("unconfirmed") before anything starts. Cache hits need no confirmation (a rerun or a
dry replay costs nothing), and the mock backend needs none.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..llm import agent_guard as guard
from ..llm import agent_prompts as prompts
from ..llm import backend
from ..llm import claude_code as cc
from ..llm import jsonfix
from ..llm.agent_guard import ActorAnswer, ClusterAnswer
from ..llm.agent_prompts import (ActorSpec, ClusterSpec, HazardSpec, Lever, Magnitude, MenuOption, NodeContext,
                                 ReactionSpec, SynthRequest)

__all__ = ["AgentLLM", "Settings", "FileCache", "LLMPause", "CallRecord", "ActorResult", "NodeResult",
           "SynthResult", "RootResult", "TextResult", "ActorSpec", "MenuOption", "Magnitude", "Lever",
           "ReactionSpec", "HazardSpec", "NodeContext", "ClusterSpec", "SynthRequest", "ActorAnswer",
           "ClusterAnswer", "actor_spec_from_menu", "action_for", "llm_status", "cache_key", "PROMPT_VERSION"]

PROMPT_VERSION = prompts.PROMPT_VERSION
RETRY_KINDS = frozenset({"timeout", "rate", "overloaded", "server"})
ONCE_KINDS = frozenset({"failed", "unparseable", "empty", "max_turns"})
PRIOR_ONLY_KINDS = frozenset({"refusal", "max_tokens", "context"})
PAUSE_KINDS = frozenset({"limit", "login", "token", "cli_missing", "model", "forbidden"})
PAUSE_REASON = {"limit": "usage_limit", "login": "auth", "token": "auth", "cli_missing": "cli_missing",
                "model": "model", "forbidden": "model", "cancelled": "cancelled", "breaker": "breaker",
                "unconfirmed": "unconfirmed"}
E_UNCONFIRMED = ("这次推演需要调用本机 Claude Code（计入你的订阅额度），但还没有得到你的确认：请在界面上查看预计用量并确认后"
                 "再开始。已缓存的调用不受影响。")
# every agent-tree call: claude-opus-5-5 at effort medium (USER_DECISIONS 2026-11-02: 「每一步其实你也可以就用 opus 5.5
# medium」); --effort is passed only when `claude --help` lists it (2.1.278 does)
DEFAULT_EFFORT = {"actor_root": "medium", "actor": "medium", "node": "medium", "synth": "medium", "root": "medium",
                  "repair": "medium", "probe": "medium", "story": "medium"}
E_HARD_CAP = ("已达到这次推演的用量上限（预计调用的 1.2 倍，{cap} 次新调用）：推演暂停，已完成的调用都在缓存里。"
              "确认后继续（会按剩余部分重新估计并提高上限）。")
STATUS_TTL_S = 60.0
DEFAULT_TIMEOUTS = {"actor": 180.0, "node": 300.0, "synth": 300.0, "root": 180.0, "repair": 90.0, "probe": 60.0,
                    "story": 180.0}
CONTAINER = {"actor": "options", "node": "actors", "synth": "clusters", "root": "h0", "story": "story",
             "probe": "year_guess"}
REFUSAL_TEXT = re.compile(r"I can(?:'|no)t help|I won't|I'm not able to|I am not able to|无法协助|不能协助|我不能扮演|"
                          r"无法完成这个请求|不能提供这类")
FAILURE_ZH = {"timeout": "超时", "rate": "请求过于频繁", "overloaded": "服务繁忙", "server": "服务暂时不可用",
              "failed": "调用失败", "unparseable": "输出无法解析", "empty": "结果为空", "max_turns": "没有给出结果",
              "refusal": "模型拒绝", "max_tokens": "输出达到长度上限", "context": "输入超出上下文窗口",
              "budget": "已达到本次推演的调用上限", "invalid_output": "输出无效"}
LATENCY_WINDOW = 20


def action_for(kind: str) -> str:
    """retry | retry_once | prior_only | pause | cancel for a ClaudeCodeError kind."""
    if kind == "cancelled":
        return "cancel"
    if kind in PAUSE_KINDS:
        return "pause"
    if kind in RETRY_KINDS:
        return "retry"
    if kind in PRIOR_ONLY_KINDS:
        return "prior_only"
    return "retry_once"


class LLMPause(Exception):
    """The run must pause (resumable): kind is the error kind ("limit", "login", …, "cancelled", "breaker"),
    reason the journal's run_paused reason, reset_at the epoch a usage limit resets (when the CLI said)."""

    def __init__(self, kind: str, message: str, *, reset_at: float | None = None, record: CallRecord | None = None):
        super().__init__(message)
        self.kind = kind
        self.reason = PAUSE_REASON.get(kind, kind)
        self.reset_at = reset_at
        self.record = record


@dataclass
class Settings:
    """Per-run LLM settings (AGENT_DESIGN §11.2). versions: the engine's hashes that enter every cache key
    (engine, registry, catalogue, …)."""
    model: str | None = None                  # default MSL_LLM_MODEL / claude-opus-5-5
    backend: str | None = None                # default MSL_LLM_BACKEND
    effort: dict = field(default_factory=lambda: dict(DEFAULT_EFFORT))
    timeouts: dict = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))
    safe_mode: bool = True                    # --safe-mode when the installed CLI lists it (U-5)
    retries: int = 2
    retry_sleep: tuple = (30.0, 120.0)
    max_calls: int | None = 60                # new calls; past it actors go prior-only (None: no cap)
    hard_cap: int | None = None               # G9: new calls; reaching it pauses the run with reason "budget"
    breaker: int = 3
    rephrase: bool = True
    concurrency: int = 3
    output_mode: str = "text"
    versions: dict = field(default_factory=dict)
    confirmed: bool = False                   # the user confirmed this run: new claude_code calls may start

    @classmethod
    def coerce(cls, s) -> Settings:
        if s is None:
            return cls()
        if isinstance(s, Settings):
            return s
        d = dict(s)
        out = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k not in ("effort", "timeouts")})
        out.effort = {**DEFAULT_EFFORT, **dict(d.get("effort") or {})}
        out.timeouts = {**DEFAULT_TIMEOUTS, **{k: float(v) for k, v in dict(d.get("timeouts") or {}).items()}}
        return out


@dataclass
class CallRecord:
    """One logical call (its retries included) for the journal and the node detail's calls[]."""
    key: str
    kind: str                                 # actor | node | synth | root | repair | story | probe
    purpose: str = "call"                     # call | repair | rephrase
    cached: bool = False
    shared: bool = False                      # answered by an identical request already in flight
    ok: bool = False
    attempts: int = 0
    error_kind: str | None = None            # why the call failed (None when it ended with an answer)
    error: str | None = None
    retried: list = field(default_factory=list)   # error kinds of the attempts that were retried
    seconds: float = 0.0
    model: str | None = None
    effort: str | None = None
    safe_mode: bool = False
    backend: str | None = None
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActorResult:
    answer: ActorAnswer
    calls: list = field(default_factory=list)
    report: list = field(default_factory=list)
    flags: list = field(default_factory=list)     # fallback, prior_only, repaired, rephrased, budget, declined

    def to_dict(self) -> dict:
        return {"answer": self.answer.to_dict(), "calls": [c.to_dict() for c in self.calls], "report": self.report,
                "flags": self.flags}


@dataclass
class NodeResult:
    actors: dict = field(default_factory=dict)     # actor id -> ActorAnswer
    reactions: dict = field(default_factory=dict)  # leader -> follower -> {leader option: {follower option: p}}
    hazards: dict = field(default_factory=dict)    # hazard id -> tilted probability (LLM; the engine pools it)
    calls: list = field(default_factory=list)
    report: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"actors": {k: v.to_dict() for k, v in self.actors.items()}, "reactions": self.reactions,
                "hazards": self.hazards, "calls": [c.to_dict() for c in self.calls], "report": self.report,
                "flags": self.flags}


@dataclass
class SynthResult:
    clusters: dict = field(default_factory=dict)   # cluster id -> ClusterAnswer (status ok | default)
    calls: list = field(default_factory=list)
    report: list = field(default_factory=list)
    flags: list = field(default_factory=list)      # synth_default (该节点后果未经大模型判断), repaired, budget

    def to_dict(self) -> dict:
        return {"clusters": {k: v.to_dict() for k, v in self.clusters.items()},
                "calls": [c.to_dict() for c in self.calls], "report": self.report, "flags": self.flags}


@dataclass
class RootResult:
    ok: bool = False
    h0: dict | None = None
    forced: list = field(default_factory=list)
    user_views: list = field(default_factory=list)
    unmapped: list = field(default_factory=list)
    error: str = ""
    calls: list = field(default_factory=list)
    report: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**{k: v for k, v in asdict(self).items() if k != "calls"}, "calls": [c.to_dict() for c in self.calls]}


@dataclass
class TextResult:
    """P6 / P7: the validated payload, or None with the error."""
    ok: bool = False
    payload: dict | None = None
    error: str = ""
    calls: list = field(default_factory=list)
    report: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "payload": self.payload, "error": self.error,
                "calls": [c.to_dict() for c in self.calls], "report": self.report}


# ---------------------------------------------------------------- cache

class FileCache:
    """<root>/<key[:2]>/<key>.json: content-addressed, atomic writes, a torn or unreadable file is a miss. The
    engine may pass its own cache instead (anything with get(key) and put(key, record))."""

    def __init__(self, root):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        try:
            rec = json.loads(self.path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return rec if isinstance(rec, dict) else None

    def put(self, key: str, record: dict) -> None:
        path = self.path(key)
        tmp = path.with_name(f".{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(record, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(tmp, path)
        except (OSError, ValueError):
            try:
                tmp.unlink()
            except OSError:
                pass


def cache_key(*, kind: str, system: str, user: str, model: str, backend_name: str, cli_version: str | None,
              effort: str | None, safe_mode: bool, output_mode: str, versions: dict, sample: int, repair: int) -> str:
    """sha256 of {engine/registry/catalogue versions, prompt version, backend, model (+ CLI version), effort, safe
    mode, output mode, sha256(system), sha256(user), sample index, repair index} (AGENT_DESIGN §11.4)."""
    h = lambda s: hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()  # noqa: E731
    blob = {"versions": dict(sorted((versions or {}).items())), "prompts": PROMPT_VERSION, "kind": kind,
            "backend": backend_name, "model": model, "cli": cli_version, "effort": effort, "safe_mode": bool(safe_mode),
            "output": output_mode, "system": h(system), "user": h(user), "sample": int(sample), "repair": int(repair)}
    return h(jsonfix.canonical_json(blob))


# ---------------------------------------------------------------- process-wide status (for /api/agents/llm_status)

_STATUS_LOCK = threading.Lock()
_LATENCY: deque = deque(maxlen=LATENCY_WINDOW)       # seconds of the last real calls (not cached, not mock)
_LIMIT: dict = {}                                    # the last usage-limit failure: {at, reset_at, message}


def _note_latency(seconds: float) -> None:
    with _STATUS_LOCK:
        _LATENCY.append(float(seconds))


def _note_limit(err: cc.ClaudeCodeError) -> None:
    with _STATUS_LOCK:
        _LIMIT.clear()
        _LIMIT.update(at=time.time(), reset_at=err.reset_at, message=str(err))


_STATUS_CACHE: dict = {}


def _cached_status(model: str, name: str) -> dict:
    """backend.status for STATUS_TTL_S (it starts `claude --version` and `claude auth status`): repeated status or
    estimate requests do not start CLI processes."""
    key = (name, model, os.environ.get("MSL_CLAUDE_BIN", ""))
    now = time.monotonic()
    with _STATUS_LOCK:
        hit = _STATUS_CACHE.get(key)
        if hit is not None and now - hit[0] < STATUS_TTL_S:
            return dict(hit[1])
    st = backend.status(model, name)
    with _STATUS_LOCK:
        _STATUS_CACHE[key] = (now, dict(st))
    return st


def clear_status_cache() -> None:
    with _STATUS_LOCK:
        _STATUS_CACHE.clear()


def llm_status(*, concurrency: int = 3) -> dict:
    """{available, backend, label, cli_version, model, reason, auth, flags, concurrency, latency, limited,
    limited_reason, reset_at, mock}: local checks only (claude --version, auth status, --help), never a model
    call. `limited` stays true for 5 hours after a usage-limit failure unless the CLI gave an earlier reset."""
    name, model = backend.backend_name(), backend.model_id()
    st = _cached_status(model, name)
    caps = backend.capabilities(name) if st.get("available") else {"flags": {}}
    with _STATUS_LOCK:
        lat = list(_LATENCY)
        limit = dict(_LIMIT)
    now = time.time()
    until = limit.get("reset_at") or (limit.get("at", 0) + 5 * 3600 if limit else 0)
    limited = bool(limit) and now < until
    latency = {"n": len(lat), "p50_s": round(statistics.median(lat), 1) if lat else None,
               "p90_s": round(sorted(lat)[max(0, int(round(0.9 * len(lat))) - 1)], 1) if lat else None}
    return {"available": bool(st.get("available")), "backend": st.get("backend", name), "label": st.get("label"),
            "cli_version": st.get("version"), "model": model, "reason": st.get("reason", ""), "auth": st.get("auth"),
            "flags": caps.get("flags") or {}, "concurrency": concurrency, "latency": latency, "limited": limited,
            "limited_reason": limit.get("message", "") if limited else "", "reset_at": until if limited else None,
            "mock": name == "mock"}


# ---------------------------------------------------------------- catalogue adapter

def _writes_text(writes, param) -> str:
    """The catalogue's option writes as one line for the menu table (e.g. "d_ff = 幅度", "d_itinv 增量 0.16×幅度，4 季")."""
    parts = []
    for w in writes or []:
        fid = w.get("field") or w.get("role")
        if not fid:
            continue
        kind, a, b, dq = w.get("kind"), w.get("a"), w.get("b"), int(w.get("dq") or 0)
        if kind == "event":
            amount = w.get("z", w.get("sd"))
            val = f"{prompts.num(amount)}{'×幅度' if w.get('per_unit') else ''}" if amount is not None else "按选项"
            text = f"{fid} 事件贡献 {val}"
        elif a is None:
            text = f"{fid} = 幅度" + (f"（{param.get('unit')}）" if param and param.get("unit") else "")
        else:
            av = prompts.num(a) + ("×幅度" if w.get("per_unit") else "")
            sd = f" ± {prompts.num(b)}" if b else ""
            text = f"{fid} {'+= ' if kind == 'shift' else '= '}{av}{sd}"
        if dq:
            text += f"（{dq} 季后生效）"
        if w.get("fallback") and not w.get("field"):
            text += f"（后备：{w['fallback']}）"
        parts.append(text)
    return "；".join(parts) or "—"


def _point(writes, param) -> dict:
    """Expected writes {field: value} of a catalogue option (the magnitude's default where it scales)."""
    out = {}
    default = (param or {}).get("default")
    for w in writes or []:
        fid = w.get("field")
        if not fid or w.get("kind") == "event":
            continue
        a = w.get("a")
        if a is None:
            a = default
        elif w.get("per_unit") and default is not None:
            a = a * default
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            out[fid] = out.get(fid, 0.0) + float(a)
    return out


def actor_spec_from_menu(menu: dict, prior: dict | None = None, **extra) -> ActorSpec:
    """An ActorSpec from the catalogue's node menu (scenariolab.agents.actors.menu(): {"actor", "zh", "main":
    {"options": [option summaries], "required": [...]}}) and the prior (priors.prior(): {"p", "source", "label",
    "kind"}). extra: any ActorSpec field (levers, facts, custom, custom_cap, instruments, packet, private,
    calendar, commitments, mandate, …)."""
    main = (menu or {}).get("main") or {}
    p = (prior or {}).get("p") or {}
    uniform = (prior or {}).get("kind") == "uniform"
    required = set(main.get("required") or [])
    options = []
    for o in main.get("options") or []:
        param = o.get("param")
        sig = o.get("signal") or {}
        options.append(MenuOption(
            id=o["id"], label=o.get("zh") or o["id"],
            base_rate=None if uniform else p.get(o["id"]), status_quo=bool(o.get("status_quo")),
            magnitude=Magnitude.coerce(param) if param else None, writes=_writes_text(o.get("writes"), param),
            point=_point(o.get("writes"), param), note=o.get("note") or "", required=o["id"] in required,
            signal=str(sig.get("text") or "")))
    kw = {"id": menu.get("actor"), "zh": menu.get("zh") or menu.get("actor"), "options": options,
          "base_rate_source": "" if uniform else str((prior or {}).get("source") or ""),
          "base_rate_condition": "" if uniform else str((prior or {}).get("condition") or "")}
    kw.update(extra)
    return ActorSpec(**kw)


# ---------------------------------------------------------------- the session

def _cancel_wait(cancel: threading.Event | None, seconds: float, sleep) -> bool:
    """Sleep `seconds`; True when the cancel event fired."""
    if seconds <= 0:
        return bool(cancel is not None and cancel.is_set())
    if sleep is not None:
        sleep(seconds)
        return bool(cancel is not None and cancel.is_set())
    if cancel is not None:
        return cancel.wait(seconds)
    time.sleep(seconds)
    return False


@dataclass
class _Outcome:
    payload: dict | None = None
    text: str | None = None
    failure: str | None = None                # error kind when there is no usable payload
    notes: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    repaired: bool = False


class AgentLLM:
    """One run's LLM access. Thread-safe: the engine may call it from several threads (calls in a level).

        llm = AgentLLM(Settings(max_calls=60, versions={...}), cache=FileCache(...), cancel=run_cancel_event,
                       on_event=journal_emit, failure_log=Path("data/agents/cli_failures.jsonl"))
        res = llm.actor(ctx, actor_spec)                    # P2 (root, promoted actors)
        res = llm.node(ctx, actor_specs, reactions=..., hazards=...)   # P1 merged
        res = llm.synth(synth_request)                      # S
        res = llm.root(ctx, narrative, settable_fields)     # P4
    """

    def __init__(self, settings=None, *, cache=None, cancel: threading.Event | None = None, failure_log=None,
                 on_event=None, sleep=None, kill: threading.Event | None = None, ledger=None):
        self.settings = Settings.coerce(settings)
        self.cache = cache
        self.cancel = cancel                     # graceful: checked before every new call
        self.kill = kill                         # force: passed to the backend, kills the CLI's process group
        self.ledger = Path(ledger) if ledger else None
        self.level = 0                           # the tree level being expanded (set by the engine; in the ledger)
        self._paid_level: dict[str, int] = {}
        if self.ledger is not None and self.ledger.exists():
            try:
                for ln in self.ledger.read_text(encoding="utf-8").splitlines():
                    k, _, lv = ln.strip().partition("\t")
                    if k:
                        self._paid_level.setdefault(k, int(lv) if lv.strip().lstrip("-").isdigit() else 0)
            except OSError:
                self._paid_level = {}
        self.paid: set[str] = set(self._paid_level)
        self._consumed: set[str] = set()
        self._halt: LLMPause | None = None
        self.failure_log = Path(failure_log) if failure_log else None
        self.on_event = on_event
        self._sleep = sleep                      # tests inject a no-op; None waits on the cancel event
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max(1, min(int(self.settings.concurrency or 1), 6)))
        self._streak: tuple[str | None, int] = (None, 0)
        self._logged: set[str] = set()
        self._caps: dict | None = None
        self.stats = {"calls_new": 0, "cached": 0, "shared": 0, "failed": 0, "retries": 0, "repairs": 0,
                      "rephrases": 0, "fallbacks": 0, "budget_hits": 0, "seconds": 0.0,
                      "usage": {k: 0 for k in cc.USAGE_KEYS}, "cost_usd": 0.0, "calls_paid": len(self.paid)}

    # ------------------------------------------------------------ identity of a call

    @property
    def backend_name(self) -> str:
        return backend.normalize(self.settings.backend) if self.settings.backend else backend.backend_name()

    @property
    def model(self) -> str:
        return self.settings.model or backend.model_id()

    def capabilities(self) -> dict:
        if self._caps is None:
            self._caps = backend.capabilities(self.backend_name)
        return self._caps

    def _effort(self, key: str) -> str | None:
        return self.settings.effort.get(key) or None

    def _timeout(self, key: str) -> float:
        return min(float(self.settings.timeouts.get(key, backend.timeout_ceiling())), backend.timeout_ceiling())

    def key(self, kind: str, system: str, user: str, *, effort: str | None, sample: int = 0, repair: int = 0) -> str:
        return cache_key(kind=kind, system=system, user=user, model=self.model, backend_name=self.backend_name,
                         cli_version=self.capabilities().get("cli_version"), effort=effort,
                         safe_mode=self.settings.safe_mode, output_mode=self.settings.output_mode,
                         versions=self.settings.versions, sample=sample, repair=repair)

    # ------------------------------------------------------------ bookkeeping

    def _emit(self, event: str, **payload) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event, **payload)
            except Exception:              # a journal failure must not lose the call
                pass

    def _check_cancel(self, rec: CallRecord | None = None) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise LLMPause("cancelled", "推演已取消", record=rec)
        if self._halt is not None:
            raise LLMPause(self._halt.kind, str(self._halt), reset_at=self._halt.reset_at, record=rec)

    def halt(self, err: LLMPause) -> None:
        """Another call of the level paused (usage limit, login, budget, …): no new call starts in this run."""
        with self._lock:
            if self._halt is None:
                self._halt = err

    def _reserve(self, key: str) -> bool:
        """Count one new call against the budget: False past the soft cap (prior-only); LLMPause at the hard cap."""
        with self._lock:
            hard = self.settings.hard_cap
            if hard is not None and self.stats["calls_new"] >= int(hard):
                self.stats["budget_hits"] += 1
                raise LLMPause("budget", E_HARD_CAP.format(cap=int(hard)))
            cap = self.settings.max_calls
            if cap is not None and self.stats["calls_new"] >= cap:
                self.stats["budget_hits"] += 1
                return False
            self.stats["calls_new"] += 1
            self._consumed.add(key)
            return True

    def _count_paid_hit(self, key: str) -> None:
        """A cache hit on a call this run paid for before a pause: it counts against the budget again, once, so the
        resumed run spends exactly what the uninterrupted run would."""
        with self._lock:
            if key in self.paid and key not in self._consumed:
                self._consumed.add(key)
                self.stats["calls_new"] += 1

    def _note_paid(self, key: str) -> None:
        with self._lock:
            if key in self.paid:
                return
            self.paid.add(key)
            self._paid_level[key] = int(self.level)
            self.stats["calls_paid"] = len(self.paid)
            if self.ledger is None:
                return
            try:
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                with self.ledger.open("a", encoding="utf-8") as fh:
                    fh.write(f"{key}\t{int(self.level)}\n")
            except OSError:
                pass

    def restore_consumed(self, before_level: int) -> None:
        """A resume at `before_level`: the calls paid in earlier levels were consumed (counted) before the
        checkpoint; the ones paid in the interrupted level are counted again when the resumed run meets them."""
        with self._lock:
            self._consumed = {k for k, lv in self._paid_level.items() if lv < int(before_level)}
            self.stats["calls_paid"] = len(self.paid)

    def _log_failure(self, err: cc.ClaudeCodeError) -> None:
        """The first verbatim (redacted) failure text per class, appended to failure_log (cli_failures.jsonl)."""
        with self._lock:
            if self.failure_log is None or err.kind in self._logged:
                return
            self._logged.add(err.kind)
        try:
            seen = set()
            if self.failure_log.exists():
                for line in self.failure_log.read_text(encoding="utf-8").splitlines():
                    try:
                        seen.add(json.loads(line).get("kind"))
                    except ValueError:
                        continue
            if err.kind in seen:
                return
            self.failure_log.parent.mkdir(parents=True, exist_ok=True)
            with self.failure_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": err.kind, "message": str(err), "detail": cc.redact(err.detail),
                                     "backend": self.backend_name, "at": time.strftime("%Y-%m-%d %H:%M:%S")},
                                    ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _streak_fail(self, kind: str, rec: CallRecord) -> None:
        with self._lock:
            last, n = self._streak
            n = n + 1 if last == kind else 1
            self._streak = (kind, n)
        if self.settings.breaker and n >= self.settings.breaker:
            raise LLMPause("breaker", f"连续 {n} 次调用以同一类失败结束（{FAILURE_ZH.get(kind, kind)}），推演已暂停。",
                           record=rec)

    def _streak_ok(self) -> None:
        with self._lock:
            self._streak = (None, 0)

    # ------------------------------------------------------------ one logical call

    def _cached(self, key: str, rec: CallRecord):
        """(text, meta, rec) from the cache, or None."""
        hit = self.cache.get(key) if self.cache is not None else None
        if hit is None or not isinstance(hit.get("text"), str):
            return None
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        rec.cached, rec.ok, rec.model = True, True, meta.get("model") or self.model
        rec.error_kind = rec.error = None
        with self._lock:
            self.stats["cached"] += 1
        self._emit("call_done", key=rec.key[:12], call=rec.kind, purpose=rec.purpose, cached=True, ok=True,
                   seconds=0.0)
        return hit["text"], meta, rec

    def ask(self, kind: str, system: str, user: str, *, effort_key: str, timeout_key: str, sample: int = 0,
            repair: int = 0, purpose: str = "call") -> tuple[str | None, dict | None, CallRecord]:
        """(text, meta, record) for one logical call: the cache, the budget, retries and pauses. text is None when
        the call failed in a way the caller falls back from (record.error_kind says why)."""
        effort = self._effort(effort_key)
        key = self.key(kind, system, user, effort=effort, sample=sample, repair=repair)
        rec = CallRecord(key=key, kind=kind, purpose=purpose, effort=effort, safe_mode=self.settings.safe_mode,
                         backend=self.backend_name, model=self.model)
        hit = self._cached(key, rec)
        if hit is not None:
            self._count_paid_hit(key)
            return hit
        self._check_cancel(rec)
        if self.backend_name == "claude_code" and not self.settings.confirmed:
            rec.error_kind, rec.error = "unconfirmed", E_UNCONFIRMED
            raise LLMPause("unconfirmed", E_UNCONFIRMED, record=rec)
        timeout = self._timeout(timeout_key)
        while True:
            self._check_cancel(rec)
            if not self._reserve(key):
                rec.error_kind, rec.error = "budget", FAILURE_ZH["budget"]
                self._emit("call_done", key=key[:12], call=kind, purpose=purpose, cached=False, ok=False,
                           error="budget")
                return None, None, rec
            rec.attempts += 1
            self._emit("call_started", key=key[:12], call=kind, purpose=purpose, attempt=rec.attempts)
            t0 = time.monotonic()
            try:
                with self._slots:
                    self._check_cancel(rec)
                    hit = self._cached(key, rec)          # answered while this call waited for a slot
                    if hit is not None:                   # (the identical call that answered consumed the key)
                        with self._lock:
                            self.stats["calls_new"] -= 1
                        self._count_paid_hit(key)
                        return hit
                    with backend.cli_slot(self.cancel):
                        self._check_cancel(rec)
                        text, meta = backend.run(system, user, self.model, timeout, effort=effort,
                                                 safe_mode=self.settings.safe_mode, cancel=self.kill,
                                                 backend=self.backend_name)
                    if self.cache is not None:            # stored inside the slot, for any identical call waiting
                        self.cache.put(key, {"key": key, "kind": kind, "purpose": purpose, "text": text, "meta": meta,
                                             "seconds": round(time.monotonic() - t0, 3),
                                             "created": time.strftime("%Y-%m-%d %H:%M:%S")})
            except cc.ClaudeCodeError as err:
                rec.seconds += time.monotonic() - t0
                rec.error_kind, rec.error = err.kind, str(err)
                self._log_failure(err)
                action = action_for(err.kind)
                self._emit("call_done", key=key[:12], call=kind, purpose=purpose, cached=False, ok=False,
                           error=err.kind, retry=rec.attempts - 1, seconds=round(rec.seconds, 2))
                if action == "cancel":
                    raise LLMPause("cancelled", str(err), record=rec) from None
                if action == "pause":
                    if err.kind == "limit":
                        _note_limit(err)
                    raise LLMPause(err.kind, str(err), reset_at=err.reset_at, record=rec) from None
                limit = self.settings.retries if action == "retry" else (1 if action == "retry_once" else 0)
                if rec.attempts <= limit:
                    rec.retried.append(err.kind)
                    with self._lock:
                        self.stats["retries"] += 1
                    wait = self.settings.retry_sleep[min(rec.attempts - 1, len(self.settings.retry_sleep) - 1)] \
                        if self.settings.retry_sleep else 0.0
                    if _cancel_wait(self.cancel, wait, self._sleep):
                        raise LLMPause("cancelled", "推演已取消", record=rec) from None
                    continue
                with self._lock:
                    self.stats["failed"] += 1
                self._streak_fail(err.kind, rec)
                return None, None, rec
            seconds = time.monotonic() - t0
            rec.seconds += seconds
            rec.ok, rec.model = True, meta.get("model") or self.model
            rec.error_kind = rec.error = None
            rec.usage, rec.cost_usd = dict(meta.get("usage") or {}), meta.get("cost_usd")
            with self._lock:
                if meta.get("shared_inflight"):
                    rec.shared = True
                    self.stats["calls_new"] -= 1
                    self.stats["shared"] += 1
                self.stats["seconds"] += seconds
                for k in cc.USAGE_KEYS:
                    self.stats["usage"][k] += int(rec.usage.get(k) or 0)
                self.stats["cost_usd"] += float(rec.cost_usd or 0.0)
            if not rec.shared:
                self._note_paid(key)
            if meta.get("backend") == "claude_code" and not rec.shared:
                _note_latency(seconds)
            self._streak_ok()
            self._emit("call_done", key=key[:12], call=kind, purpose=purpose, cached=False, ok=True,
                       retry=rec.attempts - 1, seconds=round(seconds, 2), shared=rec.shared)
            return text, meta, rec

    # ------------------------------------------------------------ parse, validate, repair

    @staticmethod
    def parse(text: str, schema: dict, container: str) -> tuple[dict | None, list[str], list[str]]:
        """(payload, notes, problems): problems are what a format repair could fix (no JSON, the wrong top level,
        a missing or mistyped main container); everything else is left to the guard and only reported."""
        obj, notes = jsonfix.extract_json(text)
        if obj is None:
            return None, notes, [notes[-1] if notes else "输出不是合法的 JSON"]
        problems, shown = [], []
        for e in jsonfix.validate(obj, schema):
            depth = e.path.count(".") + e.path.count("[")
            if e.kind in ("required", "type") and depth <= 1 and (e.path in ("$", f"$.{container}")):
                problems.append(str(e))
            elif len(shown) < 8:
                shown.append(f"格式：{e}")
        version = schema.get("properties", {}).get("schema_version", {}).get("const")
        if version and obj.get("schema_version") not in (version, None):
            shown.append(f"格式：schema_version 应为 {version}，已忽略")
        return obj, notes + shown, problems

    def _obtain(self, kind: str, system: str, user: str, schema: dict, *, effort_key: str, timeout_key: str,
                sample: int, purpose: str = "call") -> _Outcome:
        """One call and, when its text cannot be used, one P5 repair."""
        out = _Outcome()
        text, _, rec = self.ask(kind, system, user, effort_key=effort_key, timeout_key=timeout_key, sample=sample,
                                purpose=purpose)
        out.calls.append(rec)
        if text is None:
            out.failure = rec.error_kind or "failed"
            return out
        out.text = text
        payload, notes, problems = self.parse(text, schema, CONTAINER[kind])
        out.notes += notes
        if payload is None and REFUSAL_TEXT.search(text):
            out.failure = "refusal"
            return out
        if problems:
            payload, more = self.repair(kind, schema, text, problems, sample)
            out.calls += more.calls
            out.notes += more.notes
            out.repaired = True
            if payload is None:
                out.failure = more.failure or "invalid_output"
                out.problems = problems
                return out
        out.payload = payload
        return out

    def repair(self, kind: str, schema: dict, text: str, problems: list, sample: int = 0) -> tuple[dict | None, _Outcome]:
        """One P5 format-only repair (no state; it cannot add content): (payload or None, outcome)."""
        out = _Outcome()
        with self._lock:
            self.stats["repairs"] += 1
        system, user = prompts.repair_system(schema), prompts.repair_user(problems, text)
        fixed, _, rec = self.ask(kind, system, user, effort_key="repair", timeout_key="repair", sample=sample,
                                 repair=1, purpose="repair")
        out.calls.append(rec)
        if fixed is None:
            out.failure = rec.error_kind or "failed"
            out.notes.append(f"格式修复调用失败（{FAILURE_ZH.get(out.failure, out.failure)}）")
            return None, out
        payload, notes, still = self.parse(fixed, schema, CONTAINER[kind])
        out.notes += [f"修复后：{n}" for n in notes]
        if payload is None or still:
            out.failure = "invalid_output"
            out.notes.append("格式修复后仍不合格：" + "；".join(still or ["不是合法的 JSON"])[:200])
            return None, out
        out.notes.append("经过一次格式修复")
        return payload, out

    @staticmethod
    def _allowed(*texts) -> set:
        return guard.allowed_numbers(*texts)

    # ------------------------------------------------------------ P2: one actor

    def actor(self, ctx, actor, *, follower=None, sample: int = 0, variant: str = "V0",
              root: bool | None = None) -> ActorResult:
        """One actor in its own call (the root's separate mode; promoted actors; a declining actor's rephrase).
        The answer's options carry p_llm (or, for a follower, reaction rows); a fallback is prior-only."""
        ctx = NodeContext.coerce(ctx)
        actor = ActorSpec.coerce(actor)
        follower = ReactionSpec.coerce(follower) if follower is not None else None
        is_root = (ctx.quarter_done is None) if root is None else bool(root)
        return self._actor(ctx, actor, follower=follower, sample=sample, variant=variant,
                           effort_key="actor_root" if is_root else "actor", rephrase=False)

    def _actor(self, ctx: NodeContext, actor: ActorSpec, *, follower, sample: int, variant: str, effort_key: str,
               rephrase: bool) -> ActorResult:
        schema = prompts.actor_output_schema(actor, follower)
        system = prompts.actor_system(schema, ctx, actor.custom_cap)
        user = prompts.actor_user(ctx, actor, follower=follower, variant=variant, rephrase=rephrase)
        allowed = self._allowed(user)
        purpose = "rephrase" if rephrase else "call"
        o = self._obtain("actor", system, user, schema, effort_key=effort_key, timeout_key="actor", sample=sample,
                         purpose=purpose)
        result = ActorResult(answer=ActorAnswer(actor=actor.id), calls=list(o.calls), report=list(o.notes))
        if rephrase:
            result.flags.append("rephrased")
        if o.repaired:
            result.flags.append("repaired")
        if o.payload is None:
            if o.failure == "refusal" and not rephrase and self.settings.rephrase:
                return self._merge_rephrase(result, self._rephrase_actor(ctx, actor, follower, sample, variant,
                                                                         effort_key))
            result.answer = self._fallback(actor, o.failure, o.problems)
            result.flags += ["fallback", "prior_only"] + (["budget"] if o.failure == "budget" else [])
            result.report += result.answer.report
            return result
        ans = guard.guard_actor(o.payload, actor, allowed=allowed, follower=follower)
        if ans.problems and not o.repaired:
            fixed, more = self.repair("actor", schema, o.text or "", ans.problems, sample)
            result.calls += more.calls
            result.report += more.notes
            result.flags.append("repaired")
            if fixed is not None:
                again = guard.guard_actor(fixed, actor, allowed=allowed, follower=follower)
                if not again.problems:
                    ans = again
        if ans.declined and not rephrase and self.settings.rephrase:
            result.report += ans.report
            return self._merge_rephrase(result, self._rephrase_actor(ctx, actor, follower, sample, variant,
                                                                     effort_key))
        if ans.problems:
            ans = self._fallback(actor, "invalid_output", ans.problems, report=ans.report)
        result.answer = ans
        result.report += ans.report
        if ans.fallback:
            result.flags += ["fallback", "prior_only"]
        if ans.declined:
            result.flags.append("declined")
        return result

    def _rephrase_actor(self, ctx, actor, follower, sample, variant, effort_key) -> ActorResult:
        with self._lock:
            self.stats["rephrases"] += 1
        return self._actor(ctx, actor, follower=follower, sample=sample, variant=variant, effort_key=effort_key,
                           rephrase=True)

    @staticmethod
    def _merge_rephrase(first: ActorResult, second: ActorResult) -> ActorResult:
        second.calls = first.calls + second.calls
        second.report = first.report + ["拒绝评估或拒绝回答，换一种说法重问一次"] + second.report
        second.flags = list(dict.fromkeys(first.flags + second.flags))
        return second

    def _fallback(self, actor: ActorSpec, failure: str | None, problems=(), report=()) -> ActorAnswer:
        with self._lock:
            self.stats["fallbacks"] += 1
        why = FAILURE_ZH.get(failure or "", failure or "输出无效")
        if problems:
            why += "：" + "；".join(str(p) for p in problems)[:120]
        ans = guard.fallback_answer(actor, why, declined=failure == "refusal")
        ans.report = list(report) + ans.report
        return ans

    # ------------------------------------------------------------ P1: the merged node call

    def node(self, ctx, actors, *, reactions=(), hazards=(), sample: int = 0, variant: str = "V0") -> NodeResult:
        """Every active actor in one call (below the root). Actors whose block is missing, broken or declined get
        one repair (shared by the node) or one separate rephrase, then their prior-only fallback. Private items
        are never rendered here."""
        ctx = NodeContext.coerce(ctx)
        actors = [ActorSpec.coerce(a) for a in actors]
        reactions = [ReactionSpec.coerce(r) for r in reactions]
        hazards = [HazardSpec.coerce(h) for h in hazards]
        schema = prompts.node_output_schema(actors, reactions, hazards)
        system = prompts.actor_system(schema, ctx, max([a.custom_cap for a in actors] or [prompts.CUSTOM_CAP]))
        res = NodeResult()
        payload = None
        for rephrase in (False, True):
            user = prompts.node_user(ctx, actors, reactions, hazards, variant=variant)
            if rephrase:
                user = f"{prompts.REPHRASE}\n\n{user}"
                res.flags.append("rephrased")
                with self._lock:
                    self.stats["rephrases"] += 1
            o = self._obtain("node", system, user, schema, effort_key="node", timeout_key="node", sample=sample,
                             purpose="rephrase" if rephrase else "call")
            res.calls += o.calls
            res.report += o.notes
            if o.repaired:
                res.flags.append("repaired")
            declined = o.payload is not None and o.payload.get("status") == "decline"
            refused = o.payload is None and o.failure == "refusal"
            if (declined or refused) and not rephrase and self.settings.rephrase:
                res.report.append("整个节点被拒绝，换一种说法重问一次")
                continue
            payload = o.payload if not declined else None
            failure = o.failure if o.payload is None else ("refusal" if declined else None)
            break
        allowed = self._allowed(prompts.node_user(ctx, actors, reactions, hazards, variant=variant))
        if payload is None:
            for a in actors:
                res.actors[a.id] = self._fallback(a, failure, o.problems)
                res.report += res.actors[a.id].report
            res.flags += ["fallback", "prior_only"] + (["budget"] if failure == "budget" else [])
            return res
        blocks = payload.get("actors") if isinstance(payload.get("actors"), dict) else {}
        unknown = sorted(set(blocks) - {a.id for a in actors})
        if unknown:
            res.report.append(f"回答里有不在本节点的行为者 {'、'.join(unknown)}，已删除")
        answers = {a.id: guard.guard_actor(blocks.get(a.id), a, allowed=allowed) for a in actors}
        # a missing block is not repaired: the repair call has no state and must not invent one (PROMPTS §2.8)
        missing = [a.id for a in actors if not isinstance(blocks.get(a.id), dict)]
        for aid in missing:
            answers[aid] = self._fallback(next(a for a in actors if a.id == aid), "invalid_output",
                                          ["回答中缺少该行为者"])
        broken = {aid: ans.problems for aid, ans in answers.items() if ans.problems and aid not in missing}
        if broken and not o.repaired:
            problems = [p for ps in broken.values() for p in ps]
            fixed, more = self.repair("node", schema, o.text or "", problems, sample)
            res.calls += more.calls
            res.report += more.notes
            res.flags.append("repaired")
            if fixed is not None:
                fixed_blocks = fixed.get("actors") if isinstance(fixed.get("actors"), dict) else {}
                for a in actors:
                    if a.id in broken:
                        again = guard.guard_actor(fixed_blocks.get(a.id), a, allowed=allowed)
                        if not again.problems:
                            answers[a.id] = again
                payload = {**payload, **{k: fixed[k] for k in ("reactions", "hazard_tilts") if k in fixed
                                         and not payload.get(k)}}
        for a in actors:
            ans = answers[a.id]
            if ans.declined and self.settings.rephrase:
                res.report += ans.report + [f"{a.id}：拒绝评估，用单独调用换一种说法重问一次"]
                sub = self._rephrase_actor(ctx, a, None, sample, variant, "actor")
                res.calls += sub.calls
                res.report += sub.report
                ans = sub.answer
            elif ans.problems:
                ans = self._fallback(a, "invalid_output", ans.problems, report=ans.report)
                res.report += ans.report
            else:
                res.report += ans.report
            res.actors[a.id] = ans
        res.reactions = guard.guard_reactions(payload.get("reactions"), reactions, res.report)
        res.hazards = guard.guard_hazards(payload.get("hazard_tilts"), hazards, res.report)
        if any(ans.fallback for ans in res.actors.values()):
            res.flags.append("fallback")
        res.flags = list(dict.fromkeys(res.flags))
        return res

    # ------------------------------------------------------------ S: the consequence synthesizer

    def synth(self, req, *, sample: int = 0) -> SynthResult:
        """One call for the node's gated clusters. Clusters the call cannot settle (missing, declined, broken
        geometry, a failed call) are "default": the engine uses its default partition and the flag
        该节点后果未经大模型判断. Mechanism eligibility, lag discipline, direction, signatures, precision weights,
        joint rarity and the probability calculus are synth.py's."""
        req = req if isinstance(req, SynthRequest) else SynthRequest(**dict(req))
        schema = prompts.synth_output_schema(req)
        system = prompts.synth_system(schema, req.fin_axis, req.funding_axis)
        user = prompts.synth_user(req)
        res = SynthResult()
        o = self._obtain("synth", system, user, schema, effort_key="synth", timeout_key="synth", sample=sample)
        res.calls += o.calls
        res.report += o.notes
        if o.repaired:
            res.flags.append("repaired")
        if o.payload is None or o.payload.get("status") == "decline":
            why = ("拒绝分析" if o.payload is not None else FAILURE_ZH.get(o.failure or "", o.failure or "输出无效"))
            for c in req.clusters:
                res.clusters[c.id] = ClusterAnswer(cluster_id=c.id, status="default", reason=why,
                                                   declined=o.payload is not None,
                                                   report=[f"{c.id}：该节点后果未经大模型判断（{why}）"])
            res.report += [f"该节点后果未经大模型判断（{why}）"]
            res.flags += ["synth_default"] + (["budget"] if o.failure == "budget" else [])
            return res
        allowed = self._allowed(user)
        clusters = guard.guard_synth(o.payload, req, allowed=allowed)
        broken = {cid: c.problems for cid, c in clusters.items() if c.problems}
        if broken and not o.repaired:
            fixed, more = self.repair("synth", schema, o.text or "", [p for ps in broken.values() for p in ps], sample)
            res.calls += more.calls
            res.report += more.notes
            res.flags.append("repaired")
            if fixed is not None:
                again = guard.guard_synth(fixed, req, allowed=allowed)
                for cid in broken:
                    if not again[cid].problems:
                        clusters[cid] = again[cid]
        for cid, c in clusters.items():
            if c.problems:
                clusters[cid] = ClusterAnswer(cluster_id=cid, status="default", reason="输出无效",
                                              report=c.report + [f"{cid}：输出无效（{'；'.join(c.problems)[:80]}），"
                                                                 "用默认分区"])
            res.report += clusters[cid].report
        res.clusters = clusters
        if any(c.status == "default" for c in clusters.values()):
            res.flags.append("synth_default")
        res.flags = list(dict.fromkeys(res.flags))
        return res

    # ------------------------------------------------------------ P4: the opening narrative

    def root(self, ctx, narrative: str, settable=()) -> RootResult:
        """The user's narrative -> h0, forced first-quarter fields (each with the user's own words), user views
        kept apart from facts, unmapped content. The user reviews it before the tree spends calls."""
        ctx = NodeContext.coerce(ctx)
        text = str(narrative or "").strip().encode("utf-8", "replace").decode("utf-8")
        res = RootResult()
        if len(text) > prompts.MAX_NARRATIVE_CHARS:
            res.report.append(f"用户叙事过长（{len(text)} 字），已截断到 {prompts.MAX_NARRATIVE_CHARS} 字")
            text = text[:prompts.MAX_NARRATIVE_CHARS]
        levers = [Lever.coerce(f) for f in settable]
        schema = prompts.root_output_schema(levers)
        system, user = prompts.root_system(schema, ctx), prompts.root_user(ctx, text, levers)
        o = self._obtain("root", system, user, schema, effort_key="root", timeout_key="root", sample=0)
        res.calls, res.report = list(o.calls), res.report + o.notes
        if o.payload is None:
            res.error = FAILURE_ZH.get(o.failure or "", o.failure or "输出无效")
            return res
        g = guard.guard_root(o.payload, text, levers, allowed=self._allowed(user))
        res.report += g["report"]
        res.ok, res.h0, res.forced = g["ok"], g["h0"], g["forced"]
        res.user_views, res.unmapped = g["user_views"], g["unmapped"]
        if not g["ok"]:
            res.error = g["report"][0] if g["report"] else "起点翻译被拒绝"
        return res

    # ------------------------------------------------------------ P7 story, P6 leak probe

    def _text_call(self, kind: str, system: str, user: str, schema: dict, effort_key: str) -> TextResult:
        o = self._obtain(kind, system, user, schema, effort_key=effort_key, timeout_key=kind, sample=0)
        res = TextResult(calls=list(o.calls), report=list(o.notes))
        if o.payload is None:
            res.error = FAILURE_ZH.get(o.failure or "", o.failure or "输出无效")
            return res
        res.ok, res.payload = True, o.payload
        return res

    def story(self, path_lines, *, quarter_from: str = "", quarter_to: str = "") -> TextResult:
        """补写叙事 for one leaf (on demand). The story's text is scrubbed of price sentences."""
        schema = prompts.story_output_schema()
        user = prompts.story_user(path_lines, quarter_from, quarter_to)
        res = self._text_call("story", prompts.story_system(schema), user, schema, "story")
        if res.ok:
            allowed = self._allowed(user)
            p = res.payload
            res.payload = {"headline": guard.clean_text(p.get("headline"), 30, allowed, res.report, "叙事标题"),
                           "story": guard.clean_text(p.get("story"), 600, allowed, res.report, "叙事"),
                           "facts_used": [f for f in (p.get("facts_used") or []) if isinstance(f, dict)][:20]}
        return res

    def leak_probe(self, anonymized_state: str) -> TextResult:
        """Evaluation only: can the model date an anonymised state (P6)?"""
        schema = prompts.leak_probe_schema()
        return self._text_call("probe", prompts.probe_system(schema), prompts.probe_user(anonymized_state),
                               schema, "probe")
