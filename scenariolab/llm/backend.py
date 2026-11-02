"""Which model answers: MSL_LLM_BACKEND, for the agent tree (and the translator's claude_code / mock paths).

    claude_code   the default: the local Claude Code CLI on the user's own subscription login (claude_code.py)
    mock          MockClaude (mock.py): deterministic, schema-valid answers derived from the prompt; no process,
                  no usage. Every test and evaluation runs on it (tests/conftest.py makes it the default).

The translator also knows "api" (the Anthropic SDK, only when set explicitly); the agent tree does not use it.
run() has claude_code.run's contract for both backends: (text, metadata) or ClaudeCodeError with a kind.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

from ..env import load_dotenv
from . import claude_code, mock

BACKENDS = ("claude_code", "mock")
DEFAULT_BACKEND = "claude_code"
DEFAULT_MODEL = "claude-opus-5-5"                 # the translator's DEFAULT_MODEL
CAPS_TTL_S = 300.0


def normalize(raw) -> str:
    """"Claude-Code " -> "claude_code"; empty -> the default."""
    return str(raw or "").strip().lower().replace("-", "_") or DEFAULT_BACKEND


def backend_name() -> str:
    load_dotenv()
    return normalize(os.environ.get("MSL_LLM_BACKEND", ""))


def model_id() -> str:
    load_dotenv()
    return os.environ.get("MSL_LLM_MODEL", "").strip() or DEFAULT_MODEL


def timeout_ceiling() -> float:
    """MSL_LLM_TIMEOUT (600 s by default): no call waits longer, whatever its own timeout."""
    load_dotenv()
    try:
        v = float(os.environ.get("MSL_LLM_TIMEOUT", "").strip() or claude_code.DEFAULT_TIMEOUT_S)
    except ValueError:
        return claude_code.DEFAULT_TIMEOUT_S
    return v if v >= 1.0 and v == v and v != float("inf") else claude_code.DEFAULT_TIMEOUT_S


# ---------------------------------------------------------------- one call budget for the whole process
# Every model call of this process (agent runs, snapshot builds, on-demand expansions, leaf stories and the
# translator) takes one of MSL_LLM_CONCURRENCY slots (default 3, 1–6; AGENT_DESIGN §11.2: k = 3, server-owned), so
# a run plus expansions can never have more calls in flight than that.
_CLI_SLOTS: dict[int, threading.BoundedSemaphore] = {}
_CLI_SLOTS_LOCK = threading.Lock()


def cli_concurrency() -> int:
    load_dotenv()
    try:
        n = int(os.environ.get("MSL_LLM_CONCURRENCY", "").strip() or 3)
    except ValueError:
        n = 3
    return max(1, min(n, 6))


@contextmanager
def cli_slot(cancel: threading.Event | None = None, poll_s: float = 0.25, timeout: float | None = None):
    """Hold one of the process-wide call slots while a call runs; a graceful cancel stops the wait, and `timeout`
    (seconds) bounds it (ClaudeCodeError "rate": every slot stayed busy)."""
    n = cli_concurrency()
    with _CLI_SLOTS_LOCK:
        sem = _CLI_SLOTS.setdefault(n, threading.BoundedSemaphore(n))
    t_end = None if timeout is None else time.monotonic() + float(timeout)
    while not sem.acquire(timeout=poll_s):
        if cancel is not None and cancel.is_set():
            raise claude_code.ClaudeCodeError("推演已取消", "cancelled")
        if t_end is not None and time.monotonic() > t_end:
            raise claude_code.ClaudeCodeError(f"本机已有 {n} 个大模型调用在进行（MSL_LLM_CONCURRENCY={n}），"
                                              f"等了 {timeout:g} 秒仍没有空位，请稍后重试。", "rate")
    try:
        yield
    finally:
        sem.release()


def unknown_message(name: str) -> str:
    return f"MSL_LLM_BACKEND={name} 不认识：智能体推演可用 claude_code（默认，本机 Claude Code）或 mock（模拟后端，不调用模型）。"


def run(system: str, user: str, model: str, timeout: float = claude_code.DEFAULT_TIMEOUT_S, *,
        effort: str | None = None, safe_mode: bool = False, cancel: threading.Event | None = None,
        backend: str | None = None) -> tuple[str, dict]:
    """One call on `backend` (default MSL_LLM_BACKEND)."""
    name = normalize(backend) if backend else backend_name()
    if name == "mock":
        return mock.run(system, user, model, timeout, effort=effort, safe_mode=safe_mode, cancel=cancel)
    if name == "claude_code":
        return claude_code.run(system, user, model, timeout, effort=effort, safe_mode=safe_mode, cancel=cancel)
    raise claude_code.ClaudeCodeError(unknown_message(name), "cli_missing")


def status(model: str | None = None, backend: str | None = None) -> dict:
    """{"backend", "label", "available", "model", "version", "auth", "reason", …}: local checks only, no model call."""
    name = normalize(backend) if backend else backend_name()
    model = model or model_id()
    if name == "mock":
        return mock.status(model)
    if name == "claude_code":
        return claude_code.status(model)
    return {"backend": name, "label": name, "available": False, "model": model, "version": None,
            "auth": claude_code.AUTH_NONE, "reason": unknown_message(name)}


_CAPS: dict = {}
_CAPS_LOCK = threading.Lock()


def capabilities(backend: str | None = None, force: bool = False) -> dict:
    """{"backend", "cli_version", "flags": {"effort", "effort_levels", "safe_mode"}} — the CLI version (for cache
    keys) and the optional flags `claude --help` lists; reused for CAPS_TTL_S. No model call."""
    name = normalize(backend) if backend else backend_name()
    key = (name, os.environ.get("MSL_CLAUDE_BIN", ""))
    with _CAPS_LOCK:
        hit = _CAPS.get(key)
        if hit and not force and time.monotonic() - hit[0] < CAPS_TTL_S:
            return dict(hit[1])
    if name == "mock":
        caps = {"backend": "mock", "cli_version": mock.MOCK_VERSION,
                "flags": {"effort": True, "effort_levels": list(claude_code.EFFORT_LEVELS), "safe_mode": True}}
    elif name == "claude_code":
        binary, problem = claude_code.resolve_binary()
        version = None
        if not problem:
            st = claude_code.status(model_id())
            version = st.get("version")
        flags = claude_code.supported_flags(binary) if not problem else {"effort": False, "effort_levels": [],
                                                                           "safe_mode": False}
        caps = {"backend": "claude_code", "cli_version": version,
                "flags": {k: flags.get(k) for k in ("effort", "effort_levels", "safe_mode")}}
    else:
        caps = {"backend": name, "cli_version": None, "flags": {"effort": False, "effort_levels": [], "safe_mode": False}}
    with _CAPS_LOCK:
        _CAPS[key] = (time.monotonic(), dict(caps))
    return caps


def clear_caches() -> None:
    with _CAPS_LOCK:
        _CAPS.clear()
