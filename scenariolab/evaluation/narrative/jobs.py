"""The narrative test's background job: translate (runner) -> score (score) -> audit -> report, one run at a time.

The translation phase holds the translator's call slots (llm/translator.py, MSL_LLM_CONCURRENCY); the scoring phase
holds one of the server's backtest slots when the server is running (the same queue as /api/backtest/run, §8.5), so a
narrative scoring and a custom backtest never compete for the CPU. A usage limit leaves the run paused (resume by its
id with a fresh estimate and confirm: true); a cancel stops it after the call in flight.
"""
from __future__ import annotations

import threading
import time
import traceback

from . import runner as RN

_LOCK = threading.Lock()
_ACTIVE: dict = {"rid": None, "thread": None, "cancel": None, "started": None}
_LOCAL_SLOT = threading.BoundedSemaphore(1)


class Busy(RuntimeError):
    def __init__(self, rid: str):
        super().__init__(f"已有叙事检验正在运行（{rid}）：同一时间只能运行一个，所有调用共用同一份订阅额度")
        self.rid = rid


def _slot():
    """The server's backtest slots when this process is the server; a local slot otherwise (the CLI)."""
    import sys
    server = sys.modules.get("scenariolab.server")
    slots = getattr(server, "_backtest_slots", None) if server is not None else None
    return slots if slots is not None else _LOCAL_SLOT


def execute(rid: str, *, cancel: threading.Event | None = None, score_kwargs: dict | None = None) -> str:
    """Run (or resume) every phase of a run in this thread; returns the final status."""
    from . import report as RP
    from . import score as SC
    try:
        status = RN.translate_run(rid, cancel=cancel)
        if status != "translated":
            return status
        with _slot():
            SC.score_run(rid, cancel=cancel, **(score_kwargs or {}))
        if cancel is not None and cancel.is_set():
            RN.write_run(rid, status="cancelled")
            return "cancelled"
        RN.write_run(rid, status="reporting")
        RP.build(rid)
        RN.write_run(rid, status="done", error=None, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return "done"
    except Exception as err:  # noqa: BLE001 - recorded in the run, in Chinese
        RN.write_run(rid, status="failed", error=f"叙事检验出错：{type(err).__name__}: {err}",
                     traceback=traceback.format_exc()[-4000:])
        return "failed"


def start(rid: str, *, background: bool = True, score_kwargs: dict | None = None) -> str:
    """Start (or resume) a run's job. Busy when another run is active."""
    with _LOCK:
        t = _ACTIVE.get("thread")
        if t is not None and t.is_alive():
            raise Busy(_ACTIVE["rid"])
        cancel = threading.Event()
        _ACTIVE.update(rid=rid, cancel=cancel, started=time.time())
        RN.write_run(rid, status="queued", error=None)
        if not background:
            _ACTIVE["thread"] = None
    if not background:
        try:
            return execute(rid, cancel=cancel, score_kwargs=score_kwargs)
        finally:
            with _LOCK:
                _ACTIVE.update(rid=None, cancel=None)
    th = threading.Thread(target=execute, args=(rid,), kwargs={"cancel": cancel, "score_kwargs": score_kwargs},
                          daemon=True, name=f"narrative-{rid}")
    with _LOCK:
        _ACTIVE["thread"] = th
    th.start()
    return "queued"


def cancel(rid: str) -> bool:
    with _LOCK:
        if _ACTIVE.get("rid") == rid and _ACTIVE.get("cancel") is not None:
            _ACTIVE["cancel"].set()
            return True
    return False


def active() -> dict | None:
    with _LOCK:
        t = _ACTIVE.get("thread")
        if _ACTIVE.get("rid") and (t is None or t.is_alive()):
            return {"rid": _ACTIVE["rid"], "started": _ACTIVE["started"]}
    return None


def wait(timeout: float | None = None) -> None:
    with _LOCK:
        t = _ACTIVE.get("thread")
    if t is not None:
        t.join(timeout)
