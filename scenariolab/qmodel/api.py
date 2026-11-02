"""/api/q/*: the quarterly stack's endpoints (SIM_DESIGN §3.9; 试运行). An APIRouter that server.py includes before
the static mount. Chinese error details, as the rest of the server.

    GET  /api/q/card            the model card (card.json) with today's start-of-quarter states from the panel
    POST /api/q/run             {qtable, n_paths, seed, mode, continuation, cma, validation_mode} -> {id, status}
    GET  /api/q/run/{id}        {id, status: queued | running | done | error, result | error, elapsed_s}

/api/q/run reads the QTable synchronously (a 400 names the first problem; rollout.from_qtable is the structural
reading until P6's qtable.py validates quotes and refusals), then runs the quarterly stack in a background thread
(at most config_q.API_WORKERS at once; the first run fits the forward stack, a few seconds). mode "path" runs the
scenario's quarters at once (two-sided conditioning) plus `continuation` quarters without statements; mode "step"
runs its first quarter only (one-sided). An identical request returns the same job. Nothing here calls an LLM.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections import OrderedDict

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response

from .. import jsonio
from . import config_q as cq

router = APIRouter(prefix="/api/q", tags=["quarterly"])

_JOBS: OrderedDict = OrderedDict()
_KEYS: dict = {}
_GUARD = threading.Lock()
_SLOTS = threading.BoundedSemaphore(cq.API_WORKERS)


def _json(obj) -> Response:
    return Response(content=jsonio.dumps(obj), media_type="application/json")


# ---------------------------------------------------------------- the card

@router.get("/card")
def card():
    """The model card plus today's values of the start-of-quarter states (the quarterly panel's last row)."""
    from . import card as qcard
    from . import panel as qpanel
    try:
        c = qcard.load_card()
    except OSError as err:
        raise HTTPException(404, f"模型卡片 card.json 不存在：请运行 python tools/build_model_card.py（{err}）") from err
    out = dict(c)
    out["label"] = cq.TRIAL_LABEL
    try:
        Q = qpanel.load_panel()
        meta = qpanel.load_meta()["info"]
        today = Q.index[-1]
        cols = {r["id"]: (r.get("panel") or []) for r in qcard.rows(c) if r.get("group") == "s"}
        vals = {}
        for rid, pcs in cols.items():
            for col in pcs:
                if col in Q.columns:
                    v = Q.at[today, col]
                    vals[col] = None if v != v else float(v)
        out["today"] = {"quarter": str(today), "data_end_month": meta.get("data_end_month"), "states": vals}
    except (OSError, KeyError, ValueError):
        out["today"] = None                        # the panel is not built: the card alone
    return _json(out)


# ---------------------------------------------------------------- runs

def _int(body: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = body.get(key, default)
    if isinstance(v, bool):
        raise HTTPException(400, f"{key} 必须是整数")
    if isinstance(v, float) and math.isfinite(v) and v == int(v):
        v = int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        v = int(v.strip())
    if not isinstance(v, int):
        raise HTTPException(400, f"{key} 必须是整数，收到「{v}」")
    if not lo <= v <= hi:
        raise HTTPException(400, f"{key} 必须在 {lo}..{hi} 之间，收到 {v}")
    return v


def parse_request(body: dict) -> dict:
    """The run request with its QTable read into rollout inputs; HTTPException(400) with a Chinese detail."""
    from . import card as qcard
    from . import panel as qpanel
    from . import rollout as R
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是一个 JSON 对象（{ … }）")
    qt = body.get("qtable")
    if not isinstance(qt, dict):
        raise HTTPException(400, "qtable 必须是 QTable/1 对象")
    mode = body.get("mode", "path")
    if mode not in ("path", "step"):
        raise HTTPException(400, f"mode 必须是 path 或 step，收到「{mode}」")
    vmode = body.get("validation_mode", "production")
    if vmode not in ("production", "test"):
        raise HTTPException(400, f"validation_mode 必须是 production 或 test，收到「{vmode}」")
    cma = body.get("cma", cq.CMA_FORWARD)
    if cma not in cq.CMA_ARMS:
        raise HTTPException(400, f"cma 必须是 {list(cq.CMA_ARMS)} 之一，收到「{cma}」")
    n_paths = _int(body, "n_paths", cq.API_DEFAULT_PATHS, 100, cq.API_MAX_PATHS)
    seed = _int(body, "seed", 0, 0, 2 ** 31 - 1)
    try:
        Q = qpanel.load_panel()
    except OSError as err:
        raise HTTPException(503, "季度面板还没有生成：请运行 python -m scenariolab.qmodel.panel") from err
    te = R.last_data_quarter(Q)
    try:
        read = R.from_qtable(qt, None, vmode, Q=Q, train_end=te, x_fields=qcard.x_fields(qcard.load_card()))
    except ValueError as err:                         # RolloutError and the valuation block's ValueError
        raise HTTPException(400, f"QTable 有错误：{err}") from err
    H = read["H"]
    cont = _int(body, "continuation", 0, 0, cq.ROLLOUT_MAX_QUARTERS - H) if mode == "path" else 0
    pq = n_paths * ((H + cont) if mode == "path" else 1)
    if pq > cq.API_MAX_PATH_QUARTERS:                 # memory and time grow with paths x quarters
        raise HTTPException(400, f"路径数 × 季度数 = {n_paths} × {H + cont} = {pq}，超过上限 "
                                 f"{cq.API_MAX_PATH_QUARTERS}：请减少路径数或续推季度数")
    key = hashlib.sha256(json.dumps({"qtable": qt, "n_paths": n_paths, "seed": seed, "mode": mode, "cma": cma,
                                     "continuation": cont, "validation_mode": vmode}, sort_keys=True,
                                    ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    return {"read": read, "mode": mode, "n_paths": n_paths, "seed": seed, "cma": cma, "continuation": cont,
            "validation_mode": vmode, "key": key}


def execute(req: dict) -> dict:
    """Run one parsed request on the forward stack (the job body; also called directly by tests)."""
    from . import rollout as R
    read = req["read"]
    stack = R.get_stack()
    t0 = time.perf_counter()
    if req["mode"] == "step":
        stmts = [s for s in read["statements"] if max(int(q) for q in s.quarters) <= 1]
        vst = [s for s in read["valuation"] if max(s.quarters) <= 1]
        H, cont = 1, 0
    else:
        stmts, vst, H, cont = read["statements"], read["valuation"], read["H"], req["continuation"]
    roll = R.run_path(stack, stmts, vst, H=H, n_paths=req["n_paths"], seed=req["seed"], continuation=cont,
                      cma=req["cma"], cycle_odds=read.get("cycle_odds"), mode=req["validation_mode"])
    out = R.summary(roll)
    out["qtable_notes"] = read["notes"]
    out["request"] = {k: req[k] for k in ("mode", "n_paths", "seed", "cma", "continuation", "validation_mode")}
    if req["mode"] == "step" and read["H"] > 1:
        out["qtable_notes"] = out["qtable_notes"] + [{"code": "step", "text": f"逐季模式只模拟第 1 季："
                                                                              f"第 2–{read['H']} 季的陈述未作输入"}]
    out["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return out


def _run_job(jid: str, req: dict) -> None:
    try:
        with _SLOTS:                                  # queued until a slot is free; "running" only once it has one
            with _GUARD:
                job = _JOBS.get(jid)
                if job is None:                       # evicted while queued (API_MAX_JOBS): nobody can read it
                    return
                job["status"] = "running"
                job["started"] = time.time()
            res = execute(req)
        payload, err = jsonio.dumps(res), None
    except Exception as exc:  # noqa: BLE001 - reported in the job, in Chinese
        payload, err = None, f"季度模型运行出错：{type(exc).__name__}: {exc}"
    with _GUARD:
        job = _JOBS.get(jid)
        if job is None:
            return
        job.update(status="error" if err else "done", payload=payload, error=err, finished=time.time())
        if err:
            _KEYS.pop(job["key"], None)


def submit(req: dict, *, background: bool = True) -> str:
    """Queue a parsed request; an identical one that has not failed returns its job id."""
    with _GUARD:
        jid = _KEYS.get(req["key"])
        if jid is not None and jid in _JOBS:
            _JOBS.move_to_end(jid)
            return jid
        jid = uuid.uuid4().hex[:12]
        _JOBS[jid] = {"id": jid, "status": "queued", "created": time.time(), "key": req["key"], "payload": None,
                      "error": None, "mode": req["mode"]}
        _KEYS[req["key"]] = jid
        while len(_JOBS) > cq.API_MAX_JOBS:
            old, job = _JOBS.popitem(last=False)
            _KEYS.pop(job["key"], None)
    if background:
        threading.Thread(target=_run_job, args=(jid, req), daemon=True, name=f"qrun-{jid}").start()
    else:
        _run_job(jid, req)
    return jid


@router.post("/run")
def run(body: dict = Body(...)):
    """Start a quarterly-stack run of a QTable (see the module docstring); returns the job id at once."""
    req = parse_request(body)
    jid = submit(req)
    with _GUARD:
        st = _JOBS[jid]["status"]
    return _json({"id": jid, "status": st, "label": cq.TRIAL_LABEL})


@router.get("/run/{jid}")
def run_status(jid: str):
    if not (len(jid) == 12 and all(c in "0123456789abcdef" for c in jid)):
        raise HTTPException(400, f"运行编号格式不对：「{jid}」")
    with _GUARD:
        job = _JOBS.get(jid)
        if job is None:
            raise HTTPException(404, f"找不到运行「{jid}」（可能已过期：只保留最近 {cq.API_MAX_JOBS} 个）")
        job = dict(job)
    end = job.get("finished") or time.time()
    head = {"id": jid, "status": job["status"], "mode": job["mode"], "label": cq.TRIAL_LABEL,
            "elapsed_s": round(end - job.get("started", job["created"]), 2)}
    if job["status"] == "done":
        return Response(content=jsonio.dumps(head)[:-1] + ',"result":' + job["payload"] + "}",
                        media_type="application/json")
    if job["status"] == "error":
        head["error"] = job["error"]
    return _json(head)
