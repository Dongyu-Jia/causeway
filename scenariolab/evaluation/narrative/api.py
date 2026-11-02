"""/api/evaluation/narrative/*: the narrative test's endpoints (SIM_DESIGN §3.9, §8.5, §8.7). Chinese error details.

    GET  /api/evaluation/narrative                    registry summary (no windows), runs, defaults, legend, the
                                                      active job, the H-SEP data status
    GET  /api/evaluation/narrative/runs               {"runs": [...]}: the research run, then mock and real runs
    GET  /api/evaluation/narrative/runs/{run_id}      the run's report.json (report.py), or its status while running
    GET  /api/evaluation/narrative/runs/{run_id}/progress   status, progress, pause, error
    POST /api/evaluation/narrative/estimate           {episodes?, narratives?, arms?, probe?, engines?, backend?}
                                                      -> the estimate with its id (no model call); {run_id}: the
                                                      estimate of that run's own remaining calls (for /resume)
    POST /api/evaluation/narrative/run                {estimate_id, confirm: true, label?} -> {run_id, status}
    POST /api/evaluation/narrative/runs/{run_id}/resume   {estimate_id, confirm: true}: a paused run continues; the
                                                      estimate must be of THIS run's remaining calls (409 otherwise)
    POST /api/evaluation/narrative/runs/{run_id}/cancel

/run never starts without both an estimate id issued in the last 10 minutes (whose plan has not changed) and
confirm: true, whatever the backend: 400 without an estimate id, 409 with needs_confirm and the estimate without
confirm: true, 409 with stale for an old or changed estimate. Every POST answers only this machine's own pages
(the Host names this machine, no cross-site Origin, a JSON body), like the translator and the agent tree: a real run
spends the user's Claude plan. The job runs in the background (jobs.py): one run at a time.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ... import jsonio
from . import common as C

router = APIRouter(prefix="/api/evaluation/narrative", tags=["narrative"])
NOT_LOCAL = ("叙事检验只接受本机页面（http://127.0.0.1 或 http://localhost）发来的 JSON 请求：真实运行会调用本机 Claude Code，"
             "计入你自己的订阅额度。")


def _json(obj, status: int = 200) -> Response:
    return Response(content=jsonio.dumps(obj), media_type="application/json", status_code=status)


def _guard(request: Request) -> None:
    from ... import server
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if not server.local_host(request) or not server.same_origin(request) or ctype != "application/json":
        raise HTTPException(403, NOT_LOCAL)


def _mods():
    """The narrative modules, imported on first use (the server's start-up stays light)."""
    from . import episodes as EP
    from . import jobs as JB
    from . import report as RP
    from . import runner as RN
    return EP, JB, RP, RN


def _sealed() -> None:
    EP = _mods()[0]
    if not EP.sealed():
        raise HTTPException(503, EP.NOT_SEALED)


def _selection(raw) -> list[str]:
    """Registry codes of a request's episode selection (set names, registry codes, or the spec's codes)."""
    from . import key as K
    RN = _mods()[3]
    try:
        return K.resolve(raw if raw is not None else RN.DEFAULT_SETTINGS["episodes"])
    except (KeyError, K.KeyError_) as err:
        raise HTTPException(400, str(err.args[0] if err.args else err)) from err


@router.get("")
def summary():
    from . import sep as SEP
    EP, JB, RP, RN = _mods()
    reg = EP.public_summary() if EP.sealed() else {"sealed": False, "hint": EP.NOT_SEALED}
    return _json({"registry": reg, "runs": RP.runs(), "defaults": RN.DEFAULT_SETTINGS, "legend": RP.legend(),
                  "active": JB.active(), "sep": SEP.status(), "backend": RN.backend_name(),
                  "fresh_s": C.ESTIMATE_FRESH_S, "labels": C.LABELS})


@router.get("/runs")
def runs():
    RP = _mods()[2]
    return _json({"runs": RP.runs()})


@router.get("/runs/{rid}")
def run_report(rid: str):
    RP, RN = _mods()[2:]
    try:
        return _json(RP.load(rid))
    except KeyError as err:
        raise HTTPException(404, str(err.args[0] if err.args else err)) from err
    except RN.RunnerError as err:
        raise HTTPException(400, str(err)) from err


@router.get("/runs/{rid}/progress")
def run_progress(rid: str):
    JB, RN = _mods()[1], _mods()[3]
    try:
        r = RN.read_run(rid)
    except KeyError as err:
        raise HTTPException(404, str(err.args[0] if err.args else err)) from err
    except RN.RunnerError as err:
        raise HTTPException(400, str(err)) from err
    return _json({k: r.get(k) for k in ("id", "status", "progress", "pause", "error", "kind", "backend", "updated")}
                 | {"active": (JB.active() or {}).get("rid") == rid})


@router.post("/estimate")
def estimate(request: Request, body: dict = Body(...)):
    """What a run would cost (no model call); the id it returns is what /run needs within 10 minutes."""
    _guard(request)
    _sealed()
    RN = _mods()[3]
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是一个 JSON 对象（{ … }）")
    if body.get("run_id") is not None:          # what resuming this run would cost: its own remaining calls
        try:
            return _json(RN.estimate_resume(str(body["run_id"])))
        except KeyError as err:
            raise HTTPException(404, str(err.args[0] if err.args else err)) from err
        except (RN.RunnerError, OSError) as err:
            raise HTTPException(400, str(err)) from err
    raw = {k: body.get(k) for k in ("narratives", "arms", "probe", "engines", "backend", "refresh", "seed")}
    raw["episodes"] = _selection(body.get("episodes"))
    try:
        return _json(RN.estimate(raw))
    except RN.RunnerError as err:
        raise HTTPException(400, str(err)) from err


def _confirmed_estimate(body: dict):
    """The fresh estimate of a /run or /resume request, or the refusal response."""
    RN = _mods()[3]
    eid = body.get("estimate_id")
    if not isinstance(eid, str) or not eid:
        raise HTTPException(400, "缺少估算编号 estimate_id：先 POST /api/evaluation/narrative/estimate，看过用量再确认")
    try:
        est = RN.load_estimate(eid)
    except RN.EstimateError as err:
        raise HTTPException(400, str(err)) from err
    if body.get("confirm") is not True:
        return None, JSONResponse(status_code=409, content={
            "detail": "叙事检验要先看估算、再确认（confirm: true）才运行" + ("：这次会调用本机 Claude Code，计入你的订阅额度"
                                                                 if est.get("real_calls") else ""),
            "needs_confirm": True, "estimate": jsonio.clean(est)})
    try:
        est = RN.fresh_estimate(eid)
    except RN.EstimateError as err:
        return None, JSONResponse(status_code=409, content={"detail": str(err), "stale": True})
    except RN.RunnerError as err:
        raise HTTPException(400, str(err)) from err
    return est, None


@router.post("/run")
def run(request: Request, body: dict = Body(...)):
    """Start a run from a fresh, confirmed estimate (see the module docstring)."""
    _guard(request)
    _sealed()
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是一个 JSON 对象（{ … }）")
    est, refusal = _confirmed_estimate(body)
    if refusal is not None:
        return refusal
    if est.get("run_id"):
        return JSONResponse(status_code=409, content={
            "detail": f"这个估算是继续运行 {est['run_id']} 的估算，不能用来开始新的运行：请先估算新的运行", "stale": True})
    _, JB, _, RN = _mods()
    active = JB.active()
    if active:
        return JSONResponse(status_code=409, content={"detail": JB.Busy(active["rid"]).args[0],
                                                      "active_run": active["rid"]})
    label = body.get("label")
    rid = RN.create_run(est, label=str(label)[:80] if isinstance(label, str) and label.strip() else None)
    try:
        status = JB.start(rid)
    except JB.Busy as err:
        return JSONResponse(status_code=409, content={"detail": str(err), "active_run": err.rid})
    return _json({"run_id": rid, "status": status, "estimate_id": est["id"]})


@router.post("/runs/{rid}/resume")
def resume(rid: str, request: Request, body: dict = Body(...)):
    """Continue a paused (or failed) run: what is done is skipped; it needs a fresh, confirmed estimate too."""
    _guard(request)
    _sealed()
    _, JB, _, RN = _mods()
    try:
        r = RN.read_run(rid)
    except KeyError as err:
        raise HTTPException(404, str(err.args[0] if err.args else err)) from err
    if r.get("status") not in ("paused", "failed", "cancelled", "translated"):
        raise HTTPException(409, f"这次运行的状态是 {r.get('status')}：只有暂停、失败或取消的运行可以继续")
    est, refusal = _confirmed_estimate(body if isinstance(body, dict) else {})
    if refusal is not None:
        return refusal
    if est.get("run_id") != rid:                # the confirmation must be of this run's own remaining calls
        return JSONResponse(status_code=409, content={
            "detail": (f"这个估算不是运行 {rid} 剩下的调用：先 POST /api/evaluation/narrative/estimate "
                       f"{{\"run_id\": \"{rid}\"}}，看过还剩多少调用再确认"), "stale": True, "run_id": rid})
    try:
        status = JB.start(rid)
    except JB.Busy as err:
        return JSONResponse(status_code=409, content={"detail": str(err), "active_run": err.rid})
    RN.write_run(rid, resumed_with=est["id"])
    return _json({"run_id": rid, "status": status})


@router.post("/runs/{rid}/cancel")
def cancel(rid: str, request: Request):
    from ... import server
    if not server.local_host(request) or not server.same_origin(request):
        raise HTTPException(403, NOT_LOCAL)
    JB = _mods()[1]
    return _json({"run_id": rid, "cancelled": JB.cancel(rid)})
