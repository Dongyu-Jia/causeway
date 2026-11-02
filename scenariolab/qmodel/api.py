"""/api/q/*: the quarterly stack's endpoints (SIM_DESIGN §3.9; 试运行). An APIRouter that server.py includes before
the static mount. Chinese error details, as the rest of the server.

    GET  /api/q/card                 the model card (card.json) with today's start-of-quarter states from the panel
    GET  /api/q/card/view            the 变量卡片 tab: readable rows, today's value, percentile, series, evidence
    POST /api/q/run                  {qtable | scenario, narrative?, source?, n_paths, seed, mode, continuation, cma,
                                      validation_mode} -> {id, status}
    GET  /api/q/run/{id}             {id, status: queued | running | done | error, result | error, elapsed_s}
    POST /api/q/validate             {qtable, narrative?, mode?, source?} -> the validation report (qtable.validate)
    POST /api/q/translate/estimate   {narrative, mode?, n_quarters?} -> calls, tokens, minutes, cached (no call)
    POST /api/q/translate            {narrative, mode?, n_quarters?, refresh?, confirm?} -> the translation
    GET  /api/q/valuation            the valuation block on the quarterly panel (valview.py: states, identity, caveats)
    GET  /api/q/scenarios            the 13 built-in scenarios as reviewed QTables (scenarios/qtables/)
    GET  /api/q/scenarios/{id}       one of them: the QTable, its source text, the conversion record, the report
    POST /api/q/phase_spec           {qtable} -> the monthly engine's phase spec of the same QTable (to_phase_spec)

/api/q/run reads the QTable through qtable.py synchronously (strict: a 400 names the first error or refusal; the
warnings, the R1-R5 gaps and the anchors travel with the result), then runs the quarterly stack in a background
thread (at most config_q.API_WORKERS at once; the first run fits the forward stack, a few seconds). The quotes are
checked against `narrative` when the request carries one (source "translator", the default), against the built-in
scenario's own text for `scenario`, and not at all for a hand-written table (W-QUOTE); source "user" marks the
user's own cells. mode "path" runs the scenario's quarters at once (two-sided conditioning) plus `continuation`
quarters without statements; mode "step" runs its first quarter only (one-sided). An identical request returns the
same job.

/api/q/translate is the only route here that can call a model: MSL_LLM_BACKEND claude_code spends the user's plan,
so it answers only this machine's own pages and only with confirm: true after the estimate (the mock backend and a
cached answer need no confirmation). A usage limit, a missing login or CLI is a 503 with "pause" and "reset_at".
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections import OrderedDict

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, Response

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


@router.get("/card/view")
def card_view():
    """The 变量卡片 tab (P8a, SIM_DESIGN §10.2): every card row in readable Chinese with today's value, its
    percentile, the quarterly series for a sparkline and the links to its evidence (cardview.py)."""
    from . import cardview
    try:
        return _json(cardview.view())
    except OSError as err:
        raise HTTPException(404, f"模型卡片 card.json 不存在：请运行 python tools/build_model_card.py（{err}）") from err


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


def _source(body: dict) -> tuple[dict, str | None, str]:
    """(qtable, narrative, source) of a request: a built-in scenario by id, or the posted QTable with its text."""
    from . import qtable as QT
    sid = body.get("scenario")
    if sid is not None:
        try:
            env = QT.builtin(str(sid))
        except QT.QTableError as err:
            raise HTTPException(404, str(err)) from err
        return env["qtable"], env["text"], "builtin"
    qt = body.get("qtable")
    if not isinstance(qt, dict):
        raise HTTPException(400, "qtable 必须是 QTable/1 对象（或用 scenario 指定内置情景）")
    narrative = body.get("narrative")
    if narrative is not None and not isinstance(narrative, str):
        raise HTTPException(400, "narrative 必须是字符串")
    source = body.get("source", "translator")
    if source not in QT.SOURCES:
        raise HTTPException(400, f"source 必须是 {' / '.join(QT.SOURCES)} 之一，收到「{source}」")
    return qt, narrative, source


def parse_request(body: dict) -> dict:
    """The run request with its QTable read into rollout inputs; HTTPException(400) with a Chinese detail."""
    from . import card as qcard
    from . import panel as qpanel
    from . import rollout as R
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是一个 JSON 对象（{ … }）")
    qt, narrative, source = _source(body)
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
        read = R.from_qtable(qt, None, vmode, Q=Q, train_end=te, x_fields=qcard.x_fields(qcard.load_card()),
                             narrative=narrative, source=source)
    except ValueError as err:                         # RolloutError and the valuation block's ValueError
        raise HTTPException(400, f"QTable 有错误：{err}") from err
    H = read["H"]
    cont = _int(body, "continuation", 0, 0, cq.ROLLOUT_MAX_QUARTERS - H) if mode == "path" else 0
    key = hashlib.sha256(json.dumps({"qtable": qt, "narrative": narrative, "source": source, "n_paths": n_paths,
                                     "seed": seed, "mode": mode, "cma": cma, "continuation": cont,
                                     "validation_mode": vmode}, sort_keys=True,
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
    from . import qreadout                            # the scenario detail's extra readouts (P8)
    out.update(qreadout.extras(stack, roll))
    out["qtable_notes"] = read["notes"]
    out["validation"] = report_summary(read.get("report"))
    out["anchors"] = read.get("anchors") or {}
    clean = (read.get("report") or {}).get("qtable")
    if clean:                                         # what the QTable stated per quarter, in words (qview, P8)
        from . import qview
        out["cells"] = qview.cells(clean, 1 if req["mode"] == "step" else read["H"])
        out["channels"] = qview.channels()
    out["request"] = {k: req[k] for k in ("mode", "n_paths", "seed", "cma", "continuation", "validation_mode")}
    if req["mode"] == "step" and read["H"] > 1:
        out["qtable_notes"] = out["qtable_notes"] + [{"code": "step", "text": f"逐季模式只模拟第 1 季："
                                                                              f"第 2–{read['H']} 季的陈述未作输入"}]
    out["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return out


def _run_job(jid: str, req: dict) -> None:
    with _GUARD:
        _JOBS[jid]["status"] = "running"
        _JOBS[jid]["started"] = time.time()
    try:
        with _SLOTS:
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


def report_summary(report: dict | None) -> dict | None:
    """The validation report without the clean QTable (the run already carries what it ran)."""
    if not report:
        return None
    return {k: v for k, v in report.items() if k != "qtable"}


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


# ---------------------------------------------------------------- the scenario layer (P6): validate, translate, built-ins

def _mode(body: dict, key: str = "mode") -> str:
    m = body.get(key, "production")
    if m not in ("production", "test"):
        raise HTTPException(400, f"{key} 必须是 production 或 test，收到「{m}」")
    return m


def _narrative(body: dict) -> str:
    from ..llm import translator as tr
    text = body.get("narrative")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "narrative 不能为空")
    if len(text) > tr.MAX_NARRATIVE_CHARS:
        raise HTTPException(400, f"叙事过长（{len(text)} 字），请控制在 {tr.MAX_NARRATIVE_CHARS} 字以内")
    return text.strip()


def _quarters(body: dict) -> int | None:
    if body.get("n_quarters") is None:
        return None
    return _int(body, "n_quarters", 4, 1, cq.QTABLE_MAX_QUARTERS)


@router.post("/validate")
def validate(body: dict = Body(...)):
    """The validation report of a QTable (§7.3): errors, refused entries, warnings, R1-R5 gaps and the clean form."""
    from . import qtable as QT
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是一个 JSON 对象（{ … }）")
    qt, narrative, source = _source(body)
    try:
        return _json(QT.validate(qt, narrative, mode=_mode(body), source=source))
    except QT.QTableError as err:
        raise HTTPException(400, str(err)) from err


@router.post("/translate/estimate")
def translate_estimate(body: dict = Body(...)):
    """What /api/q/translate would cost (calls, tokens, seconds, cached): no model call."""
    from ..llm import qtranslator as QX
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是一个 JSON 对象（{ … }）")
    return _json(QX.estimate(_narrative(body), mode=_mode(body), n_quarters=_quarters(body)))


@router.post("/translate")
def translate(request: Request, body: dict = Body(...)):
    """Narrative -> QTable/1 (qtranslator.translate): the clean QTable, the validation report, the status (ok /
    repaired / fallback) and the call's metadata. A real backend needs this machine's own page and confirm: true."""
    from .. import server
    from ..llm import qtranslator as QX
    from ..llm import translator as tr
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是一个 JSON 对象（{ … }）")
    narrative, mode, nq = _narrative(body), _mode(body), _quarters(body)
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if not server.local_host(request) or not server.same_origin(request) or ctype != "application/json":
        raise HTTPException(403, "翻译只接受本机页面（http://127.0.0.1 或 http://localhost）发来的 JSON 请求："
                                 "Claude 订阅额度只供你自己在本机使用。")
    est = QX.estimate(narrative, mode=mode, n_quarters=nq)
    refresh = body.get("refresh") is True
    if (est["needs_confirmation"] or (refresh and est["backend"] != "mock")) and body.get("confirm") is not True:
        return JSONResponse(status_code=409, content={"detail": QX.E_CONFIRM.format(calls=2), "estimate": est})
    try:
        return _json(QX.translate(narrative, mode=mode, n_quarters=nq, refresh=refresh))
    except QX.QTablePause as err:
        return JSONResponse(status_code=503, content={"detail": str(err), "pause": err.kind,
                                                      "reset_at": err.reset_at})
    except tr.TranslatorError as err:
        raise HTTPException(400, str(err)) from err


@router.get("/valuation")
def valuation_view():
    """The valuation block on the quarterly panel (valview.view): today's states with percentiles, the CAPE history's
    percentiles, the identity decomposition by window, the re-rating drift by train end and the caveats (P8)."""
    from . import panel as qpanel
    from . import valview
    try:
        Q = qpanel.load_panel()
        info = qpanel.load_meta().get("info") or {}
    except OSError as err:
        raise HTTPException(404, "季度面板还没有生成：请运行 python -m scenariolab.qmodel.panel") from err
    return _json(valview.view(Q, info))


@router.get("/scenarios")
def scenarios():
    """The built-in scenarios as reviewed QTables (tools/qtable_from_yaml.py), each with its validation counts."""
    from . import qtable as QT
    out = []
    for sid in QT.builtin_ids():
        env = QT.builtin(sid)
        rep = QT.validate(env["qtable"], env["text"], mode="production", source="builtin")
        out.append({"id": sid, "name": env.get("name"), "n_quarters": env["qtable"]["n_quarters"],
                    "ok": rep["ok"], "counts": rep["counts"], "review": len(env.get("review") or []),
                    "dropped_views": sum(1 for c in env.get("conversion") or [] if c.get("status") == "dropped")})
    return _json({"label": cq.TRIAL_LABEL, "scenarios": out})


@router.get("/scenarios/{sid}")
def scenario(sid: str):
    from . import qtable as QT
    try:
        env = QT.builtin(sid)
    except QT.QTableError as err:
        raise HTTPException(404, str(err)) from err
    rep = QT.validate(env["qtable"], env["text"], mode="production", source="builtin")
    return _json({**env, "validation": report_summary(rep), "label": cq.TRIAL_LABEL})


@router.post("/phase_spec")
def phase_spec(body: dict = Body(...)):
    """The monthly engine's phase spec of a QTable (§7.4): the views it became and the fields it dropped."""
    from . import panel as qpanel
    from . import qtable as QT
    from . import rollout as R
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是一个 JSON 对象（{ … }）")
    qt, narrative, _ = _source(body)
    try:
        Q = qpanel.load_panel()
    except OSError as err:
        raise HTTPException(503, "季度面板还没有生成：请运行 python -m scenariolab.qmodel.panel") from err
    try:
        spec, rep = QT.to_phase_spec(qt, Q=Q, train_end=R.last_data_quarter(Q), narrative=narrative or "",
                                     mode=_mode(body))
    except QT.QTableError as err:
        raise HTTPException(400, f"QTable 有错误：{err}") from err
    return _json({"spec": spec, "report": rep})
