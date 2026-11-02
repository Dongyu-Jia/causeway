"""Local web server: static UI + JSON results + on-demand engine calls.

    python -m scenariolab.server            # http://127.0.0.1:8765
    MSL_PORT=9000 python -m scenariolab.server
"""
from __future__ import annotations

import inspect
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.utils import is_body_allowed_for_status_code
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config, jsonio
from .env import load_dotenv
from .qmodel import api as qmodel_api

load_dotenv()

RESULTS = config.RESULTS_DIR
CUSTOM = RESULTS / "custom"
ID_RE = re.compile(r"^[a-z0-9_]{1,60}$")
RESULT_NAMES = {"eras", "causality", "backtest", "diagnostics", "scenarios", "forecast", "ablation",
                "portfolio_default", "llm_translator", "valuation", "regimes", "regime_forecast", "factor_audit",
                "instrument_meta", "agents_accounts_root", "agents_priors"}
# Which command writes each result file (a 404 names it). Default-build steps can also be rebuilt alone.
BUILD = "python -m scenariolab.build"
BUILD_STEP = {"meta": "meta", "eras": "eras", "causality": "causality", "backtest": "backtest",
              "diagnostics": "diagnostics", "scenarios": "scenarios", "forecast": "forecast",
              "valuation": "valuation", "portfolio_default": "portfolio", "llm_translator": "llm",
              "regimes": "regimes", "factor_audit": "factor_audit", "instrument_meta": "instruments",
              "agents_accounts_root": "agents_accounts", "agents_priors": "agents_accounts"}
CUSTOM_EXTS = (".json", ".npz", ".instr.npz", ".macro.npz")   # a custom scenario's files (portfolio.tickers: the last two)
BUILD_HINT = {
    "ablation": f"{BUILD} --only ablation（引擎消融约需 5 分钟，默认构建不包含这一步）",
    "regime_forecast": f"{BUILD} --only regime_forecast（宏观体制的一年期预测检验约需 2 分钟，默认构建不包含这一步）",
    "factor_audit": f"{BUILD}（或只补这一步：{BUILD} --only factor_audit，缺失因子审计约需 20 秒，要先有 diagnostics.json）",
    "agents_accounts_root": f"{BUILD}（或只补这一步：{BUILD} --only agents_accounts，智能体推演的起点账户：下载 FRED、"
                            "财政部、纽约联储、SEC 等免费数据并缓存，首次约需 1–2 分钟）",
    "agents_priors": f"{BUILD}（或只补这一步：{BUILD} --only agents_accounts，用本地数据重估行为者的历史先验）",
}
NO_CACHE_PREFIXES = ("/js/", "/css/")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}       # the Host a page served by this server sends


def local_host(request) -> bool:
    """Whether the request's Host header names this machine. A page on another site that re-points its DNS
    name at 127.0.0.1 (DNS rebinding) reaches the server same-origin but still sends its own name, so this
    is what keeps other people's pages from spending the user's Claude plan through /api/llm/translate."""
    host = (request.headers.get("host") or "").strip().lower()
    if host.startswith("["):                              # [::1]:8765
        host = host[1:host.find("]")] if "]" in host else ""
    else:
        host = host.rsplit(":", 1)[0]
    return host in LOCAL_HOSTS


def _warm() -> None:
    """Fit the engine, run the engine's unconditional reference simulation (every scenario's regime readout
    compares with it; about 1 s, cached and locked, so a request that arrives meanwhile waits for this copy)
    and load the scenario paths in the background, so first requests are fast."""
    try:
        from .context import get_context
        from .portfolio import service
        from .scenarios import runner
        ctx = get_context()
    except Exception as err:  # noqa: BLE001 - a cold start is only slower
        print(f"[server] warm-up skipped: {err}")
        return
    for name, fn in (("reference run", lambda: runner.reference_simulation(ctx)), ("scenario paths", service.warm)):
        try:
            fn()
        except Exception as err:  # noqa: BLE001 - each part only makes its first request slower
            print(f"[server] warm-up of the {name} skipped: {err}")


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Causeway", version="0.1.0", lifespan=lifespan)


# Starlette's generic English details (an unknown path, a wrong method); our own details are Chinese already.
GENERIC_DETAIL = {"Not Found": "找不到这个地址", "Method Not Allowed": "这个地址不支持该请求方法"}


@app.exception_handler(StarletteHTTPException)
async def http_error(request, err: StarletteHTTPException):
    headers = getattr(err, "headers", None)
    if not is_body_allowed_for_status_code(err.status_code):
        return Response(status_code=err.status_code, headers=headers)
    detail = err.detail
    if isinstance(detail, str) and detail in GENERIC_DETAIL:
        detail = f"{GENERIC_DETAIL[detail]}：{request.url.path}"
    return JSONResponse(status_code=err.status_code, content={"detail": detail}, headers=headers)


@app.exception_handler(RequestValidationError)
async def bad_body(_request, err: RequestValidationError):
    """A body that is not a JSON object (FastAPI's own 422 is English and nested): one Chinese line."""
    first = (err.errors() or [{}])[0]
    kind = str(first.get("type", ""))
    if kind == "json_invalid":
        why = "不是合法的 JSON"
    elif kind in ("missing", "dict_type", "model_attributes_type"):
        why = "必须是一个 JSON 对象（{ … }）"
    else:
        why = str(first.get("msg") or kind or "格式不对")
    return JSONResponse(status_code=400, content={"detail": f"请求体{why}"})


@app.middleware("http")
async def agents_guard(request, call_next):
    """The agents guard before routing and body validation: every state-changing /api/agents request (and the status
    check, which starts CLI processes) from another site or without a JSON body is refused with 403, whatever the
    route would have done with it."""
    path = request.url.path
    if path.startswith("/api/agents/") and (request.method in ("POST", "PUT", "PATCH", "DELETE")
                                            or path == "/api/agents/llm_status"):
        why = agents_refusal(request) if request.method != "GET" else \
            (None if same_origin(request) else NOT_SAME_ORIGIN_AGENTS)
        if why:
            return JSONResponse(status_code=403, content={"detail": why})
    return await call_next(request)


@app.middleware("http")
async def no_cache_ui(request, call_next):
    """The UI's modules and styles must be revalidated on every load (ETag -> 304 when unchanged),
    otherwise a browser keeps running stale tab modules after an edit."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith(NO_CACHE_PREFIXES) or path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def build_hint(name: str) -> str:
    """The command that writes results/<name>.json."""
    if name in BUILD_HINT:
        return BUILD_HINT[name]
    step = BUILD_STEP.get(name)
    return f"{BUILD}（或只补这一步：{BUILD} --only {step}）" if step else BUILD


def _json_file(path, hint: str | None = None) -> Response:
    if not path.exists():
        raise HTTPException(404, f"{path.name} 还没有生成：请在项目根目录运行 {hint or build_hint(path.stem)}")
    return Response(content=path.read_bytes(), media_type="application/json")


def _json(obj) -> Response:
    return Response(content=jsonio.dumps(obj), media_type="application/json")


@app.get("/api/meta")
def meta():
    """meta.json plus `results`: the result names whose file exists now, so a page skips an optional file that
    is missing instead of requesting it (a 404 in the browser console)."""
    path = RESULTS / "meta.json"
    if not path.exists():
        return _json_file(path)
    m = jsonio.load(path)
    if isinstance(m, dict):
        m["results"] = sorted(n for n in RESULT_NAMES if (RESULTS / f"{n}.json").exists())
    return _json(m)


@app.get("/api/instruments")
def instruments_catalog():
    """The ticker catalog (标的推演, the portfolio builders' ticker list): Chinese names, recipe cards, parameters
    with defaults and ranges, validation, chips, the build's start state and statistics, and whether the ticker
    files fit the current scenario paths (status). Defined before /api/{name}, which would take the path."""
    from .portfolio import tickers
    return _json(tickers.catalog_report())


@app.get("/api/{name}")
def result(name: str):
    if name not in RESULT_NAMES:
        raise HTTPException(404, f"没有名为「{name}」的结果")
    return _json_file(RESULTS / f"{name}.json")


@app.get("/api/scenarios/custom")
def list_custom():
    items = []
    if CUSTOM.exists():
        for p in sorted(CUSTOM.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                d = jsonio.load(p)
            except ValueError:
                continue
            items.append({"id": d["id"], "name": d["name"], "summary": d.get("summary", ""),
                          "source": d.get("source", "custom"), "created": d.get("created"),
                          "prior_prob": d.get("prior_prob"), "prob": d.get("prob"),
                          "instruments": (CUSTOM / f"{d['id']}.instr.npz").exists()})
    return _json({"scenarios": items})


def builtin_ids() -> set[str]:
    """Ids of the built-in scenarios: the built result files plus every spec in the library."""
    ids = {p.stem for p in (RESULTS / "scenarios").glob("*.json")}
    try:
        from .scenarios.library import load_specs
        ids |= {str(s.get("id")) for s in load_specs()}
    except Exception:  # noqa: BLE001 - the built files still protect the built-in ids
        pass
    return ids


def _bad_id(sid: str) -> HTTPException:
    return HTTPException(400, f"情景 id「{sid}」不合法：只能包含小写字母、数字和下划线（最多 60 个字符）")


@app.get("/api/scenarios/{sid}")
def scenario(sid: str):
    if not ID_RE.match(sid):
        raise _bad_id(sid)
    for path in (RESULTS / "scenarios" / f"{sid}.json", CUSTOM / f"{sid}.json"):
        if path.exists():
            return _json_file(path)
    if sid in builtin_ids():
        raise HTTPException(404, f"内置情景「{sid}」的结果还没有生成：请在项目根目录运行 {build_hint('scenarios')}")
    raise HTTPException(404, f"没有找到情景「{sid}」：它不在内置情景库里，也不是已保存的自定义情景（可能已被删除）")


def _evaluate(fn, body: dict) -> Response:
    from .portfolio import service
    params = inspect.signature(fn).parameters
    known = list(params)
    unknown = [k for k in body if k not in known]
    missing = [k for k, p in params.items() if p.default is inspect.Parameter.empty and k not in body]
    if unknown:                                    # Python's own TypeError would be English
        raise HTTPException(400, f"不认识的参数：{'、'.join(map(str, unknown))}（可用参数：{'、'.join(known)}）")
    if missing:
        raise HTTPException(400, f"缺少参数：{'、'.join(missing)}")
    try:
        return _json(fn(**body))
    except service.PathsNotReady as err:          # missing or half-written path files
        raise HTTPException(503, str(err)) from err
    except (TypeError, ValueError) as err:
        raise HTTPException(400, str(err)) from err


@app.post("/api/portfolio/evaluate")
def portfolio_evaluate(body: dict = Body(...)):
    from .portfolio import service
    return _evaluate(service.evaluate_portfolio, body)


@app.post("/api/withdrawal/evaluate")
def withdrawal_evaluate(body: dict = Body(...)):
    from .portfolio import service
    return _evaluate(service.evaluate_withdrawal, body)


@app.post("/api/instruments/project")
def instruments_project(body: dict = Body(...)):
    """One ticker on every scenario path (portfolio.tickers.evaluate_project): fans, the comparison table, cash per
    share, attribution and correlations; `params` recompute the recipe on the stored scenario paths."""
    from .portfolio import tickers
    return _evaluate(tickers.evaluate_project, body)


# ---------------------------------------------------------------- custom backtest windows

BACKTEST_CACHE_MAX = 32                        # finished windows kept in memory (about 70-400 KB each)
# Windows computed at the same time. The work is mostly Python under the GIL, so more at once only slows every
# request (four together took 24 s each instead of 6 s, and /api/meta 0.5 s instead of 0.02 s); the rest queue.
BACKTEST_MAX_RUNNING = 2
_backtest_cache: OrderedDict[tuple, bytes] = OrderedDict()
_backtest_locks: dict[tuple, threading.Lock] = {}
_backtest_guard = threading.Lock()
_backtest_slots = threading.BoundedSemaphore(BACKTEST_MAX_RUNNING)


def _backtest_key_lock(key: tuple) -> threading.Lock:
    with _backtest_guard:
        return _backtest_locks.setdefault(key, threading.Lock())


def _backtest_cached(key: tuple) -> bytes | None:
    with _backtest_guard:
        if key in _backtest_cache:
            _backtest_cache.move_to_end(key)
            return _backtest_cache[key]
    return None


def _backtest_store(key: tuple, payload: bytes) -> None:
    with _backtest_guard:
        _backtest_cache[key] = payload
        _backtest_cache.move_to_end(key)
        while len(_backtest_cache) > BACKTEST_CACHE_MAX:
            old, _ = _backtest_cache.popitem(last=False)
            _backtest_locks.pop(old, None)


@app.post("/api/backtest/run")
def backtest_run(body: dict = Body(...)):
    """One custom holdout window (the 回测 tab's 自选时间段): train on months <= train_end, test the next
    `months`. A plain def, so FastAPI runs it in its threadpool and the server stays responsive; results
    are cached in memory by (train_end, months, n_paths, seed), and a second identical request waits for
    the first instead of computing the window again."""
    from .analysis import backtest as bt
    from .context import get_context
    ctx = get_context()
    try:
        req = bt.custom_request(ctx.panel, body)
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    key = (req["train_end"], req["months"], req["n_paths"], req["seed"])
    with _backtest_key_lock(key):
        payload = _backtest_cached(key)
        if payload is None:
            with _backtest_slots:
                t0 = time.perf_counter()
                try:
                    win = bt.run_custom(ctx.panel, ctx.spec, **req)
                except ValueError as err:             # e.g. a training sample the model cannot be fitted on
                    raise HTTPException(400, f"这个窗口无法回测：{err}") from err
                except Exception as err:  # noqa: BLE001 - a Chinese line instead of a bare "Internal Server Error"
                    raise HTTPException(500, f"回测出错：{type(err).__name__}: {err}") from err
            win["elapsed_s"] = round(time.perf_counter() - t0, 2)
            payload = jsonio.dumps(win).encode("utf-8")
            _backtest_store(key, payload)
    return Response(content=payload, media_type="application/json")


@app.get("/api/llm/status")
def llm_status():
    from .llm import translator
    return _json(translator.status())


def _horizon_months(value) -> int:
    """horizon_months of a translate request: an integer number of months within the translator's bounds."""
    from .llm import translator
    lo, hi = translator.HORIZON_BOUNDS
    msg = f"horizon_months 必须是 {lo} 到 {hi} 之间的整数（情景总月数）"
    if isinstance(value, bool):
        raise HTTPException(400, msg)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        value = int(value)
    if isinstance(value, float) and np.isfinite(value) and value == int(value):
        value = int(value)
    if not isinstance(value, int):
        raise HTTPException(400, f"{msg}，收到「{value}」")
    if not lo <= value <= hi:
        raise HTTPException(400, f"{msg}，收到 {value}")
    return value


@app.post("/api/llm/translate")
def llm_translate(request: Request, body: dict = Body(...)):
    from .context import get_context
    from .llm import translator
    narrative = str(body.get("narrative", "")).strip()
    if not narrative:
        raise HTTPException(400, "narrative 不能为空")
    if len(narrative) > translator.MAX_NARRATIVE_CHARS:
        raise HTTPException(400, f"叙事过长（{len(narrative)} 字），请控制在 {translator.MAX_NARRATIVE_CHARS} 字以内")
    raw = body.get("horizon_months")
    horizon = _horizon_months(config.HORIZON_MONTHS if raw is None else raw)
    if not local_host(request):            # before anything runs: the plan's usage is for this machine's own pages
        raise HTTPException(403, "翻译只接受本机页面（http://127.0.0.1 或 http://localhost）发来的请求："
                                 "Claude 订阅额度只供你自己在本机使用。")
    st = translator.status()               # local checks only (claude --version / auth status): no usage
    if not st.get("available"):
        raise HTTPException(503, st.get("reason") or "LLM 不可用")
    try:
        return _json(translator.translate(narrative, get_context(), horizon_months=horizon,
                                          refresh=body.get("refresh") is True))
    except translator.TranslatorError as err:
        raise HTTPException(502, str(err)) from err


@app.post("/api/scenarios/custom/run")
def custom_run(body: dict = Body(...)):
    from .context import get_context
    from .scenarios.runner import run_spec
    spec = body.get("spec")
    if not isinstance(spec, dict):
        raise HTTPException(400, "spec 必须是对象")
    spec = _custom_id(spec)
    from .scenarios.schema import validate_spec
    ctx = get_context()
    _, report = validate_spec(spec, ctx.db.frame, config.HORIZON_MONTHS, model_spec=ctx.model.spec)
    if report["errors"]:                                  # nothing usable would be simulated or saved
        raise HTTPException(400, "情景规格有错误：" + "；".join(report["errors"]))
    try:
        n_paths = int(min(max(int(body.get("n_paths", 600)), 100), 2000))
        detail, sim = run_spec(get_context(), spec, n_paths=n_paths, seed=int(body.get("seed", 7)),
                               source=str(body.get("source", "custom")))
    except (TypeError, ValueError, KeyError) as err:
        raise HTTPException(400, f"情景规格无法运行：{err}") from err
    detail["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if body.get("save", True):
        CUSTOM.mkdir(parents=True, exist_ok=True)
        _drop_custom_instruments(detail["id"])         # ticker files of an earlier run must not outlive its paths
        tmp = CUSTOM / f"{detail['id']}.tmp.npz"       # the portfolio service may be reading the old file
        np.savez_compressed(tmp, total=sim.total[None], income=sim.income[None],
                            cpi=sim.macro["cpi"][None], ids=np.array([detail["id"]]),
                            assets=np.array(sim.assets), labels=np.array(detail["labels"]))
        tmp.replace(CUSTOM / f"{detail['id']}.npz")
        detail["instruments"] = _custom_instruments(sim, ctx, detail)
        jsonio.dump(detail, CUSTOM / f"{detail['id']}.json")
    return _json(detail)


def _drop_custom_instruments(sid: str) -> None:
    for ext in CUSTOM_EXTS[2:]:
        (CUSTOM / f"{sid}{ext}").unlink(missing_ok=True)


def _custom_instruments(sim, ctx, detail: dict) -> dict:
    """The custom scenario's ticker columns (portfolio.tickers.write_custom: custom/<id>.instr.npz and .macro.npz,
    about 1-3 s). A failure keeps the scenario itself; its tickers are then unavailable and the reason is saved."""
    from .portfolio import tickers
    try:
        return tickers.write_custom(sim, ctx.db, detail["id"], CUSTOM, labels=detail["labels"])
    except Exception as err:  # noqa: BLE001 - the scenario's own paths are saved already
        _drop_custom_instruments(detail["id"])
        return {"saved": False, "reason": f"标的路径没有生成（{type(err).__name__}: {err}）：这个自定义情景只能按资产类别评估"}


def _custom_id(spec: dict) -> dict:
    """The spec with an id that cannot shadow a built-in scenario.

    GET /api/scenarios/<id> serves the built-in file first, so a custom scenario saved under a
    built-in id would never be shown: such an id gets a custom_ prefix (slugified again, so it
    keeps to the 40-character limit, until it is free). The id is the one validate_spec derives.
    A spec with nothing usable for an id (no id and a Chinese name both slugify to the generic
    "scenario") gets a unique custom_<hex> id, as the lab does, so two such runs never overwrite
    each other.
    """
    from .scenarios.schema import slugify
    sid = slugify(spec.get("id") or spec.get("name") or "")
    if sid == "scenario":
        sid = f"custom_{uuid.uuid4().hex[:10]}"
    taken = builtin_ids()
    if sid not in taken:
        return spec if spec.get("id") == sid else {**spec, "id": sid}
    new, k = slugify(f"custom_{sid}"), 1
    while new in taken:
        k += 1
        new = slugify(f"custom{k}_{sid}")
    return {**spec, "id": new}


@app.delete("/api/scenarios/custom/{sid}")
def custom_delete(sid: str):
    if not ID_RE.match(sid):
        raise _bad_id(sid)
    removed = False
    for ext in CUSTOM_EXTS:
        p = CUSTOM / f"{sid}{ext}"
        if p.exists():
            p.unlink()
            removed = True
    if not removed:
        raise HTTPException(404, f"没有找到自定义情景「{sid}」（可能已被删除）")
    return _json({"deleted": sid})


# ---------------------------------------------------------------- 智能体推演 (agent tree; AGENT_DESIGN §12.6)
# Every handler that can spend plan usage or change the user's state calls local_host first (403 otherwise).
# Runs: data/agents/runs/<id>/; snapshots: data/agents/snapshots/<id>/ (frozen core + overlay of on-demand nodes).

AGENT_NODE_RE = re.compile(r"^[0-9a-f]{16}$")
NOT_LOCAL_AGENTS = ("智能体推演只接受本机页面（http://127.0.0.1 或 http://localhost）发来的请求：它会调用本机 Claude Code，"
                    "计入你自己的订阅额度，或修改你保存的推演。")


NOT_SAME_ORIGIN_AGENTS = ("智能体推演只接受本页面自己发出的请求：这个请求来自另一个网站（Origin / Sec-Fetch-Site），"
                          "已拒绝，没有调用 Claude Code，也没有修改任何推演。")
NEEDS_JSON_AGENTS = "智能体推演的修改请求必须带 JSON 请求体（Content-Type: application/json）。"


def same_origin(request: Request) -> bool:
    """Whether a request comes from this server's own pages. Browsers send Origin on every cross-origin request
    (and Sec-Fetch-Site on all of them): another site's page may send a "simple" POST to 127.0.0.1 with a local Host
    header, so the Host check alone does not stop cross-site requests. A request without either header (curl, the
    tests) is not from a browser page."""
    site = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if site in ("cross-site", "same-site"):
        return False
    origin = (request.headers.get("origin") or "").strip().lower()
    if not origin:
        return True
    if origin == "null":
        return False
    host = (request.headers.get("host") or "").strip().lower()
    return origin in (f"http://{host}", f"https://{host}") and local_host(request)


def agents_refusal(request) -> str | None:
    """Why an agents request that can spend plan usage or change the user's state is refused, or None: the Host must
    name this machine (DNS rebinding), the request must come from this server's own pages (no cross-site request),
    and a POST / PUT / PATCH must carry a JSON body, which another site cannot send without a CORS preflight this
    server never answers."""
    if not local_host(request):
        return NOT_LOCAL_AGENTS
    if not same_origin(request):
        return NOT_SAME_ORIGIN_AGENTS
    if request.method in ("POST", "PUT", "PATCH"):
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return NEEDS_JSON_AGENTS
    return None


def _agents_local(request: Request) -> None:
    why = agents_refusal(request)
    if why:
        raise HTTPException(403, why)


def _agents_call(fn, *args, **kwargs):
    """Run a jobs / snapshot call and map its exceptions to Chinese HTTP errors."""
    from .agents import jobs
    try:
        return fn(*args, **kwargs)
    except jobs.NeedsConfirm as err:
        return JSONResponse(status_code=409, content={"detail": str(err), "needs_confirm": True,
                                                      "estimate": jsonio.clean(err.estimate)})
    except jobs.Conflict as err:
        return JSONResponse(status_code=409, content={"detail": str(err), "active_run": err.active})
    except KeyError as err:
        raise HTTPException(404, str(err.args[0] if err.args else err)) from err
    except ValueError as err:
        raise HTTPException(400, str(err)) from err


def _agents_response(obj):
    return obj if isinstance(obj, Response) else _json(obj)


def _node_key(key: str) -> str:
    if not AGENT_NODE_RE.match(key or ""):
        raise HTTPException(400, f"节点 key「{key}」不合法：应为 16 位十六进制")
    return key


@app.get("/api/agents/config")
def agents_config():
    from .agents import actors as A
    from .agents import model_adapter as MA
    from .agents import snapshot as S
    from .agents import synthesizer as SY
    from .agents import tree as T
    from .agents import overlay as OV
    adapter = _agents_adapter()
    info = OV.merge(adapter.field_info())
    resolver = A.FieldResolver(fields=set(adapter.x_fields) | set(info), info=info)
    model = MA.model_info(adapter)
    return _json({"catalogue": A.catalogue(resolver), "defaults": T.TreeSettings().to_dict(),
                  "snapshot_defaults": S.default_settings().to_dict(),
                  "limits": {"max_depth": 80, "max_B": 30, "max_K": 4, "max_M": 4, "particles": [50, 5000]},
                  "synth": {"axes": SY.AXES, "default_axes": SY.DEFAULT_AXES, "c_s": SY.C_S, "c_m": SY.C_M,
                            "caps": [SY.CAP_BRANCH, SY.CAP_TOTAL], "channel_w": SY.CHANNEL_W, "gates": SY.GATE_ZH},
                  "llm": {"model": "claude-opus-5-5", "effort": "medium", "hard_cap_factor": T.HARD_CAP_FACTOR},
                  "starts": {"today": {"quarter": T.FIRST_QUARTER, "root_quarter": T.ROOT_QUARTER, "data_end": "2026-09"}},
                  "versions": T.versions(adapter), "model": model, "transmission": _agents_transmission(adapter),
                  "export_enabled": False, "export": T.export_status(model, "none")})


def _agents_transmission(adapter) -> dict | None:
    """What the default pricing model's interventions reach (tree.transmission, cached per model), for the setup
    view and the honesty card; None for a model without the structural capability."""
    from .agents import snapshot as S
    from .agents import tree as T
    if "structural" not in (getattr(adapter, "capabilities", ()) or ()):
        return None
    try:
        return S.engine_transmission(T.Engine(T.TreeSettings(), adapter=adapter, root_accounts=None))
    except Exception:  # noqa: BLE001 - the probe is information for readers; the config must still load
        return None


def _agents_adapter():
    """The agent tree's default pricing model; an explicit MSL_AGENT_MODEL that cannot load is a 503 with the
    Chinese reason (never a traceback, never a silent fallback)."""
    from .agents import model_adapter as MA
    try:
        return MA.load_adapter()
    except ValueError as err:
        raise HTTPException(503, f"智能体推演的定价模型无法加载：{err}") from err


@app.get("/api/agents/schema")
def agents_schema():
    from .agents import model_adapter as MA
    from .agents import overlay as OV
    adapter = _agents_adapter()
    info = OV.merge(adapter.field_info())
    return _json({"version": MA.identity(adapter), "stub": bool(adapter.stub), "x_fields": list(adapter.x_fields),
                  "assets": list(adapter.assets), "fields": [{"id": k, **v} for k, v in sorted(info.items())]})


@app.get("/api/agents/llm_status")
def agents_llm_status(request: Request):
    """Local checks only (cached for a minute: no CLI process per request); never for another site's page."""
    if not same_origin(request):
        raise HTTPException(403, NOT_SAME_ORIGIN_AGENTS)
    from .agents import llm_io
    return _json(llm_io.llm_status())


@app.get("/api/agents/institutions")
def agents_institutions():
    from .agents import actors as A
    return _json({"items": A.load_institutions()})


@app.put("/api/agents/institutions")
def agents_institutions_put(request: Request, body: dict = Body(...)):
    _agents_local(request)
    from .agents import actors as A
    edits = body.get("edits", body)
    if not isinstance(edits, dict):
        raise HTTPException(400, "edits 必须是对象 {id: 值}")
    try:
        return _json({"items": A.save_institutions(edits)})
    except ValueError as err:
        raise HTTPException(400, str(err)) from err


@app.get("/api/agents/accounts/root")
def agents_accounts_root():
    return _json_file(RESULTS / "agents_accounts_root.json")


@app.post("/api/agents/estimate")
def agents_estimate(request: Request, body: dict = Body(...)):
    """Settings -> calls (new and cached), rounds, time, tokens, the hard cap, memory and (snapshot: true) size.
    Never calls the LLM. Cached levels: a saved run or snapshot with the same settings (any depth) is a prefix of
    this tree, so its finished levels are cache hits."""
    _agents_local(request)
    from .agents import jobs
    from .agents import snapshot as S
    from .agents import tree as T
    raw = body.get("settings", {})
    if not isinstance(raw, dict):
        raise HTTPException(400, "settings 必须是对象")
    try:
        st = S.default_settings(**{k: v for k, v in raw.items() if k not in ("B", "depth")},
                                beam=int(raw.get("B", S.DEFAULT_BEAM)), quarters=int(raw.get("depth", S.DEFAULT_QUARTERS))) \
            if body.get("snapshot") else T.TreeSettings.coerce(raw)
    except (TypeError, ValueError) as err:
        raise HTTPException(400, f"设置有错误：{err}") from err
    probs = st.problems()
    if probs:
        raise HTTPException(400, "设置有错误：" + "；".join(probs))
    from .agents import llm_io
    lat = (llm_io.llm_status().get("latency") or {})
    measured = (lat.get("p50_s"), lat.get("p90_s")) if (lat.get("n") or 0) >= 5 else None
    cached = jobs.manager().cached_levels(st)
    est = S.full_estimate(st, latency=measured, cached_levels=cached) if body.get("snapshot") else \
        T.estimate(st, latency=measured, cached_levels=cached)
    est["latency_basis"] = "measured" if measured else "default"
    est["cached_levels"] = cached
    return _json(est)


@app.get("/api/agents/runs")
def agents_runs(status: str | None = None, q: str | None = None):
    from .agents import jobs
    return _json({"runs": jobs.manager().list(status=status, q=q)})


@app.post("/api/agents/runs")
def agents_runs_start(request: Request, body: dict = Body(...)):
    _agents_local(request)
    from .agents import jobs
    settings = body.get("settings", {})
    if not isinstance(settings, dict):
        raise HTTPException(400, "settings 必须是对象")
    return _agents_response(_agents_call(jobs.manager().start, settings, name=body.get("name"),
                                         confirm=body.get("confirm") is True))


@app.get("/api/agents/runs/{rid}")
def agents_run(rid: str):
    from .agents import jobs
    return _agents_response(_agents_call(jobs.manager().get, rid))


@app.patch("/api/agents/runs/{rid}")
def agents_run_rename(request: Request, rid: str, body: dict = Body(...)):
    _agents_local(request)
    from .agents import jobs
    return _agents_response(_agents_call(jobs.manager().rename, rid, str(body.get("name", ""))))


@app.delete("/api/agents/runs/{rid}")
def agents_run_delete(request: Request, rid: str):
    _agents_local(request)
    from .agents import jobs
    return _agents_response(_agents_call(jobs.manager().delete, rid))


@app.get("/api/agents/runs/{rid}/progress")
def agents_run_progress(rid: str, since: int = 0):
    from .agents import jobs
    return _agents_response(_agents_call(jobs.manager().progress, rid, since))


@app.get("/api/agents/runs/{rid}/events")
def agents_run_events(request: Request, rid: str):
    """SSE: the journal after Last-Event-ID, then new events until the run stops (polling the journal file)."""
    from starlette.responses import StreamingResponse

    from .agents import jobs
    mgr = jobs.manager()
    try:
        mgr.read(rid)
    except KeyError as err:
        raise HTTPException(404, str(err.args[0])) from err
    try:
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0

    def stream():
        seq, idle = since, 0
        while True:
            pr = mgr.progress(rid, seq)
            for ev in pr["events"]:
                seq = max(seq, int(ev.get("seq", 0)))
                yield f"id: {ev.get('seq')}\ndata: {jsonio.dumps(ev)}\n\n"
            if pr["status"] not in ("running", "queued", "cancelling"):
                idle += 1
                if idle > 2:
                    yield f"event: end\ndata: {jsonio.dumps({'status': pr['status']})}\n\n"
                    return
            time.sleep(0.5)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/agents/runs/{rid}/cancel")
def agents_run_cancel(request: Request, rid: str, body: dict | None = Body(None)):
    _agents_local(request)
    from .agents import jobs
    return _agents_response(_agents_call(jobs.manager().cancel, rid, force=bool((body or {}).get("force"))))


@app.post("/api/agents/runs/{rid}/resume")
def agents_run_resume(request: Request, rid: str, body: dict | None = Body(None)):
    _agents_local(request)
    from .agents import jobs
    return _agents_response(_agents_call(jobs.manager().resume, rid, confirm=(body or {}).get("confirm") is True))


@app.post("/api/agents/runs/{rid}/extend")
def agents_run_extend(request: Request, rid: str, body: dict = Body(...)):
    _agents_local(request)
    from .agents import jobs
    try:
        quarters = int(body.get("quarters", 1))
    except (TypeError, ValueError) as err:
        raise HTTPException(400, "quarters 必须是整数") from err
    return _agents_response(_agents_call(jobs.manager().extend, rid, quarters, confirm=body.get("confirm") is True))


@app.post("/api/agents/runs/{rid}/intervene")
def agents_run_intervene(request: Request, rid: str, body: dict = Body(...), dry: int = 0):
    """{node, actor, option, prob} -> a child run with the intervention (?dry=1: the estimate only)."""
    _agents_local(request)
    from .agents import jobs
    from .agents import tree as T
    mgr = jobs.manager()
    try:
        parent = mgr.read(rid)
    except KeyError as err:
        raise HTTPException(404, str(err.args[0])) from err
    st = T.TreeSettings.coerce(parent["settings"])
    ov = {"node": _node_key(str(body.get("node", ""))), "actor": str(body.get("actor", "")),
          "option": str(body.get("option", "")), "prob": float(body.get("prob", 1.0))}
    if not 0.0 <= ov["prob"] <= 1.0:
        raise HTTPException(400, "prob 必须在 0 到 1 之间")
    st2 = T.TreeSettings.coerce({**st.to_dict(), "overrides": list(st.overrides or []) + [ov]})
    if dry:
        return _json({"estimate": T.estimate(st2)})
    return _agents_response(_agents_call(mgr.start, st2.to_dict(), name=f"{parent.get('name') or rid}：假如",
                                         confirm=body.get("confirm") is True, parent_run=rid))


def _explorer(kind: str, xid: str):
    from .agents import jobs
    try:
        return jobs.manager().explorer(kind, xid)
    except (KeyError, ValueError) as err:
        raise HTTPException(404, str(err.args[0] if err.args else err)) from err


def _level_view(ex, d: int):
    lv = ex.level(d)
    if lv is None:
        raise HTTPException(404, f"第 {d} 层不存在")
    idx = ex.overlay_index()
    nodes = []
    for r in lv["nodes"]:
        ent = idx.get(r["key"])
        nodes.append({**r, "overlay": "expanded"} if ent and ent.get("expanded") else r)
    core = {r["key"] for r in nodes}
    over = [dict(v, key=k) for k, v in idx.items() if v.get("level") == d and k not in core]   # overlay-only nodes
    return _json({"level": d, "nodes": nodes, "overlay": over})


def _node_view(ex, key: str):
    _node_key(key)
    rec = ex.record(key)
    if rec is None:
        raise HTTPException(404, f"没有找到节点「{key}」")
    return _json({"record": rec, "detail": ex.detail(key), "path": ex.path(key)})


def _expand(ex, request: Request, key: str, body: dict | None):
    """Generate an unexpanded branch with the LLM, in the overlay. A snapshot's branch needs no confirmation (the
    user's decision); a saved run's needs confirm: true with the real Claude Code (409 needs_confirm with the
    estimate otherwise), like every other action that spends usage on a run."""
    _agents_local(request)
    _node_key(key)
    from .agents import jobs
    body = body or {}
    try:
        quarters = int(body.get("quarters", 1))
    except (TypeError, ValueError) as err:
        raise HTTPException(400, "quarters 必须是整数") from err
    ok, why = ex.expandable(key)
    if not ok:
        ent = ex.overlay_entry(key)
        if ent is not None:
            return _json({"status": "done", "node": key, "entry": ent})
        raise HTTPException(409 if "已" in why else 404, why)
    confirmed = ex.kind == "snapshot" or body.get("confirm") is True or not jobs.real_backend()
    if not confirmed:
        est = ex.expand_estimate(key, quarters)
        return JSONResponse(status_code=409, content={
            "detail": f"在已保存的推演里生成这一支会调用本机 Claude Code，最多约 {est['calls_new']} 次（上限 {est['max_calls']} 次，"
                      "计入你的订阅额度）：确认后带上 confirm: true 重新提交", "needs_confirm": True,
            "estimate": jsonio.clean(est)})
    job = ex.submit(key, quarters=quarters, confirmed=True)
    wait = min(max(float(body.get("wait", 0) or 0), 0.0), 120.0)
    t_end = time.time() + wait
    while wait and time.time() < t_end and job["status"] in ("queued", "running"):
        time.sleep(0.1)
        job = ex.job(job["id"]) or job
    out = {"job": job, "node": key}
    if job["status"] == "done":
        out["entry"] = ex.overlay_entry(key)
    return _json(out)


@app.get("/api/agents/runs/{rid}/levels/{d}")
def agents_run_level(rid: str, d: int):
    return _level_view(_explorer("run", rid), d)


@app.get("/api/agents/runs/{rid}/nodes/{key}")
def agents_run_node(rid: str, key: str):
    return _node_view(_explorer("run", rid), key)


@app.post("/api/agents/runs/{rid}/nodes/{key}/expand")
def agents_run_expand(request: Request, rid: str, key: str, body: dict | None = Body(None)):
    _agents_local(request)
    return _expand(_explorer("run", rid), request, key, body)


@app.get("/api/agents/runs/{rid}/jobs/{jid}")
def agents_run_job(rid: str, jid: str):
    job = _explorer("run", rid).job(jid)
    if job is None:
        raise HTTPException(404, f"没有找到任务「{jid}」")
    return _json(job)


@app.get("/api/agents/runs/{rid}/branches")
def agents_run_branches(rid: str, depth: int | None = None):
    return _json(_explorer("run", rid).branches(depth))


@app.get("/api/agents/runs/{rid}/prior_trees/{tree}/branches")
def agents_run_prior_branches(rid: str, tree: str, depth: int | None = None):
    """The 历史基率树 (base_rate) or 模型树 (model) of a run: its leaf mixture (0 LLM calls)."""
    from .agents import jobs
    from .agents import snapshot as S
    if tree not in ("base_rate", "model"):
        raise HTTPException(400, "tree 只能是 base_rate 或 model")
    try:
        root = jobs.manager().run_dir(rid) / "prior_trees" / tree
    except KeyError as err:
        raise HTTPException(404, str(err.args[0])) from err
    if not root.exists():
        raise HTTPException(404, f"推演「{rid}」没有{'历史基率树' if tree == 'base_rate' else '模型树'}（prior_trees 关闭或尚未完成）")
    return _json(S.Explorer(root).branches(depth))


def _story(ex, request: Request, key: str, body: dict | None):
    """补写叙事 (P7) for one node's path: one call, cached; with the real CLI it needs confirm: true."""
    _agents_local(request)
    _node_key(key)
    from .agents import jobs
    from .agents import tree as T
    if jobs.real_backend() and not (body or {}).get("confirm") is True:
        return JSONResponse(status_code=409, content={"detail": "补写叙事会调用一次本机 Claude Code（计入你的订阅额度）："
                                                                "确认后带上 confirm: true 重新提交", "needs_confirm": True})
    path = ex.path(key)
    lines = [((ex.record(k) or {}).get("memory_line") or "") for k in path[1:]]
    lines = [x for x in lines if x]
    if not lines:
        raise HTTPException(400, "起点没有可写的路径")
    import dataclasses
    st = dataclasses.replace(ex.settings(), confirmed=True)
    llm = T.make_llm(st, cancel=None, journal=None, cache_dir=jobs.manager().cache_dir, max_calls=2)
    res = llm.story(lines, quarter_from=(ex.record(path[1]) or {}).get("quarter", ""),
                    quarter_to=(ex.record(key) or {}).get("quarter", ""))
    if not res.ok:
        raise HTTPException(502, f"补写叙事失败：{res.error}")
    return _json({"node": key, "story": res.payload, "calls": [c.to_dict() for c in res.calls]})


@app.post("/api/agents/runs/{rid}/nodes/{key}/story")
def agents_run_story(request: Request, rid: str, key: str, body: dict | None = Body(None)):
    _agents_local(request)
    return _story(_explorer("run", rid), request, key, body)


@app.post("/api/agents/snapshots/{sid}/nodes/{key}/story")
def agents_snapshot_story(request: Request, sid: str, key: str, body: dict | None = Body(None)):
    _agents_local(request)
    return _story(_explorer("snapshot", sid), request, key, body)


@app.post("/api/agents/runs/{rid}/export_set")
def agents_run_export(request: Request, rid: str):
    _agents_local(request)
    return JSONResponse(status_code=409, content={
        "detail": "组合评估与提取服务尚不支持季度情景集：交给组合评估的按钮暂时禁用（桩模型的推演也不能导出）",
        "enabled": False})


@app.get("/api/agents/sets")
def agents_sets():
    return _json({"sets": [], "enabled": False, "reason": "组合评估与提取服务尚不支持季度情景集"})


@app.get("/api/agents/snapshots")
def agents_snapshots():
    from .agents import jobs
    from .agents import snapshot as S
    return _json({"snapshots": S.list_snapshots(jobs.manager().snaps_dir)})


@app.post("/api/agents/snapshots")
def agents_snapshot_build(request: Request, body: dict = Body(...)):
    _agents_local(request)
    from .agents import jobs
    sid = str(body.get("id") or "").strip()
    if not sid:
        raise HTTPException(400, "id 不能为空")
    return _agents_response(_agents_call(jobs.manager().start_snapshot, sid, body.get("settings") or {},
                                         confirm=body.get("confirm") is True, resume=body.get("resume") is True))


@app.get("/api/agents/snapshots/{sid}")
def agents_snapshot(sid: str):
    return _json(_explorer("snapshot", sid).summary())


@app.get("/api/agents/snapshots/{sid}/levels/{d}")
def agents_snapshot_level(sid: str, d: int):
    return _level_view(_explorer("snapshot", sid), d)


@app.get("/api/agents/snapshots/{sid}/nodes/{key}")
def agents_snapshot_node(sid: str, key: str):
    return _node_view(_explorer("snapshot", sid), key)


@app.get("/api/agents/snapshots/{sid}/path/{key}")
def agents_snapshot_path(sid: str, key: str):
    ex = _explorer("snapshot", sid)
    _node_key(key)
    if ex.record(key) is None:
        raise HTTPException(404, f"没有找到节点「{key}」")
    return _json({"path": [ex.record(k) for k in ex.path(key)]})


def _best_leaf(ex, key: str):
    _node_key(key)
    if ex.record(key) is None:
        raise HTTPException(404, f"没有找到节点「{key}」")
    return _json({"node": key, "leaf": ex.best_leaf(key)})


@app.get("/api/agents/snapshots/{sid}/best_leaf/{key}")
def agents_snapshot_best_leaf(sid: str, key: str):
    """The most probable kept descendant of `key` at the deepest level its subtree reaches (the comic's path)."""
    return _best_leaf(_explorer("snapshot", sid), key)


@app.get("/api/agents/runs/{rid}/best_leaf/{key}")
def agents_run_best_leaf(rid: str, key: str):
    return _best_leaf(_explorer("run", rid), key)


@app.post("/api/agents/snapshots/{sid}/cancel")
def agents_snapshot_cancel(request: Request, sid: str, body: dict | None = Body(None)):
    """Stop a snapshot build (graceful: calls in flight finish and are cached; force: killed) or its auto-resume."""
    _agents_local(request)
    from .agents import jobs
    return _agents_response(_agents_call(jobs.manager().cancel_snapshot, sid, force=bool((body or {}).get("force"))))


@app.post("/api/agents/snapshots/{sid}/nodes/{key}/expand")
def agents_snapshot_expand(request: Request, sid: str, key: str, body: dict | None = Body(None)):
    """An unexpanded branch, generated on demand by the LLM (no confirmation dialog: the user's decision) and saved
    in the overlay; the frozen core is never written."""
    _agents_local(request)
    return _expand(_explorer("snapshot", sid), request, key, body)


@app.get("/api/agents/snapshots/{sid}/jobs/{jid}")
def agents_snapshot_job(sid: str, jid: str):
    job = _explorer("snapshot", sid).job(jid)
    if job is None:
        raise HTTPException(404, f"没有找到任务「{jid}」")
    return _json(job)


@app.get("/api/agents/snapshots/{sid}/branches")
def agents_snapshot_branches(sid: str, depth: int | None = None):
    return _json(_explorer("snapshot", sid).branches(depth))


@app.get("/api/agents/cache")
def agents_cache():
    from .agents import tree as T
    root = T.CACHE_DIR
    files = list(root.glob("*/*.json")) if root.exists() else []
    return _json({"entries": len(files), "bytes": sum(p.stat().st_size for p in files)})


# 季度模型（试运行）: /api/q/card, /api/q/run, /api/q/run/{id} (SIM_DESIGN §3.9; scenariolab/qmodel/api.py).
# Included before the static mount, which would otherwise answer every path.
app.include_router(qmodel_api.router)

app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")


def main() -> None:
    import uvicorn
    port = int(os.environ.get("MSL_PORT", "8765"))
    print(f"Causeway -> http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
