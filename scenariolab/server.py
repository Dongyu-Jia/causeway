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

load_dotenv()

RESULTS = config.RESULTS_DIR
CUSTOM = RESULTS / "custom"
ID_RE = re.compile(r"^[a-z0-9_]{1,60}$")
RESULT_NAMES = {"eras", "causality", "backtest", "diagnostics", "scenarios", "forecast", "ablation",
                "portfolio_default", "llm_translator", "valuation", "regimes", "regime_forecast", "factor_audit"}
# Which command writes each result file (a 404 names it). Default-build steps can also be rebuilt alone.
BUILD = "python -m scenariolab.build"
BUILD_STEP = {"meta": "meta", "eras": "eras", "causality": "causality", "backtest": "backtest",
              "diagnostics": "diagnostics", "scenarios": "scenarios", "forecast": "forecast",
              "valuation": "valuation", "portfolio_default": "portfolio", "llm_translator": "llm",
              "regimes": "regimes", "factor_audit": "factor_audit"}
BUILD_HINT = {
    "ablation": f"{BUILD} --only ablation（引擎消融约需 5 分钟，默认构建不包含这一步）",
    "regime_forecast": f"{BUILD} --only regime_forecast（宏观体制的一年期预测检验约需 2 分钟，默认构建不包含这一步）",
    "factor_audit": f"{BUILD}（或只补这一步：{BUILD} --only factor_audit，缺失因子审计约需 20 秒，要先有 diagnostics.json）",
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
                          "prior_prob": d.get("prior_prob"), "prob": d.get("prob")})
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
        tmp = CUSTOM / f"{detail['id']}.tmp.npz"       # the portfolio service may be reading the old file
        np.savez_compressed(tmp, total=sim.total[None], income=sim.income[None],
                            cpi=sim.macro["cpi"][None], ids=np.array([detail["id"]]),
                            assets=np.array(sim.assets), labels=np.array(detail["labels"]))
        tmp.replace(CUSTOM / f"{detail['id']}.npz")
        jsonio.dump(detail, CUSTOM / f"{detail['id']}.json")
    return _json(detail)


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
    for ext in (".json", ".npz"):
        p = CUSTOM / f"{sid}{ext}"
        if p.exists():
            p.unlink()
            removed = True
    if not removed:
        raise HTTPException(404, f"没有找到自定义情景「{sid}」（可能已被删除）")
    return _json({"deleted": sid})


app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")


def main() -> None:
    import uvicorn
    port = int(os.environ.get("MSL_PORT", "8765"))
    print(f"Causeway -> http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
