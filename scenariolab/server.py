"""Local web server: static UI + JSON results + on-demand engine calls.

    python -m scenariolab.server            # http://127.0.0.1:8765
    MSL_PORT=9000 python -m scenariolab.server
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import config, jsonio
from .env import load_dotenv

load_dotenv()

app = FastAPI(title="Macro Scenario Lab", version="0.1.0")
RESULTS = config.RESULTS_DIR
CUSTOM = RESULTS / "custom"
ID_RE = re.compile(r"^[a-z0-9_]{1,60}$")


def _json_file(path) -> Response:
    if not path.exists():
        raise HTTPException(404, f"{path.name} 还没有生成：请先运行 python -m scenariolab.build")
    return Response(content=path.read_bytes(), media_type="application/json")


def _json(obj) -> Response:
    return Response(content=jsonio.dumps(obj), media_type="application/json")


@app.get("/api/meta")
def meta():
    return _json_file(RESULTS / "meta.json")


@app.get("/api/{name}")
def result(name: str):
    if name not in {"eras", "causality", "backtest", "diagnostics", "scenarios"}:
        raise HTTPException(404, "unknown result")
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
                          "prior_prob": d.get("prior_prob")})
    return _json({"scenarios": items})


@app.get("/api/scenarios/{sid}")
def scenario(sid: str):
    if not ID_RE.match(sid):
        raise HTTPException(400, "bad id")
    for path in (RESULTS / "scenarios" / f"{sid}.json", CUSTOM / f"{sid}.json"):
        if path.exists():
            return _json_file(path)
    raise HTTPException(404, f"scenario {sid} not found")


@app.post("/api/portfolio/evaluate")
def portfolio_evaluate(body: dict = Body(...)):
    from .portfolio import service
    try:
        return _json(service.evaluate_portfolio(**body))
    except (TypeError, ValueError) as err:
        raise HTTPException(400, str(err)) from err


@app.post("/api/withdrawal/evaluate")
def withdrawal_evaluate(body: dict = Body(...)):
    from .portfolio import service
    try:
        return _json(service.evaluate_withdrawal(**body))
    except (TypeError, ValueError) as err:
        raise HTTPException(400, str(err)) from err


@app.get("/api/llm/status")
def llm_status():
    from .llm import translator
    return _json(translator.status())


@app.post("/api/llm/translate")
def llm_translate(body: dict = Body(...)):
    from .context import get_context
    from .llm import translator
    narrative = str(body.get("narrative", "")).strip()
    if not narrative:
        raise HTTPException(400, "narrative 不能为空")
    st = translator.status()
    if not st.get("available"):
        raise HTTPException(503, st.get("reason") or "LLM 不可用")
    try:
        return _json(translator.translate(narrative, get_context(),
                                          horizon_months=int(body.get("horizon_months", config.HORIZON_MONTHS))))
    except translator.TranslatorError as err:
        raise HTTPException(502, str(err)) from err


@app.post("/api/scenarios/custom/run")
def custom_run(body: dict = Body(...)):
    from .context import get_context
    from .scenarios.runner import run_spec
    spec = body.get("spec")
    if not isinstance(spec, dict):
        raise HTTPException(400, "spec 必须是对象")
    n_paths = int(min(max(int(body.get("n_paths", 600)), 100), 2000))
    detail, sim = run_spec(get_context(), spec, n_paths=n_paths, seed=int(body.get("seed", 7)),
                           source=str(body.get("source", "custom")))
    detail["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if body.get("save", True):
        CUSTOM.mkdir(parents=True, exist_ok=True)
        jsonio.dump(detail, CUSTOM / f"{detail['id']}.json")
        np.savez_compressed(CUSTOM / f"{detail['id']}.npz", total=sim.total[None], income=sim.income[None],
                            cpi=sim.macro["cpi"][None], ids=np.array([detail["id"]]),
                            assets=np.array(sim.assets), labels=np.array(detail["labels"]))
    return _json(detail)


@app.delete("/api/scenarios/custom/{sid}")
def custom_delete(sid: str):
    if not ID_RE.match(sid):
        raise HTTPException(400, "bad id")
    removed = False
    for ext in (".json", ".npz"):
        p = CUSTOM / f"{sid}{ext}"
        if p.exists():
            p.unlink()
            removed = True
    if not removed:
        raise HTTPException(404, "not found")
    return _json({"deleted": sid})


app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")


def main() -> None:
    import uvicorn
    port = int(os.environ.get("MSL_PORT", "8765"))
    print(f"Macro Scenario Lab -> http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
