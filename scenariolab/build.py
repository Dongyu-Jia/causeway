"""Build every result the web UI shows.

    python -m scenariolab.build                 # cached downloads
    python -m scenariolab.build --refresh       # re-download all data first
    python -m scenariolab.build --only scenarios meta
    python -m scenariolab.build --paths 1000    # more simulated paths per scenario

Order matters: causality writes the structural shock series that later steps
can use; diagnostics writes model_selection.json, which backtest, scenarios
and the server then load as "today's transmission".
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import time
import traceback

from . import config, jsonio
from .context import EngineContext
from .data.panel import load_meta, save_panel

STEPS = ["panel", "causality", "eras", "diagnostics", "backtest", "scenarios", "meta"]


def _run(module: str, ctx: EngineContext, **kwargs):
    return importlib.import_module(f"scenariolab.{module}").run(ctx, **kwargs)


def build_meta(ctx: EngineContext) -> dict:
    panel_meta = load_meta()
    last = ctx.panel.index.max()
    row = ctx.panel.loc[last]
    try:
        from .llm import translator
        llm = translator.status()
    except Exception as err:  # noqa: BLE001 - optional feature
        llm = {"available": False, "reason": str(err)}
    return {
        "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_start": panel_meta["start"], "data_end": panel_meta["end"],
        "model_start": config.MODEL_START, "horizon_months": config.HORIZON_MONTHS,
        "assets": config.ASSETS, "factors": config.FACTORS, "states": config.STATES, "eras": config.ERAS,
        "today": {"date": last.strftime("%Y-%m"),
                  "states": {s["id"]: row.get(f"s_{s['id']}") for s in config.STATES},
                  "income_yield": ctx.anchor.income_yield},
        "model": ctx.model.summary(), "cma": ctx.cma,
        "notes": panel_meta.get("notes", []), "sources": panel_meta.get("sources", {}),
        "asset_start": panel_meta.get("asset_start", {}), "llm": llm,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build Macro Scenario Lab results")
    ap.add_argument("--refresh", action="store_true", help="re-download all source data")
    ap.add_argument("--only", nargs="*", choices=STEPS, help="run only these steps")
    ap.add_argument("--skip", nargs="*", choices=STEPS, default=[], help="skip these steps")
    ap.add_argument("--paths", type=int, default=600, help="simulated paths per scenario")
    args = ap.parse_args(argv)
    steps = [s for s in (args.only or STEPS) if s not in args.skip]
    out = config.RESULTS_DIR
    failures: list[str] = []
    t_all = time.time()

    def step(name: str, fn):
        if name not in steps:
            return None
        t = time.time()
        print(f"[build] {name} ...", flush=True)
        try:
            res = fn()
            print(f"[build] {name} done in {time.time() - t:.1f}s", flush=True)
            return res
        except ModuleNotFoundError as err:
            print(f"[build] {name} skipped (module not available: {err})", flush=True)
        except Exception:  # noqa: BLE001 - report and keep building the rest
            traceback.print_exc()
            failures.append(name)
            print(f"[build] {name} FAILED", flush=True)
        return None

    if "panel" in steps:
        step("panel", lambda: save_panel(refresh=args.refresh))
    elif not config.PANEL_PATH.exists():
        save_panel(refresh=args.refresh)

    ctx = EngineContext.load(selection=False)
    step("causality", lambda: jsonio.dump(_run("analysis.causality", ctx), out / "causality.json"))
    ctx = EngineContext.load(selection=False)          # merge freshly written shocks
    step("eras", lambda: jsonio.dump(_run("analysis.eras", ctx), out / "eras.json"))
    step("diagnostics", lambda: jsonio.dump(_run("analysis.diagnostics", ctx), out / "diagnostics.json"))
    ctx = EngineContext.load(selection=True)           # today's transmission = selected model
    step("backtest", lambda: jsonio.dump(_run("analysis.backtest", ctx), out / "backtest.json"))
    step("scenarios", lambda: _run("scenarios.library", ctx, n_paths=args.paths))
    step("meta", lambda: jsonio.dump(build_meta(ctx), out / "meta.json"))

    print(f"[build] finished in {time.time() - t_all:.1f}s" + (f"; failed: {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
