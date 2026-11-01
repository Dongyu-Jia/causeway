"""Build every result the web UI shows.

    python -m scenariolab.build                 # cached downloads
    python -m scenariolab.build --refresh       # re-download all data first
    python -m scenariolab.build --only scenarios meta
    python -m scenariolab.build --paths 1000    # more simulated paths per scenario
    python -m scenariolab.build --only ablation # the engine ablation study (~6 min, not run by default)
    python -m scenariolab.build --all           # every step, the ablation included

Order matters: causality writes the structural shock series that later steps
can use; diagnostics writes model_selection.json, which backtest, scenarios
and the server then load as "today's transmission"; forecast scores rolling
one-year-ahead forecasts and needs the scenario paths; valuation tests the
valuation state on the backtest windows and the forecast and checks itself
against backtest.json, scenarios.json and forecast.json; portfolio evaluates
the preset portfolios on the scenario paths; llm writes what the translator
would send the model (no API key needed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import time
import traceback

from . import config, jsonio
from .context import EngineContext
from .data.panel import load_meta, meta_sources, save_panel

STEPS = ["panel", "causality", "eras", "diagnostics", "backtest", "scenarios", "forecast", "valuation", "portfolio",
         "llm", "meta", "ablation"]
OPT_IN = {"ablation"}          # slow; run with --only ablation (or list it in --only)
OPTIONAL_PACKAGES = ("anthropic",)   # a step that needs only these and finds them missing is skipped, not failed


def optional_missing(err: ModuleNotFoundError) -> bool:
    """Whether a ModuleNotFoundError is an optional package (the llm extra) rather than a broken install."""
    name = str(getattr(err, "name", None) or "")
    return any(name == p or name.startswith(f"{p}.") for p in OPTIONAL_PACKAGES)


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
        "notes": panel_meta.get("notes", []), "sources": meta_sources(panel_meta),
        "asset_start": panel_meta.get("asset_start", {}), "llm": llm,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build Causeway results")
    ap.add_argument("--refresh", action="store_true", help="re-download all source data")
    ap.add_argument("--only", nargs="*", choices=STEPS, help="run only these steps")
    ap.add_argument("--skip", nargs="*", choices=STEPS, default=[], help="skip these steps")
    ap.add_argument("--all", action="store_true", help="run every step, the opt-in ones (ablation) included")
    ap.add_argument("--paths", type=int, default=600, help="simulated paths per scenario")
    args = ap.parse_args(argv)
    default = STEPS if args.all else [s for s in STEPS if s not in OPT_IN]
    steps = [s for s in (args.only or default) if s not in args.skip]
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
            if optional_missing(err):        # the llm extra is not installed: nothing to build, not an error
                print(f"[build] {name} skipped (optional package not installed: {err})", flush=True)
                return None
            traceback.print_exc()            # any other missing module is a broken install or a bug
            failures.append(name)
            print(f"[build] {name} FAILED", flush=True)
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
    step("causality", lambda: jsonio.dump(_run("analysis.causality", ctx, refresh=args.refresh),
                                          out / "causality.json"))
    ctx = EngineContext.load(selection=False)          # merge freshly written shocks
    step("eras", lambda: jsonio.dump(_run("analysis.eras", ctx), out / "eras.json"))
    step("diagnostics", lambda: jsonio.dump(_run("analysis.diagnostics", ctx), out / "diagnostics.json"))
    ctx = EngineContext.load(selection=True)           # today's transmission = selected model
    step("backtest", lambda: jsonio.dump(_run("analysis.backtest", ctx), out / "backtest.json"))
    step("scenarios", lambda: _run("scenarios.library", ctx, n_paths=args.paths))
    step("forecast", lambda: jsonio.dump(_run("analysis.forecast", ctx), out / "forecast.json"))
    step("valuation", lambda: jsonio.dump(_run("analysis.valuation", ctx), out / "valuation.json"))
    step("portfolio", lambda: _run("portfolio.service", ctx))
    step("llm", lambda: _run("llm.translator", ctx))
    step("meta", lambda: jsonio.dump(build_meta(ctx), out / "meta.json"))
    step("ablation", lambda: jsonio.dump(_run("analysis.ablation", ctx), out / "ablation.json"))

    print(f"[build] finished in {time.time() - t_all:.1f}s" + (f"; failed: {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
