"""Build every result the web UI shows.

    python -m scenariolab.build                 # cached downloads
    python -m scenariolab.build --refresh       # re-download all data first
    python -m scenariolab.build --only scenarios meta
    python -m scenariolab.build --only factor_audit   # the missing-factor audit (~20 s; needs diagnostics.json)
    python -m scenariolab.build --paths 1000    # more simulated paths per scenario
    python -m scenariolab.build --only ablation # the engine ablation study (~5 min, not run by default)
    python -m scenariolab.build --only regime_forecast   # regimes in the one-year forecast (~2 min, opt-in)
    python -m scenariolab.build --only quarterly qmodel  # the quarterly stack: its panel (~10 s), T-F and T-C (~2 min)
    python -m scenariolab.build --only evaluation        # pre-registration records: integrity and reproduction (opt-in)
    python -m scenariolab.build --all           # every step, the opt-in ones included

Order matters: causality writes the structural shock series that later steps
can use; diagnostics writes model_selection.json, which backtest, scenarios
and the server then load as "today's transmission" (eras attributes with the
selected model but runs before diagnostics: a build that runs both attributes
the eras again when the walk-forward changed the selection, as on a fresh
clone's first build, and --only diagnostics prints a note); factor_audit replicates the
research round's archive and evaluates the changes against that selected model
(it reads diagnostics.json's walk-forward); regimes reuses the engine's
unconditional reference run that the scenario readouts computed (rebuild it
after --only scenarios); forecast scores rolling one-year-ahead forecasts and
needs the scenario paths; valuation tests the valuation state on the backtest
windows and the forecast and checks itself against backtest.json,
scenarios.json and forecast.json; portfolio evaluates the preset portfolios on
the scenario paths; llm writes what the translator would send the model (it
never calls a model: its status runs only `claude --version` and `claude auth
status`, which spend no usage; its regime section reads regimes.json). The quarterly stack (SIM_DESIGN §3.9,
试运行): quarterly builds data/processed/panel_quarterly.csv from the monthly panel and the raw files the
quarterly import cached (python -m scenariolab.qmodel.sources_q import ...; it never downloads), qmodel fits the
factor layer and writes T-F and T-C to data/results/qmodel.json (it needs model_selection.json and the quarterly
panel), and the opt-in evaluation checks the pre-registration records and reproduces those whose inputs are the
frozen ones. A quarterly-stack step whose inputs are not on disk is skipped with the missing file named, not failed:
a fresh clone builds the monthly app without them.
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
from .engine.factor_model import ModelSpec
from .env import load_dotenv

STEPS = ["panel", "quarterly", "causality", "eras", "diagnostics", "factor_audit", "backtest", "scenarios", "regimes",
         "forecast", "valuation", "portfolio", "llm", "qmodel", "meta", "ablation", "regime_forecast", "evaluation"]
OPT_IN = {"ablation", "regime_forecast", "evaluation"}   # slow; run with --only <step> (or --all)
SKIP_IF_MISSING = {"quarterly", "qmodel", "evaluation"}   # the quarterly stack: skipped when its inputs are not cached
OPTIONAL_PACKAGES = ("anthropic",)   # a step that needs only these and finds them missing is skipped, not failed


def optional_missing(err: ModuleNotFoundError) -> bool:
    """Whether a ModuleNotFoundError is an optional package (the llm extra) rather than a broken install."""
    name = str(getattr(err, "name", None) or "")
    return any(name == p or name.startswith(f"{p}.") for p in OPTIONAL_PACKAGES)


def eras_model_stale(out) -> bool:
    """Whether out/eras.json was attributed with another transmission than the selected one (out/
    model_selection.json; the base spec without it), compared as ModelSpecs with the name aside (the eras page
    warns on the same mismatch). False when eras.json is missing or unreadable: there is nothing to redo (a failed
    step is reported)."""
    try:
        used = jsonio.load(out / "eras.json")["attribution_info"]["model"]["spec"]
    except (OSError, KeyError, TypeError, ValueError):
        return False
    try:
        selected = jsonio.load(out / "model_selection.json")["spec"]
    except (OSError, KeyError, TypeError, ValueError):
        selected = ModelSpec().to_dict()
    try:
        return ModelSpec.from_dict({**used, "name": ""}) != ModelSpec.from_dict({**selected, "name": ""})
    except (TypeError, ValueError):
        return True


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
        "view_concepts": config.VIEW_CONCEPTS,          # the view variables' concept names (the method tab groups them)
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
    ap.add_argument("--all", action="store_true",
                    help="run every step, the opt-in ones (ablation, regime_forecast) included")
    ap.add_argument("--paths", type=int, default=600, help="simulated paths per scenario")
    args = ap.parse_args(argv)
    load_dotenv()                 # the optional FRED_API_KEY (data/sources.py) lives in the git-ignored .env
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
        except FileNotFoundError as err:
            if name in SKIP_IF_MISSING:      # the quarterly stack's raw files or panel are not cached here
                print(f"[build] {name} skipped (missing input: {err})", flush=True)
                return None
            traceback.print_exc()
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
    step("quarterly", lambda: _run("qmodel.panel", ctx))
    step("causality", lambda: jsonio.dump(_run("analysis.causality", ctx, refresh=args.refresh),
                                          out / "causality.json"))
    ctx = EngineContext.load(selection=False)          # merge freshly written shocks
    step("eras", lambda: jsonio.dump(_run("analysis.eras", ctx), out / "eras.json"))
    step("diagnostics", lambda: jsonio.dump(_run("analysis.diagnostics", ctx), out / "diagnostics.json"))
    ctx = EngineContext.load(selection=True)           # today's transmission = selected model
    if "eras" in steps and "diagnostics" in steps and eras_model_stale(out):
        # the walk-forward picked another model than the one eras just attributed with (a fresh clone's first
        # build, a new factor): attribute again, so eras.json and the scenarios use the same transmission
        print("[build] eras: diagnostics changed the selected model, attributing the eras again", flush=True)
        step("eras", lambda: jsonio.dump(_run("analysis.eras", ctx), out / "eras.json"))
    step("factor_audit", lambda: jsonio.dump(_run("analysis.factor_audit", ctx), out / "factor_audit.json"))
    step("backtest", lambda: jsonio.dump(_run("analysis.backtest", ctx), out / "backtest.json"))
    step("scenarios", lambda: _run("scenarios.library", ctx, n_paths=args.paths))
    step("regimes", lambda: jsonio.dump(_run("analysis.regimes", ctx), out / "regimes.json"))
    step("forecast", lambda: jsonio.dump(_run("analysis.forecast", ctx), out / "forecast.json"))
    step("valuation", lambda: jsonio.dump(_run("analysis.valuation", ctx), out / "valuation.json"))
    step("portfolio", lambda: _run("portfolio.service", ctx))
    step("llm", lambda: _run("llm.translator", ctx))
    step("qmodel", lambda: _run("qmodel.factor_checks", ctx))      # writes data/results/qmodel.json itself
    step("meta", lambda: jsonio.dump(build_meta(ctx), out / "meta.json"))
    step("ablation", lambda: jsonio.dump(_run("analysis.ablation", ctx), out / "ablation.json"))
    step("regime_forecast", lambda: jsonio.dump(_run("analysis.regime_forecast", ctx), out / "regime_forecast.json"))
    step("evaluation", lambda: _run("evaluation.protocol", ctx))   # writes data/results/evaluation.json itself

    if "diagnostics" in steps and "eras" not in steps and eras_model_stale(out):
        print("[build] note: the selected model changed; run --only eras too (the eras attribution uses it)", flush=True)
    if "scenarios" in steps and "regimes" not in steps and (out / "regimes.json").exists():
        # the scenario readouts and regimes.json share the engine's reference run; the regimes tab warns
        # when they come from different builds
        print("[build] note: scenarios were rebuilt without regimes; run --only regimes too", flush=True)
    print(f"[build] finished in {time.time() - t_all:.1f}s" + (f"; failed: {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
