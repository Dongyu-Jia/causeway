# Architecture and module contracts

This file is the contract every module codes against. If you change a shape
here, change every producer and consumer of it.

Example numbers come from the build of 2026-11-01 (`meta.json` `built_at`
2026-11-01T17:50:45Z, data through 2026-09). They move with every data
update; the shapes and rules do not.

## Layout

```
scenariolab/
  __init__.py              package version
  config.py                shared ids and engine settings: paths, MODEL_START, HORIZON_MONTHS, BLOCK_MONTHS,
                           RARE_MONTHS, RESIDUALS, EXOGENOUS_SHOCKS, ERAS, ASSETS, FACTORS, STATES, VALUATION
  env.py                   minimal .env loader (KEY=VALUE lines; never overrides a variable that is already set)
  jsonio.py                JSON I/O: numpy/pandas-safe, NaN/inf -> null, floats to 6 decimals, atomic writes
  context.py               EngineContext (panel + selected model + MonthDB + Anchor + CMA); get_context()
  build.py                 CLI: python -m scenariolab.build (see "The build")
  server.py                FastAPI app + static UI: python -m scenariolab.server (see "HTTP API")
  data/sources.py          cached downloaders: FRED graph CSV, Ken French zips, Nareit workbook, gold CSV,
                           Shiller ie_data.xls (optional: a failed download leaves CAPE empty); the FRED API
                           client and the publisher downloaders (see "The missing-factor round")
  data/import_raw.py       one-off import of the research round's downloads -> data/raw/ + SOURCES.json
  data/panel.py            builds data/processed/panel_monthly.csv + panel_meta.json (incl. column_info)
  engine/factor_model.py   ModelSpec + AssetFactorModel (linear / lags / inflation regime / ridge / window / MLP)
  engine/ep.py             entropy pooling: entropy_pool(), apply_views(), blend(), support(), weighted_corr()
  engine/pooling.py        phase weights: views solved on block starts; MIN_ENS / MIN_EPISODES; shrink lambda
  engine/sampler.py        weighted block bootstrap: BlockProbs, sample_phases()
  engine/residuals.py      residual draw modes: coupled, decoupled, detrended, trend_decoupled, partial, recentered
  engine/simulate.py       MonthDB, Anchor, capital_market_assumptions(), ValuationModel, simulate(),
                           Attribution (the additive split of each simulated return, attribution=True)
  engine/summary.py        wealth paths, fans, representative path, Brownian-bridge candles, correlations
  scenarios/schema.py      spec schema (YAML and the LLM JSON schema), validate_spec(), allowed_variables()
  scenarios/runner.py      run_spec(ctx, spec) -> (detail dict, SimOutput); phase_pools(); summarize(); headline();
                           attribution_summary() (assets[a].attribution)
  scenarios/library.py     runs specs/*.yaml -> results/scenarios.json, scenarios/<id>.json, scenario_paths.npz
  scenarios/probabilities.py  prior x likelihood(today)^gamma -> posterior scenario probabilities
  scenarios/calibrate.py   income-distribution -> consumption calculator (a spec's `calibration` block)
  scenarios/specs/*.yaml   the 13 built-in scenarios (the schema the LLM emits, plus reversion / calibration)
  analysis/eras.py         -> results/eras.json (incl. each era's return attribution)
  analysis/causality.py    -> results/causality.json + data/processed/structural_shocks.csv
  analysis/walkforward.py  diagnose -> fix -> untouched-test loop that picks the transmission model
  analysis/diagnostics.py  -> results/diagnostics.json + results/model_selection.json
  analysis/backtest.py     -> results/backtest.json (transmission R^2, four holdout windows); run_custom() for
                           the custom windows of POST /api/backtest/run
  analysis/forecast.py     -> results/forecast.json (rolling one-year forecasts since 1984-12, live forecast,
                           its reconciliation with the scenario mixture) + data/ledger/ (the live record)
  analysis/valuation.py    -> results/valuation.json (P/E, ROIC, FCF as factors? valuation-state variants)
  analysis/ablation.py     -> results/ablation.json (engine ablation; opt-in build step)
  analysis/factor_tests.py the research harness's validated core (same-sample tests, BH, placebos, residuals)
  analysis/factor_audit.py -> results/factor_audit.json (缺失因子审计: replication of the archive + current model)
  analysis/audit_archive/  research_2026_11_01.json: the frozen research archive (package data)
  portfolio/metrics.py     portfolio wealth (rebalance monthly / annual / none), path metrics, weighted stats
  portfolio/withdrawal.py  three withdrawal strategies, ruin, critical (safe) withdrawal rates
  portfolio/service.py     evaluate_portfolio(), evaluate_withdrawal() for the API; build step ->
                           portfolio_default.json + continuation_paths.npz (withdrawal years 16-40)
  llm/translator.py        narrative -> scenario spec via Claude (structured output); build step -> llm_translator.json
web/
  index.html               shell: sidebar nav, K-line colour select, vendor/echarts/echarts.min.js, js/app.js
  vendor/echarts/          Apache ECharts 5.5.1 (echarts.min.js, LICENSE, NOTICE; Apache-2.0), from the npm
                           tarball (sha512 checked against the registry): no CDN, the UI works offline
  css/app.css              tokens and layout; --up / --down follow the K-line setting
  js/app.js                hash router, lazy-loaded tabs
  js/api.js                fetch wrapper (GET results cached per page load)
  js/ui.js                 DOM helpers, number formats, colour classes, META lookups
  js/charts.js             ECharts helpers and their lifecycle, K-line colours
  js/tabs/<id>.js          one module per tab (12, see "Web app")
  js/tabs/_scenario_detail.js, js/tabs/_weights.js   shared helpers (a leading underscore = not a tab)
  js/tabs/_factor_audit.js the diagnostics tab's 「缺失因子审计」 section (factor_audit.json; deep links
                           #diagnostics?audit, ?audit-families, ?audit-assets, ?audit-concepts)
tests/                     pytest suite (21 files, see "Testing")
data/                      raw/, processed/, results/, ledger/ (git-ignored, built locally; ledger/ is the
                           append-only record of live one-year forecasts, see "Forecast ledger")
docs/                      ARCHITECTURE.md (this file), screenshots/ (the README's four PNGs),
                           research/2026-11-01-factor-audit/ (the research round's reports, statistics only)
.github/workflows/ci.yml   CI on every push and pull request (see "Testing")
CONTRIBUTING.md            setup, build, tests, adding a scenario or a data source (Chinese + English summary)
.env.example               template for .env: ANTHROPIC_API_KEY, FRED_API_KEY (optional), MSL_LLM_MODEL, MSL_PORT
```

## Conventions

* Monthly data, index = month start (`pd.Timestamp("2026-09-01")`). JSON dates
  are `"YYYY-MM"` strings for monthly series and `"YYYY"` for yearly ones; a few
  metadata fields are `"YYYY-MM-DD"` (`meta.data_start` / `data_end`,
  `meta.asset_start`, `meta.model.train_start` / `train_end`, a scenario
  detail's `start`).
* Returns are decimal simple returns (0.01 = 1%). Rates, yields, spreads, CPI
  YoY and the other `s_` levels are in percent (4.5 = 4.5%). `f_` factor
  changes are monthly, in the unit `config.FACTORS` gives (Δlog or Δpp). P/D is
  a multiple (price / trailing 12-month dividends).
* Panel column families (data/panel.py): `s_<state>`, `f_<factor>`,
  `r_<asset>` (total), `p_<asset>` (price), `d_<asset>` (income; r = p + d),
  `dy_<asset>` (trailing 12-month income yield, %), `lvl_<name>` (raw levels
  for re-anchoring), `e_<shock>` (structural shocks, merged from
  processed/structural_shocks.csv by `context.load_panel_with_shocks()` when
  present), `x_<name>` (stored, audit-only old definitions; never views or
  model inputs; see "The missing-factor round" for them, release dating and
  `column_info`).
* Models are estimated from `config.MODEL_START` (1973-01). The MonthDB holds
  the months from 1973-02 to the last panel month that have every required
  column: 644 months in the current build.
* All JSON goes through `scenariolab.jsonio`: `dump()` writes atomically (tmp
  file + rename) with `ensure_ascii=False`; NaN/inf become `null`, floats are
  rounded to 6 decimals, Timestamps become `"YYYY-MM"`, and a rounded negative
  zero in prose ("-0.0%") loses its sign (web/js/api.js applies the same rule
  to every response).
* UI text and API error details are Simplified Chinese. Code, ids and comments
  are English.
* Every analysis module exposes `run(ctx, ...) -> dict`. build.py writes the
  returned dict for causality, eras, diagnostics, backtest, forecast,
  valuation, meta and ablation; scenarios.library, portfolio.service and
  llm.translator write their own files; diagnostics also writes
  model_selection.json, causality also writes
  processed/structural_shocks.csv, forecast appends its live forecast to
  data/ledger/ and portfolio.service writes results/continuation_paths.npz.
  Nothing writes outside `data/`.
* Experiments go elsewhere: `scenarios.library.run(ctx, out_dir=<dir>,
  ids=[...])` writes a subset (probabilities renormalized over it), and the
  ablation and valuation studies run their scenario variants in a temporary
  (or `--scen-dir`) directory and refuse one inside data/results.

## The build (build.py)

```
python -m scenariolab.build                     # every default step (all but ablation)
python -m scenariolab.build --refresh           # re-download every source first
python -m scenariolab.build --skip panel        # everything from the cached panel
python -m scenariolab.build --only scenarios forecast valuation portfolio llm
python -m scenariolab.build --paths 1000        # paths per built-in scenario (default 600)
python -m scenariolab.build --only ablation     # the opt-in engine ablation (~5 min)
python -m scenariolab.build --all               # every step, ablation included
```

`STEPS = [panel, causality, eras, diagnostics, factor_audit, backtest,
scenarios, regimes, forecast, valuation, portfolio, llm, meta, ablation,
regime_forecast]` always run in this order, whatever order `--only` lists
them in. `OPT_IN = {"ablation", "regime_forecast"}` is left out of the
default set: `--all` adds it, `--only` picks any steps (opt-in ones included),
`--skip` removes steps from either. `--paths N` changes only the scenarios step
(the backtest's 400, the forecast's 1000, and the studies' 400 per window and
600 per scenario run are fixed in their modules). Without `--refresh` the panel step downloads only the
files that are missing (the optional Shiller workbook included: 2 attempts, 60 s
timeout, cached only when the body is an .xls; a failure leaves CAPE empty);
`--refresh` re-downloads every source in the panel step and the quarterly FRED
series of the causality step. A failing step prints its traceback and the build goes
on (exit code 1, the failed steps listed at the end). Only a missing optional
package skips a step (`build.OPTIONAL_PACKAGES = ("anthropic",)`, the llm extra;
`build.optional_missing`); any other `ModuleNotFoundError` fails it. When panel is not among the steps and
data/processed/panel_monthly.csv does not exist, the panel is built (and its
data downloaded) anyway.

| Step | Code | Reads | Writes (data/…) |
|---|---|---|---|
| panel | `data.panel.save_panel` | data/raw/** (downloads files that are missing) | processed/panel_monthly.csv, processed/panel_meta.json |
| causality | `analysis.causality.run` | panel; FRED PRS85006173, GDPC1, PCECC96, CP, GDP (cached in data/raw/fred) | results/causality.json, processed/structural_shocks.csv |
| eras | `analysis.eras.run` | panel, model_selection.json (the model its attribution splits by) | results/eras.json |
| diagnostics | `analysis.diagnostics.run` | panel + shocks | results/diagnostics.json, results/model_selection.json |
| factor_audit | `analysis.factor_audit.run` (selection context) | panel + shocks, diagnostics.json, model_selection.json, the frozen archive | results/factor_audit.json |
| backtest | `analysis.backtest.run` | panel, model_selection.json | results/backtest.json |
| scenarios | `scenarios.library.run` | specs/*.yaml, model_selection.json | results/scenarios.json, results/scenarios/<id>.json, results/scenario_paths.npz |
| forecast | `analysis.forecast.run` | panel, model_selection.json, scenario_paths.npz (live scenario mixture), results/scenarios/<id>.json (each built-in's validated spec, re-run with runner.run_spec at library.seed_for(id) and the npz path count for its year-one macro paths; this reproduces the npz exactly, `live.reconciliation.paths_reproduced`), data/ledger/*.json | results/forecast.json; appends data/ledger/forecast_<origin>.json |
| valuation | `analysis.valuation.run` | panel, forecast.json (analog_ens, reproduction check), backtest.json and scenarios.json (consistency checks), data/raw/shiller (cache only, never downloads) | results/valuation.json |
| portfolio | `portfolio.service.run` | the context (build_continuation), scenario_paths.npz, scenarios.json | results/continuation_paths.npz, results/portfolio_default.json |
| llm | `llm.translator.run` | specs/*.yaml (few-shot examples), the MonthDB | results/llm_translator.json |
| meta | `build.build_meta` | panel_meta.json + `data.panel.SOURCES` (`meta_sources`), the context, `translator.status()` | results/meta.json |
| ablation (opt-in) | `analysis.ablation.run` | panel, backtest.json (reproduction check), scenarios.json | results/ablation.json |

The context is loaded with `selection=False` before causality and again after
it (to merge the fresh shocks); after diagnostics it is loaded with
`selection=True`, so backtest and every later step use "today's transmission"
from model_selection.json. A default build without downloads takes about 4
minutes, most of it the valuation step. The last full build (`--all --skip
panel`, 2026-11-01) took 532 s with a peak RSS of 2.6 GB: causality 2.7 s, eras
0.4, diagnostics 6.9, backtest 21.5, scenarios 6.7, forecast 13.7, valuation
151.2 (`valuation.json` `settings.seconds` 151.1), portfolio 19.7 (most of it
the 40-year continuation), llm 0.8, meta 0.0, ablation 307.9; 224 s without
the ablation.

Partial rebuilds:

* After editing a built-in spec, or after `--only scenarios`, rebuild
  forecast, valuation, portfolio and llm as well
  (`--only scenarios forecast valuation portfolio llm`): forecast mixes the
  scenario paths into its live forecast, valuation compares its own copy of
  the scenario results with scenarios.json (the 估值与盈利 page asks for
  `--only valuation` when they differ), portfolio_default.json is computed on
  the paths, and the translator's few-shot examples are built-in specs.
  The portfolio step also rebuilds continuation_paths.npz, whose start must
  match the scenario paths' first label.
* A different model in model_selection.json changes every later step. The
  eras step fits the selected model for its attribution but runs before
  diagnostics, so after a walk-forward that changes the selection run
  `--only eras` again (the eras page warns when `attribution_info.model.spec`
  differs from `meta.model.spec`).
* After a data update, scenarios must run before forecast (a full build does):
  scenario paths that start in another month are never mixed into the live
  forecast, reconciled or recorded in the ledger (`live.calibration.reason`
  names `--only scenarios forecast`).
* The server keeps one EngineContext (`context.get_context()`) for custom runs,
  custom backtest windows and translations: restart it after a rebuild. Result
  files are read from disk on every request, and the portfolio service reloads
  the path files (scenario_paths.npz, continuation_paths.npz, custom/*.npz) when
  their mtime changes.

Module entry points: `python -m scenariolab.analysis.ablation [--paths 400]
[--seed 11] [--extra-seeds 12 13 14 15] [--scen-dir DIR] [--scen-paths 600]
[--no-scenarios]`; `python -m scenariolab.analysis.valuation [--paths 400]
[--seeds 11 12 13 14 15] [--scen-dir DIR] [--scen-paths 600] [--no-scenarios]
[--no-forecast] [--forecast-paths 1000]`; `python -m scenariolab.llm.translator
"<叙事>" [--run] [--yaml OUT] [--horizon N] [--paths N] [--model ID]
[--print-prompt]`. analysis.backtest, forecast, diagnostics, eras and
causality also run as `python -m scenariolab.analysis.<name>` and write their
own result file (forecast also appends to the ledger); `python -m
scenariolab.portfolio.service` loads an EngineContext itself and writes
continuation_paths.npz and portfolio_default.json.

## Engine settings (config.py)

| Setting | Current value | What it changes |
|---|---|---|
| `MODEL_START` | `"1973-01"` | first month of every model fit and of the MonthDB; earlier eras are descriptive only |
| `HORIZON_MONTHS` | 180 | months per scenario path (15 years); validate_spec pads or trims the phases to it |
| `BLOCK_MONTHS` | 6 | sampler block length; each phase's views are solved on block starts of length `min(6, phase months)` |
| `RARE_MONTHS` | 2020-03..2020-08, weight 0.25 | prior weight of the 2020 lockdown months; a block start that would draw any of them gets the factor in full |
| `RESIDUALS` | `{"resid_mode": "partial", "resid_coupling": 0.5}` | the residual draw of every simulation (scenarios, backtest, forecast, the valuation study's baseline); chosen by analysis/ablation.py on the 2026-11-01 build, whose own baseline is the pre-adoption coupled draw (its `partial_50` variant is the engine); on the missing-factor round's build the same rule picks `decoupled` (validation MAE 2.80% vs 3.47%, test 4.81% vs 4.99%): the engine is not switched until that is reviewed (a switch needs a full rebuild) |
| `EXOGENOUS_SHOCKS` | 7 events, 1973-10 (war and oil embargo) .. 2026-03 (oil supply shock, WTI +54% in a month) | one-year forecast years that overlap one are flagged, every forecast score is also reported without them, and a large miss in such a year is attributed to the shock |
| `VALUATION` | see below | the valuation state |
| `FACTORS` | 27 ids (roles: see "The missing-factor round") | `core` (10): the fixed comparator `ModelSpec()` of the walk-forward (the selected model and the driftable factors come from model_selection.json); `endogenous` (wealth, valuation, cape): built partly from asset prices, never accepted by the walk-forward, a view on them warns; `optional` (cape): its source may be missing, and a missing month counts as no change |
| `STATES` | 60 ids | level variables (era fingerprints, views, today's state); `endogenous` (cape) warns as a view; 30 of them are the scenario view variables of the missing-factor round (`VIEW_STATE_IDS`) |
| `ERAS` | 9 eras, 1955-01 .. today | the era of a month (`simulate.era_of`), fingerprints, a phase's era mix; the simulator draws only months from 1973 on |
| `ASSETS` | 13 ids | ids, names, data proxies, `group`, and the `income` source (dividend / coupon / interest / none) |

`FACTORS`: core = oil, ip, cpi_goods, cpi_services, ff, y10, spread, usd,
unrate, labor_share; candidates = profit_share, capex_share, wealth (end.),
white_collar, wage, sentiment, valuation (end.), real_y10, profit_growth,
fcf_share, cape (end., optional). The simulator also tracks `cpi` (f_cpi), so
a MonthDB month needs every non-optional `f_` column.

`VALUATION`:

| Key | Value | Meaning |
|---|---|---|
| `enabled` | False | the pull toward a fitted fair value; off for scenarios, backtests and the one-year forecast (all 9 variants lost, see valuation.json) |
| `track` | True | with `enabled` off, scenario runs still track the market P/D at kappa = 0, so every return is unchanged: detail `macro.pd`, headline `pd_end_p50`; not used in backtests or the forecast |
| `infl_half_life_months` | 36 | EWMA half-life of the inflation-expectation proxy (panel `s_infl_expect` and the simulated paths) |
| `release_lag_months` | 5 | a NIPA quarter (dated at its first month) is known 5 months later: `s_profit_share_rt`, `s_profit_growth_yoy`, `s_fcf_share`, `f_profit_growth`, `f_fcf_share` |
| `features` | real_y10, profit_share, infl_expect | regressors of the fair value ln P/D* |
| `kappa` | 0.1 | annual pull speed when enabled (a number, or "ecm" for the in-sample error-correction estimate) |
| `cma` | "historical" | the drift the pull is added to: "historical" or "valuation_state" |
| `income` | "constant" | equity income: the anchor yield, or "path" = yield_0 x (P/D_0 / P/D_t)^beta |

Engine constants outside config.py, each set by the evidence in its module
docstring: `engine/pooling.py` MIN_ENS 30, MIN_EPISODES 6, EPISODE_MONTHS 12,
SHRINK_STEPS 14; `engine/simulate.py` DEFAULT_REVERSION, RATE_SCALING,
SPREAD_WIDENING_FLOOR 0.85, OIL_VIEW_VARS; `engine/residuals.py`
DEFAULT_WINDOW 36, SEED_OFFSET 7919; the spec bounds in `scenarios/schema.py`;
`scenarios/calibrate.py` UNRATE_VOL_MULT 0.62; `scenarios/probabilities.py`
GAMMA 0.5, WINDOW 6, KEY_STATES; `scenarios/runner.py` ATTRIBUTION True (the
module switch for run_spec's attribution), ATTRIBUTION_BAND (10, 90);
`analysis/eras.py` ATTR_MIN_COVERAGE 0.5, OWN_BETA_MONTHS_PER_PARAM 5;
`analysis/backtest.py` WINDOWS, VIEW_VARS, PHASE_MONTHS 24, N_PATHS 400,
SEED 11, CUSTOM_PATHS 200, CUSTOM_PATHS_RANGE (100, 400), CUSTOM_MONTHS_RANGE
(12, 180), MIN_TRAIN_MONTHS_CUSTOM 180, SETUP_CACHE_MAX 8;
`analysis/forecast.py` FIRST_ORIGIN 1984-12, KEY_STATES, ENS_GRID, SPLIT
2004-12, N_PATHS 1000, SEED 23, SKILL_MIN_T 1.0, SKILL_REFS (uncond, naive),
LEDGER_DIR (data/ledger), LEDGER_SCHEMA 1, LEDGER_TAUS (100 levels),
LEDGER_METHODS, LEDGER_MAX_DELAY_MONTHS 3; `analysis/walkforward.py`
MIN_GAIN 0.003, MAX_ACCEPT 3, ITERATIONS; `portfolio/service.py`
HORIZON_BOUNDS (10, 40), DEFAULT_HORIZON 30, CONTINUATION_FILE,
CONTINUATION_CMA "valuation_state", CONTINUATION_SEED crc32("continuation"),
CONTINUATION_CHUNK 600 (and `continuation_chunk_seed()`); `server.py`
BACKTEST_CACHE_MAX 32, BACKTEST_MAX_RUNNING 2.

## EngineContext (context.py)

```python
ctx = EngineContext.load(selection=True, spec=None, cma_method="valuation_adjusted")
ctx.panel    # DataFrame: panel + e_* shock columns (processed/structural_shocks.csv) when present
ctx.meta     # processed/panel_meta.json
ctx.spec     # ModelSpec: results/model_selection.json when selection=True and it is readable, else
             # ModelSpec() (the 10 core factors); shock columns the panel lacks are dropped
ctx.model    # AssetFactorModel(spec) fitted on MODEL_START..end
ctx.db       # MonthDB(panel, model): the months the simulator draws
ctx.anchor   # Anchor at the last panel month (levels, last 12 CPI changes, trailing income yields)
ctx.cma      # capital_market_assumptions(db, anchor, method=cma_method)
ctx.fit(spec, start, end) -> AssetFactorModel     # helper for backtests
get_context() / reset_context()                    # the server's cached process-wide context
```

## Engine pipeline

`scenarios.runner.run_spec(ctx, spec, n_paths=600, seed=0, source="builtin",
attribution=None)` runs validate_spec -> `pool_phase` per phase ->
`with_oil_lock` -> `simulate` -> `summarize` -> `attribution_summary`.
Built-ins use the seed crc32(id) (`library.seed_for`); custom runs default
to 7. `attribution=None` means the module switch `runner.ATTRIBUTION` (True:
built-ins, custom and LLM runs through the API, and the studies that call
library.run all get `assets[a].attribution`, about 30 ms per 600-path run);
a study that needs only the returns can set it to False, as it swaps
`runner.RESIDUALS` and `runner.VALUATION`. Every simulated number is
bit-identical either way.

### Transmission model (engine/factor_model.py)

`ModelSpec(factors=<10 core>, lags=0, regime=None | "inflation", kind="linear"
| "mlp", ridge=0.0, window_years=None, target="p", shock_cols=(), hidden=16,
name="base")`. One regression per model asset (every asset but cash) of the
month's price return on the same month's factor changes. `regime="inflation"`
adds interactions with a dummy for CPI YoY > 4% in the previous month; `mlp`
is a 3-seed ensemble of one tanh hidden layer. Treasury, crude oil and cash
are fitted for the transmission charts only; simulations build them
structurally. The walk-forward (analysis/walkforward.py) selected F:
11 factors (the core without labor share, plus GZ and the term-premium x
high-core-inflation term), linear, no lags, full sample, with per-asset
equations for energy (+ metals/oil) and gold (fitted from 1982-01)
(model_selection.json; see "The missing-factor round").

### MonthDB (engine/simulate.py)

The months from MODEL_START that have every non-optional `f_` column, the
model's base columns, its residuals and the previous month's ff / y10 /
spread levels. It holds `frame` (every panel column, for views), `factors`
(the `f_` changes incl. f_cpi), `resid` (the fitted model's residuals),
`prev_levels`, `eras`, `rare` (RARE_MONTHS factors), `prior()` (uniform x
rare, optionally decayed toward recent months), `uncond_frequency()` (each
month's share of an unconditional 15-year path's draws: the weight of every
"historical average" the engine projects forward), `oil_log_real_median` and
`oil_real_drift`.

### Phase weights: entropy pooling over block starts (engine/pooling.py, engine/ep.py)

For a phase of n months, B = min(BLOCK_MONTHS, n, T):

1. **Pooling frame.** One row per feasible block start s (T - B + 1 rows). A
   view variable's value at s is the expected mean of the months the phase
   draws from s: the B-month mean of x[s:s+B], blended with the shorter last
   block when n is not a multiple of B, ((n // B)·B·full + (n % B)·part) / n.
   Starts that would draw a month where a view variable is missing are left
   out of that phase: a view is judged only where it is observed.
2. **Prior over starts** (`start_prior`): block starts weighted by the month
   prior, each multiplied by the smallest RARE_MONTHS factor among the months
   it would draw, so a rare month is not diluted by the ordinary months of its
   block.
3. **Solve.** `ep.apply_views` solves min KL(q || p) subject to the views (the
   dual over standardized constraints, L-BFGS-B; converged when the largest
   violation is below 1e-4). The sampler draws starts from q, so the expected
   mean of the drawn months equals each target, up to Monte Carlo noise.
   Pooling single months instead would let a view on a fat-tailed flow be met
   by a few spike months whose block neighbours undo it.
4. **View forms** (in the column's unit; monthly for `f_` columns):
   `mean: x` -> E_q = x; `mean_annual: x` -> E_q = x / 12; `shift_sd: k` ->
   prior mean + k x prior sd (the sd of the start values, i.e. of phase
   averages); `min` / `max` (and `min_annual` / `max_annual`) -> inequalities on
   the mean; `vol_mult: k` (item 6). Only the first of mean, mean_annual,
   shift_sd is used.
5. **Support clipping.** A target outside the 2nd-98th percentile of the start
   values is clipped (`clipped`, `requested` in the view report). validate_spec
   applies the same bound beforehand (`pooling.view_support` for the phase's
   own length), so the two agree.
6. **vol_mult is a second moment.** The pooling frame carries `<var>__sq` (the
   start values of x^2) and the constraint is E_q[mean of x^2 over the drawn
   months] = m^2 + k^2·(E_p[mean of x^2] - prior mean^2): the drawn months'
   monthly variance around m (the view's mean if it has one, else the prior
   mean) is k^2 times the prior's. A view with only `vol_mult` bounds the
   variance alone (the scenario page shows it as 波动 × k, with — under
   引擎采用). Without a bound a mean view tilts toward the blocks with the
   most extreme means, and the lockdown blocks (+10.4 pp unemployment in
   April 2020) are the most extreme even at their 0.25 prior weight. The
   built-ins bound f_unrate where their other views would pull those blocks
   in: energy_transition phase 1 (an oil vol_mult) and financial_crisis
   phase 1 (spread up with an oil crash, an emergency cut and falling
   sentiment) carry a variance bound alone (0.62 and 1.0); the f_unrate mean
   views carry vol_mult 0.62 in calm phases (baseline 2, financial_crisis 3,
   the calibrated phases k_shaped 1-2 and agi_boom 2), 1.0 in recessions
   (financial_crisis 2, oil_war 2) and 0.7 in japanification 1. Three mean
   views carry none (ai_bust 1, deglobalization 1, reacceleration 1). Phases
   without a bound still draw the 2020-03..05 blocks at up to about twice the
   unconditional rate (about 2x in agi_boom 3 and stagflation 1, 1.4-1.8x in
   ai_bust 1, energy_transition 2, fiscal_dominance 2 and stagflation 2).
7. **Shrink lambda.** The views as requested must converge and rest on at
   least MIN_ENS = 30 effective months (exp of the entropy of the drawn-month
   frequency) and MIN_EPISODES = 6 effective 12-month episodes (the effective
   number of 12-month calendar bins of block starts; overlapping starts
   inflate the month count, so one 2-3 year stretch alone can reach 30
   months). Both are checked on the pooled solution before any confidence
   blending, so a lower confidence cannot make the paths more repetitive.
   Otherwise every view of the phase is shrunk toward what the prior already
   satisfies: means used = prior + lambda·(target - prior), binding bounds
   likewise, variance targets base + lambda·(target - base), with lambda the
   largest value in [0, 1] that passes (14 bisection steps). `shrink_reason`
   is "infeasible" (no mix of blocks meets the views) or "thin" (feasible but
   below a threshold; `ens_requested` = the effective months of the views as
   requested). The runner adds a Chinese warning with lambda to the
   validation report. A phase where no start can be judged falls back to the
   prior (lambda = 0).
8. **Confidence.** c < 1 blends the pooled start probabilities toward the
   prior, q = c·q + (1 - c)·p (opinion pooling, `ep.blend`).
9. **Report** (`runner.ep_report`, detail `phases[].ep`): ens (effective months
   after blending), ens_ratio, block, episodes, shrink, shrink_reason,
   ens_requested, kl, converged, max_violation, views[] (var, prior_mean,
   prior_sd, kind, target, target_used, posterior_mean, posterior_sd, clipped,
   requested, vol_mult; on the pooling frame, so they describe phase
   averages), era_weights, top_months[] (date, weight_ratio = drawn share /
   prior share, weight).

### Sampling (engine/sampler.py)

Per phase, each path draws ceil(n / B) block starts from the phase's
`BlockProbs` (or from the block-start probabilities of month weights) and
concatenates the blocks; the last one is cut to the phase length. Phases are
concatenated in order.

### simulate() (engine/simulate.py)

In order:

1. **Overrides** per phase (annual values, / 12 per month). A `factor_drift` on
   ff, y10 or spread is added after level scaling, so it is delivered as asked;
   on any other simulated factor (the core factors and cpi) it is added to the
   drawn changes; on profit_share it also moves the valuation state's
   profit-share path (validate_spec allows it only while VALUATION is
   enabled). `asset_drift`, `resid_scale` and `r_star` act in steps 3 and 8.
   **Oil lock**: a phase is `oil_locked` when its views name f_oil, s_oil_yoy
   or s_real_oil (`with_oil_lock`, OIL_VIEW_VARS) or it has an oil
   factor_drift.
2. **CPI YoY** at every level date, from the anchor's last 12 actual monthly
   log changes plus the simulated ones; the rate anchors use
   pi = clip(YoY, -2, 15).
3. **Rates, spread, unemployment**, month by month:
   * level scaling of the drawn change: x clip((now + s) / (then + s), 0.3,
     1.5), s = 1.5 pp for ff and y10 and 1.0 pp for the spread, "then" = the
     level in the month before the drawn month (RATE_SCALING). A 1980 +3 pp
     month is much smaller at today's levels; months from near-zero rates are
     multiplied at most 1.5 times. A widening spread month keeps at least
     SPREAD_WIDENING_FLOOR = 0.85 of its size, and the mean change this
     asymmetric scaling would add to an unconditional spread path at the
     current level (`spread_scaling_bias` on a 0.3..10 grid) is subtracted,
     so the long-run spread does not settle above spread_level;
   * plus the phase's ff / y10 / spread drift;
   * reversion: Δff += kappa_ff·(r* + pi - ff); Δy10 += kappa_y10·(r* + pi +
     term_premium - y10); Δspread += kappa_spread·(spread_level - spread);
     Δunrate = drawn change (+ drift) + kappa_unrate·(unrate_level - unrate);
     r* is the phase's `r_star` override or the default;
   * clipped to ff 0..25, y10 0.1..20, spread 0.3..10, unrate 2..25 (%); the
     simulated changes replace the drawn ones before the transmission.
4. **Oil.** The real log oil price runs in anchor-month dollars. In months
   that are not oil-locked the drawn change gets -oil_real_drift (the sample's
   average real drift, as an unconditional path draws it) and
   -kappa_oil·(log real oil - target), target = the 1973+ median log real oil
   + log anchor CPI + `oil_target_shift`. Shift 0 is about $68 in 2026
   dollars; the schema bounds -0.7..+0.4 are about $34..$101 (the 5th and
   77th percentiles since 1973). Locked months follow the drawn months.
5. **Macro level paths** (n, T+1; index 0 = anchor): cpi, ff, y10, spread,
   unrate, cpi_yoy, oil, ip, usd, labor_share.
6. **Factor-explained price returns**: today's model on the simulated factor
   changes (lags filled from the anchor's last months; the regime dummy from
   the simulated YoY).
7. **Residuals**: `residuals.draw` with config.RESIDUALS (below).
8. **Price return** = systematic + CMA adjustment / 12 + asset_drift / 12 +
   residual x resid_scale.
9. **Valuation state**, when a ValuationModel is passed (below).
10. **Totals**: crude_oil = expm1(oil change); treasury_10y = a 10-year par
    bond bought at last month's yield and repriced at this month's
    (semiannual coupons) + coupon / 12; cash = max(ff - (anchor ff - anchor
    TB3), 0) / 1200; every other asset = price return + the anchor's trailing
    income yield / 12 (or the path income). Totals are floored at -95%.
    Output: `SimOutput(total (n, T, A), income, macro (n, T+1), idx (n, T),
    assets, start, attribution)`, float32 (attribution None unless asked).
11. **Attribution** (`simulate(..., attribution=True)` only; off by default, so
    backtests, the forecast, the ablation and the valuation study pay
    nothing). It only reads arrays the steps above already built:
    `SimOutput.attribution = Attribution(terms, factor_terms, values (n, P, A, K),
    total (n, P, A), phase_months, assets, kinds, method, reason)`, `values` =
    per path and phase the mean monthly contribution x 12, `total` = the phase
    mean of the floored float64 totals x 12, and the terms sum to `total`
    exactly (6.7e-16 at most on the 13 built-ins). Terms of a model asset
    (`kinds` "model"): one per transmission feature = beta x the SIMULATED
    factor change (after level scaling, reversion, the oil pull or lock and the
    factor drifts, so a factor drift shows up in its factor's term), then
    `intercept`, `cma` (the CMA adjustment), `overrides` (asset_drift only),
    `valuation_pull` (only with a ValuationModel of kappa > 0), `residual`
    (x resid_scale), `income` and `floor` (the -95% clip, 0 in practice).
    Structural assets: treasury_10y = `y10` (par-bond repricing) + `income`,
    cash = `income`, crude_oil = `oil`. A spec with kind mlp, lags or a regime
    gets one `unattributed` term with a Chinese reason
    (`simulate.attribution_method`). Every simulated number is bit-identical
    with or without it.

`DEFAULT_REVERSION` (monthly pull shares and percent levels; a spec's
top-level `reversion` overrides it within the schema bounds):

| Key | Default | Bounds | Meaning |
|---|---|---|---|
| kappa_ff | 0.02 | 0..0.1 | policy rate toward r* + inflation |
| kappa_y10 | 0.015 | 0..0.1 | 10-year yield toward r* + inflation + term_premium |
| kappa_spread | 0.03 | 0..0.1 | credit spread toward spread_level |
| kappa_unrate | 0.015 | 0..0.1 | unemployment toward unrate_level |
| kappa_oil | 0.015 | 0..0.1 | log real oil toward its target |
| r_star | 0.75 | -3..3 | neutral real policy rate (a phase's `overrides.r_star` replaces it there) |
| term_premium | 0.75 | -1..3 | |
| spread_level | 2.0 | 0.5..6 | |
| unrate_level | 4.8 | 3..10 | |
| oil_target_shift | 0.0 | -0.7..0.4 | log shift of the real-oil target; acts only on phases without an oil view or oil drift |

### Capital-market assumptions (simulate.capital_market_assumptions)

An annual drift adjustment per model asset, added to the price return.
`historical`: none (the sample-average drift). `valuation_adjusted` (the
default: ctx.cma, the scenarios, the one-year forecast): the market loses the
sample's re-rating tailwind (the average Δlog P/D) and gains
reversion·ln(median P/D / P/D now) / horizon_years (reversion 0.3 over 15
years); every other equity gets beta x that market adjustment minus
shrink_relative (0.5) x its drift relative to the market (net of the oil
factor's real drift, which the simulator already removes); gold loses its
sample real appreciation and keeps pace with inflation (gold_reversion 0: no
pull toward its median real price); treasury, crude oil and cash get none.
`valuation_state`: valuation_adjusted without the constant reversion term,
for the valuation-state mode. Every average is weighted the way an
unconditional path draws the months. Current values (meta.json `cma`): P/D now
97.4, median 51.6, re-rating tailwind +1.71%/yr, reversion term -1.27%/yr,
market adjustment -2.98%/yr. Backtest fans use historical, re-run with
valuation_adjusted for comparison (`fan_cond_va`).

### Residual modes (engine/residuals.py)

`simulate(resid_mode, resid_window, resid_coupling, resid_seed)`:
`coupled` (each drawn month's own residual), `decoupled` (residuals from an
independent block bootstrap under the prior, seed + 7919), `detrended` (the
drawn residual minus a centered rolling mean estimated on the MonthDB months,
default 36, plus the constant that restores the prior mean),
`trend_decoupled` (the fast part coupled, the slow part from the independent
draw), `partial` (per path and sampler block, the coupled residual with
probability `coupling`, otherwise the decoupled one), `recentered` (coupled,
shifted per phase by (1 - coupling)·(E_q[resid] - E_prior[resid])). Every mode
keeps the coupled draw's unconditional expected residual. config.RESIDUALS =
partial 0.5: a view that selects a few episodes brings along only half of
their era-specific drift (bubbles, busts). That is why the built-ins state
valuation busts as explicit asset_drift (-0.08 on us_equity and nasdaq in
ai_bust phase 1, on those plus financials and real_estate in
financial_crisis phases 1-2, and -0.08 / -0.06 on the same four in
japanification phases 1-2): the 2000-02 and 2007-09 falls were mostly
residual, and the low-rate months japanification draws bring a re-rating of
their own.

### Valuation state (ValuationModel, fit_valuation)

log P/D_{t+1} = log P/D_t + log(1 + market price return_t) - g_t, with g_t
the drawn month's trailing-dividend growth (log1p(p_us_equity) -
f_valuation). The fair value ln P/D* = a + b·real_y10 + c·profit_share +
d·infl_expect is fitted on the MonthDB months (Newey-West, 60 lags) and moves
with the simulated levels: infl_expect by the same EWMA recursion on the
simulated CPI, real_y10 = y10 - infl_expect, profit_share by the drawn
months' published changes (plus a profit_share drift). With kappa > 0 equity
prices get -kappa / 12·(ln P/D - ln P/D*) a month, scaled by each asset's beta
to the market (gold, treasury, crude oil and cash: none).
`runner.valuation_for(ctx)` picks the mode: `enabled` -> the pull, the CMA of
VALUATION["cma"], macro `pd` and `pd_fair`; off with `track` (the default) ->
`fit_valuation(kappa=0, income="constant")` on ctx.db, returns bit-identical,
macro `pd` only; neither -> no state. A panel without the valuation columns
runs untracked. The states are cached per MonthDB; with the pull off the CMA
is always the caller's `ctx.cma`, so a context that shares the db but carries
another CMA (the ablation's historical-CMA scenario run) gets its own drift.

### Summaries (runner.summarize, engine/summary.py)

Wealth paths from the totals (index 0 = 1); real paths deflated by each
path's own CPI. The representative path is the one whose us_equity terminal
value and max drawdown ranks are both closest to the median; its monthly
candles have exact opens and closes and Brownian-bridge wicks (21 steps, the
asset's monthly vol), and yearly candles aggregate calendar years. Fans are
the 5/25/50/75/95th percentiles. `corr` pools every path and month;
`corr_prior` is the prior-weighted correlation of historical total returns.

### Scenario probabilities (scenarios/probabilities.py)

posterior ∝ prior_prob x L^gamma (gamma = 0.5), L = exp(-1/2·mean_k z_k^2),
z_k = (today_k - E_q[s_k]) / sd_k over KEY_STATES (cpi_yoy, ip_yoy, unrate,
ff, spread, oil_yoy, y10); today = the mean of the last 6 months, E_q over the
months the first phase draws, sd over the MonthDB. A transparent heuristic:
the data tilt the subjective priors but cannot overrule them.

## Scenario spec (scenarios/schema.py): YAML and LLM output

One schema serves the built-in YAML files, specs edited in the lab and the
LLM. `validate_spec(spec, frame, horizon=180) -> (clean_spec, report)` is the
guard rail between a narrative and the engine: it rejects unknown variables,
bounds every number to what history can support, pads or trims the phases and
reports every change. An illustrative YAML spec with every field (values taken
from several built-ins):

```yaml
id: energy_transition            # slugified to [a-z0-9_]+, <= 40 chars (default: the YAML file name)
name: 能源转型与气候冲击          # <= 60 chars
summary: 一句话概括               # <= 400 chars
narrative: |-                    # <= 6000 chars
  起因、传导机制、与哪些历史时期相似、哪些部分超出历史
tags: [能源转型, 气候]            # <= 8 tags of <= 24 chars
prior_prob: 0.05                 # 0..1 (default 0.05); renormalized across the scenario set
analogs: ["2021-03:2022-09"]     # <= 8 windows YYYY-MM:YYYY-MM; shown in the UI, never used for the weights
causal_chain:                    # <= 30 edges; from / to <= 40 chars, lag <= 20, mechanism <= 300
  - {from: 上游投资不足, to: 油价, sign: "+", lag: "12-36月", mechanism: "供给投资下降"}   # sign: + | - | ±
phases:                          # <= 8, in time order; the months are padded / trimmed to 180
  - name: 绿色通胀               # <= 30 chars
    months: 36                   # integer >= 1 (invalid -> 12, with a warning)
    views:                       # <= 12 per phase: entropy-pooling views on MonthDB columns
      - {var: f_oil, mean_annual: 0.10, vol_mult: 1.3, why: "油价每年约 +10% 且波动更大"}
      - {var: s_cpi_yoy, mean: 3.8}
      - {var: e_oil_supply, shift_sd: 1.0}
      - {var: f_unrate, vol_mult: 0.62}       # a variance bound alone
    overrides:                   # optional, bounded
      factor_drift: {unrate: 0.13}            # per year, in the factor's unit
      asset_drift: {energy: -0.02}            # per year, added to the price return
      resid_scale: 1.2
      r_star: -1.0
    confidence: 1.0              # 0.3..1.0 (default 1.0)
reversion:                       # optional, spec level: overrides of simulate.DEFAULT_REVERSION
  oil_target_shift: -0.35        # -0.7..0.4: log shift of the real-oil target (0 = the 1973+ median, about $68)
calibration:                     # optional, built-in library only (see below)
  - {kind: distribution, phase: 0, params: {labor_income_shock: {mid40: -0.05}, equity_wealth_change: 0.25}}
```

`reversion` takes any key of the DEFAULT_REVERSION table with its bounds;
`oil_target_shift` acts only on phases without an oil view or oil drift
(energy_transition uses -0.35, the 1986-2003 median of about $48). The other
fields and their bounds:

| Field | Rule |
|---|---|
| view `var` | starts with `s_`, `f_` or `e_` and is a MonthDB column with history; anything else is dropped with a warning |
| `mean`, `min`, `max` | clipped to the 2nd-98th percentile of what the phase can average (`pooling.view_support`: means of B-month runs, much narrower than single months) |
| `mean_annual`, `min_annual`, `max_annual` | the same bounds x 12 (the value / 12 is the monthly target); `min_annual` / `max_annual` exist in YAML only |
| `shift_sd` | -3..3 prior sds of the phase average |
| `vol_mult` | 0.3..3 |
| `why` | <= 200 chars |
| a view with no target | dropped with a warning |
| endogenous variables | a view on f_valuation, f_wealth, f_cape or s_cape warns that it dictates equity prices |
| IDENTITIES | s_real_ff = s_ff - s_cpi_yoy and s_term_spread = s_y10 - s_ff: when a phase gives a mean-type target (mean, mean_annual or shift_sd) for all three, the derived one is dropped |
| `factor_drift` | only `drift_factor_ids(model_spec)`: the active model's factors and extras, cpi and labor_share (plus profit_share while VALUATION is enabled; without a model spec the 10 core factors and cpi; see "The missing-factor round" for W12-W14). Other FACTORS ids are dropped with a warning (they only pick months; a drift on them changes no path), unknown ids too. Bound: +-2 x the factor's annualized monthly-change sd on the MonthDB (llm_translator.json `factor_drift_bounds`; currently unrate +-3.24 pp/yr, oil +-0.71 and labor_share +-0.024 log/yr) |
| `asset_drift` | equity, REIT and gold assets, +-0.08 per year; cash, treasury_10y and crude_oil are structural and dropped |
| `resid_scale` | 0.5..2.0 |
| `r_star` | -3..3 |
| `confidence` | 0.3..1.0 |
| `reversion` | an unknown or non-numeric key warns and is dropped; a clipped value warns and is listed with `phase: null` |
| phases | more than 8: the rest dropped; total > horizon: trimmed from the back (each phase keeps >= 1 month); total < horizon: a view-free phase 回归常态 is appended |
| other top-level keys | dropped; kept: id, name, summary, narrative, tags, prior_prob, analogs, causal_chain, phases, reversion, source |

Only a `phases` value that is not a list, or a phase that is not an object, is
an error (`report.ok` false; the custom-run API refuses such a spec); every
other fix is a warning. Report (an actual run on the current build):

```json
{"ok": true, "errors": [],
 "warnings": ["阶段 1：s_cpi_yoy 的 mean=14 超出历史可支撑范围，截断为 12.32",
              "阶段 1：因子 unrate 的额外漂移 5/年 截断为 3.241",
              "阶段 1：资产 nasdaq 的额外漂移 0.1/年 截断为 0.08",
              "均值回归参数 oil_target_shift=-1 超出边界，截断为 -0.7"],
 "clipped": [{"phase": 1, "var": "s_cpi_yoy", "key": "mean", "requested": 14.0, "used": 12.3223},
             {"phase": 1, "var": "factor_drift.unrate", "requested": 5.0, "used": 3.2407},
             {"phase": 1, "var": "asset_drift.nasdaq", "requested": 0.1, "used": 0.08},
             {"phase": 1, "var": "resid_scale", "requested": 3.0, "used": 2.0},
             {"phase": null, "var": "reversion.oil_target_shift", "requested": -1.0, "used": -0.7}],
 "horizon_months": 180, "default_reversion": {"kappa_ff": 0.02, "…": "…", "oil_target_shift": 0.0}}
```

`resid_scale` / `r_star` clipping is listed in `clipped` without a warning.
run_spec appends the shrink warnings of the phase pooling to the same
`warnings`. The UI shows the report next to every result (a `phase: null` row
shows — as its phase).

**LLM form** (`SPEC_JSON_SCHEMA`, used as the structured-output schema). The
same fields with these differences: every view carries all of var, mean,
mean_annual, shift_sd, min, max, vol_mult and why (null when unused; no
`*_annual` bounds); overrides always have the list form
`factor_drift: [{factor, annual, why}]`, `asset_drift: [{asset, annual, why}]`,
`resid_scale`, `r_star` (null when unused), which `_normalize_overrides` turns
into the dict form; there is no `reversion` and no `calibration`, so the
translator cannot set either. Numeric bounds are not in the JSON schema
(structured outputs do not support minimum / maximum); validate_spec enforces
them. Before validation `translator.repair()` prefixes the id with `llm_`,
sets `source: "llm"`, reads a prior_prob or confidence between 1.5 and 100 as
a percentage, merges duplicate views of a variable, turns mean_annual on a
non-`f_` variable into mean, keeps only the first of mean / mean_annual /
shift_sd, and warns about valuation variables and overrides without a reason.

**Calibration block** (built-ins only: `library.resolve_calibration`, then
`calibrate.distribution_views`). `calibration: {kind: distribution, phase:
<0-based index>, params: {...}}`, or a list of them. Params (defaults in
`calibrate.DEFAULTS`; group dicts are keyed bottom50 / mid40 / top10):
consumption_share, labor_income_shock, mpc_labor, labor_income_ratio,
equity_ownership, worker_share, wealth_mpc (0.03), equity_wealth_change,
equity_wealth_to_consumption (2.7), consumption_gdp_share (0.68),
job_loss_fraction (0.5), okun (0.5), ip_beta (1.7), services_slope (0.2);
`months` is the phase's length, and unknown keys are listed in
`impact.ignored_params`. The calculator's result enters that phase as views
that replace same-variable views there: s_ip_yoy (mean: today's 12-month
average + ip_beta x growth), s_services_yoy (mean), f_unrate (mean_annual = the
Okun part, vol_mult = UNRATE_VOL_MULT 0.62: the prior's monthly sd of f_unrate
is 0.281 pp with 2020-03..08 at 0.25x, and 0.62 x 0.281 = 0.175 pp, the sd
without 2020) and f_labor_share (mean_annual 0, so the labour-share shift comes
only from the drift); plus the factor drifts unrate (direct job losses /
years) and labor_share (log change / years), added to the phase's own drifts
(k_shaped phase 2 adds its own unrate +0.13 on top). The detail JSON gets
`calibration[]` (see "Results files"). validate_spec drops the key, so a spec
POSTed to the API must carry the resolved views; the lab's templates come from
`detail.spec`, which does. Calibrated built-ins: k_shaped, agi_boom.

## Results files

Everything lives in data/results/ and is served as `/api/<name>` (see "HTTP
API"; model_selection.json is not served). Units follow "Conventions"; every
analysis file also carries its own `units` and `notes`. Shapes below use
`key: type` notation; `{asset: …}` means one entry per asset id.

### meta.json (build.build_meta)

```text
{built_at: "2026-11-01T17:50:45Z", data_start: "1947-01-01", data_end: "2026-09-01",
 model_start: "1973-01", horizon_months: 180,
 assets: [config.ASSETS], factors: [config.FACTORS], states: [config.STATES], eras: [config.ERAS],
 today: {date: "2026-09", states: {state: level}, income_yield: {asset: trailing 12-month income, decimal}},
 model: {spec, describe, train_start, train_end, r2: {asset}, resid_sd: {asset},
         coef: {asset: {feature}}, tstat: {asset: {feature}}, intercept: {asset}},
 cma: {method, reversion, horizon_years, shrink_relative, gold_reversion,
       historical_price_drift: {asset}, adjustment: {asset}, beta_to_market: {asset}, relative_drift: {asset},
       gold: {real_drift_hist, real_gap_log, reversion_term},
       pd_now, pd_median, rerating_tailwind, reversion_term, market_adjustment, target_price_drift: {asset}},
 notes: [str], sources: {name: url}, asset_start: {asset: "YYYY-MM-DD"},
 llm: {available, model, reason}}
```

Drifts and adjustments are annual decimals, P/D values multiples. The
method page builds its step 7 (估值锚定) from `cma` (reversion, horizon_years,
shrink_relative, gold_reversion). `sources` is `data.panel.meta_sources(panel_meta)`:
`panel.SOURCES` (FRED, Ken French Data Library, FTSE Nareit, 「Gold Prices（DataHub
核心数据集，月度金价）」, 「Robert Shiller 的 ie_data.xls（席勒 CAPE）」) plus any
other entry of panel_meta.json, without the legacy gold label, so meta.json is
current even when panel_meta.json predates a SOURCES change.

### eras.json (analysis/eras.py)

```text
{reference: {start: "1955-01", end},
 eras: [{id, name, start, end, ongoing, summary, n_months,
         states: {state: {mean, start, end, coverage}}, factors: {factor: {mean_annual, vol_annual}},
         assets: {asset: {cagr, real_cagr, vol, max_dd, avg_income}}, corr: {assets, matrix},
         stock_bond_corr, oil_cpi_passthrough, oil_cpi_passthrough_se, oil_price_moves,
         phillips_slope, phillips_measure, signature: [{state, name, z, text}], narrative,
         attribution: {method, n_months, coverage, months: [start, end] | null, units, reason?, missing?,
                       assets: {asset: {kind: model | cash, terms: {<factor ids>, intercept, residual, income}
                                        (cash: {income}), factor_sum, r2_in_era,
                                        own_beta: {terms, factor_sum, intercept, r2, n_months} | null,
                                        price_mean, total_mean, cagr, reconciliation, n_months}}}}],
 attribution_info: {method, reason, beta: "full_sample", model: {spec, describe, train_start, train_end},
                    factor_terms, min_coverage: 0.5, own_beta_min_months: 55, units},
 fingerprint: {rows, row_names, cols, col_names, col_units, ref_mean, ref_sd, z: [[…]], values: [[…]]},
 today_similarity: {date, window, database, n_months_db, excluded_eras, method: "soft_entropy_pooling",
                    ens_min: 60, tau, replayed_by_engine, weights: {era: share}, ens, ens_ratio, kl,
                    views: [{var, state, name, today, target, prior_mean, posterior_mean}],
                    residual: [{state, name, unit, in_view, today, matched, gap, gap_z}],
                    top_months: [{date, era, weight}], interpretation},
 timeline: {dates, names: {state: name}, series: {state: […]}, bands: [{id, name, start, end}]},
 notes: [str]}
```

z = (era mean - 1955+ mean) / 1955+ sd. today_similarity matches the last 12
months against 1955 up to 24 months ago and before the ongoing era, with
softened views (at least 60 effective months); `residual` lists what history
cannot reproduce.

`eras[].attribution` (传导归因) splits each era's mean annual price return
(mean monthly x 12) into the selected model's full-sample beta x the era's
actual mean factor changes (one term per factor), `intercept` and `residual`;
`price_mean` is their sum over the months that have every feature, `+ income`
gives `total_mean`, and `reconciliation` = `cagr` over the same months -
`total_mean`. `r2_in_era` is the full-sample model's R^2 inside the era (it
can be negative). `own_beta` refits the model on the era alone (only eras
with at least 55 months, 5 per coefficient); its intercept absorbs the factor
means, so its split is a comparison only (科技繁荣 美股大盘: factors -22.5%,
intercept +46.5%, R^2 0.32). An era needs every feature in at least half of
its months: 战后黄金期 and 通胀起步 have none (`reason`, `missing: ["usd"]`:
the trade-weighted dollar starts in 1973-02), and 大通胀 covers 119 of its 120
months (no 1973-01), so its `cagr` differs a little from the asset table's.
treasury_10y and crude_oil are split by the regression here (`kind` model),
although the scenario engine builds them structurally. The step fits the
model of model_selection.json (else the base spec) and records it in
`attribution_info.model`.

### causality.json (+ processed/structural_shocks.csv)

```text
{svar: {variables: [{id, name, unit}],                    # oil, ip, cpi, ff, spread
        shocks: [{id, name, restrictions: [{var, sign, horizon: [h0, h1]}], text}],   # oil_supply, demand, monetary, financial
        lags: 6, horizon: 36, periods: [{id, name, start, end}],   # full, oil_era, great_moderation, post_gfc
        irf: {period: {shock: {var: {p16, p50, p84}}}}, n_accepted: {period: 500}, draws: {period: {…}},
        normalized: {period: {…}}, median_target: {period, draw, n_rotation_draws, acceptance_rate, irf, impact},
        shock_corr: [[…]], excluded: {months, rows, …}, method},
 historical_decomposition: {dates, variable: "cpi", unit, actual, baseline,
        contrib: {oil_supply, demand, monetary, financial, other, covid}, names},
 distribution: {shock, horizon_q, periods: [{id, name, start, end, shock_start, shock_end}],   # 1960-1999, 2000-today
        variables: [{id, name, unit}], responses: {period: {var: {coef, lo, hi, se}}},
        n_obs, shock_sd_pct, band, method, alternative},
 asset_betas: {periods, factors, assets, beta_std: {period: [[…]]}, tstat, r2: {period: {asset}}},
 chains: [{id, title, nodes: [{id, label, group}], edges: [{from, to, sign, label, evidence}]}]}
```

IRFs are per 1-sd shock (cumulative % for oil, ip, cpi; pp for ff and
spread); the decomposition is in pp of 12-month CPI inflation; the labour-share
local projections are scaled to a 1% decline. structural_shocks.csv:
`date,e_oil_supply,e_demand,e_monetary,e_financial` (monthly, unit variance;
2020-03..06 and the rows that use them as lags are blank).

### diagnostics.json (analysis/diagnostics.py) and model_selection.json

```text
diagnostics: {questions: {missing_factor, missing_mechanism, tuning, neural_net},
  candidates: [{id, group, question, hypothesis, change, eligible, note}],          # 22 candidates
  score_assets, min_gain: 0.003, max_accept: 3, require_test_gain: true, r2_benchmark: "training_mean",
  units, notes,
  iterations: [{iter, train, validate, test, start, baseline, findings, trials: [{round, candidate, …, val_score, delta}],
                selected, test_result: {base, selected}, test_ablation: [{candidate, val_delta, test_score, test_delta}],
                carry_over}],
  final: {spec, describe, changes, is_base, test_used_for_choice, val_score, test_score, test_months,
          test_per_asset, base_val_score, base_test_score, reason, last_selected, selected_on, trained_on},
  residual_analysis: {spec, describe, sample, break_date, acf_lags, assets, leakage, rolling_betas, summary},
  recommendations: [{priority, group, question, title, detail, evidence}]}
model_selection: {spec: ModelSpec dict, describe, source: "walkforward", changes, val_score, test_score,
  base_val_score, base_test_score, test_used_for_choice, r2_benchmark, reason,
  selected_on: {train, validate, test}, created}
```

Scores are mean out-of-sample monthly R^2 of price returns over the 10
non-structural assets, with SST around the training mean (> 0 beats the
historical mean). Iterations: train 1973-1994 / validate 1995-2004 / test
2005-2014, then train 1973-2004 / validate 2005-2014 / test 2015-today. The
current final spec is F (「11 个因子、能源股 +1 个专门因子、黄金从 1982 年起估计」); the
candidates, the incremental gate, stage B and the new keys are in "The
missing-factor round". model_selection.json is read by
EngineContext only; the UI sees the model through `meta.model` and
`diagnostics.final`.

### backtest.json (analysis/backtest.py)

```text
{assets, asset_names, risk_assets,                      # risk assets = equities, REITs, gold
 spec, describe, n_paths: 400, seed: 11, view_vars,     # the 8 views: s_cpi_yoy, s_ip_yoy, f_ff, f_y10, f_spread, f_oil, f_unrate, f_usd
 units, notes,
 shocks: config.EXOGENOUS_SHOCKS [{start, end, name, kind}],
 custom: {paths: 200, paths_range: [100, 400], months_range: [12, 180], min_train_months: 180,
          earliest_train_end: "1987-12", last_month: "2026-09"},          # the bounds of POST /api/backtest/run
 transmission: {dates, start, end, oos_start, refit_every: 60, refits, basis: "conditional_median",
                r2_benchmark, structural, series: {asset: {actual, fitted, fitted_oos}},
                r2: {asset}, r2_oos: {asset}, r2_in_sample_oos_months: {asset},
                r2_risk_avg, r2_oos_risk_avg, r2_in_sample_oos_months_risk_avg},
 windows: [{id, name, train: [start, end], test: [start, end], phase_months: 24, phases, n_paths, dates,
            assets: {asset: {dates, actual, oracle, fan_cond, fan_uncond, fan_cond_va: {p5, p25, p50, p75, p95},
                             cagr_actual, cagr_oracle, r2_oracle,
                             coverage_cond, pit_terminal_cond, cagr_pred_cond_p50,
                             coverage_uncond, pit_terminal_uncond, cagr_pred_uncond_p50, …_va,
                             decomposition: {total, scenario, engine, transmission, *_annual}, decomposition_va}},
            corr: {assets, realized, posterior, prior, mae_posterior, mae_prior},
            macro: {cpi_yoy | ff | y10 | unrate: {actual, cond, uncond_p50}},
            ep: [{phase, ens, ens_ratio, kl, converged, max_violation, block, shrink, shrink_reason,
                  ens_requested, episodes, clipped, views, era_weights, top_months, max_abs_drawn_gap_sd}],
            view_gap_sd, model_r2_train, shocks: [the EXOGENOUS_SHOCKS its test months overlap], summary: {…}}],
 summary: {windows: [{id, name, test, coverage_cond_risk, coverage_uncond_risk, cagr_mae_cond_risk,
                      cagr_mae_uncond_risk, scenario_abs_annual, engine_abs_annual, transmission_abs_annual, …}],
           coverage_cond_risk, coverage_uncond_risk, cagr_mae_cond_risk, cagr_mae_uncond_risk,
           scenario_abs_annual, engine_abs_annual, transmission_abs_annual,
           pit_tail_share_cond, n_pit, text}}
```

Windows (model, MonthDB and anchor see only the training months):
tech_bubble (train to 1994-12, test 1995-01..2004-12), gfc_qe (2004-12;
2005-01..2014-12), covid_ai (2014-12; 2015-01..today), post_covid (2019-12;
2020-01..today). `fan_cond` gets 24-month phases whose views are the realized
test-period means of the 8 view variables ("had we known the macro path");
`fan_uncond` has no views; both use CMA historical. Indexes start at 1;
coverage = share of test months inside the 5-95% band; MAE = mean absolute
error of the median CAGR over the risk assets; decomposition in log points of
the terminal value (`*_annual` per year): scenario = uncond - cond, engine =
cond - oracle, transmission = oracle - actual.

Custom windows (the 回测 tab's 自选时间段, `POST /api/backtest/run`) run the
same `run_window` on any training cutoff from 1987-12 (at least 15 years of
training months from MODEL_START) and 12..180 test months that end within the
panel: `custom_window` / `custom_request` validate the request (Chinese
ValueErrors), `window_setup` holds what every seed and path count of a window
shares (the test months, the phase plan, the model, MonthDB and anchor fitted
on the training months, and the pooled phase weights: nearly all of the run
time), and `run_custom` reuses it through an 8-entry cache (`SETUP_CACHE_MAX`,
keyed by train_end, months and the model spec, checked against a weak
reference to the panel; concurrent requests for one window wait for a single
setup). A custom window with a built-in window's parameters (e.g. 2004-12 +
120 months at 400 paths, seed 11) reproduces that window of backtest.json
exactly, apart from `id`, `name`, `label` and `custom`. Output: the window JSON
with `id: "custom"`, `name: "自选窗口"`, plus `label` ("自选窗口：1999-12 之后
5 年"), `custom: {train_end, months, n_paths, seed, param: "YYYY-MM:months"}`
and, from the server, `elapsed_s`. A first run takes 0.1-10 s at 200 paths
(1987-12 + 180 months is the slowest); another seed or path count of the same
window 0.1-0.4 s.

### scenarios.json, scenarios/<id>.json, scenario_paths.npz (scenarios/library.py)

Index (scenarios ordered by posterior probability):

```text
{start: "2026-09", horizon_months: 180, n_paths: 600,
 anchor: {date, ff, y10, tb3, spread, oil, cpi, unrate, usd, ip, labor_share, income_yield: {asset}},
 cma: {… as meta.cma}, probability_method: str,
 probability: {gamma: 0.5, window_months: 6, states, today: {state}, scales: {state}, date},
 scenarios: [{id, name, summary, tags, source: "builtin", prior_prob (normalized), likelihood, prob,
   headline: {us_equity_cagr_p50, us_equity_real_cagr_p50, cpi_avg (%), ff_end_p50 (%),
              pd_end_p50 (multiple: the P/D after 15 years, median path), us_equity_max_dd_p50,
              nasdaq_cagr_p50, gold_cagr_p50, treasury_10y_cagr_p50, cash_real_cagr_p50,
              y10_end_p50 (%), unrate_max_p50 (%)},
   ens_min, analogs, warnings (count), calibrated (bool),
   assets: {asset: {cagr_p5, cagr_p50, cagr_p95, real_cagr_p50, max_dd_p50}}}]}
```

Detail (`runner.run_spec` + library extras):

```text
{id, name, summary, narrative, tags, source, prior_prob, analogs,
 spec (the validated spec), validation (the report),
 causal_chain: {nodes: [{id, label}], edges: [{from, to, sign, lag, mechanism}]},
 phases: [{name, start_month, months, views, overrides, confidence, ep: {…}}],
 start: "2026-09-01", n_paths, corr_prior: {assets, matrix},
 labels: ["2026-09", …, "2041-09"],                      # T + 1 monthly labels, index 0 = the anchor
 years: ["2026", …, "2041"],
 assets: {asset: {fan: {p5, p25, p50, p75, p95},          # wealth multiple of 1 at the anchor
                  fan_real,                               # deflated by each path's own CPI
                  candles_monthly: [[label, open, close, low, high]],   # representative path, starts at 100
                  candles_yearly: [[year, open, close, low, high]],
                  stats: {cagr_p5, cagr_p50, cagr_p95, cagr_mean, real_cagr_p5, real_cagr_p50, real_cagr_p95,
                          vol, max_dd_p50, income_yield},
                  attribution: {kind: model | treasury | cash | crude_oil, method: linear | unattributed | structural,
                                units, reason?,
                                phases: [{name, start_month, months, terms: {term: annual}, factor_sum,
                                          factor_band: [p10, p90], arithmetic, cagr_p50, reconciliation}],
                                total: {months, terms, factor_sum, factor_band, arithmetic, cagr_p50, reconciliation}}}},
 macro: {cpi_yoy, ff, y10, unrate, spread (%), oil ($ a barrel, nominal), pd (multiple): {fan: {p5, …, p95}}},
 corr: {assets, matrix}, era_composition: {era: share of the drawn months}, representative_path,
 calibration: [{kind, title, text, base, impact, views, overrides, override_notes, param_notes, formulae,
                engine_mapping, caveats, phase, phase_name, start_month, months,
                engine_check: {equity_wealth_change_assumed, us_equity_price_return_p50, us_equity_total_return_p50,
                               unemployment_change_assumed_pp, unemployment_change_p50_pp,
                               unemployment_change_mean_pp, labor_share_log_change_assumed,
                               labor_share_log_change_p50}}],   # calibrated built-ins only
 probability: {prior, likelihood, posterior, distance, z: {state}, phase_means: {state}, today, method},
 prob}
```

`macro.pd` is the P/D each path implies with the valuation state tracked at
kappa = 0 (index 0 = today's 97.4); `headline.pd_end_p50` is its last p50
(baseline 83.1; 76.7 probability-weighted over the 13 built-ins). While the
pull is off there is no `macro.pd_fair`; the scenario page's P/D card then
says 仅记录，不做估值回归 and takes its historical percentiles from
valuation.json `data.pd` (if a build ever writes `macro.pd_fair`, the older
fair-value card is used). `engine_check` compares the calculator's
assumptions with what the engine simulated over that phase.

`assets[a].attribution` (传导归因, `runner.attribution_summary` from
`SimOutput.attribution`): per phase and over the whole path (`total`), the
mean over the paths of each term, in annual decimals (mean monthly
contribution x 12). The terms sum exactly to `arithmetic` (the mean monthly
total return x 12); `factor_sum` is the sum of the factor terms and
`factor_band` its 10th-90th percentile over the paths; `reconciliation` =
`cagr_p50` - `arithmetic` (compounding, volatility drag and the path median
against the mean; the page calls it 复利、波动与中位数修正, and it is positive
in short skewed phases, e.g. financial_crisis 金融股 phase 1 +5.0 pp), and
`total.cagr_p50` is exactly `stats.cagr_p50`. `overrides`, `income` and
`floor` are listed only where they move the asset (gold has no income; cash
has `income` only, treasury_10y `y10` + `income`, crude_oil `oil`). The
`intercept` term is the model's return at zero factor change and, like `cma`
and an equity's `income`, is the same in every scenario (美股大盘 +8.9%/年,
纳斯达克100 +15.0%/年), so a model asset's scenarios differ through the factor
terms, the overrides and the residual. Example,
oil_war 石油能源股 冲击期 (6 months): oil +18.7%, usd +3.3%, unrate +2.5%,
factors +22.9% (paths 10-90%: -0.9% to +45.2%), intercept +7.0%, cma -3.4%,
residual -10.9%, income +2.8% -> arithmetic +18.4%, reconciliation -1.4% ->
median CAGR +17.0%. A detail grows about 22 KB with it; custom runs
(POST /api/scenarios/custom/run) carry it too.

scenario_paths.npz: `total` (S, N, T, A) float32 monthly total returns,
`income` (S, N, T, A), `cpi` (S, N, T+1) CPI levels, `ids` (S,), `assets` (A,),
`labels` (T+1,), `probs` (S,); rows in index order. Currently S = 13, N = 600,
T = 180, A = 13.

### forecast.json (analysis/forecast.py)

```text
{horizon_months: 12, n_paths: 1000, key_states, analog_ens (chosen),
 methods: {uncond | analog | naive | scenarios: {name, description}},
 assets, asset_names, risk_assets, shocks: config.EXOGENOUS_SHOCKS, notes,
 selection: {grid: [{ens, val_crps_analog, val_crps_uncond, val_crps_naive, val_coverage,
                     test_crps_analog, test_crps_uncond, test_crps_naive, test_coverage, test_rank_ic}],
             chosen, split: "2004-12", rule, gain_val, gain_test, text},
 origins: [{date, end, shocks: [name], inflation, analogs: [{date, weight, distance}], bandwidth,
            assets: {asset: {realized,
                             uncond | analog: {p5, p25, p50, p75, p95, mean, prob_loss, pit, crps},
                             naive: {p5, p50, p95, mean, pit, crps}}},
            macro: {unrate | ff | cpi_yoy | spread | oil: {realized, uncond | analog | naive: {p5, p50, p95, mean, pit, crps}}}}],
 scores: {all | ex_shock: {uncond | analog | naive: {assets: {asset: {n, coverage90, pit_mad, crps, mae_p50, bias_p50}},
                                                    risk_avg: {coverage90, pit_mad, crps, mae_p50}, all_avg: {…},
                                                    direction: {hit_rate, base_rate_up, n},
                                                    rank_ic: {mean, t, n, by_origin}},
                           crps_skill_risk: {analog_vs_uncond, analog_vs_naive, uncond_vs_naive}}},
 n_shock_years, misses: [{origin, asset, realized, p5, p50, p95, z, cause: shock | macro | pricing, why}],
 text,
 macro_skill: {rule, split: "2004-12", min_t: 1.0, n_origins: 42, refs: ["uncond", "naive"], skilled: [ids],
               vars: {k: {name, unit, skilled, beats: {uncond, naive}, coverage90: {analog, uncond, naive},
                          all | val | test | ex_shock: {n, crps_analog, crps_uncond, skill, t},
                          naive: {all | val | test | ex_shock: {n, crps_analog, crps_naive, skill, t}}}}},
 live: {origin, state: {key_state}, z_today, analog_ens, bandwidth, display_ens,
        analogs: [{date, weight, distance, next12: {asset}, macro}],
        assets: {asset: {uncond | analog | scenarios | scenarios_calibrated: {p5, p25, p50, p75, p95, mean, prob_loss,
                                                                            real_p50}}},   # scenarios_calibrated only when calibrated
        macro: {key: {uncond | analog: {p5, p50, p95, mean}}}, scenario_medians: {asset: {id}}, scenario_probs: {id},
        scenario_probs_calibrated: {id: q} | null,
        calibration: {available, reason, rule, verdicts, candidates, skilled, prior: {id},
                      probs, kl, ens, ens_prior, converged, max_violation, shrink,                   # when available
                      views: [{var, name, target, target_used, prior_mean, posterior_mean}],
                      shifts: [{id, name, prior, calibrated, shift}], text},
        reconciliation: {assets, n_paths, paths_reproduced, not_reproduced, missing, text: [paragraphs],
                         summary: {asset: {analog | scenarios | calibrated: {p50, mean, prob_loss} | null,
                                           gap_p50, gap_mean, gap_prob_loss, calibrated_gap_p50?}},
                         contrib_mean | contrib_prob_loss: {asset: {id: p_s x (scenario value - analog value)}},
                         scenarios: [{id, name, prob, prob_calibrated, year1: {asset: {p50, mean, prob_loss}},
                                      cagr15_p50: {asset}, macro: {k: mean 12-month change}, front_load: {asset}}],
                         macro: {k: {name, unit, analog_mean, uncond_mean, scenarios_mean, calibrated_mean, skilled,
                                     skill, t, skill_naive, t_naive}},
                         sensitivity: [{vars, skilled, kl, ens, converged, shrink, us_equity: {p50, mean, prob_loss}}]}},
 ledger: {dir, n_entries, entries: [{origin, created, matures, status: pending | matured, version, model, analog_ens,
                                     methods: {m: {asset: bands, no q}}, realized, scores: {m: {asset: {covered, pit,
                                     crps, err_p50}}}}],
          matured, pending, first_maturity, scores: {m: {assets, risk_avg, n_entries}, crps_skill_risk} | null,
          appended, skipped, errors, text, methods: {m: name}}}
```

Origins: every December from 1984-12 with a full realized year after it, plus
the latest such month (42 origins, 1984-12..2025-09). At each origin the
model, MonthDB, anchor and CMA are refit on the months up to it. uncond = one
12-month block drawn under the prior; analog = kernel weights on block starts
whose origin state (KEY_STATES: cpi_yoy, unrate, real_ff, term_spread, spread,
div_yield, ip_yoy, oil_yoy) resembles today's, with the effective number of
starts chosen from ENS_GRID on origins up to 2004-12 (576 now); naive = a
normal with the training mean and sd of 12-month returns. Returns are 12-month
total returns (decimal); macro changes are pp (unrate, ff, cpi_yoy, spread)
or a log change (oil). CRPS: lower is better; skill = 1 - CRPS / CRPS of the
benchmark, on the risk-asset average. `misses` are the 15 largest misses of
the analog method (realized outside its 90% band), each tagged shock (an
EXOGENOUS_SHOCKS year), macro (a realized macro change outside its 90% range)
or pricing. `live.scenarios` mixes the first 12 months of every built-in
scenario by probability (scenario_paths.npz); paths from another start month
than the live origin are never mixed in (then `live` has no mixture,
reconciliation or calibration, and `calibration.reason` says to run `--only
scenarios forecast`).

`origins[].macro[k].naive` is a normal with the mean and sd of the training
months' 12-month changes over [MODEL_START, origin] (`naive_macro`).
`macro_skill`: a macro variable is skilled when the analog forecast of its
12-month change beats both benchmarks, the unconditional draw and that naive
normal: CRPS skill > 0 over all 42 origins and in both halves (split 2004-12),
with a paired t >= SKILL_MIN_T 1.0 against each. The naive benchmark matters:
both engine methods draw past months' CPI changes without re-anchoring them,
so over the 42 origins the analog CPI y/y forecast beats the unconditional
draw (skill +1.3%, t 1.4) but is 45.1% worse than naive (t -3.0), while
unemployment, the policy rate and the credit spread beat naive (+15.3%,
+15.5%, +11.2%) but not the unconditional draw, and oil beats neither. Today
no variable is skilled, so `calibration.available` is false,
`scenario_probs_calibrated` is null and `assets[a].scenarios_calibrated` is
absent; the 「按一年期预测校准」 choice of the portfolio and withdrawal tabs is
then disabled with `calibration.reason`, and 模型后验 stays the default.

Reconciliation (情景概率与一年期预测的对账). The mixture's first year is more
bearish than the scored analog forecast (美股大盘 median +4.6% against +7.8%,
P(loss) 40% against 32%), because the shock scenarios put their shocks in the
first year. `contrib_mean` and `contrib_prob_loss` split the gap exactly by
scenario (probability x the scenario's first-year value minus the analog
value; each sums to its gap). Each built-in is re-run from its detail spec at
its library seed to get the 12-month macro changes of the same paths
(`paths_reproduced` checks them against the npz). The calibration, when a
variable is skilled, is entropy pooling over the 13 scenarios with the model
posterior as the prior and, as views, the analog method's live mean 12-month
change of each skilled variable; an LP finds the largest reachable shrink
lambda (each q >= 0.001 x its prior). `sensitivity` is a what-if per
candidate variable and for all five together (unemployment alone: KL 0.26,
美股大盘 median +7.2%); it is never used.

`ledger` is the scored view of data/ledger/ (see "Forecast ledger"): today
1 entry (origin 2026-09, pending, matures when the data reach 2027-09),
`appended` false when the origin was already recorded, `skipped` the reason a
build did not record (null when it did or had nothing new).

### Forecast ledger (data/ledger/)

The build's own forecast run (default origins, `n_paths` 1000, seed 23, no
valuation override, `results_dir` None) records its live forecast as
data/ledger/forecast_<origin>.json; studies and other settings never append,
and `run(ledger_dir=None)` turns the ledger off. Schema 1:

```text
{schema: 1, origin: "2026-09", matures: "2027-09", horizon_months: 12, created (UTC), version, model,
 analog_ens, n_paths, seed, taus (q[i] = the (i + 0.5)% quantile, i = 0..99), units,
 methods: {analog | uncond | scenarios | scenarios_calibrated: {name, assets: {asset: {p5, p25, p50, p75, p95,
                                                                                     mean, prob_loss, q[100]}}},
           naive: {name, assets: {asset: {mu, sd}}}},
 macro: {analog | uncond: {k: {p5, p50, p95, mean, q[100]}}}, state,
 scenario_probs, scenario_probs_calibrated}
```

* Written to a unique temporary file and hard-linked into place: an entry
  appears complete or not at all, is never replaced, and a rebuild on the same
  panel adds nothing (two builds racing on one origin: one wins).
* `ledger_refusal`: the build only appends in time order (no origin older than
  one already recorded: a backfill would be made after the later data were
  known) and only within LEDGER_MAX_DELAY_MONTHS = 3 months after the origin
  month (a stale panel); otherwise `ledger.skipped` gives the reason and the
  page shows it. A scenario mixture from paths of another start month is never
  recorded.
* `load_ledger` keeps one entry per origin (the file named
  forecast_<origin>.json first) and lists unreadable files and duplicates in
  `ledger.errors`.
* Scoring: once the panel covers an entry's 12 months, every build scores it
  (`score_ledger`: covered by the 90% band, PIT, CRPS, error of the median, per
  method and asset; CRPS and PIT from the stored quantiles are within 0.14% /
  0.005 of the sample scores). This is the true out-of-sample record: the
  rolling scorecard above is a replay with today's code and model.
* The ledger lives in data/, so it is git-ignored: each installation keeps its
  own. The first entry (origin 2026-09, created 2026-11-01T16:06:16Z) was
  written while the calibration still accepted CPI y/y on the unconditional
  benchmark alone, so it carries a `scenarios_calibrated` method and
  `scenario_probs_calibrated` (KL 0.019) that today's forecast no longer has;
  its analog, uncond and scenarios bands equal today's live forecast exactly.

### ablation.json (analysis/ablation.py; opt-in)

```text
{question, text, windows: [{id, name, test, train_end, role: validation | test}],
 hypotheses: [{id: H1..H6, name, test, result, verdict: supported | rejected | inconclusive, evidence, numbers}],
 variants: [{id, name, description, hypothesis, params, adoptable,
             windows: {window: {cagr_mae_cond_risk, cagr_mae_uncond_risk, coverage_cond_risk, pit_tail_share,
                                engine_abs_annual, transmission_abs_annual, scenario_abs_annual, …,
                                cagr_mae_cond_risk_main_seed, assets}},
             validation_mae, validation_coverage, validation_tail_share, validation_gain, validation_gain_se,
             test_mae, test_coverage, test_tail_share, test_gain, test_gain_se,
             forward_fan (engine variants), eligible, scenarios}],
 selection: {rule, chosen, adopt, checks, eligible, reason, engine_variant},
 transmission_gap: {windows, context, text},
 scenario_effects: {n_paths, keys, baseline: {id: {<keys>, prob, name}}, baseline_prob_weighted, units,
                    variants: {variant: {headline, delta, prob_weighted, prob_weighted_delta,
                                         max_abs_us_equity_delta, max_scenario}},
                    asset_drift_busts: [ids of built-ins with a negative us_equity asset_drift],
                    engine_variant, engine_matches_results},   # the engine variant's run vs scenarios.json
 settings: {n_paths: 400, seeds: [11, 12, 13, 14, 15], coverage_drop: 0.05, min_gain: 0.002, fan_drop: 0.05,
            fan_paths: 4000, engine_variant, engine_reproduces_backtest, baseline_validation_mae,
            baseline_test_mae},
 units}
```

H1 residual contamination, H2 phase length, H3 view set, H4 CMA, H5 missing
transmission, H6 MIN_ENS. Only engine variants (H1, H4, H6) can be adopted;
the rule (validation windows tech_bubble and gfc_qe, one confirmation on
covid_ai and post_covid) is in `selection.rule`. `engine_variant` is the
variant config.RESIDUALS matches (partial_50 now), and
`engine_reproduces_backtest` checks it against backtest.json
(`scenario_effects.engine_matches_results` against scenarios.json). The
`baseline` variant is the engine before the adoption (coupled residuals), so
its numbers no longer match either file.

### valuation.json (analysis/valuation.py)

```text
{question, text, text_paragraphs: [str],
 data: {dates (1955-01 ..), pd, pd_fair, real_yield, profit_share, profit_growth_yoy, pe_proxy (CAPE),
        fcf_proxy, infl_expect, eras, units},
 regression: {formula, coefficients: {const | real_y10 | profit_share | infl_expect: {beta, t, name}}, r2, sample,
              n_obs, gap_sd, ecm, hac_lags: 60, alternatives: [{formula, …, gap_log_today}], cape, by_window,
              stability, theory_check},
 kappa: {value, half_life_years, how_chosen, grid: [0.05, 0.1, 0.2, "ecm"], ecm_full_sample, ecm_by_window,
         persistence: {log_pd_ar1_monthly, log_pd_ar1_annual, …, ecm_real_time, oos_r2_price_12m: {r2, n},
                       variance_ratio}, literature},
 today: {date, pd, pd_fair, gap_log, gap_percentile, gap_max: {value, date}, gap_log_alternatives, gap_log_range,
         kappa, implied_annual_drag, horizon_years, implied_annual_drag_15y, drivers, driver_names, levels,
         sample_means, pd_fair_at_sample_means, pd_percentile, pd_median, cape, cape_percentile, …},
 implied_pd: {valuation_adjusted | historical: {pd_end: {p5, p50, p95}, pd_fair_end_p50, log_price_sd,
              valuation_part_sd, dividend_part_sd, valuation_dividend_corr, variance_ratio,
              valuation_change_mean, pd_p50_by_year}},
 roic_proxy: {depreciation, corr_with_profit_share, sample, coef_in_fair_value, t_in_fair_value, r2, level_today},
 variants: [{id, name, description, adoptable, cma, settings, same_backtest_as, windows,
             validation_mae, validation_mae_uncond, validation_coverage, validation_gain, validation_gain_se,
             test_mae, test_mae_uncond, test_coverage, test_gain, test_gain_se,
             forward_fan, fan_width_ratio_min, forecast_crps_skill, forecast_skill, forecast, eligible, scenarios}],
 selection: {rule, chosen, adopt, checks, eligible, reason, best_candidate},
 factor_candidates: [{id, name, new, endogenous, release_dated, validation_delta, test_delta,
                      periods: [{period, delta}], spanned_r2, accepted_in_walkforward, corr_same_month_equity,
                      verdict, why}],
 factor_walkforward: {final, iterations, min_gain},
 scenario_effects: {n_paths, keys, baseline: {id: {<keys>, pd_end_p50, prob}}, baseline_prob_weighted,
                    units, modes: {variant: {prob_weighted, prob_weighted_delta, per_scenario,
                                             max_abs_us_equity_delta, max_scenario, pd_end_p50_prob_weighted}},
                    baseline_matches_results},
 settings: {n_paths, seeds, validation, test, coverage_drop, min_gain, fan_drop, fan_paths, forecast_tol,
            forecast_ic_tol, forecast_paths, forecast_split, analog_ens, n_origins, residuals,
            engine_valuation, baseline_reproduces_backtest, baseline_reproduces_forecast, seconds},
 units, notes}
```

Variants: three references (baseline 「现行引擎（不加估值拉力）」, cma_historical
「参照：全部用历史漂移」, cma_va 「参照：全部用估值调整 CMA」) and 9 adoptable valuation-state variants (kappa 0.05,
0.10, 0.20 or the ecm estimate, on historical or valuation_state drift; path
income with kappa 0.10 or 0; a fair value without inflation). The selection
rule is the ablation's (validation windows, one test confirmation, seeds
11-15, coverage, the 15-year fan width) plus a fourth test: the one-year
forecast (42 origins, split at 2004-12) must not get worse (CRPS at most 1%
worse, rank IC at most 0.05 lower). `gap_log` = ln(P/D) - ln(fair P/D), > 0 =
dear; `gap_percentile` is the share of the regression months whose in-sample
gap (today's coefficients) is smaller, `gap_max` the largest such gap, and
`pd_percentile` ranks the P/D itself (today 99% against 94% for the gap).
`scenario_effects.baseline` is the analysis's own copy of the 13
scenarios' headlines (+ pd_end_p50, prob), and `baseline_matches_results`
says whether it still matches scenarios.json (the 估值与盈利 page checks the
same and asks for `--only valuation` otherwise). The page's 'variants ± SE'
column is seed-to-seed simulation noise only; the evidence base is 2
validation plus 2 test windows.

### portfolio_default.json (portfolio/service.py run)

```text
{source: {paths_file, scenario_ids, official_paths, paths_modified, continuation_file, continuation_modified,
          computed_at},
 default_weights: {us_equity: 0.6, treasury_10y: 0.4},
 portfolio: evaluate_portfolio(60/40) response,
 withdrawal: evaluate_withdrawal(60/40) response at DEFAULT_HORIZON (30 years, the withdrawal page's default),
 continuation: {file, label, scenario_years: 15, max_years: 40, …continuation_paths.npz info} | null,
 presets: [{id, name, weights, description, portfolio: {weighted stats, no fans}, worst_id,
            ruin_prob_weighted: {strategy}, safe_rate_weighted: {strategy},          # at withdrawal_horizon_years
            withdrawal_horizon_years: 15,
            withdrawal_by_horizon: {"15" | "30": {ruin_prob_weighted: {strategy}, safe_rate_weighted: {strategy}}}}]}
```

Presets: classic_60_40, all_equity, all_weather, income, inflation_hedge. A
preset's `ruin_prob_weighted` / `safe_rate_weighted` stay at the scenario
length (15 years); the preset tooltips of both pages show the safe rate at
both horizons from `withdrawal_by_horizon` (60/40: 15 年 5.29% · 30 年 2.76%).

### continuation_paths.npz (portfolio/service.py, build step portfolio)

The withdrawal years after the scenario paths (16..40). `build_continuation`
runs 13 view-free 40-year `simulate()` runs of 600 paths (CONTINUATION_CHUNK)
from today's anchor (the prior over 6-month block starts, config.RESIDUALS,
CMA "valuation_state" = valuation_adjusted without its 15-year reversion term),
seeded `continuation_chunk_seed(crc32("continuation"), c)`, and keeps months
181..480 as yearly factors; `write_continuation` writes it atomically.

```text
total_y, price_y, income_y: (N, 25, A) float32     wd.year_factors of months 181..480 (N = 7800)
cpi_y: (N, 26) float64                              year-end CPI relative to month 180 (index 0 = 1)
assets (A,), labels (26 year-end labels, "2041-09" .. "2066-09"), start (anchor "YYYY-MM"), offset (180)
info (JSON): {method: "unconditional", seed, n_paths: 7800, chunk_paths: 600, chunk_seeds: [13], offset_months: 180,
              months: 300, block_months: 6, residuals, anchor, cma: "valuation_state", cma_adjustment,
              market_adjustment, rerating_tailwind, scenario_cma: "valuation_adjusted", scenario_market_adjustment,
              reversion_term_dropped, at_start: {y10, ff, spread, unrate, cpi_yoy}, inflation_p50,
              real_cagr_p50: {us_equity, treasury_10y, cash}, pd: {p50_at_start, p50_end, dlog_per_year}}
```

About 26 MB. One continuation path per built-in scenario path: row r of the
stacked scenarios goes on with path r mod N, so every built-in path has its
own future; custom rows appended after the built-ins start over at path 0
(the response says so: `continuation.shared`, `shared_scenarios`). The file
is used only when `start` equals the scenario paths' first label, `offset`
their month count and the asset lists match; a file in the earlier monthly
format (total / income / cpi) is refused as 旧格式. Either way a horizon past
15 years is then a 503 naming `--only portfolio`. Today: the continuation
starts with y10 5.4%, ff 4.7%, unemployment 4.3%, CPI y/y 3.4% (medians at
month 180); median inflation over years 16..40 3.9% a year; median real
CAGR 3.6% for 美股大盘, 1.5% for 10 年期美债; the tracked P/D stays about flat
(median 90.8 -> 88.6, -0.16% a year). The module docstring of
portfolio/service.py has the reasons and sensitivities (the CMA choice, one
path per row instead of 600 shared ones, the comparison with history).

### llm_translator.json (llm/translator.py run)

```text
{status: {available, model, reason}, model, horizon_months, today,
 request: {max_tokens: 16000, effort: "high", betas: ["server-side-fallback-2026-07-01"], fallbacks: "default",
           output_format: "json_schema"},
 prompt: {system, chars, sha256, examples: ["oil_war", "k_shaped"]},
 variables: [{var, name, kind: state | factor | shock, unit, p2, median, p98, mean, sd,
              block_p2, block_p98, block_sd, asset_driver, today, today_date, outside_support | last12_mean}],
 factor_drift_bounds: {factor: +- per year}, asset_drift: {bound: 0.08, assets},
 schema: SPEC_JSON_SCHEMA, examples: [LLM-form spec], translation: null}
```

What the translator would send the model, built without an API key. p2 /
median / p98 / sd describe single months, block_* the means of 6-month runs
(what a phase's views can reach). `examples` are the prompt's few-shot
built-ins (oil_war, k_shaped) in the LLM form with their whole narratives
(`translator.lab_examples`): the lab's 载入示例 shows them as narrative ① and
spec ②, labelled as not a live translation.

### custom/<id>.json and custom/<id>.npz (server.py)

Written by `POST /api/scenarios/custom/run` with `save` (the default): the
run_spec detail plus `created: "YYYY-MM-DD HH:MM:SS"` (source as posted:
"custom", "llm"; no probability, prob or calibration), and an npz with the
scenario_paths arrays for S = 1 (`total`, `income`, `cpi`, `ids`, `assets`,
`labels`; no `probs`). The npz is written to a temporary file first, because
the portfolio service may be reading the old one.

## Portfolio and withdrawal (portfolio/service.py)

```python
evaluate_portfolio(weights: dict, scenario_probs: dict | None = None,
                   include_custom: list[str] = (), rebalance: str = "monthly") -> dict
evaluate_withdrawal(weights: dict, initial: float = 1e6, rate: float = 0.04,
                    strategy: str = "all", inflation_indexed: bool = True,
                    scenario_probs: dict | None = None, include_custom: list[str] = (),
                    horizon_years: int = 30) -> dict
```

Inputs. `weights` = {asset id: weight >= 0}, normalized to 1 (an unknown
asset, a negative or non-numeric weight, or all zeros is a ValueError, i.e.
HTTP 400). Built-in scenarios take their `prob` from scenarios.json (else the
npz `probs`, else uniform); an included custom scenario (`include_custom`:
saved ids) takes its own `prob`, else `prior_prob`, else 0.05.
`scenario_probs` = {id: prob} replaces any of them and the whole vector is
renormalized, so 0 drops a scenario from the weighted mixture (it is still
reported on its own). `rebalance`: monthly | annual | none. Worst / best
scenarios are ranked in real terms (median real CAGR; for withdrawals the
ruin probability, ties broken by lower real terminal wealth). The path files
are cached by mtime and size; a missing or half-written npz raises
`PathsNotReady` (HTTP 503). All inputs are validated before any path is read.

Withdrawal strategies (annual, no other rebalancing):
* `pro_rata` (按比例提取): income is reinvested in the asset that paid it; at
  each year end the withdrawal is raised by selling every holding in
  proportion to its current value.
* `div_then_pro_rata` (先用股息/利息，不足按比例): income is paid out to a cash
  pot during the year (the pot earns nothing); at year end the pot funds the
  withdrawal; any surplus is reinvested pro-rata; any shortfall is sold
  pro-rata.
* `div_then_winners` (先用股息，不足按当年收益从高到低卖): like the previous one,
  but the shortfall is raised by selling holdings in descending order of that
  year's total return (sell the best performer first, exhaust it, move on).
* The withdrawal is `rate * initial` at every year end, indexed to the path's
  own CPI when `inflation_indexed` (the first year end already includes the
  first year's inflation). Ruin = holdings plus the pot cannot cover a year's
  withdrawal; what is left is paid out and wealth stays 0. The safe rate is
  the highest rate in 0..20% whose ruin probability is at most 5% (floored to
  0.01%); `ruin_curve` evaluates rates 0..10% in steps of 0.25%.
* `horizon_years`: a whole number of years in HORIZON_BOUNDS (10..40, default
  DEFAULT_HORIZON 30); anything else is a ValueError (HTTP 400:
  期限（horizon_years）必须是 10 到 40 之间的整数年（收到 …）, a non-integer shown in
  full). Years 1..15 are the scenario paths; years 16..40 come from
  continuation_paths.npz (no scenario theme: 「第 16 年起没有情景主题：按历史月份
  无条件续抽」). A missing, stale or old-format continuation is a
  `PathsNotReady` (HTTP 503) naming `python -m scenariolab.build --only
  portfolio`, for horizons past 15 years only. Every figure up to year 15 is
  bit-identical to a 15-year run (`scenario_horizon` repeats them).
* Dividend-first rules take equity and REIT income as the anchor yield / 12 on
  the current value, so the pot falls with prices. A sticky-dividend variant
  (smoothed price, half-life 12-48 months, same total return) raised the
  30-year ruin by 0.03-0.77 pp and moved safe rates by at most 0.02 pp; it was
  not adopted (portfolio/withdrawal.py docstring).

Response shapes (fans are wealth multiples of 1 at t = 0 for the portfolio,
amounts in the currency of `initial` for withdrawals):

```text
portfolio: {years, dates, horizon_months, weights, rebalance,
  per_scenario: [{id, name, source, prob, n_paths, cagr_p5, cagr_p50, cagr_p95, real_cagr_p5, real_cagr_p50,
                  real_cagr_p95, vol, sharpe, sortino, max_dd_p50, max_dd_p95, terminal_p50, terminal_real_p50,
                  cvar5_terminal, cvar5_terminal_real, prob_loss_real, fan, fan_real}],
  aggregate: {weighted, equal_weight: {same stats + fan, fan_real}, worst, best: {a per_scenario entry}},
  notes}
withdrawal: {years, dates, horizon_years, scenario_years: 15, horizon_bounds: [10, 40],
  initial, rate, annual_withdrawal, inflation_indexed, weights,
  strategies: {strategy: {name, description,
    per_scenario: [{id, name, source, prob, n_paths, ruin_prob, survival: [P(not ruined by year k)],
                    terminal_real_p5, terminal_real_p50, terminal_real_p95, terminal_nominal_p50,
                    depletion_year_median, wealth_fan_real}],
    aggregate: {ruin_prob_weighted, ruin_prob_worst, worst_id, ruin_prob_best, best_id, survival_weighted,
                terminal_real_p50_weighted, terminal_real_p5_weighted, terminal_real_p95_weighted,
                terminal_nominal_p50_weighted, depletion_year_median_weighted, wealth_fan_real_weighted}}},
  safe_rate: {strategy: {weighted, per_scenario: {id: rate}}},
  ruin_curve: {rates: [0 .. 0.1], strategy: {weighted: [...], per_scenario: {id: [...]}}},
  safe_ruin_prob: 0.05,
  scenario_horizon: null | {years: 15, date: "2041-09",                # the same runs cut at year 15
    strategies: {strategy: {aggregate: {ruin_prob_weighted, ruin_prob_worst, worst_id, ruin_prob_best, best_id,
                                        terminal_real_p5_weighted, terminal_real_p50_weighted,
                                        terminal_real_p95_weighted, terminal_nominal_p50_weighted,
                                        depletion_year_median_weighted},
                            per_scenario: {id: {ruin_prob, terminal_real_p5, terminal_real_p50, terminal_real_p95,
                                                terminal_nominal_p50, depletion_year_median}}}},
    safe_rate: {strategy: {weighted, per_scenario}}, ruin_curve: {rates, strategy: {weighted, per_scenario}}},
  continuation: null | {…continuation_paths.npz info, label, first_year: 16, years, start_date: "2041-09",
                        n_paths: 7800, rows, shared (bool: some rows reuse a continuation path),
                        shared_scenarios: [names]},
  notes}
```

`scenario_horizon` and `continuation` are null for horizons of 15 years or
less; `years` and `dates` run to the chosen horizon (dates past 2041-09 are
computed). 60/40 at 4%, indexed, probability-weighted (2026-11-01 build):
ruin over 15 / 30 / 40 years 0.045% / 37.0% / 59.0% (pro_rata), 0.16% / 38.8% /
58.8% (div_then_pro_rata), 0.25% / 37.9% / 57.3% (div_then_winners); safe
rates 5.29% / 2.76% / 2.17%, 5.17% / 2.62% / 2.04%, 5.10% / 2.61% / 2.03%. The
same withdrawal code on actual history (rolling 30-year windows from 1953)
fails 6-7% of the time: the gap comes from the engine's capital-market
assumptions (service.py docstring), not from the continuation. The first
request at a new horizon takes about 2.5-3 s (mostly bisecting the critical
rates of div_then_winners at the horizon and at 15 years), another rate at the
same horizon about 0.3 s.

Sharpe and Sortino are medians of per-path ratios against the cash asset;
`max_dd_p95` is the bad tail (only 5% of paths draw down further);
`cvar5_terminal` is the mean terminal multiple of the worst 5% of paths;
`prob_loss_real` is P(real terminal wealth < 1).

## HTTP API (server.py)

`python -m scenariolab.server` serves http://127.0.0.1:8765 (`MSL_PORT`
changes the port; it binds to 127.0.0.1 only). It loads .env at import, and a
background thread fits the EngineContext and loads the scenario paths at
startup so the first requests are fast.

| Method | Path | Body -> result |
|---|---|---|
| GET | `/`, `/js/…`, `/css/…`, `/vendor/…` | the static UI from web/ (`Cache-Control: no-cache` on `/`, `/index.html`, `/js/*`, `/css/*`, so edited tab modules are revalidated; `/vendor/echarts/*` gets ETag and Last-Modified, a conditional GET a 304, and no Cache-Control; a directory such as `/vendor/` is a 404, no listing) |
| GET | `/api/meta` | meta.json |
| GET | `/api/{name}` | results/{name}.json for name in eras, causality, backtest, diagnostics, scenarios, forecast, ablation, portfolio_default, llm_translator, valuation, regimes, regime_forecast, factor_audit |
| GET | `/api/scenarios/{id}` | results/scenarios/{id}.json, else results/custom/{id}.json (built-ins win) |
| GET | `/api/scenarios/custom` | `{scenarios: [{id, name, summary, source, created, prior_prob, prob}]}`, newest first |
| POST | `/api/scenarios/custom/run` | `{spec, save = true, n_paths = 600 (clamped 100..2000), seed = 7, source = "custom"}` -> the detail JSON + `created`; with `save`, also custom/{id}.json + .npz |
| DELETE | `/api/scenarios/custom/{id}` | `{deleted: id}` |
| POST | `/api/portfolio/evaluate` | the keyword arguments of `evaluate_portfolio` (`weights` required) -> its result |
| POST | `/api/withdrawal/evaluate` | the keyword arguments of `evaluate_withdrawal` (`weights` required; `horizon_years` 10..40, default 30) -> its result |
| POST | `/api/backtest/run` | `{train_end "YYYY-MM", months 12..180, n_paths 100..400 = 200, seed = 11}` -> one custom holdout window: the `run_window` JSON (id "custom", name "自选窗口") + `label`, `custom`, `shocks`, `elapsed_s` (see backtest.json) |
| GET | `/api/llm/status` | `{available, model, reason}` (local checks only, no network call) |
| POST | `/api/llm/translate` | `{narrative (<= 20000 chars), horizon_months = 180 (12..600)}` -> `{spec, validation, model, requested_model, usage, raw, elapsed_s, preview}` |

`/api/backtest/run` is a plain def, so FastAPI runs it in its threadpool and
the server stays responsive. Finished windows are cached in memory by
(train_end, months, n_paths, seed), up to BACKTEST_CACHE_MAX = 32; an
identical request waits on a per-key lock instead of computing again; at most
BACKTEST_MAX_RUNNING = 2 windows compute at once (the work is mostly Python
under the GIL) and the rest queue. Requests for the same window with other
seeds or path counts share one pooled setup (`backtest._cached_setup`).

Custom ids: the server slugifies `spec.id` (or the name) exactly as
validate_spec does. An id that is a built-in id (a built file or a spec in
the library) becomes `custom_<id>` (then `custom2_<id>`, …) so a saved custom
scenario can never be hidden behind a built-in; a spec with nothing usable
for an id (no id and a Chinese name both slugify to "scenario") gets a unique
`custom_<10 hex>`. The lab applies the same rules before it runs.

Errors are JSON `{"detail": "<Chinese text>"}`:

| Status | When | Detail |
|---|---|---|
| 400 | body not JSON / not an object | 请求体不是合法的 JSON / 请求体必须是一个 JSON 对象（{ … }） |
| 400 | unknown or missing evaluate parameters | 不认识的参数：…（可用参数：…） / 缺少参数：weights |
| 400 | bad values (weights, probabilities, strategy, rate, …) | the service's own Chinese message |
| 400 | bad scenario id (`^[a-z0-9_]{1,60}$`) | 情景 id「…」不合法：只能包含小写字母、数字和下划线（最多 60 个字符） |
| 400 | custom run: `spec` not an object / validation errors / the engine cannot run it | spec 必须是对象 / 情景规格有错误：… / 情景规格无法运行：… (nothing is simulated or saved) |
| 400 | translate: empty narrative / longer than 20000 characters / horizon_months not an integer in 12..600 (checked before the credentials; null means 180) | narrative 不能为空 / 叙事过长（… 字），请控制在 20000 字以内 / horizon_months 必须是 12 到 600 之间的整数（情景总月数）… |
| 400 | backtest/run: unknown or missing parameters; train_end not YYYY-MM or outside the panel (a year pandas cannot format, such as 0000, included); fewer than 15 training years (earliest 1987-12); a test past the last panel month (the message gives the room left); months / n_paths / seed out of range or not integers; a training sample the model cannot be fitted on | 不认识的参数：… / 缺少参数：months（…） / train_end … 不在数据范围内（1947-01 至 2026-09） / 训练期至少需要 15 年：… 最早可选 1987-12 / 测试期会到 …，超出数据的最后一个月 …：… / months 必须是 12 到 180 之间的整数（测试期月数），收到 … / 这个窗口无法回测：… |
| 400 | withdrawal: horizon_years not a whole number in 10..40 | 期限（horizon_years）必须是 10 到 40 之间的整数年（收到 …） / 期限（horizon_years）必须是数字 |
| 404 | a result file is missing | `<file> 还没有生成：请在项目根目录运行 python -m scenariolab.build（或只补这一步：python -m scenariolab.build --only <step>）`; ablation.json names `--only ablation（引擎消融约需 6 分钟，默认构建不包含这一步）` |
| 404 | unknown result name | 没有名为「…」的结果 |
| 404 | built-in scenario not built / unknown scenario / unknown custom id on delete | the build hint with `--only scenarios` / 没有找到情景「…」：… / 没有找到自定义情景「…」（可能已被删除） |
| 404 / 405 | unknown path / wrong method | 找不到这个地址：<path> / 这个地址不支持该请求方法：<path> |
| 500 | backtest/run: an unexpected engine error | 回测出错：<type>: <message> |
| 502 | translate: the API call failed (401, 403, 404 model, 429, 5xx, timeout, refusal, max_tokens, invalid JSON) | the translator's Chinese message |
| 503 | evaluate: the path files are missing or being rewritten | 无法读取 scenario_paths.npz（可能正在重新生成，请稍后重试）… / scenario_paths.npz 还没有生成… |
| 503 | withdrawal past 15 years: continuation_paths.npz missing, stale (another start month or length, other assets) or in the old format | 期限 … 年超过情景路径的 15 年，需要无条件续抽路径：…（names `python -m scenariolab.build --only portfolio`） |
| 503 | translate: no credentials or no anthropic package | the status reason (copy .env.example to .env, set ANTHROPIC_API_KEY, restart) |

The build hint in a 404 is what the UI shows under a failed page
(`ui.buildHint`).

## LLM translator (llm/translator.py)

The rule: the LLM never produces prices, returns or K-lines. It only turns a
narrative into the scenario spec above (shocks, views, phases, bounded
overrides, causal chain); `validate_spec` clips it to the historical support,
and the engine produces every number. The UI shows the validation report next
to the results.

* Credentials: `status()` is available when the `anthropic` package is
  installed and ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN is set (in the
  environment or .env), or an `ant` CLI profile is found. The model is
  `MSL_LLM_MODEL`, default `claude-opus-5-5`. Never commit .env (it is in
  .gitignore).
* Request: `client.beta.messages.create(model, max_tokens=16000, system=[the
  prompt, cache_control ephemeral], messages=[narrative],
  betas=["server-side-fallback-2026-07-01"], fallbacks="default",
  output_config={effort: "high", format: {type: "json_schema", schema:
  SPEC_JSON_SCHEMA}})`; parameters the installed SDK does not name go through
  `extra_body` / `extra_headers`.
* `build_system_prompt(ctx)` is deterministic for a build (no timestamps,
  fixed ordering) so prompt caching hits: the engine's semantics (6-month
  block pooling, the shrink lambda, level scaling and reversion targets with
  the real-oil target computed per build, about $68 a barrel, and today's $97
  drifting toward it in phases without an oil view), the hard rules, every
  variable's support (6-month block p2 / median / p98 / sd) and today's
  value, the eras, the drift bounds, and two built-in specs in the LLM form
  (oil_war, k_shaped; calibrations resolved). The examples' narratives,
  summaries and causal chains contain engine numbers filled in after the run;
  the prompt says to restate only the model's own views and overrides and to
  keep the narrative to 2-4 paragraphs. The residual rule (item 4 and the
  resid_scale line: half of the residual follows the drawn months) is
  `residual_rule()`, generated from config.RESIDUALS; the prompt is 20,882
  characters now. `llm_form(spec, source, narrative_chars=1200)` and
  `few_shot_examples(ctx, horizon, narrative_chars)` cut the examples'
  narratives (None keeps all); `lab_examples(ctx)` keeps them whole for
  llm_translator.json `examples` (the lab's 载入示例).
* validate_spec warns when a posted spec carries a `calibration` block: only
  the built-in library resolves it into views and drifts.
* `translate(narrative, ctx, horizon_months)`: narrative <= 20000 chars,
  horizon 12..600 months (the engine always simulates 180: a shorter horizon
  gets a view-free tail phase and a warning, a longer spec is trimmed from the
  back); `parse_response` turns refusals,
  max_tokens, an exceeded context window and invalid JSON into Chinese
  `TranslatorError`s; then `repair` -> validate_spec -> `preview` (per phase:
  ens, shrink, era weights, top months, and every view's `drawn_mean`, the
  exact expectation over simulate()'s draws, with `realized_share` = (drawn -
  prior) / (target - prior)).
* CLI: `python -m scenariolab.llm.translator "叙事" --run` prints the spec,
  the validation and the engine's headline; `--yaml OUT` writes it in the
  built-in format; `--print-prompt` prints the system prompt without calling
  the API.

## Web app (web/)

No build step: index.html loads the vendored ECharts 5.5.1
(`vendor/echarts/echarts.min.js`, Apache-2.0, LICENSE and NOTICE beside it)
and `js/app.js` as an ES module. No file under web/ references an external
URL, so the UI works offline and with every CDN blocked (checked with a
browser that resolves only 127.0.0.1: every tab renders, no request leaves
the machine). To upgrade, download the version's tgz from the npm registry,
check it against the registry's `dist.integrity`, keep only
`dist/echarts.min.js`, LICENSE and NOTICE, and update the version comment in
index.html and the README.

* **Router** (app.js). `TABS` lists 12 tabs in four groups: 历史与因果
  (overview 总览, eras 历史时代画像, causality 因果传导), 验证 (forecast
  一年期预测, backtest 回测：模拟 vs 实际, diagnostics 诊断与迭代, valuation
  估值与盈利), 未来 (scenarios 未来情景, lab 情景实验室（LLM）, portfolio
  组合评估, withdrawal 提取与破产概率), 说明 (method 方法与数据). A route
  `#<tab>?<param>` lazy-imports `./tabs/<tab>.js` and calls
  `render(root, ctx)`; root is a fresh container that is detached when the
  reader navigates away, so a slow render can never draw on the next page.
  `ctx = {api, charts, ui, meta, param, route, signal, alive, rerender}`:
  `param` is the decoded deep link after "?"; `signal` is an AbortSignal
  aborted on the next navigation or re-render (stop timers and observers);
  `alive()` turns false once the reader has moved on (drop stale responses);
  `rerender` is true when the same page is drawn again in place (the K-line
  toggle), so drafts and picks are kept. render may return an Element (a
  deep-linked section) to scroll to; a failed render shows the error and, for
  a 404, the build command (`ui.buildHint`). Deep links include
  `#scenarios?<id>`, `#lab?<custom id>`, `#lab?tpl=<built-in id>`,
  `#portfolio?preset=income&rebalance=annual&scenario=ai_bust`,
  `#withdrawal?rate=3.5&horizon=10&nominal&strategy=div_then_winners`
  (`horizon` or `years`: 10..40, default 30; numbers are rounded and clamped
  with a notice, a non-number keeps 30 with a notice),
  `#backtest?<window>/<asset>/log` and
  `#backtest?custom=YYYY-MM:months[:paths[:seed]][/<asset>][/log]` (a shared
  custom window runs again on load; a `custom=` token that does not parse
  shows a notice and the built-in windows),
  `#valuation?answer | today | history | regression | kappa | variants |
  factors | scenarios`, `#eras?today`, `#diagnostics?ablation`. Pages that
  rewrite their own deep link use `ui.setRoute(tab, param, alive)`, which
  replaces the address without a re-render and only while that page is still
  current.
* **api.js**: `get(path, {fresh})` caches GET promises per page load (a failed
  one is evicted), `post`, `del`, `invalidate(prefix)`; an error carries
  `status` and the server's `detail`; strings in every response lose a
  rounded negative zero.
* **ui.js**: `h()` (element builder), `esc`, `card`, `chartBox`, `wideChart`
  (a dense chart scrolls inside its card, never the page), `kpi`, `table`
  (`stack` turns rows into labelled blocks on phones), `seg`, `select`,
  `setRoute`, `buildHint`, `loading`, `notice`, `chip`; formats `fmtPct`,
  `fmtPP`, `fmtNum`, `fmtMoney` (never "-0.0"); `signCls(x)` -> `pos` / `neg`
  by the sign of a number, `goodCls(ok)` -> `is-good` / `is-bad` for a
  judgement; META lookups filled from /api/meta (`assetName`, `factorName`,
  `stateName`, `eraName`, `varName`).
* **charts.js**: `mount` / `unmount` / `disposeWithin(root)` (app.js disposes
  every chart and observer of the previous page), `line`, `fan`, `candles`,
  `heatmap`, `bars`, `dag` / `dagLayout`, `scatter`, `assetColor`, `palette`,
  `fmtValue`; options are only ever added, so existing calls keep working.
  When `window.echarts` is missing (the vendor file blocked or deleted),
  `mount()` puts 「图表库 ECharts 没有加载，这里的图表无法显示（表格和文字不受影响）。」 in
  the chart box and returns an inert stand-in, so tables and text still render.
* **K-line colours.** The default is CN (红涨绿跌: red up, green down); the
  topbar select switches to US (绿涨红跌), stored in localStorage
  (`kline-color`). `charts.getKlineScheme / setKlineScheme / klineColors /
  klineDiverging` drive candles and return heatmaps; app.js sets
  `body.kline-cn` / `body.kline-us`, which set the CSS `--up` / `--down`
  tokens. `.pos` / `.neg` (the sign of a number) follow them; `.is-good` /
  `.is-bad` (a judgement, green better and red worse) do not.
* **Tabs and the files they read**: overview (eras, forecast, scenarios),
  eras (eras, causality, meta), causality (causality), forecast (forecast,
  scenarios), backtest (backtest, POST backtest/run), diagnostics
  (diagnostics, optional ablation), valuation (valuation, optional
  scenarios), scenarios (scenarios, scenarios/{id}, scenarios/custom), lab
  (llm/status, llm_translator, scenarios, scenarios/{id}, scenarios/custom,
  POST llm/translate, POST scenarios/custom/run, DELETE
  scenarios/custom/{id}), portfolio and withdrawal (portfolio_default,
  scenarios, scenarios/custom, forecast for the calibrated probabilities, POST
  portfolio/evaluate or withdrawal/evaluate), method (meta, optional
  valuation). `_scenario_detail.js` renders one scenario detail for the
  scenarios and lab tabs (its P/D card fetches valuation once); `_weights.js`
  holds the weight, probability and custom-scenario inputs shared by
  portfolio and withdrawal. The valuation tab injects its own styles
  (`style#style-valuation`; app.css is unchanged), and its scenario section
  compares scenarios.json with `valuation.scenario_effects.baseline`.
* **传导归因 on the scenario pages** (`_scenario_detail.js`, a card under
  因果传导链): an asset select (default: the asset the chain links to, e.g.
  石油能源股 for oil_war), a phase choice (全程 + each phase), a one-line
  reading generated from the numbers (the three largest factor channels, with
  利率 = ff + y10 and 通胀 = goods + services, then 其余因子合计, 未解释部分, any
  情景覆盖 / 估值回归 / −95% 截断, and 常数项、估值调整和收入合计, so the parts add
  up to 算术年化), a horizontal waterfall in K-line colours by sign (factors
  -> ＝因子合计 -> 常数项 / 估值调整 / 情景覆盖 / 未解释部分 / 收入 -> ＝算术年化合计
  -> 复利、波动与中位数修正 -> ＝中位数年化回报) and a per-phase table (stacked
  blocks on phones). Causal-chain edges from a box that names a factor (or
  高估值, the CMA term of us_equity only) to a box that names an asset (exact
  labels) carry the engine number of the selected phase in the edge label,
  the tooltip and a line under the mechanism: oil_war and energy_transition
  油价→能源股, dollar_decline 美元→黄金, fiscal_dominance 美元走弱→黄金,
  baseline 高估值→股票长期回报.
* **传导归因 on the eras page** (`eras.js`, under 资产表现): a heatmap of the
  eras since 1973 x channels for a chosen asset (9 factor columns with 通胀 =
  goods + services, ＝因子合计, 常数项, 未解释部分 outlined dashed, ＝价格回报,
  收入) in K-line colours, and a sentence for the selected era (the three
  largest channels plus 其余, then 复利与波动修正 from the arithmetic total to the
  CAGR: compounding less volatility drag, positive in strong eras such as
  科技繁荣; R^2 in the era and the era's own beta; the whole
  era's CAGR from the asset table when the attribution covers fewer months;
  for treasury_10y and crude_oil a note that the engine builds them
  structurally). A red notice appears when `attribution_info.model.spec`
  differs from `meta.model.spec`.
* **回测**: a 「自选时间段」 card (presets 1987-12 + 15 年, 1999-12 + 5 年,
  2007-06 + 3 年, 2019-12 + 2 年, 2021-12 + 3 年, free entry, paths and seed
  under 路径数与随机种子). A run adds a 自选窗口 pill (the last 4 are kept for the
  page load), every chart and table of the page switches to it, the address
  becomes a shareable deep link and 复制链接 copies it. Exogenous shocks are
  shaded on the main, small, macro and transmission charts (legend item 外生冲击),
  named in tooltips and listed under the main chart.
* **一年期预测**: new sections 情景概率与一年期预测的对账 (the gap split by
  scenario for every asset, each scenario's first year against its own 15
  years, macro skill against both benchmarks, the calibration and the
  what-ifs) and 实时样本外记录 (the ledger table per asset, stacked on phones,
  and the scores once entries mature).
* **Scenario probabilities** (`_weights.js` probEditor, both portfolio and
  withdrawal): 模型后验 (the default: no `scenario_probs` sent) / 按一年期预测校准
  (forecast.json `live.scenario_probs_calibrated` for the built-ins; custom
  scenarios keep their prior weight; disabled with `calibration.reason` while
  no variable is skilled, as today; a notice when forecast.json and
  scenarios.json come from different builds) / 手动 (sliders, 等权), and a
  一年期校准 column. `shared.probMode` remembers the choice across the two tabs;
  `mode()` and `label()` give the tabs their status line (按一年期预测校准的概率 /
  手动调整的概率 / 模型后验概率). Preset tooltips show the 15- and 30-year safe
  rates.
* **提取与破产概率**: a 10-40 year horizon slider (default 30) with a hint, a
  note on the continuation (第 16 年起没有情景主题), the 15-year figures beside
  each KPI and in the tables, a card comparing 15 years with the chosen
  horizon, thin 15-year curves on the ruin-vs-rate chart, the continuation
  years shaded on the survival and wealth charts, and extra heatmap bands
  (15-30%, 30-50%, >50%) past 15 years; tables stack on phones and keep their
  first column pinned on tablets (`style#style-withdrawal-horizon`).
* **组合评估** and **提取与破产概率**: clearing every weight hides the previous
  portfolio's results with a notice until a valid portfolio is computed
  (restoring the same weights shows them again at once).
* **情景实验室**: 载入示例 puts a built-in's narrative in ① and its hand-written
  spec in the LLM form in ② (llm_translator.json `examples`, else
  scenarios/{id}), labelled 「这不是 LLM 的实时翻译，也没有调用任何 API」, with a
  per-phase table and a button that runs it through the engine; no API key
  needed.

## Testing

`.venv/bin/python -m pytest -q` runs the whole suite: 494 tests in 21 files
at the time of writing, about 4 minutes (494 passed in about 240 s after the full
build of 2026-11-02, none skipped). Tests that need local data skip themselves until
it exists: `needs_panel` / an inline check skip without
data/processed/panel_monthly.csv; test_server.py's `needs_build` skips
without data/results/meta.json; others check the specific result file they
read (backtest.json, ablation.json, scenario_paths.npz, continuation_paths.npz
for the real-data portfolio tests). test_translator.py
needs the `anthropic` package (the llm extra) and never calls the API: it
injects fake clients and checks that the system prompt is deterministic.
test_build.py covers the build steps and the sources with the network mocked.
The server tests that run a custom scenario write into data/results/custom (and
delete it) or into a temporary directory. A fresh clone (no data/) without the
llm extra, as CI installs it (`.[dev]`), runs 294 passed / 172 skipped (about
50 s; test_translator.py is skipped as a whole).

Files: test_ablation, test_attribution (bit-identity of simulate and run_spec
with and without the attribution, additivity, the reconciliation, a pure oil
shock, the structural assets, the lagged-spec fallback, the valuation pull, and
that the factor terms use the simulated, not the drawn, factor changes),
test_backtest (incl. the custom windows: validation, no look-ahead, the setup
cache), test_build, test_causality, test_diagnostics, test_engine, test_eras
(incl. era_attribution on synthetic and real data), test_forecast (incl. the
ledger: append-only, refusals, duplicates, scoring on stored quantiles; the
skill rule and the calibration), test_portfolio (incl. the continuation: the
first 15 years bit-identical, one path per row, stale and old-format files
refused), test_scenarios, test_server, test_translator, test_valuation,
test_withdrawal.

CI (.github/workflows/ci.yml) runs on every push and pull request, on Python
3.10 and 3.12: `pip install -e ".[dev]"` (no llm extra), `python -m
compileall -q scenariolab`, `ruff check scenariolab tests` (pyproject
`[tool.ruff.lint] select = ["E9", "F"]`; the dev extra pins
`ruff>=0.16,<0.17`) and `pytest -q`. It never downloads data or runs the
build: on a fresh clone the data-dependent tests skip.

## The missing-factor round (缺失因子审计, 2026-11-01)

This section is self-contained: it records everything the missing-factor round changed or added, and where an
earlier section of this file says something else, this section wins. The research behind it (six families, a
verifier and a critique) is archived as statistics in `docs/research/2026-11-01-factor-audit/` and as data in
`scenariolab/analysis/audit_archive/research_2026_11_01.json`; both are frozen once the round lands. Example
numbers are from the full build of 2026-11-02 (01:53 UTC, data through 2026-09); panel_meta, eras,
diagnostics, factor_audit and meta were rebuilt at 03:26 UTC for text and audit-field fixes (the panel, the
selected model and every number quoted here unchanged).

### Conventions: column families and timing

* Panel column families gain `x_<name>`: stored audit-only columns (`x_spread_mixed`, today's old spread
  timing; `x_labor_share_interp`, the old interpolated labor share), with which the audit rebuilds the research
  round's base model. An `x_` column is never a view, a drift or a model input (`schema.ALLOWED_PREFIXES =
  ("s_", "f_", "e_", "rg_")`). Two `f_` columns are audit-only as well: the walk-forward's report-only stage-B
  regressors `f_usd_em` (Δlog of the EM dollar: TWEXOMTH monthly average to 2005-12, DTWEXEMEGS month-end from
  2006-01, chained) and `f_mort_spread` (Δ `s_mort_spread`). They are not `config.FACTORS` ids
  (`panel.STAGE_B_AUDIT`, role `audit` in `column_info`): `validate_spec` refuses a view on them with W14 (as on
  any `f_` column of the frame that is no config factor, `f_cpi` aside: `schema.audit_only`), they cannot be
  drifted and the variable table leaves them out.
* **Reference month vs release dated.** Monthly macro data sit at the month they describe. Every quarterly
  series is *release dated*: `release_dated(q, index, lag)` puts the quarter labelled at its first month m0 in
  month m0 + lag and holds it until the next release, at most `QUARTER_HOLD` = 3 months (Z.1 wealth before 1952 is
  annual, so a few 1947-52 months stay empty). Ratios (profit share, capex share, wealth / income, IT investment
  share, tariff rate, deficit / GDP) are formed quarterly first. Nothing is interpolated any more except single
  missing months of monthly series (the 2025-10 CPI); market prices (month-end, last week) carry at most one
  month. `f_labor_share` and the other release-dated `f_` columns are therefore non-zero only in release
  months (labor share: Feb / May / Aug / Nov). `s_profit_share` is now the same series as `s_profit_share_rt`.
* **Spread timing.** `s_spread = lvl_baa − lvl_y10s` with both legs at the same time of month: monthly averages
  to 1961-12, the last weekly WBAA and WGS10YR of the month 1962-01..1985-12, DBAA and DGS10 month-end from
  1986-01. `lvl_y10` (the yield factor) is unchanged. 1973-85 levels correlate 0.969 with the old series.
* **Sentiment.** Before 1978 the quarterly Michigan survey is held until the next survey (at most 3 months)
  instead of interpolated.
* `config.RELEASE_LAGS` (months from a quarter's first month to its first publication): PRS85006173, OPHNFB,
  ULCNFB, A679RC1Q027SBEA 4; CP, GDP, PNFI, DPI, FGDEF, FYGFGDQ188S, HLW 5; B235RC1Q027SBEA 3; TNWBSHNO 6.
  `VALUATION["release_lag_months"]` stays 5 and must equal `RELEASE_LAGS["CP"]` (test).
* `panel_meta.json` `column_info`: `{col: {sources, alignment: reference_month | release_dated | month_end |
  monthly_average | last_week | mixed, lag_months, first, last, late, full_history, revised, copyright, role:
  model | state | view | audit}}` for every `s_`, `f_` and `x_` column (`panel.COLUMN_INFO` must have an entry
  for each; the build fails otherwise). `late` = first observed more than 12 months after 1973-02;
  `full_history` = observed in every month from 1973-02. panel_meta also gains `release_lags` and the notes on
  release dating, the spread legs, GZ / ACM revisions and the house-price splice.

### Engine settings (config.py)

| Setting | Value | What it changes |
|---|---|---|
| `FACTORS` | 27 ids | roles: `core` (10, the fixed comparator `ModelSpec()`), candidates (`core: False`), `interaction` (`tp10_x_core4`: `{"factor": "tp10", "state": "core_yoy", "lag": 1, "above": 4.0}`), `dedicated` (`metals_oil`: `["energy"]`), `role: "audit"` (`tp10_x_infl4`, `ebp`: comparison columns only, `AUDIT_FACTOR_IDS`) |
| `STATES` | 60 ids | the 30 earlier states plus 30 scenario view variables (`VIEW_STATE_IDS`), each with `concept` (a key of `VIEW_CONCEPTS`, concept id → Chinese name) and `source`; `late: True` for `tips10` (2003-01) and `mich1y` (1978-01) |
| `RELEASE_LAGS` | see above | release dating of every quarterly series |
| `FRED` | `{"api_url": "https://api.stlouisfed.org/fred", "key_env": "FRED_API_KEY", "min_interval_s": 0.6, "attempts": 4}` | the FRED API client |

New factors: `gz` (GZ 信用利差变动, Fed EBP csv `gz_spread`, from 1973-02), `tp10` (Δ ACMTP10, NY Fed ACM,
month-end, from 1961-07), `tp10_x_core4` (`f_tp10[t] · 1{s_core_yoy[t−1] > 4.0}`, NaN where the lag is NaN; the
threshold is `factor_model.HIGH_INFLATION` and `config.REGIMES["params"]["high_inflation_pct"]`, test),
`metals_oil` (Δlog of the World Bank Pink Sheet base-metals index over the average crude price, both monthly
averages, from 1960-02). Three "high inflation" switches read the same 4%: the regime label `high_infl`
(headline CPI y/y, same month), `ModelSpec(regime="inflation")` (headline, previous month; still rejected) and
the TP term (core CPI, previous month).

### Transmission model: per-asset equations (engine/factor_model.py)

The selected model is **F** (`model_selection.json`): `factors = [oil, ip, cpi_goods, cpi_services, ff, y10,
spread, usd, unrate, gz, tp10_x_core4]` (labor share dropped), linear, no lags, full sample, plus
`asset_factors = {energy: [metals_oil]}` and `asset_start = {gold: "1982-01"}`; `describe()` 「11 个因子、能源股 +1
个专门因子、黄金从 1982 年起估计」.

* `ModelSpec` gains `asset_factors` (`((asset, (factor ids)), ...)`) and `asset_start` (`((asset, "YYYY-MM"),
  ...)`): sorted tuples in the dataclass, dicts in JSON (`to_dict`); `from_dict` accepts dicts, lists or an
  absent key, so older files load unchanged. `describe()` appends 「、能源股 +1 个专门因子」 (「、{n} 个资产有专门因子」
  when several assets have extras) and 「、黄金从 1982 年起估计」.
* Columns: `base_columns(spec)` = common `f_` columns + `shock_cols` + the sorted union of the extras;
  `asset_columns(spec, asset)` and `asset_feature_names(spec, asset)` give one equation's columns and
  features. Extras enter contemporaneously; lags and the regime interaction apply to the common columns.
  Feature order: common, common lags, `_hi` terms, `hi_infl`, extras.
* `AssetFactorModel.fit` keeps the joint row mask (every union feature and every model asset's return present)
  and groups the assets by (extras, start); each group is its own OLS / ridge on its own columns and months.
  `coef` stays one table, assets × union features, with 0.0 outside an equation (`tstat` NaN there, null in
  JSON); `stats.r2` / `n_obs` / `resid_sd` are per asset over its own rows; `asset_train_start = {asset:
  "YYYY-MM"}`. A spec without extras or own starts takes the previous single-equation code path (bit-identical).
  Predictions keep the joint standardisation round trip (equal to `x @ coef.T + intercept` to about 1e-16).
* Short own windows fall back: an `asset_start` is honoured only when its own window holds at least
  `MIN_ASSET_WINDOW` = 120 months, else that asset uses the common start and `asset_start_used[asset]` is None.
  The forecast's origins before 1991-12 fit gold from 1973; `fit(..., honor_asset_start=False)` turns the
  windows off (causality's per-era betas). `kind="mlp"` with extras or a start raises `ValueError` (the networks
  are report-only).
* `contributions`, the attribution (`Attribution.asset_factor_terms`, each asset's own terms) and
  `summary()` (adds `asset_factors`, `asset_start`, `asset_train_start`, `asset_start_used`) follow the per-asset
  equations; a feature outside an equation contributes exactly 0.
* `capital_market_assumptions` reads `coef` as before; gold's oil beta now comes from its 1982+ equation.

### MonthDB (engine/simulate.py)

`SIM_FACTORS = sorted((set(FACTOR_IDS) − AUDIT_FACTOR_IDS) | {"cpi"})`: gz, tp10, tp10_x_core4 and metals_oil
are tracked like any factor (drawn months' values, optional `factor_drift`). `MonthDB.needed` = the
non-optional `SIM_FACTORS` columns + `base_columns(model.spec)`. Every new input covers 1973-02 onward, so the
MonthDB keeps **644 months, 1973-02..2026-09** with F (test). GZ, the TP term and metals/oil enter simulations
with the drawn month's own values: no level scaling and no reversion (unlike `f_spread`). The TP term's dummy is
the drawn month's own (its previous month's core CPI): 「期限溢价项只在抽中的历史月份本身处于高核心通胀时起作用」, so a
phase that wants it must draw such months (a view on `s_core_yoy` or the regime layer's high-inflation flag).
The dummy is on in 231 of the 644 months, last in 2023-12; today core CPI y/y is 2.57, so it is off. Gold's
residuals for 1973-02..1981-12 come from its 1982+ fit. Simulated CPI, IP and unemployment for a given draw do
not depend on the transmission, so the regime layer's reference run (2000 paths, seed 4242) keeps
bit-identical regime mixes (checked: base on the old panel, base and F on the new panel, all 13 regime arrays
and the drawn months equal).

### Walk-forward (analysis/walkforward.py)

* **Candidates.** 28 = the earlier 22 + `add_gz`, `add_tp10_x_core4`, `drop_labor_share` (eligible) and
  `add_tp10_x_infl4`, `add_tp10`, `add_ebp` (report-only comparisons). `MLP_ELIGIBLE = False`: `mlp16` / `mlp32`
  are report-only with the note 「传导模型必须是线性的：界面显示每个因子的 β，情景回报按因子拆分；神经网络只作对照（见缺失因子审计）」.
  The old `QUARTERLY` set is folded into `RELEASE_DATED` (「季度数据按通常的发布滞后对齐，只在发布月变化（个别推迟发布的季度会早
  1–2 个月出现）。」: the fixed lags of `RELEASE_LAGS` are the usual ones; ALFRED shows a few quarters published later, e.g.
  the labor share's 2018-Q4 and 2025-Q3/Q4 and the tariff rate's 2013-Q3, 2018-Q4 and 2025-Q3/Q4, which the panel shows
  1-2 months early).
* **Incremental gate** (`GATE = "incremental"`). An iteration's selection is kept only when it differs from the
  iteration's start and beats the *start* on the untouched test by more than `MIN_GAIN`:
  `test_result.delta_vs_start` (and `delta` against the base, as before). The next iteration starts from the
  kept spec; the final common spec is the last selection if it passed, else the last start. Iteration 1
  (start = base) behaves as before.
* **Stage B** (`dedicated_pass`), on the common spec C: `DEDICATED_CANDIDATES` = `energy_metals_oil` and
  `gold_window` (options 1982-01 / 1985-01) eligible, `industrials_usd_em`, `gold_tips10_avg` (not evaluated:
  the panel has no `f_tips10_avg`, TIPS start in 2003) and `gold_mortgage_spread` report-only. Per iteration: the
  best option on the asset's own validation R² (benchmark: the asset's mean over the whole training window) is
  chosen when it gains more than `MIN_GAIN_ASSET` = 0.003, and accepted when its test gain clears it too; the
  final spec takes what passes in the last iteration (`final_rule: "last_iteration"`, `test_used_for_choice:
  true`).
* `run_walkforward(panel, base, iterations, diagnose, dedicated=None, gate=GATE, mlp_eligible=None)`:
  `dedicated=[]` turns stage B off; `gate="base"` and `mlp_eligible=True` rerun the comparison protocols the
  audit shows. It returns `stage_b`, `common_robustness` (ridge 0.01 / 0.05 on the common spec per decade),
  `gate`, `mlp_eligible`, `min_gain_asset` and `final.common` / `final.dedicated`.

Outcome on this build (the design's check values in brackets where they differ):

| | value |
|---|---|
| iteration 1 (train 1973-94, validate 1995-2004, test 2005-14) | add_gz, add_tp10_x_core4, drop_labor_share; validation −0.0114 → 0.0372 → 0.0625 → 0.0754; test 0.0162 → 0.1679: kept |
| iteration 2 (train 1973-2004, validate 2005-14, test 2015-01..2026-09) | add_real_y10, shocks_replace_oil, add_profit_growth; validation 0.1679 → 0.1867 → 0.2045 → 0.2112 [0.1951, 0.2016]; test on 131 months: base 0.1996, start 0.2795, selected 0.2681 [0.2707]; vs start −0.0114 [−0.0088], vs base +0.0685 [+0.0711]: not kept |
| common spec C | the iteration-1 selection, 「11 个因子」 |
| stage B (asset R² gains, iteration 1 val / test, iteration 2 val / test) | energy + metals_oil +0.0033 / +0.0078 / +0.0078 / +0.0150, accepted; gold from 1982-01 +0.0815 / +0.1711 / +0.1711 / +0.1081, accepted (1985-01 +0.0814 / +0.1583 / +0.1583 / +0.0864, not chosen); industrials + usd_em −0.0033 / +0.0313 / +0.0313 / +0.0057 [+0.0319 / +0.0068: the design added metals_oil to every equation]; gold + mortgage spread +0.0631 / +0.0804 / +0.0804 / +0.0546 (report-only) |
| final F | validation 0.1858, test 0.2678 on 141 months (base 0.2128) |

The iteration-2 numbers in brackets were computed with the structural shocks identified on the old spread
timing; the causality step re-identifies them on the fixed spread (e_financial correlates 0.70 with the old
series), which moves only iteration 2 (the decision and F are unchanged; F has no shocks).

### Results files

* **diagnostics.json** gains top-level `gate`, `mlp_eligible`, `min_gain_asset`, `stage_b` (`{candidates, common,
  min_gain_asset, min_coverage, r2_benchmark, final_rule, test_used_for_choice, iterations: [{iter, train,
  validate, test, rows: [{id, asset, option, label, change, eligible, evaluated, reason, val, test, val_gain,
  test_gain, chosen, accepted}]}], accepted, accepted_options, change, text}`) and `common_robustness`;
  `iterations[].test_result` gains `start`, `delta_vs_start` and `gate`; `iterations[].start` gains `label` and
  `chain`; `final` gains `common {spec, describe, changes, kept}`, `dedicated` and `last_selected.{delta,
  delta_vs_start, improved}`; `residual_analysis` gains `asset_train_start`, `asset_start_used`,
  `assets[a].fit_start`, `assets[a].own_factors` (Chow, Ljung-Box and the rolling betas use each asset's own
  equation and first month) and `leakage.wealth.quarter`. The recommendations answer the four questions
  (factor: 是; mechanism: 部分; tuning: ridge 0.05 not selected; network: 只作对照) and keep the regime layer's
  lookup (the 通胀状态依赖 title, `evidence.oos_delta`, `evidence.changes = ["add_tp10_x_core4"]`).
* **model_selection.json**: `spec` carries `asset_factors` / `asset_start`; new key `dedicated` (the accepted
  stage-B ids).
* **factor_audit.json** (`analysis/factor_audit.py`, new step): `{version, data_version {panel_sha1,
  model_selection_sha1, data_end}, archive {id, access_date, panel_sha1: "3869fdde3437", tag, docs, source},
  model {spec, describe, is_expected, expected_describe, common_describe}, summary, coverage, data_fixes,
  changes, plausible, families, per_asset, bh, refutation, forecast_states, upper_bound, residuals,
  model_vs_factor, out_of_universe, protocol, concepts, concept_counts, today, replication, untested, no_source,
  issues, notes}`. The sections that are wholly archived or recomputed say so with `source` (`archive`:
  archive, families, plausible, concepts, bh, upper_bound, model_vs_factor, `out_of_universe.archive_U4`;
  `recomputed`: residuals, `out_of_universe.final_common`, the recomputed refutation rows; `mixed`:
  forecast_states; protocol says which its rows are). The derived sections (summary, coverage, data_fixes, changes,
  per_asset, today, replication, concept_counts) carry no `source`: `summary.sources` names the origin of each
  summary number (counts and bounds archive, scores recomputed, the unexplained share mixed), and a caveat that
  quotes archived numbers is marked in `changes[].caveat_sources` (`"archive"` or null, aligned with `caveats`).
  Archived numbers carry the tag 「研究轮存档（2026-11-01，面板 3869fdde3437）」 on the page and are never presented as
  recomputed. `changes[]` (spread_timing, release_dating, sentiment_hold, add_gz, add_tp10_x_core4,
  drop_labor_share, energy_metals_oil, gold_window) each carry `definition`, `alignment`, `coverage`,
  `walkforward`, `replication`, `refutation`, `loo_in_final`, `loo_in_common`, `table {values, basis}`, `status`,
  `verdict` and `caveats` (the mandatory UI sentences with the build's numbers); a dedicated change's
  `loo_in_final` is its own asset's R² change and `dedicated.score_gain` the 10-asset average.
  `protocol.alternatives` are recomputed on this build (`source: "recomputed"`; the archive's rows stay in
  `archive_alternatives`). `replication.checks.sci` repeats the replication differences as text (the worst R² difference
  `"4.99e-07"`), because results files round every float to 6 decimals and it would read 0.0.
  `upper_bound.per_asset[].honest_range_T3` is each asset's 2015-26 range over the six honest fits (sets A and B
  × ridge, lasso, PCR, penalty chosen on the 10-asset score; the archive's `critique.upper_bound_common_per_asset`,
  from the critique's `upper_bound_r2.csv`), `upper_bound.beta_drift` the hindsight by-decade betas' adjusted R²
  on the U5 residuals (range 0.06-0.29). `residuals.fallback` / `residuals.note` say which refits fit an
  asset with its own start from the common start (gold in the 1985-1991 refits, the engine's 120-month rule).
  The design's d05 fitted gold from 1982-01 at every refit (36 months at the first one), so two of its check
  values differ by this rule, beyond its 1e-3 tolerance for gold: SSE kept vs the base 0.898 (d05 0.923; the
  other nine assets agree to 1e-6) and the PC1 share 0.5135 (0.5141, inside the tolerance).
  `today` percentiles are over the MonthDB's months (1973-02 on), as the design's d07.
* **panel_meta.json**: `column_info`, `release_lags`, the new notes and sources (see Conventions and Data
  sources).
* **meta.json** gains `view_concepts` (`config.VIEW_CONCEPTS`, concept id → Chinese name): the method tab's step 1
  groups the 30 view variables by it. The notes and sources follow panel_meta.json.
* **regimes.json** (regime layer): `transmission` fits each asset's own equation (`asset_features`,
  `asset_fit_start`; `engine_coef_check_max_abs_diff` 3.6e-13 with F) and says `labor_share_in_model`; the note
  reads 「劳动收入占比按发布月对齐，只在发布月变化」, and with labor share out of the model
  `n_sig_excl_labor_share` equals `n_sig_5pct`. `walk_forward_verdict.changes` is read from the diagnostics
  recommendation, and `model_use.transmission.reason` names the one kept inflation-state term when it is
  there.

### Scenario spec: W12-W14 and the driftable factors

* `drift_factor_ids(spec=None)`: with a spec, the model's common factors + its extras + `cpi` + `labor_share` (+
  `profit_share` while the valuation state is on); without one, as before. runner, translator, the server's
  custom run and the regime layer's conflict example pass `ctx.model.spec` to `validate_spec(...,
  model_spec=...)`. With F a `factor_drift` can move `gz` (bound ±2 × the annualised sd of `f_gz`),
  `tp10_x_core4` (in every month of the phase, so it amounts to assuming the phase is in high core inflation) and
  `metals_oil`; audit factors are refused.
* Warnings continue the regime layer's W1-W11:
  **W12** a view on a variable first observed more than 12 months after the MonthDB's first month:
  「阶段 {k}：{var} 从 {first} 才有数据，这一阶段只会抽 {first} 以后的历史月份（占 1973 年以来月份的 {share:.0%}）」
  (s_tips10 2003-01, 44%; s_mich1y 1978-01, 91%);
  **W13** a labour-share drift while labour share is not in the transmission (kept, warned; k_shaped and agi_boom
  under F): 「阶段 {k}：劳动收入占比不在今天的传导模型里（按发布月对齐后它不提高样本外 R²），它的漂移只改变模拟的劳动收入占比路径，不直接改变资产回报」;
  **W14** a view on an audit-only variable (`f_ebp`, `f_tp10_x_infl4`, the stage-B regressors `f_usd_em` and
  `f_mort_spread`, or any other `f_` column of the frame that is no config factor, `f_cpi` aside):
  「阶段 {k}：{var} 只用于审计对照，不能作为情景视图，已忽略」.
  `x_` columns are refused as unknown prefixes.
* The calculator (`library.resolve_calibration(spec, base, model_spec)`) adds `calibrate.LABOR_SHARE_CAVEAT` to
  the explanation and `engine_check.labor_share_in_transmission` (false under F).
* Scenario details gain `transmission {describe, factors, notes, interactions: [{factor, phases: [{drawn_share,
  database_share}]}]}`; each model asset's attribution lists only its own equation's terms (metals_oil only for
  energy). Eras attribution follows each asset's equation too; the 30 view variables are in each era's `states`
  but stay out of the era signatures.
* The 13 built-in specs are not edited in this round. The one new clip: financial_crisis phase 2 `s_spread
  mean 3.6` → 3.532 (the support's upper bound after the timing fix); `tests/test_scenarios.py` whitelists it
  (`KNOWN_CLIPS`) until the post-merge content review.

### LLM translator

The root-cause list is generated from `ctx.model.spec` and the `FACTORS` names (under F it no longer names
`f_labor_share`, and adds 「期限溢价（f_tp10；上月核心 CPI 同比 > 4% 的月份直接进入传导模型）」 and
「能源股另含基础金属相对油价（f_metals_oil）」). Variable rows gain `asset_driver_assets` (null = every modelled
asset; `["energy"]` for `f_metals_oil`; the prompt prints 是 / 是（仅能源股） / 否) and `late_from` / `late_share`.
A section 「## 新增的情景变量」 follows the variable table: the 30 view variables by concept (`VIEW_CONCEPTS` order),
the late ones with their first month and the W12 rule, the release-dating note, the GZ rule (only when `f_gz`
is a model input) and the TP-term note. The regime layer's 「## 宏观体制（regime 字段）」 stays before 「## 覆盖项」.
The prompt stays deterministic per build.

### Data sources

* **FRED through the API, graph CSV as fallback** (`data/sources.py`). `fred(series_id, eop=False,
  refresh=False, agg=None)` reads the API JSON cache (`data/raw/fred/<ID>.json`, `<ID>_m_eop.json`,
  `<ID>_m_avg.json`), then the graph CSV cache (`<ID>.csv`, `<ID>_eop.csv`), then downloads: through
  `https://api.stlouisfed.org/fred/series/observations` when `FRED_API_KEY` is set (`env.load_dotenv()`; build.main
  calls it right after parsing its arguments), else the graph CSV. When the API fails with a key set (FRED
  rejects the key with HTTP 400, or the API is down after its retries), the keyless graph CSV is tried once, so a
  bad key in `.env` does not stop a build; the warning and any final error name only the series, the status and
  the exception type. `fred_meta(series_id)` caches
  `/fred/series` metadata as `<ID>_meta.json`. Weekly and daily series keep their observation dates;
  panel.py aggregates them (`last_in_month`, `mean_in_month`). A graph CSV download (keyless, or the fallback)
  moves an older API JSON of the same request to `fred/superseded/` so the new CSV is not shadowed.
* **Key hygiene.** The key is only the `api_key` query parameter, sent only to `api.stlouisfed.org` (redirects
  off). It is never in a file name, a cache body, a log line, an exception, panel_meta.json, meta.json or
  SOURCES.json: a failed call raises a fresh `RuntimeError("FRED API 下载失败：<ID>（HTTP <status> / <type>）")`
  (a `requests` message contains the full URL, so it is never re-raised or printed), recorded URLs read
  `…?series_id=<ID>&api_key=REDACTED&file_type=json`, and a body that echoes the key is refused. Politeness: at
  least 0.6 s between calls, 4 attempts with 1.5·k s backoff on HTTP 429 / 5xx and timeouts; a body without an
  `observations` list is a failure.
* **Publisher downloaders** (keyless; each checks the content before caching): `fed_ebp()` (Federal Reserve
  Board EBP csv; required for `gz`), `nyfed_acm()` (ACM term premium, xls; required for the TP term),
  `worldbank_pinksheet()` (the monthly Pink Sheet, link discovered on the World Bank's commodity-markets page,
  restricted to worldbank.org; required for `metals_oil`), and the optional `freddiemac_fmhpi()`,
  `shiller_home()` (Fig. 3-1), `iacoviello_gpr()`, `epu_news()` and `nyfed_hlw()`. A required source raises when
  neither a cache nor a download is available; an optional one returns None and leaves its columns empty with a
  panel_meta note. New FRED ids: WBAA and WGS10YR (required) plus 28 optional view inputs. `openpyxl>=3.1`
  joins the dependencies (Pink Sheet, HLW, EPU).
* **Importing the research downloads** (one-off): `python -m scenariolab.data.import_raw <factors/raw>
  [--force] [--dest DIR] [--families DIR]` copies bytes only (FRED responses and the families' sources.json are
  parsed as JSON data; nothing is executed or opened with macros), checks recorded SHA-256 (and the 12-digit
  SHA-1 prefixes some families used), refuses to overwrite a different file without `--force`, refuses any file
  that contains the configured key, admits a FRED observations file to the cache only if it is a full, unpaged,
  untransformed request (`observation_start` 1600-01-01, `observation_end` 9999-12-31, `units` lin, `count ==
  len(observations) <= limit`; else `fred/alfred/`), files vintages under `fred/alfred/` and searches under
  `fred/searches/`, and writes `data/raw/SOURCES.json` (one record per file: url with `api_key=REDACTED`,
  publisher, access_date, terms, frequency, first_month, publication_lag, revised, copyright, sha256) and
  `data/raw/import_log.json`. Where a series exists both as the project's CSV and an imported JSON, the JSON
  wins (a test checks they agree).
* Third-party copyright (SOURCES.json, `column_info.copyright`, the method tab's data card): Moody's (BAA, DBAA,
  WBAA via FRED), Freddie Mac (PMMS, FMHPI), University of Michigan (UMCSENT, MICH), Robert J. Shiller (ie_data,
  Fig. 3-1). `data/` stays git-ignored: nothing is redistributed.

### The build

`STEPS = [panel, causality, eras, diagnostics, factor_audit, backtest, scenarios, regimes, forecast, valuation,
portfolio, llm, meta, ablation, regime_forecast]` (the order shared with the regime layer), `OPT_IN =
{"ablation", "regime_forecast"}`.

| Step | Code | Reads | Writes (data/…) |
|---|---|---|---|
| factor_audit | `analysis.factor_audit.run` (selection context) | panel + shocks, diagnostics.json, model_selection.json, the frozen archive (package data) | results/factor_audit.json |

`factor_audit` runs on the selected model after diagnostics and before backtest; it needs diagnostics.json
(without it the walk-forward rows are empty and `issues` says so) and reads nothing outside `data/` and the
package. Two modes: *replication* rebuilds the research round's base B0 from the `x_` columns and recomputes
the registered candidates (`REGISTERED`: the spread fix, GZ, the TP term and metals/oil required, plus
tp10_x_infl4 and the mortgage spread) and three forecast states; it must reproduce the archive to 1e-6 on the
deltas, 1e-4 on t / Stouffer z (the archive's inputs were stored with 6 significant digits) and exactly on the
counts (`replication.checks.ok`). *Current* scores B0, B1 and the selected model, the data-fix deltas, each
change alone on B1, the dedicated gains, leave-one-out in F and in the common spec, the residual anatomy, the
six Fama-French industries the project does not use, coverage from `column_info`, today's readings, and the
§2.3 protocols (one walk-forward run with the networks eligible; the old gate's rows are read off the
two-iteration runs). CLI: `python -m scenariolab.analysis.factor_audit`.

`analysis/factor_tests.py` is the validated core of the research harness: the same-sample transmission test
(`Bench.transmission`: delta and `delta_same_sample`, Diebold-Mariano and Clark-West t with Newey-West, Stouffer
z over decades with ≥ 120 training months), the pooled 10-asset score test, `bh()`, `circular_placebo` (seed
`[20261101, crc32(id)]`), `refutation`, `predictive_test` (the KEY_STATES baseline, monthly and year-end origins),
`oos_residuals` (through `AssetFactorModel`, so per-asset extras and windows follow the 120-month fallback) and
`residual_pca`.

Timings of the full build of 2026-11-02 (`--all --skip panel`, other jobs on the machine at the same time, so
the times are upper bounds): causality 6.5 s, eras 1.1, diagnostics 15.7, factor_audit 99.7 (18.6 s when rerun alone after the protocol rows were changed to one walk-forward run per candidate set), backtest 68.0,
scenarios 29.6, regimes 92.8, forecast 37.3, valuation 358.7, portfolio 29.1, llm 1.8, meta 0.0, ablation
508.6, regime_forecast 96.6; 1346 s in all. The panel step (no downloads) takes about 3 s. The forecast step refused to
append a second 2026-09 entry to `data/ledger/` (an origin is recorded once); the ledger is re-recorded after
this branch and the regime layer merge.

### HTTP API and web app

`GET /api/factor_audit` serves factor_audit.json (`server.RESULT_NAMES`, `BUILD_STEP["factor_audit"] =
"factor_audit"`); a missing file gives 404 naming `--only factor_audit` and its prerequisite diagnostics.json.
The diagnostics tab gains the section 「缺失因子审计」 (drawn by `web/js/tabs/_factor_audit.js`; deep link
`#diagnostics?audit`, sub-anchors `?audit-families`, `?audit-assets`, `?audit-concepts`; loaded optionally like
the ablation), a stage-B card 「专门因子（只作用于单个资产）」 and the gate wording. Inside the section the
archived numbers carry the 「研究轮存档」 chip and the recomputed ones 「本次构建重算」 (the protocol table by its
`source`, a caveat by `caveat_sources`); the seven PLAUSIBLE transmission candidates get their own table; the
coverage table calls a factor column 传导模型 only when the selected model uses it (else 候选因子). The method
tab gets step 12 「缺失因子审计」 after the regime layer's step 11, and step 6 says what the TP term needs (drawn
high-core-inflation months); the overview gets the 「信用与期限」 line under today's macro state; a scenario
detail whose model has the TP term shows, per phase, the share of drawn months in which it is on next to the
month database's (`transmission.interactions`). The lab's variable table marks a dedicated factor 「是（仅石油能源股）」
and a late variable's first month; the eras tab's model check compares `asset_factors` / `asset_start` too. When
this build's engine ablation picks a variant other than the one the engine runs, the diagnostics tab says so
instead of 「引擎已经改用消融选出的…」.

### Testing

New files: `test_sources.py` (FRED API and CSV paths, the keyless graph-CSV fallback when the API rejects the key
or is down, key hygiene, politeness, the publisher parsers; network mocked, fake key), `test_panel.py` (spread
timing, release dating, a 20-draw no-look-ahead property that cuts each quarterly input at its publication month,
so a quarter shown before it was published fails, the TP term, coverage and `column_info`, the Törnqvist index), `test_factor_audit.py` (replication, BH and placebos,
residual anatomy, the JSON contract, out of universe, the protocol rows). Additions in test_engine and
test_attribution (per-asset equations, fallbacks, bit-identity of the base path), test_diagnostics (28
candidates, the incremental gate, stage B, the walk-forward on the panel: test 19 checks iteration 2 against
the value of whichever structural shocks are on disk), test_scenarios / test_translator / test_eras (W12-W14,
driftable factors, the root list, the new prompt section, eras by equation), test_server and test_build (the
endpoint, the shared step order, `load_dotenv` before the panel).
