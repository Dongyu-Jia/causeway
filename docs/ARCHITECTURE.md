# Architecture and module contracts

This file is the contract every module codes against. If you change a shape
here, change every producer and consumer of it.

Example numbers come from the build of 2026-11-02 (`meta.json` `built_at`
2026-11-02T01:02:07Z, data through 2026-09). They move with every data
update; the shapes and rules do not.

## Layout

```
scenariolab/
  __init__.py              package version
  config.py                shared ids and engine settings: paths, MODEL_START, HORIZON_MONTHS, BLOCK_MONTHS,
                           RARE_MONTHS, RESIDUALS, EXOGENOUS_SHOCKS, ERAS, ASSETS, FACTORS, STATES, VALUATION,
                           REGIMES
  env.py                   minimal .env loader (KEY=VALUE lines; never overrides a variable that is already set)
  jsonio.py                JSON I/O: numpy/pandas-safe, NaN/inf -> null, floats to 6 decimals, atomic writes
  context.py               EngineContext (panel + selected model + MonthDB + Anchor + CMA); get_context()
  build.py                 CLI: python -m scenariolab.build (see "The build")
  server.py                FastAPI app + static UI: python -m scenariolab.server (see "HTTP API")
  data/sources.py          cached downloaders: FRED graph CSV, Ken French zips, Nareit workbook, gold CSV,
                           Shiller ie_data.xls (optional: a failed download leaves CAPE empty)
  data/panel.py            builds data/processed/panel_monthly.csv + panel_meta.json
  engine/factor_model.py   ModelSpec + AssetFactorModel (linear / lags / inflation regime / ridge / window / MLP)
  engine/ep.py             entropy pooling: entropy_pool(), apply_views(), blend(), support(), weighted_corr()
  engine/pooling.py        phase weights: views solved on block starts; MIN_ENS / MIN_EPISODES; shrink lambda
  engine/sampler.py        weighted block bootstrap: BlockProbs, sample_phases()
  engine/residuals.py      residual draw modes: coupled, decoupled, detrended, trend_decoupled, partial, recentered
  engine/simulate.py       MonthDB, Anchor, capital_market_assumptions(), ValuationModel, simulate(),
                           Attribution (the additive split of each simulated return, attribution=True)
  engine/summary.py        wealth paths, fans, representative path, Brownian-bridge candles, correlations
  engine/regimes.py        the macro-regime classifier (numpy, pandas, config only): classify(), classify_panel(),
                           classify_sim(), regime_columns() / asof_columns() (the rg_ columns), nowcast(),
                           validate_nber(), rule_text() and the conventions (runs, window_mask, rng_for, …)
  scenarios/schema.py      spec schema (YAML and the LLM JSON schema), validate_spec(), allowed_variables()
  scenarios/runner.py      run_spec(ctx, spec) -> (detail dict, SimOutput); phase_pools(); summarize(); headline();
                           attribution_summary() (assets[a].attribution); reference_simulation() (the engine's
                           unconditional reference run the regime readouts compare with)
  scenarios/regime_views.py   the `regime` phase key (aliases, expansion into rg_ views), checks W1-W11, the
                           regime readout (detail["regimes"]), the library's regime fields, the translator's rule
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
  analysis/regimes.py      -> results/regimes.json (macro regimes: nowcast, NBER validation, assets, episodes,
                           transitions, forward returns, eras, the engine's regimes, why they stay out of the model)
  analysis/regime_forecast.py  -> results/regime_forecast.json (does the real-time regime improve the one-year
                           forecast? opt-in build step)
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
  js/tabs/<id>.js          one module per tab (13, see "Web app")
  js/tabs/_scenario_detail.js, js/tabs/_weights.js   shared helpers (a leading underscore = not a tab)
tests/                     pytest suite (18 files, see "Testing")
data/                      raw/, processed/, results/, ledger/ (git-ignored, built locally; ledger/ is the
                           append-only record of live one-year forecasts, see "Forecast ledger")
docs/                      ARCHITECTURE.md (this file), screenshots/ (the README's four PNGs)
.github/workflows/ci.yml   CI on every push and pull request (see "Testing")
CONTRIBUTING.md            setup, build, tests, adding a scenario or a data source (Chinese + English summary)
.env.example               template for .env: ANTHROPIC_API_KEY, MSL_LLM_MODEL, MSL_PORT
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
  present).
* The `rg_` family (macro regimes, engine/regimes.py) is computed on load and
  never stored: `load_panel_with_shocks()` appends `regimes.regime_columns(panel)`
  after the `e_` merge when the panel has the three inputs (s_cpi_yoy,
  lvl_unrate, lvl_ip), about 0.06 s. 18 columns, 0/1 floats, NaN where the
  label is unknown (before 1948-01, or a month without its own CPI,
  unemployment and IP readings): the ex-post label one-hot `rg_expansion`
  `rg_inflation` `rg_deflation` `rg_recession` `rg_depression`
  `rg_stagflation`, the ex-post flags `rg_contraction` `rg_severe`
  `rg_high_infl` `rg_cpi_neg`, and the real-time (one-sided) `rg_rt_<label>`
  (6), `rg_rt_contraction`, `rg_rt_severe`. They are never written to
  panel_monthly.csv: an ex-post label depends on later data, so a stored column
  would be right only for windows ending today, and a parameter change needs no
  panel rebuild. Scenario views may use the ex-post ones (see "Scenario spec");
  `rg_rt_` views are refused.
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
  returned dict for causality, eras, diagnostics, backtest, regimes, forecast,
  valuation, meta, ablation and regime_forecast; scenarios.library, portfolio.service and
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
python -m scenariolab.build --only scenarios regimes forecast valuation portfolio llm
python -m scenariolab.build --paths 1000        # paths per built-in scenario (default 600)
python -m scenariolab.build --only ablation     # the opt-in engine ablation (~5 min)
python -m scenariolab.build --only regime_forecast   # the opt-in regime forecast test (~2 min)
python -m scenariolab.build --all               # every step, both opt-in steps included
```

`STEPS = [panel, causality, eras, diagnostics, backtest, scenarios, regimes,
forecast, valuation, portfolio, llm, meta, ablation, regime_forecast]` always
run in this order, whatever order `--only` lists them in. `OPT_IN =
{"ablation", "regime_forecast"}` is left out of the default set: `--all` adds
them, `--only` picks any steps (opt-in ones included), `--skip` removes steps
from either. `--paths N` changes only the scenarios step
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
| backtest | `analysis.backtest.run` | panel, model_selection.json | results/backtest.json |
| scenarios | `scenarios.library.run` | specs/*.yaml, model_selection.json | results/scenarios.json, results/scenarios/<id>.json, results/scenario_paths.npz |
| regimes | `analysis.regimes.run` | panel (+ the rg_ columns), the MonthDB, the cached reference run (`runner.reference_simulation`, shared with the scenario readouts), diagnostics.json (the transmission walk-forward verdict, optional), data/raw/fred (the 2025-10 interpolation note, cache only, optional) | results/regimes.json |
| forecast | `analysis.forecast.run` | panel, model_selection.json, scenario_paths.npz (live scenario mixture), results/scenarios/<id>.json (each built-in's validated spec, re-run with runner.run_spec at library.seed_for(id) and the npz path count for its year-one macro paths; this reproduces the npz exactly, `live.reconciliation.paths_reproduced`), data/ledger/*.json | results/forecast.json; appends data/ledger/forecast_<origin>.json |
| valuation | `analysis.valuation.run` | panel, forecast.json (analog_share, reproduction check), backtest.json and scenarios.json (consistency checks), data/raw/shiller (cache only, never downloads) | results/valuation.json |
| portfolio | `portfolio.service.run` | the context (build_continuation), scenario_paths.npz, scenarios.json | results/continuation_paths.npz, results/portfolio_default.json |
| llm | `llm.translator.run` | specs/*.yaml (few-shot examples), the MonthDB | results/llm_translator.json |
| meta | `build.build_meta` | panel_meta.json + `data.panel.SOURCES` (`meta_sources`), the context, `translator.status()` | results/meta.json |
| ablation (opt-in) | `analysis.ablation.run` | panel, backtest.json (reproduction check), scenarios.json | results/ablation.json |
| regime_forecast (opt-in) | `analysis.regime_forecast.run` | panel (real-time labels from `panel.loc[:t]` only), model_selection.json, forecast.json (optional: only the check that the module's own re-selection gives forecast.json's share and analog medians, `baseline.forecast_json`) | results/regime_forecast.json |

The context is loaded with `selection=False` before causality and again after
it (to merge the fresh shocks); after diagnostics it is loaded with
`selection=True`, so backtest and every later step use "today's transmission"
from model_selection.json. A default build without downloads takes about 5
minutes, most of it the valuation step. The last build of every step but panel
and ablation (`--only causality eras diagnostics backtest scenarios regimes
forecast valuation portfolio llm meta regime_forecast`, 2026-11-02) took 388 s
(peak RSS 2.1 GB): causality 2.8 s, eras 0.4, diagnostics 9.7, backtest 22.2,
scenarios 11.3 (13 regime readouts and the reference run included), regimes
37.5, forecast 16.1, valuation 164.2 (`valuation.json` `settings.seconds`
164.1), portfolio 22.0 (most of it the 40-year continuation), llm 1.0, meta
0.0, regime_forecast 100.7 (its walk-forward 80 s); 287 s for the default
steps. `--only regimes llm` ran once more after it (41 s), so regimes.json's
`forward.forecast_test` sees the new regime_forecast.json. The opt-in ablation
took 308 s in an earlier full build (peak RSS 2.6 GB).

Partial rebuilds:

* After editing a built-in spec, or after `--only scenarios`, rebuild
  regimes, forecast, valuation, portfolio and llm as well
  (`--only scenarios regimes forecast valuation portfolio llm`): regimes.json's
  `engine.reference` and the scenario details' `regimes.phases[].reference`
  blocks come from the same cached reference run (the build prints a reminder
  when scenarios ran without regimes, and the 宏观体制 tab warns when
  `regimes.data_end` differs from the scenario start), forecast mixes the
  scenario paths into its live forecast, valuation compares its own copy of
  the scenario results with scenarios.json (the 估值与盈利 page asks for
  `--only valuation` when they differ), portfolio_default.json is computed on
  the paths, and the translator's few-shot examples are built-in specs.
  The portfolio step also rebuilds continuation_paths.npz, whose start must
  match the scenario paths' first label.
* After `--only regimes`, rebuild llm too: the translator's prompt quotes the
  study numbers of regimes.json `engine.studies` only when its `data_end` is
  the anchor month (a full build runs llm after regimes).
* After `--only forecast`, rebuild valuation (its forecast block re-runs the
  42 origins at forecast.json's `analog_share`, `settings.baseline_reproduces_forecast`)
  and, if it is kept, regime_forecast.
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
[--print-prompt]`. analysis.backtest, forecast, diagnostics, eras, causality,
regimes and regime_forecast also run as `python -m scenariolab.analysis.<name>`
and write their own result file (forecast also appends to the ledger); `python -m
scenariolab.portfolio.service` loads an EngineContext itself and writes
continuation_paths.npz and portfolio_default.json.

## Engine settings (config.py)

| Setting | Current value | What it changes |
|---|---|---|
| `MODEL_START` | `"1973-01"` | first month of every model fit and of the MonthDB; earlier eras are descriptive only |
| `HORIZON_MONTHS` | 180 | months per scenario path (15 years); validate_spec pads or trims the phases to it |
| `BLOCK_MONTHS` | 6 | sampler block length; each phase's views are solved on block starts of length `min(6, phase months)` |
| `RARE_MONTHS` | 2020-03..2020-08, weight 0.25 | prior weight of the 2020 lockdown months; a block start that would draw any of them gets the factor in full |
| `RESIDUALS` | `{"resid_mode": "partial", "resid_coupling": 0.5}` | the residual draw of every simulation (scenarios, backtest, forecast, the valuation study's baseline); chosen by analysis/ablation.py, whose own baseline is the pre-adoption coupled draw (its `partial_50` variant is the engine) |
| `EXOGENOUS_SHOCKS` | 7 events, 1973-10 (war and oil embargo) .. 2026-03 (oil supply shock, WTI +54% in a month) | one-year forecast years that overlap one are flagged, every forecast score is also reported without them, and a large miss in such a year is attributed to the shock |
| `VALUATION` | see below | the valuation state |
| `FACTORS` | 21 ids | `core` (10): the default transmission and the factors a factor_drift can move; `endogenous` (wealth, valuation, cape): built partly from asset prices, never accepted by the walk-forward, a view on them warns; `optional` (cape): its source may be missing, and a missing month counts as no change |
| `STATES` | 30 ids | level variables (era fingerprints, views, today's state); `endogenous` (cape) warns as a view |
| `ERAS` | 9 eras, 1955-01 .. today | the era of a month (`simulate.era_of`), fingerprints, a phase's era mix; the simulator draws only months from 1973 on |
| `ASSETS` | 13 ids | ids, names, data proxies, `group`, and the `income` source (dividend / coupon / interest / none) |
| `REGIMES` | see below | the macro-regime classifier (engine/regimes.py) and its conventions; changes no simulated number |

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

`REGIMES` (the one block of conventions every regime consumer uses):

| Key | Value | Meaning |
|---|---|---|
| `params` | ip_swing_pct 3.0, u_rise_pp 0.5, u_smooth_months 3, severe_ip_decline_pct 15.0, high_inflation_pct 4.0, cpi_neg_pct 0.0, core_low_pct 1.0 | contraction = IP 3% below its peak confirmed by a 0.5 pp rise of the 3-month unemployment rate from its low in the leg; severe = the leg's IP fall reaches 15%; high inflation = CPI y/y > 4% in the same month (the transmission dummy of `ModelSpec.regime` uses the previous month; a test asserts both thresholds are 4.0); cpi_neg = CPI y/y < 0; core_low = core CPI y/y < 1% |
| `labels` | expansion 扩张, inflation 通胀, deflation 物价负增长 (term 通缩), recession 衰退, depression 深度衰退 (term 萧条), stagflation 滞胀 | id, Chinese name, the user's term it maps, colour and description, in code order 0..5. Precedence: 滞胀 > 深度衰退 > 衰退 > 物价负增长 > 通胀 > 扩张 |
| `first_month` | "1948-01" | earlier months are unknown (NaN rg_ columns) |
| `nber_window` | ("1948-06", "2020-12") | the window of the NBER validation (NBER dates only score the rule, they never enter a label) |
| `min_episodes` | 4 | fewer episodes = thin (†, no placebo p; no interval below 2) |
| `bootstrap` | 4000 replicates, seed 20261101, 90% (5, 95) | episode bootstrap; every cell seeds its own generator, `regimes.rng_for(section, window, group, asset, stat)` |
| `hac_lags` | monthly 6, forward 12 | Newey-West lags |
| `forward_lag_months` | 1 | a label is known the month after: forward windows run t+2..t+13 (t+1..t+12 is the `base` sensitivity) |
| `placebo_min_shift` | 24 | circular shifts of the real-time labels by at least 24 months |
| `reference` | n_paths 2000, seed 4242 | the engine's unconditional reference run (`runner.reference_simulation`) |
| `longest_contraction_months` | 18 | the longest contraction since 1948 (W7, W11) |
| `thin_runs` | 3 | W9: fewer runs since 1973 than this |

Engine constants outside config.py, each set by the evidence in its module
docstring: `engine/pooling.py` MIN_ENS 30, MIN_EPISODES 6, EPISODE_MONTHS 12,
SHRINK_STEPS 14; `engine/simulate.py` DEFAULT_REVERSION, RATE_SCALING,
SPREAD_WIDENING_FLOOR 0.85, OIL_VIEW_VARS; `engine/residuals.py`
DEFAULT_WINDOW 36, SEED_OFFSET 7919; the spec bounds in `scenarios/schema.py`;
`scenarios/calibrate.py` UNRATE_VOL_MULT 0.62; `scenarios/probabilities.py`
GAMMA 0.5, WINDOW 6, KEY_STATES; `scenarios/runner.py` ATTRIBUTION True (the
module switch for run_spec's attribution), ATTRIBUTION_BAND (10, 90),
REGIME_READOUT True (the switch for detail["regimes"]), _REFERENCE_KEEP 2
(reference runs kept per MonthDB); `scenarios/regime_views.py`
SUPPORT_MONTHS 12, DELIVERY_TOL 0.005, CONFLICT_SD 1.5, CONFLICT_GAIN 0.05,
CONTRACTION_SHARE_MIN 0.3, NOTE_REQUEST_GAP 0.10, NOTE_CATCH_UP_GAP 0.20,
NOTE_TERM_SHARE 0.5, NAME_TERMS; `engine/regimes.py` THIN (= min_episodes),
EPS 1e-9;
`analysis/eras.py` ATTR_MIN_COVERAGE 0.5, OWN_BETA_MONTHS_PER_PARAM 5;
`analysis/backtest.py` WINDOWS, VIEW_VARS, PHASE_MONTHS 24, N_PATHS 400,
SEED 11, CUSTOM_PATHS 200, CUSTOM_PATHS_RANGE (100, 400), CUSTOM_MONTHS_RANGE
(12, 180), MIN_TRAIN_MONTHS_CUSTOM 180, SETUP_CACHE_MAX 8;
`analysis/forecast.py` FIRST_ORIGIN 1984-12, KEY_STATES, ANALOG_SHARES (1/32,
1/16, 1/8, 1/4, 1/2 of the starts available at each origin), ANALOG_SHARE 1/4
(without a selection), FLAT_BANDWIDTH 999, CLAIM_SKILL 0.005, CLAIM_T 2.0, SPLIT
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
attribution=None, regimes=None)` runs validate_spec -> `pool_phase` per phase
-> the regime checks W8-W11 (`regime_views.spec_checks`, appended to
`validation.warnings`) -> `with_oil_lock` -> `simulate` -> `summarize` ->
`attribution_summary` -> the regime readout. Built-ins use the seed crc32(id)
(`library.seed_for`); custom runs default to 7. `attribution=None` means the
module switch `runner.ATTRIBUTION` (True: built-ins, custom and LLM runs
through the API, and the studies that call library.run all get
`assets[a].attribution`, about 30 ms per 600-path run); a study that needs
only the returns can set it to False, as it swaps `runner.RESIDUALS` and
`runner.VALUATION`. `regimes=None` means `runner.REGIME_READOUT` (True): the
detail gets `regimes` (see "Results files"), about 0.25 s per 600 paths
(`regimes.classify_sim`); the ablation, the valuation study, the forecast's
reconciliation re-runs and the regimes step's own studies pass
`regimes=False` (`library.run(..., regimes=False)` passes it through). Every
simulated number is bit-identical with and without either (tested on
baseline, stagflation and financial_crisis, with and without the rg_
columns).

`runner.reference_simulation(ctx) -> dict` is the engine's own unconditional
run that every regime readout and regimes.json `engine.reference` compare
with: the spec `{phases: [{name: 回归常态, months: 180, views: []}]}` through
run_spec's own steps (validate_spec, phase_weights, with_oil_lock,
valuation_for, simulate) with `config.REGIMES["reference"]` (2000 paths, seed
4242), without detail, attribution or readout, so it is bit-identical to
`run_spec(ctx, <that spec>, 2000, 4242)` and never calls run_spec. Returns
`{sim, regimes: {the classify_sim label, flag and episode arrays, (2000, 181),
column 0 = the anchor}, drawn: (2000, 180) label codes of the drawn months,
n_paths, seed, months, spec}`; every array is read-only and shared. It is
cached per MonthDB (a WeakKeyDictionary under a lock, so a request that
arrives during the first run waits instead of computing a second copy), keyed
by the anchor date, paths, seed, horizon, `runner.RESIDUALS`,
`runner.VALUATION` and the CMA (method and adjustment); a db keeps the 2 most
recently used settings (`_REFERENCE_KEEP`). About 1 s the first time and
72 MB; the server warms it at startup.

### Transmission model (engine/factor_model.py)

`ModelSpec(factors=<10 core>, lags=0, regime=None | "inflation", kind="linear"
| "mlp", ridge=0.0, window_years=None, target="p", shock_cols=(), hidden=16,
name="base")`. One regression per model asset (every asset but cash) of the
month's price return on the same month's factor changes. `regime="inflation"`
adds interactions with a dummy for CPI YoY > 4% in the previous month; `mlp`
is a 3-seed ensemble of one tanh hidden layer. Treasury, crude oil and cash
are fitted for the transmission charts only; simulations build them
structurally. The walk-forward (analysis/walkforward.py) kept the base spec:
10 core factors, linear, no lags, full sample (model_selection.json).

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

**As-of regime columns.** `MonthDB(panel, model, start, end)` with an `end`
before the panel's last month replaces the frame's ex-post `rg_` columns by
`regimes.asof_columns(panel, end)`, computed from `panel.loc[:end]` only;
without it a window ending 2008-06 would see 2008-01..06 already labelled
收缩 / 深度衰退, and one ending 2020-03 would see 2020-03. The `rg_rt_` columns
are one-sided and stay as they are; `db.dates` are unchanged (644 months).
`db.panel` keeps the full-sample rg_ columns, so an early window must read
regimes from `db.frame`. The cache is keyed by `id(panel)`, checked against a
weak reference (dropped with the panel), then by the last month, a blake2b
digest of the input series up to it and the params (an in-place edit
invalidates it), at most 256 windows per panel; about 35 ms per new window.
No current backtest or forecast view uses rg_ columns: this guards any future
one.

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
    regime: {通胀: 0.5}          # optional: shares of macro-regime months among the drawn ones (see below)
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
| view `var` | starts with `s_`, `f_`, `e_` or `rg_` and is a MonthDB column with history; anything else is dropped with a warning; a view on a real-time `rg_rt_*` column is dropped with W4 |
| `regime` | `{id or Chinese alias: share}` (YAML) or `[{id, share, why}]` (LLM form). Ids: the six labels expansion, inflation, deflation, recession, depression, stagflation plus the flags contraction and high_infl; aliases 扩张 / 金发姑娘 / 软着陆, 通胀 / 过热, 通缩 / 物价负增长, 衰退, 萧条 / 深度衰退, 滞胀, 收缩, 高通胀. Each entry becomes a plain view `{var: rg_<id>, mean: share, why: 宏观体制：…}`, placed first, replacing a view on the same column and not counted in the 12-view cap; shares are clipped to 0..1 (listed in `clipped` as `regime.<id>`), and labels summing to more than 1 are rescaled (W3). The clean phase keeps `regime: {id: share}` only when one was asked, so re-validation is idempotent and the 13 built-ins (none uses it) are unchanged. The views are clipped to the block support like any view (衰退 1.0 -> 0.373) |
| `mean`, `min`, `max` | clipped to the 2nd-98th percentile of what the phase can average (`pooling.view_support`: means of B-month runs, much narrower than single months) |
| `mean_annual`, `min_annual`, `max_annual` | the same bounds x 12 (the value / 12 is the monthly target); `min_annual` / `max_annual` exist in YAML only |
| `shift_sd` | -3..3 prior sds of the phase average |
| `vol_mult` | 0.3..3 |
| `why` | <= 200 chars |
| a view with no target | dropped with a warning |
| endogenous variables | a view on f_valuation, f_wealth, f_cape or s_cape warns that it dictates equity prices |
| IDENTITIES | s_real_ff = s_ff - s_cpi_yoy and s_term_spread = s_y10 - s_ff: when a phase gives a mean-type target (mean, mean_annual or shift_sd) for all three, the derived one is dropped |
| `factor_drift` | only `drift_factor_ids()`: the 10 core factors and cpi (plus profit_share while VALUATION is enabled). Other FACTORS ids are dropped with a warning (they only pick months; a drift on them changes no path), unknown ids too. Bound: +-2 x the factor's annualized monthly-change sd on the MonthDB (llm_translator.json `factor_drift_bounds`; currently unrate +-3.24 pp/yr, oil +-0.71 and labor_share +-0.024 log/yr) |
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
`warnings`.

**Regime checks** (scenarios/regime_views.py). Asking for a regime is not
getting it, so every regime request (and every plain view on an ex-post rg_
column) is checked. validate_spec runs W1-W7 without the database; run_spec
(after pooling) and `translator.preview` append W8-W11. Every number in a
warning comes from the frame or `config.REGIMES` at validation time; how much
a request delivers in simulated paths lives in regimes.json `engine.studies`
and is never typed into a warning.

| id | trigger | warning (abridged) |
|---|---|---|
| W1 | unknown regime id | 阶段 k：未知的宏观体制 …，已忽略（可用：…） |
| W2 | share not a number | 阶段 k：宏观体制 … 的占比不是数值，已忽略 |
| W3 | the six labels sum to more than 1 | 六种体制互斥，占比之和 … 超过 1，已按比例缩到合计 1 |
| W4 | a view on `rg_rt_*` | … 是实时标签（只用于现况判断和一年期预测检验），情景视图请用事后标签 rg_<体制>，已忽略 |
| W5 | deflation or cpi_neg requested | the MonthDB's 物价负增长 months (8 now: 2009-07..10, 2015-01..04) are base effects; use f_cpi mean_annual (support lower bound about -0.008/年) with 收缩 0 and an f_unrate view |
| W6 | an inflation-type id (inflation, stagflation, high_infl) in a phase under 12 months | simulated CPI y/y takes about 12 months to catch up with the drawn months |
| W7 | a contraction-type id (contraction, recession, depression, stagflation) with share x months > 18 (`longest_contraction_months`) | more contraction months than the longest contraction since 1948; split the phase |
| W8 | share above what a 12-month phase can deliver (`max_delivered` of the support table, + 0.005) | … 最多只能占抽中月份的 …（要求 …） |
| W9 | fewer than `thin_runs` = 3 runs since 1973, or fewer than 2.5 effective episodes | only these few episodes would be repeated (the engine's own thinness check counts every drawn month and cannot see it) |
| W10 | another mean-type view more than 1.5 block-sd from the regime's own drawn mean | … 的目标 … 与所选体制的历史月份（平均 …）相差 … 个标准差 |
| W10b | the regime views pool alone at lambda_base but with the other views only at a lower lambda | names the views whose removal lifts lambda most (leave-one-out, gains >= 0.05) |
| W11 | any spec: consecutive phases drawing >= 30% contraction months add up to more than 18 expected contraction months | 区块首尾相接会造出比历史上任何一次都更深更长的收缩 |

The support table (`regime_views.support_table(db)`, cached per MonthDB;
llm_translator.json `regime_support`, regimes.json `engine.support`) is what a
12-month phase with the view at 1 delivers: today 扩张 100% (15 runs), 通胀
100%, 收缩 100%, 高通胀 100%, 滞胀 93% (4 runs, lambda 0.928), 深度衰退 76% (2 runs,
1.01 effective episodes: 2008-09), 衰退 37% (1 run, 2001), 物价负增长 33% (2
runs). No built-in trips W1-W11. Example: stagflation phase 1 with 滞胀 0.8 is
infeasible with its own views (lambda 0.65; s_real_ff and s_ip_yoy named by
W10b). The pools of W10 / W10b (one per pair and per left-out view, a second or
more each when the views conflict) are cached per MonthDB and (months, views),
so `translator.preview` and the run of the same clean spec pool them once
(about 13 s, then 2.6 s for the run, for that example at 200 paths).
 The UI shows the report next to every result (a `phase: null` row
shows — as its phase).

**LLM form** (`SPEC_JSON_SCHEMA`, used as the structured-output schema). The
same fields with these differences: every view carries all of var, mean,
mean_annual, shift_sd, min, max, vol_mult and why (null when unused; no
`*_annual` bounds); overrides always have the list form
`factor_drift: [{factor, annual, why}]`, `asset_drift: [{asset, annual, why}]`,
`resid_scale`, `r_star` (null when unused), which `_normalize_overrides` turns
into the dict form; every phase has a required `regime` array of `{id (enum of
the 8 ids), share, why}` (`additionalProperties: false`; `[]` when unused);
there is no `reversion` and no `calibration`, so the
translator cannot set either. Numeric bounds are not in the JSON schema
(structured outputs do not support minimum / maximum); validate_spec enforces
them. Before validation `translator.repair()` prefixes the id with `llm_`,
sets `source: "llm"`, reads a prior_prob or confidence between 1.5 and 100 as
a percentage, merges duplicate views of a variable, turns mean_annual on a
non-`f_` variable into mean, keeps only the first of mean / mean_annual /
shift_sd, maps Chinese regime ids to ids (with a note) and drops unknown ones
with the W1 text, and warns about valuation variables and overrides without a
reason.

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
{built_at: "2026-11-02T01:02:07Z", data_start: "1947-01-01", data_end: "2026-09-01",
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
current final spec is the base (10 core factors): the last selection (12
factors, 4 shocks, ridge 0.05, 15-year window) lost on the untouched test
(0.115 vs 0.188 on the same 131 months). model_selection.json is read by
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
   assets: {asset: {cagr_p5, cagr_p50, cagr_p95, real_cagr_p50, max_dd_p50}},
   regimes: {year1: {label: share}, whole: {label: share}, p_contraction_year1, dominant_by_phase: [label]}}],
 regimes: {year1_weighted: {label: share}, p_contraction_year1_weighted,           # probability-weighted
           reference: {year1, p_contraction_year1, n_paths: 2000, seed: 4242},       # runner.reference_simulation
           history: {shares_1973: {label: share}, p_contraction_12m_window}}}
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
 calibration: [{kind, title, text, base, impact, views, overrides, override_notes, param_names, param_notes, formulae,
                engine_mapping, caveats, phase, phase_name, start_month, months,
                engine_check: {equity_wealth_change_assumed, us_equity_price_return_p50, us_equity_total_return_p50,
                               unemployment_change_assumed_pp, unemployment_change_p50_pp,
                               unemployment_change_mean_pp, labor_share_log_change_assumed,
                               labor_share_log_change_p50}}],   # calibrated built-ins only
 regimes: {version: 1, labels: [{id, name, color}],                    # the regime readout (REGIME_READOUT)
           anchor: {month, label_rt, label_expost_mix: {label: share of paths}},
           phases: [{name, start_month, months, requested: {id: share} | null,
                     simulated, simulated_rt, drawn: {label: share}, dominant,
                     paths_with_contraction, contraction_share,
                     reference: {simulated, paths_with_contraction},   # the same months of the reference run
                     history_base_rate}],        # share of 1973-02.. windows of `months` months holding a contraction month
           windows: {year1 | years2_5 | years6_15 | whole: {simulated, simulated_rt, drawn, paths_with_contraction,
                                                            contraction_share, reference: {simulated, paths_with_contraction}}},
           years: [{year: 1..15, simulated, paths_with_contraction}],
           episodes: {per_path, length_mean, share_longer_than_18m, share_severe,
                      reference: {the same four}, history: {episodes, per_15y, length_mean, longer_than_18m, severe}},
           notes: [str]},
 probability: {prior, likelihood, posterior, distance, z: {state}, phase_means: {state}, today, method},
 prob}
```

`regimes` (宏观体制 readout, `regime_views.readout`): `simulated` is the ex-post
label of each simulated month (the paths classified with the panel history
prepended, `regimes.classify_sim`), `simulated_rt` its real-time label, and
`drawn` the ex-post label of the historical month it was drawn from
(`SimOutput.idx`); the two differ because simulated CPI y/y and IP accumulate
from today's levels. Every share dict sums to 1; `anchor.label_rt` is the same
on every path (today: 扩张), `label_expost_mix` the anchor's ex-post label over
the paths (a later simulated fall can make it a contraction month). The
`reference` blocks are the same months of `runner.reference_simulation` (the
same in every scenario of a build), `history_base_rate` the share of
historical windows of the phase's length that hold a contraction month (12
months 24.2%, 36 months 45.2%). Notes, generated from the numbers: N1
requested - drawn > 0.10; N2 drawn - simulated > 0.20 for a requested label;
N3 a name term (滞胀, 通缩, 萧条, 衰退, 通胀 / 过热; the phase name's own, else the
scenario name's for every phase with views; the text says 名称里有 or 情景名称里有)
whose labels hold less than half of that phase's simulated months; N4 always:
the reference's share of months 25-180 with CPI y/y above 4% against history
(41% against 33%). Today N3 fires 6 times (stagflation 工资-物价螺旋 26% and 沃尔克式紧缩
44% 滞胀, japanification 去杠杆与降息 3% and 长期停滞 4% 物价负增长, oil_war 衰退与油价回落
41% and energy_transition 绿色通胀 41%), never for baseline or financial_crisis.
Probability-weighted year 1: simulated 扩张 50.8 / 通胀 32.3 / 物价负增长 0.2 / 衰退
9.0 / 深度衰退 2.3 / 滞胀 5.4 %, P(a contraction month in year 1) 25.7%, against
the reference's 19.2% and history's 24.2%. A detail grows about 10 KB with it.

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

### regimes.json (analysis/regimes.py)

The macro-regime layer's evidence: what each month was, how well the rule
dates recessions, what assets and factors did in each regime, how regimes
move, what the engine does with them, and why they stay out of the
transmission model and the one-year forecast. About 36 s; `python -m
scenariolab.analysis.regimes` writes it too.

```text
{version: 1, data_end: "2026-09", monthdb: {first: "1973-02", last: "2026-09", months: 644},
 definitions: {params, labels: [{id, code, name, term, color, desc}], flags: [{id, name, desc, history_only, rt}],
               precedence, rule_text: {growth, severity, inflation, precedence, realtime, data, timing},
               terms: [{term, labels, text}], names_changed: [{from, to, why}],
               grid: {rows, cols, cells: [[label]], note}, conventions: {id: {name, text}} (ordered),
               interpolations: [{month: "2025-10", series: [cpi, unrate, core], values, how}], swing_pct},
 nowcast: {month, known_from, label, name, label_expost, readings: {…},
           checks: [{id, name, rule, reading, threshold, distance, on}], missing_after: {series: [months]},
           provisional_from, last_high_infl: {month, cpi_yoy}, recent: [24 x {month, label, label_rt, cpi_yoy,
           ip_below_peak_rt, u_rise_rt}], note},
 timeline: {dates (945), label, label_rt, provisional_from, flags: {flag: [0/1]}, series: {cpi_yoy, core_yoy,
            infl_trend_12m, unrate_3m, sahm_gap, ip_yoy, ip_below_peak_rt, ip_drawdown_24m, u_rise_rt},
            runs: [{label, name, start, end, months}], runs_rt, nber: [{peak, trough, first_month}]},
 counts: {"1948" | "1973": {expost | realtime: {months, labels: {id: {months, share, runs}}}, flags: {id: {months, runs}},
                            inflation_falling: {months_with_trend, falling, first_month}}, agreement_rt_expost, confusion_1948: [[6x6]], runs_expost_1948,
          cpi_near_threshold_1973},
 validation: {window, expost | realtime: {balanced_accuracy, tpr, tnr, precision, detected, recessions,
              mean_start_err, mean_end_err, mean_abs_start_err, mean_abs_end_err, false_alarms: [[start, end]],
              per_recession}, months_flagged_after_window, alternatives: [{id, name, kind, ba, …, fa_runs}],
              legs, legs_margin, sensitivity, out_of_sample: {loro, split, grid_points_ba_ge_0955, grid_size, …},
              revisions: {cases, paths: 300, seed}, near_miss_2022_24},
 assets: {windows: {"1973" | "1948": {start, end, months, assets, groups: {gid: {kind, id, name, months, episodes,
            episode_list, thin, contraction_cell, survivors: [asset], assets: {asset: {months,
            ann_real | ann_nominal: {value, ci90}, vs_rest: {value, ci90, rest_value, survives}, vol, sharpe,
            share_positive, worst_12m, worst_12m_real}}, corr_stock_bond, corr_gold_equity}}}},
          inflation_by_era: [{span, start, end, months, episodes, thin, equity_real, bond_real, corr_sb}],
          corr_by_half, corr_by_cpi_bucket},
 episodes: [{label, start, end, months, ip_decline_pct, u_rise_pp, cpi_mean, equity_real_total,
             equity_real_to_low, equity_low_month, equity_real_after_low, months_after_low, bond_real_total,
             worst_window, path: [real equity index, 100 the month before the start]}],
 episodes_summary: {stagflation_after_low_1973: {after, of, episodes}, stagflation_low_before_trough_1973: [lo, hi]},
 factors: {window, reference, factors_meta, states_meta, table_states,
           groups: {gid: {…, factors: {f: {z, ci90, sig, mean_annual}}, states: {s: {mean, z, sig}},
                          distinctive_factors, distinctive_states}}},
 transitions: {"1948" | "1973": {monthly, durations: {id: {mean, median, max, min, thin, …}}, rt_to_expost_12m,
                                 base_rate_12m, p_contraction_12m: {by_label, by_flag, all},
                                 inflation_to_stagflation_hits: {hits, years}},
               walk_forward: {exactly_t12 | any_12m | new_contraction: {all | 1973_1999 | 2000_end}}},
 forward: {lag_months: 1, windows: {"1973" | "1948": {groups: {gid: {name, kind, assets: {asset: {origins,
           episodes, thin, mean (real), uncond, diff, ci90 (of diff), ci90_mean, placebo_p (null if thin), t_hac,
           diff_vs_rest, halves, robust}}}}, global_placebo: {tests, actual, null_mean, null_p95, p}, n_tests,
           eligible_tests, n_survive, n_robust, holm_discoveries, bh10_discoveries, base: {… t+1..t+12 …}}},
           mechanism: {origins, within_12m_of_contraction, mean_within, mean_not, …},
           forecast_test: {available, file, data_end, adopted}},
 eras: [{id, name, start, end, months, dominant, expost, realtime, flags, runs_expost}],
 transmission: {fit_rows, splits: [{id, desc, wald_p: {asset}, n_sig_5pct, n_tests, n_sig_excl_labor_share, …,
                equity_y10}], walk_forward_verdict: {title, detail, oos_r2_delta, n_reject}},
 engine: {support: {id: {name, column, months, runs, episodes, support_hi, max_delivered, shrink, reason, ens,
                        engine_episodes, regime_effective_episodes, thin}},
          reference: {n_paths, seed, months, windows: {year1 | years2_5 | years6_15 | whole | m25_180: {…}},
                      anchor, cpi_gt4_share, cpi_pct, label_changes_per_year, mean_run_months,
                      contraction_episodes},
          history: {shares, cpi_gt4_share, cpi_pct, label_changes_per_year, mean_run_months, contraction_episodes,
                    contraction_base_rate_by_window_months},
          comovement, drawn_vs_sim_crosstab, simulated_inflation_from_expansion_share,
          studies: {n_paths: 600, seed: 20261101, duration_seed: 7,
                    drawn_vs_simulated: [{case, views, months: 12 | 48, drawn_share, simulated_share, lambda, …,
                                          unrate_end_p50}], duration: [{share, months, …}],
                    conflict_example: {spec, phase, regime, lambda_alone, lambda, drawn_share, leave_one_out}}},
 model_use: {transmission | forecast | scenarios: {used, reason, …}; transmission | forecast: tested, forecast: adopted},
 settings: {bootstrap, seed, ci, min_episodes, hac_lags, placebo_min_shift, forward_lag_months, seconds},
 units, notes}
```

Group ids: `label:<id>` (the six labels; real-time labels in `forward`),
`flag:contraction | severe | high_infl | cpi_neg | core_low` (`flag:contraction_rt`,
`flag:severe_rt` in `forward`), `all`. Conventions (`config.REGIMES` and
engine/regimes.py, one block for every consumer): the "1973" window is the
MonthDB months (1973-02..2026-09, 644), the "1948" window 1948-01..today
(945; US equities, Treasuries from 1953-05 and cash only); real returns are
geometric (`expm1(12 x mean(log1p(r) - f_cpi))`); forward windows run t+2..t+13
(a label is known the month after); an episode is a run with at least one month
in the window; fewer than 4 episodes is thin (†, no placebo p); intervals are
90% episode-bootstrap intervals with one generator per cell; HAC 6 / 12 lags;
placebo = circular shifts of >= 24 months. Today (2026-09): real time 扩张,
known from 2026-10, CPI y/y 0.28 pp under the 4% line (last above it 2026-05,
4.17%), no month provisional. NBER: ex post balanced accuracy 0.960 in sample
and 12 / 12 recessions, real time 0.786 with one false alarm (1959-11) and
signals 3.9 months late at the start, 2.9 at the end; leave-one-recession-out
0.909-0.933. The real-time 通胀 tilt of US equities a year later (-6.7 pp) has a
joint placebo p of 0.17 over 130 tests (Holm 0 of 65 eligible), and the
labels have no skill for a new contraction (Brier skill +0.04, t 0.6). The
engine's reference run has CPI y/y above 4% in 41% of months 25-180 (history
33%) and 2.12 contractions per path (history 1.96 per 15 years).
`forward.forecast_test` is `{available, file, data_end, adopted}`: available
only when regime_forecast.json exists and its `data_end` is this build's data
month. That step runs later, so the first build of a new data month reports it
as not available until `--only regimes` runs again after it; `model_use.forecast`
(`tested`, `adopted`) and the last note state the forecast test's verdict only
when it is available (「两者都做过样本外检验，没有通过」), otherwise they say the test
is the opt-in step and this build has no result for its data. Every survives /
robust / significant sentence in the UI is generated from the flags, not typed.

### forecast.json (analysis/forecast.py)

```text
{horizon_months: 12, n_paths: 1000, key_states,
 analog_share (the share of the available starts every origin used), analog_ens (today's effective starts),
 methods: {uncond | analog | naive | scenarios: {name, description}},
 assets, asset_names, risk_assets, shocks: config.EXOGENOUS_SHOCKS, notes,
 selection: {grid: [{share, label ("1/32" ..), ens_first, ens_last, flat_origins,
                     val_crps_analog, val_crps_uncond, val_crps_naive, val_coverage, val_t,
                     test_crps_analog, test_crps_uncond, test_crps_naive, test_coverage, test_rank_ic, test_t}],
             chosen (a share), chosen_label, unit, split: "2004-12", rule, common_random_numbers: true,
             gain_val, gain_test, t_val, t_test, flat_origins, n_origins, claim: {min_skill, min_t},
             baseline_name, text},
 origins: [{date, end, shocks: [name], inflation, analogs: [{date, weight, distance}], bandwidth,
            flat (bandwidth at the cap: analog = uncond), analog_ens, n_starts,
            assets: {asset: {realized,
                             uncond | analog: {p5, p25, p50, p75, p95, mean, prob_loss, pit, crps},
                             naive: {p5, p50, p95, mean, pit, crps}}},
            macro: {unrate | ff | cpi_yoy | spread | oil: {realized, uncond | analog | naive: {p5, p50, p95, mean, pit, crps}}}}],
 scores: {all | ex_shock: {uncond | analog | naive: {assets: {asset: {n, coverage90, pit_mad, crps, mae_p50, bias_p50}},
                                                    risk_avg: {coverage90, pit_mad, crps, mae_p50}, all_avg: {…},
                                                    direction: {hit_rate, base_rate_up, n},
                                                    rank_ic: {mean, t, n, by_origin}},
                           crps_skill_risk: {analog_vs_uncond, analog_vs_naive, uncond_vs_naive},
                           crps_paired_risk: {analog_vs_uncond | analog_vs_naive | uncond_vs_naive: {diff, t, n}}}},
 n_shock_years, misses: [{origin, asset, realized, p5, p50, p95, z, cause: shock | macro | pricing, why}],
 text,
 macro_skill: {rule, split: "2004-12", min_t: 1.0, n_origins: 42, refs: ["uncond", "naive"], skilled: [ids],
               vars: {k: {name, unit, skilled, beats: {uncond, naive}, coverage90: {analog, uncond, naive},
                          all | val | test | ex_shock: {n, crps_analog, crps_uncond, skill, t},
                          naive: {all | val | test | ex_shock: {n, crps_analog, crps_naive, skill, t}}}}},
 live: {origin, state: {key_state}, z_today, analog_ens, bandwidth, analog_share, flat, n_starts, display_ens,
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
 ledger: {dir, n_entries, entries: [{origin, created, matures, status: pending | matured, version, model, analog_ens, analog_share,
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
div_yield, ip_yoy, oil_yoy) resembles today's; naive = a normal with the
training mean and sd of 12-month returns. The analog kernel's effective number
of starts is a share of the starts available at the origin (every start but
the first: 131 at 1984-12, 632 today), chosen from ANALOG_SHARES (1/32 .. 1/2,
so every value is reachable at every origin) by the lowest validation risk
CRPS on origins up to 2004-12; the test origins only check it. Today the
choice is 1/2 (66 effective starts at 1984-12, 310 at 2025-09, 316 live),
the flattest candidate: every share is worse than uncond, less so toward
flat, and `selection.text` says both (generated: 「选中的 1/2 是候选里最平的一档（最接近
无条件抽样），而验证期每一档的 CRPS 都比无条件抽样高（技能 …，t …）」). uncond and analog draw with the same seed at each origin (seed + i;
common random numbers), so they differ only by their weights and a flat
kernel (bandwidth at the cap, FLAT_BANDWIDTH 999; `origins[].flat`) gives the
unconditional forecast bit for bit; `selection.flat_origins` counts those (0
of 42 now). Texts are generated from the numbers: a CRPS gain or loss is
stated only when |skill| > CLAIM_SKILL 0.5% and the paired |t| >= CLAIM_T 2,
and no gain is claimed when half the origins or more are flat (today
「相似起点法与无条件抽样差不多（技能 -0.8%，t = -0.7，在随机误差之内）」). Returns are 12-month
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
both engine methods draw past months' CPI changes without re-anchoring them.
Over the 42 origins the analog CPI y/y forecast beats the unconditional draw
(skill +31.8%, t 5.5) but only ties naive (-0.3%, t 0.0); unemployment and the
policy rate beat naive (+15.2%, +15.6%) but not the unconditional draw, and
the credit spread and oil beat neither. Today
no variable is skilled, so `calibration.available` is false,
`scenario_probs_calibrated` is null and `assets[a].scenarios_calibrated` is
absent; the 「按一年期预测校准」 choice of the portfolio and withdrawal tabs is
then disabled with `calibration.reason`, and 模型后验 stays the default.

Reconciliation (情景概率与一年期预测的对账). The mixture's first year is more
bearish than the scored analog forecast (美股大盘 median +4.6% against +8.3%,
P(loss) 40% against 27%; the analog's 0.265 is written 27% in
the sentence and in the tables alike: the text rounds a half up, as the page does), because the shock scenarios put their shocks in the
first year (7 scenarios with 48% combined probability; their first-year
unemployment rises 0.60 pp on average, the analog's -0.01 pp). `contrib_mean` and `contrib_prob_loss` split the gap exactly by
scenario (probability x the scenario's first-year value minus the analog
value; each sums to its gap). Each built-in is re-run from its detail spec at
its library seed to get the 12-month macro changes of the same paths
(`paths_reproduced` checks them against the npz). The calibration, when a
variable is skilled, is entropy pooling over the 13 scenarios with the model
posterior as the prior and, as views, the analog method's live mean 12-month
change of each skilled variable; an LP finds the largest reachable shrink
lambda (each q >= 0.001 x its prior). `sensitivity` is a what-if per
candidate variable and for all five together (unemployment alone: KL 0.63,
美股大盘 median +7.5%); it is never used.

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
 analog_ens, analog_share?, common_random_numbers?, n_paths, seed,
 taus (q[i] = the (i + 0.5)% quantile, i = 0..99), units,
 methods: {analog | uncond | scenarios | scenarios_calibrated: {name, assets: {asset: {p5, p25, p50, p75, p95,
                                                                                     mean, prob_loss, q[100]}}},
           naive: {name, assets: {asset: {mu, sd}}}},
 macro: {analog | uncond: {k: {p5, p50, p95, mean, q[100]}}}, state,
 scenario_probs, scenario_probs_calibrated}
```

* `analog_share` (the share of the available starts the analog used) and
  `common_random_numbers: true` (uncond and analog drew with the same seed)
  were added after the 2026-09 entry, which keeps its recorded form; readers
  treat both as optional. Method names are the forecast module's display names
  (`METHOD_CN`: 相似起点法, 无条件抽样, 历史均值基准, 情景加权, 情景加权（校准）).
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
  own. The entry for origin 2026-09 was recorded once more on 2026-11-01
  (created 2026-11-01T23:04:51Z), before it could mature: the first recording
  carried a calibrated method set that was later withdrawn and the analog
  setting the feasible share grid replaced (ENS 576), so it was removed by hand
  and the corrected forecast went in through the module's own append path
  (methods uncond, analog, scenarios, naive; analog_ens 316; no calibrated
  probabilities). Its uncond, scenarios and naive bands equal the first
  recording's; its bands equal today's live forecast exactly.

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

### regime_forecast.json (analysis/regime_forecast.py; opt-in)

Does knowing today's real-time regime improve the one-year forecast out of
sample? The pre-registered walk-forward of the regime study, against the
current forecast. About 100 s (`--only regime_forecast`, or `python -m
scenariolab.analysis.regime_forecast`).

```text
{version, data_end: "2026-09", question, text,
 settings: {n_paths: 1000, seed_bases: [23, 1023, 2023, 3023, 4023], origins (42), split: "2004-12",
            share_grid, share_labels, lambdas: [0, 0.25, 0.5], guard: {min_ens: 30, min_episodes: 6},
            common_random_numbers: true, claim: {min_skill: 0.005, min_t: 2.0}},
 baseline: {name, share, share_label, flat_origins, val_crps_by_share,
            forecast_json: {same_share, max_abs_diff_us_equity_p50}, risk_crps_seed23, coverage90_seed23,
            uncond_risk_crps_seed23, aa_noise: {seed: {all | val | test: {diff, t}}}, aa_note},
 families: [{id: a1 | a2 | b | bc | c | cc, name, selected_by_seed, selected_text,
             risk_crps_rel | risk_crps_diff: {val | test | all | ex_shock: {mean, min, max}}, t: {val, test},
             rank_ic_delta, coverage90, coverage90_current, macro: {cpi_yoy | unrate | ff: {rel_vs_current,
             skill_vs_naive}}, fragility_seed23: {test, test_rel, best_origins, best_contrib, test_without_best,
             test_without_best_rel}, better_both_halves_seeds, checks, passes}],
 oracle: {c_oracle | cc_oracle: {name, eligible: false, risk_crps_rel, better_both_halves_seeds}},
 guard_tests: {truncation, scrambled_future, power, code_guards},
 by_label_seed23, labels_at_origins, verdict: {adopted, rule, text}, seconds}
```

Every origin i draws every method with seed base + i (common random numbers
with the current forecast, which uses the forecast module's own selection:
the share 1/2 today; `baseline.forecast_json` checks that it reproduces
forecast.json exactly). Families: a1 / a2 add real-time regime indicators to
the analog state; b / bc weight the analog kernel by 1 for starts with the
origin's real-time label (6 labels or 3 groups) and lambda otherwise; c / cc
draw only from such starts; a thin pool shrinks toward the plain kernel or
the prior. Adoption: lower risk-asset CRPS than the current forecast in both
halves in every seed set, the seed-averaged rank IC down by at most 0.05 and
90% coverage by at most 0.02. The look-ahead guards: `check_frame` refuses a
frame past the origin or with ex-post / NBER columns; truncation (150 months
checked) and a scrambled future (42 origins: weights and forecasts unmoved)
pass, and the power check shows the ex-post labels and the oracle do move.
Which forecast the file tested: `data_end`, `settings.origins[-1]` and
`baseline.share`. The 宏观体制 tab compares them with regimes.json `data_end`
and forecast.json (`origins[-1].date`, `analog_share`); when one differs (the
step is opt-in, so a default build that refreshes the forecast leaves it
behind) it marks the result 旧结果, names the current forecast and the
command, and the forward-return section no longer says the test failed. A
test-half gain in `text` (and on the tab) carries its seed-averaged t and,
unless it passes the forecast module's claim rule (`settings.claim`: beyond
0.5% with |t| >= 2), 「在随机误差之内」 (`gain_note`).
Today: adopted none; no family is better in both halves in any seed set
(validation / test relative CRPS, positive = worse: a1 +0.47% / +0.21%, a2
+0.60% / +0.05%, b +0.57% / -0.22%, bc +0.66% / -0.28%, c +1.77% / -0.96%, cc
+4.08% / -2.83%, the cc test gain resting on 2007-12 and 2021-12); the
oracle c_oracle (ex-post labels of the whole panel) is better in both halves
in 5 of 5 seed sets, which is what a look-ahead would be worth. Real-time
labels at the 42 origins: 扩张 32, 通胀 8, 衰退 2.

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
            forecast_ic_tol, forecast_paths, forecast_split, analog_share, n_origins, residuals,
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
 regime_support: {id: {name, column, months, runs, episodes, max_delivered, shrink, …}},   # the 8 ids + cpi_neg
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
background thread fits the EngineContext, runs the engine's reference
simulation (`runner.reference_simulation`, about 1 s; every custom run's
regime readout compares with it, and a request that arrives meanwhile waits
for this copy) and loads the scenario paths at startup so the first requests
are fast; a failure of one part only makes its first request slower.

| Method | Path | Body -> result |
|---|---|---|
| GET | `/`, `/js/…`, `/css/…`, `/vendor/…` | the static UI from web/ (`Cache-Control: no-cache` on `/`, `/index.html`, `/js/*`, `/css/*`, so edited tab modules are revalidated; `/vendor/echarts/*` gets ETag and Last-Modified, a conditional GET a 304, and no Cache-Control; a directory such as `/vendor/` is a 404, no listing) |
| GET | `/api/meta` | meta.json plus `results`: the result names (of the list below) whose file exists at request time, so pages skip a missing optional file instead of requesting it (`api.optional`) |
| GET | `/api/{name}` | results/{name}.json for name in eras, causality, backtest, diagnostics, scenarios, regimes, forecast, ablation, regime_forecast, portfolio_default, llm_translator, valuation |
| GET | `/api/scenarios/{id}` | results/scenarios/{id}.json, else results/custom/{id}.json (built-ins win) |
| GET | `/api/scenarios/custom` | `{scenarios: [{id, name, summary, source, created, prior_prob, prob}]}`, newest first |
| POST | `/api/scenarios/custom/run` | `{spec, save = true, n_paths = 600 (clamped 100..2000), seed = 7, source = "custom"}` -> the detail JSON (with `regimes`; W8-W11 in `validation.warnings`; a phase's `regime` key in the dict or list form) + `created`; with `save`, also custom/{id}.json + .npz |
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
| 404 | a result file is missing | `<file> 还没有生成：请在项目根目录运行 python -m scenariolab.build（或只补这一步：python -m scenariolab.build --only <step>）`; ablation.json names `--only ablation（引擎消融约需 5 分钟，默认构建不包含这一步）`, regime_forecast.json `--only regime_forecast（宏观体制的一年期预测检验约需 2 分钟，默认构建不包含这一步）` |
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
  value, the eras, the drift bounds, the section 「## 宏观体制（regime 字段）」
  (between 历史时期 and 覆盖项; `regime_views.TRANSLATOR_RULE` via
  `translator.regime_rule`), and two built-in specs in the LLM form
  (oil_war, k_shaped; calibrations resolved). The regime section states the
  classifier's rule from config.REGIMES, what a 12-month phase can deliver per
  regime from the MonthDB support table (e.g. 衰退（recession）37%，1 段（2001）),
  and ten usage rules (prefer contraction, high_infl and continuous views;
  stagflation <= 0.8 and <= 18 months; 萧条 = depression <= about 0.76, one
  episode 2008-09; 通缩 = f_cpi near its lower support with contraction 0 and an
  f_unrate view, never the deflation label; share x months <= 18 for the
  contraction types; …). Its study numbers (12-month stagflation 0.8 is 60%
  simulated 滞胀, the f_cpi recipe ends with median unemployment 2.3%, the
  conflict example's lambda 0.65) are read from regimes.json `engine.studies`
  only when its `data_end` equals the anchor month (rows matched on views and
  months; the parsed file is cached by mtime); otherwise those clauses keep
  their rule without the number, so the prompt never states a number the
  build did not compute. The variable table does not list rg_ columns: the
  `regime` key is the only way the model targets regimes. The examples' narratives,
  summaries and causal chains contain engine numbers filled in after the run;
  the prompt says to restate only the model's own views and overrides and to
  keep the narrative to 2-4 paragraphs. The residual rule (item 4 and the
  resid_scale line: half of the residual follows the drawn months) is
  `residual_rule()`, generated from config.RESIDUALS; the prompt is 22,613
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
  prior) / (target - prior)), which appends the regime checks W8-W11 to the
  validation warnings.
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

* **Router** (app.js). `TABS` lists 13 tabs in four groups: 历史与因果
  (overview 总览, eras 历史时代画像, regimes 宏观体制, causality 因果传导), 验证 (forecast
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
  factors | scenarios`, `#regimes?today | words | timeline | validation |
  assets | episodes | factors | transitions | forward | eras | engine |
  forecast | conventions`, `#eras?today`, `#diagnostics?ablation`. Pages that
  rewrite their own deep link use `ui.setRoute(tab, param, alive)`, which
  replaces the address without a re-render and only while that page is still
  current.
* **api.js**: `get(path, {fresh})` caches GET promises per page load (a failed
  one is evicted), `optional(name, meta)` (null without a request when
  `meta.results` says the file is missing, else `get` with errors as null),
  `post`, `del`, `invalidate(prefix)`; an error carries
  `status` and the server's `detail`; strings in every response lose a
  rounded negative zero.
* **ui.js**: `h()` (element builder), `esc`, `card`, `chartBox`, `wideChart`
  (a dense chart scrolls inside its card, never the page), `kpi`, `table`
  (`stack` turns rows into labelled blocks on phones), `watchTables(root)` /
  `fitTable(wrap)` (app.js watches the page once: a table wider than its card
  first wraps its cells, `.fit-wrap`, then stacks its rows into labelled
  blocks, `.fit-stack`, and each step is undone once the card is wide enough
  again; a table with merged cells keeps scrolling), `seg`, `select`,
  `setRoute`, `buildHint`, `loading`, `notice`, `chip`; formats `fmtPct`,
  `fmtPP`, `fmtNum`, `fmtMoney` (never "-0.0"); `signCls(x)` -> `pos` / `neg`
  by the sign of a number, `goodCls(ok)` -> `is-good` / `is-bad` for a
  judgement; META lookups filled from /api/meta (`assetName`, `factorName`,
  `stateName`, `eraName`, `varName`).
* **charts.js**: `mount` / `unmount` / `disposeWithin(root)` (app.js disposes
  every chart and observer of the previous page), `line` (`xValue` puts numeric
  x on a value axis, spaced by value), `fan`, `candles`, `heatmap` (cell labels
  white on dark cells and ink on light ones, whichever contrasts more), `bars`
  (`valueMax` fixes the value axis, 1 for stacked shares), `dag` / `dagLayout`, `scatter`, `assetColor`, `palette`,
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
* **Tabs and the files they read**: overview (eras, forecast, scenarios,
  optional regimes), eras (eras, causality, meta, optional regimes), regimes
  (regimes; optional regime_forecast, forecast for the current baseline's name
  and to check that regime_forecast.json tested it, and scenarios for the
  library mix),
  causality (causality), forecast (forecast, scenarios), backtest (backtest, POST backtest/run), diagnostics
  (diagnostics, optional ablation), valuation (valuation, optional
  scenarios), scenarios (scenarios, scenarios/{id}, scenarios/custom), lab
  (llm/status, llm_translator, scenarios, scenarios/{id}, scenarios/custom,
  POST llm/translate, POST scenarios/custom/run, DELETE
  scenarios/custom/{id}), portfolio and withdrawal (portfolio_default,
  scenarios, scenarios/custom, forecast for the calibrated probabilities, POST
  portfolio/evaluate or withdrawal/evaluate), method (meta, optional
  valuation and regimes); optional files go through `api.optional`, so a
  missing one is not requested (and the regimes tab shows its build hint
  without requesting regimes.json when `meta.results` lacks it). `_scenario_detail.js` renders one scenario detail for the
  scenarios and lab tabs (its P/D card fetches valuation once) and exports
  the regime helpers the scenarios, eras and lab tabs share: `REGIME_LABELS`,
  `regimeLabels` / `regimeName` / `regimeColor` / `regimeVarName`, `regimeRequest`
  (a phase's `regime` key in either form as {id: share}), `regimeMix` (chips of
  the shares that show as 1% or more), `roundShares` (whole percents adding up to
  100, largest remainder), `regimeBars` (100% stacked bars with a wrapping legend
  and row labels that wrap on a phone) and `fitTable`; `_weights.js`
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
* **宏观体制** (`regimes.js`, after 历史时代画像). Sections A-M: 今天处在什么体制
  (the real-time label with 「单边、修订后数据」, the readings and the signed
  distance to every threshold, the last 36 months, the provisional and
  interpolation notes); 你问的四个词怎么对应到数据 (the 3 x 3 grid of
  `definitions.grid`, the user's four words 通胀 / 通缩 / 滞胀 / 萧条 mapped to the
  labels, 通缩 -> 物价负增长 and 萧条 -> 深度衰退 explicitly, and the six labels'
  counts); the 1948+ timeline (ex post or real time, NBER bands for comparison
  only); the NBER validation (in and out of sample, alternatives, revisions,
  2024 as 「接近触发」); assets by regime (a K-line-coloured heatmap, contraction
  cells outlined as 包含先跌后涨); 收缩期的逐段路径 (every contraction as a real
  equity path from the month before its start: the fall to the equity low and
  the rebound, never 「滞胀对股票不坏」); factors (fingerprint palette); transitions
  and the walk-forward 「对新的收缩没有预测力」; forward returns with the
  publication lag, the recession-onset mechanism and the joint placebo next
  to every number; eras; 引擎里的体制 (the support table, history vs engine
  mixes, what a request delivers, co-movement); 一年期预测里用不用体制 (the
  regime_forecast families and verdict, or the opt-in command when the file
  is missing; the current baseline's name from forecast.json, and a 旧结果 notice
  naming the command when the file tested another forecast: data month, last origin or analog share); 口径说明 (`definitions.conventions`).
  Every number in a sentence is filled from the JSON; † marks thin cells with
  the episode list in its tooltip; regime categories use the
  `definitions.labels[].color` palette, never K-line red / green; tables stack
  on phones. The eras tab adds a 这一时代的宏观体制 card (ex post / real time
  bars and the runs) and 各时代的宏观体制构成 when regimes.json exists (omitted
  without an error otherwise). The scenario detail (`_scenario_detail.js`)
  adds 宏观体制：模拟路径处在什么状态 after 分阶段视图 (year-1 contraction
  probability against the reference and history, the anchor's real-time
  label, the 15 yearly mixes, the per-phase table 要求 / 抽中月份 / 模拟路径 /
  出现收缩的路径 / 引擎无条件 / 历史基准, and the notes N1-N4); a detail without
  `regimes` (an older build) says how to get it. The scenario list adds
  各情景第 1 年的宏观体制 from the index fields; the lab's phase table shows a 体制
  column when a phase sets `regime`; the overview adds one line linking to
  `#regimes?today`; the method page adds step 11 宏观体制 and a limitation line.
* **一年期预测**: the selection table by share of the available starts (with
  each share's flat origins and paired t; the KPI is coloured only when the
  claim rule passes, and every selection sentence comes from
  `selection.text`); the tables wrap or stack instead of scrolling at 900 and
  390 px; sections 情景概率与一年期预测的对账 (the gap split by
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

`.venv/bin/python -m pytest -q` runs the whole suite: 412 tests in 18 files
at the time of writing, about 3 minutes (412 passed in 167 s after the full
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
delete it) or into a temporary directory. A fresh clone (no data/) runs 253
passed / 159 skipped with the llm extra and 248 passed / 140 skipped without it
(about 30 s). The exact check-value tests of the regime layer skip unless
panel_meta.json ends in 2026-09 (`CHECK_MONTH` in tests/test_regimes.py), so a
data update skips them instead of failing.

Files: test_ablation, test_attribution (bit-identity of simulate and run_spec
with and without the attribution, additivity, the reconciliation, a pure oil
shock, the structural assets, the lagged-spec fallback, the valuation pull, and
that the factor terms use the simulated, not the drawn, factor changes),
test_backtest (incl. the custom windows: validation, no look-ahead, the setup
cache), test_build (incl. the STEPS order and OPT_IN), test_causality,
test_diagnostics, test_engine, test_eras (incl. era_attribution on synthetic
and real data), test_forecast (incl. the ledger: append-only, refusals,
duplicates, scoring on stored quantiles; the skill rule and the calibration;
a flat kernel equals uncond bit for bit, every share reachable, the claim
rule of every generated text), test_portfolio (incl. the continuation: the
first 15 years bit-identical, one path per row, stale and old-format files
refused), test_regime_forecast (the look-ahead guards: check_frame refusals,
truncation equality, a scrambled future moves no weight at any origin; the
adoption rule on synthetic score tables; a test-half gain stated with its t
and the claim rule), test_regimes (46 tests: synthetic
cases, real time = the ex-post rule on data through t, the own-month unknown
mask, the labels of 945 months, nowcast with trailing gaps, an empty panel,
provisional_from, the rg_ columns and the MonthDB as-of window, bit-identity
with and without rg_ columns and REGIME_READOUT, NBER scores, the
conventions, the check values, REGIMES high_inflation_pct = the transmission
threshold), test_regimes_analysis (the regimes.json contract and its check
values; the forecast test counts only for the same data month), test_scenarios
(incl. the regime key, W1-W11 and the pools a second check reuses, the readout
and the index fields), test_server (incl. /api/regimes, the regime_forecast 404
hint and `results` in /api/meta), test_translator (incl. the regime schema, repair and the prompt
section), test_valuation, test_withdrawal.

CI (.github/workflows/ci.yml) runs on every push and pull request, on Python
3.10 and 3.12: `pip install -e ".[dev]"` (no llm extra), `python -m
compileall -q scenariolab`, `ruff check scenariolab tests` (pyproject
`[tool.ruff.lint] select = ["E9", "F"]`; the dev extra pins
`ruff>=0.16,<0.17`) and `pytest -q`. It never downloads data or runs the
build: on a fresh clone the data-dependent tests skip.
