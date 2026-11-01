# Architecture and module contracts

This file is the contract every module codes against. If you change a shape
here, change every producer and consumer of it.

## Layout

```
scenariolab/
  config.py            paths, ERAS, ASSETS, FACTORS, STATES (shared ids)
  data/sources.py      cached downloaders (FRED, Ken French, Nareit, gold)
  data/panel.py        builds data/processed/panel_monthly.csv
  engine/ep.py         entropy pooling: apply_views(frame, views) -> EPResult
  engine/sampler.py    weighted block bootstrap
  engine/factor_model.py  ModelSpec + AssetFactorModel (linear / lags / regime / MLP)
  engine/simulate.py   MonthDB, Anchor, capital_market_assumptions, simulate()
  engine/summary.py    fans, candles, correlations, stats
  context.py           EngineContext: panel + selected model + db + anchor + cma
  scenarios/schema.py  scenario spec schema + validate_spec()
  scenarios/runner.py  run_spec(ctx, spec) -> (detail json, SimOutput)
  scenarios/specs/*.yaml   the built-in scenarios (same schema the LLM emits)
  scenarios/library.py     build all built-ins -> results/scenarios*.json + npz
  scenarios/probabilities.py  prior x likelihood(today) -> posterior probs
  scenarios/calibrate.py      income-distribution -> consumption calculator
  analysis/eras.py         -> results/eras.json
  analysis/causality.py    -> results/causality.json + processed/structural_shocks.csv
  analysis/backtest.py     -> results/backtest.json
  analysis/diagnostics.py  -> results/diagnostics.json (+ model_selection.json)
  analysis/walkforward.py  the diagnose -> fix -> holdout loop used by diagnostics
  portfolio/metrics.py     portfolio stats on scenario paths
  portfolio/withdrawal.py  withdrawal strategies + ruin
  portfolio/service.py     evaluate_portfolio(), evaluate_withdrawal() for the API
  llm/translator.py        narrative -> scenario spec via Claude (structured output)
  build.py             CLI: python -m scenariolab.build [--refresh] [--paths N] [--skip ...]
  server.py            FastAPI app: python -m scenariolab.server
web/                   static single-page UI (ECharts via CDN, no build step)
data/                  raw/, processed/, results/  (git-ignored)
```

## Conventions

* Monthly data, index = month start (`pd.Timestamp("2026-09-01")`). JSON dates
  are `"YYYY-MM"` strings for monthly series, `"YYYY"` for yearly.
* Returns are decimal simple returns (0.01 = 1%). Rates, yields and spreads are
  in percent (4.5 = 4.5%). JSON numbers are rounded to <= 6 decimals.
* Panel column families: `s_<state>`, `f_<factor>`, `r_<asset>` (total),
  `p_<asset>` (price), `d_<asset>` (income), `dy_<asset>` (trailing 12m income
  yield, %), `lvl_<name>` (raw levels), `e_<shock>` (structural shocks, merged
  from processed/structural_shocks.csv by EngineContext when present).
* Models are estimated from `config.MODEL_START` (1973-01).
* All JSON written with `ensure_ascii=False`. NaN/inf must become `null`
  (use `scenariolab.jsonio.dump`).
* UI text is Simplified Chinese. Code, ids and comments are English.
* Every analysis module exposes `run(ctx: EngineContext) -> dict` that returns
  the JSON-ready dict; `build.py` writes it. Modules must not write outside
  `data/`.

## EngineContext (context.py)

```python
ctx = EngineContext.load(selection=True)   # reads results/model_selection.json if present
ctx.panel      # DataFrame, panel + e_* shock columns if available
ctx.meta       # panel meta dict
ctx.spec       # ModelSpec in use
ctx.model      # AssetFactorModel fitted on MODEL_START..end
ctx.db         # MonthDB over MODEL_START..end
ctx.anchor     # Anchor at the last month
ctx.cma        # capital_market_assumptions(db, anchor)  (valuation_adjusted)
ctx.fit(spec, start, end) -> AssetFactorModel   # helper for backtests
```

## Scenario spec (scenarios/schema.py) — also the LLM output schema

```yaml
id: oil_war                      # [a-z0-9_]+
name: 中东战争与石油供给冲击
summary: 一句话概括
narrative: 多段说明（为什么、怎么传导）
tags: [oil, war, stagflation]
prior_prob: 0.07                 # subjective prior, 0..1 (renormalized across set)
analogs: ["1973-10:1975-03", "1990-08:1991-03"]   # historical analog windows
causal_chain:                    # explanation graph shown in the UI
  - {from: 霍尔木兹海峡受阻, to: 原油供给下降, sign: "+", lag: "0-1月", mechanism: "..."}
phases:
  - name: 冲击期
    months: 6
    views:                       # entropy-pooling views on MonthDB columns
      - {var: f_oil, mean_annual: 0.9, why: "油价 6 个月涨约 45%"}
      - {var: e_oil_supply, shift_sd: 1.5}
      - {var: s_cpi_yoy, mean: 6.0}
    overrides:                   # optional, all bounded by validate_spec
      factor_drift: {ip: -0.02}  # annual change added to a factor (beyond history)
      asset_drift: {nasdaq: 0.03}  # annual price drift added (beyond history)
      resid_scale: 1.2           # 0.5..2.0
      r_star: 0.5                # neutral real policy rate, -3..3
    confidence: 1.0              # 0.3..1.0 (blend posterior toward prior)
reversion: {}                    # optional overrides of simulate.DEFAULT_REVERSION
```

View keys: `mean`, `mean_annual`, `shift_sd`, `min`, `max`, `min_annual`,
`max_annual`, `vol_mult` (see engine/ep.apply_views). `var` must be a MonthDB
frame column; allowed prefixes `s_`, `f_`, `e_`. Phase months must sum to the
horizon (180); `validate_spec` pads with a view-free phase or trims the last
phase, clips out-of-range values and returns a report:

```json
{"ok": true, "errors": [], "warnings": ["phase 2: s_cpi_yoy 14.0 clipped to 11.3 (98th pct)"],
 "clipped": [{"phase": 1, "var": "s_cpi_yoy", "requested": 14.0, "used": 11.3}]}
```

## Results files

### results/meta.json (build.py)
`{built_at, data_start, data_end, model_start, assets[], factors[], states[], eras[],
today: {date, states: {id: value}, income_yield: {asset: value}}, model: AssetFactorModel.summary(),
cma: {...}, notes[], sources{}, llm: {available: bool, model: str}}`

### results/eras.json (analysis/eras.py)
```json
{
 "eras": [{"id","name","start","end","summary","n_months",
   "states": {"cpi_yoy": {"mean": 7.9, "start": 3.6, "end": 3.8}},
   "factors": {"oil": {"mean_annual": 0.12, "vol_annual": 0.35}},
   "assets": {"us_equity": {"cagr": 0.07, "real_cagr": -0.01, "vol": 0.17, "max_dd": -0.45, "avg_income": 0.04}},
   "corr": {"assets": ["us_equity", ...], "matrix": [[1, 0.3], [0.3, 1]]},
   "stock_bond_corr": 0.35, "oil_cpi_passthrough": 0.04, "phillips_slope": -0.2,
   "signature": [{"state": "cpi_yoy", "z": 2.1, "text": "通胀显著高于长期均值"}],
   "narrative": "自动生成的中文描述"}],
 "fingerprint": {"rows": ["golden_age", ...], "cols": ["cpi_yoy", ...], "z": [[...]], "values": [[...]]},
 "today_similarity": {"date": "2026-09", "weights": {"great_inflation": 0.31, ...}, "ens": 120.5,
   "views": [{"var": "s_cpi_yoy", "today": 3.7}], "residual": [{"state": "...", "today": 1, "matched": 2, "gap_z": 0.4}]},
 "timeline": {"dates": ["1955-01", ...], "series": {"cpi_yoy": [...], ...}}
}
```

### results/causality.json (analysis/causality.py)
```json
{
 "svar": {"variables": [{"id","name","unit"}], "shocks": [{"id","name","restrictions"}],
   "horizon": 36, "periods": [{"id","name","start","end"}],
   "irf": {"<period>": {"<shock>": {"<var>": {"p16": [...], "p50": [...], "p84": [...]}}}},
   "n_accepted": {"<period>": 500}},
 "historical_decomposition": {"dates": [...], "variable": "cpi", "contrib": {"<shock>": [...]}},
 "distribution": {"shock": "劳动收入占比下降 1%", "horizon_q": 12, "periods": [...],
   "responses": {"<period>": {"<var>": {"coef": [...], "lo": [...], "hi": [...]}}}},
 "asset_betas": {"periods": [...], "factors": [...], "assets": [...],
   "beta_std": {"<period>": [[...]]}, "r2": {"<period>": {"<asset>": 0.2}}},
 "chains": [{"id","title","nodes": [{"id","label","group"}], "edges": [{"from","to","sign","label","evidence"}]}]
}
```
Also writes `data/processed/structural_shocks.csv` with columns
`date,e_oil_supply,e_demand,e_monetary,e_financial` (unit variance, monthly).

### results/backtest.json, results/diagnostics.json, results/model_selection.json
See analysis module docstrings; shapes:
```json
backtest: {"assets": [...],
  "transmission": {"dates": [...], "series": {"<asset>": {"actual": [...], "fitted": [...], "fitted_oos": [...]}},
                   "r2": {...}, "r2_oos": {...}},
  "windows": [{"id","name","train": ["1973-01","1999-12"],"test": ["2000-01","2009-12"],"phase_months": 24,
     "assets": {"<asset>": {"dates": [...], "actual": [...], "oracle": [...],
        "fan_cond": {"p5": [], "p25": [], "p50": [], "p75": [], "p95": []}, "fan_uncond": {...},
        "coverage_cond": 0.8, "coverage_uncond": 0.7, "pit_terminal_cond": 0.4, "pit_terminal_uncond": 0.6,
        "cagr_actual": 0.01, "cagr_pred_cond_p50": 0.02, "cagr_pred_uncond_p50": 0.08}},
     "corr": {"assets": [...], "realized": [[...]], "posterior": [[...]], "prior": [[...]], "mae_posterior": 0.1, "mae_prior": 0.2},
     "summary": {...}}]}
diagnostics: {"candidates": [...], "iterations": [{"iter","train","validate","test","baseline","findings","trials","selected","test_result"}],
  "final": {"spec": {...}, "trained_on": [...]}, "residual_analysis": {...}, "recommendations": [...]}
model_selection: {"spec": {ModelSpec dict}, "source": "...", "val_score": 0.2, "test_score": 0.18}
```

### results/scenarios.json + results/scenarios/<id>.json + results/scenario_paths.npz
Index:
```json
{"start": "2026-09", "horizon_months": 180, "n_paths": 600, "anchor": {...}, "cma": {...},
 "probability_method": "...",
 "scenarios": [{"id","name","summary","tags","source": "builtin","prior_prob","likelihood","prob",
   "headline": {"us_equity_cagr_p50": 0.05, "us_equity_real_cagr_p50": 0.02, "cpi_avg": 3.1, "ff_end_p50": 3.5},
   "ens_min": 210.0}]}
```
Detail (`runner.run_spec` output):
```json
{"id","name","summary","narrative","tags","source","spec": {...},"validation": {...},
 "causal_chain": {"nodes": [{"id","label"}], "edges": [{"from","to","sign","lag","mechanism"}]},
 "phases": [{"name","start_month","months","views": [...],
    "ep": {"ens","ens_ratio","kl","converged","era_weights": {...},"top_months": [{"date","weight_ratio"}],"views": [...]}}],
 "labels": ["2026-09", "2026-10", ...],          // T+1 monthly labels
 "years": ["2026", "2027", ...],
 "assets": {"<asset>": {"fan": {"p5": [...], ...}, "fan_real": {...},
    "candles_monthly": [["2026-10", open, close, low, high], ...],
    "candles_yearly": [["2027", open, close, low, high], ...],
    "stats": {"cagr_p5","cagr_p50","cagr_p95","real_cagr_p50","vol","max_dd_p50"}}},
 "macro": {"cpi_yoy": {"fan": {...}}, "ff": {...}, "y10": {...}, "unrate": {...}, "oil": {...}, "spread": {...}},
 "corr": {"assets": [...], "matrix": [[...]]}, "corr_prior": {...},
 "era_composition": {"great_inflation": 0.4, ...},
 "representative_path": 17}
```
npz arrays: `total` (S,N,T,A) float32, `income` (S,N,T,A) float32, `cpi` (S,N,T+1)
float32, `ids` (S,), `assets` (A,), `labels` (T+1,). Custom (LLM/user) scenarios:
`results/custom/<id>.json` + `results/custom/<id>.npz` (same arrays with S=1).

## Portfolio and withdrawal (portfolio/service.py)

```python
evaluate_portfolio(weights: dict, scenario_probs: dict | None = None,
                   include_custom: list[str] = (), rebalance: str = "monthly") -> dict
evaluate_withdrawal(weights: dict, initial: float = 1e6, rate: float = 0.04,
                    strategy: str = "all", inflation_indexed: bool = True,
                    scenario_probs: dict | None = None, include_custom: list[str] = (),
                    horizon_years: int = 15) -> dict
```

Withdrawal strategies (annual, no other rebalancing):
* `pro_rata` (按比例提取): income is reinvested in the asset that paid it; at
  each year end the withdrawal is raised by selling every holding in
  proportion to its current value.
* `div_then_pro_rata` (先用股息/利息，不足按比例): income is paid out to a cash
  pot during the year; at year end the pot funds the withdrawal; any surplus
  is reinvested pro-rata; any shortfall is sold pro-rata.
* `div_then_winners` (先用股息，不足按当年收益从高到低卖): like the previous one,
  but the shortfall is raised by selling holdings in descending order of that
  year's total return (sell the best performer first, exhaust it, move on).
* The withdrawal amount is `rate * initial`, indexed to the path's own CPI when
  `inflation_indexed`. Ruin = holdings cannot cover a year's withdrawal.

Response shapes:
```json
portfolio: {"years": [...], "per_scenario": [{"id","name","prob","cagr_p5","cagr_p50","cagr_p95","real_cagr_p50",
   "vol","sharpe","sortino","max_dd_p50","max_dd_p95","cvar5_terminal","prob_loss_real","fan": {...},"fan_real": {...}}],
   "aggregate": {"weighted": {...}, "worst": {...}, "best": {...}, "equal_weight": {...}}}
withdrawal: {"years": [...], "strategies": {"<strategy>": {"per_scenario": [{"id","name","prob","ruin_prob",
   "survival": [...], "terminal_real_p5","terminal_real_p50","terminal_real_p95","terminal_nominal_p50",
   "depletion_year_median","wealth_fan_real": {...}}],
   "aggregate": {"ruin_prob_weighted","ruin_prob_worst","ruin_prob_best","survival_weighted": [...],"terminal_real_p50_weighted"}}},
 "safe_rate": {"<strategy>": {"weighted": 0.041, "per_scenario": {"<id>": 0.05}}}}
```

## HTTP API (server.py)

| Method | Path | Body / result |
|---|---|---|
| GET | `/` | web/index.html |
| GET | `/api/meta` `/api/eras` `/api/causality` `/api/backtest` `/api/diagnostics` `/api/scenarios` | results JSON |
| GET | `/api/scenarios/{id}` | built-in or custom detail JSON |
| POST | `/api/portfolio/evaluate` | evaluate_portfolio body -> result |
| POST | `/api/withdrawal/evaluate` | evaluate_withdrawal body -> result |
| GET | `/api/llm/status` | `{available, model, reason}` |
| POST | `/api/llm/translate` | `{narrative, horizon_months?}` -> `{spec, validation, model, usage}`; 503 without key |
| POST | `/api/scenarios/custom/run` | `{spec, save?}` -> detail JSON (engine run, no LLM) |
| GET | `/api/scenarios/custom` | list of saved custom scenarios |
| DELETE | `/api/scenarios/custom/{id}` | remove one |

## LLM translator rule

The LLM never produces prices, returns or K-lines. It only turns a narrative
into the scenario spec above (shocks, views, phases, bounded overrides, causal
chain). `validate_spec` clips it to the historical support, and the engine
produces every number. The UI must show the validation report next to results.
