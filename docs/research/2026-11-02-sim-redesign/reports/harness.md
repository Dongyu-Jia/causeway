# Oracle-macro evaluation harness (research round, role `harness`)

*What the role did:* the evaluation harness every later role used (ported into the app as scenariolab/evaluation/harness.py).

Rendered from the role's final report in the research workflow's journal (run `wf_cc7fe8c3-afc`, result `harness`). The text is unchanged except that local absolute paths are written as the spec writes them (`scratchpad/sim_redesign/...`, `scratchpad/...`). It is the report the critic and the designer read; the numbers are the role's own, and the critique ([`critic/critique.md`](../critic/critique.md)) and the spec ([`SIM_DESIGN.md`](../SIM_DESIGN.md)) say which of them survived.

## Summary

I built the oracle-macro evaluation harness, checked it against every number I could reproduce, and ran the baselines B0 and F on both protocols plus the variance anatomy. I wrote nothing in MAIN, RWT or FWT: all scripts ran with -B, and the only MAIN file newer than this round's start is MAIN/.env, changed at 16:25:16 by another process (the harness never reads it).

HARNESS API. Module: &lt;H>/simharness.py, where H = scratchpad/sim_redesign/harness
- How to run: cd MAIN && MAIN/.venv/bin/python -B script.py (-I also works). In the script: sys.path.insert(0, H); import simharness as sh, before importing anything from scenariolab. The import sets dont_write_bytecode and puts MAIN on sys.path.
- load_panel(fixed=True, valuation=True, extra=False) -> DataFrame. This is MAIN's load_panel_with_shocks plus extra columns; MAIN's own columns are never overwritten.
  - fixed adds the factor round's columns: f_spread_fixed, f_gz, f_tp10_x_core4, f_metals_oil, f_tp10, f_ebp and f_labor_share_rt4. They are read from scratchpad/factors/&lt;family>/candidates.csv plus the release-dated PRS85006173. I re-implemented d02/d05's load() and hold_quarterly (= scenariolab release_dated) rather than importing the factor harness.
  - valuation adds v_* log levels: per asset v_lp_/v_ld12_/v_lpd_ (price index, trailing-12m dividends, P/D), and for the S&P v_sp_lp/le/ld/lpe/lpd/lcape/lcpi/le10r. The Shiller workbook is the newer file in factors/raw (E and D to 2026-06).
  - extra adds p_/d_/r_buseq (Ken French BusEq).
- conditional_paths(model_fn, panel, protocol="split"|"rolling", horizons_q=(1,4,12,20), assets=SCORE_ASSETS, name=None, band=True, n_paths=500, seed=11, splits=SPLITS, roll_first="1985-01", roll_refit=12, roll_last=None, inside=True, bench_start="1973-01")
  - Returns one row per origin x horizon x asset: model, protocol, group, train_end, origin, end, h_q, months, asset, y, yhat, bench, q10, q50, q90, pit, n_fit. y is the realized sum of log1p(p); bench is months x the training-mean monthly log return.
  - Origins are quarter starts. With inside=True a window must end inside its test decade.
  - "split" trains to 1994-12 / 2004-12 / 2014-12.
  - "rolling" refits every January on data to the previous December; the April, July and October origins reuse that fit. Origins run from 1985-01.
- metrics(paths, spans=None, n_boot=1000, seed=0, score_assets=None) -> one row per model x protocol x group x asset (plus "mean10") x h_q. Columns: n, n_indep, r2 (vs the training-mean drift), r2_lo90/r2_hi90 (moving-block bootstrap over origins, block = max(h,4) quarters, only when n_indep >= 3), r2_nonoverlap, corr, mae, mae_bench, bias (yhat - y), hit, hit_excess, cov80, below10, above90, pit_mad, sd_y, sd_yhat, and r2_median on the mean10 rows.
- mean_table(m, value="r2", asset="mean10") pivots one value by horizon.
- Models:
  - linear_model(LinearSpec), where LinearSpec(name, common, extra={asset: cols}, start={asset: first month}, assets) is a per-asset OLS on a joint training mask.
  - Built-in specs: B0, B1, S, F.
  - drift_model() is the benchmark itself, so its R^2 is exactly 0.
- Helpers:
  - visible_panel(P, origin, end)
  - known_at(V, col, origin, lag=KNOWN_LAG default)
  - engine_residual_draw(dates, months, n_paths, seed), which is the engine's own residuals.prior_draw: 6-month blocks, 2020-03..08 at weight 0.25.
  - decomposition_levels / decompose_windows(P, asset, origins, months, measure in {pd, pe, cape, sp_pe, sp_cape})
  - variance_anatomy(P, paths, assets, measures, spans, horizons_q, protocol_pref, model)
  - lead_check(), selfcheck()

PLUGGING IN A MODEL
- Write model_fn(train, train_end) -> Fitted. `train` is panel.loc[:train_end]; nothing later exists for the fit.
- The Fitted object needs `assets` and predict_path(vis, origin, months), which returns a months x assets DataFrame of monthly LOG price returns. Only window sums are scored, so a quarterly or horizon-level model may put its whole prediction in the last month.
- What the model can see:
  - `vis` is the panel cut at the window's last month, with every asset-derived column blanked from the origin on. Blanked: the p_/r_/d_/dy_/v_ prefixes, lvl_gold, lvl_nasdaq, s_div_yield, f_valuation, s_cape, f_cape, s_wealth_income, f_wealth and lvl_networth.
  - Macro factors are visible through the end of the window (oracle macro).
  - Everything in months up to origin-1, valuations included, is known at the start. known_at() applies publication lags: Shiller E-based columns lag 3 months.
- Bands, optional: set `resid` to the training monthly log residuals and the harness draws them with the engine sampler, or provide simulate_paths(vis, origin, months, n_paths, seed) -> (n_paths, months, assets).
- For B0 and F the band is engine-faithful: simple-return residual added to the simple prediction, floored at -95% a month. This is the "decoupled" analog, since the factor path is fixed. The point forecast is sum(log1p(phat)) + c_a, where c_a is the in-sample log-bias correction.
- example_plugin.py has two worked examples.

CHECKS, all passed
- LinearFitted B0 matches AssetFactorModel to 4.7e-16, and its residuals match MonthDB's to 1.8e-15.
- The engine draw with my shim is identical to prior_draw on a real MonthDB.
- Masking works. v_lpd_us_equity equals log(100/s_div_yield) to 1.8e-15. The drift model scores |R^2| &lt; 7e-16. Single windows checked by hand match to 4 decimals.
- Lead's quick numbers (simple sums, calendar periods, M/Q/Y): -0.024/-0.041/-0.102, -0.107/-0.430/-0.245, +0.172/+0.280/+0.299. Maximum deviation 0.0005, which is rounding. In the harness's log space they become 1Q -0.046/-0.414/+0.284 and non-overlapping 4Q -0.093/-0.230/+0.306.
- Factor-round monthly scores reproduced exactly: B0 -0.0239/-0.1071/+0.1722, B1 -0.0114/+0.0162/+0.2128, F +0.0839/+0.1858/+0.2678. S scores +0.0754/+0.1679/+0.2555.

BASELINES. Mean over the 10 score assets of the OOS R^2 of the cumulative log price return, by horizon 1Q / 4Q / 12Q / 20Q. Independent windows per decade: 40/10/3/2 (47/11/3/2 for 2015-26); pooled rolling 1985-2026: 167/41/13/8.

| Model, protocol | Decade | 1Q | 4Q | 12Q | 20Q |
|---|---|---|---|---|---|
| B0 split | 1995-2004 | -0.05 | -0.19 | -0.58 | -0.67 |
| B0 split | 2005-2014 | -0.41 | -0.29 | -0.42 | -0.61 |
| B0 split | 2015-2026 | +0.28 | +0.23 | -0.02 | -0.34 |
| F split | 1995-2004 | +0.13 | -0.03 | -0.34 | -0.43 |
| F split | 2005-2014 | +0.26 | +0.30 | +0.21 | +0.20 |
| F split | 2015-2026 | +0.44 | +0.35 | +0.05 | -0.36 |
| B0 rolling | 1985-1994 | -0.01 | +0.23 | -0.00 | -0.05 |
| B0 rolling | pooled 1985-2026 | +0.02 | +0.01 | -0.26 [-0.55,+0.01] | -0.28 [-0.60,-0.09] |
| F rolling | 1985-1994 | -0.01 | +0.27 | +0.06 | -0.02 |
| F rolling | pooled 1985-2026 | +0.23 [+0.13,+0.32] | +0.22 [+0.09,+0.33] | -0.01 [-0.22,+0.17] | -0.03 [-0.25,+0.18] |

Brackets are 90% block-bootstrap intervals.

us_equity, split, 1Q / 4Q / 12Q / 20Q:

| Model | 1995-2004 | 2005-2014 | 2015-2026 |
|---|---|---|---|
| B0 | -0.15 / -0.42 / -0.62 / -0.68 | -0.34 / -0.19 / -0.31 / -0.36 | +0.38 / +0.35 / -0.18 / -0.73 |
| F | +0.18 / -0.15 / -0.34 / -0.37 | +0.49 / +0.56 / +0.50 / +0.58 | +0.65 / +0.56 / +0.04 / -0.44 |

nasdaq, split, 1Q / 4Q / 12Q / 20Q:

| Model | 1995-2004 | 2005-2014 | 2015-2026 |
|---|---|---|---|
| B0 | -0.09 / -0.24 / -0.34 / -0.35 | -0.24 / -0.04 / -0.19 / -0.24 | +0.32 / +0.34 / -0.08 / -0.49 |
| F | +0.08 / -0.14 / -0.25 / -0.27 | +0.45 / +0.50 / +0.69 / +0.79 | +0.46 / +0.41 / +0.07 / -0.27 |

Coverage of the engine's 80% band, mean10, split, 1Q / 4Q / 12Q / 20Q:

| Model | 1995-2004 | 2005-2014 | 2015-2026 |
|---|---|---|---|
| B0 | .77 / .65 / .53 / .59 | .74 / .76 / .65 / .59 | .83 / .89 / .91 / .90 |
| F | .77 / .68 / .57 / .59 | .81 / .81 / .79 / .83 | .86 / .90 / .90 / .90 |

- When the realized value falls outside the band it is mostly below: below10 is 0.26-0.31 at 12-20Q in 1995-2014 for B0.
- Sign hit, pooled rolling 1Q / 4Q / 12Q / 20Q: B0 .68/.75/.82/.87, F .71/.77/.82/.86. The raw sign hit is inflated by positive drift at long horizons; hit_excess (sign relative to the drift) is in the CSV.

VARIANCE ANATOMY. us_equity, B0, CAPE decomposition. Each cell is share_m | r2_orc_m | B0 r2 | corr(yhat, d multiple) | corr(miss, d multiple)^2, at 1Q / 4Q / 12Q / 20Q.
- share_m = Cov(dp, dm)/Var(dp), the multiple's share of price-change variance.
- r2_orc_m = the R^2 you would get knowing only the realized multiple path, with fundamentals at drift.
- 1985-94 uses the rolling protocol; the other decades use split.

| Decade | 1Q | 4Q | 12Q | 20Q |
|---|---|---|---|---|
| 1975-84 (share_m only) | 1.02 | 1.03 | 0.97 | 1.19 |
| 1985-94 | 1.00\|.97\|+.00\|+.17\|.85 | 1.03\|.83\|+.44\|+.68\|.62 | 1.08\|.24\|+.06\|+.61\|.24 | 1.01\|.17\|+.38\|+.57\|.55 |
| 1995-2004 | 1.01\|.99\|-.15\|-.09\|.90 | .98\|.99\|-.42\|-.40\|.90 | .96\|.99\|-.62\|-.61\|.94 | .94\|.99\|-.68\|-.86\|.98 |
| 2005-14 | .95\|.96\|-.34\|-.22\|.83 | .92\|.93\|-.19\|-.04\|.81 | .80\|.87\|-.31\|-.18\|.77 | .84\|.88\|-.36\|-.27\|.96 |
| 2015-26 | 1.03\|.97\|+.38\|+.73\|.86 | 1.10\|.92\|+.35\|+.83\|.83 | 1.02\|.77\|-.18\|+.85\|.48 | 1.10\|.63\|-.73\|+.76\|.43 |

- CRSP P/D gives the same picture, except 2005-14 at 12-20Q where share_m is 0.56-0.60 (the 2009 dividend cut).
- Six industries plus REITs, each with its own P/D: share_m 0.72-1.07, corr(miss, dm)^2 0.43-0.89.
- F changes 2005-14: corr(yhat, dm) +0.71 to +0.92 and corr(miss, dm)^2 0.12-0.34 at 1-12Q. 1995-2004 stays 0.83-0.96.
- 2015-26 bias, B0: the realized return exceeds the model by 0.18 log points (12Q) and 0.30 (20Q). The re-rating beyond the 1973-2014 average explains 0.13/0.20 of that on P/D but only 0.07/0.07 on CAPE. On CAPE the rest is CPI x E10 growth above its drift (0.24 vs 0.18 at 12Q, 0.43 vs 0.30 at 20Q).
- Naive CAPE-pull demo (not tuned): pooled 12Q R^2 goes from -0.26 to -0.59. For us_equity at 12Q it helps 1995-2004 (-0.62 to -0.29) and 2005-14 (-0.31 to +0.01) but wrecks 2015-26 (-0.18 to -1.49).

## Key findings

1. The harness is validated end to end: it reproduces the lead's quick check (maximum deviation 0.0005) and the factor round's B0/B1/F walk-forward scores to 4 decimals. B0 matches the project's AssetFactorModel to 5e-16, the residual draw is identical to the engine's own residuals.prior_draw, the drift-only model scores R^2 = 0 to 7e-16, and single windows checked by hand match.
2. On both models R^2 does not improve beyond one year; it falls. Pooled rolling 1985-2026, mean over 10 assets, 1Q/4Q/12Q/20Q: B0 +0.02/+0.01/-0.26/-0.28, F +0.23/+0.22/-0.01/-0.03 (independent windows 167/41/13/8). Quarterly and annual are about equal; multi-year is worse. The only decade where F holds up at long horizons is 2005-14 (+0.21 at 12Q, +0.20 at 20Q).
3. Every multi-year statistic rests on very few independent windows: 3 per decade at 12Q and 2 at 20Q. The 12Q bootstrap intervals use about 3 blocks and are unreliable; 20Q has no interval.
4. In 1995-2004 the macro factors pointed the wrong way: B0's prediction correlates negatively with the realized us_equity move at long horizons (-0.65 at 12Q, -0.87 at 20Q). F does not fix this decade (-0.34/-0.37 at 12/20Q).
5. In 2015-26 the long-horizon miss is a level bias, not a direction error. us_equity correlation is still 0.71/0.54 (B0) and 0.76/0.77 (F) at 12/20Q with a 100% sign hit rate, but the models are 0.16-0.30 log points too low over 3-5 years. On P/D, the re-rating beyond the 1973-2014 average accounts for 0.13/0.20 of it; on CAPE only 0.07/0.07, and the rest is 10-year earnings growth above its training drift.
6. The lead's hypothesis holds: the long-horizon misses are valuation swings. On the CAPE or P/D split the multiple carries about 80-110% of us_equity price-change variance at every horizon in every decade. The share of B0's miss variance explained by the multiple change, corr(miss, d multiple)^2, is 0.90-0.98 in 1995-2004, 0.77-0.96 in 2005-14 and 0.43-0.86 in 2015-26. Knowing the realized multiple path alone would give R^2 0.77-0.99 at 1-12Q from 1995 on, against B0's -0.62 to +0.38.
7. F's gain in 2005-14 comes from pricing the multiple move: corr(yhat, d multiple) +0.71 to +0.92, and its misses there are no longer aligned with the multiple (0.12-0.34 at 1-12Q). The likely reason is that the GZ credit spread is itself a risk-premium market price, so this gain is partly endogenous.
8. The engine's own residual band (6-month blocks) is calibrated at 1Q (coverage 0.74-0.86) but too narrow at 12-20Q in 1995-2014 (0.53-0.65, with misses mostly below the band), and too wide in 2015-26 (about 0.90). It cannot produce multi-year valuation swings.
9. Trailing-12m P/E is not a usable fundamentals split: the 2008-09 write-downs give it a multiple share of -1.1 to -2.5 in 2005-14. CAPE and P/D are usable. For nasdaq the 'P/D' is BusEq's dividend yield, and BusEq's own P/D breaks around 2004-05 (Microsoft's special dividend), so there is no clean tech fundamentals series in the allowed sources.
10. A naive valuation state does not work: a CAPE gap against its 1881..train-end mean (example_plugin.py, not tuned) helps us_equity at 12-20Q in 1995-2014 but wrecks 1985-94 and 2015-26 (pooled 12Q R^2 -0.26 to -0.59). A fixed long-run anchor fails after 1990, consistent with the earlier valuation.json result.

## Open questions

1. Should windows be allowed to cross decade ends (conditional_paths inside=False)? That raises the 12-20Q sample from 2-3 to about 3-4 independent windows per split, but mixes decades: a 1995-2004 statistic would include the 2005-09 GFC.
2. The band question is for the engine design: the 6-month-block residual draw cannot produce multi-year valuation swings, so 12-20Q coverage is 0.53-0.65 in 1995-2014. Options are a valuation-state component (the persistence of the multiple) or longer residual blocks.
3. How should the multiple enter the model? A fixed CAPE anchor fails. The candidates are a moving fair value (real yield, profit share: the project's ValuationModel) or a multiple path stated by the scenario. A scenario-stated path is realized valuation, so the harness can only score it as an oracle upper bound (r2_orc_m), not as a model input.
4. For QQQ: the allowed sources have no clean NASDAQ-100 earnings or FCF series (BusEq dividends break in 2004-05; the nasdaq 'P/D' is BusEq's yield). Should we use the S&P (Shiller) proxy, or build a sector-profit proxy?
5. F's gold equation (fitted from 1982) is much worse at long horizons in 2015-26 (-0.98/-2.63 at 12/20Q vs B0 -0.53/-0.84), and in rolling refits before 1992 it has only 36-120 training months. Is that acceptable for the final model?
6. The engine layer (backtest fan_cond with LLM-translated phase views, the user's headline end-to-end test) is not wrapped by this harness. It scores the transmission layer under oracle macro. A bridge from a translated spec to backtest.run_custom or the scenario runner is still needed.

## Files the role listed

Paths are relative to `scratchpad/sim_redesign/`. Only the files marked are in this archive; the rest (scripts, work outputs, data) stay in the research folder.

- `harness/simharness.py`
- `harness/example_plugin.py`
- `harness/work/r01_checks.py`
- `harness/work/r01_checks.json`
- `harness/work/r02_baselines.py`
- `harness/work/paths_baselines.csv`
- `harness/work/metrics_baselines.csv`
- `harness/work/r04_anatomy.py`
- `harness/work/anatomy_B0.csv`
- `harness/work/anatomy_F.csv`
- `harness/work/r06_summary.py`
- `harness/work/r06_summary.log`
- `harness/work/r04_anatomy.log`
- `harness/work/metrics_example_capepull.csv`
- `harness/work/example_plugin.log`
- `harness/work/r03_tables.py`
- `harness/work/r05_spot.py`
