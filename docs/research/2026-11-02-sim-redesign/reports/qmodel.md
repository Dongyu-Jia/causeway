# The quarterly three-input model (research round, role `qmodel`)

*What the role did:* the model G on [s_q, x_q x m, m], the masked NN comparison, masking results, the plug-in.

Rendered from the role's final report in the research workflow's journal (run `wf_cc7fe8c3-afc`, result `qmodel`). The text is unchanged except that local absolute paths are written as the spec writes them (`scratchpad/sim_redesign/...`, `scratchpad/...`). It is the report the critic and the designer read; the numbers are the role's own, and the critique ([`critic/critique.md`](../critic/critique.md)) and the spec ([`SIM_DESIGN.md`](../SIM_DESIGN.md)) say which of them survived.

## Summary

I built the quarterly three-input model from the user's fourth message, compared it walk-forward with L0, L1, L2, G and NN, and wrapped the best model as a plug-in. The best model is G, a structured Gaussian. With the quarter's x fully given (oracle), it roughly ties today's monthly transmission (L0) at one quarter. The NN loses everywhere. Under a narrative that states only macro and policy, quarterly returns are not predictable at all. I wrote nothing in MAIN, RWT or FWT: every script ran with -B, nothing was downloaded, and no git command ran. MAIN's .git metadata changed during the round because of other worktree rounds; RWT has the regime round's edits.

DATA (qdata.py -> work/qpanel.csv, 216 quarters 1973Q1..2026Q4; source, timing and lag of every column in qpanel_meta.json)
- Outputs y:
  - the 12 harness assets as quarterly log price returns (sum of the monthly log1p(p)); they match the harness exactly;
  - cash as the bill total return;
  - y_mult = dlog(P/E-hat), with E-hat = the 4-quarter mean of Shiller trailing EPS, plus y_eps = y_us_equity - y_mult (identity error 1e-5).
- Inputs x: the 52 final-schema fields.
  - Taken from the schema rounds' quarterly builds; spot checks against raw files were exact (d_bs, d_oil, x_lending_std).
  - d_tips10 (DFII10) added; d_lf with the January population-control jumps removed.
  - x_oil_cause by the Kilian same-sign rule (56% of quarters are supply), carried as d_oil_sup = d_oil x 1{supply}.
  - Late series are masked before they start, not invented.
- Five policy shocks u_ff, u_bs, u_def, u_tr, u_ptax come from reaction functions and stabilisers fitted to each train end. They reproduce the schema's R2 exactly: u_ff 0.54/0.51/0.46 on n = 86/126/142.
- States s (known at the start; lags in the meta):
  - 17 states: fed funds; term spread; Baa-10y; real 10y; core y/y; EWMA inflation trend; unemployment; Sahm gap; CRSP P/D; CAPE; E/E10; NFC FCF yield; real oil; real dollar; volatility; zero-bound flag; core > 4% flag;
  - 1- and 4-quarter momentum of all 13 assets;
  - the previous quarter's x.
- Where I differ from your candidate list, I followed the final schema:
  - fed funds instead of Wu-Xia;
  - monetary base/GDP instead of WALCL, which starts in 2002;
  - PCTR/PI;
  - Philadelphia Fed delivery times instead of GSCPI;
  - d_lf instead of CIVPART;
  - NIPA FGDEF;
  - GZ and Baa-10y (EBP is price-derived);
  - policy stance through d_y2 and d_ff.
- c_q (the pre-quarter text) maps to three things:
  - states: stance = s_ff, s_term; stress = s_spread, s_vol; euphoria = P/D, CAPE, E/E10, momentum;
  - the x rows of future quarters, for announced paths;
  - stated output legs y_mult and y_eps.
  Two soft states are dropped because they have no measured proxy that is not a price: sentiment beyond valuation and momentum (VIX and EBP are price-derived and never inputs), and policy credibility.

MODELS (qcore.py)
- L0: the harness's monthly F, aggregated to quarters.
- L1: quarterly ridge on the schema's tier-1 fields, with the F coefficients shrunk toward L0's monthly betas.
- L2: L1 plus regime states, own momentum and three state x shock interactions.
- L2-allstates: L2 plus the valuation and macro states.
- G: x | s Gaussian, with missing-data EM covariance (factor-analytic or shrinkage, chosen by masked-entry error), plus the L2 equations.
  - Masked x is filled in by exact conditioning, and its uncertainty is carried into the returns.
  - With x fully given, G equals L2.
- G-full: an unstructured joint Gaussian of [x, r] given s.
- NN: a masked MLP in numpy, with inputs [s, x*m, m] and outputs [r, x].
  - One tanh layer with a linear skip connection, an L2 penalty chosen by time-ordered validation, and L-BFGS on mask-augmented data.
  - 4 seeds per fit.
- NN-res: G plus an MLP correction.
- Every model saturates standardized inputs at +-4 training sd, fixed a priori; mechanical own-price inputs (oil for crude, the 10y yield for the bond) are exempt.
  - Without saturation, G's O-A R2 for 2015-26 is -0.578, against -0.035 with it.
  - Without saturation, 2008Q4's d_bs (about 77 training sd) and the 2020 macro moves break the linear completion.
- There are no episode flags.

RESULTS
Mean R2 over the 10 score assets, against the training drift.
- Split protocol: test windows 1995-2004 / 2005-14 / 2015-26, with 40/40/47 quarters.
- Rolling protocol: pooled 1985-2026, 167 quarters.

| model | split 1995-2004 | split 2005-14 | split 2015-26 | rolling pooled |
|---|---|---|---|---|
| L0 | +0.126 | +0.261 | +0.441 | +0.226 |
| G | +0.152 | +0.226 | +0.492 | +0.222 |
| NN | -0.062 | +0.004 | +0.185 | +0.084 |
| NN-res | +0.148 | +0.258 | +0.476 | +0.231 |

1Q, x fully given:
- G minus L0, split: +0.026 [+0.004, +0.050], -0.034 [-0.095, +0.061], +0.051 [+0.022, +0.093].
- G's CRPS skill and 80% coverage, split: CRPSS +0.073/+0.160/+0.273, coverage 0.78/0.79/0.86.

Masking:
- Half of x masked at random, G: R2 +0.047/+0.091/+0.277 (rolling +0.119); 80% coverage about 0.80.
- Half masked, L2 with the masked x set to its training mean: R2 +0.046/+0.115/+0.276; coverage about 0.76.
- So the learned completion improves calibration but not accuracy.
- O-A (E0 and E0p given only), G: -0.038/-0.291/-0.035; L2 0.000/-0.110/-0.005; rolling about -0.03. Macro and policy alone explain nothing quarterly.
- Completion of masked market fields from macro is at or below zero out of sample under O-A (median completion R2 -0.06, pooled rolling). The zero bound and 2008 break the inferred rates: d_ff -1.22 and d_y2 -1.4 to -1.9.

Recursive paths with realized x, G against L0:

| horizon | split 1995-2004 | split 2005-14 | split 2015-26 | rolling pooled | independent windows (split; rolling) |
|---|---|---|---|---|---|
| 12Q | -0.476 vs -0.336 | +0.251 vs +0.215 | +0.149 vs +0.049 | -0.127 vs -0.007 | 3 per decade; 13 |
| 20Q | -0.574 vs -0.431 | +0.293 vs +0.201 | -0.052 vs -0.363 | -0.197 vs -0.030 | 2 per decade; 8 |

- G - L0 pooled rolling: 12Q -0.119 [-0.197, -0.052]; 20Q -0.167 [-0.255, -0.088]. The deficit is driven by 1985-2004, when the quarterly model had 47-127 training quarters.
- 80% coverage at 12-20Q in 1995-2004 is 0.48-0.53.

2020-21 (refits at 2019Q4 and 2020Q4, oracle x, no pandemic flag):
- Pooled over 8 quarters x 10 assets, G R2 0.638 with 80% coverage 0.86; L0 0.636 with coverage 0.76.
- us_equity 2020Q1-Q4: G -0.21/+0.15/+0.09/+0.11 against an actual -0.23/+0.20/+0.09/+0.14.
- Under O-A the sequence is wrong. The completion has spreads rising in Q2 (+0.70) when they fell (-1.49), and the dollar falling in Q1 (-0.97) when it rose (+6.63). Stating d_gz, d_y10 and d_oil restores the path.

PLUG-IN (qplugin.py; README.md documents it)
- predict(origin_month, x_table, n_paths, seed, model="G") fits on quarters before the origin. It conditions on whatever the x_table gives (NaN = inferred) and samples the masked inputs. It rolls macro and asset states path by path and returns simulated quarterly paths for the 13 assets plus the valuation leg.
- It accepts policy interventions u_* (do-operator, with the identity raw field = systematic part + u imposed exactly) and stated legs y_mult and y_eps (V-PRICE identity exact on every path). It is deterministic given the seed.
- harness_model_fn() plugs it into simharness.conditional_paths and reproduces these scores exactly (0.152/0.226/0.492).

THE NN QUESTION
- The NN does not beat G or L2 in any split. At 1Q full x the gap is -0.21/-0.22/-0.31, and every interval excludes 0. Its only non-negative cells are rolling 1985-94 and two O-A cells, none significant.
- Seed sd of the NN's R2 is 0.011-0.014. NN-res is within +-0.05 of G and never significant; its validation mostly chose near-zero corrections.

## Key findings

1. One quarterly model was built as specified. The plug-in model is G, a structured Gaussian. Inputs are the 52 final-schema x fields (maskable), 5 policy shocks, and states known at the start, including P/D, CAPE, E/E10 and FCF yield. Outputs are 13 asset returns plus the valuation leg y_mult. The reaction functions reproduce the schema's numbers exactly (u_ff R2 0.54/0.51/0.46, n = 86/126/142). Late series are masked before they start, and there are no episode flags.
2. 1Q with x fully given (split; 40/40/47 quarters): G (= L2) mean10 R2 +0.152/+0.226/+0.492 against L0 +0.126/+0.261/+0.441. G - L0 = +0.026 [+0.004, +0.050], -0.034 [-0.095, +0.061], +0.051 [+0.022, +0.093]. Pooled rolling 1985-2026: G +0.222 [+0.14, +0.31] vs L0 +0.226 (167 quarters), a tie. Quarterly fitting only competes because the F coefficients are shrunk toward the monthly betas: without that prior, the quarterly F scores 0.141/0.154/0.358.
3. The NN does not beat G or L2 in any split. At 1Q full x it scores -0.062/+0.004/+0.185 (rolling +0.084), and NN - G = -0.214, -0.222 and -0.307, with every 90% interval excluding 0. Its only non-negative cells are rolling 1985-94 and two O-A cells, none significant. Across 4 seeds, the sd of its R2 is 0.011-0.014. NN-res (G plus an MLP correction) is within +-0.05 of G and never significant: validation picked heavy penalties in 2 of 3 splits.
4. Masking: with half of x masked at random, G scores R2 +0.047/+0.091/+0.277 (rolling +0.119) against L2 with masked inputs set to their training mean at +0.046/+0.115/+0.276. Gaussian completion buys calibration (80% coverage 0.79-0.83 vs 0.76-0.81) but not accuracy. Median completion R2 of the masked fields is about 0.12-0.15; d_y10 0.50, d_ip 0.38 and d_unrate 0.31 complete well, while d_ff (-1.22) and d_usd (0.04) do not.
5. With only macro and policy given (O-A), quarterly returns are unpredictable: G -0.038/-0.291/-0.035, L2 0.000/-0.110/-0.005, rolling about -0.03. Inferring market prices from macro has non-positive out-of-sample value (median completion R2 -0.06 pooled). The zero bound and 2008 break the inferred rates, because the x-block covariance does not depend on the regime. A narrative must state financial conditions (credit spread, rates, dollar) for the model to say anything.
6. Recursive paths with realized x, G vs L0. 4Q: -0.047/+0.250/+0.365 vs -0.031/+0.302/+0.347 (10/10/11 independent windows). 12Q: -0.476/+0.251/+0.149 vs -0.336/+0.215/+0.049 (3 per decade). 20Q: -0.574/+0.293/-0.052 vs -0.431/+0.201/-0.363 (2 per decade). Pooled rolling, G is behind L0 at 12Q (-0.119 [-0.197, -0.052]) and 20Q (-0.167 [-0.255, -0.088]) because of the short training samples of 1985-2004. 80% coverage at 12-20Q in 1995-2004 is only 0.48-0.53.
7. Valuation as a state adds nothing out of sample: L2-allstates minus L2 is +-0.02 in every split (for example -0.008/+0.008/+0.002 at 1Q full). Valuation enters usefully as a stated leg in the plug-in. Example, 2022-23 with oracle x: the us_equity median is +0.16 (actual -0.015). Adding the realized multiple leg moves it to -0.15; stating both legs fixes it exactly and tightens the other equities. Energy's +41% in 2022-23 is missed in every setting.
8. 2020-21 without a pandemic flag: the generic channels reproduce the equity path under oracle x. Pooled over 8 quarters x 10 assets, G R2 is 0.638 with 80% coverage 0.86, against L0 at 0.636 with coverage 0.76. us_equity 2020Q1-Q4: G -0.21/+0.15/+0.09/+0.11 vs actual -0.23/+0.20/+0.09/+0.14. Under O-A the timing is wrong: the completion has credit spreads rising in Q2 (+0.70) when they fell (-1.49). Stating d_gz, d_y10 and d_oil restores the path.
9. Saturating inputs at +-4 training sd (fixed a priori) is essential for any linear Gaussian completion. Without it, G's O-A R2 for 2015-26 is -0.578, against -0.035 with it. Raw d_bs in 2008Q4 is about 77 training sd and d_deficit in 2020 about 33 sd. Mechanical own-price inputs (crude ~ d_oil, the 10y bond ~ d_y10) are exempt; with that, crude R2 is >= 0.9998 and the 10y bond 0.96-0.99.
10. The plug-in qplugin.predict(origin_month, x_table, n_paths, seed) fits only on data before the origin and gives deterministic joint paths. It accepts the 52 schema fields, policy interventions u_* (the do-operator: a -1pp intervention cut leaves unemployment unchanged and gives d_ff -0.77, the systematic part plus u) and stated legs y_mult/y_eps. It rolls states path by path and also runs 10-year roll-forwards from 2026Q4. harness_model_fn() plugs it into simharness and reproduces the scores exactly (0.152/0.226/0.492).

## Open questions

1. The covariance that fills in missing inputs is the same in every regime, so at the zero bound (2009-15, 2020-21) and in 2008 it infers rate cuts that cannot happen (completion R2 d_ff -1.22, d_y2 -1.4 to -1.9). Should G get separate covariances for the zero-bound and high-inflation regimes, estimated where the training data contain them and shrunk toward the pooled one? Or should a narrative have to state the rate path?
2. Inferring market prices (spreads, rates, dollar) from macro alone has non-positive out-of-sample value at the quarterly frequency. Should the translator be required to state financial conditions (d_gz, d_y10, d_usd at least)? Or should the plug-in shrink those inferences toward the drift when they are not stated?
3. From macro, the model's earnings and valuation legs are weak or wrong. 2022-23: EPS rose about 10% while the model's earnings draw was about 0. 2020-21: the valuation leg was -0.27 against +0.20. Should the scenario layer always state both legs (X01/X27), and how should their bounds be calibrated? The plug-in already accepts y_mult and y_eps exactly.
4. Long-horizon bands are too narrow in 1995-2004: 80% coverage 0.48-0.53 at 12-20 quarters, because quarterly residuals are drawn iid. Should we add a persistent valuation-state residual or multi-quarter residual blocks?
5. In the rolling protocol G trails L0 at 12-20 quarters (-0.12/-0.17 pooled), because early refits have 47-127 training quarters. Options: a stronger monthly prior while samples are short, a minimum training length, or using L0 for the early decades.
6. Evaluation caveats to accept or fix: HLW r* is the current vintage, not real time (it is only in the auxiliary states, not the final model). The previous quarter's NIPA-based x is treated as known at the start even though it is released weeks into the quarter (the oracle-macro assumption). The q05 rows for crude oil and the 10-year bond come from a run before their own-price inputs were exempted from saturation; q14 gives the final values, and the mean of the 10 score assets is unaffected.
7. The headline end-to-end test is still pending: the written narratives (E01-E12) have not been run through the translator into the plug-in. The translator must emit exactly the x_table columns: the 52 schema fields, x_oil_cause, the u_* shocks and the y_mult/y_eps legs. Does the scorer call qplugin.predict directly or through harness_model_fn?

## Files the role listed

Paths are relative to `scratchpad/sim_redesign/`. Only the files marked are in this archive; the rest (scripts, work outputs, data) stay in the research folder.

- [`qmodel/README.md`](../qmodel/README.md) (in this archive)
- `qmodel/qplugin.py`
- `qmodel/qcore.py`
- `qmodel/qdata.py`
- `qmodel/qeval.py`
- `qmodel/q05_run.py`
- `qmodel/q06_report.py`
- `qmodel/q13_tables.py`
- `qmodel/work/qpanel.csv`
- `qmodel/work/qpanel_meta.json`
- `qmodel/work/q13_tables.log`
- `qmodel/work/q06_report.log`
- `qmodel/work/q06_metrics.csv`
- `qmodel/work/q06_diffs.csv`
- `qmodel/work/q06_seeds_split.csv`
- `qmodel/work/q06_completion_rolling.csv`
- `qmodel/work/q05_split_rows.csv.gz`
- `qmodel/work/q05_rolling_rows.csv.gz`
- `qmodel/work/q11_covid.log`
- `qmodel/work/q10_plugin_test2.log`
- `qmodel/work/q09_plugin_test.log`
- `qmodel/work/q12_harness_adapter.log`
- `qmodel/work/q07_clip.log`
- `qmodel/work/q04_ablation.log`
- `qmodel/work/q14_mechanical.log`
