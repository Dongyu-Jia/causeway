# Joint quarterly factor model (research round, role `joint`)

*What the role did:* the quarterly BVAR, Waggoner-Zha conditioning, the C' cycle hybrid, persistence diagnostics.

Rendered from the role's final report in the research workflow's journal (run `wf_cc7fe8c3-afc`, result `joint`). The text is unchanged except that local absolute paths are written as the spec writes them (`scratchpad/sim_redesign/...`, `scratchpad/...`). It is the report the critic and the designer read; the numbers are the role's own, and the critique ([`critic/critique.md`](../critic/critique.md)) and the spec ([`SIM_DESIGN.md`](../SIM_DESIGN.md)) say which of them survived.

## Summary

## Recommendation: C′, a quarterly joint engine split by variable

Use a quarterly Bayesian VAR whose conditional simulation fills every factor the scenario leaves out. The fill is two-sided across quarters. Four add-ons each answer a failure measured below:

1. **Business cycle.** Where the scenario does not state the cycle, draw the IP path from the engine's own historical 6-month blocks and pass it to the BVAR as constraints. Linear dynamics cannot produce short, sharp contractions.
2. **Explicit anchors.** The scenario layer sets the long-run real policy rate, real 10-year, inflation and unemployment. Defaults are the engine's `DEFAULT_REVERSION` values, not the VAR's 1973+ intercepts.
3. **Zero bound.** The policy rate is censored at 0.1.
4. **USD.** Zero drift plus noise unless the scenario states it.

**Frequency.** The dynamics engine, the LLM's statements and the qmodel all run quarterly. Monthly chart paths come from splitting each simulated quarter with a historical quarter's within-quarter profile, which keeps quarterly totals exact. That split is designed but not tested.

**Why not one of the three options as posed.**
- (A), the engine: keeps the cycle's shape. It breaks co-movement and inflation persistence, and it delivers a stated path only on average.
- (B), a plain BVAR: gets co-movement, persistence and exact delivery right. Its business cycle is wrong.
- The lead's (C), all macro from blocks plus a quarterly asset mapping: inherits A's macro numbers exactly. The useful hybrid splits by variable, not by layer.

## What I built and checked

- **Read-only.** Everything ran under a write-guard covering MAIN, RWT and FWT, with 0 blocked writes. No MAIN file is newer than my start stamp. RWT and FWT files that changed during the run were edited by the other implementation rounds.
- **Quarterly data.** Built from MAIN's panel plus the factor round's level files. A quarter's change equals the window sum of the monthly factor changes; checked, the maximum error is 2e-15.
- **Variables (10).**
  - Headline and services CPI inflation.
  - IP growth.
  - Log real oil.
  - Fed funds and the 10-year, nominal or real (rate minus a 36-month-half-life CPI moving average).
  - The fixed Baa−10y spread and the GZ spread.
  - Log USD.
  - Unemployment.
- **BVAR(2).**
  - Minnesota and sum-of-coefficients dummy priors.
  - λ chosen by marginal likelihood on each training window only (0.25–0.35).
  - Pandemic quarters 2020Q2–2021Q2 excluded.
  - 100 posterior parameter draws; explosive draws are redrawn (22% of raw draws).
  - Whitened residual vectors bootstrapped.
- **Constraints.** Hard or soft, imposed by projecting the shocks (Waggoner-Zha).
- **Checks.**
  - The OLS limit matches to 6e-8.
  - Constraints are met to 5e-14 on every path.
  - The Gaussian Monte Carlo mean equals the analytic conditional mean (maximum |z| 2.5 over 120 cells).
- **Variants.**
  - B3: real rates.
  - B3ss: B3 with the engine's anchors.
  - B4: an IP-drawdown "bounce-back" state plus the zero bound.
  - B5: B4 plus a contraction regime (intercept shift, regime shock pools, empirical durations).
  - C′: the recommended hybrid.

## Persistence

Unconditional runs from 2026Q3, 2000 paths, quarters 9–60. History is 1973Q1–2026Q3 (215 quarters), labelled with the regime classifier on quarters. The monthly classifier reproduces the engine reference exactly (40.77% / 17.29%).

| | CPI y/y > 4% | CPI p10 / 50 / 90 | CPI autocorr. at 4q | high-inflation spell (q) | contractions per 15y | mean length (q) | longer than 6q | bounce-back |
|---|---|---|---|---|---|---|---|---|
| History | 32.6% | 1.43 / 3.02 / 8.54 | +0.19 | 7.8 | 1.95 | 4.1 | 0 of 7 | −0.48 [−0.82, −0.30] |
| A, as built | 40.7% | 1.77 / 3.54 / 6.86 | −0.05 | 4.5 | 1.94 | 4.3 | 18% | +0.02 |
| A, 24-month blocks | 33.5% | 1.48 / 3.16 / 7.60 | +0.23 | 5.9 | 1.88 | 4.8 | 17% | — |
| A, state-matched chain | 16.8% | 1.58 / 2.90 / 4.56 | +0.07 | 3.5 | 0.91 | 6.0 | 35% | — |
| B | 41.4% | −0.06 / 3.41 / 6.98 | +0.30 | 7.2 | 2.50 | 8.6 | 55% | −0.11 |
| B4 | 45.3% | 0.20 / 3.67 / 7.62 | +0.30 | 7.9 | 2.61 | 7.7 | 47% | −0.32 |
| B5 | 43.1% | −0.04 / 3.51 / 7.54 | +0.35 | 7.8 | 1.93 | 7.5 | 40% | — |
| C′ | 43.4% | 0.08 / 3.55 / 7.23 | +0.31 | 7.5 | 1.83 | 4.6 | 22% | — |
| C′ with anchors | 36.6% | −0.46 / 3.04 / 6.78 | +0.31 | 6.8 | 1.81 | 4.6 | 22% | — |

C′ anchors: CPI 2.5, services 3.0, real policy rate 0.75, real 10-year 1.5, unemployment 4.8, spread 2.0.

## Consistency under the constraint

The constraint, from 2026Q3: CPI 6% a year for 2 years, IP −5% over the 2 years, oil +50%.

**What history says.**
- OLS on 195 overlapping 8-quarter windows (24 independent), Newey-West se.
- The nearest analogs are 1974-10, 1989-01 and 2006-10; none is a close match.

**2-year change of each unstated variable.**

| | History (se) | A | B |
|---|---|---|---|
| Fed funds | +1.00 (0.60) | +0.23 | +2.24 |
| 10-year | +0.80 (0.25) | +0.81 | +0.57 |
| Baa spread | +1.13 (0.18) | +0.78 | +1.05 |
| GZ | +1.28 (0.34) | −0.01 | +1.30 |
| Unemployment | +1.71 (0.25) | +1.79 | +1.15 |
| Services CPI | 6.25 (0.18) | 6.35 | 6.39 |

**Delivery.**
- B meets the stated values on every path.
- A meets them only on average: for a stated −5% the IP 2-year change runs −17% to +5% (p10/p90), and oil runs −28% to +111% for a stated +50%. Its pooling did not need to shrink (effective sample 469 months, 40.7 episodes) and put 40% of the weight on 1973–82 months.

**Within-quarter co-movement** against history's 90% block-bootstrap interval:
- B sits inside the interval on all five slopes tested.
- A is outside on two:
  - Policy rate on unemployment: −0.43 vs −1.44 [−2.02, −0.88].
  - Okun's slope: −0.064 vs −0.041 [−0.047, −0.034].
- C′ gives −1.43 and −0.033.

## Harness: partial oracle

Set-up:
- A scenario states realized CPI, IP and oil every quarter (S1); the engine fills the rest.
- One fixed transmission, fitted to the train end, maps factors to assets.
- Split protocol; scored with `sh.metrics`.

**Asset fit.** Pooled 1995–2026 mean R² over the 10 score assets, transmission with GZ. Independent windows: 126 / 31 / 10 / 6.

| | 1Q | 4Q | 12Q | 20Q |
|---|---|---|---|---|
| Full oracle | +0.22 [+0.13, +0.31] | +0.12 | −0.08 | −0.16 |
| A, CPI/IP/oil stated | +0.03 | −0.09 | −0.36 | −0.56 |
| B, CPI/IP/oil stated | −0.17 | −0.31 | −0.15 | −0.19 |
| A, nothing stated | −0.00 | −0.02 | −0.05 | −0.06 |
| B, nothing stated | −0.19 | −0.33 | −0.27 | −0.06 |

Stating the fed funds rate as well changes nothing.

**Factor fill-in.** R² of each unstated factor's window change against no change, pooled, 4Q / 12Q.

| Factor | A | Best BVAR variant |
|---|---|---|
| Fed funds | +0.47 / +0.71 | B4 +0.24 / +0.19 (plain B −0.13 / +0.05) |
| 10-year | +0.38 / +0.65 | B3ss +0.08 / −0.25 (plain B −0.28 / −1.59) |
| 10-year, fed funds also stated | +0.36 / +0.64 | B3ss +0.46 / +0.62 |
| Baa spread | +0.28 / +0.23 | B3ss +0.44 / +0.60 |
| GZ | −0.04 / −0.28 | B −0.10 / +0.30 |
| Unemployment | +0.39 / +0.49 | B +0.54 / +0.61 |
| Services CPI | −1.31 / −1.87 | B +0.11 / −0.32 |
| USD | −0.72 / −2.62 | negative everywhere (B −2.22 / −4.64) |

**80% band coverage at 20Q**, 1995–2004 / 2005–14:

| B, CPI/IP/oil stated | A, CPI/IP/oil stated | Full oracle |
|---|---|---|
| 0.80 / 0.76 | 0.64 / 0.63 | 0.64 / 0.70 |

## How it plugs into the qmodel rollout

1. **Schema.** The LLM's output schema is the BVAR variable list, each a measured quarterly series.
   - A statement is a linear row on the stacked path: a level, change, average or y/y over named quarters.
   - It is hard, or soft with an sd taken from the stated confidence.
   - A stated fed funds rate becomes a linear row on the real rate and the CPI path.
   - A range is handled by accept/reject over the ensemble.
2. **Joint draw.** From the anchor state (the last 2 quarters), the BVAR draws N complete 40-quarter factor paths, exact on every hard statement.
   - The draw is two-sided: a 2029 statement shapes 2027–28.
   - Spans without a stated cycle first get an IP path from the blocks, entropy-tilted if the scenario states regime odds.
   - Each statement reports a plausibility statistic: the shock size it requires. The stagflation set needs 0.81 sd per constraint.
   - This is the masking model: trained on measured data, run on stated values.
3. **qmodel, per path and quarter.**
   - s_q: assets, valuations and factor lags to q−1 from that path.
   - c_q: the path's earlier quarters.
   - x_q: the path's complete factor vector for quarter q.
   - It returns quarter-q asset returns and the new valuation state, which become s_{q+1}.
   - The qmodel never sees a missing input.
   - Valuation stays a qmodel state, not a BVAR variable, so realized multiples cannot leak into the factor fill.
   - Influence runs one way, macro to assets.
4. **Monthly UI.** Split each quarter with a kNN-matched historical quarter's monthly profile. Use the same historical quarter for every variable and asset, and shift so the quarterly totals stay exact.

## Key findings

1. The engine's CPI y/y > 4% excess (40.7% of quarters vs 32.6% in history) is a long-run level question, not a dynamics fix. Every generator whose mean is the 1973+ sample mean sits at 40-45%: plain BVAR 41.4%, BVAR with bounce-back and zero bound (B4) 45.3%, C' hybrid 43.4%. Anchoring long-run CPI at 2.5% gives 36.6%. One historical path is weak evidence: across the 4 independent 13-year windows of history the share runs 0-83% (median 13.5%).
2. The engine's real persistence defect: CPI y/y has no memory beyond a year. Its 4-quarter autocorrelation within 13-year paths is -0.05 vs +0.19 in history, and high-inflation spells last 4.5 vs 7.8 quarters. The BVAR restores both (+0.30, 7.2q); 24-month blocks restore them partly (+0.23, 5.9q).
3. History has a strong recovery bounce-back: each 1pp of IP drawdown from its 8-quarter peak predicts +0.48pp faster IP growth over the next 4 quarters (90% CI 0.30-0.82; 61 independent 4-quarter windows). The engine's block bootstrap has none (+0.02), the plain BVAR a fifth of it (-0.11), and B4 with a drawdown state -0.32.
4. Contraction durations, quarterly regime classifier, share longer than 6 quarters: history 0 of 7 (mean 4.1q, 13.6% of quarters in contraction), engine 18% (mean 4.3q, 13.8%), plain BVAR 55% (mean 8.6q, about 36% of quarters), B4 47%, B5 with regime 40%, C' 22% (mean 4.6q, 14.1%). With 7 episodes the engine's 18% is not rejected (P(0 of 7) = 0.24); the BVAR's 55% is (0.004). A linear VAR cannot reproduce short, sharp contractions. A state-matched block chain made it worse (35%, CPI>4 only 17%).
5. Under the constraint (CPI 6% for 2 years, IP -5%, oil +50%), the BVAR's 2-year changes match history's regression (24 independent windows) on spreads (Baa +1.05 vs +1.13 ± 0.18; GZ +1.30 vs +1.28 ± 0.34) and on services CPI. It is too hawkish on fed funds (+2.24 vs +1.00 ± 0.60) and too mild on unemployment (+1.15 vs +1.71 ± 0.25). The engine gets unemployment and the 10-year right but leaves GZ flat (-0.01) and fed funds +0.23.
6. The engine delivers a stated path only on average: for a stated -5%, the IP 2-year change runs -17% to +5% (p10/p90); for a stated +50%, oil runs -28% to +111%. The BVAR meets hard statements exactly on every path (to 5e-14) and reports a plausibility statistic (the stagflation set needs 0.81 sd of shocks per constraint).
7. Within-quarter co-movement against history's 90% intervals: the BVAR is inside on all five slopes tested. The engine is outside on two: policy rate on unemployment -0.43 vs -1.44 [-2.02, -0.88], because level scaling shrinks rate moves but not unemployment moves; and Okun's slope -0.064 vs -0.041 [-0.047, -0.034]. C' gives -1.43 and -0.033.
8. Partial oracle on the harness (realized CPI, IP and oil stated every quarter): neither engine turns a correct narrative into asset skill. Pooled 1995-2026 mean R^2 over the 10 score assets, transmission with GZ, 1Q/4Q/12Q/20Q (126/31/10/6 independent windows): full oracle +0.22 [+0.13, +0.31] / +0.12 / -0.08 / -0.16; engine +0.03 / -0.09 / -0.36 / -0.56; BVAR -0.17 / -0.31 / -0.15 / -0.19. Stating fed funds too changes nothing. The full oracle's short-horizon skill comes from realized financial variables that such a story does not pin down.
9. Factor fill-in (pooled R^2 of the unstated factor's change vs no change, 4Q/12Q). The BVAR is better on unemployment (+0.54/+0.61 vs +0.39/+0.49), the Baa spread (+0.44/+0.60 vs +0.28/+0.23), GZ at 12-20Q (+0.30 vs -0.28) and services CPI. The engine is better on fed funds (+0.47/+0.71 vs at best +0.24/+0.19). USD has no fill skill anywhere (all negative), so no change is best.
10. The engine's rate advantage is mostly its anchors. r* 0.75 and term premium 0.75 are fixed constants used in every backtest window, including the one trained to 1994, where they are post-2008 levels. Given the same anchors the BVAR matches the engine on the 10-year when fed funds is also stated (+0.46/+0.62 vs +0.36/+0.64). Censoring the policy rate at the zero bound removes the BVAR's 2009 misses of about -3.5pp; the 1Q fed funds fill improves from -2.9 to -1.0.
11. Long-horizon bands: with CPI, IP and oil stated, BVAR 80% coverage at 20Q is 0.80/0.76 in 1995-2004/2005-14 vs engine 0.64/0.63 and full oracle 0.64/0.70. Factor and parameter uncertainty in the joint model moves multi-year bands toward calibration. Coverage in 2015-26 is about 0.9 for all, too wide.
12. Open defects of every generator: unemployment falls below 3% in 12-20% of simulated quarters (engine 15%, plain BVAR 12%, C' 12%, C' with a 4.8 anchor 20%; history never below 3.4% since 1973). The plain BVAR also takes fed funds below zero in 14% of quarters, which the zero-bound censoring fixes.

## Open questions

1. Unemployment floor: every generator puts unemployment below 3% in 12-20% of simulated quarters (history never below 3.4% since 1973). Should the joint model use a bound, a log or logit transform, or a firmer natural-rate anchor? The 4.8 default made it worse (20%) because block-drawn IP growth (median 2.1% y/y) is higher than the VAR's own IP mean (1.2%).
2. Anchors are the scenario layer's decision. The engine's r* 0.75 and term premium 0.75 are fixed constants also used in the 1995-2014 backtests, where they are post-2008 levels. Should backtests use anchors knowable at the train end (for example r* of about 2% in 1994)? What long-run inflation should be the default anchor? It alone moves CPI>4 from 43% to 37%.
3. Cycle when the scenario is silent: C' uses the engine's 6-month blocks for IP, and 22% of contractions still exceed 6 quarters (history 0 of 7, P = 0.18). A block chain conditioned on IP drawdown (bounce-back aware) is untested; my state-matched chain on CPI y/y, IP y/y and the 12-month unemployment change made durations worse (35%).
4. Headline end-to-end test: a correct CPI/IP/oil narrative gives no asset skill under any dynamics engine (pooled 1Q R^2 +0.03 engine, -0.17 BVAR, vs +0.22 full oracle). Should the LLM translator be required to state financial conditions (credit spreads, GZ, yields, the valuation path)? Those are market prices, so stating them approaches an asset oracle; that decision belongs to the lead and the qmodel round.
5. At 1-4Q the transmission amplifies fill-in errors in rates and USD: the BVAR's asset R^2 is below drift despite better unemployment and spread fills. Should the qmodel take the conditional ensemble (path draws) rather than the conditional mean? Should variables with no fill skill (USD) be held at no change in point paths?
6. Not tested here: inequality statements (ranges) by accept/reject, the regime label as a statable variable (the regime layer's contraction flag), the monthly disaggregation of quarterly paths for the UI, and feedback from asset or valuation states to macro. GZ is partly endogenous, and the design keeps influence one-way, macro to assets.

## Files the role listed

Paths are relative to `scratchpad/sim_redesign/`. Only the files marked are in this archive; the rest (scripts, work outputs, data) stay in the research folder.

- `joint/jq.py`
- `joint/jq2.py`
- `joint/jq3.py`
- `joint/jq4.py`
- `joint/jengine.py`
- `joint/guard.py`
- `joint/j01_bvar_check.py`
- `joint/j02_persistence.py`
- `joint/j02_stats.py`
- `joint/j03_conditional.py`
- `joint/j04_partial_oracle.py`
- `joint/j05_b2_persistence.py`
- `joint/j06_fill_accuracy.py`
- `joint/j07_b3.py`
- `joint/j08_bounce.py`
- `joint/j09_b4.py`
- `joint/j10_b5_cycle.py`
- `joint/j11_hybrid.py`
- `joint/j12_b3ss.py`
- `joint/j13_hybrid_anchor.py`
- `joint/j14_pooled.py`
- `joint/work/j01_bvar_check.json`
- `joint/work/j02_persistence.json`
- `joint/work/j03_conditional.json`
- `joint/work/j04_metrics.csv`
- `joint/work/j04_paths.csv`
- `joint/work/j04.log`
- `joint/work/j06_fill_accuracy.csv`
- `joint/work/j06_fill_pooled.csv`
- `joint/work/j07.log`
- `joint/work/j08_bounce.json`
- `joint/work/j09.log`
- `joint/work/j09_b4.json`
- `joint/work/j10_b5_cycle.json`
- `joint/work/j11_hybrid.json`
- `joint/work/j12.log`
- `joint/work/j13_hybrid_anchor.json`
- `joint/work/j14_pooled.csv`
