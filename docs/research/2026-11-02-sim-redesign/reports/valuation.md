# Valuation block (research round, role `valuation`)

*What the role did:* valuation series with explicit timing; the negative results on pulls and level states.

Rendered from the role's final report in the research workflow's journal (run `wf_cc7fe8c3-afc`, result `valuation`). The text is unchanged except that local absolute paths are written as the spec writes them (`scratchpad/sim_redesign/...`, `scratchpad/...`). It is the report the critic and the designer read; the numbers are the role's own, and the critique ([`critic/critique.md`](../critic/critique.md)) and the spec ([`SIM_DESIGN.md`](../SIM_DESIGN.md)) say which of them survived.

## Summary

I built the valuation block, wrote the quarterly series with timing for qmodel, and tested the block on the harness (oracle macro, both protocols, 1/4/12/20 quarters). The short answer: the starting valuation level does not improve out-of-sample (OOS) returns after 1985, in any form I tried. The multiple therefore belongs in the scenario layer as an input, not in the transmission as a pull. I wrote nothing in MAIN, RWT or FWT: scripts ran with -B, or with -I for scripts that only read data, and no new .pyc or other file appeared in MAIN. FRED was called 13 times (7 searches, 6 series), with the key read from file and logged as api_key=REDACTED.

**For qmodel (logged in READY.md).** The data is in `valuation_quarterly.csv`: 1947Q1 to 2026Q4, 191 columns, described in `valuation_columns.json`.
- `x` is the level at the quarter's last month.
- `d_x` is the realized change during the quarter. It is a target only.
- `x__known` is the value known at the start of the quarter, after publication lags: prices and dividends 0 months, CPI 1, Shiller E and D 3, NIPA and Z.1 5. French book equity is used from July of each year.
- The 2026Q4 row holds today's state: CAPE 41.2 (97.5th percentile since 1948), CRSP P/D 97.4, trailing P/E 26.0, ECY 0.79%, BusEq P/B 12.2 (15.1 at the 2000Q1 peak), real-yield proxy 2.04%.
- The nonfinancial-corporate free-cash-flow (FCF) yield is 1.96%, the 45th percentile of its own history. By FCF, today is not expensive. By after-tax profit yield (2.71%, 2nd percentile) it is as expensive as 2000.

**Structure tested.**
- (a) Δlog P = Δlog fundamentals + Δlog multiple. This is exact for CRSP P/D (to 1e-15). The multiple carries 97% of quarterly price variance on P/D and 101% on CAPE.
- (b) Fundamentals equation, 1973–2026, HAC t in brackets:
  - Dividend growth: CPI 0.44 (2.3), own lag 0.30 (2.8); profit share and nominal GDP add nothing; R² 0.13.
  - CPI × 10-year average real EPS (the CAPE fundamental): CPI pass-through 0.97 (8.9), R² 0.56.
- (c) Multiple equation, Δlog P/D:
  - Δ real 10-year yield −0.027 per point (−3.2), Δ inflation expectations −0.116 (−3.3), Δ term premium −0.012 (−0.9), Δ GZ spread −0.135 (−7.8), Δ CPI y/y about 0. R² 0.41.
  - Error-correction coefficient κ: 0.0008 per quarter (t 0.07). By split end it is 0.051, 0.007 and 0.009, none significant.
  - Shiller log CAPE: κ 0.018 per quarter over 1881–2026 (half-life about 10 years, t 1.8). Over 1948–2026, bias-corrected, it is a unit root.
  - The level shifted: average CAPE 16.5 / 11.4 / 16.1 / 11.0 / 25.8 / 27.2 by era from 1881.
- (c) evaluated as inference of the multiple from the macro path alone (pooled rolling 1985–2026, 1/4/12/20Q): R² 0.30 / 0.21 / −0.35 / −0.27, corr 0.58 / 0.53 / 0.15 / −0.05. In 2005–14 corr is 0.81–0.87 at 4–12Q (the credit crisis), but the 1995–2004 bubble cannot be inferred (corr ≤ 0.06 at 12–20Q).

**Best specification (us_equity).** The P/D block with no pull (κ = 0), the multiple's drift set to zero (no historical re-rating), and the spec (b) and (c) regressors. OOS R² at 1 / 4 / 12 / 20Q:

| Model | 1995–2004 | 2005–14 | 2015–26 | Pooled rolling 1985–2026 |
|---|---|---|---|---|
| Best block | +0.33 / 0.00 / −0.32 / −0.63 | +0.01 / +0.11 / +0.22 / +0.50 | +0.64 / +0.42 / −0.46 / −1.31 | +0.31 / +0.20 / −0.05 / −0.04 |
| F | +0.18 / −0.15 / −0.34 / −0.37 | +0.49 / +0.56 / +0.50 / +0.58 | +0.65 / +0.56 / +0.04 / −0.44 | +0.35 / +0.30 / −0.09 / −0.11 |
| B0 | −0.15 / −0.42 / −0.62 / −0.68 | −0.34 / −0.19 / −0.31 / −0.36 | +0.38 / +0.35 / −0.18 / −0.73 | +0.02 / −0.02 / −0.41 / −0.44 |

- Pooled 90% intervals at 12/20Q: block [−0.46, 0.24] / [−0.32, 0.33], F [−0.44, 0.23] / [−0.48, 0.25]. Independent windows: 167 / 41 / 13 / 8 pooled; 3 and 2 per decade at 12 and 20Q.
- So the block matches F at long horizons, which is not significant; it loses at 1–4Q, and in 2005–14 at every horizon. Its value is a consistent decomposition into fundamentals and multiple, not accuracy.
- Its 80% band coverage at 12/20Q is 0.74 / 0.76 against F's 0.78 / 0.80. Bootstrap block length (1 to 12 quarters) moves it by 0.02 or less. In 1995–2004, 45% of 12Q outcomes land above the band.

**What fails**, in pooled 12/20Q R² unless noted:
- Error correction with κ clipped at ≥ 0: constant anchor −0.47 / −0.75, real-yield anchor −0.25 / −0.27, 20-year rolling anchor −0.32 / −0.56. CAPE versions are worse.
- Fixed pulls of 5% or 10% a year: −0.10 / −0.12 and −0.25 / −0.32.
- Starting-level corrections to F, against F's −0.09 / −0.11:
  - CAPE −0.51 / −0.78; CAPE vs real-yield anchor −0.61 / −0.63; CAPE vs its 20-year mean −0.33 / −0.63; ECY −0.46 / −0.55; P/D −0.73 / −0.73; profit yield −0.43 / −0.40; FCF yield −0.14 / −0.26.
  - They help 1995–2004 a little and hurt 2015–26 and 1985–94 badly.
- NIPA profits as fundamentals, even with realized profits: −0.66 / −0.66 / −1.25 / −1.30 at 1–20Q.
- Trailing P/E as fundamentals: −1.20 / −1.01 / −1.10 / −1.34 (2008–09 write-downs).
- Per-asset blocks for the other equities: mean over the 10 assets +0.10 / +0.03 / −0.16 / −0.18 against F's +0.23 / +0.22 / −0.01 / −0.03. They are worse for energy (no oil term), industrials, staples, REITs and nasdaq; about equal for financials and utilities; slightly better for healthcare.
- A pure quarterly regression (Fq) is much worse than F's monthly-fitted betas: us_equity −0.09 / −0.72 / −1.05 at 4/12/20Q. This matters for qmodel's design.

**Does the starting valuation explain 5- and 10-year returns OOS?** Only in rank, not in level, after 1980. I used Shiller real total returns, an expanding window from 1881, and 90% block-bootstrap intervals.
- 10-year, CAPE: +0.08 [−0.04, 0.43] over 1921–2016 (10 independent windows), +0.14 over 1950–2016 (7), −0.15 over 1980–2016 (4), −0.27 over 1995–2016 (3). Correlation is +0.43 to +0.91, but forecasts are 0.23–0.45 log points too low.
- 10-year, excess CAPE yield (ECY), the only one positive after 1950: +0.29 [0.05, 0.45] over 1950–2016 (7), +0.33 [−0.04, 0.61] over 1995–2016 (3).
- 10-year, P/D: −0.98 over 1980–2016. FCF yield (data from 1953): 0.00, corr −0.39 (4 windows).
- 5-year: CAPE −0.07 (21 windows), ECY +0.16 over 1950–2021 (15).
- In-sample 1881–2016: CAPE slope −0.61 (t −4.9, R² 0.26) on only 13 independent decades.

**CMA drift adjustment.** Its valuation part (minus the re-rating tailwind, plus a 30%, 15-year reversion to the median P/D) is the block's multiple drift plus κ in reduced form, so the two must never both be on.
- When the block or valuation state is active, use the engine's `valuation_state` CMA method (reversion 0).
- Default the multiple's drift to zero: pooled neutral-to-better, but split by decade (+0.50 vs +0.22 at 20Q in 2005–14, −1.31 vs −0.71 in 2015–26). Over 1973–2026 the P/D re-rated by +1.7% a year (39 → 97), and the default should not extend that.
- Any reversion should be an explicit scenario statement with half-life of 14 years or more, never a hidden default: every OOS test of a pull loses.
- Nominal fundamentals growth then comes from the macro path (CPI pass-through).
- The sector relative-drift shrink and gold's real-drift removal are not valuation and can stay in the CMA.

**Data and coverage per asset.**
- us_equity:
  - CRSP P/D from 1948Q1, exact identity.
  - Shiller CAPE, P/E, P/D and ECY from 1881, with actual S&P EPS to 2026-06. MAIN's NIPA-spliced `s_cape` reads 4–12% too high in 2024–26.
  - FCF and profit yields from 1953Q1. FCF is negative in 29 quarters between 1970 and 2001, so use the yield, not log P/FCF.
- nasdaq: there is no free NASDAQ-100 P/E or FCF.
  - BusEq P/D breaks in 2003–05 (843 in 1999, 58 in 2004, 163 in 2007).
  - BusEq P/B from the French annual book-to-market table (from 1948, July timing) is the only continuous tech valuation. BusEq price moves track NASDAQ with quarterly correlation 0.95.
  - Even so, the block built on it loses to F (−0.33 / −0.36 vs −0.21 / −0.28 at 12/20Q).
- French industries: own P/D (lumpy dividends) and annual P/B.
- REITs: Nareit P/D from 1972. Its fast in-sample reversion (0.18 per quarter) does not hold OOS.
- Gold: no cash-flow valuation; keep F.

## Key findings

1. Valuation series are delivered for qmodel in valuation_quarterly.csv (1947Q1-2026Q4, 191 columns, logged in READY.md). Each series comes in three timings: x (end of quarter), d_x (realized change, target only) and x__known (known at the start of the quarter, with publication lags). The 2026Q4 row is today's state: CAPE 41.2 (97.5th percentile since 1948), P/D 97.4, ECY 0.79%, BusEq P/B 12.2 (2000 peak 15.1). The nonfinancial-corporate FCF yield is 1.96%, its 45th percentile: by FCF, today is not expensive.
2. The multiple carries 97% (P/D) to 101% (CAPE) of quarterly us_equity price variance. The multiple equation, 1973-2026, HAC t: Δ real y10 -0.027 (-3.2), Δ inflation expectations -0.116 (-3.3), Δ term premium -0.012 (-0.9), Δ GZ -0.135 (-7.8); R2 0.41. The dividend equation has R2 0.13; the CAPE fundamental (CPI x E10) has R2 0.56, almost all CPI pass-through.
3. There is no usable error correction after 1973. κ is 0.0008 per quarter (t 0.07) on the full sample and insignificant at every split end (0.051, 0.007, 0.009). Over 1881-2026 log CAPE has κ 0.018 (half-life about 10 years, t 1.8); over 1948-2026, bias-corrected, it is a unit root. The CAPE level shifted around 1990 (era means 16.5 / 11.4 / 16.1 / 11.0 / 25.8 / 27.2).
4. Every pull toward an anchor lowers pooled 12/20Q R2, with κ clipped at >= 0. For the P/D block: constant anchor -0.47 / -0.75, real-yield anchor -0.25 / -0.27, 20-year rolling anchor -0.32 / -0.56, fixed 5% a year -0.10 / -0.12, fixed 10% a year -0.25 / -0.32. With no pull it is -0.06 / -0.09.
5. Seven starting-level states added to F all lower pooled 12/20Q R2 for us_equity: CAPE -0.51 / -0.78, ECY -0.46 / -0.55, P/D -0.73 / -0.73, FCF yield -0.14 / -0.26, against F's -0.09 / -0.11. They help 1995-2004 slightly and hurt 2015-26 and 1985-94 badly.
6. Best specification: the P/D block with no pull and zero re-rating drift. us_equity pooled 1/4/12/20Q: +0.31 / +0.20 / -0.05 / -0.04, against F +0.35 / +0.30 / -0.09 / -0.11 and B0 +0.02 / -0.02 / -0.41 / -0.44. The 90% intervals overlap (12Q [-0.46, 0.24] vs [-0.44, 0.23]), and 2005-14 is worse than F at every horizon. It matches F; it does not beat it.
7. Per-asset blocks lose to F: the 10-asset mean is +0.10 / +0.03 / -0.16 / -0.18 against +0.23 / +0.22 / -0.01 / -0.03. The losses are in energy, industrials, staples, REITs and nasdaq. A pure quarterly regression (Fq) is far worse than F's monthly-fitted betas (us_equity 4/12/20Q -0.09 / -0.72 / -1.05), so the quarterly model should borrow monthly-estimated betas.
8. Inferring the multiple from the macro path alone: R2 0.30 / 0.21 / -0.35 / -0.27 (1/4/12/20Q), corr 0.58 / 0.53 / 0.15 / -0.05. Correlation is 0.81-0.87 in 2005-14 and 2015-26 at 4-12Q, but the 1995-2004 bubble cannot be inferred (corr &lt;= 0.06 at 12-20Q). The multiple should therefore be a maskable scenario input (d_us_equity_lpd or d_us_equity_lm_cape), or left to the band.
9. The long-horizon band is not fixed. The block's 80% coverage at 12/20Q is 0.74 / 0.76 (F 0.78 / 0.80), and bootstrap block length from 1 to 12 quarters moves it by 0.02 or less. In 1995-2004, 45% of 12Q outcomes fall above the band.
10. Starting valuation and 5- and 10-year returns OOS (expanding window from 1881, block-bootstrap 90% intervals): CAPE ranks decades correctly (corr +0.43 to +0.91) but misses the level after 1980. 10-year R2: +0.08 over 1921-2016 (10 independent windows), -0.15 over 1980-2016 (4), -0.27 over 1995-2016 (3). Only ECY is positive after 1950: +0.29 [0.05, 0.45] (7 windows) and +0.33 over 1995-2016 (3). FCF yield is about 0 with negative correlation; P/D is -0.98 after 1980.
11. CMA: its valuation part (remove the re-rating, then a 15-year reversion to the median P/D) is the block's drift plus κ in reduced form. Never run both: use the engine's valuation_state CMA method with no reversion. Default the multiple's drift to zero (P/D re-rated +1.7% a year from 1973 to 2026), with mixed decade evidence. Any reversion should be an explicit scenario statement, since every OOS test of a pull loses. Sector drift shrink and gold stay in the CMA.
12. Data limits. NASDAQ-100 has no free P/E or FCF. BusEq P/D breaks in 2003-05 (Microsoft's special dividend), so BusEq P/B is the only continuous tech valuation, and the block built on it still loses to F. Trailing P/E is unusable (2008-09 write-downs; R2 about -1.1). NIPA profits as fundamentals fail even when the realized values are used (-1.25 / -1.30 at 12/20Q).

## Open questions

1. Design decision for the lead: should valuation enter the quarterly model only as a known state plus a maskable scenario input (the realized multiple change), with no pull in the transmission? The evidence supports this. If a 10-year scenario needs a default valuation path, the only OOS-positive candidate after 1950 is the ECY-based level, which rests on 3-7 independent decades and is too pessimistic post-1990.
2. The engine's forward scenarios and one-year forecast use the CMA 'valuation_adjusted' reversion term. Every OOS pull I tested loses after 1985. Should that term be switched off (the valuation_state method, reversion 0) and replaced by an explicit, stated valuation view in each scenario?
3. For qmodel: a pure quarterly regression on about 170 quarters is far worse than the monthly-fitted F at 4-20Q (us_equity -0.09 / -0.72 / -1.05 at 4/12/20Q). Can the quarterly model borrow monthly-estimated betas, or monthly within-quarter features, instead of refitting at quarterly frequency?
4. The 12-20Q band stays under-dispersed in bubble decades (1995-2004: 45% of 12Q outcomes above the band). Longer bootstrap blocks do not help. Fat-tailed or regime-dependent multiple volatility, or scenario-stated valuation ranges, are open options.
5. The headline end-to-end test: for historical episodes, valuation statements in a narrative are realized valuation and are circular. Should the narrative translator be barred from emitting the multiple (keep it masked or inferred) when scoring, and allowed to emit it only for forward scenarios?
6. Which valuation measure should translated text map to? CAPE and P/D say today is extreme (97.5th-98.4th percentile); the FCF yield says normal (45th). I recommend log P/D (exact identity with the CRSP price) or log CAPE as the multiple, with FCF shown as context only. Does the lead agree?
7. NASDAQ/QQQ: no allowed source has NASDAQ-100 earnings or FCF. BusEq P/B is the best free state, but the block built on it loses to F. Keep nasdaq on F plus BusEq P/B as a state only, or look for another source?
8. Two timing assumptions need confirmation. The GZ level is treated as known with lag 0 (reference month, as in the factor round), although the Fed posts it with a short delay. 2026Q3 NIPA profits are unpublished, so the block uses its no-NIPA fundamentals equation for that quarter.

## Files the role listed

Paths are relative to `scratchpad/sim_redesign/`. Only the files marked are in this archive; the rest (scripts, work outputs, data) stay in the research folder.

- [`valuation/READY.md`](../valuation/READY.md) (in this archive)
- `valuation/valuation_quarterly.csv`
- `valuation/valuation_columns.json`
- `valuation/valuation_monthly_ref.csv`
- `valuation/valuation_monthly_known.csv`
- `valuation/valuation_results.json`
- `valuation/valblock.py`
- `valuation/vmodel.py`
- `valuation/fredclient.py`
- `valuation/work/v05_us_equity_grid.py`
- `valuation/work/metrics_v05.csv`
- `valuation/work/metrics_v21_clip.csv`
- `valuation/work/v10_state_correction.py`
- `valuation/work/metrics_v10.csv`
- `valuation/work/v11_multiple_inference.py`
- `valuation/work/v11_summary.csv`
- `valuation/work/v13_all_equity.py`
- `valuation/work/metrics_v13.csv`
- `valuation/work/v06_longhorizon.py`
- `valuation/work/v06_longhorizon.json`
- `valuation/work/v18_longhorizon_ci.json`
- `valuation/work/v14_long_kappa.json`
- `valuation/work/v17_coefs.json`
- `valuation/work/metrics_v16.csv`
- `valuation/work/v09_fcf.py`
- `valuation/work/fcf_quarterly_raw.csv`
- `valuation/downloads/fred_fcf_20261101/requests.log`
