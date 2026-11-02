# Critique of the sim_redesign research round (role: critic)

Everything here is read-only on MAIN, RWT and FWT. Every script ran as `cd MAIN && MAIN/.venv/bin/python -B <script>` (or `-I -B` for scripts that only read files), from this folder. Nothing was downloaded. Scripts are `c01..c25*.py` in this folder; their outputs are in `work/c*.log`.

Verdict codes: **CONFIRMED** (re-run or checked in code, matches), **PARTLY** (numbers right, interpretation or scope wrong), **REFUTED** (the numbers do not support the claim), **UNVERIFIED** (not re-run; reason given).

---

## 0. The ten findings that matter for the design

1. **The oracle-macro transmission has no macro skill. All of it is credit-spread skill.** I ablated the harness's F model; c10, c11 and c18 match the harness's stored F/B0 metrics exactly. Rolling protocol, pooled 1985–2026, mean R² over the 10 score assets at 1/4/12/20Q (167/41/13/7–8 independent windows):

   | model (realized same-month inputs) | 1Q | 4Q | 12Q | 20Q |
   |---|---|---|---|---|
   | F (8 core + Baa spread + GZ + TP×core4; energy + metals/oil) | +0.226 | +0.223 | −0.007 | −0.030 |
   | F without GZ | +0.128 | +0.118 | −0.119 | −0.133 |
   | F without GZ and without the Baa spread | −0.003 | −0.016 | −0.258 | −0.285 |
   | ip, CPI goods, CPI services, ff, unemployment only | −0.081 | −0.145 | −0.373 | −0.421 |
   | the same + oil + 10y + dollar | −0.036 | −0.035 | −0.258 | −0.297 |
   | **GZ spread alone** | **+0.165** | **+0.149** | **+0.110** | **+0.108** |

   - Paired block bootstrap. F − F without GZ is +0.10 at every horizon, and every 90% interval excludes 0. F − (no credit at all) is +0.23 to +0.26.
   - F − GZ-only is +0.061 [+0.004, +0.120] at 1Q, +0.075 [−0.043, +0.155] at 4Q, −0.117 [−0.305, +0.012] at 12Q and −0.137 [−0.292, −0.013] at 20Q. The eight macro factors add a little at 1Q and **subtract** from 12Q on.
   - In 2005–14 (split), F without GZ scores −0.075/−0.012/−0.087/−0.128.
   - us_equity, pooled: GZ alone +0.30/+0.28/+0.17/+0.16, against F +0.35/+0.30/−0.09/−0.11 and the valuation block +0.31/+0.20/−0.05/−0.04.
   - GZ is a bond-market risk premium (adj R² 0.40 on same-quarter asset returns, re-run in c20). "Oracle macro" is therefore an oracle *credit market*, not an oracle macro path.
   - This also explains why the joint and qmodel agents found that a narrative giving CPI, IP, oil and policy cannot produce asset skill under any dynamics engine: those inputs carry none, even when they are known exactly.

2. **"The misses are valuation swings" is an accounting identity, not evidence.** Δlog P = Δlog D + Δlog(P/D), and dividends are smooth: var(Δlog D)/var(Δlog P) = 0.074 quarterly (c14). So *any* forecast that does not track prices has misses that "are" multiple changes.
   - **Placebo in the episode test** (c04). A constant forecast gives corr(miss, ΔP/D)² = **0.854** across the 11 episodes. That is higher than every model the scorer reports (E-n 0.66–0.78, oracles 0.38–0.47).
   - **Placebo on the harness** (c12, CAPE split). A constant forecast gives err_r2_m 0.83–1.00, at least B0's value in every decade × horizon cell.
   - The multiple's share of variance is about 1 at 1Q as well, so it cannot explain why long horizons are worse. The decay with horizon comes from the macro betas: pure macro falls from −0.08 to −0.42 while GZ-only stays near +0.11.
   - The informative statistic is the correlation between forecast and outcome across episodes. For the narrative runs it is −0.08 to +0.19 (sd of the medians 0.07–0.11, against 0.31 realized).
3. **The narrative test was not blind, and it was more contaminated than the reports say.**
   - **Shared context.** The translator's task prompt was `${CTX}` + QSCHEMA + ROLE (`scratchpad/sim_research_tail.js`). Assuming CTX is the same shared header the critic received, it contained three things: the model start 1973-02 (so every prompt's line "T0−XyYm 至 T0 共 N 个月" gives T0 exactly); the user's 2022–23 example (rate hikes, oil war, reopening); and the lead's hypothesis naming 1995–2000, 2000–02 and 2008–09.
   - **Few-shot examples.** They name events with Y0 offsets, for example "Y0−14 年海湾战争".
   - **schema.json, which the translator loaded** (`translator/qtable.py`). Its templates G01–G35, behind the `template_sign` entries, carry quarter-dated realized footprints of the test episodes: "2008Q4：GZ +6.8 sd" (G04), "2020Q1 GZ +5.5 sd" (G16), GZ in 2009Q1–Q3 (G08), and the realized equity earnings and multiple legs of 1995–2000 and 2023–26 (G06). The schema also lists 354 year tokens in scenario_rules.
   - **Direct evidence.** The number-free E08 n3 table's d_gz path is [+2.0, −1.2, −0.3, −0.2, −0.1, 0.0], against realized [+2.33, −1.49, −0.27, −0.57, −0.16, −0.15]: **corr 0.99, MAE 0.20pp** against a realized sd of 1.16. For E05 n3, corr 0.78. This is the variable that carries all the transmission skill (point 1).
4. **"A perfect translation would not help either" is overstated** (c21, c22, c23).
   - The pooled oracle loss (MSE skill −0.41 [−0.77, −0.02]) depends on E03, which trains on 80 DB months. Without E03, E-oracle is −0.27 [−0.50, +0.28], and the F, VB and Q oracles are +0.07, +0.11 and +0.03.
   - Excluding also the two equity busts (E05, E12; n = 8, a post-hoc subset): the E-oracle has MSE skill **+0.27 [+0.17, +0.45]** at the window end and +0.26 at 4Q. The narrative runs stay near 0 (E-n1 −0.01, E-n2 +0.01, E-n3 +0.21), and E-cond is +0.17.
   - So in ordinary episodes the oracle path does help, and the translation and engine layers lose that gain. In the two busts the transmission fails outright. "Model loss dominates" holds only for the pooled sample, which the busts dominate.
5. **Look-ahead in every engine run.**
   - `analysis/backtest.run_window` and the scorer's `s02_engine.py` call `simulate()` without `reversion`. Every window, including the one trained to 1994, therefore uses `DEFAULT_REVERSION` from `engine/simulate.py`: r* 0.75, term premium 0.75, spread level 2.0, unemployment 4.8. These are post-2008 levels chosen with full-sample knowledge.
   - The translator's r_star overrides replace the r* part during the narrative phases only. The key is present in all 188 phases of the 36 specs; it is numeric (−3 to +3) in 131 phases, and every spec has at least one numeric value.
6. **Why the narrative medians had the wrong sign in the busts: mostly drift, not rate cuts** (c16, B0 = MAIN's AssetFactorModel).
   - **E05.** The intercept over the window contributes +0.198, the factors +0.078 (y10 +0.055, ff +0.023, unemployment +0.028, spread −0.069), against realized −0.468.
   - **E12.** Intercept +0.271, factors +0.133.
   - **E10, with hikes.** Intercept +0.168, factors −0.082.
   - **The unemployment beta is positive.** Rising unemployment adds to predicted returns, the anticipation effect.
7. **The qmodel's G-over-L0 margin is not out-of-sample evidence.**
   - G − L0 at 1Q is +0.026/−0.034/+0.051, about the same as the gain of the schema's tier-1 fields over F (L1 − L1-Fp = +0.023/+0.003/+0.054).
   - Those fields were selected with the same three test decades: the schema's own tier rule says so.
   - On top of that, 16 variants were scored on the test decades (`qmodel/work/q03_variants.log`) before L2-T1p-reg was chosen.
   - F itself used 1995–2004 as its validation period in the factor round (FACTORS_DESIGN §2).
   - The conclusion "G ≈ L0, and the NN loses" stands. "G beats L0 in 1995–2004 and 2015–26" does not.
8. **Data errors confirmed.**
   - **MAIN's spliced `s_cape` is 4.0–12.5% too high since 2024.** It reads 43.7 against Shiller's 40.6 in 2026-09. It feeds the production translator's state table (`llm/translator.py`).
   - **P/Ê is distorted by write-downs.** In E05 the P/Ê "multiple" rises +1.005 while price falls 0.468; in E12, +0.256 against −0.383.
   - **About 30% of the P/D re-rating is a falling payout ratio.** Over 1973→2025 (Shiller annual means), Δlog P/D 0.88 = Δlog P/E 0.63 + 0.26 from the payout ratio. The valuation report's "+1.7%/yr re-rating" and its "zero P/D drift" default mix valuation with buybacks.
   - **x_oil_cause conventions are opposite.** qdata uses 1 = supply; the translator used 0 = supply.
   - **The newer Shiller workbook ends E and D at 2026-06.** Trailing E rises 23% in two quarters (240.6 → 295.4). Whether those values are reported or estimated cannot be told from the file.
9. **Small samples.** What can and cannot be concluded is set out in section 5. In short, 12Q and 20Q decade cells are 2–3 independent windows each; the episode test has 11 units, with 7 reaching 8Q and 2 busts. 1 of the 30 narrative-versus-no-narrative intervals favours the narrative and 3 disfavour it, about what chance gives at 90%.
10. **The scorer's audit is incomplete.**
    - It omits 16 `f_sentiment` views: E06 n1/n2/n3, E08 n3, E11 and E12. Their translator-chosen magnitudes are "取历史下沿附近", and sentiment is endogenous (adj R² 0.24). It also omits 3 `f_profit_share` views (E09).
    - It misses the schema.json and CTX channels and the E08 n3 GZ recall.
    - The fan_cond row it calls the only clearly positive one takes realized means of y10, the Baa spread and the dollar, which are market prices.

---

## 1. What I re-ran

| script | what | output |
|---|---|---|
| c01_scoreboard.py | the scorer's scoreboard, paired differences, MSE skill, PIT and coverage, recomputed from `scorer/out/score_rows.csv` | work/c01.log |
| c02_narrative_scan.py | narratives: years, digits in n3, asset/corporate/sector words | work/c02.log |
| c03_spec_audit.py, c03b.py | 36 specs, 36 tables: banned views, overrides, r_star, endogenous views | work/c03*.log |
| c04_valmiss.py | the scorer's valuation-miss claim against a constant-forecast placebo | work/c04.log |
| c05_panel_cols.py | which harness columns stay visible after masking | — |
| c10_harness_gz.py, c11_harness_macro.py, c18_gz_paired.py | F ablations (GZ, credit, pure macro, GZ alone), split and rolling, paired block bootstrap | work/c10.log, c11.log, c18.log |
| c12_anatomy_placebo.py | the harness's variance anatomy against a constant-forecast placebo | work/c12.log |
| c13_qmodel_rows.py | qmodel 1Q scores recomputed from `q05_split_rows.csv.gz`, G − L0 bootstrap | work/c13.log |
| c14_valuation.py, hac.py | multiple share, P/D and CAPE error-correction κ (HAC), today's percentiles, MAIN s_cape against Shiller | work/c14.log |
| c15_joint_hist.py | history CPI > 4% share, 13-year windows, autocorrelation | work/c15.log |
| c16_contrib.py | B0 factor contributions to us_equity in E02/E05/E10/E12 and in 1995–2004 | work/c16.log |
| c17_n2_1q.py | where the n2 1Q gain sits, by asset; multiple-testing count | work/c17.log |
| c19_cape_today.py | MAIN s_cape against Shiller at the panel end | — |
| c20_endog.py | endogeneity (adj R² of factor changes on same-quarter asset returns) | work/c20.log |
| c21_endc.py, c22_mse_exE03.py, c23_narr_subsets.py | common-end comparisons; robustness to E03 and to the busts | work/c21–c23.log |
| c24_selfcheck.py | `simharness.selfcheck()` and `lead_check()` | work/c24_head.log |
| c25_payout.py | P/D change split into P/E and the payout ratio | work/c25.log |

---

## 2. Verdicts by report

### 2.1 harness

| claim | verdict | evidence |
|---|---|---|
| Harness validated (B0 = AssetFactorModel, residual draw identical, masking, drift R² = 0, lead's numbers) | **CONFIRMED** | `selfcheck()`: b0 4.72e-16, resid 1.80e-15, draw identical, mask ok, v_lpd 1.78e-15, drift 6.66e-16. `lead_check()` means match (1995–2004 monthly −0.024). My independent `conditional_paths` runs reproduce the stored B0/F mean10 tables to 3 decimals. |
| R² does not improve beyond one year; it falls | **CONFIRMED** (and explained) | F rolling pooled +0.226/+0.223/−0.007/−0.030. The decay belongs to the macro betas: pure macro −0.08 → −0.42, GZ alone +0.17 → +0.11. |
| Multi-year statistics rest on very few windows | **CONFIRMED** | 3 and 2 per decade at 12Q and 20Q; 13 and 7–8 pooled. |
| 1995–2004: the macro factors pointed the wrong way | **CONFIRMED**, refined | The largest B0 contribution in 12Q windows is the interpolated labor share (β −2.2 trained to 1995-06; contribution sd 0.071, corr −0.69 with the realized return), then y10 (sd 0.040, corr −0.31). Every contribution is small next to the realized sd of 0.425. |
| 2015–26: a level bias | **UNVERIFIED** | Not re-run. Caveat: part of the P/D "re-rating" is the falling payout ratio (finding 8). |
| The lead's hypothesis holds: long-horizon misses are valuation swings | **REFUTED as evidence** | A constant-forecast placebo gives err_r2_m 0.83–1.00 (CAPE), at least B0's in every cell. share_m ≈ 1 at 1Q too, so it cannot explain the horizon decay. r2_orc_m is an identity, not a bound anyone could reach. |
| F's 2005–14 gain comes from pricing the multiple; GZ partly endogenous | **CONFIRMED and stronger** | Without GZ, F's 2005–14 R² is −0.075/−0.012/−0.087/−0.128. Pooled F − F_noGZ is +0.10 at every horizon (90% interval > 0). It is entirely, not partly, GZ. |
| Engine band calibrated at 1Q, too narrow at 12–20Q in 1995–2014 | **PARTLY** | Correct as computed, but 12–20Q coverage rests on 2–3 independent windows per decade, so 0.53–0.65 cannot be told apart from 0.80. |
| Trailing P/E unusable as a fundamentals split | **CONFIRMED** | E05 P/Ê leg +1.005 while price falls 0.468; E12 +0.256 against −0.383 (scorer/out/valuation_misses.csv). |
| Naive CAPE pull does not work | **CONFIRMED** (consistent) | The valuation report's wider tests agree; my error-correction κ is insignificant after 1973. |
| Note on F's decades | caveat | F's components were selected in the factor round with 1995–2004 as the validation period (FACTORS_DESIGN §2: GZ "selected in iteration 1"; gold from 1982 "chosen on validation"). F's 1995–2004 cells are therefore not clean out-of-sample results. |
| Note on visible columns | caveat | The e_* structural shocks (full-sample SVAR), f_ebp (full-sample default model) and f_tp10 (ACM, re-estimated on the full sample) are visible to any plug-in. B0 and F do not use the e_* shocks or EBP. F's TP × core4 term is zero in 1995–2020. |

### 2.2 layers

| claim | verdict | evidence |
|---|---|---|
| Horizon hypothesis supported in 2 of 3 decades | **REFUTED as stated** (weak at best) | Their own l02_evidence.json: corr(annual oracle error, same-year ΔlogPD) +0.57 / −0.57 / +0.89 on n = 10/10/11 (the ±0.57 cells give p ≈ 0.08); pooled 0.056 on n = 31. The sign flips in 2005–14. |
| Valuation is the missing input, and the model cannot infer it | **PARTLY** | The "upper bound" adds ΔP/D, about 97% of price variance, to explain price: it is circular. "Cannot infer" holds for 1995–2004 only. In 2005–14 and 2015–26 the credit spread infers much of it (finding 1; valuation corr 0.81–0.87). |
| Valuation level is a slow leading state only | **CONFIRMED** (consistent) | κ ≈ 0 after 1973 (c14). Long-horizon results rest on 7–14 independent windows. |
| Anticipation: ip and unrate priced about a quarter ahead; S+Lc | **UNVERIFIED / PLAUSIBLE** | The two variables were chosen from full-sample lead–lag evidence, and the pooled 90% interval touches 0. Supporting fact (c16): B0's unemployment beta is positive, so rising unemployment adds +0.028 to E05's oracle prediction. |
| KEY_STATES verdict | **UNVERIFIED** | Not re-run. |
| Endogeneity table | **CONFIRMED** | c20 (n = 214): spread 0.32, GZ 0.40, y10 0.27, oil 0.45, P/D 0.93, sentiment 0.24 (layers 0.27), EPU 0.19 (0.29), dollar 0.21, ip 0.09, unemployment 0.01. |
| Contract R5: credit stress "only when the narrative has a crisis" | **design flaw** | R5 is the input that carries all of the transmission's skill (finding 1). Making it optional removes the one channel that works. |
| "The oracle-macro arm is the leakage-free reference" | **PARTLY** | It is free of LLM leakage, but it feeds realized credit-market prices (GZ, Baa) and the 10y; for the bond and credit these are asset prices. |
| The LLM is a leakage channel | **CONFIRMED, and larger** | See 2.8 and finding 3. |

### 2.3 writer

| claim | verdict | evidence |
|---|---|---|
| No years in the narratives; n3 has no digits | **CONFIRMED** | c02: no 19xx/20xx tokens in any of the 36 files; 0 digits in every n3. |
| None names a person, country or event | **PARTLY** | No proper names, but the events are unique descriptions: E05 (investment bank rescued by takeover, investment bank bankrupt, insurer bailed out, GSEs taken over, two automakers in managed bankruptcy, stress tests); E10 (mid-size banks fail within days in a run, debt-ceiling deal); E12 (domestic terror attack, accounting-fraud bankruptcies); E11 (savings-and-loan failures). 12 of 12 were identified. |
| Nothing about asset prices, valuation, earnings or sectors | **PARTLY REFUTED** | No prices or valuations. But E12 names a sector ("企业（尤其是信息技术和通信设备领域）此前严重过度投资") and earnings fraud ("几家大型企业相继曝出财务造假并破产"); E02 the IT investment boom; E05, E10 and E11 financial-firm failures. These are statements about sector and corporate outcomes. |
| prompt.md anchored at T0, dates tokenized | **CONFIRMED**, with leaks | T0 = 1973-02 + offset, read off each sample line (E01 "T0−31y4m 至 T0 共 377 个" gives 2004-06). The few-shot examples name dated events ("Y0−14 年海湾战争", "Y0−31 至 Y0−30 年禁运"); "Y0+22年美元" gives 2026. |
| E06 leave-window-out; E03 has 80 DB months | **CONFIRMED** | E06 sample line "T0−7m 至 T0+53y 共 623"; E03 "T0−6y7m 至 T0 共 80". |
| N2 = N1 + financial conditions | **CONFIRMED**; caveat | N2 states realized 10y, Baa spread and dollar paths, which are market prices. The treasury result follows from the 10y mechanically. N2's only significant gain (1Q) sits in financials (−0.023 [−0.034, −0.012]) and the treasury (−0.015), the assets those prices drive (c17). |
| realized_macro magnitudes | **UNVERIFIED** | Not re-run line by line. The history CPI > 4% share (c15) matches the panel. |

### 2.4 valuation

| claim | verdict | evidence |
|---|---|---|
| Today's state: CAPE 41.2 at the 97.5th percentile, P/D 97.4, FCF yield 1.96% at the 45th, profit yield 2.71% at the 2nd | **CONFIRMED** | c14: CAPE 41.2 at the 97.8th percentile since 1948Q1; FCF 1.96% at the 44.3rd (1952Q3–2026Q2, n = 296, median 2.08); profit yield 2.71% at the 3.0th; 29 negative FCF quarters, all in 1970–2001. |
| "By FCF, today is not expensive" | **NOT SUPPORTED** | The FCF yield is a NIPA/Z.1 proxy for all nonfinancial corporations: domestic profits + depreciation − capex, over the market value of all NFC equity, including estimated private firms. It is not an SPY/QQQ FCF yield. The two NIPA yields disagree by 41 percentile points. The conclusion depends on the proxy, and the user asked about SPY/QQQ. |
| Multiple carries 97% (P/D) / 101% (CAPE) of quarterly price variance | **CONFIRMED**, mechanical | 0.973 / 1.011 on n = 214. var(Δlog D)/var(Δlog P) = 0.074: it follows from smooth dividends. |
| Multiple equation (real y10, inflation expectations, TP, GZ) | **UNVERIFIED**; mislabelled | Not re-run. ΔGZ is the dominant regressor (t −7.8), so "inferring the multiple from the macro path alone" means inferring it mostly from the credit spread. |
| No usable error correction after 1973 | **CONFIRMED** | My simple error-correction model on log P/D, κ per quarter (HAC t): to 1994 +0.065 (1.72), to 2004 +0.012 (0.77), to 2014 +0.017 (1.22), full sample +0.012 (0.97). Their spec adds regressors: 0.0008 (0.07). Shiller annual log CAPE: 1881–2026 +0.078/yr (2.20), about 0.019/q, matching their 0.018 (t 1.8); 1948–2026 +0.059 (1.53). |
| Every pull and every starting-level state lowers R² | **UNVERIFIED** (consistent) | Not re-run; consistent with κ ≈ 0. These are negative results, so the risk of selection is low. |
| Best spec (P/D block, no pull) matches F | **UNVERIFIED** for the block; F confirmed | F's us_equity pooled +0.348/+0.297/−0.092/−0.111 reproduced. GZ alone beats both at 12–20Q (+0.17/+0.16). The block's structure adds nothing beyond the credit spread at long horizons. The best spec was chosen from a grid scored on the same decades. |
| MAIN s_cape reads 4–12% too high in 2024–26 | **CONFIRMED** | MAIN/Shiller ratio: 1.039 (2023H2), 1.070 (2024), 1.083 (2025), 1.091 (2026), range 1.040–1.125. 2026-09: 43.72 against 40.58. Exact before 2023-07 (max diff 0.0005). |
| Shiller E and D "actual" to 2026-06 | **UNVERIFIED** | The file ends E/D at 2026-06 with no notes on estimates. Trailing E goes 240.6 (2025-12) → 295.4 (2026-06). |
| 5- and 10-year out-of-sample results | **UNVERIFIED**; underpowered | 3–10 independent windows per cell. ECY "+0.29 [0.05, 0.45] on 7 windows" comes from a bootstrap over 7 decade-blocks, which is not a reliable interval. |
| CMA: zero P/D drift by default | **PARTLY** | 1973→2025 Δlog P/D 0.88 = Δlog P/E 0.63 + 0.26 from the payout ratio (c25). Zero P/D drift implies a constant payout; it is not zero re-rating. |
| BusEq P/D break and per-asset limits | **CONFIRMED** (documented) | — |

### 2.5 joint

| claim | verdict | evidence |
|---|---|---|
| History: CPI y/y > 4% in 32.6% of quarters | **CONFIRMED** | 0.326 quarterly, 0.332 monthly (n = 215 / 645). |
| "Across the 4 independent 13-year windows the share runs 0–83% (median 13.5%)" | **PARTLY** | The range and median come from overlapping windows (step 4q). Non-overlapping windows: 83 / 27 / 10 / 15% (median 21%). |
| CPI persistence (4q autocorrelation +0.19 in history against −0.05 in the engine) | **PARTLY** | The direction holds. The history value depends on the definition: non-overlapping 13-year windows give 0.56 / 0.49 / −0.24 / 0.46 (mean 0.32). The engine value was not re-run. |
| Bounce-back +0.48 per pp, 61 independent windows | **UNVERIFIED** | The sample is 1960Q2–2026Q3 (n = 245, j08), not 1973+. |
| Contractions: 0 of 7 longer than 6q; P values | **arithmetic CONFIRMED** | 0.82^7 = 0.25 and 0.45^7 = 0.004. The classifier was not re-run. |
| The BVAR meets hard constraints exactly | **true by construction** | Conditioning by projection. |
| Partial oracle: no asset skill from CPI, IP and oil under any engine | **CONFIRMED** (consistent, and explained) | c11: realized CPI, IP, ff, unemployment and oil carry no skill (pooled −0.08 to −0.42). No dynamics engine can create skill from inputs that have none. |
| The engine's rate advantage comes from its anchors, which are post-2008 constants used in every backtest window | **CONFIRMED** in code | `backtest.run_window` calls `simulate()` without `reversion`, so `DEFAULT_REVERSION` (r* 0.75, TP 0.75, spread 2.0, unemployment 4.8) applies in every window. The scorer's engine runs do the same. |
| Recommendation C′ | design (no verdict) | Its fill-in claims for rates and spreads matter less than its central point: whatever fills the credit spread decides the asset outcome. |

### 2.6 schema

| claim | verdict | evidence |
|---|---|---|
| Policy reaction functions, R² 0.40–0.54 | **CONFIRMED** | The qmodel's independent implementation reproduces u_ff R² 0.54 / 0.51 / 0.46 (n = 86 / 126 / 142). These are current-vintage data, not real time. |
| QE cannot be learned | **CONFIRMED** as "cannot be learned" | 26 quarters and 4 programs; no inference beyond that. |
| With CAPE as the multiple, the multiple is the price | **CONFIRMED** | Share 1.011 (c14). |
| s_ecyc predicts both legs, not the price | **UNVERIFIED** | 15 non-overlapping windows; the out-of-sample part from 1995 has about 5. |
| Duration ×1.01 | **UNVERIFIED** | Not re-run. |
| λ_π: 2022 "−0.23 real-rate leg + −0.10 inflation leg ≈ −0.33 vs −0.365" | **PARTLY** | The inflation leg uses the pre-2003 coefficient, which is fine. The real-rate leg is imposed by the present-value duration (H ≈ 13.5), while the same regressions estimate b_real ≈ 0 (−0.044 / −0.016 / +0.107 / +0.015 per pp; v05) and the valuation block −0.027 per pp. The identity's rate sensitivity is 3–5 times the estimated one, and 2022 is a single-episode check. |
| Oil-cause label | **UNVERIFIED**; convention **CONFIRMED broken** | qdata: "0 demand / 1 supply"; translator: 0 = supply; the scorer patched it. |
| V-AGG matches the direct equation | **UNVERIFIED** | Not re-run. |
| Tiers selected with all three test decades | **CONFIRMED** (their admission; schema.json tiers/1) | Consequence for the qmodel: 2.7. |
| schema.json used as reference for the translator | **leak** | Templates and scenario_rules contain quarter-dated realized values of test episodes, including GZ and equity legs (finding 3). |

### 2.7 qmodel

| claim | verdict | evidence |
|---|---|---|
| One quarterly model, fitted on quarters ≤ train end | **CONFIRMED** in code | Std (mean/sd) and ridge CV on training rows only (qcore.RidgeModel.fit); missing-data EM within training; late series masked. |
| 1Q full x: G +0.152/+0.226/+0.492 against L0 +0.126/+0.261/+0.441 | **CONFIRMED** numbers | c13 from q05 rows; G − L0 +0.026 [+0.005, +0.051], −0.034 [−0.097, +0.063], +0.051 [+0.019, +0.093]. |
| G competes or beats L0 | **PARTLY** | The margin equals the tier-1 gain over F (L1 − L1-Fp = +0.023 / +0.003 / +0.054), and the tiers were selected on these decades. 16 variants were compared on the test decades (q03). Also L1 (no states) = 0.153 / 0.209 / 0.491, so states add nothing at 1Q. |
| NN loses everywhere; not seed luck | **CONFIRMED** | −0.062 / +0.004 / +0.185; seed sd 0.011–0.014. |
| Macro and policy alone (O-A) unpredictable | **CONFIRMED** (consistent) | Matches c11 (pure macro negative). |
| Valuation as a state adds nothing | **PARTLY** | Level states: L2-allstates − L2 = −0.008 / +0.008 / +0.002 (c13). But L2 and G contain Δy10 × P/D, whose removal costs −0.063 mean10 and −0.374 nasdaq in 2005–14 (q04, scored on the test decade). |
| "Stating both legs fixes it exactly" (2022–23) | **tautology** | The V-PRICE identity: stating y_eps + y_mult states the price. |
| 2020–21 R² 0.638 | **PARTLY** | Pooled over 8 quarters × 10 assets, with 2020Q1–Q2 dominating the variance (about 2 independent events). Oracle x includes realized GZ, y10 and the dollar. |
| Saturation at ±4 sd essential; fixed a priori | **CONFIRMED**, not cherry-picked | q07: at full x, no clip or ±6 scores higher in 1995–2004 (0.161 vs 0.152). It matters for O-A (2015–26 −0.578 → −0.035). The exemption for mechanical own-price inputs came after q05 and affects crude and the bond only. |
| Recursive 12–20Q paths | **UNVERIFIED** | 2–3 windows per decade; not re-run. |

### 2.8 translator

| claim | verdict | evidence |
|---|---|---|
| 36 specs and 36 tables; no valuation views; no asset_drift | **CONFIRMED** | c03: no VALUATION_VARS or wealth views, no asset_drift or factor_drift, analogs empty. |
| The test is not blind | **CONFIRMED, more channels** | The prompt sample line plus a model start known from CTX; the user's 2022–23 example in CTX; few-shot named events; schema.json footprints. |
| Protocol slips (E01 n3; the diff printout) | **UNVERIFIED** (self-reported) | Independent evidence of recall: E08 n3 d_gz corr 0.99, MAE 0.20pp; E05 n3 corr 0.78. |
| E06 engine-side look-ahead | **CONFIRMED** | "T0−7m 至 T0+53y". |
| r_star overrides almost everywhere | **CONFIRMED** | The r_star key appears in all 188 phases; it is numeric (−3 to +3) in 131 of them, and all 36 specs have at least one numeric override. |
| E1/E2 fields only when stated in words, with two exceptions | **PARTLY REFUTED** | It does not list 16 f_sentiment views with chosen magnitudes ("取历史下沿附近": E06 n1–n3, E08 n3, E11 n1–n3, E12 n1–n3) or 3 f_profit_share views (E09). Their closeness to realized is low (median abs z 0.94–2.05). They are chosen magnitudes on an endogenous variable, which pull in months with bad equity residuals through the 0.5 coupling. |
| Inflation derived; u_* sign-only | **CONFIRMED** (scorer audit) | — |

### 2.9 scorer

| claim | verdict | evidence |
|---|---|---|
| Scoreboard, paired comparisons, PIT, coverage | **CONFIRMED** | c01 recomputes them from score_rows.csv: E-n1 +0.004 [−0.020, +0.032], E-n2 −0.005, E-n3 −0.007; E-cond −0.023 [−0.041, −0.005]; PIT 0.25 (engine narratives) against 0.30–0.32 (qmodel). |
| A correct narrative buys nothing over no narrative | **CONFIRMED** (pooled), with power caveat | 1 of 30 paired equity9 intervals favours a narrative (E-n2 1Q) and 3 disfavour one (Q-n1 2Q and 4Q, Q-n2 8Q): chance level. With 11 episodes, differences under about 0.02–0.03 log points cannot be detected. In the 8 non-bust episodes the narratives are about 0 while the oracle is +0.27 (c23). |
| Even a perfect translation would not help; the oracle is worse than no information | **PARTLY REFUTED** | Driven by E03 (80-month training) and the two busts. Without E03, −0.27 [−0.50, +0.28]; without E03, E05 and E12, +0.27 [+0.17, +0.45]. "Excluding E03 changes none of these conclusions" is also refuted for this row. |
| Model loss dominates translation loss | **PARTLY** | True pooled. In ordinary episodes the oracle's gain is lost in translation and the engine (E-cond keeps +0.17 of the oracle's +0.27; the narratives keep about 0). |
| The misses are valuation-multiple swings; P/D explains 83% | **PARTLY**, mechanical | Share 0.827 confirmed. A constant forecast has corr(miss, ΔP/D)² 0.854, above every model. The finding is "no cross-episode skill": corr(median, realized) −0.08 to +0.19 for narratives, +0.55 for VB-oracle. |
| Wrong sign in every down episode "because rate cuts and lower yields load positively" | **sign CONFIRMED, cause PARTLY REFUTED** | B0 decomposition: drift +0.20 (E05) and +0.27 (E12) against factors +0.08 and +0.13; positive unemployment beta. In E10, with hikes, the drift is +0.17. |
| The engine beats the quarterly model on narratives | **CONFIRMED** | Q − E at the common end: +0.056 [+0.013, +0.101], +0.067 [+0.022, +0.129], +0.064 [+0.019, +0.123]; the qmodel wins 2, 1 and 3 of 11. Under oracle macro they tie: −0.017 [−0.073, +0.021]. |
| The factor layer under-delivers (40–70%) | **UNVERIFIED** | Not re-run; consistent with the EP shrink statistics. |
| 2022–23 deep dive | **UNVERIFIED** in detail | B0 attribution for E10 confirmed in sign: intercept +0.168, factors −0.082, y10 −0.085. |
| Audit: no asset or valuation leakage | **PARTLY REFUTED** | Correct for explicit fields. Missed: sentiment and profit-share views, the CTX and schema.json channels, the E08 n3 GZ recall. |
| The test is not blind; 12 of 12 guesses right | **CONFIRMED** | — |
| x_oil_cause bug | **CONFIRMED** | qdata.py line 201: "0 demand / 1 supply". |
| P/Ê distorted by write-downs | **CONFIRMED** | valuation_misses.csv. |
| E-cond is the only clearly positive row | **CONFIRMED**, caveat | Its views are realized means of VIEW_VARS, which include f_y10, f_spread and f_usd (backtest.py line 72): partly a market-price oracle. |

---

## 3. Look-ahead inventory

| channel | where | severity | status |
|---|---|---|---|
| T0 derivable from the prompt sample line, with model start 1973-02 in the shared context | writer prompts + CTX | high, for blindness | confirmed |
| User's 2022–23 example and the lead's named episodes in the translator's context | sim_research_tail.js (CTX) | high, for blindness (E10, E02/E12, E05) | confirmed, assuming CTX = the shared header |
| schema.json templates and scenario_rules with quarter-dated realized footprints (GZ, lending standards, equity legs) | translator reads schema.json | high: the most important variable (GZ) | confirmed; E08 n3 d_gz corr 0.99 |
| Narratives with unique corporate and financial events | writer n1–n3 | medium (recognition) | confirmed |
| Fixed reversion anchors (r* 0.75, TP 0.75, spread 2.0, unemployment 4.8) in every window | MAIN `DEFAULT_REVERSION`, backtest and scorer engine runs | medium | confirmed in code |
| E06 fitted on 53 years after T0 | writer and scorer | high for E06 (reported apart) | confirmed |
| Interpolated quarterly series (labor share in B0) carry part of the next quarter | MAIN panel | low under oracle macro; it is B0's largest 1995–2004 contribution | confirmed (harness note) |
| e_* SVAR shocks (full sample), EBP (full-sample default model), ACM TP (re-estimated), HLW r* (current vintage), Kilian index (revised) | panel and qmodel aux states | low to medium; EBP is used in the layers' leading-state results | noted; B0 and F do not use the e_* shocks or EBP |
| Current-vintage NIPA and Z.1 (FCF yield, profits) as states known at the start | valuation and qmodel | low | noted |
| Shiller monthly E is interpolated from quarterly; a 3-month lag can be about a month short | valuation and qmodel `__known` | low | noted |
| Standardization and masking statistics | qmodel | none found | training rows only (code inspection) |

## 4. Overfitting and selection inventory

- **Schema tiers.** Selected with all three test decades (the schema's own rule). The qmodel's L1, L2 and G inherit this; their margin over L0 equals the tier-1 gain.
- **qmodel variants.** 16 variants scored on the test decades (q03) before L2-T1p-reg was chosen. The spread between variants (0.01–0.02 at 1Q) is the size of the reported G − L0 margins.
- **F (the factor round).** 1995–2004 was the validation period that selected GZ, TP × core4, the dropped labor share and gold from 1982. Only 2005–14 and 2015–26 are test periods for F.
- **Valuation block.** The best spec was chosen from a grid on the same decades. The mostly negative grid results limit the damage.
- **Layers, S+Lc.** The leads were chosen from full-sample lead–lag evidence.
- **Seeds.** No sign of seed luck. The NN seed sd is 0.011–0.014. The scorer's 400-path medians have Monte Carlo error of about 0.01 per asset and episode, small next to the bootstrap intervals over episodes.
- **Multiple testing in the narrative test.** 30 paired intervals: 1 favourable, 3 unfavourable.

## 5. Small samples: what can and cannot be concluded

| evidence | independent units | can conclude | cannot conclude |
|---|---|---|---|
| Harness 1Q / 4Q pooled rolling | 167 / 41 | differences of about 0.05–0.1 in mean10 R² (e.g. GZ's +0.10) | per-decade rankings within ±0.05 |
| Harness 12Q / 20Q per decade | 3 / 2 | nothing beyond anecdote; the sign only of very large effects | coverage (0.53 vs 0.80), R² rankings, valuation "explaining" a decade |
| Harness 12Q / 20Q pooled | 13 / 7–8 | large effects (F − F_nocredit +0.25, interval > 0) | decade-specific claims; band calibration |
| Episode test | 11 anchored (7 reach 8Q; 2 busts; E02 and E07 adjacent) | that narratives add less than about 0.02–0.03 log points of equity9 mean absolute error; that the qmodel loses to the engine by about 0.06 | busts versus ordinary episodes (2 busts); conditional calibration; per-asset-group claims |
| Valuation, 10-year | 3–14 decades | the CAPE rank correlation is positive over 1881–2016 | out-of-sample level skill after 1980 (4 windows); "ECY works" (7) |
| Joint contractions / inflation windows | 7 / 4 | the plain BVAR's long contractions are implausible (p 0.004) | the engine's 18% excess (p 0.25); 13-year inflation-share comparisons |
| Schema QE / λ_π / s_ecyc | 4 programs / about 2 swings / 15 windows | nothing statistical | a calibrated λ_π or κ pair |
| 2020–21 qmodel check | 2 extreme quarters | the generic channels have the right sign under oracle x | that no pandemic flag is needed |

## 6. Data errors and measurement issues

1. **MAIN `s_cape` after 2023-06** (NIPA splice): 4.0–12.5% too high; 43.7 against 40.6 in 2026-09. Used by the production translator prompt.
2. **The project's Shiller copy ends E at 2023-06 and CAPE at 2023-09.** The harness, valuation and qmodel use the newer workbook (E/D to 2026-06); the layers' 1881–2023 tests use the old one (harmless for 10-year returns to 2016).
3. **Trailing P/E (and the schema's P/Ê y_mult)** is distorted by write-downs in 2001–02 and 2008–09.
4. **P/D trend.** About 30% of the 1973–2025 rise is the payout ratio (0.26 of 0.88 log).
5. **French-implied dividends.**
   - BusEq P/D breaks in 2004 (Microsoft's special dividend). Industry P/D is lumpy and also moves when constituents change at the annual June/July reclassification.
   - The market P/D is clean as an identity, but the 2009 dividend cuts move it against price (2005–14 share_m 0.56–0.60 at 12–20Q).
6. **x_oil_cause** codes are opposite between the translator (0 = supply) and qdata (1 = supply).
7. **NFC FCF yield**: an all-NFC NIPA/Z.1 proxy, not a market FCF yield for SPY or QQQ.

## 7. What the design stage should take from this

1. Treat **financial conditions (a credit-spread or risk-premium path) as the scenario's main asset lever**: a required field, not R5's "only in a crisis". Say plainly that stating it is close to stating the equity risk premium. Score "macro only" and "macro + stated spreads" as separate arms.
2. **Report placebo-corrected anatomy.** Replace corr(miss, Δmultiple)² with corr(forecast, outcome), or with the share of the multiple change that the model predicts.
3. **Rebuild the narrative test so it is blind**:
   - prompts without the sample length and without named-event few-shots;
   - a translator context without CTX examples or schema footprints (field definitions only);
   - perturbed or synthetic episodes and a wrong-narrative placebo;
   - N3 checked for recall of E1 fields (as with E08 GZ).
4. **Fit reversion anchors (r*, TP, spread, unemployment) to the train end in every backtest window**, then re-run the backtest tables.
5. **Pre-register the qmodel's field set and states before scoring**, or score only on 2015–26 using fields frozen on 1995–2014. Keep L0 or F as the default, since G's margin is selection-sized.
6. **Data fixes**:
   - MAIN s_cape: Shiller's newer file, or the E10 splice re-based;
   - the x_oil_cause convention;
   - y_mult defined on CRSP P/D or CAPE, not P/Ê;
   - payout-adjusted P/D, or CAPE, for the CMA's re-rating term.
7. **Re-state the headline result.** A correct narrative did not help in these 11 episodes. An exact factor path helps in ordinary episodes (about +0.27 MSE skill, n = 8, post hoc) and fails in the two equity busts, where the multiple collapsed while credit spreads, rates and drift pointed up.
