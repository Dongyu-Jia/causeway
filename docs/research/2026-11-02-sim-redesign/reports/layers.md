# Timing classes, anticipation and endogeneity (research round, role `layers`)

*What the role did:* same-quarter against leading inputs, the endogeneity table, the first model card, the forecast-tab proposal.

Rendered from the role's final report in the research workflow's journal (run `wf_cc7fe8c3-afc`, result `layers`). The text is unchanged except that local absolute paths are written as the spec writes them (`scratchpad/sim_redesign/...`, `scratchpad/...`). It is the report the critic and the designer read; the numbers are the role's own, and the critique ([`critic/critique.md`](../critic/critique.md)) and the spec ([`SIM_DESIGN.md`](../SIM_DESIGN.md)) say which of them survived.

## Summary

Role "layers": the timing taxonomy, the three-layer contract and the forecast-layer proposal. Everything is read-only on MAIN, RWT and FWT. Scripts and outputs are in the scratch folder sim_redesign/layers. l01_timing.py and its CSV come from an earlier start of this same role; I re-read them and built on them.

TAXONOMY. The full model card is l04_model_card.csv: 50 variables, each with its Chinese name, panel columns, users, publication lag, timing class, role and evidence string. The sample is quarterly 1973Q1–2026Q3 (n=215 quarters), with 52 Q4 origins for the one-year leads and 36 expanding out-of-sample origins from 1989Q4. Seven classes:
(1) 同期驱动 (same-period driver). 油价 oil, 商品/服务通胀 cpi_goods/cpi_services, 政策利率 ff, 10年期收益率 y10, 美元 usd, 实际利率 real_y10, 期限溢价 tp10 and tp10_x_core4, 金属/油价 metals_oil.
(2) 同期驱动·市场约提前一季定价 (driver priced about one quarter ahead). 工业产出 ip and 失业率 unrate. Their same-quarter link to asset returns is significant for 0 of 10 assets. The previous quarter's asset returns are significantly correlated with this quarter's change for 9 of 10 assets.
(3) 同期·内生 (same-period but endogenous: the price itself, or built partly from prices). 信用利差 spread (assets explain its same-quarter change, adj R² 0.32), GZ (0.40), 信心 sentiment, 初请 claims, EPU, 家庭财富 wealth, and the valuation changes f_valuation and f_cape (adj R² 0.93 and 0.88).
(4) 领先状态·宏观 (leading state for the macro block; PLAUSIBLE only). Annual out-of-sample n=36, best of 4 macro targets:
   - 期限利差 term spread → Δunemployment: r −0.46, OOS +0.13
   - 实际政策利率 real_ff → Δff: −0.33, +0.17
   - EBP → Δunemployment: +0.43, +0.26
   - 工资 wage → ΔCPI: +0.56, +0.27
   - 资本开支 capex share → IP: −0.41, +0.22
   - also 房地产贷款 re_loans and 费城资本开支 philly_capex
(5) 领先状态·资产（仅多年期）(leading for assets, multi-year only). The P/D and CAPE levels.
(6) 起点状态 (start state for re-anchoring and mean reversion). real oil, CPI y/y, core CPI y/y, the inflation-expectation proxy, r*, Michigan 1-year expectations, and rg_high_infl.
(7) 滞后/描述 (lagging or descriptive). The interpolated labor share (it looks ahead), profit share, profit growth, fcf_share, white collar, productivity, saving rate, house prices, GPR, every ex-post rg_ label, rg_rt_contraction, sahm_gap and infl_trend_12m.

No variable leads one-year asset returns. Across all 50, the best us_equity OOS R² is +0.04, and no lead is significant. This is why the analog kernel over KEY_STATES collapses to ENS 576.

CONTRACT.
- What a scenario must state, per phase in whole quarters (at least 6 months): policy (f_ff or r_star), growth (ip and/or unrate), inflation, oil, credit stress when the narrative has it, and a new valuation field. Its mode is inferred (default), hold or rerate. Rerate is stated qualitatively and mapped to preset sizes of the residual multiple. It is never a P/E target written by the LLM.
- What the model infers: every other factor, jointly. Asset paths come from:
  - same-quarter factor changes, plus the next quarter's ip and unrate changes (legitimate because the scenario supplies the whole factor path);
  - the macro-implied part of the multiple;
  - the residual and CMA/income terms;
  - today's valuation as the initial condition.
- Leading states may tilt scenario probabilities. They never move asset paths directly.
- In backtests the LLM runs blind: no asset_drift, and no valuation magnitudes.

FORECAST TAB. Headline: the probability-weighted 4-quarter mixture of the scenarios, in quarterly bands. Beside it go each scenario's quarterly median and a table of what each scenario assumes ('what you must believe'). The analog method is demoted to the reference band 无情景基准 and kept only in the ledger. An honest validation has three parts:
- the end-to-end conditional test, with arms for oracle macro, a macro-only narrative and a narrative plus qualitative valuation, and a check that the right narrative scores better than a wrong one;
- coherence and base-rate checks of the probabilities;
- prospective scoring in the ledger, plus contemporaneous Fed staff/SEP forecasts used as the hypothesis written at the time.

The probabilities of the 13 hand-written scenarios cannot be calibrated.

## Key findings

1. HORIZON HYPOTHESIS: SUPPORTED IN 2 OF 3 DECADES. The quarterly refit S uses 10 factors: the core with the timing-fixed spread, GZ added and the labor share dropped. Its mean OOS R² over 10 assets is -0.146/-0.238/+0.179 quarterly and -0.780/-0.612/+0.210 annual (1995-2004/2005-14/2015-26; 40/40/46 quarters, 10/10/11 years). us_equity's annual oracle error correlates with the same-year change in log P/D at +0.57 (n 10), -0.57 (n 10) and +0.89 (n 11). The bubble years 1996-99 have errors of +0.25 to +0.36 log, with re-ratings of +0.13 to +0.22. In 2005-14 the sign flips: the biggest miss is 2009 at -0.82 log, a linear overshoot on the GZ/spread narrowing plus P/D distorted by dividend cuts, not a valuation swing. Pooled over 31 years the correlation is 0.06.
2. VALUATION IS THE MISSING INPUT, AND THE MODEL CANNOT INFER IT. As a NON-admissible upper bound, adding the same-quarter change in log P/D to S lifts the mean quarterly R² to +0.265/+0.370/+0.457 and us_equity's to 0.90/0.74/0.93. Same-quarter factors explain 40-50% of the quarterly change in log P/D in sample, and 0.17/0.17/0.55 out of sample. The residual multiple has quarterly AR1 of about 0.00-0.03, so its level wanders like a random walk and no later quarter gives it back. Using the start log P/D level as a linear input (S+V) hurts: -0.684 in 1995-2004. About 95% of the annual variance of us_equity price returns is the change in log P/D (n 52 years; P/D overstates this relative to P/E, because dividends are smooth).
3. VALUATION LEVEL IS A SLOW LEADING STATE ONLY. Shiller 1881-2023, non-overlapping January origins. Start log CAPE against the next 1/3/5/10-year real return: corr -0.19/-0.29/-0.32/-0.52 (n 142/47/28/14). Expanding OOS R² from 1950: -0.02/-0.18/-0.04/+0.13 (n 73/24/14/7). Log P/D at 10 years: corr -0.30, OOS -0.21. In the panel since 1973, one-year us_equity OOS R² is -0.20 (P/D) and -0.31 (CAPE). valuation.json also says the real-time error-correction slope is -0.067 (t -0.8). Conclusion: today's valuation enters as the initial condition (income, CMA, the implied-multiple path), plus an optional slow reversion that the scenario chooses. It is not a one-year predictor and not an input to the transmission.
4. ANTICIPATION: ip AND unrate ARE PRICED ABOUT ONE QUARTER AHEAD. Their same-quarter link to asset returns is significant for 0 of 10 assets. The previous quarter's returns are significant for 9 of 10. Adding the next quarter's changes in ip and unrate (S+Lc) raises quarterly mean R² from -0.146/-0.238/+0.179 to -0.096/-0.159/+0.202, better for 9/8/7 of 10 assets, and annual R² from -0.780/-0.612 to -0.414/-0.270 (2015-26: 0.210 to 0.192). Moving-block bootstrap (4 quarters) of the variance-scaled squared-error gain: P>0 = 0.98/0.71/0.68 by decade, pooled 0.94 (n 126 quarters, 90% CI -0.03 to +1.71). PLAUSIBLE, not confirmed: the two variables were chosen from full-sample lead-lag evidence, and leading all 10 factors (S+La) overfits (-0.364/-0.396/+0.191). In a simulation this is admissible, because the scenario supplies the whole factor path.
5. KEY_STATES VERDICT, one year ahead, Q4 origins (r = in-sample correlation, OOS = out-of-sample R², n=36):
   - cpi_yoy: start state, mean reversion (ΔCPI r -0.41, OOS +0.13)
   - unrate: same-period, priced ahead (level OOS about 0)
   - real_ff: leads Δff (OOS +0.17)
   - term_spread: leads Δunemployment (OOS +0.13)
   - spread: endogenous, not leading (macro OOS -0.39; the timing-fixed spread -0.51)
   - div_yield: multi-year only (one-year OOS -0.16)
   - ip_yoy: same-period
   - oil_yoy: start state (CPI base effect)
   None of the 8 leads one-year asset returns, so an analog over them is effectively unconditional. That matches ENS 576 at 38 of 42 origins and the uncond method's rank IC of 0.187 (t 3.08), which the analog does not beat.
6. ENDOGENEITY: share of each variable's same-quarter change explained by the 10 asset returns (adj R²) — a view on these partly dictates the asset outcome:
   - P/D 0.93
   - CAPE 0.88
   - oil 0.45
   - GZ 0.40
   - spread 0.32 (timing-fixed 0.32)
   - EBP 0.31
   - EPU 0.29
   - y10 and real_y10 0.27
   - sentiment 0.27
   - claims 0.21
   validate_spec today warns only for valuation and wealth. The endogenous-variable warning should extend to spread, GZ, EBP, sentiment, claims and EPU.
7. THREE-LAYER CONTRACT, LAYER 1 (scenario hypothesis; the only predictive layer). Every phase is a multiple of 3 months, at least 6. It must state:
   - R1 policy: f_ff or r_star;
   - R2 growth: ip and/or unrate, with recession yes/no and depth;
   - R3 inflation: s_cpi_yoy or the goods/services split;
   - R4 oil: f_oil or e_oil_supply;
   - R5 credit stress: spread/gz, only when the narrative has a crisis;
   - R6 a new field valuation = {mode: inferred|hold|rerate, size: 温和/明显/极端 = 0.5/1/2 sd of the annual residual multiple, why}.
   Optional: usd, fiscal, factor_drift, regime phase key, prior_prob. The translator prompt gets a 'five things you must give' section. validate_spec reports which of R1-R4 are missing and fills them as model-inferred, so the user sees what the scenario left to the model.
8. THREE-LAYER CONTRACT, LAYERS 2 AND 3 (what the model infers).
   - Factor layer: one joint model for every variable a scenario leaves unspecified. Today that is the EP-reweighted 6-month block bootstrap, with rate scaling and mean reversion to start states. The contract does not depend on which joint model fills it.
   - Asset layer: assets per quarter = B0 · same-quarter factor changes + B1 · next-quarter ip and unrate changes (if adopted) + the macro-implied multiple (already inside the betas) + the residual multiple (zero mean unless the scenario says rerate; if it says hold, the factor-implied part is offset; reversion only if the scenario opts in) + the residual + CMA + income.
   - Today's P/D, CAPE and FCF proxy enter only as initial conditions.
   - Outputs: the implied P/D (and P/E) path and the regime labels. Both are descriptive, so nothing is counted twice.
   - Every config FACTOR/STATE gets a timing field from the model card: same, same_anticipated, endogenous, leading_macro, leading_multiyear, start_state, descriptive. The UI and the prompt show it.
9. FORECAST TAB.
   - Rename it 未来一年：情景条件模拟.
   - Headline: the probability-weighted mixture of the scenarios' first 4 quarters, in quarter-end bands (no monthly).
   - Show each scenario's quarterly median line and a table of what each scenario assumes: ff, unrate, CPI, oil, spread, valuation mode.
   - Let the user set the probabilities, and show how much each asset's mean moves with each scenario's probability.
   - The analog method becomes the reference band 无情景基准（历史常态）, which is what it already is in practice. Keep it in the append-only ledger for continuity, but not as a method choice.
   - calibrate_probs may tilt scenario probabilities only through leading-macro states that pass macro_skill (candidates: term spread → unemployment, real_ff → ff). Leading states belong to the hypothesis layer, never to asset paths.
10. HONEST VALIDATION.
   (a) The user's end-to-end test is a conditional-simulation test, with 4 arms per episode: oracle macro (no LLM); a macro-only narrative translated blind (no asset_drift, no valuation magnitudes, narrative checked for asset-outcome words); the narrative plus a qualitative valuation statement; and a wrong-narrative control, which must score worse than the right one (discrimination). Metrics: CRPS skill against unconditional, 90% coverage, cross-asset rank IC, per quarter and per year. Use about 12-15 episodes since 1973 with HAC or block bootstrap, and report the number of independent quarters per episode.
   (b) The probabilities of 13 overlapping 15-year scenarios have one realization: they cannot be calibrated. What can be tested: coherence (they sum to 1, EP λ/KL, support); the mixture's implied macro base rates (P(recession in 2 years), P(average CPI > 4%)) against history and against contemporaneous Fed SEP/Greenbook forecasts from federalreserve.gov; prospective Brier/CRPS scoring in the ledger, which takes years.
   (c) Contemporaneous Fed staff forecasts used as the hypothesis written at the time test the whole chain without hindsight, at about one origin per year.
   (d) Promote the backtest's error decomposition (hypothesis error vs simulation error) to the headline.
11. THE LLM IS A LEAKAGE CHANNEL. The translator knows how history turned out: for example, that the Nasdaq fell about 33% in 2022. In end-to-end tests asset_drift must be stripped, valuation must be qualitative and mapped to preset sizes, and VALUATION_VARS views must be refused, not just warned about (today repair() only warns). Each translation is recorded with the model id and the narrative hash. The oracle-macro arm is the leakage-free reference.

## Open questions

1. Should the next-quarter ip/unrate lead (S+Lc) be adopted in the transmission? The evidence is PLAUSIBLE: pooled bootstrap P 0.94, 90% CI touches 0, and the two variables were picked from full-sample lead-lag evidence. It needs a pre-registered entry in FWT's walk-forward protocol on the fixed panel before it goes into the engine. Attribution and the monthly simulator would also need a quarterly or lead-aware design.
2. Which valuation mode applies when a scenario states none (the user's call)? Options: 'inferred' (only the macro-implied multiple; the residual multiple has zero drift) or a slow reversion toward fair value, which the 10-year evidence supports weakly (corr -0.52 with CAPE, n 14; OOS +0.13, n 7). The fair-value regression itself is unstable: sign flips in 2 of 3 features (valuation.json).
3. Which valuation measure? P/D is distorted by the 2008-09 dividend cuts and by buybacks. The project's Shiller copy has earnings only to 2023-06, and another agent appears to be parsing a newer file. No free SPY/QQQ FCF-yield or QQQ P/E history exists, and fcf_share is a NIPA macro proxy (profits minus capex over GDP), not a market FCF yield. The user's emphasis on SPY/QQQ P/E and FCF can therefore be met only through CAPE/P/E for the S&P and the NIPA proxy.
4. The 2009 oracle miss (-0.82 log) looks like a linear overshoot on the record GZ/spread narrowing. Does the asset layer need a level-dependent or capped credit beta? FWT notes that GZ changes are not scaled to today's spread level.
5. Are the leading-macro states used to tilt scenario probabilities (calibrate_probs)? They are PLAUSIBLE only: n=36 annual OOS origins, best of 4 targets. Before any use they need the same macro_skill test (beat both uncond and naive in both halves).

## Files the role listed

Paths are relative to `scratchpad/sim_redesign/`. Only the files marked are in this archive; the rest (scripts, work outputs, data) stay in the research folder.

- `layers/l04_model_card.csv`
- `layers/l04_model_card.py`
- `layers/l01_timing.py`
- `layers/l01_timing.csv`
- `layers/l02_evidence.py`
- `layers/l02_evidence.json`
- `layers/l03_multiple.py`
- `layers/l05_boot.py`
- `layers/rg_export.py`
- `layers/rg_cols.csv`
