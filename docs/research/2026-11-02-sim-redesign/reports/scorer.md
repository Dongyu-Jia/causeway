# End-to-end narrative test (research version) (research round, role `scorer`)

*What the role did:* the scoreboard by episode and asset group, the decomposition, the audit and the deep dive.

Rendered from the role's final report in the research workflow's journal (run `wf_cc7fe8c3-afc`, result `scorer`). The text is unchanged except that local absolute paths are written as the spec writes them (`scratchpad/sim_redesign/...`, `scratchpad/...`). It is the report the critic and the designer read; the numbers are the role's own, and the critique ([`critic/critique.md`](../critic/critique.md)) and the spec ([`SIM_DESIGN.md`](../SIM_DESIGN.md)) say which of them survived.

## Summary

I ran the user's end-to-end test on all 12 episodes with both engines: narrative → translated spec or quarterly table → factors → assets. I scored 13 assets at 1, 2, 4 and 8 quarters and at the window end (12 quarters exists only for E09). Everything is under SCORER = scratchpad/sim_redesign/scorer: scripts s02–s13, raw paths in work/, tables in out/. Nothing in MAIN, RWT or FWT was written. Every run was `cd MAIN && .venv/bin/python -B`, and no MAIN file is newer than my first script. RWT's newer files belong to the regime round.

**Short answer.** A correct narrative of the episode buys nothing over no narrative at 4 quarters or at the window end, in either engine. A perfect translation would not help either: fed the realized factor path, today's transmission is worse than no information at the window end. The binding loss is factor → asset, and the us_equity misses are almost entirely valuation-multiple swings.

## SETUP

**Today's engine**
- Contexts were rebuilt exactly as the writer built prompt.md: `fit_window` at T0, `model_selection` 'base', e_* columns dropped; E06 leave-window-out.
- Check: all 52 variable-table rows are identical to each prompt.md, in all 12 episodes.
- Each spec went through `translator.repair` → `validate_spec(180)` → `runner.phase_weights` → `with_oil_lock` → `simulate`.
  - 400 paths, seed 11, CMA historical, partial residual coupling 0.5.
  - The runner's kappa-0 P/D tracker is on; it changes no return.
- Post-processing record:
  - Repairs: 0.
  - Clips: 13, all small (E05 f_unrate and f_cpi_goods, E06-n3 f_ff, E11 f_oil ×6, E12 f_oil ×2).
  - Entropy-pooling shrink (λ &lt; 1) in 32 of 52 window phases for n1 (mean λ 0.79), 41 of 53 for n2 (0.67), 23 of 47 for n3 (0.84). In E03, λ is 0.05–0.43 on its 80 DB months.
- Runs:
  - E-uncond, E-cond: as the backtest's fan_uncond and fan_cond.
  - E-oracle: base model fed the realized factors, with the engine's residual block band. Treasury and crude are exact by the structural rules; cash follows its rule.
  - E-n1, E-n2, E-n3: the translated specs.
  - F-oracle: the factor round's F, monthly.

**Quarterly model**
- qplugin "G", 1000 paths, seed 11.
- Runs:
  - Q-n1, Q-n2, Q-n3: from q_nX.json. Nulls are masked. Sign-only and template_sign entries cannot be passed to the plug-in.
  - Q-oracle: realized x. Q-uncond: everything masked.
  - VB-oracle: valuation block pd_none_zero, with a P/D block for us_equity and F for the other 9 risk assets.
- E06 cannot run: only 2 training quarters.
- E03 trains on only 26 quarters.
- E08 starts mid-quarter. I used origin 2020Q1 and mapped the narrative quarters one-to-one, a one-month shift.
- E03 and E04 are scored to their last full quarter (1982-06 and 2015-12).
- x_oil_cause: the translator codes 0 = supply, while the qmodel data code 1 = supply. I passed it as d_oil_sup = d_oil × (1 − cause).

**Scoring**
- Measure: cumulative log price return (cash: log total return).
- Error: median − realized. Also the 80% band hit, PIT and CRPS.
- Pooled over the 11 anchored episodes; the independent unit is the episode. E02 and E07 are adjacent. Only 7 episodes reach 8 quarters.
- 90% intervals come from a bootstrap over episodes. E06 is reported separately.

**Timing of inputs**
- Translated views, tables and oracle factors are all same-period.
- Contexts are anchored at T0. Qmodel states are known at the start, with publication lags: CPI 1 month, Shiller E 3 months, NIPA release-dated.
- Fits use data up to T0 only, except E06.
- No realized asset price or valuation change after T0 enters any run.
- Caveats:
  - The base model's labor share is interpolated, so it contains part of the next quarter.
  - F's GZ spread is a bond-market price.
  - The qmodel treats the previous quarter's x as known.

## SCOREBOARD (equity9 = us_equity, nasdaq, 6 sectors, REITs; mean |median error| in log points; 11 anchored episodes)

| run | 1Q | 4Q | 8Q (7 episodes) | end | 80% hit at end |
|---|---|---|---|---|---|
| E-uncond | .087 | .144 | .286 | .251 [.18, .34] | .71 |
| E-cond | .084 | .141 | .268 | .228 | .76 |
| E-oracle | .080 | .155 | .384 | .282 | .66 |
| E-n1 | .089 | .163 | .308 | .255 | .72 |
| E-n2 | .076 | .144 | .302 | .246 | .71 |
| E-n3 | .089 | .161 | .298 | .244 | .75 |
| F-oracle | .093 | .159 | .369 | .263 | .63 |
| Q-uncond | .098 | .147 | .300 | .262 | .69 |
| Q-oracle | .096 | .169 | .370 | .269 | .64 |
| Q-n1 | .096 | .182 | .380 | .306 | .72 |
| Q-n2 | .090 | .167 | .392 | .306 | .61 |
| Q-n3 | .095 | .173 | .391 | .301 | .71 |
| VB-oracle | .093 | .160 | .362 | .261 | .66 |

- **PIT.** Mean |PIT − .5| at the end is .25 for the engine narratives (calibrated) and .30–.32 for the qmodel (over-confident).
- **us_equity at the end:**

| run | error |
|---|---|
| E-uncond | .230 |
| E-cond | .196 |
| E-oracle | .250 |
| E-n1 / n2 / n3 | .227 / .213 / .221 |
| Q-oracle | .232 |
| Q-n1 / n2 / n3 | .279 / .283 / .288 |
| VB-oracle | .194 |

- **Other groups at the end:**
  - Gold: E-uncond .193, E-n2 .173, Q-oracle .147.
  - Crude: E-uncond .240, E-cond .103, E-n .16–.24, Q-n1 and Q-n2 .04–.05 (a stated d_oil drives crude mechanically).
  - Treasury + cash: E-uncond .058, E-n .025–.033, Q-n2 .014, Q-n1 .059.
- **Narrative minus no narrative (equity9, window end):**
  - E-n1 +.004 [−.019, +.031]; E-n2 −.005 [−.019, +.010]; E-n3 −.007 [−.028, +.019].
  - E-cond −.023 [−.041, −.006].
  - E-oracle +.031 [−.017, +.084], MSE skill −.41 [−.76, −.01].
  - Q-n1 +.044 [−.009, +.102]; Q-n2 +.044 [−.010, +.112]; Q-n3 +.040 [−.022, +.113].
  - Q-oracle +.007 [−.053, +.077].
- **The one significant narrative gain:** n2 in the engine at 1Q, −.010 [−.020, −.002], MSE skill +.24 [+.07, +.43].
- Excluding E03 changes none of these conclusions.

**By episode** (equity9 at the window end). Each row: E-uncond / E-cond / E-oracle / E-n1 / E-n2 / E-n3 ‖ Q-uncond / Q-oracle / Q-n1 / Q-n2 / Q-n3 ‖ VB-oracle; realized us_equity in brackets.
- E01: .21/.18/.20/.20/.20/.19 ‖ .20/.19/.26/.24/.24 ‖ .20 [+.23]
- E02: .36/.34/.33/.43/.38/.30 ‖ .42/.38/.32/.39/.31 ‖ .34 [+.55]
- E03: .20/.21/.45/.18/.21/.21 ‖ .17/.55/.38/.56/.55 ‖ .61 [+.12]
- E04: .17/.11/.11/.14/.15/.15 ‖ .14/.11/.11/.14/.11 ‖ .10 [−.04]
- E05: .63/.53/.75/.61/.59/.59 ‖ .59/.45/.79/.64/.65 ‖ .41 [−.47]
- E07: .08/.10/.08/.08/.07/.09 ‖ .10/.10/.09/.10/.09 ‖ .09 [+.14]
- E08: .15/.19/.13/.15/.18/.16 ‖ .16/.12/.16/.16/.13 ‖ .15 [+.33]
- E09: .19/.15/.14/.16/.15/.15 ‖ .19/.14/.22/.22/.24 ‖ .16 [+.36]
- E10: .16/.15/.12/.11/.17/.14 ‖ .24/.17/.14/.18/.14 ‖ .12 [−.02]
- E11: .12/.08/.09/.11/.08/.09 ‖ .16/.08/.14/.09/.12 ‖ .09 [+.17]
- E12: .50/.48/.71/.65/.54/.63 ‖ .52/.68/.76/.64/.74 ‖ .62 [−.38]

E05, E12 and E02 dominate the errors; their 80% hit rate is 0–.44.

**us_equity window-end medians, n1 / n2 / n3.** In every down episode the narrative medians have the wrong sign:
- E05 (realized −.47): engine +.12 / +.11 / +.11; qmodel +.31 / +.16 / +.14. Oracles: E +.26, Q −.13, VB −.32.
- E12 (realized −.38): engine +.35 / +.22 / +.33; qmodel +.46 / +.34 / +.44.
- E10 (realized −.02): engine +.04 / +.12 / +.07.
- E06 (realized −.14; −.57 at 4Q): engine +.08 / +.09 / +.10.

**E06** (leave-window-out, look-ahead in fitting), equity9 at the end:
- E-uncond .338, E-cond .268, E-oracle .326, E-n .26–.29, F .276, VB .260.
- us_equity: every engine run is positive; VB −.07.

## DECOMPOSITION (equity9, window end, RMS in log points)

**Engine**
- Narrative error .336–.355, against no narrative .339.
- Translation (narrative − cond): .074–.105.
- Engine part (cond − oracle: pooling, shrink, re-anchoring): .142.
- Transmission (oracle − realized): .403.
- Narrative − oracle: .11–.12. It is correlated −.53 to −.67 with the transmission error: the engine's pull toward history offsets part of the transmission's misses. That is why the narrative runs beat E-oracle (−.027 to −.038 at the end).

**Qmodel**
- Narrative error .41–.42, against no narrative .346.
- Translation (narrative − Q-oracle): .105–.163. n2 is lowest at 4Q (.070, against n1 .107).
- Model (Q-oracle − realized): .376.

**Today's engine versus the quarterly model** (common end, equity9): Q-nX − E-nX = +.056 [+.013, +.099] (n1), +.067 [+.021, +.131] (n2), +.064 [+.017, +.122] (n3). The qmodel wins only 1–3 of 11 episodes. Under oracle macro the two tie: Q-oracle − E-oracle = −.017 [−.075, +.021].

**Scenario → factor layer**
- The engine's narrative runs deliver only part of the realized macro move at the window end: policy rate 48–60%, unemployment 37–51%, CPI y/y 33–40%, oil 47–63%, 10y 54–77%. Mean gaps: ff 1.2–1.5pp against a 2.8pp move; unemployment 0.9–1.2pp against 1.9pp.
- The cause is entropy-pooling shrink, plus means met by skewed mixtures. In E10 n1, months 7–12, oil has a mean of −.20 but a median of −.04: 27% crash blocks.
- The qmodel holds stated inputs exactly. Masked financial conditions under n1 are no better than unconditional: d_y10 gap 1.41 against 1.39, d_usd 7.6 against 8.0, d_gz .72 against .60.

**Valuation**
- The realized CRSP P/D change explains 83% of the cross-episode variance of window us_equity returns.
- The narrative runs' misses correlate −.81 to −.88 with it (r² .66–.78). The oracle models' misses correlate −.59 to −.69.
- The schema's P/Ê leg is distorted by EPS write-downs: the E05 "multiple" is +1.0 while price fell .47; E12 +.26 while price fell .38.

## AUDIT

**36 specs**
- None has a view on a valuation, asset or endogenous variable; none has an asset_drift, factor_drift, analogs or a reversion block.
- r_star overrides appear in all 36 (−3 to +3, chosen by the translator, not in the narratives).
- resid_scale 1.5–2.0 in E05 and E08.
- f_spread views inferred from words in the E05 and E08 n1/n3 specs: 5 views in n1 and 5 in n3. They sit far from realized (median |z| .75).

**36 tables**
- No output legs (no y_*, y_mult or y_eps).
- E12 carries a c_q.X27 write-down note. It is text only and the plug-in ignores it.
- n1 and n2 values are all stated, interp, derived or band.
- Inferred financial conditions appear only in the E05 and E08 n3 tables (d_gz; d_cpbill in E05).
- u_* is never numeric.

**Contamination (closeness to realized, median |z|)**
- Spec views: n1 .16, n2 .21, n3 .22. A no-information baseline (the history prior's mean) scores .66–.76.
- Table core fields: n1 .21, n3 .25. The model's unconditional completion scores .42.
- E02, whose n3 was translated before the diff slip, is equally close: n3 .18 against a baseline .77 (n1 .12).
- So the number-free n3 recovers most of the realized magnitudes. That can only come from recognising the episode.

**Period guesses**
- 12 of 12 exact, both T0 and window. Confidence .95–.99; from n3 alone .75–.99.
- Accuracy on unrecognized episodes cannot be measured: there are none.
- T0 can also be computed from the prompt alone (sample length + 1973-02).

**Other checks**
- N3 duration mismatches (E02, E04, E06, E07, E11) cost nothing detectable.
- Passing x_oil_cause raw, with the flipped convention, would move medians by at most .05 (mean .01).

## DEEP DIVE: 2022–23 (E10)

**Realized** log price returns, 2022 / 2023:

| asset | 2022 | 2023 |
|---|---|---|
| us_equity | −.237 | +.222 |
| nasdaq | −.400 | +.431 |
| energy | +.460 | −.046 |
| REITs | −.321 | +.065 |
| 10y bond | −.209 | −.002 |
| crude | +.062 | −.109 |

**2022 drawdown.** No run reproduces it in its median.

| run | 2022 us_equity median |
|---|---|
| E-n1 / n2 / n3 | +.017 / +.030 / +.024 |
| E-oracle | −.031 |
| F | −.049 |
| VB | −.084 (the best: about a third) |
| Q-n | +.055 to +.063 |
| Q-oracle | +.018 |

- P(2022 below −15%): engine narratives .18–.23 (unconditional .10); qmodel narratives .08–.09.
- P(below the realized value): E-n1 .19; every qmodel run ≤ .04.
- The realized value is outside the qmodel's band and inside the engine's (PIT .12–.18).

**2023 rebound.** Only partly caught: E-n2 +.11, E-oracle +.11, VB +.11, Q-oracle +.14; Q-n1 and Q-n3 about 0.

**What drove the engine's equity path** (n1 attribution over 2022):
- Contributions: y10 −.059, usd −.029, cpi_goods −.022, spread −.021, oil +.008, intercept/drift +.084. Net about 0.
- Fed the realized factors, the same model gives −.033 against a realized −.211. Its 10y loading is −.036 per pp: the 2.4pp yield rise explains only −.086.

**What drove the qmodel**
- Q-oracle catches 2022Q2 (−.098 against −.193), through GZ −.048, term premium −.033, dollar −.014.
- Under n1 it fills d_gz with the wrong sign (−.21 against +.17 in Q1), d_y10 ≈ 0 (realized +.80 / +.66 / +.85) and d_usd ≈ 0 (realized +4.9 / +5.3).

**Valuation legs**
- Realized P/Ê multiple: −.457 by 2022Q3. Q-oracle's leg reaches −.209; Q-n reach −.11 to −.14.
- VB captures 66% of the H1-2022 P/D compression: −.194 against −.293.

**Macro delivery** (engine n1):
- CPI y/y is close: 8.6 / 9.7 / 8.5 / 6.7 against 8.6 / 9.0 / 8.2 / 6.4.
- Policy rate falls short: 2.5% against 4.1% at end-2022, 3.5% against 5.3% at end-2023.
- Unemployment drifts to 4.3% against 3.8%.
- Oil stays at 104 against 80.

**10y bond** in 2022: engine narratives −.14 to −.16; Q-n2 −.18; Q-n1 −.03.

**Missed:**
- The real-rate-driven multiple compression.
- Energy's +46% (every median between +.02 and +.08).
- nasdaq's −40% / +43% (PIT .02–.09).
- REITs' −32%.
- The 2023 re-rating.

Chart-ready numbers are in out/deepdive_E10.json and out/deepdive_E10_paths.csv.

## Key findings

1. A correct narrative buys nothing over no narrative, in either engine. Equity9 at the window end, narrative minus no narrative (log points, 11 anchored episodes, bootstrap over episodes): engine n1 +.004 [−.019, +.031], n2 −.005 [−.019, +.010], n3 −.007 [−.028, +.019]; qmodel n1/n2/n3 +.044 / +.044 / +.040 (intervals about [−.01 to −.02, +.10 to +.11]). The only significant gain is engine n2 in the first quarter: −.010 [−.020, −.002], MSE skill +.24 [+.07, +.43].
2. Even a perfect translation would not help with today's transmission. Fed the realized factor path, the base model is worse than no information at the window end: equity9 +.031 [−.017, +.084], MSE skill −.41 [−.76, −.01]. Q-oracle is about equal to no narrative (+.007). The engine's fan_cond (24-month realized-mean views) is the only clearly positive row: −.023 [−.041, −.006].
3. Model loss dominates translation loss. Equity9 at the window end, RMS: engine translation (narrative − cond) .07–.11, engine machinery (cond − oracle) .14, transmission (oracle − realized) .40; qmodel translation .11–.16 against model loss .38. The engine's pull toward history is anti-correlated with the transmission error (−.53 to −.67), which is why the narrative runs beat the engine oracle by .03–.04.
4. The misses are valuation-multiple swings. Across the 11 episodes, the realized CRSP P/D change explains 83% of the variance of window us_equity returns. The narrative runs' us_equity misses correlate −.81 to −.88 with it (r² .66–.78). In every down episode (E05 −.47, E12 −.38, E06 −.14, E10 −.02) the narrative medians have the wrong sign (+.04 to +.46), because rate cuts and lower yields load positively.
5. Today's engine beats the quarterly model on narratives: Q − E at the common end, equity9, +.056 / +.067 / +.064 for n1/n2/n3, every 90% interval above 0; the qmodel wins 1–3 of 11 episodes. Under oracle macro they tie (−.017 [−.075, +.021]). The qmodel is over-confident (mean |PIT − .5| .30–.32 against the engine's .25), and its completion of masked financial conditions is no better than unconditional, with the wrong sign in 2022 and 2007–09.
6. The scenario → factor layer of the engine under-delivers the narrated macro. At the window end the narrative runs reach 48–60% of the realized policy-rate move, 37–51% of unemployment, 33–40% of CPI y/y and 47–63% of oil. Entropy pooling shrank 32/52 (n1), 41/53 (n2) and 23/47 (n3) window phases (mean λ .79/.67/.84), and it meets mean views on skewed flows with mixtures: E10 n1 oil in months 7–12 has mean −.20, median −.04.
7. In 2022–23 no run reproduces the 2022 drawdown in its median (us_equity −.237). Engine narratives give +.02 to +.03, the qmodel narratives +.06; the best are the oracle-macro valuation block (−.084), F (−.049) and the engine oracle (−.031). The engine narratives do double the tail probability: P(&lt; −15%) is .18–.23 against .10 unconditional. The 2023 rebound (+.222) is partly caught (medians up to +.14). Energy's +46%, nasdaq's −40%/+43% and the multiple compression (P/Ê −.46 by 2022Q3) are missed.
8. The audit finds no asset or valuation leakage: no valuation or endogenous views, no asset_drift, no factor_drift, no analogs in the 36 specs, and no output legs in the 36 tables. Translator-chosen magnitudes that are not in any narrative: r_star overrides in all 36 specs, resid_scale in E05/E08, f_spread views inferred from words in E05/E08 (median |z| .75 from realized), and inferred d_gz/d_cpbill in E05/E08 n3.
9. The test is not blind. The period guesses are 12/12 exact (confidence .95–.99; .75–.99 from n3 alone), so accuracy on unrecognized episodes cannot be measured. The number-free n3 values are almost as close to realized as n1's numbers: median |z| .22 against .16 for spec views (no-information baseline .66–.76) and .25 against .21 for table fields (baseline .42). E02, whose n3 was translated before the protocol slip, is equally close.
10. The engine contexts reproduce the translator's prompts exactly: 52/52 variable-table rows match in all 12 episodes. Production post-processing made 0 repairs and 13 small clips. N3 duration mismatches (E02, E04, E06, E07, E11) cost nothing detectable.
11. There is a schema bug: the translator codes x_oil_cause 0 = supply / 1 = demand, while the qmodel's data use 1 = supply. I mapped it to d_oil_sup = d_oil × (1 − cause). Passing it raw would move medians by at most .05 (mean .01).
12. The schema's P/Ê valuation leg is distorted by EPS write-downs exactly in the episodes that matter: the E05 'multiple' is +1.00 while price fell .47, and E12 +.26 while price fell .38. CRSP P/D gives the clean identity: the multiple explains 83% of cross-episode us_equity variance.

## Open questions

1. Transmission is the binding loss: the base model fed the realized path is worse than no information at the window end, and the misses track the realized P/D change (r² .66–.78). Should the next round make the valuation multiple a scenario-stated input on CRSP P/D or CAPE, defaulting to the valuation block's inference (VB was the best oracle on us_equity at the end, .194), instead of a better macro translation? The gain would show only as an oracle upper bound, since realized valuation cannot be an input.
2. The engine's factor layer delivers only 40–70% of a narrated macro path: views are shrunk when infeasible, and phase means of skewed flows are met by mixtures (the median path misses the narrated oil fall). Should a narrated policy-rate or oil path be imposed as a level path (do-operator or path constraint) rather than through phase-mean views plus r_star? r_star overrides appear in all 36 specs, and the translator flagged that r_star also moves the 10y target.
3. The qmodel's completion of unstated financial conditions is no better than unconditional, with the wrong sign in 2022 and 2007–09. Should translations be required to state d_gz, d_y10 and d_usd, should masked E1 fields shrink to the training mean, or should the x-block get regime-specific covariances?
4. The schema's P/Ê valuation leg is distorted by write-downs (E05 'multiple' +1.0, E12 +.26 while prices fell). Should y_mult be redefined on CRSP P/D or CAPE before any narrative is allowed to state a valuation leg?
5. Blinding failed: 12/12 periods recognized, and the number-free n3 values are as close to realized as n1's numbers. A real blind test needs prompts without the T0 leak (sample length, era offsets, base years) and narratives the translator cannot date (perturbed or synthetic episodes, or a placebo: another window's prompt). Should the lead commission that before treating any narrative result as predictive evidence?
6. Which oracle should define 'model loss'? The engine oracle is beaten by its own narrative runs (the history pull offsets transmission errors); F and VB oracles are better on equities but use GZ, a bond-market price. Should the headline decomposition use F/VB as the oracle once the factor round lands?
7. The x_oil_cause conventions are opposite (translator 0 = supply, qmodel data 1 = supply); the impact here is small (at most .05). Which side should change in schema.json? Separately, sign-only and template_sign entries and all u_* (never numeric) are unusable by the plug-in: should it accept sign constraints?
8. Edge episodes: E06 can only run leave-window-out (look-ahead in fitting, reported apart; the qmodel cannot run it at all). E03 trains the qmodel on 26 quarters and the engine on 80 months (λ .05–.43). E08 needed a one-month shift for the qmodel. Conclusions do not change without E03. Should the lead's headline use the 10 clean episodes (excluding E03 as well as E06)?
9. CMA: these runs use CMA 'historical', as the backtest windows do. The production scenario runner uses 'valuation_adjusted', which this round did not test on the narratives.

## Files the role listed

Paths are relative to `scratchpad/sim_redesign/`. Only the files marked are in this archive; the rest (scripts, work outputs, data) stay in the research folder.

- `scorer/sc_common.py`
- `scorer/s02_engine.py`
- `scorer/s03_qmodel.py`
- `scorer/s04_vb.py`
- `scorer/s05_score.py`
- `scorer/s06_decomp.py`
- `scorer/s07_audit.py`
- `scorer/s08_deepdive.py`
- `scorer/s09_macro.py`
- `scorer/s10_tables.py`
- `scorer/s11_extra.py`
- `scorer/s12_valuation_misses.py`
- `scorer/s13_headline.py`
- `scorer/out/headline.json`
- `scorer/out/scoreboard_by_episode.csv`
- `scorer/out/scoreboard_groups.csv`
- `scorer/out/scoreboard.json`
- `scorer/out/score_rows.csv`
- `scorer/out/score_pooled.csv`
- `scorer/out/score_E06.csv`
- `scorer/out/paired_comparisons.csv`
- `scorer/out/decomposition_summary.csv`
- `scorer/out/decomposition_by_episode.csv`
- `scorer/out/decomposition_rows.csv`
- `scorer/out/macro_delivery_engine.csv`
- `scorer/out/macro_delivery_qmodel.csv`
- `scorer/out/valuation_misses.csv`
- `scorer/out/audit.json`
- `scorer/out/audit_specs.csv`
- `scorer/out/audit_views.csv`
- `scorer/out/audit_tables.csv`
- `scorer/out/audit_table_values.csv`
- `scorer/out/audit_guesses.csv`
- `scorer/out/check_context.csv`
- `scorer/out/check_closeness_baselines.csv`
- `scorer/out/check_oil_cause.csv`
- `scorer/out/deepdive_E10.json`
- `scorer/out/deepdive_E10_paths.csv`
- `scorer/work`
