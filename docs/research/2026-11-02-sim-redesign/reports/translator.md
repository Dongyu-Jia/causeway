# Stand-in translations (research round, role `translator`)

*What the role did:* 36 production specs and 36 quarterly tables written as the translator would, plus the period guesses.

Rendered from the role's final report in the research workflow's journal (run `wf_cc7fe8c3-afc`, result `translator`). The text is unchanged except that local absolute paths are written as the spec writes them (`scratchpad/sim_redesign/...`, `scratchpad/...`). It is the report the critic and the designer read; the numbers are the role's own, and the critique ([`critic/critique.md`](../critic/critique.md)) and the spec ([`SIM_DESIGN.md`](../SIM_DESIGN.md)) say which of them survived.

## Summary

I translated all 12 codes × 3 narratives, so nothing was left untranslatable. That is 36 production specs (spec_n1/n2/n3.json), 36 quarterly event tables (q_n1/n2/n3.json) and 12 period guesses (guess.json), all written into narratives/&lt;code>/.

Production specs:
- I treated each prompt.md as the system prompt and produced the exact SPEC_JSON_SCHEMA shape: 180 months, at most 8 phases, at most 12 views per phase, Chinese text, empty analogs.
- No view uses f_valuation, s_div_yield, f_cape or s_cape, nor the endogenous f_wealth or s_wealth_income, and there is no asset_drift. No narrative actually asked for a valuation view.
- Every spec passes my structural check against its own prompt's ranges (p2–p98 bounds, ⚠ states, identity triples, drift bounds). I did not run the project's validate_spec, because it loads the panel.

Quarterly tables:
- Every row carries all 52 L1/L2 x_q fields from schema-revise-2 plus the five shocks u_*. Anything the text does not determine is null.
- A sign the text determines goes in `sign`; a sign that comes only from a G-template is marked template_sign.
- Every value has a src code: stated, interp, derived (from yoy paths), band, or inferred (N3 only). Each table also has a units/endogeneity block, c_q (anticipated paths, regime, X09/X27 notes), cross-quarter constraints and an events list mapping each event to its template.

Scripts and checks are in sim_redesign/translator/. Guesses are below; I recognised every period with confidence ≥0.97, and even n3 alone was enough with ≥0.75.

## Key findings

1. THE TEST IS NOT BLIND. Each prompt.md gives the sample as 'T0−XyYm 至 T0 共 N 个月'. The model start (1973-02) is known, so T0 is exact. The era offsets and the base-year notes ('Y0+22年美元', 'Y0+13=100') also pin Y0. On top of that, every n3 describes historically unique events.
2. Guesses (T0, then narrative window): E01 2004-06 (2004-07..2006-12); E02 1995-06 (1995-07..1997-12); E03 1979-09 (1979-10..1982-08); E04 2014-06 (2014-07..2016-02); E05 2007-06 (2007-07..2009-06); E06 1973-09 (1973-10..1975-06); E07 1993-12 (1994-01..1995-06); E08 2020-01 (2020-02..2021-06); E09 2016-12 (2017-01..2019-12); E10 2021-12 (2022-01..2023-12); E11 1990-06 (1990-07..1991-12); E12 2000-12 (2001-01..2002-12). Confidence is 0.97–0.99 each.
3. MY PROTOCOL SLIPS, which contaminate the N3 runs. (1) E01 was the first code, and I translated its n3 after reading n1 and n2, so it is fully contaminated. (2) While working on E02, a diff command printed the first ~25 characters of every code's n1 '全期小结' line and n2 '金融条件' line. Examples: 'E04 油价 106 → 33', 'E10 政策利率 0.1% → 5.3%（第 20'. The n3 translations for E03–E12 were made after that, so their N3 magnitudes may be partly contaminated. Each q_n3 notes field says so. E02 n3 is clean. For E02–E12 I always translated n3 before opening n1 and n2.
4. ENGINE-SIDE LOOK-AHEAD IN E06. Its prompt sample is 'T0−7m 至 T0+53y'. The month library and the transmission model therefore contain the scenario window itself, so entropy pooling (EP) can draw the actual 1973–75 months. E03 also ships a prompt_lwo.md, which I did not translate; I used prompt.md only, as instructed.
5. N3 DURATIONS WERE INFERRED (from the words, plus the slips above) and often disagree with n1. Months inferred vs actual: E01 30 vs 30 (contaminated); E02 36 vs 30; E03 36 vs 35; E04 30 vs 20; E05 24 vs 24; E06 24 vs 21; E07 24 vs 18; E08 18 vs 17; E09 36 vs 36; E10 24 vs 24; E11 24 vs 18; E12 24 vs 24. The q_n3 tables therefore do not line up quarter by quarter with q_n1/q_n2 for E02, E04, E06, E07 and E11; the harness must align them or score the timing error. Partial last quarters exist in the n1/n2 tables: E03 q12, E04 q7 and E08 q6 cover 2 months each.
6. POLICY STANCE NEEDED THE r_star OVERRIDE ALMOST EVERYWHERE. The prompt says rate changes are scaled by '今天水平 ÷ 当时水平' and revert to r*+inflation. That makes narrated paths unreachable with f_ff views alone: E01 hiking from 1%; E04, E05, E08 and E10 holding at zero; E02, E07 and E11 holding a level. So most phases set r_star between −3 and 3 so that the reversion target sits near the narrated rate. The cost: the same r* also moves the 10y target (r*+infl+0.75). That fights inverted curves (E01 n2 conundrum, E03, E09, E10). I partly compensated with f_y10 views and explain this in each spec's narrative. The prompt does not say whether '今天水平' is the T0 level or the simulated level.
7. OUT-OF-SUPPORT SHOCKS. In E03 (only an 80-month pool), E05, E06, E08 and E11 the shocks exceed every historical 6-month block, so many views sit at the p2/p98 edges and will be truncated. In E03 the sample has no 6-month block with falling oil, so the narrated oil decline is left to the engine's oil reversion: no oil view in that phase. I also used resid_scale 1.5–2.0 for the panic and lockdown phases (E05, E08). Expect the engine to under-deliver shock amplitudes; that is not a translation choice.
8. INFLATION IN THE TABLES IS DERIVED. The narratives give yoy levels, so quarterly annualized goods/services/core/wage rates come from the smoothest monthly path that hits the stated yoy anchors, assuming an even pre-start year. The text only pins down 4-quarter sums. Some quarters are artifacts: E01 goods q10 +9.5%, E05 goods q8 +12.6%. They are flagged in the notes, and E01 also carries explicit 4-quarter constraints.
9. DO-OPERATOR HANDLING. n1 and n2 give full policy paths together with macro, so I filled the raw d_ff. u_* is filled only as a sign, and only for interventions the text names: emergency or intermeeting cuts, the E03 regime change, pre-emptive hikes in E07 and E09, QE/QT, tax laws, stimulus and checks, the consolidation deals. I never computed u_* values, because that needs the fitted reaction functions. Deficits are sign-only and left to the model whenever the narrative gives the fiscal components.
10. E1/E2 POLICY. In n1 and n3, E1/E2 fields are filled only when the text says them in words; otherwise they are template signs. I made two exceptions. (1) E05 and E08 describe panic and funding freezes, so q_n3 gets inferred d_gz/d_cpbill values, and the n1/n3 specs get inferred f_spread views. (2) I removed an inferred dollar view from E04 n3, because policy divergence is not a statement about the dollar. n2 specs add f_y10, f_spread and f_usd from the text, with views enlarged to offset scaling and reversion.
11. Other hard calls. x_oil_cause stays null wherever the text gives no cause (E02, E07 and several decline phases). 'Mixed' is coded 0.5 when the text names both demand and supply (E05, E08, E09, E10). E09 tariffs are sign-only in n1 and +0.2–0.3pp/quarter inferred in n3. Corporate tax cuts go in as d_corptax/u_ptax signs in the tables and as f_profit_share views in the specs. E12 accounting fraud is recorded as an X27 write-down note in c_q, not as a valuation view. E10's regional bank run follows G33, with no spread view. E11's coalition defense spending is a template sign only, because the text never says the home country fought. E03 credit controls are recorded as X09 in c_q, with a d_cons_goods sign.

## Open questions

1. Blinding failed: every prompt.md reveals its own T0, and the n3 events are recognisable. Should the lead rebuild the prompts with the sample length, era offsets and base-year notes hidden, and re-run the translation with an agent that has seen none of the n1/n2 text? Today the N1–N3 runs measure translation fidelity, not prediction from a blind description.
2. E06's month library and transmission model include the scenario window itself (T0−7m to T0+53y). Should the harness rebuild them on months outside the window, as E03's prompt_lwo.md does, before scoring E06?
3. The n3 quarter grids differ in length from n1/n2 for E02, E04, E06, E07 and E11. Should the harness align them on the narrative's own phase boundaries, or score the timing error as part of the N3 loss?
4. The u_* shocks appear only as signs. Should the harness compute them as d_ff (or d_deficit etc.) minus the fitted systematic part, or apply the sign-only interventions as bounded draws?
5. For per-quarter inflation, should the harness use my derived quarterly values or only the 4-quarter-sum constraints the text actually determines?
6. How should template_sign entries be used: as sign constraints on the joint model's completion, or ignored under N1/N3 masking?
7. My r_star overrides (−3 to 3) set the reversion targets for both the policy rate and the 10y. That makes the engine's 10y paths in n1 partly a consequence of my policy-stance choice. Should the engine expose separate ff and 10y anchors, or should the test score n1's 10y only under O-B or N2?

## Files the role listed

Paths are relative to `scratchpad/sim_redesign/`. Only the files marked are in this archive; the rest (scripts, work outputs, data) stay in the research folder.

- `narratives/E01`
- `narratives/E02`
- `narratives/E03`
- `narratives/E04`
- `narratives/E05`
- `narratives/E06`
- `narratives/E07`
- `narratives/E08`
- `narratives/E09`
- `narratives/E10`
- `narratives/E11`
- `narratives/E12`
- `translator/qtable.py`
- `translator/check_spec.py`
- `translator/paths.py`
- `translator/n3lex.py`
- `translator/final_check.py`
