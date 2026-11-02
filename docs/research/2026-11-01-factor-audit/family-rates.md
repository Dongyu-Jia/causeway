# Rates, inflation expectations, term premium, money and liquidity — candidate family report

Access date 2026-11-01. Harness v1.1, panel sha1 3869fdde3437 (last month 2026-09), model_selection sha1 58a9db10ed83. Nothing in the project directory was written. 42 series built (37 transmission candidates, 36 predictors) plus 9 mechanism (interaction) candidates. Every number below is read from `tests.json` / `work/*.json` by `scripts/s17_report.py`.

## Bottom line

1. **One rates mechanism earns its place out of sample: the change in the ACM 10-year term premium, active only when CPI inflation was above 4% the month before** (`tp10_x_infl4`, the project's own regime dummy). It raises the 10-asset score in every decade: +0.0048 / +0.0499 / +0.0097 (1995–2004 / 2005–14 / 2015–26; pooled DM t +1.47 / +1.62 / +0.88, combined one-sided p 0.011), every decade above the walk-forward's MIN_GAIN 0.003. The other regime definitions agree: CPI > 3% +0.0110/+0.0117/+0.0149, core CPI > 4% +0.0116/+0.0418/+0.0221 (p <0.001). The coefficient is negative for every equity sector (us_equity -0.039, HAC t -2.3) and positive for gold (+0.032, t +1.6). In high-inflation months, term-premium shocks hit equities beyond what the y10 change implies. That is the positive stock–bond correlation of 2022 and the 1970s. Neither the plain term premium (-0.0026/-0.0359/+0.0015) nor the expected-path part in the same regime (-0.0289/-0.0017/+0.0023) does this. Placebos are 200 noise series and 200 permutations in the same months; none reaches its total gain (share 0.000 and 0.000).
2. **No plain change factor from this family is a missing core factor.** Of the 37 transmission candidates, 1 meets the common-factor rule, and only weakly: `cle_ei1` (combined p 0.26). The 1y/2y yields, the 3-month bill, paper-bill spreads, M2 and bank credit do worse than the base in all three decades. Asset by asset, 106 of 460 non-mechanical candidate×asset pairs meet the letter of the rule. Within-family Benjamini–Hochberg (q = 0.10) keeps 14 of them, and all but one are regime interactions: `tp10_x_core4`: us_equity, industrials, nasdaq, energy, gold, financials, staples; `tp10_x_infl3`: industrials, gold, us_equity, nasdaq; `y10_x_infl4`: energy, gold; `mort_spread`: gold. The exception, mort_spread → gold, has a negative sign and no mechanism. A further 11 pairs survive on treasury_10y / crude_oil. Those gains are mechanical: the treasury price is built from the y10 path, and curve terms only fit roll-down and convexity.
3. **Forecast states (known-at, 12 months ahead, better than the KEY_STATES baseline in both halves):** real-estate-loan growth forecasts the change in CPI inflation (incremental R² +0.208 / +0.172, Clark–West t +4.46, BH q within family 0.0035). The result is robust to an extra month of lag, to dropping 2020–22 and to removing level breaks. First-release y/y (1997+) correlates 0.995 with today's, and the series forecasts core and services CPI as well. Bank credit → CPI (+0.223/+0.076) depends on 2020–22. The 1y–ff spread → Δff (+0.116/+0.086), Michigan 1y → CPI (+0.218/+0.006) and SPF 1y → Δy10 (+0.096/+0.072) are weaker. M2 → CPI (+0.203/+0.177) is **not robust**: it fails with a 1981 sample start and with first-release data.
4. **Gold's dedicated rate driver is the TIPS real yield measured like the gold price (monthly average).** Since 2003 it adds +0.316 to gold's 2015–26 R² on the same training months (DM t +2.47; full-sample HAC t -6.7). Measured at month-end it adds only +0.056. No honest pre-2003 proxy carries it: the Cleveland real 10y on the same timing adds about 0. It cannot be a core factor (it starts in 2003).
5. **A sample-window side finding.** Fitting the base model from 1982 instead of 1973 raises gold's OOS R² by +0.143 / +0.232 / +0.171. This is a window choice, not a factor. It is also why every late-starting candidate (Cleveland, Michigan, SPF, KW) seemed to help gold in the raw harness deltas. Judged on the same training months, those gains disappear.
6. **The y10 beta that flips around 2000 is an inflation-expectations beta.** With y10 split into Cleveland expected inflation and the real rate (1982+), us_equity's real-rate beta is stable (-0.018 → -0.037). The expected-inflation beta flips (-0.115, t -1.7 → +0.086, t +2.7). The ACM split does not stabilize: both parts flip (expected path -0.039 → +0.016, TP -0.053 → +0.000). The regime interaction in finding 1 is the version of this that survives out of sample.

## Ranked table

Transmission numbers: change in OOS R² (score = mean over the 10 score assets) for 1995–2004 / 2005–14 / 2015–26. Candidates that start after 1973 are compared with the base refitted on the same training months. t = pooled Diebold–Mariano t of the score, p = combined one-sided p. Asset entries show the best non-mechanical assets that meet the rule. Predictive: incremental OOS R² over KEY_STATES in 1985–2004 / 2005–2025 (monthly origins), with the Clark–West p on all origins. Coverage: first month of the monthly change; '(full)' = observed every month 1973-02..2026-09, as a core factor needs. Flag = harness endogeneity_check.

| # | candidate | mechanism | coverage | transmission | predictive (r2_incr h1/h2) | verdict | role | flag |
|---|---|---|---|---|---|---|---|---|
| 1 | tp10_x_infl4 | TP change x 1{CPI y/y > 4% last month} (project's regime dummy) | 1961-07 (ACM) (full) | score +0.005/+0.050/+0.010 (t +1.5/+1.6/+0.9; p 0.011); us_equity +0.008/+0.072/+0.018 (p 0.0092); gold +0.006/+0.050/+0.029 (p 0.012) | - | ADDS OOS: common, > 0.003 in every decade; note: pre-specified (the project's regime dummy); clears 0.003 in 1995-2004 by 0.0018 | core transmission factor (interaction term) | ok |
| 2 | tp10_x_core4 | TP change x 1{core CPI y/y > 4%} | 1961-07 (ACM) (full) | score +0.012/+0.042/+0.022 (t +2.1/+2.7/+2.7; p <0.001); us_equity +0.022/+0.062/+0.046 (p <0.001); industrials +0.019/+0.038/+0.024 (p <0.001) | - | ADDS OOS: common, > 0.003 in every decade; note: variant (core CPI); 1995-2014 test months all D = 0 | core transmission term (variant of #1) | ok |
| 3 | tp10_x_infl3 | TP change x 1{CPI y/y > 3%} | 1961-07 (ACM) (full) | score +0.011/+0.012/+0.015 (t +2.1/+0.4/+2.8; p 0.001); industrials +0.020/+0.015/+0.017 (p <0.001); gold +0.011/+0.017/+0.028 (p <0.001) | - | ADDS OOS: common, > 0.003 in every decade; note: variant (threshold 3%) | core transmission term (variant of #1) | ok |
| 4 | tp10_x_cpi_level | TP change x CPI y/y (continuous) | 1961-07 (ACM) (full) | score +0.003/+0.002/+0.010 (t +1.5/+0.2/+1.7; p 0.024); industrials +0.003/+0.002/+0.005 (p 0.018); gold +0.008/+0.005/+0.026 (p 0.026) | - | adds OOS by the rule (common); note: variant; 2005-14 below 0.003 | robustness variant | ok |
| 5 | tp10_x_infl5 | TP change x 1{CPI y/y > 5%} | 1961-07 (ACM) (full) | score +0.001/+0.003/+0.010 (t +0.4/+1.1/+1.5; p 0.045); industrials +0.001/+0.002/+0.005 (p 0.02); us_equity +0.001/+0.004/+0.016 (p 0.026) | - | adds OOS by the rule (common); note: variant; 1995-2004 below 0.003 | robustness variant | ok |
| 6 | re_loans | real-estate loan growth (housing credit) | 1947-02 (full) | score -0.003/-0.031/-0.007 (t -0.5/-2.2/-1.0; p 0.99) | cpi_yoy +0.208/+0.172 (CW p <0.001); oil +0.030/+0.221 (CW p <0.001); ret_crude_oil +0.030/+0.221 (CW p <0.001) | forecast state: cpi_yoy, oil, ret_crude_oil; note: robust to extra lag, dropping 2020-22, break adjustment and vintages; oil hits have no prior | forecast state | ok |
| 7 | bank_credit | bank credit growth | 1947-02 (full) | score -0.016/-0.003/-0.011 (t -2.7/-1.0/-1.2; p 1); energy +0.009/-0.019/+0.021 (p 0.11) | cpi_yoy +0.223/+0.076 (CW p <0.001); ret_crude_oil +0.087/+0.049 (CW p 0.0017); oil +0.087/+0.049 (CW p 0.0017) | forecast state: cpi_yoy, ret_crude_oil, oil; note: cpi: 2005-25 gain relies on 2020-22 | forecast state | ok |
| 8 | rstar | natural rate r* (HLW) | 1961-06 | - | ret_crude_oil +0.008/+0.115 (CW p 0.0013); oil +0.008/+0.115 (CW p 0.0013) | forecast state: ret_crude_oil, oil; note: hits only oil (no prior); its intended targets ff, y10, unrate fail | forecast state for oil only (no prior) |  |
| 9 | near_spread | y1-ff: priced policy change | 1962-01 | - | ff +0.116/+0.086 (CW p 0.0018); spread +0.010/+0.025 (CW p 0.062) | forecast state: ff; note: 2005-25 gain relies on 2022 | forecast state |  |
| 10 | slope_10_1 | slope y10-y1 (recession signal) | 1962-01 | - | ff +0.116/+0.086 (CW p 0.0018); spread +0.010/+0.025 (CW p 0.062) | forecast state: ff; note: same information as near_spread given the baseline | forecast state |  |
| 11 | mich1y | household 1y inflation expectations | 1978-02 | score -0.006/-0.001/-0.000 (t -1.8/-0.5/-0.1; p 0.92); real_estate +0.001/-0.000/+0.007 (p 0.12); financials +0.002/-0.002/+0.006 (p 0.13) | cpi_yoy +0.218/+0.006 (CW p 0.002); ret_gold +0.102/+0.006 (CW p 0.022); spread +0.033/+0.012 (CW p 0.084) | forecast state: cpi_yoy, ret_gold; note: 2005-25 gain only +0.006 | forecast state; scenario view variable | ok |
| 12 | mort_spread | mortgage rate - y10 (housing finance) | 1971-05 (full) | score -0.007/-0.007/+0.005 (t -1.1/-1.8/+0.9; p 0.87); gold +0.024/+0.099/+0.050 (p 0.0049) | - | adds OOS for gold; note: gold sign negative, no mechanism | none (sign without mechanism) | spanned by core factors |
| 13 | cle_ei10 | Cleveland 10y expected inflation | 1982-01 | score -0.002/-0.008/+0.006 (t -0.5/-3.0/+1.9; p 0.82); industrials +0.001/+0.012/+0.012 (p 0.035); us_equity -0.001/+0.007/+0.013 (p 0.18) | - | adds OOS for industrials | scenario view variable only | spanned by core factors |
| 14 | kw_tp10 | term premium (Kim-Wright, 1990+) | 1990-02 | score -0.108/-0.022/+0.009 (t -2.4/-2.5/+1.2; p 0.84); real_estate +0.014/+0.000/+0.001 (p 0.047); energy -0.060/+0.001/+0.020 (p 0.19) | - | adds OOS for real_estate; note: real_estate tiny; harness 'forecast state' rejected (h1 49 origins) | scenario view variable only | spanned by core factors |
| 15 | y10_x_infl3 | y10 change x 1{CPI y/y > 3%} | 1962-02 (full) | score -0.050/+0.046/+0.040 (t -2.7/+2.2/+3.1; p 0.071); us_equity -0.076/+0.077/+0.069 (p 0.044); gold -0.009/+0.026/+0.047 (p 0.051) | - | adds OOS by the rule (common); weak; note: 1995-2004 significantly worse (pooled t -2.7) | none | spanned by core factors |
| 16 | y10_x_infl4 | y10 change x 1{CPI y/y > 4%} | 1962-02 (full) | score -0.019/+0.051/+0.018 (t -2.8/+2.6/+1.8; p 0.17); energy +0.002/+0.019/+0.001 (p <0.001); gold +0.006/+0.036/+0.019 (p 0.0026) | - | adds OOS by the rule (common); weak; note: 1995-2004 significantly worse (pooled t -2.8) | none | spanned by core factors |
| 17 | ff_x_infl4 | ff change x 1{CPI y/y > 4%} | 1962-02 (full) | score +0.001/+0.004/-0.002 (t +0.1/+1.7/-0.4; p 0.22); nasdaq -0.002/+0.011/+0.012 (p 0.012); us_equity +0.001/+0.009/+0.000 (p 0.15) | - | adds OOS by the rule (common); weak; note: tiny | none | spanned by core factors |
| 18 | cle_ei1 | Cleveland 1y expected inflation | 1982-01 | score -0.015/+0.029/+0.015 (t -1.8/+1.8/+1.0; p 0.26); utilities +0.008/+0.040/+0.048 (p 0.017); us_equity -0.004/+0.041/+0.025 (p 0.047) | ret_nasdaq +0.000/+0.017 (CW p 0.15) | adds OOS by the rule (common); weak; note: weak | scenario view variable only | ok |
| 19 | real1y_spf | ex-ante real 1y rate y1 - SPF | 1970-05 | - | ret_healthcare +0.006/+0.213 (CW p 0.009); ret_financials +0.013/+0.055 (CW p 0.0093); ret_industrials +0.055/+0.083 (CW p 0.01) | forecast state: ret_healthcare, ret_financials, ret_industrials; note: equity targets, no prior | forecast state (weak, no prior) |  |
| 20 | curvature | curve hump 2*y5-y1-y10 | 1962-02 (full) | score -0.008/+0.027/-0.012 (t -2.5/+1.5/-0.6; p 0.82); staples +0.009/+0.045/-0.005 (p 0.063) | ret_healthcare +0.008/+0.186 (CW p 0.0095) | forecast state: ret_healthcare; note: healthcare only, no prior | forecast state (weak, no prior) | ok |
| 21 | m2real | real M2 growth | 1959-02 (full) | score -0.010/-0.011/-0.023 (t -0.9/-1.5/-1.8; p 0.99); real_estate +0.009/-0.009/+0.017 (p 0.32) | cpi_yoy +0.229/+0.182 (CW p 0.011); ret_crude_oil +0.022/+0.059 (CW p 0.037); oil +0.022/+0.059 (CW p 0.037) | forecast state: cpi_yoy, ret_crude_oil, oil; note: not robust (as m2) | none (not robust) | ok |
| 22 | spf_cpi1y | SPF 1y CPI expectations | 1981-09 | score -0.000/-0.003/+0.001 (t -0.1/-0.9/+0.4; p 0.63); real_estate +0.006/+0.012/+0.029 (p 0.053) | ip_yoy +0.002/+0.062 (CW p 0.015) | forecast state: ip_yoy; note: ip only; 1985-2004 +0.002 | forecast state; scenario view variable | ok |
| 23 | m2 | M2 growth (liquidity) | 1959-02 (full) | score -0.009/-0.003/-0.045 (t -1.2/-1.6/-1.5; p 0.99); real_estate +0.001/-0.011/+0.013 (p 0.44) | cpi_yoy +0.203/+0.177 (CW p 0.015); oil +0.006/+0.060 (CW p 0.051); ret_crude_oil +0.006/+0.060 (CW p 0.051) | forecast state: cpi_yoy; note: not robust (1981 start, first releases) | none (not robust) | ok |
| 24 | deposits | bank deposit growth (funding) | 1973-02 (full) | score -0.002/+0.002/-0.010 (t -0.6/+0.9/-2.0; p 0.83); energy +0.010/-0.005/+0.018 (p 0.25); nasdaq +0.000/+0.011/-0.009 (p 0.45) | cpi_yoy +0.081/+0.090 (CW p 0.019) | forecast state: cpi_yoy; note: 2005-25 gain relies on 2020-22 | forecast state | ok |
| 25 | spf_pgdp1y | SPF 1y deflator expectations (1970+) | 1970-06 (full) | score +0.002/-0.003/-0.005 (t +0.6/-0.3/-1.4; p 0.74); nasdaq -0.000/+0.002/+0.000 (p 0.4) | y10 +0.096/+0.072 (CW p 0.048); ret_staples +0.012/+0.090 (CW p 0.048); ret_us_equity +0.030/+0.026 (CW p 0.051) | forecast state: y10, ret_staples; note: weak (CW p 0.048 for y10) | forecast state | ok |
| 26 | tips10_avg | TIPS real yield, monthly average (gold timing) | 2003-02 | score n/a/n/a/+0.024 (t n/a/n/a/+2.1; p 0.017) | - | not testable in 2 decades (history); note: gold 2015-26 +0.32 vs same-sample base; history too short | gold-specific; scenario view variable (2003+) | partly market-driven |
| 27 | be10_avg | breakeven, monthly average (gold timing) | 2003-02 | score n/a/n/a/+0.023 (t n/a/n/a/+1.4; p 0.085) | - | not testable in 2 decades (history); note: gold +0.12 in 2015-26; history too short | scenario view variable (2003+) | partly market-driven; spanned by core factors |
| 28 | tips10 | 10y TIPS real yield (month-end) | 2003-02 | score n/a/n/a/-0.004 (t n/a/n/a/-0.3; p 0.63) | - | not testable in 2 decades (history) | scenario view variable only | partly market-driven; spanned by core factors |
| 29 | be10 | 10y breakeven | 2003-02 | score n/a/n/a/-0.004 (t n/a/n/a/-0.3; p 0.63) | - | not testable in 2 decades (history) | scenario view variable only | partly market-driven; spanned by core factors |
| 30 | be5y5y | 5y5y breakeven (anchoring) | 2003-02 | score n/a/n/a/-0.011 (t n/a/n/a/-0.8; p 0.78) | - | not testable in 2 decades (history) | scenario view variable only | ok |
| 31 | shadow_gap | shadow rate at the zero bound (Wu-Xia) | 1947-03 (full) | score n/a/n/a/+0.003 (t n/a/n/a/+0.3; p 0.38) | - | not testable in 2 decades (history); note: harness labels it a forecast state for ff by a numerical zero (issue 1) | none | ok |
| 32 | fed_assets | Fed total assets (QE/QT) | 2003-02 | score n/a/n/a/-0.017 (t n/a/n/a/-1.0; p 0.85) | - | not testable in 2 decades (history) | scenario view variable only | spanned by core factors |
| 33 | net_liq | Fed assets - TGA - RRP | 2003-03 | score n/a/n/a/-0.017 (t n/a/n/a/-1.1; p 0.87) | - | not testable in 2 decades (history) | scenario view variable only | ok |
| 34 | cre_loans | CRE loans (REIT financing) | 2004-07 | score n/a/n/a/-0.015 (t n/a/n/a/-1.5; p 0.94) | - | not testable in 2 decades (history) | scenario view variable only | ok |
| 35 | acm_tp10 | term premium (ACM): discount-rate news that is not expected policy | 1961-07 (full) | score -0.003/-0.036/+0.002 (t -0.4/-1.7/+0.8; p 0.76); utilities +0.008/-0.012/+0.004 (p 0.13) | - | no OOS gain | none | ok |
| 36 | y1 | 1y yield: policy news beyond the month-average ff | 1962-02 (full) | score -0.022/-0.009/-0.011 (t -2.7/-2.8/-0.8; p 1) | - | no OOS gain | none | spanned by core factors |
| 37 | y2 | 2y yield: expected policy path | 1962-02 (full) | score -0.009/-0.028/-0.009 (t -1.6/-3.2/-1.0; p 1) | - | no OOS gain | none | spanned by core factors |
| 38 | y5 | 5y yield: belly / corporate & REIT funding | 1962-02 (full) | score +0.002/-0.036/+0.001 (t +0.4/-2.1/+0.9; p 0.68) | - | no OOS gain | none | spanned by core factors |
| 39 | bill3m | 3m bill at month-end | 1954-02 (full) | score -0.042/-0.067/+0.001 (t -2.6/-2.5/+0.2; p 1) | - | no OOS gain | none | ok |
| 40 | cle_real10 | Cleveland 10y real rate | 1982-01 | score -0.001/-0.031/-0.006 (t -0.1/-3.1/-1.3; p 1) | - | no OOS gain | scenario view variable only | spanned by core factors |
| 41 | cle_real1 | Cleveland 1y real rate | 1982-01 | score -0.000/-0.032/+0.008 (t -0.1/-2.8/+0.6; p 0.91); real_estate -0.001/+0.002/+0.015 (p 0.081) | ret_nasdaq +0.001/+0.002 (CW p 0.19) | no OOS gain | scenario view variable only | ok |
| 42 | mich5y | household 5-10y expectations | 1990-05 | score +0.003/-0.029/-0.001 (t +1.1/-1.2/-0.1; p 0.84) | - | no OOS gain | scenario view variable only | ok |
| 43 | spf_cpi10y | SPF 10y CPI expectations | 1991-12 | score n/a/-0.002/-0.010 (t n/a/-0.4/-1.7; p 0.93); real_estate n/a/+0.003/+0.009 (p 0.076) | - | no OOS gain | scenario view variable only | ok |
| 44 | baa_aaa | default premium BAA-AAA (splits the core spread) | 1947-02 (full) | score +0.000/-0.009/+0.001 (t +0.1/-1.7/+0.2; p 0.8); real_estate +0.000/+0.001/+0.014 (p 0.11); us_equity -0.000/+0.010/+0.011 (p 0.11) | - | no OOS gain | none | ok |
| 45 | mbase | monetary base (QE proxy, 1959+) | 1959-02 (full) | score -0.001/+0.027/-0.039 (t -0.3/+0.5/-3.1; p 0.95); energy +0.000/+0.000/+0.012 (p 0.28) | - | no OOS gain | none | ok |
| 46 | ci_loans | C&I loan growth | 1947-02 (full) | score -0.017/-0.026/-0.031 (t -2.6/-2.3/-1.1; p 1) | - | no OOS gain | none | ok |
| 47 | cons_credit | consumer credit growth | 1947-02 | score -0.005/-0.010/-0.000 (t -1.3/-1.9/-0.2; p 0.97); healthcare +0.000/+0.011/+0.029 (p 0.053); real_estate -0.002/+0.001/+0.011 (p 0.25) | - | no OOS gain | scenario view variable only | ok |
| 48 | m2_first | M2 y/y, first releases (vintage check) | 1981-02 | - | - | no OOS gain | none |  |
| 49 | paper_bill_fin | financial CP - T-bill (funding liquidity) | 1971-05 (full) | score -0.030/-0.065/-0.005 (t -2.8/-1.4/-1.8; p 1); gold -0.012/+0.018/+0.012 (p 0.16) | - | no OOS gain | none | ok |
| 50 | paper_bill_nf | nonfinancial CP - T-bill | 1971-05 (full) | score -0.028/-0.031/-0.017 (t -2.8/-1.6/-1.7; p 1); gold -0.011/+0.016/+0.019 (p 0.1) | - | no OOS gain | none | ok |
| 51 | rny10_x_infl4 | expected-path part of y10 x 1{CPI > 4%} | 1961-07 (ACM) (full) | score -0.029/-0.002/+0.002 (t -2.7/-3.2/+0.8; p 1); gold +0.000/+0.005/-0.005 (p 0.12) | - | no OOS gain; note: expected-path part: fails | none | ok |

## Verdict rules (as in the task)

* **Adds out of sample, asset-specific:** over the decades with ≥ 120 training months, delta > 0 in at least 2, mean delta > 0 and mean DM (Newey–West) t > 0. **Common:** the same on the 10-asset score with the pooled DM t of the score series D_t = mean_a (e_base² − e_cand²)/SST_a. That series sums to the harness score_delta; the replication matches it to 5e-16. **Predictor:** incremental OOS R² > 0 in both halves, needing ≥ 60 origins per half and a non-degenerate h1 (harness issue 1 below).
* Late-starting candidates: the harness `delta` compares a candidate fitted on its own (short, late) months with a base fitted from 1973. I use `delta_same_sample` instead, with DM and Clark–West recomputed against that same-sample base. The replication matches the harness delta_same_sample to 2e-15.
* Every p value I computed is in `pvalues.csv` (21410 rows): one-sided DM, Clark–West and the combined z per decade and asset, the pooled score, residual correlations and predictive DM/CW. Use them for BH across families. Within this family alone (BH q = 0.10) the surviving non-mechanical pairs are the term-premium × inflation interactions, and the predictive pairs are re_loans → cpi_yoy and bank_credit → cpi_yoy.

## 1. Term premium × inflation regime (the one adoption candidate)

Definition (all from data already in or next to the panel): `f_tp10 = Δ ACMTP10` (month-end, pp; NY Fed ACM monthly sheet), `D_t = 1{s_cpi_yoy[t−1] > 4}`. This is the same lagged dummy `ModelSpec(regime='inflation')` uses. Candidate column = `f_tp10 × D_t`, added to every asset's equation on top of the 10 core factors.

| variant | 1995–2004 | 2005–14 | 2015–26 | pooled t | combined p | BH q (family) | > 0.003 every decade |
|---|---|---|---|---|---|---|---|
| `y10_x_infl4` | -0.0194 | +0.0510 | +0.0181 | -2.80 / +2.63 / +1.84 | 0.17 | 0.88 | no |
| `y10_x_infl3` | -0.0503 | +0.0456 | +0.0399 | -2.68 / +2.17 / +3.06 | 0.071 | 0.44 | no |
| `ff_x_infl4` | +0.0006 | +0.0043 | -0.0020 | +0.09 / +1.68 / -0.41 | 0.22 | 1 | no |
| `tp10_x_infl4` | +0.0048 | +0.0499 | +0.0097 | +1.47 / +1.62 / +0.88 | 0.011 | 0.14 | yes |
| `tp10_x_infl3` | +0.0110 | +0.0117 | +0.0149 | +2.12 / +0.41 / +2.82 | 0.001 | 0.019 | yes |
| `tp10_x_infl5` | +0.0007 | +0.0034 | +0.0097 | +0.38 / +1.06 / +1.48 | 0.045 | 0.34 | no |
| `tp10_x_core4` | +0.0116 | +0.0418 | +0.0221 | +2.14 / +2.73 / +2.74 | <0.001 | <0.001 | yes |
| `tp10_x_cpi_level` | +0.0032 | +0.0016 | +0.0104 | +1.53 / +0.17 / +1.73 | 0.024 | 0.22 | no |
| `rny10_x_infl4` | -0.0289 | -0.0017 | +0.0023 | -2.69 / -3.23 / +0.77 | 1 | 1 | no |

Per asset (`tp10_x_infl4`):

| asset | Δ 1995–2004 | Δ 2005–14 | Δ 2015–26 | DM t | coefficient (full sample) | HAC t |
|---|---|---|---|---|---|---|
| us_equity | +0.0084 | +0.0719 | +0.0181 | +1.39 / +1.61 / +1.08 | -0.0391 | -2.30 |
| nasdaq | +0.0041 | +0.0754 | +0.0136 | +0.93 / +1.81 / +0.64 | -0.0543 | -2.53 |
| energy | +0.0044 | +0.0350 | +0.0223 | +0.67 / +2.29 / +0.91 | -0.0431 | -2.40 |
| industrials | +0.0067 | +0.0399 | +0.0053 | +1.88 / +1.50 / +0.50 | -0.0380 | -1.81 |
| staples | +0.0038 | +0.0453 | -0.0258 | +1.71 / +1.55 / -1.81 | -0.0162 | -1.09 |
| healthcare | +0.0025 | +0.0983 | +0.0136 | +0.42 / +1.47 / +0.59 | -0.0346 | -1.81 |
| utilities | +0.0002 | +0.0222 | +0.0138 | +0.10 / +1.50 / +1.16 | -0.0173 | -1.34 |
| financials | +0.0081 | +0.0502 | +0.0016 | +1.38 / +1.48 / +0.10 | -0.0387 | -2.01 |
| real_estate | +0.0039 | +0.0107 | +0.0057 | +1.67 / +1.08 / +0.63 | -0.0215 | -2.11 |
| gold | +0.0059 | +0.0499 | +0.0290 | +0.86 / +1.77 / +1.26 | +0.0323 | +1.58 |
| crude_oil | -0.0000 | +0.0022 | -0.0008 | -0.22 / +1.18 / -0.84 | +0.0059 | +0.74 |
| treasury_10y | +0.0007 | +0.0023 | -0.0027 | +5.07 / +2.52 / -1.44 | +0.0023 | +1.09 |

* **Where the gain comes from.** Months with D = 1 in the test windows: 0 in 1995–2004, 14 in 2005–14 and 27 in 2015–26. Those months carry 63% and 66% of the 2005–14 and 2015–26 gains. The rest, and all of 1995–2004, comes from cleaner normal-regime betas: the 1973–94 high-inflation term-premium shocks no longer distort the y10 and other coefficients. The largest single months are 2008-10, 2008-09 and 2008-11 (CPI y/y was above 4% until October 2008), then 2020-03/04 and 2022-07/10/12 (`work/tp_regime.json`).
* **Rolling test windows:** 10-year test windows starting 1990, 1995, …, 2015 (trained from 1973): score Δ +0.0049 / +0.0048 / +0.0341 / +0.0499 / +0.0127 / +0.0118. All six are positive. The plain y10 × regime term is negative in the first three windows.
* **Placebos (200 each):** Gaussian noise with the TP's sd in the same months reaches the real candidate in 1.0% / 0.0% / 0.5% of draws. The TP permuted within the regime months does so in 1.5% / 0.0% / 2.0%. For core CPI > 4% it is 0% everywhere. The gain is not 'any variable in high-inflation months'.
* **Residuals:** correlation with PC1 of the out-of-sample residuals -0.20 (t -3.4). On 12-month sums it is -0.54 (t -3.3), significant in both halves for us_equity, staples, utilities. PC1 is the miss map's 'outside driver of the equity multiple'; this term explains part of it. Endogeneity check: ok (R² on same-month asset returns 0.060).
* **Independent term-premium model:** Kim–Wright (1990+) × the same dummy, judged on the same training months: +0.0483 / +0.0413 / +0.0476. 1995–2004 has only 59 training months, so it does not count.
* **Engine fit:** ACM starts 1961-06, so the term is observed in every MonthDB month (1973-02..). In a simulation the TP change comes with the drawn historical month, like any non-simulated column, and the dummy can be taken from the simulated CPI path (as `regime='inflation'` does). Role: **core transmission term**, an interaction on one factor rather than the full regime spec the walk-forward rejected. A scenario view can also target the TP itself.
* **Caveats.** ACM is a model estimate: today's vintage, with parameters fitted on 1961–2026. The finding comes out of 13 regime specifications I tried. The pre-specified one (the project's > 4% dummy) clears MIN_GAIN in 1995–2004 by only 0.0018, and within-family BH gives it q = 0.14. The 3% and core variants are stronger (q 0.019 and <0.001). The greedy walk-forward (selection on validation, gate on the untouched test) still has to confirm it with the other accepted changes.

## 2. Plain rates, expectations, money and credit factors (transmission)

Same-month changes on top of the 10 core factors: none adds in two of three decades with a positive mean t, except `cle_ei1` (weak; its best assets are utilities and us_equity). Highlights (full tables in tests.json → verdicts_common, verdicts_asset):

* `acm_tp10`: score -0.0026 / -0.0359 / +0.0015 (pooled t -0.4 / -1.7 / +0.8); endogeneity: ok, R² on core factors 0.42.
* `y1`: score -0.0225 / -0.0089 / -0.0109 (pooled t -2.7 / -2.8 / -0.8); endogeneity: spanned by core factors, R² on core factors 0.70.
* `y2`: score -0.0087 / -0.0275 / -0.0086 (pooled t -1.6 / -3.2 / -1.0); endogeneity: spanned by core factors, R² on core factors 0.79.
* `bill3m`: score -0.0425 / -0.0669 / +0.0008 (pooled t -2.6 / -2.5 / +0.2); endogeneity: ok, R² on core factors 0.50.
* `curvature`: score -0.0075 / +0.0266 / -0.0119 (pooled t -2.5 / +1.5 / -0.6); endogeneity: ok, R² on core factors 0.16.
* `baa_aaa`: score +0.0001 / -0.0094 / +0.0011 (pooled t +0.1 / -1.7 / +0.2); endogeneity: ok, R² on core factors 0.40.
* `mort_spread`: score -0.0071 / -0.0070 / +0.0055 (pooled t -1.1 / -1.8 / +0.9); endogeneity: spanned by core factors, R² on core factors 0.64.
* `m2`: score -0.0094 / -0.0035 / -0.0449 (pooled t -1.2 / -1.6 / -1.5); endogeneity: ok, R² on core factors 0.14.
* `mbase`: score -0.0012 / +0.0265 / -0.0395 (pooled t -0.3 / +0.5 / -3.1); endogeneity: ok, R² on core factors 0.23.
* `bank_credit`: score -0.0158 / -0.0026 / -0.0106 (pooled t -2.7 / -1.0 / -1.2); endogeneity: ok, R² on core factors 0.11.
* `re_loans`: score -0.0033 / -0.0314 / -0.0066 (pooled t -0.5 / -2.2 / -1.0); endogeneity: ok, R² on core factors 0.11.
* `deposits`: score -0.0016 / +0.0020 / -0.0103 (pooled t -0.6 / +0.9 / -2.0); endogeneity: ok, R² on core factors 0.09.
* `paper_bill_fin`: score -0.0302 / -0.0649 / -0.0048 (pooled t -2.8 / -1.4 / -1.8); endogeneity: ok, R² on core factors 0.30.
* `cle_ei1`: score -0.0154 / +0.0293 / +0.0154 (pooled t -1.8 / +1.8 / +1.0); endogeneity: ok, R² on core factors 0.16.
* `cle_ei10`: score -0.0020 / -0.0078 / +0.0059 (pooled t -0.5 / -3.0 / +1.9); endogeneity: spanned by core factors, R² on core factors 0.70.
* `mich1y`: score -0.0056 / -0.0013 / -0.0004 (pooled t -1.8 / -0.5 / -0.1); endogeneity: ok, R² on core factors 0.10.
* `spf_pgdp1y`: score +0.0018 / -0.0032 / -0.0054 (pooled t +0.6 / -0.3 / -1.4); endogeneity: ok, R² on core factors 0.03.

Late series, 2015–26 only, against the same-sample base: `tips10` -0.0036; `be10` -0.0036; `be5y5y` -0.0108; `tips10_avg` +0.0242; `be10_avg` +0.0227; `fed_assets` -0.0172; `net_liq` -0.0167; `cre_loans` -0.0151; `shadow_gap` +0.0025. QE balance-sheet measures (Fed assets, net liquidity) do not explain same-month returns beyond the core factors. For 2003+ series, 1995–2004 has no data and 2005–14 has under 120 training months.

## 3. Asset by asset: the most specific drivers tried

| asset | candidate | Δ 1995–2004 | Δ 2005–14 | Δ 2015–26 | combined p | full-sample HAC t | rule |
|---|---|---|---|---|---|---|---|
| gold | `tips10_avg` | n/a | n/a | +0.316 | 0.0067 | -6.7 | — |
| gold | `be10_avg` | n/a | n/a | +0.115 | 0.035 | +4.1 | — |
| gold | `tips10` | n/a | n/a | +0.056 | 0.1 | -1.6 | — |
| gold | `cle_real10` | -0.021 | +0.026 | -0.024 | 0.35 | +1.8 | — |
| gold | `mich1y` | -0.037 | -0.006 | -0.017 | 0.87 | +2.7 | — |
| gold | `m2` | -0.027 | +0.049 | -0.294 | 0.77 | +1.8 | — |
| gold | `mbase` | +0.017 | -0.182 | -0.003 | 0.87 | -0.7 | — |
| gold | `fed_assets` | n/a | n/a | -0.044 | 0.82 | -1.2 | — |
| gold | `mort_spread` | +0.024 | +0.099 | +0.050 | 0.0049 | -2.2 | meets |
| gold | `tp10_x_infl4` | +0.006 | +0.050 | +0.029 | 0.012 | +1.6 | meets |
| nasdaq | `tips10` | n/a | n/a | +0.009 | 0.38 | -2.7 | — |
| nasdaq | `be5y5y` | n/a | n/a | +0.017 | 0.16 | +2.0 | — |
| nasdaq | `acm_tp10` | -0.004 | -0.047 | +0.003 | 0.78 | -1.1 | — |
| nasdaq | `kw_tp10` | -0.121 | -0.155 | +0.003 | 0.89 | -0.4 | — |
| nasdaq | `y2` | -0.006 | -0.036 | +0.000 | 0.93 | +0.1 | — |
| nasdaq | `tp10_x_infl4` | +0.004 | +0.075 | +0.014 | 0.026 | -2.5 | meets |
| nasdaq | `tp10_x_core4` | +0.012 | +0.064 | +0.049 | <0.001 | -3.0 | meets |
| real_estate | `mort_spread` | +0.001 | -0.009 | -0.007 | 0.53 | -0.3 | — |
| real_estate | `re_loans` | +0.002 | +0.010 | -0.013 | 0.36 | -2.1 | — |
| real_estate | `cre_loans` | n/a | n/a | -0.060 | 0.96 | -1.0 | — |
| real_estate | `baa_aaa` | +0.000 | +0.001 | +0.014 | 0.11 | -1.7 | meets |
| real_estate | `y5` | -0.001 | -0.007 | +0.001 | 0.27 | +0.5 | — |
| real_estate | `acm_tp10` | +0.002 | -0.002 | +0.001 | 0.15 | -0.4 | — |
| real_estate | `tp10_x_infl4` | +0.004 | +0.011 | +0.006 | 0.025 | -2.1 | meets |
| utilities | `baa_aaa` | +0.023 | -0.112 | -0.030 | 0.58 | +1.0 | — |
| utilities | `acm_tp10` | +0.008 | -0.012 | +0.004 | 0.13 | -1.2 | meets |
| utilities | `y5` | -0.012 | -0.001 | +0.000 | 0.85 | +0.0 | — |
| utilities | `tips10` | n/a | n/a | -0.035 | 0.75 | -2.5 | — |
| utilities | `cle_ei1` | +0.008 | +0.040 | +0.048 | 0.017 | +3.6 | meets |
| utilities | `tp10_x_infl4` | +0.000 | +0.022 | +0.014 | 0.056 | -1.3 | meets |
| financials | `y1` | -0.029 | -0.011 | -0.003 | 1 | -2.8 | — |
| financials | `y2` | -0.012 | -0.030 | -0.011 | 1 | -1.9 | — |
| financials | `curvature` | -0.013 | +0.030 | -0.002 | 0.52 | +2.8 | — |
| financials | `deposits` | -0.001 | -0.001 | -0.003 | 0.94 | -0.0 | — |
| financials | `bank_credit` | -0.005 | -0.015 | -0.007 | 0.98 | -1.0 | — |
| financials | `ci_loans` | -0.005 | -0.034 | -0.032 | 1 | -1.4 | — |
| financials | `paper_bill_fin` | -0.047 | -0.075 | -0.010 | 1 | -0.8 | — |
| financials | `baa_aaa` | +0.002 | -0.022 | +0.016 | 0.27 | -1.2 | — |
| financials | `tp10_x_infl4` | +0.008 | +0.050 | +0.002 | 0.044 | -2.0 | meets |
| us_equity | `acm_tp10` | -0.012 | -0.050 | +0.001 | 0.92 | -0.7 | — |
| us_equity | `cle_ei1` | -0.004 | +0.041 | +0.025 | 0.047 | +3.6 | meets |
| us_equity | `m2` | -0.011 | -0.018 | -0.025 | 0.99 | +0.4 | — |
| us_equity | `paper_bill_fin` | -0.035 | -0.085 | -0.014 | 1 | -1.1 | — |
| us_equity | `tp10_x_infl4` | +0.008 | +0.072 | +0.018 | 0.0092 | -2.3 | meets |
| us_equity | `tp10_x_core4` | +0.022 | +0.062 | +0.046 | <0.001 | -3.0 | meets |
| treasury_10y | `acm_tp10` | +0.001 | +0.005 | +0.005 | <0.001 | -4.0 | meets (mechanical) |
| treasury_10y | `y1` | +0.008 | +0.019 | +0.013 | <0.001 | +7.0 | meets (mechanical) |
| treasury_10y | `curvature` | +0.006 | +0.009 | +0.009 | <0.001 | -6.9 | meets (mechanical) |
| treasury_10y | `spf_pgdp1y` | +0.000 | +0.000 | +0.001 | <0.001 | +1.7 | meets (mechanical) |

Late candidates are shown against the same-sample base. Gold's `mort_spread` meets the rule (p 0.0049), but with a negative sign and no clear mechanism. Treat it as a pointer only. Sources not reachable or not allowed: utility bond yields, CMBS spreads, bank deposit rates / NIM, central-bank gold purchases.

## 4. Forecast states (predictive_test, known-at alignment)

| candidate → target | r2_incr 1985–2004 | r2_incr 2005–25 | CW t (all) | BH q (family) | year-end origins h1/h2 | in-sample t |
|---|---|---|---|---|---|---|
| `re_loans` → cpi_yoy | +0.208 | +0.172 | +4.46 | 0.0035 | +0.165 / +0.161 | +4.4 |
| `bank_credit` → cpi_yoy | +0.223 | +0.076 | +3.62 | 0.055 | +0.189 / +0.158 | +4.0 |
| `re_loans` → oil | +0.030 | +0.221 | +3.11 | 0.12 | +0.010 / +0.195 | +4.3 |
| `re_loans` → ret_crude_oil | +0.030 | +0.221 | +3.11 | 0.12 | +0.010 / +0.195 | +4.3 |
| `rstar` → ret_crude_oil | +0.008 | +0.115 | +3.03 | 0.12 | -0.007 / +0.047 | +3.9 |
| `rstar` → oil | +0.008 | +0.115 | +3.03 | 0.12 | -0.007 / +0.047 | +3.9 |
| `bank_credit` → ret_crude_oil | +0.087 | +0.049 | +2.94 | 0.12 | +0.100 / +0.033 | +3.3 |
| `bank_credit` → oil | +0.087 | +0.049 | +2.94 | 0.12 | +0.100 / +0.033 | +3.3 |
| `near_spread` → ff | +0.116 | +0.086 | +2.93 | 0.12 | +0.065 / +0.051 | +2.3 |
| `slope_10_1` → ff | +0.116 | +0.086 | +2.93 | 0.12 | +0.065 / +0.051 | -2.3 |
| `mich1y` → cpi_yoy | +0.218 | +0.006 | +2.90 | 0.12 | +0.219 / +0.129 | +3.4 |
| `real1y_spf` → ret_healthcare | +0.006 | +0.213 | +2.37 | 0.25 | +0.022 / +0.100 | +3.6 |
| `real1y_spf` → ret_financials | +0.013 | +0.055 | +2.36 | 0.25 | +0.025 / +0.016 | +3.3 |
| `curvature` → ret_healthcare | +0.008 | +0.186 | +2.35 | 0.25 | +0.012 / +0.186 | -3.8 |
| `real1y_spf` → ret_industrials | +0.055 | +0.083 | +2.32 | 0.25 | +0.107 / -0.019 | +4.1 |
| `m2real` → cpi_yoy | +0.229 | +0.182 | +2.29 | 0.25 | +0.204 / +0.163 | +6.0 |
| `re_loans` → ret_gold | +0.005 | +0.034 | +2.27 | 0.25 | +0.013 / +0.031 | +2.6 |
| `real1y_spf` → ret_us_equity | +0.014 | +0.149 | +2.27 | 0.25 | +0.052 / +0.044 | +4.2 |
| `spf_cpi1y` → ip_yoy | +0.002 | +0.062 | +2.19 | 0.25 | -0.005 / +0.091 | +2.1 |
| `m2` → cpi_yoy | +0.203 | +0.177 | +2.18 | 0.25 | +0.170 / +0.148 | +5.9 |
| `deposits` → cpi_yoy | +0.081 | +0.090 | +2.08 | 0.29 | +0.031 / +0.116 | +3.5 |
| `mich1y` → ret_gold | +0.102 | +0.006 | +2.03 | 0.32 | +0.089 / +0.008 | +1.9 |
| `real1y_spf` → ret_energy | +0.094 | +0.011 | +2.00 | 0.32 | +0.164 / -0.044 | +3.2 |
| `bank_credit` → ret_healthcare | +0.023 | +0.058 | +1.86 | 0.42 | +0.077 / +0.143 | -2.0 |
| `m2real` → ret_crude_oil | +0.022 | +0.059 | +1.80 | 0.43 | +0.042 / +0.019 | +3.6 |
| `m2real` → oil | +0.022 | +0.059 | +1.80 | 0.43 | +0.042 / +0.019 | +3.6 |
| `bank_credit` → ret_us_equity | +0.020 | +0.035 | +1.73 | 0.48 | +0.015 / +0.078 | -2.1 |
| `spf_pgdp1y` → y10 | +0.096 | +0.072 | +1.67 | 0.49 | +0.055 / +0.114 | +2.4 |
| `spf_pgdp1y` → ret_staples | +0.012 | +0.090 | +1.67 | 0.49 | +0.012 / +0.060 | -2.1 |

Robustness of the main hits (`work/extra.json → predictive_robustness`, `work/regime.json`, `work/cpi_parts.json`):

| candidate → target | variant | all | h1 | h2 | h2 without 2020–22 | cand vs mean h1 / h2 |
|---|---|---|---|---|---|---|
| `re_loans` → cpi_yoy | as_tested | +0.185 | +0.208 | +0.172 | +0.289 | -0.600 / +0.174 |
| `re_loans` → cpi_yoy | extra_lag_1m | +0.174 | +0.178 | +0.172 | +0.284 | -0.661 / +0.175 |
| `bank_credit` → cpi_yoy | as_tested | +0.129 | +0.223 | +0.076 | -0.002 | -0.570 / +0.078 |
| `bank_credit` → cpi_yoy | extra_lag_1m | +0.124 | +0.200 | +0.082 | +0.010 | -0.616 / +0.084 |
| `m2` → cpi_yoy | as_tested | +0.187 | +0.203 | +0.177 | -0.044 | -0.610 / +0.180 |
| `m2` → cpi_yoy | extra_lag_1m | +0.173 | +0.181 | +0.168 | -0.040 | -0.654 / +0.170 |
| `m2` → cpi_yoy | from_1981_02 | -0.087 | +0.027 | -0.116 | -0.133 | -0.569 / +0.068 |
| `m2` → cpi_yoy | first_release | -0.033 | +0.099 | -0.067 | -0.171 | -0.453 / +0.110 |
| `m2real` → cpi_yoy | as_tested | +0.199 | +0.229 | +0.182 | -0.038 | -0.557 / +0.185 |
| `m2real` → cpi_yoy | extra_lag_1m | +0.181 | +0.213 | +0.162 | -0.037 | -0.590 / +0.165 |
| `deposits` → cpi_yoy | as_tested | +0.087 | +0.081 | +0.090 | -0.065 | -0.328 / +0.180 |
| `deposits` → cpi_yoy | extra_lag_1m | +0.089 | +0.077 | +0.094 | -0.056 | -0.331 / +0.183 |
| `mich1y` → cpi_yoy | as_tested | +0.079 | +0.218 | +0.006 | +0.052 | -1.284 / +0.025 |
| `near_spread` → ff | as_tested | +0.105 | +0.116 | +0.086 | +0.003 | -0.047 / +0.247 |
| `re_loans` → oil | as_tested | +0.140 | +0.030 | +0.221 | +0.260 | -0.376 / -0.073 |
| `re_loans` → oil | extra_lag_1m | +0.138 | +0.013 | +0.230 | +0.263 | -0.399 / -0.061 |
| `bank_credit` → ret_us_equity | as_tested | +0.025 | +0.020 | +0.035 | +0.150 | -0.898 / -0.026 |
| `bank_credit` → ret_us_equity | extra_lag_1m | +0.027 | +0.019 | +0.040 | +0.139 | -0.899 / -0.021 |
| `mich1y` → ret_gold | as_tested | +0.040 | +0.102 | +0.006 | -0.000 | -2.140 / -0.783 |

* Real-estate loans with the level breaks of 1958-01, 2006-10, 2008-10 removed (median growth): +0.208 / +0.130. First-release (ALFRED, 1997+) vs today's y/y: corr 0.995, mean absolute revision 0.37 pp (bank credit 0.944, M2 0.979).
* CPI parts, RE loans: core +0.364 / +0.077, services +0.242 / +0.070, goods +0.025 / +0.115. Faster housing credit is followed by higher underlying inflation. This is a household-balance-sheet demand channel the panel does not carry today.
* The baseline itself is poor for CPI (OOS R² vs the prevailing mean -1.02 in 1985–2004). In that half the candidate model is still worse than the mean; the gain is relative to the baseline, as the rule asks.
* M2 → CPI is not robust: restricted to the post-1981 sample +0.027 / -0.116, first releases +0.099 / -0.067, and without 2020–22 the second half turns negative (-0.044).
* Not forecast states, despite the harness flag: shadow_gap → ff (zero in every 1985–2004 origin; harness issue 1). kw_tp10, mich5y and spf_cpi10y have 27–49 first-half origins.

## 5. The y10 factor's own dynamics

ACM decomposition of monthly changes of the 10y zero-coupon yield (variance shares; corr = with the us_equity price return):

| period | sd Δy10 | expected path | term premium | 2·cov | corr(eq, Δy10) | corr(eq, Δpath) | corr(eq, ΔTP) |
|---|---|---|---|---|---|---|---|
| 1973-1984 | 0.426 | 1.05 | 0.51 | -0.56 | -0.28 | -0.21 | -0.09 |
| 1985-1999 | 0.311 | 0.48 | 0.53 | -0.01 | -0.29 | -0.12 | -0.28 |
| 2000-2007 | 0.263 | 0.54 | 0.58 | -0.12 | +0.33 | +0.37 | +0.07 |
| 2008-2019 | 0.269 | 0.34 | 1.14 | -0.47 | +0.24 | +0.26 | +0.08 |
| 2020-2026 | 0.281 | 0.34 | 0.62 | +0.05 | -0.32 | -0.07 | -0.36 |

Since 2020 equities move against the term premium (−0.36), not against the expected path. In 2008–19 they moved with the expected path. For 12-month changes of y10, only SPF 1y inflation expectations beat the baseline in both halves (weak). The ACM TP level, the slope, r* and the Cleveland expectations do not.

Beta stability: OLS on the core factors with y10 replaced by its parts, HAC t, diff z = difference / joint se. `base` and `acm_split` compare 1973-02..1999 with 2000–2026. `base82` and `cle_split` compare 1982..1999 with 2000–2026 (Cleveland starts in 1982). `d_y10_rest` = Δy10 − Δei10 − Δreal10 (inflation risk premium plus model error):

| asset | part | β pre | t | β post | t | diff z |
|---|---|---|---|---|---|---|
| us_equity | base: f_y10 | -0.045 | -4.4 | +0.003 | +0.3 | +3.1 |
| us_equity | acm_split: d_rny10 | -0.039 | -3.4 | +0.016 | +0.8 | +2.3 |
| us_equity | acm_split: d_tp10 | -0.053 | -4.2 | +0.000 | +0.0 | +3.3 |
| us_equity | base82: f_y10 | -0.055 | -4.7 | +0.003 | +0.3 | +3.5 |
| us_equity | cle_split: d_ei10 | -0.115 | -1.7 | +0.086 | +2.7 | +2.8 |
| us_equity | cle_split: d_real10 | -0.018 | -0.4 | -0.037 | -2.6 | -0.4 |
| us_equity | cle_split: d_y10_rest | -0.066 | -2.7 | -0.006 | -0.4 | +2.1 |
| nasdaq | base: f_y10 | -0.051 | -3.9 | +0.009 | +0.4 | +2.5 |
| nasdaq | acm_split: d_rny10 | -0.041 | -2.7 | +0.049 | +1.6 | +2.6 |
| nasdaq | acm_split: d_tp10 | -0.061 | -3.9 | -0.002 | -0.1 | +2.6 |
| nasdaq | base82: f_y10 | -0.062 | -3.7 | +0.009 | +0.4 | +2.7 |
| nasdaq | cle_split: d_ei10 | -0.168 | -2.0 | +0.104 | +1.9 | +2.7 |
| nasdaq | cle_split: d_real10 | +0.002 | +0.0 | -0.038 | -1.7 | -0.6 |
| nasdaq | cle_split: d_y10_rest | -0.095 | -2.7 | +0.007 | +0.2 | +2.2 |
| utilities | base: f_y10 | -0.058 | -6.8 | -0.039 | -4.1 | +1.5 |
| utilities | acm_split: d_rny10 | -0.066 | -6.5 | -0.009 | -0.4 | +2.5 |
| utilities | acm_split: d_tp10 | -0.059 | -5.7 | -0.042 | -4.3 | +1.2 |
| utilities | base82: f_y10 | -0.069 | -8.3 | -0.039 | -4.1 | +2.4 |
| utilities | cle_split: d_ei10 | -0.111 | -2.1 | +0.049 | +1.4 | +2.5 |
| utilities | cle_split: d_real10 | -0.043 | -1.4 | -0.081 | -4.8 | -1.1 |
| utilities | cle_split: d_y10_rest | -0.045 | -2.9 | -0.047 | -2.9 | -0.1 |
| real_estate | base: f_y10 | -0.043 | -5.0 | -0.055 | -3.6 | -0.7 |
| real_estate | acm_split: d_rny10 | -0.038 | -3.7 | -0.063 | -2.4 | -0.9 |
| real_estate | acm_split: d_tp10 | -0.039 | -4.5 | -0.048 | -3.5 | -0.6 |
| real_estate | base82: f_y10 | -0.046 | -5.5 | -0.055 | -3.6 | -0.5 |
| real_estate | cle_split: d_ei10 | -0.119 | -2.9 | +0.042 | +1.0 | +2.7 |
| real_estate | cle_split: d_real10 | -0.001 | -0.0 | -0.098 | -4.7 | -2.7 |
| real_estate | cle_split: d_y10_rest | -0.038 | -2.4 | -0.083 | -3.9 | -1.7 |
| financials | base: f_y10 | -0.063 | -5.4 | +0.025 | +1.9 | +5.1 |
| financials | acm_split: d_rny10 | -0.059 | -4.3 | +0.006 | +0.2 | +2.0 |
| financials | acm_split: d_tp10 | -0.071 | -5.0 | +0.027 | +2.2 | +5.2 |
| financials | base82: f_y10 | -0.077 | -5.4 | +0.025 | +1.9 | +5.3 |
| financials | cle_split: d_ei10 | -0.173 | -2.7 | +0.142 | +3.3 | +4.1 |
| financials | cle_split: d_real10 | -0.018 | -0.4 | -0.031 | -1.5 | -0.3 |
| financials | cle_split: d_y10_rest | -0.078 | -2.9 | +0.017 | +0.8 | +2.8 |
| energy | base: f_y10 | -0.031 | -2.2 | +0.027 | +2.2 | +3.1 |
| energy | acm_split: d_rny10 | -0.026 | -1.6 | +0.054 | +2.4 | +2.9 |
| energy | acm_split: d_tp10 | -0.041 | -2.7 | +0.021 | +1.7 | +3.2 |
| energy | base82: f_y10 | -0.047 | -4.0 | +0.027 | +2.2 | +4.4 |
| energy | cle_split: d_ei10 | -0.070 | -1.1 | +0.086 | +1.8 | +2.0 |
| energy | cle_split: d_real10 | -0.033 | -0.9 | +0.002 | +0.1 | +0.8 |
| energy | cle_split: d_y10_rest | -0.048 | -1.9 | +0.002 | +0.1 | +1.6 |
| gold | base: f_y10 | +0.011 | +1.0 | -0.027 | -3.0 | -2.6 |
| gold | acm_split: d_rny10 | +0.020 | +1.9 | -0.027 | -1.4 | -2.2 |
| gold | acm_split: d_tp10 | +0.021 | +1.5 | -0.022 | -2.3 | -2.5 |
| gold | base82: f_y10 | +0.022 | +1.7 | -0.027 | -3.0 | -3.1 |
| gold | cle_split: d_ei10 | -0.062 | -1.0 | -0.064 | -2.5 | -0.0 |
| gold | cle_split: d_real10 | +0.072 | +2.3 | -0.006 | -0.4 | -2.3 |
| gold | cle_split: d_y10_rest | +0.015 | +0.5 | -0.039 | -2.6 | -1.7 |

## 6. Gold

| candidate (gold equation) | 1995–2004 | 2005–14 | 2015–26 | full-sample coef (HAC t) |
|---|---|---|---|---|
| `tips10_avg` (same-sample) | n/a | n/a | +0.316 | -0.097 (-6.7) |
| `tips10_2m` (same-sample) | n/a | n/a | +0.273 | -0.090 (-5.6) |
| `cle_real10_2m` (same-sample) | +0.011 | -0.006 | -0.006 | +0.014 (+1.1) |
| `cle_ei10_2m` (same-sample) | +0.013 | -0.024 | -0.003 | +0.028 (+1.1) |
| `y10_minus_cle_ei10_2m` (same-sample) | -0.017 | +0.003 | +0.007 | -0.024 (-1.2) |
| `mich1y_2m` (same-sample) | -0.025 | -0.037 | +0.006 | +0.011 (+1.5) |
| `y10_avg` (full history) | -0.033 | -0.070 | -0.003 | -0.001 (-0.1) |
| `real10_avg_ewma` (full history) | -0.030 | -0.064 | +0.002 | -0.006 (-0.6) |
| `y10_eop_lag1` (full history) | -0.002 | -0.005 | +0.007 | -0.006 (-0.9) |

Window: base model fitted from 1982 instead of 1973 gives gold +0.143 / +0.232 / +0.171 (other assets mixed, e.g. utilities +0.111 in 2005–14 but -0.266 in 2015–26). For the implementation round: choose a gold-specific window on validation and gate it on test, like any change.

## 7. Coverage, alignment and engine fit

* Market prices (Treasury, TIPS, breakevens, ACM / KW term premia, Moody's, mortgage rates, shadow rate): month-end, like the panel's y10 (my DGS10 month-end equals lvl_y10 exactly). Known-at lag 0.
* Monthly averages (H.6 money, H.8 bank credit, CP and bill rates): the month's average, like the panel's ff. Known-at lag 1 (G.19 consumer credit lag 2; Fed H.4.1 lag 1).
* Cleveland Fed expectations: the value labelled month m is built from data through the end of m−1 (its change correlates 0.81 with Δy10 in m−1 and 0.16 in m), so reference month = label − 1 and known-at = label (published mid-month).
* Quarterly (SPF, HLW r*): held from release to release with `hold_quarterly` (SPF lag 1, HLW lag 5 from the quarter start), never interpolated. r* known-at: real-time vintages (latest quarter of each vintage, released ~quarter start + 5 months) from 2016-03; before that the current one-sided estimates held from quarter start + 5 (today's parameters and data: mild look-ahead).
* y2 before 1976-06 is interpolated across maturities from y1 and y5 in the same month (no look-ahead). Overlap check: corr of changes 0.981.
* Single missing months (2025-10 shutdown CPI for M2REAL, thin CP months) are interpolated for the reference-month changes, as `panel.put()` does. Known-at levels only carry forward.
* Full history 1973-02..2026-09 (MonthDB condition): ACM TP, y1, y2 (proxy), y5, bill3m, curvature, SPF PGDP (quarterly steps), shadow_gap (zero outside the ZLB), BAA−AAA, mortgage spread, M2, M2REAL, base, bank credit, C&I, RE loans, deposits, paper-bill spreads. Consumer credit misses the last month (publication lag). Late: KW (1990), TIPS/breakevens and Fed balance sheet (2003), Cleveland (1982), Michigan (1978/1990), CRE loans (2004).

## Harness issues and workarounds (harness not edited)

1. predictive_test: r2_incr_h1 = 3.3e-16 (numerical zero) counts as > 0 for a candidate that is identically zero in h1 (shadow_gap -> ff), so incr_both_halves and screen() call it a forecast state; worked around with a 1e-9 tolerance and n >= 60 origins per half.
2. transmission_test / screen: for candidates that start after 1973 the DM / Clark-West statistics and screen's assets_every_decade use the base fitted on the FULL training window, which mixes the training-window change with the factor (every 1978-1991 candidate 'helps' gold by +0.10..0.23 because its fit skips 1973-81); worked around with same-sample DM/CW (base refitted on the candidate's months; delta matches the harness delta_same_sample to 2e-15).
3. predictive_test: for late candidates the h1 half can hold very few origins (spf_cpi10y, mich5y, kw_tp10: 27-49) and still count for incr_both_halves; worked around with n >= 60.

## Files

* `factors/rates/report.md` — this report
* `factors/rates/candidates.csv` — every monthly series built, both alignments (`<id>__ref_level`, `__ref_change`, `__known_level` / `__known_yoy`) plus the mechanism series
* `factors/rates/tests.json` — all harness outputs and my additional tests
* `factors/rates/pvalues.csv` — every p value, flat, for cross-family BH
* `factors/rates/sources.json` — provenance (FRED URLs with api_key=REDACTED)
* `factors/rates/scripts/` — s01–s17, run in order with `python -B -I`
* `factors/rates/work/` — intermediate pickles / JSON and stdout logs
