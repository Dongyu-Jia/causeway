# Miss map: where the base transmission model misses

Base model = `model_selection.json`: per-asset OLS of the monthly **price** return on the same month's changes of the
10 core factors (oil, ip, cpi_goods, cpi_services, ff, y10, spread, usd, unrate, labor_share). "Out of sample" = the
test decade is predicted from the **actual** factor changes by a model fitted only on the training window, R² against
the training-window mean (walkforward semantics). Residuals for the era, autocorrelation, PCA and top-month analyses are
the expanding-window OOS residuals (refit every January, 1985-01..2026-09, 501 months). Everything below is generated
by `miss_map.py` through `harness.py` (numbers in `miss_map.json`); event labels are a short curated list of well-known
events, and a month without a label is shown with its data only.

Mean OOS R² over the 10 score assets: T1 (1995-2004) -0.024, T2 (2005-2014) -0.107, T3 (2015-2026) 0.172.

## Findings

1. **The dominant miss is the market's non-macro component (the multiple), not a missing real-economy factor.**
   PC1 of the standardized OOS residuals carries 56% of their variance (PC2 11%, PC3 9%). It loads
   positively on all nine equity/REIT series, correlates 0.95 with the us_equity residual and 0.85 with
   the same-month change of the market P/D. PC1 accounts for 90% of us_equity's residual variance
   (industrials 81%, financials 78%), but only 39% of energy's, 36% of utilities'
   and 9% of gold's. The P/D change alone absorbs 78% of the us_equity residual, but it is
   endogenous (built from the price), so the gap is an **exogenous driver of the equity multiple**: risk appetite,
   discount-rate or uncertainty news. PC1 also correlates -0.34 with the same-month change of the
   BAA−10Y spread, which is already in the model. The spread matters more out of sample than its estimated beta
   says: its us_equity beta was -0.031 per pp in 1973-1999 and -0.088 in 2000-2026 (table K). A cleaner
   credit-risk-appetite measure (excess bond premium, financial-conditions index) is the natural candidate.

2. **Crisis months, and rate/inflation betas that changed sign.** Months that are in the top-10 residual list of
   at least 3 of the 10 assets: 1987-10 (9), 2009-02 (9), 1998-08 (7), 2008-11 (7), 2008-10 (7), 2020-04 (5), 1987-01 (3), 2002-07 (3), 2009-04 (3). These are financial panics and the rebounds after them (1987-01 is
   a valuation surge). In 2008-11 the model **predicted**
   +7.4% for us_equity, +10.2% for healthcare and +11.3% for financials. The reason: yields and
   goods prices collapsed, and in 1973-1999 such moves were bullish (us_equity beta -0.045 per pp of y10,
   -1.42 per unit of goods inflation). In 2000-2026 they are not (+0.003 and -0.34); see table K.
   Without the seven GFC months 2008-09..2009-03 the T2 mean rises from -0.107 to -0.048 (staples -0.48
   → -0.25, healthcare -0.31 → -0.13). The project's inflation-regime interaction failed out of sample.
   What actually changes is whether a yield move is inflation news, term-premium news or growth news. That calls
   for a **mechanism**, not just a factor: a decomposition of Δy10 (expected short rates vs term premium) and a
   financial-stress factor.

3. **Slow drifts point to missing slow, sector-specific factors (the "dedicated factor" case).** Variance ratio
   VR(36) of the OOS residual (white noise = 1): gold 1.80, healthcare 1.65, us_equity 1.41,
   nasdaq 1.40, financials 1.28. Energy 0.78, industrials 0.71 and utilities 0.81
   instead revert. The largest 36-month cumulative residuals, against a white-noise sd of 0.23–0.40:
   * nasdaq +1.44 in 1997-2000 and -1.92 in 2000-03 (bubble)
   * financials -1.57 and REITs -0.96 in 2006-09 (housing and credit)
   * energy +1.08 in 2020-10..2023-09 (post-COVID energy rerating)
   * gold +0.84 in 2023-03..2026-02.

   By era (mean OOS residual per year, NW t):
   * AI-capex era (2024-01..2026-09): gold +21.7% (t 2.0), nasdaq +10.9% (t 1.4),
     us_equity +10.2% (t 1.7), utilities +9.3% (t 1.4)
   * tech boom: nasdaq +32.9% (t 2.9)
   * China-shock/credit era: healthcare -13.3% (t -2.6)
   * COVID/reflation: energy +14.6% (t 0.9).

   In 2000-2007 the model was worse than the era's own mean for most assets (unexplained share: us_equity
   1.21, healthcare 1.29). Each drift has an obvious sector story that no macro factor carries:
   * housing and credit for financials and REITs
   * the oil-industry profit cycle for energy
   * health policy and drug pricing for healthcare (1993 reform, 1994-97 rerating)
   * power markets and power demand for utilities (2001 California, 2002 merchant-power collapse, the AI era)
   * official-sector demand, real rates and geopolitics for gold.

4. **Gold is the worst-explained asset.** OOS R² T1 0.024, T2 -0.161, T3 -0.178. PC1 carries only
   9% of its residual. Its top misses are policy events: the Washington Agreement in 1999-10, the Plaza
   Accord in 1985-10 (the model predicted a large rise from the dollar's fall; gold did not move) and the 1990 Gulf
   crisis (predicted rise from oil). Residual acf1 0.17 (Ljung–Box p 0.007) is partly the
   monthly-average price; the REIT residual also fails Ljung–Box (p 0.000; acf12 > 0).

5. **The core labor-share factor does not earn its place.** `f_labor_share` interpolates the quarterly BLS series
   between mid-quarter months, so March 2020 already carries Q2-2020 (published in August). Dropping it **raises**
   the mean OOS R² in every decade: T1 +0.0088, T2 +0.0213, T3 +0.0074. The number of better assets is
   8/9/9 of 10, though no single asset is significant (max DM t 1.72). A release-dated
   labor share (no look-ahead) does not rescue it: lag 5 scores -0.011 / -0.094 / 0.166 and lag 4
   -0.026 / -0.077 / 0.178, against -0.015 / -0.086 / 0.180 with no labor share at all (and the
   base's -0.024 / -0.107 / 0.172); lag 5 beats dropping it in 1 of the 3 decades, lag 4 in 1. So the
   same-month change of the labor share carries no reliable out-of-sample information about monthly returns,
   with or without look-ahead. The household-income feedback the user asks about is better modeled as a slow
   state, a view or a 12-month forecast variable than as a same-month transmission factor in this form.
   Unflattering for the current spec; reported as found.

6. **The one-year forecast's big misses are macro surprises the analog cannot forecast.**
   * Of the 15 largest misses in forecast.json, 11 are 'macro' (the realized 12-month macro change fell
     outside the forecast's 90% band; 6 of them are the 2007-12 origin), 2 'shock' and 2
     'pricing'. The pricing ones: 1998-12 nasdaq +102% with the market P/D +18%; 1994-12 healthcare
     +57% with P/D +23%.
   * macro_skill: against the unconditional draw, the analog's CRPS skill on all five macro variables lies between
     -0.6% and +1.3% (|t| ≤ 1.4), so the analog weighting adds nothing.
   * Against the naive normal it is better for unemployment (+15%, t 3.4) and the policy rate
     (+16%, t 3.3), and not significantly for the spread (+11%, t 1.3) or oil (+3%,
     t 0.6). It is **worse for CPI y/y (-45%, t -3.0; -86% up to 2004)**: the inflation bias
     of drawing historical inflation without re-anchoring.
   * A plain linear model on the KEY_STATES does forecast the 12-month IP change (OOS R² 0.35; 0.46
     after 2004) and barely the unemployment change (0.02). It is negative for every risky-asset return
     (-1.21 to -0.10; +0.06 for the 10-year Treasury), so a forecast-state candidate has to beat
     both that baseline and the prevailing mean.

7. **T3 is the friendliest decade, but not because of COVID.** Excluding 2020-02..06 raises the T3 mean from
   0.172 to 0.192 (energy 0.17 → 0.30). T2's training window ends in 2004, just after the 2000
   break, and even without the GFC months its mean stays negative (-0.048).

## What the families should look for

Scenario concepts with no panel variable (tariffs, deficits, term premium, house prices, power demand, central-bank
gold, ...) are listed with candidate series in concept_gaps.md. Publishers are the allowed keyless sources; FRED
(api.stlouisfed.org, key provided) carries many of the same series in one format. Verify every FRED ID before use.

| miss | dedicated factor / mechanism to test | long-history candidates |
|---|---|---|
| PC1 (market multiple), crisis months | exogenous risk appetite, financial stress, uncertainty; run `endogeneity_check` first (market-derived series are suspect) | GZ spread and excess bond premium (federalreserve.gov csv, monthly 1973+); Chicago Fed NFCI/ANFCI (chicagofed.org or FRED, weekly 1971+); EPU (policyuncertainty.com, 1985+); VXO/VIX (cboe.com or FRED, 1986/1990+, itself a market price) |
| y10 beta flip around 2000, defensives in T2 | split Δy10 into expected short rates and term premium; survey inflation expectations instead of the EWMA proxy | ACM term premium (newyorkfed.org, 1961+); Kim–Wright (federalreserve.gov or FRED, 1990+); Michigan 1-yr expectations (sca.isr.umich.edu, monthly 1978+); SPF / Livingston (philadelphiafed.org) |
| financials, REITs (2006-09 drift, GFC months) | housing and credit supply | Freddie Mac house price index (freddiemac.com, monthly 1975+) and PMMS mortgage rate (1971+); housing starts and permits (Census via FRED, 1959/1960+); SLOOS (federalreserve.gov, 1990+); delinquency and charge-off rates (federalreserve.gov, quarterly 1985+) |
| energy (2020-10..2023-09 drift; 2020-04/05) | oil-industry cash flow beyond the spot price: natural gas, refining margins | World Bank Pink Sheet (worldbank.org, monthly averages 1960+); Henry Hub gas (EIA via FRED, 1997+); BLS PPI refined petroleum products (long history) |
| utilities (2001-02; AI era +9%/yr residual) | power demand and power prices (data centers, electrification) | Fed G.17 output of electric and gas utilities (federalreserve.gov or FRED; detail from 1972); BLS CPI electricity (1952+) |
| AI-capex era 2024-01..: us_equity +10%/yr, nasdaq +11%, gold +22% residual | AI capex beyond the aggregate capex share: information-processing equipment and software investment, semiconductor output | NIPA private fixed investment in information processing equipment and in software (BEA via FRED, quarterly 1947+, use `hold_quarterly`); Fed G.17 IP semiconductors (1972+); Census data-center construction spending (short history: a view or state at most) |
| healthcare (policy cycles; 2000-07 −13%/yr residual) | health-price and policy measures | BLS CPI medical care (1947+); BLS PPI pharmaceutical preparations (1980s+) |
| gold (drifts, policy events; AI era) | official-sector demand, real rates, geopolitics, dollar regime | GPR (matteoiacoviello.com, 1985+; historical index 1900+); TIPS real yields (federalreserve.gov H.15, 2003+: too short for core); central-bank gold purchases (WGC / IMF: not on the allowed list) |
| labor share look-ahead; the user's questions on white-collar wages and household income | rebuild distribution factors release-dated with `hold_quarterly`; test income and wage measures as slow states (`residual_test` horizon 12, `predictive_test`) as well as same-month factors | BLS Productivity and Costs (labor share, unit labor costs: api.bls.gov or FRED, quarterly 1947+); BEA personal income, wages and salaries, real disposable income (FRED, monthly 1959+); BLS CES hourly earnings of production and nonsupervisory employees in information, financial activities and professional services (1960s+; all-employee earnings only 2006+); BLS CPS earnings by decile (start date to check) |

## A. Out-of-sample R² of the base model (price returns, actual same-month factors)

|  | T1 | T2 | T3 | T3_131m | in_sample_1973_2026 |
|---|---|---|---|---|---|
| us_equity 美股大盘 | -0.109 | -0.067 | 0.319 | 0.262 | 0.165 |
| nasdaq 纳斯达克100 | -0.066 | -0.005 | 0.259 | 0.222 | 0.104 |
| energy 石油能源股 | 0.084 | 0.260 | 0.172 | 0.325 | 0.278 |
| industrials 工业制造股 | -0.059 | -0.018 | 0.295 | 0.241 | 0.170 |
| staples 必需消费股 | -0.015 | -0.476 | 0.199 | 0.165 | 0.143 |
| healthcare 医疗保健股 | -0.006 | -0.308 | 0.059 | 0.123 | 0.104 |
| utilities 公用事业股 | -0.010 | -0.049 | 0.150 | 0.231 | 0.188 |
| financials 金融股 | -0.057 | -0.320 | 0.073 | 0.067 | 0.115 |
| real_estate 房地产 REITs | -0.026 | 0.074 | 0.374 | 0.331 | 0.206 |
| gold 黄金 | 0.024 | -0.161 | -0.178 | -0.090 | 0.117 |
| crude_oil 原油 | 0.954 | 0.970 | 0.940 | 0.982 | 0.957 |
| treasury_10y 10年期美债 | 0.953 | 0.932 | 0.949 | 0.955 | 0.973 |

Mean over the 10 score assets: T1 -0.024, T2 -0.107, T3 0.172 (131-month sample 0.188).

## B. Residuals by era (score assets)

**Mean residual per year (drift the factors miss; + = asset did better than its macro factors imply)**

|  | us_equity | nasdaq | energy | industrials | staples | healthcare | utilities | financials | real_estate | gold |
|---|---|---|---|---|---|---|---|---|---|---|
| great_inflation 大通胀与石油危机（样本内） | -0.012 | -0.016 | -0.048 | +0.007 | -0.002 | -0.020 | -0.023 | -0.014 | +0.020 | +0.080 |
| disinflation 大缓和与去通胀 | -0.028 | -0.019 | -0.018 | -0.033 | -0.034 | -0.000 | -0.016 | -0.036 | -0.082 | -0.099 |
| tech_boom 科技繁荣 | +0.134 | +0.329 | +0.085 | +0.097 | -0.007 | +0.125 | +0.023 | +0.110 | -0.058 | -0.052 |
| china_credit 中国冲击与信贷扩张 | -0.104 | -0.207 | +0.034 | -0.044 | -0.080 | -0.133 | +0.042 | -0.087 | +0.043 | +0.082 |
| stagnation_qe 长期停滞与 QE | -0.008 | -0.004 | -0.074 | -0.031 | -0.051 | -0.021 | -0.002 | -0.040 | -0.003 | +0.055 |
| covid_reflation 疫情与再通胀 | +0.063 | +0.082 | +0.146 | +0.054 | -0.047 | -0.013 | +0.010 | +0.032 | -0.009 | -0.013 |
| ai_capex AI 资本周期 | +0.102 | +0.109 | +0.046 | +0.089 | -0.047 | +0.050 | +0.093 | +0.063 | +0.010 | +0.217 |

**Residual sd per year**

|  | us_equity | nasdaq | energy | industrials | staples | healthcare | utilities | financials | real_estate | gold |
|---|---|---|---|---|---|---|---|---|---|---|
| great_inflation 大通胀与石油危机（样本内） | 0.172 | 0.201 | 0.242 | 0.196 | 0.164 | 0.203 | 0.146 | 0.193 | 0.158 | 0.278 |
| disinflation 大缓和与去通胀 | 0.148 | 0.214 | 0.156 | 0.176 | 0.167 | 0.186 | 0.106 | 0.166 | 0.119 | 0.145 |
| tech_boom 科技繁荣 | 0.145 | 0.265 | 0.153 | 0.164 | 0.143 | 0.153 | 0.121 | 0.187 | 0.124 | 0.117 |
| china_credit 中国冲击与信贷扩张 | 0.154 | 0.312 | 0.184 | 0.175 | 0.117 | 0.138 | 0.164 | 0.164 | 0.149 | 0.115 |
| stagnation_qe 长期停滞与 QE | 0.146 | 0.169 | 0.165 | 0.194 | 0.132 | 0.151 | 0.120 | 0.234 | 0.218 | 0.144 |
| covid_reflation 疫情与再通胀 | 0.151 | 0.183 | 0.389 | 0.186 | 0.137 | 0.155 | 0.171 | 0.215 | 0.158 | 0.123 |
| ai_capex AI 资本周期 | 0.108 | 0.158 | 0.167 | 0.195 | 0.106 | 0.131 | 0.129 | 0.149 | 0.121 | 0.151 |

**Unexplained share SSE/SST within the era (OOS; > 1 = worse than the era's own mean)**

|  | us_equity | nasdaq | energy | industrials | staples | healthcare | utilities | financials | real_estate | gold |
|---|---|---|---|---|---|---|---|---|---|---|
| great_inflation 大通胀与石油危机（样本内） | 0.94 | 0.95 | 1.12 | 0.95 | 0.85 | 0.93 | 0.84 | 0.92 | 0.97 | 0.89 |
| disinflation 大缓和与去通胀 | 0.93 | 0.97 | 0.96 | 0.98 | 0.92 | 0.95 | 0.73 | 0.85 | 1.01 | 1.73 |
| tech_boom 科技繁荣 | 1.09 | 1.14 | 0.82 | 1.11 | 0.95 | 0.89 | 0.87 | 1.02 | 1.07 | 1.13 |
| china_credit 中国冲击与信贷扩张 | 1.21 | 1.12 | 0.96 | 1.07 | 1.16 | 1.29 | 1.13 | 1.18 | 1.01 | 0.87 |
| stagnation_qe 长期停滞与 QE | 0.91 | 0.90 | 0.61 | 0.86 | 1.14 | 1.09 | 0.88 | 1.16 | 0.89 | 1.23 |
| covid_reflation 疫情与再通胀 | 0.56 | 0.62 | 0.92 | 0.57 | 0.70 | 0.89 | 0.86 | 0.82 | 0.50 | 1.28 |
| ai_capex AI 资本周期 | 0.84 | 0.92 | 0.60 | 0.89 | 0.73 | 0.83 | 0.82 | 0.86 | 0.65 | 1.11 |

Eras: great_inflation from the in-sample fit (OOS residuals start 1985-01); disinflation OOS = 1985-1994 only.

## C. Residual autocorrelation and slow drift (OOS residuals 1985-01..2026-09)

|  | acf1 | acf2 | acf12 | lb_p | vr12 | vr36 | vr60 |
|---|---|---|---|---|---|---|---|
| us_equity | 0.085 | -0.040 | 0.037 | 0.185 | 1.160 | 1.408 | 1.335 |
| nasdaq | 0.078 | -0.014 | -0.013 | 0.125 | 1.250 | 1.399 | 1.263 |
| energy | -0.061 | -0.030 | -0.033 | 0.081 | 0.821 | 0.782 | 0.593 |
| industrials | 0.027 | -0.028 | 0.043 | 0.317 | 0.896 | 0.713 | 0.605 |
| staples | 0.073 | -0.031 | 0.015 | 0.473 | 1.035 | 1.047 | 0.864 |
| healthcare | 0.073 | -0.001 | 0.030 | 0.137 | 1.266 | 1.654 | 1.639 |
| utilities | -0.023 | -0.079 | 0.060 | 0.063 | 1.010 | 0.806 | 0.548 |
| financials | 0.111 | -0.023 | 0.065 | 0.141 | 1.156 | 1.282 | 1.390 |
| real_estate | 0.020 | -0.104 | 0.109 | 0.000 | 1.008 | 0.839 | 0.661 |
| gold | 0.174 | 0.006 | 0.057 | 0.007 | 1.337 | 1.803 | 1.812 |
| crude_oil | 0.041 | 0.117 | -0.039 | 0.389 | 1.242 | 0.912 | 0.728 |
| treasury_10y | 0.125 | -0.055 | 0.038 | 0.084 | 1.181 | 1.072 | 0.861 |

**Largest 36-month cumulative residuals (non-overlapping)** — white-noise sd of a 36-month sum in brackets

* us_equity (0.26): 2000-04..2003-03 -1.08; 2006-03..2009-02 -0.80; 1997-04..2000-03 +0.48; 2019-01..2021-12 +0.43
* nasdaq (0.40): 2000-04..2003-03 -1.92; 1997-04..2000-03 +1.44; 2006-03..2009-02 -0.75; 2019-01..2021-12 +0.53
* energy (0.34): 2020-10..2023-09 +1.08; 2017-10..2020-09 -0.62; 2007-07..2010-06 -0.59; 2003-12..2006-11 +0.55
* industrials (0.31): 2006-03..2009-02 -0.93; 2000-04..2003-03 -0.82; 2009-03..2012-02 +0.49; 1987-10..1990-09 -0.45
* staples (0.24): 2007-07..2010-06 -0.71; 2000-04..2003-03 -0.67; 1991-04..1994-03 -0.61; 2017-05..2020-04 -0.42
* healthcare (0.27): 2000-11..2003-10 -0.96; 2006-03..2009-02 -0.84; 1994-08..1997-07 +0.63; 1991-08..1994-07 -0.62
* utilities (0.23): 2003-03..2006-02 +0.55; 2007-06..2010-05 -0.51; 1999-11..2002-10 -0.41; 2021-12..2024-11 +0.39
* financials (0.34): 2006-03..2009-02 -1.57; 2000-04..2003-03 -0.73; 1995-05..1998-04 +0.63; 1987-10..1990-09 -0.59
* real_estate (0.28): 2006-04..2009-03 -0.96; 2009-04..2012-03 +0.65; 1987-10..1990-09 -0.62; 2003-04..2006-03 +0.53
* gold (0.24): 2023-03..2026-02 +0.84; 1987-11..1990-10 -0.79; 2008-10..2011-09 +0.76; 2005-04..2008-03 +0.55

## D. Residual PCA (correlation matrix, 10 score assets, OOS residuals)

|  | PC1 | PC2 | PC3 |
|---|---|---|---|
| us_equity | 0.403 | 0.213 | 0.100 |
| nasdaq | 0.307 | 0.493 | 0.248 |
| energy | 0.266 | -0.447 | -0.044 |
| industrials | 0.381 | 0.122 | 0.045 |
| staples | 0.344 | 0.016 | -0.175 |
| healthcare | 0.331 | 0.156 | -0.067 |
| utilities | 0.254 | -0.487 | -0.423 |
| financials | 0.374 | -0.007 | 0.031 |
| real_estate | 0.278 | -0.129 | -0.182 |
| gold | -0.130 | 0.467 | -0.823 |
| variance share | 0.556 | 0.107 | 0.092 |
| corr with us_equity residual | 0.951 | 0.221 | 0.096 |

Correlation of the PCs with panel series (same month):

|  | PC1 | PC2 | PC3 |
|---|---|---|---|
| f_valuation | 0.846 | 0.177 | 0.064 |
| f_cape | 0.586 | 0.106 | 0.038 |
| f_sentiment | 0.109 | 0.095 | -0.039 |
| f_wealth | 0.210 | 0.101 | 0.032 |
| f_spread | -0.335 | 0.058 | -0.123 |
| f_oil | 0.124 | -0.204 | 0.239 |
| f_y10 | 0.176 | -0.008 | 0.175 |
| f_usd | -0.207 | 0.121 | -0.158 |
| f_ff | 0.040 | -0.012 | 0.031 |
| f_unrate | 0.115 | -0.278 | 0.013 |
| e_oil_supply | 0.194 | -0.136 | 0.218 |
| e_demand | -0.025 | -0.097 | 0.162 |
| e_monetary | -0.125 | -0.045 | -0.059 |
| e_financial | -0.228 | -0.113 | 0.011 |
| p_treasury_10y | -0.199 | 0.014 | -0.185 |
| p_crude_oil | 0.107 | -0.195 | 0.242 |

Covariance-matrix PCA: shares 0.577, 0.129, 0.074; PC1 corr with us_equity residual 0.970.

Share of each asset's OOS residual variance carried by PC1, and by the (endogenous) market P/D change:

|  | us_equity | nasdaq | energy | industrials | staples | healthcare | utilities | financials | real_estate | gold |
|---|---|---|---|---|---|---|---|---|---|---|
| PC1 R² | 0.904 | 0.526 | 0.394 | 0.806 | 0.657 | 0.610 | 0.358 | 0.779 | 0.431 | 0.094 |
| market P/D change R² | 0.777 | 0.541 | 0.219 | 0.615 | 0.410 | 0.434 | 0.164 | 0.549 | 0.267 | 0.045 |

## E. The 10 largest residual months per asset (OOS)

residual = actual − predicted price return; factor moves in sd units; P/D = market price/dividend change (endogenous, shown only to tell valuation moves apart).


**us_equity 美股大盘**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 1987-10 | -0.267 | -0.228 | +0.040 | y10 -2.2σ, spread +2.0σ | -22.1% | 黑色星期一（1987-10-19） |
| 1998-08 | -0.179 | -0.157 | +0.022 | ip +2.0σ, y10 -1.3σ | -15.8% | 俄罗斯违约与 LTCM 危机（1998-08） |
| 2008-11 | -0.154 | -0.080 | +0.074 | cpi_goods -8.2σ, y10 -3.2σ | -6.4% | 全球金融危机：花旗救助（2008-11-23） |
| 2008-10 | -0.138 | -0.173 | -0.034 | spread +5.7σ, usd +4.5σ | -17.3% | 全球金融危机：雷曼倒闭后的崩盘 |
| 2000-11 | -0.128 | -0.103 | +0.025 | y10 -0.9σ, labor_share -0.7σ | -10.9% | 互联网泡沫破裂（纳斯达克 2000-11 约 −23%）与大选计票僵局 |
| 2020-04 | +0.127 | +0.135 | +0.008 | unrate +22.2σ, ip -14.7σ | +13.5% | 封锁后的反弹；WTI 期货跌到负值（2020-04-20） |
| 2002-09 | -0.121 | -0.103 | +0.017 | spread +1.6σ, y10 -1.5σ | -10.6% | 2002 年熊市二次探底 |
| 2009-02 | -0.115 | -0.104 | +0.011 | labor_share -3.8σ, usd +1.6σ | -9.1% | 全球金融危机最后一跌（2009-03-09 见底） |
| 1987-01 | +0.113 | +0.127 | +0.014 | usd -2.4σ, cpi_goods +1.2σ | +12.8% | 1987 年初美股急涨（S&P 500 单月约 +13%，估值扩张） |
| 2002-07 | -0.107 | -0.081 | +0.026 | usd -1.4σ, y10 -1.0σ | -7.5% | 会计丑闻熊市（WorldCom 2002-07-21 破产），能源交易商和电力公司股价崩盘 |

**nasdaq 纳斯达克100**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 1987-10 | -0.313 | -0.270 | +0.043 | y10 -2.2σ, spread +2.0σ | -22.1% | 黑色星期一（1987-10-19） |
| 2001-02 | -0.291 | -0.264 | +0.027 | labor_share +2.2σ, spread +1.0σ | -9.4% | 科技股崩跌继续（2001-02） |
| 1999-12 | +0.275 | +0.250 | -0.025 | labor_share +3.3σ, spread -1.0σ | +7.8% | 互联网泡沫顶峰前的狂热（纳斯达克 100 单月约 +25%） |
| 2000-11 | -0.272 | -0.236 | +0.036 | y10 -0.9σ, labor_share -0.7σ | -10.9% | 互联网泡沫破裂（纳斯达克 2000-11 约 −23%）与大选计票僵局 |
| 2008-11 | -0.213 | -0.112 | +0.101 | cpi_goods -8.2σ, y10 -3.2σ | -6.4% | 全球金融危机：花旗救助（2008-11-23） |
| 2002-10 | +0.203 | +0.189 | -0.014 | oil -1.2σ, y10 +0.9σ | +5.7% | 2002-10 熊市见底反弹 |
| 2001-09 | -0.202 | -0.205 | -0.003 | spread +1.9σ, oil -1.3σ | -9.4% | 9·11 恐怖袭击 |
| 2000-02 | +0.196 | +0.195 | -0.000 | labor_share +3.2σ, usd +1.4σ | +2.0% | 互联网泡沫最后冲刺（纳斯达克 2000-03-10 见顶） |
| 1998-08 | -0.193 | -0.172 | +0.021 | ip +2.0σ, y10 -1.3σ | -15.8% | 俄罗斯违约与 LTCM 危机（1998-08） |
| 2001-03 | -0.181 | -0.176 | +0.006 | labor_share -1.6σ, usd +1.4σ | -3.8% | 科技股崩跌继续（2001-03） |

**energy 石油能源股**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 2020-04 | +0.493 | +0.323 | -0.170 | unrate +22.2σ, ip -14.7σ | +13.5% | 封锁后的反弹；WTI 期货跌到负值（2020-04-20） |
| 1987-10 | -0.224 | -0.194 | +0.030 | y10 -2.2σ, spread +2.0σ | -22.1% | 黑色星期一（1987-10-19） |
| 2022-10 | +0.208 | +0.234 | +0.026 | labor_share -1.2σ, ff +1.0σ | +7.0% | OPEC+ 减产 200 万桶/日（2022-10-05）与熊市反弹 |
| 2021-02 | +0.202 | +0.224 | +0.022 | ip -3.6σ, labor_share -2.1σ | +3.9% | 再通胀交易：收益率和油价上涨（2021-02） |
| 2020-11 | +0.200 | +0.270 | +0.070 | labor_share +2.8σ, oil +2.3σ | +13.3% | 辉瑞疫苗（2020-11-09）与大选后的周期股轮动 |
| 2005-02 | +0.170 | +0.190 | +0.020 | labor_share -1.2σ, oil +0.7σ | -0.8% | — |
| 2020-05 | -0.166 | -0.007 | +0.160 | oil +6.0σ, unrate -3.4σ | +4.7% | 油价从负值反弹，能源股没有跟随 |
| 2009-02 | -0.158 | -0.130 | +0.028 | labor_share -3.8σ, usd +1.6σ | -9.1% | 全球金融危机最后一跌（2009-03-09 见底） |
| 2002-07 | -0.146 | -0.118 | +0.027 | usd -1.4σ, y10 -1.0σ | -7.5% | 会计丑闻熊市（WorldCom 2002-07-21 破产），能源交易商和电力公司股价崩盘 |
| 2008-07 | -0.142 | -0.157 | -0.015 | cpi_goods +1.3σ, oil -1.2σ | -1.4% | 油价 2008-07-11 见顶约 147 美元后急跌 |

**industrials 工业制造股**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 1987-10 | -0.326 | -0.286 | +0.040 | y10 -2.2σ, spread +2.0σ | -22.1% | 黑色星期一（1987-10-19） |
| 2008-10 | -0.209 | -0.238 | -0.029 | spread +5.7σ, usd +4.5σ | -17.3% | 全球金融危机：雷曼倒闭后的崩盘 |
| 2009-02 | -0.188 | -0.162 | +0.025 | labor_share -3.8σ, usd +1.6σ | -9.1% | 全球金融危机最后一跌（2009-03-09 见底） |
| 2009-04 | +0.186 | +0.208 | +0.022 | spread -2.5σ, cpi_services -1.4σ | +10.5% | 银行压力测试前的大反弹（2009-04） |
| 1998-08 | -0.179 | -0.158 | +0.021 | ip +2.0σ, y10 -1.3σ | -15.8% | 俄罗斯违约与 LTCM 危机（1998-08） |
| 2001-09 | -0.171 | -0.160 | +0.011 | spread +1.9σ, oil -1.3σ | -9.4% | 9·11 恐怖袭击 |
| 2026-06 | +0.156 | +0.147 | -0.009 | oil -2.6σ, cpi_goods -2.4σ | -2.9% | — |
| 2008-11 | -0.155 | -0.084 | +0.071 | cpi_goods -8.2σ, y10 -3.2σ | -6.4% | 全球金融危机：花旗救助（2008-11-23） |
| 2022-10 | +0.148 | +0.147 | -0.001 | labor_share -1.2σ, ff +1.0σ | +7.0% | OPEC+ 减产 200 万桶/日（2022-10-05）与熊市反弹 |
| 1999-12 | +0.145 | +0.119 | -0.026 | labor_share +3.3σ, spread -1.0σ | +7.8% | 互联网泡沫顶峰前的狂热（纳斯达克 100 单月约 +25%） |

**staples 必需消费股**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 1987-10 | -0.254 | -0.211 | +0.043 | y10 -2.2σ, spread +2.0σ | -22.1% | 黑色星期一（1987-10-19） |
| 1998-08 | -0.155 | -0.132 | +0.024 | ip +2.0σ, y10 -1.3σ | -15.8% | 俄罗斯违约与 LTCM 危机（1998-08） |
| 1987-01 | +0.134 | +0.143 | +0.009 | usd -2.4σ, cpi_goods +1.2σ | +12.8% | 1987 年初美股急涨（S&P 500 单月约 +13%，估值扩张） |
| 2018-12 | -0.134 | -0.117 | +0.016 | labor_share +1.3σ, oil -1.2σ | -11.3% | 2018-12 抛售（加息与缩表担忧） |
| 2008-11 | -0.125 | -0.046 | +0.079 | cpi_goods -8.2σ, y10 -3.2σ | -6.4% | 全球金融危机：花旗救助（2008-11-23） |
| 2008-12 | -0.122 | +0.002 | +0.124 | cpi_goods -4.2σ, labor_share -3.7σ | +2.0% | 全球金融危机：联邦基金利率降到 0–0.25%（2008-12-16），长端收益率暴跌 |
| 2008-10 | -0.118 | -0.135 | -0.017 | spread +5.7σ, usd +4.5σ | -17.3% | 全球金融危机：雷曼倒闭后的崩盘 |
| 2009-02 | -0.112 | -0.078 | +0.034 | labor_share -3.8σ, usd +1.6σ | -9.1% | 全球金融危机最后一跌（2009-03-09 见底） |
| 1998-10 | +0.107 | +0.109 | +0.002 | usd -2.0σ, oil -1.1σ | +7.0% | LTCM 救助与美联储连续降息后的反弹（1998-10） |
| 1993-04 | -0.104 | -0.094 | +0.010 | usd -1.4σ, labor_share +0.8σ | -2.5% | “万宝路星期五”（1993-04-02 菲利普莫里斯降价约 20%），品牌消费品股暴跌 |

**healthcare 医疗保健股**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 1987-10 | -0.238 | -0.203 | +0.034 | y10 -2.2σ, spread +2.0σ | -22.1% | 黑色星期一（1987-10-19） |
| 2008-11 | -0.179 | -0.077 | +0.102 | cpi_goods -8.2σ, y10 -3.2σ | -6.4% | 全球金融危机：花旗救助（2008-11-23） |
| 2020-04 | +0.162 | +0.133 | -0.029 | unrate +22.2σ, ip -14.7σ | +13.5% | 封锁后的反弹；WTI 期货跌到负值（2020-04-20） |
| 1998-08 | -0.139 | -0.125 | +0.015 | ip +2.0σ, y10 -1.3σ | -15.8% | 俄罗斯违约与 LTCM 危机（1998-08） |
| 1991-12 | +0.124 | +0.164 | +0.040 | y10 -2.0σ, spread +1.2σ | +10.4% | 美联储 1991-12-20 把贴现率下调 100bp（联邦基金目标降到 4%）后的年末大涨 |
| 2008-10 | -0.122 | -0.111 | +0.011 | spread +5.7σ, usd +4.5σ | -17.3% | 全球金融危机：雷曼倒闭后的崩盘 |
| 2002-06 | -0.119 | -0.096 | +0.023 | usd -1.5σ, cpi_services -0.8σ | -9.7% | WorldCom 财务造假曝光（2002-06-25），会计丑闻熊市 |
| 2009-02 | -0.118 | -0.112 | +0.007 | labor_share -3.8σ, usd +1.6σ | -9.1% | 全球金融危机最后一跌（2009-03-09 见底） |
| 1993-02 | -0.117 | -0.092 | +0.024 | y10 -1.1σ, cpi_services -0.6σ | -0.0% | 克林顿医改启动、抨击药价（1993 年初），医药股下跌 |
| 1987-01 | +0.116 | +0.130 | +0.014 | usd -2.4σ, cpi_goods +1.2σ | +12.8% | 1987 年初美股急涨（S&P 500 单月约 +13%，估值扩张） |

**utilities 公用事业股**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 2002-07 | -0.147 | -0.124 | +0.023 | usd -1.4σ, y10 -1.0σ | -7.5% | 会计丑闻熊市（WorldCom 2002-07-21 破产），能源交易商和电力公司股价崩盘 |
| 2020-04 | +0.147 | +0.051 | -0.096 | unrate +22.2σ, ip -14.7σ | +13.5% | 封锁后的反弹；WTI 期货跌到负值（2020-04-20） |
| 2009-02 | -0.136 | -0.130 | +0.006 | labor_share -3.8σ, usd +1.6σ | -9.1% | 全球金融危机最后一跌（2009-03-09 见底） |
| 2002-09 | -0.136 | -0.112 | +0.025 | spread +1.6σ, y10 -1.5σ | -10.6% | 2002 年熊市二次探底 |
| 2002-03 | +0.123 | +0.100 | -0.023 | oil +1.8σ, y10 +1.6σ | +5.0% | — |
| 1987-10 | -0.115 | -0.074 | +0.041 | y10 -2.2σ, spread +2.0σ | -22.1% | 黑色星期一（1987-10-19） |
| 2021-03 | +0.110 | +0.100 | -0.010 | ip +2.8σ, cpi_goods +1.1σ | +2.1% | — |
| 2020-02 | -0.108 | -0.103 | +0.005 | labor_share +2.6σ, oil -1.4σ | -8.3% | 新冠疫情冲击开始（2020-02 下旬） |
| 2001-01 | -0.106 | -0.104 | +0.003 | labor_share +2.2σ, cpi_services +1.8σ | +3.1% | 加州电力危机（2001-01 轮流停电，电力公司濒临破产） |
| 2020-06 | -0.105 | -0.054 | +0.051 | ip +6.4σ, unrate -4.7σ | +2.4% | 新冠疫情与全面封锁 |

**financials 金融股**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 1987-10 | -0.265 | -0.214 | +0.052 | y10 -2.2σ, spread +2.0σ | -22.1% | 黑色星期一（1987-10-19） |
| 2008-11 | -0.258 | -0.145 | +0.113 | cpi_goods -8.2σ, y10 -3.2σ | -6.4% | 全球金融危机：花旗救助（2008-11-23） |
| 1998-08 | -0.255 | -0.221 | +0.034 | ip +2.0σ, y10 -1.3σ | -15.8% | 俄罗斯违约与 LTCM 危机（1998-08） |
| 2009-01 | -0.227 | -0.210 | +0.017 | labor_share -3.8σ, ip -2.7σ | -6.4% | 全球金融危机：银行国有化担忧 |
| 2008-10 | -0.204 | -0.204 | -0.000 | spread +5.7σ, usd +4.5σ | -17.3% | 全球金融危机：雷曼倒闭后的崩盘 |
| 2020-04 | +0.200 | +0.118 | -0.083 | unrate +22.2σ, ip -14.7σ | +13.5% | 封锁后的反弹；WTI 期货跌到负值（2020-04-20） |
| 2009-02 | -0.180 | -0.156 | +0.024 | labor_share -3.8σ, usd +1.6σ | -9.1% | 全球金融危机最后一跌（2009-03-09 见底） |
| 2009-04 | +0.168 | +0.169 | +0.001 | spread -2.5σ, cpi_services -1.4σ | +10.5% | 银行压力测试前的大反弹（2009-04） |
| 2009-03 | +0.166 | +0.145 | -0.021 | spread +1.9σ, ip -1.8σ | +6.2% | 2009-03-09 见底后的反弹 |
| 2008-06 | -0.165 | -0.178 | -0.013 | cpi_goods +2.8σ, oil +0.9σ | -9.3% | 银行股暴跌（雷曼 2008-06-09 预告二季度亏损），油价创新高 |

**real_estate 房地产 REITs**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 2008-11 | -0.286 | -0.236 | +0.050 | cpi_goods -8.2σ, y10 -3.2σ | -6.4% | 全球金融危机：花旗救助（2008-11-23） |
| 2009-04 | +0.279 | +0.305 | +0.026 | spread -2.5σ, cpi_services -1.4σ | +10.5% | 银行压力测试前的大反弹（2009-04） |
| 2008-10 | -0.259 | -0.319 | -0.060 | spread +5.7σ, usd +4.5σ | -17.3% | 全球金融危机：雷曼倒闭后的崩盘 |
| 2009-02 | -0.240 | -0.213 | +0.028 | labor_share -3.8σ, usd +1.6σ | -9.1% | 全球金融危机最后一跌（2009-03-09 见底） |
| 2009-01 | -0.202 | -0.175 | +0.027 | labor_share -3.8σ, ip -2.7σ | -6.4% | 全球金融危机：银行国有化担忧 |
| 1987-10 | -0.184 | -0.158 | +0.025 | y10 -2.2σ, spread +2.0σ | -22.1% | 黑色星期一（1987-10-19） |
| 2004-04 | -0.141 | -0.149 | -0.008 | y10 +2.0σ, spread -0.9σ | -2.5% | 就业报告引发加息预期急升（2004-04），REITs 暴跌 |
| 2011-10 | +0.117 | +0.141 | +0.024 | labor_share -2.2σ, usd -1.8σ | +12.2% | 欧债危机缓解后的反弹（2011-10） |
| 1998-08 | -0.113 | -0.100 | +0.012 | ip +2.0σ, y10 -1.3σ | -15.8% | 俄罗斯违约与 LTCM 危机（1998-08） |
| 2011-09 | -0.111 | -0.114 | -0.003 | usd +3.4σ, labor_share -2.1σ | -7.9% | 欧债危机，美联储扭曲操作（2011-09-21） |

**gold 黄金**

| month | residual | actual | predicted | largest factor moves | market P/D | event |
|---|---|---|---|---|---|---|
| 1999-10 | +0.171 | +0.174 | +0.003 | oil -1.2σ, ip +1.1σ | +6.8% | 华盛顿黄金协议（1999-09-26，欧洲 15 家央行含欧洲央行宣布限制售金），金价急涨 |
| 2011-08 | +0.128 | +0.118 | -0.010 | spread +1.8σ, y10 -1.8σ | -8.6% | 美国主权评级下调（2011-08-05）与欧债危机，金价创新高 |
| 1985-10 | -0.126 | +0.009 | +0.135 | usd -3.1σ, labor_share +1.3σ | +5.0% | 广场协议（1985-09-22）后美元急跌，金价没有随美元反应 |
| 1990-09 | -0.124 | -0.015 | +0.108 | oil +3.5σ, cpi_goods +1.4σ | -4.0% | 海湾危机：伊拉克入侵科威特后油价飙升，金价没有随油价和通胀上涨 |
| 2009-02 | +0.118 | +0.098 | -0.020 | labor_share -3.8σ, usd +1.6σ | -9.1% | 全球金融危机最后一跌（2009-03-09 见底） |
| 2006-06 | -0.117 | -0.117 | -0.000 | usd +0.3σ, oil +0.3σ | +0.8% | 金价从 2006-05 高点回落，新兴市场与大宗商品抛售 |
| 2025-10 | +0.110 | +0.106 | -0.004 | cpi_services -0.8σ, usd +0.6σ | +2.6% | 金价突破 4000 美元（2025-10），央行购金与避险需求 |
| 2008-12 | +0.107 | +0.072 | -0.035 | cpi_goods -4.2σ, labor_share -3.7σ | +2.0% | 全球金融危机：联邦基金利率降到 0–0.25%（2008-12-16），长端收益率暴跌 |
| 2008-10 | +0.106 | -0.028 | -0.133 | spread +5.7σ, usd +4.5σ | -17.3% | 全球金融危机：雷曼倒闭后的崩盘 |
| 2015-01 | +0.105 | +0.042 | -0.063 | cpi_goods -4.0σ, usd +1.9σ | -3.6% | 瑞士央行放弃汇率上限（2015-01-15）、欧洲央行 QE，美元大涨而金价上涨 |

Months in the top-10 list of 3 or more assets: 1987-10 (9), 2009-02 (9), 1998-08 (7), 2008-11 (7), 2008-10 (7), 2020-04 (5), 1987-01 (3), 2002-07 (3), 2009-04 (3)

## F. One-year forecast misses (data/results/forecast.json, analog method)

| origin | asset | realized | p5 | p50 | p95 | cause | market P/D next 12m | why |
|---|---|---|---|---|---|---|---|---|
| 1997-12 | nasdaq | +0.858 | -0.252 | +0.077 | +0.495 | macro | +20.9% | 宏观意外：CPI 同比实际 -0.09pp，预测中位数 +2.67pp；油价实际 -31%，预测中位数 +10% |
| 1998-12 | nasdaq | +1.023 | -0.263 | +0.064 | +0.490 | pricing | +18.3% | 宏观大体符合预测，资产本身偏离：估值或盈利的变化，因子里没有 |
| 1999-12 | utilities | +0.504 | -0.150 | +0.084 | +0.260 | macro | -12.0% | 宏观意外：信用利差实际 +1.15pp，预测中位数 -0.10pp |
| 2007-12 | us_equity | -0.367 | -0.203 | +0.088 | +0.355 | macro | -37.0% | 宏观意外：失业率实际 +2.30pp，预测中位数 -0.30pp；政策利率实际 -4.08pp，预测中位数 -0.03pp；CPI 同比实际 -4.13pp，预测中位数 -0.67pp；信用利差实际 +3.30pp，预测中位数 -0.30pp；油价实际 -54%，预测中位数 -13% |
| 2007-12 | industrials | -0.449 | -0.217 | +0.100 | +0.370 | macro | -37.0% | 宏观意外：失业率实际 +2.30pp，预测中位数 -0.30pp；政策利率实际 -4.08pp，预测中位数 -0.03pp；CPI 同比实际 -4.13pp，预测中位数 -0.67pp；信用利差实际 +3.30pp，预测中位数 -0.30pp；油价实际 -54%，预测中位数 -13% |
| 2007-12 | financials | -0.498 | -0.205 | +0.099 | +0.410 | macro | -37.0% | 宏观意外：失业率实际 +2.30pp，预测中位数 -0.30pp；政策利率实际 -4.08pp，预测中位数 -0.03pp；CPI 同比实际 -4.13pp，预测中位数 -0.67pp；信用利差实际 +3.30pp，预测中位数 -0.30pp；油价实际 -54%，预测中位数 -13% |
| 2007-12 | real_estate | -0.377 | -0.167 | +0.092 | +0.338 | macro | -37.0% | 宏观意外：失业率实际 +2.30pp，预测中位数 -0.30pp；政策利率实际 -4.08pp，预测中位数 -0.03pp；CPI 同比实际 -4.13pp，预测中位数 -0.67pp；信用利差实际 +3.30pp，预测中位数 -0.30pp；油价实际 -54%，预测中位数 -13% |
| 2007-12 | treasury_10y | +0.205 | -0.080 | +0.020 | +0.122 | macro | -37.0% | 宏观意外：失业率实际 +2.30pp，预测中位数 -0.30pp；政策利率实际 -4.08pp，预测中位数 -0.03pp；CPI 同比实际 -4.13pp，预测中位数 -0.67pp；信用利差实际 +3.30pp，预测中位数 -0.30pp；油价实际 -54%，预测中位数 -13% |
| 2010-12 | treasury_10y | +0.162 | -0.082 | +0.022 | +0.103 | macro | -12.8% | 宏观意外：政策利率实际 -0.11pp，预测中位数 +0.70pp |
| 2019-12 | treasury_10y | +0.105 | -0.100 | -0.014 | +0.066 | shock | +22.8% | 外生冲击：新冠疫情与全面封锁 |
| 2008-12 | treasury_10y | -0.102 | -0.058 | +0.034 | +0.111 | macro | +46.3% | 宏观意外：失业率实际 +2.60pp，预测中位数 -0.66pp；信用利差实际 -3.28pp，预测中位数 -1.19pp |
| 2007-12 | crude_oil | -0.535 | -0.421 | -0.135 | +0.455 | macro | -37.0% | 宏观意外：失业率实际 +2.30pp，预测中位数 -0.30pp；政策利率实际 -4.08pp，预测中位数 -0.03pp；CPI 同比实际 -4.13pp，预测中位数 -0.67pp；信用利差实际 +3.30pp，预测中位数 -0.30pp；油价实际 -54%，预测中位数 -13% |
| 1994-12 | healthcare | +0.567 | -0.178 | +0.105 | +0.476 | pricing | +23.4% | 宏观大体符合预测，资产本身偏离：估值或盈利的变化，因子里没有 |
| 1999-12 | nasdaq | -0.368 | -0.259 | +0.070 | +0.604 | macro | -12.0% | 宏观意外：信用利差实际 +1.15pp，预测中位数 -0.10pp |
| 1990-12 | healthcare | +0.635 | -0.200 | +0.147 | +0.495 | shock | +31.4% | 外生冲击：伊拉克入侵科威特与海湾战争 |

Causes: {'macro': 11, 'pricing': 2, 'shock': 2}. Origins with the most risk assets outside the analog 90% band: 2007-12 (8/10), 2012-12 (5/10), 2002-12 (4/10), 1994-12 (3/10), 1990-12 (3/10), 2001-12 (2/10).

**macro_skill: analog CRPS skill vs the unconditional draw and vs the naive normal (all 42 origins)**

| variable | vs uncond | t | vs naive | t | vs naive ≤2004 | vs naive >2004 | skilled |
|---|---|---|---|---|---|---|---|
| unrate 失业率 | +0.1% | 0.2 | +15.3% | 3.4 | +12.7% | +16.9% | False |
| ff 政策利率 | +0.5% | 0.9 | +15.5% | 3.3 | +15.2% | +15.9% | False |
| cpi_yoy CPI 同比 | +1.3% | 1.4 | -45.1% | -3.0 | -86.2% | -11.7% | False |
| spread 信用利差 | -0.6% | -1.0 | +11.2% | 1.3 | +6.2% | +14.5% | False |
| oil 油价 | +0.3% | 0.5 | +3.0% | 0.6 | +3.1% | +2.9% | False |

## G. KEY_STATES linear baseline: OOS R² vs the prevailing mean (monthly origins 1985-01..2025-09)

|  | all | h1 | h2 | CW t (all) |
|---|---|---|---|---|
| unrate | 0.022 | -0.269 | 0.059 | 2.907 |
| cpi_yoy | -0.221 | -1.020 | 0.003 | 2.716 |
| ff | -0.015 | -0.184 | 0.176 | 2.470 |
| ip_yoy | 0.350 | -0.102 | 0.459 | 2.349 |
| spread | -0.045 | 0.096 | -0.088 | 1.889 |
| y10 | -0.383 | 0.039 | -1.263 | 1.519 |
| oil | -0.394 | -0.418 | -0.378 | -1.079 |
| ret_us_equity | -0.479 | -0.936 | -0.063 | -0.710 |
| ret_nasdaq | -0.497 | -0.693 | 0.080 | -1.566 |
| ret_energy | -0.323 | -0.687 | -0.221 | -1.451 |
| ret_industrials | -0.342 | -1.017 | -0.071 | -0.325 |
| ret_staples | -0.612 | -0.496 | -0.819 | 1.482 |
| ret_healthcare | -0.871 | -0.612 | -1.487 | -0.191 |
| ret_utilities | -0.341 | -0.291 | -0.406 | 0.898 |
| ret_financials | -0.341 | -0.693 | -0.154 | -0.183 |
| ret_real_estate | -0.102 | -0.406 | 0.007 | 2.059 |
| ret_gold | -1.214 | -1.977 | -0.834 | -0.408 |
| ret_crude_oil | -0.394 | -0.418 | -0.378 | -1.079 |
| ret_treasury_10y | 0.062 | 0.213 | -0.080 | 3.576 |

## H. Look-ahead in the core labor-share factor (mean OOS R² over the 10 score assets)

|  | T1 | T2 | T3 |
|---|---|---|---|
| base (interpolated labor share, look-ahead) | -0.024 | -0.107 | 0.172 |
| base without labor share | -0.015 | -0.086 | 0.180 |
| labor share release-dated, lag 4 | -0.026 | -0.077 | 0.178 |
| labor share release-dated, lag 5 | -0.011 | -0.094 | 0.166 |

`drop_test('labor_share')`: ΔR² of the base WITHOUT labor share (positive = better without it) and DM t per asset

|  | T1 | T2 | T3 | T1 DM t | T2 DM t | T3 DM t |
|---|---|---|---|---|---|---|
| us_equity | +0.052 | +0.007 | +0.002 | +1.720 | +0.350 | +0.450 |
| nasdaq | +0.037 | +0.016 | +0.008 | +1.420 | +0.910 | +0.910 |
| energy | -0.034 | +0.018 | +0.025 | -1.130 | +0.500 | +0.780 |
| industrials | +0.010 | +0.017 | +0.005 | +0.350 | +0.450 | +0.600 |
| staples | -0.025 | +0.080 | -0.025 | -0.570 | +0.750 | -0.940 |
| healthcare | +0.010 | -0.004 | +0.017 | +0.750 | -0.200 | +1.420 |
| utilities | +0.001 | +0.005 | +0.024 | +0.060 | +0.160 | +1.220 |
| financials | +0.004 | +0.052 | +0.002 | +0.100 | +1.080 | +0.780 |
| real_estate | +0.000 | +0.009 | +0.001 | +0.010 | +0.580 | +0.860 |
| gold | +0.034 | +0.014 | +0.016 | +1.050 | +0.460 | +1.220 |

## I. Existing candidates per asset (transmission ΔR², each alone on the base)

|  | score_T1 | score_T2 | score_T3 | assets_every_decade | resid_sig_both_halves | resid_sig_both_halves_h12 |
|---|---|---|---|---|---|---|
| sentiment | -0.0198 | +0.0448 | -0.0487 | [] | [] | ['industrials', 'financials'] |
| profit_share | -0.0009 | -0.0397 | +0.0015 | [] | [] | [] |
| capex_share | +0.0003 | -0.0065 | +0.0084 | ['real_estate'] | ['energy'] | ['energy', 'utilities'] |
| white_collar | +0.0022 | +0.0086 | -0.0016 | ['us_equity', 'industrials', 'utilities'] | ['healthcare'] | [] |
| wage | -0.0037 | +0.0032 | -0.0124 | ['nasdaq', 'staples'] | [] | [] |
| real_y10 | -0.0253 | +0.0397 | +0.0025 | ['staples'] | [] | ['industrials'] |
| profit_growth | +0.0041 | -0.0171 | -0.0129 | [] | [] | [] |
| fcf_share | +0.0081 | -0.0582 | -0.0118 | [] | [] | [] |

resid_sig_both_halves: |t| > 2 with the same sign in 1985-2004 and 2005-2026, month by month; _h12: the same on overlapping 12-month sums (slow factors). capex_share and profit_share are interpolated quarterly series (look-ahead), and investment and sentiment also respond to stock prices, so an h12 correlation is a pointer, not causal evidence.

Structural-shock block (4 columns jointly): T1 +0.0080 (120 m); T2 +0.0181 (120 m); T3 -0.0104 (131 m)

## J. How much of a decade's miss sits in its crisis months

|  | us_equity | nasdaq | energy | industrials | staples | healthcare | utilities | financials | real_estate | gold | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T2 all months | -0.067 | -0.005 | 0.260 | -0.018 | -0.476 | -0.308 | -0.049 | -0.320 | 0.074 | -0.161 | -0.107 |
| T2 without 2008-09..2009-03 | -0.045 | 0.027 | 0.223 | -0.001 | -0.251 | -0.135 | -0.093 | -0.239 | 0.057 | -0.021 | -0.048 |
| T3 all months | 0.319 | 0.259 | 0.172 | 0.295 | 0.199 | 0.059 | 0.150 | 0.073 | 0.374 | -0.178 | 0.172 |
| T3 without 2020-02..2020-06 | 0.281 | 0.214 | 0.300 | 0.247 | 0.180 | 0.126 | 0.243 | 0.085 | 0.340 | -0.097 | 0.192 |

## K. The rate and goods-inflation betas change sign around 2000 (in-sample coefficients, price return per unit)

y10 and spread per percentage point, cpi_goods and oil per unit of Δlog. The Chow test in diagnostics.json finds the break at 2000-01.

|  | us_equity | nasdaq | energy | industrials | staples | healthcare | utilities | financials | real_estate | gold |
|---|---|---|---|---|---|---|---|---|---|---|
| y10 1973-1999 | -0.045 | -0.051 | -0.031 | -0.041 | -0.047 | -0.035 | -0.058 | -0.063 | -0.043 | 0.011 |
| cpi_goods 1973-1999 | -1.425 | -2.018 | -1.020 | -0.936 | -1.355 | -1.934 | -0.473 | -1.862 | -0.907 | 2.308 |
| spread 1973-1999 | -0.031 | -0.040 | -0.027 | -0.034 | -0.030 | -0.003 | -0.022 | -0.038 | -0.048 | -0.005 |
| oil 1973-1999 | -0.002 | 0.003 | 0.140 | -0.024 | -0.047 | -0.037 | 0.012 | -0.030 | 0.010 | 0.066 |
| y10 2000-2026 | 0.003 | 0.009 | 0.027 | 0.011 | -0.009 | -0.011 | -0.039 | 0.025 | -0.055 | -0.027 |
| cpi_goods 2000-2026 | -0.342 | -0.178 | -0.441 | -1.117 | -0.377 | 0.150 | -0.157 | -0.504 | 0.076 | 0.944 |
| spread 2000-2026 | -0.088 | -0.119 | -0.030 | -0.107 | -0.035 | -0.055 | -0.066 | -0.084 | -0.121 | 0.012 |
| oil 2000-2026 | 0.015 | -0.024 | 0.317 | 0.024 | 0.019 | -0.027 | 0.022 | 0.006 | -0.040 | -0.004 |
