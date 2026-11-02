# Technology and capital-spending cycle: candidate factors

Family report, missing-factor round. Data version: panel sha1 3869fdde3437, model_selection 58a9db10ed83, panel last month 2026-09, harness 1.1. Access date 2026-11-01. Everything here was produced by the scripts in `scripts/` (run with `python -B -I`); nothing in the project was written. Numbers in the tables are read from `tests.json`.

**Files.** `candidates.csv` (every monthly series built, both alignments, wide, `date` column), `sources.json` (provenance of every raw file), `tests.json` (all harness outputs, the follow-up tests, a flat list of every p value under `pvalues`, and BH within the family under `bh_within_family`), `scripts/` (code), `work/` (CSV views of the same results).

## Bottom line

1. **No tech or capex series earns a place as a same-month transmission factor.** 52 reference-month series and 3 blocks (53 series built in all; the panel's own capex share was run alongside as a reference) were added one at a time to the 10-factor base. None raises the mean OOS R² by more than 0.003 in every decade; Benjamini–Hochberg (q = 0.10) inside the family rejects nothing on the Diebold–Mariano tests (1500 asset × decade tests, min p 0.0004) or on the pooled mean-over-assets test (174 tests, min p 0.0041). The large swings of every 1992+ series (orders, shipments, construction: +0.11 to +0.22 in 2005–2014, -0.27 to +0.03 in 2015–2026) are a training-window effect: on the same training months they vanish (§3).

2. **Forecast states: 6 known-at pairs pass the both-halves rule; one has a stable, mechanism-consistent sign: manufacturers' planned capital spending (Philadelphia Fed, 1968+).** A high reading forecasts *lower* 12-month returns (Lamont 2000's investment-plans effect): industrials ΔR² over the KEY_STATES baseline +0.061 (1985–2004) and +0.046 (2005–2025), Clark–West p 0.0025; us_equity +0.060 / +0.014, p 0.0138; negative slope in all three subperiods, and the same with the never-revised NSA series. Of the others, industrial electricity sales → IP growth (+0.038 / +0.057) has a stable negative slope but is a cleaner gauge of the industrial cycle's mean reversion, not a tech channel; machinery producer-price inflation → capex share (+0.004 / +0.093) and → IP growth, and the IT-investment share → market returns, have slopes that are near zero or change sign across subperiods (the IT share also has the wrong sign for an over-investment story and tracks the dividend yield). None is significant after BH inside the family (smallest q 0.25), and the baseline they improve on is itself worse than the prevailing mean for returns: forecast-state candidates for the walk-forward, not findings.

3. **The scenarios' capex anchor is the wrong variable for the AI era.** The AI boom does not show up in US tech *output* (high-tech IP y/y averages 9.5% in 2024-01..2026-09 against 35.5% in 1995–99) but in *investment in information processing equipment and software*: 5.35% of GDP in 2026-Q3, the highest on record (previous peak 4.46% in 2000-12), and computers and peripherals alone 1.53% (previous peak 1.05% in 1999-06). The IT-investment share singles out the 2000–02 bust among all recessions (AUC 0.95; the panel's capex share 0.60). It is the better view variable for agi_boom / ai_bust (§6); data-center construction (Census, 2014+) is the purest AI series but too short for anything except a view.

4. **Tech output news arrives a month late.** The monthly change of high-tech IP correlates with the *next* month's stock return (0.15, t 3.9; same month 0.02); the harness flags it 'check look-ahead'. Entered in the month it is published, computer-and-electronics IP raises the mean OOS R² in all three decades (+0.0114 / +0.0373 / +0.0050), but no decade is significant, nasdaq loses heavily in 2015–2026, and with ALFRED first releases (only available from 2015) the 2015–2026 gain turns negative (-0.0061). Not recommended.

5. **Specific drivers tried after the first nulls** (own input costs, own output prices, own demand): natural gas, coal, the power-price/gas margin and electricity sales for utilities; steel and machinery prices and aircraft orders for industrials; chip and computer prices, capacity utilization, tech employment for nasdaq. None passes after the same-sample and BH checks. Two by-products for other families: producer prices of capital equipment / machinery help **gold** in all three decades (machinery +0.024 / +0.038 / +0.096, combined z p 0.014); and the 'energy' gains of the electricity series are another training-window effect (dropping 1973–75).

## 1. Ranked table

Rule (task): *adds out of sample* = for one asset (dedicated) or the 10-asset mean (common), ΔR² > 0 in at least 2 of the 3 test decades and on average, with a positive mean HAC (Newey–West) Diebold–Mariano t; or, as a predictor, incremental OOS R² over the KEY_STATES baseline > 0 in both halves (here also ≥ 60 origins per half). Decades: T1 1995–2004 (trained 1973–94), T2 2005–14 (1973–2004), T3 2015–26 (1973–2014). 'pooled t' = HAC t of the month-by-month mean-over-assets loss gain (sums exactly to the mean ΔR²). For late starters the comparison is on the candidate's own training months ('same sample').

| # | candidate (alignment) | mechanism | assets / targets | coverage | numbers | verdict | role |
|---|---|---|---|---|---|---|---|
| 1 | philly_capex (known-at, lag 0) | planned capex high ⇒ high valuations / low expected returns (Lamont 2000); sign −, as expected | ret_industrials, ret_us_equity (12 m) | 1968-05.., full | ΔR² vs baseline ind +0.061/+0.046, CW p 0.0025; us_eq +0.060/+0.014, CW p 0.0138; coef -0.0040/pt (t -4.0); NSA same; transmission T1–T3 -0.0112/+0.0051/-0.0024 | **adds as predictor** (both halves; not BH-significant) | forecast state; view variable |
| 2 | ipe_sw_share (IT equipment + software investment / GDP; quarterly held) | IT capex intensity | nasdaq, us_equity; ret_us_equity; scenario anchor | 1947.., full | pred us_eq +0.066/+0.064, CW p 0.0230, coef +0.133/pp (sign opposite to the over-investment story; corr with div yield -0.85); transmission -0.0076/-0.0016/-0.0117; bust-vs-recessions AUC 0.95, AI-era AUC 0.96 | adds as predictor by the letter (likely a div-yield drift fix); no transmission gain | **scenario view variable** (agi_boom / ai_bust); forecast state only after a structural check |
| 3 | comp_share (computers & peripherals investment / GDP) | AI servers are 'computers' in NIPA | nasdaq; scenario anchor | 1959.., full | latest 1.53% (record), 4q change +0.58pp (record); bust-vs-recessions AUC 1.00; pred ind/us_eq strong in 1985–2004 (+0.39) but negative after (-0.04) | no (fails the second half) | scenario view variable |
| 4 | eia_elec_ind y/y (known-at, lag 3) | industrial power use = cleaner real-time industrial-activity gauge; mean reversion | ip_yoy (12 m) | 1974.. (SA from 1976) | ΔR² +0.038/+0.057, CW p 0.0224; alone OOS R² 0.25; coef -0.22 (t -2.4); slope by subperiod 1973–89 -0.188 (t -2.7); 1990–04 -0.188 (t -2.1); 2005–25 -0.169 (t -1.5) | **adds as predictor** (ip_yoy only) | forecast state (not tech-specific) |
| 5 | ppi_machinery y/y (known-at, lag 1) | capital-goods price inflation = strong equipment demand; nominal capex share also carries the relative price | capex_share, ip_yoy (12 m) | 1948.., full | capex_share ΔR² +0.004/+0.093, CW p 0.0098; ip_yoy +0.041/+0.036, CW p 0.0550; slope by subperiod capex: 1973–89 -0.024 (t -0.9); 1990–04 +0.034 (t 0.6); 2005–25 +0.090 (t 1.4); ip_yoy: 1973–89 -0.175 (t -0.9); 1990–04 -1.593 (t -1.7); 2005–25 +0.846 (t 1.6) | adds as predictor by the letter; **unstable slope** (≈0 or sign change before 2005) | none now; re-check in the walk-forward |
| 6 | ip_compelec / ip_semis / ip_hitech (release month, lag 1) | tech output news reaches prices when published | 10-asset mean; nasdaq | 1967/1972.., full | compelec +0.0114/+0.0373/+0.0050 (pooled t 1.04/0.79/0.21); semis +0.0007/+0.0652/+0.0167; nasdaq T3 -0.252 (hitech); T3 with first releases -0.0061 / -0.0001 | adds by the letter in today's vintage; **fails the real-time check** | none now (re-test with first releases if older vintages become available) |
| 7 | ip_commeq / ip_compelec / cu_hitech (reference month) | tech output ↔ utilities (power demand? 2001–02 merchant-power bust?) | utilities | 1967/1972.., full | ip_commeq utilities +0.009/+0.028/+0.028, combined z p 0.016 (BH q 0.31); ip_compelec +0.001/+0.019/+0.029, p 0.009 (q 0.20); holds without 2001–02 and without 2024–26 | dedicated by the letter; not BH-significant; weak mechanism | none (note for the utilities/power family) |
| 8 | ppi_machinery / ppi_capeq (reference month) | capital-goods inflation (not a tech mechanism) | gold (by-product) | 1950/1947.., full | machinery→gold +0.024/+0.038/+0.096 (p 0.014, BH q 0.28); capeq→gold +0.016/+0.017/+0.049; mean over assets machinery -0.0001/+0.0195/+0.0107 | dedicated (gold) by the letter, not BH-significant | hand to the inflation / gold family |
| 9 | capex_share_rt (PNFI/GDP, no interpolation) vs panel f_capex_share | the scenarios' capex variable without look-ahead | industrials, us_equity | 1947.., full | panel (interpolated) mean +0.0003/-0.0065/+0.0084, industrials +0.016/-0.008/+0.012; honest rebuild -0.0003/-0.0259/+0.0023, industrials +0.013/-0.039/+0.005 | no (the interpolated version's small gains disappear without look-ahead) | view variable (use the release-dated level) |
| 10 | core capex orders / shipments / backlog, tech orders, aircraft orders (M3, 1992+) | firms' investment decisions | industrials, us_equity, nasdaq; ip_yoy | 1992.., not full | raw T2/T3 +0.1826/-0.2260; same sample -0.0208/+0.0236; pred ip_yoy 2005–25 -0.019 | no | view variable (1992+); could feed a nowcast |
| 11 | cons_datacenter (Census C30, 2014+) | AI capex in structures; power demand | utilities, industrials, nasdaq | 2014-01.., 152 m | latest y/y 55%; $85.0 bn SAAR in 2026-08 vs $1.6 bn in 2014-01; untestable (11 training months before 2015) | untestable | scenario view variable only |
| 12 | ip_hitech, ip_semis, cu_hitech, ip_bus_eq, ppi_semis … (others, reference month) | tech / capex activity, prices | nasdaq, industrials, us_equity, utilities | mostly 1967/1972.., full | see §2: full-history series passing the common rule (raw, mean ΔR² over T1–T3): ppi_machinery +0.0100, ipe_sw_real +0.0025, relprice_ipe +0.0001 — no decade significant; ip_hitech, cu_hitech flagged 'check look-ahead' | no | view variables (ip_hitech y/y identifies the 1995–99 boom, AUC 0.995) |

## 2. Transmission test, every reference-month candidate

Monthly change in the month the data describe (quarterly NIPA: from the quarter's last month, held). score = mean ΔR² over the 10 score assets; t = pooled HAC t; 'rule' = the task's common-factor rule; 'assets passing' = dedicated rule (raw, before BH; for late starters these compare training windows, see §3); same-sample = base refitted on the candidate's training months (late starters only). Flag = `endogeneity_check`. Screen = harness `screen()` verdict.

| candidate | first | full | T1 score (t) | T2 score (t) | T3 score (t) | rule | assets passing | same-sample T2 / T3 | flag | screen |
|---|---|---|---|---|---|---|---|---|---|---|
| ip_hitech | 1967-02 | yes | -0.0151 (-3.67) | +0.0220 (+1.18) | +0.0004 (+0.02) | – | gold, real_estate | – | check look-ahead | endogenous |
| ip_semis | 1972-02 | yes | -0.0078 (-2.08) | +0.0116 (+1.34) | +0.0024 (+0.13) | – | energy, real_estate, utilities | – | ok | no-gain |
| ip_computers | 1972-02 | yes | -0.0029 (-1.27) | -0.0145 (-1.41) | -0.0065 (-2.36) | – | – | – | ok | no-gain |
| ip_commeq | 1972-02 | yes | -0.0168 (-2.65) | +0.0050 (+1.08) | +0.0038 (+0.54) | – | gold, real_estate, utilities | – | ok | no-gain |
| ip_compelec | 1972-02 | yes | -0.0138 (-2.84) | +0.0100 (+0.93) | +0.0061 (+0.47) | – | real_estate, utilities | – | ok | no-gain |
| ip_infoproc_eq | 1967-02 | yes | -0.0204 (-2.63) | -0.0076 (-1.39) | +0.0011 (+0.48) | – | gold, real_estate | – | ok | no-gain |
| ip_semis_mat | 1954-02 | yes | -0.0099 (-3.29) | +0.0266 (+1.20) | -0.0062 (-0.40) | – | industrials, staples, utilities | – | ok | no-gain |
| ip_bus_eq | 1947-02 | yes | -0.0163 (-1.80) | +0.0075 (+0.58) | -0.0002 (-0.02) | – | energy, financials, healthcare, industrials, us_equity | – | spanned by core factors | no-gain |
| ip_mfg_xhitech | 1967-02 | yes | +0.0195 (+2.69) | -0.0838 (-1.88) | -0.0041 (-0.68) | – | – | – | spanned by core factors | no-gain |
| cu_hitech | 1967-02 | yes | -0.0038 (-1.48) | +0.0039 (+0.74) | +0.0019 (+0.33) | – | financials, gold, real_estate, utilities | – | check look-ahead | endogenous |
| cu_semis | 1972-02 | yes | -0.0029 (-0.72) | -0.0023 (-1.26) | +0.0035 (+0.44) | – | financials, healthcare, real_estate, utilities | – | ok | no-gain |
| cu_compelec | 1972-02 | yes | -0.0069 (-1.19) | +0.0003 (+0.28) | +0.0017 (+0.90) | – | real_estate, utilities | – | ok | no-gain |
| cap_hitech | 1967-02 | yes | – | – | – | excluded (smooth, interpolated) | – | – | – | – |
| ip_electric | 1972-02 | yes | +0.0018 (+0.92) | -0.0107 (-1.54) | -0.0010 (-0.95) | – | real_estate | – | ok | no-gain |
| ip_elec_td | 1972-02 | yes | +0.0008 (+0.41) | -0.0123 (-1.82) | -0.0020 (-1.60) | – | – | – | ok | no-gain |
| cu_electric | 1967-02 | yes | +0.0033 (+1.35) | -0.0118 (-1.37) | -0.0021 (-1.04) | – | utilities | – | ok | no-gain |
| orders_core | 1992-03 | no | too short | +0.1826 (+1.72) | -0.2260 (-0.93) | – | gold | -0.0208 / +0.0236 | ok | no-gain |
| ship_core | 1992-02 | no | too short | +0.1960 (+1.73) | -0.1844 (-0.90) | – | gold | -0.0085 / +0.0630 | ok | no-gain |
| backlog_core | 1992-02 | no | too short | +0.1558 (+1.69) | -0.1416 (-0.85) | – | gold | -0.0487 / +0.1057 | ok | no-gain |
| orders_tech | 1992-03 | no | too short | +0.1634 (+1.63) | -0.2666 (-0.97) | – | gold | -0.0400 / -0.0170 | ok | no-gain |
| ship_tech | 1992-02 | no | too short | +0.1094 (+0.93) | -0.2217 (-0.94) | – | gold | -0.0950 / +0.0256 | ok | no-gain |
| btb_core | 1992-03 | no | too short | +0.1983 (+1.79) | -0.2486 (-0.94) | – | gold | -0.0052 / +0.0009 | ok | no-gain |
| orders_aircraft | 1992-03 | no | too short | +0.2220 (+1.85) | +0.0349 (+1.26) | pass | energy, financials, gold, healthcare, industrials, us_equity, utilities | +0.0027 / +0.0039 | ok | no-gain |
| philly_capex | 1968-06 | yes | -0.0112 (-1.57) | +0.0051 (+0.98) | -0.0024 (-0.47) | – | financials, nasdaq | – | ok | forecast-state |
| ppi_semis | 1981-07 | no | -0.0199 (-1.04) | +0.0186 (+0.87) | -0.0432 (-0.80) | – | gold | +0.0045 / -0.0010 | ok | no-gain |
| ppi_computers | 1991-01 | no | too short | +0.1615 (+2.08) | -0.1114 (-0.86) | – | gold | +0.0020 / +0.0192 | ok | no-gain |
| ppi_capeq | 1947-05 | yes | +0.0022 (+0.48) | -0.0148 (-1.64) | +0.0039 (+1.58) | – | gold | – | ok | no-gain |
| ppi_electric | 1973-02 | yes | -0.0004 (-0.08) | +0.0159 (+1.29) | -0.0014 (-0.14) | – | energy, healthcare, nasdaq, staples, utilities | – | ok | no-gain |
| ppi_transformer | 1967-02 | yes | -0.0016 (-0.61) | -0.0130 (-0.94) | -0.0083 (-1.86) | – | energy, nasdaq, utilities | – | ok | no-gain |
| ppi_natgas | 1970-02 | yes | -0.0649 (-1.43) | -0.0154 (-0.96) | -0.0301 (-2.71) | – | – | – | ok | no-gain |
| ppi_coal | 1947-02 | yes | -0.0808 (-3.46) | -0.0017 (-0.33) | -0.0043 (-0.63) | – | – | – | ok | no-gain |
| util_margin | 1973-02 | yes | -0.0763 (-1.43) | -0.0145 (-0.83) | -0.0326 (-2.74) | – | – | – | ok | no-gain |
| ppi_steel | 1947-02 | yes | -0.0112 (-0.71) | -0.0908 (-1.28) | +0.0043 (+0.64) | – | gold | – | ok | no-gain |
| ppi_machinery | 1950-02 | yes | -0.0001 (-0.02) | +0.0195 (+1.01) | +0.0107 (+0.53) | pass | energy, financials, gold, healthcare, real_estate, us_equity | – | ok | no-gain |
| emp_semis | 1985-02 | no | too short | +0.0525 (+1.25) | -0.0424 (-0.69) | – | energy, gold | -0.0220 / +0.0761 | ok | no-gain |
| emp_compelec | 1990-02 | no | too short | +0.0346 (+0.76) | -0.0718 (-0.72) | – | gold | -0.0539 / +0.1003 | ok | no-gain |
| emp_csd | 1990-02 | no | too short | +0.0864 (+1.64) | -0.1173 (-0.84) | – | gold | -0.0021 / +0.0548 | ok | no-gain |
| cons_mfg | 1993-02 | no | too short | +0.1627 (+1.70) | -0.2380 (-0.94) | – | gold | -0.0285 / -0.0052 | ok | no-gain |
| cons_office | 1993-02 | no | too short | +0.1573 (+1.56) | -0.1754 (-0.93) | – | gold | -0.0339 / +0.0573 | ok | no-gain |
| cons_datacenter | 2014-02 | no | – | – | – | untestable (too short) | – | – | – | – |
| cons_compelec | 1993-02 | no | too short | +0.1637 (+1.84) | -0.2514 (-0.94) | – | gold | -0.0275 / -0.0187 | ok | no-gain |
| cons_electric | 1993-02 | no | too short | +0.1824 (+1.75) | -0.2171 (-0.92) | – | gold | -0.0089 / +0.0156 | ok | no-gain |
| eia_elec_comm | 1976-02 | no | -0.0095 (-0.53) | +0.0217 (+0.60) | +0.0015 (+0.22) | pass raw / fail same-sample | energy, gold, us_equity | -0.0016 / -0.0004 | ok | no-gain |
| eia_elec_ind | 1976-02 | no | -0.0204 (-1.15) | +0.0309 (+0.76) | -0.0144 (-1.72) | – | energy, gold | +0.0075 / -0.0163 | ok | forecast-state |
| eia_elec_total | 1976-02 | no | -0.0038 (-0.24) | +0.0140 (+0.41) | -0.0012 (-0.17) | – | energy, gold, us_equity | -0.0094 / -0.0031 | ok | no-gain |
| ipe_sw_share | 1947-04 | yes | -0.0076 (-1.54) | -0.0016 (-0.25) | -0.0117 (-0.99) | – | – | – | ok | forecast-state |
| ipe_share | 1947-04 | yes | -0.0101 (-1.98) | -0.0011 (-0.20) | -0.0027 (-0.22) | – | financials | – | ok | no-gain |
| comp_share | 1959-04 | yes | +0.0028 (+0.68) | -0.0084 (-0.98) | -0.0229 (-1.43) | – | – | – | ok | no-gain |
| sw_share | 1959-04 | yes | -0.0015 (-0.09) | -0.0117 (-0.80) | -0.0227 (-1.69) | – | – | – | ok | no-gain |
| rnd_share | 1947-04 | yes | -0.0118 (-2.02) | -0.0218 (-1.37) | -0.0021 (-0.82) | – | – | – | ok | no-gain |
| capex_share_rt | 1947-04 | yes | -0.0003 (-0.12) | -0.0259 (-1.79) | +0.0023 (+1.05) | – | – | – | ok | no-gain |
| ipe_sw_real | 1947-04 | yes | -0.0006 (-0.18) | +0.0061 (+1.09) | +0.0018 (+0.17) | pass | energy, financials, healthcare, industrials, real_estate, us_equity, utilities | – | ok | no-gain |
| relprice_ipe | 1947-04 | yes | -0.0013 (-0.44) | +0.0006 (+0.45) | +0.0011 (+0.38) | pass | energy, nasdaq, real_estate, utilities | – | ok | no-gain |
| block_utilities_price_fuel | 1973 | yes | -0.1369 (-2.50) | -0.0036 (-0.14) | -0.0339 (-2.06) | – | – | – | block | – |
| block_industrials_steel_machinery | 1973 | yes | -0.0107 (-0.73) | -0.0501 (-1.33) | +0.0096 (+0.41) | – | gold | – | block | – |
| block_capex_output_intentions | 1973 | yes | -0.0278 (-2.04) | +0.0127 (+0.96) | -0.0016 (-0.11) | – | energy, financials, us_equity | – | block | – |
| panel_f_capex_share | 1973 | yes | +0.0003 (+0.06) | -0.0065 (-0.36) | +0.0084 (+0.47) | pass | energy, financials, industrials, real_estate, us_equity | – | ok | – |

`cap_hitech` (capacity growth) is excluded from transmission: G.17 capacity is interpolated from annual estimates (lag-1 autocorrelation of its monthly change 0.99), not a same-month factor. Blocks are tested jointly (`transmission_test` with a DataFrame).

## 3. Late starters: the training-window effect

Every series that starts in 1992–93 shows a mean ΔR² of +0.11 to +0.22 in T2 and -0.27 to +0.03 in T3 against the base trained from 1973 — orders, shipments, backlog, book-to-bill, construction alike. The difference between the raw and the same-sample delta is the effect of the training window alone. Refitting the base on the candidate's own training months (harness `delta_same_sample`, with DM t added here via `extra_stats.oos_errors`) removes it:

| candidate | raw T2 / T3 | same-sample T2 / T3 | same-sample pooled t T2 / T3 | common rule on the same sample |
|---|---|---|---|---|
| backlog_core | +0.1558 / -0.1416 | -0.0487 / +0.1057 | -1.57 / +1.07 | – |
| btb_core | +0.1983 / -0.2486 | -0.0052 / +0.0009 | -0.85 / +0.85 | – |
| cons_compelec | +0.1637 / -0.2514 | -0.0275 / -0.0187 | -1.29 / -1.06 | – |
| cons_electric | +0.1824 / -0.2171 | -0.0089 / +0.0156 | -0.74 / +1.20 | – |
| cons_mfg | +0.1627 / -0.2380 | -0.0285 / -0.0052 | -1.75 / -1.21 | – |
| cons_office | +0.1573 / -0.1754 | -0.0339 / +0.0573 | -2.29 / +0.93 | – |
| eia_elec_comm | +0.0217 / +0.0015 | -0.0016 / -0.0004 | -0.61 / -0.27 | – |
| eia_elec_ind | +0.0309 / -0.0144 | +0.0075 / -0.0163 | +0.73 / -3.19 | – |
| eia_elec_total | +0.0140 / -0.0012 | -0.0094 / -0.0031 | -1.50 / -1.59 | – |
| emp_compelec | +0.0346 / -0.0718 | -0.0539 / +0.1003 | -1.70 / +1.05 | – |
| emp_csd | +0.0864 / -0.1173 | -0.0021 / +0.0548 | -0.10 / +1.00 | – |
| emp_semis | +0.0525 / -0.0424 | -0.0220 / +0.0761 | -1.59 / +1.04 | – |
| orders_aircraft | +0.2220 / +0.0349 | +0.0027 / +0.0039 | +0.73 / +0.78 | pass |
| orders_core | +0.1826 / -0.2260 | -0.0208 / +0.0236 | -1.26 / +0.98 | – |
| orders_tech | +0.1634 / -0.2666 | -0.0400 / -0.0170 | -1.96 / -1.76 | – |
| ppi_computers | +0.1615 / -0.1114 | +0.0020 / +0.0192 | +0.21 / +0.62 | pass |
| ppi_semis | +0.0186 / -0.0432 | +0.0045 / -0.0010 | +1.34 / -0.21 | – |
| ship_core | +0.1960 / -0.1844 | -0.0085 / +0.0630 | -0.62 / +1.05 | – |
| ship_tech | +0.1094 / -0.2217 | -0.0950 / +0.0256 | -2.66 / +0.90 | – |

On the same sample no late starter passes for any asset after BH (same-sample combined-z family: 190 pairs, min p 0.0045, 0 rejections). The raw-delta 'gold' passes of these series (and the 'energy' passes of the EIA electricity series, which start in 1976 after the seasonal-adjustment warm-up and so drop the 1973–75 oil shock from training) are all window effects. For the project this is a property of the base, not of the factors: training on 1992–2004 instead of 1973–2004 helps 2005–2014 and training on 1992–2014 hurts 2015–2026.

## 4. Release-month variants (the change published during the month)

`<name>@rel` = `published(ref_chg, lag)`. Motivation: the high-tech IP change leads next month's market. Same harness test.

| candidate | T1 score (t) | T2 score (t) | T3 score (t) | nasdaq T1 / T2 / T3 |
|---|---|---|---|---|
| orders_aircraft@rel | too short | +0.2282 (+1.92) | +0.0247 (+0.85) | – / -0.009 / -0.142 |
| ip_hitech@rel | +0.0248 (+1.54) | +0.0643 (+0.90) | -0.0003 (-0.01) | +0.038 / +0.098 / -0.252 |
| ip_semis@rel | +0.0007 (+0.06) | +0.0652 (+1.19) | +0.0167 (+0.52) | +0.014 / +0.121 / -0.146 |
| ip_semis_mat@rel | +0.0017 (+0.23) | +0.0541 (+1.25) | +0.0137 (+0.43) | +0.003 / +0.100 / -0.121 |
| ip_compelec@rel | +0.0114 (+1.04) | +0.0373 (+0.79) | +0.0050 (+0.21) | +0.020 / +0.062 / -0.144 |
| backlog_core@rel | too short | +0.1769 (+1.68) | -0.1446 (-0.84) | – / -0.096 / -0.063 |
| cu_semis@rel | -0.0084 (-0.60) | +0.0253 (+0.73) | +0.0273 (+1.14) | +0.014 / +0.052 / -0.064 |
| ip_commeq@rel | +0.0242 (+1.32) | +0.0089 (+0.27) | -0.0017 (-0.10) | +0.033 / -0.042 / -0.084 |
| ppi_machinery@rel | +0.0020 (+0.25) | +0.0254 (+1.35) | +0.0010 (+0.06) | +0.000 / +0.043 / -0.065 |
| cu_hitech@rel | +0.0064 (+0.32) | -0.0012 (-0.03) | +0.0224 (+0.94) | +0.035 / -0.007 / -0.117 |
| cu_compelec@rel | +0.0021 (+0.17) | +0.0011 (+0.05) | +0.0157 (+0.91) | +0.016 / +0.002 / -0.056 |
| ppi_electric@rel | -0.0009 (-0.36) | +0.0105 (+1.97) | +0.0048 (+0.61) | +0.006 / +0.028 / -0.044 |
| eia_elec_ind@rel | -0.0060 (-0.39) | +0.0212 (+0.57) | -0.0023 (-0.26) | -0.019 / +0.020 / -0.023 |
| eia_elec_total@rel | -0.0044 (-0.27) | +0.0151 (+0.47) | +0.0002 (+0.02) | -0.017 / +0.004 / -0.008 |

Rows for 1992+ series (orders_aircraft, backlog_core) carry the training-window effect of §3.

Real-time check (ALFRED; vintages of these series start 2015-02, so only T3 can be checked): first-release monthly changes correlate 0.27 (semiconductors) and 0.75 (NAICS 334) with today's values. T3 mean ΔR² with first-release inputs: ip_compelec@rel -0.0061 (today's vintage +0.0050), ip_semis@rel -0.0001 (today's +0.0167). The lead of tech IP over returns is concentrated in 1995–2014 (corr of the high-tech IP change with next month's nasdaq return 0.28 in 1995–2004, 0.23 in 2005–14, 0.04 in 2015–26) and may partly be revisions that bring in later information, which a release-month factor would then leak. Not adoptable on this evidence.

## 5. Predictive tests (known-at levels / y/y; next 12 months)

Targets: ip_yoy, unrate, ret_nasdaq, ret_industrials, ret_us_equity, ret_utilities (harness) and capex_share (custom: next-12-month change of PNFI/GDP, reference quarter, no interpolation; `extra_stats.predictive_custom`, a replica of `predictive_test` that reproduces it exactly on ip_yoy). Monthly origins 1985-01..2025-09; ΔR² = incremental OOS R² over the KEY_STATES baseline, halves ≤2004 / ≥2005. Only pairs with ΔR² > 0 in both halves and ≥ 60 origins in each half count.

| known-at series | target | ΔR² 1985–2004 | ΔR² 2005–25 | CW p (all) | BH q | year-end ΔR² h1 / h2 | coef (in-sample t) | slope by subperiod (t) | candidate model vs mean | alone vs mean |
|---|---|---|---|---|---|---|---|---|---|---|
| philly_capex.known_level | ret_industrials | +0.061 | +0.046 | 0.0025 | 0.25 | +0.085 / +0.061 | -0.0040 (-4.0) | 1973–89 -0.0008 (-0.8); 1990–04 -0.0020 (-1.2); 2005–25 -0.0032 (-2.1) | -0.272 | -0.009 |
| ppi_machinery.known_yoy | capex_share | +0.004 | +0.093 | 0.0098 | 0.51 | +0.005 / +0.167 | 0.0662 (2.1) | 1973–89 -0.0239 (-0.9); 1990–04 +0.0340 (0.6); 2005–25 +0.0898 (1.4) | +0.283 | -0.079 |
| philly_capex.known_level | ret_us_equity | +0.060 | +0.014 | 0.0138 | 0.56 | +0.077 / +0.039 | -0.0033 (-3.3) | 1973–89 -0.0001 (-0.2); 1990–04 -0.0026 (-1.4); 2005–25 -0.0022 (-1.5) | -0.415 | -0.052 |
| eia_elec_ind.known_yoy | ip_yoy | +0.038 | +0.057 | 0.0224 | 0.56 | +0.010 / +0.115 | -0.2232 (-2.4) | 1973–89 -0.1877 (-2.7); 1990–04 -0.1885 (-2.1); 2005–25 -0.1692 (-1.5) | +0.348 | +0.254 |
| ipe_sw_share.known_level | ret_us_equity | +0.066 | +0.064 | 0.0230 | 0.56 | +0.053 / +0.021 | 0.1335 (3.5) | 1973–89 +0.0823 (1.9); 1990–04 -0.4366 (-2.7); 2005–25 +0.4078 (3.0) | -0.382 | -0.164 |
| ppi_machinery.known_yoy | ip_yoy | +0.041 | +0.036 | 0.0550 | 0.87 | +0.034 / +0.070 | 0.4187 (2.3) | 1973–89 -0.1747 (-0.9); 1990–04 -1.5927 (-1.7); 2005–25 +0.8464 (1.6) | +0.374 | -0.134 |

BH q: Benjamini–Hochberg over the 413 Clark–West p values of this family (all known-at series × 7 targets, monthly origins, full period). Slope: in-sample coefficient of the candidate on top of the KEY_STATES (Newey–West 12) within each subperiod.

Pairs with ΔR² > 0 in both halves but fewer than 60 first-half origins (1992+ series) are not counted: orders_core.known_yoy → ret_utilities (11 first-half origins); orders_tech.known_yoy → ret_us_equity (11 first-half origins); orders_tech.known_yoy → ret_utilities (11 first-half origins); btb_core.known_level → ip_yoy (23 first-half origins); orders_aircraft.known_yoy → unrate (11 first-half origins). Targets that no candidate forecasts in both halves (counted pairs): unrate, ret_nasdaq, ret_utilities (capex_share: best incremental CW p 0.010). BH inside the family on the predictive CW p values: 354 tests, 0 rejections (min p 0.00055).

**Philadelphia Fed capex intentions, robustness.** NSA (unrevised) level: industrials +0.056 / +0.042 (CW t 2.73), us_equity +0.056 / +0.015 (CW t 2.19). In-sample slope on top of the KEY_STATES by subperiod (ret_industrials): 1973-1989 -0.0008 (t -0.8); 1990-2004 -0.0020 (t -1.2); 2005-2025 -0.0032 (t -2.1). Negative in every subperiod; one standard deviation (11.3 points) × the in-sample slope -0.0040 ≈ -4.5% (log) on the next 12-month industrials return. Correlation with the dividend yield 0.00, with IP growth 0.43. The KEY_STATES baseline is overfit (its OOS R² on these pairs: industrials -0.34, us_equity -0.48), so the candidate model still loses to the prevailing mean overall; the candidate alone beats the mean after 2004 (industrials +0.049, us_equity +0.030) but not before.

**IT-investment share → us_equity** has the 'wrong' sign for an over-investment mechanism (+0.133 per pp of GDP) and correlates -0.85 with the dividend yield in the baseline: it most likely offsets the dividend yield's downward drift (buybacks) rather than adding a capex channel. Reported, not recommended without that check.

## 6. Scenario anchors: agi_boom and ai_bust

AUC = probability that a month inside the window has a higher value than a month outside it (1973-01..2026-09; 0.5 = no separation, values below 0.5 = lower inside). 'bust vs recessions' compares 2000-04..2002-10 with the other NBER recession months. Latest = last observed month; pctile = share of 1973+ months below it.

| variable | first | latest (month) | pctile | 1995–99 mean | AUC 1995–99 | 2000–02 mean | AUC 2000–02 | AUC bust vs recessions | 2024–26 mean | AUC 2024–26 |
|---|---|---|---|---|---|---|---|---|---|---|
| panel s_capex_share (interpolated) | 1973-01 | 14.98 (2026-09) | 0.988 | 13.38 | 0.60 | 13.61 | 0.66 | 0.60 | 14.20 | 0.89 |
| panel f_capex_share x12 (annualized) | 1973-01 | 0.80 (2026-09) | 0.935 | 0.45 | 0.73 | -0.66 | 0.26 | 0.33 | 0.23 | 0.54 |
| panel s_ip_yoy | 1973-01 | 2.25 (2026-09) | 0.496 | 5.40 | 0.83 | -0.27 | 0.32 | 0.65 | 0.54 | 0.31 |
| panel s_productivity_yoy | 1973-01 | 0.82 (2026-09) | 0.253 | 2.46 | 0.64 | 3.38 | 0.81 | 0.95 | 2.43 | 0.67 |
| ip_hitech y/y | 1973-01 | 16.17 (2026-09) | 0.572 | 35.54 | 0.99 | 11.68 | 0.42 | 0.50 | 9.47 | 0.32 |
| ip_semis y/y | 1973-01 | 18.15 (2026-09) | 0.598 | 41.53 | 0.98 | 21.71 | 0.59 | 0.70 | 10.50 | 0.35 |
| cu_hitech | 1973-01 | 72.87 (2026-09) | 0.264 | 83.12 | 0.81 | 71.32 | 0.37 | 0.34 | 73.41 | 0.27 |
| ipe_sw_share (IT capex % GDP) | 1973-01 | 5.35 (2026-09) | 0.998 | 3.66 | 0.65 | 4.04 | 0.84 | 0.95 | 4.46 | 0.96 |
| ipe_sw_share 4q change | 1973-01 | 0.77 (2026-09) | 0.998 | 0.19 | 0.80 | -0.18 | 0.35 | 0.37 | 0.32 | 0.77 |
| comp_share (computers capex % GDP) | 1973-01 | 1.53 (2026-09) | 0.998 | 0.93 | 0.95 | 0.87 | 0.89 | 1.00 | 0.86 | 0.70 |
| comp_share 4q change | 1973-01 | 0.58 (2026-09) | 0.998 | 0.06 | 0.72 | -0.10 | 0.19 | 0.22 | 0.24 | 0.89 |
| capex_share_rt (PNFI/GDP, no interpolation) | 1973-01 | 14.98 (2026-09) | 0.989 | 13.32 | 0.58 | 13.76 | 0.72 | 0.62 | 14.14 | 0.88 |
| capex_share_rt 4q change | 1973-01 | 0.83 (2026-09) | 0.933 | 0.46 | 0.74 | -0.55 | 0.31 | 0.32 | 0.20 | 0.51 |
| orders_tech y/y (1993+) | 1993-02 | 15.37 (2026-09) | 0.955 | 7.67 | 0.75 | -13.42 | 0.23 | 0.39 | 6.21 | 0.68 |
| philly_capex | 1973-01 | 37.10 (2026-09) | 0.947 | 16.56 | 0.42 | 12.86 | 0.33 | 0.56 | 23.21 | 0.61 |
| cons_datacenter y/y (2015+) | 2015-01 | 54.92 (2026-08) | 0.893 | – | – | – | – | – | 42.59 | 0.75 |
| eia_elec_comm y/y | 1974-01 | 3.32 (2026-06) | 0.611 | 3.78 | 0.65 | 2.71 | 0.53 | 0.64 | 3.04 | 0.58 |

Reading: today's views use `f_capex_share` (aggregate nonresidential investment / GDP, interpolated), `s_ip_yoy` and `s_productivity_yoy`. High-tech IP growth identifies the 1995–99 boom almost perfectly (AUC 0.995) but not the AI era (0.32): US chip and hardware output is not where AI capex lands (imported accelerators and servers). The AI era shows up in the **IT-investment share** (record 5.35% of GDP; AUC 0.96) and in **computers and peripherals investment** (record 1.53%; 4-quarter change +0.58 pp, AUC 0.89), and in data-center construction (Census). Recommendations for the implementation round, as views (no OOS claim):

* agi_boom phase 1: replace or complement `f_capex_share +0.35/yr` with a view on the release-dated IT-investment share (`s_ipe_sw_share`: its 4-quarter change, or its level). By calendar-year averages, an IT-share rise of at least 0.15 pp/yr selects 1977–1980, 1982, 1984, 1995–2000, 2025–2026; the aggregate view (capex share +0.35 pp/yr or more) also selects 1973, 2005–2006, 2011, 2014, 2023, years in which the IT share rose 0.10 pp or less (capex upswings outside IT), so the IT share is the more specific anchor for a tech capex boom.
* ai_bust phase 1: the IT-investment share falling from a high level singles out 2000-04..2002-10 among all recessions (AUC 0.95 for the level, computers 1.00), while the aggregate capex-share change, which falls in every recession, separates the bust less well (AUC 0.33, i.e. 0.67 in the falling direction).
* Caveat for both: entropy pooling can only reweight history; a view at today's record level (5.35%) leaves only 2025–26 months, so views should be relative (1999–2001 range 3.76–4.46%). Data-center construction (2014+) and core capex orders (1992+) can be view variables only over their own months.

## 7. Engine fit and roles

| candidate group | covers 1973-01 on? | honest long proxy | role it could play |
|---|---|---|---|
| G.17 high-tech IP, CU, business equipment, electric power (1967/1972+) | yes | – | core-eligible but no OOS gain; view variables; ip_hitech y/y as the 1995–99 boom identifier |
| Philadelphia Fed capex intentions (1968+) | yes | – | **forecast state** (12-month industrials / market returns) |
| NIPA IT investment shares (quarterly 1947/1959+, held from release) | yes (quarterly) | – | **scenario view variable** for the AI scenarios; poor same-month factor (changes 4 times a year) |
| Census M3 orders, shipments (1992+) | no | IP business equipment (1947+) and Philly capex (1968+) cover the same cycle; no splice needed since the 1992+ series add nothing on the same sample | view variable over 1992+ |
| BLS PPI semiconductors (1981+), computers (1990+) | no | – | none |
| BLS PPI capital equipment, machinery, transformers, power, gas, coal, steel (1947–1973+) | yes | – | none for this family; capex-goods prices → gold for the inflation family |
| CES tech employment (1985/1990+) | no | – | none |
| Census C30 construction (1993+; data centers 2014+) | no | – | view variable only |
| EIA electricity retail sales (1973+, NSA; SA here from 1976) | 3 years short after SA warm-up | G.17 electric power IP (1972+, SA) | eia_elec_ind y/y as a forecast state for ip_yoy |

## 8. Data, alignment and harness notes

* **Alignments.** Reference month = the month described (G.17, M3, PPI, CES, C30, EIA, Philly). Quarterly NIPA: the quarter's value from its last month, held to the next quarter (no interpolation). Known-at: monthly `published(x, lag)` with lag 1 (G.17, M3, PPI, CES), 2 (C30), 3 (EIA MER), 0 (Philly, released in the survey month); NIPA `hold_quarterly(q, 4)` (advance estimate by quarter start + 3; lag 4 = second estimate, conservative). Lags are backed by the FRED `last_updated` dates of the latest releases (sources.json). Data are today's vintage except the ALFRED and NSA checks.
* **Seasonal adjustment.** EIA retail sales, the natural gas PPI and the machinery PPI are NSA; their monthly change is adjusted with the mean change of the same calendar month over the previous 10 years (past data only); y/y rates use the NSA data. Other NSA PPIs (semiconductors, computers, transformers, coal, steel) have calendar-month R² ≤ 0.035 and are used as published.
* **Daily/weekly data**: none in this family.
* **Original publisher checks.** The G.17 text files (federalreserve.gov) equal the FRED copies exactly for all 15 G.17 series compared. BEA has a keyless flat file (apps.bea.gov/national/Release/TXT/NipaDataQ.txt, 35.6 MB); the BEA API needs a key; FRED was used for the BEA series (same codes). EIA's MER csv is keyless (GET works; a HEAD request returns 503). Census C30 xlsx files are keyless and carry the data-center line from 2014-01.
* **Harness.** No bug found; not edited. Two limitations worked around in `scripts/extra_stats.py`, validated against the harness (pooled score 2.5e-16, custom predictive 0.0e+00, per-month errors 1.3e-15): (1) `transmission_test` reports DM t only against the full-window base, so for late starters the DM / combined-z p values and `screen()`'s dedicated-asset BH compare training windows, not factors (all 15 raw combined-z BH rejections in this family are such cases); I added same-sample DM statistics. (2) `predictive_test`'s `incr_both_halves` is true for 1992+ series on 11 first-half origins; I require ≥ 60. Also: `endogeneity_check`'s 'check look-ahead' fires for monthly high-tech IP (no interpolation involved): the flag is right to warn, the reason here is late news / revisions (ALFRED). The residual test at horizon 12 marks 302 of 689 correlations significant after BH: 12-month capex and tech-output changes move with the 12-month residual of the market multiple (orders_core vs residual PC1 0.64, ship_tech 0.67), as q-theory predicts (investment follows valuations); as the README says, a same-period 12-month correlation is not causal.
* **p values for the cross-family BH.** `tests.json → pvalues` holds every p value computed (one-sided 'candidate better' for DM / CW / pooled / combined z / predictive; two-sided for the residual test), with test, candidate, asset or target, decade or period. `bh_within_family` gives the within-family BH per test type.

## 9. Verdict per candidate

trans = reference-month transmission (common rule / dedicated assets; same-sample for late starters); rel = release-month variant (common rule; full-history monthly series only); pred = counted predictive passes; sign = expected sign vs the in-sample (FULL) HAC t of the coefficient for the first mechanism asset. Every p value is in `tests.json → pvalues`.

| candidate | coverage | trans | rel | pred | flag | sign: expected / observed | verdict | role |
|---|---|---|---|---|---|---|---|---|
| ip_hitech | 1967-02.., full | –; dedicated: gold, real_estate | common | – | check look-ahead | + / nasdaq +1.36 | adds by the letter as a release-month factor; fails the ALFRED real-time check (T3) | none now |
| ip_semis | 1972-02.., full | –; dedicated: energy, real_estate, utilities | common | – | ok | + / nasdaq +0.75 | adds by the letter as a release-month factor; fails the ALFRED real-time check (T3) | none now |
| ip_computers | 1972-02.., full | – | no | – | ok | + / nasdaq +0.96 | no gain | view variable |
| ip_commeq | 1972-02.., full | –; dedicated: gold, real_estate, utilities | common | – | ok | + / nasdaq +1.13 | dedicated by the letter (not BH-significant) | none |
| ip_compelec | 1972-02.., full | –; dedicated: real_estate, utilities | common | – | ok | + / nasdaq +1.01 | adds by the letter as a release-month factor; fails the ALFRED real-time check (T3) | none now |
| ip_infoproc_eq | 1967-02.., full | –; dedicated: gold, real_estate | no | – | ok | + / nasdaq +1.07 | dedicated by the letter (not BH-significant) | none |
| ip_semis_mat | 1954-02.., full | –; dedicated: industrials, staples, utilities | common | – | ok | + / nasdaq +1.01 | adds by the letter as a release-month factor; fails the ALFRED real-time check (T3) | none now |
| ip_bus_eq | 1947-02.., full | –; dedicated: energy, financials, healthcare, industrials, us_equity | no | – | spanned by core factors | + / industrials -0.23 | dedicated by the letter (not BH-significant) | none |
| ip_mfg_xhitech | 1967-02.., full | – | no | – | spanned by core factors | + / industrials -0.34 | no gain | view variable |
| cu_hitech | 1967-02.., full | –; dedicated: financials, gold, real_estate, utilities | common | – | check look-ahead | + / nasdaq +1.16 | dedicated by the letter (not BH-significant) | none |
| cu_semis | 1972-02.., full | –; dedicated: financials, healthcare, real_estate, utilities | common | – | ok | + / nasdaq +0.70 | dedicated by the letter (not BH-significant) | none |
| cu_compelec | 1972-02.., full | –; dedicated: real_estate, utilities | common | – | ok | + / nasdaq +0.54 | dedicated by the letter (not BH-significant) | none |
| cap_hitech | 1967-02.., full | excluded (smooth) | – | – | – | + / – | not a same-month factor | none |
| ip_electric | 1972-02.., full | –; dedicated: real_estate | no | – | ok | + / utilities -0.44 | dedicated by the letter (not BH-significant) | none |
| ip_elec_td | 1972-02.., full | – | no | – | ok | + / utilities -0.74 | no gain | view variable |
| cu_electric | 1967-02.., full | –; dedicated: utilities | no | – | ok | + / utilities -0.06 | dedicated by the letter (not BH-significant) | none |
| orders_core | 1992-03.., not full | –; dedicated: financials, gold | – | – | ok | + / industrials +0.90 | dedicated by the letter (not BH-significant) | none |
| ship_core | 1992-02.., not full | –; dedicated: financials, staples | – | – | ok | + / industrials +1.34 | dedicated by the letter (not BH-significant) | none |
| backlog_core | 1992-02.., not full | –; dedicated: gold, nasdaq | – | – | ok | + / industrials +2.13 | dedicated by the letter (not BH-significant) | none |
| orders_tech | 1992-03.., not full | –; dedicated: gold, real_estate | – | – | ok | + / nasdaq +1.48 | dedicated by the letter (not BH-significant) | none |
| ship_tech | 1992-02.., not full | –; dedicated: real_estate | – | – | ok | + / nasdaq -0.21 | dedicated by the letter (not BH-significant) | none |
| btb_core | 1992-03.., not full | –; dedicated: gold | – | – | ok | + / industrials +0.21 | dedicated by the letter (not BH-significant) | none |
| orders_aircraft | 1992-03.., not full | common (same sample); dedicated: financials, staples, utilities | – | – | ok | + / industrials -1.56 | passes the common rule by the letter on the same sample (T2/T3 only); gains tiny, no decade significant | none |
| philly_capex | 1968-06.., full | –; dedicated: financials, nasdaq | – | ret_industrials, ret_us_equity | ok | + / industrials +1.42 | **adds as predictor** (returns, sign as expected) | forecast state; view variable |
| ppi_semis | 1981-07.., not full | –; dedicated: financials, industrials, nasdaq, real_estate, us_equity | – | – | ok | + / nasdaq +1.96 | dedicated by the letter (not BH-significant) | none |
| ppi_computers | 1991-01.., not full | common (same sample); dedicated: energy, healthcare, industrials, real_estate, us_equity | – | – | ok | + / nasdaq +0.72 | passes the common rule by the letter on the same sample (T2/T3 only); gains tiny, no decade significant | none |
| ppi_capeq | 1947-05.., full | –; dedicated: gold | no | – | ok | - / industrials +0.49 | dedicated by the letter (not BH-significant) | none |
| ppi_electric | 1973-02.., full | –; dedicated: energy, healthcare, nasdaq, staples, utilities | common | – | ok | + / utilities -1.10 | dedicated by the letter (not BH-significant) | none |
| ppi_transformer | 1967-02.., full | –; dedicated: energy, nasdaq, utilities | no | – | ok | + / industrials -0.38 | dedicated by the letter (not BH-significant) | none |
| ppi_natgas | 1970-02.., full | – | no | – | ok | - / utilities -0.58 | no gain | view variable |
| ppi_coal | 1947-02.., full | – | no | – | ok | - / utilities +0.11 | no gain | view variable |
| util_margin | 1973-02.., full | – | no | – | ok | + / utilities +0.50 | no gain | view variable |
| ppi_steel | 1947-02.., full | –; dedicated: gold | no | – | ok | - / industrials +0.43 | dedicated by the letter (not BH-significant) | none |
| ppi_machinery | 1950-02.., full | common; dedicated: energy, financials, gold, healthcare, real_estate, us_equity | common | ip_yoy, capex_share | ok | + / industrials -0.23 | adds as predictor (ip_yoy, capex_share); slope unstable across subperiods | none now (walk-forward re-check) |
| emp_semis | 1985-02.., not full | –; dedicated: nasdaq | – | – | ok | + / nasdaq +1.01 | dedicated by the letter (not BH-significant) | none |
| emp_compelec | 1990-02.., not full | –; dedicated: gold, nasdaq | – | – | ok | + / nasdaq +1.60 | dedicated by the letter (not BH-significant) | none |
| emp_csd | 1990-02.., not full | –; dedicated: gold, us_equity | – | – | ok | + / nasdaq +1.40 | dedicated by the letter (not BH-significant) | none |
| cons_mfg | 1993-02.., not full | –; dedicated: real_estate | – | – | ok | + / industrials -0.03 | dedicated by the letter (not BH-significant) | none |
| cons_office | 1993-02.., not full | –; dedicated: energy, staples, utilities | – | – | ok | + / real_estate -1.29 | dedicated by the letter (not BH-significant) | none |
| cons_datacenter | 2014-02.., not full | untestable | – | – | ok | + / utilities -2.06 | untestable (history too short) | scenario view variable only |
| cons_compelec | 1993-02.., not full | –; dedicated: gold | – | – | ok | + / industrials +1.37 | dedicated by the letter (not BH-significant) | none |
| cons_electric | 1993-02.., not full | –; dedicated: gold, real_estate | – | – | ok | + / utilities -1.09 | dedicated by the letter (not BH-significant) | none |
| eia_elec_comm | 1976-02.., not full | –; dedicated: gold | – | – | ok | + / utilities +0.57 | dedicated by the letter (not BH-significant) | none |
| eia_elec_ind | 1976-02.., not full | – | – | ip_yoy | ok | + / utilities +0.40 | adds as predictor (ip_yoy) | forecast state candidate |
| eia_elec_total | 1976-02.., not full | – | – | – | ok | + / utilities +0.46 | no gain | view variable over its own months |
| ipe_sw_share | 1947-04.., full | – | – | ret_us_equity | ok | + / nasdaq +2.13 | adds as predictor (ret_us_equity); slope unstable across subperiods | **scenario view variable** (agi_boom / ai_bust) |
| ipe_share | 1947-04.., full | –; dedicated: financials | – | – | ok | + / nasdaq +1.53 | dedicated by the letter (not BH-significant) | none |
| comp_share | 1959-04.., full | – | – | – | ok | + / nasdaq +1.72 | no gain | **scenario view variable** (agi_boom / ai_bust) |
| sw_share | 1959-04.., full | – | – | – | ok | + / nasdaq +2.41 | no gain | view variable |
| rnd_share | 1947-04.., full | – | – | – | ok | + / nasdaq +1.07 | no gain | view variable |
| capex_share_rt | 1947-04.., full | – | – | – | ok | + / industrials +0.76 | no gain | view variable |
| ipe_sw_real | 1947-04.., full | common; dedicated: energy, financials, healthcare, industrials, real_estate, us_equity, utilities | – | – | ok | + / nasdaq +0.70 | passes the common rule by the letter; gains tiny, no decade significant | none |
| relprice_ipe | 1947-04.., full | common; dedicated: energy, nasdaq, real_estate, utilities | – | – | ok | - / nasdaq +1.20 | passes the common rule by the letter; gains tiny, no decade significant | none |
