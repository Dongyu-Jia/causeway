# qmodel: the quarterly scenario model (plug-in documentation)

This is one quarterly model that takes three inputs:

- s_q: the hard state known at the start of the quarter;
- c_q: the pre-quarter context, translated to states;
- x_q: the in-quarter events, translated to the schema's x fields, any subset of which may be missing.

It returns the joint distribution of the quarter's 13 asset returns plus the us_equity valuation change. It also fills in every x field the text did not state. Applied quarter by quarter, it rolls a scenario forward for up to 10 years and more.

All code here is read-only with respect to MAIN, RWT and FWT. Run it as

    cd "<MAIN>" && "<MAIN>/.venv/bin/python" -B <script>

## Calling the plug-in

```python
import sys; sys.path.insert(0, "<scratch>/sim_redesign/qmodel")
import qplugin as qp, pandas as pd
x = pd.DataFrame({"d_ff": [0.5, 0.75, 0.75, 0.5],          # one row per quarter from the origin; NaN = not stated
                  "d_oil": [25, -5, None, None], "x_oil_cause": ["supply", "supply", None, None],
                  "d_gz": [0.4, None, None, None]})
out = qp.predict("2022-01", x, n_paths=1000, seed=0)        # model="G" (default) | "L2" | "NN"
out["paths"]      # (n_paths, 4, 14) quarterly log returns: 13 assets + "mult"
out["cum_q50"]    # medians of the cumulative returns; also q10/q90, mean, x_mean / x_q10 / x_q90
```

**`predict(origin_month, x_table, n_paths=1000, seed=0, model="G")`**

- `origin_month`: a quarter-start month, "YYYY-MM".
  - The model is fitted only on quarters that end before the origin quarter: train_end = the origin quarter − 1.
  - The origin states are those known at the start of the origin quarter.
  - The latest origin is 2026-10 (2026Q4 = today).
- `x_table`: one row per simulated quarter.
  - Columns: any of the 52 schema fields (units as in `qdata.build` and `schema.json`).
  - `x_oil_cause` takes "supply" or "demand"; together with `d_oil` it sets the supply part `d_oil_sup`.
  - Policy interventions `u_ff`, `u_bs`, `u_def`, `u_tr`, `u_ptax` follow the do-operator of schema section 3.0. When a shock u is given and its raw field is not, the raw field is set exactly to the systematic part plus u. The systematic part is the reaction function or stabiliser evaluated on the completed same-quarter macro and the states.
  - Stated output legs: `y_mult` (alias "mult") is the valuation leg Δlog P/Ê; `y_eps` (alias "eps") is the earnings leg; any `y_<asset>` is also accepted. Every path is shifted so the stated leg holds exactly, with us_equity = y_eps + y_mult (V-PRICE). The other assets move by their residual covariance with the stated leg.
  - NaN means not stated: the input is masked and inferred.
- Returned paths are jointly simulated.
  - Masked x fields are drawn from their conditional Gaussian, so model "G" carries their uncertainty.
  - Residual vectors are bootstrapped, which keeps cross-asset correlation, and scaled by a fitted state-volatility factor.
  - Returns are price returns, except cash, which is the bill total return.
- States roll forward path by path:
  - macro states from the completed x path: fed funds, unemployment, term and credit spreads, core y/y and the core > 4% regime, the zero-bound flag, inflation trend, real 10y, Sahm gap, real oil, real dollar;
  - asset-derived states from the simulated returns: CRSP P/D, CAPE, E/E10, FCF yield, volatility, momentum;
  - the previous quarter's x from the path.
  - Nothing realized in or after the origin quarter is read.
- Determinism: the same seed gives identical paths.

**`harness_model_fn(model="G", n_sim=300)`** returns a `model_fn` for `simharness.conditional_paths`. It runs the model under oracle macro (the realized x of the window's quarters) and places each quarter's return in the quarter's last month. It reproduces this folder's own 1Q scores exactly (G split: 0.152 / 0.226 / 0.492; `work/q12_*`).

**`realized_x(start_q, H)`** returns the realized x table, the "perfect translation", for tests.

## The model ("G", the structured Gaussian)

1. **x-block, x | s.** Each of 57 inputs (the 52 schema fields, with x_oil_cause carried by d_oil_sup, plus 5 policy shocks) is a ridge regression on its own lag and 11 start states. The regression uses blocked 5-fold CV and runs on training quarters only.
   - The residual covariance is estimated by missing-data EM, so late-starting series are masked before their start rather than invented.
   - The covariance structure is chosen by masked-entry prediction error on the last quarter of the training residuals. It is either factor-analytic with k ∈ {3, 5, 8} or shrinkage toward the diagonal. FA(3) or shrinkage 0.5 was chosen in every split.
2. **r-equations, r | x, s.** There is one ridge equation per output (13 returns + y_mult).
   - Inputs: the schema's tier-1 fields, with tier-1s fields only in their scope assets; the interactions TP × 1{core > 4}, Δy10 × P/D and oil × ZLB; the states s_vol, s_zlb, s_core4; own momentum.
   - The F-field coefficients are shrunk toward the harness's monthly-fitted F betas (L0) rather than toward 0. The penalty and a separate state-penalty multiplier are chosen by blocked CV.
3. **Conditioning.** Masked x is filled in by exact Gaussian conditioning. r is linear in x given s, so its mean is the r-equation at the completed x, and its variance adds the masked-x uncertainty through the Jacobian.
4. **Saturation.** Standardized inputs saturate at ±4 training sd. This was fixed a priori; sensitivity is in `work/q07_clip.log`. The exceptions are mechanical own-price inputs: d_oil for crude oil and d_y10 for the 10y bond. Without saturation, 2008Q4 QE (d_bs ~ 77 training sd) and the 2020 macro moves (d_deficit ~ 33 sd) produce absurd linear extrapolations.
5. **No single-episode flags.** There is no pandemic dummy.

## Timing of every input

| input | timing |
|---|---|
| x fields | same quarter (realized = the "perfect translation"; in a scenario, what the in-quarter text states) |
| d_ff, d_y10, d_tp10, d_gz, d_spread, d_usd, d_oil | end of quarter vs end of the previous quarter |
| flows (GDP, consumption, CPI) | quarterly average or quarterly annualized |
| s_ff, s_term, s_spread, s_oil_real, CRSP P/D | the month before the quarter (lag 0) |
| core and CPI y/y, unemployment, Sahm gap, real 10y, real dollar | lag 1 month |
| CAPE, E/E10 | Shiller E, lag 3 months |
| NFC FCF yield | NIPA/Z.1, release dated |
| momentum, volatility | quarterly asset returns through the previous quarter |
| previous-quarter x | part of the realized factor path; a few NIPA fields are released only weeks into the quarter (oracle-macro assumption) |

Late series are masked before their start: d_y2 (1976Q3), d_infexp1 (1978Q2), d_hpi (1975Q2), x_lending_std (1990Q2), d_infexp_lr (1990Q3), d_epu_health and d_chargeoff (1985Q2), d_tips10 (2003Q2), u_bs (2008Q3).

## Files

| file | what |
|---|---|
| `qdata.py` | builds the quarterly panel `work/qpanel.csv` and `work/qpanel_meta.json` (216 quarters 1973Q1..2026Q4; per-column source, timing, lag) |
| `qcore.py` | models: ridge L1/L2 (+ monthly-F prior), policy split, x-block EM, structured Gaussian G, joint Gaussian G-full, masked MLP NN, NN-res |
| `qeval.py` | protocols (split, rolling), 1Q tasks (full / half / O-A / O-A′), recursive 4/12/20-quarter paths, CRPS, coverage, block bootstrap |
| `q05_run.py`, `q06_report.py` | the evaluation run and the report (work/q05_*, work/q06_*) |
| `qplugin.py` | the plug-in: `predict`, `realized_x`, `harness_model_fn` |
| `work/q09_*`–`q12_*` | plug-in tests (2022–23, 10 years from today, do-operator, valuation leg), the 2020–21 check, the harness cross-check |

## Measured skill (walk-forward; every fit sees quarters up to its train end only)

Mean over the 10 score assets of the quarterly OOS R² against the training-mean drift. The split protocol trains to 1994Q4, 2004Q4 and 2014Q4; the number of test quarters is 40 / 40 / 47. Each cell gives R² | CRPSS | 80% coverage. The full tables are in `work/q06_metrics.csv` and the paired differences with 90% block-bootstrap intervals in `work/q06_diffs.csv`.

| task | model | 1995–2004 | 2005–2014 | 2015–2026 |
|---|---|---|---|---|
| 1Q, x fully given | L0 (harness monthly F) | +0.126 \| +0.053 \| 0.77 | +0.261 \| +0.163 \| 0.81 | +0.441 \| +0.250 \| 0.86 |
| | **G** (= L2 at full x) | +0.152 \| +0.073 \| 0.78 | +0.226 \| +0.160 \| 0.79 | +0.492 \| +0.273 \| 0.86 |
| | NN | −0.062 \| −0.036 \| 0.74 | +0.004 \| −0.014 \| 0.82 | +0.185 \| +0.081 \| 0.84 |
| | NN-res (G + MLP correction) | +0.148 \| +0.070 \| 0.78 | +0.258 \| +0.152 \| 0.78 | +0.476 \| +0.261 \| 0.85 |
| 1Q, half of x masked at random | G | +0.047 \| +0.027 \| 0.79 | +0.091 \| +0.096 \| 0.80 | +0.277 \| +0.156 \| 0.83 |
| | L2 (masked x → training mean) | +0.046 \| +0.028 \| 0.76 | +0.115 \| +0.086 \| 0.76 | +0.276 \| +0.157 \| 0.81 |
| 1Q, O-A (E0 + E0p only) | G | −0.038 \| −0.011 \| 0.80 | −0.291 \| −0.072 \| 0.75 | −0.035 \| −0.013 \| 0.78 |
| | L2 | −0.000 \| −0.012 \| 0.73 | −0.110 \| −0.033 \| 0.70 | −0.005 \| +0.012 \| 0.75 |
| 4Q recursive path (n_indep 10 / 10 / 11) | G | −0.047 | +0.250 | +0.365 |
| | L0 | −0.031 | +0.302 | +0.347 |
| 12Q (n_indep 3 / 3 / 3) | G | −0.476 | +0.251 | +0.149 |
| | L0 | −0.336 | +0.215 | +0.049 |
| 20Q (n_indep 2 / 2 / 2) | G | −0.574 | +0.293 | −0.052 |
| | L0 | −0.431 | +0.201 | −0.363 |

Notes on the table:
- The NN loses to G and L2 in every split and every task. At 1Q full x the gap is −0.21 to −0.31, and the 90% intervals exclude 0.
- NN-res is within ±0.03 of G, and no interval excludes 0.
- Seeds: 4 per fit. The sd across seeds of the NN's mean10 R² is 0.011–0.014; for NN-res it is 0.002–0.018.

**Rolling protocol.** The model is refitted every January on data to the previous December, and origins run 1985Q1..2026Q3. Pooled over 1985–2026, mean10 R² [90% CI]:

| task | L0 | G | NN | NN-res | n_indep |
|---|---|---|---|---|---|
| 1Q full x | +0.226 [+0.12,+0.32] | +0.222 [+0.14,+0.31] | +0.084 [+0.03,+0.14] | +0.231 [+0.13,+0.32] | 167 |
| 1Q half masked | — | +0.119 [+0.04,+0.20] | +0.017 | +0.121 | 167 |
| 1Q O-A | — | −0.031 | −0.070 | −0.045 | 167 |
| 4Q path | +0.223 | +0.169 | −0.005 | +0.177 | 41 |
| 12Q path | −0.007 | −0.127 | −0.556 | −0.165 | 13 |
| 20Q path | −0.030 | −0.197 | −1.112 | −0.432 | 8 |

G − L0 at 12Q is −0.119 [−0.197, −0.052] and at 20Q −0.167 [−0.255, −0.088]. The deficit comes from 1985–2004, when the quarterly model had only 47–127 training quarters. In 2015–26 G is ahead: +0.086 at 12Q and +0.330 at 20Q.

**2020–21, generic channels only (no pandemic flag).** Refits at 2019Q4 and 2020Q4, 1Q, oracle x. Pooled over the 8 quarters × 10 assets:

| model | R² | 80% coverage |
|---|---|---|
| G | 0.638 | 0.86 |
| L0 | 0.636 | 0.76 |
| NN | 0.274 | 0.60 |

- us_equity, 2020Q1–Q4: G −0.21 / +0.15 / +0.09 / +0.11 against an actual −0.23 / +0.20 / +0.09 / +0.14.
- Recursive 8-quarter path from the 2019Q4 fit (`work/q11_covid.log`): us_equity +0.28 [+0.02, +0.50] against an actual +0.40. 9 of 10 assets fall inside the 80% band. The misses are energy (+0.30 against −0.01) and the valuation leg (−0.27 against +0.20).
- With only macro and policy given (O-A), the sequence is wrong: 2020Q1 −0.07 and Q2 −0.07. The completion puts the credit spread up in Q2 (+0.70 against an actual −1.49) and the dollar down in Q1 (−0.97 against +6.63). Adding d_gz, d_y10 and d_oil restores the path (−0.18 / +0.15).

## Limitations

- About 214 quarters in all, and 87, 127 and 167 training quarters at the three split ends. Every multi-year statistic rests on 2–10 independent windows.
- The x-block covariance does not depend on the regime. At the zero bound it still infers rate cuts from bad macro (2008–09, 2020), and it cannot represent the flip in the stock-bond correlation beyond the TP × core > 4 term.
- Inferring market prices (E1) from macro alone (E0) has negative out-of-sample value at the quarterly frequency. A narrative must state the financial conditions it means.
- The valuation leg cannot be inferred: the 1995–2000 multiple expansion is not in the macro. State it as y_mult (and y_eps). In the evaluation these legs are always masked.
