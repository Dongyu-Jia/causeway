# valuation agent: READY (v2, final for this round, 2026-11-01)

Folder: `scratchpad/sim_redesign/valuation/`

## Files for the qmodel agent

- `valuation_quarterly.csv`: one row per calendar quarter, 1947Q1..2026Q4 (320 rows).
  - The last row, 2026Q4, holds only `__known` values: today's state at origin 2026-10.
  - Key: `quarter`, with `q_start` / `q_end` as `YYYY-MM`.
- `valuation_columns.json`: definition, source, timing, lag, and first/last quarter for each of the 191 columns.
- `valuation_monthly_ref.csv` and `valuation_monthly_known.csv`: monthly versions. The known file runs to 2026-10.
- `valuation_results.json`: harness OOS metrics for every model below (R2, 90% interval, corr, bias, cov80, n, n_indep).
- Code:
  - `valblock.py`: the data layer.
  - `vmodel.py`: the quarterly block as a simharness model.
  - `fredclient.py`: FRED client; the key is never printed, logged URLs read api_key=REDACTED.
  - `work/v*.py`: one script per experiment, with its `.log`, `.csv` and `.json` outputs.

## Timing (same as v1)

| pattern | meaning |
|---|---|
| `x` | value at `q_end` (ref timing) |
| `d_x` | realized change during the quarter. Target only; never an input for that quarter's return. |
| `x__known` | value known at the START of the quarter, publication lags applied |

Lags beyond the month before `q_start`:
- 0: CRSP, French and Nareit prices and dividends; the Shiller price; GS10.
- 1: CPI.
- 3: Shiller E and D.
- French annual book equity: row t is used from July of year t on.
- 5: NIPA and Z.1, release-dated. The Z.1 market value is rolled forward with the CRSP price to the last month-end.

FCF yield columns were added in v2: `fcf_yield_nfc`, `profit_yield_nfc` and `div_yield_z1`, each with `__known` and `d_` versions.
- Construction: nonfinancial corporations, 4-quarter mean of [profits after tax with IVA/CCAdj (W328RC1Q027SBEA) + consumption of fixed capital (B456RC1Q027SBEA) - total capex (Z.1 BOGZ1FA105050005Q)], divided by the market value of NFC equity (NCBEILQ027S).
- Coverage: 1953Q1 onward. FCF is negative in 29 quarters between 1970 and 2001, so use the yield, never log P/FCF.

## Today's state (known at the start of 2026Q4)

| measure | today | comparison |
|---|---|---|
| CAPE | 41.2 | |
| CRSP P/D | 97.4 | |
| trailing P/E | 26.0 | |
| ECY | 0.79% | |
| BusEq P/B | 12.2 | 2000Q1: 15.1 |
| NFC FCF yield | 1.96% | median 2.08% |
| NFC profit yield | 2.71% | 2000Q1: 2.62% |
| real y10 proxy | 2.04% | |

## What the evidence says (details in the final report)

1. **The starting valuation level adds nothing OOS to 1-20 quarter returns under oracle macro, after 1985.**
   - Seven states were tried (CAPE, CAPE vs its 20-year mean, CAPE vs a real-yield/inflation anchor, ECY, P/D, FCF yield, NFC profit yield), as window-level corrections to F.
   - Every one lowers pooled R2 at 12-20Q. us_equity, F vs the best state:
     - F: -0.09 / -0.11.
     - Least bad, F+fcf: -0.14 / -0.26.
     - F+cape: -0.51 / -0.78.
   - The error-correction pull of every anchor also hurts, with kappa clipped at >= 0 as in the engine's KAPPA_BOUNDS (`work/metrics_v21_clip.csv`).
   - P/D block, pooled rolling 12/20Q, against no pull at -0.06 / -0.09:
     - constant anchor: -0.47 / -0.75;
     - real-yield/inflation anchor: -0.25 / -0.27;
     - 20-year rolling anchor: -0.32 / -0.56.
   - The OLS kappa is about 0 after 1973 (full sample 0.0008/q, t 0.07).
2. **The multiple must be a scenario input (maskable) or a band, not a model inference.**
   - Inferred from oracle macro without error correction, CRSP P/D, pooled rolling 1985-2026, at 1/4/12/20Q: R2 0.30 / 0.21 / -0.35 / -0.27; corr 0.58 / 0.53 / 0.15 / -0.05.
   - The 1995-2004 bubble cannot be inferred: corr <= 0.06 at 12-20Q.
3. **The structural block (dp = df + dm) roughly matches F at long horizons for us_equity and loses for the other equities. Estimate betas on monthly data.**
   - It is the dividend equation plus the multiple equation on (real y10, inflation expectations, TP, GZ, CPI y/y), with kappa = 0 and zero re-rating drift.
   - us_equity, pooled 12/20Q: -0.05 / -0.04, against F's -0.09 / -0.11. The 90% intervals overlap.
   - Per-asset own P/D or P/B blocks lose to F: mean10 -0.16 / -0.18 against -0.01 / -0.03.
   - A pure quarterly OLS (Fq) is much worse than the monthly-fitted F: us_equity pooled 4/12/20Q -0.09 / -0.72 / -1.05 against +0.30 / -0.09 / -0.11.
4. **Suggested valuation inputs for s_q:**
   - `sp_lcape__known`, `us_equity_lpd__known`, `sp_ecy__known`, `fcf_yield_nfc__known`.
   - `buseq_lpb__known` (tech) and `{industry}_lpb__known`.
   - Targets / maskable scenario inputs: `d_us_equity_lpd` (or `d_us_equity_lm_cape`) and `d_us_equity_ld12`.
