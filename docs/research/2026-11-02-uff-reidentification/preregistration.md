# Pre-registration: re-identifying u_ff (the price puzzle) — frozen before any candidate is estimated

Written 2026-11-02 (UTC 20:56), before any candidate identification was computed. The baseline numbers quoted here
were measured on today's code only (SP/c1_probe.py, data/results/qmodel.json). The sha256 of this file and the
freeze time are in `preregistration.sha256` next to it. It is never edited after the freeze. A protocol change would
be a new file that names this one.

**Why not the project's machinery.** `tools/run_preregistered.py` / `scenariolab/evaluation/protocol.py` know only the
record ids TA, TC and TS. The argparse `choices`, `RECORD_IDS`, `TITLES` and `RUNNABLE` are fixed, and each id has its
own protocol function and runner. A model-identification record would mean extending the protocol module with a new
record type and runner, or misusing a TC record. So this markdown is frozen by its sha256 and timestamp instead, as
the brief allows. It supersedes nothing: there is no earlier record on the identification.

## 0. Base

- Worktree branch feat/uff-reidentify, from feat/causeway-v2 @ e10b44df67e8c6a95661b7373cf4b91a4abb5e5e (clean).
- data/ is the local copy. panel_quarterly.csv sha256 45353cea…0ace. The stack identity (spec.stack_identity) is
  c64c4f79…8360. The T-F/T-C identity in qmodel.json is 98a0f3d6…269b and matches today's code. The baseline
  qmodel.json is saved as SP/uff/baseline_qmodel.json: 51 of 77 T-F checks pass, the T-C gate fails (d_bdlev −0.057,
  d_tips10 −0.025), and F9 fails on the unemployment peak (q16).
- The baseline puzzle (c1_probe, N 4000) is +0.095 pp core y/y at q5, +0.087 at q8, +0.059 at q12 and +0.031 at q16.
  Unemployment is +0.062 at q8 and +0.095 at q16, still rising.

## 1. What is identified, and what must not move

The structural impact matrix is P = chol(Σ_u) in the do-operator's within-quarter order (FactorModel.do_order /
system). A stated u_ff, and a d_ff pin read as "do", set the rff column's shock. Today's order (r0) is
[E0: pc ps ip lu + the E0 satellites] → [E0p: rff + the E0p satellites] → [E1c/E1/E2: ro ry10 sp gz usd + the other
satellites] → d_bs. The rff column depends only on the set of variables ordered before rff.

The ordering candidates (b1, b2) change only this order. They are implemented as a separate rank used by
`do_order()` alone. They leave unchanged:
- the variables' endogeneity classes, which the satellite regressions use for their same-quarter terms, along with the
  completion shrinkage, F9's E0 list and the validators;
- policy.py's reaction function (the u_ff unit, shock_sd);
- the reduced-form model: core, satellites, Σ_u, pool and draws.

A config_q switch keeps r0 available.

## 2. Candidates, in this order

- **b1 `oil_first`**: real oil (core ro) ranked with E0, ahead of rff. The Fed sees same-quarter oil. ro's satellites
  (d_natgas, d_metals_oil, d_igrea) stay where they are.
- **b2 `oil_expectations_first`**: b1, plus the satellite d_infexp1 (Michigan 1-year, today E2) ranked ahead of rff,
  through the joint Cholesky of [core; satellites]. It is not moved into the core: that would start the core's sample
  at the Michigan series' start (1978Q2) and change the reduced form. Its cost is reported:
  - what the do-operator no longer moves on impact;
  - that the identification leans on a satellite with point-estimate parameters and pre-1978Q2 values masked in its
    covariance EM;
  - any change in estimation (expected none).
- **a `response_restricted`**: order r0. The draw rule becomes today's stability rule AND restriction R_a on the
  draw's linear impulse response to the rff structural shock. R_a is tested on both systems the stability rule
  already tests: the VAR, and the cycle-module system M·Φ with M = I − k e_ip′ on the stacked system, k = Σ_u[:, ip] /
  Σ_u[ip, ip]. Impact is P_d[:, rff], with responses r_h = Φ1 r_{h−1} + Φ2 r_{h−2}. Core y/y's response is
  Δcore_yoy(h) = ¼ Σ_{j=max(1,h−3)}^{h} r_j[d_core]. R_a requires Δcore_yoy(h) ≤ 0 for every h = 8…16 and < 0 for
  every h = 12…16.
  - The satellites are point estimates, so they are fitted before the draw loop. No random numbers are shared, so r0
    is unchanged.
  - Under (a) the proposals per draw are 4,000 (today 400). If any of the 100 draws falls back to the posterior mean,
    (a) fails: the restricted posterior cannot be sampled.
  - Reported: the acceptance rate (accepted among stable proposals, and among all proposals) and how the posterior
    moves (the mean of A's rff-lag columns in the pc / ps / lu equations, steady states, roots).
  - The point model (the posterior mean, used where param_uncertainty=False, e.g. F9) is not changed by (a). Its own
    IRF is reported.

Stop at the first candidate that passes every rule in §3. Report every candidate that ran. If none passes, the
default is not changed; the write-up gives the numbers and recommends the next option.

## 3. Acceptance rules (all must hold)

Measured on the forward stack: rollout.get_stack() defaults (train end 2026Q3, forward anchors, seed 0, n_boot 200,
origin 2026Q4), through QA.make_adapter(), in step mode, with each particle on its own posterior draw. The IRF is
the c1_probe protocol: quarter 1 pins d_ff (do) at each particle's own A0 draw + 0.5 against a placebo pinned at A0;
quarters 2… are free; common random numbers; differences of the particle means.

- **A1 (price puzzle).** Core y/y IRF ≤ 0 at every h = 8…16 and < 0 at every h = 12…16. The judge's literal points
  (h = 8 ≤ 0, h = 12 < 0) are reported too; the decision uses the stricter rule.
- **A2 (unemployment lag).** Mean Δu at h = 1 is 0 (it is E0). The peak of mean Δu over h = 1…40 falls in q4…q12 and
  is > 0.
- **A3 (spiral).** Paired hawk minus passive, as J07:
  - hawk: in particles whose start-of-quarter s_core_yoy > 4.0, d_ff is pinned at that quarter's free A0 draw + 0.5;
  - passive: no pins;
  - rng seed 50_000 + h for both A0 and the pinned draw; N 2000; 80 quarters; root_state(2000).
  - Pass: mean s_core_yoy (hawk − passive) at q20 ≤ 0, and no non-finite value by q80 in any particle of the hawk
    chain's public states or drawn x fields.
  - Reported: core, u and the cumulative extra tightening at q8/16/20/24/40/80, and the firing share.
- **A4 (T-F not worse).** A full factor_checks.run (seed TF_SEED, the build's settings) is compared with the
  baseline's 77 summary lines.
  - The pass count must be ≥ 51.
  - There must be no pass→fail flip on the hard checks: F1, F6_tails at every fit, F10, and F9's same-quarter-E0
    and imputes-nothing sub-results.
  - Every flip, in either direction, is listed.
- **A5 (T-C not worse).** Each E1 field's gate R² must be ≥ baseline − 0.01. The gate status must be no worse
  (baseline false). The selection (w*, regime covariance adopted) must be unchanged. For b1/b2 the T-C numbers are
  expected identical.
- **A6 (F9 literature signs).** At the forward fit: equity_impact > 0, d_y2_impact < 0, usd_impact < 0 for a −1 sd
  u_ff. F9's own unemployment-peak rule (4–8, path mode, cycle off) is reported and counts in A4's tally.
- **A7 (the agent tree's switch).** tree.transmission's ff_irf.puzzle is false on the candidate's adapter (its own
  protocol: 64 particles, seeds 91_000 + h, a q4–8 mean > 0.02 = puzzle). If it stayed true, the judge's overlay
  would stay on and cancel a right-signed response.

## 4. Verification that is not a criterion

- **b1/b2.** simulate() of the forward fit (unconditional, and with reduced-form statements on d_ip, d_cpi, d_oil and
  d_gz) gives the same x fields and levels as r0 at the same seed to 1e-9; T-C and the T-F posterior predictive
  checks are identical. Only quantities that read P may move: shocks, do pins, u_ff statements, F1's u_ff row, F9 and
  rarity with do pins.
- **a.** List what moves.
- **Every candidate.** The IRF tables (core y/y, u, ff; q1–16) and the posterior-mean analytic IRF (VAR and cycle
  systems) are reported.

## 5. Seeds and sizes (fixed)

| Item | Setting |
|---|---|
| Stack | seed 0, n_boot 200 |
| IRF probe | N 4000; A0 rng 1007; the step rng of quarter h (0-based) 700 + h; H 40 |
| Spiral | N 2000; rng 50_000 + h; H 80 |
| T-F / T-C | factor_checks.run(seed=20261102), full |
| Transmission | the tree's constants |
| (a)'s draw rng | as today, SeedSequence([seed, 7]) |
