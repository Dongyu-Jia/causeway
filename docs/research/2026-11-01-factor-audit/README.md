# 缺失因子审计：研究轮存档（2026-11-01）

These are copies of the reports of the missing-factor research round (access date 2026-11-01, research panel sha1
`3869fdde3437`). They hold statistics only; no downloaded data are copied here. The numbers they report are frozen
in `scenariolab/analysis/audit_archive/research_2026_11_01.json`, which the build step `factor_audit` reads; the
diagnostics tab shows every archived number with the tag 「研究轮存档（2026-11-01，面板 3869fdde3437）」 and never presents it as recomputed. The step's
replication mode rebuilds the research round's base model from the panel's audit-only `x_` columns and checks that
the registered candidates reproduce these numbers.

| file | what |
|---|---|
| `verified.md` | the adversarial verification: cross-family multiple testing, refutation checks, verdicts |
| `critique.md` | the completeness review: upper bound, residual anatomy, model versus factor, concept gaps |
| `miss_map.md` | where the base transmission model misses, before the round |
| `concept_gaps.md` | the economic concepts the 13 scenario specs rely on |
| `family-*.md` | the six family reports (rates, risk, labor, housing, commod, tech) |

Local paths in the reports are replaced by `factors/`, `<research>`, `<project>` or `<local path>`. Series from
Moody's, Freddie Mac, the University of Michigan, S&P and Robert J. Shiller are copyrighted by their publishers;
these reports quote statistics computed from them, not the data.
