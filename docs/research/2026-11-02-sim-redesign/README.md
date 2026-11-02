# 季度情景模拟：研究轮与设计存档（2026-11-01 至 2026-11-02）

The research round of 2026-11-01 asked how the app should simulate "scenario → factors → assets" quarter by quarter,
with valuation as a state, an LLM that translates text into the model's own inputs, and a permanent end-to-end
narrative test. Ten research roles reported, a critic re-ran their key numbers, and a designer wrote the
implementation spec, `SIM_DESIGN.md`, on 2026-11-02. The quarterly stack in `scenariolab/qmodel/` and
`scenariolab/evaluation/` implements that spec (docs/ARCHITECTURE.md, chapter 季度模型).

These are copies. They hold statistics and text, no downloaded data. Every number the app shows is recomputed by the
app's own code from its own panel, and the spec's numbers serve as check values (for example
`scenariolab/evaluation/research/oracle_ablation.json`, the narrative test's research archive in
`scenariolab/evaluation/narrative/research/`, and the model card `scenariolab/qmodel/card.json`). This folder is
frozen: a later round adds its own folder instead of editing this one.

## What is here

| File | What |
|---|---|
| [`SIM_DESIGN.md`](SIM_DESIGN.md) | the implementation spec: decisions D1–D15, the evidence after the critique (§2), the architecture (§3), asset layer (§4), factor layer (§5), valuation (§6), scenario layer and translator (§7), the narrative test (§8), the forecast tab (§9), UI (§10), interfaces (§11), landing order (§12), the user's decisions U-1–U-16 (§13), phases P0–P10 (§14), open questions (§15), the model card (App. A), QTable/1 (App. B) |
| [`critic/critique.md`](critic/critique.md) | the adversarial review: a verdict on every claim of the reports, with the critic's own re-runs (c01–c25) |
| [`reports/harness.md`](reports/harness.md) | the oracle-macro evaluation harness (ported into the app as `scenariolab/evaluation/harness.py`) |
| [`reports/layers.md`](reports/layers.md) | timing classes (same-quarter against leading), anticipation evidence, the endogeneity table, the forecast-tab proposal |
| [`reports/writer.md`](reports/writer.md) | the 12 research episodes and their narratives n1, n2, n3 (the code-to-window key is not part of the report) |
| [`reports/valuation.md`](reports/valuation.md) | the valuation block: series with timing, the identity, the negative results on pulls and level states |
| [`reports/joint.md`](reports/joint.md) | the joint quarterly factor model: BVAR, Waggoner–Zha conditioning, the C′ cycle hybrid, persistence |
| [`reports/schema.md`](reports/schema.md) | the quarterly input table after two critique and revision rounds (schema-revise-2 3.0) |
| [`reports/schema_scen.md`](reports/schema_scen.md) | the scenario-driven review: 127 scenarios decomposed into generic channels |
| [`reports/qmodel.md`](reports/qmodel.md) | the quarterly three-input model G, the masked NN, masking and completion |
| [`reports/translator.md`](reports/translator.md) | the 36 stand-in translations (specs and quarterly tables) and the period guesses |
| [`reports/scorer.md`](reports/scorer.md) | the research version of the end-to-end narrative test: scoreboard, decomposition, audit |
| [`schema/SCHEMA.md`](schema/SCHEMA.md) | the schema roles' document for the user (Chinese): the table, the policy split, templates G01–G35, the rules, and the second critic's 12 gaps with their decisions (§0.1) |
| [`schema_lit/SCHEMA_lit.md`](schema_lit/SCHEMA_lit.md) | the literature-driven half of the schema work (Chinese) |
| [`schema_scen/scenarios.md`](schema_scen/scenarios.md), [`schema_scen/gaps.md`](schema_scen/gaps.md) | the scenario-driven half: the scenario list with each one's essence, and the gaps it found (Chinese) |
| [`qmodel/README.md`](qmodel/README.md) | the research qmodel's plug-in documentation |
| [`valuation/READY.md`](valuation/READY.md) | the valuation role's hand-over note to the qmodel role (timing conventions, today's states) |
| [`narratives/README.md`](narratives/README.md) | the narrative-test inputs: file roles and the T0 / Y0 notation of the anchored prompts (Chinese) |
| [`design/d01_history_stats.py`](design/d01_history_stats.py) … [`design/d05_sizes_anchors.py`](design/d05_sizes_anchors.py) | the designer's five scripts (the `[d0x]` tags in the spec) |
| [`design/work/`](design/work/) | their outputs: [`d01_history_stats.json`](design/work/d01_history_stats.json), [`d02_anchors_outliers_episodes.json`](design/work/d02_anchors_outliers_episodes.json), [`d05_sizes_anchors.json`](design/work/d05_sizes_anchors.json), the model card [`model_card.md`](design/work/model_card.md) / [`model_card.csv`](design/work/model_card.csv) (Appendix A, from d03) and [`qtable.schema.json`](design/work/qtable.schema.json) (Appendix B's JSON schema, from d04) |

[`d03_model_card.py`](design/d03_model_card.py) generates Appendix A, [`d04_qtable_schema.py`](design/d04_qtable_schema.py)
Appendix B's schema and [`d02_anchors_outliers_episodes.py`](design/d02_anchors_outliers_episodes.py) the narrative
test's episode table (§8.2). [`d01_history_stats.py`](design/d01_history_stats.py) gives the history statistics the
spec uses as acceptance thresholds, and [`d05_sizes_anchors.py`](design/d05_sizes_anchors.py) the valuation size bins,
the CAPE-multiple drift by train end and the 80-quarter unemployment anchor. They read the research round's quarterly panels (`qpanel.csv`, `valuation_quarterly.csv`) and `schema.json`,
which are not copied: the app rebuilds the panel as `data/processed/panel_quarterly.csv`
(`python -m scenariolab.qmodel.panel`), the schema is in the package as `scenariolab/qmodel/schema.json`, and
`tools/build_model_card.py` regenerates the card with d03's rules.

## Where the reports come from

- The ten role reports (`reports/`) are the roles' final reports as the research workflow stored them (its journal,
  run `wf_cc7fe8c3-afc`): summary, key findings, open questions and the files each role listed. These are the reports
  the critic and the designer read. They are rendered as Markdown; the text is not edited apart from the paths below.
- The other documents are copied byte for byte from the research folder `scratchpad/sim_redesign/`, and this folder
  keeps the same relative paths. That is why the spec's own references (`critic/critique.md`, `design/d03_model_card.py`,
  `schema/SCHEMA.md`, `qmodel/README.md`, …) resolve here for the files that were copied.
- Local absolute paths are written the way the spec writes them: `scratchpad/sim_redesign/…` for the research folder,
  `scratchpad/…` for the rest of the session scratchpad. Only `valuation/READY.md` among the copied files had one
  (its first line). Other names in the texts: MAIN is the app's main checkout as it was then (`main` at `eb2f448`),
  RWT the regime-layer worktree (`feat/causeway-v2`), FWT the factor-round worktree.
- Every file was scanned before it was added: no local path, no user name, no e-mail address, no key-shaped string
  (FRED requests are logged as `api_key=REDACTED` in the texts that mention them).

## What is not here, and why

- **Data.** The research folder's series files and downloads (`valuation/valuation_quarterly.csv` and its monthly
  versions, `*/dl/`, `*/raw/`, `*/downloads/`, the harness's and the critic's stored paths) are third-party data or
  derived from it, under their publishers' terms; they stay local. So do the work tables the reports summarize
  (`schema_scen/coverage*.csv`, `schema/coverage_final.csv`, the scorer's `out/`). Series from Moody's, Freddie Mac,
  the University of Michigan, S&P and Robert J. Shiller are copyrighted by their publishers; these reports quote
  statistics computed from them, not the data.
- **The narratives and the stand-in tables** (`narratives/E01–E12/`). They are already in the package, converted and
  checked: `scenariolab/evaluation/narrative/research/episodes_v1.json` (the v1 narratives, the 36 stand-in tables as
  QTable/1 legacy and the translator's period guesses), with the scorer's rows and the published check values beside
  it (`python -m scenariolab.evaluation.narrative import-standin --from <research folder> --check` rebuilds them from
  the research folder and compares).
- **The schema critics' gap lists and the earlier table versions** (`schema-critic-1`, `schema-critic-2`,
  `schema-merge`, `schema-revise-1`). They were intermediate inputs: the second critic's gaps and their decisions are
  in `schema/SCHEMA.md` §0.1, and the first round's were applied by schema-revise-1, whose table SCHEMA.md
  supersedes (the research folder keeps the earlier versions under `schema-revise-*/backup_prev/`).
- **The scripts of the research roles** (harness, layers, joint, qmodel, translator, scorer, critic). They ran against
  the research scratchpad and MAIN; the reports name them, and the app's ports replace them
  (`scenariolab/evaluation/harness.py` for `simharness.py`, `scenariolab/qmodel/` for `qdata.py`, `qcore.py` and the
  joint study's `jq*.py`, `scenariolab/evaluation/narrative/` for the scorer). The designer's five scripts are kept
  because the spec cites their outputs as `[d0x]`.
- **`schema/schema.json`.** The app carries its projection, `scenariolab/qmodel/schema.json` (P0.5: the
  `x_oil_cause` enum and the redefined `y_mult`). The production translator takes its templates' signs and sd sizes
  from it; the narrative test's blind translator never sees it (its dated footprints would date an episode) and gets
  the blind view `scenariolab/qmodel/schema_view.json` instead.
