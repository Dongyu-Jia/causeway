# 预注册记录（pre-registration records）

Frozen records of the quarterly stack's pre-registered tests (SIM_DESIGN §4.2, §4.3, §4.6, §5.8, §14 P5). They are
written by `tools/run_preregistered.py` (the code is `scenariolab/evaluation/protocol.py`). **Never edit a record**: a
run, a status or a note appends to `results`, and nothing else in the file changes.

| File | Test | What it decides |
|---|---|---|
| `<date>_TA.json` | T-A | whether a registered candidate (C1 to C3) joins the asset layer L0. It sets `config_q.TA_DEFAULT` and records the report-only rows: the masked NN, and G_app = L0 + C1 + C2 |
| `<date>_TC.json` | T-C | the factor layer's completion weight w and the regime covariance, on validation; the completion gate, on the test |
| `<date>_TS.json` | T-S | the switch gate. The monthly engine stays the default until T-S passes, and it needs the narrative test (P7). Until P9 evaluates it, the record holds status entries |
| `<date>-<tag>_T?.json` | a later record of the same date | it names the record it supersedes (`frozen.supersedes`: file name and frozen hash). `2026-11-02-r2_TC.json` supersedes `2026-11-02_TC.json`: exact scoring (no Monte Carlo), initial states from published data only, §3.5's saturation |

## A record

- `frozen` holds the protocol, the candidates, thresholds and seeds, the sha256 of every input file (code, model card,
  data), the environment and `frozen_at`. `frozen_sha256` is the sha256 of its canonical JSON (sorted keys, compact).
  The code is the runs' static import closure (`qmodel.spec.code_files` of `evaluation/protocol.py` and the CLI),
  not the whole package, so a change to a file the runs never import (the server, the translator) cannot void a
  record. The first records (2026-11-02) hashed the whole package; their check compares only today's closure.
- `environment.code_commit` says whether the closure was committed at `git_head`. Commit the code before freezing:
  reproducing a record then means checking out that commit, not matching hashes against today's tree.
- `results` holds one entry per run or status: its time, the frozen hash it ran against, the inputs check, the result
  and the result's own sha256.
- Timestamps are compared as UTC instants (a record may be appended to in another time zone).
- A note (`run_preregistered.py note TA --text ...`) adds a disclosure without touching the frozen part or a run.
- A run refuses in any of these cases:
  - the record's frozen part no longer matches its hash;
  - an input file differs from its frozen hash;
  - the record already holds a run. Each test runs exactly once.
- A protocol change is a new dated record that names the one it supersedes.

## Commands

```
python tools/run_preregistered.py check [--inputs]          # integrity of every record (and today's inputs)
python tools/run_preregistered.py frozen-copy <rec> <out>   # the record as it was frozen (results empty)
python tools/run_preregistered.py freeze --date YYYY-MM-DD  # new records (refuses to overwrite)
python tools/run_preregistered.py freeze --only TC --tag r2 --supersedes 2026-11-02_TC.json   # one later record
python tools/run_preregistered.py run TA|TC --date ... [--tag r2]   # once
python tools/run_preregistered.py status TS --date ...
python tools/run_preregistered.py note TA --date ... --text "..." [--json fields.json]
python -m scenariolab.build --only evaluation               # integrity, inputs, reproduction (opt-in)
```

## Landing (commit order is checked in CI)

`tests/test_preregistered.py` requires the first committed version of each record to hold no result. Every later
version must keep the frozen part and only extend `results`. The check needs the full history: CI checks out with
`fetch-depth: 0`, a shallow clone fails with "needs the full history", and the record commits must never be
squash-merged. To land the records:

1. Commit the frozen copies first: `tools/run_preregistered.py frozen-copy <record> <record>.frozen`, then put that
   text in place of the record. They are byte-identical to the files `freeze` wrote; the scratch folder of P5 keeps
   the originals with their sha256.
2. Then commit the records with their results.

The derived rows of a run (the conditional paths of T-A) are in `data/results/preregistered/`, which is git-ignored.
Their sha256 is in the result.
