"""The end-to-end narrative test, a permanent feature (SIM_DESIGN §8; §14 P7).

A historical episode is described in words (a narrative with no years, names or places); the LLM translates the
words into a QTable under the blind protocol; the quarterly stack (and the monthly engine, on the identical
translation) simulates the episode from its origin with a model fitted only on data before it; the result is scored
against what happened, next to oracle arms that are given the realized factor paths.

    common.py     paths, arms, horizons, asset groups, labels (no episode window anywhere)
    episodes.py   the public registry: shuffled codes, sets, narratives, stand-ins, N-wrong pairs, blind state rows
    key.py        THE SEALED KEY: code -> window. Read only by the scorer, the audit, the synthetic writer and the
                  seal command. The prompt builder and the runner never import it (a test checks the imports).
    prompts.py    the blind prompt builder (the translator's test mode + the registry's percentile bands; the probe)
    runner.py     estimates and the translation phase: every byte sent recorded, N3 first, pause and resume
    score.py      the simulation and scoring phase: arms × engines × CMA arms, cells, pooled statistics, claims
    audit.py      blindness: prompt scans, recognition rate, contamination index, recall flags
    report.py     the report the evaluation tab renders (report.json; format in its docstring)
    research.py   import-standin: the research round's 36 stand-in tables and its stored scores, recomputed
    synth.py      the synthetic episodes (narratives written by code from the model's own simulated paths)
    sep.py        the H-SEP arm: the Fed's real-time SEP medians as a QTable by code (download needs approval)
    jobs.py       the background job (translate -> score -> report) behind the API
    api.py        /api/evaluation/narrative/* (estimate, run with a fresh estimate id and confirm: true, runs)

    python -m scenariolab.evaluation.narrative estimate | run | score | report | import-standin | seal | sep

Real LLM calls happen only from an explicit, confirmed run (an estimate first; the API needs confirm: true and an
estimate issued in the last 10 minutes; the CLI asks on a TTY). Every test, CI job and build step runs on the mock
backend; nothing here downloads anything unless the user approves the SEP fetch.
"""
