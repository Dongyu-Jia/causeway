"""The `narrative` build step (SIM_DESIGN §3.9): the mock backend and the stand-in scoring only. It never calls a
real LLM, whatever MSL_LLM_BACKEND says: the step sets the mock backend for its own duration.

    python -m scenariolab.evaluation.narrative build

It seals the registry when it is missing (data/evaluation/), then runs the narrative test on every episode with v1
narratives (the research round's stand-in tables replace the translations) and the synthetic episodes (their
reference translations), scores them on the quarterly stack and writes the run's report under
data/results/narrative/<run_id>/. It also checks the research round's archive against its published numbers. About a
few minutes (18 stack fits). run(ctx) has scenariolab.build's step signature; adding "narrative" to build.STEPS and
OPT_IN is left to the merge, because tests/test_build.py pins the shared step list on every branch.
"""
from __future__ import annotations

import os
from contextlib import contextmanager


@contextmanager
def mock_backend():
    old = os.environ.get("MSL_LLM_BACKEND")
    os.environ["MSL_LLM_BACKEND"] = "mock"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MSL_LLM_BACKEND", None)
        else:
            os.environ["MSL_LLM_BACKEND"] = old


def run(ctx=None, *, episodes: str = "all", engines=("Q",), n_paths: int | None = None) -> dict:
    from . import episodes as EP
    from . import jobs as JB
    from . import key as K
    from . import research as RS
    from . import runner as RN
    out = {}
    with mock_backend():
        if not EP.sealed():
            out["seal"] = K.seal()
        est = RN.estimate({"episodes": K.resolve(episodes), "narratives": "v1",
                           "arms": ["N1", "N2", "N3", "NW", "NS"], "probe": True, "engines": list(engines)})
        rid = RN.create_run(est, label="构建步骤：模拟后端 + 研究轮代译（流程检验）")
        out["run_id"] = rid
        out["status"] = JB.start(rid, background=False,
                                 score_kwargs={"n_paths": n_paths} if n_paths else None)
    if RS.available():
        chk = RS.checks(RS.recompute(intervals=False))
        out["research_checks"] = {k: chk[k] for k in ("ok", "n", "max_dev")}
    print(f"[narrative] run {out['run_id']}: {out['status']}"
          + (f"; research checks {out['research_checks']}" if "research_checks" in out else ""), flush=True)
    return out
