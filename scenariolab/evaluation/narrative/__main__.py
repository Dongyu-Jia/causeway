"""python -m scenariolab.evaluation.narrative <command> (SIM_DESIGN §8.5).

    estimate [--backend claude_code|api|mock] [--episodes clean|all|short|E05,N13|X03,X07] [--narratives v1|v2]
             [--arms N1,N2,N3,wrong,synth] [--probe | --no-probe] [--engines Q,E (default)] [--refresh]
                                     calls, tokens and minutes; writes an estimate id (no model call)
    run --estimate <id> [--i-understand-this-uses-my-plan] [--i-understand-this-is-billed]
                                     asks 「确认运行？」 on a TTY; otherwise a real backend needs the flag
                                     (claude_code: the plan; api is refused: QTables never go through the API).
                                     Whether a call is uncached is decided now, not by the estimate's old count
    resume <run_id> [--estimate <id>] [flags as for run]
                                     prices the run's OWN remaining calls now (an estimate of another plan is
                                     refused) and asks, or needs the flag, whenever one of them is not cached
    score <run_id> [--engines Q,E] [--paths N]      re-score a translated run
    report <run_id>                                  stats.json from the stored rows, then report.json
    list                                             runs (research, mock, real)
    import-standin --from <sim_redesign folder> [--check]
    research                                         recompute the research round's statistics and check them
    seal [--force] [--no-synth]                      write the sealed key and the registry (local files only)
    sep plan | sep fetch --i-approve-download        the H-SEP data (the download needs your approval)
    build [--episodes all] [--engines Q] [--paths N] the build step: mock backend + stand-ins only (build.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import common as C


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def _confirm(est: dict, args) -> bool:
    """est: an estimate priced NOW (runner.live_cost): real_calls is whether any call it would make is uncached."""
    backend = est["backend"]
    if backend == "mock" or not est.get("real_calls"):
        return True
    if backend == "claude_code" and getattr(args, "i_understand_this_uses_my_plan", False):
        return True
    if not sys.stdin.isatty():
        print("需要确认：加 --i-understand-this-uses-my-plan（会计入你的 Claude 订阅额度）", file=sys.stderr)
        return False
    return input(f"这次会调用 {est['calls']['new']} 次本机 Claude Code（计入订阅额度）。确认运行？[y/N] ").strip().lower() \
        in ("y", "yes")


def _estimate_summary(est: dict) -> str:
    c = est["calls"]
    return (f"[估算 {est['id']}] 后端 {est['backend_label']}；翻译 {c['translations']} 个、探针 {c['probes']} 个；"
            f"新调用 {c['new']} 次（缓存 {c['cached']}、代译 {c['standin']}）；输入约 {est['input_tokens']} tokens，"
            f"输出约 {est['output_tokens'][0]}–{est['output_tokens'][1]} tokens；约 {est['minutes'][0]}–{est['minutes'][1]} "
            f"分钟；缺 {est['n_missing']} 个叙事。{C.ESTIMATE_FRESH_S / 60:.0f} 分钟内有效。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scenariolab.evaluation.narrative", description="叙事检验（SIM_DESIGN §8）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--backend")
    e.add_argument("--episodes", default="all")
    e.add_argument("--narratives", default="v2", choices=["v1", "v2"])
    e.add_argument("--arms", default="N1,N2,N3,wrong,synth")
    e.add_argument("--probe", dest="probe", action="store_true", default=True)
    e.add_argument("--no-probe", dest="probe", action="store_false")
    e.add_argument("--engines", default="Q,E")
    e.add_argument("--refresh", action="store_true")
    for name in ("run", "resume"):
        r = sub.add_parser(name)
        if name == "resume":
            r.add_argument("run_id")
        r.add_argument("--estimate", required=name == "run")
        r.add_argument("--i-understand-this-uses-my-plan", action="store_true")
        r.add_argument("--i-understand-this-is-billed", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("run_id")
    s.add_argument("--engines")
    s.add_argument("--paths", type=int)
    rp = sub.add_parser("report")
    rp.add_argument("run_id")
    sub.add_parser("list")
    imp = sub.add_parser("import-standin")
    imp.add_argument("--from", dest="src", required=True, help="研究轮的 sim_redesign 文件夹")
    imp.add_argument("--check", action="store_true")
    sub.add_parser("research")
    se = sub.add_parser("seal")
    se.add_argument("--force", action="store_true")
    se.add_argument("--no-synth", action="store_true")
    sp = sub.add_parser("sep")
    sp.add_argument("action", choices=["plan", "fetch", "status"])
    sp.add_argument("--i-approve-download", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--episodes", default="all")
    b.add_argument("--engines", default="Q")
    b.add_argument("--paths", type=int)
    a = ap.parse_args(argv)

    if a.cmd == "build":
        from . import build as BD
        res = BD.run(None, episodes=a.episodes, engines=tuple(a.engines.split(",")), n_paths=a.paths)
        _print(res)
        return 0 if res.get("status") == "done" else 1

    if a.cmd == "import-standin":
        from . import research as RS
        res = RS.import_standin(Path(a.src), check=a.check)
        _print(res)
        return 0 if res["ok"] else 1
    if a.cmd == "research":
        from . import research as RS
        chk = RS.checks()
        _print({k: chk[k] for k in ("ok", "n", "max_dev", "max_interval_dev")} | {"failed": chk["failed"][:20]})
        return 0 if chk["ok"] else 1
    if a.cmd == "seal":
        from . import key as K
        _print(K.seal(force=a.force, synth=not a.no_synth))
        return 0
    if a.cmd == "sep":
        from . import sep as SEP
        if a.action == "plan":
            _print(SEP.plan())
            return 0
        if a.action == "status":
            _print(SEP.status())
            return 0
        try:
            _print(SEP.fetch(approved=a.i_approve_download))
        except SEP.SepError as err:
            print(f"[SEP] {err}", file=sys.stderr)
            return 2
        return 0
    if a.cmd == "list":
        from . import report as RP
        for r in RP.runs():
            print(f"{r['id']:28s} {r.get('status') or '':12s} {r.get('kind') or '':9s} {r.get('backend') or ''}  "
                  f"{r.get('label') or ''}")
        return 0
    from . import runner as RN
    if a.cmd == "estimate":
        from . import key as K
        if (a.backend or "").strip().lower() == "api":
            print(f"[错误] {RN.E_API}", file=sys.stderr)
            return 2
        try:
            est = RN.estimate({"episodes": K.resolve(a.episodes), "narratives": a.narratives, "arms": a.arms,
                               "probe": a.probe, "engines": a.engines, "backend": a.backend, "refresh": a.refresh})
        except (RN.RunnerError, KeyError) as err:
            print(f"[错误] {err}", file=sys.stderr)
            return 2
        print(_estimate_summary(est))
        for note in est["notes"]:
            print(f"  · {note}")
        return 0
    if a.cmd in ("run", "resume"):
        from . import jobs as JB
        if getattr(a, "i_understand_this_is_billed", False):
            print(f"[错误] {RN.E_API}", file=sys.stderr)
            return 2
        try:
            if a.cmd == "resume":
                est = RN.estimate_resume(a.run_id)       # this run's own remaining calls, priced now
                if a.estimate:
                    given = RN.fresh_estimate(a.estimate)
                    if given.get("run_id") != a.run_id:
                        raise RN.EstimateError(f"估算「{a.estimate}」不是运行 {a.run_id} 剩下的调用：不能用它继续这次运行")
            else:
                est = RN.fresh_estimate(a.estimate)
                if est.get("run_id"):
                    raise RN.EstimateError(f"估算「{a.estimate}」是继续运行 {est['run_id']} 的估算：用 resume")
                est = {**est, **RN.live_cost(est)}       # uncached now, not when the estimate was made
        except KeyError as err:
            print(f"[错误] {err.args[0] if err.args else err}", file=sys.stderr)
            return 2
        except RN.RunnerError as err:
            print(f"[错误] {err}", file=sys.stderr)
            return 2
        print(_estimate_summary(est))
        if not _confirm(est, a):
            return 1
        rid = a.run_id if a.cmd == "resume" else RN.create_run(est)
        status = JB.start(rid, background=False)
        r = RN.read_run(rid)
        print(f"[运行 {rid}] {status}" + (f"：{r['pause']['message']}" if r.get("pause") else "")
              + (f"：{r['error']}" if r.get("error") else ""))
        return 0 if status in ("done", "paused") else 1
    if a.cmd == "score":
        from . import report as RP
        from . import score as SC
        SC.score_run(a.run_id, engines=a.engines.split(",") if a.engines else None, n_paths=a.paths)
        rep = RP.build(a.run_id)
        RN.write_run(a.run_id, status="done")
        print(rep["headline"])
        return 0
    if a.cmd == "report":
        from . import report as RP
        from . import score as SC
        SC.restat(a.run_id)
        rep = RP.build(a.run_id)
        print(rep["headline"])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
