"""The narrative test's report: what the evaluation tab (情景条件检验 → C. 叙事检验, SIM_DESIGN §8.7, §10.1) renders.

    GET /api/evaluation/narrative/runs            -> {"runs": [run summary, ...]}   (runs() below)
    GET /api/evaluation/narrative/runs/{run_id}   -> report.json of that run (build()), or its status while running
    research run id: "research_2026_11_01"        -> research_report() (from the committed archive)

report.json ("narrative-report/1"). Every number comes from the run's files; every text is Chinese; no field holds
LLM text except translations[].quote-like fields inside "episodes[].translations" (rendered as text, never HTML).

    schema, id, kind (research | mock | real), status, created, backend, model, narratives (v1 | v2), engines [Q, E]
    label            the run's main label; labels [...]: every label that applies (研究轮（代译，不盲，2026-11-01） /
                     模拟后端 … 流程检验 / 研究轮代译（无引文） / 识别出年代的回放（不是盲测） / 试运行)
    headline         one paragraph generated from the pre-registered claims (it never says more than they allow)
    claims           [{id C1..C4, claim, passed, evaluated, text, mde, conditions [{a, b, assets, horizon, set,
                     diff, lo90, hi90, n, mde, passed}]}]
    tn               {id, version, sha256, file}: the pre-registration record the claims come from
    legend           {arms {id: {name, zh, kind}}, engines {id: zh}, asset_sets {id: zh}, horizons [...],
                     cma {id: zh}}: display names (the UI shows these, never the ids)
    scoreboard       {set: "clean", assets: "equity9", cma, rows [{variant "Q:N2", engine, arm, cma, horizon,
                     n_episodes, n_cells, mae, mae_lo90, mae_hi90, crps, cov80, cov80_lo90, cov80_hi90, pit_mad, bias,
                     rmse, width, corr_median_realized, anecdote}]}; anecdote is true when fewer than 5 episodes
                     reach the horizon (the UI greys those cells and shows 「只能当个案」)
    groups           {asset set: scoreboard-like rows at 4Q and the window end} for us_equity, gold, crude, rates,
                     all13 (the mechanical assets are scored apart, §4.1)
    paired           [{a, b, a_cma, b_cma, horizon, assets, set, n_episodes, diff, lo90, hi90, sd, mde, mse_skill,
                     skill_lo90, skill_hi90, a_wins_share, favourable, unfavourable, per_episode {code: diff}}]
    n_per_horizon    {set: {horizon: episodes}};  mde {value, text};  chance {tests, favourable, unfavourable,
                     expected_each_side, level}
    decomposition    {"N2|end": {n_episodes, total_mean, hypothesis_mean, simulation_mean, *_mabs, per_episode}}
    discrimination   {episodes [{code, right_mae, wrong_mae, right_better}], share_right_better,
                     right_minus_wrong {diff, lo90, hi90, n_episodes, mde}: N1 − N-wrong (negative = the right
                     narrative is closer)}
    strata           {short | bust | ordinary | synth: scoreboard rows (equity9, window end)}
    episodes         [{code, window {spec_code, start, end, set, bust, useq, dd, …}, n_quarters, skipped, reason,
                     arms {arm: {notes, statements, …}}, translations {arm: {status, label, refused, missing, …}},
                     series {variant: {asset: {p10, p50, p90, realized}}}}]: quarterly cumulative log returns for
                     us_equity, nasdaq, gold and treasury_10y (the small multiples)
    blindness        audit.py's panel (prompts, narratives, recognition, contamination, recall, translations)
    ts_inputs        Q − E on the oracle arms and the narratives, each engine's mean |PIT − 0.5| (read by P9's T-S)
    calibration      {engine: {arm: mean |PIT − 0.5| at the window end}}
    files            the run's files (score_rows.csv, fields.csv, stats.json, episodes.json, audit.json, …)
    notes            [...]
The research run adds checks {ok, n, failed, max_dev, max_interval_dev} (recomputation against the published
numbers) and corrections [...] (the critic's corrections), and its "series" are points at the scored horizons.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from ... import jsonio
from . import common as C
from . import runner as RN
from . import stats as ST

SCHEMA = "narrative-report/1"
ANECDOTE_MIN = 5


def legend() -> dict:
    return {"arms": {k: {"name": v["name"], "zh": v["zh"], "kind": v["kind"]} for k, v in C.ARMS.items()},
            "engines": dict(C.ENGINES), "asset_sets": dict(C.ASSET_SET_ZH), "horizons": list(C.HORIZONS),
            "cma": dict(C.CMA_ZH)}


def _split(v: str) -> tuple[str, str]:
    e, _, a = str(v).partition(":")
    return e, a


def _rows(pooled: list[dict], assets: str, n_per_h: dict, horizons=C.HORIZONS) -> list[dict]:
    out = []
    for r in pooled:
        if r["assets"] != assets or r["horizon"] not in horizons:
            continue
        e, a = _split(r["variant"])
        if a == "V" and assets != "us_equity":         # the identity bound exists for us_equity only
            continue
        out.append({**r, "engine": e, "arm": a, "anecdote": (r.get("n_episodes") or 0) < ANECDOTE_MIN})
    out.sort(key=lambda r: (r["engine"], list(C.ARMS).index(r["arm"]) if r["arm"] in C.ARMS else 99,
                            C.HORIZONS.index(r["horizon"]) if r["horizon"] in C.HORIZONS else 9, r.get("cma") or ""))
    return out


def headline(claims: list, kind: str, audit: dict | None, mde: float | None) -> str:
    parts = []
    if kind == "mock":
        parts.append("这是模拟后端的流程检验：翻译是研究轮的代译、合成片段的参考翻译或空表，下面的数字不是叙事检验的结论。")
    for c in claims:
        if c.get("always"):
            continue
        if not c.get("evaluated", True):          # an unevaluated claim is not stated, not even as a heading
            parts.append(f"{c['id']} 未评估（缺少需要的臂或片段）。")
        elif kind == "mock":                      # a pipeline check concludes nothing
            parts.append(f"{c['id']}：流程跑通，条件{'满足' if c.get('passed') else '不满足'}（不作结论"
                         + ("；只依赖已知因子路径的臂，不涉及翻译" if c["id"] == "C2" else "") + "）。")
        else:
            parts.append(f"{c['id']}：{c['text']}。")
    rec = ((audit or {}).get("recognition") or {})
    if rec.get("n"):
        parts.append(f"模拟探针总回答“不知道”，识别率 {rec['recognised']}/{rec['n']} 不是测量。" if kind == "mock" else
                     f"识别率 {rec['recognised']}/{rec['n']}（探针认出了年代的比例）。")
    con = ((audit or {}).get("contamination") or {}).get("overall") or {}
    if con.get("n"):
        unc = con.get("baseline_uncond")
        parts.append(f"污染指数：翻译值与实现值的中位 |z| 为 {con['median_abs_z']:.2f}（历史均值基线 "
                     f"{con['baseline_prior']:.2f}，模型无条件补全基线 {unc:.2f}）。" if unc is not None else
                     f"污染指数：翻译值与实现值的中位 |z| 为 {con['median_abs_z']:.2f}（历史均值基线 "
                     f"{con['baseline_prior']:.2f}）。")
    if mde is not None:
        parts.append(f"最小可检出差异约 {mde:.3f} log（股票类 9 个资产的窗口末 MAE）。")
    return "".join(parts)


def build(rid: str, *, audit: dict | None = None, write: bool = True) -> dict:
    """report.json of a scored run (see the module docstring)."""
    from . import audit as AU
    run = RN.read_run(rid)
    d = RN.run_dir(rid)
    stats = json.loads((d / "stats.json").read_text(encoding="utf-8"))
    eps = json.loads((d / "episodes.json").read_text(encoding="utf-8"))
    audit = audit if audit is not None else AU.audit_run(rid)
    s = run.get("settings") or {}
    head_cma = stats.get("head_cma", "historical")
    n_clean = (stats.get("n_per_horizon") or {}).get("clean") or {}
    pooled = stats.get("pooled") or {}
    clean_rows = [r for r in pooled.get("clean", []) if r.get("cma") == head_cma]
    c1 = next((c for c in stats.get("claims") or [] if c["id"] == "C1"), None)
    mde = c1.get("mde") if c1 else None
    labels = list(audit.get("labels") or [])
    if "Q" in (s.get("engines") or ["Q"]):
        labels.append(f"季度模型：{C.LABELS['trial']}")
    kind = run.get("kind") or "real"
    rep = {"schema": SCHEMA, "id": rid, "kind": kind, "status": "done", "created": run.get("created"),
           "backend": run.get("backend"), "model": run.get("model"), "narratives": s.get("narratives"),
           "engines": s.get("engines"), "label": labels[0] if labels else None, "labels": labels,
           "tn": stats.get("tn"), "legend": legend(), "cma": {"headline": head_cma, "arms": list(C.CMA_ARMS)},
           "headline": headline(stats.get("claims") or [], kind, audit, mde),
           "claims": stats.get("claims") or [],
           "scoreboard": {"set": "clean", "assets": "equity9", "cma": head_cma,
                          "rows": _rows(clean_rows, "equity9", n_clean)},
           "groups": {a: _rows(clean_rows, a, n_clean, ("4Q", "end"))
                      for a in ("us_equity", "gold", "crude", "rates", "all13")},
           "paired": [p for p in stats.get("paired") or [] if p["set"] == "clean"],
           "paired_strata": [p for p in stats.get("paired") or [] if p["set"] != "clean"],
           "n_per_horizon": stats.get("n_per_horizon"),
           "mde": {"value": mde, "text": f"最小可检出差异约 {mde:.3f}" if mde is not None else None},
           "chance": stats.get("chance"), "decomposition": stats.get("decomposition"),
           "discrimination": stats.get("discrimination"),
           "strata": {k: [r for r in v if r["assets"] == "equity9" and r["horizon"] == "end"
                          and r.get("cma") == head_cma] for k, v in pooled.items() if k != "clean"},
           "episodes": [{**{k: v for k, v in e.items() if k != "series"}, "series": e.get("series") or {}}
                        for _, e in sorted(eps.items())],
           "blindness": audit, "ts_inputs": stats.get("ts_inputs"), "calibration": stats.get("calibration"),
           "files": {k: v for k, v in {"score_rows": "score_rows.csv", "fields": "fields.csv", "stats": "stats.json",
                                       "episodes": "episodes.json", "audit": "audit.json", "manifest": "manifest.json",
                                       "plan": "plan.json"}.items() if (d / v).exists()},
           "notes": [C.LABELS["trial"] + "：季度模型通过切换关 T-S 之前只作试运行"],
           "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if write:
        (d / "report.json").write_text(jsonio.dumps(rep), encoding="utf-8")
    return rep


# ================================================================ the research run

RESEARCH_LEGEND_ARMS = {"U": "无叙事", "cond": "fan_cond：已知 8 个视图变量的阶段均值（部分是市场价格）",
                        "oracle": "已知因子路径", "N1": "叙事 N1", "N2": "叙事 N2（加金融条件）", "N3": "叙事 N3（不含数字）"}


def research_report() -> dict:
    """The research round's run (研究轮（代译，不盲，2026-11-01）) in the report format, recomputed from its stored
    rows: the original 11 episodes (the scorer's published numbers) and the clean 10 (E03 moved to the short set).
    Cached in memory and in data/results/narrative/research_2026_11_01.report.json (keyed by the archive's hash)."""
    import hashlib

    from . import research as RS
    digest = hashlib.sha256(b"".join((C.RESEARCH_DIR / f).read_bytes() for f in
                                     (RS.EPISODES_FILE, RS.ROWS_FILE, RS.PUBLISHED_FILE))
                            + f"text-{REPORT_TEXT_VERSION}".encode()).hexdigest()[:16]
    return _research_report(digest)


REPORT_TEXT_VERSION = 2          # bumped when the report's reader text changes (the research cache is rebuilt)


@lru_cache(maxsize=1)
def _research_report(digest: str) -> dict:
    from . import research as RS
    cache = C.runs_dir() / f"{RS.RESEARCH_ID}.report.json"
    try:
        hit = json.loads(cache.read_text(encoding="utf-8"))
        if hit.get("archive_digest") == digest and hit.get("schema") == SCHEMA:
            return hit
    except (OSError, ValueError):
        pass
    rep = _build_research_report()
    rep["archive_digest"] = digest
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(jsonio.dumps(rep), encoding="utf-8")
    except OSError:
        pass
    return rep


def _build_research_report() -> dict:
    from . import research as RS
    R = RS.load_rows()
    R = R[R["horizon"] != "12Q"]
    R2 = R.copy()
    R2["variant"] = [f"{e}:{a}" for e, a in zip(R2["engine"], R2["arm"])]
    sets = {"original11": list(RS.ANCHORED), "clean": list(RS.CLEAN), "short": ["E03", "E06"],
            "bust": [c for c in RS.CLEAN if c in RS.BUSTS], "ordinary": [c for c in RS.CLEAN if c not in RS.BUSTS]}
    pooled = {k: ST.pooled(R2[R2["horizon"] != "endc"], v) for k, v in sets.items()}
    npph = {k: {h: int(R2[R2["code"].isin(v) & (R2["horizon"] == h)]["code"].nunique()) for h in C.HORIZONS}
            for k, v in sets.items()}
    pairs = []
    for set_name in ("original11", "clean"):
        for h in C.HORIZONS:
            for a in ("E0:N1", "E0:N2", "E0:N3", "E0:cond", "E0:oracle", "F:oracle"):
                p = ST.paired(R2, (a, "historical"), ("E0:U", "historical"), h, "equity9", codes=sets[set_name])
                if p:
                    pairs.append({**p, "set": set_name})
            for a in ("G:N1", "G:N2", "G:N3", "G:oracle"):
                p = ST.paired(R2, (a, "historical"), ("G:U", "historical"), h, "equity9", codes=sets[set_name])
                if p:
                    pairs.append({**p, "set": set_name})
        for arm in ("N1", "N2", "N3", "oracle"):
            p = ST.paired(R2, (f"G:{arm}", "historical"), (f"E0:{arm}", "historical"), "endc", "equity9",
                          codes=sets[set_name])
            if p:
                pairs.append({**p, "set": set_name})
    chk = RS.checks(RS.recompute(intervals=False))
    arch = RS.load_archive()
    guesses = []
    for code, ep in arch["episodes"].items():
        g = (ep.get("guess") or {}).get("guess") or {}
        guesses.append({"code": code, "guess": g.get("year_range") or g.get("narrative_months"),
                        "confidence": (ep.get("guess") or {}).get("confidence"), "recognised": True})
    mde_p = next((p for p in pairs if p["a"] == "E0:N2" and p["b"] == "E0:U" and p["horizon"] == "end"
                  and p["set"] == "original11"), None)
    series = {}
    for code in RS.CODES:
        sub = R2[(R2["code"] == code) & (R2["asset"] == "us_equity")]
        series[code] = {v: {h: {k: round(float(g[k].iloc[0]), 4) for k in ("q10", "q50", "q90", "y")}
                            for h, g in gv.groupby("horizon")} for v, gv in sub.groupby("variant")}
    pub_lines = [p for p in RS.load_published()["critic"] if p["stat"] != "chance"]
    text = ("研究轮（代译，不盲）：在原来 11 个片段上，叙事没有带来可检出的改进：月度引擎的 N2 − 无叙事在窗口末为 "
            f"{mde_p['diff']:+.3f} [{mde_p['lo90']:+.3f}, {mde_p['hi90']:+.3f}]，最小可检出差异约 {mde_p['mde']:.3f}。"
            if mde_p else "研究轮（代译，不盲）。")
    text += ("30 个叙事对无叙事的成对区间里 1 个有利、3 个不利，与偶然相当。翻译者 12 个片段全部认出了年代："
             "这是识别出年代的回放，不是盲测；“Q”是研究轮的季度模型 G，不是现在的季度模型。")
    return {"schema": SCHEMA, "id": RS.RESEARCH_ID, "kind": "research", "status": "done", "created": "2026-11-01",
            "backend": "代译（研究轮的翻译角色，不是本机 Claude Code）", "model": None, "narratives": "v1",
            "engines": ["E0", "F", "G", "VB"], "label": C.LABELS["research"],
            "labels": [C.LABELS["research"], C.LABELS["legacy"], C.LABELS["recognised"]],
            "tn": None, "headline": text, "claims": [],
            "legend": {**legend(), "engines": dict(RS.ENGINE_ZH), "research_arms": RESEARCH_LEGEND_ARMS},
            "cma": {"headline": "historical", "arms": ["historical"]},
            "scoreboard": {"set": "original11", "assets": "equity9", "cma": "historical",
                           "rows": _rows(pooled["original11"], "equity9", npph["original11"])},
            "scoreboard_clean": {"set": "clean", "assets": "equity9", "cma": "historical",
                                 "rows": _rows(pooled["clean"], "equity9", npph["clean"])},
            "groups": {a: _rows(pooled["original11"], a, npph["original11"], ("4Q", "end"))
                       for a in ("us_equity", "gold", "crude", "rates", "all13")},
            "paired": [p for p in pairs if p["set"] == "original11"],
            "paired_strata": [p for p in pairs if p["set"] != "original11"],
            "n_per_horizon": npph,
            "mde": {"value": mde_p["mde"] if mde_p else None,
                    "text": f"最小可检出差异约 {mde_p['mde']:.3f}" if mde_p else None},
            "chance": {"tests": 30, "favourable": 1, "unfavourable": 3, "expected_each_side": 1.5, "level": 0.9,
                       "source": "critic/c17（重算一致）"},
            "strata": {k: [r for r in v if r["assets"] == "equity9" and r["horizon"] == "end"]
                       for k, v in pooled.items() if k not in ("original11", "clean")},
            "episodes": [{"code": c, "window": {"spec_code": c, "months": list(RS.WINDOWS[c][:2]),
                                                "mode": RS.WINDOWS[c][2], "label": RS.WINDOWS[c][3],
                                                "set": "clean" if c in RS.CLEAN else "short",
                                                "bust": c in RS.BUSTS},
                          "points": series.get(c, {})} for c in RS.CODES],
            "blindness": {"recognition": {"n": len(guesses), "recognised": len(guesses), "rate": 1.0,
                                          "probes": guesses, "source": "研究轮翻译者写完翻译后的自我判断"},
                          "labels": [C.LABELS["recognised"]]},
            "critic": [{"source": p["source"], "line": p["line"]} for p in pub_lines],
            "checks": {k: chk[k] for k in ("ok", "n", "max_dev", "max_interval_dev")} | {"failed": chk["failed"][:20]},
            "corrections": RS.corrections(),
            "files": {"score_rows": "scenariolab/evaluation/narrative/research/score_rows.csv.gz",
                      "published": "scenariolab/evaluation/narrative/research/published.json",
                      "episodes": "scenariolab/evaluation/narrative/research/episodes_v1.json"},
            "notes": [c["text"] for c in RS.corrections()]}


def runs() -> list[dict]:
    """The run selector: the research run first, then the mock and real runs (newest first)."""
    from . import research as RS
    out = []
    if RS.available():
        out.append({"id": RS.RESEARCH_ID, "kind": "research", "status": "done", "label": C.LABELS["research"],
                    "backend": "代译", "model": None, "created": "2026-11-01", "narratives": "v1",
                    "engines": ["E0", "F", "G", "VB"], "n_episodes": len(RS.CODES)})
    for r in RN.list_runs():
        r["has_report"] = (RN.run_dir(r["id"]) / "report.json").exists()
        out.append(r)
    return out


def load(rid: str) -> dict:
    """The report of a run: research_report() for the research id; report.json when it exists; else the status."""
    from . import research as RS
    if rid == RS.RESEARCH_ID:
        return research_report()
    run = RN.read_run(rid)
    p = RN.run_dir(rid) / "report.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema": SCHEMA, "id": rid, "status": run.get("status"), "progress": run.get("progress"),
            "pause": run.get("pause"), "error": run.get("error"), "kind": run.get("kind")}


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and np.isfinite(x)


def frame(rid: str) -> pd.DataFrame:
    """The score rows of a run as a DataFrame (the research run's from the archive)."""
    from . import research as RS
    if rid == RS.RESEARCH_ID:
        return RS.load_rows()
    return pd.read_csv(RN.run_dir(rid) / "score_rows.csv")
