"""季度情景集 (SIM_DESIGN §11.3, §4.6; AGENT_DESIGN §14): the 13 built-in scenarios rebuilt on the quarterly stack, in
the scenario_paths.npz layout with step_months = 3, for 组合评估 and 提取与破产概率 (portfolio/service.py, scenario_set
"quarterly"). The agent tree's export (agents/tree.export_arrays) writes the same layout, so its sets load the same way.

    python -m scenariolab.portfolio.qsets build [--paths 600] [--seed 20261102]    # about 1-2 minutes, no LLM
    python -m scenariolab.portfolio.qsets status

It is a default only behind the switch gate T-S (evaluation/switch.py): until T-S passes the services keep the monthly
engine's set, the quarterly one can be picked by hand and is labelled 试运行; the build's portfolio step rebuilds it
only once T-S has passed (scenariolab/build.py build_portfolio calls run()).

Each built-in's reviewed QTable (scenarios/qtables/<id>.json, P6) runs in path mode over its quarters (two-sided
conditioning: later statements shape earlier quarters), through rollout's public API (from_qtable, run_path), and the
same model goes on without statements to quarter CONT_QUARTERS (40 years, the withdrawal page's longest horizon):
rollout's continuation is exactly "the scenario, then the model unconditionally toward its anchors" (§3.6). The first
SCENARIO_QUARTERS (60, the monthly set's 15 years) are the set's quarterly paths; the rest are its own continuation.

Writes data/results/scenario_paths_q.npz:
  total, income   (S, N, 60, 13) float32: quarterly simple total return and income. The stack's convention: 1 + total
                  = the quarter's price growth + its income (cash: all of its return is income), which is what the
                  services' year_factors reads (price growth = 1 + total - income).
  cpi             (S, N, 61): CPI relative to the start (exp of the cumulative quarterly log inflation d_cpi / 400)
  ids, assets, labels (61 quarter-end months "2026-09", "2026-12", ...), probs (S,), step_months = 3
  cont_total_y, cont_price_y, cont_income_y (S*N, 25, 13), cont_cpi_y (S*N, 26), cont_labels (26), cont_start,
  cont_offset (60 steps), cont_info: years 16-40 as yearly factors of each path's own continuation
and data/results/scenarios_q.json: {start, horizon_months, n_paths, scenarios [{id, name, prob, prior_prob, posterior,
quality}], set {...}}.

Probabilities: the hand-set prior_prob of each QTable, normalised (§7.5, U-9: the posterior tilt is off by default);
the monthly build's posterior (scenarios.json) is kept beside it for reference.

Paths the stack flags as diverging (rollout.tail_flags: GZ > 15, unemployment > 20%, a quarterly US-equity log return
beyond ±100%; also any asset's quarterly log return beyond ±150%, income above 100% a quarter, or anything not finite)
are not kept. Inside the 60 scenario quarters such a path is replaced by a clean path of the same scenario, drawn at
random with a fixed seed; in the continuation only its continuation is replaced, by a clean continuation of the same
scenario. A scenario with more than MAX_REPLACED of its paths flagged inside its scenario quarters is left out (its
probability is spread over the others in proportion). Every share is recorded in the set's quality block and shown
with every result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from .. import config, jsonio
from . import service as svc
from . import withdrawal as wd

VERSION = 1
SCENARIO_QUARTERS = config.HORIZON_MONTHS // 3          # 60: the monthly set's 15 years
CONT_QUARTERS = 4 * svc.HORIZON_BOUNDS[1]               # 160: to year 40
DEFAULT_PATHS = 600                                     # the monthly set's paths per scenario
DEFAULT_SEED = 20261102
MAX_REPLACED = 0.5                                      # more flagged scenario paths than this: the scenario is left out
ASSET_LOG_BOUND = 1.5                                   # |quarterly log return| beyond this is divergence, not a quarter
INCOME_BOUND = 1.0                                      # income above 100% of the value in a quarter: divergence


def _seed(seed: int, sid: str) -> int:
    return zlib.crc32(f"qsets:{int(seed)}:{sid}".encode()) % 2**31


def quarter_end_labels(origin: str, T: int) -> list[str]:
    """T + 1 quarter-end months from the end of the quarter before `origin`: '2026Q4' -> '2026-09', '2026-12', ..."""
    import pandas as pd
    q0 = pd.Period(origin, "Q")
    return [(q0 - 1 + k).end_time.strftime("%Y-%m") for k in range(int(T) + 1)]


def flags(roll, assets: list) -> np.ndarray:
    """(n, T) path-quarters that left history far behind or are not finite."""
    from ..qmodel import rollout as R
    tot = np.asarray(roll.ax.total, float)
    inc = np.asarray(roll.ax.income, float)
    cpi = np.asarray(roll.fx.x["d_cpi"], float)
    us = tot[:, :, assets.index("us_equity")]
    with np.errstate(invalid="ignore"):
        bad = R.tail_flags(np.asarray(roll.fx.levels["gz"], float), np.asarray(roll.fx.levels["unrate"], float), us)
        bad |= ~np.isfinite(tot).all(axis=2) | ~np.isfinite(inc).all(axis=2) | ~np.isfinite(cpi)
        bad |= (np.abs(np.nan_to_num(tot, nan=0.0)) > ASSET_LOG_BOUND).any(axis=2)
        bad |= (np.nan_to_num(inc, nan=0.0) > INCOME_BOUND).any(axis=2)
    return bad


def clean_paths(total_log, income, cpi_q, bad, rng, head: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Replace flagged paths (module docstring): rows flagged inside the first `head` quarters take a clean row of the
    same scenario (whole path); rows flagged only later keep their head and take a clean row's continuation.
    Returns (total_log, income, cpi_q, quality) with every row finite, or quality["excluded"] when too few are clean."""
    n = total_log.shape[0]
    head_bad = bad[:, :head].any(axis=1)
    q = {"n": int(n), "head_flagged": round(float(head_bad.mean()), 4)}
    ok = np.flatnonzero(~head_bad)
    if head_bad.mean() > MAX_REPLACED or ok.size == 0:
        return total_log, income, cpi_q, {**q, "excluded": True}
    rows = np.arange(n)
    rows[head_bad] = rng.choice(ok, size=int(head_bad.sum()), replace=True)
    total_log, income, cpi_q, bad = total_log[rows], income[rows], cpi_q[rows], bad[rows]
    cont_bad = bad[:, head:].any(axis=1)
    q["cont_flagged"] = round(float(cont_bad.mean()), 4)
    good = np.flatnonzero(~cont_bad)
    if cont_bad.any():
        if good.size == 0:
            return total_log, income, cpi_q, {**q, "excluded": False, "no_continuation": True}
        donor = rng.choice(good, size=int(cont_bad.sum()), replace=True)
        idx = np.flatnonzero(cont_bad)
        total_log = total_log.copy()
        income = income.copy()
        cpi_q = cpi_q.copy()
        total_log[idx, head:] = total_log[donor, head:]
        income[idx, head:] = income[donor, head:]
        cpi_q[idx, head:] = cpi_q[donor, head:]
    return total_log, income, cpi_q, {**q, "excluded": False, "no_continuation": False}


def headline(tot, cpi, dcpi, ff_end, assets: list, H: int) -> dict:
    """A scenario's numbers over its H quarters, in the units of the monthly set's scenarios.json headline: US equity
    nominal and real CAGR medians (decimal), the median path's average CPI inflation and the median policy rate at the
    end (percent)."""
    us = tot[:, :H, assets.index("us_equity")]
    years = H / 4.0
    with np.errstate(all="ignore"):
        g = np.prod(1.0 + us, axis=1)
        real = g / (cpi[:, H] / cpi[:, 0])
        out = {"us_equity_cagr_p50": float(np.median(g ** (1.0 / years) - 1.0)),
               "us_equity_real_cagr_p50": float(np.median(real ** (1.0 / years) - 1.0)),
               "cpi_avg": float(np.median(dcpi[:, :H].mean(axis=1))),
               "ff_end_p50": float(np.median(ff_end)) if np.size(ff_end) else None}
    return {k: (round(v, 6) if isinstance(v, float) and np.isfinite(v) else None) for k, v in out.items()}


def _posterior(results: Path) -> dict:
    try:
        d = jsonio.load(results / svc.INDEX_FILE)
    except (OSError, ValueError):
        return {}
    return {str(s.get("id")): s.get("prob") for s in d.get("scenarios", []) if isinstance(s, dict)}


def build(*, n_paths: int = DEFAULT_PATHS, seed: int = DEFAULT_SEED, results: Path | None = None,
          write: bool = True, ids=None, progress=None) -> dict:
    """Run every built-in QTable on the quarterly stack and write the set (module docstring). Returns its summary."""
    from ..evaluation import switch
    from ..qmodel import qtable as QT
    from ..qmodel import rollout as R
    t0 = time.perf_counter()
    results = Path(results or svc.RESULTS_DIR)
    stack = R.get_stack()
    assets = list(R.ASSETS)
    H0 = SCENARIO_QUARTERS
    ids = list(ids or QT.builtin_ids())
    post = _posterior(results)
    heads, conts, kept, notes, quality = [], [], [], [], {}
    for i, sid in enumerate(ids):
        if progress is not None:
            progress(i + 1, len(ids))
        env = QT.builtin(sid)
        name = env.get("name") or sid
        try:
            read = R.from_qtable(env["qtable"], stack, "production", narrative=env["text"], source="builtin")
            H = int(read["H"])
            if H > H0:
                raise R.RolloutError(f"情景有 {H} 个季度，季度情景集只收 {H0} 个季度以内的情景")
            with np.errstate(all="ignore"):
                roll = R.run_path(stack, read["statements"], read["valuation"], H=H, n_paths=int(n_paths),
                                  seed=_seed(seed, sid), cycle_odds=read.get("cycle_odds"),
                                  continuation=CONT_QUARTERS - H, states=False, mode="production")
        except Exception as err:  # noqa: BLE001 - one scenario fails: the set says so and goes on
            quality[sid] = {"excluded": True, "error": f"{type(err).__name__}: {err}"}
            notes.append(f"「{name}」没有算出来（{type(err).__name__}）：这个情景集里没有它，它的概率按比例分给其余情景。")
            continue
        bad = flags(roll, assets)
        with np.errstate(all="ignore"):
            tl, inc, dcpi, q = clean_paths(np.asarray(roll.ax.total, float), np.asarray(roll.ax.income, float),
                                           np.asarray(roll.fx.x["d_cpi"], float), bad,
                                           np.random.default_rng(_seed(seed, f"clean:{sid}")), H0)
        q["tail_share_rollout"] = round(float((roll.info or {}).get("tail_share") or 0.0), 4)
        quality[sid] = q
        if q.get("excluded"):
            notes.append(f"「{name}」有 {100 * q['head_flagged']:.0f}% 的路径在情景的 {H0} 个季度里发散（模型自己标为"
                         "不可信）：这个情景集里没有它，它的概率按比例分给其余情景。")
            continue
        if q.get("no_continuation"):
            notes.append(f"「{name}」的续推全部发散：情景集没有第 16 年以后的续推。")
        cpi = np.concatenate([np.ones((dcpi.shape[0], 1)), np.exp(np.cumsum(dcpi / 400.0, axis=1))], axis=1)
        tot = np.expm1(tl)
        head = headline(tot, cpi, dcpi, np.asarray(roll.fx.levels["ff"], float)[:, H0 - 1][~bad[:, :H0].any(axis=1)],
                        assets, H0)
        heads.append((tot[:, :H0].astype(np.float32), inc[:, :H0].astype(np.float32), cpi[:, :H0 + 1]))
        conts.append(wd.year_factors(tot[:, H0:], inc[:, H0:], cpi[:, H0:] / cpi[:, H0:H0 + 1], months=4))
        kept.append({"id": sid, "name": name, "prior_prob": env.get("prior_prob"), "posterior": post.get(sid),
                     "quality": q, "headline": head})
        if q["head_flagged"] > 0 or q.get("cont_flagged", 0) > 0:
            notes.append(f"「{name}」：{100 * q['head_flagged']:.1f}% 的路径在情景的 {H0} 个季度里发散、"
                         f"{100 * q.get('cont_flagged', 0):.1f}% 的路径在续推里发散（模型自己标为不可信），"
                         "已换成同一情景里没有发散的路径（固定种子随机抽取）。")
    if not kept:
        raise RuntimeError("没有一个情景算出可用的路径：季度情景集没有写")
    prior = np.array([float(k["prior_prob"]) if isinstance(k["prior_prob"], (int, float)) else np.nan for k in kept])
    if not np.isfinite(prior).all() or prior.sum() <= 0:
        prior = np.ones(len(kept))
    probs = prior / prior.sum()
    for k, p in zip(kept, probs):
        k["prob"] = round(float(p), 6)
    labels = quarter_end_labels(stack.origin, H0)
    cont_labels = quarter_end_labels(stack.origin, CONT_QUARTERS)[H0::4]
    st = switch.state(results)
    trial = not st.get("quarterly")
    sem = switch.semantic_version(stack.version)
    n_cont = sum(1 for k in kept if not k["quality"].get("no_continuation"))
    meta = {"engine": "quarterly", "step_months": 3, "name": svc.set_name("quarterly", len(kept)), "version": VERSION,
            "model": sem, "stack": stack.version, "origin": stack.origin, "train_end": stack.train_end,
            "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "n_paths": int(n_paths),
            "seed": int(seed), "scenario_quarters": H0, "continuation_quarters": CONT_QUARTERS - H0,
            "horizon_months": 3 * H0, "trial_when_built": trial,
            "excluded": [s for s in ids if quality.get(s, {}).get("excluded")],
            "quality": quality,
            "probabilities": "手设先验（每个情景的主观先验，归一化；按今天状态的后验倾斜默认关闭）",
            "text": (f"季度情景集：{len(kept)} 个内置情景的季度情景表在季度模型上运行（路径模式，{int(n_paths)} 条路径，"
                     f"起点 {stack.origin}，训练期到 {stack.train_end}），每步一个季度，{H0} 个季度之后同一个模型不加陈述"
                     f"续推到第 40 年。情景概率默认用手设先验。"),
            "notes": notes}
    out = {"set": meta, "kept": [k["id"] for k in kept], "notes": notes, "elapsed_s": round(time.perf_counter() - t0, 1)}
    if not write:
        return out
    S, N = len(kept), int(n_paths)
    total = np.stack([h[0] for h in heads])
    income = np.stack([h[1] for h in heads])
    cpi = np.stack([h[2] for h in heads])
    cf = wd.YearFactors.concat(conts)
    cont_info = {"method": "quarterly_stack", "offset_steps": H0, "quarters": CONT_QUARTERS - H0,
                 "text": "季度模型在情景的 60 个季度之后不加陈述接着往下推（每条路径从它自己的季末状态出发），向长期锚回归",
                 "complete": n_cont == S}
    results.mkdir(parents=True, exist_ok=True)
    path = results / svc.QUARTERLY_FILE
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    arrays = {"total": total, "income": income, "cpi": cpi.astype(np.float64), "ids": np.array([k["id"] for k in kept]),
              "assets": np.array(assets), "labels": np.array(labels), "probs": probs, "step_months": np.array(3),
              "engine": np.array("quarterly"), "version": np.array(str(stack.version))}
    if n_cont == S:
        arrays.update(cont_total_y=cf.total.astype(np.float32), cont_price_y=cf.price.astype(np.float32),
                      cont_income_y=cf.income.astype(np.float32), cont_cpi_y=cf.cpi, cont_labels=np.array(cont_labels),
                      cont_start=np.array(labels[0]), cont_offset=np.array(H0), cont_info=np.array(jsonio.dumps(cont_info)))
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    index = {"start": labels[0], "horizon_months": 3 * H0, "n_paths": N, "step_months": 3,
             "scenarios": [{k2: k[k2] for k2 in ("id", "name", "prob", "prior_prob", "posterior", "headline")} for k in kept],
             "set": meta}
    jsonio.dump(index, results / svc.QUARTERLY_INDEX)
    svc.clear_cache()
    out.update(file=str(path), index=str(results / svc.QUARTERLY_INDEX), shape=list(total.shape),
               continuation=n_cont == S)
    return out


def run(ctx=None, *, results: Path | None = None, **kw) -> dict:
    """scenariolab.build's step signature (the portfolio step calls it once the switch gate T-S has passed)."""
    return build(results=results, **kw)


def status(results: Path | None = None) -> dict:
    results = Path(results or svc.RESULTS_DIR)
    rep = svc.sets_report(results)
    q = next((s for s in rep["sets"] if s["id"] == "quarterly"), {})
    return {"available": q.get("available"), "issue": q.get("issue"), "created": q.get("created"),
            "trial": q.get("trial"), "default": rep["default"], "switch": rep["switch"].get("default_text"),
            "quality": q.get("quality")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scenariolab.portfolio.qsets", description="季度情景集（SIM_DESIGN §11.3）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(jsonio.clean(status()), ensure_ascii=False, indent=1))
        return 0
    res = build(n_paths=a.paths, seed=a.seed, progress=lambda i, n: print(f"[qsets] {i}/{n}", flush=True))
    print(f"[qsets] {res.get('file')}: {res.get('shape')}, continuation {res.get('continuation')}, "
          f"{res['elapsed_s']} s; left out: {res['set']['excluded'] or '无'}")
    for line in res["notes"]:
        print(f"  · {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
