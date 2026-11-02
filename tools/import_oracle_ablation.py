"""The research round's oracle-macro ablation (SIM_DESIGN §2 E1, §10.1 B): how much of the factor round's skill under
a known factor path is credit-spread skill. Writes scenariolab/evaluation/research/oracle_ablation.json, which the
evaluation tab (情景条件检验 → B. 已知因子路径时能复现多少) shows as its explanation card.

    python tools/import_oracle_ablation.py --from <scratchpad>/sim_redesign            # write the committed file
    python tools/import_oracle_ablation.py --from <scratchpad>/sim_redesign --check    # exit 1 if it differs

The research folder is read, never written (MSL_RESEARCH_DIR is the default for --from). Its inputs:

* critic/work/c10_paths.csv.gz: the stored conditional paths of F, F without GZ and F without credit spreads
  (rolling and split protocols). The app's harness (scenariolab/evaluation/harness.py, the port of simharness)
  recomputes their metrics from those paths: R² against the training-mean drift with its 90% moving-block bootstrap
  interval (1000 draws, seed 0) and n_indep. The point values reproduce the critic's c10.log exactly (checked).
* critic/work/c11_metrics.csv: the pure-macro and GZ-only rows as the critic stored them (its own 300-draw interval;
  c11 kept no paths).
* critic/work/c18.log: the paired block-bootstrap differences F − ablation (2000 draws), parsed from the log.

Every model is a linear transmission fitted on data to the train end and fed the realized factor path; the score is
the mean R² over the 10 risk assets of the cumulative log price return, pooled over rolling origins 1985–2026
(refit every January). The research runs drew no bands (band=False), so there is no coverage or CRPSS here. The
check values are SIM_DESIGN §2 E1's; nothing here calls an LLM or downloads anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "scenariolab" / "evaluation" / "research" / "oracle_ablation.json"
SCHEMA = "oracle-ablation/1"
HORIZONS = (1, 4, 12, 20)
GROUP = "1985-2026"
# (id, Chinese name, source)
MODELS = [
    ("F", "因子轮完整模型 F：宏观、油价、利率、美元、Baa 利差、GZ 信用利差、期限溢价 × 高核心通胀", "c10"),
    ("F_noGZ", "F 去掉 GZ 信用利差", "c10"),
    ("F_nocredit", "F 去掉 GZ 和 Baa 两个信用利差", "c10"),
    ("MACRO5", "只用宏观：工业产出、商品与服务通胀、政策利率、失业率", "c11"),
    ("GZonly", "只用 GZ 信用利差", "c11"),
]
SHORT = {"F": "完整模型 F", "F_noGZ": "去掉 GZ", "F_nocredit": "去掉两个信用利差", "MACRO5": "只用宏观",
         "GZonly": "只用 GZ"}
# SIM_DESIGN §2 E1 (pooled rolling mean10 R², 1/4/12/20Q)
CHECK = {"F": (0.226, 0.223, -0.007, -0.030), "F_noGZ": (0.128, 0.118, -0.119, -0.133),
         "F_nocredit": (-0.003, -0.016, -0.258, -0.285), "MACRO5": (-0.081, -0.145, -0.373, -0.421),
         "GZonly": (0.165, 0.149, 0.110, 0.108)}
CHECK_TOL = 0.0005
INPUTS = ("critic/work/c10_paths.csv.gz", "critic/work/c11_metrics.csv", "critic/work/c18.log", "critic/work/c10.log")
PAIR_RE = re.compile(r"F-(\w+) ([+-]\d+\.\d+) \[([+-]\d+\.\d+),([+-]\d+\.\d+)\]")
HEAD_RE = re.compile(r"h=\s*(\d+)Q n_origins=(\d+) n_indep=(\d+):")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(r) -> dict:
    def f(x):
        x = float(x)
        return round(x, 4) + 0.0 if x == x else None          # + 0.0: no "-0.0" in the file
    return {"r2": f(r["r2"]), "lo90": f(r["r2_lo90"]), "hi90": f(r["r2_hi90"]), "n_indep": int(r["n_indep"])}


def build(src: Path) -> dict:
    import pandas as pd

    from scenariolab.evaluation import harness as eh
    missing = [p for p in INPUTS if not (src / p).exists()]
    if missing:
        raise SystemExit(f"研究轮的文件不在 {src}：{', '.join(missing)}")
    paths = pd.read_csv(src / INPUTS[0], parse_dates=["train_end", "origin", "end"])
    m10 = eh.metrics(paths, n_boot=1000, seed=0)
    m11 = pd.read_csv(src / INPUTS[1])
    rows = []
    for mid, zh, source in MODELS:
        m = m10 if source == "c10" else m11
        cells, eq = {}, {}
        for h in HORIZONS:
            sel = m[(m["model"] == mid) & (m["protocol"] == "rolling") & (m["group"] == GROUP) & (m["h_q"] == h)]
            r = sel[sel["asset"] == "mean10"]
            if len(r) != 1:
                raise SystemExit(f"{mid} {h}Q: expected one mean10 row, found {len(r)}")
            cells[str(h)] = _cell(r.iloc[0])
            u = sel[sel["asset"] == "us_equity"]
            eq[str(h)] = round(float(u.iloc[0]["r2"]), 4) + 0.0 if len(u) == 1 else None
        rows.append({"id": mid, "zh": zh, "short": SHORT[mid], "source": source, "cells": cells,
                     "us_equity_r2": eq,
                     "interval": ("应用的 harness 用研究轮存下的路径重算（1000 次块自助法）" if source == "c10"
                                  else "研究轮 c11 存下的指标（300 次块自助法）")})
    checks = []
    for r in rows:
        for h, want in zip(HORIZONS, CHECK[r["id"]]):
            got = r["cells"][str(h)]["r2"]
            checks.append({"model": r["id"], "h": h, "want": want, "got": got, "ok": abs(got - want) <= CHECK_TOL})
    paired = []
    for line in (src / INPUTS[2]).read_text(encoding="utf-8").splitlines():
        hm = HEAD_RE.search(line)
        if not hm:
            continue
        h, n_or, n_ind = (int(x) for x in hm.groups())
        for b, d, lo, hi in PAIR_RE.findall(line):
            paired.append({"a": "F", "b": b, "h": h, "diff": float(d), "lo90": float(lo), "hi90": float(hi),
                           "n_origins": n_or, "n_indep": n_ind})
    if len(paired) != 12:
        raise SystemExit(f"c18.log: expected 12 paired differences, parsed {len(paired)}")
    by = {r["id"]: r["cells"] for r in rows}

    def g(mid, h):
        t = f"{by[mid][str(h)]['r2']:+.2f}"
        return "0.00" if t in ("+0.00", "-0.00") else t
    text = (f"已知因子路径时（1985–2026 年滚动起点合并，1 个季度 {by['F']['1']['n_indep']} 个独立窗口），"
            f"只用宏观的 R² 在 1 个季度和 4 个季度是 {g('MACRO5', 1)} 和 {g('MACRO5', 4)}；只用 GZ 信用利差是 "
            f"{g('GZonly', 1)} 和 {g('GZonly', 4)}；因子轮的完整模型 F 是 {g('F', 1)} 和 {g('F', 4)}，去掉两个信用利差后"
            f"是 {g('F_nocredit', 1)} 和 {g('F_nocredit', 4)}。已知宏观的能力几乎全部来自信用利差：情景对金融条件怎么说，"
            "比它对宏观怎么说更决定资产结果。")
    return {"schema": SCHEMA, "label": "研究轮（批评复核，2026-11-02）",
            "title": "已知因子路径时，能力来自哪里：因子轮模型 F 的消融",
            "protocol": "滚动起点：每年 1 月用到上年 12 月的数据重新拟合，起点 1985–2026 年合并；已知的是实现了的因子路径；"
                        "R² 相对训练期平均漂移，10 个风险资产取平均；区间是 90% 移动块自助法",
            "group": GROUP, "assets": "mean10", "horizons": list(HORIZONS),
            "models": rows, "paired": paired, "checks": checks, "text": text,
            "notes": ["这几次研究轮运行没有画区间，所以没有 80% 覆盖率和 CRPSS。",
                      "12 和 20 个季度：每个十年只有 2–3 个独立窗口，只能当个案。",
                      "这是月度传导模型在已知因子路径下的上界检验，不是情景的模拟能力：情景说错时，能力会更低。"],
            "sources": {p: _sha(src / p) for p in INPUTS},
            "spec": "SIM_DESIGN §2 E1（批评 c10、c11、c18）",
            "generated_by": "tools/import_oracle_ablation.py"}


def dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="研究轮的已知因子路径消融 → oracle_ablation.json")
    ap.add_argument("--from", dest="src", default=os.environ.get("MSL_RESEARCH_DIR"),
                    help="研究轮目录 sim_redesign（默认取 MSL_RESEARCH_DIR）")
    ap.add_argument("--check", action="store_true", help="与已提交的文件比较（不写）")
    a = ap.parse_args(argv)
    if not a.src:
        print("需要 --from <sim_redesign>（或设置 MSL_RESEARCH_DIR）", file=sys.stderr)
        return 2
    rep = build(Path(a.src))
    bad = [c for c in rep["checks"] if not c["ok"]]
    for c in bad:
        print(f"检查值不符：{c['model']} {c['h']}Q 期望 {c['want']:+.3f}，得到 {c['got']:+.4f}", file=sys.stderr)
    text = dumps(rep)
    if a.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print("一致" if same else f"{OUT} 与重新导入的结果不同")
        return 0 if same and not bad else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"写入 {OUT.relative_to(ROOT)}：{len(rep['models'])} 个模型、{len(rep['paired'])} 个成对差，"
          f"检查值 {len(rep['checks']) - len(bad)}/{len(rep['checks'])} 一致")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
