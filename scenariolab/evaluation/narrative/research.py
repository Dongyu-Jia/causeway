"""The research round's narrative test, imported (SIM_DESIGN §8.5: import-standin; §14 P7).

The research round (2026-11-01) ran the end-to-end test on 12 episodes with a stand-in translator: 36 tables
(sim_redesign/narratives/E01–E12/q_n1..3.json), scored by sim_redesign/scorer (out/score_rows.csv, 13 variants) and
re-checked by the critic (critic/work/c01, c17, c21–c23). Its translations are not re-run (USER_DECISIONS); this
module keeps them in the repository (research/, committed) and recomputes the published statistics from the
scorer's stored rows.

    python -m scenariolab.evaluation.narrative import-standin --from <sim_redesign folder> [--check]

writes three files under scenariolab/evaluation/narrative/research/ (deterministic: --check rebuilds them in
memory and compares byte for byte):

    episodes_v1.json    per research code: the window as the research round ran it, the v1 narratives n1/n2/n3, the
                        stand-in translator's own period guess (guess.json), and the 36 tables converted to QTable/1
                        (convert_standin) with the conversion record
    score_rows.csv.gz   the scorer's stored cells (code, variant, asset, horizon, months, y, q10, q50, q90, mean, err,
                        abs_err, width, crps, cov80, pit, degenerate)
    published.json      the check values: the scorer's pooled table (out/score_pooled.csv: 11 anchored episodes) and
                        every number the critic printed for the paired comparisons, the MSE skills on its subsets
                        (without E03; without E03, E05 and E12) and the chance count (c01, c17, c21, c22, c23 logs)

Conversion of a stand-in table (convert_standin). The research quarterly model read the tables' numeric x values only
(sim_redesign/scorer/s03_qmodel.py: "sign-only entries are NOT used"); the import keeps exactly those, so a replay
runs the translation the research round ran:
    a numeric value          -> {"value", "src", "quote": ""}: src stated -> stated; interp, derived and band ->
                                derived; inferred (N3's words mapped to a number by the translator itself, critic 10)
                                -> derived, counted in the record
    x_oil_cause              -> the enum: the research translator wrote 0 = supply, 1 = demand, 0.5 = mixed (the
                                opposite of qdata's 0/1, critic 8); the fix is recorded
    sign-only entries, the cross-quarter constraints, the announced paths, the regime flags and the events are not
    imported (the research model did not read them); their counts are recorded. fc.stance is 未说明 in every row.
Every converted table is QTable/1 read with source "legacy": no quotes, so the quote rules are skipped and every
result from it carries the label 研究轮代译（无引文）.

Recomputation (recompute, checks). The scorer's pooled numbers are recomputed from the rows with its own rule (the
mean over the 11 anchored episodes of each episode's mean over the set's assets); the critic's numbers with exact
ports of its scripts (the same bootstrap generators and call order, so the intervals reproduce as well). checks()
compares every published value with its recomputation: points to ± half a unit of the printed precision (the
scorer's MAE to ± 0.002, the exit test of §14 P7).

The critic's corrections the evaluation tab applies to this run (corrections()): E06 apart (leave-window-out); E03
moved to the short-training set, so the clean research set is 10 episodes; busts reported as a stratum; the number
of paired intervals against chance (1 of 30 favourable); the minimum detectable effect; the label
「识别出年代的回放（不是盲测）」 (the translator recognised 12 of 12 periods); the monthly engine ran B0 with
post-2008 reversion anchors (look-ahead) and "Q" is the research qmodel G, not this quarterly stack.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C

RESEARCH_ID = "research_2026_11_01"
SCHEMA = "narrative-research/1"
EPISODES_FILE = "episodes_v1.json"
ROWS_FILE = "score_rows.csv.gz"
PUBLISHED_FILE = "published.json"
# the research round's episode windows (sim_redesign/scorer/sc_common.EPISODES): months, the context mode its
# prompts were built in, and the scorer's description. Public in the spec (§8.2) as the episodes' spec codes.
WINDOWS = {
    "E01": ("2004-07", "2006-12", "anchored", "measured tightening, oil doubling"),
    "E02": ("1995-07", "1997-12", "anchored", "soft landing, IT boom, strong dollar"),
    "E03": ("1979-10", "1982-08", "anchored", "Volcker disinflation, double dip (80 DB months)"),
    "E04": ("2014-07", "2016-02", "anchored", "oil collapse, strong dollar"),
    "E05": ("2007-07", "2009-06", "anchored", "credit crisis"),
    "E06": ("1973-10", "1975-06", "lwo", "oil embargo, stagflation (leave-window-out)"),
    "E07": ("1994-01", "1995-06", "anchored", "1994 pre-emptive tightening"),
    "E08": ("2020-02", "2021-06", "anchored", "pandemic collapse and reopening"),
    "E09": ("2017-01", "2019-12", "anchored", "late cycle, QT, tariffs"),
    "E10": ("2022-01", "2023-12", "anchored", "2022-23: fastest hikes, energy war"),
    "E11": ("1990-07", "1991-12", "anchored", "Gulf war oil spike, mild recession"),
    "E12": ("2001-01", "2002-12", "anchored", "IT-capex bust, 9/11"),
}
CODES = tuple(WINDOWS)
ANCHORED = tuple(c for c in CODES if WINDOWS[c][2] == "anchored")         # the scorer's 11 pooled episodes
CLEAN = tuple(c for c in ANCHORED if c != "E03")                         # §8.2: E03 has 27 training quarters
BUSTS = ("E05", "E08", "E10", "E12")                                      # §8.2 table: DD <= -0.20
EQ9 = list(C.EQUITY9)
ROW_COLUMNS = ["code", "variant", "asset", "horizon", "months", "y", "q10", "q50", "q90", "mean", "err", "abs_err",
               "width", "crps", "cov80", "pit", "degenerate"]
# research variant -> (engine, arm): E0 = the monthly engine as the research round ran it (B0, fixed reversion
# anchors), F = the harness's F fed realized factors, G = the research qmodel, VB = its valuation block
VARIANTS = {"E-uncond": ("E0", "U"), "E-cond": ("E0", "cond"), "E-oracle": ("E0", "oracle"), "E-n1": ("E0", "N1"),
            "E-n2": ("E0", "N2"), "E-n3": ("E0", "N3"), "F-oracle": ("F", "oracle"), "Q-uncond": ("G", "U"),
            "Q-oracle": ("G", "oracle"), "Q-n1": ("G", "N1"), "Q-n2": ("G", "N2"), "Q-n3": ("G", "N3"),
            "VB-oracle": ("VB", "oracle")}
ENGINE_ZH = {"E0": "月度引擎（研究轮：B0 模型、固定长期锚）", "F": "因子轮 F（已知因子路径）", "G": "研究轮季度模型 G（不是现在的季度模型）",
             "VB": "研究轮估值模块"}
ARM_ZH = {"U": "无叙事", "cond": "fan_cond：已知 8 个视图变量的阶段均值（部分是市场价格）", "oracle": "已知因子路径（oracle）",
          "N1": "叙事 N1", "N2": "叙事 N2（加金融条件）", "N3": "叙事 N3（不含数字）"}
SRC_MAP = {"stated": "stated", "interp": "derived", "derived": "derived", "band": "derived", "inferred": "derived"}
OIL_CAUSE_FIX = {0.0: "supply", 1.0: "demand", 0.5: "mixed"}
OIL_FIX_NOTE = ("x_oil_cause：研究轮翻译器写 0 = 供给、1 = 需求、0.5 = 混合，与 qdata 的 0/1 约定相反（批评第 8 条）；"
                "导入时换成枚举 supply / demand / mixed")
STANDIN_NOTE = "研究轮代译（无引文）：从研究轮的季度表导入，只导入数值条目（研究轮的季度模型只读数值）；x_oil_cause 约定已修正"


# ================================================================ conversion

def _q(v) -> int:
    return int(str(v).lstrip("qQ"))


def convert_standin(table: dict, x_fields=None) -> dict:
    """One research stand-in table -> {"qtable": QTable/1 (legacy), "conversion": what was kept, changed, dropped}."""
    known = set(x_fields) if x_fields is not None else None
    rows_in = table.get("rows") or []
    H = int((table.get("timing") or {}).get("n_quarters") or len(rows_in))
    rec = {"code": table.get("code"), "narrative": table.get("narrative"), "n_quarters": H,
           "schema_version": table.get("schema_version"), "values": 0, "src": {}, "inferred": 0,
           "oil_cause": {"supply": 0, "demand": 0, "mixed": 0}, "dropped": {"sign_only": 0, "unknown_field": [],
                                                                            "constraints": len(table.get("constraints") or []),
                                                                            "announced": 0, "events": len(table.get("events") or [])},
           "notes": [OIL_FIX_NOTE]}
    rec["dropped"]["announced"] = len(((table.get("c_q") or {}).get("anticip")) or [])
    rows = []
    for r in rows_in:
        q = _q(r["q"])
        src = r.get("src") or {}
        x, u = {}, {}
        for f, v in (r.get("x") or {}).items():
            if v is None:
                continue
            if f == "x_oil_cause":
                cause = OIL_CAUSE_FIX.get(float(v))
                if cause is None:
                    rec["dropped"]["unknown_field"].append(f"{f}={v}")
                    continue
                x[f] = {"value": cause, "src": "stated" if src.get(f) == "stated" else "stated_qual", "quote": ""}
                rec["oil_cause"][cause] += 1
                continue
            if known is not None and f not in known and not f.startswith("u_"):
                rec["dropped"]["unknown_field"].append(f)
                continue
            s = SRC_MAP.get(src.get(f), "derived")
            rec["src"][src.get(f) or "none"] = rec["src"].get(src.get(f) or "none", 0) + 1
            if src.get(f) == "inferred":
                rec["inferred"] += 1
            entry = {"value": float(v), "src": s, "quote": ""}
            if f.startswith("u_"):
                u[f] = entry
            else:
                x[f] = entry
            rec["values"] += 1
        rec["dropped"]["sign_only"] += len(r.get("sign") or {})
        row = {"q": q, "x": dict(sorted(x.items())), "fc": {"stance": "未说明"}}
        if u:
            row["u"] = dict(sorted(u.items()))
        rows.append(row)
    rows.sort(key=lambda r: r["q"])
    qt = {"schema": "qtable/1", "model_card": "card/1 (schema-revise-2 3.0 + SIM_DESIGN)", "n_quarters": H,
          "rows": rows, "c_q": {}, "events": [], "notes": [STANDIN_NOTE]}
    rec["dropped"]["unknown_field"] = sorted(set(rec["dropped"]["unknown_field"]))
    rec["src"] = dict(sorted(rec["src"].items()))
    if rec["dropped"]["sign_only"]:
        rec["notes"].append(f"研究轮只写方向、没有幅度的条目 {rec['dropped']['sign_only']} 个没有导入：QTable/1 的方向条目必须带幅度档，"
                            "研究轮的季度模型也没有用它们")
    if rec["inferred"]:
        rec["notes"].append(f"{rec['inferred']} 个数值是翻译器把定性文字换成的数字（研究轮的 inferred），按 derived 导入：它们不是文字给出的")
    return {"qtable": qt, "conversion": rec}


# ================================================================ the published check values (logs of the critic)

_NUM = r"([+\-−]?\d+\.\d+)"
_PAIRED = re.compile(rf"^(1Q|2Q|4Q|8Q|end|endc) eq9 (?:excl E03 )?([A-Za-z]+-[a-z0-9]+) ?- ?([A-Za-z]+-[a-z0-9]+): "
                     rf"{_NUM} \[{_NUM},{_NUM}\](?: n=(\d+))?(?: (?:wins for Q|Q wins): (\d+)/(\d+))?")
_SKILL_C01 = re.compile(rf"^MSE skill (end) eq9 ([A-Za-z]+-[a-z0-9]+) vs ([A-Za-z]+-[a-z0-9]+): {_NUM} \[{_NUM},{_NUM}\]")
_SKILL_SUB = re.compile(rf"^(?:(4Q|end) )?excl (\[[^\]]*\]): MSE skill ([A-Za-z]+-[a-z0-9]+) vs ([A-Za-z]+-[a-z0-9]+) "
                        rf"{_NUM} \[{_NUM},{_NUM}\] n_ep=(\d+)")
_CHANCE = re.compile(r"paired narrative-vs-no-narrative tests: (\d+)\s+intervals excluding 0: (\[.*\])")


def _num(s: str) -> float:
    return float(s.replace("−", "-"))


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def parse_critic_logs(work: Path) -> list[dict]:
    """Every paired difference, MSE skill and chance count the critic printed (c01, c17, c21, c22, c23)."""
    out = []
    for name in ("c01", "c21", "c22", "c23"):
        path = Path(work) / f"{name}.log"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            m = _PAIRED.match(line)
            if m:
                h, a, b, v, lo, hi = m.group(1, 2, 3, 4, 5, 6)
                excl = ["E06", "E03"] if "excl E03" in line else ["E06"]
                out.append({"source": f"critic/{name}", "line": line, "stat": "paired_mae", "horizon": h, "a": a,
                            "b": b, "excluded": excl, "value": _num(v), "lo": _num(lo), "hi": _num(hi),
                            "decimals": _decimals(v), "n": int(m.group(7)) if m.group(7) else None,
                            "a_wins": int(m.group(8)) if m.group(8) else None})
                continue
            m = _SKILL_C01.match(line)
            if m:
                h, a, b, v, lo, hi = m.groups()
                out.append({"source": f"critic/{name}", "line": line, "stat": "mse_skill", "horizon": h, "a": a,
                            "b": b, "excluded": ["E06"], "value": _num(v), "lo": _num(lo), "hi": _num(hi),
                            "decimals": _decimals(v), "n": None})
                continue
            m = _SKILL_SUB.match(line)
            if m:
                h, excl, a, b, v, lo, hi, n = m.groups()
                out.append({"source": f"critic/{name}", "line": line, "stat": "mse_skill", "horizon": h or "end",
                            "a": a, "b": b, "excluded": json.loads(excl.replace("'", '"')), "value": _num(v),
                            "lo": _num(lo), "hi": _num(hi), "decimals": _decimals(v), "n": int(n)})
    c17 = Path(work) / "c17.log"
    if c17.exists():
        for line in c17.read_text(encoding="utf-8").splitlines():
            m = _CHANCE.search(line)
            if m:
                sig = re.findall(r"\('([^']+)', '([^']+)', np\.float64\(([^)]+)\), np\.float64\(([^)]+)\), "
                                 r"np\.float64\(([^)]+)\), (\d+)\)", m.group(2))
                items = [{"horizon": h, "variant": v, "diff": float(d), "lo": float(lo), "hi": float(hi), "n": int(n)}
                         for h, v, d, lo, hi, n in sig]
                out.append({"source": "critic/c17", "line": line[:300], "stat": "chance", "tests": int(m.group(1)),
                            "significant": items, "favourable": sum(1 for s in items if s["hi"] < 0),
                            "unfavourable": sum(1 for s in items if s["lo"] > 0)})
    return out


# ================================================================ the archive

def _dump(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def _gzip(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(data)
    return buf.getvalue()


def build_archive(src: Path, x_fields=None) -> dict[str, bytes]:
    """{file name: bytes} of the three archive files from the research folder (sim_redesign/)."""
    src = Path(src)
    narr, scorer, critic = src / "narratives", src / "scorer", src / "critic"
    if not (narr / "E01" / "q_n1.json").exists() or not (scorer / "out" / "score_rows.csv").exists():
        raise FileNotFoundError(f"不是研究轮的 sim_redesign 文件夹（缺 narratives/E01/q_n1.json 或 scorer/out/score_rows.csv）：{src}")
    if x_fields is None:
        from ...qmodel import card as qcard
        x_fields = qcard.x_fields(qcard.load_card())
    episodes = {}
    for code, (s, e, mode, label) in WINDOWS.items():
        d = narr / code
        ep = {"window_months": [s, e], "mode": mode, "label": label, "narratives": {}, "standin": {}}
        for n in C.NARRATIVES:
            ep["narratives"][n] = (d / f"{n}.md").read_text(encoding="utf-8")
            ep["standin"][n] = convert_standin(json.loads((d / f"q_{n}.json").read_text(encoding="utf-8")), x_fields)
        g = d / "guess.json"
        if g.exists():
            guess = json.loads(g.read_text(encoding="utf-8"))
            ep["guess"] = {k: guess.get(k) for k in ("guess", "confidence", "confidence_from_narrative_only",
                                                       "basis_narrative", "n3_blind")}
        episodes[code] = ep
    arch = {"schema": SCHEMA, "id": RESEARCH_ID, "label": C.LABELS["research"],
            "source": {"narratives": "sim_redesign/narratives/E01–E12（n1–n3.md、q_n1–q_n3.json、guess.json）",
                       "scorer": "sim_redesign/scorer（out/score_rows.csv、out/score_pooled.csv）",
                       "critic": "sim_redesign/critic/work（c01、c17、c21、c22、c23 的输出）"},
            "episodes": episodes}
    rows = pd.read_csv(scorer / "out" / "score_rows.csv")
    rows = rows[ROW_COLUMNS]
    buf = io.StringIO()
    rows.to_csv(buf, index=False, float_format="%.10g", lineterminator="\n")
    pooled = pd.read_csv(scorer / "out" / "score_pooled.csv")
    pooled = pooled[pooled["assets"].isin(["equity9", "us_equity", "all13", "gold", "crude", "rates"])]
    pub = {"schema": SCHEMA, "scorer_pooled": {"source": "scorer/out/score_pooled.csv（11 个锚定片段）",
                                               "episodes": list(ANCHORED),
                                               "rows": json.loads(pooled.to_json(orient="records", double_precision=10))},
           "critic": parse_critic_logs(critic / "work")}
    return {EPISODES_FILE: _dump(arch), ROWS_FILE: _gzip(buf.getvalue().encode("utf-8")), PUBLISHED_FILE: _dump(pub)}


def import_standin(src: Path, *, check: bool = False, out_dir: Path | None = None) -> dict:
    """Write the archive (or, with check, compare a rebuild with the committed files). Returns a summary."""
    out_dir = Path(out_dir or C.RESEARCH_DIR)
    files = build_archive(src)
    summary = {"dir": str(out_dir), "files": {}, "ok": True}
    for name, data in files.items():
        path = out_dir / name
        if check:
            if name == ROWS_FILE:          # compare the decompressed text (gzip headers are deterministic anyway)
                same = path.exists() and gzip.decompress(path.read_bytes()) == gzip.decompress(data)
            else:
                same = path.exists() and path.read_bytes() == data
            summary["files"][name] = "same" if same else "differs"
            summary["ok"] = summary["ok"] and same
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            summary["files"][name] = len(data)
    if not check:
        load_archive.cache_clear()
        load_rows.cache_clear()
    return summary


@lru_cache(maxsize=1)
def load_archive() -> dict:
    return json.loads((C.RESEARCH_DIR / EPISODES_FILE).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_published() -> dict:
    return json.loads((C.RESEARCH_DIR / PUBLISHED_FILE).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_rows() -> pd.DataFrame:
    """The scorer's stored cells, with the app's columns added (engine, arm, cma, set, bust)."""
    R = pd.read_csv(C.RESEARCH_DIR / ROWS_FILE, compression="gzip")
    R["engine"] = R["variant"].map(lambda v: VARIANTS.get(v, ("?", "?"))[0])
    R["arm"] = R["variant"].map(lambda v: VARIANTS.get(v, ("?", "?"))[1])
    R["cma"] = "historical"
    R["set"] = np.where(R["code"].isin(CLEAN), "clean", "short")
    R["bust"] = R["code"].isin(BUSTS)
    R["quarters"] = (R["months"] // 3).astype(int)
    return R


def available() -> bool:
    return all((C.RESEARCH_DIR / f).exists() for f in (EPISODES_FILE, ROWS_FILE, PUBLISHED_FILE))


# ================================================================ recomputation: the scorer's rule and exact critic ports

def scorer_pooled(R: pd.DataFrame | None = None, codes=ANCHORED) -> pd.DataFrame:
    """The scorer's pooled point estimates (mae, rmse, bias, cov80, pit_mad, crps) per variant × horizon × set on
    `codes`: the mean over episodes of each episode's mean over the set's assets."""
    R = load_rows() if R is None else R
    sets = {"all13": C.ASSETS, "equity9": EQ9, "us_equity": ["us_equity"], "gold": ["gold"], "crude": ["crude_oil"],
            "rates": ["treasury_10y", "cash"]}
    sub = R[R["code"].isin(codes)]
    out = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # degenerate bands have no PIT: an all-nan mean
        groups = list(sub.groupby(["variant", "horizon"]))
    for (v, h), g in groups:
        for sname, al in sets.items():
            gg = g[g["asset"].isin(al)]
            if gg.empty:
                continue
            per = gg.groupby("code")
            out.append({"variant": v, "horizon": h, "assets": sname, "n_episodes": int(per.ngroups),
                        "mae": float(per["abs_err"].mean().mean()),
                        "rmse": float(np.sqrt(np.nanmean(per["err"].apply(lambda s: (s ** 2).mean())))),
                        "bias": float(np.nanmean(per["err"].mean())), "cov80": float(per["cov80"].mean().mean()),
                        "pit_mad": _nanmean(per["pit"].apply(lambda s: (s - 0.5).abs().mean())),
                        "crps": float(np.nanmean(per["crps"].mean()))})
    return pd.DataFrame(out)


def _nanmean(s) -> float:
    a = np.asarray(s, float)
    return float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else float("nan")


def _boot_np(rng, d, B):
    d = np.asarray(d)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def _skill_boot(rng, e: pd.DataFrame, v: str, base: str, draws: int) -> tuple[float, float, float]:
    """The critic's MSE-skill bootstrap: each draw resamples episodes (rng.choice over the codes) and sums the squared
    errors of the drawn episodes' cells. Summing per-episode sums is the same number as summing the concatenated
    frames the critic built, and choosing indices draws the same stream as choosing the codes."""
    sk = 1 - (e[v] ** 2).sum() / (e[base] ** 2).sum()
    codes = e.index.get_level_values(0).unique()
    sv = np.array([float((e.loc[c, v] ** 2).sum()) for c in codes])
    sb = np.array([float((e.loc[c, base] ** 2).sum()) for c in codes])
    pick = np.arange(len(codes))
    bs = np.empty(draws)
    for k in range(draws):
        i = rng.choice(pick, len(codes))
        bs[k] = 1 - sv[i].sum() / sb[i].sum()
    return float(sk), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


def critic_c01(R: pd.DataFrame) -> list[dict]:
    """sim_redesign/critic/c01_scoreboard.py: the paired end differences, MSE skills and 1Q differences (rng 0)."""
    rng = np.random.default_rng(0)
    anch = [c for c in sorted(R.code.unique()) if c != "E06"]
    r = R[R.code.isin(anch) & R.asset.isin(EQ9)]
    ep = r[r.horizon == "end"].groupby(["variant", "code"]).abs_err.mean().unstack()
    out = []
    for base, others in [("E-uncond", ["E-cond", "E-oracle", "E-n1", "E-n2", "E-n3", "F-oracle"]),
                         ("Q-uncond", ["Q-oracle", "Q-n1", "Q-n2", "Q-n3"]),
                         ("E-uncond", ["Q-uncond", "Q-n1", "Q-n2", "Q-n3", "Q-oracle", "VB-oracle"])]:
        for o in others:
            if o not in ep.index:
                continue
            d = (ep.loc[o] - ep.loc[base]).dropna()
            m, lo, hi = _boot_np(rng, d.values, 20000)
            out.append({"source": "critic/c01", "stat": "paired_mae", "horizon": "end", "a": o, "b": base,
                        "excluded": ["E06"], "value": m, "lo": lo, "hi": hi, "n": len(d)})
    for a, b in [("Q-n1", "E-n1"), ("Q-n2", "E-n2"), ("Q-n3", "E-n3"), ("Q-oracle", "E-oracle")]:
        d = (ep.loc[a] - ep.loc[b]).dropna()
        m, lo, hi = _boot_np(rng, d.values, 20000)
        out.append({"source": "critic/c01", "stat": "paired_mae", "horizon": "end", "a": a, "b": b,
                    "excluded": ["E06"], "value": m, "lo": lo, "hi": hi, "n": len(d), "a_wins": int((d < 0).sum())})
    e = r[r.horizon == "end"].pivot_table(index=["code", "asset"], columns="variant", values="err")
    for v in ["E-oracle", "E-cond", "E-n1", "E-n2", "E-n3", "Q-oracle", "Q-n1", "F-oracle", "VB-oracle"]:
        if v in e.columns:
            sk, lo, hi = _skill_boot(rng, e, v, "E-uncond", 5000)
            out.append({"source": "critic/c01", "stat": "mse_skill", "horizon": "end", "a": v, "b": "E-uncond",
                        "excluded": ["E06"], "value": sk, "lo": lo, "hi": hi, "n": None})
    ep1 = r[r.horizon == "1Q"].groupby(["variant", "code"]).abs_err.mean().unstack()
    for o in ["E-n1", "E-n2", "E-n3", "E-cond", "E-oracle"]:
        d = (ep1.loc[o] - ep1.loc["E-uncond"]).dropna()
        m, lo, hi = _boot_np(rng, d.values, 20000)
        out.append({"source": "critic/c01", "stat": "paired_mae", "horizon": "1Q", "a": o, "b": "E-uncond",
                    "excluded": ["E06"], "value": m, "lo": lo, "hi": hi, "n": len(d)})
    return out


def critic_c17(R: pd.DataFrame) -> dict:
    """sim_redesign/critic/c17_n2_1q.py: the 30 narrative-versus-no-narrative intervals counted (rng 3)."""
    R = R[R.code != "E06"]
    rng = np.random.default_rng(3)
    r1 = R[R.horizon == "1Q"]
    for a in EQ9 + ["gold", "treasury_10y", "crude_oil"]:
        x = r1[r1.asset == a].pivot_table(index="code", columns="variant", values="abs_err")
        _boot_np(rng, (x["E-n2"] - x["E-uncond"]).values, 20000)
        _boot_np(rng, (x["E-n1"] - x["E-uncond"]).values, 20000)
    tests, sig = 0, []
    for h in ["1Q", "2Q", "4Q", "8Q", "end"]:
        rh = R[(R.horizon == h) & R.asset.isin(EQ9)]
        ep = rh.groupby(["variant", "code"]).abs_err.mean().unstack()
        for base, vs in (("E-uncond", ["E-n1", "E-n2", "E-n3"]), ("Q-uncond", ["Q-n1", "Q-n2", "Q-n3"])):
            for v in vs:
                d = (ep.loc[v] - ep.loc[base]).dropna()
                if len(d) < 5:
                    continue
                m, lo, hi = _boot_np(rng, d.values, 20000)
                tests += 1
                if lo > 0 or hi < 0:
                    sig.append({"horizon": h, "variant": v, "diff": m, "lo": lo, "hi": hi, "n": len(d)})
    return {"stat": "chance", "tests": tests, "significant": sig, "favourable": sum(1 for s in sig if s["hi"] < 0),
            "unfavourable": sum(1 for s in sig if s["lo"] > 0)}


def critic_c21(R: pd.DataFrame) -> list[dict]:
    """sim_redesign/critic/c21_endc.py: Q − E at the common end; differences without E03 (rng 5)."""
    R = R[R.code != "E06"]
    rng = np.random.default_rng(5)
    out = []
    r = R[(R.horizon == "endc") & R.asset.isin(EQ9)]
    ep = r.groupby(["variant", "code"]).abs_err.mean().unstack()
    for a, b in [("Q-n1", "E-n1"), ("Q-n2", "E-n2"), ("Q-n3", "E-n3"), ("Q-oracle", "E-oracle")]:
        d = (ep.loc[a] - ep.loc[b]).dropna()
        m, lo, hi = _boot_np(rng, d.values, 20000)
        out.append({"source": "critic/c21", "stat": "paired_mae", "horizon": "endc", "a": a, "b": b,
                    "excluded": ["E06"], "value": m, "lo": lo, "hi": hi, "n": len(d), "a_wins": int((d < 0).sum())})
    r2 = R[(R.horizon == "end") & R.asset.isin(EQ9) & (R.code != "E03")]
    ep2 = r2.groupby(["variant", "code"]).abs_err.mean().unstack()
    for v in ["E-oracle", "F-oracle", "VB-oracle", "E-n1", "E-n2", "E-n3", "E-cond"]:
        d = (ep2.loc[v] - ep2.loc["E-uncond"]).dropna()
        m, lo, hi = _boot_np(rng, d.values, 20000)
        out.append({"source": "critic/c21", "stat": "paired_mae", "horizon": "end", "a": v, "b": "E-uncond",
                    "excluded": ["E06", "E03"], "value": m, "lo": lo, "hi": hi, "n": len(d)})
    return out


def critic_c22(R: pd.DataFrame) -> list[dict]:
    """sim_redesign/critic/c22_mse_exE03.py: MSE skill of the oracles without E03 and without the busts (rng 9)."""
    rng = np.random.default_rng(9)
    out = []
    for excl in (["E06"], ["E06", "E03"], ["E06", "E03", "E05", "E12"]):
        r = R[(R.horizon == "end") & R.asset.isin(EQ9) & ~R.code.isin(excl)]
        e = r.pivot_table(index=["code", "asset"], columns="variant", values="err")
        n = len(e.index.get_level_values(0).unique())
        for v in ["E-oracle", "F-oracle", "VB-oracle", "Q-oracle", "E-cond"]:
            sk, lo, hi = _skill_boot(rng, e, v, "E-uncond", 4000)
            out.append({"source": "critic/c22", "stat": "mse_skill", "horizon": "end", "a": v, "b": "E-uncond",
                        "excluded": excl, "value": sk, "lo": lo, "hi": hi, "n": n})
    return out


def critic_c23(R: pd.DataFrame) -> list[dict]:
    """sim_redesign/critic/c23_narr_subsets.py: the narratives' MSE skill in the 8 non-bust episodes (rng 11)."""
    rng = np.random.default_rng(11)
    out = []
    excl = ["E06", "E03", "E05", "E12"]
    for h in ("4Q", "end"):
        r = R[(R.horizon == h) & R.asset.isin(EQ9) & ~R.code.isin(excl)]
        e = r.pivot_table(index=["code", "asset"], columns="variant", values="err")
        n = len(e.index.get_level_values(0).unique())
        for v in ["E-n1", "E-n2", "E-n3", "Q-n1", "Q-n2", "Q-n3", "E-oracle", "Q-oracle"]:
            sk, lo, hi = _skill_boot(rng, e, v, "E-uncond", 3000)
            out.append({"source": "critic/c23", "stat": "mse_skill", "horizon": h, "a": v, "b": "E-uncond",
                        "excluded": excl, "value": sk, "lo": lo, "hi": hi, "n": n})
    return out


def recompute(R: pd.DataFrame | None = None, *, intervals: bool = True) -> dict:
    """{"scorer_pooled": DataFrame, "critic": [...], "chance": {...}} recomputed from the stored rows. intervals=False
    skips the critic's bootstrap ports (the points are then computed directly; a few seconds faster)."""
    R = load_rows() if R is None else R
    R = R[ROW_COLUMNS] if set(ROW_COLUMNS) <= set(R.columns) else R
    out = {"scorer_pooled": scorer_pooled(R)}
    if intervals:
        out["critic"] = critic_c01(R) + critic_c21(R) + critic_c22(R) + critic_c23(R)
        out["chance"] = critic_c17(R)
    else:
        out["critic"] = [dict(p, value=point(R, p), lo=None, hi=None) for p in load_published()["critic"]
                         if p["stat"] != "chance"]
        out["chance"] = None
    return out


def point(R: pd.DataFrame, p: dict) -> float:
    """The point value of one published critic statistic, computed directly from the rows."""
    r = R[(R.horizon == p["horizon"]) & R.asset.isin(EQ9) & ~R.code.isin(p["excluded"])]
    if p["stat"] == "paired_mae":
        ep = r.groupby(["variant", "code"]).abs_err.mean().unstack()
        return float((ep.loc[p["a"]] - ep.loc[p["b"]]).dropna().mean())
    e = r.pivot_table(index=["code", "asset"], columns="variant", values="err")
    return float(1 - (e[p["a"]] ** 2).sum() / (e[p["b"]] ** 2).sum())


def _key(p: dict) -> tuple:
    return (p.get("source"), p["stat"], p["horizon"], p["a"], p["b"], tuple(p["excluded"]))


def checks(rec: dict | None = None, *, mae_tol: float = 0.002) -> dict:
    """Every published value against its recomputation. Scorer: mae (± mae_tol), rmse, bias, cov80, pit_mad and crps
    (± 1e-6: the CSV is full precision). Critic: points to ± half a unit of the printed precision (+ 1e-9), intervals
    reported with their largest deviation. {"ok", "n", "failed", "max_dev", "items"}."""
    rec = rec or recompute()
    pub = load_published()
    items, failed = [], []
    P = rec["scorer_pooled"].set_index(["variant", "horizon", "assets"])
    for row in pub["scorer_pooled"]["rows"]:
        k = (row["variant"], row["horizon"], row["assets"])
        if k not in P.index:
            failed.append({"what": "scorer", "key": list(k), "why": "没有重算出来"})
            continue
        for col, tol in (("mae", mae_tol), ("rmse", 1e-6), ("bias", 1e-6), ("cov80", 1e-6), ("pit_mad", 1e-6),
                         ("crps", 1e-6)):
            pv, rv = row.get(col), P.at[k, col]
            if pv is None or (isinstance(pv, float) and not np.isfinite(pv)):
                continue
            dev = abs(float(pv) - float(rv))
            it = {"what": f"scorer.{col}", "key": list(k), "published": pv, "recomputed": rv, "dev": dev, "tol": tol,
                  "ok": dev <= tol}
            items.append(it)
            if not it["ok"]:
                failed.append(it)
    mine = {_key(p): p for p in rec["critic"]}
    max_iv = 0.0
    for p in pub["critic"]:
        if p["stat"] == "chance":
            ch = rec.get("chance")
            if ch is not None:
                ok = (ch["tests"], ch["favourable"], ch["unfavourable"]) == (p["tests"], p["favourable"],
                                                                             p["unfavourable"])
                it = {"what": "critic.chance", "key": p["source"], "published": [p["tests"], p["favourable"],
                                                                                  p["unfavourable"]],
                      "recomputed": [ch["tests"], ch["favourable"], ch["unfavourable"]], "ok": ok}
                items.append(it)
                if not ok:
                    failed.append(it)
            continue
        r = mine.get(_key(p))
        if r is None:
            failed.append({"what": "critic", "key": p["line"], "why": "没有重算出来"})
            continue
        tol = 0.5 * 10 ** (-p["decimals"]) + 1e-9
        dev = abs(p["value"] - r["value"])
        it = {"what": "critic.point", "key": p["line"], "published": p["value"], "recomputed": r["value"], "dev": dev,
              "tol": tol, "ok": dev <= tol}
        if r.get("lo") is not None and p.get("lo") is not None:
            iv = max(abs(p["lo"] - r["lo"]), abs(p["hi"] - r["hi"]))
            it["interval_dev"] = iv
            max_iv = max(max_iv, iv)
        items.append(it)
        if not it["ok"]:
            failed.append(it)
    return {"ok": not failed, "n": len(items), "failed": failed,
            "max_dev": max([it.get("dev", 0.0) for it in items] or [0.0]), "max_interval_dev": max_iv,
            "items": items}


def corrections() -> list[dict]:
    """The critic's corrections the evaluation tab applies to the research run (Chinese, for the run's notes)."""
    return [
        {"id": "E06", "text": "E06 用窗口以外的全部月份拟合（前视），单独报告，不进合并统计（研究轮已这样做）"},
        {"id": "E03", "text": "E03 只有 27 个训练季度（80 个月度样本），按片段的训练期规则移到训练期短的一组：干净集是 10 个片段，"
                              "原来的 11 个片段的数字照旧保留"},
        {"id": "busts", "text": "大跌片段（窗口内最大回撤 ≤ −20%：E05、E08、E10、E12）单独作为预注册的次要分层；"
                                "去掉 E03、E05、E12 的 8 个片段是事后挑出的子集"},
        {"id": "chance", "text": "30 个叙事对无叙事的成对区间里，1 个有利、3 个不利：与偶然相当"},
        {"id": "mde", "text": "11 个片段时，小于约 0.02–0.03 log 的股票类 MAE 差异检不出来"},
        {"id": "blind", "text": "翻译者 12 个片段全部认出了年代：识别出年代的回放（不是盲测）"},
        {"id": "engine", "text": "月度引擎跑的是 B0 模型（不是因子轮 F），长期锚用的是 2008 年以后才知道的固定值（前视）"},
        {"id": "qmodel", "text": "“Q”是研究轮的季度模型 G，不是现在的季度模型"},
    ]
