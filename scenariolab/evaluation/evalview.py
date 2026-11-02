"""情景条件检验 (the evaluation tab, SIM_DESIGN §10.1): one read-only view of the pre-registered records and the
computed checks of the quarterly stack. Nothing here writes a record, runs a test or calls an LLM.

    GET /api/evaluation                 summary() below
    GET /api/evaluation/records/{rid}   one frozen record in full: its JSON, sha256 and file

summary() -> {
  schema "evaluation-view/1", label 试运行, created, panel (whether the quarterly panel exists: section D needs it)
  records       G. every frozen record found in PREREG_DIRS (scenariolab/qmodel/preregistered for P5's T-A / T-C /
                T-S, scenariolab/evaluation/preregistered, scenariolab/evaluation/narrative/preregistered for T-N):
                id (file stem), test, test_zh, date, file, sha256, version, title, frozen, selection (what was selected
                on which data), candidates, thresholds, seeds, hashes, n_results, last_result {date, status, passed,
                text}, status (registered | done | unreadable)
  tests         {TA, TC, TS, TN: {status: missing | registered | done | unreadable, record, zh, what}}
  asset_layer   B. T-A's latest result, normalized: {date, asset_layer, adopted, rows [{model, model_zh, horizon, r2,
                lo90, hi90, crpss, cov80, n_indep, report_only, anecdote}], conditions, text}; None without results
  conditional   B. the evaluation step's conditional-path tables (results/evaluation.json "conditional"), or None
  ablation      B. the research round's oracle ablation (research/oracle_ablation.json, committed), or None
  anatomy       D. predicted against realized multiple with the placebo row (results/evaluation.json "anatomy")
  factor_layer  E. qmodel.json's computed T-F and T-C (pass / fail shown, not frozen: the records are P5's)
  delivery      E. qtable_delivery.json: the built-in QTables' views delivered on the quarterly stack
  switch_gate   A. T-S as the switch reads it (evaluation/switch.py state(): the newest valid evaluation, from the
                record's appended results or results/switch_gate.json, else not evaluated with the reasons):
                {status, passed, text, conditions [{id, text, passed, detail}], reasons, source, default_text}
  ledger        F. the forward ledger's state (forecast_mix.json "ledger")
  narrative     C. {headline_run, headline_kind (real | research | None), n_real, n_mock}: the run section A quotes
                (the newest real run with a report, else the research run; a mock run never)
  conclusion    A. {parts [{key, kind, text, href}]}: everything but the narrative test; the page adds the headline
                run's own headline (narrative/report.py builds it from the pre-registered claims only)
  contamination_notes, notes
}

Every text is Chinese and every number comes from a file. A record whose shape this module does not know still
shows in G with its raw JSON (the page prints it under 原始记录); the normalizers read the keys the spec names
(§4.2, §4.6, §5.8, §14 P5) and their usual aliases.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import config, jsonio
from ..qmodel import config_q as cq
from .switch import not_evaluated_text as _switch_not_evaluated

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

PACKAGE_DIR = Path(__file__).resolve().parent
# where frozen pre-registration records live (read at call time, so a test can point them at a fixture folder)
PREREG_DIRS = [cq.QMODEL_DIR / "preregistered", PACKAGE_DIR / "preregistered", PACKAGE_DIR / "narrative" / "preregistered"]
RESULTS_DIR = config.RESULTS_DIR
ABLATION_PATH = PACKAGE_DIR / "research" / "oracle_ablation.json"
SCHEMA = "evaluation-view/1"
RECORD_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_([A-Za-z0-9-]+?)(?:\.results)?$")
RID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")

TESTS = {
    "TA": {"zh": "资产层候选检验（T-A）", "what": "资产层在默认的 L0 之外是否加入候选（字段、状态、交互、领先项、神经网络）"},
    "TC": {"zh": "补全检验（T-C）", "what": "文字没说的金融市场字段怎样补全：收缩权重与体制协方差"},
    "TS": {"zh": "切换关（T-S）", "what": "季度模型何时取代月度引擎，成为情景、一年期页、组合与提取的默认"},
    "TF": {"zh": "因子引擎检验（T-F）", "what": "因子层的模拟能否复现历史的通胀、衰退、联动与边界"},
    "TN": {"zh": "叙事检验（T-N）", "what": "叙事检验的片段、臂、统计与可以说的话"},
}
MAIN_TESTS = ("TA", "TC", "TS", "TN")
HORIZON_Q = {"1": "1Q", "1q": "1Q", "1Q": "1Q", "2Q": "2Q", "4": "4Q", "4q": "4Q", "4Q": "4Q", "8Q": "8Q", "12": "12Q",
             "12q": "12Q", "12Q": "12Q", "20": "20Q", "20q": "20Q", "20Q": "20Q"}
ANECDOTE_H = ("12Q", "20Q")
MODEL_ZH = {"L0": "L0：因子轮 F（默认资产层）", "G": "G：L0 加通过选择的候选", "NN": "神经网络（只作对照）",
            "NN-res": "G 加神经网络修正（只作对照）", "NN_res": "G 加神经网络修正（只作对照）",
            "C1": "C1：F 之外的一级字段", "C2": "C2：状态与交互", "C3": "C3：下一季的工业产出与失业率",
            "C4": "C4：按行业加总的美股（只作对照）", "OA": "O-A：已知宏观", "OAp": "O-A′：加已知商品价格",
            "OB": "O-B：加已知金融市场", "OF": "O-full：全部字段已知", "placebo": "安慰剂（常数）",
            "drift": "只有历史漂移", "E": "月度引擎", "Q": "季度模型"}
REPORT_ONLY = {"NN", "NN-res", "NN_res", "C4"}
CONTAMINATION_NOTES = [
    "2015–2026 年已经被之前每一轮看过：因子轮用它的检验期做了选择，字段分层用了它，研究轮的季度模型也在它上面给各种变体打过分。"
    "所以 T-A 的检验期是预注册的重测，不是新的留出样本；唯一干净的留出是从 2026 年第 4 季开始的前瞻账本。",
    "因子轮的 F 以 1995–2004 年为验证期选入 GZ 信用利差：那个十年对 F 不是干净的检验。",
    "字段分层（一级、二级）用三个十年全部的数据选出：候选的证据不是独立的。",
    "长期锚：回测与检验用训练期末可知的规则；研究轮的引擎用的是 2008 年以后才知道的固定值（前视），它的数字只作对照。",
    "叙事检验：翻译者知道历史，识别率每次都量；识别率 100% 时结果是“识别出年代的回放（不是盲测）”。",
]
TF_STATS = {
    "F2_cpi_gt4": "CPI 同比 > 4% 的季度占比",
    "F2_cpi_acf4": "CPI 同比的 4 季自相关（不重叠的 13 年窗口）",
    "F2_spell": "高通胀持续期的平均长度（季）",
    "F3_per_15y": "每 15 年的衰退次数",
    "F3_mean_q": "衰退的平均长度（季）",
    "F3_gt6q": "长于 6 个季度的衰退占比（规则：≤ 25%）",
    "F4_bounce": "反弹：4 季工业产出增长对 8 季高点回撤的斜率",
    "F5_dff_on_du": "Δ政策利率对 Δ失业率的斜率",
    "F5_okun": "奥肯系数：Δ失业率对工业产出增长",
    "F5_corr_dgz_useq": "GZ 信用利差变动与美股季度回报的相关",
    "F5_corr_dspread_useq": "Baa 利差变动与美股季度回报的相关",
    "F5_stock_bond_core4_0": "股债相关（核心通胀 ≤ 4%）",
    "F5_stock_bond_core4_1": "股债相关（核心通胀 > 4%）",
    "F6_bounds": "边界：政策利率 ≥ 0.1；失业率 ≥ 2.5，低于 3% 的季度 ≤ 5%；CPI 同比 > 15% ≤ 0.5%",
}
TF_SHORT = {"F2_cpi_gt4": "高通胀占比", "F2_cpi_acf4": "CPI 同比的自相关", "F2_spell": "高通胀持续期",
            "F3_per_15y": "衰退次数", "F3_mean_q": "衰退长度", "F3_gt6q": "长衰退占比", "F4_bounce": "反弹",
            "F5_dff_on_du": "政策利率对失业率", "F5_okun": "奥肯系数", "F5_corr_dgz_useq": "GZ 利差与美股的相关",
            "F5_corr_dspread_useq": "Baa 利差与美股的相关", "F5_stock_bond_core4_0": "低通胀时的股债相关",
            "F5_stock_bond_core4_1": "高通胀时的股债相关", "F6_bounds": "边界", "F1": "交付", "F7": "补全技能",
            "F8": "覆盖率", "F9": "干预（do 算子）", "F10": "逐季与整段一致"}
TF_GROUPS = [("F2", "通胀"), ("F3", "衰退"), ("F4", "反弹"), ("F5", "联动"), ("F6", "边界")]
F7_FIELD = {"ps": "d_cpi_serv", "ff": "d_ff", "y10": "d_y10", "sp": "d_spread", "gz": "d_gz", "ur": "d_unrate",
            "usd": "d_usd"}
TC_PATTERN = {"a": "随机缺一半字段", "b": "只给宏观与政策（像叙事那样）", "c": "只给金融条件"}
ENDO_ZH = {"E0": "外生宏观", "E0p": "政策", "E1c": "商品价格", "E1": "金融市场", "E2": "调查与新闻"}
ANCHORS_ZH = {"train_end": "训练期末可知的锚", "forward": "前瞻判断值"}
SWITCH_DEFAULT = _switch_not_evaluated()     # the switch's own text (evaluation/switch.py, P9)


def _json(obj, status: int = 200) -> Response:
    return Response(content=jsonio.dumps(obj), media_type="application/json", status_code=status)


def _load(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x else None


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(config.ROOT))
    except ValueError:
        return str(path)


# ================================================================ records

def _test_id(data: dict, stem: str) -> str:
    raw = str((data or {}).get("test") or (data or {}).get("id") or "")
    m = RECORD_RE.match(stem)
    for cand in (raw, m.group(2) if m else stem.rsplit("_", 1)[-1]):
        t = re.sub(r"[^A-Za-z]", "", cand).upper()
        if t in TESTS:
            return t
    return "other"


def _results(data: dict, path: Path) -> list:
    """The appended results of a record: its own `results` list (or `result` dict), plus a sibling append-only file
    (<stem>.results.json / .jsonl) when the round keeps them apart."""
    out = []
    res = data.get("results") if isinstance(data, dict) else None
    if isinstance(res, list):
        out += [r for r in res if isinstance(r, dict)]
    elif isinstance(res, dict):
        out.append(res)
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        out.append(data["result"])
    for sib in (path.with_name(f"{path.stem}.results.json"), path.with_name(f"{path.stem}.results.jsonl")):
        if not sib.exists():
            continue
        try:
            text = sib.read_text(encoding="utf-8")
        except OSError:
            continue
        if sib.suffix == ".jsonl":
            for line in text.splitlines():
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        else:
            try:
                obj = json.loads(text)
            except ValueError:
                obj = None
            if isinstance(obj, list):
                out += [r for r in obj if isinstance(r, dict)]
            elif isinstance(obj, dict):
                out += [r for r in obj.get("results", [obj]) if isinstance(r, dict)]
    return out


def _passed(res: dict):
    for k in ("passed", "pass", "ok", "adopt", "adopted_any"):
        v = res.get(k)
        if isinstance(v, bool):
            return v
    st = str(res.get("status") or res.get("verdict") or "").lower()
    if st in ("passed", "pass", "adopted"):
        return True
    if st in ("failed", "fail", "rejected", "not_adopted"):
        return False
    return None


def _text(obj, *keys) -> str | None:
    for k in keys:
        v = obj.get(k) if isinstance(obj, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _selection_text(data: dict) -> str | None:
    """What was selected on which data, from the record's protocol (strings, or the usual span / train-end keys)."""
    proto = data.get("protocol")
    if isinstance(proto, str):
        return proto
    parts = []
    for key, zh in (("selection", "选择"), ("validation", "验证"), ("test", "检验")):
        v = (proto or {}).get(key) if isinstance(proto, dict) else None
        if v is None:
            v = data.get(key) if key != "test" else None
        if isinstance(v, str):
            parts.append(f"{zh}：{v}")
        elif isinstance(v, dict):
            span = v.get("span") or v.get("origins") or v.get("period") or v.get("decades")
            te = v.get("train_end")
            bits = []
            if isinstance(span, (list, tuple)) and span:
                bits.append("–".join(str(s) for s in span[:2]) if len(span) == 2 else "、".join(str(s) for s in span))
            elif isinstance(span, str):
                bits.append(span)
            if te:
                bits.append(f"训练到 {te}")
            txt = _text(v, "text", "zh", "rule", "description")
            if txt:
                bits.append(txt)
            if bits:
                parts.append(f"{zh}：{'；'.join(bits)}")
    if not parts:
        return _text(data, "selection_text", "what", "summary")
    return "。".join(parts)


def _record_summary(path: Path) -> dict:
    raw = path.read_bytes()
    stem = path.stem
    m = RECORD_RE.match(stem)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        data = None
    test = _test_id(data if isinstance(data, dict) else {}, stem)
    out = {"id": stem, "test": test, "test_zh": TESTS.get(test, {}).get("zh", "其他记录"),
           "file": _rel(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
           "date": (m.group(1) if m else None)}
    if not isinstance(data, dict):
        return {**out, "status": "unreadable", "error": "不是合法的 JSON 对象"}
    res = _results(data, path)
    last = res[-1] if res else None
    cands = data.get("candidates")
    if isinstance(cands, dict):
        cands = [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in cands.items()]
    out.update({
        "date": str(data.get("date") or out["date"] or "") or None, "version": data.get("version"),
        "title": _text(data, "title", "zh", "name"), "spec": _text(data, "spec"), "frozen": _text(data, "frozen"),
        "selection": _selection_text(data),
        "candidates": [{"id": str(c.get("id") or c.get("name") or i + 1), "text": _text(c, "zh", "text", "name"),
                        "report_only": bool(c.get("report_only")) or str(c.get("id")) in REPORT_ONLY}
                       for i, c in enumerate(cands or []) if isinstance(c, dict)],
        "thresholds": data.get("thresholds") if isinstance(data.get("thresholds"), (dict, list)) else None,
        "seeds": data.get("seeds") if isinstance(data.get("seeds"), (dict, list, int)) else None,
        "hashes": ({k: str(v) for k, v in data["hashes"].items()} if isinstance(data.get("hashes"), dict) else None),
        "n_results": len(res),
        "last_result": ({"date": str(last.get("date") or last.get("created") or last.get("run_at") or "") or None,
                         "status": last.get("status"), "passed": _passed(last),
                         "text": _text(last, "text", "zh", "summary", "conclusion", "decision")} if last else None),
        "status": "done" if res else "registered",
    })
    return out


def _record_files() -> list[Path]:
    seen, out = set(), []
    for d in PREREG_DIRS:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.endswith(".results.json") or p.resolve() in seen:
                continue
            seen.add(p.resolve())
            out.append(p)
    return out


def records() -> list[dict]:
    """Every frozen record, newest first within a test (tests in the order T-A, T-C, T-S, T-N, others)."""
    recs = [_record_summary(p) for p in _record_files()]
    order = {t: i for i, t in enumerate(list(MAIN_TESTS) + ["TF", "other"])}
    recs.sort(key=lambda r: (order.get(r["test"], 9), "" if r.get("date") is None else r["date"], r["id"]))
    out = []
    for t in order:                                   # newest first per test, stable test order
        out += sorted([r for r in recs if r["test"] == t], key=lambda r: (r.get("date") or "", r["id"]), reverse=True)
    return out


def record(rid: str) -> dict:
    """One record in full (its JSON as written, the appended results, sha256 and file)."""
    if not RID_RE.match(rid or ""):
        raise KeyError(f"记录编号格式不对：「{rid}」")
    for p in _record_files():
        if p.stem == rid:
            data = _load(p)
            return {**_record_summary(p), "record": data, "results": _results(data, p) if isinstance(data, dict) else []}
    raise KeyError(f"找不到预注册记录「{rid}」")


def _latest(recs: list[dict], test: str) -> dict | None:
    return next((r for r in recs if r["test"] == test), None)


def tests_status(recs: list[dict]) -> dict:
    out = {}
    for t in list(MAIN_TESTS):
        r = _latest(recs, t)
        out[t] = {"zh": TESTS[t]["zh"], "what": TESTS[t]["what"], "record": r["id"] if r else None,
                  "status": r["status"] if r else "missing", "date": r.get("date") if r else None,
                  "n_results": r.get("n_results", 0) if r else 0}
    return out


# ================================================================ normalizers (T-A, T-S)

def _first(d: dict, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def _row(model, horizon, cell: dict) -> dict | None:
    h = HORIZON_Q.get(str(horizon).strip()) if horizon is not None else None
    r2 = _num(_first(cell, "r2", "R2", "dr2", "value"))
    if h is None or (r2 is None and _num(_first(cell, "crpss", "cov80")) is None):
        return None
    m = str(model)
    return {"model": m, "model_zh": MODEL_ZH.get(m, m), "horizon": h, "r2": r2,
            "lo90": _num(_first(cell, "r2_lo90", "lo90", "lo", "r2_lo")),
            "hi90": _num(_first(cell, "r2_hi90", "hi90", "hi", "r2_hi")),
            "crpss": _num(_first(cell, "crpss", "crps_skill")), "cov80": _num(_first(cell, "cov80", "coverage80")),
            "n_indep": _first(cell, "n_indep", "n_windows", "n"), "report_only": m in REPORT_ONLY or bool(cell.get("report_only")),
            "anecdote": h in ANECDOTE_H}


def table_rows(block) -> list[dict]:
    """Model x horizon cells from a list of row dicts ({model | variant | arm | candidate, horizon | h | h_q, r2, …})
    or a nested {model: {horizon: {r2, …}}} mapping."""
    rows = []
    if isinstance(block, dict) and isinstance(block.get("rows"), list):
        block = block["rows"]
    if isinstance(block, list):
        for c in block:
            if not isinstance(c, dict):
                continue
            r = _row(_first(c, "model", "variant", "arm", "candidate", "name", "id"), _first(c, "horizon", "h", "h_q"), c)
            if r:
                rows.append(r)
    elif isinstance(block, dict):
        for model, by_h in block.items():
            if not isinstance(by_h, dict):
                continue
            for h, cell in by_h.items():
                if isinstance(cell, dict):
                    r = _row(model, h, cell)
                    if r:
                        rows.append(r)
    order = {h: i for i, h in enumerate(("1Q", "2Q", "4Q", "8Q", "12Q", "20Q"))}
    mo = {m: i for i, m in enumerate(MODEL_ZH)}
    rows.sort(key=lambda r: (mo.get(r["model"], 50), r["model"], order.get(r["horizon"], 9)))
    return rows


def conditions(block) -> list[dict]:
    """A gate's conditions ([{id, text, passed, …numbers}] or {id: {…}}), each with a one-line Chinese detail."""
    items = []
    if isinstance(block, dict):
        block = [{"id": k, **(v if isinstance(v, dict) else {"passed": v})} for k, v in block.items()]
    for i, c in enumerate(block or []):
        if not isinstance(c, dict):
            continue
        nums = []
        diff, lo, hi = _num(_first(c, "diff", "value", "delta")), _num(c.get("lo90")), _num(c.get("hi90"))
        bound = _num(_first(c, "bound", "threshold", "margin", "hi90_below", "lo90_above"))
        if diff is not None:
            nums.append(f"{diff:+.3f}" + (f" [{lo:+.3f}, {hi:+.3f}]" if lo is not None and hi is not None else ""))
        if bound is not None:
            nums.append(f"界线 {bound:+.3f}")
        if c.get("n") is not None:
            nums.append(f"n {c['n']}")
        items.append({"id": str(c.get("id") or i + 1), "text": _text(c, "text", "zh", "rule", "name") or str(c.get("id") or ""),
                      "passed": _passed(c), "detail": "，".join(nums) or None,
                      "evaluated": c.get("evaluated", True) is not False})
    return items


def asset_layer(recs: list[dict]) -> dict | None:
    """T-A's latest result as the section-B table (L0, G when adopted, NN report-only), or None without results."""
    r = _latest(recs, "TA")
    if not r or not r.get("n_results"):
        return None
    full = record(r["id"])
    res = full["results"][-1]
    test = res.get("test") if isinstance(res.get("test"), (dict, list)) else None
    block = _first(res, "rows", "table")
    if block is None:                                 # {test: {rows: [...]}} or {test: {model: {horizon: {...}}}}
        block = test.get("rows", test) if isinstance(test, dict) else test
    rows = table_rows(block or [])
    conds = conditions(_first(res, "conditions", "gate") or (test.get("conditions") if isinstance(test, dict) else None) or [])
    adopted = res.get("adopted")
    if isinstance(adopted, str):
        adopted = [adopted]
    return {"record": r["id"], "date": r["last_result"]["date"] if r.get("last_result") else None,
            "asset_layer": _text(res, "asset_layer", "default") or ("L0" if not adopted else None),
            "adopted": [str(a) for a in adopted or []], "rows": rows, "conditions": conds,
            "passed": _passed(res), "text": _text(res, "text", "zh", "summary", "conclusion"),
            "selection": table_rows(res.get("selection")) if isinstance(res.get("selection"), (list, dict)) else []}


def switch_gate(recs: list[dict]) -> dict:
    """A. T-S as the switch reads it (evaluation/switch.py state(), P9): the newest stored evaluation, checked (this
    module's schema, consistent, against the record that is still there, on today's model version), else why nothing
    is evaluated yet. A result another tool wrote is shown but switches nothing (status invalid; a failed one reads
    as failed). {status, passed, text, conditions [{id, text, passed, detail, evaluated}], reasons, source, date,
    default_text, commands, command_note, label}."""
    from . import switch
    st = switch.state(RESULTS_DIR)
    return {"status": st["status"], "passed": bool(st["passed"]), "text": st["text"],
            "conditions": conditions(st.get("conditions") or []), "reasons": list(st.get("reasons") or []),
            "source": (st.get("source") or {}).get("id"), "date": (str(st.get("evaluated_at") or "")[:10] or None),
            "default_text": st.get("default_text"), "commands": st.get("commands"),
            "command_note": st.get("command_note"), "label": st.get("label"), "defaults": st.get("defaults")}


# ================================================================ the factor layer's computed checks (qmodel.json)

def _card_names() -> dict:
    try:
        from ..qmodel import card as qcard
        return {r["id"]: r["zh"] for r in qcard.load_card()["rows"]}
    except (OSError, ValueError, KeyError):
        return {}


def _ppc_cell(c: dict) -> dict:
    return {k: c.get(k) for k in ("p05", "p50", "p95", "pass", "n", "share", "windows_inside", "share_of_quarters")
            if k in c}


def _label(text):
    """A statement label of qmodel.json as reader text ("fc 收紧" -> "金融条件 收紧")."""
    return re.sub(r"(?<![A-Za-z])fc(?![A-Za-z])", "金融条件", str(text)) if text else text


def _reader_notes(notes) -> list[str]:
    """Notes without the build plan's phase or section ids (those live in the docs, not on the page)."""
    return [n for n in notes or [] if isinstance(n, str) and not re.search(r"§|(?<![A-Za-z])P\d+(?![0-9])", n)]


def factor_layer() -> dict | None:
    q = _load(Path(RESULTS_DIR) / cq.QMODEL_RESULTS_PATH.name)
    fl = (q or {}).get("factor_layer") if isinstance(q, dict) else None
    if not isinstance(fl, dict):
        return None
    names = _card_names()
    tf = fl.get("T_F") or {}
    fits = tf.get("fits") or {}
    ends = list(fits)
    rows = []
    for sid, zh in TF_STATS.items():
        cells = {e: _ppc_cell((fits[e].get("ppc") or {}).get(sid) or {}) for e in ends if (fits[e].get("ppc") or {}).get(sid)}
        if not cells:
            continue
        first = (fits[ends[0]].get("ppc") or {}).get(sid) or {}
        hv = first.get("history_windows") if "history_windows" in first else first.get("history")
        if sid == "F6_bounds":
            cells = {e: {**{k: (fits[e]["ppc"][sid]).get(k) for k in ("ff_min", "u_min", "u_below3", "cpi_gt15", "pass")}}
                     for e in cells}
        rows.append({"id": sid, "group": sid[:2], "zh": zh, "short": TF_SHORT.get(sid, zh), "history": hv, "fits": cells,
                     "passed": all(bool(c.get("pass")) for c in cells.values())})
    last = fits.get(ends[-1], {}) if ends else {}
    f1 = last.get("F1") or {}
    deliv = [{"label": _label(d.get("label")), "field": names.get(d.get("var"), d.get("var")), "kind": d.get("kind"),
              "quarters": d.get("quarters"), "value": d.get("value"), "mean": d.get("mean"), "sd": d.get("sd"),
              "share_within_1sd": d.get("share_within_1sd"), "max_abs_error_sci": d.get("max_abs_error_sci"),
              "share_inside": d.get("share_inside"), "lo": d.get("lo"), "hi": d.get("hi")} for d in f1.get("delivery") or []]
    for d in deliv:                                   # the panel's own column ids are not reader text
        if isinstance(d["field"], str) and re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)+", d["field"]):
            d["field"] = {"cpi_yoy": "CPI 同比"}.get(d["field"], "派生变量")
    f7 = {}
    for h, blk in (tf.get("F7") or {}).items():
        if not isinstance(blk, dict):
            continue
        f7[h] = {"rows": [{"key": k, "zh": names.get(F7_FIELD.get(k, ""), k), "r2": v,
                           "engine": (blk.get("engine") or {}).get(k), "check": (blk.get("checks") or {}).get(k)}
                          for k, v in (blk.get("r2") or {}).items()],
                 "usd_held": (blk.get("checks") or {}).get("usd_held"), "n_origins": blk.get("n_origins"),
                 "n_indep": blk.get("n_indep"), "first": blk.get("first"), "last": blk.get("last"),
                 "pass": blk.get("pass"), "anecdote": f"{h}Q" in ANECDOTE_H}
    f8 = tf.get("F8") or {}
    f9 = last.get("F9") or {}
    f10 = last.get("F10") or {}
    summ = tf.get("summary") or {}
    tc = fl.get("T_C") or {}
    tc_test = {}
    for pat, by in (tc.get("test") or {}).items():
        if not isinstance(by, dict):
            continue
        tc_test[pat] = {"zh": TC_PATTERN.get(pat, pat), "rows": sorted(
            [{"field": f, "zh": names.get(f, f), "cls": v.get("cls"), "cls_zh": ENDO_ZH.get(v.get("cls"), v.get("cls")),
              "n": v.get("n"), "r2": v.get("r2_vs_uncond"),
              "crps_ratio": (v["crps"] / v["crps_uncond"]) if _num(v.get("crps")) and _num(v.get("crps_uncond")) else None,
              "cov80": v.get("cov80")} for f, v in by.items() if isinstance(v, dict)],
            key=lambda r: (list(ENDO_ZH).index(r["cls"]) if r["cls"] in ENDO_ZH else 9, r["zh"]))}
    gate = tc.get("gate") or {}
    sel = tc.get("selection") or {}
    return {
        "created": fl.get("created"), "version": fl.get("version"), "notes": _reader_notes(fl.get("notes")),
        "label": "计算值：只给出通过 / 未通过，不作为关卡；冻结的预注册记录见 G 节",
        "fits": [{"end": e, "anchors": fits[e].get("anchors"), "anchors_zh": ANCHORS_ZH.get(fits[e].get("anchors"), ""),
                  "origin": fits[e].get("origin")} for e in ends],
        "tf_rows": rows, "groups": [{"id": g, "zh": z} for g, z in TF_GROUPS],
        "f1": {"n_paths": f1.get("n_paths"), "hard_max_abs_error_sci": f1.get("hard_max_abs_error_sci"),
               "soft_share_within_1sd": f1.get("soft_share_within_1sd"), "delivery": deliv, "pass": f1.get("pass"),
               "rarity": f1.get("rarity")} if f1 else None,
        "f7": f7 or None,
        "f8": {"windows": f8.get("windows"), "first": f8.get("first"), "last": f8.get("last"),
               "coverage": f8.get("coverage"), "pass": f8.get("pass"),
               "per_field": [{"field": f, "zh": names.get(f, f), "coverage": v} for f, v in (f8.get("per_field") or {}).items()]}
        if f8 else None,
        "f9": {"same_quarter_E0_unchanged": f9.get("same_quarter_E0_unchanged"),
               "same_quarter_E0_max_abs_change_sci": f9.get("same_quarter_E0_max_abs_change_sci"),
               "signs": {k: v for k, v in (f9.get("signs") or {}).items() if k != "unrate_response"},
               "unrate_response": (f9.get("signs") or {}).get("unrate_response"),
               "signs_as_literature": f9.get("signs_as_literature"), "imputes_nothing": f9.get("imputes_nothing"),
               "pass": f9.get("pass")} if f9 else None,
        "f10": {k: f10.get(k) for k in ("n_draws", "ks_min_p", "ks_fields", "same_seed_max_abs_diff_sci", "pass")} if f10 else None,
        "summary": {"total": len(summ), "passed": sum(1 for v in summ.values() if v is True),
                    "failed": sorted(k for k, v in summ.items() if v is False)},
        "tc": {"validation": (f"验证：每年第 4 季用到上一年年底的数据重新拟合，给之后 4 个季度打分，起点 {cq.TC_VALIDATION[0]}–"
                              f"{cq.TC_VALIDATION[1]}（{tc.get('n_validation_origins') or '—'} 个起点）"),
               "test_end": tc.get("test_end"),
               "n_validation_origins": tc.get("n_validation_origins"), "n_test_origins": tc.get("n_test_origins"),
               "completion_w": sel.get("completion_w"), "by_w": sel.get("by_w"),
               "regime_cov_adopted": (sel.get("regime_cov") or {}).get("adopted"),
               "gate_rule": "情形 b（只给宏观与政策）：每个金融市场字段的补全 R² ≥ −0.02",
               "gate_r2": [{"field": f, "zh": names.get(f, f), "r2": v} for f, v in (gate.get("E1_r2") or {}).items()],
               "gate_pass": gate.get("pass"), "pass": tc.get("pass"), "test": tc_test} if tc else None,
    }


def delivery() -> dict | None:
    d = _load(Path(RESULTS_DIR) / "qtable_delivery.json")
    if not isinstance(d, dict) or not isinstance(d.get("scenarios"), dict):
        return None
    try:
        from ..qmodel import qtable as QT
        names = {sid: (QT.builtin(sid).get("name") or sid) for sid in d["scenarios"]}
    except Exception:  # noqa: BLE001 - the built-in envelopes are optional here
        names = {}
    views, per = [], []
    for sid, s in d["scenarios"].items():
        vs = [v for v in s.get("views") or [] if isinstance(v, dict)]
        for v in vs:
            views.append({"scenario": names.get(sid, sid), "phase": v.get("phase_name"), "target": v.get("target"),
                          "quarterly": v.get("quarterly"), "engine": v.get("engine"), "gap": v.get("gap"),
                          "engine_gap": v.get("engine_gap"), "tolerance": v.get("tolerance"), "ok": v.get("ok")})
        gaps = sorted(abs(v["gap"]) / v["tolerance"] for v in vs if _num(v.get("gap")) is not None and _num(v.get("tolerance")))
        eg = sorted(abs(v["engine_gap"]) / v["tolerance"] for v in vs
                    if _num(v.get("engine_gap")) is not None and _num(v.get("tolerance")))
        per.append({"scenario": names.get(sid, sid), "views": len(vs), "ok": sum(1 for v in vs if v.get("ok")),
                    "median_gap_tol": gaps[len(gaps) // 2] if gaps else None,
                    "median_engine_gap_tol": eg[len(eg) // 2] if eg else None,
                    "dropped": len(s.get("dropped") or []), "excluded_paths": s.get("excluded_paths")})
    return {"label": d.get("label"), "n_paths": d.get("n_paths"), "statistic": "各路径阶段均值的中位数",
            "n_views": len(views), "n_ok": sum(1 for v in views if v.get("ok")), "scenarios": per, "views": views,
            "ok": d.get("ok")}


# ================================================================ the rest of the page

def ledger() -> dict | None:
    m = _load(Path(RESULTS_DIR) / "forecast_mix.json")
    led = (m or {}).get("ledger") if isinstance(m, dict) else None
    if not isinstance(led, dict):
        return None
    ents = led.get("entries") or []
    return {"n_entries": led.get("n_entries", len(ents)), "text": led.get("text"),
            "origins": sorted({str(e.get("origin")) for e in ents if e.get("origin")}),
            "matured": sum(int(e.get("matured") or 0) for e in ents), "summary": led.get("summary") or {},
            "engines": sorted({str(e.get("engine")) for e in ents if e.get("engine")}), "created": m.get("created")}


def narrative_runs() -> dict:
    try:
        from .narrative import report as RP
        runs = RP.runs()
    except Exception:  # noqa: BLE001 - the narrative package's archive is optional for this view
        runs = []
    real = [r for r in runs if r.get("kind") == "real" and r.get("status") == "done" and r.get("has_report")]
    research = next((r for r in runs if r.get("kind") == "research"), None)
    head = real[0] if real else research
    return {"headline_run": head["id"] if head else None,
            "headline_kind": ("real" if real else "research") if head else None,
            "n_real": len(real), "n_mock": sum(1 for r in runs if r.get("kind") == "mock"), "n_runs": len(runs)}


def _ablation() -> dict | None:
    d = _load(ABLATION_PATH)
    return d if isinstance(d, dict) and d.get("schema") == "oracle-ablation/1" else None


def conclusion(tests: dict, al: dict | None, gate: dict, fl: dict | None, abl: dict | None, nar: dict) -> dict:
    """Section A without the narrative test (the page adds the headline run's headline): every sentence is a status,
    a pre-registered gate's result or a labelled research finding; nothing claims an improvement."""
    parts = [{"key": "intro", "kind": "info",
              "text": "情景条件检验回答两个问题：情景给定以后，模型的模拟离实际有多远；把叙事换成错的，结果会不会更差。"
                      "它不检验预测能力：预测只在情景概率里。", "href": None}]
    if nar.get("n_real", 0) == 0:
        parts.append({"key": "narrative_missing", "kind": "warn",
                      "text": "真实大模型的叙事检验还没有按盲测规范运行过（v2 叙事还没有写）：C 节的数字来自研究轮的代译回放"
                              "和模拟后端的流程检验。真实运行只在你看过估算并确认之后才开始。", "href": "#evaluation?narrative"})
    parts.append({"key": "switch", "kind": "good" if gate.get("passed") else "warn", "text": gate["text"],
                  "href": "#evaluation?records"})
    ta = tests.get("TA") or {}
    if al:
        txt = al.get("text") or (f"资产层候选检验 T-A（{al.get('date') or ta.get('date')}）：采纳 {'、'.join(al['adopted'])}。"
                                 if al.get("adopted") else
                                 f"资产层候选检验 T-A（{al.get('date') or ta.get('date')}）：没有候选通过，资产层用 L0（因子轮 F）。")
        parts.append({"key": "asset_layer", "kind": "info", "text": txt, "href": "#evaluation?oracle"})
    elif ta.get("status") == "registered":
        parts.append({"key": "asset_layer", "kind": "info",
                      "text": f"资产层候选检验 T-A 已预注册（{ta.get('date')}），还没有结果：资产层用默认的 L0（因子轮 F）。",
                      "href": "#evaluation?records"})
    else:
        parts.append({"key": "asset_layer", "kind": "info",
                      "text": "资产层候选检验 T-A 尚无记录：资产层用默认的 L0（因子轮 F），F 以外的字段只经由因子层影响资产。",
                      "href": "#evaluation?records"})
    if fl:
        s = fl.get("summary") or {}
        tc = fl.get("tc") or {}
        bad = []
        for k in s.get("failed") or []:
            name = TF_SHORT.get(k.split(":")[-1], k.split(":")[-1])
            if name not in bad:
                bad.append(name)
        txt = (f"因子引擎检验 T-F（模拟能否复现历史的统计特征；计算值，{len(fl.get('fits') or [])} 个训练期末）：{s.get('passed', 0)}/{s.get('total', 0)} 项通过"
               + (f"；没有通过的：{'、'.join(bad)}" if bad else "") + "。")
        if tc:
            low = [g["zh"] for g in tc.get("gate_r2") or [] if _num(g.get("r2")) is not None and g["r2"] < -0.02]
            txt += (f"补全检验 T-C（计算值）：只给宏观时，{'、'.join(low)}的补全 R² 低于 −0.02，关卡未通过。" if low else
                    "补全检验 T-C（计算值）：只给宏观时，金融市场字段的补全都不比无条件抽样差 0.02 以上。"
                    + ("关卡通过。" if tc.get("gate_pass") else ""))
        parts.append({"key": "factor_layer", "kind": "info", "text": txt + "冻结的预注册记录见 G 节。",
                      "href": "#evaluation?engine"})
    else:
        parts.append({"key": "factor_layer", "kind": "info",
                      "text": "因子引擎检验尚无结果：季度模型这一步构建之后才有 T-F 与 T-C 的计算值。",
                      "href": "#evaluation?engine"})
    if abl:
        parts.append({"key": "ablation", "kind": "info", "text": f"{abl.get('label')}：{abl.get('text')}",
                      "href": "#evaluation?oracle"})
    return {"parts": parts}


def summary() -> dict:
    recs = records()
    tests = tests_status(recs)
    al = asset_layer(recs)
    gate = switch_gate(recs)
    fl = factor_layer()
    abl = _ablation()
    nar = narrative_runs()
    ev = _load(Path(RESULTS_DIR) / "evaluation.json")
    ev = ev if isinstance(ev, dict) else {}
    cond = ev.get("conditional") if isinstance(ev.get("conditional"), (dict, list)) else None
    if cond is not None:
        cond = {**(cond if isinstance(cond, dict) else {}), "rows": table_rows(cond)}
    anat = ev.get("anatomy") if isinstance(ev.get("anatomy"), (dict, list)) else None
    if isinstance(anat, list):
        anat = {"rows": anat}
    return {
        "schema": SCHEMA, "label": cq.TRIAL_LABEL, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "panel": Path(cq.PANEL_Q_PATH).exists(),        # D reads /api/q/valuation only when the quarterly panel exists
        "records": recs, "tests": tests, "asset_layer": al, "conditional": cond, "ablation": abl, "anatomy": anat,
        "factor_layer": fl, "delivery": delivery(), "switch_gate": gate, "ledger": ledger(), "narrative": nar,
        "conclusion": conclusion(tests, al, gate, fl, abl, nar), "contamination_notes": list(CONTAMINATION_NOTES),
        "notes": ["每个数字都来自构建结果：预注册记录（G 节）、因子层的检验结果、条件路径表、叙事检验的报告与研究轮的存档。",
                  "12 和 20 个季度的数字：每个十年只有 2–3 个独立窗口，只能当个案。"],
    }


# ================================================================ endpoints

@router.get("")
def evaluation_summary():
    return _json(summary())


@router.get("/records/{rid}")
def evaluation_record(rid: str):
    try:
        return _json(record(rid))
    except KeyError as err:
        raise HTTPException(404, str(err.args[0] if err.args else err)) from err
