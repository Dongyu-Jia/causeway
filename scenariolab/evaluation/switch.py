"""切换关 T-S (SIM_DESIGN §4.6, §11.3, §14 P9): when the quarterly stack replaces the monthly engine as the default for
the 13 built-in scenarios, the 未来一年 tab, 组合评估 and 提取与破产概率.

Two parts, both read-only with respect to the models (nothing here fits, simulates or calls an LLM):

  evaluate()  the gate. Inputs: P5's frozen T-S record (scenariolab/qmodel/preregistered/<date>_TS.json: the protocol
              and its thresholds; the record's own numbers win, SIM_DESIGN §4.6's are the defaults and the result says
              which were used), the factor layer's T-F result (a frozen TF record's latest result, else qmodel.json's
              computed checks) and one narrative-test run with both engines (P7: data/results/narrative/<run_id>/
              score_rows.csv, the per-cell scores; episodes.json for the engine versions; translations/ for where each
              narrative table came from). It recomputes every comparison from the stored rows with the narrative
              test's own statistics (narrative/stats.py: episodes are the unit, 2000 bootstrap draws, 90% intervals).
  state()     the switch every consumer reads: the portfolio and withdrawal services' default scenario set, the forecast
              tab's default engine, the scenarios tab and the evaluation tab. It is on only when a stored evaluation is
              this module's (schema switch-gate/1), passed, internally consistent, made against the T-S record that is
              still there unchanged, and made on the quarterly stack's current model version. Anything else (no
              record, no evaluation, a failed or hand-written result, a changed record or model) switches nothing,
              and the reasons travel with the state.

Conditions (§4.6, on the narrative test's clean episode set, equity9, the headline CMA "historical"):

  TF        T-F passes: every check of the factor layer's posterior predictive tests (§5.8)
  O:<arm>:<h>  oracle arms O-A, O-B, O-full: Q − E equity9 MAE (paired over episodes) has a 90% upper bound below
            +0.02, at 4Q and at the window end; E is the monthly engine running the selected model F
  PIT       calibration: Q's mean |PIT − 0.5| is at most E's + 0.02, on the cells both engines have (those arms and
            horizons, equity9, clean set; per episode, then the mean over episodes)
  N:<arm>:<h>  narrative arms N1–N3 on identical translations (each QTable converted for E by to_phase_spec): Q − E
            upper bound below +0.03, at 4Q and at the window end. Evaluated only on a run of kind "real" (a real
            LLM's translations); a mock run (the token-free build: the mock backend and the research round's
            stand-ins, non-blind and without quotes) is a pipeline check, and its narrative conditions are "not
            evaluated" unless the frozen record itself lists "standin" in its narrative_sources
  IN:*      the inputs: the run is finished and scored by both engines; E ran the selected F; one quarterly-stack
            version across the run; every clean episode is in every oracle comparison and in every narrative
            comparison (the record may set narrative_coverage "with_narratives" to accept the episodes that have
            the narrative; the condition then says how many it covers); the narrative tables are a real LLM's
            translations (or stand-ins, when the record opts in) and never the minimal fallback table

A condition whose inputs are missing is "not evaluated", and T-S then does not pass: the gate fails closed.

Where results go (evaluate --write / --append): data/results/switch_gate.json (what state() reads first) and one line
appended to the record's sibling <stem>.results.jsonl (append-only; the frozen record itself is never edited). The
real evaluation runs by hand after the P5 records are merged and the v2 narratives are written (a real run spends the
user's plan; the token-free `narrative build` is a pipeline check that cannot pass T-S, and --append refuses it):

    python -m scenariolab.evaluation.narrative estimate --backend claude_code --episodes clean --engines Q,E
    python -m scenariolab.evaluation.narrative run --estimate <id>       # asks before it spends
    python -m scenariolab.evaluation.switch evaluate --run latest --write --append

    python -m scenariolab.evaluation.switch status                       # the switch as the app reads it
    python -m scenariolab.evaluation.switch inputs                       # what an evaluation would read, what is missing
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .. import config, jsonio

VERSION = 1
SCHEMA = "switch-gate/1"                    # what evaluate() writes
STATE_SCHEMA = "switch-state/1"             # what state() returns
EVALUATOR = "scenariolab.evaluation.switch"
GATE_FILE = "switch_gate.json"
RESULTS_DIR = config.RESULTS_DIR            # module attribute: tests point it at a fixture folder
MODEL_SELECTION = "model_selection.json"
CONSUMERS = ("scenarios", "forecast", "portfolio", "withdrawal")
CONSUMER_ZH = {"scenarios": "13 个内置情景", "forecast": "未来一年页", "portfolio": "组合评估", "withdrawal": "提取与破产概率"}
ENGINE_ZH = {"monthly": "月度引擎", "quarterly": "季度模型"}
TRIAL = "试运行"
DEFAULT_TAG = "默认"

# §4.6 as numbers (the defaults when the frozen record does not state them)
DEFAULT_PROTOCOL = {
    "set": "clean", "assets": "equity9", "cma": "historical", "horizons": ["4Q", "end"],
    "oracle_arms": ["OA", "OB", "OF"], "oracle_hi90_below": 0.02,
    "narrative_arms": ["N1", "N2", "N3"], "narrative_hi90_below": 0.03,
    "pit_margin": 0.02, "pit_arms": None,          # None: the oracle and narrative arms above
    "narrative_sources": ["llm"],                  # stand-ins only when the frozen record lists "standin"
    "narrative_coverage": "clean",                 # every clean episode ("with_narratives": those with the narrative)
    "tf": "all_checks",
}
COVERAGES = ("clean", "with_narratives")
# record keys (and their usual aliases) for the protocol's numbers and lists
PROTOCOL_ALIASES = {
    "oracle_hi90_below": ("oracle_hi90_below", "oracle_margin", "margin_oracle", "oracle", "oracle_bound"),
    "narrative_hi90_below": ("narrative_hi90_below", "narrative_margin", "margin_narrative", "narrative",
                             "narrative_bound"),
    "pit_margin": ("pit_margin", "calibration_margin", "pit_mad_margin", "pit", "calibration"),
    "horizons": ("horizons",), "oracle_arms": ("oracle_arms",), "narrative_arms": ("narrative_arms",),
    "set": ("set", "episode_set"), "assets": ("assets", "asset_set"), "cma": ("cma", "head_cma"),
    "narrative_sources": ("narrative_sources", "translation_sources"),
    "narrative_coverage": ("narrative_coverage", "coverage"),
}
ARM_ZH = {"OA": "已知宏观（O-A）", "OAp": "加已知商品价格（O-A′）", "OB": "已知宏观与金融市场（O-B）",
          "OF": "全部字段已知（O-full）", "N1": "叙事 N1", "N2": "叙事 N2（加金融条件）", "N3": "叙事 N3（不含数字）",
          "U": "无叙事"}
H_ZH = {"1Q": "1 个季度", "2Q": "2 个季度", "4Q": "4 个季度", "8Q": "8 个季度", "end": "窗口末"}
ARM_NARRATIVE = {"N1": "n1", "N2": "n2", "N3": "n3"}
SOURCE_ZH = {"llm": "真实大模型的翻译", "standin": "研究轮代译（无引文）", "mock": "模拟后端生成的表（没有内容）",
             "fallback": "翻译失败后的最小表", "missing": "没有翻译记录"}
TRIAL_WHERE = "情景实验室和这些页面的对照"          # where the trial runs (agent_tree_where() adds 智能体推演 when it does)
NOT_EVALUATED_TEXT = ("切换关 T-S 还没有评估。在它通过之前，13 个内置情景、未来一年页、组合评估与提取都用月度引擎；"
                      "季度模型只在{where}里作试运行。")
COMMANDS = ["python -m scenariolab.evaluation.narrative estimate --backend claude_code --episodes clean --engines Q,E",
            "python -m scenariolab.evaluation.narrative run --estimate <估算编号>",
            "python -m scenariolab.evaluation.switch evaluate --run latest --write --append"]
COMMAND_NOTE = ("预注册记录合并、盲测规范 v2 的叙事写好之后，在项目根目录手动运行：第一行只估算（不调用模型），第二行在你确认后调用"
                "本机 Claude Code（计入你的订阅额度），第三行按冻结的 T-S 记录评估并写下结果。不花额度的叙事检验构建（模拟后端加研究轮"
                "代译）只是流程检验：它的叙事臂不评估，T-S 不会因此通过。")
MOCK_RUN_TEXT = ("这是模拟后端的运行（流程检验：研究轮代译不盲、没有引文）：叙事臂不评估。只有真实大模型翻译的运行（或冻结记录明确"
                 "接受代译）才能评估叙事臂")


def trial_where() -> str:
    """Where the quarterly stack runs as a trial: the lab and the comparisons on these pages, and the agent tree only
    when its model adapter is the quarterly one (by default it runs the Gaussian stub)."""
    import os
    name = (os.environ.get("MSL_AGENT_MODEL") or "stub").strip()      # agents/model_adapter.load_adapter's choice
    return "情景实验室、智能体推演和这些页面的对照" if "qmodel" in name else TRIAL_WHERE


def not_evaluated_text() -> str:
    return NOT_EVALUATED_TEXT.format(where=trial_where())


class SwitchError(ValueError):
    """An evaluation request the gate cannot serve (Chinese message)."""


# ================================================================ small helpers

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x):
    if isinstance(x, bool) or x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _signed(x, nd: int = 3) -> str:
    v = _num(x)
    return "—" if v is None else f"{v:+.{nd}f}"


def _codes_text(codes, k: int = 6) -> str:
    codes = list(codes)
    return "、".join(codes[:k]) + (f" 等 {len(codes)} 个" if len(codes) > k else "")


def semantic_version(version: str | None) -> str | None:
    """'qmodel-r1-f1-a1/2000Q4/train_end/s0/b200/…' -> 'qmodel-r1-f1-a1': the layers' versions, without the train end,
    the anchors, the seed and the data digest (a data update does not change what T-S certified; a layer does)."""
    if not version or not isinstance(version, str):
        return None
    head = version.split("/", 1)[0].strip()
    return head if head.startswith("qmodel-") else None


def current_model_version() -> str:
    """The quarterly stack's semantic version now (rollout, factor layer, asset layer)."""
    from ..qmodel import assets as AS
    from ..qmodel import factors as F
    from ..qmodel import rollout as R
    return f"qmodel-r{R.VERSION}-f{F.VERSION}-a{AS.VERSION}"


# ================================================================ the frozen T-S record and the protocol

def _ev():
    from . import evalview
    return evalview


def protocol_digest(data) -> str | None:
    """sha256 of a record without its appended results (`results` / `result`): what an evaluation was made against.
    Results appended inside the file or beside it leave it unchanged; an edit to the protocol changes it."""
    if not isinstance(data, dict):
        return None
    body = {k: v for k, v in data.items() if k not in ("results", "result")}
    return _sha(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def ts_record(record_id: str | None = None) -> dict | None:
    """The newest frozen T-S record (or the named one): {id, file, path, sha256, protocol_sha256, date, data,
    results, status}."""
    EV = _ev()
    recs = [r for r in EV.records() if r["test"] == "TS"]
    if record_id:
        recs = [r for r in recs if r["id"] == record_id]
        if not recs:
            raise SwitchError(f"找不到 T-S 记录「{record_id}」")
    if not recs:
        return None
    r = recs[0]
    full = EV.record(r["id"])
    path = next((p for p in EV._record_files() if p.stem == r["id"]), None)
    return {"id": r["id"], "file": r.get("file"), "path": path, "sha256": r.get("sha256"), "date": r.get("date"),
            "protocol_sha256": protocol_digest(full.get("record")),
            "data": full.get("record"), "results": full.get("results") or [], "status": r.get("status")}


def _alias(block: dict, key: str):
    for k in PROTOCOL_ALIASES.get(key, (key,)):
        if isinstance(block, dict) and k in block and block[k] is not None:
            return block[k]
    return None


def protocol(record: dict | None) -> tuple[dict, dict]:
    """(the effective protocol, {source, from_record [keys], defaults [keys], problems [Chinese], notes [Chinese]}).

    The record's numbers come from its `thresholds`, `protocol` or `gate` block (a number, or {hi90_below | margin |
    value}). A block the reader does not understand is a problem: the evaluation then cannot pass."""
    proto = json.loads(json.dumps(DEFAULT_PROTOCOL))
    info = {"source": "spec", "from_record": [], "defaults": [], "problems": [], "notes": []}
    data = (record or {}).get("data") if record else None
    if not isinstance(data, dict):
        info["defaults"] = sorted(proto)
        if record is not None:
            info["problems"].append("T-S 记录不是合法的 JSON 对象：读不出协议")
        return proto, info
    blocks = []
    for key in ("thresholds", "protocol", "gate"):
        b = data.get(key)
        if isinstance(b, dict):
            blocks.append(b)
            for sub in ("thresholds", "margins"):
                if isinstance(b.get(sub), dict):
                    blocks.append(b[sub])
        elif b is not None and not isinstance(b, (str, list)):
            info["problems"].append(f"T-S 记录的「{key}」既不是对象也不是文字：读不出门槛")
    for key in proto:
        val = None
        for b in blocks:
            val = _alias(b, key)
            if val is not None:
                break
        if val is None:
            info["defaults"].append(key)
            continue
        if key in ("oracle_hi90_below", "narrative_hi90_below", "pit_margin"):
            if isinstance(val, str) and _num(val) is None:      # a description in words: the number stays §4.6's
                info["defaults"].append(key)
                info["notes"].append(f"记录里的「{key}」是文字说明，不是数字：门槛按设计默认")
                continue
            if isinstance(val, dict):
                val = next((val[k] for k in ("hi90_below", "margin", "value", "bound") if _num(val.get(k)) is not None),
                           None)
            v = _num(val)
            if v is None:
                info["problems"].append(f"T-S 记录的门槛「{key}」不是数字：{str(val)[:40]}")
                continue
            proto[key] = v
        elif key in ("horizons", "oracle_arms", "narrative_arms", "narrative_sources"):
            if not isinstance(val, list) or not all(isinstance(x, str) for x in val) or not val:
                info["problems"].append(f"T-S 记录的「{key}」不是非空的文字列表")
                continue
            proto[key] = [str(x) for x in val]
        else:
            if not isinstance(val, str):
                info["problems"].append(f"T-S 记录的「{key}」不是文字")
                continue
            proto[key] = val
        info["from_record"].append(key)
    _conditions_list(data, proto, info)
    info["source"] = "record" if info["from_record"] else "spec"
    for key in ("oracle_hi90_below", "narrative_hi90_below", "pit_margin"):
        if key in info["from_record"] and abs(proto[key] - DEFAULT_PROTOCOL[key]) > 1e-12:
            info["notes"].append(f"记录的门槛 {key} = {proto[key]:+.3f}，与设计默认的 "
                                 f"{DEFAULT_PROTOCOL[key]:+.3f} 不同：按记录")
    if info["defaults"] and info["from_record"]:
        info["notes"].append("记录没有写的协议项按设计默认：" + "、".join(info["defaults"]))
    elif not info["from_record"]:
        info["notes"].append("T-S 记录没有写门槛：全部按设计默认（预言机臂 +0.02、叙事臂 +0.03、PIT +0.02）")
    for arm in proto["oracle_arms"] + proto["narrative_arms"]:
        if arm not in ARM_ZH:
            info["problems"].append(f"T-S 记录里的臂「{arm}」不认识")
    for hz in proto["horizons"]:
        if hz not in H_ZH:
            info["problems"].append(f"T-S 记录里的期限「{hz}」不认识")
    # what this evaluator can compute (anything else is a problem, and the spec's value is kept so nothing crashes)
    for key, ok in (("assets", ("equity9",)), ("set", ("clean",)), ("cma", ("historical", "no_rerate"))):
        v = proto[key]
        if v in ok:
            continue
        hit = [o for o in ok if isinstance(v, str) and o in v]
        if len(hit) == 1:                       # "clean（§8.2 的 14 个片段）": the token inside a description
            proto[key] = hit[0]
        elif isinstance(v, str) and (len(v) > 24 or re.search(r"[\u4e00-\u9fff\s]", v)):
            info["notes"].append(f"记录的「{key}」是文字说明：按设计默认（{DEFAULT_PROTOCOL[key]}）")
            proto[key] = DEFAULT_PROTOCOL[key]
        else:
            info["problems"].append(f"T-S 记录的「{key}」是「{v}」：评估器只支持 {' / '.join(ok)}")
            proto[key] = DEFAULT_PROTOCOL[key]
    if not set(proto["narrative_sources"]) <= set(SOURCE_ZH):
        info["problems"].append("T-S 记录里的翻译来源不认识：" + "、".join(proto["narrative_sources"]))
        proto["narrative_sources"] = list(DEFAULT_PROTOCOL["narrative_sources"])
    if proto["narrative_coverage"] not in COVERAGES:
        info["problems"].append(f"T-S 记录的叙事覆盖「{proto['narrative_coverage']}」不认识（clean 或 with_narratives）")
        proto["narrative_coverage"] = DEFAULT_PROTOCOL["narrative_coverage"]
    return proto, info


ORACLE_ARMS = ("OA", "OAp", "OB", "OF")
NARRATIVE_ARMS = ("N1", "N2", "N3")


def _conditions_list(data: dict, proto: dict, info: dict) -> None:
    """A record that lists its comparisons as conditions, the T-N record's way ({a: "Q:OA", b: "E:OA", horizon,
    assets, set, hi90_below}, and a calibration item {pit_margin | margin, ...}): the arms, horizons and margins come
    from them. One margin per group (oracle, narrative); an item the reader cannot place is a problem."""
    items = None
    for key in ("conditions", "all", "gate"):
        v = data.get(key)
        if isinstance(v, list):
            items = v
            break
        if isinstance(v, dict) and isinstance(v.get("conditions"), list):
            items = v["conditions"]
            break
    proto_block = data.get("protocol")
    if items is None and isinstance(proto_block, dict) and isinstance(proto_block.get("conditions"), list):
        items = proto_block["conditions"]
    if not items:
        return
    groups = {"oracle": {"arms": [], "h": [], "m": set()}, "narrative": {"arms": [], "h": [], "m": set()}}
    for it in items:
        if not isinstance(it, dict):
            info["problems"].append("T-S 记录的条件列表里有不是对象的项")
            continue
        a, b = str(it.get("a") or ""), str(it.get("b") or "")
        if a.startswith("Q:") and b.startswith("E:") and a[2:] == b[2:]:
            arm = a[2:]
            g = "oracle" if arm in ORACLE_ARMS else "narrative" if arm in NARRATIVE_ARMS else None
            m = _num(it.get("hi90_below"))
            hz = str(it.get("horizon") or "")
            if g is None or m is None or hz not in H_ZH:
                info["problems"].append(f"T-S 记录的条件「{a} − {b}」读不懂（臂、期限或门槛）")
                continue
            for k, want in (("assets", proto["assets"]), ("set", proto["set"])):
                if it.get(k) is not None and it.get(k) != want:
                    info["problems"].append(f"T-S 记录的条件「{a} − {b}」用的{k}是「{it.get(k)}」，评估器只支持「{want}」")
            G = groups[g]
            if arm not in G["arms"]:
                G["arms"].append(arm)
            if hz not in G["h"]:
                G["h"].append(hz)
            G["m"].add(m)
            continue
        txt = json.dumps(it, ensure_ascii=False).lower()
        m = next((_num(it.get(k)) for k in ("pit_margin", "margin", "bound") if _num(it.get(k)) is not None), None)
        if ("pit" in txt or "calibration" in txt) and m is not None:
            proto["pit_margin"] = m
            info["from_record"].append("pit_margin")
            continue
        if str(it.get("id") or it.get("test") or "").replace("-", "").upper() in ("TF", "TFPASS"):
            continue                                   # T-F passes: always a condition here
        info["problems"].append(f"T-S 记录的条件读不懂：{txt[:60]}")
    for g, key in (("oracle", "oracle_hi90_below"), ("narrative", "narrative_hi90_below")):
        G = groups[g]
        if not G["arms"]:
            continue
        if len(G["m"]) > 1:
            info["problems"].append(f"T-S 记录给{g}臂的门槛不止一个：{sorted(G['m'])}")
            continue
        proto[key] = G["m"].pop()
        proto[f"{g}_arms"] = G["arms"]
        info["from_record"] += [key, f"{g}_arms"]
        hs = sorted(set(G["h"]), key=lambda h: list(H_ZH).index(h))
        if proto.get("_h_from_list") and proto["horizons"] != hs:
            info["problems"].append("T-S 记录里两组条件的期限不同：评估器要求同一组期限")
        proto["horizons"], proto["_h_from_list"] = hs, True
        info["from_record"].append("horizons")
    proto.pop("_h_from_list", None)
    info["from_record"] = sorted(set(info["from_record"]))
    info["defaults"] = [k for k in info["defaults"] if k not in info["from_record"]]


def required_ids(proto: dict) -> list[str]:
    ids = ["TF"]
    ids += [f"O:{a}:{h}" for a in proto["oracle_arms"] for h in proto["horizons"]]
    ids.append("PIT")
    ids += [f"N:{a}:{h}" for a in proto["narrative_arms"] for h in proto["horizons"]]
    return ids


# ================================================================ T-F

def tf_result(results_dir: Path | None = None) -> dict | None:
    """The factor layer's T-F: a frozen TF record's latest result, else qmodel.json's computed checks (every one must
    pass). {source, passed, n, failed [ids], label, date, sha256, quick} or None when neither exists."""
    EV = _ev()
    recs = [r for r in EV.records() if r["test"] == "TF" and r.get("n_results")]
    if recs:
        r = recs[0]
        last = r.get("last_result") or {}
        return {"source": "record", "record": r["id"], "sha256": r.get("sha256"), "date": last.get("date"),
                "passed": last.get("passed") is True, "n": None, "failed": [], "quick": False,
                "label": f"冻结记录 {r['id']}", "text": last.get("text")}
    path = Path(results_dir or RESULTS_DIR) / "qmodel.json"
    try:
        raw = path.read_bytes()
        q = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError):
        return None
    fl = q.get("factor_layer") if isinstance(q, dict) else None
    summ = ((fl or {}).get("T_F") or {}).get("summary")
    if not isinstance(summ, dict) or not summ:
        return None
    failed = sorted(k for k, v in summ.items() if v is not True)
    return {"source": "qmodel.json", "sha256": _sha(raw), "date": (fl or {}).get("created"),
            "passed": not failed, "n": len(summ), "failed": failed, "quick": bool((fl or {}).get("quick")),
            "label": "因子层拟合时的计算值（未冻结）", "text": None}


def _tf_failed_text(keys) -> str:
    """'1994Q4:F2_cpi_acf4', '2004Q4:F2_cpi_acf4', 'F7' -> 'CPI 同比的自相关（2 个训练期末）、补全技能'."""
    EV = _ev()
    by: dict = {}
    for k in keys:
        end, _, sid = k.partition(":") if ":" in k else ("", "", k)
        by.setdefault(sid, []).append(end)
    parts = []
    for sid, ends in by.items():
        name = EV.TF_SHORT.get(sid) or EV.TF_STATS.get(sid) or sid
        n = len([e for e in ends if e])
        parts.append(f"{name}（{n} 个训练期末）" if n > 1 else f"{name}（{ends[0]}）" if n == 1 else name)
    return "、".join(parts)


def tf_condition(tf: dict | None) -> dict:
    c = {"id": "TF", "group": "tf", "text": "因子引擎检验 T-F 全部通过"}
    if tf is None:
        return {**c, "evaluated": False, "passed": None, "detail": "还没有 T-F 结果（qmodel.json 或冻结的 TF 记录）"}
    if tf.get("quick"):
        return {**c, "evaluated": False, "passed": None, "source": tf["source"],
                "detail": "qmodel.json 是快速版本（路径数少）：不能用于切换关，请运行完整的 qmodel 构建"}
    failed = tf.get("failed") or []
    n = tf.get("n")
    if tf["passed"]:
        detail = f"{n} 项全部通过（{tf['label']}）" if n else f"通过（{tf['label']}）"
    else:
        detail = (f"{len(failed)} / {n} 项没有通过（{tf['label']}）：" + _tf_failed_text(failed)) if n \
            else f"没有通过（{tf['label']}）"
    return {**c, "evaluated": True, "passed": bool(tf["passed"]), "source": tf["source"], "n_failed": len(failed),
            "n": n, "failed": failed, "detail": detail}


# ================================================================ the narrative run

def _narrative():
    from .narrative import common as C
    return C


def list_runs() -> list[dict]:
    """Narrative-test runs (run.json), newest first, with the engines they were scored on."""
    root = _narrative().runs_dir()
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        p = d / "run.json"
        if not (d.is_dir() and p.exists()):
            continue
        r = _load(p)
        if not isinstance(r, dict):
            continue
        st = _load(d / "stats.json") or {}
        engines = list((st.get("settings") or {}).get("engines") or (r.get("settings") or {}).get("engines") or [])
        out.append({"id": r.get("id") or d.name, "status": r.get("status"), "kind": r.get("kind"),
                    "backend": r.get("backend"), "created": r.get("created"), "engines": engines,
                    "narratives": (r.get("settings") or {}).get("narratives"), "dir": d,
                    "scored": (d / "score_rows.csv").exists()})
    out.sort(key=lambda r: (str(r.get("created") or ""), r["id"]), reverse=True)
    return out


def latest_run() -> str | None:
    """The newest finished run scored on both engines (Q and E): a real LLM's run first (only it can pass T-S), else
    the newest mock run (a pipeline check: its narrative conditions are not evaluated)."""
    done = [r for r in list_runs() if r["status"] == "done" and r["scored"] and {"Q", "E"} <= set(r["engines"])]
    real = [r for r in done if r.get("kind") == "real"]
    return (real or done or [{"id": None}])[0]["id"]


def _classify(rec: dict | None) -> str:
    """Where a narrative arm's table came from: llm | standin | mock | fallback | missing."""
    if not isinstance(rec, dict):
        return "missing"
    st = str(rec.get("status") or "")
    if st == "standin" or rec.get("source_kind") == "legacy":
        return "standin"
    if st in ("fallback", "error", "paused", "refused") or not isinstance(rec.get("qtable"), dict):
        return "fallback"
    if str(rec.get("backend") or "") == "mock":
        return "mock"
    return "llm" if st in ("ok", "repaired") else "fallback"


@dataclass
class Inputs:
    """What one evaluation reads (load_inputs builds it from the files; tests build it from fixtures)."""
    record: dict | None = None
    tf: dict | None = None
    run: dict | None = None
    rows: object = None                       # pandas DataFrame of score_rows.csv
    stats: dict | None = None
    episodes: dict | None = None
    translations: dict = field(default_factory=dict)      # (code, arm) -> translation record (or None)
    registry: dict | None = None              # {"clean": [codes], "narratives": {code: {version: [n1, ...]}}}
    selected_model: str | None = None         # model_selection.json spec name (the F the engine arm must run)
    model_now: str | None = None              # current_model_version()
    notes: list = field(default_factory=list)


def _registry() -> dict | None:
    try:
        from .narrative import episodes as EP
        reg = EP.load_registry()
    except Exception:  # noqa: BLE001 - no registry: completeness is checked against the run's own episodes
        return None
    codes = reg.get("codes") or {}
    return {"clean": sorted(c for c, e in codes.items() if e.get("set") == "clean"),
            "narratives": {c: (e.get("narratives") or {}) for c, e in codes.items()},
            "standin": {c: (e.get("standin") or {}) for c, e in codes.items()}}


def load_inputs(run_id: str | None, *, record_id: str | None = None, results_dir: Path | None = None) -> Inputs:
    """Read every input of an evaluation from the files (no simulation)."""
    import pandas as pd
    results_dir = Path(results_dir or RESULTS_DIR)
    inp = Inputs(record=ts_record(record_id), tf=tf_result(results_dir), registry=_registry())
    sel = _load(results_dir / MODEL_SELECTION)
    inp.selected_model = ((sel or {}).get("spec") or {}).get("name") if isinstance(sel, dict) else None
    try:
        inp.model_now = current_model_version()
    except Exception as err:  # noqa: BLE001 - recorded; the state check then cannot match
        inp.notes.append(f"读不出季度模型的当前版本：{type(err).__name__}")
    rid = latest_run() if run_id in (None, "", "latest") else run_id
    if rid is None:
        return inp
    from .narrative import runner as RN
    try:
        d = RN.run_dir(rid)
    except RN.RunnerError as err:
        raise SwitchError(str(err)) from err
    run = _load(d / "run.json")
    if not isinstance(run, dict):
        raise SwitchError(f"没有找到叙事检验的运行「{rid}」")
    rows_path = d / "score_rows.csv"
    rows_sha = None
    if rows_path.exists():
        raw = rows_path.read_bytes()
        rows_sha = _sha(raw)
        inp.rows = pd.read_csv(rows_path)
    inp.stats = _load(d / "stats.json")
    inp.episodes = _load(d / "episodes.json")
    for p in sorted((d / "translations").glob("*.json")) if (d / "translations").is_dir() else []:
        rec = _load(p)
        if isinstance(rec, dict) and rec.get("code") and rec.get("arm"):
            inp.translations[(str(rec["code"]), str(rec["arm"]))] = rec
    st = inp.stats or {}
    inp.run = {"id": rid, "kind": run.get("kind"), "status": run.get("status"), "backend": run.get("backend"),
               "model": run.get("model"), "created": run.get("created"), "label": run.get("label"),
               "narratives": (run.get("settings") or {}).get("narratives"),
               "engines": list((st.get("settings") or {}).get("engines") or (run.get("settings") or {}).get("engines")
                               or []),
               "rows_sha256": rows_sha, "dir": str(d)}
    return inp


# ================================================================ the conditions

def _paired_condition(R, arm: str, hz: str, bound: float, proto: dict, expected, group: str) -> dict:
    from .narrative import stats as ST
    cid = f"{'O' if group == 'oracle' else 'N'}:{arm}:{hz}"
    text = (f"{ARM_ZH.get(arm, arm)}·{H_ZH.get(hz, hz)}：季度模型减月度引擎的股票类 9 个 MAE，"
            f"90% 上界低于 {bound:+.3f}")
    c = {"id": cid, "group": group, "arm": arm, "horizon": hz, "text": text, "bound": bound, "hi90_below": bound,
         "a": f"Q:{arm}", "b": f"E:{arm}", "assets": proto["assets"], "set": proto["set"], "cma": proto["cma"]}
    if R is None or R.empty:
        return {**c, "evaluated": False, "passed": None, "detail": "没有可用的评分行"}
    codes = list(expected) if expected is not None else None
    p = ST.paired(R, (f"Q:{arm}", proto["cma"]), (f"E:{arm}", proto["cma"]), hz, proto["assets"], codes=codes)
    if not p:
        return {**c, "evaluated": False, "passed": None, "n": 0, "detail": "没有两个引擎都有的片段"}
    got = list(p["episodes"])
    c.update(diff=p["diff"], lo90=p["lo90"], hi90=p["hi90"], n=p["n_episodes"], mde=p.get("mde"), episodes=got,
             per_episode=p.get("per_episode"))
    missing = sorted(set(codes or []) - set(got))
    if missing:
        return {**c, "evaluated": False, "passed": None, "missing": missing,
                "detail": f"缺 {len(missing)} 个片段（{_codes_text(missing)}）：比较必须覆盖干净集里应有的每个片段"}
    if p.get("hi90") is None:
        return {**c, "evaluated": False, "passed": None, "detail": f"只有 {p['n_episodes']} 个片段：算不出区间"}
    passed = bool(p["hi90"] < bound)
    return {**c, "evaluated": True, "passed": passed,
            "detail": (f"{_signed(p['diff'])} [{_signed(p['lo90'])}, {_signed(p['hi90'])}]，上界"
                       f"{'低于' if passed else '不低于'} {bound:+.3f}（{p['n_episodes']} 个片段）")}


def _pit_condition(R, proto: dict, expected) -> dict:
    """Q's mean |PIT − 0.5| against E's + margin, on the cells (episode, arm, asset, horizon) both engines have."""
    import numpy as np
    from .narrative import common as C
    margin = proto["pit_margin"]
    arms = proto["pit_arms"] or (proto["oracle_arms"] + proto["narrative_arms"])
    c = {"id": "PIT", "group": "calibration", "bound": margin, "arms": arms, "horizons": proto["horizons"],
         "text": f"校准：季度模型的平均 |PIT − 0.5| 不超过月度引擎的 {margin:+.3f}（同一批单元）"}
    if R is None or R.empty:
        return {**c, "evaluated": False, "passed": None, "detail": "没有可用的评分行"}
    al = C.ASSET_SETS[proto["assets"]]
    sub = R[(R["cma"] == proto["cma"]) & R["horizon"].isin(proto["horizons"]) & R["asset"].isin(al)
            & R["arm"].isin(arms) & R["engine"].isin(["Q", "E"])]
    if expected is not None:
        sub = sub[sub["code"].isin(list(expected))]
    sub = sub[np.isfinite(sub["pit"].astype(float))]
    keys = ["code", "arm", "asset", "horizon"]
    q = sub[sub["engine"] == "Q"].set_index(keys)["pit"].astype(float)
    e = sub[sub["engine"] == "E"].set_index(keys)["pit"].astype(float)
    q = q[~q.index.duplicated()]
    e = e[~e.index.duplicated()]
    ix = q.index.intersection(e.index)
    if len(ix) == 0:
        return {**c, "evaluated": False, "passed": None, "detail": "没有两个引擎都有的单元"}
    dq = (q.loc[ix] - 0.5).abs().groupby(level="code").mean()
    de = (e.loc[ix] - 0.5).abs().groupby(level="code").mean()
    got = sorted(dq.index)
    vq, ve = float(dq.mean()), float(de.mean())
    c.update(q=round(vq, 6), e=round(ve, 6), diff=round(vq - ve, 6), n=len(got), n_cells=int(len(ix)), episodes=got,
             value=round(vq - ve, 6))
    missing = sorted(set(expected or []) - set(got))
    if missing:
        return {**c, "evaluated": False, "passed": None, "missing": missing,
                "detail": f"缺 {len(missing)} 个片段（{_codes_text(missing)}）"}
    passed = bool(vq <= ve + margin)
    return {**c, "evaluated": True, "passed": passed,
            "detail": (f"季度模型 {vq:.3f}，月度引擎 {ve:.3f}，差 {vq - ve:+.3f}"
                       f"{'，不超过' if passed else '，超过'} {margin:+.3f}（{len(got)} 个片段、{len(ix)} 个单元）")}


def _expected(inp: Inputs, R, proto: dict) -> tuple[list | None, dict, list]:
    """(clean codes every oracle comparison must cover, {arm: codes every narrative comparison must cover}, notes)."""
    notes = []
    run_clean = sorted(R[R["set"] == proto["set"]]["code"].unique()) if R is not None and not R.empty else []
    if inp.registry and inp.registry.get("clean") and proto["set"] == "clean":
        clean = list(inp.registry["clean"])
    else:
        clean = run_clean
        notes.append("没有片段登记表：完整性按运行自己的干净集核对")
    version = (inp.run or {}).get("narratives") or "v1"
    by_arm = {}
    if proto.get("narrative_coverage", "clean") == "clean":
        for arm in proto["narrative_arms"]:          # every clean episode, whether or not its narrative is written
            by_arm[arm] = list(clean)
        return clean, by_arm, notes
    for arm in proto["narrative_arms"]:
        n = ARM_NARRATIVE.get(arm)
        if inp.registry and n:
            have = [c for c in clean if n in ((inp.registry["narratives"].get(c) or {}).get(version) or [])
                    or n in ((inp.registry.get("standin", {}).get(c) or {}).get(version) or [])]
        else:
            have = sorted(R[(R["set"] == proto["set"]) & (R["variant"] == f"Q:{arm}")]["code"].unique()) \
                if R is not None and not R.empty else []
        by_arm[arm] = have
    for arm, have in by_arm.items():
        if clean and len(have) < len(clean):
            notes.append(f"叙事臂 {arm} 只覆盖 {len(have)} / {len(clean)} 个干净片段（T-S 记录接受只比较写了叙事的片段）")
    return clean, by_arm, notes


def narrative_block(inp: Inputs, proto: dict) -> str | None:
    """Why the narrative conditions cannot be evaluated on this run (None: they can). Only a real LLM's translations
    can certify the narrative arms; the record may accept the research round's stand-ins explicitly."""
    run = inp.run or {}
    if run.get("kind") == "real":
        return None
    if "standin" in proto.get("narrative_sources", []) or "mock" in proto.get("narrative_sources", []):
        return None
    return MOCK_RUN_TEXT if run.get("kind") == "mock" else "这次运行不是真实大模型的翻译：叙事臂不评估"


def _input_checks(inp: Inputs, R, proto: dict, narr_expected: dict) -> tuple[list[dict], dict]:
    """The IN:* conditions and a provenance summary of the narrative tables."""
    out = []
    run = inp.run
    if run is None:
        out.append({"id": "IN:run", "group": "inputs", "text": "有一次同时用季度模型和月度引擎评分的叙事检验运行",
                    "evaluated": False, "passed": None, "detail": "还没有这样的运行"})
        return out, {}
    ok = run.get("status") == "done" and R is not None and not R.empty and run.get("kind") in ("mock", "real")
    why = []
    if run.get("status") != "done":
        why.append(f"运行状态是「{run.get('status')}」")
    if R is None or R.empty:
        why.append("没有 score_rows.csv")
    if run.get("kind") not in ("mock", "real"):
        why.append("研究轮导入的运行不是这个应用的季度模型与月度引擎")
    kind_zh = {"mock": "模拟后端的运行（不调用模型）", "real": "真实大模型的运行"}.get(run.get("kind"), "导入的运行")
    out.append({"id": "IN:run", "group": "inputs", "text": f"叙事检验运行 {run['id']} 已完成并有评分行",
                "evaluated": True, "passed": ok,
                "detail": "；".join(why) or f"{kind_zh}，{str(run.get('created') or '')[:10]}"})
    engines = sorted(set(R["engine"].unique())) if R is not None and not R.empty else []
    out.append({"id": "IN:engines", "group": "inputs", "text": "两个引擎都评了分（季度模型 Q 与月度引擎 E）",
                "evaluated": True, "passed": {"Q", "E"} <= set(engines),
                "detail": "引擎：" + ("、".join(engines) or "—")})
    # E ran the selected F; one quarterly-stack version across the run
    eps = inp.episodes if isinstance(inp.episodes, dict) else {}
    e_models = sorted({str((v.get("engine_E") or {}).get("model")) for v in eps.values()
                       if isinstance(v, dict) and isinstance(v.get("engine_E"), dict)
                       and not (v.get("engine_E") or {}).get("skipped")})
    if not e_models:
        m_ok, m_detail = False, "运行里没有记下月度引擎跑的是哪个模型"
    elif inp.selected_model:
        m_ok = e_models == [inp.selected_model]
        m_detail = ("与因子轮选中的 F 一致" if m_ok else
                    f"月度引擎跑的不是因子轮选中的 F（运行里有 {len(e_models)} 个模型名，与选中的不同）")
    else:
        m_ok = all(m not in ("", "base", "None") for m in e_models)
        m_detail = "没有因子轮的选择记录：只核对不是研究轮的基础模型" + ("" if m_ok else "，而运行用的正是基础模型")
    out.append({"id": "IN:model", "group": "inputs", "text": "月度引擎跑的是因子轮选中的模型 F（不是研究轮用的 base）",
                "evaluated": True, "passed": m_ok, "detail": m_detail})
    sem = sorted({s for s in (semantic_version(v.get("stack")) for v in eps.values() if isinstance(v, dict)) if s})
    s_ok = len(sem) == 1
    out.append({"id": "IN:stack", "group": "inputs", "text": "整次运行用同一个版本的季度模型",
                "evaluated": True, "passed": s_ok,
                "detail": ("一个版本" if s_ok else f"有 {len(sem)} 个版本") if sem else "运行里没有记下季度模型的版本"})
    # the narrative tables
    allowed = set(proto["narrative_sources"])
    prov = {}
    bad = []
    for arm, codes in narr_expected.items():
        for code in codes:
            k = _classify(inp.translations.get((code, arm)))
            prov[k] = prov.get(k, 0) + 1
            if k not in allowed:
                bad.append(f"{code}·{arm}（{SOURCE_ZH.get(k, k)}）")
    detail = "；".join(f"{SOURCE_ZH.get(k, k)} {n} 张" for k, n in sorted(prov.items())) or "没有叙事表"
    if bad:
        detail += "。不能用的：" + "、".join(bad[:6]) + ("等" if len(bad) > 6 else "")
    out.append({"id": "IN:sources", "group": "inputs",
                "text": "叙事臂用的是翻译（" + " 或 ".join(SOURCE_ZH.get(s, s) for s in proto["narrative_sources"])
                        + "），不是最小表", "evaluated": True, "passed": not bad and bool(prov), "detail": detail})
    return out, prov


def _cross_check(inp: Inputs, conds: list[dict]) -> list[str]:
    """stats.json's ts_inputs (P7) against the recomputed comparisons: a mismatch means stats.json is older than the
    rows; the rows win."""
    ts = (inp.stats or {}).get("ts_inputs") if isinstance(inp.stats, dict) else None
    if not isinstance(ts, dict):
        return []
    bad = []
    for c in conds:
        if c["group"] not in ("oracle", "narrative") or c.get("diff") is None:
            continue
        ref = (ts.get("oracle" if c["group"] == "oracle" else "narrative") or {}).get(f"{c['arm']}|{c['horizon']}")
        if not isinstance(ref, dict):
            continue
        for k in ("diff", "lo90", "hi90"):
            a, b = _num(c.get(k)), _num(ref.get(k))
            if a is not None and b is not None and abs(a - b) > 1e-5:
                bad.append(c["id"])
                break
    if bad:
        return [f"stats.json 的 T-S 输入与 score_rows.csv 重算的不一致（{'、'.join(bad[:4])}）：stats.json 可能是旧的，以重算为准"]
    return []


def gate(inp: Inputs) -> dict:
    """The T-S result from its inputs (pure: no file is read or written here)."""
    proto, pinfo = protocol(inp.record)
    R = inp.rows
    if R is not None and not R.empty:
        R = R[R["set"] == proto["set"]] if "set" in R.columns else R
    conds = [tf_condition(inp.tf)]
    notes = list(inp.notes) + pinfo["notes"]
    clean, narr_expected, enotes = _expected(inp, inp.rows, proto) if inp.rows is not None else ([], {}, [])
    notes += enotes
    checks, prov = _input_checks(inp, inp.rows, proto, narr_expected)
    for arm in proto["oracle_arms"]:
        for hz in proto["horizons"]:
            conds.append(_paired_condition(R, arm, hz, proto["oracle_hi90_below"], proto, clean, "oracle"))
    conds.append(_pit_condition(R, proto, clean))
    block = narrative_block(inp, proto) if inp.run is not None else None
    for arm in proto["narrative_arms"]:
        for hz in proto["horizons"]:
            c = _paired_condition(R, arm, hz, proto["narrative_hi90_below"], proto, narr_expected.get(arm), "narrative")
            if block:                                  # computed for the record, never certified
                c = {**c, "evaluated": False, "passed": None, "computed": c.get("detail"), "detail": block}
            elif c.get("n") is not None and clean and proto.get("narrative_coverage") != "clean":
                c = {**c, "detail": f"{c.get('detail')}；覆盖 {c['n']} / {len(clean)} 个干净片段"}
            conds.append(c)
    if pinfo["problems"]:
        checks.append({"id": "IN:protocol", "group": "inputs", "text": "T-S 记录的协议读得懂", "evaluated": True,
                       "passed": False, "detail": "；".join(pinfo["problems"])})
    if inp.record is None:
        checks.insert(0, {"id": "IN:record", "group": "inputs", "text": "有冻结的 T-S 预注册记录",
                          "evaluated": False, "passed": None,
                          "detail": "记录还不在：预注册那一轮合并之后才有"})
    notes += _cross_check(inp, conds)
    allc = checks + conds
    failed = [c for c in allc if c["evaluated"] and c["passed"] is False]
    missing = [c for c in allc if not c["evaluated"]]
    passed = not failed and not missing and inp.record is not None
    when = _now()
    run = dict(inp.run or {})
    run.pop("dir", None)
    eps = inp.episodes if isinstance(inp.episodes, dict) else {}
    sem = sorted({s for s in (semantic_version(v.get("stack")) for v in eps.values() if isinstance(v, dict)) if s})
    result = {
        "schema": SCHEMA, "evaluator": EVALUATOR, "version": VERSION, "test": "TS", "evaluated_at": when,
        "status": "passed" if passed else "failed", "passed": passed, "incomplete": bool(missing),
        "record": ({"id": inp.record["id"], "file": inp.record.get("file"), "sha256": inp.record.get("sha256"),
                    "protocol_sha256": inp.record.get("protocol_sha256") or protocol_digest(inp.record.get("data")),
                    "date": inp.record.get("date")} if inp.record else None),
        "protocol": proto, "protocol_source": pinfo["source"], "protocol_from_record": pinfo["from_record"],
        "required": required_ids(proto), "run": run or None,
        "tf": ({k: v for k, v in inp.tf.items() if k != "text"} if inp.tf else None),
        "model": {"semantic": sem[0] if len(sem) == 1 else None, "run_versions": sem, "now": inp.model_now,
                  "engine_E": inp.selected_model},
        "translations": prov, "conditions": conds, "checks": checks,
        "notes": notes, "commands": COMMANDS,
    }
    result["reasons"] = reasons(result)
    result["text"] = result_text(result)
    return result


def _cond_line(c: dict) -> str:
    return f"「{c['text']}」（{c['detail']}）" if c.get("detail") else f"「{c['text']}」"


def reasons(result: dict) -> list[str]:
    """One Chinese line per input or condition that failed or could not be evaluated, with its numbers."""
    out = []
    for c in (result.get("checks") or []) + (result.get("conditions") or []):
        if c.get("evaluated") and c.get("passed") is False:
            out.append(f"未满足：{c['text']}（{c.get('detail') or '—'}）")
        elif not c.get("evaluated"):
            out.append(f"不能评估：{c['text']}（{c.get('detail') or '缺少输入'}）")
    return out


def result_text(result: dict) -> str:
    when = str(result.get("evaluated_at") or "")[:10]
    if result.get("passed"):
        return (f"切换关 T-S 通过（{when}）：13 个内置情景、未来一年页、组合评估与提取的默认引擎换成季度模型；"
                "月度引擎留作对照。")
    allc = (result.get("checks") or []) + (result.get("conditions") or [])
    failed = [c for c in allc if c.get("evaluated") and c.get("passed") is False]
    missing = [c for c in allc if not c.get("evaluated")]
    parts = []
    if failed:
        parts.append("未满足的条件：" + "；".join(_cond_line(c) for c in failed[:6])
                     + (f"；另有 {len(failed) - 6} 条" if len(failed) > 6 else ""))
    if missing:
        parts.append(f"{len(missing)} 个条件缺少输入、不能评估：" + "；".join(f"「{c['text']}」" for c in missing[:4])
                     + ("等" if len(missing) > 4 else ""))
    return (f"切换关 T-S 没有通过（{when}），" + "。".join(parts) + "。默认引擎仍是月度引擎，季度模型继续作试运行。")


def evaluate(run_id: str | None = "latest", *, record_id: str | None = None, results_dir: Path | None = None) -> dict:
    """evaluate the gate on a narrative run (default: the newest finished run with both engines)."""
    return gate(load_inputs(run_id, record_id=record_id, results_dir=results_dir))


def write_result(result: dict, *, results_dir: Path | None = None, append: bool = False) -> dict:
    """Write the result as results/switch_gate.json (atomic) and, with append, as one JSON line appended to the
    T-S record's sibling <stem>.results.jsonl (the frozen record itself is never edited)."""
    if result.get("schema") != SCHEMA:
        raise SwitchError("只能写本模块评估出的结果")
    if not result.get("record"):
        raise SwitchError("没有冻结的 T-S 预注册记录：评估结果不能作为切换的依据，没有写入")
    if append and (result.get("run") or {}).get("kind") != "real":
        raise SwitchError("模拟后端的运行只是流程检验：结果不追加到 T-S 记录的结果文件（只有真实大模型翻译的运行才追加）")
    out = {"gate_file": str(jsonio.dump(result, Path(results_dir or RESULTS_DIR) / GATE_FILE))}
    if append:
        rec = ts_record(result["record"]["id"])
        if rec is None or rec.get("path") is None:
            raise SwitchError(f"找不到 T-S 记录「{result['record']['id']}」：没有追加")
        if rec.get("protocol_sha256") != result["record"].get("protocol_sha256"):
            raise SwitchError("T-S 记录在评估之后改过：没有追加，请重新评估")
        side = Path(rec["path"]).with_name(f"{Path(rec['path']).stem}.results.jsonl")
        with open(side, "a", encoding="utf-8") as fh:
            fh.write(jsonio.dumps(result) + "\n")
        out["appended"] = str(side)
    clear_cache()
    return out


# ================================================================ the state every consumer reads

def _recheck(res: dict) -> list[str]:
    """Why a stored passed result is not internally consistent ([] when it is)."""
    probs = []
    conds = res.get("conditions") or []
    have = {c.get("id") for c in conds if isinstance(c, dict)}
    need = res.get("required") or []
    lost = [i for i in need if i not in have]
    if not need or lost:
        probs.append("结果缺少应有的条件：" + ("、".join(lost[:6]) if lost else "没有列出应有的条件"))
    for c in list(conds) + list(res.get("checks") or []):
        if not isinstance(c, dict) or not c.get("evaluated") or c.get("passed") is not True:
            probs.append(f"条件 {c.get('id') if isinstance(c, dict) else '?'} 没有评估或没有通过")
            continue
        hi, b = _num(c.get("hi90")), _num(c.get("bound"))
        if c.get("group") in ("oracle", "narrative") and (hi is None or b is None or not hi < b):
            probs.append(f"条件 {c['id']} 的上界与门槛对不上")
        if c.get("group") == "calibration":
            q, e = _num(c.get("q")), _num(c.get("e"))
            if q is None or e is None or b is None or q > e + b + 1e-12:
                probs.append(f"条件 {c['id']} 的数字与门槛对不上")
    if not res.get("record"):
        probs.append("结果没有对应的 T-S 预注册记录")
    return probs


def _foreign(res: dict) -> dict:
    """A result this module did not write (hand-written, or another round's placeholder): shown, never trusted to
    switch. Its own failed verdict still reads as failed."""
    EV = _ev()
    conds = EV.conditions(res.get("conditions") or res.get("gate") or res.get("checks") or [])
    said = EV._passed(res)
    if said is None and conds:
        said = all(c["passed"] is True for c in conds)
    failed = [c for c in conds if c.get("passed") is False]
    if said is False or failed:
        text = res.get("text") if isinstance(res.get("text"), str) and res.get("text") else (
            "切换关 T-S 没有通过，未满足的条件：" + "；".join(_cond_line(c) for c in failed)
            + "。默认引擎仍是月度引擎，季度模型继续作试运行。")
        return {"status": "failed", "conditions": conds, "text": text, "reasons": [f"未满足：{c['text']}" for c in failed]}
    return {"status": "invalid", "conditions": conds,
            "text": ("切换关的这条结果不是评估工具写的（没有版本、应有的条件和对应的预注册记录）：不据此切换。"
                     "默认引擎仍是月度引擎。"),
            "reasons": ["这条结果不是 python -m scenariolab.evaluation.switch evaluate 写的：不据此切换"]}


def _candidates(results_dir: Path) -> list[dict]:
    """Stored T-S results: the newest T-S record's appended results (latest one) and results/switch_gate.json."""
    out = []
    try:
        rec = ts_record()
    except Exception:  # noqa: BLE001 - an unreadable record shows in the evaluation tab's G section
        rec = None
    # a record's status snapshots and disclosure notes (P5 appends a 'status' entry while T-S is not evaluable) are
    # not evaluations: only the other entries can switch anything
    evals = [x for x in ((rec or {}).get("results") or []) if not (isinstance(x, dict) and x.get("kind") in
                                                                    ("status", "note"))]
    if rec and evals:
        out.append({"kind": "record", "id": rec["id"], "result": evals[-1], "record": rec})
    d = _load(Path(results_dir) / GATE_FILE)
    if isinstance(d, dict) and d:
        out.append({"kind": "file", "id": GATE_FILE, "result": d, "record": rec})
    return out


def _scan(results_dir: Path) -> list[str]:
    """Why nothing has been evaluated yet: the record, T-F and the narrative runs, as Chinese lines."""
    lines = []
    try:
        rec = ts_record()
    except Exception:  # noqa: BLE001
        rec = None
    if rec is None:
        lines.append("切换关 T-S 的预注册记录（冻结的协议与门槛）还不在：预注册那一轮合并之后才能评估")
    else:
        lines.append(f"T-S 已预注册（{rec.get('date') or rec['id']}），还没有评估结果")
    tf = tf_result(results_dir)
    if tf is None:
        lines.append("还没有因子引擎检验 T-F 的结果（qmodel 构建步骤写 qmodel.json）")
    elif not tf["passed"]:
        c = tf_condition(tf)
        lines.append(f"因子引擎检验 T-F 现在没有通过：{c['detail']}")
    try:
        both = [r for r in list_runs() if r["status"] == "done" and r["scored"] and {"Q", "E"} <= set(r["engines"])]
    except Exception:  # noqa: BLE001
        both = []
    if not both:
        lines.append("还没有同时用季度模型和月度引擎评分的叙事检验运行")
    else:
        lines.append(f"已有 {len(both)} 次两个引擎都评了分的叙事检验运行（最新 {both[0]['id']}）")
    return lines


_CACHE: dict = {}
_LOCK = threading.Lock()


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _stamp(results_dir: Path) -> tuple:
    EV = _ev()
    files = [Path(results_dir) / GATE_FILE, Path(results_dir) / "qmodel.json"]
    for d in EV.PREREG_DIRS:
        d = Path(d)
        if d.is_dir():
            files += sorted(d.glob("*_TS*.json*")) + sorted(d.glob("*_TF*.json*"))
    try:
        from .narrative import common as C
        root = C.runs_dir()
        files.append(root)
    except Exception:  # noqa: BLE001
        pass
    out = []
    for p in files:
        try:
            st = p.stat()
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), None, None))
    return tuple(out)


def state(results_dir: Path | str | None = None) -> dict:
    """The switch as every consumer reads it (module docstring). Cached on the input files' mtimes."""
    results_dir = Path(results_dir or RESULTS_DIR)
    key = (str(results_dir), _stamp(results_dir))
    with _LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return json.loads(json.dumps(hit))
    out = _state(results_dir)
    with _LOCK:
        _CACHE.clear()
        _CACHE[key] = out
    return json.loads(json.dumps(out))


def _state(results_dir: Path) -> dict:
    base = {"schema": STATE_SCHEMA, "commands": COMMANDS, "command_note": COMMAND_NOTE, "consumers": CONSUMER_ZH}
    cands = _candidates(results_dir)
    mine = [c for c in cands if isinstance(c["result"], dict) and c["result"].get("schema") == SCHEMA]
    chosen = max(mine, key=lambda c: (str(c["result"].get("evaluated_at") or ""), c["kind"] == "record")) if mine \
        else (cands[0] if cands else None)
    if chosen is None:
        st = {"status": "not_evaluated", "text": not_evaluated_text(), "conditions": [], "reasons": _scan(results_dir),
              "source": None, "evaluated_at": None}
    elif chosen["result"].get("schema") != SCHEMA:
        st = {**_foreign(chosen["result"]), "source": {"kind": chosen["kind"], "id": chosen["id"]},
              "evaluated_at": chosen["result"].get("date") or chosen["result"].get("evaluated_at")}
    else:
        res = chosen["result"]
        st = {"status": res.get("status"), "text": res.get("text") or result_text(res),
              "conditions": (res.get("checks") or []) + (res.get("conditions") or []),
              "reasons": list(res.get("reasons") or reasons(res)), "source": {"kind": chosen["kind"], "id": chosen["id"]},
              "evaluated_at": res.get("evaluated_at"), "run": res.get("run"), "protocol": res.get("protocol"),
              "record": res.get("record"), "model": res.get("model"), "notes": res.get("notes") or []}
        if res.get("passed") is True:
            probs = _recheck(res)
            if probs:
                st.update(status="invalid", reasons=probs,
                          text="切换关的结果前后不一致：" + "；".join(probs[:3]) + "。不据此切换，默认引擎仍是月度引擎。")
            else:
                stale = []
                rec = chosen.get("record")
                want = res.get("record") or {}
                if rec is None or rec.get("id") != want.get("id") \
                        or rec.get("protocol_sha256") != want.get("protocol_sha256"):
                    stale.append("评估所依据的 T-S 预注册记录不见了或改过")
                sem = (res.get("model") or {}).get("semantic")
                try:
                    now = current_model_version()
                except Exception:  # noqa: BLE001
                    now = None
                if not sem or sem != now:
                    stale.append(f"评估之后季度模型的版本变了（评估时 {sem or '—'}，现在 {now or '—'}）")
                if stale:
                    st.update(status="stale", reasons=stale,
                              text="切换关的结果过期了：" + "；".join(stale) + "。请重新评估；在那之前默认引擎仍是月度引擎。")
        elif res.get("status") == "passed":
            st.update(status="invalid", reasons=["结果写着通过，但 passed 不是 true"],
                      text="切换关的结果前后不一致：不据此切换，默认引擎仍是月度引擎。")
    on = st["status"] == "passed"
    defaults = {k: ("quarterly" if on else "monthly") for k in CONSUMERS}
    return {**base, **st, "passed": on, "quarterly": on, "default_engine": "quarterly" if on else "monthly",
            "label": DEFAULT_TAG if on else TRIAL, "defaults": defaults,
            "default_text": ("默认引擎：季度模型（切换关 T-S 已通过）" if on
                             else "默认引擎：月度引擎（季度模型是试运行：切换关 T-S "
                             + {"not_evaluated": "还没有评估", "failed": "没有通过", "stale": "的结果过期了",
                                "invalid": "的结果无效"}.get(st["status"], "没有通过") + "）")}


def public(st: dict | None = None) -> dict:
    """The state for the pages (GET /api/evaluation/switch): everything but the per-episode numbers."""
    st = st if st is not None else state()
    out = dict(st)
    out["conditions"] = [{k: v for k, v in c.items() if k not in ("per_episode", "episodes", "failed")}
                         for c in st.get("conditions") or []]
    return out


# ================================================================ endpoint

def _router():
    from fastapi import APIRouter
    from fastapi.responses import Response
    r = APIRouter(prefix="/api/evaluation", tags=["switch"])

    @r.get("/switch")
    def switch_state():
        """The switch as every page reads it (the default engine and why), GET /api/evaluation/switch."""
        return Response(content=jsonio.dumps(public()), media_type="application/json")

    return r


router = _router()


# ================================================================ CLI

def _print(obj) -> None:
    print(json.dumps(jsonio.clean(obj), ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scenariolab.evaluation.switch", description="切换关 T-S（SIM_DESIGN §4.6）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="the switch as the app reads it (no evaluation)")
    sub.add_parser("inputs", help="what an evaluation would read, and what is missing")
    ev = sub.add_parser("evaluate", help="evaluate T-S on a narrative run with both engines")
    ev.add_argument("--run", default="latest", help="narrative run id, or latest (the newest finished run with Q and E)")
    ev.add_argument("--record", help="T-S record id (default: the newest)")
    ev.add_argument("--write", action="store_true", help="write data/results/switch_gate.json")
    ev.add_argument("--append", action="store_true", help="also append the result to the record's .results.jsonl")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        st = state()
        print(st["text"])
        for line in st.get("reasons") or []:
            print(f"  · {line}")
        print(st["default_text"])
        return 0
    if a.cmd == "inputs":
        for line in _scan(Path(RESULTS_DIR)):
            print(f"  · {line}")
        print(COMMAND_NOTE)
        for c in COMMANDS:
            print(f"    {c}")
        return 0
    try:
        res = evaluate(a.run, record_id=a.record)
    except SwitchError as err:
        print(f"[错误] {err}", file=sys.stderr)
        return 2
    print(res["text"])
    for line in res["reasons"]:
        print(f"  · {line}")
    if a.write or a.append:
        try:
            _print(write_result(res, append=a.append))
        except SwitchError as err:
            print(f"[错误] {err}", file=sys.stderr)
            return 2
    else:
        print("（只是评估，没有写入：加 --write 写 data/results/switch_gate.json，加 --append 追加到 T-S 记录的结果文件）")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
