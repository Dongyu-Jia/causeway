"""Narrative -> QTable/1: the production translator of the quarterly stack (SIM_DESIGN §7.2, Appendix B; §14 P6).

The LLM's output schema is the model's input schema. The model writes a QTable (scenariolab/qmodel/qtable.py):
for each quarter the fields the text states, each with a quote from the narrative, the financial-conditions stance
and the c_q block. qtable.validate is the guard rail: refusals are removed, the R1-R5 gaps are reported with the
narrative's evidence sentence, and the quarterly stack completes everything the text does not state.

System prompt, in the order of §7.2:
    1. the role                       2. the card's x_q / u / c_q fields (compact; definitions; quote keywords)
    3. the size-bin table              4. the generic templates G01-G35 (production: signs and sd sizes;
    5. the rules (decomposition into generic channels, R1-R6, refusals,      test: signs only, the blind view)
       quotes, quarter indices)       6. the state table (production: today's levels; test: percentile bands)
    7. the output format (a compact JSON schema, one line, under "## 输出格式（JSON schema）")
plus three synthetic few-shot examples (a demand recession, an oil supply shock, a banking panic) with no years,
names or dated events. The user message is the narrative only (production may add the requested length).

Modes: "production" (the lab) and "test" (the narrative test: the blind schema view, percentile bands, no
template sizes; prompt_scan() must find no year token, month count, quarter label or footprint in it).

Calls (translate): MSL_LLM_BACKEND claude_code (the local Claude Code CLI on the user's own plan login) or mock
(every test); the api backend is not offered for QTables. Answers are cached in data/llm_cache/ by (backend, model,
system prompt, user message), as the monthly translator does (mock answers are not cached), within one of the
process-wide call slots. What each failure maps to (the agents' kinds, claude_code.KINDS):

    answer not JSON / not a readable QTable / form refusals ... one format repair call, then the fallback
    refusal (stop_reason or plain text), max_tokens, context, empty, failed, timeout, 429 / 529 / 5xx ... fallback
    usage limit, login, token, CLI missing, model not allowed ...... QTablePause (the caller pauses; resumable)
    cancel ........................................................... QTablePause("cancelled")

The fallback is the minimal QTable (every field absent, fc 未说明): the run is the model's unconditional
completion, and the result says why. A real call happens only from an explicit, confirmed action (the API needs
confirm: true and shows estimate() first; the CLI asks on a TTY); tests and builds use the mock.

    python -m scenariolab.llm.qtranslator --print-prompt [--mode test]
    python -m scenariolab.llm.qtranslator "霍尔木兹海峡被封锁……" [--quarters 8] [--yes]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time

from ..qmodel import card as qcard
from ..qmodel import config_q as cq
from ..qmodel import qtable as QT
from . import claude_code, jsonfix
from . import translator as tr

SCHEMA_HEADER = "## 输出格式（JSON schema）"         # the header MockClaude reads the schema under
MODE_LINE = {"production": "模式：生产（情景实验室）", "test": "模式：叙事检验（盲测）"}
PROMPT_VERSION = "qtable-prompt-2"          # 2: the test mode's synthetic decomposition example and blind templates
EFFORT = "high"
BACKENDS = ("claude_code", "mock")
PAUSE_KINDS = frozenset({"limit", "login", "token", "cli_missing", "model", "forbidden"})
FALLBACK_KINDS = frozenset({"refusal", "max_tokens", "context", "empty", "failed", "timeout", "rate", "overloaded",
                            "server", "unparseable", "max_turns"})
REFUSAL_TEXT = re.compile(r"I can(?:'|no)t help|I won't|I'm not able to|I am not able to|无法协助|不能协助|"
                          r"无法完成这个请求|不能提供这类|我不能翻译")
LENGTH_RE = (re.compile(r"情景长度：\s*([0-9]+)\s*个季度"), re.compile(r"覆盖期约\s*([0-9]+)\s*个季度"))
OUTPUT_TOKENS = (3000, 8000)                         # SIM_DESIGN §8.5: 3-8k output tokens per translation
SECONDS = (20, 60)
LABEL = "季度情景表翻译（QTable/1）"
E_API = "季度情景表的翻译只用本机 Claude Code（MSL_LLM_BACKEND=claude_code）或模拟后端（mock）：不走 Anthropic API。"
E_CONFIRM = ("这次翻译要调用本机 Claude Code（计入你的订阅额度，约 {calls} 次调用）：请先看预计用量，再确认（confirm: true）。"
             "模拟后端与已缓存的翻译不需要确认。")
FALLBACK_NOTE = "翻译没有得到可用的情景表（{why}）：按“文字没有给”处理，全部由模型补全"
CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")
# the state table: (panel column, 中文名, unit, how to show it: "level" / "exp" (log level shown as its level) /
# "band" (a percentile band only, also in production: a log level means nothing to a translator))
STATES = (("s_ff", "政策利率（联邦基金利率）", "%", "level"), ("s_unrate", "失业率", "%", "level"),
          ("s_cpi_yoy", "CPI 同比", "%", "level"), ("s_core_yoy", "核心 CPI 同比", "%", "level"),
          ("s_y10", "10 年期收益率", "%", "level"), ("s_term", "期限利差（10 年 − 政策利率）", "pp", "level"),
          ("s_spread", "信用利差（Baa − 10 年）", "pp", "level"), ("s_gz", "GZ 信用利差", "pp", "level"),
          ("s_tp10", "10 年期限溢价", "pp", "level"), ("s_infexp1", "短期通胀预期（密歇根 1 年）", "%", "level"),
          ("s_cape", "席勒 CAPE", "倍", "exp"), ("s_oil_real", "实际油价", "", "band"),
          ("s_usd_real", "实际美元", "", "band"))
# the decomposition example of the rules: production keeps the spec's own (§7.2); the blind test mode gets a
# synthetic one (a generic demand recession with many historical instances), never a dated episode's arc
DECOMPOSE_EXAMPLE = {
    "production": "疫情封锁 → 服务需求塌陷、劳动供给下降、转移支付、降息到下限、扩表、信用利差先升后降",
    "test": "企业投资收缩 → 产出下降、失业上升、通胀回落、信用利差走阔、政策利率随后下调"}
# test mode (§7.2, §8.4): templates with a single historical instance are left out (their channels and phases date the
# episode), and names carrying an era or event word become generic; phase labels lose their durations (「（1–2 季）」)
BLIND_DROP = ("G01",)
BLIND_RENAME = {"G03": "石油供给冲击", "G10": "房地产繁荣后的下跌", "G30": "估值重估（倍数部分）", "G31": "全球利率冲击"}
DURATION = re.compile(r"[（(][^（）()]*(?:季|月|年)[^（）()]*[）)]")
BANDS = ((5, "极低（历史 5 分位以下）"), (20, "低（历史 5–20 分位）"), (40, "偏低（历史 20–40 分位）"),
         (60, "中等（历史 40–60 分位）"), (80, "偏高（历史 60–80 分位）"), (95, "高（历史 80–95 分位）"),
         (101, "极高（历史 95 分位以上）"))


class QTablePause(Exception):
    """The translation must pause (usage limit, login, CLI missing, model not allowed, cancelled): resumable.
    kind is claude_code's error kind; reset_at the epoch a usage limit resets (when the CLI said)."""

    def __init__(self, kind: str, message: str, reset_at: float | None = None):
        super().__init__(message)
        self.kind = kind
        self.reset_at = reset_at


def check_mode(mode: str) -> str:
    if mode not in QT.MODES:
        raise tr.TranslatorError(f"mode 只能是 production 或 test，收到「{mode}」。")
    return mode


def tokens(text: str) -> int:
    """The design docs' convention: one token per CJK character, one per four other characters."""
    s = str(text or "")
    cjk = len(CJK.findall(s))
    return cjk + (len(s) - cjk + 3) // 4


# ================================================================ the output schema (compact, for the prompt)

def prompt_schema(card: dict | None = None) -> dict:
    """QTable/1 for the prompt: the card-generated schema with one entry definition for every field (a property
    name list instead of 51 repeated objects) and without bounds (jsonfix.transport_schema); qtable.validate checks
    the full schema."""
    card = card or qcard.load_card()
    full = jsonfix.transport_schema(qcard.qtable_json_schema(card))
    rows = full["properties"]["rows"]["items"]
    xprops = rows["properties"]["x"]["properties"]
    entry = next(v for k, v in xprops.items() if k != "x_oil_cause")
    rows["properties"]["x"] = {"type": "object", "propertyNames": {"enum": sorted(xprops)},
                               "properties": {"x_oil_cause": xprops["x_oil_cause"]}, "additionalProperties": entry}
    uprops = rows["properties"]["u"]["properties"]
    rows["properties"]["u"] = {"type": "object", "propertyNames": {"enum": sorted(uprops)},
                               "additionalProperties": entry}
    an = full["properties"]["c_q"]["properties"]["anchors"]
    one = next(iter(an["properties"].values()))
    full["properties"]["c_q"]["properties"]["anchors"] = {"type": "object", "propertyNames": {"enum": list(an["properties"])},
                                                          "additionalProperties": one}
    return full


def schema_section(card: dict | None = None) -> str:
    return "\n".join([SCHEMA_HEADER, "",
                      "回答只有一个 JSON 对象，符合下面的 JSON schema：不要放进 ```json 代码块，前后不要写任何其他文字。"
                      "rows[].x 的键只能是 propertyNames 列出的字段，每个值是一个条目对象。", "",
                      jsonfix.compact_json(prompt_schema(card))])


# ================================================================ prompt sections

def role_section(mode: str) -> str:
    lines = ["# 角色", "",
             "你是宏观情景的翻译器。你把一段情景叙事翻译成季度情景表 QTable/1：它就是季度模型的输入。",
             "你只翻译文字说了什么。文字没有说的字段一律不写（缺省 = 没有说明），季度模型会用历史联合分布补全；"
             "不要猜测、不要补充文字没有给的数值，也不要预测资产价格。", MODE_LINE[mode]]
    if mode == "test":
        lines.append("这是盲测：不要猜测叙事对应的年代，输出的任何字符串里都不要写年份或月份计数。")
    return "\n".join(lines)


def field_section(card: dict, mode: str) -> str:
    """The card's compact field list (card.prompt_fields), the definitions (production: the card's; test: the blind
    schema view's, every dated clause removed) and the words a quote must name for the stated-only fields."""
    out = ["# 字段（季度模型的输入：x_q 字段、政策冲击 u、c_q 锚）", "", qcard.prompt_fields(card, mode), "",
           "## 定义（每个字段对应一个实测序列；文字到字段的映射就是这个定义）"]
    if mode == "test":
        view = json.loads(cq.SCHEMA_VIEW_PATH.read_text(encoding="utf-8"))
        defs = {f["id"]: f.get("definition") or "" for f in view["fields"]}
    else:
        defs = {r["id"]: r.get("definition") or "" for r in card["rows"]}
    for r in card["rows"]:
        if r["group"] in ("x", "u") and defs.get(r["id"]):
            out.append(f"- {r['id']}：{defs[r['id']]}")
    out += ["", "## 只在文字明说时才写的字段（市场价格、商品价格、调查与新闻指数）：引文必须提到下列词之一"]
    for r in qcard.rows(card, "x"):
        if r["permission"] == "stated_only":
            out.append(f"- {r['id']}（{r['zh']}）：{' / '.join(r['keywords'])}")
    return "\n".join(out)


def size_section() -> str:
    lines = ["# 定性幅度", "",
             "文字只给方向和定性的程度（“大幅”“小幅”“温和”）时，写 sign + size，不写数字："]
    zh = {"小": "小幅、略微", "温和": "温和、有所", "明显": "明显、显著", "大幅": "大幅、急剧", "极端": "极端、史无前例"}
    for k, v in cq.SIZE_BINS.items():
        lines.append(f"- {k} = {v:g} sd（{zh[k]}）")
    lines.append("sd 是该字段季度变动的训练期标准差，由模型换算；幅度档只有这五个。")
    return "\n".join(lines)


def template_section(mode: str) -> str:
    """G01-G35. Production: each template's channels with their signs and sd sizes (schema.json's `x`); test: the
    blind view's signs only (no sizes, no dated footprints, no footprint quarters)."""
    out = ["# 通用模板（把事件拆成通用渠道；名字不是输入，只有足迹是）", ""]
    if mode == "test":
        for t in blind_templates():
            parts = []
            for ph in t.get("phases") or []:
                signs = "，".join(f"{f} {s}" for f, s in ph["signs"].items())
                parts.append(f"{ph['phase']}：{signs}" if ph.get("phase") else signs)
            out.append(f"- {t['id']} {t['name']}：{'｜'.join(parts)}")
        out.append("（盲测只给方向：+ 上升，- 下降，0 基本不变，? 视情况）")
    else:
        schema = qcard.load_schema()
        for t in schema["templates"]:
            out.append(f"- {t['id']} {t['zh']}：{t.get('x', '')}")
    return "\n".join(out)


def blind_templates() -> list[dict]:
    """The blind schema view's templates as the test-mode prompt shows them: without BLIND_DROP, with BLIND_RENAME,
    and with phase labels stripped of durations (a phase named only by its duration keeps no label)."""
    view = json.loads(cq.SCHEMA_VIEW_PATH.read_text(encoding="utf-8"))
    out = []
    for t in view["templates"]:
        if t["id"] in BLIND_DROP:
            continue
        phases = []
        for ph in t.get("phases") or []:
            label = DURATION.sub("", str(ph.get("phase") or "")).strip()
            phases.append({**ph, "phase": label})
        out.append({**t, "name": BLIND_RENAME.get(t["id"], t["name"]), "phases": phases})
    return out


def rules_section(mode: str) -> str:
    rules = [
        "每个事件都拆成至少有两次历史先例的通用渠道（上面的模板）；在 events 里写它用了哪些模板和渠道（channels），"
        f"how_decomposed 用一句话写怎么拆的，例如「{DECOMPOSE_EXAMPLE[mode]}」。"
        "没有通用足迹的事件（例如一家公司的财务造假）只写进 events（generic_footprint: false）或 notes，不是输入。",
        "季度用序号 q = 1, 2, …（第 1 季是情景的第一个季度），不写日期。",
        "每个条目三选一：value（文字里的数字；可带 sd 表示“大约”）；sign + size（文字只有方向和定性幅度）；"
        "min / max（文字给了区间）。每个条目都写 src：stated（文字里的数字）、stated_qual（定性的文字）、"
        "derived（对文字数字的算术，例如把同比路径换成季度变动）；以及 quote：叙事原文的一段（不超过 120 字，必须逐字出现在叙事里）。"
        "文字没有数字时不能写数字：src 是 stated 或 derived 的 value / min / max，引文里必须有数字；“不变”“按兵不动”写 sign \"0\"。",
        "单位按字段表：%年化、pp/季、log%/季（100 × 对数变动）等。例如“一个季度内上涨约三成”= d_oil value 26（log%/季）。",
        "金融条件 fc.stance 每个季度必填：明显收紧 / 收紧 / 中性 / 宽松 / 明显宽松 / 未说明；不是“未说明”时 quote 必须提到信用利差、"
        "金融条件或融资条件。文字没有说就写“未说明”。",
        "R1 政策：写 d_ff（因经济而动的利率变动），或写 u_ff（文字写明的干预：紧急或临时降息、意外加息、政策体制变化）。"
        "「因为衰退而降息」写衰退的宏观（失业率、产出），利率留给反应函数；文字明说「由反应函数决定」时写进 notes。",
        "R2 增长：d_ip / d_rgdp / d_unrate 或体制请求；R3 通胀：d_cpi_goods / d_cpi_serv / d_core 或同比约束；"
        "R4 油价：d_oil 加 x_oil_cause（supply / demand / mixed，不写 0/1）；R5 金融条件：fc.stance；R6 估值模式（缺省 inferred）。"
        "文字没有说的要求不要编：模型会补全并在报告里标出。",
        "u_* 冲击只写文字点名的干预：紧急或意外的降息加息、体制变化、扩表缩表（QE/QT）、减税或加税法案、刺激支票。"
        "写 u_bs 时还要写它对 d_tp10、d_y2、d_gz 的作用（扩表本身对资产的作用学不出来）。",
        "产出 = 生产率 × 劳动投入：d_rgdp、d_lprod、劳动投入（d_lf 或 d_unrate）同一季度最多写两个；"
        "叙事给了财政分项（d_defense、d_transfers、d_perstax）时不写 d_deficit。",
        "市场价格与调查值（E1、E1c、E2 字段：利率、利差、美元、油价与商品价格、房价、调查与新闻指数）只在文字明说时写，"
        "引文要提到这个变量；文字没有说的不能由你决定幅度。",
        "不能写：资产价格或回报、估值倍数或估值数字、y_* 腿、期初状态 s_*。"
        + ("估值的说法（“估值泡沫”“估值压缩”）写进 c_q.valuation 的 rerate（方向 + 温和 / 明显 / 极端），"
           "“估值不变”写 hold；这相当于直接假设股价。" if mode == "production" else
           "盲测里不写任何估值陈述（c_q.valuation 留空）。"),
        "c_q：anchors（文字给的长期假设，例如中性利率，带 from_q 与引文）；announced（已宣布的路径，例如前瞻指引）；"
        "constraints（跨季度的陈述：sum / mean / yoy / level / range）；regime（体制请求：label + share）。",
    ]
    if mode == "test":
        rules.append("盲测：任何字符串（引文、事件、备注）里都不能出现 19xx / 20xx 年份或月份计数；不要写真实的人名、机构名或地名。")
    return "\n".join(["# 规则", ""] + [f"{i}. {r}" for i, r in enumerate(rules, 1)])


def _band(pct: float) -> str:
    for hi, text in BANDS:
        if pct < hi:
            return text
    return BANDS[-1][1]


def state_rows(mode: str, *, Q=None, quarter: str | None = None) -> list[str]:
    """The state table at the start of `quarter` (default: the panel's last row, today): production shows levels
    (and the percentile band where a level means nothing); test mode shows percentile bands against the history
    before the quarter only, because combinations of levels date an episode."""
    import numpy as np
    import pandas as pd
    if Q is None:
        try:
            from ..qmodel import panel as qpanel
            Q = qpanel.load_panel()
        except (OSError, ValueError):
            return ["（季度面板还没有生成：没有状态表）"]
    q = pd.Period(quarter, "Q") if quarter else Q.index[-1]
    if q not in Q.index:
        return [f"（季度面板里没有 {q if mode == 'production' else '这个起点'}：没有状态表）"]
    out = []
    for col, zh, unit, how in STATES:
        if col not in Q.columns:
            continue
        v = Q.at[q, col]
        hist = Q[col].loc[:q].dropna()
        if v != v or len(hist) < 20:
            continue
        pct = float((hist < v).mean() * 100.0)
        band = _band(pct)
        if mode == "test" or how == "band":
            out.append(f"- {zh}：{band}")
        else:
            val = float(np.exp(v)) if how == "exp" else float(v)
            out.append(f"- {zh}：{val:.2f}{unit}（{band}）")
    return out


def state_section(mode: str, *, Q=None, quarter: str | None = None, rows: list[str] | None = None) -> str:
    """The state table section. rows: precomputed lines (the narrative test's sealed registry stores each episode's
    percentile bands, so its prompt builder never needs the episode's quarter); otherwise state_rows()."""
    head = "# 季初状态（今天的水平）" if mode == "production" else "# 季初状态（历史分位）"
    if rows is None:
        return "\n".join([head, ""] + state_rows(mode, Q=Q, quarter=quarter))
    return "\n".join([head, ""] + (list(rows) or ["（起点之前没有足够的历史：没有状态表）"]))


# ================================================================ synthetic few-shot examples (no years, no names)

def _ex_entry(value=None, sd=None, sign=None, size=None, src="stated", quote=""):
    e = {}
    if value is not None:
        e["value"] = value
        if sd is not None:
            e["sd"] = sd
    if sign is not None:
        e["sign"] = sign
        e["size"] = size
    e.update(src=src, quote=quote)
    return e


EXAMPLES = (
    {"name": "需求衰退（合成）",
     "narrative": "企业投资先行收缩，第一个季度工业产出下降约 2%，失业率上升 0.4 个百分点。央行随后两个季度各降息 50 个基点。"
                  "信用利差明显走阔，通胀温和回落。第三个季度起失业率仍缓慢上升，金融条件趋于稳定。",
     "qtable": {"schema": "qtable/1", "n_quarters": 3, "rows": [
         {"q": 1, "x": {"d_ip": _ex_entry(-2.0, src="stated", quote="工业产出下降约 2%"),
                        "d_unrate": _ex_entry(0.4, src="stated", quote="失业率上升 0.4 个百分点"),
                        "d_cpi_serv": _ex_entry(sign="-", size="温和", src="stated_qual", quote="通胀温和回落")},
          "fc": {"stance": "收紧", "quote": "信用利差明显走阔"}},
         {"q": 2, "x": {"d_ff": _ex_entry(-0.5, src="stated", quote="央行随后两个季度各降息 50 个基点")},
          "fc": {"stance": "未说明"}},
         {"q": 3, "x": {"d_ff": _ex_entry(-0.5, src="stated", quote="央行随后两个季度各降息 50 个基点"),
                        "d_unrate": _ex_entry(sign="+", size="小", src="stated_qual", quote="失业率仍缓慢上升")},
          "fc": {"stance": "中性", "quote": "金融条件趋于稳定"}}],
         "c_q": {}, "events": [{"text": "企业投资先行收缩", "quarters": [1], "templates": [], "channels": ["d_ip", "d_unrate"],
                                "generic_footprint": True, "how_decomposed": "需求衰退 → 产出下降、失业上升、通胀回落、信用利差走阔"}],
         "notes": []}},
    {"name": "石油供给冲击（合成）",
     "narrative": "一个产油地区爆发冲突，原油供应中断，油价一个季度内上涨约四成，地缘政治风险急剧上升。商品价格明显上涨。"
                  "第二个季度冲突平息，油价回落约一成，央行按兵不动。",
     "qtable": {"schema": "qtable/1", "n_quarters": 2, "rows": [
         {"q": 1, "x": {"d_oil": _ex_entry(34.0, sd=6.0, src="stated", quote="油价一个季度内上涨约四成"),
                        "x_oil_cause": {"value": "supply", "src": "stated_qual", "quote": "原油供应中断"},
                        "d_gpr": _ex_entry(sign="+", size="大幅", src="stated_qual", quote="地缘政治风险急剧上升"),
                        "d_cpi_goods": _ex_entry(sign="+", size="明显", src="stated_qual", quote="商品价格明显上涨")},
          "fc": {"stance": "未说明"}},
         {"q": 2, "x": {"d_oil": _ex_entry(-10.5, sd=3.0, src="derived", quote="油价回落约一成"),
                        "x_oil_cause": {"value": "supply", "src": "stated_qual", "quote": "冲突平息"},
                        "d_ff": {"sign": "0", "src": "stated_qual", "quote": "央行按兵不动"}},
          "fc": {"stance": "未说明"}}],
         "c_q": {}, "events": [{"text": "产油地区爆发冲突", "quarters": [1, 2], "templates": ["G02"],
                                "channels": ["d_oil", "x_oil_cause", "d_gpr", "d_cpi_goods"], "generic_footprint": True,
                                "how_decomposed": "能源生产地区的冲突 → 石油供给冲击、地缘风险上升、商品通胀"}], "notes": []}},
    {"name": "银行业恐慌（合成）",
     "narrative": "几家中型银行出现挤兑，融资市场压力骤升，信用利差急剧走阔。央行召开临时会议紧急降息，并宣布扩表购买国债，"
                  "期限溢价明显下降。下一个季度恐慌消退，信用利差有所收窄。",
     "qtable": {"schema": "qtable/1", "n_quarters": 2, "rows": [
         {"q": 1, "x": {"d_cpbill": _ex_entry(sign="+", size="大幅", src="stated_qual", quote="融资市场压力骤升"),
                        "d_tp10": _ex_entry(sign="-", size="明显", src="stated_qual", quote="期限溢价明显下降")},
          "u": {"u_ff": _ex_entry(sign="-", size="明显", src="stated_qual", quote="央行召开临时会议紧急降息"),
                "u_bs": _ex_entry(sign="+", size="明显", src="stated_qual", quote="宣布扩表购买国债")},
          "fc": {"stance": "明显收紧", "quote": "信用利差急剧走阔"}},
         {"q": 2, "x": {}, "fc": {"stance": "宽松", "quote": "信用利差有所收窄"}}],
         "c_q": {}, "events": [{"text": "几家中型银行出现挤兑", "quarters": [1, 2], "templates": [],
                                "channels": ["d_cpbill", "u_ff", "u_bs", "d_tp10"], "generic_footprint": True,
                                "how_decomposed": "银行挤兑 → 融资压力、信用利差走阔、紧急降息、扩表压低期限溢价"}], "notes": []}},
)


def examples_section() -> str:
    out = ["# 示例（合成的叙事，不对应任何真实时期）"]
    for ex in EXAMPLES:
        out += ["", f"## {ex['name']}", "", f"<narrative>\n{ex['narrative']}\n</narrative>", "",
                jsonfix.compact_json(ex["qtable"])]
    return "\n".join(out)


def build_system_prompt(mode: str = "production", *, card: dict | None = None, Q=None, quarter: str | None = None,
                        state_rows: list[str] | None = None) -> str:
    """The QTable translator's system prompt (§7.2), production or test mode, with the output schema section.
    state_rows: the state table's lines as given (the narrative test's blind percentile bands), instead of reading
    the panel at `quarter`."""
    check_mode(mode)
    card = card or qcard.load_card()
    parts = [role_section(mode), field_section(card, mode), size_section(), template_section(mode),
             rules_section(mode), state_section(mode, Q=Q, quarter=quarter, rows=state_rows), examples_section(),
             schema_section(card)]
    return "\n\n".join(parts)


def user_message(narrative: str, n_quarters: int | None = None, mode: str = "production") -> str:
    """The narrative only (production may add the length the user asked for)."""
    head = "请把下面的叙事翻译成季度情景表（QTable/1）。"
    if n_quarters and mode == "production":
        head += f"\n情景长度：{int(n_quarters)} 个季度（文字没有说的季度留空，由模型补全）。"
    return f"{head}\n\n<narrative>\n{narrative}\n</narrative>"


def repair_message(narrative: str, previous: str, problems: list[str], n_quarters: int | None = None,
                   mode: str = "production") -> str:
    """One format repair: the previous answer, what was wrong with it and the narrative again (quotes come from it)."""
    prev = str(previous or "").replace("<", "＜").replace(">", "＞")[:20000]
    lines = "\n".join(f"- {p}" for p in problems[:20]) or "- 输出不是合法的 JSON"
    return (f"{user_message(narrative, n_quarters, mode)}\n\n上一次的回答不是可用的季度情景表：\n{lines}\n\n"
            f"<previous_output>\n{prev}\n</previous_output>\n\n只输出改正后的一个 JSON 对象。")


def prompt_scan(text: str) -> list[str]:
    """The blind protocol's scan of a test-mode prompt (§8.4): year tokens, quarter labels, month counts, a T0
    offset, footprint keys. [] = clean."""
    out = []
    for m in qcard.YEAR_TOKEN.finditer(text):
        out.append(f"年份「{m.group(0)}」")
    for m in re.finditer(r"(?<![0-9A-Za-z_])(?:19|20)?[0-9]{2}Q[1-4](?![0-9])", text):
        out.append(f"季度标签「{m.group(0)}」")
    for m in QT.MONTH_COUNT.finditer(text):
        out.append(f"月份计数「{m.group(0)}」")
    for key in ("footprint_quarters", "measured", "样本期", "模型起点"):
        if key in text:
            out.append(f"「{key}」")
    return out


# ================================================================ calls

def backend_name() -> str:
    return tr.backend_name()


def complete_text(system: str, user: str, model: str, backend: str, refresh: bool = False,
                  cancel: threading.Event | None = None) -> tuple[str, dict]:
    """(the answer's text, metadata): from data/llm_cache/ when this exact request was answered before (no call),
    else one call within a call slot. Only an answer that parses as a JSON object is cached (mock never).
    ClaudeCodeError propagates with its kind."""
    from . import backend as llm_backend
    from . import mock
    key = tr.cache_key(backend, model, system, user)
    if not refresh and backend != "mock":
        hit = tr.cache_load(key)
        if hit is not None:
            return hit["text"], {**hit["meta"], "cached": True, "cached_at": hit.get("created")}
    with tr._slot():
        if not refresh and backend != "mock":
            hit = tr.cache_load(key)
            if hit is not None:
                return hit["text"], {**hit["meta"], "cached": True, "cached_at": hit.get("created")}
        run = claude_code.run if backend == "claude_code" else mock.run
        with llm_backend.cli_slot(cancel=cancel, timeout=tr.timeout_s()):
            text, meta = run(system, user, model, tr.timeout_s(), effort=EFFORT, cancel=cancel)
        obj, _ = jsonfix.extract_json(text)
        if backend != "mock" and isinstance(obj, dict):
            tr.cache_store(key, text, meta)
    return text, {**meta, "cached": False}


def is_cached(system: str, user: str, model: str, backend: str) -> bool:
    return backend != "mock" and tr.cache_load(tr.cache_key(backend, model, system, user)) is not None


def requested_quarters(narrative: str, n_quarters: int | None) -> int:
    if n_quarters:
        return max(1, min(int(n_quarters), cq.QTABLE_MAX_QUARTERS))
    for rx in LENGTH_RE:
        m = rx.search(str(narrative or ""))
        if m:
            return max(1, min(int(m.group(1)), cq.QTABLE_MAX_QUARTERS))
    return QT.DEFAULT_QUARTERS


def estimate(narrative: str, *, mode: str = "production", n_quarters: int | None = None, backend: str | None = None,
             model: str | None = None) -> dict:
    """What one translation would cost, without a model call: calls (1, at most 2 with the format repair), input
    tokens (the design docs' convention), the output-token range of §8.5, minutes, whether it is cached."""
    check_mode(mode)
    backend = backend or backend_name()
    model = model or tr.model_id()
    system = build_system_prompt(mode)
    user = user_message(str(narrative or ""), n_quarters, mode)
    cached = is_cached(system, user, model, backend)
    label = {"mock": "模拟后端（不调用模型，不消耗订阅额度）", "claude_code": claude_code.LABEL}.get(backend, backend)
    return {"label": LABEL, "backend": backend, "backend_label": label, "model": model, "mode": mode,
            "calls": 0 if cached or backend == "mock" else 1, "max_calls": 0 if cached or backend == "mock" else 2,
            "input_tokens": tokens(system) + tokens(user), "output_tokens": list(OUTPUT_TOKENS),
            "seconds": list(SECONDS), "cached": cached, "needs_confirmation": backend != "mock" and not cached,
            "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(), "prompt_chars": len(system)}


def _problems(report: dict | None, notes: list[str]) -> list[str]:
    if report is None:
        return notes[-1:] or ["输出不是合法的 JSON"]
    out = [f"{it.get('where') + '：' if it.get('where') else ''}{it['text']}" for it in report["errors"]]
    out += [f"{it.get('where') + '：' if it.get('where') else ''}{it['text']}" for it in report["refused"]
            if it.get("category") == "form"]
    return out


def _needs_repair(report: dict | None) -> bool:
    return report is None or not report["readable"] or report.get("form_problems", 0) > 0


def translate(narrative: str, *, mode: str = "production", n_quarters: int | None = None, refresh: bool = False,
              model: str | None = None, backend: str | None = None, quarter: str | None = None, Q=None,
              cancel: threading.Event | None = None, stats: dict | None = None,
              state_rows: list[str] | None = None, numbers: bool = True) -> dict:
    """Narrative -> {"status": ok | repaired | fallback, "qtable" (clean), "validation", "raw", "repair", "error",
    "backend", "model", "cached", "calls", "usage", "elapsed_s", "prompt", "mode", "label"}.

    One call (cached when answered before); one format repair when the answer is not a readable QTable or has form
    problems; the minimal QTable as the fallback (refusal, timeout, overload, an unrepairable answer); QTablePause on
    the usage limit, login, a missing CLI, a model not allowed or a cancel. quarter (test mode): the episode's
    origin, whose state the prompt shows as percentile bands; state_rows: those lines precomputed (the narrative
    test's blind registry), so the caller needs no date. numbers=False: the narrative test's N3 arm (market prices
    and surveys only as size bins, qtable R-N3). out["messages"] holds the system prompt and the user message exactly
    as sent and out["repair"]["user"] the repair message (every byte sent is the caller's to record)."""
    check_mode(mode)
    text = str(narrative or "").strip().encode("utf-8", "replace").decode("utf-8")
    if not text:
        raise tr.TranslatorError("叙事不能为空。")
    if len(text) > tr.MAX_NARRATIVE_CHARS:
        raise tr.TranslatorError(f"叙事过长（{len(text)} 字），请控制在 {tr.MAX_NARRATIVE_CHARS} 字以内。")
    backend = backend or backend_name()
    if backend not in BACKENDS:
        raise tr.TranslatorError(E_API if backend == "api" else tr.status()["reason"])
    model = model or tr.model_id()
    if backend == "claude_code":            # local checks only (claude --version / auth status): no usage
        st = tr.status() if tr.backend_name() == "claude_code" else claude_code.status(model)
        if not st["available"]:
            raise QTablePause("cli_missing" if "找不到" in str(st.get("reason")) else "login", st["reason"])
    H = requested_quarters(text, n_quarters)
    system = build_system_prompt(mode, Q=Q, quarter=quarter, state_rows=state_rows)
    user = user_message(text, n_quarters, mode)
    t0 = time.perf_counter()
    out = {"label": LABEL, "mode": mode, "backend": backend, "model": model, "status": "ok", "raw": None,
           "repair": None, "error": None, "cached": False, "calls": 0, "usage": {},
           "prompt": {"version": PROMPT_VERSION, "mode": mode, "chars": len(system),
                      "sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
                      "narrative_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}}

    def ask(u: str) -> tuple[str | None, dict | None]:
        try:
            answer, meta = complete_text(system, u, model, backend, refresh=refresh, cancel=cancel)
        except claude_code.ClaudeCodeError as err:
            if err.kind in PAUSE_KINDS or err.kind == "cancelled":
                raise QTablePause(err.kind, str(err), err.reset_at) from err
            out["error"] = {"kind": err.kind, "message": str(err), "action": "fallback"}
            return None, None
        except tr.TranslatorError as err:          # no free call slot
            out["error"] = {"kind": "rate", "message": str(err), "action": "fallback"}
            return None, None
        out["calls"] += 0 if meta.get("cached") else 1
        out["cached"] = out["cached"] or bool(meta.get("cached"))
        for k, v in (meta.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                out["usage"][k] = out["usage"].get(k, 0) + v
        return answer, meta

    def read(answer: str | None) -> tuple[dict | None, dict | None, list[str]]:
        if answer is None:
            return None, None, []
        obj, notes = jsonfix.extract_json(answer)
        if obj is None:
            return None, None, notes
        return obj, QT.validate(obj, text, mode=mode, source="translator", stats=stats, numbers=numbers), notes

    answer, _ = ask(user)
    if answer is not None and REFUSAL_TEXT.search(answer) and jsonfix.extract_json(answer)[0] is None:
        out["error"] = {"kind": "refusal", "message": claude_code.E_REFUSAL, "action": "fallback"}
        answer = None
    obj, report, notes = read(answer)
    out["raw"] = obj
    if answer is not None and _needs_repair(report):
        problems = _problems(report, notes)
        r_user = repair_message(text, answer, problems, n_quarters, mode)
        r_answer, _ = ask(r_user)
        r_obj, r_report, r_notes = read(r_answer)
        out["repair"] = {"problems": problems, "raw": r_obj, "ok": r_report is not None and r_report["readable"],
                         "user": r_user, "user_sha256": hashlib.sha256(r_user.encode("utf-8")).hexdigest()}
        if r_report is not None and r_report["readable"]:
            obj, report = r_obj, r_report
            out["status"] = "repaired"
        elif report is None or not report["readable"]:
            out["error"] = out["error"] or {"kind": "unparseable", "message": "修复后仍不是可用的季度情景表",
                                            "action": "fallback"}
            report = None
    if report is None or not report["readable"]:
        why = (out["error"] or {}).get("message") or "输出无法读取"
        qt = QT.minimal(H, note=FALLBACK_NOTE.format(why=why)[:QT.NOTE_MAX])
        report = QT.validate(qt, text, mode=mode, source="translator", stats=stats, numbers=numbers)
        out["status"] = "fallback"
        if out["error"] is None:
            out["error"] = {"kind": "unparseable", "message": why, "action": "fallback"}
    out["qtable"] = report.pop("qtable")
    out["validation"] = report
    out["elapsed_s"] = round(time.perf_counter() - t0, 3)
    out["messages"] = {"system": system, "user": user}     # exactly what was sent (the narrative is stripped first)
    return out


# ================================================================ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="叙事 → 季度情景表（QTable/1）")
    ap.add_argument("narrative", nargs="?", help="叙事全文（或用 --file）")
    ap.add_argument("--file", help="从文件读叙事")
    ap.add_argument("--mode", default="production", choices=QT.MODES)
    ap.add_argument("--quarters", type=int, default=None, help="情景长度（季度）")
    ap.add_argument("--print-prompt", action="store_true", help="只打印系统提示词（不调用模型）")
    ap.add_argument("--yes", action="store_true", help="不再询问，直接调用（会计入订阅额度）")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)
    if a.print_prompt:
        text = build_system_prompt(a.mode)
        print(text)
        if a.mode == "test":
            problems = prompt_scan(text)
            print(f"\n[盲测扫描] {'没有问题' if not problems else '；'.join(problems)}", file=sys.stderr)
        return 0
    narrative = open(a.file, encoding="utf-8").read() if a.file else (a.narrative or "")
    if not narrative.strip():
        ap.error("需要叙事（参数或 --file）")
    est = estimate(narrative, mode=a.mode, n_quarters=a.quarters)
    print(f"[预计] 后端 {est['backend_label']}，调用 {est['calls']} 次（最多 {est['max_calls']} 次），"
          f"输入约 {est['input_tokens']} tokens，输出约 {est['output_tokens'][0]}–{est['output_tokens'][1]} tokens",
          file=sys.stderr)
    if est["needs_confirmation"] and not a.yes:
        if not sys.stdin.isatty():
            print("需要确认：加 --yes（会计入你的 Claude 订阅额度）", file=sys.stderr)
            return 2
        if input("确认运行？[y/N] ").strip().lower() not in ("y", "yes"):
            return 1
    try:
        res = translate(narrative, mode=a.mode, n_quarters=a.quarters, refresh=a.refresh)
    except QTablePause as err:
        print(f"[暂停] {err}", file=sys.stderr)
        return 3
    except tr.TranslatorError as err:
        print(f"[错误] {err}", file=sys.stderr)
        return 1
    print(json.dumps({"status": res["status"], "qtable": res["qtable"],
                      "refused": res["validation"]["refused"], "warnings": res["validation"]["warnings"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
