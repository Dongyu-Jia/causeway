"""QTable/1: the scenario layer's contract (SIM_DESIGN §3.4, §7, Appendix B; §14 P6).

The LLM's output schema is the model's input schema. A QTable states, per quarter, any subset of the card's 52 x_q
fields and 5 policy shocks (each a value with an optional sd, a sign + size bin, or a range; each with a source code
and a quote), the financial-conditions stance (required in every row), and the cross-quarter c_q block (anchors,
announced paths, constraints, valuation statements, regime requests), plus events and notes that are never inputs.
An absent field means "not stated": the factor layer completes it.

    validate(qt, narrative, mode=, source=)  the validation report (§7.3): structural errors, refused entries
                                             (removed), warnings, the R1-R5 requirements a quarter leaves to the
                                             model (with the narrative's evidence sentence) and the clean QTable
    check(qt, narrative, ...)                the same, but a QTableError naming the first problem unless clean
    constraints(qt, stack=... | Q=...)       the clean QTable as rollout inputs: factors.Statement list, valuation
                                             statements, the cycle module's regime odds, notes (what /api/q/run runs)
    to_phase_spec(qt, ...)                   the monthly engine's phase spec from the same QTable (§7.4): the
                                             narrative test's engine arms run on identical translations
    minimal(H)                               every field absent, fc 未说明 (the mock's and the fallback's answer)
    builtin(sid), builtin_ids()              the 13 reviewed built-in scenarios as QTables (scenarios/qtables/,
                                             written by tools/qtable_from_yaml.py)
    scan_blind(obj)                          year tokens and month counts in any string (the blind protocol)

Modes: "production" (the lab, the built-in scenarios, the agent tree) and "test" (the narrative test). Sources:
"translator" (an LLM translation; every rule), "user" (the user's own cells in the lab: the user is the scenario's
author, so the E1 keyword rule and the intervention check do not apply), "builtin" (a reviewed conversion of a
YAML spec; its quotes are checked against the YAML's prose) and "legacy" (the research round's stand-in tables:
no quotes, so the quote rules are skipped and every result says so).

Rules (§7.3; codes in the report, messages in Chinese):

    R-SHAPE   not a QTable (not an object, no rows, quarters out of 1..60)                  error
    R-FIELD   unknown field / shock / anchor / template                                    refused
    R-E3      asset prices or returns, valuation legs, E3 fields                            refused
    R-STATE   a start-of-quarter state written as an input                                  refused
    R-FORM    wrong entry form (not exactly one of value / sign + size / min-max; types)    refused
    R-SIZE    a size outside the five bins                                                  refused
    R-SRC     src not stated / stated_qual / derived, or words written as a number: a value  refused
              or range marked stated / derived whose quote carries no number (digits, 百分之…, a
              Chinese numeral with a unit such as 三成 / 五十个基点, 一半); translations only
    R-QUOTE   no quote, a quote shorter than QUOTE_MIN characters (translations), or a quote  refused
              that is not in the narrative (also a credit stance's 未说明 quote)
    R-E1      an E1 / E1c / E2 entry (or a credit stance) whose quote does not name it      refused
    R-N3      numbers=False (the narrative test's N3 arm, §8.4): an E1 / E1c / E2 value or   refused (legacy:
              range; the research round's stand-ins are re-expressed as the nearest size bin   re-expressed, W-N3)
    R-RANGE   beyond 1.5 x the historical range (y/y constraints against the level's range;  refused / warning
              a policy shock beyond ±4 shock sd): refused in test mode, clipped (W-RANGE)
    R-VAL     a valuation statement in test mode; a number in a valuation statement         refused
    R-YEAR    a 19xx / 20xx token or a month count in any string, test mode                 refused
    W-QE      u_bs without its intervention paths on d_tp10 / d_y2 / d_gz                   warning
    W-IDENT   more than two of output / productivity / labour input, deficit + components   warning (one dropped)
    W-U       a policy shock whose quote names no intervention                              warning
    W-FC, W-R1..W-R4, W-OIL   R1-R5 left to the model (with the evidence sentence)           warning
    W-ANCHOR  scenario anchors are recorded; the factor layer's steady state is fixed at fit warning
    W-VAL     a valuation statement: 估值假设相当于直接假设股价                              warning
    W-QUOTE   no narrative to check quotes against (a hand-written or imported table)       warning
    W-EVENT   an event without a generic footprint is a record, not an input                warning

Nothing here calls an LLM or downloads anything. The quarterly stack is 试运行 until the switch gate T-S passes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import unicodedata
from pathlib import Path

from . import card as qcard
from . import config_q as cq

SCHEMA_ID = cq.QTABLE_SCHEMA_ID
MODES = ("production", "test")
SOURCES = ("translator", "user", "builtin", "legacy")
QTABLES_DIR = Path(__file__).resolve().parent.parent / "scenarios" / "qtables"   # the 13 reviewed built-ins
ENTRY_KEYS = ("value", "sd", "sign", "size", "min", "max", "src", "quote")
VALUATION_KEYS = ("quarters", "mode", "direction", "size", "reversion_half_life_years", "quote")
CQ_ITEM_KEYS = {"anchors": ("value", "from_q", "quote"), "announced": ("field", "quarters", "values", "sd", "quote"),
                "constraints": ("kind", "field", "quarters", "value", "sd", "min", "max", "quote"),
                "regime": ("quarters", "label", "share", "quote")}
SIGNS = {"+": 1.0, "-": -1.0, "0": 0.0}
TOP_KEYS = ("schema", "model_card", "n_quarters", "rows", "c_q", "events", "notes")
CQ_KEYS = ("anchors", "announced", "constraints", "valuation", "regime")
CONSTRAINT_KINDS = ("sum", "mean", "yoy", "level", "range")
YOY_VARS = {"s_cpi_yoy": "cpi_yoy", "s_core_yoy": "core_yoy", "d_core": "core_yoy"}   # y/y has a CPI observable only
LEVEL_VARS = {"s_ff": "ff", "s_unrate": "unrate", "s_cpi_yoy": "cpi_yoy", "s_core_yoy": "core_yoy"}
EVENT_TEXT_MAX = 200
NOTE_MAX = 200
RANGE_WIDEN = 1.5                       # §7.3: beyond 1.5 x the historical range
SHOCK_RANGE_SD = cq.SATURATION_SD       # a policy shock beyond ±4 shock sd (the factor layer's saturation)
YOY_RANGE = {"s_cpi_yoy": "s_cpi_yoy", "s_core_yoy": "s_core_yoy", "d_core": "s_core_yoy"}   # y/y: the level's range
QUOTE_MIN = 2                           # a translation's quote: at least two characters of the narrative
NUMBER_CLASSES = ("E1", "E1c", "E2")    # N3 (§8.4): market prices, commodity prices and surveys only as size bins
# a quote that carries a number: digits (ASCII or full-width), 百分之…, a Chinese numeral with a unit (三成, 两倍,
# 五十个基点, 一个百分点) or a number word (一半, 减半, 翻番). Durations (三个季度) do not count: they are not the value.
NUMERAL = re.compile(r"[0-9０-９]|百分之|[零〇一二两三四五六七八九十百千万亿]+(?:成|倍|个?百分点|个?基点|分之|%|％)|"
                     r"一半|减半|过半|翻番|翻倍|翻一番")
QUAL_SD = cq.QTABLE_QUAL_SD             # a sign + size entry: soft at size x sd with sd = 0.25 training sd
SIGN_ZERO_SD = cq.SIZE_BINS["小"]        # sign "0" (unchanged): soft at 0 with the 小 bin as its sd
ANNOUNCED_SD = cq.SIZE_BINS["温和"]      # an announced path without sd: soft, sd = the 温和 bin of the field
DEFAULT_QUARTERS = 4                    # the fallback's horizon when nobody said how long the scenario is
FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")
YEAR_TOKEN = qcard.YEAR_TOKEN
# a month count that can date an episode (a sample position such as "第 640 个月", "T0+18m"): three or more digits
# before 月, or a T0 offset. Durations such as "6 个月" in an N1 narrative's month timing are not counts.
MONTH_COUNT = re.compile(r"(?<![0-9])[0-9]{3,}\s*个?月|T0\s*[+\-−]")
SENTENCE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
# the credit stance names financial conditions: the keywords of d_gz and d_spread plus these
FC_WORDS = ("金融条件", "融资条件", "融资环境", "信贷", "利差", "流动性", "违约", "风险偏好", "风险溢价", "融资")
# what names an intervention (the card's 只在文字写明“干预”时): warning W-U when a shock's quote has none
INTERVENTION_WORDS = {
    "u_ff": ("紧急", "意外", "临时", "非常规", "干预", "被迫", "抢先", "预防性", "政治压力", "体制", "换帅", "出人意料",
             "突然", "超预期", "提前"),
    "u_bs": ("扩表", "缩表", "量化宽松", "量化紧缩", "QE", "QT", "购债", "资产购买", "收益率曲线控制", "YCC", "购买国债"),
    "u_def": ("刺激", "法案", "支出计划", "财政扩张", "财政紧缩", "削减开支", "减支", "一揽子"),
    "u_tr": ("支票", "补贴", "转移支付", "救助", "UBI", "发放", "失业救济", "现金"),
    "u_ptax": ("减税", "加税", "税改", "税法", "退税", "法案"),
}
# evidence sentences for the requirements a quarter leaves to the model (§7.1, §10.4)
TOPIC_WORDS = {
    "R1": ("利率", "降息", "加息", "央行", "美联储", "联储", "货币政策", "基点", "政策利率"),
    "R2": ("增长", "衰退", "失业", "就业", "产出", "GDP", "经济", "收缩", "复苏", "裁员"),
    "R3": ("通胀", "物价", "CPI", "价格"),
    "R4": ("油价", "原油", "石油"),
    "R5": ("信用", "利差", "金融条件", "融资", "信贷", "违约", "流动性"),
}
GROWTH_X = ("d_ip", "d_rgdp", "d_unrate")
INFL_X = ("d_cpi_goods", "d_cpi_serv", "d_core")
FC_X = ("d_gz", "d_spread")
INFL_LABELS = ("inflation", "stagflation", "deflation", "high_infl")
# anchors (c_q.anchors): plausible bounds of a long-run assumption (判断值): r* and the term premium within the
# real-rate and term-premium history since 1973, π_lr within the inflation-expectation history, the spread and
# unemployment anchors within their level history; oil and dollar anchors are log shifts of the training medians
ANCHOR_BOUNDS = {"r_star": (-3.0, 4.0), "pi_lr": (0.0, 8.0), "a_tp": (-1.5, 4.0), "a_spread": (0.8, 5.0),
                 "a_ustar": (3.0, 10.0), "a_oil": (-1.0, 1.0), "a_usd": (-0.5, 0.5)}
ANCHOR_ZH = {"r_star": "中性实际利率 r*", "pi_lr": "长期通胀", "a_tp": "期限溢价锚", "a_spread": "信用利差锚",
             "a_ustar": "失业率锚", "a_oil": "实际油价锚移位", "a_usd": "实际美元锚移位"}
REGIME_ZH = {"expansion": "扩张", "inflation": "通胀", "deflation": "物价负增长", "recession": "衰退",
             "depression": "深度衰退", "stagflation": "滞胀", "contraction": "收缩", "high_infl": "高通胀"}
REGIME_ALIASES = {"扩张": "expansion", "通胀": "inflation", "过热": "inflation", "通缩": "deflation",
                  "物价负增长": "deflation", "衰退": "recession", "萧条": "depression", "深度衰退": "depression",
                  "滞胀": "stagflation", "收缩": "contraction", "高通胀": "high_infl"}
CONTRACTION_LABELS = ("contraction", "recession", "depression", "stagflation")
# the regime classifier's thresholds (config.REGIMES params): CPI y/y above 4 is high inflation, below 0 falling
# prices; a deep contraction loses at least 15% of IP from its peak
HIGH_INFL, CPI_NEG, DEEP_IP = 4.0, 0.0, 15.0
REGIME_MAJORITY = 0.5                    # a share at or above it imposes the label's inequality on the quarters' mean

MSG = {
    "not_object": "QTable 必须是 JSON 对象（{ … }）",
    "no_rows": "qtable.rows 必须是非空列表",
    "n_quarters": "n_quarters 必须是 1..{hi} 之间的整数（写的是 {v!r}）",
    "row_q": "行的季度 q 必须是 1..{H} 之间的整数（写的是 {v!r}）",
    "row_dup": "第 {q} 季出现了两次：只保留第一行",
    "row_shape": "每一行必须是对象，并有 q、x 与 fc",
    "top_key": "QTable 里有不认识的键「{k}」：已删除",
    "schema": "schema 必须是 qtable/1（写的是 {v!r}）：已按 qtable/1 读取",
    "field": "字段不存在：{f}",
    "field_x": "字段不存在：{f}（没有可测的对应变量，未作输入）",
    "shock": "未知的政策冲击：{f}（只有 u_ff、u_bs、u_def、u_tr、u_ptax）",
    "e3": "资产价格与估值倍数不是输入：{f}",
    "state": "「{f}」是期初状态（来自数据），不是情景输入",
    "form": "条目必须写 value（可带 sd）、sign + size 或 min / max 之中的一种",
    "keys": "条目里有不认识的键：{keys}",
    "number": "{k} 必须是有限的数字（写的是 {v!r}）",
    "sd": "sd 必须是正数（写的是 {v!r}）",
    "sign": "sign 只能是 '+'、'-' 或 '0'（写的是 {v!r}）",
    "size": "定性幅度只能是 小 / 温和 / 明显 / 大幅 / 极端（写的是「{v}」）：文字没有说的幅度不能由翻译器决定",
    "no_size": "写了方向就要写幅度档 size（小 / 温和 / 明显 / 大幅 / 极端）",
    "minmax": "min 大于 max",
    "src": "src 必须是 stated（文字里的数字）、stated_qual（文字的定性幅度）或 derived（对文字数字的推算）",
    "qual_number": "定性的文字只能写成方向 + 幅度档（stated_qual），不能写成数字：数字要有文字依据",
    "oil_cause": "油价原因 x_oil_cause 只写 supply / demand / mixed（不再写 0/1：两套旧约定正好相反）",
    "oil_src": "x_oil_cause 的 src 只能是 stated 或 stated_qual",
    "quote_missing": "每个值都要带引文 quote：这个值没有文字依据",
    "quote": "引文在叙事里找不到：这个值没有文字依据",
    "quote_cut": "引文超过 {n} 个字：已截短",
    "quote_short": "引文太短（至少 {n} 个字）：这个值没有文字依据",
    "src_number": "写成数字（src {src}）的值要有文字里的数字作依据：引文里没有数字。文字只有定性说法时写 sign + size，src 写 stated_qual",
    "n3": "N3 叙事不含数字：{zh} 只能写方向 + 幅度档（sign + size），不能写数字（市场价格与调查值的数字可能来自记忆）",
    "n3_bin": "研究轮代译的 N3 数字已换成最接近的幅度档：{zh} {v:g} → {sign}{size}",
    "item_keys": "{what}里有不认识的键：{keys}（已删除）",
    "val_number": "估值假设相当于直接假设股价：估值陈述只写模式、方向和幅度档，不写数字（{keys}）",
    "e1": "文字没有说的市场价格和调查值不能由翻译器决定：引文没有提到{zh}",
    "fc_stance": "金融条件 fc.stance 必须是 {stances} 之一（写的是 {v!r}）：按未说明处理",
    "range_test": "超出历史范围：{v:g} 超过历史区间的 1.5 倍（{lo:g} 到 {hi:g}）",
    "range_clip": "超出历史范围：{v:g} 超过历史区间的 1.5 倍，按 {used:g} 处理",
    "u_range_test": "超出历史范围：冲击 {v:g} 超过 ±{k:g} 个冲击标准差（{lo:g} 到 {hi:g}）",
    "u_range_clip": "超出历史范围：冲击 {v:g} 超过 ±{k:g} 个冲击标准差，按 {used:g} 处理",
    "val_test": "估值假设相当于直接假设股价：叙事检验不允许写估值陈述",
    "val_warn": "估值假设：这一条相当于直接假设股价（第 {qs} 季，{what}）",
    "year": "盲测：输出里不能出现年份或月份计数（「{tok}」）",
    "qe": "扩表对资产的作用学不出来，请写明它对期限溢价、2 年期收益率和信用利差的作用（第 {q} 季只写了资产负债表冲击）",
    "ident_output": "产出 = 生产率 × 劳动投入：同一季度最多写其中两项；第 {q} 季的劳动生产率增长由联合模型推出（与文字不一致时报告）",
    "ident_deficit": "赤字 = 支出 + 转移 − 税收：第 {q} 季叙事给了分项，赤字留给联合模型推断",
    "intervention": "冲击「{f}」只用于文字写明的干预（紧急或意外的政策动作、扩表缩表、退税支票、减税法案等）：引文里没有找到干预的说法",
    "fc": "金融条件未说明：信用利差由模型补全。历史检验里，补全的信用利差与无条件抽样相当，而资产结果主要取决于它",
    "R1": "政策利率没有说明：由反应函数补全",
    "R2": "增长没有说明：由周期模块补全",
    "R3": "通胀没有说明：由模型补全",
    "R4": "油价没有说明：由模型补全",
    "oil": "油价变动没有写原因（供给冲击 / 需求推动 / 混合）：模型按历史上供给与需求的混合补全",
    "no_evidence": "叙事里没有找到相关的句子",
    "no_narrative": "没有叙事原文：引文没有核对（手写或导入的情景表）",
    "legacy": "研究轮代译（无引文）：引文规则没有核对",
    "anchor": "情景的长期锚已记录（{names}），但这一版季度模型的稳态先验在拟合时固定：这次运行用默认的长期锚（判断值）",
    "anchor_range_test": "长期锚 {a} = {v:g} 超出合理范围 {lo:g} 到 {hi:g}",
    "anchor_range": "长期锚 {a} = {v:g} 超出合理范围 {lo:g} 到 {hi:g}：按 {used:g} 处理",
    "quarters": "quarters 必须是 1..{H} 之内的整数列表",
    "values_len": "预告路径 {f}：quarters 与 values 的长度不同",
    "kind": "约束的 kind 必须是 sum / mean / yoy / level / range",
    "yoy": "同比约束只对 CPI 同比与核心 CPI 同比有观测（s_cpi_yoy、s_core_yoy、d_core）：{f} 不能写同比",
    "no_value": "约束必须写 value，或 min / max",
    "range_kind": "range 约束必须写 min 或 max",
    "regime_label": "体制请求的标签必须是 {labels} 之一（写的是 {v!r}）",
    "regime_share": "体制请求的 share 必须是 0..1 之间的数（写的是 {v!r}）",
    "regime_alias": "体制「{v}」已按 {id}（{zh}）读取",
    "event_shape": "事件必须写 text、quarters、channels、generic_footprint 与 how_decomposed",
    "event_template": "事件「{t}」的模板 {g} 不存在：已删除",
    "event_channel": "事件「{t}」的渠道 {c} 不是 QTable 字段：已删除",
    "event_nofoot": "事件「{t}」没有通用足迹：只作记录，不是输入",
    "event_unstated": "事件「{t}」列出的渠道在第 {qs} 季没有写入数值：它只作记录",
    "note_cut": "备注超过 {n} 个字：已截短",
}


class QTableError(ValueError):
    """A QTable that cannot be run as written (Chinese message); .report holds the full validation report."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report


# ================================================================ card, statistics, text helpers

_CARD: dict | None = None
_TEMPLATES: set | None = None
_STATS: dict = {}
_STATS_LOCK = threading.Lock()


def _card(card: dict | None = None) -> dict:
    global _CARD
    if card is not None:
        return card
    if _CARD is None:
        _CARD = qcard.load_card()
    return _CARD


def template_ids() -> set:
    """G01..G35 (the blind schema view's templates)."""
    global _TEMPLATES
    if _TEMPLATES is None:
        try:
            view = json.loads(cq.SCHEMA_VIEW_PATH.read_text(encoding="utf-8"))
            _TEMPLATES = {t["id"] for t in view.get("templates", [])}
        except (OSError, ValueError, KeyError):
            _TEMPLATES = {f"G{k:02d}" for k in range(1, 36)}
    return _TEMPLATES


def norm(text) -> str:
    """NFKC and no whitespace: the form quotes and narratives are compared in (a model that writes 「加息50个基点」
    for 「加息 50 个基点」, or full-width digits, still quotes the text)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or "")))


def quote_in(quote: str, narrative: str) -> bool:
    q = norm(quote)
    return bool(q) and q in norm(narrative)


def names(quote: str, words) -> bool:
    q = norm(quote).lower()
    return any(norm(w).lower() in q for w in words if w)


def sentences(narrative: str) -> list[str]:
    return [s.strip() for s in SENTENCE.findall(str(narrative or "")) if s.strip()]


def evidence(narrative: str | None, rule: str) -> str | None:
    """The narrative's first sentence on the requirement's topic (R1..R5), or None."""
    for s in sentences(narrative or ""):
        if names(s, TOPIC_WORDS[rule]):
            return s[:cq.QUOTE_MAX]
    return None


def scan_blind(obj, path: str = "$") -> list[tuple[str, str]]:
    """[(path, token)] of every 19xx / 20xx token or month count in any string of `obj` (the blind protocol)."""
    out = []
    if isinstance(obj, str):
        for m in YEAR_TOKEN.finditer(obj):
            out.append((path, m.group(0)))
        for m in MONTH_COUNT.finditer(obj):
            out.append((path, m.group(0)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += scan_blind(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += scan_blind(v, f"{path}[{i}]")
    return out


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and math.isfinite(v) and v == int(v):
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _train_end(Q) -> str:
    try:
        from . import panel as qpanel
        return str(qpanel.load_meta()["info"]["last_data_quarter"])
    except (OSError, KeyError, ValueError):
        return str(Q.index[-2])


def stats_table(Q=None, train_end: str | None = None, *, shocks: bool = False) -> dict:
    """{field: {"sd", "lo", "hi"}} on the quarters up to the train end (default: the last data quarter): the
    training sd of every x field (the size bins' unit; the same numbers as panel.field_stats), its historical
    range, and for the level states a constraint may name (s_cpi_yoy, s_core_yoy, s_unrate, s_ff) the range of the
    level with the sd of its quarterly change. shocks=True adds the policy shocks' sd (policy.shock_sd). Empty when
    the quarterly panel is not built (the range rule is then skipped and the report says so)."""
    import numpy as np
    import pandas as pd

    from . import panel as qpanel
    if Q is None:
        try:
            stamp = cq.PANEL_Q_PATH.stat().st_mtime_ns
        except OSError:
            return {}
    else:
        stamp = (id(Q), len(Q), str(Q.index[-1]), len(Q.columns))
    key = (stamp, train_end, shocks)
    with _STATS_LOCK:
        hit = _STATS.get(key)
    if hit is not None:
        return hit
    if Q is None:
        try:
            Q = qpanel.load_panel()
        except (OSError, ValueError):
            return {}
    te = pd.Period(train_end or _train_end(Q), "Q")
    xf = [f for f in qcard.x_fields(_card()) if f in Q.columns and f != "x_oil_cause"]
    fs = qpanel.field_stats(Q, xf, str(te))
    out: dict = {}
    for f in xf:
        s = Q[f].loc[:te].dropna()
        if len(s) < 3:
            continue
        out[f] = {"sd": float(fs.at[f, "sd"]), "lo": float(s.min()), "hi": float(s.max())}
    for st in LEVEL_VARS:
        if st in Q.columns:
            s = Q[st].loc[:te].dropna()
            if len(s) > 3:
                out[st] = {"sd": float(np.std(np.diff(s.to_numpy(float)), ddof=1)), "lo": float(s.min()),
                           "hi": float(s.max())}
    if shocks:
        from . import policy
        try:
            for u, v in policy.shock_sd(policy.fit(Q, str(te))).items():
                if v is not None and math.isfinite(float(v)):
                    out[u] = {"sd": float(v), "lo": None, "hi": None}
        except Exception:  # noqa: BLE001 - a sign + size shock then has no unit and is refused in constraints()
            pass
    with _STATS_LOCK:
        if len(_STATS) > 8:
            _STATS.clear()
        _STATS[key] = out
    return out


def minimal(n_quarters: int = DEFAULT_QUARTERS, *, note: str | None = None) -> dict:
    """A minimal valid QTable: every field absent (the model completes everything), fc 未说明 in every row."""
    H = max(1, min(int(n_quarters or DEFAULT_QUARTERS), cq.QTABLE_MAX_QUARTERS))
    qt = {"schema": SCHEMA_ID, "model_card": cq.CARD_ID, "n_quarters": H,
          "rows": [{"q": q, "x": {}, "fc": {"stance": "未说明"}} for q in range(1, H + 1)],
          "c_q": {}, "events": [], "notes": []}
    if note:
        qt["notes"].append(str(note)[:NOTE_MAX])
    return qt


def schema_errors(qt, card: dict | None = None) -> list[str]:
    """The QTable/1 JSON schema's errors for `qt` (the card-generated schema; jsonfix's validator)."""
    from ..llm import jsonfix
    return [str(e) for e in jsonfix.validate(qt, qcard.qtable_json_schema(_card(card)))]


def digest(qt: dict) -> str:
    return hashlib.sha256(json.dumps(qt, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


# ================================================================ validation

class _Ctx:
    """One validation pass: the card's field classes, the narrative, the mode and the collected issues."""

    def __init__(self, narrative, mode, source, stats, card, numbers=True):
        if mode not in MODES:
            raise QTableError(f"未知的模式 {mode!r}（production 或 test）")
        if source not in SOURCES:
            raise QTableError(f"未知的来源 {source!r}（{' / '.join(SOURCES)}）")
        self.card = _card(card)
        self.narrative = None if narrative is None else str(narrative)
        self.mode, self.source = mode, source
        self.numbers = bool(numbers)
        self.stats = stats if stats is not None else stats_table()
        self.x = set(qcard.x_fields(self.card))
        self.u = set(qcard.shocks(self.card))
        self.rows = {r["id"]: r for r in self.card["rows"]}
        self.states = {r["id"] for r in qcard.rows(self.card, "s")}
        self.outputs = {r["id"] for r in qcard.rows(self.card, "y")} | {r["id"] for r in qcard.rows(self.card, "v")}
        self.errors: list[dict] = []
        self.refused: list[dict] = []
        self.warnings: list[dict] = []
        self.clipped: list[dict] = []
        self.dropped: list[dict] = []
        self.counts = {"entries": 0, "stated": 0, "stated_qual": 0, "derived": 0}
        self.binned: list[dict] = []           # legacy N3 numbers re-expressed as size bins

    def issue(self, level: str, code: str, text: str, where: str = "", path: str = "", **extra) -> dict:
        it = {"code": code, "level": level, "where": where, "path": path, "text": text,
              **{k: v for k, v in extra.items() if v is not None}}
        {"error": self.errors, "refused": self.refused, "warning": self.warnings}[level].append(it)
        return it

    def refuse(self, code, text, where="", path="", category="rule", **extra):
        return self.issue("refused", code, text, where, path, category=category, **extra)

    def warn(self, code, text, where="", path="", **extra):
        return self.issue("warning", code, text, where, path, **extra)

    # the quote rules shared by every quoted item
    def quote_ok(self, quote, where: str, path: str, *, field: str | None = None, required: bool = True,
                 words=None, zh: str | None = None, q: int | None = None, number: str | None = None) -> str | None:
        """The quote to keep (cut to 120 characters), or None when the item is refused. number: the src of a value
        written as a number (stated / derived): a translation's quote must then carry a number (R-SRC)."""
        if self.source == "legacy":
            return str(quote or "")[:cq.QUOTE_MAX]
        if not isinstance(quote, str) or not quote.strip():
            if not required:
                return ""
            self.refuse("R-QUOTE", MSG["quote_missing"], where, path, field=field, q=q)
            return None
        quote = quote.strip()
        if self.source == "translator" and len(norm(quote)) < QUOTE_MIN:
            self.refuse("R-QUOTE", MSG["quote_short"].format(n=QUOTE_MIN), where, path, field=field, q=q,
                        quote=quote[:cq.QUOTE_MAX])
            return None
        if self.narrative is not None and not quote_in(quote, self.narrative):
            self.refuse("R-QUOTE", MSG["quote"], where, path, field=field, q=q, quote=quote[:cq.QUOTE_MAX])
            return None
        if words is not None and self.source != "user" and not names(quote, words):
            self.refuse("R-E1", MSG["e1"].format(zh=zh or field or ""), where, path, field=field, q=q,
                        quote=quote[:cq.QUOTE_MAX])
            return None
        if number and self.source == "translator" and not NUMERAL.search(norm(quote)):
            self.refuse("R-SRC", MSG["src_number"].format(src=number), where, path, field=field, q=q,
                        quote=quote[:cq.QUOTE_MAX])
            return None
        if self.mode == "test":
            hits = scan_blind(quote)
            if hits:
                self.refuse("R-YEAR", MSG["year"].format(tok=hits[0][1]), where, path, field=field, q=q)
                return None
        if len(quote) > cq.QUOTE_MAX:
            self.warn("W-QUOTE", MSG["quote_cut"].format(n=cq.QUOTE_MAX), where, path, field=field, q=q)
            quote = quote[:cq.QUOTE_MAX]
        return quote

    def keywords(self, field: str):
        r = self.rows.get(field) or {}
        return r.get("keywords") if r.get("permission") == "stated_only" else None

    def zh(self, field: str) -> str:
        r = self.rows.get(field) or {}
        return f"「{r.get('zh', field)}」"

    def name(self, field: str) -> str:
        """The field's Chinese name alone (reader text: no code id)."""
        return str((self.rows.get(field) or {}).get("zh") or ANCHOR_ZH.get(field) or field)

    def endo(self, field: str) -> str | None:
        return (self.rows.get(field) or {}).get("endo_code")

    def band(self, field: str):
        s = self.stats.get(field) or {}
        lo, hi = s.get("lo"), s.get("hi")
        if lo is None or hi is None:
            return None
        c, w = (lo + hi) / 2.0, (hi - lo) / 2.0 * RANGE_WIDEN
        return c - w, c + w

    def ranged(self, field: str, v: float, where: str, path: str, q: int | None = None, scale: float = 1.0,
               stat: str | None = None):
        """(the value to keep, ok): beyond 1.5 x the historical range -> refused in test mode, clipped otherwise.
        stat: the statistics row to read (a y/y constraint reads its level's range)."""
        b = self.band(stat or field)
        if b is None or v is None:
            return v, True
        lo, hi = b[0] * scale, b[1] * scale
        if lo <= v <= hi:
            return v, True
        if self.mode == "test":
            self.refuse("R-RANGE", MSG["range_test"].format(v=v, lo=lo, hi=hi), where, path, field=field, q=q)
            return v, False
        used = float(min(max(v, lo), hi))
        self.warn("W-RANGE", MSG["range_clip"].format(v=v, used=used), where, path, field=field, q=q)
        self.clipped.append({"where": where, "path": path, "field": field, "asked": v, "used": used})
        return used, True

    def shock_ranged(self, u: str, v: float, where: str, path: str, q: int | None = None):
        """A policy shock against ±SHOCK_RANGE_SD shock sd (the factor layer saturates there): refused in test mode,
        clipped otherwise; no shock sd (no fit) -> not checked."""
        sd = (self.stats.get(u) or {}).get("sd")
        if sd is None or not math.isfinite(float(sd)) or sd <= 0 or v is None:
            return v, True
        k = SHOCK_RANGE_SD
        lo, hi = -k * float(sd), k * float(sd)
        if lo <= v <= hi:
            return v, True
        if self.mode == "test":
            self.refuse("R-RANGE", MSG["u_range_test"].format(v=v, k=k, lo=lo, hi=hi), where, path, field=u, q=q)
            return v, False
        used = float(min(max(v, lo), hi))
        self.warn("W-RANGE", MSG["u_range_clip"].format(v=v, k=k, used=used), where, path, field=u, q=q)
        self.clipped.append({"where": where, "path": path, "field": u, "asked": v, "used": used})
        return used, True

    def n3_bin(self, f: str, out: dict, where: str, path: str, q: int | None):
        """numbers=False: an E1 / E1c / E2 value or range. Legacy (the research round's stand-ins): the value as the
        nearest size bin in training sd (W-N3); otherwise refused (R-N3). Returns the entry to keep or None."""
        sd = (self.stats.get(f) or {}).get("sd")
        if self.source == "legacy" and "value" in out and sd and math.isfinite(float(sd)) and sd > 0:
            z = out["value"] / float(sd)
            if abs(z) < cq.SIZE_BINS["小"] / 2.0:
                sign, size = "0", None
            else:
                sign = "+" if z > 0 else "-"
                size = min(cq.SIZE_BINS, key=lambda k: abs(cq.SIZE_BINS[k] - abs(z)))
            new = {"sign": sign, **({"size": size} if size else {}), "src": "stated_qual"}
            self.binned.append({"q": q, "field": f, "asked": out["value"], "sign": sign, "size": size})
            self.warn("W-N3", MSG["n3_bin"].format(zh=self.name(f), v=out["value"], sign=sign, size=size or ""),
                      where, path, field=f, q=q)
            return new
        self.refuse("R-N3", MSG["n3"].format(zh=self.name(f)), where, path, field=f, q=q)
        return None


def field_name(field: str) -> str:
    """A field's Chinese name (the card's zh; anchors, the stance and the c_q items in words): reader text never shows
    a code id. An id the card does not know is shown as written (it is what the table wrote)."""
    try:
        rows = {r["id"]: r for r in _card()["rows"]}
    except Exception:  # noqa: BLE001 - no card
        rows = {}
    if field in rows and rows[field].get("zh"):
        return str(rows[field]["zh"])
    return {"fc": "金融条件", "valuation": "估值陈述", "regime": "体制请求", **ANCHOR_ZH}.get(field, field)


def _where(q: int | None, field: str | None = None) -> str:
    parts = [f"第 {q} 季"] if q else []
    if field:
        parts.append(field_name(field))
    return " · ".join(parts)


def _entry(ctx: _Ctx, f: str, e, q: int, path: str, *, shock: bool) -> dict | None:
    """One x or u entry, checked; the clean entry or None (refused)."""
    where = _where(q, f)
    if not shock and f not in ctx.x:
        if f in ctx.u:
            ctx.refuse("R-FIELD", f"{f} 是政策冲击：写在 u 里，不写在 x 里", where, path, "form", field=f, q=q)
        elif f in ctx.outputs or f.startswith("y_") or f in cq.ASSETS or f in ("s_pe", "s_cape", "s_pd", "pe", "cape"):
            ctx.refuse("R-E3", MSG["e3"].format(f=f), where, path, field=f, q=q)
        elif f in ctx.states:
            ctx.refuse("R-STATE", MSG["state"].format(f=field_name(f)), where, path, field=f, q=q)
        else:
            ctx.refuse("R-FIELD", MSG["field_x"].format(f=f), where, path, "form", field=f, q=q)
        return None
    if shock and f not in ctx.u:
        ctx.refuse("R-FIELD", MSG["shock"].format(f=f), where, path, "form", field=f, q=q)
        return None
    if not isinstance(e, dict):
        ctx.refuse("R-FORM", MSG["form"], where, path, "form", field=f, q=q)
        return None
    if f == "x_oil_cause":
        return _oil_cause(ctx, e, q, path)
    bad = sorted(set(e) - set(ENTRY_KEYS))
    if bad:
        ctx.refuse("R-FORM", MSG["keys"].format(keys="、".join(bad)), where, path, "form", field=f, q=q)
        return None
    has_val = e.get("value") is not None
    has_sign = e.get("sign") is not None or e.get("size") is not None
    has_rng = e.get("min") is not None or e.get("max") is not None
    if has_val + has_sign + has_rng != 1:
        ctx.refuse("R-FORM", MSG["form"], where, path, "form", field=f, q=q)
        return None
    out: dict = {}
    if has_val:
        if not _finite(e["value"]):
            ctx.refuse("R-FORM", MSG["number"].format(k="value", v=e["value"]), where, path, "form", field=f, q=q)
            return None
        out["value"] = float(e["value"])
        if e.get("sd") is not None:
            if not _finite(e["sd"]) or e["sd"] <= 0:
                ctx.refuse("R-FORM", MSG["sd"].format(v=e["sd"]), where, path, "form", field=f, q=q)
                return None
            out["sd"] = float(e["sd"])
    elif has_sign:
        sign, size = e.get("sign"), e.get("size")
        if sign not in SIGNS:
            ctx.refuse("R-FORM", MSG["sign"].format(v=sign), where, path, "form", field=f, q=q)
            return None
        if size is None and sign != "0":
            ctx.refuse("R-SIZE", MSG["no_size"], where, path, "form", field=f, q=q)
            return None
        if size is not None and size not in cq.SIZE_BINS:
            ctx.refuse("R-SIZE", MSG["size"].format(v=size), where, path, "form", field=f, q=q)
            return None
        out["sign"] = sign
        if size is not None:
            out["size"] = size
    else:
        for k in ("min", "max"):
            if e.get(k) is not None:
                if not _finite(e[k]):
                    ctx.refuse("R-FORM", MSG["number"].format(k=k, v=e[k]), where, path, "form", field=f, q=q)
                    return None
                out[k] = float(e[k])
        if "min" in out and "max" in out and out["min"] > out["max"]:
            ctx.refuse("R-FORM", MSG["minmax"], where, path, "form", field=f, q=q)
            return None
    src = e.get("src")
    if ctx.source == "legacy" and src is None:
        src = "stated"
    if src not in cq.SOURCES_OF_VALUES:
        ctx.refuse("R-SRC", MSG["src"], where, path, "form", field=f, q=q)
        return None
    if src == "stated_qual" and not has_sign:
        ctx.refuse("R-SRC", MSG["qual_number"], where, path, field=f, q=q)
        return None
    if not ctx.numbers and not shock and not has_sign and ctx.endo(f) in NUMBER_CLASSES:
        binned = ctx.n3_bin(f, out, where, path, q)          # N3: market prices and surveys only as size bins
        if binned is None:
            return None
        out, src, has_sign = binned, binned["src"], True
    out["src"] = src
    words = None if shock else ctx.keywords(f)
    quote = ctx.quote_ok(e.get("quote"), where, path, field=f, words=words, zh=ctx.zh(f), q=q,
                         number=src if src in ("stated", "derived") and not has_sign else None)
    if quote is None:
        return None
    out["quote"] = quote
    if shock and ctx.source not in ("user", "legacy") and not names(quote, INTERVENTION_WORDS.get(f, ())):
        ctx.warn("W-U", MSG["intervention"].format(f=field_name(f)), where, path, field=f, q=q)
    for k in ("value", "min", "max"):
        if k in out:
            v, ok = (ctx.shock_ranged(f, out[k], where, path, q) if shock else ctx.ranged(f, out[k], where, path, q))
            if not ok:
                return None
            out[k] = v
    ctx.counts["entries"] += 1
    ctx.counts[src] = ctx.counts.get(src, 0) + 1
    return out


def _item_keys(ctx: _Ctx, item: dict, kind: str, where: str, path: str) -> None:
    """Unknown keys of a c_q item: a warning (the clean item keeps only the known keys)."""
    extra = sorted(k for k in item if k not in CQ_ITEM_KEYS[kind])
    if extra:
        ctx.warn("W-FORM", MSG["item_keys"].format(what=where, keys="、".join(extra)), where, path)


def _oil_cause(ctx: _Ctx, e: dict, q: int, path: str) -> dict | None:
    where = _where(q, "x_oil_cause")
    v = e.get("value")
    if v not in cq.OIL_CAUSE:
        ctx.refuse("R-FORM", MSG["oil_cause"], where, path, "form", field="x_oil_cause", q=q)
        return None
    bad = sorted(set(e) - {"value", "src", "quote"})
    if bad:
        ctx.refuse("R-FORM", MSG["keys"].format(keys="、".join(bad)), where, path, "form", field="x_oil_cause", q=q)
        return None
    src = e.get("src") or ("stated" if ctx.source == "legacy" else None)
    if src not in ("stated", "stated_qual"):
        ctx.refuse("R-SRC", MSG["oil_src"], where, path, "form", field="x_oil_cause", q=q)
        return None
    quote = ctx.quote_ok(e.get("quote"), where, path, field="x_oil_cause", q=q)
    if quote is None:
        return None
    ctx.counts["entries"] += 1
    ctx.counts[src] = ctx.counts.get(src, 0) + 1
    return {"value": v, "src": src, "quote": quote}


def _fc(ctx: _Ctx, fc, q: int, path: str) -> dict:
    where = _where(q, "fc")
    if not isinstance(fc, dict):
        return {"stance": "未说明"}
    stance = fc.get("stance", "未说明")
    if stance not in cq.FC_STANCES:
        ctx.refuse("R-FORM", MSG["fc_stance"].format(stances=" / ".join(cq.FC_STANCES), v=stance), where, path,
                   "form", field="fc", q=q)
        return {"stance": "未说明"}
    if stance == "未说明":
        out = {"stance": "未说明"}
        if isinstance(fc.get("quote"), str) and fc["quote"].strip():
            quote = ctx.quote_ok(fc["quote"], where, path, field="fc", required=False, q=q)
            if quote:
                out["quote"] = quote
        return out
    words = tuple(ctx.keywords("d_gz") or ()) + tuple(ctx.keywords("d_spread") or ()) + FC_WORDS
    quote = ctx.quote_ok(fc.get("quote"), where, path, field="fc", words=words, zh="金融条件或信用利差", q=q)
    if quote is None:
        return {"stance": "未说明"}
    ctx.counts["entries"] += 1
    return {"stance": stance, "quote": quote}


def _quarters(ctx: _Ctx, qs, H: int, where: str, path: str) -> tuple | None:
    if not isinstance(qs, (list, tuple)) or not qs:
        ctx.refuse("R-FORM", MSG["quarters"].format(H=H), where, path, "form")
        return None
    out = []
    for v in qs:
        k = _int(v)
        if k is None or not 1 <= k <= H:
            ctx.refuse("R-FORM", MSG["quarters"].format(H=H), where, path, "form")
            return None
        out.append(k)
    return tuple(sorted(set(out)))


def _rows(ctx: _Ctx, rows: list, H: int) -> list[dict]:
    by_q: dict[int, dict] = {}
    for i, r in enumerate(rows):
        path = f"rows[{i}]"
        if not isinstance(r, dict):
            ctx.refuse("R-SHAPE", MSG["row_shape"], f"第 {i + 1} 行", path, "form")
            continue
        q = _int(r.get("q"))
        if q is None or not 1 <= q <= H:
            ctx.refuse("R-SHAPE", MSG["row_q"].format(H=H, v=r.get("q")), f"第 {i + 1} 行", path, "form")
            continue
        if q in by_q:
            ctx.refuse("R-SHAPE", MSG["row_dup"].format(q=q), _where(q), path, "form", q=q)
            continue
        extra = sorted(set(r) - {"q", "x", "u", "fc"})
        if extra:
            ctx.refuse("R-FORM", MSG["keys"].format(keys="、".join(extra)), _where(q), path, "form", q=q)
        x_in = r.get("x") if isinstance(r.get("x"), dict) else {}
        u_in = r.get("u") if isinstance(r.get("u"), dict) else {}
        if r.get("x") is not None and not isinstance(r.get("x"), dict):
            ctx.refuse("R-FORM", "rows[].x 必须是对象（字段 → 条目）", _where(q), f"{path}.x", "form", q=q)
        if r.get("u") is not None and not isinstance(r.get("u"), dict):
            ctx.refuse("R-FORM", "rows[].u 必须是对象（冲击 → 条目）", _where(q), f"{path}.u", "form", q=q)
        x, u = {}, {}
        for f, e in x_in.items():
            ce = _entry(ctx, str(f), e, q, f"{path}.x.{f}", shock=False)
            if ce is not None:
                x[str(f)] = ce
        for f, e in u_in.items():
            ce = _entry(ctx, str(f), e, q, f"{path}.u.{f}", shock=True)
            if ce is not None:
                u[str(f)] = ce
        fc = _fc(ctx, r.get("fc"), q, f"{path}.fc")
        _identities(ctx, x, u, q, path)
        row = {"q": q, "x": x, "fc": fc}
        if u:
            row["u"] = u
        by_q[q] = row
    return [by_q.get(q) or {"q": q, "x": {}, "fc": {"stance": "未说明"}} for q in range(1, H + 1)]


def _identities(ctx: _Ctx, x: dict, u: dict, q: int, path: str) -> None:
    """Schema rule 2 (output = productivity x labour input; deficit = spending + transfers - taxes) and the QE rule."""
    labour = [f for f in ("d_lf", "d_unrate") if f in x]
    if "d_rgdp" in x and "d_lprod" in x and labour:
        ctx.dropped.append({"q": q, "field": "d_lprod", "entry": x.pop("d_lprod"), "rule": "W-IDENT"})
        ctx.warn("W-IDENT", MSG["ident_output"].format(q=q), _where(q, "d_lprod"), f"{path}.x.d_lprod",
                 field="d_lprod", q=q)
    if "d_deficit" in x and any(f in x for f in ("d_defense", "d_transfers", "d_perstax")):
        ctx.dropped.append({"q": q, "field": "d_deficit", "entry": x.pop("d_deficit"), "rule": "W-IDENT"})
        ctx.warn("W-IDENT", MSG["ident_deficit"].format(q=q), _where(q, "d_deficit"), f"{path}.x.d_deficit",
                 field="d_deficit", q=q)
    if "u_bs" in u and not any(f in x for f in ("d_tp10", "d_y2", "d_gz")):
        ctx.warn("W-QE", MSG["qe"].format(q=q), _where(q, "u_bs"), f"{path}.u.u_bs", field="u_bs", q=q)


def _anchors(ctx: _Ctx, anchors, H: int) -> dict:
    out = {}
    if anchors is None:
        return out
    if not isinstance(anchors, dict):
        ctx.refuse("R-FORM", "c_q.anchors 必须是对象（锚 → {value, from_q, quote}）", "c_q.anchors", "c_q.anchors", "form")
        return out
    for a, e in anchors.items():
        path, where = f"c_q.anchors.{a}", f"长期锚「{field_name(a)}」"
        if a not in cq.QTABLE_ANCHORS:
            ctx.refuse("R-FIELD", MSG["field"].format(f=a), where, path, "form", field=a)
            continue
        if not isinstance(e, dict) or not _finite(e.get("value")):
            ctx.refuse("R-FORM", MSG["number"].format(k="value", v=(e or {}).get("value") if isinstance(e, dict) else e),
                       where, path, "form", field=a)
            continue
        fq = _int(e.get("from_q", 1))
        if fq is None or not 1 <= fq <= H:
            ctx.refuse("R-FORM", MSG["quarters"].format(H=H), where, path, "form", field=a)
            continue
        _item_keys(ctx, e, "anchors", where, path)
        quote = ctx.quote_ok(e.get("quote"), where, path, field=a)
        if quote is None:
            continue
        v = float(e["value"])
        lo, hi = ANCHOR_BOUNDS[a]
        if not lo <= v <= hi:
            if ctx.mode == "test":
                ctx.refuse("R-RANGE", MSG["anchor_range_test"].format(a=a, v=v, lo=lo, hi=hi), where, path, field=a)
                continue
            used = float(min(max(v, lo), hi))
            ctx.warn("W-RANGE", MSG["anchor_range"].format(a=a, v=v, lo=lo, hi=hi, used=used), where, path, field=a)
            ctx.clipped.append({"where": where, "path": path, "field": a, "asked": v, "used": used})
            v = used
        out[a] = {"value": v, "from_q": fq, "quote": quote}
    return out


def _announced(ctx: _Ctx, items, H: int) -> list:
    out = []
    for i, a in enumerate(items or []):
        path = f"c_q.announced[{i}]"
        if not isinstance(a, dict):
            ctx.refuse("R-FORM", "预告路径必须是对象", "预告路径", path, "form")
            continue
        f = str(a.get("field"))
        where = f"预告路径「{field_name(f)}」"
        if f not in ctx.x or f == "x_oil_cause":
            ctx.refuse("R-FIELD", MSG["field"].format(f=f), where, path, "form", field=f)
            continue
        if not ctx.numbers and ctx.endo(f) in NUMBER_CLASSES:
            ctx.refuse("R-N3", MSG["n3"].format(zh=ctx.name(f)), where, path, field=f)
            continue
        _item_keys(ctx, a, "announced", where, path)
        qs = a.get("quarters")
        vals = a.get("values")
        if not isinstance(qs, (list, tuple)) or not isinstance(vals, (list, tuple)) or len(qs) != len(vals):
            ctx.refuse("R-FORM", MSG["values_len"].format(f=f), where, path, "form", field=f)
            continue
        quarters = _quarters(ctx, qs, H, where, path)
        if quarters is None or len(quarters) != len(qs):
            if quarters is not None:
                ctx.refuse("R-FORM", MSG["quarters"].format(H=H), where, path, "form", field=f)
            continue
        if any(v is not None and not _finite(v) for v in vals):
            ctx.refuse("R-FORM", MSG["number"].format(k="values", v=vals), where, path, "form", field=f)
            continue
        sd = a.get("sd")
        if sd is not None and (not _finite(sd) or sd <= 0):
            ctx.refuse("R-FORM", MSG["sd"].format(v=sd), where, path, "form", field=f)
            continue
        quote = ctx.quote_ok(a.get("quote"), where, path, field=f, words=ctx.keywords(f), zh=ctx.zh(f))
        if quote is None:
            continue
        clean_vals, ok = [], True
        for v in vals:
            if v is None:
                clean_vals.append(None)
                continue
            used, ok = ctx.ranged(f, float(v), where, path)
            if not ok:
                break
            clean_vals.append(used)
        if not ok:
            continue
        pairs = sorted(zip((int(q) for q in qs), clean_vals))
        item = {"field": f, "quarters": [p[0] for p in pairs], "values": [p[1] for p in pairs], "quote": quote}
        if sd is not None:
            item["sd"] = float(sd)
        out.append(item)
    return out


def _constraints(ctx: _Ctx, items, H: int) -> list:
    out = []
    allowed = ctx.x | set(cq.CONSTRAINT_STATES)
    for i, c in enumerate(items or []):
        path = f"c_q.constraints[{i}]"
        if not isinstance(c, dict):
            ctx.refuse("R-FORM", "约束必须是对象", "约束", path, "form")
            continue
        kind, f = c.get("kind"), str(c.get("field"))
        where = f"约束「{field_name(f)}」"
        if kind not in CONSTRAINT_KINDS:
            ctx.refuse("R-FORM", MSG["kind"], where, path, "form", field=f)
            continue
        if f not in allowed or f == "x_oil_cause":
            ctx.refuse("R-FIELD", MSG["field"].format(f=f), where, path, "form", field=f)
            continue
        if kind == "yoy" and f not in YOY_VARS:
            ctx.refuse("R-FORM", MSG["yoy"].format(f=f), where, path, "form", field=f)
            continue
        if not ctx.numbers and ctx.endo(f) in NUMBER_CLASSES:
            ctx.refuse("R-N3", MSG["n3"].format(zh=ctx.name(f)), where, path, field=f)
            continue
        _item_keys(ctx, c, "constraints", where, path)
        quarters = _quarters(ctx, c.get("quarters"), H, where, path)
        if quarters is None:
            continue
        item = {"kind": kind, "field": f, "quarters": list(quarters)}
        bad = False
        for k in ("value", "sd", "min", "max"):
            if c.get(k) is not None:
                if not _finite(c[k]) or (k == "sd" and c[k] <= 0):
                    ctx.refuse("R-FORM", (MSG["sd"] if k == "sd" else MSG["number"]).format(k=k, v=c[k]),
                               where, path, "form", field=f)
                    bad = True
                    break
                item[k] = float(c[k])
        if bad:
            continue
        if kind == "range" and "min" not in item and "max" not in item:
            ctx.refuse("R-FORM", MSG["range_kind"], where, path, "form", field=f)
            continue
        if kind != "range" and "value" not in item and "min" not in item and "max" not in item:
            ctx.refuse("R-FORM", MSG["no_value"], where, path, "form", field=f)
            continue
        if "min" in item and "max" in item and item["min"] > item["max"]:
            ctx.refuse("R-FORM", MSG["minmax"], where, path, "form", field=f)
            continue
        words = ctx.keywords(f) if f in ctx.x else None
        quote = ctx.quote_ok(c.get("quote"), where, path, field=f, words=words, zh=ctx.zh(f))
        if quote is None:
            continue
        item["quote"] = quote
        scale = float(len(quarters)) if kind == "sum" else 1.0
        ok = True
        for k in ("value", "min", "max"):
            if k in item:
                item[k], ok = ctx.ranged(f, item[k], where, path, scale=scale,
                                         stat=YOY_RANGE.get(f) if kind == "yoy" else None)
                if not ok:
                    break
        if ok:
            out.append(item)
    return out


def _valuation(ctx: _Ctx, items, H: int) -> list:
    from . import valuation as val
    out = []
    for i, e in enumerate(items or []):
        path, where = f"c_q.valuation[{i}]", "估值陈述"
        if not isinstance(e, dict):
            ctx.refuse("R-FORM", "估值陈述必须是对象", where, path, "form")
            continue
        extra = sorted(k for k in e if k not in VALUATION_KEYS)
        numeric = [k for k in extra if _finite(e[k])]
        if numeric:
            ctx.refuse("R-VAL", MSG["val_number"].format(keys="、".join(numeric)), where, path)
            continue
        if extra:
            ctx.warn("W-FORM", MSG["item_keys"].format(what="估值陈述", keys="、".join(extra)), where, path)
            e = {k: v for k, v in e.items() if k in VALUATION_KEYS}
        try:
            st = val.parse_statements([e], H, mode="production")[0]
        except ValueError as err:
            ctx.refuse("R-FORM", str(err), where, path, "form")
            continue
        if st.active and ctx.mode == "test":
            ctx.refuse("R-VAL", MSG["val_test"], where, path)
            continue
        quote = ctx.quote_ok(e.get("quote"), where, path, field="valuation")
        if quote is None:
            continue
        item = {"quarters": list(st.quarters), "mode": st.mode, "quote": quote}
        if st.mode == "rerate":
            item["direction"] = "+" if st.direction > 0 else "-"
            item["size"] = st.size
        if st.half_life_years is not None:
            item["reversion_half_life_years"] = float(st.half_life_years)
        if st.active:
            if st.mode == "hold":
                what = "倍数不变"
            elif st.mode == "rerate":
                what = f"倍数{'上' if st.direction > 0 else '下'}调（{st.size}）"
            else:
                what = f"均值回归，半衰期 {st.half_life_years:g} 年"
            ctx.warn("W-VAL", MSG["val_warn"].format(qs=_span(st.quarters), what=what), where, path)
        out.append(item)
    try:                                        # overlapping hold / rerate / reversion quarters
        val.parse_statements(out, H, mode="production")
    except ValueError as err:
        ctx.refuse("R-FORM", str(err), "估值陈述", "c_q.valuation", "form")
        return []
    return out


def _regime(ctx: _Ctx, items, H: int) -> list:
    out = []
    for i, rg in enumerate(items or []):
        path, where = f"c_q.regime[{i}]", "体制请求"
        if not isinstance(rg, dict):
            ctx.refuse("R-FORM", "体制请求必须是对象", where, path, "form")
            continue
        label = rg.get("label")
        if label not in cq.REGIME_LABELS:
            alias = REGIME_ALIASES.get(str(label or "").strip())
            if alias is None:
                ctx.refuse("R-FORM", MSG["regime_label"].format(labels=" / ".join(cq.REGIME_LABELS), v=label), where,
                           path, "form")
                continue
            ctx.warn("W-REGIME", MSG["regime_alias"].format(v=label, id=alias, zh=REGIME_ZH[alias]), where, path)
            label = alias
        _item_keys(ctx, rg, "regime", where, path)
        share = rg.get("share")
        if not _finite(share) or not 0.0 <= share <= 1.0:
            ctx.refuse("R-FORM", MSG["regime_share"].format(v=share), where, path, "form")
            continue
        quarters = _quarters(ctx, rg.get("quarters"), H, where, path)
        if quarters is None:
            continue
        quote = ctx.quote_ok(rg.get("quote"), where, path, field="regime")
        if quote is None:
            continue
        out.append({"quarters": list(quarters), "label": label, "share": float(share), "quote": quote})
    return out


def _events(ctx: _Ctx, items, H: int, rows: list) -> list:
    out = []
    stated = {r["q"]: set(r["x"]) | set(r.get("u") or {}) for r in rows}
    for i, ev in enumerate(items or []):
        path = f"events[{i}]"
        if not isinstance(ev, dict) or not isinstance(ev.get("text"), str) or not ev["text"].strip():
            ctx.refuse("R-FORM", MSG["event_shape"], "事件", path, "form")
            continue
        text = ev["text"].strip()[:EVENT_TEXT_MAX]
        how = str(ev.get("how_decomposed") or "")[:EVENT_TEXT_MAX]
        if ctx.mode == "test":
            hits = scan_blind([text, how])
            if hits:
                ctx.refuse("R-YEAR", MSG["year"].format(tok=hits[0][1]), f"事件「{text[:20]}」", path)
                continue
        qs = ev.get("quarters") or []
        quarters = tuple(sorted({k for k in (_int(v) for v in qs) if k is not None and 1 <= k <= H}))
        templates = []
        for g in ev.get("templates") or []:
            if g in template_ids():
                templates.append(g)
            else:
                ctx.warn("W-EVENT", MSG["event_template"].format(t=text[:20], g=g), f"事件「{text[:20]}」", path)
        channels = []
        known = ctx.x | ctx.u
        for c in ev.get("channels") or []:
            if c in known:
                channels.append(c)
            else:
                ctx.warn("W-EVENT", MSG["event_channel"].format(t=text[:20], c=c), f"事件「{text[:20]}」", path)
        foot = ev.get("generic_footprint") is True
        if not foot:
            ctx.warn("W-EVENT", MSG["event_nofoot"].format(t=text[:20]), f"事件「{text[:20]}」", path)
        elif channels and quarters:
            missing = [q for q in quarters if not (stated.get(q, set()) & set(channels))]
            if missing:
                ctx.warn("W-EVENT", MSG["event_unstated"].format(t=text[:20], qs=_span(missing)),
                         f"事件「{text[:20]}」", path)
        out.append({"text": text, "quarters": list(quarters), "templates": templates, "channels": channels,
                    "generic_footprint": foot, "how_decomposed": how})
    return out


def _notes(ctx: _Ctx, items) -> list:
    out = []
    for i, n in enumerate(items or []):
        if not isinstance(n, str) or not n.strip():
            continue
        if ctx.mode == "test":
            hits = scan_blind(n)
            if hits:
                ctx.refuse("R-YEAR", MSG["year"].format(tok=hits[0][1]), "备注", f"notes[{i}]")
                continue
        if len(n) > NOTE_MAX:
            ctx.warn("W-NOTE", MSG["note_cut"].format(n=NOTE_MAX), "备注", f"notes[{i}]")
        out.append(n.strip()[:NOTE_MAX])
    return out


def _span(qs) -> str:
    """[1, 2, 3, 5] -> "1–3、5"."""
    qs = sorted(set(int(q) for q in qs))
    parts, start, prev = [], None, None
    for q in qs:
        if start is None:
            start = prev = q
        elif q == prev + 1:
            prev = q
        else:
            parts.append(f"{start}" if start == prev else f"{start}–{prev}")
            start = prev = q
    if start is not None:
        parts.append(f"{start}" if start == prev else f"{start}–{prev}")
    return "、".join(parts)


def _covers(items, fields, q) -> bool:
    return any(it["field"] in fields and q in it["quarters"] for it in items)


def _requirements(ctx: _Ctx, clean: dict) -> list:
    """R1-R5 per quarter (§7.1): one warning per requirement listing the quarters it leaves to the model."""
    rows, cqx = clean["rows"], clean["c_q"]
    cons = cqx.get("constraints", []) + [{"field": a["field"], "quarters": a["quarters"]}
                                         for a in cqx.get("announced", [])]
    regime = cqx.get("regime", [])
    text_r1 = " ".join(clean.get("notes", []) + [e["text"] + e["how_decomposed"] for e in clean.get("events", [])])
    explicit_r1 = "反应函数" in text_r1
    miss: dict[str, list] = {k: [] for k in ("R1", "R2", "R3", "R4", "R5", "oil")}
    for r in rows:
        q, x, u = r["q"], r["x"], r.get("u") or {}
        if not ("d_ff" in x or "u_ff" in u or _covers(cons, ("d_ff", "s_ff"), q) or explicit_r1):
            miss["R1"].append(q)
        if not (any(f in x for f in GROWTH_X) or _covers(cons, GROWTH_X + ("s_unrate",), q)
                or any(q in rg["quarters"] for rg in regime)):
            miss["R2"].append(q)
        if not (any(f in x for f in INFL_X) or _covers(cons, INFL_X + ("s_cpi_yoy", "s_core_yoy"), q)
                or any(q in rg["quarters"] and rg["label"] in INFL_LABELS for rg in regime)):
            miss["R3"].append(q)
        if not ("d_oil" in x or _covers(cons, ("d_oil",), q)):
            miss["R4"].append(q)
        elif "d_oil" in x and "x_oil_cause" not in x:
            miss["oil"].append(q)
        if r["fc"]["stance"] == "未说明" and not (any(f in x for f in FC_X) or _covers(cons, FC_X, q)):
            miss["R5"].append(q)
    out = []
    for rule, qs in miss.items():
        if not qs:
            continue
        code = {"R5": "W-FC", "oil": "W-OIL"}.get(rule, f"W-{rule}")
        text = MSG["fc"] if rule == "R5" else MSG[rule]
        ev = evidence(ctx.narrative, "R4" if rule == "oil" else rule) if ctx.narrative else None
        item = {"code": code, "rule": "R4" if rule == "oil" else rule, "quarters": qs,
                "text": f"第 {_span(qs)} 季：{text}",
                "evidence": ev if ev else (MSG["no_evidence"] if ctx.narrative else None)}
        out.append(item)
        ctx.warn(code, item["text"], f"第 {_span(qs)} 季", "rows", quarters=qs, evidence=item["evidence"])
    return out


def validate(qt, narrative: str | None = None, *, mode: str = "production", source: str = "translator",
             stats: dict | None = None, card: dict | None = None, numbers: bool = True) -> dict:
    """The validation report of a QTable (§7.3) with its clean form.

    narrative: the text the quotes must come from (None: quotes are not checked, W-QUOTE). Refused entries are
    removed from the clean QTable; clipped values are clipped; one of an over-determined identity is dropped. The
    report: ok (nothing refused, no errors), readable (a clean QTable exists), errors, refused, warnings,
    missing (R1-R5 with evidence), clipped, dropped, anchors (recorded), counts, n_quarters, qtable (the clean
    QTable; it validates against the card-generated JSON schema). numbers=False (the narrative test's N3 arm):
    E1 / E1c / E2 fields only as sign + size (R-N3; the research round's stand-ins re-expressed, W-N3)."""
    ctx = _Ctx(narrative, mode, source, stats, card, numbers)
    report = {"schema": SCHEMA_ID, "mode": mode, "source": source, "ok": False, "readable": False}

    def finish(clean):
        cqx_ = (clean or {}).get("c_q") or {}
        report.update(errors=ctx.errors, refused=ctx.refused, warnings=ctx.warnings, clipped=ctx.clipped,
                      dropped=ctx.dropped, binned=ctx.binned, numbers=ctx.numbers, counts={**ctx.counts, **{k: len(cqx_.get(k) or ()) for k in CQ_KEYS},
                                                   "events": len((clean or {}).get("events") or ()),
                                                   "refused": len(ctx.refused), "warnings": len(ctx.warnings)},
                      qtable=clean)
        report["readable"] = clean is not None
        report["ok"] = clean is not None and not ctx.errors and not ctx.refused
        report["form_problems"] = sum(1 for it in ctx.refused + ctx.errors if it.get("category") == "form"
                                      or it["level"] == "error")
        return report

    if not isinstance(qt, dict):
        ctx.issue("error", "R-SHAPE", MSG["not_object"])
        return finish(None)
    rows = qt.get("rows")
    if not isinstance(rows, list) or not rows:
        ctx.issue("error", "R-SHAPE", MSG["no_rows"], path="rows")
        return finish(None)
    for k in qt:
        if k not in TOP_KEYS:
            ctx.refuse("R-FORM", MSG["top_key"].format(k=k), "QTable", k, "form")
    if qt.get("schema") not in (None, SCHEMA_ID):
        ctx.warn("W-SCHEMA", MSG["schema"].format(v=qt.get("schema")), "QTable", "schema")
    nq = qt.get("n_quarters")
    H = _int(nq) if nq is not None else None
    if nq is None:
        qs = [_int(r.get("q")) for r in rows if isinstance(r, dict)]
        H = max([q for q in qs if q is not None] or [0])
    if H is None or not 1 <= H <= cq.QTABLE_MAX_QUARTERS:
        ctx.issue("error", "R-SHAPE", MSG["n_quarters"].format(hi=cq.QTABLE_MAX_QUARTERS, v=nq), path="n_quarters")
        return finish(None)
    report["n_quarters"] = H
    if narrative is None and source not in ("legacy",):
        ctx.warn("W-QUOTE", MSG["no_narrative"], "QTable", "")
    if source == "legacy":
        ctx.warn("W-QUOTE", MSG["legacy"], "QTable", "")
    clean_rows = _rows(ctx, rows, H)
    cqx_in = qt.get("c_q") if isinstance(qt.get("c_q"), dict) else {}
    if qt.get("c_q") is not None and not isinstance(qt.get("c_q"), dict):
        ctx.refuse("R-FORM", "c_q 必须是对象", "c_q", "c_q", "form")
    for k in cqx_in:
        if k not in CQ_KEYS:
            ctx.refuse("R-FORM", MSG["top_key"].format(k=f"c_q.{k}"), "c_q", f"c_q.{k}", "form")
    cqx = {}
    anchors = _anchors(ctx, cqx_in.get("anchors"), H)
    if anchors:
        cqx["anchors"] = anchors
    for key, fn in (("announced", _announced), ("constraints", _constraints), ("valuation", _valuation),
                    ("regime", _regime)):
        items = cqx_in.get(key)
        if items is not None and not isinstance(items, list):
            ctx.refuse("R-FORM", f"c_q.{key} 必须是列表", "c_q", f"c_q.{key}", "form")
            continue
        got = fn(ctx, items, H)
        if got:
            cqx[key] = got
    events = qt.get("events") if isinstance(qt.get("events"), list) else []
    clean = {"schema": SCHEMA_ID, "model_card": cq.CARD_ID, "n_quarters": H, "rows": clean_rows, "c_q": cqx,
             "events": _events(ctx, events, H, clean_rows), "notes": _notes(ctx, qt.get("notes") or [])}
    report["range_checked"] = bool(ctx.stats)
    report["missing"] = _requirements(ctx, clean)
    report["anchors"] = anchors
    if anchors:
        ctx.warn("W-ANCHOR", MSG["anchor"].format(names="、".join(f"{ANCHOR_ZH[a]} {v['value']:g}"
                                                                     f"（第 {v['from_q']} 季起）"
                                                                     for a, v in anchors.items())),
                 "c_q.anchors", "c_q.anchors")
    report["defaults"] = {"valuation": "inferred" if not cqx.get("valuation") else None}
    problems = schema_errors(clean, ctx.card)
    if problems:                                   # the clean form always validates; anything else is a bug here
        ctx.issue("error", "R-SHAPE", f"整理后的情景表仍不符合 QTable/1：{problems[0]}")
    return finish(clean)


def first_problem(report: dict) -> str | None:
    for key in ("errors", "refused"):
        if report.get(key):
            it = report[key][0]
            return f"{it['where']}：{it['text']}" if it.get("where") else it["text"]
    return None


def check(qt, narrative: str | None = None, **kw) -> dict:
    """validate(), or QTableError with the first error or refusal (the API's strict reading)."""
    report = validate(qt, narrative, **kw)
    if not report["ok"]:
        raise QTableError(first_problem(report) or "QTable 无法读取", report)
    return report


# ================================================================ QTable -> rollout inputs

def _stmt_entry(F, f: str, e: dict, q: int, unit: float | None):
    """One clean entry as a factors.Statement (the label is its quote)."""
    label = str(e.get("quote") or "")
    if f == "x_oil_cause":
        return F.Statement(f, (q,), cq.OIL_CAUSE[e["value"]], label=label)
    if "value" in e:
        return F.Statement(f, (q,), float(e["value"]), sd=e.get("sd"), label=label)
    if "sign" in e:
        if unit is None or not math.isfinite(unit) or unit <= 0:
            raise QTableError(f"{_where(q, f)}：没有训练期标准差，不能写定性幅度")
        if e["sign"] == "0":
            return F.Statement(f, (q,), 0.0, sd=SIGN_ZERO_SD * unit, label=label)
        size = e.get("size") or "小"
        return F.Statement(f, (q,), SIGNS[e["sign"]] * cq.SIZE_BINS[size] * unit, sd=QUAL_SD * unit, label=label)
    return F.Statement(f, (q,), lo=e.get("min"), hi=e.get("max"), label=label)


def regime_statements(F, regime: list) -> tuple[list, dict | None, list]:
    """Regime requests through the classifier's rules (engine/regimes.py). The contraction share of the requests on
    the same quarters (the contraction flag's share, or the sum of the mutually exclusive recession / depression /
    stagflation labels inside it; 1 - share for expansion) tilts the C' cycle module's IP blocks in those quarters
    (requests on different quarters: their quarter-weighted share, with a note); at a share >= 0.5 a label's
    inequality is imposed on the quarters' mean (high inflation: mean CPI y/y >= 4; stagflation: CPI y/y >= 4 as
    well; falling prices: mean CPI y/y <= 0; deep contraction: Σ d_ip <= -15; expansion: 0 <= mean CPI y/y <= 4);
    a share of 0 imposes the opposite inequality where one exists (no deep contraction: Σ d_ip >= -15)."""
    stmts, notes = [], []
    groups: dict[tuple, dict] = {}            # requests on the same quarters: the contraction share they imply
    for rg in regime:
        qs, s, lab = tuple(rg["quarters"]), float(rg["share"]), rg["label"]
        g = groups.setdefault(qs, {"flag": None, "labels": 0.0, "any": False, "expansion": None})
        if lab == "contraction":
            g["flag"], g["any"] = s, True
        elif lab in ("recession", "depression", "stagflation"):
            g["labels"] += s                  # mutually exclusive labels inside the contraction flag
            g["any"] = g["any"] or s > 0
        elif lab == "expansion":
            g["expansion"] = s
    tilts = []
    for qs, g in groups.items():
        if g["any"]:
            tilts.append((qs, min(1.0, max(g["flag"] or 0.0, g["labels"]))))
        elif g["expansion"] is not None:
            tilts.append((qs, 1.0 - g["expansion"]))
        elif g["flag"] == 0.0:
            tilts.append((qs, 0.0))
    for rg in regime:
        qs, s, lab = tuple(rg["quarters"]), float(rg["share"]), rg["label"]
        tag = f"regime:{lab}:{s:g}"
        major = s >= REGIME_MAJORITY
        if lab in ("inflation", "high_infl", "stagflation") and major:
            stmts.append(F.Statement("cpi_yoy", qs, lo=HIGH_INFL, agg="mean", label=tag))
        elif lab in ("inflation", "high_infl") and s == 0.0:
            stmts.append(F.Statement("cpi_yoy", qs, hi=HIGH_INFL, agg="mean", label=tag))
        if lab == "deflation" and major:
            stmts.append(F.Statement("cpi_yoy", qs, hi=CPI_NEG, agg="mean", label=tag))
        elif lab == "deflation" and s == 0.0:
            stmts.append(F.Statement("cpi_yoy", qs, lo=CPI_NEG, agg="mean", label=tag))
        if lab == "depression" and major:
            stmts.append(F.Statement("d_ip", qs, hi=-DEEP_IP, agg="sum", label=tag))
        elif lab == "depression" and s == 0.0:
            stmts.append(F.Statement("d_ip", qs, lo=-DEEP_IP, agg="sum", label=tag))
        if lab == "expansion" and major:
            stmts.append(F.Statement("cpi_yoy", qs, lo=CPI_NEG, hi=HIGH_INFL, agg="mean", label=tag))
        if 0.0 < s < REGIME_MAJORITY and lab not in CONTRACTION_LABELS + ("expansion",):
            notes.append({"code": "regime", "text": f"体制请求「{REGIME_ZH[lab]}」占 {s:.0%}：少于一半的季度，"
                                                    "不能写成每条路径都满足的不等式，未作输入"})
    odds = None
    if tilts:
        quarters = sorted({q for qs, _ in tilts for q in qs})
        n = sum(len(qs) for qs, _ in tilts)
        share = sum(len(qs) * s for qs, s in tilts) / n
        if len({round(s, 6) for _, s in tilts}) > 1:
            notes.append({"code": "regime", "text": f"几条收缩请求的占比不同：周期模块按季度加权的 {share:.0%} 倾斜"})
        odds = {"contraction": float(share), "quarters": quarters}
    return stmts, odds, notes


def constraints(qt: dict, *, stack=None, Q=None, train_end: str | None = None, x_fields=None,
                mode: str = "production", narrative: str | None = None, source: str = "translator",
                report: dict | None = None, strict: bool = True, numbers: bool = True) -> dict:
    """The QTable as rollout inputs: {statements (factors.Statement), valuation (ValuationStatement), cycle_odds,
    H, notes, anchors, report}. Validates first unless a report of this QTable is given: strict (the API) raises
    QTableError naming the first error or refusal; strict=False (the narrative test: refusals are part of its
    result) runs the clean QTable and only an unreadable one raises. Needs the stack, or the quarterly panel Q (with
    the train end and the x fields the factor layer carries; no fit). Every sign + size entry is a soft statement at
    sign x bin x training sd with sd = 0.25 training sd; fc stances are soft statements on d_gz and d_spread at
    FC_STANCE x their training sd (sd 0.5 x it) unless the row states the field; announced paths are soft (sd 0.5
    training sd unless given)."""
    from . import factors as F
    from . import valuation as val
    if stack is not None:
        Q, train_end, x_fields = stack.Q, stack.train_end, stack.x_fields
    if Q is None:
        raise QTableError("constraints() 需要拟合好的模型，或季度面板与训练期末")
    te = train_end or _train_end(Q)
    stats = stats_table(Q, te, shocks=True)
    if report is None:
        report = (check if strict else validate)(qt, narrative, mode=mode, source=source, stats=stats, numbers=numbers)
    if not report["readable"]:
        raise QTableError(first_problem(report) or "QTable 无法读取", report)
    clean = report["qtable"]
    H = int(clean["n_quarters"])
    have = set(x_fields) if x_fields is not None else set(qcard.x_fields(_card()))
    sd = {f: v["sd"] for f, v in stats.items()}
    stmts, notes = [], []
    for r in clean["rows"]:
        q = r["q"]
        for f, e in r["x"].items():
            if f not in have:
                raise QTableError(f"{_where(q, f)}：字段不存在或这个训练期没有估计：{f}", report)
            stmts.append(_stmt_entry(F, f, e, q, sd.get(f)))
        for u, e in (r.get("u") or {}).items():
            stmts.append(_stmt_entry(F, u, e, q, sd.get(u)))
        stance = r["fc"]["stance"]
        if stance != "未说明":
            for f in cq.FC_FIELDS:
                if f in r["x"] or f not in sd:
                    continue
                stmts.append(F.Statement(f, (q,), cq.FC_STANCE[stance] * sd[f], sd=cq.FC_STANCE_SD * sd[f],
                                         label=f"fc:{stance}"))
    cqx = clean["c_q"]
    for c in cqx.get("constraints", []):
        kind, f, qs = c["kind"], c["field"], tuple(c["quarters"])
        if kind == "yoy":
            var, agg = YOY_VARS[f], "each"
        elif f in LEVEL_VARS:
            var, agg = LEVEL_VARS[f], {"sum": "sum", "mean": "mean"}.get(kind, "each")
            if var == "ff" and agg != "each":
                var = "ff_shadow"       # the observed rate's floor is per quarter: a mean is stated on the shadow rate
        else:
            var, agg = f, {"sum": "sum", "mean": "mean"}.get(kind, "each")
        label = str(c.get("quote") or "")
        if kind == "range" or "value" not in c:
            stmts.append(F.Statement(var, qs, lo=c.get("min"), hi=c.get("max"), agg="each" if kind == "range" else agg,
                                     label=label))
        else:
            stmts.append(F.Statement(var, qs, float(c["value"]), sd=c.get("sd"), agg=agg, label=label))
    for a in cqx.get("announced", []):
        f = a["field"]
        unit = sd.get(f)
        s_a = a.get("sd") or (ANNOUNCED_SD * unit if unit else None)
        for q, v in zip(a["quarters"], a["values"]):
            if v is not None:
                stmts.append(F.Statement(f, (int(q),), float(v), sd=s_a, label=str(a.get("quote") or "")))
    rst, odds, rnotes = regime_statements(F, cqx.get("regime", []))
    stmts += rst
    notes += rnotes
    vst = val.parse_statements(cqx.get("valuation", []), H, mode=mode)
    for w in report["warnings"]:
        if w["code"] in ("W-FC", "W-R1", "W-R2", "W-R3", "W-R4", "W-OIL", "W-ANCHOR", "W-QE", "W-IDENT", "W-RANGE", "W-N3",
                         "W-VAL", "W-QUOTE", "W-U"):
            notes.append({"code": w["code"], "text": w["text"], **({"quarters": w["quarters"]}
                                                                  if w.get("quarters") else {})})
    return {"statements": stmts, "valuation": vst, "cycle_odds": odds, "H": H, "notes": notes,
            "anchors": report.get("anchors") or {}, "report": report}


# ================================================================ QTable -> the monthly engine's phase spec (§7.4)

# QTable field -> (monthly view variable, divisor): the monthly view's mean = the quarterly value / divisor
MONTHLY_VIEWS = {
    "d_ff": ("f_ff", 3.0), "d_cpi_goods": ("f_cpi_goods", 1200.0), "d_cpi_serv": ("f_cpi_services", 1200.0),
    "d_ip": ("f_ip", 300.0), "d_oil": ("f_oil", 300.0), "d_usd": ("f_usd", 300.0), "d_unrate": ("f_unrate", 3.0),
    "d_y10": ("f_y10", 3.0), "d_spread": ("f_spread", 3.0), "d_gz": ("f_gz", 3.0), "d_tp10": ("f_tp10", 3.0),
    "d_sent": ("f_sentiment", 3.0), "d_metals_oil": ("f_metals_oil", 300.0), "d_lshare": ("f_labor_share", 300.0),
    "d_wage": ("f_wage", 1200.0),
}
STATE_VIEWS = {"s_cpi_yoy": "s_cpi_yoy", "s_core_yoy": "s_core_yoy", "s_unrate": "s_unrate", "s_ff": "s_ff",
               "d_core": "s_core_yoy"}
VIEW_PRIORITY = ("f_ff", "f_unrate", "f_ip", "f_cpi_goods", "f_cpi_services", "f_oil", "f_spread", "f_gz", "f_y10",
                 "f_usd", "f_tp10", "s_cpi_yoy", "s_core_yoy", "s_unrate", "s_ff", "f_cape")
MONTHLY_NOTE = "月度引擎没有这个字段"
MAX_PHASES = 8
MAX_VIEWS = 12


def _entry_value(f: str, e: dict, sd: dict) -> tuple[float | None, float | None, float | None]:
    """(value, min, max) of a clean entry in the field's own units (sign + size at the training sd)."""
    if "value" in e:
        return float(e["value"]), None, None
    if "sign" in e:
        unit = sd.get(f)
        if unit is None:
            return None, None, None
        if e["sign"] == "0":
            return 0.0, None, None
        return SIGNS[e["sign"]] * cq.SIZE_BINS[e.get("size") or "小"] * unit, None, None
    return None, e.get("min"), e.get("max")


def to_phase_spec(qt: dict, *, Q=None, train_end: str | None = None, stats: dict | None = None,
                  name: str | None = None, narrative: str = "", horizon_months: int | None = None,
                  validated: bool = False, mode: str = "production") -> tuple[dict, dict]:
    """The monthly engine's phase spec for a (clean) QTable (§7.4): (spec, report).

    Consecutive quarters with the same stated set (fields, stance, the c_q statements covering them, regime and
    valuation requests) merge into a phase of at least 2 quarters (a lone quarter joins its successor, the last one its predecessor; beyond the engine's 8
    phases the shortest neighbours merge). Each stated field becomes a monthly view in MONTHLY_VIEWS units (f_ff =
    d_ff / 3 pp a month, f_cpi_goods = d_cpi_goods / 1200, f_ip / f_oil / f_usd = / 300, f_unrate, f_y10, f_spread,
    f_gz = / 3, ...), averaged over the phase's stated quarters; a range keeps its bounds; a fc stance becomes f_spread
    and f_gz views at the size bins; u_ff becomes an f_ff view of the same size (the monthly engine has no
    do-operator); level constraints become s_ views; regime requests the phase's regime; valuation statements f_cape
    views with the warning; r_star and a_tp anchors the phase overrides. A field without a monthly column is dropped
    with 「月度引擎没有这个字段」. The phases after the QTable's horizon are one 回归常态 phase without views."""
    from .. import config
    stats = stats if stats is not None else stats_table(Q, train_end, shocks=True)
    if not validated:                      # the structure only: quotes are the translation's business, not the engine's
        report_v = validate(qt, None, mode=mode, source="user", stats=stats)
        if not report_v["readable"]:
            raise QTableError(first_problem(report_v) or "QTable 无法读取", report_v)
        qt = report_v["qtable"]
    sd = {f: v["sd"] for f, v in stats.items()}
    horizon = int(horizon_months or config.HORIZON_MONTHS)
    H = int(qt["n_quarters"])
    rows = {r["q"]: r for r in qt["rows"]}
    cqx = qt.get("c_q") or {}
    dropped: dict[str, list] = {}
    notes, warnings = [], []

    def drop(field, q, text=MONTHLY_NOTE):
        dropped.setdefault(field, {"field": field, "quarters": [], "text": text})["quarters"].append(q)

    # per quarter: {view var: [(value, min, max)]}, the regime and valuation items, and the merge signature
    per_q: dict[int, dict] = {}
    sig: dict[int, tuple] = {}
    for q in range(1, H + 1):
        r = rows.get(q) or {"x": {}, "fc": {"stance": "未说明"}}
        views: dict[str, list] = {}
        keys = []
        for f, e in r["x"].items():
            keys.append(f)
            if f == "x_oil_cause":
                drop(f, q, "月度引擎没有这个字段（油价原因只能用 SVAR 冲击表达，而它们用全样本估计，有前视）")
                continue
            if f not in MONTHLY_VIEWS:
                drop(f, q)
                continue
            var, div = MONTHLY_VIEWS[f]
            v, lo, hi = _entry_value(f, e, sd)
            if v is None and lo is None and hi is None:
                drop(f, q, "没有训练期标准差，定性幅度换不成月度视图")
                continue
            views.setdefault(var, []).append(tuple(None if t is None else t / div for t in (v, lo, hi)))
        for u, e in (r.get("u") or {}).items():
            keys.append(u)
            if u == "u_ff" and "d_ff" not in r["x"]:
                v, lo, hi = _entry_value(u, e, sd)
                if v is not None:
                    views.setdefault("f_ff", []).append((v / 3.0, None, None))
                    notes.append(f"第 {q} 季：月度引擎没有干预算子，u_ff 按同幅度的政策利率变动写成 f_ff 视图")
                    continue
            drop(u, q, "月度引擎没有干预冲击")
        stance = r["fc"]["stance"]
        if stance != "未说明":
            keys.append(f"fc:{stance}")
            for f in cq.FC_FIELDS:
                if f in r["x"] or f not in sd:
                    continue
                var, div = MONTHLY_VIEWS[f]
                views.setdefault(var, []).append((cq.FC_STANCE[stance] * sd[f] / div, None, None))
        for ci, c in enumerate(cqx.get("constraints", [])):
            if q not in c["quarters"]:
                continue
            f, kind = c["field"], c["kind"]
            keys.append(f"c{ci}:{f}:{kind}")         # one statement: quarters it covers are one stated set
            if f in LEVEL_VARS or (f == "d_core" and kind == "yoy"):
                if kind == "sum":
                    drop(f"{f}:sum", q, "水平的加总没有月度视图")
                    continue
                var, div = STATE_VIEWS[f], 1.0           # a level (or CPI y/y): the phase's s_ view
            elif f in MONTHLY_VIEWS:
                var, div = MONTHLY_VIEWS[f]
                if kind == "sum":
                    div *= len(c["quarters"])            # a sum over the quarters: each quarter's share
            else:
                drop(f, q)
                continue
            vals = tuple(None if c.get(k) is None else c[k] / div for k in ("value", "min", "max"))
            views.setdefault(var, []).append(vals)
        for a in cqx.get("announced", []):
            if q in a["quarters"]:
                v = a["values"][a["quarters"].index(q)]
                if v is None:
                    continue
                keys.append(f"a:{a['field']}")
                if a["field"] not in MONTHLY_VIEWS:
                    drop(a["field"], q)
                    continue
                var, div = MONTHLY_VIEWS[a["field"]]
                views.setdefault(var, []).append((v / div, None, None))
        reg = tuple(sorted((rg["label"], rg["share"]) for rg in cqx.get("regime", []) if q in rg["quarters"]))
        vals_ = tuple(sorted((v["mode"], v.get("direction"), v.get("size")) for v in cqx.get("valuation", [])
                             if q in v["quarters"] and v["mode"] != "inferred"))
        per_q[q] = {"views": views, "regime": reg, "valuation": vals_}
        sig[q] = (tuple(sorted(keys)), reg, vals_)
    # runs of identical signatures -> phases of at least 2 quarters, at most MAX_PHASES (one kept for the tail)
    runs: list[list[int]] = []
    for q in range(1, H + 1):
        if runs and sig[runs[-1][-1]] == sig[q]:
            runs[-1].append(q)
        else:
            runs.append([q])
    k = 0
    while k < len(runs):
        if len(runs[k]) < 2 and len(runs) > 1:
            if k + 1 < len(runs):
                runs[k + 1] = runs[k] + runs[k + 1]
            else:
                runs[k - 1] = runs[k - 1] + runs[k]
            runs.pop(k)
            continue
        k += 1
    cap = MAX_PHASES - (1 if 3 * H < horizon else 0)
    merged_any = False
    while len(runs) > cap:
        j = min(range(len(runs) - 1), key=lambda i: len(runs[i]) + len(runs[i + 1]))
        runs[j:j + 2] = [runs[j] + runs[j + 1]]
        merged_any = True
    if merged_any:
        warnings.append(f"月度引擎最多 {MAX_PHASES} 个阶段：相邻的短阶段已合并，合并阶段里的视图取有陈述季度的平均")
    phases, report_phases = [], []
    anchors = cqx.get("anchors") or {}
    for run in runs:
        acc: dict[str, list] = {}
        seen: dict[str, set] = {}
        for q in run:
            for var, items in per_q[q]["views"].items():
                acc.setdefault(var, []).extend(items)
                seen.setdefault(var, set()).add(q)
        views = []
        for var, items in acc.items():
            v = [t[0] for t in items if t[0] is not None]
            lo = [t[1] for t in items if t[1] is not None]
            hi = [t[2] for t in items if t[2] is not None]
            view = {"var": var}
            if v and var.startswith("f_") and len(seen[var]) < len(run):
                # a change stated in some quarters of a merged phase keeps its cumulative size over the phase
                per_quarter = {}
                for q in run:
                    for t in per_q[q]["views"].get(var, []):
                        if t[0] is not None:
                            per_quarter.setdefault(q, []).append(t[0])
                total = sum(sum(x) / len(x) for x in per_quarter.values())
                view["mean"] = round(total / len(run), 10)
                notes.append(f"第 {_span(run)} 季：{var} 只在第 {_span(seen[var])} 季有陈述，按累计变动摊到整个阶段")
            elif v:
                view["mean"] = round(sum(v) / len(v), 10)
            if lo:
                view["min"] = round(sum(lo) / len(lo), 10)
            if hi:
                view["max"] = round(sum(hi) / len(hi), 10)
            view["why"] = f"季度情景表第 {_span(run)} 季的陈述（换算为月度单位）"
            views.append(view)
        # valuation statements: f_cape views (Σ y_mult over the statement's span, per month)
        for vs in cqx.get("valuation", []):
            qs = [q for q in vs["quarters"] if q in run]
            if not qs or vs["mode"] == "inferred":
                continue
            if vs["mode"] == "hold":
                target = 0.0
            else:
                target = None
                if Q is not None:
                    from . import valuation as val
                    try:
                        bins = val.size_bins(Q, train_end or _train_end(Q), len(vs["quarters"]))
                        target = (1 if vs["direction"] == "+" else -1) * bins[vs["size"]] / (3.0 * len(vs["quarters"]))
                    except ValueError:
                        target = None
                if target is None:
                    drop("valuation", qs[0], "没有季度面板：重估幅度换不成月度视图")
                    continue
            views = [v for v in views if v["var"] != "f_cape"]
            views.append({"var": "f_cape", "mean": round(target, 10),
                          "why": f"估值陈述（{vs['mode']}）：估值视图几乎直接决定股价"})
            warnings.append(f"第 {_span(qs)} 季：估值陈述写成了 f_cape 视图，它几乎直接决定股市估值和股价")
        order = {v: i for i, v in enumerate(VIEW_PRIORITY)}
        views.sort(key=lambda v: order.get(v["var"], len(order)))
        if len(views) > MAX_VIEWS:
            for v in views[MAX_VIEWS:]:
                drop(v["var"], run[0], f"月度引擎每个阶段最多 {MAX_VIEWS} 个视图：已删除")
            views = views[:MAX_VIEWS]
        regime = {}
        for q in run:
            for lab, share in per_q[q]["regime"]:
                regime.setdefault(lab, share)
        overrides = {}
        if "r_star" in anchors and run[0] >= anchors["r_star"]["from_q"]:
            overrides["r_star"] = anchors["r_star"]["value"]
        if "a_tp" in anchors and run[0] >= anchors["a_tp"]["from_q"]:
            overrides["term_premium"] = anchors["a_tp"]["value"]
        ph = {"name": f"第 {_span(run)} 季", "months": 3 * len(run), "views": views, "confidence": 1.0}
        if regime:
            ph["regime"] = regime
        if overrides:
            ph["overrides"] = overrides
        phases.append(ph)
        report_phases.append({"name": ph["name"], "quarters": run, "months": ph["months"],
                              "views": [v["var"] for v in views]})
    for a in anchors:
        if a not in ("r_star", "a_tp"):
            drop(f"anchor:{a}", anchors[a]["from_q"], "月度引擎的阶段覆盖项只有 r_star 与 term_premium：这个锚没有换算")
    if 3 * H < horizon:
        phases.append({"name": "回归常态", "months": horizon - 3 * H, "views": [], "confidence": 1.0})
    sha = digest(qt)[:8]
    spec = {"id": f"qtable_{sha}", "name": (name or f"季度情景表 {sha}")[:60],
            "summary": "由季度情景表（QTable/1）换算的月度规格", "narrative": str(narrative or "")[:6000],
            "tags": ["QTable"], "prior_prob": 0.05, "analogs": [], "causal_chain": [], "phases": phases,
            "source": "qtable"}
    rep = {"phases": report_phases, "dropped": list(dropped.values()), "notes": notes, "warnings": warnings,
           "horizon_months": horizon}
    return spec, rep


# ================================================================ the built-in scenarios as QTables (§7.4)

def builtin_ids() -> list[str]:
    return sorted(p.stem for p in QTABLES_DIR.glob("*.json"))


def builtin(sid: str) -> dict:
    """The reviewed QTable of a built-in scenario: {id, name, qtable, text, source, conversion, review, ...}."""
    if not re.fullmatch(r"[a-z0-9_]+", str(sid or "")):
        raise QTableError(f"内置情景编号格式不对：「{sid}」")
    path = QTABLES_DIR / f"{sid}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise QTableError(f"没有内置情景「{sid}」的季度情景表（scenarios/qtables/{sid}.json）") from err


def builtin_report(sid: str, stats: dict | None = None) -> dict:
    """validate() of a built-in QTable against its own source text (source "builtin")."""
    env = builtin(sid)
    return validate(env["qtable"], env["text"], mode="production", source="builtin", stats=stats)
