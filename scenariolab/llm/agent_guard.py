"""Deterministic guardrails on parsed agent answers (AGENT_DESIGN §10.4, PROMPTS §5.2, CONSEQUENCES §10).

Every fix is one Chinese line in the call's report. Nothing here calls a model; repairs that need one are returned
as `problems` for the caller (scenariolab/agents/llm_io.py) to send through one P5 format repair.

    G1  powers         option ids from the node's menu (+ custom where allowed); writes / facts only on the custom
                       option and only the actor's own levers and fact types; reactions only catalogue pairs
    G2  no prices      sentences pairing a price / market term with a number that does not restate the state, with
                       a forecast word (将 / 会 / 预计 / 可能 / 跌至 / 跌到 / 升到 / 达到 …) or with a direction word
                       (上涨 / 下跌 / 回落 …, unless every number in it restates the state) are deleted; the text is
                       first NFKC-normalised (full-width digits) and Chinese numerals become digits (一万五千 →
                       15000, 百分之五 → 5%), so a forecast cannot hide in them (also the synthesizer's G-S2)
    G3  magnitudes     the option's catalogue range (unit slips ×100 / ÷100 fixed, clipped, step-rounded); custom
                       writes against the field's support: p01–p99 fine, up to the extreme 罕见, beyond it clipped
    G5  probabilities  percentages ÷ 100; |sum − 1| ≤ 0.10 renormalised, larger -> repair or fallback; the 0.03
                       rule (merge into the nearest option, mass conserved; two options: raise to 0.03);
                       duplicates merged; a status-quo option exists; custom ≤ its cap; reaction rows sum to 1
    G6  integrity      numbers in rationales and signals that do not restate the state are flagged; required
                       options (commitments due, hard gates) that were left out are flagged for the engine
    G8  length         every string capped at its schema length (cut and noted, never a failure)
    G10 hazard tilts   catalogue hazards only, inside [m/2, 3m]; above 2m only with a reason
    synth (structural part of G-S1, G-S2, G-S3, G-S10, G-S11, G-S12): shifts only on mechanism fields the
                       cluster's actors did not write, axis only a partition axis not written, partition geometry
                       on the z grid (contiguous, covering, one central), probabilities, numbers in memory, new
                       exogenous events, facts and exit hazards. Mechanism eligibility, lag discipline, direction,
                       signatures, precision weighting, joint rarity and the probability calculus are synth.py's.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field

from .agent_prompts import (CUSTOM_ID, MIN_BRANCH_PROB, MIN_PROB, Z_GRID, ActorSpec, HazardSpec, Lever,
                            ReactionSpec, SynthRequest)
from .jsonfix import truncate

SUM_TOL = 0.10
MAX_OPTIONS = 4
PRICE_TERMS = re.compile(
    r"(美股|股市|股价|股票|标普|纳指|纳斯达克|道指|指数点位|点位|金价|黄金价格|油价|原油价格|布伦特|WTI|房地产信托|REITs?"
    r"|房价|国债价格|收益率|利差|汇率|美元指数|估值|市盈率|市净率|股息率|CAPE|波动率|VIX)", re.I)
# forecast words; 会 only as the modal verb (not 国会、委员会、理事会、议会、社会、机会、峰会 …, not 会议、会谈)
FUTURE_WORDS = re.compile(r"将(?!近)|预计|预期|有望|料将|料|可能|或将|或许|跌至|涨至|升至|降至|跌到|涨到|升到|降到|回落到|"
                          r"回升到|达到|跌破|突破|年底前|明年|未来"
                          r"|(?<![国议委员事社机协大理学工峰年览省都宴晚聚展])会(?![议谈晤计员期所址])")
DIRECTION_WORDS = re.compile(r"上涨|下跌|大涨|大跌|反弹|回落|回升|走高|走低|走强|走弱|飙升|暴跌|崩盘|攀升|下挫|上扬|走软|走稳")
CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}
CN_BIG = {"万": 10_000, "亿": 100_000_000}
CN_NUMERAL = re.compile(r"(百分之)?([零〇一二两三四五六七八九十百千万亿]+)(?:(点)([零〇一二三四五六七八九]+))?")
EXO_EVENTS = re.compile(r"战争爆发|爆发战争|开战|疫情|大流行|地震|飓风|恐怖袭击|恐袭|政变|网络攻击")
FIELD_ID = re.compile(r"(?<![A-Za-z0-9_])[a-z][a-z0-9]*(?:_[a-z0-9.]+)+(?![A-Za-z0-9_])")
QUARTER_LABEL = re.compile(r"(?:19|20)\d{2}\s*Q[1-4]|(?<![A-Za-z])Q[1-4](?![0-9])")
DATE_LIKE = re.compile(r"(?:19|20)\d{2}\s*[-/年.]\s*\d{1,2}(?:\s*[-/月.]\s*\d{1,2}\s*日?)?")
NUMBER = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
COUNT_AFTER = re.compile(r"\s*(?:年期|年|个月|月|日|天|周|个季度|季度|季|次|名|位|家|个|项|条|轮|段|成)")
SENTENCE = re.compile(r"(?<=[。；;！!？?\n])")


# ---------------------------------------------------------------- numbers and price words

def _cn_int(t: str) -> int | None:
    """一万五千 -> 15000, 五千二百 -> 5200, 十二 -> 12, 三 -> 3; None when it is not a well-formed numeral."""
    total, section, digit = 0, 0, None
    for ch in t:
        if ch in CN_DIGIT:
            if digit is not None:
                return None                    # 五五: two digits in a row is not a numeral
            digit = CN_DIGIT[ch]
        elif ch in CN_UNIT:
            section += (1 if digit is None else digit) * CN_UNIT[ch]
            digit = None
        elif ch in CN_BIG:
            if digit is None and section == 0:
                return None                    # 万一 (in case): no leading number
            total += (section + (digit or 0)) * CN_BIG[ch]
            section, digit = 0, None
    return total + section + (digit or 0)


def normalize_numbers(text: str) -> str:
    """NFKC (full-width digits and signs to ASCII) and Chinese numerals to digits: 一万五千点 -> 15000点, 百分之五 ->
    5%, 三点五 -> 3.5. A lone numeral character (一 in 一致, 两 in 两党) is left alone."""
    s = unicodedata.normalize("NFKC", str(text or ""))

    def sub(m):
        pct, body, dot, frac = m.group(1), m.group(2), m.group(3), m.group(4)
        if len(body) < 2 and not pct and not dot and body not in CN_UNIT:
            return m.group(0)
        n = _cn_int(body)
        if n is None:
            return m.group(0)
        out = str(n) + ("." + "".join(str(CN_DIGIT[c]) for c in frac) if dot and frac else "")
        return out + "%" if pct else out
    return CN_NUMERAL.sub(sub, s)


def _value(token: str) -> float:
    return float(token.replace(",", "").replace("−", "-").replace("+", ""))


def _strip_ids(text: str) -> str:
    """Field ids, dates and quarter labels blanked out, keeping every other character at its position."""
    s = str(text or "")
    for rx in (FIELD_ID, DATE_LIKE, QUARTER_LABEL):
        s = rx.sub(lambda m: " " * len(m.group(0)), s)
    return s


def numbers_in(text) -> list[float]:
    """The values a text states: field ids, dates, quarter labels, ordinals (第 3 季) and counts (8 个季度, 10 年期)
    removed."""
    s = _strip_ids(text)
    out = []
    for m in NUMBER.finditer(s):
        before = s[max(0, m.start() - 1):m.start()]
        if before == "第" or COUNT_AFTER.match(s, m.end()):
            continue
        try:
            out.append(_value(m.group(0)))
        except ValueError:
            continue
    return out


def allowed_numbers(*texts) -> set[float]:
    """Every number shown to the model (state table, packets, digest, board): what a text may restate."""
    out: set[float] = set()
    for t in texts:
        for m in NUMBER.finditer(FIELD_ID.sub(" ", str(t or ""))):
            try:
                out.add(_value(m.group(0)))
            except ValueError:
                continue
    return out


def restates(x: float, allowed) -> bool:
    if float(x).is_integer() and 0 <= abs(x) <= 4:            # counters and ordinals
        return True
    return any(abs(x - a) <= max(0.051, 0.006 * abs(a)) for a in allowed)


def _price_claim(sentence: str) -> bool:
    for m in PRICE_TERMS.finditer(sentence):
        window = sentence[max(0, m.start() - 10):m.end() + 16]
        if numbers_in(window):
            return True
    return False


def scrub_prices(text, allowed, report: list, tag: str) -> str:
    """G2 / G-S2: delete every sentence that states or predicts a price: a price term with a forecast word, with a
    number that does not restate the state, or with a direction word unless every number in the sentence restates
    the state. Matching runs on the normalised sentence (normalize_numbers); the kept text is the original."""
    out = []
    for sent in SENTENCE.split(str(text or "")):
        norm = normalize_numbers(sent)
        if not sent or not PRICE_TERMS.search(norm):
            out.append(sent)
            continue
        nums = numbers_in(norm)
        restated = bool(nums) and all(restates(n, allowed) for n in nums)
        if FUTURE_WORDS.search(norm):
            why = "预测价格"
        elif _price_claim(norm) and not restated:
            why = "价格数值不是状态表里的"
        elif DIRECTION_WORDS.search(norm) and not restated:
            why = "价格方向"
        else:
            out.append(sent)
            continue
        report.append(f"{tag}：删除涉及价格的句子（{why}）「{truncate(sent.strip(), 40)}」")
    return "".join(out).strip()


def scrub_events(text, report: list, tag: str) -> str:
    """G-S2 (synthesizer): no new exogenous events in the text when nature drew none."""
    out = []
    for sent in SENTENCE.split(str(text or "")):
        if sent and EXO_EVENTS.search(sent):
            report.append(f"{tag}：删除新的外生事件「{truncate(sent.strip(), 40)}」")
            continue
        out.append(sent)
    return "".join(out).strip()


def drop_numbers(text, allowed) -> tuple[str, list[str]]:
    """(text without the stated values that restate nothing in `allowed`, the removed tokens); counts, ordinals,
    dates and field ids are kept (G-S11)."""
    s = str(text or "")
    masked = _strip_ids(s)                               # same length as s
    cut, removed = [], []
    for m in NUMBER.finditer(masked):
        before = s[max(0, m.start() - 1):m.start()]
        if before == "第" or COUNT_AFTER.match(s, m.end()):
            continue
        try:
            v = _value(m.group(0))
        except ValueError:
            continue
        if not restates(abs(v), allowed):
            cut.append((m.start(), m.end()))
            removed.append(m.group(0))
    for a, b in reversed(cut):
        s = s[:a] + s[b:]
    return re.sub(r"\s{2,}", " ", s).strip(), removed


def flag_numbers(text, allowed, report: list, tag: str) -> None:
    """G6: numbers that do not restate the state are flagged (not deleted)."""
    bad = [n for n in numbers_in(text) if not restates(n, allowed)]
    if bad:
        shown = "、".join(f"{x:g}" for x in bad[:3])
        report.append(f"{tag}：引用的数值 {shown} 不在状态表中（请核对）")


def _text(value, limit: int, allowed, report: list, tag: str, *, prices: bool = True) -> str:
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(value if value is not None else "")).strip()
    if prices:
        s = scrub_prices(s, allowed, report, tag)
    if len(s) > limit:
        report.append(f"{tag}：超过 {limit} 字，已截断")
        s = truncate(s, limit)
    return s


clean_text = _text      # public name: control characters stripped, price sentences deleted, capped at `limit`


def _opt_text(value, limit: int, allowed, report: list, tag: str) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    s = _text(value, limit, allowed, report, tag)
    return s or None


def _num(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ---------------------------------------------------------------- probabilities

def percent_fix(values: list, report: list, tag: str) -> list:
    """Percentages (a sum in (1.5, 100.5]) -> decimals."""
    s = sum(values)
    if 1.5 < s <= 100.5:
        report.append(f"{tag}：概率看起来是百分数，已除以 100")
        return [v / 100.0 for v in values]
    return values


def normalise(values: list, report: list, tag: str) -> tuple[list, str | None]:
    """(values summing to 1, None), or (values, the problem) when the sum is more than SUM_TOL away from 1."""
    values = percent_fix(list(values), report, tag)
    if any(v < 0 for v in values):
        return values, f"{tag}：概率不能为负"
    s = sum(values)
    if s <= 0:
        return values, f"{tag}：概率之和为 0"
    if abs(s - 1.0) > SUM_TOL:
        return values, f"{tag}：概率之和 {s:.2f} 偏离 1 太多"
    if abs(s - 1.0) > 0.005:
        report.append(f"{tag}：概率之和 {s:.2f}，已归一化")
    return [v / s for v in values], None


def _round_probs(values: list) -> list:
    """Four decimals that sum exactly to 1 (the largest absorbs the rounding)."""
    if not values:
        return values
    r = [round(v, 4) for v in values]
    k = max(range(len(r)), key=lambda i: r[i])
    r[k] = round(r[k] + 1.0 - sum(r), 4)
    return r


# ---------------------------------------------------------------- actors

@dataclass
class ActorAnswer:
    """One actor's guarded answer. options[].p_llm is the LLM's probability (None for a fallback); the engine
    pools it with the prior (prior_only means pool with c = 0)."""
    actor: str
    status: str = "ok"                       # ok | fallback
    options: list = field(default_factory=list)
    vs_base_rate: str | None = None
    private_note: dict | None = None
    reaction: dict | None = None             # follower rows {leader option: {follower option: p}}
    fallback: bool = False
    prior_only: bool = False
    declined: bool = False
    reason: str = ""                         # why the fallback (Chinese)
    missing_required: list = field(default_factory=list)
    report: list = field(default_factory=list)
    problems: list = field(default_factory=list)   # unresolved: a P5 repair could fix them

    def to_dict(self) -> dict:
        return asdict(self)


def _distance(a: dict, b: dict, actor: ActorSpec) -> float:
    """Standardised distance of two options' writes (the menu's point dicts; custom writes), else menu order."""
    pa, pb = a.get("_point") or {}, b.get("_point") or {}
    if pa or pb:
        sds = {lv.id: (lv.sd or 1.0) for lv in actor.levers}
        keys = set(pa) | set(pb)
        return math.sqrt(sum(((pa.get(k, 0.0) - pb.get(k, 0.0)) / (sds.get(k) or 1.0)) ** 2 for k in keys))
    return float(abs(a.get("_index", 0) - b.get("_index", 0)))


def _merge_into_nearest(opts: list, victim: dict, actor: ActorSpec, report: list, tag: str, why: str) -> None:
    others = [o for o in opts if o is not victim]
    target = min(others, key=lambda o: (_distance(victim, o, actor), o["_index"]))
    target["p_llm"] += victim["p_llm"]
    opts.remove(victim)
    report.append(f"{tag}/{victim['id']}：{why}，并入最接近的选项 {target['id']}")


def _magnitude(o: dict, spec, report: list, tag: str):
    m = spec.magnitude
    raw = o.get("magnitude")
    if m is None:
        if raw is not None:
            report.append(f"{tag}：选项没有幅度参数，已忽略 magnitude")
        return None
    v = _num(raw)
    if v is None:
        if raw is not None:
            report.append(f"{tag}：幅度不是数字，按默认值 {m.default:g} 处理")
        return m.default
    if not m.lo - 1e-9 <= v <= m.hi + 1e-9:
        for factor, unit in ((0.01, "基点"), (100.0, "百分点之百分之一")):
            if m.lo - 1e-9 <= v * factor <= m.hi + 1e-9:
                report.append(f"{tag}：幅度 {v:g} 看起来是{unit}，按 {v * factor:g} 处理")
                v = v * factor
                break
        else:
            clipped = min(max(v, m.lo), m.hi)
            report.append(f"{tag}：幅度 {v:g} 超出范围 [{m.lo:g}, {m.hi:g}]，截断为 {clipped:g}")
            v = clipped
    if m.step:
        r = round(round(v / m.step) * m.step, 6)
        r = min(max(r, m.lo), m.hi)
        if abs(r - v) > 1e-9:
            report.append(f"{tag}：幅度 {v:g} 取整为 {r:g}（{m.step:g} 的整数倍）")
        v = r
    return v


def _support(value: float, lever: Lever, beyond: str | None, report: list, tag: str) -> float:
    """G3 on a custom write: p01–p99 fine, up to the extreme flagged 罕见, beyond it clipped (the reason kept)."""
    lo, hi = lever.min, lever.max
    if lever.step:
        r = round(round(value / lever.step) * lever.step, 6)
        if abs(r - value) > 1e-9:
            report.append(f"{tag}：{lever.id}={value:g} 取整为 {r:g}")
        value = r
    if lo is not None and hi is not None and not lo <= value <= hi:
        clipped = min(max(value, lo), hi)
        report.append(f"{tag}：{lever.id}={value:g} 超出历史极值 [{lo:g}, {hi:g}]，截断为 {clipped:g}"
                      + (f"（理由：{beyond}）" if beyond else "（未给理由）"))
        return clipped
    if lever.p01 is not None and lever.p99 is not None and not lever.p01 <= value <= lever.p99:
        report.append(f"{tag}：{lever.id}={value:g} 罕见（历史 1%–99% 分位之外），保留并标记")
    if lo is None and lever.p01 is None:
        report.append(f"{tag}：{lever.id} 没有历史区间，幅度未检查（未验证）")
    return value


def _signal(raw, actor: ActorSpec, allowed, report: list, tag: str) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        report.append(f"{tag}：表态格式不对，已删除")
        return None
    d = raw.get("direction")
    if isinstance(d, bool) or _num(d) is None or int(d) != d or int(d) not in (-1, 0, 1):
        report.append(f"{tag}：表态的方向不是 −1/0/1，已删除表态")
        return None
    h = _num(raw.get("horizon_q"))
    horizon = int(min(max(round(h), 1), 4)) if h is not None else 1
    if h is None or horizon != h:
        report.append(f"{tag}：表态期限已调整为 {horizon} 季")
    inst = raw.get("instrument")
    if inst is not None and inst not in actor.instruments:
        report.append(f"{tag}：表态的工具 {inst} 不是本行为者的工具，已置空")
        inst = None
    cond = raw.get("condition")
    if cond not in actor.signal_conditions:
        report.append(f"{tag}：表态条件 {cond} 不在目录中，已置空")
        cond = None
    text = _text(raw.get("text"), 40, allowed, report, f"{tag} 表态")
    flag_numbers(text, allowed, report, f"{tag} 表态")
    return {"text": text, "instrument": inst, "direction": int(d), "horizon_q": horizon, "condition": cond,
            "commit": bool(raw.get("commit"))}


def fallback_answer(actor, reason: str, *, declined: bool = False) -> ActorAnswer:
    """回退：历史基率 — the status quo plus the top base-rate options (up to 4 in all), base rates renormalised
    (each ≥ 0.03), flagged prior-only; without base rates a uniform choice labelled 无历史先例."""
    actor = ActorSpec.coerce(actor)
    sq = actor.status_quo
    rated = sorted((o for o in actor.options if o is not sq), key=lambda o: -(o.base_rate or 0.0))
    chosen = ([sq] if sq else []) + rated[:MAX_OPTIONS - (1 if sq else 0)]
    if actor.has_base_rates:
        weights = [max(o.base_rate or 0.0, MIN_PROB) for o in chosen]
        label = "回退：历史基率"
    else:
        weights = [1.0] * len(chosen)
        label = "回退：均匀（无历史先例）"
    total = sum(weights) or 1.0
    probs = _round_probs([w / total for w in weights])
    options = [{"id": o.id, "label": o.label, "p_llm": None, "p_fallback": p, "magnitude":
                (o.magnitude.default if o.magnitude else None), "writes": {}, "facts": [], "rationale": label,
                "beyond_history": None, "signal": None, "custom": False, "source": "base_rate"}
               for o, p in zip(chosen, probs)]
    return ActorAnswer(actor=actor.id, status="fallback", options=options, fallback=True, prior_only=True,
                       declined=declined, reason=reason, report=[f"{actor.id}：{label}（{reason}）"])


def guard_rows(rows, leader_options, follower_options, report: list, tag: str) -> dict:
    """Reaction rows -> {leader option: {follower option: p}}: unknown options dropped, percentages fixed, each row
    renormalised within SUM_TOL (else dropped: the prior row is used), duplicates merged."""
    out: dict = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        lo = row.get("leader_option")
        if lo not in leader_options:
            report.append(f"{tag}：leader 选项 {lo} 不在菜单中，该行已删除")
            continue
        if lo in out:
            report.append(f"{tag}：leader 选项 {lo} 重复，保留第一行")
            continue
        probs: dict = {}
        for fp in row.get("follower_probs") or []:
            opt = fp.get("option") if isinstance(fp, dict) else None
            p = _num(fp.get("prob")) if isinstance(fp, dict) else None
            if opt not in follower_options or p is None:
                report.append(f"{tag}/{lo}：follower 选项 {opt} 无效，已删除")
                continue
            probs[opt] = probs.get(opt, 0.0) + p
        if not probs:
            report.append(f"{tag}/{lo}：没有有效的条件概率，用历史条件概率")
            continue
        keys = list(probs)
        vals, problem = normalise([probs[k] for k in keys], report, f"{tag}/{lo}")
        if problem:
            report.append(f"{problem}，该行已删除（用历史条件概率）")
            continue
        out[lo] = dict(zip(keys, _round_probs(vals)))
    missing = [lo for lo in leader_options if lo not in out]
    if missing and out:
        report.append(f"{tag}：缺少 leader 选项 {'、'.join(missing)} 的行（用历史条件概率）")
    return out


def guard_actor(block, actor, *, allowed=(), follower: ReactionSpec | None = None) -> ActorAnswer:
    """G1/G2/G3/G5/G6/G8 on one actor's block (P2's whole answer or one entry of P1's actors)."""
    actor = ActorSpec.coerce(actor)
    tag = actor.id
    ans = ActorAnswer(actor=actor.id)
    rep = ans.report
    if not isinstance(block, dict):
        ans.problems.append(f"{tag}：缺少行为者的回答")
        return ans
    if block.get("status") == "decline":
        reason = _text(block.get("decline_reason") or "未说明原因", 120, allowed, rep, f"{tag} 拒绝理由")
        out = fallback_answer(actor, f"拒绝评估：{reason}", declined=True)
        out.report = rep + out.report
        return out
    index = {o.id: k for k, o in enumerate(actor.options)}           # the catalogue's order (ordered bins)
    index[CUSTOM_ID] = len(index)
    opts: list[dict] = []
    seen: dict = {}
    for raw in block.get("options") or []:
        if not isinstance(raw, dict):
            continue
        oid = raw.get("id")
        otag = f"{tag}/{oid}"
        if oid == CUSTOM_ID and actor.custom:
            spec = None
        elif actor.option(str(oid)) is not None:
            spec = actor.option(str(oid))
        else:
            rep.append(f"{otag}：不是本节点菜单中的选项，已删除")
            continue
        p = raw.get("prob")
        p = _num(p) if p is not None else None
        if oid in seen:
            prev = seen[oid]
            if p is not None:
                prev["p_llm"] = (prev["p_llm"] or 0.0) + p
            rep.append(f"{otag}：重复的选项已合并")
            continue
        o = {"id": oid, "p_llm": p, "custom": spec is None, "source": "llm", "_index": index.get(oid, len(index))}
        if spec is not None:
            o["label"] = _text(raw.get("label") or spec.label, 12, allowed, rep, f"{otag} 名称", prices=False) \
                or spec.label
            o["magnitude"] = _magnitude(raw, spec, rep, otag)
            o["_point"] = dict(spec.point)
            if raw.get("writes") or raw.get("facts"):
                rep.append(f"{otag}：规范选项不写字段或事实（由程序按选项写入），已删除 writes / facts")
            o["writes"], o["facts"] = {}, []
        else:
            o["label"] = _text(raw.get("label") or "自定义", 12, allowed, rep, f"{otag} 名称", prices=False) or "自定义"
            o["magnitude"] = None
            beyond = raw.get("beyond_history")
            writes = {}
            levers = {lv.id: lv for lv in actor.levers}
            for fid, v in (raw.get("writes") or {}).items() if isinstance(raw.get("writes"), dict) else []:
                if fid not in levers:
                    rep.append(f"{otag}：越权字段 {fid} 已删除")
                    continue
                if _num(v) is None:
                    rep.append(f"{otag}：{fid} 的写入值不是数字，已删除")
                    continue
                writes[fid] = _support(_num(v), levers[fid], beyond, rep, otag)
            facts = []
            for fact in raw.get("facts") or []:
                if fact in actor.facts and fact not in facts:
                    facts.append(fact)
                else:
                    rep.append(f"{otag}：事实类型 {fact} 不允许，已删除")
            if not writes and not facts:
                rep.append(f"{otag}：自定义选项没有可执行的写入，已删除（概率并入现状选项）")
                o["_drop_to_sq"] = True
            o["writes"], o["facts"], o["_point"] = writes, facts, dict(writes)
        o["rationale"] = _text(raw.get("rationale"), 80, allowed, rep, f"{otag} 理由")
        flag_numbers(o["rationale"], allowed, rep, f"{otag} 理由")
        o["beyond_history"] = _opt_text(raw.get("beyond_history"), 80, allowed, rep, f"{otag} 超出历史")
        o["signal"] = _signal(raw.get("signal"), actor, allowed, rep, otag)
        seen[oid] = o
        opts.append(o)

    ans.vs_base_rate = _opt_text(block.get("vs_base_rate"), 120, allowed, rep, f"{tag} 基率说明")
    note = block.get("private_note")
    if isinstance(note, dict):
        clean = {k: _text(note.get(k), 40, allowed, rep, f"{tag} 私下打算") for k in ("plan", "reservation")
                 if str(note.get(k) or "").strip()}
        ans.private_note = clean or None

    if follower is not None:
        reaction = block.get("reaction") if isinstance(block.get("reaction"), dict) else None
        rows = guard_rows((reaction or {}).get("rows"), follower.leader_options, actor.option_ids, rep,
                          f"{tag} 季内反应")
        ans.reaction = rows or None
        if not rows:
            rep.append(f"{tag}：没有有效的季内反应行（用历史条件概率）")

    # custom options with nothing to write: their mass goes to the status quo
    sq_spec = actor.status_quo
    for o in [o for o in opts if o.pop("_drop_to_sq", False)]:
        sq = next((x for x in opts if sq_spec is not None and x["id"] == sq_spec.id), None)
        if sq is not None or sq_spec is None:
            opts.remove(o)
            if sq is not None and o["p_llm"] is not None and sq["p_llm"] is not None:
                sq["p_llm"] += o["p_llm"]
            continue
        o.update(id=sq_spec.id, label=sq_spec.label, custom=False, source="code", writes={}, facts=[],
                 magnitude=sq_spec.magnitude.default if sq_spec.magnitude else None, _index=index[sq_spec.id],
                 _point=dict(sq_spec.point))
        rep.append(f"{tag}：无效的自定义选项改为现状选项 {sq_spec.id}")

    if not opts:
        ans.problems.append(f"{tag}：没有有效的选项")
        return ans

    need_probs = follower is None or any(o["p_llm"] is not None for o in opts)
    if need_probs:
        missing = [o["id"] for o in opts if o["p_llm"] is None]
        if missing:
            if follower is not None:
                for o in opts:
                    o["p_llm"] = None
                need_probs = False
            else:
                ans.problems.append(f"{tag}：选项 {'、'.join(missing)} 缺少概率")
                return ans
    if need_probs:
        vals, problem = normalise([o["p_llm"] for o in opts], rep, tag)
        if problem:
            ans.problems.append(problem)
            return ans
        for o, v in zip(opts, vals):
            o["p_llm"] = v
        _probability_rules(opts, actor, rep, tag)
    else:
        _structure_rules(opts, actor, rep, tag)

    # R9: a departure over 0.15 from a base rate needs a reason
    if need_probs and not ans.vs_base_rate:
        far = [o["id"] for o in opts if actor.option(o["id"]) is not None
               and actor.option(o["id"]).base_rate is not None
               and abs(o["p_llm"] - actor.option(o["id"]).base_rate) > 0.15]
        if far:
            rep.append(f"{tag}：{'、'.join(far)} 与历史基率相差超过 0.15，但没有写 vs_base_rate")
    listed = {o["id"] for o in opts}
    ans.missing_required = [o.id for o in actor.options if o.required and o.id not in listed]
    if ans.missing_required:
        rep.append(f"{tag}：必须列出的选项 {'、'.join(ans.missing_required)} 没有列出（程序在池化时按历史基率补入）")
    ans.options = [{k: v for k, v in o.items() if not k.startswith("_")} for o in opts]
    return ans


def _code_option(spec, actor: ActorSpec, p, why: str) -> dict:
    return {"id": spec.id, "label": spec.label, "p_llm": p,
            "magnitude": spec.magnitude.default if spec.magnitude else None, "writes": {}, "facts": [],
            "rationale": why, "beyond_history": None, "signal": None, "custom": False, "source": "code",
            "_index": [o.id for o in actor.options].index(spec.id), "_point": dict(spec.point)}


def _ensure_status_quo(opts: list, actor: ActorSpec, rep: list, tag: str, with_probs: bool) -> None:
    sq = actor.status_quo
    if sq is None or any(o["id"] == sq.id for o in opts):
        return
    if with_probs:
        for o in opts:
            o["p_llm"] *= (1.0 - MIN_PROB)
    opts.append(_code_option(sq, actor, MIN_PROB if with_probs else None, "程序补入的现状选项"))
    rep.append(f"{tag}：缺少维持现状选项（规则 R3），已按 {MIN_PROB:g} 补入")


def _second_option(opts: list, actor: ActorSpec, rep: list, tag: str, with_probs: bool) -> None:
    if len(opts) >= 2:
        return
    have = {o["id"] for o in opts}
    rest = sorted((o for o in actor.options if o.id not in have), key=lambda o: -(o.base_rate or 0.0))
    if not rest:
        return
    if with_probs:
        for o in opts:
            o["p_llm"] *= (1.0 - MIN_PROB)
    opts.append(_code_option(rest[0], actor, MIN_PROB if with_probs else None, "程序补入（选项至少 2 个）"))
    rep.append(f"{tag}：只有 1 个选项，已按 {MIN_PROB:g} 补入 {rest[0].id}")


def _structure_rules(opts: list, actor: ActorSpec, rep: list, tag: str) -> None:
    """A follower's option list without marginal probabilities: status quo present, 2–4 options."""
    _ensure_status_quo(opts, actor, rep, tag, with_probs=False)
    _second_option(opts, actor, rep, tag, with_probs=False)
    while len(opts) > MAX_OPTIONS:
        victim = max(opts, key=lambda o: o["_index"])
        opts.remove(victim)
        rep.append(f"{tag}/{victim['id']}：选项超过 {MAX_OPTIONS} 个，已删除")


def _probability_rules(opts: list, actor: ActorSpec, rep: list, tag: str) -> None:
    """Status quo, the custom cap, 2–4 options, the 0.03 rule; probabilities rounded to sum exactly 1."""
    _ensure_status_quo(opts, actor, rep, tag, with_probs=True)
    for o in opts:
        if o["custom"] and o["p_llm"] > actor.custom_cap + 1e-9:
            excess = o["p_llm"] - actor.custom_cap
            o["p_llm"] = actor.custom_cap
            others = [x for x in opts if x is not o]
            base = sum(x["p_llm"] for x in others)
            for x in others:
                x["p_llm"] += excess * (x["p_llm"] / base if base > 0 else 1.0 / len(others))
            rep.append(f"{tag}/{CUSTOM_ID}：自定义选项的概率超过 {actor.custom_cap:g}，已截为上限")
    while len(opts) > MAX_OPTIONS:
        victim = min(opts, key=lambda o: (o["p_llm"], -o["_index"]))
        _merge_into_nearest(opts, victim, actor, rep, tag, f"选项超过 {MAX_OPTIONS} 个")
    _second_option(opts, actor, rep, tag, with_probs=True)
    small = [o for o in opts if o["p_llm"] < MIN_PROB - 1e-9]
    for o in sorted(small, key=lambda o: o["p_llm"]):
        if o not in opts:
            continue
        if len(opts) <= 2:
            deficit = MIN_PROB - o["p_llm"]
            o["p_llm"] = MIN_PROB
            other = next(x for x in opts if x is not o)
            other["p_llm"] -= deficit
            rep.append(f"{tag}/{o['id']}：概率低于 {MIN_PROB:g}，提高到下限")
            continue
        _merge_into_nearest(opts, o, actor, rep, tag, f"概率低于 {MIN_PROB:g}")
    probs = _round_probs([o["p_llm"] for o in opts])
    for o, p in zip(opts, probs):
        o["p_llm"] = p


def guard_reactions(raw, pairs, report: list) -> dict:
    """G1/G5 on P1's reactions: {leader: {follower: {leader option: {follower option: p}}}}, catalogue pairs only."""
    pairs = {(r.leader, r.follower): r for r in (ReactionSpec.coerce(p) for p in pairs)}
    out: dict = {}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        leader, follower = item.get("leader"), item.get("follower")
        spec = pairs.get((leader, follower))
        if spec is None:
            report.append(f"季内反应 {leader} → {follower} 不在本节点的行为者对中，已删除")
            continue
        if follower in out.get(leader, {}):
            report.append(f"季内反应 {leader} → {follower} 重复，保留第一个")
            continue
        rows = guard_rows(item.get("rows"), spec.leader_options, spec.follower_options, report,
                          f"季内反应 {leader} → {follower}")
        if rows:
            out.setdefault(leader, {})[follower] = rows
    return out


def guard_hazards(raw, hazards, report: list) -> dict:
    """G10: {hazard id: p} inside [m/2, 3m]; above 2m only with a reason (else capped at 2m)."""
    hazards = {h.id: h for h in (HazardSpec.coerce(x) for x in hazards)}
    out: dict = {}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        hid, p = item.get("hazard"), _num(item.get("prob"))
        h = hazards.get(hid)
        tag = f"自然风险 {hid}"
        if h is None:
            report.append(f"{tag}：不在目录中，已删除")
            continue
        if p is None or hid in out:
            report.append(f"{tag}：概率无效或重复，已删除")
            continue
        if 1.0 < p <= 100.0:
            p /= 100.0
            report.append(f"{tag}：概率看起来是百分数，已除以 100")
        why = str(item.get("why") or "").strip()
        clipped = min(max(p, h.lo), h.hi)
        if abs(clipped - p) > 1e-12:
            report.append(f"{tag}：概率 {p:.3f} 超出 [{h.lo:.3f}, {h.hi:.3f}]，已截断为 {clipped:.3f}")
            p = clipped
        if p > 2 * h.mass + 1e-12 and not why:
            report.append(f"{tag}：超过模型尾部质量的 2 倍且没有依据，已截为 {2 * h.mass:.3f}")
            p = 2 * h.mass
        out[hid] = round(p, 6)
    return out


# ---------------------------------------------------------------- synthesizer (structural part)

@dataclass
class ClusterAnswer:
    """One cluster's guarded branches; status "default" means the engine uses the default partition."""
    cluster_id: str
    status: str = "ok"                       # ok | default
    axis: str | None = None
    branches: list = field(default_factory=list)
    reason: str = ""
    declined: bool = False
    report: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _on_grid(z) -> bool:
    return z is None or any(abs(z - g) < 1e-9 for g in Z_GRID)


def _geometry(branches: list, report: list, tag: str) -> bool:
    """G-S3: the non-mechanism regions are contiguous on the z grid, cover the axis, and exactly one is central."""
    regions = [b for b in branches if b["kind"] in ("central", "partition")]
    if sum(1 for b in regions if b["kind"] == "central") != 1:
        report.append(f"{tag}：必须有且只有一个模型中枢分支")
        return False
    for b in regions:
        r = b.get("region")
        if not isinstance(r, dict):
            report.append(f"{tag}/{b['id']}：分区缺少区间")
            return False
        lo, hi = _num(r.get("lo_z")), _num(r.get("hi_z"))
        if (r.get("lo_z") is not None and lo is None) or (r.get("hi_z") is not None and hi is None):
            report.append(f"{tag}/{b['id']}：区间端点不是数字")
            return False
        if not (_on_grid(lo) and _on_grid(hi)):
            report.append(f"{tag}/{b['id']}：区间端点不在 −2.5…2.5 的半整数网格上")
            return False
        if lo is not None and hi is not None and lo >= hi:
            report.append(f"{tag}/{b['id']}：区间下端不小于上端")
            return False
        b["region"] = {"lo_z": lo, "hi_z": hi}
    ordered = sorted(regions, key=lambda b: -math.inf if b["region"]["lo_z"] is None else b["region"]["lo_z"])
    if len(ordered) == 1:
        only = ordered[0]["region"]
        if only["lo_z"] is not None or only["hi_z"] is not None:
            report.append(f"{tag}：只有一个区间时它必须覆盖整条轴")
            return False
        return True
    if ordered[0]["region"]["lo_z"] is not None or ordered[-1]["region"]["hi_z"] is not None:
        report.append(f"{tag}：区间没有覆盖整条轴")
        return False
    for a, b in zip(ordered, ordered[1:]):
        if a["region"]["hi_z"] is None or b["region"]["lo_z"] is None or abs(a["region"]["hi_z"]
                                                                               - b["region"]["lo_z"]) > 1e-9:
            report.append(f"{tag}：区间不首尾相接")
            return False
    return True


def guard_synth(payload: dict, req: SynthRequest, *, allowed=()) -> dict:
    """{cluster id: ClusterAnswer} for every cluster of the request (missing or broken ones are "default")."""
    req = req if isinstance(req, SynthRequest) else SynthRequest(**req)
    out: dict = {}
    axes, shiftable, anchors = set(req.axes), set(req.shift_fields), set(req.anchor_ids)
    by_id = {c.id: c for c in req.clusters}
    for raw in (payload or {}).get("clusters") or []:
        if not isinstance(raw, dict):
            continue
        cid = raw.get("cluster_id")
        if cid not in by_id or cid in out:
            continue
        spec = by_id[cid]
        ans = ClusterAnswer(cluster_id=cid)
        rep = ans.report
        out[cid] = ans
        if raw.get("status") == "decline":
            reason = _text(raw.get("decline_reason") or "未说明原因", 120, allowed, rep, f"{cid} 拒绝理由")
            ans.status, ans.declined, ans.reason = "default", True, f"拒绝分析：{reason}"
            continue
        written = set(spec.written_fields)
        axis = raw.get("axis")
        if axis is not None and (axis not in axes or axis in written):
            rep.append(f"{cid}：分区轴 {axis} 不允许（不是分区轴或本季已被行为者写入）")
            axis = None
        ans.axis = axis
        causes = set(spec.causes)
        branches, seen = [], set()
        for b in raw.get("branches") or []:
            if not isinstance(b, dict):
                continue
            bid = str(b.get("id") or f"b{len(branches) + 1}")
            if bid in seen:
                bid = f"{bid}_{len(branches) + 1}"
            seen.add(bid)
            tag = f"{cid}/{bid}"
            kind = b.get("kind")
            if kind not in ("central", "partition", "mechanism"):
                rep.append(f"{tag}：分支类型 {kind} 无效，已删除")
                continue
            p = _num(b.get("prob"))
            if p is None:
                ans.problems.append(f"{tag}：缺少概率")
                continue
            nb = {"id": bid, "label": _text(b.get("label") or bid, 12, allowed, rep, f"{tag} 名称", prices=False),
                  "kind": kind, "p_llm": p, "region": b.get("region"), "shifts": {}}
            if kind == "mechanism":
                nb["region"] = None
                for fid, s in (b.get("shifts") or {}).items() if isinstance(b.get("shifts"), dict) else []:
                    if fid not in shiftable or fid in written:
                        rep.append(f"{tag}：不能对 {fid} 写机制增量（不允许或本季已被行为者写入），已删除")
                        continue
                    if fid in set(getattr(spec, "channel_fields", None) or ()):
                        rep.append(f"{tag}：{fid} 本季已由模板通道写入，不能再写机制增量，已删除")
                        continue
                    mean = _num((s or {}).get("mean")) if isinstance(s, dict) else None
                    sd = _num((s or {}).get("sd")) if isinstance(s, dict) else None
                    if mean is None:
                        rep.append(f"{tag}：{fid} 的增量不是数字，已删除")
                        continue
                    nb["shifts"][fid] = {"mean": mean, "sd": abs(sd) if sd is not None else 0.0}
                if not nb["shifts"]:
                    rep.append(f"{tag}：机制分支没有可写的增量，并入模型中枢")
                    nb["_merge_central"] = True
            elif b.get("shifts"):
                rep.append(f"{tag}：非机制分支不能写增量，已删除 shifts")
            cause = b.get("cause") if isinstance(b.get("cause"), dict) else {}
            opts = [x for x in cause.get("options") or [] if isinstance(x, str)]
            kept = [x for x in opts if x in causes]
            if len(kept) < len(opts):
                rep.append(f"{tag}：原因中有不属于本动作组的选项，已删除")
            nb["cause"] = {"options": kept,
                           "state": [truncate(str(x), 40) for x in cause.get("state") or []][:4],
                           "facts": [truncate(str(x), 40) for x in cause.get("facts") or []][:3]}
            for key, limit in (("mechanism", 80), ("memory", 80)):
                text = _text(b.get(key), limit, allowed, rep, f"{tag} {key}")
                if not req.nature_events:
                    text = scrub_events(text, rep, f"{tag} {key}")
                nb[key] = text
            # G-S11: numbers in the memory sentence must be in the state or this branch's shifts
            shift_vals = {abs(s["mean"]) for s in nb["shifts"].values()}
            nb["memory"], bad = drop_numbers(nb["memory"], set(allowed) | shift_vals)
            if bad:
                rep.append(f"{tag}：记忆中的数字 {'、'.join(bad[:3])} 不在状态表或本支字段中，已删除")
            nb["anchors"] = [a for a in b.get("anchors") or [] if not anchors or a in anchors][:4]
            nb["unintended"] = bool(b.get("unintended"))
            nb["beyond_history"] = _opt_text(b.get("beyond_history"), 80, allowed, rep, f"{tag} 超出历史")
            nb["vs_prior"] = _opt_text(b.get("vs_prior"), 80, allowed, rep, f"{tag} 先验说明")
            facts = []
            for f in (b.get("facts") or [])[:2]:
                if not isinstance(f, dict) or not str(f.get("text") or "").strip():
                    continue
                hz = _num(f.get("exit_hazard"))
                hz2 = 0.5 if hz is None else min(max(hz, 0.1), 0.9)
                if hz is not None and abs(hz2 - hz) > 1e-12:
                    rep.append(f"{tag}：持续事实的结束概率 {hz:g} 已截到 [0.1, 0.9]")
                facts.append({"text": _text(f.get("text"), 40, allowed, rep, f"{tag} 持续事实"), "exit_hazard": hz2})
            if len(b.get("facts") or []) > 2:
                rep.append(f"{tag}：持续事实超过 2 条，只保留前 2 条")
            nb["facts"] = facts
            branches.append(nb)
        central = next((x for x in branches if x["kind"] == "central"), None)
        for nb in [x for x in branches if x.pop("_merge_central", False)]:
            branches.remove(nb)
            if central is not None:
                central["p_llm"] += nb["p_llm"]
        if ans.problems:
            continue
        if axis is None and any(x["kind"] == "partition" for x in branches):
            rep.append(f"{cid}：没有有效的分区轴，分区分支改为默认分区")
            ans.status, ans.reason = "default", "分区轴无效"
            continue
        if not branches or not _geometry(branches, rep, cid):
            ans.status, ans.reason = "default", "分区几何不合格"
            continue
        if len(branches) < 2 and axis is not None:
            rep.append(f"{cid}：分支少于 2 个，改为默认分区")
            ans.status, ans.reason = "default", "分支少于 2 个"
            continue
        vals, problem = normalise([x["p_llm"] for x in branches], rep, cid)
        if problem:
            ans.problems.append(problem)
            continue
        vals = _round_probs(vals)
        small = [x["id"] for x, v in zip(branches, vals) if v < MIN_BRANCH_PROB - 1e-9]
        if small:
            rep.append(f"{cid}：分支 {'、'.join(small)} 的概率低于 {MIN_BRANCH_PROB:g}（由池化与合并规则处理）")
        for x, v in zip(branches, vals):
            x["p_llm"] = v
        ans.branches = branches
    for cid in by_id:
        if cid not in out:
            out[cid] = ClusterAnswer(cluster_id=cid, status="default", reason="回答中缺少该动作组",
                                     report=[f"{cid}：回答中缺少该动作组，用默认分区"])
    return out


# ---------------------------------------------------------------- root translation (P4)

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def guard_root(payload: dict, narrative: str, settable, *, allowed=()) -> dict:
    """{"ok", "h0", "forced", "user_views", "unmapped", "report"}: forced fields only from the settable list,
    clipped to their history, each with a quote that really is in the user's narrative."""
    report: list = []
    if not isinstance(payload, dict) or payload.get("status") == "decline":
        reason = (payload or {}).get("decline_reason") if isinstance(payload, dict) else None
        return {"ok": False, "h0": None, "forced": [], "user_views": [], "unmapped": [],
                "report": [f"起点翻译被拒绝：{reason or '未说明原因'}"]}
    levers = {lv.id: lv for lv in (Lever.coerce(f) for f in settable)}
    h = payload.get("h0") if isinstance(payload.get("h0"), dict) else {}
    h0 = {"headline": _text(h.get("headline"), 30, allowed, report, "起点记忆 headline"),
          "summary": _text(h.get("summary"), 500, allowed, report, "起点记忆 summary"), "facts_used": []}
    for f in (h.get("facts_used") or [])[:20]:
        if isinstance(f, dict) and _num(f.get("value")) is not None:
            if not restates(_num(f["value"]), allowed):
                report.append(f"起点记忆引用的数值 {f.get('field')}={f['value']} 与状态表不符")
            h0["facts_used"].append({"field": str(f.get("field") or ""), "value": _num(f["value"])})
    whole = _norm(narrative)
    forced, used = [], set()
    for item in (payload.get("forced") or [])[:8]:
        if not isinstance(item, dict):
            continue
        fid, v, quote = item.get("field"), _num(item.get("value")), str(item.get("quote") or "")
        tag = f"固定字段 {fid}"
        if fid not in levers:
            report.append(f"{tag}：不是可设定的字段，已删除")
            continue
        if v is None:
            report.append(f"{tag}：数值无效，已删除")
            continue
        if not quote.strip() or _norm(quote) not in whole:
            report.append(f"{tag}：引文不是用户原文，已删除")
            continue
        if fid in used:
            report.append(f"{tag}：重复，保留第一个")
            continue
        forced.append({"field": fid, "value": _support(v, levers[fid], None, report, tag),
                       "quote": truncate(quote.strip(), 80)})
        used.add(fid)
    views = [truncate(str(x).strip(), 80) for x in (payload.get("user_views") or []) if str(x or "").strip()][:8]
    unmapped = [truncate(str(x).strip(), 80) for x in (payload.get("unmapped") or []) if str(x or "").strip()][:8]
    return {"ok": True, "h0": h0, "forced": forced, "user_views": views, "unmapped": unmapped, "report": report}
