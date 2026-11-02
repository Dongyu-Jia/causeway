"""The blind prompt builder of the narrative test (SIM_DESIGN §8.4). It never imports the key.

Translation prompt: the QTable translator's test mode (llm/qtranslator.py: the blind schema view, template signs
only, synthetic few-shot examples, the rule against years and month counts) with the episode's state table taken
from the registry, which stores percentile bands computed when the registry was sealed. Nothing in this module knows
an episode's window, origin or train end; the user message is the narrative only (test mode adds no length line).

Recognition probe (§8.4): one separate call per narrative probed, 「这段叙事描述的是哪段时期？不知道就写
unknown」; its answer is scored against the sealed key by audit.py and never fed back into a translation.

Scans (§8.4 automated checks): scan_prompt (year tokens, quarter labels, month counts, a T0 offset, footprint keys,
and era names or event words in the prompt) and scan_narrative (years, digits in an N3 narrative of protocol v2,
names, places and the banned asset / valuation / firm-outcome words of writer protocol v2). Every prompt and every
user message is hashed; the run manifest keeps the hashes and the texts themselves (every byte sent).
"""
from __future__ import annotations

import hashlib
import json
import re

from ...llm import jsonfix
from ...llm import qtranslator as QX
from . import common as C
from . import episodes as EP

PROBE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "title": C.PROBE_SCHEMA_VERSION, "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "known", "period", "start_year", "end_year", "confidence", "clues"],
    "properties": {
        "schema_version": {"const": C.PROBE_SCHEMA_VERSION},
        "known": {"type": "boolean"},
        "period": {"type": "string", "maxLength": 80},
        "start_year": {"type": ["integer", "null"]},
        "end_year": {"type": ["integer", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "clues": {"type": "array", "maxItems": 5, "items": {"type": "string", "maxLength": 80}},
    },
}
PROBE_QUESTION = "这段叙事描述的是哪段时期？不知道就写 unknown。"
# era-identifying names (people, firms, unique events): kept out of prompts and of v2 narratives
ERA_NAMES = ("格林斯潘", "伯南克", "耶伦", "鲍威尔", "沃尔克", "雷曼", "贝尔斯登", "房利美", "房地美", "通用汽车", "克莱斯勒",
             "安然", "世通", "9·11", "九一一", "海湾战争", "伊拉克", "科威特", "新冠", "SARS", "次贷", "卡特", "里根", "克林顿",
             "布什", "奥巴马", "特朗普", "拜登", "华尔街", "硅谷")
# event words that point at one historical episode (§8.4 "scan for era names and event words"): kept out of the
# test-mode prompt (its decomposition example and template list are synthetic and generic)
EVENT_WORDS = ("疫情", "封锁", "大流行", "重启", "欧债", "禁运", "石油危机", "泡沫", "互联网", "科网", "储贷", "黑色星期一",
               "脱欧", "缩减恐慌", "JGB", "债务上限危机", "广场协议", "亚洲金融危机", "长期资本")
# places and institutions: writer protocol v2 keeps them out of narratives (the card's field definitions name a few
# data sources, e.g. the German 10-year yield and the Philadelphia Fed survey, so prompts are not scanned for them)
PLACES = ("美联储", "联储", "俄罗斯", "乌克兰", "中东", "OPEC", "欧佩克", "美国", "中国", "日本", "欧洲", "欧元区", "英国", "德国")
NAMES = ERA_NAMES + PLACES
BANNED = ("股价", "股市", "股票", "纳斯达克", "纳指", "标普", "道琼斯", "估值", "市盈率", "市净率", "盈利", "财报", "破产保护",
          "收购", "并购", "接管", "注资", "泡沫", "科技股", "互联网")
YEAR = re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}(?![0-9])")
DIGIT = re.compile(r"[0-9０-９]")


def sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", "surrogatepass")).hexdigest()


# ---------------------------------------------------------------- the translation prompt

def translation_system(code: str) -> str:
    """The test-mode QTable prompt with the episode's blind state table (percentile bands from the registry)."""
    return QX.build_system_prompt("test", state_rows=EP.state_rows(code))


def translation_user(narrative: str) -> str:
    """The narrative only (test mode: no length line, no episode code)."""
    return QX.user_message(narrative, None, "test")


# ---------------------------------------------------------------- the recognition probe

def probe_system() -> str:
    return "\n".join([
        "# 角色", "",
        "你读一段宏观经济叙事，判断它描述的是哪段真实的历史时期（大致的起止年份）。",
        "只根据叙事本身判断，不要编造；不知道就写 unknown：known 写 false，period 写 unknown，年份写 null，confidence 写 0。",
        "", QX.SCHEMA_HEADER, "",
        "回答只有一个 JSON 对象，符合下面的 JSON schema：不要放进 ```json 代码块，前后不要写任何其他文字。", "",
        jsonfix.compact_json(PROBE_SCHEMA)])


def probe_user(narrative: str) -> str:
    return f"{PROBE_QUESTION}\n\n<narrative>\n{narrative}\n</narrative>"


def read_probe(text: str | None) -> dict:
    """The probe's answer as {"known", "period", "start_year", "end_year", "confidence", "clues", "valid", "problems"}
    (an unreadable answer counts as unknown)."""
    obj, notes = jsonfix.extract_json(text or "")
    if not isinstance(obj, dict):
        return {"known": False, "period": "unknown", "start_year": None, "end_year": None, "confidence": 0.0,
                "clues": [], "valid": False, "problems": notes[-1:] or ["回答不是 JSON"]}
    problems = [str(e) for e in jsonfix.validate(obj, PROBE_SCHEMA)]
    out = {k: obj.get(k) for k in ("known", "period", "start_year", "end_year", "confidence", "clues")}
    for k in ("start_year", "end_year"):
        v = out.get(k)
        out[k] = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 1900 <= v <= 2100 else None
    out["known"] = bool(out.get("known")) and (out["start_year"] is not None or out["end_year"] is not None)
    out["confidence"] = float(out["confidence"]) if isinstance(out.get("confidence"), (int, float)) else 0.0
    out["clues"] = [str(c)[:80] for c in (out.get("clues") or [])][:5] if isinstance(out.get("clues"), list) else []
    out["period"] = str(out.get("period") or "unknown")[:80]
    out["valid"] = not problems
    out["problems"] = problems[:5]
    return out


# ---------------------------------------------------------------- scans

def scan_prompt(text: str) -> list[str]:
    """[] when the prompt is clean: the translator's own scan (years, quarter labels, month counts, T0 offsets,
    footprint keys) plus era-identifying names (people, firms, unique events) and event words (EVENT_WORDS)."""
    out = list(QX.prompt_scan(text))
    for w in ERA_NAMES:
        if w in text:
            out.append(f"名称「{w}」")
    for w in EVENT_WORDS:
        if w in text:
            out.append(f"事件词「{w}」")
    return out


def scan_narrative(text: str, version: str, n: str) -> dict:
    """Years, digits (in a protocol-v2 N3), names / places and banned words of one narrative: a report, not a
    refusal (the v1 narratives were written before protocol v2 and are kept unchanged)."""
    years = YEAR.findall(text)
    digits = len(DIGIT.findall(text)) if (n == "n3" and version == "v2") else None
    names = [w for w in NAMES if w in text]
    banned = {w: text.count(w) for w in BANNED if w in text}
    clean = not years and not names and not banned and not digits
    return {"years": years, "n3_digits": digits, "names": names, "banned": banned, "clean": clean}


def manifest_entry(system: str, user: str) -> dict:
    return {"system_sha256": sha256(system), "user_sha256": sha256(user), "system_chars": len(system),
            "user_chars": len(user), "input_tokens": QX.tokens(system) + QX.tokens(user)}


def prompt_digest(codes) -> str:
    """One hash over every translation prompt of these episodes (an estimate is stale when it changes)."""
    blob = json.dumps({c: sha256(translation_system(c)) for c in sorted(codes)}, sort_keys=True)
    return hashlib.sha256((blob + QX.PROMPT_VERSION).encode("utf-8")).hexdigest()
