"""JSON out of a model's text answer, and a small JSON-schema validator for the subset the agents' schemas use.

No `jsonschema` package is installed in either supported Python, and the agents need Chinese messages anyway.

extract_json(text) -> (object or None, notes): the answer's JSON object with the slips models make fixed, each fix
listed in Chinese (AGENT_DESIGN §10.4 item 2): code fences and surrounding prose stripped; trailing commas, NaN and
Infinity removed; Python literals (True/False/None) fixed; full-width quotes, colons and commas fixed only where they
stand for JSON syntax, never inside a string (a blanket replacement corrupted 「失去席位」 in the PROMPTS prototype).

validate(instance, schema) -> [SchemaError]: type (number means finite, bool is not a number), enum, const,
required, properties, additionalProperties false, items, min/maxItems, min/maxLength, minimum/maximum, pattern,
local $ref into $defs.

transport_schema(schema): the strict schema without the keywords a prompt or --json-schema need not carry (bounds,
lengths, counts, patterns, descriptions), const turned into a one-value enum (PROMPTS §3.1).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", re.S)
OPEN_QUOTES = "“”„‟〝〞＂"
FULL_COLON, FULL_COMMA = "：", "，"
STRUCTURAL_AFTER = set(":：,，}]")
NOT_JSON_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class SchemaError:
    path: str
    kind: str          # type | enum | const | required | additional | min_items | max_items | max_length | ...
    message: str

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------- canonical JSON

def canonical_json(obj) -> str:
    """Sorted keys, no spaces, UTF-8 characters kept: the text hashed into cache keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def compact_json(obj) -> str:
    """Key order kept, no spaces: the schema text rendered into a prompt."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


# ---------------------------------------------------------------- extraction

def _balanced_object(s: str, start: int) -> int | None:
    """Index just past the '}' closing the object that opens at s[start], respecting strings; None if unclosed."""
    depth, i, n, in_str = 0, start, len(s), None
    while i < n:
        c = s[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c == '"':
            in_str = '"'
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _outside_strings(s: str, fn) -> str:
    """Apply fn to every stretch of s that is outside a JSON string (ASCII double quotes)."""
    out, buf, i, n = [], [], 0, len(s)
    in_str = False
    while i < n:
        c = s[i]
        if in_str:
            buf.append(c)
            if c == "\\" and i + 1 < n:
                buf.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                out.append("".join(buf))
                buf, in_str = [], False
        elif c == '"':
            out.append(fn("".join(buf)))
            buf, in_str = ['"'], True
        else:
            buf.append(c)
        i += 1
    out.append("".join(buf) if in_str else fn("".join(buf)))
    return "".join(out)


def _fix_commas_and_literals(s: str) -> str:
    def fix(chunk: str) -> str:
        chunk = re.sub(r",\s*([}\]])", r"\1", chunk)
        chunk = re.sub(r"\b(?:NaN|-?Infinity)\b", "null", chunk)
        chunk = re.sub(r"\bTrue\b", "true", chunk)
        chunk = re.sub(r"\bFalse\b", "false", chunk)
        return re.sub(r"\bNone\b", "null", chunk)
    # a trailing comma can sit right before a string's end only outside strings; run twice for ",\n]" after fixes
    return _outside_strings(_outside_strings(s, fix), fix)


def _fix_fullwidth(s: str) -> str:
    """Full-width quotes / colon / comma used as JSON syntax become ASCII; text inside ASCII strings is untouched.
    A string opened by a curly quote ends at the next curly quote that is followed by a structural character."""
    out, i, n = [], 0, len(s)
    state = "out"
    while i < n:
        c = s[i]
        if state == "ascii":
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                state = "out"
        elif state == "curly":
            if c in OPEN_QUOTES:
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                if j >= n or s[j] in STRUCTURAL_AFTER:
                    out.append('"')
                    state = "out"
                else:
                    out.append(c)
            elif c == '"':
                out.append('\\"')
            elif c == "\\" and i + 1 < n:
                out.append(s[i:i + 2])
                i += 2
                continue
            else:
                out.append(c)
        else:
            if c == '"':
                out.append(c)
                state = "ascii"
            elif c in OPEN_QUOTES:
                out.append('"')
                state = "curly"
            elif c == FULL_COLON:
                out.append(":")
            elif c == FULL_COMMA:
                out.append(",")
            else:
                out.append(c)
        i += 1
    return "".join(out)


def _loads(s: str):
    return json.loads(s, strict=False)          # strict=False: raw newlines inside strings are accepted


def extract_json(text) -> tuple[dict | None, list[str]]:
    """(the first JSON object in `text`, Chinese notes of every fix), or (None, notes) when nothing parses.
    The last note then names the position of the parse error."""
    notes: list[str] = []
    s = str(text or "").strip().lstrip("﻿")
    if not s:
        return None, ["没有输出"]
    fenced = FENCE_RE.search(s)
    if fenced and "{" in fenced.group(1):
        s2 = fenced.group(1).strip()
        notes.append("去掉了代码块标记")
        if s[:fenced.start()].strip() or s[fenced.end():].strip():
            notes.append("去掉了 JSON 之外的文字")
        s = s2
    elif s.startswith("```"):
        s = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        notes.append("去掉了代码块标记")
    i = s.find("{")
    if i < 0:
        return None, notes + ["输出里没有 JSON 对象"]
    end = _balanced_object(s, i)
    candidate = s[i:end] if end is not None else s[i:]
    if (i > 0 or (end is not None and s[end:].strip())) and "去掉了 JSON 之外的文字" not in notes:
        notes.append("去掉了 JSON 之外的文字")
    last_err = None
    try:
        obj = _loads(candidate)
        return (obj, notes) if isinstance(obj, dict) else (None, notes + ["模型输出的 JSON 不是对象"])
    except json.JSONDecodeError as err:
        last_err = err
    for note, fix in (("删除了尾随逗号 / NaN / Python 字面量", _fix_commas_and_literals),
                      ("把用作 JSON 符号的中文引号、冒号、逗号改为半角", _fix_fullwidth)):
        fixed = fix(candidate)
        if fixed == candidate:
            continue
        notes.append(note)
        candidate = fixed
        if end is None:                         # the object may close once curly quotes are fixed
            j = _balanced_object(candidate, 0)
            if j is not None:
                candidate = candidate[:j]
        try:
            obj = _loads(candidate)
            return (obj, notes) if isinstance(obj, dict) else (None, notes + ["模型输出的 JSON 不是对象"])
        except json.JSONDecodeError as err:
            last_err = err
    where = f"第 {last_err.lineno} 行第 {last_err.colno} 列" if last_err is not None else "未知位置"
    tail = "（输出可能被截断）" if end is None else ""
    return None, notes + [f"不是合法的 JSON（{where}）{tail}"]


# ---------------------------------------------------------------- validation

_PY = {"string": str, "object": dict, "array": list, "boolean": bool, "null": type(None)}


def _is_type(v, name: str) -> bool:
    if name == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if name == "integer":
        return (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer())
    return isinstance(v, _PY.get(name, object))


def _resolve(schema: dict, root: dict) -> dict:
    ref = schema.get("$ref")
    if not ref:
        return schema
    node = root
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def validate(instance, schema: dict, path: str = "$", root: dict | None = None) -> list[SchemaError]:
    """Errors of `instance` against `schema` (the subset above), each with a Chinese message."""
    root = root or schema
    schema = _resolve(schema, root)
    errs: list[SchemaError] = []
    t = schema.get("type")
    types = t if isinstance(t, list) else ([t] if t else [])
    if types and not any(_is_type(instance, n) for n in types):
        return [SchemaError(path, "type", f"{path}：类型应为 {'/'.join(types)}")]
    if "const" in schema and instance != schema["const"]:
        errs.append(SchemaError(path, "const", f"{path}：应为 {schema['const']!r}"))
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(SchemaError(path, "enum", f"{path}：{instance!r} 不在允许值中"))
    if isinstance(instance, str):
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errs.append(SchemaError(path, "max_length", f"{path}：长度 {len(instance)} 超过 {schema['maxLength']}"))
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errs.append(SchemaError(path, "min_length", f"{path}：长度 {len(instance)} 不足 {schema['minLength']}"))
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errs.append(SchemaError(path, "pattern", f"{path}：{instance!r} 不符合格式 {schema['pattern']}"))
    if _is_type(instance, "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append(SchemaError(path, "minimum", f"{path}：{instance} 小于 {schema['minimum']}"))
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append(SchemaError(path, "maximum", f"{path}：{instance} 大于 {schema['maximum']}"))
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in instance:
                errs.append(SchemaError(f"{path}.{k}", "required", f"{path}：缺少 {k}"))
        for k, v in instance.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}", root)
            elif schema.get("additionalProperties") is False:
                errs.append(SchemaError(f"{path}.{k}", "additional", f"{path}：不允许的键 {k}"))
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errs.append(SchemaError(path, "min_items", f"{path}：至少 {schema['minItems']} 项"))
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errs.append(SchemaError(path, "max_items", f"{path}：至多 {schema['maxItems']} 项"))
        if "items" in schema:
            for n, v in enumerate(instance):
                errs += validate(v, schema["items"], f"{path}[{n}]", root)
    return errs


TRANSPORT_DROP = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern",
                  "minItems", "maxItems", "format", "$id", "$schema", "description", "title"}


def transport_schema(schema):
    """The strict schema minus TRANSPORT_DROP; const -> a one-value enum (PROMPTS §3.1)."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k in TRANSPORT_DROP:
                continue
            if k == "const":
                out["enum"] = [v]
                continue
            out[k] = ({pk: transport_schema(pv) for pk, pv in v.items()} if k in ("properties", "$defs")
                      else transport_schema(v))
        return out
    if isinstance(schema, list):
        return [transport_schema(x) for x in schema]
    return schema


# ---------------------------------------------------------------- strings for prompts

def truncate(text, limit: int) -> str:
    """At most `limit` characters (an ellipsis when cut)."""
    s = str(text or "")
    return s if len(s) <= limit else s[:max(0, limit - 1)].rstrip() + "…"


def one_line(text) -> str:
    return NOT_JSON_SPACE.sub(" ", str(text or "")).strip()
