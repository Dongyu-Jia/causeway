"""LLM backends and JSON extraction.

A backend answers ``complete(CallRequest) -> Completion`` or raises LLMError:

- OpenRouterBackend: one chat completion; structured outputs (response_format json_schema, strict) when the model
  lists them, else json_object; a json_schema request the provider refuses (HTTP 400/404/422) is retried once as
  json_object. Usage accounting on; the cost comes from OpenRouter's usage.cost, else from the catalogue's prices.
- MockBackend: deterministic, schema-valid answers built from the request's context (no key, no network). A script
  ``{call key: [behaviour, ...]}`` (call key "actor:<id>" or "synth"; behaviour ok | error | bad_json | timeout |
  invalid) makes the next calls fail in a chosen way, for tests and demos.

extract_json(text) -> (object | None, error | None): the JSON object in a model's text (code fences, prose around it,
trailing commas, Python literals and smart quotes outside strings are tolerated).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import config
from . import model as M
from .openrouter import ModelCatalog, OpenRouterClient, OpenRouterError, Transport
from .settings import redact


@dataclass
class CallRequest:
    purpose: str                         # actor | synth
    model: str
    messages: list[dict]
    schema_name: str
    json_schema: dict
    temperature: float | None = None
    reasoning_effort: str = ""
    max_tokens: int | None = None
    timeout_s: float = 240
    context: dict = field(default_factory=dict)      # structured inputs (the mock answers from them)
    repair: bool = False


@dataclass
class Completion:
    text: str
    usage: dict
    cost: float | None
    model: str
    response_format: str
    meta: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    def __init__(self, message: str, kind: str = "error", retryable: bool = True, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status = status


# ------------------------------------------------------------------------------------------------ JSON extraction

FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", re.S)
SMART = {"“": '"', "”": '"', "„": '"', "＂": '"'}


def _balanced(s: str, start: int) -> int | None:
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _outside_strings(s: str, fn) -> str:
    """Apply fn to the parts of s outside JSON strings."""
    out, buf, in_str, esc = [], [], False, False
    for c in s:
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            out.append(fn("".join(buf)))
            buf = []
            out.append(c)
            in_str = True
        else:
            buf.append(c)
    out.append(fn("".join(buf)))
    return "".join(out)


def _fix(seg: str) -> str:
    seg = re.sub(r",(\s*[}\]])", r"\1", seg)
    seg = re.sub(r"\bTrue\b", "true", seg)
    seg = re.sub(r"\bFalse\b", "false", seg)
    seg = re.sub(r"\bNone\b", "null", seg)
    seg = re.sub(r"\bNaN\b|-?\bInfinity\b", "null", seg)
    return seg.replace("：", ":").replace("，", ",")


def extract_json(text: Any) -> tuple[Any, str | None]:
    if isinstance(text, (dict, list)):
        return text, None
    s = str(text or "").strip()
    if not s:
        return None, "the answer is empty"
    try:
        obj = json.loads(s)
        return (obj, None) if isinstance(obj, dict) else (None, "the answer is JSON but not an object")
    except ValueError:
        pass
    candidates = [m.group(1).strip() for m in FENCE_RE.finditer(s)] + [s]
    last_err = "no JSON object found"
    for cand in candidates:
        for c0 in (cand, "".join(SMART.get(ch, ch) for ch in cand)):
            start = c0.find("{")
            while start >= 0:
                end = _balanced(c0, start)
                if end is None:
                    last_err = "the JSON object is not closed (the answer may be truncated)"
                    break
                chunk = c0[start:end]
                for attempt in (chunk, _outside_strings(chunk, _fix)):
                    try:
                        obj = json.loads(attempt)
                        if isinstance(obj, dict):
                            return obj, None
                    except ValueError as err:
                        last_err = f"invalid JSON: {err}"
                start = c0.find("{", end)
    return None, last_err


# ------------------------------------------------------------------------------------------------ OpenRouter

class OpenRouterBackend:
    name = "openrouter"

    def __init__(self, api_key: str, catalog: ModelCatalog | None = None, transport: Transport | None = None):
        self._client = OpenRouterClient(api_key, transport=transport)
        self._key = api_key
        self.catalog = catalog or ModelCatalog(transport=transport)

    def _format_for(self, model_id: str) -> str:
        info = self.catalog.lookup(model_id)
        if info is None:
            return "json_object"
        if info.get("structured_outputs"):
            return "json_schema"
        return "json_object" if info.get("response_format") else "none"

    def _body(self, req: CallRequest, fmt: str) -> dict:
        body: dict[str, Any] = {"model": req.model, "messages": req.messages, "usage": {"include": True}}
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.max_tokens:
            body["max_tokens"] = req.max_tokens
        if req.reasoning_effort:
            # the reasoning text is not needed (exclude); its tokens are still billed and reported in usage
            body["reasoning"] = {"effort": req.reasoning_effort, "exclude": True}
        if fmt == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": req.schema_name, "strict": True,
                                                       "schema": req.json_schema}}
            body["provider"] = {"require_parameters": True}
        elif fmt == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(self, req: CallRequest) -> Completion:
        fmt = self._format_for(req.model)
        try:
            data = self._client.chat(self._body(req, fmt), req.timeout_s)
        except OpenRouterError as err:
            if fmt == "json_schema" and err.status in (400, 404, 422):
                fmt = "json_object"
                try:
                    data = self._client.chat(self._body(req, fmt), req.timeout_s)
                except OpenRouterError as err2:
                    raise LLMError(redact(str(err2), [self._key]), err2.kind, err2.retryable, err2.status) from None
            else:
                raise LLMError(redact(str(err), [self._key]), err.kind, err.retryable, err.status) from None
        return self._parse(req, data, fmt)

    def _parse(self, req: CallRequest, data: dict, fmt: str) -> Completion:
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LLMError("OpenRouter returned no choices", "empty")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if isinstance(choice.get("error"), dict):
            raise LLMError(redact(f"provider error: {choice['error'].get('message', '')}"[:400], [self._key]), "http")
        msg = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str) or not content.strip():
            fin = choice.get("finish_reason") or choice.get("native_finish_reason") or "unknown"
            raise LLMError(f"empty answer (finish reason: {fin})", "empty")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
        u = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
             "reasoning_tokens": details.get("reasoning_tokens"), "total_tokens": usage.get("total_tokens")}
        cost = usage.get("cost")
        cost_source = "openrouter"
        if not isinstance(cost, (int, float)):
            cost, cost_source = price(self.catalog.lookup(req.model), u), "computed"
        return Completion(text=content, usage=u, cost=float(cost) if cost is not None else None,
                          model=str(data.get("model") or req.model), response_format=fmt,
                          meta={"id": data.get("id"), "provider": data.get("provider"),
                                "finish_reason": choice.get("finish_reason"), "cost_source": cost_source})


def price(info: dict | None, usage: dict) -> float | None:
    if not info or info.get("prompt_price") is None or info.get("completion_price") is None:
        return None
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(pt, (int, float)) or not isinstance(ct, (int, float)):
        return None
    return pt * info["prompt_price"] + ct * info["completion_price"]


# ------------------------------------------------------------------------------------------------ mock

def _h(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _zh(language: str) -> bool:
    lang = (language or "").lower()
    return "中" in lang or "chinese" in lang or lang.startswith("zh")


def est_tokens(text: str) -> int:
    """A rough token count: CJK characters count about one each, other text about four characters per token."""
    s = str(text or "")
    cjk = sum(1 for c in s if "　" <= c <= "鿿" or "＀" <= c <= "￯")
    return int(cjk + (len(s) - cjk) / 4) + 1


class MockBackend:
    """Deterministic answers from the request context: actors pick an action by hash of (world, step, actor); the
    synthesizer moves about a third of the numeric fields a little (inside their plausible range), keeps every other
    value, and writes a narrative from the actions. Free of charge (cost 0)."""
    name = "mock"

    def __init__(self, delay_s: float | None = None, script: dict[str, list[str]] | None = None):
        self.delay_s = config.mock_delay_s() if delay_s is None else delay_s
        self._script = {k: list(v) for k, v in (script or {}).items()}
        self._lock = threading.Lock()
        self.calls: list[CallRequest] = []

    def _next(self, key: str) -> str:
        with self._lock:
            queue = self._script.get(key)
            return queue.pop(0) if queue else "ok"

    def complete(self, req: CallRequest) -> Completion:
        with self._lock:
            self.calls.append(req)
        if self.delay_s:
            time.sleep(self.delay_s)
        ctx = req.context or {}
        key = f"actor:{ctx.get('actor', {}).get('id')}" if req.purpose == "actor" else "synth"
        behaviour = self._next(key + (":repair" if req.repair else ""))
        if behaviour == "error":
            raise LLMError("mock: injected failure", "http")
        if behaviour == "timeout":
            raise LLMError("mock: injected timeout", "timeout")
        if behaviour == "bad_json":
            text = "Sure! Here is my answer: {\"action_id\": oops, not json"
        elif req.purpose == "actor":
            obj = self._actor(ctx)
            if behaviour == "invalid":
                obj["action_id"] = "no_such_action"
            text = json.dumps(obj, ensure_ascii=False)
        else:
            obj = self._synth(ctx)
            if behaviour == "invalid":
                obj = {"summary": "missing values"}
            text = json.dumps(obj, ensure_ascii=False)
        prompt = "".join(m.get("content", "") for m in req.messages)
        usage = {"prompt_tokens": est_tokens(prompt), "completion_tokens": est_tokens(text), "reasoning_tokens": 0}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return Completion(text=text, usage=usage, cost=0.0, model=f"mock/{req.model}", response_format="mock",
                          meta={"cost_source": "mock"})

    @staticmethod
    def _actor(ctx: dict) -> dict:
        actor = ctx.get("actor") or {}
        world = ctx.get("world") or {}
        state = ctx.get("state") or {}
        zh = _zh(world.get("language", ""))
        actions = actor.get("action_space") or []
        idx = _h(world.get("id"), state.get("step"), actor.get("id")) % (len(actions) + (1 if actor.get("allow_free_action") else 0) or 1)
        if idx < len(actions):
            act = actions[idx]
            aid, label = act["id"], act["label"]
            detail = (f"（模拟）{actor.get('name')}选择：{label}。" if zh else f"(mock) {actor.get('name')} chooses: {label}.")
        else:
            aid = M.FREE_ACTION
            label = "采取一项清单外的小步调整" if zh else "take a small step outside the list"
            detail = ("（模拟）一项不在清单上的小步调整：先观察其他各方的动作。" if zh
                      else "(mock) a small step outside the list: wait and watch the others.")
        cares = actor.get("cares_about") or []
        first = cares[0] if cares else ("自身目标" if zh else "its goals")
        if zh:
            return {"action_id": aid, "action_detail": detail,
                    "rationale": f"（模拟）基于{state.get('period', '')}的状态，优先考虑：{first}。",
                    "public_statement": f"（模拟）{actor.get('name')}：我们将{label}，并密切关注局势。",
                    "expected_consequences": [f"（模拟）{label}对局势的影响有限且可控。"]}
        return {"action_id": aid, "action_detail": detail,
                "rationale": f"(mock) Given the state at {state.get('period', '')}, the priority is: {first}.",
                "public_statement": f"(mock) {actor.get('name')}: we will {label.lower()} and watch closely.",
                "expected_consequences": [f"(mock) {label} has a limited, contained effect."]}

    @staticmethod
    def _synth(ctx: dict) -> dict:
        world = ctx.get("world") or {}
        state = ctx.get("state") or {}
        schema = ctx.get("schema") or {"fields": []}
        results = ctx.get("actor_results") or []
        zh = _zh(world.get("language", ""))
        step = int(state.get("step") or 0) + 1
        cur = state.get("values") or {}
        values: dict[str, dict] = {}
        moved = 0
        for f in schema.get("fields", []):
            fid, prev = f["id"], cur.get(f["id"])
            val, reason = prev, ""
            h = _h(world.get("id"), step, fid)
            if f["type"] in M.NUMERIC_TYPES and isinstance(prev, (int, float)) and not isinstance(prev, bool) and h % 3 == 0:
                lo, hi = f.get("min"), f.get("max")
                span = (hi - lo) if (lo is not None and hi is not None and hi > lo) else 0.0
                step_size = 0.01 * abs(prev) + 0.001 * span + (0.01 if not prev and not span else 0.0)
                delta = ((h // 3) % 7 - 3) / 3 * step_size              # small, deterministic, either sign
                new = prev + delta
                if lo is not None:
                    new = max(lo, new)
                if hi is not None:
                    new = min(hi, new)
                new = int(round(new)) if f["type"] == "integer" else round(new, 2)
                if new != prev:
                    val = new
                    moved += 1
                    reason = ("（模拟）各方行动的综合结果带来小幅变动。" if zh
                              else "(mock) a small move from the actors' combined actions.")
            if val is None:
                val = {"number": 0.0, "percent": 0.0, "integer": 0, "boolean": False,
                       "enum": (f.get("options") or [""])[0], "text": ""}.get(f["type"], "")
                reason = "（模拟）此前没有数值，取默认值。" if zh else "(mock) no previous value: a default."
            values[fid] = {"value": val, "reason": reason}
        acted = [r for r in results if r.get("status") == "ok"]
        silent = [r for r in results if r.get("status") != "ok"]
        period = ctx.get("next_period", "")
        if zh:
            parts = [f"【模拟】{period}：" + "；".join(f"{r['name']}{r['output']['action_label']}" for r in acted) + "。"]
            if silent:
                parts.append("未回应：" + "、".join(r["name"] for r in silent) + "。")
            parts.append(f"共有 {moved} 个数值小幅变动，其余保持不变。")
            summary = f"（模拟）{len(acted)} 方行动，{moved} 个字段变动"
        else:
            parts = [f"[mock] {period}: " + "; ".join(f"{r['name']} — {r['output']['action_label']}" for r in acted) + "."]
            if silent:
                parts.append("No response: " + ", ".join(r["name"] for r in silent) + ".")
            parts.append(f"{moved} values moved slightly; the rest held.")
            summary = f"(mock) {len(acted)} actors acted, {moved} fields moved"
        text = state.get("text") or {}
        questions = [dict(q) for q in text.get("key_questions") or []]
        open_q = [q for q in questions if q.get("status") == "open"]
        if open_q:
            q = open_q[(step - 1) % len(open_q)]
            q["answer"] = (f"（模拟）截至 {period}：仍待观察。" if zh else f"(mock) as of {period}: still open.")
        attributes = [dict(a) for a in text.get("attributes") or []]
        return {"summary": summary, "values": values, "narrative": " ".join(parts), "key_questions": questions,
                "attributes": attributes}


_SCRIPTED: dict[str, MockBackend] = {}


def make_backend(settings: dict, catalog: ModelCatalog | None = None, transport: Transport | None = None):
    """The backend for these settings (WORLDSIM_LLM overrides the choice in Settings). WORLDSIM_MOCK_SCRIPT (JSON,
    e.g. {"synth": ["error"]}) gives the mock one scripted queue for the whole process, to try the failure paths."""
    name = config.forced_backend() or settings.get("backend") or "openrouter"
    if name == "mock":
        raw = config.mock_script()
        if raw:
            if raw not in _SCRIPTED:
                try:
                    script = json.loads(raw)
                except ValueError:
                    script = {}
                _SCRIPTED[raw] = MockBackend(script=script if isinstance(script, dict) else {})
            return _SCRIPTED[raw]
        return MockBackend()
    return OpenRouterBackend(settings.get("api_key") or "", catalog=catalog, transport=transport)
