"""A fake OpenRouter for WorldSim's tests: a transport that records every request and answers like
POST /api/v1/chat/completions, GET /api/v1/key and GET /api/v1/models would (shapes as documented and as seen in
the public model list). No test ever reaches the network."""
import copy
import json

TEST_KEY = "sk-or-v1-TESTKEY0123456789abcdef0123456789abcdef"


def chat_response(content, cost=0.0123, model="google/gemini-4-argon", usage=None):
    u = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200, "cost": cost,
         "completion_tokens_details": {"reasoning_tokens": 50}, "prompt_tokens_details": {"cached_tokens": 0}}
    if usage is not None:
        u = usage
    return {"id": "gen-123", "provider": "Google", "model": model, "object": "chat.completion", "created": 1793600000,
            "choices": [{"index": 0, "finish_reason": "stop", "native_finish_reason": "STOP",
                         "message": {"role": "assistant", "content": content}}], "usage": u}


def answer_for(body):
    """A valid answer for an actor or synthesizer request, read from its response_format schema when present."""
    rf = body.get("response_format") or {}
    schema = (rf.get("json_schema") or {}).get("schema")
    text = json.dumps(body.get("messages"), ensure_ascii=False)
    if (schema and "action_id" in schema["properties"]) or (not schema and '"action_id"' in text):
        ids = schema["properties"]["action_id"]["enum"] if schema else ["free"]
        return {"action_id": ids[0], "action_detail": "detail", "rationale": "because", "public_statement": "we act",
                "expected_consequences": ["calm"]}
    props = schema["properties"]["values"]["properties"] if schema else {}
    values = {}
    for fid, spec in props.items():
        t = spec["properties"]["value"]
        values[fid] = {"value": {"number": 1.5, "integer": 2, "boolean": True}.get(t["type"], (t.get("enum") or ["x"])[0]),
                       "reason": "moved"}
    return {"summary": "one line", "values": values, "narrative": "What happened.",
            "key_questions": [{"question": "Q?", "status": "open", "answer": "not yet"}],
            "attributes": [{"name": "Mood", "value": "calm"}]}


class FakeTransport:
    """transport(method, url, headers, body, timeout) -> (status, data, text). `script` is a list of answers used
    first, each a (status, data) pair, an Exception to raise, or "auto" (a valid answer for the request)."""

    def __init__(self, script=None, models=None, key_info=None):
        self.requests = []
        self.script = list(script or [])
        self.models = models
        self.key_info = key_info or {"data": {"label": "sk-or-v1-abc...xyz", "limit": 10.0, "limit_remaining": 7.5,
                                              "usage": 2.5, "is_free_tier": False}}

    def __call__(self, method, url, headers, body, timeout):
        self.requests.append({"method": method, "url": url, "headers": dict(headers), "body": copy.deepcopy(body),
                              "timeout": timeout})
        if url.endswith("/models"):
            data = {"data": self.models or []}
            return 200, data, json.dumps(data)
        if url.endswith("/key"):
            return 200, self.key_info, json.dumps(self.key_info)
        step = self.script.pop(0) if self.script else "auto"
        if isinstance(step, Exception):
            raise step
        if step == "auto":
            data = chat_response(json.dumps(answer_for(body), ensure_ascii=False))
            return 200, data, json.dumps(data)
        status, data = step
        return status, data, json.dumps(data) if data is not None else "not json"

    @property
    def chats(self):
        return [r for r in self.requests if r["url"].endswith("/chat/completions")]


MODELS = [
    {"id": "google/gemini-4-argon", "name": "Google: Gemini 4 Argon", "context_length": 1048576,
     "pricing": {"prompt": "0.000002", "completion": "0.00001"},
     "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort", "response_format", "structured_outputs",
                              "temperature"],
     "reasoning": {"mandatory": True, "supported_efforts": ["high", "medium", "low"], "default_effort": "medium"}},
    {"id": "acme/json-only", "name": "Acme JSON", "context_length": 32000, "pricing": {"prompt": "0.000001", "completion": "0.000002"},
     "supported_parameters": ["response_format", "temperature"], "reasoning": None},
    {"id": "acme/plain", "name": "Acme Plain", "context_length": 8000, "pricing": {"prompt": "0", "completion": "0"},
     "supported_parameters": ["temperature"]},
    {"id": "openrouter/auto", "name": "Auto", "pricing": {"prompt": "-1", "completion": "-1"}},
    "not a model",
]


def small_world(n_actors=3):
    return {"id": "w1", "name": "Test world", "language": "English", "description": "A small test world.",
            "schema": {"fields": [
                {"id": "rate", "label": "Rate", "group": "Policy", "type": "percent", "min": 0, "max": 20},
                {"id": "tension", "label": "Tension", "group": "Politics", "type": "number", "min": 0, "max": 10},
                {"id": "phase", "label": "Phase", "group": "Politics", "type": "enum", "options": ["calm", "crisis"]}]},
            "actors": [{"id": f"a{i}", "name": f"Actor {i}", "role": f"role {i}", "cares_about": ["peace"],
                        "action_space": [{"id": "hold", "label": "Hold"}, {"id": "push", "label": "Push"}],
                        "private_info": f"SECRET-{i}"} for i in range(n_actors)],
            "synthesizer": {"instructions": "Be consistent."},
            "initial_state": {"period": "2026Q4", "values": {"rate": 3.88, "tension": 5, "phase": "calm"},
                              "text": {"narrative": "Start.", "key_questions": [{"question": "Q?", "status": "open"}],
                                       "attributes": [{"name": "Mood", "value": "uneasy"}]}}}
