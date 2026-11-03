"""Prompts and answer schemas for the two kinds of calls.

Actor call: the world description, the actor's role, what it cares about, its action space and private information,
the current state (only its visible fields) and a short recent history -> one JSON object
{action_id, action_detail, rationale, public_statement, expected_consequences [str]}.

Synthesizer call: the world description, the synthesizer's instructions, the schema definition, the current state,
every actor's output and the recent history -> the next state as one JSON object
{summary, values {field id: {value, reason}}, narrative, key_questions [{question, status, answer}],
 attributes [{name, value}]}.

The chrome is English; the world's own text (roles, instructions, state) is in the world's language, and every
generated text is asked for in that language.
"""
from __future__ import annotations

import json
from typing import Any

from . import model as M

ACTOR_SCHEMA_NAME = "actor_decision"
SYNTH_SCHEMA_NAME = "next_state"
NARRATIVE_CHARS = 700


def fmt_value(field: dict, value: Any) -> str:
    if value is None:
        return "(unknown)"
    if field.get("type") == "boolean":
        return "true" if value else "false"
    if field.get("type") in M.NUMERIC_TYPES and isinstance(value, (int, float)):
        s = M._num_text(value)
        unit = field.get("unit") or ""
        if field.get("type") == "percent" and "%" not in unit:
            unit = ("% " + unit).strip()
        return f"{s} {unit}".strip()
    return str(value)


def _state_lines(schema: dict, state: dict, visible: list[str] | None = None, with_reasons: bool = False) -> list[str]:
    fields = schema.get("fields", [])
    if visible is not None:
        keep = set(visible)
        fields = [f for f in fields if f["id"] in keep]
    lines: list[str] = []
    group = None
    vals = state.get("values") or {}
    reasons = state.get("reasons") or {}
    for f in fields:
        if f["group"] != group:
            group = f["group"]
            lines.append(f"[{group}]")
        tag = f" ({f['tag']})" if f.get("tag") else ""
        line = f"- {f['label']} [{f['id']}]{tag}: {fmt_value(f, vals.get(f['id']))}"
        if with_reasons and reasons.get(f["id"]):
            line += f"  — last change: {reasons[f['id']][:160]}"
        lines.append(line)
    return lines


def _text_lines(text: dict) -> list[str]:
    lines = ["Narrative:", (text.get("narrative") or "(none)").strip(), "", "Key questions:"]
    qs = text.get("key_questions") or []
    if not qs:
        lines.append("- (none)")
    for q in qs:
        ans = f" — current answer: {q['answer']}" if q.get("answer") else ""
        lines.append(f"- [{q.get('status', 'open')}] {q['question']}{ans}")
    lines += ["", "Attributes:"]
    attrs = text.get("attributes") or []
    if not attrs:
        lines.append("- (none)")
    for a in attrs:
        lines.append(f"- {a['name']}: {a['value']}")
    return lines


def _history_lines(history: list[dict], me: str | None = None, include_current_narrative: bool = False) -> list[str]:
    if not history:
        return ["(no earlier steps)"]
    lines: list[str] = []
    for i, h in enumerate(history):
        last = i == len(history) - 1
        lines.append(f"Step {h['step']} ({h['period']}): {h.get('summary') or ''}".rstrip(": "))
        if h.get("actions"):
            lines.append("  Actions that led to it:")
            for a in h["actions"]:
                who = f"{a['name']}{' (you)' if me and a.get('actor_id') == me else ''}"
                if a.get("status") != "ok":
                    lines.append(f"  - {who}: no response")
                    continue
                stmt = f' — said: "{a["public_statement"][:200]}"' if a.get("public_statement") else ""
                detail = f" — {a['action_detail'][:200]}" if a.get("action_detail") else ""
                lines.append(f"  - {who}: {a['action_label']}{detail}{stmt}")
        if h.get("narrative") and (include_current_narrative or not last):
            lines.append(f"  Narrative then: {h['narrative'][:NARRATIVE_CHARS]}")
    return lines


def _world_lines(world: dict) -> list[str]:
    return [f"# World: {world['name']}", world.get("description") or "(no description)"]


# ------------------------------------------------------------------------------------------------ actor

def actor_schema(actor: dict) -> dict:
    ids = [a["id"] for a in actor.get("action_space") or []]
    if actor.get("allow_free_action") or not ids:
        ids = ids + [M.FREE_ACTION]
    return {"type": "object", "additionalProperties": False,
            "required": ["action_id", "action_detail", "rationale", "public_statement", "expected_consequences"],
            "properties": {
                "action_id": {"type": "string", "enum": ids},
                "action_detail": {"type": "string"},
                "rationale": {"type": "string"},
                "public_statement": {"type": "string"},
                "expected_consequences": {"type": "array", "items": {"type": "string"}}}}


def actor_messages(world: dict, actor: dict, state: dict, history: list[dict], next_period: str) -> list[dict]:
    lang = world.get("language") or "English"
    free = actor.get("allow_free_action")
    system = "\n".join([
        f'You play one actor in a turn-based world simulation ("{world["name"]}"). Stay in role: decide what '
        f"{actor['name']} does during the coming period, from {state.get('period')} to {next_period}, given what it "
        "cares about, its private information and the situation. Be realistic about its constraints.",
        "Choose exactly one action from your action space by its id"
        + (', or "free" with your own action described in action_detail' if free else "") + ".",
        "Answer with one JSON object only (no prose, no code fences) with these keys:",
        '  "action_id": the chosen action id' + (' or "free"' if free else ""),
        '  "action_detail": what exactly you do (size, timing, conditions)',
        '  "rationale": why, in a few sentences (private: only the synthesizer reads it)',
        '  "public_statement": what you say in public (may be empty)',
        '  "expected_consequences": a list of 1-4 short expected consequences',
        f"Write every text value in {lang}. Do not invent data you were not given; reason from it.",
    ])
    actions = actor.get("action_space") or []
    lines = _world_lines(world) + ["", f"# You are: {actor['name']}", actor.get("role") or "(no role given)", "",
                                   "# What you care about"]
    lines += [f"- {c}" for c in actor.get("cares_about") or []] or ["- (not specified)"]
    lines += ["", "# Your action space"]
    lines += [f"- {a['id']}: {a['label']}" + (f" — {a['description']}" if a.get("description") else "") for a in actions]
    lines.append("- free: an action of your own (describe it in action_detail)" if free else
                 "(free actions are not allowed: pick one of the ids above)")
    if actor.get("private_info"):
        lines += ["", "# Private information (only you see this)", actor["private_info"]]
    lines += ["", f"# Current state: step {state.get('step')}, period {state.get('period')}"]
    lines += _state_lines(world["schema"], state, actor.get("visible_fields"))
    lines += [""] + _text_lines(state.get("text") or {})
    lines += ["", "# Recent history (oldest first)"] + _history_lines(history, me=actor["id"])
    lines += ["", f"Decide {actor['name']}'s action for {state.get('period')} → {next_period}. JSON only."]
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


# ------------------------------------------------------------------------------------------------ synthesizer

def _value_schema(field: dict) -> dict:
    t = field["type"]
    if t in ("number", "percent"):
        return {"type": "number"}
    if t == "integer":
        return {"type": "integer"}
    if t == "boolean":
        return {"type": "boolean"}
    if t == "enum":
        return {"type": "string", "enum": list(field.get("options") or [])}
    return {"type": "string"}


def synth_schema(schema: dict) -> dict:
    fields = schema.get("fields", [])
    props = {f["id"]: {"type": "object", "additionalProperties": False, "required": ["value", "reason"],
                       "properties": {"value": _value_schema(f), "reason": {"type": "string"}}} for f in fields}
    return {"type": "object", "additionalProperties": False,
            "required": ["summary", "values", "narrative", "key_questions", "attributes"],
            "properties": {
                "summary": {"type": "string"},
                "values": {"type": "object", "additionalProperties": False, "required": [f["id"] for f in fields],
                           "properties": props},
                "narrative": {"type": "string"},
                "key_questions": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["question", "status", "answer"],
                    "properties": {"question": {"type": "string"}, "status": {"type": "string", "enum": list(M.QUESTION_STATUS)},
                                   "answer": {"type": "string"}}}},
                "attributes": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["name", "value"],
                    "properties": {"name": {"type": "string"}, "value": {"type": "string"}}}}}}


def _schema_lines(schema: dict) -> list[str]:
    lines: list[str] = []
    group = None
    for f in schema.get("fields", []):
        if f["group"] != group:
            group = f["group"]
            lines.append(f"[{group}]")
        bits = [f"type {f['type']}"]
        if f.get("unit"):
            bits.append(f"unit {f['unit']}")
        if f["type"] == "enum":
            bits.append("options: " + " | ".join(f.get("options") or []))
        if f.get("min") is not None or f.get("max") is not None:
            lo = M._num_text(f["min"]) if f.get("min") is not None else "-inf"
            hi = M._num_text(f["max"]) if f.get("max") is not None else "+inf"
            bits.append(f"plausible {lo} … {hi}")
        if f.get("tag"):
            bits.append(f"tag {f['tag']}")
        line = f"- {f['id']} — {f['label']} ({'; '.join(bits)})"
        if f.get("hist"):
            line += f"; history: {f['hist']}"
        if f.get("description"):
            line += f"\n    {f['description'][:400]}"
        lines.append(line)
    return lines


def _actor_output_lines(results: list[dict]) -> list[str]:
    lines: list[str] = []
    for r in results:
        if r.get("status") != "ok":
            lines += [f"## {r['name']} [{r['actor_id']}]: NO RESPONSE ({r.get('error') or 'failed'})",
                      "  Treat this actor as having taken no new action this period (status quo), and say so if it matters.", ""]
            continue
        o = r["output"]
        lines += [f"## {r['name']} [{r['actor_id']}]",
                  f"- action: {o['action_label']} [{o['action_id']}]",
                  f"- detail: {o.get('action_detail') or ''}",
                  f"- rationale: {o.get('rationale') or ''}",
                  f"- public statement: {o.get('public_statement') or ''}",
                  "- expected consequences: " + "; ".join(o.get("expected_consequences") or []), ""]
    return lines or ["(no actors)"]


def synth_messages(world: dict, state: dict, results: list[dict], history: list[dict], next_period: str) -> list[dict]:
    lang = world.get("language") or "English"
    system = "\n".join([
        f'You are the synthesizer (game master) of a world simulation ("{world["name"]}"). Resolve every actor\'s '
        f"action jointly into ONE coherent next state for period {next_period}, following the world's instructions.",
        "Answer with one JSON object only (no prose, no code fences):",
        '  "summary": one line on what happened this period',
        '  "values": an object with EVERY schema field id as a key, each {"value": <typed value>, "reason": <why it '
        'changed, or "" when unchanged>}. Use the field\'s type and unit; an enum takes one of its options exactly.',
        '  "narrative": the situation at the start of the new period (a few paragraphs)',
        '  "key_questions": the updated list of {"question", "status": "open"|"answered", "answer"} — answer what '
        "was resolved, keep what is still open (update its current answer), add new questions that arose, drop only "
        "questions that no longer matter",
        '  "attributes": the updated list of {"name", "value"}',
        "Rules: keep every value consistent with the other values and with the actions; change only what this "
        "period's events move; give a reason for every changed value; keep values plausible and say so in the reason "
        "when a value goes beyond its plausible range or its history; nothing an actor did not do happens by "
        "assumption unless it follows from the situation.",
        f"Write every text value (summary, reasons, narrative, questions, answers, attributes) in {lang}.",
    ])
    lines = _world_lines(world)
    if (world.get("synthesizer") or {}).get("instructions"):
        lines += ["", "# Synthesizer instructions", world["synthesizer"]["instructions"]]
    lines += ["", "# Schema (every field must get a value)"] + _schema_lines(world["schema"])
    lines += ["", f"# Current state: step {state.get('step')}, period {state.get('period')}"]
    lines += _state_lines(world["schema"], state)
    lines += [""] + _text_lines(state.get("text") or {})
    lines += ["", "# Recent history (oldest first)"] + _history_lines(history)
    lines += ["", f"# Actors' decisions for {state.get('period')} → {next_period}"] + _actor_output_lines(results)
    lines += [f"Write the state at {next_period} (step {int(state.get('step') or 0) + 1}). JSON only."]
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


# ------------------------------------------------------------------------------------------------ repair

def repair_messages(messages: list[dict], bad_text: str, error: str) -> list[dict]:
    return messages + [
        {"role": "assistant", "content": (bad_text or "")[:8000]},
        {"role": "user", "content": f"That answer could not be used: {error}. Reply again with ONLY the corrected "
                                    "JSON object in the required format: no prose, no code fences."}]


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
