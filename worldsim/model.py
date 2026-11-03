"""The world's data model as plain dicts: normalization of a world (schema, actors, synthesizer, initial state), value
coercion, period arithmetic, and the validation of the synthesizer's next state against the schema.

World (format worldsim-world/1)::

    {id, name, description, language, period_step {unit quarter|month|year, count}, history_window,
     model {model, temperature, reasoning_effort}           blank = the global default from Settings
     schema {fields: [Field]}, actors [Actor], synthesizer {instructions, model},
     initial_state State, created, updated}

Field:  {id, label, group, type number|percent|integer|text|enum|boolean, unit, description, options [enum],
         min, max (plausible bounds: a value outside is kept and flagged), hist (historical range note), tag}
State:  {step, period, values {field id: value | None}, text {narrative, key_questions [{question, status open|answered,
         answer}], attributes [{name, value}]}, summary}
Actor:  {id, name, enabled, role, cares_about [str], action_space [{id, label, description}], allow_free_action,
         private_info, visible_fields null (all) | [field id], model}

Nothing here calls a model, touches the disk or the network.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any

FORMAT = "worldsim-world/1"
FIELD_TYPES = ("number", "percent", "integer", "text", "enum", "boolean")
NUMERIC_TYPES = ("number", "percent", "integer")
PERIOD_UNITS = {"month": 1, "quarter": 3, "year": 12}
QUESTION_STATUS = ("open", "answered")
FREE_ACTION = "free"
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
WORLD_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
REASONING_EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh", "max")
LIMITS = {"name": 120, "label": 160, "short": 400, "text": 20000, "fields": 300, "actors": 60, "actions": 30,
          "options": 60, "questions": 40, "attributes": 60, "cares": 20}


class WorldError(ValueError):
    """A world (or a part of it) that does not validate: `problems` lists every reason, in English."""

    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems) or "invalid world")


# ---------------------------------------------------------------------------------------------------------- helpers

def _text(raw: Any, limit: int = LIMITS["text"]) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if isinstance(raw, (int, float)):
        return _num_text(raw)
    if not isinstance(raw, str):
        raw = str(raw)
    return raw.strip()[:limit]


def _num_text(x: float) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        return str(x)
    return f"{x:.6g}" if abs(x) < 1e15 else str(x)


def _finite(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if math.isfinite(float(raw)) else None
    return None


def slug(text: str, fallback: str = "item") -> str:
    """A lower-case id from free text ("Policy rate (%)" -> "policy_rate"); `fallback` when nothing is left."""
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    if not s or not s[0].isalpha():
        s = f"{fallback}_{s}" if s else fallback
    return s[:48].rstrip("_") or fallback


def unique_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    for i in range(2, 10000):
        cand = f"{base[:44]}_{i}"
        if cand not in taken:
            return cand
    raise WorldError(f"could not make a unique id from {base!r}")


# ---------------------------------------------------------------------------------------------------------- periods

Q_RE = re.compile(r"^(\d{4})\s*[Qq]([1-4])$")
M_RE = re.compile(r"^(\d{4})-(\d{1,2})$")
Y_RE = re.compile(r"^(\d{4})$")
TAIL_RE = re.compile(r"^(.*?)(\d+)$")


def advance_period(period: str, unit: str = "quarter", count: int = 1) -> str:
    """The period `count` units after `period`: 2026Q4 + 1 quarter = 2027Q1, 2026-11 + 1 month = 2026-12,
    2026 + 1 year = 2027, "Turn 3" + 1 = "Turn 4"; anything else gets " +n"."""
    p = str(period or "").strip()
    months = PERIOD_UNITS.get(unit, 3) * max(1, int(count or 1))
    m = Q_RE.match(p)
    if m:
        idx = int(m.group(1)) * 12 + (int(m.group(2)) - 1) * 3 + months
        y, mm = divmod(idx, 12)
        return f"{y}Q{mm // 3 + 1}"
    m = M_RE.match(p)
    if m and 1 <= int(m.group(2)) <= 12:
        idx = int(m.group(1)) * 12 + int(m.group(2)) - 1 + months
        y, mm = divmod(idx, 12)
        return f"{y}-{mm + 1:02d}"
    m = Y_RE.match(p)
    if m:
        return str(int(m.group(1)) + max(1, months // 12))
    m = TAIL_RE.match(p)
    if m:
        return f"{m.group(1)}{int(m.group(2)) + max(1, int(count or 1))}"
    return f"{p} +{max(1, int(count or 1))}" if p else f"+{max(1, int(count or 1))}"


# ---------------------------------------------------------------------------------------------------------- values

def coerce_value(field: dict, raw: Any) -> tuple[Any, str | None, bool]:
    """(value, note, ok) for `raw` as a value of `field`. ok False: the value cannot be read as this type (the caller
    decides what to do: carry the previous value forward, keep null). A note says what was changed to read it."""
    ftype = field.get("type")
    if raw is None:
        return None, None, False
    if ftype in NUMERIC_TYPES:
        num = _finite(raw)
        note = None
        if num is None and isinstance(raw, str):
            cleaned = (raw.strip().replace("，", "").replace(",", "").replace("％", "%").replace("−", "-")
                       .replace("pp", "").replace("%", "").replace("+", "").strip())
            try:
                num = float(cleaned)
                note = f"read {raw!r} as a number"
            except ValueError:
                num = None
            if num is not None and not math.isfinite(num):
                num = None
        if num is None:
            return None, None, False
        if ftype == "integer":
            if num != int(num):
                return int(round(num)), f"rounded {_num_text(num)} to an integer", True
            return int(num), note, True
        return float(num), note, True
    if ftype == "boolean":
        if isinstance(raw, bool):
            return raw, None, True
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw), f"read {raw!r} as a boolean", True
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in ("true", "yes", "y", "1", "是", "真", "对", "有"):
                return True, f"read {raw!r} as true", True
            if s in ("false", "no", "n", "0", "否", "假", "不", "无"):
                return False, f"read {raw!r} as false", True
        return None, None, False
    if ftype == "enum":
        options = field.get("options") or []
        if not isinstance(raw, str):
            raw = _text(raw)
        if raw in options:
            return raw, None, True
        key = re.sub(r"\s+", "", raw).lower()
        for opt in options:
            if re.sub(r"\s+", "", opt).lower() == key:
                return opt, f"matched {raw!r} to the option {opt!r}", True
        return None, None, False
    # text
    if isinstance(raw, (dict, list)):
        return None, None, False
    if isinstance(raw, str):
        return raw.strip()[:LIMITS["text"]], None, True
    return _text(raw), f"read {raw!r} as text", True


def values_equal(field: dict, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if field.get("type") in NUMERIC_TYPES:
        fa, fb = _finite(a), _finite(b)
        if fa is None or fb is None:
            return a == b
        return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))
    return a == b


def range_note(field: dict, value: Any) -> str | None:
    """Why a numeric value lies outside the field's plausible [min, max], or None."""
    if field.get("type") not in NUMERIC_TYPES:
        return None
    v = _finite(value)
    if v is None:
        return None
    lo, hi = field.get("min"), field.get("max")
    if lo is not None and v < lo:
        return f"{_num_text(v)} is below the plausible minimum {_num_text(lo)}"
    if hi is not None and v > hi:
        return f"{_num_text(v)} is above the plausible maximum {_num_text(hi)}"
    return None


# ---------------------------------------------------------------------------------------------------------- schema

def normalize_field(raw: Any, index: int = 0) -> tuple[dict, list[str]]:
    problems: list[str] = []
    if not isinstance(raw, dict):
        return {}, [f"field #{index + 1} is not an object"]
    fid = _text(raw.get("id"), 48)
    label = _text(raw.get("label"), LIMITS["label"]) or fid
    where = f"field {fid or '#' + str(index + 1)}"
    if not ID_RE.match(fid):
        problems.append(f"{where}: id must be lower-case letters, digits and _ (starting with a letter, at most 48)")
    ftype = _text(raw.get("type"), 20).lower() or "number"
    if ftype not in FIELD_TYPES:
        problems.append(f"{where}: type must be one of {', '.join(FIELD_TYPES)}")
    field = {"id": fid, "label": label, "group": _text(raw.get("group"), 80) or "General", "type": ftype,
             "unit": _text(raw.get("unit"), 40), "description": _text(raw.get("description"), 2000),
             "options": [], "min": None, "max": None, "hist": _text(raw.get("hist"), 400),
             "tag": _text(raw.get("tag"), 40)}
    if ftype == "enum":
        opts = raw.get("options")
        if isinstance(opts, str):
            opts = [o for o in re.split(r"[\n,，;；|]", opts)]
        seen: list[str] = []
        for o in opts or []:
            t = _text(o, LIMITS["label"])
            if t and t not in seen:
                seen.append(t)
        if not seen:
            problems.append(f"{where}: an enum needs at least one option")
        field["options"] = seen[:LIMITS["options"]]
    if ftype in NUMERIC_TYPES:
        for key in ("min", "max"):
            v = raw.get(key)
            if v in (None, ""):
                continue
            num, _, ok = coerce_value({"type": "number"}, v)
            if not ok:
                problems.append(f"{where}: {key} must be a number")
            else:
                field[key] = num
        if field["min"] is not None and field["max"] is not None and field["min"] > field["max"]:
            problems.append(f"{where}: min is greater than max")
    return field, problems


def normalize_schema(raw: Any) -> tuple[dict, list[str]]:
    fields_raw = raw.get("fields") if isinstance(raw, dict) else raw
    if not isinstance(fields_raw, list):
        return {"fields": []}, ["schema.fields must be a list"]
    problems: list[str] = []
    fields: list[dict] = []
    seen: set[str] = set()
    if len(fields_raw) > LIMITS["fields"]:
        problems.append(f"a schema has at most {LIMITS['fields']} fields")
    for i, f in enumerate(fields_raw[:LIMITS["fields"]]):
        field, probs = normalize_field(f, i)
        problems += probs
        if field.get("id") in seen:
            problems.append(f"field id {field['id']} is used twice")
        seen.add(field.get("id", ""))
        fields.append(field)
    return {"fields": fields}, problems


def field_map(schema: dict) -> dict[str, dict]:
    return {f["id"]: f for f in (schema or {}).get("fields", [])}


# ---------------------------------------------------------------------------------------------------------- state text

def normalize_questions(raw: Any) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    out: list[dict] = []
    for i, q in enumerate(raw if isinstance(raw, list) else []):
        if isinstance(q, str):
            q = {"question": q}
        if not isinstance(q, dict):
            notes.append(f"key question #{i + 1} is not an object: dropped")
            continue
        text = _text(q.get("question"), LIMITS["short"])
        if not text:
            notes.append(f"key question #{i + 1} has no question text: dropped")
            continue
        status = _text(q.get("status"), 20).lower()
        if status not in QUESTION_STATUS:
            if status:
                notes.append(f"key question {text[:40]!r}: status {status!r} read as 'open'")
            status = "open"
        out.append({"question": text, "status": status, "answer": _text(q.get("answer"), 2000)})
    if len(out) > LIMITS["questions"]:
        notes.append(f"only the first {LIMITS['questions']} key questions were kept")
        out = out[:LIMITS["questions"]]
    return out, notes


def normalize_attributes(raw: Any) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    out: list[dict] = []
    if isinstance(raw, dict):
        raw = [{"name": k, "value": v} for k, v in raw.items()]
    seen: set[str] = set()
    for i, a in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(a, dict):
            notes.append(f"attribute #{i + 1} is not an object: dropped")
            continue
        name = _text(a.get("name"), LIMITS["label"])
        if not name:
            notes.append(f"attribute #{i + 1} has no name: dropped")
            continue
        if name in seen:
            notes.append(f"attribute {name!r} appears twice: the first was kept")
            continue
        seen.add(name)
        out.append({"name": name, "value": _text(a.get("value"), 2000)})
    if len(out) > LIMITS["attributes"]:
        notes.append(f"only the first {LIMITS['attributes']} attributes were kept")
        out = out[:LIMITS["attributes"]]
    return out, notes


def empty_text() -> dict:
    return {"narrative": "", "key_questions": [], "attributes": []}


def normalize_state(schema: dict, raw: Any, *, step: int = 0) -> tuple[dict, list[str]]:
    """An initial (or imported) state: values coerced to their field types (a value that cannot be read is an error;
    a missing value is null), unknown value ids dropped, the text normalized. Returns (state, problems)."""
    problems: list[str] = []
    raw = raw if isinstance(raw, dict) else {}
    fmap = field_map(schema)
    vals_raw = raw.get("values") if isinstance(raw.get("values"), dict) else {}
    values: dict[str, Any] = {}
    for fid, field in fmap.items():
        v = vals_raw.get(fid)
        if v is None or (isinstance(v, str) and v.strip() == "" and field["type"] != "text"):
            values[fid] = None
            continue
        val, _, ok = coerce_value(field, v)
        if not ok:
            problems.append(f"value of {fid} ({field['label']}) cannot be read as {field['type']}: {v!r}")
            values[fid] = None
        else:
            values[fid] = val
    text_raw = raw.get("text") if isinstance(raw.get("text"), dict) else {}
    questions, _ = normalize_questions(text_raw.get("key_questions"))
    attributes, _ = normalize_attributes(text_raw.get("attributes"))
    state = {"step": int(raw.get("step", step) or step), "period": _text(raw.get("period"), 40) or "T0",
             "values": values,
             "text": {"narrative": _text(text_raw.get("narrative")), "key_questions": questions,
                      "attributes": attributes},
             "summary": _text(raw.get("summary"), LIMITS["short"])}
    for key in ("flags", "changes", "created", "reasons", "usage", "cost", "cost_complete"):
        if key in raw:
            state[key] = copy.deepcopy(raw[key])
    return state, problems


# ---------------------------------------------------------------------------------------------------------- actors

def normalize_actor(raw: Any, index: int = 0, field_ids: set[str] | None = None) -> tuple[dict, list[str]]:
    problems: list[str] = []
    if not isinstance(raw, dict):
        return {}, [f"actor #{index + 1} is not an object"]
    aid = _text(raw.get("id"), 48)
    name = _text(raw.get("name"), LIMITS["name"]) or aid
    where = f"actor {aid or '#' + str(index + 1)}"
    if not ID_RE.match(aid):
        problems.append(f"{where}: id must be lower-case letters, digits and _ (starting with a letter)")
    cares = raw.get("cares_about")
    if isinstance(cares, str):
        cares = [c for c in cares.splitlines()]
    cares_list = [t for t in (_text(c, LIMITS["short"]) for c in (cares or [])) if t][:LIMITS["cares"]]
    actions: list[dict] = []
    seen: set[str] = set()
    for j, a in enumerate(raw.get("action_space") or []):
        if not isinstance(a, dict):
            problems.append(f"{where}: action #{j + 1} is not an object")
            continue
        act_id = _text(a.get("id"), 48)
        if not ID_RE.match(act_id):
            problems.append(f"{where}: action id {act_id!r} must be lower-case letters, digits and _")
        elif act_id == FREE_ACTION:
            problems.append(f"{where}: action id 'free' is reserved for free actions")
        elif act_id in seen:
            problems.append(f"{where}: action id {act_id} is used twice")
        seen.add(act_id)
        actions.append({"id": act_id, "label": _text(a.get("label"), LIMITS["label"]) or act_id,
                        "description": _text(a.get("description"), 2000)})
    if len(actions) > LIMITS["actions"]:
        problems.append(f"{where}: at most {LIMITS['actions']} actions")
    allow_free = bool(raw.get("allow_free_action", True))
    if not actions and not allow_free:
        problems.append(f"{where}: needs at least one action, or free actions allowed")
    visible = raw.get("visible_fields")
    if visible is not None:
        if not isinstance(visible, list):
            problems.append(f"{where}: visible_fields must be a list of field ids or null (all)")
            visible = None
        else:
            visible = [_text(v, 48) for v in visible if _text(v, 48)]
            if field_ids is not None:
                visible = [v for v in visible if v in field_ids]
    actor = {"id": aid, "name": name, "enabled": bool(raw.get("enabled", True)),
             "role": _text(raw.get("role")), "cares_about": cares_list, "action_space": actions,
             "allow_free_action": allow_free, "private_info": _text(raw.get("private_info")),
             "visible_fields": visible, "model": _text(raw.get("model"), 200)}
    return actor, problems


def normalize_actors(raw: Any, field_ids: set[str] | None = None) -> tuple[list[dict], list[str]]:
    if not isinstance(raw, list):
        return [], ["actors must be a list"]
    problems: list[str] = []
    out: list[dict] = []
    seen: set[str] = set()
    if len(raw) > LIMITS["actors"]:
        problems.append(f"a world has at most {LIMITS['actors']} actors")
    for i, a in enumerate(raw[:LIMITS["actors"]]):
        actor, probs = normalize_actor(a, i, field_ids)
        problems += probs
        if actor.get("id") in seen:
            problems.append(f"actor id {actor['id']} is used twice")
        seen.add(actor.get("id", ""))
        out.append(actor)
    return out, problems


def normalize_synthesizer(raw: Any) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {"instructions": _text(raw.get("instructions")), "model": _text(raw.get("model"), 200)}


def normalize_model_settings(raw: Any) -> tuple[dict, list[str]]:
    raw = raw if isinstance(raw, dict) else {}
    problems: list[str] = []
    temp = raw.get("temperature")
    if temp in ("", None):
        temp = None
    else:
        t, _, ok = coerce_value({"type": "number"}, temp)
        if not ok or not 0 <= t <= 2:
            problems.append("model.temperature must be between 0 and 2 (or blank)")
            temp = None
        else:
            temp = t
    effort = _text(raw.get("reasoning_effort"), 20).lower()
    if effort not in REASONING_EFFORTS:
        problems.append(f"model.reasoning_effort must be one of {', '.join(e or '(default)' for e in REASONING_EFFORTS)}")
        effort = ""
    return {"model": _text(raw.get("model"), 200), "temperature": temp, "reasoning_effort": effort}, problems


# ---------------------------------------------------------------------------------------------------------- world

def normalize_world(raw: Any, *, require_id: bool = True) -> tuple[dict, list[str]]:
    """A whole world. Returns (world, problems); the caller refuses the world when problems is not empty."""
    if not isinstance(raw, dict):
        return {}, ["a world must be a JSON object"]
    problems: list[str] = []
    wid = _text(raw.get("id"), 64)
    if require_id and not WORLD_ID_RE.match(wid):
        problems.append("world id must be lower-case letters, digits, _ or - (at most 64)")
    schema, probs = normalize_schema(raw.get("schema") or {"fields": []})
    problems += probs
    fids = set(field_map(schema))
    actors, probs = normalize_actors(raw.get("actors") or [], fids)
    problems += probs
    model, probs = normalize_model_settings(raw.get("model"))
    problems += probs
    step = raw.get("period_step") if isinstance(raw.get("period_step"), dict) else {}
    unit = _text(step.get("unit"), 20).lower() or "quarter"
    if unit not in PERIOD_UNITS:
        problems.append("period_step.unit must be month, quarter or year")
        unit = "quarter"
    try:
        count = int(step.get("count") or 1)
    except (TypeError, ValueError):
        count = 1
    if not 1 <= count <= 120:
        problems.append("period_step.count must be between 1 and 120")
        count = max(1, min(120, count))
    try:
        window = int(raw.get("history_window", 3))
    except (TypeError, ValueError):
        window = 3
    window = max(0, min(20, window))
    initial, probs = normalize_state(schema, raw.get("initial_state") or {}, step=0)
    problems += [f"initial state: {p}" for p in probs]
    initial["step"] = 0
    world = {"format": FORMAT, "id": wid, "name": _text(raw.get("name"), LIMITS["name"]) or "Untitled world",
             "description": _text(raw.get("description")), "language": _text(raw.get("language"), 40) or "English",
             "period_step": {"unit": unit, "count": count}, "history_window": window, "model": model,
             "schema": schema, "actors": actors, "synthesizer": normalize_synthesizer(raw.get("synthesizer")),
             "initial_state": initial, "created": _text(raw.get("created"), 40), "updated": _text(raw.get("updated"), 40)}
    return world, problems


def blank_world(wid: str, name: str = "New world") -> dict:
    """A small valid world to start from."""
    world, problems = normalize_world({
        "id": wid, "name": name, "language": "English",
        "description": "Describe the world, its scope and what one step means.",
        "schema": {"fields": [
            {"id": "tension", "label": "Tension", "group": "General", "type": "number", "unit": "0-10",
             "description": "How tense the situation is (0 calm, 10 crisis).", "min": 0, "max": 10},
            {"id": "phase", "label": "Phase", "group": "General", "type": "enum", "options": ["calm", "dispute", "crisis"],
             "description": "The phase of the situation."}]},
        "actors": [{"id": "actor_a", "name": "Actor A", "role": "Describe who this actor is.",
                    "cares_about": ["What this actor wants"],
                    "action_space": [{"id": "status_quo", "label": "Keep the status quo", "description": "Do nothing new."},
                                     {"id": "escalate", "label": "Escalate", "description": "Raise the pressure."}],
                    "allow_free_action": True}],
        "synthesizer": {"instructions": "Resolve all actions jointly into one coherent next state."},
        "initial_state": {"period": "T0", "values": {"tension": 3, "phase": "calm"},
                          "text": {"narrative": "The starting situation.", "key_questions": [
                              {"question": "Will the dispute escalate?", "status": "open", "answer": ""}],
                              "attributes": [{"name": "Mood", "value": "uneasy"}]}},
    })
    if problems:                                     # pragma: no cover - a fixed literal
        raise WorldError(problems)
    return world


# ---------------------------------------------------------------------------------------------------------- next state

def _flag(fid: str | None, level: str, code: str, message: str) -> dict:
    return {"field": fid, "level": level, "code": code, "message": message}


def validate_next_state(schema: dict, current: dict, output: Any, *, period: str, step: int) -> tuple[dict, list[dict], list[dict]]:
    """The synthesizer's answer checked against the schema -> (next state, flags, changes).

    Every schema field gets a value: the synthesizer's (coerced to the field type), else the current value carried
    forward and flagged (missing, unreadable, an enum outside its options, null). A numeric value outside the plausible
    range is kept and flagged. A changed field without a reason is flagged. Unknown field ids are ignored and flagged:
    nothing is invented. The text (narrative, key questions, attributes) falls back to the current text, flagged."""
    out = output if isinstance(output, dict) else {}
    flags: list[dict] = []
    changes: list[dict] = []
    fmap = field_map(schema)
    cur_vals = (current or {}).get("values") or {}
    raw_vals = out.get("values")
    if not isinstance(raw_vals, dict):
        flags.append(_flag(None, "error", "values", "the answer has no 'values' object: every value was carried forward"))
        raw_vals = {}
    values: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for fid, field in fmap.items():
        prev = cur_vals.get(fid)
        entry = raw_vals.get(fid, None)
        has = fid in raw_vals
        reason = ""
        if isinstance(entry, dict) and ("value" in entry or "reason" in entry):
            reason = _text(entry.get("reason"), 600)
            raw = entry.get("value")
        else:
            raw = entry
        if not has:
            values[fid] = prev
            flags.append(_flag(fid, "warn", "missing", "no value given: the previous value was carried forward"))
            continue
        if raw is None:
            values[fid] = prev
            flags.append(_flag(fid, "warn", "null", "the value was null: the previous value was carried forward"))
            continue
        val, note, ok = coerce_value(field, raw)
        if not ok:
            values[fid] = prev
            what = f"not one of the options ({', '.join(field.get('options') or [])})" if field["type"] == "enum" \
                else f"not a valid {field['type']}"
            flags.append(_flag(fid, "error", "type", f"{_text(raw, 80)!r} is {what}: the previous value was carried forward"))
            continue
        if note:
            flags.append(_flag(fid, "info", "coerced", note))
        rng = range_note(field, val)
        if rng:
            flags.append(_flag(fid, "warn", "range", f"{rng} (kept; check it)"))
        values[fid] = val
        if not values_equal(field, prev, val):
            if not reason:
                flags.append(_flag(fid, "warn", "no_reason", "the value changed without a reason"))
            reasons[fid] = reason
            changes.append({"field": fid, "label": field["label"], "from": prev, "to": val, "reason": reason})
        elif reason:
            reasons[fid] = reason
    for fid in raw_vals:
        if fid not in fmap:
            flags.append(_flag(fid, "warn", "unknown_field", "not a schema field: ignored"))
    cur_text = (current or {}).get("text") or empty_text()
    narrative = _text(out.get("narrative"))
    if not narrative:
        narrative = cur_text.get("narrative", "")
        flags.append(_flag(None, "error", "narrative", "no narrative: the previous narrative was carried forward"))
    if isinstance(out.get("key_questions"), list):
        questions, notes = normalize_questions(out.get("key_questions"))
        flags += [_flag(None, "info", "key_questions", n) for n in notes]
    else:
        questions = copy.deepcopy(cur_text.get("key_questions") or [])
        flags.append(_flag(None, "warn", "key_questions", "no key questions list: the previous questions were carried forward"))
    if isinstance(out.get("attributes"), (list, dict)):
        attributes, notes = normalize_attributes(out.get("attributes"))
        flags += [_flag(None, "info", "attributes", n) for n in notes]
    else:
        attributes = copy.deepcopy(cur_text.get("attributes") or [])
        flags.append(_flag(None, "warn", "attributes", "no attributes list: the previous attributes were carried forward"))
    summary = _text(out.get("summary"), LIMITS["short"])
    if not summary:
        flags.append(_flag(None, "info", "summary", "no one-line summary"))
    state = {"step": step, "period": period, "values": values,
             "text": {"narrative": narrative, "key_questions": questions, "attributes": attributes},
             "summary": summary, "reasons": reasons, "flags": flags, "changes": changes}
    return state, flags, changes


def text_changes(prev_text: dict, new_text: dict) -> dict:
    """What changed in the text between two states: new / answered / dropped questions, changed attributes."""
    prev_q = {q["question"]: q for q in (prev_text or {}).get("key_questions", [])}
    new_q = {q["question"]: q for q in (new_text or {}).get("key_questions", [])}
    prev_a = {a["name"]: a["value"] for a in (prev_text or {}).get("attributes", [])}
    new_a = {a["name"]: a["value"] for a in (new_text or {}).get("attributes", [])}
    return {
        "new_questions": [q for q in new_q if q not in prev_q],
        "answered_questions": [q for q, v in new_q.items() if v["status"] == "answered"
                               and prev_q.get(q, {}).get("status") != "answered"],
        "dropped_questions": [q for q in prev_q if q not in new_q],
        "changed_attributes": [{"name": n, "from": prev_a.get(n), "to": v} for n, v in new_a.items() if prev_a.get(n) != v],
        "dropped_attributes": [n for n in prev_a if n not in new_a],
    }
