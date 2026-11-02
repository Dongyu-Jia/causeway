"""MockClaude: a deterministic fake Claude Code (MSL_LLM_BACKEND=mock). No process, no network, no plan usage.

It answers exactly like claude_code.run (the answer's text and metadata, or a ClaudeCodeError with a kind) and reads
only what a real model would read: the system prompt and the user message. The output schema it must satisfy is the
one the prompt carries (under "## 输出格式（JSON schema）"); its schema_version names the call type:

    agents-actor-2 / agents-node-2   per actor, the history prior's modal option (MSL_LLM_MOCK_MODE=modal, the
                                     default) or a seeded draw from the prior (draw) gets the mass, the other listed
                                     options 0.03 each; "prior" reports the prior itself (top four options). The
                                     prior is the 历史基率 column of the actor's menu table; followers answer with the
                                     prior rows of their reaction table.
    agents-synth-1                   partitions only: each cluster is cut on its first permitted axis at the regions
                                     the prompt lists, with the model's own masses (one central branch), so the
                                     synthesizer layer is neutral.
    agents-root-1                    h0 from the narrative, no forced fields, opinion sentences as user_views.
    agents-story-1, agents-leak-1    fixed, schema-valid answers.
    a P5 repair                      the previous output, re-parsed and with probabilities renormalised.
    the translator's spec            a valid spec built from keywords of the narrative (oil, recession, inflation,
                                     rate cuts or hikes), in the shape SPEC_JSON_SCHEMA describes.

Fixtures turn a call into an edge case (FIXTURES): malformed or fenced JSON, trailing commas, full-width quotes,
percent probabilities, a bad probability sum, a price sentence, an out-of-power write, a decline, a refusal (as
the envelope's stop_reason, or as plain text), a timeout, a hang until cancelled, usage-limit texts (the epoch form
and the "resets 3pm" form, classified by claude_code's own classify_failure), login failures, 429 / 529 / 5xx,
max_tokens, context, an empty answer, a missing CLI and a crash. Queue them with push(name, match=None, times=1)
(match: a call kind such as "actor", "node", "synth", "repair"; "actor:fed" for a kind plus a substring of the
user message; any other string is a substring; or a callable(kind, system, user)), or persistently with
MSL_LLM_MOCK_FIXTURE=name[@match][,name@match]. Recorded answers (cassettes) are read first from
MSL_LLM_MOCK_CASSETTES=<dir>/<claude_code.request_key>.json holding {"text": …} or {"error": {"kind", "message"}}.

Settings, read at every call: MSL_LLM_MOCK_MODE (modal | draw | prior), MSL_LLM_MOCK_SEED (an integer, for draw),
MSL_LLM_MOCK_DELAY_S (simulated latency, at most 30 s; a cancel event interrupts it).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from . import claude_code as cc
from .agent_prompts import (ACTOR_HEADER, CLUSTER_HEADER, CUSTOM_ID, MENU_COLUMNS, MIN_BRANCH_PROB, MIN_PROB,
                            REACTION_HEADER, SCHEMA_HEADER, VERSIONS)
from .jsonfix import extract_json, truncate

MOCK_VERSION = "mock-1"
LABEL = "模拟后端（MSL_LLM_BACKEND=mock：不调用模型，不消耗订阅额度）"
MODES = ("modal", "draw", "prior")
MAX_DELAY_S = 30.0
HANG_MAX_S = 30.0
CALL_LOG = 500
PRICE_SENTENCE = "加息后美股预计将下跌 12%，纳指跌至 18000 点。"
OPINION = re.compile(r"我认为|我觉得|我相信|我预计|我判断|一定会|肯定会|必然会")
CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")


@dataclass
class Fixture:
    name: str
    match: object = None          # None | str | callable(kind, system, user) -> bool
    times: int | None = 1         # None: every matching call


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _tokens(text: str) -> int:
    """The convention of the design docs: one token per CJK character, one per four other characters."""
    s = str(text or "")
    cjk = len(CJK.findall(s))
    return cjk + (len(s) - cjk + 3) // 4


def _two_decimals(values: list) -> list:
    """Probabilities rounded to two decimals that still sum to exactly 1."""
    if not values:
        return []
    total = sum(values) or 1.0
    r = [round(v / total, 2) for v in values]
    k = max(range(len(r)), key=lambda i: r[i])
    r[k] = round(r[k] + 1.0 - sum(r), 2)
    return r


# ---------------------------------------------------------------- reading the prompt

def schema_from_system(system: str) -> dict | None:
    """The JSON schema a prompt carries (the line after SCHEMA_HEADER that is a JSON object), or None."""
    s = str(system or "")
    i = s.find(SCHEMA_HEADER)
    if i < 0:
        return None
    for line in s[i + len(SCHEMA_HEADER):].splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                return None
            return obj if isinstance(obj, dict) else None
    return None


def schema_version(schema: dict | None) -> str | None:
    sv = ((schema or {}).get("properties") or {}).get("schema_version") or {}
    if "const" in sv:
        return sv["const"]
    enum = sv.get("enum") or []
    return enum[0] if enum else None


def call_kind(schema: dict | None, user: str) -> str:
    """actor | node | synth | root | probe | story | repair | translator | generic."""
    if "<previous_output>" in str(user or ""):
        return "repair"
    version = schema_version(schema)
    for kind, v in VERSIONS.items():
        if version == v:
            return kind
    if "phases" in ((schema or {}).get("properties") or {}):
        return "translator"
    return "generic"


def _top_sections(text: str) -> str:
    """Text up to the next top-level ("# ") header."""
    out = []
    for line in text.splitlines():
        if line.startswith("# ") and out:
            break
        out.append(line)
    return "\n".join(out)


def actor_sections(user: str) -> dict[str, str]:
    """{actor id: its "## 行为者：" section} of a P1 / P2 user message."""
    out = {}
    for part in re.split(rf"(?m)^(?={re.escape(ACTOR_HEADER)})", str(user or "")):
        m = re.match(rf"{re.escape(ACTOR_HEADER)}.*?id: ([a-z0-9_]+)", part)
        if m:
            out[m.group(1)] = _top_sections(part)
    return out


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _first_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(text or "").replace(",", ""))
    return float(m.group(0)) if m else None


def menu_rows(section: str) -> dict[str, dict]:
    """{option id: {"status_quo", "default", "base_rate"}} from an actor section's menu table."""
    out = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if len(cells) != len(MENU_COLUMNS) or cells[0] == MENU_COLUMNS[0] or set(cells[0]) <= {"-"}:
            continue
        out[cells[0]] = {"status_quo": cells[2] == "是",
                         "default": None if cells[3] in ("", "—") else _first_number(cells[3]),
                         "base_rate": None if cells[5] in ("", "—") else _first_number(cells[5])}
    return out


def reaction_tables(user: str) -> dict[tuple, dict]:
    """{(leader, follower): {leader option: {follower option: p}}} from the "## 季内反应" tables."""
    out: dict = {}
    for part in re.split(rf"(?m)^(?={re.escape(REACTION_HEADER)})", str(user or "")):
        m = re.match(rf"{re.escape(REACTION_HEADER)}([a-z0-9_]+) → ([a-z0-9_]+)", part)
        if not m:
            continue
        lines = []
        for ln in part.splitlines()[1:]:            # the table right under the header, nothing after it
            if not ln.startswith("|"):
                break
            lines.append(ln)
        if len(lines) < 2:
            continue
        header = _cells(lines[0])[1:]
        rows = {}
        for ln in lines[2:]:
            cells = _cells(ln)
            probs = {f: _first_number(v) for f, v in zip(header, cells[1:]) if v not in ("", "—")}
            rows[cells[0]] = {f: p for f, p in probs.items() if p is not None}
        out[(m.group(1), m.group(2))] = rows
    return out


REGION = re.compile(r"\((-∞|-?\d+(?:\.\d+)?),(\+∞|-?\d+(?:\.\d+)?)[\])]\s*(.*?)\s+(\d+(?:\.\d+)?)\s*$")


def cluster_sections(user: str) -> dict[str, dict]:
    """{cluster id: {"written": [...], "masses": {axis: [(lo, hi, name, mass)]}}} from an S1 user message."""
    out = {}
    for part in re.split(rf"(?m)^(?={re.escape(CLUSTER_HEADER)})", str(user or "")):
        m = re.match(rf"{re.escape(CLUSTER_HEADER)}([a-z0-9_]+)", part)
        if not m:
            continue
        section = _top_sections(part)
        written: list[str] = []
        w = re.search(r"本季已由动作写入的字段[^：]*：(.*)", section)
        if w and w.group(1).strip() != "无":
            written = [x.strip() for x in w.group(1).split("、") if x.strip()]
        masses: dict = {}
        for line in section.splitlines():
            a = re.match(r"- ([a-z][a-z0-9_]*)(?:（[^）]*）)?：(.*)", line)
            if not a:
                continue
            regions = []
            for piece in a.group(2).split("｜"):
                r = REGION.search(piece.strip())
                if r:
                    lo = None if r.group(1) == "-∞" else float(r.group(1))
                    hi = None if r.group(2) == "+∞" else float(r.group(2))
                    regions.append((lo, hi, r.group(3).strip(), float(r.group(4))))
            if regions:
                masses[a.group(1)] = regions
        out[m.group(1)] = {"written": written, "masses": masses}
    return out


def _tag(user: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", str(user or ""), re.S)
    return m.group(1).strip() if m else ""


def _enum(schema: dict, *path) -> list:
    node = schema
    for p in path:
        node = (node or {}).get(p) or {}
    return list(node.get("enum") or ([node["const"]] if "const" in node else []))


# ---------------------------------------------------------------- answers

def _rng(*parts) -> random.Random:
    seed = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(seed[:16], 16))


def _options(ids: list, menu: dict, mode: str, rng: random.Random, follower: bool) -> list[dict]:
    """The listed options of one actor: the chosen option (modal or drawn) with the mass, the status quo and the
    runner-up at 0.03; or, in "prior" mode, the top four options at their base rates."""
    ids = [i for i in ids if i != CUSTOM_ID]
    if not ids:
        return []
    base = {i: menu[i]["base_rate"] for i in ids if menu.get(i) and menu[i]["base_rate"] is not None}
    sq = next((i for i in ids if (menu.get(i) or {}).get("status_quo")), ids[0])
    ranked = sorted(ids, key=lambda i: (-base.get(i, 0.0), ids.index(i))) if base else [sq] + [i for i in ids if i != sq]
    if mode == "prior" and base:
        listed = ranked[:4] if sq in ranked[:4] else ranked[:3] + [sq]
        probs = _two_decimals([max(base.get(i, 0.0), MIN_PROB) for i in listed])
        why = "模拟后端：照抄历史基率"
    else:
        if mode == "draw":
            weights = [base.get(i, 0.0) for i in ids] if base and sum(base.values()) > 0 else [1.0] * len(ids)
            chosen = rng.choices(ids, weights=weights, k=1)[0]
            why = "模拟后端：按历史基率抽样选择"
        else:
            chosen = ranked[0]
            why = "模拟后端：选择历史基率的众数选项"
        listed = [chosen] + [i for i in [sq] + ranked if i != chosen]
        listed = list(dict.fromkeys(listed))[:3]
        if len(listed) < 2:
            listed = (listed + [i for i in ids if i not in listed])[:2]
        probs = _two_decimals([1.0 - MIN_PROB * (len(listed) - 1)] + [MIN_PROB] * (len(listed) - 1))
    return [{"id": i, "prob": None if follower else p, "magnitude": (menu.get(i) or {}).get("default"),
             "rationale": why if k == 0 else "模拟后端：保留的备选", "beyond_history": None, "signal": None}
            for k, (i, p) in enumerate(zip(listed, probs))]      # no label: the guard takes the menu's


def _actor_block(aid: str, ids: list, user: str, mode: str, seed: str, follower: bool) -> dict:
    menu = menu_rows(actor_sections(user).get(aid, ""))
    opts = _options(ids, menu, mode, _rng(seed, aid, user), follower)
    return {"status": "ok", "options": opts,
            "vs_base_rate": None if mode == "prior" else "模拟后端：概率集中于所选选项，不代表任何判断"}


def _rows(leader_options: list, follower_options: list, prior: dict, menu: dict) -> list[dict]:
    rows = []
    base = {f: (menu.get(f) or {}).get("base_rate") for f in follower_options}
    for lo in leader_options:
        row = {f: p for f, p in (prior.get(lo) or {}).items() if f in follower_options and p and p > 0}
        if not row:
            row = {f: p for f, p in base.items() if p}
        if not row:
            row = {f: 1.0 for f in follower_options[:2]}
        keys = list(row)
        rows.append({"leader_option": lo, "follower_probs": [{"option": f, "prob": p}
                                                             for f, p in zip(keys, _two_decimals(list(row.values())))]})
    return rows


def actor_answer(schema: dict, user: str, mode: str, seed: str) -> dict:
    props = schema.get("properties") or {}
    aid = _enum(schema, "properties", "actor")[0]
    ids = _enum(props.get("options") or {}, "items", "properties", "id")
    reaction = props.get("reaction")
    block = _actor_block(aid, ids, user, mode, seed, follower=reaction is not None)
    out = {"schema_version": VERSIONS["actor"], "actor": aid, **block, "private_note": None}
    if reaction is not None:
        leader = _enum(reaction, "properties", "leader")[0]
        rows_schema = ((reaction.get("properties") or {}).get("rows") or {}).get("items") or {}
        leader_options = _enum(rows_schema, "properties", "leader_option")
        follower_options = _enum(rows_schema, "properties", "follower_probs", "items", "properties", "option")
        prior = reaction_tables(user).get((leader, aid), {})
        menu = menu_rows(actor_sections(user).get(aid, ""))
        out["reaction"] = {"leader": leader, "rows": _rows(leader_options, follower_options, prior, menu)}
    return out


def node_answer(schema: dict, user: str, mode: str, seed: str) -> dict:
    actors_schema = ((schema.get("properties") or {}).get("actors") or {}).get("properties") or {}
    actors = {}
    for aid, block_schema in actors_schema.items():
        ids = _enum(block_schema, "properties", "options", "items", "properties", "id")
        actors[aid] = _actor_block(aid, ids, user, mode, seed, follower=False)
    reactions = []
    sections = actor_sections(user)
    for (leader, follower), prior in sorted(reaction_tables(user).items()):
        if leader not in actors or follower not in actors:
            continue
        leader_options = list(prior)
        f_ids = _enum(actors_schema[follower], "properties", "options", "items", "properties", "id")
        f_ids = [i for i in f_ids if i != CUSTOM_ID]
        menu = menu_rows(sections.get(follower, ""))
        reactions.append({"leader": leader, "follower": follower,
                          "rows": _rows(leader_options, f_ids, prior, menu)})
    return {"schema_version": VERSIONS["node"], "status": "ok", "actors": actors, "reactions": reactions,
            "hazard_tilts": []}


def _partition(axis: str, regions: list) -> list[dict]:
    """Branches of one axis from (lo, hi, name, mass) regions: small regions (< 0.05) join their inner neighbour;
    the region holding z = 0 is the central branch; probabilities are the model's masses."""
    regs = [list(r) for r in regions]

    def holds_zero(r):
        return (r[0] is None or r[0] < 0) and (r[1] is None or r[1] > 0) or r[0] == 0 or r[1] == 0

    centre = next((k for k, r in enumerate(regs) if holds_zero(r)), len(regs) // 2)
    changed = True
    while changed and len(regs) > 2:
        changed = False
        for k, r in enumerate(regs):
            if r[3] < MIN_BRANCH_PROB and k != centre:
                j = k + 1 if k < centre else k - 1
                a, b = sorted((k, j))
                merged = [regs[a][0], regs[b][1], regs[j][2], regs[a][3] + regs[b][3]]
                regs[a:b + 1] = [merged]
                centre = centre - 1 if k < centre else centre
                changed = True
                break
    probs = _two_decimals([max(r[3], 1e-6) for r in regs])
    out = []
    for k, (r, p) in enumerate(zip(regs, probs)):
        central = k == centre
        out.append({"id": "central" if central else f"part{k + 1}",
                    "label": "模型中枢" if central else truncate(r[2] or f"区间{k + 1}", 12),
                    "kind": "central" if central else "partition", "prob": p,
                    "region": {"lo_z": r[0], "hi_z": r[1]}, "shifts": {}, "cause": {"options": []},
                    "mechanism": "模拟后端：模型自身分布的一段", "anchors": [], "unintended": False,
                    "beyond_history": None, "facts": [], "memory": f"模拟后端：按模型分布分区（{axis}）",
                    "vs_prior": None})
    return out


def synth_answer(schema: dict, user: str) -> dict:
    clusters_schema = ((schema.get("properties") or {}).get("clusters") or {}).get("items") or {}
    ids = _enum(clusters_schema, "properties", "cluster_id")
    axes = [a for a in _enum(clusters_schema, "properties", "axis") if a is not None]
    sections = cluster_sections(user)
    clusters = []
    for cid in ids:
        sec = sections.get(cid, {"written": [], "masses": {}})
        usable = [a for a in sec["masses"] if a in axes and a not in sec["written"]]
        if usable:
            axis = usable[0]
            regions = sec["masses"][axis]
        else:
            axis = next((a for a in axes if a not in sec["written"]), None)
            regions = [(None, -1.0, "低于中枢", 0.159), (-1.0, 1.0, "中枢", 0.682), (1.0, None, "高于中枢", 0.159)]
        if axis is None:
            clusters.append({"cluster_id": cid, "status": "decline", "decline_reason": "模拟后端：没有可用的分区轴",
                             "axis": None, "branches": []})
            continue
        clusters.append({"cluster_id": cid, "status": "ok", "axis": axis, "branches": _partition(axis, regions)})
    return {"schema_version": VERSIONS["synth"], "status": "ok", "clusters": clusters}


def root_answer(schema: dict, user: str) -> dict:
    narrative = _tag(user, "user_narrative")
    narrative = "" if narrative == "（空）" else narrative
    plain = re.sub(r"\s+", " ", narrative).strip()
    views = [truncate(s.strip(), 80) for s in re.split(r"(?<=[。！？!?；;])", narrative) if OPINION.search(s)][:8]
    return {"schema_version": VERSIONS["root"], "status": "ok",
            "h0": {"headline": truncate(plain, 30) or "模拟起点：没有用户叙事", "summary": truncate(plain, 500),
                   "facts_used": []},
            "forced": [], "user_views": views, "unmapped": []}


def story_answer(schema: dict, user: str) -> dict:
    lines = [ln.strip() for ln in _tag(user, "path").splitlines() if ln.strip()]
    return {"schema_version": VERSIONS["story"], "headline": "模拟叙事",
            "story": truncate("；".join(lines) or "模拟后端：没有路径。", 600), "facts_used": []}


def probe_answer(schema: dict, user: str) -> dict:
    return {"schema_version": VERSIONS["probe"], "year_guess": 2000, "quarter_guess": 1, "confidence": 0.0,
            "episode": "模拟后端：不识别", "clues": []}


THEMES = (  # keyword -> views (the INLINE_EXAMPLE's validated magnitudes) and one causal link
    (r"石油|原油|油价|霍尔木兹|欧佩克|OPEC", [("f_oil", "模拟：油价上行，波动放大", {"mean_annual": 0.9, "vol_mult": 1.5}),
                                       ("s_cpi_yoy", "模拟：能源推高整体通胀", {"mean": 6.0})],
     ("油价", "通胀", "+", "1-3月", "模拟后端：能源成本进入物价")),
    (r"衰退|失业|萧条|裁员", [("f_unrate", "模拟：失业率上升", {"mean_annual": 0.6}),
                       ("f_ip", "模拟：工业产出下降", {"mean_annual": -0.03})],
     ("需求", "失业率", "-", "3-12月", "模拟后端：需求走弱后企业裁员")),
    (r"通胀|滞胀|物价", [("s_cpi_yoy", "模拟：通胀偏高", {"mean": 6.0})],
     ("物价", "政策利率", "+", "3-9月", "模拟后端：通胀上行，政策收紧")),
    (r"降息", [("f_ff", "模拟：美联储降息", {"mean_annual": -0.6})],
     ("政策利率", "需求", "-", "6-12月", "模拟后端：降息支撑需求")),
    (r"加息", [("f_ff", "模拟：美联储加息", {"mean_annual": 0.6})],
     ("政策利率", "信用利差", "+", "6-12月", "模拟后端：高利率提高违约风险")),
)
VIEW_KEYS = ("mean", "mean_annual", "shift_sd", "min", "max", "vol_mult")


def translator_answer(schema: dict, user: str) -> dict:
    """A spec in SPEC_JSON_SCHEMA's shape (every key present), built from keywords of the narrative."""
    narrative = _tag(user, "narrative") or str(user or "")
    m = re.search(r"翻译成 (\d+) 个月", str(user or ""))
    horizon = int(m.group(1)) if m else 180
    digest = hashlib.sha256(narrative.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    views, chain, tags = [], [], ["模拟"]
    for pattern, items, link in THEMES:
        if re.search(pattern, narrative, re.I):
            for var, why, targets in items:
                if all(v["var"] != var for v in views):
                    views.append({"var": var, **{k: targets.get(k) for k in VIEW_KEYS}, "why": why})
            chain.append(dict(zip(("from", "to", "sign", "lag", "mechanism"), link)))
    first = min(24, horizon)
    no_overrides = {"factor_drift": [], "asset_drift": [], "resid_scale": None, "r_star": None}
    phases = [{"name": "情景期", "months": first, "regime": [], "views": views[:6], "overrides": no_overrides,
               "confidence": 0.8}]
    if horizon > first:
        phases.append({"name": "回归常态", "months": horizon - first, "regime": [], "views": [],
                       "overrides": copy.deepcopy(no_overrides), "confidence": 1.0})
    plain = re.sub(r"\s+", " ", narrative).strip()
    return {"id": f"mock_{digest}", "name": truncate(re.sub(r"[，。！？,.!?].*$", "", plain), 12) or "模拟情景",
            "summary": truncate(plain, 60) or "模拟后端生成的情景", "narrative": plain or "（空）", "tags": tags,
            "prior_prob": 0.05, "analogs": [], "causal_chain": chain, "phases": phases}


def generic_answer(schema: dict) -> object:
    """A minimal instance of a schema (first enum value, empty strings and arrays, zero numbers)."""
    if not isinstance(schema, dict):
        return None
    if "enum" in schema:
        return schema["enum"][0] if schema["enum"] else None
    t = schema.get("type")
    t = t[0] if isinstance(t, list) else t
    if t == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        return {k: generic_answer(props.get(k, {})) for k in schema.get("required", [])}
    return {"array": [], "string": "", "number": 0, "integer": 0, "boolean": False, "null": None}.get(t)


def _renormalise(obj):
    """Every list of {"prob": number} dicts scaled to sum 1 (percentages included): the mock's format repair."""
    if isinstance(obj, dict):
        return {k: _renormalise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [_renormalise(v) for v in obj]
        probs = [x.get("prob") for x in items if isinstance(x, dict)]
        if probs and len(probs) == len(items) and all(isinstance(p, (int, float)) and not isinstance(p, bool)
                                                      and p >= 0 for p in probs) and sum(probs) > 0:
            for x, p in zip(items, _two_decimals(probs)):
                x["prob"] = p
        return items
    return obj


def repair_answer(schema: dict | None, user: str, mode: str = "modal", seed: str = "0") -> object:
    """A P5 repair: the previous output re-parsed with its probabilities renormalised; when it does not parse, a
    fresh answer from the schema alone (a repair call sees no state, so menus come from the schema's enums)."""
    previous = _tag(user, "previous_output").replace("＜", "<").replace("＞", ">")
    obj, _ = extract_json(previous)
    if isinstance(obj, dict):
        return _renormalise(obj)
    kind = call_kind(schema, "")
    builders = {"actor": actor_answer, "node": node_answer}
    if kind in builders:
        return builders[kind](schema, "", mode, seed)
    simple = {"synth": synth_answer, "root": root_answer, "story": story_answer, "probe": probe_answer,
              "translator": translator_answer}
    if kind in simple:
        return simple[kind](schema, "")
    return generic_answer(schema or {"type": "object"})


# ---------------------------------------------------------------- fixtures

def _limit_epoch() -> int:
    return int(time.time()) + 3 * 3600


def _cli_failure(model: str, envelope: dict | None, returncode: int = 1, stderr: str = "", stdout: str = ""):
    env = None if envelope is None else {"type": "result", "subtype": "success", "result": None, **envelope}
    return cc.failure_error(env, returncode, stderr, model, False, stdout=stdout)


ERRORS = {
    "refusal": lambda m, t: cc.ClaudeCodeError(cc.E_REFUSAL, "refusal"),
    "timeout": lambda m, t: cc.ClaudeCodeError(cc.E_TIMEOUT.format(seconds=t), "timeout"),
    "usage_limit": lambda m, t: _cli_failure(m, {"is_error": True, "api_error_status": 429,
                                                 "result": f"Claude AI usage limit reached|{_limit_epoch()}"}),
    "usage_limit_text": lambda m, t: _cli_failure(m, {"is_error": True, "result":
                                                      "5-hour limit reached ∙ resets 3pm (America/Los_Angeles)"}),
    "weekly_limit": lambda m, t: _cli_failure(m, {"is_error": True, "result": "You've hit your weekly limit · resets Mon 9am"}),
    "login": lambda m, t: _cli_failure(m, {"is_error": True, "api_error_status": 401,
                                           "result": "Failed to authenticate. API Error: 401 OAuth access token is invalid."}),
    "rate_limit": lambda m, t: _cli_failure(m, {"is_error": True, "api_error_status": 429,
                                                "result": "API Error: 429 rate_limit_error"}),
    "overloaded": lambda m, t: _cli_failure(m, {"is_error": True, "api_error_status": 529,
                                                "result": "API Error: 529 overloaded_error"}),
    "server_error": lambda m, t: _cli_failure(m, {"is_error": True, "api_error_status": 500, "result": "API Error: 500"}),
    "model_missing": lambda m, t: _cli_failure(m, {"is_error": True, "api_error_status": 404,
                                                   "result": f"There's an issue with the selected model ({m})."}),
    "max_tokens": lambda m, t: cc.ClaudeCodeError(cc.E_MAX_TOKENS, "max_tokens"),
    "context": lambda m, t: cc.ClaudeCodeError(cc.E_CONTEXT, "context"),
    "empty": lambda m, t: cc.ClaudeCodeError(cc.E_EMPTY, "empty"),
    "cli_missing": lambda m, t: cc.ClaudeCodeError(cc.INSTALL_HINT, "cli_missing"),
    "crash": lambda m, t: _cli_failure(m, None, 1, "Segmentation fault", "Error: not json"),
}


def _first_actor_block(answer: dict) -> dict | None:
    if isinstance(answer.get("actors"), dict) and answer["actors"]:
        return next(iter(answer["actors"].values()))
    if "options" in answer:
        return answer
    return None


def _prob_lists(answer: dict) -> list[list]:
    out = []
    if "options" in answer:
        out.append(answer["options"])
    for block in (answer.get("actors") or {}).values():
        out.append(block.get("options") or [])
    for c in answer.get("clusters") or []:
        out.append(c.get("branches") or [])
    return out


def _shift_fields(schema: dict | None) -> list:
    """The mechanism fields a synthesizer answer may shift (from its schema)."""
    try:
        br = schema["properties"]["clusters"]["items"]["properties"]["branches"]["items"]
        return list(br["properties"]["shifts"]["properties"])
    except (KeyError, TypeError):
        return []


def _mutate(name: str, answer: dict, schema: dict | None = None) -> dict:
    a = copy.deepcopy(answer)
    block = _first_actor_block(a)
    if name == "synth_mechanism":
        # a mechanism branch next to the partitions (the UI's 机制 rows; not the mock's default behaviour)
        fields = [f for f in _shift_fields(schema) if f in ("d_sent", "d_cons_serv", "d_capex", "d_unrate")] or \
            _shift_fields(schema)
        for c in a.get("clusters") or []:
            if c.get("status") != "ok" or not c.get("branches") or not fields:
                continue
            cen = next((b for b in c["branches"] if b.get("kind") == "central"), c["branches"][0])
            take = min(0.12, max(0.0, float(cen.get("prob") or 0) - 0.05))
            if take <= 0:
                continue
            cen["prob"] = round(float(cen["prob"]) - take, 2)
            c["branches"] = c["branches"][:3] + [{
                "id": "mech1", "label": "信心冲击", "kind": "mechanism", "prob": round(take, 2), "region": None,
                "shifts": {fields[0]: {"mean": -8.0, "sd": 1.0}}, "cause": {"options": [], "state": ["模拟后端"], "facts": []},
                "mechanism": "模拟后端：模型看不到的信心通道（机制增量）", "anchors": [], "unintended": True,
                "beyond_history": None, "facts": [], "memory": "模拟后端：信心受到冲击", "vs_prior": None}]
        return a
    if name == "percent_probs":
        for lst in _prob_lists(a):
            for o in lst:
                if isinstance(o.get("prob"), (int, float)):
                    o["prob"] = round(o["prob"] * 100)
    elif name == "bad_sum":
        lst = next((x for x in _prob_lists(a) if x), None)
        if lst:
            lst[0]["prob"] = 0.10 if lst[0].get("prob", 0) > 0.2 else 0.9
    elif name == "price_sentence":
        if block and block.get("options"):
            block["options"][0]["rationale"] = PRICE_SENTENCE
        for c in a.get("clusters") or []:
            if c.get("branches"):
                c["branches"][0]["memory"] = PRICE_SENTENCE
                break
    elif name == "out_of_power":
        if block and block.get("options"):
            block["options"][0]["writes"] = {"d_ff": -0.5}
    elif name == "decline":
        a["status"] = "decline"
        a["decline_reason"] = "模拟后端：拒绝"
    elif name == "decline_actor":
        if block is not None:
            block.update(status="decline", decline_reason="模拟后端：该行为者拒绝评估", options=[])
        for c in (a.get("clusters") or [])[:1]:
            c.update(status="decline", decline_reason="模拟后端：拒绝分析", branches=[])
    elif name == "missing_actor":
        if isinstance(a.get("actors"), dict) and a["actors"]:
            a["actors"].pop(next(iter(a["actors"])))
    return a


TEXT_FIXTURES = {"malformed_json", "fenced", "trailing_comma", "fullwidth", "refusal_text"}
ANSWER_FIXTURES = {"percent_probs", "bad_sum", "price_sentence", "out_of_power", "decline", "decline_actor",
                   "missing_actor", "synth_mechanism"}
FIXTURES = tuple(sorted(TEXT_FIXTURES | ANSWER_FIXTURES | set(ERRORS) | {"hang"}))


def _text_fixture(name: str, text: str) -> str:
    if name == "malformed_json":
        return text[:max(10, int(len(text) * 0.6))]
    if name == "fenced":
        return f"好的，以下是结果：\n```json\n{text}\n```\n以上。"
    if name == "trailing_comma":
        k = text.rfind("}")
        return text[:k] + ", " + text[k:] if k > 0 else text
    if name == "fullwidth":
        return text.replace('"status":', "“status”：", 1)
    if name == "refusal_text":
        return "I can't help with role-playing a real head of state."
    return text


# ---------------------------------------------------------------- the backend

class MockClaude:
    """A deterministic, schema-aware fake of claude_code.run. Thread-safe; one process-wide instance (MOCK)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queue: list[Fixture] = []
        self.calls: deque = deque(maxlen=CALL_LOG)

    # fixtures
    def push(self, name: str, match=None, times: int | None = 1) -> None:
        if name not in FIXTURES and not callable(name):
            raise ValueError(f"unknown mock fixture {name!r}; known: {', '.join(FIXTURES)}")
        with self._lock:
            self._queue.append(Fixture(name, match, times))

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self.calls.clear()

    @staticmethod
    def _matches(match, kind: str, system: str, user: str) -> bool:
        if match is None:
            return True
        if callable(match):
            return bool(match(kind, system, user))
        m = str(match)
        if m == kind:
            return True
        if ":" in m:
            k, _, detail = m.partition(":")
            return k == kind and detail in user
        return m in user

    def _take(self, kind: str, system: str, user: str) -> str | None:
        with self._lock:
            for f in self._queue:
                if self._matches(f.match, kind, system, user):
                    if f.times is not None:
                        f.times -= 1
                        if f.times <= 0:
                            self._queue.remove(f)
                    return f.name
        for item in filter(None, (x.strip() for x in _env("MSL_LLM_MOCK_FIXTURE").split(","))):
            name, _, match = item.partition("@")
            if name in FIXTURES and self._matches(match or None, kind, system, user):
                return name
        return None

    # answers
    def answer(self, system: str, user: str) -> tuple[str, object]:
        """(call kind, the derived answer object) for a prompt (no fixture applied)."""
        schema = schema_from_system(system)
        kind = call_kind(schema, user)
        mode = _env("MSL_LLM_MOCK_MODE", "modal").lower()
        mode = mode if mode in MODES else "modal"
        seed = _env("MSL_LLM_MOCK_SEED", "0")
        if kind == "repair":
            return kind, repair_answer(schema, user, mode, seed)
        if schema is None:
            return kind, {}
        builders = {"actor": lambda: actor_answer(schema, user, mode, seed),
                    "node": lambda: node_answer(schema, user, mode, seed),
                    "synth": lambda: synth_answer(schema, user), "root": lambda: root_answer(schema, user),
                    "story": lambda: story_answer(schema, user), "probe": lambda: probe_answer(schema, user),
                    "translator": lambda: translator_answer(schema, user)}
        return kind, builders.get(kind, lambda: generic_answer(schema))()

    @staticmethod
    def _cassette(model: str, system: str, user: str, effort, safe_mode) -> tuple[str | None, dict | None]:
        root = _env("MSL_LLM_MOCK_CASSETTES")
        if not root:
            return None, None
        path = Path(root) / f"{cc.request_key(model, system, user, effort, safe_mode)}.json"
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, None
        if isinstance(rec.get("error"), dict):
            return None, rec["error"]
        return (rec["text"], None) if isinstance(rec.get("text"), str) else (None, None)

    @staticmethod
    def _wait(seconds: float, timeout: float, cancel) -> None:
        seconds = min(max(seconds, 0.0), MAX_DELAY_S)
        if seconds <= 0:
            return
        limit = min(seconds, timeout)
        stopped = cancel.wait(limit) if cancel is not None else (time.sleep(limit) or False)
        if stopped:
            raise cc.ClaudeCodeError(cc.E_CANCELLED, "cancelled")
        if seconds >= timeout:
            raise cc.ClaudeCodeError(cc.E_TIMEOUT.format(seconds=timeout), "timeout")

    def run(self, system: str, user: str, model: str, timeout: float = cc.DEFAULT_TIMEOUT_S, *,
            effort: str | None = None, safe_mode: bool = False, cancel: threading.Event | None = None
            ) -> tuple[str, dict]:
        """claude_code.run's contract: (text, metadata) or ClaudeCodeError (with a kind)."""
        if cancel is not None and cancel.is_set():
            raise cc.ClaudeCodeError(cc.E_CANCELLED, "cancelled")
        t0 = time.perf_counter()
        kind, answer = self.answer(system, user)
        fixture = self._take(kind, system, user)
        if callable(fixture):                     # a custom fixture: (kind, system, user, answer) -> text, or raises
            custom, fixture = fixture, getattr(fixture, "__name__", "custom")
        else:
            custom = None
        record = {"kind": kind, "fixture": fixture, "model": model, "effort": effort, "safe_mode": safe_mode,
                  "chars": len(system) + len(user), "user_sha": hashlib.sha256(user.encode("utf-8", "surrogatepass"))
                  .hexdigest()[:12]}
        with self._lock:
            self.calls.append(record)
        self._wait(float(_env("MSL_LLM_MOCK_DELAY_S", "0") or 0), timeout, cancel)
        text, error = self._cassette(model, system, user, effort, safe_mode)
        if error is not None:
            raise cc.ClaudeCodeError(str(error.get("message") or "录制的失败"), str(error.get("kind") or "failed"))
        if text is None and custom is not None:
            text = str(custom(kind, system, user, copy.deepcopy(answer)))
        if text is None:
            if fixture == "hang":
                stopped = cancel.wait(min(timeout, HANG_MAX_S)) if cancel is not None else False
                if stopped:
                    raise cc.ClaudeCodeError(cc.E_CANCELLED, "cancelled")
                raise cc.ClaudeCodeError(cc.E_TIMEOUT.format(seconds=timeout), "timeout")
            if fixture in ERRORS:
                raise ERRORS[fixture](model, timeout)
            if fixture in ANSWER_FIXTURES and isinstance(answer, dict):
                answer = _mutate(fixture, answer, schema_from_system(system) if fixture == "synth_mechanism" else None)
            text = json.dumps(answer, ensure_ascii=False)
            if fixture in TEXT_FIXTURES:
                text = _text_fixture(fixture, text)
        wall = time.perf_counter() - t0
        usage = {"input_tokens": _tokens(system) + _tokens(user), "output_tokens": _tokens(text),
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        meta = {"backend": "mock", "model": model, "models": [model], "duration_ms": int(wall * 1000),
                "duration_api_ms": 0, "num_turns": 1, "stop_reason": "end_turn", "usage": usage,
                "usage_by_model": {model: {"inputTokens": usage["input_tokens"], "outputTokens": usage["output_tokens"],
                                           "costUSD": 0.0}},
                "cost_usd": 0.0, "wall_s": round(wall, 4), "effort": effort, "safe_mode": bool(safe_mode),
                "mock": {"kind": kind, "fixture": fixture, "version": MOCK_VERSION}}
        return text, meta

    def status(self, model: str) -> dict:
        return {"backend": "mock", "label": LABEL, "available": True, "model": model, "binary": None,
                "version": MOCK_VERSION, "auth": "模拟（不调用模型）",
                "reason": "MSL_LLM_BACKEND=mock：所有调用由本地模拟后端回答，不调用模型、不消耗订阅额度。"}


MOCK = MockClaude()


def run(system: str, user: str, model: str, timeout: float = cc.DEFAULT_TIMEOUT_S, **kwargs) -> tuple[str, dict]:
    return MOCK.run(system, user, model, timeout, **kwargs)


def status(model: str) -> dict:
    return MOCK.status(model)


def push(name: str, match=None, times: int | None = 1) -> None:
    MOCK.push(name, match, times)


def reset() -> None:
    MOCK.reset()


def calls() -> list[dict]:
    with MOCK._lock:
        return list(MOCK.calls)
