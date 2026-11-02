"""Estimates and the translation phase of the narrative test (SIM_DESIGN §8.5). It never imports the key.

A run has two phases. This module does the first: it plans the calls, estimates them, and translates. The second
(simulation and scoring, score.py) needs the sealed key and runs only after the translations are on disk.

Plan (plan()). For each selected episode, in an order shuffled with the run's seed: N3 first (§8.4), then N1, N2 and
N-wrong (the registry's pre-registered partner's N1, translated with THIS episode's prompt) in a shuffled order, then
the recognition probes (on N1 and N3). Synthetic episodes get one translation of their n2 text (N-synth) and no probe.
A narrative that is not written (v2 before the writer round; v1 of the six new episodes) is listed as missing.
On the mock backend the research round's stand-in tables replace the v1 translations (no call: QTable/1 legacy,
labelled 研究轮代译（无引文）) and a synthetic episode's reference translation is registered as the mock's replay.

Estimate (estimate()). Calls, the ones answered from data/llm_cache/ (free), the stand-ins, input tokens (the design
docs' convention), output tokens (§8.5 assumptions: 3–8k per translation, 0.1–0.4k per probe), minutes at the
translator's concurrency, the backend's billing, and whether a confirmation is needed. It is saved with an id
(data/results/narrative/estimates/<id>.json); a run needs an estimate issued in the last ESTIMATE_FRESH_S seconds
whose plan has not changed since (fresh_estimate()).

Translation (translate_run()). Every call is separate (no memory), through llm/qtranslator.translate in test mode
with the registry's blind state rows; probes through the translator's cached call path. The run manifest records
every call's prompt and narrative hashes (and a format repair's message hash), and the texts sent are kept under
sent/<sha256>.txt (every byte sent: the system prompt, the narrative message and the repair message). N3 is
translated with numbers=False (§8.4: market prices and surveys only as size bins, qtable R-N3).
A usage limit, a missing login or CLI pauses the run (status paused, with reset_at when the CLI said); a paused run
resumes by its id and skips what is done (cached answers are free). Under pytest only the mock backend may run.

    data/results/narrative/<run_id>/run.json              status, settings, estimate, progress, pause, error
    data/results/narrative/<run_id>/manifest.json         backend, model, prompt version, every call's hashes
    data/results/narrative/<run_id>/translations/<call>.json   clean QTable, validation report, status, call metadata
    data/results/narrative/<run_id>/probes/<call>.json    the probe's answer (scored by audit.py, never fed back)
    data/results/narrative/<run_id>/sent/<sha256>.txt     every system prompt and user message sent
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from ...llm import claude_code
from ...llm import mock as llm_mock
from ...llm import qtranslator as QX
from ...llm import translator as tr
from ...qmodel import qtable as QT
from . import common as C
from . import episodes as EP
from . import prompts as PR

RUN_PREFIX = "n_"
TERMINAL = ("done", "failed", "cancelled")
DEFAULT_SETTINGS = {"episodes": "all", "narratives": "v2", "arms": list(C.DEFAULT_RUN_ARMS), "probe": True,
                    "engines": ["Q", "E"], "seed": C.SEED, "refresh": False}
E_API = ("季度情景表的翻译只用本机 Claude Code（MSL_LLM_BACKEND=claude_code）或模拟后端：不走按 token 计费的 Anthropic API"
         "（USER_DECISIONS）。")
E_BACKEND = "服务器配置的后端是 {have}（MSL_LLM_BACKEND）：估算与运行只能用它，收到「{want}」。"
E_PYTEST = "测试里只能用模拟后端（MSL_LLM_BACKEND=mock）：叙事检验在测试中永远不调用真实大模型。"
BILLING = {"mock": "模拟后端：不调用模型，不消耗订阅额度", "claude_code": "本机 Claude Code：计入你的订阅额度（不按 token 计费）"}


class RunnerError(ValueError):
    """A request the runner refuses (Chinese message)."""


class EstimateError(RunnerError):
    """No usable estimate: unknown, stale (older than ESTIMATE_FRESH_S) or for another plan."""


# ================================================================ settings and the plan

def backend_name() -> str:
    return tr.backend_name()


def _check_backend(want: str | None) -> str:
    have = backend_name()
    want = (str(want).strip().lower().replace("-", "_") if want else None) or have
    if want == "api" or have == "api":
        raise RunnerError(E_API)
    if want != have:
        raise RunnerError(E_BACKEND.format(have=have, want=want))
    if have not in QX.BACKENDS:
        raise RunnerError(tr.status()["reason"])
    return have


def normalize(raw: dict | None) -> dict:
    """The run settings with defaults; episodes become registry codes (a spec code needs key.resolve first)."""
    raw = dict(raw or {})
    s = {**DEFAULT_SETTINGS, **{k: v for k, v in raw.items() if v is not None}}
    s["episodes"] = EP.select(s["episodes"])
    if s["narratives"] not in ("v1", "v2"):
        raise RunnerError(f"narratives 只能是 v1 或 v2（收到「{s['narratives']}」）")
    arms = s["arms"].split(",") if isinstance(s["arms"], str) else list(s["arms"])
    alias = {"wrong": "NW", "synth": "NS", "n1": "N1", "n2": "N2", "n3": "N3"}
    arms = [alias.get(a.strip(), a.strip()) for a in arms if str(a).strip()]
    bad = [a for a in arms if a not in C.TRANSLATED_ARMS]
    if bad:
        raise RunnerError(f"不认识的叙事臂：{'、'.join(bad)}（可选 N1、N2、N3、wrong、synth）")
    s["arms"] = arms
    engines = s["engines"].split(",") if isinstance(s["engines"], str) else list(s["engines"])
    if not engines or any(e not in C.ENGINES for e in engines):
        raise RunnerError("engines 只能是 Q（季度模型）和 / 或 E（月度引擎）")
    s["engines"] = engines
    s["probe"] = bool(s["probe"])
    s["refresh"] = s.get("refresh") is True
    s["seed"] = int(s["seed"])
    s["backend"] = _check_backend(raw.get("backend"))
    s["model"] = str(raw.get("model") or tr.model_id())
    return s


def _call(kind: str, code: str, arm: str | None, src: tuple, prompt_code: str) -> dict:
    cid = f"{code}__{arm}" if kind != "probe" else f"{code}__probe_{src[2]}"
    return {"id": cid, "kind": kind, "code": code, "arm": arm,
            "source": {"code": src[0], "version": src[1], "n": src[2]}, "prompt_code": prompt_code}


def plan(settings: dict) -> dict:
    """{"calls": [...], "missing": [...]} for normalized settings (see the module docstring)."""
    rng = np.random.default_rng([int(settings["seed"]), 31])
    version = settings["narratives"]
    arms = settings["arms"]
    episodes = [c for c in settings["episodes"] if EP.entry(c)["set"] != "synth"]
    synth = [c for c in settings["episodes"] if EP.entry(c)["set"] == "synth"]
    if "NS" in arms and not synth:
        synth = EP.codes("synth")
    calls, missing = [], []
    order = [episodes[i] for i in rng.permutation(len(episodes))] if episodes else []
    for code in order:
        mine = []
        for arm in ("N1", "N2", "N3"):
            if arm not in arms:
                continue
            n = C.ARMS[arm]["narrative"]
            if EP.has_narrative(code, version, n):
                mine.append(_call("translate", code, arm, (code, version, n), code))
            else:
                missing.append({"code": code, "arm": arm, "version": version, "text": _missing_text(code, version)})
        if "NW" in arms and EP.entry(code)["set"] == "clean":
            partner = EP.pair(code)
            if partner and EP.has_narrative(partner, version, "n1"):
                mine.append(_call("translate", code, "NW", (partner, version, "n1"), code))
            else:
                missing.append({"code": code, "arm": "NW", "version": version,
                                "text": f"配对片段 {partner or '（没有）'} 的 N1 没有写"})
        first = [c for c in mine if c["arm"] == "N3"]
        rest = [c for c in mine if c["arm"] != "N3"]
        rest = [rest[i] for i in rng.permutation(len(rest))] if rest else []
        calls += first + rest
        if settings["probe"]:
            for n in ("n1", "n3"):
                if EP.has_narrative(code, version, n):
                    calls.append(_call("probe", code, None, (code, version, n), code))
    if "NS" in arms:
        for code in synth:
            calls.append(_call("translate", code, "NS", (code, "synth", C.ARMS["NS"]["narrative"]), code))
    backend = settings["backend"]
    for c in calls:
        c["standin"] = bool(backend == "mock" and c["kind"] == "translate" and c["source"]["version"] == "v1"
                            and EP.standin(c["source"]["code"], c["source"]["n"]) is not None)
        c["reference"] = bool(c["kind"] == "translate" and c["source"]["version"] == "synth")
    return {"calls": calls, "missing": missing}


def _missing_text(code: str, version: str) -> str:
    return C.LABELS["no_v2"] if version == "v2" else "研究轮没有这个片段的叙事（新增片段，等 v2 写作）"


def call_texts(call: dict) -> tuple[str, str, str]:
    """(system, user, narrative) of a planned call."""
    src = call["source"]
    text = EP.narrative(src["code"], src["version"], src["n"])
    if text is None:
        raise RunnerError(f"叙事不存在：{src['code']}/{src['version']}/{src['n']}")
    if call["kind"] == "probe":
        return PR.probe_system(), PR.probe_user(text), text
    return PR.translation_system(call["prompt_code"]), PR.translation_user(text), text


def plan_digest(settings: dict, calls: list) -> str:
    """A hash of what the run would send (prompts and narratives): an estimate is stale once it changes."""
    parts = []
    for c in calls:
        sysm, user, _ = call_texts(c)
        parts.append([c["id"], PR.sha256(sysm), PR.sha256(user), c["standin"]])
    blob = json.dumps({"calls": parts, "backend": settings["backend"], "model": settings["model"],
                       "prompt": QX.PROMPT_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ================================================================ the estimate

def _cost(s: dict, calls: list) -> dict:
    """Calls, cache hits, stand-ins, tokens and minutes of these planned calls (no model call)."""
    backend, model = s["backend"], s["model"]
    n_tr = n_pr = cached = standin = new_tr = new_pr = 0
    tokens_in = 0
    for c in calls:
        if c["kind"] == "probe":
            n_pr += 1
        else:
            n_tr += 1
        if c["standin"]:
            standin += 1
            continue
        sysm, user, _ = call_texts(c)
        if backend == "mock":
            continue
        if not s["refresh"] and QX.is_cached(sysm, user, model, backend):
            cached += 1
            continue
        if c["kind"] == "probe":
            new_pr += 1
        else:
            new_tr += 1
        tokens_in += QX.tokens(sysm) + QX.tokens(user)
    new = new_tr + new_pr
    out_lo = new_tr * C.TRANSLATION_OUTPUT_TOKENS[0] + new_pr * C.PROBE_OUTPUT_TOKENS[0]
    out_hi = new_tr * C.TRANSLATION_OUTPUT_TOKENS[1] + new_pr * C.PROBE_OUTPUT_TOKENS[1]
    conc = max(1, tr.concurrency())
    minutes = [round(new * C.CALL_SECONDS[0] / conc / 60.0, 1), round(new * C.CALL_SECONDS[1] / conc / 60.0, 1)]
    return {"calls": {"translations": n_tr, "probes": n_pr, "total": len(calls), "new": new, "cached": cached,
                      "standin": standin, "new_translations": new_tr, "new_probes": new_pr,
                      "max_with_repairs": new + new_tr},
            "input_tokens": tokens_in, "output_tokens": [out_lo, out_hi], "minutes": minutes,
            "concurrency": conc, "real_calls": backend != "mock" and new > 0}


def _save_estimate(est: dict) -> None:
    C.estimates_dir().mkdir(parents=True, exist_ok=True)
    (C.estimates_dir() / f"{est['id']}.json").write_text(json.dumps(est, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")


def estimate(raw: dict | None = None, *, save: bool = True) -> dict:
    """What a run would cost, without any model call (see the module docstring); saved with an id."""
    s = normalize(raw)
    p = plan(s)
    calls = p["calls"]
    backend, model = s["backend"], s["model"]
    now = time.time()
    est = {"id": uuid.uuid4().hex[:12], "created": now, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "fresh_s": C.ESTIMATE_FRESH_S, "settings": {k: v for k, v in s.items()},
           "backend": backend, "backend_label": BILLING.get(backend, backend), "model": model,
           "billing": BILLING.get(backend, backend), **_cost(s, calls), "needs_confirmation": True,
           "missing": p["missing"], "n_missing": len(p["missing"]),
           "plan_digest": plan_digest(s, calls),
           "notes": ["每次翻译约 2–2.5 万输入 token、3–8 千输出 token、20–60 秒：这些是设计文档的假设，第一次真实运行后用实测值替换"]}
    if backend == "mock":
        est["notes"].append(C.LABELS["mock"])
    if s["narratives"] == "v2" and p["missing"]:
        est["notes"].append(C.LABELS["no_v2"] + "：可以先用 v1（研究轮的叙事，不是盲测规范 v2）")
    if save:
        _save_estimate(est)
    return est


def remaining(rid: str) -> tuple[dict, list, list]:
    """(the run's settings, its planned calls that are not done yet, all its planned calls)."""
    run = read_run(rid)
    s = normalize(run["settings"])
    calls = json.loads((run_dir(rid) / "plan.json").read_text(encoding="utf-8"))["calls"]
    return s, [c for c in calls if not _done_path(rid, c).exists()], calls


def estimate_resume(rid: str, *, save: bool = True) -> dict:
    """What resuming a run would cost: its OWN remaining calls (not done yet), with the cache as it is now. The
    estimate names the run (run_id), so /resume accepts it for that run only, and its plan digest covers exactly the
    remaining calls, so it goes stale when anything is done or changes in between."""
    s, todo, calls = remaining(rid)
    now = time.time()
    est = {"id": uuid.uuid4().hex[:12], "created": now, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "fresh_s": C.ESTIMATE_FRESH_S, "run_id": rid, "settings": {k: v for k, v in s.items()},
           "backend": s["backend"], "backend_label": BILLING.get(s["backend"], s["backend"]), "model": s["model"],
           "billing": BILLING.get(s["backend"], s["backend"]), **_cost(s, todo), "needs_confirmation": True,
           "remaining": len(todo), "done": len(calls) - len(todo), "missing": [], "n_missing": 0,
           "plan_digest": plan_digest(s, todo),
           "notes": [f"继续运行 {rid}：还剩 {len(todo)} 个调用（已完成 {len(calls) - len(todo)} 个，不再调用）"]}
    if s["backend"] == "mock":
        est["notes"].append(C.LABELS["mock"])
    if save:
        _save_estimate(est)
    return est


def load_estimate(estimate_id: str) -> dict:
    if not isinstance(estimate_id, str) or not (len(estimate_id) == 12 and all(c in "0123456789abcdef"
                                                                               for c in estimate_id)):
        raise EstimateError(f"估算编号格式不对：「{estimate_id}」")
    p = C.estimates_dir() / f"{estimate_id}.json"
    if not p.exists():
        raise EstimateError(f"没有找到估算「{estimate_id}」：请先估算（estimate）")
    return json.loads(p.read_text(encoding="utf-8"))


def fresh_estimate(estimate_id: str, *, now: float | None = None, max_age_s: float = C.ESTIMATE_FRESH_S) -> dict:
    """The estimate if it was issued in the last max_age_s seconds and its plan is unchanged; else EstimateError."""
    est = load_estimate(estimate_id)
    age = (now if now is not None else time.time()) - float(est.get("created", 0))
    if age > max_age_s or age < -60:
        raise EstimateError(f"估算「{estimate_id}」已经过期（{age / 60:.0f} 分钟前；只接受 {max_age_s / 60:.0f} 分钟内的估算）："
                            "请重新估算再确认")
    if est.get("run_id"):                       # a resume estimate: the run's remaining calls, now
        try:
            s, todo, _ = remaining(est["run_id"])
        except (KeyError, OSError, ValueError) as err:
            raise EstimateError(f"估算「{estimate_id}」对应的运行不在了：{err}") from err
        if plan_digest(s, todo) != est["plan_digest"]:
            raise EstimateError("估算之后这次运行剩下的调用变了：请重新估算再确认")
        return est
    s = normalize(est["settings"])
    if plan_digest(s, plan(s)["calls"]) != est["plan_digest"]:
        raise EstimateError("估算之后叙事、提示词或缓存的范围变了：请重新估算再确认")
    return est


def live_cost(est: dict) -> dict:
    """The estimate's calls priced with the cache as it is NOW (a CLI confirmation must not trust an old count):
    a resume estimate's remaining calls, or a run estimate's whole plan."""
    if est.get("run_id"):
        s, todo, _ = remaining(est["run_id"])
        return _cost(s, todo)
    s = normalize(est["settings"])
    return _cost(s, plan(s)["calls"])


# ================================================================ runs on disk

def run_dir(run_id: str) -> Path:
    if not isinstance(run_id, str) or not run_id or "/" in run_id or ".." in run_id \
            or not run_id.replace("_", "").replace("-", "").isalnum():
        raise RunnerError(f"运行编号不合法：「{run_id}」")
    return C.runs_dir() / run_id


def read_run(run_id: str) -> dict:
    p = run_dir(run_id) / "run.json"
    if not p.exists():
        raise KeyError(f"没有找到叙事检验的运行「{run_id}」")
    return json.loads(p.read_text(encoding="utf-8"))


_WRITE_LOCK = threading.Lock()


def _atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_run(run_id: str, **upd) -> dict:
    with _WRITE_LOCK:
        p = run_dir(run_id) / "run.json"
        cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        cur.update(upd, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        _atomic(p, cur)
        return cur


def new_run_id() -> str:
    return f"{RUN_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def create_run(est: dict, *, label: str | None = None) -> str:
    """A new run directory from an estimate (the caller has checked it is fresh and confirmed)."""
    s = normalize(est["settings"])
    p = plan(s)
    rid = new_run_id()
    kind = "mock" if s["backend"] == "mock" else "real"
    reg = EP.load_registry()
    write_run(rid, id=rid, status="queued", kind=kind, label=label or (C.LABELS["mock"] if kind == "mock" else None),
              backend=s["backend"], model=s["model"], created=time.strftime("%Y-%m-%dT%H:%M:%S"), settings=s,
              estimate_id=est["id"], estimate={k: est[k] for k in ("calls", "input_tokens", "output_tokens", "minutes",
                                                                   "billing")},
              registry={"sealed_at": reg.get("sealed_at"), "key_sha256": reg.get("key_sha256"),
                        "panel": reg.get("panel")},
              progress={"phase": "translate", "done": 0, "total": len(p["calls"])}, pause=None, error=None)
    _atomic(run_dir(rid) / "plan.json", p)
    _atomic(run_dir(rid) / "manifest.json", {
        "run_id": rid, "backend": s["backend"], "model": s["model"], "effort": QX.EFFORT,
        "prompt_version": QX.PROMPT_VERSION, "mode": "test", "estimate_id": est["id"],
        "registry": {"sealed_at": reg.get("sealed_at"), "key_sha256": reg.get("key_sha256")},
        "isolation": ("claude_code 后端以 --tools \"\" 在空的临时目录里运行：翻译者只收到系统提示词和叙事，"
                      "看不到仓库、文件或工具") if s["backend"] == "claude_code" else "模拟后端：没有发出任何内容",
        "calls": {}})
    return rid


def _done_path(rid: str, call: dict) -> Path:
    sub = "probes" if call["kind"] == "probe" else "translations"
    return run_dir(rid) / sub / f"{call['id']}.json"


def _record_sent(rid: str, text: str) -> str:
    h = PR.sha256(text)
    p = run_dir(rid) / "sent" / f"{h}.txt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return h


def _manifest_call(rid: str, call: dict, entry: dict) -> None:
    with _WRITE_LOCK:
        p = run_dir(rid) / "manifest.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["calls"][call["id"]] = entry
        _atomic(p, m)


# ================================================================ one call

def _translate(rid: str, call: dict, s: dict, cancel) -> dict:
    sysm, user, text = call_texts(call)
    src = call["source"]
    base = {"id": call["id"], "kind": "translate", "code": call["code"], "arm": call["arm"], "source": src,
            "prompt_code": call["prompt_code"], "mode": "test", "backend": s["backend"], "model": s["model"],
            "narrative_sha256": PR.sha256(text), **PR.manifest_entry(sysm, user)}
    if call.get("standin"):
        rec = EP.standin(src["code"], src["n"])
        report = QT.validate(rec["qtable"], text, mode="test", source="legacy")
        qt = report.pop("qtable")
        return {**base, "kind": "standin", "status": "standin", "label": C.LABELS["legacy"], "source_kind": "legacy",
                "qtable": qt, "validation": report, "conversion": rec.get("conversion"), "calls": 0, "cached": False,
                "usage": {}, "elapsed_s": 0.0, "sent": False}
    if call.get("reference") and s["backend"] == "mock":
        ref = EP.reference(src["code"], src["n"])
        if ref is not None:
            llm_mock.register_qtable(text, ref)
    res = QX.translate(text, mode="test", backend=s["backend"], model=s["model"], refresh=s["refresh"],
                       cancel=cancel, state_rows=EP.state_rows(call["prompt_code"]), numbers=call["arm"] != "N3")
    sent = s["backend"] != "mock"
    repair = dict(res.get("repair") or {}) or None
    if repair is not None:
        r_user = repair.pop("user", None)
        if sent and isinstance(r_user, str):
            repair["user_sha256"] = _record_sent(rid, r_user)     # the format repair is sent too (every byte sent)
            base["repair_user_sha256"] = repair["user_sha256"]
            base["repair_user_chars"] = len(r_user)
    if sent:
        msgs = res.get("messages") or {}
        sys_sent, user_sent = msgs.get("system") or sysm, msgs.get("user") or user
        base["system_sha256"] = _record_sent(rid, sys_sent)          # exactly the bytes sent, not the plan's copy
        base["user_sha256"] = _record_sent(rid, user_sent)
        base.update(system_chars=len(sys_sent), user_chars=len(user_sent))
    return {**base, "status": res["status"], "label": None, "source_kind": "translator", "qtable": res["qtable"],
            "validation": res["validation"], "raw": res.get("raw"), "repair": repair,
            "error": res.get("error"), "calls": res.get("calls", 0), "cached": res.get("cached", False),
            "usage": res.get("usage") or {}, "elapsed_s": res.get("elapsed_s"), "prompt": res.get("prompt"),
            "sent": sent}


def _probe(rid: str, call: dict, s: dict, cancel) -> dict:
    sysm, user, text = call_texts(call)
    src = call["source"]
    base = {"id": call["id"], "kind": "probe", "code": call["code"], "source": src, "backend": s["backend"],
            "model": s["model"], "narrative_sha256": PR.sha256(text), **PR.manifest_entry(sysm, user)}
    t0 = time.perf_counter()
    try:
        answer, meta = QX.complete_text(sysm, user, s["model"], s["backend"], refresh=s["refresh"], cancel=cancel)
    except claude_code.ClaudeCodeError as err:
        if err.kind in QX.PAUSE_KINDS or err.kind == "cancelled":
            raise QX.QTablePause(err.kind, str(err), err.reset_at) from err
        return {**base, "status": "error", "answer": PR.read_probe(None), "error": {"kind": err.kind,
                                                                                    "message": str(err)},
                "elapsed_s": round(time.perf_counter() - t0, 3), "sent": s["backend"] != "mock"}
    except tr.TranslatorError as err:
        return {**base, "status": "error", "answer": PR.read_probe(None), "error": {"kind": "rate", "message": str(err)},
                "elapsed_s": round(time.perf_counter() - t0, 3), "sent": False}
    if s["backend"] != "mock":
        _record_sent(rid, sysm)
        _record_sent(rid, user)
    return {**base, "status": "ok", "answer": PR.read_probe(answer), "raw_text": str(answer)[:2000],
            "cached": bool(meta.get("cached")), "usage": meta.get("usage") or {},
            "elapsed_s": round(time.perf_counter() - t0, 3), "sent": s["backend"] != "mock"}


def run_one(rid: str, call: dict, s: dict, cancel=None) -> dict:
    """Execute one planned call and store its record (QTablePause propagates: the run pauses)."""
    if os.environ.get("PYTEST_CURRENT_TEST") and s["backend"] != "mock":
        raise RunnerError(E_PYTEST)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec = _probe(rid, call, s, cancel) if call["kind"] == "probe" else _translate(rid, call, s, cancel)
    rec["started"] = started
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic(_done_path(rid, call), rec)
    _manifest_call(rid, call, {k: rec.get(k) for k in ("kind", "code", "arm", "source", "system_sha256", "user_sha256",
                                                       "repair_user_sha256", "repair_user_chars",
                                                       "narrative_sha256", "system_chars", "user_chars",
                                                       "input_tokens", "status", "cached", "calls", "usage", "sent",
                                                       "started", "finished")})
    return rec


def translate_run(rid: str, *, cancel: threading.Event | None = None, progress=None) -> str:
    """Translate every planned call of a run that is not done yet. Returns the status it leaves: translated, paused
    or cancelled (failed raises)."""
    run = read_run(rid)
    s = normalize(run["settings"])
    if os.environ.get("PYTEST_CURRENT_TEST") and s["backend"] != "mock":
        raise RunnerError(E_PYTEST)
    p = json.loads((run_dir(rid) / "plan.json").read_text(encoding="utf-8"))
    calls = p["calls"]
    todo = [c for c in calls if not _done_path(rid, c).exists()]
    write_run(rid, status="translating", pause=None, progress={"phase": "translate", "done": len(calls) - len(todo),
                                                               "total": len(calls)})
    done = len(calls) - len(todo)
    for c in todo:
        if cancel is not None and cancel.is_set():
            write_run(rid, status="cancelled", progress={"phase": "translate", "done": done, "total": len(calls)})
            return "cancelled"
        try:
            run_one(rid, c, s, cancel)
        except QX.QTablePause as err:
            status = "cancelled" if err.kind == "cancelled" else "paused"
            write_run(rid, status=status, pause={"reason": err.kind, "message": str(err), "reset_at": err.reset_at,
                                                 "call": c["id"]},
                      progress={"phase": "translate", "done": done, "total": len(calls)})
            return status
        done += 1
        write_run(rid, progress={"phase": "translate", "done": done, "total": len(calls)})
        if progress is not None:
            progress(done, len(calls))
    write_run(rid, status="translated", progress={"phase": "translate", "done": done, "total": len(calls)})
    return "translated"


def translations(rid: str) -> dict[str, dict]:
    """{call id: translation record} of a run."""
    d = run_dir(rid) / "translations"
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def probes(rid: str) -> dict[str, dict]:
    d = run_dir(rid) / "probes"
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def list_runs() -> list[dict]:
    """Every run directory's run.json (newest first), without the large parts."""
    root = C.runs_dir()
    out = []
    if not root.exists():
        return out
    for d in root.iterdir():
        p = d / "run.json"
        if d.is_dir() and p.exists():
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({k: r.get(k) for k in ("id", "status", "kind", "label", "backend", "model", "created",
                                               "updated", "progress", "pause", "error", "estimate")}
                       | {"narratives": (r.get("settings") or {}).get("narratives"),
                          "engines": (r.get("settings") or {}).get("engines"),
                          "n_episodes": len((r.get("settings") or {}).get("episodes") or [])})
    out.sort(key=lambda r: r.get("created") or "", reverse=True)
    return out
