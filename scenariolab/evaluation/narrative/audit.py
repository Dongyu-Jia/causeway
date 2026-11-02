"""The blindness audit of a narrative-test run (SIM_DESIGN §8.4), permanent and recomputed with every run.

    prompts          every distinct translation prompt the run used: its sha256, size and the blind scan
                     (prompts.scan_prompt: years, quarter labels, month counts, T0 offsets, footprint keys, era names)
    narratives       per narrative used: years, digits in a protocol-v2 N3, names / places, banned words
    recognition      the probes' answers scored against the sealed key: recognised when the guessed years overlap
                     the window's years widened by one year on each side; the rate per narrative (n1, n3) and overall
    contamination    the contamination index (§8.4): the median |z| of the translated values against realized
                     (z in training sd of the field's quarterly change), next to two baselines on the same cells: the
                     history prior's mean (the training mean; the research round: 0.66–0.76) and the model's
                     unconditional completion (the U arm's median; 0.42); per narrative arm, and for E1 / E1c only
    recall           a flag per (episode, N3 translation, E1 / E1c / E2 field) stated in at least 3 quarters whose path
                     correlates with the realized path above 0.9 (the research round's E08 n3 d_gz: 0.99). Only N3:
                     N1 and N2 give numbers, so a close path there is the text, not memory
    delivery         the simulated median of each stated value against the value (training sd), per arm and form
    translations     statuses (ok / repaired / fallback / standin) and the refusal codes
    label            「识别出年代的回放（不是盲测）」 when every probe recognised its period; the stand-ins' label; the
                     mock's label

It reads the key (recognition) and the run's fields.csv (the scorer's stated and realized values).
"""
from __future__ import annotations

import collections
import json

import numpy as np
import pandas as pd

from . import common as C
from . import episodes as EP
from . import key as K
from . import prompts as PR
from . import runner as RN

E1_CLASSES = ("E1", "E1c", "E2")
RECALL_CORR = 0.9
RECALL_MIN_Q = 3
RECALL_ARMS = ("N3",)                     # the number-free narrative: a close path can only come from memory


def _years(q: str) -> int:
    return pd.Period(q, "Q").year


def recognised(answer: dict, start: str, end: str) -> bool:
    """Whether a probe's guessed years overlap the window's years (± 1 year)."""
    if not answer or not answer.get("known"):
        return False
    gs, ge = answer.get("start_year"), answer.get("end_year")
    gs = gs if gs is not None else ge
    ge = ge if ge is not None else gs
    if gs is None:
        return False
    lo, hi = min(gs, ge), max(gs, ge)
    return max(lo, _years(start) - 1) <= min(hi, _years(end) + 1)


def recognition(probes: dict, key: dict) -> dict:
    rows, by_n = [], collections.defaultdict(list)
    for cid, p in probes.items():
        e = key["episodes"].get(p["code"])
        if e is None:
            continue
        ans = p.get("answer") or {}
        ok = recognised(ans, e["start"], e["end"])
        n = (p.get("source") or {}).get("n")
        rows.append({"code": p["code"], "narrative": n, "known": bool(ans.get("known")), "recognised": ok,
                     "guess": [ans.get("start_year"), ans.get("end_year")], "confidence": ans.get("confidence"),
                     "status": p.get("status")})
        by_n[n].append(ok)
    rate = float(np.mean([r["recognised"] for r in rows])) if rows else None
    return {"n": len(rows), "recognised": sum(r["recognised"] for r in rows), "rate": rate,
            "by_narrative": {n: {"n": len(v), "rate": float(np.mean(v))} for n, v in sorted(by_n.items())},
            "probes": rows}


def contamination(fields: pd.DataFrame, endo: dict) -> dict:
    """The contamination index per narrative arm and overall (stated cells), with the two baselines."""
    if fields is None or fields.empty:
        return {"n": 0}
    st = fields[(~fields["completed"].astype(bool)) & np.isfinite(fields["stated"]) & np.isfinite(fields["realized"])
                & (fields["sd"] > 0)].copy()
    if st.empty:
        return {"n": 0}
    st["z_trans"] = (st["stated"] - st["realized"]) / st["sd"]
    st["z_prior"] = (st["prior_mean"] - st["realized"]) / st["sd"]
    st["z_uncond"] = (st["u_q50"] - st["realized"]) / st["sd"]
    st["e1"] = st["field"].map(lambda f: endo.get(f) in E1_CLASSES)

    def block(g):
        return {"n": int(len(g)), "median_abs_z": round(float(g["z_trans"].abs().median()), 4),
                "baseline_prior": round(float(g["z_prior"].abs().median()), 4),
                "baseline_uncond": round(float(g["z_uncond"].abs().median()), 4) if g["z_uncond"].notna().any()
                else None}

    out = {"overall": block(st), "by_arm": {a: block(g) for a, g in st.groupby("arm")}}
    e1 = st[st["e1"]]
    out["e1"] = block(e1) if len(e1) else {"n": 0}
    out["e1_by_arm"] = {a: block(g) for a, g in e1.groupby("arm")} if len(e1) else {}
    comp = fields[fields["completed"].astype(bool) & np.isfinite(fields["realized"]) & (fields["sd"] > 0)]
    if len(comp):
        z = ((comp["sim_q50"] - comp["realized"]) / comp["sd"]).abs()
        out["completion"] = {"n": int(len(comp)), "median_abs_z": round(float(z.median()), 4)}
    return out


def delivery(fields: pd.DataFrame) -> dict:
    """Factor delivery (§8.6): the simulated median of every stated value against the value, in training sd (0 for a
    hard value; within the stated sd for a soft one; large where the ±4 sd saturation clipped it), per arm."""
    if fields is None or fields.empty:
        return {"n": 0}
    st = fields[(~fields["completed"].astype(bool)) & np.isfinite(fields["stated"]) & np.isfinite(fields["sim_q50"])
                & (fields["sd"] > 0)].copy()
    if st.empty:
        return {"n": 0}
    st["gap"] = ((st["sim_q50"] - st["stated"]) / st["sd"]).abs()

    def block(g):
        return {"n": int(len(g)), "median_gap_sd": round(float(g["gap"].median()), 4),
                "share_within_half_sd": round(float((g["gap"] <= 0.5).mean()), 4),
                "max_gap_sd": round(float(g["gap"].max()), 4)}

    return {"overall": block(st), "by_arm": {a: block(g) for a, g in st.groupby("arm")},
            "by_form": {f: block(g) for f, g in st.groupby(st["form"].fillna("value"))}}


def recall(fields: pd.DataFrame, endo: dict) -> list[dict]:
    if fields is None or fields.empty:
        return []
    st = fields[(~fields["completed"].astype(bool)) & np.isfinite(fields["stated"]) & np.isfinite(fields["realized"])]
    out = []
    for (code, arm, f), g in st.groupby(["code", "arm", "field"]):
        if arm not in RECALL_ARMS or endo.get(f) not in E1_CLASSES or len(g) < RECALL_MIN_Q:
            continue
        a, b = g["stated"].to_numpy(float), g["realized"].to_numpy(float)
        if a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > RECALL_CORR:
            out.append({"code": code, "arm": arm, "field": f, "quarters": int(len(g)), "corr": round(r, 3),
                        "mae_sd": round(float(np.mean(np.abs(a - b) / g["sd"].to_numpy(float))), 3)})
    return out


def translation_summary(trans: dict) -> dict:
    status = collections.Counter(t.get("status") for t in trans.values())
    codes = collections.Counter()
    for t in trans.values():
        for it in (t.get("validation") or {}).get("refused") or []:
            codes[it.get("code")] += 1
    labels = sorted({t.get("label") for t in trans.values() if t.get("label")})
    return {"n": len(trans), "status": dict(status), "refused": dict(codes), "labels": labels,
            "fallback": [t["id"] for t in trans.values() if t.get("status") == "fallback"]}


def audit_run(rid: str) -> dict:
    run = RN.read_run(rid)
    trans, probes = RN.translations(rid), RN.probes(rid)
    key = K.load_key()
    from .score import card_classes
    endo = card_classes()
    prompts, seen = [], set()
    for t in trans.values():
        h = t.get("system_sha256")
        if h in seen or t.get("kind") == "standin":
            continue
        seen.add(h)
        text = PR.translation_system(t["prompt_code"])
        prompts.append({"sha256": h, "rebuilt_sha256": PR.sha256(text), "chars": len(text),
                        "scan": PR.scan_prompt(text), "code": t["prompt_code"]})
    narr = []
    used = sorted({(t["source"]["code"], t["source"]["version"], t["source"]["n"]) for t in trans.values()})
    for code, version, n in used:
        text = EP.narrative(code, version, n)
        if text is not None:
            narr.append({"code": code, "version": version, "narrative": n, **PR.scan_narrative(text, version, n)})
    fpath = RN.run_dir(rid) / "fields.csv"
    fields = pd.read_csv(fpath) if fpath.exists() else pd.DataFrame()
    rec = recognition(probes, key)
    labels = []
    if run.get("kind") == "mock":
        labels.append(C.LABELS["mock"])
    tsum = translation_summary(trans)
    if C.LABELS["legacy"] in tsum["labels"]:
        labels.append(C.LABELS["legacy"])
    if rec["n"] and rec["recognised"] == rec["n"]:
        labels.append(C.LABELS["recognised"])
    out = {"prompts": prompts, "prompt_clean": all(not p["scan"] for p in prompts),
           "narratives": narr, "narratives_clean": all(n["clean"] for n in narr) if narr else None,
           "recognition": rec, "contamination": contamination(fields, endo), "recall": recall(fields, endo),
           "delivery": delivery(fields),
           "translations": tsum, "labels": labels,
           "manifest": {"calls": len(json.loads((RN.run_dir(rid) / "manifest.json").read_text(encoding="utf-8"))
                                     .get("calls") or {}),
                        "sent_files": len(list((RN.run_dir(rid) / "sent").glob("*.txt")))
                        if (RN.run_dir(rid) / "sent").exists() else 0},
           "baselines_research": {"prior": [0.66, 0.76], "uncond": 0.42}}
    from ... import jsonio
    (RN.run_dir(rid) / "audit.json").write_text(jsonio.dumps(out), encoding="utf-8")     # NaN -> null
    return out
