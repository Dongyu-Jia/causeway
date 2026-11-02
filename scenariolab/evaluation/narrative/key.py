"""THE SEALED KEY of the narrative test (SIM_DESIGN §8.2): which shuffled code is which historical window.

Read only by the scorer (score.py), the audit (audit.py), the seal command and the CLI / API when they turn a spec
code such as E05 into a registry code. The blind prompt builder (prompts.py) and the runner (runner.py) never import
this module, directly or through another module; tests/test_narrative.py checks the import graph and that building
every prompt and running a mock translation never opens key.json.

    python -m scenariolab.evaluation.narrative seal [--force] [--no-synth]

writes, under data/evaluation/ (git-ignored, like every local result):
    key.json           code -> {spec_code, research_code, start, end, origin, train_end, n_quarters, train_quarters,
                       set, useq, dd, bust, checks}; synthetic codes -> {train_end, origin, path, seed, truth}
    registry.json      the public part (episodes.py): sets, lengths, narratives, stand-ins, N-wrong pairs, the blind
                       state rows of each origin, the key's sha256 (so a changed key is noticed)
    narratives/<code>/<version>/<n>.md and standin/<code>/v1/<n>.json   (from the committed research archive)

The episode set is the spec's pre-registered table (§8.2): 14 clean episodes with at least 56 training quarters
(the headline) and 4 short-training episodes reported apart. Codes are shuffled with a fixed seed (SEED), so sealing
again gives the same registry; the N-wrong pairing is computed by code (nearest window length among clean episodes
whose origin is at least 5 years away, ties by registry code) and frozen. Busts are DD <= -0.20 (the window's
maximum drawdown of the quarterly us_equity price path, starting from 0). useq, DD and the training quarters are
checked against design/d02's printed values (three decimals; the spec's table rounds them to two) and every miss is
listed in key.json["checks"].

The windows are public in the spec; the key keeps the mapping from the codes the translator's runs see to those
windows out of every code path that builds or sends a prompt.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C

SCHEMA = "narrative-key/1"
SEED = 20261102
# spec code -> (first quarter, last quarter, set, research code, training quarters, useq and DD as design/d02
# printed them (3 decimals; the check values), and the spec table's two-decimal useq and DD (§8.2; informational:
# the table rounded d02's 0.025 and -0.175 once more, to 0.03 and -0.18))
EPISODES = {
    "E11": ("1990Q3", "1991Q4", "clean", "E11", 70, 0.168, -0.177, 0.17, -0.18),
    "N13": ("1987Q3", "1988Q4", "clean", None, 58, -0.093, -0.272, -0.09, -0.27),
    "E07": ("1994Q1", "1995Q2", "clean", "E07", 84, 0.140, -0.061, 0.14, -0.06),
    "E02": ("1995Q3", "1997Q4", "clean", "E02", 90, 0.554, 0.000, 0.55, 0.00),
    "N14": ("1998Q1", "2000Q4", "clean", None, 100, 0.284, -0.170, 0.28, -0.17),
    "E12": ("2001Q1", "2002Q4", "clean", "E12", 112, -0.383, -0.456, -0.38, -0.46),
    "E01": ("2004Q3", "2006Q4", "clean", "E01", 126, 0.233, -0.028, 0.23, -0.03),
    "E05": ("2007Q3", "2009Q2", "clean", "E05", 138, -0.468, -0.625, -0.47, -0.63),
    "N15": ("2010Q2", "2012Q4", "clean", None, 149, 0.201, -0.175, 0.20, -0.18),
    "E04": ("2014Q3", "2015Q4", "clean", "E04", 166, 0.025, -0.083, 0.03, -0.08),
    "E09": ("2017Q1", "2019Q4", "clean", "E09", 176, 0.363, -0.162, 0.36, -0.16),
    "E08": ("2020Q1", "2021Q2", "clean", "E08", 188, 0.329, -0.230, 0.33, -0.23),
    "E10": ("2022Q1", "2023Q4", "clean", "E10", 196, -0.015, -0.296, -0.02, -0.30),
    "N18": ("2024Q1", "2026Q2", "clean", None, 204, 0.449, -0.056, 0.45, -0.06),
    "E06": ("1973Q4", "1975Q2", "short", "E06", 3, -0.144, -0.569, None, None),
    "E03": ("1979Q4", "1982Q2", "short", "E03", 27, 0.038, -0.211, None, None),
    "N16": ("1977Q1", "1979Q3", "short", None, 16, 0.114, -0.122, None, None),
    "N17": ("1983Q1", "1984Q4", "short", None, 40, 0.158, -0.129, None, None),
}
CLEAN_QUARTERS = 119                      # §8.2: the clean set has 119 quarters
PAIR_MIN_GAP_Q = 20                       # N-wrong: the partner's origin at least 5 years away
CHECK_TOL = 0.0005 + 1e-9                 # d02 printed useq and DD to three decimals


class KeyError_(KeyError):
    """The key is missing or does not match the registry (Chinese message)."""


def key_path() -> Path:
    return C.eval_dir() / "key.json"


def assign_codes(seed: int = SEED) -> dict[str, str]:
    """{spec code: registry code X01..X18}: a seeded permutation, so the code says nothing about the window."""
    spec = sorted(EPISODES)
    order = np.random.default_rng(seed).permutation(len(spec))
    return {spec[j]: f"{C.CODE_PREFIX['episode']}{i + 1:02d}" for i, j in enumerate(order)}


def window_stats(Q: pd.DataFrame, start: str, end: str) -> dict:
    """The window's realized statistics (design/d02): quarters, training quarters (from the panel's first quarter),
    the us_equity price return over the window, its maximum drawdown from the start, nasdaq and Σ d_gz."""
    a, b = pd.Period(start, "Q"), pd.Period(end, "Q")
    y = Q.loc[a:b, "y_us_equity"].astype(float)
    c = np.r_[0.0, y.cumsum().to_numpy()]
    dd = float(np.min(c - np.maximum.accumulate(c)))
    idx = list(Q.index)
    return {"n_quarters": int(len(y)), "train_quarters": int(idx.index(a)), "useq": float(c[-1]), "dd": dd,
            "nasdaq": float(Q.loc[a:b, "y_nasdaq"].sum()), "dgz_sum": float(Q.loc[a:b, "d_gz"].sum()),
            "origin": str(a), "train_end": str(a - 1), "start": str(a), "end": str(b)}


def pairs(eps: dict) -> dict[str, str | None]:
    """N-wrong partners: each clean episode -> the clean episode of the nearest window length whose origin is at least
    PAIR_MIN_GAP_Q quarters away, ties broken by registry code order."""
    clean = sorted(c for c, e in eps.items() if e["set"] == "clean")
    out = {}
    for c in clean:
        o = pd.Period(eps[c]["origin"], "Q")
        cand = [d for d in clean if d != c and abs((pd.Period(eps[d]["origin"], "Q") - o).n) >= PAIR_MIN_GAP_Q]
        cand.sort(key=lambda d: (abs(eps[d]["n_quarters"] - eps[c]["n_quarters"]), d))
        out[c] = cand[0] if cand else None
    return out


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n"


def seal(*, Q: pd.DataFrame | None = None, force: bool = False, synth: bool = True, n_boot: int | None = None,
         seed: int = SEED) -> dict:
    """Write key.json, registry.json, the narratives and the stand-ins (see the module docstring). An existing key is
    kept unless force=True (sealing is part of the pre-registration; with the same seed the codes are the same)."""
    from ...llm import qtranslator as QX
    from ...qmodel import panel as qpanel
    from . import research as RS
    from . import synth as SY
    if key_path().exists() and not force:
        return {"sealed": False, "reason": "已经封存（加 --force 重新封存；同一个种子给出同样的编号）", "key": str(key_path())}
    t0 = time.perf_counter()
    Q = qpanel.load_panel() if Q is None else Q
    meta = {}
    try:
        info = qpanel.load_meta()["info"]
        meta = {"last_data_quarter": info.get("last_data_quarter"), "inputs_digest": info.get("inputs_digest"),
                "card_digest": info.get("card_digest")}
    except (OSError, KeyError, ValueError):
        meta = {"last_data_quarter": str(Q.index[-2])}
    codes = assign_codes(seed)
    arch = RS.load_archive() if RS.available() else {"episodes": {}}
    key_eps, reg_codes, checks, missing = {}, {}, [], []
    for spec, (start, end, set_name, research, c_train, c_useq, c_dd, t_useq, t_dd) in EPISODES.items():
        code = codes[spec]
        st = window_stats(Q, start, end)
        bust = st["dd"] <= C.BUST_DD
        for name, want, got, table in (("useq", c_useq, st["useq"], t_useq), ("dd", c_dd, st["dd"], t_dd),
                                       ("train_quarters", c_train, st["train_quarters"], c_train)):
            tol = 0 if name == "train_quarters" else CHECK_TOL
            ok = abs(float(got) - float(want)) <= tol
            checks.append({"spec_code": spec, "what": name, "check": want, "computed": round(float(got), 4),
                           "spec_table": table, "ok": ok})
        key_eps[code] = {"spec_code": spec, "research_code": research, "set": set_name, "bust": bool(bust), **st}
        narr, stand = {}, {}
        ep = (arch.get("episodes") or {}).get(research or "")
        if ep:
            for n in C.NARRATIVES:
                _write_text(C.narratives_dir() / code / "v1" / f"{n}.md", ep["narratives"][n])
                _write_text(C.standin_dir() / code / "v1" / f"{n}.json", _dump(ep["standin"][n]))
            narr["v1"] = list(C.NARRATIVES)
            stand["v1"] = list(C.NARRATIVES)
        else:
            missing.append({"code": code, "version": "v1", "text": "研究轮没有这个片段的叙事（新增片段，等 v2 写作）"})
        missing.append({"code": code, "version": "v2", "text": C.LABELS["no_v2"]})
        reg_codes[code] = {"kind": "episode", "set": set_name, "n_quarters": st["n_quarters"], "narratives": narr,
                           "standin": stand, "pair": None,
                           "state_rows": QX.state_rows("test", Q=Q, quarter=st["origin"])}
    for c, p in pairs({c: {**v, "origin": v["origin"]} for c, v in key_eps.items()}).items():
        reg_codes[c]["pair"] = p
    key_synth = {}
    if synth:
        sy = SY.episodes(Q, n_boot=n_boot)
        order = np.random.default_rng([seed, 7]).permutation(len(sy))
        for i, j in enumerate(order):
            e = sy[int(j)]
            code = f"{C.CODE_PREFIX['synth']}{i + 1:02d}"
            for n in ("n1", "n2"):
                _write_text(C.narratives_dir() / code / "synth" / f"{n}.md", e["texts"][n])
            key_synth[code] = {"kind": "synth", "set": "synth", "train_end": e["train_end"], "origin": e["origin"],
                               "path": e["path"], "seed": e["seed"], "n_quarters": e["n_quarters"], "start": e["start"],
                               "stack": e["stack"], "truth": e["truth"]}
            reg_codes[code] = {"kind": "synth", "set": "synth", "n_quarters": e["n_quarters"],
                               "narratives": {"synth": ["n1", "n2"]}, "standin": {}, "pair": None,
                               "reference": e["texts"]["reference"],
                               "state_rows": QX.state_rows("test", Q=Q, quarter=e["origin"])}
    clean_q = sum(v["n_quarters"] for v in key_eps.values() if v["set"] == "clean")
    checks.append({"spec_code": "*", "what": "clean_quarters", "check": CLEAN_QUARTERS, "computed": clean_q,
                   "spec_table": CLEAN_QUARTERS, "ok": clean_q == CLEAN_QUARTERS})
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    key = {"schema": SCHEMA, "seed": seed, "sealed_at": stamp, "panel": meta, "episodes": key_eps,
           "synth": key_synth, "checks": checks}
    key_text = _dump(key)
    _write_text(key_path(), key_text)
    reg = {"schema": C.REGISTRY_SCHEMA, "seed": seed, "sealed_at": stamp, "panel": meta,
           "key_sha256": hashlib.sha256(key_text.encode("utf-8")).hexdigest(),
           "codes": dict(sorted(reg_codes.items())), "missing": missing}
    _write_text(C.registry_path(), _dump(reg))
    from . import episodes as EP
    EP.clear_cache()
    _KEY.clear()
    return {"sealed": True, "codes": len(key_eps), "synth": len(key_synth), "checks_failed": [c for c in checks
                                                                                               if not c["ok"]],
            "missing": len(missing), "seconds": round(time.perf_counter() - t0, 1), "key": str(key_path()),
            "registry": str(C.registry_path())}


_KEY: dict = {}


def load_key(verify: bool = True) -> dict:
    """key.json; verify=True checks it against the registry's sha256 (a key edited after sealing is refused)."""
    p = key_path()
    if not p.exists():
        raise KeyError_("叙事检验的密钥 key.json 还没有生成：请运行 python -m scenariolab.evaluation.narrative seal")
    stamp = p.stat().st_mtime_ns
    hit = _KEY.get("key")
    if hit is not None and hit[0] == stamp and hit[1] == str(p):
        return hit[2]
    text = p.read_text(encoding="utf-8")
    if verify:
        from . import episodes as EP
        want = EP.load_registry().get("key_sha256")
        got = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if want != got:
            raise KeyError_("key.json 与登记表记录的哈希不一致：密钥在封存之后被改过，请重新封存（seal --force）")
    key = json.loads(text)
    _KEY["key"] = (stamp, str(p), key)
    return key


def episode(code: str) -> dict:
    """One code's key entry (episodes and synthetic episodes)."""
    k = load_key()
    e = k["episodes"].get(code) or k["synth"].get(code)
    if e is None:
        raise KeyError_(f"密钥里没有片段「{code}」")
    return e


def resolve(selection) -> list[str]:
    """Registry codes for a user's selection: clean / short / synth / all / everything, or a comma list of registry
    codes (X05) and spec codes (E05, N13)."""
    from . import episodes as EP
    if isinstance(selection, str) and selection.strip() in ("clean", "short", "synth", "all", "everything"):
        return EP.select(selection.strip())
    items = [str(i).strip() for i in (selection.split(",") if isinstance(selection, str) else list(selection or []))]
    items = [x for x in items if x]
    known = EP.entries()
    if all(x in known for x in items):                 # registry codes only: no need to open the key
        return EP.select(items)
    by_spec = {v["spec_code"]: c for c, v in load_key()["episodes"].items()}
    return EP.select([by_spec.get(x, x) for x in items])


def unsealed(codes) -> dict:
    """{code: what the report may show after scoring: spec code, window, set, bust, useq, dd}."""
    k = load_key()
    out = {}
    for c in codes:
        e = k["episodes"].get(c)
        if e is not None:
            out[c] = {"spec_code": e["spec_code"], "research_code": e.get("research_code"), "start": e["start"],
                      "end": e["end"], "set": e["set"], "bust": e["bust"], "useq": round(e["useq"], 4),
                      "dd": round(e["dd"], 4), "train_quarters": e["train_quarters"], "n_quarters": e["n_quarters"]}
        elif c in k["synth"]:
            s = k["synth"][c]
            out[c] = {"spec_code": None, "research_code": None, "start": s["origin"], "end": None, "set": "synth",
                      "bust": None, "train_end": s["train_end"], "n_quarters": s["n_quarters"]}
    return out
