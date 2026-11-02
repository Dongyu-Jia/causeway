"""The public episode registry of the narrative test (SIM_DESIGN §8.2). It never reads the key.

data/evaluation/registry.json (written by `seal`, key.py) lists, per shuffled code:

    set            clean (the headline), short (fewer than 56 training quarters: reported apart) or synth
    n_quarters     the window's length (the narratives say 「覆盖期约 N 个季度」 or give it in months)
    narratives     {version: [n1, n2, n3]} present under data/evaluation/narratives/<code>/<version>/
    standin        {version: [n1, ...]}: the research round's stand-in translations (v1 only), QTable/1 legacy
    pair           the N-wrong partner (pre-registered by code, frozen at seal time)
    state_rows     the blind state table of the episode's origin: percentile bands only (computed at seal time by
                   the sealing tool, so the prompt builder needs no date to show them)
    reference      synthetic episodes only: the reference translation the writer knows (the mock replays it)

plus the seal's metadata (seed, the key's sha256, the panel digest). Codes are X01..X18 (episodes) and S01..S08
(synthetic), in an order shuffled with a fixed seed: nothing in a code says when an episode happened.
"""
from __future__ import annotations

import json
from functools import lru_cache

from . import common as C


class RegistryError(FileNotFoundError):
    """The registry is not sealed yet (Chinese message)."""


NOT_SEALED = ("叙事检验的片段登记表还没有生成：请运行 python -m scenariolab.evaluation.narrative seal"
              "（需要季度面板；只写本地文件，不调用模型）")


def sealed() -> bool:
    return C.registry_path().exists()


@lru_cache(maxsize=4)
def _load(path: str, mtime: int) -> dict:
    return json.loads(open(path, encoding="utf-8").read())


def load_registry() -> dict:
    p = C.registry_path()
    if not p.exists():
        raise RegistryError(NOT_SEALED)
    return _load(str(p), p.stat().st_mtime_ns)


def clear_cache() -> None:
    _load.cache_clear()


def entries() -> dict:
    return load_registry()["codes"]


def entry(code: str) -> dict:
    e = entries().get(code)
    if e is None:
        raise KeyError(f"登记表里没有片段「{code}」")
    return e


def codes(set_name: str | None = None) -> list[str]:
    """Registry codes, optionally of one set (clean / short / synth), in code order."""
    out = [c for c, e in entries().items() if set_name is None or e["set"] == set_name]
    return sorted(out)


def select(selection) -> list[str]:
    """Registry codes for a selection: "clean", "short", "synth", "all" (episodes, no synthetic), "everything", or
    a comma list / list of registry codes. Spec codes (E05, N13) need the key: key.resolve() turns them into
    registry codes (the CLI and the API call it); this function only knows registry codes."""
    if isinstance(selection, str):
        s = selection.strip()
        if s in ("clean", "short", "synth"):
            return codes(s)
        if s == "all":
            return codes("clean") + codes("short")
        if s == "everything":
            return codes()
        items = [x.strip() for x in s.split(",") if x.strip()]
    else:
        items = [str(x).strip() for x in (selection or [])]
    known = entries()
    bad = [x for x in items if x not in known]
    if bad:
        raise KeyError(f"登记表里没有这些片段：{'、'.join(bad)}（片段编号是 X01–X18 与 S01–S08；E05、N13 这类设计编号要用 key.resolve 换算）")
    return list(dict.fromkeys(items))


def narrative(code: str, version: str, n: str) -> str | None:
    """The narrative text (data/evaluation/narratives/<code>/<version>/<n>.md), or None when it is not written."""
    p = C.narratives_dir() / code / version / f"{n}.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


def has_narrative(code: str, version: str, n: str) -> bool:
    return n in (entry(code).get("narratives") or {}).get(version, [])


def standin(code: str, n: str, version: str = "v1") -> dict | None:
    """The research round's stand-in translation as QTable/1 legacy ({"qtable", "conversion"}), or None."""
    p = C.standin_dir() / code / version / f"{n}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def reference(code: str, n: str) -> dict | None:
    """A synthetic episode's reference translation (what the writer knows it wrote), or None."""
    return ((entry(code).get("reference") or {}).get(n)) or None


def pair(code: str) -> str | None:
    return entry(code).get("pair")


def state_rows(code: str) -> list[str]:
    return list(entry(code).get("state_rows") or [])


def n_quarters(code: str) -> int:
    return int(entry(code)["n_quarters"])


def public_summary() -> dict:
    """What the evaluation tab may show before any run: codes, sets, lengths, what is written (no windows)."""
    reg = load_registry()
    out = {"schema": reg.get("schema"), "sealed_at": reg.get("sealed_at"), "seed": reg.get("seed"),
           "key_sha256": reg.get("key_sha256"), "panel": reg.get("panel"), "codes": {}}
    for c, e in reg["codes"].items():
        out["codes"][c] = {k: e.get(k) for k in ("set", "n_quarters", "narratives", "standin", "pair", "kind")}
    out["missing"] = reg.get("missing") or []
    return out
