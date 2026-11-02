"""The quarterly stack's identity (SIM_DESIGN §3.9 spec.py): content digests of its code and inputs.

A fitted stack is a deterministic function of its code, its constants (config_q, part of the code), its input files
and its arguments (train end, anchors, seed, refits). Its version is built from those contents only -- never from
file modification times, which change when nothing changed -- so a version identifies the model: the agent tree keys
its LLM cache and its run manifests on it, and a checkpoint made under one version must not resume under another.

    code_files(entries)   the static import closure of the entry modules inside the package (every
                          `import scenariolab...` / `from . import ...` reachable from them, functions included)
    code_digest(entries)  sha256 over those files' bytes (relative path + sha256 per file)
    file_digest(paths)    sha256 over the files' bytes (missing files count as "missing")
    inputs_digest()       the stack's data identity: the quarterly panel and its meta (whose inputs digest hashes
                          every raw file the panel read), the monthly panel, model_selection.json and the model card
    stack_identity()      {code, inputs, digest}: what Stack.version carries

Nothing here imports the model modules (the closure is read from source), downloads or calls an LLM.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from .. import config
from . import config_q as cq

ROOT = config.ROOT
PACKAGE = "scenariolab"
STACK_ENTRIES = ("scenariolab.qmodel.rollout", "scenariolab.qmodel.adapter", "scenariolab.qmodel.api")


def _module_file(name: str, root: Path = ROOT) -> Path | None:
    parts = name.split(".")
    base = root.joinpath(*parts)
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _module_name(path: Path, root: Path = ROOT) -> str:
    rel = path.resolve().relative_to(root.resolve()).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


_MEMO: dict = {}


def _stamp(path: Path) -> tuple:
    st = Path(path).stat()
    return (str(path), st.st_size, st.st_mtime_ns)      # a memo key only: identities are the contents' sha256


def _imports(path: Path, root: Path = ROOT) -> set[str]:
    """Package modules a file imports anywhere in its body (absolute and relative imports, inside functions too)."""
    key = ("imports", _stamp(path), str(root))
    if key not in _MEMO:
        _MEMO[key] = _scan_imports(path, root)
    return set(_MEMO[key])


def _scan_imports(path: Path, root: Path) -> set[str]:
    mod = _module_name(path, root)
    pkg = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == PACKAGE or a.name.startswith(PACKAGE + "."):
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = pkg.split(".")
                if node.level > 1:
                    base_parts = base_parts[:len(base_parts) - (node.level - 1)]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if not (base == PACKAGE or base.startswith(PACKAGE + ".")):
                continue
            out.add(base)
            for a in node.names:                        # `from . import factors`: the name may be a module
                out.add(f"{base}.{a.name}")
    return out


def code_files(entries=STACK_ENTRIES, root: Path = ROOT, extra_paths=()) -> list[Path]:
    """The static import closure (package files only) of the entry modules (dotted names or paths), plus every
    package __init__ on the way and `extra_paths` (e.g. a tool script, scanned too)."""
    todo: list[Path] = []
    for e in list(entries) + list(extra_paths):
        p = Path(e) if (isinstance(e, Path) or str(e).endswith(".py")) else _module_file(str(e), root)
        if p is None or not p.exists():
            raise FileNotFoundError(f"找不到模块 {e}")
        todo.append(p.resolve())
    seen: set[Path] = set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            names = _imports(p, root)
        except ValueError:                              # a file outside the package root (a tool): scan imports only
            names = _imports_outside(p)
        for name in names:
            parts = name.split(".")
            for k in range(1, len(parts) + 1):          # the package __init__ files on the way
                f = _module_file(".".join(parts[:k]), root)
                if f is not None and f.resolve() not in seen:
                    todo.append(f.resolve())
    return sorted(seen)


def _imports_outside(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.startswith(PACKAGE)}
        elif isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").startswith(PACKAGE):
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def rel(path: Path, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    key = ("sha", _stamp(p))
    hit = _MEMO.get(key)
    if hit is None:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        hit = _MEMO[key] = h.hexdigest()
    return hit


def digest_of(files: dict[str, str | None]) -> str:
    return hashlib.sha256("\n".join(f"{k} {v or 'missing'}" for k, v in sorted(files.items())).encode()).hexdigest()


def file_digest(paths) -> dict:
    files = {rel(p): sha256_file(p) for p in paths}
    return {"digest": digest_of(files), "files": files}


def code_digest(entries=STACK_ENTRIES) -> dict:
    return file_digest(code_files(entries))


def input_paths() -> list[Path]:
    """The stack's input files: the quarterly panel and its meta (the meta's inputs digest covers every raw file the
    panel read), the monthly panel and its meta, the selected model F and the model card."""
    return [cq.PANEL_Q_PATH, cq.PANEL_Q_META_PATH, config.PANEL_PATH, config.PANEL_META_PATH,
            config.RESULTS_DIR / "model_selection.json", cq.CARD_PATH]


def inputs_digest() -> dict:
    return file_digest(input_paths())


def stack_identity(entries=STACK_ENTRIES) -> dict:
    """{code, inputs, digest} of the stack fitted from the files: sha256 of contents only (a touched or rewritten
    file with the same bytes keeps the identity; any changed byte changes it)."""
    code = code_digest(entries)["digest"]
    data = inputs_digest()["digest"]
    return {"code": code, "inputs": data, "digest": hashlib.sha256(f"{code}|{data}".encode()).hexdigest()}
