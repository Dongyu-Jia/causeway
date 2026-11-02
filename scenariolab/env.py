"""Minimal .env loader (KEY=VALUE lines) so tokens and keys never live in code."""
from __future__ import annotations

import os
import re
from pathlib import Path

from .config import ROOT

KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_dotenv(path=None) -> None:
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def dotenv_problems(path=None) -> list[tuple[int, str | None]]:
    """The .env lines that are not blank, a comment or KEY=VALUE, as (line number, the key of the last good
    line above it): usually a value that wrapped onto the next line, of which load_dotenv read only the part
    before the break. Never the values themselves."""
    path = Path(path or ROOT / ".env")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    out: list[tuple[int, str | None]] = []
    key = None
    for n, line in enumerate(lines, start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        name = s.split("=", 1)[0].strip() if "=" in s else ""
        if KEY_RE.fullmatch(name):
            key = name
        else:
            out.append((n, key))
    return out
