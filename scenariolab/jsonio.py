"""JSON helpers: numpy/pandas-safe, NaN -> null, compact output."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

DECIMALS = 6
# A tiny negative number rounded in prose reads "-0.00" / "-0.0%"; written text drops that sign.
# Dates ("2020-01"), exponents ("3.6e-08") and real negatives ("-0.05") are left alone.
NEG_ZERO = re.compile(r"(^|[^\d.A-Za-z_])[-−](0(?:\.0+)?)(?![\d.])")


def tidy(text: str) -> str:
    return NEG_ZERO.sub(r"\1\2", text) if "0" in text and ("-0" in text or "−0" in text) else text


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return round(f, DECIMALS) if math.isfinite(f) else None
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m")
    if isinstance(obj, (str, np.str_)):
        return tidy(str(obj))
    return obj


def dumps(obj) -> str:
    return json.dumps(clean(obj), ensure_ascii=False, separators=(",", ":"))


def dump(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(obj), encoding="utf-8")
    tmp.replace(path)
    return path


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
