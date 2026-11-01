"""JSON helpers: numpy/pandas-safe, NaN -> null, compact output."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DECIMALS = 6


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
    if isinstance(obj, np.str_):
        return str(obj)
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
