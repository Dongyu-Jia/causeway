"""Tickers (标的) in the portfolio and withdrawal services (INSTRUMENTS_DESIGN §6.1, §6.4, §7.3).

The instrument layer (scenariolab/instruments) turns every core path into 16 ticker columns in the shape of the 13
asset classes: monthly total returns and cash income per start-of-month value (a payout rule only splits the total
into cash and price). This module is the services' side of it.

Files (all under results/):
  instrument_paths.npz         the built-in scenarios (instruments.build.run, the build's instruments step)
  instrument_meta.json         its summary; run() adds paths_fingerprint (which scenario_paths.npz it belongs to),
                               the GOLD start level and the continuation's summary
  instrument_continuation.npz  years 16..40 (build_continuation): the unconditional continuation of
                               continuation_paths.npz itself (the portfolio step's simulate() chunks, the same seeds,
                               one 480-month path per scenario path), the layer run on every month of it and the
                               yearly factors after month 180 kept; a row keeps its scenario alpha across the seam.
                               It records the fingerprint of the asset factors it ran alongside, so it is only ever
                               chained onto the continuation_paths.npz it belongs to
  custom/<id>.instr.npz        a custom scenario's tickers (write_custom, from server.custom_run): the layout of
                               instrument_paths.npz with S = 1; custom/<id>.macro.npz its macro paths (what the
                               project endpoint recomputes on)

compose(data) gives a ScenarioSet's ticker columns row for row with its asset paths. Every file must belong to the
paths it is mixed with (scenario ids, paths per scenario, start month and the fingerprint of the stored paths), else
PathsNotReady (HTTP 503) naming the build command, or for a custom scenario saved before the layer a ValueError (400).

Weights: an asset id (config.ASSET_IDS) or a ticker id, case-insensitive ("brk.b" is BRK.B); GOLD (金价) is the gold
asset class itself (instruments.ALIASES) and never adds a column.

The API helpers: catalog_report() for GET /api/instruments and evaluate_project() for POST /api/instruments/project
(one ticker on every scenario path: fans, statistics, cash per share, attribution and correlations, recomputed on the
stored scenario paths with the request's parameters; with the defaults the numbers are the build's).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import zipfile
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config, jsonio
from ..instruments import ALIASES, ESTIMATED, LEVERAGED, STRUCTURAL, TICKERS
from . import metrics
from . import withdrawal as wd

PATHS_FILE = "instrument_paths.npz"
META_FILE = "instrument_meta.json"
CONTINUATION_FILE = "instrument_continuation.npz"
MACRO_FILE = "scenario_macro.npz"
CUSTOM_SUFFIX = ".instr.npz"
CUSTOM_MACRO_SUFFIX = ".macro.npz"
ALPHA_MODES = ("shrunk", "half", "raw")
ALPHA_VARIANTS = ("half", "raw")            # stored as yearly factors next to the default ("shrunk") monthly paths
ALPHA_NAMES = {"shrunk": "收缩后（默认）", "half": "减半", "raw": "原始"}
VOL_SOURCES = ("model", "drawn", "fixed")    # what the monthly engine can drive ("core" needs a core that models RV)
VOL_TICKERS = LEVERAGED + ("QQQI",) + ESTIMATED    # the recipes the volatility state enters (drag, premiums, residuals)
FIXED_VOL_BOUNDS = (0.05, 1.5)
BUILD_INSTRUMENTS = "python -m scenariolab.build --only instruments"
CONTINUATION_FORMAT = 1
MONTHS = metrics.MONTHS_PER_YEAR
GOLD = "GOLD"
PROJECT_TICKERS = TICKERS + (GOLD,)
_UPPER = {t.upper(): t for t in TICKERS}
_FACTOR_KEYS = ("total", "price", "income")
_LOCK = threading.RLock()


def catalog() -> dict:
    from ..instruments import params as prm
    return prm.catalog()


def error_text(key: str, **kw) -> str:
    return catalog()["errors"][key].format(**kw)


def ticker_name(t: str) -> str:
    return catalog()["tickers"].get(t, {}).get("name", t)


def resolve(aid, assets) -> tuple[str | None, str | None]:
    """("asset", id) for an asset class or the GOLD alias, ("ticker", id) for a ticker, (None, None) otherwise."""
    if not isinstance(aid, str):
        return None, None
    if aid in assets:
        return "asset", aid
    key = aid.strip().upper()
    if key in ALIASES and ALIASES[key] in assets:
        return "asset", ALIASES[key]
    if key in _UPPER:
        return "ticker", _UPPER[key]
    return None, None


def parse_alpha_mode(value) -> str:
    mode = "shrunk" if value is None else value
    if not isinstance(mode, str) or mode not in ALPHA_MODES:
        raise ValueError(f"alpha_mode 必须是 {' / '.join(ALPHA_MODES)} 之一（估计型标的的 α：收缩后、减半、原始）")
    return mode


# ---------------------------------------------------------------- fingerprints

def paths_fingerprint(ids, first_paths) -> str:
    """instruments.build.paths_fingerprint of scenario paths: the ids and the first two paths of each scenario
    ((S, >= 2, T, A), float32 as the npz stores them)."""
    from ..instruments.build import paths_fingerprint as fp
    return fp({"ids": np.array([str(i) for i in ids]), "total": np.asarray(first_paths, dtype=np.float32)})


def factors_fingerprint(total) -> str:
    """sha256 of yearly total-return factors as continuation_paths.npz stores them (float32)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(total).astype(np.float32)).tobytes()).hexdigest()


def finite_factors(arrays: dict) -> tuple[dict, dict]:
    """(arrays with every non-finite yearly factor set to 0, {name: how many there were}).

    A guard, expected to count nothing: the payout rule used to pay a holding's whole value out (the 2% price
    floor compounding until the price underflowed, then a 0 / 0 split: 3 of 7,800 continuation rows of BME), which
    the explicit distribution cut (distributions.py) and the layer's safe division now prevent. Should anything
    non-finite appear again it is kept out of every file and every result, and the count is in the files' info
    and printed by the build."""
    out, bad = {}, {}
    for k, v in arrays.items():
        a = np.asarray(v)
        n = int(a.size - np.isfinite(a).sum())
        if n:
            bad[k] = n
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        out[k] = a
    return out, bad


# ---------------------------------------------------------------- ticker files

@dataclass
class TickerBlock:
    """One ticker file (instrument_paths.npz or custom/<id>.instr.npz), flattened to a scenario-major path axis."""
    path: str
    ids: list[str]
    n_paths: int
    tickers: list[str]
    estimated: list[str]
    labels: list[str]                    # T + 1 monthly labels
    total: np.ndarray                    # (P, T, K) float32 monthly total returns
    factors: wd.YearFactors              # (P, Y, K) yearly factors of the default alpha (cpi: ones, unused)
    alpha: dict                          # {"half" | "raw": (total, price, income) (P, Y, E) float32 yearly factors}
    start: np.ndarray                    # (K,) USD start prices
    alive_y: np.ndarray | None           # (P, Y, L) the leveraged funds alive at each year end (LEVERAGED order)
    fingerprint: str | None              # the scenario paths it was run on (custom files store it)

    @property
    def months(self) -> int:
        return int(self.total.shape[1])

    def col(self, t: str) -> int:
        return self.tickers.index(t)


def load_block(path: Path) -> TickerBlock:
    from .service import PathsNotReady
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            tickers = [str(x) for x in z["tickers"]]
            ids = [str(x) for x in z["ids"]]
            total = np.asarray(z["total"], dtype=np.float32)
            income = np.asarray(z["income"], dtype=np.float32)
            labels = [str(x) for x in z["labels"]]
            start = np.asarray(z["start"], dtype=float)
            est = [str(x) for x in z["estimated"]] if "estimated" in z.files else [t for t in tickers if t in ESTIMATED]
            alive = np.asarray(z["alive"]) if "alive" in z.files else None
            alpha = {m: tuple(np.asarray(z[f"alpha_{m}_{k}"], dtype=np.float32) for k in _FACTOR_KEYS)
                     for m in ALPHA_VARIANTS if f"alpha_{m}_total" in z.files}
            fp = str(z["paths_fingerprint"]) if "paths_fingerprint" in z.files else None
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as err:
        raise PathsNotReady(f"无法读取 {path.name}（可能正在重新生成，请稍后重试）：{err}") from err
    bad = PathsNotReady(f"{path.name} 的数组形状不一致：请运行 {BUILD_INSTRUMENTS}")
    if total.ndim != 4 or income.shape != total.shape:
        raise bad
    S, N, T, K = total.shape
    Y = T // MONTHS
    if len(ids) != S or len(tickers) != K or len(labels) != T + 1 or start.shape != (K,) or Y < 1:
        raise bad
    fin, _ = finite_factors({"total": total.reshape(S * N, T, K), "income": income.reshape(S * N, T, K)})
    total, income = fin["total"], fin["income"]
    factors = wd.year_factors(total, income, np.ones((S * N, T + 1)))
    del income, fin
    fin, _ = finite_factors({"total": factors.total, "price": factors.price, "income": factors.income})
    factors = wd.YearFactors(fin["total"], fin["price"], fin["income"], factors.cpi, factors.order)
    for m, arrs in alpha.items():
        if any(a.shape != (S, N, Y, len(est)) for a in arrs):
            raise bad
        fin, _ = finite_factors(dict(zip(_FACTOR_KEYS, arrs)))
        alpha[m] = tuple(fin[k].reshape(S * N, Y, len(est)) for k in _FACTOR_KEYS)
    alive_y = None
    if alive is not None and alive.shape[:3] == (S, N, T):
        alive_y = alive[:, :, MONTHS - 1:Y * MONTHS:MONTHS, :].reshape(S * N, Y, alive.shape[3])
    return TickerBlock(path=str(path), ids=ids, n_paths=N, tickers=tickers, estimated=est, labels=labels, total=total,
                       factors=factors, alpha=alpha, start=start, alive_y=alive_y, fingerprint=fp)


@dataclass
class TickerCont:
    """instrument_continuation.npz: the tickers' years after the scenario paths (build_continuation)."""
    tickers: list[str]
    estimated: list[str]
    offset: int                          # months simulated and dropped first (= the scenario paths' length)
    start: str                           # anchor month of the unconditional run
    labels: list[str]                    # year-end labels from month `offset` on
    factors: wd.YearFactors              # (R, Y, K) default alpha; cpi (R, Y + 1) relative to month `offset`
    alpha: dict                          # {"half" | "raw": (total, price, income) (R, Y, E) float32}
    info: dict

    @property
    def n_paths(self) -> int:
        return int(self.factors.total.shape[0])

    @property
    def years(self) -> int:
        return self.factors.years


def load_continuation(path: Path) -> TickerCont:
    from .service import PathsNotReady
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            if "asset_fingerprint" not in z.files or "offset" not in z.files:
                raise PathsNotReady(f"{path.name} 是旧格式（没有记录它接在哪一组续抽路径之后）：请运行 "
                                    f"{BUILD_INSTRUMENTS} 重新生成")
            tickers = [str(x) for x in z["tickers"]]
            est = [str(x) for x in z["estimated"]]
            fin, _ = finite_factors({k: np.asarray(z[f"{k}_y"], dtype=np.float64) for k in _FACTOR_KEYS})
            tot, pri, inc = (fin[k] for k in _FACTOR_KEYS)
            cpi = np.asarray(z["cpi_y"], dtype=np.float64)
            alpha = {}
            for m in ALPHA_VARIANTS:
                if f"alpha_{m}_total_y" in z.files:
                    fin, _ = finite_factors({k: np.asarray(z[f"alpha_{m}_{k}_y"], dtype=np.float32) for k in _FACTOR_KEYS})
                    alpha[m] = tuple(fin[k] for k in _FACTOR_KEYS)
            labels = [str(x) for x in z["labels"]]
            start, offset = str(z["start"]), int(z["offset"])
            info = json.loads(str(z["info"]))
            info["asset_fingerprint"] = str(z["asset_fingerprint"])
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as err:
        if isinstance(err, PathsNotReady):
            raise
        raise PathsNotReady(f"无法读取 {path.name}（可能正在重新生成，请稍后重试）：{err}") from err
    R, Y, K = tot.shape
    if (pri.shape != tot.shape or inc.shape != tot.shape or cpi.shape != (R, Y + 1) or len(tickers) != K
            or len(labels) != Y + 1 or any(a.shape != (R, Y, len(est)) for v in alpha.values() for a in v)):
        raise PathsNotReady(f"{path.name} 的数组形状不一致：请运行 {BUILD_INSTRUMENTS} 重新生成")
    return TickerCont(tickers=tickers, estimated=est, offset=offset, start=start, labels=labels,
                      factors=wd.YearFactors(tot, pri, inc, cpi, np.argsort(-tot, axis=2, kind="stable")),
                      alpha=alpha, info=info)


def write_continuation(path: Path, tickers, estimated, factors: dict, alpha: dict, cpi, labels, start: str,
                       offset: int, info: dict) -> Path:
    """Write instrument_continuation.npz atomically (the server may be reading the old one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    arrays = {"format": np.array(CONTINUATION_FORMAT), "tickers": np.array(list(tickers)),
              "estimated": np.array(list(estimated)), "cpi_y": np.asarray(cpi, dtype=np.float64),
              "labels": np.array([str(x) for x in labels]), "start": np.array(str(start)),
              "offset": np.array(int(offset)), "info": np.array(jsonio.dumps(info)),
              "asset_fingerprint": np.array(str(info["asset_fingerprint"]))}
    arrays.update({f"{k}_y": np.asarray(factors[k], dtype=np.float32) for k in _FACTOR_KEYS})
    for m, arrs in alpha.items():
        arrays.update({f"alpha_{m}_{k}_y": np.asarray(a, dtype=np.float32) for k, a in zip(_FACTOR_KEYS, arrs)})
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------- a scenario set's ticker columns

@dataclass
class TickerSet:
    """A ScenarioSet's ticker columns, row for row with its asset paths (compose)."""
    tickers: list[str]
    estimated: list[str]
    blocks: list[TickerBlock]            # one per ScenarioSet part, in row order
    factors: wd.YearFactors              # (P, Y, K) default alpha over every row, Y = the scenario years
    alpha: dict                          # {mode: (total, price, income) (P, Y, E) float32}: the modes every part has
    meta: dict                           # instrument_meta.json
    continuation: TickerCont | None
    continuation_issue: str
    key: tuple

    def year_columns(self, tickers: list[str], mode: str = "shrunk") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(total, price, income) (P, Y, k) float64 of `tickers` in the scenario years, alpha variant `mode`."""
        return _columns(self.factors, self.alpha, self.tickers, self.estimated, tickers, mode)

    def cont_columns(self, tickers: list[str], mode: str = "shrunk") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.continuation
        return _columns(c.factors, c.alpha, c.tickers, c.estimated, tickers, mode)

    def monthly(self, k: int, tickers: list[str], months: int) -> np.ndarray:
        """(P_k, months, k) float32 monthly total returns of `tickers` on part k's rows."""
        b = self.blocks[k]
        return b.total[:, :months][:, :, [b.col(t) for t in tickers]]

    def alive_y(self, t: str) -> np.ndarray | None:
        """(P, Y) whether leveraged fund t is still alive at each scenario year end (None: not a leveraged fund)."""
        if t not in LEVERAGED or any(b.alive_y is None for b in self.blocks):
            return None
        j = LEVERAGED.index(t)
        Y = self.factors.years
        return np.concatenate([b.alive_y[:, :Y, j] for b in self.blocks])


def _columns(f: wd.YearFactors, alpha: dict, have: list[str], est: list[str], tickers: list[str], mode: str):
    idx = [have.index(t) for t in tickers]
    out = [np.asarray(getattr(f, k)[:, :, idx], dtype=np.float64) for k in _FACTOR_KEYS]    # fancy index: copies
    if mode != "shrunk" and any(t in est for t in tickers):
        if mode not in alpha:
            raise ValueError(f"标的文件里没有 α 变体「{ALPHA_NAMES.get(mode, mode)}」：请运行 {BUILD_INSTRUMENTS}"
                             "（自定义情景请重新运行）")
        Y = f.years
        for j, t in enumerate(tickers):
            if t in est:
                e = est.index(t)
                for n in range(3):
                    out[n][:, :, j] = alpha[mode][n][:, :Y, e]
    return tuple(out)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def ticker_key(data) -> tuple:
    """The stamps of every file a ScenarioSet's tickers come from (a change reloads them)."""
    from .service import _stamp
    root = Path(data.root)
    files = [root / PATHS_FILE, root / META_FILE, root / CONTINUATION_FILE]
    files += [Path(npz).with_name(f"{cid}{CUSTOM_SUFFIX}") for kind, npz, cid in data.files if kind == "custom"]
    return tuple((str(p), _stamp(p)) for p in files)


def _meta(root: Path) -> dict:
    from .service import _cached, _load_json
    p = Path(root) / META_FILE
    return _cached(p, _load_json) if p.exists() else {}


def builtin_issue(data, meta: dict | None = None) -> str:
    """Why instrument_paths.npz cannot be mixed with data's built-in paths ("" when it can), from the files' stamps
    and instrument_meta.json only (nothing large is loaded)."""
    root = Path(data.root)
    p = root / PATHS_FILE
    if not p.exists():
        return error_text("paths_missing")
    meta = _meta(root) if meta is None else meta
    blk = data.parts[0][0]
    again = f"请运行 {BUILD_INSTRUMENTS}"
    ids, n = meta.get("scenario_ids"), meta.get("n_paths")
    if ids is not None and (list(ids) != blk.ids or n != blk.n_paths):
        return f"{PATHS_FILE} 是按另一组情景生成的（{len(ids)} 个情景、每个 {n} 条路径），与 scenario_paths.npz 不一致：{again}"
    fp = meta.get("paths_fingerprint")
    if fp is not None:
        if fp != blk.fingerprint():
            return f"{PATHS_FILE} 是按旧的情景路径生成的（情景已重新生成）：{again}"
    elif _mtime(p) < _mtime(root / "scenario_paths.npz"):
        return f"{PATHS_FILE} 比 scenario_paths.npz 旧（情景已重新生成）：{again}"
    return ""


def continuation_issue(c: TickerCont, data, tickers: list[str]) -> str:
    """Why the ticker continuation cannot follow data's paths and their asset continuation ("" when it can)."""
    again = f"请运行 {BUILD_INSTRUMENTS} 重新生成"
    ac = data.continuation
    if ac is None:
        return data.continuation_issue or "资产的续抽路径不可用"
    if c.tickers != tickers:
        return f"{CONTINUATION_FILE} 的标的列表与 {PATHS_FILE} 不一致：{again}"
    if c.offset != ac.offset or c.start != ac.start or c.n_paths != ac.n_paths or c.years != ac.years:
        return (f"{CONTINUATION_FILE}（从 {c.start} 起、接在第 {c.offset} 个月后、{c.n_paths} 条 × {c.years} 年）与资产的"
                f"续抽路径（{ac.start}、第 {ac.offset} 个月、{ac.n_paths} 条 × {ac.years} 年）不一致：{again}")
    if c.info.get("asset_fingerprint") != ac.fingerprint():
        return f"{CONTINUATION_FILE} 接的是另一组资产续抽路径（continuation_paths.npz 已重新生成）：{again}"
    return ""


def compose(data) -> TickerSet:
    """data's ticker columns (service.ScenarioSet.tickers() caches this by ticker_key)."""
    from .service import PathsNotReady, _cached
    root = Path(data.root)
    meta = _meta(root)
    issue = builtin_issue(data, meta)
    if issue:
        raise PathsNotReady(issue)
    blocks, s0 = [], 0
    for (kind, npz, cid), (blk, _cols) in zip(data.files, data.parts):
        names = data.names[s0:s0 + len(blk.ids)]
        s0 += len(blk.ids)
        if kind == "builtin":
            tb = _cached(root / PATHS_FILE, load_block)
            if tb.ids != blk.ids or tb.n_paths != blk.n_paths or tb.months < data.months or tb.labels[0] != blk.labels[0]:
                raise PathsNotReady(f"{PATHS_FILE} 与 scenario_paths.npz 不一致（情景、路径数或起点不同）："
                                    f"请运行 {BUILD_INSTRUMENTS}")
        else:
            p = Path(npz).with_name(f"{cid}{CUSTOM_SUFFIX}")
            label = "」「".join(names) or cid
            if not p.exists():
                raise ValueError(f"自定义情景「{label}」没有标的路径：{error_text('custom_old')}")
            tb = _cached(p, load_block)
            if (tb.ids != blk.ids or tb.n_paths != blk.n_paths or tb.months < data.months
                    or tb.labels[0] != blk.labels[0]
                    or (tb.fingerprint is not None and tb.fingerprint != blk.fingerprint())):
                raise ValueError(f"自定义情景「{label}」的标的路径与它的情景路径不一致：请重新运行该情景")
            if tb.tickers != blocks[0].tickers:
                raise ValueError(f"自定义情景「{label}」的标的列表与内置情景不一致：{error_text('custom_old')}")
        blocks.append(tb)
    Y = data.years
    factors = wd.YearFactors.concat([b.factors.select(None, Y) for b in blocks])
    alpha = {}
    for m in ALPHA_VARIANTS:
        if all(m in b.alpha for b in blocks):
            parts = [tuple(a[:, :Y] for a in b.alpha[m]) for b in blocks]
            alpha[m] = parts[0] if len(parts) == 1 else tuple(np.concatenate([p[n] for p in parts]) for n in range(3))
    tickers, est = blocks[0].tickers, blocks[0].estimated
    cont, cissue = None, ""
    cp = root / CONTINUATION_FILE
    if data.continuation is None:
        cissue = data.continuation_issue or "资产的续抽路径不可用"
    elif not cp.exists():
        cissue = f"标的的续抽路径（{CONTINUATION_FILE}）还没有生成：请运行 {BUILD_INSTRUMENTS}"
    else:
        try:
            c = _cached(cp, load_continuation)
            cissue = continuation_issue(c, data, tickers)
            cont = None if cissue else c
        except PathsNotReady as err:
            cissue = str(err)
    return TickerSet(tickers=tickers, estimated=est, blocks=blocks, factors=factors, alpha=alpha, meta=meta,
                     continuation=cont, continuation_issue=cissue, key=ticker_key(data))


# ---------------------------------------------------------------- what a portfolio / withdrawal result shows

def flag_union(ts: TickerSet, tickers: list[str]) -> dict:
    """{chip key: [tickers]} over the tickers' flags (instrument_meta.json)."""
    out: dict = {}
    flags = (ts.meta or {}).get("flags") or {}
    for t in tickers:
        for f in flags.get(t, []):
            out.setdefault(f, []).append(t)
    return out


def describe(ts: TickerSet, tickers: list[str], alpha_mode: str, path_weights=None) -> dict:
    """The response block for the tickers a portfolio holds (names, chips, start state, alpha, wipe-outs)."""
    cat = catalog()
    chips = cat.get("chips", {})
    meta = ts.meta or {}
    out = {"tickers": list(tickers), "names": {t: ticker_name(t) for t in tickers},
           "kinds": {t: cat["tickers"].get(t, {}).get("kind") for t in tickers},
           "flags": {t: list((meta.get("flags") or {}).get(t, [])) for t in tickers},
           "chips": {k: chips.get(k, k) for k in flag_union(ts, tickers)},
           "start": {t: (meta.get("start") or {}).get(t) for t in tickers},
           "alpha_mode": alpha_mode, "alpha_mode_name": ALPHA_NAMES[alpha_mode],
           "estimated": [t for t in tickers if t in ts.estimated],
           "built_at": meta.get("built_at"), "anchor_month": meta.get("anchor_month"),
           "params_version": meta.get("params_version"), "vintages": meta.get("vintages")}
    if path_weights is not None:
        out["wipeout_prob_scenario_end"] = {}
        for t in tickers:
            a = ts.alive_y(t)
            if a is not None:
                out["wipeout_prob_scenario_end"][t] = metrics.weighted_mean(~a[:, -1], path_weights)
    return out


def notes(ts: TickerSet, tickers: list[str], alpha_mode: str, withdrawal: bool, scenario_years: int,
          horizon: int | None = None) -> list[str]:
    cat = catalog()
    month = (ts.meta or {}).get("anchor_month") or ""
    out = [f"标的：{'、'.join(ticker_name(t) for t in tickers)}。每个标的是作用在同一批情景路径上的配方"
           "（QQQ、QLD、TQQQ、QQQI、TLT、GLD 按产品机制推导；其余为 2013 年起的季度数据估计），在每条路径上与资产类别一致"
           + (f"；起点为 {month} 月末" if month else "") + "。"]
    chips = cat.get("chips", {})
    for k, held in flag_union(ts, tickers).items():
        if k != "tax":
            out.append(f"（{'、'.join(held)}）{chips.get(k, k)}。")
    lev = [t for t in tickers if t in LEVERAGED]
    if lev:
        out.append(f"{'、'.join(lev)} 在某天纳指跌幅达到 1/L 时归零（爆仓抽签）；按月再平衡的组合在归零后这部分权重按 0 回报计"
                   "（相当于不计息的现金），提取策略里归零的持仓不再有价值。")
    if withdrawal:
        out.extend(cat.get("withdrawal_note") or [])
    if any(t in ts.estimated for t in tickers):
        out.append(f"{cat.get('alpha_note', '')}本次：{ALPHA_NAMES[alpha_mode]}"
                   + ("（三个版本只差 α 这一项）。" if withdrawal else "（组合页按默认的收缩后 α 计算；提取页可切换）。"))
    if horizon is not None and horizon > scenario_years:
        out.append(f"第 {scenario_years + 1} 年起，标的接在同一条无条件续抽路径上继续推演（同一配方、同一波动率状态的延续）；"
                   "估计型标的的 α 按路径固定，续抽前后不变。")
    out.append("标的全部为税前；QQQI 的分配大部分是资本返还，MLP 为 K-1 收入，这里不区分税务性质。")
    return out


# ---------------------------------------------------------------- the build step (after instruments.build.run)

def paths_info(root: Path) -> dict:
    """ids, paths per scenario, months, labels and the fingerprint of results/scenario_paths.npz."""
    with np.load(Path(root) / "scenario_paths.npz", allow_pickle=False) as z:
        total = z["total"]
        ids = [str(x) for x in z["ids"]]
        return {"ids": ids, "n_paths": int(total.shape[1]), "months": int(total.shape[2]),
                "labels": [str(x) for x in z["labels"]], "assets": [str(a) for a in z["assets"]],
                "fingerprint": paths_fingerprint(ids, np.asarray(total[:, :2], dtype=np.float32))}


_ANCHORS: dict = {}


def anchor_for(month: str) -> dict:
    """The start state the build uses (instruments.anchors.current(month, prefer_data=True)), cached per month and
    per version of instrument_meta.json (a rebuild may have read newer data)."""
    from ..instruments import anchors as anc
    from . import service as svc
    key = (month, svc._stamp(Path(svc.RESULTS_DIR) / META_FILE))
    with _LOCK:
        if key not in _ANCHORS:
            _ANCHORS.clear()
            _ANCHORS[key] = anc.current(month, prefer_data=True)
        return _ANCHORS[key]


def gold_start(ctx, month: str) -> dict | None:
    """GOLD (金价): the core's gold asset is the London monthly-average price; its level at the anchor month."""
    try:
        s = ctx.panel["lvl_gold"].dropna()
        s = s[np.asarray(s.index.strftime("%Y-%m")) <= month]
        return {"price": float(s.iloc[-1]), "asof": s.index[-1].strftime("%Y-%m"),
                "basis": "伦敦金月均价（引擎的 gold 资产，美元/盎司）"}
    except (KeyError, AttributeError, IndexError, TypeError, ValueError):
        return None


def build_continuation(ctx, root: Path, info: dict, write: bool = True) -> dict:
    """instrument_continuation.npz: portfolio.service.build_continuation (the portfolio step's chunks, its seeds,
    one 480-month path per scenario path) with the layer run on each chunk (instruments.layer.continuation_factors:
    every ticker plus the alpha variants, a row keeping its scenario's alpha draw), the yearly factors after month
    `offset` kept. The asset factors simulated alongside are fingerprinted, and compared with continuation_paths.npz
    when that file exists (T2: expected 0)."""
    from ..instruments import params as prm
    from ..instruments.layer import continuation_factors, rows_for_chunk
    from . import service as svc
    t0 = time.time()
    ids, N, T = info["ids"], info["n_paths"], info["months"]
    anchor = anchor_for(info["labels"][0][:7])

    def hook(sim, r0: int, chunk: int) -> dict:
        g, i = rows_for_chunk(ids, N, r0, int(sim.total.shape[0]))
        return continuation_factors(sim, ctx.db, T, g, i, chunk=chunk, tickers=TICKERS, anchor=anchor,
                                    alpha_modes=ALPHA_MODES)

    arrays = svc.build_continuation(ctx, offset=T, n_paths=len(ids) * N, yearly=True, hook=hook)
    hooks = arrays.pop("hook")
    f = arrays["factors"]
    est = [t for t in TICKERS if t in ESTIMATED]
    factors, non_finite = finite_factors({k: np.concatenate([h[k] for h in hooks]) for k in _FACTOR_KEYS})
    alpha = {}
    for m in ALPHA_VARIANTS:
        fin, bad = finite_factors({k: np.concatenate([h["alpha"][m][n] for h in hooks]) for n, k in enumerate(_FACTOR_KEYS)})
        alpha[m] = tuple(fin[k] for k in _FACTOR_KEYS)
        non_finite.update({f"alpha_{m}_{k}": v for k, v in bad.items()})
    where = {}
    if non_finite:                                    # which tickers and rows (the layer's payout rule; see finite_factors)
        raw = np.concatenate([h["price"] for h in hooks])
        for r, _y, k in np.argwhere(~np.isfinite(raw)):
            where.setdefault(TICKERS[k], set()).add(int(r))
        where = {t: sorted(rs) for t, rs in where.items()}
        print(f"[instruments] warning: {non_finite} non-finite yearly factors in the continuation set to 0 "
              f"(a holding paid out in full); rows by ticker: {where}", flush=True)
    check = None
    ref = Path(root) / svc.CONTINUATION_FILE
    if ref.exists():
        try:
            with np.load(ref, allow_pickle=False) as z:
                stored = np.asarray(z["total_y"])
            if stored.shape == f.total.shape:
                check = float(np.abs(f.total.astype(np.float32) - stored).max())
        except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile, zlib.error):
            check = None
    Y = f.years
    labels = [str(x) for x in arrays["labels"]][0:Y * MONTHS + 1:MONTHS]
    cinfo = arrays["info"]
    out = {"method": "portfolio.service.build_continuation + instruments.layer.continuation_factors",
           "rows": int(f.total.shape[0]), "years": int(Y), "offset_months": int(arrays["offset"]),
           "seed": cinfo.get("seed"), "chunk_seeds": cinfo.get("chunk_seeds"), "chunk_paths": cinfo.get("chunk_paths"),
           "anchor": arrays["start"], "asset_fingerprint": factors_fingerprint(f.total),
           "paths_fingerprint": info["fingerprint"], "params_version": prm.version(),
           "start_source": anchor.get("source"), "alpha_modes": list(ALPHA_MODES),
           "asset_factors_max_abs_diff_vs_continuation_paths": check,
           "non_finite_set_to_zero": non_finite, "non_finite_rows": where, "seconds": round(time.time() - t0, 2)}
    if write:
        p = write_continuation(Path(root) / CONTINUATION_FILE, TICKERS, est, factors, alpha, f.cpi, labels,
                               arrays["start"], arrays["offset"], out)
        out["file_bytes"] = p.stat().st_size
    return out


def run(ctx, results_dir: str | Path | None = None, write: bool = True) -> dict:
    """The instruments step after instruments.build.run (scenariolab.build.build_instruments): write
    instrument_continuation.npz (build_continuation) and record in instrument_meta.json which scenario paths the
    ticker paths belong to (paths_fingerprint), GOLD's start level and the continuation's summary."""
    root = Path(results_dir or config.RESULTS_DIR)
    info = paths_info(root)
    meta_path = root / META_FILE
    meta = jsonio.load(meta_path) if meta_path.exists() else {}
    if meta and (list(meta.get("scenario_ids") or []) != info["ids"] or meta.get("n_paths") != info["n_paths"]):
        raise RuntimeError(f"{META_FILE} 是按另一组情景生成的：先运行 instruments.build.run")
    cont = build_continuation(ctx, root, info, write=write)
    if write and meta:
        meta["paths_fingerprint"] = info["fingerprint"]
        g = gold_start(ctx, info["labels"][0][:7])
        if g is not None:
            meta.setdefault("start", {})[GOLD] = g
        meta["continuation"] = {"file": CONTINUATION_FILE, **{k: cont.get(k) for k in (
            "rows", "years", "offset_months", "anchor", "seed", "asset_factors_max_abs_diff_vs_continuation_paths",
            "non_finite_set_to_zero", "non_finite_rows", "file_bytes", "seconds", "params_version")}}
        jsonio.dump(meta, meta_path)
    print(f"[instruments] {CONTINUATION_FILE}: {cont.get('rows')} rows x {cont.get('years')} years, "
          f"{(cont.get('file_bytes') or 0) / 1e6:.1f} MB, T2 max |diff| "
          f"{cont.get('asset_factors_max_abs_diff_vs_continuation_paths')}, {cont.get('seconds')} s; "
          f"{PATHS_FILE}: {_size(root / PATHS_FILE)}", flush=True)
    return {"continuation": cont, "paths_fingerprint": info["fingerprint"]}


def _size(path: Path) -> str:
    try:
        return f"{path.stat().st_size:,} bytes"
    except OSError:
        return "missing"


# ---------------------------------------------------------------- custom scenarios

def custom_files(custom_dir: Path, cid: str) -> list[Path]:
    return [Path(custom_dir) / f"{cid}{CUSTOM_SUFFIX}", Path(custom_dir) / f"{cid}{CUSTOM_MACRO_SUFFIX}"]


def _core(kind: str, sid: str, cid: str | None, total, income, cpi, macro: dict, labels, assets, drawn=None):
    """The CoreView the build ran a scenario's tickers on (the same streams: set id, key)."""
    from ..instruments.core import CoreView
    custom = kind == "custom"
    return CoreView.from_arrays("M", total, income, {"cpi": cpi, **{k: macro[k] for k in ("y10", "ff", "spread")}},
                                labels, assets, scenario_id=cid if custom else sid,
                                set_id=f"custom:{cid}" if custom else "builtin", drawn=drawn)


def write_custom(sim, db, cid: str, custom_dir: str | Path, labels: list[str] | None = None,
                 anchor: dict | None = None) -> dict:
    """custom/<id>.instr.npz (instrument_paths.npz's layout with S = 1, the alpha variants included) and
    custom/<id>.macro.npz (scenario_macro.npz's layout) for a custom scenario's SimOutput, as server.custom_run
    saves it: the layer on its paths with streams keyed ("custom:<id>", id). The core is built from the arrays the
    custom npz keeps (float32) and the float32 macro paths, so /api/instruments/project recomputes the same numbers.
    Returns a short summary for the scenario's JSON."""
    from ..engine.summary import month_labels
    from ..instruments import estimated as est
    from ..instruments.layer import run_layer
    t0 = time.time()
    custom_dir = Path(custom_dir)
    n, T, _ = sim.total.shape
    labels = [str(x) for x in (labels or month_labels(sim.start, T))]
    macro = {k: np.asarray(sim.macro[k], dtype=np.float32) for k in ("y10", "ff", "spread")}
    dates = np.asarray(db.dates.strftime("%Y-%m")).astype("U7")
    idx = np.asarray(sim.idx)
    core = _core("custom", cid, cid, sim.total, sim.income, sim.macro["cpi"], macro, labels, list(sim.assets),
                 dates[idx])
    anchor = anchor or anchor_for(labels[0][:7])
    out = run_layer(core, TICKERS, anchor=anchor)
    stacked, non_finite = finite_factors(dict(zip(("total", "income"), out.stacked(TICKERS))))
    total, income = stacked["total"], stacked["income"]
    alive = np.stack([out.paths[t].alive if out.paths[t].alive is not None else np.ones((n, T), bool)
                      for t in LEVERAGED], axis=2)
    est_t = [t for t in TICKERS if t in ESTIMATED]
    arrays = {"tickers": np.array(TICKERS), "ids": np.array([cid]), "probs": np.array([1.0]),
              "labels": np.array(labels), "estimated": np.array(est_t), "leveraged": np.array(LEVERAGED),
              "start": np.array([float((anchor["prices"].get(t) or {}).get("price", np.nan)) for t in TICKERS]),
              "total": total[None].astype(np.float32), "income": income[None].astype(np.float32),
              "alive": alive[None]}
    ones = np.ones((n, T + 1))
    for mode in ALPHA_VARIANTS:
        e = est.simulate(core, out.vol.mkt_vol_quarterly(), est_t, None, anchor["estimated"], alpha_mode=mode)
        g = wd.year_factors(np.stack([e[t].total for t in est_t], axis=2), np.stack([e[t].income for t in est_t], axis=2),
                            ones)
        f, bad = finite_factors({k: getattr(g, k) for k in _FACTOR_KEYS})
        non_finite.update({f"alpha_{mode}_{k}": v for k, v in bad.items()})
        for k in _FACTOR_KEYS:
            arrays[f"alpha_{mode}_{k}"] = f[k][None].astype(np.float32)
    fp = paths_fingerprint([cid], np.asarray(sim.total, dtype=np.float32)[None, :2])
    arrays["paths_fingerprint"] = np.array(fp)
    custom_dir.mkdir(parents=True, exist_ok=True)
    instr, mac = custom_files(custom_dir, cid)
    tmp = custom_dir / f"{cid}.instr.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(instr)
    tmp = custom_dir / f"{cid}.macro.tmp.npz"
    np.savez_compressed(tmp, ids=np.array([cid]), **{k: v[None] for k, v in macro.items()},
                        drawn_idx=idx[None].astype(np.int16), db_dates=dates, paths_fingerprint=np.array(fp))
    tmp.replace(mac)
    return {"saved": True, "files": [instr.name, mac.name], "tickers": list(TICKERS),
            "anchor_month": anchor.get("month"), "start_source": anchor.get("source"),
            "non_finite_set_to_zero": non_finite, "seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- GET /api/instruments

def status(data=None) -> dict:
    """Whether the ticker files can be used with the current scenario paths (cheap: stamps and the meta only)."""
    from . import service as svc
    root = Path(svc.RESULTS_DIR)
    out = {"paths": (root / PATHS_FILE).exists(), "continuation": (root / CONTINUATION_FILE).exists(),
           "meta": (root / META_FILE).exists(), "available": False, "issue": "", "build": BUILD_INSTRUMENTS}
    try:
        data = data if data is not None else svc.load_scenarios()
    except svc.PathsNotReady as err:
        out["issue"] = str(err)
        return out
    out["issue"] = builtin_issue(data)
    out["available"] = not out["issue"]
    out["horizon_years"] = {"scenario": data.years, "max": data.max_years}
    return out


def catalog_report() -> dict:
    """GET /api/instruments: the catalog (instruments.catalog.entries: Chinese names, kinds, recipe cards with fees
    from the fee table, parameters with defaults and ranges, validation, sources and limits, chips) with the build's
    start state, flags, statistics, data vintages and diagnostics, plus what the services accept."""
    from ..instruments import catalog as cat
    from . import service as svc
    root = Path(svc.RESULTS_DIR)
    meta = svc._load_json(root / META_FILE) if (root / META_FILE).exists() else {}
    out = cat.entries(meta)
    gold = (meta.get("start") or {}).get(GOLD) if meta else None
    if gold and GOLD in out["tickers"]:
        out["tickers"][GOLD]["start"] = gold
    out.update({
        "status": status(),
        "order": [t for g in out.get("groups", []) for t in g.get("tickers", [])],
        "stored": list(TICKERS), "structural": list(STRUCTURAL), "estimated": list(ESTIMATED),
        "leveraged": list(LEVERAGED), "aliases": dict(ALIASES),
        "alpha_modes": [{"id": m, "name": ALPHA_NAMES[m]} for m in ALPHA_MODES],
        "vol_sources": list(VOL_SOURCES), "fixed_vol_bounds": list(FIXED_VOL_BOUNDS),
        "presets": [dict(p) for p in svc.INSTRUMENT_PRESETS],
        "continuation": (meta or {}).get("continuation"),
    })
    return out


# ---------------------------------------------------------------- POST /api/instruments/project

PART_NAMES = {"index": "指数", "vol_drag": "波动损耗", "third_moment": "三阶矩", "fourth_moment": "四阶矩", "dividend": "股息", "financing": "融资",
              "fee": "费率", "premium": "期权费收入", "capped_upside": "被封顶的上涨", "coupon": "票息", "duration": "久期",
              "convexity": "凸性", "cash": "现金传导", "credit_spread": "信用利差", "alpha": "α", "residual": "残差",
              "gold": "金价"}
KEY_OPTION_PARAMS = {"s_eff", "dist_rule", "alpha_mode"}      # options the layer takes by name (others by value)
_PROJECT: OrderedDict = OrderedDict()                          # finished responses
_MAX_PROJECT = 12
_ARRAYS: OrderedDict = OrderedDict()                           # scenario income / CPI / macro arrays per part
_MAX_ARRAYS = 4


def parse_ticker(value) -> str:
    if isinstance(value, str):
        key = value.strip().upper()
        if key == GOLD:
            return GOLD
        if key in _UPPER:
            return _UPPER[key]
    raise ValueError(f"未知标的：{value}（可选：{'、'.join(PROJECT_TICKERS)}）")


def param_specs(t: str) -> dict:
    """{param id: catalog spec} a ticker accepts (its catalog params, the estimated tickers' alpha, a start price)."""
    cat = catalog()
    e = cat["tickers"].get(t, {})
    specs = {p["id"]: p for p in (e.get("params") or [])}
    if e.get("kind") == "estimated":
        specs.update({p["id"]: p for p in (cat.get("estimated_common", {}).get("params") or [])})
    specs["price"] = {"id": "price", "name": "起点价格（美元）", "range": [0.0, None]}
    return specs


def parse_params(t: str, params) -> tuple[dict, dict]:
    """(the layer's params for ticker t, the effective parameters shown: the defaults overridden by the request)."""
    from .service import _flag, _number
    specs = param_specs(t)
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params 必须是 {参数 id: 值} 形式的对象")
    layer: dict = {}
    shown = {k: s.get("default") for k, s in specs.items() if k != "price"}
    for k, v in params.items():
        if k not in specs:
            raise ValueError(f"{t} 没有参数「{k}」（可用：{'、'.join(specs)}）")
        s = specs[k]
        name = s.get("name", k)
        if "options" not in s and "range" not in s:          # a switch (the ruin draw)
            layer[k] = shown[k] = _flag(v, f"{t} 的{name}（{k}）", bool(s.get("default", True)))
            continue
        opts = s.get("options") or {}
        if isinstance(v, str) and v in opts:
            layer[k] = v if k in KEY_OPTION_PARAMS else opts[v]
            shown[k] = v
            continue
        if "range" not in s or (isinstance(v, str) and not _numeric(v)):
            raise ValueError(f"{t} 的{name}（{k}）必须是 {' / '.join(map(str, opts))} 之一"
                             + ("，或范围内的数字" if "range" in s else ""))
        x = _number(v, f"{t} 的{name}（{k}）")
        lo, hi = s["range"]
        if (lo is not None and x < lo) or (hi is not None and x > hi) or (k == "price" and x <= 0):
            raise ValueError(f"{t} 的{name}（{k}）必须在 {lo:g} 到 {'∞' if hi is None else f'{hi:g}'} 之间（收到 {x:g}）")
        layer[k] = shown[k] = x
    return layer, shown


def _numeric(v: str) -> bool:
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


def wfan(values: np.ndarray, weights=None, pcts=metrics.PCTS, decimals: int = 6) -> dict:
    """metrics.fan, vectorized over columns: weighted percentile bands (plotting position (S_i - w_i / 2) / W, linear
    interpolation, clamped; equal weights give np.percentile's hazen) of every column of (n, K) values. NaN values
    and zero-weight rows are left out."""
    v = np.asarray(values, dtype=float)
    if weights is None and v.shape[0] and np.isfinite(v).all():          # equal weights: hazen, vectorized
        out = np.percentile(v, pcts, axis=0, method="hazen")
        return {f"p{p}": [round(float(x), decimals) for x in row] for p, row in zip(pcts, out)}
    w = np.ones(v.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    keep = w > 0
    v, w = v[keep], w[keep]
    out = np.full((len(pcts), v.shape[1]), np.nan)
    if v.shape[0]:
        bad = ~np.isfinite(v)
        order = np.argsort(np.where(bad, np.inf, v), axis=0, kind="stable")
        vs = np.take_along_axis(v, order, axis=0)
        ws = np.where(np.take_along_axis(bad, order, axis=0), 0.0, w[order])
        q = np.asarray(pcts, dtype=float) / 100.0
        for k in range(v.shape[1]):
            m = ws[:, k] > 0
            if m.any():
                cw = np.cumsum(ws[m, k])
                out[:, k] = np.interp(q, (cw - 0.5 * ws[m, k]) / cw[-1], vs[m, k])
    return {f"p{p}": [None if not math.isfinite(x) else round(float(x), decimals) for x in row]
            for p, row in zip(pcts, out)}


def _wq(x, p, w=None) -> float | None:
    v = metrics.weighted_percentile(x, p, w)
    return None if not math.isfinite(v) else float(v)


def _wm(x, w=None) -> float | None:
    v = metrics.weighted_mean(x, w)
    return None if not math.isfinite(v) else float(v)


def qlog(total) -> np.ndarray:
    """(n, T) monthly simple returns -> (n, T // 3) quarterly log returns (a wipe-out floored at -99.9999%)."""
    lt = np.log1p(np.maximum(np.asarray(total, dtype=np.float64), -0.999999))
    n, T = lt.shape
    return lt[:, :T // 3 * 3].reshape(n, T // 3, 3).sum(axis=2)


def path_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row correlation of (n, Q) a and b (NaN where either is flat)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=1) / np.where(den > 0, den, 1.0), np.nan)


def _scenario_arrays(data) -> list[dict]:
    """Per ScenarioSet part: the monthly income, CPI and macro paths the layer needs next to the ScenarioSet's own
    asset totals (scenario_paths.npz + scenario_macro.npz, or custom/<id>.npz + custom/<id>.macro.npz)."""
    from .service import PathsNotReady, _stamp
    out = []
    for (kind, npz, cid), (blk, cols) in zip(data.files, data.parts):
        npz = Path(npz)
        mac = Path(data.root) / MACRO_FILE if kind == "builtin" else npz.with_name(f"{cid}{CUSTOM_MACRO_SUFFIX}")
        if not mac.exists():
            if kind == "builtin":
                raise PathsNotReady(f"{MACRO_FILE} 还没有生成（情景的利率与利差路径）：请运行 {BUILD_INSTRUMENTS}")
            raise ValueError(f"自定义情景 {cid} 没有宏观路径文件：{error_text('custom_old')}")
        key = (str(npz), _stamp(npz), str(mac), _stamp(mac))
        with _LOCK:
            arr = _ARRAYS.get(key)
            if arr is None:
                arr = _load_scenario_arrays(npz, mac, blk)
                _ARRAYS[key] = arr
                while len(_ARRAYS) > _MAX_ARRAYS:
                    _ARRAYS.popitem(last=False)
            _ARRAYS.move_to_end(key)
        out.append({**arr, "kind": kind, "cid": cid, "cols": cols, "blk": blk})
    return out


def _load_scenario_arrays(npz: Path, mac: Path, blk) -> dict:
    from .service import PathsNotReady
    try:
        with np.load(npz, allow_pickle=False) as z:
            income = np.asarray(z["income"], dtype=np.float32)
            cpi = np.asarray(z["cpi"], dtype=np.float32)
        with np.load(mac, allow_pickle=False) as m:
            ids = [str(x) for x in m["ids"]]
            macro = {k: np.asarray(m[k], dtype=np.float32) for k in ("y10", "ff", "spread")}
            dates = np.asarray(m["db_dates"]).astype(str) if "db_dates" in m.files else None
            drawn = dates[np.asarray(m["drawn_idx"])] if dates is not None and "drawn_idx" in m.files else None
            fp = str(m["paths_fingerprint"]) if "paths_fingerprint" in m.files else None
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as err:
        raise PathsNotReady(f"无法读取 {npz.name} / {mac.name}（可能正在重新生成，请稍后重试）：{err}") from err
    S, N = len(blk.ids), blk.n_paths
    if ids != blk.ids or macro["y10"].shape[:2] != (S, N) or (fp is not None and fp != blk.fingerprint()):
        raise PathsNotReady(f"{mac.name} 与 {npz.name} 不一致（情景已重新生成）：请运行 {BUILD_INSTRUMENTS}")
    T = income.shape[2]
    return {"income": income.reshape(S * N, T, -1), "cpi": cpi.reshape(S * N, T + 1),
            "macro": {k: v.reshape(S * N, -1) for k, v in macro.items()},
            "drawn": None if drawn is None else drawn.reshape(S * N, T)}


def _meta_now() -> dict:
    from . import service as svc
    return _meta(Path(svc.RESULTS_DIR))


def _ticker_paths(t: str, core, layer_params: dict, vol_source: str, fixed_vol, alpha_mode: str, anchor: dict | None,
                  price: float | None) -> dict:
    """One scenario's paths of ticker t: total, income, level (USD per share), log parts, flags, alive."""
    from ..instruments.layer import run_layer
    if t == GOLD:                                  # the core's gold asset itself
        g_tot, g_inc = core.asset_total("gold"), core.asset_income("gold")
        start = price or float((((_meta_now().get("start") or {}).get(GOLD)) or {}).get("price") or 1.0)
        growth = np.cumprod(np.maximum(1.0 + g_tot - g_inc, 0.0), axis=1)
        level = start * np.concatenate([np.ones((core.n, 1)), growth], axis=1)
        return {"total": g_tot, "income": g_inc, "level": level, "parts": {"gold": np.log1p(g_tot)},
                "flags": ("gold_avg", "tax"), "alive": None, "meta": {}}
    p = dict(layer_params)
    if price is not None and t in ESTIMATED:
        p["price"] = price
    out = run_layer(core, [t], params={t: p}, vol_source=vol_source, anchor=anchor, alpha_mode=alpha_mode,
                    fixed_vol=fixed_vol)
    ip = out.paths[t]
    level = ip.level
    if price is not None and t not in ESTIMATED and level[0, 0] > 0:
        level = level * (price / float(level[0, 0]))
    return {"total": ip.total, "income": ip.income, "level": level, "parts": ip.parts, "flags": tuple(ip.flags),
            "alive": ip.alive, "meta": ip.meta}


def path_stats(tot, inc, level, cpi, alive=None) -> dict:
    """Per-path statistics of one ticker on one scenario's monthly paths (arrays (n,) and the fan inputs)."""
    tot = np.asarray(tot, dtype=np.float64)
    n, T = tot.shape
    mult = np.concatenate([np.ones((n, 1)), np.cumprod(1.0 + tot, axis=1)], axis=1)
    cpi = np.asarray(cpi, dtype=np.float64)
    real = mult / (cpi[:, :T + 1] / cpi[:, :1])
    Y = T // MONTHS
    cash = np.asarray(inc, dtype=np.float64) * level[:, :-1]          # USD per share paid in each month
    dps_y = cash[:, :Y * MONTHS].reshape(n, Y, MONTHS).sum(axis=2)
    p0 = np.maximum(level[:, 0], 1e-12)
    h10 = min(10, Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        payer = dps_y[:, 0] > 0
        lowest = dps_y[:, 1:h10].min(axis=1) if h10 > 1 else dps_y[:, 0]
        cut10 = np.where(payer, lowest < 0.9 * dps_y[:, 0], np.nan)
        cut20 = np.where(payer, lowest < 0.8 * dps_y[:, 0], np.nan)
        d10 = np.where(payer, dps_y[:, h10 - 1] / np.where(payer, dps_y[:, 0], 1.0), np.nan)
    growth = np.full((n, T), np.nan)                                  # undefined after a wipe-out
    np.divide(mult[:, 1:], mult[:, :-1], out=growth, where=mult[:, :-1] > 0)
    vol = metrics.annualized_vol(growth - 1.0)
    m10 = min(120, T)
    return {"mult": mult, "real": real, "level": level, "dps_y": dps_y,
            "terminal": mult[:, -1], "terminal_real": real[:, -1],
            "cagr10": metrics.cagr(mult[:, m10], m10 / MONTHS), "cagr15": metrics.cagr(mult[:, -1], T / MONTHS),
            "real_cagr15": metrics.cagr(real[:, -1], T / MONTHS),
            "price_cagr10": metrics.cagr(level[:, m10] / p0, m10 / MONTHS),
            "max_dd": metrics.max_drawdown(mult), "max_dd_q": metrics.max_drawdown(mult[:, ::3]),
            "vol": vol, "income_y1": dps_y[:, 0] / p0, "cash10_per_10k": dps_y[:, :h10].sum(axis=1) / p0 * 1e4,
            "cut10": cut10, "cut20": cut20, "dps10_over_1": d10,
            "dead": None if alive is None else ~np.asarray(alive)[:, -1]}


def summary(st: dict, w=None) -> dict:
    """Distribution summary of path_stats arrays (optionally path-weighted): the 标的推演 tab's comparison table."""
    return {"cagr10_p10": _wq(st["cagr10"], 10, w), "cagr10_p50": _wq(st["cagr10"], 50, w),
            "cagr10_p90": _wq(st["cagr10"], 90, w), "cagr15_p50": _wq(st["cagr15"], 50, w),
            "real_cagr15_p50": _wq(st["real_cagr15"], 50, w), "price_cagr10_p50": _wq(st["price_cagr10"], 50, w),
            "max_dd_p50": _wq(st["max_dd"], 50, w), "max_dd_q_p50": _wq(st["max_dd_q"], 50, w),
            "vol": _wm(st["vol"], w), "prob_loss15": _wm(st["terminal"] < 1.0, w),
            "prob_loss15_real": _wm(st["terminal_real"] < 1.0, w),
            "income_y1_p50": _wq(st["income_y1"], 50, w), "cash10_per_10k_p10": _wq(st["cash10_per_10k"], 10, w),
            "cash10_per_10k_p50": _wq(st["cash10_per_10k"], 50, w), "cash10_per_10k_p90": _wq(st["cash10_per_10k"], 90, w),
            "cut10_prob": _wm(st["cut10"], w), "cut20_prob": _wm(st["cut20"], w),
            "dps10_over_1_p50": _wq(st["dps10_over_1"], 50, w),
            "wipeout_prob": None if st["dead"] is None else _wm(st["dead"], w)}


def fans(st: dict, w=None) -> dict:
    return {"total": wfan(st["mult"], w), "total_real": wfan(st["real"], w), "price": wfan(st["level"], w, decimals=4),
            "dps": wfan(st["dps_y"], w, decimals=4)}


def attribution(parts: dict, log_total: np.ndarray, years: float) -> dict:
    """Annualized additive log parts: on the median path (by total log return) and the mean over the paths whose
    parts are all defined (a wiped-out leveraged fund has none after its wipe-out)."""
    if not parts:
        return {"median_path": {}, "mean": {}}
    full = np.ones(log_total.shape[0], dtype=bool)
    for v in parts.values():
        full &= np.isfinite(np.asarray(v, dtype=np.float64)).all(axis=1)
    order = np.argsort(np.where(np.isfinite(log_total), log_total, -np.inf), kind="stable")
    med = int(order[len(order) // 2])
    out = {"median_path": {}, "mean": {}}
    for k, v in parts.items():
        s = np.asarray(v, dtype=np.float64)
        out["median_path"][k] = float(s[med].sum() / years) if np.isfinite(s[med]).all() else None
        out["mean"][k] = float(s[full].sum(axis=1).mean() / years) if full.any() else None
    return out


def _mix_mean(values, probs) -> float | None:
    pairs = [(v, p) for v, p in zip(values, probs) if v is not None and p > 0]
    if not pairs:
        return None
    return float(sum(v * p for v, p in pairs) / sum(p for _, p in pairs))


def _asset_name(a: str) -> str:
    return {x["id"]: x.get("name", x["id"]) for x in config.ASSETS}.get(a, a)


def _brief_meta(metas: list[dict]) -> dict:
    """The recipe's run information (mean over scenarios for numbers, e.g. the running c_LR)."""
    nums: dict = {}
    text: dict = {}
    for m in metas:
        for k, v in (m or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                nums.setdefault(k, []).append(float(v))
            elif isinstance(v, str):
                text.setdefault(k, v)
    return {**text, **{k: float(np.mean(v)) for k, v in nums.items()}}


def _project_notes(t: str, entry: dict, flags: set, cat: dict, vol_source: str, mode: str,
                   credit: dict | None = None, ruin: bool | None = None) -> list[str]:
    chips = cat.get("chips", {})
    out = ["预测（带判断的假设）只发生在情景层：标的按配方在同一批情景路径上推出，同一条路径上标的与资产类别、标的之间永远一致；"
           "加权汇总按情景概率混合各情景的路径。",
           "总回报假设分配全部再投资；价格为每股除息后的美元价格；现金分配为每股每年美元（不再投资）。"]
    for f in sorted(flags):
        if f == "tax":
            continue
        line = f"{chips.get(f, f)}。"
        if f in ("credit_link", "credit_link_ok") and credit and credit.get("sim_lo") is not None:
            line = (f"{chips.get(f, f)}：本次与美股的季度相关模拟 {credit['sim_lo']:.2f}–{credit['sim_hi']:.2f}、历史 "
                    f"{credit['hist']:.2f}，{credit['n_over']}/{credit['n']} 个情景低于历史超过 {credit['tol']:.2f}。")
        out.append(line)
    if ruin is False:
        out.append("爆仓抽签已关闭：归零概率没有计算，结果只看日度再平衡的连续部分（敏感性）。")
    if vol_source != "model" and t in VOL_TICKERS:
        out.append({"drawn": "期内波动来自抽到的历史月份（只作对照）。",
                    "fixed": "期内波动为固定值：敏感性，不是推演。"}[vol_source])
    elif vol_source != "model":
        out.append("期内波动的选择不影响这个标的（它的配方不用期内波动）。")
    if t in ESTIMATED:
        out.append(f"{cat.get('alpha_note', '')}本次：{ALPHA_NAMES[mode]}。")
    out.extend(entry.get("limits") or [])
    out.append("全部为税前。")
    return out


_QCOLS: OrderedDict = OrderedDict()      # (set key, ticker key) -> {column: (P, Q) quarterly log returns}


def _quarterly_columns(data, arrays: list[dict], ts: TickerSet) -> dict:
    """Quarterly log total returns of every asset class and stored ticker (default parameters) on data's rows."""
    key = (data.key, ts.key)
    with _LOCK:
        if key in _QCOLS:
            _QCOLS.move_to_end(key)
            return _QCOLS[key]
    T = data.months
    cols = {a: qlog(np.concatenate([(arr["blk"].total[:, :T, j] if arr["cols"] is None
                                     else arr["blk"].total[:, :T, arr["cols"][j]]) for arr in arrays])).astype(np.float32)
            for j, a in enumerate(data.assets)}
    cols.update({t: qlog(np.concatenate([b.total[:, :T, b.col(t)] for b in ts.blocks])).astype(np.float32)
                 for t in ts.tickers})
    with _LOCK:
        _QCOLS[key] = cols
        while len(_QCOLS) > 1:
            _QCOLS.popitem(last=False)
    return cols


def _correlations(data, t: str, q_t: np.ndarray, arrays: list[dict], ts: TickerSet, pw: np.ndarray) -> dict:
    """Quarterly log total-return correlations of ticker t with every asset class and every other stored ticker:
    per path, then the median per scenario and the weighted median over the mixture; history where known."""
    from ..instruments import params as prm
    groups = data.groups
    out = {"with": {}, "hist_us_equity": prm.bridges().get("hist_corr_with_us_equity", {}).get(t),
           "method": "季度对数总回报，逐路径计算相关系数，再取中位数（加权汇总按情景概率）"}
    for name, q in _quarterly_columns(data, arrays, ts).items():
        if name == t:
            continue
        c = path_corr(q_t, q.astype(np.float64))
        out["with"][name] = {"weighted": _wq(c, 50, pw),
                             "per_scenario": {sid: _wq(c[groups == k], 50) for k, sid in enumerate(data.ids)}}
    return out


def _no_drawn(kind: str, cid) -> Exception:
    from .service import PathsNotReady
    if kind == "builtin":
        return PathsNotReady(f"{MACRO_FILE} 没有抽到的历史月份：请运行 {BUILD_INSTRUMENTS}")
    return ValueError(f"自定义情景 {cid} 没有抽到的历史月份：请重新运行该情景")


def project_report(data, ticker, params=None, scenario_probs=None, vol_source: str = "model",
                   fixed_vol=None, alpha_mode: str = "shrunk") -> dict:
    """evaluate_project() on already loaded scenarios."""
    from . import service as svc
    from .service import _stamp
    t0 = time.time()
    t = parse_ticker(ticker)
    layer_params, shown = parse_params(t, params)
    price = layer_params.pop("price", None)
    if vol_source not in VOL_SOURCES:
        raise ValueError(f"vol_source 必须是 {' / '.join(VOL_SOURCES)} 之一（期内波动：模型、抽到的历史月、固定）")
    if vol_source == "fixed":
        if fixed_vol is None:
            raise ValueError("vol_source 为 fixed 时需要 fixed_vol（年化波动，例如 0.25）")
        fixed_vol = svc._number(fixed_vol, "固定波动（fixed_vol）")
        lo, hi = FIXED_VOL_BOUNDS
        if not lo <= fixed_vol <= hi:
            raise ValueError(f"固定波动（fixed_vol）必须在 {lo} 到 {hi} 之间（年化，0.25 = 25%）")
    elif fixed_vol is not None:
        raise ValueError("fixed_vol 只在 vol_source 为 fixed 时使用")
    mode = parse_alpha_mode(layer_params.pop("alpha_mode", alpha_mode))    # the ticker's own alpha param wins
    if t in ESTIMATED:
        shown["alpha_mode"] = mode
    probs = svc.scenario_probabilities(data, scenario_probs)
    ts = data.tickers()                         # the ticker files must belong to these paths (503 / 400 otherwise)
    root = Path(data.root)
    key = (data.key, ts.key, tuple((str(p), _stamp(p)) for p in (root / MACRO_FILE,)), t,
           json.dumps(layer_params, sort_keys=True, default=str), price, vol_source, fixed_vol, mode,
           tuple(np.round(probs, 12)))
    with _LOCK:
        hit = _PROJECT.get(key)
        if hit is not None:
            _PROJECT.move_to_end(key)
            return {**hit, "cached": True}
    arrays = _scenario_arrays(data)
    labels, T = data.labels, data.months
    anchor = anchor_for(labels[0][:7]) if t != GOLD else None
    per, stats_all, qlogs, flags, metas = [], [], [], set(), []
    k_s = 0
    for arr in arrays:
        blk, cols, kind, cid = arr["blk"], arr["cols"], arr["kind"], arr["cid"]
        tot_a = blk.total[:, :T] if cols is None else blk.total[:, :T][:, :, cols]
        inc_a = arr["income"][:, :T] if cols is None else arr["income"][:, :T][:, :, cols]
        if vol_source == "drawn" and arr["drawn"] is None:
            raise _no_drawn(kind, cid)
        for j, sid in enumerate(blk.ids):
            a, b = j * blk.n_paths, (j + 1) * blk.n_paths
            core = _core(kind, sid, cid, tot_a[a:b], inc_a[a:b], arr["cpi"][a:b, :T + 1],
                         {k: v[a:b, :T + 1] for k, v in arr["macro"].items()}, labels[:T + 1], data.assets,
                         None if arr["drawn"] is None else arr["drawn"][a:b, :T])
            tp = _ticker_paths(t, core, layer_params, vol_source, fixed_vol, mode, anchor, price)
            st = path_stats(tp["total"], tp["income"], tp["level"], core.cpi, tp["alive"])
            lt = np.log1p(np.maximum(np.asarray(tp["total"], dtype=np.float64), -0.999999)).sum(axis=1)
            per.append({"id": data.ids[k_s], "name": data.names[k_s], "source": data.sources[k_s],
                        "prob": float(probs[k_s]), "n_paths": b - a, **summary(st), "fans": fans(st),
                        "attribution": attribution(tp["parts"], lt, T / MONTHS)})
            k_s += 1
            stats_all.append(st)
            qlogs.append(qlog(tp["total"]))
            flags |= set(tp["flags"])
            metas.append(tp["meta"])
    pw = metrics.mixture_weights(data.groups, probs)
    pooled = {k: np.concatenate([s[k] for s in stats_all]) for k in stats_all[0] if k != "dead"}
    pooled["dead"] = None if stats_all[0]["dead"] is None else np.concatenate([s["dead"] for s in stats_all])
    del stats_all
    weighted = {**summary(pooled, pw), "fans": fans(pooled, pw)}
    parts = list(per[0]["attribution"]["mean"]) if per else []
    weighted["attribution"] = {"mean": {k: _mix_mean([e["attribution"]["mean"].get(k) for e in per],
                                                     [e["prob"] for e in per]) for k in parts}}
    del pooled
    corr = _correlations(data, t, np.concatenate(qlogs), arrays, ts, pw)
    credit = None
    if "credit_link" in flags:                     # T18 on this request's own paths (as the build does per scenario)
        from ..instruments.layer import credit_link_status, resolve_credit_link
        us = corr["with"].get("us_equity") or {}
        credit = credit_link_status(us.get("per_scenario") or {}, corr.get("hist_us_equity"))
        flags = set(resolve_credit_link(sorted(flags), credit))
    cat = catalog()
    entry = cat["tickers"].get(t, {})
    meta = ts.meta or {}
    start_info = dict((meta.get("start") or {}).get(t) or {})
    p0 = per[0]["fans"]["price"]["p50"][0] if per else None
    if price is not None:                          # the reader's own start price replaces the build's
        flags.discard("start_est")
        start_info = {"price": price, "user_price": price,
                      "asof": start_info.get("asof") or (data.labels[0] if data.labels else None),
                      "basis": "user", "build": start_info}
    defaults = {k: s.get("default") for k, s in param_specs(t).items() if k != "price"}
    ruin = bool(layer_params.get("ruin", True)) if t in LEVERAGED else None
    res = {
        "ticker": t, "name": entry.get("name", t), "kind": entry.get("kind"), "native": entry.get("native"),
        "labels": list(labels[:T + 1]), "quarter_index": list(range(0, T + 1, 3)),
        "years": data.year_labels(data.years), "dates": data.year_dates(data.years),
        "start": {**start_info, "price": p0, "user_price": price},
        "params": shown, "defaults": defaults, "vol_source": vol_source, "fixed_vol": fixed_vol,
        "vol_applies": t in VOL_TICKERS, "ruin": ruin, "credit_link": credit,
        "alpha_mode": mode if t in ESTIMATED else None,
        "recomputed": bool(layer_params or price is not None or (vol_source != "model" and t in VOL_TICKERS)
                           or (t in ESTIMATED and mode != "shrunk")),
        "flags": sorted(flags), "chips": {f: cat.get("chips", {}).get(f, f) for f in sorted(flags)},
        "part_names": {k: PART_NAMES.get(k, _asset_name(k)) for k in parts},
        "per_scenario": per, "aggregate": {"weighted": weighted}, "corr": corr, "recipe_meta": _brief_meta(metas),
        "notes": _project_notes(t, entry, flags, cat, vol_source, mode, credit, ruin),
        "built_at": meta.get("built_at"), "vintages": meta.get("vintages"),
        "elapsed_s": round(time.time() - t0, 2), "cached": False,
    }
    res = json.loads(jsonio.dumps(res))
    with _LOCK:
        _PROJECT[key] = res
        _PROJECT.move_to_end(key)
        while len(_PROJECT) > _MAX_PROJECT:
            _PROJECT.popitem(last=False)
    return res


def clear_cache() -> None:
    with _LOCK:
        _PROJECT.clear()
        _ARRAYS.clear()
        _ANCHORS.clear()
        _QCOLS.clear()


def evaluate_project(ticker, params: dict | None = None, include_custom: list[str] = (),
                     scenario_probs: dict | None = None, vol_source: str = "model", fixed_vol: float | None = None,
                     alpha_mode: str = "shrunk") -> dict:
    """POST /api/instruments/project: one ticker on every scenario path (INSTRUMENTS_DESIGN §6.4, §7.1).

    ticker: a stored ticker or GOLD; params: that ticker's recipe parameters (GET /api/instruments lists them, plus
    price = a start price in USD), recomputed on the stored scenario paths with the build's random streams, so the
    defaults reproduce the build; vol_source model / drawn / fixed (+ fixed_vol); alpha_mode for an estimated ticker.
    ValueError (400) for a bad input, PathsNotReady (503) when the files are missing or stale."""
    from . import service as svc
    return project_report(svc.load_scenarios(include_custom), ticker, params, scenario_probs, vol_source, fixed_vol,
                          alpha_mode)
