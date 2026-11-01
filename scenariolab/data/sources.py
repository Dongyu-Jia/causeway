"""Downloaders for the public data sources, with a plain-file cache.

No API keys are needed: FRED's graph CSV endpoint, Ken French's data library,
Nareit's monthly workbook, the datasets/gold-prices CSV and Shiller's ie_data.xls
are all open. Every download is cached under data/raw/<source>/: a missing file
is downloaded, a cached one is reused unless `refresh=True`.
"""
from __future__ import annotations

import re
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .. import __version__
from ..config import RAW_DIR

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
NAREIT_URL = "https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls"
GOLD_URL = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"

REPO_URL = "https://github.com/Dongyu-Jia/causeway"
USER_AGENT = f"causeway/{__version__} (+{REPO_URL})"


def _download(url: str, dest: Path, refresh: bool, params: dict | None = None, attempts: int = 4,
              timeout: float = 90, check: Callable[[bytes], bool] | None = None) -> Path:
    """dest, downloaded from url unless it is already cached (non-empty) and refresh is False.

    `check(content)` must accept the body before it is cached (a rejected body counts as a failed
    attempt); after `attempts` failures a RuntimeError is raised and nothing is written.
    """
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            if check is not None and not check(resp.content):
                raise ValueError("unexpected content")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return dest
        except Exception as err:  # network hiccups are common on FRED
            last_err = err
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url} ({last_err})")


def _to_month_start(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx)).to_period("M").to_timestamp()


# ---------------------------------------------------------------- FRED

def fred(series_id: str, eop: bool = False, refresh: bool = False) -> pd.Series:
    """One FRED series as a float Series indexed by month (or quarter) start.

    eop=True asks FRED to aggregate a daily series to month-end values, which
    lines market variables up with month-end asset returns.
    """
    suffix = "_eop" if eop else ""
    dest = RAW_DIR / "fred" / f"{series_id}{suffix}.csv"
    params = {"id": series_id}
    if eop:
        params.update({"fq": "Monthly", "fam": "eop"})
    _download(FRED_URL, dest, refresh, params=params)
    df = pd.read_csv(dest)
    date_col = df.columns[0]
    values = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s = pd.Series(values.to_numpy(), index=pd.to_datetime(df[date_col]), name=series_id)
    s = s.dropna()
    s.index = _to_month_start(s.index)
    return s[~s.index.duplicated(keep="last")]


# ---------------------------------------------------------------- Ken French

def _parse_french_tables(text: str) -> dict[str, pd.DataFrame]:
    """Split a Ken French CSV into its titled tables (monthly rows only)."""
    tables: dict[str, pd.DataFrame] = {}
    lines = text.splitlines()
    i = 0
    title = None
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith(",") and title is not None:
            header = [h.strip() for h in stripped.split(",")[1:]]
            rows = []
            i += 1
            while i < len(lines) and re.match(r"^\s*\d{6}\s*,", lines[i]):
                parts = [p.strip() for p in lines[i].split(",")]
                rows.append([parts[0]] + parts[1:len(header) + 1])
                i += 1
            if rows:
                df = pd.DataFrame(rows, columns=["date"] + header)
                df.index = pd.to_datetime(df.pop("date"), format="%Y%m")
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.mask(df <= -99.99)
                tables[title] = df
            title = None
            continue
        if stripped and not re.match(r"^\s*\d", stripped):
            title = stripped
        i += 1
    return tables


def french(name: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """All monthly tables of a Ken French zip, e.g. '12_Industry_Portfolios'."""
    dest = RAW_DIR / "french" / f"{name}_CSV.zip"
    _download(FRENCH_URL.format(name=name), dest, refresh)
    with zipfile.ZipFile(dest) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(member).decode("latin-1")
    return _parse_french_tables(text)


def french_table(name: str, title_prefix: str, refresh: bool = False) -> pd.DataFrame:
    tables = french(name, refresh=refresh)
    for title, df in tables.items():
        if title.startswith(title_prefix):
            return df
    raise KeyError(f"{title_prefix!r} not found in {name}: {list(tables)}")


def french_factors(refresh: bool = False) -> pd.DataFrame:
    """Fama-French 3 factors, monthly, in decimal (Mkt-RF, SMB, HML, RF)."""
    dest = RAW_DIR / "french" / "F-F_Research_Data_Factors_CSV.zip"
    _download(FRENCH_URL.format(name="F-F_Research_Data_Factors"), dest, refresh)
    with zipfile.ZipFile(dest) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(member).decode("latin-1")
    # The monthly block is the first table and has no title line before the
    # header, so parse it directly.
    rows = []
    header = None
    for line in text.splitlines():
        s = line.strip()
        if header is None and s.startswith(",Mkt-RF"):
            header = [h.strip() for h in s.split(",")[1:]]
            continue
        if header is not None:
            if re.match(r"^\d{6}\s*,", s):
                parts = [p.strip() for p in s.split(",")]
                rows.append(parts[:len(header) + 1])
            elif rows:
                break
    df = pd.DataFrame(rows, columns=["date"] + header)
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m")
    return df.apply(pd.to_numeric, errors="coerce") / 100.0


# ---------------------------------------------------------------- Nareit

def nareit_equity_reits(refresh: bool = False) -> pd.DataFrame:
    """FTSE Nareit All Equity REITs: total, price and income return (decimal)."""
    dest = RAW_DIR / "nareit" / "MonthlyHistoricalReturns.xls"
    _download(NAREIT_URL, dest, refresh)
    raw = pd.read_excel(dest, sheet_name="Index Data", header=None, engine="xlrd")
    # Locate the "All Equity REITs" block by its label in the header rows.
    label_row = raw.iloc[5]
    col = next(c for c in range(raw.shape[1]) if str(label_row[c]).strip() == "All Equity REITs")
    body = raw.iloc[8:].copy()
    dates = pd.to_datetime(body[0], errors="coerce")
    out = pd.DataFrame({
        "total": pd.to_numeric(body[col], errors="coerce") / 100.0,
        "price": pd.to_numeric(body[col + 2], errors="coerce") / 100.0,
        "income": pd.to_numeric(body[col + 4], errors="coerce") / 100.0,
        "dividend_yield": pd.to_numeric(body[col + 5], errors="coerce") / 100.0,
    })
    out.index = dates
    out = out[out.index.notna()]
    out.index = _to_month_start(out.index)
    return out.dropna(subset=["total"])


# ---------------------------------------------------------------- Gold

def gold(refresh: bool = False) -> pd.Series:
    """Monthly gold price in USD/oz (London fix average, datasets/gold-prices)."""
    dest = RAW_DIR / "gold" / "monthly.csv"
    _download(GOLD_URL, dest, refresh)
    df = pd.read_csv(dest)
    s = pd.Series(pd.to_numeric(df["Price"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["Date"], format="%Y-%m"), name="gold")
    s = s.dropna()
    s.index = _to_month_start(s.index)
    return s[~s.index.duplicated(keep="last")]


# ---------------------------------------------------------------- Shiller

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
SHILLER_COLUMNS = {"P": "price", "D": "dividend", "E": "earnings", "CPI": "cpi", "CAPE": "cape"}
XLS_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")      # an Excel 97-2003 workbook (OLE2 compound file)
SHILLER_ATTEMPTS, SHILLER_TIMEOUT = 2, 60          # optional source: give up sooner than the required ones


def shiller(refresh: bool = False, download: bool = True) -> pd.DataFrame | None:
    """Shiller's monthly S&P composite data (price, dividend, earnings, CPI, CAPE), or None.

    Read from data/raw/shiller/ie_data.xls. Like every other source, the file is downloaded when
    it is not cached yet (so a first build gets CAPE) and re-downloaded only with refresh=True;
    download=False reads the cache only (analysis code that must never go online). It is an
    optional source: any failure (no network, a body that is not an .xls workbook, a changed
    layout) returns None and the panel is built without CAPE instead of failing; nothing is
    cached then, so the next build tries again.
    """
    dest = RAW_DIR / "shiller" / "ie_data.xls"
    if download:
        try:
            _download(SHILLER_URL, dest, refresh, attempts=SHILLER_ATTEMPTS, timeout=SHILLER_TIMEOUT,
                      check=lambda body: body[:8] == XLS_MAGIC)
        except RuntimeError:
            pass
    if not dest.exists() or dest.stat().st_size == 0:
        return None
    try:
        raw = pd.read_excel(dest, sheet_name="Data", header=None, engine="xlrd")
        head = next(i for i in range(min(len(raw), 40)) if str(raw.iat[i, 0]).strip() == "Date")
        cols = {}
        for c in range(raw.shape[1]):
            name = str(raw.iat[head, c]).strip()
            if name in SHILLER_COLUMNS and SHILLER_COLUMNS[name] not in cols:
                cols[SHILLER_COLUMNS[name]] = c
        body = raw.iloc[head + 1:]
        stamp = pd.to_numeric(body[0], errors="coerce")
        body, stamp = body[stamp.notna()], stamp[stamp.notna()].to_numpy(float)
        year = np.floor(stamp).astype(int)
        month = np.rint((stamp - year) * 100).astype(int)     # 1871.01 = January, 1871.1 = October
        index = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))
        out = pd.DataFrame({k: pd.to_numeric(body[c], errors="coerce").to_numpy(float) for k, c in cols.items()},
                           index=pd.DatetimeIndex(index))
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out if {"price", "earnings", "cpi", "cape"} <= set(out.columns) else None
    except Exception:  # noqa: BLE001 - optional source: a layout change must not break the panel
        return None


def interp_quarterly_to_monthly(q: pd.Series) -> pd.Series:
    """Place each quarterly value on the middle month and interpolate linearly.

    The last quarter is carried flat to the end of that quarter. This is used
    only for slow-moving distribution and capex variables.
    """
    q = q.dropna()
    mid = q.copy()
    mid.index = mid.index + pd.DateOffset(months=1)
    end = q.index.max() + pd.DateOffset(months=2)
    idx = pd.date_range(q.index.min(), end, freq="MS")
    m = mid.reindex(idx).interpolate(limit_area="inside")
    return m.ffill().bfill()


def splice(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """Use `primary` where it exists; before it, chain `fallback` by ratio."""
    primary = primary.dropna()
    fallback = fallback.dropna()
    start = primary.index.min()
    overlap = fallback.index.intersection(primary.index)
    if len(overlap) == 0:
        return pd.concat([fallback[fallback.index < start], primary]).sort_index()
    ratio = primary.loc[overlap[0]] / fallback.loc[overlap[0]]
    early = fallback[fallback.index < start] * ratio
    return pd.concat([early, primary]).sort_index()


def splice_level(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """Like `splice` for rates: use primary where present, fallback before it."""
    primary = primary.dropna()
    early = fallback.dropna()
    early = early[early.index < primary.index.min()]
    return pd.concat([early, primary]).sort_index()


def safe_log_diff(s: pd.Series) -> pd.Series:
    return np.log(s).diff()
