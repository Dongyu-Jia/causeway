"""Downloaders for the public data sources, with a plain-file cache.

No API keys are needed: FRED's graph CSV endpoint, Ken French's data library,
Nareit's monthly workbook and the datasets/gold-prices CSV are all open.
Every download is cached under data/raw/<source>/ and reused unless
`refresh=True`.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import RAW_DIR

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
NAREIT_URL = "https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls"
GOLD_URL = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"

USER_AGENT = "macro-scenario-lab/0.1 (+https://github.com)"


def _download(url: str, dest: Path, refresh: bool, params: dict | None = None) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=90,
                                headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return dest
        except Exception as err:  # network hiccups are common on FRED
            last_err = err
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
