"""Downloaders for the public data sources, with a plain-file cache.

FRED series come from the FRED API (api.stlouisfed.org) when FRED_API_KEY is set (env.load_dotenv() reads it
from the git-ignored .env) and from FRED's keyless graph CSV otherwise, or when the API fails (a rejected key,
an outage). Everything else is open: Ken French's
data library, Nareit's monthly workbook, the datasets/gold-prices CSV, Shiller's workbooks, and the publishers
of the missing-factor round (Federal Reserve Board GZ spread / excess bond premium, NY Fed ACM term premium and
HLW r*, World Bank Pink Sheet, Freddie Mac FMHPI, Caldara-Iacoviello GPR, Baker-Bloom-Davis EPU). Every download
is cached under data/raw/<source>/: a missing file is downloaded, a cached one is reused unless `refresh=True`.
`python -m scenariolab.data.import_raw` fills the same cache from the research round's downloads.

The FRED key travels only as the api_key query parameter to api.stlouisfed.org. It is never part of a file
name, a cached body (FRED does not echo it), a log line or an exception: requests' messages contain the full
URL with the query string, so they are never re-raised or printed, and logged URLs say api_key=REDACTED.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit

import numpy as np
import pandas as pd
import requests

from .. import __version__, config
from ..config import RAW_DIR

log = logging.getLogger(__name__)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = config.FRED["api_url"]                    # https://api.stlouisfed.org/fred
FRED_API_HOST = "api.stlouisfed.org"                 # the only host that ever receives the key
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
NAREIT_URL = "https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls"
GOLD_URL = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"

REPO_URL = "https://github.com/Dongyu-Jia/causeway"
USER_AGENT = f"causeway/{__version__} (+{REPO_URL})"

XLS_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")      # an Excel 97-2003 workbook (OLE2 compound file)
ZIP_MAGIC = b"PK\x03\x04"                           # .xlsx workbooks are zip archives
OPTIONAL_ATTEMPTS, OPTIONAL_TIMEOUT = 2, 60         # optional sources: give up sooner than the required ones


def _download(url: str, dest: Path, refresh: bool, params: dict | None = None, attempts: int = 4,
              timeout: float = 90, check: Callable[[bytes], bool] | None = None) -> Path:
    """dest, downloaded from url unless it is already cached (non-empty) and refresh is False.

    `check(content)` must accept the body before it is cached (a rejected body counts as a failed
    attempt); after `attempts` failures a RuntimeError is raised and nothing is written. Keyless URLs
    only: the FRED API goes through _fred_get, which never puts its URL in an error.
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


def _cached(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _atomic_write(dest: Path, body: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)


def _to_month_start(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx)).to_period("M").to_timestamp()


# ---------------------------------------------------------------- FRED

AGG_SUFFIX = {None: "", "eop": "_m_eop", "avg": "_m_avg"}    # API responses (the research cache's names)
CSV_SUFFIX = {None: "", "eop": "_eop", "avg": "_avg"}        # graph CSV downloads
_LAST_CALL: list = [None]                                    # time.monotonic() of this process's last API call


def fred_key() -> str | None:
    """FRED_API_KEY from the environment after env.load_dotenv() (the git-ignored .env), or None."""
    key = os.environ.get(config.FRED["key_env"], "").strip()
    return key or None


def redacted_url(path: str, params: dict) -> str:
    """An API request as it may be logged or recorded: the key is always api_key=REDACTED."""
    query = {k: v for k, v in params.items() if k not in ("api_key", "file_type")}
    return f"{FRED_API}{path}?{urlencode({**query, 'api_key': 'REDACTED', 'file_type': 'json'})}"


def _now() -> float:
    return time.monotonic()


def _throttle() -> None:
    """At least config.FRED["min_interval_s"] between two API calls of this process (FRED allows 120 a minute)."""
    gap = float(config.FRED["min_interval_s"])
    if _LAST_CALL[0] is not None:
        wait = gap - (_now() - _LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
    _LAST_CALL[0] = _now()


def _fred_get(path: str, params: dict, dest: Path, field: str, series_id: str) -> Path:
    """GET FRED_API + path with the key and write the body to dest once it holds a `field` list.

    HTTP 429 / 5xx, timeouts, connection errors and bodies without the list are retried
    (config.FRED["attempts"] in all, 1.5 k s apart); any other status fails at once. Every failure raises
    a RuntimeError that names only the series, the HTTP status and the exception type.
    """
    key = fred_key()
    if key is None:
        raise RuntimeError(f"FRED API 下载失败：{series_id}（没有设置 {config.FRED['key_env']}）")
    url = FRED_API + path
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != FRED_API_HOST:      # the key may go to this host only
        raise RuntimeError(f"FRED API 地址必须是 https://{FRED_API_HOST}：{series_id}")
    shown = redacted_url(path, params)
    attempts = int(config.FRED["attempts"])
    status, err_name = None, None
    for attempt in range(attempts):
        _throttle()
        status, err_name, retry = None, None, True
        try:
            resp = requests.get(url, params={**params, "api_key": key, "file_type": "json"}, timeout=90,
                                headers={"User-Agent": USER_AGENT}, allow_redirects=False)
            status = int(resp.status_code)
            if status == 200:
                body = resp.content
                try:
                    data = json.loads(body)
                except ValueError:
                    data = None
                if isinstance(data, dict) and isinstance(data.get(field), list) and key.encode() not in body:
                    _atomic_write(dest, body)
                    log.info("FRED API %s -> %s", shown, dest.name)
                    return dest
                err_name = "UnexpectedBody"
            else:
                err_name = "HTTPError"
                retry = status == 429 or status >= 500
        except (requests.Timeout, requests.ConnectionError) as err:
            err_name = type(err).__name__
        except Exception as err:  # noqa: BLE001 - never re-raised: its message may hold the URL with the key
            err_name, retry = type(err).__name__, False
        if not retry:
            break
        if attempt + 1 < attempts:
            time.sleep(1.5 * (attempt + 1))
    log.warning("FRED API failed: %s (HTTP %s / %s)", shown, status, err_name)
    raise RuntimeError(f"FRED API 下载失败：{series_id}（HTTP {status if status is not None else '—'} / {err_name}）")


def _fred_series(dates, values, name: str) -> pd.Series:
    """Both cache formats parse to this: float values ("." or empty = NaN), sorted by date."""
    s = pd.Series(pd.to_numeric(pd.Series(list(values), dtype=object), errors="coerce").to_numpy(float),
                  index=pd.DatetimeIndex(pd.to_datetime(list(dates))), name=name)
    return s[~s.index.duplicated(keep="last")].sort_index()


def read_fred_json(path: Path, series_id: str | None = None) -> pd.Series:
    """A cached API response (/fred/series/observations) as a Series on its observation dates."""
    obs = json.loads(Path(path).read_bytes())["observations"]
    return _fred_series([o["date"] for o in obs], [o["value"] for o in obs], series_id or Path(path).stem)


def read_fred_csv(path: Path, series_id: str | None = None) -> pd.Series:
    """A cached graph CSV (observation_date,<ID>) as a Series on its observation dates."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return _fred_series(df.iloc[:, 0], df.iloc[:, 1], series_id or Path(path).stem)


def _fred_csv_check(series_id: str) -> Callable[[bytes], bool]:
    heads = {f"observation_date,{series_id}".lower().encode(), f"date,{series_id}".lower().encode()}
    return lambda body: body.lstrip(b"\xef\xbb\xbf").split(b"\n", 1)[0].strip().lower() in heads


def _fred_csv_download(series_id: str, agg: str | None, jpath: Path, cpath: Path) -> Path:
    """The keyless graph CSV of one request (it never carries the key)."""
    params = {"id": series_id}
    if agg:
        params.update({"fq": "Monthly", "fam": agg})
    _download(FRED_URL, cpath, True, params=params, check=_fred_csv_check(series_id))
    if jpath.exists():
        # an older API copy of the same request would shadow the CSV just downloaded on the next build
        old = jpath.parent / "superseded" / jpath.name
        old.parent.mkdir(parents=True, exist_ok=True)
        jpath.replace(old)
    return cpath


def _fred_download(series_id: str, agg: str | None, jpath: Path, cpath: Path) -> Path:
    """Download one request in the configured mode: the API with a key, the graph CSV without one.

    When the API fails with a key set (FRED rejects the key, or the API is down) the keyless graph CSV is
    tried once, so a bad key in .env does not stop the build; the warning and any final error name only the
    series, the HTTP status and the exception type (never the key or a requests message)."""
    if not fred_key():
        return _fred_csv_download(series_id, agg, jpath, cpath)
    params = {"series_id": series_id}
    if agg:
        params.update({"frequency": "m", "aggregation_method": agg})
    try:
        return _fred_get("/series/observations", params, jpath, "observations", series_id)
    except RuntimeError as err:
        api_err = str(err)                # _fred_get's own message: series, status and exception type only
    log.warning("FRED API failed for %s; trying the keyless graph CSV", series_id)
    try:
        return _fred_csv_download(series_id, agg, jpath, cpath)
    except RuntimeError:
        pass                              # raised below, outside the handler: no chained requests message
    raise RuntimeError(f"{api_err}；graph CSV 也下载失败")


def fred(series_id: str, eop: bool = False, refresh: bool = False, agg: str | None = None) -> pd.Series:
    """One FRED series as a float Series; missing values ("." in FRED's files) are dropped.

    Monthly and quarterly series are indexed by their month (quarter) start; a weekly or daily series
    asked for without `agg` keeps its observation dates (panel.py takes the last value in the month or the
    month's mean). agg="eop" / "avg" (eop=True is agg="eop") asks FRED for the month-end / monthly average
    of a daily series, which lines market variables up with month-end asset returns.

    Cache data/raw/fred/: API responses <ID>.json, <ID>_m_eop.json, <ID>_m_avg.json and graph CSVs <ID>.csv,
    <ID>_eop.csv, <ID>_avg.csv. A request reads its JSON, else its CSV, else downloads: through the API when
    FRED_API_KEY is set (the graph CSV when the API fails), the graph CSV otherwise. refresh=True downloads
    again in that mode and reads what it downloaded (a CSV download moves the request's older JSON to
    fred/superseded/).
    """
    agg = "eop" if eop else agg
    if agg not in AGG_SUFFIX:
        raise ValueError(f"agg must be None, 'eop' or 'avg', not {agg!r}")
    fdir = RAW_DIR / "fred"
    jpath = fdir / f"{series_id}{AGG_SUFFIX[agg]}.json"
    cpath = fdir / f"{series_id}{CSV_SUFFIX[agg]}.csv"
    if refresh:
        path = _fred_download(series_id, agg, jpath, cpath)
    elif _cached(jpath):
        path = jpath
    elif _cached(cpath):
        path = cpath
    else:
        path = _fred_download(series_id, agg, jpath, cpath)
    s = (read_fred_json if path.suffix == ".json" else read_fred_csv)(path, series_id).dropna()
    if len(s) and (s.index.day == 1).all():      # FRED dates monthly and quarterly observations at the 1st
        s.index = _to_month_start(s.index)
    return s


def fred_meta(series_id: str, refresh: bool = False, download: bool = True) -> dict:
    """FRED /fred/series metadata (title, units, frequency, seasonal_adjustment, observation_start/end,
    last_updated, notes), cached as <ID>_meta.json; {} when it is not cached and cannot be fetched (no key,
    or download=False). copyright = the notes mention a copyright (provenance only)."""
    dest = RAW_DIR / "fred" / f"{series_id}_meta.json"
    if (refresh or not _cached(dest)) and download and fred_key():
        _fred_get("/series", {"series_id": series_id}, dest, "seriess", series_id)
    if not _cached(dest):
        return {}
    meta = (json.loads(dest.read_bytes()).get("seriess") or [{}])[0]
    out = {k: meta.get(k) for k in ("id", "title", "units", "frequency", "seasonal_adjustment",
                                    "observation_start", "observation_end", "last_updated", "notes")}
    out["copyright"] = bool(re.search(r"copyright|©", out.get("notes") or "", re.IGNORECASE))
    return out


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
SHILLER_ATTEMPTS, SHILLER_TIMEOUT = OPTIONAL_ATTEMPTS, OPTIONAL_TIMEOUT


def shiller(refresh: bool = False, download: bool = True) -> pd.DataFrame | None:
    """Shiller's monthly S&P composite data (price, dividend, earnings, CPI, CAPE), or None.

    Read from data/raw/shiller/ie_data.xls. Like every other source, the file is downloaded when
    it is not cached yet (so a first build gets CAPE) and re-downloaded only with refresh=True;
    download=False reads the cache only (analysis code that must never go online). It is an
    optional source: any failure (no network, a body that is not an .xls workbook, a changed
    layout) returns None and the panel is built without CAPE instead of failing; nothing is
    cached then, so the next build tries again.

    A refresh never replaces the cached workbook with one whose earnings end earlier: P0.3
    imported the 2026 workbook (E to 2026-06) while SHILLER_URL's copy had E only to 2023-06,
    and with the NIPA splice retired an older workbook would end CAPE (and the quarterly
    stack's s_cape) in 2023. The download goes to a side file; a cached workbook that parses
    is replaced only by one whose last earnings month is not earlier (an unreadable download
    never replaces it).
    """
    dest = RAW_DIR / "shiller" / "ie_data.xls"
    if download:
        if refresh and _cached(dest):
            side = dest.with_name("ie_data.download.xls")
            try:
                _download(SHILLER_URL, side, True, attempts=SHILLER_ATTEMPTS, timeout=SHILLER_TIMEOUT,
                          check=lambda body: body[:8] == XLS_MAGIC)
                old, new = _shiller_parse(dest), _shiller_parse(side)
                old_e = None if old is None else old["earnings"].last_valid_index()
                new_e = None if new is None else new["earnings"].last_valid_index()
                if old_e is not None and (new_e is None or new_e < old_e):
                    log.warning("shiller: the downloaded workbook's earnings end %s, before the cached one's %s; "
                                "the cached workbook is kept", new_e, old_e)
                else:
                    side.replace(dest)
            except RuntimeError:
                pass
            finally:
                if side.exists():
                    side.unlink()
        else:
            try:
                _download(SHILLER_URL, dest, refresh, attempts=SHILLER_ATTEMPTS, timeout=SHILLER_TIMEOUT,
                          check=lambda body: body[:8] == XLS_MAGIC)
            except RuntimeError:
                pass
    if not dest.exists() or dest.stat().st_size == 0:
        return None
    return _shiller_parse(dest)


def _shiller_parse(dest: Path) -> pd.DataFrame | None:
    """The workbook's Data sheet as a monthly frame, or None (a layout change must not break the panel)."""
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


# ---------------------------------------------------------------- publishers of the missing-factor round
# Each file is checked before it is cached (a moved file often answers with an HTML page). Required sources
# (they feed model inputs) raise when neither a cache nor a download is available, so the model never changes
# silently; optional ones (view variables) return None and the panel leaves their columns empty.

EBP_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
PINK_PAGE_URL = "https://www.worldbank.org/en/research/commodity-markets"
PINK_FILE = "CMO-Historical-Data-Monthly.xlsx"
FMHPI_URL = "https://www.freddiemac.com/fmac-resources/research/docs/fmhpi_master_file.csv"
SHILLER_HOME_URL = "http://www.econ.yale.edu/~shiller/data/Fig3-1.xls"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
EPU_URL = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"
EPU_HISTORICAL_URL = "https://www.policyuncertainty.com/media/US_Historical_EPU_data.xlsx"
HLW_URL = ("https://www.newyorkfed.org/medialibrary/media/research/economists/williams/data/"
           "Holston_Laubach_Williams_current_estimates.xlsx")
PINK_MONTH = re.compile(r"^\s*(\d{4})M(\d{2})\s*$")


def is_xls(body: bytes) -> bool:
    return body[:8] == XLS_MAGIC


def is_zip(body: bytes) -> bool:
    return body[:4] == ZIP_MAGIC


def is_ebp_csv(body: bytes) -> bool:
    return body.lstrip(b"\xef\xbb\xbf").lstrip().startswith(b"date,gz_spread,ebp")


def is_fmhpi_csv(body: bytes) -> bool:
    head = {h.strip().strip('"') for h in body.lstrip(b"\xef\xbb\xbf").split(b"\n", 1)[0].decode("latin-1").split(",")}
    return {"GEO_Type", "Year", "Month", "Index_SA"} <= head


def _required(url: str, dest: Path, refresh: bool, check: Callable[[bytes], bool], name: str) -> Path:
    try:
        return _download(url, dest, refresh, check=check)
    except RuntimeError:
        if _cached(dest):              # a failed refresh keeps the cached copy (the data do not change)
            log.warning("%s: download failed, using the cached %s", name, dest.name)
            return dest
        raise RuntimeError(f"必需的数据源不可用：{name}（{url}；没有缓存 {dest.name}）") from None


def _optional(url: str, dest: Path, refresh: bool, check: Callable[[bytes], bool]) -> Path | None:
    try:
        return _download(url, dest, refresh, check=check, attempts=OPTIONAL_ATTEMPTS, timeout=OPTIONAL_TIMEOUT)
    except RuntimeError:
        return dest if _cached(dest) else None


def _xls_rows(path: Path, sheet: str | int = 0) -> list[list]:
    """Cell values of one sheet of an .xls workbook (xlrd: data only, nothing evaluated); date cells as datetimes."""
    import xlrd
    wb = xlrd.open_workbook(str(path), on_demand=True)
    try:
        sh = wb.sheet_by_name(sheet) if isinstance(sheet, str) else wb.sheet_by_index(sheet)
        rows = []
        for r in range(sh.nrows):
            row = []
            for c in range(sh.ncols):
                cell = sh.cell(r, c)
                if cell.ctype == xlrd.XL_CELL_DATE:
                    row.append(xlrd.xldate_as_datetime(cell.value, wb.datemode))
                else:
                    row.append(cell.value)
            rows.append(row)
        return rows
    finally:
        wb.release_resources()


def _xlsx_rows(path: Path, sheet: str) -> list[tuple]:
    """Cell values of one sheet of an .xlsx workbook (openpyxl read-only, cached values; no macros, no formulas run)."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return [tuple(r) for r in wb[sheet].iter_rows(values_only=True)]
    finally:
        wb.close()


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _monthly(values: dict, name: str) -> pd.Series:
    s = pd.Series(values, dtype=float, name=name)
    s.index = _to_month_start(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()


def parse_ebp_csv(text: str) -> pd.DataFrame:
    """ebp_csv.csv: date, gz_spread, ebp (pp), est_prob -> gz_spread and ebp by month."""
    df = pd.read_csv(io.StringIO(text))
    if not {"date", "gz_spread", "ebp"} <= set(df.columns):
        raise ValueError(f"EBP csv: unexpected columns {list(df.columns)[:6]}")
    out = df[["gz_spread", "ebp"]].apply(pd.to_numeric, errors="coerce")
    out.index = _to_month_start(df["date"])
    return out[~out.index.duplicated(keep="last")].sort_index()


def parse_acm_rows(rows: list) -> pd.Series:
    """ACM Monthly sheet rows (header DATE, ..., ACMTP10, ...; dates like 30-Jun-1961) -> ACMTP10 by month."""
    head = [str(h).strip() for h in rows[0]]
    if "DATE" not in head or "ACMTP10" not in head:
        raise ValueError("ACM Monthly: DATE / ACMTP10 header not found")
    d, v = head.index("DATE"), head.index("ACMTP10")
    vals = {}
    for r in rows[1:]:
        if len(r) <= max(d, v) or r[d] in (None, ""):
            continue
        date = r[d] if hasattr(r[d], "year") else pd.to_datetime(str(r[d]).strip(), format="%d-%b-%Y")
        vals[pd.Timestamp(date)] = _num(r[v])
    return _monthly(vals, "ACMTP10").dropna()


def _pink_rows(rows: list, first_col: int, last_col: int) -> pd.DataFrame:
    data = {}
    for r in rows:
        m = PINK_MONTH.match(str(r[0])) if r and r[0] is not None else None
        if m:
            data[pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)] = [_num(x) for x in r[first_col:last_col + 1]]
    return pd.DataFrame.from_dict(data, orient="index").sort_index()


def parse_pink_sheet(prices: list, indices: list) -> pd.DataFrame:
    """Pink Sheet rows -> crude_avg ("Crude oil, average", $/bbl, sheet Monthly Prices) and base_metals ("Base
    Metals (ex. iron ore)", 2010 = 100, sheet Monthly Indices), monthly averages; columns found by their labels
    ("…" = missing)."""
    head = next((r for r in prices[:12] if any(str(c).strip() == "Crude oil, average" for c in r if c is not None)),
                None)
    if head is None:
        raise ValueError("Pink Sheet: 'Crude oil, average' not found in Monthly Prices")
    oil_col = [str(c).strip() if c is not None else "" for c in head].index("Crude oil, average")
    metal_col = None
    for j in range(max(len(r) for r in indices[5:9])):
        label = " ".join(str(r[j]) for r in indices[5:9] if j < len(r) and r[j] is not None)
        if "Base Metals (ex. iron ore)" in " ".join(label.split()):
            metal_col = j
            break
    if metal_col is None:
        raise ValueError("Pink Sheet: 'Base Metals (ex. iron ore)' not found in Monthly Indices rows 6-9")
    oil = _pink_rows(prices, oil_col, oil_col)
    metals = _pink_rows(indices, metal_col, metal_col)
    out = pd.DataFrame({"crude_avg": oil.iloc[:, 0] if len(oil) else pd.Series(dtype=float),
                        "base_metals": metals.iloc[:, 0] if len(metals) else pd.Series(dtype=float)})
    return out.sort_index()


def parse_fmhpi_csv(path_or_buffer) -> pd.Series:
    """fmhpi_master_file.csv -> the US seasonally adjusted index (Index_SA) by month."""
    df = pd.read_csv(path_or_buffer, usecols=["Year", "Month", "GEO_Type", "Index_SA"], dtype={"GEO_Type": str})
    us = df[df["GEO_Type"].str.strip() == "US"]
    idx = pd.to_datetime(pd.DataFrame({"year": us["Year"].astype(int), "month": us["Month"].astype(int), "day": 1}))
    s = pd.Series(pd.to_numeric(us["Index_SA"], errors="coerce").to_numpy(float), index=pd.DatetimeIndex(idx),
                  name="fmhpi_sa")
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


def parse_fig31_rows(rows: list) -> pd.Series:
    """Shiller Fig3-1.xls rows -> his nominal home price index (column 8) on the monthly decimal dates of column 7
    (1953.0417 = January 1953); the annual rows before 1953 are left out."""
    vals = {}
    for r in rows[7:]:
        if len(r) < 9 or not isinstance(r[7], float) or not isinstance(r[8], float) or r[7] < 1953:
            continue
        year = int(np.floor(r[7] + 1e-9))
        month = int(round((r[7] - year) * 12 + 0.5))
        if 1 <= month <= 12:
            vals[pd.Timestamp(year, month, 1)] = r[8]
    return _monthly(vals, "shiller_home").dropna()


def parse_gpr_rows(rows: list, column: str = "GPRH") -> pd.Series:
    """data_gpr_export.xls rows (header month, GPR, ..., GPRH, ...; the month column holds dates) -> `column`."""
    head = [str(h).strip() for h in rows[0]]
    if "month" not in head or column not in head:
        raise ValueError(f"GPR: 'month' / {column!r} header not found")
    d, v = head.index("month"), head.index(column)
    vals = {pd.Timestamp(r[d]): _num(r[v]) for r in rows[1:] if len(r) > max(d, v) and hasattr(r[d], "year")}
    return _monthly(vals, column).dropna()


def parse_epu_rows(rows: list, column: str = "News_Based_Policy_Uncert_Index") -> pd.Series:
    """US_Policy_Uncertainty_Data.xlsx, sheet Main News Index (Year, Month, `column`) -> the index by month."""
    head = [str(h).strip() if h is not None else "" for h in rows[0]]
    if not {"Year", "Month", column} <= set(head):
        raise ValueError(f"EPU: Year / Month / {column!r} header not found")
    y, m, v = head.index("Year"), head.index("Month"), head.index(column)
    vals = {}
    for r in rows[1:]:
        yy, mm = _num(r[y]) if len(r) > y else np.nan, _num(r[m]) if len(r) > m else np.nan
        if np.isfinite(yy) and np.isfinite(mm) and 1 <= mm <= 12:
            vals[pd.Timestamp(int(yy), int(mm), 1)] = _num(r[v])
    return _monthly(vals, "epu").dropna()


def parse_hlw_rows(rows: list) -> pd.Series:
    """HLW Estimates sheet rows -> the one-sided US r* (percent), quarterly, dated at the quarter's first month.
    The column is the one whose group label (the row above 'Date') names r* and whose country is US."""
    hdr = next((i for i, r in enumerate(rows[:20]) if r and str(r[0]).strip() == "Date"), None)
    if hdr is None or hdr == 0:
        raise ValueError("HLW: 'Date' header row not found")
    groups, countries = rows[hdr - 1], rows[hdr]
    col, group = None, ""
    for j in range(len(countries)):
        if j < len(groups) and groups[j]:
            group = str(groups[j])
        if "r*" in group and str(countries[j]).strip() == "US":
            col = j
            break
    if col is None:
        raise ValueError("HLW: the US r* column not found")
    vals = {pd.Timestamp(r[0]): _num(r[col]) for r in rows[hdr + 1:] if r and hasattr(r[0], "year") and len(r) > col}
    return _monthly(vals, "rstar").dropna()


def fed_ebp(refresh: bool = False) -> pd.DataFrame:
    """Federal Reserve Board GZ credit spread and excess bond premium (pp), monthly from 1973-01 (Favara,
    Gilchrist, Lewis and Zakrajsek; the whole history may be revised each month). Required: the GZ spread
    is a transmission candidate."""
    dest = _required(EBP_URL, RAW_DIR / "federalreserve" / "ebp_csv.csv", refresh, is_ebp_csv,
                     "美联储 GZ 信用利差与超额债券溢价")
    return parse_ebp_csv(dest.read_text(encoding="utf-8-sig"))


def nyfed_acm(refresh: bool = False) -> pd.Series:
    """NY Fed Adrian-Crump-Moench 10-year term premium ACMTP10 (pp), month-end, from 1961-06. Required (the
    high-core-inflation term premium is a transmission candidate)."""
    dest = _required(ACM_URL, RAW_DIR / "nyfed" / "ACMTermPremium.xls", refresh, is_xls, "纽约联储 ACM 期限溢价")
    return parse_acm_rows(_xls_rows(dest, "ACM Monthly"))


def _pink_sheet_url() -> str:
    """The current Pink Sheet link: the href ending CMO-Historical-Data-Monthly.xlsx on the commodity-markets
    page (the page is untrusted text: only searched for that link, which must stay on worldbank.org)."""
    resp = requests.get(PINK_PAGE_URL, timeout=OPTIONAL_TIMEOUT, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    for href in re.findall(r'href=["\']([^"\']+)["\']', resp.text):
        if href.split("?")[0].endswith(PINK_FILE):
            url = urljoin(PINK_PAGE_URL, href)
            host = urlsplit(url).hostname or ""
            if urlsplit(url).scheme == "https" and (host == "worldbank.org" or host.endswith(".worldbank.org")):
                return url
    raise RuntimeError(f"Pink Sheet link not found on {PINK_PAGE_URL}")


def worldbank_pinksheet(refresh: bool = False) -> pd.DataFrame:
    """World Bank Pink Sheet monthly averages: crude_avg (Crude oil, average, $/bbl) and base_metals (Base Metals
    ex. iron ore index). Required (base metals relative to oil is the energy equation's dedicated candidate)."""
    dest = RAW_DIR / "worldbank" / PINK_FILE
    if refresh or not _cached(dest):
        try:
            url = _pink_sheet_url()
        except Exception as err:  # noqa: BLE001 - no link: fall back to the cache, or fail below
            if not _cached(dest):
                raise RuntimeError(f"必需的数据源不可用：世界银行大宗商品价格（{PINK_PAGE_URL}：{err}）") from None
            url = None
        if url is not None:
            _required(url, dest, refresh, is_zip, "世界银行大宗商品价格（Pink Sheet）")
    return parse_pink_sheet(_xlsx_rows(dest, "Monthly Prices"), _xlsx_rows(dest, "Monthly Indices"))


def _optional_parse(path: Path | None, parse: Callable[[Path], pd.Series]) -> pd.Series | None:
    if path is None:
        return None
    try:
        s = parse(path)
        return s if len(s) else None
    except Exception:  # noqa: BLE001 - optional source: a layout change leaves its columns empty
        log.warning("optional source %s could not be read", path.name)
        return None


def freddiemac_fmhpi(refresh: bool = False) -> pd.Series | None:
    """Freddie Mac House Price Index, US, seasonally adjusted, monthly from 1975-01 (© Freddie Mac), or None."""
    dest = _optional(FMHPI_URL, RAW_DIR / "freddiemac" / "fmhpi_master_file.csv", refresh, is_fmhpi_csv)
    return _optional_parse(dest, parse_fmhpi_csv)


def shiller_home(refresh: bool = False) -> pd.Series | None:
    """Shiller's monthly nominal US home price index from 1953 (Fig3-1.xls; © Robert J. Shiller), or None. The
    house-price view uses it only for 1972-74, before the FMHPI starts."""
    dest = _optional(SHILLER_HOME_URL, RAW_DIR / "shiller" / "Fig3-1.xls", refresh, is_xls)
    return _optional_parse(dest, lambda p: parse_fig31_rows(_xls_rows(p, 0)))


def iacoviello_gpr(refresh: bool = False) -> pd.Series | None:
    """Caldara-Iacoviello historical geopolitical risk index GPRH, monthly from 1900, or None."""
    dest = _optional(GPR_URL, RAW_DIR / "iacoviello" / "data_gpr_export.xls", refresh, is_xls)
    return _optional_parse(dest, lambda p: parse_gpr_rows(_xls_rows(p, 0)))


def epu_news(refresh: bool = False) -> pd.Series | None:
    """Baker-Bloom-Davis news-based US economic policy uncertainty, monthly from 1900, or None.

    The Main News Index of US_Policy_Uncertainty_Data.xlsx is their historical newspaper index up to 1984
    (rescaled by a constant factor: identical by construction) and the 10-paper index from 1985; the historical
    workbook is kept for provenance only.
    """
    _optional(EPU_HISTORICAL_URL, RAW_DIR / "policyuncertainty" / "US_Historical_EPU_data.xlsx", refresh, is_zip)
    dest = _optional(EPU_URL, RAW_DIR / "policyuncertainty" / "US_Policy_Uncertainty_Data.xlsx", refresh, is_zip)
    return _optional_parse(dest, lambda p: parse_epu_rows(_xlsx_rows(p, "Main News Index")))


def nyfed_hlw(refresh: bool = False) -> pd.Series | None:
    """NY Fed Holston-Laubach-Williams one-sided US r* (percent, quarterly, dated at the quarter's first month), or
    None."""
    dest = _optional(HLW_URL, RAW_DIR / "nyfed" / "Holston_Laubach_Williams_current_estimates.xlsx", refresh, is_zip)
    return _optional_parse(dest, lambda p: parse_hlw_rows(_xlsx_rows(p, "HLW Estimates")))


# ---------------------------------------------------------------- series helpers

def interp_quarterly_to_monthly(q: pd.Series) -> pd.Series:
    """Place each quarterly value on the middle month and interpolate linearly.

    The last quarter is carried flat to the end of that quarter. A month then contains part of the next
    quarter (look-ahead): the panel uses it only for the audit column x_labor_share_interp, which rebuilds
    the research round's base model; every model and view column is release dated instead.
    """
    q = q.dropna()
    mid = q.copy()
    mid.index = mid.index + pd.DateOffset(months=1)
    end = q.index.max() + pd.DateOffset(months=2)
    idx = pd.date_range(q.index.min(), end, freq="MS")
    m = mid.reindex(idx).interpolate(limit_area="inside")
    return m.ffill().bfill()


def splice(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """Use `primary` where it exists; before it, chain `fallback` by ratio (no primary yet: the fallback)."""
    primary = primary.dropna()
    fallback = fallback.dropna()
    if primary.empty:
        return fallback.sort_index()
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
    if primary.empty:
        return early.sort_index()
    early = early[early.index < primary.index.min()]
    return pd.concat([early, primary]).sort_index()


def safe_log_diff(s: pd.Series) -> pd.Series:
    return np.log(s).diff()
