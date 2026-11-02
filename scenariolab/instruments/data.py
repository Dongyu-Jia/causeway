"""The instrument data layer: loaders for data/raw/instruments/ (filled by import_raw, refreshed by refresh).

Every loader reads a file only through `path()`, which requires a provenance record in SOURCES.json and refuses
the excluded sources (T15). Parsing is plain data parsing (CSV, JSON, the N-PORT XML, the 13F TSV extracts); no
research code runs. Results are cached per process (lru_cache): call `clear_cache()` after a refresh.

Conventions: daily series on their observation dates; monthly frames on a monthly PeriodIndex; quarterly on a
quarterly PeriodIndex (or quarter-end dates where the research tables use them); returns decimal; rates, yields
and vol indexes percent unless a name says otherwise.

The complete-month rule (INSTRUMENTS_DESIGN §5.1, defect 3): a month is complete only when every source used
together has its last trading day (last_trading_day: the last weekday, skipping the two NYSE holidays that can
fall on one, Good Friday and Memorial Day). An incomplete month enters no fit, replay or start state. Example: on
2026-11-01 ProShares has NAVs to 10-30 but FRED's NASDAQ-100 stops at 10-29, so 2026-10 is incomplete (to 10-29
QLD / TQQQ are -7.2% / -11.2%, to 10-30 -13.7% / -20.5%: the earlier mismatch was a vintage, not an error).
"""
from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..data.sources import read_fred_csv, read_fred_json
from . import fees

ROOT = config.RAW_DIR / "instruments"
MANUAL = ROOT / "manual"                    # user-supplied files (data/raw/instruments/manual/<TICKER>.csv)


class DataMissing(FileNotFoundError):
    """A file the instrument layer needs is not in data/raw/instruments (run import_raw or refresh)."""


# ---------------------------------------------------------------- provenance-checked access

@lru_cache(maxsize=1)
def sources() -> dict:
    p = ROOT / "SOURCES.json"
    if not p.exists():
        return {"sources": {}, "files": {}, "excluded": []}
    return json.loads(p.read_text(encoding="utf-8"))


def path(rel: str) -> Path:
    """The file data/raw/instruments/<rel>, after checking its provenance record (T15)."""
    doc = sources()
    rec = doc.get("files", {}).get(rel)
    p = ROOT / rel
    if rec is None:
        if rel.startswith("manual/") and p.exists():       # the user's own files carry no record
            return p
        raise DataMissing(f"没有 {rel} 的来源记录（data/raw/instruments/SOURCES.json）：请先运行 "
                          "python -m scenariolab.instruments.import_raw 或 refresh")
    if rec.get("source") in set(doc.get("excluded", [])):
        raise PermissionError(f"{rel} 来自被排除的来源 {rec.get('source')}")
    if not p.exists():
        raise DataMissing(f"缺少文件 data/raw/instruments/{rel}")
    return p


def available(*rels: str) -> bool:
    try:
        for r in rels:
            path(r)
    except (DataMissing, PermissionError):
        return False
    return True


def clear_cache() -> None:
    for f in (sources, fred_daily, us_market_daily, cboe, proshares_nav, proshares_distributions, nport_tables,
              _13f_rows, _prices_13f, _derived, _ndx_rv_table):
        f.cache_clear()


# ---------------------------------------------------------------- calendar and the complete-month rule

def _easter(y: int) -> pd.Timestamp:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return pd.Timestamp(y, month, day)


def last_trading_day(month) -> pd.Timestamp:
    """The month's last NYSE trading day (the last weekday that is not Good Friday or Memorial Day)."""
    p = pd.Period(month, "M")
    d = p.to_timestamp(how="end").normalize()
    holidays = {_easter(p.year) - pd.Timedelta(days=2)}
    may31 = pd.Timestamp(p.year, 5, 31)
    holidays.add(may31 - pd.Timedelta(days=may31.weekday()))     # Memorial Day: the last Monday of May
    while d.weekday() >= 5 or d in holidays:
        d -= pd.Timedelta(days=1)
    return d


def month_complete(s: pd.Series, month) -> bool:
    """A daily series has the month's last trading day (or later, for a series dated on calendar days)."""
    p = pd.Period(month, "M")
    idx = pd.DatetimeIndex(s.dropna().index)
    inm = idx[(idx >= p.start_time) & (idx <= p.end_time)]
    return bool(len(inm)) and inm.max().normalize() >= last_trading_day(p)


def complete_months(*series: pd.Series, monthly: tuple[pd.Series, ...] = ()) -> pd.PeriodIndex:
    """Months complete in every daily series (month_complete) and present in every monthly one (PeriodIndex)."""
    sets = []
    for s in series:
        idx = pd.DatetimeIndex(s.dropna().index)
        months = pd.PeriodIndex(idx.to_period("M").unique(), freq="M")
        last = idx.groupby(idx.to_period("M"))
        sets.append({m for m in months if pd.Timestamp(last[m].max()).normalize() >= last_trading_day(m)})
    for s in monthly:
        sets.append(set(pd.PeriodIndex(s.dropna().index, freq="M")))
    if not sets:
        return pd.PeriodIndex([], freq="M")
    common = set.intersection(*sets)
    return pd.PeriodIndex(sorted(common), freq="M")


def month_end(s: pd.Series, complete: bool = True) -> pd.Series:
    """Last observation of every month (monthly PeriodIndex); with `complete`, only complete months."""
    s = s.dropna()
    m = s.groupby(s.index.to_period("M")).last()
    if complete:
        ok = complete_months(s)
        m = m[m.index.isin(ok)]
    return m


# ---------------------------------------------------------------- FRED, Ken French, Cboe

@lru_cache(maxsize=None)
def fred_daily(series_id: str) -> pd.Series:
    """A FRED series as imported or refreshed (data/raw/instruments/fred/<ID>.json, or the keyless graph CSV
    <ID>.csv), read by the project's FRED parsers."""
    if available(f"fred/{series_id}.json"):
        return read_fred_json(path(f"fred/{series_id}.json"), series_id).dropna()
    return read_fred_csv(path(f"fred/{series_id}.csv"), series_id).dropna()


@lru_cache(maxsize=1)
def us_market_daily() -> pd.Series:
    """Daily log total return of the US market (Ken French Mkt-RF + RF, CRSP), 1926-07 on."""
    p = path("french/F-F_Research_Data_Factors_daily_CSV.zip")
    with zipfile.ZipFile(p) as z:
        txt = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in txt.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append((parts[0], float(parts[1]), float(parts[4])))
    df = pd.DataFrame(rows, columns=["d", "mkt_rf", "rf"])
    s = pd.Series(np.log1p((df["mkt_rf"] + df["rf"]).to_numpy() / 100.0),
                  index=pd.to_datetime(df["d"], format="%Y%m%d"), name="us_market")
    return s.sort_index()


@lru_cache(maxsize=None)
def cboe(name: str, column: str = "CLOSE") -> pd.Series:
    """A Cboe daily history (VXN, VIX: CLOSE; BXN, BXNT, BXM: the index column)."""
    df = pd.read_csv(path(f"cboe/{name}_History.csv"))
    df.columns = [c.strip().upper() for c in df.columns]
    col = column.upper() if column.upper() in df.columns else name.upper()
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y"), name=name.lower())
    return s.dropna().sort_index()


def vxn_daily() -> pd.Series:
    """VXN close (percent): FRED VXNCLS from 2001, the Cboe file filling its gaps."""
    f = fred_daily("VXNCLS")
    try:
        return f.combine_first(cboe("VXN")).sort_index()
    except DataMissing:
        return f


# ---------------------------------------------------------------- NASDAQ-100 and realized-variance tables

def ndx_daily() -> pd.Series:
    return fred_daily("NASDAQ100")


def ndx_rv_table(complete: bool = True) -> pd.DataFrame:
    """A copy of the cached table (see _ndx_rv_table)."""
    t = _ndx_rv_table(complete)
    out = t.copy()
    out.attrs = dict(t.attrs)
    return out


@lru_cache(maxsize=2)
def _ndx_rv_table(complete: bool = True) -> pd.DataFrame:
    """Monthly realized variance of the nasdaq asset's index (rv_by_month): x (month-end log price return), rv
    (sum of squared daily log returns), rs3 (sum of cubes), n (days), source. NDX from 1985-11; before it the
    NASDAQ Composite scaled by the 1985-11..2014-12 NDX / Composite variance ratio (rv x r, rs3 x r^1.5,
    x x sqrt r), as the engine's nasdaq asset is spliced. Used by vol_source "drawn" and the vol fits."""
    def table(px):
        px = px.dropna()
        x = np.log(px).diff().dropna()
        k = x.index.to_period("M")
        g = pd.DataFrame({"x": x, "x2": x ** 2, "x3": x ** 3, "k": k}).groupby("k")
        me = px.groupby(px.index.to_period("M")).last()
        T = pd.DataFrame({"rv": g["x2"].sum(), "rs3": g["x3"].sum(), "n": g["x"].count()})
        T["x"] = np.log(me).diff().reindex(T.index)
        if complete:
            T = T[T.index.isin(complete_months(px))]
        return T
    N, C = table(ndx_daily()), table(fred_daily("NASDAQCOM"))
    ov = N.index.intersection(C.index)
    ov = ov[(ov >= pd.Period("1985-11", "M")) & (ov <= pd.Period("2014-12", "M"))]
    ratio = float(N.loc[ov, "rv"].mean() / C.loc[ov, "rv"].mean())
    good = N[(N["n"] >= 15) & N["x"].notna()].copy()
    good["source"] = "NASDAQ-100"
    early = C[~C.index.isin(good.index) & (C["n"] >= 15) & C["x"].notna()].copy()
    early["rv"] *= ratio
    early["rs3"] *= ratio ** 1.5
    early["x"] *= np.sqrt(ratio)
    early["source"] = "NASDAQ Composite x variance ratio"
    T = pd.concat([early, good]).sort_index()
    T.attrs["variance_ratio"] = ratio
    return T[["x", "rv", "rs3", "n", "source"]]


def mkt_rv(freq: str = "M", complete: bool = True) -> pd.DataFrame:
    """US market realized variance per period (Ken French daily): rv (sum of squared daily log returns), rv_std_ann
    (std of the daily log returns x sqrt 252: instr-mapped's rv_mkt), n. Monthly PeriodIndex or quarterly."""
    x = us_market_daily()
    if complete:
        ok = complete_months(x)
        x = x[x.index.to_period("M").isin(ok)]
    k = x.index.to_period(freq)
    g = x.groupby(k)
    out = pd.DataFrame({"rv": (x ** 2).groupby(k).sum(), "rv_std_ann": g.std() * np.sqrt(252), "n": g.count()})
    if freq == "Q" and complete:
        months = x.index.to_period("M").unique()
        q_full = pd.Series(months.asfreq("Q")).value_counts()
        out = out[out.index.isin(q_full[q_full == 3].index)]
    return out


# ---------------------------------------------------------------- QQQ's dividend yield (the NDX dividend input)

def qqq_fy() -> dict[int, float]:
    """QQQ fiscal-year (Oct-Sep) gross dividend yield = net investment income ratio + expense ratio (annual
    reports, EDGAR); FY2004 was not parsed: the mean of FY2003 and FY2006 (as instr-structural)."""
    fy = derived("qqq_fy.csv")
    dy = {int(r.fy): float(r.gross_div_yield) for r in fy.itertuples()
          if pd.notna(r.gross_div_yield) and r.gross_div_yield > 0}
    if 2003 in dy and 2006 in dy:
        dy[2004] = (dy[2003] + dy[2006]) / 2
    return dy


def ndx_div_yield(dates) -> np.ndarray:
    """The NDX dividend yield at each date (QQQ's fiscal year; the nearest year outside the table)."""
    dy = qqq_fy()
    lo, hi = min(dy), max(dy)
    out = []
    for d in pd.DatetimeIndex(pd.PeriodIndex(dates, freq="M").to_timestamp() if isinstance(dates, pd.PeriodIndex)
                              else dates):
        f = d.year + 1 if d.month >= 10 else d.year
        out.append(dy.get(f, dy[hi] if f > hi else dy[lo]))
    return np.asarray(out, float)


# ---------------------------------------------------------------- ProShares (QLD, TQQQ)

@lru_cache(maxsize=None)
def proshares_nav(ticker: str) -> pd.Series:
    """Daily NAV per share (split-adjusted, ex-distribution, as ProShares publishes it)."""
    df = pd.read_csv(path(f"proshares/{ticker}-historical_nav.csv"))
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    return df["NAV"].astype(float).rename(f"{ticker.lower()}_nav")


@lru_cache(maxsize=None)
def proshares_distributions(ticker: str) -> pd.Series:
    """Cash per share by ex-date from the distribution API answers (already on the NAV file's share basis)."""
    rows = []
    rels = sorted(k for k in sources().get("files", {}) if k.startswith(f"proshares/distributions/{ticker}_"))
    for rel in rels:
        for r in json.loads(path(rel).read_text()):
            amt = sum(float(r.get(k) or 0.0) for k in ("CashDividendPerShare", "ShortTermCapGains", "LongTermCapGains",
                                                         "ReturnOfCapital", "SpecialPerShare"))
            rows.append((pd.Timestamp(r["ExDate"][:10]), amt))
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows, columns=["ex", "amt"]).groupby("ex")["amt"].sum().sort_index()


# Distributions missing from the ProShares API, which starts in 2013 for QLD (defect 3): QLD's NAV drops by
# 6.07% / 5.13% on these year-end ex-dates against a 2 x NDX day (daily tracking sd 0.02%).
IMPUTED_DISTRIBUTIONS = {"QLD": ("2006-12-20", "2007-12-20")}


def imputed_distributions(ticker: str) -> pd.Series:
    """The missing distributions per share: the one-day residual of the NAV against L x the NDX price return less
    one day of carry (fed funds / 360 x (L - 1), the fee / 365, plus L x the NDX dividend yield / 365)."""
    days = IMPUTED_DISTRIBUTIONS.get(ticker, ())
    if not days:
        return pd.Series(dtype=float)
    L = {"QLD": 2, "TQQQ": 3}[ticker]
    nav, ndx, dff = proshares_nav(ticker), ndx_daily(), fred_daily("DFF")
    out = {}
    for d in days:
        d = pd.Timestamp(d)
        i = nav.index.get_loc(d)
        p0 = nav.index[i - 1]
        n_days = (d - p0).days
        R = ndx.loc[:d].iloc[-1] / ndx.loc[:p0].iloc[-1] - 1
        b = float(dff.loc[:p0].iloc[-1]) / 100
        q = float(ndx_div_yield(pd.DatetimeIndex([p0]))[0])
        expected = (L * R - (L - 1) * b * n_days / 360 - fees.fee(ticker, p0) * n_days / 365
                    + L * q * n_days / 365)
        actual = nav.loc[d] / nav.loc[p0] - 1
        out[d] = float((expected - actual) * nav.loc[p0])
    return pd.Series(out, dtype=float)


def proshares_total_return(ticker: str) -> pd.Series:
    """Daily NAV total-return index: (NAV_t + D_t) / NAV_t-1 chained, the API's distributions plus the imputed
    QLD 2006-12 / 2007-12 ones; an ex-date that is not a NAV day goes to the next NAV day."""
    nav = proshares_nav(ticker)
    dist = proshares_distributions(ticker)
    imp = imputed_distributions(ticker)
    if len(imp):
        dist = pd.concat([dist, imp]).groupby(level=0).sum().sort_index()
    d_on = dist.reindex(nav.index).fillna(0.0)
    for ex in [e for e in dist.index if e not in nav.index]:
        nxt = nav.index[nav.index > ex]
        if len(nxt):
            d_on.loc[nxt[0]] += dist.loc[ex]
    return ((nav + d_on) / nav.shift(1)).fillna(1.0).cumprod().rename(f"{ticker.lower()}_tr")


def letf_monthly(ticker: str, complete_with=None) -> pd.Series:
    """Monthly NAV total returns (PeriodIndex) over months complete in the fund's NAV file and in every series of
    `complete_with` (default: the FRED NASDAQ-100 the replay is driven by)."""
    tr = proshares_total_return(ticker)
    others = (ndx_daily(),) if complete_with is None else tuple(complete_with)
    ok = complete_months(tr, *others)
    m = tr.groupby(tr.index.to_period("M")).last().pct_change()
    return m[m.index.isin(ok)].dropna()


# ---------------------------------------------------------------- SEC N-PORT

def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
        for k in list(el.attrib):
            if "}" in k:
                el.attrib[k.split("}", 1)[1]] = el.attrib.pop(k)
    return root


def _txt(el, p, default=None):
    x = el.find(p) if el is not None else None
    return default if x is None or x.text is None else x.text.strip()


def _num(el, p):
    try:
        return float(_txt(el, p))
    except (TypeError, ValueError):
        return None


def _month_add(ym: str, k: int) -> str:
    return str(pd.Period(ym, "M") + k)


# QYLD (Global X) files rtn1..rtn3 in reverse order (rtn1 = the report month): reversed, its months correlate 0.987
# with the BXNT index, as filed 0.29. Every other fund follows the form (QLD / TQQQ match their daily NAVs to 1e-5).
NPORT_REVERSED = {"QYLD"}


@lru_cache(maxsize=1)
def nport_tables() -> dict[str, pd.DataFrame]:
    """Every imported N-PORT-P document parsed: returns (fund, rep_date, month, rtn), bonds (TLT debt holdings),
    options (written / purchased calls: strike, expiry, contracts), equity (equity value, net assets)."""
    rets, bonds, opts, eqs = [], [], [], []
    for rel in sorted(k for k in sources().get("files", {}) if k.startswith("sec/nport/")):
        fund = Path(rel).stem.split("_")[0]
        try:
            root = _strip_ns(ET.parse(path(rel)).getroot())
        except ET.ParseError:
            continue
        gi, fi = root.find(".//genInfo"), root.find(".//fundInfo")
        rep = _txt(gi, "repPdDate")
        if rep is None:
            continue
        na = _num(fi, "netAssets")
        keys = ("rtn3", "rtn2", "rtn1") if fund in NPORT_REVERSED else ("rtn1", "rtn2", "rtn3")
        for mt in root.iter("monthlyTotReturn"):
            for k, key in enumerate(keys):
                v = mt.attrib.get(key)
                if v not in (None, "", "N/A"):
                    rets.append((fund, rep, _month_add(rep[:7], k - 2), float(v) / 100.0, na, Path(rel).name))
        eq_val = 0.0
        for inv in root.iter("invstOrSec"):
            val = _num(inv, "valUSD") or 0.0
            bal = _num(inv, "balance")
            cat = _txt(inv, "assetCat") or ""
            ds = inv.find("debtSec")
            if ds is not None:
                bonds.append((fund, rep, _txt(inv, "cusip"), _txt(inv, "title", ""), _txt(ds, "maturityDt"),
                              _num(ds, "annualizedRt"), bal, val, _txt(inv, "issuerCat")))
            if cat == "EC":
                eq_val += val
            di = inv.find("derivativeInfo")
            o = di.find("optionSwaptionWarrantDeriv") if di is not None else None
            if o is not None:
                opts.append((fund, rep, _txt(inv, "title", ""), _txt(o, "putOrCall"), _txt(o, "writtenOrPur"),
                             _num(o, "exercisePrice"), _txt(o, "expDt"), bal, _num(o, "shareNo"), val,
                             _txt(o, ".//indexName")))
        eqs.append((fund, rep, eq_val, na))
    R = pd.DataFrame(rets, columns=["fund", "rep_date", "month", "rtn", "net_assets", "file"])
    return {"returns": R,
            "bonds": pd.DataFrame(bonds, columns=["fund", "rep_date", "cusip", "title", "maturity", "coupon", "par",
                                                  "val_usd", "issuer_cat"]),
            "options": pd.DataFrame(opts, columns=["fund", "rep_date", "title", "put_call", "written_pur", "strike",
                                                   "expiry", "contracts", "multiplier", "val_usd", "index_name"]),
            "equity": pd.DataFrame(eqs, columns=["fund", "rep_date", "equity_val", "net_assets"])}


def nport_returns(fund: str) -> pd.Series:
    """Monthly NAV total returns filed on N-PORT (PeriodIndex; a month filed twice keeps the later report)."""
    R = nport_tables()["returns"]
    R = R[R.fund == fund].sort_values("rep_date").drop_duplicates("month", keep="last")
    return pd.Series(R["rtn"].to_numpy(), index=pd.PeriodIndex(R["month"], freq="M"), name=fund).sort_index()


# ---------------------------------------------------------------- SEC 13F quarter-end prices (2013 on)

# ticker -> CUSIPs (instr-mapped s06); the reference securities are kept for cross-checks
CUSIPS_13F = {
    "XLG": ["46137V233", "78355W205", "18383M100", "18385P101"],
    "MAIN": ["56035L104"], "ARCC": ["04010L103"], "CSWC": ["140501107"], "MPLX": ["55336V100"],
    "EPD": ["293792107"], "SCHD": ["808524797"], "MO": ["02209S103"], "BME": ["09250W107"],
    "BRK.B": ["084670702"], "QQQ": ["46090E103", "73935A104"], "QLD": ["74347R206"], "TQQQ": ["74347X831"],
    "QQQI": ["78433H675"], "TLT": ["464287432"], "GLD": ["78463V107"], "QYLD": ["37954Y483"], "SPY": ["78462F103"],
}
SPLIT_FACTORS = (1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50)


@lru_cache(maxsize=1)
def _13f_rows() -> pd.DataFrame:
    """The 13F information-table rows of the instruments: original 13F-HR, SH, no options, value in dollars
    (thousands before 2023-01-03), quarter-end periods only."""
    files = sources().get("files", {})
    infos = []
    for rel in sorted(k for k in files if k.startswith("sec/13f/info_")):
        stem = Path(rel).name[5:-7]
        srel = f"sec/13f/sub_{stem}.tsv.gz"
        if srel not in files:
            continue
        with gzip.open(path(rel), "rt", encoding="utf-8") as f:
            a = pd.read_csv(f, sep="\t", dtype=str, quoting=3, keep_default_na=False)
        with gzip.open(path(srel), "rt", encoding="utf-8") as f:
            b = pd.read_csv(f, sep="\t", dtype=str, quoting=3, keep_default_na=False)
        infos.append(a.merge(b.drop_duplicates("ACCESSION_NUMBER"), on="ACCESSION_NUMBER", how="left"))
    if not infos:
        raise DataMissing("没有 13F 数据（data/raw/instruments/sec/13f）")
    df = pd.concat(infos, ignore_index=True)
    df = df.drop_duplicates()
    df["CUSIP"] = df["CUSIP"].str.strip().str.upper().str[:9]
    df = df[(df["SUBMISSIONTYPE"] == "13F-HR") & (df["ISAMENDMENT"].str.upper() != "Y")]
    df = df[(df["SSHPRNAMTTYPE"].str.upper() == "SH") & (df["PUTCALL"].str.strip() == "")]
    df["value"] = pd.to_numeric(df["VALUE"].str.replace(",", ""), errors="coerce")
    df["shares"] = pd.to_numeric(df["SSHPRNAMT"].str.replace(",", ""), errors="coerce")
    df = df[(df["value"] > 0) & (df["shares"] > 0)]
    df["filed"] = pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    df["period"] = pd.to_datetime(df["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["filed", "period"])
    df["usd"] = np.where(df["filed"] >= pd.Timestamp("2023-01-03"), df["value"], df["value"] * 1000.0)
    return df[df["period"].dt.is_quarter_end & df["CUSIP"].isin({c for v in CUSIPS_13F.values() for c in v})]


def _central_price(g: pd.DataFrame) -> tuple[float, int]:
    f = g.groupby("CIK").agg(usd=("usd", "sum"), sh=("shares", "sum"))
    f = f[f["usd"] >= 2e5]
    if len(f) < 3:
        return np.nan, len(f)
    px = f["usd"] / f["sh"]
    m = px.median()
    px = px[(px > m / 30) & (px < m * 30)]
    m = px.median()
    w = px[(px - m).abs() <= 0.03 * m]
    if len(w) < 3:
        return np.nan, len(w)
    return float(w.median()), int(len(w))


def prices_13f() -> pd.DataFrame:
    """A copy of _prices_13f() (callers may change it)."""
    return _prices_13f().copy()


@lru_cache(maxsize=1)
def _prices_13f() -> pd.DataFrame:
    """Split-adjusted quarter-end prices (today's share basis) from the 13F rows: per filer value / shares, the
    median over filers with >= $200k, then the median within +-3% of it; a k:1 split between two quarters when the
    median share ratio of common filers is within 6% of k and the price ratio within 25% of 1/k (instr-mapped
    s06; the maximum error against filed market prices was 0.07%). Index: quarter-end dates."""
    df = _13f_rows()
    periods = sorted(df["period"].unique())
    adj = {}
    for t, cus in CUSIPS_13F.items():
        sub = df[df["CUSIP"].isin(cus)]
        ser = {}
        for p in periods:
            g = sub[sub["period"] == p]
            if g.empty:
                continue
            c = g.groupby("CUSIP")["CIK"].nunique().idxmax()
            ser[p] = _central_price(g[g["CUSIP"] == c])[0]
        s = pd.Series(ser, dtype=float).sort_index()
        f = pd.Series(1.0, index=s.index)
        v = s.dropna()
        for p0, p1 in zip(v.index[:-1], v.index[1:]):
            a = sub[sub["period"] == p0].groupby("CIK")["shares"].sum()
            b = sub[sub["period"] == p1].groupby("CIK")["shares"].sum()
            j = pd.concat([a, b], axis=1, join="inner").dropna()
            j = j[(j.iloc[:, 0] > 0) & (j.iloc[:, 1] > 0)]
            if len(j) < 10:
                continue
            sr, pr = float(np.median(j.iloc[:, 1] / j.iloc[:, 0])), v[p1] / v[p0]
            for k in SPLIT_FACTORS:
                for kk in (k, 1.0 / k):
                    if abs(sr / kk - 1) < 0.06 and abs(pr * kk - 1) < 0.25:
                        f[f.index < p1] *= kk
        adj[t] = s / f
    A = pd.DataFrame(adj)
    A.index = pd.DatetimeIndex(A.index, name="qend")
    return A


def prices_13f_table() -> pd.DataFrame | None:
    """The 13F quarter-end price table: derived/p13f_adj.csv (written by the research round, rebuilt from the
    extracts by refresh after a new data set), else computed from the extracts; None without either."""
    if available("derived/p13f_adj.csv"):
        t = derived("p13f_adj.csv").copy()
        t.index = pd.DatetimeIndex(pd.to_datetime(t.pop("qend")), name="qend")
        return t
    if available("sec/13f/manifest.jsonl"):
        return prices_13f()
    return None


# ---------------------------------------------------------------- research-derived tables

@lru_cache(maxsize=None)
def _derived(name: str) -> pd.DataFrame:
    return pd.read_csv(path(f"derived/{name}"))


def derived(name: str) -> pd.DataFrame:
    """A research-derived table (a copy: callers may change it)."""
    return _derived(name).copy()


def instr_quarterly(ticker: str | None = None) -> pd.DataFrame:
    """Ticker x quarter-end: P (13F price), D (cash per share with ex-date in the quarter), y = D / P-1,
    lr (log total return), lp (log price return), nav, nii, div_src."""
    d = derived("instr_quarterly.csv").copy()
    d["qend"] = pd.to_datetime(d["qend"])
    return d if ticker is None else d[d["ticker"] == ticker].set_index("qend").sort_index()


# ---------------------------------------------------------------- the core's quarterly dataset (estimated recipes)

CORE_ASSETS = list(config.ASSET_IDS)


def core_quarterly(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """The core panel aggregated to calendar quarters (instr-mapped's core_q, rebuilt from the project's panel):
    lr_/lp_/inc_<asset> (sum of monthly log total / log price / income returns), ff_avg, ff_end, tb3_end, y10_end,
    baa10_end (Baa - 10y, pp), gz_end (FRB GZ spread), vix_avg, rv_mkt (std of daily US market log returns x
    sqrt 252), rv_mkt_ss (sum of squares), and the changes d_baa10, d_gz, d_y10, d_ff, d_tb3. Index: quarter-end
    dates; only whole quarters."""
    from ..data.panel import load_panel
    P = (panel if panel is not None else load_panel()).loc["1960-01":]
    q = pd.PeriodIndex(pd.DatetimeIndex(P.index), freq="Q")
    out = {}
    for a in CORE_ASSETS:
        out[f"lr_{a}"] = np.log1p(P[f"r_{a}"].astype(float)).groupby(q).sum(min_count=3)
        out[f"lp_{a}"] = np.log1p(P[f"p_{a}"].astype(float)).groupby(q).sum(min_count=3)
        out[f"inc_{a}"] = P[f"d_{a}"].astype(float).groupby(q).sum(min_count=3)
    last = lambda c: P[c].astype(float).groupby(q).last()          # noqa: E731
    out["ff_avg"] = P["lvl_ff"].astype(float).groupby(q).mean()
    out["ff_end"] = last("lvl_ff")
    out["tb3_end"] = last("lvl_tb3")
    out["y10_end"] = last("lvl_y10")
    out["baa10_end"] = last("lvl_baa") - last("lvl_y10")
    df = pd.DataFrame(out)
    n_months = pd.Series(1, index=q).groupby(level=0).count()
    df = df[df.index.isin(n_months[n_months == 3].index)]
    gz = Path(config.RAW_DIR) / "federalreserve" / "ebp_csv.csv"
    if gz.exists():
        e = pd.read_csv(gz, parse_dates=["date"]).set_index("date")
        df["gz_end"] = e["gz_spread"].groupby(pd.PeriodIndex(e.index, freq="Q")).last().reindex(df.index)
    try:
        v = cboe("VIX")
        df["vix_avg"] = v.groupby(pd.PeriodIndex(v.index, freq="Q")).mean().reindex(df.index)
    except DataMissing:
        pass
    try:
        rq = mkt_rv("Q")
        df["rv_mkt"] = rq["rv_std_ann"].reindex(df.index)
        df["rv_mkt_ss"] = rq["rv"].reindex(df.index)
    except DataMissing:
        pass
    for c, d in (("baa10_end", "d_baa10"), ("gz_end", "d_gz"), ("y10_end", "d_y10"), ("tb3_end", "d_tb3")):
        if c in df.columns:
            df[d] = df[c].diff()
    df["d_ff"] = df["ff_avg"].diff()
    df.index = df.index.to_timestamp(how="end").normalize()
    df.index.name = "qend"
    return df


def manual_prices(ticker: str) -> pd.DataFrame | None:
    """Optional user prices: data/raw/instruments/manual/<TICKER>.csv with date, close[, adj_close][, dividend]."""
    p = MANUAL / f"{ticker}.csv"
    if not p.exists():
        return None
    f = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    f["adj_close"] = f["adj_close"] if "adj_close" in f.columns else f["close"]
    f["dividend"] = f["dividend"] if "dividend" in f.columns else 0.0
    return f[["close", "adj_close", "dividend"]]


def vintages() -> dict:
    """The last date of every main series (for instrument_meta.json["vintages"] and the UI's 数据截至)."""
    out = {}
    for name, fn in (("ndx", ndx_daily), ("vxn", vxn_daily), ("us_market_daily", us_market_daily),
                     ("dgs20", lambda: fred_daily("DGS20")), ("dgs30", lambda: fred_daily("DGS30")),
                     ("qld_nav", lambda: proshares_nav("QLD")), ("tqqq_nav", lambda: proshares_nav("TQQQ"))):
        try:
            out[name] = str(fn().index.max().date())
        except (DataMissing, KeyError, ValueError):
            out[name] = None
    try:
        out["prices_13f"] = str(instr_quarterly()["qend"][instr_quarterly()["P"].notna()].max().date())
    except (DataMissing, KeyError, ValueError):
        out["prices_13f"] = None
    try:
        r = nport_tables()["returns"]
        out["nport"] = str(r["month"].max()) if len(r) else None
    except DataMissing:
        out["nport"] = None
    return out
