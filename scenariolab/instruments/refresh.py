"""Refresh the instrument data (INSTRUMENTS_DESIGN §5.5); explicit only, never run by a build or a test.

    python -m scenariolab.instruments.refresh --sources fred french cboe proshares nport 13f [--refresh]

Without --refresh a file that exists is never downloaded again (only missing ones are fetched; for the 13F data
sets and the N-PORT filings "missing" means a quarter / filing not in the local manifest yet). Afterwards the
derived 13F price table is rebuilt from the extracts (derived/p13f_adj.csv) and every new file gets a provenance
record in data/raw/instruments/SOURCES.json. Refitting is a separate, explicit step (fit.py).

Access rules:
  FRED       through the project's client (scenariolab.data.sources): the key goes only to api.stlouisfed.org as
             the api_key parameter, logged URLs say api_key=REDACTED; without a key the keyless graph CSV
  SEC        the User-Agent is SEC_USER_AGENT from the environment (env.load_dotenv reads the git-ignored .env),
             sent only to sec.gov hosts (redirects elsewhere are refused), never printed or written; <= 4 requests
             a second
  ProShares  the public all-funds NAV file (accounts.profunds.com), QLD / TQQQ rows kept; the distribution API
             answers are not refreshed automatically (new distributions are a few hundredths of a percent a year)
  Cboe       the daily history CSVs (cdn.cboe.com, redirect to cdn-api.cboe.com only)
  never      BlackRock / iShares, Invesco (terms forbid automated access), Stooq / Schwab (bot protection)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import re
import time
import urllib.parse
import zipfile
from pathlib import Path

import requests

from .. import jsonio
from ..data import sources as src
from . import data as D
from . import import_raw as IR

ROOT = D.ROOT
SEC_HOSTS = {"www.sec.gov", "sec.gov", "data.sec.gov", "efts.sec.gov"}
SEC_GAP_S = 0.25
CBOE_HOSTS = ("cdn.cboe.com", "cdn-api.cboe.com")
PROSHARES_URL = "https://accounts.profunds.com/etfdata/historical_nav.csv"
FORM13F_PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
NPORT_SERIES = {"TLT": "S000004360", "QLD": "S000006827", "TQQQ": "S000024908", "QYLD": "S000063352",
                "QQQI": "S000083903", "QQQ": "S000101292"}
_last_sec = [0.0]

# the 13F rows kept from each data set (instr-mapped s03): the known CUSIPs and the issuers' names
KNOWN_13F = {c for v in D.CUSIPS_13F.values() for c in v}
NAME_13F = re.compile(
    r"ARES CAP|MAIN STR|CAPITAL SOUTHW|MPLX|ENTERPRISE PROD|ALTRIA|BERKSHIRE HATH|BLACKROCK HEALTH|SCHWAB STR"
    r"|TOP 50|RUSSELL TOP|RYDEX|GUGGENHEIM|POWERSHARES QQQ|INVESCO QQQ|PROSHARES|NEOS ETF|SPDR GOLD"
    r"|GLOBAL X FD|SELECT SECTOR|SPDR S&P 500 ETF", re.I)


# ---------------------------------------------------------------- provenance

def _record(rel: str, url: str | None, source: str, kind: str, body: bytes, **extra) -> None:
    p = ROOT / "SOURCES.json"
    doc = jsonio.load(p) if p.exists() else {"version": 1, "sources": IR.SOURCES, "excluded": list(IR.EXCLUDED_SOURCES),
                                             "files": {}}
    doc.setdefault("files", {})[rel] = {"path": rel, "kind": kind, "source": source,
                                        "url": IR.redact(url) if url else None,
                                        "access_date": dt.date.today().isoformat(),
                                        "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                                        "generated_by": "scenariolab.instruments.refresh", **extra}
    jsonio.dump(doc, p)
    D.clear_cache()


def _write(rel: str, body: bytes) -> Path:
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return dest


def _have(rel: str) -> bool:
    p = ROOT / rel
    return p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------- SEC

def _sec_user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("SEC_USER_AGENT 没有设置（.env）：SEC 要求声明联系方式")
    return ua


def sec_get(url: str, timeout: float = 120.0) -> bytes:
    """GET a sec.gov URL with the declared User-Agent (no redirect off sec.gov; the UA is never logged)."""
    host = urllib.parse.urlsplit(url).hostname or ""
    if host not in SEC_HOSTS:
        raise ValueError(f"SEC 请求只发往 sec.gov：{host}")
    ua = _sec_user_agent()
    last_err = None
    for attempt in range(4):
        wait = SEC_GAP_S - (time.monotonic() - _last_sec[0])
        if wait > 0:
            time.sleep(wait)
        _last_sec[0] = time.monotonic()
        try:
            r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=timeout,
                             allow_redirects=False)
            if r.status_code in (301, 302, 303, 307, 308):
                nxt = urllib.parse.urljoin(url, r.headers.get("Location", ""))
                if (urllib.parse.urlsplit(nxt).hostname or "") not in SEC_HOSTS:
                    raise ValueError("拒绝跳转到 sec.gov 以外的主机")
                url = nxt
                continue
            if r.status_code == 200:
                return r.content
            last_err = f"HTTP {r.status_code}"
            if r.status_code in (403, 404):
                break
        except requests.RequestException as err:
            last_err = type(err).__name__          # never the message: no header or URL detail leaks
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"SEC 下载失败：{url}（{last_err}）")


def refresh_nport(funds=tuple(NPORT_SERIES), refresh: bool = False) -> list[str]:
    """New N-PORT-P filings of each fund (EDGAR atom feed per series) -> sec/nport/<FUND>_<filing date>.xml."""
    new = []
    for fund in funds:
        feed = sec_get("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="
                       f"{NPORT_SERIES[fund]}&type=NPORT-P&dateb=&owner=include&start=0&count=100&output=atom")
        text = feed.decode("utf-8", errors="replace")
        for date, href in zip(re.findall(r"<filing-date>([^<]+)</filing-date>", text),
                              re.findall(r"<filing-href>([^<]+)</filing-href>", text)):
            rel = f"sec/nport/{fund}_{date.strip()}.xml"
            if _have(rel) and not refresh:
                continue
            url = href.strip().rsplit("/", 1)[0] + "/primary_doc.xml"
            body = sec_get(url)
            _write(rel, body)
            _record(rel, url, "sec_edgar", "nport", body)
            new.append(rel)
    return new


def _extract_13f(body: bytes) -> tuple[list[list[str]], list[str], int]:
    """(matched info rows, submission + cover rows of their filings, rows scanned) of one 13F data-set zip."""
    z = zipfile.ZipFile(io.BytesIO(body))
    names = {n.upper().split("/")[-1]: n for n in z.namelist()}

    def table(member, cols):
        with z.open(names[member]) as f:
            rd = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            hdr = rd.readline().rstrip("\r\n").split("\t")
            ix = {h: i for i, h in enumerate(hdr)}
            return [[p[ix[c]] if c in ix and ix[c] < len(p) else "" for c in cols]
                    for p in (line.rstrip("\r\n").split("\t") for line in rd)]
    want = ["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]
    kept, n = [], 0
    with z.open(names["INFOTABLE.TSV"]) as f:
        rd = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        hdr = rd.readline().rstrip("\r\n").split("\t")
        ix = {h: i for i, h in enumerate(hdr)}
        for line in rd:
            n += 1
            p = line.rstrip("\r\n").split("\t")
            if len(p) < len(hdr):
                continue
            if p[ix["CUSIP"]].strip().upper()[:9] in KNOWN_13F or NAME_13F.search(p[ix["NAMEOFISSUER"]]):
                kept.append([p[ix[c]] if c in ix else "" for c in want])
    acc = {r[0] for r in kept}
    sub = table("SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
    cov = {r[0]: r[1:] for r in table("COVERPAGE.TSV", ["ACCESSION_NUMBER", "REPORTCALENDARORQUARTER",
                                                         "ISAMENDMENT", "AMENDMENTTYPE", "FILINGMANAGER_NAME"])}
    subs = ["\t".join(x.replace("\t", " ") for x in (r + cov.get(r[0], ["", "", "", ""]))) for r in sub if r[0] in acc]
    return [want] + kept, subs, n


def refresh_13f(refresh: bool = False) -> list[str]:
    """New Form 13F data sets: each zip is read in memory with the SEC User-Agent and only the instruments' rows
    are kept (the disk has no room for the 3 GB of zips), then derived/p13f_adj.csv is rebuilt."""
    page = sec_get(FORM13F_PAGE).decode("utf-8", errors="replace")
    links = sorted(set(re.findall(r'href="([^"]*13f[^"]*\.zip)"', page, flags=re.I)))
    man_rel = "sec/13f/manifest.jsonl"
    done = set()
    if _have(man_rel):
        done = {json.loads(x)["file"] for x in (ROOT / man_rel).read_text().splitlines() if x.strip()}
    new = []
    for link in links:
        fname = link.split("/")[-1]
        if fname in done and not refresh:
            continue
        url = urllib.parse.urljoin("https://www.sec.gov", link)
        body = sec_get(url, timeout=900)
        rows, subs, n = _extract_13f(body)
        stem = fname.replace("_form13f.zip", "")
        info = gzip.compress(("\n".join("\t".join(x.replace("\t", " ") for x in r) for r in rows) + "\n").encode())
        sub = gzip.compress(("ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT\t"
                             "REPORTCALENDARORQUARTER\tISAMENDMENT\tAMENDMENTTYPE\tFILINGMANAGER_NAME\n"
                             + "\n".join(subs) + "\n").encode())
        for rel, b in ((f"sec/13f/info_{stem}.tsv.gz", info), (f"sec/13f/sub_{stem}.tsv.gz", sub)):
            _write(rel, b)
            _record(rel, url, "sec_edgar", "13f_extract", b, zip_sha256=hashlib.sha256(body).hexdigest())
        line = json.dumps({"file": fname, "url": url, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                           "rows_scanned": n, "rows_kept": len(rows) - 1, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        with open(ROOT / man_rel, "a", encoding="utf-8") as m:
            m.write(line + "\n")
        _record(man_rel, None, "sec_edgar", "13f_manifest", (ROOT / man_rel).read_bytes())
        new.append(fname)
        del body
    if new:
        rebuild_13f_prices()
    return new


def rebuild_13f_prices() -> Path:
    """derived/p13f_adj.csv from the 13F extracts (data.prices_13f), with its provenance record."""
    D.clear_cache()
    A = D.prices_13f()
    body = A.to_csv(float_format="%.6g").encode()
    rel = "derived/p13f_adj.csv"
    _write(rel, body)
    _record(rel, None, "sec_edgar", "derived", body, produced_by="scenariolab.instruments.data.prices_13f",
            built_from="sec/13f extracts")
    return ROOT / rel


# ---------------------------------------------------------------- FRED, Ken French, Cboe, ProShares

def refresh_fred(series=IR.FRED_SERIES, refresh: bool = False) -> list[str]:
    """FRED series through the project's client into data/raw/instruments/fred/ (API JSON, or the graph CSV)."""
    new = []
    for sid in series:
        jpath, cpath = ROOT / "fred" / f"{sid}.json", ROOT / "fred" / f"{sid}.csv"
        if (_have(f"fred/{sid}.json") or _have(f"fred/{sid}.csv")) and not refresh:
            continue
        p = src._fred_download(sid, None, jpath, cpath)
        rel = f"fred/{p.name}"
        _record(rel, src.redacted_url("/series/observations", {"series_id": sid}) if p.suffix == ".json"
                else f"{src.FRED_URL}?id={sid}", "fred_api", "fred_observations", p.read_bytes(), series_id=sid)
        new.append(rel)
    return new


def refresh_french(refresh: bool = False) -> list[str]:
    name = "F-F_Research_Data_Factors_daily_CSV.zip"
    rel = f"french/{name}"
    if _have(rel) and not refresh:
        return []
    url = src.FRENCH_URL.format(name=name.replace("_CSV.zip", ""))
    p = src._download(url, ROOT / rel, refresh=True, check=lambda b: b[:4] == src.ZIP_MAGIC)
    _record(rel, url, "ken_french", "publisher", p.read_bytes())
    return [rel]


def _get_public(url: str, hosts: tuple[str, ...], timeout: float = 120.0) -> bytes:
    for _ in range(4):
        r = requests.get(url, headers={"User-Agent": src.USER_AGENT}, timeout=timeout, allow_redirects=False)
        if r.status_code in (301, 302, 303, 307, 308):
            url = urllib.parse.urljoin(url, r.headers.get("Location", ""))
            if (urllib.parse.urlsplit(url).hostname or "") not in hosts:
                raise ValueError(f"拒绝跳转到 {url}")
            continue
        r.raise_for_status()
        return r.content
    raise RuntimeError(f"too many redirects: {url}")


def refresh_cboe(names=("VXN", "VIX", "BXN", "BXNT", "BXM"), refresh: bool = False) -> list[str]:
    new = []
    for n in names:
        rel = f"cboe/{n}_History.csv"
        if _have(rel) and not refresh:
            continue
        url = IR.URL_TEMPLATES["cboe"].format(name=f"{n}_History.csv")
        body = _get_public(url, CBOE_HOSTS)
        _write(rel, body)
        _record(rel, url, "cboe", "publisher", body)
        new.append(rel)
    return new


def refresh_proshares(tickers=("QLD", "TQQQ"), refresh: bool = False) -> list[str]:
    rels = [f"proshares/{t}-historical_nav.csv" for t in tickers]
    if all(_have(r) for r in rels) and not refresh:
        return []
    body = _get_public(PROSHARES_URL, ("accounts.profunds.com",), timeout=300)
    lines = body.decode("utf-8", errors="replace").splitlines()
    hdr, rows = lines[0], lines[1:]
    for t, rel in zip(tickers, rels):
        keep = [r for r in rows if r.split(",")[2:3] == [t]] if "," in hdr else []
        b = ("\n".join([hdr] + keep) + "\n").encode()
        _write(rel, b)
        _record(rel, PROSHARES_URL, "proshares", "publisher", b, note=f"the {t} rows of the all-funds NAV file")
    return rels


SOURCES_FN = {"fred": refresh_fred, "french": refresh_french, "cboe": refresh_cboe, "proshares": refresh_proshares,
              "nport": refresh_nport, "13f": refresh_13f}


def main(argv=None) -> int:
    from ..env import load_dotenv
    ap = argparse.ArgumentParser(description="Refresh the instrument data (downloads; explicit only)")
    ap.add_argument("--sources", nargs="+", choices=sorted(SOURCES_FN), required=True)
    ap.add_argument("--refresh", action="store_true", help="download again even where a file exists")
    args = ap.parse_args(argv)
    load_dotenv()
    for s in args.sources:
        new = SOURCES_FN[s](refresh=args.refresh)
        print(f"[instruments.refresh] {s}: {len(new)} new file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
