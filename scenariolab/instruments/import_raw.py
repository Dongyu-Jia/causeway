"""Import the instrument research round's downloads into data/raw/instruments/ (INSTRUMENTS_DESIGN §5); run once.

    python -m scenariolab.instruments.import_raw <instruments research dir> [--factors-raw DIR] [--force]

The research dir holds instr-data/, instr-structural/, instr-mapped/ (and instr-design/); --factors-raw is the
missing-factor round's raw folder (it has the Ken French daily market file the volatility state is fitted on).

Data only, as scenariolab.data.import_raw does it: bytes are copied, no research script is imported or executed,
and the manifests (download logs, sources.json files) are parsed as JSON / text data. What is imported (RULES):

  fred/<ID>.json (+ _meta)   FRED API responses (full unpaged requests only: data.import_raw.admissible)
  french/                    Ken French daily factors (the US market's daily returns -> rv_mkt)
  cboe/                      VXN, VIX, BXN, BXNT, BXM daily histories
  proshares/                 QLD / TQQQ daily NAV files, the split table, the distribution API answers
  sec/nport/                 N-PORT-P primary documents (QLD, TQQQ, QQQ, QQQI, QYLD, TLT)
  sec/13f/                   the 13F data sets' matched rows (the research extracted them in memory: info_*/sub_*
                             TSVs and the manifest with each original zip's URL and SHA-256)
  derived/                   tables the research rounds built from SEC filings (XBRL, N-CSR, N-30B-2), NEOS and the
                             13F rows, each with the script that produced it (provenance; never executed here)

Never imported: anything from issuers whose terms forbid automated access (BlackRock / iShares, Invesco), bot-
protection probes (Stooq, Schwab), terms pages and HTML probes, and any file that contains the FRED key or the
SEC User-Agent. Each file's SHA-256 is checked against the research manifests where they recorded one; an
existing different file is never overwritten without --force. Writes data/raw/instruments/SOURCES.json (the
source registry with the terms as read, and one provenance record per file) and import_log.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
from pathlib import Path

from .. import config, jsonio
from ..data.import_raw import _copy, admissible, redact
from ..env import load_dotenv

DEST = config.RAW_DIR / "instruments"
ACCESS_DATE = "2026-11-01"
API = "https://api.stlouisfed.org/fred"

# The source registry: terms as the research rounds read them on 2026-11-01 (instr-data/sources.json).
SOURCES = {
    "fred_api": {
        "publisher": "Federal Reserve Bank of St. Louis (FRED API)", "status": "used",
        "terms_url": "https://fred.stlouisfed.org/docs/api/terms_of_use.html",
        "terms": "API terms accepted with the key. Third-party series stay their owners' property ('Before using "
                 "data series owned by third parties for anything other than your own personal use, you must "
                 "contact the data owner'); Nasdaq index series carry 'Copyright (c) 2025, NASDAQ, Inc.', VXN/VIX "
                 "'Copyright, 2016, Chicago Board Options Exchange'. Personal research use; never redistributed.",
        "access": "the key goes only to api.stlouisfed.org (scenariolab.data.sources); recorded URLs say "
                  "api_key=REDACTED"},
    "ken_french": {
        "publisher": "Kenneth R. French Data Library (Tuck School, Dartmouth; built from CRSP)", "status": "used",
        "terms_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        "terms": "Free public research library; no separate terms of use; cite the library."},
    "cboe": {
        "publisher": "Cboe Global Markets (cdn.cboe.com -> cdn-api.cboe.com)", "status": "used",
        "terms_url": "https://www.cboe.com/terms",
        "terms": "'You may view, print and download one copy of the Materials for your personal non-commercial use'; "
                 "no copying, redistribution or derivative works. One copy each, local only."},
    "sec_edgar": {
        "publisher": "U.S. Securities and Exchange Commission (EDGAR, Form 13F data sets, XBRL)", "status": "used",
        "terms_url": "https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data",
        "terms": "Public U.S.-government data. Fair-access policy: a declared User-Agent with contact (SEC_USER_AGENT, "
                 "sent only to sec.gov hosts) and fewer than 10 requests a second."},
    "proshares": {
        "publisher": "ProShares (accounts.profunds.com)", "status": "used (not on the pre-approved list; D3 default)",
        "terms_url": "https://www.proshares.com/legal/terms-and-conditions",
        "terms": "Content 'provided for information, education and non-commercial purposes only'; no permission to "
                 "sell, copy, publish, reproduce, distribute, mirror or frame; no clause on automated access. "
                 "Downloaded once for personal non-commercial research; local only, never redistributed."},
    "neos": {
        "publisher": "NEOS Investments (neosfunds.com)", "status": "used",
        "terms_url": None,
        "terms": "No terms-of-use page found (the QQQI page links only a privacy policy); public fund page and "
                 "shareholder reports fetched once each for personal research."},
    "research_derived": {
        "publisher": "Causeway instrument research rounds (instr-data, instr-structural, instr-mapped)",
        "status": "used (tables built from the sources above; the producing scripts are named per file)",
        "terms_url": None,
        "terms": "Each table inherits the terms of the sources it was built from (named per file)."},
    # never used: recorded so the registry says why (T15)
    "ishares_blackrock": {"publisher": "BlackRock / iShares", "status": "EXCLUDED (terms forbid automated access)",
                          "terms_url": "https://www.blackrock.com/corporate/compliance/terms-and-conditions",
                          "terms": "Prohibits 'any robot, spider, intelligent agent, other automatic device, or "
                                   "manual process to search, monitor or copy' the site or its data."},
    "invesco": {"publisher": "Invesco", "status": "EXCLUDED (terms forbid automated access)",
                "terms_url": "https://www.invesco.com/us/en/resources/terms-of-use.html",
                "terms": "Prohibits 'data mining, robots, or similar data gathering and extraction tools'."},
    "schwab": {"publisher": "Schwab Asset Management", "status": "BLOCKED (HTTP 403 bot protection; not bypassed)",
               "terms_url": "https://www.schwabassetmanagement.com/terms-of-use", "terms": None},
    "stooq": {"publisher": "Stooq", "status": "BLOCKED (JavaScript proof-of-work bot check; not bypassed)",
              "terms_url": None, "terms": None},
    "global_x": {"publisher": "Global X ETFs", "status": "NOT USED (terms page unreadable)",
                 "terms_url": "https://www.globalxetfs.com/terms-of-use/", "terms": None},
}
EXCLUDED_SOURCES = ("ishares_blackrock", "invesco", "schwab", "stooq", "global_x")
# path fragments and hosts that are never imported, whatever a rule says
EXCLUDED_PATH = re.compile(r"blackrock|ishares|invesco|_excluded|stooq|schwab|globalx|global_x|probe|terms",
                           re.IGNORECASE)
EXCLUDED_HOSTS = ("blackrock.com", "ishares.com", "invesco.com", "schwabassetmanagement.com", "stooq.com",
                  "globalxetfs.com")

FRED_SERIES = ("NASDAQ100", "NASDAQCOM", "DFF", "DTB3", "DGS1MO", "DGS3MO", "DGS2", "DGS5", "DGS10", "DGS20",
               "DGS30", "GS10", "GS20", "GS30", "FEDFUNDS", "TB3MS", "VXNCLS", "VIXCLS", "BAA10Y", "SOFR")

# derived research tables: research path -> (dest name, produced_by, built from)
DERIVED = {
    "instr-mapped/work/instr_quarterly.csv": (
        "instr_quarterly.csv", "instr-mapped/scripts/s10_dataset.py",
        "13F quarter-end prices (p13f_adj.csv) + distributions per share from SEC XBRL (MO, BDCs, EPD, MPLX), "
        "N-CSR financial highlights (SCHD, XLG, BME); ticker x quarter P, D, y, lr, lp, nav, nii"),
    "instr-mapped/work/p13f_adj.csv": (
        "p13f_adj.csv", "instr-mapped/scripts/s06_13f_prices.py",
        "sec/13f extracts: median over filers of value / shares, split adjusted (today's share basis)"),
    "instr-mapped/work/p13f_raw.csv": ("p13f_raw.csv", "instr-mapped/scripts/s06_13f_prices.py",
                                       "sec/13f extracts: unadjusted price, filers, dispersion, CUSIP per quarter"),
    "instr-mapped/work/p13f_splits.csv": ("p13f_splits.csv", "instr-mapped/scripts/s06_13f_prices.py",
                                          "sec/13f extracts: detected splits"),
    "instr-mapped/work/dist_annual.csv": ("dist_annual.csv", "instr-mapped/scripts/s15_dist.py",
                                          "instr_quarterly.csv: calendar-year distributions and prices"),
    "instr-mapped/work/bme_ncsr.csv": ("bme_ncsr.csv", "instr-mapped/scripts/s08_bme_ncsr_parse.py",
                                       "BME N-CSR / N-CSRS financial highlights (EDGAR): NAV, price, distributions"),
    "instr-mapped/work/core_quarterly.csv": (
        "core_quarterly_research.csv", "instr-mapped/scripts/core_q.py",
        "the core panel's quarterly aggregates (cross-check of data.core_quarterly only)"),
    "instr-structural/work/qqq_fy.csv": (
        "qqq_fy.csv", "instr-structural/scripts/s04_parse_qqq_ar.py",
        "QQQ annual reports (N-30B-2, EDGAR): fiscal-year expense ratio, net investment income, gross dividend yield"),
    "instr-structural/tables/letf_spreads.csv": (
        "letf_spreads.csv", "instr-structural/scripts/s07_letf.py",
        "N-PORT swap terms of QLD / TQQQ: notional-weighted spread, exposure, s_eff per borrowed unit"),
    "instr-structural/work/cc_option_positions.csv": (
        "cc_option_positions.csv", "instr-structural/scripts/s02_parse_nport.py + s09_covered_call.py",
        "N-PORT written NDX calls of QQQI and QYLD: strike, contracts, coverage per report"),
    "instr-structural/tables/tlt_holdings_profile.csv": (
        "tlt_holdings_profile.csv", "instr-structural/scripts/s10_tlt.py",
        "TLT N-PORT bond holdings: portfolio yield, modified duration, convexity per report"),
    "instr-structural/rv_by_month.csv": (
        "rv_by_month_research.csv", "instr-structural/scripts/s17_rv_table.py",
        "FRED NASDAQ100 / NASDAQCOM daily: monthly realized variance (cross-check of data.ndx_rv_table only)"),
    "instr-data/built/qqqi_distributions.csv": ("qqqi_distributions.csv", "instr-data/scripts/build_all.py",
                                                "NEOS QQQI fund page distribution table (ex, record, pay dates)"),
    "instr-data/built/qqqi_financial_highlights.csv": (
        "qqqi_financial_highlights.csv", "instr-data/scripts/build_all.py",
        "NEOS annual / semi-annual financial statements: NAV, distributions, return of capital"),
    "instr-data/built/sec_dividends_quarterly.csv": (
        "sec_dividends_quarterly.csv", "instr-data/scripts/build_all.py",
        "SEC XBRL companyfacts: MO dividends declared, EPD / MPLX distributions per unit"),
    "instr-data/built/sec_bdc_periodic.csv": (
        "sec_bdc_periodic.csv", "instr-data/scripts/build_all.py",
        "SEC XBRL companyfacts (financial highlights): BDC NAV, market price, distributions, NII per period"),
}
DERIVED_SOURCE = {"p13f": "sec_edgar", "instr_quarterly": "sec_edgar", "dist_annual": "sec_edgar",
                  "bme_ncsr": "sec_edgar", "qqq_fy": "sec_edgar", "letf_spreads": "sec_edgar",
                  "cc_option": "sec_edgar", "tlt_holdings": "sec_edgar", "sec_": "sec_edgar", "qqqi": "neos",
                  "rv_by_month": "fred_api", "core_quarterly": "research_derived"}

# (root, glob relative to the root, destination: a folder ending in "/" or a file name, source id, kind)
RULES = [
    ("factors", "french/ff_daily_*/F-F_Research_Data_Factors_daily_CSV.zip",
     "french/F-F_Research_Data_Factors_daily_CSV.zip", "ken_french", "publisher"),
    ("ins", "instr-structural/dl_cboe/*_History.csv", "cboe/", "cboe", "publisher"),
    ("ins", "instr-structural/dl_proshares/QLD-historical_nav.csv", "proshares/", "proshares", "publisher"),
    ("ins", "instr-structural/dl_proshares/TQQQ-historical_nav.csv", "proshares/", "proshares", "publisher"),
    ("ins", "instr-structural/dl_proshares/etf_splits.csv", "proshares/", "proshares", "publisher"),
    ("ins", "instr-structural/dl_proshares/distributions/*.json", "proshares/distributions/", "proshares",
     "publisher"),
    ("ins", "instr-structural/dl_sec/nport/*.xml", "sec/nport/", "sec_edgar", "nport"),
    ("ins", "instr-mapped/dl_sec_13f/*.tsv.gz", "sec/13f/", "sec_edgar", "13f_extract"),
    ("ins", "instr-mapped/dl_sec_13f/manifest.jsonl", "sec/13f/", "sec_edgar", "13f_manifest"),
]
URL_TEMPLATES = {
    "cboe": "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}",
    "proshares_nav": "https://accounts.profunds.com/etfdata/{name}",
    "proshares_dist": "https://www.proshares.com/api/distributionsummary (ticker and year of {name})",
    "french": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}",
}


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def research_hashes(ins: Path, factors: Path | None) -> dict[str, dict]:
    """{sha256: {url, utc, publisher}} from every research manifest that recorded a hash (data, parsed as JSON)."""
    out: dict[str, dict] = {}
    log = Path(ins) / "instr-data" / "raw" / "download_log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and isinstance(r.get("sha256"), str):
                out[r["sha256"].lower()] = {"url": r.get("url"), "utc": r.get("utc"), "manifest": str(log.name)}
    roots = [Path(factors).parent] if factors else []
    for root in roots:
        for p in sorted(root.glob("*/sources.json")):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            stack = [doc]
            while stack:
                o = stack.pop()
                if isinstance(o, dict):
                    if isinstance(o.get("sha256"), str) and len(o["sha256"]) == 64:
                        out.setdefault(o["sha256"].lower(), {"url": o.get("url"), "utc": o.get("utc"),
                                                            "publisher": o.get("publisher"),
                                                            "manifest": f"{p.parent.name}/sources.json"})
                    stack.extend(o.values())
                elif isinstance(o, list):
                    stack.extend(o)
    return out


def _nport_urls(ins: Path) -> dict[str, str]:
    """N-PORT file name -> its EDGAR URL, from instr-structural's request log (tab separated text)."""
    out: dict[str, str] = {}
    log = Path(ins) / "instr-structural" / "dl_sec" / "requests.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) >= 5 and parts[1].startswith("https://"):
                out[Path(parts[4]).name] = parts[1]
    return out


def _excluded(rel: str, url: str | None) -> str | None:
    if EXCLUDED_PATH.search(rel):
        return "excluded source or probe (terms / bot protection)"
    host = re.sub(r"^https?://([^/]+).*$", r"\1", url or "").lower()
    if any(host == h or host.endswith("." + h) for h in EXCLUDED_HOSTS):
        return f"excluded host {host}"
    return None


def plan(ins: Path, factors: Path | None) -> tuple[list[dict], list[dict]]:
    """(copies, skipped): every file to import with its destination (relative to DEST) and provenance hints."""
    ins = Path(ins)
    copies: list[dict] = []
    skipped: list[dict] = []
    roots = {"ins": ins, "factors": Path(factors) if factors else None}
    for sid in FRED_SERIES:
        for suffix, kind in (("", "fred_observations"), ("_meta", "fred_meta")):
            p = ins / "instr-structural" / "dl_fred" / f"{sid}{suffix}.json"
            if p.exists():
                copies.append({"src": p, "rel": p.relative_to(ins).as_posix(), "dest": f"fred/{sid}{suffix}.json",
                               "source": "fred_api", "kind": kind, "series_id": sid})
            else:
                skipped.append({"source": f"instr-structural/dl_fred/{sid}{suffix}.json", "reason": "not downloaded"})
    for root, pattern, dest, source, kind in RULES:
        base = roots.get(root)
        if base is None:
            skipped.append({"source": f"{root}:{pattern}", "reason": "root not given"})
            continue
        for p in sorted(Path(x) for x in glob.glob(str(base / pattern))):
            rel = p.relative_to(base).as_posix()
            target = dest + p.name if dest.endswith("/") else dest
            copies.append({"src": p, "rel": f"{root}:{rel}", "dest": target, "source": source, "kind": kind})
    for rel, (name, script, built_from) in DERIVED.items():
        p = ins / rel
        if not p.exists():
            skipped.append({"source": rel, "reason": "not built by the research round"})
            continue
        src_id = next((v for k, v in DERIVED_SOURCE.items() if name.startswith(k)), "research_derived")
        copies.append({"src": p, "rel": rel, "dest": f"derived/{name}", "source": src_id, "kind": "derived",
                       "produced_by": script, "built_from": built_from})
    return copies, skipped


def _url(c: dict, nport: dict[str, str]) -> str | None:
    name = Path(c["dest"]).name
    if c["kind"] == "fred_observations":
        return f"{API}/series/observations?series_id={c['series_id']}&api_key=REDACTED&file_type=json"
    if c["kind"] == "fred_meta":
        return f"{API}/series?series_id={c['series_id']}&api_key=REDACTED&file_type=json"
    if c["kind"] == "nport":
        return nport.get(name)
    if c["source"] == "cboe":
        return URL_TEMPLATES["cboe"].format(name=name)
    if c["source"] == "ken_french":
        return URL_TEMPLATES["french"].format(name=name)
    if c["source"] == "proshares":
        return URL_TEMPLATES["proshares_dist" if "/distributions/" in c["dest"] else "proshares_nav"].format(name=name)
    if c["kind"] in ("13f_extract", "13f_manifest"):
        return "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets (zips named in manifest.jsonl)"
    return None


def _secrets() -> list[bytes]:
    """The configured secrets (FRED key, SEC User-Agent) as bytes, to refuse files that contain them."""
    out = []
    for var in (config.FRED["key_env"], "SEC_USER_AGENT"):
        v = os.environ.get(var, "").strip()
        if len(v) >= 8:
            out.append(v.encode())
    return out


def run(ins: Path, factors: Path | None = None, dest_root: Path | None = None, force: bool = False) -> dict:
    """Copy the instrument research downloads into dest_root (data/raw/instruments); returns the import log."""
    ins = Path(ins).resolve()
    factors = Path(factors).resolve() if factors else None
    dest_root = Path(dest_root or DEST)
    hashes = research_hashes(ins, factors)
    nport = _nport_urls(ins)
    secrets = _secrets()
    copies, skipped = plan(ins, factors)
    src_path = dest_root / "SOURCES.json"
    doc = jsonio.load(src_path) if src_path.exists() else {}
    records = dict(doc.get("files", {}))
    rows = []
    counts = {"copied": 0, "unchanged": 0, "conflict": 0, "rejected": 0, "skipped": len(skipped)}
    for c in copies:
        url = _url(c, nport)
        why = _excluded(c["rel"], url)
        if why:
            counts["skipped"] += 1
            rows.append({"source": c["rel"], "dest": c["dest"], "action": "skipped", "reason": why})
            continue
        body = c["src"].read_bytes()
        digest = _sha(body)
        dest = dest_root / c["dest"]
        row = {"source": c["rel"], "dest": c["dest"], "kind": c["kind"], "sha256": digest}
        if any(s in body for s in secrets):
            row.update(action="rejected", reason="the file contains a configured secret (FRED key / SEC User-Agent)")
        elif c["kind"] == "fred_observations" and not admissible(body)[0]:
            row.update(action="rejected", reason=f"not a full FRED request ({admissible(body)[1]})")
        elif dest.exists() and dest.read_bytes() == body:
            row.update(action="unchanged")
        elif dest.exists() and not force:
            row.update(action="conflict", reason="a different file exists (use --force to replace it)")
        else:
            existed = dest.exists()
            _copy(c["src"], dest)
            row.update(action="copied", replaced=existed)
        counts[row["action"]] += 1
        if row["action"] in ("copied", "unchanged"):
            known = hashes.get(digest, {})
            rec = {"path": c["dest"], "research_path": c["rel"], "kind": c["kind"], "source": c["source"],
                   "url": known.get("url") or url, "access_date": (known.get("utc") or ACCESS_DATE)[:10],
                   "sha256": digest, "bytes": len(body),
                   "verified": f"sha256 = {known['manifest']}" if known else None}
            if c.get("series_id"):
                rec["series_id"] = c["series_id"]
            if c["kind"] == "derived":
                rec.update(produced_by=c["produced_by"], built_from=c["built_from"],
                           note="research output, data only (the producing script is never run here)")
            records[c["dest"]] = {k: (redact(v) if isinstance(v, str) else v) for k, v in rec.items()}
        rows.append(row)
    rows += [{**s, "action": "skipped"} for s in skipped]
    sources_doc = {
        "version": 1, "generated_by": "scenariolab.instruments.import_raw", "access_date": ACCESS_DATE,
        "note": "provenance of every instrument file; URLs never carry a key; excluded sources are never read",
        "sources": SOURCES, "excluded": list(EXCLUDED_SOURCES), "files": dict(sorted(records.items()))}
    log = {"run_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": str(ins),
           "factors_raw": str(factors) if factors else None, "force": bool(force), "counts": counts,
           "bytes_copied": int(sum(r.get("bytes", 0) for r in records.values()
                                   if any(x.get("dest") == r["path"] and x.get("action") == "copied" for x in rows))),
           "files": rows}
    jsonio.dump(json.loads(redact(json.dumps(sources_doc, ensure_ascii=False))), src_path)
    jsonio.dump(json.loads(redact(json.dumps(log, ensure_ascii=False))), dest_root / "import_log.json")
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import the instrument research downloads into data/raw/instruments")
    ap.add_argument("src", type=Path, help="the instruments research directory (instr-data, instr-structural, ...)")
    ap.add_argument("--factors-raw", type=Path, default=None, help="the missing-factor round's raw directory")
    ap.add_argument("--dest", type=Path, default=None, help="destination (default: data/raw/instruments)")
    ap.add_argument("--force", action="store_true", help="replace existing files that differ")
    args = ap.parse_args(argv)
    if not args.src.is_dir():
        ap.error(f"not a directory: {args.src}")
    load_dotenv()                   # only so that configured secrets can be refused and redacted
    log = run(args.src, args.factors_raw, args.dest, force=args.force)
    c = log["counts"]
    print(f"[instruments.import_raw] copied {c['copied']}, unchanged {c['unchanged']}, conflicts {c['conflict']}, "
          f"rejected {c['rejected']}, skipped {c['skipped']} ({log['bytes_copied'] / 1e6:.1f} MB)")
    for r in log["files"]:
        if r["action"] in ("conflict", "rejected"):
            print(f"[instruments.import_raw] {r['action']}: {r['source']} -> {r.get('dest')} ({r.get('reason')})")
    return 1 if c["conflict"] or c["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
