"""Import the research round's downloads into data/raw (FACTORS_DESIGN §4.3); run once, no new downloads.

    python -m scenariolab.data.import_raw <factors/raw dir> [--force]

Bytes are copied, nothing else: no file from the source directory is imported, executed or opened as a
workbook (FRED responses and the families' sources.json are parsed as JSON data). A FRED observations file
enters the canonical cache (fred/<ID>.json, <ID>_m_eop.json, <ID>_m_avg.json; the research cache's <ID>_obs.json
is the same request as <ID>.json) only when it is a full, unpaged, untransformed request; any other FRED file
(ALFRED vintages and first releases, release lists, searches) is kept under fred/alfred/ or fred/searches/ for
provenance, and a series whose file did not qualify is fetched again by the build. The publisher files go to
the paths data/sources.py reads. Files the build does not read (HTML pages, Cboe, EIA, Census, OECD, BLS API
responses, extra French files, ...) are not copied.

Each file's SHA-256 (or the 12-digit SHA-1 prefix some families recorded) is checked against the families'
sources.json (the folder above the raw directory holds rates/, risk/, ... with one each) where recorded. An
existing different file is never overwritten without --force. Writes data/raw/SOURCES.json (one provenance
record per imported file: url with api_key=REDACTED, publisher, access_date, terms, frequency, first_month,
publication_lag, revised, copyright, sha256) and data/raw/import_log.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from .. import config, jsonio
from ..env import load_dotenv

FAMILIES = ("rates", "risk", "labor", "housing", "commod", "tech")
ACCESS_DATE = "2026-11-01"                      # the research round's downloads (the families' access_date)
API = "https://api.stlouisfed.org/fred"

# research path (glob, relative to the raw directory) -> project path; a trailing "/" keeps the file name
PUBLISHER_FILES = [
    ("federalreserve/ebp_*/ebp_csv.csv", "federalreserve/ebp_csv.csv"),
    ("nyfed/acm_xls_*/ACMTermPremium.xls", "nyfed/ACMTermPremium.xls"),
    ("nyfed/Holston_Laubach_Williams_current_estimates_*/*.xlsx", "nyfed/"),
    ("nyfed/Holston_Laubach_Williams_real_time_estimates_*/*.xlsx", "nyfed/"),
    ("worldbank/pinksheet_monthly_*/CMO-Historical-Data-Monthly.xlsx", "worldbank/"),
    ("freddiemac/fmhpi_master_*/fmhpi_master_file.csv", "freddiemac/"),
    ("shiller/fig3-1_*/Fig3-1.xls", "shiller/Fig3-1.xls"),
    ("iacoviello/gpr_*/data_gpr_export.xls", "iacoviello/"),
    ("policyuncertainty/us_epu_*/*.xlsx", "policyuncertainty/"),
    ("policyuncertainty/historical_epu_*/*.xlsx", "policyuncertainty/"),
]
FRED_DATA = re.compile(r"^(?P<id>[A-Z0-9]+)(?P<suffix>_obs|_m_eop|_m_avg)?\.json$")
FRED_META = re.compile(r"^(?P<id>[A-Z0-9]+)_meta\.json$")
FRED_ALFRED = re.compile(r"^(?P<id>[A-Z0-9]+)_(rt.*|first.*|initial.*|vintage.*|vintagedates|release|realtime_all)\.json$")
FRED_RELEASE_SOURCES = re.compile(r"^_release.*_sources\.json$")
FRED_SEARCH = re.compile(r"^(SEARCH_|_search_|search_).*\.json$")

FILE_KEYS = ("file", "local_path", "raw_file", "cache_file", "cache_files", "cached_files", "files", "path", "cache")
URL_KEYS = ("url", "urls", "url_observations", "metadata_url", "url_meta", "url_metadata")
FIELDS = ("url", "publisher", "access_date", "terms", "frequency", "first_month", "publication_lag", "revised",
          "copyright")
KEY_IN_URL = re.compile(r"(api_key=)[^&\s\"'<>]+")


def redact(text: str) -> str:
    """Every api_key=<value> becomes api_key=REDACTED (and the configured key itself, should it appear bare)."""
    text = KEY_IN_URL.sub(r"\1REDACTED", text)
    key = os.environ.get(config.FRED["key_env"], "").strip()
    return text.replace(key, "REDACTED") if len(key) >= 8 else text


def _clean(value):
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _sha(data: bytes, name: str = "sha256") -> str:
    return hashlib.new(name, data).hexdigest()


# ---------------------------------------------------------------- the families' provenance

def _walk(obj, out: list) -> None:
    if isinstance(obj, dict):
        out.append(obj)
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)


def _file_refs(d: dict) -> list[str]:
    """Paths a provenance record names, relative to the raw directory ('fred/X.json', 'nyfed/.../x.xls')."""
    out = []
    for k in FILE_KEYS:
        v = d.get(k)
        for x in (v if isinstance(v, list) else [v]):
            if not isinstance(x, str) or not x.strip():
                continue
            x = x.strip()
            if "/raw/" in x:
                out.append(x.split("/raw/", 1)[1])
            elif x.startswith("raw/"):
                out.append(x[4:])
            elif re.match(r"^[A-Za-z0-9_]+\.json$", x):           # a bare FRED cache name
                out.append("fred/" + x)
    return out


def _urls(d: dict) -> list[str]:
    out = []
    for k in URL_KEYS:
        v = d.get(k)
        out += [u for u in (v if isinstance(v, list) else [v]) if isinstance(u, str) and u.startswith("http")]
    return out


def _series_id(d: dict) -> str | None:
    for k in ("fred_id", "series_id"):
        if isinstance(d.get(k), str) and re.match(r"^[A-Z0-9]+$", d[k]):
            return d[k]
    for u in _urls(d):
        m = re.search(r"api\.stlouisfed\.org/fred/series(?:/observations)?\?(?:[^#]*&)?series_id=([A-Z0-9]+)", u)
        if m:
            return m.group(1)
    return None


def _first(*values):
    return next((v for v in values if v not in (None, "", [], {})), None)


def _record(d: dict, family_date: str | None) -> dict:
    urls = _urls(d)
    obs_urls = [u for u in urls if "/series/observations" in u]
    first = _first(d.get("first_month"), d.get("first_obs"), d.get("first_observation"), d.get("observation_start"))
    if isinstance(first, str) and re.match(r"^\d{4}-\d{2}-\d{2}", first):
        first = first[:7]
    terms = _first(d.get("terms"), d.get("terms_of_use"))
    in_terms = next((p.strip() for p in str(terms or "").split(";") if re.search(r"©|copyright", p, re.I)), None)
    third_party = "third-party copyright (FRED notes)" if d.get("third_party_copyright_in_notes") is True else None
    return {
        "url": _first(d.get("url") if isinstance(d.get("url"), str) else None, *obs_urls, *urls),
        "publisher": _first(d.get("publisher")),
        "access_date": _first(d.get("access_date"), family_date),
        "terms": terms,
        "frequency": _first(d.get("frequency"), d.get("frequency_native"), d.get("frequency_used")),
        "first_month": first,
        "publication_lag": _first(d.get("publication_lag"), d.get("publication_lag_months"), d.get("lag_used")),
        "revised": _first(d.get("revised")),
        "copyright": _first(d.get("copyright"), in_terms, third_party),
        "sha256": d.get("sha256") if isinstance(d.get("sha256"), str) else None,
        "sha1": d.get("sha1") if isinstance(d.get("sha1"), str) else None,
    }


def _merge(into: dict, rec: dict, family: str) -> None:
    for k, v in rec.items():
        if into.get(k) in (None, "") and v not in (None, ""):
            into[k] = v
    into.setdefault("families", [])
    if family not in into["families"]:
        into["families"].append(family)


def family_provenance(families_root: Path) -> tuple[dict, dict]:
    """({research path: merged record}, {FRED series id: merged record}) from every family's sources.json."""
    by_file: dict = {}
    by_series: dict = {}
    for fam in FAMILIES:
        path = Path(families_root) / fam / "sources.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        dicts: list = []
        _walk(doc, dicts)
        for d in dicts:
            rec = _record(d, doc.get("access_date") if isinstance(doc, dict) else None)
            for ref in _file_refs(d):
                _merge(by_file.setdefault(ref, {}), rec, fam)
            sid = _series_id(d)
            if sid and any(rec[k] for k in ("publisher", "terms", "revised", "publication_lag", "copyright")):
                _merge(by_series.setdefault(sid, {}), {k: v for k, v in rec.items() if k not in ("sha256", "sha1")},
                       fam)
    return by_file, by_series


# ---------------------------------------------------------------- what to copy where

def admissible(body: bytes) -> tuple[bool, str]:
    """Whether a FRED observations response is a full, unpaged, untransformed request (the canonical cache)."""
    try:
        d = json.loads(body)
    except ValueError:
        return False, "not JSON"
    obs = d.get("observations") if isinstance(d, dict) else None
    if not isinstance(obs, list):
        return False, "no observations list"
    for k, want in (("observation_start", "1600-01-01"), ("observation_end", "9999-12-31"), ("units", "lin")):
        if d.get(k) != want:
            return False, f"{k} = {d.get(k)!r}"
    count, limit = d.get("count"), d.get("limit")
    if not (isinstance(count, int) and isinstance(limit, int) and count == len(obs) <= limit):
        return False, "paged or truncated (count, limit)"
    if d.get("offset", 0) != 0 or d.get("output_type", 1) != 1:
        return False, "offset or output_type (not the plain request)"
    if d.get("realtime_start") != d.get("realtime_end"):
        return False, "a real-time range (vintages)"
    return True, ""


def plan(src: Path) -> tuple[list[dict], list[dict]]:
    """(copies, skipped): copies = [{source, dest, kind, series_id}] (source and dest relative), skipped =
    [{source, reason}] for files that stay in the research archive."""
    src = Path(src)
    copies: list[dict] = []
    skipped: list[dict] = []
    taken: set[str] = set()
    for pattern, dest in PUBLISHER_FILES:
        for p in sorted(src.glob(pattern)):           # dated folders sort by date: the newest wins
            rel = p.relative_to(src).as_posix()
            target = dest + p.name if dest.endswith("/") else dest
            copies = [c for c in copies if c["dest"] != target]
            copies.append({"source": rel, "dest": target, "kind": "publisher", "series_id": None})
            taken.add(rel)
    fred_dir = src / "fred"
    data: dict[str, list] = {}
    for p in sorted(fred_dir.iterdir()) if fred_dir.is_dir() else []:
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        name = p.name
        m = FRED_DATA.match(name)
        if m:
            suffix = m.group("suffix") if m.group("suffix") in ("_m_eop", "_m_avg") else ""
            data.setdefault(f"fred/{m.group('id')}{suffix}.json", []).append((p, rel, m.group("id"), suffix))
        elif FRED_META.match(name):
            copies.append({"source": rel, "dest": rel, "kind": "fred_meta", "series_id": FRED_META.match(name)["id"]})
        elif FRED_ALFRED.match(name) or FRED_RELEASE_SOURCES.match(name):
            sid = FRED_ALFRED.match(name)["id"] if FRED_ALFRED.match(name) else None
            copies.append({"source": rel, "dest": f"fred/alfred/{name}", "kind": "alfred", "series_id": sid})
        elif FRED_SEARCH.match(name):
            copies.append({"source": rel, "dest": f"fred/searches/{name}", "kind": "fred_search", "series_id": None})
        else:
            skipped.append({"source": rel, "reason": "unrecognised FRED file name"})
        taken.add(rel)
    for dest, cands in sorted(data.items()):
        ok = []
        for p, rel, sid, suffix in cands:
            body = p.read_bytes()
            good, why = admissible(body)
            if good:
                d = json.loads(body)
                ok.append((d.get("realtime_start") or "", 0 if rel.endswith(f"{sid}{suffix}.json") else 1, rel, sid, d))
            else:
                copies.append({"source": rel, "dest": f"fred/alfred/{p.name}", "kind": "alfred", "series_id": sid,
                               "note": f"not a full request ({why}); fetched again by the build"})
        if not ok:
            continue
        ok.sort(key=lambda t: (t[0], -t[1]))        # the latest vintage; the canonical name on a tie
        best = ok[-1]
        kind = "fred_aggregate" if dest.endswith(("_m_eop.json", "_m_avg.json")) else "fred_observations"
        copies.append({"source": best[2], "dest": dest, "kind": kind, "series_id": best[3],
                       "first_month": _first_month(best[4])})
        for t in ok[:-1]:
            same = t[4].get("observations") == best[4].get("observations")
            skipped.append({"source": t[2], "reason": f"the same request as {best[2]}"
                            + (" (identical observations)" if same else f" (older vintage {t[0]})")})
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src).as_posix()
        if p.is_file() and rel not in taken:
            skipped.append({"source": rel, "reason": "not read by the build (research archive only)"})
    return copies, skipped


def _first_month(d: dict) -> str | None:
    return next((o["date"][:7] for o in d.get("observations", []) if o.get("value") not in (".", "", None)), None)


def _fred_url(kind: str, sid: str, dest: str) -> str | None:
    if kind == "fred_meta":
        return f"{API}/series?series_id={sid}&api_key=REDACTED&file_type=json"
    if kind == "fred_observations":
        return f"{API}/series/observations?series_id={sid}&api_key=REDACTED&file_type=json"
    if kind == "fred_aggregate":
        agg = "eop" if dest.endswith("_m_eop.json") else "avg"
        return (f"{API}/series/observations?series_id={sid}&frequency=m&aggregation_method={agg}"
                f"&api_key=REDACTED&file_type=json")
    return None


def _fred_meta(src: Path, sid: str) -> dict:
    p = Path(src) / "fred" / f"{sid}_meta.json"
    try:
        return (json.loads(p.read_bytes()).get("seriess") or [{}])[0]
    except (OSError, ValueError, AttributeError):
        return {}


def _copyright_note(notes: str) -> str | None:
    """The sentence of FRED's notes that states a copyright, if any (shortened)."""
    for sentence in re.split(r"(?<=[.!?])\s+", notes or ""):
        if re.search(r"copyright|©", sentence, re.IGNORECASE):
            return " ".join(sentence.split())[:240]
    return None


def provenance(c: dict, src: Path, by_file: dict, by_series: dict, sha256: str, size: int) -> dict:
    """The SOURCES.json record of one copied file."""
    fam = dict(by_file.get(c["source"], {}))
    sid = c.get("series_id")
    if sid and c["kind"] != "fred_search":
        for k, v in by_series.get(sid, {}).items():
            if k == "families":
                fam["families"] = sorted(set(fam.get("families", [])) | set(v))
            elif fam.get(k) in (None, "") and v not in (None, ""):
                fam[k] = v
    rec = {"path": c["dest"], "research_path": c["source"], "kind": c["kind"], "series_id": sid}
    rec.update({k: fam.get(k) for k in FIELDS})
    if c["kind"] in ("fred_observations", "fred_aggregate", "fred_meta"):
        meta = _fred_meta(src, sid)
        rec["url"] = _fred_url(c["kind"], sid, c["dest"])
        rec["publisher"] = rec["publisher"] or "FRED (Federal Reserve Bank of St. Louis)"
        rec["frequency"] = rec["frequency"] or meta.get("frequency")
        rec["copyright"] = _copyright_note(meta.get("notes", "")) or rec["copyright"]     # FRED's own note first
        rec["terms"] = rec["terms"] or "FRED API Terms of Use (https://fred.stlouisfed.org/docs/api/terms_of_use.html)"
        if c.get("first_month"):
            rec["first_month"] = c["first_month"]
    elif c["kind"] == "alfred" and rec["url"] and not re.search(r"realtime|vintage|output_type|/release", rec["url"]):
        rec["url"] = None              # a family record listing this file next to the plain request's URL
    rec["access_date"] = rec["access_date"] or ACCESS_DATE
    rec.update({"sha256": sha256, "bytes": size, "families": sorted(fam.get("families", []))})
    if c.get("note"):
        rec["note"] = c["note"]
    return {k: _clean(v) if k != "families" else v for k, v in rec.items()}


# ---------------------------------------------------------------- run

def _copy(src_file: Path, dest_file: Path) -> None:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_file.with_name(dest_file.name + ".part")
    shutil.copyfile(src_file, tmp)                    # bytes only
    tmp.replace(dest_file)


def run(src: Path, dest_root: Path | None = None, families_root: Path | None = None, force: bool = False) -> dict:
    """Copy the research downloads under `src` into `dest_root` (data/raw); returns the import log."""
    src = Path(src).resolve()
    dest_root = Path(dest_root or config.RAW_DIR)
    families_root = Path(families_root or src.parent)
    by_file, by_series = family_provenance(families_root)
    copies, skipped = plan(src)
    key = os.environ.get(config.FRED["key_env"], "").strip()
    sources_path = dest_root / "SOURCES.json"
    sources = jsonio.load(sources_path) if sources_path.exists() else {}
    records = dict(sources.get("files", {}))
    rows = []
    counts = {"copied": 0, "unchanged": 0, "conflict": 0, "rejected": 0, "skipped": len(skipped)}
    for c in copies:
        source, dest = src / c["source"], dest_root / c["dest"]
        body = source.read_bytes()
        digest = _sha(body)
        recorded = by_file.get(c["source"], {})
        row = {"source": c["source"], "dest": c["dest"], "kind": c["kind"], "sha256": digest}
        if recorded.get("sha256") and recorded["sha256"].lower() != digest:
            row.update(action="rejected", reason="SHA-256 differs from the family's sources.json")
        elif recorded.get("sha1") and not _sha(body, "sha1").startswith(recorded["sha1"].lower()):
            row.update(action="rejected", reason="SHA-1 differs from the family's sources.json")
        elif len(key) >= 8 and key.encode() in body:
            row.update(action="rejected", reason="the file contains the FRED API key")
        elif dest.exists() and dest.read_bytes() == body:
            row.update(action="unchanged")
        elif dest.exists() and not force:
            row.update(action="conflict", reason="a different file exists (use --force to replace it)")
        else:
            existed = dest.exists()
            _copy(source, dest)
            row.update(action="copied", replaced=existed)
        counts[row["action"]] += 1
        if row["action"] in ("copied", "unchanged"):
            records[c["dest"]] = provenance(c, src, by_file, by_series, digest, len(body))
            row["verified"] = "sha256" if recorded.get("sha256") else "sha1" if recorded.get("sha1") else None
        rows.append(row)
    rows += [{**s, "action": "skipped"} for s in skipped]
    sources_doc = {"version": 1, "generated_by": "scenariolab.data.import_raw", "access_date": ACCESS_DATE,
                   "note": "provenance of every file imported from the research round; URLs never carry a key",
                   "files": dict(sorted(records.items()))}
    log = {"run_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": str(src),
           "families_root": str(families_root), "force": bool(force), "counts": counts,
           "bytes_copied": int(sum((src / r["source"]).stat().st_size for r in rows if r.get("action") == "copied")),
           "files": rows}
    jsonio.dump(json.loads(redact(json.dumps(sources_doc, ensure_ascii=False))), sources_path)
    jsonio.dump(json.loads(redact(json.dumps(log, ensure_ascii=False))), dest_root / "import_log.json")
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import the research round's downloads into data/raw")
    ap.add_argument("src", type=Path, help="the research raw directory (factors/raw)")
    ap.add_argument("--force", action="store_true", help="replace existing files that differ")
    ap.add_argument("--families", type=Path, default=None,
                    help="folder with the families' sources.json (default: the raw directory's parent)")
    ap.add_argument("--dest", type=Path, default=None, help="destination (default: data/raw)")
    args = ap.parse_args(argv)
    if not args.src.is_dir():
        ap.error(f"not a directory: {args.src}")
    load_dotenv()                           # only so that the key, if configured, can be redacted and refused
    log = run(args.src, args.dest, args.families, force=args.force)
    c = log["counts"]
    print(f"[import_raw] copied {c['copied']}, unchanged {c['unchanged']}, conflicts {c['conflict']}, "
          f"rejected {c['rejected']}, not copied {c['skipped']} ({log['bytes_copied'] / 1e6:.1f} MB)")
    for r in log["files"]:
        if r["action"] in ("conflict", "rejected"):
            print(f"[import_raw] {r['action']}: {r['source']} -> {r['dest']} ({r['reason']})")
    return 1 if c["conflict"] or c["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
