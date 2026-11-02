"""Raw inputs of the quarterly panel: cache-only readers and the import of the research rounds' downloads.

The quarterly panel (qmodel/panel.py) reads the monthly panel plus series the monthly app never needed: NIPA, Z.1
and FRED series of the schema rounds (labour force, real consumption and investment, broker-dealer leverage, the
German 10-year yield, the real dollar, bank securities, NFC free cash flow, ...), the Philadelphia Fed delivery-time
index, the Michigan 5-10-year inflation expectation, the categorical health-care EPU and the Shiller 2026 workbook.
They were downloaded by the research rounds of 2026-11-01 and are imported here once:

    python -m scenariolab.qmodel.sources_q import <scratchpad directory> [--force]

As scenariolab.data.import_raw does for the factor round: bytes are copied, nothing in the source directory is
executed or opened as a workbook, each file's SHA-1 / SHA-256 is checked against the hash its round recorded,
a FRED file is accepted only when it is a full, unpaged, untransformed request, an existing different file is never
replaced without --force, and every copied file gets a provenance record in data/raw/SOURCES.json (url with
api_key=REDACTED, publisher, access date, terms, frequency, first month, publication lag, copyright, sha256). The
log goes to data/raw/import_log_quarterly.json (the factor round's import_log.json is left alone). The Shiller
2026 workbook goes to shiller/ie_data.xls, the file scenariolab.data.import_raw also imports (P0.3): identical bytes
are "unchanged", and an existing record of another importer is kept.

Readers never download: a missing file raises FileNotFoundError naming the import command.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, jsonio
from ..data import sources as src
from ..data.import_raw import _copyright_note, redact
from . import config_q as cq

ACCESS_DATE = "2026-11-01"
IMPORT_HINT = "python -m scenariolab.qmodel.sources_q import <scratchpad 目录>"
FRED_TERMS = ("FRED API Terms of Use (https://fred.stlouisfed.org/docs/api/terms_of_use.html); the data belong to the "
              "original source; the app does not redistribute data (data/ is git-ignored)")
BEA, Z1 = "U.S. Bureau of Economic Analysis (NIPA)", "Federal Reserve Board, Financial Accounts of the United States (Z.1)"
FRED_IMPORTS = {
    # series id: (research folder under the scratchpad, publisher, publication lag text)
    "A008RA3Q086SBEA": ("sim_redesign/schema_lit/dl/fred_20261101", BEA, "advance estimate about 1 month"),
    "A011RA3Q086SBEA": ("sim_redesign/schema_lit/dl/fred_20261101", BEA, "advance estimate about 1 month"),
    "EXPGSC1": ("sim_redesign/schema_lit/dl/fred_20261101", BEA, "advance estimate about 1 month"),
    "A053RC1Q027SBEA": ("sim_redesign/schema_lit/dl/fred_20261101", BEA, "with corporate profits, about 2-3 months"),
    "W055RC1Q027SBEA": ("sim_redesign/schema_lit/dl/fred_20261101", BEA, "advance estimate about 1 month"),
    "DGDSRA3M086SBEA": ("sim_redesign/schema_lit/dl/fred_20261101", BEA, "end of the next month"),
    "CLF16OV": ("sim_redesign/schema_lit/dl/fred_20261101", "U.S. Bureau of Labor Statistics (CPS)",
                "first Friday of the next month"),
    "BOGZ1FL664090005Q": ("sim_redesign/schema_lit/dl/fred_20261101", Z1, "about 10 weeks"),
    "BOGZ1FL664190005Q": ("sim_redesign/schema_lit/dl/fred_20261101", Z1, "about 10 weeks"),
    "NCBEILQ027S": ("sim_redesign/schema_lit/dl/fred_20261101", Z1, "about 10 weeks"),
    "BOGZ1FA105050005Q": ("sim_redesign/schema_lit/dl/fred_20261101", Z1, "about 10 weeks"),
    "BOGZ1FA106121075Q": ("sim_redesign/schema_lit/dl/fred_20261101", Z1, "about 10 weeks"),
    "W328RC1Q027SBEA": ("sim_redesign/valuation/downloads/fred_fcf_20261101", BEA,
                        "with corporate profits, about 2-3 months"),
    "B456RC1Q027SBEA": ("sim_redesign/valuation/downloads/fred_fcf_20261101", BEA,
                        "with corporate profits, about 2-3 months"),
    "IRLTLT01DEM156N": ("sim_redesign/schema_scen/raw/fred_obs_20261101", "OECD Main Economic Indicators",
                        "about 1 month"),
    "BOGZ1FL075035503Q": ("sim_redesign/schema_scen/raw/fred_obs_20261101", Z1, "about 10 weeks"),
    "BCNSDODNS": ("sim_redesign/schema_scen/raw/fred_obs_20261101", Z1, "about 10 weeks"),
    "NCBCEBQ027S": ("sim_redesign/schema_scen/raw/fred_obs_20261101", Z1, "about 10 weeks"),
    "TWEXBPA": ("sim_redesign/schema-revise-1/dl/fred_20261101", "Federal Reserve Board (H.10, discontinued 2019)",
                "about 1 month"),
    "RTWEXBGS": ("sim_redesign/schema-revise-1/dl/fred_20261101", "Federal Reserve Board (H.10)", "about 1 month"),
    "TLAACBM027SBOG": ("sim_redesign/schema-revise-1/dl/fred_20261101", "Federal Reserve Board (H.8)",
                       "about 1 week after the month"),
    "INVEST": ("sim_redesign/schema-revise-1/dl/fred_20261101b", "Federal Reserve Board (H.8)",
               "about 1 week after the month"),
}
META_DIRS = {"sim_redesign/schema_scen/raw/fred_obs_20261101": "sim_redesign/schema_scen/raw/fred_meta_20261101"}
# publisher files: research path -> (data/raw path, hash kind, expected hash, provenance)
PUBLISHER_IMPORTS = [
    ("factors/raw/shiller/ie_data_shillerdata_20261101/ie_data.xls", cq.SHILLER_FILES[0], "sha1",
     "aa647cf3944d31368359264027800dbf703ebd61",
     {"url": "https://shillerdata.com/ (ie_data.xls, the 2026 workbook)", "frequency": "monthly",
      "publisher": "Robert J. Shiller, 'Irrational Exuberance' data (S&P composite price, dividends, earnings, CPI)",
      "terms": "free to use with attribution", "copyright": "Robert J. Shiller", "first_month": "1871-01",
      "publication_lag": "S&P quarterly earnings complete about 2 months after the quarter (known-at lag 3)",
      "revised": "yes (estimates replaced by reported values)"}),
    ("factors/raw/philfed/mbos_bos_dif_20261101/bos_dif.csv", "philfed/bos_dif.csv", "sha1",
     "de48250d6935c83cbf9632f8be846cae26e3d7c8",
     {"url": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/MBOS/Historical-Data/"
             "Diffusion-Indexes/bos_dif.csv", "frequency": "monthly",
      "publisher": "Federal Reserve Bank of Philadelphia, Manufacturing Business Outlook Survey",
      "terms": "public; cite the Federal Reserve Bank of Philadelphia", "copyright": None, "first_month": "1968-05",
      "publication_lag": "third Thursday of the survey month (known-at lag 0)", "revised": "seasonal factors"}),
    ("factors/raw/umich/tbmpx1px5_20261101/tbmpx1px5.csv", "umich/tbmpx1px5.csv", "sha1", "60f16891172c",
     {"url": "https://www.sca.isr.umich.edu/files/tbmpx1px5.csv", "frequency": "monthly",
      "publisher": "University of Michigan, Surveys of Consumers",
      "terms": "Copyright University of Michigan Surveys of Consumers; cite the source; not for redistribution",
      "copyright": "University of Michigan", "first_month": "1978-01",
      "publication_lag": "0 (final reading at the end of the month)", "revised": "no"}),
    ("factors/raw/policyuncertainty/categorical_epu_20261101/Categorical_EPU_Data.xlsx",
     "policyuncertainty/Categorical_EPU_Data.xlsx", "sha256",
     "99b5bb553df81bf742ef1157004efe82d4a85a477d93021b13692961ddd041d3",
     {"url": "https://www.policyuncertainty.com/media/Categorical_EPU_Data.xlsx", "frequency": "monthly",
      "publisher": "Baker, Bloom and Davis, www.PolicyUncertainty.com",
      "terms": "free use with attribution to the authors, the paper and the website", "copyright": None,
      "first_month": "1985-01", "publication_lag": "about one month (lag 1)", "revised": "small"}),
]


# ---------------------------------------------------------------- import

def fred_full_request(body: bytes) -> tuple[bool, str]:
    """A FRED observations response that holds the whole series: unpaged, untransformed, one vintage, and no
    observation_start after the first observation (import_raw's rule, which also accepts an explicit start that cuts
    nothing off, as the schema rounds requested 1947-01-01)."""
    try:
        d = json.loads(body)
    except ValueError:
        return False, "not JSON"
    obs = d.get("observations") if isinstance(d, dict) else None
    if not isinstance(obs, list) or not obs:
        return False, "no observations"
    if d.get("observation_end") != "9999-12-31" or d.get("units") != "lin":
        return False, "observation_end or units"
    count, limit = d.get("count"), d.get("limit")
    if not (isinstance(count, int) and isinstance(limit, int) and count == len(obs) <= limit):
        return False, "paged or truncated (count, limit)"
    if d.get("offset", 0) != 0 or d.get("output_type", 1) != 1 or d.get("realtime_start") != d.get("realtime_end"):
        return False, "offset, output_type or a real-time range"
    if str(d.get("observation_start", "1600-01-01")) > str(obs[0]["date"]):
        return False, "observation_start after the first observation"
    return True, ""


def _sha(body: bytes, name: str = "sha256") -> str:
    return hashlib.new(name, body).hexdigest()


def plan(scratch: Path) -> list[dict]:
    """Every file to import: {source (absolute), dest (relative to data/raw), kind, series_id, check, provenance}."""
    scratch = Path(scratch)
    out = []
    for sid, (folder, publisher, lag) in FRED_IMPORTS.items():
        meta = scratch / META_DIRS.get(folder, folder) / f"{sid}_meta.json"
        prov = {"publisher": publisher, "publication_lag": lag, "meta": meta if meta.exists() else None}
        out.append({"source": scratch / folder / f"{sid}.json", "dest": f"fred/{sid}.json", "kind": "fred_observations",
                    "series_id": sid, "check": None, "provenance": prov})
        if meta.exists():
            out.append({"source": meta, "dest": f"fred/{sid}_meta.json", "kind": "fred_meta", "series_id": sid,
                        "check": None, "provenance": prov})
    for research, dest, kind, digest, prov in PUBLISHER_IMPORTS:
        out.append({"source": scratch / research, "dest": dest, "kind": "publisher", "series_id": None,
                    "check": (kind, digest), "provenance": prov})
    return out


def _fred_record(c: dict, body: bytes) -> dict:
    sid = c["series_id"]
    meta = {}
    if c["provenance"].get("meta") is not None:
        try:
            meta = (json.loads(Path(c["provenance"]["meta"]).read_bytes()).get("seriess") or [{}])[0]
        except (ValueError, AttributeError):
            meta = {}
    first = None
    if c["kind"] == "fred_observations":
        obs = json.loads(body)["observations"]
        first = next((o["date"][:7] for o in obs if o.get("value") not in (".", "", None)), None)
    path = "/series" if c["kind"] == "fred_meta" else "/series/observations"
    return {"url": f"https://api.stlouisfed.org/fred{path}?series_id={sid}&api_key=REDACTED&file_type=json",
            "publisher": c["provenance"]["publisher"], "terms": FRED_TERMS, "frequency": meta.get("frequency"),
            "first_month": first or (str(meta.get("observation_start", ""))[:7] or None),
            "publication_lag": c["provenance"]["publication_lag"], "revised": "yes (current vintage)",
            "copyright": _copyright_note(meta.get("notes", ""))}


def run_import(scratch: Path, dest_root: Path | None = None, force: bool = False) -> dict:
    """Copy the research downloads the quarterly panel needs into data/raw; returns the import log."""
    scratch = Path(scratch).resolve()
    dest_root = Path(dest_root or config.RAW_DIR)
    sources_path = dest_root / "SOURCES.json"
    doc = jsonio.load(sources_path) if sources_path.exists() else {"version": 1, "files": {}}
    records = dict(doc.get("files", {}))
    rows, counts = [], {"copied": 0, "unchanged": 0, "conflict": 0, "rejected": 0, "missing": 0}
    for c in plan(scratch):
        source, dest = Path(c["source"]), dest_root / c["dest"]
        row = {"source": str(source.relative_to(scratch)) if source.is_relative_to(scratch) else str(source),
               "dest": c["dest"], "kind": c["kind"]}
        if not source.exists():
            row.update(action="missing")
            counts["missing"] += 1
            rows.append(row)
            continue
        body = source.read_bytes()
        digest = _sha(body)
        row["sha256"] = digest
        reason = None
        if c["check"]:
            kind, want = c["check"]
            if not _sha(body, kind).startswith(want.lower()):
                reason = f"{kind.upper()} differs from the research round's record"
        if c["kind"] == "fred_observations":
            ok, why = fred_full_request(body)
            reason = reason or (None if ok else f"not a full FRED request ({why})")
        if reason:
            row.update(action="rejected", reason=reason)
        elif dest.exists() and dest.read_bytes() == body:
            row.update(action="unchanged")
        elif dest.exists() and not force:
            row.update(action="conflict", reason="a different file exists (use --force to replace it)")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            shutil.copyfile(source, tmp)
            tmp.replace(dest)
            row.update(action="copied")
        counts[row["action"]] += 1
        if row["action"] == "copied" or (row["action"] == "unchanged" and c["dest"] not in records):
            prov = _fred_record(c, body) if c["kind"].startswith("fred") else dict(c["provenance"])
            records[c["dest"]] = {"path": c["dest"], "research_path": row["source"], "kind": c["kind"],
                                  "series_id": c["series_id"], **prov, "access_date": ACCESS_DATE,
                                  "sha256": digest, "bytes": len(body), "families": ["quarterly"],
                                  "imported_by": "scenariolab.qmodel.sources_q"}
        rows.append(row)
    doc["files"] = dict(sorted(records.items()))
    jsonio.dump(json.loads(redact(json.dumps(doc, ensure_ascii=False))), sources_path)
    log = {"run_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": str(scratch),
           "force": bool(force), "counts": counts, "files": rows}
    jsonio.dump(json.loads(redact(json.dumps(log, ensure_ascii=False))), dest_root / "import_log_quarterly.json")
    return log


# ---------------------------------------------------------------- cache-only readers

def raw_path(rel: str) -> Path:
    return config.RAW_DIR / rel


def _need(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"缺少原始文件 {path.relative_to(config.ROOT) if path.is_relative_to(config.ROOT) else path}"
                                f"（季度面板不下载数据；先运行 {IMPORT_HINT}）")
    return path


def fred(series_id: str, agg: str | None = None) -> pd.Series:
    """A cached FRED series on month-start dates (quarterly series at the quarter's first month), never downloaded.
    agg 'eop' / 'avg' reads the monthly aggregate of a daily series."""
    fdir = config.RAW_DIR / "fred"
    stem = series_id + src.AGG_SUFFIX[agg]
    for path, reader in ((fdir / f"{stem}.json", src.read_fred_json),
                         (fdir / f"{series_id}{src.CSV_SUFFIX[agg]}.csv", src.read_fred_csv)):
        if path.exists() and path.stat().st_size > 0:
            s = reader(path, series_id).dropna()
            if len(s) and (s.index.day == 1).all():
                s.index = s.index.to_period("M").to_timestamp()
            return s.sort_index()
    raise FileNotFoundError(f"缺少 FRED 序列 {series_id}（data/raw/fred/{stem}.json；先运行 {IMPORT_HINT}）")


def shiller_workbook() -> Path:
    """The Shiller workbook the quarterly panel reads: among config_q.SHILLER_FILES, the first whose earnings reach
    SHILLER_MIN_EARNINGS (the 2026 workbook, E and D to 2026-06); else the existing one with the latest earnings."""
    best, best_end = None, None
    for rel in cq.SHILLER_FILES:
        p = raw_path(rel)
        if not p.exists():
            continue
        e = shiller_frame(p)["earnings"].last_valid_index()
        if e is not None and e >= pd.Timestamp(cq.SHILLER_MIN_EARNINGS):
            return p
        if best is None or (e is not None and e > best_end):
            best, best_end = p, e
    if best is None:
        raise FileNotFoundError(f"缺少 Shiller 工作簿（{', '.join(cq.SHILLER_FILES)}；先运行 {IMPORT_HINT}）")
    return best


def shiller_frame(path: Path | None = None) -> pd.DataFrame:
    """Shiller's ie_data 'Data' sheet: price, dividend, earnings, cpi, gs10, cape, tr_cape, ecy (monthly, month start).
    Columns are found by their header names (valuation/valblock.load_shiller_full); 'Yield' is his excess CAPE yield."""
    path = _need(Path(path) if path else shiller_workbook())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_excel(path, sheet_name="Data", header=None, engine="xlrd")
    head = next(i for i in range(min(len(raw), 40)) if str(raw.iat[i, 0]).strip() == "Date")
    want = {"P": "price", "D": "dividend", "E": "earnings", "CPI": "cpi", "Rate GS10": "gs10", "CAPE": "cape",
            "TR CAPE": "tr_cape", "Yield": "ecy"}
    cols: dict[str, int] = {}
    for c in range(raw.shape[1]):
        name = str(raw.iat[head, c]).strip()
        if name in want and want[name] not in cols:
            cols[want[name]] = c
    body = raw.iloc[head + 1:]
    stamp = pd.to_numeric(body[0], errors="coerce")
    body, stamp = body[stamp.notna()], stamp[stamp.notna()].to_numpy(float)
    year = np.floor(stamp).astype(int)
    month = np.rint((stamp - year) * 100).astype(int)          # 1871.01 = January, 1871.1 = October
    index = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))
    out = pd.DataFrame({k: pd.to_numeric(body[c], errors="coerce").to_numpy(float) for k, c in cols.items()},
                       index=pd.DatetimeIndex(index))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["path"] = str(path)
    return out


def french_book_to_market() -> pd.DataFrame:
    """Ken French 12 industries, annual 'Sum of BE / Sum of ME' (row t = BE of fiscal year t-1 over ME of December
    t-1; portfolios are formed at the end of June of t), dated at June of year t. -99.99 = missing."""
    path = _need(config.RAW_DIR / "french" / "12_Industry_Portfolios_CSV.zip")
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        lines = zf.read(member).decode("latin-1").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == "Sum of BE / Sum of ME")
    header = [h.strip() for h in lines[i + 1].split(",")[1:]]
    rows = []
    for ln in lines[i + 2:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 4:
            break
        rows.append([int(parts[0])] + [float(x) for x in parts[1:len(header) + 1]])
    df = pd.DataFrame(rows, columns=["year"] + header).set_index("year")
    df.index = pd.DatetimeIndex([pd.Timestamp(year=int(y), month=6, day=1) for y in df.index])
    return df.mask(df <= -99.0)


def french_tables(name: str) -> dict[str, pd.DataFrame]:
    """All monthly tables of a cached Ken French zip (data/raw/french/<name>_CSV.zip), never downloaded."""
    path = _need(config.RAW_DIR / "french" / f"{name}_CSV.zip")
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return src._parse_french_tables(zf.read(member).decode("latin-1"))


def french_table(name: str, title_prefix: str) -> pd.DataFrame:
    for title, df in french_tables(name).items():
        if title.startswith(title_prefix):
            return df
    raise KeyError(f"{title_prefix!r} not found in {name}")


def hlw() -> pd.DataFrame:
    """NY Fed HLW one-sided US estimates: r* and trend growth g (percent), dated at the quarter's first month."""
    rows = src._xlsx_rows(_need(config.RAW_DIR / "nyfed" / "Holston_Laubach_Williams_current_estimates.xlsx"),
                          "HLW Estimates")
    hdr = next(i for i, r in enumerate(rows[:20]) if r and str(r[0]).strip() == "Date")
    groups, countries = rows[hdr - 1], rows[hdr]
    want = {"rstar": "Natural Rate", "g": "Trend Growth"}
    cols: dict[str, int] = {}
    group = ""
    for j in range(len(countries)):
        if j < len(groups) and groups[j]:
            group = str(groups[j])
        for key, label in want.items():
            if label in group and str(countries[j]).strip() == "US" and key not in cols:
                cols[key] = j
    if set(cols) != set(want):
        raise ValueError("HLW: US r* / trend growth columns not found")
    body = [r for r in rows[hdr + 1:] if r and hasattr(r[0], "year")]
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]).to_period("Q").to_timestamp() for r in body])
    return pd.DataFrame({k: [src._num(r[c]) if len(r) > c else np.nan for r in body] for k, c in cols.items()},
                        index=idx)


def mbos_dtc() -> pd.Series:
    """Philadelphia Fed Manufacturing Business Outlook Survey: current delivery-time diffusion index DTC (SA)."""
    df = pd.read_csv(_need(config.RAW_DIR / "philfed" / "bos_dif.csv"))
    idx = pd.to_datetime(df.pop("DATE"), format="%b-%y")
    idx = pd.DatetimeIndex([t - pd.DateOffset(years=100) if t.year > 2030 else t for t in idx])   # 68 -> 1968
    return pd.Series(pd.to_numeric(df["DTC"], errors="coerce").to_numpy(float), index=idx).sort_index().dropna()


def umich_px5() -> pd.Series:
    """University of Michigan median expected inflation over the next 5-10 years (PX5_MD), monthly."""
    df = pd.read_csv(_need(config.RAW_DIR / "umich" / "tbmpx1px5.csv"))
    months = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                              "August", "September", "October", "November", "December"])}
    idx = pd.to_datetime(pd.DataFrame({"year": df["YYYY"].astype(int), "month": df["Month"].map(months), "day": 1}))
    return pd.Series(pd.to_numeric(df["PX5_MD"], errors="coerce").to_numpy(float),
                     index=pd.DatetimeIndex(idx)).sort_index().dropna()


def epu_health() -> pd.Series:
    """Baker-Bloom-Davis categorical EPU: '5. Health care' (monthly, from 1985)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(_need(config.RAW_DIR / "policyuncertainty" / "Categorical_EPU_Data.xlsx"))
    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    idx = pd.to_datetime(pd.DataFrame({"year": df["Year"].astype(int), "month": df["Month"].astype(int), "day": 1}))
    col = next(c for c in df.columns if str(c).strip().lower().endswith("health care"))
    return pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(float),
                     index=pd.DatetimeIndex(idx)).sort_index().dropna()


def epu_news() -> pd.Series:
    """Baker-Bloom-Davis news-based EPU, one scale throughout (the project's Main News Index reader)."""
    return src.parse_epu_rows(src._xlsx_rows(_need(config.RAW_DIR / "policyuncertainty" /
                                                   "US_Policy_Uncertainty_Data.xlsx"), "Main News Index"))


def gpr_historical() -> pd.Series:
    """Caldara-Iacoviello historical geopolitical risk index GPRH, monthly."""
    return src.parse_gpr_rows(src._xls_rows(_need(config.RAW_DIR / "iacoviello" / "data_gpr_export.xls"), 0))


def fmhpi_us() -> pd.Series:
    """Freddie Mac House Price Index, US, seasonally adjusted, monthly from 1975-01."""
    return src.parse_fmhpi_csv(_need(config.RAW_DIR / "freddiemac" / "fmhpi_master_file.csv"))


def file_digest(path: Path) -> str:
    return _sha(Path(path).read_bytes())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="季度面板的原始数据：从研究轮的下载导入 data/raw（只复制字节并核对哈希）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="import the research rounds' downloads (no network)")
    imp.add_argument("scratch", type=Path, help="the research scratchpad (holds factors/ and sim_redesign/)")
    imp.add_argument("--force", action="store_true", help="replace existing files that differ")
    args = ap.parse_args(argv)
    if not args.scratch.is_dir():
        ap.error(f"not a directory: {args.scratch}")
    log = run_import(args.scratch, force=args.force)
    c = log["counts"]
    print(f"[sources_q] copied {c['copied']}, unchanged {c['unchanged']}, conflicts {c['conflict']}, "
          f"rejected {c['rejected']}, missing {c['missing']}")
    for r in log["files"]:
        if r["action"] in ("conflict", "rejected", "missing"):
            print(f"[sources_q] {r['action']}: {r['source']} -> {r['dest']} ({r.get('reason', '')})")
    return 1 if c["conflict"] or c["rejected"] or c["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
