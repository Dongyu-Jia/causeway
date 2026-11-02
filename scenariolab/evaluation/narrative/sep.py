"""The H-SEP arm (SIM_DESIGN §8.3, U-15): the Fed's own real-time forecast as a scenario hypothesis, without an LLM.

At each year's last Summary of Economic Projections (the December release, November in 2007), the FOMC's medians for
the coming year (unemployment in Q4, real GDP growth Q4/Q4, PCE and core PCE inflation Q4/Q4; the fed funds median
only from 2012 on) become a QTable BY CODE (to_qtable):
    unemployment   level statements on the quarter-end rate, linear from the origin's rate to the Q4 median
    GDP growth     d_rgdp (annualized) equal to the Q4/Q4 median in each of the four quarters
    PCE inflation  CPI y/y = PCE + the training-period gap (CPI − PCE y/y over the 40 quarters to the train end),
                   linear from the origin's CPI y/y to the Q4 value; core PCE -> core CPI y/y the same way
    fed funds      (2012+) level statements on the observed rate, linear to the year-end median; before 2012 the
                   policy rate is left to the reaction function
    fc.stance      未说明 (H-SEP) or 中性 in every quarter (H-SEP+FC0: "no credit shock")
Each annual window runs from Q1 of the year after the SEP for 4 quarters, with the model fitted to the SEP's quarter
(about 18 non-overlapping windows from 2008). (H-SEP − O-A) is the hypothesis error of the Fed's forecast (§8.6).

Data. The real-time medians come from ALFRED (FRED's vintage archive) through the FRED API, which needs the project's
FRED_API_KEY (.env; it is sent only to api.stlouisfed.org, as data/sources.py does). The download needs the user's
approval and is never automatic:

    python -m scenariolab.evaluation.narrative sep plan                  # what would be downloaded (no network)
    python -m scenariolab.evaluation.narrative sep fetch --i-approve-download

fetch() first asks FRED for each series' metadata and records the title, units and first observation, so the ids
below are confirmed against FRED before any value is used; a series whose title does not name the Summary of
Economic Projections is not used. Files go to data/raw/sep/ (their own folder). Until then the arm reports
「需要先下载美联储 SEP 数据（需要你的同意）」.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ... import config
from ...qmodel import config_q as cq

RAW_DIR = config.RAW_DIR / "sep"
API_URL = "https://api.stlouisfed.org/fred"
REALTIME_START = "2007-10-01"
# role -> (median id, central-tendency midpoint id (fallback where the median is not published), what)
SERIES = {
    "unrate": ("UNRATEMD", "UNRATECTM", "失业率（第四季度均值）"),
    "gdp": ("GDPC1MD", "GDPC1CTM", "实际 GDP 增长（第四季度同比）"),
    "pce": ("PCECTPIMD", "PCECTPICTM", "PCE 通胀（第四季度同比）"),
    "core_pce": ("JCXFEMD", "JCXFECTM", "核心 PCE 通胀（第四季度同比）"),
    "ff": ("FEDTARMD", "FEDTARCTM", "联邦基金利率（年末，2012 年起）"),
}
GAP_SERIES = {"pce_index": "PCEPI", "core_pce_index": "PCEPILFE"}      # the CPI − PCE gap (current vintage)
FF_FROM_YEAR = 2012
GAP_QUARTERS = 40
FIRST_YEAR = 2008
H = 4
NEEDS_DATA = "需要先下载美联储当时的预测数据（下载要经你同意，在本机手动运行，见“给维护者”）"
FETCH_COMMAND = "python -m scenariolab.evaluation.narrative sep fetch --i-approve-download"
QUOTE = "美联储 SEP 中位数（由代码换算，不是叙事）"


class SepError(RuntimeError):
    """The SEP arm cannot run (Chinese message)."""


def series_ids() -> list[str]:
    ids = []
    for med, ctm, _ in SERIES.values():
        ids += [med] + ([ctm] if ctm else [])
    return ids + list(GAP_SERIES.values())


def plan() -> dict:
    """What fetch() would download: no network, nothing written."""
    reqs = []
    for sid in series_ids():
        reqs.append({"series_id": sid, "metadata": f"{API_URL}/series?series_id={sid}&file_type=json",
                     "observations": f"{API_URL}/series/observations?series_id={sid}&realtime_start={REALTIME_START}"
                                     "&realtime_end=9999-12-31&file_type=json",
                     "dest": str(RAW_DIR / f"{sid}.json")})
    return {"approved": False, "host": "api.stlouisfed.org", "needs": "FRED_API_KEY（.env）",
            "requests": reqs, "n_requests": 2 * len(reqs), "dest_dir": str(RAW_DIR),
            "text": "下载美联储 SEP 的实时版本（ALFRED）：每个序列先取元数据确认编号，再取全部历史版本。需要你的同意。"}


def fetch(*, approved: bool = False, refresh: bool = False) -> dict:
    """Download the plan's series into data/raw/sep/ (only with approved=True; never called by a test or a build)."""
    if not approved:
        raise SepError("下载美联储 SEP 数据需要你的同意：加 --i-approve-download（只下载到 data/raw/sep/）")
    import os
    import urllib.parse
    import urllib.request

    from ...data import sources
    from ...env import load_dotenv
    load_dotenv()
    key = sources.fred_key()
    if not key:
        raise SepError("没有配置 FRED_API_KEY（.env）：ALFRED 的历史版本只能经 FRED API 取得")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = {"downloaded": [], "skipped": [], "confirmed": {}, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for sid in series_ids():
        dest = RAW_DIR / f"{sid}.json"
        if dest.exists() and not refresh:
            out["skipped"].append(sid)
            continue
        body = {}
        for part, params in (("series", {"series_id": sid}),
                             ("series/observations", {"series_id": sid, "realtime_start": REALTIME_START,
                                                      "realtime_end": "9999-12-31"})):
            q = urllib.parse.urlencode({**params, "file_type": "json", "api_key": key})
            with urllib.request.urlopen(f"{API_URL}/{part}?{q}", timeout=60) as resp:   # noqa: S310 - fixed host
                body[part] = json.loads(resp.read().decode("utf-8"))
            time.sleep(0.6)
        meta = (body["series"].get("seriess") or [{}])[0]
        title = str(meta.get("title") or "")
        out["confirmed"][sid] = {"title": title, "units": meta.get("units"),
                                 "observation_start": meta.get("observation_start"),
                                 "sep": "Summary of Economic Projections" in title or sid in GAP_SERIES.values()}
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"series": meta, "observations": body["series/observations"].get("observations")},
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dest)
        out["downloaded"].append(sid)
    (RAW_DIR / "fetch_log.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def available(raw_dir: Path | None = None) -> bool:
    d = Path(raw_dir or RAW_DIR)
    return all((d / f"{SERIES[r][0]}.json").exists() for r in ("unrate", "gdp", "pce", "core_pce"))


def load(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """{series id: observations (date, value, realtime_start, realtime_end)} of the downloaded files."""
    d = Path(raw_dir or RAW_DIR)
    if not available(d):
        raise SepError(f"{NEEDS_DATA}：{FETCH_COMMAND}")
    out = {}
    for sid in series_ids():
        p = d / f"{sid}.json"
        if not p.exists():
            continue
        obs = json.loads(p.read_text(encoding="utf-8")).get("observations") or []
        df = pd.DataFrame(obs)
        if df.empty:
            continue
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        for c in ("date", "realtime_start", "realtime_end"):
            df[c] = pd.to_datetime(df[c].replace("9999-12-31", "2262-04-11"), errors="coerce")
        out[sid] = df.dropna(subset=["value"])
    return out


def vintage(df: pd.DataFrame, as_of) -> dict[int, float]:
    """{projection year: value} as known on `as_of` (the observations whose real-time period contains it)."""
    t = pd.Timestamp(as_of)
    v = df[(df["realtime_start"] <= t) & (df["realtime_end"] >= t)]
    return {int(d.year): float(x) for d, x in zip(v["date"], v["value"])}


def sep_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    """The release dates (vintage starts) of a SEP series."""
    return sorted(pd.to_datetime(df["realtime_start"].unique()))


def last_release(df: pd.DataFrame, year: int) -> pd.Timestamp | None:
    dates = [d for d in sep_dates(df) if d.year == year]
    return dates[-1] if dates else None


def _value(data: dict, role: str, as_of, year: int) -> tuple[float | None, str | None]:
    med, ctm, _ = SERIES[role]
    for sid in (med, ctm):
        if sid and sid in data:
            v = vintage(data[sid], as_of).get(year)
            if v is not None and np.isfinite(v):
                return v, sid
    return None, None


def gap(Q: pd.DataFrame, train_end: str, data: dict, which: str = "headline") -> float | None:
    """The training-period mean of CPI y/y − PCE y/y (headline or core) over GAP_QUARTERS quarters to the train end."""
    sid = GAP_SERIES["pce_index" if which == "headline" else "core_pce_index"]
    if sid not in data:
        return None
    df = data[sid].sort_values(["date", "realtime_start"]).drop_duplicates("date", keep="last")
    s = df.set_index("date")["value"]
    yoy = 100.0 * np.log(s / s.shift(12))
    q = yoy.groupby(yoy.index.to_period("Q")).last()
    col = "s_cpi_yoy" if which == "headline" else "s_core_yoy"
    te = pd.Period(train_end, "Q")
    both = pd.concat([Q[col].shift(-1), q], axis=1, keys=["cpi", "pce"]).loc[:te].dropna().iloc[-GAP_QUARTERS:]
    return float((both["cpi"] - both["pce"]).mean()) if len(both) else None


def windows(data: dict, last_quarter: str) -> list[dict]:
    """The annual windows: origin Q1 of year Y, the last SEP of Y − 1, train end (Y − 1)Q4, 4 quarters."""
    ref = data[SERIES["unrate"][0]]
    out = []
    last = pd.Period(last_quarter, "Q")
    for y in range(FIRST_YEAR, last.year + 1):
        rel = last_release(ref, y - 1)
        if rel is None or pd.Period(f"{y}Q4", "Q") > last:
            continue
        out.append({"code": f"SEP{y}", "year": y, "release": str(rel.date()), "origin": f"{y}Q1",
                    "train_end": f"{y - 1}Q4", "n_quarters": H})
    return out


def _lin(a: float, b: float, n: int = H) -> list[float]:
    return [a + (b - a) * (k + 1) / n for k in range(n)]


def to_qtable(win: dict, data: dict, Q: pd.DataFrame, *, fc0: bool = False) -> tuple[dict, dict]:
    """The window's QTable by code (see the module docstring) and a record of what it used."""
    y, rel = win["year"], win["release"]
    origin = pd.Period(win["origin"], "Q")
    rec = {"release": rel, "year": y, "used": {}, "notes": []}
    rows = [{"q": q, "x": {}, "fc": ({"stance": "中性", "quote": QUOTE} if fc0 else {"stance": "未说明"})}
            for q in range(1, H + 1)]
    cons = []
    u, sid = _value(data, "unrate", rel, y)
    if u is not None and "s_unrate" in Q.columns:
        u0 = float(Q.at[origin, "s_unrate"])
        for q, v in enumerate(_lin(u0, u), 1):
            cons.append({"kind": "level", "field": "s_unrate", "quarters": [q], "value": round(v, 4), "sd": 0.1,
                         "quote": QUOTE})
        rec["used"]["unrate"] = {"series": sid, "value": u}
    g, sid = _value(data, "gdp", rel, y)
    if g is not None:
        for r in rows:
            r["x"]["d_rgdp"] = {"value": round(g, 4), "sd": 0.5, "src": "derived", "quote": QUOTE}
        rec["used"]["gdp"] = {"series": sid, "value": g}
    for role, col, which in (("pce", "s_cpi_yoy", "headline"), ("core_pce", "s_core_yoy", "core")):
        p, sid = _value(data, role, rel, y)
        if p is None or col not in Q.columns:
            continue
        gp = gap(Q, win["train_end"], data, which)
        if gp is None:
            rec["notes"].append(f"没有 {which} 的 CPI − PCE 差：{role} 没有换算")
            continue
        target = p + gp
        start = float(Q.at[origin, col])
        for q, v in enumerate(_lin(start, target), 1):
            cons.append({"kind": "yoy" if col == "s_core_yoy" else "level", "field": "d_core" if col == "s_core_yoy"
                         else col, "quarters": [q], "value": round(v, 4), "sd": 0.2, "quote": QUOTE})
        rec["used"][role] = {"series": sid, "value": p, "gap": gp, "cpi_target": target}
    if y >= FF_FROM_YEAR:
        f, sid = _value(data, "ff", rel, y)
        if f is not None and "s_ff" in Q.columns:
            f0 = float(Q.at[origin, "s_ff"])
            for q, v in enumerate(_lin(f0, f), 1):
                cons.append({"kind": "level", "field": "s_ff", "quarters": [q], "value": round(max(v, cq.FF_FLOOR), 4),
                             "sd": 0.1, "quote": QUOTE})
            rec["used"]["ff"] = {"series": sid, "value": f}
    else:
        rec["notes"].append("2012 年以前 SEP 没有联邦基金利率中位数：政策利率由反应函数决定")
    qt = {"schema": cq.QTABLE_SCHEMA_ID, "model_card": cq.CARD_ID, "n_quarters": H, "rows": rows,
          "c_q": {"constraints": cons} if cons else {}, "events": [],
          "notes": ["由反应函数决定" if y < FF_FROM_YEAR else "政策利率按 SEP 的年末中位数", "SEP 由代码换算成情景表（不用大模型）"]}
    return qt, rec


def status() -> dict:
    """What the evaluation tab shows for the H-SEP arm."""
    if not available():
        return {"available": False, "text": NEEDS_DATA, "command": FETCH_COMMAND, "plan": plan()}
    log = RAW_DIR / "fetch_log.json"
    return {"available": True, "fetch_log": json.loads(log.read_text(encoding="utf-8")) if log.exists() else None}


def score_sep(*, data: dict | None = None, Q: pd.DataFrame | None = None, n_paths: int = 1000, n_boot=None,
              seed: int = 11, out_dir: Path | None = None) -> dict:
    """Score the H-SEP arms on every annual window (U, O-A, H-SEP, H-SEP+FC0; the quarterly stack, CMA historical):
    the rows, pooled tables and the paired differences (H-SEP − U, H-SEP − O-A, H-SEP+FC0 − U, O-A − U), written to
    data/results/narrative/sep_<date>/ (stats.json, score_rows.csv). Needs the downloaded SEP data."""
    from ...qmodel import panel as qpanel
    from ...qmodel import qtable as QT
    from . import common as C
    from . import score as SC
    from . import stats as ST
    data = data if data is not None else load()
    Q = qpanel.load_panel() if Q is None else Q
    wins = windows(data, str(Q.index[-2]))
    rows, notes = [], {}
    for i, w in enumerate(wins):
        entry = {"kind": "episode", "set": "sep", "bust": None, "n_quarters": H, "train_end": w["train_end"],
                 "origin": w["origin"]}
        try:
            ep = SC.Episode(w["code"], entry, Q=Q, seed=seed + i, n_paths=n_paths, n_boot=n_boot,
                            cmas=("historical",))
        except Exception as err:  # noqa: BLE001
            notes[w["code"]] = f"拟合不出来：{type(err).__name__}: {err}"
            continue
        ep.run("U")
        rows += ep.rows("U")
        vals = SC.oracle_values(Q, ep.quarters, C.ARMS["OA"]["classes"], ep.stack.x_fields)
        ep.run("OA", SC.oracle_statements(vals))
        rows += ep.rows("OA")
        for arm, fc0 in (("HS", False), ("HS0", True)):
            qt, rec = to_qtable(w, data, Q, fc0=fc0)
            read = QT.constraints(qt, stack=ep.stack, mode="test", narrative=None, source="user", strict=False)
            ep.run(arm, read["statements"], read["valuation"], read.get("cycle_odds"))
            rows += ep.rows(arm)
            notes.setdefault(w["code"], rec)
    R = pd.DataFrame(rows)
    codes = sorted(R["code"].unique()) if len(R) else []
    pairs = []
    for h in ("1Q", "2Q", "4Q", "end"):
        for a, b in (("Q:HS", "Q:U"), ("Q:HS", "Q:OA"), ("Q:HS0", "Q:U"), ("Q:OA", "Q:U")):
            p = ST.paired(R, (a, "historical"), (b, "historical"), h, "equity9", codes=codes) if len(R) else {}
            if p:
                pairs.append({**p, "set": "sep"})
    out = {"windows": wins, "pooled": ST.pooled(R, codes) if len(R) else [], "paired": pairs, "notes": notes,
           "label": "美联储当时的预测（SEP）：事前的假说，不用大模型"}
    d = Path(out_dir or (C.runs_dir() / f"sep_{time.strftime('%Y%m%d_%H%M%S')}"))
    d.mkdir(parents=True, exist_ok=True)
    R.to_csv(d / "score_rows.csv", index=False, float_format="%.8g")
    from ... import jsonio
    (d / "stats.json").write_text(jsonio.dumps(out), encoding="utf-8")
    return out
