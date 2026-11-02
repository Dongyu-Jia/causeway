"""The quarterly panel of the quarterly stack (SIM_DESIGN §3.2, §3.9, §14 P1).

Ports the research round's qmodel/qdata.py (the 52 x_q fields, the start-of-quarter states, the asset targets) and
valuation/valblock.py (valuation levels with explicit timing) onto the project's own data: the monthly panel
(data/processed/panel_monthly.csv) and the cached raw files in data/raw (qmodel/sources_q.py; nothing downloads).

    python -m scenariolab.qmodel.panel          # writes data/processed/panel_quarterly.csv + panel_quarterly_meta.json

One row per calendar quarter, 1973Q1 .. the quarter after the last complete data quarter ("today", start-of-quarter
states only; 2026Q4 on the 2026-09 monthly panel). Column roles (panel_quarterly_meta.json "columns"):

  x       the card's 52 x_q fields, realized during the quarter (同期): the "perfect translation" in training, what the
          in-quarter text states in a scenario. The L0 fields are the quarter's sum of the monthly factor changes
          times a unit scale (config_q.L0_FIELDS), exactly the monthly model's quarterly input.
  x_aux   model variables that are not QTable fields: d_cpi (headline CPI inflation, the BVAR's pc), d_oil_sup.
  s       the card's s_q states, known at the START of the quarter after publication lags (前定).
  s_aux   model states that are not card rows (inflation trend, Sahm gap, ZLB and core>4 flags, realized volatility,
          momentum, own P/D of each equity asset, ...), also known at the start.
  y       the card's outputs: y_<asset> (13 quarterly log price returns; cash = the bill's total return), y_mult
          (Δlog CAPE multiple), y_fund (Δlog CPI + Δlog E10 real), y_pd (Δlog CRSP P/D) and y_eps (display only).
  y_aux   log total returns yt_<asset>, simple income inc_<asset> = exp(yt) - exp(y), CRSP dividend growth.
  view    the card's views (display only, never model inputs).
  level   quarter-end reference levels (lvl_*) that the d_ fields are changes of (the factor layer's core levels).
  known   <field>_l1__known / <field>_l2__known: the x field's value in quarter q-1 / q-2 if it was published by the
          start of quarter q, else NaN (release lag = the slowest of the field's inputs, config_q.SERIES_LAG).

Every start-of-quarter value uses only observations that were published before the quarter began: each raw input
has a release lag (months after its reference period ends); tests rebuild the panel from inputs cut at those lags
and check the s / s_aux / view / known columns are unchanged (the truncation test). Data are today's vintage.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, jsonio
from ..data import panel as monthly_panel
from ..data import sources as src
from . import config_q as cq
from . import sources_q as sq

VERSION = "1"
QUARTER_START_MONTHS = (1, 4, 7, 10)
QUARTER_END_MONTHS = (3, 6, 9, 12)
CARD_ROLES = ("x", "s", "y", "view")
REALIZED_ROLES = ("x", "x_aux", "y", "y_aux", "level")  # NaN in the "today" row

# monthly panel columns read (asset p_ / r_ / d_ columns are added for every asset; all lag 0)
PANEL_COLUMNS = sorted(k.split(":", 1)[1] for k in cq.SERIES_LAG if k.startswith("panel:"))
FRED_MONTHLY = {"BOGMBASE": None, "PCTR": None, "PI": None, "UMCSENT": None, "MICH": None, "IGREA": None,
                "TWEXOMTH": None, "DTWEXEMEGS": "eop", "IRLTLT01DEM156N": None, "WPU0531": None, "CP3M": None,
                "CPF3M": None, "TB3MS": None, "CLF16OV": None, "DGDSRA3M086SBEA": None, "DSERRA3M086SBEA": None,
                "DFII10": "eop", "DGS5": "eop", "TWEXBPA": None, "RTWEXBGS": None, "CUSR0000SEHA": None,
                "INVEST": None, "TLAACBM027SBOG": None, "DSPI": None}
FRED_QUARTERLY = ["GDP", "GDPC1", "FDEFX", "W055RC1Q027SBEA", "B679RA3Q086SBEA", "A008RA3Q086SBEA",
                  "A011RA3Q086SBEA", "EXPGSC1", "B235RC1Q027SBEA", "A255RC1Q027SBEA", "FGDEF", "CP",
                  "A053RC1Q027SBEA", "W328RC1Q027SBEA", "B456RC1Q027SBEA", "OPHNFB", "PRS85006173",
                  "A679RC1Q027SBEA", "FYGFGDQ188S", "DRTSCILM", "CORALACBS", "BOGZ1FL664090005Q",
                  "BOGZ1FL664190005Q", "BOGZ1FL075035503Q", "NCBEILQ027S", "BOGZ1FA105050005Q",
                  "BOGZ1FA106121075Q", "NCBCEBQ027S", "BCNSDODNS", "CMDEBT", "FYGFDPUN", "FDHBFIN", "TDSP",
                  "USSTHPI"]
FRENCH_WEIGHT_GROUPS = {"energy": ["Enrgy"], "industrials": ["Manuf"], "staples": ["NoDur"], "healthcare": ["Hlth"],
                        "utilities": ["Utils"], "financials": ["Money"], "buseq": ["BusEq"],
                        "other": ["Durbl", "Chems", "Telcm", "Shops", "Other"]}
GROUPS = {"y_assets": [f"y_{a}" for a in cq.ASSETS],
          "s_w": [f"s_w_{g}" for g in FRENCH_WEIGHT_GROUPS]}   # card rows that are a block of panel columns


# ---------------------------------------------------------------- inputs

@dataclass
class Inputs:
    """Every raw series the build reads, keyed as in config_q.SERIES_LAG ('panel:f_oil', 'fred:GDP', ...).

    Monthly series sit on month-start dates; quarterly ones at the quarter's first month (FRED convention); the
    French annual book-to-market rows at June of their year. `freq` is "M", "Q" or "A"."""
    series: dict[str, pd.Series]
    freq: dict[str, str]
    files: dict = field(default_factory=dict)        # provenance of the files read (path -> sha256, notes)

    def __getitem__(self, key: str) -> pd.Series:
        return self.series[key]

    def lag(self, key: str) -> int:
        return series_lag(key)


def series_lag(key: str) -> int:
    """Release lag of an input key (months after its reference period's last month)."""
    if key in cq.SERIES_LAG:
        return int(cq.SERIES_LAG[key])
    base = ":".join(key.split(":")[:2])
    if base in cq.SERIES_LAG:
        return int(cq.SERIES_LAG[base])
    if key.startswith(("panel:p_", "panel:r_", "panel:d_", "french:w:", "french:p:")):
        return 0                                       # month-end asset prices, dividends, market values
    raise KeyError(f"no release lag for input {key!r} (add it to config_q.SERIES_LAG)")


def _month_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_inputs(panel: pd.DataFrame | None = None) -> Inputs:
    """Read every raw input from the cache (the monthly panel, data/raw); raises FileNotFoundError naming the import
    command when a file is missing (the quarterly panel never downloads)."""
    P = monthly_panel.load_panel() if panel is None else panel
    series: dict[str, pd.Series] = {}
    freq: dict[str, str] = {}
    files: dict = {"panel_monthly.csv": _digest(config.PANEL_PATH) if panel is None else "in memory"}
    sq.start_recording()
    try:
        _load_series(P, series, freq, files)
    finally:
        read = sq.stop_recording()
    files["raw"] = {_rel(p): _digest(p) for p in read}
    return Inputs(series=series, freq=freq, files=files)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(config.ROOT.resolve()))
    except ValueError:
        return str(p)


def _load_series(P: pd.DataFrame, series: dict, freq: dict, files: dict) -> None:
    def add(key, s, f="M"):
        series[key] = s.astype(float).sort_index()
        freq[key] = f

    for col in PANEL_COLUMNS:
        add(f"panel:{col}", P[col])
    for a in cq.ASSETS:
        for k in ("p", "r", "d"):
            add(f"panel:{k}_{a}", P[f"{k}_{a}"])
    for sid, agg in FRED_MONTHLY.items():
        add(f"fred:{sid}", sq.fred(sid, agg))
    for sid in FRED_QUARTERLY:
        s = sq.fred(sid)
        s.index = s.index.to_period("Q").to_timestamp()
        add(f"fred:{sid}", s[~s.index.duplicated(keep="last")], "Q")
    add("fmhpi:us_sa", _month_index(sq.fmhpi_us()))
    add("epu:news", _month_index(sq.epu_news()))
    add("epu:health", _month_index(sq.epu_health()))
    add("gpr:gprh", _month_index(sq.gpr_historical()))
    add("mbos:dtc", _month_index(sq.mbos_dtc()))
    add("umich:px5", _month_index(sq.umich_px5()))
    h = sq.hlw()
    add("hlw:rstar", h["rstar"].dropna(), "Q")
    add("hlw:g", h["g"].dropna(), "Q")
    wb = sq.shiller_workbook()
    S = sq.shiller_frame(wb)
    files["shiller"] = {"path": str(wb.relative_to(config.ROOT)) if wb.is_relative_to(config.ROOT) else str(wb),
                        "sha256": _digest(wb), "last_earnings": _month(S["earnings"].last_valid_index()),
                        "last_cape": _month(S["cape"].last_valid_index())}
    for col in ("price", "cpi", "earnings", "dividend", "cape"):
        add(f"shiller:{col}", S[col].dropna())
    bm = sq.french_book_to_market()
    for ind in sorted(set(cq.FRENCH_INDUSTRY.values())):
        add(f"french:bm:{ind}", bm[ind].dropna(), "A")
    nfirms = sq.french_table("12_Industry_Portfolios", "Number of Firms in Portfolios")
    size = sq.french_table("12_Industry_Portfolios", "Average Firm Size")
    mv = (nfirms * size).dropna(how="all")
    for ind in mv.columns:
        add(f"french:w:{ind}", mv[ind].dropna())
    ex_div = sq.french_table("12_Industry_Portfolios_Wout_Div", "Average Value Weighted Returns -- Monthly") / 100.0
    for ind in sorted(set(cq.FRENCH_INDUSTRY.values())):
        add(f"french:p:{ind}", ex_div[ind].dropna())


def truncate_inputs(inputs: Inputs, asof: str | pd.Timestamp) -> Inputs:
    """The inputs as published by the END of month `asof`: each series keeps the observations whose reference
    period ended at least its release lag before (monthly: month <= asof - lag; quarterly: the quarter's last month
    <= asof - lag; annual French rows, dated June: <= asof - lag)."""
    asof = pd.Timestamp(asof).to_period("M").to_timestamp()
    out = {}
    for key, s in inputs.series.items():
        cut = asof - pd.DateOffset(months=series_lag(key))
        ref_end = s.index + pd.DateOffset(months=2) if inputs.freq[key] == "Q" else s.index
        out[key] = s[ref_end <= cut]
    return Inputs(series=out, freq=dict(inputs.freq), files={**inputs.files, "truncated_asof": asof.strftime("%Y-%m")})


# ---------------------------------------------------------------- quarterly helpers (pandas Period "Q")

def _to_q(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).to_period("Q")
    return s[~s.index.duplicated(keep="last")]


def q_sum(m: pd.Series, scale: float = 1.0, min_count: int = 3) -> pd.Series:
    """scale x the sum of the quarter's monthly values; NaN unless all `min_count` months are observed."""
    g = m.groupby(pd.DatetimeIndex(m.index).to_period("Q"))
    out = g.sum(min_count=min_count)
    out[g.count() < min_count] = np.nan
    return scale * out


def q_end(m: pd.Series) -> pd.Series:
    """The value in the quarter's last month (NaN when that month is missing)."""
    return _to_q(m[pd.DatetimeIndex(m.index).month.isin(QUARTER_END_MONTHS)])


def q_avg(m: pd.Series, min_count: int = 3) -> pd.Series:
    g = m.groupby(pd.DatetimeIndex(m.index).to_period("Q"))
    out = g.mean()
    out[g.count() < min_count] = np.nan
    return out


def q_native(s: pd.Series) -> pd.Series:
    """A quarterly series dated at its first month -> quarters."""
    return _to_q(s)


def at_start(m: pd.Series, lag: int = 0) -> pd.Series:
    """The monthly value known at the start of each quarter q: the value of month (first month of q) - 1 - lag."""
    s = m.copy()
    s.index = pd.DatetimeIndex(s.index) + pd.DateOffset(months=1 + lag)
    return _to_q(s[s.index.month.isin(QUARTER_START_MONTHS)])


def quarters_back(lag: int) -> int:
    """How many quarters back the latest quarterly value known at a quarter's start lies (lag after its end)."""
    return 1 + int(math.ceil(lag / 3.0))


def qshift(q: pd.Series, k: int) -> pd.Series:
    """Move a quarterly series k quarters later by its labels (the value of quarter j lands on j + k)."""
    out = q.copy()
    out.index = pd.PeriodIndex(out.index, freq="Q") + k
    return out


def known_quarterly(q: pd.Series, lag: int) -> pd.Series:
    """A quarterly value (by its own quarter) as known at the start of each quarter: the value of q - k, k from the lag."""
    return qshift(q, quarters_back(lag))


def dlog(s: pd.Series, k: float = 400.0) -> pd.Series:
    return k * np.log(s.where(s > 0)).diff()


def _digest(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def inputs_digest(inputs: Inputs) -> str:
    """SHA-256 of the inputs as files: the bytes of every raw file the build read (and the monthly panel's), the input
    keys and the truncation cut, if any -- not the parsed floats, which pandas 2.x and 3.x read one ulp apart (the
    digest is the panel's identity in the freshness test and the adapter version, so it must not depend on the
    environment). An in-memory panel (load_inputs(panel=...)) falls back to values_digest."""
    f = inputs.files
    if f.get("panel_monthly.csv") == "in memory" or not f.get("raw"):
        return values_digest(inputs)
    h = hashlib.sha256()
    h.update(f"panel_monthly.csv {f['panel_monthly.csv']}\n".encode())
    for rel_p, sha in sorted(f["raw"].items()):
        h.update(f"{rel_p} {sha}\n".encode())
    h.update(("keys " + ",".join(sorted(inputs.series)) + "\n").encode())
    h.update(f"truncated {f.get('truncated_asof', '')}".encode())
    return h.hexdigest()


def values_digest(inputs: Inputs) -> str:
    """SHA-256 over every input series (name, dates, values): exact, so environment-dependent (pandas parses some files
    one ulp differently); recorded for reference only."""
    h = hashlib.sha256()
    for key in sorted(inputs.series):
        s = inputs.series[key]
        h.update(key.encode())
        h.update(pd.DatetimeIndex(s.index).strftime("%Y-%m").str.cat(sep=",").encode())
        h.update(np.ascontiguousarray(s.to_numpy(float)).tobytes())
    return h.hexdigest()


def _month(t) -> str | None:
    return None if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m")


def _ratio_q(a: pd.Series, b: pd.Series, scale: float = 100.0) -> pd.Series:
    return scale * a / b.reindex(a.index)


# ---------------------------------------------------------------- build

def build(inputs: Inputs) -> tuple[pd.DataFrame, dict]:
    """The quarterly panel and its column registry from `inputs` (load_inputs() or truncate_inputs(...))."""
    inp = inputs
    pm = lambda c: inp[f"panel:{c}"]          # noqa: E731
    fr = lambda s: inp[f"fred:{s}"]           # noqa: E731
    p_eq = pm("p_us_equity").dropna()
    data_end = p_eq.index.max()
    last_q = data_end.to_period("Q") if data_end.month in QUARTER_END_MONTHS else data_end.to_period("Q") - 1
    today = last_q + 1
    QI = pd.period_range(cq.FIRST_QUARTER, today, freq="Q")
    cols: dict[str, pd.Series] = {}
    meta: dict[str, dict] = {}

    def put(name, s, role, unit, timing, sources, text, card=None, note=""):
        cols[name] = s.astype(float).reindex(QI)
        lag = max((series_lag(k) for k in sources), default=0) if sources else 0
        meta[name] = {"role": role, "timing": timing, "unit": unit, "card": card if card is not None else
                      (name if role in CARD_ROLES else None), "sources": list(sources), "lag_months": lag,
                      "source": text, "note": note}

    # ======================== x: the 52 x_q fields (same quarter)
    UNIT = {"d_oil": "log%/季", "d_ip": "log%/季", "d_cpi_goods": "%年化", "d_cpi_serv": "%年化", "d_ff": "pp/季",
            "d_y10": "pp/季", "d_spread": "pp/季", "d_usd": "log%/季", "d_unrate": "pp/季", "d_gz": "pp/季",
            "d_tp10": "pp/季", "d_metals_oil": "log%/季"}
    for f, (col, k) in cq.WINDOW_SUM_FIELDS.items():
        put(f, q_sum(pm(col), k), "x", UNIT[f], "same_q", [f"panel:{col}"],
            f"{k:g} x the quarter's sum of the monthly panel's {col} (L0: F's quarterly input)")
    gdp = q_native(fr("GDP"))
    put("d_bs", (100.0 * q_avg(fr("BOGMBASE")) / gdp.reindex(q_avg(fr("BOGMBASE")).index)).diff(), "x", "pp GDP/季",
        "same_q", ["fred:BOGMBASE", "fred:GDP"], "monetary base (quarterly average) / nominal GDP, change",
        note="the monetary base includes currency (NSA)")
    put("d_deficit", _ratio_q(-q_native(fr("FGDEF")), gdp).diff(), "x", "pp GDP/季", "same_q",
        ["fred:FGDEF", "fred:GDP"], "-FGDEF / GDP (NIPA federal net lending), change")
    pi_q = q_avg(fr("PI"))
    put("d_transfers", _ratio_q(q_avg(fr("PCTR")), pi_q).diff(), "x", "pp/季", "same_q", ["fred:PCTR", "fred:PI"],
        "personal current transfer receipts / personal income (quarterly averages), change")
    put("d_defense", _ratio_q(q_native(fr("FDEFX")), gdp).diff(), "x", "pp GDP/季", "same_q",
        ["fred:FDEFX", "fred:GDP"], "national defense consumption and investment / GDP, change")
    cpq = q_native(fr("CP"))
    put("d_corptax", (100.0 * (1.0 - cpq / q_native(fr("A053RC1Q027SBEA")).reindex(cpq.index))).diff(), "x",
        "pp/季", "same_q", ["fred:CP", "fred:A053RC1Q027SBEA"],
        "1 - after-tax book profits (CP) / pre-tax book profits (A053RC1Q027SBEA), change")
    put("d_perstax", _ratio_q(q_native(fr("W055RC1Q027SBEA")), pi_q).diff(), "x", "pp/季", "same_q",
        ["fred:W055RC1Q027SBEA", "fred:PI"], "personal current taxes / personal income, change")
    put("d_tariff", _ratio_q(q_native(fr("B235RC1Q027SBEA")), q_native(fr("A255RC1Q027SBEA"))).diff(), "x", "pp/季",
        "same_q", ["fred:B235RC1Q027SBEA", "fred:A255RC1Q027SBEA"], "customs duties / goods imports (NIPA), change")
    sent = fr("UMCSENT")          # quarterly surveys before 1978: a value holds for the next two months
    sent = sent.reindex(pd.date_range(sent.index.min(), data_end, freq="MS")).ffill(limit=2) if len(sent) else sent
    put("d_sent", q_end(sent).diff(), "x", "指数点/季", "same_q", ["fred:UMCSENT"],
        "Michigan sentiment at the quarter's last month, change (quarterly surveys before 1978 held two months)")
    put("d_hpi", dlog(q_end(inp["fmhpi:us_sa"])), "x", "%年化", "same_q", ["fmhpi:us_sa"],
        "Freddie Mac HPI (US, SA), quarter-end log change, annualized")
    put("x_lending_std", q_native(fr("DRTSCILM")), "x", "%", "same_q", ["fred:DRTSCILM"],
        "SLOOS net share of banks tightening C&I standards (large and middle-market firms)",
        note="FRED dates the April survey in Q2; it asks about the past three months (timing 滞后一季)")
    put("d_igrea", q_avg(fr("IGREA")).diff(), "x", "指数/季", "same_q", ["fred:IGREA"],
        "Kilian index of global real economic activity, quarterly average, change")
    em = src.splice(fr("DTWEXEMEGS"), fr("TWEXOMTH"))
    put("d_usd_em", dlog(q_end(em), 100.0), "x", "log%/季", "same_q",
        ["fred:TWEXOMTH", "fred:DTWEXEMEGS"],
        "dollar vs EM currencies: OITP index (monthly average) to 2005-12, EME index (month-end) from 2006-01, "
        "chained by their ratio in 2006-01; quarter-end log change", note="research version spliced without "
        "chaining (one jump in 2006Q1); here the two indices are chained")
    put("d_bund", q_end(fr("IRLTLT01DEM156N")).diff(), "x", "pp/季", "same_q", ["fred:IRLTLT01DEM156N"],
        "German 10-year government bond yield (OECD, monthly average) at the quarter's last month, change")
    put("d_natgas", dlog(q_avg(fr("WPU0531")), 100.0), "x", "log%/季", "same_q", ["fred:WPU0531"],
        "PPI natural gas, quarterly average, log change")
    put("x_supply_chain", q_avg(inp["mbos:dtc"]), "x", "扩散指数", "same_q", ["mbos:dtc"],
        "Philadelphia Fed MBOS current delivery-time diffusion index (DTC), quarterly average")
    put("d_itinv", dlog(q_native(fr("B679RA3Q086SBEA"))), "x", "%年化", "same_q", ["fred:B679RA3Q086SBEA"],
        "real investment in information processing equipment and software (quantity index), annualized")
    put("d_lprod", dlog(q_native(fr("OPHNFB"))), "x", "%年化", "same_q", ["fred:OPHNFB"],
        "nonfarm business output per hour, annualized")
    clf = fr("CLF16OV")
    dl = np.log(clf).diff()
    jan = dl.index.month == 1
    dl_clean = dl.where(~jan, (dl.shift(1) + dl.shift(-1)) / 2.0)
    lf_level = np.exp(dl_clean.fillna(0.0).cumsum())
    put("d_lf", dlog(q_avg(lf_level)), "x", "%年化", "same_q", ["fred:CLF16OV"],
        "civilian labor force, quarterly average, annualized; each January change replaced by the mean of the "
        "December and February changes (population controls)")
    put("d_lshare", dlog(q_native(fr("PRS85006173")), 100.0), "x", "log%/季", "same_q", ["fred:PRS85006173"],
        "nonfarm business labor share (BLS), log change")
    cp_rate = pd.concat([fr("CP3M")[fr("CP3M").index < pd.Timestamp("1997-01-01")],
                         fr("CPF3M")[fr("CPF3M").index >= pd.Timestamp("1997-01-01")]]).sort_index()
    put("d_cpbill", q_avg(cp_rate - fr("TB3MS").reindex(cp_rate.index)).diff(), "x", "pp/季", "same_q",
        ["fred:CP3M", "fred:CPF3M", "fred:TB3MS"],
        "3-month commercial paper (CP3M to 1996-12, CPF3M from 1997-01) minus the 3-month bill, quarterly average, "
        "change")
    ta, tl = q_native(fr("BOGZ1FL664090005Q")), q_native(fr("BOGZ1FL664190005Q"))
    lev = ta / (ta - tl.reindex(ta.index))
    put("d_bdlev", dlog(lev.where(lev > 0), 100.0), "x", "log%/季", "same_q",
        ["fred:BOGZ1FL664090005Q", "fred:BOGZ1FL664190005Q"],
        "Z.1 L.130 broker-dealer assets / (assets - liabilities), log change")
    put("d_epu", dlog(q_avg(inp["epu:news"]), 100.0), "x", "log%/季", "same_q", ["epu:news"],
        "Baker-Bloom-Davis news-based EPU (one scale throughout), quarterly average, log change")
    put("d_gpr", dlog(q_avg(inp["gpr:gprh"]), 100.0), "x", "log%/季", "same_q", ["gpr:gprh"],
        "Caldara-Iacoviello historical GPR index, quarterly average, log change")
    put("d_epu_health", dlog(q_avg(inp["epu:health"]), 100.0), "x", "log%/季", "same_q", ["epu:health"],
        "categorical EPU, health care, quarterly average, log change")
    put("d_infexp1", q_avg(fr("MICH")).diff(), "x", "pp/季", "same_q", ["fred:MICH"],
        "Michigan 1-year inflation expectation, quarterly average, change")
    px5 = inp["umich:px5"]
    px5 = px5[px5.index >= pd.Timestamp("1990-04-01")]
    put("d_infexp_lr", q_avg(px5).diff(), "x", "pp/季", "same_q", ["umich:px5"],
        "Michigan 5-10-year inflation expectation (median), quarterly average from 1990Q2, change")
    put("d_rgdp", dlog(q_native(fr("GDPC1"))), "x", "%年化", "same_q", ["fred:GDPC1"], "real GDP, annualized")
    put("d_core", dlog(q_end(pm("lvl_core"))), "x", "%年化", "same_q", ["panel:lvl_core"],
        "core CPI at the quarter's last month, annualized log change")
    put("d_wage", dlog(q_end(pm("lvl_wage"))), "x", "%年化", "same_q", ["panel:lvl_wage"],
        "production and nonsupervisory average hourly earnings, annualized log change")
    put("d_y2", q_end(pm("lvl_y2")).diff(), "x", "pp/季", "same_q", ["panel:lvl_y2"],
        "2-year Treasury yield, month-end of the quarter's last month, change")
    put("d_tips10", q_end(fr("DFII10")).diff(), "x", "pp/季", "same_q", ["fred:DFII10"],
        "10-year TIPS yield, month-end of the quarter's last month, change")
    put("d_cons_goods", dlog(q_avg(fr("DGDSRA3M086SBEA"))), "x", "%年化", "same_q", ["fred:DGDSRA3M086SBEA"],
        "real PCE goods (quantity index), quarterly average, annualized")
    put("d_cons_serv", dlog(q_avg(fr("DSERRA3M086SBEA"))), "x", "%年化", "same_q", ["fred:DSERRA3M086SBEA"],
        "real PCE services (quantity index), quarterly average, annualized")
    put("d_capex", dlog(q_native(fr("A008RA3Q086SBEA"))), "x", "%年化", "same_q", ["fred:A008RA3Q086SBEA"],
        "real nonresidential fixed investment (quantity index), annualized")
    put("d_resinv", dlog(q_native(fr("A011RA3Q086SBEA"))), "x", "%年化", "same_q", ["fred:A011RA3Q086SBEA"],
        "real residential investment (quantity index), annualized")
    put("d_exports", dlog(q_native(fr("EXPGSC1"))), "x", "%年化", "same_q", ["fred:EXPGSC1"],
        "real exports of goods and services, annualized")
    put("d_profits", dlog(cpq, 100.0), "x", "log%/季", "same_q", ["fred:CP"],
        "corporate profits after tax without IVA/CCAdj (NIPA CP), log change")
    put("d_cre", dlog(q_native(fr("BOGZ1FL075035503Q")), 100.0), "x", "log%/季", "same_q",
        ["fred:BOGZ1FL075035503Q"], "Z.1 commercial real estate price index, log change")
    put("d_chargeoff", q_native(fr("CORALACBS")).diff(), "x", "pp/季", "same_q", ["fred:CORALACBS"],
        "charge-off rate on all bank loans, change")
    oil, igrea = cols["d_oil"], cols["d_igrea"]
    cause = pd.Series(np.where(oil.notna() & igrea.notna(), (np.sign(oil) != np.sign(igrea)).astype(float), np.nan),
                      index=QI)
    put("x_oil_cause", cause, "x", "0 需求 / 1 供给", "same_q", meta["d_oil"]["sources"] + meta["d_igrea"]["sources"],
        "training label by the sign rule: demand (0) when oil and global activity move the same way, supply (1) "
        "otherwise; a QTable writes supply / demand / mixed (= 0.5)")
    put("d_oil_sup", oil * cause, "x_aux", "log%/季", "same_q", meta["x_oil_cause"]["sources"],
        "d_oil x 1{supply}: the supply part of the oil move (the oil loading splits by cause)")
    put("d_cpi", q_sum(pm("f_cpi"), 400.0), "x_aux", "%年化", "same_q", ["panel:f_cpi"],
        "headline CPI inflation, 400 x the quarter's sum of monthly log changes (the factor layer's pc)")

    # ======================== levels at the quarter's end (the d_ fields are their changes)
    lv_ff, lv_y10 = q_end(pm("lvl_ff")), q_end(pm("lvl_y10"))
    put("lvl_ff", lv_ff, "level", "%", "ref_end", ["panel:lvl_ff"], "fed funds, monthly average of the last month")
    put("lvl_y10", lv_y10, "level", "%", "ref_end", ["panel:lvl_y10"], "10-year yield, month-end of the last month")
    put("lvl_y2", q_end(pm("lvl_y2")), "level", "%", "ref_end", ["panel:lvl_y2"], "2-year yield, month-end")
    put("lvl_tb3", q_end(pm("lvl_tb3")), "level", "%", "ref_end", ["panel:lvl_tb3"], "3-month bill, monthly average")
    put("lvl_spread", q_end(pm("s_spread")), "level", "pp", "ref_end", ["panel:s_spread"],
        "Baa - 10y, both legs at the same time of the month (factor round timing fix)")
    put("lvl_gz", q_end(pm("s_gz")), "level", "pp", "ref_end", ["panel:s_gz"], "GZ credit spread")
    put("lvl_tp10", q_end(pm("s_tp10")), "level", "pp", "ref_end", ["panel:s_tp10"], "ACM 10-year term premium")
    put("lvl_usd", 100.0 * np.log(q_end(pm("lvl_usd"))), "level", "100 log", "ref_end", ["panel:lvl_usd"],
        "100 x log broad dollar (month-end from 2006)")
    put("lvl_unrate", q_end(pm("lvl_unrate")), "level", "%", "ref_end", ["panel:lvl_unrate"], "unemployment rate")
    put("lvl_cpi", 100.0 * np.log(q_end(pm("lvl_cpi"))), "level", "100 log", "ref_end", ["panel:lvl_cpi"],
        "100 x log CPI (all items, SA)")
    put("lvl_oil_real", 100.0 * np.log(q_end(pm("lvl_oil")) / q_end(pm("lvl_cpi"))), "level", "100 log", "ref_end",
        ["panel:lvl_oil", "panel:lvl_cpi"], "100 x log(WTI / CPI) at the quarter's last month")
    put("lvl_infl_trend", q_end(pm("s_infl_expect")), "level", "%", "ref_end", ["panel:s_infl_expect"],
        f"pi_trend: EWMA of annualized monthly CPI inflation, {cq.EWMA_HALF_LIFE_MONTHS}-month half-life")

    # ======================== y: asset outputs (quarterly log returns) and the valuation legs
    for a in cq.ASSETS:
        if a == "cash":
            put("y_cash", q_sum(np.log1p(pm("r_cash"))), "y", "log", "target", ["panel:r_cash"],
                "3-month bill total return (its price return is identically 0)", card="y_assets")
        else:
            put(f"y_{a}", q_sum(np.log1p(pm(f"p_{a}"))), "y", "log", "target", [f"panel:p_{a}"],
                f"sum of the quarter's monthly log price returns log1p(p_{a})", card="y_assets")
        put(f"yt_{a}", q_sum(np.log1p(pm(f"r_{a}"))), "y_aux", "log", "target", [f"panel:r_{a}"],
            f"sum of the quarter's monthly log total returns log1p(r_{a})")
        put(f"inc_{a}", np.exp(cols[f"yt_{a}"]) - np.exp(cols[f"y_{a}"]) if a != "cash" else
            np.exp(cols["yt_cash"]) - 1.0, "y_aux", "simple", "target", [f"panel:r_{a}", f"panel:p_{a}"],
            "simple quarterly income: total minus price simple return over the quarter")
    V = valuation_monthly(inp)
    put("y_mult", q_end(V["us_equity_lm_cape"]).diff(), "y", "log", "target",
        ["panel:p_us_equity", "shiller:price", "shiller:cpi", "shiller:cape"],
        "Δlog CAPE multiple: CRSP quarter-end log price - (log CPI + log E10 real), E10 real implied by Shiller's "
        "CAPE (valuation us_equity_lm_cape)", note="replaces the schema's Δlog P/Ê (write-downs: 2007Q3-2009Q2 "
        "P/Ê +1.005 while the price fell 0.468)")
    put("y_fund", (q_end(V["sp_lcpi"]) + q_end(V["sp_le10r"])).diff(), "y", "log", "target",
        ["shiller:price", "shiller:cpi", "shiller:cape"],
        "Δlog CPI + Δlog E10 real (Shiller): the fundamentals leg, y_us_equity = y_mult + y_fund (V-PRICE)")
    put("y_pd", q_end(V["us_equity_lpd"]).diff(), "y", "log", "target", ["panel:p_us_equity", "panel:d_us_equity"],
        "Δlog CRSP P/D (price over trailing 12-month dividends): the income leg's ratio, not a re-rating measure",
        note="about 30% of the 1973-2025 P/D rise is payout decline")
    eps_q = np.exp(q_end(V["sp_le"]))
    put("y_eps", np.log(eps_q.rolling(4).mean()).diff(), "y", "log", "target", ["shiller:earnings"],
        "Δlog Ê, Ê = 4-quarter mean of Shiller's trailing 12-month S&P EPS at quarter ends (display only)",
        note="write-downs included; never an identity leg")
    put("y_dg", q_end(V["us_equity_ld12"]).diff(), "y_aux", "log", "target",
        ["panel:p_us_equity", "panel:d_us_equity"], "Δlog CRSP trailing 12-month dividends (the dividend equation)")

    # ======================== s: states known at the start of the quarter
    s_ff = at_start(pm("lvl_ff"), 0)
    put("s_ff", s_ff, "s", "%", "start", ["panel:lvl_ff"], "fed funds (monthly average) of the month before q")
    put("s_term", at_start(pm("lvl_y10") - pm("lvl_ff"), 0), "s", "pp", "start", ["panel:lvl_y10", "panel:lvl_ff"],
        "10-year yield (month-end) minus fed funds, the month before q")
    put("s_spread", at_start(pm("s_spread"), 0), "s", "pp", "start", ["panel:s_spread"],
        "Baa - 10y (same time of the month), the month before q")
    put("s_real_y10", at_start(pm("s_real_y10"), 1), "s", "%", "start", ["panel:s_real_y10"],
        "10-year yield minus pi_trend (the factor layer's ry10), CPI lag 1",
        note="the schema's alternative from 2003 is the TIPS yield (s_tips10_aux); the factor layer uses y10 - pi_trend")
    core = at_start(pm("s_core_yoy"), 1)
    put("s_core_yoy", core, "s", "%", "start", ["panel:s_core_yoy"], "core CPI y/y, lag 1")
    put("s_unrate", at_start(pm("lvl_unrate"), 1), "s", "%", "start", ["panel:lvl_unrate"], "unemployment rate, lag 1")
    put("s_infexp1", at_start(fr("MICH"), 0), "s", "%", "start", ["fred:MICH"],
        "Michigan 1-year inflation expectation, the month before q (from 1978)")
    put("s_infexp_lr", at_start(px5, 0), "s", "%", "start", ["umich:px5"],
        "Michigan 5-10-year inflation expectation (median), the month before q (continuous from 1990-04)",
        note="before 1990 the anchor rule (SIM_DESIGN §5.5) uses the trailing 40-quarter mean of core CPI y/y")
    rstar = known_quarterly(q_native(inp["hlw:rstar"]), series_lag("hlw:rstar"))
    put("s_rstar", rstar, "s", "%", "start", ["hlw:rstar"], "HLW one-sided r* (current vintage), quarter q-2",
        note="current-vintage estimates: display and oracle reference only; backtests use the train-end rule")
    put("s_trend_g", known_quarterly(q_native(inp["hlw:g"]), series_lag("hlw:g")), "s", "%", "start", ["hlw:g"],
        "HLW trend growth g (current vintage), quarter q-2")
    infl_trend = at_start(pm("s_infl_expect"), 1)
    pol_infl = cols["s_infexp1"].where(cols["s_infexp1"].notna(), infl_trend.reindex(QI))
    put("s_policy_gap", s_ff.reindex(QI) - pol_infl - rstar.reindex(QI), "s", "pp", "start",
        ["panel:lvl_ff", "fred:MICH", "panel:s_infl_expect", "hlw:rstar"],
        "s_ff - s_infexp1 - s_rstar (pi_trend in place of s_infexp1 before 1978)")
    put("s_pshare", at_start(pm("s_profit_share_rt"), 0), "s", "%", "start", ["panel:s_profit_share_rt"],
        "after-tax book profits / GDP as published by the end of the month before q (monthly panel, release dated)")
    nfc_debt = _ratio_q(q_native(fr("BCNSDODNS")) / 1000.0, gdp)
    put("s_nfc_debt", known_quarterly(nfc_debt, max(series_lag("fred:BCNSDODNS"), series_lag("fred:GDP"))), "s", "%",
        "start", ["fred:BCNSDODNS", "fred:GDP"], "NFC debt securities and loans / GDP (Z.1), quarter q-2")
    put("s_oil_real", at_start(np.log(pm("lvl_oil")), 0) - at_start(np.log(pm("lvl_cpi")), 1), "s", "log", "start",
        ["panel:lvl_oil", "panel:lvl_cpi"], "log WTI (the month before q) / CPI (two months before q)")
    put("s_usd_real", at_start(np.log(chain_forward(fr("TWEXBPA"), fr("RTWEXBGS"), "2006-01")), 1), "s", "log",
        "start", ["fred:TWEXBPA", "fred:RTWEXBGS"],
        "log real broad dollar: TWEXBPA to 2005-12, RTWEXBGS from 2006-01 chained onto it by the 2006-01 ratio, lag 1",
        note="the research rescaled the old index by the 2006 mean ratio (a constant log shift of the level and a "
        "look-ahead in it); here the old level is kept and the new index is chained forward")
    put("s_pd", at_start(V["us_equity_lpd"], 0), "s", "log", "start", ["panel:p_us_equity", "panel:d_us_equity"],
        "log CRSP price / trailing 12-month dividends, the month before q")
    K = valuation_known(V)
    put("s_cape", K["sp_lcape"], "s", "log", "start", ["shiller:price", "shiller:cpi", "shiller:cape"],
        "log CAPE known at the start: S&P price (the month before q) / [CPI (lag 1) x E10 real (lag 3)], Shiller 2026",
        note="valuation sp_lcape__known; E of the last two quarters may be estimates (SIM_DESIGN §6.5)")
    put("s_ecyc", K["sp_lcape"] - K["sp_lpe"], "s", "log", "start",
        ["shiller:price", "shiller:cpi", "shiller:cape", "shiller:earnings"],
        "earnings cycle: log CAPE - log trailing P/E, both known at the start (= log real E / E10 real)")
    put("s_pe", K["sp_lpe_hat"], "s", "log", "start", ["shiller:price", "shiller:earnings"],
        "log P/Ê: S&P price (the month before q) / Ê, Ê = 4-quarter mean of trailing 12-month EPS at the quarter ends "
        "known at the start (E lag 3)", note="display only (write-downs distort it); the S&P price is used because the "
        "CRSP index level is arbitrary")
    put("s_fcfy", fcf_yield_known(inp, QI, V), "s", "%", "start",
        ["fred:W328RC1Q027SBEA", "fred:B456RC1Q027SBEA", "fred:BOGZ1FA105050005Q", "fred:NCBEILQ027S",
         "panel:p_us_equity"],
        "NFC free-cash-flow yield: 4-quarter mean of [profits after tax with IVA/CCAdj + CFC - capex] over NFC equity "
        "market value, latest published quarter, market value rolled to the month before q with the CRSP price",
        note="全部非金融公司的代理，不是 SPY/QQQ 的自由现金流收益率 (display only)")
    put("s_payout", payout_known(inp), "s", "%", "start",
        ["fred:BOGZ1FA106121075Q", "fred:NCBCEBQ027S", "fred:NCBEILQ027S"],
        "NFC net payout yield (net dividends - net equity issuance, 4-quarter mean, over equity market value), "
        "20-quarter mean, quarter q-2 (Z.1)")
    for g, inds in FRENCH_WEIGHT_GROUPS.items():
        tot = sum(inp[f"french:w:{i}"] for i in sorted({c for v in FRENCH_WEIGHT_GROUPS.values() for c in v}))
        put(f"s_w_{g}", at_start(100.0 * sum(inp[f"french:w:{i}"] for i in inds) / tot, 0), "s", "%", "start",
            [f"french:w:{i}" for v in FRENCH_WEIGHT_GROUPS.values() for i in v],
            f"French 12-industry market-value weight of {'+'.join(inds)} (firms x average size), the month before q",
            card="s_w")

    # ======================== s_aux: model states that are not card rows
    put("s_cpi_yoy", at_start(pm("s_cpi_yoy"), 1), "s_aux", "%", "start", ["panel:s_cpi_yoy"], "CPI y/y, lag 1")
    put("s_infl_trend", infl_trend, "s_aux", "%", "start", ["panel:s_infl_expect"],
        "pi_trend (EWMA CPI inflation, 36-month half-life), lag 1")
    un = pm("lvl_unrate")
    u3 = un.rolling(3).mean()
    put("s_sahm", at_start(u3 - u3.shift(1).rolling(12).min(), 1), "s_aux", "pp", "start", ["panel:lvl_unrate"],
        "Sahm gap: 3-month average unemployment minus its minimum over the prior 12 months, lag 1")
    put("s_gz", at_start(pm("s_gz"), 0), "s_aux", "pp", "start", ["panel:s_gz"], "GZ spread, the month before q",
        note="the Board posts a month in the first week of the next (research convention lag 0 kept)")
    put("s_y10", at_start(pm("lvl_y10"), 0), "s_aux", "%", "start", ["panel:lvl_y10"], "10-year yield (month-end)")
    put("s_tb3", at_start(pm("lvl_tb3"), 0), "s_aux", "%", "start", ["panel:lvl_tb3"], "3-month bill")
    put("s_tp10", at_start(pm("s_tp10"), 0), "s_aux", "pp", "start", ["panel:s_tp10"], "ACM term premium")
    put("s_tips10_aux", at_start(pm("s_tips10"), 0), "s_aux", "%", "start", ["panel:s_tips10"],
        "10-year TIPS yield (from 2003)")
    put("s_zlb", (cols["s_ff"] < cq.ZLB_FF).astype(float).where(cols["s_ff"].notna()), "s_aux", "0/1", "start",
        ["panel:lvl_ff"], f"1{{s_ff < {cq.ZLB_FF}}}")
    put("s_core4", (cols["s_core_yoy"] > cq.CORE4).astype(float).where(cols["s_core_yoy"].notna()), "s_aux", "0/1",
        "start", ["panel:s_core_yoy"], f"1{{core CPI y/y > {cq.CORE4:g}}} (the factor round's TP x core regime)")
    yq = {}
    for a in cq.ASSETS:
        s = q_sum(np.log1p(pm(f"p_{a}") if a != "cash" else pm("r_cash")))
        yq[a] = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="Q")) if len(s) else s
    yeq = yq["us_equity"]
    put("s_vol", np.sqrt((qshift(yeq, 1) ** 2 + qshift(yeq, 2) ** 2) / 2.0), "s_aux", "log", "start",
        ["panel:p_us_equity"], "sqrt(mean of the squared us_equity quarterly log returns of q-1 and q-2)")
    for a in cq.ASSETS:
        srcs = ["panel:r_cash"] if a == "cash" else [f"panel:p_{a}"]
        put(f"m1_{a}", qshift(yq[a], 1), "s_aux", "log", "start", srcs, f"y_{a} of q-1")
        put(f"m4_{a}", qshift(yq[a].rolling(4).sum(), 1), "s_aux", "log", "start", srcs,
            f"sum of y_{a} over q-4..q-1")
    for a in cq.EQUITY_LIKE:
        put(f"s_pd_{a}", at_start(V[f"{a}_lpd"], 0), "s_aux", "log", "start", [f"panel:p_{a}", f"panel:d_{a}"],
            f"log own price / trailing 12-month dividends of {a}, the month before q (display; nasdaq uses BusEq's "
            "dividend yield on the NASDAQ price)")
    pb = book_to_market_monthly(inp)
    for a in ("buseq", "energy", "industrials", "staples", "healthcare", "utilities", "financials"):
        if f"{a}_lpb" in pb:
            put(f"s_pb_{a}", at_start(pb[f"{a}_lpb"], 0), "s_aux", "log", "start",
                [f"french:bm:{cq.FRENCH_INDUSTRY[a]}", f"french:p:{cq.FRENCH_INDUSTRY[a]}"],
                f"log P/B of the French {cq.FRENCH_INDUSTRY[a]} industry (annual BE/ME from July, drifted with its "
                "ex-dividend price index), the month before q (display)")

    # ======================== views (display only)
    put("s_debt", known_quarterly(q_native(fr("FYGFGDQ188S")), series_lag("fred:FYGFGDQ188S")), "view", "%",
        "start", ["fred:FYGFGDQ188S"], "federal debt held by the public / GDP, quarter q-2")
    put("s_foreign_hold", known_quarterly(_ratio_q(q_native(fr("FDHBFIN")) * 1000.0, q_native(fr("FYGFDPUN"))),
                                          series_lag("fred:FDHBFIN")), "view", "%", "start",
        ["fred:FDHBFIN", "fred:FYGFDPUN"], "foreign holdings of federal debt / debt held by the public, quarter q-2")
    put("s_gold_real", at_start(np.log(pm("lvl_gold")), 0) - at_start(np.log(pm("lvl_cpi")), 1), "view", "log",
        "start", ["panel:lvl_gold", "panel:lvl_cpi"], "log gold (the month before q) / CPI (two months before q)")
    hh = _ratio_q(q_native(fr("CMDEBT")) / 1000.0, q_avg(fr("DSPI")))
    put("s_hh_debt", known_quarterly(hh, max(series_lag("fred:CMDEBT"), series_lag("fred:DSPI"))), "view", "%",
        "start", ["fred:CMDEBT", "fred:DSPI"], "household debt / disposable income, quarter q-2")
    hpi_fhfa = q_native(fr("USSTHPI"))
    house = np.log(hpi_fhfa / q_avg(fr("CUSR0000SEHA")).reindex(hpi_fhfa.index))
    put("s_house_pr", known_quarterly(house, max(series_lag("fred:USSTHPI"), series_lag("fred:CUSR0000SEHA"))),
        "view", "log", "start", ["fred:USSTHPI", "fred:CUSR0000SEHA"],
        "log FHFA house price index / CPI rent of primary residence (quarterly average), quarter q-2")
    debt_gdp = _ratio_q(q_native(fr("FYGFDPUN")) / 1000.0, gdp)
    y10_q = q_avg(pm("lvl_y10"))
    burden = y10_q.rolling(20).mean().reindex(debt_gdp.index) * debt_gdp / 100.0
    put("s_int_burden", known_quarterly(burden, max(series_lag("fred:FYGFDPUN"), series_lag("fred:GDP"))), "view",
        "% GDP", "start", ["fred:FYGFDPUN", "fred:GDP", "panel:lvl_y10"],
        "identity: (mean 10-year yield over the past 20 quarters) x federal debt held by the public / GDP, q-2")
    put("s_it_share", known_quarterly(_ratio_q(q_native(fr("A679RC1Q027SBEA")), gdp),
                                      max(series_lag("fred:A679RC1Q027SBEA"), series_lag("fred:GDP"))), "view", "%",
        "start", ["fred:A679RC1Q027SBEA", "fred:GDP"], "nominal IT equipment and software investment / GDP, q-2")
    put("s_lshare", known_quarterly(q_native(fr("PRS85006173")), series_lag("fred:PRS85006173")), "view", "指数",
        "start", ["fred:PRS85006173"], "nonfarm business labor share (index), quarter q-2")
    put("s_tdsp", known_quarterly(q_native(fr("TDSP")), series_lag("fred:TDSP")), "view", "%", "start",
        ["fred:TDSP"], "household debt service / disposable income (current vintage, from 2005), quarter q-2")
    y5 = q_end(fr("DGS5"))
    share = at_start(fr("INVEST") / fr("TLAACBM027SBOG"), series_lag("fred:INVEST"))
    bank = share * sum(qshift(y5.diff(), j) for j in range(1, 9))
    put("s_bank_dur", bank, "view", "pp", "start", ["fred:DGS5", "fred:INVEST", "fred:TLAACBM027SBOG"],
        "(bank securities / bank assets, H.8, latest month published) x the sum of the 5-year yield's quarterly "
        "changes over q-8..q-1", note="the research used the share at the end of q-1, published a week into q")
    put("s_old65", pd.Series(np.nan, index=QI), "view", "%", "start", [],
        "share of the population aged 65 and over (World Bank via FRED SPPOP65UPTOZSUSA)",
        note="not available: the series was never downloaded (no download in this round)")
    put("wc_gap", q_avg(pm("s_wc_wage_gap")), "view", "pp", "same_q", ["panel:s_wc_wage_gap"],
        "white-collar minus goods-producing hourly earnings growth (y/y, quarterly average; the monthly panel's "
        "s_wc_wage_gap)")

    # ======================== known: release-lagged lags of every x field
    xcols = [c for c, m in meta.items() if m["role"] in ("x", "x_aux")]
    for c in xcols:
        lag = meta[c]["lag_months"]
        for k in (1, 2):
            known = 3 * (k - 1) >= lag
            v = cols[c].shift(k) if known else pd.Series(np.nan, index=QI)
            put(f"{c}_l{k}__known", v, "known", meta[c]["unit"], "known", meta[c]["sources"],
                f"{c} of quarter q-{k} if published by the start of q (release lag {lag} months after the quarter)",
                note="" if known else "published after the quarter starts: impute it (the model's joint distribution)")

    Q = pd.DataFrame(cols, index=QI)
    for c, m in meta.items():
        if m["role"] in REALIZED_ROLES:
            Q.loc[Q.index > last_q, c] = np.nan
    Q.index.name = "quarter"
    for c, m in meta.items():
        s = Q[c].dropna()
        m.update({"first": str(s.index.min()) if len(s) else None, "last": str(s.index.max()) if len(s) else None,
                  "n": int(len(s))})
    info = {"version": VERSION, "first_quarter": cq.FIRST_QUARTER, "last_data_quarter": str(last_q),
            "today": str(today), "data_end_month": _month(data_end), "n_quarters": int(len(Q)),
            "groups": GROUPS, "inputs": inputs.files, "inputs_digest": inputs_digest(inputs),
            "values_digest": values_digest(inputs),
            "card_digest": _digest(cq.CARD_PATH) if cq.CARD_PATH.exists() else None,
            "notes": [
                "季度面板：每行一个日历季度；最后一行（今天）只有季初已知的状态。",
                "x 字段是季内实现值（训练时的“完美翻译”）；L0 字段等于该季三个月度因子变动之和乘以单位系数。",
                "s 状态与 __known 列只用季度开始前已经发布的数据（每个原始序列的发布滞后见 config_q.SERIES_LAG）；"
                "数据是今天的修订版，不是当时的实时版本。",
                "y_mult 是 CAPE 倍数的对数变化（CRSP 价格 /（CPI × 标普 10 年平均实际盈利））；"
                "y_us_equity = y_mult + y_fund 是恒等式。Shiller 最近两个季度的 E 可能是估计值。",
                "2026Q3 的 y_mult / y_fund 用的是 Shiller 工作簿在最后一个 E 月之后给出的 CAPE。",
                "s_old65 没有数据（本轮不下载）。",
            ]}
    return Q, {"info": info, "columns": meta}


def chain_forward(old: pd.Series, new: pd.Series, at: str) -> pd.Series:
    """`old` before month `at`, then `new` rescaled by old / new in month `at` (causal: no value before `at` depends
    on later data). Without an overlap at `at`, old alone."""
    old, new = old.dropna(), new.dropna()
    t0 = pd.Timestamp(at)
    if t0 not in old.index or t0 not in new.index:
        return old.sort_index()
    ratio = float(old.loc[t0] / new.loc[t0])
    return pd.concat([old[old.index < t0], new[new.index >= t0] * ratio]).sort_index()


# ---------------------------------------------------------------- valuation block (port of valuation/valblock.py)

def _log_index(ret: pd.Series) -> pd.Series:
    """log price index from simple returns: cumulated log1p from the first return, 0 in the month before it."""
    lr = np.log1p(ret.astype(float))
    out = pd.Series(np.nan, index=ret.index)
    first = lr.first_valid_index()
    if first is None:
        return out
    out.loc[first:] = lr.loc[first:].fillna(0.0).cumsum()
    i = out.index.get_loc(first)
    if i > 0:
        out.iloc[i - 1] = 0.0
    return out


def valuation_monthly(inp: Inputs) -> pd.DataFrame:
    """Monthly reference levels (log): CRSP / French / Nareit price index, trailing 12-month dividends and P/D per
    equity asset; Shiller's S&P price, E, D, CPI, CAPE and the implied 10-year average real EPS; the CRSP price over
    S&P fundamentals (the CAPE multiple us_equity_lm_cape, level arbitrary)."""
    p = inp["panel:p_us_equity"]
    idx = pd.date_range(min(p.index.min(), inp["shiller:price"].index.min()), p.index.max(), freq="MS")
    R = {}
    for a in cq.EQUITY_LIKE:
        pa, da = inp[f"panel:p_{a}"].reindex(idx), inp[f"panel:d_{a}"].reindex(idx)
        lp = _log_index(pa)
        d12 = (da * np.exp(lp).shift(1)).rolling(12).sum()
        R[f"{a}_lp"] = lp
        R[f"{a}_ld12"] = np.log(d12.where(d12 > 0))
        R[f"{a}_lpd"] = lp - R[f"{a}_ld12"]
    sh = {c: inp[f"shiller:{c}"].reindex(idx) for c in ("price", "cpi", "earnings", "dividend", "cape")}
    R["sp_lp"] = np.log(sh["price"])
    R["sp_le"] = np.log(sh["earnings"].where(sh["earnings"] > 0))
    R["sp_ld"] = np.log(sh["dividend"].where(sh["dividend"] > 0))
    R["sp_lcpi"] = np.log(sh["cpi"])
    R["sp_lcape"] = np.log(sh["cape"])
    R["sp_le10r"] = R["sp_lp"] - R["sp_lcpi"] - R["sp_lcape"]
    R["us_equity_lm_cape"] = R["us_equity_lp"] - (R["sp_lcpi"] + R["sp_le10r"])
    return pd.DataFrame(R, index=idx)


def valuation_known(V: pd.DataFrame) -> dict[str, pd.Series]:
    """Shiller valuation states known at the start of each quarter (valblock.monthly_known at quarter starts):
    price of the month before q, CPI with lag 1, E and E10 with lag 3."""
    lag_e = series_lag("shiller:earnings")
    lp1 = at_start(V["sp_lp"], 0)
    le = at_start(V["sp_le"], lag_e)
    out = {"sp_lpe": lp1 - le,
           "sp_lcape": lp1 - at_start(V["sp_lcpi"], series_lag("shiller:cpi")) - at_start(V["sp_le10r"], lag_e)}
    # Ê known at the start: mean of trailing E at the last four quarter ends that are known (E lag 3)
    e_end = np.exp(q_end(V["sp_le"]))
    e_hat = e_end.rolling(4).mean()
    out["sp_lpe_hat"] = lp1 - np.log(known_quarterly(e_hat, lag_e))
    return out


def fcf_yield_known(inp: Inputs, QI: pd.PeriodIndex, V: pd.DataFrame) -> pd.Series:
    """NFC free-cash-flow yield known at the start of each quarter (valuation/work/v09_fcf.py): the latest quarter
    published by then (release lag), its 4-quarter mean FCF over its market value rolled forward with the CRSP price
    to the month before q."""
    fq = lambda s: q_native(inp[f"fred:{s}"])        # noqa: E731
    mv = fq("NCBEILQ027S") / 1000.0
    fcf = fq("W328RC1Q027SBEA") + fq("B456RC1Q027SBEA").reindex(mv.index) - fq("BOGZ1FA105050005Q").reindex(mv.index) / 1000.0
    fcf4 = fcf.rolling(4).mean()
    lag = max(series_lag(k) for k in ("fred:NCBEILQ027S", "fred:W328RC1Q027SBEA", "fred:B456RC1Q027SBEA",
                                      "fred:BOGZ1FA105050005Q"))
    k = quarters_back(lag)
    lp = V["us_equity_lp"]
    out = pd.Series(np.nan, index=QI)
    for q in QI:
        j = q - k
        if j not in fcf4.index or j not in mv.index:
            continue
        t_prev = q.start_time.to_period("M").to_timestamp() - pd.DateOffset(months=1)
        qe = j.end_time.to_period("M").to_timestamp()
        if t_prev not in lp.index or qe not in lp.index:
            continue
        roll = np.exp(lp.get(t_prev, np.nan) - lp.get(qe, np.nan))
        out[q] = 100.0 * fcf4[j] / (mv[j] * roll)
    return out


def payout_known(inp: Inputs) -> pd.Series:
    """NFC net payout yield, 20-quarter mean, as known at the start of each quarter (Z.1 lag)."""
    fq = lambda s: q_native(inp[f"fred:{s}"])        # noqa: E731
    mv = fq("NCBEILQ027S")
    net = (fq("BOGZ1FA106121075Q") - fq("NCBCEBQ027S").reindex(fq("BOGZ1FA106121075Q").index)).rolling(4).mean()
    y = (100.0 * net / mv.reindex(net.index)).rolling(20).mean()
    lag = max(series_lag(k) for k in ("fred:NCBEILQ027S", "fred:BOGZ1FA106121075Q", "fred:NCBCEBQ027S"))
    return known_quarterly(y, lag)


def book_to_market_monthly(inp: Inputs) -> pd.DataFrame:
    """log P/B per French industry, monthly (valblock.book_to_market_monthly): row t of the annual BE/ME (book of
    fiscal year t-1 over market of December t-1) is used from July of t to June of t+1, drifted with the industry's
    ex-dividend price index from December t-1."""
    out = {}
    for a, ind in cq.FRENCH_INDUSTRY.items():
        bm = inp[f"french:bm:{ind}"]
        lmv = _log_index(inp[f"french:p:{ind}"])
        v = pd.Series(np.nan, index=lmv.index)
        for t0, b in bm.items():
            if not np.isfinite(b) or b <= 0:
                continue
            year = t0.year
            dec = pd.Timestamp(year=year - 1, month=12, day=1)
            if dec not in lmv.index or not np.isfinite(lmv.get(dec, np.nan)):
                continue
            months = (lmv.index >= pd.Timestamp(year=year, month=7, day=1)) & (
                lmv.index <= pd.Timestamp(year=year + 1, month=6, day=1))
            v[months] = -np.log(b) + lmv[months] - lmv[dec]
        out[f"{a}_lpb"] = v
    return pd.DataFrame(out)


# ---------------------------------------------------------------- policy regressors (P3's policy.py fits them)

def reaction_frame(Q: pd.DataFrame, field: str) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """(y, X, sample) of the systematic policy equation of `field` (config_q.REACTION; schema-revise-2 v01): the
    regressors are same-quarter E0 macro and start-of-quarter states, l_ = the previous quarter's realized value,
    eq4_lag = the previous four quarters' us_equity log price return. sample marks the quarters the equation is
    fitted on (non-ZLB quarters for d_ff, from 2008Q3 for d_bs); the caller cuts it at REACTION_START..train end."""
    spec = cq.REACTION[field]
    ff_end = Q["s_ff"].shift(-1).fillna(Q["s_ff"] + Q["d_ff"])
    zlb = ((Q["s_ff"] < cq.ZLB_FF) & (ff_end < cq.ZLB_FF)).astype(float)
    base = pd.DataFrame({
        "d_ip": Q["d_ip"], "d_unrate": Q["d_unrate"], "d_cpi_goods": Q["d_cpi_goods"],
        "d_cpi_serv": Q["d_cpi_serv"], "s_core_yoy": Q["s_core_yoy"], "s_unrate": Q["s_unrate"], "s_ff": Q["s_ff"],
        "l_d_ff": Q["d_ff"].shift(1), "l_d_gz": Q["d_gz"].shift(1), "l_d_unrate": Q["d_unrate"].shift(1),
        "eq4_lag": Q["y_us_equity"].rolling(4).sum().shift(1), "zlb": zlb, "zlb_x_dun": zlb * Q["d_unrate"],
        "l_d_bs": Q["d_bs"].shift(1), "l_d_perstax": Q["d_perstax"].shift(1)}, index=Q.index)
    if spec["sample"] == "non_zlb":
        sample = zlb == 0
    elif spec["sample"] == "from_2008Q3":
        sample = pd.Series(Q.index >= pd.Period(cq.BS_POLICY_START, "Q"), index=Q.index)
    else:
        sample = pd.Series(True, index=Q.index)
    return Q[field], base[spec["x"]], sample


def ols_r2(y: pd.Series, X: pd.DataFrame) -> tuple[float, int, pd.Series]:
    """OLS with a constant on the complete rows: (R², n, coefficients)."""
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    Z = np.column_stack([np.ones(len(df)), df.drop(columns="_y").to_numpy(float)])
    b, *_ = np.linalg.lstsq(Z, df["_y"].to_numpy(float), rcond=None)
    e = df["_y"].to_numpy(float) - Z @ b
    yy = df["_y"].to_numpy(float)
    return float(1.0 - (e @ e) / ((yy - yy.mean()) @ (yy - yy.mean()))), int(len(df)), pd.Series(
        b, index=["const"] + list(X.columns))


def reaction_r2(Q: pd.DataFrame, field: str, train_end: str) -> tuple[float, int]:
    """In-sample R² and n of the systematic equation fitted on REACTION_START..train_end (v01's check)."""
    y, X, sample = reaction_frame(Q, field)
    window = (Q.index >= pd.Period(cq.REACTION_START, "Q")) & (Q.index <= pd.Period(train_end, "Q"))
    keep = window & sample.to_numpy(bool)
    r2, n, _ = ols_r2(y[keep], X[keep])
    return r2, n


# ---------------------------------------------------------------- helpers for the later phases

def lagged(Q: pd.DataFrame, cols: list[str], k: int = 1) -> pd.DataFrame:
    """Realized values k quarters back (the oracle lag; use the __known columns for what was published)."""
    return Q[cols].shift(k).add_suffix(f"_l{k}")


HIST_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def field_stats(Q: pd.DataFrame, fields: list[str], train_end: str | None = None) -> pd.DataFrame:
    """Per field, on the quarters up to `train_end` (default: all data quarters): n, mean, sd (the "training sd"
    the QTable size bins scale: 小 / 温和 / 明显 / 大幅 / 极端 = 0.25 / 0.5 / 1 / 2 / 3 sd), the schema's display sd on
    1975Q2-2019Q4, the 1985-2019 sd, and the p01..p99 quantiles (the adapter's field_info hist)."""
    end = pd.Period(train_end, "Q") if train_end else Q.index.max()
    rows = {}
    for f in fields:
        s = Q[f].loc[:end].dropna()
        rows[f] = {"n": int(len(s)), "mean": float(s.mean()) if len(s) else np.nan,
                   "sd": float(s.std()) if len(s) > 2 else np.nan,
                   "sd_1975_2019": float(Q[f].loc[cq.SD_WINDOW[0]:cq.SD_WINDOW[1]].std()),
                   "sd_1985_2019": float(Q[f].loc["1985Q1":"2019Q4"].std()),
                   **{f"p{int(round(100 * p)):02d}": float(s.quantile(p)) if len(s) else np.nan
                      for p in HIST_QUANTILES}}
    return pd.DataFrame(rows).T


def columns(meta: dict, role: str | tuple[str, ...]) -> list[str]:
    roles = (role,) if isinstance(role, str) else role
    return [c for c, m in meta["columns"].items() if m["role"] in roles]


def start_columns(meta: dict) -> list[str]:
    """Every column known at the start of its quarter: the states, the start-of-quarter views and the __known lags
    (the information set of an origin; the truncation test covers exactly these)."""
    return [c for c, m in meta["columns"].items() if m["timing"] == "start" or m["role"] == "known"]


def card_columns(meta: dict) -> dict[str, list[str]]:
    """{card row id: [panel columns]} for every panel column that is a card row (or a block of one)."""
    out: dict[str, list[str]] = {}
    for c, m in meta["columns"].items():
        if m.get("card"):
            out.setdefault(m["card"], []).append(c)
    return out


def build_panel() -> tuple[pd.DataFrame, dict]:
    return build(load_inputs())


def save_panel() -> pd.DataFrame:
    """Build the quarterly panel and write it atomically (CSV + meta JSON)."""
    Q, meta = build_panel()
    cq.PANEL_Q_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = cq.PANEL_Q_PATH.with_name(cq.PANEL_Q_PATH.name + ".tmp")
    Q.to_csv(tmp, index_label="quarter")          # full precision: the identities are tested to 1e-12
    tmp.replace(cq.PANEL_Q_PATH)
    jsonio.dump(meta, cq.PANEL_Q_META_PATH)
    return Q


def run(ctx=None, **_) -> dict:
    """Build step (python -m scenariolab.build --only quarterly, once registered there): write the panel and return
    a summary. `ctx` (an EngineContext) is accepted for the build's step signature; the monthly panel is read from
    data/processed either way."""
    Q = save_panel()
    info = load_meta()["info"]
    return {"path": str(cq.PANEL_Q_PATH), "quarters": int(len(Q)), "columns": int(Q.shape[1]),
            "today": info["today"], "last_data_quarter": info["last_data_quarter"]}


def load_panel() -> pd.DataFrame:
    """The quarterly panel with a quarterly PeriodIndex."""
    Q = pd.read_csv(cq.PANEL_Q_PATH, index_col="quarter")
    Q.index = pd.PeriodIndex(Q.index, freq="Q")
    return Q


def load_meta() -> dict:
    return json.loads(cq.PANEL_Q_META_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    out = save_panel()
    m = load_meta()["info"]
    print(f"[qmodel.panel] {out.shape[0]} quarters x {out.shape[1]} columns, {m['first_quarter']}..{m['today']} "
          f"(data to {m['last_data_quarter']}) -> {cq.PANEL_Q_PATH}")
