"""Build the monthly panel every model reads.

Column families (all monthly, indexed by month start):
  s_<state>    level / state variables (percent units unless noted)
  f_<factor>   monthly factor changes (log diffs or percentage-point diffs)
  r_<asset>    total return, decimal
  p_<asset>    price (capital) return, decimal
  d_<asset>    income return (dividend / coupon / interest), decimal; r = p + d
  lvl_<name>   raw levels kept for re-anchoring simulations (cpi index, yields...)
  x_<name>     audit-only columns (the missing-factor audit rebuilds the research round's base model from them);
               never a view, a model input or a scenario variable

Timing. Monthly macro data sit at the month they describe (reference month); market prices are month-end
values where the source allows (the credit spread uses the last weekly values of the month in 1962-1985);
quarterly series are dated at the month they are first published (config.RELEASE_LAGS) and held until the
next release, never interpolated. panel_meta.json's column_info records each column's sources, timing, lag,
first month and role.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config
from . import sources as src

# The public sources the panel is built from, {name: url} (panel_meta.json and meta.json "sources",
# listed on the 方法与数据 page). Shiller's workbook is optional: without it CAPE is empty. No URL carries a key.
SOURCES = {
    "FRED": "https://fred.stlouisfed.org",
    "Ken French Data Library": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
    "FTSE Nareit": "https://www.reit.com/data-research/reit-indexes/monthly-index-values-returns",
    "Gold Prices（DataHub 核心数据集，月度金价）": "https://github.com/datasets/gold-prices",
    "Robert Shiller 的 ie_data.xls（席勒 CAPE）": src.SHILLER_URL,
    "FRED API（api.stlouisfed.org；有 FRED_API_KEY 时使用，否则用 fredgraph.csv）": "https://fred.stlouisfed.org/docs/api/fred/",
    "美联储 GZ 信用利差与超额债券溢价": src.EBP_URL,
    "纽约联储 ACM 期限溢价": "https://www.newyorkfed.org/research/data_indicators/term-premia-tabs",
    "世界银行大宗商品价格（Pink Sheet）": src.PINK_PAGE_URL,
    "Freddie Mac 房价指数（FMHPI）": "https://www.freddiemac.com/research/indices/house-price-index",
    "Shiller 美国房价（Fig. 3-1）": src.SHILLER_HOME_URL,
    "Caldara–Iacoviello 地缘政治风险指数": "https://www.matteoiacoviello.com/gpr.htm",
    "Baker–Bloom–Davis 经济政策不确定性指数": "https://www.policyuncertainty.com",
    "纽约联储 Holston–Laubach–Williams 中性利率": "https://www.newyorkfed.org/research/policy/rstar",
}
# Source names an older panel build wrote to panel_meta.json, replaced by the entries above.
LEGACY_SOURCE_NAMES = ("Gold (datasets/gold-prices)",)

QUARTER_HOLD = 3          # a quarterly value is held at most until the next quarter's release (3 months)
WC_INDUSTRIES = [("CES5000000008", "CES5000000006"),       # information
                 ("CES5500000008", "CES5500000006"),       # financial activities
                 ("CES6000000008", "CES6000000006")]       # professional and business services
CPI_WC_GOODS = "CES0600000008"                             # goods-producing P&NS hourly earnings


def meta_sources(panel_meta: dict) -> dict:
    """SOURCES, plus any other source a panel_meta.json lists (legacy names dropped).

    build_meta uses it, so meta.json names every source even when panel_meta.json was written by
    an older panel build (which lacked the Shiller entry and named gold differently).
    """
    extra = {k: v for k, v in (panel_meta.get("sources") or {}).items()
             if k not in SOURCES and k not in LEGACY_SOURCE_NAMES and v not in SOURCES.values()}
    return {**SOURCES, **extra}


def par_bond_return(y_prev: pd.Series, y_now: pd.Series, years: int = 10) -> tuple[pd.Series, pd.Series]:
    """Monthly return of a par bond bought at y_prev and repriced at y_now.

    Semiannual coupons; yields in percent. Returns (price_return, income_return).
    """
    c = (y_prev / 100.0).to_numpy()
    y = np.maximum((y_now / 100.0).to_numpy(), 1e-6)
    n = 2 * years
    disc = (1.0 + y / 2.0) ** (-n)
    price = (c / 2.0) * (1.0 - disc) / (y / 2.0) + disc
    price_ret = pd.Series(price - 1.0, index=y_now.index)
    income = pd.Series(c / 12.0, index=y_now.index)
    return price_ret, income


def _yoy(s: pd.Series) -> pd.Series:
    return 100.0 * (s / s.shift(12) - 1.0)


def _q_yoy(q: pd.Series) -> pd.Series:
    """Change over four quarters (percent) of a quarterly series dated at quarter starts."""
    q = q.dropna()
    prior = q.reindex(q.index - pd.DateOffset(years=1)).to_numpy(float)
    return pd.Series(100.0 * (q.to_numpy(float) / prior - 1.0), index=q.index)


def inflation_expectation(cpi: pd.Series, half_life: float | None = None) -> pd.Series:
    """Inflation-expectation proxy (percent a year): EWMA of past annualized monthly CPI inflation.

    Adaptive expectations with a `half_life`-month half-life (config.VALUATION, 36). Only past
    and current CPI prints enter, and the simulator runs the same recursion on simulated CPI
    paths (engine/simulate.py), so the real yield it implies is defined the same way in history
    and in a scenario. Checked against approximate 10-year TIPS real yields at 14 dates in 2003-2024
    (reference values, not panel data), y10 minus this proxy tracks them with a correlation of
    about 0.85, y10 minus 12-month CPI inflation about 0.57.
    """
    hl = float(half_life or config.VALUATION["infl_half_life_months"])
    pi = 1200.0 * np.log(cpi).diff()
    return pi.ewm(alpha=1.0 - 0.5 ** (1.0 / hl), adjust=False, ignore_na=True).mean().where(pi.notna().cumsum() > 0)


def release_dated(q: pd.Series, index: pd.DatetimeIndex, lag_months: int | None = None,
                  hold_months: int | None = None) -> pd.Series:
    """Quarterly values on the monthly `index`, dated at the month they are first published.

    A quarter dated at its first month (the FRED convention) becomes known `lag_months` later
    (config.RELEASE_LAGS; default config.VALUATION's 5, i.e. by the end of the third month after the
    quarter: NIPA profits come with the second GDP estimate, Q4 with the third) and is held until the
    next release, so a month only sees quarters already published by its end. hold_months caps the
    hold (a quarterly series holds for 3 months; a value whose successor is overdue is not carried
    further). Data are today's vintage (revisions are not undone).
    """
    lag = int(config.VALUATION["release_lag_months"] if lag_months is None else lag_months)
    q = q.dropna().copy()
    if q.empty:
        return pd.Series(np.nan, index=index)
    q.index = q.index + pd.DateOffset(months=lag)
    full = pd.date_range(min(q.index.min(), index.min()), max(q.index.max(), index.max()), freq="MS")
    out = q.reindex(full)
    out = out.ffill(limit=hold_months - 1) if hold_months else out.ffill()
    return out.reindex(index)


def last_in_month(s: pd.Series) -> pd.Series:
    """Daily or weekly observations -> the last observed value in each month (by observation date)."""
    s = s.dropna()
    g = s.groupby(s.index.to_period("M")).last()
    g.index = g.index.to_timestamp()
    return g


def last_dates_in_month(s: pd.Series) -> pd.Series:
    """Daily or weekly observations -> the date of the last observation in each month (month-start index)."""
    d = s.dropna().index.to_series()
    g = d.groupby(d.index.to_period("M")).max()
    g.index = g.index.to_timestamp()
    return g


def mean_in_month(s: pd.Series) -> pd.Series:
    """Daily or weekly observations -> the mean of the month's observations (by observation date)."""
    s = s.dropna()
    g = s.groupby(s.index.to_period("M")).mean()
    g.index = g.index.to_timestamp()
    return g


def segments(*parts: tuple[pd.Series, str | None]) -> pd.Series:
    """One series from consecutive pieces [(series, first month), ...]: each piece is used from its first
    month (None = from its start) until the next piece's first month."""
    out = []
    for i, (s, start) in enumerate(parts):
        s = s.dropna()
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if i + 1 < len(parts) and parts[i + 1][1] is not None:
            s = s[s.index < pd.Timestamp(parts[i + 1][1])]
        out.append(s)
    return pd.concat(out).sort_index()


def tornqvist(growth: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Chained Törnqvist index of several components, 1.0 in the first month where every component exists.

    growth: each component's Δlog; weights: each component's (nominal) weight, e.g. payroll = hourly
    earnings x employees. Month t's index change is exp(Σ_i w̄_i,t g_i,t) with w̄ the mean of the
    components' shares at t-1 and t. A missing component after the start ends the index (NaN from there).
    """
    share = weights.div(weights.sum(axis=1, min_count=weights.shape[1]), axis=0)
    w = 0.5 * (share + share.shift(1))
    step = (w * growth).sum(axis=1, min_count=growth.shape[1])
    start = share.dropna().index.min()
    if start is None or pd.isna(start):
        return pd.Series(np.nan, index=growth.index)
    step = step.where(step.index > start, np.nan)
    step.loc[start] = 0.0
    return np.exp(step.loc[start:].cumsum(skipna=False)).reindex(growth.index)


def high_inflation_term(change: pd.Series, state: pd.Series, lag: int, above: float) -> pd.Series:
    """change x 1{state[t-lag] > above}; NaN where the lagged state is unknown."""
    prev = state.shift(lag)
    return change * (prev > above).astype(float).where(prev.notna())


def house_price_index(fmhpi: pd.Series | None, shiller_home: pd.Series | None) -> pd.Series:
    """Freddie Mac HPI (US, SA) from 1975-01; before it, 1972-01..1974-12, Shiller's monthly nominal home price
    index seasonally adjusted by its own 1953-74 month-of-year mean log changes and chained to the FMHPI
    level of 1975-01 (so the 1975-01 change is 0, documented)."""
    fm = fmhpi.dropna() if fmhpi is not None else pd.Series(dtype=float)
    pieces = [fm]
    if shiller_home is not None and len(shiller_home.dropna()):
        dl = np.log(shiller_home.dropna()).diff().loc["1953-02":"1974-12"]
        if len(dl):
            seas = dl.groupby(dl.index.month).mean() - dl.mean()
            dl_sa = dl - dl.index.month.map(seas).to_numpy()
            pre = dl_sa.loc["1972-02":"1974-12"]
            months = pd.date_range("1972-01-01", "1974-12-01", freq="MS")
            if len(pre) == len(months) - 1:
                anchor = float(np.log(fm.iloc[0])) if len(fm) and fm.index[0] == pd.Timestamp("1975-01-01") else 0.0
                # the 1974-12 level is the 1975-01 level; walk back with the adjusted changes
                back = pre.iloc[::-1].cumsum().iloc[::-1]          # sum of the changes after each month
                lvl = pd.Series(anchor, index=months)
                lvl.iloc[:-1] = anchor - back.to_numpy()
                pieces.insert(0, np.exp(lvl))
    out = pd.concat(pieces).sort_index() if pieces else pd.Series(dtype=float)
    return out[~out.index.duplicated(keep="last")]


def shiller_cape(shiller: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.Series:
    """Shiller's own CAPE (S&P composite price / 10-year average real earnings) on the monthly `index`.

    The CAPE column of his workbook as it is (data/raw/shiller/ie_data.xls; `python -m
    scenariolab.data.import_raw` brings in the 2026 workbook of the research round, E and D to 2026-06), so
    s_cape equals his CAPE wherever both exist. Months after his last CAPE stay empty: the NIPA-profit
    extension this replaced read 1.04-1.13 times his values once his newer file came out (SIM_DESIGN §6.5).
    None (no cached file) gives an all-NaN series.
    """
    if shiller is None or "cape" not in shiller or shiller["cape"].dropna().empty:
        return pd.Series(np.nan, index=index)
    return shiller["cape"].reindex(index).astype(float)


def shiller_notes(shiller: pd.DataFrame | None) -> list[str]:
    """The panel notes on Shiller's workbook: what s_cape is, and the caveat on his latest earnings (SIM_DESIGN
    §6.5: the file does not say whether the last two quarters of E are reported or estimated)."""
    if shiller is None or "cape" not in shiller or shiller["cape"].dropna().empty:
        return ["席勒 CAPE 取自 Shiller 的 ie_data.xls（缓存）；缺少该文件时 CAPE 为空。"]
    cape_end = shiller["cape"].last_valid_index()
    out = [f"席勒 CAPE 直接取自 Shiller 的 ie_data.xls（缓存，CAPE 到 {cape_end:%Y-%m}）；他的最后一个 CAPE 月份之后为空，"
           "不再用 NIPA 利润外推。"]
    e = shiller["earnings"].dropna() if "earnings" in shiller else pd.Series(dtype=float)
    if len(e) > 6:
        e_end = e.index[-1]
        before = e.loc[:e_end - pd.DateOffset(months=6)]
        if len(before):
            out.append(f"Shiller 的盈利 E 到 {e_end:%Y-%m}：最近两个季度从 {before.iloc[-1]:.1f}（{before.index[-1]:%Y-%m}）"
                       f"变为 {e.iloc[-1]:.1f}，文件里分不清是已公布值还是估计值：E 的最近两个季度可能是估计值，"
                       "今天的市盈率和 10 年平均盈利都依赖它。")
    return out


class _ColumnBag:
    """Dict of columns that reads like a DataFrame while the panel is built."""

    def __init__(self, index: pd.DatetimeIndex):
        self.index = index
        self._cols: dict[str, pd.Series] = {}

    def __setitem__(self, name: str, s: pd.Series):
        self._cols[name] = pd.Series(s, index=self.index) if not isinstance(s, pd.Series) else s.reindex(self.index)

    def __getitem__(self, name: str) -> pd.Series:
        return self._cols[name]

    def __contains__(self, name: str) -> bool:
        return name in self._cols

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._cols, index=self.index)


# ---------------------------------------------------------------- column_info
# column -> (sources, timing, revised[, RELEASE_LAGS key]). Timing is the column's timing within the model
# sample (1973 on): "mixed" when it changes there (oil and the dollar switch from monthly averages to month-end
# values, the spread from last-week to month-end values) or combines two timings. Sources are FRED series ids
# or publisher files (data/raw/SOURCES.json has their provenance).
REF, AVG, EOM, WEEK, RD, MIX = "reference_month", "monthly_average", "month_end", "last_week", "release_dated", "mixed"
SPREAD_SOURCES = ["DBAA", "WBAA", "BAA", "DGS10", "WGS10YR", "GS10"]
WC_SOURCES = [sid for pair in WC_INDUSTRIES for sid in pair]
FRENCH = ["french_ff_factors", "french_12_industry"]
COLUMN_INFO = {
    "s_cpi_yoy": (["CPIAUCSL"], REF, True), "s_core_yoy": (["CPILFESL"], REF, True),
    "s_goods_yoy": (["CUSR0000SAC"], REF, True), "s_services_yoy": (["CUSR0000SAS"], REF, True),
    "s_ip_yoy": (["INDPRO"], REF, True), "s_unrate": (["UNRATE"], REF, True), "s_ff": (["FEDFUNDS"], AVG, False),
    "s_real_ff": (["FEDFUNDS", "CPIAUCSL"], AVG, True), "s_y10": (["DGS10", "GS10"], EOM, False),
    "s_term_spread": (["DGS10", "GS10", "FEDFUNDS"], MIX, False), "s_spread": (SPREAD_SOURCES, MIX, False),
    "s_oil_yoy": (["DCOILWTICO", "WTISPLC"], MIX, False), "s_real_oil": (["DCOILWTICO", "WTISPLC", "CPIAUCSL"], MIX, True),
    "s_usd_yoy": (["DTWEXBGS", "TWEXMMTH"], MIX, False), "s_labor_share": (["PRS85006173"], RD, True, "PRS85006173"),
    "s_profit_share": (["CP", "GDP"], RD, True, "CP"), "s_capex_share": (["PNFI", "GDP"], RD, True, "PNFI"),
    "s_wealth_income": (["TNWBSHNO", "DPI"], RD, True, "TNWBSHNO"),
    "s_white_collar_share": (["USPBS", "USINFO", "USFIRE", "PAYEMS"], REF, True),
    "s_real_wage_yoy": (["AHETPI", "CPIAUCSL"], REF, True), "s_productivity_yoy": (["OPHNFB"], RD, True, "OPHNFB"),
    "s_sentiment": (["UMCSENT"], REF, False), "s_saving_rate": (["PSAVERT"], REF, True),
    "s_div_yield": (FRENCH, EOM, True), "s_infl_expect": (["CPIAUCSL"], REF, True),
    "s_real_y10": (["DGS10", "GS10", "CPIAUCSL"], MIX, True), "s_profit_share_rt": (["CP", "GDP"], RD, True, "CP"),
    "s_profit_growth_yoy": (["CP"], RD, True, "CP"), "s_fcf_share": (["CP", "PNFI", "GDP"], RD, True, "CP"),
    "s_cape": (["shiller_ie_data"], AVG, True),
    "f_oil": (["DCOILWTICO", "WTISPLC"], MIX, False), "f_ip": (["INDPRO"], REF, True), "f_cpi": (["CPIAUCSL"], REF, True),
    "f_cpi_goods": (["CUSR0000SAC"], REF, True), "f_cpi_services": (["CUSR0000SAS"], REF, True),
    "f_ff": (["FEDFUNDS"], AVG, False), "f_y10": (["DGS10", "GS10"], EOM, False), "f_spread": (SPREAD_SOURCES, MIX, False),
    "f_usd": (["DTWEXBGS", "TWEXMMTH"], MIX, False), "f_unrate": (["UNRATE"], REF, True),
    "f_labor_share": (["PRS85006173"], RD, True, "PRS85006173"), "f_profit_share": (["CP", "GDP"], RD, True, "CP"),
    "f_capex_share": (["PNFI", "GDP"], RD, True, "PNFI"), "f_wealth": (["TNWBSHNO", "DPI"], RD, True, "TNWBSHNO"),
    "f_white_collar": (["USPBS", "USINFO", "USFIRE", "PAYEMS"], REF, True), "f_wage": (["AHETPI"], REF, True),
    "f_sentiment": (["UMCSENT"], REF, False), "f_valuation": (FRENCH, EOM, True),
    "f_real_y10": (["DGS10", "GS10", "CPIAUCSL"], MIX, True), "f_profit_growth": (["CP"], RD, True, "CP"),
    "f_fcf_share": (["CP", "PNFI", "GDP"], RD, True, "CP"), "f_cape": (["shiller_ie_data"], AVG, True),
    # missing-factor round: model inputs, their levels and the audit comparisons
    "s_gz": (["federalreserve_ebp"], REF, True), "f_gz": (["federalreserve_ebp"], REF, True),
    "s_ebp": (["federalreserve_ebp"], REF, True), "f_ebp": (["federalreserve_ebp"], REF, True),
    "s_tp10": (["nyfed_acm"], EOM, True), "f_tp10": (["nyfed_acm"], EOM, True),
    "f_tp10_x_core4": (["nyfed_acm", "CPILFESL"], EOM, True), "f_tp10_x_infl4": (["nyfed_acm", "CPIAUCSL"], EOM, True),
    "f_metals_oil": (["worldbank_pinksheet"], AVG, True), "s_metals_oil_yoy": (["worldbank_pinksheet"], AVG, True),
    # scenario view variables (FACTORS_DESIGN §1.6)
    "s_mort_spread": (["MORTGAGE30US", "WGS10YR"], WEEK, False),
    "s_hpi_yoy": (["freddiemac_fmhpi", "shiller_fig3_1"], REF, True),
    "s_re_loans_yoy": (["REALLN"], REF, True), "s_real_pce_yoy": (["DPCERA3M086SBEA"], REF, True),
    "s_real_dpi_yoy": (["DSPIC96"], REF, True), "s_real_wages_yoy": (["A576RC1", "PCEPI"], REF, True),
    "s_wc_wage_yoy": (WC_SOURCES, REF, True), "s_wc_real_wage_yoy": (WC_SOURCES + ["CPIAUCSL"], REF, True),
    "s_wc_wage_gap": (WC_SOURCES + [CPI_WC_GOODS], REF, True), "s_wc_emp_yoy": (["USINFO", "USFIRE", "USPBS"], REF, True),
    "s_claims": (["ICSA"], AVG, True), "s_ulc_yoy": (["ULCNFB"], RD, True, "ULCNFB"),
    "s_it_invest_share": (["A679RC1Q027SBEA", "GDP"], RD, True, "A679RC1Q027SBEA"),
    "s_philly_capex": (["CEFDFSA066MSFRBPHI"], REF, True), "s_ip_power_yoy": (["IPG2211S"], REF, True),
    "s_ip_oilgas_yoy": (["IPG211S"], REF, True), "s_food_yoy": (["CUSR0000SAF11"], REF, True),
    "s_tariff_rate": (["B235RC1Q027SBEA", "A255RC1Q027SBEA"], RD, True, "B235RC1Q027SBEA"),
    "s_deficit_gdp": (["FGDEF", "GDP"], RD, True, "FGDEF"), "s_debt_gdp": (["FYGFGDQ188S"], RD, True, "FYGFGDQ188S"),
    "s_usd_em_yoy": (["DTWEXEMEGS", "TWEXOMTH"], MIX, False), "s_rstar": (["nyfed_hlw"], RD, True, "HLW"),
    "s_gpr": (["iacoviello_gpr"], REF, True), "s_epu": (["policyuncertainty_epu"], REF, True),
    "s_tips10": (["DFII10"], EOM, False), "s_mich1y": (["MICH"], REF, False),
    # audit only: the report-only stage-B regressors (walkforward.DEDICATED_CANDIDATES), never views
    "f_usd_em": (["DTWEXEMEGS", "TWEXOMTH"], MIX, False), "f_mort_spread": (["MORTGAGE30US", "WGS10YR"], WEEK, False),
    # audit only: the research round's base model (old spread timing, interpolated labor share)
    "x_spread_mixed": (["DBAA", "BAA", "DGS10", "GS10"], MIX, False),
    "x_labor_share_interp": (["PRS85006173"], MIX, True),
}
# third-party copyright by source (SOURCES.json has the full notes); FRED's own data are public
COPYRIGHT = {"BAA": "Moody's", "DBAA": "Moody's", "WBAA": "Moody's", "MORTGAGE30US": "Freddie Mac",
             "freddiemac_fmhpi": "Freddie Mac", "UMCSENT": "University of Michigan", "MICH": "University of Michigan",
             "shiller_fig3_1": "Robert J. Shiller", "shiller_ie_data": "Robert J. Shiller"}
# f_ columns that are not config factors: the walk-forward's report-only stage-B comparisons (audit role)
STAGE_B_AUDIT = {"usd_em", "mort_spread"}
LATE_AFTER_MONTHS = 12          # first observed more than this after the MonthDB's first month: a late variable


def column_info(P: pd.DataFrame) -> dict:
    """{col: {sources, alignment, lag_months, first, last, late, full_history, revised, copyright, role}} for
    every s_, f_ and x_ column (panel_meta.json; the audit's coverage table and the method tab read it)."""
    view, audit = set(config.VIEW_STATE_IDS), config.AUDIT_FACTOR_IDS
    factors = set(config.FACTOR_IDS)
    db_start = pd.Timestamp(config.MODEL_START) + pd.DateOffset(months=1)     # the MonthDB's first drawn month
    out = {}
    for col in P.columns:
        if not col.startswith(("s_", "f_", "x_")):
            continue
        if col not in COLUMN_INFO:
            raise KeyError(f"column_info: no entry for {col} (add it to panel.COLUMN_INFO)")
        sources, timing, revised, *lag_key = COLUMN_INFO[col]
        s = P[col]
        ok = s.notna()
        first, last = (s.index[ok].min(), s.index[ok].max()) if ok.any() else (None, None)
        if col.startswith("x_") or (col.startswith("f_") and col[2:] in audit | STAGE_B_AUDIT):
            role = "audit"
        elif col.startswith("f_") and col[2:] in factors:
            role = "model"
        elif col.startswith("s_") and col[2:] in view:
            role = "view"
        else:
            role = "state"
        rights = sorted({COPYRIGHT[x] for x in sources if x in COPYRIGHT})
        out[col] = {
            "sources": list(sources), "alignment": timing,
            "lag_months": int(config.RELEASE_LAGS[lag_key[0]]) if lag_key else None,
            "first": first.strftime("%Y-%m") if first is not None else None,
            "last": last.strftime("%Y-%m") if last is not None else None,
            "late": bool(first is None or first > db_start + pd.DateOffset(months=LATE_AFTER_MONTHS)),
            "full_history": bool(ok.loc[db_start:].all()) if len(s.loc[db_start:]) else False,
            "revised": bool(revised), "copyright": "; ".join(rights) or None, "role": role,
        }
    return out


def _notes(meta_extra: list[str], cape_lines: list[str] | None = None) -> list[str]:
    lags = config.RELEASE_LAGS
    cape_lines = shiller_notes(None) if cape_lines is None else cape_lines
    return [
        f"季度序列（劳动收入占比、生产率、单位劳动成本：发布滞后 {lags['PRS85006173']} 个月；"
        f"利润、GDP、投资、可支配收入、关税、赤字、债务：{lags['CP']} 个月或 {lags['B235RC1Q027SBEA']} 个月；"
        f"家庭净值：{lags['TNWBSHNO']} 个月）按通常的发布滞后对齐，之前沿用上一次发布值，不插值；个别推迟发布的季度"
        "（如 2013、2018–19 和 2025 年美国政府停摆期间）会比实际发布早 1–2 个月出现，用的也是今天的修订版数据。",
        "信用利差 1962–1985 年用周度 Baa 与 10 年期收益率在每月最后一周的值（同一周），1986 年起用日度月末值；更早用月均值。",
        "10年期收益率、WTI、美元、纳斯达克用月末值；更早年份用月均值拼接。",
        "原油是现货价格回报，不含期货展期收益；黄金为月度均价。",
        "纳斯达克的股息用 Ken French 科技行业（BusEq）的股息收益近似。",
        "消费者信心 1978 年以前是季度调查，两次调查之间沿用上一次调查的值，不插值。",
        f"通胀预期代理 = 过去月度 CPI 通胀（年化）的指数加权均值，半衰期 {config.VALUATION['infl_half_life_months']} 个月；"
        "10 年期实际收益率代理 = 10 年期收益率 − 通胀预期代理。",
        *cape_lines,
        "GZ 信用利差与超额债券溢价为美联储当前版本，整段历史每月都可能修订。",
        "期限溢价为纽约联储 ACM 模型的当前估计（全样本估计的参数）。",
        "房价同比 1973–74 年用 Shiller 的月度房价指数（按 1953–74 年的月份均值季节调整）衔接 Freddie Mac 房价指数。",
        *meta_extra,
    ]


def build_panel(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    f = lambda sid, eop=False: src.fred(sid, eop=eop, refresh=refresh)  # noqa: E731
    q = src.interp_quarterly_to_monthly
    lags = config.RELEASE_LAGS
    state_name = {s["id"]: s["name"] for s in config.STATES}
    unavailable: list[str] = []

    def optional_fred(sid: str, eop: bool = False) -> pd.Series | None:
        try:
            return f(sid, eop=eop)
        except RuntimeError:
            unavailable.append(sid)
            return None

    # ---------------- macro levels
    cpi = f("CPIAUCSL")
    core = f("CPILFESL")
    goods = f("CUSR0000SAC")
    services = f("CUSR0000SAS")
    ip = f("INDPRO")
    unrate = f("UNRATE")
    ff = f("FEDFUNDS")
    gs10, dgs10 = f("GS10"), f("DGS10", eop=True)
    y10 = src.splice_level(dgs10, gs10)
    y2 = f("DGS2", eop=True)
    tb3 = f("TB3MS")
    baa_m, baa_d = f("BAA"), f("DBAA", eop=True)
    # credit spread, both legs at the same time of month: monthly averages to 1961, the last weekly values of
    # the month (Moody's Baa and the 10-year yield of the same H.15 week) 1962-1985, daily month-end from 1986
    wy10_weekly = f("WGS10YR")
    wbaa, wy10 = last_in_month(f("WBAA")), last_in_month(wy10_weekly)
    baa = segments((baa_m, None), (wbaa, "1962-01"), (baa_d, "1986-01"))
    y10s = segments((gs10, None), (wy10, "1962-01"), (dgs10, "1986-01"))
    baa_mixed = src.splice_level(baa_d, baa_m)            # the old construction, kept for the audit
    oil = src.splice(f("DCOILWTICO", eop=True), f("WTISPLC"))
    usd = src.splice(f("DTWEXBGS", eop=True), f("TWEXMMTH"))
    labor_q = f("PRS85006173")
    cp_q, gdp_q, pnfi_q = f("CP"), f("GDP"), f("PNFI")      # quarterly, dated at their release below
    networth_q, dpi_q, productivity_q = f("TNWBSHNO"), f("DPI"), f("OPHNFB")
    wage = f("AHETPI")
    payems, pbs, info, fire = f("PAYEMS"), f("USPBS"), f("USINFO"), f("USFIRE")
    sentiment = f("UMCSENT")
    saving = f("PSAVERT")
    nasdaq = src.splice(f("NASDAQ100", eop=True), f("NASDAQCOM", eop=True))

    # ---------------- Ken French industries (VW, percent -> decimal)
    ind_tot = src.french_table("12_Industry_Portfolios", "Average Value Weighted Returns -- Monthly", refresh) / 100.0
    ind_ex = src.french_table("12_Industry_Portfolios_Wout_Div", "Average Value Weighted Returns -- Monthly", refresh) / 100.0
    nfirms = src.french_table("12_Industry_Portfolios", "Number of Firms in Portfolios", refresh)
    avgsize = src.french_table("12_Industry_Portfolios", "Average Firm Size", refresh)
    ind_div = (ind_tot - ind_ex)
    weights = (nfirms * avgsize)
    weights = weights.div(weights.sum(axis=1), axis=0)
    mkt_div = (ind_div * weights).sum(axis=1, min_count=1)
    ff3 = src.french_factors(refresh)
    mkt_tot = ff3["Mkt-RF"] + ff3["RF"]

    reits = src.nareit_equity_reits(refresh)
    gold = src.gold(refresh)
    shiller = src.shiller(refresh)

    # ---------------- missing-factor round: model inputs (required) and view variables (optional)
    ebp = src.fed_ebp(refresh)
    acm = src.nyfed_acm(refresh)
    pink = src.worldbank_pinksheet(refresh)
    fmhpi, shiller_home = src.freddiemac_fmhpi(refresh), src.shiller_home(refresh)
    gpr, epu, hlw = src.iacoviello_gpr(refresh), src.epu_news(refresh), src.nyfed_hlw(refresh)
    view = {sid: optional_fred(sid) for sid in
            ["MORTGAGE30US", "REALLN", "DPCERA3M086SBEA", "DSPIC96", "A576RC1", "PCEPI", *WC_SOURCES, CPI_WC_GOODS,
             "ICSA", "ULCNFB", "A679RC1Q027SBEA", "CEFDFSA066MSFRBPHI", "IPG2211S", "IPG211S", "CUSR0000SAF11",
             "B235RC1Q027SBEA", "A255RC1Q027SBEA", "FGDEF", "FYGFGDQ188S", "TWEXOMTH", "MICH"]}
    view["DTWEXEMEGS"] = optional_fred("DTWEXEMEGS", eop=True)
    view["DFII10"] = optional_fred("DFII10", eop=True)

    idx = pd.date_range("1947-01-01", max(cpi.index.max(), ind_tot.index.max()), freq="MS")
    # Columns are collected in a dict and turned into one frame at the end;
    # inserting ~150 columns one by one fragments a pandas DataFrame.
    P = _ColumnBag(idx)

    def prep(s: pd.Series | None, ffill_limit: int | None = None, market: bool = False) -> pd.Series:
        if s is None:                                  # an optional source that is not available
            return pd.Series(np.nan, index=idx)
        s = s.reindex(idx)
        if market:
            # month-end / last-week market prices are never interpolated: a missing month inside the
            # series carries the previous month's value (at most one month)
            carried = s.ffill(limit=1)
            last = s.last_valid_index()
            s = carried.where(carried.index <= last) if last is not None else s
        else:
            # Single missing months (e.g. the October 2025 shutdown left no CPI or
            # unemployment print) are filled by interpolation, not dropped.
            s = s.interpolate(limit=2, limit_area="inside")
        if ffill_limit:
            s = s.ffill(limit=ffill_limit)
        return s

    def put(name: str, s: pd.Series | None, ffill_limit: int | None = None, market: bool = False):
        P[name] = prep(s, ffill_limit, market)

    def rd(qs: pd.Series | None, lag: int) -> pd.Series:
        """A quarterly series dated at its release month and held until the next one."""
        if qs is None:
            return pd.Series(np.nan, index=idx)
        return release_dated(qs, idx, lag, hold_months=QUARTER_HOLD)

    # keep raw levels for re-anchoring simulations
    for name, s in {"cpi": cpi, "core": core, "goods": goods, "services": services, "ip": ip,
                    "unrate": unrate, "ff": ff, "y10": y10, "y2": y2, "tb3": tb3,
                    "oil": oil, "usd": usd, "gold": gold, "nasdaq": nasdaq, "wage": wage}.items():
        put(f"lvl_{name}", s)
    put("lvl_baa", baa, market=True)
    put("lvl_y10s", y10s, market=True)                     # the 10-year leg of the spread (lvl_y10 from 1986)
    # Michigan sentiment was quarterly before 1978: a survey value holds until the next survey (at most 3 months)
    P["lvl_sentiment"] = sentiment.reindex(idx).ffill(limit=3)
    # quarterly levels, dated at their first publication (the simulator's labor-share anchor is the latest
    # published quarter)
    P["lvl_labor_share"] = rd(labor_q, lags["PRS85006173"])
    P["lvl_profits"] = rd(cp_q, lags["CP"])
    P["lvl_gdp"] = rd(gdp_q, lags["GDP"])
    P["lvl_pnfi"] = rd(pnfi_q, lags["PNFI"])
    P["lvl_networth"] = rd(networth_q, lags["TNWBSHNO"])
    P["lvl_dpi"] = rd(dpi_q, lags["DPI"])
    P["lvl_productivity"] = rd(productivity_q, lags["OPHNFB"])
    put("lvl_white_collar_share", 100.0 * (pbs + info + fire) / payems)
    put("lvl_saving_rate", saving)

    L = lambda n: P[f"lvl_{n}"]  # noqa: E731

    # ---------------- states
    P["s_cpi_yoy"] = _yoy(L("cpi"))
    P["s_core_yoy"] = _yoy(L("core"))
    P["s_goods_yoy"] = _yoy(L("goods"))
    P["s_services_yoy"] = _yoy(L("services"))
    P["s_ip_yoy"] = _yoy(L("ip"))
    P["s_unrate"] = L("unrate")
    P["s_ff"] = L("ff")
    P["s_real_ff"] = L("ff") - P["s_cpi_yoy"]
    P["s_y10"] = L("y10")
    P["s_term_spread"] = L("y10") - L("ff")
    P["s_spread"] = L("baa") - L("y10s")
    P["s_oil_yoy"] = _yoy(L("oil"))
    last_cpi = L("cpi").dropna().iloc[-1]
    P["s_real_oil"] = L("oil") * last_cpi / L("cpi")
    P["s_usd_yoy"] = _yoy(L("usd"))
    P["s_labor_share"] = L("labor_share")
    # ratios are formed at quarterly frequency, then dated at the release
    P["s_profit_share"] = 100.0 * rd(cp_q / gdp_q.reindex(cp_q.index), lags["CP"])
    P["s_capex_share"] = 100.0 * rd(pnfi_q / gdp_q.reindex(pnfi_q.index), lags["PNFI"])
    P["s_wealth_income"] = rd((networth_q / 1000.0) / dpi_q.reindex(networth_q.index), lags["TNWBSHNO"])
    P["s_white_collar_share"] = L("white_collar_share")
    P["s_real_wage_yoy"] = _yoy(L("wage")) - P["s_cpi_yoy"]
    P["s_productivity_yoy"] = rd(_q_yoy(productivity_q), lags["OPHNFB"])
    P["s_sentiment"] = L("sentiment")
    P["s_saving_rate"] = L("saving_rate")

    # ---------------- factor changes
    ld = lambda s: np.log(s).diff()  # noqa: E731
    P["f_oil"] = ld(L("oil"))
    P["f_ip"] = ld(L("ip"))
    P["f_cpi"] = ld(L("cpi"))
    P["f_cpi_goods"] = ld(L("goods"))
    P["f_cpi_services"] = ld(L("services"))
    P["f_ff"] = L("ff").diff()
    P["f_y10"] = L("y10").diff()
    P["f_spread"] = P["s_spread"].diff()
    P["f_usd"] = ld(L("usd"))
    P["f_unrate"] = L("unrate").diff()
    P["f_labor_share"] = ld(L("labor_share"))             # these quarterly changes are non-zero in release months only
    P["f_profit_share"] = P["s_profit_share"].diff()
    P["f_capex_share"] = P["s_capex_share"].diff()
    P["f_wealth"] = ld(P["s_wealth_income"])
    P["f_white_collar"] = ld(L("white_collar_share"))
    P["f_wage"] = ld(L("wage"))
    P["f_sentiment"] = L("sentiment").diff()

    # ---------------- assets
    def put_asset(aid: str, total: pd.Series, income: pd.Series):
        total = total.reindex(idx)
        income = income.reindex(idx)
        P[f"r_{aid}"] = total
        P[f"d_{aid}"] = income
        P[f"p_{aid}"] = total - income

    put_asset("us_equity", mkt_tot, mkt_div)
    nas_price = nasdaq.pct_change()
    nas_div = ind_div["BusEq"].reindex(nas_price.index)
    put_asset("nasdaq", nas_price + nas_div, nas_div)
    for aid, col in {"energy": "Enrgy", "industrials": "Manuf", "staples": "NoDur",
                     "healthcare": "Hlth", "utilities": "Utils", "financials": "Money"}.items():
        put_asset(aid, ind_tot[col], ind_div[col])
    put_asset("real_estate", reits["total"], reits["income"])
    g = gold.pct_change()
    put_asset("gold", g, g * 0.0)
    o = oil.pct_change()
    put_asset("crude_oil", o, o * 0.0)
    yv = P["lvl_y10"]
    bp, bi = par_bond_return(yv.shift(1), yv)
    put_asset("treasury_10y", bp + bi, bi)
    cash = P["lvl_tb3"].shift(1) / 1200.0
    put_asset("cash", cash, cash)

    # ---------------- market valuation (price / trailing-12m dividends)
    mkt_price_ret = P["p_us_equity"]
    price_idx = (1.0 + mkt_price_ret.fillna(0.0)).cumprod()
    div_cash = P["d_us_equity"] * price_idx.shift(1)
    d12 = div_cash.rolling(12).sum()
    pd_ratio = price_idx / d12
    P["s_div_yield"] = 100.0 / pd_ratio
    P["f_valuation"] = np.log(pd_ratio).diff()

    # ---------------- valuation and earnings channel: every series is known at the end of its month
    # (CPI and yields are monthly; NIPA quarters are dated at their release, release_dated)
    P["s_infl_expect"] = inflation_expectation(L("cpi"))
    P["s_real_y10"] = L("y10") - P["s_infl_expect"]
    cp_rt = rd(cp_q, lags["CP"])
    P["s_profit_share_rt"] = P["s_profit_share"]          # the same release-dated ratio (the valuation module reads _rt)
    P["s_profit_growth_yoy"] = 100.0 * (rd(cp_q / cp_q.shift(4), lags["CP"]) - 1.0)
    P["s_fcf_share"] = 100.0 * rd((cp_q - pnfi_q.reindex(cp_q.index)) / gdp_q.reindex(cp_q.index), lags["CP"])
    P["s_cape"] = shiller_cape(shiller, idx)
    P["f_real_y10"] = P["s_real_y10"].diff()
    P["f_profit_growth"] = np.log(cp_rt).diff()          # non-zero only in release months
    P["f_fcf_share"] = P["s_fcf_share"].diff()
    P["f_cape"] = np.log(P["s_cape"]).diff()

    # ---------------- missing-factor round: credit, term premium and global demand (model inputs)
    put("s_gz", ebp["gz_spread"])
    put("s_ebp", ebp["ebp"])
    P["f_gz"] = P["s_gz"].diff()
    P["f_ebp"] = P["s_ebp"].diff()                        # audit comparison only
    put("s_tp10", acm, market=True)                     # ACM month-end: a missing month is carried, not interpolated
    P["f_tp10"] = P["s_tp10"].diff()
    tp = next(fa["interaction"] for fa in config.FACTORS if fa["id"] == "tp10_x_core4")
    P["f_tp10_x_core4"] = high_inflation_term(P[f"f_{tp['factor']}"], P[f"s_{tp['state']}"], tp["lag"], tp["above"])
    P["f_tp10_x_infl4"] = high_inflation_term(P["f_tp10"], P["s_cpi_yoy"], tp["lag"], tp["above"])   # audit only
    put("lvl_metals_oil", pink["base_metals"] / pink["crude_avg"])
    P["f_metals_oil"] = ld(L("metals_oil"))
    P["s_metals_oil_yoy"] = _yoy(L("metals_oil"))

    # ---------------- scenario view variables (FACTORS_DESIGN §1.6; never model inputs)
    V = view

    def ratio(a: pd.Series | None, b: pd.Series | None, scale: float = 1.0) -> pd.Series | None:
        """scale * a / b on a's dates (a quarterly ratio is formed before it is release dated)."""
        return None if a is None or b is None else scale * a / b.reindex(a.index)

    def yoy(s: pd.Series | None) -> pd.Series:
        return _yoy(prep(s))

    mort = V["MORTGAGE30US"]
    # the last weekly PMMS survey of the month (a Thursday) minus the last weekly 10-year yield (week ending Friday),
    # as the research round built it; in a month that ends on a Thursday the two legs are a week apart
    put("s_mort_spread", None if mort is None else last_in_month(mort) - wy10, market=True)
    mort_gap_days = (None if mort is None else
                     (last_dates_in_month(mort) - last_dates_in_month(wy10_weekly)).dt.days.dropna())
    P["s_hpi_yoy"] = yoy(house_price_index(fmhpi, shiller_home) if fmhpi is not None else None)
    P["s_re_loans_yoy"] = yoy(V["REALLN"])
    P["s_real_pce_yoy"] = yoy(V["DPCERA3M086SBEA"])
    P["s_real_dpi_yoy"] = yoy(V["DSPIC96"])
    P["s_real_wages_yoy"] = yoy(ratio(V["A576RC1"], V["PCEPI"]))
    if all(V[a] is not None and V[e] is not None for a, e in WC_INDUSTRIES):
        # white-collar P&NS hourly earnings: Törnqvist chain over information, financial activities and
        # professional & business services, weighted by payroll (hourly earnings x P&NS employees)
        ahe = pd.DataFrame({a: V[a] for a, _ in WC_INDUSTRIES}).reindex(idx)
        payroll = pd.DataFrame({a: V[a] * V[e] for a, e in WC_INDUSTRIES}).reindex(idx)
        P["s_wc_wage_yoy"] = _yoy(tornqvist(np.log(ahe).diff(), payroll))
    else:
        P["s_wc_wage_yoy"] = pd.Series(np.nan, index=idx)
    P["s_wc_real_wage_yoy"] = P["s_wc_wage_yoy"] - P["s_cpi_yoy"]
    P["s_wc_wage_gap"] = P["s_wc_wage_yoy"] - yoy(V[CPI_WC_GOODS])
    P["s_wc_emp_yoy"] = yoy(info + fire + pbs)
    put("s_claims", None if V["ICSA"] is None else mean_in_month(V["ICSA"]) / 1000.0)
    P["s_ulc_yoy"] = rd(None if V["ULCNFB"] is None else _q_yoy(V["ULCNFB"]), lags["ULCNFB"])
    P["s_it_invest_share"] = rd(ratio(V["A679RC1Q027SBEA"], gdp_q, 100.0), lags["A679RC1Q027SBEA"])
    put("s_philly_capex", V["CEFDFSA066MSFRBPHI"])
    P["s_ip_power_yoy"] = yoy(V["IPG2211S"])
    P["s_ip_oilgas_yoy"] = yoy(V["IPG211S"])
    P["s_food_yoy"] = yoy(V["CUSR0000SAF11"])
    P["s_tariff_rate"] = rd(ratio(V["B235RC1Q027SBEA"], V["A255RC1Q027SBEA"], 100.0), lags["B235RC1Q027SBEA"])
    P["s_deficit_gdp"] = rd(ratio(V["FGDEF"], gdp_q, -100.0), lags["FGDEF"])
    P["s_debt_gdp"] = rd(V["FYGFGDQ188S"], lags["FYGFGDQ188S"])
    # EM dollar: the OITP index (monthly average) to 2005, the EME index (month-end) from 2006, chained by ratio
    usd_em = (src.splice(V["DTWEXEMEGS"], V["TWEXOMTH"])
              if V["DTWEXEMEGS"] is not None and V["TWEXOMTH"] is not None else None)
    P["s_usd_em_yoy"] = yoy(usd_em)
    # the report-only stage-B regressors (walk-forward, audit role): industrials + the EM dollar's log change,
    # gold + the mortgage spread's change; never views or model inputs
    P["f_usd_em"] = np.log(prep(usd_em)).diff()
    P["s_rstar"] = rd(hlw, lags["HLW"])
    put("s_gpr", gpr)
    put("s_epu", epu)
    put("s_tips10", V["DFII10"], market=True)
    put("s_mich1y", V["MICH"])
    P["f_mort_spread"] = P["s_mort_spread"].diff()

    # ---------------- audit only: the research round's base model (FACTORS_DESIGN §4.4)
    P["x_spread_mixed"] = prep(baa_mixed) - L("y10")
    put("x_labor_share_interp", q(labor_q), ffill_limit=6)

    # trailing dividend yields per asset (percent), for display and income modelling
    for a in config.ASSET_IDS:
        P[f"dy_{a}"] = 100.0 * P[f"d_{a}"].rolling(12).sum()

    P = P.frame()

    # trim to the last month where the core asset set is complete
    core_cols = [f"r_{a}" for a in config.ASSET_IDS] + ["f_cpi", "f_ip"]
    last = P[core_cols].dropna().index.max()
    P = P.loc[:last]

    extra = []
    for name, ok, cols in (
            ("Freddie Mac 房价指数（FMHPI）", fmhpi is not None, ["hpi_yoy"]),
            ("Shiller 美国房价（Fig. 3-1）", shiller_home is not None, ["hpi_yoy"]),
            ("Caldara–Iacoviello 地缘政治风险指数", gpr is not None, ["gpr"]),
            ("Baker–Bloom–Davis 经济政策不确定性指数", epu is not None, ["epu"]),
            ("纽约联储 Holston–Laubach–Williams 中性利率", hlw is not None, ["rstar"])):
        if not ok:
            extra.append(f"{name}不可用，{'、'.join(state_name[c] for c in cols)}为空。")
    if unavailable:
        empty = [state_name[c[2:]] for c in P.columns if c.startswith("s_") and c[2:] in config.VIEW_STATE_IDS
                 and P[c].isna().all()]
        extra.append(f"FRED 序列 {'、'.join(unavailable)} 不可用" + (f"，{'、'.join(empty)}为空。" if empty else "。"))
    if mort_gap_days is not None:
        gap = mort_gap_days.loc[pd.Timestamp(config.MODEL_START) + pd.DateOffset(months=1):last]
        if len(gap):
            extra.append(f"房贷利差 = 当月最后一次周度房贷利率调查（周四）− 当月最后一周的 10 年期收益率（截至周五的一周）；"
                         f"月末是周四的月份两条腿相差一周（{gap.index[0]:%Y-%m} 以来 {len(gap)} 个月里有 "
                         f"{int((gap.abs() > 3).sum())} 个）。")
    model_inputs = [fa for fa in config.FACTORS if fa.get("role") != "audit" and not fa.get("optional")]
    for fa in model_inputs:
        col = f"f_{fa['id']}"
        lv = P[col].last_valid_index() if P[col].notna().any() else None
        if lv is not None and lv < last:
            extra.append(f"{fa['name']}的数据只到 {lv:%Y-%m}，之后的月份不能被情景抽到（不向后沿用）。")

    meta = {
        "start": str(P.index.min().date()),
        "end": str(last.date()),
        "model_start": config.MODEL_START,
        "n_months": int(len(P)),
        "columns": list(P.columns),
        "asset_start": {a: str(P[f"r_{a}"].first_valid_index().date()) for a in config.ASSET_IDS},
        "factor_start": {fid: (str(P[f"f_{fid}"].first_valid_index().date())
                               if P[f"f_{fid}"].notna().any() else None) for fid in config.FACTOR_IDS},
        "sources": dict(SOURCES),
        "notes": _notes(extra, shiller_notes(shiller)),
        "cape_source": ("shiller" if P["s_cape"].notna().any() else None),
        "release_lags": dict(config.RELEASE_LAGS),
        "column_info": column_info(P),
    }
    return P, meta


def save_panel(refresh: bool = False) -> pd.DataFrame:
    """Build the panel and write it atomically (other processes read the CSV while it is rebuilt)."""
    P, meta = build_panel(refresh=refresh)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.PANEL_PATH.with_name(config.PANEL_PATH.name + ".tmp")
    P.to_csv(tmp, index_label="date")
    tmp.replace(config.PANEL_PATH)
    tmp_meta = config.PANEL_META_PATH.with_name(config.PANEL_META_PATH.name + ".tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    tmp_meta.replace(config.PANEL_META_PATH)
    return P


def load_panel() -> pd.DataFrame:
    P = pd.read_csv(config.PANEL_PATH, index_col="date", parse_dates=["date"])
    P.index.freq = "MS"
    return P


def load_meta() -> dict:
    return json.loads(config.PANEL_META_PATH.read_text())
