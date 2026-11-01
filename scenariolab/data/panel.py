"""Build the monthly panel every model reads.

Column families (all monthly, indexed by month start):
  s_<state>    level / state variables (percent units unless noted)
  f_<factor>   monthly factor changes (log diffs or percentage-point diffs)
  r_<asset>    total return, decimal
  p_<asset>    price (capital) return, decimal
  d_<asset>    income return (dividend / coupon / interest), decimal; r = p + d
  lvl_<name>   raw levels kept for re-anchoring simulations (cpi index, yields...)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config
from . import sources as src

# The public sources the panel is built from, {name: url} (panel_meta.json and meta.json "sources",
# listed on the 方法与数据 page). Shiller's workbook is optional: without it CAPE is empty.
SOURCES = {
    "FRED": "https://fred.stlouisfed.org",
    "Ken French Data Library": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
    "FTSE Nareit": "https://www.reit.com/data-research/reit-indexes/monthly-index-values-returns",
    "Gold Prices（DataHub 核心数据集，月度金价）": "https://github.com/datasets/gold-prices",
    "Robert Shiller 的 ie_data.xls（席勒 CAPE）": src.SHILLER_URL,
}
# Source names an older panel build wrote to panel_meta.json, replaced by the entries above.
LEGACY_SOURCE_NAMES = ("Gold (datasets/gold-prices)",)


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


def release_dated(q: pd.Series, index: pd.DatetimeIndex, lag_months: int | None = None) -> pd.Series:
    """Quarterly values on the monthly `index`, dated at the month they are first published.

    A quarter dated at its first month (the FRED convention) becomes known `lag_months` later
    (config.VALUATION: 5, i.e. by the end of the third month after the quarter; NIPA profits come
    with the second GDP estimate, Q4 with the third) and is held until the next release, so a
    month only sees quarters already published by its end. Data are today's vintage (revisions
    are not undone).
    """
    lag = int(config.VALUATION["release_lag_months"] if lag_months is None else lag_months)
    q = q.dropna().copy()
    if q.empty:
        return pd.Series(np.nan, index=index)
    q.index = q.index + pd.DateOffset(months=lag)
    full = pd.date_range(min(q.index.min(), index.min()), max(q.index.max(), index.max()), freq="MS")
    return q.reindex(full).ffill().reindex(index)


def spliced_cape(shiller: pd.DataFrame | None, price_index: pd.Series, profits_rt: pd.Series,
                 cpi: pd.Series) -> pd.Series:
    """Shiller's CAPE (S&P composite price / 10-year average real earnings), extended past his last month.

    After Shiller's last earnings month, earnings grow with NIPA after-tax profits as published
    (`profits_rt`) and the price with the CRSP market price index; the extension is chained to
    his last CAPE value. None (no cached file) gives an all-NaN series.
    """
    idx = price_index.index
    if shiller is None or shiller["cape"].dropna().empty:
        return pd.Series(np.nan, index=idx)
    sh = shiller.reindex(pd.date_range(min(shiller.index.min(), idx.min()), idx.max(), freq="MS"))
    cape_end = sh["cape"].last_valid_index()
    e_end = sh["earnings"].last_valid_index()
    cpi_end = sh["cpi"].last_valid_index()
    out = sh["cape"].copy()
    if cape_end >= idx.max():
        return out.reindex(idx)
    # earnings and CPI past Shiller's last prints, chained to them
    prof = profits_rt.reindex(sh.index)
    earn = sh["earnings"].copy()
    later = earn.index > e_end
    earn[later] = earn[e_end] * prof[later] / prof[e_end]
    c = sh["cpi"].copy()
    later_c = c.index > cpi_end
    c[later_c] = c[cpi_end] * cpi.reindex(sh.index)[later_c] / cpi.get(cpi_end, np.nan)
    real_e10 = (earn / c).rolling(120, min_periods=120).mean()
    pidx = price_index.reindex(sh.index)
    price = sh["price"].copy()
    later_p = price.index > cape_end
    price[later_p] = price[cape_end] * pidx[later_p] / pidx[cape_end]
    own = (price / c) / real_e10
    out[later_p] = sh["cape"][cape_end] * own[later_p] / own[cape_end]
    return out.reindex(idx)


class _ColumnBag:
    """Dict of columns that reads like a DataFrame while the panel is built."""

    def __init__(self, index: pd.DatetimeIndex):
        self.index = index
        self._cols: dict[str, pd.Series] = {}

    def __setitem__(self, name: str, s: pd.Series):
        self._cols[name] = pd.Series(s, index=self.index) if not isinstance(s, pd.Series) else s.reindex(self.index)

    def __getitem__(self, name: str) -> pd.Series:
        return self._cols[name]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._cols, index=self.index)


def build_panel(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    f = lambda sid, eop=False: src.fred(sid, eop=eop, refresh=refresh)  # noqa: E731
    q = src.interp_quarterly_to_monthly

    # ---------------- macro levels
    cpi = f("CPIAUCSL")
    core = f("CPILFESL")
    goods = f("CUSR0000SAC")
    services = f("CUSR0000SAS")
    ip = f("INDPRO")
    unrate = f("UNRATE")
    ff = f("FEDFUNDS")
    y10 = src.splice_level(f("DGS10", eop=True), f("GS10"))
    y2 = f("DGS2", eop=True)
    tb3 = f("TB3MS")
    baa = src.splice_level(f("DBAA", eop=True), f("BAA"))
    oil = src.splice(f("DCOILWTICO", eop=True), f("WTISPLC"))
    usd = src.splice(f("DTWEXBGS", eop=True), f("TWEXMMTH"))
    labor = q(f("PRS85006173"))
    cp_q, gdp_q, pnfi_q = f("CP"), f("GDP"), f("PNFI")      # quarterly, for the release-dated series
    profits = q(cp_q)
    gdp = q(gdp_q)
    pnfi = q(pnfi_q)
    networth = q(f("TNWBSHNO"))
    dpi = q(f("DPI"))
    productivity = q(f("OPHNFB"))
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

    idx = pd.date_range("1947-01-01", max(cpi.index.max(), ind_tot.index.max()), freq="MS")
    # Columns are collected in a dict and turned into one frame at the end;
    # inserting ~150 columns one by one fragments a pandas DataFrame.
    P = _ColumnBag(idx)

    def put(name: str, s: pd.Series, ffill_limit: int | None = None):
        s = s.reindex(idx)
        # Single missing months (e.g. the October 2025 shutdown left no CPI or
        # unemployment print) are filled by interpolation, not dropped.
        s = s.interpolate(limit=2, limit_area="inside")
        if ffill_limit:
            s = s.ffill(limit=ffill_limit)
        P[name] = s

    # keep raw levels for re-anchoring simulations
    for name, s in {"cpi": cpi, "core": core, "goods": goods, "services": services, "ip": ip,
                    "unrate": unrate, "ff": ff, "y10": y10, "y2": y2, "tb3": tb3, "baa": baa,
                    "oil": oil, "usd": usd, "gold": gold, "nasdaq": nasdaq, "wage": wage,
                    "sentiment": sentiment.reindex(idx).interpolate(limit_area="inside")}.items():
        put(f"lvl_{name}", s)
    for name, s in {"labor_share": labor, "profits": profits, "gdp": gdp, "pnfi": pnfi,
                    "networth": networth, "dpi": dpi, "productivity": productivity}.items():
        put(f"lvl_{name}", s, ffill_limit=6)
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
    P["s_spread"] = L("baa") - L("y10")
    P["s_oil_yoy"] = _yoy(L("oil"))
    last_cpi = L("cpi").dropna().iloc[-1]
    P["s_real_oil"] = L("oil") * last_cpi / L("cpi")
    P["s_usd_yoy"] = _yoy(L("usd"))
    P["s_labor_share"] = L("labor_share")
    P["s_profit_share"] = 100.0 * L("profits") / L("gdp")
    P["s_capex_share"] = 100.0 * L("pnfi") / L("gdp")
    P["s_wealth_income"] = (L("networth") / 1000.0) / L("dpi")
    P["s_white_collar_share"] = L("white_collar_share")
    P["s_real_wage_yoy"] = _yoy(L("wage")) - P["s_cpi_yoy"]
    P["s_productivity_yoy"] = _yoy(L("productivity"))
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
    P["f_labor_share"] = ld(L("labor_share"))
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
    cp_rt = release_dated(cp_q, idx)
    P["s_profit_share_rt"] = 100.0 * release_dated(cp_q / gdp_q.reindex(cp_q.index), idx)
    P["s_profit_growth_yoy"] = 100.0 * (release_dated(cp_q / cp_q.shift(4), idx) - 1.0)
    P["s_fcf_share"] = 100.0 * release_dated((cp_q - pnfi_q.reindex(cp_q.index)) / gdp_q.reindex(cp_q.index), idx)
    P["s_cape"] = spliced_cape(shiller, price_idx, cp_rt, L("cpi"))
    P["f_real_y10"] = P["s_real_y10"].diff()
    P["f_profit_growth"] = np.log(cp_rt).diff()          # non-zero only in release months
    P["f_fcf_share"] = P["s_fcf_share"].diff()
    P["f_cape"] = np.log(P["s_cape"]).diff()

    # trailing dividend yields per asset (percent), for display and income modelling
    for a in config.ASSET_IDS:
        P[f"dy_{a}"] = 100.0 * P[f"d_{a}"].rolling(12).sum()

    P = P.frame()

    # trim to the last month where the core asset set is complete
    core_cols = [f"r_{a}" for a in config.ASSET_IDS] + ["f_cpi", "f_ip"]
    last = P[core_cols].dropna().index.max()
    P = P.loc[:last]

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
        "notes": [
            "季度序列（劳动收入占比、利润、GDP、投资、净值、生产率）放在季度中间月份后线性插值成月度。",
            "10年期收益率、BAA、WTI、美元、纳斯达克用月末值；更早年份用月均值拼接。",
            "原油是现货价格回报，不含期货展期收益；黄金为月度均价。",
            "纳斯达克的股息用 Ken French 科技行业（BusEq）的股息收益近似。",
            f"估值与盈利渠道的季度序列（利润占比、利润同比、利润减资本开支）按发布月对齐：季度数据从季度第一个月起 "
            f"{config.VALUATION['release_lag_months']} 个月后才算已知，之前沿用上一次发布值，不插值、不前视（用的是今天的修订版数据）。",
            f"通胀预期代理 = 过去月度 CPI 通胀（年化）的指数加权均值，半衰期 {config.VALUATION['infl_half_life_months']} 个月；"
            "10 年期实际收益率代理 = 10 年期收益率 − 通胀预期代理。",
            "席勒 CAPE 取自 Shiller 的 ie_data.xls（缓存）；他的最后一个月之后，盈利按 NIPA 税后利润（按发布月）增长、"
            "价格按 CRSP 全市场价格指数外推，并与他的最后一个 CAPE 值衔接。缺少该文件时 CAPE 为空。",
        ],
        "cape_source": ("shiller+nipa_extension" if shiller is not None and P["s_cape"].notna().any() else None),
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
