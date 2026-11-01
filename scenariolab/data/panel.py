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
    profits = q(f("CP"))
    gdp = q(f("GDP"))
    pnfi = q(f("PNFI"))
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
        "sources": {
            "FRED": "https://fred.stlouisfed.org",
            "Ken French Data Library": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
            "FTSE Nareit": "https://www.reit.com/data-research/reit-indexes/monthly-index-values-returns",
            "Gold (datasets/gold-prices)": "https://github.com/datasets/gold-prices",
        },
        "notes": [
            "季度序列（劳动收入占比、利润、GDP、投资、净值、生产率）放在季度中间月份后线性插值成月度。",
            "10年期收益率、BAA、WTI、美元、纳斯达克用月末值；更早年份用月均值拼接。",
            "原油是现货价格回报，不含期货展期收益；黄金为月度均价。",
            "纳斯达克的股息用 Ken French 科技行业（BusEq）的股息收益近似。",
        ],
    }
    return P, meta


def save_panel(refresh: bool = False) -> pd.DataFrame:
    P, meta = build_panel(refresh=refresh)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    P.to_csv(config.PANEL_PATH, index_label="date")
    config.PANEL_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return P


def load_panel() -> pd.DataFrame:
    P = pd.read_csv(config.PANEL_PATH, index_col="date", parse_dates=["date"])
    P.index.freq = "MS"
    return P


def load_meta() -> dict:
    return json.loads(config.PANEL_META_PATH.read_text())
