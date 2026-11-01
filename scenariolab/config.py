"""Shared configuration: paths, eras, assets and macro factors.

Everything that more than one module needs to agree on lives here, so the
data layer, the models and the web UI all speak the same ids.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
WEB_DIR = ROOT / "web"

PANEL_PATH = PROCESSED_DIR / "panel_monthly.csv"
PANEL_META_PATH = PROCESSED_DIR / "panel_meta.json"

# Floating dollar, free gold market and the OPEC era all start around here, so
# every model is estimated on data from this month on. Eras before it are
# shown descriptively only.
MODEL_START = "1973-01"

# Months per simulated scenario path (15 years).
HORIZON_MONTHS = 180

# The simulator draws blocks of this many consecutive historical months.
BLOCK_MONTHS = 6

# Months the simulator treats as rare events: their prior weight (how often an
# unconditioned path draws them, and the base that views tilt from) is scaled
# down. The 2020 lockdown moved unemployment +10pp and output -14% in a month;
# at full weight about one 10-year path in six would replay it.
RARE_MONTHS = [
    {"start": "2020-03", "end": "2020-08", "weight": 0.25, "name": "新冠封锁与重启"},
]

# Where a simulated month's unexplained residual comes from (engine/residuals.py).
# Each sampled block keeps its own months' residuals with probability 0.5 and
# otherwise takes residuals from an independent unconditional draw, so a view
# that selects a few historical episodes brings along only half of their
# era-specific drift (bubbles, busts). Chosen by the engine ablation
# (analysis/ablation.py) on 1995-2014 and confirmed on 2015-2026.
RESIDUALS = {"resid_mode": "partial", "resid_coupling": 0.5}

# Exogenous shocks (wars, pandemics, attacks, embargoes) that no macro factor
# could have anticipated. A one-year forecast whose year contains one is
# flagged, and forecast scores are also reported without those years: a miss
# caused by such an event is not a model error. The last entry is described by
# what the data show (WTI 67 -> 103 in one month), not by its cause.
EXOGENOUS_SHOCKS = [
    {"start": "1973-10", "end": "1974-03", "name": "第四次中东战争与石油禁运", "kind": "war"},
    {"start": "1979-01", "end": "1979-12", "name": "伊朗革命与第二次石油危机", "kind": "war"},
    {"start": "1990-08", "end": "1991-02", "name": "伊拉克入侵科威特与海湾战争", "kind": "war"},
    {"start": "2001-09", "end": "2001-10", "name": "9·11 恐怖袭击", "kind": "attack"},
    {"start": "2020-02", "end": "2020-06", "name": "新冠疫情与全面封锁", "kind": "pandemic"},
    {"start": "2022-02", "end": "2022-06", "name": "俄乌战争", "kind": "war"},
    {"start": "2026-03", "end": "2026-05", "name": "2026 年 3 月石油供给冲击（油价单月 +54%）", "kind": "oil"},
]

ERAS = [
    {"id": "golden_age", "name": "战后黄金期", "start": "1955-01", "end": "1965-12",
     "summary": "高增长、低通胀；工会强，工资跟着生产率涨；布雷顿森林固定汇率。"},
    {"id": "inflation_onset", "name": "通胀起步", "start": "1966-01", "end": "1972-12",
     "summary": "越战与伟大社会财政扩张，通胀抬头；布雷顿森林体系瓦解。"},
    {"id": "great_inflation", "name": "大通胀与石油危机", "start": "1973-01", "end": "1982-12",
     "summary": "两次石油供给冲击，工资-物价螺旋，沃尔克加息；股债正相关。"},
    {"id": "disinflation", "name": "大缓和与去通胀", "start": "1983-01", "end": "1994-12",
     "summary": "通胀和利率长期下行，利润占比回升，全球化起步。"},
    {"id": "tech_boom", "name": "科技繁荣", "start": "1995-01", "end": "1999-12",
     "summary": "生产率和 IT 资本开支繁荣，劳动力紧张、各收入层工资普涨，估值抬升。"},
    {"id": "china_credit", "name": "中国冲击与信贷扩张", "start": "2000-01", "end": "2007-12",
     "summary": "互联网泡沫破裂后，制造业蓝领被全球化压制，劳动收入占比下降，房地产信贷扩张。"},
    {"id": "stagnation_qe", "name": "长期停滞与 QE", "start": "2008-01", "end": "2019-12",
     "summary": "金融危机后低增长、低通胀、零利率和 QE；资产价格上涨、财富差距扩大；股债负相关。"},
    {"id": "covid_reflation", "name": "疫情与再通胀", "start": "2020-01", "end": "2023-12",
     "summary": "巨额财政转移、供应链冲击，低收入工资涨得最快，9% 通胀后急速加息；股债重新正相关。"},
    {"id": "ai_capex", "name": "AI 资本周期", "start": "2024-01", "end": None,
     "summary": "AI 资本开支集中在少数巨头，利润占比高，入门级白领岗位受冲击，关税与高赤字。"},
]

# Broad asset classes. `income` says where the cash yield comes from, which the
# dividend-first withdrawal strategies need.
ASSETS = [
    {"id": "us_equity", "name": "美股大盘", "group": "equity",
     "proxy": "CRSP 全市场市值加权（SPY 代理，Ken French）", "income": "dividend"},
    {"id": "nasdaq", "name": "纳斯达克100", "group": "equity",
     "proxy": "NASDAQ-100 月末价（1986 年前用纳斯达克综合指数）+ 科技行业股息", "income": "dividend"},
    {"id": "energy", "name": "石油能源股", "group": "equity",
     "proxy": "Ken French 12 行业 Enrgy", "income": "dividend"},
    {"id": "industrials", "name": "工业制造股", "group": "equity",
     "proxy": "Ken French 12 行业 Manuf", "income": "dividend"},
    {"id": "staples", "name": "必需消费股", "group": "equity",
     "proxy": "Ken French 12 行业 NoDur", "income": "dividend"},
    {"id": "healthcare", "name": "医疗保健股", "group": "equity",
     "proxy": "Ken French 12 行业 Hlth", "income": "dividend"},
    {"id": "utilities", "name": "公用事业股", "group": "equity",
     "proxy": "Ken French 12 行业 Utils", "income": "dividend"},
    {"id": "financials", "name": "金融股", "group": "equity",
     "proxy": "Ken French 12 行业 Money", "income": "dividend"},
    {"id": "real_estate", "name": "房地产 REITs", "group": "real_estate",
     "proxy": "FTSE Nareit All Equity REITs", "income": "dividend"},
    {"id": "gold", "name": "黄金", "group": "commodity",
     "proxy": "伦敦金价月度均价（datasets/gold-prices）", "income": "none"},
    {"id": "crude_oil", "name": "原油", "group": "commodity",
     "proxy": "WTI 现货（月末；不含期货展期收益）", "income": "none"},
    {"id": "treasury_10y", "name": "10年期美债", "group": "bond",
     "proxy": "按 10 年期收益率月度重定价的平价债券", "income": "coupon"},
    {"id": "cash", "name": "现金", "group": "cash",
     "proxy": "3 个月国库券（TB3MS）", "income": "interest"},
]
ASSET_IDS = [a["id"] for a in ASSETS]

# Monthly factor changes (column prefix `f_` in the panel). `core` factors feed
# the default asset model; the rest are candidates the diagnostics can add.
# `endogenous` factors are built partly from asset prices themselves (the
# market P/D; household net worth, which holds equities and is interpolated
# from quarterly data), so they cannot explain asset returns and a view on
# them amounts to dictating the market.
FACTORS = [
    {"id": "oil", "name": "油价变动", "unit": "Δlog", "core": True, "group": "price"},
    {"id": "ip", "name": "工业产出增长", "unit": "Δlog", "core": True, "group": "cycle"},
    {"id": "cpi_goods", "name": "商品通胀", "unit": "Δlog", "core": True, "group": "price"},
    {"id": "cpi_services", "name": "服务通胀", "unit": "Δlog", "core": True, "group": "price"},
    {"id": "ff", "name": "政策利率变动", "unit": "Δpp", "core": True, "group": "money"},
    {"id": "y10", "name": "10年期收益率变动", "unit": "Δpp", "core": True, "group": "money"},
    {"id": "spread", "name": "信用利差变动", "unit": "Δpp", "core": True, "group": "money"},
    {"id": "usd", "name": "美元变动", "unit": "Δlog", "core": True, "group": "money"},
    {"id": "unrate", "name": "失业率变动", "unit": "Δpp", "core": True, "group": "cycle"},
    {"id": "labor_share", "name": "劳动收入占比变动", "unit": "Δlog", "core": True, "group": "distribution"},
    {"id": "profit_share", "name": "利润占GDP变动", "unit": "Δpp", "core": False, "group": "distribution"},
    {"id": "capex_share", "name": "资本开支占GDP变动", "unit": "Δpp", "core": False, "group": "capital"},
    {"id": "wealth", "name": "家庭净值/收入变动", "unit": "Δlog", "core": False, "group": "distribution",
     "endogenous": True},
    {"id": "white_collar", "name": "白领就业占比变动", "unit": "Δlog", "core": False, "group": "distribution"},
    {"id": "wage", "name": "普通工资增长", "unit": "Δlog", "core": False, "group": "distribution"},
    {"id": "sentiment", "name": "消费者信心变动", "unit": "Δ", "core": False, "group": "cycle"},
    {"id": "valuation", "name": "估值变动（市场 P/D）", "unit": "Δlog", "core": False, "group": "capital",
     "endogenous": True},
    # Valuation and earnings channel (data/panel.py). Quarterly NIPA series are dated at the month they
    # are first published (VALUATION["release_lag_months"]), so none of them looks ahead.
    {"id": "real_y10", "name": "10年期实际收益率变动", "unit": "Δpp", "core": False, "group": "money"},
    {"id": "profit_growth", "name": "企业利润增长（按发布月）", "unit": "Δlog", "core": False, "group": "capital"},
    {"id": "fcf_share", "name": "利润减资本开支/GDP变动", "unit": "Δpp", "core": False, "group": "capital"},
    {"id": "cape", "name": "估值变动（席勒 CAPE）", "unit": "Δlog", "core": False, "group": "capital",
     "endogenous": True, "optional": True},
]
FACTOR_IDS = [f["id"] for f in FACTORS]
CORE_FACTOR_IDS = [f["id"] for f in FACTORS if f["core"]]
ENDOGENOUS_FACTOR_IDS = [f["id"] for f in FACTORS if f.get("endogenous")]

# Level ("state") variables used for era fingerprints and scenario views.
STATES = [
    {"id": "cpi_yoy", "name": "CPI 同比", "unit": "%"},
    {"id": "core_yoy", "name": "核心 CPI 同比", "unit": "%"},
    {"id": "goods_yoy", "name": "商品 CPI 同比", "unit": "%"},
    {"id": "services_yoy", "name": "服务 CPI 同比", "unit": "%"},
    {"id": "ip_yoy", "name": "工业产出同比", "unit": "%"},
    {"id": "unrate", "name": "失业率", "unit": "%"},
    {"id": "ff", "name": "联邦基金利率", "unit": "%"},
    {"id": "real_ff", "name": "实际政策利率", "unit": "%"},
    {"id": "y10", "name": "10年期收益率", "unit": "%"},
    {"id": "term_spread", "name": "期限利差(10Y-FF)", "unit": "pp"},
    {"id": "spread", "name": "信用利差(BAA-10Y)", "unit": "pp"},
    {"id": "oil_yoy", "name": "油价同比", "unit": "%"},
    {"id": "real_oil", "name": "实际油价(2026美元)", "unit": "$"},
    {"id": "usd_yoy", "name": "美元同比", "unit": "%"},
    {"id": "labor_share", "name": "劳动收入占比(指数)", "unit": "2017=100"},
    {"id": "profit_share", "name": "企业税后利润/GDP", "unit": "%"},
    {"id": "capex_share", "name": "非住宅投资/GDP", "unit": "%"},
    {"id": "wealth_income", "name": "家庭净值/可支配收入", "unit": "倍"},
    {"id": "white_collar_share", "name": "白领行业就业占比", "unit": "%"},
    {"id": "real_wage_yoy", "name": "普通实际工资同比", "unit": "%"},
    {"id": "productivity_yoy", "name": "劳动生产率同比", "unit": "%"},
    {"id": "sentiment", "name": "消费者信心", "unit": "指数"},
    {"id": "div_yield", "name": "美股股息率", "unit": "%"},
    {"id": "saving_rate", "name": "个人储蓄率", "unit": "%"},
    {"id": "infl_expect", "name": "通胀预期代理（过去通胀的指数加权均值）", "unit": "%"},
    {"id": "real_y10", "name": "10年期实际收益率代理", "unit": "%"},
    {"id": "profit_share_rt", "name": "企业税后利润/GDP（按发布月）", "unit": "%"},
    {"id": "profit_growth_yoy", "name": "企业税后利润同比（按发布月）", "unit": "%"},
    {"id": "fcf_share", "name": "利润减资本开支/GDP（自由现金流代理）", "unit": "%"},
    {"id": "cape", "name": "席勒市盈率 CAPE", "unit": "倍", "endogenous": True},
]
STATE_IDS = [s["id"] for s in STATES]

# Valuation state (engine/simulate.py ValuationModel, analysis/valuation.py). A price is earnings x multiple,
# so the multiple is never a same-month driver of the price; it is a state: today's market P/D, a fair
# value estimated on the training months from the real 10-year yield, the profit share and expected
# inflation, and a pull of the price toward it. `enabled` is the engine default for scenarios, backtests
# and the one-year forecast; analysis/valuation.py decides it on the backtest windows (validation first,
# one test confirmation) and writes the evidence to results/valuation.json.
VALUATION = {
    "enabled": False,
    # with the mode off, scenario runs still track the market P/D at kappa 0 (every return unchanged), so
    # each scenario shows the multiple its paths imply (macro.pd); off in backtests and the forecast
    "track": True,
    "infl_half_life_months": 36,          # inflation expectation = EWMA of past CPI inflation (panel and paths)
    "release_lag_months": 5,              # NIPA quarter (dated at its first month) is published by start + 5
    # before enabling: check results/valuation.json regression.theory_check and regression.stability -- with
    # these features the fitted real-yield sign decides whether a real-rate shock lowers or raises the fair value
    "features": ("real_y10", "profit_share", "infl_expect"),
    "kappa": 0.1,                         # annual pull toward the fair value (or "ecm": estimated)
    "cma": "historical",                  # drift the pull is added to ("historical" | "valuation_state")
    "income": "constant",                 # equity income: "constant" anchor yield or "path" (1 / P/D path)
}
