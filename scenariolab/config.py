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
# market P/D; household net worth, which holds equities), so they cannot
# explain asset returns and a view on them amounts to dictating the market.
# `interaction` factors are another factor's change switched on by a lagged
# state (built in data/panel.py), `dedicated` ones are meant for the listed
# assets' equations only, and `role: "audit"` factors are comparison columns of
# the missing-factor audit: never simulated, driftable, a view or listed in the
# variable table.
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
    # Missing-factor round (FACTORS_DESIGN): candidates the walk-forward may add. The GZ spread is the Fed's
    # monthly corporate-bond spread, the term premium the NY Fed's ACM month-end estimate; the interaction is
    # the term-premium change in months whose previous-month core CPI y/y is above 4% (the same 4.0 as
    # factor_model.HIGH_INFLATION and REGIMES["params"]["high_inflation_pct"]; a test checks it).
    {"id": "gz", "name": "GZ 信用利差变动", "unit": "Δpp", "core": False, "group": "money"},
    {"id": "tp10", "name": "10年期限溢价变动（ACM）", "unit": "Δpp", "core": False, "group": "money"},
    {"id": "tp10_x_core4", "name": "期限溢价变动×高核心通胀", "unit": "Δpp", "core": False, "group": "money",
     "interaction": {"factor": "tp10", "state": "core_yoy", "lag": 1, "above": 4.0}},
    {"id": "metals_oil", "name": "基础金属相对油价变动", "unit": "Δlog", "core": False, "group": "price",
     "dedicated": ["energy"]},
    # audit-only comparison columns: never in SIM_FACTORS, views, drifts or the variable table
    {"id": "tp10_x_infl4", "name": "期限溢价变动×高通胀（对照）", "unit": "Δpp", "core": False, "group": "money",
     "role": "audit"},
    {"id": "ebp", "name": "超额债券溢价变动（对照）", "unit": "Δpp", "core": False, "group": "money", "role": "audit"},
]
FACTOR_IDS = [f["id"] for f in FACTORS]
CORE_FACTOR_IDS = [f["id"] for f in FACTORS if f["core"]]
ENDOGENOUS_FACTOR_IDS = [f["id"] for f in FACTORS if f.get("endogenous")]
AUDIT_FACTOR_IDS = {f["id"] for f in FACTORS if f.get("role") == "audit"}

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
    # Scenario view variables of the missing-factor round (FACTORS_DESIGN §1.6): never model inputs (s_gz and
    # s_tp10 are the levels of model factors). `concept` is the concept gap the variable closes (a key of
    # VIEW_CONCEPTS), `source` where it comes from; `late` ones start well after 1973, so a view on them only
    # draws months where they are observed (validation warning W12). Quarterly ones are release dated
    # (RELEASE_LAGS); data/panel.py's column_info has each column's sources, timing and coverage.
    {"id": "gz", "name": "GZ 信用利差", "unit": "pp", "concept": "risk_appetite",
     "source": "美联储 GZ 信用利差与超额债券溢价"},
    {"id": "ebp", "name": "超额债券溢价", "unit": "pp", "concept": "risk_appetite",
     "source": "美联储 GZ 信用利差与超额债券溢价"},
    {"id": "tp10", "name": "10 年期限溢价（ACM）", "unit": "pp", "concept": "term_premium",
     "source": "纽约联储 ACM 期限溢价"},
    {"id": "mort_spread", "name": "房贷利差（30 年房贷 − 10 年期）", "unit": "pp", "concept": "house_prices",
     "source": "FRED：Freddie Mac PMMS（MORTGAGE30US）与 H.15（WGS10YR）"},
    {"id": "hpi_yoy", "name": "房价同比", "unit": "%", "concept": "house_prices",
     "source": "Freddie Mac 房价指数（FMHPI）；1973–74 年衔接 Shiller 美国房价"},
    {"id": "re_loans_yoy", "name": "银行房地产贷款同比", "unit": "%", "concept": "credit_supply",
     "source": "FRED：美联储 H.8（REALLN）"},
    {"id": "real_pce_yoy", "name": "实际消费同比", "unit": "%", "concept": "consumption_income",
     "source": "FRED：BEA（DPCERA3M086SBEA）"},
    {"id": "real_dpi_yoy", "name": "实际可支配收入同比", "unit": "%", "concept": "consumption_income",
     "source": "FRED：BEA（DSPIC96）"},
    {"id": "real_wages_yoy", "name": "实际工资薪金总额同比", "unit": "%", "concept": "consumption_income",
     "source": "FRED：BEA（A576RC1、PCEPI）"},
    {"id": "wc_wage_yoy", "name": "白领普通员工时薪同比", "unit": "%", "concept": "white_collar_income",
     "source": "FRED：BLS CES（信息、金融、专业与商业服务）"},
    {"id": "wc_real_wage_yoy", "name": "白领实际时薪同比", "unit": "%", "concept": "white_collar_income",
     "source": "FRED：BLS CES、CPI"},
    {"id": "wc_wage_gap", "name": "白领与商品生产部门时薪增速差", "unit": "pp", "concept": "white_collar_income",
     "source": "FRED：BLS CES（CES0600000008）"},
    {"id": "wc_emp_yoy", "name": "白领行业就业同比", "unit": "%", "concept": "white_collar_jobs",
     "source": "FRED：BLS CES（USINFO、USFIRE、USPBS）"},
    {"id": "claims", "name": "首次申领失业金（月均）", "unit": "千人", "concept": "tightness",
     "source": "FRED：美国劳工部（ICSA）"},
    {"id": "ulc_yoy", "name": "单位劳动成本同比", "unit": "%", "concept": "bargaining",
     "source": "FRED：BLS 生产率与成本（ULCNFB）"},
    {"id": "it_invest_share", "name": "IT 设备与软件投资/GDP", "unit": "%", "concept": "ai_capex",
     "source": "FRED：BEA（A679RC1Q027SBEA、GDP）"},
    {"id": "philly_capex", "name": "费城联储制造业未来资本开支指数", "unit": "指数", "concept": "ai_capex",
     "source": "FRED：费城联储制造业调查（CEFDFSA066MSFRBPHI）"},
    {"id": "ip_power_yoy", "name": "电力生产同比", "unit": "%", "concept": "power",
     "source": "FRED：美联储 G.17（IPG2211S）"},
    {"id": "ip_oilgas_yoy", "name": "油气开采产出同比", "unit": "%", "concept": "oil_supply",
     "source": "FRED：美联储 G.17（IPG211S）"},
    {"id": "metals_oil_yoy", "name": "基础金属相对油价同比", "unit": "%", "concept": "china",
     "source": "世界银行大宗商品价格（Pink Sheet）"},
    {"id": "food_yoy", "name": "居家食品 CPI 同比", "unit": "%", "concept": "climate",
     "source": "FRED：BLS CPI（CUSR0000SAF11）"},
    {"id": "tariff_rate", "name": "有效关税率", "unit": "%", "concept": "tariffs",
     "source": "FRED：BEA（B235RC1Q027SBEA、A255RC1Q027SBEA）"},
    {"id": "deficit_gdp", "name": "联邦赤字/GDP（NIPA）", "unit": "%", "concept": "fiscal_deficit",
     "source": "FRED：BEA（FGDEF、GDP）"},
    {"id": "debt_gdp", "name": "联邦公众持有债务/GDP", "unit": "%", "concept": "debt",
     "source": "FRED（FYGFGDQ188S）"},
    {"id": "usd_em_yoy", "name": "美元对新兴市场货币同比", "unit": "%", "concept": "china",
     "source": "FRED：美联储 H.10（TWEXOMTH、DTWEXEMEGS）"},
    {"id": "rstar", "name": "中性实际利率 r*（HLW）", "unit": "%", "concept": "r_star",
     "source": "纽约联储 Holston–Laubach–Williams 中性利率"},
    {"id": "gpr", "name": "地缘政治风险（历史指数）", "unit": "指数", "concept": "geopolitics",
     "source": "Caldara–Iacoviello 地缘政治风险指数"},
    {"id": "epu", "name": "经济政策不确定性（新闻指数）", "unit": "指数", "concept": "political_pressure",
     "source": "Baker–Bloom–Davis 经济政策不确定性指数"},
    {"id": "tips10", "name": "10 年期 TIPS 实际收益率", "unit": "%", "concept": "real_yield",
     "source": "FRED：美联储 H.15（DFII10）", "late": True},
    {"id": "mich1y", "name": "密歇根 1 年期通胀预期", "unit": "%", "concept": "infl_expect",
     "source": "FRED：密歇根大学（MICH）", "late": True},
]
STATE_IDS = [s["id"] for s in STATES]
VIEW_STATE_IDS = [s["id"] for s in STATES if s.get("concept")]

# The concept gaps the view variables close (ids of the research round's concept table, harness/concept_gaps;
# names as the audit's 概念缺口 table shows them), in that table's order.
VIEW_CONCEPTS = {
    "risk_appetite": "风险偏好 / 金融条件",
    "term_premium": "期限溢价",
    "china": "中国 / 全球需求",
    "house_prices": "房价",
    "credit_supply": "信贷供给",
    "consumption_income": "家庭消费 / 实际收入",
    "bargaining": "议价能力 / 工资–物价螺旋",
    "tightness": "劳动力市场紧张度",
    "real_yield": "实际收益率（TIPS）",
    "ai_capex": "AI 资本开支",
    "climate": "极端天气 / 食品价格",
    "infl_expect": "通胀预期",
    "tariffs": "关税",
    "fiscal_deficit": "财政赤字",
    "debt": "国债供给 / 债务",
    "r_star": "中性利率 r*",
    "white_collar_income": "白领劳动收入",
    "white_collar_jobs": "白领岗位",
    "power": "电力需求",
    "oil_supply": "原油供给",
    "geopolitics": "地缘政治",
    "political_pressure": "政策不确定性 / 对央行的政治压力",
}

# Months from a quarter's first month (the FRED date of a quarterly observation) to its first publication.
# data/panel.py dates every quarterly series at that month (release_dated) and holds it until the next
# release: no interpolation, no look-ahead (today's revised vintage). Evidence: ALFRED first-release dates
# (BLS Productivity and Costs: lag 4 in 95% of quarters; NIPA profits with the second GDP estimate; Z.1
# about ten weeks after the quarter); HLW is the NY Fed's one-sided r* (FACTORS_DESIGN §1.1, §1.6).
RELEASE_LAGS = {"PRS85006173": 4, "OPHNFB": 4, "ULCNFB": 4, "A679RC1Q027SBEA": 4, "CP": 5, "GDP": 5, "PNFI": 5,
                "DPI": 5, "FGDEF": 5, "FYGFGDQ188S": 5, "HLW": 5, "B235RC1Q027SBEA": 3, "TNWBSHNO": 6}

# FRED access (data/sources.py): the API when FRED_API_KEY is set (loaded from the git-ignored .env), the
# keyless graph CSV otherwise. The key goes only to api_url's host, never into a file, log line or error.
FRED = {"api_url": "https://api.stlouisfed.org/fred", "key_env": "FRED_API_KEY", "min_interval_s": 0.6,
        "attempts": 4}

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
                                          # (= RELEASE_LAGS["CP"]; a test checks it)
    # before enabling: check results/valuation.json regression.theory_check and regression.stability -- with
    # these features the fitted real-yield sign decides whether a real-rate shock lowers or raises the fair value
    "features": ("real_y10", "profit_share", "infl_expect"),
    "kappa": 0.1,                         # annual pull toward the fair value (or "ecm": estimated)
    "cma": "historical",                  # drift the pull is added to ("historical" | "valuation_state")
    "income": "constant",                 # equity income: "constant" anchor yield or "path" (1 / P/D path)
}

# Macro regimes (engine/regimes.py): every month of history and of every simulated path gets one label from
# CPI y/y, the unemployment rate and the industrial production level (the three series the simulator also
# produces). The labels describe; they feed no model component (transmission, forecast and residual mode
# each failed their own walk-forward with regime terms). A parameter changes only here, and every change
# re-runs the NBER validation, the out-of-sample grid and the revision study (build step `regimes`).
# high_inflation_pct equals factor_model.HIGH_INFLATION (a test checks it); the labels read the same month's
# CPI y/y, the transmission's inflation state the previous month's.
REGIMES = {
    "params": {
        "ip_swing_pct": 3.0,              # IP zigzag: % fall from the peak / rise from the trough that confirms it
        "u_rise_pp": 0.5,                 # unemployment confirmation: pp above its low in the down-leg
        "u_smooth_months": 3,             # trailing mean of the unemployment rate
        "severe_ip_decline_pct": 15.0,    # leg decline (peak to lowest level) of a deep contraction
        "high_inflation_pct": 4.0,        # CPI y/y strictly above it: high inflation (month; episode average)
        "cpi_neg_pct": 0.0,               # CPI y/y strictly below it: prices falling
        "core_low_pct": 1.0,              # core CPI y/y strictly below it (history only)
    },
    # display order = the integer code the label arrays carry (-1 = unknown); first match wins in the
    # order stagflation > depression > recession > deflation > inflation > expansion. Ids keep the words
    # the user asked about; `term` is that word where the display name differs from it.
    "labels": [
        {"id": "expansion", "name": "扩张", "term": None, "color": "#a5d8ff",
         "desc": "增长未收缩，当月 CPI 同比在 0–4% 之间"},
        {"id": "inflation", "name": "通胀", "term": None, "color": "#fab005",
         "desc": "增长未收缩，当月 CPI 同比高于 4%（包括过热的年份，也包括衰退之后通胀从高位回落的年份；看通胀趋势区分上行和回落）"},
        {"id": "deflation", "name": "物价负增长", "term": "通缩", "color": "#15aabf",
         "desc": "增长未收缩，当月 CPI 同比低于 0"},
        {"id": "recession", "name": "衰退", "term": None, "color": "#9775fa",
         "desc": "收缩，这段收缩期间 CPI 同比平均不高于 4%，本轮工业产出从高点的跌幅不到 15%"},
        {"id": "depression", "name": "深度衰退", "term": "萧条", "color": "#343a40",
         "desc": "深度收缩：这段收缩期间 CPI 同比平均不高于 4%，本轮工业产出从高点累计下跌 15% 以上"},
        {"id": "stagflation", "name": "滞胀", "term": None, "color": "#b35c00",
         "desc": "收缩，且这段收缩期间 CPI 同比平均高于 4%（不论收缩深浅）"},
    ],
    "first_month": "1948-01",             # labels exist from here (unemployment starts 1948-01)
    "nber_window": ("1948-06", "2020-12"),  # NBER dates only score the rule; they never enter a label
    "min_episodes": 4,                    # a statistic resting on fewer episodes is marked thin (†)
    "bootstrap": {"replicates": 4000, "seed": 20261101, "ci": (5.0, 95.0)},   # whole-episode bootstrap
    "hac_lags": {"monthly": 6, "forward": 12},     # Newey-West lags: monthly regressions / 12-month windows
    "forward_lag_months": 1,              # month t's label is known during t+1: forward windows start at t+2
    "placebo_min_shift": 24,              # circular shifts of the real-time labels by at least this many months
    "reference": {"n_paths": 2000, "seed": 4242},  # the engine's unconditional reference run (runner)
    "longest_contraction_months": 18,     # the longest contraction since 1948 (1973-75, 2008-09)
    "thin_runs": 3,                       # a regime with fewer runs since 1973 gets the thin-history warning
}
