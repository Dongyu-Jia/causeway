"""Build WorldSim's default world 「美国与世界 · 2026Q4」 from Causeway's local data.

    python tools/build_worldsim_default.py            # writes worldsim/defaults/us_2026q4.json
    python tools/build_worldsim_default.py --check    # rebuilds in memory and reports differences, writes nothing

Inputs (all local; data/ is git-ignored, so the output file is committed and CI never needs data/):
- data/processed/panel_monthly.csv, panel_quarterly.csv  today's values (2026-09 / start of 2026Q4) and the history
  each field's range note and plausible bounds come from
- data/raw/fred/*.json                                    payrolls, housing starts, mortgage rate, breakeven, wages,
                                                          the Fed's balance sheet (series the panels do not carry)
- data/results/agents_accounts_root.json                  the actors' books (balance sheets, income, cash flow): each
                                                          actor's private information, and fields such as the EFFR
- data/agents/institutions.json                           the pinned facts (Fed, IEEPA, oil, midterms, China truce,
                                                          OPEC+): every one still unconfirmed is tagged 未确认
- scenariolab/qmodel/card.json                            the quarterly model card's definitions
- scenariolab/agents/actors.py                            the actor catalogue: roles and action spaces of the ten
                                                          policy / state / corporate actors (labels are the catalogue's)
- the statesynth design note (§7)                         the four private-sector actors' action spaces

Today = the start of 2026Q4: the panels end 2026-09. A few fields have no September reading and use the latest one
in the accounts (stated in their description, e.g. the EFFR of 2026-10-29). No asset price levels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
OUT = ROOT / "worldsim" / "defaults" / "us_2026q4.json"
TODAY_M = "2026-09-01"


# ------------------------------------------------------------------------------------------------ data access

class Data:
    def __init__(self):
        need = [DATA / "processed" / "panel_monthly.csv", DATA / "processed" / "panel_quarterly.csv",
                DATA / "results" / "agents_accounts_root.json", DATA / "agents" / "institutions.json"]
        missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
        if missing:
            sys.exit("missing local data: " + ", ".join(missing) + " (run Causeway's build first)")
        self.pm = pd.read_csv(need[0], index_col=0, parse_dates=True)
        self.pq = pd.read_csv(need[1], index_col=0)
        self.acct = json.loads(need[2].read_text(encoding="utf-8"))
        self.inst = {it["id"]: it for it in json.loads(need[3].read_text(encoding="utf-8"))["items"]}
        self.card = {r["id"]: r for r in json.loads((ROOT / "scenariolab" / "qmodel" / "card.json").read_text(
            encoding="utf-8"))["rows"]}

    # monthly panel
    def pm_series(self, col: str) -> pd.Series:
        return self.pm[col].astype(float).dropna()

    def pm_today(self, col: str) -> tuple[float, str]:
        s = self.pm_series(col)
        return float(s.iloc[-1]), s.index[-1].strftime("%Y-%m")

    # quarterly panel
    def pq_series(self, col: str) -> pd.Series:
        return self.pq[col].astype(float).dropna()

    def pq_today(self, col: str) -> tuple[float, str]:
        s = self.pq_series(col)
        return float(s.iloc[-1]), str(s.index[-1])

    # raw FRED observations
    @staticmethod
    def fred(series: str) -> pd.Series:
        path = DATA / "raw" / "fred" / series
        if path.suffix == ".csv":
            rows = list(csv.reader(path.open(encoding="utf-8")))[1:]
            pairs = [(r[0], r[1]) for r in rows if len(r) > 1]
        else:
            obs = json.loads(path.read_text(encoding="utf-8"))["observations"]
            pairs = [(o["date"], o["value"]) for o in obs]
        vals = {pd.Timestamp(d): float(v) for d, v in pairs if v not in ("", ".")}
        return pd.Series(vals).sort_index()

    def line(self, book: str, key: str) -> dict:
        return self.acct["books"][book]["lines"][key]

    def val(self, book: str, key: str) -> float:
        return float(self.line(book, key)["value"])


# ------------------------------------------------------------------------------------------------ formatting

def r(x: float, d: int = 2) -> float:
    return float(round(x, d)) + 0.0


def fmt(x: float, d: int = 2) -> str:
    if abs(x) >= 1000:
        return f"{x:,.{0 if d <= 1 else 1}f}"
    return f"{x:.{d}f}"


def hist_note(s: pd.Series, d: int = 2, freq: str = "m") -> str:
    s = s.dropna()
    if s.empty:
        return ""

    def when(ix):
        return ix.strftime("%Y-%m") if hasattr(ix, "strftime") else str(ix)
    lo_i, hi_i = s.idxmin(), s.idxmax()
    first, last = when(s.index[0])[:4], when(s.index[-1])[:4]
    return (f"{first}–{last}：最低 {fmt(float(s.min()), d)}（{when(lo_i)}），最高 {fmt(float(s.max()), d)}（{when(hi_i)}），"
            f"中位 {fmt(float(s.median()), d)}")


def nice_bounds(s: pd.Series, nonneg: bool, pad: float = 0.25) -> tuple[float, float]:
    lo, hi = float(s.min()), float(s.max())
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    lo2, hi2 = lo - pad * span, hi + pad * span
    if nonneg:
        lo2 = max(0.0, lo2)
    mag = 10 ** math.floor(math.log10(max(abs(hi2), abs(lo2), 1e-9))) / 10
    return float(f"{math.floor(lo2 / mag) * mag:.6g}"), float(f"{math.ceil(hi2 / mag) * mag:.6g}")


def yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    return (s / s.shift(periods) - 1) * 100


# ------------------------------------------------------------------------------------------------ the schema

def field(fid, label, group, ftype, value, unit="", description="", hist="", lo=None, hi=None, options=None, tag=""):
    f = {"id": fid, "label": label, "group": group, "type": ftype, "unit": unit, "description": description,
         "options": options or [], "min": lo, "max": hi, "hist": hist, "tag": tag}
    return f, value


def num_field(D, fid, label, group, ftype, unit, value, as_of, description, series, nonneg=True, d=2, freq="m",
              lo=None, hi=None, tag=""):
    if series is not None and not series.dropna().empty:
        blo, bhi = nice_bounds(series, nonneg)
        hist = hist_note(series, d, freq)
    else:
        blo, bhi, hist = lo, hi, "本地数据没有这一序列的历史：合理区间为判断值"
    lo = blo if lo is None else lo
    hi = bhi if hi is None else hi
    desc = f"{description}（读数：{as_of}）" if as_of else description
    return field(fid, label, group, ftype, r(value, d), unit, desc, hist, lo, hi, tag=tag)


def unconfirmed(D, iid) -> str:
    return "未确认" if D.inst.get(iid, {}).get("status") == "unconfirmed" else ""


def build_schema(D: Data):
    G_RATE, G_PRICE, G_GROW, G_LAB = "政策利率与收益率曲线", "物价与通胀预期", "增长与投资", "劳动力市场"
    G_CRED, G_FISC, G_HOUSE, G_CORP = "信用与金融条件", "财政", "住房", "企业与收入分配"
    G_HH, G_EXT, G_CONF, G_POL = "居民", "外部、能源与贸易", "信心与不确定性", "政治与制度（定性）"
    rows = []
    add = rows.append

    # --- policy rate and the curve
    effr = D.line("fed", "effr")
    ff_m = D.pm_series("lvl_ff")
    add(num_field(D, "policy_rate", "联邦基金利率（有效）", G_RATE, "percent", "%", effr["value"], effr["as_of"],
                  "有效联邦基金利率。数据显示 2026 年 9 月中旬上调 25bp：7–8 月月均 3.63%、9 月月均 3.75%，"
                  f"10 月下旬 {effr['value']:.2f}%（目标区间 3.75–4.00%，准备金利率 {D.val('fed', 'iorb'):.2f}%）。"
                  "10 月 27–28 日会议按数据推断为维持（未确认）。历史区间按联邦基金利率月均计算", ff_m))
    walcl = D.fred("WALCL_m_eop.json")
    ta = D.line("fed", "total_assets")
    add(num_field(D, "fed_assets", "美联储总资产", G_RATE, "number", "十亿美元", ta["value"], ta["as_of"],
                  "H.4.1 美联储总资产。近 13 周国库券增持约 "
                  f"{D.val('fed', 'soma_bills_change_13w'):.0f} 十亿美元（准备金管理购买），MBS 到期滚落约 "
                  f"{-D.val('fed', 'soma_mbs_change_13w'):.0f} 十亿美元", walcl / 1000, d=1))
    y2, a2 = D.pm_today("lvl_y2")
    add(num_field(D, "y2", "2 年期国债收益率", G_RATE, "percent", "%", y2, a2 + " 月末",
                  f"2 年期国债收益率（月末）。10 月末为 {D.val('macro', 'y2'):.2f}%", D.pm_series("lvl_y2")))
    y10, a10 = D.pm_today("lvl_y10")
    add(num_field(D, "y10", "10 年期国债收益率", G_RATE, "percent", "%", y10, a10 + " 月末",
                  f"10 年期国债收益率（月末）。10 月末为 {D.val('macro', 'y10'):.2f}%，10 月月均 5.14%",
                  D.pm_series("lvl_y10")))
    tips, at = D.pm_today("s_tips10")
    add(num_field(D, "tips10", "10 年期 TIPS 实际收益率", G_RATE, "percent", "%", tips, at + " 月末",
                  "10 年期通胀保值国债收益率（DFII10，月末）。10 月末为 2.63%", D.pm_series("s_tips10"), nonneg=False))
    tp, atp = D.pm_today("s_tp10")
    add(num_field(D, "term_premium", "10 年期限溢价（ACM）", G_RATE, "number", "百分点", tp, atp + " 月末",
                  "纽约联储 ACM 模型的 10 年期限溢价", D.pm_series("s_tp10"), nonneg=False))

    # --- prices and expectations
    cpi, ac = D.pm_today("s_cpi_yoy")
    add(num_field(D, "cpi_yoy", "CPI 同比", G_PRICE, "percent", "% 同比", cpi, ac,
                  f"消费者价格指数同比。能源推动：WTI 同比 {D.pm_today('s_oil_yoy')[0]:+.0f}%", D.pm_series("s_cpi_yoy"),
                  nonneg=False))
    core, acc = D.pm_today("s_core_yoy")
    add(num_field(D, "core_cpi_yoy", "核心 CPI 同比", G_PRICE, "percent", "% 同比", core, acc,
                  "剔除食品与能源的 CPI 同比（CPILFESL）", D.pm_series("s_core_yoy"), nonneg=False))
    pce = D.line("fed", "core_pce_yoy")
    add(num_field(D, "core_pce_yoy", "核心 PCE 同比", G_PRICE, "percent", "% 同比", pce["value"], pce["as_of"][:7],
                  f"美联储的目标口径（目标 2%）：通胀缺口 {D.val('fed', 'inflation_gap'):.2f} 个百分点", None,
                  lo=-2.0, hi=12.0))
    m1, am1 = D.pm_today("s_mich1y")
    add(num_field(D, "infl_exp_1y", "1 年期通胀预期（密歇根）", G_PRICE, "percent", "%", m1, am1,
                  "密歇根大学调查的 1 年通胀预期中位数", D.pm_series("s_mich1y"), nonneg=False, d=1))
    lr, alr = D.pq_today("s_infexp_lr")
    add(num_field(D, "infl_exp_lr", "长期通胀预期（密歇根 5–10 年）", G_PRICE, "percent", "%", lr, f"{alr} 季初",
                  D.card["s_infexp_lr"]["definition"].split("；")[0], D.pq_series("s_infexp_lr"), d=1, freq="q"))
    be = D.fred("T10YIE_m_avg.json")
    add(num_field(D, "breakeven10", "10 年期盈亏平衡通胀率", G_PRICE, "percent", "%", float(be.loc[TODAY_M]), "2026-09 月均",
                  "10 年期名义国债与 TIPS 收益率之差（T10YIE）", be, nonneg=False))

    # --- growth and investment
    g, ag = D.pq_today("d_rgdp")
    add(num_field(D, "real_gdp_growth", "实际 GDP 增长（季度年化）", G_GROW, "percent", "% 年化", g, ag,
                  D.card["d_rgdp"]["definition"], D.pq_series("d_rgdp"), nonneg=False, d=1, freq="q"))
    it, ait = D.pm_today("s_it_invest_share")
    add(num_field(D, "it_invest_gdp", "IT 设备与软件投资/GDP", G_GROW, "percent", "% GDP", it, ait,
                  "名义信息处理设备与软件投资 / 名义 GDP（最新可得季度）", D.pm_series("s_it_invest_share")))
    cap = D.line("bigtech", "capex_ttm")
    add(num_field(D, "hyperscaler_capex_ttm", "超大型科技公司资本开支（近 4 季合计）", G_GROW, "number", "十亿美元",
                  cap["value"], cap["as_of"],
                  "微软、Alphabet、亚马逊、Meta、甲骨文（SEC 季报）。一年前为 "
                  f"{D.val('bigtech', 'capex_ttm_prev'):.0f}；2026 年五家计划合计约 "
                  f"{D.inst['hyperscaler_capex_2026_bn']['value']:.0f}（二手来源，未确认）", None, lo=0.0, hi=2000.0, d=1))

    # --- labour
    u, au = D.pm_today("lvl_unrate")
    add(num_field(D, "unemployment", "失业率", G_LAB, "percent", "%", u, au, "U-3 失业率",
                  D.pm_series("lvl_unrate"), d=1))
    pay = D.fred("PAYEMS.json")
    pay3 = (pay - pay.shift(3)) / 3
    m = pay.diff()
    add(num_field(D, "payrolls_3m", "非农就业月均新增（近 3 个月）", G_LAB, "number", "千人/月", float(pay3.loc[TODAY_M]),
                  "2026-07~09",
                  f"非农就业人数的 3 个月平均月度变动。7、8、9 月分别为 {m.loc['2026-07-01']:+.0f}、"
                  f"{m.loc['2026-08-01']:+.0f}、{m.loc['2026-09-01']:+.0f} 千人", pay3, nonneg=False, d=1))
    cl, acl = D.pm_today("s_claims")
    add(num_field(D, "initial_claims", "首次申领失业救济（月均）", G_LAB, "number", "千人/周", cl, acl,
                  "周度首次申领的月均值（ICSA）。10 月各周为 194–200 千人", D.pm_series("s_claims"), d=1))
    ahe = D.fred("AHETPI.json")
    wy = yoy(ahe)
    add(num_field(D, "wage_growth", "时薪同比（生产与非管理人员）", G_LAB, "percent", "% 同比", float(wy.loc[TODAY_M]), "2026-09",
                  f"平均时薪（AHETPI，{ahe.loc[TODAY_M]:.2f} 美元）同比", wy, nonneg=False))
    ls, als = D.pm_today("lvl_labor_share")
    add(num_field(D, "labor_share", "劳动收入占比（指数，2017=100）", G_LAB, "number", "指数", ls, als,
                  "非农企业部门劳动收入占比（最新可得季度）", D.pm_series("lvl_labor_share"), d=1))

    # --- credit and financial conditions
    sp, asp = D.pm_today("s_spread")
    add(num_field(D, "baa_spread", "Baa 信用利差（相对 10 年国债）", G_CRED, "number", "百分点", sp, asp + " 月末",
                  "穆迪 Baa 公司债收益率与 10 年期国债收益率之差（同一时点）", D.pm_series("s_spread")))
    gz, agz = D.pm_today("s_gz")
    add(num_field(D, "gz_spread", "GZ 信用利差", G_CRED, "number", "百分点", gz, agz,
                  f"Gilchrist–Zakrajšek 公司债利差（美联储）；超额债券溢价 EBP {D.pm_today('s_ebp')[0]:.2f}"
                  "（低于 0 表示风险偏好偏高）", D.pm_series("s_gz")))
    lstd, alstd = D.pq_today("x_lending_std")
    add(num_field(D, "lending_std", "银行放贷标准（SLOOS 净收紧比例）", G_CRED, "percent", "% 净收紧", lstd, alstd,
                  "高级信贷员调查：对大中型企业收紧工商业贷款标准的银行净占比（正 = 收紧）", D.pq_series("x_lending_std"),
                  nonneg=False, d=1, freq="q", lo=-100.0, hi=100.0))

    # --- fiscal
    dfc, adf = D.pm_today("s_deficit_gdp")
    add(num_field(D, "deficit_gdp", "联邦赤字/GDP", G_FISC, "percent", "% GDP", dfc, "最新可得季度 2026Q2",
                  f"NIPA 联邦净借款 / GDP。2026 财年赤字 {D.val('fiscal', 'deficit_fy') / 1000:.2f} 万亿美元"
                  f"（收入 {D.val('fiscal', 'receipts_fy') / 1000:.2f}、支出 {D.val('fiscal', 'outlays_fy') / 1000:.2f} 万亿）",
                  D.pm_series("s_deficit_gdp"), nonneg=False))
    debt, _ = D.pm_today("s_debt_gdp")
    add(num_field(D, "debt_gdp", "联邦公众持有债务/GDP", G_FISC, "percent", "% GDP", debt, "最新可得季度 2026Q2",
                  f"公众持有债务 {D.val('fiscal', 'debt_held_public') / 1000:.1f} 万亿美元（2026-10-29）",
                  D.pm_series("s_debt_gdp"), d=2))
    ni = D.line("fiscal", "net_interest_share_receipts")
    add(num_field(D, "net_interest_share", "净利息支出/联邦收入", G_FISC, "percent", "%", ni["value"], "2026 财年",
                  f"2026 财年净利息 {D.val('fiscal', 'net_interest_fy'):.0f} 十亿美元（上年 "
                  f"{D.val('fiscal', 'net_interest_prior_fy'):.0f}）；可交易国债平均利率 "
                  f"{D.val('fiscal', 'avg_rate_marketable'):.2f}%", None, lo=0.0, hi=50.0))
    hr = D.line("fiscal", "limit_headroom")
    add(num_field(D, "debt_limit_headroom", "债务上限余量", G_FISC, "number", "十亿美元", hr["value"], hr["as_of"],
                  f"法定上限 {D.val('fiscal', 'statutory_limit') / 1000:.2f} 万亿美元减受限债务；按近 3 个月月均新增 "
                  f"{D.val('fiscal', 'debt_pace_3m'):.0f} 十亿美元约 {D.val('fiscal', 'months_to_limit_at_3m_pace'):.1f} "
                  "个月触及（X 日期未锁定，未确认）；上限被提高或暂停时这一数值随之重设", None, lo=0.0, hi=10000.0, d=1))

    # --- housing
    hs = D.fred("HOUST.json")
    add(num_field(D, "housing_starts", "新屋开工（年化）", G_HOUSE, "number", "千套", float(hs.loc[TODAY_M]), "2026-09",
                  f"新屋开工年化（HOUST）；6 月为 {hs.loc['2026-06-01']:.0f}", hs, d=0))
    hp, ahp = D.pm_today("s_hpi_yoy")
    add(num_field(D, "hpi_yoy", "房价同比（Freddie Mac）", G_HOUSE, "percent", "% 同比", hp, ahp,
                  "Freddie Mac 全国房价指数同比", D.pm_series("s_hpi_yoy"), nonneg=False))
    mo = D.fred("MORTGAGE30US_m_eop.json")
    add(num_field(D, "mortgage30", "30 年期抵押贷款利率", G_HOUSE, "percent", "%", float(mo.loc[TODAY_M]), "2026-09 月末",
                  f"Freddie Mac 30 年期固定利率；10 月末 {mo.iloc[-1]:.2f}%", mo))

    # --- corporate sector and distribution
    ps, aps = D.pm_today("s_profit_share")
    add(num_field(D, "profit_share", "企业税后利润/GDP", G_CORP, "percent", "% GDP", ps, aps + "（2026Q2 数据）",
                  "NIPA 税后企业账面利润 / 名义 GDP", D.pm_series("s_profit_share")))
    dy, ady = D.pm_today("s_div_yield")
    add(num_field(D, "dividend_yield", "美股股息率", G_CORP, "percent", "%", dy, ady + " 月末",
                  "美国股市的股息率（Ken French 数据库，过去 12 个月股息 / 市值）：一个估值比率，不是价格水平",
                  D.pm_series("s_div_yield")))

    # --- households
    sr, asr = D.pm_today("lvl_saving_rate")
    add(num_field(D, "saving_rate", "个人储蓄率", G_HH, "percent", "% 可支配收入", sr, asr,
                  f"个人储蓄 / 可支配收入；1959 年以来第 1 百分位为 {D.val('macro', 'psavert_p1_since_1959'):.1f}%",
                  D.pm_series("lvl_saving_rate"), d=1))
    tdsp, atd = D.pq_today("s_tdsp")
    add(num_field(D, "hh_debt_service", "家庭偿债负担", G_HH, "percent", "% 可支配收入", tdsp, f"{atd} 季初",
                  "家庭偿债支出 / 可支配收入（美联储 DSR）", D.pq_series("s_tdsp"), freq="q"))
    se, ase = D.pm_today("lvl_sentiment")
    add(num_field(D, "consumer_sentiment", "消费者信心（密歇根）", G_HH, "number", "指数（1966Q1=100）", se, ase,
                  "密歇根大学消费者信心指数（所说的“消费者信心 48.1”即此指数，不是世界大型企业联合会的指数）",
                  D.pm_series("lvl_sentiment"), d=1))

    # --- external, energy and trade
    usd, ausd = D.pm_today("lvl_usd")
    add(num_field(D, "usd_index", "贸易加权美元指数（广义）", G_EXT, "number", "指数", usd, ausd + " 月末",
                  f"美联储广义贸易加权美元指数（DTWEXBGS）；同比 {D.pm_today('s_usd_yoy')[0]:+.1f}%",
                  D.pm_series("lvl_usd"), d=1))
    wti, aw = D.pm_today("lvl_oil")
    wti_oct = float(D.fred("DCOILWTICO_eop.csv").iloc[-1])
    add(num_field(D, "wti", "WTI 原油价格", G_EXT, "number", "美元/桶", wti, aw + " 月末",
                  f"WTI 现货（月末）。10 月末 {wti_oct:.1f}；布伦特 10-27 为 {D.val('opec', 'brent_spot'):.1f}。"
                  "2026 年冲突导致的供给中断仍在持续（未确认）", D.pm_series("lvl_oil"), d=1))
    gas = D.line("president", "gasoline_price")
    add(num_field(D, "gasoline", "汽油零售价", G_EXT, "number", "美元/加仑", gas["value"], gas["as_of"],
                  "美国常规汽油全国均价（EIA 周度）", None, lo=1.0, hi=8.0))
    spr = D.line("president", "spr_inventory")
    add(num_field(D, "spr", "战略石油储备库存", G_EXT, "number", "百万桶", spr["value"], spr["as_of"],
                  f"近一年减少 {-D.val('president', 'spr_change_52w'):.1f} 百万桶；美国原油产量 "
                  f"{D.val('opec', 'us_crude_production'):.2f} 百万桶/日", None, lo=0.0, hi=750.0, d=1))
    tr = D.line("president", "effective_tariff_rate")
    tr_panel, _ = D.pm_today("s_tariff_rate")
    add(num_field(D, "tariff_rate", "有效关税率", G_EXT, "percent", "%", tr["value"], "2026Q3（NIPA）",
                  f"关税收入 / 商品进口额（NIPA，年化 {D.val('fiscal', 'customs_nipa_saar'):.0f} / "
                  f"{D.val('fiscal', 'goods_imports_saar'):.0f} 十亿美元）。面板的最新读数 {tr_panel:.2f}% 是 2026Q2；"
                  f"三季度的回落包含 5–9 月 {D.val('fiscal', 'customs_refunds_since_2026_05'):.1f} 十亿美元的关税退款（原因未确认）",
                  D.pm_series("s_tariff_rate")))
    cny = D.line("china", "cny_per_usd")
    add(num_field(D, "cny_usd", "人民币兑美元", G_EXT, "number", "元/美元", cny["value"], cny["as_of"],
                  "在岸人民币汇率（每美元）", None, lo=5.5, hi=8.5))

    # --- confidence and uncertainty
    epu, aepu = D.pm_today("s_epu")
    add(num_field(D, "epu", "经济政策不确定性指数", G_CONF, "number", "指数", epu, aepu,
                  "Baker–Bloom–Davis 新闻 EPU 指数（月度）", D.pm_series("s_epu"), d=1))
    gpr, agpr = D.pm_today("s_gpr")
    add(num_field(D, "gpr", "地缘政治风险指数", G_CONF, "number", "指数", gpr, agpr,
                  "Caldara–Iacoviello 地缘政治风险指数", D.pm_series("s_gpr"), d=1))

    # --- qualitative fields from the pinned facts (institutions.json)
    add(field("fed_stance", "美联储政策立场", G_POL, "enum", "偏鹰观望（暂停加息）", options=[
        "偏鸽（降息周期）", "中性观望", "偏鹰观望（暂停加息）", "偏鹰（继续加息）"],
        description="由数据判断：9 月加息 25bp、10 月会议维持（未确认）；通胀高于 3%", tag="判断"))
    fomc = D.inst["fomc_2026_10"]
    add(field("fomc_last", "最近一次议息会议（10 月 27–28 日）", G_POL, "enum", "维持",
              options=["维持", "降息 25bp", "降息 50bp", "加息 25bp"], description=fomc["default_reason"],
              tag=unconfirmed(D, "fomc_2026_10")))
    add(field("fed_chair", "美联储主席与理事会人事", G_POL, "text", "按数据日期的现任；主席倾向中性",
              description=D.inst["fed_chair"]["default_reason"], tag=unconfirmed(D, "fed_chair")))
    add(field("ieepa_status", "IEEPA 关税的法律状态", G_POL, "enum", "有效",
              options=["有效", "诉讼待决", "已被裁定无效", "已裁定无效并退款"],
              description=f"{D.inst['ieepa_status']['default_reason']}。{D.inst['refunds_cause']['default_reason']}",
              tag=unconfirmed(D, "ieepa_status")))
    add(field("oil_disruption", "石油供给中断（2026 年冲突）", G_POL, "enum", "中断持续",
              options=["无中断", "中断持续", "中断升级", "部分恢复", "全面恢复"],
              description=f"{D.inst['belligerents_active']['source']}；冲突方{D.inst['belligerents_identity']['value']}",
              tag=unconfirmed(D, "belligerents_active")))
    add(field("political_control", "两院与白宫的多数党", G_POL, "text",
              "白宫：共和党；参议院：共和党 53–47；众议院：共和党微弱多数",
              description=D.inst["political_control"]["default_reason"], tag=unconfirmed(D, "political_control")))
    add(field("house_control", "众议院控制权（2026-11-03 中期选举）", G_POL, "enum", "选举前：共和党微弱多数（结果未揭晓）",
              options=["选举前：共和党微弱多数（结果未揭晓）", "共和党保住众议院", "民主党赢得众议院"],
              description="2027-01-03 新国会就任；" + D.inst["election_2026_result"]["default_reason"]
              + "；没有市场赔率时按历史基率（执政党在中期选举中通常失去众议院席位）",
              tag=unconfirmed(D, "election_2026_result")))
    add(field("china_truce", "中美贸易休战", G_POL, "enum", "休战有效（约 2026-11-10 到期）",
              options=["休战有效（约 2026-11-10 到期）", "休战延长", "休战破裂（恢复加征）", "达成更广泛的协议"],
              description=D.inst["china_truce_expiry"]["default_reason"], tag=unconfirmed(D, "china_truce_expiry")))
    add(field("opec_policy", "欧佩克+ 产量政策", G_POL, "enum", "维持配额", options=["维持配额", "增产", "减产"],
              description=D.inst["opec_decision"]["default_reason"] + "；可达市场的闲置产能未锁定（EIA STEO）",
              tag=unconfirmed(D, "opec_decision")))
    add(field("debt_limit_status", "债务上限状态", G_POL, "enum", "接近上限",
              options=["余量充足", "接近上限", "动用非常规措施", "已提高或暂停", "对峙中"],
              description=f"由账户判断：余量约 {hr['value']:.0f} 十亿美元，按近 3 个月速度约 "
                          f"{D.val('fiscal', 'months_to_limit_at_3m_pace'):.1f} 个月；非常规措施容量（判断值 "
                          f"{D.inst['em_capacity_bn']['value']:.0f} 十亿美元）未确认", tag="判断"))
    add(field("chip_controls", "对华芯片出口管制与稀土管制", G_POL, "text", "按数据日期的现状",
              description=D.inst["chip_controls"]["default_reason"], tag=unconfirmed(D, "chip_controls")))
    fields = [f for f, _ in rows]
    values = {f["id"]: v for f, v in rows}
    return fields, values


# ------------------------------------------------------------------------------------------------ text

def build_text(D: Data, v: dict) -> dict:
    b = lambda k, n: D.val(k, n)  # noqa: E731
    narrative = "\n\n".join([
        f"2026 年第四季度开始时（本地数据截至 2026 年 9 月），美国经济在高通胀与放缓的就业之间拉扯。失业率 {v['unemployment']:.1f}%，"
        f"近三个月非农就业月均只新增约 {v['payrolls_3m']:.0f} 千人，首次申领约 {v['initial_claims']:.0f} 千人/周；三季度实际 GDP 年化增长 "
        f"{v['real_gdp_growth']:.1f}%。CPI 同比 {v['cpi_yoy']:.1f}%（主要来自能源），核心 CPI {v['core_cpi_yoy']:.1f}%，美联储关注的核心 PCE "
        f"{v['core_pce_yoy']:.1f}%；1 年期通胀预期 {v['infl_exp_1y']:.1f}%，消费者信心 {v['consumer_sentiment']:.1f}，处在历史低位。",
        f"美联储 9 月加息 25bp，有效联邦基金利率约 {v['policy_rate']:.2f}%；10 月 27–28 日的会议按数据推断为维持（未确认）。10 年期国债收益率"
        f" 9 月底 {v['y10']:.2f}%（10 月底回落到 {D.val('macro', 'y10'):.2f}%），10 年 TIPS 实际收益率 {v['tips10']:.2f}%，期限溢价 "
        f"{v['term_premium']:.2f} 个百分点；30 年期房贷利率 {v['mortgage30']:.2f}%，新屋开工降至年化 {v['housing_starts']:.0f} 千套。",
        f"2026 年的冲突导致石油供给中断仍在持续（未确认）：WTI 9 月底 {v['wti']:.0f} 美元/桶，10 月底约 103 美元，布伦特 10 月 27 日 "
        f"{b('opec', 'brent_spot'):.0f} 美元；IMF 预计科威特与伊拉克 2026 年经济分别收缩 {-b('opec', 'kwt_real_gdp_growth_2026'):.1f}% 与 "
        f"{-b('opec', 'irq_real_gdp_growth_2026'):.1f}%。美国汽油均价 {v['gasoline']:.2f} 美元/加仑，战略石油储备 {v['spr']:.0f} 百万桶，"
        "比一年前少约 127 百万桶。欧佩克+ 的配额按最近一次公开声明（未确认）。",
        f"财政仍在扩张：2026 财年赤字 {b('fiscal', 'deficit_fy') / 1000:.2f} 万亿美元，公众持有债务/GDP {v['debt_gdp']:.1f}%，净利息占联邦收入 "
        f"{v['net_interest_share']:.1f}%。债务上限余量约 {v['debt_limit_headroom']:.0f} 十亿美元，按近三个月的速度约 "
        f"{b('fiscal', 'months_to_limit_at_3m_pace'):.1f} 个月触及（X 日期未锁定）。有效关税率三季度为 {v['tariff_rate']:.2f}%，"
        f"5–9 月关税退款合计 {b('fiscal', 'customs_refunds_since_2026_05'):.0f} 十亿美元，原因未确认；IEEPA 关税按仍然有效处理（未确认），"
        "中美一年期暂停措施约在 11 月 10 日到期（未确认）。",
        f"11 月 3 日中期选举的结果尚未揭晓：选举前白宫与两院都由共和党控制（众议院微弱多数，未确认）。AI 投资周期仍在加速：五家超大型科技公司近"
        f"四季资本开支 {v['hyperscaler_capex_ttm']:.0f} 十亿美元（一年前 {b('bigtech', 'capex_ttm_prev'):.0f}），自由现金流接近零，近四季"
        f"新增债务 {b('bigtech', 'debt_change_4q'):.0f} 十亿美元；IT 投资/GDP {v['it_invest_gdp']:.2f}%。企业利润/GDP {v['profit_share']:.1f}%，"
        f"劳动收入占比指数 {v['labor_share']:.1f}，美股股息率 {v['dividend_yield']:.2f}%，个人储蓄率 {v['saving_rate']:.1f}%。",
    ])
    questions = [
        ("通胀高于 3% 时，美联储下一步是继续加息、按兵不动，还是转向降息？",
         f"9 月加息 25bp，10 月维持（未确认）；核心 PCE {v['core_pce_yoy']:.1f}%、失业率 {v['unemployment']:.1f}%，方向未定"),
        ("11 月 3 日中期选举后，执政的共和党能否保住众议院？", "结果未揭晓；历史上执政党在中期选举中通常失去席位"),
        ("IEEPA 关税的法律命运如何：维持、被裁定无效，还是无效并退款？",
         f"按仍然有效处理（未确认）；5–9 月已退款 {D.val('fiscal', 'customs_refunds_since_2026_05'):.0f} 十亿美元，原因未确认"),
        ("石油供给中断会升级、僵持还是恢复？油价能否回到 90 美元以下？", f"中断持续（未确认）；WTI 约 {v['wti']:.0f}–103 美元"),
        ("超大型科技公司的 AI 资本开支周期会继续加速，还是开始消化？",
         f"近四季 {v['hyperscaler_capex_ttm']:.0f} 十亿美元，约为一年前的 1.9 倍；自由现金流接近零、靠举债融资"),
        ("财政路径：债务上限能否在 X 日期前提高或暂停？赤字会扩大还是收缩？",
         f"余量约 {v['debt_limit_headroom']:.0f} 十亿美元，约 2.7 个月；2026 财年赤字约 2.0 万亿美元"),
        ("中美一年期休战到期后会延长、破裂，还是达成更广泛的协议？", "约 2026-11-10 到期（未确认）；尚无消息"),
        ("劳动力市场是否在走弱：失业率会升破 4.5% 吗？",
         f"失业率 {v['unemployment']:.1f}%，近三个月就业月均新增约 {v['payrolls_3m']:.0f} 千人，初请仍低"),
    ]
    attributes = [
        ("美联储独立性", "形式上维持；主席倾向中性（未确认）"),
        ("货币政策", f"偏鹰观望：9 月加息后利率约 {v['policy_rate']:.2f}%"),
        ("关税制度", f"IEEPA 关税按有效处理（未确认）；有效关税率 {v['tariff_rate']:.2f}%"),
        ("石油供给", f"冲突导致的中断持续（未确认）；WTI 约 {v['wti']:.0f}–103 美元"),
        ("财政立场", "扩张：赤字约 6% GDP，债务上限临近"),
        ("政治格局", "共和党控制白宫与两院（选举前，未确认）"),
        ("AI 投资周期", "加速：资本开支一年翻倍，越来越依赖举债"),
        ("中美关系", "一年期休战，约 11 月 10 日到期（未确认）"),
        ("居民部门", f"信心低迷（{v['consumer_sentiment']:.1f}），储蓄率 {v['saving_rate']:.1f}% 偏低"),
    ]
    return {"narrative": narrative,
            "key_questions": [{"question": q, "status": "open", "answer": a} for q, a in questions],
            "attributes": [{"name": n, "value": x} for n, x in attributes]}


# ------------------------------------------------------------------------------------------------ actors

UNIT_ZH = {"$bn": "十亿美元", "$bn SAAR": "十亿美元（年化）", "$bn/q": "十亿美元/季", "$bn/4q": "十亿美元/近 4 季",
           "$bn/13w": "十亿美元/13 周", "$bn/month": "十亿美元/月", "$/gal": "美元/加仑", "$/bbl": "美元/桶",
           "mn bbl": "百万桶", "mb/d": "百万桶/日", "% GDP": "% GDP", "% DPI": "% 可支配收入", "% NW": "% 净资产",
           "ratio": "倍", "x": "倍", "pp": "个百分点", "index": "（指数）", "months": "个月", "CNY/USD": "元/美元",
           "pctile": "百分位", "%": "%"}
BIGTECH_ZH = {"revenue": "收入", "operating_income": "营业利润", "cfo": "经营现金流", "capex": "资本开支", "fcf": "自由现金流",
              "buybacks": "回购", "dividends": "股息", "cash": "现金", "securities": "有价证券", "debt": "债务",
              "rpo": "剩余履约义务（RPO）", "depreciation": "折旧", "op_margin": "营业利润率",
              "capex_to_cfo": "资本开支/经营现金流", "lease_principal": "融资租赁本金"}


def book_lines(D: Data, book: str, keys: list[str] | None = None, title: str | None = None) -> list[str]:
    lines = D.acct["books"][book]["lines"]
    out = [f"【{title}】"] if title else []
    for k in keys or list(lines):
        if k not in lines:
            continue
        ln = lines[k]
        zh = ln.get("zh") or BIGTECH_ZH.get(k, k)
        unit = UNIT_ZH.get(ln.get("unit") or "", ln.get("unit") or "")
        v = ln.get("value")
        if not isinstance(v, (int, float)):
            continue
        d = 1 if abs(v) >= 100 else 2
        sep = "" if unit.startswith("%") or unit.startswith("（") else " "
        out.append(f"- {zh}：{fmt(float(v), d)}{sep}{unit}（{ln.get('as_of')}）")
    return out


INST_VALUE_ZH = {"hold": "维持", "cut_25": "降息 25bp", "cut_50": "降息 50bp", "hike_25": "加息 25bp",
                 "in_force": "有效", "pending": "诉讼待决", "struck_down": "已被裁定无效",
                 "struck_down_refunds": "已裁定无效并退款", "unknown": "未知", "house_holds": "执政党保住众议院",
                 "house_flips": "执政党失去众议院", "none": "无", "pending_this_quarter": "本季待决"}


def inst_lines(D: Data, actor_ids: tuple[str, ...]) -> list[str]:
    out = ["【制度现状（来自 institutions.json；“未确认”表示用户尚未锁定）】"]
    for it in D.inst.values():
        if not set(it.get("affects") or []) & set(actor_ids):
            continue
        val = it.get("value")
        if val in (None, ""):
            val = "未锁定"
        elif isinstance(val, bool):
            val = "是" if val else "否"
        elif isinstance(val, str):
            val = INST_VALUE_ZH.get(val, val)
        status = "未确认" if it.get("status") == "unconfirmed" else "已确认"
        out.append(f"- {it['zh']}：{val}{(' ' + it['unit']) if it.get('unit') and val != '未锁定' else ''}（{status}；"
                   f"{it.get('default_reason') or ''}）")
    return out


def catalogue():
    from scenariolab.agents import actors as A
    return A.ACTORS


def cat_action(cat, actor_id: str, domain: str, option: str, description: str, label: str | None = None) -> dict:
    opt = cat[actor_id].domain(domain).option(option)
    return {"id": option, "label": label or opt.zh, "description": description}


def build_actors(D: Data, v: dict) -> list[dict]:
    cat = catalogue()
    A = []

    def actor(aid, name, role, cares, actions, private):
        A.append({"id": aid, "name": name, "enabled": True, "role": role, "cares_about": cares,
                  "action_space": actions, "allow_free_action": True, "private_info": "\n".join(private),
                  "visible_fields": None, "model": ""})

    actor("fed", "美联储", "美国联邦储备体系的联邦公开市场委员会（作为机构，不模仿任何个人的语气）。本季有两次议息会议，"
          "决定联邦基金利率目标区间与资产负债表安排；在通胀高于目标、就业放缓、油价冲击与财政扩张同时出现时权衡双重使命。",
          ["把通胀（核心 PCE）拉回 2% 的目标并稳住通胀预期", "避免不必要的失业上升（就业缺口接近 0）",
           "金融稳定：信用利差、银行与国债市场的运转", "维护政策独立性与可信度", "资产负债表与准备金的平稳运行"],
          [cat_action(cat, "fed", "rate", "hold", "两次会议都维持目标区间（现状）。"),
           cat_action(cat, "fed", "rate", "hike_25", "本季累计加息 25bp。"),
           cat_action(cat, "fed", "rate", "hike_50", "本季累计加息 50bp（两次各 25bp 或一次 50bp）。"),
           cat_action(cat, "fed", "rate", "cut_25", "本季累计降息 25bp。"),
           cat_action(cat, "fed", "rate", "cut_50", "本季累计降息 50bp。"),
           cat_action(cat, "fed", "balance_sheet", "bs_runoff", "停止准备金管理购买，恢复缩表（资产/GDP 每季约 −0.5pp，判断值）。"),
           cat_action(cat, "fed", "balance_sheet", "bs_reserve_purchases", "扩大准备金管理购买（增持国库券）。")],
          book_lines(D, "fed", None, "美联储账户（H.4.1 与数据）") + inst_lines(D, ("fed",)))

    actor("president", "总统与白宫", "美国总统与白宫（作为机构，不模仿任何人的语气）。掌握贸易与关税（122、301、232 条款与 IEEPA 授权）、"
          "能源（战略石油储备）、移民执法与对外谈判，也会对美联储公开施压；关注中期选举前后的政治处境。",
          ["选民的生活成本感受：汽油价格与通胀", "中期选举与执政党的国会席位", "贸易逆差、产业回流与对华谈判筹码",
           "股市与就业的表现", "关税的法律授权与收入"],
          [cat_action(cat, "president", "trade", "tariff_hold", "维持现有关税（现状；有效关税率仍随进口构成与退款波动）。"),
           cat_action(cat, "president", "trade", "tariff_up_small", "对特定行业或国家加征关税。"),
           cat_action(cat, "president", "trade", "tariff_up_large", "广泛加征关税（约 +2pp 有效关税率，历史上罕见）。"),
           cat_action(cat, "president", "trade", "tariff_down", "降低关税或扩大豁免。"),
           cat_action(cat, "president", "trade", "deal", "与主要伙伴（含中国）达成协议：下调关税并换取承诺。"),
           cat_action(cat, "president", "energy", "spr_release", "动用战略石油储备以压低油价。"),
           cat_action(cat, "president", "fed_pressure", "pressure_public", "公开向美联储施压，要求降息。", "公开向美联储施压")],
          book_lines(D, "president", None, "白宫关注的指标") + inst_lines(D, ("president",)))

    actor("congress", "国会", "美国国会两院（作为机构）。决定最终成为法律的财政立法：拨款与续命、税收、转移支付，以及债务上限的提高、暂停或对峙；"
          "11 月 3 日中期选举后，新国会 2027-01-03 就任，跛脚鸭会期内的谈判筹码会变化。",
          ["避免债务违约与政府停摆的政治代价", "选区选民的减税与支出诉求", "赤字与利息负担的可持续性",
           "选举结果带来的权力格局", "关税退款与财政收入"],
          [cat_action(cat, "congress", "fiscal", "no_new_law", "本季没有新的财政立法，只有自动稳定器（现状）。"),
           cat_action(cat, "congress", "fiscal", "debt_limit_raise", "提高债务上限（2011 年以来的约束期都得到解决）。"),
           cat_action(cat, "congress", "fiscal", "debt_limit_suspend", "暂停债务上限一段时间。"),
           cat_action(cat, "congress", "fiscal", "debt_limit_standoff", "债务上限对峙，本季未解决。"),
           cat_action(cat, "congress", "fiscal", "stimulus_small", "小规模刺激：退税或直接支付（约 0.8% GDP）。"),
           cat_action(cat, "congress", "fiscal", "consolidation", "财政整顿：削减支出或增税。"),
           cat_action(cat, "congress", "deadline", "shutdown", "拨款到期未续，政府停摆。")],
          book_lines(D, "fiscal", ["deficit_fy", "receipts_fy", "outlays_fy", "net_interest_fy", "net_interest_share_receipts",
                                   "debt_held_public", "statutory_limit", "limit_headroom", "debt_pace_3m",
                                   "months_to_limit_at_3m_pace", "customs_refunds_since_2026_05", "deficit_nipa_gdp",
                                   "debt_gdp"], "联邦财政账户") + inst_lines(D, ("congress",)))

    actor("treasury", "财政部", "美国财政部（作为机构）。管理国债发行结构（季度再融资：短债占比与期限）、回购、现金余额（TGA），"
          "在触及债务上限时动用非常规措施，并向国会估计 X 日期。",
          ["按最低成本为赤字融资", "国债市场的流动性与平稳运行", "在债务上限约束下避免违约", "现金余额（TGA）的安全垫"],
          [cat_action(cat, "treasury", "issuance", "refunding_unchanged", "季度再融资不改变发行结构（现状）。"),
           cat_action(cat, "treasury", "issuance", "bills_up", "提高短期国库券占比。"),
           cat_action(cat, "treasury", "issuance", "bills_down", "拉长发行期限，增加中长期国债。"),
           cat_action(cat, "treasury", "issuance", "buybacks_up", "扩大国债回购以改善流动性。"),
           {"id": "extraordinary_measures", "label": "动用非常规措施",
            "description": "触及上限前后动用非常规措施并消耗 TGA，向国会通报 X 日期（容量判断值，未确认）。"}],
          book_lines(D, "fiscal", ["tga_dts", "debt_held_public", "bill_share", "mspd_bills", "mspd_notes", "mspd_bonds",
                                   "avg_rate_marketable", "avg_rate_marketable_12m_ago", "foreign_holdings_tic",
                                   "foreign_official_holdings", "limit_headroom", "months_to_limit_at_3m_pace"],
                     "财政部现金与债务管理") + inst_lines(D, ("treasury",)))

    actor("courts", "最高法院与法院", "美国最高法院与联邦法院（作为机构）。在有待决案件时作出裁决：IEEPA 关税授权、美联储理事的免职保护、"
          "大型科技公司的反垄断救济；没有到期的案件时本季不作裁决。只依据法律与先例，不考虑市场反应。",
          ["法律文本与先例", "行政权的边界（关税授权、独立机构的免职保护）", "裁决的可执行性（例如退款）", "司法机构的独立性"],
          [{"id": "no_ruling", "label": "本季不作相关裁决", "description": "案件继续审理或没有到期的裁决（现状）。"},
           cat_action(cat, "courts", "tariff_authority", "uphold", "维持 IEEPA 关税授权。", "维持 IEEPA 关税授权"),
           cat_action(cat, "courts", "tariff_authority", "strike_down", "裁定 IEEPA 关税授权无效，但不要求退税。",
                      "IEEPA 授权无效（不退税）"),
           cat_action(cat, "courts", "tariff_authority", "strike_down_refunds", "裁定授权无效并要求退税（一次性财政事件）。",
                      "IEEPA 授权无效并退税"),
           cat_action(cat, "courts", "fed_removal", "allow_removal", "允许总统罢免美联储理事（独立性冲击）。",
                      "允许罢免美联储理事"),
           cat_action(cat, "courts", "antitrust", "structural", "对大型科技公司判处结构性救济（拆分或剥离）。",
                      "反垄断：结构性救济（拆分/剥离）")],
          inst_lines(D, ("courts",)))

    actor("voters", "选民", "美国选民（作为整体）。本季在 11 月 3 日中期选举中投票，决定众议院（和部分参议院席位）的归属；"
          "他们的判断来自生活成本、就业与对执政党的观感。",
          ["生活成本：汽油、食品与房租", "就业与收入的安全感", "对执政党政策的评价", "移民、关税等议题"],
          [cat_action(cat, "voter", "midterm", "house_holds", "执政的共和党保住众议院（延续现状）。"),
           cat_action(cat, "voter", "midterm", "house_flips", "执政党失去众议院：新国会 2027-01-03 起分治。"),
           {"id": "both_flip", "label": "执政党失去两院", "description": "执政党同时失去众议院与参议院（较罕见）。"}],
          book_lines(D, "president", ["cpi_yoy", "gasoline_price", "consumer_sentiment", "real_dpi_pc_yoy",
                                      "us_equity_drawdown"], "选民感受到的经济")
          + book_lines(D, "households", ["saving_rate", "debt_service_ratio", "nw_share_top1", "nw_share_bottom50"],
                       "居民账户") + inst_lines(D, ("voter",)))

    actor("china", "中国", "中国中央政府与人民银行（作为机构）。应对美国的关税与出口管制（对等关税、稀土与磁材管制、谈判），"
          "决定国内宏观刺激力度与人民币汇率，并在台海问题上表态。",
          ["经济增长目标与就业", "对美出口与产业链安全（芯片、稀土）", "人民币汇率稳定与资本流动", "房地产下行与地方债务",
           "台海与地缘政治"],
          [cat_action(cat, "china", "trade_response", "no_response", "不作新的对美回应（现状）。"),
           cat_action(cat, "china", "trade_response", "retaliate_tariffs", "对等加征关税。"),
           cat_action(cat, "china", "trade_response", "retaliate_export_controls", "加征关税并收紧稀土等出口管制。"),
           cat_action(cat, "china", "trade_response", "negotiate_truce", "谈判延长或扩大休战。"),
           cat_action(cat, "china", "domestic", "major_stimulus", "推出大规模财政与信贷刺激。"),
           cat_action(cat, "china", "fx", "cny_weaker", "允许人民币有管理地贬值。", "人民币有管理的贬值"),
           cat_action(cat, "china", "taiwan", "tw_large_exercise", "在台海举行大规模军演。", "台海大规模军演")],
          book_lines(D, "china", None, "中国账户（IMF、TIC、BIS）") + inst_lines(D, ("china",)))

    actor("opec", "欧佩克+", "欧佩克+（以沙特为摆动产油国，作为机构）。决定产量配额；在冲突导致供给中断、油价高企时权衡市场份额、"
          "财政收入与美国页岩油的竞争。",
          ["成员国财政收入（沙特财政赤字）", "市场份额与美国页岩油的竞争", "联盟团结与配额纪律", "与美国的关系", "避免需求被高油价摧毁"],
          [cat_action(cat, "opec", "quota", "quota_hold", "维持配额（现状）。"),
           cat_action(cat, "opec", "quota", "quota_raise", "增产（受可达市场的闲置产能约束）。"),
           cat_action(cat, "opec", "quota", "quota_cut", "减产以托住油价。")],
          book_lines(D, "opec", None, "产油国账户（IMF WEO、EIA）") + inst_lines(D, ("opec",)))

    actor("belligerents", "冲突方", "2026 年导致石油供给中断的武装冲突中的产油方与交战各方（身份未锁定，不具名）。他们的行动决定航道、设施与供给的恢复或进一步中断。",
          ["军事与政治目标", "石油收入与设施安全", "外部调停与制裁压力", "国内经济的承受力"],
          [cat_action(cat, "belligerents", "conflict", "stalemate", "冲突僵持，中断持续（现状）。"),
           cat_action(cat, "belligerents", "conflict", "escalate", "冲突升级，供给进一步中断，地缘风险上升。"),
           cat_action(cat, "belligerents", "conflict", "ceasefire", "停火或航道恢复，供给逐步回升。")],
          book_lines(D, "opec", ["brent_spot", "kwt_real_gdp_growth_2026", "irq_real_gdp_growth_2026",
                                 "kwt_current_account_gdp_2026", "irq_current_account_gdp_2026"], "受冲突影响的产油国")
          + inst_lines(D, ("belligerents",)))

    actor("bigtech", "超大型科技公司 CEO", "微软、Alphabet、亚马逊、Meta、甲骨文的管理层（CEO、CFO，作为整体；不用任何在世人物的姓名）。"
          "每季财报时决定 AI 资本开支指引、用工、股东回报与 AI 定价。",
          ["AI 竞赛中的位置与算力", "资本开支的回报与自由现金流", "债务融资成本与评级", "股东回报与股价",
           "监管（反垄断、出口管制）"],
          [cat_action(cat, "bigtech", "capex", "capex_hold", "维持或小幅调整资本开支指引（现状）。", "资本开支：维持或小幅调整"),
           cat_action(cat, "bigtech", "capex", "capex_raise_large", "大幅上调资本开支指引（持续约 4 季的承诺）。",
                      "资本开支：大幅上调"),
           cat_action(cat, "bigtech", "capex", "capex_cut", "下调资本开支，进入消化期。", "资本开支：下调（消化）"),
           cat_action(cat, "bigtech", "capex", "capex_slash", "大幅削减资本开支（2001–02 型）。", "资本开支：大幅削减（2001–02 型）"),
           cat_action(cat, "bigtech", "workforce", "hire_freeze_ai", "冻结招聘，用 AI 替代部分白领岗位。"),
           cat_action(cat, "bigtech", "payout", "buyback_cut", "削减回购为资本开支融资。"),
           cat_action(cat, "bigtech", "pricing", "price_war", "AI 服务价格战。")],
          book_lines(D, "bigtech", None, "五家合计（SEC 季报，2026Q3）")
          + book_lines(D, "bigtech_msft", ["capex", "fcf", "rpo"], "微软")
          + book_lines(D, "bigtech_googl", ["capex", "fcf", "rpo"], "Alphabet")
          + book_lines(D, "bigtech_amzn", ["capex", "fcf", "debt"], "亚马逊")
          + book_lines(D, "bigtech_meta", ["capex", "fcf", "debt"], "Meta")
          + book_lines(D, "bigtech_orcl", ["capex", "fcf", "debt", "rpo"], "甲骨文") + inst_lines(D, ("bigtech",)))

    actor("households", "消费者/居民", "美国家庭与消费者（作为整体）。决定消费与储蓄、是否去杠杆，以及买房还是观望；"
          "财富高度集中，股票占金融资产的比重处在历史高位。",
          ["收入与就业安全", "生活成本：汽油、食品、房租与利率", "财富与股市", "债务负担与信贷可得性", "买房的时机"],
          [{"id": "spend_normal", "label": "按收入正常消费", "description": "消费随收入增长（现状）。"},
           {"id": "precautionary_saving", "label": "预防性储蓄", "description": "压缩消费、提高储蓄率（储蓄率上升 0.75pp 以上）。"},
           {"id": "dissave", "label": "动用储蓄维持消费", "description": "储蓄率进一步下降以维持消费。"},
           {"id": "deleverage", "label": "去杠杆", "description": "偿还债务，家庭债务/收入下降。"},
           {"id": "housing_wait", "label": "住房：观望", "description": "推迟买房，住宅投资下降。"},
           {"id": "housing_rush", "label": "住房：抢先入市", "description": "预期利率或房价上行而提前买房。"}],
          book_lines(D, "households", None, "居民账户（Z.1、DFA、BEA）"))

    actor("investors", "华尔街投资者", "资产管理者、对冲基金、券商与散户的整体（作为市场力量）。决定风险配置与科技仓位；"
          "他们的行动通过信用利差、期限溢价与美元影响世界，而不是由模拟写出资产价格。",
          ["风险调整后的回报", "利率与通胀的路径", "流动性与保证金", "AI 主题的估值与集中度", "政策与地缘风险"],
          [{"id": "hold_allocation", "label": "维持配置", "description": "仓位不变（现状）。"},
           {"id": "risk_on", "label": "加仓风险资产", "description": "提高股票与信用仓位，信用利差收窄。"},
           {"id": "de_risk", "label": "降低风险", "description": "减仓、转向现金与短债，信用利差走阔。"},
           {"id": "flight_to_safety", "label": "避险抛售", "description": "恐慌性抛售与避险，利差大幅走阔、美元走强。"},
           {"id": "tech_overweight", "label": "科技仓位：加仓", "description": "继续集中于 AI 与科技。"},
           {"id": "tech_rotate_out", "label": "科技仓位：轮动出去", "description": "从科技转向价值、能源与防御板块。"}],
          book_lines(D, "investors", None, "投资者账户（Z.1、H.6、FINRA）")
          + book_lines(D, "households", ["equity_share_fin_assets", "equity_share_pctile_since_1990"], "居民的股票配置"))

    actor("banks", "银行/放贷机构", "美国商业银行与其他放贷机构（作为整体）。决定贷款标准（SLOOS）与信贷投放；关注资产质量、"
          "存款成本与证券浮亏。",
          ["资产质量：拖欠与核销", "净息差与存款成本", "资本与监管要求", "证券组合的利率风险", "贷款需求"],
          [{"id": "standards_unchanged", "label": "标准不变", "description": "贷款标准不变（现状）。"},
           {"id": "ease", "label": "放松", "description": "放松贷款标准，信贷扩张。"},
           {"id": "tighten", "label": "收紧", "description": "收紧贷款标准（SLOOS 净收紧 10–30%）。"},
           {"id": "tighten_hard", "label": "大幅收紧", "description": "大幅收紧（净收紧 30% 以上），信贷紧缩。"}],
          book_lines(D, "banks", None, "银行账户（FDIC、H.8）"))

    actor("firms", "非科技企业", "美国的非科技企业（制造、服务、零售、能源等，作为整体）。决定投资、招聘与定价；"
          "面对高利率、关税成本与油价冲击。",
          ["利润率与成本（能源、关税、工资）", "需求前景", "融资成本与信用条件", "定价能力与市场份额"],
          [{"id": "business_as_planned", "label": "按计划经营", "description": "投资与招聘按计划（现状）。"},
           {"id": "expand", "label": "扩张", "description": "增加资本开支与招聘。"},
           {"id": "slow_down", "label": "放缓", "description": "推迟投资、放慢招聘。"},
           {"id": "contract", "label": "收缩", "description": "削减投资并裁员。"},
           {"id": "price_pass_through", "label": "定价：提价转嫁", "description": "把关税与能源成本转嫁给消费者。"},
           {"id": "price_cut_share", "label": "定价：让价保份额", "description": "压缩利润率以保住市场份额。"}],
          book_lines(D, "nfc", None, "非金融企业账户（Z.1、NIPA）"))
    return A


SYNTH_INSTRUCTIONS = """你是「美国与世界」的世界合成器。每一步代表一个季度：给定季初状态与所有行为者在本季的行动，写出下一季季初的状态。

1. 联合求解：把所有行动放在一起考虑它们的相互作用（例如美联储加息与财政扩张同时发生、美国加征关税与中国报复同时发生、冲突升级与欧佩克+ 增产同时发生），得到一个前后一致的结果，而不是逐条叠加。
2. 一致性：每个字段都要与其他字段、与行动一致。美联储加息推高短端利率（政策利率按行动的幅度变动，2 年期收益率通常同向）；油价变动影响 CPI 同比与汽油价格；关税行动改变有效关税率；选举结果改变「两院与白宫的多数党」与「众议院控制权」；债务上限的提高或暂停改变「债务上限状态」与余量；字段之间的会计关系（赤字与债务、利率与利息负担）要说得通。
3. 只改变本季事件确实会移动的字段；没有理由变动的字段保持原值，reason 留空。慢变量（劳动收入占比、利润/GDP、债务/GDP、IT 投资/GDP）一个季度只会小幅移动。
4. 每个变动的字段都给出简短理由，指明是哪个行动或机制。
5. 合理性：数值放在历史区间内的合理位置；若数值超出其历史区间或合理区间，必须在理由里说明为什么。
6. 定性字段（枚举）只能取给定选项之一；标注“未确认”的事实，在被行动或事件改变之前保持原样。
7. 关键问题要诚实更新：已经有答案的标记为 answered 并写出答案；仍未定的保持 open 并更新当前判断；出现新的关键问题就加入；不要假装确定。
8. 属性反映本季之后的制度与格局。
9. 这是世界模拟，不做资产定价：不要写股票指数点位、个股价格、金价、估值倍数等资产价格水平；schema 里已有的宏观字段（收益率、利差、美元指数、油价、股息率）照常更新。
10. 叙事写成新季度季初的局势概述（3–6 段）：各方做了什么、结果如何、还有哪些悬念。没有回应的行为者视为维持现状。"""


def build_world(D: Data) -> dict:
    fields, values = build_schema(D)
    text = build_text(D, values)
    actors = build_actors(D, values)
    return {
        "format": "worldsim-world/1", "id": "us_2026q4", "name": "美国与世界 · 2026Q4",
        "description": ("美国与世界政治经济的季度推演，起点是 2026 年第四季度季初。数值来自 Causeway 的本地数据（面板截至 2026-09；"
                        "部分账户读数截至 10 月底，见各字段说明），制度现状来自 institutions.json（多数未确认）。"
                        "每一步一个季度：14 个行为者各自行动，合成器写出下一季季初的状态。只做世界模拟，不做资产定价。"),
        "language": "中文", "period_step": {"unit": "quarter", "count": 1}, "history_window": 3,
        "model": {"model": "", "temperature": None, "reasoning_effort": ""},
        "schema": {"fields": fields}, "actors": actors,
        "synthesizer": {"instructions": SYNTH_INSTRUCTIONS, "model": ""},
        "initial_state": {"step": 0, "period": "2026Q4", "values": values, "text": text,
                          "summary": "起点：2026Q4 季初（数据截至 2026-09）"},
        "created": "", "updated": "",
        "source": {"built_by": "tools/build_worldsim_default.py", "data_end": "2026-09",
                   "accounts": D.acct.get("version"), "catalogue": catalogue_version()},
    }


def catalogue_version() -> str:
    from scenariolab.agents import actors as A
    return A.CATALOGUE_VERSION.split(" (")[0]


def check_user_numbers(values: dict) -> list[str]:
    quoted = {"tips10": 2.93, "debt_gdp": 97.37, "consumer_sentiment": 48.1, "initial_claims": 200.5,
              "labor_share": 93.4, "profit_share": 13.14, "dividend_yield": 1.03, "it_invest_gdp": 5.05}
    out = []
    for k, q in quoted.items():
        got = values.get(k)
        tol = 0.051 if k in ("consumer_sentiment", "initial_claims", "labor_share") else 0.0051
        out.append(f"{k}: data {got} vs quoted {q} -> {'match' if got is not None and abs(got - q) <= tol else 'DIFFERENT'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with the committed file")
    args = ap.parse_args(argv)
    D = Data()
    world = build_world(D)
    from worldsim import model as M
    norm, problems = M.normalize_world(world)
    if problems:
        print("the built world does not validate:\n  " + "\n  ".join(problems))
        return 1
    text = json.dumps(world, ensure_ascii=False, indent=1) + "\n"
    for line in check_user_numbers(world["initial_state"]["values"]):
        print(line)
    print(f"{len(world['schema']['fields'])} fields, {len(world['actors'])} actors: "
          + ", ".join(a["name"] for a in world["actors"]))
    if args.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print("committed file is up to date" if same else "committed file differs from a fresh build")
        return 0 if same else 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.encode()):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
