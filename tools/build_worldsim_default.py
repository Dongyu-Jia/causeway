"""Build WorldSim's default world 「美国与世界 · 2026Q4」 from Causeway's local data.

WorldSim lives in its own repository (https://github.com/Dongyu-Jia/worldsim) and ships the built file as
worldsim/defaults/us_2026q4.json. This script writes that file into a WorldSim checkout, and uses the checkout's
worldsim.model to validate the world:

    git clone https://github.com/Dongyu-Jia/worldsim.git ../worldsim
    python tools/build_worldsim_default.py                      # writes ../worldsim/worldsim/defaults/us_2026q4.json
    python tools/build_worldsim_default.py --check              # rebuilds in memory and compares, writes nothing
    python tools/build_worldsim_default.py --worldsim PATH      # another checkout (or set WORLDSIM_REPO)

Inputs (all local; data/ is git-ignored and never published):
- data/processed/panel_monthly.csv, panel_quarterly.csv  today's values (2026-09 / start of 2026Q4) and the history
  each field's range note and plausible bounds come from
- data/raw/fred/*.json                                    payrolls, housing starts, mortgage rate, breakeven, wages,
                                                          the Fed's balance sheet (series the panels do not carry)
- data/results/agents_accounts_root.json                  the actors' books (balance sheets, income, cash flow): each
                                                          actor's private information, and fields such as the EFFR
- data/agents/institutions.json                           the pinned facts (Fed, IEEPA, oil, midterms, China truce,
                                                          OPEC+): every one still unconfirmed is tagged 未确认
- scenariolab/qmodel/card.json                            the quarterly model card's definitions
- data/raw/shiller/ie_data.xls, data/raw/gold/monthly.csv the S&P composite and gold prices; FRED NASDAQ100, VIXCLS,
  data/raw/fred/{NASDAQ100_eop.csv, VIXCLS, CSUSHPISA,    CSUSHPISA and HPIPONM226S the other asset prices
  HPIPONM226S}.json
- data/results/instrument_meta.json                       QQQ and TLT start prices (Causeway's estimates, tagged 估算)
- scenariolab/agents/actors.py                           the actors' levers, which their decision questions follow

Written here: the world description, the 22 predefined state questions with today's status (grounded in the data and
the pinned facts, 未确认 where unconfirmed), the ~100-字 narrative, the actors' decision questions (guidance in words:
units, typical options and magnitudes) with their autonomy guidance, and the synthesizer's instructions.

Today = the start of 2026Q4: the panels end 2026-09. A few fields have no September reading and use the latest one
in the accounts (stated in their description, e.g. the EFFR of 2026-10-29). Asset prices are ordinary state fields
(no pricing model); a price with no local source is left out rather than invented.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
OUT_REL = Path("worldsim") / "defaults" / "us_2026q4.json"
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
        self.instr = json.loads((DATA / "results" / "instrument_meta.json").read_text(encoding="utf-8"))
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

    # Robert Shiller's monthly S&P composite (the file Causeway's panel reads its CAPE from)
    @staticmethod
    def shiller_price() -> pd.Series:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_excel(DATA / "raw" / "shiller" / "ie_data.xls", sheet_name="Data", header=7)
        df = df[pd.to_numeric(df["Date"], errors="coerce").notna() & pd.to_numeric(df["P"], errors="coerce").notna()]
        idx = [pd.Timestamp(f"{float(d):.2f}".replace(".", "-") + "-01") for d in df["Date"]]
        return pd.Series(df["P"].astype(float).values, index=idx).sort_index()

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
    G_ASSET = "资产价格"
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

    # --- asset prices: ordinary state fields (no pricing model); the synthesizer moves them with a reason
    spx = D.shiller_price()
    add(num_field(D, "spx", "标普 500 指数", G_ASSET, "number", "点", float(spx.loc[TODAY_M]), "2026-09 月均",
                  "S&P 综合指数的月均收盘点位（Robert Shiller 的 data/raw/shiller/ie_data.xls，Causeway 面板的 CAPE 也取自此文件）；"
                  f"席勒 CAPE {D.pm_today('s_cape')[0]:.1f}；文件里最新的 10 月（可能不是完整月份）为 {spx.iloc[-1]:,.0f}", spx, d=1))
    ndx = D.fred("NASDAQ100_eop.csv")
    add(num_field(D, "ndx", "纳斯达克 100 指数", G_ASSET, "number", "点", float(ndx.loc[TODAY_M]), "2026-09 月末",
                  f"FRED NASDAQ100（月末收盘，data/raw/fred/NASDAQ100_eop.csv）；10 月末 {ndx.iloc[-1]:,.0f}", ndx, d=1))
    qqq = D.instr["start"]["QQQ"]
    ratio = qqq["price"] / float(ndx.loc[TODAY_M])
    f_qqq, v_qqq = num_field(D, "qqq", "QQQ（纳斯达克 100 ETF）", G_ASSET, "number", "美元", qqq["price"], qqq["asof"],
                             "Causeway 的起点价格（data/results/instrument_meta.json）：2026-06-30 的 13F 报告价格按纳斯达克 100 的"
                             f"价格回报滚动到 9 月底（不是成交价，估算）；约为指数的 {ratio:.4f} 倍，与指数同步变动。本地没有 QQQ 的"
                             "价格历史：历史区间与合理区间按纳斯达克 100 换算", ndx * ratio, d=2, tag="估算")
    f_qqq["hist"] = "按纳斯达克 100 换算（QQQ 1999 年才上市）：" + f_qqq["hist"]
    add((f_qqq, v_qqq))
    gold = D.pm_series("lvl_gold")
    add(num_field(D, "gold", "黄金价格", G_ASSET, "number", "美元/盎司", float(gold.iloc[-1]),
                  gold.index[-1].strftime("%Y-%m") + " 月均",
                  "伦敦金价的月均值（data/raw/gold/monthly.csv，Causeway 面板 lvl_gold）；本地没有 10 月读数", gold, d=0))
    tlt = D.instr["start"]["TLT"]
    add(num_field(D, "tlt", "TLT（20 年以上美国国债 ETF）", G_ASSET, "number", "美元", tlt["price"], tlt["asof"],
                  "Causeway 的起点价格（data/results/instrument_meta.json）：2026-06-30 的 13F 报告价格按久期模型与 20/30 年期收益率"
                  "的变动滚动到 9 月底（不是成交价，估算）；与长端收益率反向", None, lo=30.0, hi=200.0, tag="估算"))
    vix = D.fred("VIXCLS.json")
    vix_sep = vix.loc[:"2026-09-30"]
    add(num_field(D, "vix", "VIX 波动率指数", G_ASSET, "number", "指数", float(vix_sep.iloc[-1]),
                  vix_sep.index[-1].strftime("%Y-%m-%d"),
                  f"CBOE VIX（FRED VIXCLS，日度收盘）；最新 {vix.iloc[-1]:.1f}（{vix.index[-1]:%Y-%m-%d}）", vix, d=1))
    cs = D.fred("CSUSHPISA.json")
    add(num_field(D, "case_shiller", "Case-Shiller 全国房价指数", G_ASSET, "number", "指数（2000-01=100）",
                  float(cs.iloc[-1]), cs.index[-1].strftime("%Y-%m"),
                  f"S&P Cotality Case-Shiller 美国全国房价指数，季调（FRED CSUSHPISA；发布滞后约两个月，最新为 8 月）；同比 "
                  f"{(cs.iloc[-1] / cs.iloc[-13] - 1) * 100:+.1f}%。成屋销售中位价（NAR）本地没有，未收录", cs, d=1))
    fh = D.fred("HPIPONM226S.json")
    add(num_field(D, "fhfa_hpi", "FHFA 房价指数（仅购买）", G_ASSET, "number", "指数（1991-01=100）", float(fh.iloc[-1]),
                  fh.index[-1].strftime("%Y-%m"),
                  f"FHFA 美国仅购买房价指数，季调（FRED HPIPONM226S，最新为 8 月）；同比 "
                  f"{(fh.iloc[-1] / fh.iloc[-13] - 1) * 100:+.1f}%", fh, d=1))

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


# ------------------------------------------------------------------------------------------------ the state's text

WORLD_DESCRIPTION = """「美国与世界 · 2026Q4」是一个以美国为中心的政治经济季度推演世界。起点是 2026 年第四季度季初（本地数据截至 2026 年 9 月，部分读数到 10 月底），每一步推进一个季度。

范围：美国的货币政策、财政与债务上限、贸易与关税、法院、中期选举与国内政治，劳动力市场、通胀、消费者、企业盈利与 AI 投资周期、银行与金融条件、房地产与主要资产价格；外部世界包括中国、欧佩克+ 与 2026 年导致石油供给中断的武装冲突。

行为者共 14 个：美联储、总统与白宫、国会、财政部、法院、选民、中国、欧佩克+、冲突方、超大型科技公司、居民、华尔街投资者、银行与非科技企业。它们每季回答各自的决策问题，并可采取自主行动；合成器把所有决策放在一起，写出下一季季初的状态。

状态由三部分组成：指标数值（宏观、金融与资产价格，来自 Causeway 的本地数据，出处见各字段说明）；预设状态问题的现状描述（制度事实多数尚未确认，标注“未确认”）；以及一段约 100 字的总结。资产价格是普通的状态字段，由合成器随事件一致地更新，没有定价模型。"""


def build_state_text(D: Data, v: dict) -> tuple[list[dict], dict, str]:
    """The predefined state questions (id, text, group, guidance), each one's status today, and the ~100-字
    narrative of the initial state."""
    b = D.val
    G_MON, G_FIS, G_EXT, G_ENE, G_ECO, G_MKT = ("货币与金融", "财政与国内政治", "贸易与地缘政治", "能源与冲突",
                                                "经济与企业", "市场与资产")
    ebp = D.pm_today("s_ebp")[0]
    oil_yoy = D.pm_today("s_oil_yoy")[0]
    usd_yoy = D.pm_today("s_usd_yoy")[0]
    cape = D.pm_today("s_cape")[0]
    tariff_q2 = D.pm_today("s_tariff_rate")[0]
    mort_oct = D.fred("MORTGAGE30US_m_eop.json").iloc[-1]
    wti_oct = D.fred("DCOILWTICO_eop.csv").iloc[-1]
    ndx_oct = D.fred("NASDAQ100_eop.csv").iloc[-1]
    rows = [
        ("fed_policy", "美联储的政策立场与独立性如何？", G_MON,
         "利率路径的倾向（偏鹰 / 中性 / 偏鸽）、最近一次会议、资产负债表、人事与独立性受到的压力",
         f"9 月加息 25bp 后，有效联邦基金利率约 {v['policy_rate']:.2f}%（目标区间 3.75–4.00%），10 月 27–28 日的会议按数据推断为维持"
         f"（未确认）。核心 PCE {v['core_pce_yoy']:.1f}%，比 2% 的目标高约 {b('fed', 'inflation_gap'):.1f} 个百分点，立场为偏鹰观望。"
         f"资产负债表约 {v['fed_assets'] / 1000:.2f} 万亿美元，近 13 周增持国库券约 {b('fed', 'soma_bills_change_13w'):.0f} 十亿美元"
         "（准备金管理购买）。主席与理事会人事按数据日期的现任（未确认），独立性在形式上维持。"),
        ("inflation", "通胀的成因与动态如何？", G_MON,
         "能源、关税、工资与预期各自的贡献，通胀在上行、见顶还是回落",
         f"CPI 同比 {v['cpi_yoy']:.1f}%，主要由能源推动（WTI 同比 {oil_yoy:+.0f}%）；核心 CPI {v['core_cpi_yoy']:.1f}%、"
         f"核心 PCE {v['core_pce_yoy']:.1f}%。1 年期通胀预期 {v['infl_exp_1y']:.1f}% 偏高，长期预期 {v['infl_exp_lr']:.1f}%，"
         f"10 年盈亏平衡通胀率 {v['breakeven10']:.2f}%。时薪同比 {v['wage_growth']:.1f}%，工资没有加速，关税的价格传导仍在进行（判断）。"),
        ("financial_conditions", "金融条件与信用状况如何？", G_MON,
         "长端利率与期限溢价、信用利差、风险偏好、杠杆",
         f"10 年期国债收益率 9 月底 {v['y10']:.2f}%（10 月底 {b('macro', 'y10'):.2f}%），期限溢价 {v['term_premium']:.2f} 个百分点，"
         f"实际利率（10 年 TIPS）{v['tips10']:.2f}%。信用市场仍宽松：Baa 利差 {v['baa_spread']:.2f}、GZ 利差 {v['gz_spread']:.2f} 个百分点，"
         f"超额债券溢价 {ebp:.2f}（低于 0 表示风险偏好偏高）。保证金负债 {b('investors', 'margin_debt') / 1000:.2f} 万亿美元，"
         f"同比 {b('investors', 'margin_debt_yoy'):+.0f}%。"),
        ("banks", "银行体系是否稳健？", G_MON,
         "资产质量、存款与证券浮亏、放贷意愿、商业地产敞口",
         f"银行贷款 {b('banks', 'bank_loans') / 1000:.1f} 万亿美元、存款 {b('banks', 'bank_deposits') / 1000:.1f} 万亿美元（H.8，10 月）；"
         f"贷款拖欠率 {b('banks', 'bank_delinquency'):.2f}%、核销率 {b('banks', 'bank_chargeoff'):.2f}%（2026Q2），资产质量平稳。"
         f"放贷标准净收紧比例 {v['lending_std']:.0f}%，大体不变；高利率下证券组合的浮亏仍是压力点（判断）。"),
        ("fiscal", "财政立场与赤字路径如何？", G_FIS,
         "赤字规模与方向、利息负担、新的财政立法",
         f"2026 财年赤字 {b('fiscal', 'deficit_fy') / 1000:.2f} 万亿美元（收入 {b('fiscal', 'receipts_fy') / 1000:.2f}、支出 "
         f"{b('fiscal', 'outlays_fy') / 1000:.2f} 万亿），赤字/GDP 约 {v['deficit_gdp']:.1f}%（NIPA），公众持有债务/GDP {v['debt_gdp']:.1f}%。"
         f"净利息 {b('fiscal', 'net_interest_fy'):.0f} 十亿美元，占联邦收入 {v['net_interest_share']:.1f}%，比上年多 "
         f"{b('fiscal', 'net_interest_fy') - b('fiscal', 'net_interest_prior_fy'):.0f} 十亿。财政仍在扩张，尚无新的整顿或刺激立法。"),
        ("debt_limit", "债务上限与 X 日期的局势如何？", G_FIS,
         "余量、触及时间、非常规措施、国会的处理方式",
         f"距法定上限约 {v['debt_limit_headroom']:.0f} 十亿美元，按近 3 个月的速度约 {b('fiscal', 'months_to_limit_at_3m_pace'):.1f} "
         f"个月触及。X 日期未锁定，非常规措施容量（判断值 {D.inst['em_capacity_bn']['value']:.0f} 十亿美元）未确认；"
         f"TGA 现金约 {b('fiscal', 'tga_dts'):.0f} 十亿美元。国会尚无提高或暂停上限的方案。"),
        ("politics", "中期选举与国内政治格局如何？", G_FIS,
         "选举结果（揭晓前后）、两院与白宫的格局、支持率与政治动态",
         "11 月 3 日中期选举的结果尚未揭晓。选举前白宫与参议院（53–47）由共和党控制，众议院为共和党微弱多数（未确认）；"
         "没有市场赔率时按历史基率，执政党在中期选举中通常失去众议院席位。新国会 2027-01-03 就任，选举后的跛脚鸭会期会影响财政谈判。"
         "总统支持率本地没有数据（未确认）。"),
        ("courts", "关键诉讼（IEEPA 关税、美联储理事免职、反垄断）进展如何？", G_FIS,
         "哪些案件待决、何时可能裁决、裁决的影响",
         f"IEEPA 关税按仍然有效处理（未确认）；2026 年 5–9 月关税退款合计 {b('fiscal', 'customs_refunds_since_2026_05'):.0f} 十亿美元，"
         "法律原因未确认。美联储理事免职案与大型科技公司反垄断救济本季没有已知的到期裁决（未确认）。"),
        ("tariffs", "关税制度与其法律状态如何？", G_EXT,
         "有效关税率、主要关税的授权依据、豁免与退款、对物价和贸易的影响",
         f"有效关税率三季度为 {v['tariff_rate']:.2f}%（NIPA；二季度 {tariff_q2:.2f}%），回落部分来自关税退款。关税收入年化约 "
         f"{b('fiscal', 'customs_nipa_saar'):.0f} 十亿美元，商品进口年化 {b('fiscal', 'goods_imports_saar') / 1000:.2f} 万亿。"
         "IEEPA 关税按有效处理（未确认），122、232、301 条款的关税照常执行。"),
        ("us_china", "中美关系（贸易休战、出口管制、台海）如何？", G_EXT,
         "休战到期前后的走向、芯片与稀土管制、汇率、台海局势",
         f"一年期贸易休战约 2026-11-10 到期（未确认），尚无延长或扩大的消息。对华芯片出口管制与中方稀土管制按数据日期的现状"
         f"（未确认）。美国自华进口近 12 个月 {b('china', 'us_imports_from_china_12m'):.0f} 十亿美元、对华出口 "
         f"{b('china', 'us_exports_to_china_12m'):.0f} 十亿；人民币兑美元 {v['cny_usd']:.2f}，中国持有美债 "
         f"{b('china', 'us_treasury_holdings'):.0f} 十亿美元（一年减少 {-b('china', 'us_treasury_holdings_change_12m'):.0f}）。"
         "台海维持紧张中的常态（判断）。"),
        ("global_growth", "欧洲与全球增长如何？", G_EXT,
         "主要经济体的增长与衰退风险、能源冲击的外溢",
         f"本地数据没有欧元区与全球增长的最新读数（未确认）。IMF WEO 预测中国 2026 年增长 {b('china', 'real_gdp_growth_2026'):.1f}%"
         f"（2025 年 {b('china', 'real_gdp_growth_2025'):.1f}%）；受冲突影响的海湾产油国收缩：沙特 {b('opec', 'sau_real_gdp_growth_2026'):.1f}%、"
         f"阿联酋 {b('opec', 'are_real_gdp_growth_2026'):.1f}%、科威特 {b('opec', 'kwt_real_gdp_growth_2026'):.1f}%、伊拉克 "
         f"{b('opec', 'irq_real_gdp_growth_2026'):.1f}%，俄罗斯 {b('opec', 'rus_real_gdp_growth_2026'):.1f}%。欧洲作为能源净进口方承受油价冲击（判断）。"),
        ("dollar_flows", "美元与资本流动如何？", G_EXT,
         "美元强弱、外国对美国资产的需求、经常账户",
         f"广义贸易加权美元指数 {v['usd_index']:.1f}，同比 {usd_yoy:+.1f}%，大体持平。外国持有美国国债 "
         f"{b('fiscal', 'foreign_holdings_tic') / 1000:.2f} 万亿美元（其中官方 {b('fiscal', 'foreign_official_holdings') / 1000:.2f} 万亿，8 月）；"
         f"经常账户逆差约占 GDP {-b('row', 'current_account_gdp'):.1f}%，国际投资净头寸 {b('row', 'niip') / 1000:.1f} 万亿美元。"),
        ("oil_conflict", "石油供给中断与冲突的态势如何？", G_ENE,
         "冲突的走向、受影响的设施与航道、中断规模、调停与外交",
         f"2026 年武装冲突导致的石油供给中断仍在持续（未确认），冲突方身份未锁定。WTI 9 月底 {v['wti']:.0f} 美元/桶、10 月底约 "
         f"{wti_oct:.0f}，布伦特 10 月 27 日 {b('opec', 'brent_spot'):.0f} 美元；IMF 预计科威特与伊拉克 2026 年经济分别收缩 "
         f"{-b('opec', 'kwt_real_gdp_growth_2026'):.1f}% 与 {-b('opec', 'irq_real_gdp_growth_2026'):.1f}%。地缘政治风险指数 {v['gpr']:.0f}。"),
        ("opec", "欧佩克+ 的产量政策如何？", G_ENE,
         "配额、闲置产能、联盟纪律、与美国页岩油的竞争",
         f"按最近一次公开声明维持配额（未确认）；可达市场的闲置产能未锁定（EIA STEO），配额数值本地没有。沙特外汇储备 "
         f"{b('opec', 'saudi_reserves_ex_gold'):.0f} 十亿美元，IMF 预测其 2026 年财政赤字约 GDP 的 {-b('opec', 'sau_gg_net_lending_gdp_2026'):.1f}%。"
         f"美国原油产量 {b('opec', 'us_crude_production'):.2f} 百万桶/日，一年增加 {b('opec', 'us_crude_production_change_52w'):.2f}。"),
        ("energy_supply", "美国的能源供应与油价负担如何？", G_ENE,
         "汽油价格、战略石油储备、国内产量对冲击的缓冲",
         f"汽油均价 {v['gasoline']:.2f} 美元/加仑（10 月下旬）。战略石油储备 {v['spr']:.0f} 百万桶，比一年前少约 "
         f"{-b('president', 'spr_change_52w'):.0f} 百万桶，进一步释放的空间有限（判断）。国内产量接近历史高位，部分缓冲了进口中断。"),
        ("labor", "劳动力市场的状况如何？", G_ECO,
         "就业增长、失业率、裁员与招聘、工资",
         f"失业率 {v['unemployment']:.1f}%（CBO 自然失业率约 {b('fed', 'nrou_cbo'):.1f}%），近三个月非农就业月均新增约 "
         f"{v['payrolls_3m']:.0f} 千人，首次申领约 {v['initial_claims']:.0f} 千人/周：裁员仍少但招聘明显放缓。时薪同比 {v['wage_growth']:.1f}%，"
         f"劳动收入占比指数 {v['labor_share']:.1f}，处在低位。"),
        ("consumers", "消费者与居民的状况如何？", G_ECO,
         "信心、收入与储蓄、债务负担、财富分布",
         f"消费者信心 {v['consumer_sentiment']:.1f}，处在历史低位；人均实际可支配收入同比只有 {b('president', 'real_dpi_pc_yoy'):.1f}%，"
         f"储蓄率 {v['saving_rate']:.1f}% 偏低，偿债负担 {v['hh_debt_service']:.1f}% 可支配收入。财富高度集中：最富 1% 持有约 "
         f"{b('households', 'nw_share_top1'):.1f}% 的净资产，后 50% 只有 {b('households', 'nw_share_bottom50'):.1f}%；消费主要靠高收入家庭支撑（判断）。"),
        ("growth", "美国经济增长的势头与衰退风险如何？", G_ECO,
         "增长的来源、放缓的迹象、衰退风险",
         f"三季度实际 GDP 年化增长 {v['real_gdp_growth']:.1f}%，AI 相关投资（IT 投资/GDP {v['it_invest_gdp']:.2f}%）与高收入家庭消费是主要支撑；"
         f"就业放缓、信心低迷与高油价是拖累。经济政策不确定性指数 {v['epu']:.0f}，处在高位。短期衰退风险上升但尚未显现（判断）。"),
        ("ai_capex", "AI 资本开支周期处在什么阶段？", G_ECO,
         "超大型科技公司的投资、融资方式、回报与电力等约束",
         f"五家超大型科技公司近四季资本开支 {v['hyperscaler_capex_ttm']:.0f} 十亿美元（一年前 {b('bigtech', 'capex_ttm_prev'):.0f}），"
         f"三季度单季 {b('bigtech', 'capex'):.0f}；自由现金流近四季只有 {b('bigtech', 'fcf_ttm'):.0f} 十亿，近四季新增债务 "
         f"{b('bigtech', 'debt_change_4q'):.0f} 十亿，收入同比 {b('bigtech', 'revenue_growth_4q'):.0f}%。2026 年计划合计约 "
         f"{D.inst['hyperscaler_capex_2026_bn']['value']:.0f} 十亿美元（二手来源，未确认）。周期仍在加速，但越来越依赖举债。"),
        ("profits", "企业盈利与利润率如何？", G_ECO,
         "利润水平与分化、成本压力（关税、能源、利息）、定价能力",
         f"企业税后利润/GDP {v['profit_share']:.1f}%，处在历史高位；非金融企业利润率（税后利润/增加值）{b('nfc', 'profit_margin_gva'):.1f}%，"
         f"利息覆盖倍数 {b('nfc', 'interest_coverage'):.1f}。利润高度集中在大型科技公司，非科技企业面临关税、能源与利息成本的挤压（判断）。"),
        ("housing", "房地产市场的状况如何？", G_ECO,
         "房贷利率、成交与开工、房价与可负担性",
         f"30 年期房贷利率 9 月底 {v['mortgage30']:.2f}%（10 月底 {mort_oct:.2f}%），新屋开工年化 {v['housing_starts']:.0f} 千套；"
         f"Case-Shiller 全国房价指数 8 月 {v['case_shiller']:.1f}，房价同比约 {v['hpi_yoy']:.1f}%（Freddie Mac），涨幅放缓。"
         "高利率下成交低迷，可负担性处在多年低位（判断）。"),
        ("markets", "资产市场的状况与估值如何？", G_MKT,
         "股市水平与估值、科技股集中度、黄金、长债与波动率、风险偏好",
         f"标普 500 9 月月均 {v['spx']:,.0f} 点，纳斯达克 100 9 月底 {v['ndx']:,.0f} 点（10 月底 {ndx_oct:,.0f}）；席勒 CAPE 约 {cape:.1f}，"
         f"股息率 {v['dividend_yield']:.2f}%，估值处在历史高位，居民的股票占金融资产比重处在 1990 年以来的第 "
         f"{b('households', 'equity_share_pctile_since_1990'):.0f} 百分位。黄金 9 月均价约 {v['gold']:,.0f} 美元/盎司，"
         f"TLT 约 {v['tlt']:.0f} 美元（长端收益率高），VIX 9 月底 {v['vix']:.1f}。"),
    ]
    questions = [{"id": i, "text": t, "group": g, "guidance": gd} for i, t, g, gd, _ in rows]
    statuses = {i: st for i, _, _, _, st in rows}
    narrative = (f"2026Q4 季初：通胀 {v['cpi_yoy']:.1f}% 偏高、就业放缓（失业率 {v['unemployment']:.1f}%），美联储加息后观望；"
                 f"冲突使油价约 {v['wti']:.0f} 美元，财政扩张、债务上限临近；AI 投资高涨、股市估值处于高位；"
                 "中期选举与中美休战到期悬而未决。")
    return questions, statuses, narrative



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


def Q(qid: str, text: str, guidance: str, required: bool = True) -> dict:
    return {"id": qid, "text": text, "guidance": guidance, "required": required}


def build_actors(D: Data, v: dict) -> list[dict]:
    """The fourteen actors. Their decision questions follow the levers in Causeway's actor catalogue
    (scenariolab/agents/actors.py); the guidance carries the constraints in words
    (units, typical options and magnitudes)."""
    A = []

    def actor(aid, name, role, cares, questions, autonomy, private, max_actions=3):
        A.append({"id": aid, "name": name, "enabled": True, "role": role, "cares_about": cares,
                  "decision_questions": questions,
                  "autonomy": {"enabled": True, "max_actions": max_actions, "guidance": autonomy},
                  "private_info": "\n".join(private), "visible_fields": None, "model": ""})

    actor("fed", "美联储", "美国联邦储备体系的联邦公开市场委员会（作为机构，不模仿任何个人的语气）。本季有两次议息会议，"
          "决定联邦基金利率目标区间与资产负债表安排；在通胀高于目标、就业放缓、油价冲击与财政扩张同时出现时权衡双重使命。",
          ["把通胀（核心 PCE）拉回 2% 的目标并稳住通胀预期", "避免不必要的失业上升（就业缺口接近 0）",
           "金融稳定：信用利差、银行与国债市场的运转", "维护政策独立性与可信度", "资产负债表与准备金的平稳运行"],
          [Q("rate", "本季政策利率怎么调、调多少？",
             "本季有两次议息会议。以 25bp 为单位给出本季累计变动（维持、加息 25/50bp、降息 25/50bp，通常在 −100bp 到 +100bp "
             "之间），说明每次会议的安排和新的目标区间。"),
           Q("balance_sheet", "资产负债表怎么安排？",
             "在三者中选择并说明节奏：继续准备金管理购买（增持国库券，给出每月规模，十亿美元）、维持规模、恢复缩表（给出每月滚落上限）。"),
           Q("guidance", "前瞻指引的基调是什么？",
             "偏鹰、中性或偏鸽，用一两句话说明对之后几次会议的暗示，以及改变路径的条件（通胀、就业、金融条件的门槛）。")],
          "例如：紧急流动性工具或回购便利、监管与压力测试的表态、与财政部或外国央行的协调（美元互换额度）、对市场的特别沟通。"
          "不要重复上面问题已经决定的利率与资产负债表安排。",
          book_lines(D, "fed", None, "美联储账户（H.4.1 与数据）") + inst_lines(D, ("fed",)))

    actor("president", "总统与白宫", "美国总统与白宫（作为机构，不模仿任何人的语气）。掌握贸易与关税（122、301、232 条款与 IEEPA 授权）、"
          "能源（战略石油储备）、移民执法与对外谈判，也会对美联储公开施压；关注中期选举前后的政治处境。",
          ["选民的生活成本感受：汽油价格与通胀", "中期选举与执政党的国会席位", "贸易逆差、产业回流与对华谈判筹码",
           "股市与就业的表现", "关税的法律授权与收入"],
          [Q("tariffs", "本季关税政策怎么调整？",
             "维持、对特定行业或国家加征、广泛加征、降低或扩大豁免、与伙伴达成协议之一或组合；尽量给出对象、税率（%）、生效时间与所依据的"
             "授权（IEEPA、122、232、301 条款）。广泛加征约使有效关税率上升 2 个百分点，历史上罕见。"),
           Q("china", "对中国的谈判与出口管制立场是什么？",
             "中美一年期休战约 2026-11-10 到期（未确认）：延长、扩大成更广泛的协议、放任到期，还是升级？芯片与稀土相关的管制是放松、维持还是收紧？"),
           Q("spr", "是否动用战略石油储备、动用多少？",
             f"不动用，或给出释放量（百万桶）与节奏；库存见状态，比一年前已少约 {-D.val('president', 'spr_change_52w'):.0f} 百万桶。"),
           Q("fed", "对美联储采取什么姿态？",
             "不评论、温和表态、公开施压要求降息，或人事行动（提名、试图罢免理事）；说明力度。", required=False)],
          "例如：行政令、移民执法、对外访问与协议、制裁、能源许可、对国会的立法施压、选举相关的政治动作。",
          book_lines(D, "president", None, "白宫关注的指标") + inst_lines(D, ("president",)))

    actor("congress", "国会", "美国国会两院（作为机构）。决定最终成为法律的财政立法：拨款与续命、税收、转移支付，以及债务上限的提高、暂停或对峙；"
          "11 月 3 日中期选举后，新国会 2027-01-03 就任，跛脚鸭会期内的谈判筹码会变化。",
          ["避免债务违约与政府停摆的政治代价", "选区选民的减税与支出诉求", "赤字与利息负担的可持续性",
           "选举结果带来的权力格局", "关税退款与财政收入"],
          [Q("fiscal", "本季财政立法的方向与规模是什么？",
             "无新立法（只有自动稳定器）、拨款续命、小规模刺激（退税或直接支付，约 0.8% GDP）、减税、财政整顿（削减支出或增税）之一或组合；"
             "给出规模（十亿美元或占 GDP 百分比）与生效时间。"),
           Q("debt_limit", "债务上限怎么处理？",
             "提高（给出新上限或增加额，万亿美元）、暂停（到何时）、对峙未解决，或本季不处理；参考状态中的债务上限余量与触及时间。"),
           Q("shutdown", "拨款到期时是否发生政府停摆？",
             "不停摆（续命或全年拨款）、短暂停摆（给出天数）或长期停摆；说明到期日与各方的筹码。")],
          "例如：听证与调查、限制或确认总统关税授权的立法、对美联储人事的确认或立法、跛脚鸭会期的其他议案。",
          book_lines(D, "fiscal", ["deficit_fy", "receipts_fy", "outlays_fy", "net_interest_fy", "net_interest_share_receipts",
                                   "debt_held_public", "statutory_limit", "limit_headroom", "debt_pace_3m",
                                   "months_to_limit_at_3m_pace", "customs_refunds_since_2026_05", "deficit_nipa_gdp",
                                   "debt_gdp"], "联邦财政账户") + inst_lines(D, ("congress",)))

    actor("treasury", "财政部", "美国财政部（作为机构）。管理国债发行结构（季度再融资：短债占比与期限）、回购、现金余额（TGA），"
          "在触及债务上限时动用非常规措施，并向国会估计 X 日期。",
          ["按最低成本为赤字融资", "国债市场的流动性与平稳运行", "在债务上限约束下避免违约", "现金余额（TGA）的安全垫"],
          [Q("issuance", "本季发行结构怎么调整？",
             f"短期国库券占比提高、不变或降低（拉长期限），给出大致幅度（百分点，当前约 {D.val('fiscal', 'bill_share'):.0f}%）；季度再融资中中长期国债的发行规模是否改变。"),
           Q("debt_limit_ops", "如何应对债务上限？",
             "是否启动非常规措施、季末 TGA 目标余额（十亿美元）、向国会通报的 X 日期估计。"),
           Q("buybacks", "国债回购的规模怎么安排？", "不变、扩大或缩小，尽量给出每季规模（十亿美元）。", required=False)],
          "例如：与美联储或外国财政部的协调、制裁与外资审查（OFAC、CFIUS）、关税退款的执行、市场稳定措施。",
          book_lines(D, "fiscal", ["tga_dts", "debt_held_public", "bill_share", "mspd_bills", "mspd_notes", "mspd_bonds",
                                   "avg_rate_marketable", "avg_rate_marketable_12m_ago", "foreign_holdings_tic",
                                   "foreign_official_holdings", "limit_headroom", "months_to_limit_at_3m_pace"],
                     "财政部现金与债务管理") + inst_lines(D, ("treasury",)))

    actor("courts", "最高法院与法院", "美国最高法院与联邦法院（作为机构）。在有待决案件时作出裁决：IEEPA 关税授权、美联储理事的免职保护、"
          "大型科技公司的反垄断救济；没有到期的案件时本季不作裁决。只依据法律与先例，不考虑市场反应。",
          ["法律文本与先例", "行政权的边界（关税授权、独立机构的免职保护）", "裁决的可执行性（例如退款）", "司法机构的独立性"],
          [Q("ieepa", "本季是否就 IEEPA 关税授权作出裁决？结果是什么？",
             "不裁决（案件继续审理）、维持授权、裁定授权无效但不要求退税、裁定无效并要求退税（一次性财政事件）；若裁决，说明范围与生效时间。"),
           Q("fed_removal", "关于总统罢免美联储理事的权力，本季有无裁决？",
             "无裁决、维持免职保护，或允许罢免（独立性冲击）；说明所依据的先例。"),
           Q("antitrust", "对大型科技公司的反垄断案有无裁决或救济？",
             "无、行为救济，或结构性救济（拆分或剥离）；说明涉及的公司与业务。", required=False)],
          "例如：对行政命令的临时禁令或暂停、紧急裁定、程序性安排（加快审理、发回重审）。只依据法律，不因市场反应而行动。",
          inst_lines(D, ("courts",)), max_actions=2)

    actor("voters", "选民", "美国选民（作为整体）。本季在 11 月 3 日中期选举中投票，决定众议院（和部分参议院席位）的归属；"
          "他们的判断来自生活成本、就业与对执政党的观感。",
          ["生活成本：汽油、食品与房租", "就业与收入的安全感", "对执政党政策的评价", "移民、关税等议题"],
          [Q("midterm", "中期选举的结果如何？",
             "若本季包含 11 月 3 日的投票：给出众议院与参议院的归属和大致席位（例如“民主党以约 225 席对 210 席赢得众议院，共和党保住参议院"
             " 52 席”）；若结果已在之前的季度揭晓，写“已揭晓”并简述结果。"),
           Q("approval", "你们对总统与执政党的支持怎么变化？",
             "给出方向和大致幅度（百分点），以及主要原因（生活成本、就业、关税、移民等）。"),
           Q("priorities", "现在最在意的两三个议题是什么？参与政治的热情怎样？",
             "按重要性列出议题，并说明投票与参与的意愿是高、中还是低。", required=False)],
          "例如：抗议与集会、地方公投、对特定政策的舆论压力、消费者抵制。作为整体的选民，不扮演具体的人。",
          book_lines(D, "president", ["cpi_yoy", "gasoline_price", "consumer_sentiment", "real_dpi_pc_yoy",
                                      "us_equity_drawdown"], "选民感受到的经济")
          + book_lines(D, "households", ["saving_rate", "debt_service_ratio", "nw_share_top1", "nw_share_bottom50"],
                       "居民账户") + inst_lines(D, ("voter",)), max_actions=2)

    actor("china", "中国", "中国中央政府与人民银行（作为机构）。应对美国的关税与出口管制（对等关税、稀土与磁材管制、谈判），"
          "决定国内宏观刺激力度与人民币汇率，并在台海问题上表态。",
          ["经济增长目标与就业", "对美出口与产业链安全（芯片、稀土）", "人民币汇率稳定与资本流动", "房地产下行与地方债务",
           "台海与地缘政治"],
          [Q("trade", "对美国的关税与出口管制，本季如何回应？",
             "不回应、对等加征关税（税率与范围）、加征关税并收紧稀土等出口管制、谈判延长或扩大休战（约 2026-11-10 到期）；说明条件与时间。"),
           Q("cny", "人民币汇率如何管理？",
             "保持稳定、允许有管理的贬值或引导升值；给出季末的大致目标（元/美元，当前见状态）与所用工具（中间价、逆周期因子、外汇存款准备金）。"),
           Q("stimulus", "国内宏观刺激的力度如何？",
             "不加码、温和加码或大规模刺激；尽量给出规模（万亿元或占 GDP 百分比）与方向（财政、信贷、房地产、消费）。"),
           Q("taiwan", "台海方面本季的动作是什么？", "常态、加强军事存在或大规模军演；说明时间与要释放的信号。", required=False)],
          "例如：购买美国农产品或飞机、实体清单与反制裁、稀土出口许可的节奏、外交访问、调整美债持有。",
          book_lines(D, "china", None, "中国账户（IMF、TIC、BIS）") + inst_lines(D, ("china",)))

    actor("opec", "欧佩克+", "欧佩克+（以沙特为摆动产油国，作为机构）。决定产量配额；在冲突导致供给中断、油价高企时权衡市场份额、"
          "财政收入与美国页岩油的竞争。",
          ["成员国财政收入（沙特财政赤字）", "市场份额与美国页岩油的竞争", "联盟团结与配额纪律", "与美国的关系", "避免需求被高油价摧毁"],
          [Q("quota", "本季产量配额变化多少（万桶/日）？",
             "给出相对当前配额的变化（例如 0、+41、−100 万桶/日）与生效月份；增产受可达市场的闲置产能约束，说明谁来增减。"),
           Q("discipline", "如何处理配额纪律与沙特的角色？",
             "是否容忍成员国超产，沙特是否单边增减产或额外自愿减产。", required=False)],
          "例如：调整官方售价、与美国或中国的外交、对冲突方的调停、长期供货合同。",
          book_lines(D, "opec", None, "产油国账户（IMF WEO、EIA）") + inst_lines(D, ("opec",)))

    actor("belligerents", "冲突方", "2026 年导致石油供给中断的武装冲突中的产油方与交战各方（身份未锁定，不具名）。他们的行动决定航道、设施与供给的恢复或进一步中断。",
          ["军事与政治目标", "石油收入与设施安全", "外部调停与制裁压力", "国内经济的承受力"],
          [Q("conflict", "本季冲突走向如何：僵持、升级还是停火？",
             "说明具体的军事与外交动作，以及对石油设施与航道的影响。"),
           Q("supply", "石油供给中断的程度怎么变化（万桶/日）？",
             "给出中断量的大致变化（例如“中断再增加 50 万桶/日”或“恢复 100 万桶/日”），以及航道是否通行。")],
          "例如：针对特定设施的行动、扣押船只、接受或拒绝调停、与外部大国接触。不具名，只描述对世界状态的影响。",
          book_lines(D, "opec", ["brent_spot", "kwt_real_gdp_growth_2026", "irq_real_gdp_growth_2026",
                                 "kwt_current_account_gdp_2026", "irq_current_account_gdp_2026"], "受冲突影响的产油国")
          + inst_lines(D, ("belligerents",)))

    actor("bigtech", "超大型科技公司 CEO", "微软、Alphabet、亚马逊、Meta、甲骨文的管理层（CEO、CFO，作为整体；不用任何在世人物的姓名）。"
          "每季财报时决定 AI 资本开支指引、用工、股东回报与 AI 定价。",
          ["AI 竞赛中的位置与算力", "资本开支的回报与自由现金流", "债务融资成本与评级", "股东回报与股价",
           "监管（反垄断、出口管制）"],
          [Q("capex", "未来四个季度的 AI 资本开支指引怎么调整？",
             "维持或小幅调整、大幅上调、下调（进入消化期）或大幅削减（2001–02 型）；给出五家合计的年化金额（十亿美元，近四季合计见状态）"
             "或变化百分比，以及融资方式（现金、发债、租赁）。"),
           Q("hiring", "招聘与用工怎么安排？", "扩招、冻结或裁员，以及用 AI 替代白领岗位的程度。"),
           Q("payout", "股东回报（回购、股息）与 AI 定价怎么调整？",
             "回购增减（十亿美元）；AI 服务价格上调、不变或降价（价格战）。", required=False)],
          "例如：发债融资、数据中心选址与电力合同、芯片采购、与政府或监管机构的协议、并购。",
          book_lines(D, "bigtech", None, "五家合计（SEC 季报，2026Q3）")
          + book_lines(D, "bigtech_msft", ["capex", "fcf", "rpo"], "微软")
          + book_lines(D, "bigtech_googl", ["capex", "fcf", "rpo"], "Alphabet")
          + book_lines(D, "bigtech_amzn", ["capex", "fcf", "debt"], "亚马逊")
          + book_lines(D, "bigtech_meta", ["capex", "fcf", "debt"], "Meta")
          + book_lines(D, "bigtech_orcl", ["capex", "fcf", "debt", "rpo"], "甲骨文") + inst_lines(D, ("bigtech",)))

    actor("households", "消费者/居民", "美国家庭与消费者（作为整体）。决定消费与储蓄、是否去杠杆，以及买房还是观望；"
          "财富高度集中，股票占金融资产的比重处在历史高位。",
          ["收入与就业安全", "生活成本：汽油、食品、房租与利率", "财富与股市", "债务负担与信贷可得性", "买房的时机"],
          [Q("spending", "本季支出与储蓄怎么平衡？",
             "按收入正常消费、预防性储蓄或动用储蓄维持消费；给出储蓄率的大致变化（百分点，当前见状态），并区分高收入与中低收入家庭。"),
           Q("housing", "买房意愿如何？", "观望、正常或抢先入市；结合房贷利率与房价说明。"),
           Q("debt", "借贷还是去杠杆？", "增加信用卡与汽车贷款、维持，或偿还债务；说明还款压力。", required=False)],
          "例如：换工作或增加工时、搬迁、调整股票等资产配置、减少旅行与大件消费。作为整体，不扮演具体的人。",
          book_lines(D, "households", None, "居民账户（Z.1、DFA、BEA）"))

    actor("investors", "华尔街投资者", "资产管理者、对冲基金、券商与散户的整体（作为市场力量）。决定风险配置、股债配置与科技仓位；"
          "他们的行动通过资产价格、信用利差、期限溢价与美元影响世界。",
          ["风险调整后的回报", "利率与通胀的路径", "流动性与保证金", "AI 主题的估值与集中度", "政策与地缘风险"],
          [Q("risk", "风险偏好怎么变化？",
             "加仓风险资产、维持、降低风险或避险抛售；说明对信用利差与波动率的方向性影响。"),
           Q("allocation", "股债配置怎么调整？",
             "股票、长期国债、短债与现金、黄金的超配或低配；说明期限选择（对期限溢价与长端收益率的影响）。"),
           Q("tech", "科技股仓位怎么调整？",
             "加仓、维持或轮动出去（转向价值、能源、防御板块）；说明对 AI 主题估值与集中度的看法。")],
          "例如：对冲特定风险、增减美元或黄金、去杠杆或加杠杆、参与 IPO 与发债认购、基差交易。",
          book_lines(D, "investors", None, "投资者账户（Z.1、H.6、FINRA）")
          + book_lines(D, "households", ["equity_share_fin_assets", "equity_share_pctile_since_1990"], "居民的股票配置"))

    actor("banks", "银行/放贷机构", "美国商业银行与其他放贷机构（作为整体）。决定贷款标准（SLOOS）与信贷投放；关注资产质量、"
          "存款成本与证券浮亏。",
          ["资产质量：拖欠与核销", "净息差与存款成本", "资本与监管要求", "证券组合的利率风险", "贷款需求"],
          [Q("standards", "本季放贷标准怎么调整？",
             "放松、不变、收紧（SLOOS 净收紧 10–30%）或大幅收紧（30% 以上）；给出净收紧比例的大致变化（百分点），可以分企业贷款、"
             "商业地产与消费贷款。"),
           Q("deposits", "存款定价与证券组合怎么处理？",
             "存款利率跟随市场利率的程度、是否出售浮亏证券，以及久期拉长或缩短。", required=False)],
          "例如：增加拨备、发行资本工具、收缩商业地产敞口、调整回购与分红、向贴现窗口借款。",
          book_lines(D, "banks", None, "银行账户（FDIC、H.8）"))

    actor("firms", "非科技企业", "美国的非科技企业（制造、服务、零售、能源等，作为整体）。决定投资、招聘与定价；"
          "面对高利率、关税成本与油价冲击。",
          ["利润率与成本（能源、关税、工资）", "需求前景", "融资成本与信用条件", "定价能力与市场份额"],
          [Q("hiring", "本季招聘计划如何？", "扩招、按计划、放缓或裁员；尽量给出对非农就业的大致影响（千人/月）。"),
           Q("capex", "资本开支怎么调整？", "增加、维持、推迟或削减，说明原因（利率、关税、需求）。"),
           Q("pricing", "定价怎么调整：转嫁成本还是让价保份额？",
             "提价转嫁关税与能源成本（给出大致幅度，%）、维持价格，或压缩利润率保住份额。")],
          "例如：调整供应链（转移产地、囤货）、发债或回购、申请关税豁免、游说、自动化投资。",
          book_lines(D, "nfc", None, "非金融企业账户（Z.1、NIPA）"))
    return A


SYNTH_INSTRUCTIONS = """你是「美国与世界」的世界合成器。每一步代表一个季度：给定季初状态与所有行为者在本季的决策（每个决策问题的文字回答与自主行动），写出下一季季初的状态。

1. 联合求解：把所有决策放在一起考虑它们的相互作用（例如美联储加息与财政扩张同时发生、美国加征关税与中国报复同时发生、冲突升级与欧佩克+ 增产同时发生），得到一个前后一致的结果，而不是逐条叠加。
2. 分步工作：先写本季要事（主要事件、各方决策如何相互作用或冲突、哪些力量占主导），再分批写指标数值，然后写每个预设问题的现状，最后写总结。每一步只写被要求的键，并与之前各步已经写下的内容保持一致。
3. 一致性：每个字段都要与其他字段、与决策一致。美联储调整利率时，政策利率按回答的幅度变动，2 年期收益率通常同向；油价变动影响 CPI 同比与汽油价格；关税决策改变有效关税率；选举结果改变「两院与白宫的多数党」与「众议院控制权」；债务上限的提高或暂停改变「债务上限状态」与余量；字段之间的会计关系（赤字与债务、利率与利息负担）要说得通。
4. 只改变本季事件确实会移动的字段；没有理由变动的字段保持原值，reason 留空。慢变量（劳动收入占比、利润/GDP、债务/GDP、IT 投资/GDP）一个季度只会小幅移动。
5. 每个变动的字段都给出简短理由，指明是哪个决策或机制。
6. 合理性：数值放在历史区间内的合理位置；若数值超出其历史区间或合理区间，必须在理由里说明为什么。
7. 定性字段（枚举）只能取给定选项之一；标注“未确认”的事实，在被决策或事件改变之前保持原样。
8. 资产价格（标普 500、纳斯达克 100、QQQ、黄金、TLT、VIX、房价指数）是普通的状态字段：按本季事件与其他字段一致地移动——利率与期限溢价、企业盈利与 AI 投资、风险偏好与信用利差、美元、通胀与避险需求——每个变动给出理由，并对照历史区间保持合理（一个季度的涨跌通常不超过历史上常见的季度波动）。QQQ 与纳斯达克 100 同步变动；TLT 与 20–30 年期收益率反向变动；房价指数变动缓慢。
9. 预设问题的现状：每个问题都要写出季末的现状（2–4 句；没有变化也要完整重述），写清楚已经确定的和仍不确定的，不要假装确定；仍未确认的事实继续标注“未确认”。
10. 总结约 100 字（不超过 120 字），概括全部指标与全部问题现状的要点。
11. 没有回应的行为者视为维持现状；回答被标为“未回答”的问题，不要替行为者编造答案。"""


def build_world(D: Data) -> dict:
    fields, values = build_schema(D)
    questions, statuses, narrative = build_state_text(D, values)
    actors = build_actors(D, values)
    return {
        "format": "worldsim-world/2", "id": "us_2026q4", "name": "美国与世界 · 2026Q4",
        "description": WORLD_DESCRIPTION,
        "language": "中文", "period_step": {"unit": "quarter", "count": 1}, "history_window": 3,
        "model": {"model": "", "temperature": None, "reasoning_effort": ""},
        "schema": {"fields": fields, "questions": questions}, "actors": actors,
        "synthesizer": {"instructions": SYNTH_INSTRUCTIONS, "model": "", "max_items_per_call": 6},
        "initial_state": {"step": 0, "period": "2026Q4", "values": values, "statuses": statuses,
                          "narrative": narrative, "summary": "起点：2026Q4 季初（数据截至 2026-09）"},
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
    ap.add_argument("--worldsim", default=os.environ.get("WORLDSIM_REPO") or str(ROOT.parent / "worldsim"),
                    help="a checkout of github.com/Dongyu-Jia/worldsim (default: $WORLDSIM_REPO, else ../worldsim)")
    args = ap.parse_args(argv)
    repo = Path(args.worldsim).expanduser().resolve()
    if not (repo / "worldsim" / "model.py").is_file():
        print(f"no WorldSim checkout at {repo}: git clone https://github.com/Dongyu-Jia/worldsim.git there, "
              "or pass --worldsim PATH")
        return 1
    sys.path.insert(0, str(repo))
    out = repo / OUT_REL
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
    narrative = world["initial_state"]["narrative"]
    if len(narrative) > M.narrative_limits(world["language"])[1]:
        print(f"the initial narrative has {len(narrative)} characters: over the limit")
        return 1
    print(f"{len(world['schema']['fields'])} fields, {len(world['schema']['questions'])} state questions, "
          f"narrative {len(narrative)} characters, {len(world['actors'])} actors: "
          + ", ".join(f"{a['name']} ({len(a['decision_questions'])})" for a in world["actors"]))
    if args.check:
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print("committed file is up to date" if same else "committed file differs from a fresh build")
        return 0 if same else 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.encode()):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
