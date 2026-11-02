"""Per-actor accounts, sector accounts and observation packets (LEDGERS.md; AGENT_DESIGN.md §5).

Accounts are bookkeeping, not a second model:

- **L1 one writer per number.** Flows the quarterly model draws (deficit change, consumption, profits, capex, house
  prices, asset returns) are booked, never recomputed; rules create only the lines the model lacks (federal
  interest, the TGA, debt subject to limit and extraordinary measures, Fed income and the deferred asset, customs,
  hyperscaler depreciation and cash, ...).
- **L2′ no learned model inputs.** The debt-type s fields are views in the schema; the accounts compute them for
  packets and the UI (:meth:`Accounts.views`, identity I9) and hand the model nothing.
- **L3 constraints and reasons, never prices.** :func:`gates` turns the books into menu gates and prior tilts
  (G-L1…G-L9); :func:`sector_summary` gives the synthesizer its 部门账户摘要.

Pieces:

1. :data:`BOOKS` / :data:`LINES`: 15 books, 126 lines (statement, unit, update class, calibration, packet, visibility).
2. The build step (``python -m scenariolab.build --only agents_accounts``): :class:`Fetcher` downloads the free
   endpoints with a plain-file cache under ``data/raw/agents/`` (FRED through the project's shared FRED cache);
   SEC requests carry ``SEC_USER_AGENT`` from the environment, go only to ``sec.gov`` hosts, at most 10 a second,
   and are never logged; without the variable the cached filings are used and labelled 未刷新（缺少 SEC_USER_AGENT）.
   :func:`build_root` writes ``data/results/agents_accounts_root.json``; :func:`import_research` fills the cache
   from the research round's downloads (bytes copied, nothing executed).
3. :class:`Accounts`: the per-particle books (vectorised over N particles; never averaged; they move with their
   particles), the quarterly update A1–A16 (:meth:`Accounts.step`), the identities I1–I10 (:meth:`Accounts.check`).
4. Packets: :func:`own_block`, :func:`news_digest` (with the T-L10 guard), :func:`signal_board`,
   :class:`Credibility`, :func:`packet`, :func:`regime_line`.

Units: $bn; flows per quarter unless "SAAR"; rates in percent. Nothing here calls a model or an LLM.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from .. import config, jsonio

log = logging.getLogger(__name__)

ACCOUNTS_VERSION = "agents-accounts-3 (2026-11-02; per-particle actions; the deficit's pull to its baseline, debt >= 0)"
ROOT_PATH = config.RESULTS_DIR / "agents_accounts_root.json"
RAW_AGENTS = config.RAW_DIR / "agents"
TOL = 1e-6

# =====================================================================================================================
# 1. Books and lines (LEDGERS §2–§3, ledgers.json)
# =====================================================================================================================

BOOKS: dict[str, dict] = {
    "fed": {"zh": "美联储", "owner": "fed", "kind": "actor"},
    "fiscal": {"zh": "联邦政府（国会、财政部、总统共用）", "owner": "congress, treasury, president", "kind": "actor"},
    "president": {"zh": "总统与白宫（政治资本）", "owner": "president", "kind": "actor"},
    "china": {"zh": "中国", "owner": "china", "kind": "actor"},
    "opec": {"zh": "欧佩克+（以沙特为摆动产油国）", "owner": "opec", "kind": "actor"},
    "bigtech": {"zh": "超大型科技公司（五家合计，可分公司）", "owner": "bigtech", "kind": "actor"},
    "voter": {"zh": "选民（家庭部门的分布）", "owner": "voter", "kind": "actor"},
    "courts": {"zh": "法院（案卷，不是财务报表）", "owner": "courts", "kind": "actor"},
    "belligerents": {"zh": "冲突方（模板）", "owner": "belligerents", "kind": "actor"},
    "households": {"zh": "家庭部门", "owner": "sector", "kind": "sector"},
    "nfc": {"zh": "非金融企业", "owner": "sector", "kind": "sector"},
    "banks": {"zh": "银行", "owner": "sector", "kind": "sector"},
    "investors": {"zh": "投资者", "owner": "sector", "kind": "sector"},
    "row": {"zh": "外国部门", "owner": "sector", "kind": "sector"},
    "sectoral": {"zh": "部门余额（恒等式）", "owner": "identity", "kind": "check"},
}
UPDATE_CLASSES = {"identity": "恒等式", "model": "模型", "action": "动作", "rule": "规则", "link": "链接",
                  "carried": "延续", "user": "锁定", "event": "事件"}

LINES = [  # book, line, 中文, statement, unit, update class, calibration, packet, visibility, source, series
    ("fed", "total_assets", "总资产", "balance_sheet", "$bn", "identity", None, True, "public", "FRED_H41", "WALCL"),
    ("fed", "soma_treasuries", "持有国债", "balance_sheet", "$bn", "action", None, True, "public", "FRED_H41", "TREAST"),
    ("fed", "soma_bills", "其中国库券", "balance_sheet", "$bn", "action", None, True, "public", "NYFED_SOMA", "summary.bills"),
    ("fed", "soma_mbs", "持有 MBS", "balance_sheet", "$bn", "rule", None, True, "public", "FRED_H41", "WSHOMCB"),
    ("fed", "other_assets", "其他资产（溢价、贷款等）", "balance_sheet", "$bn", "carried", None, False, "public", "FRED_H41", "WALCL-TREAST-WSHOMCB"),
    ("fed", "currency", "流通中货币", "balance_sheet", "$bn", "rule", "A10", True, "public", "FRED_H41", "WCURCIR"),
    ("fed", "reserves", "银行准备金", "balance_sheet", "$bn", "identity", None, True, "public", "FRED_H41", "WRESBAL"),
    ("fed", "reverse_repo_total", "逆回购（外国官方 + 隔夜）", "balance_sheet", "$bn", "carried", None, True, "public", "FRED_H41", "WLRRAL"),
    ("fed", "tga", "财政部一般账户", "balance_sheet", "$bn", "link", None, True, "public", "FRED_H41", "WDTGAL"),
    ("fed", "deferred_asset", "递延资产", "balance_sheet", "$bn", "rule", None, True, "public", "FRED_H41", "-RESPPLLOPNWW"),
    ("fed", "interest_income", "利息收入（扣除运营成本）", "income", "$bn/q", "rule", "A5", False, "public", "derived", "A5"),
    ("fed", "interest_expense", "利息支出（准备金与逆回购）", "income", "$bn/q", "rule", None, False, "public", "derived", "IORB x reserves + RRP rate x RRP"),
    ("fed", "net_income", "净收入", "income", "$bn/q", "identity", None, True, "public", "derived", "-Δ deferred asset"),
    ("fed", "remittances", "上缴财政部", "cash_flow", "$bn/q", "identity", None, False, "public", "derived", "identity"),
    ("fed", "rmp_purchases_q", "准备金管理购买", "cash_flow", "$bn/q", "action", None, True, "public", "NYFED_SOMA", "Δ bills"),
    ("fed", "inflation_gap", "通胀缺口", "mandate", "pp", "model", None, True, "public", "FRED_BEA", "PCEPILFE"),
    ("fed", "unemployment_gap", "就业缺口", "mandate", "pp", "model", None, True, "public", "FRED_CBO", "UNRATE - NROU"),
    ("fiscal", "deficit_nipa_gdp", "联邦净借款/GDP（NIPA，模型口径）", "income", "% GDP", "model", None, False, "public", "FRED_BEA", "-FGDEF/GDP"),
    ("fiscal", "deficit_cash_q", "现金赤字（MTS）", "income", "$bn/q", "rule", "A2", True, "public", "FRED_MTS", "-MTSDS133FMS"),
    ("fiscal", "receipts_q", "收入", "income", "$bn/q", "rule", "A14", True, "public", "FRED_MTS", "MTSR133FMS"),
    ("fiscal", "outlays_q", "支出", "income", "$bn/q", "identity", None, True, "public", "FRED_MTS", "MTSO133FMS"),
    ("fiscal", "net_interest", "净利息", "income", "$bn/q", "rule", "A4", True, "public", "FD_MTS", "MTS table 9 Net Interest"),
    ("fiscal", "customs_q", "关税收入", "income", "$bn/q", "identity", None, True, "public", "FD_MTS4", "Customs duties"),
    ("fiscal", "customs_refunds", "关税退款", "cash_flow", "$bn/q", "event", None, True, "public", "FD_MTS4", "Customs refunds"),
    ("fiscal", "fed_remittances", "美联储上缴", "income", "$bn/q", "link", None, False, "public", "derived", "fed.remittances"),
    ("fiscal", "debt_held_public", "公众持有债务", "balance_sheet", "$bn", "identity", "A1", True, "public", "FD_DTS", "Debt Held by the Public"),
    ("fiscal", "intragov_holdings", "政府内部持有", "balance_sheet", "$bn", "rule", None, False, "public", "FD_DTS", "Intragovernmental Holdings"),
    ("fiscal", "statutory_limit", "法定债务上限", "balance_sheet", "$bn", "action", None, True, "public", "FD_DTS", "Statutory Debt Limit"),
    ("fiscal", "limit_headroom", "上限余量", "balance_sheet", "$bn", "identity", None, True, "public", "derived", "limit - subject"),
    ("fiscal", "extraordinary_measures_used", "已用非常规措施", "balance_sheet", "$bn", "rule", None, True, "public", "judgment", "capacity 350bn"),
    ("fiscal", "tga", "TGA 现金", "balance_sheet", "$bn", "identity", None, True, "public", "FD_DTS", "Operating cash balance"),
    ("fiscal", "arrears", "逾期未付（越过 X 日期后）", "balance_sheet", "$bn", "identity", None, False, "public", "derived", "identity"),
    ("fiscal", "avg_rate_marketable", "可交易国债平均利率", "balance_sheet", "%", "rule", "A3", True, "public", "FD_AVG", "Total Marketable"),
    ("fiscal", "bill_share", "国库券占比", "balance_sheet", "%", "action", None, True, "public", "FD_MSPD", "Bills / marketable"),
    ("fiscal", "holders_fed", "美联储持有", "balance_sheet", "$bn", "link", None, False, "public", "derived", "fed.soma_treasuries"),
    ("fiscal", "holders_foreign", "外国持有", "balance_sheet", "$bn", "link", None, False, "public", "TIC_T5", "Grand Total"),
    ("fiscal", "house_senate_control", "两院控制与席位差", "political", "seats", "event", None, True, "public", "CONGRESS_SEATS", "House Clerk / Senate"),
    ("fiscal", "appropriations_deadline", "拨款到期日", "political", "date", "user", None, True, "public", "user", "CR expiry"),
    ("president", "approval", "支持率", "political", "%", "user", None, True, "public", "APP_GALLUP", "Gallup job approval"),
    ("president", "gasoline_price", "汽油零售价", "scorecard", "$/gal", "rule", None, True, "public", "FRED_EIA", "GASREGW"),
    ("president", "cpi_yoy", "CPI 同比", "scorecard", "%", "model", None, True, "public", "MAIN_PANEL", "s_cpi_yoy"),
    ("president", "real_dpi_pc_yoy", "人均实际可支配收入同比", "scorecard", "%", "link", None, True, "public", "FRED_BEA", "A229RX0"),
    ("president", "us_equity_drawdown", "美股距高点", "scorecard", "%", "model", None, True, "public", "MAIN_PANEL", "p_us_equity"),
    ("president", "effective_tariff_rate", "有效关税率", "instruments", "%", "link", None, True, "public", "FRED_BEA", "B235RC1Q027SBEA / A255RC1Q027SBEA"),
    ("president", "spr_inventory", "战略石油储备", "instruments", "mn bbl", "action", None, True, "public", "EIA_WPSR", "WCSSTUS1"),
    ("president", "tariff_authority", "关税授权状态", "instruments", "fact", "event", None, True, "public", "SCOTUS", "court rulings"),
    ("china", "fx_reserves", "外汇储备", "balance_sheet", "$bn", "rule", "A7", True, "public", "FRED_IMF_IFS", "TRESEGCNM052N"),
    ("china", "us_treasury_holdings", "持有美债", "balance_sheet", "$bn", "carried", None, True, "public", "TIC_T5", "China, Mainland"),
    ("china", "current_account_gdp", "经常账户/GDP", "income", "% GDP", "carried", None, True, "public", "IMF_WEO_DM", "BCA_NGDPD/CHN"),
    ("china", "gg_deficit_gdp", "广义政府赤字/GDP", "income", "% GDP", "action", None, True, "public", "IMF_WEO_DM", "GGXCNL_NGDP/CHN"),
    ("china", "gg_debt_gdp", "政府债务/GDP", "balance_sheet", "% GDP", "identity", None, True, "public", "IMF_WEO_DM", "GGXWDG_NGDP/CHN"),
    ("china", "private_credit_gdp", "私人信贷/GDP", "balance_sheet", "% GDP", "action", None, True, "public", "FRED_BIS", "QCNPAM770A"),
    ("china", "real_house_price_vs_peak", "实际房价距峰值", "balance_sheet", "%", "carried", None, True, "public", "FRED_BIS", "QCNR628BIS"),
    ("china", "growth_vs_target", "增长与目标", "income", "pp", "user", None, True, "public", "IMF_WEO_DM", "NGDP_RPCH/CHN"),
    ("china", "cny_per_usd", "人民币中间价", "instruments", "CNY/USD", "action", None, True, "public", "FRED_FED_H10", "DEXCHUS"),
    ("china", "us_bilateral_trade", "对美贸易", "income", "$bn/12m", "rule", None, True, "public", "FRED_CENSUS", "IMPCH, EXPCH"),
    ("opec", "brent", "布伦特", "market", "$/bbl", "model", None, True, "public", "FRED_EIA", "DCOILBRENTEU"),
    ("opec", "production", "产量（沙特、集团）", "operations", "mb/d", "action", None, True, "public", "OPEC_MOMR", "secondary sources"),
    ("opec", "spare_capacity_reachable", "可达市场的闲置产能", "operations", "mb/d", "user", None, True, "own", "EIA_STEO", "surplus capacity"),
    ("opec", "oil_revenue", "石油收入", "income", "$bn/q", "identity", None, True, "public", "EIA_OPEC_REV", "net export revenue"),
    ("opec", "fiscal_balance", "财政余额（沙特）", "income", "% GDP", "identity", None, True, "public", "IMF_WEO_DM", "GGXCNL_NGDP/SAU"),
    ("opec", "fiscal_breakeven", "财政盈亏平衡油价", "income", "$/bbl", "identity", None, True, "public", "IMF_REO", "breakeven"),
    ("opec", "reserves", "外汇储备（沙特）", "balance_sheet", "$bn", "identity", None, True, "public", "FRED_IMF_IFS", "TRESEGSAM052N"),
    ("opec", "members_growth", "成员增长", "income", "%", "carried", None, True, "public", "IMF_WEO_DM", "NGDP_RPCH"),
    ("opec", "us_crude_production", "美国原油产量（竞争者）", "market", "mb/d", "rule", None, True, "public", "EIA_WPSR", "WCRFPUS2"),
    ("bigtech", "revenue", "收入", "income", "$bn/q", "rule", "A8", True, "public", "SEC_XBRL", "revenue concepts"),
    ("bigtech", "depreciation", "折旧", "income", "$bn/q", "rule", "A8", True, "public", "SEC_XBRL", "depreciation concepts"),
    ("bigtech", "operating_income", "经营利润", "income", "$bn/q", "identity", None, True, "public", "SEC_XBRL", "OperatingIncomeLoss"),
    ("bigtech", "interest", "利息支出", "income", "$bn/q", "rule", None, False, "public", "derived", "debt x (y10 + 0.6)"),
    ("bigtech", "cfo", "经营现金流", "cash_flow", "$bn/q", "rule", None, True, "public", "SEC_XBRL", "NetCashProvidedByUsedInOperatingActivities"),
    ("bigtech", "capex", "资本开支", "cash_flow", "$bn/q", "action", None, True, "public", "SEC_XBRL", "capex concepts"),
    ("bigtech", "lease_principal", "融资租赁本金", "cash_flow", "$bn/q", "carried", None, False, "public", "SEC_XBRL", "FinanceLeasePrincipalPayments"),
    ("bigtech", "fcf", "自由现金流", "cash_flow", "$bn/q", "identity", None, True, "public", "derived", "CFO - capex - lease principal"),
    ("bigtech", "buybacks", "回购", "cash_flow", "$bn/q", "action", None, True, "public", "SEC_XBRL", "PaymentsForRepurchaseOfCommonStock"),
    ("bigtech", "dividends", "股息", "cash_flow", "$bn/q", "carried", None, True, "public", "SEC_XBRL", "dividend concepts"),
    ("bigtech", "debt_issuance", "新增债务", "cash_flow", "$bn/q", "rule", None, True, "public", "derived", "financing rule"),
    ("bigtech", "cash_securities", "现金与证券", "balance_sheet", "$bn", "identity", None, True, "public", "SEC_XBRL", "cash + securities"),
    ("bigtech", "debt", "债务", "balance_sheet", "$bn", "identity", None, True, "public", "SEC_XBRL", "debt concepts"),
    ("bigtech", "rpo", "未履约合同（RPO）", "balance_sheet", "$bn", "carried", None, True, "public", "SEC_XBRL", "RevenueRemainingPerformanceObligation"),
    ("bigtech", "headcount", "员工数", "balance_sheet", "persons", "action", None, False, "public", "SEC_XBRL", "10-K text (not tagged in XBRL)"),
    ("voter", "nw_distribution", "净资产分布", "balance_sheet", "% NW", "rule", None, True, "public", "FRED_DFA", "WFRBST01134, WFRBSN09161, WFRBSN40188, WFRBSB50215"),
    ("voter", "real_dpi_pc_growth", "人均实际可支配收入增长", "income", "%", "link", None, True, "public", "FRED_BEA", "A229RX0"),
    ("voter", "saving_rate", "储蓄率", "income", "%", "link", None, True, "public", "FRED_BEA", "PMSAVE/DSPI"),
    ("voter", "debt_service", "偿债率", "cash_flow", "% DPI", "carried", None, True, "public", "FRED_FED", "TDSP"),
    ("voter", "election_odds", "选举赔率", "political", "probability", "user", None, True, "public", "user", "market odds"),
    ("courts", "docket", "待决案件", "docket", "list", "user", None, True, "public", "SCOTUS", "docket"),
    ("courts", "money_at_stake", "涉及金额", "docket", "$bn", "link", None, True, "public", "FD_MTS4", "refunds paid / collections at issue"),
    ("belligerents", "oil_exports", "石油出口量与收入", "income", "mb/d, $bn/q", "user", None, True, "public", "user", "pinned"),
    ("belligerents", "fiscal_balance", "财政余额", "income", "% GDP", "carried", None, True, "public", "IMF_WEO_DM", "GGXCNL_NGDP/{ISO3}"),
    ("belligerents", "reserves", "储备（含被冻结部分）", "balance_sheet", "$bn", "carried", None, True, "public", "CBR", "reserves"),
    ("belligerents", "military_spending", "军费", "income", "% GDP", "carried", None, True, "public", "SIPRI", "milex"),
    ("belligerents", "sanctions", "制裁与航道状态", "facts", "fact", "event", None, True, "public", "user", "pinned"),
    ("households", "equities", "股票与基金", "balance_sheet", "$bn", "model", None, False, "public", "FRED_Z1", "HNOCEAQ027S + HNOMFAQ027S"),
    ("households", "real_estate", "房地产", "balance_sheet", "$bn", "model", None, False, "public", "FRED_Z1", "HNOREMQ027S"),
    ("households", "other_assets", "其他资产", "balance_sheet", "$bn", "rule", None, False, "public", "FRED_Z1", "TNWBSHNO + TLBSHNO - equities - real estate"),
    ("households", "liabilities", "负债", "balance_sheet", "$bn", "carried", None, False, "public", "FRED_Z1", "TLBSHNO"),
    ("households", "net_worth", "净资产", "balance_sheet", "$bn", "identity", "A6", False, "public", "FRED_Z1", "TNWBSHNO"),
    ("households", "dpi", "可支配收入", "income", "$bn SAAR", "carried", "A15", False, "public", "FRED_BEA", "DSPI"),
    ("households", "pce", "消费", "income", "$bn SAAR", "model", None, False, "public", "FRED_BEA", "PCE"),
    ("households", "saving_rate", "储蓄率", "income", "%", "identity", None, False, "public", "FRED_BEA", "PMSAVE/DSPI"),
    ("households", "debt_service_ratio", "偿债率", "cash_flow", "% DPI", "carried", None, False, "public", "FRED_FED", "TDSP"),
    ("nfc", "profits_pbt", "税前利润", "income", "$bn SAAR", "model", None, False, "public", "FRED_Z1_NIPA", "A464RC1Q027SBEA"),
    ("nfc", "net_interest", "净利息", "income", "$bn SAAR", "rule", None, False, "public", "FRED_Z1_NIPA", "B471RC1Q027SBEA"),
    ("nfc", "interest_coverage", "利息覆盖", "income", "x", "identity", None, False, "public", "derived", "(PBT + net interest)/net interest"),
    ("nfc", "capex", "资本开支", "cash_flow", "$bn SAAR", "model", None, False, "public", "FRED_Z1_NIPA", "BOGZ1FA105050005Q"),
    ("nfc", "net_equity_issuance", "股票净发行", "cash_flow", "$bn SAAR", "carried", None, False, "public", "FRED_Z1_NIPA", "NCBCEBQ027S"),
    ("nfc", "debt", "债务", "balance_sheet", "$bn", "identity", None, False, "public", "FRED_Z1_NIPA", "BCNSDODNS"),
    ("banks", "total_assets", "总资产", "balance_sheet", "$bn", "identity", None, False, "public", "FRED_FDIC_H8", "QBPBSTAS"),
    ("banks", "reserves", "准备金", "balance_sheet", "$bn", "link", None, False, "public", "FRED_H41", "WRESBAL"),
    ("banks", "securities", "证券", "balance_sheet", "$bn", "carried", None, False, "public", "FRED_FDIC_H8", "SBCACBW027SBOG"),
    ("banks", "unrealized_losses", "证券未实现损失", "balance_sheet", "$bn", "rule", None, False, "public", "FRED_FDIC_H8", "duration rule"),
    ("banks", "loans", "贷款", "balance_sheet", "$bn", "rule", None, False, "public", "FRED_FDIC_H8", "TOTLL"),
    ("banks", "deposits", "存款", "balance_sheet", "$bn", "identity", None, False, "public", "FRED_FDIC_H8", "DPSACBW027SBOG"),
    ("banks", "nii", "净利息收入", "income", "$bn/q", "carried", None, False, "public", "FRED_FDIC_H8", "QBPQYNTIY"),
    ("banks", "chargeoffs", "核销", "income", "% loans", "model", None, False, "public", "FRED_FDIC_H8", "CORALACBS"),
    ("investors", "margin_debt", "保证金负债", "balance_sheet", "$bn", "rule", "A9", False, "public", "FINRA_MARGIN", "Debit balances"),
    ("investors", "mmf_assets", "货币基金", "balance_sheet", "$bn", "carried", None, False, "public", "FRED_Z1", "MMMFFAQ027S"),
    ("investors", "household_equity_share", "家庭股票占金融资产", "balance_sheet", "%", "identity", None, False, "public", "derived", "households.equities / financial assets"),
    ("investors", "broker_dealer_assets", "券商资产", "balance_sheet", "$bn", "model", None, False, "public", "FRED_Z1", "BOGZ1FL664090005Q"),
    ("row", "foreign_holdings", "外国持有美债", "balance_sheet", "$bn", "rule", "A16", False, "public", "TIC_T5", "Grand Total"),
    ("row", "foreign_official", "其中官方", "balance_sheet", "$bn", "carried", None, False, "public", "TIC_T5", "Of which: Foreign Official"),
    ("row", "current_account_gdp", "经常账户/GDP", "income", "% GDP", "carried", None, False, "public", "FRED_BEA_ITA", "IEABC"),
    ("row", "niip", "国际投资净头寸", "balance_sheet", "$bn", "carried", None, False, "public", "FRED_BEA_ITA", "IIPUSNETIQ"),
    ("sectoral", "government_nl", "政府净借出", "flow", "$bn SAAR", "identity", None, False, "public", "FRED_BEA", "AD01RC1Q027SBEA"),
    ("sectoral", "nation_nl", "国家对外净借出", "flow", "$bn SAAR", "carried", None, False, "public", "FRED_BEA", "W162RC1Q027SBEA"),
    ("sectoral", "private_nl", "私人部门净借出", "flow", "$bn SAAR", "identity", None, False, "public", "FRED_BEA", "W994RC1Q027SBEA"),
]
LINE_KEYS = ("book", "id", "zh", "statement", "unit", "update", "calibration", "packet", "visibility", "src", "series")


def lines(book: str | None = None) -> list[dict]:
    """The line catalogue as dicts (one book, or all)."""
    return [dict(zip(LINE_KEYS, row)) for row in LINES if book is None or row[0] == book]


IDENTITIES = {
    "I1": "美联储：国债 + MBS + 其他资产 = 货币 + 准备金 + 逆回购 + TGA + 其他负债 − 递延资产",
    "I2": "美联储收益：Δ递延资产 = −min(净收入, 0) − 抵扣；上缴 = 净收入 − 抵扣",
    "I3": "TGA 现金：ΔTGA = 借款 − 现金赤字 − 其他融资（为负时记逾期未付）",
    "I4": "公众持有债务：Δ = 借款；受上限约束的债务 ≤ 上限",
    "I5": "持有人：美联储 + 外国 ≤ 公众持有债务",
    "I6": "家庭：Δ净资产 = 储蓄 + 股票重估 + 房地产重估 + 其他资产重估",
    "I7": "公司现金：Δ现金 = 经营现金流 − 资本开支 − 租赁本金 − 回购 − 股息 + 新增债务",
    "I8": "部门余额：私人 + 政府 = 国家对外（NIPA 表 5.1）",
    "I9": "视图 = 账户比率（s_debt、s_int_burden、s_hh_debt、s_nfc_debt、s_foreign_hold）",
    "I10": "银行：贷款 + 证券（含未实现损益）+ 其他资产 = 存款 + 其他负债 + 权益（含未实现损益）",
}

# Calibrations (LEDGERS §5.3; work/l03_anchors.json): accounting rules tested on history; judgment marked
CALIBRATIONS = {
    "A1": {"rule": "Δ公众持有债务 = 现金赤字 + ΔTGA + 其他融资", "omf_q": 18.4, "resid_sd_q": 138.1, "n": 93,
           "sample": "2003Q2–2026Q2"},
    "A2": {"rule": "现金赤字（MTS）= −52.8/4 + 1.055 × NIPA 净借款", "const_q": -13.2, "slope": 1.055, "r2": 0.957},
    "A3": {"rule": "可交易国债平均利率向 5 年期收益率部分调整", "lambda_q": 0.129, "oos_rmse_pp": 0.248,
           "sample": "2001–2026；2016–2026 样本外"},
    "A4": {"rule": "NIPA 联邦利息 = 158.1 + 1.075 × 平均利率 × 公众持有债务（年化）", "const_saar": 158.1, "slope": 1.075,
           "r2": 0.986},
    "A5": {"rule": "美联储净收入 = 组合净收益率 × SOMA − 准备金利率 × 准备金 − 逆回购利率 × 逆回购", "soma_net_yield": 1.973,
           "r2": 0.935, "per_25bp_iorb_bn_per_year": 8.3},
    "A6": {"rule": "家庭净资产 = 上季 + 储蓄 + 股票重估 + 房地产重估（系数 1）", "r2": 0.886, "sample": "1991–2026"},
    "A7": {"rule": "Δlog 中国储备 = 0.038% − 0.401·Δlog 美元（月度）", "const_pct_m": 0.038, "slope": -0.401, "r2": 0.553},
    "A8": {"rule": "超大型科技公司：混合使用年限 7 年的直线折旧；广告收入对名义 GDP 的弹性 2.19（只有 Alphabet 显著）",
           "life_years": 7.0, "ads_beta": 2.19},
    "A9": {"rule": "保证金负债月度对数变化 = 0.374% + 0.566 × 美股收益", "drift_pct_m": 0.374, "beta": 0.566,
           "r2": 0.336},
    "A10": {"rule": "流通货币每季 +1.44%（与名义 GDP 无关，R² 0.03）", "growth_q": 0.01437},
    "A11": {"rule": "债务上限约束期：2011 年以来 10 段，中位 136 天，0 次违约", "median_days": 135.5, "n_since_2011": 10},
    "A12": {"rule": "部门余额恒等式：146 季最大残差 1,600 万美元", "max_abs_resid_bn": 0.016,
            "private_nl_gdp_p01_p99": (-4.23, 15.95)},
    "A13": {"rule": "GDP 平减指数通胀 − CPI 通胀 = −0.42pp（1990–2019）", "wedge_pp": -0.42},
    "A14": {"rule": "联邦收入对名义 GDP 的弹性 1.73（1990–2026）", "elasticity": 1.73},
    "A15": {"rule": "可支配收入 = 名义 GDP × 根节点份额（增长回归 R² 0.06 失败，所以延续份额）"},
    "A16": {"rule": "外国持有占新增债务 0.318（近 8 季）", "share": 0.318},
}


# =====================================================================================================================
# 2. Downloads with a plain-file cache
# =====================================================================================================================

FD = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
SEC_HOST_SUFFIX = "sec.gov"
SEC_MIN_INTERVAL = 0.12          # at most ~8 requests a second (SEC fair access: <= 10)
NO_SEC_UA = "未刷新（缺少 SEC_USER_AGENT）"
STALE_DOWNLOAD = "未刷新（下载失败，使用缓存）"

FRED_SERIES = [   # every FRED series the snapshot reads (the endpoints LEDGERS §4 verified)
    "WALCL", "TREAST", "WSHOMCB", "WUPSHO", "WUDSHO", "WLCFLPCL", "WCURCIR", "WRESBAL", "WLRRAL", "RRPONTSYD", "WDTGAL",
    "RESPPLLOPNWW", "IORB", "RRPONTSYAWARD", "DFF", "PCEPILFE", "UNRATE", "NROU", "MTSR133FMS", "MTSO133FMS",
    "MTSDS133FMS", "GDP", "A091RC1Q027SBEA", "B235RC1Q027SBEA", "A255RC1Q027SBEA", "FYGFDPUN", "FYGFGDQ188S", "FDHBFIN",
    "FGDEF", "W994RC1Q027SBEA", "AD01RC1Q027SBEA", "AD02RC1Q027SBEA", "W162RC1Q027SBEA", "W995RC1Q027SBEA", "GASREGW",
    "UMCSENT", "A229RX0", "TRESEGCNM052N", "QCNPAM770A", "QCNR628BIS", "IMPCH", "EXPCH", "DEXCHUS", "TRESEGSAM052N",
    "DCOILBRENTEU", "POILBREUSDM", "TNWBSHNO", "HNOCEAQ027S", "HNOMFAQ027S", "TFAABSHNO", "HNOREMQ027S", "TLBSHNO",
    "CMDEBT", "DSPI", "PCE", "PMSAVE", "TDSP", "WFRBST01134", "WFRBSN09161", "WFRBSN40188", "WFRBSB50215", "BCNSDODNS",
    "A464RC1Q027SBEA", "W328RC1Q027SBEA", "B471RC1Q027SBEA", "A455RC1Q027SBEA", "BOGZ1FA105050005Q",
    "BOGZ1FA105005305Q", "BOGZ1FA106121075Q", "NCBCEBQ027S", "NCBEILQ027S", "QBPBSTAS", "QBPQYNTIY", "TOTLL",
    "SBCACBW027SBOG", "TASACBW027SBOG", "DPSACBW027SBOG", "DRALACBS", "CORALACBS", "MMMFFAQ027S", "WRMFNS",
    "BOGZ1FL664090005Q", "IEABC", "IIPUSNETIQ", "DGS2", "DGS10", "PSAVERT"]
FRED_VARIANT = {"DGS2": "eop", "DGS10": "eop"}          # month-end daily series
JSON_SOURCES = {   # cache path -> (url, params)
    "fiscal/debt_to_penny.json": (FD + "v2/accounting/od/debt_to_penny", {"sort": "-record_date", "page[size]": "5"}),
    "fiscal/debt_to_penny_full.json": (FD + "v2/accounting/od/debt_to_penny",
                                       {"fields": "record_date,debt_held_public_amt,intragov_hold_amt,tot_pub_debt_out_amt",
                                        "sort": "record_date", "page[size]": "10000"}),
    "fiscal/dts_operating_cash_balance.json": (FD + "v1/accounting/dts/operating_cash_balance",
                                               {"sort": "-record_date", "page[size]": "12"}),
    "fiscal/dts_debt_subject_to_limit.json": (FD + "v1/accounting/dts/debt_subject_to_limit",
                                              {"sort": "-record_date", "page[size]": "20"}),
    "fiscal/mspd_table_1.json": (FD + "v1/debt/mspd/mspd_table_1", {"sort": "-record_date", "page[size]": "30"}),
    "fiscal/avg_interest_rates_total_marketable.json": (FD + "v2/accounting/od/avg_interest_rates",
                                                        {"filter": "security_desc:eq:Total Marketable",
                                                         "sort": "-record_date", "page[size]": "400"}),
    "fiscal/mts_table_9_latest.json": (FD + "v1/accounting/mts/mts_table_9", {"sort": "-record_date", "page[size]": "60"}),
    "fiscal/mts_table_4_customs.json": (FD + "v1/accounting/mts/mts_table_4",
                                        {"filter": "classification_desc:eq:Customs Duties", "sort": "record_date",
                                         "page[size]": "10000"}),
    "nyfed/soma_summary.json": ("https://markets.newyorkfed.org/api/soma/summary.json", None),
    "imf/GGXCNL_NGDP.json": ("https://www.imf.org/external/datamapper/api/v1/GGXCNL_NGDP/CHN/SAU/ARE/KWT/IRQ/RUS/USA", None),
    "imf/GGXWDG_NGDP.json": ("https://www.imf.org/external/datamapper/api/v1/GGXWDG_NGDP/CHN/SAU/ARE/KWT/IRQ/RUS/USA", None),
    "imf/BCA_NGDPD.json": ("https://www.imf.org/external/datamapper/api/v1/BCA_NGDPD/CHN/SAU/ARE/KWT/IRQ/RUS/USA", None),
    "imf/NGDP_RPCH.json": ("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/CHN/SAU/ARE/KWT/IRQ/RUS/USA", None),
    "worldbank/oil_rents.json": ("https://api.worldbank.org/v2/country/SAU;IRQ;KWT;ARE;RUS/indicator/NY.GDP.PETR.RT.ZS",
                                 {"format": "json", "per_page": "60", "date": "2018:2025"}),
}
FILE_SOURCES = {   # cache path -> url (binary or text files)
    "tic/slt_table5.txt": "https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/slt_table5.txt",
    "eia/WCSSTUS1w.xls": "https://www.eia.gov/dnav/pet/hist_xls/WCSSTUS1w.xls",
    "eia/WCRFPUS2w.xls": "https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls",
    "finra/margin-statistics.xlsx": "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx",
}
CIK = {"MSFT": "0000789019", "GOOGL": "0001652044", "AMZN": "0001018724", "META": "0001326801", "ORCL": "0001341439"}
BT = {   # XBRL concepts per company (LEDGERS §2.6; l02 BT)
    "MSFT": {"rev": "RevenueFromContractWithCustomerExcludingAssessedTax", "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
             "div": "PaymentsOfDividendsCommonStock", "sec": ["ShortTermInvestments"],
             "debt": ["LongTermDebtNoncurrent", "LongTermDebtCurrent", "CommercialPaper"], "dep": "Depreciation"},
    "GOOGL": {"rev": "Revenues", "capex": "PaymentsToAcquirePropertyPlantAndEquipment", "div": "PaymentsOfDividends",
              "sec": ["MarketableSecuritiesCurrent"], "debt": ["LongTermDebtNoncurrent", "LongTermDebtCurrent", "CommercialPaper"],
              "dep": "Depreciation"},
    "AMZN": {"rev": "RevenueFromContractWithCustomerExcludingAssessedTax", "capex": "PaymentsToAcquireProductiveAssets",
             "div": None, "sec": ["MarketableSecuritiesCurrent"], "debt": ["LongTermDebtNoncurrent", "LongTermDebtCurrent"],
             "dep": "DepreciationDepletionAndAmortization"},
    "META": {"rev": "RevenueFromContractWithCustomerExcludingAssessedTax", "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
             "div": "PaymentsOfDividends", "sec": ["MarketableSecuritiesCurrent"],
             "debt": ["LongTermDebtNoncurrent", "LongTermDebtCurrent"], "dep": "DepreciationDepletionAndAmortization"},
    "ORCL": {"rev": "RevenueFromContractWithCustomerExcludingAssessedTax", "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
             "div": "PaymentsOfDividendsCommonStock", "sec": ["AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
             "debt": ["LongTermNotesPayable", "CommercialPaper"], "dep": "Depreciation"},
}
SEC_COMMON = ["OperatingIncomeLoss", "NetIncomeLoss", "NetCashProvidedByUsedInOperatingActivities",
              "FinanceLeasePrincipalPayments", "PaymentsForRepurchaseOfCommonStock",
              "CashAndCashEquivalentsAtCarryingValue", "RevenueRemainingPerformanceObligation"]


def sec_concepts(tk: str) -> list[str]:
    m = BT[tk]
    out = [m["rev"], m["capex"], m["dep"], *m["sec"], *m["debt"], *SEC_COMMON]
    if m["div"]:
        out.append(m["div"])
    return list(dict.fromkeys(out))


def is_sec_host(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return urlsplit(url).scheme == "https" and (host == SEC_HOST_SUFFIX or host.endswith("." + SEC_HOST_SUFFIX))


def _atomic_write(dest: Path, body: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)


class Fetcher:
    """The build step's downloads. A cached, non-empty file is reused unless `refresh`; a failed download falls
    back to the cache and is labelled 未刷新. Records one status per source (no URL with a key, no header)."""

    def __init__(self, cache: Path | None = None, *, refresh: bool = False, download: bool = True,
                 session=None, sec_user_agent: str | None = None, sleep=time.sleep):
        from ..data import sources
        self.cache = Path(cache) if cache else RAW_AGENTS
        self.refresh = refresh
        self.download = download
        self.status: dict[str, dict] = {}
        self._ua = sources.USER_AGENT
        self._session = session
        self._sec_ua = sec_user_agent if sec_user_agent is not None else (os.environ.get("SEC_USER_AGENT") or "").strip()
        self._sleep = sleep
        self._last_sec = 0.0

    # -- plumbing
    def _get(self, url: str, params=None, headers=None, timeout: float = 45.0, *, sec: bool = False):
        import requests
        getter = self._session.get if self._session is not None else requests.get
        if sec:     # SEC_USER_AGENT goes only to sec.gov: a redirect elsewhere is never followed with it (HTTP 3xx fails)
            return getter(url, params=params, headers=headers or {"User-Agent": self._ua}, timeout=timeout,
                          allow_redirects=False)
        return getter(url, params=params, headers=headers or {"User-Agent": self._ua}, timeout=timeout)

    def _fetch(self, key: str, rel: str, url: str, params=None, *, check=None, sec: bool = False) -> Path | None:
        dest = self.cache / rel
        cached = dest.exists() and dest.stat().st_size > 0
        if cached and not self.refresh:
            self.status[key] = {"status": "cached", "file": rel, "bytes": dest.stat().st_size}
            return dest
        absent = dest.with_name(dest.name + ".absent")
        if sec and absent.exists() and not self.refresh:       # a 404 is cached too: no request until --refresh
            self.status[key] = {"status": "absent", "file": rel, "why": "该公司不用这个 XBRL 概念（404，已缓存）"}
            return None
        if sec and not self._sec_ua:
            self.status[key] = {"status": "stale" if cached else "missing", "file": rel, "why": NO_SEC_UA}
            return dest if cached else None
        if not self.download:
            self.status[key] = {"status": "stale" if cached else "missing", "file": rel, "why": "未下载（离线构建）"}
            return dest if cached else None
        err = None
        for attempt in range(2):
            try:
                if sec:
                    if not is_sec_host(url):
                        raise ValueError("SEC 请求只发往 sec.gov")
                    wait = SEC_MIN_INTERVAL - (time.monotonic() - self._last_sec)
                    if wait > 0:
                        self._sleep(wait)
                    self._last_sec = time.monotonic()
                    resp = self._get(url, headers={"User-Agent": self._sec_ua, "Accept-Encoding": "gzip, deflate"},
                                     sec=True)
                else:
                    resp = self._get(url, params=params)
                code = int(getattr(resp, "status_code", 0))
                if code == 404 and sec:
                    _atomic_write(absent, b"404\n")
                    self.status[key] = {"status": "absent", "file": rel, "why": "该公司不用这个 XBRL 概念（404）"}
                    return dest if cached else None
                if code != 200:
                    raise RuntimeError(f"HTTP {code}")
                body = resp.content
                if check is not None and not check(body):
                    raise ValueError("内容不对")
                _atomic_write(dest, body)
                self.status[key] = {"status": "downloaded", "file": rel, "bytes": len(body)}
                return dest
            except Exception as e:  # noqa: BLE001 - recorded by type only (never a header or a key)
                err = f"{type(e).__name__}: {str(e)[:80]}" if not sec else type(e).__name__
                if attempt == 0:
                    self._sleep(1.0)
        self.status[key] = {"status": "stale" if cached else "missing", "file": rel,
                            "why": STALE_DOWNLOAD if cached else f"下载失败（{err}）"}
        return dest if cached else None

    # -- sources
    def json_source(self, rel: str) -> dict | list | None:
        url, params = JSON_SOURCES[rel]
        p = self._fetch(rel, rel, url, params, check=_is_json)
        return json.loads(p.read_text(encoding="utf-8")) if p else None

    def file_source(self, rel: str) -> Path | None:
        url = FILE_SOURCES[rel]
        chk = _is_xls if rel.endswith(".xls") else _is_zip if rel.endswith(".xlsx") else None
        return self._fetch(rel, rel, url, check=chk)

    def sec(self, tk: str, concept: str) -> dict | None:
        rel = f"sec/{tk}_{concept}.json"
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{CIK[tk]}/us-gaap/{concept}.json"
        p = self._fetch(f"sec:{tk}:{concept}", rel, url, check=_is_json, sec=True)
        return json.loads(p.read_text(encoding="utf-8")) if p else None

    def fred(self, sid: str) -> pd.Series:
        """FRED through the project's shared cache (data/raw/fred; API with FRED_API_KEY, graph CSV otherwise), then
        the research import (data/raw/agents/fred, windowed), then a download into the shared cache."""
        from ..data import sources
        variant = FRED_VARIANT.get(sid)
        key = f"fred:{sid}"
        shared = [config.RAW_DIR / "fred" / n for n in
                  ((f"{sid}_m_eop.json", f"{sid}_eop.csv") if variant == "eop" else (f"{sid}.json", f"{sid}.csv"))]
        local = self.cache / "fred" / f"{sid}.csv"
        if self.refresh and self.download:
            try:
                s = sources.fred(sid, eop=variant == "eop", refresh=True)
                self.status[key] = {"status": "downloaded", "file": "fred/" + sid}
                return s
            except Exception as e:  # noqa: BLE001
                why = f"{STALE_DOWNLOAD}：{type(e).__name__}"
        else:
            why = None
        if any(p.exists() and p.stat().st_size > 0 for p in shared):
            s = sources.fred(sid, eop=variant == "eop")
            self.status[key] = {"status": "stale" if why else "cached", "file": "fred/" + sid, **({"why": why} if why else {})}
            return s
        if local.exists() and local.stat().st_size > 0:
            s = sources.read_fred_csv(local, sid).dropna()
            if len(s) and (s.index.day == 1).all():
                s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
            self.status[key] = {"status": "stale" if why else "cached", "file": f"agents/fred/{sid}.csv",
                                **({"why": why} if why else {})}
            return s
        if not self.download:
            self.status[key] = {"status": "missing", "why": "未下载（离线构建）"}
            raise FileNotFoundError(sid)
        try:
            s = sources.fred(sid, eop=variant == "eop")
            self.status[key] = {"status": "downloaded", "file": "fred/" + sid}
            return s
        except Exception as e:  # noqa: BLE001
            self.status[key] = {"status": "missing", "why": f"下载失败（{type(e).__name__}）"}
            raise FileNotFoundError(sid) from None

    @property
    def has_sec_user_agent(self) -> bool:
        return bool(self._sec_ua)

    def summary(self) -> dict:
        out: dict[str, int] = {}
        for v in self.status.values():
            out[v["status"]] = out.get(v["status"], 0) + 1
        return out


def _is_json(body: bytes) -> bool:
    try:
        json.loads(body)
        return True
    except ValueError:
        return False


def _is_xls(body: bytes) -> bool:
    return body[:8] == bytes.fromhex("d0cf11e0a1b11ae1")


def _is_zip(body: bytes) -> bool:
    return body[:4] == b"PK\x03\x04"


RESEARCH_MAP = [   # (glob under the research dl/ folder, cache path; "{name}" keeps the file name)
    ("fred_*/{name}.csv", "fred/{name}.csv"),
    ("fiscal_*/debt_to_penny.json", "fiscal/debt_to_penny.json"),
    ("fiscal_*/dts_operating_cash_balance.json", "fiscal/dts_operating_cash_balance.json"),
    ("fiscal_*/dts_debt_subject_to_limit.json", "fiscal/dts_debt_subject_to_limit.json"),
    ("fiscal_*/mspd_table_1.json", "fiscal/mspd_table_1.json"),
    ("fiscal_*/avg_interest_rates_total_marketable.json", "fiscal/avg_interest_rates_total_marketable.json"),
    ("fiscal_*/mts_table_9_latest.json", "fiscal/mts_table_9_latest.json"),
    ("fed_*/nyfed_soma_summary.json", "nyfed/soma_summary.json"),
    ("tic_*/slt_table5.txt", "tic/slt_table5.txt"),
    ("intl_*/imf_dm_{name}.json", "imf/{name}.json"),
    ("opec_*/wb_oil_rents.json", "worldbank/oil_rents.json"),
    ("eia_*/WCSSTUS1w.xls", "eia/WCSSTUS1w.xls"),
    ("eia_*/WCRFPUS2w.xls", "eia/WCRFPUS2w.xls"),
    ("finra_*/margin-statistics.xlsx", "finra/margin-statistics.xlsx"),
    ("sec_*/{name}.json", "sec/{name}.json"),
]
RESEARCH_FACTORS_MAP = [
    ("fiscaldata/debt_to_penny_*/debt_to_penny.json", "fiscal/debt_to_penny_full.json"),
    ("fiscaldata/mts_table_4_customs_*/mts_table_4_customs.json", "fiscal/mts_table_4_customs.json"),
]
_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def import_research(dl_dir: Path, factors_raw: Path | None = None, dest: Path | None = None, *,
                    force: bool = False) -> dict:
    """Copy the research round's downloads into the cache (bytes only: no file is executed or imported).

    A FRED ``<ID>_long.csv`` window wins over ``<ID>.csv``. An existing different file is kept unless `force`.
    Writes ``import_log.json`` (file, sha256, bytes, source) next to the cache."""
    dl_dir = Path(dl_dir)
    dest = Path(dest) if dest else RAW_AGENTS
    rows, skipped = [], []

    def copy(src: Path, rel: str):
        target = dest / rel
        body = src.read_bytes()
        if target.exists() and target.read_bytes() != body and not force:
            skipped.append({"file": rel, "why": "已有不同的文件（加 --force 覆盖）"})
            return
        _atomic_write(target, body)
        rows.append({"file": rel, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                     "source": str(src.relative_to(src.parents[1])) if len(src.parents) > 1 else src.name})

    for pattern, rel in RESEARCH_MAP:
        if "{name}" in pattern:
            folder_glob, name_pat = pattern.split("/", 1)
            for folder in sorted(dl_dir.glob(folder_glob)):
                prefix, suffix = name_pat.split("{name}")
                for src in sorted(folder.glob(prefix + "*" + suffix)):
                    name = src.name[len(prefix):len(src.name) - len(suffix)]
                    if not _NAME_RE.match(name):
                        continue
                    if name.endswith("_long"):
                        continue
                    long_ = folder / f"{prefix}{name}_long{suffix}"
                    copy(long_ if long_.exists() else src, rel.replace("{name}", name))
        else:
            for src in sorted(dl_dir.glob(pattern)):
                copy(src, rel)
    if factors_raw:
        for pattern, rel in RESEARCH_FACTORS_MAP:
            for src in sorted(Path(factors_raw).glob(pattern)):
                copy(src, rel)
    jsonio.dump({"imported": rows, "skipped": skipped, "at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                dest / "import_log.json")
    return {"imported": len(rows), "skipped": len(skipped), "dest": str(dest)}


# =====================================================================================================================
# 3. The root snapshot (port of LEDGERS l02_snapshot.py)
# =====================================================================================================================

HQ = 8   # quarters of history kept per line


def _r(x, d=3):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, d) if math.isfinite(xf) else None


def _q_end(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.to_period("Q")).last()


def _q_mean(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.to_period("Q")).mean()


def _q_sum(s: pd.Series, min_count: int = 3) -> pd.Series:
    g = s.groupby(s.index.to_period("Q"))
    out, n = g.sum(min_count=1), g.count()
    return out[n >= min_count]


def _yoy(s: pd.Series, periods: int) -> pd.Series:
    return (s / s.shift(periods) - 1.0) * 100.0


def _last(s: pd.Series):
    s = s.dropna()
    if not len(s):
        raise ValueError("空序列")
    return s.index[-1], float(s.iloc[-1])


class _Snap:
    """Collects lines {value, unit, as_of, src, series, zh} and short histories, book by book."""

    def __init__(self):
        self.books: dict[str, dict] = {}
        self.missing: list[dict] = []

    def line(self, book, lid, value, unit, as_of, src, series, zh=None, note=None, digits=3):
        v = _r(value, digits)
        if v is None:
            self.missing.append({"book": book, "line": lid, "why": "数值缺失"})
            return
        self.books.setdefault(book, {"lines": {}, "history": {}})["lines"][lid] = {
            "value": v, "unit": unit, "as_of": str(as_of)[:10], "src": src, "series": series,
            **({"zh": zh} if zh else {}), **({"note": note} if note else {})}

    def hist(self, book, lid, s: pd.Series, digits=3):
        s = s.dropna().tail(HQ)
        self.books.setdefault(book, {"lines": {}, "history": {}})["history"][lid] = {str(k): _r(v, digits)
                                                                                 for k, v in s.items()}

    def get(self, book, lid):
        return self.books[book]["lines"][lid]["value"]


class _Reader:
    """Cached-file readers (all parse data only; nothing is executed)."""

    def __init__(self, fetch: Fetcher):
        self.f = fetch
        self._json: dict = {}

    def fg(self, sid: str) -> pd.Series:
        return self.f.fred(sid)

    def jfile(self, rel: str):
        if rel not in self._json:
            self._json[rel] = self.f.json_source(rel)
        if self._json[rel] is None:
            raise FileNotFoundError(rel)
        return self._json[rel]

    def fiscal(self, name: str) -> list[dict]:
        return self.jfile(f"fiscal/{name}.json")["data"]

    def debt_to_penny_full(self) -> pd.DataFrame:
        rows = []
        for rel in ("fiscal/debt_to_penny_full.json", "fiscal/debt_to_penny.json"):
            try:
                rows += self.jfile(rel)["data"]
            except (FileNotFoundError, KeyError, TypeError):
                continue
        if not rows:
            raise FileNotFoundError("debt_to_penny")
        df = pd.DataFrame(rows).drop_duplicates("record_date").set_index("record_date").sort_index()
        df.index = pd.to_datetime(df.index)
        for c in ("debt_held_public_amt", "intragov_hold_amt", "tot_pub_debt_out_amt"):
            df[c] = pd.to_numeric(df[c].replace("null", np.nan), errors="coerce") / 1e9
        return df[["debt_held_public_amt", "intragov_hold_amt", "tot_pub_debt_out_amt"]]

    def customs_monthly(self) -> pd.DataFrame:
        rows = self.jfile("fiscal/mts_table_4_customs.json")["data"]
        df = pd.DataFrame(rows)
        df = df[df["classification_desc"].str.startswith("Customs")]
        df.index = pd.to_datetime(df["record_date"])
        out = pd.DataFrame({k: pd.to_numeric(df[f"current_month_{k}_amt"].replace("null", np.nan), errors="coerce") / 1e9
                            for k in ("gross_rcpt", "refund", "net_rcpt")})
        return out.sort_index()

    def sec_facts(self, tk: str, concept: str) -> pd.DataFrame | None:
        d = self.f.sec(tk, concept)
        if not d:
            return None
        arr = (d.get("units") or {}).get("USD") or []
        if not arr:
            return None
        df = pd.DataFrame(arr)
        df["end"] = pd.to_datetime(df["end"])
        if "start" in df:
            df["start"] = pd.to_datetime(df["start"])
        return df.sort_values(["end", "filed"])

    def xbrl_quarterly(self, tk: str, concept: str, kind: str = "duration") -> pd.Series:
        """Quarterly values by calendar quarter of the period end. Duration items: a 3-month fact when present,
        else the difference of two year-to-date facts with the same start (10-Q cash-flow statements are YTD)."""
        df = self.sec_facts(tk, concept)
        if df is None:
            return pd.Series(dtype=float)
        if kind == "instant":
            df = df[df["start"].isna()] if "start" in df else df
            s = df.groupby("end")["val"].last()
            s.index = pd.DatetimeIndex(s.index).to_period("Q")
            return s.groupby(level=0).last().astype(float)
        df = df.dropna(subset=["start"]).copy()
        last = df.groupby(["start", "end"])["val"].last()
        vals = {}
        for e in sorted({e for (_, e) in last.index}):
            cands = [(s, v) for (s, ee), v in last.items() if ee == e]
            q = [v for s, v in cands if 80 <= (e - s).days <= 100]
            if q:
                vals[e] = float(q[-1])
                continue
            for s, v in sorted(cands, key=lambda x: x[0]):
                prev = [(ee, vv) for (ss, ee), vv in last.items() if ss == s and 80 <= (e - ee).days <= 100]
                if prev:
                    vals[e] = float(v) - float(prev[-1][1])
                    break
        s = pd.Series(vals, dtype=float).sort_index()
        s.index = pd.DatetimeIndex(s.index).to_period("Q")
        return s.groupby(level=0).last()

    def imf_dm(self, indicator: str) -> dict:
        d = self.jfile(f"imf/{indicator}.json")
        return (d.get("values") or {}).get(indicator, {})

    def wb(self, rel: str) -> pd.DataFrame:
        d = self.jfile(rel)
        rows = d[1] if isinstance(d, list) and len(d) > 1 and d[1] else []
        return pd.DataFrame([{"country": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]} for r in rows])

    def tic_table5(self) -> pd.DataFrame:
        p = self.f.file_source("tic/slt_table5.txt")
        if p is None:
            raise FileNotFoundError("tic/slt_table5.txt")
        txt = p.read_text(encoding="utf-8", errors="replace")
        rows = [ln for ln in txt.splitlines() if ln.strip()]
        hdr = next(i for i, ln in enumerate(rows) if ln.startswith("Country"))
        cols = rows[hdr].split("\t")
        data = []
        for ln in rows[hdr + 1:]:
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            vals = []
            for x in parts[1:len(cols)]:
                try:
                    vals.append(float(x))
                except ValueError:
                    vals.append(np.nan)
            data.append([parts[0].strip()] + vals)
        width = max(len(r) for r in data)
        data = [r + [np.nan] * (width - len(r)) for r in data]
        return pd.DataFrame(data, columns=["country"] + [c.strip() for c in cols[1:width]]).set_index("country")

    def eia_weekly(self, name: str) -> pd.Series:
        p = self.f.file_source(f"eia/{name}")
        if p is None:
            raise FileNotFoundError(name)
        x = pd.read_excel(p, sheet_name="Data 1", header=2, engine="xlrd").iloc[:, :2].dropna()
        x.columns = ["date", "value"]
        return pd.Series(pd.to_numeric(x["value"], errors="coerce").values, index=pd.to_datetime(x["date"])).dropna().sort_index()

    def finra_margin(self) -> pd.DataFrame:
        p = self.f.file_source("finra/margin-statistics.xlsx")
        if p is None:
            raise FileNotFoundError("finra")
        return pd.read_excel(p, engine="openpyxl")

    def soma_summary(self) -> pd.DataFrame:
        d = self.jfile("nyfed/soma_summary.json")
        df = pd.DataFrame((d.get("soma") or {}).get("summary") or [])
        df.index = pd.to_datetime(df.pop("asOfDate"))
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_index()

    def panel(self) -> pd.DataFrame:
        if not config.PANEL_PATH.exists():
            raise FileNotFoundError(config.PANEL_PATH.name)
        P = pd.read_csv(config.PANEL_PATH)
        P.index = pd.to_datetime(P.pop("date"))
        return P


def _section(snap: _Snap, name: str, fn, *args) -> None:
    try:
        fn(snap, *args)
    except (FileNotFoundError, KeyError, ValueError, IndexError, TypeError, StopIteration) as err:
        snap.missing.append({"book": name, "line": "*", "why": f"{type(err).__name__}: {str(err)[:120]}"})
        log.warning("accounts root: section %s incomplete (%s)", name, err)


def _fed(S: _Snap, R: _Reader) -> None:
    bil = 1e3
    walcl, treast, mbs = R.fg("WALCL") / bil, R.fg("TREAST") / bil, R.fg("WSHOMCB") / bil
    curr, res, rrp, tga = R.fg("WCURCIR") / bil, R.fg("WRESBAL") / bil, R.fg("WLRRAL") / bil, R.fg("WDTGAL") / bil
    onrrp, remit = R.fg("RRPONTSYD"), R.fg("RESPPLLOPNWW") / bil
    d, v = _last(walcl)
    S.line("fed", "total_assets", v, "$bn", d, "FRED_H41", "WALCL", "总资产")
    S.line("fed", "soma_treasuries", _last(treast)[1], "$bn", d, "FRED_H41", "TREAST", "持有国债")
    S.line("fed", "soma_mbs", _last(mbs)[1], "$bn", d, "FRED_H41", "WSHOMCB", "持有 MBS")
    S.line("fed", "currency", _last(curr)[1], "$bn", d, "FRED_H41", "WCURCIR (week avg)", "流通中货币")
    S.line("fed", "reserves", _last(res)[1], "$bn", d, "FRED_H41", "WRESBAL (week avg)", "银行准备金")
    S.line("fed", "reverse_repo_total", _last(rrp)[1], "$bn", d, "FRED_H41", "WLRRAL", "逆回购（含外国官方）")
    S.line("fed", "on_rrp", _last(onrrp)[1], "$bn", _last(onrrp)[0], "FRED_NYFED", "RRPONTSYD", "隔夜逆回购")
    S.line("fed", "tga", _last(tga)[1], "$bn", d, "FRED_H41", "WDTGAL", "财政部一般账户（TGA）")
    S.line("fed", "deferred_asset", -_last(remit)[1], "$bn", d, "FRED_H41", "-RESPPLLOPNWW", "递延资产（累计未弥补亏损）")
    other = _last(walcl)[1] - (_last(curr)[1] + _last(res)[1] + _last(rrp)[1] + _last(tga)[1] + _last(remit)[1])
    S.line("fed", "other_liabilities_capital_resid", other, "$bn", d, "identity",
           "WALCL - (currency + reserves + RRP + TGA + remittances due)", "其他负债与资本（残差）")
    for k, s in [("total_assets", walcl), ("soma_treasuries", treast), ("soma_mbs", mbs), ("reserves", res),
                 ("reverse_repo_total", rrp), ("tga", tga), ("deferred_asset", -remit)]:
        S.hist("fed", k, _q_end(s))
    da = _q_end(-remit)
    ni = -(da.diff())
    S.hist("fed", "net_income_q", ni)
    S.line("fed", "net_income_last4q", ni.dropna().tail(4).sum(), "$bn/4q", str(ni.dropna().index[-1]), "derived",
           "-Δ deferred asset, last 4 quarters", "近4季净收入")
    S.line("fed", "deferred_asset_peak", da.max(), "$bn", str(da.idxmax()), "derived", "max of -RESPPLLOPNWW", "递延资产峰值")
    iorb, rrpr, dff = R.fg("IORB"), R.fg("RRPONTSYAWARD"), R.fg("DFF")
    S.line("fed", "iorb", _last(iorb)[1], "%", _last(iorb)[0], "FRED_FED", "IORB", "准备金利率")
    S.line("fed", "on_rrp_rate", _last(rrpr)[1], "%", _last(rrpr)[0], "FRED_NYFED", "RRPONTSYAWARD", "隔夜逆回购利率")
    S.line("fed", "effr", _last(dff)[1], "%", _last(dff)[0], "FRED_NYFED", "DFF", "有效联邦基金利率")
    try:
        pre = float(iorb.loc[:"2026-10-27"].iloc[-1])
        post = float(iorb.loc["2026-10-30":].iloc[0])
        S.line("fed", "iorb_change_at_oct28_meeting", post - pre, "pp", "2026-10-30", "derived",
               "IORB 2026-10-30 minus 2026-10-27", "10月28日议息会议前后准备金利率变化",
               note="0 表示目标区间维持在 3.75–4.00%")
    except IndexError:
        pass
    S.line("fed", "interest_expense_runrate_q", (_last(res)[1] * _last(iorb)[1] + _last(rrp)[1] * _last(rrpr)[1]) / 400,
           "$bn/q", d, "derived", "(reserves*IORB + RRP*RRP rate)/4", "当前负债的季度利息支出")
    so = R.soma_summary()
    sd = so.index[-1]
    S.line("fed", "soma_bills", so["bills"].iloc[-1] / 1e9, "$bn", sd, "NYFED_SOMA", "summary.bills", "其中国库券（面值）")
    if len(so) > 14:
        S.line("fed", "soma_bills_change_13w", (so["bills"].iloc[-1] - so["bills"].iloc[-14]) / 1e9, "$bn/13w", sd,
               "NYFED_SOMA", "Δ bills over 13 weeks", "近13周国库券增持（准备金管理购买）")
        S.line("fed", "soma_mbs_change_13w", (so["mbs"].iloc[-1] - so["mbs"].iloc[-14]) / 1e9, "$bn/13w", sd,
               "NYFED_SOMA", "Δ MBS over 13 weeks", "近13周 MBS 变化（到期滚落）")
    pdt, pv = _last(_yoy(R.fg("PCEPILFE"), 12))
    S.line("fed", "core_pce_yoy", pv, "%", pdt, "FRED_BEA", "PCEPILFE", "核心 PCE 同比")
    S.line("fed", "inflation_gap", pv - 2.0, "pp", pdt, "derived", "core PCE yoy - 2", "通胀缺口")
    udt, uv = _last(R.fg("UNRATE"))
    nrou = R.fg("NROU")
    nv = float(nrou.loc[:pd.Timestamp(udt)].iloc[-1])
    S.line("fed", "unemployment", uv, "%", udt, "FRED_BLS", "UNRATE", "失业率")
    S.line("fed", "nrou_cbo", nv, "%", str(nrou.loc[:pd.Timestamp(udt)].index[-1]), "FRED_CBO", "NROU", "CBO 自然失业率")
    S.line("fed", "unemployment_gap", uv - nv, "pp", udt, "derived", "UNRATE - NROU", "就业缺口")


def _fiscal(S: _Snap, R: _Reader) -> None:
    rows = R.fiscal("dts_debt_subject_to_limit")
    latest = max(x["record_date"] for x in rows)
    rr = [x for x in rows if x["record_date"] == latest]

    def get(cat):
        return sum(float(x["close_today_bal"]) for x in rr if x["debt_catg"] == cat) / 1e3
    pub, intra = get("Debt Held by the Public"), get("Intragovernmental Holdings")
    not_subj, guar, limit = get("Debt Not Subject to Limit"), get("Other Debt Subject to Limit"), get("Statutory Debt Limit")
    subj = pub + intra - not_subj + guar
    S.line("fiscal", "debt_held_public", pub, "$bn", latest, "FD_DTS", "Table III-C: Debt Held by the Public", "公众持有债务")
    S.line("fiscal", "intragov_holdings", intra, "$bn", latest, "FD_DTS", "Table III-C", "政府内部持有")
    S.line("fiscal", "not_subject_to_limit", not_subj - guar, "$bn", latest, "FD_DTS",
           "not subject - other subject", "不受上限约束的债务（净）")
    S.line("fiscal", "debt_subject_to_limit", subj, "$bn", latest, "FD_DTS", "public + intragov - not subject + guaranteed",
           "受上限约束的债务")
    S.line("fiscal", "statutory_limit", limit, "$bn", latest, "FD_DTS", "Table III-C: Statutory Debt Limit", "法定债务上限")
    S.line("fiscal", "limit_headroom", limit - subj, "$bn", latest, "derived", "limit - subject", "上限余量")
    try:
        dtp = R.debt_to_penny_full()
        tot = dtp["tot_pub_debt_out_amt"].dropna()
        for m in (3, 6, 12):
            a = float(tot.loc[:pd.Timestamp(latest) - pd.DateOffset(months=m)].iloc[-1])
            pace = (float(tot.iloc[-1]) - a) / m
            S.line("fiscal", f"debt_pace_{m}m", pace, "$bn/month", latest, "FD_D2P", f"Δ total public debt over {m} months / {m}",
                   f"近{m}个月月均新增债务")
            if pace > 0:
                S.line("fiscal", f"months_to_limit_at_{m}m_pace", (limit - subj) / pace, "months", latest, "derived",
                       "headroom / pace", f"按近{m}月速度触及上限的月数")
        ig = dtp["intragov_hold_amt"].dropna()
        if len(ig) > 260:
            S.line("fiscal", "intragov_growth_q", (float(ig.iloc[-1]) - float(ig.loc[:ig.index[-1] - pd.DateOffset(months=12)]
                                                                               .iloc[-1])) / 4, "$bn/q", latest,
                   "FD_D2P", "Δ intragovernmental over 12 months / 4", "政府内部持有每季增长")
    except (FileNotFoundError, IndexError, KeyError):
        S.missing.append({"book": "fiscal", "line": "debt_pace", "why": "Debt to the Penny 缺失"})
    ocb = R.fiscal("dts_operating_cash_balance")
    lt = max(x["record_date"] for x in ocb)
    tga = [x for x in ocb if x["record_date"] == lt and "Closing Balance" in x["account_type"]]
    if tga:
        val = tga[0].get("open_today_bal") or tga[0].get("close_today_bal")
        S.line("fiscal", "tga_dts", float(val) / 1e3, "$bn", lt, "FD_DTS", "Operating cash balance, TGA closing", "TGA 现金余额")
    rec, outl, dfc = R.fg("MTSR133FMS") / 1e3, R.fg("MTSO133FMS") / 1e3, R.fg("MTSDS133FMS") / 1e3
    fy_end = pd.Timestamp(f"{rec.index.max().year if rec.index.max().month >= 9 else rec.index.max().year - 1}-09-01")
    fy = slice(fy_end - pd.DateOffset(months=11), fy_end)
    S.line("fiscal", "receipts_fy", rec.loc[fy].sum(), "$bn", str(fy_end.date()), "FRED_MTS", "MTSR133FMS", "上一财年收入")
    S.line("fiscal", "outlays_fy", outl.loc[fy].sum(), "$bn", str(fy_end.date()), "FRED_MTS", "MTSO133FMS", "上一财年支出")
    S.line("fiscal", "deficit_fy", -dfc.loc[fy].sum(), "$bn", str(fy_end.date()), "FRED_MTS", "-MTSDS133FMS", "上一财年赤字")
    rq = _q_sum(rec)
    S.line("fiscal", "receipts_q", float(rq.iloc[-1]), "$bn/q", str(rq.index[-1]), "FRED_MTS", "MTSR133FMS (quarter sum)",
           "上季收入")
    S.hist("fiscal", "receipts_q", rq)
    S.hist("fiscal", "outlays_q", _q_sum(outl))
    S.hist("fiscal", "deficit_q", -_q_sum(dfc))
    try:
        t9 = R.fiscal("mts_table_9_latest")
        last9 = max(x["record_date"] for x in t9)
        ni = [x for x in t9 if x["record_date"] == last9 and x["classification_desc"] == "Net Interest"][0]
        S.line("fiscal", "net_interest_fy", float(ni["current_fytd_rcpt_outly_amt"]) / 1e9, "$bn", last9, "FD_MTS",
               "MTS Table 9: Net Interest", "本财年至今净利息")
        S.line("fiscal", "net_interest_prior_fy", float(ni["prior_fytd_rcpt_outly_amt"]) / 1e9, "$bn", last9, "FD_MTS",
               "MTS Table 9: Net Interest (prior FYTD)", "上年同期净利息")
        S.line("fiscal", "net_interest_share_receipts", float(ni["current_fytd_rcpt_outly_amt"]) / 1e9 / rec.loc[fy].sum() * 100,
               "%", last9, "derived", "net interest / receipts", "净利息占收入")
    except (FileNotFoundError, IndexError, KeyError, ValueError):
        S.missing.append({"book": "fiscal", "line": "net_interest", "why": "MTS 表 9 缺失"})
    gdp, nipa_int = R.fg("GDP"), R.fg("A091RC1Q027SBEA")
    g_dt, g_v = _last(gdp)
    S.line("macro", "gdp_saar", g_v, "$bn SAAR", g_dt, "FRED_BEA", "GDP", "名义 GDP")
    S.line("fiscal", "nipa_interest_saar", float(nipa_int.iloc[-1]), "$bn SAAR", nipa_int.index[-1], "FRED_BEA",
           "A091RC1Q027SBEA", "NIPA 联邦利息支出")
    S.line("fiscal", "interest_burden_nipa", float(nipa_int.iloc[-1]) / float(gdp.loc[nipa_int.index[-1]]) * 100, "% GDP",
           nipa_int.index[-1], "FRED_BEA", "A091RC1Q027SBEA / GDP", "联邦利息支出/GDP（NIPA）")
    S.hist("fiscal", "interest_burden_nipa", (nipa_int / gdp * 100).dropna())
    fg = R.fg("FGDEF")
    S.line("fiscal", "deficit_nipa_gdp", -float(fg.iloc[-1]) / float(gdp.loc[fg.index[-1]]) * 100, "% GDP", fg.index[-1],
           "FRED_BEA", "-FGDEF / GDP", "联邦净借款/GDP（NIPA，模型口径）")
    dhp_q = R.fg("FYGFDPUN") / 1e3
    S.line("fiscal", "debt_held_public_nipa_q", float(dhp_q.loc[:nipa_int.index[-1]].iloc[-1]), "$bn", nipa_int.index[-1],
           "FRED_FISCAL", "FYGFDPUN", "公众持有债务（NIPA 季度）")
    S.hist("fiscal", "debt_held_public_q", _q_end(dhp_q))
    ar = R.fiscal("avg_interest_rates_total_marketable")
    ars = pd.Series({x["record_date"]: float(x["avg_interest_rate_amt"]) for x in ar}).sort_index()
    ars.index = pd.to_datetime(ars.index)
    S.line("fiscal", "avg_rate_marketable", float(ars.iloc[-1]), "%", ars.index[-1], "FD_AVG",
           "avg_interest_rates: Total Marketable", "可交易国债平均利率")
    S.line("fiscal", "avg_rate_marketable_12m_ago", float(ars.loc[:ars.index[-1] - pd.DateOffset(months=12)].iloc[-1]), "%",
           ars.index[-1] - pd.DateOffset(months=12), "FD_AVG", "same, 12 months earlier", "一年前")
    m1 = R.fiscal("mspd_table_1")
    lm = max(x["record_date"] for x in m1)
    M = {x["security_class_desc"]: float(x["total_mil_amt"]) / 1e3 for x in m1
         if x["record_date"] == lm and x["security_type_desc"] == "Marketable"}
    tot = M.get("Total Marketable") or sum(M.values())
    for k in ("Bills", "Notes", "Bonds", "Treasury Inflation-Protected Securities", "Floating Rate Notes"):
        if k in M:
            S.line("fiscal", "mspd_" + k.split()[0].lower(), M[k], "$bn", lm, "FD_MSPD", f"MSPD table 1: {k}", f"可交易国债：{k}")
    if "Bills" in M and tot:
        S.line("fiscal", "bill_share", M["Bills"] / tot * 100, "%", lm, "derived", "bills / total marketable", "短期国库券占比")
    try:
        cm = R.customs_monthly()
        lc = cm.index[-1]
        S.line("fiscal", "customs_gross_3m", cm["gross_rcpt"].tail(3).sum(), "$bn/3m", lc, "FD_MTS4", "Customs duties gross",
               "关税毛收入（近3月）")
        S.line("fiscal", "customs_refunds_since_2026_05", cm.loc["2026-05-01":, "refund"].sum(), "$bn", lc, "FD_MTS4",
               "Customs duties refunds", "2026年5月以来关税退款合计",
               note="退款从每月约 10–20 亿美元跳到 2026 年 5 月 220 亿、6 月 492 亿；法律原因是锁定事实")
        S.hist("fiscal", "customs_net_q", _q_sum(cm["net_rcpt"]))
    except (FileNotFoundError, KeyError, IndexError):
        S.missing.append({"book": "fiscal", "line": "customs", "why": "MTS 表 4 缺失"})
    cu, im = R.fg("B235RC1Q027SBEA"), R.fg("A255RC1Q027SBEA")
    S.line("fiscal", "customs_nipa_saar", _last(cu)[1], "$bn SAAR", _last(cu)[0], "FRED_BEA", "B235RC1Q027SBEA", "关税（NIPA 年化）")
    S.line("fiscal", "goods_imports_saar", _last(im)[1], "$bn SAAR", _last(im)[0], "FRED_BEA", "A255RC1Q027SBEA", "商品进口（年化）")
    tr = (100 * cu / im).dropna()
    S.line("president", "effective_tariff_rate", _last(tr)[1], "%", _last(tr)[0], "FRED_BEA",
           "B235RC1Q027SBEA / A255RC1Q027SBEA", "有效关税率")
    S.hist("president", "effective_tariff_rate", _q_end(tr), 2)
    S.line("fiscal", "debt_gdp", float(R.fg("FYGFGDQ188S").iloc[-1]), "% GDP", R.fg("FYGFGDQ188S").index[-1], "FRED_FISCAL",
           "FYGFGDQ188S", "公众持有债务/GDP")
    t5 = R.tic_table5()
    gt = [i for i in t5.index if i.lower().startswith("grand total")]
    if gt:
        S.line("fiscal", "foreign_holdings_tic", float(t5.loc[gt[0]].iloc[0]), "$bn", t5.columns[0], "TIC_T5",
               "Table 5 Grand Total", "外国持有美债（TIC 表 5）")
    fo = [i for i in t5.index if "official" in i.lower()]
    if fo:
        S.line("fiscal", "foreign_official_holdings", float(t5.loc[fo[0]].iloc[0]), "$bn", t5.columns[0], "TIC_T5",
               f"Table 5: {fo[0]}", "其中官方")
    if "China, Mainland" in t5.index:
        cn = t5.loc["China, Mainland"]
        S.line("china", "us_treasury_holdings", float(cn.iloc[0]), "$bn", cn.index[0], "TIC_T5", "Table 5: China, Mainland",
               "持有美债")
        if len(cn) > 12 and pd.notna(cn.iloc[12]):
            S.line("china", "us_treasury_holdings_change_12m", float(cn.iloc[0] - cn.iloc[12]), "$bn", cn.index[0], "TIC_T5",
                   "Δ12m", "近12月变化")
    for sid, k, zh in [("W994RC1Q027SBEA", "nl_private", "私人部门净借出"), ("AD01RC1Q027SBEA", "nl_government", "政府部门净借出"),
                       ("AD02RC1Q027SBEA", "nl_federal", "联邦政府净借出"),
                       ("W162RC1Q027SBEA", "nl_nation_row", "国家对外净借出（≈经常账户）"),
                       ("W995RC1Q027SBEA", "nl_business", "企业部门净借出")]:
        s = R.fg(sid)
        S.line("sectoral", k, float(s.iloc[-1]), "$bn SAAR", s.index[-1], "FRED_BEA", sid, zh)
        S.line("sectoral", k + "_gdp", float(s.iloc[-1]) / float(gdp.loc[s.index[-1]]) * 100, "% GDP", s.index[-1], "derived",
               f"{sid} / GDP", zh + "/GDP")
    L = S.books["sectoral"]["lines"]
    S.line("sectoral", "identity_residual", L["nl_private"]["value"] + L["nl_government"]["value"] - L["nl_nation_row"]["value"],
           "$bn SAAR", L["nl_private"]["as_of"], "identity", "private + government - nation", "部门恒等式残差")


def _macro(S: _Snap, R: _Reader) -> None:
    for sid, k, zh in (("DGS2", "y2", "2 年期国债收益率（月末）"), ("DGS10", "y10", "10 年期国债收益率（月末）")):
        s = R.fg(sid)
        S.line("macro", k, _last(s)[1], "%", _last(s)[0], "FRED_TREASURY", f"{sid} month-end", zh)
    y10 = R.fg("DGS10")
    q = _q_mean(y10).dropna().tail(20)
    S.books.setdefault("macro", {"lines": {}, "history": {}})["history"]["y10_q_20"] = {str(k): _r(v, 4) for k, v in q.items()}
    ps = R.fg("PSAVERT")
    S.line("macro", "psavert_p1_since_1959", float(np.percentile(ps.loc["1959":].values, 1)), "%", str(ps.index[-1])[:10],
           "FRED_BEA", "PSAVERT 1st percentile since 1959", "储蓄率 1959 年以来第 1 百分位")


def _president(S: _Snap, R: _Reader) -> None:
    gas = R.fg("GASREGW")
    S.line("president", "gasoline_price", _last(gas)[1], "$/gal", _last(gas)[0], "FRED_EIA", "GASREGW", "汽油零售价")
    S.hist("president", "gasoline_price", _q_mean(gas), 2)
    try:
        P = R.panel()
        cpi = P["s_cpi_yoy"].dropna()
        S.line("president", "cpi_yoy", float(cpi.iloc[-1]), "%", cpi.index[-1], "MAIN_PANEL", "s_cpi_yoy", "CPI 同比")
        px = (1 + P["p_us_equity"].fillna(0)).cumprod()
        dd = (px / px.cummax() - 1) * 100
        S.line("president", "us_equity_drawdown", float(dd.iloc[-1]), "%", dd.index[-1], "MAIN_PANEL", "p_us_equity cumulative",
               "美股距高点回撤")
    except (FileNotFoundError, KeyError, IndexError):
        S.missing.append({"book": "president", "line": "cpi_yoy/us_equity_drawdown", "why": "项目面板缺失"})
    sent = R.fg("UMCSENT")
    S.line("president", "consumer_sentiment", _last(sent)[1], "index", _last(sent)[0], "FRED_UMICH", "UMCSENT", "消费者信心")
    d_, v_ = _last(_yoy(R.fg("A229RX0"), 12))
    S.line("president", "real_dpi_pc_yoy", v_, "%", d_, "FRED_BEA", "A229RX0", "人均实际可支配收入同比")
    spr = R.eia_weekly("WCSSTUS1w.xls") / 1e3
    S.line("president", "spr_inventory", _last(spr)[1], "mn bbl", _last(spr)[0], "EIA_WPSR", "WCSSTUS1", "战略石油储备库存")
    if len(spr) > 53:
        S.line("president", "spr_change_52w", float(spr.iloc[-1] - spr.iloc[-53]), "mn bbl", _last(spr)[0], "EIA_WPSR", "Δ52w",
               "近一年变化")


def _china(S: _Snap, R: _Reader) -> None:
    res = R.fg("TRESEGCNM052N") / 1e3
    d_, v_ = _last(res)
    S.line("china", "fx_reserves_ex_gold", v_, "$bn", d_, "FRED_IMF_IFS", "TRESEGCNM052N", "外汇储备（不含黄金）")
    S.line("china", "fx_reserves_change_12m", v_ - float(res.loc[:d_ - pd.DateOffset(months=12)].iloc[-1]), "$bn", d_,
           "FRED_IMF_IFS", "Δ12m", "近12月变化")
    S.hist("china", "fx_reserves_ex_gold", _q_end(res), 1)
    for ind, k, zh in [("GGXCNL_NGDP", "gg_net_lending_gdp", "广义政府净借出/GDP（IMF 官方口径）"),
                       ("GGXWDG_NGDP", "gg_gross_debt_gdp", "广义政府总债务/GDP"),
                       ("BCA_NGDPD", "current_account_gdp", "经常账户/GDP"), ("NGDP_RPCH", "real_gdp_growth", "实际 GDP 增长")]:
        try:
            v = R.imf_dm(ind).get("CHN", {})
        except FileNotFoundError:
            S.missing.append({"book": "china", "line": k, "why": f"IMF {ind} 缺失"})
            continue
        for y in ("2025", "2026"):
            if y in v:
                S.line("china", f"{k}_{y}", v[y], "%", f"{y}-12-31", "IMF_WEO_DM", f"{ind}/CHN/{y}",
                       f"{zh}（{y}{'，WEO 预测' if y == '2026' else ''}）")
    cr = R.fg("QCNPAM770A")
    S.line("china", "private_credit_gdp", _last(cr)[1], "% GDP", _last(cr)[0], "FRED_BIS", "QCNPAM770A", "私人非金融部门信贷/GDP")
    if len(cr) > 4:
        S.line("china", "private_credit_gdp_change_4q", float(cr.iloc[-1] - cr.iloc[-5]), "pp", _last(cr)[0], "FRED_BIS", "Δ4q",
               "近4季变化")
    hp = R.fg("QCNR628BIS")
    S.line("china", "real_house_price_vs_peak", (float(hp.iloc[-1]) / float(hp.max()) - 1) * 100, "%", _last(hp)[0],
           "FRED_BIS", "QCNR628BIS", "实际住宅价格距峰值")
    imp, exp_ = R.fg("IMPCH") / 1e3, R.fg("EXPCH") / 1e3
    S.line("china", "us_imports_from_china_12m", imp.tail(12).sum(), "$bn", imp.index[-1], "FRED_CENSUS", "IMPCH", "美国自华进口（12月）")
    S.line("china", "us_exports_to_china_12m", exp_.tail(12).sum(), "$bn", exp_.index[-1], "FRED_CENSUS", "EXPCH", "美国对华出口（12月）")
    fx = R.fg("DEXCHUS")
    S.line("china", "cny_per_usd", _last(fx)[1], "CNY/USD", _last(fx)[0], "FRED_FED_H10", "DEXCHUS", "人民币兑美元")


def _opec(S: _Snap, R: _Reader) -> None:
    sa = R.fg("TRESEGSAM052N") / 1e3
    S.line("opec", "saudi_reserves_ex_gold", _last(sa)[1], "$bn", _last(sa)[0], "FRED_IMF_IFS", "TRESEGSAM052N", "沙特储备（不含黄金）")
    S.line("opec", "saudi_reserves_change_12m", float(sa.iloc[-1] - sa.loc[:_last(sa)[0] - pd.DateOffset(months=12)].iloc[-1]),
           "$bn", _last(sa)[0], "FRED_IMF_IFS", "Δ12m", "近12月变化")
    for ind, k, zh in [("GGXCNL_NGDP", "gg_net_lending_gdp", "财政余额/GDP"), ("BCA_NGDPD", "current_account_gdp", "经常账户/GDP"),
                       ("GGXWDG_NGDP", "gg_gross_debt_gdp", "政府债务/GDP"), ("NGDP_RPCH", "real_gdp_growth", "实际增长")]:
        try:
            vals = R.imf_dm(ind)
        except FileNotFoundError:
            S.missing.append({"book": "opec", "line": k, "why": f"IMF {ind} 缺失"})
            continue
        for c in ("SAU", "ARE", "KWT", "IRQ", "RUS"):
            v = vals.get(c, {})
            if "2026" in v:
                S.line("opec", f"{c.lower()}_{k}_2026", v["2026"], "%", "2026-12-31", "IMF_WEO_DM", f"{ind}/{c}/2026",
                       f"{c} {zh}（2026，WEO 预测）")
    try:
        w = R.wb("worldbank/oil_rents.json")
        for c in ("SAU", "IRQ", "KWT", "ARE", "RUS"):
            x = w[(w.country == c) & w.value.notna()].sort_values("year")
            if len(x):
                S.line("opec", f"{c.lower()}_oil_rents_gdp", float(x.value.iloc[-1]), "% GDP", f"{int(x.year.iloc[-1])}-12-31",
                       "WB_WDI", f"NY.GDP.PETR.RT.ZS/{c}", f"{c} 石油租金/GDP")
    except (FileNotFoundError, KeyError, AttributeError):
        S.missing.append({"book": "opec", "line": "oil_rents", "why": "World Bank 缺失"})
    br = R.fg("DCOILBRENTEU")
    S.line("opec", "brent_spot", _last(br)[1], "$/bbl", _last(br)[0], "FRED_EIA", "DCOILBRENTEU", "布伦特现货")
    S.hist("opec", "brent_quarterly_avg", _q_mean(R.fg("POILBREUSDM")), 1)
    us = R.eia_weekly("WCRFPUS2w.xls") / 1e3
    S.line("opec", "us_crude_production", _last(us)[1], "mb/d", _last(us)[0], "EIA_WPSR", "WCRFPUS2", "美国原油产量（页岩竞争者）")
    if len(us) > 53:
        S.line("opec", "us_crude_production_change_52w", float(us.iloc[-1] - us.iloc[-53]), "mb/d", _last(us)[0], "EIA_WPSR",
               "Δ52w", "近一年变化")


def _bigtech(S: _Snap, R: _Reader) -> None:
    agg, capex = {}, None
    for tk, m in BT.items():
        q = {"revenue": R.xbrl_quarterly(tk, m["rev"]), "operating_income": R.xbrl_quarterly(tk, "OperatingIncomeLoss"),
             "cfo": R.xbrl_quarterly(tk, "NetCashProvidedByUsedInOperatingActivities"),
             "capex": R.xbrl_quarterly(tk, m["capex"]), "lease_principal": R.xbrl_quarterly(tk, "FinanceLeasePrincipalPayments"),
             "buybacks": R.xbrl_quarterly(tk, "PaymentsForRepurchaseOfCommonStock"),
             "dividends": R.xbrl_quarterly(tk, m["div"]) if m["div"] else pd.Series(dtype=float),
             "depreciation": R.xbrl_quarterly(tk, m["dep"])}
        inst = {"cash": R.xbrl_quarterly(tk, "CashAndCashEquivalentsAtCarryingValue", "instant")}
        sec = pd.Series(dtype=float)
        for c in m["sec"]:
            sec = sec.add(R.xbrl_quarterly(tk, c, "instant"), fill_value=0)
        inst["securities"] = sec
        debt = pd.Series(dtype=float)
        for c in m["debt"]:
            debt = debt.add(R.xbrl_quarterly(tk, c, "instant"), fill_value=0)
        inst["debt"] = debt
        inst["rpo"] = R.xbrl_quarterly(tk, "RevenueRemainingPerformanceObligation", "instant")
        frame = pd.DataFrame({k: v / 1e9 for k, v in {**q, **inst}.items()}).sort_index()
        if frame.empty or "capex" not in frame or frame["capex"].dropna().empty:
            S.missing.append({"book": f"bigtech_{tk.lower()}", "line": "*", "why": "XBRL 缺失"})
            continue
        for k in ("cash", "securities", "debt", "rpo"):
            if k in frame:
                frame[k] = frame[k].ffill()
        cx = frame["capex"].dropna()
        cx = cx.loc[pd.Period("2017Q3"):]
        capex = cx if capex is None else capex.add(cx, fill_value=0.0)
        frame = frame.loc[pd.Period("2024Q1"):]
        frame["fcf"] = frame["cfo"] - frame["capex"] - frame["lease_principal"].fillna(0)
        frame["capex_to_cfo"] = frame["capex"] / frame["cfo"]
        frame["op_margin"] = frame["operating_income"] / frame["revenue"] * 100
        lastq = frame.dropna(subset=["capex", "cfo"]).index[-1]
        f = frame.loc[lastq]
        led = f"bigtech_{tk.lower()}"
        for k in ("revenue", "operating_income", "cfo", "capex", "lease_principal", "fcf", "buybacks", "dividends", "cash",
                  "securities", "debt", "rpo", "depreciation", "op_margin", "capex_to_cfo"):
            if k in f and pd.notna(f[k]):
                S.line(led, k, f[k], "%" if k == "op_margin" else ("ratio" if k == "capex_to_cfo" else "$bn"), lastq,
                       "SEC_XBRL", f"{tk}:{k}")
        for k in ("capex", "fcf", "cfo", "debt", "buybacks", "operating_income"):
            S.hist(led, k, frame[k], 2)
        agg[tk] = frame
    if not agg:
        raise FileNotFoundError("SEC XBRL")
    keys = ["revenue", "operating_income", "cfo", "capex", "lease_principal", "fcf", "buybacks", "dividends", "cash",
            "securities", "debt", "depreciation"]
    A = sum(agg[tk][keys].fillna(0) for tk in agg)
    lastq = min(agg[tk].dropna(subset=["capex", "cfo"]).index[-1] for tk in agg)
    A = A.loc[:lastq]
    for k in keys:
        S.hist("bigtech", k, A[k], 1)
        S.line("bigtech", k, A.loc[lastq, k], "$bn", str(lastq), "SEC_XBRL", f"sum of {len(agg)} companies: {k}")
    S.line("bigtech", "capex_ttm", A["capex"].tail(4).sum(), "$bn/4q", str(lastq), "SEC_XBRL", "sum of last 4 quarters", "资本开支（近4季）")
    S.line("bigtech", "capex_ttm_prev", A["capex"].iloc[-8:-4].sum(), "$bn/4q", str(lastq - 4), "SEC_XBRL", "4 quarters a year earlier",
           "一年前")
    S.line("bigtech", "fcf_ttm", A["fcf"].tail(4).sum(), "$bn/4q", str(lastq), "SEC_XBRL", "CFO - capex - lease principal", "自由现金流（近4季）")
    S.line("bigtech", "payouts_ttm", (A["buybacks"] + A["dividends"]).tail(4).sum(), "$bn/4q", str(lastq), "SEC_XBRL",
           "buybacks + dividends", "股东回报（近4季）")
    S.line("bigtech", "debt_change_4q", A["debt"].iloc[-1] - A["debt"].iloc[-5], "$bn", str(lastq), "SEC_XBRL", "Δ debt over 4 quarters",
           "近4季债务增加")
    S.line("bigtech", "net_cash", A["cash"].iloc[-1] + A["securities"].iloc[-1] - A["debt"].iloc[-1], "$bn", str(lastq), "SEC_XBRL",
           "cash + securities - debt", "净现金")
    S.line("bigtech", "revenue_growth_4q", (A["revenue"].iloc[-1] / A["revenue"].iloc[-5] - 1) * 100, "%", str(lastq), "SEC_XBRL",
           "revenue / revenue 4 quarters earlier - 1", "收入同比")
    if capex is not None:
        capex = capex.loc[:lastq]
        S.books["bigtech"]["capex_history"] = {"quarters": [str(q) for q in capex.index],
                                               "values": [_r(v, 3) for v in capex.values]}


def _households(S: _Snap, R: _Reader) -> None:
    nw, eq, mf = R.fg("TNWBSHNO") / 1e3, R.fg("HNOCEAQ027S") / 1e3, R.fg("HNOMFAQ027S") / 1e3
    tfa, re_, liab, debt = R.fg("TFAABSHNO") / 1e3, R.fg("HNOREMQ027S") / 1e3, R.fg("TLBSHNO") / 1e3, R.fg("CMDEBT") / 1e3
    dt_ = nw.index[-1]
    for k, s, sid, zh in (("net_worth", nw, "TNWBSHNO", "净资产"), ("equities_direct", eq, "HNOCEAQ027S", "直接持有股票"),
                          ("mutual_funds", mf, "HNOMFAQ027S", "共同基金份额"), ("financial_assets", tfa, "TFAABSHNO", "金融资产"),
                          ("real_estate", re_, "HNOREMQ027S", "房地产（市值）"), ("liabilities", liab, "TLBSHNO", "负债"),
                          ("debt", debt, "CMDEBT", "债务（证券与贷款）")):
        S.line("households", k, s.loc[:dt_].iloc[-1], "$bn", dt_, "FRED_Z1", sid, zh)
    share = ((eq + mf) / tfa * 100).dropna()
    S.line("households", "equity_share_fin_assets", float(share.iloc[-1]), "%", dt_, "derived",
           "(equities + mutual funds) / financial assets", "股票与基金占金融资产")
    S.line("households", "equity_share_pctile_since_1990", float((share < share.iloc[-1]).mean() * 100), "pctile", dt_, "derived",
           "share of quarters below", "1990 年以来分位")
    S.hist("households", "net_worth", _q_end(nw), 0)
    dspi, pce, sav = R.fg("DSPI"), R.fg("PCE"), R.fg("PMSAVE")
    S.line("households", "dpi_saar", _last(dspi)[1], "$bn SAAR", _last(dspi)[0], "FRED_BEA", "DSPI", "可支配收入")
    S.line("households", "pce_saar", _last(pce)[1], "$bn SAAR", _last(pce)[0], "FRED_BEA", "PCE", "消费支出")
    S.line("households", "saving_rate", _last(sav)[1] / _last(dspi)[1] * 100, "%", _last(sav)[0], "FRED_BEA", "PMSAVE / DSPI", "储蓄率")
    dpi_at = float(dspi.loc[:dt_ + pd.offsets.QuarterEnd(0)].iloc[-1])
    S.line("households", "net_worth_to_dpi", float(nw.iloc[-1]) / dpi_at, "ratio", dt_, "derived", "net worth / DPI", "净资产/可支配收入")
    S.line("households", "debt_to_dpi", float(debt.iloc[-1]) / dpi_at * 100, "%", dt_, "derived", "debt / DPI (= s_hh_debt)",
           "债务/可支配收入")
    tdsp = R.fg("TDSP")
    S.line("households", "debt_service_ratio", _last(tdsp)[1], "% DPI", _last(tdsp)[0], "FRED_FED", "TDSP", "偿债支出/可支配收入")
    for sid, k, zh in [("WFRBST01134", "nw_share_top1", "前1%"), ("WFRBSN09161", "nw_share_90_99", "90-99%"),
                       ("WFRBSN40188", "nw_share_50_90", "50-90%"), ("WFRBSB50215", "nw_share_bottom50", "后50%")]:
        s = R.fg(sid)
        S.line("households", k, _last(s)[1], "% NW", _last(s)[0], "FRED_DFA", sid, f"净资产份额：{zh}")
        S.hist("households", k, _q_end(s), 1)


def _sectors(S: _Snap, R: _Reader) -> None:
    gdp = R.fg("GDP")
    for sid, k, zh, unit in [("BCNSDODNS", "nfc_debt", "非金融企业债务", "$bn"),
                             ("A464RC1Q027SBEA", "nfc_pbt", "税前利润（不含 IVA/CCAdj）", "$bn SAAR"),
                             ("W328RC1Q027SBEA", "nfc_pat_adj", "税后利润（含 IVA/CCAdj）", "$bn SAAR"),
                             ("B471RC1Q027SBEA", "nfc_net_interest", "净利息与杂项支付", "$bn SAAR"),
                             ("A455RC1Q027SBEA", "nfc_gva", "增加值", "$bn SAAR"),
                             ("BOGZ1FA105050005Q", "nfc_capex", "资本开支", "$bn SAAR"),
                             ("BOGZ1FA105005305Q", "nfc_financing_gap", "融资缺口（资本开支 - 内部资金）", "$bn SAAR"),
                             ("BOGZ1FA106121075Q", "nfc_net_dividends", "净股息", "$bn SAAR"),
                             ("NCBCEBQ027S", "nfc_net_equity_issuance", "股票净发行（负 = 净回购）", "$bn SAAR"),
                             ("NCBEILQ027S", "nfc_equity_mv", "股票市值", "$bn")]:
        s = R.fg(sid)
        scale = 1e3 if abs(s.iloc[-1]) > 5e4 and sid != "A455RC1Q027SBEA" else 1.0
        s = s / scale
        S.line("nfc", k, s.iloc[-1], unit, s.index[-1], "FRED_Z1_NIPA", sid, zh)
        S.hist("nfc", k, _q_end(s), 1)
    L = S.books["nfc"]["lines"]
    S.line("nfc", "interest_coverage", (L["nfc_pbt"]["value"] + L["nfc_net_interest"]["value"]) / L["nfc_net_interest"]["value"],
           "x", L["nfc_pbt"]["as_of"], "derived", "(PBT + net interest) / net interest", "利息覆盖倍数")
    S.line("nfc", "nfc_debt_gdp", L["nfc_debt"]["value"] / float(gdp.loc[L["nfc_debt"]["as_of"]]) * 100, "% GDP",
           L["nfc_debt"]["as_of"], "derived", "BCNSDODNS / GDP (= s_nfc_debt)", "企业债务/GDP")
    S.line("nfc", "profit_margin_gva", L["nfc_pat_adj"]["value"] / L["nfc_gva"]["value"] * 100, "%", L["nfc_pat_adj"]["as_of"],
           "derived", "after-tax profits / GVA", "利润率（税后利润/增加值）")
    for sid, k, zh, scale in [("QBPBSTAS", "bank_total_assets", "总资产（FDIC）", 1e3), ("QBPQYNTIY", "bank_nii_q", "季度净利息收入（FDIC）", 1e3),
                              ("TOTLL", "bank_loans", "贷款（H.8）", 1), ("SBCACBW027SBOG", "bank_securities", "证券（H.8）", 1),
                              ("TASACBW027SBOG", "bank_treasury_agency", "国债与机构债（H.8）", 1),
                              ("DPSACBW027SBOG", "bank_deposits", "存款（H.8）", 1), ("DRALACBS", "bank_delinquency", "贷款拖欠率", 1),
                              ("CORALACBS", "bank_chargeoff", "核销率（年化）", 1)]:
        s = R.fg(sid) / scale
        S.line("banks", k, _last(s)[1], "%" if k in ("bank_delinquency", "bank_chargeoff") else "$bn", _last(s)[0],
               "FRED_FDIC_H8", sid, zh)
    B = S.books["banks"]["lines"]
    S.line("banks", "nii_to_assets_ann", B["bank_nii_q"]["value"] * 4 / B["bank_total_assets"]["value"] * 100, "%",
           B["bank_nii_q"]["as_of"], "derived", "4 x NII / total assets", "净利息收入/总资产（年化）")
    mmf = R.fg("MMMFFAQ027S") / 1e3
    S.line("investors", "mmf_assets", _last(mmf)[1], "$bn", _last(mmf)[0], "FRED_Z1", "MMMFFAQ027S", "货币基金资产（Z.1）")
    rm = R.fg("WRMFNS")
    S.line("investors", "retail_mmf", _last(rm)[1], "$bn", _last(rm)[0], "FRED_H6", "WRMFNS", "零售货币基金（H.6）")
    bd = R.fg("BOGZ1FL664090005Q") / 1e3
    S.line("investors", "broker_dealer_assets", _last(bd)[1], "$bn", _last(bd)[0], "FRED_Z1", "BOGZ1FL664090005Q", "券商总金融资产")
    try:
        fm = R.finra_margin()
        col = [c for c in fm.columns if "Debit" in str(c)][0]
        fm = fm.dropna(subset=[col])
        S.line("investors", "margin_debt", float(fm[col].iloc[0]) / 1e3, "$bn", str(fm["Year-Month"].iloc[0]) + "-01",
               "FINRA_MARGIN", col, "保证金负债")
        if len(fm) > 12:
            S.line("investors", "margin_debt_yoy", (fm[col].iloc[0] / fm[col].iloc[12] - 1) * 100, "%",
                   str(fm["Year-Month"].iloc[0]) + "-01", "FINRA_MARGIN", "12m change", "保证金负债同比")
    except (FileNotFoundError, IndexError, KeyError):
        S.missing.append({"book": "investors", "line": "margin_debt", "why": "FINRA 缺失"})
    ca = R.fg("IEABC") / 1e3
    S.line("row", "current_account_q", _last(ca)[1], "$bn/q", _last(ca)[0], "FRED_BEA_ITA", "IEABC", "经常账户（季度）")
    S.line("row", "current_account_gdp", _last(ca)[1] * 4 / float(gdp.loc[_last(ca)[0]]) * 100, "% GDP", _last(ca)[0], "derived",
           "4 x IEABC / GDP", "经常账户/GDP")
    nii = R.fg("IIPUSNETIQ") / 1e3
    S.line("row", "niip", _last(nii)[1], "$bn", _last(nii)[0], "FRED_BEA_ITA", "IIPUSNETIQ", "国际投资净头寸")


def a11_spells(tot: pd.Series) -> list[dict]:
    """A11: debt-limit binding spells in Debt to the Penny (total public debt within 0.05% over every 30-day window,
    a run of >= 45 days, then a jump >= 0.8% within 15 observations)."""
    tot = tot.dropna()
    tot = tot[~tot.index.duplicated()].sort_index()
    dates, vals, n = tot.index, tot.values, len(tot)
    flat = np.zeros(n, bool)
    for k in range(n):
        j = np.searchsorted(dates, dates[k] + pd.Timedelta(days=30))
        if j < n and abs(vals[j] - vals[k]) / vals[k] < 0.0005:
            flat[k:j + 1] = True
    eps, k = [], 0
    while k < n:
        if flat[k]:
            s = k
            while k < n and flat[k]:
                k += 1
            e = k - 1
            dur = (dates[e] - dates[s]).days
            jump = float(vals[min(e + 15, n - 1)] - vals[e])
            if dur >= 45 and jump / vals[e] >= 0.008:
                eps.append({"start": str(dates[s].date()), "end": str(dates[e].date()), "days": int(dur)})
        k += 1
    return eps


def build_root(fetch: Fetcher | None = None) -> dict:
    """The 2026Q4 starting accounts: every line with its value, unit, data date, source and series."""
    fetch = fetch or Fetcher()
    R = _Reader(fetch)
    S = _Snap()
    for name, fn in (("macro", _macro), ("fed", _fed), ("fiscal", _fiscal), ("president", _president), ("china", _china),
                     ("opec", _opec), ("bigtech", _bigtech), ("households", _households), ("sectors", _sectors)):
        _section(S, name, fn, R)
    checks = {}
    try:
        eps = a11_spells(R.debt_to_penny_full()["tot_pub_debt_out_amt"])
        since = [e["days"] for e in eps if e["start"] >= "2011"]
        checks["A11"] = {"n_since_2011": len(since), "median_days": float(np.median(since)) if since else None,
                         "reproduced": len(since) == CALIBRATIONS["A11"]["n_since_2011"]}
    except (FileNotFoundError, KeyError, ValueError):
        checks["A11"] = {"reproduced": None, "why": "Debt to the Penny 缺失"}
    sl = (S.books.get("sectoral") or {}).get("lines") or {}
    if "identity_residual" in sl:
        checks["A12"] = {"residual_bn_saar": sl["identity_residual"]["value"],
                         "reproduced": abs(sl["identity_residual"]["value"]) <= 0.1}
    stale = []
    fred_stale = {k[5:]: v.get("why") for k, v in fetch.status.items()
                  if k.startswith("fred:") and v["status"] in ("stale", "missing")}
    sec_stale = {k.split(":")[1]: v.get("why") for k, v in fetch.status.items()
                 if k.startswith("sec:") and v["status"] == "stale"}
    for book, b in S.books.items():
        for lid, ln in b["lines"].items():
            why = None
            if ln.get("src") == "SEC_XBRL":
                if not fetch.has_sec_user_agent:
                    why = NO_SEC_UA
                else:
                    tks = [tk for tk in BT if tk.lower() in book or tk in str(ln.get("series", ""))] or list(BT)
                    why = next((sec_stale[tk] for tk in tks if tk in sec_stale), None)
            else:
                tokens = set(re.findall(r"[A-Z0-9]{3,}", str(ln.get("series", ""))))
                why = next((fred_stale[t] for t in tokens if t in fred_stale), None)
            if why:
                ln["stale"] = why
                stale.append({"book": book, "line": lid, "why": why})
    out = {"version": ACCOUNTS_VERSION, "as_of_run": dt.date.today().isoformat(), "books": S.books, "missing": S.missing,
           "stale": stale, "sources": fetch.status, "source_summary": fetch.summary(), "checks": checks,
           "calibrations": CALIBRATIONS, "identities": IDENTITIES}
    out["hash"] = hashlib.sha256(jsonio.dumps(S.books).encode("utf-8")).hexdigest()[:16]
    return out


def load_root(path: Path | None = None) -> dict | None:
    """The built root snapshot, or None (the API then shows the build command)."""
    p = Path(path) if path else ROOT_PATH
    try:
        d = jsonio.load(p)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, Mapping) and d.get("books") else None


def root_value(root: Mapping, book: str, line: str, default=None):
    try:
        return root["books"][book]["lines"][line]["value"]
    except (KeyError, TypeError):
        return default


# =====================================================================================================================
# 4. The per-particle accounts and their quarterly update
# =====================================================================================================================

ACCOUNT_INPUTS: dict[str, tuple[str, ...]] = {   # what the accounts read from the model's draw (21 x fields)
    "ff": ("d_ff", "d_policy"), "core": ("d_core",), "unrate": ("d_unrate",), "deficit": ("d_deficit",),
    "y2": ("d_y2",), "y10": ("d_y10",), "cpi_goods": ("d_cpi_goods",), "cpi_serv": ("d_cpi_serv",),
    "rgdp": ("d_rgdp",), "cons_goods": ("d_cons_goods",), "cons_serv": ("d_cons_serv",), "hpi": ("d_hpi",),
    "profits": ("d_profits",), "capex": ("d_capex",), "itinv": ("d_itinv",), "chargeoff": ("d_chargeoff",),
    "lending_std": ("x_lending_std",), "tariff": ("d_tariff",), "usd": ("d_usd",), "oil": ("d_oil",),
    "bdlev": ("d_bdlev",)}
ACCOUNT_ASSETS = ("us_equity", "treasury_10y", "cash", "crude_oil")
VIEW_FIELDS = ("s_debt", "s_int_burden", "s_hh_debt", "s_nfc_debt", "s_foreign_hold")


def resolve_inputs(resolver=None) -> dict[str, str]:
    """Account input key -> schema field id (aliases resolved; the first candidate the schema has)."""
    from .actors import FieldResolver, _resolver
    r = _resolver(resolver) if resolver is not None else FieldResolver.load()
    out = {}
    for k, cands in ACCOUNT_INPUTS.items():
        chosen = None
        for c in cands:
            try:
                c2 = r.canonical(c)
            except KeyError:
                continue
            if r.fields is None or c2 in r.fields:
                chosen = c2
                break
        out[k] = chosen or cands[0]
    return out


@dataclass
class Params:
    # Fed (A5, A10)
    soma_net_yield: float = 1.973
    bill_spread_to_iorb: float = -0.10
    rrp_spread_to_iorb: float = -0.15
    mbs_paydown_q: float = 0.025
    currency_growth_q: float = 0.01437
    rmp_hold_q: float = 57.0               # reserve-management bill purchases per quarter at the root pace
    rmp_expanded_q: float = 100.0          # 判断值
    runoff_q: float = -75.0                # 判断值 (Treasury runoff cap of $25bn a month)
    qe_q: float = 600.0                    # 判断值
    ample_reserves: float = 2900.0         # reserves at the end of runoff (Dec 2025; author's reading, 待核)
    # federal (A1-A4, A13, A14, A16)
    a2_const_q: float = -13.2
    a2_slope: float = 1.055
    omf_q: float = 18.4
    intragov_growth_q: float = 73.3
    not_subject_to_limit: float = 202.9
    lam_q: float = 0.129
    nipa_int_const: float = 158.1
    nipa_int_slope: float = 1.075
    tga_target: float = 850.0              # 判断值 unless pinned (U-15)
    tga_adjust: float = 0.5
    tga_floor: float = 150.0
    em_capacity: float = 350.0             # 判断值 unless pinned (U-15)
    foreign_share_issuance: float = 0.318
    receipts_elasticity: float = 1.73
    deflator_wedge: float = -0.42
    tariff_import_elasticity: float = 1.0  # imports fall 1% per pp of the effective tariff (判断值)
    # households (A6, A15)
    dpi_share_gdp: float = 0.7358
    outlays_other_share_dpi: float = 0.0381
    goods_share_pce: float = 0.31
    ofa_cash_weight: float = 0.7
    liab_growth_q: float = 0.012
    pop_growth_ann: float = 0.5
    # hyperscalers (A8)
    life_years: float = 7.0
    ebitda_margin: float = 0.366
    tax_rate: float = 0.16
    cfo_factor: float = 1.215
    rev_growth_lr: float = 0.10
    rev_growth_halflife_q: float = 8.0
    ads_share: float = 0.31
    ads_beta: float = 2.19
    cash_floor_rev: float = 0.6
    debt_spread: float = 0.6
    # president, China, OPEC+, banks, investors
    gasoline_oil_elasticity: float = 0.4
    china_res_drift_q: float = 3 * 0.038
    china_res_usd_beta: float = -0.401
    cny_weaker_step: float = 0.04
    china_stimulus_credit_pp: float = 2.0
    china_stimulus_deficit_pp: float = 1.0
    shale_response: float = 0.02           # US crude output per quarter per log oil change two quarters back (判断值)
    bank_duration: float = 5.0             # 判断值 (inside Jiang, Matvos, Piskorski & Seru 2023)
    bank_equity_share: float = 0.10        # 判断值
    margin_drift_q: float = 3 * 0.00374
    margin_beta: float = 0.566
    # sectoral
    nation_nl_gdp: float = -3.354
    state_local_nl_saar: float = -450.5
    private_nl_p01_p99: tuple = (-4.23, 15.95)
    saving_rate_p1: float = 2.30
    refunds_remaining_bn: float = 0.0      # pinned (未确认 default 0): paid in the quarter of a refund ruling
    # the federal deficit's quarterly pull toward its baseline (the root's level plus actions' discretionary changes):
    # the cumulated d_deficit regressed on its own lag, 1973–2019, gives −0.047 a quarter (half-life 3.7 years). The
    # quarterly models draw d_deficit without its level (the stack's satellite has no own lag), so without the pull the
    # books' deficit is a random walk (P14 review: −21% … +42% of GDP by 2046 and public debt below zero)
    deficit_pull_q: float = 0.047

    @classmethod
    def from_root(cls, root: Mapping | None, institutions: Mapping | None = None) -> "Params":
        p = cls()
        if root:
            v = lambda b, l: root_value(root, b, l)   # noqa: E731
            if v("fiscal", "intragov_growth_q"):
                p.intragov_growth_q = float(v("fiscal", "intragov_growth_q"))
            if v("fiscal", "not_subject_to_limit") is not None:
                p.not_subject_to_limit = float(v("fiscal", "not_subject_to_limit"))
            if v("sectoral", "nl_nation_row_gdp") is not None:
                p.nation_nl_gdp = float(v("sectoral", "nl_nation_row_gdp"))
            ng, nf = v("sectoral", "nl_government"), v("sectoral", "nl_federal")
            if ng is not None and nf is not None:
                p.state_local_nl_saar = float(ng) - float(nf)
            dpi, gdp = v("households", "dpi_saar"), v("macro", "gdp_saar")
            if dpi and gdp:
                p.dpi_share_gdp = float(dpi) / float(gdp)
            if v("macro", "psavert_p1_since_1959") is not None:
                p.saving_rate_p1 = float(v("macro", "psavert_p1_since_1959"))
            bills = v("fed", "soma_bills_change_13w")
            if bills is not None:
                p.rmp_hold_q = max(float(bills), 0.0)
        inst = dict(institutions or {})
        if inst.get("em_capacity_bn") is not None:
            p.em_capacity = float(inst["em_capacity_bn"])
        if inst.get("tga_target_bn") is not None:
            p.tga_target = float(inst["tga_target_bn"])
        if inst.get("refunds_remaining_bn") is not None:
            p.refunds_remaining_bn = float(inst["refunds_remaining_bn"])
        return p


def foreign_share(S: Mapping) -> np.ndarray:
    """Foreign holdings / debt held by the public, %: 0 where surpluses have repaid the debt (no 0/0)."""
    dhp = np.asarray(S["dhp"], float)
    return np.divide(np.asarray(S["fh"], float) * 100.0, dhp, out=np.zeros_like(dhp), where=dhp > 1e-9)


def _full(v, n) -> np.ndarray:
    return np.full(n, float(v))


# ------------------------------------------------------------------------------------------------- account actions
# Accounts.step takes the account actions either as one dict for every particle (actors.account_actions) or per
# particle: {key: array of length N} with None (or a missing key) where the particle's options set nothing. The
# tree passes the per-particle form, so every particle's books follow its own options (never another particle's).

def _act_raw(a: Mapping, key: str, n: int) -> np.ndarray:
    """The action `key` per particle as an object array (None where not set)."""
    v = a.get(key)
    if isinstance(v, np.ndarray):
        if v.shape[:1] != (n,):
            raise ValueError(f"账户动作 {key} 的长度 {v.shape[:1]} 与粒子数 {n} 不一致")
        return v.astype(object)
    out = np.empty(n, dtype=object)
    out[:] = [v] * n
    return out


def _act_set(a: Mapping, key: str, n: int) -> np.ndarray:
    """Which particles set `key` (the key is present and not None)."""
    if key not in a:
        return np.zeros(n, bool)
    return np.array([x is not None for x in _act_raw(a, key, n)], bool)


def _act_true(a: Mapping, key: str, n: int) -> np.ndarray:
    """Which particles set `key` to a true value."""
    if key not in a:
        return np.zeros(n, bool)
    return np.array([bool(x) for x in _act_raw(a, key, n)], bool)


def _act_float(a: Mapping, key: str, n: int, default: float = 0.0) -> np.ndarray:
    """`key` as a float per particle (`default` where it is not set or empty)."""
    if key not in a:
        return np.full(n, float(default))
    return np.array([float(x) if x not in (None, "", False) else float(default) for x in _act_raw(a, key, n)], float)


def per_particle_actions(columns: Iterable[tuple[np.ndarray, list]], n: int) -> dict:
    """Per-particle account actions from (option code per particle, [account dict per option]) columns, merged in
    column order exactly as actors.account_actions merges one profile's payloads (a later column overrides an
    earlier one for the same key; a capex plan percentage also sets bt_new_guidance)."""
    out: dict = {}
    for code, accs in columns:
        code = np.asarray(code, int)
        for k, acc in enumerate(accs):
            if not acc:
                continue
            sel = code == k
            if not sel.any():
                continue
            for key, val in acc.items():
                arr = out.get(key)
                if arr is None:
                    arr = out[key] = np.full(n, None, dtype=object)
                arr[sel] = val
                if key == "bt_capex_plan_pct":
                    g = out.get("bt_new_guidance")
                    if g is None:
                        g = out["bt_new_guidance"] = np.full(n, None, dtype=object)
                    g[sel] = True
    return out


class Accounts:
    """The books of N particles. `s` holds arrays of shape (N,) (and windows of shape (N, k)); nothing is ever
    averaged across particles: reduction moves particles, and their books, with their mass (take / concat)."""

    WINDOWS = ("_capex_hist", "_y10_win", "_cpi_win", "_res_win", "_oil_lags", "_dpi_win", "_brent_win")

    def __init__(self, n: int, s: dict, carried: dict, params: Params, quarter: str = "2026Q3", q: int = 0,
                 resolver=None, as_of: str = ""):
        self.n = int(n)
        self.s = s
        self.carried = carried                 # scalar root lines the step never changes (shown in packets)
        self.p = params
        self.quarter = quarter                 # the quarter whose end the books describe
        self.q = q
        self.flows: dict = {}
        self.resid: dict = {}
        self.inputs_missing: list[str] = []
        self._fields = resolve_inputs(resolver) if resolver is not None else resolve_inputs()
        self.as_of = as_of

    # ---------------------------------------------------------------------------------------------- construction
    @classmethod
    def from_root(cls, root: Mapping, n: int, *, params: Params | None = None, institutions: Mapping | None = None,
                  resolver=None, quarter: str = "2026Q3") -> "Accounts":
        """The root books for every particle (data values). Missing root lines raise a KeyError naming them."""
        p = params or Params.from_root(root, institutions)
        need = [("fed", "iorb"), ("fed", "soma_treasuries"), ("fed", "soma_bills"), ("fed", "soma_mbs"),
                ("fed", "total_assets"), ("fed", "currency"), ("fed", "reverse_repo_total"), ("fed", "tga"),
                ("fed", "deferred_asset"), ("fed", "other_liabilities_capital_resid"), ("fiscal", "debt_held_public"),
                ("fiscal", "intragov_holdings"), ("fiscal", "statutory_limit"), ("fiscal", "avg_rate_marketable"),
                ("fiscal", "foreign_holdings_tic"), ("households", "equities_direct"), ("households", "mutual_funds"),
                ("households", "real_estate"), ("households", "liabilities"), ("households", "net_worth"),
                ("households", "pce_saar"), ("bigtech", "revenue"), ("bigtech", "capex"), ("bigtech", "depreciation"),
                ("bigtech", "dividends"), ("bigtech", "buybacks"), ("bigtech", "lease_principal"), ("bigtech", "debt"),
                ("bigtech", "cash"), ("bigtech", "securities"), ("macro", "gdp_saar"), ("macro", "y2"), ("macro", "y10")]
        miss = [f"{b}.{k}" for b, k in need if root_value(root, b, k) is None]
        if miss:
            raise KeyError("起点账户缺少这些行（运行 python -m scenariolab.build --only agents_accounts）：" + "、".join(miss))
        g = lambda b, k, d=None: root_value(root, b, k, d)   # noqa: E731
        N = int(n)
        S: dict = {}
        S["gdp"] = _full(g("macro", "gdp_saar"), N)
        S["y2"], S["y10"], S["iorb"] = _full(g("macro", "y2"), N), _full(g("macro", "y10"), N), _full(g("fed", "iorb"), N)
        S["tsy"], S["bills"], S["mbs"] = (_full(g("fed", "soma_treasuries"), N), _full(g("fed", "soma_bills"), N),
                                          _full(g("fed", "soma_mbs"), N))
        S["fed_other_a"] = _full(g("fed", "total_assets") - g("fed", "soma_treasuries") - g("fed", "soma_mbs"), N)
        S["cur"], S["rrp"], S["tga"] = _full(g("fed", "currency"), N), _full(g("fed", "reverse_repo_total"), N), _full(
            g("fed", "tga"), N)
        S["da"], S["fed_other_l"] = _full(g("fed", "deferred_asset"), N), _full(g("fed", "other_liabilities_capital_resid"), N)
        S["res"] = S["tsy"] + S["mbs"] + S["fed_other_a"] - (S["cur"] + S["rrp"] + S["tga"] + S["fed_other_l"] - S["da"])
        soma0, bills0 = g("fed", "soma_treasuries") + g("fed", "soma_mbs"), g("fed", "soma_bills")
        S["_y_coupon"] = _full((p.soma_net_yield * soma0 - (g("fed", "iorb") + p.bill_spread_to_iorb) * bills0)
                               / max(soma0 - bills0, 1.0), N)
        S["dhp"], S["intra"], S["limit"] = (_full(g("fiscal", "debt_held_public"), N), _full(g("fiscal", "intragov_holdings"), N),
                                            _full(g("fiscal", "statutory_limit"), N))
        S["em_used"], S["em_overrun"], S["arrears"] = np.zeros(N), np.zeros(N), np.zeros(N)
        S["susp_q"] = np.zeros(N)
        S["def_gdp"] = _full(g("fiscal", "deficit_nipa_gdp", 5.2), N)
        S["def_base"] = S["def_gdp"].copy()          # the baseline the deficit returns to (moved by actions only)
        S["r_avg"] = _full(g("fiscal", "avg_rate_marketable"), N)
        nipa_int = g("fiscal", "nipa_interest_saar")
        dhp_q = g("fiscal", "debt_held_public_nipa_q", g("fiscal", "debt_held_public"))
        S["_int_adj"] = _full((nipa_int - (p.nipa_int_const + p.nipa_int_slope * g("fiscal", "avg_rate_marketable") / 100.0
                                            * dhp_q)) if nipa_int is not None else 0.0, N)
        S["int_burden_nipa"] = _full(g("fiscal", "interest_burden_nipa", 4.0), N)
        S["tau"] = _full(g("president", "effective_tariff_rate", 6.85), N)
        S["imports_saar"] = _full(g("fiscal", "goods_imports_saar", 4000.0), N)
        S["receipts_q"] = _full(g("fiscal", "receipts_q", 1250.0), N)
        S["fh"] = _full(g("fiscal", "foreign_holdings_tic"), N)
        S["refunds_q"] = np.zeros(N)
        S["hh_eq"] = _full(g("households", "equities_direct") + g("households", "mutual_funds"), N)
        S["hh_re"], S["hh_liab"] = _full(g("households", "real_estate"), N), _full(g("households", "liabilities"), N)
        S["hh_ofa"] = _full(g("households", "net_worth") + g("households", "liabilities") - S["hh_eq"][0] - S["hh_re"][0], N)
        S["pce_saar"] = _full(g("households", "pce_saar"), N)
        S["saving_rate"] = _full(g("households", "saving_rate", 4.0), N)
        for k in ("revenue", "capex", "depreciation", "dividends", "buybacks", "lease_principal", "debt"):
            S["bt_" + k] = _full(g("bigtech", k), N)
        S["bt_cash"] = _full(g("bigtech", "cash") + g("bigtech", "securities"), N)
        S["bt_rev_g"] = _full((g("bigtech", "revenue_growth_4q", 10.0) or 10.0) / 100.0, N)
        hist = ((root.get("books") or {}).get("bigtech") or {}).get("capex_history") or {}
        vals = [float(v) for v in (hist.get("values") or []) if v is not None]
        n_life = int(round(4 * p.life_years))
        if len(vals) < n_life:
            vals = [vals[0] if vals else g("bigtech", "capex")] * (n_life - len(vals)) + vals
        S["_capex_hist"] = np.tile(np.array(vals[-n_life:], float), (N, 1))
        base = S["_capex_hist"].sum(axis=1) / n_life
        S["_dep_scale"] = S["bt_depreciation"] / np.maximum(base, 1e-9)
        S["bt_capex_plan_q"] = np.zeros(N)          # quarterly growth of capex while a plan runs (x/4 %)
        S["bt_plan_left"] = np.zeros(N)
        S["bt_buyback_mult"] = np.ones(N)
        y10h = list((((root.get("books") or {}).get("macro") or {}).get("history") or {}).get("y10_q_20", {}).values())
        y10h = [float(v) for v in y10h if v is not None] or [float(g("macro", "y10"))]
        S["_y10_win"] = np.tile(np.array((y10h * 20)[-20:] if len(y10h) < 20 else y10h[-20:], float), (N, 1))
        cpi = g("president", "cpi_yoy", 3.0)
        S["_cpi_win"] = np.tile(np.full(4, float(cpi)), (N, 1))
        S["cpi_yoy"] = _full(cpi, N)
        S["unrate"] = _full(g("fed", "unemployment", 4.3), N)
        S["core_pce"] = _full(g("fed", "core_pce_yoy", 3.0), N)
        S["nrou"] = _full(g("fed", "nrou_cbo", 4.3), N)
        S["gas"] = _full(g("president", "gasoline_price", 3.5), N)
        S["spr"] = _full(g("president", "spr_inventory", 400.0), N)
        S["eq_index"] = np.ones(N)
        dd = g("president", "us_equity_drawdown", 0.0) or 0.0
        S["eq_peak"] = _full(1.0 / (1.0 + dd / 100.0), N)
        rdpi = g("president", "real_dpi_pc_yoy", 1.0)
        S["_dpi_win"] = np.tile(np.full(4, float(rdpi) + 0.0), (N, 1))
        S["real_dpi_pc_yoy"] = _full(rdpi, N)
        S["cn_res"] = _full(g("china", "fx_reserves_ex_gold", 3300.0), N)
        S["_res_win"] = np.tile(np.full(4, float(g("china", "fx_reserves_ex_gold", 3300.0))), (N, 1))
        S["cn_ust"] = _full(g("china", "us_treasury_holdings", 600.0), N)
        S["cny"] = _full(g("china", "cny_per_usd", 7.0), N)
        S["cn_credit"] = _full(g("china", "private_credit_gdp", 200.0), N)
        S["cn_def"] = _full(-(g("china", "gg_net_lending_gdp_2026", -8.0) or -8.0), N)
        S["cn_imports_us"] = _full(g("china", "us_imports_from_china_12m", 270.0), N)
        S["brent"] = _full(g("opec", "brent_spot", 80.0), N)
        bq = list((((root.get("books") or {}).get("opec") or {}).get("history") or {}).get("brent_quarterly_avg", {}).values())
        bq = [float(v) for v in bq if v is not None] or [float(g("opec", "brent_spot", 80.0))]
        S["_brent_win"] = np.tile(np.array(([bq[0]] * 2 + bq)[-2:], float), (N, 1))
        S["us_crude"] = _full(g("opec", "us_crude_production", 13.5), N)
        S["_oil_lags"] = np.zeros((N, 2))
        S["sa_res"] = _full(g("opec", "saudi_reserves_ex_gold", 450.0), N)
        S["opec_quota_chg"] = np.zeros(N)
        S["nfc_debt"] = _full(g("nfc", "nfc_debt", 15700.0), N)
        S["nfc_pbt"] = _full(g("nfc", "nfc_pbt", 3500.0), N)
        S["nfc_ni"] = _full(g("nfc", "nfc_net_interest", 360.0), N)
        S["nfc_capex"] = _full(g("nfc", "nfc_capex", 3250.0), N)
        ta = g("banks", "bank_total_assets", 26000.0)
        S["bk_loans"] = _full(g("banks", "bank_loans", 0.54 * ta), N)
        S["bk_sec_cost"] = _full(g("banks", "bank_securities", 0.22 * ta), N)
        S["bk_unreal"] = np.zeros(N)
        S["bk_other_a"] = _full(ta - S["bk_loans"][0] - S["bk_sec_cost"][0], N)
        S["bk_dep"] = _full(g("banks", "bank_deposits", 0.75 * ta), N)
        S["bk_equity_book"] = _full(p.bank_equity_share * ta, N)
        S["bk_other_l"] = _full(ta - S["bk_dep"][0] - S["bk_equity_book"][0], N)
        S["bk_chargeoff"] = _full(g("banks", "bank_chargeoff", 0.5), N)
        S["bk_nii_q"] = _full(g("banks", "bank_nii_q", 190.0), N)
        S["margin"] = _full(g("investors", "margin_debt", 1000.0), N)
        S["row_niip"] = _full(g("row", "niip", -20000.0), N)
        S["row_ca_q"] = _full(g("row", "current_account_q", -250.0), N)
        carried = {}
        for book, b in (root.get("books") or {}).items():
            for lid, ln in (b.get("lines") or {}).items():
                carried[f"{book}.{lid}"] = ln.get("value")
        acc = cls(N, S, carried, p, quarter=quarter, resolver=resolver, as_of=str(root.get("as_of_run", "")))
        acc._record_root()
        return acc

    def _record_root(self):
        p, S = self.p, self.s
        mts0 = p.a2_const_q + p.a2_slope * S["def_gdp"] / 100.0 * S["gdp"] / 4.0     # A2 at the root's deficit
        self.flows = {"fed_ni": np.zeros(self.n), "fed_remit": np.zeros(self.n), "borrow": np.zeros(self.n),
                      "mts_def_q": mts0, "limit_binding": np.zeros(self.n), "xdate": np.zeros(self.n),
                      "bt_fcf": self.s["bt_revenue"] * 0 + float(self.carried.get("bigtech.fcf") or 0.0),
                      "bt_d_debt": np.zeros(self.n), "hh_saving_rate": self.s["saving_rate"].copy(),
                      "dpi_saar": self.p.dpi_share_gdp * self.s["gdp"], "customs_q": self.s["tau"] / 100.0
                      * self.s["imports_saar"] / 4.0, "sect_private_nl_gdp": np.full(
                          self.n, float(self.carried.get("sectoral.nl_private_gdp") or 0.0)),
                      "tga_prev": self.s["tga"].copy(), "bt_ebitda": self.s["bt_revenue"] * self.p.ebitda_margin,
                      "nipa_int_saar": np.full(self.n, float(self.carried.get("fiscal.nipa_interest_saar") or 0.0))}

    # ---------------------------------------------------------------------------------------------- particles
    def copy(self) -> "Accounts":
        out = Accounts(self.n, {k: v.copy() for k, v in self.s.items()}, dict(self.carried), copy.copy(self.p),
                       quarter=self.quarter, q=self.q, as_of=self.as_of)
        out._fields = dict(self._fields)
        out.flows = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.flows.items()}
        out.resid = dict(self.resid)
        return out

    def take(self, idx) -> "Accounts":
        """The books of the particles `idx` (resampling or reassignment moves accounts with their particles)."""
        idx = np.asarray(idx, int)
        out = Accounts(len(idx), {k: v[idx].copy() for k, v in self.s.items()}, dict(self.carried), copy.copy(self.p),
                       quarter=self.quarter, q=self.q, as_of=self.as_of)
        out._fields = dict(self._fields)
        out.flows = {k: (v[idx].copy() if isinstance(v, np.ndarray) and v.shape[:1] == (self.n,) else v)
                     for k, v in self.flows.items()}
        out.resid = dict(self.resid)
        return out

    @classmethod
    def concat(cls, parts: list["Accounts"]) -> "Accounts":
        """Pool several clouds' books (particles keep their own books; nothing is averaged)."""
        if not parts:
            raise ValueError("没有可以合并的账户")
        a0 = parts[0]
        s = {k: np.concatenate([p.s[k] for p in parts], axis=0) for k in a0.s}
        out = cls(sum(p.n for p in parts), s, dict(a0.carried), copy.copy(a0.p), quarter=a0.quarter, q=a0.q, as_of=a0.as_of)
        out._fields = dict(a0._fields)
        out.flows = {k: np.concatenate([p.flows[k] for p in parts], axis=0) for k in a0.flows
                     if all(isinstance(p.flows.get(k), np.ndarray) for p in parts)}
        return out

    # ---------------------------------------------------------------------------------------------- inputs
    def _x(self, x: Mapping, key: str) -> np.ndarray:
        fid = self._fields.get(key)
        v = x.get(fid) if fid else None
        if v is None:
            for c in ACCOUNT_INPUTS[key]:
                if c in x:
                    v = x[c]
                    break
        if v is None:
            self.inputs_missing.append(fid or key)
            return np.zeros(self.n)
        arr = np.asarray(v, float)
        return np.broadcast_to(arr, (self.n,)).astype(float) if arr.ndim == 0 else arr.astype(float)

    def _y(self, y: Mapping, key: str) -> np.ndarray:
        v = y.get(key)
        if v is None:
            self.inputs_missing.append(f"y:{key}")
            return np.zeros(self.n)
        arr = np.asarray(v, float)
        return np.broadcast_to(arr, (self.n,)).astype(float) if arr.ndim == 0 else arr.astype(float)

    # ---------------------------------------------------------------------------------------------- one quarter
    def step(self, x: Mapping, y: Mapping, actions: Mapping | None = None) -> dict:
        """One quarter for every particle. x: schema field id -> array (the model's draw, aliases accepted);
        y: asset id -> quarterly log return; actions: actors.account_actions(...) for every particle, or per particle
        ({key: array of length N}, per_particle_actions). Returns the step report."""
        a = dict(actions or {})
        p, S, N = self.p, self.s, self.n
        self.inputs_missing = []
        X = {k: self._x(x, k) for k in ACCOUNT_INPUTS}
        Y = {k: self._y(y, k) for k in ACCOUNT_ASSETS}
        F: dict = {}
        # ---- macro aggregates (A13: GDP deflator ~ CPI + wedge)
        # headline CPI inflation (%年化): the model's own d_cpi when its draw carries one (the quarterly stack's
        # training weights, so the books' CPI y/y follows the model's), else the goods/services split at 0.37/0.63
        cpi_x = x.get("d_cpi") if hasattr(x, "get") else None
        cpi_q = (np.broadcast_to(np.asarray(cpi_x, float), (N,)).astype(float) if cpi_x is not None
                 else 0.37 * X["cpi_goods"] + 0.63 * X["cpi_serv"])
        ngdp_g = (X["rgdp"] + cpi_q + p.deflator_wedge) / 400.0
        S["gdp"] = S["gdp"] * np.exp(ngdp_g)
        dy10 = X["y10"]
        S["y2"] = S["y2"] + X["y2"]
        S["y10"] = S["y10"] + dy10
        S["iorb"] = np.maximum(S["iorb"] + X["ff"], 0.15)
        S["_y10_win"] = np.concatenate([S["_y10_win"][:, 1:], S["y10"][:, None]], axis=1)
        S["_cpi_win"] = np.concatenate([S["_cpi_win"][:, 1:], cpi_q[:, None]], axis=1)
        S["cpi_yoy"] = S["_cpi_win"].mean(axis=1)
        S["unrate"] = S["unrate"] + X["unrate"]
        S["core_pce"] = S["core_pce"] + 0.25 * (X["core"] - S["core_pce"])
        # ---- Fed (A5, A10): bill purchases by the balance-sheet attribute; MBS principal reinvested in bills
        rmp_of = {"hold": p.rmp_hold_q, "reserve_purchases": p.rmp_expanded_q, "runoff": p.runoff_q, "qe": p.qe_q}
        rmp = np.array([rmp_of.get(b or "hold", p.rmp_hold_q) for b in _act_raw(a, "fed_bs", N)], float)
        own_rmp = _act_set(a, "fed_rmp_q", N)
        if own_rmp.any():
            rmp = np.where(own_rmp, _act_float(a, "fed_rmp_q", N), rmp)
        paydown = p.mbs_paydown_q * S["mbs"]
        bills_yield = S["iorb"] + p.bill_spread_to_iorb
        income = (S["_y_coupon"] * (S["tsy"] - S["bills"] + S["mbs"]) + bills_yield * S["bills"]) / 400.0
        expense = (S["iorb"] * S["res"] + (S["iorb"] + p.rrp_spread_to_iorb) * S["rrp"]) / 400.0
        ni = income - expense
        da_prev = S["da"].copy()
        repay = np.where(ni > 0, np.minimum(S["da"], ni), 0.0)
        remit = np.where(ni > 0, ni - repay, 0.0)
        S["da"] = S["da"] + np.where(ni < 0, -ni, 0.0) - repay
        S["tsy"] = S["tsy"] + rmp + paydown
        S["bills"] = np.maximum(S["bills"] + rmp + paydown, 0.0)
        S["mbs"] = S["mbs"] - paydown
        S["cur"] = S["cur"] * (1 + p.currency_growth_q)
        F.update(fed_ni=ni, fed_remit=remit, fed_income=income, fed_expense=expense, fed_repay=repay, fed_da_prev=da_prev,
                 fed_rmp=rmp.copy())
        # ---- federal government (A1-A4, A14, A16): the model's d_deficit is the NIPA total (one writer); the level
        # returns to its baseline at deficit_pull_q a quarter, and the baseline moves with actions' discretionary
        # changes (x["_agent_deficit_action"], written by the engine), so an enacted consolidation or stimulus stays
        act = x.get("_agent_deficit_action") if hasattr(x, "get") else None
        act = np.zeros(N) if act is None else np.broadcast_to(np.asarray(act, float), (N,)).astype(float)
        base = S.get("def_base")
        base = S["def_gdp"].copy() if base is None else base
        S["def_gdp"] = S["def_gdp"] + X["deficit"] - p.deficit_pull_q * (S["def_gdp"] - base)
        S["def_base"] = base + act
        nipa_q = S["def_gdp"] / 100.0 * S["gdp"] / 4.0
        refunds = _act_float(a, "customs_refunds_q", N)
        ordered = _act_set(a, "customs_refunds_bn", N)
        if ordered.any():                                 # a court-ordered refund: the pinned liability, paid once
            raw = _act_raw(a, "customs_refunds_bn", N)
            refunds = refunds + np.array([float(p.refunds_remaining_bn if v == "pinned" else v or 0.0) if o else 0.0
                                          for v, o in zip(raw, ordered)], float)
        mts_q = p.a2_const_q + p.a2_slope * nipa_q + refunds
        tga_prev, dhp_prev = S["tga"].copy(), S["dhp"].copy()
        raise_ = _act_float(a, "debt_limit_raise_bn", N)
        susp = _act_float(a, "debt_limit_suspend_q", N)
        lifted = (raise_ > 0) | (susp > 0)
        if lifted.any():                                  # a raise or a suspension restores extraordinary measures
            S["intra"] = np.where(lifted, S["intra"] + S["em_used"], S["intra"])
            S["em_used"] = np.where(lifted, 0.0, S["em_used"])
            S["limit"] = np.where(raise_ > 0, S["limit"] + raise_, S["limit"])
            S["susp_q"] = np.where(susp > 0, susp, S["susp_q"])
        suspended = S["susp_q"] > 0
        tga_desired = S["tga"] + p.tga_adjust * (p.tga_target - S["tga"])
        need = mts_q + p.omf_q + (tga_desired - S["tga"])
        if suspended.any():          # during a suspension the limit follows the debt (no binding)
            S["limit"] = np.where(suspended, S["dhp"] + S["intra"] - p.not_subject_to_limit + np.maximum(need, 0.0)
                                  + p.intragov_growth_q + 1.0, S["limit"])
        subj = S["dhp"] + S["intra"] - p.not_subject_to_limit
        over = subj + np.maximum(need, 0.0) + p.intragov_growth_q - S["limit"]
        em_left = np.maximum(p.em_capacity - S["em_used"], 0.0)
        em_now = np.minimum(np.maximum(over, 0.0), em_left)
        S["em_used"] = S["em_used"] + em_now
        S["intra"] = S["intra"] + p.intragov_growth_q - em_now
        excess = np.maximum(S["dhp"] + S["intra"] - p.not_subject_to_limit - S["limit"], 0.0)
        S["intra"] = S["intra"] - excess
        S["em_overrun"] = S["em_overrun"] + excess
        room = S["limit"] - (S["dhp"] + S["intra"] - p.not_subject_to_limit)
        borrow = np.minimum(need, np.maximum(room, 0.0))
        borrow = np.maximum(borrow, -S["dhp"])            # a surplus repays debt down to zero; the rest stays in the TGA
        short = np.maximum(need - borrow, 0.0)
        S["dhp"] = S["dhp"] + borrow
        tga_new = S["tga"] + borrow - mts_q - p.omf_q
        arrears_new = np.maximum(-tga_new, 0.0)
        S["arrears"] = S["arrears"] + arrears_new
        S["tga"] = np.maximum(tga_new, 0.0)
        xdate = S["tga"] < p.tga_floor
        S["r_avg"] = S["r_avg"] + p.lam_q * ((S["y2"] + S["y10"]) / 2.0 - S["r_avg"])
        nipa_int = p.nipa_int_const + p.nipa_int_slope * S["r_avg"] / 100.0 * dhp_prev + S["_int_adj"]
        S["_int_adj"] = S["_int_adj"] * 0.5 ** 0.25
        S["int_burden_nipa"] = nipa_int / S["gdp"] * 100.0
        S["tau"] = np.maximum(S["tau"] + X["tariff"], 0.0)
        S["imports_saar"] = S["imports_saar"] * np.exp((X["cons_goods"] + 0.37 * X["cpi_goods"]) / 400.0
                                                       - p.tariff_import_elasticity * X["tariff"] / 100.0)
        customs_q = S["tau"] / 100.0 * S["imports_saar"] / 4.0
        S["receipts_q"] = S["receipts_q"] * np.exp(p.receipts_elasticity * ngdp_g) + remit
        outlays_q = S["receipts_q"] + mts_q
        S["fh"] = S["fh"] + p.foreign_share_issuance * borrow
        # holders cannot hold more than the debt outstanding (I5): when surpluses shrink the debt, foreign holdings
        # give way first and the Fed's bill purchases stop at what is left (a quarterly model's deficit level
        # wanders, so a path can run surpluses for years)
        over = S["fh"] + S["tsy"] - S["dhp"]
        if np.any(over > 0):
            cut_fh = np.minimum(np.maximum(over, 0.0), S["fh"])
            S["fh"] = S["fh"] - cut_fh
            cut_tsy = np.maximum(S["fh"] + S["tsy"] - S["dhp"], 0.0)
            S["tsy"] = S["tsy"] - cut_tsy
            S["bills"] = np.maximum(S["bills"] - cut_tsy, 0.0)
        S["susp_q"] = np.maximum(S["susp_q"] - 1.0, 0.0)
        ended = suspended & (S["susp_q"] == 0)
        if ended.any():              # the suspension ends: the limit is reset at the debt then outstanding
            S["limit"] = np.where(ended, S["dhp"] + S["intra"] - p.not_subject_to_limit, S["limit"])
        F.update(nipa_def_q=nipa_q, mts_def_q=mts_q, borrow=borrow, em_now=em_now, customs_q=customs_q, refunds_q=refunds.copy(),
                 outlays_q=outlays_q, xdate=xdate.astype(float), nipa_int_saar=nipa_int, need=need,
                 limit_binding=((short > 1e-9) | (em_now > 1e-9)).astype(float), tga_prev=tga_prev, dhp_prev=dhp_prev,
                 arrears_new=arrears_new)
        # ---- Fed liabilities: reserves are the balancing item (TGA drawdown adds reserves; losses are paid in reserves)
        S["res"] = S["tsy"] + S["mbs"] + S["fed_other_a"] - (S["cur"] + S["rrp"] + S["tga"] + S["fed_other_l"] - S["da"])
        # ---- households (A6, A15): revaluation from the model's returns; saving = income minus consumption
        dpi_saar = p.dpi_share_gdp * S["gdp"]
        real_c = p.goods_share_pce * X["cons_goods"] + (1 - p.goods_share_pce) * X["cons_serv"]
        S["pce_saar"] = S["pce_saar"] * np.exp((real_c + cpi_q + p.deflator_wedge) / 400.0)
        saving_q = (dpi_saar * (1 - p.outlays_other_share_dpi) - S["pce_saar"]) / 4.0
        eq_reval = S["hh_eq"] * (np.exp(Y["us_equity"]) - 1.0)
        re_reval = S["hh_re"] * X["hpi"] / 400.0
        r_ofa = p.ofa_cash_weight * (np.exp(Y["cash"]) - 1) + (1 - p.ofa_cash_weight) * (np.exp(Y["treasury_10y"]) - 1)
        ofa_reval = S["hh_ofa"] * r_ofa
        nw_prev = S["hh_eq"] + S["hh_re"] + S["hh_ofa"] - S["hh_liab"]
        d_liab = S["hh_liab"] * p.liab_growth_q
        S["hh_eq"] = S["hh_eq"] + eq_reval
        S["hh_re"] = S["hh_re"] + re_reval
        S["hh_ofa"] = S["hh_ofa"] + ofa_reval + saving_q + d_liab
        S["hh_liab"] = S["hh_liab"] + d_liab
        S["saving_rate"] = saving_q * 4 / dpi_saar * 100
        dpi_real_g = 400.0 * ngdp_g - cpi_q - p.pop_growth_ann
        S["_dpi_win"] = np.concatenate([S["_dpi_win"][:, 1:], dpi_real_g[:, None]], axis=1)
        S["real_dpi_pc_yoy"] = S["_dpi_win"].mean(axis=1)
        F.update(hh_saving_q=saving_q, hh_saving_rate=S["saving_rate"], hh_eq_reval=eq_reval, hh_re_reval=re_reval,
                 hh_ofa_reval=ofa_reval, hh_nw_prev=nw_prev, dpi_saar=dpi_saar)
        # ---- hyperscalers (A8): capex = the action (a guidance revision runs 4 quarters); cash identity I7
        guided = _act_true(a, "bt_new_guidance", N)
        if guided.any():
            x_pct = _act_float(a, "bt_capex_plan_pct", N)
            S["bt_capex_plan_q"] = np.where(guided, x_pct / 4.0, S["bt_capex_plan_q"])
            S["bt_plan_left"] = np.where(guided, 4.0, S["bt_plan_left"])
        own_bb = _act_set(a, "bt_buyback_mult", N)
        if own_bb.any():
            S["bt_buyback_mult"] = np.where(own_bb, _act_float(a, "bt_buyback_mult", N, 1.0), S["bt_buyback_mult"])
        running = S["bt_plan_left"] > 0
        S["bt_capex"] = np.where(running, S["bt_capex"] * (1 + S["bt_capex_plan_q"] / 100.0), S["bt_capex"])
        S["bt_plan_left"] = np.maximum(S["bt_plan_left"] - 1.0, 0.0)
        S["_capex_hist"] = np.concatenate([S["_capex_hist"][:, 1:], S["bt_capex"][:, None]], axis=1)
        n_life = S["_capex_hist"].shape[1]
        S["bt_depreciation"] = S["_dep_scale"] * S["_capex_hist"].sum(axis=1) / n_life
        decay = 0.5 ** (1.0 / p.rev_growth_halflife_q)
        S["bt_rev_g"] = p.rev_growth_lr + decay * (S["bt_rev_g"] - p.rev_growth_lr)
        cyc = p.ads_share * p.ads_beta * (ngdp_g * 4 - 0.045)
        S["bt_revenue"] = S["bt_revenue"] * (1 + (S["bt_rev_g"] + cyc) / 4.0)
        ebitda = S["bt_revenue"] * p.ebitda_margin
        opinc = ebitda - S["bt_depreciation"]
        interest = S["bt_debt"] * (S["y10"] + p.debt_spread) / 400.0
        cfo = p.cfo_factor * ((opinc - interest) * (1 - p.tax_rate) + S["bt_depreciation"])
        bb = S["bt_buybacks"] * S["bt_buyback_mult"]
        S["bt_buyback_mult"] = np.ones(N)
        payouts = bb + S["bt_dividends"]
        fcf = cfo - S["bt_capex"] - S["bt_lease_principal"]
        floor_ = p.cash_floor_rev * S["bt_revenue"]
        pre = S["bt_cash"] + fcf - payouts
        d_debt = np.maximum(floor_ - pre, 0.0)
        cash_prev, debt_prev = S["bt_cash"].copy(), S["bt_debt"].copy()
        S["bt_cash"] = pre + d_debt
        S["bt_debt"] = S["bt_debt"] + d_debt
        S["bt_buybacks"] = bb
        F.update(bt_opinc=opinc, bt_cfo=cfo, bt_fcf=fcf, bt_payouts=payouts, bt_d_debt=d_debt, bt_cash_prev=cash_prev,
                 bt_debt_prev=debt_prev, bt_interest=interest, bt_ebitda=ebitda)
        # ---- president's scorecard, China (A7), OPEC+, banks, investors (A9), NFC, rest of world
        S["gas"] = S["gas"] * np.exp(p.gasoline_oil_elasticity * X["oil"] / 100.0)
        S["spr"] = np.maximum(S["spr"] - _act_float(a, "spr_release_mbd", N) * 91.25, 0.0)
        S["eq_index"] = S["eq_index"] * np.exp(Y["us_equity"])
        S["eq_peak"] = np.maximum(S["eq_peak"], S["eq_index"])
        S["cn_res"] = S["cn_res"] * np.exp((p.china_res_drift_q + p.china_res_usd_beta * X["usd"]) / 100.0)
        S["_res_win"] = np.concatenate([S["_res_win"][:, 1:], S["cn_res"][:, None]], axis=1)
        S["cn_ust"] = S["cn_ust"] * np.exp(Y["treasury_10y"])
        weaker = _act_true(a, "cny_weaker", N)
        if weaker.any():
            S["cny"] = np.where(weaker, S["cny"] * (1 + p.cny_weaker_step), S["cny"])
        stim = _act_true(a, "cn_stimulus", N)
        if stim.any():
            S["cn_credit"] = np.where(stim, S["cn_credit"] + p.china_stimulus_credit_pp, S["cn_credit"])
            S["cn_def"] = np.where(stim, S["cn_def"] + p.china_stimulus_deficit_pp, S["cn_def"])
        S["cn_imports_us"] = S["cn_imports_us"] * np.exp((X["cons_goods"] + 0.37 * X["cpi_goods"]) / 400.0
                                                         - p.tariff_import_elasticity * X["tariff"] / 100.0)
        S["brent"] = S["brent"] * np.exp(X["oil"] / 100.0)
        S["_brent_win"] = np.concatenate([S["_brent_win"][:, 1:], S["brent"][:, None]], axis=1)
        S["us_crude"] = S["us_crude"] * (1 + p.shale_response * S["_oil_lags"][:, 0] / 100.0)
        S["_oil_lags"] = np.concatenate([S["_oil_lags"][:, 1:], X["oil"][:, None]], axis=1)
        S["opec_quota_chg"] = S["opec_quota_chg"] + _act_float(a, "opec_quota_chg_mbd", N)
        g = np.exp(ngdp_g)
        unreal_prev = S["bk_unreal"].copy()
        for k in ("bk_loans", "bk_sec_cost", "bk_other_a", "bk_dep", "bk_other_l", "bk_equity_book"):
            S[k] = S[k] * g
        S["bk_unreal"] = S["bk_unreal"] - p.bank_duration * dy10 / 100.0 * S["bk_sec_cost"]
        S["bk_chargeoff"] = np.maximum(S["bk_chargeoff"] + X["chargeoff"], 0.0)
        S["bk_nii_q"] = S["bk_nii_q"] * g
        S["margin"] = S["margin"] * np.exp(p.margin_drift_q + p.margin_beta * Y["us_equity"])
        S["nfc_debt"] = S["nfc_debt"] * g
        S["nfc_pbt"] = S["nfc_pbt"] * np.exp(X["profits"] / 100.0)
        S["nfc_ni"] = S["nfc_ni"] * g
        S["nfc_capex"] = S["nfc_capex"] * np.exp((X["capex"] + cpi_q + p.deflator_wedge) / 400.0)
        S["row_ca_q"] = p.nation_nl_gdp / 100.0 * S["gdp"] / 4.0
        S["row_niip"] = S["row_niip"] + S["row_ca_q"]
        F.update(bk_unreal_prev=unreal_prev)
        # ---- sectoral balances (A12): federal (accounts) + state & local (carried); nation carried; private = residual
        fed_nl = -nipa_q * 4.0
        gov_nl = fed_nl + p.state_local_nl_saar
        nation = p.nation_nl_gdp / 100.0 * S["gdp"]
        priv = nation - gov_nl
        F.update(sect_private_nl_gdp=priv / S["gdp"] * 100, sect_gov_nl=gov_nl, sect_nation_nl=nation, sect_private_nl=priv)
        self.flows = F
        self.q += 1
        from .actors import quarter_add
        self.quarter = quarter_add(self.quarter, 1)
        self.check()
        return {"quarter": self.quarter, "ok": self.ok(), "residuals": dict(self.resid),
                "inputs_missing": sorted(set(self.inputs_missing))}

    # ---------------------------------------------------------------------------------------------- identities
    def check(self) -> dict:
        """Identities I1–I10 (max over particles; relative where the line is a stock)."""
        S, F, p = self.s, self.flows, self.p
        R = {}
        assets = S["tsy"] + S["mbs"] + S["fed_other_a"]
        liab = S["cur"] + S["res"] + S["rrp"] + S["tga"] + S["fed_other_l"] - S["da"]
        R["I1_fed_balance_sheet"] = float(np.max(np.abs(assets - liab) / np.abs(assets)))
        if "fed_ni" in F and "fed_da_prev" in F:
            dda = (S["da"] - F["fed_da_prev"]) - (np.where(F["fed_ni"] < 0, -F["fed_ni"], 0.0) - F["fed_repay"])
            rem = F["fed_remit"] - np.where(F["fed_ni"] > 0, F["fed_ni"] - F["fed_repay"], 0.0)
            R["I2_fed_income"] = float(np.max(np.abs(dda)) + np.max(np.abs(rem)))
        if all(k in F for k in ("borrow", "tga_prev", "mts_def_q", "arrears_new", "dhp_prev")):
            R["I3_tga_cash"] = float(np.max(np.abs((S["tga"] - F["arrears_new"]) - F["tga_prev"]
                                                   - (F["borrow"] - F["mts_def_q"] - p.omf_q))))
            R["I4_public_debt"] = float(np.max(np.abs((S["dhp"] - F["dhp_prev"]) - F["borrow"])))
        R["I4_debt_le_limit"] = float(np.max(np.maximum(S["dhp"] + S["intra"] - p.not_subject_to_limit - S["limit"], 0.0)
                                             / S["limit"]))
        R["I5_holders"] = float(np.max(np.maximum(S["fh"] + S["tsy"] - S["dhp"], 0.0)))
        if "hh_nw_prev" in F:
            nw = S["hh_eq"] + S["hh_re"] + S["hh_ofa"] - S["hh_liab"]
            flow = F["hh_nw_prev"] + F["hh_saving_q"] + F["hh_eq_reval"] + F["hh_re_reval"] + F["hh_ofa_reval"]
            R["I6_household_stock_flow"] = float(np.max(np.abs(nw - flow) / np.abs(nw)))
        if "bt_cash_prev" in F:
            chk = F["bt_cash_prev"] + F["bt_fcf"] - F["bt_payouts"] + (S["bt_debt"] - F["bt_debt_prev"])
            R["I7_company_cash"] = float(np.max(np.abs(S["bt_cash"] - chk) / np.maximum(np.abs(S["bt_cash"]), 1.0)))
        if "sect_private_nl" in F:
            R["I8_sectoral"] = float(np.max(np.abs(F["sect_private_nl"] + F["sect_gov_nl"] - F["sect_nation_nl"]) / S["gdp"]))
        v = self.views()
        R["I9_views"] = float(max(np.max(np.abs(v["s_debt"] - S["dhp"] / S["gdp"] * 100)),
                                  np.max(np.abs(v["s_foreign_hold"] - foreign_share(S)))))
        bk_a = S["bk_loans"] + S["bk_sec_cost"] + S["bk_unreal"] + S["bk_other_a"]
        bk_l = S["bk_dep"] + S["bk_other_l"] + S["bk_equity_book"] + S["bk_unreal"]
        R["I10_banks"] = float(np.max(np.abs(bk_a - bk_l) / np.abs(bk_a)))
        self.resid = R
        return R

    def ok(self) -> bool:
        return all(v <= TOL for k, v in self.resid.items() if not k.startswith("I3") and not k.startswith("I4_public")) \
            and self.resid.get("I3_tga_cash", 0.0) <= TOL * 10 and self.resid.get("I4_public_debt", 0.0) <= TOL * 10

    # ---------------------------------------------------------------------------------------------- views and facts
    def views(self) -> dict[str, np.ndarray]:
        """I9: the schema's view states from the books (s_int_burden by its v3 definition: the past 20 quarters'
        mean 10-year yield × debt held by the public / GDP). The model is handed none of them."""
        S, F = self.s, self.flows
        dpi = F.get("dpi_saar")
        if dpi is None:
            dpi = self.p.dpi_share_gdp * S["gdp"]
        debt_gdp = S["dhp"] / S["gdp"] * 100
        hh_debt = S["hh_liab"] * float(self.carried.get("households.debt") or 1.0) / float(
            self.carried.get("households.liabilities") or 1.0)
        return {"s_debt": debt_gdp, "s_int_burden": S["_y10_win"].mean(axis=1) * debt_gdp / 100.0,
                "s_hh_debt": hh_debt / dpi * 100, "s_nfc_debt": S["nfc_debt"] / S["gdp"] * 100,
                "s_foreign_hold": foreign_share(S)}

    def fact_shares(self) -> dict[str, float]:
        """Shares of particles in the discrete account states (debt limit binding, X-date, arrears, FCF < 0)."""
        S, F = self.s, self.flows
        return {"debt_limit_binding": float(np.mean(F.get("limit_binding", np.zeros(self.n)))),
                "xdate_passed": float(np.mean(S["tga"] < self.p.tga_floor)),
                "arrears": float(np.mean(S["arrears"] > 1e-6)),
                "hyperscaler_fcf_negative": float(np.mean(F.get("bt_fcf", np.zeros(self.n)) < 0)),
                "saving_rate_below_p1": float(np.mean(S["saving_rate"] < self.p.saving_rate_p1))}

    def facts(self, threshold: float = 0.5) -> dict[str, bool]:
        """Account facts that join the node's facts (nodes that differ in them never merge)."""
        sh = self.fact_shares()
        return {k: sh[k] >= threshold for k in ("debt_limit_binding", "xdate_passed", "arrears")}

    # ---------------------------------------------------------------------------------------------- summaries
    def _values(self) -> dict[str, np.ndarray | float]:
        S, F = self.s, self.flows
        z = np.zeros(self.n)
        v = self.views()
        return {
            "fed.total_assets": S["tsy"] + S["mbs"] + S["fed_other_a"], "fed.soma_treasuries": S["tsy"],
            "fed.soma_bills": S["bills"], "fed.soma_mbs": S["mbs"], "fed.currency": S["cur"], "fed.reserves": S["res"],
            "fed.reverse_repo_total": S["rrp"], "fed.tga": S["tga"], "fed.deferred_asset": S["da"],
            "fed.interest_income": F.get("fed_income", z), "fed.interest_expense": F.get("fed_expense", z),
            "fed.net_income": F.get("fed_ni", z), "fed.remittances": F.get("fed_remit", z),
            "fed.rmp_purchases_q": F.get("fed_rmp", z), "fed.inflation_gap": S["core_pce"] - 2.0,
            "fed.unemployment_gap": S["unrate"] - S["nrou"], "fed.iorb": S["iorb"],
            "fiscal.deficit_nipa_gdp": S["def_gdp"], "fiscal.deficit_cash_q": F.get("mts_def_q", z),
            "fiscal.receipts_q": S["receipts_q"], "fiscal.outlays_q": F.get("outlays_q", z),
            "fiscal.net_interest": F.get("nipa_int_saar", z) / 4.0, "fiscal.customs_q": F.get("customs_q", z),
            "fiscal.customs_refunds": F.get("refunds_q", z), "fiscal.fed_remittances": F.get("fed_remit", z),
            "fiscal.debt_held_public": S["dhp"], "fiscal.intragov_holdings": S["intra"],
            "fiscal.statutory_limit": S["limit"],
            "fiscal.limit_headroom": S["limit"] - (S["dhp"] + S["intra"] - self.p.not_subject_to_limit),
            "fiscal.extraordinary_measures_used": S["em_used"], "fiscal.tga": S["tga"], "fiscal.arrears": S["arrears"],
            "fiscal.avg_rate_marketable": S["r_avg"], "fiscal.holders_fed": S["tsy"], "fiscal.holders_foreign": S["fh"],
            "fiscal.interest_burden_nipa": S["int_burden_nipa"], "fiscal.debt_gdp": v["s_debt"],
            "fiscal.limit_binding_share": F.get("limit_binding", z), "fiscal.xdate_flag": F.get("xdate", z),
            "president.gasoline_price": S["gas"], "president.cpi_yoy": S["cpi_yoy"],
            "president.real_dpi_pc_yoy": S["real_dpi_pc_yoy"],
            "president.us_equity_drawdown": (S["eq_index"] / S["eq_peak"] - 1) * 100,
            "president.effective_tariff_rate": S["tau"], "president.spr_inventory": S["spr"],
            "president.unemployment": S["unrate"],
            "china.fx_reserves": S["cn_res"], "china.fx_reserves_change_4q": (S["cn_res"] / S["_res_win"][:, 0] - 1) * 100,
            "china.us_treasury_holdings": S["cn_ust"], "china.gg_deficit_gdp": S["cn_def"],
            "china.private_credit_gdp": S["cn_credit"], "china.cny_per_usd": S["cny"],
            "china.us_bilateral_trade": S["cn_imports_us"],
            "opec.brent": S["brent"], "opec.brent_q_avg_last2": S["_brent_win"].mean(axis=1),
            "opec.us_crude_production": S["us_crude"], "opec.quota_change": S["opec_quota_chg"], "opec.reserves": S["sa_res"],
            "bigtech.revenue": S["bt_revenue"], "bigtech.depreciation": S["bt_depreciation"],
            "bigtech.operating_income": F.get("bt_opinc", S["bt_revenue"] * self.p.ebitda_margin - S["bt_depreciation"]),
            "bigtech.interest": F.get("bt_interest", z), "bigtech.cfo": F.get("bt_cfo", z), "bigtech.capex": S["bt_capex"],
            "bigtech.lease_principal": S["bt_lease_principal"], "bigtech.fcf": F.get("bt_fcf", z),
            "bigtech.buybacks": S["bt_buybacks"], "bigtech.dividends": S["bt_dividends"],
            "bigtech.debt_issuance": F.get("bt_d_debt", z), "bigtech.cash_securities": S["bt_cash"],
            "bigtech.debt": S["bt_debt"], "bigtech.net_cash": S["bt_cash"] - S["bt_debt"],
            "bigtech.op_margin": F.get("bt_opinc", z) / S["bt_revenue"] * 100,
            "voter.saving_rate": S["saving_rate"], "voter.real_dpi_pc_growth": S["real_dpi_pc_yoy"],
            "households.equities": S["hh_eq"], "households.real_estate": S["hh_re"], "households.other_assets": S["hh_ofa"],
            "households.liabilities": S["hh_liab"], "households.net_worth": S["hh_eq"] + S["hh_re"] + S["hh_ofa"] - S["hh_liab"],
            "households.dpi": F.get("dpi_saar", self.p.dpi_share_gdp * S["gdp"]), "households.pce": S["pce_saar"],
            "households.saving_rate": S["saving_rate"],
            "nfc.profits_pbt": S["nfc_pbt"], "nfc.net_interest": S["nfc_ni"],
            "nfc.interest_coverage": (S["nfc_pbt"] + S["nfc_ni"]) / S["nfc_ni"], "nfc.capex": S["nfc_capex"],
            "nfc.debt": S["nfc_debt"],
            "banks.total_assets": S["bk_loans"] + S["bk_sec_cost"] + S["bk_unreal"] + S["bk_other_a"],
            "banks.reserves": S["res"], "banks.securities": S["bk_sec_cost"] + S["bk_unreal"],
            "banks.unrealized_losses": -S["bk_unreal"], "banks.loans": S["bk_loans"], "banks.deposits": S["bk_dep"],
            "banks.nii": S["bk_nii_q"], "banks.chargeoffs": S["bk_chargeoff"],
            "investors.margin_debt": S["margin"],
            "investors.household_equity_share": S["hh_eq"] / (S["hh_eq"] + S["hh_ofa"]) * 100,
            "row.foreign_holdings": S["fh"], "row.foreign_share": foreign_share(S), "row.niip": S["row_niip"],
            "row.current_account_q": S["row_ca_q"],
            "sectoral.government_nl": F.get("sect_gov_nl", z), "sectoral.nation_nl": F.get("sect_nation_nl", z),
            "sectoral.private_nl": F.get("sect_private_nl", z), "sectoral.private_nl_gdp": F.get("sect_private_nl_gdp", z),
            **{f"views.{k}": arr for k, arr in v.items()},
        }

    def summary(self) -> dict[str, dict]:
        """line key -> {p10, median, p90, mean} over particles (the branch view actors see)."""
        out = {}
        for k, v in self._values().items():
            arr = np.asarray(v, float)
            if arr.ndim == 0:
                arr = np.full(self.n, float(arr))
            arr = arr[np.isfinite(arr)]
            if not len(arr):
                continue
            out[k] = {"p10": float(np.percentile(arr, 10)), "median": float(np.median(arr)),
                      "p90": float(np.percentile(arr, 90)), "mean": float(arr.mean())}
        return out

    def detail(self, root: Mapping | None = None) -> dict:
        """The node detail's accounts{book}: lines {id, label, median, p10, p90, unit, source, as_of, update},
        the identities and the gates (§12.4)."""
        summ = self.summary()
        books = {}
        for row in LINES:
            ln = dict(zip(LINE_KEYS, row))
            key = f"{ln['book']}.{ln['id']}"
            s = summ.get(key)
            entry = {"id": ln["id"], "label": ln["zh"], "unit": ln["unit"], "statement": ln["statement"],
                     "update": ln["update"], "update_zh": UPDATE_CLASSES.get(ln["update"], ln["update"]),
                     "calibration": ln["calibration"], "source": ln["src"], "series": ln["series"],
                     "visibility": ln["visibility"]}
            if s:
                entry.update({"median": s["median"], "p10": s["p10"], "p90": s["p90"]})
            elif self.carried.get(key) is not None:
                entry.update({"median": self.carried[key], "p10": self.carried[key], "p90": self.carried[key],
                              "carried": True})
            if root:
                rl = (((root.get("books") or {}).get(ln["book"]) or {}).get("lines") or {}).get(ln["id"])
                if rl:
                    entry["as_of"] = rl.get("as_of")
                    if rl.get("stale"):
                        entry["stale"] = rl["stale"]
            books.setdefault(ln["book"], {"zh": BOOKS[ln["book"]]["zh"], "lines": []})["lines"].append(entry)
        return {"quarter": self.quarter, "n": self.n, "books": books,
                "identities": {k: {"residual": v, "ok": v <= (TOL * 10 if k.startswith(("I3", "I4_public")) else TOL),
                                   "zh": IDENTITIES.get(k.split("_")[0], "")} for k, v in self.resid.items()},
                "gates": gates(self), "views": {k: summ.get(f"views.{k}") for k in VIEW_FIELDS},
                "facts": self.fact_shares()}


# =====================================================================================================================
# 5. Gates (LEDGERS §7.1): accounts -> menus and prior tilts for the next quarter
# =====================================================================================================================

def gates(acc: Accounts) -> dict:
    """G-L1…G-L9 from the books (per actor); hard gates add options, soft gates multiply priors (判断值)."""
    p, S, F = acc.p, acc.s, acc.flows
    room = S["limit"] - (S["dhp"] + S["intra"] - p.not_subject_to_limit)
    next_need = np.maximum(F.get("mts_def_q", np.full(acc.n, 450.0)) + p.omf_q + p.intragov_growth_q, 0.0)
    suspended = S["susp_q"] > 0
    p_bind = float(np.mean((room < next_need) & ~suspended))
    em_left = np.maximum(p.em_capacity - S["em_used"], 0.0)
    p_x = float(np.mean((S["tga"] + em_left + np.maximum(room, 0.0) < next_need + p.tga_floor) & ~suspended))
    out = {"congress": {"p_limit_binds_next_q": p_bind, "p_xdate_next_q": p_x,
                        "require_options": (["debt_limit_raise", "debt_limit_suspend", "debt_limit_standoff"]
                                            if p_bind >= 0.5 else []),
                        "interest_burden_nipa": float(np.median(S["int_burden_nipa"])),
                        "note": "G-L1 硬门（P ≥ 0.5，判断值）；A11：2011 年以来 10/10 次解决，中位 136 天，0 次违约；"
                                "G-L3 利息负担只显示"},
           "treasury": {"require_options": ["extraordinary_measures"] if p_bind >= 0.5 else [], "tga_floor": p.tga_floor,
                        "p_xdate_next_q": p_x}}
    out["synth"] = {"xdate_mechanism_allowed": p_x >= 0.5,
                    "saving_rate_below_p1": float(np.median(S["saving_rate"])) < p.saving_rate_p1,
                    "hyperscaler_fcf_negative": float(np.median(F.get("bt_fcf", np.zeros(acc.n)))) < 0,
                    "note": "G-L2：越过 X 日期是合成器允许的机制（超出历史）；G-L9：储蓄率低于 1959 年以来第 1 百分位是需求轴"
                            "有界倾斜的理由，本身不是机制"}
    res_med = float(np.median(S["res"]))
    out["fed"] = {"deferred_asset": float(np.median(S["da"])), "ni_q": float(np.median(F.get("fed_ni", np.zeros(acc.n)))),
                  "reserves": res_med, "reserves_below_ample": res_med < p.ample_reserves,
                  "tilt": {"bs_runoff": 0.5} if res_med < p.ample_reserves else {},
                  "per_25bp_iorb_bn_per_year": CALIBRATIONS["A5"]["per_25bp_iorb_bn_per_year"],
                  "note": "G-L4 只显示（递延资产会计让美联储可以带亏运行）；G-L5：准备金低于缩表结束时的水平，恢复缩表 ×0.5（判断值）"}
    chg4 = float(np.median(S["cn_res"] / S["_res_win"][:, 0] - 1.0))
    out["china"] = {"reserves_change_4q_pct": chg4 * 100, "reserves_falling": chg4 < -0.10,
                    "tilt": {"cny_weaker": 0.5} if chg4 < -0.10 else {},
                    "note": "G-L6：储备 4 季降超 10%，允许人民币走弱的先验 ×0.5（判断值，只有 2015–16 一例）"}
    cash_room = S["bt_cash"] - p.cash_floor_rev * S["bt_revenue"]
    fcf = F.get("bt_fcf", np.zeros(acc.n))
    raise_need = np.maximum(-(cash_room + 4 * (fcf - 0.15 * S["bt_capex"])), 0.0)
    ebitda4 = np.maximum(4 * F.get("bt_ebitda", S["bt_revenue"] * p.ebitda_margin), 1e-9)
    nd_ebitda = float(np.median((S["bt_debt"] - S["bt_cash"]) / ebitda4))
    payout_fcf = float(np.median(fcf - S["bt_buybacks"] - S["bt_dividends"]))
    out["bigtech"] = {"debt_needed_if_raise_15pct_4q": float(np.median(raise_need)), "net_debt_to_ebitda": nd_ebitda,
                      "fcf_after_payout": payout_fcf,
                      "tilt": {"capex_raise_large": 0.5 if nd_ebitda > 1.0 else 1.0,
                               "buyback_up": 0.5 if float(np.median(fcf)) < 0 else 1.0},
                      "note": "G-L7（判断值）：派现后自由现金流为负 → 加速回购 ×0.5；净债务/EBITDA > 1 倍 → 大幅上调 ×0.5"}
    be = acc.carried.get("opec.fiscal_breakeven")
    below = bool(be) and bool(np.median(S["_brent_win"].max(axis=1)) < float(be))
    out["opec"] = {"below_breakeven_2q": below, "breakeven": be,
                   "note": "G-L8：布伦特连续 2 季低于财政盈亏平衡油价 → 崩跌行先验（盈亏平衡未锁定时不触发）"}
    return out


# =====================================================================================================================
# 6. Observation packets (LEDGERS §6; AGENT_DESIGN §5.4)
# =====================================================================================================================

# actors whose own lines come from quarterly releases (Z.1, NIPA, DFA, 10-Q, IMF): on a branch they are the model's
# estimate, labelled 估计（官方数据尚未发布）; H.4.1, DTS and MTS lines count as published (U-14)
ESTIMATED_BOOKS = {"bigtech", "voter", "china", "opec", "belligerents"}
ZH = {"fed": "美联储", "congress": "国会", "treasury": "财政部", "president": "总统与白宫", "china": "中国", "opec": "欧佩克+",
      "bigtech": "超大型科技公司", "voter": "选民", "courts": "法院", "belligerents": "冲突方", "nature": "自然"}
OWN_BLOCK_MAX = 400


def sig3(v) -> str:
    """Display precision (3 significant figures; particle noise below it does not change the text or the cache key)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def est_tokens(s: str) -> int:
    """PROMPTS' convention (unmeasured): one token per CJK character, four ASCII characters per token."""
    cjk = sum(1 for ch in s if "一" <= ch <= "鿿" or "　" <= ch <= "〿" or "＀" <= ch <= "￯")
    return int(round(cjk + (len(s) - cjk) / 4.0))


def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@dataclass
class Signal:
    actor: str
    text: str
    instrument: str | None = None
    direction: int = 0
    horizon_q: int = 1
    condition: str | None = None
    commit: bool = False
    private_note: str | None = None      # never rendered for other actors
    quarter: str | None = None


@dataclass
class Credibility:
    """Beta(2,2) per actor (判断值): a matured signal is kept if the realised direction matches, excused if its
    stated condition fired, broken otherwise; shown as 可信度 0.xx（k/n）."""
    a: float = 2.0
    b: float = 2.0
    kept: int = 0
    total: int = 0

    def update(self, kept: bool, excused: bool = False) -> None:
        if excused:
            return
        self.total += 1
        if kept:
            self.kept += 1
            self.a += 1
        else:
            self.b += 1

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def label(self) -> str:
        return f"可信度 {self.mean:.2f}（{self.kept}/{self.total}）"


def settle_signal(sig: Signal | Mapping, realized_direction: int, condition_fired: bool = False) -> tuple[bool, bool]:
    """(kept, excused) for a matured signal."""
    d = sig.direction if isinstance(sig, Signal) else int(sig.get("direction", 0))
    if condition_fired:
        return False, True
    return (int(np.sign(realized_direction)) == int(np.sign(d))), False


class _View:
    """Formats a line for an own block: data values at the root, median (10–90%) on a branch."""

    def __init__(self, root: Mapping | None = None, acc: Accounts | None = None):
        self.root = root
        self.acc = acc
        self._summ = acc.summary() if acc is not None else {}

    @property
    def branch(self) -> bool:
        return self.acc is not None

    def num(self, key: str, default=None):
        if self.acc is not None and key in self._summ:
            return self._summ[key]["median"]
        if self.root is not None:
            b, line = key.split(".", 1)
            return root_value(self.root, b, line, default)
        return default

    def fmt(self, key: str, unit: str = "") -> str:
        if self.acc is not None and key in self._summ:
            s = self._summ[key]
            if abs(s["p90"] - s["p10"]) < 1e-9 or sig3(s["p10"]) == sig3(s["p90"]):
                return f"{sig3(s['median'])}{unit}"
            return f"{sig3(s['median'])}{unit}（{sig3(s['p10'])}–{sig3(s['p90'])}）"
        v = self.num(key)
        return "—" if v is None else f"{sig3(v)}{unit}"


def _hist(root: Mapping | None, book: str, line: str, n: int = 4) -> list:
    if not root:
        return []
    h = (((root.get("books") or {}).get(book) or {}).get("history") or {}).get(line) or {}
    return [v for v in list(h.values())[-n:] if v is not None]


def own_block(actor: str, root: Mapping | None = None, acc: Accounts | None = None, *, quarter_label: str | None = None,
              alerts: list[str] | None = None, private_note: str | None = None, show_private: bool = False,
              institutions: Mapping | None = None) -> str:
    """The actor's own account block (≤ 6 lines + ≤ 2 history rows + constraint alerts + ≤ 2 watched lines).

    At the root (acc None) the data values with data dates; on a branch the branch's median (10–90%). The private
    note is rendered only with show_private (the actor's own separate call)."""
    v = _View(root, acc)
    inst = dict(institutions or {})
    head = quarter_label or (f"{acc.quarter} 季末；本分支中位数（10–90%）" if acc is not None else
                             _root_label(actor, root))
    if acc is not None and actor in ESTIMATED_BOOKS:      # U-14: quarterly releases lag; H.4.1, DTS, MTS do not
        head += "；估计（官方数据尚未发布）"
    L: list[str] = []
    hist: dict[str, list] = {}
    al = list(alerts or [])
    g = gates(acc) if acc is not None else None
    if actor == "fed":
        L.append(f"资产：总资产 {v.fmt('fed.total_assets')}｜国债 {v.fmt('fed.soma_treasuries')}（其中国库券 {v.fmt('fed.soma_bills')}）｜"
                 f"MBS {v.fmt('fed.soma_mbs')}")
        L.append(f"负债：准备金 {v.fmt('fed.reserves')}｜逆回购 {v.fmt('fed.reverse_repo_total')}｜TGA {v.fmt('fed.tga')}｜"
                 f"流通货币 {v.fmt('fed.currency')}")
        if v.branch:
            L.append(f"损益：本季净收入 {v.fmt('fed.net_income')}｜递延资产 {v.fmt('fed.deferred_asset')}")
            L.append(f"使命：核心通胀缺口 {v.fmt('fed.inflation_gap', 'pp')}｜就业缺口 {v.fmt('fed.unemployment_gap', 'pp')}")
        else:
            L.append(f"损益：近4季净收入 {v.fmt('fed.net_income_last4q')}｜递延资产 {v.fmt('fed.deferred_asset')}"
                     f"（峰值 {v.fmt('fed.deferred_asset_peak')}）｜准备金利率每 25bp ≈ 8.30/年")
            L.append(f"操作：近13周国库券 {_signed(v.num('fed.soma_bills_change_13w'))}（准备金管理购买）｜"
                     f"MBS {_signed(v.num('fed.soma_mbs_change_13w'))}（到期）")
            L.append(f"使命：核心 PCE {v.fmt('fed.core_pce_yoy', '%')}（缺口 {_signed(v.num('fed.inflation_gap'))}pp）｜"
                     f"失业率 {v.fmt('fed.unemployment', '%')}，CBO 自然率 {v.fmt('fed.nrou_cbo', '%')}")
            hist["准备金"] = _hist(root, "fed", "reserves")
        if v.num("fed.on_rrp", 100) is not None and (v.num("fed.on_rrp", 100) or 100) < 20 and not v.branch:
            al.append("隔夜逆回购已近零：TGA 每增 1，准备金减 1")
        months = v.num("fiscal.months_to_limit_at_6m_pace")
        if not v.branch and months is not None:          # watched line: the federal headroom and the TGA path
            al.append(f"债务上限约 {months:.1f} 个月后触及，其间 TGA 下降会推高准备金")
        al += _limit_alert(v, g, "TGA 下降会推高准备金")
    elif actor in ("congress", "treasury"):
        if actor == "congress":
            if v.branch:
                L.append(f"收支：现金赤字 {v.fmt('fiscal.deficit_cash_q')}/季｜收入 {v.fmt('fiscal.receipts_q')}｜"
                         f"利息/GDP {v.fmt('fiscal.interest_burden_nipa', '%')}")
            else:
                L.append(f"上一财年：收入 {v.fmt('fiscal.receipts_fy')}｜支出 {v.fmt('fiscal.outlays_fy')}｜"
                         f"赤字 {v.fmt('fiscal.deficit_fy')}｜净利息 {v.fmt('fiscal.net_interest_fy')}"
                         f"（占收入 {v.fmt('fiscal.net_interest_share_receipts', '%')}）")
            L.append(f"债务：公众持有 {v.fmt('fiscal.debt_held_public')}（GDP 的 {v.fmt('fiscal.debt_gdp', '%')}）｜"
                     f"上限 {v.fmt('fiscal.statutory_limit')}｜余量 {v.fmt('fiscal.limit_headroom')}")
            if not v.branch:
                L.append(f"利息：可交易国债平均利率 {v.fmt('fiscal.avg_rate_marketable', '%')}（一年前 "
                         f"{v.fmt('fiscal.avg_rate_marketable_12m_ago', '%')}）；利息/GDP {v.fmt('fiscal.interest_burden_nipa', '%')}")
                L.append(f"关税：2026 年 5 月以来退款 {v.fmt('fiscal.customs_refunds_since_2026_05')}；近 3 月毛收入 "
                         f"{v.fmt('fiscal.customs_gross_3m')}")
                hist["赤字（季）"] = _hist(root, "fiscal", "deficit_q")
            else:
                L.append(f"非常规措施已用 {v.fmt('fiscal.extraordinary_measures_used')}｜TGA {v.fmt('fiscal.tga')}")
            L.append("政治：两院多数与席位差、拨款到期日（制度现状，用户锁定）")
        else:
            L.append(f"可交易国债：国库券 {v.fmt('fiscal.mspd_bills')}｜中期 {v.fmt('fiscal.mspd_notes')}｜长期 "
                     f"{v.fmt('fiscal.mspd_bonds')}｜国库券占比 {v.fmt('fiscal.bill_share', '%')}（TBAC 建议 15–20%）")
            L.append(f"现金：TGA {v.fmt('fiscal.tga') if v.branch else v.fmt('fiscal.tga_dts')}｜持有人：美联储 "
                     f"{v.fmt('fed.soma_treasuries')}｜外国 {v.fmt('fiscal.foreign_holdings_tic')}")
        pace = v.num("fiscal.debt_pace_6m")
        months = v.num("fiscal.months_to_limit_at_6m_pace")
        if not v.branch and pace and months is not None:
            al.append(f"近 6 个月月均新增债务 {sig3(pace)}，约 {months:.1f} 个月触及上限；2011 年以来 10 次触及上限，"
                      f"中位 136 天后解决，0 次违约")
        al += _limit_alert(v, g, "")
        if actor == "treasury":
            al.append(f"非常规措施容量 {sig3(inst.get('em_capacity_bn') or 350)}、TGA 目标 {sig3(inst.get('tga_target_bn') or 850)}"
                      "（判断值，请锁定）")
    elif actor == "president":
        L.append(f"选民成绩单：汽油 {v.fmt('president.gasoline_price')} 美元/加仑｜CPI {v.fmt('president.cpi_yoy', '%')}｜"
                 f"失业率 {v.fmt('president.unemployment' if v.branch else 'fed.unemployment', '%')}｜"
                 f"人均实际可支配收入 {v.fmt('president.real_dpi_pc_yoy', '%')}")
        L.append(f"工具存量：有效关税率 {v.fmt('president.effective_tariff_rate', '%')}｜战略石油储备 "
                 f"{v.fmt('president.spr_inventory')} 百万桶" + ("" if v.branch else
                                                         f"（一年 {_signed(v.num('president.spr_change_52w'))}）"))
        L.append(f"美股距高点 {v.fmt('president.us_equity_drawdown', '%')}（已实现，不预测）")
        appr = inst.get("approval")
        L.append(f"支持率 {sig3(appr)}%（锁定）" if appr is not None else "支持率与选举赔率：用户锁定（Gallup / APP；市场赔率）")
    elif actor == "china":
        L.append(f"对外：外汇储备 {v.fmt('china.fx_reserves' if v.branch else 'china.fx_reserves_ex_gold')}｜持有美债 "
                 f"{v.fmt('china.us_treasury_holdings')}｜人民币 {v.fmt('china.cny_per_usd')}")
        if v.branch:
            L.append(f"财政与信贷：广义政府赤字 {v.fmt('china.gg_deficit_gdp', '%')} GDP｜私人信贷 {v.fmt('china.private_credit_gdp', '%')}")
        else:
            gg = v.num("china.gg_net_lending_gdp_2026")
            L.append(f"财政与信贷：广义政府赤字 {sig3(-gg) if gg is not None else '—'}% GDP（IMF 官方口径）｜政府债务 "
                     f"{v.fmt('china.gg_gross_debt_gdp_2026', '%')}｜私人信贷 {v.fmt('china.private_credit_gdp', '%')}")
            L.append(f"增长：2026 年 {v.fmt('china.real_gdp_growth_2026', '%')}（WEO）｜目标："
                     + (f"{sig3(inst['china_growth_target'])}%" if inst.get("china_growth_target") else "用户锁定")
                     + f"｜实际房价距峰值 {v.fmt('china.real_house_price_vs_peak', '%')}")
        L.append(f"对美贸易：美国自华进口 {v.fmt('china.us_bilateral_trade' if v.branch else 'china.us_imports_from_china_12m')}/12 月")
        al.append("美元每升 1%，储备估值约降 0.40%（A7）")
        if g and g["china"]["reserves_falling"]:
            al.append("储备 4 季降超 10%")
    elif actor == "opec":
        bq = _hist(root, "opec", "brent_quarterly_avg")
        L.append(f"油价：布伦特 {v.fmt('opec.brent' if v.branch else 'opec.brent_spot')}" +
                 ("" if v.branch else "｜季均 " + "/".join(sig3(x) for x in bq)))
        L.append(f"沙特：储备 {v.fmt('opec.reserves' if v.branch else 'opec.saudi_reserves_ex_gold')}｜财政 "
                 f"{v.fmt('opec.sau_gg_net_lending_gdp_2026', '%')} GDP｜增长 {v.fmt('opec.sau_real_gdp_growth_2026', '%')}（2026 WEO）")
        L.append(f"成员增长（2026 WEO）：科威特 {v.fmt('opec.kwt_real_gdp_growth_2026', '%')}｜伊拉克 "
                 f"{v.fmt('opec.irq_real_gdp_growth_2026', '%')}｜阿联酋 {v.fmt('opec.are_real_gdp_growth_2026', '%')}")
        L.append(f"竞争者：美国原油产量 {v.fmt('opec.us_crude_production')} 百万桶/日")
        L.append("配额、实际产量、财政盈亏平衡油价：用户锁定（OPEC 月报、EIA STEO、IMF REO）")
        if show_private and inst.get("opec_spare_mbd") is not None:
            L.append(f"（仅你可见）可达市场的闲置产能 {sig3(inst['opec_spare_mbd'])} 百万桶/日")
    elif actor == "bigtech":
        L.append(f"损益：收入 {v.fmt('bigtech.revenue')}｜经营利润 {v.fmt('bigtech.operating_income')}｜折旧 {v.fmt('bigtech.depreciation')}")
        L.append(f"现金流：经营 {v.fmt('bigtech.cfo')}｜资本开支 {v.fmt('bigtech.capex')}" +
                 ("" if v.branch else f"（近 4 季 {v.fmt('bigtech.capex_ttm')}，一年前 {v.fmt('bigtech.capex_ttm_prev')}）") +
                 f"｜自由现金流 {v.fmt('bigtech.fcf')}")
        L.append(f"股东回报：回购 {v.fmt('bigtech.buybacks')}｜股息 {v.fmt('bigtech.dividends')}" +
                 ("" if v.branch else f"（近 4 季合计 {v.fmt('bigtech.payouts_ttm')}）"))
        if v.branch:
            L.append(f"资产负债：现金与证券 {v.fmt('bigtech.cash_securities')}｜债务 {v.fmt('bigtech.debt')}｜净现金 "
                     f"{v.fmt('bigtech.net_cash')}｜新增债务 {v.fmt('bigtech.debt_issuance')}")
        else:
            cs = (v.num("bigtech.cash") or 0) + (v.num("bigtech.securities") or 0)
            L.append(f"资产负债：现金与证券 {sig3(cs)}｜债务 {v.fmt('bigtech.debt')}（4 季 {_signed(v.num('bigtech.debt_change_4q'))}）｜"
                     f"净现金 {v.fmt('bigtech.net_cash')}")
            ratios = [f"{tk} {sig3(root_value(root, 'bigtech_' + tk.lower(), 'capex_to_cfo'))}" for tk in BT
                      if root_value(root, "bigtech_" + tk.lower(), "capex_to_cfo") is not None]
            if ratios:
                L.append("资本开支/经营现金流：" + "｜".join(ratios))
            hist["资本开支"] = _hist(root, "bigtech", "capex")
        if (v.num("bigtech.fcf") or 0) < 0:
            al.append("资本开支已超过经营现金流：再上调需发债或削减派息")
        if g:
            need = g["bigtech"]["debt_needed_if_raise_15pct_4q"]
            if need > 0:
                al.append(f"若未来 4 季上调 15%，约需新增债务 {sig3(need)}")
    elif actor == "voter":
        L.append(f"财富：净资产 {v.fmt('households.net_worth')}｜股票与基金占金融资产 "
                 f"{v.fmt('investors.household_equity_share' if v.branch else 'households.equity_share_fin_assets', '%')}")
        if not v.branch:
            L.append(f"分布（净资产份额）：前 1% {v.fmt('households.nw_share_top1', '%')}｜90–99% {v.fmt('households.nw_share_90_99', '%')}｜"
                     f"50–90% {v.fmt('households.nw_share_50_90', '%')}｜后 50% {v.fmt('households.nw_share_bottom50', '%')}")
        L.append(f"收支：储蓄率 {v.fmt('households.saving_rate', '%')}｜人均实际可支配收入 "
                 f"{v.fmt('president.real_dpi_pc_yoy', '%')}｜CPI {v.fmt('president.cpi_yoy', '%')}｜汽油 {v.fmt('president.gasoline_price')}")
    elif actor == "courts":
        L.append("待决案件、阶段、预计判决季度：制度现状，用户锁定")
        L.append(f"涉及金额：关税退款已付 {v.fmt('fiscal.customs_refunds_since_2026_05')}（2026 年 5 月以来，MTS 表 4）；剩余义务 "
                 f"{sig3(inst.get('refunds_remaining_bn') or 0)}")
    elif actor == "belligerents":
        L.append("模板（身份与数字由用户锁定）：石油出口量与收入｜财政余额（IMF WEO）｜储备（含被冻结部分）｜军费（SIPRI）｜制裁状态")
        L.append(f"可见的足迹：布伦特 {v.fmt('opec.brent' if v.branch else 'opec.brent_spot')}｜科威特与伊拉克 2026 年增长 "
                 f"{v.fmt('opec.kwt_real_gdp_growth_2026', '%')} / {v.fmt('opec.irq_real_gdp_growth_2026', '%')}（WEO）")
    else:
        L.append("（没有账户）")
    al = [x for x in al if x]
    out = [f"## 你的账户：{ZH.get(actor, actor)}（{head}；十亿美元，除非注明）"] + L[:6]
    for k, vals in list(hist.items())[:2]:
        if vals:
            out.append(f"近{len(vals)}季{k}：" + "/".join(sig3(x) for x in vals))
    w = _watched(actor, v)
    if w:
        out.append("关注（他人账户）：" + "｜".join(w[:2]))
    if al:
        out.append("约束提示：" + "；".join(dict.fromkeys(al)))
    if private_note and show_private:
        out.append(f"（仅你可见）你上季的内部打算：{private_note}")
    return "\n".join(out)


def _watched(actor: str, v: "_View") -> list[str]:
    """<= 2 lines the actor watches in other books (LEDGERS §6.3): the Fed the federal headroom and TGA, OPEC+ the
    US SPR, China the US customs receipts, Congress the voters' real income."""
    out: list[str] = []
    if actor == "fed":
        out.append(f"联邦上限余量 {v.fmt('fiscal.limit_headroom')}")
        out.append(f"财政部 TGA {v.fmt('fiscal.tga' if v.branch else 'fiscal.tga_dts')}")
    elif actor == "opec":
        out.append(f"美国战略石油储备 {v.fmt('president.spr_inventory')} 百万桶")
    elif actor == "china":
        out.append(f"美国关税收入 {v.fmt('fiscal.customs_q')}/季" if v.branch else
                   f"美国关税毛收入 {v.fmt('fiscal.customs_gross_3m')}/3 月")
    elif actor == "congress":
        out.append(f"选民人均实际可支配收入 {v.fmt('president.real_dpi_pc_yoy', '%')}")
    return [x for x in out if "—" not in x]


def account_events(prev: Mapping[str, float] | None, acc: Accounts, threshold: float = 0.5) -> list[tuple[str, str]]:
    """News-digest items (type "ledger") for account thresholds crossed this quarter, from fact shares before and
    after (prev = the parent's Accounts.fact_shares(); None at the root): only what has happened."""
    cur = acc.fact_shares()
    before = dict(prev or {})
    items = []
    texts = {"debt_limit_binding": "联邦债务触及法定上限，财政部动用非常规措施",
             "xdate_passed": "TGA 低于下限：越过 X 日期的风险出现", "arrears": "联邦出现逾期未付（技术性违约）",
             "hyperscaler_fcf_negative": "超大型科技公司自由现金流为负", "saving_rate_below_p1": "家庭储蓄率跌破 1959 年以来第 1 百分位"}
    for k, text in texts.items():
        if cur.get(k, 0.0) >= threshold > before.get(k, 0.0):
            items.append(("ledger", f"{text}（{100 * cur[k]:.0f}% 粒子）"))
    return items


def _root_label(actor: str, root: Mapping | None) -> str:
    book = {"congress": "fiscal", "treasury": "fiscal", "voter": "households"}.get(actor, actor)
    line = {"fed": "total_assets", "fiscal": "statutory_limit", "president": "gasoline_price", "china": "fx_reserves_ex_gold",
            "opec": "brent_spot", "bigtech": "capex", "households": "net_worth"}.get(book)
    try:
        return str(root["books"][book]["lines"][line]["as_of"]) if (root and line) else "用户锁定"
    except (KeyError, TypeError):
        return "数据日期见账户"


def _signed(v) -> str:
    if v is None:
        return "—"
    s = sig3(v)
    return s if s.startswith("-") else "+" + s


def _limit_alert(v: _View, g: dict | None, tail: str) -> list[str]:
    if g is None:
        return []
    pb = g["congress"]["p_limit_binds_next_q"]
    px = g["congress"]["p_xdate_next_q"]
    out = []
    if pb >= 0.05:
        out.append(f"下季债务上限约束的粒子 {100 * pb:.0f}%" + (f"：{tail}" if tail else ""))
    if px >= 0.05:
        out.append(f"X 日期风险的粒子 {100 * px:.0f}%")
    return out


DIGEST_ORDER = {"event": 0, "action": 1, "ledger": 2, "consequence": 3, "data": 4, "fact": 5}
_PRICE = r"(股价|美股|纳指|标普|收益率|油价|金价|美元指数|汇率|利差|估值|市盈率|房价)"
_FUT = r"(将|会|预计|有望|料将|可能会|跌至|涨至|升至|降至|突破)"
PRICE_FORECAST_RE = re.compile(_FUT + r".{0,12}" + _PRICE + "|" + _PRICE + r".{0,12}" + _FUT)


def digest_guard(items: Iterable[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[str]]:
    """T-L10: drop any digest item that forecasts a market price (only what has happened is news)."""
    kept, dropped = [], []
    for kind, text in items:
        if PRICE_FORECAST_RE.search(text):
            dropped.append(text)
        else:
            kept.append((kind, text))
    return kept, dropped


def news_digest(items: Iterable[tuple[str, str]]) -> str:
    """≤ 6 items, ≤ 50 characters each, fixed type order (event, action, ledger, consequence, data, fact)."""
    kept, _ = digest_guard(items)
    kept = sorted(kept, key=lambda t: DIGEST_ORDER.get(t[0], 9))[:6]
    return "# 新闻摘要（上季已发生的事；没有任何预测）\n" + "\n".join(f"- {t[:50]}" for _, t in kept)


def root_digest_items(root: Mapping, institutions: Mapping | None = None) -> list[tuple[str, str]]:
    """The root's news digest from the data (code-filled; the pinned facts decide the FOMC line's wording)."""
    v = lambda b, l: root_value(root, b, l)   # noqa: E731
    inst = dict(institutions or {})
    items = []
    fomc = inst.get("fomc_2026_10", "hold")
    chg = v("fed", "iorb_change_at_oct28_meeting")
    if fomc == "hold" and (chg is None or abs(chg) < 1e-9):
        items.append(("action", f"10 月 28 日议息会议：利率区间维持（准备金利率前后均为 {sig3(v('fed', 'iorb'))}%）"))
    else:
        items.append(("action", f"10 月 28 日议息会议：{ {'cut_25': '降息 25bp', 'cut_50': '降息 50bp', 'hike_25': '加息 25bp'}.get(fomc, '维持') }"))
    if v("fiscal", "limit_headroom") is not None:
        items.append(("ledger", f"联邦债务距法定上限 {v('fiscal', 'limit_headroom'):.0f} 十亿美元"))
    if v("bigtech", "capex") is not None and v("bigtech", "cfo") is not None:
        more = "超过" if v("bigtech", "capex") > v("bigtech", "cfo") else "低于"
        items.append(("ledger", f"五家云公司上季资本开支 {v('bigtech', 'capex'):.0f} 十亿美元，{more}经营现金流"))
    if v("fiscal", "customs_refunds_since_2026_05"):
        items.append(("ledger", f"2026 年 5 月以来退还关税 {v('fiscal', 'customs_refunds_since_2026_05'):.0f} 十亿美元"))
    if v("fed", "core_pce_yoy") is not None and v("fed", "unemployment") is not None:
        items.append(("data", f"最新核心 PCE 同比 {v('fed', 'core_pce_yoy'):.1f}%，失业率 {v('fed', 'unemployment'):.1f}%"))
    if v("opec", "brent_spot") is not None and v("president", "spr_change_52w") is not None:
        items.append(("fact", f"布伦特最新 {v('opec', 'brent_spot'):.0f} 美元；战略石油储备一年减少 "
                              f"{abs(v('president', 'spr_change_52w')):.0f} 百万桶"))
    return items


def signal_board(signals: Iterable[Signal], cred: Mapping[str, Credibility], viewer: str | None = None) -> str:
    """其他人的意思: each other actor's public signal from last quarter, tagged 表态 / 承诺, with credibility.
    Private notes never appear here (they belong only to their owner's own separate call)."""
    rows = ["# 各方上季的公开表态（可信度 = 过去表态兑现的比例，Beta 后验均值）"]
    for s in signals:
        if viewer and s.actor == viewer:
            continue
        c = cred.get(s.actor, Credibility())
        tag = "承诺" if s.commit else "表态"
        rows.append(f"- {ZH.get(s.actor, s.actor)}［{tag}，{c.label()}］{s.text[:40]}")
    return "\n".join(rows)


def regime_line(state: Mapping, facts: Mapping, quarter: str, acc: Accounts | None = None) -> str:
    """The shared state table's one regime line (core > 4%, credit-stress percentile, oil disruption, debt limit
    binding, election quarter)."""
    from .actors import ELECTIONS
    parts = []
    core = state.get("s_core_yoy")
    if core is not None:
        parts.append(f"通胀体制：{'高通胀（核心 > 4%）' if float(core) > 4 else '常态'}")
    pct = state.get("s_spread_pctile")
    if pct is not None:
        parts.append(f"信用压力分位 {float(pct):.0f}")
    parts.append("石油供给中断：" + ("持续" if facts.get("oil_disruption_ongoing") else "无"))
    if acc is not None:
        sh = acc.fact_shares()["debt_limit_binding"]
        parts.append(f"债务上限约束：{100 * sh:.0f}% 粒子")
    elif facts.get("debt_limit_binding"):
        parts.append("债务上限约束：是")
    e = ELECTIONS.get(quarter)
    parts.append(f"选举季：{e['zh']}" if e else "选举季：否")
    return "体制：" + "｜".join(parts)


def packet(actor: str, own: str, digest: str, board: str, dashboard_ref: str = "（共享状态表见第一节）") -> dict:
    """The actor's packet text with its sizes and hash (equal states give equal text: cache hits)."""
    text = "\n\n".join([own, digest, board, dashboard_ref])
    return {"actor": actor, "text": text, "chars": len(text), "tokens_est": est_tokens(text), "hash": text_hash(text),
            "own_chars": len(own), "own_tokens_est": est_tokens(own)}


def root_packets(root: Mapping, *, signals: Iterable[Signal] = (), cred: Mapping | None = None,
                 institutions: Mapping | None = None, actors: Iterable[str] | None = None,
                 show_private_for: str | None = None) -> dict[str, dict]:
    """Every root actor's packet (separate mode renders the owner's private items only for `show_private_for`)."""
    from .actors import institution_values
    inst = dict(institutions) if institutions is not None else institution_values()
    dig = news_digest(root_digest_items(root, inst))
    sig = list(signals)
    out = {}
    for a in (actors or ("voter", "courts", "belligerents", "president", "congress", "china", "opec", "bigtech", "fed")):
        board = signal_board(sig, cred or {}, viewer=a)
        own = own_block(a, root, None, institutions=inst, show_private=(show_private_for == a),
                        private_note=next((s.private_note for s in sig if s.actor == a and show_private_for == a), None))
        out[a] = packet(a, own, dig, board)
    return out


def sector_summary(acc: Accounts | None = None, root: Mapping | None = None) -> str:
    """部门账户摘要 for the synthesizer (≤ 250 characters; LEDGERS §6.3)."""
    v = _View(root, acc)
    parts = []
    sr = v.num("households.saving_rate")
    if sr is not None:
        parts.append(f"家庭储蓄率 {sig3(sr)}%" + ("（低于 1959 年以来第 1 百分位）" if sr < (acc.p.saving_rate_p1 if acc else 2.30) else ""))
    if acc is not None:
        ul = acc.summary()['banks.unrealized_losses']['median']
        parts.append(f"银行证券未实现{'损失' if ul >= 0 else '收益'} {sig3(abs(ul))}")
        parts.append(f"企业利息覆盖 {sig3(acc.summary()['nfc.interest_coverage']['median'])} 倍")
        parts.append(f"新增债务的外国份额 {acc.p.foreign_share_issuance:.2f}")
        pnl = acc.summary().get("sectoral.private_nl_gdp", {}).get("median")
        if pnl is not None:
            lo, hi = acc.p.private_nl_p01_p99
            parts.append(f"私人净借出 {sig3(pnl)}% GDP（{'在' if lo <= pnl <= hi else '超出'}历史区间）")
        fs = acc.fact_shares()
        flags = [z for z, k in (("上限约束", "debt_limit_binding"), ("X 日期", "xdate_passed"),
                                ("科技公司自由现金流 < 0", "hyperscaler_fcf_negative")) if fs[k] >= 0.5]
        if flags:
            parts.append("约束标志：" + "、".join(flags))
    else:
        md = root_value(root or {}, "investors", "margin_debt_yoy")
        if md is not None:
            parts.append(f"保证金负债同比 {sig3(md)}%")
        cov = root_value(root or {}, "nfc", "interest_coverage")
        if cov is not None:
            parts.append(f"企业利息覆盖 {sig3(cov)} 倍")
    return ("部门账户摘要：" + "；".join(parts))[:250]


# =====================================================================================================================
# 7. The build step
# =====================================================================================================================

def run(ctx=None, *, refresh: bool = False, download: bool = True, fetch: Fetcher | None = None,
        results_dir: Path | None = None) -> dict:
    """``python -m scenariolab.build --only agents_accounts``: the root accounts, the history priors re-estimated
    from local data, and the pinned-facts file (written with its 未确认 defaults only when missing)."""
    from . import actors, priors
    out_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    actors.ensure_institutions_file()
    fetch = fetch or Fetcher(refresh=refresh, download=download)
    root = build_root(fetch)
    n_lines = sum(len(b.get("lines", {})) for b in root["books"].values())
    if n_lines == 0:
        raise RuntimeError("账户数据全部缺失：没有缓存，下载也失败（检查网络后重试）")
    jsonio.dump(root, out_dir / ROOT_PATH.name)
    tables = priors.build_priors()
    jsonio.dump(tables, out_dir / priors.PRIORS_PATH.name)
    summary = {"lines": n_lines, "stale": len(root["stale"]), "missing": len(root["missing"]),
               "sources": root["source_summary"], "checks": root["checks"], "priors_reproduced": tables.get("summary")}
    print(f"[agents_accounts] {n_lines} lines; stale {summary['stale']}; missing {summary['missing']}; "
          f"sources {summary['sources']}; priors reproduced {summary['priors_reproduced']}", flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="智能体推演的起点账户（构建步骤 agents_accounts 与研究下载的导入）")
    ap.add_argument("--import", dest="import_dir", help="research dl/ folder to copy into data/raw/agents")
    ap.add_argument("--factors-raw", help="research factors/raw folder (Debt to the Penny, MTS table 4)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--offline", action="store_true", help="build from the cache only (no downloads)")
    args = ap.parse_args(argv)
    if args.import_dir:
        print(json.dumps(import_research(Path(args.import_dir), Path(args.factors_raw) if args.factors_raw else None,
                                         force=args.force), ensure_ascii=False))
        return 0
    from ..env import load_dotenv
    load_dotenv()
    run(download=not args.offline)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
