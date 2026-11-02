"""Portfolio and withdrawal evaluation across scenarios (the API backend).

Reads results/scenario_paths.npz (+ results/scenarios.json for names and
probabilities) and, on request, results/custom/<id>.npz/.json. Loaded arrays
and their yearly growth factors are cached by file mtime, so a request only
pays for the portfolio arithmetic. Response shapes: docs/ARCHITECTURE.md.

Probabilities: built-in scenarios use their `prob` from scenarios.json (else
the npz `probs`, else uniform); an included custom scenario uses its own
`prob` / `prior_prob` (default 0.05). `scenario_probs` replaces any of them
and the whole vector is renormalized to 1, so a probability of 0 drops a
scenario from the weighted mixture (it is still reported on its own).

Worst / best scenarios are ranked in real terms (median real CAGR; for
withdrawals the ruin probability, ties broken by real terminal wealth), so a
high-inflation scenario cannot look "best" on nominal growth alone.

Withdrawal horizons beyond the scenario paths (HORIZON_BOUNDS, 10..40 years;
the paths are config.HORIZON_MONTHS = 180 months long). Past month 180 there is
no scenario theme, so the years after it come from unconditional engine
simulations, drawn the same way for every scenario (build_continuation,
written by the build's portfolio step as CONTINUATION_FILE next to the paths):
  * simulate() with no views from today's anchor (the prior over 6-month block
    starts, config.RESIDUALS), 40 years long; its first 180 months are dropped
    so the continuation starts from where an unconditional path stands after 15
    years (rates, spread, unemployment and real oil back near their anchors).
    One continuation path per built-in scenario path (13 x 600 = 7800, run in
    chunks of CONTINUATION_CHUNK paths with seeds derived from
    CONTINUATION_SEED): path row r of the stacked scenarios goes on with
    continuation path r mod N, so every built-in path gets its own independent
    future (a custom scenario appended after them reuses paths from the start).
    The file stores yearly growth factors (float32), which is all the
    withdrawal loop reads. The first 180 months stay exactly the scenario
    paths, so every statistic up to year 15 is bit-identical, and scenarios
    differ after year 15 only through the wealth and holdings they leave there.
  * Why one path per row: an earlier version appended the same 600 paths to
    every scenario (row k -> path k), so the 13 scenarios shared their futures
    and the Monte Carlo noise of the probability-weighted figures did not
    average out across scenarios. On the 2026-11-01 build, the 30-year ruin of
    60/40 at 4% (pro_rata) over six continuation seeds was 37.1% on average
    with a seed-to-seed SD of 0.72 pp (the seed then in use gave the highest,
    38.5%); with one path per row the mean was 37.0% and the SD 0.32 pp. Safe
    rates move by about 0.01 pp either way.
  * Capital-market assumptions: CONTINUATION_CMA = "valuation_state", i.e.
    valuation_adjusted without its reversion term, the only part of it that is
    a 15-year target (reversion x ln(median P/D / P/D now) / 15 years). What
    stays is the engine's standing view that the sample's re-rating, its sector
    winners and gold's 1970s repricing do not repeat. "historical" (no
    adjustment) would add the +1.71%/yr re-rating back: the P/D tracked at
    kappa 0 rises 1.54%/yr over months 181..480 (x1.47 in 25 years: the
    scenarios' probability-weighted 77 at year 15 would end near 113, above
    the 1973+ maximum of 102); keeping the reversion would pull it on toward
    the 51.6 median (-1.43%/yr). With valuation_state it stays about flat
    (mean -0.16%/yr, median 90.8 -> 88.6). On the 2026-11-01 build the
    30-year ruin of 60/40 at 4% (pro_rata) is 37.0% this way, 33.0% with
    historical and 40.4% with the reversion continued (safe rates 2.76%,
    2.83%, 2.71%).
  * Why that is far above the historical 4% rule (2026-11-01 build, 60/40,
    4%, pro_rata, 30 years): the same withdrawal code on actual history (this
    panel's us_equity and treasury_10y, rolling windows starting 1953..1996)
    fails 6-7% of the time (monthly or December starts; worst starts 1968-11
    and 1965-12, lowest surviving rates 3.75% and 3.87%), and a
    plain 6-month block bootstrap of the actual months fails about 4% (safe
    rate 4.1-4.2%): the Trinity / Bengen range. The engine's view-free paths
    from today (2400 each) fail 15% with the historical drift (today's 1.1%
    equity yield instead of the sample's 2.8%, today's rates; median real US
    equity return 5.1% a year against 7.4% in history), 28% without the
    re-rating tailwind (valuation_state, 3.3%) and 37% with the reversion term
    in the first 15 years (valuation_adjusted, then this continuation), the
    same as with the 13 scenarios' first 15 years (37.0%). The gap is the
    capital-market assumptions, not the continuation or the scenario themes.
  * CPI and income continue: the continuation's CPI is chained onto each
    path's month-180 level (inflation-indexed withdrawals keep t = 0 purchasing
    power) and its income is the engine's (anchor yields on equities, y10 / 12,
    cash interest). The macro levels themselves are the continuation's own
    from month 181, not each scenario's month-180 levels: no theme means no
    memory of one.

Tickers (INSTRUMENTS_DESIGN §6.1, §7.3; tickers.py): `weights` may name the 16
tickers of the instrument layer (QQQ ... BRK.B, case-insensitive) alone or
mixed with asset classes; GOLD (金价) is the gold asset class itself. A request
that holds no ticker runs exactly the asset-class code above (the same arrays
and operations, so every such result is bit-identical to what it was before
tickers existed); one that holds tickers evaluates the columns with a positive
weight: the asset classes' and, row for row, the tickers' paths
(instrument_paths.npz, custom/<id>.instr.npz) and, past year 15, the tickers'
continuation (instrument_continuation.npz, the layer run on the very
continuation paths the asset classes use). The dividend-first rules pay out
each ticker's own distributions (its payout model's cash per start-of-month
value: sticky payouts for the BDCs, MLPs, MO, SCHD and BME, 14.3% of NAV for
QQQI, the coupon yield for TLT, index dividends for QQQ and XLG, none for the
leveraged funds and BRK.B); the asset classes keep the engine's income. The
withdrawal page's alpha_mode (shrunk / half / raw) swaps the estimated
tickers' yearly factors for the build's precomputed variants (they differ in
the alpha scalar only). The ticker files are loaded on the first request that
holds a ticker (about 160 MB), never at start.

Scenario sets (SIM_DESIGN §11.3, §4.6; AGENT_DESIGN §14). `scenario_set` picks the paths a request runs on:
  monthly        the monthly engine's 13 built-ins (scenario_paths.npz above; "builtin" is the same set). Every
                 statistic of a monthly request is computed exactly as before sets existed.
  quarterly      the 13 built-ins' reviewed QTables on the quarterly stack (scenario_paths_q.npz + scenarios_q.json,
                 portfolio/qsets.py): step_months = 3, 60 quarters, and the set's own continuation to year 40 (its
                 cont_* arrays, the same model run on without statements) instead of continuation_paths.npz.
  agent:<id>     a quarterly set the agent tree exports (data/agents/sets/<id>.npz + .json, the same layout).
  None           the default: monthly until the switch gate T-S passes (evaluation/switch.py state()), then quarterly
                 (falling back to monthly, with a note, while the quarterly set is not built).
A set with step_months = s has 12 / s steps a year: volatility is annualized by sqrt(12 / s) (x sqrt(4) for a quarterly
set), Sharpe and Sortino likewise; rebalancing is every step for "monthly" (refused on a quarterly set: there is no
month inside a quarter) and every 3 / s steps for "quarterly" (offered on both sets, so they can be compared on the
same rule), every 12 / s steps for "annual"; drawdowns of a quarterly set are quarter-end figures (季末口径). Yearly
withdrawals are unchanged: the withdrawal loop reads yearly factors, built from the steps of each year. A monthly
custom scenario included in a quarterly set is summed to quarter-ends (total returns exactly; its yearly factors are
its own monthly ones, so its withdrawal figures equal the monthly set's). Tickers need monthly paths: a quarterly set
refuses them.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import zipfile
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .. import config, jsonio
from . import metrics
from . import tickers as tk
from . import withdrawal as wd

RESULTS_DIR = config.RESULTS_DIR          # module level so tests can point it elsewhere
PATHS_FILE = "scenario_paths.npz"
INDEX_FILE = "scenarios.json"
CUSTOM_DIR = "custom"
OUTPUT_FILE = "portfolio_default.json"
ID_RE = re.compile(r"^[a-z0-9_]{1,60}$")
DEFAULT_CUSTOM_PROB = 0.05
MONTHS = metrics.MONTHS_PER_YEAR
RUIN_CURVE_RATES = tuple(round(0.0025 * k, 4) for k in range(41))   # 0% .. 10% per year
HORIZON_BOUNDS = (10, 40)            # withdrawal horizons (whole years) the API accepts
DEFAULT_HORIZON = 30                 # the withdrawal page's default: the horizon the 4% convention is about
CONTINUATION_FILE = "continuation_paths.npz"
CONTINUATION_CMA = "valuation_state"     # valuation_adjusted without its 15-year reversion term (module docstring)
CONTINUATION_SEED = zlib.crc32(b"continuation") % 2**31     # library.seed_for("continuation")
CONTINUATION_CHUNK = 600             # paths per simulate() call of build_continuation (memory: 480 months each)
BUILD_PORTFOLIO = "python -m scenariolab.build --only portfolio"
# scenario sets (module docstring)
QUARTERLY_FILE = "scenario_paths_q.npz"
QUARTERLY_INDEX = "scenarios_q.json"
AGENT_SETS_DIR = config.DATA_DIR / "agents" / "sets"     # module attribute: tests point it elsewhere
BUILD_QSETS = "python -m scenariolab.portfolio.qsets build"
SET_ALIASES = {"monthly": "monthly", "builtin": "monthly", "quarterly": "quarterly", "builtin_q": "quarterly"}
AGENT_SET_RE = re.compile(r"^agent:([A-Za-z0-9_-]{1,80})$")
SET_NAMES = {"monthly": "13 个内置情景 · 月度引擎", "quarterly": "13 个内置情景 · 季度模型"}   # with 13 scenarios

REBALANCE = ("monthly", "quarterly", "annual", "none")       # rebalancing rules a request may name
REBALANCE_MONTHS = {"monthly": 1, "quarterly": 3, "annual": 12}
QUARTER_END_DD = {"us_equity": 0.888, "nasdaq": 0.932, "treasury_10y": 0.918}   # quarter-end / monthly max drawdown
QUARTERLY_DD_NOTE = ("季末口径：季度情景集只有季末的数值，最大回撤按季末计，会低估月内的最深点（1973–2026 年滚动 10 年窗口，"
                     "季末回撤 ÷ 月度最大回撤的中位数：美股 0.888（第 10 百分位 0.776）、纳指 0.932、10 年期美债 0.918）；"
                     "不做月内插值。")


def set_name(set_id: str, n: int | None = None, meta: dict | None = None) -> str:
    """'13 个内置情景 · 月度引擎', '11 个内置情景 · 季度模型' (a set may leave a diverging scenario out), or the agent
    set's own name."""
    if set_id in ("monthly", "quarterly"):
        return SET_NAMES[set_id] if not n else f"{n} 个内置情景 · {'月度引擎' if set_id == 'monthly' else '季度模型'}"
    return str((meta or {}).get("name") or set_id)


DEFAULT_WEIGHTS = {"us_equity": 0.6, "treasury_10y": 0.4}
PRESETS = [
    {"id": "classic_60_40", "name": "经典股债 60/40", "weights": DEFAULT_WEIGHTS,
     "description": "60% 美股大盘 + 40% 10 年期美债。"},
    {"id": "all_equity", "name": "全仓美股", "weights": {"us_equity": 1.0},
     "description": "100% 美股大盘，收益和回撤都最大。"},
    {"id": "all_weather", "name": "简化全天候", "weights": {"us_equity": 0.3, "treasury_10y": 0.55,
                                                       "gold": 0.075, "crude_oil": 0.075},
     "description": "30% 股票、55% 国债、各 7.5% 黄金和原油，近似桥水全天候的配比。"},
    {"id": "income", "name": "收息组合", "weights": {"utilities": 0.2, "real_estate": 0.2, "staples": 0.15,
                                                 "financials": 0.1, "treasury_10y": 0.25, "cash": 0.1},
     "description": "偏重高股息板块、REITs 和债券，现金流多、适合先用股息的提取策略。"},
    {"id": "inflation_hedge", "name": "抗通胀组合", "weights": {"us_equity": 0.25, "energy": 0.2, "gold": 0.2,
                                                           "real_estate": 0.15, "cash": 0.2},
     "description": "能源股、黄金、REITs 和现金，针对高通胀情景。"},
]
# Ticker portfolios (INSTRUMENTS_DESIGN §4.4, §7.2): evaluated by run() into portfolio_default.json's
# instrument_presets when the instrument files are built; PRESETS above stay the asset-class presets.
INSTRUMENT_PRESETS = [
    {"id": "income_tickers", "name": "示例：收息标的",
     "weights": {"MAIN": 0.10, "ARCC": 0.10, "CSWC": 0.05, "QQQI": 0.15, "MPLX": 0.075, "EPD": 0.075, "SCHD": 0.10,
                 "MO": 0.05, "BME": 0.05, "TLT": 0.10, "QQQ": 0.10, "BRK.B": 0.05},
     "description": "只为展示标的层，不是建议：三家 BDC、两家 MLP、QQQI、SCHD、MO、BME 收息，TLT、QQQ、BRK.B 平衡。"},
]


class PathsNotReady(ValueError):
    """The scenario path files are missing or unreadable (e.g. a build is rewriting them)."""


# ---------------------------------------------------------------- loading and caching

@dataclass
class _Block:
    """One npz file, flattened to a scenario-major path axis."""
    ids: list[str]
    n_paths: int                 # paths per scenario
    total: np.ndarray            # (S*N, T, A) float32 monthly total returns
    cpi: np.ndarray              # (S*N, T+1) CPI relative to t=0
    assets: list[str]
    labels: list[str]            # T+1 monthly labels
    probs: np.ndarray | None     # (S,) if the npz stores them
    factors: wd.YearFactors      # (S*N, Y, A)
    _fingerprint: str | None = field(default=None, repr=False)
    step_months: int = 1         # months per step: 1 (the monthly engine), 3 (a quarterly set)
    cont: "_Continuation | None" = None     # a set file's own continuation (its cont_* arrays), if it has one
    cont_issue: str = ""                    # why a set file's continuation could not be read

    @property
    def months(self) -> int:
        """The number of steps T (months for the monthly engine's files, quarters for a quarterly set)."""
        return self.total.shape[1]

    def fingerprint(self) -> str:
        """instruments.build.paths_fingerprint of these paths (the ids and each scenario's first two paths): what
        the ticker files record of the scenario paths they were run on."""
        if self._fingerprint is None:
            S = len(self.ids)
            first = self.total.reshape(S, self.n_paths, *self.total.shape[1:])[:, :2]
            self._fingerprint = tk.paths_fingerprint(self.ids, first)
        return self._fingerprint


CONT_KEYS = ("cont_total_y", "cont_price_y", "cont_income_y", "cont_cpi_y")   # a set file's own continuation


def _load_block(path: Path) -> _Block:
    cont_arrays = None
    try:
        with np.load(path, allow_pickle=False) as z:
            total = np.asarray(z["total"], dtype=np.float32)
            income = np.asarray(z["income"], dtype=np.float32)
            cpi = np.asarray(z["cpi"], dtype=np.float64)
            ids = [str(x) for x in z["ids"]]
            assets = [str(x) for x in z["assets"]]
            labels = [str(x) for x in z["labels"]] if "labels" in z.files else []
            probs = np.asarray(z["probs"], dtype=float) if "probs" in z.files else None
            step = int(z["step_months"]) if "step_months" in z.files else 1
            if all(k in z.files for k in CONT_KEYS):
                cont_arrays = {"total_y": z["cont_total_y"], "price_y": z["cont_price_y"],
                               "income_y": z["cont_income_y"], "cpi_y": z["cont_cpi_y"], "assets": z["assets"],
                               "labels": z["cont_labels"] if "cont_labels" in z.files else None,
                               "start": str(z["cont_start"]) if "cont_start" in z.files else None,
                               "offset": int(z["cont_offset"]) if "cont_offset" in z.files else None,
                               "info": json.loads(str(z["cont_info"])) if "cont_info" in z.files else {}}
    except (OSError, KeyError, ValueError, TypeError, EOFError, zipfile.BadZipFile, zlib.error) as err:
        raise PathsNotReady(f"无法读取 {path.name}（可能正在重新生成，请稍后重试）：{err}") from err
    if step < 1 or MONTHS % step:
        raise PathsNotReady(f"{path.name} 的步长 step_months = {step} 不是 12 的约数")
    S, N, T, A = total.shape
    if income.shape != total.shape or cpi.shape != (S, N, T + 1) or len(ids) != S or len(assets) != A:
        raise PathsNotReady(f"{path.name} 的数组形状不一致")
    total = total.reshape(S * N, T, A)
    cpi = cpi.reshape(S * N, T + 1)
    cpi = cpi / cpi[:, :1]
    if len(labels) != T + 1:
        labels = [str(k) for k in range(T + 1)]
    blk = _Block(ids=ids, n_paths=N, total=total, cpi=cpi, assets=assets, labels=labels,
                 probs=probs if probs is not None and probs.shape == (S,) else None,
                 factors=wd.year_factors(total, income.reshape(S * N, T, A), cpi, months=MONTHS // step),
                 step_months=step)
    if cont_arrays is not None:
        if cont_arrays["labels"] is None or cont_arrays["start"] is None or cont_arrays["offset"] is None:
            blk.cont_issue = f"{path.name} 的续推数组缺少 cont_labels / cont_start / cont_offset"
        else:
            try:
                blk.cont = _continuation_from(cont_arrays)
            except PathsNotReady as err:
                blk.cont_issue = f"{path.name} 的续推数组：{err}"
    return blk


def _aggregate_block(blk: _Block, step: int) -> _Block:
    """A monthly block summed to steps of `step` months (a custom scenario in a quarterly set): total returns compound
    exactly (prod(1 + r) - 1 over the step) and the CPI is taken at the step ends; that is all the portfolio report
    reads per step. The yearly factors (the withdrawal loop's input: total, price and income growth per year) are the
    block's own monthly ones, so its withdrawal figures are exactly the monthly set's."""
    if blk.step_months == step:
        return blk
    if blk.step_months != 1 or step % blk.step_months or blk.months < step:
        raise ValueError(f"无法把步长 {blk.step_months} 个月的路径换成 {step} 个月一步")
    P, T, A = blk.total.shape
    K = T // step
    r = blk.total[:, :K * step].astype(np.float64).reshape(P, K, step, A)
    tot = np.prod(1.0 + r, axis=2) - 1.0
    return _Block(ids=blk.ids, n_paths=blk.n_paths, total=tot.astype(np.float32), cpi=blk.cpi[:, 0:K * step + 1:step],
                  assets=blk.assets, labels=blk.labels[0:K * step + 1:step], probs=blk.probs, factors=blk.factors,
                  step_months=step)


def _load_json(path: Path) -> dict:
    try:
        d = jsonio.load(path)
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


# ---------------------------------------------------------------- unconditional continuation (years 16..40)

def continuation_label(scenario_years: int) -> str:
    return f"第 {scenario_years + 1} 年起没有情景主题：按历史月份无条件续抽"


def continuation_label_q(scenario_years: int) -> str:
    return f"第 {scenario_years + 1} 年起没有情景主题：同一个模型不加陈述续推"


@dataclass
class _Continuation:
    """CONTINUATION_FILE: what follows month `offset` of every scenario path (module docstring)."""
    start: str                   # anchor month of the unconditional run ("YYYY-MM"), = the paths' first label
    offset: int                  # months simulated and dropped first (= the scenario paths' length)
    assets: list[str]
    labels: list[str]            # year-end labels from month `offset` on (index 0 = month `offset`)
    factors: wd.YearFactors      # (N, Y, A); cpi relative to month `offset`
    info: dict = field(default_factory=dict)    # how it was made (build_continuation), shown with the results
    _fingerprint: str | None = field(default=None, repr=False)

    @property
    def n_paths(self) -> int:
        return self.factors.total.shape[0]

    @property
    def years(self) -> int:
        return self.factors.years

    def factors_for(self, assets: list[str]) -> wd.YearFactors:
        return self.factors if assets == self.assets else self.factors.select([self.assets.index(a) for a in assets])

    def fingerprint(self) -> str:
        """sha256 of the yearly total factors as CONTINUATION_FILE stores them: the ticker continuation records the
        one it was simulated alongside (tickers.build_continuation)."""
        if self._fingerprint is None:
            self._fingerprint = tk.factors_fingerprint(self.factors.total)
        return self._fingerprint


YEARLY_KEYS = ("total_y", "price_y", "income_y", "cpi_y")       # CONTINUATION_FILE arrays (write_continuation)


def _continuation_from(arrays: dict) -> _Continuation:
    """A _Continuation from build_continuation's output (monthly total, income, cpi or yearly `factors`;
    monthly labels) or from CONTINUATION_FILE's yearly factors (YEARLY_KEYS, year-end labels)."""
    assets = [str(a) for a in arrays["assets"]]
    labels = [str(x) for x in arrays["labels"]]
    bad = PathsNotReady(f"{CONTINUATION_FILE} 的数组形状不一致：请运行 {BUILD_PORTFOLIO} 重新生成")
    if arrays.get("factors") is not None:
        factors = arrays["factors"]
        if factors.total.shape[2] != len(assets) or len(labels) < factors.years * MONTHS + 1:
            raise bad
        labels = labels[0:factors.years * MONTHS + 1:MONTHS]
    elif "total_y" in arrays:
        total, price, income = (np.asarray(arrays[k], dtype=np.float64) for k in YEARLY_KEYS[:3])
        cpi = np.asarray(arrays["cpi_y"], dtype=np.float64)
        if total.ndim != 3:
            raise bad
        N, Y, A = total.shape
        if (price.shape != total.shape or income.shape != total.shape or cpi.shape != (N, Y + 1)
                or len(assets) != A or len(labels) != Y + 1 or Y < 1 or not np.all(cpi[:, 0] == 1.0)):
            raise bad
        factors = wd.YearFactors(total, price, income, cpi, np.argsort(-total, axis=2, kind="stable"))
    else:
        total = np.asarray(arrays["total"], dtype=np.float32)
        income = np.asarray(arrays["income"], dtype=np.float32)
        cpi = np.asarray(arrays["cpi"], dtype=np.float64)
        if total.ndim != 3:
            raise bad
        N, T, A = total.shape
        if (income.shape != total.shape or cpi.shape != (N, T + 1) or len(assets) != A or len(labels) != T + 1
                or T < MONTHS):
            raise bad
        factors = wd.year_factors(total, income, cpi)
        labels = labels[0:factors.years * MONTHS + 1:MONTHS]
    return _Continuation(start=str(arrays["start"]), offset=int(arrays["offset"]), assets=assets, labels=labels,
                         factors=factors, info=dict(arrays.get("info") or {}))


def _load_continuation(path: Path) -> _Continuation:
    try:
        with np.load(path, allow_pickle=False) as z:
            if "total_y" not in z.files:          # the earlier monthly format (600 paths shared by every scenario)
                raise PathsNotReady(f"{path.name} 是旧格式（所有情景共用同一组续抽路径）：请运行 {BUILD_PORTFOLIO} 重新生成")
            arrays = {k: z[k] for k in (*YEARLY_KEYS, "assets", "labels")}
            arrays.update(start=str(z["start"]), offset=int(z["offset"]), info=json.loads(str(z["info"])))
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as err:
        if isinstance(err, PathsNotReady):
            raise
        raise PathsNotReady(f"无法读取 {path.name}（可能正在重新生成，请稍后重试）：{err}") from err
    return _continuation_from(arrays)


def write_continuation(path: str | Path, total=None, income=None, cpi=None, assets=(), labels=(), start: str = "",
                       offset: int = 0, info: dict | None = None, factors: wd.YearFactors | None = None) -> Path:
    """Write a continuation file atomically (a temporary file, then a rename: the server may be reading it).

    Takes build_continuation's output (monthly total / income / cpi, or with yearly=True its `factors`) and
    stores what the withdrawal loop reads: the yearly growth factors (wd.year_factors; total, price-only and
    income, float32, (N, Y, A)), the year-end CPI relative to month `offset` (float64, (N, Y+1)) and the
    year-end labels. That is 1/12 of the monthly size, so every scenario path can have its own continuation
    path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = factors if factors is not None else wd.year_factors(
        np.asarray(total, dtype=np.float32), np.asarray(income, dtype=np.float32), np.asarray(cpi, dtype=np.float64))
    labels = [str(x) for x in labels][0:f.years * MONTHS + 1:MONTHS]
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(tmp, total_y=f.total.astype(np.float32), price_y=f.price.astype(np.float32),
                        income_y=f.income.astype(np.float32), cpi_y=f.cpi, assets=np.array([str(a) for a in assets]),
                        labels=np.array(labels), start=np.array(str(start)),
                        offset=np.array(int(offset)), info=np.array(jsonio.dumps(info or {})))
    tmp.replace(path)
    return path


def continuation_chunk_seed(seed: int, chunk: int) -> int:
    """Seed of chunk `chunk` of build_continuation: the base seed for the first, a hash of (seed, chunk) after."""
    return int(seed) if chunk == 0 else zlib.crc32(f"continuation:{int(seed)}:{int(chunk)}".encode()) % 2**31


def build_continuation(ctx, offset: int = config.HORIZON_MONTHS, n_paths: int = 600,
                       seed: int = CONTINUATION_SEED, years: int = HORIZON_BOUNDS[1],
                       chunk: int = CONTINUATION_CHUNK, yearly: bool = False, hook=None) -> dict:
    """The unconditional engine paths that follow month `offset` (write_continuation's arguments).

    hook(sim, first_row, chunk_index), when given, is called with every run's SimOutput (all `years` x 12 months)
    and its results are returned as the list out["hook"] (tickers.build_continuation runs the instrument layer on
    each run this way); nothing else changes.

    simulate() of `years` x 12 months with no views: the prior over 6-month block starts (exactly what a
    view-free scenario phase draws), today's anchor, the default reversion, config.RESIDUALS and the
    CONTINUATION_CMA drift; `n_paths` paths in runs of at most `chunk` (run c is seeded with
    continuation_chunk_seed(seed, c), so the first run is simulate(seed=seed) itself). The first `offset`
    months are dropped. The P/D is tracked at kappa 0 (returns unchanged) only to report what the drift
    assumption implies. Returns the monthly total / income / cpi, or with `yearly` only their yearly
    factors (`factors`, wd.year_factors per run), which keeps the build's memory at one run's months
    (13 x 600 paths x 300 months would otherwise hold about 250 MB, more with --paths).
    """
    from ..engine.pooling import BLOCK, pool_phase
    from ..engine.simulate import capital_market_assumptions, fit_valuation, simulate
    from ..engine.summary import month_labels

    T = int(years) * MONTHS
    if not 0 < offset < T or offset % MONTHS:
        raise ValueError(f"offset must be a whole number of years below {T} months")
    n_paths, chunk = int(n_paths), max(1, int(chunk))
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    db, anchor = ctx.db, ctx.anchor
    pool = pool_phase(db.frame, [], db.prior(), T, BLOCK, rare=db.rare)
    cma = capital_market_assumptions(db, anchor, method=CONTINUATION_CMA)
    try:
        vm = fit_valuation(db, anchor, kappa=0.0, income="constant")
    except (KeyError, ValueError, np.linalg.LinAlgError):      # a panel without the valuation columns
        vm = None
    levels = ("y10", "ff", "spread", "unrate", "cpi_yoy")          # reported at month `offset` (info.at_start)
    shown = ("us_equity", "treasury_10y", "cash")                    # real CAGRs reported in info
    parts, factors, seeds, start, assets, hooked = [], [], [], None, None, []
    for c, a in enumerate(range(0, n_paths, chunk)):
        seeds.append(continuation_chunk_seed(seed, c))
        sim = simulate(db, anchor, [pool.blocks], [T], n_paths=min(chunk, n_paths - a), seed=seeds[-1], block=BLOCK,
                       cma=cma, valuation=vm, **config.RESIDUALS)
        start, assets = sim.start, list(sim.assets)
        if hook is not None:
            hooked.append(hook(sim, a, c))
        m = sim.macro
        tot, inc, cpi_c = sim.total[:, offset:], sim.income[:, offset:], m["cpi"][:, offset:].astype(np.float64)
        infl = cpi_c[:, -1] / cpi_c[:, 0]
        part = {"infl": infl, **{k: m[k][:, offset].astype(np.float64) for k in levels if k in m},
                **{f"g_{x}": np.prod(1.0 + tot[:, :, assets.index(x)].astype(np.float64), axis=1) / infl
                   for x in shown if x in assets},
                **({"pd0": m["pd"][:, offset].astype(np.float64), "pd1": m["pd"][:, -1].astype(np.float64)}
                   if "pd" in m else {})}
        if yearly:
            factors.append(wd.year_factors(tot, inc, cpi_c))
        else:
            part.update(total=np.ascontiguousarray(tot), income=np.ascontiguousarray(inc), cpi=cpi_c)
        parts.append(part)
        del sim, m, tot, inc
    cat = {k: (np.concatenate([p[k] for p in parts]) if len(parts) > 1 else parts[0][k]) for k in parts[0]}
    del parts
    labels = month_labels(start, T)
    yrs = (T - offset) / MONTHS

    def real_cagr(asset: str) -> float:
        return float(np.median(cat[f"g_{asset}"] ** (1.0 / yrs) - 1.0))

    pd_info = None
    if "pd0" in cat:
        pd_info = {"p50_at_start": float(np.median(cat["pd0"])), "p50_end": float(np.median(cat["pd1"])),
                   "dlog_per_year": float(np.log(cat["pd1"] / cat["pd0"]).mean() / yrs)}
    scen_cma = getattr(ctx, "cma", None) or {}
    info = {
        "method": "unconditional", "seed": int(seed), "n_paths": n_paths, "chunk_paths": min(chunk, n_paths),
        "chunk_seeds": seeds, "offset_months": int(offset),
        "months": T - offset, "block_months": BLOCK, "residuals": dict(config.RESIDUALS),
        "anchor": labels[0], "cma": CONTINUATION_CMA,
        "cma_adjustment": {a: float(v) for a, v in cma["adjustment"].items()},
        "market_adjustment": cma.get("market_adjustment"), "rerating_tailwind": cma.get("rerating_tailwind"),
        "scenario_cma": scen_cma.get("method"), "scenario_market_adjustment": scen_cma.get("market_adjustment"),
        "reversion_term_dropped": scen_cma.get("reversion_term"),
        "at_start": {k: float(np.median(cat[k])) for k in levels if k in cat},
        "inflation_p50": float(np.median(cat["infl"] ** (1.0 / yrs) - 1.0)),
        "real_cagr_p50": {a: real_cagr(a) for a in shown if a in assets},
        "pd": pd_info,
    }
    out = {"assets": assets, "labels": labels[offset:], "start": labels[0], "offset": int(offset), "info": info}
    if hook is not None:
        out["hook"] = hooked
    if yearly:
        return {"factors": wd.YearFactors.concat(factors), **out}
    return {"total": cat["total"], "income": cat["income"], "cpi": cat["cpi"], **out}


def _shift_label(label: str, months: int) -> str:
    """"2026-09" + 12 -> "2027-09" (a non-date label: the month number)."""
    try:
        y, m = int(label[:4]), int(label[5:7])
    except (TypeError, ValueError):
        return str(months)
    k = y * 12 + (m - 1) + months
    return f"{k // 12:04d}-{k % 12 + 1:02d}"


_LOCK = threading.RLock()
_FILES: OrderedDict = OrderedDict()      # str(path) -> (stamp, object)
_SETS: OrderedDict = OrderedDict()       # composition key -> ScenarioSet
_CRIT: OrderedDict = OrderedDict()       # (set key, weights, strategy, indexed, years) -> critical rates
_MAX_FILES, _MAX_SETS, _MAX_CRIT = 24, 4, 96    # a set with customs copies ~50 MB of year factors


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _remember(cache: OrderedDict, key, value, limit: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _cached(path: Path, loader):
    """Load `path` once per (mtime, size); reload when the file changes."""
    stamp, key = _stamp(path), str(path)
    with _LOCK:
        hit = _FILES.get(key)
        if hit is not None and hit[0] == stamp:
            _FILES.move_to_end(key)
            return hit[1]
        return _remember(_FILES, key, (stamp, loader(path)), _MAX_FILES)[1]


def clear_cache() -> None:
    with _LOCK:
        _FILES.clear()
        _SETS.clear()
        _CRIT.clear()
        _BOOKS.clear()
    tk.clear_cache()


@dataclass
class ScenarioSet:
    """The scenarios one request evaluates, with all paths on one axis."""
    ids: list[str]
    names: list[str]
    sources: list[str]                   # "builtin" | "custom"
    default_probs: np.ndarray            # (S,), sums to 1
    ranges: list[tuple[int, int]]        # path rows [a, b) of each scenario
    assets: list[str]
    labels: list[str]                    # T+1 monthly labels, index 0 = start
    months: int                          # T
    parts: list[tuple[_Block, list[int] | None]]   # blocks in row order + asset reindex
    factors: wd.YearFactors              # (P, Y, A) over all rows
    key: tuple
    notes: list[str] = field(default_factory=list)   # composition caveats shown with every result
    continuation: _Continuation | None = None        # what follows the paths' last month (CONTINUATION_FILE)
    continuation_issue: str = ""                     # why there is none (shown when a horizon needs it)
    files: list = field(default_factory=list)        # (kind, npz path, custom id) of each part, in row order
    root: Path | None = None                         # the results directory of the built-in files
    _extended: wd.YearFactors | None = field(default=None, repr=False)   # factors + continuation, built once
    _tickers: tuple | None = field(default=None, repr=False)   # (tickers.ticker_key, tickers.TickerSet), on first use
    step_months: int = 1                             # months per step (`months` above counts steps: T)
    set_id: str = "monthly"                          # monthly | quarterly | agent:<id> (module docstring)
    set_meta: dict = field(default_factory=dict)     # the set file's own description (scenarios_q.json "set", …)

    @property
    def spy(self) -> int:
        """Steps per year: 12 for the monthly engine's paths, 4 for a quarterly set."""
        return MONTHS // self.step_months

    @property
    def years(self) -> int:
        """Years with a scenario theme: the length of the scenario paths."""
        return self.months // self.spy

    @property
    def max_years(self) -> int:
        return self.years + (self.continuation.years if self.continuation is not None else 0)

    @property
    def groups(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.ids)), [b - a for a, b in self.ranges])

    def _label(self, step: int) -> str:
        return self.labels[step] if step < len(self.labels) else _shift_label(self.labels[0], step * self.step_months)

    def year_labels(self, years: int) -> list[str]:
        return [self._label(self.spy * k)[:4] for k in range(years + 1)]

    def year_dates(self, years: int) -> list[str]:
        return [self._label(self.spy * k)[:7] for k in range(years + 1)]

    def continuation_rows(self) -> np.ndarray:
        """The continuation path each row goes on with: row r of the stacked scenarios -> path r mod N.

        With one continuation path per built-in path (the build's N), every built-in row has its own; rows
        past N (custom scenarios appended after the built-ins) start over from path 0.
        """
        n = self.continuation.n_paths
        return np.arange(self.factors.total.shape[0]) % n

    def continuation_shared(self) -> list[str]:
        """Names of the scenarios with rows that reuse a continuation path another row also goes on with."""
        n = self.continuation.n_paths
        return [name for name, (a, b) in zip(self.names, self.ranges) if b > n]

    def factors_for(self, years: int) -> wd.YearFactors:
        """Year factors for `years` year ends: the scenario paths, then the continuation if needed.

        Up to self.years the result is self.factors itself (sliced), so nothing changes there.
        PathsNotReady when the horizon needs a continuation that is missing, stale or too short.
        """
        if years <= self.years:
            return self.factors.select(None, years)
        self.check_continuation(years)
        with _LOCK:
            if self._extended is None:
                self._extended = self.factors.chain(self.continuation.factors_for(self.assets), self.continuation_rows())
        return self._extended.select(None, years)

    def check_continuation(self, years: int) -> None:
        """PathsNotReady when a horizon of `years` needs a continuation that is missing, stale or too short."""
        if self.continuation is None:
            raise PathsNotReady(f"期限 {years} 年超过情景路径的 {self.years} 年，需要无条件续抽路径："
                                f"{self.continuation_issue or f'{CONTINUATION_FILE} 不可用'}")
        if years > self.max_years:
            raise PathsNotReady(f"无条件续抽路径只够到第 {self.max_years} 年：请运行 {BUILD_PORTFOLIO} 重新生成")

    def with_continuation(self, cont: _Continuation | None, issue: str = "") -> "ScenarioSet":
        """A copy with `cont` attached when it fits these paths (else none, and the reason)."""
        if cont is not None:
            issue = _continuation_issue(self, cont)
            if issue:
                cont = None
        return replace(self, continuation=cont, continuation_issue=issue, _extended=None, _tickers=None,
                       notes=list(self.notes), key=self.key + (("continuation", id(cont)),))

    def tickers(self) -> "tk.TickerSet":
        """The tickers' columns for these rows (tickers.compose): loaded on the first request that holds a ticker
        and again whenever one of their files changes. PathsNotReady when the built-in ticker files are missing or
        belong to other scenario paths; ValueError when an included custom scenario was saved without them, or when
        the set is not monthly (the ticker layer runs on the monthly engine's paths)."""
        if self.step_months != 1:
            raise ValueError(TICKERS_MONTHLY_ONLY)
        key = tk.ticker_key(self)
        with _LOCK:
            if self._tickers is None or self._tickers[0] != key:
                self._tickers = (key, tk.compose(self))
            return self._tickers[1]


TICKERS_MONTHLY_ONLY = ("季度情景集暂不支持标的：标的层（QQQ、TQQQ、BDC 等）按月度引擎的情景路径计算；"
                        "请换回月度情景集，或只用 13 类资产")


def _continuation_issue(data: ScenarioSet, cont: _Continuation) -> str:
    """Why `cont` cannot follow `data`'s paths ("" when it can)."""
    monthly = data.step_months == 1
    name = f"{CONTINUATION_FILE} " if monthly else "情景集文件里的续推数组"
    again = (f"请运行 {BUILD_PORTFOLIO} 重新生成" if monthly else "请重新导出这个情景集" if data.set_id.startswith("agent:")
             else f"请运行 {BUILD_QSETS} 重新生成")
    unit = "个月" if monthly else f"步（每步 {data.step_months} 个月）"
    if sorted(cont.assets) != sorted(data.assets):
        return f"{name}的资产列表与情景路径不一致：{again}"
    if cont.offset != data.months or data.months % data.spy:
        return f"{name}接在第 {cont.offset} {unit}之后，而情景路径有 {data.months} {unit}：{again}"
    if cont.start != data.labels[0]:
        return f"{name}是按 {cont.start} 起的数据生成的，情景路径从 {data.labels[0]} 开始：{again}"
    return ""


def _first_number(*values) -> float | None:
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f >= 0:
            return f
    return None


def _custom_ids(include_custom) -> list[str]:
    if include_custom is None:
        return []
    items = [include_custom] if isinstance(include_custom, str) else list(include_custom)
    out = []
    for cid in items:
        cid = str(cid)
        if not ID_RE.match(cid):
            raise ValueError(f"自定义情景 id 不合法：{cid}")
        if cid not in out:
            out.append(cid)
    return out


def _compose(parts: list[tuple[str, Path, Path, str | None]], key: tuple, cont_path: Path | None = None) -> ScenarioSet:
    data = _compose_paths(parts, key)
    if cont_path is None:
        return data
    if not cont_path.exists():
        out = data.with_continuation(None, f"{CONTINUATION_FILE} 还没有生成：请运行 {BUILD_PORTFOLIO}")
    else:
        try:
            out = data.with_continuation(_cached(cont_path, _load_continuation))
        except PathsNotReady as err:          # half-written or unreadable: horizons past the paths wait for it
            out = data.with_continuation(None, str(err))
    out.key = key                             # the continuation file's stamp is part of the key already
    return out


def _compose_paths(parts: list[tuple[str, Path, Path, str | None]], key: tuple, set_id: str = "monthly") -> ScenarioSet:
    loaded = [(kind, _cached(npz, _load_block), _cached(js, _load_json) if js.exists() else {}, cid)
              for kind, npz, js, cid in parts]
    main = loaded[0][1]
    step = main.step_months
    spy = MONTHS // step
    if step != 1:          # monthly custom scenarios in a quarterly set: summed to its steps (_aggregate_block)
        loaded = [(kind, blk if kind != "custom" else _aggregate_block(blk, step), info, cid)
                  for kind, blk, info, cid in loaded]
    T = min(b.months for _, b, _, _ in loaded)
    if T < spy:
        raise PathsNotReady("情景路径不足一年")
    meta = {str(s.get("id")): s for s in loaded[0][2].get("scenarios", []) if isinstance(s, dict)}
    ids, names, sources, probs, ranges, blocks, factors, notes = [], [], [], [], [], [], [], []
    row = 0
    for kind, blk, info, cid in loaded:
        if sorted(blk.assets) != sorted(main.assets):
            raise ValueError(f"自定义情景 {cid} 的资产列表与内置情景不一致")
        if kind == "custom" and blk.labels[0] != main.labels[0]:
            notes.append(f"自定义情景 {cid} 的路径从 {blk.labels[0]} 开始，内置情景从 {main.labels[0]} 开始"
                         "（可能是数据更新前保存的）；混合时按同一起点对齐，建议重新运行该情景。")
        cols = None if blk.assets == main.assets else [blk.assets.index(a) for a in main.assets]
        for k, sid in enumerate(blk.ids):
            if kind in ("builtin", "agent"):
                m = meta.get(sid, {})
                p = _first_number(m.get("prob"), None if blk.probs is None else blk.probs[k])
                name = str(m.get("name") or sid)
            else:
                sid = cid if len(blk.ids) == 1 else f"{cid}_{sid}"
                p = _first_number(info.get("prob"), info.get("prior_prob"), DEFAULT_CUSTOM_PROB)
                name = str(info.get("name") or sid)
                if sid in ids:
                    sid = f"custom:{sid}"
                notes.append(f"自定义情景「{name}」默认以先验概率 {p:.1%} 与内置情景一起重新归一化后计入加权汇总"
                             "（可用 scenario_probs 修改）。")
                if step != 1:
                    notes.append(f"自定义情景「{name}」来自月度引擎：按季末汇总后与这个季度情景集一起混合"
                                 "（总回报按月复利精确汇总；提取用它自己的年度数据）。")
            ids.append(sid)
            names.append(name)
            sources.append(kind)
            probs.append(np.nan if p is None else p)
            ranges.append((row, row + blk.n_paths))
            row += blk.n_paths
        blocks.append((blk, cols))
        factors.append(blk.factors.select(cols, T // spy))
    p = np.asarray(probs, dtype=float)
    p[np.isnan(p)] = 1.0 / len(p)
    if p.sum() <= 0:
        p[:] = 1.0
    return ScenarioSet(ids=ids, names=names, sources=sources, default_probs=p / p.sum(), ranges=ranges,
                       assets=list(main.assets), labels=main.labels[:T + 1], months=T, parts=blocks,
                       factors=wd.YearFactors.concat(factors), key=key, notes=notes,
                       files=[(kind, Path(npz), cid) for kind, npz, _, cid in parts], root=Path(parts[0][1]).parent,
                       step_months=step, set_id=set_id,
                       set_meta=dict(loaded[0][2].get("set") or {}) if isinstance(loaded[0][2], dict) else {})


def _compose_set(parts: list[tuple[str, Path, Path, str | None]], key: tuple, set_id: str) -> ScenarioSet:
    """A set file (quarterly or agent) with its own continuation (its cont_* arrays) after its last step."""
    data = _compose_paths(parts, key, set_id)
    main = data.parts[0][0]
    if main.cont is not None:
        out = data.with_continuation(main.cont)
    else:
        out = data.with_continuation(None, main.cont_issue or "这个情景集没有续推数组：期限只能到情景路径的长度")
    out.key = key
    return out


def normalize_set(scenario_set) -> str:
    """'monthly' | 'quarterly' | 'agent:<id>' for an explicit scenario_set (ValueError with a Chinese message)."""
    if not isinstance(scenario_set, str):
        raise ValueError("scenario_set 必须是文字：monthly、quarterly 或 agent:<编号>")
    s = scenario_set.strip()
    if s in SET_ALIASES:
        return SET_ALIASES[s]
    if AGENT_SET_RE.match(s):
        return s
    raise ValueError(f"未知情景集：{s[:60]}（可选 monthly / quarterly / agent:<编号>，不写则按切换关的默认）")


def set_files(set_id: str, root: Path) -> tuple[Path, Path]:
    """(npz, index json) of a set."""
    if set_id == "monthly":
        return root / PATHS_FILE, root / INDEX_FILE
    if set_id == "quarterly":
        return root / QUARTERLY_FILE, root / QUARTERLY_INDEX
    m = AGENT_SET_RE.match(set_id)
    if not m:
        raise ValueError(f"未知情景集：{set_id}")
    base = Path(AGENT_SETS_DIR)
    return base / f"{m.group(1)}.npz", base / f"{m.group(1)}.json"


def load_scenarios(include_custom=(), results_dir: str | Path | None = None, scenario_set: str = "monthly") -> ScenarioSet:
    """Built-in scenarios (of one scenario set: the monthly engine's by default) plus the requested custom ones
    (cached by file mtimes)."""
    root = Path(RESULTS_DIR if results_dir is None else results_dir)
    sid = normalize_set(scenario_set)
    if sid != "monthly":
        return _load_set(sid, include_custom, root)
    main = root / PATHS_FILE
    if not main.exists():
        raise PathsNotReady(f"{PATHS_FILE} 还没有生成：请先运行 python -m scenariolab.build")
    parts = [("builtin", main, root / INDEX_FILE, None)]
    for cid in _custom_ids(include_custom):
        npz = root / CUSTOM_DIR / f"{cid}.npz"
        if not npz.exists():
            raise ValueError(f"自定义情景 {cid} 不存在或没有保存路径数据")
        parts.append(("custom", npz, root / CUSTOM_DIR / f"{cid}.json", cid))
    cont = root / CONTINUATION_FILE
    key = tuple((str(npz), _stamp(npz), _stamp(js)) for _, npz, js, _ in parts) + ((str(cont), _stamp(cont)),)
    with _LOCK:
        hit = _SETS.get(key)
        if hit is not None:
            _SETS.move_to_end(key)
            return hit
        return _remember(_SETS, key, _compose(parts, key, cont), _MAX_SETS)


def _load_set(sid: str, include_custom, root: Path) -> ScenarioSet:
    main, index = set_files(sid, root)
    if not main.exists():
        if sid == "quarterly":
            raise PathsNotReady(f"季度情景集（{QUARTERLY_FILE}）还没有生成：请运行 {BUILD_QSETS}")
        raise ValueError(f"情景集 {sid} 不存在")
    parts = [("builtin" if sid == "quarterly" else "agent", main, index, None)]
    for cid in _custom_ids(include_custom):
        npz = root / CUSTOM_DIR / f"{cid}.npz"
        if not npz.exists():
            raise ValueError(f"自定义情景 {cid} 不存在或没有保存路径数据")
        parts.append(("custom", npz, root / CUSTOM_DIR / f"{cid}.json", cid))
    key = ("set", sid) + tuple((str(npz), _stamp(npz), _stamp(js)) for _, npz, js, _ in parts)
    with _LOCK:
        hit = _SETS.get(key)
        if hit is not None:
            _SETS.move_to_end(key)
            return hit
        return _remember(_SETS, key, _compose_set(parts, key, sid), _MAX_SETS)


def _switch_state(root: Path) -> dict:
    try:
        from ..evaluation import switch
        return switch.state(root)
    except Exception as err:  # noqa: BLE001 - an unreadable switch switches nothing
        return {"status": "invalid", "quarterly": False, "passed": False, "label": "试运行",
                "text": f"切换关的状态读不出来（{type(err).__name__}）：默认引擎是月度引擎。", "reasons": [],
                "default_text": "默认引擎：月度引擎"}


def _switch_brief(st: dict) -> dict:
    return {**{k: st.get(k) for k in ("status", "passed", "quarterly", "text", "default_text", "label", "evaluated_at",
                                      "reasons")},
            "certified_model": (st.get("model") or {}).get("semantic") if isinstance(st.get("model"), dict) else None}


STALE_QSET_NOTE = ("切换关 T-S 已通过，但磁盘上的季度情景集是用另一个版本的季度模型生成的（不是切换关认证的版本）："
                   f"请重新生成季度情景集（{BUILD_QSETS}）；在那之前默认用月度引擎的情景集，季度情景集只作试运行。")


def _set_model(meta: dict) -> str | None:
    """The quarterly stack's semantic version a set was built on (its index's set.model, or the head of set.stack)."""
    m = meta.get("model")
    if isinstance(m, str) and m.startswith("qmodel-"):
        return m.split("/", 1)[0]
    st = meta.get("stack")
    return st.split("/", 1)[0] if isinstance(st, str) and st.startswith("qmodel-") else None


def qset_certified(meta: dict, sw: dict) -> bool:
    """A quarterly set may serve as the switched default only when it was built on the model T-S certified."""
    want = sw.get("certified_model")
    if want is None and isinstance(sw.get("model"), dict):
        want = sw["model"].get("semantic")
    got = _set_model(meta or {})
    return bool(want) and got == want


def _qset_meta(root: Path) -> dict:
    idx = root / QUARTERLY_INDEX
    info = _load_json(idx) if idx.exists() else {}
    return dict((info or {}).get("set") or {}) if isinstance(info, dict) else {}


def resolve_set(scenario_set=None, results_dir: str | Path | None = None) -> dict:
    """Which set a request runs on: {id, requested, default (bool: chosen by the switch), note, switch}.
    None / "" / "default": the switch's default (switch.state(): monthly until T-S passes), falling back to the
    monthly set with a note while the quarterly set is not built."""
    root = Path(RESULTS_DIR if results_dir is None else results_dir)
    st = _switch_state(root)
    want, note = ("quarterly" if st.get("quarterly") else "monthly"), None
    if want == "quarterly" and not (root / QUARTERLY_FILE).exists():
        want = "monthly"
        note = f"切换关 T-S 已通过，但季度情景集还没有生成（{BUILD_QSETS}）：这次先用月度引擎的情景集。"
    elif want == "quarterly" and not qset_certified(_qset_meta(root), st):
        want, note = "monthly", STALE_QSET_NOTE
    if scenario_set is None or (isinstance(scenario_set, str) and scenario_set.strip() in ("", "default")):
        return {"id": want, "requested": None, "default": True, "note": note, "switch": _switch_brief(st),
                "effective_default": want}
    return {"id": normalize_set(scenario_set), "requested": str(scenario_set), "default": False, "note": None,
            "switch": _switch_brief(st), "effective_default": want}


def set_block(data: ScenarioSet, sinfo: dict | None) -> dict:
    """The response's description of the set it ran on."""
    sinfo = sinfo or {"id": data.set_id, "default": False, "note": None, "switch": None}
    sw = sinfo.get("switch") or {}
    switched = bool(sw.get("quarterly"))
    if switched and data.set_id == "quarterly" and not qset_certified(data.set_meta, sw):
        switched = False                         # built on another model version than T-S certified: a trial set
    engine = "monthly" if data.set_id == "monthly" else str(data.set_meta.get("engine") or "quarterly")
    stub = bool(data.set_meta.get("stub"))
    trial = engine != "monthly" and not switched
    name = set_name(data.set_id, sum(1 for x in data.sources if x == "builtin") or None, data.set_meta)
    return {"id": data.set_id, "name": name, "engine": engine, "engine_zh": "月度引擎" if engine == "monthly" else "季度模型",
            "step_months": data.step_months, "steps_per_year": data.spy, "trial": trial,
            "label": ("桩模型（不是资产模型）" if stub else "试运行" if trial else "默认" if engine != "monthly" and switched
                      else ""),
            "is_default": (data.set_id == sinfo["effective_default"] if sinfo.get("effective_default")
                           else (data.set_id == "quarterly") == switched) if data.set_id in ("monthly", "quarterly")
            else False,
            "chosen_by_default": bool(sinfo.get("default")), "note": sinfo.get("note"),
            "switch": {k: sw.get(k) for k in ("status", "passed", "text", "default_text")} if sw else None,
            "meta": {k: data.set_meta.get(k) for k in ("version", "created", "origin", "n_paths", "quality", "probabilities",
                                                        "text") if k in data.set_meta}}


def warm(results_dir: str | Path | None = None) -> bool:
    """Pre-load the built-in paths and their continuation (e.g. at server start). False if not built."""
    try:
        data = load_scenarios((), results_dir)
    except PathsNotReady:
        return False
    if data.continuation is not None:
        data.factors_for(min(DEFAULT_HORIZON, data.max_years))
    return True


# ---------------------------------------------------------------- request validation

def _number(value, what: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{what}必须是数字")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}必须是数字") from None
    if not math.isfinite(f):
        raise ValueError(f"{what}必须是有限数字")
    return f


def parse_horizon(value) -> int:
    """A withdrawal horizon: a whole number of years in HORIZON_BOUNDS (30 or 30.0; not 30.5, 0 or 41)."""
    lo, hi = HORIZON_BOUNDS
    x = _number(value, "期限（horizon_years）")
    if x != math.floor(x) or not lo <= x <= hi:
        got = f"{x:g}" if x == math.floor(x) else repr(x)       # 30.0000001 is not shown as "30"
        raise ValueError(f"期限（horizon_years）必须是 {lo} 到 {hi} 之间的整数年（收到 {got}）")
    return int(x)


def _normalized(w: np.ndarray) -> np.ndarray:
    if w.max() <= 0:
        raise ValueError("权重不能全为 0")
    with np.errstate(over="ignore"):
        if not np.isfinite(w.sum()):     # e.g. 1e308 + 1e308: rescale instead of zeroing every weight
            w = w / w.max()
    return w / w.sum()


def parse_book_weights(weights, assets: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    """{asset or ticker id: weight} -> ((A,) asset weights, {ticker: weight}), raw (not yet normalized).

    Asset ids are exact; tickers are case-insensitive ("brk.b" is BRK.B) and GOLD (金价) is the gold asset class
    (tickers.resolve). An id that is neither: ValueError 未知资产或标的."""
    if not isinstance(weights, dict) or not weights:
        raise ValueError("weights 必须是 {资产或标的 id: 权重} 形式的非空对象")
    w = np.zeros(len(assets))
    held: dict[str, float] = {}
    for aid, val in weights.items():
        kind, key = tk.resolve(aid, assets)
        if kind is None:
            raise ValueError(error_unknown(aid))
        what = "资产" if kind == "asset" else "标的"
        x = _number(val, f"{what} {aid} 的权重")
        if x < 0:
            raise ValueError(f"{what} {aid} 的权重不能为负")
        if kind == "asset":
            w[assets.index(key)] += x
        else:
            held[key] = held.get(key, 0.0) + x
    return w, held


def error_unknown(aid) -> str:
    return f"未知资产或标的：{aid}"


def parse_weights(weights, assets: list[str]) -> np.ndarray:
    """{asset_id: weight} -> (A,) non-negative weights over `assets`, summing to 1 (asset classes only; a ticker
    with a positive weight is refused here: _book takes both)."""
    w, held = parse_book_weights(weights, assets)
    if any(x > 0 for x in held.values()):
        raise ValueError(f"这里只接受资产类别：{'、'.join(t for t, x in held.items() if x > 0)} 是标的")
    return _normalized(w)


@dataclass
class _Book:
    """The columns one request holds. Without a ticker: every asset class in data.assets order with the weights
    parse_weights gives, i.e. exactly the asset-class evaluation. With tickers: the asset classes and the tickers
    with a positive weight (assets first, in data.assets order, then tickers in instruments.TICKERS order)."""
    columns: list[str]
    w: np.ndarray                        # (C,) summing to 1
    asset_cols: list[int] | None         # data.assets indices of the asset columns (None: all of them)
    tickers: list[str]
    alpha_mode: str = "shrunk"

    @property
    def key(self) -> tuple:
        return () if not self.tickers else (tuple(self.columns), self.alpha_mode)

    def weights_out(self) -> dict:
        return {a: float(x) for a, x in zip(self.columns, self.w) if x > 0}


def _book(data: ScenarioSet, weights, alpha_mode=None) -> _Book:
    mode = tk.parse_alpha_mode(alpha_mode)
    wa, held = parse_book_weights(weights, data.assets)
    held = {t: x for t, x in held.items() if x > 0}
    if not held:
        return _Book(columns=list(data.assets), w=_normalized(wa), asset_cols=None, tickers=[], alpha_mode=mode)
    if data.step_months != 1:
        raise ValueError(TICKERS_MONTHLY_ONLY)
    cols = [i for i, x in enumerate(wa) if x > 0]
    tickers = [t for t in tk.TICKERS if t in held]
    w = _normalized(np.concatenate([wa[cols], [held[t] for t in tickers]]))
    return _Book(columns=[data.assets[i] for i in cols] + tickers, w=w, asset_cols=cols, tickers=tickers,
                 alpha_mode=mode)


def _book_monthly(data: ScenarioSet, book: _Book, k: int, asset_total: np.ndarray) -> np.ndarray:
    """Part k's monthly total returns in the book's columns (the asset totals themselves without a ticker)."""
    if not book.tickers:
        return asset_total
    T = asset_total.shape[1]
    return np.concatenate([asset_total[:, :, book.asset_cols], data.tickers().monthly(k, book.tickers, T)], axis=2)


def _joined(af: wd.YearFactors, cols: list[int], tf: tuple, cpi: np.ndarray) -> wd.YearFactors:
    cat = [np.concatenate([getattr(af, k)[:, :, cols], t], axis=2) for k, t in zip(("total", "price", "income"), tf)]
    return wd.YearFactors(cat[0], cat[1], cat[2], cpi, np.argsort(-cat[0], axis=2, kind="stable"))


_BOOKS: OrderedDict = OrderedDict()      # (set key, ticker key, book key, years) -> YearFactors
_MAX_BOOKS = 2


def _book_factors(data: ScenarioSet, book: _Book, years: int) -> wd.YearFactors:
    """data.factors_for(years) in the book's columns: the scenario years, then the continuation (the asset classes'
    and the tickers', row for row) past them."""
    if not book.tickers:
        return data.factors_for(years)
    ts = data.tickers()
    key = (data.key, ts.key, book.key, years)
    with _LOCK:
        hit = _BOOKS.get(key)
        if hit is not None:
            _BOOKS.move_to_end(key)
            return hit
    head = _joined(data.factors, book.asset_cols, ts.year_columns(book.tickers, book.alpha_mode), data.factors.cpi)
    if years <= data.years:
        f = head.select(None, years)
    else:
        data.check_continuation(years)
        if ts.continuation is None:
            raise PathsNotReady(f"期限 {years} 年超过情景路径的 {data.years} 年，标的需要续抽路径：{ts.continuation_issue}")
        cf = data.continuation.factors_for(data.assets)
        tail = _joined(cf, book.asset_cols, ts.cont_columns(book.tickers, book.alpha_mode), cf.cpi)
        f = head.chain(tail, data.continuation_rows()).select(None, years)
    with _LOCK:
        return _remember(_BOOKS, key, f, _MAX_BOOKS)


def _flag(value, what: str, default: bool) -> bool:
    """Strict boolean: true/false (also 1/0 and their strings); None -> default."""
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0"):
        return value.strip().lower() in ("true", "1")
    raise ValueError(f"{what}必须是 true 或 false")


def scenario_probabilities(data: ScenarioSet, scenario_probs=None) -> np.ndarray:
    """Default probabilities with optional {id: prob} replacements, renormalized to 1."""
    p = data.default_probs.copy()
    if scenario_probs:
        if not isinstance(scenario_probs, dict):
            raise ValueError("scenario_probs 必须是 {情景id: 概率} 形式的对象")
        for sid, val in scenario_probs.items():
            if sid not in data.ids:
                raise ValueError(f"未知情景：{sid}（自定义情景需要同时放进 include_custom）")
            x = _number(val, f"情景 {sid} 的概率")
            if x < 0:
                raise ValueError(f"情景 {sid} 的概率不能为负")
            p[data.ids.index(sid)] = x
    return metrics.normalize_probs(p)


def _strategies(strategy) -> list[str]:
    items = list(wd.STRATEGIES) if strategy in (None, "all") else (
        [strategy] if isinstance(strategy, str) else list(strategy))
    for s in items:
        if s not in wd.STRATEGIES:
            raise ValueError(f"未知提取策略：{s}（可选 all / {' / '.join(wd.STRATEGIES)}）")
    return items


def _extremes(entries: list[dict], rank) -> tuple[dict, dict]:
    """(lowest, highest) entry by the tuple rank(entry) among scenarios with positive probability."""
    def finite(e: dict) -> bool:
        try:
            return all(math.isfinite(x) for x in rank(e))
        except TypeError:                # a None statistic
            return False

    live = [e for e in entries if e["prob"] > 0 and finite(e)] or [e for e in entries if finite(e)]
    if not live:
        return entries[0], entries[0]
    return min(live, key=rank), max(live, key=rank)


def _real_growth(e: dict) -> tuple:
    """Portfolio ranking: real median CAGR (nominal growth flatters high-inflation scenarios)."""
    return e["real_cagr_p50"], e["cagr_p50"]


def _ruin_rank(e: dict) -> tuple:
    """Withdrawal ranking: ruin probability, ties broken by lower real terminal wealth."""
    return e["ruin_prob"], -e["terminal_real_p50"]


# ---------------------------------------------------------------- portfolio

def rebalance_rule(rebalance, step_months: int = 1) -> tuple[str, str, int]:
    """(the rule's name, metrics.portfolio_wealth mode, period in steps) of a request's rebalancing on a set with
    step_months: monthly every step (monthly sets only), quarterly every 3 / s steps, annual every 12 / s, none.
    None: monthly on a monthly set, quarterly on a quarterly one."""
    if rebalance is None:
        rebalance = "monthly" if step_months == 1 else "quarterly"
    if rebalance not in REBALANCE:
        raise ValueError(f"rebalance 必须是 {' / '.join(REBALANCE)} 之一")
    if rebalance == "none":
        return rebalance, "none", 0
    months = REBALANCE_MONTHS[rebalance]
    if months < step_months or months % step_months:
        raise ValueError(f"这个情景集每步 {step_months} 个月，没有月内路径：不能每月再平衡"
                         f"（可选 {' / '.join(r for r in REBALANCE if r != 'monthly')}）")
    k = months // step_months
    return rebalance, ("monthly" if k == 1 else "annual"), k


def _quarterly_notes(data: ScenarioSet) -> list[str]:
    """The notes a set with steps longer than a month adds to every result."""
    if data.step_months == 1:
        return []
    s = data.step_months
    out = [f"这个情景集每步 {s} 个月（每年 {data.spy} 步）：波动率按 {s} 个月收益的标准差 × √{data.spy} 年化，"
           f"夏普和索提诺同样按每年 {data.spy} 步；“每季再平衡”就是每一步调回目标权重，没有月内路径，所以不能每月再平衡。",
           QUARTERLY_DD_NOTE]
    text = data.set_meta.get("text")
    if isinstance(text, str) and text:
        out.append(text)
    for line in data.set_meta.get("notes") or []:
        if isinstance(line, str) and line:
            out.append(line)
    return out


def portfolio_report(data: ScenarioSet, weights, scenario_probs=None, rebalance: str | None = None,
                     set_info: dict | None = None) -> dict:
    """evaluate_portfolio() on already loaded scenarios. Fans are wealth multiples of 1 at t=0."""
    book = _book(data, weights)
    w = book.w
    probs = scenario_probabilities(data, scenario_probs)
    rebalance, mode, period = rebalance_rule(rebalance, data.step_months)
    T, Y, spy = data.months, data.years, data.spy
    ts = data.tickers() if book.tickers else None
    cash = data.assets.index("cash") if "cash" in data.assets else None
    parts, yearly, yearly_real = [], [], []
    for k, (blk, cols) in enumerate(data.parts):
        atot = blk.total[:, :T] if cols is None else blk.total[:, :T][:, :, cols]
        total = _book_monthly(data, book, k, atot)
        wealth = (metrics.portfolio_wealth(total, w, mode) if mode != "annual"
                  else metrics.portfolio_wealth(total, w, mode, period=period))
        deflator = blk.cpi[:, :T + 1]
        rf = 0.0 if cash is None else atot[:, :, cash].astype(np.float64)
        parts.append(metrics.path_metrics(wealth, deflator, rf, periods=spy))
        yearly.append(wealth[:, 0:Y * spy + 1:spy])
        yearly_real.append(yearly[-1] / deflator[:, 0:Y * spy + 1:spy])
    m = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    yearly, yearly_real = np.concatenate(yearly), np.concatenate(yearly_real)

    def summary(rows: slice = slice(None), path_weights=None) -> dict:
        return {**metrics.summarize({k: v[rows] for k, v in m.items()}, path_weights),
                "fan": metrics.fan(yearly[rows], path_weights),
                "fan_real": metrics.fan(yearly_real[rows], path_weights)}

    per = [{"id": sid, "name": name, "source": src, "prob": float(probs[k]), "n_paths": b - a,
            **summary(slice(a, b))}
           for k, (sid, name, src, (a, b)) in enumerate(zip(data.ids, data.names, data.sources, data.ranges))]
    groups = data.groups
    worst, best = _extremes(per, _real_growth)
    out = {
        "years": data.year_labels(Y), "dates": data.year_dates(Y), "horizon_months": T * data.step_months,
        "weights": book.weights_out(),
        "rebalance": rebalance,
        "per_scenario": per,
        "aggregate": {
            "weighted": summary(slice(None), metrics.mixture_weights(groups, probs)),
            "worst": worst, "best": best,
            "equal_weight": summary(slice(None), metrics.mixture_weights(groups, np.ones(len(data.ids)))),
        },
        "notes": [
            "财富以期初 1 元计；real 为按各路径自身 CPI 折算成期初购买力。",
            "CAGR、回撤等先在每条路径上计算，再取分位数；加权汇总把各情景的路径按情景概率混合。",
            "夏普 = 相对现金资产的年化超额收益 / 年化波动；索提诺的分母只计跑输现金的"
            + ("月份。" if data.step_months == 1 else "季度。"),
            "max_dd_p95 是回撤的坏尾部：只有 5% 的路径回撤比它更深；cvar5_terminal 是最差 5% 路径的期末财富倍数均值。",
            "最差/最好情景按扣除通胀后的实际 CAGR 中位数排序（名义收益会高估高通胀情景）。",
            *_quarterly_notes(data),
            *data.notes,
        ],
        "step_months": data.step_months,
        "scenario_set": set_block(data, set_info),
    }
    if set_info and set_info.get("note"):
        out["notes"].insert(0, set_info["note"])
    if ts is not None:
        out["instruments"] = tk.describe(ts, book.tickers, book.alpha_mode, metrics.mixture_weights(groups, probs))
        out["notes"] += tk.notes(ts, book.tickers, book.alpha_mode, withdrawal=False, scenario_years=Y)
    return out


def evaluate_portfolio(weights: dict, scenario_probs: dict | None = None,
                       include_custom: list[str] = (), rebalance: str | None = None,
                       scenario_set: str | None = None) -> dict:
    """Portfolio statistics per scenario and aggregated (docs/ARCHITECTURE.md). scenario_set: monthly | quarterly |
    agent:<id>, or None for the switch's default (module docstring); rebalance: monthly | quarterly | annual | none
    (None: monthly on a monthly set, quarterly on a quarterly one)."""
    sinfo = resolve_set(scenario_set)
    return portfolio_report(load_scenarios(include_custom, scenario_set=sinfo["id"]), weights, scenario_probs,
                            rebalance, set_info=sinfo)


# ---------------------------------------------------------------- withdrawal

def _critical(data: ScenarioSet, book: _Book, strategy: str, indexed: bool, years: int) -> np.ndarray:
    ts_key = data.tickers().key if book.tickers else ()
    key = (data.key, ts_key, book.key, tuple(np.round(book.w, 12)), strategy, indexed, years)
    with _LOCK:
        hit = _CRIT.get(key)
        if hit is not None:
            _CRIT.move_to_end(key)
            return hit
    crit = wd.critical_rates(_book_factors(data, book, years), book.w, strategy, indexed, years)
    with _LOCK:
        return _remember(_CRIT, key, crit, _MAX_CRIT)


def _rates(data: ScenarioSet, crit: np.ndarray, pw: np.ndarray) -> tuple[dict, dict]:
    """(safe rate, ruin curve) weighted and per scenario, from per-path critical rates."""
    by = list(zip(data.ids, data.ranges))
    return ({"weighted": wd.safe_rate(crit, pw), "per_scenario": {sid: wd.safe_rate(crit[a:b]) for sid, (a, b) in by}},
            {"weighted": wd.ruin_curve(crit, RUIN_CURVE_RATES, pw),
             "per_scenario": {sid: wd.ruin_curve(crit[a:b], RUIN_CURVE_RATES) for sid, (a, b) in by}})


_THEMED_KEYS = ("ruin_prob", "terminal_real_p5", "terminal_real_p50", "terminal_real_p95", "terminal_nominal_p50",
                "depletion_year_median")
DIVIDEND_NOTE = ("股票和 REITs 的分红按今天的股息率乘以当时的市值计入现金池，熊市里随股价同比例下降（真实分红更有粘性）；"
                 "改用平滑的粘性分红只是把同样的总收益在现金和股价之间重新划分，在 5 个预设组合上让 30 年破产概率最多高出约 0.8 个百分点"
                 "（不是降低）、安全提取率变化不超过 0.02 个百分点，所以沿用引擎口径。")


def _signed(x) -> str:
    return "—" if x is None else f"{x * 100:+.2f}%"


DIVIDEND_NOTE_Q = ("季度情景集的分红：每个派息资产按它季初的市价股息比计（P/D 随价格与股息增长逐季滚动），10 年期美债为票息、"
                   "现金为全部回报，按季计入、年末一起进现金池；每季的总回报 = 价格变化 + 当季收入。")


def _continuation_notes_q(data: ScenarioSet, horizon: int) -> list[str]:
    """The continuation of a set file (its own cont_* arrays): what follows its last step."""
    c, S = data.continuation, data.years
    i = c.info or {}
    how = i.get("text") or ("同一个模型在情景的最后一季之后不加任何陈述接着往下推（每条路径从它自己的季末状态出发），"
                            "向长期锚回归")
    shared = data.continuation_shared()
    return [f"第 {S + 1} 年起没有情景主题：{how}。前 {S} 年是各情景自己的路径，第 {S + 1}–{horizon} 年接的是这个续推。",
            (f"续推路径共 {c.n_paths} 条，少于全部 {data.factors.total.shape[0]} 条情景路径："
             f"「{'」「'.join(shared)}」的部分路径与其他路径按序号循环共用续推路径。" if shared
             else f"每条情景路径各接一条续推路径（共 {c.n_paths} 条）。"),
            "CPI、分红和利息在续推部分照常计算，随通胀上调的提取额按整条路径的 CPI 换算。"]


def _continuation_notes(data: ScenarioSet, horizon: int) -> list[str]:
    if data.step_months != 1:
        return _continuation_notes_q(data, horizon)
    c, S = data.continuation, data.years
    i = c.info or {}
    shared = data.continuation_shared()
    notes = [
        f"{continuation_label(S)}。前 {S} 年是各情景自己的路径；第 {S + 1}–{horizon} 年接的是引擎的无条件模拟"
        f"（不加任何视图，同样从今天的状态出发、按 {i.get('block_months', 6)} 个月一段抽历史月份，残差的抽法与情景相同），"
        f"去掉这些模拟的前 {S} 年，把其后的年份按路径序号接在情景路径后面。",
        (f"续抽路径共 {c.n_paths} 条，少于全部 {data.factors.total.shape[0]} 条情景路径："
         f"「{'」「'.join(shared)}」的部分路径与其他路径按序号循环共用续抽路径；" if shared
         else f"每条情景路径各接一条独立的续抽路径（共 {c.n_paths} 条，抽法相同）；")
        + f"第 {S} 年之后各情景的差别只来自前 {S} 年留下的财富和持仓。",
    ]
    adj, scen, tail = i.get("market_adjustment"), i.get("scenario_market_adjustment"), i.get("rerating_tailwind")
    if adj is not None and scen is not None and tail is not None:
        notes.append(f"续抽部分的长期回报假设去掉了只为前 {S} 年设定的估值回归项：美股价格漂移每年调整 {_signed(adj)}"
                     f"（前 {S} 年为 {_signed(scen)}），市价股息比（P/D）既不再向历史中位数回归，"
                     f"也不加回 1973 年以来每年约 {tail:.2%} 的估值抬升。")
    at, infl = i.get("at_start") or {}, i.get("inflation_p50")
    level = (f"续抽从无条件路径自己的利率和通胀开始（第 {S} 年末中位数：10 年期收益率 {at['y10']:.1f}%、"
             f"联邦基金利率 {at['ff']:.1f}%；续抽期通胀中位数约每年 {infl:.1%}），"
             if "y10" in at and "ff" in at and infl is not None else "续抽从无条件路径自己的利率和通胀开始，")
    notes.append("CPI、分红和利息在续抽部分照常计算，随通胀上调的提取额按整条路径的 CPI 换算；"
                 f"{level}不承接各情景第 {S} 年末的水平。")
    return notes


COVERAGE_HALF = 0.5          # the cash-flow KPI: some year's pot below half the withdrawal (INSTRUMENTS_DESIGN §7.3)


def _cash_flow(res: wd.WithdrawalPaths, f: wd.YearFactors, initial: float, rate: float, indexed: bool,
               pw: np.ndarray, data: ScenarioSet) -> dict | None:
    """One strategy's pot (the cash the holdings paid) against each year's withdrawal: the weighted fan of
    pot / withdrawal over the paths alive at the start of the year, and the probability that some year's pot is
    below COVERAGE_HALF of the withdrawal (weighted and per scenario). None at a withdrawal rate of 0."""
    if rate <= 0:
        return None
    flows = wd.cash_flows(res, f, rate * initial, indexed)
    cover = np.where(flows["alive"], flows["pot"] / flows["need"], np.nan)
    short = (flows["alive"] & (flows["pot"] < COVERAGE_HALF * flows["need"])).any(axis=1)
    return {"coverage_fan": metrics.fan(cover, pw, decimals=4),
            "prob_pot_below_half": metrics.weighted_mean(short, pw),
            "prob_pot_below_half_per_scenario": {sid: metrics.weighted_mean(short[a:b])
                                                 for sid, (a, b) in zip(data.ids, data.ranges)}}


def _cash_sources(f: wd.YearFactors, book: _Book, initial: float, rate: float, indexed: bool, pw: np.ndarray) -> dict:
    """What pays the withdrawal (strategy-independent): the weighted median of each holding's year-1 cash over the
    year-1 withdrawal (and of their sum), and each holding's distribution-cut pressure over the first 10 years
    (wd.cut_flags: one unit held, some year at least 10% / 20% below year 1)."""
    out = {"pot_share_y1": None, "cut_pressure": {}, "cut_years": min(10, f.years)}
    held = [j for j, x in enumerate(book.w) if x > 0]
    if rate > 0:
        need1 = rate * initial * (f.cpi[:, 1] if indexed else 1.0)
        shares = {book.columns[j]: initial * book.w[j] * f.income[:, 0, j] / need1 for j in held}
        out["pot_share_y1"] = {"total": metrics.weighted_percentile(sum(shares.values()), 50, pw),
                               "columns": {c: metrics.weighted_percentile(s, 50, pw) for c, s in shares.items()}}
    for j in held:
        c10, c20 = wd.cut_flags(f, j, 10, 0.1), wd.cut_flags(f, j, 10, 0.2)
        if np.isfinite(c20).any():
            out["cut_pressure"][book.columns[j]] = {"cut10_prob": metrics.weighted_mean(c10, pw),
                                                    "cut20_prob": metrics.weighted_mean(c20, pw)}
    return out


def _withdrawal_set_notes(data: ScenarioSet) -> list[str]:
    """What a set with steps longer than a month adds to a withdrawal result (the rules themselves are yearly)."""
    if data.step_months == 1:
        return []
    out = [f"这个情景集每步 {data.step_months} 个月：提取仍是每年年末一次，用每年 {data.spy} 步合成的年度数据计算，"
           "规则与月度情景集完全相同。"]
    text = data.set_meta.get("text")
    if isinstance(text, str) and text:
        out.append(text)
    for line in data.set_meta.get("notes") or []:
        if isinstance(line, str) and line:
            out.append(line)
    return out


def withdrawal_report(data: ScenarioSet, weights, initial: float = 1e6, rate: float = 0.04,
                      strategy: str = "all", inflation_indexed: bool = True,
                      scenario_probs=None, horizon_years: int = DEFAULT_HORIZON, alpha_mode: str = "shrunk",
                      set_info: dict | None = None) -> dict:
    """evaluate_withdrawal() on already loaded scenarios. Wealth is in units of `initial`.

    Every figure is at `horizon_years`. Past the scenario paths (data.years) the years come from the
    continuation (module docstring), and `scenario_horizon` gives ruin, terminal wealth, safe rates and
    ruin curves at the end of the scenario paths: the same withdrawal runs cut there, so they equal a
    run of exactly that length. Every input is checked before any path is touched.
    """
    book = _book(data, weights, alpha_mode)
    w = book.w
    probs = scenario_probabilities(data, scenario_probs)
    strategies = _strategies(strategy)
    initial = _number(initial, "初始资产")
    if initial <= 0:
        raise ValueError("初始资产必须大于 0")
    rate = _number(rate, "提取率")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("提取率必须在 0 到 1 之间（0.04 = 每年 4%）")
    Y = parse_horizon(horizon_years)
    indexed = _flag(inflation_indexed, "inflation_indexed ", True)
    S = data.years
    f = _book_factors(data, book, Y)
    longer = Y > S
    pw = metrics.mixture_weights(data.groups, probs)
    decimals = int(np.clip(6 - np.floor(np.log10(initial)), 0, 6))     # ~6 significant digits
    rows = list(enumerate(zip(data.ids, data.names, data.sources, data.ranges)))
    out, safe, curve = {}, {}, {"rates": list(RUIN_CURVE_RATES)}
    flows = {**_cash_sources(f, book, initial, rate, indexed, pw), "strategies": {}}
    part = ({"years": S, "date": data.year_dates(S)[-1], "strategies": {}, "safe_rate": {},
             "ruin_curve": {"rates": list(RUIN_CURVE_RATES)}} if longer else None)
    for s in strategies:
        res = wd.simulate(f, w, initial, rate, s, indexed, Y, record_holdings=True)
        flows["strategies"][s] = _cash_flow(res, f, initial, rate, indexed, pw, data)
        res.holdings = None
        per = [{"id": sid, "name": name, "source": src, "prob": float(probs[k]), "n_paths": b - a,
                **wd.summarize(res.rows(a, b), f.cpi[a:b], decimals=decimals)}
               for k, (sid, name, src, (a, b)) in rows]
        mix = wd.summarize(res, f.cpi, pw, decimals=decimals)
        best, worst = _extremes(per, _ruin_rank)
        out[s] = {
            **wd.STRATEGIES[s],
            "per_scenario": per,
            "aggregate": {
                "ruin_prob_weighted": mix["ruin_prob"],
                "ruin_prob_worst": worst["ruin_prob"], "worst_id": worst["id"],
                "ruin_prob_best": best["ruin_prob"], "best_id": best["id"],
                "survival_weighted": mix["survival"],
                "terminal_real_p50_weighted": mix["terminal_real_p50"],
                "terminal_real_p5_weighted": mix["terminal_real_p5"],
                "terminal_real_p95_weighted": mix["terminal_real_p95"],
                "terminal_nominal_p50_weighted": mix["terminal_nominal_p50"],
                "depletion_year_median_weighted": mix["depletion_year_median"],
                "wealth_fan_real_weighted": mix["wealth_fan_real"],
            },
        }
        safe[s], curve[s] = _rates(data, _critical(data, book, s, indexed, Y), pw)
        if part is not None:
            head = res.head(S)
            hper = [{"id": sid, "prob": float(probs[k]), **wd.summarize(head.rows(a, b), f.cpi[a:b], with_fan=False)}
                    for k, (sid, _, _, (a, b)) in rows]
            hmix = wd.summarize(head, f.cpi, pw, with_fan=False)
            hbest, hworst = _extremes(hper, _ruin_rank)
            part["strategies"][s] = {
                "aggregate": {"ruin_prob_weighted": hmix["ruin_prob"],
                              "ruin_prob_worst": hworst["ruin_prob"], "worst_id": hworst["id"],
                              "ruin_prob_best": hbest["ruin_prob"], "best_id": hbest["id"],
                              **{f"{k}_weighted": hmix[k] for k in _THEMED_KEYS if k != "ruin_prob"}},
                "per_scenario": {e["id"]: {k: e[k] for k in _THEMED_KEYS} for e in hper},
            }
            part["safe_rate"][s], part["ruin_curve"][s] = _rates(data, _critical(data, book, s, indexed, S), pw)
    ts = data.tickers() if book.tickers else None
    notes = [
        "每年年末提取一次，金额 = 提取率 × 初始资产" + (
            "（按期初购买力计），按各路径自身的 CPI 换算成当年名义金额，第 1 年末就已含第 1 年的通胀。"
            if indexed else "（名义金额固定）。"),
        "除提取外不做再平衡；股息/利息策略的现金池在年内不计息。",
        "破产 = 持仓加现金池不够支付当年提取额；之后财富记为 0。terminal_real 以期初购买力计。",
        f"安全提取率 = 破产概率不超过 {wd.SAFE_RUIN_PROB:.0%} 的最高提取率（在 0–{wd.RATE_BOUNDS[1]:.0%} 内搜索，"
        f"达到 {wd.RATE_BOUNDS[1]:.0%} 表示上限）。",
        *(_continuation_notes(data, Y) if longer else []),
        DIVIDEND_NOTE if data.step_months == 1 else DIVIDEND_NOTE_Q,
        *_withdrawal_set_notes(data),
        *data.notes,
        *([] if ts is None else tk.notes(ts, book.tickers, book.alpha_mode, withdrawal=True, scenario_years=S,
                                         horizon=Y)),
    ]
    if set_info and set_info.get("note"):
        notes.insert(0, set_info["note"])
    flows["note"] = CASH_FLOW_NOTE
    cont = None
    if longer:
        c = data.continuation
        shared = data.continuation_shared()
        cont = {**(c.info or {}), "label": continuation_label(S) if data.step_months == 1 else continuation_label_q(S),
                "first_year": S + 1, "years": Y - S,
                "start_date": data.year_dates(S)[-1], "n_paths": c.n_paths, "rows": int(f.total.shape[0]),
                "shared": bool(shared), "shared_scenarios": shared}
        if ts is not None:
            cont["instruments"] = {k: (ts.continuation.info or {}).get(k) for k in (
                "rows", "years", "offset_months", "anchor", "seed", "params_version")}
    report = {
        "years": data.year_labels(Y), "dates": data.year_dates(Y), "horizon_years": Y,
        "scenario_years": S, "horizon_bounds": list(HORIZON_BOUNDS),
        "initial": initial, "rate": rate, "annual_withdrawal": rate * initial, "inflation_indexed": indexed,
        "weights": book.weights_out(),
        "strategies": out, "safe_rate": safe, "ruin_curve": curve,
        "safe_ruin_prob": wd.SAFE_RUIN_PROB, "scenario_horizon": part, "continuation": cont, "notes": notes,
        "cash_flow": flows, "step_months": data.step_months, "scenario_set": set_block(data, set_info),
    }
    if ts is not None:
        report["instruments"] = tk.describe(ts, book.tickers, book.alpha_mode, pw)
    return report


CASH_FLOW_NOTE = ("现金流（cash_flow）：pot_share_y1 = 第一年各持仓的现金（股息、利息、分配）占第一年提取额的比例（中位数）；"
                  "coverage_fan = 每年现金池 / 当年提取额的分布（只计当年年初还没破产的路径；按比例提取时现金留在原资产里再投资，"
                  "这里仍按它产生的现金计）；prob_pot_below_half = 某年现金池不足提取额一半的概率；"
                  "cut_pressure = 持有一份、不再投资时，前 10 年里某年现金比第一年低 10% / 20% 以上的概率。")


def evaluate_withdrawal(weights: dict, initial: float = 1e6, rate: float = 0.04,
                        strategy: str = "all", inflation_indexed: bool = True,
                        scenario_probs: dict | None = None, include_custom: list[str] = (),
                        horizon_years: int = DEFAULT_HORIZON, alpha_mode: str = "shrunk",
                        scenario_set: str | None = None) -> dict:
    """Withdrawal strategies, ruin and safe rates per scenario and aggregated (docs/ARCHITECTURE.md).

    horizon_years: a whole number of years in HORIZON_BOUNDS (10..40, default 30). Years past the
    scenario paths (15) come from the unconditional continuation; scenario_horizon repeats the
    figures at year 15. ValueError (HTTP 400) for an invalid horizon, PathsNotReady (503) when a horizon
    needs the continuation and it is missing or stale. weights may hold tickers (module docstring);
    alpha_mode (shrunk / half / raw) picks the estimated tickers' alpha variant. scenario_set: monthly |
    quarterly | agent:<id>, or None for the switch's default (module docstring).
    """
    sinfo = resolve_set(scenario_set)
    return withdrawal_report(load_scenarios(include_custom, scenario_set=sinfo["id"]), weights, initial, rate, strategy,
                             inflation_indexed, scenario_probs, horizon_years, alpha_mode, set_info=sinfo)


# ---------------------------------------------------------------- the sets a page can pick

def _set_entry(set_id: str, root: Path, st: dict) -> dict:
    npz, index = set_files(set_id, root)
    info = _load_json(index) if index.exists() else {}
    meta = dict(info.get("set") or {})
    engine = "monthly" if set_id == "monthly" else str(meta.get("engine") or "quarterly")
    switched = bool(st.get("quarterly"))
    stale = switched and set_id == "quarterly" and npz.exists() and not qset_certified(meta, st)
    if stale:
        switched = False                         # the monthly set stays the default until the set is rebuilt
    n_index = len([x for x in info.get("scenarios", []) if isinstance(x, dict)]) if isinstance(info, dict) else 0
    out = {"id": set_id, "name": set_name(set_id, n_index or None, meta), "engine": engine,
           "step_months": 1 if set_id == "monthly" else int(meta.get("step_months") or 3),
           "available": npz.exists(), "issue": None,
           "trial": engine != "monthly" and not switched,
           "is_default": (set_id == "quarterly") == switched if set_id in ("monthly", "quarterly") else False,
           "stub": bool(meta.get("stub")), "n_scenarios": None, "n_paths": None, "years": None,
           "created": (meta.get("created") or _modified(npz)) if npz.exists() else None,
           "text": meta.get("text"), "quality": meta.get("quality"), "probabilities": meta.get("probabilities")}
    if stale:
        out["issue"] = STALE_QSET_NOTE
    if not npz.exists():
        out["issue"] = (f"还没有生成：请运行 {BUILD_QSETS}" if set_id == "quarterly"
                        else "还没有生成：请运行 python -m scenariolab.build" if set_id == "monthly" else "文件不在")
        return out
    sc = [s for s in info.get("scenarios", []) if isinstance(s, dict)]
    out["n_scenarios"] = len(sc) or None
    out["n_paths"] = meta.get("n_paths") or info.get("n_paths")
    hm = info.get("horizon_months") or meta.get("horizon_months")
    out["years"] = int(hm) // MONTHS if isinstance(hm, (int, float)) and hm else None
    out["scenarios"] = [{"id": s.get("id"), "name": s.get("name"), "prob": s.get("prob"),
                         "prior_prob": s.get("prior_prob"), "excluded": s.get("excluded"),
                         "headline": s.get("headline") if set_id != "monthly" else None} for s in sc]
    return out


def sets_report(results_dir: str | Path | None = None) -> dict:
    """The scenario sets a page can pick (GET /api/portfolio/sets): {default, sets [...], switch}."""
    root = Path(RESULTS_DIR if results_dir is None else results_dir)
    st = _switch_state(root)
    sets = [_set_entry("monthly", root, st), _set_entry("quarterly", root, st)]
    base = Path(AGENT_SETS_DIR)
    if base.is_dir():
        for p in sorted(base.glob("*.npz")):
            sid = f"agent:{p.stem}"
            if AGENT_SET_RE.match(sid):
                sets.append(_set_entry(sid, root, st))
    sinfo = resolve_set(None, root)
    for e in sets:
        if e["id"] in ("monthly", "quarterly"):
            e["is_default"] = e["id"] == sinfo["id"]
    return {"default": sinfo["id"], "default_note": sinfo.get("note"), "sets": sets, "switch": _switch_brief(st),
            "rebalance": {"monthly": list(REBALANCE), "quarterly": [r for r in REBALANCE if r != "monthly"]},
            "quarter_end_drawdown": QUARTER_END_DD, "quarter_end_note": QUARTERLY_DD_NOTE}


# ---------------------------------------------------------------- build step

def _brief(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if not k.startswith("fan")}


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.ROOT))
    except ValueError:                   # outside the project (tests, scratch builds): no local path
        return path.name


def _modified(path: Path) -> str | None:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
    except OSError:
        return None


def run(ctx=None, results_dir: str | Path | None = None, out_dir: str | Path | None = None,
        write: bool = True) -> dict:
    """Evaluate the default 60/40 portfolio and the presets on the built scenarios.

    With a context (the build's), the unconditional continuation of the scenario paths is simulated
    first (build_continuation: one 40-year path per built-in scenario path, 13 runs of 600, about 7 s)
    and written next to the paths as
    CONTINUATION_FILE when `write` (else it is only used here), so it is computed once per build and
    read by the server like the paths. Without one (tests), an existing CONTINUATION_FILE in
    `results_dir` is used. Writes <out_dir or results_dir>/portfolio_default.json when `write`;
    `source` records which files it was computed from. The default withdrawal is at DEFAULT_HORIZON
    (what the withdrawal page shows first); a preset's ruin_prob_weighted / safe_rate_weighted stay at
    the scenario length (withdrawal_horizon_years, 15: what the preset tooltips of both pages showed
    so far), and withdrawal_by_horizon has them at that length and at DEFAULT_HORIZON.
    """
    root = Path(RESULTS_DIR if results_dir is None else results_dir)
    data = load_scenarios((), root)
    paths, cont_path = root / PATHS_FILE, root / CONTINUATION_FILE
    if ctx is not None:                      # one continuation path per built-in scenario path
        arrays = build_continuation(ctx, offset=data.months, n_paths=data.factors.total.shape[0], yearly=True)
        if write:
            write_continuation(cont_path, **arrays)
            data = load_scenarios((), root)
        else:
            data = data.with_continuation(_continuation_from(arrays))
    source = {"paths_file": _shown(paths), "scenario_ids": data.ids,
              "official_paths": paths.resolve() == (config.RESULTS_DIR / PATHS_FILE).resolve(),
              "paths_modified": _modified(paths),
              "continuation_file": _shown(cont_path) if data.continuation is not None and cont_path.exists() else None,
              "continuation_modified": _modified(cont_path) if data.continuation is not None else None,
              "computed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    base = int(np.clip(data.years, *HORIZON_BOUNDS))
    horizons = sorted({base, DEFAULT_HORIZON})
    presets = [_preset(data, p, base, horizons) for p in PRESETS]
    cont = data.continuation
    out = {"source": source, "default_weights": DEFAULT_WEIGHTS,
           "portfolio": portfolio_report(data, DEFAULT_WEIGHTS),
           "withdrawal": withdrawal_report(data, DEFAULT_WEIGHTS, horizon_years=DEFAULT_HORIZON),
           "continuation": None if cont is None else {
               "file": source["continuation_file"], "label": continuation_label(data.years),
               "scenario_years": data.years, "max_years": data.max_years, **(cont.info or {})},
           "presets": presets}
    try:                                     # the ticker presets, when the instruments step has run
        data.tickers()
        out["instrument_presets"] = [_preset(data, p, base, horizons, tickers=True) for p in INSTRUMENT_PRESETS]
        out["instrument_presets_issue"] = None
    except (PathsNotReady, ValueError) as err:
        out["instrument_presets"], out["instrument_presets_issue"] = None, str(err)
    if write:
        jsonio.dump(out, Path(out_dir or root) / OUTPUT_FILE)
    return out


def _preset(data: ScenarioSet, p: dict, base: int, horizons: list[int], tickers: bool = False) -> dict:
    """A preset's weighted portfolio statistics and its ruin / safe rates at each horizon (pro-rata etc.)."""
    pr = portfolio_report(data, p["weights"])
    by, cash = {}, {}
    for H in horizons:
        wr = withdrawal_report(data, p["weights"], horizon_years=H)
        by[str(H)] = {"ruin_prob_weighted": {s: v["aggregate"]["ruin_prob_weighted"] for s, v in wr["strategies"].items()},
                      "safe_rate_weighted": {s: v["weighted"] for s, v in wr["safe_rate"].items()}}
        if tickers:
            share = wr["cash_flow"]["pot_share_y1"]
            cash[str(H)] = {"cash_y1_over_withdrawal_p50": None if share is None else share["total"],
                            "prob_pot_below_half": {s: (v or {}).get("prob_pot_below_half")
                                                    for s, v in wr["cash_flow"]["strategies"].items()}}
    out = {**p, "portfolio": _brief(pr["aggregate"]["weighted"]),
           "worst_id": pr["aggregate"]["worst"]["id"],
           "ruin_prob_weighted": by[str(base)]["ruin_prob_weighted"],
           "safe_rate_weighted": by[str(base)]["safe_rate_weighted"],
           "withdrawal_horizon_years": base, "withdrawal_by_horizon": by}
    if tickers:
        out["cash_flow_by_horizon"] = cash
    return out


if __name__ == "__main__":
    from ..context import EngineContext

    t0 = time.time()
    res = run(EngineContext.load(selection=True))
    agg = res["portfolio"]["aggregate"]["weighted"]
    wdr = res["withdrawal"]
    print(f"60/40 weighted CAGR p50 {agg['cagr_p50']:.4f}, real {agg['real_cagr_p50']:.4f}, "
          f"max_dd_p50 {agg['max_dd_p50']:.3f}; {wdr['horizon_years']}-year ruin at 4% "
          f"{ {s: round(v['aggregate']['ruin_prob_weighted'], 4) for s, v in wdr['strategies'].items()} }, safe rates "
          f"{ {s: v['weighted'] for s, v in wdr['safe_rate'].items()} } ({time.time() - t0:.1f}s)")
