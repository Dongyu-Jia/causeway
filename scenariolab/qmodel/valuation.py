"""The valuation block of the quarterly stack (SIM_DESIGN §6).

The identity (V-PRICE)
    log P_q (CRSP) = log M_q + log CPI_q + log E10r_q
M is the CAPE multiple on the CRSP price (level arbitrary), E10r the 10-year average of S&P real EPS and CPI
Shiller's (2026 workbook). So y_us_equity = y_mult + y_fund with y_mult = Δlog M and y_fund = Δlog CPI + Δlog E10r;
the quarterly panel builds both legs (P1 tests the identity to 1e-10). The trailing P/E is display only (write-downs
distort it) and the P/D is not a re-rating measure (about 30% of its 1973-2025 rise is payout decline).

What this module does
  * The legs' training equations. y_fund (smooth: mostly CPI pass-through) and y_dg (the growth of CRSP trailing
    12-month dividends, the income leg) are OLS on quarterly-stack fields (headline CPI, IP, unemployment, real GDP;
    config_q.FUND_REGRESSORS) plus their latest own growth known at the start of the quarter. The valuation report's
    own equations used NIPA nominal GDP and the profit share, which are not quarterly-stack fields;
    research_equations() refits them for the check values (SIM_DESIGN §6.1).
  * The scenario's valuation statements (c_q.valuation, §6.3). No statement = inferred: the multiple does whatever
    the macro and credit path implies, y_mult = price - y_fund, and the residual multiple has zero drift (the asset
    layer's no_rerate CMA removes the historical re-rating, U-5). hold and rerate are soft constraints on the sum of
    y_mult over the stated quarters, with sizes from historical quantiles of the same span; a reversion half-life
    (>= 14 years, inferred only) adds an explicit slow pull. Per path the us_equity residual is shifted so the
    statement holds and the other equities move by their residual regression on it (the qmodel's method), so nasdaq
    and the sectors inherit the market's re-rating. Never stated in test mode (the narrative test).
  * Implied CAPE and P/D paths, today's valuation states with percentiles and caveats, and the UI wording of a
    statement (「相当于直接假设股价」).
No pull: nothing moves the multiple toward a fair value unless a scenario states a reversion (every out-of-sample
pull lost, §6.2).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config_q as cq

ASSETS = list(cq.ASSETS)
I_EQ = ASSETS.index("us_equity")
FOLLOWERS = [a for a in cq.EQUITY_LIKE if a != "us_equity"]     # inherit the market's re-rating (§6.3)
FIRST_LEG_QUARTER = "1973Q2"            # the first quarter with a change of every quarterly level (panel from 1973Q1)
SIZES = list(cq.VALUATION_SIZES)        # 温和 / 明显 / 极端
STATEMENT_MODES = list(cq.VALUATION_MODES)
HAC_LAGS = 4

# UI wording (SIM_DESIGN §6.3, §10.6): a valuation statement is a price assumption
WORDING = "估值假设：这一条相当于直接假设美股价格在这 {n} 个季度里 {price}（其中基本面腿 {fund}）"
CAVEATS = {
    "cape": "Shiller 2026 版工作簿；E 的最近两个季度可能是估计值",
    "ecy": "超额 CAPE 收益率 = 100/CAPE − 10 年实际利率（10 年期收益率 − 过去 10 年年化 CPI 通胀）；只作展示",
    "pd": "CRSP 全市场价格 / 过去 12 个月股息；约 30% 的 1973–2025 年 P/D 上升是派息率下降，不当作估值",
    "pe": "滚动市盈率被 2001–02、2008–09 的减记扭曲：只展示，不进模型",
    "fcfy": "全部非金融公司的代理，不是 SPY/QQQ 的自由现金流收益率；水平状态样本外无增益",
    "pb_buseq": "Ken French BusEq（科技）市净率：唯一连续的科技估值，只作展示；纳指的估值只能用全市场 CAPE 近似",
}


# ---------------------------------------------------------------- legs and their history

def leg_frame(Q: pd.DataFrame) -> pd.DataFrame:
    """The us_equity legs and the identity gap per quarter: y_us_equity, y_mult, y_fund, y_pd, y_dg, and
    gap = y_us_equity - y_mult - y_fund (0 up to rounding)."""
    out = Q[["y_us_equity", "y_mult", "y_fund", "y_pd", "y_dg"]].copy()
    out["gap"] = out["y_us_equity"] - out["y_mult"] - out["y_fund"]
    return out


def _train(Q: pd.DataFrame, col: str, train_end, start: str = FIRST_LEG_QUARTER) -> pd.Series:
    return Q[col].loc[pd.Period(start, "Q"):pd.Period(str(train_end), "Q")].dropna()


def mult_drift(Q: pd.DataFrame, train_end) -> float:
    """Mean quarterly Δlog CAPE multiple over 1973Q2..train_end (the historical re-rating m̄ of §4.4)."""
    return float(_train(Q, "y_mult", train_end).mean())


def mult_drift_pct(Q: pd.DataFrame, train_end) -> float:
    """m̄ in % a year (400 × the quarterly mean): +1.18 / +1.80 / +1.47 / +1.89 at 1994Q4 / 2004Q4 / 2014Q4 /
    2026Q3 in the design's d05."""
    return 400.0 * mult_drift(Q, train_end)


def size_bins_from(y_mult: np.ndarray, span: int) -> dict[str, float]:
    """温和 / 明显 / 极端 = p50 / p75 / p95 of |Σ Δlog M| over every window of `span` consecutive quarters."""
    y = np.asarray(y_mult, float)
    span = int(span)
    if span < 1 or len(y) < span:
        raise ValueError(f"跨度 {span} 个季度超过训练样本（{len(y)} 个季度）")
    c = np.convolve(y, np.ones(span), mode="valid")
    a = np.abs(c)
    return {s: float(np.quantile(a, cq.VALUATION_SIZE_QUANTILE[s])) for s in SIZES}


def size_bins(Q: pd.DataFrame, train_end, span: int) -> dict[str, float]:
    """The re-rating size bins for a statement over `span` quarters, from the training quarters 1973Q2..train_end
    (SIM_DESIGN §6.3; on 1973Q2..2014Q4: 4Q 0.12 / 0.20 / 0.34 ... 20Q 0.30 / 0.44 / 0.62)."""
    y = _train(Q, "y_mult", train_end)
    full = pd.period_range(y.index.min(), y.index.max(), freq="Q")
    if len(full) != len(y):
        raise ValueError("训练期的倍数变化不连续，无法计算重估幅度")
    return size_bins_from(y.to_numpy(float), span)


# ---------------------------------------------------------------- the legs' training equations

def fund_regressors(frame) -> dict[str, np.ndarray]:
    """The equations' inputs from quarterly-stack fields (a DataFrame or a dict of arrays): dcpi = d_cpi/400
    (headline CPI log change), dip = d_ip/100, dun = d_unrate, drgdp = d_rgdp/400."""
    get = (lambda c: frame[c].to_numpy(float)) if isinstance(frame, pd.DataFrame) else \
        (lambda c: np.asarray(frame[c], float))
    return {"dcpi": get("d_cpi") / 400.0, "dip": get("d_ip") / 100.0, "dun": get("d_unrate"),
            "drgdp": get("d_rgdp") / 400.0}


def hac_ols(y: np.ndarray, X: np.ndarray, lags: int = HAC_LAGS) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """OLS of y on [1, X] with Newey-West (Bartlett) t statistics: (beta, t, R², residuals)."""
    A = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    e = y - A @ b
    inv = np.linalg.pinv(A.T @ A)
    u = A * e[:, None]
    S = u.T @ u
    for L in range(1, min(lags, len(y) - 1) + 1):
        G = u[L:].T @ u[:-L]
        S += (1.0 - L / (lags + 1.0)) * (G + G.T)
    se = np.sqrt(np.maximum(np.diag(inv @ S @ inv), 0.0))
    t = np.divide(b, se, out=np.full_like(b, np.nan), where=se > 0)
    sst = float(((y - y.mean()) ** 2).sum())
    return b, t, (1.0 - float(e @ e) / sst) if sst > 0 else float("nan"), e


@dataclass
class LegEquation:
    """target_q = c + Σ b_k x_k,q + g · target_{q-lag} + e_q, fitted by OLS on the training quarters."""
    target: str
    lag: int
    names: list
    coef: list
    t: list
    r2: float
    n: int
    resid_sd: float
    first: str
    train_end: str
    resid: pd.Series = field(default=None, repr=False)

    def predict(self, reg: dict, own_lag) -> np.ndarray:
        """Mean of the target for regressor arrays `reg` (fund_regressors) and the own lagged growth (broadcast)."""
        b = dict(zip(self.names, self.coef))
        out = b["const"] + b["g_lag"] * np.asarray(own_lag, float)
        for k in cq.FUND_REGRESSORS:
            out = out + b[k] * np.asarray(reg[k], float)
        return out

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("resid")
        return d


def fit_leg(Q: pd.DataFrame, target: str, train_end, lag: int | None = None) -> LegEquation:
    """Fit the y_fund or y_dg equation on the quarters FIRST_QUARTER..train_end where the target, the regressors and
    the lagged target exist (the lag: config_q.FUND_OWN_LAG; Shiller E is published with a lag of 3 months, so the
    latest y_fund known at the start of q is that of q-2; CRSP dividends are known at once, so y_dg of q-1)."""
    lag = int(cq.FUND_OWN_LAG[target] if lag is None else lag)
    te = pd.Period(str(train_end), "Q")
    reg = pd.DataFrame(fund_regressors(Q), index=Q.index)
    D = pd.concat([Q[target].rename("y"), reg, Q[target].shift(lag).rename("g_lag")], axis=1)
    D = D.loc[pd.Period(cq.FIRST_QUARTER, "Q"):te].dropna()
    if len(D) < 3 * (len(cq.FUND_REGRESSORS) + 2):
        raise ValueError(f"{target}：训练期只有 {len(D)} 个季度，无法估计")
    cols = list(cq.FUND_REGRESSORS) + ["g_lag"]
    b, t, r2, e = hac_ols(D["y"].to_numpy(float), D[cols].to_numpy(float))
    return LegEquation(target=target, lag=lag, names=["const"] + cols, coef=[float(x) for x in b],
                       t=[float(x) for x in t], r2=float(r2), n=int(len(D)), resid_sd=float(np.std(e)),
                       first=str(D.index.min()), train_end=str(te), resid=pd.Series(e, index=D.index))


def research_equations(Q: pd.DataFrame, train_end: str = "2026Q3") -> dict[str, dict]:
    """The valuation report's equations (b) (valuation/vmodel.py FUND_SET, v17_coefs.json): the target on NIPA
    nominal GDP growth, IP, the profit-share change (pp), CPI, unemployment and the lagged own growth, HAC(4) t.
    Reads NIPA GDP and CP from the raw cache (never downloads). The check values (SIM_DESIGN §6.1): y_fund CPI 0.97
    (t 8.9, R² 0.56); y_dg CPI 0.44, own lag 0.30 (R² 0.13)."""
    from . import sources_q as sq

    def quarterly(sid):
        s = sq.fred(sid)
        s.index = pd.DatetimeIndex(s.index).to_period("Q")
        return s[~s.index.duplicated(keep="last")]

    gdp, cp = quarterly("GDP"), quarterly("CP")
    nipa = pd.DataFrame({"dlgdp": np.log(gdp).diff(), "dps": (100.0 * cp / gdp.reindex(cp.index)).diff()})
    nipa = nipa.reindex(Q.index)
    out = {}
    for target in ("y_fund", "y_dg"):
        lag = cq.FUND_OWN_LAG[target]
        D = pd.DataFrame({"y": Q[target], "dlgdp": nipa["dlgdp"], "dip": Q["d_ip"] / 100.0, "dps": nipa["dps"],
                          "dcpi": Q["d_cpi"] / 400.0, "dun": Q["d_unrate"], "g_lag": Q[target].shift(lag)})
        D = D.loc[pd.Period(cq.FIRST_QUARTER, "Q"):pd.Period(train_end, "Q")].dropna()
        cols = ["dlgdp", "dip", "dps", "dcpi", "dun", "g_lag"]
        b, t, r2, _ = hac_ols(D["y"].to_numpy(float), D[cols].to_numpy(float))
        out[target] = {"n": int(len(D)), "r2": float(r2), "coef": dict(zip(["const"] + cols, map(float, b))),
                       "t": dict(zip(["const"] + cols, map(float, t)))}
    return out


# ---------------------------------------------------------------- the block fitted to a train end

@dataclass
class ValuationBlock:
    """The valuation block fitted on the quarters up to `train_end` (inclusive)."""
    train_end: str
    fund: LegEquation
    dg: LegEquation
    m_bar: float                    # mean quarterly y_mult over 1973Q2..train_end (the historical re-rating)
    m_star: float                   # training median of the start-of-quarter log CAPE (the reversion target)
    y_mult_train: np.ndarray = field(repr=False, default=None)
    _bins: dict = field(default_factory=dict, repr=False)

    @classmethod
    def fit(cls, Q: pd.DataFrame, train_end) -> "ValuationBlock":
        te = str(pd.Period(str(train_end), "Q"))
        y = _train(Q, "y_mult", te)
        cape = Q["s_cape"].loc[:pd.Period(te, "Q")].dropna()
        return cls(train_end=te, fund=fit_leg(Q, "y_fund", te), dg=fit_leg(Q, "y_dg", te),
                   m_bar=float(y.mean()), m_star=float(cape.median()), y_mult_train=y.to_numpy(float))

    def bins(self, span: int) -> dict[str, float]:
        span = int(span)
        if span not in self._bins:
            self._bins[span] = size_bins_from(self.y_mult_train, span)
        return self._bins[span]

    def summary(self) -> dict:
        return {"train_end": self.train_end, "fund": self.fund.summary(), "dg": self.dg.summary(),
                "m_bar_q": self.m_bar, "m_bar_pct_per_year": 400.0 * self.m_bar, "m_star_log_cape": self.m_star,
                "m_star_cape": math.exp(self.m_star),
                "bins": {str(h): self.bins(h) for h in (4, 8, 12, 20) if h <= len(self.y_mult_train)}}


# ---------------------------------------------------------------- valuation statements (c_q.valuation)

@dataclass(frozen=True)
class ValuationStatement:
    quarters: tuple                 # 1-based quarter indices of the path (q = 1 .. H)
    mode: str                       # inferred / hold / rerate
    direction: int = 0              # +1 / -1 (rerate)
    size: str | None = None         # 温和 / 明显 / 极端 (rerate)
    half_life_years: float | None = None     # explicit reversion (inferred only, >= 14 years)
    quote: str = ""

    @property
    def active(self) -> bool:
        return self.mode in ("hold", "rerate") or self.half_life_years is not None


def parse_statements(entries, n_quarters: int, mode: str = "production") -> list[ValuationStatement]:
    """Validate a QTable's c_q.valuation list (Appendix B) into statements; raises ValueError with a Chinese message.

    Test mode (the narrative test) refuses every statement other than inferred (SIM_DESIGN §6.3). The quarters of
    hold, rerate and reversion statements may not overlap."""
    if mode not in ("production", "test"):
        raise ValueError(f"未知的模式 {mode!r}（production 或 test）")
    out = []
    for i, e in enumerate(entries or []):
        if not isinstance(e, dict):
            raise ValueError(f"第 {i + 1} 条估值陈述不是对象")
        m = e.get("mode")
        if m not in STATEMENT_MODES:
            raise ValueError(f"第 {i + 1} 条估值陈述的模式 {m!r} 不在 {STATEMENT_MODES} 里")
        qs = e.get("quarters")
        if not isinstance(qs, (list, tuple)) or not qs or not all(isinstance(q, (int, np.integer))
                                                                  and not isinstance(q, bool) for q in qs):
            raise ValueError(f"第 {i + 1} 条估值陈述的 quarters 必须是非空的整数列表")
        qs = tuple(sorted(set(int(q) for q in qs)))
        if qs[0] < 1 or qs[-1] > int(n_quarters):
            raise ValueError(f"第 {i + 1} 条估值陈述的季度超出范围 1..{n_quarters}")
        hl = e.get("reversion_half_life_years")
        if hl is not None:
            hl = float(hl)
            if not hl >= cq.MIN_REVERSION_HALF_LIFE_YEARS:
                raise ValueError(f"均值回归的半衰期至少 {cq.MIN_REVERSION_HALF_LIFE_YEARS} 年（写的是 {hl:g} 年）")
            if m != "inferred":
                raise ValueError("均值回归只能和 inferred 一起写：hold / rerate 已经直接规定了倍数")
        direction, size = 0, None
        if m == "rerate":
            d = e.get("direction")
            if d not in ("+", "-"):
                raise ValueError("rerate 必须写方向 direction：'+' 或 '-'")
            size = e.get("size")
            if size not in SIZES:
                raise ValueError(f"rerate 的幅度 size 必须是 {SIZES} 之一")
            direction = 1 if d == "+" else -1
        st = ValuationStatement(qs, m, direction, size, hl, str(e.get("quote") or ""))
        if mode == "test" and st.active:
            raise ValueError("叙事检验（测试模式）不允许写估值假设：估值只能由模型推断")
        out.append(st)
    act = [s for s in out if s.active]
    seen: dict[int, int] = {}
    for k, s in enumerate(act):
        for q in s.quarters:
            if q in seen:
                raise ValueError(f"估值陈述在第 {q} 季重叠：每个季度最多一条 hold / rerate / 均值回归")
            seen[q] = k
    return out


def statement_target(st: ValuationStatement, bins: dict[str, float]) -> tuple[float, float]:
    """(target, sd) of Σ y_mult over the statement's quarters (hold: 0 ± 0.25 × 温和; rerate: ±size ± 0.25 × size)."""
    if st.mode == "hold":
        return 0.0, cq.VALUATION_HOLD_SD * bins["温和"]
    if st.mode == "rerate":
        v = bins[st.size]
        return st.direction * v, cq.VALUATION_RERATE_SD * v
    raise ValueError(f"{st.mode} 不是有幅度的估值陈述")


def reversion_kappa(half_life_years: float) -> float:
    """κ per quarter for a stated half-life in years: ln 2 / (4 HL)."""
    return math.log(2.0) / (4.0 * float(half_life_years))


def shift_vector(resid_beta: dict | None) -> np.ndarray:
    """The per-asset loading of a us_equity residual shift (1 for us_equity, the residual regression coefficient for
    the other equity-like assets, 0 for gold and the mechanical assets)."""
    v = np.zeros(len(ASSETS))
    v[I_EQ] = 1.0
    for a in FOLLOWERS:
        v[ASSETS.index(a)] = float((resid_beta or {}).get(a, 0.0))
    return v


def apply_statements(price: np.ndarray, y_fund: np.ndarray, statements: list[ValuationStatement], *,
                     bins, resid_beta: dict | None, rng: np.random.Generator, log_cape0=None,
                     m_star: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Impose the valuation statements on simulated paths.

    price (n, H, 13) quarterly log price returns in config_q.ASSETS order and y_fund (n, H); bins(span) gives the
    size bins (ValuationBlock.bins). hold / rerate: per path the target of Σ y_mult is drawn from N(target, sd²) (a
    soft statement is delivered within its sd; sd = 0 would be exact) and the gap to the path's own Σ (price - y_fund)
    is spread evenly over the stated quarters as a shift of the us_equity residual, the other equities moving by
    their residual regression on it. reversion: each stated quarter adds -κ (log CAPE at its start - m_star), log CAPE
    rolled from log_cape0 with the path's (shifted) y_mult. Returns (price', shift), shift = price' - price."""
    P = np.array(price, dtype=float, copy=True)
    F = np.asarray(y_fund, float)
    n, H, _ = P.shape
    shift = np.zeros_like(P)
    v = shift_vector(resid_beta)
    for st in sorted([s for s in statements if s.active], key=lambda s: s.quarters[0]):
        qi = np.asarray(st.quarters, int) - 1
        if qi.max() >= H:
            raise ValueError(f"估值陈述的季度超出模拟长度 {H}")
        if st.mode in ("hold", "rerate"):
            target, sd = statement_target(st, bins(len(qi)))
            S = (P[:, qi, I_EQ] - F[:, qi]).sum(axis=1)
            goal = target + sd * rng.standard_normal(n)
            d = ((goal - S) / len(qi))[:, None] * v[None, :]
            P[:, qi, :] += d[:, None, :]
            shift[:, qi, :] += d[:, None, :]
        else:
            if log_cape0 is None or m_star is None:
                raise ValueError("均值回归需要起点 CAPE 与训练期中位数")
            kappa = reversion_kappa(st.half_life_years)
            lc = np.broadcast_to(np.asarray(log_cape0, float), (n,)).copy()
            stated = set(int(q) for q in qi)
            for k in range(int(qi.max()) + 1):
                if k in stated:
                    d = (-kappa * (lc - m_star))[:, None] * v[None, :]
                    P[:, k, :] += d
                    shift[:, k, :] += d
                lc = lc + P[:, k, I_EQ] - F[:, k]
    return P, shift


def implied_paths(log_cape0, log_pd0, y_mult: np.ndarray, y_pd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log CAPE and log P/D at the start of each quarter and at the end of the last: (n, H + 1) each.
    log CAPE rolls with y_mult (the CRSP-price multiple stands in for Shiller's S&P-price CAPE), log P/D with
    y_pd = y_us_equity - y_dg."""
    ym, yp = np.atleast_2d(np.asarray(y_mult, float)), np.atleast_2d(np.asarray(y_pd, float))
    n = ym.shape[0]
    c0 = np.broadcast_to(np.asarray(log_cape0, float), (n,))
    p0 = np.broadcast_to(np.asarray(log_pd0, float), (n,))
    cape = np.concatenate([c0[:, None], c0[:, None] + np.cumsum(ym, axis=1)], axis=1)
    pdr = np.concatenate([p0[:, None], p0[:, None] + np.cumsum(yp, axis=1)], axis=1)
    return cape, pdr


def statement_text(n_quarters: int, price_change: float, fund_change: float) -> str:
    """「估值假设：这一条相当于直接假设美股价格在这 N 个季度里 ±X%（其中基本面腿 +Y%）」 from the medians of the cumulative
    log price change and of the fundamentals leg over the stated quarters."""
    def pct(x):
        v = 100.0 * (math.exp(float(x)) - 1.0)
        return f"{v:+.0f}%"
    return WORDING.format(n=int(n_quarters), price=pct(price_change), fund=pct(fund_change))


# ---------------------------------------------------------------- today's valuation states (display)

def _at_start(m: pd.Series, lag: int) -> pd.Series:
    """The monthly value known at the start of each quarter: month (first month of q) - 1 - lag."""
    s = m.copy()
    s.index = pd.DatetimeIndex(s.index) + pd.DateOffset(months=1 + lag)
    s = s[s.index.month.isin((1, 4, 7, 10))]
    s.index = s.index.to_period("Q")
    return s[~s.index.duplicated(keep="last")]


def shiller_known(shiller: pd.DataFrame | None = None) -> pd.DataFrame:
    """Shiller valuation states known at the start of every quarter since his data begin: log CAPE (price of the
    month before q, CPI lag 1, E10 real lag 3: the quarterly panel's s_cape) and the excess CAPE yield (100/CAPE -
    (GS10 of the month before q - trailing 10-year annualized CPI inflation, CPI lag 1))."""
    from . import sources_q as sq
    S = sq.shiller_frame() if shiller is None else shiller
    lag_e, lag_cpi = 3, 1
    lp, lcpi, lcape = np.log(S["price"]), np.log(S["cpi"]), np.log(S["cape"])
    le10r = lp - lcpi - lcape
    log_cape = _at_start(lp, 0) - _at_start(lcpi, lag_cpi) - _at_start(le10r, lag_e)
    infl10 = 100.0 * (np.exp((lcpi - lcpi.shift(120)) / 10.0) - 1.0)
    real10 = _at_start(S["gs10"], 0) - _at_start(infl10, lag_cpi)
    ecy = 100.0 * np.exp(-log_cape) - real10.reindex(log_cape.index)
    return pd.DataFrame({"log_cape": log_cape, "ecy": ecy})


def percentile(history: pd.Series, value: float) -> float:
    """Share of history at or below value, in percent (NaN without history)."""
    h = pd.Series(history).dropna().to_numpy(float)
    if not len(h) or not np.isfinite(value):
        return float("nan")
    return float(100.0 * np.mean(h <= value))


def state_table(Q: pd.DataFrame, quarter=None, shiller: pd.DataFrame | None = None) -> list[dict]:
    """Today's valuation states (§6.2) with their history percentiles and caveats: CAPE (percentile since 1948),
    ECY, CRSP P/D, trailing P/E, NFC FCF yield and BusEq P/B. Display only: none of them is a predictor or a pull."""
    q = pd.Period(str(quarter), "Q") if quarter is not None else Q.index.max()
    K = shiller_known(shiller)
    hist_from = pd.Period(cq.VALUATION_HISTORY_FROM, "Q")
    rows = []

    def row(sid, zh, value, unit, hist, use, ref=None):
        rec = {"id": sid, "zh": zh, "quarter": str(q), "value": value, "unit": unit,
               "percentile": percentile(hist, value) if hist is not None else float("nan"),
               "history_from": str(hist.dropna().index.min()) if hist is not None and hist.notna().any() else None,
               "use": use, "caveat": CAVEATS.get(sid, "")}
        if ref:
            rec.update(ref)
        rows.append(rec)

    lc = float(Q.at[q, "s_cape"]) if "s_cape" in Q and pd.notna(Q.at[q, "s_cape"]) else float(K["log_cape"].get(q))
    cape_hist = np.exp(K["log_cape"].loc[hist_from:q])
    row("cape", "席勒 CAPE", math.exp(lc), "倍", cape_hist, "初始条件；估值陈述的参照；隐含路径")
    ecy = float(K["ecy"].get(q, np.nan))
    row("ecy", "超额 CAPE 收益率（ECY）", ecy, "%", K["ecy"].loc[hist_from:q], "展示")
    pdv = math.exp(float(Q.at[q, "s_pd"]))
    row("pd", "CRSP 市价/股息", pdv, "倍", np.exp(Q["s_pd"].loc[:q]), "收入腿；y_pd")
    if "s_pe" in Q and pd.notna(Q.at[q, "s_pe"]):
        row("pe", "滚动市盈率（P/Ê）", math.exp(float(Q.at[q, "s_pe"])), "倍", np.exp(Q["s_pe"].loc[:q]), "只展示")
    if "s_fcfy" in Q and pd.notna(Q.at[q, "s_fcfy"]):
        row("fcfy", "非金融公司自由现金流收益率（代理）", float(Q.at[q, "s_fcfy"]), "%", Q["s_fcfy"].loc[:q], "只展示")
    if "s_pb_buseq" in Q and pd.notna(Q.at[q, "s_pb_buseq"]):
        ref = {}
        peak = pd.Period("2000Q2", "Q")             # known at the start of 2000Q2 = the end of 2000Q1 (the tech peak)
        if peak in Q.index and pd.notna(Q.at[peak, "s_pb_buseq"]):
            ref = {"reference": {"2000Q1 季末": math.exp(float(Q.at[peak, "s_pb_buseq"]))}}
        row("pb_buseq", "科技行业（BusEq）市净率", math.exp(float(Q.at[q, "s_pb_buseq"])), "倍",
            np.exp(Q["s_pb_buseq"].loc[:q]), "只展示", ref)
    return rows
