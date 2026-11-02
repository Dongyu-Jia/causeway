"""Withdrawal strategies and ruin on simulated scenario paths.

A retiree invests `initial` at the target weights at t=0 and takes
rate * initial out at every year end, indexed to the path's own CPI when
inflation_indexed (docs/ARCHITECTURE.md). Nothing else is rebalanced.

  pro_rata            income stays in the asset that paid it; the withdrawal
                      sells every holding in proportion to its value.
  div_then_pro_rata   income goes to a cash pot during the year (the pot earns
                      nothing); at year end the pot pays the withdrawal, a
                      surplus is reinvested pro-rata, a shortfall is sold
                      pro-rata.
  div_then_winners    like div_then_pro_rata, but the shortfall is sold from
                      the holding with the best total return that year first,
                      then the next best once it is exhausted, and so on.

Ruin: holdings (plus the pot) cannot cover the year's withdrawal. What is left
is paid out, the ruin year is recorded and wealth stays 0 afterwards.

What the dividend-first rules use as "dividend": the engine's income return of
each month (simulate(): the anchor's trailing 12-month yield / 12 on equities
and REITs, i.e. a constant yield on the current price; y10 / 12 on the
treasury; the whole return on cash), paid on the holding's price-only value
(year_factors). So in a bust the equity part of the pot falls one-for-one with
prices, where real dividends are sticky. A sticky split was tested on the
2026-11-01 build (30-year horizon with the continuation, 4%, both
dividend-first rules, every preset) and left out: dividends per share
following a smoothed price (error correction toward the price-proportional
level, half-life 12 / 24 / 48 months, trend = the engine's unconditional
price drift, total returns unchanged) raised the ruin probability by 0.03 to
0.77 pp (re-checked on the one-path-per-row continuation: 60/40 +0.04..+0.22,
all equity +0.03..+0.05, all weather +0.16..+0.39, income +0.14..+0.51,
inflation hedge +0.14..+0.77; slightly *worse*: the extra cash paid in a bust
comes out of the price return, so the pro-rata reinvestment holds less of the
asset that then recovers) and moved no safe rate by more than 0.02 pp. Only the
cash / price split moves, the total return is fixed, and today's equity yields
are low (US equity 1.1%), so the pot is mostly bond coupons and cash interest.

A ticker column (portfolio.tickers, INSTRUMENTS_DESIGN §7.3) brings its own
distributions instead: its payout model's cash per start-of-month value (sticky
payouts for the BDCs, MLPs, MO, SCHD and BME, 14.3% of NAV for QQQI, the coupon
yield for TLT, index dividends for QQQ and XLG, nothing for the leveraged funds
and BRK.B), through the same year_factors. Nothing here depends on where a
column comes from. cash_flows() and cut_flags() report the pot against the
withdrawal and a holding's distribution cuts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import MONTHS_PER_YEAR, clean_sample, fan, weighted_mean, weighted_percentile

STRATEGIES = {
    "pro_rata": {
        "name": "按比例提取",
        "description": "股息和利息留在原资产里再投资；每年年末按各持仓市值的比例卖出，凑足当年提取额。"},
    "div_then_pro_rata": {
        "name": "先用股息/利息，不足按比例",
        "description": "年内收到的股息和利息先放进现金池（不计息）；年末先用现金池支付，"
                       "有剩余按持仓比例再投资，不够的部分按持仓比例卖出。"},
    "div_then_winners": {
        "name": "先用股息，不足按当年收益从高到低卖",
        "description": "和上一种一样先用现金池；不够的部分从当年总收益最高的资产开始卖，卖光了再卖下一个。"},
}
LINEAR_STRATEGIES = ("pro_rata", "div_then_pro_rata")   # composition never depends on the withdrawal
SAFE_RUIN_PROB = 0.05           # a rate is "safe" when at most 5% of paths are ruined
RATE_BOUNDS = (0.0, 0.20)       # search interval for safe rates (fraction of initial per year)


@dataclass
class YearFactors:
    """Per path, year and asset growth over each year: the withdrawal loop's inputs."""
    total: np.ndarray      # (P, Y, A) total-return growth factor (income reinvested)
    price: np.ndarray      # (P, Y, A) price-only growth factor
    income: np.ndarray     # (P, Y, A) income paid out per unit of start-of-year value
    cpi: np.ndarray        # (P, Y+1) CPI at year ends relative to t=0 (index 0 = 1)
    order: np.ndarray      # (P, Y, A) assets by descending total growth (sale order)

    @property
    def years(self) -> int:
        return self.total.shape[1]

    def rows(self, a: int, b: int) -> "YearFactors":
        return YearFactors(self.total[a:b], self.price[a:b], self.income[a:b], self.cpi[a:b], self.order[a:b])

    def take(self, idx: np.ndarray) -> "YearFactors":
        """The paths idx (an index array, repeats allowed), in that order."""
        idx = np.asarray(idx, dtype=np.int64)
        return YearFactors(self.total[idx], self.price[idx], self.income[idx], self.cpi[idx], self.order[idx])

    def chain(self, tail: "YearFactors", rows: np.ndarray | None = None) -> "YearFactors":
        """These years followed by tail's: path k goes on with tail path rows[k] (tail's own k when None).

        The head is copied unchanged, so everything up to self.years is exactly what self alone gives.
        tail.cpi is relative to its own start (index 0 = 1) and is chained onto each path's last
        year-end CPI, so an inflation-indexed withdrawal keeps the purchasing power of t = 0.
        """
        t = tail if rows is None else tail.take(rows)
        if t.total.shape[0] != self.total.shape[0] or t.total.shape[2] != self.total.shape[2]:
            raise ValueError("chain: the tail must have one path per head path and the same assets")
        cat = lambda a, b: np.concatenate([a, b], axis=1)                     # noqa: E731
        return YearFactors(cat(self.total, t.total), cat(self.price, t.price), cat(self.income, t.income),
                           cat(self.cpi, self.cpi[:, -1:] * (t.cpi[:, 1:] / t.cpi[:, :1])), cat(self.order, t.order))

    def select(self, assets: list[int] | None = None, years: int | None = None) -> "YearFactors":
        """Reorder assets and/or keep the first `years` years."""
        Y = self.years if years is None else years
        cols = slice(None) if assets is None else assets
        f = YearFactors(self.total[:, :Y, cols], self.price[:, :Y, cols], self.income[:, :Y, cols],
                        self.cpi[:, :Y + 1], self.order[:, :Y])
        if assets is not None:
            f.order = np.argsort(-f.total, axis=2, kind="stable")
        return f

    @staticmethod
    def concat(parts: list["YearFactors"]) -> "YearFactors":
        if len(parts) == 1:
            return parts[0]
        return YearFactors(*(np.concatenate([getattr(p, k) for p in parts])
                             for k in ("total", "price", "income", "cpi", "order")))


def year_factors(total: np.ndarray, income: np.ndarray, cpi: np.ndarray,
                 months: int = MONTHS_PER_YEAR) -> YearFactors:
    """Monthly (P, T, A) total and income returns + (P, T+1) CPI levels -> YearFactors.

    The income of month m is income_m times the holding's value at the start
    of month m, the holding compounding at the price return (total - income)
    only, because the income is paid out. Only whole years are kept.
    """
    P, T, A = total.shape
    Y = T // months
    gt, gp, iy = (np.empty((P, Y, A)) for _ in range(3))
    for y in range(Y):
        r = total[:, y * months:(y + 1) * months, :].astype(np.float64)
        inc = income[:, y * months:(y + 1) * months, :].astype(np.float64)
        price = np.cumprod(1.0 + r - inc, axis=1)
        gt[:, y] = np.prod(1.0 + r, axis=1)
        gp[:, y] = price[:, -1]
        iy[:, y] = inc[:, 0] + (inc[:, 1:] * price[:, :-1]).sum(axis=1)
    c = np.asarray(cpi, dtype=np.float64)
    cpi_y = c[:, 0:Y * months + 1:months] / c[:, :1]
    return YearFactors(gt, gp, iy, cpi_y, np.argsort(-gt, axis=2, kind="stable"))


@dataclass
class WithdrawalPaths:
    wealth: np.ndarray                  # (P, Y+1) nominal wealth after each year-end withdrawal
    ruin_year: np.ndarray               # (P,) first year (1..Y) not fully paid, 0 = never ruined
    holdings: np.ndarray | None = None  # (P, Y+1, A) when record_holdings

    def rows(self, a: int, b: int) -> "WithdrawalPaths":
        return WithdrawalPaths(self.wealth[a:b], self.ruin_year[a:b],
                               None if self.holdings is None else self.holdings[a:b])

    def head(self, years: int) -> "WithdrawalPaths":
        """The first `years` year ends: what a run of exactly `years` years gives (a withdrawal never
        depends on later years), with a ruin after them not counted."""
        ry = np.where(self.ruin_year > years, 0, self.ruin_year).astype(self.ruin_year.dtype)
        return WithdrawalPaths(self.wealth[:, :years + 1], ry,
                               None if self.holdings is None else self.holdings[:, :years + 1])


def _sell_in_order(h: np.ndarray, amount: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Holdings after selling `amount` (P,) from h (P, A), exhausting assets in `order`."""
    hs = np.take_along_axis(h, order, axis=1)
    sold = np.clip(amount[:, None] - (np.cumsum(hs, axis=1) - hs), 0.0, hs)
    out = np.empty_like(h)
    np.put_along_axis(out, order, hs - sold, axis=1)
    return out


def simulate(f: YearFactors, weights: np.ndarray, initial: float = 1.0, rate=0.04,
             strategy: str = "pro_rata", inflation_indexed: bool = True, years: int | None = None,
             record_holdings: bool = False) -> WithdrawalPaths:
    """Run one strategy on every path for `years` year-end withdrawals.

    rate: scalar or (P,) fraction of `initial` withdrawn per year; the year-y
    amount is rate * initial * CPI_y / CPI_0 when inflation_indexed, else
    rate * initial. weights: (A,) summing to 1.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知提取策略：{strategy}")
    P, _, A = f.total.shape
    Y = f.years if years is None else int(years)
    w = np.asarray(weights, dtype=float)
    base = np.broadcast_to(np.asarray(rate, dtype=float) * float(initial), (P,))
    h = np.outer(np.full(P, float(initial)), w)
    wealth = np.empty((P, Y + 1))
    wealth[:, 0] = h.sum(axis=1)
    hold = np.empty((P, Y + 1, A)) if record_holdings else None
    if hold is not None:
        hold[:, 0] = h
    ruin = np.zeros(P, dtype=np.int32)
    for y in range(Y):
        need = base * f.cpi[:, y + 1] if inflation_indexed else base
        if strategy == "pro_rata":
            grown, pot = h * f.total[:, y], 0.0
        else:
            grown, pot = h * f.price[:, y], np.einsum("pa,pa->p", h, f.income[:, y])
        H = grown.sum(axis=1)
        avail = H + pot
        broke = avail < need
        ruin[broke & (ruin == 0)] = y + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(H > 0, (avail - need) / H, 0.0)
        h = grown * scale[:, None]                     # pro-rata sale or pro-rata reinvestment
        if strategy == "div_then_winners":
            short = need - pot
            if (short > 0).any():
                sold = _sell_in_order(grown, np.maximum(short, 0.0), f.order[:, y])
                h = np.where((short > 0)[:, None], sold, h)
        h[broke] = 0.0
        wealth[:, y + 1] = h.sum(axis=1)
        if hold is not None:
            hold[:, y + 1] = h
    return WithdrawalPaths(wealth, ruin, hold)


def cash_flows(res: WithdrawalPaths, f: YearFactors, amount, inflation_indexed: bool = True) -> dict:
    """Per path and year: the cash the start-of-year holdings paid during the year (their income; the pot of the
    dividend-first rules, reinvested by pro_rata), the year's withdrawal and whether the path was still alive at
    the start of the year. amount: rate * initial (scalar or (P,)), the year-y withdrawal being amount * CPI_y /
    CPI_0 when inflation_indexed. Needs simulate(..., record_holdings=True).

    Returns {"pot" (P, Y), "need" (P, Y), "alive" (P, Y)}; a ruined path's pot is 0 from the year after its ruin."""
    if res.holdings is None:
        raise ValueError("cash_flows needs simulate(..., record_holdings=True)")
    Y = res.wealth.shape[1] - 1
    h = res.holdings[:, :Y]                                   # holdings at the start of each year
    pot = np.einsum("pya,pya->py", h, f.income[:, :Y])
    base = np.broadcast_to(np.asarray(amount, dtype=float), (h.shape[0],))
    need = base[:, None] * (f.cpi[:, 1:Y + 1] if inflation_indexed else np.ones((1, Y)))
    ry = res.ruin_year[:, None]
    alive = (ry == 0) | (ry >= np.arange(1, Y + 1)[None, :])
    return {"pot": pot, "need": need, "alive": alive}


def cut_flags(f: YearFactors, column: int, years: int = 10, cut: float = 0.2) -> np.ndarray:
    """Per path, whether one unit held from t = 0 (distributions not reinvested) receives in some year 2..`years`
    at least `cut` less cash than in year 1 (NaN on a path whose year-1 cash is 0): the distribution-cut pressure
    of a holding (INSTRUMENTS_DESIGN §4.2, §7.3)."""
    Y = min(int(years), f.years)
    inc, pr = f.income[:, :Y, column], f.price[:, :Y, column]
    unit = np.concatenate([np.ones((inc.shape[0], 1)), np.cumprod(pr[:, :-1], axis=1)], axis=1)
    cash = inc * unit
    first = cash[:, 0]
    if Y < 2:
        return np.where(first > 0, False, np.nan)
    low = cash[:, 1:].min(axis=1)
    return np.where(first > 0, low < (1.0 - cut) * first, np.nan)


def critical_rates(f: YearFactors, weights: np.ndarray, strategy: str, inflation_indexed: bool = True,
                   years: int | None = None, tol: float = 1e-5) -> np.ndarray:
    """Per path, the largest rate (fraction of initial per year) that survives `years`.

    Clipped to RATE_BOUNDS. pro_rata and div_then_pro_rata never let the
    withdrawal change the portfolio's composition, so wealth follows
    W_y = W_{y-1} R_y - D_y with fixed path returns R_y. Dividing by the
    no-withdrawal multiple M_y gives W_y / M_y = W_0 - sum_{k<=y} D_k / M_k,
    so with D_k = rate * W_0 * c_k (c_k = CPI_k / CPI_0 or 1) a path survives
    iff rate <= 1 / sum_k c_k / M_k. div_then_winners sells a rate-dependent
    mix, so it is bisected per path on RATE_BOUNDS to `tol`.
    """
    lo, hi = RATE_BOUNDS
    P = f.total.shape[0]
    Y = f.years if years is None else int(years)
    if strategy in LINEAR_STRATEGIES:
        M = simulate(f, weights, 1.0, 0.0, strategy, inflation_indexed, Y).wealth[:, 1:]
        c = f.cpi[:, 1:Y + 1] if inflation_indexed else 1.0
        return np.clip(1.0 / (c / M).sum(axis=1), lo, hi)
    lo_v, hi_v = np.full(P, lo), np.full(P, hi)
    lo_v[simulate(f, weights, 1.0, hi_v, strategy, inflation_indexed, Y).ruin_year == 0] = hi
    for _ in range(int(np.ceil(np.log2((hi - lo) / tol)))):
        mid = 0.5 * (lo_v + hi_v)
        alive = simulate(f, weights, 1.0, mid, strategy, inflation_indexed, Y).ruin_year == 0
        lo_v = np.where(alive, mid, lo_v)
        hi_v = np.where(alive, hi_v, mid)
    return lo_v


def ruin_curve(crit: np.ndarray, rates, weights=None) -> list[float]:
    """Ruin probability P(r* < rate) at each of `rates`, from critical_rates() output."""
    v, w = clean_sample(crit, weights)
    if v.size == 0:
        return [float("nan")] * len(np.atleast_1d(rates))
    order = np.argsort(v, kind="stable")
    cw = np.concatenate([[0.0], np.cumsum(w[order])]) / w.sum()
    return cw[np.searchsorted(v[order], np.atleast_1d(rates), side="left")].tolist()


def ruin_probability(crit: np.ndarray, rate: float, weights=None) -> float:
    """Probability of ruin at `rate`, from critical_rates() output."""
    return ruin_curve(crit, [rate], weights)[0]


def safe_rate(crit: np.ndarray, weights=None, max_ruin: float = SAFE_RUIN_PROB) -> float | None:
    """Largest rate in RATE_BOUNDS whose ruin probability P(r* < rate) is <= max_ruin.

    crit: per-path critical rates. This is the limit of bisecting the
    (monotone) ruin probability on RATE_BOUNDS; the result is floored to 0.01%
    so the reported rate itself is safe. None for an empty sample.
    """
    v, w = clean_sample(crit, weights)
    if v.size == 0:
        return None
    order = np.argsort(v, kind="stable")
    v, cw = v[order], np.cumsum(w[order]) / w.sum()
    j = min(int(np.searchsorted(cw, max_ruin + 1e-12, side="right")), v.size - 1)
    return float(np.floor(min(v[j], RATE_BOUNDS[1]) * 1e4 + 1e-9) / 1e4)


def summarize(res: WithdrawalPaths, cpi: np.ndarray, weights=None, decimals: int = 6,
              with_fan: bool = True) -> dict:
    """Ruin and wealth statistics of simulate() output under optional path weights.

    cpi: (P, >= Y+1) CPI at year ends relative to t=0. Wealth figures are in
    the currency of `initial`; *_real ones are deflated to t=0 money by each
    path's own CPI. survival[k] = probability of not being ruined by year end
    k (survival[0] = 1). depletion_year_median: median ruin year among ruined
    paths (1 = the first year end), None when no path is ruined. `decimals`
    rounds the wealth fan (left out when not with_fan).
    """
    Y = res.wealth.shape[1] - 1
    real = res.wealth / cpi[:, :Y + 1]
    ry = res.ruin_year
    ruined = ry > 0
    w = np.ones(ry.size) if weights is None else np.asarray(weights, dtype=float)
    depletion = weighted_percentile(ry[ruined], 50, w[ruined]) if ruined.any() else float("nan")
    out = {
        "ruin_prob": weighted_mean(ruined, w),
        "survival": [1.0 - weighted_mean(ruined & (ry <= k), w) for k in range(Y + 1)],
        "terminal_real_p5": weighted_percentile(real[:, -1], 5, w),
        "terminal_real_p50": weighted_percentile(real[:, -1], 50, w),
        "terminal_real_p95": weighted_percentile(real[:, -1], 95, w),
        "terminal_nominal_p50": weighted_percentile(res.wealth[:, -1], 50, w),
        "depletion_year_median": None if np.isnan(depletion) else depletion,
    }
    if with_fan:
        out["wealth_fan_real"] = fan(real, w, decimals=decimals)
    return out
