"""Long-run anchors knowable at a train end: the pre-registered rule of SIM_DESIGN §5.5.

Backtests and evaluations must not take their long-run levels from today's knowledge. The monthly engine's
DEFAULT_REVERSION (r* 0.75, term premium 0.75, Baa - 10y spread 2.0, unemployment 4.8) are post-2008 judgement
values for forward scenarios (SIM_DESIGN §3.8); a window trained to 1994 has to pull toward what was knowable
then. The rule was fixed before any result was seen and is crude on purpose (it is never tuned):

  r_star   trailing 40-quarter mean of (fed funds - core CPI y/y)      -> the engine's r_star
  term     trailing 40-quarter mean of (10-year yield - fed funds)     -> term_premium (the 10-year's target
                                                                          is r* + inflation + term_premium)
  spread   trailing 40-quarter mean of Baa - 10-year                    -> spread_level
  unrate   trailing 80-quarter mean of the unemployment rate            -> unrate_level
  pi_lr    the Michigan 5-10-year median at the train end from 1990Q2 on; before that, or while the panel
           carries no such series, the trailing 40-quarter mean of core CPI y/y (the monthly engine has no
           pi_lr: its policy-rate target follows the simulated inflation itself)

Timing. A quarter's value is the one known at its end, as the quarterly states date them (sim_redesign qdata):
the policy rate, the 10-year yield, the spread and the Michigan survey at the quarter's last month; core CPI y/y
and the unemployment rate one month earlier (they are released during the following month). The trailing
quarters of a train end `end` are the samples three months apart that end at `end` (calendar quarters for a
quarter-end month), so only panel rows up to `end` are ever read, and a panel truncated at `end` gives the
same anchors as the full one.

Check values (SIM_DESIGN §5.5 from design/d02 and d05; tests/test_anchors.py): 1994Q4 2.19 / 1.67 / 2.01 / 7.05,
2004Q4 1.79 / 1.24 / 2.26 / 5.73, 2014Q4 -0.35 / 1.73 / 2.71 / 6.01, 2026Q2 -0.74 / 0.46 / 2.04 / 5.77
(r* / term / spread / unemployment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUARTERS = 40                      # r*, the term and spread anchors, the core-CPI rule of pi_lr
UNRATE_QUARTERS = 80               # the unemployment anchor
STEP_MONTHS = 3                    # one quarter between samples
MICH_LR_START = "1990-04"          # 1990Q2: the Michigan 5-10-year question is asked every month from here
MICH_LR_COLUMN = "s_mich5y"        # the panel column of the Michigan 5-10-year median, once a panel carries it
MICH_LR_MAX_AGE = 2                # months: an older survey value does not count as "at the train end"
# panel column and months of release lag of each input (the lag is applied before sampling)
INPUTS = {"ff": ("s_ff", 0), "y10": ("s_y10", 0), "spread": ("s_spread", 0), "core_yoy": ("s_core_yoy", 1),
          "unrate": ("s_unrate", 1)}
# the anchors -> simulate()'s reversion keys (engine/simulate.DEFAULT_REVERSION); kappas and the oil target keep
# their defaults (the real-oil target is already the month database's own training median)
REVERSION_KEYS = {"r_star": "r_star", "term": "term_premium", "spread": "spread_level", "unrate": "unrate_level"}
RULES = {
    "r_star": "过去 40 个季度（联邦基金利率 − 核心 CPI 同比）的均值",
    "term": "过去 40 个季度（10 年期收益率 − 联邦基金利率）的均值",
    "spread": "过去 40 个季度 Baa − 10 年期利差的均值",
    "unrate": "过去 80 个季度失业率的均值",
    "pi_lr": "1990 年第 2 季度起取训练期末的密歇根 5–10 年通胀预期中位数，之前（或面板没有这一序列时）取过去 40 个季度核心 CPI 同比的均值",
}


def _month(end) -> pd.Timestamp:
    t = pd.Timestamp(end)
    return pd.Timestamp(year=t.year, month=t.month, day=1)


def sample_months(end, n: int, lag: int = 0) -> list[pd.Timestamp]:
    """The n quarter samples of a train end, newest first: end - lag, end - lag - 3, ... (month starts)."""
    t = _month(end)
    return [t - pd.DateOffset(months=STEP_MONTHS * k + lag) for k in range(n)]


def _samples(P: pd.DataFrame, name: str, end, n: int) -> pd.Series:
    col, lag = INPUTS[name]
    if col not in P.columns:
        raise ValueError(f"面板缺少 {col}，无法计算训练期末的长期锚")
    months = sample_months(end, n, lag)
    s = P[col].reindex(months)
    if s.isna().any():
        first = months[-1].strftime("%Y-%m")
        raise ValueError(f"训练期末 {_month(end):%Y-%m} 之前的 {n} 个季度里 {col} 有缺失（最早需要 {first}），无法计算长期锚")
    return s


def michigan_lr(P: pd.DataFrame, end, series: pd.Series | None = None) -> float | None:
    """The Michigan 5-10-year median at the train end (the latest value at most MICH_LR_MAX_AGE months old), or
    None before 1990Q2 or when neither `series` nor the panel's MICH_LR_COLUMN holds one."""
    t = _month(end)
    if t < pd.Timestamp(MICH_LR_START):
        return None
    s = series if series is not None else (P[MICH_LR_COLUMN] if MICH_LR_COLUMN in P.columns else None)
    if s is None:
        return None
    s = s.loc[:t].dropna()
    if s.empty or s.index[-1] < t - pd.DateOffset(months=MICH_LR_MAX_AGE):
        return None
    return float(s.iloc[-1])


def train_end_anchors(panel: pd.DataFrame, end, mich_lr: pd.Series | None = None) -> dict:
    """The §5.5 anchors knowable at the train end `end` (a month; only rows up to it are read).

    Returns {"end", "r_star", "term", "spread", "unrate", "pi_lr", "pi_lr_source", "from", "rules"}: levels in
    percent (pp for the two spreads); "pi_lr_source" is "michigan_5_10y" or "core_cpi_40q"; "from" the first
    month each anchor reads. `mich_lr` is a monthly Michigan 5-10-year series to use instead of the panel's
    column. A ValueError (Chinese) when a train end has fewer than the rule's quarters of history.
    """
    t = _month(end)
    P = panel.loc[:t]
    ff, core = _samples(P, "ff", t, QUARTERS), _samples(P, "core_yoy", t, QUARTERS)
    y10, spread = _samples(P, "y10", t, QUARTERS), _samples(P, "spread", t, QUARTERS)
    unrate = _samples(P, "unrate", t, UNRATE_QUARTERS)
    out = {"end": t.strftime("%Y-%m"),
           "r_star": float(np.mean(ff.to_numpy(float) - core.to_numpy(float))),
           "term": float(np.mean(y10.to_numpy(float) - ff.to_numpy(float))),
           "spread": float(spread.mean()), "unrate": float(unrate.mean())}
    mich = michigan_lr(P, t, mich_lr)
    if mich is not None:
        out["pi_lr"], out["pi_lr_source"] = mich, "michigan_5_10y"
    else:
        out["pi_lr"], out["pi_lr_source"] = float(core.mean()), "core_cpi_40q"
    out["from"] = {"r_star": core.index[-1].strftime("%Y-%m"), "term": ff.index[-1].strftime("%Y-%m"),
                   "spread": spread.index[-1].strftime("%Y-%m"), "unrate": unrate.index[-1].strftime("%Y-%m")}
    out["rules"] = dict(RULES)
    return out


def reversion(anchors: dict) -> dict:
    """simulate()'s `reversion` overrides from train_end_anchors: r_star, term_premium, spread_level, unrate_level."""
    return {key: float(anchors[name]) for name, key in REVERSION_KEYS.items()}


def window_reversion(panel: pd.DataFrame, end) -> dict:
    """The reversion a backtest window trained to `end` simulates with (reversion(train_end_anchors(...)))."""
    return reversion(train_end_anchors(panel, end))
