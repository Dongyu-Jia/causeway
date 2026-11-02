"""The valuation block as the quarterly stack reads it (SIM_DESIGN §6; §10.3, §10.5; §14 P8): GET /api/q/valuation.

From the quarterly panel (panel.py; nothing is fitted here):

  today       the start-of-quarter valuation states with their percentile in the panel's history since 1973:
              CAPE (Shiller 2026 workbook), the excess CAPE yield 100/CAPE - real 10-year (the panel's real yield:
              y10 - the 36-month inflation trend, not Shiller's trailing-10-year one), CRSP P/D, trailing P/Ê
              (display only), the earnings-cycle position log(Ê/E10), the NFC free-cash-flow yield proxy (display
              only, labelled) and the BusEq P/B (display only; the only continuous tech valuation)
  cape_q      the CAPE history's percentiles 0..100 (the scenario detail places a simulated CAPE in history)
  identity    log P = log M + log CPI + log E10r: per window the summed log price return and its legs (the CAPE
              multiple y_mult, the fundamentals y_fund = Δlog CPI + Δlog E10r), the CRSP P/D change and the schema's
              P/Ê change (price minus Δlog Ê, which write-downs distort); and over all quarters each leg's
              correlation with the price return and its share of the price variance (cov / var)
  rerating    the mean CAPE-multiple drift per year from 1973 to each train end (§4.4: removed from forward drift)
  notes       the caveats the UI prints (E estimates, the FCF proxy, the QQQ limitation, the identity is not evidence)

No LLM, no download.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

WINDOWS = [("2007Q3", "2009Q2", "金融危机"), ("2001Q1", "2002Q4", "互联网泡沫破裂"), ("1995Q3", "2000Q1", "互联网泡沫"),
           ("2022Q1", "2022Q3", "通胀与加息")]
RERATING_ENDS = ("1994Q4", "2004Q4", "2014Q4")
NOTES = [
    "股价 = 倍数 × 基本面是恒等式：倍数一腿几乎承载了全部股价波动，所以写出一条倍数路径就等于写出一条股价路径。"
    "“没算准的部分主要是估值变化”不能当证据：常数（安慰剂）也能得到同样高的比例；只能报告模拟的倍数变化与实际倍数变化的相关、"
    "实际对模拟的斜率，并附安慰剂一行（情景条件检验的估值账本）。",
    "倍数用 CAPE 倍数（CRSP 价格 /（CPI × 标普 10 年平均实际盈利））：滚动 P/E 被 2001–02、2008–09 的减记扭曲，"
    "股价下跌时它的“倍数”反而上升，只作显示。",
    "E 的最近两个季度可能是估计值（Shiller 工作簿没有标明）：今天的 P/E、E10 和盈利周期位置都依赖它。",
    "自由现金流收益率是全部非金融公司的代理（NIPA / Z.1），不是 SPY / QQQ 的自由现金流收益率；它与非金融公司利润收益率的分位相差很大，只作显示。",
    "纳指的估值只能用全市场 CAPE 近似：允许的数据源里没有纳斯达克 100 的市盈率、盈利或自由现金流；BusEq 行业市净率是唯一连续的科技估值，只作显示。",
    "起点估值对 1–20 个季度的回报没有样本外增益，任何“拉回均值”的设定都更差（12–20 个季度的部分每个十年只有 2–3 个独立窗口，只能当个案）："
    "估值是状态和初始条件，不是预测量；情景可以明说“估值重估”，那相当于直接假设股价。",
]


def _f(x, d: int = 4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _pct_rank(hist: np.ndarray, v: float) -> float | None:
    h = hist[np.isfinite(hist)]
    if not h.size or not math.isfinite(v):
        return None
    return float((h <= v).mean())


def _col(Q: pd.DataFrame, name: str) -> pd.Series:
    return Q[name].astype(float) if name in Q.columns else pd.Series(np.nan, index=Q.index)


def view(Q: pd.DataFrame, info: dict | None = None) -> dict:
    """The valuation view of the quarterly panel Q (index = quarter labels; the last row is today's start)."""
    Q = Q.copy()
    Q.index = [str(i) for i in Q.index]
    today = Q.index[-1]
    first = "1973Q1"
    H = Q.loc[first:]
    cape = np.exp(_col(H, "s_cape"))
    pd_ = np.exp(_col(H, "s_pd"))
    pe = np.exp(_col(H, "s_pe"))
    ry = _col(H, "s_real_y10")
    ecy = 100.0 / cape - ry
    fcf = _col(H, "s_fcfy")
    pb = np.exp(_col(H, "s_pb_buseq"))
    ecyc = _col(H, "s_ecyc")
    rows = []

    def state(key, zh, series, unit, use, fmt_d=1, extra=None):
        v = float(series.iloc[-1]) if len(series) else float("nan")
        hist = series.to_numpy(float)
        rows.append({"key": key, "zh": zh, "value": _f(v, 3), "unit": unit, "use": use,
                     "percentile": _f(_pct_rank(hist[:-1], v), 3), "since": first, "digits": fmt_d, **(extra or {})})

    state("cape", "CAPE（席勒周期调整市盈率）", cape, "倍", "初始条件；估值陈述的参照；隐含路径")
    state("ecy", "超额 CAPE 收益率（100/CAPE − 10 年实际收益率）", ecy, "%",
          "只显示：1950 年后唯一有正的 10 年样本外 R² 的口径，区间不可靠；这里的实际收益率是 10 年期收益率减 36 个月通胀趋势", 2)
    state("pd", "CRSP 市价/股息（P/D）", pd_, "倍", "收入腿；P/D 路径是输出（约 30% 的上升是派息率下降，不当作估值重估）", 1)
    state("pe", "滚动市盈率 P/Ê", pe, "倍", "只显示：被减记扭曲", 1)
    state("ecyc", "盈利周期位置 log(Ê/E10)", ecyc, "", "状态；E 的最近两个季度可能是估计值", 2)
    state("fcf", "自由现金流收益率（全部非金融公司代理）", fcf, "%",
          "只显示：全部非金融公司的代理，不是 SPY/QQQ 的自由现金流收益率", 2)
    pb_extra = {}
    if "2000Q2" in H.index:                    # the end of 2000Q1 (the bubble's peak) = the start of 2000Q2
        pb_extra["ref"] = {"quarter": "2000Q1 末", "value": _f(pb.loc["2000Q2"], 2)}
    state("pb_buseq", "BusEq（科技）行业市净率", pb, "倍", "只显示：唯一连续的科技估值；纳指的估值只能用全市场 CAPE 近似", 1,
          pb_extra)
    cape_hist = cape.to_numpy(float)[:-1]
    cape_hist = cape_hist[np.isfinite(cape_hist)]
    cape_q = [_f(x, 3) for x in np.percentile(cape_hist, np.arange(0, 101))] if cape_hist.size else []

    price = _col(Q, "y_us_equity")
    legs = {"mult": _col(Q, "y_mult"), "fund": _col(Q, "y_fund"), "pd": _col(Q, "y_pd"),
            "pe": price - _col(Q, "y_eps")}
    windows = []
    for a, b, zh in WINDOWS:
        if a not in Q.index or b not in Q.index:
            continue
        sl = slice(a, b)
        windows.append({"from": a, "to": b, "zh": zh, "n": int(price.loc[sl].notna().sum()),
                        "price": _f(price.loc[sl].sum()), **{k: _f(v.loc[sl].sum()) for k, v in legs.items()}})
    ok = price.notna()
    for v in legs.values():
        ok &= v.notna()
    P = price[ok]
    all_q = {"n": int(ok.sum()), "from": str(P.index[0]) if len(P) else None, "to": str(P.index[-1]) if len(P) else None}
    for k, v in legs.items():
        x = v[ok]
        all_q[k] = {"corr": _f(np.corrcoef(x, P)[0, 1], 3) if len(P) > 2 else None,
                    "share": _f(np.cov(x, P)[0, 1] / np.var(P, ddof=1), 3) if len(P) > 2 else None}
    ident = price - legs["mult"] - legs["fund"]
    ymult = _col(Q, "y_mult")
    last = str(ymult.dropna().index[-1]) if ymult.notna().any() else None
    rerate = []
    for end in (*RERATING_ENDS, last):
        if end is None or end not in Q.index:
            continue
        m = ymult.loc["1973Q2":end].dropna()      # the multiple's level from the end of 1973Q1 to the end (d05)
        rerate.append({"from": f"{first} 末", "to": end, "per_year": _f(4.0 * m.mean(), 4), "n": int(len(m))})
    return {"today": {"quarter": today, "data_end_month": (info or {}).get("data_end_month"), "states": rows},
            "cape_q": cape_q, "cape_since": first,
            "identity": {"windows": windows, "all": all_q,
                         "max_abs_error": _f(float(np.nanmax(np.abs(ident.to_numpy(float)))) if ident.notna().any()
                                             else None, 12),
                         "text": "log P = log 倍数 + log CPI + log 10 年平均实际盈利：价格回报 = 倍数腿 + 基本面腿（恒等式）"},
            "rerating": rerate, "notes": NOTES,
            "label": "季度模型的估值口径（试运行）"}
