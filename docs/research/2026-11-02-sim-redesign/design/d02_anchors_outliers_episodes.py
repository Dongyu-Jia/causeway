"""d02 -- (a) train-end-knowable anchors vs DEFAULT_REVERSION, (b) a generic (undated) outlier rule on quarterly core
factor changes, (c) candidate narrative-test episodes: realized us_equity path and training length.
Read-only: reads the qmodel quarterly panel only. Run: cd MAIN && .venv/bin/python -I -B d02_....py <qpanel.csv>
"""
import sys, json
import numpy as np, pandas as pd
Q = pd.read_csv(sys.argv[1]).set_index("quarter").loc[:"2026Q3"]
out = {}
# (a) anchors: trailing 40-quarter means known at the train end (states are start-of-quarter values, so the row after
# the train end holds the train-end-known level; use the train-end quarter's own end-of-quarter level = next row's s_)
def trailing(col, end, n=40):
    idx = list(Q.index)
    j = idx.index(end)
    nxt = idx[j + 1] if j + 1 < len(idx) else None
    s = Q[col].shift(-1)  # level at the END of each quarter (= next quarter's start state)
    w = s.loc[:end].iloc[-n:]
    return round(float(w.mean()), 2)
anch = {}
for end in ["1994Q4", "2004Q4", "2014Q4", "2026Q2"]:
    rff = (Q["s_ff"] - Q["s_core_yoy"]).shift(-1).loc[:end].iloc[-40:].mean()
    anch[end] = {"real_ff_10y": round(float(rff), 2), "term_10y": trailing("s_term", end),
                 "spread_10y": trailing("s_spread", end), "unrate_10y": trailing("s_unrate", end),
                 "core_10y": trailing("s_core_yoy", end)}
out["anchors_trailing_10y"] = anch
out["DEFAULT_REVERSION"] = {"r_star": 0.75, "term_premium": 0.75, "spread_level": 2.0, "unrate_level": 4.8}
# (b) generic outlier rule: Mahalanobis norm of the standardized core change vector, iterated robust covariance
core = ["d_ff", "d_y10", "d_spread", "d_gz", "d_usd", "d_unrate", "d_ip", "d_cpi_goods", "d_cpi_serv", "d_oil"]
X = Q[core].dropna()
keep = np.ones(len(X), bool)
from math import isfinite
thr = 25.19   # chi2(10) 0.995 quantile
for it in range(20):
    mu = X[keep].mean().values; S = np.cov(X[keep].values.T)
    Si = np.linalg.inv(S)
    D = X.values - mu
    m2 = np.einsum("ij,jk,ik->i", D, Si, D)
    new = m2 <= thr
    if (new == keep).all():
        break
    keep = new
flag = X.index[~keep]
out["outlier_rule"] = {"threshold_chi2_10_995": thr, "iterations": it + 1, "n_quarters": int(len(X)),
                       "flagged": [(q, round(float(v), 1)) for q, v in zip(X.index, m2) if v > thr]}
# (c) episodes
eps = {"E06": ("1973Q4", "1975Q2"), "E03": ("1979Q4", "1982Q2"), "E11": ("1990Q3", "1991Q4"), "E07": ("1994Q1", "1995Q2"),
       "E02": ("1995Q3", "1997Q4"), "E12": ("2001Q1", "2002Q4"), "E01": ("2004Q3", "2006Q4"), "E05": ("2007Q3", "2009Q2"),
       "E04": ("2014Q3", "2015Q4"), "E09": ("2017Q1", "2019Q4"), "E08": ("2020Q1", "2021Q2"), "E10": ("2022Q1", "2023Q4"),
       "N13": ("1987Q3", "1988Q4"), "N14": ("1998Q1", "2000Q4"), "N15": ("2010Q2", "2012Q4"), "N16": ("1977Q1", "1979Q3"),
       "N17": ("1983Q1", "1984Q4"), "N18": ("2024Q1", "2026Q2")}
idx = list(Q.index)
rows = {}
for k, (a, b) in eps.items():
    y = Q.loc[a:b, "y_us_equity"]
    c = y.cumsum()
    dd = float((c - np.maximum.accumulate(np.r_[0.0, c.values][:-1].clip(min=None)).clip(min=0)).min()) if len(c) else None
    peak = np.maximum.accumulate(np.r_[0.0, c.values])
    trough = float(np.min(np.r_[0.0, c.values] - peak))
    rows[k] = {"window": f"{a}..{b}", "n_q": int(len(y)), "train_q": idx.index(a), "useq_window": round(float(c.iloc[-1]), 3),
               "useq_maxdd": round(trough, 3), "nasdaq_window": round(float(Q.loc[a:b, "y_nasdaq"].sum()), 3),
               "dgz_sum": round(float(Q.loc[a:b, "d_gz"].sum()), 2)}
out["episodes"] = rows
# overlap check
spans = sorted((idx.index(a), idx.index(b), k) for k, (a, b) in eps.items())
ov = [(spans[i][2], spans[i + 1][2]) for i in range(len(spans) - 1) if spans[i + 1][0] <= spans[i][1]]
out["overlaps"] = ov
print(json.dumps(out, indent=1, ensure_ascii=False))
