"""d05 -- (a) unemployment anchor with a 20-year trailing mean; (b) size bins for stated valuation re-ratings: quantiles
of |cumulative dlog CAPE-multiple| over spans of 4/8/12/20 quarters, 1973Q2..2014Q4 and full sample; (c) quarterly sd
of the CAPE multiple change. Read-only. Run: <venv python> -I -B d05_sizes_anchors.py <qpanel.csv> <valuation_quarterly.csv>
"""
import sys, json
import numpy as np, pandas as pd
Q = pd.read_csv(sys.argv[1]).set_index("quarter").loc[:"2026Q3"]
V = pd.read_csv(sys.argv[2]).set_index("quarter").loc[:"2026Q3"]
out = {}
u = Q["s_unrate"].shift(-1)
out["unrate_trailing_80q"] = {e: round(float(u.loc[:e].iloc[-80:].mean()), 2) for e in ["1994Q4", "2004Q4", "2014Q4", "2026Q2"]}
dm = V["us_equity_lm_cape"].diff().loc["1973Q2":]
out["sd_q_dm_cape_1973_2014"] = round(float(dm.loc[:"2014Q4"].std()), 3)
out["sd_q_dm_cape_full"] = round(float(dm.std()), 3)
for per, sl in [("1973_2014", slice("1973Q2", "2014Q4")), ("full", slice("1973Q2", "2026Q3"))]:
    d = dm.loc[sl]
    res = {}
    for h in [4, 8, 12, 20]:
        c = d.rolling(h).sum().dropna().abs()
        res[h] = {"p50": round(float(c.quantile(0.5)), 3), "p75": round(float(c.quantile(0.75)), 3),
                  "p95": round(float(c.quantile(0.95)), 3), "max": round(float(c.max()), 3), "n_overlap": int(len(c)),
                  "n_indep": int(len(d) // h)}
    out[f"abs_cum_dm_cape_{per}"] = res
# today's log CAPE multiple percentile (1948+) is in the valuation report; here: trailing 40q drift of the multiple
for e in ["1994Q4", "2004Q4", "2014Q4", "2026Q3"]:
    s = V["us_equity_lm_cape"].loc["1973Q1":e]
    out.setdefault("mult_drift_pct_per_year_since_1973", {})[e] = round(float((s.iloc[-1] - s.iloc[0]) / ((len(s) - 1) / 4) * 100), 2)
print(json.dumps(out, indent=1))
