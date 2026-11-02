"""d01 -- history statistics used as acceptance thresholds in SIM_DESIGN (read-only; reads the qmodel and valuation
quarterly panels only). Run: cd MAIN && .venv/bin/python -I -B d01_history_stats.py <qpanel.csv> <valuation_quarterly.csv>
"""
import sys, json
import numpy as np, pandas as pd
Q = pd.read_csv(sys.argv[1]).set_index("quarter")
V = pd.read_csv(sys.argv[2]).set_index("quarter")
Q = Q.loc[:"2026Q3"]
out = {}
def corr(a, b, rows=None):
    d = pd.concat([Q[a], Q[b]], axis=1).dropna()
    if rows is not None: d = d.loc[rows[0]:rows[1]]
    return round(float(d.corr().iloc[0, 1]), 3), int(len(d))
for per in [("1973Q1", "2026Q3"), ("1973Q1", "2019Q4"), ("1985Q1", "2026Q3"), ("2015Q1", "2026Q3")]:
    out[f"corr_dgz_useq_{per[0]}_{per[1]}"] = corr("d_gz", "y_us_equity", per)
    out[f"corr_dspread_useq_{per[0]}_{per[1]}"] = corr("d_spread", "y_us_equity", per)
    out[f"corr_dy10_useq_{per[0]}_{per[1]}"] = corr("d_y10", "y_us_equity", per)
    out[f"corr_useq_tsy_{per[0]}_{per[1]}"] = corr("y_us_equity", "y_treasury_10y", per)
# multiple legs: y_mult (P/E-hat) vs CAPE multiple on CRSP price vs P/D
lm = V["us_equity_lm_cape"].loc[:"2026Q3"]
lpd = V["us_equity_lpd"].loc[:"2026Q3"]
dm_cape = lm.diff()
dm_pd = lpd.diff()
y = Q["y_us_equity"]
d = pd.concat([y, dm_cape.rename("dm_cape"), dm_pd.rename("dm_pd"), Q["y_mult"]], axis=1).dropna()
d = d.loc["1973Q2":]
out["n_q_mult"] = int(len(d))
for c in ["dm_cape", "dm_pd", "y_mult"]:
    out[f"corr_price_{c}"] = round(float(d[["y_us_equity", c]].corr().iloc[0, 1]), 3)
    out[f"share_{c}"] = round(float(np.cov(d["y_us_equity"], d[c])[0, 1] / d["y_us_equity"].var()), 3)
# episode check: 2007Q3..2009Q2 and 2001Q1..2002Q4 sums
for a, b in [("2007Q3", "2009Q2"), ("2001Q1", "2002Q4"), ("1995Q3", "2000Q1"), ("2022Q1", "2022Q3")]:
    s = d.loc[a:b].sum()
    out[f"legs_{a}_{b}"] = {k: round(float(s[k]), 3) for k in ["y_us_equity", "dm_cape", "dm_pd", "y_mult"]}
# unemployment floor since 1973 and CPI>4 share
u = Q["s_unrate"].dropna()
out["unrate_min_since1973"] = round(float(u.min()), 2)
cy = Q["s_cpi_yoy"].dropna()
out["share_cpi_yoy_gt4"] = round(float((cy > 4).mean()), 3)
# GZ alone: quarterly std of d_gz, autocorr
out["sd_dgz"] = round(float(Q["d_gz"].std()), 3)
out["ac1_dgz"] = round(float(Q["d_gz"].autocorr(1)), 3)
# share of equity quarters with |return| > 15%
print(json.dumps(out, indent=1, ensure_ascii=False))
