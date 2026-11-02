"""Historical era fingerprints and how today compares with them.

For every era in config.ERAS: state levels, factor drift and volatility, asset
returns (nominal and CPI-deflated), cross-asset correlations, the stock-bond
correlation, the oil -> CPI pass-through, the Phillips-curve slope, a
signature (the states furthest from their 1955+ average, in sd units) and an
auto-generated narrative.

today_similarity describes the last 12 months as an entropy-pooling mixture of
historical months: 1955 up to 24 months ago and before the ongoing era starts
(today is inside that era by definition, so matching against its own months
says nothing about history). Today sits at the edge of history on
some states (record profit share, record-low dividend yield), where hard mean
views are jointly infeasible and the EP dual collapses onto a single month. So
the views are solved softly: minimise KL(q||p) + |E_q[z] - z_today|^2 / (2 tau),
with tau the smallest value that keeps at least ENS_MIN effective months. The
resulting posterior means are then passed to engine.ep.apply_views as ordinary
mean views, which reproduces the same weights and the engine's own report. The
per-view gaps that remain are what history cannot reproduce.

eras[].attribution splits each era's realized average annual price return
(mean monthly price return x 12) per asset into the transmission the engine
uses (the selected model, fitted on 1973 onward) applied to the era's actual
factor changes, one term per factor, plus the intercept and the residual,
which is what that transmission does not explain in the era; income makes it
the total return, and `reconciliation` takes it to the era's CAGR. The era's
own regression (its own betas) is reported next to it where the era is long
enough. Eras without every factor (before 1973 there is no trade-weighted
dollar) get no attribution, with the reason. A model with per-asset equations
(asset_factors, asset_start) splits each asset along its own equation: an extra
regressor of another asset (energy's metals-vs-oil factor) is no term of it, and
an asset fitted from a later month (gold from 1982) carries those betas into the
earlier eras too, as the scenario engine does.

The scenario view variables of config.STATES (the ones with a `concept`) are
reported in every era's state levels but stay out of the signatures, which keep
the long-standing macro states (the view variables have shorter or overlapping
histories and would crowd them out).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .. import config, jsonio
from ..engine.ep import apply_views, effective_number, kl_divergence
from ..engine.factor_model import AssetFactorModel, ModelSpec, asset_feature_names, design_matrix
from ..engine.simulate import attribution_method, era_of

FULL_START = "1955-01"           # reference window for z-scores and the similarity database
MIN_COVERAGE = 0.5               # share of an era's months a series needs to be reported
PASSTHROUGH_LAGS = 6
# Before 1973 WTI was an administered posted price that changed a handful of
# times per era; a lag regression on so few moves is identified by one or two
# events (Feb 1957: 0.13 from the Suez move alone), so it is not reported.
MIN_OIL_MOVES = 24
SIGNATURE_SIZE = 5

FINGERPRINT_STATES = [
    "cpi_yoy", "core_yoy", "ip_yoy", "unrate", "real_ff", "y10", "spread", "real_oil",
    "labor_share", "profit_share", "capex_share", "wealth_income", "white_collar_share",
    "real_wage_yoy", "productivity_yoy", "div_yield", "saving_rate", "sentiment",
]
SIMILARITY_STATES = ["cpi_yoy", "ip_yoy", "unrate", "real_ff", "y10", "spread", "oil_yoy",
                     "profit_share", "div_yield"]
TODAY_MONTHS = 12
EXCLUDE_RECENT = 24
ENS_MIN = 60.0
ATTR_MIN_COVERAGE = 0.5          # share of an era's months that must have every feature of the transmission
OWN_BETA_MONTHS_PER_PARAM = 5    # the era's own regression needs at least 5 months per coefficient
ATTR_UNITS = "年化：每月算术回报的均值 ×12（小数）；price_mean = 各因子项 + 常数项 + 未解释部分，再加收入 = total_mean"

STATE_INFO = {s["id"]: s for s in config.STATES}
SIGNATURE_STATES = {s["id"] for s in config.STATES if not s.get("concept")}   # view variables stay out
ERA_NAMES = {e["id"]: e["name"] for e in config.ERAS}
ASSET_NAMES = {a["id"]: a["name"] for a in config.ASSETS}
FACTOR_NAMES = {f["id"]: f["name"] for f in config.FACTORS}
UNIT_TEXT = {"%": "%", "pp": "pp", "$": " 美元", "2017=100": "", "倍": " 倍", "指数": ""}
PHILLIPS_LABEL = {"wage_yoy": "工资增速", "core_yoy": "核心通胀", "cpi_yoy": "通胀"}

# Highly correlated states share a family; a signature keeps one per family so
# four inflation measures do not crowd out everything else.
FAMILY = {"cpi_yoy": "inflation", "core_yoy": "inflation", "goods_yoy": "inflation",
          "services_yoy": "inflation", "infl_expect": "inflation", "ff": "policy", "real_ff": "policy",
          "y10": "rates", "term_spread": "rates", "real_y10": "rates", "oil_yoy": "oil", "real_oil": "oil",
          # the release-dated profit share is the same series as profit_share; CAPE and the dividend yield
          # are both the market multiple
          "profit_share": "profits", "profit_share_rt": "profits", "div_yield": "valuation", "cape": "valuation"}

# (text when above the long-run mean, text when below); {d} is the degree word.
STATE_TEXT = {
    "cpi_yoy": ("通胀{d}偏高", "通胀{d}偏低"),
    "core_yoy": ("核心通胀{d}偏高", "核心通胀{d}偏低"),
    "goods_yoy": ("商品通胀{d}偏高", "商品通胀{d}偏低"),
    "services_yoy": ("服务通胀{d}偏高", "服务通胀{d}偏低"),
    "ip_yoy": ("工业增长{d}偏强", "工业增长{d}偏弱"),
    "unrate": ("失业率{d}偏高、劳动力市场松弛", "失业率{d}偏低、劳动力市场紧张"),
    "ff": ("政策利率{d}偏高", "政策利率{d}偏低"),
    "real_ff": ("实际政策利率{d}偏高（货币偏紧）", "实际政策利率{d}偏低（货币偏松）"),
    "y10": ("长期利率{d}偏高", "长期利率{d}偏低"),
    "term_spread": ("收益率曲线{d}偏陡", "收益率曲线{d}偏平"),
    "spread": ("信用利差{d}偏宽（风险溢价高）", "信用利差{d}偏窄（风险偏好强）"),
    "oil_yoy": ("油价涨幅{d}偏大", "油价{d}偏弱"),
    "real_oil": ("实际油价{d}偏高", "实际油价{d}偏低"),
    "usd_yoy": ("美元{d}走强", "美元{d}走弱"),
    "labor_share": ("劳动收入占比{d}偏高", "劳动收入占比{d}偏低（收入向资本倾斜）"),
    "profit_share": ("企业利润占 GDP {d}偏高", "企业利润占 GDP {d}偏低"),
    "capex_share": ("企业投资占 GDP {d}偏高", "企业投资占 GDP {d}偏低"),
    "wealth_income": ("家庭净值/收入{d}偏高（资产价格高）", "家庭净值/收入{d}偏低"),
    "white_collar_share": ("白领行业就业占比{d}偏高", "白领行业就业占比{d}偏低"),
    "real_wage_yoy": ("普通工人实际工资增长{d}偏快", "普通工人实际工资增长{d}偏慢"),
    "productivity_yoy": ("劳动生产率增长{d}偏快", "劳动生产率增长{d}偏慢"),
    "sentiment": ("消费者信心{d}偏强", "消费者信心{d}偏弱"),
    "div_yield": ("股息率{d}偏高（估值低）", "股息率{d}偏低（估值高）"),
    "saving_rate": ("储蓄率{d}偏高", "储蓄率{d}偏低"),
    "infl_expect": ("通胀预期代理{d}偏高", "通胀预期代理{d}偏低"),
    "real_y10": ("10 年期实际利率{d}偏高", "10 年期实际利率{d}偏低"),
    "profit_share_rt": ("企业利润占 GDP（按发布月）{d}偏高", "企业利润占 GDP（按发布月）{d}偏低"),
    "profit_growth_yoy": ("企业利润增长{d}偏快", "企业利润增长{d}偏慢"),
    "fcf_share": ("利润减资本开支/GDP（自由现金流代理）{d}偏高", "利润减资本开支/GDP（自由现金流代理）{d}偏低"),
    "cape": ("席勒 CAPE {d}偏高（估值高）", "席勒 CAPE {d}偏低（估值低）"),
}


# ---------------------------------------------------------------- helpers

def _ym(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def _fmt_state(sid: str, value: float) -> str:
    unit = UNIT_TEXT.get(STATE_INFO[sid]["unit"], "")
    return f"{value:.1f}{unit}"


def _pct(x: float, signed: bool = True) -> str:
    if signed and round(x * 100, 1) == 0.0:          # "0.0%", not "+0.0%" / "-0.0%"
        return "0.0%"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _degree(z: float) -> str:
    a = abs(z)
    return "显著" if a >= 1.5 else ("明显" if a >= 0.75 else "略")


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their classical covariance (X includes the constant)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    return beta, XtX_inv * float(resid @ resid) / dof


def era_bounds(era: dict, panel: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(era["end"]) if era["end"] else panel.index.max()
    return pd.Timestamp(era["start"]), min(end, panel.index.max())


def reference_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd of every state over FULL_START..end (rows: state ids)."""
    cols = [f"s_{s['id']}" for s in config.STATES if f"s_{s['id']}" in panel]
    X = panel.loc[FULL_START:, cols]
    out = pd.DataFrame({"mean": X.mean(), "sd": X.std()})
    out.index = [c[2:] for c in out.index]
    return out


# ---------------------------------------------------------------- per-era blocks

def state_block(P: pd.DataFrame) -> dict:
    out = {}
    for s in config.STATES:
        col = f"s_{s['id']}"
        if col not in P:
            continue
        x = P[col].dropna()
        coverage = len(x) / max(len(P), 1)
        if coverage < MIN_COVERAGE:
            continue
        out[s["id"]] = {"mean": float(x.mean()), "start": float(x.iloc[0]), "end": float(x.iloc[-1]),
                        "coverage": round(coverage, 3)}
    return out


def factor_block(P: pd.DataFrame) -> dict:
    """Annualized mean (x12) and volatility (x sqrt 12) of monthly factor changes, in factor units."""
    out = {}
    for fid in [f for f in config.FACTOR_IDS if f not in config.AUDIT_FACTOR_IDS] + ["cpi"]:
        col = f"f_{fid}"
        if col not in P:
            continue
        x = P[col].dropna()
        if len(x) < max(12, MIN_COVERAGE * len(P)):
            continue
        out[fid] = {"mean_annual": float(x.mean() * 12.0), "vol_annual": float(x.std() * np.sqrt(12.0))}
    return out


def asset_block(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Return statistics for assets with total-return data in every month of the era.

    cagr / real_cagr: geometric annual (decimal), real = deflated by the CPI
    index (lvl_cpi) from the month before the era to its last month.
    vol: annualized sd of monthly total returns. max_dd: worst peak-to-trough
    of the nominal total-return index. avg_income: mean monthly income x 12.
    """
    P = panel.loc[start:end]
    n = len(P)
    prev = start - pd.DateOffset(months=1)
    cpi0 = panel["lvl_cpi"].get(prev, np.nan)
    if not np.isfinite(cpi0):
        cpi0 = float(P["lvl_cpi"].iloc[0])
    cpi_ratio = float(P["lvl_cpi"].iloc[-1]) / float(cpi0)
    out = {}
    for a in config.ASSET_IDS:
        r = P[f"r_{a}"]
        if len(r) == 0 or r.isna().any():
            continue
        w = np.cumprod(1.0 + r.to_numpy(float))
        peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
        out[a] = {
            "cagr": float(w[-1] ** (12.0 / n) - 1.0),
            "real_cagr": float((w[-1] / cpi_ratio) ** (12.0 / n) - 1.0),
            "vol": float(r.std() * np.sqrt(12.0)),
            "max_dd": float(min((w / peak - 1.0).min(), 0.0)),
            "avg_income": float(P[f"d_{a}"].mean() * 12.0),
        }
    return out


def corr_block(P: pd.DataFrame, assets: list[str]) -> dict:
    ids = [a for a in assets if a != "cash"]
    if not ids:
        return {"assets": [], "matrix": []}
    c = P[[f"r_{a}" for a in ids]].corr().to_numpy()
    return {"assets": ids, "matrix": [[round(float(v), 3) for v in row] for row in c]}


def oil_price_moves(panel: pd.DataFrame, start, end) -> int:
    """Months in [start, end] with a non-zero oil-price change."""
    return int((panel["f_oil"].loc[start:end].abs() > 1e-9).sum())


def oil_cpi_passthrough(panel: pd.DataFrame, start, end, lags: int = PASSTHROUGH_LAGS) -> tuple[float, float]:
    """Sum of the f_oil lag 0..lags coefficients in an OLS of monthly f_cpi (both log changes).

    0.04 means a 10% oil-price rise lifts the CPI level by about 0.4% within
    `lags` months. Lags may reach before the era start (they are observed).
    Returns (sum, its OLS standard error); NaN when the oil price moved in
    fewer than MIN_OIL_MOVES months of the window.
    """
    if oil_price_moves(panel, start, end) < MIN_OIL_MOVES:
        return float("nan"), float("nan")
    X = pd.concat({f"l{k}": panel["f_oil"].shift(k) for k in range(lags + 1)}, axis=1)
    D = pd.concat([panel["f_cpi"].rename("y"), X], axis=1).loc[start:end].dropna()
    if len(D) < lags + 12:
        return float("nan"), float("nan")
    Z = np.column_stack([np.ones(len(D)), D.drop(columns="y").to_numpy(float)])
    beta, cov = _ols(D["y"].to_numpy(float), Z)
    w = np.r_[0.0, np.ones(lags + 1)]
    return float(w @ beta), float(np.sqrt(w @ cov @ w))


def phillips_slope(panel: pd.DataFrame, start, end) -> tuple[float, str]:
    """OLS slope of 12-month nominal wage growth (%) on the unemployment rate (%).

    Falls back to core CPI YoY (then headline) when the wage series covers too
    little of the era (it starts in 1964).
    """
    P = panel.loc[start:end]
    wage_yoy = (100.0 * (panel["lvl_wage"] / panel["lvl_wage"].shift(12) - 1.0)).loc[start:end]
    for name, y in (("wage_yoy", wage_yoy), ("core_yoy", P["s_core_yoy"]), ("cpi_yoy", P["s_cpi_yoy"])):
        D = pd.concat([y.rename("y"), P["s_unrate"].rename("u")], axis=1).dropna()
        if len(D) >= max(24, MIN_COVERAGE * len(P)):
            Z = np.column_stack([np.ones(len(D)), D["u"].to_numpy(float)])
            beta, _ = _ols(D["y"].to_numpy(float), Z)
            return float(beta[1]), name
    return float("nan"), "none"


def signature(states: dict, ref: pd.DataFrame, k: int = SIGNATURE_SIZE) -> list[dict]:
    rows = []
    for sid, st in states.items():
        if sid not in SIGNATURE_STATES or sid not in ref.index or not ref.loc[sid, "sd"] > 0:
            continue
        z = (st["mean"] - ref.loc[sid, "mean"]) / ref.loc[sid, "sd"]
        rows.append((abs(z), sid, z))
    # an exact tie keeps the shorter id: the release-dated profit share is now the same series as profit_share
    rows.sort(key=lambda r: (-r[0], r[1]))
    out, families = [], set()
    for _, sid, z in rows:
        fam = FAMILY.get(sid, sid)
        if fam in families:
            continue
        families.add(fam)
        name = STATE_INFO[sid]["name"]
        sep = " " if name[-1:].isascii() and name[-1:].isalnum() else ""        # "CAPE 明显偏高", not "CAPE明显偏高"
        hi, lo = STATE_TEXT.get(sid, (name + sep + "{d}偏高", name + sep + "{d}偏低"))
        phrase = (hi if z > 0 else lo).format(d=_degree(z))
        text = (f"{phrase}，均值 {_fmt_state(sid, states[sid]['mean'])}"
                f"（1955 年以来 {_fmt_state(sid, ref.loc[sid, 'mean'])}）")
        out.append({"state": sid, "name": STATE_INFO[sid]["name"], "z": float(z), "text": text})
        if len(out) == k:
            break
    return out


def narrative(era: dict, rec: dict) -> str:
    parts = [f"{era['name']}（{rec['start']} 至 {rec['end']}，{rec['n_months']} 个月）：{era['summary']}"]
    if rec["signature"]:
        parts.append("数据指纹：" + "；".join(s["text"] for s in rec["signature"][:3]) + "。")
    assets = rec["assets"]
    ranked = sorted(((v["real_cagr"], a) for a, v in assets.items() if a != "cash"), reverse=True)
    if ranked:
        best, worst = ranked[0], ranked[-1]
        parts.append(f"实际年化回报最好的是{ASSET_NAMES[best[1]]}（{_pct(best[0])}），"
                     f"最差的是{ASSET_NAMES[worst[1]]}（{_pct(worst[0])}）。")
    eq = assets.get("us_equity")
    if eq:
        parts.append(f"美股名义年化 {_pct(eq['cagr'])}、扣除通胀后 {_pct(eq['real_cagr'])}，"
                     f"期间最大回撤 {_pct(eq['max_dd'], signed=False)}。")
    if "cash" in assets:
        parts.append(f"现金实际年化 {_pct(assets['cash']['real_cagr'])}。")
    sbc = rec.get("stock_bond_corr")
    if sbc is not None and np.isfinite(sbc):
        kind = "同涨同跌，债券难以对冲股票" if sbc > 0.1 else ("反向波动，债券能对冲股票" if sbc < -0.1 else "基本不相关")
        parts.append(f"股债月度相关性 {sbc:+.2f}（{kind}）。")
    pt, se = rec.get("oil_cpi_passthrough"), rec.get("oil_cpi_passthrough_se")
    if pt is not None and np.isfinite(pt):
        noisy = "，估计误差较大" if se is not None and se > abs(pt) / 2 else ""
        parts.append(f"油价对 CPI 的 6 个月累计传导 {pt:.3f}（油价涨 10%，CPI 约多涨 {pt * 10:.2f}%{noisy}）。")
    elif rec.get("oil_price_moves", MIN_OIL_MOVES) < MIN_OIL_MOVES:
        moves = rec["oil_price_moves"]
        why = (f"油价是管制挂牌价，期间只变动过 {moves} 个月" if moves < 0.5 * rec["n_months"]
               else f"油价有变动的月份太少（{moves} 个月）")
        parts.append(f"{why}，无法估计油价对 CPI 的传导。")
    ps = rec.get("phillips_slope")
    if ps is not None and np.isfinite(ps):
        label = PHILLIPS_LABEL.get(rec.get("phillips_measure"), "通胀")
        parts.append(f"菲利普斯曲线斜率 {ps:+.2f}（失业率每高 1 个百分点，{label}约{'低' if ps < 0 else '高'} "
                     f"{abs(ps):.2f} 个百分点）。")
    return "".join(parts)


def era_record(panel: pd.DataFrame, era: dict, ref: pd.DataFrame) -> dict:
    start, end = era_bounds(era, panel)
    P = panel.loc[start:end]
    assets = asset_block(panel, start, end)
    sbc = float(P["r_us_equity"].corr(P["r_treasury_10y"])) if {"us_equity", "treasury_10y"} <= assets.keys() else None
    pt, pt_se = oil_cpi_passthrough(panel, start, end)
    slope, measure = phillips_slope(panel, start, end)
    states = state_block(P)
    rec = {
        "id": era["id"], "name": era["name"], "start": _ym(start), "end": _ym(end),
        "ongoing": era["end"] is None, "summary": era["summary"], "n_months": int(len(P)),
        "states": states, "factors": factor_block(P), "assets": assets,
        "corr": corr_block(P, list(assets)),
        "stock_bond_corr": sbc, "oil_cpi_passthrough": pt, "oil_cpi_passthrough_se": pt_se,
        "oil_price_moves": oil_price_moves(panel, start, end),
        "phillips_slope": slope, "phillips_measure": measure,
        "signature": signature(states, ref),
    }
    rec["narrative"] = narrative(era, rec)
    return rec


# ---------------------------------------------------------------- transmission attribution

def attribution_model(ctx) -> AssetFactorModel:
    """The transmission the scenarios run (results/model_selection.json, else the base spec), fitted on
    config.MODEL_START onward: ctx.model when it is that model, otherwise fitted here (the build hands this
    step the base-spec context, the tests a bare panel)."""
    from ..context import selected_spec

    panel = ctx.panel
    spec = selected_spec() or ModelSpec()
    if any(c not in panel.columns for c in spec.shock_cols):        # as EngineContext.load drops them
        spec = ModelSpec.from_dict({**spec.to_dict(), "shock_cols": []})
    model = getattr(ctx, "model", None)
    if isinstance(model, AssetFactorModel) and model.spec == spec and model.train_start is not None:
        return model
    return AssetFactorModel(spec).fit(panel, config.MODEL_START, None)


def _own_beta(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict | None:
    """The era's own OLS of monthly price returns on its factor changes: each factor's beta x its mean
    change (x 12), the intercept (x 12; the in-sample residual has mean 0) and R^2. None when the era has
    fewer than OWN_BETA_MONTHS_PER_PARAM months per coefficient."""
    n, k = X.shape
    if n < OWN_BETA_MONTHS_PER_PARAM * (k + 1):
        return None
    Z = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    e = y - Z @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    terms = {f: float(12.0 * beta[i + 1] * X[:, i].mean()) for i, f in enumerate(names)}
    return {"terms": terms, "factor_sum": float(sum(terms.values())), "intercept": float(12.0 * beta[0]),
            "r2": float(1.0 - float(e @ e) / sst) if sst > 0 else None, "n_months": int(n)}


def era_attribution(panel: pd.DataFrame, model: AssetFactorModel, X: pd.DataFrame, fitted: pd.DataFrame,
                    start, end) -> dict:
    """Each asset's average annual price return in [start, end] split along the transmission.

    X: the model's features on the whole panel (design_matrix), fitted: model.predict(panel). Over the era
    months that have every feature, mean(p) = sum_k beta_k mean(x_k) + intercept + mean(residual) exactly
    (the residual is p minus the fitted value); with income it is the mean total return, and
    reconciliation = CAGR over those months - that arithmetic mean (compounding and volatility drag).
    The terms k of an asset are the features of its own equation (factor_model.asset_feature_names);
    the era's own regression (own_beta) uses the same features.
    """
    method, why = attribution_method(model.spec)
    P = panel.loc[start:end]
    Xe = X.loc[start:end]
    feat_ok = Xe.notna().all(axis=1)
    covered = Xe.index[feat_ok]
    out = {"method": method, "n_months": int(feat_ok.sum()),
           "coverage": round(float(feat_ok.mean()), 3) if len(P) else 0.0,
           "months": [_ym(covered[0]), _ym(covered[-1])] if len(covered) else None, "units": ATTR_UNITS,
           "assets": {}}
    if method != "linear":
        out["reason"] = why
    if not len(P) or feat_ok.mean() < ATTR_MIN_COVERAGE:
        missing = [c for c in Xe.columns if Xe[c].isna().mean() > 1.0 - ATTR_MIN_COVERAGE]
        names = "、".join(FACTOR_NAMES.get(c, c) for c in missing) or "部分因子"
        out["missing"] = missing
        out["reason"] = (f"大部分月份缺少传导模型的因子（{names}）"
                         + ("，贸易加权美元指数 1973 年才开始；传导模型也只用 1973 年以后的数据估计"
                            if "usd" in missing else ""))
        return out
    linear = method == "linear"
    for a in list(model.assets) + ["cash"]:
        p, d, r = (P.get(f"{k}_{a}") for k in ("p", "d", "r"))
        if p is None or d is None or r is None:
            continue
        ok = feat_ok & p.notna() & d.notna() & r.notna()
        if ok.sum() < max(12, ATTR_MIN_COVERAGE * len(P)):
            continue
        pe, de, re_ = (s[ok].to_numpy(float) for s in (p, d, r))
        n = len(pe)
        price_mean, total_mean = 12.0 * float(pe.mean()), 12.0 * float(re_.mean())
        cagr = float(np.prod(1.0 + re_) ** (12.0 / n) - 1.0)
        if a == "cash" or a not in model.assets:
            rec = {"kind": "cash", "terms": {"income": 12.0 * float(de.mean())}}
        else:
            feats = asset_feature_names(model.spec, a) if linear else []
            fit = fitted.loc[ok[ok].index, a].to_numpy(float)
            resid = pe - fit
            if linear:
                xm = Xe.loc[ok, feats].to_numpy(float).mean(axis=0)
                terms = {f: 12.0 * float(model.coef.loc[a, f]) * float(xm[i]) for i, f in enumerate(feats)}
                terms["intercept"] = 12.0 * float(model.intercept[a])
            else:
                terms = {"unattributed": 12.0 * float(fit.mean())}
            terms["residual"] = 12.0 * float(resid.mean())
            terms["income"] = 12.0 * float(de.mean())
            sst = float(((pe - pe.mean()) ** 2).sum())
            rec = {"kind": "model", "terms": terms,
                   "factor_sum": float(sum(terms[f] for f in feats)) if linear else None,
                   "r2_in_era": float(1.0 - float(resid @ resid) / sst) if sst > 0 else None,
                   "own_beta": _own_beta(pe, Xe.loc[ok, feats].to_numpy(float), feats) if linear else None}
        rec.update({"price_mean": price_mean, "total_mean": total_mean, "cagr": cagr,
                    "reconciliation": cagr - total_mean, "n_months": int(n)})
        out["assets"][a] = rec
    return out


def attribution_info(model: AssetFactorModel) -> dict:
    """How the attribution was made. factor_terms: the union of the equations' features (an asset's own
    terms are asset_factor_terms[a] where its equation differs from the common one); own_beta_min_months:
    the common equation's (an asset with extras needs OWN_BETA_MONTHS_PER_PARAM more per extra);
    asset_train_start: each equation's first fit month (later than train_start for an asset_start)."""
    method, why = attribution_method(model.spec)
    linear = method == "linear"
    own = {a: asset_feature_names(model.spec, a) for a in model.assets} if linear else {}
    common = min((len(v) for v in own.values()), default=len(model.features))
    return {"method": method, "reason": why or None, "beta": "full_sample",
            "model": {"spec": model.spec.to_dict(), "describe": model.spec.describe(),
                      "train_start": _ym(model.train_start), "train_end": _ym(model.train_end),
                      "asset_train_start": dict(getattr(model, "asset_train_start", {}) or {})},
            "factor_terms": list(model.features) if linear else [],
            "asset_factor_terms": {a: f for a, f in own.items() if f != list(model.features)},
            "min_coverage": ATTR_MIN_COVERAGE, "own_beta_min_months": OWN_BETA_MONTHS_PER_PARAM * (common + 1),
            "units": ATTR_UNITS}


def asset_start_note(model: AssetFactorModel) -> str | None:
    """A note for the eras page when some equation starts later than the common sample, else None."""
    late = {a: m for a, m in (getattr(model, "asset_start_used", {}) or {}).items() if m}
    if not late:
        return None
    items = "、".join(f"{ASSET_NAMES.get(a, a)}从 {m[:4]} 年" for a, m in sorted(late.items()))
    return (f"传导模型里{items}起估计：更早时期的归因也用这套β（情景引擎同样如此），这些时期不在它的估计样本里。")


# ---------------------------------------------------------------- cross-era views

def fingerprint(records: list[dict], ref: pd.DataFrame, cols: list[str] = FINGERPRINT_STATES) -> dict:
    """Era x state heatmap: z = (era mean - 1955+ mean) / 1955+ sd, plus the raw era means."""
    z, values = [], []
    for rec in records:
        zr, vr = [], []
        for sid in cols:
            st = rec["states"].get(sid)
            if st is None or sid not in ref.index:
                zr.append(None)
                vr.append(None)
                continue
            vr.append(st["mean"])
            zr.append((st["mean"] - ref.loc[sid, "mean"]) / ref.loc[sid, "sd"])
        z.append(zr)
        values.append(vr)
    return {"rows": [r["id"] for r in records], "row_names": [r["name"] for r in records],
            "cols": cols, "col_names": [STATE_INFO[c]["name"] for c in cols],
            "col_units": [STATE_INFO[c]["unit"] for c in cols],
            "ref_mean": [float(ref.loc[c, "mean"]) for c in cols],
            "ref_sd": [float(ref.loc[c, "sd"]) for c in cols], "z": z, "values": values}


def soft_pool(Z: np.ndarray, b: np.ndarray, tau: float) -> np.ndarray:
    """Uniform-prior entropy pooling with quadratic view penalties.

    Z: (k, T) standardized state rows, b: (k,) standardized targets. Solves
    min_q KL(q||p) + |Z q - b|^2 / (2 tau), whose dual is the hard EP dual plus
    tau/2 |theta|^2 (always finite). Returns q.
    """
    T = Z.shape[1]
    logp = np.full(T, -np.log(T))

    def dual(theta):
        z = logp - Z.T @ theta
        lse = logsumexp(z)
        q = np.exp(z - lse)
        return lse + theta @ b + 0.5 * tau * theta @ theta, b - Z @ q + tau * theta

    res = minimize(dual, np.zeros(len(b)), jac=True, method="L-BFGS-B",
                   options={"maxiter": 3000, "gtol": 1e-11, "ftol": 1e-15})
    z = logp - Z.T @ res.x
    return np.exp(z - logsumexp(z))


def match_today(X: np.ndarray, target: np.ndarray, ens_min: float = ENS_MIN) -> tuple[np.ndarray, float]:
    """Smallest view tolerance tau with ENS >= ens_min (bisection on log tau). Returns (q, tau)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = ((X - mu) / sd).T
    b = (target - mu) / sd
    lo, hi = -6.0, 3.0
    q = soft_pool(Z, b, 10 ** lo)
    if effective_number(q) >= ens_min:
        return q, 10 ** lo
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        q_mid = soft_pool(Z, b, 10 ** mid)
        if effective_number(q_mid) >= ens_min:
            hi = mid
        else:
            lo = mid
    return soft_pool(Z, b, 10 ** hi), 10 ** hi


def similarity_end(panel: pd.DataFrame, eras: list[dict] = config.ERAS) -> tuple[pd.Timestamp, list[str]]:
    """Last month of the similarity database and the ongoing era ids it leaves out.

    The database stops EXCLUDE_RECENT months before the last month and before
    any era that has no end yet.
    """
    end = panel.index.max() - pd.DateOffset(months=EXCLUDE_RECENT)
    ongoing = [e for e in eras if e["end"] is None and pd.Timestamp(e["start"]) <= panel.index.max()]
    for e in ongoing:
        end = min(end, pd.Timestamp(e["start"]) - pd.DateOffset(months=1))
    return end, [e["id"] for e in ongoing]


def today_similarity(panel: pd.DataFrame, ref: pd.DataFrame, states: list[str] = SIMILARITY_STATES,
                     ens_min: float = ENS_MIN) -> dict:
    """Today (mean of the last 12 months) as a mixture of historical months."""
    last = panel.index.max()
    cols = [f"s_{s}" for s in states]
    recent = panel.iloc[-TODAY_MONTHS:]
    today = recent[cols].mean()
    db_end, excluded = similarity_end(panel)
    db = panel.loc[FULL_START:db_end].dropna(subset=cols)
    X = db[cols].to_numpy(float)
    q_soft, tau = match_today(X, today.to_numpy(float), ens_min)
    res = apply_views(db[cols], [{"var": c, "mean": float(m)} for c, m in zip(cols, q_soft @ X)])
    replayed = res.converged and not any(v["clipped"] for v in res.views)
    q = res.q if replayed else q_soft
    ens = res.ens if replayed else effective_number(q)
    era_ids = np.array([era_of(d) for d in db.index])
    weights = {e["id"]: float(q[era_ids == e["id"]].sum()) for e in config.ERAS if e["id"] not in excluded}

    views = [{"var": c, "state": sid, "name": STATE_INFO[sid]["name"], "today": float(today[c]),
              "target": float(v["target"]), "prior_mean": float(v["prior_mean"]),
              "posterior_mean": float(q @ X[:, j])}
             for j, (sid, c, v) in enumerate(zip(states, cols, res.views))]
    residual = []
    for sid in list(states) + [s for s in FINGERPRINT_STATES if s not in states]:
        x = db[f"s_{sid}"].to_numpy(float)
        ok = np.isfinite(x)
        now = float(recent[f"s_{sid}"].mean())
        if not ok.any() or q[ok].sum() < 1e-9 or not np.isfinite(now):
            continue
        matched = float(q[ok] @ x[ok] / q[ok].sum())
        residual.append({"state": sid, "name": STATE_INFO[sid]["name"], "unit": STATE_INFO[sid]["unit"],
                         "in_view": sid in states, "today": now, "matched": matched, "gap": now - matched,
                         "gap_z": (now - matched) / float(ref.loc[sid, "sd"])})
    residual.sort(key=lambda r: -abs(r["gap_z"]))
    top = np.argsort(-q)[:12]
    out = {
        "date": _ym(last), "window": [_ym(recent.index[0]), _ym(last)],
        "database": [_ym(db.index.min()), _ym(db.index.max())], "n_months_db": int(len(db)),
        "excluded_eras": excluded,
        "method": "soft_entropy_pooling", "ens_min": ens_min, "tau": tau, "replayed_by_engine": bool(replayed),
        "weights": weights, "ens": float(ens), "ens_ratio": float(ens / len(db)),
        "kl": kl_divergence(q, np.full(len(q), 1.0 / len(q))), "views": views, "residual": residual,
        "top_months": [{"date": _ym(db.index[i]), "era": str(era_ids[i]), "weight": float(q[i])} for i in top],
    }
    out["interpretation"] = similarity_text(out)
    return out


def similarity_text(sim: dict) -> str:
    ranked = sorted(sim["weights"].items(), key=lambda kv: -kv[1])
    top = [f"{ERA_NAMES.get(e, e)}（{w * 100:.0f}%）" for e, w in ranked[:3] if w >= 0.03]
    skipped = "、".join(ERA_NAMES.get(e, e) for e in sim.get("excluded_eras", []))
    text = (f"用过去 12 个月（{sim['window'][0]} 至 {sim['window'][1]}）{len(sim['views'])} 个关键宏观状态的均值，"
            f"在 {sim['database'][0]} 至 {sim['database'][1]} 的 {sim['n_months_db']} 个历史月份里做熵池重新加权"
            f"（保留至少 {sim['ens_min']:.0f} 个有效月份"
            + (f"；不含今天所处的「{skipped}」本身" if skipped else "") + "），今天最像" + "、".join(top) + "的混合，"
            f"有效样本约 {sim['ens']:.0f} 个月。")
    gaps = [r for r in sim["residual"] if abs(r["gap_z"]) >= 0.4][:4]
    if gaps:
        items = [f"{r['name']}（今天 {_fmt_state(r['state'], r['today'])}，历史匹配 {_fmt_state(r['state'], r['matched'])}）"
                 for r in gaps]
        text += "历史无法复制的部分：" + "；".join(items) + "。这些差距就是今天“新”的地方，情景推演时要额外考虑。"
    else:
        text += "各项状态都能在历史中找到相近的组合。"
    return text


def timeline(panel: pd.DataFrame, cols: list[str] = FINGERPRINT_STATES) -> dict:
    P = panel.loc[FULL_START:]
    return {"dates": [_ym(d) for d in P.index], "names": {c: STATE_INFO[c]["name"] for c in cols},
            "series": {c: P[f"s_{c}"].to_numpy(float) for c in cols},
            "bands": [{"id": e["id"], "name": e["name"], "start": e["start"], "end": e["end"] or _ym(panel.index.max())}
                      for e in config.ERAS]}


def run(ctx, **_) -> dict:
    panel = ctx.panel
    ref = reference_stats(panel)
    records = [era_record(panel, era, ref) for era in config.ERAS]
    model = attribution_model(ctx)
    X, fitted = design_matrix(panel, model.spec), model.predict(panel)
    for rec, era in zip(records, config.ERAS):
        rec["attribution"] = era_attribution(panel, model, X, fitted, *era_bounds(era, panel))
    return {
        "reference": {"start": FULL_START, "end": _ym(panel.index.max())},
        "eras": records,
        "attribution_info": attribution_info(model),
        "fingerprint": fingerprint(records, ref),
        "today_similarity": today_similarity(panel, ref),
        "timeline": timeline(panel),
        "notes": [
            "z 值 = (时期均值 - 1955 年以来均值) / 1955 年以来标准差。",
            "实际年化回报用 CPI 指数平减；资产只统计在该时期每个月都有数据的。",
            "劳动收入占比、利润、投资、净值等季度数据按通常的发布滞后对齐，只在发布月变化，其月度因子变化集中在这些月份。",
            "今天的相似度：过去 12 个月均值作为观点，在 24 个月以前、且早于当前时代开始的历史月份上做熵池重新加权；"
            "无法同时满足的观点会被软化，剩下的差距列在 residual 中。",
            f"油价有变动的月份不到 {MIN_OIL_MOVES} 个的时期（1973 年以前 WTI 是管制挂牌价）不报告油价对 CPI 的传导。",
            ("传导归因：时期内每月价格回报的均值 ×12 = Σ（1973 年以来全样本β × 该时期因子的平均月度变化 ×12）+ 常数项"
             " + 未解释部分（残差均值），再加收入 = 总回报的算术年化；复利与波动修正（复利的抬升减去波动拖累，可正可负）把它换算到年化回报（CAGR）。"
             "1973 年以前没有贸易加权美元指数，不做归因。"),
            *([note] if (note := asset_start_note(model)) else []),
        ],
    }


if __name__ == "__main__":
    from ..context import EngineContext

    jsonio.dump(run(EngineContext.load(selection=False)), config.RESULTS_DIR / "eras.json")
