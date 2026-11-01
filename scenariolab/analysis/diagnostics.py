"""Why is the prediction wrong? Statistical diagnostics of the transmission model.

run(ctx) drives the walk-forward loop (walkforward.py) with `diagnose` as its
first step, analyses the residuals of the final spec on the full sample,
derives recommendations and writes results/model_selection.json, the spec
EngineContext then refits on all data. It returns the diagnostics.json dict.

Tests, on monthly price returns of the non-structural assets (the nested
F-tests use the linear OLS version of the spec):
  missing factor   correlation of the model residuals with each candidate
                   series, train (in-sample) and validate (out of sample)
  missing lags     Ljung-Box Q(12) of the residuals + joint F-test of 1-3
                   month lags of every factor
  regime           joint F-test of the inflation-regime interactions
  parameter break  Chow test, training vs validation months
  nonlinearity     RESET test (squared and cubed fitted values)
  drift bias       mean out-of-sample residual and its share of the MSE
  overfit          in-sample vs out-of-sample R^2
t-statistics treat months as independent; quarterly-interpolated series are
smooth, so their significance is overstated.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from .. import config, jsonio
from ..context import EngineContext
from ..engine.factor_model import MODEL_ASSETS, AssetFactorModel, ModelSpec, design_matrix
from . import walkforward as wf

SCORE_ASSETS = wf.SCORE_ASSETS
NAMES = {a["id"]: a["name"] for a in config.ASSETS}
FACTOR_NAMES = {**wf.FACTOR_NAMES, "e_oil_supply": "石油供给冲击", "e_demand": "总需求冲击",
                "e_monetary": "货币政策冲击", "e_financial": "金融冲击"}
ALPHA = 0.05
MIN_FLAG = 3                  # assets that must reject before a finding is flagged
ACF_LAGS = 12
BREAK_DATE = "2000-01"
BETA_FACTORS = ("oil", "y10", "spread", "cpi_services")
BETA_WINDOW = 120
SELECTION_PATH = config.RESULTS_DIR / "model_selection.json"
UNITS = {
    "score": "非结构性资产（股票、REITs、黄金）月度价格收益样本外 R² 的平均值，用实际因子变化预测；"
             "R² = 1 − SSE/SST，SST 以训练期均值为基准，R² > 0 表示好过只用历史均值",
    "bias_annual": "验证期平均残差 × 12（小数，正数 = 模型低估）",
    "beta_per_sd": "因子变动 1 个标准差时资产当月价格收益（小数）",
    "p": "检验的 p 值（< 0.05 视为显著）",
}
NOTES = [
    "所有 t 统计量把月份当作相互独立；季度数据插值成的月度序列很平滑，显著性会被高估。",
    "F 检验（滞后、状态、Chow、RESET）用同一组特征的普通最小二乘线性模型；神经网络没有对应的检验。",
    "Chow 检验假设两段样本的残差方差相同；金融收益的波动率随时代变化，检验会偏向拒绝。",
]


def _ym(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


# ---------------------------------------------------------------- statistics

def acf(x, nlags: int = ACF_LAGS) -> np.ndarray:
    x = np.asarray(x, float)
    d = x[np.isfinite(x)]
    d = d - d.mean()
    den = (d ** 2).sum()
    return np.array([(d[k:] * d[:-k]).sum() / den for k in range(1, nlags + 1)])


def ljung_box(x, lags: int = ACF_LAGS) -> tuple[float, float]:
    x = np.asarray(x, float)
    n = int(np.isfinite(x).sum())
    r = acf(x, lags)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def corr_t(r: float, n: int) -> float:
    return float(r * np.sqrt(max(n - 2, 1) / max(1.0 - r * r, 1e-12)))


def _ssr(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, int]:
    """Residual sum of squares per column of y on X plus a constant, and the rank of that design."""
    Z = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return ((y - Z @ beta) ** 2).sum(axis=0), int(np.linalg.matrix_rank(Z))


def nested_f(y: np.ndarray, X_r: np.ndarray, X_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F-test that the extra columns of X_u add nothing over X_r, per column of y."""
    s_r, k_r = _ssr(y, X_r)
    s_u, k_u = _ssr(y, X_u)
    q, dof = max(k_u - k_r, 1), len(y) - k_u
    F = ((s_r - s_u) / q) / (s_u / dof)
    return F, stats.f.sf(F, q, dof)


def chow(y1: np.ndarray, X1: np.ndarray, y2: np.ndarray, X2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chow F-test that two samples share the same coefficients, per column of y."""
    s_p, k = _ssr(np.vstack([y1, y2]), np.vstack([X1, X2]))
    s1, _ = _ssr(y1, X1)
    s2, _ = _ssr(y2, X2)
    dof = len(y1) + len(y2) - 2 * k
    F = ((s_p - s1 - s2) / k) / ((s1 + s2) / dof)
    return F, stats.f.sf(F, k, dof)


def reset_test(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RESET: do squared and cubed fitted values add explanatory power? Per column of y."""
    Z = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    fit = Z @ beta
    F, p = np.empty(y.shape[1]), np.empty(y.shape[1])
    for j in range(y.shape[1]):
        f = (fit[:, j] - fit[:, j].mean()) / fit[:, j].std()
        Fj, pj = nested_f(y[:, [j]], X, np.column_stack([X, f ** 2, f ** 3]))
        F[j], p[j] = Fj[0], pj[0]
    return F, p


def _xy(panel: pd.DataFrame, spec: ModelSpec, start, end, assets=SCORE_ASSETS,
        other: ModelSpec | None = None) -> tuple[np.ndarray, ...]:
    """Targets and design over [start, end]; with `other`, both designs on their common months."""
    X = design_matrix(panel, spec).loc[start:end]
    Y = panel.loc[start:end, [f"p_{a}" for a in assets]]
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    if other is None:
        return Y[ok].to_numpy(float), X[ok].to_numpy(float)
    X2 = design_matrix(panel, other).loc[start:end]
    ok &= X2.notna().all(axis=1)
    return Y[ok].to_numpy(float), X[ok].to_numpy(float), X2[ok].to_numpy(float)


def _corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    pair = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(pair) < 12:
        return float("nan"), len(pair)
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), len(pair)


def _names(assets) -> str:
    return "、".join(NAMES[a] for a in assets) or "无"


def _finding(fid: str, group: str, test: str, flagged: bool, text: str, suggests=(), **extra) -> dict:
    return {"id": fid, "group": group, "question": wf.QUESTIONS.get(group, group), "test": test,
            "flagged": bool(flagged), "suggests": list(suggests), "text": text, **extra}


# ---------------------------------------------------------------- findings

def _candidate_columns(candidates) -> dict[str, tuple[str, bool]]:
    """Panel column -> (candidate id, eligible) for every candidate that adds a series."""
    out = {}
    for c in candidates:
        if c.change.get("drop_factors"):
            continue
        for col in [f"f_{f}" for f in c.change.get("add_factors", ())] + list(c.change.get("shock_cols", ())):
            out.setdefault(col, (c.id, c.eligible))
    return out


def share_text(x: float) -> str:
    """A share near 1 keeps a decimal: an R^2 of 0.996 is not "100%" (that would read as an identity)."""
    return f"{x:.1%}" if 0.995 <= x < 1.0 else f"{x:.0%}"


def spanned_r2(panel: pd.DataFrame, spec: ModelSpec, col: str, rows) -> float:
    """R^2 of a candidate series on the spec's current base features: how much it duplicates them."""
    X = design_matrix(panel, replace(spec, lags=0, regime=None)).loc[rows]
    y = panel.loc[rows, col]
    ok = X.notna().all(axis=1) & y.notna()
    if ok.sum() < 3 * (X.shape[1] + 2):
        return float("nan")
    yv = y[ok].to_numpy(float)[:, None]
    ssr, _ = _ssr(yv, X[ok].to_numpy(float))
    return float(1.0 - ssr[0] / ((yv - yv.mean()) ** 2).sum())


def _missing_factor(panel, spec, resid, rows_tr, rows_va, candidates) -> dict:
    table = []
    for col, (cid, eligible) in _candidate_columns(candidates).items():
        per = {}
        for a in SCORE_ASSETS:
            row = {}
            for tag, rows in (("train", rows_tr), ("validate", rows_va)):
                r, n = _corr(resid.loc[rows, a], panel.loc[rows, col])
                row[f"corr_{tag}"], row[f"t_{tag}"] = r, corr_t(r, n) if np.isfinite(r) else float("nan")
            per[a] = row
        t_va = {a: abs(per[a]["t_validate"]) for a in SCORE_ASSETS if np.isfinite(per[a]["t_validate"])}
        best = max(t_va, key=t_va.get) if t_va else None
        both = [a for a in SCORE_ASSETS if abs(per[a]["t_train"]) > 2 and abs(per[a]["t_validate"]) > 2
                and np.sign(per[a]["corr_train"]) == np.sign(per[a]["corr_validate"])]
        table.append({
            "column": col, "name": FACTOR_NAMES.get(col[2:] if col.startswith("f_") else col, col),
            "candidate": cid, "eligible": eligible,
            "n_sig_train": int(sum(abs(per[a]["t_train"]) > 2 for a in SCORE_ASSETS)),
            "n_sig_validate": int(sum(abs(per[a]["t_validate"]) > 2 for a in SCORE_ASSETS)),
            "n_sig_both": len(both), "spanned_r2": spanned_r2(panel, spec, col, rows_tr),
            "mean_abs_corr_validate": float(np.nanmean([abs(per[a]["corr_validate"]) for a in SCORE_ASSETS])),
            "best": {"asset": best, "corr": per[best]["corr_validate"], "t": per[best]["t_validate"]} if best else None,
            "per_asset": per,
        })
    table.sort(key=lambda r: (-r["n_sig_both"], -r["n_sig_validate"], -r["mean_abs_corr_validate"]))
    good = [r for r in table if r["eligible"] and r["n_sig_both"] >= MIN_FLAG]
    late = [r for r in table if r["eligible"] and r["n_sig_validate"] >= MIN_FLAG and r["n_sig_train"] < MIN_FLAG]
    n = len(SCORE_ASSETS)
    shown = sorted(table, key=lambda r: (-r["n_sig_validate"], -r["mean_abs_corr_validate"]))[:3]
    parts = [f"「{r['name']}」{'' if r['eligible'] else '（内生，只作对照）'}：训练期 {r['n_sig_train']}/{n}、"
             f"验证期 {r['n_sig_validate']}/{n}" for r in shown]
    text = "与残差显著相关（|t|>2）的资产数，最多的是 " + "；".join(parts) + "。"
    text += (f"在训练期和验证期都显著、方向一致的可用因子：{'、'.join(r['name'] for r in good)}，值得一试。" if good
             else f"没有一个可用的候选因子在训练期和验证期都对 {MIN_FLAG} 个以上资产显著，缺因子不是主要问题。")
    if late:
        text += (f"{'、'.join(r['name'] for r in late)} 只在验证期显著：训练期残差与它们无关，"
                 f"验证期的相关更像是已有因子的系数漂移，而不是缺了一个新因子。")
    # candidates the current spec already holds (e.g. one adopted in an earlier iteration) are spanned trivially
    in_spec = {f"f_{f}" for f in spec.factors} | set(spec.shock_cols)
    dup = [r for r in table if np.isfinite(r["spanned_r2"]) and r["spanned_r2"] > 0.5 and r["column"] not in in_spec]
    if dup:
        text += ("候选里 " + "、".join(f"{r['name']}（{share_text(r['spanned_r2'])}）" for r in dup)
                 + " 的大部分变化已经能由现有因子线性解释（括号内为对现有因子回归的 R²），新增的信息很少。")
    return _finding("missing_factor", "missing_factor", "残差与候选因子的相关系数（t 统计量，|t|>2 视为显著）",
                    bool(good), text, sorted({r["candidate"] for r in good}), table=table)


def _missing_lags(panel, lin, resid, rows_tr, sample) -> dict:
    per = {}
    for a in SCORE_ASSETS:
        e = resid.loc[rows_tr, a]
        q, p = ljung_box(e)
        per[a] = {"acf": acf(e), "acf1": float(acf(e, 1)[0]), "lb_q": q, "lb_p": p}
    if lin.lags < 3:
        y, Xr, Xu = _xy(panel, lin, *sample, other=replace(lin, lags=3))
        F, p = nested_f(y, Xr, Xu)
        for j, a in enumerate(SCORE_ASSETS):
            per[a].update({"lag_f": float(F[j]), "lag_p": float(p[j])})
    lb = [a for a in SCORE_ASSETS if per[a]["lb_p"] < ALPHA]
    lag = [a for a in SCORE_ASSETS if per[a].get("lag_p", 1.0) < ALPHA]
    text = (f"残差 Ljung-Box(12) 显著自相关的资产：{_names(lb)}（{len(lb)}/{len(SCORE_ASSETS)}）；"
            f"1–3 个月滞后因子联合显著的资产：{_names(lag)}（{len(lag)}/{len(SCORE_ASSETS)}）。")
    if per["gold"]["acf1"] > 0.15:
        text += (f"黄金残差的一阶自相关为 {per['gold']['acf1']:.2f}：数据用的是月均金价，时间平均本身就会产生约 0.25 的"
                 f"一阶自相关，这不是真实的传导滞后。")
    return _finding("missing_lags", "missing_mechanism", "残差 Ljung-Box(12) + 1–3 阶滞后因子联合 F 检验",
                    len(lag) >= MIN_FLAG, text, ["lags1", "lags3"] if len(lag) >= MIN_FLAG else [],
                    per_asset=per, n_ljung_box_reject=len(lb), n_lag_reject=len(lag))


def _regime(panel, lin, sample) -> dict:
    if lin.regime:
        return _finding("regime", "missing_mechanism", "通胀状态交互项联合 F 检验", False, "当前模型已含通胀状态交互。")
    y, Xr, Xu = _xy(panel, lin, *sample, other=replace(lin, regime="inflation"))
    F, p = nested_f(y, Xr, Xu)
    per = {a: {"f": float(F[j]), "p": float(p[j])} for j, a in enumerate(SCORE_ASSETS)}
    rej = [a for a in SCORE_ASSETS if per[a]["p"] < ALPHA]
    text = (f"高通胀（CPI 同比 > 4%）与低通胀月份的传导系数显著不同的资产：{_names(rej)}"
            f"（{len(rej)}/{len(SCORE_ASSETS)}）。")
    return _finding("regime", "missing_mechanism", "通胀状态交互项联合 F 检验", len(rej) >= MIN_FLAG, text,
                    ["regime_inflation"] if len(rej) >= MIN_FLAG else [], per_asset=per, n_reject=len(rej))


def _parameter_break(panel, lin, sample, validate) -> dict:
    y1, X1 = _xy(panel, lin, *sample)
    y2, X2 = _xy(panel, lin, *validate)
    F, p = chow(y1, X1, y2, X2)
    per = {a: {"f": float(F[j]), "p": float(p[j])} for j, a in enumerate(SCORE_ASSETS)}
    rej = [a for a in SCORE_ASSETS if per[a]["p"] < ALPHA]
    text = (f"训练期（{_ym(sample[0])} 至 {_ym(sample[1])}）和验证期（{validate[0]} 至 {validate[1]}）的系数显著不同的资产："
            f"{_names(rej)}（{len(rej)}/{len(SCORE_ASSETS)}）。"
            + ("传导关系本身在变，这是样本外 R² 低的直接原因之一。" if len(rej) >= MIN_FLAG
               else "多数资产的系数没有显著变化，样本外误差主要不是来自系数漂移。"))
    return _finding("parameter_break", "tuning", "Chow 检验（训练期与验证期系数是否相同）", len(rej) >= MIN_FLAG, text,
                    ["window_15", "window_25", "ridge_0.05"] if len(rej) >= MIN_FLAG else [],
                    per_asset=per, n_reject=len(rej))


def _nonlinearity(panel, lin, sample) -> dict:
    y, X = _xy(panel, lin, *sample)
    F, p = reset_test(y, X)
    per = {a: {"f": float(F[j]), "p": float(p[j])} for j, a in enumerate(SCORE_ASSETS)}
    rej = [a for a in SCORE_ASSETS if per[a]["p"] < ALPHA]
    text = (f"RESET 检验显示存在非线性的资产：{_names(rej)}（{len(rej)}/{len(SCORE_ASSETS)}）。"
            + ("非线性有统计证据，可以试神经网络，但要看样本外是否真的更好。" if len(rej) >= MIN_FLAG
               else "多数资产没有明显的非线性，神经网络的潜在收益有限。"))
    return _finding("nonlinearity", "neural_net", "RESET 检验（拟合值的平方、立方项是否显著）", len(rej) >= MIN_FLAG, text,
                    ["mlp16", "mlp32"] if len(rej) >= MIN_FLAG else [], per_asset=per, n_reject=len(rej))


def _drift_bias(resid, rows_va) -> dict:
    per = {}
    for a in SCORE_ASSETS:
        e = resid.loc[rows_va, a].dropna()
        m, sd = float(e.mean()), float(e.std(ddof=1))
        per[a] = {"bias_annual": 12.0 * m, "t": m / (sd / np.sqrt(len(e))), "bias_share": m ** 2 / float((e ** 2).mean())}
    share = float(np.mean([per[a]["bias_share"] for a in SCORE_ASSETS]))
    worst = max(SCORE_ASSETS, key=lambda a: abs(per[a]["bias_annual"]))
    sig = [a for a in SCORE_ASSETS if abs(per[a]["t"]) > 2]
    text = (f"验证期平均预测偏差最大的是{NAMES[worst]}（每年 {per[worst]['bias_annual']:+.1%}，t={per[worst]['t']:.1f}）；"
            f"偏差显著的资产 {len(sig)}/{len(SCORE_ASSETS)} 个。平均漂移偏差只占样本外均方误差的 {share:.1%}，"
            f"月度误差主要来自因子解释不了的波动，长期偏差则会随复利放大。")
    return _finding("drift_bias", "tuning", "验证期平均残差（年化）与偏差占均方误差的比例", len(sig) >= MIN_FLAG, text,
                    per_asset=per, mean_bias_share=share, worst=worst)


def _overfit(model, ev) -> dict:
    per = {a: {"train_r2": model.stats.r2[a], "validate_r2": ev["per_asset"][a]} for a in SCORE_ASSETS}
    tr = float(np.mean([per[a]["train_r2"] for a in SCORE_ASSETS]))
    va = ev["score"]
    neg = [a for a in SCORE_ASSETS if per[a]["validate_r2"] < 0]
    text = (f"训练期平均 R² {tr:.3f}，验证期（样本外）{va:.3f}；样本外 R² 为负的资产：{_names(neg)}。"
            + ("样本外几乎没有解释力，收缩系数或缩短窗口值得一试。" if va < tr / 2 else "样本外保留了大部分解释力。"))
    return _finding("overfit", "tuning", "训练期 R² 与验证期 R² 的差距", va < tr / 2, text,
                    ["ridge_0.01", "ridge_0.05"] if va < tr / 2 else [], per_asset=per, train_r2=tr, validate_r2=va)


def diagnose(panel: pd.DataFrame, spec: ModelSpec, train: tuple, validate: tuple, candidates) -> list[dict]:
    """Step (a) of each walk-forward iteration: why does `spec` miss on train / validate?"""
    model = AssetFactorModel(spec).fit(panel, *train)
    resid = model.residuals(panel)[SCORE_ASSETS]
    sample = (model.train_start, model.train_end)
    rows_tr = resid.loc[sample[0]:sample[1]].dropna().index
    rows_va = resid.loc[validate[0]:validate[1]].dropna().index
    lin = replace(spec, kind="linear", ridge=0.0)
    ev = wf.evaluate(panel, spec, train, validate)
    return [
        _overfit(model, ev),
        _missing_factor(panel, spec, resid, rows_tr, rows_va, candidates),
        _missing_lags(panel, lin, resid, rows_tr, sample),
        _regime(panel, lin, sample),
        _parameter_break(panel, lin, sample, validate),
        _nonlinearity(panel, lin, sample),
        _drift_bias(resid, rows_va),
    ]


# ---------------------------------------------------------------- full-sample residuals

def rolling_betas(panel: pd.DataFrame, spec: ModelSpec, factors=BETA_FACTORS, window: int = BETA_WINDOW) -> dict:
    """Trailing `window`-month OLS betas, one point per calendar year, per 1-sd factor move."""
    lin = ModelSpec(factors=tuple(dict.fromkeys([*spec.factors, *factors])), shock_cols=spec.shock_cols)
    X = design_matrix(panel, lin).loc[config.MODEL_START:]
    Y = panel.loc[config.MODEL_START:, [f"p_{a}" for a in MODEL_ASSETS]]
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X, Y = X[ok], Y[ok]
    cols = [list(X.columns).index(f) + 1 for f in factors]          # +1 for the constant
    sd = X[list(factors)].std()
    years = []
    beta = {a: {f: [] for f in factors} for a in MODEL_ASSETS}
    se = {a: {f: [] for f in factors} for a in MODEL_ASSETS}
    for year in range(X.index[0].year, X.index[-1].year + 1):
        idx = X.index[X.index <= pd.Timestamp(f"{year}-12-01")][-window:]
        if len(idx) < window or idx[-1].year != year:
            continue
        Z = np.column_stack([np.ones(window), X.loc[idx].to_numpy(float)])
        y = Y.loc[idx].to_numpy(float)
        b, *_ = np.linalg.lstsq(Z, y, rcond=None)
        s2 = ((y - Z @ b) ** 2).sum(axis=0) / (window - Z.shape[1])
        diag = np.diag(np.linalg.pinv(Z.T @ Z))
        years.append(str(year))
        for j, a in enumerate(MODEL_ASSETS):
            for f, c in zip(factors, cols):
                beta[a][f].append(float(b[c, j] * sd[f]))
                se[a][f].append(float(np.sqrt(diag[c] * s2[j]) * sd[f]))
    return {"years": years, "factors": list(factors), "window_months": window,
            "factor_sd": {f: float(sd[f]) for f in factors},
            "assets": {a: {"beta": beta[a], "se": se[a]} for a in MODEL_ASSETS}}


def residual_analysis(panel: pd.DataFrame, spec: ModelSpec, candidates) -> dict:
    model = AssetFactorModel(spec).fit(panel, config.MODEL_START, None)
    resid = model.residuals(panel).loc[model.train_start:model.train_end, MODEL_ASSETS]
    lin = replace(spec, kind="linear", ridge=0.0)
    brk = pd.Timestamp(BREAK_DATE)
    y1, X1 = _xy(panel, lin, model.train_start, brk - pd.DateOffset(months=1), MODEL_ASSETS)
    y2, X2 = _xy(panel, lin, brk, model.train_end, MODEL_ASSETS)
    k = X1.shape[1] + 1
    F, p = chow(y1, X1, y2, X2) if min(len(y1), len(y2)) > 3 * k else (np.full(len(MODEL_ASSETS), np.nan),) * 2
    cols = _candidate_columns(candidates)
    assets = {}
    for j, a in enumerate(MODEL_ASSETS):
        e = resid[a].dropna()
        q, pv = ljung_box(e)
        corr = {}
        for col, (cid, eligible) in cols.items():
            r, n = _corr(e, panel[col])
            corr[col] = {"corr": r, "t": corr_t(r, n) if np.isfinite(r) else None, "candidate": cid, "eligible": eligible}
        assets[a] = {"name": NAMES[a], "r2": model.stats.r2[a], "resid_sd": model.stats.resid_sd[a],
                     "acf": acf(e), "ljung_box_q": q, "ljung_box_p": pv, "factor_corr": corr,
                     "chow_f": float(F[j]), "chow_p": float(p[j])}
    chow_rej = [a for a in SCORE_ASSETS if assets[a]["chow_p"] < ALPHA]
    lb_rej = [a for a in SCORE_ASSETS if assets[a]["ljung_box_p"] < ALPHA]
    sample = (model.train_start, model.train_end)
    leakage = {f: wf.leakage_evidence(panel, f"f_{f}", sample) for f in wf.ENDOGENOUS if f"f_{f}" in panel}
    return {
        "spec": spec.to_dict(), "describe": spec.describe(),
        "sample": [_ym(model.train_start), _ym(model.train_end)], "break_date": BREAK_DATE,
        "acf_lags": list(range(1, ACF_LAGS + 1)), "assets": assets,
        "leakage": leakage,
        "rolling_betas": rolling_betas(panel, spec),
        "summary": {"r2_risk_avg": float(np.mean([assets[a]["r2"] for a in SCORE_ASSETS])),
                    "n_chow_reject": len(chow_rej), "chow_reject": chow_rej,
                    "n_ljung_box_reject": len(lb_rej), "ljung_box_reject": lb_rej},
    }


# ---------------------------------------------------------------- recommendations

def _period(p) -> str:
    return f"{p[0][:4]}–{p[1][:4]}"


def oos_deltas(its: list[dict], cid: str) -> list[dict]:
    """R^2 change vs the base when `cid` is added alone, per out-of-sample period.

    The walk-forward periods chain (iteration k's test = iteration k+1's validation, both fitted
    on the same months), so the first validation plus every test give one score per decade.
    """
    abl = [{r["candidate"]: r for r in it["test_ablation"]} for it in its]
    if not all(cid in a for a in abl):
        return []
    out = [{"period": _period(its[0]["validate"]), "delta": abl[0][cid]["val_delta"]}]
    return out + [{"period": _period(it["test"]), "delta": a[cid]["test_delta"]} for it, a in zip(its, abl)]


def _fmt(rows: list[dict]) -> str:
    return "、".join(f"{r['period']} {r['delta']:+.3f}" for r in rows)


def recommendations(res: dict, resid: dict) -> list[dict]:
    its = res["iterations"]
    fin = res["final"]
    recs = []

    def add(priority: str, group: str, title: str, detail: str, **evidence):
        recs.append({"priority": priority, "group": group, "question": wf.QUESTIONS.get(group, ""),
                     "title": title, "detail": detail, "evidence": evidence})

    base_oos = ([{"period": _period(its[0]["validate"]), "r2": its[0]["baseline"]["val_score"]}]
                + [{"period": _period(it["test"]), "r2": it["test_result"]["base_all_months"]["score"]} for it in its])
    base_txt = "、".join(f"{r['period']} {r['r2']:+.3f}" for r in base_oos)
    if fin["is_base"]:
        add("高", "tuning", f"保留基准线性模型（{len(fin['spec']['factors'])} 个核心宏观因子，全样本估计）",
            fin["reason"] + f" 基准模型各样本外十年的平均 R²：{base_txt}。", base_oos_r2=base_oos,
            last_selected=fin["last_selected"])
    else:
        add("高", "tuning", f"采用 {fin['describe']}",
            fin["reason"] + f" 测试期平均 R² {fin['base_test_score']:.3f} → {fin['test_score']:.3f}；"
            f"基准各样本外十年：{base_txt}。", changes=fin["changes"], base_oos_r2=base_oos)

    endo = {c: oos_deltas(its, c) for c in ("add_valuation", "add_cape", "add_wealth")}
    endo = {c: d for c, d in endo.items() if d}
    if endo:
        parts = "；".join(f"{FACTOR_NAMES[c.removeprefix('add_')]}：{_fmt(d)}" for c, d in endo.items())
        lk = resid.get("leakage", {})
        why = []
        if "add_valuation" in endo and "valuation" in lk:
            why.append(f"估值变化由美股价格本身构造（全样本与当月美股价格收益相关 {lk['valuation']['same_month']:.2f}）")
        if "add_cape" in endo and "cape" in lk:
            why.append(f"席勒 CAPE 的变化同样几乎就是股价本身（与当月美股价格收益相关 {lk['cape']['same_month']:.2f}）")
        if "add_wealth" in endo and "wealth" in lk:
            why.append(f"家庭净值包含股票市值，且季度插值会把之后几个月的股市涨跌提前带进当月（与下月美股价格收益相关 "
                       f"{lk['wealth']['next_month']:.2f}，与当月 {lk['wealth']['same_month']:.2f}）")
        add("高", "missing_factor", "不要把估值变化（P/D、CAPE）、家庭净值当成传导因子",
            f"单独加入后样本外 R² 的变化：{parts}。" + ("但" + "；".join(why) + "。" if why else "")
            + "这是用资产价格解释资产价格，情景里也无法独立设定，所以只作对照、不参与选择。",
            oos_delta=endo, leakage=lk)

    win = {c: oos_deltas(its, c) for c in ("window_15", "window_25")}
    win = {c: d for c, d in win.items() if d}
    rb = resid["rolling_betas"]
    us = rb["assets"]["us_equity"]["beta"]["y10"]
    pre = [b for y, b in zip(rb["years"], us) if int(y) <= 1999]
    post = [b for y, b in zip(rb["years"], us) if int(y) >= 2005]
    if win and pre and post:
        helps = [c for c, d in win.items() if all(r["delta"] > wf.MIN_GAIN for r in d)]
        worst = min(r["delta"] for d in win.values() for r in d)
        flip = np.sign(np.mean(pre)) != np.sign(np.mean(post))
        deltas = "；".join(f"最近 {c.split('_')[1]} 年窗口 {_fmt(d)}" for c, d in win.items())
        head = (f"Chow 检验：{resid['summary']['n_chow_reject']}/{len(SCORE_ASSETS)} 个风险资产的系数在 {BREAK_DATE[:4]} 年前后"
                f"显著不同；10 年期收益率上升 1 个标准差时，美股当月的反应从 1999 年前平均 {np.mean(pre):+.2%} 变为 2005 年后 "
                f"{np.mean(post):+.2%}" + ("（股债相关性换了符号）。" if flip else "。"))
        if helps:
            title = "传导系数随时代变化，缩短估计窗口在每个样本外十年都有帮助"
            body = head + f"单独使用时样本外 R² 的变化：{deltas}。"
        else:
            title = "传导系数随时代变化，但不能靠缩短估计窗口解决"
            body = (head + f"但缩短估计窗口不是解药，单独使用时样本外 R² 的变化：{deltas}。"
                    + ("最近窗口只是押注上一个时代会延续，时代一换就大幅失效；" if worst < -0.05 else
                       "最近窗口没有稳定的增益；")
                    + "更稳妥的是保留全样本系数，在情景里用熵池化把权重放到相似时代的月份上。")
        add("高", "tuning", title, body,
            chow_reject=resid["summary"]["chow_reject"], us_equity_y10_beta_pre1999=float(np.mean(pre)),
            us_equity_y10_beta_post2005=float(np.mean(post)), oos_delta=win)

    nn = {c: oos_deltas(its, c) for c in ("mlp16", "mlp32")}
    nn = {c: d for c, d in nn.items() if d}
    if nn:
        worth = all(r["delta"] > wf.MIN_GAIN for d in nn.values() for r in d)
        reset = [next(f for f in it["findings"] if f["id"] == "nonlinearity")["n_reject"] for it in its]
        gains = {c: [r["period"] for r in d if r["delta"] > wf.MIN_GAIN] for c, d in nn.items()}
        gain_txt = "，".join(f"{c[3:]} 个隐藏单元" + (f"只在 {'、'.join(g)} 有增益" if g else "没有一个十年有增益")
                             for c, g in gains.items())
        add("中", "neural_net", "小型神经网络有稳定增益，可以采用" if worth else "小型神经网络目前不值得",
            "单独使用时样本外 R² 的变化：" + "；".join(f"{c[3:]} 个隐藏单元 {_fmt(d)}" for c, d in nn.items())
            + "。RESET 检验显示非线性的资产数（各轮训练期）：" + "、".join(f"{r}/{len(SCORE_ASSETS)}" for r in reset) + "。"
            + ("增益在每个样本外十年都成立。" if worth else
               gain_txt + "，没有稳定重现；而且神经网络没有可解释的 β，情景页的因子归因和传导链条会失去依据。"),
            oos_delta=nn, reset_reject=reset)

    gold = resid["assets"]["gold"]
    others = float(np.mean([resid["assets"][a]["acf"][0] for a in SCORE_ASSETS if a != "gold"]))
    lags = {c: oos_deltas(its, c) for c in ("lags1", "lags3")}
    lags = {c: d for c, d in lags.items() if d}
    if gold["acf"][0] > 0.15:
        lag_helps = any(all(r["delta"] > wf.MIN_GAIN for r in d) for d in lags.values())
        add("中", "missing_mechanism", "黄金的残差自相关来自月均价数据，不是传导滞后：建议改用月末金价",
            f"全样本黄金残差一阶自相关 {gold['acf'][0]:.2f}（Ljung-Box p={gold['ljung_box_p']:.2g}），其他风险资产平均 "
            f"{others:.2f}。月均价把上月末到本月末的变化摊平，随机游走的月均值变化本身就有约 0.25 的一阶自相关。"
            + ("加滞后项后样本外 R² 的变化：" + "；".join(f"{c[4:]} 阶 {_fmt(d)}" for c, d in lags.items())
               + ("，滞后项只会去拟合这个数据伪影。" if not lag_helps else "，滞后项的增益里有一部分来自这个数据伪影。")
               if lags else "")
            + "应在数据层改用月末金价。",
            gold_acf1=gold["acf"][0], others_acf1=others, oos_delta=lags)

    drift = [next(f for f in it["findings"] if f["id"] == "drift_bias") for it in its]
    add("中", "tuning", "宏观因子只解释一小部分月度波动，长期收益要靠 CMA 而不是历史截距",
        f"全样本风险资产平均 R² {resid['summary']['r2_risk_avg']:.0%}，基准模型样本外：{base_txt}。验证期预测偏差最大的："
        + "；".join(f"{_period(it['validate'])} {NAMES[d['worst']]} 每年 {d['per_asset'][d['worst']]['bias_annual']:+.1%}"
                   for it, d in zip(its, drift))
        + "。漂移偏差在月度误差里占比很小，但复利 10 年后就是终值误差的主要来源，所以情景引擎里的长期漂移"
          "不应照搬训练期的平均收益，而要由资本市场假设（CMA）单独设定；历史漂移与估值调整两种 CMA 的样本外对比见回测页。",
        r2_risk_avg=resid["summary"]["r2_risk_avg"],
        bias={it["iter"]: {"asset": d["worst"], "bias_annual": d["per_asset"][d["worst"]]["bias_annual"]}
              for it, d in zip(its, drift)})

    cands = {c["id"]: c for c in res["candidates"]}
    factors = {cid: oos_deltas(its, cid) for cid, c in cands.items()
               if c["eligible"] and (c["change"].get("add_factors") or c["change"].get("shock_cols"))}
    factors = {c: d for c, d in factors.items() if d}
    if factors:
        robust = [c for c, d in factors.items() if all(r["delta"] > wf.MIN_GAIN for r in d)]
        label = {c: FACTOR_NAMES.get(c.removeprefix("add_"), cands[c]["hypothesis"]) for c in factors}
        add("低", "missing_factor",
            f"「{'、'.join(label[c] for c in robust)}」在每个样本外十年都稳定提升" if robust
            else "没有一个候选因子在每个样本外十年都稳定提升",
            "单独加入后样本外 R² 的变化：" + "；".join(f"{label[c]} {_fmt(d)}" for c, d in factors.items())
            + "。" + ("可以作为可选因子加入。" if robust else "缺一个因子不是预测误差的主要原因。"),
            robust=robust, oos_delta=factors)

    reg_d = oos_deltas(its, "regime_inflation")
    if reg_d:
        reg = [next(f for f in it["findings"] if f["id"] == "regime") for it in its]
        n_rej = [r.get("n_reject", 0) for r in reg]
        stable = all(r["delta"] > wf.MIN_GAIN for r in reg_d)
        title = ("通胀状态依赖的传导在样本外稳定有效" if stable else
                 "通胀状态依赖的传导：" + ("统计上存在，但样本外不稳定" if max(n_rej) >= MIN_FLAG
                                           else "统计上不显著，样本外也不稳定"))
        add("低", "missing_mechanism", title,
            "通胀状态交互项显著的资产数（各轮训练期）：" + "、".join(f"{r}/{len(SCORE_ASSETS)}" for r in n_rej)
            + f"；单独加入后样本外 R² 的变化：{_fmt(reg_d)}。高通胀月份主要来自 1970 年代，交互项的估计误差很大；"
              "更稳妥的做法是在情景里用熵池化把权重放到相应时代的月份上。",
            n_reject=n_rej, oos_delta=reg_d)
    order = {"高": 0, "中": 1, "低": 2}
    return sorted(recs, key=lambda r: order[r["priority"]])[:8]


# ---------------------------------------------------------------- entry point

def run(ctx: EngineContext) -> dict:
    panel = ctx.panel
    end = _ym(panel.index.max())
    res = wf.run_walkforward(panel, ModelSpec(), diagnose=diagnose)
    final = ModelSpec.from_dict(res["final"]["spec"])
    resid = residual_analysis(panel, final, wf.build_candidates(panel, (config.MODEL_START, end)))
    fin = {**res["final"], "trained_on": [config.MODEL_START, end]}
    selection = {
        "spec": final.to_dict(), "describe": final.describe(), "source": "walkforward", "changes": fin["changes"],
        "val_score": fin["val_score"], "test_score": fin["test_score"],
        "base_val_score": fin["base_val_score"], "base_test_score": fin["base_test_score"],
        "test_used_for_choice": fin["test_used_for_choice"], "r2_benchmark": res["r2_benchmark"],
        "reason": fin["reason"], "selected_on": fin["selected_on"],
        "created": dt.date.today().isoformat(),
    }
    jsonio.dump(selection, SELECTION_PATH)
    return {
        "questions": wf.QUESTIONS, "candidates": res["candidates"], "score_assets": res["score_assets"],
        "min_gain": res["min_gain"], "max_accept": res["max_accept"], "require_test_gain": res["require_test_gain"],
        "r2_benchmark": res["r2_benchmark"], "units": UNITS, "notes": NOTES,
        "iterations": res["iterations"], "final": fin, "residual_analysis": resid,
        "recommendations": recommendations(res, resid),
    }


def main() -> None:
    ctx = EngineContext.load(selection=False)
    jsonio.dump(run(ctx), config.RESULTS_DIR / "diagnostics.json")


if __name__ == "__main__":
    main()
