"""The statistical tests of the missing-factor audit (缺失因子审计): the validated core of the research harness.

The research round of 2026-11-01 tested its candidates with a stand-alone harness on a copy of the panel; its
verifier re-derived every adopted number with the same functions. This module ports the parts the audit step
re-runs, on any panel it is given (nothing is read from disk here):

  Bench.transmission  each candidate alone on top of a plain linear base spec, per asset and test decade: the
                      change in out-of-sample R^2 on identical months with walk-forward semantics (fit on the
                      training window, predict the test months from the actual same-month factors, R^2 against
                      the training-window mean price return). delta_same_sample refits the base on the
                      candidate's own training months, so a late series is not credited with a shorter window;
                      dm_t / cw_t are the Diebold-Mariano and Clark-West t of that same-sample comparison
                      (Newey-West), and the pooled table tests the 10-asset score month by month.
  stouffer            the decades combined: sum of t over the decades with >= 120 training months / sqrt(k).
  bh                  Benjamini-Hochberg false-discovery control.
  circular_placebo    the candidate's own observed values shifted in time (deterministic shifts per candidate).
  refutation          the verifier's battery: decades, ex-crisis, rolling 10-year windows, one month later and
                      one month earlier (a look-ahead placebo), circular placebo.
  predictive_test     a state known at the origin month on top of a const + forecast.KEY_STATES baseline,
                      12-month targets, expanding window, Clark-West and DM (monthly and year-end origins).
  oos_residuals       expanding-window out-of-sample residuals of any linear spec (per-asset extra factors and
                      fit windows through the engine's own AssetFactorModel).
  residual_pca        principal components of a residual panel.
  spec_score/compare  a spec scored like walkforward.evaluate / compare (per-asset equations included).
  external_r2         the same windows for a return series outside the model (out-of-universe industries).
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd
from scipy import stats

from .. import config
from ..engine.factor_model import MODEL_ASSETS, STRUCTURAL_ASSETS, AssetFactorModel, ModelSpec, asset_extras, design_matrix

SCORE_ASSETS = [a for a in MODEL_ASSETS if a not in STRUCTURAL_ASSETS]   # = walkforward.SCORE_ASSETS
MIN_TRAIN = 120                       # decades with fewer training months do not enter the Stouffer z
PLACEBO_SEED = 20261101               # with crc32(candidate id): one fixed set of shifts per candidate
PLACEBO_N = 100
HORIZON = 12

# Test decades, each refit on its own training window (T1/T2 = walk-forward iteration 1 validation / test,
# T3 = iteration 2 test).
DECADES = {
    "T1": {"train": ("1973-01", "1994-12"), "test": ("1995-01", "2004-12")},
    "T2": {"train": ("1973-01", "2004-12"), "test": ("2005-01", "2014-12")},
    "T3": {"train": ("1973-01", "2014-12"), "test": ("2015-01", None)},      # None = the panel's last month
}
DECADE_LABELS = {"T1": "1995–2004", "T2": "2005–14", "T3": "2015–26"}
# rolling 10-year test windows, each trained from 1973 to the year before
ROLLING = {f"R{y}": {"train": ("1973-01", f"{y - 1}-12"), "test": (f"{y}-01", f"{y + 9}-12" if y < 2015 else None)}
           for y in (1990, 1995, 2000, 2005, 2010, 2015)}
CRISES = (("2008-01", "2009-12"), ("2020-01", "2020-12"))


# ============================================================================ statistics

def nw_lag(n: int) -> int:
    """Newey-West (1994) rule of thumb floor(4 (n/100)^(2/9)): 4 for 120-264 months, 5 for about 500."""
    return int(math.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def _nw_sum(v: np.ndarray, lag: int) -> np.ndarray:
    """sum_t v_t^2 + 2 sum_{l<=lag} (1 - l/(lag+1)) sum_t v_t v_{t-l}, per column (no demeaning)."""
    v = np.asarray(v, float)
    if v.ndim == 1:
        v = v[:, None]
    s = (v * v).sum(axis=0)
    for lg in range(1, lag + 1):
        if lg >= len(v):
            break
        s = s + 2.0 * (1.0 - lg / (lag + 1.0)) * (v[lg:] * v[:-lg]).sum(axis=0)
    return s


def nw_mean_t(d, lag: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean of each column of d, its Newey-West t statistic and the lag used."""
    d = np.asarray(d, float)
    if d.ndim == 1:
        d = d[:, None]
    n = len(d)
    if n < 3:
        return np.full(d.shape[1], np.nan), np.full(d.shape[1], np.nan), 0
    lag = nw_lag(n) if lag is None else int(lag)
    m = d.mean(axis=0)
    se = np.sqrt(np.maximum(_nw_sum(d - m, lag) / (n * n), 1e-300))
    return m, m / se, lag


def p_one(t, n: int):
    """One-sided p of H1: t > 0 (the candidate is better), t(n-1)."""
    return stats.t.sf(t, df=max(n - 1, 1))


def p_two(t, n: int):
    return 2.0 * stats.t.sf(np.abs(t), df=max(n - 1, 1))


def ols_hac(y, X, lag: int | None = None, const: bool = True) -> dict:
    """OLS of y on X (+ constant) with Newey-West standard errors; rows with any NaN are dropped.
    Returns coef / t / se (without the constant), n, lag and r2."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    out = {"coef": np.full(k, np.nan), "t": np.full(k, np.nan), "se": np.full(k, np.nan), "n": int(n),
           "lag": 0, "r2": np.nan}
    if n < k + 5:
        return out
    X1 = np.column_stack([np.ones(n), X]) if const else X
    try:
        Qinv = np.linalg.inv(X1.T @ X1)
    except np.linalg.LinAlgError:
        return out
    beta = Qinv @ (X1.T @ y)
    e = y - X1 @ beta
    lag = nw_lag(n) if lag is None else int(lag)
    off = 1 if const else 0
    se = np.array([math.sqrt(max(float(_nw_sum((X1 @ Qinv[:, j + off]) * e, lag)[0]), 1e-300)) for j in range(k)])
    sst = ((y - y.mean()) ** 2).sum()
    out.update({"coef": beta[off:], "se": se, "t": beta[off:] / se, "lag": lag,
                "r2": float(1.0 - (e ** 2).sum() / sst) if sst > 0 else np.nan})
    return out


def bh(pvals, q: float = 0.10) -> pd.DataFrame:
    """Benjamini-Hochberg at false-discovery rate q. NaN = not tested (left out of m).
    Returns p, q_value (adjusted, monotone) and reject (q_value <= q) on the input's index."""
    s = pd.Series(pvals, dtype=float) if isinstance(pvals, dict) else (
        pvals.astype(float) if isinstance(pvals, pd.Series) else pd.Series(np.asarray(pvals, float)))
    out = pd.DataFrame({"p": s, "q_value": np.nan, "reject": False}, index=s.index)
    ok = s.notna().to_numpy()
    p = s.to_numpy()[ok]
    m = len(p)
    if m == 0:
        return out
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty(m)
    qv[order] = np.minimum(adj, 1.0)
    out.loc[ok, "q_value"] = qv
    out.loc[ok, "reject"] = qv <= q
    out["reject"] = out["reject"].astype(bool)
    return out


# ============================================================================ helpers

def month_start(panel: pd.DataFrame) -> pd.DataFrame:
    """The panel with a month-start index (the panel's own convention; a copy)."""
    P = panel.copy()
    P.index = pd.DatetimeIndex(P.index).to_period("M").to_timestamp()
    return P


def window_rows(index: pd.DatetimeIndex, start, end) -> np.ndarray:
    """Boolean mask of the months in [start, end] (end None = the last month)."""
    m = index >= pd.Timestamp(start)
    if end is not None:
        m &= index <= pd.Timestamp(end)
    return np.asarray(m)


def crisis_free(index: pd.DatetimeIndex, crises=CRISES) -> pd.Series:
    """True outside the crisis windows (2008-01..2009-12 and 2020)."""
    m = pd.Series(True, index=index)
    for a, b in crises:
        m[(index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))] = False
    return m


def ym(ts) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m")


def _decades(decades) -> dict:
    if decades is None:
        return DECADES
    if isinstance(decades, dict):
        return decades
    return {k: DECADES[k] for k in decades}


class _Fit:
    """One OLS / ridge fit of every MODEL_ASSET on the same months (AssetFactorModel._fit_linear)."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, rows: np.ndarray, ridge: float = 0.0):
        mask = rows & np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        n, k = int(mask.sum()), X.shape[1]
        if n < 3 * (k + 2):
            raise ValueError(f"not enough months to fit: {n} for {k} features")
        x, y = X[mask], Y[mask]
        xm, xsd = x.mean(axis=0), x.std(axis=0)
        xsd[xsd < 1e-12] = 1.0
        z = (x - xm) / xsd
        ym_ = y.mean(axis=0)
        A = z.T @ z + ridge * n * np.eye(k)
        if np.linalg.cond(A) > 1e12:
            raise np.linalg.LinAlgError("the design is (nearly) collinear: the candidate duplicates a base factor")
        self.coef = np.linalg.solve(A, z.T @ (y - ym_)) / xsd[:, None]
        self.intercept = ym_ - xm @ self.coef
        self.mask = mask
        self.X = X

    def predict(self, rows: np.ndarray, cols: list[int]) -> np.ndarray:
        return self.X[rows] @ self.coef[:, cols] + self.intercept[cols]

    def coef_hac(self, Y: np.ndarray, cols: list[int], assets_idx: list[int]) -> np.ndarray:
        """Newey-West standard errors of the coefficients `cols` per asset, on the fit months."""
        x = self.X[self.mask]
        X1 = np.column_stack([np.ones(len(x)), x])
        E = Y[self.mask][:, assets_idx] - (x @ self.coef[:, assets_idx] + self.intercept[assets_idx])
        lag = nw_lag(len(x))
        try:
            Qinv = np.linalg.inv(X1.T @ X1)
        except np.linalg.LinAlgError:
            return np.full((len(cols), len(assets_idx)), np.nan)
        return np.vstack([np.sqrt(np.maximum(_nw_sum((X1 @ Qinv[:, c + 1])[:, None] * E, lag), 1e-300))
                          for c in cols])


# ============================================================================ transmission

class Bench:
    """A plain linear base spec on one panel: the comparator every transmission test adds a candidate to."""

    def __init__(self, panel: pd.DataFrame, spec: ModelSpec | None = None):
        spec = spec or ModelSpec()
        if spec.kind != "linear" or spec.lags or spec.regime or spec.window_years or asset_extras(spec) \
                or spec.asset_start:
            raise ValueError("Bench takes a plain linear spec (no lags, regime, window or per-asset equations)")
        self.panel = month_start(panel)
        self.spec = spec
        self.index = self.panel.index
        self.Y = self.panel[[f"{spec.target}_{a}" for a in MODEL_ASSETS]].to_numpy(float)
        self.Xb_frame = design_matrix(self.panel, spec)
        self.Xb = self.Xb_frame.to_numpy(float)
        self._fits: dict = {}

    def align(self, s) -> pd.Series:
        """A monthly series on the panel's month-start index (two values in one month raise)."""
        s = pd.Series(s).copy()
        idx = s.index.to_timestamp() if isinstance(s.index, pd.PeriodIndex) else pd.DatetimeIndex(s.index)
        idx = idx.to_period("M").to_timestamp()
        if idx.has_duplicates:
            raise ValueError(f"{s.name}: several values in one month")
        s.index = idx
        return pd.to_numeric(s, errors="coerce").astype(float).reindex(self.index)

    def rows(self, start, end) -> np.ndarray:
        return window_rows(self.index, start, end)

    def _base_fit(self, key, rows, Xb):
        k = (key, Xb.shape[1])
        if k not in self._fits:
            self._fits[k] = _Fit(Xb, self.Y, rows, self.spec.ridge)
        return self._fits[k]

    def design(self, cand, drop=()) -> tuple[np.ndarray, list[int]]:
        """The base design without `drop` and with the candidate column(s) appended; returns (X, their indices)."""
        bad = [d for d in drop if d not in self.spec.factors]
        if bad:
            raise ValueError(f"drop={bad}: not factors of the base spec")
        keep = [c for c in self.Xb_frame.columns if c not in drop]
        if cand is None:
            return self.Xb_frame[keep].to_numpy(float), []
        cdf = cand if isinstance(cand, pd.DataFrame) else pd.DataFrame({"cand": cand})
        cdf = pd.DataFrame({f"__c{j}": self.align(cdf[c]) for j, c in enumerate(cdf.columns)}, index=self.index)
        X = pd.concat([self.Xb_frame[keep], cdf], axis=1).to_numpy(float)
        return X, list(range(len(keep), X.shape[1]))

    def transmission(self, cands: dict, drop=(), assets=None, decades=None, score_mask=None,
                     train_exclude=None, base_drop=False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Each candidate alone on top of the base, per decade x asset; returns (rows, pooled).

        rows: candidate, decade, asset, r2_base, r2_base_ss, r2_cand, delta (vs the base fitted on the whole
          training window), delta_ss (vs the base refitted on the candidate's training months), dm_t / cw_t
          (same-sample Diebold-Mariano and Clark-West t, Newey-West, positive = candidate better), dm_t_raw,
          coef / coef_t (the candidate in the decade's training fit, HAC t), n_test, n_train, train_start,
          train_end, too_short, nw_lag. pooled: candidate x decade, score_ss (mean delta_ss over the score
          assets) and pooled_t (NW t of the month-by-month score gain).
        score_mask: months allowed in scoring (bool Series); train_exclude: months removed from every training
        window; base_drop: compare with the base that also drops `drop` (else with the full base)."""
        drop = tuple(drop or ())
        assets = list(assets or SCORE_ASSETS)
        aidx = [MODEL_ASSETS.index(a) for a in assets]
        act_all = self.panel[[f"p_{a}" for a in assets]].to_numpy(float)
        Xb = self.design(None, drop)[0] if (base_drop and drop) else self.Xb
        decs = _decades(decades)
        smask = None if score_mask is None else score_mask.reindex(self.index).fillna(False).to_numpy(bool)
        tex = None if train_exclude is None else train_exclude.reindex(self.index).fillna(False).to_numpy(bool)
        rows, pooled = [], []
        for name, cand in cands.items():
            try:
                Xc, cidx = self.design(cand, drop)
            except (ValueError, KeyError) as err:
                rows.append({"candidate": name, "error": str(err)})
                continue
            for key, d in decs.items():
                tr = self.rows(*d["train"])
                if tex is not None:
                    tr = tr & ~tex
                te = self.rows(*d["test"])
                try:
                    fb = (self._base_fit((key, d["train"], drop if base_drop else ()), tr, Xb) if tex is None
                          else _Fit(Xb, self.Y, tr, self.spec.ridge))
                    fc = _Fit(Xc, self.Y, tr, self.spec.ridge)
                    fbs = _Fit(Xb, self.Y, fc.mask, self.spec.ridge) if fc.mask.sum() < fb.mask.sum() else fb
                except (ValueError, np.linalg.LinAlgError) as err:
                    rows.append({"candidate": name, "decade": key, "error": str(err)})
                    continue
                pb, pc, pbs = fb.predict(te, aidx), fc.predict(te, aidx), fbs.predict(te, aidx)
                y = act_all[te]
                common = (np.isfinite(pb).all(axis=1) & np.isfinite(pc).all(axis=1) & np.isfinite(y).all(axis=1)
                          & np.isfinite(pbs).all(axis=1))
                if smask is not None:
                    common &= smask[te]
                n_te = int(common.sum())
                if n_te < 12:
                    rows.append({"candidate": name, "decade": key, "error": f"only {n_te} test months"})
                    continue
                bench = np.nanmean(act_all[self.rows(*d["train"])], axis=0)
                yy = y[common]
                eb, ec, ebs = yy - pb[common], yy - pc[common], yy - pbs[common]
                sst = ((yy - bench) ** 2).sum(axis=0)
                r2b = 1.0 - (eb ** 2).sum(axis=0) / sst
                r2c = 1.0 - (ec ** 2).sum(axis=0) / sst
                r2bs = 1.0 - (ebs ** 2).sum(axis=0) / sst
                dl = ebs ** 2 - ec ** 2
                _, dm_t, lag = nw_mean_t(dl)
                _, cw_t, _ = nw_mean_t(dl + (pbs[common] - pc[common]) ** 2)
                _, dm_t_raw, _ = nw_mean_t(eb ** 2 - ec ** 2)
                if len(cidx) == 1:
                    se = fc.coef_hac(self.Y, cidx, aidx)
                    coef = fc.coef[np.ix_(cidx, aidx)]
                idx = self.index[fc.mask]
                n_tr = int(fc.mask.sum())
                for j, a in enumerate(assets):
                    rows.append({"candidate": name, "decade": key, "asset": a, "r2_base": r2b[j],
                                 "r2_base_ss": r2bs[j], "r2_cand": r2c[j], "delta": r2c[j] - r2b[j],
                                 "delta_ss": r2c[j] - r2bs[j], "dm_t": dm_t[j], "cw_t": cw_t[j],
                                 "cw_p": float(p_one(cw_t[j], n_te)), "dm_t_raw": dm_t_raw[j],
                                 "coef": coef[0, j] if len(cidx) == 1 else np.nan,
                                 "coef_t": coef[0, j] / se[0, j] if len(cidx) == 1 else np.nan,
                                 "n_test": n_te, "n_train": n_tr, "train_start": ym(idx.min()),
                                 "train_end": ym(idx.max()), "too_short": n_tr < MIN_TRAIN, "nw_lag": lag})
                sc = [j for j, a in enumerate(assets) if a in SCORE_ASSETS]
                if sc:
                    D = (dl[:, sc] / sst[sc]).mean(axis=1)
                    _, pt, _ = nw_mean_t(D)
                    pooled.append({"candidate": name, "decade": key, "score_ss": float(D.sum()),
                                   "score": float(((eb[:, sc] ** 2 - ec[:, sc] ** 2) / sst[sc]).mean(axis=1).sum()),
                                   "pooled_t": float(pt[0]), "n_test": n_te, "n_train": n_tr,
                                   "train_start": ym(idx.min()), "too_short": n_tr < MIN_TRAIN})
        return pd.DataFrame(rows), pd.DataFrame(pooled)


def stouffer(values, n_train=None, min_train: int = MIN_TRAIN) -> tuple[float, int]:
    """Stouffer z = sum t / sqrt(k) over the decades with at least `min_train` training months."""
    t = np.asarray(values, float)
    ok = np.isfinite(t)
    if n_train is not None:
        ok &= np.asarray(n_train) >= min_train
    k = int(ok.sum())
    return (float(t[ok].sum() / math.sqrt(k)) if k else float("nan")), k


def combined_z(rows: pd.DataFrame, pooled: pd.DataFrame, assets) -> dict:
    """{asset: (z, k)} from the same-sample DM t, plus 'common' from the pooled score t."""
    out = {}
    for a in assets:
        g = rows[rows["asset"] == a]
        out[a] = stouffer(g["dm_t"], g["n_train"])
    out["common"] = stouffer(pooled["pooled_t"], pooled["n_train"])
    return out


def placebo_shifts(cand_id: str, n_obs: int, n: int = PLACEBO_N) -> np.ndarray:
    """The deterministic circular shifts of a candidate: seed [PLACEBO_SEED, crc32(id)], 24..n_obs-24 months."""
    rng = np.random.default_rng([PLACEBO_SEED, zlib.crc32(cand_id.encode("utf-8"))])
    return rng.integers(24, n_obs - 24, size=n)


def circular_placebo(bench: Bench, cand_id: str, x: pd.Series, assets, drop=(), n: int = PLACEBO_N,
                     z_obs: dict | None = None) -> dict:
    """Circular shifts of the candidate's observed values (same distribution and autocorrelation, wrong
    months). Per asset (and 'common', the pooled score): n, the observed z, the empirical one-sided p
    (count(z_shift >= z_obs) + 1) / (n + 1) and the 95th percentile of the shifted z."""
    obs = bench.align(x).dropna()
    vals = obs.to_numpy()
    cands = {f"plc{i}": pd.Series(np.roll(vals, int(sh)), index=obs.index)
             for i, sh in enumerate(placebo_shifts(cand_id, len(vals), n))}
    rows, pooled = bench.transmission(cands, drop=drop, assets=SCORE_ASSETS)
    zs = {a: [] for a in list(assets) + ["common"]}
    for c in cands:
        g = rows[rows["candidate"] == c] if len(rows) else rows
        for a in assets:
            ga = g[(g["asset"] == a) & (g["n_train"] >= MIN_TRAIN)]
            if len(ga):
                zs[a].append(float(ga["dm_t"].sum() / math.sqrt(len(ga))))
        gp = pooled[(pooled["candidate"] == c) & (pooled["n_train"] >= MIN_TRAIN)] if len(pooled) else pooled
        if len(gp):
            zs["common"].append(float(gp["pooled_t"].sum() / math.sqrt(len(gp))))
    if z_obs is None:
        r0, p0 = bench.transmission({cand_id: x}, drop=drop, assets=SCORE_ASSETS)
        z_obs = {k: v[0] for k, v in combined_z(r0, p0, assets).items()}
    out = {}
    for a, v in zs.items():
        v = np.asarray(v, float)
        zo = z_obs[a]
        out[a] = {"n": int(len(v)), "z_obs": zo,
                  "p": float((np.sum(v >= zo) + 1) / (len(v) + 1)) if len(v) else None,
                  "q95": float(np.percentile(v, 95)) if len(v) else None}
    return out


def summarize(rows: pd.DataFrame, pooled: pd.DataFrame, assets) -> dict:
    """{asset: {decade: {delta, delta_ss, dm_t, cw_t, n_train, n_test, coef_t}}, 'common': {decade: {score_ss,
    pooled_t, n_train, n_test}}} (the verifier's battery layout)."""
    out = {}
    for a in assets:
        g = rows[rows["asset"] == a].set_index("decade")
        out[a] = {d: {"delta": float(g.loc[d, "delta"]), "delta_ss": float(g.loc[d, "delta_ss"]),
                      "dm_t": float(g.loc[d, "dm_t"]), "cw_t": float(g.loc[d, "cw_t"]),
                      "n_train": int(g.loc[d, "n_train"]), "n_test": int(g.loc[d, "n_test"]),
                      "coef_t": float(g.loc[d, "coef_t"])} for d in g.index}
    gp = pooled.set_index("decade")
    out["common"] = {d: {"score_ss": float(gp.loc[d, "score_ss"]), "pooled_t": float(gp.loc[d, "pooled_t"]),
                         "n_train": int(gp.loc[d, "n_train"]), "n_test": int(gp.loc[d, "n_test"])}
                     for d in gp.index}
    return out


def refutation(bench: Bench, cand_id: str, x: pd.Series, assets, drop=(), placebo_n: int = PLACEBO_N) -> dict:
    """The verifier's refutation battery for one candidate: the three decades, the score without the crisis
    months (and also trained without them), rolling 10-year windows, the candidate one month later (x[t-1])
    and one month earlier (x[t+1], a look-ahead placebo), and the circular-shift placebo."""
    x = bench.align(x)
    excl = crisis_free(bench.index)
    out = {"assets": list(assets), "drop": list(drop)}
    r, p = bench.transmission({cand_id: x}, drop=drop)
    out["base"] = summarize(r, p, assets)
    out["z"] = {k: list(v) for k, v in combined_z(r, p, assets).items()}
    r, p = bench.transmission({cand_id: x}, drop=drop, score_mask=excl)
    out["ex_crisis_score"] = summarize(r, p, assets)
    r, p = bench.transmission({cand_id: x}, drop=drop, score_mask=excl, train_exclude=~excl)
    out["ex_crisis_train_score"] = summarize(r, p, assets)
    r, p = bench.transmission({cand_id: x}, drop=drop, decades=ROLLING)
    out["rolling"] = summarize(r, p, assets)
    r, p = bench.transmission({cand_id: x.shift(1)}, drop=drop)
    out["lag1"] = summarize(r, p, assets)
    r, p = bench.transmission({cand_id: x.shift(-1)}, drop=drop)
    out["lead1"] = summarize(r, p, assets)
    out["placebo_circular"] = circular_placebo(bench, cand_id, x, assets, drop, placebo_n,
                                               z_obs={k: v[0] for k, v in out["z"].items()})
    return out


# ============================================================================ predictive test

def _expanding_forecasts(W: np.ndarray, y: np.ndarray, start_i: int, origins_i: np.ndarray,
                         min_train: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS forecasts at each origin index t from the pairs s in [start_i, t-HORIZON] (all finite); W includes
    the constant. Returns (forecast, the mean of the training targets, n_train)."""
    ok = np.isfinite(W).all(axis=1) & np.isfinite(y)
    ok[:start_i] = False
    W = W.copy()
    fin = np.isfinite(W).all(axis=1)
    if fin.any():            # sample-wide scaling (the constant makes OLS invariant to it; conditioning only)
        m, s = W[fin].mean(axis=0), W[fin].std(axis=0)
        var = s > 1e-12
        W[:, var] = (W[:, var] - m[var]) / s[var]
    p = W.shape[1]
    Wz = np.where(ok[:, None], W, 0.0)
    yz = np.where(ok, y, 0.0)
    cxx = np.cumsum(Wz[:, :, None] * Wz[:, None, :], axis=0)
    cxy = np.cumsum(Wz * yz[:, None], axis=0)
    cy = np.cumsum(yz)
    cn = np.cumsum(ok.astype(float))
    fc = np.full(len(origins_i), np.nan)
    mu = np.full(len(origins_i), np.nan)
    last = origins_i - HORIZON
    valid = last >= 0
    lv = np.where(valid, last, 0)
    n = np.where(valid, cn[lv], 0)
    good = valid & (n >= max(min_train, 3 * (p + 1))) & np.isfinite(W[origins_i]).all(axis=1)
    if good.any():
        beta = np.einsum("nij,nj->ni", np.linalg.pinv(cxx[lv[good]]), cxy[lv[good]])
        fc[good] = (W[origins_i[good]] * beta).sum(axis=1)
        mu[good] = cy[lv[good]] / n[good]
    return fc, mu, n.astype(int)


def _oos_stats(y, f_base, f_cand, mu, lag) -> dict:
    ok = np.isfinite(y) & np.isfinite(f_base) & np.isfinite(f_cand) & np.isfinite(mu)
    n = int(ok.sum())
    if n < 8:
        return {"n": n}
    y, fb, fc, m = y[ok], f_base[ok], f_cand[ok], mu[ok]
    eb, ec, em = y - fb, y - fc, y - m
    sse_b, sse_c, sse_m = (eb ** 2).sum(), (ec ** 2).sum(), (em ** 2).sum()
    _, t_i, _ = nw_mean_t(eb ** 2 - ec ** 2, lag)
    _, cw_i, _ = nw_mean_t(eb ** 2 - ec ** 2 + (fb - fc) ** 2, lag)
    return {"n": n, "r2_base": float(1 - sse_b / sse_m), "r2_cand": float(1 - sse_c / sse_m),
            "r2_incr": float(1 - sse_c / sse_b), "dm_t": float(t_i[0]), "dm_p": float(p_one(t_i, n)[0]),
            "cw_t": float(cw_i[0]), "cw_p": float(p_one(cw_i, n)[0]), "nw_lag": lag}


def predictive_targets(panel: pd.DataFrame) -> dict:
    """Next-12-month targets at each origin: macro changes v[t+12]-v[t] (log for oil) and next-12-month log total
    returns of every asset except cash (the research harness's 19 targets)."""
    P = panel
    macro = {"unrate": ("lvl_unrate", "diff"), "cpi_yoy": ("s_cpi_yoy", "diff"), "ff": ("lvl_ff", "diff"),
             "ip_yoy": ("s_ip_yoy", "diff"), "spread": ("s_spread", "diff"), "y10": ("lvl_y10", "diff"),
             "oil": ("lvl_oil", "log")}
    out = {}
    for t, (col, kind) in macro.items():
        v = P[col].astype(float)
        out[t] = np.log(v.shift(-HORIZON) / v) if kind == "log" else v.shift(-HORIZON) - v
    for a in config.ASSET_IDS:
        if a != "cash":
            out[f"ret_{a}"] = np.log1p(P[f"r_{a}"].astype(float)).rolling(HORIZON).sum().shift(-HORIZON)
    return out


def predictive_test(panel: pd.DataFrame, candidates_known_at: dict, targets, first_origin: str = "1985-01",
                    split: str = "2004-12", ffill_limit: int = 12, min_train: int = MIN_TRAIN,
                    train_start: str = config.MODEL_START, lag_monthly: int = 12, lag_year_end: int = 1) -> pd.DataFrame:
    """Does a state known at the origin forecast the next 12 months on top of const + forecast.KEY_STATES?

    candidates_known_at: {name: Series dated at the month by which the value is published}, held forward up to
    `ffill_limit` months. At each origin t the baseline and the candidate model are fitted on the same pairs
    s in [train_start, t-12] (where the candidate exists); benchmark = the prevailing mean. Rows per candidate x
    target x origins (monthly from first_origin, NW lag 12; year_end = forecast.forecast_origins, NW lag 1) x
    period (all / h1 <= split / h2): n, r2_base, r2_cand (vs the mean), r2_incr = 1 - SSE_cand/SSE_base, dm_t /
    dm_p, cw_t / cw_p (Clark-West, one-sided). KEY_STATES are today's vintage at the month they describe."""
    from .forecast import KEY_STATES, forecast_origins
    P = month_start(panel)
    idx = P.index
    Yt = predictive_targets(P)
    S = P[[f"s_{k}" for k in KEY_STATES]].ffill().to_numpy(float)
    start_i = int(idx.get_indexer([pd.Timestamp(train_start)])[0])
    sets = {"monthly": (idx[idx >= pd.Timestamp(first_origin)], lag_monthly),
            "year_end": (pd.DatetimeIndex(forecast_origins(P)), lag_year_end)}
    cut = pd.Timestamp(split)
    ones = np.ones((len(idx), 1))
    rows = []
    for name, cand in candidates_known_at.items():
        c = pd.Series(cand).copy()
        c.index = pd.DatetimeIndex(c.index).to_period("M").to_timestamp()
        x = c.reindex(idx).ffill(limit=ffill_limit).to_numpy(float)
        for tname in targets:
            y = Yt[tname].to_numpy(float)
            Wb = np.where(np.isfinite(x)[:, None], np.column_stack([ones, S]), np.nan)
            Wc = np.column_stack([ones, S, x])
            for oname, (oset, lag) in sets.items():
                oi = idx.get_indexer(oset)
                oi = oi[oi >= 0]
                fb, mu, _ = _expanding_forecasts(Wb, y, start_i, oi, min_train)
                fc, _, _ = _expanding_forecasts(Wc, y, start_i, oi, min_train)
                yo, dates = y[oi], idx[oi]
                for period, sel in (("all", np.ones(len(oi), bool)), ("h1", dates <= cut), ("h2", dates > cut)):
                    rows.append({"candidate": name, "target": tname, "origins": oname, "period": period,
                                 **_oos_stats(yo[sel], fb[sel], fc[sel], mu[sel], lag)})
    return pd.DataFrame(rows)


# ============================================================================ residuals

def oos_residuals(panel: pd.DataFrame, spec: ModelSpec, first: str = "1985-01", step: int = 12,
                  assets=None, start: str = config.MODEL_START) -> pd.DataFrame:
    """Expanding-window out-of-sample residuals (actual price return minus the prediction from the actual
    same-month factors): at each refit month R = first, first + step, ... the engine's AssetFactorModel is
    fitted on start..R-1 and predicts R..R+step-1. Per-asset extras and fit windows follow the engine
    (an asset_start is honoured once its own window holds MIN_ASSET_WINDOW months)."""
    P = month_start(panel)
    assets = list(assets or MODEL_ASSETS)
    act = P[[f"{spec.target}_{a}" for a in assets]].to_numpy(float)
    out = np.full((len(P), len(assets)), np.nan)
    idx = P.index
    R = pd.Timestamp(first)
    while R <= idx.max():
        model = AssetFactorModel(spec).fit(P, start, R - pd.DateOffset(months=1))
        te = (idx >= R) & (idx <= R + pd.DateOffset(months=step - 1))
        pred = model.predict(P.loc[te])[assets].to_numpy(float)
        out[te] = act[te] - pred
        R = R + pd.DateOffset(months=step)
    return pd.DataFrame(out, index=idx, columns=assets).loc[first:]


def residual_pca(resid: pd.DataFrame, n: int = 3) -> dict:
    """PCA of a residual panel on the correlation matrix (months with every asset). Each component is signed so
    its us_equity loading is >= 0. Returns share (of the first n), share_all, loadings, scores,
    corr_with_us_equity_resid and months."""
    R = resid.dropna()
    Z = (R - R.mean()) / R.std(ddof=0)
    w, V = np.linalg.eigh(np.cov(Z.to_numpy().T, ddof=0))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    cols = list(R.columns)
    ref = cols.index("us_equity") if "us_equity" in cols else int(np.argmax(np.abs(V[:, 0])))
    for k in range(V.shape[1]):
        if V[ref, k] < 0 or (abs(V[ref, k]) < 1e-9 and V[np.argmax(np.abs(V[:, k])), k] < 0):
            V[:, k] *= -1
    scores = pd.DataFrame(Z.to_numpy() @ V[:, :n], index=R.index, columns=[f"PC{k + 1}" for k in range(n)])
    corr_us = ([float(np.corrcoef(scores[c], R["us_equity"])[0, 1]) for c in scores] if "us_equity" in cols else [])
    return {"share": (w[:n] / w.sum()).tolist(), "share_all": (w / w.sum()).tolist(),
            "loadings": pd.DataFrame(V[:, :n], index=cols, columns=scores.columns), "scores": scores,
            "corr_with_us_equity_resid": corr_us, "months": int(len(R))}


def r2_on(y: pd.Series, x: pd.Series) -> float:
    """In-sample R^2 of y on a constant and x (months with both)."""
    d = pd.concat([y, x], axis=1).dropna()
    A = np.column_stack([np.ones(len(d)), d.iloc[:, 1].to_numpy(float)])
    yv = d.iloc[:, 0].to_numpy(float)
    e = yv - A @ np.linalg.lstsq(A, yv, rcond=None)[0]
    return float(1 - (e ** 2).sum() / ((yv - yv.mean()) ** 2).sum())


# ============================================================================ scoring specs

def spec_score(panel: pd.DataFrame, spec: ModelSpec, train: tuple, test: tuple, assets=None,
               months: pd.DatetimeIndex | None = None) -> dict:
    """One spec scored like walkforward.evaluate: fit on `train`, predict `test` from the actual factors, R^2 per
    asset against its mean price return over the whole training window; `months` restricts the scoring."""
    assets = list(assets or SCORE_ASSETS)
    model = AssetFactorModel(spec).fit(panel, train[0], train[1])
    pred = model.predict(panel).loc[test[0]:test[1], assets]
    act = panel.loc[test[0]:test[1], [f"p_{a}" for a in assets]].set_axis(assets, axis=1)
    bench = panel.loc[train[0]:train[1], [f"p_{a}" for a in assets]].mean().set_axis(assets)
    ok = pred.notna().all(axis=1) & act.notna().all(axis=1)
    if months is not None:
        ok &= pred.index.isin(months)
    per = {}
    for a in assets:
        y, p = act.loc[ok, a].to_numpy(float), pred.loc[ok, a].to_numpy(float)
        per[a] = float(1.0 - ((y - p) ** 2).sum() / ((y - bench[a]) ** 2).sum())
    return {"score": float(np.mean(list(per.values()))), "per_asset": per, "n_months": int(ok.sum()),
            "index": pred.index[ok]}


def compare(panel: pd.DataFrame, specs: dict, train: tuple, test: tuple, assets=None) -> dict:
    """Several specs on the months all of them can predict (walkforward.compare)."""
    res = {k: spec_score(panel, s, train, test, assets) for k, s in specs.items()}
    common = res[next(iter(res))]["index"]
    for r in res.values():
        common = common.intersection(r["index"])
    for k, s in specs.items():
        if len(res[k]["index"]) > len(common):
            res[k] = spec_score(panel, s, train, test, assets, months=common)
    return res


def decade_window(key: str, panel: pd.DataFrame, decades=None) -> tuple[tuple, tuple]:
    d = _decades(decades)[key]
    end = d["test"][1] or ym(panel.index.max())
    return tuple(d["train"]), (d["test"][0], end)


def external_r2(panel: pd.DataFrame, features: list[str], y: pd.Series, decades=None) -> dict:
    """OOS R^2 of a return series outside the model (an out-of-universe industry) per decade: OLS on `features`
    fitted on the training months where every feature and every MODEL_ASSETS price return exists, predicted on
    the test months from the actual factors, against the training-window mean of y."""
    P = month_start(panel)
    y = pd.Series(y).copy()
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = y.reindex(P.index)
    X = P[features]
    fit_ok = X.notna().all(axis=1) & P[[f"p_{a}" for a in MODEL_ASSETS]].notna().all(axis=1) & y.notna()
    out = {}
    for key, d in _decades(decades).items():
        t0, t1 = d["train"]
        s0, s1 = d["test"][0], d["test"][1] or ym(P.index.max())
        tr = fit_ok & (P.index >= pd.Timestamp(t0)) & (P.index <= pd.Timestamp(t1))
        A = np.column_stack([np.ones(int(tr.sum())), X.loc[tr].to_numpy(float)])
        b = np.linalg.lstsq(A, y[tr].to_numpy(float), rcond=None)[0]
        te = X.notna().all(axis=1) & y.notna() & (P.index >= pd.Timestamp(s0)) & (P.index <= pd.Timestamp(s1))
        pred = np.column_stack([np.ones(int(te.sum())), X.loc[te].to_numpy(float)]) @ b
        yt = y[te].to_numpy(float)
        bench = float(y.loc[t0:t1].mean())
        out[key] = float(1 - ((yt - pred) ** 2).sum() / ((yt - bench) ** 2).sum())
    return out
