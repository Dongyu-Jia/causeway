"""History priors of the actors (AGENT_DESIGN.md §9.2 item 3; ACTORS v1 work a01–a08, ACTORS_V2 §4 and §9).

The order of prior sources is market odds the user supplies → the model's own A0 bins (engine) → these tables →
uniform (无历史先例). Every table here is either estimated from local files (``data/raw`` and the project panel;
nothing is downloaded) or a dated, sourced decision record counted in code, or a judgment row marked 判断值:

======================  =======================================================================================
Fed                     7-bin ordered probit on the quarterly change (W_A01: n 117, 1987Q3–2026Q3 ex-ELB, McFadden
                        R² 0.20, OOS log score −1.47 vs −1.86 unconditional); a ``schema`` variant without the HLW
                        gap (every covariate is a schema state) is the run-time default; within-branch slopes (e01 B)
Fiscal (Congress)       Sahm triggers → stimulus law within −1/+4 quarters 4/9; shutdowns 7 lapses in 36 FYs; 0
                        defaults in 78 limit actions; A11 binding spells since 2011 (median 136 days)
Tariffs (President)     effective rate = customs / goods imports: normal-times P(|Δ| ≥ 0.5pp) 11/260; active
                        regime 10 up / 4 down of 15 quarters
China                   8/8 retaliations 2018–25; stimulus and depreciation episode counts; Taiwan crises 4 in
                        308 quarters, no blockade
OPEC+                   real-oil crashes → cut within 2 quarters 5/8 (dated OPEC decisions); spike rows
Nature                  oil disruptions 9/215, GPR episodes above the 1985–2019 p95, financial accidents 5/215,
                        pandemic ≈ 2%/yr (Marani et al. 2021)
Belligerents            exit hazard 13/31 from runs of real-oil y/y > 40%
Hyperscalers            boom / normal / bust rows (判断值) anchored on P(real IT investment falls next quarter |
                        Nasdaq 4q ≤ −20%) = 9/16
Voters                  midterm House losses of the president's party when it held the House 7/12 (5/6 since 1994)
======================  =======================================================================================

:func:`build_priors` re-estimates everything from the local data and reports, per table, whether it reproduces
the research round's numbers; :func:`load_priors` returns the built file when present, else the frozen values;
:func:`prior` answers the engine: the probability of each option in a node's menu, with its source and a Chinese
label for the UI (历史基率（外部视角）).
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, jsonio

PRIORS_VERSION = "agents-priors-1 (2026-11-01)"
PRIORS_PATH = config.RESULTS_DIR / "agents_priors.json"


# =====================================================================================================================
# 0. Counting helpers
# =====================================================================================================================

def jeffreys(k: int, n: int, a: float = 0.05) -> list[float | None]:
    """Jeffreys interval for a binomial share (Brown, Cai & DasGupta 2001)."""
    from scipy import stats
    if n <= 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 0.5, n - k + 0.5))
    return [round(lo, 3), round(hi, 3)]


def rate(k: int, n: int) -> dict:
    """{k, n, p, jeffreys_mean, jeffreys95}: p = k / n; the Jeffreys mean (k + ½) / (n + 1) is used when k is 0."""
    return {"k": int(k), "n": int(n), "p": round(k / n, 4) if n else None,
            "jeffreys_mean": round((k + 0.5) / (n + 1), 4) if n >= 0 else None, "jeffreys95": jeffreys(k, n)}


def normalize(p: Mapping[str, float], ids: Sequence[str], floor: float = 0.0) -> dict[str, float]:
    """p restricted to ids, missing entries at `floor`, renormalized (uniform when everything is 0)."""
    vals = {i: max(float(p.get(i, floor)), 0.0) for i in ids}
    tot = sum(vals.values())
    if tot <= 0:
        return {i: 1.0 / len(ids) for i in ids} if ids else {}
    return {i: v / tot for i, v in vals.items()}


# =====================================================================================================================
# 1. Local data (read only; never downloads)
# =====================================================================================================================

class LocalData:
    """Readers for the files the project build and the research import left on disk (FRED cache, NY Fed HLW,
    Caldara–Iacoviello GPR, the monthly panel). A missing file raises FileNotFoundError."""

    def __init__(self, raw_dir: Path | None = None, panel_path: Path | None = None):
        self.raw = Path(raw_dir) if raw_dir else config.RAW_DIR
        self.panel_path = Path(panel_path) if panel_path else config.PANEL_PATH
        self._panel = None

    def fred(self, sid: str, variant: str | None = None) -> pd.Series:
        from ..data import sources
        names = {None: [f"{sid}.json", f"{sid}.csv"], "eop": [f"{sid}_m_eop.json", f"{sid}_eop.csv"],
                 "avg": [f"{sid}_m_avg.json", f"{sid}_avg.csv"]}[variant]
        for name in names:
            p = self.raw / "fred" / name
            if p.exists() and p.stat().st_size > 0:
                s = (sources.read_fred_json if p.suffix == ".json" else sources.read_fred_csv)(p, sid).dropna()
                s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp() if (s.index.day == 1).all() else s.index
                return s
        raise FileNotFoundError(f"本地没有 FRED 序列 {sid}（{'/'.join(names)}）")

    def hlw(self) -> pd.DataFrame:
        p = self.raw / "nyfed" / "Holston_Laubach_Williams_current_estimates.xlsx"
        if not p.exists():
            raise FileNotFoundError("本地没有纽约联储 HLW 估计（nyfed/Holston_Laubach_Williams_current_estimates.xlsx）")
        v = pd.read_excel(p, sheet_name="HLW Estimates", header=None)
        v = v.iloc[6:, [0, 10, 14]].dropna(subset=[0])
        v.columns = ["date", "rstar", "gap"]
        v.index = pd.PeriodIndex(pd.to_datetime(v.pop("date")), freq="Q")
        return v.astype(float)

    def gpr(self) -> pd.Series:
        p = self.raw / "iacoviello" / "data_gpr_export.xls"
        if not p.exists():
            raise FileNotFoundError("本地没有 GPR 数据（iacoviello/data_gpr_export.xls）")
        df = pd.read_excel(p)
        col = [c for c in df.columns if str(c).lower() in ("month", "date")][0]
        s = pd.to_numeric(df["GPR"], errors="coerce")
        s.index = pd.DatetimeIndex(pd.to_datetime(df[col])).to_period("M").to_timestamp()
        return s.dropna()

    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            if not self.panel_path.exists():
                raise FileNotFoundError(f"本地没有项目面板 {self.panel_path.name}")
            P = pd.read_csv(self.panel_path, index_col=0, parse_dates=True)
            P.index = pd.DatetimeIndex(P.index).to_period("M").to_timestamp()
            self._panel = P
        return self._panel


def _monthly(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))


def _yoy(s: pd.Series, k: int = 12) -> pd.Series:
    m = _monthly(s)
    return (100.0 * (m / m.shift(k) - 1.0)).dropna()


def _q_last(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s.groupby(s.index.to_period("Q")).last()


def _q_mean(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s.groupby(s.index.to_period("Q")).mean()


def _q_first_next(s: pd.Series) -> pd.Series:
    """The value in the first month of the next quarter, labelled by the quarter that ended (W_A01: the policy level
    after the quarter's last meeting)."""
    s = s.dropna()
    first = s[s.index.month.isin([1, 4, 7, 10])]
    q = first.copy()
    q.index = (first.index - pd.DateOffset(months=1)).to_period("Q")
    return q


def _episodes(flag: pd.Series) -> list:
    return [q for q, v in flag.items() if v and not flag.get(q - 1, False)]


# =====================================================================================================================
# 2. The Fed: ordered probit (W_A01)
# =====================================================================================================================

FED_EDGES = (-math.inf, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, math.inf)
FED_LABELS = ("<=-0.75", "-0.50", "-0.25", "0", "+0.25", "+0.50", ">=+0.75")
FED_OPTIONS = ("cut_75plus", "cut_50", "cut_25", "hold", "hike_25", "hike_50", "hike_75plus")
FED_MID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.875)


@dataclass(frozen=True)
class OrderedProbit:
    """P(bin k | x) = Φ(c_k − z·b) − Φ(c_{k−1} − z·b), z = (x − mean) / sd (standardised covariates)."""
    cols: tuple[str, ...]
    coef: tuple[float, ...]
    cuts: tuple[float, ...]
    x_mean: tuple[float, ...]
    x_sd: tuple[float, ...]
    n: int
    sample: tuple[str, str]
    counts: tuple[int, ...]
    mcfadden_r2: float
    spec: str
    source: str

    def probs(self, x: Mapping[str, float]) -> np.ndarray:
        from scipy import stats
        z = np.array([(float(x[c]) - m) / s for c, m, s in zip(self.cols, self.x_mean, self.x_sd)])
        xb = float(z @ np.array(self.coef))
        cdf = np.concatenate([[0.0], stats.norm.cdf(np.array(self.cuts) - xb), [1.0]])
        p = np.clip(np.diff(cdf), 0.0, 1.0)
        return p / p.sum()

    def prior(self, x: Mapping[str, float]) -> dict[str, float]:
        return dict(zip(FED_OPTIONS, (float(v) for v in self.probs(x))))

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Mapping) -> "OrderedProbit":
        return cls(**{k: (tuple(d[k]) if isinstance(d[k], list) else d[k]) for k in cls.__dataclass_fields__})


# Fitted on the worktree's local data on 2026-11-01 (FEDFUNDS, CPILFESL, UNRATE, DBAA, DGS10 month-end, NY Fed HLW
# current estimates, the panel's us_equity): it reproduces W_A01 to the published three decimals (n 117, bin
# counts 12/15/10/40/17/15/8, the 2026Q4 macro prior 3.2/10.9/9.2/47.9/18.4/8.9/1.5%). build_priors refits it.
FED_PROBIT_MACRO = OrderedProbit(
    cols=("core", "gap", "du2", "lag", "rgap", "eq", "dspr"),
    coef=(0.077196, 0.097726, -0.398568, 0.722785, -0.341261, 0.183361, -0.068543),
    cuts=(-1.836839, -1.057414, -0.708711, 0.580936, 1.284351, 2.203071),
    x_mean=(2.992029, 0.533746, -0.02906, -0.00641, -1.205288, 0.020068, 0.004017),
    x_sd=(1.114573, 1.601115, 0.368193, 0.521403, 1.499056, 0.079249, 0.305362),
    n=117, sample=("1987Q3", "2026Q3"), counts=(12, 15, 10, 40, 17, 15, 8), mcfadden_r2=0.200, spec="macro",
    source="W_A01（agent_design/actors/scripts/a01_fed.py）在本地数据上重估")
FED_PROBIT_SCHEMA = OrderedProbit(
    cols=("core", "du2", "lag", "rgap", "eq", "dspr"),
    coef=(0.058912, -0.386742, 0.776576, -0.336022, 0.192931, -0.048746),
    cuts=(-1.82858, -1.045119, -0.698163, 0.576492, 1.275413, 2.203399),
    x_mean=(2.992029, -0.02906, -0.00641, -1.205288, 0.020068, 0.004017),
    x_sd=(1.114573, 0.368193, 0.521403, 1.499056, 0.079249, 0.305362),
    n=117, sample=("1987Q3", "2026Q3"), counts=(12, 15, 10, 40, 17, 15, 8), mcfadden_r2=0.198, spec="schema",
    source="W_A01 的宏观设定去掉 HLW 产出缺口（输入表没有这个状态；z 0.82，不显著），其余协变量都是输入表状态")
FED_RESEARCH = {
    "n": 117, "counts": [12, 15, 10, 40, 17, 15, 8], "mcfadden_r2": 0.20,
    "oos_2000_2026": {"n": 67, "mean_logscore_model": -1.473, "mean_logscore_uncond": -1.86,
                      "mean_logscore_persist": -1.53, "brier_model": 0.72, "brier_uncond": 0.824},
    "today_x_2026Q4": {"core": 2.566, "gap": -0.114, "du2": -0.1, "lag": 0.25, "rgap": 0.313, "eq": 0.015,
                       "dspr": -0.06},
    "prior_2026Q4_macro": {"cut_75plus": 0.032, "cut_50": 0.109, "cut_25": 0.092, "hold": 0.479, "hike_25": 0.184,
                           "hike_50": 0.089, "hike_75plus": 0.015},
    "source": "agent_design/actors/work/a01_fed.json",
}
# within-branch slopes (§6.6; e01 B: quarterly Δff on lagged information, 1987Q3-2026Q3 ex-ELB, n 117, R² 0.53, NW 4)
FED_SLOPES = {"us_equity_logret": {"b": 1.045, "se": 0.579}, "d_unrate": {"b": -0.513, "se": 0.204},
              "core_gap": {"b": 0.098, "se": 0.039}, "d_ff_lag": {"b": 0.533, "se": 0.090},
              "ff_level": {"b": -0.065, "se": 0.017}, "clip_pp": 0.75, "same_quarter": 0.0,
              "source": "engine-tree/work/e01_evidence.json B；同季斜率不可识别（美股 0.58，se 0.58），默认 0"}


def fed_dataset(data: LocalData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(d, X) of W_A01: the quarterly decision (changes rounded to 25bp) and the covariates known at the start."""
    ff = data.fred("FEDFUNDS")
    lvl = _q_first_next(ff)
    if pd.Period("2026Q3") not in lvl.index and ff.index.max() >= pd.Timestamp("2026-09-01"):
        # October 2026 not yet published: September averaged 3.75 after 3.63 in August and the meeting's change took
        # effect on the 17th, so the post-meeting level x solves (16*3.63 + 14*x)/30 = 3.75 (W_A01)
        sep, aug = float(ff.loc["2026-09-01"]), float(ff.loc["2026-08-01"])
        lvl.loc[pd.Period("2026Q3")] = round((sep * 30 - 16 * aug) / 14, 3)
    lvl = lvl.sort_index()
    d = pd.DataFrame({"level": lvl})
    d["chg"] = (d["level"].diff() / 0.25).round() * 0.25
    d["core"] = _q_last(_yoy(data.fred("CPILFESL")))
    d["u"] = _q_last(data.fred("UNRATE"))
    h = data.hlw()
    d["rstar"] = h["rstar"]
    d["gap_hlw"] = h["gap"]
    P = data.panel()
    d["eq"] = np.log1p(P["p_us_equity"]).groupby(P.index.to_period("Q")).sum()
    d["spr"] = _q_last(data.fred("DBAA", "eop") - data.fred("DGS10", "eop"))
    last = d["level"].dropna().index.max()
    d = d.loc[:last]
    d = d.reindex(pd.period_range(d.index.min(), last + 1, freq="Q"))
    X = pd.DataFrame(index=d.index)
    X["core"] = d["core"].shift(1)
    X["gap"] = d["gap_hlw"].shift(2)
    X["du2"] = (d["u"] - d["u"].shift(2)).shift(1)
    X["lag"] = d["chg"].shift(1)
    X["rgap"] = (d["level"] - d["core"]).shift(1) - d["rstar"].shift(2)
    X["eq"] = d["eq"].shift(1)
    X["dspr"] = d["spr"].diff().shift(1)
    X["start_level"] = d["level"].shift(1)
    return d, X


def _oprobit_nll(theta, X, y, K):
    from scipy import stats
    k = X.shape[1]
    cuts = np.concatenate([[theta[k]], theta[k] + np.cumsum(np.exp(theta[k + 1:]))])
    xb = X @ theta[:k]
    upper = np.where(y < K - 1, cuts[np.minimum(y, K - 2)], np.inf)
    lower = np.where(y > 0, cuts[np.maximum(y - 1, 0)], -np.inf)
    p = stats.norm.cdf(upper - xb) - stats.norm.cdf(lower - xb)
    return -np.sum(np.log(np.clip(p, 1e-12, None)))


def fit_ordered_probit(X: np.ndarray, y: np.ndarray, K: int, cols: Sequence[str], *, spec: str = "",
                       sample: tuple[str, str] = ("", ""), source: str = "") -> OrderedProbit:
    """Maximum likelihood (BFGS, then Nelder–Mead polish if needed) on standardised covariates."""
    from scipy import optimize, stats
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    k = Z.shape[1]
    freq = np.bincount(y, minlength=K) / len(y)
    c = stats.norm.ppf(np.clip(np.cumsum(freq)[:-1], 1e-3, 1 - 1e-3))
    th0 = np.concatenate([np.zeros(k), [c[0]], np.log(np.maximum(np.diff(c), 1e-3))])
    r = optimize.minimize(_oprobit_nll, th0, args=(Z, y, K), method="BFGS", options={"maxiter": 5000, "gtol": 1e-6})
    if not np.all(np.isfinite(r.x)):
        r = optimize.minimize(_oprobit_nll, th0, args=(Z, y, K), method="Nelder-Mead",
                              options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-9})
    cuts = np.concatenate([[r.x[k]], r.x[k] + np.cumsum(np.exp(r.x[k + 1:]))])
    counts = np.bincount(y, minlength=K)
    ll0 = float(np.sum(np.log(np.clip(counts[y] / len(y), 1e-12, None))))
    return OrderedProbit(cols=tuple(cols), coef=tuple(round(float(v), 6) for v in r.x[:k]),
                         cuts=tuple(round(float(v), 6) for v in cuts), x_mean=tuple(round(float(v), 6) for v in mu),
                         x_sd=tuple(round(float(v), 6) for v in sd), n=int(len(y)), sample=tuple(sample),
                         counts=tuple(int(v) for v in counts), mcfadden_r2=round(1.0 + float(r.fun) / ll0, 3),
                         spec=spec, source=source)


def fit_fed_probit(data: LocalData, spec: str = "schema") -> tuple[OrderedProbit, dict]:
    """The probit refitted on local data, and the covariates of the next decision (the last row of X)."""
    d, X = fed_dataset(data)
    y = pd.cut(d["chg"], list(FED_EDGES), labels=False).rename("y")
    df = X.join(y).loc[pd.Period("1987Q3"):]
    cols = list(FED_PROBIT_MACRO.cols if spec == "macro" else FED_PROBIT_SCHEMA.cols)
    last_obs = d["chg"].dropna().index.max()
    fit_df = df.loc[:last_obs]
    fit_df = fit_df[fit_df["start_level"] >= 0.40].dropna(subset=cols + ["y"])
    pr = fit_ordered_probit(fit_df[cols].to_numpy(float), fit_df["y"].astype(int).to_numpy(), len(FED_LABELS), cols,
                            spec=spec, sample=(str(fit_df.index.min()), str(fit_df.index.max())),
                            source=f"本地数据重估（{config.RAW_DIR.name}/fred、nyfed HLW、项目面板）")
    nxt = X.iloc[-1]
    today = {c: (round(float(nxt[c]), 3) if pd.notna(nxt[c]) else None) for c in FED_PROBIT_MACRO.cols}
    return pr, {"quarter": str(X.index[-1]), "x": today, "start_level": float(nxt["start_level"])}


def fed_covariates(state: Mapping[str, float]) -> dict[str, float | None]:
    """The schema-variant covariates from a node's state (s fields and last quarter's x values as <id>_l1).

    core ← s_core_yoy; du2 ← d_unrate_l1 + d_unrate_l2 (or `du2`); lag ← d_ff_l1; rgap ← s_ff − s_core_yoy −
    s_rstar; eq ← us_equity_l1 (last quarter's log price return); dspr ← d_spread_l1. Explicit keys win."""
    g = lambda k: (None if state.get(k) is None else float(state[k]))   # noqa: E731
    du2 = g("du2")
    if du2 is None and g("d_unrate_l1") is not None:
        du2 = g("d_unrate_l1") + (g("d_unrate_l2") or 0.0)
    rgap = g("rgap")
    if rgap is None and None not in (g("s_ff"), g("s_core_yoy"), g("s_rstar")):
        rgap = g("s_ff") - g("s_core_yoy") - g("s_rstar")
    return {"core": g("core") if g("core") is not None else g("s_core_yoy"), "du2": du2,
            "lag": g("lag") if g("lag") is not None else g("d_ff_l1"), "rgap": rgap,
            "eq": g("eq") if g("eq") is not None else g("us_equity_l1"),
            "dspr": g("dspr") if g("dspr") is not None else g("d_spread_l1")}


def fed_pinned_renormalize(p: Mapping[str, float], first_meeting: str) -> dict[str, float]:
    """The quarter's options compatible with a pinned first meeting: its move plus −25 / 0 / +25bp at the second
    (no inter-meeting moves), renormalised (ACTORS §9: held in October → −25/0/+25 at 12.2/63.4/24.4%)."""
    first = {"hold": 0.0, "cut_25": -0.25, "cut_50": -0.5, "hike_25": 0.25}.get(first_meeting)
    if first is None:
        return dict(p)
    ok = {}
    for opt, mid in zip(FED_OPTIONS, FED_MID):
        tot = round(mid / 0.25) * 0.25
        if any(abs((first + s) - tot) < 1e-9 for s in (-0.25, 0.0, 0.25)) and opt not in ("cut_75plus", "hike_75plus"):
            ok[opt] = p.get(opt, 0.0)
    return normalize(ok, list(ok)) if sum(ok.values()) > 0 else dict(p)


def fed_prior(state: Mapping | None = None, *, probit: OrderedProbit | None = None, quarter: str | None = None,
              pinned_first_meeting: str | None = None) -> dict:
    """{option: p} over the seven bins, with the source. Without a full covariate vector the frozen 2026Q4 macro
    prior is used (labelled). A pinned first meeting (2026Q4: the Oct 27–28 decision) renormalises."""
    pr = probit or FED_PROBIT_SCHEMA
    cov = fed_covariates(state or {})
    if all(cov.get(c) is not None for c in pr.cols):
        p = pr.prior(cov)
        src, label = f"fed_probit_{pr.spec}", "有序 probit（W_A01，按期初状态）"
    else:
        p = dict(FED_RESEARCH["prior_2026Q4_macro"])
        src, label = "fed_probit_2026Q4_frozen", "有序 probit 的 2026Q4 先验（W_A01；缺少状态时冻结值）"
    if pinned_first_meeting and quarter == "2026Q4":
        p = fed_pinned_renormalize(p, pinned_first_meeting)
        label += f"；10 月会议锁定为 {pinned_first_meeting}，在相容选项上重新归一"
    return {"p": p, "source": src, "label": label, "kind": "history"}


# =====================================================================================================================
# 3. Decision records (dated lists from the public record; the author's tabulations are marked 待核)
# =====================================================================================================================

STIMULUS_LAWS = [   # countercyclical federal stimulus laws (signing date, name, headline size % of GDP); W_A02
    ("1975-03-29", "Tax Reduction Act of 1975", 1.4), ("1991-11-15", "Emergency Unemployment Compensation Act", 0.1),
    ("2001-06-07", "EGTRRA (2001 rebates)", 0.4), ("2002-03-09", "Job Creation and Worker Assistance Act", 0.4),
    ("2008-02-13", "Economic Stimulus Act of 2008", 1.0), ("2009-02-17", "ARRA", 5.5),
    ("2020-03-27", "CARES Act", 10.0), ("2020-12-27", "Consolidated Appropriations Act, 2021", 4.2),
    ("2021-03-11", "American Rescue Plan Act", 8.5)]
SHUTDOWNS = {"lapses_with_furloughs_FY1991_FY2026": ["1990-10", "1995-11", "1995-12/1996-01", "2013-10", "2018-01",
                                                     "2018-12/2019-01", "2025-10/11"], "fiscal_years": 36,
             "source": "CRS RS20348（2025 年一项待核）"}
DEBT_LIMIT = {"limit_actions_since_1960": 78, "defaults": 0, "spells_since_2011_days": [54, 75, 77, 91, 134, 137, 170,
                                                                                         182, 219, 248],
              "source": "Treasury；A11（Debt to the Penny：30 天内变动 < 0.05% 且随后跳升 ≥ 0.8%）"}
PROCYCLICAL_LAWS = {"count": 5, "years": 45, "laws": ["ERTA 1981", "EGTRRA 2001", "JGTRRA 2003", "TCJA 2017",
                                                       "OBBBA 2025"], "source": "作者计数（待核）"}
CONSOLIDATION_LAWS = {"count": 6, "years": 36, "laws": ["OBRA 1990", "OBRA 1993", "BBA 1997", "BCA 2011", "ATRA 2013",
                                                         "FRA 2023"], "source": "作者计数（待核）"}
US_ESCALATIONS_CN = {"rounds": ["2018-07", "2018-08", "2018-09", "2019-05", "2019-09", "2025-02", "2025-03",
                                "2025-04"], "retaliated": 8, "export_controls_rounds": ["2025-04", "2025-10"],
                     "source": "Bown (2021) 与 2025 年公告（W_A03）"}
CN_STIMULUS_EPISODES = {"starts": ["2008Q4", "2012Q2", "2015Q2", "2018Q3", "2020Q1", "2024Q3"], "quarters": 72,
                        "span": ["2008Q4", "2026Q3"], "source": "CHINA_STIM（作者整理）"}
CN_DEPRECIATION_EPISODES = {"starts": ["2015Q3", "2018Q2", "2019Q3", "2022Q2"], "quarters": 44,
                            "span": ["2015Q4", "2026Q3"], "source": "W_A07"}
TAIWAN_CRISES = {"crises": ["1954-09", "1958-08", "1995-07", "2022-08"], "quarters": 308, "span": ["1949Q4", "2026Q3"],
                 "blockades": 0, "source": "TAIWAN_HIST（W_A03）"}
OPEC_DECISIONS = [  # (date, kind): cut | raise | market_share | ceiling; W_A03 (the author's reading of OPEC communiques)
    ("1982-03-20", "ceiling"), ("1983-03-14", "cut"), ("1984-10-31", "cut"), ("1985-12-09", "market_share"),
    ("1986-08-05", "cut"), ("1986-12-20", "cut"), ("1988-11-28", "cut"), ("1990-08-29", "raise"),
    ("1997-11-29", "raise"), ("1998-03-22", "cut"), ("1998-06-24", "cut"), ("1999-03-23", "cut"),
    ("2000-03-28", "raise"), ("2001-01-17", "cut"), ("2001-03-17", "cut"), ("2001-07-25", "cut"),
    ("2001-12-28", "cut"), ("2003-03-11", "raise"), ("2004-06-03", "raise"), ("2006-10-20", "cut"),
    ("2006-12-14", "cut"), ("2008-10-24", "cut"), ("2008-12-17", "cut"), ("2011-06-08", "raise"),
    ("2014-11-27", "market_share"), ("2015-12-04", "market_share"), ("2016-11-30", "cut"), ("2018-06-22", "raise"),
    ("2018-12-07", "cut"), ("2019-12-06", "cut"), ("2020-03-06", "market_share"), ("2020-04-12", "cut"),
    ("2021-07-18", "raise"), ("2022-10-05", "cut"), ("2023-04-02", "cut"), ("2023-06-04", "cut"),
    ("2023-11-30", "cut"), ("2025-03-03", "raise"), ("2025-05-03", "raise")]
OIL_DISRUPTIONS = {"list": ["1973-10 阿以战争与禁运", "1978-11 伊朗革命", "1980-09 两伊战争", "1990-08 海湾战争",
                            "2002-12 委内瑞拉罢工 / 2003-03 伊拉克战争", "2011-02 利比亚", "2019-09 阿布盖格袭击",
                            "2022-02 俄乌战争", "2026-03 供给冲击（项目数据）"], "quarters": 215,
                   "source": "Kilian (2008) 外生供给冲击 1973–2004 加之后的事件（W_A03）"}
FIN_ACCIDENTS = {"list": ["1984-05 Continental Illinois", "1990-91 新英格兰银行倒闭与信贷紧缩", "1998-09 LTCM",
                          "2007-08 次贷与金融危机", "2023-03 SVB 与地区银行"], "quarters": 215, "systemic": 2,
                 "source": "Laeven–Valencia (2018) 与 FOMC 记录（W_A03）"}
SPR_DRAWDOWNS = {"emergency": ["1991-01", "2005-09", "2011-06", "2022-03"], "quarters": 143,
                 "given_disruption": "1990 年以来 6 次中断约 4 次（作者计数，待核）", "source": "DOE"}
# president's party in midterms when it held the House before the election (1946-2022): lost / held
MIDTERMS_HELD = {"lost": [1946, 1954, 1994, 2006, 2010, 2018, 2022], "held": [1950, 1962, 1966, 1978, 2002],
                 "senate_lost_given_held": {"k": 6, "n": 15}, "source": "APP 中期选举表（作者整理，待核）"}
PRESIDENTIAL = {"incumbent_party_won_1952_2024": {"k": 8, "n": 19}, "source": "选举结果（作者整理）"}


# =====================================================================================================================
# 4. Re-estimation from local data
# =====================================================================================================================

def sahm_triggers(u: pd.Series) -> list[pd.Timestamp]:
    """Real-time-form Sahm triggers on the final vintage: the 3-month mean ≥ 0.5pp above the minimum of the previous
    12 months' 3-month means; a new trigger only 24 months after the last (W_A02)."""
    m3 = _monthly(u).interpolate(limit=1).rolling(3).mean()
    low = m3.rolling(12).min().shift(1)
    s = (m3 - low) >= 0.5
    first = s & ~s.shift(1, fill_value=False)
    out, last = [], None
    for d, v in first.items():
        if v and (last is None or (d - last).days > 730):
            out.append(d)
            last = d
    return out


def fiscal_record(data: LocalData | None = None) -> dict:
    out = {"shutdown": {**rate(len(SHUTDOWNS["lapses_with_furloughs_FY1991_FY2026"]), SHUTDOWNS["fiscal_years"]),
                        "source": SHUTDOWNS["source"]},
           "procyclical_per_q": round(PROCYCLICAL_LAWS["count"] / (4 * PROCYCLICAL_LAWS["years"]), 4),
           "consolidation_per_q": round(CONSOLIDATION_LAWS["count"] / (4 * CONSOLIDATION_LAWS["years"]), 4),
           "debt_limit": {"defaults": DEBT_LIMIT["defaults"], "actions": DEBT_LIMIT["limit_actions_since_1960"],
                          "rule_of_three_upper": round(3 / DEBT_LIMIT["limit_actions_since_1960"], 3),
                          "spells_since_2011": len(DEBT_LIMIT["spells_since_2011_days"]),
                          "median_days": float(np.median(DEBT_LIMIT["spells_since_2011_days"])),
                          "resolved": len(DEBT_LIMIT["spells_since_2011_days"]), "source": DEBT_LIMIT["source"]}}
    stim = {"k": 4, "n": 9, "p": 0.444, "jeffreys95": [0.173, 0.746], "reproduced": False, "why": "冻结值（W_A02）"}
    if data is not None:
        try:
            trig = [t for t in sahm_triggers(data.fred("UNRATE")) if t >= pd.Timestamp("1970-01-01")]
            laws = pd.to_datetime([d for d, _, _ in STIMULUS_LAWS])
            k = sum(1 for t in trig if any(t - pd.Timedelta(days=92) <= ld <= t + pd.Timedelta(days=366) for ld in laws))
            stim = {**rate(k, len(trig)), "triggers": [str(t.date()) for t in trig],
                    "reproduced": (k, len(trig)) == (4, 9), "why": "本地 UNRATE 重算"}
        except (FileNotFoundError, KeyError, ValueError) as err:
            stim["why"] = f"冻结值（{err}）"
    out["stimulus_given_sahm"] = stim
    out["stimulus_hazard_per_q_while_triggered"] = round(1 - (1 - stim["p"]) ** (1 / 5), 3) if stim.get("p") else 0.089
    return out


def tariff_record(data: LocalData | None = None) -> dict:
    frozen = {"normal_big_move": {"k": 11, "n": 260, "p": 0.0423, "jeffreys95": [0.023, 0.072]},
              "active_regime": {"n": 15, "up": 10, "down": 4, "flat": 1}, "reproduced": False, "why": "冻结值（W_A02）"}
    if data is None:
        return frozen
    try:
        cu, im = data.fred("B235RC1Q027SBEA"), data.fred("A255RC1Q027SBEA")
    except FileNotFoundError as err:
        return {**frozen, "why": f"冻结值（{err}）"}
    tr = (100 * cu / im).dropna()
    tq = tr.groupby(tr.index.to_period("Q")).last()
    dtq = tq.diff()
    base = dtq.loc[pd.Period("1960Q1"):pd.Period("2024Q4")].dropna()
    k = int((base.abs() >= 0.5).sum())
    act = pd.concat([dtq.loc[pd.Period("2018Q1"):pd.Period("2019Q4")], dtq.loc[pd.Period("2025Q1"):pd.Period("2026Q3")]])
    act = act.dropna()
    out = {"normal_big_move": rate(k, len(base)),
           "active_regime": {"n": int(len(act)), "up": int((act > 0.1).sum()), "down": int((act < -0.1).sum()),
                             "flat": int((act.abs() <= 0.1).sum()),
                             "changes": {str(q): round(float(v), 2) for q, v in act.items()}},
           "latest_rate": {"quarter": str(tq.index[-1]), "pct": round(float(tq.iloc[-1]), 2)},
           "why": "本地 FRED（海关关税 / 商品进口）重算"}
    out["reproduced"] = (k, len(base)) == (11, 260) and (out["active_regime"]["up"], out["active_regime"]["down"]) == (10, 4)
    return out


def _real_oil_q(data: LocalData) -> pd.Series:
    wti, cpi = data.fred("WTISPLC"), data.fred("CPIAUCSL")
    real = (wti / _monthly(cpi).interpolate(limit=1).reindex(wti.index)).dropna()
    return np.log(_q_mean(real))


def opec_record(data: LocalData | None = None) -> dict:
    frozen = {"cut_given_crash": {"k": 5, "n": 8, "p": 0.625, "jeffreys95": [0.295, 0.881]}, "reproduced": False,
              "why": "冻结值（W_A03）"}
    rows = {"crash": {"quota_cut": 0.625, "quota_hold": 0.25, "quota_raise": 0.125},
            "spike": {"quota_raise": 0.30, "quota_hold": 0.55, "quota_cut": 0.15},
            "inside": {"quota_hold": 0.70, "quota_cut": 0.15, "quota_raise": 0.15}}
    out = {"rows": rows, "rows_note": "崩跌行：2 季内减产 5/8，其余在维持与争夺份额（增产）之间按 2:1（判断值）；"
                                      "飙升行：增产 0.30 / 维持 0.55 / 减产 0.15（W_A03）；区间内一行判断值", **frozen}
    if data is None:
        return out
    try:
        rq = _real_oil_q(data)
    except FileNotFoundError as err:
        return {**out, "why": f"冻结值（{err}）"}
    d2 = rq.diff(2)
    rng = slice(pd.Period("1974Q1"), pd.Period("2026Q3"))
    crash = [q for q in _episodes((d2 <= -0.30).loc[rng]) if q <= pd.Period("2025Q4")]
    dec = pd.DataFrame(OPEC_DECISIONS, columns=["date", "kind"])
    dec["q"] = pd.PeriodIndex(pd.to_datetime(dec["date"]), freq="Q")
    k = sum(1 for q in crash if ((dec["q"] >= q) & (dec["q"] <= q + 2) & (dec["kind"] == "cut")).any())
    out["cut_given_crash"] = rate(k, len(crash))
    out["crash_quarters"] = [str(q) for q in crash]
    out["reproduced"] = (k, len(crash)) == (5, 8)
    out["why"] = "本地 FRED（WTISPLC / CPIAUCSL）与日期化的欧佩克决定重算"
    return out


def belligerents_record(data: LocalData | None = None) -> dict:
    frozen = {"exit": {"k": 13, "n": 31, "p": 0.419, "jeffreys95": [0.259, 0.594]}, "reproduced": False,
              "why": "冻结值（W_A06）"}
    rows = {"ceasefire": 0.42, "escalate": 0.15, "stalemate": 0.43}
    out = {"rows": rows, "rows_note": "停火或航道恢复 = 退出风险率（W_A06）；升级 0.15、僵持 0.43（判断值）", **frozen}
    if data is None:
        return out
    try:
        rq = _real_oil_q(data)
    except FileNotFoundError as err:
        return {**out, "why": f"冻结值（{err}）"}
    flag = ((rq - rq.shift(4)) > np.log(1.4)).loc[pd.Period("1974Q1"):pd.Period("2026Q3")]
    runs, cur = [], None
    for q, v in flag.items():
        if v and cur is None:
            cur = [q, q]
        elif v:
            cur[1] = q
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    done = [(a, b) for a, b in runs if b < pd.Period("2026Q3")]
    exposure = sum(int((b - a).n + 1) for a, b in done)
    out["exit"] = rate(len(done), exposure)
    out["runs"] = [{"start": str(a), "end": str(b)} for a, b in runs]
    out["reproduced"] = (len(done), exposure) == (13, 31)
    out["why"] = "本地 FRED 实际油价同比 > 40% 的持续期重算"
    return out


def nature_record(data: LocalData | None = None) -> dict:
    n_oil = len(OIL_DISRUPTIONS["list"])
    out = {"oil_disruption": {**rate(n_oil, OIL_DISRUPTIONS["quarters"]), "list": OIL_DISRUPTIONS["list"],
                              "source": OIL_DISRUPTIONS["source"]},
           "financial_accident": {**rate(len(FIN_ACCIDENTS["list"]), FIN_ACCIDENTS["quarters"]),
                                  "list": FIN_ACCIDENTS["list"], "systemic_share": rate(FIN_ACCIDENTS["systemic"],
                                                                                       len(FIN_ACCIDENTS["list"])),
                                  "source": FIN_ACCIDENTS["source"]},
           "pandemic": {"p": 0.005, "basis": "约每年 2%（Marani et al. 2021, PNAS）", "jeffreys95": [None, None]},
           "gpr_episode": {"k": 8, "n": 167, "p": 0.0479, "jeffreys95": [0.023, 0.088], "reproduced": False,
                           "why": "冻结值（W_A03）"}}
    if data is not None:
        try:
            g = data.gpr()
            gq = g.groupby(g.index.to_period("Q")).mean()
            p95 = float(gq.loc[pd.Period("1985Q1"):pd.Period("2019Q4")].quantile(0.95))
            sp = gq.loc[pd.Period("1985Q1"):pd.Period("2026Q3")] > p95
            eps = _episodes(sp)
            out["gpr_episode"] = {**rate(len(eps), int(sp.notna().sum())), "p95": round(p95, 1),
                                  "episodes": [str(e) for e in eps], "why": "本地 GPR（Caldara–Iacoviello）重算",
                                  "reproduced": len(eps) == 8}
        except (FileNotFoundError, KeyError, ValueError) as err:
            out["gpr_episode"]["why"] = f"冻结值（{err}）"
    return out


def china_record() -> dict:
    st, dp, tw = CN_STIMULUS_EPISODES, CN_DEPRECIATION_EPISODES, TAIWAN_CRISES
    return {"retaliation": {**rate(US_ESCALATIONS_CN["retaliated"], len(US_ESCALATIONS_CN["rounds"])),
                            "source": US_ESCALATIONS_CN["source"]},
            "stimulus": {**rate(len(st["starts"]), st["quarters"]), "source": st["source"]},
            "depreciation": {**rate(len(dp["starts"]), dp["quarters"]), "source": dp["source"]},
            "taiwan_crisis": {**rate(len(tw["crises"]), tw["quarters"]), "source": tw["source"]},
            "taiwan_blockade": {**rate(tw["blockades"], tw["quarters"]),
                                "rule_of_three_upper": round(3 / tw["quarters"], 4)}}


def bigtech_record(data: LocalData | None = None) -> dict:
    rows = {"boom": {"capex_raise_large": 0.35, "capex_hold": 0.50, "capex_cut": 0.12, "capex_slash": 0.03},
            "normal": {"capex_raise_large": 0.20, "capex_hold": 0.60, "capex_cut": 0.15, "capex_slash": 0.05},
            "bust": {"capex_raise_large": 0.05, "capex_hold": 0.35, "capex_cut": 0.40, "capex_slash": 0.20}}
    out = {"rows": rows, "status": "判断值（锚定 NIPA 条件频率）",
           "regime_rule": "繁荣：纳指 4 季收益 > 0 且上季实际 IT 投资年化增速 > 15%；萧条：纳指 4 季跌 ≥ 20% 或上季 GZ 利差 +0.3"
                          "（v2 删除了 EBP，改用 GZ，判断值）；其余为平常",
           "anchor": {"k": 9, "n": 16, "p": 0.562, "jeffreys95": [0.326, 0.778], "otherwise": 0.063,
                      "reproduced": False, "why": "冻结值（W_C02）"}}
    if data is None:
        return out
    try:
        it = data.fred("B679RA3Q086SBEA")
        P = data.panel()
    except FileNotFoundError as err:
        out["anchor"]["why"] = f"冻结值（{err}）"
        return out
    itq = it.groupby(it.index.to_period("Q")).last()
    g = 100.0 * ((itq / itq.shift(1)) ** 4 - 1.0)
    if "lvl_nasdaq" not in P.columns:
        out["anchor"]["why"] = "冻结值（面板没有 lvl_nasdaq）"
        return out
    nd = P["lvl_nasdaq"].dropna()
    ndq = np.log(nd[nd.index.month.isin([3, 6, 9, 12])])
    ndq.index = ndq.index.to_period("Q")
    frame = pd.DataFrame({"r4": ndq - ndq.shift(4), "g_next": g.shift(-1)}).dropna().loc[pd.Period("1986Q1"):]
    bad = frame["r4"] <= np.log(0.8)
    k = int((frame.loc[bad, "g_next"] < 0).sum())
    out["anchor"] = {**rate(k, int(bad.sum())), "otherwise": round(float((frame.loc[~bad, "g_next"] < 0).mean()), 3),
                     "reproduced": (k, int(bad.sum())) == (9, 16), "why": "本地 NIPA 实际 IT 投资与面板纳指重算"}
    return out


def voter_record() -> dict:
    lost, held = MIDTERMS_HELD["lost"], MIDTERMS_HELD["held"]
    since94 = [y for y in lost if y >= 1994], [y for y in held if y >= 1994]
    return {"midterm_house_lost_given_held": {**rate(len(lost), len(lost) + len(held)), "lost": lost, "held": held},
            "since_1994": rate(len(since94[0]), len(since94[0]) + len(since94[1])),
            "senate_lost_given_held": rate(**MIDTERMS_HELD["senate_lost_given_held"]),
            "presidential_incumbent_party": rate(**PRESIDENTIAL["incumbent_party_won_1952_2024"]),
            "source": MIDTERMS_HELD["source"]}


def frozen_priors() -> dict:
    """The tables without local data (fresh clones, CI): the research round's numbers and the dated lists."""
    return {"version": PRIORS_VERSION, "built": False,
            "fed": {"macro": FED_PROBIT_MACRO.to_dict(), "schema": FED_PROBIT_SCHEMA.to_dict(), "research": FED_RESEARCH,
                    "slopes": FED_SLOPES, "reproduced": None, "why": "冻结值（本地重估见 build_priors）"},
            "fiscal": fiscal_record(None), "tariff": tariff_record(None), "opec": opec_record(None),
            "belligerents": belligerents_record(None), "nature": nature_record(None), "china": china_record(),
            "bigtech": bigtech_record(None), "voter": voter_record(), "treasury": TREASURY_ROW}


def build_priors(data: LocalData | None = None) -> dict:
    """Every table re-estimated from local data, with `reproduced` per table against the research round."""
    data = data or LocalData()
    out = frozen_priors()
    out["built"] = True
    try:
        macro, today = fit_fed_probit(data, "macro")
        schema, _ = fit_fed_probit(data, "schema")
        p_today = macro.prior(today["x"]) if all(today["x"].get(c) is not None for c in macro.cols) else None
        rep = (macro.n == FED_RESEARCH["n"] and list(macro.counts) == FED_RESEARCH["counts"]
               and p_today is not None and all(abs(p_today[k] - v) < 0.0015 for k, v in
                                               FED_RESEARCH["prior_2026Q4_macro"].items()))
        out["fed"] = {**out["fed"], "macro": macro.to_dict(), "schema": schema.to_dict(), "next_decision": today,
                      "prior_next_macro": p_today, "reproduced": bool(rep), "why": "本地数据重估"}
    except (FileNotFoundError, KeyError, ValueError, IndexError) as err:
        out["fed"]["why"] = f"冻结值（{err}）"
    out["fiscal"] = fiscal_record(data)
    out["tariff"] = tariff_record(data)
    out["opec"] = opec_record(data)
    out["belligerents"] = belligerents_record(data)
    out["nature"] = nature_record(data)
    out["bigtech"] = bigtech_record(data)
    out["summary"] = {k: out[k].get("reproduced") if isinstance(out[k], Mapping) else None
                      for k in ("fed", "tariff", "opec", "belligerents")}
    out["summary"].update({"stimulus_given_sahm": out["fiscal"]["stimulus_given_sahm"].get("reproduced"),
                           "gpr_episode": out["nature"]["gpr_episode"].get("reproduced"),
                           "bigtech_anchor": out["bigtech"]["anchor"].get("reproduced")})
    return out


def load_priors(path: Path | None = None) -> dict:
    """The built tables (data/results/agents_priors.json) when present, else the frozen ones."""
    p = Path(path) if path else PRIORS_PATH
    try:
        d = jsonio.load(p)
        if isinstance(d, Mapping) and d.get("version") == PRIORS_VERSION:
            return d
    except (OSError, ValueError):
        pass
    return frozen_priors()


def fed_probit_of(tables: Mapping | None = None, spec: str = "schema") -> OrderedProbit:
    try:
        return OrderedProbit.from_dict((tables or {})["fed"][spec])
    except (KeyError, TypeError):
        return FED_PROBIT_SCHEMA if spec == "schema" else FED_PROBIT_MACRO


# =====================================================================================================================
# 5. The prior of a node's menu
# =====================================================================================================================

TREASURY_ROW = {"refunding_unchanged": 0.85, "bills_up": 0.06, "bills_down": 0.06, "buybacks_up": 0.03,
                "note": "TBAC 长期建议短债占比约 15–20%（判断值行）"}
ATTRIBUTE_ROWS = {   # non-branching attributes (判断值 unless noted)
    ("fed", "balance_sheet"): {"bs_hold": 0.80, "bs_reserve_purchases": 0.12, "bs_runoff": 0.05, "bs_qe": 0.03},
    ("president", "export_controls"): {"ec_hold": 0.80, "ec_tighten": 0.15, "ec_ease": 0.05},
    ("president", "energy"): {"spr_hold": 0.75, "spr_release": 0.25},
    ("president", "immigration"): {"imm_hold": 0.85, "imm_tighten": 0.15},
    ("president", "fed_pressure"): {"pressure_none": 0.60, "pressure_public": 0.30, "nominate_dovish": 0.08,
                                    "attempt_removal": 0.02},
    ("congress", "deadline"): {"cr_on_time": 0.806, "shutdown": 0.194},          # 7 lapses in 36 FYs (CRS)
    ("china", "domestic"): {"targeted_support": 0.911, "major_stimulus": 0.089},  # Jeffreys mean of 6 in 72 q
    ("china", "fx"): {"fx_hold": 0.90, "cny_weaker": 0.10},                       # Jeffreys mean of 4 in 44 q
    ("china", "taiwan"): {"tw_status_quo": 0.8884, "tw_large_exercise": 0.11, "tw_blockade": 0.0016},
    ("bigtech", "workforce"): {"hire_hold": 0.70, "hire_expand": 0.10, "hire_freeze_ai": 0.15, "layoffs": 0.05},
    ("bigtech", "payout"): {"buyback_hold": 0.70, "buyback_up": 0.10, "buyback_cut": 0.20},
    ("bigtech", "pricing"): {"price_hold": 0.80, "price_war": 0.10, "monetize": 0.10},
}


def _uniform(ids: Sequence[str], why: str = "无历史先例") -> dict:
    return {"p": {i: 1.0 / len(ids) for i in ids} if ids else {}, "source": "uniform", "label": why, "kind": "uniform"}


def _ctx(ctx) -> dict:
    if ctx is None:
        return {}
    if isinstance(ctx, Mapping):
        return dict(ctx)
    return {k: getattr(ctx, k) for k in ("quarter", "state", "facts", "political", "institutions", "gates",
                                         "leader_choices", "stress_mode") if hasattr(ctx, k)}


def opec_regime(state: Mapping) -> str:
    """crash: real oil −30% over 2 quarters (log) or Brent below the pinned breakeven; spike: +25% in a quarter or
    a disruption; inside: otherwise. Keys: real_oil_d2 / real_oil_d1 (log), brent_below_breakeven_2q (bool)."""
    d2, d1 = state.get("real_oil_d2"), state.get("real_oil_d1")
    if state.get("brent_below_breakeven_2q") or (d2 is not None and float(d2) <= -0.30):
        return "crash"
    if (d1 is not None and float(d1) >= 0.25) or state.get("oil_disruption"):
        return "spike"
    return "inside"


def bigtech_regime(state: Mapping) -> str:
    r4, g, dgz = state.get("nasdaq_4q"), state.get("d_itinv_l1"), state.get("d_gz_l1")
    if (r4 is not None and float(r4) <= math.log(0.8)) or (dgz is not None and float(dgz) >= 0.3):
        return "bust"
    if r4 is not None and float(r4) > 0 and g is not None and float(g) > 15:
        return "boom"
    return "normal"


def prior(actor_id: str, option_ids: Sequence[str], ctx=None, *, domain: str | None = None,
          tables: Mapping | None = None) -> dict:
    """{"p": {option: p}, "source", "label", "kind", "notes"} for the live options of one domain at a node.

    kind: market (user odds; the actor is prior-only) | history | judgment | uniform. Gates from the accounts
    (G-L5/6/7/8) multiply soft tilts; reaction rows give followers' conditionals (from ctx["leader_choices"]).
    Options a table lacks get a floor of 0.01 before renormalising (noted)."""
    from . import actors as A
    c = _ctx(ctx)
    state = dict(c.get("state") or {})
    facts = dict(c.get("facts") or {})
    inst = dict(c.get("institutions") or {})
    gates = dict(c.get("gates") or {})
    leaders = dict(c.get("leader_choices") or {})
    quarter = c.get("quarter")
    tables = tables or load_priors()
    ids = list(option_ids)
    if not ids:
        return _uniform(ids)
    actor = A.ACTORS[actor_id]
    dom = actor.domain(domain) if domain else actor.main
    notes: list[str] = []
    key = (actor_id, dom.id)
    if key in ATTRIBUTE_ROWS:
        row = dict(ATTRIBUTE_ROWS[key])
        if key == ("president", "energy"):
            if facts.get("oil_disruption_ongoing"):
                e = A.ELECTIONS.get(A.quarter_add(quarter, 1)) if quarter else None
                row = {"spr_release": 0.5 if (e or leaders.get("belligerents") == "escalate") else 0.25}
                row["spr_hold"] = 1 - row["spr_release"]
                notes.append("中断持续：释放 0.25；两季内有选举或冲突升级时 0.5（v1 规则，判断值）")
            else:
                row = {"spr_hold": 0.972, "spr_release": 0.028}
        if key == ("fed", "balance_sheet") and ((gates.get("fed") or {}).get("tilt") or {}).get("bs_runoff"):
            row["bs_runoff"] *= float(gates["fed"]["tilt"]["bs_runoff"])
            notes.append("G-L5：准备金低于缩表结束时的水平，恢复缩表 ×0.5（判断值）")
        if key == ("china", "fx") and (gates.get("china") or {}).get("reserves_falling"):
            row["cny_weaker"] *= 0.5
            notes.append("G-L6：储备 4 季降超 10%，“允许贬值”先验 ×0.5")
        if key == ("bigtech", "payout") and ((gates.get("bigtech") or {}).get("tilt") or {}).get("buyback_up", 1.0) < 1:
            row["buyback_up"] *= (gates["bigtech"]["tilt"]["buyback_up"])
            notes.append("G-L7：派现后自由现金流为负，加速回购 ×0.5")
        out = {"p": normalize(row, ids, 0.01), "source": f"attribute:{actor_id}.{dom.id}", "kind": "judgment",
               "label": "判断值（属性）", "notes": notes}
        return out
    prior_id = dom.prior
    if prior_id == "fed_probit":
        res = fed_prior(state, probit=fed_probit_of(tables), quarter=quarter,
                        pinned_first_meeting=inst.get("fomc_2026_10") if quarter == "2026Q4" else None)
        p = dict(res["p"])
        tilt = float((c.get("political") or {}).get("chair_tilt") or 0.0)
        if tilt:
            for opt, mid in zip(FED_OPTIONS, FED_MID):
                p[opt] = p.get(opt, 0.0) * math.exp(0.5 * tilt * (1 if mid > 0 else -1 if mid < 0 else 0))
            notes.append(f"主席倾向 {tilt:+.1f}：每单位使降息对加息的对数几率移动 0.5（判断值）")
        if (gates.get("fed") or {}).get("reserves_below_ample"):
            notes.append("G-L5：准备金低于缩表结束时的水平（只影响资产负债表属性）")
        return {"p": normalize(p, ids, 0.0) if sum(p.get(i, 0) for i in ids) > 0 else _uniform(ids)["p"],
                "source": res["source"], "label": res["label"], "kind": "history", "notes": notes}
    if prior_id == "president_trade":
        regime = (c.get("political") or {}).get("trade_regime", "active")
        if regime == "active":
            row = {"tariff_up_large": 0.30, "tariff_up_small": 0.25, "tariff_hold": 0.20, "tariff_down": 0.15,
                   "deal": 0.10}
            label = "活跃关税体制：15 季中 10 升 4 降（W_A06）；广泛/行业的拆分与降级/协议的拆分是判断值"
        else:
            p_big = float(tables["tariff"]["normal_big_move"]["p"] or 0.0423)
            row = {"tariff_hold": 1 - p_big, "tariff_up_small": p_big / 2, "tariff_down": p_big / 2,
                   "tariff_up_large": 0.0, "deal": 0.0}
            label = f"平常体制：P(|Δ| ≥ 0.5pp) = {p_big:.3f}（11/260，W_A02）"
        return {"p": normalize(row, ids, 0.005), "source": "tariff_record", "label": label, "kind": "history",
                "notes": notes}
    if prior_id == "uniform":
        return {**_uniform(ids), "notes": ["只有一例：没有历史基率"]}
    if prior_id == "congress_fiscal":
        f = tables["fiscal"]
        row: dict[str, float] = {}
        if "stimulus_small" in ids or "stimulus_large" in ids:
            h = float(f.get("stimulus_hazard_per_q_while_triggered") or 0.089)
            if "stimulus_large" in ids and facts.get("pandemic"):
                row["stimulus_large"] = 0.9
                notes.append("疫情后大规模刺激 0.9（一例，2020）")
            else:
                row["stimulus_small"] = h * 0.6
                row["stimulus_large"] = h * 0.4
                notes.append(f"萨姆规则触发期内每季刺激风险率 {h:.3f}（4/9 在 5 季内，W_A02）；大小拆分判断值")
        row["procyclical_tax_cut"] = float(f.get("procyclical_per_q") or 0.028)
        row["consolidation"] = float(f.get("consolidation_per_q") or 0.042)
        if any(i.startswith("debt_limit") for i in ids):
            spells = f["debt_limit"]
            med = float(spells.get("median_days") or 136)
            hz = 1 - 0.5 ** (91.25 / med)
            row.update({"debt_limit_raise": hz / 2, "debt_limit_suspend": hz / 2, "debt_limit_standoff": 1 - hz})
            row["no_new_law"] = 0.0            # while the limit binds, "no law" is the standoff
            notes.append(f"上限约束：2011 年以来 10/10 次解决、中位 {med:.0f} 天 → 本季解决约 {hz:.2f}；提高与暂停对半"
                         "（判断值）；78 次上限行动 0 次违约")
            others = sum(v for k, v in row.items() if not k.startswith("debt_limit") and k != "no_new_law")
            for k in [k for k in row if k.startswith("debt_limit")]:
                row[k] *= max(1.0 - others, 0.0)
        else:
            row["no_new_law"] = max(1.0 - sum(row.values()), 0.0)
        return {"p": normalize(row, ids, 0.01), "source": "fiscal_record", "label": "国会决策记录（W_A02、A11）",
                "kind": "history", "notes": notes}
    if prior_id == "treasury":
        return {"p": normalize(TREASURY_ROW, ids, 0.01), "source": "treasury", "label": TREASURY_ROW["note"],
                "kind": "judgment", "notes": notes}
    if prior_id == "china_trade":
        rows = A.reaction_rows("china", leaders)
        rr = next((x for x in rows if x["follower_domain"] == "trade_response"), None)
        if rr:
            return {"p": normalize(rr["row"], ids, 0.01), "source": f"reaction:president.{rr['leader_option']}",
                    "label": "条件表：8/8 次报复（Bown 2021）；其余行判断值", "kind": "history", "notes": notes}
        expiry_q = facts.get("trade_truce_expiry_q")
        if expiry_q and quarter and expiry_q == quarter:
            row = {"no_response": 0.45, "negotiate_truce": 0.40, "retaliate_tariffs": 0.10,
                   "retaliate_export_controls": 0.05}
            notes.append("休战本季到期：3 次休战 12 个月内破裂 2 次（判断值行）")
        else:
            row = {"no_response": 0.80, "negotiate_truce": 0.15, "retaliate_tariffs": 0.05}
        return {"p": normalize(row, ids, 0.01), "source": "china_record", "label": "没有美方升级时的行（判断值）",
                "kind": "judgment", "notes": notes}
    if prior_id == "opec_quota":
        rows = A.reaction_rows("opec", leaders)
        rr = next((x for x in rows if x["follower_domain"] == "quota"), None)
        if rr:
            return {"p": normalize(rr["row"], ids, 0.01), "source": "reaction:belligerents.escalate",
                    "label": "冲突升级时的条件行（判断值）", "kind": "judgment", "notes": notes}
        st = {**state, "oil_disruption": bool(facts.get("oil_disruption_ongoing"))}
        if (gates.get("opec") or {}).get("below_breakeven_2q"):
            st["brent_below_breakeven_2q"] = True
        reg = opec_regime(st)
        row = tables["opec"]["rows"][reg]
        lab = {"crash": "崩跌行：2 季内减产 5/8（W_A03）", "spike": "飙升或高于区间（W_A03）",
               "inside": "区间内（判断值）"}[reg]
        return {"p": normalize(row, ids, 0.01), "source": f"opec_record:{reg}", "label": lab, "kind": "history",
                "notes": notes}
    if prior_id == "bigtech_capex":
        reg = bigtech_regime(state)
        row = dict(tables["bigtech"]["rows"][reg])
        for rr in A.reaction_rows("bigtech", leaders):
            for k, m in rr["row"].items():
                row[k] = row.get(k, 0.0) * float(m)
            notes.append(f"{rr['leader']}·{rr['leader_option']}：{'、'.join(rr['row'])} ×1.5（判断值）")
        t = (gates.get("bigtech") or {}).get("tilt") or {}
        if t.get("capex_raise_large", 1.0) < 1.0:
            row["capex_raise_large"] *= float(t["capex_raise_large"])
            notes.append("G-L7：净债务/EBITDA > 1 倍，大幅上调 ×0.5")
        return {"p": normalize(row, ids, 0.01), "source": f"bigtech:{reg}",
                "label": {"boom": "繁荣", "normal": "平常", "bust": "萧条"}[reg] + "体制（判断值，锚定 NIPA 条件频率）",
                "kind": "judgment", "notes": notes}
    if prior_id == "belligerents":
        return {"p": normalize(tables["belligerents"]["rows"], ids, 0.01), "source": "belligerents_record",
                "label": "停火 = 退出风险率 0.42（W_A06）；升级、僵持判断值", "kind": "history", "notes": notes}
    if prior_id == "courts":
        odds = inst.get("ruling_tariff_odds") if dom.id == "tariff_authority" else None
        if odds is not None:
            o = float(odds)
            row = {"uphold": 1 - o, "strike_down": o / 2, "strike_down_refunds": o / 2}
            return {"p": normalize(row, ids, 0.0), "source": "user_odds", "label": "用户或市场给的概率", "kind": "market",
                    "notes": ["裁定无效的两种退款结果对半（判断值）"]}
        return {**_uniform(ids, "无可靠基率（0.5）"), "notes": notes}
    if prior_id == "voter_midterm":
        odds = inst.get("house_flip_odds")
        if odds is not None:
            o = float(odds)
            return {"p": normalize({"house_flips": o, "house_holds": 1 - o}, ids, 0.0), "source": "market_odds",
                    "label": "用户输入的市场赔率（选民只用先验，c = 0）", "kind": "market", "notes": notes}
        r = tables["voter"]["midterm_house_lost_given_held"]
        p = float(r["p"])
        return {"p": normalize({"house_flips": p, "house_holds": 1 - p}, ids, 0.0), "source": "voter_record",
                "label": f"历史基率：执政党失去众议院 {r['k']}/{r['n']}（1994 年以来 5/6；待核），已标注", "kind": "history",
                "notes": ["没有市场赔率：用历史基率"]}
    if prior_id == "voter_presidential":
        r = tables["voter"]["presidential_incumbent_party"]
        p = float(r["p"])
        return {"p": normalize({"incumbent_party_wins": p, "opposition_wins": 1 - p}, ids, 0.0),
                "source": "voter_record", "label": f"执政党赢得白宫 {r['k']}/{r['n']}（1952–2024）", "kind": "history",
                "notes": notes}
    return _uniform(ids)


def hazard_table(tables: Mapping | None = None) -> list[dict]:
    """Nature's outside view (shown beside the model's own tail mass, §6.5)."""
    t = (tables or load_priors())["nature"]
    rows = []
    for hid, key in (("oil_disruption", "oil_disruption"), ("gpr_episode", "gpr_episode"),
                     ("financial_accident", "financial_accident"), ("pandemic", "pandemic")):
        r = t[key]
        rows.append({"id": hid, "p_q": r.get("p"), "jeffreys95": r.get("jeffreys95"), "k": r.get("k"), "n": r.get("n"),
                     "basis": r.get("basis") or r.get("source") or r.get("why")})
    return rows


def priors_summary(tables: Mapping | None = None) -> dict:
    """A compact view for /api/agents/config: what each prior rests on and whether it was reproduced locally."""
    t = tables or load_priors()
    return {"version": t.get("version"), "built": t.get("built", False), "reproduced": copy.deepcopy(t.get("summary")),
            "fed_today": (t.get("fed") or {}).get("prior_next_macro") or FED_RESEARCH["prior_2026Q4_macro"],
            "hazards": hazard_table(t)}
