"""Does knowing today's real-time macro regime improve the one-year forecast out of sample?

Opt-in build step regime_forecast -> results/regime_forecast.json. The protocol is the pre-registered
walk-forward of the regime study, run against the current one-year forecast (analysis/forecast.py):

  origins   forecast.forecast_origins(panel): every December from 1984 with a full realized year after it,
            plus the latest such month (42 today); N_PATHS paths
  seeds     SEED_BASES. At origin i every method draws with seed base + i: common random numbers with the
            current forecast, whose unconditional and analog draws share that seed, so paired differences
            carry the weights' effect and little Monte Carlo noise
  current   the analog kernel at the share of the available starts that the forecast module's own selection
            picks (the lowest validation CRPS at seed base forecast.SEED, i.e. forecast.json's choice); the
            same in every seed set
  families  a1, a2   the analog kernel with real-time regime indicators added to the state
            b, bc    the analog kernel x (1 for starts whose origin had the origin's real-time label, else
                     lambda): six labels or three groups; a thin pool shrinks toward the plain kernel
            c, cc    the prior restricted to starts whose origin had the origin's real-time label / group;
                     a thin pool shrinks toward the prior
            a and b take their share (and lambda) per seed set from the validation half (origins <= SPLIT),
            the way the forecast module chooses its own
  adoption  ADOPTION: lower risk-asset CRPS than the current forecast in both halves in every seed set, the
            seed-averaged rank IC down by at most MAX_RANK_IC_DROP, 90% coverage down by at most
            MAX_COVERAGE_DROP. The oracle (ex-post labels of the whole panel) shows what a look-ahead would
            be worth and is never eligible.

Look-ahead guard. A variant only sees realtime_regime(panel, t): the panel cut at t, classified, and only the
classifier's one-sided outputs kept. check_frame refuses a frame that reaches past the origin or carries an
ex-post or NBER column. guard_tests() checks the guard on the data: truncation (the frame built at t equals
the whole-panel run's one-sided outputs up to t), a scrambled future (every row after t replaced: the weights
and the forecasts at t must not move) and its power (the ex-post labels and the oracle do move).

    python -m scenariolab.analysis.regime_forecast        # writes results/regime_forecast.json (a few minutes)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from .. import config, jsonio
from ..context import EngineContext
from ..engine import regimes
from ..engine.ep import effective_number
from ..engine.pooling import MIN_ENS, MIN_EPISODES, SHRINK_STEPS, episode_ens, month_frequency
from . import forecast as F

H = F.HORIZON
N_PATHS = F.N_PATHS
SEED_BASES = (F.SEED, 1023, 2023, 3023, 4023)
SHARES = F.ANALOG_SHARES
LAMBDAS = (0.0, 0.25, 0.5)
A1 = ("infl_trend_12m", "sahm_gap")
A2 = ("infl_trend_12m", "sahm_gap", "ip_below_peak_rt", "u_rise_rt")
MAX_RANK_IC_DROP = 0.05
MAX_COVERAGE_DROP = 0.02
MACRO_KEYS = ("cpi_yoy", "unrate", "ff")
KEEP = ("p5", "p25", "p50", "p75", "p95", "mean", "prob_loss", "pit", "crps")

# the classifier's one-sided outputs: the value at month m uses data through m only
REALTIME_KEYS = ("label_rt", "contraction_rt", "severe_rt", "high_infl", "cpi_neg", "cpi_yoy", "infl_trend_12m",
                 "infl_trend_6m", "unrate_3m", "sahm_gap", "ip_yoy", "ip_drawdown_24m", "ip_below_peak_rt",
                 "ip_leg_decline_rt", "u_rise_rt", "episode_cpi_mean_rt")
# two-sided outputs: they use months after m and never enter a forecast made at an origin
EXPOST_KEYS = ("label", "contraction", "severe", "episode", "episode_ip_decline", "episode_cpi_mean")

# three groups: 0 = no contraction and CPI y/y <= 4% (扩张, 物价负增长), 1 = 通胀, 2 = any contraction
COARSE = {regimes.CODE["expansion"]: 0, regimes.CODE["deflation"]: 0, regimes.CODE["inflation"]: 1,
          regimes.CODE["recession"]: 2, regimes.CODE["depression"]: 2, regimes.CODE["stagflation"]: 2, -1: -1}
COARSE_NAME = {0: "非收缩·通胀≤4%", 1: "通胀", 2: "收缩"}

FAMILIES = {
    "a1": "相似起点 + 通胀趋势、Sahm 差值",
    "a2": "相似起点 + 通胀趋势、Sahm、工业产出离高点、失业率升幅",
    "b": "相似起点 × 同一实时体制加权（6 类）",
    "bc": "相似起点 × 同一实时体制加权（3 组：非收缩·通胀≤4% / 通胀 / 收缩）",
    "c": "只从同一实时体制的起点抽（6 类）",
    "cc": "只从同一实时体制的起点抽（3 组）",
}
ORACLE = {"c_oracle": "只从同一事后体制的起点抽（6 类，前视演示）", "cc_oracle": "只从同一事后体制的起点抽（3 组，前视演示）"}


def _tag(share: float) -> str:
    """Key suffix of a share: 1/32 -> "32", 1/2 -> "2"."""
    return str(round(1.0 / share))


def _lam(lam: float) -> str:
    return {0.0: "0", 0.25: "25", 0.5: "50"}[lam]


def settings_of(family: str) -> list[str]:
    """The method keys a family chooses among on the validation half."""
    if family in ("a1", "a2"):
        return [f"{family}_{_tag(s)}" for s in SHARES]
    if family in ("b", "bc"):
        return [f"{family}{_lam(lam)}_{_tag(s)}" for lam in LAMBDAS for s in SHARES]
    return [family]


def setting_text(key: str) -> str:
    """A method key in words: "b50_2" -> "λ 0.5，可用起点的 1/2"."""
    if key in ("c", "cc", "c_oracle", "cc_oracle"):
        return "按体制限定的无条件抽样"
    head, share = key.rsplit("_", 1)
    txt = f"可用起点的 1/{share}"
    lam = head.lstrip("abc")
    return f"λ {int(lam) / 100:g}，{txt}" if head.startswith("b") and lam else txt


# ---------------------------------------------------------------- the look-ahead guard

class LookAheadError(RuntimeError):
    pass


def check_frame(frame: pd.DataFrame, t: pd.Timestamp) -> None:
    """Raise unless `frame` stops at the origin and holds real-time columns only."""
    if len(frame) and frame.index.max() > t:
        raise LookAheadError(f"regime frame reaches {frame.index.max():%Y-%m}, past the origin {t:%Y-%m}")
    bad = [c for c in frame.columns
           if c in EXPOST_KEYS or (str(c).startswith(("nber", "rg_")) and not str(c).startswith("rg_rt_"))]
    if bad:
        raise LookAheadError(f"ex-post columns at an origin: {bad}")


def realtime_regime(panel: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Real-time regime labels and indicators of every month <= t, computed from data through t only."""
    P = panel.loc[:t]
    if not len(P) or P.index.max() != t:
        raise LookAheadError(f"origin {t} is not a panel month")
    r = regimes.classify(*(P[regimes.INPUTS[k]].to_numpy(float) for k in ("cpi_yoy", "unrate", "ip")))
    df = pd.DataFrame({k: r[k] for k in REALTIME_KEYS}, index=P.index)
    df["label_rt"] = df["label_rt"].astype(int)
    df["coarse_rt"] = df["label_rt"].map(COARSE).astype(int)
    check_frame(df, t)
    return df


def expost_regime(panel: pd.DataFrame) -> pd.DataFrame:
    """Ex-post labels of the whole panel. LOOK-AHEAD: only for the guard tests and the oracle demonstration."""
    r = regimes.classify(*(panel[regimes.INPUTS[k]].to_numpy(float) for k in ("cpi_yoy", "unrate", "ip")))
    df = pd.DataFrame({"label": r["label"].astype(int)}, index=panel.index)
    df["coarse"] = df["label"].map(COARSE).astype(int)
    return df


# ---------------------------------------------------------------- weights at one origin

class Origin:
    """The one-year forecast's engine at origin t (forecast.origin_engine) and the origin's state."""

    def __init__(self, panel: pd.DataFrame, spec, t: pd.Timestamp):
        self.t = t
        eng = F.origin_engine(panel, spec, t)
        self.db, self.anchor, self.cma, self.vm, self.prior = eng["db"], eng["anchor"], eng["cma"], eng["vm"], eng["prior"]
        self.state = F.state_at(panel, t)
        self.n_avail = F.available_starts(self.db)

    def simulate(self, probs: np.ndarray, n_paths: int, seed: int) -> dict:
        return F.simulate_year(self.db, self.anchor, self.cma, probs, n_paths, seed, valuation=self.vm)


def start_labels(frame: pd.DataFrame, db, col: str, block: int = H) -> np.ndarray:
    """Label of the origin month s - 1 of every block start s (start 0 has none: -1)."""
    lab = frame[col].reindex(db.dates).to_numpy()
    n_starts = len(db) - block + 1
    out = np.full(n_starts, -1, int)
    out[1:] = lab[: n_starts - 1]
    return out


def thinness(q: np.ndarray, T: int, block: int = H) -> tuple[float, float]:
    """Effective months and effective 12-month episodes of block-start probabilities (the engine's checks)."""
    return effective_number(month_frequency(q, block, H, T)), episode_ens(q)


def guard(q_target: np.ndarray, q_fallback: np.ndarray, T: int) -> tuple[np.ndarray, float]:
    """The engine's rule for thin view sets: keep q_target if it rests on at least MIN_ENS effective months
    and MIN_EPISODES effective episodes, else mix toward q_fallback with the largest weight that does."""
    def good(q):
        m, e = thinness(q, T)
        return m >= MIN_ENS and e >= MIN_EPISODES

    if q_target.sum() <= 0:
        return q_fallback, 0.0
    if good(q_target):
        return q_target, 1.0
    lo, hi, best = 0.0, 1.0, q_fallback
    for _ in range(SHRINK_STEPS):
        mid = 0.5 * (lo + hi)
        q = (1.0 - mid) * q_fallback + mid * q_target
        if good(q):
            lo, best = mid, q
        else:
            hi = mid
    return best, lo


def weights_at(o: Origin, reg: pd.DataFrame, expost: pd.DataFrame | None = None) -> tuple[dict, dict]:
    """Every method's block-start probabilities at origin o.t and their diagnostics. an_<k> is the current
    forecast's kernel at the share 1/k (forecast.analog_weights)."""
    t, db = o.t, o.db
    check_frame(reg, t)
    T = len(db)
    ks = db.frame[[f"s_{k}" for k in F.KEY_STATES]].to_numpy(float)
    W, D = {"uncond": o.prior}, {}
    for s in SHARES:
        W[f"an_{_tag(s)}"], info = F.analog_weights(db, o.state, share=s)
        D[f"an_{_tag(s)}"] = {"flat": info["flat"]}
    for name, extra in (("a1", A1), ("a2", A2)):
        X = np.concatenate([ks, reg[list(extra)].reindex(db.dates).to_numpy(float)], axis=1)
        x_now = np.concatenate([o.state, reg.loc[t, list(extra)].to_numpy(float)])
        for s in SHARES:
            W[f"{name}_{_tag(s)}"], _, _ = F.kernel_weights(db, X, x_now, s * o.n_avail)
    for tag, col in (("b", "label_rt"), ("bc", "coarse_rt")):
        same = start_labels(reg, db, col) == int(reg.loc[t, col])
        for lam in LAMBDAS:
            mult = np.where(same, 1.0, lam)
            for s in SHARES:
                plain = W[f"an_{_tag(s)}"]
                if same[1:].any():
                    q, _, _ = F.kernel_weights(db, ks, o.state, s * o.n_avail, mult=mult)
                    q, kappa = guard(q, plain, T)
                else:                              # no start in the origin's regime: the plain kernel
                    q, kappa = plain, 0.0
                key = f"{tag}{_lam(lam)}_{_tag(s)}"
                W[key], D[key] = q, {"kappa": kappa, "pool": int(same[1:].sum())}
    sources = [("c", reg, "label_rt"), ("cc", reg, "coarse_rt")]
    if expost is not None:
        sources += [("c_oracle", expost, "label"), ("cc_oracle", expost, "coarse")]
    for key, frame, col in sources:
        now = int(frame.loc[t, col])
        same = start_labels(frame, db, col) == now
        q = o.prior * same
        if q.sum() > 0:
            q, kappa = guard(q / q.sum(), o.prior, T)
        else:
            q, kappa = o.prior, 0.0
        W[key], D[key] = q, {"kappa": kappa, "pool": int(same[1:].sum()), "label": now}
    return W, D


# ---------------------------------------------------------------- the walk-forward

def _compact(entry: dict) -> dict:
    return {k: (v if not isinstance(v, dict) else {f: v[f] for f in KEEP if f in v}) for k, v in entry.items()}


def walk_forward(panel: pd.DataFrame, spec, dates: list[pd.Timestamp], seed_bases=SEED_BASES,
                 n_paths: int = N_PATHS, log=None) -> tuple[dict, list[dict]]:
    """{seed base: rows} in forecast_origin's format (one key per method), and per origin the weights'
    diagnostics. The weights do not depend on the seed: each origin is set up once and simulated per seed;
    methods with identical weights share one simulation."""
    expost = expost_regime(panel)                      # oracle only
    rows = {sb: [] for sb in seed_bases}
    diag = []
    for i, t in enumerate(dates):
        o = Origin(panel, spec, t)
        reg = realtime_regime(panel, t)                # data through t only
        W, D = weights_at(o, reg, expost)
        for sb in seed_bases:
            runs, seen = {}, {}
            for key, q in W.items():
                h = q.tobytes()
                if h not in seen:
                    seen[h] = o.simulate(q, n_paths, sb + i)
                runs[key] = seen[h]
            real, assets, macro = F.score_runs(panel, t, runs)
            rows[sb].append({"date": t.strftime("%Y-%m"), "shocks": F.shock_names(t + pd.DateOffset(months=1), real["end"]),
                             "label_rt": int(reg.loc[t, "label_rt"]), "coarse_rt": int(reg.loc[t, "coarse_rt"]),
                             "label_expost": int(expost.loc[t, "label"]),
                             "assets": {a: _compact(e) for a, e in assets.items()}, "macro": macro})
        diag.append({"date": t.strftime("%Y-%m"), "label_rt": regimes.ID_OF[int(reg.loc[t, "label_rt"])],
                     "flat": {k: d["flat"] for k, d in D.items() if "flat" in d},
                     "kappa": {k: d["kappa"] for k, d in D.items() if "kappa" in d},
                     "pool": {k: d["pool"] for k, d in D.items() if k in ("b0_2", "bc0_2", "c", "cc")}})
        if log:
            log(f"{t:%Y-%m} {regimes.NAME[regimes.ID_OF[int(reg.loc[t, 'label_rt'])]]}")
    return rows, diag


# ---------------------------------------------------------------- scoring

def subsets(rows: list[dict], split: str = F.SPLIT) -> dict[str, np.ndarray]:
    cut = pd.Timestamp(split)
    d = np.array([pd.Timestamp(r["date"]) <= cut for r in rows])
    s = np.array([bool(r["shocks"]) for r in rows])
    idx = np.arange(len(rows))
    return {"all": idx, "val": idx[d], "test": idx[~d], "ex_shock": idx[~s]}


def paired(x: np.ndarray, y: np.ndarray) -> dict:
    """Mean of x - y over the finite pairs, its paired t statistic (ddof 1) and n."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 2:
        return {"diff": float(d.mean()) if n else None, "t": None, "n": n}
    sd = float(d.std(ddof=1))
    return {"diff": float(d.mean()), "t": float(d.mean() / sd * np.sqrt(n)) if sd > 0 else None, "n": n}


def risk_crps(rows: list[dict], key: str) -> float | None:
    """The forecast module's risk-asset CRPS of method `key` (per asset over origins, then over assets)."""
    return F.method_scores(rows, key)["risk_avg"]["crps"]


def select(rows: list[dict], family: str, sel: np.ndarray) -> tuple[str, dict]:
    """The forecast module's rule: the family's setting with the lowest validation risk-asset CRPS."""
    val = {k: risk_crps([rows[i] for i in sel], k) for k in settings_of(family)}
    return min(val, key=val.get), val


def _macro_crps(rows: list[dict], key: str, k: str) -> np.ndarray:
    return np.array([r["macro"][k][key]["crps"] for r in rows], float)


def compare(rows: list[dict], key: str, ref: str) -> dict:
    """Scores of `key` and its paired differences against `ref` (CRPS: key - ref, positive = worse) on every
    subset; rank IC, coverage and the macro changes over all origins."""
    parts = subsets(rows)
    pk, pr = F.per_origin_crps(rows, key), F.per_origin_crps(rows, ref)
    out = {}
    for tag, sel in parts.items():
        sub = [rows[i] for i in sel]
        ref_crps = risk_crps(sub, ref)
        d = paired(pk[sel], pr[sel])
        out[tag] = {"n": int(len(sel)), "crps": risk_crps(sub, key), "ref_crps": ref_crps, **d,
                    "rel": d["diff"] / ref_crps if d["diff"] is not None and ref_crps else None}
    ks, rs = F.method_scores(rows, key), F.method_scores(rows, ref)
    out["rank_ic"], out["ref_rank_ic"] = ks["rank_ic"]["mean"], rs["rank_ic"]["mean"]
    out["coverage90"], out["ref_coverage90"] = ks["risk_avg"]["coverage90"], rs["risk_avg"]["coverage90"]
    macro = {}
    for k in MACRO_KEYS:
        recs = [r for r in rows if "naive" in r["macro"][k]]
        a, n = _macro_crps(recs, key, k), np.array([r["macro"][k]["naive"]["crps"] for r in recs], float)
        mk, mr = _macro_crps(rows, key, k), _macro_crps(rows, ref, k)
        macro[k] = {"rel_vs_current": float((mk.mean() - mr.mean()) / mr.mean()),
                    "skill_vs_naive": float(1.0 - a.mean() / n.mean()) if len(recs) and n.mean() > 0 else None}
    out["macro"] = macro
    return out


def by_label(rows: list[dict], key: str, ref: str) -> dict:
    """Risk-asset CRPS of `key` minus `ref` grouped by the origin's real-time label: {name: {n, diff, t}}."""
    pk, pr = F.per_origin_crps(rows, key), F.per_origin_crps(rows, ref)
    out = {}
    for code in sorted({r["label_rt"] for r in rows}):
        sel = np.array([r["label_rt"] == code for r in rows])
        out[regimes.NAME[regimes.ID_OF[code]]] = {"n": int(sel.sum()), **paired(pk[sel], pr[sel])}
    return out


def fragility(rows: list[dict], key: str, ref: str, k: int = 2) -> dict:
    """How much of the test-half difference of `key` (minus `ref`, risk-asset CRPS) rests on its k best
    origins (largest gains); `*_rel` divide by `ref`'s mean CRPS on the same origins."""
    test = subsets(rows)["test"]
    pr = F.per_origin_crps(rows, ref)[test]
    d = F.per_origin_crps(rows, key)[test] - pr
    order = np.argsort(d)[:k]
    keep = np.setdiff1d(np.arange(d.size), order)
    rest = float(d[keep].mean()) if keep.size else None
    return {"test": float(d.mean()), "test_rel": float(d.mean() / pr.mean()),
            "best_origins": [rows[test[j]]["date"] for j in order], "best_contrib": [float(d[j]) for j in order],
            "test_without_best": rest, "test_without_best_rel": rest / float(pr[keep].mean()) if keep.size else None}


def adoption(per_seed: list[dict], max_ic_drop: float = MAX_RANK_IC_DROP,
             max_cov_drop: float = MAX_COVERAGE_DROP) -> dict:
    """The pre-registered adoption rule on a family's per-seed results ({val, test: CRPS difference against the
    current forecast, positive = worse; ic_delta; cov_delta}): lower CRPS in both halves in every seed set, the
    seed-averaged rank IC down by at most `max_ic_drop`, 90% coverage down by at most `max_cov_drop`."""
    both = [s["val"] < 0 and s["test"] < 0 for s in per_seed]
    ic = float(np.mean([s["ic_delta"] for s in per_seed])) if per_seed else float("nan")
    cov = float(np.mean([s["cov_delta"] for s in per_seed])) if per_seed else float("nan")
    checks = {"both_halves_every_seed": bool(per_seed) and all(both), "rank_ic": ic >= -max_ic_drop,
              "coverage": cov >= -max_cov_drop}
    return {"better_both_halves_seeds": int(sum(both)), "n_seeds": len(per_seed), "rank_ic_delta": ic,
            "coverage90_delta": cov, "checks": checks, "passes": all(checks.values())}


def _agg(values: list) -> dict:
    x = np.array([v for v in values if v is not None], float)
    return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())} if x.size else \
        {"mean": None, "min": None, "max": None}


# ---------------------------------------------------------------- guard tests on the data

def scramble(panel: pd.DataFrame, t: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Every row after t replaced: numeric columns shuffled and sign-flipped, IP 40% lower, unemployment +5 pp,
    CPI y/y 12%."""
    rng = np.random.default_rng(seed)
    Q = panel.copy()
    fut = Q.index > t
    k = int(fut.sum())
    if k == 0:
        return Q
    perm = rng.permutation(k)
    for c in [c for c in Q.columns if pd.api.types.is_numeric_dtype(Q[c])]:
        v = Q.loc[fut, c].to_numpy(float)
        Q.loc[fut, c] = -v[perm]
    Q.loc[fut, "lvl_ip"] = panel.loc[fut, "lvl_ip"].to_numpy(float) * 0.6
    Q.loc[fut, "lvl_unrate"] = panel.loc[fut, "lvl_unrate"].to_numpy(float) + 5.0
    Q.loc[fut, "s_cpi_yoy"] = 12.0
    return Q


def _same_frame(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return (a.index.equals(b.index) and list(a.columns) == list(b.columns)
            and all(np.array_equal(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True) for c in a.columns))


def guard_tests(panel: pd.DataFrame, spec, current: str, dates: list[pd.Timestamp], sims: bool = True) -> dict:
    """G1 truncation, G2 scrambled future (weights and, with `sims`, a few forecasts), G3 its power, G4 the
    code guards."""
    whole = realtime_regime(panel, panel.index[-1])
    months = sorted(set(dates) | set(panel.loc["1973-01":].index[::6]))
    g1 = [f"{t:%Y-%m}" for t in months if not _same_frame(realtime_regime(panel, t), whole.loc[:t])]
    expost_true = expost_regime(panel)
    g2_fail, g3_labels, g3_oracle = [], 0, 0
    for i, t in enumerate(dates):
        scrambled = scramble(panel, t, 100 + i)
        o1, o2 = Origin(panel, spec, t), Origin(scrambled, spec, t)
        r1, r2 = realtime_regime(panel, t), realtime_regime(scrambled, t)
        ex2 = expost_regime(scrambled)
        W1, _ = weights_at(o1, r1, expost_true)
        W2, _ = weights_at(o2, r2, ex2)
        same_w = all(np.array_equal(W1[k], W2[k]) for k in W1 if "oracle" not in k)
        same_s = True
        if sims:
            for k in dict.fromkeys(("uncond", current, "a1_8", "b0_4", "bc25_2", "c", "cc")):
                s1, s2 = o1.simulate(W1[k], 200, 7 + i), o2.simulate(W2[k], 200, 7 + i)
                same_s &= bool(np.array_equal(s1["growth"], s2["growth"])
                               and all(np.array_equal(s1["macro"][m], s2["macro"][m]) for m in s1["macro"]))
        if not (_same_frame(r1, r2) and same_w and same_s):
            g2_fail.append(f"{t:%Y-%m}")
        g3_labels += int((expost_true.loc[:t, "label"] != ex2.loc[:t, "label"]).any())
        g3_oracle += int(any(not np.array_equal(W1[k], W2[k]) for k in ("c_oracle", "cc_oracle")))
    errs = []
    t = dates[min(10, len(dates) - 1)]
    reg = realtime_regime(panel, t)
    for frame, what in ((whole, "frame past the origin"), (reg.assign(label=0), "ex-post column"),
                        (reg.assign(nber_recession=0), "NBER column"), (reg.assign(rg_contraction=0.0), "rg_ ex-post column")):
        try:
            check_frame(frame, t)
            errs.append(f"not refused: {what}")
        except LookAheadError:
            pass
    return {"truncation": {"months_checked": len(months), "mismatches": g1, "pass": not g1},
            "scrambled_future": {"origins": len(dates), "failures": g2_fail, "pass": not g2_fail, "sims_checked": sims},
            "power": {"origins_where_expost_labels_moved": g3_labels, "origins_where_oracle_weights_moved": g3_oracle},
            "code_guards": {"pass": not errs, "errors": errs}}


# ---------------------------------------------------------------- report

def current_share(rows23: list[dict]) -> tuple[float, dict]:
    """The forecast module's own choice from the seed-base-SEED rows (= forecast.json's selection): the share
    whose plain kernel has the lowest validation risk-asset CRPS."""
    val = subsets(rows23)["val"]
    sub = [rows23[i] for i in val]
    crps = {s: risk_crps(sub, f"an_{_tag(s)}") for s in SHARES}
    return min(crps, key=crps.get), crps


def _pct(x: float | None, d: int = 1) -> str:
    return "—" if x is None else f"{x:+.{d}%}"


def gain_note(rel: float | None, t: float | None) -> str:
    """The forecast module's claim rule on a seed-averaged relative CRPS change `rel` (negative = better) whose
    paired t of the per-origin differences is `t` (negative = better): "（t = -2.4）" when the gain passes
    (beyond CLAIM_SKILL with |t| of at least CLAIM_T), else with 「在随机误差之内」."""
    if rel is None:
        return ""
    if F._verdict(-rel, None if t is None else -t) > 0:
        return f"（t = {t:.1f}）"
    return f"（t = {t:.1f}，在随机误差之内）" if t is not None else "（在随机误差之内）"


def report_text(fams: list[dict], baseline: dict, oracle: dict, n: int, seeds: int, split: str) -> tuple[str, str]:
    """(summary text, verdict text), generated from the numbers. CRPS changes are relative to the current
    forecast, seed-averaged; negative = better. A test-half gain is stated with its seed-averaged t and, unless
    it passes the forecast module's claim rule (gain_note), as within the noise."""
    passing = [f for f in fams if f["passes"]]
    if passing:
        v = "采用：" + "、".join(f"{f['name']}" for f in passing) + "通过了全部采用条件。"
    else:
        best = max(fams, key=lambda f: f["better_both_halves_seeds"])
        most = (f"最多的是「{best['name']}」，{best['better_both_halves_seeds']} 组" if best["better_both_halves_seeds"]
                else "没有一类在任何一组种子里两半都更好")
        v = (f"未采用：{len(fams)} 类做法中没有一类在 {seeds} 组随机种子里都同时在验证期和检验期好于现行预测"
             f"（{most}）。一年期预测不使用宏观体制。")
    lead = min(fams, key=lambda f: f["risk_crps_rel"]["test"]["mean"])
    rt = lead["risk_crps_rel"]
    frag = lead.get("fragility_seed23") or {}
    frag_txt = ""
    if rt["test"]["mean"] < 0:
        frag_txt = (f"检验期 CRPS 降得最多的是「{lead['name']}」：相对现行预测检验期 {_pct(rt['test']['mean'], 2)}"
                    f"{gain_note(rt['test']['mean'], (lead.get('t') or {}).get('test'))}、"
                    f"验证期 {_pct(rt['val']['mean'], 2)}（负 = 更好）")
        best_o, rest = frag.get("best_origins") or [], frag.get("test_without_best_rel")
        if best_o and rest is not None:
            frag_txt += (f"，但这主要来自 {'、'.join(best_o)} 这 {len(best_o)} 个起点：去掉它们后检验期为 {_pct(rest, 2)}"
                         "（随机种子 23）。" if rest >= 0 else
                         f"；随机种子 23 下改进最多的是 {'、'.join(best_o)}，去掉这 {len(best_o)} 个起点后检验期为 "
                         f"{_pct(rest, 2)}。")
        else:
            frag_txt += "。"
    parts = [f"{grp}验证期 {_pct(o['risk_crps_rel']['val']['mean'], 2)}、检验期 {_pct(o['risk_crps_rel']['test']['mean'], 2)}"
             f"（{o['better_both_halves_seeds']} 组种子两半都更好）"
             for grp, o in (("6 类", oracle.get("c_oracle") or {}), ("3 组", oracle.get("cc_oracle") or {}))
             if o.get("risk_crps_rel")]
    orc_txt = (f"作为对照，用事后才知道的体制标签（前视）只从同一体制的起点抽，相对现行预测的 CRPS：{'；'.join(parts)}"
               "（负 = 更好）。这只说明前视信息值多少，不能用。") if parts else ""
    text = (f"问题：知道今天的实时宏观体制，能不能让一年期预测在样本外更准？做法：{n} 个起点、{seeds} 组随机种子，"
            "每个起点只用当时的数据和当时的实时体制标签（单边、修订后数据）；有参数的 4 类做法各自在 "
            f"{split} 及以前的起点上选集中度（和 λ），之后的起点只用来检验；对照是{baseline['name']}，"
            f"与它用同一组随机数。{v}{frag_txt}{orc_txt}")
    return text, v


def _forecast_json_check(share: float, rows: list[dict], current: str) -> dict | None:
    """Whether results/forecast.json (when present, same origins) used the same share and the same analog
    forecasts as the current forecast here (its rows are this walk-forward's main seed set)."""
    try:
        fj = jsonio.load(config.RESULTS_DIR / "forecast.json")
    except (OSError, ValueError):
        return None
    ref = fj.get("origins") or []
    if [r.get("date") for r in ref] != [r["date"] for r in rows] or not isinstance(fj.get("analog_share"), (int, float)):
        return {"same_share": False, "max_abs_diff_us_equity_p50": None}
    diff = max(abs(r["assets"]["us_equity"][current]["p50"] - x["assets"]["us_equity"]["analog"]["p50"])
               for r, x in zip(rows, ref))
    return {"same_share": abs(fj["analog_share"] - share) < 1e-12, "max_abs_diff_us_equity_p50": diff}


def run(ctx: EngineContext, seed_bases=SEED_BASES, n_paths: int = N_PATHS, origins: list | None = None,
        guards: bool = True, log=None) -> dict:
    """regime_forecast.json (see the module docstring)."""
    t0 = time.time()
    panel, spec = ctx.panel, ctx.spec
    dates = [pd.Timestamp(t) for t in (origins or F.forecast_origins(panel))]
    seed_bases = tuple(seed_bases)
    main_seed = F.SEED if F.SEED in seed_bases else seed_bases[0]
    rows, diag = walk_forward(panel, spec, dates, seed_bases, n_paths, log=log)
    seconds_walk = time.time() - t0
    main = rows[main_seed]
    # the current forecast: the forecast module's own choice, from the main seed set's plain kernels (with
    # the module's seeds these rows are the module's rows); every seed set compares against the same share
    share, val_crps = current_share(main)
    cur = f"an_{_tag(share)}"
    fams, by_label_main = [], {}
    for fam, name in FAMILIES.items():
        per_seed, chosen, cmp_all = [], [], []
        for sb in seed_bases:
            r = rows[sb]
            best, _ = select(r, fam, subsets(r)["val"])
            c = compare(r, best, cur)
            chosen.append(best)
            cmp_all.append(c)
            per_seed.append({"val": c["val"]["diff"], "test": c["test"]["diff"],
                             "ic_delta": c["rank_ic"] - c["ref_rank_ic"], "cov_delta": c["coverage90"] - c["ref_coverage90"]})
        rule = adoption(per_seed)
        best_main = chosen[seed_bases.index(main_seed)]
        by_label_main[fam] = {"setting": best_main, "labels": by_label(main, best_main, cur)}
        fams.append({
            "id": fam, "name": name, "selected_by_seed": {str(sb): k for sb, k in zip(seed_bases, chosen)},
            "selected_text": {str(sb): setting_text(k) for sb, k in zip(seed_bases, chosen)},
            "risk_crps_rel": {p: _agg([c[p]["rel"] for c in cmp_all]) for p in ("val", "test", "all", "ex_shock")},
            "risk_crps_diff": {p: _agg([c[p]["diff"] for c in cmp_all]) for p in ("val", "test", "all", "ex_shock")},
            "t": {p: _agg([c[p]["t"] for c in cmp_all])["mean"] for p in ("val", "test")},
            "rank_ic_delta": rule["rank_ic_delta"], "coverage90": _agg([c["coverage90"] for c in cmp_all])["mean"],
            "coverage90_current": _agg([c["ref_coverage90"] for c in cmp_all])["mean"],
            "macro": {k: {"rel_vs_current": _agg([c["macro"][k]["rel_vs_current"] for c in cmp_all])["mean"],
                          "skill_vs_naive": _agg([c["macro"][k]["skill_vs_naive"] for c in cmp_all])["mean"]}
                      for k in MACRO_KEYS},
            "fragility_seed23": fragility(main, best_main, cur),
            "better_both_halves_seeds": rule["better_both_halves_seeds"], "checks": rule["checks"],
            "passes": rule["passes"]})
    oracle = {}
    for key, name in ORACLE.items():
        cmp_all = [compare(rows[sb], key, cur) for sb in seed_bases]
        oracle[key] = {"name": name, "eligible": False,
                       "risk_crps_rel": {p: _agg([c[p]["rel"] for c in cmp_all]) for p in ("val", "test", "all")},
                       "better_both_halves_seeds": sum(1 for c in cmp_all if c["val"]["diff"] < 0 and c["test"]["diff"] < 0)}
    current = F.method_scores(main, cur)["risk_avg"]
    uncond = F.method_scores(main, "uncond")["risk_avg"]
    aa = {}
    for sb in seed_bases:
        pu, pa = F.per_origin_crps(rows[sb], "uncond"), F.per_origin_crps(rows[sb], cur)
        aa[str(sb)] = {p: {k: v for k, v in paired(pu[sel], pa[sel]).items() if k != "n"}
                       for p, sel in subsets(rows[sb]).items() if p != "ex_shock"}
    flat = sum(1 for d in diag if d["flat"].get(cur))
    label = F.share_label(share)
    baseline = {"name": f"现行一年期预测（相似起点法，可用起点的 {label}；{len(dates)} 个起点中 {flat} 个等于无条件抽样）",
                "share": share, "share_label": label, "flat_origins": flat,
                "val_crps_by_share": {F.share_label(s): v for s, v in val_crps.items()},
                "forecast_json": _forecast_json_check(share, main, cur),
                "risk_crps_seed23": current["crps"], "coverage90_seed23": current["coverage90"],
                "uncond_risk_crps_seed23": uncond["crps"], "aa_noise": aa,
                "aa_note": "无条件抽样减现行预测的风险资产 CRPS（逐起点配对，正 = 现行预测更好）；两者用同一组随机数，差别来自权重"}
    labels = {}
    for r in main:
        nm = regimes.NAME[regimes.ID_OF[r["label_rt"]]]
        labels[nm] = labels.get(nm, 0) + 1
    gt = guard_tests(panel, spec, cur, dates) if guards else None
    rule_txt = (f"一类做法要被采用，它在验证期选出的设定必须：在全部 {len(seed_bases)} 组随机种子里，验证期（起点 ≤ {F.SPLIT}）和检验期"
                f"的风险资产 CRPS 都低于现行预测；{len(seed_bases)} 组平均的排序相关 IC 下降不超过 {MAX_RANK_IC_DROP:g}；"
                f"90% 区间覆盖率下降不超过 {MAX_COVERAGE_DROP:g}。用事后体制标签的前视对照不参与")
    text, vtext = report_text(fams, baseline, oracle, len(dates), len(seed_bases), F.SPLIT)
    return {
        # data_end and the baseline's share and origins say which forecast this test belongs to: the regimes tab
        # compares them with regimes.json and forecast.json and warns when they differ (an opt-in step goes stale)
        "version": 1, "data_end": f"{panel.index.max():%Y-%m}",
        "question": "知道今天的实时宏观体制，能不能让一年期预测在样本外更准？", "text": text,
        "settings": {"n_paths": n_paths, "seed_bases": list(seed_bases), "origins": [f"{t:%Y-%m}" for t in dates],
                     "split": F.SPLIT, "share_grid": list(SHARES), "share_labels": [F.share_label(s) for s in SHARES],
                     "lambdas": list(LAMBDAS), "guard": {"min_ens": MIN_ENS, "min_episodes": MIN_EPISODES},
                     "common_random_numbers": True, "claim": {"min_skill": F.CLAIM_SKILL, "min_t": F.CLAIM_T}},
        "baseline": baseline, "families": fams, "oracle": oracle, "guard_tests": gt,
        "by_label_seed23": by_label_main, "labels_at_origins": labels,
        "verdict": {"adopted": [f["id"] for f in fams if f["passes"]], "rule": rule_txt, "text": vtext},
        "seconds": {"walk_forward": seconds_walk, "total": time.time() - t0},
    }


def main() -> None:
    ctx = EngineContext.load(selection=True)
    jsonio.dump(run(ctx, log=lambda m: print(f"[regime_forecast] {m}", file=sys.stderr, flush=True)),
                config.RESULTS_DIR / "regime_forecast.json")


if __name__ == "__main__":
    main()
