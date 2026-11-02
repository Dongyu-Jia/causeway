"""Rollout of the quarterly stack: path mode, step mode, the state roll Φ, continuation and export (SIM_DESIGN §3.6,
§3.7, §11; §14 P4).

The stack (Stack) is the factor layer (factors.FactorModel, P3) and the asset layer (assets.AssetLayer, P2) fitted to
one train end, with the quarterly panel they were fitted on. get_stack() caches the forward stack (train end = the
last complete data quarter, forward anchors §3.8, origin = today's quarter).

Path mode (run_path): the factor layer draws the whole H-quarter factor path at once, conditioned on every statement
(two-sided: a statement about a later quarter shapes the earlier ones, §3.6), the asset layer then runs over the
completed paths, and the state roll Φ produces each quarter's start state. A continuation of K quarters after the
scenario is the same model run on without statements: the Waggoner-Zha projection only touches the innovations of
quarters a statement constrains, so simulating H + K quarters with statements in 1..H is exactly "the scenario,
then the model unconditionally toward its anchors".

Step mode (Stepper): one quarter's conditional draw for N particles that each carry their own state, given
per-particle point pins (the agent tree branches every quarter, §3.7). It is the factor layer's own one-quarter
system (posterior draw per particle, bootstrapped whitened innovations, the C' cycle module with its block state
carried in the state, exact conditioning) and the asset layer's (L0, identities, the residual pool with its block
state, the legs, income), vectorized over particles with different states:

    y_q = C_d + Φ1_d z_{q-1} + Φ2_d z_{q-2} + P_d ε,    ε = P_d⁻¹ L_d u   (u a whitened pool vector)

Pins are restrictions on y_q. A reduced-form pin is an observation: ε is projected onto R P ε = τ - R pred in the
identity metric, which is the Waggoner-Zha projection of the reduced-form innovation in the Σ metric (§5.3). An
intervention ("do") on a field with a policy shock (d_ff -> u_ff on the shadow real rate, d_bs, d_deficit,
d_transfers, d_perstax) fixes its structural shock so that the field hits the pin: E0 macro, ordered before it in
the recursion, is unmoved in the quarter; E1 / E2 respond on impact (the do-operator of §5.3). "do_last" sets a
satellite after the quarter's draw (nothing else responds in the quarter; its own lags carry it on). The policy
rate's floor (observed ff = max(shadow, 0.1)), the log-shifted variables (u = 2 + exp(lu), spreads = 0.1 + exp(.)),
the ±4 sd saturation (d_oil and d_y10 pass through) and the dollar hold (config_q.USD_HOLD: where d_usd is not
pinned, its expected change is zero) follow the factor layer.

The state (a dict of arrays, first axis = particle; keys with "_" are private) carries the factor layer's stacked
vector of the last two quarters (_z), the inflation trend, the CPI level and history, the block states of the cycle
module and the residual pool, each particle's posterior draw and asset-layer refit, the valuation and income
states, and the public states the agent tree reads (s_ff, s_unrate, s_core_yoy, ..., "<x>_l1", "<asset>_l1",
nasdaq_4q, real_oil_d1/d2). At the root the levels the factor layer conditions on (ff, unemployment, the spreads,
y10, pi_trend) are the realized values at the end of the quarter before the origin, the research convention the
factor layer uses; the published start-of-quarter states (core and headline CPI y/y, the Sahm gap, Michigan
expectations, CAPE, P/D, the bill) come from the panel. Rolled states are the model's quarter-end values.

The state roll Φ (Stepper.roll) is accounting only: levels = previous level + the quarter's change (the pinned
value where a field was pinned), pi_trend's EWMA, the CPI level and y/y, s_zlb = 1{s_ff < 0.5}, s_core4 =
1{core y/y > 4}, the Sahm gap on quarter-end unemployment, CAPE and P/D rolled by y_mult and y_pd, s_vol, the legs'
lags, and the factor layer's stacked vector recomputed from the delivered fields (an exact bijection given the
previous state, so the step engine's internal state always agrees with what it reported). Path mode's states are the
same Φ applied to the path draws, so path and step modes share one state roll.

Exports: core_view_draws (the instruments' CoreView.from_quarterly input), disaggregate (monthly paths for monthly
consumers during the transition, labelled 季内月度路径为历史季度形态拆分（假设）), summary (the API's JSON),
attribution (stated / completed / drift / valuation / residual, §4.4) and from_qtable (a minimal QTable reading for
/api/q/run; P6's qtable.py owns validation).

The quarterly stack is 试运行 until the switch gate T-S passes. Nothing here downloads or calls an LLM.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import chi2

from . import assets as AS
from . import config_q as cq
from . import factors as F
from . import valuation as val

VERSION = "1"
ASSETS = list(cq.ASSETS)
IA = {a: i for i, a in enumerate(ASSETS)}
RISK_COLS = [IA[a] for a in AS.RISK]
NC = F.NC
I = F.IDX
ALPHA = F.ALPHA
FLOOR = cq.FF_FLOOR
CODE = {"free": 0, "reduced": 1, "do": 2, "do_last": 3, "shock": 4}
# private per-particle columns a step draw carries for the state roll (Draws.x["_q"] in the agent adapter)
PRIV = ("shadow", "d_usd_raw", "x_oil_raw", "y_dg", "y_fund", "y_eq", "res_pos", "res_left", "cyc_pos", "cyc_left",
        "floor_bound", "bound_hit", "saturated")
PV = {k: j for j, k in enumerate(PRIV)}
# the public state keys every state carries (besides <x>_l1 for every x field, d_cpi_l1, d_unrate_l2, <asset>_l1)
PUBLIC_STATES = ("s_ff", "s_unrate", "s_core_yoy", "s_cpi_yoy", "s_spread", "s_gz", "s_y10", "s_term", "s_real_y10",
                 "s_infl_trend", "s_infexp1", "s_infexp_lr", "s_rstar", "s_policy_gap", "s_tp10", "s_tb3",
                 "s_oil_real", "s_usd_real", "s_sahm", "s_zlb", "s_core4", "s_cape", "s_pd", "s_vol", "nasdaq_4q",
                 "real_oil_d1", "real_oil_d2") + tuple(f"s_pd_{a}" for a in AS.DIVIDEND_ASSETS if a != "us_equity")
STATE_ZH = {"s_ff": "政策利率（季末）", "s_unrate": "失业率（季末）", "s_core_yoy": "核心 CPI 同比",
            "s_cpi_yoy": "CPI 同比", "s_spread": "Baa − 10 年利差", "s_gz": "GZ 信用利差", "s_y10": "10 年期收益率",
            "s_term": "期限利差（10 年 − 政策利率）", "s_real_y10": "10 年实际收益率（减通胀趋势）",
            "s_infl_trend": "通胀趋势（36 个月半衰期）", "s_infexp1": "1 年通胀预期（密歇根）",
            "s_infexp_lr": "长期通胀预期（密歇根 5–10 年）", "s_rstar": "中性实际利率 r*（HLW，保持不变）",
            "s_policy_gap": "政策立场缺口", "s_tp10": "10 年期限溢价（ACM）", "s_tb3": "3 个月国库券利率",
            "s_oil_real": "实际油价（log）", "s_usd_real": "实际美元（log）", "s_sahm": "萨姆规则缺口（季末失业率）",
            "s_zlb": "零下限（s_ff < 0.5）", "s_core4": "核心通胀 > 4%", "s_cape": "log 席勒 CAPE",
            "s_pd": "log 市价/股息（CRSP）", "s_vol": "已实现波动（美股季度回报）", "nasdaq_4q": "纳指过去 4 季对数回报",
            "real_oil_d1": "实际油价 1 季变化", "real_oil_d2": "实际油价 2 季变化", "s_tariff": "有效关税率"}
FLOOR_NOTE = "政策利率钉在下限 0.1% 以下：按下限交付"
BOUND_NOTE = "失业率或信用利差钉到模型的下界以下：按下界交付"
# far outside history (GZ max about 8, quarterly unemployment max about 13): the factor layer's long-horizon tails
TAIL_GZ, TAIL_U, TAIL_RETURN = 15.0, 20.0, 1.0
TAIL_NOTE = ("因子层的长期尾部发散（GZ 信用利差 > 15、失业率 > 20% 或美股单季对数回报超过 ±100%）：这些粒子的结果"
             "远在历史之外，不可信")


def tail_flags(gz, unrate, us_total) -> np.ndarray:
    """Particles (or path-quarters) whose state left history far behind, or whose returns are not finite."""
    gz, u, r = (np.asarray(a, float) for a in (gz, unrate, us_total))
    return ~np.isfinite(r) | ~np.isfinite(gz) | (gz > TAIL_GZ) | (u > TAIL_U) | (np.abs(r) > TAIL_RETURN)


class RolloutError(ValueError):
    """A request the rollout cannot run (Chinese message)."""


# ================================================================ the fitted stack

@dataclass
class Stack:
    """The factor and asset layers fitted to one train end, with the panels they were fitted on."""
    factors: F.FactorModel
    assets: AS.AssetLayer
    Q: pd.DataFrame
    P: pd.DataFrame
    train_end: str
    origin: str
    anchors: str
    seed: int
    bill_spread: float              # training mean of the 3-month bill minus fed funds (tb3 = ff + this, §11.2)
    init: F.InitState
    version: str
    info: dict = field(default_factory=dict)
    x_fields: list = field(default_factory=list)      # the card's x fields the factor layer carries (card order)
    _systems: list = field(default_factory=list, repr=False)

    def system(self, d: int) -> dict:
        """The joint system of posterior draw d (-1: the posterior mean), with P⁻¹ and the IP innovation gain."""
        return self._systems[d]

    @property
    def n_draws(self) -> int:
        return len(self.factors.draws)


def _systems(model: F.FactorModel) -> list:
    out = []
    for d in list(range(len(model.draws))) + [None]:
        s = dict(model.system(d))
        s["Pinv"] = np.linalg.inv(s["P"])
        s["kgain"] = s["Sigma"][:, I["ip"]] / s["Sigma"][I["ip"], I["ip"]]
        out.append(s)
    return out


def _panel_digest() -> str:
    try:
        from . import panel as qpanel
        meta = qpanel.load_meta()["info"]
        stamp = f"{meta.get('inputs_digest', '')}|{meta.get('card_digest', '')}|{cq.PANEL_Q_PATH.stat().st_mtime_ns}"
    except (OSError, KeyError, ValueError):
        stamp = "no-meta"
    sel = AS.SELECTION_PATH
    stamp += f"|{sel.stat().st_mtime_ns if sel.exists() else 0}"
    return hashlib.sha256(stamp.encode("utf-8")).hexdigest()[:12]


def last_data_quarter(Q: pd.DataFrame) -> str:
    """The last quarter with realized data (the panel's last row is today's start-of-quarter states only)."""
    try:
        from . import panel as qpanel
        return str(qpanel.load_meta()["info"]["last_data_quarter"])
    except (OSError, KeyError, ValueError):
        return str(Q.index[-2])


def fit_stack(train_end: str | None = None, *, origin: str | None = None, anchors: str = "forward", seed: int = 0,
              n_boot: int | None = None, Q: pd.DataFrame | None = None, P: pd.DataFrame | None = None) -> Stack:
    """Fit both layers on the quarters up to `train_end` (default: the last data quarter) and the origin's
    initial state (default origin: the quarter after the train end). anchors: 'forward' (§3.8 judgment values,
    scenarios and the agent tree) or 'train_end' (§5.5, backtests)."""
    from ..data import panel as monthly_panel
    from . import panel as qpanel
    t0 = time.perf_counter()
    Q = qpanel.load_panel() if Q is None else Q
    P = monthly_panel.load_panel() if P is None else P
    te = str(pd.Period(train_end or last_data_quarter(Q), "Q"))
    org = str(pd.Period(origin, "Q")) if origin else str(pd.Period(te, "Q") + 1)
    if pd.Period(org, "Q") not in Q.index:
        raise RolloutError(f"起点 {org} 不在季度面板里（{Q.index[0]}..{Q.index[-1]}）")
    fm = F.fit(Q, te, anchors=anchors, P=P, seed=seed)
    al = AS.AssetLayer.fit(te, panel=P, Q=Q, n_boot=cq.PARAM_BOOT if n_boot is None else int(n_boot), seed=seed)
    tq = pd.Period(te, "Q")
    bill = float((Q["lvl_tb3"] - Q["lvl_ff"]).loc[:tq].mean())
    init = fm.initial_state(Q, org)
    digest = _panel_digest()
    version = (f"qmodel-r{VERSION}-f{F.VERSION}-a{AS.VERSION}/{te}/{anchors}/s{seed}"
               f"/b{len(al.params) if al.params is not None else 0}/{digest}")
    from . import card as qcard
    have = set(F.CORE_FIELDS) | set(fm.sats.fields)
    xf = [f for f in qcard.x_fields(qcard.load_card()) if f in have]
    st = Stack(factors=fm, assets=al, Q=Q, P=P, train_end=te, origin=org, anchors=anchors, seed=int(seed),
               bill_spread=bill, init=init, version=version, x_fields=xf, _systems=_systems(fm))
    st.info = {"version": version, "train_end": te, "origin": org, "anchors": anchors, "seed": int(seed),
               "factor": {k: fm.info[k] for k in ("lambda", "nu", "rejection_rate", "n_draws", "satellites")},
               "asset_param_draws": len(al.params) if al.params is not None else 0, "bill_spread": bill,
               "root_imputed": list(init.imputed), "fit_seconds": round(time.perf_counter() - t0, 2),
               "label": cq.TRIAL_LABEL}
    return st


_STACKS: dict = {}
_STACK_LOCK = threading.Lock()


def get_stack(train_end: str | None = None, *, origin: str | None = None, anchors: str = "forward", seed: int = 0,
              n_boot: int | None = None) -> Stack:
    """A cached fit_stack (in-process; the key includes the panels' digest, so a rebuilt panel refits)."""
    key = (train_end, origin, anchors, int(seed), n_boot, _panel_digest())
    with _STACK_LOCK:
        hit = _STACKS.get(key)
        if hit is None:
            hit = fit_stack(train_end, origin=origin, anchors=anchors, seed=seed, n_boot=n_boot)
            _STACKS.clear()                     # one fitted stack per process is enough (a few MB)
            _STACKS[key] = hit
        return hit


# ================================================================ the root state

def _qval(Q: pd.DataFrame, col: str, q, default=np.nan) -> float:
    if col not in Q.columns or q not in Q.index:
        return float(default)
    v = Q.at[q, col]
    return float(v) if pd.notna(v) else float(default)


def _tariff_level(q_last) -> float:
    """The effective tariff rate (customs duties / imports, %) of the latest published quarter, from the cache."""
    try:
        from . import sources_q as sq
        cd, im = sq.fred("B235RC1Q027SBEA", None), sq.fred("A255RC1Q027SBEA", None)
        r = (100.0 * cd / im).dropna()
        return float(r.iloc[-1]) if len(r) else float("nan")
    except Exception:  # noqa: BLE001 - display state only
        return float("nan")


def x_of_z(stack: Stack, z1: np.ndarray, z0: np.ndarray, pi0, pi_m1, ff_prev, u_prev) -> dict:
    """The x fields of the quarter whose stacked vector is z1, given the quarter before (z0, its pi_trend pi_m1,
    observed ff_prev and unemployment u_prev): the inverse of the state roll's bijection (used at the root)."""
    m = stack.factors
    g = {v: z1[..., I[v]] for v in F.CORE}
    h = {v: z0[..., I[v]] for v in F.CORE}
    sh = F.EXP_SHIFT
    lvl = lambda zz, v: (sh[v] + np.exp(zz[v])) if v in sh else zz[v]      # noqa: E731
    out = {"d_cpi": g["pc"], "d_cpi_serv": g["ps"], "d_cpi_goods": (g["pc"] - m.ws * g["ps"]) / m.wg,
           "d_ip": g["ip"] / 4.0, "d_unrate": cq.FACTOR_U_SHIFT + np.exp(g["lu"]) - u_prev,
           "d_ff": np.maximum(g["rff"] + pi0, FLOOR) - ff_prev,
           "d_y10": (g["ry10"] + pi0) - (h["ry10"] + pi_m1), "d_spread": lvl(g, "sp") - lvl(h, "sp"),
           "d_gz": lvl(g, "gz") - lvl(h, "gz"), "d_usd": g["usd"] - h["usd"], "d_oil": g["ro"] - h["ro"] + g["pc"] / 4}
    for j, f in enumerate(m.sats.fields):
        out[f] = z1[..., NC + j]
    if "x_oil_cause" in out:
        out["x_oil_cause"] = np.clip(out["x_oil_cause"], 0.0, 1.0)
    return out


def root_state(stack: Stack, n: int, *, draw_index=None, param_index=None) -> dict:
    """The start-of-quarter state at the stack's origin for n particles: copies of the start state, each particle
    with its posterior draw of the factor layer and its refit of the asset layer (balanced over the draws unless
    given, e.g. a path-mode run's own)."""
    n = int(n)
    if n < 1:
        raise RolloutError("粒子数必须是正整数")
    Q, init = stack.Q, stack.init
    q0 = pd.Period(stack.origin, "Q")
    ql, ql2 = q0 - 1, q0 - 2
    z = np.asarray(init.z, float)
    z1, z0 = z[1], z[0]
    sh = F.EXP_SHIFT
    lvl = lambda zz, v: (sh[v] + math.exp(zz[I[v]])) if v in sh else float(zz[I[v]])   # noqa: E731
    full = lambda v: np.full(n, float(v))                                            # noqa: E731
    pi0 = float(init.pi0)
    s: dict = {}
    s["_z"] = np.broadcast_to(z, (n,) + z.shape).copy()
    s["_pi"] = full(pi0)
    s["_lcpi"] = full(init.lvl_cpi0)
    s["_pc_hist"] = np.tile(np.asarray(init.pc_hist, float), (n, 1))
    s["_core_hist"] = np.tile(np.asarray(init.core_hist, float), (n, 1))
    u_win = Q["lvl_unrate"].reindex(pd.period_range(q0 - 4, ql, freq="Q")).to_numpy(float)
    s["_u_win"] = np.tile(u_win, (n, 1))
    D = stack.n_draws
    if draw_index is None:
        per = int(math.ceil(n / max(D, 1)))
        draw_index = np.minimum(np.arange(n) // per, D - 1) if D else np.full(n, -1)
    s["_draw"] = np.asarray(draw_index, float).copy()
    B = len(stack.assets.params) if stack.assets.params is not None else 0
    if param_index is None:
        param_index = (np.arange(n) * B) // n if B else np.full(n, -1)
    s["_pidx"] = np.asarray(param_index, float).copy()
    for k in ("_cyc_pos", "_cyc_left", "_res_pos", "_res_left"):
        s[k] = np.zeros(n)
    a0 = stack.assets.start_state(Q, q0)
    s["_y_eq_l1"] = full(a0["y_eq_l1"])
    s["_fund_lags"] = np.tile(np.asarray(a0["y_fund_lags"], float), (n, 1))
    s["_dg_l1"] = full(a0["y_dg_l1"])
    # ---- public levels: the quarter-end values the factor layer conditions on
    ff0, u0 = float(init.ff0), float(init.u0)
    y10_0 = float(z1[I["ry10"]]) + pi0
    s["s_ff"], s["s_unrate"] = full(ff0), full(u0)
    s["s_spread"], s["s_gz"] = full(lvl(z1, "sp")), full(lvl(z1, "gz"))
    s["s_y10"] = full(y10_0)
    s["s_term"] = s["s_y10"] - s["s_ff"]
    s["s_real_y10"] = s["s_y10"] - pi0
    s["s_infl_trend"] = full(pi0)
    # ---- published start-of-quarter states (panel)
    for k, col in (("s_core_yoy", "s_core_yoy"), ("s_cpi_yoy", "s_cpi_yoy"), ("s_infexp1", "s_infexp1"),
                   ("s_infexp_lr", "s_infexp_lr"), ("s_rstar", "s_rstar"), ("s_tb3", "s_tb3"),
                   ("s_oil_real", "s_oil_real"), ("s_usd_real", "s_usd_real"), ("s_sahm", "s_sahm"),
                   ("s_core4", "s_core4"), ("s_cape", "s_cape"), ("s_pd", "s_pd"), ("s_vol", "s_vol")):
        s[k] = full(_qval(Q, col, q0))
    s["s_tp10"] = full(_qval(Q, "lvl_tp10", ql, _qval(Q, "s_tp10", q0)))
    s["s_zlb"] = (s["s_ff"] < cq.ZLB_FF).astype(float)
    s["s_policy_gap"] = s["s_ff"] - s["s_infexp1"] - s["s_rstar"]
    tl = _tariff_level(ql)
    if math.isfinite(tl):
        s["s_tariff"] = full(tl)
    for a in AS.DIVIDEND_ASSETS:
        if a != "us_equity":
            s[f"s_pd_{a}"] = full(a0[f"s_pd_{a}"])
    ro_m3 = _qval(Q, "lvl_oil_real", q0 - 3, float(z0[I["ro"]]))
    s["real_oil_d1"] = full((z1[I["ro"]] - z0[I["ro"]]) / 100.0)
    s["real_oil_d2"] = full((z1[I["ro"]] - ro_m3) / 100.0)
    # ---- last quarter's x values (published, else as the factor layer imputed them) and asset returns
    pi_m1 = _qval(Q, "lvl_infl_trend", ql2, pi0)
    xz = x_of_z(stack, z1, z0, pi0, pi_m1, _qval(Q, "lvl_ff", ql2, ff0), _qval(Q, "lvl_unrate", ql2, u0))
    for f in stack.x_fields + ["d_cpi"]:
        v = _qval(Q, f, ql)
        s[f"{f}_l1"] = full(v if math.isfinite(v) else float(xz[f]))
    s["d_unrate_l2"] = full(_qval(Q, "d_unrate", ql2, 0.0))
    for a in ASSETS:
        s[f"{a}_l1"] = full(_qval(Q, f"yt_{a}", ql, _qval(Q, f"y_{a}", ql, 0.0)))
    nas = Q["yt_nasdaq"].reindex(pd.period_range(q0 - 4, ql, freq="Q")).to_numpy(float)
    nas = np.where(np.isfinite(nas), nas, 0.0)
    s["_nas_win"] = np.tile(nas, (n, 1))
    s["nasdaq_4q"] = full(nas.sum())
    return s


# ================================================================ step mode (per-particle states)

def _ss(state: dict, key: str) -> np.ndarray:
    return np.asarray(state[key], float)


@dataclass
class StepDraw:
    """One quarter's joint draw per particle: x fields (card units, + d_cpi), asset log total returns and y_mult,
    income (simple), the CPI log change, residual-quarter labels and the private columns the state roll needs."""
    x: dict
    y: dict
    inc: dict
    cpi_q: np.ndarray
    priv: np.ndarray                      # (N, len(PRIV))
    drawn: np.ndarray                     # (N,) the asset residual's quarter label
    drawn_factor: np.ndarray              # (N,) the factor innovation's quarter label
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.cpi_q))


class _Fields:
    """Observables of the pinnable fields on the stacked vector y_q, and their per-particle targets."""

    def __init__(self, stack: Stack):
        self.stack = stack
        m = stack.factors
        self.m = m
        self.nz = m.n
        self.sat = {f: NC + j for j, f in enumerate(m.sats.fields)}
        self.vec: dict = {}

    def known(self, f: str) -> bool:
        return f in self.sat or f in F.CORE_FIELDS or f in ("d_cpi",) or f in F.SHOCKS

    def obs(self, f: str) -> np.ndarray:
        v = self.vec.get(f)
        if v is not None:
            return v
        m, v = self.m, np.zeros(self.nz)
        if f in self.sat:
            v[self.sat[f]] = 1.0
        elif f == "d_cpi_serv":
            v[I["ps"]] = 1.0
        elif f == "d_cpi_goods":
            v[I["pc"]], v[I["ps"]] = 1.0 / m.wg, -m.ws / m.wg
        elif f == "d_cpi":
            v[I["pc"]] = 1.0
        elif f == "d_ip":
            v[I["ip"]] = 0.25
        elif f == "d_oil":
            v[I["ro"]], v[I["pc"]] = 1.0, 0.25
        elif f == "d_y10":
            v[I["ry10"]], v[I["pc"]] = 1.0, ALPHA
        elif f == "d_usd":
            v[I["usd"]] = 1.0
        elif f == "d_unrate":
            v[I["lu"]] = 1.0
        elif f == "d_spread":
            v[I["sp"]] = 1.0
        elif f == "d_gz":
            v[I["gz"]] = 1.0
        elif f == "d_ff":
            v[I["rff"]], v[I["pc"]] = 1.0, ALPHA
        else:
            raise F.FactorError(f"字段不存在或这个训练期没有估计：{f}")
        self.vec[f] = v
        return v

    @staticmethod
    def exp_var(f: str) -> str | None:
        return {"d_unrate": "lu", "d_spread": "sp", "d_gz": "gz"}.get(f) if f in F.EXP_OBS else None

    def lin_offset(self, f: str, S: dict) -> np.ndarray | float:
        """b_f: card value = obs . y + b_f for the linear fields."""
        if f == "d_oil":
            return -S["z1"][:, I["ro"]]
        if f == "d_y10":
            return -(S["y10_0"] - (1.0 - ALPHA) * S["pi0"])
        if f == "d_usd":
            return -S["z1"][:, I["usd"]]
        if f in ("d_spread", "d_gz") and self.exp_var(f) is None:
            return -S["z1"][:, I["sp" if f == "d_spread" else "gz"]]
        return 0.0

    def level0(self, f: str, S: dict) -> np.ndarray:
        return {"d_unrate": S["u0"], "d_spread": S["sp0"], "d_gz": S["gz0"]}[f]

    def target(self, f: str, v: np.ndarray, S: dict) -> tuple[np.ndarray, np.ndarray]:
        """(tau on obs . y, the card value delivered) for value pins v (not d_ff at the floor)."""
        ev = self.exp_var(f)
        if ev is not None:
            lo = cq.PIN_U_MIN if f == "d_unrate" else cq.PIN_SPREAD_MIN
            lv0 = self.level0(f, S)
            lv1 = np.maximum(lv0 + v, lo)
            return np.log(lv1 - F.EXP_SHIFT[ev]), lv1 - lv0
        if f == "d_ff":
            return S["ff0"] + v - (1.0 - ALPHA) * S["pi0"], v
        return v - self.lin_offset(f, S), v


def _take(S: dict, rows: np.ndarray, n: int) -> dict:
    """The per-particle entries (first axis n) of a dict of start arrays, at rows."""
    return {k: (v[rows] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n else v) for k, v in S.items()}


def _innov(sysm: dict, pred: np.ndarray, U: np.ndarray, target: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Reduced-form innovations L u; in the cycle module's particles the quarter's own innovation is moved so IP
    growth meets its target, the other variables along their innovation regression on IP (factors._run_paths)."""
    E = U @ sysm["L"].T
    if act.any():
        gap = target - (pred[:, I["ip"]] + E[:, I["ip"]])
        E = E + np.where(act, gap, 0.0)[:, None] * sysm["kgain"][None, :]
    return E


def _censored_moments(mu, sd, a):
    """Mean and sd of max(X, a), X ~ N(mu, sd^2)."""
    sd = np.maximum(sd, 1e-12)
    zz = (a - mu) / sd
    Pz, pz = ndtr(zz), np.exp(-0.5 * zz ** 2) / math.sqrt(2.0 * math.pi)
    m1 = a * Pz + mu * (1.0 - Pz) + sd * pz
    m2 = a ** 2 * Pz + (mu ** 2 + sd ** 2) * (1.0 - Pz) + sd * (mu + a) * pz
    return m1, np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))


def _solve(sysm: dict, pred: np.ndarray, E: np.ndarray, red: list, do: list, shock: list) -> np.ndarray:
    """y = pred + P ε for the draw pred + E after the restrictions, in structural coordinates ε = P⁻¹ E.

    red: [(obs (n,), tau (m,))] reduced-form rows; do: [(pos, obs, tau)] interventions (the structural shock at pos
    is set so obs . y = tau); shock: [(pos, value (m,))] stated structural shocks in field units (P[pos,pos] ε_pos).
    The free shocks are projected onto the reduced rows in the identity metric (the WZ projection in Σ)."""
    P = sysm["P"]
    eps = E @ sysm["Pinv"].T
    n = P.shape[0]
    fixed = [p for p, _ in shock] + [p for p, _, _ in do]
    for p, v in shock:
        eps[:, p] = v / P[p, p]
    free = np.ones(n, bool)
    free[fixed] = False
    PN = P[:, free]
    base = pred + (eps[:, [p for p, _ in shock]] @ P[:, [p for p, _ in shock]].T if shock else 0.0)
    if do:
        Dp = [p for p, _, _ in do]
        O = np.stack([o for _, o, _ in do])                              # (kD, n)
        K = np.linalg.inv(O @ P[:, Dp])                                  # (kD, kD)
        OPN = O @ PN
        cD = np.column_stack([t for _, _, t in do]) - base @ O.T         # (m, kD)
    if red:
        R = np.stack([o for o, _ in red])                                # (kR, n)
        tau = np.column_stack([t for _, t in red])                       # (m, kR)
        if do:
            RPD = R @ P[:, Dp]
            A = R @ PN - RPD @ K @ OPN
            b = tau - base @ R.T - cD @ K.T @ RPD.T
        else:
            A = R @ PN
            b = tau - base @ R.T
        eN = eps[:, free]
        AA = A @ A.T
        rhs = (b - eN @ A.T).T
        try:
            alpha = np.linalg.solve(AA, rhs).T
        except np.linalg.LinAlgError:
            alpha = (np.linalg.pinv(AA) @ rhs).T
        eps[:, free] = eN + alpha @ A
    if do:
        eps[:, Dp] = (cD - eps[:, free] @ OPN.T) @ K.T
    return pred + eps @ P.T


class Stepper:
    """Step mode for N particles with their own states (§3.7): draw() one quarter's joint draw given per-particle
    pins, roll() the state roll Φ, rarity() the §3.5 statistic per particle, impute() the root's unpublished
    fields.

    Options: param_uncertainty (each particle's own posterior draw and asset refit; else the posterior mean and the
    point fit), cycle (the C' module in quarters without a growth pin), usd_hold (config_q.USD_HOLD), saturate
    (±4 sd), cma ("no_rerate", the forward default, or "historical"), block (asset residual block), gaussian
    (N(0, I) whitened innovations instead of the bootstrap)."""

    def __init__(self, stack: Stack, *, param_uncertainty: bool = True, cycle: bool = True,
                 usd_hold: bool | None = None, saturate: bool = True, cma: str = cq.CMA_FORWARD,
                 block: int | None = None, gaussian: bool = False):
        self.stack = stack
        self.m = stack.factors
        self.al = stack.assets
        self.param_uncertainty = bool(param_uncertainty)
        self.cycle = bool(cycle)
        self.usd_hold = cq.USD_HOLD if usd_hold is None else bool(usd_hold)
        self.saturate = bool(saturate)
        self.cma = cma
        self.block = int(block or cq.RESID_BLOCK)
        self.gaussian = bool(gaussian)
        self.fields = _Fields(stack)
        m = self.m
        self.nz = m.n
        self.sat = self.fields.sat
        self.ip_hist = np.asarray(m.ip_hist, float)
        self.cyc_n = len(self.ip_hist) - cq.CYCLE_BLOCK + 1
        self.cyc_sd = cq.CYCLE_SD_FACTOR * float(np.std(self.ip_hist, ddof=1))
        self.cyc_new_mean = float(self.ip_hist[:self.cyc_n].mean())
        self.res_starts = self.al.pool.starts(self.block)
        self.pool_mean = m.pool.mean(axis=0)
        self.do_pos = {f: m.names.index(cq.POLICY_DO[u]) for f, u in cq.DO_FIELDS.items()
                       if cq.POLICY_DO[u] in m.names}
        self.shock_pos = {u: m.names.index(v) for u, v in cq.POLICY_DO.items() if v in m.names}
        self.j_oil = self.sat.get("x_oil_cause")
        self.use_params = self.param_uncertainty and self.al.params is not None
        self.ip_fund = self.al.pool.columns.index("y_fund")
        self.ip_dg = self.al.pool.columns.index("y_dg")

    # ------------------------------------------------------------ helpers
    def root(self, n: int) -> dict:
        return root_state(self.stack, n)

    def _start(self, state: dict) -> dict:
        z = _ss(state, "_z")
        z1, z0 = z[:, 1, :], z[:, 0, :]
        pi0 = _ss(state, "_pi")
        ff0 = _ss(state, "s_ff")
        sh = F.EXP_SHIFT
        S = {"z1": z1, "z0": z0, "pi0": pi0, "ff0": ff0, "u0": cq.FACTOR_U_SHIFT + np.exp(z1[:, I["lu"]]),
             "sp0": sh["sp"] + np.exp(z1[:, I["sp"]]) if "sp" in sh else z1[:, I["sp"]],
             "gz0": sh["gz"] + np.exp(z1[:, I["gz"]]) if "gz" in sh else z1[:, I["gz"]],
             "y10_0": z1[:, I["ry10"]] + pi0}
        S["shadow0"] = np.where(ff0 > FLOOR + 1e-9, ff0, z1[:, I["rff"]] + pi0)
        d = _ss(state, "_draw").astype(int) if self.param_uncertainty else np.full(len(ff0), -1)
        S["draw"] = np.where(d < 0, -1, np.minimum(d, self.stack.n_draws - 1))
        return S

    def _pins(self, pins: dict | None, structural: dict | None, modes: dict | None, N: int):
        """(ids, values (N, k), codes (N, k)): the pinned fields per particle and how each is imposed."""
        modes = dict(cq.STEP_GIVEN_MODES if modes is None else modes)
        pins = {k: np.broadcast_to(np.asarray(v, float), (N,)) for k, v in (pins or {}).items()}
        structural = {k: np.broadcast_to(np.asarray(v, float), (N,)) for k, v in (structural or {}).items()}
        ids = sorted(set(pins) | set(structural))
        V = np.full((N, len(ids)), np.nan)
        C = np.zeros((N, len(ids)), int)
        for j, f in enumerate(ids):
            if not self.fields.known(f):
                raise F.FactorError(f"字段不存在或这个训练期没有估计：{f}")
            if f in F.SHOCKS:
                if f not in self.shock_pos:
                    raise F.FactorError(f"{f}：这个训练期没有对应的政策方程")
                src = structural.get(f, pins.get(f))
                ok = np.isfinite(src)
                V[ok, j], C[ok, j] = src[ok], CODE["shock"]
                continue
            if f in pins:
                ok = np.isfinite(pins[f])
                mode = modes.get(f, "reduced")
                if mode not in cq.PIN_MODES:
                    raise F.FactorError(f"{f}：未知的固定方式 {mode!r}（{cq.PIN_MODES}）")
                if mode == "do" and f not in self.do_pos:
                    mode = "reduced"
                if mode == "do_last" and f not in self.sat:
                    mode = "reduced"
                V[ok, j], C[ok, j] = pins[f][ok], CODE[mode]
            if f in structural:
                ok = np.isfinite(structural[f])
                if f in self.do_pos:
                    code = CODE["do"]
                elif f in self.sat:
                    code = CODE["do_last"]
                else:
                    raise F.FactorError(f"{f} 没有对应的政策冲击，不能写成干预（只能作为观测值固定）")
                V[ok, j], C[ok, j] = structural[f][ok], code
        return ids, V, C

    def _prior(self, S: dict) -> tuple[np.ndarray, np.ndarray]:
        sm = self.stack.system(-1)
        P1, P2 = sm["Phi"]
        return sm["C"] + S["z1"] @ P1.T + S["z0"] @ P2.T, sm["Sigma"]

    def _saturate(self, ids, V, C, S, prior):
        """Pinned values clipped to the prior predictive mean ± SATURATION_SD sd (posterior mean, no cycle module):
        the factor layer's saturation; d_oil and d_y10 pass through, do_last pins are not conditioned on."""
        if not self.saturate or not len(ids):
            return V.copy(), np.zeros(V.shape, bool)
        pm, Sig = prior
        Vu = V.copy()
        clipped = np.zeros(V.shape, bool)
        for j, f in enumerate(ids):
            act = np.isfinite(V[:, j]) & (C[:, j] != CODE["do_last"])
            if not act.any() or f in cq.MECHANICAL_FIELDS:
                continue
            if f in F.SHOCKS:
                sdv = float(self.stack.system(-1)["P"][self.shock_pos[f], self.shock_pos[f]])
                mu = np.zeros(len(V))
            else:
                o = self.fields.obs(f)
                s_lin = float(math.sqrt(max(o @ Sig @ o, 0.0)))
                ml = pm @ o
                ev = self.fields.exp_var(f)
                if ev is not None:
                    e = np.exp(ml + 0.5 * s_lin ** 2)
                    mu = F.EXP_SHIFT[ev] + e - self.fields.level0(f, S)
                    sdv = e * math.sqrt(math.expm1(s_lin ** 2))
                elif f == "d_ff":
                    mu, sdv = _censored_moments(ml + (1.0 - ALPHA) * S["pi0"], s_lin, FLOOR)
                    mu = mu - S["ff0"]
                else:
                    mu, sdv = ml + self.fields.lin_offset(f, S), s_lin
            lo, hi = mu - cq.SATURATION_SD * sdv, mu + cq.SATURATION_SD * sdv
            v = V[:, j]
            c = np.clip(v, lo, hi)
            hit = act & (np.abs(c - v) > 1e-12)
            Vu[hit, j] = c[hit] if np.ndim(c) else c
            clipped[hit, j] = True
        return Vu, clipped

    def _rows(self, ids, codes, Vg, S_g, floor_rows: bool):
        """Restrictions of one group (identical codes): (red, do, shock, last, ff_floor_j)."""
        red, do, shock, last = [], [], [], []
        floor_j = None
        for j, f in enumerate(ids):
            c = codes[j]
            if c == CODE["free"]:
                continue
            v = Vg[:, j]
            if c == CODE["shock"]:
                shock.append((self.shock_pos[f], v))
                continue
            if c == CODE["do_last"]:
                last.append((self.sat[f], v))
                continue
            if f == "d_ff" and np.all(S_g["ff0"] + v <= FLOOR + 1e-9):
                floor_j = j
                if not floor_rows:
                    continue
                tau = np.full(len(v), FLOOR) - (1.0 - ALPHA) * S_g["pi0"]
            else:
                tau, _ = self.fields.target(f, v, S_g)
            o = self.fields.obs(f)
            if c == CODE["do"]:
                do.append((self.do_pos[f], o, tau))
            else:
                red.append((o, tau))
        return red, do, shock, last, floor_j

    def _draw_u(self, rng: np.random.Generator, m: int):
        if self.gaussian:
            return rng.standard_normal((m, self.nz)), np.full(m, "gauss", dtype=object)
        k = rng.integers(0, len(self.m.pool), size=m)
        return self.m.pool[k], self.m.pool_labels[k].astype(object)

    # ------------------------------------------------------------ the factor layer's quarter
    def _factor(self, state, S, ids, V, C, rng):
        N = len(S["ff0"])
        U, lab = self._draw_u(rng, N)
        cyc_new = _ss(state, "_cyc_left") <= 0
        cyc_start = rng.integers(0, self.cyc_n, size=N)
        cyc_noise = rng.standard_normal(N)
        cyc_pos = np.where(cyc_new, cyc_start, _ss(state, "_cyc_pos").astype(int) + 1)
        cyc_pos = np.minimum(cyc_pos, len(self.ip_hist) - 1)
        cyc_left = np.where(cyc_new, cq.CYCLE_BLOCK - 1, _ss(state, "_cyc_left") - 1)
        cyc_target = self.ip_hist[cyc_pos] + self.cyc_sd * cyc_noise
        cyc_mean = np.where(cyc_new, self.cyc_new_mean, self.ip_hist[cyc_pos])
        growth = np.zeros(N, bool)
        for j, f in enumerate(ids):
            if f in cq.GROWTH_FIELDS:
                growth |= np.isfinite(V[:, j]) & (C[:, j] != CODE["free"])
        active = (~growth) & self.cycle
        prior = self._prior(S)
        Vu, clipped = self._saturate(ids, V, C, S, prior)
        floor_flag = np.zeros(N, bool)
        if "d_ff" in ids:
            j = ids.index("d_ff")
            floor_flag = np.isfinite(Vu[:, j]) & (S["ff0"] + Vu[:, j] <= FLOOR + 1e-9) & \
                np.isin(C[:, j], (CODE["reduced"], CODE["do"]))
        key = np.column_stack([S["draw"], C, floor_flag.astype(int)])
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        Y = np.empty((N, self.nz))
        Ym = np.empty((N, self.nz)) if self.usd_hold else None
        mean_u = np.zeros(self.nz) if self.gaussian else self.pool_mean
        for g in range(int(inv.max()) + 1):
            rows = np.flatnonzero(inv == g)
            r0 = rows[0]
            sysm = self.stack.system(int(S["draw"][r0]))
            Sg = _take(S, rows, N)
            P1, P2 = sysm["Phi"]
            pred = sysm["C"] + Sg["z1"] @ P1.T + Sg["z0"] @ P2.T
            act = active[rows]
            codes = C[r0]
            Vg = Vu[rows]
            red, do, shock, last, floor_j = self._rows(ids, codes, Vg, Sg, floor_rows=False)
            E = _innov(sysm, pred, U[rows], cyc_target[rows], act)
            y = _solve(sysm, pred, E, red, do, shock)
            if floor_j is not None:
                y, lab_g = self._floor(sysm, pred, y, Sg, ids, codes, Vg, act, cyc_target[rows], rng, lab[rows])
                lab[rows] = lab_g
            for p, v in last:
                y[:, p] = v
            Y[rows] = y
            if Ym is not None:
                Em = _innov(sysm, pred, np.broadcast_to(mean_u, (len(rows), self.nz)), cyc_mean[rows], act)
                ym = _solve(sysm, pred, Em, red, do, shock)
                for p, v in last:
                    ym[:, p] = v
                Ym[rows] = ym
        cyc = {"pos": cyc_pos, "left": cyc_left}
        return Y, Ym, lab, cyc, Vu, clipped, floor_flag

    def _floor(self, sysm, pred, y, Sg, ids, codes, Vg, act, target, rng, lab):
        """A d_ff pin whose level is at or below the floor: the shadow rate must end at or below FLOOR. Reduced
        form: rejection rounds with fresh innovations (the truncated conditional distribution), then projection onto
        the floor; do: the policy shock is set so the shadow sits at the floor where the drawn one is above it, so
        E0 is unmoved."""
        do_mode = codes[ids.index("d_ff")] == CODE["do"]

        def shadow(yy, sub):
            return yy[:, I["rff"]] + sub["pi0"] + ALPHA * (yy[:, I["pc"]] - sub["pi0"])

        m = len(y)
        bad = shadow(y, Sg) > FLOOR + 1e-12
        lab = np.asarray(lab, dtype=object).copy()
        if not do_mode:
            for _ in range(cq.FLOOR_ROUNDS):
                if not bad.any():
                    break
                b = np.flatnonzero(bad)
                sub = _take(Sg, b, m)
                Ub, lb = self._draw_u(rng, len(b))
                Eb = _innov(sysm, pred[b], Ub, target[b], act[b])
                red, do, shock, _, _ = self._rows(ids, codes, Vg[b], sub, floor_rows=False)
                yb = _solve(sysm, pred[b], Eb, red, do, shock)
                ok = shadow(yb, sub) <= FLOOR + 1e-12
                take = b[ok]
                y[take] = yb[ok]
                lab[take] = lb[ok]
                bad[take] = False
        if bad.any():
            b = np.flatnonzero(bad)
            sub = _take(Sg, b, m)
            red, do, shock, _, _ = self._rows(ids, codes, Vg[b], sub, floor_rows=True)
            y[b] = _solve(sysm, pred[b], y[b] - pred[b], red, do, shock)     # the least change that reaches the floor
        return y, lab

    # ------------------------------------------------------------ one quarter
    def draw(self, state: dict, pins: dict | None, rng: np.random.Generator, *, structural: dict | None = None,
             modes: dict | None = None) -> StepDraw:
        """One quarter for every particle of `state`. pins {field: (N,)}: point pins in card units (NaN = free),
        read by `modes` ({field: reduced | do | do_last}; default config_q.STEP_GIVEN_MODES); structural {field:
        (N,)}: interventions (do for the policy fields, do_last for other satellites). u_* ids pin structural shocks
        in their field's units. Pinned values come back unchanged in x, except a policy rate pinned below the floor
        (delivered at 0.1) and unemployment / spreads pinned below the model's bounds (delivered at the bound)."""
        N = len(_ss(state, "s_ff"))
        S = self._start(state)
        ids, V, C = self._pins(pins, structural, modes, N)
        Y, Ym, lab, cyc, Vu, clipped, floor_bound = self._factor(state, S, ids, V, C, rng)
        res_new = _ss(state, "_res_left") <= 0
        res_start = self.res_starts[rng.integers(0, len(self.res_starts), size=N)]
        res_pos = np.where(res_new, res_start, _ss(state, "_res_pos").astype(int) + 1)
        res_pos = np.minimum(res_pos, len(self.al.pool) - 1)
        res_left = np.where(res_new, self.block - 1, _ss(state, "_res_left") - 1)
        x, priv, notes = self._outputs(S, Y, Ym, ids, V, C, clipped, floor_bound)
        x_assets, y, inc, apriv = self._assets(state, x, res_pos)
        priv[:, PV["y_dg"]] = apriv["y_dg"]
        priv[:, PV["y_fund"]] = apriv["y_fund"]
        priv[:, PV["y_eq"]] = apriv["y_eq"]
        priv[:, PV["res_pos"]] = res_pos
        priv[:, PV["res_left"]] = res_left
        priv[:, PV["cyc_pos"]] = cyc["pos"]
        priv[:, PV["cyc_left"]] = cyc["left"]
        tail = tail_flags(_ss(state, "s_gz") + x["d_gz"], _ss(state, "s_unrate") + x["d_unrate"], y["us_equity"])
        if tail.any():
            notes.append({"code": "tail", "particles": int(tail.sum()), "text": TAIL_NOTE})
        return StepDraw(x=x, y=y, inc=inc, cpi_q=x["d_cpi"] / 400.0, priv=priv,
                        drawn=np.asarray(self.al.pool.label_array(res_pos), dtype=object),
                        drawn_factor=np.asarray(lab, dtype=object), notes=notes)

    def _outputs(self, S, Y, Ym, ids, V, C, clipped, floor_bound):
        m = self.m
        N = len(S["ff0"])
        g = {v: Y[:, I[v]] for v in F.CORE}
        sh = F.EXP_SHIFT
        pi1 = S["pi0"] + ALPHA * (g["pc"] - S["pi0"])
        shadow1 = g["rff"] + pi1
        ff1 = np.maximum(shadow1, FLOOR)
        lv = lambda v: (sh[v] + np.exp(g[v])) if v in sh else g[v]           # noqa: E731
        d_usd_raw = g["usd"] - S["z1"][:, I["usd"]]
        hold = np.zeros(N)
        if Ym is not None:
            pinned_usd = np.zeros(N, bool)
            if "d_usd" in ids:
                pinned_usd = np.isfinite(V[:, ids.index("d_usd")])
            hold = np.where(pinned_usd, 0.0, Ym[:, I["usd"]] - S["z1"][:, I["usd"]])
        x = {"d_cpi": g["pc"], "d_cpi_serv": g["ps"], "d_cpi_goods": (g["pc"] - m.ws * g["ps"]) / m.wg,
             "d_ip": g["ip"] / 4.0, "d_unrate": cq.FACTOR_U_SHIFT + np.exp(g["lu"]) - S["u0"],
             "d_ff": ff1 - S["ff0"], "d_y10": (g["ry10"] + pi1) - S["y10_0"], "d_spread": lv("sp") - S["sp0"],
             "d_gz": lv("gz") - S["gz0"], "d_usd": d_usd_raw - hold,
             "d_oil": g["ro"] - S["z1"][:, I["ro"]] + g["pc"] / 4.0}
        for j, f in enumerate(m.sats.fields):
            x[f] = Y[:, NC + j].copy()
        priv = np.full((N, len(PRIV)), np.nan)
        if self.j_oil is not None:
            priv[:, PV["x_oil_raw"]] = Y[:, self.j_oil]
            x["x_oil_cause"] = np.clip(x["x_oil_cause"], 0.0, 1.0)
        priv[:, PV["shadow"]] = shadow1
        priv[:, PV["d_usd_raw"]] = d_usd_raw
        bound = np.zeros(N, bool)
        notes = []
        # pinned values come back unchanged (the conditioning used their saturated values)
        for j, f in enumerate(ids):
            ok = np.isfinite(V[:, j]) & (C[:, j] != CODE["free"])
            if not ok.any() or f in F.SHOCKS:
                continue
            v = V[:, j]
            if f == "d_ff":
                at_floor = ok & (S["ff0"] + v <= FLOOR + 1e-9)
                keep = ok & ~at_floor
                x[f] = np.where(keep, v, x[f])
                x[f] = np.where(at_floor, FLOOR - S["ff0"], x[f])
                if at_floor.any():
                    notes.append({"field": f, "particles": int(at_floor.sum()), "text": FLOOR_NOTE})
                continue
            ev = self.fields.exp_var(f)
            if ev is not None:
                lo = cq.PIN_U_MIN if f == "d_unrate" else cq.PIN_SPREAD_MIN
                lv0 = self.fields.level0(f, S)
                low = ok & (lv0 + v < lo)
                x[f] = np.where(ok & ~low, v, x[f])
                x[f] = np.where(low, lo - lv0, x[f])
                bound |= low
                if low.any():
                    notes.append({"field": f, "particles": int(low.sum()), "text": BOUND_NOTE})
                continue
            x[f] = np.where(ok, v, x[f])
            if f == "d_usd":
                priv[:, PV["d_usd_raw"]] = np.where(ok, v, priv[:, PV["d_usd_raw"]])
            if f == "x_oil_cause" and self.j_oil is not None:
                priv[:, PV["x_oil_raw"]] = np.where(ok, v, priv[:, PV["x_oil_raw"]])
        x["d_cpi"] = m.wg * x["d_cpi_goods"] + m.ws * x["d_cpi_serv"]
        nclip = clipped.sum(axis=1)
        for j, f in enumerate(ids):
            if clipped[:, j].any():
                notes.append({"field": f, "particles": int(clipped[:, j].sum()), "text": f"{f}：{F.CLIP_TEXT}"})
        priv[:, PV["floor_bound"]] = floor_bound.astype(float)
        priv[:, PV["bound_hit"]] = bound.astype(float)
        priv[:, PV["saturated"]] = nclip
        return x, priv, notes

    # ------------------------------------------------------------ the asset layer's quarter
    def asset_frame(self, state: dict, x: dict) -> dict:
        """The asset layer's inputs for one quarter (N, 1): the L0 fields and legs' regressors from x, the start
        states s_core4, s_y10, s_tb3 and the quarter-end bill max(ff + bill spread, 0)."""
        ff1 = _ss(state, "s_ff") + x["d_ff"]
        fr = {f: np.asarray(x[f], float)[:, None] for f in ("d_oil", "d_ip", "d_cpi_goods", "d_cpi_serv", "d_ff",
                                                           "d_y10", "d_spread", "d_usd", "d_unrate", "d_gz",
                                                           "d_tp10", "d_metals_oil", "d_cpi", "d_rgdp") if f in x}
        fr["s_core4"] = _ss(state, "s_core4")[:, None]
        fr["s_zlb"] = _ss(state, "s_zlb")[:, None]
        fr["s_y10"] = _ss(state, "s_y10")[:, None]
        fr["s_tb3"] = _ss(state, "s_tb3")[:, None]
        fr["lvl_tb3"] = np.maximum(ff1 + self.stack.bill_spread, 0.0)[:, None]
        return fr

    def _assets(self, state: dict, x: dict, res_pos: np.ndarray, pidx: np.ndarray | None = None):
        """The asset layer for one quarter, per particle (AssetLayer.simulate with H = 1, written out so the
        residual block, the refit and the start states come from each particle's state)."""
        al = self.al
        N = len(res_pos)
        fr = self.asset_frame(state, x)
        Fq = AS.factor_inputs(fr, al.features)
        if pidx is None:
            pidx = _ss(state, "_pidx").astype(int)
        draw = pidx if (self.use_params and np.all(pidx >= 0)) else None
        sysm = np.zeros((N, len(ASSETS)))
        sysm[:, RISK_COLS] = al._risk_mean(Fq, self.cma, draw, shape=(N, 1))[:, 0, :]
        sv = _ss(state, "s_vol")
        if al.candidates:
            sysm[:, RISK_COLS] += al._cand_mean({c: v[:, 0] for c, v in fr.items()},
                                                {"s_vol": sv, "s_pd": _ss(state, "s_pd")}, (N,))
        mech = AS.mechanical_returns(fr)
        for a, (p, _) in mech.items():
            sysm[:, IA[a]] = p[:, 0]
        e = al.pool.draws(res_pos, sv)
        price = sysm.copy()
        price[:, RISK_COLS] += e[:, RISK_COLS]
        reg = val.fund_regressors({c: v[:, 0] for c, v in fr.items()})
        f_lags = _ss(state, "_fund_lags")
        y_fund = al.valuation.fund.predict(reg, f_lags[:, 0]) + e[:, self.ip_fund]
        y_dg = al.valuation.dg.predict(reg, _ss(state, "_dg_l1")) + e[:, self.ip_dg]
        income = np.zeros_like(price)
        with np.errstate(over="ignore", invalid="ignore"):    # only particles in the factor layer's diverging tails
            for a in AS.DIVIDEND_ASSETS:
                lpd = _ss(state, "s_pd") if a == "us_equity" else _ss(state, f"s_pd_{a}")
                income[:, IA[a]] = al.income_k[a] * np.exp(-lpd) / 4.0
            income[:, IA["treasury_10y"]] = mech["treasury_10y"][1][:, 0]
            income[:, IA["cash"]] = np.expm1(price[:, IA["cash"]])
            total = np.log(np.exp(price) + income)
        total[:, IA["cash"]] = price[:, IA["cash"]]
        y_eq = price[:, IA["us_equity"]]
        y = {a: total[:, IA[a]].copy() for a in ASSETS}
        y["mult"] = y_eq - y_fund
        inc = {a: income[:, IA[a]].copy() for a in ASSETS}
        return fr, y, inc, {"y_dg": y_dg, "y_fund": y_fund, "y_eq": y_eq, "price": price, "systematic": sysm,
                            "resid": price - sysm}

    # ------------------------------------------------------------ the state roll Φ
    def roll(self, state: dict, draw) -> dict:
        """The next start-of-quarter state from a state and one quarter's draw (a StepDraw or any object with x,
        y, inc and optionally priv / x["_q"]). Accounting only (see the module docstring)."""
        m = self.m
        x = draw.x
        y = draw.y
        inc = getattr(draw, "inc", {}) or {}
        N = len(_ss(state, "s_ff"))
        q = getattr(draw, "priv", None)
        if q is None:
            q = x.get("_q")
        q = np.full((N, len(PRIV)), np.nan) if q is None else np.asarray(q, float)
        col = lambda k: q[:, PV[k]]                                           # noqa: E731
        X = lambda f: np.asarray(x[f], float)                                 # noqa: E731
        z = _ss(state, "_z")
        z1 = z[:, 1, :]
        pi0 = _ss(state, "_pi")
        sh = F.EXP_SHIFT
        out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in state.items()}
        # ---- the factor layer's stacked vector of the quarter, recomputed from the delivered fields
        pc = m.wg * X("d_cpi_goods") + m.ws * X("d_cpi_serv")
        pi1 = pi0 + ALPHA * (pc - pi0)
        ff1 = _ss(state, "s_ff") + X("d_ff")
        at_floor = np.where(np.isfinite(col("shadow")), col("shadow"), z1[:, I["rff"]] + pi0)
        shadow = np.where(ff1 > FLOOR + 1e-9, ff1, np.minimum(at_floor, FLOOR))
        u1 = _ss(state, "s_unrate") + X("d_unrate")
        sp1 = _ss(state, "s_spread") + X("d_spread")
        gz1 = _ss(state, "s_gz") + X("d_gz")
        y10_1 = _ss(state, "s_y10") + X("d_y10")
        d_usd_raw = np.where(np.isfinite(col("d_usd_raw")), col("d_usd_raw"), X("d_usd"))
        zn = np.empty_like(z1)
        zn[:, I["pc"]] = pc
        zn[:, I["ps"]] = X("d_cpi_serv")
        zn[:, I["ip"]] = 4.0 * X("d_ip")
        zn[:, I["lu"]] = np.log(np.maximum(u1 - cq.FACTOR_U_SHIFT, 1e-4))
        zn[:, I["rff"]] = shadow - pi1
        zn[:, I["ro"]] = z1[:, I["ro"]] + X("d_oil") - pc / 4.0
        zn[:, I["ry10"]] = y10_1 - pi1
        zn[:, I["sp"]] = np.log(np.maximum(sp1 - sh["sp"], 1e-4)) if "sp" in sh else sp1
        zn[:, I["gz"]] = np.log(np.maximum(gz1 - sh["gz"], 1e-4)) if "gz" in sh else gz1
        zn[:, I["usd"]] = z1[:, I["usd"]] + d_usd_raw
        for j, f in enumerate(m.sats.fields):
            zn[:, NC + j] = X(f)
        if self.j_oil is not None:
            raw = col("x_oil_raw")
            zn[:, self.j_oil] = np.where(np.isfinite(raw), raw, X("x_oil_cause"))
        out["_z"] = np.stack([z1, zn], axis=1)
        out["_pi"] = pi1
        out["_lcpi"] = _ss(state, "_lcpi") + pc / 4.0
        pch = np.column_stack([_ss(state, "_pc_hist"), pc])
        out["_pc_hist"] = pch[:, 1:]
        out["s_cpi_yoy"] = 100.0 * np.expm1(pch.sum(axis=1) / 400.0)
        if "d_core" in x:
            chh = np.column_stack([_ss(state, "_core_hist"), X("d_core")])
            out["_core_hist"] = chh[:, 1:]
            out["s_core_yoy"] = 100.0 * np.expm1(chh.sum(axis=1) / 400.0)
        out["s_core4"] = (out["s_core_yoy"] > cq.CORE4).astype(float)
        # ---- public levels
        out["s_ff"] = ff1
        out["s_unrate"] = u1
        out["s_spread"] = sp1
        out["s_gz"] = gz1
        out["s_y10"] = y10_1
        out["s_term"] = y10_1 - ff1
        out["s_real_y10"] = y10_1 - pi1
        out["s_infl_trend"] = pi1
        out["s_zlb"] = (ff1 < cq.ZLB_FF).astype(float)
        uw = _ss(state, "_u_win")
        out["s_sahm"] = u1 - uw.min(axis=1)
        out["_u_win"] = np.column_stack([uw[:, 1:], u1])
        for k, f in (("s_infexp1", "d_infexp1"), ("s_infexp_lr", "d_infexp_lr"), ("s_tp10", "d_tp10"),
                     ("s_tariff", "d_tariff")):
            if k in state and f in x:
                out[k] = _ss(state, k) + X(f)
        out["s_policy_gap"] = ff1 - out["s_infexp1"] - _ss(state, "s_rstar")
        out["s_tb3"] = np.maximum(ff1 + self.stack.bill_spread, 0.0)
        dro = zn[:, I["ro"]] - z1[:, I["ro"]]
        out["s_oil_real"] = _ss(state, "s_oil_real") + dro / 100.0
        out["real_oil_d1"] = dro / 100.0
        out["real_oil_d2"] = (zn[:, I["ro"]] - z[:, 0, I["ro"]]) / 100.0
        out["s_usd_real"] = _ss(state, "s_usd_real") + X("d_usd") / 100.0
        # ---- the asset layer's states: the P/D rolls with the delivered price return (total less income), the
        # volatility state with the return the asset layer scaled its residuals by (priv "y_eq": the same in step
        # mode; in path mode the return before a valuation statement's shift, as AssetLayer.simulate rolls it)
        inc_us = np.asarray(inc.get("us_equity", np.zeros(N)), float)
        with np.errstate(divide="ignore", invalid="ignore"):
            price_us = np.log(np.maximum(np.exp(np.asarray(y["us_equity"], float)) - inc_us, 1e-300))
        y_eq = np.where(np.isfinite(col("y_eq")), col("y_eq"), price_us)
        mult = np.asarray(y["mult"], float)
        y_fund = np.where(np.isfinite(col("y_fund")), col("y_fund"), price_us - mult)
        y_dg = col("y_dg")
        if not np.isfinite(y_dg).all():
            reg = val.fund_regressors({"d_cpi": pc, "d_ip": X("d_ip"), "d_unrate": X("d_unrate"),
                                       "d_rgdp": X("d_rgdp") if "d_rgdp" in x else np.zeros(N)})
            y_dg = np.where(np.isfinite(y_dg), y_dg, self.al.valuation.dg.predict(reg, _ss(state, "_dg_l1")))
        out["s_vol"] = np.sqrt((y_eq ** 2 + _ss(state, "_y_eq_l1") ** 2) / 2.0)
        out["_y_eq_l1"] = y_eq
        out["s_cape"] = _ss(state, "s_cape") + mult
        out["s_pd"] = _ss(state, "s_pd") + price_us - y_dg
        for a in AS.DIVIDEND_ASSETS:
            if a == "us_equity":
                continue
            tot = np.asarray(y[a], float)
            with np.errstate(over="ignore", invalid="ignore"):
                pa = np.log(np.maximum(np.exp(tot) - np.asarray(inc.get(a, np.zeros(N)), float), 1e-300))
            out[f"s_pd_{a}"] = _ss(state, f"s_pd_{a}") + pa - self.al.div_growth[a]
        fl = _ss(state, "_fund_lags")
        out["_fund_lags"] = np.column_stack([fl[:, 1], y_fund])
        out["_dg_l1"] = y_dg
        for k in ("res_pos", "res_left", "cyc_pos", "cyc_left"):
            v = col(k)
            out[f"_{k}"] = np.where(np.isfinite(v), v, 0.0)
        # ---- last quarter's values
        out["d_unrate_l2"] = _ss(state, "d_unrate_l1")
        for f in list(self.stack.x_fields) + ["d_cpi"]:
            if f in x:
                out[f"{f}_l1"] = X(f) if f != "d_cpi" else pc
        for a in ASSETS:
            out[f"{a}_l1"] = np.asarray(y[a], float).copy()
        nw = np.column_stack([_ss(state, "_nas_win")[:, 1:], np.asarray(y["nasdaq"], float)])
        out["_nas_win"] = nw
        out["nasdaq_4q"] = nw.sum(axis=1)
        return out

    # ------------------------------------------------------------ rarity and impute
    def rarity(self, state: dict, pins: dict | None, *, structural: dict | None = None,
               modes: dict | None = None) -> dict:
        """The §3.5 statistic per particle: m2 = the squared norm of the smallest structural-shock adjustment the
        pins need from the prior mean (posterior-mean model, no cycle module); under the model m2 ~ chi2(df), df =
        the number of pins; percentile and sd per constraint (the joint study's stagflation set needed 0.81)."""
        N = len(_ss(state, "s_ff"))
        S = self._start(state)
        ids, V, C = self._pins(pins, structural, modes, N)
        sysm = self.stack.system(-1)
        pred, Sig = self._prior(S)
        Vu, _ = self._saturate(ids, V, C, S, (pred, Sig))
        m2 = np.zeros(N)
        df = np.zeros(N, int)
        if ids:
            # a policy rate pinned at the floor needs no adjustment where the prior shadow rate is already below it
            floor_need = np.zeros(N, bool)
            if "d_ff" in ids:
                j = ids.index("d_ff")
                at = np.isfinite(Vu[:, j]) & (S["ff0"] + Vu[:, j] <= FLOOR + 1e-9)
                prior_shadow = pred @ self.fields.obs("d_ff") + (1.0 - ALPHA) * S["pi0"]
                floor_need = at & (prior_shadow > FLOOR)
                C = C.copy()
                C[at & ~floor_need, j] = CODE["free"]
            key = np.column_stack([C, np.isfinite(V), floor_need])
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            for g in range(int(inv.max()) + 1):
                rows = np.flatnonzero(inv == g)
                Sg = _take(S, rows, N)
                red, do, shock, _, _ = self._rows(ids, C[rows[0]], Vu[rows], Sg, floor_rows=True)
                y = _solve(sysm, pred[rows], np.zeros((len(rows), self.nz)), red, do, shock)
                eps = (y - pred[rows]) @ sysm["Pinv"].T
                m2[rows] = (eps ** 2).sum(axis=1)
                df[rows] = len(red) + len(do) + len(shock)
        pct = np.where(df > 0, chi2.cdf(m2, np.maximum(df, 1)), 0.0)
        return {"m2": m2, "df": df, "percentile": pct, "sd_per_constraint": np.sqrt(m2 / np.maximum(df, 1)),
                "warn": pct > cq.RARITY_WARN_PCT}

    def impute(self, state: dict, rng: np.random.Generator) -> dict:
        """The root's unpublished fields (the quarter before the origin) redrawn per particle from their conditional
        distribution given the published ones (the "impute" capability; root_state uses the conditional mean)."""
        stack = self.stack
        miss = [nm.split("@")[0] for nm in stack.init.imputed if nm.endswith(f"@{pd.Period(stack.origin, 'Q') - 1}")]
        pos = [stack.factors.names.index(v) for v in miss if v in stack.factors.names and v != "rff"]
        if not pos:
            return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in state.items()}
        Q, m = stack.Q, stack.factors
        q0 = pd.Period(stack.origin, "Q")
        z2 = m._row_at(Q, q0 - 3)
        z = np.asarray(stack.init.z, float)
        sysm = m.system()
        mu = sysm["C"] + sysm["Phi"][0] @ z[0] + sysm["Phi"][1] @ z2
        obs = np.ones(m.n, bool)
        obs[pos] = False
        mm, Cc = F._cond_gauss(mu, sysm["Sigma"], obs, np.where(obs, z[1], 0.0))
        L = np.linalg.cholesky(Cc + 1e-12 * np.eye(len(pos)))
        N = len(_ss(state, "s_ff"))
        draw = mm[~obs][None, :] + rng.standard_normal((N, len(pos))) @ L.T
        out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in state.items()}
        zz = _ss(state, "_z").copy()
        zz[:, 1, pos] = draw
        out["_z"] = zz
        for k, p in enumerate(pos):
            f = m.names[p]
            key = f"{f}_l1"
            if key in out:
                v = draw[:, k]
                out[key] = np.clip(v, 0.0, 1.0) if f == "x_oil_cause" else v
        return out


# ================================================================ path mode

@dataclass
class Rollout:
    """A path-mode run: factor paths (fx), asset paths (ax), the asset layer's inputs, the start-of-quarter states
    (n, T + 1) and the state after the last quarter (step mode can continue from it)."""
    stack: Stack
    H: int
    T: int
    labels: list
    fx: F.SimResult
    ax: AS.SimPaths
    frame: dict
    states: dict
    end_state: dict | None
    statements: list
    valuation: list
    info: dict
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.ax.total.shape[0])


def quarter_labels(origin: str, T: int) -> list[str]:
    q0 = pd.Period(origin, "Q")
    return [str(q0 + k) for k in range(int(T))]


def _seed(seed: int, purpose: int) -> int:
    return int(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(purpose)]).generate_state(1)[0])


def asset_inputs(stack: Stack, fx: F.SimResult, root: dict) -> dict:
    """The asset layer's per-path inputs (n, T) from completed factor paths: the x fields, the start states s_core4,
    s_zlb, s_y10, s_tb3 rolled from the factor levels, and the quarter-end bill max(ff + bill spread, 0)."""
    n, T = fx.levels["ff"].shape
    x = fx.x
    fr = {f: x[f] for f in ("d_oil", "d_ip", "d_cpi_goods", "d_cpi_serv", "d_ff", "d_y10", "d_spread", "d_usd",
                            "d_unrate", "d_gz", "d_tp10", "d_metals_oil", "d_cpi", "d_rgdp") if f in x}

    def lagged(a, first):
        return np.concatenate([np.broadcast_to(np.asarray(first, float), (n,))[:, None], a[:, :-1]], axis=1)

    c4 = fx.levels.get("core4")
    c4 = np.asarray(c4, float) if c4 is not None else np.broadcast_to(root["s_core4"][:, None], (n, T))
    fr["s_core4"] = lagged(c4, root["s_core4"])
    fr["s_zlb"] = lagged(np.asarray(fx.levels["ff"] < cq.ZLB_FF, float), root["s_zlb"])
    fr["s_y10"] = lagged(fx.levels["y10"], root["s_y10"])
    tb3 = np.maximum(fx.levels["ff"] + stack.bill_spread, 0.0)
    fr["lvl_tb3"] = tb3
    fr["s_tb3"] = lagged(tb3, root["s_tb3"])
    return fr


def path_draw(stack: Stack, fx: F.SimResult, ax: AS.SimPaths, k: int, z_prev: np.ndarray,
              pool_pos: dict | None = None) -> StepDraw:
    """Quarter k of a path-mode run as a step draw (the state roll's input): x, asset total returns, income, y_mult
    and the private columns (the shadow rate, the raw dollar change, the raw oil-cause label, the legs, the residual
    block position)."""
    n = ax.total.shape[0]
    if pool_pos is None:
        pool_pos = {lab: i for i, lab in enumerate(stack.assets.pool.labels)}
    x = {f: np.asarray(v[:, k], float) for f, v in fx.x.items()}
    y = {a: ax.total[:, k, IA[a]] for a in ASSETS}
    y["mult"] = ax.y_mult[:, k]
    inc = {a: ax.income[:, k, IA[a]] for a in ASSETS}
    priv = np.full((n, len(PRIV)), np.nan)
    priv[:, PV["shadow"]] = fx.levels["ff_shadow"][:, k]
    priv[:, PV["d_usd_raw"]] = fx.core[:, k, I["usd"]] - z_prev[:, I["usd"]]
    m = stack.factors
    if "x_oil_cause" in m.sats.fields:
        priv[:, PV["x_oil_raw"]] = fx.sats[:, k, m.sats.fields.index("x_oil_cause")]
    priv[:, PV["y_dg"]] = ax.y_dg[:, k]
    priv[:, PV["y_fund"]] = ax.y_fund[:, k]
    priv[:, PV["y_eq"]] = ax.price[:, k, IA["us_equity"]] - ax.shift[:, k, IA["us_equity"]]
    priv[:, PV["res_pos"]] = [pool_pos.get(str(v), np.nan) for v in ax.drawn[:, k]]
    B = int(cq.RESID_BLOCK)
    priv[:, PV["res_left"]] = B - 1 - (k % B)
    return StepDraw(x=x, y=y, inc=inc, cpi_q=x["d_cpi"] / 400.0, priv=priv, drawn=ax.drawn[:, k],
                    drawn_factor=fx.drawn[:, k])


def _parse_valuation(valuation, H: int, mode: str) -> list:
    if not valuation:
        return []
    if all(isinstance(v, val.ValuationStatement) for v in valuation):
        return list(valuation)
    return val.parse_statements(list(valuation), H, mode=mode)


def run_path(stack: Stack, statements=(), valuation=(), H: int | None = None, n_paths: int = 1000, seed: int = 0,
             *, continuation: int = 0, cma: str | None = None, cycle: bool = True, cycle_odds: dict | None = None,
             usd_hold: bool | None = None, param_uncertainty: bool = True, block: int | None = None,
             gaussian: bool = False, states: bool = True, mode: str = "production") -> Rollout:
    """Path mode: H scenario quarters from the stack's origin, every factor statement imposed at once (two-sided),
    then `continuation` quarters of the same model without statements; the asset layer over the whole path; the
    state roll for every quarter (states=True). statements: factors.Statement list (quarters 1..H); valuation:
    c_q.valuation entries or valuation.ValuationStatement list (quarters 1..H); cma: "no_rerate" (default, U-5) or
    "historical"; mode: "production" or "test" (test refuses valuation statements)."""
    t0 = time.perf_counter()
    H = int(cq.ROLLOUT_H if H is None else H)
    K = int(continuation)
    T = H + K
    n = int(n_paths)
    if H < 1 or K < 0 or T > cq.ROLLOUT_MAX_QUARTERS:
        raise RolloutError(f"季度数必须在 1..{cq.ROLLOUT_MAX_QUARTERS} 之间（情景 {H} + 续推 {K}）")
    if not 1 <= n <= cq.ROLLOUT_MAX_PATHS:
        raise RolloutError(f"路径数必须在 1..{cq.ROLLOUT_MAX_PATHS} 之间")
    stmts = list(statements or ())
    for st in stmts:
        if max(int(q) for q in st.quarters) > H:
            raise RolloutError(f"{st.var}：陈述的季度 {st.quarters} 超出情景长度 {H}（续推期不接受陈述）")
    vst = _parse_valuation(valuation, H, mode)
    labels = quarter_labels(stack.origin, T)
    odds = None
    if cycle_odds:
        odds = dict(cycle_odds)
        odds.setdefault("quarters", list(range(1, H + 1)))
    fx = F.simulate(stack.factors, stack.init, T, n_paths=n, seed=seed, statements=stmts, cycle=cycle,
                    cycle_odds=odds, gaussian=gaussian, param_uncertainty=param_uncertainty, usd_hold=usd_hold)
    root = root_state(stack, n, draw_index=fx.draw_index, param_index=None)
    frame = asset_inputs(stack, fx, root)
    start = stack.assets.start_state(stack.Q, stack.origin)
    use_params = param_uncertainty and stack.assets.params is not None
    with np.errstate(invalid="ignore", over="ignore"):
        ax = stack.assets.simulate(frame, start, n_paths=n, seed=_seed(seed, 2), cma=cma or cq.CMA_FORWARD,
                                   block=block, params=use_params, statements=vst, quarters=labels)
    notes = list(fx.notes)
    st_out, end = {}, None
    if states:
        st = root_state(stack, n, draw_index=fx.draw_index, param_index=ax.param_index)
        stepper = Stepper(stack, param_uncertainty=param_uncertainty, cma=cma or cq.CMA_FORWARD, block=block)
        keys = [k for k in st if not k.startswith("_") and np.ndim(st[k]) == 1]
        rec = {k: [st[k]] for k in keys}
        z_prev = st["_z"][:, 1, :]
        pool_pos = {lab: i for i, lab in enumerate(stack.assets.pool.labels)}
        for k in range(T):
            d = path_draw(stack, fx, ax, k, z_prev, pool_pos)
            st = stepper.roll(st, d)
            z_prev = st["_z"][:, 1, :]
            for kk in keys:
                rec[kk].append(st[kk])
        st_out = {k: np.column_stack(v) for k, v in rec.items()}
        end = st
    tails = tail_flags(fx.levels["gz"], fx.levels["unrate"], ax.total[:, :, IA["us_equity"]]).any(axis=1)
    if tails.any():
        notes.append({"code": "tail", "paths": int(tails.sum()), "share": float(tails.mean()), "text": TAIL_NOTE})
    info = {"tail_share": float(tails.mean()),
            "version": stack.version, "rollout": VERSION, "origin": stack.origin, "train_end": stack.train_end,
            "anchors": stack.anchors, "H": H, "continuation": K, "T": T, "n_paths": n, "seed": int(seed),
            "cma": cma or cq.CMA_FORWARD, "cycle": bool(cycle), "usd_hold": fx.info.get("usd_hold"),
            "param_uncertainty": bool(param_uncertainty), "mode": mode, "label": cq.TRIAL_LABEL,
            "factor": fx.info, "elapsed_s": round(time.perf_counter() - t0, 3)}
    return Rollout(stack=stack, H=H, T=T, labels=labels, fx=fx, ax=ax, frame=frame, states=st_out, end_state=end,
                   statements=list(fx.statements), valuation=vst, info=info, notes=notes)


def run_step(stack: Stack, statements=(), n_paths: int = 500, seed: int = 0, **kw) -> Rollout:
    """Step mode at the stack's origin (one quarter, one-sided): path mode with H = 1 (the lab's 逐季 view)."""
    return run_path(stack, statements, (), 1, n_paths, seed, **kw)


def continue_steps(stack: Stack, state: dict, K: int, seed: int = 0, **step_kw) -> tuple[dict, list]:
    """K quarters of the step engine without pins from a state (e.g. Rollout.end_state): (final state, the step
    draws). The same model as a longer path-mode run, with each particle's own state carried."""
    stp = Stepper(stack, **step_kw)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(K)):
        d = stp.draw(state, {}, rng)
        state = stp.roll(state, d)
        draws.append(d)
    return state, draws


# ================================================================ export

def stated_mask(statements, H: int, fields) -> dict[str, np.ndarray]:
    """{field: (H,) bool}: the quarters in which a statement constrains the field (directly, through a level
    observable or, for fc stances, d_gz / d_spread); used by the attribution's stated / completed split."""
    obs = {"ff": "d_ff", "ff_shadow": "d_ff", "rff": "d_ff", "y10": "d_y10", "ry10": "d_y10", "unrate": "d_unrate",
           "spread": "d_spread", "gz": "d_gz", "usd": "d_usd", "real_oil": "d_oil", "u_ff": "d_ff",
           "cpi_yoy": ("d_cpi_goods", "d_cpi_serv"), "pc": ("d_cpi_goods", "d_cpi_serv"),
           "d_cpi": ("d_cpi_goods", "d_cpi_serv")}
    out = {f: np.zeros(H, bool) for f in fields}
    for st in statements or ():
        tgt = obs.get(st.var, st.var)
        for f in (tgt if isinstance(tgt, tuple) else (tgt,)):
            if f in out:
                for q in st.quarters:
                    if 1 <= int(q) <= H:
                        out[f][int(q) - 1] = True
    return out


def attribution(roll: Rollout) -> dict:
    """The §4.4 split of every path's quarterly log price return: stated + completed + drift + candidates =
    systematic; + valuation shift + residual = the return. Per asset: the means and medians over paths of each part,
    cumulated over the scenario quarters (log)."""
    ax, st = roll.ax, roll.stack
    al = st.assets
    n, T = roll.n, roll.T
    H = roll.H
    F_ = AS.factor_inputs(roll.frame, al.features)
    use = al.params is not None and np.all(ax.param_index >= 0)
    cols = [al.f_assets.index(a) for a in AS.RISK]
    contrib = {}
    for k, f in enumerate(al.features):
        fld = AS.FIELD_OF[f]
        if use:
            b = al.params.beta[:, cols, k][ax.param_index]                 # (n, 10)
            c = F_[f][:, :, None] * b[:, None, :]
        else:
            c = F_[f][:, :, None] * al.beta.loc[AS.RISK, f].to_numpy(float)[None, None, :]
        full = np.zeros((n, T, len(ASSETS)))
        full[:, :, RISK_COLS] = c
        contrib[fld] = contrib.get(fld, 0.0) + full
    mech = AS.mechanical_returns(roll.frame)
    for fld, a in (("d_oil", "crude_oil"), ("d_y10", "treasury_10y")):
        full = contrib.setdefault(fld, np.zeros((n, T, len(ASSETS))))
        full[:, :, IA[a]] = mech[a][0]
    carry = AS.cash_return(roll.frame["s_tb3"], roll.frame["s_tb3"])
    full = contrib.setdefault("d_ff", np.zeros((n, T, len(ASSETS))))
    full[:, :, IA["cash"]] = mech["cash"][0] - carry
    drift = ax.systematic - sum(contrib.values())
    mask = stated_mask(roll.statements, T, list(contrib))
    stated = sum(contrib[f] * mask[f][None, :, None] for f in contrib)
    completed = sum(contrib[f] * (~mask[f])[None, :, None] for f in contrib)
    parts = {"stated": stated, "completed": completed, "drift": drift, "valuation": ax.shift, "residual": ax.resid}
    out = {}
    for a in ASSETS:
        j = IA[a]
        out[a] = {p: {"mean": float(np.mean(v[:, :H, j].sum(axis=1))), "median": float(np.median(v[:, :H, j].sum(axis=1)))}
                  for p, v in parts.items()}
    return {"assets": out, "zh": {"stated": "情景说了什么", "completed": "模型补了什么", "drift": "历史常态漂移",
                                  "valuation": "估值假设", "residual": "残差（均值为 0）"},
            "identity_max_abs": float(np.max(np.abs(stated + completed + drift + ax.shift + ax.resid - ax.price)))}


def core_view_draws(roll: Rollout, *, float32: bool = False) -> dict:
    """The instruments layer's input (CoreView.from_quarterly, SIM_DESIGN §11.2): per path and quarter the 13
    assets' log total returns and simple income, the start levels and changes of ff / y10 / spread, d_gz, d_tp10,
    the CPI path, the quarter-end bill and the residual-quarter labels (drawn, for pairing)."""
    ax, fx = roll.ax, roll.fx
    T = roll.T
    cast = (lambda a: np.asarray(a, np.float32)) if float32 else (lambda a: np.asarray(a, float))   # noqa: E731
    infl = np.asarray(fx.x["d_cpi"], float) / 400.0
    cpi = np.concatenate([np.ones((roll.n, 1)), np.exp(np.cumsum(infl, axis=1))], axis=1)
    q0 = pd.Period(roll.stack.origin, "Q")
    months = [(q0 - 1 + k).end_time.strftime("%Y-%m") for k in range(T + 1)]
    st = roll.states or root_state(roll.stack, 1)
    s0 = lambda k: np.broadcast_to(np.asarray(st[k], float)[..., 0] if np.ndim(st[k]) == 2   # noqa: E731
                                   else np.asarray(st[k], float)[0], (roll.n,)).copy()
    return {"freq": "Q", "labels": months, "quarters": [str(q0 - 1)] + roll.labels, "assets": list(ASSETS),
            "y_assets": cast(ax.total), "inc_assets": cast(ax.income),
            "s_ff": s0("s_ff"), "s_y10": s0("s_y10"), "s_spread": s0("s_spread"), "s_tb3": s0("s_tb3"),
            "d_ff": cast(fx.x["d_ff"]), "d_y10": cast(fx.x["d_y10"]), "d_spread": cast(fx.x["d_spread"]),
            "d_gz": cast(fx.x["d_gz"]), "d_tp10": cast(fx.x["d_tp10"]) if "d_tp10" in fx.x else None,
            "infl_q": cast(infl), "cpi": cast(cpi), "tb3": cast(roll.frame["lvl_tb3"]),
            "drawn": np.asarray(ax.drawn, dtype=object), "drawn_factor": np.asarray(fx.drawn, dtype=object),
            "label": cq.TRIAL_LABEL, "version": roll.stack.version, "origin": roll.stack.origin,
            "scenario_quarters": roll.H}


# ---------------------------------------------------------------- monthly disaggregation (export only)

_PROFILES: dict = {}


def historical_profiles(stack: Stack) -> dict:
    """Historical quarters with complete monthly data for the 13 assets up to the train end: monthly log total
    returns (n_q, 3, 13), monthly income (n_q, 3, 13), their quarter labels and the per-asset sd of the quarterly
    sums (the kNN metric)."""
    key = (stack.version,)
    hit = _PROFILES.get(key)
    if hit is not None:
        return hit
    Pm = stack.P
    end = pd.Period(stack.train_end, "Q").end_time.to_period("M").to_timestamp()
    W = Pm.loc[:end]
    lr = np.column_stack([np.log1p(W[f"r_{a}"].to_numpy(float)) for a in ASSETS])
    dv = np.column_stack([W[f"d_{a}"].to_numpy(float) if f"d_{a}" in W else np.zeros(len(W)) for a in ASSETS])
    qidx = W.index.to_period("Q")
    labels, L, D = [], [], []
    for q in pd.unique(qidx):
        rows = np.flatnonzero(qidx == q)
        if len(rows) != 3:
            continue
        a, b = lr[rows], dv[rows]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        labels.append(str(q))
        L.append(a)
        D.append(b)
    L, D = np.stack(L), np.stack(D)
    out = {"labels": np.array(labels, dtype=object), "lr": L, "inc": D, "sd": L.sum(axis=1).std(axis=0),
           "months": [[str((pd.Period(q, "Q").start_time + pd.DateOffset(months=j)).strftime("%Y-%m"))
                       for j in range(3)] for q in labels]}
    _PROFILES.clear()
    _PROFILES[key] = out
    return out


def disaggregate(cv: dict, stack: Stack, *, seed: int = 0, k: int | None = None, chunk: int = 20000) -> dict:
    """Monthly paths from a quarterly CoreView dict (core_view_draws): each path-quarter takes the monthly profile
    of one of the k nearest historical quarters (13-asset log total returns standardized by their historical sd,
    one drawn uniformly), shifted so the three months' log total returns sum to the quarter's exactly; income is
    split in that quarter's monthly proportions (sums exact); levels are interpolated linearly within the quarter.
    Export only, labelled DISAGG_LABEL."""
    k = int(k or cq.DISAGG_K)
    prof = historical_profiles(stack)
    y = np.asarray(cv["y_assets"], float)
    inc = np.asarray(cv["inc_assets"], float)
    n, T, A = y.shape
    sd = np.where(prof["sd"] > 1e-12, prof["sd"], 1.0)
    Hq = prof["lr"].sum(axis=1) / sd                      # (nh, 13)
    Yf = (y / sd).reshape(-1, A)
    rng = np.random.default_rng(seed)
    pick = np.empty(len(Yf), int)
    hn = (Hq ** 2).sum(axis=1)
    for s in range(0, len(Yf), chunk):
        blk = Yf[s:s + chunk]
        d2 = (blk ** 2).sum(axis=1)[:, None] - 2.0 * blk @ Hq.T + hn[None, :]
        near = np.argpartition(d2, min(k, d2.shape[1] - 1), axis=1)[:, :k]
        pick[s:s + chunk] = near[np.arange(len(blk)), rng.integers(0, near.shape[1], size=len(blk))]
    pick = pick.reshape(n, T)
    h = prof["lr"][pick]                                  # (n, T, 3, 13)
    m_lr = h + ((y - h.sum(axis=2)) / 3.0)[:, :, None, :]
    w = prof["inc"][pick]
    ws = w.sum(axis=2, keepdims=True)
    share = np.where(np.abs(ws) > 1e-12, w / np.where(np.abs(ws) > 1e-12, ws, 1.0), 1.0 / 3.0)
    m_inc = inc[:, :, None, :] * share
    total = np.expm1(m_lr).reshape(n, 3 * T, A)
    income = m_inc.reshape(n, 3 * T, A)

    def interp(lvl):
        lvl = np.asarray(lvl, float)
        steps = [lvl[:, :1]]
        for t in range(T):
            for j in (1, 2, 3):
                steps.append((lvl[:, t] + (j / 3.0) * (lvl[:, t + 1] - lvl[:, t]))[:, None])
        return np.concatenate(steps, axis=1)

    def levels(start, d):
        s0 = np.broadcast_to(np.asarray(start, float), (n,))[:, None]
        return np.concatenate([s0, s0 + np.cumsum(np.asarray(d, float), axis=1)], axis=1)

    q0 = pd.Period(cv["quarters"][0], "Q") if cv.get("quarters") else None
    months = []
    if q0 is not None:
        m0 = q0.end_time.to_period("M")
        months = [str(m0 + j) for j in range(3 * T + 1)]
    drawn = np.array([prof["months"][p] for p in pick.ravel()], dtype=object).reshape(n, T, 3).reshape(n, 3 * T)
    out = {"freq": "M", "labels": months, "assets": list(cv["assets"]), "total": total, "income": income,
           "cpi": interp(cv["cpi"]), "y10": interp(levels(cv["s_y10"], cv["d_y10"])),
           "ff": interp(levels(cv["s_ff"], cv["d_ff"])), "spread": interp(levels(cv["s_spread"], cv["d_spread"])),
           "d_gz": np.repeat(np.asarray(cv["d_gz"], float) / 3.0, 3, axis=1),
           "tb3": interp(np.concatenate([np.broadcast_to(np.asarray(cv.get("s_tb3", np.asarray(cv["tb3"])[:, 0]),
                                                                    float), (n,))[:, None],
                                         np.asarray(cv["tb3"], float)], axis=1)),
           "drawn": drawn, "matched_quarter": prof["labels"][pick], "label": cq.DISAGG_LABEL, "k": k}
    if cv.get("d_tp10") is not None:
        out["d_tp10"] = np.repeat(np.asarray(cv["d_tp10"], float) / 3.0, 3, axis=1)
    return out


# ---------------------------------------------------------------- the API's summary

def _q(a: np.ndarray, qs=cq.API_QUANTILES) -> dict:
    a = np.asarray(a, float)
    return {f"p{q}": np.round(np.nanpercentile(a, q, axis=0), 6).tolist() for q in qs}


SUMMARY_FIELDS = ("d_ff", "d_unrate", "d_ip", "d_rgdp", "d_cpi_goods", "d_cpi_serv", "d_core", "d_oil", "d_y10",
                  "d_y2", "d_gz", "d_spread", "d_usd", "d_tp10", "d_sent")
SUMMARY_LEVELS = ("ff", "unrate", "y10", "spread", "gz", "cpi_yoy", "core_yoy")


def summary(roll: Rollout) -> dict:
    """JSON-ready quarterly fans (p10 / p50 / p90): each asset's cumulative log total return, the main factor
    fields and levels, the implied CAPE and P/D paths, plus delivery, rarity, notes and the attribution."""
    ax, fx = roll.ax, roll.fx
    assets = {a: {"cum_total": _q(np.cumsum(ax.total[:, :, IA[a]], axis=1)),
                  "quarter_total": _q(ax.total[:, :, IA[a]])} for a in ASSETS}
    factors = {f: _q(fx.x[f]) for f in SUMMARY_FIELDS if f in fx.x}
    levels = {v: _q(fx.levels[v]) for v in SUMMARY_LEVELS if v in fx.levels}
    stated = {}
    for st in roll.statements:
        for q in st.quarters:
            stated.setdefault(int(q), set()).add(st.var)
    return {"label": cq.TRIAL_LABEL, "zh": "季度模型（试运行）：情景条件模拟，不是预测", "info": roll.info,
            "quarters": roll.labels, "scenario_quarters": roll.H,
            "assets": assets, "factors": factors, "levels": levels,
            "valuation": {"cape": _q(np.exp(ax.log_cape)), "pd": _q(np.exp(ax.log_pd)), "y_mult": _q(ax.y_mult),
                          "statements": [{"quarters": list(s.quarters), "mode": s.mode, "direction": s.direction,
                                          "size": s.size} for s in roll.valuation]},
            "stated": {str(q): sorted(v) for q, v in sorted(stated.items())},
            "delivery": fx.delivery, "rarity": {k: v for k, v in (fx.rarity or {}).items()},
            "notes": roll.notes, "attribution": attribution(roll)}


# ================================================================ a minimal QTable reading (for /api/q/run)

def _sd_table(Q: pd.DataFrame, train_end: str, fields) -> dict:
    from . import panel as qpanel
    fs = qpanel.field_stats(Q, [f for f in fields if f in Q.columns], train_end)
    return {f: float(fs.at[f, "sd"]) for f in fs.index}


def _entry(f: str, e: dict, q: int, sd: dict, shock_sd: dict, where: str):
    if not isinstance(e, dict):
        raise RolloutError(f"{where} {f}：条目必须是对象")
    if f == "x_oil_cause":
        v = e.get("value")
        if v not in cq.OIL_CAUSE:
            raise RolloutError(f"{where} x_oil_cause 必须是 supply / demand / mixed（写的是 {v!r}）")
        return F.Statement(f, (q,), cq.OIL_CAUSE[v], label=str(e.get("quote") or ""))
    unit = shock_sd.get(f) if f in F.SHOCKS else sd.get(f)
    if "value" in e and e["value"] is not None:
        v = float(e["value"])
        s = e.get("sd")
        return F.Statement(f, (q,), v, sd=float(s) if s else None, label=str(e.get("quote") or ""))
    if "sign" in e or "size" in e:
        size = e.get("size")
        if size not in cq.SIZE_BINS:
            raise RolloutError(f"{where} {f}：幅度 size 必须是 {list(cq.SIZE_BINS)} 之一")
        sign = {"+": 1.0, "-": -1.0}.get(e.get("sign"))
        if sign is None:
            raise RolloutError(f"{where} {f}：sign 必须是 '+' 或 '-'")
        if not unit or not math.isfinite(unit):
            raise RolloutError(f"{where} {f}：没有训练期标准差，不能写定性幅度")
        return F.Statement(f, (q,), sign * cq.SIZE_BINS[size] * unit, sd=cq.QTABLE_QUAL_SD * unit,
                           label=str(e.get("quote") or ""))
    if "min" in e or "max" in e:
        lo = None if e.get("min") is None else float(e["min"])
        hi = None if e.get("max") is None else float(e["max"])
        return F.Statement(f, (q,), lo=lo, hi=hi, label=str(e.get("quote") or ""))
    raise RolloutError(f"{where} {f}：条目必须写 value、sign + size 或 min / max 之一")


def from_qtable(qt: dict, stack: Stack | None = None, mode: str = "production", *, Q: pd.DataFrame | None = None,
                train_end: str | None = None, x_fields=None) -> dict:
    """A QTable/1 (Appendix B) as rollout inputs: {statements, valuation, cycle_odds, H, notes}. The structural
    reading only: P6's qtable.py validates quotes, refusals and R1-R6 (this is what /api/q/run runs on until it
    lands). fc.stance becomes soft statements on d_gz and d_spread (config_q.FC_STANCE); 未说明 adds the W-FC note.
    Needs the stack, or the quarterly panel Q with the train end and x fields (no fit: the API validates with it)."""
    from . import policy
    if mode not in ("production", "test"):
        raise RolloutError(f"未知的模式 {mode!r}（production 或 test）")
    if not isinstance(qt, dict):
        raise RolloutError("qtable 必须是 JSON 对象")
    rows = qt.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RolloutError("qtable.rows 必须是非空列表")
    try:
        H = int(qt.get("n_quarters") or max(int(r.get("q", 0)) for r in rows if isinstance(r, dict)))
    except (TypeError, ValueError) as err:
        raise RolloutError("qtable.n_quarters 必须是整数") from err
    if not 1 <= H <= cq.QTABLE_MAX_QUARTERS:
        raise RolloutError(f"n_quarters 必须在 1..{cq.QTABLE_MAX_QUARTERS} 之间")
    if stack is not None:
        Q, train_end, x_fields = stack.Q, stack.train_end, stack.x_fields
    if Q is None or train_end is None or x_fields is None:
        raise RolloutError("from_qtable 需要拟合好的模型，或季度面板、训练期末和字段列表")
    have = set(x_fields)
    sd = _sd_table(Q, train_end, sorted(have))
    try:
        shock_sd = policy.shock_sd(policy.fit(Q, train_end))
    except Exception:  # noqa: BLE001 - sign + size on a shock then raises a Chinese error
        shock_sd = {}
    try:
        return _read_qtable(qt, rows, H, have, sd, shock_sd, mode)
    except RolloutError:
        raise
    except (TypeError, ValueError, KeyError) as err:
        raise RolloutError(f"QTable 无法读取：{err}") from err


def _read_qtable(qt: dict, rows: list, H: int, have: set, sd: dict, shock_sd: dict, mode: str) -> dict:
    stmts, notes = [], []
    for r in rows:
        if not isinstance(r, dict):
            raise RolloutError("qtable.rows 的每一行必须是对象")
        q = int(r.get("q", 0))
        if not 1 <= q <= H:
            raise RolloutError(f"行的季度 q = {q} 超出 1..{H}")
        where = f"第 {q} 季"
        for f, e in (r.get("x") or {}).items():
            if f not in have:
                raise RolloutError(f"{where}：字段不存在或这个训练期没有估计：{f}")
            stmts.append(_entry(f, e, q, sd, shock_sd, where))
        for u, e in (r.get("u") or {}).items():
            if u not in F.SHOCKS:
                raise RolloutError(f"{where}：未知的政策冲击 {u}")
            stmts.append(_entry(u, e, q, sd, shock_sd, where))
        fc = r.get("fc") or {}
        stance = fc.get("stance", "未说明")
        if stance not in cq.FC_STANCES:
            raise RolloutError(f"{where}：金融条件 fc.stance 必须是 {cq.FC_STANCES} 之一")
        if stance == "未说明":
            notes.append({"q": q, "code": "W-FC", "text": "金融条件未说明：信用利差由模型补全。历史检验里，补全的信用利差与"
                                                          "无条件抽样相当，而资产结果主要取决于它"})
        else:
            for f in cq.FC_FIELDS:
                if f in (r.get("x") or {}) or f not in sd:
                    continue
                stmts.append(F.Statement(f, (q,), cq.FC_STANCE[stance] * sd[f], sd=cq.FC_STANCE_SD * sd[f],
                                         label=f"fc:{stance}"))
    cqx = qt.get("c_q") or {}
    for c in cqx.get("constraints") or []:
        kind, f = c.get("kind"), c.get("field")
        qs = tuple(int(v) for v in c.get("quarters") or ())
        if not qs or min(qs) < 1 or max(qs) > H:
            raise RolloutError(f"约束 {f}：quarters 必须在 1..{H} 之内")
        var = cq.QTABLE_LEVEL_VARS.get(f, f)
        if kind == "yoy":
            var = {"d_cpi_goods": "cpi_yoy", "d_cpi": "cpi_yoy", "d_core": "core_yoy"}.get(f, var)
        agg = {"sum": "sum", "mean": "mean"}.get(kind, "each")
        if kind == "range":
            stmts.append(F.Statement(var, qs, lo=c.get("min"), hi=c.get("max"), agg="each"))
        else:
            v = c.get("value")
            if v is None:
                stmts.append(F.Statement(var, qs, lo=c.get("min"), hi=c.get("max"), agg=agg))
            else:
                stmts.append(F.Statement(var, qs, float(v), sd=float(c["sd"]) if c.get("sd") else None, agg=agg))
    for a in cqx.get("announced") or []:
        qs = list(a.get("quarters") or ())
        vals = list(a.get("values") or ())
        if len(qs) != len(vals):
            raise RolloutError(f"预告路径 {a.get('field')}：quarters 与 values 的长度不同")
        for q, v in zip(qs, vals):
            stmts.append(F.Statement(a.get("field"), (int(q),), float(v), sd=float(a.get("sd") or 0) or None))
    odds = None
    for rg in cqx.get("regime") or []:
        if rg.get("label") in cq.CONTRACTION_LABELS:
            odds = {"contraction": float(rg.get("share", 1.0)), "quarters": [int(q) for q in rg.get("quarters") or []]}
        else:
            notes.append({"code": "regime", "text": f"体制请求 {rg.get('label')!r} 还没有接到分类器规则（P6），未作输入"})
    if cqx.get("anchors"):
        notes.append({"code": "anchors", "text": "情景的长期锚（c_q.anchors）要等 P6 接入：这次运行用 §3.8 的默认锚"})
    for a in cqx.get("announced") or []:
        if a.get("field") not in have:
            raise RolloutError(f"预告路径：字段不存在或这个训练期没有估计：{a.get('field')}")
    try:
        vst = val.parse_statements(list(cqx.get("valuation") or []), H, mode=mode)
    except ValueError as err:                       # the valuation block's own Chinese message
        raise RolloutError(str(err)) from err
    return {"statements": stmts, "valuation": vst, "cycle_odds": odds, "H": H, "notes": notes}
