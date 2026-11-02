"""The consequence synthesizer's engine side (AGENT_DESIGN §7; CONSEQUENCES §4–§6).

The LLM call and its structural guard live in scenariolab/agents/llm_io.py (AgentLLM.synth) and
scenariolab/llm/agent_guard.py (guard_synth: powers, written fields, price and event text, partition geometry on the
z grid, probabilities). This module does everything that needs the particles or the registry:

* the partition axes of §7.3 and the masses of the model's own pre-run distribution B_k on each axis (the prior of
  every partition, shown to the LLM as 标准分区的模型质量);
* the gate of §7.5 (G1 levers outside support or on thin/absent channels, G2 joint rarity, G3 regime, G4 a branched
  nature event, G5 the root) and the default partition (the stress / calm split, prior-only);
* the semantic guard of §7.4 on mechanisms (eligibility, lag discipline, precision weights and caps by channel
  class, joint rarity, the minimum shift, merging near-duplicates) and the probability calculus of §7.6:

      partition j:   q_j = c_s · clip(p_LLM,j / Σ p_LLM, m_j/3, 3 m_j) + (1 − c_s) · m_j    (renormalised)
      mechanism j:   π_j = min(c_m · p_LLM,j, 0.25);  Σ π ≤ 0.35 (no caps in stress mode)
      final:         P(mechanism j | k) = π_j;   P(partition j | k) = (1 − Σπ) · q_j / Σq

  With c_s = c_m = 0 the layer is the identity (the leaf mixture equals B_k);
* the realisation of a branch from B_k's particles (partition = regroup and resample; mechanism = B_k + Δ_used +
  noise, then the model re-completes the rest), and the attribution of §7.7.

Every number marked 判断值 is a judgment the forward ledger may recalibrate (§9.7).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

# §7.3: axis -> field candidates (first that exists), default region names low / centre / high (z units)
AXES: dict[str, dict] = {
    "growth": {"fields": ("d_unrate", "d_rgdp"), "names": ("好于中枢", "中枢", "走弱"), "kind": "e0"},
    "investment": {"fields": ("d_capex",), "alt": "d_itinv", "names": ("收缩", "中枢", "加速"), "kind": "e0"},
    "demand": {"fields": ("d_cons_serv", "d_cons_goods"), "names": ("预防性储蓄", "中枢", "动用储蓄"), "kind": "e0"},
    "inflation": {"fields": ("d_core", "d_cpi_goods"), "names": ("低于中枢", "中枢", "高于中枢"), "kind": "e0"},
    "expectations": {"fields": ("d_infexp1",), "names": ("低于中枢", "中枢", "高于中枢"), "kind": "e2"},
    "supply": {"fields": ("x_supply_chain",), "names": ("缓解", "中枢", "紧张"), "kind": "e0"},
    "sentiment": {"fields": ("d_sent",), "names": ("恶化", "中枢", "改善"), "kind": "e2"},
    "fin_conditions": {"fields": ("d_gz",), "names": ("宽松", "中枢", "收紧", "恐慌"), "kind": "fin",
                       "switch": "fin_conditions"},
    "funding": {"fields": ("d_cpbill",), "names": ("正常", "紧张", "严重"), "kind": "fin", "switch": "funding"},
    "equity_leg": {"fields": ("mult",), "names": ("估值收缩", "中枢", "估值扩张"), "kind": "equity",
                   "switch": "equity_leg"},
}
AXIS_OF_FIELD = {f: a for a, spec in AXES.items() for f in spec["fields"]}
DEFAULT_AXES = {"fin_conditions": True, "funding": True, "equity_leg": False}     # U-8, U-10
# §7.4 precision weights (判断值): channel class -> (w, cap in model sd); well/moderate in normal states: refused
CHANNEL_W: dict[str, tuple[float, float]] = {"well": (0.2, 1.0), "moderate": (0.5, 1.5), "regime": (0.5, 1.5),
                                             "thin": (0.8, 2.5), "absent": (1.0, 2.5)}
CLASS_RANK = {"well": 0, "moderate": 1, "regime": 2, "thin": 3, "absent": 4}
LAG_PROTECTED = ("d_unrate", "d_rgdp", "d_core", "d_cpi_serv", "d_wage", "d_capex", "d_itinv", "d_cons_serv")
EXPECTATION_FIELDS = ("d_sent", "d_epu", "d_infexp1", "d_infexp_lr")
# fields an E0-quantity or E2-survey mechanism may shift (synth ∈ {shift, both} in the overlay; never prices)
SHIFT_FIELDS = ("d_unrate", "d_rgdp", "d_ip", "d_cpi_goods", "d_cpi_serv", "d_core", "d_wage", "d_cons_goods",
                "d_cons_serv", "d_capex", "d_resinv", "d_exports", "d_profits", "d_lprod", "d_lshare",
                "x_supply_chain", "d_chargeoff", "d_itinv", "d_sent", "d_epu", "d_infexp1", "d_infexp_lr",
                "d_epu_health")
C_S, C_M = 0.4, 0.5                     # pool weights of partitions and mechanisms (判断值; U-3)
CAP_BRANCH, CAP_TOTAL = 0.25, 0.35      # mechanism mass caps (判断值)
MIN_BRANCH = 0.05                       # a partition below this after pooling joins its neighbour (§8.2)
MIN_SHIFT_SD = 0.25                     # a mechanism whose largest used shift is below this joins the centre
MERGE_DIST = 0.5                        # two mechanisms closer than this (standardised) merge
M_MAX = 3                               # consequence branches kept per cluster
# default partition (ENGINE §2.7 / §7.5): stress = US equity quarter < −15%, Δu ≥ +0.5pp or Baa spread ≥ +0.75pp
STRESS_BASE = 0.130                     # share of 1973–2026 quarters with any trigger (e01b)
STRESS_MULT, STRESS_MAX, STRESS_MIN_PATH = 2.0, 0.90, 0.02
STRESS_EQ_LOG = math.log(0.85)
# joint rarity (§7.4): empirical quantiles of the χ² of k standardised shifts (W_C01 C gives k = 3: p95 8.9, p99
# 21.4 against 7.81 / 11.34 for the χ² law); other k scale the χ² quantiles by the same ratios (判断值)
CHI2_P95 = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07, 6: 12.59}
CHI2_P99 = {1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28, 5: 15.09, 6: 16.81}
EMP_P95_RATIO, EMP_P99_RATIO = 8.9 / 7.81, 21.4 / 11.34
HAZARD_REGION = {"oil_disruption": ("d_metals_oil", "le", -2.0), "gpr_episode": ("d_gpr", "ge", 2.0),
                 "financial_accident": ("d_gz", "ge", 2.5)}


def chi2_limits(k: int) -> tuple[float, float]:
    k = max(1, min(int(k), 6))
    return CHI2_P95[k] * EMP_P95_RATIO, CHI2_P99[k] * EMP_P99_RATIO


# =====================================================================================================================
# Axes and masses
# =====================================================================================================================

def regions_for(axis: str, *, panic: bool = False) -> list[tuple]:
    """(lo_z, hi_z, name) regions of an axis on the half-integer grid; None = infinite."""
    names = AXES[axis]["names"]
    if axis == "fin_conditions":
        if panic:
            return [(None, -1.0, names[0]), (-1.0, 1.0, names[1]), (1.0, 2.5, names[2]), (2.5, None, names[3])]
        return [(None, -1.0, names[0]), (-1.0, 1.0, names[1]), (1.0, None, names[2])]
    if axis == "funding":
        return [(None, 1.0, names[0]), (1.0, 2.0, names[1]), (2.0, None, names[2])]
    return [(None, -1.0, names[0]), (-1.0, 1.0, names[1]), (1.0, None, names[2])]


def axis_list(available: Iterable[str], switches: Mapping | None = None, *, written: Iterable[str] = (),
              bigtech_wrote_itinv: bool = False) -> list[tuple[str, str]]:
    """[(axis, field)] usable at a cluster: the field exists, the axis is switched on and no actor wrote it."""
    sw = {**DEFAULT_AXES, **dict(switches or {})}
    have, wrote = set(available), set(written)
    out = []
    for axis, spec in AXES.items():
        if spec.get("switch") and not sw.get(spec["switch"], False):
            continue
        cands = list(spec["fields"])
        if axis == "investment" and not bigtech_wrote_itinv and spec.get("alt"):
            cands.append(spec["alt"])
        fid = next((f for f in cands if f in have and f not in wrote), None)
        if fid:
            out.append((axis, fid))
    return out


def standardise(v: np.ndarray) -> tuple[np.ndarray, float, float]:
    v = np.asarray(v, float)
    mu = float(np.mean(v))
    sd = float(np.std(v))
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return (v - mu) / sd, mu, sd


def in_region(z: np.ndarray, lo, hi) -> np.ndarray:
    ok = np.ones(len(z), bool)
    if lo is not None:
        ok &= z > lo
    if hi is not None:
        ok &= z <= hi
    return ok


def masses(values: np.ndarray, regions: list[tuple]) -> list[dict]:
    z, _, _ = standardise(values)
    n = max(len(z), 1)
    return [{"lo": lo, "hi": hi, "name": name, "mass": float(np.sum(in_region(z, lo, hi))) / n}
            for lo, hi, name in regions]


def cluster_masses(x: Mapping[str, np.ndarray], axes: list[tuple[str, str]], *, panic: bool = False) -> dict:
    """{field: [{lo, hi, name, mass}]} on every usable axis (the prompt's 标准分区的模型质量)."""
    return {fid: masses(x[fid], regions_for(axis, panic=panic)) for axis, fid in axes if fid in x}


def stress_flags(x: Mapping[str, np.ndarray], y: Mapping[str, np.ndarray]) -> np.ndarray:
    n = len(next(iter(y.values())))
    s = np.zeros(n, bool)
    if "us_equity" in y:
        s |= np.asarray(y["us_equity"]) < STRESS_EQ_LOG
    if "d_unrate" in x:
        s |= np.asarray(x["d_unrate"]) >= 0.5
    if "d_spread" in x:
        s |= np.asarray(x["d_spread"]) >= 0.75
    return s


def default_partition(x, y, path_mass: float) -> list[dict]:
    """ENGINE's stress / calm split, prior-only: split when the stress share is ≥ 2 × 13% and ≤ 90% and the stress
    child keeps ≥ 2% of path mass; otherwise one central child. Probabilities are the model's own shares."""
    s = stress_flags(x, y)
    share = float(np.mean(s)) if len(s) else 0.0
    if share >= STRESS_MULT * STRESS_BASE and share <= STRESS_MAX and path_mass * share >= STRESS_MIN_PATH:
        return [{"id": "calm", "kind": "default", "label": "平稳情形", "rule": "calm", "m": 1 - share, "p": 1 - share,
                 "memory": "", "p_llm": None},
                {"id": "stress", "kind": "default", "label": "压力情形", "rule": "stress", "m": share, "p": share,
                 "memory": "", "p_llm": None}]
    return [{"id": "central", "kind": "central", "label": "模型中枢", "rule": "all", "m": 1.0, "p": 1.0, "memory": "",
             "p_llm": None, "stress_share": share}]


# =====================================================================================================================
# Gates (§7.5)
# =====================================================================================================================

def gates(*, root: bool, support_flags: list, thin_levers: list, rarity: float | None, rarity_k: int,
          regime: list, facts: list, nature_branched: bool, requested: bool = False) -> list[str]:
    out = []
    if support_flags or thin_levers:
        out.append("G1")
    if rarity is not None and rarity > chi2_limits(rarity_k)[0]:
        out.append("G2")
    if regime or facts:
        out.append("G3")
    if nature_branched:
        out.append("G4")
    if root or requested:
        out.append("G5")
    return out


GATE_ZH = {"G1": "杠杆超出 1–99% 区间或走薄弱通道", "G2": "动作组合的联合罕见度高于经验 95%", "G3": "体制或持续事实",
           "G4": "自然事件分叉", "G5": "根节点或用户要求"}


def regime_lines(state_med: Mapping[str, float], facts: Mapping, stress_last: bool) -> list[str]:
    out = []
    ff, core = state_med.get("s_ff"), state_med.get("s_core_yoy")
    if ff is not None and ff < 0.5:
        out.append(f"政策利率接近零下限（{ff:.2f}%）")
    if core is not None and core > 4.0:
        out.append(f"核心通胀高于 4%（{core:.2f}%）")
    if stress_last:
        out.append("上季为压力季度")
    return out


def lever_rarity(written: Mapping[str, float], sds: Mapping[str, float]) -> tuple[float, int]:
    """χ² of the cluster's lever writes standardised by each field's quarterly sd, and the number of levers."""
    zs = [(v / sds.get(f, 1.0)) ** 2 for f, v in written.items() if sds.get(f)]
    return (float(sum(zs)), len(zs)) if zs else (0.0, 0)


# =====================================================================================================================
# The synthesizer request's tables
# =====================================================================================================================

def field_table(meta: Mapping[str, dict], axes: list[tuple[str, str]]) -> list[dict]:
    """The consequence-field table: id, zh, unit, partition (axis), mechanism (shift), p01 / p50 / p99."""
    axis_fields = {f for _, f in axes}
    rows = []
    for fid in sorted(set(SHIFT_FIELDS) | axis_fields):
        m = meta.get(fid) or {}
        h = m.get("hist") or {}
        if fid not in meta and fid != "mult":
            continue
        rows.append({"id": fid, "zh": m.get("zh") or ("估值倍数变化（美股 Δlog CAPE 倍数）" if fid == "mult" else fid),
                     "unit": m.get("unit") or "", "partition": fid in axis_fields, "mechanism": fid in SHIFT_FIELDS,
                     "p01": h.get("p01"), "p50": h.get("p50"), "p99": h.get("p99")})
    return rows


def channel_table() -> list[dict]:
    return [{"channel": k, "cls": k, "w": w, "cap": cap} for k, (w, cap) in CHANNEL_W.items()]


SIGNATURES = [{"key": "x_supply_chain↑", "need": "商品价格不下跌，且产出或资本开支不上升"},
              {"key": "d_lf↓（移民收紧）", "need": "失业率变化在 0.25 个标准差以内"},
              {"key": "石油供给", "need": "只经事件合成量（d_metals_oil↓、d_gpr↑、d_igrea 不上升）"}]
DIRECTIONS = [  # §7.4 direction table (CONSEQUENCES §4.3.1), as rule-table ids the LLM may cite
    {"id": "A-TAR-1", "text": "关税 → 商品 CPI +、EPU +（每 pp 约 +6.6 log%）、信心 −", "now": "", "q4": "",
     "source": "Amiti–Redding–Weinstein 2019；本地估计"},
    {"id": "A-MON-1", "text": "加息 → 信用条件收紧（只作分区）；产出与就业 2–4 季后才响应", "now": "", "q4": "",
     "source": "Romer–Romer 2004；Bauer–Swanson 2023"},
    {"id": "A-FIS-1", "text": "财政扩张 → GDP、消费 +", "now": "", "q4": "", "source": "Parker et al. 2013；Ramey 2019"},
    {"id": "A-OIL-1", "text": "石油供给中断 → 商品 CPI +、通胀预期 +、信心 −", "now": "", "q4": "",
     "source": "Baumeister–Hamilton 2019"},
    {"id": "A-EXC-1", "text": "出口管制 → 供应链压力 +", "now": "", "q4": "", "source": "CONSEQUENCES §4.3.1"},
    {"id": "A-CAP-1", "text": "超大型科技公司削减资本开支 → IT 投资 −、信用条件 +", "now": "", "q4": "",
     "source": "W_C02"},
    {"id": "A-IND-1", "text": "施压美联储 / 独立性受损 → 通胀预期 +", "now": "", "q4": "", "source": "Drechsel 2024"},
    {"id": "A-IMM-1", "text": "移民收紧 → 就业 −，失业率基本不变", "now": "", "q4": "", "source": "CONSEQUENCES §4.3.1"},
]


# =====================================================================================================================
# Mechanisms and the probability calculus (§7.4, §7.6, §8.2)
# =====================================================================================================================

def cause_class(classes: Iterable[str]) -> str | None:
    """The cluster's strongest eligible cause class (absent > thin > regime > moderate > well)."""
    cl = [c for c in classes if c in CLASS_RANK]
    return max(cl, key=lambda c: CLASS_RANK[c]) if cl else None


def consequences(answer, x: Mapping[str, np.ndarray], *, written: Iterable[str], causes: Iterable[str],
                 classes: Iterable[str], regime: bool, axes: list[tuple[str, str]], c_s: float = C_S,
                 c_m: float = C_M, stress_mode: bool = False, panic: bool = False, m_max: int = M_MAX) -> tuple[list, list]:
    """Guarded ClusterAnswer (status ok) + the cluster's B_k draws -> (branch specs with p | cluster, report lines).

    Branch spec: {id, kind central|partition|mechanism, label, axis, region {lo_z, hi_z}, m, p_llm, q, p, shifts
    {field: {mean_raw, sd_raw, mean, sd, w, cap, cls}}, memory, mechanism, cause, facts, anchors, unintended,
    beyond_history}."""
    rep: list[str] = []
    written = set(written)
    axis = getattr(answer, "axis", None)
    raw = [dict(b) for b in (getattr(answer, "branches", None) or [])]
    usable_axes = {f for _, f in axes}
    if axis is not None and (axis not in usable_axes or axis not in x):
        rep.append(f"分区轴 {axis} 在本动作组不可用，分区改为模型中枢")
        axis = None
    regions = [b for b in raw if b.get("kind") in ("central", "partition")]
    mechs = [b for b in raw if b.get("kind") == "mechanism"]
    # ---- partitions: the model's own masses on the LLM's regions
    z = standardise(x[axis])[0] if axis is not None else None
    parts = []
    for b in regions:
        r = b.get("region") or {}
        lo, hi = r.get("lo_z"), r.get("hi_z")
        if z is None:
            lo = hi = None
        m = float(np.mean(in_region(z, lo, hi))) if z is not None else 1.0
        parts.append({**b, "axis": axis, "region": {"lo_z": lo, "hi_z": hi}, "m": m, "p_llm": float(b.get("p_llm") or 0)})
    if z is None and parts:
        central = next((b for b in parts if b["kind"] == "central"), parts[0])
        central.update(kind="central", m=1.0, p_llm=sum(b["p_llm"] for b in parts), region={"lo_z": None, "hi_z": None})
        parts = [central]
    parts.sort(key=lambda b: -math.inf if b["region"]["lo_z"] is None else b["region"]["lo_z"])
    # ---- mechanisms: eligibility, lag discipline, weights and caps, rarity, minimum shift
    cls = cause_class(classes)
    if regime and (cls is None or CLASS_RANK.get(cls, 0) < CLASS_RANK["regime"]):
        cls = "regime"
    eligible = cls in ("thin", "absent", "regime") or stress_mode
    monetary_only = bool(causes) and all(str(c).startswith("fed.") for c in causes)
    kept_m = []
    for b in mechs:
        tag = b.get("id", "m")
        if not eligible:
            rep.append(f"{tag}：机制不合格（原因通道已由模型表达：well/moderate，且非体制），并入模型中枢")
            _to_central(parts, b)
            continue
        w, cap = CHANNEL_W.get(cls or "thin", CHANNEL_W["thin"])
        shifts = {}
        for fid, s in (b.get("shifts") or {}).items():
            if fid in written or fid not in x:
                continue
            if monetary_only and not regime and fid in LAG_PROTECTED:
                rep.append(f"{tag}：只由货币政策引起的机制不能在当季写滞后保护字段 {fid}，已删除")
                continue
            sd_f = float(np.std(x[fid])) or 1.0
            mean_raw, sd_raw = float(s.get("mean", 0.0)), abs(float(s.get("sd", 0.0) or 0.0))
            mean = w * mean_raw
            if not stress_mode:
                mean = float(np.clip(mean, -cap * sd_f, cap * sd_f))
            shifts[fid] = {"mean_raw": mean_raw, "sd_raw": sd_raw, "mean": mean, "sd": w * sd_raw, "w": w, "cap": cap,
                           "cls": cls, "sd_model": sd_f}
        if not shifts:
            rep.append(f"{tag}：机制没有可用的增量，并入模型中枢")
            _to_central(parts, b)
            continue
        chi2 = sum((v["mean"] / v["sd_model"]) ** 2 for v in shifts.values())
        p95, p99 = chi2_limits(len(shifts))
        if chi2 > p99 and not b.get("beyond_history"):
            scale = math.sqrt(p99 / chi2)
            for v in shifts.values():
                v["mean"] *= scale
            rep.append(f"{tag}：增量的联合罕见度 {chi2:.1f} 高于经验 99%（{p99:.1f}），按比例缩回")
        elif chi2 > p95:
            rep.append(f"{tag}：增量的联合罕见度 {chi2:.1f} 高于经验 95%（需理由）")
        biggest = max(abs(v["mean"]) / v["sd_model"] for v in shifts.values())
        if biggest < MIN_SHIFT_SD:
            rep.append(f"{tag}：折扣后的最大增量 {biggest:.2f} 个模型标准差，低于 {MIN_SHIFT_SD}，并入模型中枢")
            _to_central(parts, b)
            continue
        kept_m.append({**b, "kind": "mechanism", "axis": None, "region": None, "shifts": shifts,
                       "p_llm": float(b.get("p_llm") or 0), "m": 0.0})
    kept_m = _merge_close(kept_m, rep)
    kept_m.sort(key=lambda b: -b["p_llm"])
    for b in kept_m[2:]:
        rep.append(f"{b.get('id')}：每组最多两个机制分支，并入模型中枢")
        _to_central(parts, b)
    kept_m = kept_m[:2]
    # ---- the calculus
    tot_llm = sum(b["p_llm"] for b in parts)
    for b in parts:
        share = b["p_llm"] / tot_llm if tot_llm > 0 else b["m"]
        lo_b, hi_b = b["m"] / 3.0, 3.0 * b["m"]
        b["q"] = c_s * float(np.clip(share, lo_b, hi_b)) + (1.0 - c_s) * b["m"]
    pis = [min(c_m * b["p_llm"], 1.0 if stress_mode else CAP_BRANCH) for b in kept_m]
    if not stress_mode and sum(pis) > CAP_TOTAL:
        scale = CAP_TOTAL / sum(pis)
        pis = [p * scale for p in pis]
        rep.append(f"机制分支的总质量超过 {CAP_TOTAL}，按比例缩到上限")
    sum_pi = sum(pis)
    sq = sum(b["q"] for b in parts) or 1.0
    for b in parts:
        b["p"] = (1.0 - sum_pi) * b["q"] / sq
    for b, p in zip(kept_m, pis):
        b["p"] = p
    # ---- merges: partitions below 0.05 join their neighbour; at most m_max branches in all
    parts = [b for b in parts if b["m"] > 0 or b["kind"] == "central"]
    parts = _merge_small(parts, rep, MIN_BRANCH)
    while len(parts) + len(kept_m) > m_max and len(parts) > 1:
        smallest = min((b for b in parts if b["kind"] != "central"), key=lambda b: b["p"], default=None)
        if smallest is None:
            break
        _merge_into_neighbour(parts, smallest, rep, f"每组最多保留 {m_max} 个分支")
    branches = parts + [b for b in kept_m if b["p"] > 0]
    total = sum(b["p"] for b in branches)
    for b in branches:
        b["p"] = b["p"] / total if total > 0 else 1.0 / len(branches)
    return branches, rep


def _to_central(parts: list, b: dict) -> None:
    central = next((p for p in parts if p["kind"] == "central"), None)
    if central is not None:
        central["p_llm"] = central.get("p_llm", 0.0) + float(b.get("p_llm") or 0.0)


def _merge_close(mechs: list, rep: list) -> list:
    out: list = []
    for b in sorted(mechs, key=lambda b: -b["p_llm"]):
        hit = None
        for o in out:
            keys = set(b["shifts"]) | set(o["shifts"])
            d = math.sqrt(sum(((b["shifts"].get(k, {}).get("mean", 0.0) - o["shifts"].get(k, {}).get("mean", 0.0))
                               / ((b["shifts"].get(k) or o["shifts"].get(k))["sd_model"])) ** 2 for k in keys))
            if d < MERGE_DIST:
                hit = o
                break
        if hit is None:
            out.append(b)
            continue
        w1, w2 = hit["p_llm"], b["p_llm"]
        for k in set(b["shifts"]) | set(hit["shifts"]):
            a, c = hit["shifts"].get(k), b["shifts"].get(k)
            if a and c:
                a["mean"] = (w1 * a["mean"] + w2 * c["mean"]) / max(w1 + w2, 1e-12)
            elif c:
                hit["shifts"][k] = dict(c)
        hit["p_llm"] = w1 + w2
        rep.append(f"{b.get('id')}：与 {hit.get('id')} 的标准化距离小于 {MERGE_DIST}，两支机制合并")
    return out


def _merge_into_neighbour(parts: list, victim: dict, rep: list, why: str) -> None:
    i = parts.index(victim)
    j = i - 1 if i > 0 else i + 1
    if not 0 <= j < len(parts):
        return
    nb = parts[j]
    lo = victim["region"]["lo_z"] if j > i else nb["region"]["lo_z"]
    hi = nb["region"]["hi_z"] if j > i else victim["region"]["hi_z"]
    if j < i:
        lo, hi = nb["region"]["lo_z"], victim["region"]["hi_z"]
    else:
        lo, hi = victim["region"]["lo_z"], nb["region"]["hi_z"]
    nb["region"] = {"lo_z": lo, "hi_z": hi}
    for k in ("m", "q", "p", "p_llm"):
        nb[k] = nb.get(k, 0.0) + victim.get(k, 0.0)
    parts.remove(victim)
    rep.append(f"{victim.get('id')}：{why}，并入相邻分区 {nb.get('id')}")


def _merge_small(parts: list, rep: list, floor: float) -> list:
    changed = True
    while changed and len(parts) > 1:
        changed = False
        for b in sorted(parts, key=lambda b: b["p"]):
            if b["p"] < floor and b["kind"] != "central":
                _merge_into_neighbour(parts, b, rep, f"池化后概率 {b['p']:.3f} 低于 {floor}")
                changed = True
                break
    return parts


# =====================================================================================================================
# Realisation and attribution
# =====================================================================================================================

def branch_rows(branch: dict, x: Mapping[str, np.ndarray], y: Mapping[str, np.ndarray]) -> np.ndarray:
    """Indices of B_k's particles a partition / default / central branch keeps (mechanisms keep them all)."""
    n = len(next(iter(y.values())))
    kind = branch.get("kind")
    if kind == "default":
        s = stress_flags(x, y)
        return np.flatnonzero(s if branch.get("rule") == "stress" else ~s)
    if kind in ("central", "partition") and branch.get("axis") and branch.get("region"):
        src = y if branch["axis"] == "mult" else x
        if branch["axis"] not in src:
            return np.arange(n)
        z = standardise(src[branch["axis"]])[0]
        r = branch["region"]
        return np.flatnonzero(in_region(z, r.get("lo_z"), r.get("hi_z")))
    return np.arange(n)


def mechanism_pins(branch: dict, x: Mapping[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """x_f = B_k,f + Δ_used + N(0, σ_used²) for every shifted field (§6.4)."""
    out = {}
    for fid, s in sorted((branch.get("shifts") or {}).items()):
        base = np.asarray(x[fid], float)
        out[fid] = base + float(s["mean"]) + float(s.get("sd") or 0.0) * rng.standard_normal(len(base))
    return out


def attribution(a0_y: Mapping, bk_y: Mapping, child_y: Mapping, kind: str, axis: str | None,
                assets: Iterable[str]) -> dict:
    """Per asset, the median differences of §7.7 for one child (from particles only)."""
    out = {}
    fin = axis in ("d_gz", "d_cpbill")
    for a in assets:
        if a not in child_y or a not in bk_y or a not in a0_y:
            continue
        m0, mk, mc = (float(np.median(v[a])) for v in (a0_y, bk_y, child_y))
        row = {"actions": mk - m0, "partition": 0.0, "fin_conditions": 0.0, "equity_leg": 0.0, "mechanism": 0.0}
        if kind == "mechanism":
            row["mechanism"] = mc - mk
        elif axis == "mult":
            row["equity_leg"] = mc - mk
        elif fin:
            row["fin_conditions"] = mc - mk
        else:
            row["partition"] = mc - mk
        out[a] = {k: round(v, 5) for k, v in row.items()}
    return out


def hazard_masses(x: Mapping[str, np.ndarray]) -> dict[str, dict]:
    """Nature's tail masses in A0 (§6.5): the share of particles in each hazard's region, in A0's own sd units."""
    out = {}
    for h, (fid, op, zc) in HAZARD_REGION.items():
        if fid not in x:
            continue
        z = standardise(x[fid])[0]
        mask = z <= zc if op == "le" else z >= zc
        out[h] = {"field": fid, "op": op, "z": zc, "m": float(np.mean(mask))}
    return out


def hazard_mask(x: Mapping[str, np.ndarray], hazard: str) -> np.ndarray:
    fid, op, zc = HAZARD_REGION[hazard]
    z = standardise(x[fid])[0]
    return z <= zc if op == "le" else z >= zc
