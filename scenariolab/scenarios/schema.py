"""Scenario spec schema and validation.

The same schema is used by the hand-written built-in scenarios (YAML) and by
the LLM translator. validate_spec() is the guard rail between a narrative and
the engine: it rejects unknown variables, bounds every number to what history
can support, pads or trims phases to the horizon, and reports every change so
the UI can show exactly what was adjusted. A phase's `regime` key (macro
regimes, regime_views.py) becomes plain views on the ex-post rg_ columns.
A phase's overrides may also set `term_premium` (the 10-year's reversion target
over r* + inflation, SIM_DESIGN P0.6) next to `r_star`; the LLM's output schema
(SPEC_JSON_SCHEMA) does not offer it, YAML and edited specs may.

Views take the s_ / f_ / e_ / rg_ column families; the audit-only x_ columns and
the audit-only f_ columns (those of audit-role factors, config.FACTORS "role":
"audit", and the walk-forward's report-only stage-B regressors, which are no
config factor at all) are refused.
The checks of the missing-factor round (FACTORS_DESIGN §3.4) follow the regime
layer's W1-W11: W12 a view on a variable first observed well after the month
database starts, W13 a labour-share drift while the labour share is not in the
transmission, W14 a view on an audit-only variable.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np
import pandas as pd

from .. import config
from ..data.panel import STAGE_B_AUDIT
from ..engine.factor_model import ModelSpec, asset_extras
from ..engine.pooling import BLOCK, start_prior, start_values, view_support
from ..engine.sampler import phase_block
from ..engine.simulate import DEFAULT_REVERSION
from . import regime_views

ALLOWED_PREFIXES = ("s_", "f_", "e_", "rg_")    # x_ columns (audit-only, stored in the panel) are refused
# W14: the factor audit's comparison columns (audit-role factors) and the walk-forward's report-only stage-B
# regressors (role "audit" in panel_meta's column_info)
AUDIT_VARS = {f"f_{fid}" for fid in config.AUDIT_FACTOR_IDS} | {f"f_{c}" for c in STAGE_B_AUDIT}
LATE_MONTHS = 12                  # W12: a view variable first observed more than this after the database starts
REALTIME_PREFIX = "rg_rt_"        # real-time regime labels: nowcast and forecast test only, never a view
# States defined as a difference of two others: a phase that sets all three
# says the same thing twice (and EP cannot meet a contradictory triple).
IDENTITIES = {"s_real_ff": ("s_ff", "s_cpi_yoy"), "s_term_spread": ("s_y10", "s_ff")}
MEAN_KEYS = ("mean", "mean_annual", "shift_sd")
TARGET_KEYS = ("mean", "mean_annual", "shift_sd", "min", "max", "min_annual", "max_annual", "vol_mult")
MAX_PHASES = 8
MAX_VIEWS_PER_PHASE = 12
ASSET_DRIFT_BOUND = 0.08          # +-8%/yr of extra price drift
RESID_SCALE_BOUNDS = (0.5, 2.0)
R_STAR_BOUNDS = (-3.0, 3.0)
# a phase's term premium (pp): the 10-year's reversion target over r* + inflation (simulate overrides, P0.6), so a
# phase that states its policy path through r_star can keep the 10-year target; the reversion's own bounds
TERM_PREMIUM_BOUNDS = (-1.0, 3.0)
CONFIDENCE_BOUNDS = (0.3, 1.0)
SHIFT_SD_BOUNDS = (-3.0, 3.0)
VOL_MULT_BOUNDS = (0.3, 3.0)
REVERSION_BOUNDS = {
    "kappa_ff": (0.0, 0.1), "kappa_y10": (0.0, 0.1), "kappa_spread": (0.0, 0.1),
    "kappa_unrate": (0.0, 0.1), "kappa_oil": (0.0, 0.1), "r_star": R_STAR_BOUNDS, "term_premium": TERM_PREMIUM_BOUNDS,
    "spread_level": (0.5, 6.0), "unrate_level": (3.0, 10.0),
    # log shift of the real-oil target (1973+ median, about $68 in 2026 dollars): -0.7 ~ $34, its 5th
    # percentile since 1973; +0.4 ~ $101, its 77th
    "oil_target_shift": (-0.7, 0.4),
}
SIGNS = {"+", "-", "±"}
ENDOGENOUS_STATE_IDS = {st["id"] for st in config.STATES if st.get("endogenous")}

# JSON schema handed to the LLM (structured output). Numeric bounds are not
# expressed here on purpose (structured outputs do not support minimum /
# maximum): validate_spec enforces them and reports clipping.
NULLABLE_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}
VIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "var": {"type": "string"},
        "mean": NULLABLE_NUMBER,
        "mean_annual": NULLABLE_NUMBER,
        "shift_sd": NULLABLE_NUMBER,
        "min": NULLABLE_NUMBER,
        "max": NULLABLE_NUMBER,
        "vol_mult": NULLABLE_NUMBER,
        "why": {"type": "string"},
    },
    "required": ["var", "mean", "mean_annual", "shift_sd", "min", "max", "vol_mult", "why"],
    "additionalProperties": False,
}
REGIME_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "enum": list(regime_views.IDS)},
        "share": {"type": "number"},
        "why": {"type": "string"},
    },
    "required": ["id", "share", "why"],
    "additionalProperties": False,
}
SPEC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "summary": {"type": "string"},
        "narrative": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "prior_prob": {"type": "number"},
        "analogs": {"type": "array", "items": {"type": "string"}},
        "causal_chain": {"type": "array", "items": {
            "type": "object",
            "properties": {"from": {"type": "string"}, "to": {"type": "string"},
                           "sign": {"type": "string", "enum": ["+", "-", "±"]},
                           "lag": {"type": "string"}, "mechanism": {"type": "string"}},
            "required": ["from", "to", "sign", "lag", "mechanism"],
            "additionalProperties": False}},
        "phases": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "months": {"type": "integer"},
                "regime": {"type": "array", "items": REGIME_SCHEMA},
                "views": {"type": "array", "items": VIEW_SCHEMA},
                "overrides": {"type": "object", "properties": {
                    "factor_drift": {"type": "array", "items": {"type": "object", "properties": {
                        "factor": {"type": "string"}, "annual": {"type": "number"}, "why": {"type": "string"}},
                        "required": ["factor", "annual", "why"], "additionalProperties": False}},
                    "asset_drift": {"type": "array", "items": {"type": "object", "properties": {
                        "asset": {"type": "string"}, "annual": {"type": "number"}, "why": {"type": "string"}},
                        "required": ["asset", "annual", "why"], "additionalProperties": False}},
                    "resid_scale": NULLABLE_NUMBER,
                    "r_star": NULLABLE_NUMBER},
                    "required": ["factor_drift", "asset_drift", "resid_scale", "r_star"],
                    "additionalProperties": False},
                "confidence": {"type": "number"},
            },
            "required": ["name", "months", "regime", "views", "overrides", "confidence"],
            "additionalProperties": False}},
    },
    "required": ["id", "name", "summary", "narrative", "tags", "prior_prob", "analogs",
                 "causal_chain", "phases"],
    "additionalProperties": False,
}


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s[:40] or "scenario"


def _num(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _normalize_overrides(ov) -> dict:
    """Accept both the YAML dict form and the LLM list form of overrides."""
    if not ov:
        return {}
    ov = dict(ov)
    for key, item_key in (("factor_drift", "factor"), ("asset_drift", "asset")):
        val = ov.get(key)
        if isinstance(val, list):
            ov[key] = {d.get(item_key): d.get("annual") for d in val if isinstance(d, dict) and d.get(item_key)}
    return {k: v for k, v in ov.items() if v is not None and v != {} and v != []}


def audit_only(var: str, columns=()) -> bool:
    """Whether a view variable is an audit-only f_ column (W14): AUDIT_VARS, or any other f_ column of the frame
    (`columns`) that is no config factor (CPI's own change f_cpi aside), so a comparison column added to the
    panel later is refused too."""
    return var in AUDIT_VARS or (var in columns and var.startswith("f_") and var[2:] not in config.FACTOR_IDS
                                 and var != "f_cpi")


def transmission_factors(spec: ModelSpec) -> list[str]:
    """Factor ids that enter some asset equation of `spec`: the common factors, then each asset's extras."""
    out = [f for f in spec.factors if f not in config.AUDIT_FACTOR_IDS]
    for ids in asset_extras(spec).values():
        out += [f for f in ids if f not in out and f not in config.AUDIT_FACTOR_IDS]
    return out


def drift_factor_ids(spec: ModelSpec | None = None) -> list[str]:
    """Factors a factor_drift can move.

    With the active model's spec: its transmission (common factors and the assets' extras), CPI, the
    labour share (a tracked macro path the income-distribution calculator drives, whether or not it is
    in the transmission; W13) and the profit share when the valuation state is on (it moves the fair
    value). Without one: the core factors and CPI (plus the profit share), the base model's list. The
    other factors only serve as views that pick historical months; a drift on them would change no path.
    """
    ids = transmission_factors(spec) if spec is not None else list(config.CORE_FACTOR_IDS)
    extra = ["cpi"] + (["labor_share"] if spec is not None else [])
    if config.VALUATION.get("enabled"):
        extra.append("profit_share")
    return ids + [f for f in extra if f not in ids]


def factor_drift_bounds(frame: pd.DataFrame, driftable: list[str] | None = None) -> dict[str, float]:
    """+-2x the annualized historical volatility of each driftable factor's monthly change
    (`driftable` defaults to drift_factor_ids())."""
    out = {}
    for fid in drift_factor_ids() if driftable is None else driftable:
        col = f"f_{fid}"
        if col in frame:
            sd = float(frame[col].std()) * math.sqrt(12.0)
            out[fid] = 2.0 * sd if sd > 0 else 0.0
    return out


def late_start(frame: pd.DataFrame, var: str) -> tuple[str, float] | None:
    """(first month "YYYY-MM", share of the database's months where it is observed) of a variable first
    observed more than LATE_MONTHS months after the month database starts (W12), else None."""
    first = frame[var].first_valid_index()
    if first is None:
        return None
    start = frame.index[0]
    if (first.year - start.year) * 12 + first.month - start.month <= LATE_MONTHS:
        return None
    return first.strftime("%Y-%m"), float(frame[var].notna().mean())


def late_view_text(k: int, var: str, first: str, share: float, since: str) -> str:
    """W12."""
    return f"阶段 {k}：{var} 从 {first} 才有数据，这一阶段只会抽 {first} 以后的历史月份（占 {since} 年以来月份的 {share:.0%}）"


def labor_share_drift_text(k: int) -> str:
    """W13."""
    return (f"阶段 {k}：劳动收入占比不在今天的传导模型里（按发布月对齐后它不提高样本外 R²），它的漂移只改变模拟的劳动收入"
            f"占比路径，不直接改变资产回报")


def validate_spec(spec: dict, frame: pd.DataFrame, horizon: int = config.HORIZON_MONTHS,
                  block: int = BLOCK, driftable: list[str] | None = None,
                  model_spec: ModelSpec | None = None) -> tuple[dict, dict]:
    """Return (clean_spec, report). `frame` is the MonthDB frame (view support).

    A view target is bounded by what the phase can average: the 2nd-98th
    percentile of the means of the `block`-month runs the sampler draws
    (pooling.view_support), which for monthly flows is much narrower than
    the range of single months.

    `model_spec` is the active transmission (runner.run_spec and the translator pass ctx.model.spec): it
    sets the factors a factor_drift may move (drift_factor_ids(model_spec), unless `driftable` lists
    them) and warns when a labour-share drift no longer moves asset returns (W13). Without either, the
    base model's list applies.
    """
    s = copy.deepcopy(spec or {})
    errors: list[str] = []
    warnings: list[str] = []
    clipped: list[dict] = []

    s["name"] = str(s.get("name") or "未命名情景")[:60]
    s["id"] = slugify(s.get("id") or s["name"])
    if not re.fullmatch(r"[a-z0-9_]+", s["id"]):
        s["id"] = slugify(s["id"])
    s["summary"] = str(s.get("summary") or "")[:400]
    s["narrative"] = str(s.get("narrative") or "")[:6000]
    s["tags"] = [str(t)[:24] for t in (s.get("tags") or [])][:8]
    pp = _num(s.get("prior_prob"))
    s["prior_prob"] = _clip(pp, 0.0, 1.0) if pp is not None else 0.05

    analogs = []
    for a in s.get("analogs") or []:
        m = re.fullmatch(r"\s*(\d{4}-\d{2})\s*:\s*(\d{4}-\d{2})\s*", str(a))
        if m and m.group(1) <= m.group(2):
            analogs.append(f"{m.group(1)}:{m.group(2)}")
        else:
            warnings.append(f"忽略无法解析的历史类比区间：{a}")
    s["analogs"] = analogs[:8]

    chain = []
    for e in s.get("causal_chain") or []:
        if not isinstance(e, dict) or not e.get("from") or not e.get("to"):
            continue
        sign = str(e.get("sign", "+"))
        chain.append({"from": str(e["from"])[:40], "to": str(e["to"])[:40],
                      "sign": sign if sign in SIGNS else "+", "lag": str(e.get("lag", ""))[:20],
                      "mechanism": str(e.get("mechanism", ""))[:300]})
    s["causal_chain"] = chain[:30]

    if driftable is None:
        driftable = drift_factor_ids(model_spec)
    drift_bounds = factor_drift_bounds(frame, driftable)
    labor_outside = model_spec is not None and "labor_share" not in transmission_factors(model_spec)
    columns = set(frame.columns)
    since = f"{frame.index[0]:%Y}" if len(frame) else ""
    phases_in = s.get("phases") or []
    if not isinstance(phases_in, list):
        errors.append("phases 必须是列表")
        phases_in = []
    # phase lengths first (trimmed to the horizon), so each view is bounded by its final phase length
    entries = []
    for k, ph in enumerate(phases_in[:MAX_PHASES]):
        if not isinstance(ph, dict):
            errors.append(f"阶段 {k + 1} 不是对象")
            continue
        months = int(_num(ph.get("months")) or 0)
        if months < 1:
            warnings.append(f"阶段 {k + 1} 月数无效，已设为 12")
            months = 12
        entries.append([k, ph, months])
    total = sum(e[2] for e in entries)
    if total > horizon:
        excess = total - horizon
        for e in reversed(entries):
            cut = min(excess, e[2] - 1)
            e[2] -= cut
            excess -= cut
            if excess == 0:
                break
        warnings.append(f"阶段总月数 {total} 超过 {horizon}，已从后往前截短")
    phases = []
    for k, ph, months in entries:
        # the regime key: plain views on the ex-post rg_ columns, placed first (not counted in the view cap)
        reg, reg_warnings, reg_clipped = regime_views.normalize_regime(ph.get("regime"), k + 1)
        warnings += reg_warnings
        clipped += reg_clipped
        raw_views = regime_views.expand_regime(reg, ph.get("views")) or []
        views = []
        for v in raw_views[:MAX_VIEWS_PER_PHASE + len(reg)]:
            if not isinstance(v, dict):
                continue
            var = str(v.get("var", ""))
            if var.startswith(REALTIME_PREFIX):
                warnings.append(regime_views.realtime_view_text(k + 1, var))
                continue
            if audit_only(var, columns):
                warnings.append(f"阶段 {k + 1}：{var} 只用于审计对照，不能作为情景视图，已忽略")       # W14
                continue
            if not var.startswith(ALLOWED_PREFIXES) or var not in columns:
                warnings.append(f"阶段 {k + 1}：变量 {var} 不在引擎支持的列表中，已忽略")
                continue
            if not np.isfinite(frame[var].to_numpy(dtype=float)).any():
                warnings.append(f"阶段 {k + 1}：变量 {var} 没有历史数据，已忽略")
                continue
            lo, hi = view_support(frame, var, months, block)
            clean_v = {"var": var}
            for key in TARGET_KEYS:
                val = _num(v.get(key))
                if val is None:
                    continue
                if key in ("shift_sd",):
                    used = _clip(val, *SHIFT_SD_BOUNDS)
                elif key == "vol_mult":
                    used = _clip(val, *VOL_MULT_BOUNDS)
                elif key.endswith("_annual"):
                    used = _clip(val, lo * 12.0, hi * 12.0)
                else:
                    used = _clip(val, lo, hi)
                if abs(used - val) > 1e-9:
                    clipped.append({"phase": k + 1, "var": var, "key": key, "requested": val, "used": used})
                    warnings.append(f"阶段 {k + 1}：{var} 的 {key}={val:.4g} 超出历史可支撑范围，截断为 {used:.4g}")
                clean_v[key] = used
            if len(clean_v) == 1:
                warnings.append(f"阶段 {k + 1}：变量 {var} 没有给出目标值，已忽略")
                continue
            if v.get("why"):
                clean_v["why"] = str(v["why"])[:200]
            if (var.startswith("f_") and var[2:] in config.ENDOGENOUS_FACTOR_IDS) or (
                    var.startswith("s_") and var[2:] in ENDOGENOUS_STATE_IDS):
                warnings.append(f"阶段 {k + 1}：{var} 部分由资产价格本身构成（内生），对它设视图相当于直接规定股市涨跌，"
                                f"请确认这是叙事本意")
            late = late_start(frame, var)
            if late:
                warnings.append(late_view_text(k + 1, var, *late, since))                         # W12
            views.append(clean_v)
        by_var = {v["var"]: v for v in views}
        for derived, (a, b) in IDENTITIES.items():
            if all(any(key in by_var.get(c, {}) for key in MEAN_KEYS) for c in (derived, a, b)):
                views = [v for v in views if v["var"] != derived]
                warnings.append(f"阶段 {k + 1}：{derived} = {a} − {b}，三者同时设定是重复的（矛盾时无法同时满足），"
                                f"已忽略 {derived}")

        ov = _normalize_overrides(ph.get("overrides"))
        clean_ov: dict = {}
        fd = {}
        for fid, val in (ov.get("factor_drift") or {}).items():
            val = _num(val)
            if fid not in drift_bounds or val is None:
                if fid in config.AUDIT_FACTOR_IDS and val is not None:
                    warnings.append(f"阶段 {k + 1}：因子 {fid} 只用于审计对照，不能加漂移，已忽略")
                elif fid in config.FACTOR_IDS and val is not None:
                    warnings.append(f"阶段 {k + 1}：因子 {fid} 不在今天的传导模型里（只能作为视图来挑选历史月份），"
                                    f"给它加漂移不会改变任何资产或宏观路径，已忽略")
                else:
                    warnings.append(f"阶段 {k + 1}：未知因子漂移 {fid}，已忽略")
                continue
            b = drift_bounds[fid]
            used = _clip(val, -b, b)
            if abs(used - val) > 1e-12:
                clipped.append({"phase": k + 1, "var": f"factor_drift.{fid}", "requested": val, "used": used})
                warnings.append(f"阶段 {k + 1}：因子 {fid} 的额外漂移 {val:.4g}/年 截断为 {used:.4g}")
            fd[fid] = used
            if fid == "labor_share" and labor_outside:
                warnings.append(labor_share_drift_text(k + 1))                                    # W13
        if fd:
            clean_ov["factor_drift"] = fd
        ad = {}
        for aid, val in (ov.get("asset_drift") or {}).items():
            val = _num(val)
            if aid not in config.ASSET_IDS or aid in ("cash", "treasury_10y", "crude_oil") or val is None:
                warnings.append(f"阶段 {k + 1}：资产 {aid} 不能设额外漂移，已忽略")
                continue
            used = _clip(val, -ASSET_DRIFT_BOUND, ASSET_DRIFT_BOUND)
            if abs(used - val) > 1e-12:
                clipped.append({"phase": k + 1, "var": f"asset_drift.{aid}", "requested": val, "used": used})
                warnings.append(f"阶段 {k + 1}：资产 {aid} 的额外漂移 {val:.4g}/年 截断为 {used:.4g}")
            ad[aid] = used
        if ad:
            clean_ov["asset_drift"] = ad
        for key, bounds in (("resid_scale", RESID_SCALE_BOUNDS), ("r_star", R_STAR_BOUNDS),
                            ("term_premium", TERM_PREMIUM_BOUNDS)):
            val = _num(ov.get(key))
            if val is not None:
                used = _clip(val, *bounds)
                if abs(used - val) > 1e-12:
                    clipped.append({"phase": k + 1, "var": key, "requested": val, "used": used})
                clean_ov[key] = used
        conf = _num(ph.get("confidence"))
        conf = _clip(conf, *CONFIDENCE_BOUNDS) if conf is not None else 1.0
        clean_ph = {"name": str(ph.get("name") or f"阶段{k + 1}")[:30], "months": months}
        kept = {v["var"] for v in views}
        reg = {r: s for r, s in reg.items() if f"rg_{r}" in kept}
        if reg:                                       # what was asked, so the detail shows it
            clean_ph["regime"] = reg
        clean_ph.update({"views": views, "overrides": clean_ov, "confidence": conf})
        warnings += regime_views.static_checks(clean_ph, k + 1, frame,
                                               regime_views.phase_requests(clean_ph, k + 1, clipped))
        phases.append(clean_ph)
    if len(phases_in) > MAX_PHASES:
        warnings.append(f"阶段数超过 {MAX_PHASES}，多余阶段已忽略")

    total = sum(p["months"] for p in phases)
    if total < horizon:
        phases.append({"name": "回归常态", "months": horizon - total, "views": [], "overrides": {}, "confidence": 1.0})
        if phases[:-1]:
            warnings.append(f"阶段总月数 {total} 少于 {horizon}，已补一个无视图的回归常态阶段")
    s["phases"] = phases

    rev = {}
    rev_in = s.get("reversion") or {}
    for key, val in (rev_in.items() if isinstance(rev_in, dict) else []):
        val = _num(val)
        if key not in REVERSION_BOUNDS or val is None:
            warnings.append(f"均值回归参数 {key} 未知或不是数值，已忽略")
            continue
        used = _clip(val, *REVERSION_BOUNDS[key])
        if abs(used - val) > 1e-12:
            clipped.append({"phase": None, "var": f"reversion.{key}", "requested": val, "used": used})
            warnings.append(f"均值回归参数 {key}={val:.4g} 超出边界，截断为 {used:.4g}")
        rev[key] = used
    s["reversion"] = rev
    # Only the built-in library resolves a calibration block into views and drifts
    # (library.resolve_calibration, before validation); anywhere else it would be dropped silently.
    if s.get("calibration") not in (None, [], {}):
        warnings.append("calibration（收入分配计算器）只在内置情景库里换算成视图和因子漂移，这份规格里的 calibration 块已忽略；"
                        "要保留它的效果，请把换算出的视图和漂移直接写进对应阶段（从内置情景载入的模板已经包含它们）")
    for key in list(s.keys()):
        if key not in {"id", "name", "summary", "narrative", "tags", "prior_prob", "analogs",
                       "causal_chain", "phases", "reversion", "source"}:
            s.pop(key)
    report = {"ok": not errors, "errors": errors, "warnings": warnings, "clipped": clipped,
              "horizon_months": horizon, "default_reversion": DEFAULT_REVERSION}
    return s, report


def allowed_variables(frame: pd.DataFrame, block: int = BLOCK, prior: np.ndarray | None = None,
                      rare: np.ndarray | None = None) -> list[dict]:
    """Variables a spec may put views on, with the historical support (for the LLM prompt and UI).

    p2/median/p98/sd describe single months; block_p2/block_p98/block_sd the
    means of `block`-month runs, which is what a phase's views can reach
    (block_sd is weighted the way the engine draws blocks, so it is the unit
    of shift_sd for a phase of `block` months).
    """
    names = {f"s_{s['id']}": s["name"] for s in config.STATES}
    names.update({f"f_{f['id']}": f["name"] + "（月度）" for f in config.FACTORS
                  if f["id"] not in config.AUDIT_FACTOR_IDS})          # audit-only columns are never views
    names["f_cpi"] = "CPI 月度变化（log）"
    shock_names = {"e_oil_supply": "石油供给冲击（标准化）", "e_demand": "总需求冲击（标准化）",
                   "e_monetary": "货币政策冲击（标准化）", "e_financial": "金融/风险冲击（标准化）"}
    names.update(shock_names)
    out = []
    for col, name in names.items():
        if col not in frame:
            continue
        x = frame[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        p2, p50, p98 = np.percentile(x, [2, 50, 98])
        B = phase_block(block, block, len(frame))
        means = start_values(frame[col].to_numpy(dtype=float), B, B, propagate_nan=True)
        ok = np.isfinite(means)
        b2, b98 = np.percentile(means[ok], [2, 98])
        w = start_prior(prior if prior is not None else np.full(len(frame), 1.0 / len(frame)), B, rare)[ok]
        w = w / w.sum()
        m = float(w @ means[ok])
        out.append({"var": col, "name": name, "p2": float(p2), "median": float(p50), "p98": float(p98),
                    "mean": float(x.mean()), "sd": float(x.std()),
                    "block_p2": float(b2), "block_p98": float(b98),
                    "block_sd": float(np.sqrt(w @ (means[ok] - m) ** 2))})
    return out
