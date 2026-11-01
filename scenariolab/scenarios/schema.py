"""Scenario spec schema and validation.

The same schema is used by the hand-written built-in scenarios (YAML) and by
the LLM translator. validate_spec() is the guard rail between a narrative and
the engine: it rejects unknown variables, bounds every number to what history
can support, pads or trims phases to the horizon, and reports every change so
the UI can show exactly what was adjusted.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np
import pandas as pd

from .. import config
from ..engine.pooling import BLOCK, start_prior, start_values, view_support
from ..engine.sampler import phase_block
from ..engine.simulate import DEFAULT_REVERSION

ALLOWED_PREFIXES = ("s_", "f_", "e_")
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
CONFIDENCE_BOUNDS = (0.3, 1.0)
SHIFT_SD_BOUNDS = (-3.0, 3.0)
VOL_MULT_BOUNDS = (0.3, 3.0)
REVERSION_BOUNDS = {
    "kappa_ff": (0.0, 0.1), "kappa_y10": (0.0, 0.1), "kappa_spread": (0.0, 0.1),
    "kappa_unrate": (0.0, 0.1), "kappa_oil": (0.0, 0.1), "r_star": R_STAR_BOUNDS, "term_premium": (-1.0, 3.0),
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
            "required": ["name", "months", "views", "overrides", "confidence"],
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


def drift_factor_ids() -> list[str]:
    """Factors a factor_drift can move: today's transmission (the core factors) and CPI, plus the
    profit share when the valuation state is on (it moves the fair value). The other factors only
    serve as views that pick historical months; a drift on them would change no path."""
    ids = list(config.CORE_FACTOR_IDS) + ["cpi"]
    if config.VALUATION.get("enabled"):
        ids.append("profit_share")
    return ids


def factor_drift_bounds(frame: pd.DataFrame) -> dict[str, float]:
    """+-2x the annualized historical volatility of each driftable factor's monthly change."""
    out = {}
    for fid in drift_factor_ids():
        col = f"f_{fid}"
        if col in frame:
            sd = float(frame[col].std()) * math.sqrt(12.0)
            out[fid] = 2.0 * sd if sd > 0 else 0.0
    return out


def validate_spec(spec: dict, frame: pd.DataFrame, horizon: int = config.HORIZON_MONTHS,
                  block: int = BLOCK) -> tuple[dict, dict]:
    """Return (clean_spec, report). `frame` is the MonthDB frame (view support).

    A view target is bounded by what the phase can average: the 2nd-98th
    percentile of the means of the `block`-month runs the sampler draws
    (pooling.view_support), which for monthly flows is much narrower than
    the range of single months.
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

    drift_bounds = factor_drift_bounds(frame)
    columns = set(frame.columns)
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
        views = []
        for v in (ph.get("views") or [])[:MAX_VIEWS_PER_PHASE]:
            if not isinstance(v, dict):
                continue
            var = str(v.get("var", ""))
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
                if fid in config.FACTOR_IDS and val is not None:
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
        for key, bounds in (("resid_scale", RESID_SCALE_BOUNDS), ("r_star", R_STAR_BOUNDS)):
            val = _num(ov.get(key))
            if val is not None:
                used = _clip(val, *bounds)
                if abs(used - val) > 1e-12:
                    clipped.append({"phase": k + 1, "var": key, "requested": val, "used": used})
                clean_ov[key] = used
        conf = _num(ph.get("confidence"))
        conf = _clip(conf, *CONFIDENCE_BOUNDS) if conf is not None else 1.0
        phases.append({"name": str(ph.get("name") or f"阶段{k + 1}")[:30], "months": months,
                       "views": views, "overrides": clean_ov, "confidence": conf})
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
    names.update({f"f_{f['id']}": f["name"] + "（月度）" for f in config.FACTORS})
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
