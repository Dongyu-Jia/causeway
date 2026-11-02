"""Narrative -> scenario spec via Claude ("LLM-driven, not LLM-invented").

The model only translates a story into the spec the engine already speaks:
structural shocks, entropy-pooling views on real historical months, phases, a
few bounded overrides and a causal chain. It never emits prices, returns or
K-lines. validate_spec clips the spec to the historical support and the
engine (EP over 6-month blocks -> block bootstrap -> factor transmission)
produces every number.

Backends (MSL_LLM_BACKEND): "claude_code", the default, runs the local Claude
Code CLI in print mode on the user's own subscription login (claude_code.py:
personal local use only, every call counts against the plan's limits);
"api" calls the Anthropic Messages API through the SDK, only when set
explicitly. Answers are cached in data/llm_cache/ by (backend, model, system
prompt, user message), so rerunning a narrative is free; at most
MSL_LLM_CONCURRENCY (2) calls run at once.

    python -m scenariolab.llm.translator "霍尔木兹海峡被封锁半年……" --run
    python -m scenariolab.llm.translator --print-prompt
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .. import config, jsonio
from ..engine.factor_model import STRUCTURAL_ASSETS, base_columns
from ..engine.simulate import DEFAULT_REVERSION
from ..env import dotenv_problems, load_dotenv
from ..scenarios import regime_views
from ..scenarios.schema import (ASSET_DRIFT_BOUND, MAX_PHASES, MAX_VIEWS_PER_PHASE, SPEC_JSON_SCHEMA,
                                allowed_variables, factor_drift_bounds, slugify, validate_spec)
from . import claude_code

try:
    import anthropic
except ImportError:  # optional dependency: pip install -e ".[llm]"
    anthropic = None

if TYPE_CHECKING:
    from ..context import EngineContext

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"   # pairs with fallbacks="default"
EFFORT = "high"
MAX_TOKENS = 16000                                   # non-streaming; thinking counts toward it
MAX_NARRATIVE_CHARS = 20000
HORIZON_BOUNDS = (12, 600)
ID_PREFIX = "llm_"                                   # keeps translations from shadowing built-in ids
SPECS_DIR = Path(__file__).resolve().parent.parent / "scenarios" / "specs"
PREFERRED_EXAMPLES = ("oil_war", "k_shaped")
EXAMPLE_NARRATIVE_CHARS = 1200
RESULT_PATH = config.RESULTS_DIR / "llm_translator.json"
REGIMES_PATH = config.RESULTS_DIR / "regimes.json"           # engine studies the regime section quotes
VIEW_KEYS = ("mean", "mean_annual", "shift_sd", "min", "max", "vol_mult")
TARGETS = ("mean", "mean_annual", "shift_sd")       # ep.apply_views uses the first one present
REFERENCE_FACTORS = ("f_oil", "f_ip", "f_cpi", "f_ff", "f_y10", "f_spread", "f_unrate")
VALUATION_VARS = ("f_valuation", "s_div_yield", "f_cape", "s_cape")   # views here come close to dictating equity prices
DEFAULT_OVERRIDE_WHY = "历史样本中没有先例的结构性效应（理由见 narrative）"
DRIFT_UNITS = {"Δlog": "Δlog/年", "Δpp": "pp/年", "Δ": "点/年"}
BACKENDS = ("claude_code", "api")
DEFAULT_BACKEND = "claude_code"                      # the local Claude Code CLI on the user's own subscription
API_LABEL = "Anthropic API（按 API 计费）"
CACHE_DIR = config.DATA_DIR / "llm_cache"            # <sha256>.json: an answer's text and its metadata only
DEFAULT_CONCURRENCY = 2                              # MSL_LLM_CONCURRENCY: model calls at once
STATUS_TTL_S = 60.0                                  # claude_code status: how long an answer is reused
STATUS_RETRY_S = 5.0                                 # ... when it is "unavailable" (a fresh login shows at once)


class TranslatorError(Exception):
    """Translation failed; the message is user-facing (Chinese)."""


# ---------------------------------------------------------------- status

def model_id() -> str:
    load_dotenv()
    return os.environ.get("MSL_LLM_MODEL", "").strip() or DEFAULT_MODEL


def backend_name() -> str:
    """MSL_LLM_BACKEND: "claude_code" (the default) or "api" (only when set explicitly); another value is
    returned as given, and status() then reports it as unknown."""
    load_dotenv()
    return os.environ.get("MSL_LLM_BACKEND", "").strip().lower().replace("-", "_") or DEFAULT_BACKEND


def _env_number(name: str, default, cast, minimum):
    """A numeric setting from the environment / .env; a missing, malformed or too small value gives the default."""
    load_dotenv()
    raw = os.environ.get(name, "").strip()
    try:
        value = cast(raw) if raw else default
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= minimum else default


def timeout_s() -> float:
    """MSL_LLM_TIMEOUT: seconds one Claude Code call may take (the CLI has no timeout flag)."""
    return float(_env_number("MSL_LLM_TIMEOUT", claude_code.DEFAULT_TIMEOUT_S, float, 1.0))


def concurrency() -> int:
    """MSL_LLM_CONCURRENCY: model calls that may run at once (the rest wait for a slot)."""
    return int(_env_number("MSL_LLM_CONCURRENCY", DEFAULT_CONCURRENCY, int, 1))


def _profile_configured() -> bool:
    """Whether the SDK would auto-discover an `ant auth login` profile or workload identity."""
    try:
        from anthropic.lib.credentials._constants import _has_auto_discoverable_credentials
        return bool(_has_auto_discoverable_credentials())
    except Exception:  # private SDK helper: absent or changed in other SDK versions
        return False


def _api_status(model: str) -> dict:
    if anthropic is None:
        return {"available": False, "model": model,
                "reason": "未安装 anthropic Python 包：在项目目录运行 pip install -e \".[llm]\" 后重启。"}
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if os.environ.get(var, "").strip():
            return {"available": True, "model": model, "reason": f"已配置 {var}"}
    if _profile_configured():
        return {"available": True, "model": model, "reason": "使用 ant CLI 登录配置（profile）中的凭据"}
    return {"available": False, "model": model,
            "reason": "未配置 Anthropic API 凭据：把 .env.example 复制为 .env 并填写 ANTHROPIC_API_KEY，然后重启服务。"}


def _token_problem(token: str) -> str | None:
    """Why CLAUDE_CODE_OAUTH_TOKEN cannot be a whole `claude setup-token` token (never quoting it), or None."""
    if not token.strip():
        return None
    if any(c.isspace() for c in token.strip()):
        return "CLAUDE_CODE_OAUTH_TOKEN 里有空格或换行：把 claude setup-token 生成的令牌完整地写成一行后重启服务。"
    for line, key in dotenv_problems():
        if key == claude_code.TOKEN_VAR:
            return (f"CLAUDE_CODE_OAUTH_TOKEN 在 .env 里被换行拆成了两行（第 {line} 行没有被读取，只读到了令牌的前半段）："
                    "把令牌合并成一行后重启服务。")
    return None


_STATUS: dict = {}                     # (model, MSL_CLAUDE_BIN, token set?) -> (_now() of the check, status)
_STATUS_LOCK = threading.Lock()
_now = time.monotonic


def _claude_code_status(model: str, force: bool = False) -> dict:
    token = os.environ.get(claude_code.TOKEN_VAR, "")
    key = (model, os.environ.get("MSL_CLAUDE_BIN", ""), bool(token.strip()))
    with _STATUS_LOCK:
        hit = _STATUS.get(key)
        if hit is not None and not force:
            at, st = hit
            if _now() - at < (STATUS_TTL_S if st["available"] else STATUS_RETRY_S):
                return dict(st)
        st = claude_code.status(model)
        problem = _token_problem(token) if st["available"] and st.get("auth") == claude_code.AUTH_TOKEN else None
        if problem:
            st.update(available=False, reason=problem)
        _STATUS[key] = (_now(), st)
        return dict(st)


def status(force: bool = False) -> dict:
    """{"backend", "label", "available", "model", "reason", ...}: local checks only, never a model call (no usage).

    claude_code adds binary, version and auth ("Claude Code 登录" / "CLAUDE_CODE_OAUTH_TOKEN" / "未登录"), from
    `claude --version` and `claude auth status --json` run with the call's own sanitized environment; the
    answer is reused for STATUS_TTL_S (an unavailable one for STATUS_RETRY_S) unless force=True.
    """
    backend, model = backend_name(), model_id()
    if backend == "api":
        return {"backend": "api", "label": API_LABEL, **_api_status(model)}
    if backend != "claude_code":
        return {"backend": backend, "label": backend, "available": False, "model": model,
                "reason": f"MSL_LLM_BACKEND={backend} 不认识：可用 claude_code（默认，本机 Claude Code）或 api（Anthropic API）。"}
    return _claude_code_status(model, force)


# ---------------------------------------------------------------- prompt pieces

def _fmt(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    ax = abs(float(x))
    if ax >= 99.95:
        return f"{x:.1f}"
    if ax >= 0.9995:
        return f"{x:.2f}"
    if ax == 0:
        return "0"
    return np.format_float_positional(float(x), precision=3, unique=False, fractional=False, trim="-")


def variable_table(ctx: "EngineContext") -> list[dict]:
    """allowed_variables + unit, kind, today's value and whether the asset model uses it."""
    hist = ctx.panel.loc[:ctx.anchor.date]
    units = {f"s_{s['id']}": s["unit"] for s in config.STATES}
    units.update({f"f_{f['id']}": f"{f['unit']}/月" for f in config.FACTORS})
    units["f_cpi"] = "Δlog/月"
    drivers = set(base_columns(ctx.model.spec))
    rows = []
    for v in allowed_variables(ctx.db.frame, prior=ctx.db.prior(), rare=ctx.db.rare):
        col = v["var"]
        kind = {"s": "state", "f": "factor", "e": "shock"}[col[0]]
        row = {**v, "kind": kind, "unit": units.get(col, "标准差"), "asset_driver": col in drivers}
        series = hist[col].dropna()
        if kind == "state" and len(series):
            row["today"] = float(series.iloc[-1])
            row["today_date"] = series.index[-1].strftime("%Y-%m")
            row["outside_support"] = not v["p2"] <= row["today"] <= v["p98"]
        elif kind != "state":
            recent = hist[col].iloc[-12:].dropna()
            row["last12_mean"] = float(recent.mean()) if len(recent) >= 6 else None
        rows.append(row)
    return rows


def _shock_links(frame, col: str) -> str:
    pairs = []
    for ref in REFERENCE_FACTORS:
        if ref in frame:
            c = float(frame[col].corr(frame[ref]))
            if np.isfinite(c):
                pairs.append((abs(c), ref, c))
    pairs.sort(reverse=True)
    return "、".join(f"{ref} {c:+.2f}" for _, ref, c in pairs[:2]) or "—"


def _view(var: str, why: str, **targets) -> dict:
    return {"var": var, **{k: targets.get(k) for k in VIEW_KEYS}, "why": why}


def _no_overrides() -> dict:
    return {"factor_drift": [], "asset_drift": [], "resid_scale": None, "r_star": None}


# Used only while scenarios/specs/ holds no built-in YAML to show the model.
INLINE_EXAMPLE = {
    "id": "oil_war",
    "name": "中东战争与石油供给冲击",
    "summary": "霍尔木兹海峡航运受阻，原油供给骤减，油价急涨推高通胀，美联储推迟降息，经济经历一段滞胀后逐步修复。",
    "narrative": (
        "中东爆发持续数月的军事冲突，霍尔木兹海峡航运受阻，全球约五分之一的海运原油供给中断。短期内原油需求缺乏弹性，"
        "供给缺口主要靠价格上涨出清：油价在半年内大幅上行，波动急剧放大。\n\n"
        "能源和运输成本迅速传导到商品通胀，并通过预期向服务价格扩散。面对供给冲击带来的通胀，美联储推迟降息并小幅加息；"
        "高油价挤压居民实际收入和企业利润，工业产出下滑、失业率上升、信用利差走阔，形成类似 1973–1974 年和 1979–1980 年的滞胀。\n\n"
        "冲突缓和后供给逐步恢复，油价部分回落，通胀和利率随之下降，之后回到无条件的历史常态。整个过程都在历史经验之内，"
        "因此不使用覆盖项。"
    ),
    "tags": ["石油", "战争", "滞胀"],
    "prior_prob": 0.07,
    "analogs": ["1973-10:1975-03", "1979-01:1980-12", "1990-08:1991-03"],
    "causal_chain": [
        {"from": "霍尔木兹海峡受阻", "to": "原油供给", "sign": "-", "lag": "0-1月",
         "mechanism": "约五分之一的海运原油经过海峡，航运中断直接减少供给"},
        {"from": "原油供给", "to": "油价", "sign": "-", "lag": "0-2月", "mechanism": "短期需求缺乏弹性，供给缺口靠价格上涨出清"},
        {"from": "油价", "to": "商品通胀", "sign": "+", "lag": "1-3月", "mechanism": "汽油、运输和化工成本直接进入商品 CPI"},
        {"from": "商品通胀", "to": "政策利率", "sign": "+", "lag": "3-9月", "mechanism": "通胀和通胀预期上行，美联储推迟降息甚至加息"},
        {"from": "油价", "to": "工业产出", "sign": "-", "lag": "3-12月", "mechanism": "能源成本挤压企业利润和居民实际收入，需求走弱"},
        {"from": "政策利率", "to": "信用利差", "sign": "+", "lag": "6-12月", "mechanism": "高利率叠加盈利下滑，违约风险上升"},
        {"from": "工业产出", "to": "失业率", "sign": "-", "lag": "6-18月", "mechanism": "产出下滑后企业裁员"},
        {"from": "油价", "to": "能源股", "sign": "+", "lag": "0-3月", "mechanism": "油价上涨直接提高能源企业的现金流"},
    ],
    "phases": [
        {"name": "冲击期", "months": 6, "views": [
            _view("f_oil", "油价半年累计上涨约 0.45 个对数单位（约 +57%），波动明显放大", mean_annual=0.9, vol_mult=1.5),
            _view("s_cpi_yoy", "多抽高通胀时期的月份：能源价格迅速推高整体通胀", mean=6.0),
            _view("f_cpi_goods", "商品价格年化上涨约 8%", mean_annual=0.08),
        ], "overrides": _no_overrides(), "confidence": 0.9},
        {"name": "滞胀期", "months": 18, "views": [
            _view("s_cpi_yoy", "通胀维持在高位", mean=6.5),
            _view("f_ip", "工业产出每年下降约 3%", mean_annual=-0.03),
            _view("f_unrate", "失业率每年上升约 0.6 个百分点", mean_annual=0.6),
            _view("f_spread", "信用利差每年走阔约 0.4 个百分点", mean_annual=0.4),
        ], "overrides": _no_overrides(), "confidence": 0.8},
        {"name": "修复期", "months": 36, "views": [
            _view("f_oil", "供给恢复，油价每年回落约 12%", mean_annual=-0.12),
            _view("s_cpi_yoy", "通胀回落但仍高于 2%", mean=3.5),
            _view("f_ff", "美联储每年降息约 0.6 个百分点", mean_annual=-0.6),
        ], "overrides": _no_overrides(), "confidence": 0.7},
        {"name": "回归常态", "months": 120, "views": [], "overrides": _no_overrides(), "confidence": 1.0},
    ],
}


def _override_whys(spec: dict) -> dict:
    """(phase index, override key, id) -> why, from list-form overrides."""
    out = {}
    for k, ph in enumerate(spec.get("phases") or []):
        ov = ph.get("overrides") if isinstance(ph, dict) else None
        for key, item_key in (("factor_drift", "factor"), ("asset_drift", "asset")):
            items = (ov or {}).get(key)
            if isinstance(items, list):
                for d in items:
                    if isinstance(d, dict) and d.get(item_key) and d.get("why"):
                        out[(k, key, str(d[item_key]))] = str(d["why"])
    return out


def _regime_items(ph: dict, source_ph) -> list[dict]:
    """A clean phase's regime key in the list form, with the source's own why where it gave one."""
    whys = {}
    raw = source_ph.get("regime") if isinstance(source_ph, dict) else None
    for d in raw if isinstance(raw, list) else []:
        rid = regime_views.regime_id(d.get("id")) if isinstance(d, dict) else None
        if rid and str(d.get("why") or "").strip():
            whys.setdefault(rid, str(d["why"]))
    return [{"id": r, "share": x, "why": whys.get(r) or regime_views.why_text(r, x)}
            for r, x in (ph.get("regime") or {}).items()]


def llm_form(spec: dict, source: dict | None = None, narrative_chars: int | None = EXAMPLE_NARRATIVE_CHARS) -> dict:
    """A validated spec in the structured-output shape: every view key present, overrides and the regime
    key as lists (the views the regime key expands to are not repeated as views).

    The narrative is cut to `narrative_chars` (the prompt's examples); None keeps all of it.
    """
    whys = _override_whys(source or spec)
    source_phases = (source or spec).get("phases") or []
    phases = []
    for k, ph in enumerate(spec.get("phases") or []):
        regime = _regime_items(ph, source_phases[k] if k < len(source_phases) else None)
        from_regime = {f"rg_{d['id']}" for d in regime}
        views = []
        for v in ph.get("views") or []:
            if v["var"] in from_regime:
                continue
            item = {"var": v["var"], **{key: v.get(key) for key in VIEW_KEYS}}
            for key in ("min", "max"):
                if item[key] is None and v.get(f"{key}_annual") is not None:
                    item[key] = float(v[f"{key}_annual"]) / 12.0
            item["why"] = str(v.get("why") or "")
            views.append(item)
        ov = ph.get("overrides") or {}
        drift = {key: [{item_key: i, "annual": a, "why": whys.get((k, key, i), DEFAULT_OVERRIDE_WHY)}
                       for i, a in (ov.get(key) or {}).items()]
                 for key, item_key in (("factor_drift", "factor"), ("asset_drift", "asset"))}
        phases.append({"name": ph.get("name", ""), "months": int(ph.get("months", 0)), "regime": regime,
                       "views": views,
                       "overrides": {**drift, "resid_scale": ov.get("resid_scale"), "r_star": ov.get("r_star")},
                       "confidence": ph.get("confidence", 1.0)})
    narrative = str(spec.get("narrative") or "")
    if narrative_chars is not None and len(narrative) > narrative_chars:
        narrative = narrative[:narrative_chars].rstrip() + "……"
    return {"id": spec.get("id", ""), "name": spec.get("name", ""), "summary": spec.get("summary", ""),
            "narrative": narrative, "tags": list(spec.get("tags") or []),
            "prior_prob": spec.get("prior_prob", 0.05), "analogs": list(spec.get("analogs") or []),
            "causal_chain": list(spec.get("causal_chain") or []), "phases": phases}


def _builtin_specs() -> dict[str, dict]:
    try:
        import yaml
    except ImportError:
        return {}
    out: dict[str, dict] = {}
    for path in sorted(SPECS_DIR.glob("*.yaml")):
        try:
            d = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(d, dict) and isinstance(d.get("phases"), list) and d["phases"]:
            out.setdefault(str(d.get("id") or path.stem), d)
    return out


def few_shot_examples(ctx: "EngineContext", horizon_months: int = config.HORIZON_MONTHS,
                      narrative_chars: int | None = EXAMPLE_NARRATIVE_CHARS) -> list[dict]:
    """Up to two built-in specs (preferring oil_war, k_shaped), else the inline example; validated, LLM form.

    A built-in's income-distribution calibration is resolved first, so the example shows
    the views and drifts the engine actually runs. narrative_chars: see llm_form.
    """
    from ..scenarios.library import calibration_base, resolve_calibration

    specs = _builtin_specs()
    order = [i for i in PREFERRED_EXAMPLES if i in specs] + sorted(i for i in specs if i not in PREFERRED_EXAMPLES)
    chosen = [specs[i] for i in order[:2]] or [INLINE_EXAMPLE]
    base = calibration_base(ctx)
    out = []
    for raw in chosen:
        resolved = resolve_calibration(raw, base)[0]
        clean, _ = validate_spec(resolved, ctx.db.frame, horizon_months)
        out.append(jsonio.clean(llm_form(clean, resolved, narrative_chars)))
    return out


def lab_examples(ctx: "EngineContext", horizon_months: int = config.HORIZON_MONTHS) -> list[dict]:
    """The prompt's few-shot examples with their whole narratives, for the lab's 载入示例 (no API key needed).

    Each is a built-in spec written by hand, laid out in the LLM's output form (SPEC_JSON_SCHEMA, as
    the prompt shows it): narrative -> spec -> engine end to end, labelled there as not a translation.
    """
    return few_shot_examples(ctx, horizon_months, narrative_chars=None)


def residual_rule() -> str:
    """Prompt text on the residuals the engine adds (config.RESIDUALS; fixed for a build)."""
    mode = config.RESIDUALS.get("resid_mode", "coupled")
    c = float(config.RESIDUALS.get("resid_coupling", 1.0))
    if mode == "coupled":
        return "再加上被抽中月份自身的残差（保留因子解释不了的跨资产联动）"
    if mode == "partial":
        return (f"再加上历史残差（因子解释不了的部分）：每条路径的每个 {config.BLOCK_MONTHS} 个月区块以 {c:.0%} 的概率"
                "沿用被抽中月份自己的残差（保留它们的跨资产联动），否则改用按无条件历史权重独立抽取的另一段残差，"
                f"所以视图挑中的历史片段只带进约 {c:.0%} 它们自己的泡沫或崩盘漂移")
    return "再加上历史残差（因子解释不了的部分，按引擎的残差设定从历史月份中抽取）"


def _state_rows(rows: list[dict], today: str) -> list[str]:
    out = ["| 变量 | 名称 | 单位 | 6个月均值 p2 | 中位数 | 6个月均值 p98 | 6个月均值标准差 | 今天 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        now = _fmt(r.get("today"))
        if r.get("today_date") and r["today_date"] != today:
            now += f"（{r['today_date']}）"
        if r.get("outside_support"):
            now += " ⚠"
        out.append(f"| {r['var']} | {r['name']} | {r['unit']} | {_fmt(r['block_p2'])} | {_fmt(r['median'])} | "
                   f"{_fmt(r['block_p98'])} | {_fmt(r['block_sd'])} | {now} |")
    return out


def _factor_rows(rows: list[dict]) -> list[str]:
    out = ["| 变量 | 名称 | 单位 | 进资产模型 | 6个月均值 p2 | 中位数 | 6个月均值 p98 | 6个月均值标准差 | 近12月均值 | 近12月折年(×12) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r.get("last12_mean")
        out.append(f"| {r['var']} | {r['name']} | {r['unit']} | {'是' if r['asset_driver'] else '否'} | "
                   f"{_fmt(r['block_p2'])} | {_fmt(r['median'])} | {_fmt(r['block_p98'])} | {_fmt(r['block_sd'])} | "
                   f"{_fmt(m)} | {_fmt(12.0 * m if m is not None else None)} |")
    return out


def _shock_section(rows: list[dict], frame) -> list[str]:
    if not rows:
        return ["本次构建没有结构冲击变量（e_*，因果分析尚未生成），请用 f_ / s_ 变量表达冲击。"]
    out = ["e_ 是因果分析识别出的月度结构冲击，已标准化（均值 0、标准差 1）。用 shift_sd（或以标准差为单位的 mean）"
           "表达；符号含义看它与因子的相关性（例如与 f_oil 正相关，意味着正值对应推高油价的冲击）。", "",
           "| 变量 | 名称 | 6个月均值 p2 | 中位数 | 6个月均值 p98 | 6个月均值标准差 | 进资产模型 | 近12月均值 | 与因子的相关 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['var']} | {r['name']} | {_fmt(r['block_p2'])} | {_fmt(r['median'])} | {_fmt(r['block_p98'])} | "
                   f"{_fmt(r['block_sd'])} | {'是' if r['asset_driver'] else '否'} | {_fmt(r.get('last12_mean'))} | "
                   f"{_shock_links(frame, r['var'])} |")
    return out


def _era_lines(model_start: str) -> list[str]:
    out = []
    for e in config.ERAS:
        span = f"{e['start']} 至 {e['end']}" if e["end"] else f"{e['start']} 至今"
        note = "（早于模型样本，只能作类比，不会被抽样）" if e["end"] and e["end"] < model_start else ""
        out.append(f"- {e['id']} {e['name']}（{span}）{note}：{e['summary']}")
    return out


def example_json(ex: dict) -> str:
    """An LLM-form spec as JSON with one view / edge per line (compact, still valid JSON)."""
    def d(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)

    head = [f" {d(k)}: {d(v)}" for k, v in ex.items() if k not in ("causal_chain", "phases")]
    chain = ",\n".join(f"  {d(e)}" for e in ex["causal_chain"])
    phases = []
    for ph in ex["phases"]:
        views = ",\n".join(f"    {d(v)}" for v in ph["views"])
        views = f"[\n{views}\n   ]" if views else "[]"
        regime = f', "regime": {d(ph["regime"])}' if "regime" in ph else ""
        phases.append(f'  {{"name": {d(ph["name"])}, "months": {d(ph["months"])}{regime},\n'
                      f'   "views": {views},\n'
                      f'   "overrides": {d(ph["overrides"])}, "confidence": {d(ph["confidence"])}}}')
    body = head + [f' "causal_chain": [\n{chain}\n ]', ' "phases": [\n' + ",\n".join(phases) + "\n ]"]
    return "{\n" + ",\n".join(body) + "\n}"


def valuation_rule() -> str:
    """Prompt text on valuation: a state, not a same-month driver; whether the engine pulls it toward a fair value."""
    pulled = config.VALUATION.get("enabled")
    engine = ("引擎会跟踪市场的市价/股息比（P/D），并按估计的公允值把股价向它拉回" if pulled else
              "引擎不做估值的路径回归（估值回归拉力在回测中没有通过检验），只在资本市场假设里按今天的估值水平做固定的长期漂移调整；"
              "每条模拟路径会记录它隐含的 P/D，供用户核对")
    return ("   - 估值（市盈率、市价/股息比、席勒 CAPE）不是同月驱动因子：它的月度变化几乎就是股价变化本身。"
            f"{engine}。叙事里的估值重估或压缩，要显式写成对 s_div_yield 的视图或有上限的 asset_drift。"
            "s_real_y10、s_infl_expect、s_profit_share_rt 可以作为状态视图；f_profit_growth、f_fcf_share 只在季度数据发布的月份变化，"
            "不进资产模型，只用来挑选历史月份。")


def _drift_rows(frame) -> list[str]:
    names = {f["id"]: (f["name"], DRIFT_UNITS.get(f["unit"], f["unit"] + "/年")) for f in config.FACTORS}
    names["cpi"] = ("CPI 总体", "Δlog/年")
    out = ["| 因子 | 名称 | 单位 | 上限（±） |", "|---|---|---|---|"]
    for fid, bound in factor_drift_bounds(frame).items():
        name, unit = names.get(fid, (fid, ""))
        out.append(f"| {fid} | {name} | {unit} | {_fmt(bound)} |")
    return out


_STUDIES: dict = {}


def regime_studies(ctx: "EngineContext") -> dict | None:
    """engine.studies of the build's regimes.json (what a regime request delivers in simulated paths, quoted
    by the prompt's regime section), or None when the file is missing or is for another data month: the
    clauses that quote a study then keep their rule without the number."""
    try:
        st = REGIMES_PATH.stat()
    except OSError:
        return None
    key = (str(REGIMES_PATH), st.st_mtime_ns, st.st_size)
    if key not in _STUDIES:
        try:
            res = jsonio.load(REGIMES_PATH)
        except (OSError, ValueError):
            res = None
        _STUDIES.clear()
        _STUDIES[key] = res if isinstance(res, dict) else None
    res = _STUDIES[key]
    if res is None or str(res.get("data_end")) != ctx.anchor.date.strftime("%Y-%m"):
        return None
    studies = (res.get("engine") or {}).get("studies")
    return studies if isinstance(studies, dict) else None


def regime_rule(ctx: "EngineContext") -> str:
    """The prompt section 「## 宏观体制（regime 字段）」: the rules from config.REGIMES, the history from the
    month database's support table, the study numbers from this build's regimes.json."""
    if not any(c in ctx.db.frame.columns for c in regime_views.VIEW_IDS):
        return ""
    return regime_views.translator_rule(ctx.db, regime_studies(ctx))


def build_system_prompt(ctx: "EngineContext", horizon_months: int = config.HORIZON_MONTHS,
                        examples: list[dict] | None = None) -> str:
    """Role, hard rules, engine semantics, variable support, today's state, eras, bounds and examples.

    Deterministic for a given build (no timestamps, fixed ordering) so prompt caching hits.
    """
    frame = ctx.db.frame
    rows = variable_table(ctx)
    today = ctx.anchor.date.strftime("%Y-%m")
    start, end = ctx.db.dates.min().strftime("%Y-%m"), ctx.db.dates.max().strftime("%Y-%m")
    shocks = [r for r in rows if r["kind"] == "shock"]
    examples = few_shot_examples(ctx, horizon_months) if examples is None else examples
    rev = DEFAULT_REVERSION
    oil_target = float(np.exp(ctx.db.oil_log_real_median + rev["oil_target_shift"]) * ctx.anchor.cpi)
    oil_gap = np.log(ctx.anchor.oil / oil_target)
    oil_drift = (f"，所以没有油价视图的阶段油价会从今天的 {ctx.anchor.oil:.0f} 美元缓慢"
                 f"{'回落' if oil_gap > 0 else '回升'}" if abs(oil_gap) > 0.1 else "")
    root = "、".join(([f"e_ 结构冲击（{'、'.join(r['var'] for r in shocks)}）"] if shocks else []) + [
        "油价（f_oil）", "政策利率（f_ff，或 overrides.r_star）", "信用利差（f_spread）",
        "劳动收入占比（f_labor_share）", "美元（f_usd）"])
    drift_assets = "、".join(f"{a['id']}（{a['name']}）" for a in config.ASSETS if a["id"] not in STRUCTURAL_ASSETS)
    years = f"{horizon_months / 12:g}"
    regime = regime_rule(ctx)

    lines = [
        "你是「宏观情景实验室」的情景翻译器。用户用自然语言描述一个宏观叙事（战争、技术变革、政策转向、金融危机、收入分配变化……），"
        "你把它翻译成一份情景规格 JSON，交给基于真实历史数据的模拟引擎执行。",
        "",
        "原则是「LLM 驱动，而不是 LLM 编造」：你负责把叙事转化为冲击、传导路径和对宏观变量的视图；资产价格、收益率和 K 线"
        "全部由引擎根据真实历史月份计算。你不输出任何价格、收益率、指数点位或 K 线，也不在任何文字中预测资产涨跌幅。",
        "",
        "## 引擎如何执行你的规格",
        "",
        f"1. 样本：{start} 至 {end} 共 {len(frame)} 个真实历史月份。",
        "2. 每个阶段的 views 约束的是「被抽中历史月份的平均值」：引擎把历史切成连续 6 个月的区块，用熵池（Entropy Pooling）"
        "给区块重新赋权，使被抽中区块的平均值满足视图，同时让权重尽量接近等权（相对熵最小），再按这些权重抽区块。"
        "每个模拟月份都是历史上真实发生过的某个月。视图描述的是阶段内的平均状态和平均变化速度，而不是某一个月的尖峰，"
        "所以 f_ 变量能设定的范围是「6 个月均值」的历史范围（见变量表），比单月窄得多。"
        "同一阶段的视图在历史上找不到足够多同时满足的区块时，引擎会把该阶段所有视图按同一比例 λ 向历史均值收缩，并报告 λ。",
        f"3. 模拟从今天（{today}）的实际水平出发，逐月累加被抽中月份的因子变化（f_ 变量）。因此：",
        "   - s_ 视图决定抽什么样的历史环境（高通胀、利差走阔、衰退……的月份），模拟路径继承这些月份的变化动态，"
        "而不是复制它们的绝对水平。例如 s_cpi_yoy mean=7 会多抽 1970 年代和 2021–2022 年的月份，它们较大的月度物价涨幅会逐步推高模拟通胀。",
        "   - 要让某个水平朝特定方向移动（油价、利率、失业率、劳动收入占比……），对相应的 f_ 变量设 mean_annual。",
        f"   - 利率类变化按「今天水平 ÷ 当时水平」缩放（1980 年利率 15% 时 +3pp 的月份，在今天 {ctx.anchor.ff:.2f}% 的政策利率下幅度小得多；"
        f"近零利率时期的变化最多放大 1.5 倍），"
        f"并缓慢均值回归：联邦基金利率 → r* + 通胀，10 年期收益率 → r* + 通胀 + {rev['term_premium']:g}，"
        f"信用利差 → {rev['spread_level']:g}，失业率 → {rev['unrate_level']:g}，"
        f"实际油价 → 1973 年以来的中位数（按今天的价格约 {oil_target:.0f} 美元/桶{oil_drift}；"
        f"某个阶段只要对油价设了视图（f_oil、s_oil_yoy、s_real_oil）或油价 factor_drift，这个阶段就不做油价回归，油价完全按视图走）；"
        f"默认 r* = {rev['r_star']:g}。因此 f_ 视图设定的是被抽中月份的变化速度，模拟出的水平路径还叠加了缩放和均值回归。",
        "   - s_ 同比类视图（如 s_cpi_yoy）挑的是「当时同比处于该水平」的月份，模拟的同比路径要用约 12 个月才追上它；"
        "要在阶段一开始就快速推高或压低某个水平，配合对应的 f_ 变化视图。",
        f"4. 资产收益完全由引擎计算：今天拟合的「因子 → 资产」传导系数作用于模拟的因子变化，{residual_rule()}；"
        "以及按今天估值水平做的长期漂移调整（资本市场假设，所以不必为高估值另设 asset_drift）；"
        "10 年期美债、原油和现金由收益率、油价和政策利率路径结构化生成。变量表中「进资产模型」为“是”的变量直接进入传导模型。",
        valuation_rule(),
        "5. 校验：超出历史 2%–98% 分位的视图目标、超出边界的覆盖项都会被截断，每一处调整都会报告给用户。",
        "",
        "## 硬性规则",
        "",
        "1. 只输出符合给定 JSON schema 的情景规格，不要输出其他文字。",
        "2. views 只能使用下方变量表中的变量。不得对资产收益或价格设视图（没有 r_、p_ 变量）；也不要用 f_valuation、"
        "s_div_yield、f_cape、s_cape 等估值变量间接规定股市涨跌，除非叙事本身就是估值重估（泡沫形成或破裂），且幅度温和。",
        f"3. 优先刻画根因：{root}。通胀、增长、就业和资产表现应当经由引擎从根因传导出来，而不是逐个规定结果。",
        "4. 留在历史支撑之内：目标值（f_ 变量按月度单位比较，即 mean_annual ÷ 12；与表中「6 个月均值」各列比较）尽量落在"
        "「中位数 ± 1.5 个标准差」以内，不要超出 p2–p98；同一阶段的几个视图在历史上要能同时出现（例如高通胀、极低失业、"
        "油价暴跌很少同时发生），否则只剩极少数区块可用，引擎会把整组视图向历史均值收缩。s_real_ff = s_ff − s_cpi_yoy、"
        "s_term_spread = s_y10 − s_ff，同一阶段不要三者同时设定。今天已处在历史区间之外的状态变量（表中标 ⚠）用 f_ 变化或 "
        "factor_drift 表达，不要用 s_ 水平视图。",
        f"5. phases 按时间顺序排列，months 为正整数，总和必须等于 {horizon_months} 个月（{years} 年），最多 {MAX_PHASES} 个阶段。",
        "6. factor_drift 和 asset_drift 只用于历史上没有先例、因子模型表达不了的效应，幅度要温和，并在每一项的 why 中说明理由；"
        "历史上发生过的冲击（油价、加息、衰退、金融危机）用 views 表达。r_star 只用来设定某阶段的货币政策取向，"
        "resid_scale 只用来设定整体波动环境；不需要时一律留空。",
        "7. 文字字段（name、summary、narrative、why、causal_chain、tags）一律用简体中文；id 用小写英文。",
        "8. causal_chain 写 4–12 条边，从根因出发经过宏观变量，可以延伸到资产类别，但只写方向（sign）、时滞（lag）"
        "和机制（mechanism）；文字里可以引用你设定的宏观视图数值，但不写资产价格、收益率或涨跌幅的数字。",
        "9. 叙事里直接给出资产结果时（如“纳指翻倍”“房价腰斩”），把它还原成背后的宏观或结构性原因（利率、利润、信用、"
        "估值），必要时用温和的 asset_drift，并在 narrative 中说明资产结果由引擎计算。",
        "10. 叙事信息不足时，按最可能的宏观含义补全假设，并在 narrative 中写明这些假设。",
        "",
        "## 视图的写法和单位",
        "",
        "- 每个视图只设 mean、mean_annual、shift_sd 三者之一，不用的字段一律为 null；同一阶段同一变量只出现一次；"
        "why 用一句话说明它对应叙事的哪一部分。",
        "- s_ 变量是水平值，单位见表（多为 % 或 pp）。用 mean（阶段平均水平），或 min / max（阶段平均的下限 / 上限）。",
        "- f_ 变量是月度变化：Δlog 类 0.01 ≈ 当月上涨 1%；Δpp 类 0.25 = 当月上升 0.25 个百分点。通常用 mean_annual = "
        "月均变化 × 12（每年的变化速度）。例：f_oil mean_annual=0.9 持续 6 个月 → 累计对数变化 0.45（油价约 +57%）；"
        "f_ff mean_annual=2.0 持续 12 个月 → 政策利率约 +2pp（再叠加均值回归）。mean_annual 只用于 f_ 变量。",
        "- shift_sd：把阶段平均值移动 k 个标准差（−3 到 3；以 6 个月均值的历史标准差为单位，见表），对任何变量都可用，"
        "也是 e_ 结构冲击的自然写法。",
        "- vol_mult：把该变量的波动放大或缩小为 k 倍（0.3–3），例如战争期间油价波动加剧。",
        "- min / max 的单位与 mean 相同（f_ 变量为月度单位），一般只用于 s_ 变量。",
        f"- 每阶段 2–5 个视图通常足够（最多 {MAX_VIEWS_PER_PHASE} 个）；views 为空的阶段按无条件历史抽样，适合作最后的「回归常态」。",
        "- confidence（0.3–1.0）：对该阶段视图的把握，小于 1 时把重新赋权的结果向无条件历史回拉；远期阶段通常更低。",
        "",
        f"## 可用变量（样本 {start} 至 {end}；「今天」= {today}）",
        "",
        "### 状态变量 s_（水平；p2/p98/标准差为连续 6 个月均值的分布，即视图能设定的范围；⚠ = 今天的单月值在历史 2%–98% 分位之外）",
        "",
        *_state_rows([r for r in rows if r["kind"] == "state"], today),
        "",
        "### 因子变量 f_（月度变化；p2/p98/标准差为连续 6 个月均值的分布，即视图能设定的范围；近12月折年 = 当前每年的变化速度，可直接与 mean_annual 比较）",
        "",
        *_factor_rows([r for r in rows if r["kind"] == "factor"]),
        "",
        "### 结构冲击 e_",
        "",
        *_shock_section(shocks, frame),
        "",
        "## 历史时期（类比词汇）",
        "",
        *_era_lines(config.MODEL_START),
        "",
        *([*regime.split("\n"), ""] if regime else []),
        "## 覆盖项 overrides（可选、有边界）",
        "",
        "- factor_drift：每年额外加到某个因子上的变化（单位同该因子，按年计），作用于该阶段的每个模拟月份，"
        "用于历史上没有先例的持续趋势。例：{\"factor\": \"labor_share\", \"annual\": -0.01, \"why\": \"AI 替代入门级白领岗位，"
        "劳动收入占比每年额外下降约 1%，历史上没有同类持续冲击\"}。各因子上限为历史年化波动的 2 倍：",
        "",
        *_drift_rows(frame),
        "",
        f"- asset_drift：每年额外加到某个资产价格回报上的漂移（小数，0.02 = 每年 +2%），只用于因子模型无法表达的结构性效应"
        f"（例如利润集中到少数公司）。上限 ±{ASSET_DRIFT_BOUND:g}，通常不超过 ±0.03。可用资产：{drift_assets}。"
        "cash、treasury_10y、crude_oil 由利率和油价路径生成，不能设。",
        "- resid_scale：该阶段全部资产残差（因子解释不了的部分，抽法见上文引擎第 4 条）的缩放（0.5–2.0），"
        "大于 1 表示因子解释不了的波动更大；不需要时为 null。",
        f"- r_star：该阶段的中性实际政策利率（−3 到 3，默认 {rev['r_star']:g}），决定利率均值回归的目标；"
        "金融抑制或长期停滞可取更低值；不需要时为 null。",
        "- 不需要覆盖项时写 {\"factor_drift\": [], \"asset_drift\": [], \"resid_scale\": null, \"r_star\": null}。",
        "",
        "## 其他字段",
        "",
        "- id：小写英文和下划线（不超过 40 个字符），如 ai_capex_bust。name：中文名称（不超过 20 字）。summary：一句话概括。",
        "- narrative：2–4 段：起因、传导机制、与哪些历史时期相似或不同、哪些部分超出历史因而使用了 overrides。"
        "可以复述你设定的宏观视图（如“油价半年上涨约五成”），但不要写任何资产价格、收益率或涨跌幅预测。",
        "- tags：3–6 个简短的中文标签。",
        f"- prior_prob：你对该情景在未来 {years} 年内大体成立的主观先验概率（0–1；常见情景 0.1–0.3，尾部情景 0.02–0.08），"
        "之后会在所有情景之间归一化。",
        "- analogs：1–4 个最相似的历史区间，格式 YYYY-MM:YYYY-MM；只用于展示和概率评估，不直接决定抽样权重（权重只由 views 决定）。",
        "- causal_chain 的节点名用简短中文（不超过 12 字）；sign 为 \"+\"（同向）、\"-\"（反向）或 \"±\"（方向取决于条件）；"
        "lag 如 \"0-3月\"、\"1-2年\"。",
        "",
        "## 示例",
        "",
        "以下示例展示格式和粒度；数值要根据新的叙事重新推导，不要照抄。示例来自内置情景库：它们的 narrative、summary 和"
        "因果链里「模拟的……」「权重最高的是……」这类数字是引擎运行之后回填的，你翻译时看不到引擎结果，所以只复述自己设定的"
        "视图和覆盖项，不要预测引擎会模拟出的水平或抽中哪些月份；narrative 按上面的要求写 2–4 段，比示例短。",
    ]
    for ex in examples:
        lines += ["", "```json", example_json(ex), "```"]
    lines += ["", "只输出一个 JSON 对象。"]
    return "\n".join(lines)


def schema_section() -> str:
    """The output format for the claude_code backend, which answers in plain text: the JSON schema that the
    API path passes as output_config.format instead (compact, one line)."""
    return "\n".join(["## 输出格式（JSON schema）", "",
                      "回答只有一个 JSON 对象，符合下面的 JSON schema：不要放进 ```json 代码块，前后不要写任何其他文字。", "",
                      json.dumps(SPEC_JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))])


def system_prompt(ctx: "EngineContext", horizon_months: int = config.HORIZON_MONTHS,
                  examples: list[dict] | None = None, backend: str | None = None) -> str:
    """The system prompt as `backend` (default: the configured one) sends it: build_system_prompt, plus the
    schema section for claude_code."""
    base = build_system_prompt(ctx, horizon_months, examples)
    return f"{base}\n\n{schema_section()}" if (backend or backend_name()) == "claude_code" else base


def _user_message(narrative: str, horizon_months: int) -> str:
    return f"请把下面的叙事翻译成 {horizon_months} 个月的情景规格。\n\n<narrative>\n{narrative}\n</narrative>"


# ---------------------------------------------------------------- API call

def _named_params(fn) -> set[str] | None:
    """Named parameters of `fn`, or None when it takes **kwargs (anything is accepted)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params}


def request_kwargs(create, model: str, system: str, user: str) -> dict:
    """Arguments for client.beta.messages.create; params this SDK does not name go via extra_body/headers."""
    kwargs = {
        "model": model, "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
    }
    optional = {"betas": [FALLBACK_BETA], "fallbacks": "default",
                "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SPEC_JSON_SCHEMA}}}
    named = _named_params(create)
    extra_body = {}
    for key, val in optional.items():
        if named is None or key in named:
            kwargs[key] = val
        elif key == "betas":
            kwargs["extra_headers"] = {"anthropic-beta": ",".join(val)}
        else:
            extra_body[key] = val
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _call(client, model: str, system: str, user: str):
    create = client.beta.messages.create
    kwargs = request_kwargs(create, model, system, user)
    if anthropic is None:
        return create(**kwargs)
    try:
        return create(**kwargs)
    except anthropic.AuthenticationError as err:
        raise TranslatorError("Anthropic API 凭据无效或已过期（401）：请检查 .env 中的 ANTHROPIC_API_KEY。") from err
    except anthropic.PermissionDeniedError as err:
        raise TranslatorError(f"当前 API 凭据无权使用模型 {model}（403）。") from err
    except anthropic.NotFoundError as err:
        raise TranslatorError(f"找不到模型 {model}（404）：请检查环境变量 MSL_LLM_MODEL。") from err
    except anthropic.RateLimitError as err:
        wait = err.response.headers.get("retry-after") if err.response is not None else None
        raise TranslatorError(f"请求过于频繁或额度已用尽（429），请{f'在 {wait} 秒后' if wait else '稍后'}重试。") from err
    except anthropic.BadRequestError as err:
        raise TranslatorError(f"API 拒绝了这次请求（400）：{err.message}") from err
    except anthropic.APIStatusError as err:
        if err.status_code >= 500:
            raise TranslatorError(f"Anthropic 服务暂时不可用（{err.status_code}），请稍后重试。") from err
        raise TranslatorError(f"Anthropic API 返回错误（{err.status_code}）：{err.message}") from err
    except anthropic.APITimeoutError as err:
        raise TranslatorError("请求超时：模型可能仍在长时间思考，请稍后重试或简化叙事。") from err
    except anthropic.APIConnectionError as err:
        raise TranslatorError("无法连接 Anthropic API：请检查网络或代理设置（ANTHROPIC_BASE_URL）。") from err
    except anthropic.CredentialsError as err:
        raise TranslatorError(f"无法获取 Anthropic API 凭据：{err}") from err
    except anthropic.AnthropicError as err:
        raise TranslatorError(f"调用 Claude 失败：{err}") from err


def parse_response(response) -> dict:
    """Spec dict from a Messages response; refusal / truncation / non-JSON raise TranslatorError."""
    return parse_spec_text(response_text(response))


def response_text(response) -> str:
    """The text of a Messages response; refusal / truncation / no text raise TranslatorError."""
    stop = getattr(response, "stop_reason", None)
    if stop == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
        suggested = getattr(details, "recommended_model", None)
        msg = "模型拒绝了这个请求" + (f"（类别：{category}）" if category else "") + "。"
        if explanation:
            msg += f"说明：{explanation}。"
        if suggested:
            msg += f"服务端建议改用 {suggested}（设置 MSL_LLM_MODEL）。"
        raise TranslatorError(msg + "请换一种表述，聚焦宏观经济机制后重试。")
    if stop == "max_tokens":
        raise TranslatorError(f"模型输出达到长度上限（max_tokens={MAX_TOKENS}），情景规格不完整：请缩短或简化叙事后重试。")
    if stop == "model_context_window_exceeded":
        raise TranslatorError("输入超出模型上下文窗口：请缩短叙事。")
    text = next((b.text for b in getattr(response, "content", None) or [] if getattr(b, "type", None) == "text"), "")
    if not text.strip():
        raise TranslatorError("模型没有返回情景规格（响应中没有文本）。")
    return text


FENCE_RE = re.compile(r"```[A-Za-z]*[ \t]*\n(.*?)```", re.S)


def parse_spec_text(text) -> dict:
    """The spec object in an answer's text: the whole text as JSON, else the first ``` block or the outermost
    {...} that is a JSON object (a plain-text answer may wrap it despite the prompt); TranslatorError otherwise."""
    s = str(text or "").strip()
    if not s:
        raise TranslatorError("模型没有返回情景规格（响应中没有文本）。")
    try:
        raw = json.loads(s)
    except json.JSONDecodeError as err:
        candidates = [m.group(1) for m in FENCE_RE.finditer(s)]
        if 0 <= s.find("{") < s.rfind("}"):
            candidates.append(s[s.find("{"):s.rfind("}") + 1])
        for c in candidates:
            try:
                raw = json.loads(c)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict):
                return raw
        raise TranslatorError(f"模型输出不是合法的 JSON（第 {err.lineno} 行第 {err.colno} 列）。") from err
    if not isinstance(raw, dict):
        raise TranslatorError("模型输出的 JSON 不是对象。")
    return raw


def usage_dict(response) -> dict:
    u = getattr(response, "usage", None)
    out = {}
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        val = getattr(u, key, None)
        if val is not None:
            out[key] = int(val)
    iterations = getattr(u, "iterations", None) or []
    if iterations:
        out["fallback_used"] = any(getattr(i, "type", None) == "fallback_message" for i in iterations)
    return out


# ---------------------------------------------------------------- backends, response cache, concurrency

def cache_key(backend: str, model: str, system: str, user: str) -> str:
    """sha256 of (backend, model, system prompt, user message): the response cache's file name."""
    blob = json.dumps([backend, model, system, user], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()     # a lone surrogate never raises


def cache_load(key: str) -> dict | None:
    """The cached {"text", "meta", "created"} for a key, or None (missing or unreadable)."""
    try:
        entry = json.loads((CACHE_DIR / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) or not isinstance(entry.get("meta"), dict):
        return None
    return entry


def cache_store(key: str, text: str, meta: dict) -> None:
    """data/llm_cache/<key>.json: the answer's text (verbatim) and its metadata, nothing else (no prompt, no
    narrative, no credentials), written atomically. A failed write (a full disk, text no encoder takes) only means
    the next rerun calls again."""
    path = CACHE_DIR / f"{key}.json"
    entry = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text, "meta": meta}
    tmp = path.with_name(f".{key}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, ValueError):
        try:
            tmp.unlink()
        except OSError:
            pass


_SLOTS: dict[int, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()


@contextmanager
def _slot():
    """One of MSL_LLM_CONCURRENCY call slots, waiting up to MSL_LLM_TIMEOUT for one to free up."""
    n, wait = concurrency(), timeout_s()
    with _SLOTS_LOCK:
        sem = _SLOTS.setdefault(n, threading.BoundedSemaphore(n))
    if not sem.acquire(timeout=wait):
        raise TranslatorError(f"已有 {n} 个翻译在进行（MSL_LLM_CONCURRENCY={n}），等了 {wait:g} 秒仍没有空位，请稍后重试。")
    try:
        yield
    finally:
        sem.release()


def _ask(backend: str, system: str, user: str, model: str, client=None) -> tuple[str, dict]:
    """One model call on `backend`: (the answer's text, metadata)."""
    if backend == "claude_code":
        try:
            return claude_code.run(system, user, model, timeout_s())
        except claude_code.ClaudeCodeError as err:
            raise TranslatorError(str(err)) from err
    if backend != "api":
        raise TranslatorError(f"MSL_LLM_BACKEND={backend} 不认识：可用 claude_code 或 api。")
    if client is None:
        if anthropic is None:
            raise TranslatorError(_api_status(model)["reason"])
        client = anthropic.Anthropic()
    response = _call(client, model, system, user)
    text = response_text(response)
    return text, {"backend": "api", "model": getattr(response, "model", None) or model,
                  "usage": usage_dict(response), "stop_reason": getattr(response, "stop_reason", None)}


def complete(system: str, user: str, model: str, backend: str = DEFAULT_BACKEND, client=None,
             refresh: bool = False) -> tuple[dict, dict]:
    """The spec object the model answers for (system, user) and the call's metadata.

    From data/llm_cache/ when this exact request was answered before (no call, no usage; meta["cached"]),
    else one call on `backend`, at most MSL_LLM_CONCURRENCY at once. Only an answer that parses as a JSON
    object is cached; refresh=True skips the lookup and replaces the entry.
    """
    key = cache_key(backend, model, system, user)
    if not refresh:
        hit = cache_load(key)
        if hit is not None:
            try:
                return parse_spec_text(hit["text"]), {**hit["meta"], "cached": True, "cached_at": hit.get("created")}
            except TranslatorError:
                pass                                   # an unreadable entry is asked again and replaced
    with _slot():
        text, meta = _ask(backend, system, user, model, client)
    raw = parse_spec_text(text)
    cache_store(key, text, meta)
    return raw, {**meta, "cached": False}


def request_info(backend: str, model: str) -> dict:
    """What one translation call sends, for llm_translator.json (no credentials)."""
    if backend == "api":
        return {"backend": "api", "max_tokens": MAX_TOKENS, "effort": EFFORT, "betas": [FALLBACK_BETA],
                "fallbacks": "default", "output_format": "json_schema"}
    if backend != "claude_code":
        return {"backend": backend}
    return {"backend": backend, "argv": claude_code.build_args("claude", model, "<system prompt file>"),
            "stdin": "user message", "cwd": "empty temporary directory", "timeout_s": timeout_s(),
            "concurrency": concurrency(), "output_format": "text (schema in the system prompt)",
            "env_removed": "ANTHROPIC_*, CLAUDECODE, CLAUDE_*, credential-named (*_API_KEY, *_TOKEN, *_SECRET, *_PASSWORD, ...) "
                           "(kept: CLAUDE_CONFIG_DIR, a non-empty CLAUDE_CODE_OAUTH_TOKEN)"}


# ---------------------------------------------------------------- repair + validation

def repair(raw: dict) -> tuple[dict, list[str]]:
    """Fix unambiguous shape / unit slips before validate_spec; every fix is reported."""
    spec = copy.deepcopy(raw)
    notes: list[str] = []
    sid = slugify(spec.get("id") or spec.get("name") or "")
    spec["id"] = sid if sid.startswith(ID_PREFIX) else ID_PREFIX + sid
    spec["source"] = "llm"
    pp = spec.get("prior_prob")
    if isinstance(pp, (int, float)) and 1.5 < pp <= 100:
        spec["prior_prob"] = pp / 100.0
        notes.append(f"prior_prob={pp:g} 看起来是百分数，已按 {pp / 100.0:g} 处理")
    for k, ph in enumerate(spec.get("phases") or [], start=1):
        if not isinstance(ph, dict):
            continue
        conf = ph.get("confidence")
        if isinstance(conf, (int, float)) and 1.5 < conf <= 100:
            ph["confidence"] = conf / 100.0
            notes.append(f"阶段 {k}：confidence={conf:g} 看起来是百分数，已按 {conf / 100.0:g} 处理")
        if isinstance(ph.get("regime"), list):
            items = []
            for item in ph["regime"]:
                if not isinstance(item, dict):
                    continue
                raw_id = item.get("id")
                rid = regime_views.regime_id(raw_id)
                if rid is None:
                    notes.append(regime_views.unknown_text(k, raw_id))
                    continue
                if rid != raw_id:
                    notes.append(f"阶段 {k}：宏观体制「{raw_id}」已按{regime_views.NAMES[rid]}处理")
                items.append({**item, "id": rid})
            ph["regime"] = items
        merged: dict[str, dict] = {}
        for v in ph.get("views") or []:
            if not isinstance(v, dict) or not v.get("var"):
                continue
            var = str(v["var"])
            v = {key: val for key, val in v.items() if val is not None}
            if var not in merged:
                merged[var] = v
                continue
            base = merged[var]
            for key, val in v.items():
                base.setdefault(key, val)
            notes.append(f"阶段 {k}：变量 {var} 出现多次，已合并为一个视图（冲突时保留第一个）")
        views = []
        for var, v in merged.items():
            if not var.startswith("f_") and "mean_annual" in v:
                val = v.pop("mean_annual")
                if not any(t in v for t in TARGETS):
                    v["mean"] = val
                    notes.append(f"阶段 {k}：mean_annual 只适用于 f_ 变量，{var} 已按 mean={val:g} 处理")
                else:
                    notes.append(f"阶段 {k}：mean_annual 只适用于 f_ 变量，已从 {var} 删除")
            given = [t for t in TARGETS if t in v]
            for t in given[1:]:
                v.pop(t)
                notes.append(f"阶段 {k}：{var} 同时给出 {given[0]} 和 {t}，引擎只使用 {given[0]}，已删除 {t}")
            if var in VALUATION_VARS:
                notes.append(f"阶段 {k}：{var} 几乎直接决定股市估值和股价，请确认叙事确实是估值重估")
            views.append(v)
        ph["views"] = views
        ov = ph.get("overrides") if isinstance(ph.get("overrides"), dict) else {}
        for key, item_key in (("factor_drift", "factor"), ("asset_drift", "asset")):
            for item in ov.get(key) or []:
                if isinstance(item, dict) and item.get(item_key) and not str(item.get("why") or "").strip():
                    notes.append(f"阶段 {k}：覆盖项 {key}.{item[item_key]} 没有给出理由")
    return spec, notes


def drawn_weights(weights: np.ndarray, n_months: int, block: int) -> np.ndarray:
    """Exact expected frequency with which sampler.sample_phase draws each month, given month weights."""
    from ..engine.pooling import month_frequency
    from ..engine.sampler import block_start_probs, phase_block

    T = len(weights)
    B = phase_block(block, n_months, T)
    return month_frequency(block_start_probs(weights, B), B, n_months, T)


def preview(ctx: "EngineContext", spec: dict, warnings: list[str], clipped=()) -> list[dict]:
    """Per phase: which real months the views select and what the sampler draws on average.

    `drawn_mean` is the exact expectation over simulate()'s draws (runner.phase_pools ->
    block bootstrap); `realized_share` = (drawn - prior) / (target - prior), which equals
    the phase's confidence unless its views were shrunk toward the prior. The regime checks of
    the pooled phases (as run_spec appends them) go to `warnings`; `clipped` is validate_spec's.
    """
    from ..scenarios.runner import ep_report, phase_pools

    pools = phase_pools(ctx, spec, warnings)
    warnings.extend(regime_views.spec_checks(ctx.db, spec, pools, clipped))
    frame = ctx.db.frame
    out, start = [], 0
    for ph, pool in zip(spec["phases"], pools):
        ep = ep_report(ctx.db, pool)
        views = []
        for v in ep["views"]:
            x = frame[v["var"]].to_numpy(float)
            x = np.where(np.isfinite(x), x, np.nanmean(x))
            item = {**v, "drawn_mean": float(pool.month_weights @ x)}
            want = (v.get("target") - v["prior_mean"]) if v.get("kind") == "mean" else 0.0
            if abs(want) > 1e-12:
                item["realized_share"] = (item["drawn_mean"] - v["prior_mean"]) / want
            views.append(item)
        eras = sorted(((str(e), float(w)) for e, w in ep["era_weights"].items() if w > 0), key=lambda kv: -kv[1])
        out.append({"name": ph["name"], "start_month": start, "months": ph["months"],
                    "ens": ep["ens"], "ens_ratio": ep["ens_ratio"], "converged": ep["converged"],
                    "shrink": ep["shrink"], "shrink_reason": ep["shrink_reason"], "ens_requested": ep["ens_requested"],
                    "episodes": ep["episodes"], "block": ep["block"], "confidence": ph.get("confidence", 1.0),
                    "era_weights": dict(eras), "top_months": ep["top_months"][:5], "views": views})
        start += ph["months"]
    return out


def translate(narrative: str, ctx: "EngineContext", horizon_months: int = config.HORIZON_MONTHS,
              client=None, model: str | None = None, refresh: bool = False) -> dict:
    """Narrative -> {"spec", "validation", "model", "usage", "raw", "elapsed_s", "backend", "cached", "meta", ...}.

    `spec` is the validated, clipped spec (ready for runner.run_spec); `raw` is what the model returned;
    `meta` records the call (model, duration, turns, usage; `cached` when it came from data/llm_cache/).
    The backend is MSL_LLM_BACKEND (claude_code by default); an injected SDK `client` (tests) means the
    api backend. refresh=True asks the model again instead of reusing a cached answer.
    """
    # a lone surrogate (valid JSON: "\ud800", from a half-pasted emoji) fails every UTF-8 encoder downstream
    text = str(narrative or "").strip().encode("utf-8", "replace").decode("utf-8")
    if not text:
        raise TranslatorError("叙事不能为空。")
    if len(text) > MAX_NARRATIVE_CHARS:
        raise TranslatorError(f"叙事过长（{len(text)} 字），请控制在 {MAX_NARRATIVE_CHARS} 字以内。")
    try:
        horizon = int(horizon_months)
    except (TypeError, ValueError):
        raise TranslatorError("horizon_months 必须是整数月数。") from None
    if not HORIZON_BOUNDS[0] <= horizon <= HORIZON_BOUNDS[1]:
        raise TranslatorError(f"情景时长必须在 {HORIZON_BOUNDS[0]}–{HORIZON_BOUNDS[1]} 个月之间。")
    model = model or model_id()
    backend = "api" if client is not None else backend_name()
    if client is None:
        st = status()
        if not st["available"]:
            raise TranslatorError(st["reason"])

    system = system_prompt(ctx, horizon, backend=backend)
    t0 = time.perf_counter()
    raw, meta = complete(system, _user_message(text, horizon), model, backend, client=client, refresh=refresh)
    elapsed = time.perf_counter() - t0

    spec, notes = repair(raw)
    clean, report = validate_spec(spec, ctx.db.frame, config.HORIZON_MONTHS)    # the engine always runs the full horizon
    if horizon < config.HORIZON_MONTHS:
        report["warnings"].append(f"引擎总是模拟 {config.HORIZON_MONTHS} 个月：叙事覆盖的 {horizon} 个月之后补了一个无视图的回归常态阶段")
    report["warnings"] = notes + report["warnings"]
    if not any(p["views"] or p["overrides"] for p in clean["phases"]):
        report["warnings"].append("规格中没有任何有效的视图或覆盖项，结果等同于无条件历史基准情景。")
    try:
        phases_preview = preview(ctx, clean, report["warnings"], report["clipped"])
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as err:
        phases_preview = []
        report["warnings"].append(f"无法预览各阶段的历史月份权重：{err}")
    return {"spec": clean, "validation": report, "model": meta.get("model") or model,
            "requested_model": model, "usage": meta.get("usage") or {}, "raw": raw,
            "elapsed_s": round(elapsed, 2), "preview": phases_preview,
            "backend": backend, "cached": bool(meta.get("cached")), "meta": meta}


# ---------------------------------------------------------------- module run + CLI

def run(ctx: "EngineContext", narrative: str | None = None, horizon_months: int = config.HORIZON_MONTHS,
        client=None, write: bool = True) -> dict:
    """What the LLM sees (prompt, schema, variable support, bounds) + status -> results/llm_translator.json.

    `examples` holds the prompt's few-shot examples with whole narratives (lab_examples: the lab's
    载入示例). Without a narrative (the build's llm step) it never calls a model: status() only runs
    `claude --version` and `claude auth status`, which spend no usage. With a narrative (and an available
    backend or an injected client) the translation is included.
    """
    examples = few_shot_examples(ctx, horizon_months)
    st = status()
    system = system_prompt(ctx, horizon_months, examples, backend=st["backend"])
    out = {
        "status": st, "backend": st["backend"], "model": st["model"], "horizon_months": horizon_months,
        "today": ctx.anchor.date, "request": request_info(st["backend"], st["model"]),
        "prompt": {"system": system, "chars": len(system),
                   "sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
                   "examples": [e["id"] for e in examples]},
        "variables": variable_table(ctx),
        "factor_drift_bounds": factor_drift_bounds(ctx.db.frame),
        "asset_drift": {"bound": ASSET_DRIFT_BOUND,
                        "assets": [a for a in config.ASSET_IDS if a not in STRUCTURAL_ASSETS]},
        "schema": SPEC_JSON_SCHEMA,
        "regime_support": regime_views.support_rows(ctx.db),
        "examples": lab_examples(ctx, horizon_months),
        "translation": translate(narrative, ctx, horizon_months, client=client) if narrative else None,
    }
    if write:
        jsonio.dump(out, RESULT_PATH)
    return jsonio.clean(out)


def spec_to_yaml(spec: dict) -> str:
    """Validated spec as YAML in the built-in format (scenarios/specs/*.yaml)."""
    import yaml

    d = jsonio.clean(spec)
    if not d.get("reversion"):
        d.pop("reversion", None)
    return yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=1000)


def _print_headline(detail: dict) -> None:
    from ..scenarios.runner import headline

    h = headline(detail)
    print(f"\n引擎结果（{detail['n_paths']} 条路径，起点 {detail['start'][:7]}，{len(detail['labels']) - 1} 个月）：")
    print(f"  美股大盘 名义年化中位数 {h['us_equity_cagr_p50']:.2%}，实际 {h['us_equity_real_cagr_p50']:.2%}；"
          f"平均 CPI 同比 {h['cpi_avg']:.2f}%；期末联邦基金利率中位数 {h['ff_end_p50']:.2f}%；最少有效月份数 {h['ens_min']:.0f}")
    names = {a["id"]: a["name"] for a in config.ASSETS}
    for aid, a in detail["assets"].items():
        s = a["stats"]
        print(f"  {names.get(aid, aid):<10} 年化 p5/p50/p95 {s['cagr_p5']:7.2%} {s['cagr_p50']:7.2%} {s['cagr_p95']:7.2%}"
              f"  实际 p50 {s['real_cagr_p50']:7.2%}  最大回撤 p50 {s['max_dd_p50']:7.2%}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scenariolab.llm.translator",
                                description="用 Claude 把叙事翻译成情景规格，可选直接交给引擎运行。默认用本机 Claude Code"
                                            "（你自己的订阅登录，计入订阅用量）；MSL_LLM_BACKEND=api 时改用 Anthropic API。")
    p.add_argument("narrative", nargs="?", help="情景叙事；'-' 表示从标准输入读取")
    p.add_argument("--yaml", metavar="OUT", help="把校验后的规格写成 YAML（与内置情景同一格式）")
    p.add_argument("--run", action="store_true",
                   help=f"用引擎运行规格并打印核心统计（按 {config.HORIZON_MONTHS} 个月运行）")
    p.add_argument("--horizon", type=int, default=config.HORIZON_MONTHS, help="情景总月数")
    p.add_argument("--paths", type=int, default=600, help="--run 时的模拟路径数")
    p.add_argument("--model", help="覆盖 MSL_LLM_MODEL")
    p.add_argument("--refresh", action="store_true", help="不用 data/llm_cache/ 里的缓存，重新调用模型")
    p.add_argument("--print-prompt", action="store_true", help="只打印系统提示词，不调用模型")
    args = p.parse_args(argv)

    from ..context import EngineContext

    ctx = EngineContext.load(selection=True)
    if args.print_prompt:
        print(system_prompt(ctx, args.horizon))
        return 0
    if not args.narrative:
        p.error("需要提供叙事（或使用 --print-prompt）")
    narrative = sys.stdin.read() if args.narrative == "-" else args.narrative
    try:
        result = translate(narrative, ctx, args.horizon, model=args.model, refresh=args.refresh)
    except TranslatorError as err:
        print(f"翻译失败：{err}", file=sys.stderr)
        return 1
    print(json.dumps(jsonio.clean(result["spec"]), ensure_ascii=False, indent=2))
    v = result["validation"]
    meta = result.get("meta") or {}
    turns = f"，{meta['num_turns']} 轮" if meta.get("num_turns") else ""
    cached = "（来自 data/llm_cache/ 的缓存，没有调用模型；--refresh 重新调用）" if result.get("cached") else ""
    print(f"\n模型 {result['model']}，后端 {result['backend']}，用时 {result['elapsed_s']} 秒{turns}{cached}，"
          f"tokens {result['usage']}")
    print(f"校验：{'通过' if v['ok'] else '有错误'}，{len(v['warnings'])} 条提示，{len(v['clipped'])} 处截断")
    for line in v["errors"] + v["warnings"]:
        print(f"  - {line}")
    for ph in result["preview"]:
        eras = "、".join(f"{e} {w:.0%}" for e, w in list(ph["era_weights"].items())[:3])
        print(f"  阶段「{ph['name']}」{ph['months']} 个月：有效历史月份 {ph['ens']:.0f}，主要来自 {eras}")
    if args.yaml:
        Path(args.yaml).write_text(spec_to_yaml(result["spec"]), encoding="utf-8")
        print(f"\n已写入 {args.yaml}")
    if args.run:
        from ..scenarios.runner import run_spec

        detail, _ = run_spec(ctx, result["spec"], n_paths=args.paths, source="llm")
        _print_headline(detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
