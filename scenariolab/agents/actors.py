"""The actor registry of the agent tree (AGENT_DESIGN.md §4, §12.1; ACTORS_V2; actors.json v1).

Who acts, what each may choose, and how a chosen option reaches the input table:

- **roles, not field ids.** Every write names a lever *role* (``policy_rate_chg``, ``tariff_chg``, ...). A
  :class:`FieldResolver` binds roles to schema fields through the agents overlay's role and alias tables
  (``d_policy -> d_ff``, ``d_lfpr -> d_lf``, ...). A role that does not resolve falls back as the catalogue says
  (the hyperscaler capex instrument writes a shift on IT investment; China's instruments write facts; OPEC+'s
  supply writes the oil-supply composite), and an option left with nothing to write is disabled with a reason.
- **actors** (:data:`ACTORS`): kind, stage, when active, the LLM pool weight c (判断值), custom-option cap, own
  levers, own account book, and one *main* domain (the options that branch) plus *attribute* domains (chosen per
  option or allocated across particles, never a branch of their own).
- **options** are catalogue options: a canonical id, at most one magnitude inside a range in schema units, the
  writes (point / normal / shift / bin / label, commitments ``dq`` quarters later), facts, political-state updates,
  account actions, a gate and a default public signal. :func:`menu` applies calendars, gates, pinned facts and the
  account gates; :func:`option_payload` turns a choice into writes, clipped to the field's history;
  :func:`combine_events` sums the shared event channels (the oil-supply composite and GPR).
- **reactions** (:data:`REACTIONS`): same-quarter conditional rows for catalogue pairs; :func:`lagged_rules` the
  next-quarter code rules.
- **pinned facts** (制度现状, :data:`INSTITUTION_DEFAULTS`): dated, sourced items the user confirms; every shipped
  default is marked 未确认. ``data/agents/institutions.json`` holds the user's edits (never an LLM's).

Nothing here calls a model or an LLM. Magnitudes that rest on no data are marked 判断值 in ``note``.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .. import config, jsonio

CATALOGUE_VERSION = "actors-v2.1 (2026-11-01; AGENT_DESIGN §4 on schema-revise-2 3.0)"
AGENTS_DATA_DIR = config.DATA_DIR / "agents"
INSTITUTIONS_PATH = AGENTS_DATA_DIR / "institutions.json"
OVERLAY_PATH = Path(__file__).resolve().parent / "overlay.json"     # written by the registry builder (if present)

# =====================================================================================================================
# 1. Roles and the alias table: the only place where lever roles meet schema field ids
# =====================================================================================================================

ROLES: dict[str, dict] = {
    # instruments: one owner each. `default` = the resolution on schema-revise-2 3.0 (94 fields, 2026-11-01), used
    # only when no overlay or schema field list is loaded; None = no field yet (the fallback applies)
    "policy_rate_chg": {"candidates": ("d_ff", "d_policy"), "default": "d_ff", "kind": "instrument", "who": ("fed",)},
    "fed_bs_chg": {"candidates": ("d_bs",), "default": "d_bs", "kind": "instrument", "who": ("fed",)},
    "tariff_chg": {"candidates": ("d_tariff",), "default": "d_tariff", "kind": "instrument", "who": ("president",)},
    "hyperscaler_capex_g": {"candidates": ("hyperscaler_capex_g", "x_hyperscaler_capex"), "default": None,
                            "kind": "instrument", "who": ("bigtech",), "fallback": "it_invest_g"},
    "bill_share_chg": {"candidates": ("x_bill_share_chg", "d_bill_share"), "default": None, "kind": "instrument",
                       "who": ("treasury",)},
    "cn_tariff_chg": {"candidates": ("x_cn_tariff_chg",), "default": None, "kind": "instrument", "who": ("china",),
                      "fallback": "fact"},
    "cny_logchg": {"candidates": ("x_cny_logchg",), "default": None, "kind": "instrument", "who": ("china",),
                   "fallback": "fact"},
    "cn_credit_impulse": {"candidates": ("x_cn_credit_impulse",), "default": None, "kind": "instrument",
                          "who": ("china",), "fallback": "fact"},
    "opec_supply_chg": {"candidates": ("x_opec_supply_chg",), "default": None, "kind": "instrument", "who": ("opec",),
                        "fallback": "oil_composite"},
    # levers on E0 quantities and E0p policy fields: listed writers, shifts only (on an E0p field a shift is the
    # discretionary shock u on top of the model's systematic part, schema v3 rule)
    "deficit_chg": {"candidates": ("d_deficit",), "default": "d_deficit", "kind": "shift_target", "who": ("congress",)},
    "transfers_chg": {"candidates": ("d_transfers",), "default": "d_transfers", "kind": "shift_target",
                      "who": ("congress",)},
    "corp_tax_chg": {"candidates": ("d_corptax",), "default": "d_corptax", "kind": "shift_target", "who": ("congress",)},
    "pers_tax_chg": {"candidates": ("d_perstax",), "default": "d_perstax", "kind": "shift_target", "who": ("congress",)},
    "defense_chg": {"candidates": ("d_defense",), "default": "d_defense", "kind": "shift_target", "who": ("congress",)},
    "labor_force_g": {"candidates": ("d_lf", "d_lfpr"), "default": "d_lf", "kind": "shift_target",
                      "who": ("president",)},
    "it_invest_g": {"candidates": ("d_itinv",), "default": "d_itinv", "kind": "shift_target", "who": ("bigtech",)},
    "capex_g": {"candidates": ("d_capex",), "default": "d_capex", "kind": "shift_target", "who": ("bigtech",)},
    # shared event channels (contributions are summed by combine_events)
    "oil_supply_shock": {"candidates": ("x_oil_supply_shock", "e_oil_supply"), "default": None, "kind": "event",
                         "who": ("nature", "belligerents", "opec", "president"), "fallback": "oil_supply_signature"},
    "oil_supply_signature": {"candidates": ("d_metals_oil",), "default": "d_metals_oil", "kind": "event_signature",
                             "who": ("nature", "belligerents", "opec", "president")},
    "oil_cause_label": {"candidates": ("x_oil_cause",), "default": "x_oil_cause", "kind": "event_label",
                        "who": ("nature", "belligerents", "opec", "president"), "fallback": "none"},
    "gpr_chg": {"candidates": ("d_gpr",), "default": "d_gpr", "kind": "event", "who": ("nature", "belligerents", "china")},
}

# old ids -> schema ids (schema-revise-1/scripts/xform.py); None = dropped, with the reason shown
ALIASES: dict[str, str | None] = {"d_policy": "d_ff", "d_lfpr": "d_lf", "y_lpd": "y_mult", "s_ebp": "s_spread",
                                  "s_gz": "s_spread", "d_ebp": None}
DROPPED_WHY = {"d_ebp": "EBP 用同季股价构造，输入表 v2 已删除；金融条件改用 GZ 利差 d_gz 的分区"}

# Field supports used when no registry `hist` is given: the 1985-2019 quarterly sd of the event channels (d01)
FALLBACK_SD = {"gpr_chg": 22.15, "oil_supply_signature": 16.26}


class FieldResolver:
    """Role -> schema field id, and old id -> current id, from the overlay (or the built-in tables).

    `fields` is the set of schema field ids; None means the schema is unknown, and a role then resolves to its
    first candidate that is not a known-missing id (the overlay's own resolution when one was loaded)."""

    def __init__(self, fields: Iterable[str] | None = None, roles: Mapping | None = None,
                 aliases: Mapping | None = None, resolved: Mapping | None = None, info: Mapping | None = None):
        self.fields = frozenset(fields) if fields is not None else None
        self.roles = {k: dict(v) for k, v in ROLES.items()}
        for k, v in (roles or {}).items():
            cand = tuple(v.get("candidates") or ([v["field"]] if v.get("field") else []))
            self.roles[k] = {**self.roles.get(k, {}), **{kk: vv for kk, vv in v.items() if kk != "candidates"},
                             "candidates": cand or self.roles.get(k, {}).get("candidates", ())}
        self.aliases = {**ALIASES, **dict(aliases or {})}
        self._resolved = dict(resolved or {})
        self.info = dict(info or {})            # field id -> overlay entry (cls, endo, entry, hist, ...)

    @classmethod
    def from_overlay(cls, overlay: Mapping) -> "FieldResolver":
        """An overlay (or a schema.json) dict: `fields` as {id: entry} or [{"id": ...}, ...], `roles`, `aliases`."""
        roles = overlay.get("roles") or {}
        resolved = {k: v.get("field") for k, v in roles.items() if isinstance(v, Mapping) and "field" in v}
        fields = overlay.get("fields")
        if isinstance(fields, list) and fields and isinstance(fields[0], Mapping):
            fields = {f["id"]: f for f in fields if isinstance(f, Mapping) and f.get("id")}
        ids = list(fields) if fields else None
        return cls(fields=ids, roles={k: v for k, v in roles.items() if isinstance(v, Mapping)},
                   aliases=overlay.get("aliases"), resolved=resolved,
                   info=fields if isinstance(fields, Mapping) else None)

    @classmethod
    def load(cls, path: Path | None = None) -> "FieldResolver":
        """The overlay next to this module when it exists (the registry builder writes it), else the built-in."""
        p = Path(path) if path else OVERLAY_PATH
        try:
            return cls.from_overlay(jsonio.load(p))
        except (OSError, ValueError, AttributeError, TypeError):
            return cls()

    def canonical(self, fid: str) -> str:
        """The current id of a field (aliases resolved). A dropped id raises a KeyError with a Chinese reason."""
        if fid in self.aliases:
            new = self.aliases[fid]
            if new is None:
                raise KeyError(f"字段「{fid}」已从输入表删除：{DROPPED_WHY.get(fid, '没有替代字段')}")
            return new
        return fid

    def field(self, role: str) -> str | None:
        """The schema field a role writes, or None when it does not resolve (the catalogue's fallback applies)."""
        spec = self.roles.get(role)
        if spec is None:
            raise KeyError(f"未知的杠杆角色「{role}」")
        if self.fields is None:
            if role in self._resolved:
                return self._resolved[role]
            return spec.get("default")
        for c in spec.get("candidates", ()):
            try:
                c = self.canonical(c)
            except KeyError:
                continue
            if c in self.fields:
                return c
        return None

    def __call__(self, role: str) -> str | None:
        return self.field(role)

    def hist(self, fid: str | None) -> dict | None:
        if not fid:
            return None
        h = (self.info.get(fid) or {}).get("hist") if self.info else None
        return h if isinstance(h, Mapping) else None

    def entry(self, fid: str | None) -> str | None:
        return (self.info.get(fid) or {}).get("entry") if (fid and self.info) else None

    def problems(self) -> list[str]:
        """Roles that resolve to nothing and have no declared fallback (shown, not fatal)."""
        out = []
        for role, spec in self.roles.items():
            if self.field(role) is None and not spec.get("fallback"):
                out.append(f"杠杆角色「{role}」在输入表里没有字段，也没有声明后备写法")
        return out

    def hash(self) -> str:
        import hashlib
        body = json.dumps({"fields": sorted(self.fields) if self.fields else None,
                           "roles": {k: list(v.get("candidates", ())) for k, v in sorted(self.roles.items())},
                           "aliases": self.aliases, "resolved": self._resolved}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def default_resolver() -> FieldResolver:
    return FieldResolver.load()


def _resolver(r) -> FieldResolver:
    if r is None:
        return default_resolver()
    if isinstance(r, FieldResolver):
        return r
    if isinstance(r, Mapping):
        return FieldResolver.from_overlay(r)
    if callable(r):                          # any callable role -> field (the engine's registry)
        fr = FieldResolver()
        fr.field = r                         # type: ignore[method-assign]
        return fr
    raise TypeError("resolver must be a FieldResolver, an overlay dict or a callable role -> field")


# =====================================================================================================================
# 2. Catalogue types
# =====================================================================================================================

@dataclass(frozen=True)
class Param:
    """The one magnitude an option may carry, in schema units (the LLM never types a field value)."""
    name: str
    zh: str
    unit: str
    lo: float
    hi: float
    default: float
    step: float | None = None

    def clip(self, v) -> tuple[float, list[str]]:
        flags = []
        try:
            x = float(v)
        except (TypeError, ValueError):
            return self.default, [f"幅度「{v}」不是数，用默认值 {self.default:g}"]
        if not math.isfinite(x):
            return self.default, [f"幅度「{v}」不是有限数，用默认值 {self.default:g}"]
        if x < self.lo or x > self.hi:
            flags.append(f"{self.zh} {x:g}{self.unit} 超出选项范围 [{self.lo:g}, {self.hi:g}]，已截断")
            x = min(max(x, self.lo), self.hi)
        if self.step:
            r = round(x / self.step) * self.step
            if abs(r - x) > 1e-9:
                flags.append(f"{self.zh} 按 {self.step:g}{self.unit} 取整：{x:g} → {r:g}")
            x = r
        return x, flags


@dataclass(frozen=True)
class Write:
    """One write on a lever role. kind: point | normal | shift | bin | label.

    per_unit: a and b are per unit of the option's magnitude; magnitude=True: a is the magnitude itself (points).
    dq > 0 makes it a commitment due dq quarters later (cancellable by the same actor's later option)."""
    role: str
    kind: str
    a: float = 0.0
    b: float = 0.0
    per_unit: bool = False
    magnitude: bool = False
    lo: float = -math.inf
    hi: float = math.inf
    dq: int = 0
    note: str = ""


@dataclass(frozen=True)
class Option:
    id: str
    zh: str
    status_quo: bool = False
    param: Param | None = None
    writes: tuple[Write, ...] = ()
    oil_z: tuple[float, float, bool] | None = None      # (a, b, per_unit): oil-supply composite contribution, z units
    gpr_sd: tuple[float, float] | None = None           # GPR contribution in quarterly sd units (1985-2019)
    facts: tuple[tuple[str, object], ...] = ()          # (fact id, value) set when chosen
    clears: tuple[str, ...] = ()                        # facts the option ends
    political: tuple[tuple[str, object], ...] = ()      # political-state updates
    account: tuple[tuple[str, object], ...] = ()        # account actions ("magnitude" = the option's magnitude)
    gate: str | None = None                             # gate rule id (see GATES)
    bin: tuple[float, float] | None = None              # bin of the instrument's own A0 distribution
    signal: tuple[str, int, int] = ("", 0, 1)           # default public signal: text, direction, horizon (q)
    protocol: str | None = None                         # conditioning protocol (§6.3): lagged | response_table | ...
    flags: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Domain:
    id: str
    zh: str
    role: str                                           # main | attribute
    options: tuple[Option, ...]
    live: str = "always"                                # live rule id (see LIVE)
    max_options: int = 4
    prior: str = ""                                     # prior id in priors.py
    branch_when: str = ""                               # documented promotion rule for attributes (engine decides)
    note: str = ""

    def option(self, oid: str) -> Option:
        for o in self.options:
            if o.id == oid:
                return o
        raise KeyError(oid)

    @property
    def status_quo(self) -> Option:
        return next(o for o in self.options if o.status_quo)


@dataclass(frozen=True)
class Actor:
    id: str
    zh: str
    kind: str                                           # policy | calendar | state | corporate | code
    stage: int
    active: str                                         # activity rule id (see ACTIVE)
    pool_c: float                                       # LLM weight in the linear pool (判断值)
    domains: tuple[Domain, ...]
    levers: tuple[str, ...] = ()                        # roles this actor may write
    custom_cap: float = 0.0                             # > 0: one custom option allowed, probability capped
    book: str | None = None                             # own account book (accounts.py)
    watches: tuple[str, ...] = ()                       # <= 2 lines watched in other books ("book.line")
    naming: str = ""
    prior_note: str = ""
    note: str = ""

    @property
    def main(self) -> Domain:
        return next(d for d in self.domains if d.role == "main")

    def domain(self, did: str) -> Domain:
        for d in self.domains:
            if d.id == did:
                return d
        raise KeyError(did)

    def find(self, oid: str) -> tuple[Domain, Option]:
        for d in self.domains:
            for o in d.options:
                if o.id == oid:
                    return d, o
        raise KeyError(f"{self.zh}没有选项「{oid}」")


def _o(oid, zh, **kw) -> Option:
    for k in ("writes", "facts", "clears", "political", "account", "flags"):
        if k in kw and not isinstance(kw[k], tuple):
            kw[k] = tuple(kw[k])
    return Option(id=oid, zh=zh, **kw)


INF = math.inf
FED_STEP = 0.25

# =====================================================================================================================
# 3. The actors (AGENT_DESIGN §4.2 and §4.4)
# =====================================================================================================================

FED = Actor(
    id="fed", zh="美联储", kind="policy", stage=4, active="always", pool_c=0.5, book="fed",
    levers=("policy_rate_chg", "fed_bs_chg"), watches=("fiscal.limit_headroom", "fiscal.tga"),
    prior_note="模型自己的 A0 分箱；模型就绪前用 7 档有序 probit（W_A01：2000–2026 样本外对数得分 −1.47，无条件 −1.86）",
    domains=(
        Domain("rate", "利率（本季两次会议合计）", "main", prior="fed_probit", max_options=4, options=(
            _o("cut_75plus", "紧急降息（≥75bp）", param=Param("size", "降幅", "pp", -1.75, -0.75, -1.0, FED_STEP),
               writes=[Write("policy_rate_chg", "point", magnitude=True)], gate="fed_emergency",
               bin=(-INF, -0.625), signal=("必要时将采取一切工具稳定金融条件", -1, 2), protocol="lagged",
               note="只在金融事故或疫情事实、上季美股 ≤ −15% 或 Baa 利差 +0.75 之后开放（v1 的 EBP 触发改为利差：v2 删除了 EBP）"),
            _o("cut_50", "降息 50bp", writes=[Write("policy_rate_chg", "point", a=-0.5)], bin=(-0.625, -0.375),
               signal=("政策需要更快回到中性", -1, 2), protocol="lagged"),
            _o("cut_25", "降息 25bp", writes=[Write("policy_rate_chg", "point", a=-0.25)], bin=(-0.375, -0.125),
               signal=("进一步调整取决于数据", -1, 2), protocol="lagged"),
            _o("hold", "按兵不动", status_quo=True, writes=[Write("policy_rate_chg", "point", a=0.0)],
               bin=(-0.125, 0.125), signal=("政策处于观望位置，取决于数据", 0, 1), protocol="lagged"),
            _o("hike_25", "加息 25bp", writes=[Write("policy_rate_chg", "point", a=0.25)], bin=(0.125, 0.375),
               signal=("进一步收紧可能合适", 1, 2), protocol="lagged"),
            _o("hike_50", "加息 50bp", writes=[Write("policy_rate_chg", "point", a=0.5)], bin=(0.375, 0.625),
               signal=("通胀风险要求更快收紧", 1, 2), protocol="lagged"),
            _o("hike_75plus", "大幅加息（≥75bp）", param=Param("size", "升幅", "pp", 0.75, 1.5, 1.0, FED_STEP),
               writes=[Write("policy_rate_chg", "point", magnitude=True)], gate="fed_big_hike", bin=(0.625, INF),
               signal=("将坚决把通胀压回 2%", 1, 2), protocol="lagged",
               note="只在核心通胀 ≥ 4% 或 1 年通胀预期 ≥ 4.5% 且失业率 < 5% 时开放；锚：2022Q2–Q4 +1.25/+1.50/+1.25pp"),
        ), note="零下限：利率变动的下限是 −s_ff（逐粒子）；25bp 取整；菜单规则：覆盖先验 ≥ 90% 的最少档位，最多 4 个"),
        Domain("balance_sheet", "资产负债表（属性，并入利率选项）", "attribute", options=(
            _o("bs_hold", "维持现行资产负债表安排", status_quo=True,
               account=[("fed_bs", "hold")], signal=("资产负债表安排不变", 0, 2)),
            _o("bs_runoff", "恢复缩表", writes=[Write("fed_bs_chg", "normal", a=-0.5, b=0.2)],
               account=[("fed_bs", "runoff")], signal=("将恢复资产负债表缩减", -1, 4), protocol="lagged",
               note="幅度判断值（v1：资产/GDP 每季约 −0.5pp）"),
            _o("bs_reserve_purchases", "准备金管理购买（扩大）", writes=[Write("fed_bs_chg", "normal", a=0.1, b=0.1)],
               account=[("fed_bs", "reserve_purchases")], signal=("将购买国库券以维持充足准备金", 1, 4),
               protocol="lagged", note="近 13 周国库券 +570 亿美元（纽约联储 SOMA）"),
            _o("bs_qe", "量化宽松", writes=[Write("fed_bs_chg", "normal", a=3.0, b=2.0)], gate="fed_qe",
               account=[("fed_bs", "qe")], signal=("将大规模购买资产以支持市场运行", 1, 4), protocol="lagged",
               note="只在零下限或压力事实下开放；锚：2008Q4 资产/GDP +7.2pp、2020Q2 +11.3pp（W_A06）"),
        )),
    ),
    note="最后一个阶段（两次会议，看得到本季所有动作）；10 月 27–28 日会议是锁定事实，本季选项在相容的选项上重新归一",
)

PRESIDENT = Actor(
    id="president", zh="总统与白宫", kind="policy", stage=2, active="always", pool_c=0.7, custom_cap=0.15,
    book="president", watches=("fiscal.customs_q", "households.saving_rate"),
    levers=("tariff_chg", "labor_force_g", "oil_supply_signature", "oil_supply_shock", "oil_cause_label"),
    prior_note="关税体制：2018–19 与 2025–26 的 15 个活跃季度中 10 升 4 降；平常时期 P(|Δ| ≥ 0.5pp) = 11/260（W_A02）；"
               "战略储备 0.028/季（W_A02，待核）",
    domains=(
        Domain("trade", "关税", "main", prior="president_trade", max_options=4, options=(
            _o("tariff_hold", "维持现有关税", status_quo=True, writes=[Write("tariff_chg", "bin", lo=-0.25, hi=0.25)],
               bin=(-0.25, 0.25), signal=("关税维持现行水平", 0, 2), protocol="thin",
               note="维持时有效关税率仍随进口构成与退款波动：取 A0 在 ±0.25pp 内的粒子"),
            _o("tariff_up_small", "行业性加征", param=Param("pp", "有效关税率变动", "pp", 0.25, 1.5, 0.6),
               writes=[Write("tariff_chg", "point", magnitude=True)], bin=(0.25, 1.5),
               signal=("将对特定行业加征关税", 1, 2), protocol="thin"),
            _o("tariff_up_large", "广泛加征", param=Param("pp", "有效关税率变动", "pp", 1.5, 5.4, 2.0),
               writes=[Write("tariff_chg", "point", magnitude=True)], bin=(1.5, INF),
               signal=("将全面提高关税", 1, 2), protocol="thin",
               note="+2pp 在 1985–2019 年是 z ≈ 15，落在 p99（1.49）与历史最大（5.44，2025Q2）之间：罕见，不截断"),
            _o("tariff_down", "降级或豁免", param=Param("pp", "有效关税率变动", "pp", -3.0, -0.25, -1.0),
               writes=[Write("tariff_chg", "point", magnitude=True)], bin=(-INF, -0.25),
               signal=("将下调部分关税", -1, 2), protocol="thin"),
            _o("deal", "达成协议（下调并承诺）", param=Param("pp", "有效关税率变动", "pp", -3.0, -0.25, -1.0),
               writes=[Write("tariff_chg", "point", magnitude=True),
                       Write("tariff_chg", "point", a=0.5, per_unit=True, dq=1, note="协议分两季执行：下季再降一半")],
               facts=[("trade_truce", True)], bin=(-INF, -0.25), signal=("已与主要伙伴达成关税协议", -1, 2),
               protocol="thin"),
        ), note="法律授权菜单按制度现状（122 条款 ≤ 15% 且 ≤ 150 天；301；232；IEEPA）；法院裁定授权无效时换成替代授权菜单"),
        Domain("trade_substitute", "关税（授权被裁定无效后的替代授权）", "main", live="tariff_struck_down",
               prior="uniform", max_options=3, options=(
                   _o("replace_122", "按 122 条款临时替代（≤ 15%，≤ 150 天）",
                      param=Param("pp", "有效关税率变动", "pp", -3.0, 0.0, -1.0),
                      writes=[Write("tariff_chg", "point", magnitude=True)], signal=("将以其他授权维持关税", 0, 2),
                      protocol="thin", note="替代后的净变动是判断值：取决于被裁无效的部分"),
                   _o("replace_301_232", "改用 301/232 条款", param=Param("pp", "有效关税率变动", "pp", -2.0, 1.0, -0.5),
                      writes=[Write("tariff_chg", "point", magnitude=True)], signal=("将改用行业与贸易法条款", 0, 2),
                      protocol="thin"),
                   _o("accept_lapse", "接受失效", status_quo=True, param=Param("pp", "有效关税率变动", "pp", -5.0, -1.0, -3.0),
                      writes=[Write("tariff_chg", "point", magnitude=True)], signal=("遵守裁决", -1, 2), protocol="thin"),
               ), note="只有一例：没有历史基率（无历史先例，均匀先验），LLM 给条件表"),
        Domain("export_controls", "出口管制与芯片（属性；有待决事项时可分叉）", "attribute",
               branch_when="出口管制或芯片关税有待决事项时", options=(
                   _o("ec_hold", "维持", status_quo=True, signal=("出口管制维持现状", 0, 2)),
                   _o("ec_tighten", "收紧", facts=[("export_controls_level", +1)],
                      signal=("将收紧对华先进芯片出口", 1, 2),
                      note="通道缺失：合成器可在 x_supply_chain、d_capex 写机制"),
                   _o("ec_ease", "放松", facts=[("export_controls_level", -1)], signal=("将放宽部分出口许可", -1, 2)),
               )),
        Domain("energy", "能源（属性；供给中断时可分叉）", "attribute", branch_when="石油供给中断持续时", options=(
            _o("spr_hold", "不动用战略储备", status_quo=True, signal=("战略储备不动用", 0, 1)),
            _o("spr_release", "动用战略石油储备", param=Param("mbd", "释放量", "百万桶/日", 0.5, 1.0, 0.75),
               oil_z=(-0.32, 0.10, True), account=[("spr_release_mbd", "magnitude")], gate="spr_release",
               facts=[("spr_release", True)], signal=("将释放战略石油储备", -1, 1), protocol="signature",
               note="−0.32 z / 百万桶日（判断值：+3.5% 油价 / 1% 世界供给 ÷ d01 的 10.9 log% / z）"),
        )),
        Domain("immigration", "移民执法（属性）", "attribute", options=(
            _o("imm_hold", "维持", status_quo=True, signal=("移民执法维持现状", 0, 2)),
            _o("imm_tighten", "收紧", param=Param("pp", "劳动力增速变动", "pp 年化", -1.5, -0.2, -0.5),
               writes=[Write("labor_force_g", "shift", magnitude=True, b=0.5, per_unit=False)],
               facts=[("imm_tightening", True)], signal=("将加强移民执法", -1, 4), protocol="thin",
               note="劳动供给签名：失业率在 0.25 个标准差内（模型补全后检查）；幅度判断值"),
        )),
        Domain("fed_pressure", "对美联储（属性）", "attribute", options=(
            _o("pressure_none", "不额外施压", status_quo=True, signal=("尊重美联储的决定", 0, 1)),
            _o("pressure_public", "公开施压", political=[("independence_risk", 1)], facts=[("fed_pressure_public", True)],
               signal=("利率应该更低", -1, 1)),
            _o("nominate_dovish", "提名偏鸽人选", political=[("chair_tilt_pending", -0.5)],
               facts=[("dovish_nominee", True)], signal=("将提名支持降息的人选", -1, 2),
               note="主席倾向在确认后 1–2 季生效（v1 规则）"),
            _o("attempt_removal", "试图罢免理事", political=[("independence_risk", 2)], gate="attempt_removal",
               facts=[("fed_removal_attempt", True)], signal=("将撤换美联储理事", -1, 1),
               note="法院裁定维持免职保护时关闭"),
        )),
    ),
    naming="总统与白宫（作为机构，不模仿任何人的语气）",
)

CONGRESS = Actor(
    id="congress", zh="国会", kind="policy", stage=2, active="always", pool_c=0.6, book="fiscal",
    watches=("households.real_dpi_pc_yoy",),
    levers=("transfers_chg", "pers_tax_chg", "corp_tax_chg", "defense_chg", "deficit_chg"),
    prior_note="萨姆规则触发后 4 季内刺激 4/9（0.44，Jeffreys 0.17–0.75）；停摆每个截止年 0.19；78 次上限行动 0 次违约；"
               "2011 年以来 10/10 次约束期解决、中位 136 天（W_A02，A11）",
    domains=(
        Domain("fiscal", "立法（最终成为法律的结果）", "main", prior="congress_fiscal", max_options=4, options=(
            _o("no_new_law", "无新立法（只有自动稳定器）", status_quo=True, signal=("没有新的财政立法", 0, 2)),
            _o("stimulus_small", "小规模刺激（退税/直接支付）",
               param=Param("size", "规模", "% GDP", 0.3, 1.5, 0.8),
               writes=[Write("transfers_chg", "shift", a=2.5, b=1.25, per_unit=True),
                       Write("transfers_chg", "shift", a=-2.5, b=1.25, per_unit=True, dq=1, note="一次性支付下季回落")],
               gate="stimulus_small", signal=("将通过针对家庭的直接支付", 1, 1), protocol="response_table",
               note="规模 0.8% GDP → 转移支付/个人收入 +2.0 ± 1.0pp，下季回落（v1；2008 年支付约 1% GDP）"),
            _o("stimulus_large", "大规模刺激（分 4–6 季支出）", param=Param("size", "规模", "% GDP", 2.0, 10.0, 4.0),
               writes=[Write("transfers_chg", "shift", a=0.5, b=0.375, per_unit=True),
                       Write("deficit_chg", "shift", a=0.625, b=0.25, per_unit=True, note="转移支付以外的剩余支出"),
                       Write("deficit_chg", "shift", a=0.25, b=0.2, per_unit=True, dq=1),
                       Write("deficit_chg", "shift", a=-0.25, b=0.2, per_unit=True, dq=3),
                       Write("deficit_chg", "shift", a=-0.375, b=0.25, per_unit=True, dq=5)],
               gate="stimulus_large", signal=("将通过大规模经济救助法案", 1, 4), protocol="response_table",
               flags=("规模超过 6% GDP 时超出可测历史（只有 2020–21）",),
               note="锚：2009Q1–Q2 赤字/GDP 4.8 → 7.6 → 9.1；2020Q2 4.9 → 26.8（W_A06）；分配比例判断值"),
            _o("procyclical_tax_cut", "顺周期减税（无衰退）", param=Param("size", "规模", "% GDP/年", 0.3, 2.0, 1.0),
               writes=[Write("pers_tax_chg", "shift", a=-0.65, b=0.25, per_unit=True),
                       Write("pers_tax_chg", "shift", a=-0.65, b=0.25, per_unit=True, dq=1)],
               gate="unified_government", signal=("将延长并扩大减税", 1, 4), protocol="response_table",
               note="ERTA 1981、EGTRRA 2001、JGTRRA 2003、TCJA 2017、OBBBA 2025；分两季生效（判断值）"),
            _o("consolidation", "财政整顿", param=Param("size", "规模", "% GDP/年", -1.5, -0.3, -0.7),
               writes=[Write("deficit_chg", "shift", a=0.3, b=0.15, per_unit=True)]
               + [Write("deficit_chg", "shift", a=0.3, b=0.15, per_unit=True, dq=k) for k in (1, 2, 3)],
               signal=("将削减赤字", -1, 4), protocol="response_table",
               note="OBRA 1990/1993、BBA 1997、BCA 2011、ATRA 2013、FRA 2023（36 年 6 部）"),
            _o("debt_limit_raise", "提高债务上限", param=Param("bn", "提高额", "十亿美元", 500.0, 6000.0, 2500.0),
               account=[("debt_limit_raise_bn", "magnitude")], clears=("debt_limit_binding", "debt_limit_standoff"),
               gate="debt_limit_live", signal=("将提高债务上限，避免违约", 0, 1),
               note="基率：2011 年以来 10/10 次约束期解决，中位 136 天，0 次违约（A11）"),
            _o("debt_limit_suspend", "暂停债务上限", param=Param("q", "暂停季数", "季", 2.0, 12.0, 8.0, 1.0),
               account=[("debt_limit_suspend_q", "magnitude")], clears=("debt_limit_binding", "debt_limit_standoff"),
               facts=[("debt_limit_suspended", True)], gate="debt_limit_live", signal=("将暂停债务上限", 0, 1)),
            _o("debt_limit_standoff", "债务上限对峙（未解决）", facts=[("debt_limit_standoff", True)],
               gate="debt_limit_live", signal=("不会无条件提高上限", 0, 1),
               note="只写事实；不确定性与风险偏好的后果由合成器写"),
        ), note="选项是最终成为法律的结果（签署或否决已含在内）；赤字由模型按恒等式推断（SCHEMA 规则 2）"),
        Domain("deadline", "拨款截止（属性；截止季度）", "attribute", live="deadline_quarter", options=(
            _o("cr_on_time", "按时拨款或续命", status_quo=True, clears=("shutdown",), signal=("政府将正常运转", 0, 1)),
            _o("shutdown", "政府停摆", param=Param("days", "停摆天数", "天", 1.0, 60.0, 21.0, 1.0),
               facts=[("shutdown", "magnitude")], account=[("shutdown_days", "magnitude")],
               signal=("拨款谈判陷入僵局", 0, 1),
               note="CBO：2018–19 年 35 天停摆使实际 GDP 少 30 亿（2018Q4）与 80 亿（2019Q1），其中 30 亿永久损失"),
        )),
    ),
)

TREASURY = Actor(
    id="treasury", zh="财政部", kind="policy", stage=2, active="bill_share_field", pool_c=0.5, book="fiscal",
    levers=("bill_share_chg",), prior_note="TBAC 长期建议短债占比约 15–20%",
    domains=(Domain("issuance", "国债发行结构（季度再融资）", "main", prior="treasury", options=(
        _o("refunding_unchanged", "发行结构不变", status_quo=True, signal=("发行计划不变", 0, 2)),
        _o("bills_up", "提高短债占比", writes=[Write("bill_share_chg", "normal", a=1.0, b=0.5)],
           account=[("bill_share_chg", 1.0)], signal=("将更多依靠短期国库券融资", 1, 2)),
        _o("bills_down", "拉长发行期限", writes=[Write("bill_share_chg", "normal", a=-1.0, b=0.5)],
           account=[("bill_share_chg", -1.0)], signal=("将增加中长期国债发行", -1, 2)),
        _o("buybacks_up", "扩大回购", writes=[Write("bill_share_chg", "normal", a=0.3, b=0.3)],
           signal=("将扩大国债回购", 0, 2)),
    )),),
    note="条件参与者：输入表有短债占比字段时开启（schema v2/v3 没有：关闭）；TGA 与非常规措施仍是代码规则",
)

CHINA = Actor(
    id="china", zh="中国", kind="policy", stage=3, active="always", pool_c=0.6, custom_cap=0.15, book="china",
    watches=("fiscal.customs_q",), levers=("gpr_chg", "cn_tariff_chg", "cny_logchg", "cn_credit_impulse"),
    prior_note="2018–25 年 8/8 次对美方升级报复（Jeffreys 0.73–1.0）；刺激 0.089/季；有管理的贬值 0.10/季；"
               "台海危机 0.013/季，封锁 Jeffreys 均值 0.0016/季（W_A03、W_A07）",
    domains=(
        Domain("trade_response", "对美贸易回应（跟随总统·关税）", "main", prior="china_trade", options=(
            _o("no_response", "不回应", status_quo=True, signal=("中方保持克制", 0, 1)),
            _o("retaliate_tariffs", "对等加征关税", facts=[("cn_retaliation", True)], clears=("trade_truce",),
               signal=("将采取对等反制措施", 1, 1), note="中国对美关税字段缺失：只写事实（后果经由合成器）"),
            _o("retaliate_export_controls", "加征关税并收紧出口管制（稀土等）",
               facts=[("cn_retaliation", True), ("export_controls_cn", +1)], clears=("trade_truce",),
               signal=("将限制关键矿产出口", 1, 2)),
            _o("negotiate_truce", "谈判或休战", facts=[("trade_truce", True)], clears=("cn_retaliation",),
               signal=("愿意通过谈判解决分歧", -1, 2)),
        )),
        Domain("domestic", "国内宏观政策（属性）", "attribute", options=(
            _o("targeted_support", "定向支持（现状）", status_quo=True, signal=("政策保持连续稳定", 0, 2)),
            _o("major_stimulus", "大规模刺激（财政/信贷）", facts=[("cn_stimulus", True)],
               account=[("cn_stimulus", True)], signal=("将加大逆周期调节力度", 1, 2),
               note="信贷脉冲字段缺失：只写事实；不再用金属与新兴市场美元价格作后备（ACTORS_V2 §4.6）"),
        )),
        Domain("fx", "人民币（属性）", "attribute", options=(
            _o("fx_hold", "汇率稳定", status_quo=True, signal=("保持人民币汇率基本稳定", 0, 2)),
            _o("cny_weaker", "有管理的贬值", facts=[("cny_weaker", True)], account=[("cny_weaker", True)],
               signal=("汇率弹性将增强", 1, 2)),
        )),
        Domain("taiwan", "台海（属性；演习概率 ≥ 0.10 时分叉）", "attribute", branch_when="大规模演习概率 ≥ 0.10",
               options=(
                   _o("tw_status_quo", "现状", status_quo=True, signal=("维护台海和平稳定", 0, 2)),
                   _o("tw_large_exercise", "大规模军演", gpr_sd=(1.0, 0.4), facts=[("taiwan_exercise", True)],
                      signal=("将举行例行军事演习", 1, 1), note="GPR +1 sd（判断值）"),
                   _o("tw_blockade", "封锁（尾部情景）", gpr_sd=(3.0, 1.0), facts=[("taiwan_blockade", True)],
                      gate="stress_mode", signal=("将采取一切必要措施", 1, 1),
                      flags=("超出可测历史：GPR +3 sd 超过 p99，需要 beyond_history 理由",),
                      note="默认作粒子混合；只有打开压力模式才显式分叉（U-7）"),
               )),
    ),
    naming="中国中央政府与人民银行（作为机构）",
)

OPEC = Actor(
    id="opec", zh="欧佩克+", kind="policy", stage=3, active="always", pool_c=0.5, book="opec",
    watches=("opec.us_crude_production", "president.spr_inventory"),
    levers=("opec_supply_chg", "oil_supply_signature", "oil_supply_shock", "oil_cause_label"),
    prior_note="油价崩跌后 2 季内减产 5/8（0.625，Jeffreys 0.30–0.88）；飙升或高于区间：增产 0.30 / 维持 0.55 / 减产 0.15；"
               "区间内维持 0.70（W_A03；区间内一行是判断值）",
    domains=(Domain("quota", "产量配额", "main", prior="opec_quota", options=(
        _o("quota_hold", "维持配额", status_quo=True, signal=("将维持现行产量政策", 0, 1)),
        _o("quota_cut", "减产", param=Param("mbd", "减产量", "百万桶/日", 0.5, 2.0, 1.0),
           oil_z=(0.32, 0.10, True), account=[("opec_quota_chg_mbd", "-magnitude")], facts=[("opec_cut", True)],
           signal=("将采取措施稳定市场", -1, 2), protocol="signature", note="+0.32 z / 百万桶日（判断值）"),
        _o("quota_raise", "增产", param=Param("mbd", "增产量", "百万桶/日", 0.5, 2.0, 0.8),
           oil_z=(-0.32, 0.10, True), account=[("opec_quota_chg_mbd", "magnitude")], gate="quota_raise",
           facts=[("opec_raise", True)], signal=("将逐步增加供应", 1, 2), protocol="signature",
           note="受可达市场的闲置产能约束（私人锁定事实；被封锁海峡后的产能不算）"),
    )),),
    naming="欧佩克+（以沙特为摆动产油国，作为机构）",
)

BIGTECH = Actor(
    id="bigtech", zh="超大型科技公司", kind="corporate", stage=3, active="always", pool_c=0.6, book="bigtech",
    watches=("households.saving_rate",), levers=("hyperscaler_capex_g", "it_invest_g", "capex_g"),
    naming="大型科技公司管理层（CEO、CFO）作为整体：微软、Alphabet、亚马逊、Meta、甲骨文；不用任何在世人物的姓名，不模仿个人语气",
    prior_note="体制先验（判断值，锚定 NIPA：纳指 4 季跌 20% 后下季实际 IT 投资下降 56%，16 次，Jeffreys 0.33–0.78；"
               "其他时候 6%，W_C02）",
    domains=(
        Domain("capex", "资本开支指引（每季财报）", "main", prior="bigtech_capex", options=(
            _o("capex_raise_large", "大幅上调", param=Param("x_pct", "未来 4 季计划变动", "%", 5.0, 40.0, 15.0),
               writes=[Write("hyperscaler_capex_g", "point", magnitude=True)],
               account=[("bt_capex_plan_pct", "magnitude")], signal=("将继续扩大 AI 基础设施投资", 1, 4),
               protocol="thin", note="后备：d_itinv 增量 0.16·x pp（年化），作为 4 季承诺（s_bt 0.16 判断值）"),
            _o("capex_hold", "维持或小幅调整", status_quo=True, param=Param("x_pct", "未来 4 季计划变动", "%", -5.0, 5.0, 0.0),
               account=[("bt_capex_plan_pct", "magnitude")], signal=("资本开支按既定计划执行", 0, 4), protocol="thin"),
            _o("capex_cut", "下调（消化）", param=Param("x_pct", "未来 4 季计划变动", "%", -30.0, -5.0, -15.0),
               writes=[Write("hyperscaler_capex_g", "point", magnitude=True)],
               account=[("bt_capex_plan_pct", "magnitude")], signal=("将放缓部分数据中心建设", -1, 4), protocol="thin"),
            _o("capex_slash", "大幅削减（2001–02 型）", param=Param("x_pct", "未来 4 季计划变动", "%", -60.0, -30.0, -35.0),
               writes=[Write("hyperscaler_capex_g", "point", magnitude=True)],
               account=[("bt_capex_plan_pct", "magnitude")], signal=("将大幅削减资本开支以保护现金流", -1, 4),
               protocol="thin", note="打开合成器金融条件轴的收紧区（倾斜 ×1.5）"),
        ), note="字段 hyperscaler_capex_g 不存在时：d_itinv 写 0.16·x 的增量，持续 4 季（承诺，可被下一季指引修改）"),
        Domain("workforce", "用工（属性）", "attribute", options=(
            _o("hire_hold", "维持", status_quo=True, signal=("招聘按计划进行", 0, 2)),
            _o("hire_expand", "扩招", facts=[("bigtech_hiring", +1)], signal=("将继续招聘 AI 人才", 1, 2)),
            _o("hire_freeze_ai", "冻结招聘（AI 替代白领）", facts=[("bigtech_hiring", -1)], signal=("将以 AI 提升效率", -1, 2)),
            _o("layoffs", "裁员潮", facts=[("bigtech_layoffs", True)], signal=("将精简组织", -1, 1),
               note="信息业就业只占非农 1.72%：一次裁员潮约 0.1 个季度标准差，不分叉"),
        )),
        Domain("payout", "股东回报（属性）", "attribute", options=(
            _o("buyback_hold", "维持", status_quo=True, account=[("bt_buyback_mult", 1.0)], signal=("股东回报政策不变", 0, 2)),
            _o("buyback_up", "加速回购", account=[("bt_buyback_mult", 1.5)], signal=("将增加股票回购", 1, 2)),
            _o("buyback_cut", "削减回购为资本开支融资", account=[("bt_buyback_mult", 0.5)], signal=("将优先投资而非回购", -1, 2)),
        )),
        Domain("pricing", "定价与变现（属性）", "attribute", options=(
            _o("price_hold", "维持", status_quo=True, signal=("定价策略不变", 0, 2)),
            _o("price_war", "AI 价格战", facts=[("ai_price_war", True)], signal=("将降低 AI 服务价格", -1, 2)),
            _o("monetize", "提价变现", facts=[("ai_monetize", True)], signal=("将推出付费 AI 服务", 1, 2),
               note="只倾斜下一季资本开支先验（变现顺利 → 上调更可能）"),
        )),
    ),
)

BELLIGERENTS = Actor(
    id="belligerents", zh="冲突方", kind="state", stage=1, active="oil_disruption", pool_c=0.4, book="belligerents",
    levers=("oil_supply_signature", "oil_supply_shock", "oil_cause_label", "gpr_chg"),
    prior_note="退出（停火/航道恢复）0.42/季，Jeffreys 0.26–0.59：实际油价同比 > 40% 的持续期 31 季中 13 次结束（W_A06）；"
               "升级 0.15、僵持 0.43（判断值）",
    domains=(Domain("conflict", "冲突走向", "main", prior="belligerents", options=(
        _o("escalate", "升级", oil_z=(1.5, 0.75, False), gpr_sd=(0.5, 0.25), facts=[("escalation", True)],
           signal=("冲突可能进一步扩大", 1, 1), protocol="signature", note="+1.5 z 与 GPR +0.5 sd（判断值）"),
        _o("stalemate", "僵持", status_quo=True, signal=("局势僵持", 0, 1), protocol="signature"),
        _o("ceasefire", "停火或航道恢复", oil_z=(-1.5, 0.75, False), gpr_sd=(-0.5, 0.25), facts=[("ceasefire", True)],
           clears=("oil_disruption_ongoing",), signal=("同意停火，恢复航运", -1, 1), protocol="signature"),
    )),),
    naming="冲突方（身份由用户锁定；未锁定时不具名）",
    note="状态触发：持续事实“武装冲突导致的石油供给中断”成立时开启；开启时自然对同一中断的退出风险率停用",
)

COURTS = Actor(
    id="courts", zh="法院", kind="calendar", stage=0, active="pending_ruling", pool_c=0.3, book="courts",
    prior_note="用户或市场给的概率；没有时 0.5，标“无可靠基率”",
    domains=(
        Domain("tariff_authority", "关税授权（IEEPA）", "main", live="ruling:tariff_authority", prior="courts", options=(
            _o("uphold", "维持授权", status_quo=True, facts=[("tariff_authority", "broad")], signal=("", 0, 1)),
            _o("strike_down", "授权无效（不退税）", facts=[("tariff_authority", "struck_down")], signal=("", 0, 1),
               note="同一季重新生成总统的关税菜单（替代授权）"),
            _o("strike_down_refunds", "授权无效并退税", facts=[("tariff_authority", "struck_down_refunds")],
               account=[("customs_refunds_bn", "pinned")], signal=("", 0, 1), note="退税进入联邦账户（一次性事件）"),
        )),
        Domain("fed_removal", "美联储理事免职保护", "main", live="ruling:fed_removal", prior="courts", options=(
            _o("uphold_protection", "维持免职保护", status_quo=True, facts=[("fed_removal_protected", True)],
               political=[("independence_risk", 0)], signal=("", 0, 1)),
            _o("allow_removal", "允许罢免", facts=[("fed_removal_allowed", True)], political=[("independence_risk", 2)],
               signal=("", 0, 1), note="打开合成器的独立性机制（d_infexp_lr、d_infexp1；通道缺失）"),
        )),
        Domain("antitrust", "反垄断救济", "main", live="ruling:antitrust", prior="courts", options=(
            _o("dismiss_or_behavioral", "驳回或行为性救济", status_quo=True, signal=("", 0, 1)),
            _o("structural", "结构性救济（拆分/剥离）", facts=[("antitrust_structural", True)], signal=("", 0, 1),
               note="下一季超大型科技公司削减回购的先验 ×1.5（判断值）"),
        )),
    ),
    note="日历型：制度现状里有预计本季判决的案件时开启；已判决的案件只能作为锁定事实输入（LLM 知道结果）",
)

VOTER = Actor(
    id="voter", zh="选民", kind="calendar", stage=0, active="election_quarter", pool_c=0.2, book="voter",
    prior_note="总统所在党在其掌握众议院的 12 次中期选举中失去众议院 7 次，1994 年以来 6 次中 5 次（v1，待核）；"
               "有市场赔率时用赔率且 c = 0",
    domains=(
        Domain("midterm", "中期选举（众参两院合并）", "main", live="election:midterm", prior="voter_midterm", options=(
            _o("house_holds", "执政党保住众议院", status_quo=True, signal=("", 0, 1)),
            _o("house_flips", "执政党失去众议院", political=[("house_next", "flip")], signal=("", 0, 1),
               note="新国会 2027-01-03 起改变国会的门槛与倾斜（下一季生效）"),
        )),
        Domain("presidential", "总统选举", "main", live="election:presidential", prior="voter_presidential", options=(
            _o("incumbent_party_wins", "执政党赢得白宫", status_quo=True, signal=("", 0, 1)),
            _o("opposition_wins", "在野党赢得白宫", political=[("pres_next", "flip")], signal=("", 0, 1)),
        )),
    ),
    note="只在联邦选举季度（2026Q4、2028Q4、2030Q4）；结果已锁定时不再分叉",
)

ACTORS: dict[str, Actor] = {a.id: a for a in (VOTER, COURTS, BELLIGERENTS, PRESIDENT, CONGRESS, TREASURY, CHINA,
                                               OPEC, BIGTECH, FED)}
STAGE_ORDER = sorted(ACTORS, key=lambda k: (ACTORS[k].stage, list(ACTORS).index(k)))

# Nature is code, not an LLM actor: hazards partition the model's own event tails (§6.5)
HAZARDS: dict[str, dict] = {
    "oil_disruption": {"zh": "石油供给中断（战争/袭击/禁运）", "role": "oil_supply_signature", "region": ("le", -2.0),
                       "fact": "oil_disruption_ongoing", "exit": 0.42, "oil_z": (4.0, 1.5),
                       "note": "A0 的 d_metals_oil 条件 z ≤ −2 是模型自己的尾部质量；新的大中断 +4 z（b 1.5；v1）"},
    "gpr_episode": {"zh": "非石油地缘冲击（GPR 进入 p95 以上）", "role": "gpr_chg", "region": ("ge", 2.0),
                    "fact": "gpr_episode", "exit": 0.5},
    "financial_accident": {"zh": "金融事故（银行/非银挤兑）", "role": None, "field_hint": "d_gz", "region": ("ge", 2.5),
                           "fact": "financial_accident", "exit": 0.5},
    "pandemic": {"zh": "严重疫情", "role": None, "region": None, "fact": "pandemic", "exit": 0.5,
                 "note": "没有通道：只作为锁定事实或压力模式里的事实，打开体制机制与美联储的紧急选项"},
}
NATURE_POOL_C = 0.2
HAZARD_TILT_BOUNDS = (0.5, 3.0)        # the LLM may move a hazard to [m/2, 3m] with a state-based reason (v1)
HAZARD_BRANCH_MIN = 0.05               # below 5% (and stress mode off) a hazard stays a particle mixture

# =====================================================================================================================
# 4. Facts, political state and the calendar
# =====================================================================================================================

FACTS: dict[str, dict] = {
    "oil_disruption_ongoing": {"zh": "武装冲突导致的石油供给中断持续", "exit": 0.42, "src": "W_A06"},
    "financial_accident": {"zh": "金融事故（挤兑/系统性压力）", "exit": 0.5},
    "pandemic": {"zh": "严重疫情", "exit": 0.5},
    "gpr_episode": {"zh": "地缘政治风险处于高位", "exit": 0.5},
    "trade_truce": {"zh": "中美贸易休战", "exit": None},
    "cn_retaliation": {"zh": "中国对美报复性关税", "exit": 0.25},
    "export_controls_level": {"zh": "对华芯片出口管制等级（相对起点）", "exit": None, "level": True},
    "export_controls_cn": {"zh": "中国关键矿产出口管制等级（相对起点）", "exit": None, "level": True},
    "cn_stimulus": {"zh": "中国大规模刺激", "exit": 0.5},
    "cny_weaker": {"zh": "人民币有管理的贬值", "exit": 0.5},
    "taiwan_exercise": {"zh": "台海大规模军演", "exit": 0.9},
    "taiwan_blockade": {"zh": "台海封锁", "exit": 0.3},
    "tariff_authority": {"zh": "关税授权状态", "exit": None, "values": ("broad", "struck_down", "struck_down_refunds")},
    "fed_removal_protected": {"zh": "美联储理事免职保护得到维持", "exit": None},
    "fed_removal_allowed": {"zh": "法院允许罢免美联储理事", "exit": None},
    "fed_removal_attempt": {"zh": "总统试图罢免美联储理事", "exit": 0.5},
    "fed_pressure_public": {"zh": "总统公开施压美联储", "exit": 0.7},
    "dovish_nominee": {"zh": "偏鸽人选待确认", "exit": 0.5},
    "antitrust_structural": {"zh": "反垄断结构性救济", "exit": None},
    "shutdown": {"zh": "联邦政府停摆（天）", "exit": 0.9},
    "debt_limit_binding": {"zh": "债务上限约束（非常规措施中）", "exit": None, "src": "accounts"},
    "debt_limit_standoff": {"zh": "债务上限对峙（X 日期临近）", "exit": None},
    "debt_limit_suspended": {"zh": "债务上限暂停", "exit": None},
    "xdate_passed": {"zh": "越过 X 日期（逾期未付）", "exit": None, "src": "accounts"},
    "imm_tightening": {"zh": "移民执法收紧", "exit": 0.2},
    "spr_release": {"zh": "战略石油储备释放中", "exit": 0.8},
    "opec_cut": {"zh": "欧佩克+ 减产", "exit": 0.5},
    "opec_raise": {"zh": "欧佩克+ 增产", "exit": 0.5},
    "escalation": {"zh": "冲突升级", "exit": 0.7},
    "ceasefire": {"zh": "停火或航道恢复", "exit": 0.3},
    "bigtech_hiring": {"zh": "大型科技公司用工方向", "exit": 0.5, "level": True},
    "bigtech_layoffs": {"zh": "大型科技公司裁员潮", "exit": 0.7},
    "ai_price_war": {"zh": "AI 价格战", "exit": 0.5},
    "ai_monetize": {"zh": "AI 提价变现", "exit": 0.5},
    "new_congress": {"zh": "新一届国会就职", "exit": 1.0},
}

POLITICAL_DEFAULT: dict[str, object] = {
    "pres_party": "R", "house": "R", "senate": "R", "house_margin": None, "senate_seats": "53-47",
    "chair_tilt": 0.0, "independence_risk": 0, "trade_regime": "active", "tariff_authority": "broad",
}

ELECTIONS = {"2026Q4": {"date": "2026-11-03", "kind": "midterm", "zh": "11 月 3 日中期选举"},
             "2028Q4": {"date": "2028-11-07", "kind": "presidential", "zh": "11 月 7 日总统与国会选举"},
             "2030Q4": {"date": "2030-11-05", "kind": "midterm", "zh": "11 月 5 日中期选举"}}
FOMC_PINNED = {"2026Q4": {"id": "fomc_2026_10", "date": "2026-10-28", "zh": "10 月 27–28 日议息会议"}}
EARNINGS = "1 月底、4 月底、7 月底、10 月底（每季财报与资本开支指引）"

_Q_RE = re.compile(r"^(\d{4})Q([1-4])$")


def quarter_add(q: str, k: int) -> str:
    m = _Q_RE.match(str(q))
    if not m:
        raise ValueError(f"季度格式必须是 YYYYQn：{q}")
    n = int(m.group(1)) * 4 + int(m.group(2)) - 1 + int(k)
    return f"{n // 4}Q{n % 4 + 1}"


def quarter_of(date: str | dt.date) -> str:
    d = dt.date.fromisoformat(str(date)[:10]) if not isinstance(date, dt.date) else date
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def quarters_between(a: str, b: str) -> int:
    ma, mb = _Q_RE.match(a), _Q_RE.match(b)
    if not (ma and mb):
        raise ValueError(f"季度格式必须是 YYYYQn：{a}、{b}")
    return (int(mb.group(1)) * 4 + int(mb.group(2))) - (int(ma.group(1)) * 4 + int(ma.group(2)))


# =====================================================================================================================
# 5. Pinned facts (制度现状): defaults marked 未确认, the user's edits in data/agents/institutions.json
# =====================================================================================================================

def _item(iid, zh, typ, value, *, unit="", options=None, source="", date="2026-11-01", affects=(), visibility="public",
          required=True, why="", suggested=None):
    return {"id": iid, "zh": zh, "type": typ, "value": value, "unit": unit, "options": list(options or []),
            "status": "unconfirmed", "source": source, "date": date, "affects": list(affects),
            "visibility": visibility, "required": required, "default_reason": why,
            "suggested": suggested or {"value": value, "why": why}}


INSTITUTION_DEFAULTS: list[dict] = [
    _item("fomc_2026_10", "10 月 27–28 日议息会议结果", "enum", "hold", options=["hold", "cut_25", "cut_50", "hike_25"],
          source="FRED IORB、RRPONTSYAWARD、DFF", date="2026-10-30", affects=["fed"],
          why="由数据推断为维持：会议前后准备金利率都是 3.90%，隔夜逆回购利率 3.75%，有效联邦基金利率 3.88%（10-29），"
              "目标区间 3.75–4.00%"),
    _item("fed_chair", "美联储主席与待确认的理事席位", "text", "按数据日期的现任；主席倾向 0", affects=["fed", "president"],
          why="未锁定：按数据日期的人事，主席倾向取 0"),
    _item("fed_chair_tilt", "主席倾向（−1 偏鸽 … +1 偏鹰）", "number", 0.0, unit="", affects=["fed"], required=False,
          why="未锁定时取 0（每单位使降息对加息的对数几率移动 0.5，判断值）"),
    _item("ieepa_status", "IEEPA 关税的法律状态", "enum", "in_force",
          options=["in_force", "pending", "struck_down", "struck_down_refunds"], affects=["president", "courts"],
          source="最高法院案卷（用户锁定）", why="未锁定：按仍然有效处理"),
    _item("refunds_cause", "关税退款的法律原因", "text", "未确认", affects=["courts", "congress"],
          source="MTS 表 4（2026 年 5–9 月海关退款合计 1,344 亿美元）",
          why="2026 年 5–9 月退款 1,344 亿美元；原因需用户锁定"),
    _item("refunds_remaining_bn", "剩余的关税退款义务", "number", 0.0, unit="十亿美元", affects=["courts", "congress"],
          source="MTS 表 4", why="未锁定时取 0（已付 1,344 亿）"),
    _item("china_truce_expiry", "中美一年期暂停措施到期日", "date", "2026-11-10", affects=["china", "president"],
          why="约 2026-11-10 到期（2025 年 10 月底的约定，待核）：按 2026Q4 到期处理"),
    _item("opec_decision", "欧佩克+ 最近一次产量决定", "text", "按最近一次公开声明", affects=["opec"],
          source="OPEC 月报（对脚本返回 403，需手动）", why="未锁定：配额按最近的公开声明"),
    _item("opec_quota_mbd", "欧佩克+ 配额（八国自愿减产组合计）", "number", None, unit="百万桶/日", affects=["opec"],
          required=False, why="未锁定：账户不计算石油收入"),
    _item("opec_spare_mbd", "可达市场的闲置产能（仅欧佩克+ 本人可见）", "number", None, unit="百万桶/日", affects=["opec"],
          visibility="opec", source="EIA STEO", why="未锁定：没有超出 EIA STEO 的可达闲置产能；增产上限按选项范围并标注"),
    _item("saudi_breakeven", "沙特财政盈亏平衡油价", "number", None, unit="美元/桶", affects=["opec"], required=False,
          source="IMF REO 中东中亚", why="未锁定：盈亏平衡门（G-L8）只显示，不触发"),
    _item("belligerents_active", "2026 年石油供给中断：冲突方开启", "bool", True, affects=["belligerents", "opec", "president"],
          source="数据：WTI 10 月底 103.4、布伦特 10-27 123；科威特与伊拉克 2026 年增长 −16.7% / −16.0%（WEO）",
          why="数据显示中断仍在持续：冲突方开启"),
    _item("belligerents_identity", "冲突方身份、航道与设施状态", "text", "不具名（中断持续）", affects=["belligerents"],
          why="未锁定：不具名"),
    _item("house_flip_odds", "11 月 3 日选举：执政党失去众议院的市场概率", "number", None, unit="概率",
          affects=["voter"], source="预测市场（用户输入）", why="没有赔率时用历史基率并标注"),
    _item("senate_flip_odds", "11 月 3 日选举：执政党失去参议院的市场概率", "number", None, unit="概率",
          affects=["voter"], required=False, why="没有赔率时用历史基率并标注"),
    _item("house_margin", "现任众议院多数党席位差", "number", None, unit="席", affects=["voter", "congress"], required=False,
          why="未锁定"),
    _item("election_2026_result", "2026-11-03 选举结果（揭晓后锁定）", "enum", "unknown",
          options=["unknown", "house_holds", "house_flips"], affects=["voter", "congress"],
          why="结果揭晓前为未知：选民在 2026Q4 分叉"),
    _item("cr_expiry", "FY2027 拨款或续命（CR）到期日", "date", None, affects=["congress"],
          why="未锁定：截止季度按财年起点（每年 Q4）"),
    _item("debt_limit_xdate", "财政部估计的 X 日期", "date", None, affects=["congress", "treasury", "fed"],
          why="未锁定：由账户按非常规措施容量与 TGA 推算"),
    _item("em_capacity_bn", "非常规措施容量", "number", 350.0, unit="十亿美元", affects=["congress", "treasury"],
          source="判断值（LEDGERS）", why="判断值 3,500 亿美元：请按财政部致国会的信锁定（U-15）"),
    _item("tga_target_bn", "季末 TGA 目标", "number", 850.0, unit="十亿美元", affects=["congress", "treasury", "fed"],
          source="判断值（LEDGERS）", why="判断值 8,500 亿美元：请按季度再融资声明锁定（U-15）"),
    _item("hyperscaler_guidance", "超大型科技公司 10 月底的三季度财报与资本开支指引", "text",
          "按 2026 年 4 月的计划（2026 年约 7,000–7,250 亿美元，二手来源）", affects=["bigtech"],
          why="本轮没有检索到 10 月底的更新"),
    _item("hyperscaler_capex_2026_bn", "2026 年五家资本开支计划合计", "number", 712.0, unit="十亿美元", affects=["bigtech"],
          required=False, source="Yahoo Finance、24/7 Wall St 2026 年 4–5 月报道（待核）", why="区间 7,000–7,250 亿的中点"),
    _item("chip_controls", "对华芯片出口管制、232 条款半导体关税、中国稀土与磁材管制", "text", "按数据日期的现状",
          affects=["president", "china", "bigtech"], why="未锁定：按数据日期的现状并标注"),
    _item("antitrust_pending", "反垄断案件（司法部诉 Google、FTC 诉 Meta 与亚马逊）待决情况", "enum", "none",
          options=["none", "pending_this_quarter"], affects=["courts", "bigtech"], why="未锁定：没有本季待决裁决"),
    _item("ruling_tariff_quarter", "最高法院关税授权裁决的预计季度", "text", None, affects=["courts", "president"],
          why="未锁定：没有待决裁决 → 法院关闭"),
    _item("ruling_tariff_odds", "关税授权被裁定无效的概率（用户或市场）", "number", None, unit="概率", affects=["courts"],
          required=False, why="没有时 0.5，标“无可靠基率”"),
    _item("ruling_fed_removal_quarter", "美联储理事免职案的预计判决季度", "text", None, affects=["courts", "fed"],
          why="未锁定：没有待决裁决"),
    _item("ruling_antitrust_quarter", "反垄断救济的预计判决季度", "text", None, affects=["courts", "bigtech"],
          required=False, why="未锁定：没有待决裁决"),
    _item("approval", "总统支持率", "number", None, unit="%", affects=["president", "voter"], required=False,
          source="Gallup / UCSB APP", why="未锁定：不进观察包"),
    _item("china_growth_target", "中国 2026 年增长目标", "number", None, unit="%", affects=["china"], required=False,
          source="人大", why="未锁定：不进观察包"),
    _item("political_control", "两院与白宫的多数党", "text", "R 白宫；R 众议院（微弱多数）；R 参议院 53–47",
          affects=["congress", "voter", "president"], why="第 119 届国会开始时的格局（v1，请确认）"),
]

_TYPES = {"enum", "number", "date", "text", "bool"}


def institution_defaults() -> list[dict]:
    return copy.deepcopy(INSTITUTION_DEFAULTS)


def load_institutions(path: Path | None = None) -> list[dict]:
    """The pinned facts: shipped defaults (未确认) overlaid with the user's file, item by item (unknown ids kept)."""
    items = {d["id"]: d for d in institution_defaults()}
    p = Path(path) if path else INSTITUTIONS_PATH
    try:
        saved = jsonio.load(p)
    except (OSError, ValueError):
        saved = None
    rows = saved.get("items") if isinstance(saved, Mapping) else saved
    for row in rows or []:
        if not isinstance(row, Mapping) or not row.get("id"):
            continue
        base = items.get(row["id"], {})
        merged = {**base, **{k: v for k, v in row.items() if k in ("value", "status", "date", "source", "edited_at",
                                                                   "note")}}
        if not base:
            merged = dict(row)
        items[row["id"]] = merged
    return list(items.values())


def validate_institution(item: Mapping, value) -> tuple[object, str | None]:
    """(clean value, Chinese error or None) for a user edit of one pinned fact."""
    typ = item.get("type")
    if typ not in _TYPES:
        return value, f"未知的类型「{typ}」"
    if value is None:
        return None, None
    if typ == "number":
        if isinstance(value, bool):
            return value, f"「{item.get('zh')}」必须是数"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return value, f"「{item.get('zh')}」必须是数，收到「{value}」"
        if not math.isfinite(v):
            return value, f"「{item.get('zh')}」必须是有限数"
        if item.get("unit") == "概率" and not 0.0 <= v <= 1.0:
            return value, f"「{item.get('zh')}」是概率，必须在 0 到 1 之间"
        return v, None
    if typ == "enum":
        if value not in (item.get("options") or []):
            return value, f"「{item.get('zh')}」只能是 {'、'.join(map(str, item.get('options') or []))} 之一"
        return value, None
    if typ == "date":
        try:
            return dt.date.fromisoformat(str(value)[:10]).isoformat(), None
        except ValueError:
            return value, f"「{item.get('zh')}」必须是日期 YYYY-MM-DD"
    if typ == "bool":
        if isinstance(value, bool):
            return value, None
        return value, f"「{item.get('zh')}」必须是 true 或 false"
    text = str(value).strip()
    if len(text) > 400:
        return value, f"「{item.get('zh')}」不能超过 400 字"
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text), None


def save_institutions(edits: Mapping[str, object], path: Path | None = None, *, confirm: bool = True) -> list[dict]:
    """Apply user edits {id: value} (typed validation; the edit time is recorded) and write the file atomically.

    Raises ValueError with one Chinese line per bad edit; nothing is written then."""
    items = {d["id"]: d for d in load_institutions(path)}
    errors = []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for iid, value in edits.items():
        if iid not in items:
            errors.append(f"没有名为「{iid}」的制度现状项")
            continue
        v, err = validate_institution(items[iid], value)
        if err:
            errors.append(err)
            continue
        items[iid] = {**items[iid], "value": v, "status": "confirmed" if confirm else "unconfirmed", "edited_at": now}
    if errors:
        raise ValueError("；".join(errors))
    p = Path(path) if path else INSTITUTIONS_PATH
    jsonio.dump({"version": 1, "note": "制度现状与锁定事实：只由用户编辑，从不由大模型编辑", "items": list(items.values())}, p)
    return list(items.values())


def ensure_institutions_file(path: Path | None = None) -> Path:
    """Write the defaults (all 未确认) when the file is missing; an existing file is never touched."""
    p = Path(path) if path else INSTITUTIONS_PATH
    if not p.exists():
        jsonio.dump({"version": 1, "note": "制度现状与锁定事实：只由用户编辑，从不由大模型编辑；status: confirmed | "
                                           "unconfirmed | default", "items": institution_defaults()}, p)
    return p


def institution_values(items: Iterable[Mapping] | None = None) -> dict:
    return {d["id"]: d.get("value") for d in (items if items is not None else load_institutions())}


def root_state(items: Iterable[Mapping] | None = None) -> dict:
    """Facts and political state at the root from the pinned facts (the engine's node 0)."""
    v = institution_values(items)
    facts: dict = {}
    if v.get("belligerents_active", True):
        facts["oil_disruption_ongoing"] = True
    expiry = v.get("china_truce_expiry")
    if expiry:
        facts["trade_truce"] = True
        facts["trade_truce_expiry_q"] = quarter_of(expiry)
    status = v.get("ieepa_status") or "in_force"
    political = dict(POLITICAL_DEFAULT)
    political["tariff_authority"] = {"in_force": "broad", "pending": "broad"}.get(status, status)
    try:
        political["chair_tilt"] = float(v.get("fed_chair_tilt") or 0.0)
    except (TypeError, ValueError):
        political["chair_tilt"] = 0.0
    if v.get("house_margin") is not None:
        political["house_margin"] = v.get("house_margin")
    res = v.get("election_2026_result") or "unknown"
    if res == "house_flips":
        political["house_next"] = "flip"
    return {"facts": facts, "political": political, "values": v}


# =====================================================================================================================
# 6. Node context, activity, live rules and gates
# =====================================================================================================================

@dataclass
class NodeContext:
    """What the menus read at a node. `state`: schema s fields (medians) and last quarter's x values as <id>_l1
    (also `us_equity_l1`: last quarter's US-equity log return); `facts`: fact id -> value; `gates`: the accounts'
    gates (accounts.gates); `institutions`: pinned values by id (institution_values)."""
    quarter: str = "2026Q4"
    state: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    political: dict = field(default_factory=lambda: dict(POLITICAL_DEFAULT))
    institutions: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    commitments: list = field(default_factory=list)
    stress_mode: bool = False
    resolver: object = None
    leader_choices: dict = field(default_factory=dict)
    pinned_meeting: bool = True              # False: the model tree's menus (no trimming for a pinned meeting)

    @classmethod
    def of(cls, ctx) -> "NodeContext":
        if isinstance(ctx, NodeContext):
            return ctx
        if ctx is None:                      # no context: the root node (facts and politics from the pinned facts)
            items = load_institutions()
            st = root_state(items)
            return cls(institutions=st["values"], facts=dict(st["facts"]), political=dict(st["political"]))
        known = {k: v for k, v in dict(ctx).items() if k in cls.__dataclass_fields__}
        out = cls(**known)
        if not out.institutions:
            out.institutions = institution_values()
        return out

    def s(self, key: str, default=None):
        v = self.state.get(key, default)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    def fact(self, key: str):
        return self.facts.get(key)


def _unified(ctx: NodeContext) -> bool:
    p = ctx.political
    return p.get("pres_party") == p.get("house") == p.get("senate")


def _pending_rulings(ctx: NodeContext) -> dict[str, bool]:
    inst = ctx.institutions
    out = {}
    for case, key in (("tariff_authority", "ruling_tariff_quarter"), ("fed_removal", "ruling_fed_removal_quarter"),
                      ("antitrust", "ruling_antitrust_quarter")):
        q = inst.get(key)
        out[case] = bool(q) and str(q).strip().upper() == ctx.quarter
    if inst.get("antitrust_pending") == "pending_this_quarter":
        out["antitrust"] = True
    if inst.get("ieepa_status") == "pending" and not inst.get("ruling_tariff_quarter"):
        out["tariff_authority"] = out.get("tariff_authority", False)
    return out


def _election(ctx: NodeContext) -> dict | None:
    e = ELECTIONS.get(ctx.quarter)
    if not e:
        return None
    if e["kind"] == "midterm" and ctx.quarter == "2026Q4" and ctx.institutions.get("election_2026_result") not in (
            None, "unknown"):
        return None                                     # the result is pinned: no branch
    return e


def actor_activity(actor_id: str, ctx=None) -> tuple[bool, str]:
    """(active, Chinese reason) for one actor at a node."""
    c = NodeContext.of(ctx)
    a = ACTORS[actor_id]
    rule = a.active
    if rule == "always":
        return True, "每季"
    if rule == "election_quarter":
        e = _election(c)
        if e:
            return True, f"{e['zh']}"
        if ELECTIONS.get(c.quarter):
            return False, "选举结果已锁定"
        return False, "不是联邦选举季度"
    if rule == "pending_ruling":
        pend = [k for k, v in _pending_rulings(c).items() if v]
        return (True, "本季有待决裁决：" + "、".join(pend)) if pend else (False, "无待决裁决")
    if rule == "oil_disruption":
        return (True, "石油供给中断持续") if c.fact("oil_disruption_ongoing") else (False, "没有武装冲突导致的供给中断")
    if rule == "bill_share_field":
        r = _resolver(c.resolver)
        return (True, "输入表有短债占比字段") if r.field("bill_share_chg") else (False, "输入表无短债占比字段")
    return False, f"未知的活动规则 {rule}"


def active_actors(ctx=None) -> list[dict]:
    """Every actor in stage order with {id, zh, stage, kind, active, why, llm}. Nature is listed as code."""
    c = NodeContext.of(ctx)
    out = [{"id": "nature", "zh": "自然", "stage": 0, "kind": "code", "active": True, "why": "代码风险率（每季）",
            "llm": False}]
    for aid in STAGE_ORDER:
        on, why = actor_activity(aid, c)
        out.append({"id": aid, "zh": ACTORS[aid].zh, "stage": ACTORS[aid].stage, "kind": ACTORS[aid].kind,
                    "active": on, "why": why, "llm": True})
    return out


def _deadline_quarter(ctx: NodeContext) -> bool:
    exp = ctx.institutions.get("cr_expiry")
    if exp:
        try:
            if quarter_of(exp) == ctx.quarter:
                return True
        except ValueError:
            pass
    return ctx.quarter.endswith("Q4")


LIVE: dict[str, Callable[[NodeContext], tuple[bool, str]]] = {
    "always": lambda c: (True, ""),
    "deadline_quarter": lambda c: (_deadline_quarter(c), "拨款截止季度" if _deadline_quarter(c) else "不是截止季度"),
    "tariff_struck_down": lambda c: (str(c.facts.get("tariff_authority") or c.political.get("tariff_authority"))
                                     .startswith("struck_down"), "关税授权被裁定无效"),
    "ruling:tariff_authority": lambda c: (_pending_rulings(c).get("tariff_authority", False), "关税授权裁决"),
    "ruling:fed_removal": lambda c: (_pending_rulings(c).get("fed_removal", False), "免职保护裁决"),
    "ruling:antitrust": lambda c: (_pending_rulings(c).get("antitrust", False), "反垄断裁决"),
    "election:midterm": lambda c: ((_election(c) or {}).get("kind") == "midterm", "中期选举"),
    "election:presidential": lambda c: ((_election(c) or {}).get("kind") == "presidential", "总统选举"),
}


def _du2(c: NodeContext) -> float | None:
    if c.state.get("du2") is not None:
        return c.s("du2")
    a, b = c.s("d_unrate_l1"), c.s("d_unrate_l2")
    return a + b if a is not None and b is not None else a


def _gate_fed_emergency(c):
    if c.fact("financial_accident") or c.fact("pandemic") or c.stress_mode:
        return True, "金融事故、疫情或压力模式"
    eq = c.s("us_equity_l1")
    if eq is not None and eq <= math.log(0.85):
        return True, "上季美股 ≤ −15%"
    ds = c.s("d_spread_l1")
    if ds is not None and ds >= 0.75:
        return True, "上季 Baa 利差 +0.75 以上"
    return False, "没有金融事故或疫情事实，上季美股与利差也未触发"


def _gate_big_hike(c):
    core, inf1, u = c.s("s_core_yoy"), c.s("s_infexp1"), c.s("s_unrate")
    hot = (core is not None and core >= 4.0) or (inf1 is not None and inf1 >= 4.5)
    if hot and (u is None or u < 5.0):
        return True, "核心通胀 ≥ 4% 或 1 年通胀预期 ≥ 4.5%，失业率 < 5%"
    return False, "核心通胀 < 4% 且 1 年通胀预期 < 4.5%（或失业率 ≥ 5%）"


def _gate_qe(c):
    ff = c.s("s_ff")
    if ff is not None and ff <= 0.25:
        return True, "零下限"
    if c.fact("financial_accident") or c.fact("pandemic") or c.stress_mode:
        return True, "压力事实"
    return False, "不在零下限，也没有压力事实"


def _gate_spr(c):
    if not (c.fact("oil_disruption_ongoing") or c.fact("escalation")):
        return False, "没有石油供给中断"
    inv, floor = c.s("spr_inventory_mb"), float(c.institutions.get("spr_floor_mb") or 200.0)
    if inv is not None and inv < floor + 30.0:
        return False, f"战略储备 {inv:.0f} 百万桶已接近下限 {floor:.0f}"
    return True, "石油供给中断持续"


def _gate_debt_limit(c):
    req = ((c.gates.get("congress") or {}).get("require_options") or [])
    if req or c.fact("debt_limit_binding") or c.fact("debt_limit_standoff"):
        return True, "债务上限约束（G-L1）"
    x = c.institutions.get("debt_limit_xdate")
    if x:
        try:
            if quarters_between(c.quarter, quarter_of(x)) <= 1:
                return True, "X 日期在本季或下季"
        except ValueError:
            pass
    return False, "债务上限本季不约束"


def _gate_stimulus_small(c):
    sahm, du2 = c.s("s_sahm"), _du2(c)
    if (sahm is not None and sahm >= 0.3) or (du2 is not None and du2 >= 0.5):
        return True, "萨姆缺口 ≥ 0.3 或两季失业率上升 ≥ 0.5pp"
    e = ELECTIONS.get(quarter_add(c.quarter, 1)) or ELECTIONS.get(quarter_add(c.quarter, 2))
    if e and _unified(c):
        return True, "两季内有选举且一党执政"
    return False, "没有衰退信号"


def _gate_stimulus_large(c):
    sahm, du2 = c.s("s_sahm"), _du2(c)
    if c.fact("pandemic") or (c.fact("financial_accident") and c.stress_mode):
        return True, "疫情或系统性金融事故"
    if sahm is not None and sahm >= 0.5 and du2 is not None and du2 >= 1.0:
        return True, "衰退信号（萨姆缺口 ≥ 0.5 且两季失业率 +1pp）"
    return False, "没有衰退信号或疫情"


def _gate_quota_raise(c):
    spare = c.institutions.get("opec_spare_mbd")
    if spare is None:
        return True, "可达闲置产能未确认（按选项范围，已标注）"
    try:
        return (float(spare) >= 0.5), f"可达闲置产能 {float(spare):g} 百万桶/日"
    except (TypeError, ValueError):
        return True, "可达闲置产能未确认"


GATES: dict[str, Callable[[NodeContext], tuple[bool, str]]] = {
    "fed_emergency": _gate_fed_emergency,
    "fed_big_hike": _gate_big_hike,
    "fed_qe": _gate_qe,
    "spr_release": _gate_spr,
    "attempt_removal": lambda c: ((not c.fact("fed_removal_protected")), "法院维持了免职保护"
                                  if c.fact("fed_removal_protected") else ""),
    "stimulus_small": _gate_stimulus_small,
    "stimulus_large": _gate_stimulus_large,
    "unified_government": lambda c: (_unified(c), "一党执政" if _unified(c) else "分治政府"),
    "debt_limit_live": _gate_debt_limit,
    "quota_raise": _gate_quota_raise,
    "stress_mode": lambda c: (bool(c.stress_mode), "压力模式" if c.stress_mode else "压力模式关闭：封锁作粒子混合"),
}


def _effect_free(o: Option, r: FieldResolver) -> bool:
    """True when nothing in the option resolves: no field write, fact, political update, account action or event."""
    if o.facts or o.clears or o.political or o.account or o.oil_z or o.gpr_sd or o.status_quo:
        return False
    for w in o.writes:
        if r.field(w.role) is not None or ROLES.get(w.role, {}).get("fallback"):
            return False
    return True


def menu(actor_id: str, ctx=None) -> dict:
    """The node's menu for one actor: {actor, active, why, main: {...}, attributes: [...], disabled: [...], notes}.

    main/attributes: {"domain", "zh", "options": [option summary], "required": [ids], "max_options"}; every gate,
    calendar and pinned fact is applied; account gates add required options (G-L1); a commitment due this quarter
    adds the 如期执行 option; options with nothing to write are disabled with a reason."""
    c = NodeContext.of(ctx)
    r = _resolver(c.resolver)
    a = ACTORS[actor_id]
    on, why = actor_activity(actor_id, c)
    out = {"actor": actor_id, "zh": a.zh, "active": on, "why": why, "main": None, "attributes": [], "disabled": [],
           "notes": []}
    if not on:
        return out
    mains = [d for d in a.domains if d.role == "main" and LIVE.get(d.live, LIVE["always"])(c)[0]]
    if actor_id == "president" and any(d.id == "trade_substitute" for d in mains):
        mains = [d for d in mains if d.id == "trade_substitute"]        # the substitute menu replaces the trade menu
        out["notes"].append("关税授权被裁定无效：关税菜单换成替代授权（无历史先例）")
    elif actor_id == "president":
        mains = [d for d in mains if d.id == "trade"]
    if not mains:
        out["active"], out["why"] = False, "本季没有可选的领域"
        return out
    for i, d in enumerate(mains):
        block = _domain_block(a, d, c, r, out["disabled"])
        if i == 0:
            out["main"] = block
        else:
            block["role"] = "main_extra"
            out["attributes"].append(block)
    for d in a.domains:
        if d.role != "attribute" or not LIVE.get(d.live, LIVE["always"])(c)[0]:
            continue
        out["attributes"].append(_domain_block(a, d, c, r, out["disabled"]))
    if actor_id == "fed" and out["main"] is not None and c.quarter in FOMC_PINNED and c.pinned_meeting:
        _apply_pinned_meeting(out, c)
    # account gates: required options (G-L1 for Congress); commitments due: the 如期执行 option
    req = (c.gates.get(actor_id) or {}).get("require_options") or []
    if req and out["main"] is not None:
        have = {o["id"] for o in out["main"]["options"]}
        out["main"]["required"] = [o for o in req if o in have]
        if out["main"]["required"]:
            out["notes"].append("账户约束门要求菜单包含：" + "、".join(out["main"]["required"]))
    due = [cm for cm in c.commitments if cm.get("actor") == actor_id and int(cm.get("due_in", 0)) == 0
           and cm.get("cancellable", True)]
    if due and out["main"] is not None:
        out["main"]["options"].insert(0, {"id": "as_committed", "zh": "如期执行（" + "；".join(
            str(cm.get("zh") or cm.get("option")) for cm in due)[:60] + "）", "status_quo": False, "param": None,
            "commitment": True, "signal": {"text": "按已公布的计划执行", "direction": 0, "horizon_q": 1}})
        out["main"]["required"] = list(dict.fromkeys(["as_committed", *out["main"].get("required", [])]))
    return out


FED_OPTION_VALUE = {"cut_50": -0.5, "cut_25": -0.25, "hold": 0.0, "hike_25": 0.25, "hike_50": 0.5}
PINNED_MEETING_MOVE = {"hold": 0.0, "cut_25": -0.25, "cut_50": -0.5, "hike_25": 0.25}


def _apply_pinned_meeting(out: dict, c: NodeContext) -> None:
    """2026Q4: the Oct 27–28 decision is pinned (默认：维持，未确认), so the quarter's total is that move plus the
    December meeting's −25 / 0 / +25bp; other regular options leave the menu (the emergency ones keep their gates)."""
    pin = FOMC_PINNED[c.quarter]
    first = PINNED_MEETING_MOVE.get(str(c.institutions.get(pin["id"]) or "hold"))
    if first is None:
        return
    out["main"]["pinned_first"] = {"meeting": pin["id"], "move": first,
                                   "option": str(c.institutions.get(pin["id"]) or "hold")}
    ok = {round(first + s, 2) for s in (-0.25, 0.0, 0.25)}
    keep = []
    for o in out["main"]["options"]:
        v = FED_OPTION_VALUE.get(o["id"])
        if v is not None and round(v, 2) not in ok:
            out["disabled"].append({"domain": "rate", "id": o["id"], "zh": o["zh"],
                                    "why": f"{pin['zh']}已锁定（{c.institutions.get(pin['id']) or 'hold'}）：本季只剩 12 月一次会议"})
            continue
        keep.append(o)
    out["main"]["options"] = keep
    out["notes"].append(f"{pin['zh']}是锁定事实：本季选项在相容的选项上重新归一")


def fed_cut_room(o: Option, c: NodeContext) -> tuple[bool, str]:
    """A rate cut is on the menu only if the model can deliver it: the policy rate s_ff (branch median) minus the
    cut stays above the model's floor (ctx.state["ff_floor"]: the quarterly stack's 0.1, the stub's 0) by more than
    half a 25bp step. Below that the cut would be delivered as a smaller one (or a hold) and duplicate it."""
    ff = c.s("ff_target", c.s("s_ff"))           # the end-of-quarter target (the engine's state), else s_ff
    if ff is None:
        return True, ""
    floor = float(c.s("ff_floor", 0.0) or 0.0)
    if o.param is not None:
        size = abs(float(o.param.hi))                  # the smallest cut the option allows (cut_75plus: 0.75)
    else:
        size = abs(float(FED_OPTION_VALUE.get(o.id, 0.0)))
    if size <= 0:
        return True, ""
    room = ff - floor
    if room < size - FED_STEP / 2:
        return False, f"政策利率 {ff:.2f}% 离模型的下限 {floor:g}% 只有 {max(room, 0):.2f} 个百分点：降息 {size:g} 无法交付"
    return True, ""


def option_value_range(actor_id: str, option_id: str, domain_id: str | None = None) -> tuple[float, float] | None:
    """The values an instrument option realizes on its field: a point write (a, a), a magnitude point write the
    param's [lo, hi], a bin write its [lo, hi]; None for options without an instrument write. A0 bins of adjacent
    menu options split at the midpoint of the nearest ends of these ranges (the catalogue bins when all are kept)."""
    a = ACTORS[actor_id]
    d = a.domain(domain_id) if domain_id else a.main
    o = d.option(option_id)
    for w in o.writes:
        if ROLES.get(w.role, {}).get("kind") != "instrument" or w.dq:
            continue
        if w.kind == "bin":
            return float(w.lo), float(w.hi)
        if w.kind == "point":
            if w.magnitude and o.param is not None:
                return float(min(o.param.lo, o.param.hi)), float(max(o.param.lo, o.param.hi))
            return float(w.a), float(w.a)
    return None


def _domain_block(a: Actor, d: Domain, c: NodeContext, r: FieldResolver, disabled: list) -> dict:
    opts = []
    for o in d.options:
        if o.gate:
            ok, why = GATES[o.gate](c)
            if not ok:
                disabled.append({"domain": d.id, "id": o.id, "zh": o.zh, "why": why})
                continue
        if a.id == "fed" and d.id == "rate" and o.id.startswith("cut"):
            ok, why = fed_cut_room(o, c)
            if not ok:
                disabled.append({"domain": d.id, "id": o.id, "zh": o.zh, "why": why})
                continue
        if _effect_free(o, r):
            disabled.append({"domain": d.id, "id": o.id, "zh": o.zh, "why": "输入表里没有它能写的字段，也没有事实或政治效果"})
            continue
        opts.append(option_summary(a.id, o, r))
    return {"domain": d.id, "zh": d.zh, "role": d.role, "max_options": d.max_options, "prior": d.prior,
            "options": opts, "required": [], "branch_when": d.branch_when, "note": d.note}


def option_summary(actor_id: str, o: Option, resolver=None) -> dict:
    """What the prompt and the UI show for one option (canonical writes as a table, the range, the signal)."""
    r = _resolver(resolver)
    writes = []
    for w in o.writes:
        f = r.field(w.role)
        fb = ROLES.get(w.role, {}).get("fallback")
        writes.append({"role": w.role, "field": f, "kind": w.kind, "dq": w.dq,
                       "fallback": None if f else (fb or "disabled"),
                       "a": None if w.magnitude else w.a, "b": w.b, "per_unit": w.per_unit, "note": w.note})
    if o.oil_z:
        writes.append({"role": "oil_supply_signature", "field": r.field("oil_supply_signature"), "kind": "event",
                       "z": o.oil_z[0], "per_unit": o.oil_z[2], "note": "石油供给合成量（z 单位）"})
    if o.gpr_sd:
        writes.append({"role": "gpr_chg", "field": r.field("gpr_chg"), "kind": "event", "sd": o.gpr_sd[0]})
    p = o.param
    return {"id": o.id, "zh": o.zh, "status_quo": o.status_quo,
            "param": None if p is None else {"name": p.name, "zh": p.zh, "unit": p.unit, "lo": p.lo, "hi": p.hi,
                                             "default": p.default, "step": p.step},
            "writes": writes, "facts": [k for k, _ in o.facts], "political": [k for k, _ in o.political],
            "bin": list(o.bin) if o.bin else None, "protocol": o.protocol, "flags": list(o.flags), "note": o.note,
            "signal": {"text": o.signal[0], "direction": o.signal[1], "horizon_q": o.signal[2]}}


def instrument_bins(actor_id: str, domain_id: str | None = None) -> list[tuple[str, float, float]]:
    """(option id, lo, hi) of the options that are bins of the instrument's own A0 distribution, sorted by lo:
    the edges the engine reads the model's history prior from (§9.2 item 2; the prior-only trees' payloads)."""
    a = ACTORS[actor_id]
    d = a.domain(domain_id) if domain_id else a.main
    rows = sorted(((o.id, float(o.bin[0]), float(o.bin[1])) for o in d.options if o.bin and o.id != "deal"),
                  key=lambda t: t[1])
    return rows


# typed facts a custom option may set: the actor's own decisions only (never an exogenous event or an account fact)
CUSTOM_FACTS = {"president": ("export_controls_level", "trade_truce", "imm_tightening", "spr_release",
                              "fed_pressure_public", "dovish_nominee", "fed_removal_attempt"),
                "china": ("cn_retaliation", "export_controls_cn", "trade_truce", "cn_stimulus", "cny_weaker",
                          "taiwan_exercise")}


def custom_option_rules(actor_id: str, resolver=None) -> dict:
    """What a custom option may write (§4.3): only the actor's own levers (resolved fields) or typed facts, with
    its probability capped (0.15 for the President and China; no custom option for anyone else)."""
    a = ACTORS[actor_id]
    r = _resolver(resolver)
    fields = sorted({f for f in (r.field(role) for role in a.levers if ROLES.get(role, {}).get("kind") in
                                 ("instrument", "shift_target")) if f})
    return {"allowed": a.custom_cap > 0, "cap": a.custom_cap, "writable_fields": fields,
            "facts": list(CUSTOM_FACTS.get(actor_id, ())),
            "kinds": {"point": [f for f in fields if f == r.field("tariff_chg")],
                                              "shift": [f for f in fields if f != r.field("tariff_chg")]}}


def validate_custom_option(actor_id: str, option: Mapping, resolver=None) -> tuple[dict | None, list[str]]:
    """Guard G1 for a custom option: drop write keys outside the actor's levers, unknown facts and a probability
    above the cap. Returns (clean option or None when the actor may not add one, Chinese flags)."""
    rules = custom_option_rules(actor_id, resolver)
    zh = ACTORS[actor_id].zh
    if not rules["allowed"]:
        return None, [f"{zh}没有自定义选项"]
    flags: list[str] = []
    writes = {}
    for k, v in dict(option.get("writes") or {}).items():
        if k not in rules["writable_fields"]:
            flags.append(f"{zh}的自定义选项不能写「{k}」：已删除")
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            flags.append(f"{zh}的自定义选项「{k}」不是数：已删除")
            continue
        if not math.isfinite(fv):
            flags.append(f"{zh}的自定义选项「{k}」不是有限数：已删除")
            continue
        writes[k] = fv
    facts = [f for f in (option.get("facts") or []) if f in rules["facts"]]
    if len(facts) != len(option.get("facts") or []):
        flags.append(f"{zh}的自定义选项含不属于自己的事实（外生事件只来自自然与冲突方）：已删除")
    try:
        prob = float(option.get("prob", 0.0) or 0.0)
    except (TypeError, ValueError):
        prob = 0.0
    if prob > rules["cap"]:
        flags.append(f"{zh}的自定义选项概率 {prob:.2f} 超过上限 {rules['cap']:.2f}：已截断")
        prob = rules["cap"]
    return {**dict(option), "id": "custom", "writes": writes, "facts": facts, "prob": max(prob, 0.0)}, flags


def trim_menu_by_prior(option_ids: list[str], prior: Mapping[str, float], *, cover: float = 0.90, max_options: int = 4,
                       keep: Iterable[str] = ()) -> list[str]:
    """The fewest options covering >= `cover` of the prior, at most `max_options`, always including `keep` (v1 rule)."""
    keep = [k for k in keep if k in option_ids]
    ranked = sorted((o for o in option_ids if o not in keep), key=lambda o: -float(prior.get(o, 0.0)))
    out = list(keep)
    for o in ranked:
        if len(out) >= max_options or sum(float(prior.get(k, 0.0)) for k in out) >= cover:
            break
        out.append(o)
    return [o for o in option_ids if o in out]


# =====================================================================================================================
# 7. Payloads: an option's writes on schema fields
# =====================================================================================================================

OIL_COMPOSITE = {
    "beta_metals_oil_per_z": -6.8,      # log% of d_metals_oil per z (d01: NW se 1.6, R2 0.18, 1985-2019, n 140)
    "oil_logpct_per_z": 10.9,           # d01: d_oil on z, 1985-2019 (for the z <-> mb/d chain only)
    "z_per_mbd": 0.32,                  # 判断值: +3.5% oil per 1% of world supply / 10.9 log% per z
    "cause_label_min_abs_z": 0.5,       # |sum z| >= 0.5: the quarter's oil move is labelled supply (x_oil_cause = 1)
    "p_z_gt_2": 0.0239,                 # base rate check (d01)
    "source": "MAIN data/processed/structural_shocks.csv e_oil_supply，季内求和，按 1985–2019 标准差标准化（d01）",
}


def _support_flags(fid: str | None, value: float, hist: Mapping | None) -> tuple[float, list[str], bool]:
    """Clip a requested value to the field's history: p01-p99 fine; up to the extreme flagged 罕见; beyond clipped."""
    if not hist or fid is None:
        return value, [], False
    lo99, hi99 = hist.get("p01"), hist.get("p99")
    mn, mx = hist.get("min"), hist.get("max")
    flags, beyond = [], False
    if lo99 is not None and hi99 is not None and (value < lo99 or value > hi99):
        if (mn is None or value >= mn) and (mx is None or value <= mx):
            flags.append(f"{fid} = {value:g} 在 1–99% 区间之外：罕见")
        else:
            beyond = True
            clipped = min(max(value, mn if mn is not None else value), mx if mx is not None else value)
            flags.append(f"{fid} = {value:g} 超出历史极值，截断到 {clipped:g}（超出可测历史，原值留给合成器）")
            value = clipped
    return value, flags, beyond


def option_payload(actor_id: str, option_id: str, magnitude=None, *, resolver=None, support: Mapping | None = None) -> dict:
    """The writes, facts, political updates and account actions of one chosen option.

    support: field id -> hist {p01, p99, min, max, sd_q_1985_2019} (the registry's); the resolver's overlay hist
    is used when not given. Returns {"actor", "option", "magnitude", "requested", "writes": [...], "events": {...},
    "facts": [...], "clears": [...], "political": {...}, "account": {...}, "unresolved": [...], "flags": [...]}.
    Each write: {field, role, kind, a, b, lo, hi, dq, source, beyond_history, requested}."""
    r = _resolver(resolver)
    a = ACTORS[actor_id]
    dom, o = a.find(option_id)
    flags: list[str] = []
    mag = None
    if o.param is not None:
        mag, f = o.param.clip(o.param.default if magnitude is None else magnitude)
        flags += f
    elif magnitude not in (None, 0, 0.0):
        flags.append(f"选项「{o.id}」没有幅度参数，忽略 {magnitude}")
    out = {"actor": actor_id, "domain": dom.id, "option": o.id, "zh": o.zh, "magnitude": mag,
           "requested": magnitude, "writes": [], "events": {"oil_z": [], "gpr": []}, "facts": [], "clears": list(o.clears),
           "political": {}, "account": {}, "unresolved": [], "flags": flags, "protocol": o.protocol,
           "source": f"actor:{actor_id}"}

    def hist_of(fid):
        if support is not None and fid in support:
            return support[fid]
        return r.hist(fid)

    for w in o.writes:
        fid = r.field(w.role)
        role = w.role
        a_, b_ = w.a, w.b
        if w.magnitude:
            a_ = mag if mag is not None else 0.0
        if w.per_unit:
            m = mag if mag is not None else 1.0
            a_, b_ = w.a * m, w.b * abs(m)
        if fid is None:
            fb = ROLES.get(role, {}).get("fallback")
            if role == "hyperscaler_capex_g" and fb == "it_invest_g":
                # ACTORS_V2 §4.1.6: d_itinv += s_bt * x for 4 quarters (commitments revisable at the next earnings)
                fid2 = r.field("it_invest_g")
                if fid2 is None:
                    out["unresolved"].append(role)
                    continue
                x = a_
                for k in range(4):
                    out["writes"].append(_w(fid2, "it_invest_g", "shift", S_BT * x, 0.5 * S_BT * abs(x), k,
                                            out["source"], note=f"后备：s_bt {S_BT} × {x:g}%（判断值）"))
                flags.append(f"字段 {role} 不存在：写 IT 投资增量 {S_BT * x:+.2f}pp，持续 4 季（承诺）")
                continue
            out["unresolved"].append(role)
            continue
        if w.kind == "point":
            v, f, beyond = _support_flags(fid, float(a_), hist_of(fid))
            flags += f
            out["writes"].append(_w(fid, role, "point", v, 0.0, w.dq, out["source"], beyond=beyond,
                                    requested=float(a_) if beyond else None, note=w.note))
        elif w.kind == "bin":
            out["writes"].append(_w(fid, role, "bin", 0.0, 0.0, w.dq, out["source"], lo=float(w.lo), hi=float(w.hi),
                                    note=w.note))
        else:                                   # normal (absolute) or shift (A0 draw + Δ)
            out["writes"].append(_w(fid, role, w.kind, float(a_), float(b_), w.dq, out["source"], note=w.note))
    if o.oil_z:
        za, zb, per = o.oil_z
        m = abs(mag) if (per and mag is not None) else 1.0
        out["events"]["oil_z"].append({"a": za * m, "b": zb * m, "actor": actor_id, "option": o.id})
    if o.gpr_sd:
        out["events"]["gpr"].append({"a_sd": o.gpr_sd[0], "b_sd": o.gpr_sd[1], "actor": actor_id, "option": o.id})
    for k, v in o.facts:
        out["facts"].append({"id": k, "value": mag if v == "magnitude" else v, "exit": FACTS.get(k, {}).get("exit"),
                             "zh": FACTS.get(k, {}).get("zh", k)})
    for k, v in o.political:
        out["political"][k] = v
    for k, v in o.account:
        out["account"][k] = (mag if v == "magnitude" else (-mag if v == "-magnitude" and mag is not None else v))
    if out["unresolved"] and not (out["writes"] or out["facts"] or out["events"]["oil_z"] or out["events"]["gpr"]
                                  or out["account"] or out["political"]):
        flags.append("没有可写的字段：该选项只进入记忆与新闻")
    return out


S_BT = 0.16          # hyperscaler US equipment capex / US IT equipment and software investment (判断值, ACTORS_V2)


def _w(fid, role, kind, a, b, dq, source, *, lo=None, hi=None, beyond=False, requested=None, note=""):
    """One resolved write; lo/hi are set only for a bin (the A0 particles inside it are resampled)."""
    return {"field": fid, "role": role, "kind": kind, "a": a, "b": b, "lo": lo, "hi": hi, "dq": int(dq),
            "source": source, "beyond_history": bool(beyond), "requested": requested, "note": note}


def hazard_payload(hazard_id: str) -> dict:
    """Nature's event contribution for a branched (or pinned) hazard, in the same shape as option_payload."""
    h = HAZARDS[hazard_id]
    out = {"actor": "nature", "option": hazard_id, "zh": h["zh"], "writes": [], "events": {"oil_z": [], "gpr": []},
           "facts": [{"id": h["fact"], "value": True, "exit": h.get("exit"), "zh": FACTS.get(h["fact"], {}).get("zh")}],
           "clears": [], "political": {}, "account": {}, "unresolved": [], "flags": [], "source": "event:nature"}
    if h.get("oil_z"):
        out["events"]["oil_z"].append({"a": h["oil_z"][0], "b": h["oil_z"][1], "actor": "nature", "option": hazard_id})
    return out


def combine_events(payloads: Iterable[Mapping], *, resolver=None, support: Mapping | None = None) -> list[dict]:
    """The shared event channels of one joint profile, summed over actors (§4.4).

    Oil: contributions add in z units; with a structural oil-supply field the sum is written there (normal);
    otherwise the schema's own supply signature: a shift β·Σz on d_metals_oil (β −6.8 log% per z), and the cause
    label x_oil_cause = 1 (supply) when |Σz| ≥ 0.5. GPR: contributions in sd units add as one shift on d_gpr."""
    r = _resolver(resolver)
    zs = [e for p in payloads for e in (p.get("events") or {}).get("oil_z", [])]
    gs = [e for p in payloads for e in (p.get("events") or {}).get("gpr", [])]
    out = []
    srcs = sorted({f"event:{e['actor']}" for e in zs})
    if zs:
        za = sum(float(e["a"]) for e in zs)
        zb = math.sqrt(sum(float(e["b"]) ** 2 for e in zs))
        f_struct = r.field("oil_supply_shock")
        if f_struct:
            out.append(_w(f_struct, "oil_supply_shock", "normal", za, zb, 0, "+".join(srcs), note="石油供给冲击（z）"))
        else:
            fid = r.field("oil_supply_signature")
            if fid:
                beta = OIL_COMPOSITE["beta_metals_oil_per_z"]
                out.append(_w(fid, "oil_supply_signature", "shift", beta * za, abs(beta) * zb, 0, "+".join(srcs),
                              note=f"石油供给合成量 Σz = {za:+.2f}（β {beta} log%/z）"))
        lab = r.field("oil_cause_label")
        if lab and abs(za) >= OIL_COMPOSITE["cause_label_min_abs_z"]:
            out.append(_w(lab, "oil_cause_label", "point", 1.0, 0.0, 0, "+".join(srcs), note="油价变动原因：供给"))
    if gs:
        fid = r.field("gpr_chg")
        if fid:
            h = (support or {}).get(fid) or r.hist(fid) or {}
            sd = float(h.get("sd_q_1985_2019") or FALLBACK_SD["gpr_chg"])
            a_sd = sum(float(e["a_sd"]) for e in gs)
            b_sd = math.sqrt(sum(float(e["b_sd"]) ** 2 for e in gs))
            val, flags, beyond = _support_flags(fid, a_sd * sd, h)
            w = _w(fid, "gpr_chg", "shift", val, b_sd * sd, 0, "+".join(sorted({f"event:{e['actor']}" for e in gs})),
                   beyond=beyond, requested=a_sd * sd if beyond else None, note=f"GPR 贡献合计 {a_sd:+.2f} sd")
            w["flags"] = flags
            out.append(w)
    return out


def account_actions(payloads: Iterable[Mapping]) -> dict:
    """Merge the account actions of a joint profile (the accounts' `actions` argument)."""
    acts: dict = {}
    for p in payloads:
        for k, v in (p.get("account") or {}).items():
            if k == "bt_capex_plan_pct":
                acts["bt_new_guidance"] = True
            acts[k] = v
    return acts


def commitments_of(payload: Mapping, quarter: str) -> list[dict]:
    """The writes due in later quarters (dq > 0) as commitments {actor, option, due_quarter, write, cancellable}.
    Each write keeps the conditioning protocol of the option that made it (write["protocol"]), so the quarter it
    falls due reads it as the first quarter did (an intervention stays an intervention, §6.3)."""
    out = []
    for w in payload.get("writes", []):
        if int(w.get("dq", 0)) > 0:
            out.append({"actor": payload["actor"], "option": payload["option"], "zh": payload.get("zh"),
                        "due_quarter": quarter_add(quarter, int(w["dq"])),
                        "write": {**w, "dq": 0, "protocol": w.get("protocol") or payload.get("protocol")},
                        "cancellable": True})
    return out


# =====================================================================================================================
# 8. Interactions (§4.5)
# =====================================================================================================================

REACTIONS: list[dict] = [
    {"follower": ("china", "trade_response"), "leader": ("president", "trade"), "kind": "rows", "llm_may_adjust": True,
     "rows": {"tariff_up_large": {"retaliate_tariffs": 0.50, "retaliate_export_controls": 0.40, "negotiate_truce": 0.05,
                                  "no_response": 0.05},
              "tariff_up_small": {"retaliate_tariffs": 0.35, "retaliate_export_controls": 0.15, "negotiate_truce": 0.10,
                                  "no_response": 0.40},
              "tariff_hold": {"no_response": 0.80, "negotiate_truce": 0.15, "retaliate_tariffs": 0.05},
              "tariff_down": {"negotiate_truce": 0.80, "no_response": 0.20},
              "deal": {"negotiate_truce": 0.80, "no_response": 0.20}},
     "source": "2018–25 年 8/8 次报复（Bown 2021，W_A03）；后三行判断值"},
    {"follower": ("opec", "quota"), "leader": ("belligerents", "conflict"), "kind": "rows", "llm_may_adjust": True,
     "rows": {"escalate": {"quota_raise": 0.45, "quota_hold": 0.50, "quota_cut": 0.05}},
     "source": "判断值（v1：1990、2003 年沙特增产；2022 年欧佩克+ 维持）；受可达闲置产能约束"},
    {"follower": ("president", "energy"), "leader": ("belligerents", "conflict"), "kind": "rows", "llm_may_adjust": True,
     "rows": {"escalate": {"spr_release": 0.5, "spr_hold": 0.5}}, "source": "判断值（v1：1990 年以来 6 次中断约 4 次释放）"},
    {"follower": ("bigtech", "capex"), "leader": ("president", "export_controls"), "kind": "tilt",
     "rows": {"ec_tighten": {"capex_cut": 1.5, "capex_hold": 1.5}}, "source": "判断值（ACTORS_V2 §8）"},
    {"follower": ("bigtech", "capex"), "leader": ("china", "trade_response"), "kind": "tilt",
     "rows": {"retaliate_export_controls": {"capex_cut": 1.5, "capex_hold": 1.5}}, "source": "判断值（ACTORS_V2 §8）"},
    {"follower": ("president", "trade"), "leader": ("courts", "tariff_authority"), "kind": "menu",
     "rows": {"strike_down": "trade_substitute", "strike_down_refunds": "trade_substitute"},
     "source": "只有一例：菜单换成替代授权，LLM 给条件表，没有基率"},
]


def reaction_rows(follower: str, leader_choices: Mapping[str, str] | None = None) -> list[dict]:
    """The reaction rows that apply to `follower` given same-quarter leader choices {actor: option id}.

    Each: {follower_domain, leader, leader_option, kind (rows | tilt | menu), row, source, llm_may_adjust}."""
    out = []
    for rr in REACTIONS:
        fa, fd = rr["follower"]
        if fa != follower:
            continue
        la, ld = rr["leader"]
        lo = (leader_choices or {}).get(la)
        if lo is None or lo not in rr["rows"]:
            continue
        out.append({"follower_domain": fd, "leader": la, "leader_domain": ld, "leader_option": lo, "kind": rr["kind"],
                    "row": copy.deepcopy(rr["rows"][lo]), "source": rr["source"],
                    "llm_may_adjust": rr.get("llm_may_adjust", False)})
    return out


def reaction_pairs() -> list[dict]:
    """Every catalogue pair (for the prompt's reaction table and the output schema's `reactions` enum)."""
    return [{"follower": list(rr["follower"]), "leader": list(rr["leader"]), "kind": rr["kind"],
             "leader_options": list(rr["rows"]), "source": rr["source"]} for rr in REACTIONS]


def lagged_rules(prev: Mapping | None = None, ctx=None) -> list[dict]:
    """Next-quarter code rules (§4.5): prior tilts and menu changes from last quarter's choices and markets.

    prev: {"choices": {actor: option}, "facts": {...}, "us_equity_q": log return, "d_y10": pp, "quarter": "YYYYQn"}."""
    p = dict(prev or {})
    ch = p.get("choices") or {}
    facts = p.get("facts") or {}
    out = []
    if ch.get("voter") == "house_flips":
        out.append({"rule": "new_congress", "actor": "congress", "effect": "flip_house",
                    "zh": "新国会（2027-01-03）：众议院易手，财政门槛与倾斜按分治政府"})
    if ch.get("courts") == "allow_removal" or ch.get("president") == "nominate_dovish":
        out.append({"rule": "chair_tilt", "actor": "fed", "effect": {"chair_tilt": -0.5, "lag_q": 1},
                    "zh": "主席倾向在确认或裁决后 1–2 季转向偏鸽（v1 规则）"})
    if ch.get("courts") == "structural" or facts.get("antitrust_structural"):
        out.append({"rule": "antitrust_payout", "actor": "bigtech", "effect": {"tilt": {"buyback_cut": 1.5}},
                    "zh": "结构性救济后下一季削减回购的先验 ×1.5（判断值）"})
    eq, dy = p.get("us_equity_q"), p.get("d_y10")
    if (eq is not None and eq <= math.log(0.9)) or (dy is not None and dy >= 0.4):
        out.append({"rule": "market_retreat", "actor": "president", "effect": {"allow_shift_to": ["tariff_down", "deal"]},
                    "zh": "上季美股 ≤ −10% 或 10 年期 +0.4pp：大模型可以把概率移向降级（2018-12 与 2025-04 两例）"})
    if facts.get("pandemic"):
        out.append({"rule": "pandemic_fiscal", "actor": "congress", "effect": {"prior": {"stimulus_large": 0.9}},
                    "zh": "疫情后下一季大规模刺激 0.9（一例，2020）"})
    if facts.get("financial_accident") and facts.get("systemic"):
        out.append({"rule": "systemic_backstop", "actor": "congress", "effect": {"prior": {"stimulus_large": 0.95}},
                    "zh": "系统性金融事故后的托底立法 ≈ 1（两例：TARP 型）"})
    if ch.get("bigtech_pricing") == "monetize" or facts.get("ai_monetize"):
        out.append({"rule": "monetize_capex", "actor": "bigtech", "effect": {"tilt": {"capex_raise_large": 1.5}},
                    "zh": "变现顺利：下一季上调资本开支的先验 ×1.5（判断值）"})
    return out


# =====================================================================================================================
# 9. Signals (其他人的意思): default texts and the condition enum
# =====================================================================================================================

SIGNAL_CONDITIONS = (None, "inflation_up", "inflation_down", "unemp_up_0.3", "equity_down_10", "spread_up_0.5",
                     "oil_up_20", "tariff_up_partner", "export_controls_up", "debt_limit_binding", "ruling_against",
                     "ceasefire", "escalation")
SIGNAL_INSTRUMENT_ROLE = {"fed": "policy_rate_chg", "president": "tariff_chg", "congress": "deficit_chg",
                          "bigtech": "hyperscaler_capex_g", "opec": "opec_supply_chg", "china": "gpr_chg",
                          "treasury": "bill_share_chg", "belligerents": "gpr_chg"}


def default_signal(actor_id: str, option_id: str, *, resolver=None) -> dict:
    """The structured public signal an option carries when the LLM writes none (LEDGERS §6.4)."""
    a = ACTORS[actor_id]
    _, o = a.find(option_id)
    role = SIGNAL_INSTRUMENT_ROLE.get(actor_id)
    r = _resolver(resolver)
    inst = r.field(role) if role else None
    return {"actor": actor_id, "option": option_id, "text": o.signal[0][:40], "instrument": inst or role,
            "direction": int(o.signal[1]), "horizon_q": int(min(max(o.signal[2], 1), 4)), "condition": None,
            "commit": False}


def validate_signal(sig: Mapping) -> tuple[dict, list[str]]:
    """Clean an LLM signal: text <= 40 chars, direction in {-1,0,1}, horizon 1-4, condition from the enum."""
    flags = []
    text = str(sig.get("text") or "")[:40]
    try:
        d = int(sig.get("direction", 0))
    except (TypeError, ValueError):
        d, _ = 0, flags.append("表态方向不是整数，按 0")
    if d not in (-1, 0, 1):
        flags.append(f"表态方向 {d} 不在 −1/0/1，按符号处理")
        d = (d > 0) - (d < 0)
    try:
        h = int(sig.get("horizon_q", 1))
    except (TypeError, ValueError):
        h = 1
    h = min(max(h, 1), 4)
    cond = sig.get("condition")
    if cond not in SIGNAL_CONDITIONS:
        flags.append(f"表态条件「{cond}」不在枚举里，已删除")
        cond = None
    return {**dict(sig), "text": text, "direction": d, "horizon_q": h, "condition": cond,
            "commit": bool(sig.get("commit", False))}, flags


# =====================================================================================================================
# 10. Catalogue views and validation
# =====================================================================================================================

FORBIDDEN_CLASSES = {"financial_price", "commodity_price", "output", "state", "anchor", "view"}


def validate_catalogue(resolver=None) -> list[str]:
    """Load-time checks (§3.2), Chinese messages: instruments have exactly one owner; no actor writes a price,
    output, state, anchor or view field (the declared oil event signature excepted); every catalogue lever resolves
    or falls back; every main domain has exactly one status-quo option; reaction rows sum to 1."""
    r = _resolver(resolver)
    problems = []
    owners: dict[str, set] = {}
    for a in ACTORS.values():
        for role in a.levers:
            if role not in r.roles:
                problems.append(f"{a.zh}的杠杆角色「{role}」不在角色表里")
                continue
            if r.roles[role].get("kind") == "instrument":
                owners.setdefault(role, set()).add(a.id)
        for d in a.domains:
            sq = [o for o in d.options if o.status_quo]
            if len(sq) != 1:
                problems.append(f"{a.zh}·{d.zh} 必须恰好有一个现状选项（现有 {len(sq)} 个）")
            for o in d.options:
                for w in o.writes:
                    if w.role not in a.levers and not (w.role == "it_invest_g" and a.id == "bigtech"):
                        problems.append(f"{a.zh}·{o.id} 写了不属于自己的杠杆「{w.role}」")
                    fid = r.field(w.role)
                    cls = (r.info.get(fid) or {}).get("cls") if (fid and r.info) else None
                    if cls in FORBIDDEN_CLASSES:
                        problems.append(f"{a.zh}·{o.id} 写了 {cls} 类字段 {fid}：行为者不能写价格、输出、状态、锚或视图")
                    if fid is None and not ROLES.get(w.role, {}).get("fallback") and a.active != "bill_share_field":
                        problems.append(f"{a.zh}·{o.id} 的杠杆「{w.role}」没有字段也没有后备")
                if o.param is not None and not (o.param.lo <= o.param.default <= o.param.hi):
                    problems.append(f"{a.zh}·{o.id} 的默认幅度不在范围内")
                # shared event channels: only the declared signature / event classes may receive contributions
                for role in (("oil_supply_signature", "oil_supply_shock") if o.oil_z else ()) + (
                        ("gpr_chg",) if o.gpr_sd else ()):
                    if role not in a.levers:
                        problems.append(f"{a.zh}·{o.id} 向事件通道「{role}」写入，但它不是自己的杠杆")
                    fid = r.field(role)
                    cls = (r.info.get(fid) or {}).get("cls") if (fid and r.info) else None
                    if cls in FORBIDDEN_CLASSES:
                        problems.append(f"{a.zh}·{o.id} 的事件贡献落在 {cls} 类字段 {fid}：只允许声明过的事件签名")
    for role, who in owners.items():
        if len(who) != 1:
            problems.append(f"工具「{role}」必须恰好有一个所有者（现有 {sorted(who)}）")
    for rr in REACTIONS:
        if rr["kind"] != "rows":
            continue
        for lo, row in rr["rows"].items():
            if abs(sum(row.values()) - 1.0) > 1e-9:
                problems.append(f"反应表 {rr['follower']} | {rr['leader']}={lo} 之和不是 1")
            fa, fd = rr["follower"]
            ids = {o.id for o in ACTORS[fa].domain(fd).options}
            for k in row:
                if k not in ids:
                    problems.append(f"反应表 {rr['follower']} 的选项「{k}」不在菜单里")
    return problems


def catalogue(resolver=None) -> dict:
    """The JSON view /api/agents/config shows: actors with kinds, stages, levers, domains and options."""
    r = _resolver(resolver)
    actors = []
    for aid in STAGE_ORDER:
        a = ACTORS[aid]
        actors.append({
            "id": a.id, "zh": a.zh, "kind": a.kind, "stage": a.stage, "active_rule": a.active, "pool_c": a.pool_c,
            "custom_cap": a.custom_cap, "book": a.book, "watches": list(a.watches), "naming": a.naming,
            "levers": [{"role": role, "field": r.field(role), "fallback": ROLES.get(role, {}).get("fallback")}
                       for role in a.levers],
            "prior_note": a.prior_note, "note": a.note,
            "domains": [{"id": d.id, "zh": d.zh, "role": d.role, "live": d.live, "max_options": d.max_options,
                         "prior": d.prior, "branch_when": d.branch_when, "note": d.note,
                         "options": [option_summary(a.id, o, r) for o in d.options]} for d in a.domains]})
    return {"version": CATALOGUE_VERSION, "resolver": r.hash(), "actors": actors,
            "nature": {"pool_c": NATURE_POOL_C, "tilt_bounds": list(HAZARD_TILT_BOUNDS),
                       "branch_min": HAZARD_BRANCH_MIN, "hazards": copy.deepcopy(HAZARDS)},
            "reactions": reaction_pairs(), "facts": copy.deepcopy(FACTS), "elections": copy.deepcopy(ELECTIONS),
            "oil_composite": dict(OIL_COMPOSITE), "s_bt": S_BT, "problems": validate_catalogue(r)}


def catalogue_hash(resolver=None) -> str:
    import hashlib
    return hashlib.sha256(jsonio.dumps(catalogue(resolver)).encode("utf-8")).hexdigest()[:16]
