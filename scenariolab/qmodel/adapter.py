"""The quarterly stack as the agent tree's ModelAdapter (SIM_DESIGN §11.1) and as the instruments layer's core
(§11.2).

    MSL_AGENT_MODEL=scenariolab.qmodel.adapter:make_adapter      # the agent tree loads it (load_adapter)
    python -m scenariolab.qmodel.adapter --check                  # the contract check, the agent tree's own

The adapter is rollout.Stepper on the forward stack (rollout.get_stack: train end = the last data quarter, forward
anchors §3.8, origin = today's quarter, 2026Q4 on the 2026-09 panel). It follows the contract documented in the agent
tree's scenariolab/agents/model_adapter.py:

    version, stub = False, capabilities = {"structural", "rarity", "impute"}
    x_fields  the card's 52 x ids (card order); assets  config.ASSET_IDS order
    field_info()  {id: {zh, unit, endo, cls, synth, axis, channel_class, lag_discipline, key, hist{p01 p05 p50 p95
                  p99 min max sd_q sd_q_1985_2019}, ...}} for every x field and every public state key
    root_state(n)  the start-of-quarter state at the origin, n particles (copies; each with its own posterior draw
                  and asset-layer refit, balanced); REQUIRED_STATE and every "<x>_l1", "<asset>_l1"
    complete(state, given, rng, *, structural=None) -> Draws
    update_state(state, draws) -> state         the state roll Φ (rollout.Stepper.roll)
    rarity(state, given) -> (N,) m2             the §3.5 statistic (chi-square with df = the number of pins)
    impute(state, rng) -> state                 the root's unpublished fields redrawn from the joint distribution

Pins (Given). Values are in card units; x_oil_cause is 0 (demand), 0.5 (mixed) or 1 (supply). Every pin comes back
unchanged in Draws.x, with two bounded exceptions the field_info documents: a policy rate pinned below the floor is
delivered at 0.1 (d_ff then reports the delivered change), and unemployment or a spread pinned below the model's
bounds (u >= 2.05, spreads >= 0.11) is delivered at the bound. Completion conditions on the pinned values after the
factor layer's ±4 sd saturation (d_oil and d_y10 pass through). How a pin is read (config_q.STEP_GIVEN_MODES,
field_info()[f]["given_mode"]):

    reduced   an observation: the rest of the quarter is drawn from the model's conditional distribution
    do        an intervention on a policy field with a structural shock (d_ff -> u_ff, d_bs -> u_bs): same-quarter E0
              macro is not moved by it, markets respond on impact (the recursive identification of §5.3)
    do_last   set after the quarter's draw (d_tariff, d_lf: thin levers with no identified same-quarter effect)

Because the adapter is "structural", the tree drops its two-step protocol (lag-protected E0 pinned at A0): the do
reading of the Fed's instruments and the do_last reading of the thin levers are what replace it. structural={field:
values} marks further interventions explicitly (d_deficit, d_transfers, d_perstax by their shocks; any other
satellite do_last). u_* ids pin structural shocks in their field's units.

Draws. x: every x field (+ d_cpi, headline CPI, %年化) and "_q", the adapter's private per-particle columns that
update_state reads (a 2-D float array; take / concat carry it); y: quarterly LOG total returns of the 13 assets and
y["mult"] = Δlog CAPE multiple (R-A2); inc: simple income; cpi_q: the quarter's CPI log change; drawn (attribute,
R-A4): the historical quarter of each particle's asset residual vector; notes: saturation, floor and bound notes.

Instruments: core_view_draws(roll) is CoreView.from_quarterly's input (rollout.core_view_draws); core_view(roll)
builds the CoreView when the instruments package is importable (resid_drawn = the residual quarters, for pairing).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from . import card as qcard
from . import config_q as cq
from . import rollout as R

ADAPTER_VERSION = "1"
REQUIRED_STATE = ("s_ff", "s_unrate", "s_core_yoy", "s_spread", "s_term", "s_infexp1", "s_rstar", "s_sahm", "s_cape",
                  "d_ff_l1", "d_unrate_l1", "d_unrate_l2", "d_spread_l1", "d_itinv_l1", "d_gz_l1", "us_equity_l1",
                  "nasdaq_4q", "real_oil_d1", "real_oil_d2")          # the agent tree's REQUIRED_STATE
CAPABILITIES = frozenset({"structural", "rarity", "impute"})
HIST_KEYS = ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99")

# ---------------------------------------------------------------- the agent tree's containers (or replicas)
try:                                                    # the agent tree's own classes when it is merged
    from ..agents.model_adapter import Draws as _TreeDraws
    from ..agents.model_adapter import Given as _TreeGiven
except ImportError:                                     # pragma: no cover - depends on the merge state
    _TreeDraws = _TreeGiven = None


@dataclass
class Given:
    """Per-particle point constraints (the agent tree's Given): values[i, j] pins field ids[j] for particle i."""
    ids: tuple
    values: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "Given":
        return cls((), np.zeros((int(n), 0)))

    @classmethod
    def from_dict(cls, pins: Mapping, n: int) -> "Given":
        ids = tuple(sorted(pins))
        vals = np.full((int(n), len(ids)), np.nan)
        for j, k in enumerate(ids):
            vals[:, j] = np.broadcast_to(np.asarray(pins[k], float), (int(n),))
        return cls(ids, vals)

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    def to_dict(self) -> dict:
        return {k: self.values[:, j].copy() for j, k in enumerate(self.ids)}

    def take(self, idx) -> "Given":
        return Given(self.ids, self.values[np.asarray(idx, int)].copy())


@dataclass
class Draws:
    """One joint draw per particle (the agent tree's Draws, with the optional `drawn` of R-A4)."""
    x: dict
    y: dict
    inc: dict = field(default_factory=dict)
    cpi_q: np.ndarray | None = None
    drawn: np.ndarray | None = None

    @property
    def n(self) -> int:
        for d in (self.x, self.y):
            for v in d.values():
                return int(len(v))
        return 0

    def take(self, idx) -> "Draws":
        idx = np.asarray(idx, int)
        return Draws({k: v[idx].copy() for k, v in self.x.items()}, {k: v[idx].copy() for k, v in self.y.items()},
                     {k: v[idx].copy() for k, v in self.inc.items()},
                     None if self.cpi_q is None else self.cpi_q[idx].copy(),
                     None if self.drawn is None else np.asarray(self.drawn)[idx].copy())

    @staticmethod
    def concat(parts: list) -> "Draws":
        p0 = parts[0]
        cat = lambda key: {k: np.concatenate([getattr(p, key)[k] for p in parts]) for k in getattr(p0, key)}  # noqa: E731
        cpi = None if p0.cpi_q is None else np.concatenate([p.cpi_q for p in parts])
        drawn = None if getattr(p0, "drawn", None) is None else np.concatenate([p.drawn for p in parts])
        return Draws(cat("x"), cat("y"), cat("inc"), cpi, drawn)


DrawsClass = _TreeDraws or Draws
GivenClass = _TreeGiven or Given

# ---------------------------------------------------------------- the agents overlay's keys (AGENT_DESIGN §2.2, §6.3, §7.3)
# The tree reads cls / synth / axis / channel_class / lag_discipline / key from field_info. These are the agents
# overlay's values on schema v2 (the tree's stub_model.json carries them for the 50 shared fields); the overlay's own
# file wins when the agents package is importable, and R-A1 asks the agent round to rebuild it on schema 3.0.
INSTRUMENTS = ("d_ff", "d_bs", "d_tariff")
EVENT_FIELDS = {"d_gpr": "event", "d_metals_oil": "event_signature"}
AXES = {"d_sent": ("both", "sentiment_axis"), "x_supply_chain": ("both", "supply_axis"),
        "d_infexp1": ("both", "expectations_axis"), "d_unrate": ("both", "growth_axis"),
        "d_core": ("both", "inflation_axis"), "d_cons_serv": ("both", "demand_axis"),
        "d_capex": ("both", "investment_axis"), "d_cpbill": ("partition", "funding_axis"),
        "d_gz": ("partition", "fin_conditions_axis")}
SHIFT_FIELDS = ("d_itinv", "d_lprod", "d_lshare", "d_epu", "d_epu_health", "d_infexp_lr", "d_rgdp", "d_ip",
                "d_cpi_goods", "d_cpi_serv", "d_wage", "d_cons_goods", "d_resinv", "d_exports", "d_profits",
                "d_chargeoff")
CHANNEL = {"d_ff": "well", "d_bs": "moderate", "d_deficit": "moderate", "d_transfers": "moderate",
           "d_perstax": "moderate", "d_defense": "moderate", "d_corptax": "moderate", "d_metals_oil": "moderate",
           "d_gpr": "moderate", "d_capex": "moderate", "d_tariff": "thin", "d_itinv": "thin", "d_lf": "thin"}
LAG_DISCIPLINE = ("d_unrate", "d_rgdp", "d_core", "d_cpi_serv", "d_wage", "d_capex", "d_itinv", "d_cons_serv")
KEY_STATES = ("s_ff", "s_unrate", "s_core_yoy", "s_term", "s_spread", "s_infexp1", "s_infexp_lr", "s_rstar",
              "s_cape", "s_real_y10", "s_pd", "s_oil_real")
STATE_PANEL = {"nasdaq_4q": "m4_nasdaq", "s_tp10": "s_tp10"}       # public state key -> panel column for its hist


def agent_keys(fid: str, endo: str) -> dict:
    """The overlay's agent keys of one x field by the AGENT_DESIGN rules (the fallback table)."""
    if fid in INSTRUMENTS:
        cls = "instrument"
    elif fid in EVENT_FIELDS:
        cls = EVENT_FIELDS[fid]
    else:
        cls = {"E0": "quantity", "E0p": "quantity", "E1": "financial_price", "E1c": "commodity_price",
               "E2": "survey_news"}.get(endo, "quantity")
    synth, axis = AXES.get(fid, ("shift" if fid in SHIFT_FIELDS else "none", None))
    return {"cls": cls, "synth": synth, "axis": axis, "channel_class": CHANNEL.get(fid),
            "lag_discipline": fid in LAG_DISCIPLINE, "key": False}


def _tree_overlay() -> dict:
    """{field: agent keys} from the agent tree's own overlay (overlay.json, else stub_model.json) when importable."""
    try:
        from ..agents import actors as tree_actors
        from ..agents import model_adapter as tree_ma
    except ImportError:
        return {}
    for path in (getattr(tree_actors, "OVERLAY_PATH", None), getattr(tree_ma, "STUB_PATH", None)):
        try:
            if path is not None and path.exists():
                fields = json.loads(path.read_text(encoding="utf-8")).get("fields") or {}
                if isinstance(fields, list):
                    fields = {f["id"]: f for f in fields if isinstance(f, Mapping) and f.get("id")}
                return {k: {kk: v.get(kk) for kk in ("cls", "synth", "axis", "channel_class", "lag_discipline", "key")
                            if kk in v} for k, v in fields.items() if isinstance(v, Mapping)}
        except (OSError, ValueError, AttributeError):
            continue
    return {}


def _hist(series) -> dict | None:
    s = series.dropna()
    if len(s) < 3:
        return None
    q = {f"p{int(round(100 * p)):02d}": float(s.quantile(p)) for p in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90,
                                                                       0.95, 0.99)}
    q.update({"min": float(s.min()), "max": float(s.max()), "sd_q": float(s.std()), "n": int(len(s)),
              "first": str(s.index[0]), "last": str(s.index[-1])})
    w = s.loc["1985Q1":"2019Q4"] if hasattr(s.index, "freqstr") else s
    q["sd_q_1985_2019"] = float(w.std()) if len(w) > 2 else q["sd_q"]
    return q


# ================================================================ the adapter

class QAdapter:
    """The quarterly stack behind the agent tree's ModelAdapter contract (see the module docstring)."""

    stub = False
    capabilities = CAPABILITIES

    def __init__(self, stack: R.Stack | None = None, *, given_modes: Mapping | None = None, **step_kw):
        self.stack = stack or R.get_stack()
        self.stepper = R.Stepper(self.stack, **step_kw)
        self.given_modes = dict(cq.STEP_GIVEN_MODES if given_modes is None else given_modes)
        for f, mo in self.given_modes.items():
            if mo not in cq.PIN_MODES:
                raise ValueError(f"{f}：未知的固定方式 {mo!r}（{cq.PIN_MODES}）")
        self.x_fields = tuple(self.stack.x_fields)
        self.assets = tuple(cq.ASSETS)
        modes = hashlib.sha256(json.dumps(sorted(self.given_modes.items())).encode()).hexdigest()[:6]
        opts = hashlib.sha256(json.dumps(sorted((k, str(v)) for k, v in step_kw.items())).encode()).hexdigest()[:6]
        self.version = f"qadapter-{ADAPTER_VERSION}/{self.stack.version}/m{modes}/o{opts}"
        self._info: dict | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- metadata
    def field_info(self) -> dict:
        with self._lock:
            if self._info is None:
                self._info = self._field_info()
        return json.loads(json.dumps(self._info))

    def _field_info(self) -> dict:
        card = qcard.load_card()
        Q = self.stack.Q.loc[:self.stack.train_end]
        tree = _tree_overlay()
        sm = self.stack.system(-1)
        out = {}
        for f in self.x_fields:
            row = qcard.row(card, f)
            endo = row.get("endo_code") or ""
            info = {"zh": row.get("zh", f), "unit": row.get("unit", ""), "endo": endo, **agent_keys(f, endo),
                    "role": "x", "first": row.get("first"), "permission": row.get("permission"),
                    "hist": _hist(Q[f].astype(float)) if f in Q.columns else None,
                    "given_mode": self.given_modes.get(f, "reduced")}
            if f in cq.DO_FIELDS:
                info["structural_shock"] = cq.DO_FIELDS[f]
            if f == "d_ff":
                info["level_floor"] = cq.FF_FLOOR
                info["note"] = "政策利率不低于 0.1%：钉在下限以下的值按下限交付（d_ff 返回交付值）"
            if f == "d_unrate":
                info["level_floor"] = cq.PIN_U_MIN
            if f in ("d_spread", "d_gz"):
                info["level_floor"] = cq.PIN_SPREAD_MIN
            if f == "x_oil_cause":
                info["model_value"] = dict(cq.OIL_CAUSE)
                info["note"] = "Given 用浮点数：0 = demand（需求）、0.5 = mixed、1 = supply（供给）"
            if f in cq.MECHANICAL_FIELDS:
                info["saturation"] = "none (mechanical own-price input)"
            info.update({k: v for k, v in (tree.get(f) or {}).items() if v is not None or k in ("axis",
                                                                                                 "channel_class")})
            out[f] = info
        for u, var in cq.POLICY_DO.items():
            if var in self.stack.factors.names:
                p = self.stack.factors.names.index(var)
                out[u] = {"zh": qcard.row(card, u).get("zh", u), "unit": qcard.row(card, u).get("unit", ""),
                          "endo": "E0p", "cls": "shock", "synth": "none", "axis": None, "channel_class": None,
                          "lag_discipline": False, "key": False, "role": "shock",
                          "sd_model": float(sm["P"][p, p]), "hist": None}
        root = self.stepper.root(1)
        for k in sorted(k for k in root if not k.startswith("_") and (k.startswith("s_") or k in R.PUBLIC_STATES)):
            try:
                row = qcard.row(card, k)
            except KeyError:
                row = {}
            col = STATE_PANEL.get(k, k)
            out[k] = {"zh": row.get("zh") or R.STATE_ZH.get(k, k), "unit": row.get("unit", ""), "endo": "S",
                      "cls": "state", "synth": "none", "axis": None, "channel_class": None, "lag_discipline": False,
                      "key": k in KEY_STATES, "role": "s",
                      "hist": _hist(self.stack.Q[col].astype(float)) if col in self.stack.Q.columns else None}
        return out

    def sd(self, fid: str) -> float:
        h = (self.field_info().get(fid) or {}).get("hist") or {}
        v = h.get("sd_q_1985_2019") or h.get("sd_q")
        return float(v) if v else 1.0

    # ---------------------------------------------------------------- states
    def root_state(self, n: int) -> dict:
        return self.stepper.root(int(n))

    def impute(self, state: dict, rng: np.random.Generator) -> dict:
        return self.stepper.impute(state, rng)

    # ---------------------------------------------------------------- the quarter
    def _pins(self, given, n: int) -> tuple[dict, dict]:
        pins, extra = {}, {}
        if given is not None and len(getattr(given, "ids", ())):
            vals = np.asarray(given.values, float)
            for j, fid in enumerate(given.ids):
                (pins if self.stepper.fields.known(fid) else extra)[fid] = vals[:, j]
        return pins, extra

    def complete(self, state: dict, given, rng: np.random.Generator, *, structural: dict | None = None):
        """One joint draw per particle of every x field and asset return for the coming quarter (see the module
        docstring for how pins are read). Ids the model does not carry pass through into x unconditioned."""
        n = len(np.asarray(state["s_ff"]))
        pins, extra = self._pins(given, n)
        d = self.stepper.draw(state, pins, rng, structural=structural, modes=self.given_modes)
        x = dict(d.x)
        for k, v in extra.items():
            x[k] = np.asarray(v, float).copy()
        x["_q"] = d.priv
        out = DrawsClass(x, d.y, d.inc, d.cpi_q)
        try:
            out.drawn = d.drawn
            out.notes = d.notes + ([{"fields": sorted(extra), "text": "模型没有这些字段：原样返回，未作条件"}]
                                   if extra else [])
        except AttributeError:                       # pragma: no cover - a slotted container
            pass
        return out

    def update_state(self, state: dict, draws) -> dict:
        return self.stepper.roll(state, draws)

    def rarity(self, state: dict, given, *, structural: dict | None = None) -> np.ndarray:
        """m2 per particle: the squared Mahalanobis size of the structural adjustment the pins need (chi-square with
        df = the number of pins under the model; rarity_detail has df, the percentile and sd per constraint)."""
        return self.rarity_detail(state, given, structural=structural)["m2"]

    def rarity_detail(self, state: dict, given, *, structural: dict | None = None) -> dict:
        n = len(np.asarray(state["s_ff"]))
        pins, _ = self._pins(given, n)
        return self.stepper.rarity(state, pins, structural=structural, modes=self.given_modes)


_ADAPTER: QAdapter | None = None
_ADAPTER_LOCK = threading.Lock()


def make_adapter() -> QAdapter:
    """The factory MSL_AGENT_MODEL names: the adapter on the cached forward stack (fitted once per process)."""
    global _ADAPTER
    with _ADAPTER_LOCK:
        if _ADAPTER is None or _ADAPTER.stack is not R.get_stack():
            _ADAPTER = QAdapter(R.get_stack())
        return _ADAPTER


# ================================================================ the contract check

def validate_adapter(adapter, n: int = 64, seed: int = 0) -> list[str]:
    """The agent tree's validate_adapter (scenariolab/agents/model_adapter.py), replicated with the same checks and
    messages for when the agents package is not importable; empty when the adapter satisfies the contract."""
    try:
        from ..agents.model_adapter import validate_adapter as tree_validate
    except ImportError:
        tree_validate = None
    if tree_validate is not None:                    # pragma: no cover - depends on the merge state
        return tree_validate(adapter, n=n, seed=seed)
    problems: list[str] = []
    for attr in ("version", "stub", "capabilities", "x_fields", "assets"):
        if not hasattr(adapter, attr):
            problems.append(f"缺少属性 {attr}")
    for meth in ("field_info", "root_state", "complete", "update_state"):
        if not callable(getattr(adapter, meth, None)):
            problems.append(f"缺少方法 {meth}()")
    if problems:
        return problems
    unknown = set(adapter.capabilities) - CAPABILITIES
    if unknown:
        problems.append(f"未知的能力 {sorted(unknown)}")
    try:
        s0 = adapter.root_state(n)
        miss = [k for k in REQUIRED_STATE if k not in s0]
        if miss:
            problems.append("起点状态缺少：" + "、".join(miss))
        bad = [k for k, v in s0.items() if np.asarray(v).shape[:1] != (n,)]
        if bad:
            problems.append("状态数组的第一维不是粒子数：" + "、".join(bad[:5]))
        rng = np.random.default_rng(seed)
        pin_f = adapter.x_fields[0]
        g = GivenClass.from_dict({pin_f: np.full(n, 0.123)}, n)
        d = adapter.complete(s0, g, rng)
        if not np.allclose(d.x.get(pin_f, np.nan), 0.123):
            problems.append(f"固定的字段 {pin_f} 没有原样返回")
        miss_x = [f for f in adapter.x_fields if f not in d.x]
        if miss_x:
            problems.append("补全结果缺少 x 字段：" + "、".join(miss_x[:5]))
        miss_y = [a for a in adapter.assets if a not in d.y]
        if miss_y:
            problems.append("补全结果缺少资产回报：" + "、".join(miss_y[:5]))
        d2 = adapter.complete(s0, g, np.random.default_rng(seed))
        if not all(np.array_equal(d.y[a], d2.y[a]) for a in adapter.assets if a in d.y and a in d2.y):
            problems.append("同一种子的两次补全结果不同（必须可复现）")
        s1 = adapter.update_state(s0, d)
        miss1 = [k for k in REQUIRED_STATE if k not in s1]
        if miss1:
            problems.append("滚动后的状态缺少：" + "、".join(miss1))
    except Exception as err:  # noqa: BLE001 - reported, not raised
        problems.append(f"调用失败：{type(err).__name__}: {err}")
    return problems


# ================================================================ instruments (CoreView.from_quarterly)

def core_view_draws(roll: R.Rollout, **kw) -> dict:
    """CoreView.from_quarterly's input for a path-mode run (rollout.core_view_draws)."""
    return R.core_view_draws(roll, **kw)


def core_view(roll: R.Rollout, *, scenario_id: str = "", set_id: str = "quarterly"):
    """The instruments layer's CoreView of a path-mode run, with resid_drawn = the residual quarters (pairing, D16;
    from_quarterly does not take it yet). Needs scenariolab.instruments (feat/instruments)."""
    try:
        from ..instruments.core import CoreView
    except ImportError as err:
        raise ImportError("标的层（scenariolab.instruments，feat/instruments）还没有合并：用 core_view_draws() 的字典") \
            from err
    cv = R.core_view_draws(roll)
    core = CoreView.from_quarterly(cv, labels=cv["labels"], scenario_id=scenario_id, set_id=set_id)
    return dataclasses.replace(core, resid_drawn=np.asarray(cv["drawn"]).astype(str))


def correlations(y_us: np.ndarray, d_gz: np.ndarray, d_spread: np.ndarray) -> dict:
    """T18 / F5: per-path quarterly corr(d_gz, us_equity) and corr(d_spread, us_equity) (median and [p05, p95]), and
    the pooled ones, from (n, T) arrays."""
    def per_path(a, b):
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
        return np.where(den > 0, (a * b).sum(axis=1) / np.where(den > 0, den, 1.0), np.nan)

    out = {}
    for k, v in (("d_gz", d_gz), ("d_spread", d_spread)):
        c = per_path(np.asarray(v, float), np.asarray(y_us, float))
        out[k] = {"median": float(np.nanmedian(c)), "p05": float(np.nanpercentile(c, 5)),
                  "p95": float(np.nanpercentile(c, 95)),
                  "pooled": float(np.corrcoef(np.ravel(v), np.ravel(y_us))[0, 1])}
    return out


# ================================================================ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="季度模型的智能体推演适配器（试运行）")
    ap.add_argument("--check", action="store_true", help="run the agent tree's contract check")
    ap.add_argument("-n", type=int, default=64)
    args = ap.parse_args(argv)
    ad = make_adapter()
    print(f"[qmodel.adapter] {ad.version}  origin {ad.stack.origin}  x_fields {len(ad.x_fields)}")
    if args.check:
        problems = validate_adapter(ad, n=args.n)
        print("contract: " + ("ok" if not problems else "; ".join(problems)))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["QAdapter", "make_adapter", "validate_adapter", "Given", "Draws", "DrawsClass", "GivenClass",
           "core_view_draws", "core_view", "correlations", "agent_keys", "REQUIRED_STATE", "CAPABILITIES"]
