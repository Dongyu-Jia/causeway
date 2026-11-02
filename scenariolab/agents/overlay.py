"""The agents overlay on the input table (AGENT_DESIGN §3.1; R-A1 of SIM_DESIGN §11.1): schema-revise-2 3.0 + the
agent tree's own keys, written to scenariolab/agents/overlay.json.

    python -m scenariolab.agents.overlay            # rebuild overlay.json from scenariolab/qmodel/schema.json
    python -m scenariolab.agents.overlay --check    # exit 1 when the committed file is stale

Per field: the schema's id, zh, unit, endo and entry, and the tree's keys

    cls             instrument | event | event_signature | event_label | quantity | financial_price |
                    commodity_price | survey_news | state | view | anchor | output (the permission matrix of §2.2)
    synth           none | shift | partition | both | partition_optional (the synthesizer's powers, §7.3)
    axis            the partition axis a field carries (growth_axis, fin_conditions_axis, …)
    channel_class   well | moderate | thin (CONSEQUENCES §4.2; 判断值) for the levers actors write
    lag_discipline  the lag-protected E0 set (§6.3)
    key             a state the node distances read
    display_only    shown to readers, never a model input nor read by priors, gates or distances
    enum            categorical fields as floats (x_oil_cause: demand 0, mixed 0.5, supply 1; the qdata convention)

What R-A1 changed against the schema-v2 overlay: x_oil_cause is the enum above (the oil-supply composite writes 1 =
supply when |Σz| ≥ 0.5), y_mult is Δlog CAPE multiple (price / (CPI × 10-year average real EPS); the equity-leg
partition is on it), s_ecyc (the earnings-cycle position log(Ê/E10)) and y_eps (Δlog Ê, write-downs included) are
new and display-only in the tree, and s_pe (log P/Ê) is display-only (write-downs distort it). d_tips10 (new in
3.0) is a financial price. Field metadata the model fits (hist) is not here: it comes from the adapter's
field_info(), which this overlay's keys override (merge()).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY_PATH = HERE / "overlay.json"
SCHEMA_PATH = HERE.parent / "qmodel" / "schema.json"
OVERLAY_VERSION = "agents-overlay-3 (2026-11-02; schema-revise-2 3.0, R-A1)"

INSTRUMENTS = ("d_ff", "d_bs", "d_tariff")
EVENTS = {"d_gpr": "event", "d_metals_oil": "event_signature", "x_oil_cause": "event_label"}
CLASS_OF_ENDO = {"E0": "quantity", "E0p": "quantity", "E1": "financial_price", "E1c": "commodity_price",
                 "E2": "survey_news", "S": "state", "V": "view", "C": "anchor", "O": "output"}
# §7.3 axes (synthesizer.AXES) and §2.2 shift powers (synthesizer.SHIFT_FIELDS minus the axes)
AXES = {"d_unrate": ("both", "growth_axis"), "d_capex": ("both", "investment_axis"),
        "d_cons_serv": ("both", "demand_axis"), "d_core": ("both", "inflation_axis"),
        "d_infexp1": ("both", "expectations_axis"), "x_supply_chain": ("both", "supply_axis"),
        "d_sent": ("both", "sentiment_axis"), "d_gz": ("partition", "fin_conditions_axis"),
        "d_cpbill": ("partition", "funding_axis"), "y_mult": ("partition_optional", "equity_leg_axis")}
SHIFT = ("d_itinv", "d_lprod", "d_lshare", "d_epu", "d_epu_health", "d_infexp_lr", "d_rgdp", "d_ip", "d_cpi_goods",
         "d_cpi_serv", "d_wage", "d_cons_goods", "d_resinv", "d_exports", "d_profits", "d_chargeoff")
CHANNEL = {"d_ff": "well", "d_bs": "moderate", "d_deficit": "moderate", "d_transfers": "moderate",
           "d_perstax": "moderate", "d_defense": "moderate", "d_corptax": "moderate", "d_metals_oil": "moderate",
           "d_gpr": "moderate", "d_capex": "moderate", "d_tariff": "thin", "d_itinv": "thin", "d_lf": "thin"}
LAG_DISCIPLINE = ("d_unrate", "d_rgdp", "d_core", "d_cpi_serv", "d_wage", "d_capex", "d_itinv", "d_cons_serv")
KEY = ("s_ff", "s_unrate", "s_core_yoy", "s_term", "s_spread", "s_cape", "s_pd")
DISPLAY_ONLY = {"s_pe": "滚动 P/E 被 2001–02、2008–09 的减记扭曲：只展示，不进模型，也不进先验、门控与节点距离",
                "y_eps": "滚动 EPS 的变化（含减记）：只展示；恒等式的盈利腿是 y_fund = Δlog CPI + Δlog E10 实际",
                "s_ecyc": "盈利周期位置：模型提供时只在观察包与界面展示（树不读它）"}
OIL_CAUSE = {"demand": 0.0, "mixed": 0.5, "supply": 1.0}
OIL_CAUSE_ZH = {"demand": "需求推动", "mixed": "供需混合", "supply": "供给推动"}
# the engine's view of the multiple leg (Draws.y["mult"]; the schema's y_mult)
MULT = {"zh": "估值倍数变化（美股 Δlog CAPE 倍数）", "unit": "log", "cls": "output", "synth": "partition_optional",
        "axis": "equity_leg_axis", "note": "价格 /（CPI × 10 年平均实际 EPS）的对数变化；取代 schema v2 的 Δlog P/Ê"}
AGENT_KEYS = ("cls", "synth", "axis", "channel_class", "lag_discipline", "key", "display_only", "enum", "enum_zh")


def keys_of(fid: str, endo: str) -> dict:
    """The tree's keys of one schema field by the rules above."""
    if fid in INSTRUMENTS:
        cls = "instrument"
    elif fid in EVENTS:
        cls = EVENTS[fid]
    else:
        cls = CLASS_OF_ENDO.get(endo, "quantity")
    synth, axis = AXES.get(fid, ("shift" if fid in SHIFT else "none", None))
    out = {"cls": cls, "synth": synth, "axis": axis, "channel_class": CHANNEL.get(fid),
           "lag_discipline": fid in LAG_DISCIPLINE, "key": fid in KEY, "display_only": fid in DISPLAY_ONLY}
    if fid == "x_oil_cause":
        out.update(enum=dict(OIL_CAUSE), enum_zh=dict(OIL_CAUSE_ZH))
    return out


def build(schema_path: Path | None = None) -> dict:
    """The overlay dict from schema.json (fields keyed by id; the aliases of actors.ALIASES)."""
    from .actors import ALIASES
    schema = json.loads(Path(schema_path or SCHEMA_PATH).read_text(encoding="utf-8"))
    fields = {}
    for f in schema["fields"]:
        fid = f["id"]
        row = {"zh": f.get("zh") or fid, "unit": f.get("unit") or "", "endo": f.get("endo"), "entry": f.get("entry"),
               "layer": f.get("layer"), **keys_of(fid, f.get("endo") or "")}
        if fid in DISPLAY_ONLY:
            row["note"] = DISPLAY_ONLY[fid]
        if fid == "y_mult":
            row["note"] = MULT["note"]
        fields[fid] = row
    counts: dict = {}
    for v in fields.values():
        counts[v["cls"]] = counts.get(v["cls"], 0) + 1
    return {"version": OVERLAY_VERSION, "schema_version": schema.get("version"), "fields": fields,
            "aliases": {k: v for k, v in ALIASES.items()}, "counts": dict(sorted(counts.items())),
            "built_by": "python -m scenariolab.agents.overlay"}


def write(path: Path | None = None, schema_path: Path | None = None) -> dict:
    ov = build(schema_path)
    p = Path(path or OVERLAY_PATH)
    p.write_text(json.dumps(ov, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return ov


_CACHE: dict = {}
_LOCK = threading.Lock()


def load(path: Path | None = None) -> dict:
    """The committed overlay (cached per file mtime); {} when it does not exist."""
    p = Path(path or OVERLAY_PATH)
    try:
        stamp = p.stat().st_mtime_ns
    except OSError:
        return {}
    with _LOCK:
        hit = _CACHE.get(str(p))
        if hit is not None and hit[0] == stamp:
            return hit[1]
    try:
        ov = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    with _LOCK:
        _CACHE[str(p)] = (stamp, ov)
    return ov


def merge(info: Mapping, overlay: Mapping | None = None) -> dict:
    """An adapter's field_info with the overlay's tree keys on top (zh, unit and hist stay the model's), plus the
    display-only entries the adapter lacks (s_pe, s_ecyc, y_eps) and the multiple leg as "mult"."""
    ov = (overlay if overlay is not None else load()).get("fields") or {}
    out = {k: dict(v) for k, v in info.items()}
    for fid, row in out.items():
        o = ov.get(fid)
        if not o:
            continue
        for k in AGENT_KEYS:
            if k in o:
                row[k] = o[k]
    for fid in DISPLAY_ONLY:
        if fid in ov and fid not in out:
            o = ov[fid]
            out[fid] = {"zh": o.get("zh", fid), "unit": o.get("unit", ""), "endo": o.get("endo"), "hist": None,
                        "role": "display", **{k: o[k] for k in AGENT_KEYS if k in o}, "note": o.get("note")}
    if "mult" not in out:
        out["mult"] = {**MULT, "endo": "O", "hist": None, "role": "y", "lag_discipline": False, "key": False,
                       "channel_class": None, "display_only": False}
    return out


def enum_label(fid: str, value, info: Mapping | None = None) -> str | None:
    """The Chinese label of a categorical field's float (x_oil_cause 0.93 -> 供给推动; nearest category)."""
    row = (info or {}).get(fid) or (load().get("fields") or {}).get(fid) or {}
    enum, zh = row.get("enum"), row.get("enum_zh") or {}
    if not enum or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    name = min(enum, key=lambda k: abs(float(enum[k]) - v))
    return zh.get(name, name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scenariolab.agents.overlay",
                                 description="重建智能体推演在输入表 3.0 上的覆盖表（overlay.json）")
    ap.add_argument("--check", action="store_true", help="只检查已提交的 overlay.json 是否与 schema.json 一致")
    args = ap.parse_args(argv)
    if args.check:
        want = build()
        have = load()
        stale = (have.get("fields") != want["fields"]) or (have.get("schema_version") != want["schema_version"])
        print("overlay.json " + ("需要重建" if stale else "与 schema.json 一致") + f"（{len(want['fields'])} 个字段）")
        return 1 if stale else 0
    ov = write()
    print(f"已写入 {OVERLAY_PATH}：{len(ov['fields'])} 个字段，{ov['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
