"""What a QTable states, quarter by quarter, in words (SIM_DESIGN §10.3, §9 section 3; §14 P8).

The scenario detail, the lab and the forecast tab show, for each quarter and each of the main channels, either what
the text stated (with the badge of its source) or that the model completed it:

    文字数字  src "stated": a number in the text          文字定性  src "stated_qual": words mapped to a size bin
    推算      src "derived": arithmetic on stated numbers   模型补全  nothing stated: the joint model completes it
    锚        a long-run assumption (c_q.anchors)           干预      a policy shock u_* (the do-operator)

Constraints and announced paths (c_q) carry numbers with a quote but no src: they count as 文字数字. A regime request
is words (文字定性). The financial-conditions stance is words unless 未说明 (then the model completes the credit
spreads, warning W-FC). No valuation statement means mode inferred: the multiple does what the macro and credit path
imply, which is the model's completion (估值假设 always reads 「相当于直接假设股价」).

    cells(qt, H=None, card=None) -> [{"q": q, "cells": {channel: {"items": [...], "completed": bool}}}]
    CHANNELS                      the channels in display order: key, Chinese name, the run summary's series

The text never shows code ids: field names come from the card's Chinese names. Pure function of the (clean) QTable;
nothing here runs the model, calls an LLM or downloads anything.
"""
from __future__ import annotations

import math

from . import card as qcard

BADGES = {"stated": "文字数字", "stated_qual": "文字定性", "derived": "推算", "completed": "模型补全",
          "anchor": "锚", "intervention": "干预"}

# (key, 中文名, x fields, shocks, c_q fields, the run summary's series for the completed median: (group, id, unit))
CHANNELS = [
    {"key": "policy", "zh": "政策利率", "x": ("d_ff",), "u": ("u_ff",), "c": ("d_ff", "s_ff"),
     "series": ("levels", "ff", "%")},
    {"key": "growth", "zh": "增长与就业", "x": ("d_ip", "d_rgdp", "d_unrate"), "u": (),
     "c": ("d_ip", "d_rgdp", "d_unrate", "s_unrate"), "series": ("levels", "unrate", "%"), "regime": True},
    {"key": "inflation", "zh": "通胀", "x": ("d_cpi_goods", "d_cpi_serv", "d_core"), "u": (),
     "c": ("d_cpi_goods", "d_cpi_serv", "d_core", "s_cpi_yoy", "s_core_yoy"), "series": ("levels", "cpi_yoy", "%"),
     "regime_infl": True},
    {"key": "oil", "zh": "油价", "x": ("d_oil", "x_oil_cause"), "u": (), "c": ("d_oil",),
     "series": ("factors", "d_oil", "log%")},
    {"key": "fc", "zh": "金融条件", "x": ("d_gz", "d_spread"), "u": (), "c": ("d_gz", "d_spread"),
     "series": ("levels", "spread", "%"), "stance": True},
    {"key": "rates", "zh": "10 年期收益率", "x": ("d_y10", "d_tp10", "d_y2"), "u": ("u_bs",),
     "c": ("d_y10", "d_tp10", "d_y2"), "series": ("levels", "y10", "%")},
    {"key": "valuation", "zh": "估值", "x": (), "u": (), "c": (), "series": ("valuation", "y_mult", "log"),
     "valuation": True},
]
MAIN_X = {f for ch in CHANNELS for f in ch["x"]} | {u for ch in CHANNELS for u in ch["u"]}
LEVEL_ZH = {"s_ff": "政策利率水平", "s_unrate": "失业率水平", "s_cpi_yoy": "CPI 同比", "s_core_yoy": "核心 CPI 同比"}
UNIT_ZH = {"pp/季": " 个百分点/季", "log%/季": "%（对数）/季", "%年化": "%（年化）", "指数点/季": " 点/季",
           "pp GDP/季": " 个百分点（占 GDP）/季", "%": "%", "pp": " 个百分点", "指数/季": "（指数）/季", "0/1": "",
           "扩散指数": "（扩散指数）"}
LEVEL_UNIT = {"s_ff": "%", "s_unrate": "%", "s_cpi_yoy": "%", "s_core_yoy": "%"}
KIND_ZH = {"mean": "平均", "sum": "合计", "level": "水平", "yoy": "同比", "range": "范围"}
OIL_CAUSE_ZH = {"supply": "供给冲击", "demand": "需求推动", "mixed": "供给与需求混合"}
REGIME_ZH = {"expansion": "扩张", "inflation": "通胀", "deflation": "物价负增长", "recession": "衰退",
             "depression": "深度衰退", "stagflation": "滞胀", "contraction": "收缩", "high_infl": "高通胀"}
REGIME_GROWTH = ("expansion", "recession", "depression", "stagflation", "contraction")
REGIME_INFL = ("inflation", "deflation", "stagflation", "high_infl", "expansion")
ANCHOR_ZH = {"r_star": "中性实际利率 r*", "pi_lr": "长期通胀", "a_tp": "期限溢价锚", "a_spread": "信用利差锚",
             "a_ustar": "失业率锚", "a_oil": "实际油价锚移位", "a_usd": "实际美元锚移位"}
VAL_WORDING = "相当于直接假设股价"


def _names(card: dict | None) -> dict:
    c = card or qcard.load_card()
    return {r["id"]: r for r in qcard.rows(c)}


def _g(v) -> str:
    """A number without trailing zeros, at most 2 decimals: 0.5 -> "0.5", 18.5 -> "18.5", -0.125 -> "-0.13"."""
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return "—"
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _signed(v) -> str:
    t = _g(v)
    return t if t.startswith("-") or t in ("0", "—") else f"+{t}"


def _span(qs) -> str:
    qs = sorted(int(q) for q in qs)
    if not qs:
        return ""
    out, a, b = [], qs[0], qs[0]
    for q in qs[1:]:
        if q == b + 1:
            b = q
            continue
        out.append(f"{a}" if a == b else f"{a}–{b}")
        a = b = q
    out.append(f"{a}" if a == b else f"{a}–{b}")
    return "、".join(out)


def _zh(fid: str, names: dict) -> str:
    if fid in LEVEL_ZH:
        return LEVEL_ZH[fid]
    row = names.get(fid)
    return (row or {}).get("zh") or fid


def _unit(fid: str, names: dict) -> str:
    if fid in LEVEL_UNIT:
        return LEVEL_UNIT[fid]
    u = (names.get(fid) or {}).get("unit") or ""
    return UNIT_ZH.get(u, f" {u}" if u else "")


def entry_text(fid: str, e: dict, names: dict) -> str:
    """One QTable entry in words: 「政策利率变动 +0.5 个百分点/季」, 「商品通胀上升（明显）」, 「失业率变动 -0.2 到 0.3 …」."""
    zh = _zh(fid, names)
    if fid == "x_oil_cause":
        return f"油价变动的原因：{OIL_CAUSE_ZH.get(str(e.get('value')), str(e.get('value')))}"
    unit = _unit(fid, names)
    if "sign" in e:
        word = {"+": "上升", "-": "下降", "0": "不变"}.get(str(e["sign"]), str(e["sign"]))
        return f"{zh}{word}（{e.get('size', '—')}）" if word != "不变" else f"{zh}不变"
    if "value" in e:
        sd = e.get("sd")
        return f"{zh} {_signed(e['value']) if fid.startswith(('d_', 'u_')) else _g(e['value'])}{unit}" \
               + (f"（±{_g(sd)}）" if isinstance(sd, (int, float)) else "（硬约束）")
    lo, hi = e.get("min"), e.get("max")
    if lo is not None and hi is not None:
        return f"{zh}在 {_g(lo)} 到 {_g(hi)}{unit} 之间"
    if lo is not None:
        return f"{zh}不低于 {_g(lo)}{unit}"
    if hi is not None:
        return f"{zh}不高于 {_g(hi)}{unit}"
    return zh


def constraint_text(c: dict, names: dict) -> str:
    """A c_q constraint or announced path in words, e.g. 「CPI 同比 第 3–6 季平均 4.5%（±0.22）」."""
    fid = c.get("field", "")
    zh, unit = _zh(fid, names), _unit(fid, names)
    span = _span(c.get("quarters") or [])
    if "values" in c:                                         # an announced path
        vals = "、".join(_g(v) for v in c.get("values") or [])
        return f"预告路径：{zh} 第 {span} 季依次 {vals}{unit}"
    kind = c.get("kind", "mean")
    sd = c.get("sd")
    tail = f"（±{_g(sd)}）" if isinstance(sd, (int, float)) else ""
    if kind == "range" or ("value" not in c and ("min" in c or "max" in c)):
        lo, hi = c.get("min"), c.get("max")
        rng = (f"在 {_g(lo)} 到 {_g(hi)}{unit} 之间" if lo is not None and hi is not None
               else f"不低于 {_g(lo)}{unit}" if lo is not None else f"不高于 {_g(hi)}{unit}")
        return f"{zh} 第 {span} 季{KIND_ZH.get(kind, '') if kind != 'range' else ''}{rng}{tail}"
    return f"{zh} 第 {span} 季{KIND_ZH.get(kind, kind)} {_g(c.get('value'))}{unit}{tail}"


def _item(badge: str, text: str, quote=None, **extra) -> dict:
    out = {"badge": badge, "badge_zh": BADGES[badge], "text": text}
    if quote:
        out["quote"] = str(quote)
    out.update(extra)
    return out


def _covers(c: dict, q: int) -> bool:
    return q in {int(x) for x in (c.get("quarters") or [])}


def valuation_text(v: dict) -> str:
    mode = v.get("mode")
    if mode == "hold":
        what = "倍数保持不变"
    elif mode == "rerate":
        what = f"倍数{'上调' if v.get('direction') == '+' else '下调'}（{v.get('size', '—')}）"
    elif mode == "inferred":
        what = "由宏观与信用隐含"
    else:
        what = str(mode)
    hl = v.get("reversion_half_life_years")
    if isinstance(hl, (int, float)):
        what += f"，向历史中位数回归（半衰期 {_g(hl)} 年）"
    return f"估值假设：{what}（{VAL_WORDING}）" if mode != "inferred" else f"估值：{what}"


def cells(qt: dict, H: int | None = None, card: dict | None = None) -> list[dict]:
    """Per quarter 1..H (default: the QTable's n_quarters), the stated items of each channel with their badges, the
    other stated fields, and the anchors in force; a channel with no item is completed by the model."""
    names = _names(card)
    rows = {int(r.get("q")): r for r in qt.get("rows") or [] if isinstance(r, dict) and isinstance(r.get("q"), int)}
    H = int(H or qt.get("n_quarters") or (max(rows) if rows else 0))
    cqx = qt.get("c_q") or {}
    cons = list(cqx.get("constraints") or []) + list(cqx.get("announced") or [])
    regime = cqx.get("regime") or []
    vals = cqx.get("valuation") or []
    anchors = cqx.get("anchors") or {}
    out = []
    for q in range(1, H + 1):
        r = rows.get(q) or {}
        x, u = r.get("x") or {}, r.get("u") or {}
        fc = r.get("fc") or {"stance": "未说明"}
        q_cells = {}
        for ch in CHANNELS:
            items = []
            for f in ch["x"]:
                if f in x and isinstance(x[f], dict):
                    items.append(_item(x[f].get("src") if x[f].get("src") in BADGES else "stated",
                                       entry_text(f, x[f], names), x[f].get("quote"), field=f))
            for s in ch["u"]:
                if s in u and isinstance(u[s], dict):
                    items.append(_item("intervention", f"干预：{entry_text(s, u[s], names)}", u[s].get("quote"),
                                       field=s, src=u[s].get("src")))
            for c in cons:
                if c.get("field") in ch["c"] and _covers(c, q):
                    items.append(_item("stated", constraint_text(c, names), c.get("quote"), field=c.get("field"),
                                       constraint=True))
            if ch.get("regime") or ch.get("regime_infl"):
                labels = REGIME_GROWTH if ch.get("regime") else REGIME_INFL
                for rg in regime:
                    if rg.get("label") in labels and _covers(rg, q):
                        items.append(_item("stated_qual", f"体制请求：{REGIME_ZH.get(rg['label'], rg['label'])}"
                                                          f"占 {round(100 * float(rg.get('share', 0)))}%",
                                           rg.get("quote"), regime=rg.get("label")))
            if ch.get("stance"):
                st = fc.get("stance", "未说明")
                if st != "未说明":
                    items.insert(0, _item("stated_qual", f"金融条件：{st}", fc.get("quote"), stance=st))
            if ch.get("valuation"):
                for v in vals:
                    if _covers(v, q):
                        items.append(_item("stated_qual", valuation_text(v), v.get("quote"), mode=v.get("mode")))
            cell = {"items": items, "completed": not items}
            if ch.get("stance"):
                cell["stance"] = fc.get("stance", "未说明")
            if ch.get("valuation") and not items:
                cell["mode"] = "inferred"
                cell["text"] = valuation_text({"mode": "inferred"})
            q_cells[ch["key"]] = cell
        other = [_item(e.get("src") if e.get("src") in BADGES else "stated", entry_text(f, e, names), e.get("quote"),
                       field=f)
                 for f, e in sorted(x.items()) if f not in MAIN_X and isinstance(e, dict)]
        other += [_item("intervention", f"干预：{entry_text(s, e, names)}", e.get("quote"), field=s)
                  for s, e in sorted(u.items()) if s not in MAIN_X and isinstance(e, dict)]
        other += [_item("stated", constraint_text(c, names), c.get("quote"), field=c.get("field"), constraint=True)
                  for c in cons if _covers(c, q) and not any(c.get("field") in ch["c"] for ch in CHANNELS)]
        anchor_items = [_item("anchor", f"{ANCHOR_ZH.get(a, a)} {_g(v.get('value'))}", v.get("quote"), anchor=a)
                        for a, v in sorted(anchors.items()) if isinstance(v, dict) and int(v.get("from_q", 1)) <= q]
        out.append({"q": q, "cells": q_cells, "other": other, "anchors": anchor_items})
    return out


def channels() -> list[dict]:
    """The channels for the page: key, Chinese name, the run summary series that holds the completed median."""
    return [{"key": ch["key"], "zh": ch["zh"], "series": list(ch["series"])} for ch in CHANNELS]
