"""The blind schema view for the translator (SIM_DESIGN §8.4, P0.5): scenariolab/qmodel/schema_view.json.

    python tools/schema_view.py            # writes schema_view.json
    python tools/schema_view.py --check    # exit 1 if it differs from a fresh build (or is not blind)

schema.json is not a translator reference: its templates and scenario rules carry dated realized footprints
(`measured`, `footprint_quarters`, 354 year tokens in the templates alone), so handing it over would let a translation
recall history. The view keeps only what a blind translation may use: field definitions (with every dated clause
removed), units, timing, endogeneity, permissions and financial-conditions roles from the model card, the generic
templates as channel SIGNS (no sizes, no footprints, no dated examples; era-identifying names made generic), and the
QTable vocabulary. It never contains a 19xx / 20xx token, a quarter label or `footprint_quarters`: a scan enforces it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenariolab.qmodel import card as qcard  # noqa: E402
from scenariolab.qmodel import config_q as cq  # noqa: E402

VIEW_ID = "schema_view/1 (blind)"
# template names that point at one era or one country become generic
BLIND_NAMES = {"G06": "通用技术资本开支繁荣与破裂", "G09": "去通胀政策冲击", "G14": "收入从劳动转向资本（自动化替代）",
               "G19": "需求推动的大宗商品繁荣", "G32": "长期停滞（零利率与低通胀）"}
FIELD = re.compile(r"(?<![A-Za-z0-9_])((?:d|x|u|s|a|y)_[a-z0-9_]+|wc_gap)")
PAREN = re.compile(r"[（(][^（）()]*[）)]")
CAUSE = {"供给": "supply", "需求": "demand"}
FORBIDDEN_KEYS = ("footprint_quarters", "measured", "latest", "coverage", "evidence", "vintage", "lit")


def dated(text: str) -> bool:
    return bool(qcard.YEAR_TOKEN.search(text) or qcard.QUARTER_TOKEN.search(text))


def scrub(text: str) -> str:
    """Remove every parenthetical, clause or sentence that names a year or a quarter; [refs] go too."""
    text = re.sub(r"\[[^\]]*\]", "", text or "")
    prev = None
    while prev != text:
        prev = text
        text = PAREN.sub(lambda m: "" if dated(m.group(0)) else m.group(0), text)
    parts = [p for p in re.split(r"(；|。)", text)]
    keep = []
    for i in range(0, len(parts), 2):
        clause = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if clause.strip() and not dated(clause):
            keep.append(clause.strip() + sep)
    out = "".join(keep).strip("；。 ")
    sub = [c for c in re.split(r"，", out) if not dated(c)]
    return "，".join(c.strip() for c in sub if c.strip())


def _sign(after: str) -> str:
    """The direction stated right after a field token in a template text (sizes are never kept). An operator between
    two fields ("d_y10 − d_infexp_lr") is not a direction; "0~+1 sd" reads as +."""
    a = after.lstrip()
    if a[:1] in "+＋−-" and FIELD.match(a[1:].lstrip()):
        return "?"
    if a.startswith("0~") or a.startswith("0～"):
        a = a[2:]
    if a.startswith("="):
        val = a[1:].lstrip()
        for zh, en in CAUSE.items():
            if val.startswith(zh):
                return en
        return "?"
    for token, sign in (("±", "±"), ("+", "+"), ("＋", "+"), ("−", "-"), ("-", "-"), ("↑", "+"), ("↓", "-"),
                        ("≈0", "0"), ("≈ 0", "0"), ("降到 0", "-"), ("在 0", "0"), ("0", "0"), ("先 +", "+-"),
                        ("由叙事写明", "narrative"), ("维持", "0"), ("不升", "0"), ("不降", "0")):
        if a.startswith(token):
            return sign
    return "?"


def template_signs(text: str, known: set[str]) -> list[dict]:
    """[{phase, signs: {field: sign}}] from a template's channel text (phases split at '｜')."""
    phases = []
    for part in (text or "").split("｜"):
        name, body = "", part
        head = part.split("：", 1)
        if len(head) == 2 and len(head[0]) <= 24 and not FIELD.search(head[0]):
            name, body = head[0].strip(), head[1]
        signs: dict[str, str] = {}
        for m in FIELD.finditer(body):
            f = m.group(1)
            if f in known and f not in signs:
                signs[f] = _sign(body[m.end():m.end() + 24])
        if signs:
            phases.append({"phase": "" if dated(name) else name, "signs": signs})
    return phases


def build_view(schema: dict, card: dict) -> dict:
    known = {r["id"] for r in card["rows"]}
    fields = []
    for r in card["rows"]:
        if r["group"] not in ("x", "u", "c", "s"):
            continue
        item = {"id": r["id"], "zh": r["zh"], "group": r["group"], "unit": r["unit"], "timing": r["timing"],
                "endogeneity": r["endo"], "translator": r["translator"], "financial_conditions": r["fc"] or None}
        if r["group"] in ("x", "u", "c"):
            item["definition"] = scrub(r["definition"])
        if r.get("enum"):
            item["enum"] = r["enum"]
        if r.get("keywords"):
            item["keywords"] = r["keywords"]
        fields.append(item)
    templates = []
    for t in schema["templates"]:
        templates.append({"id": t["id"], "name": BLIND_NAMES.get(t["id"], scrub(t["zh"])),
                          "phases": template_signs(t.get("x", ""), known)})
    view = {"view": VIEW_ID, "card": card["card"],
            "note": "盲测视图：只有定义、单位、时点、内生性、翻译器权限与模板的方向；没有日期、幅度、历史足迹",
            "fields": fields, "templates": templates,
            "never_inputs": sorted(r["id"] for r in card["rows"] if r["group"] in ("y", "v")),
            "vocabulary": {"size_bins_sd": cq.SIZE_BINS, "fc_stances": cq.FC_STANCES,
                           "sources_of_values": cq.SOURCES_OF_VALUES, "valuation_modes": cq.VALUATION_MODES,
                           "valuation_sizes": cq.VALUATION_SIZES, "regime_labels": cq.REGIME_LABELS,
                           "anchors": cq.QTABLE_ANCHORS, "quote_max_chars": cq.QUOTE_MAX}}
    return view


def blind_problems(view: dict) -> list[str]:
    """Why a view is not blind (empty = blind): year or quarter tokens anywhere, or a dated key."""
    text = json.dumps(view, ensure_ascii=False)
    out = [f"year token {y}" for y in sorted(set(qcard.year_tokens(text)))]
    out += [f"quarter token {q}" for q in sorted(set(qcard.QUARTER_TOKEN.findall(text)))]
    out += [f"key {k}" for k in FORBIDDEN_KEYS if f'"{k}"' in text]
    return out


def render(view: dict) -> str:
    return json.dumps(view, ensure_ascii=False, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--schema", type=Path, default=cq.SCHEMA_PATH)
    ap.add_argument("--card", type=Path, default=cq.CARD_PATH)
    ap.add_argument("--out", type=Path, default=cq.SCHEMA_VIEW_PATH)
    ap.add_argument("--check", action="store_true", help="compare with the existing view instead of writing it")
    args = ap.parse_args(argv)
    view = build_view(qcard.load_schema(args.schema), qcard.load_card(args.card))
    problems = blind_problems(view)
    if problems:
        print("[schema_view] the view is not blind: " + "; ".join(problems))
        return 1
    text = render(view)
    if args.check:
        same = args.out.exists() and args.out.read_text(encoding="utf-8") == text
        print(f"[schema_view] {args.out} {'is up to date' if same else 'differs from a fresh build'}")
        return 0 if same else 1
    args.out.write_text(text, encoding="utf-8")
    print(f"[schema_view] {len(view['fields'])} fields, {len(view['templates'])} templates -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
