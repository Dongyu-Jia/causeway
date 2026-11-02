"""d04 -- the QTable JSON schema (SIM_DESIGN Appendix B), generated from schema.json so the LLM's output schema equals
the model's input schema. Read-only except this folder's work/. Run: <venv python> -I -B d04_qtable_schema.py <schema.json> <outdir>
"""
import json, sys
from pathlib import Path
schema = json.load(open(sys.argv[1])); out = Path(sys.argv[2])
F = [f for f in schema["fields"] if f["layer"] in ("L1", "L2") and f["endo"] != "V"]
X_IDS = sorted(f["id"] for f in F if f["id"] != "x_oil_cause")
E12 = sorted(f["id"] for f in F if f["endo"] in ("E1", "E1c", "E2"))
U_IDS = ["u_ff", "u_bs", "u_def", "u_tr", "u_ptax"]
ANCHORS = ["r_star", "pi_lr", "a_tp", "a_spread", "a_ustar", "a_oil", "a_usd"]
SIZES = ["小", "温和", "明显", "大幅", "极端"]          # 0.25 / 0.5 / 1 / 2 / 3 training sd of the field's quarterly change
SRC = ["stated", "stated_qual", "derived"]            # template signs are not inputs; inferred values never appear
entry = {"type": "object", "additionalProperties": False,
         "properties": {"value": {"type": "number"}, "sd": {"type": "number", "exclusiveMinimum": 0},
                        "sign": {"enum": ["+", "-", "0"]}, "size": {"enum": SIZES},
                        "min": {"type": "number"}, "max": {"type": "number"},
                        "src": {"enum": SRC}, "quote": {"type": "string", "maxLength": 120}},
         "required": ["src", "quote"]}
row = {"type": "object", "additionalProperties": False,
       "properties": {"q": {"type": "integer", "minimum": 1},
                      "x": {"type": "object", "additionalProperties": False,
                            "properties": {**{k: entry for k in X_IDS},
                                           "x_oil_cause": {"type": "object", "additionalProperties": False,
                                                           "properties": {"value": {"enum": ["supply", "demand", "mixed"]},
                                                                          "src": {"enum": ["stated", "stated_qual"]},
                                                                          "quote": {"type": "string", "maxLength": 120}},
                                                           "required": ["value", "src", "quote"]}}},
                      "u": {"type": "object", "additionalProperties": False, "properties": {k: entry for k in U_IDS}},
                      "fc": {"type": "object", "additionalProperties": False,
                             "properties": {"stance": {"enum": ["明显收紧", "收紧", "中性", "宽松", "明显宽松", "未说明"]},
                                            "quote": {"type": "string", "maxLength": 120}},
                             "required": ["stance"]}},
       "required": ["q", "x", "fc"]}
qtable = {
  "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "QTable/1",
  "description": "季度情景表：LLM 的输出 = 季度模型的输入。null/缺省 = 文字没有给（模型补全）。",
  "type": "object", "additionalProperties": False,
  "properties": {
    "schema": {"const": "qtable/1"}, "model_card": {"type": "string"},
    "n_quarters": {"type": "integer", "minimum": 1, "maximum": 60},
    "rows": {"type": "array", "items": row},
    "c_q": {"type": "object", "additionalProperties": False, "properties": {
        "anchors": {"type": "object", "additionalProperties": False,
                    "properties": {a: {"type": "object", "additionalProperties": False,
                                       "properties": {"value": {"type": "number"}, "from_q": {"type": "integer", "minimum": 1},
                                                      "quote": {"type": "string", "maxLength": 120}},
                                       "required": ["value", "quote"]} for a in ANCHORS}},
        "announced": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "field": {"enum": X_IDS}, "quarters": {"type": "array", "items": {"type": "integer"}},
            "values": {"type": "array", "items": {"type": ["number", "null"]}}, "sd": {"type": "number"},
            "quote": {"type": "string", "maxLength": 120}}, "required": ["field", "quarters", "quote"]}},
        "constraints": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "kind": {"enum": ["sum", "mean", "yoy", "level", "range"]}, "field": {"enum": X_IDS + ["s_cpi_yoy", "s_core_yoy", "s_unrate", "s_ff"]},
            "quarters": {"type": "array", "items": {"type": "integer"}}, "value": {"type": "number"}, "sd": {"type": "number"},
            "min": {"type": "number"}, "max": {"type": "number"}, "quote": {"type": "string", "maxLength": 120}},
            "required": ["kind", "field", "quarters", "quote"]}},
        "valuation": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "quarters": {"type": "array", "items": {"type": "integer"}}, "mode": {"enum": ["inferred", "hold", "rerate"]},
            "direction": {"enum": ["+", "-"]}, "size": {"enum": ["温和", "明显", "极端"]},
            "reversion_half_life_years": {"type": "number", "minimum": 14}, "quote": {"type": "string", "maxLength": 120}},
            "required": ["quarters", "mode", "quote"]}},
        "regime": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "quarters": {"type": "array", "items": {"type": "integer"}},
            "label": {"enum": ["expansion", "inflation", "deflation", "recession", "depression", "stagflation", "contraction", "high_infl"]},
            "share": {"type": "number", "minimum": 0, "maximum": 1}, "quote": {"type": "string", "maxLength": 120}},
            "required": ["quarters", "label", "share", "quote"]}}}},
    "events": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
        "text": {"type": "string", "maxLength": 200}, "quarters": {"type": "array", "items": {"type": "integer"}},
        "templates": {"type": "array", "items": {"type": "string", "pattern": "^G[0-9]{2}$"}},
        "channels": {"type": "array", "items": {"enum": X_IDS + U_IDS + ["x_oil_cause"]}},
        "generic_footprint": {"type": "boolean"}, "how_decomposed": {"type": "string", "maxLength": 200}},
        "required": ["text", "quarters", "channels", "generic_footprint", "how_decomposed"]}},
    "notes": {"type": "array", "items": {"type": "string", "maxLength": 200}}},
  "required": ["schema", "n_quarters", "rows", "c_q", "events"]}
(out / "qtable.schema.json").write_text(json.dumps(qtable, ensure_ascii=False, indent=1), encoding="utf-8")
print("x fields", len(X_IDS) + 1, "E1/E1c/E2 fields", len(E12), "->", ",".join(E12))
print("bytes", len(json.dumps(qtable, ensure_ascii=False)))
