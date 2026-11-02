"""JSON out of model text and the mini schema validator (scenariolab/llm/jsonfix.py). No model, no data."""
import json

from scenariolab.llm import jsonfix as jf

OBJ = {"status": "ok", "options": [{"id": "hold", "prob": 0.6}, {"id": "cut_25", "prob": 0.4}], "note": "失去席位"}
TEXT = json.dumps(OBJ, ensure_ascii=False)


def test_plain_fenced_and_wrapped_answers_parse():
    assert jf.extract_json(TEXT) == (OBJ, [])
    obj, notes = jf.extract_json(f"```json\n{TEXT}\n```")
    assert obj == OBJ and notes == ["去掉了代码块标记"]
    obj, notes = jf.extract_json(f"好的，以下是结果：\n```json\n{TEXT}\n```\n以上。")
    assert obj == OBJ and "去掉了 JSON 之外的文字" in notes
    obj, notes = jf.extract_json(f"结果如下 {TEXT} 谢谢 {{不是 JSON}}")          # a later brace pair is ignored
    assert obj == OBJ and notes == ["去掉了 JSON 之外的文字"]


def test_trailing_commas_nan_and_python_literals_are_fixed_outside_strings():
    text = '{"a": [1, 2, ], "b": NaN, "c": True, "d": None, "e": "NaN, True,]",}'
    obj, notes = jf.extract_json(text)
    assert obj == {"a": [1, 2], "b": None, "c": True, "d": None, "e": "NaN, True,]"}
    assert notes == ["删除了尾随逗号 / NaN / Python 字面量"]


def test_full_width_syntax_is_fixed_but_full_width_text_in_strings_survives():
    obj, notes = jf.extract_json('{“headline”：“议员失去席位”，"n"：1}')
    assert obj == {"headline": "议员失去席位", "n": 1}
    assert notes == ["把用作 JSON 符号的中文引号、冒号、逗号改为半角"]
    inner = '{"text": "他说“失去席位”，然后：离开", "k": 2}'                # valid JSON: never touched
    assert jf.extract_json(inner) == ({"text": "他说“失去席位”，然后：离开", "k": 2}, [])
    obj, _ = jf.extract_json('{“a”：“x“y”}')                                # an inner curly quote stays text
    assert obj == {"a": "x“y"}


def test_raw_newlines_in_strings_are_accepted():
    obj, _ = jf.extract_json('{"a": "第一行\n第二行"}')
    assert obj == {"a": "第一行\n第二行"}


def test_unparseable_and_non_objects():
    obj, notes = jf.extract_json(TEXT[:40])
    assert obj is None and "输出可能被截断" in notes[-1] and "不是合法的 JSON" in notes[-1]
    assert jf.extract_json("") == (None, ["没有输出"])
    assert jf.extract_json("我不能回答")[0] is None
    assert jf.extract_json("[1, 2]")[1][-1] == "输出里没有 JSON 对象"


SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["id", "items"],
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z_]+$", "maxLength": 6},
        "v": {"const": "x"}, "kind": {"enum": ["a", "b", None]},
        "n": {"type": "number", "minimum": 0, "maximum": 1}, "k": {"type": "integer"},
        "items": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"$ref": "#/$defs/item"}},
    },
    "$defs": {"item": {"type": "object", "required": ["p"], "properties": {"p": {"type": ["number", "null"]}}}},
}


def test_validate_reports_each_keyword_in_chinese():
    assert jf.validate({"id": "ab", "items": [{"p": None}], "n": 0.5, "k": 3, "kind": None, "v": "x"}, SCHEMA) == []
    bad = {"id": "AB-toolong", "v": "y", "kind": "c", "n": 1.5, "k": 1.5, "items": [{}, {"p": "x"}, {"p": 1}],
           "extra": 1}
    kinds = sorted(e.kind for e in jf.validate(bad, SCHEMA))
    assert kinds == ["additional", "const", "enum", "max_items", "max_length", "maximum", "pattern", "required",
                     "type", "type"]
    errs = jf.validate({"items": []}, SCHEMA)
    assert {e.kind for e in errs} == {"required", "min_items"} and "缺少 id" in str(errs[0])
    assert jf.validate(True, {"type": "number"})[0].kind == "type"                     # a bool is not a number
    assert jf.validate(float("nan"), {"type": "number"})[0].kind == "type"
    assert jf.validate(2.0, {"type": "integer"}) == []


def test_transport_schema_drops_bounds_and_turns_const_into_enum():
    t = jf.transport_schema({"$schema": "x", "title": "t", "description": "d", "properties": {
        "a": {"type": "string", "maxLength": 3, "pattern": "x", "description": "keep the key, drop this"},
        "minimum": {"type": "number", "minimum": 0},                    # a property named like a keyword survives
        "c": {"const": "agents-actor-2"}}, "type": "object"})
    assert t == {"properties": {"a": {"type": "string"}, "minimum": {"type": "number"},
                                "c": {"enum": ["agents-actor-2"]}}, "type": "object"}


def test_canonical_json_is_order_free_and_truncate_marks_cuts():
    assert jf.canonical_json({"b": 1, "a": "中"}) == '{"a":"中","b":1}'
    assert jf.truncate("一二三四五", 3) == "一二…" and jf.truncate("一二", 3) == "一二"
