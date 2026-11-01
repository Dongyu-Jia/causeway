import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scenariolab import config, jsonio
from scenariolab.llm import translator as tr
from scenariolab.engine.pooling import view_support
from scenariolab.scenarios.schema import SPEC_JSON_SCHEMA, allowed_variables, validate_spec

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")
from anthropic.types.beta import BetaMessage  # noqa: E402

FORBIDDEN_PARAMS = ("temperature", "top_p", "top_k", "thinking", "tool_choice", "tools")
NARRATIVE = "中东战争升级，霍尔木兹海峡航运受阻半年，油价急涨，随后经济滑向滞胀。"


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    monkeypatch.setattr(tr, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("MSL_LLM_MODEL", raising=False)


@pytest.fixture(scope="module")
def ctx():
    if not config.PANEL_PATH.exists():
        pytest.skip("panel not built")
    from scenariolab.context import EngineContext
    return EngineContext.load(selection=False)


def message(text=None, stop_reason="end_turn", stop_details=None, content=None, model="claude-opus-5-5", usage=None):
    if content is None:
        content = [{"type": "thinking", "thinking": "", "signature": "sig"}]
        if text is not None:
            content.append({"type": "text", "text": text})
    return BetaMessage.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": content,
        "stop_reason": stop_reason, "stop_details": stop_details,
        "usage": usage or {"input_tokens": 120, "output_tokens": 2400,
                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 9100}})


class FakeMessages:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeClient:
    def __init__(self, reply):
        self.beta = SimpleNamespace(messages=FakeMessages(reply))


def spec_json(spec=None) -> str:
    return json.dumps(spec or tr.INLINE_EXAMPLE, ensure_ascii=False)


def assert_request_shape(kwargs: dict, model: str = tr.DEFAULT_MODEL):
    assert kwargs["model"] == model and kwargs["max_tokens"] == tr.MAX_TOKENS
    assert kwargs["betas"] == [tr.FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    assert kwargs["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": SPEC_JSON_SCHEMA}}
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["messages"][0]["role"] == "user"
    for key in FORBIDDEN_PARAMS:
        assert key not in kwargs


def test_valid_response_becomes_validated_spec(ctx):
    client = FakeClient(message(spec_json()))
    out = tr.translate(NARRATIVE, ctx, client=client)
    (kwargs,) = client.beta.messages.calls
    assert_request_shape(kwargs)
    assert NARRATIVE in kwargs["messages"][0]["content"]
    spec, report = out["spec"], out["validation"]
    assert spec["id"] == "llm_oil_war" and spec["source"] == "llm"
    assert sum(p["months"] for p in spec["phases"]) == config.HORIZON_MONTHS
    assert report["ok"] and not report["errors"] and not report["clipped"]
    assert out["raw"] == tr.INLINE_EXAMPLE
    assert out["model"] == "claude-opus-5-5"
    assert out["usage"] == {"input_tokens": 120, "output_tokens": 2400,
                            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 9100}
    first = spec["phases"][0]
    assert first["overrides"] == {} and first["views"][0] == {
        "var": "f_oil", "mean_annual": 0.9, "vol_mult": 1.5, "why": tr.INLINE_EXAMPLE["phases"][0]["views"][0]["why"]}
    assert [p["name"] for p in out["preview"]] == [p["name"] for p in spec["phases"]]
    shock = out["preview"][0]
    assert 20 < shock["ens"] <= len(ctx.db) and shock["converged"]
    assert "great_inflation" in shock["era_weights"] and shock["top_months"]
    json.loads(jsonio.dumps(out))


def test_out_of_range_values_are_clipped_and_reported(ctx):
    raw = copy.deepcopy(tr.INLINE_EXAMPLE)
    shock = raw["phases"][0]
    shock["views"][1]["mean"] = 40.0                                   # s_cpi_yoy far beyond history
    shock["views"].append(tr._view("r_us_equity", "直接规定股市回报（不允许）", mean=0.05))
    shock["overrides"] = {
        "factor_drift": [{"factor": "labor_share", "annual": -0.5, "why": "AI 替代"},
                         {"factor": "bogus", "annual": 0.1, "why": "不存在的因子"}],
        "asset_drift": [{"asset": "nasdaq", "annual": 0.5, "why": "利润集中"},
                        {"asset": "cash", "annual": 0.02, "why": "不允许"}],
        "resid_scale": 5.0, "r_star": -9.0}
    out = tr.translate(NARRATIVE, ctx, client=FakeClient(message(spec_json(raw))))
    clipped = {c["var"]: c for c in out["validation"]["clipped"]}
    p98 = view_support(ctx.db.frame, "s_cpi_yoy", raw["phases"][0]["months"])[1]   # what a 6-month phase can average
    assert clipped["s_cpi_yoy"]["requested"] == 40.0 and abs(clipped["s_cpi_yoy"]["used"] - p98) < 1e-9
    assert clipped["asset_drift.nasdaq"]["used"] == pytest.approx(0.08)
    bound = tr.factor_drift_bounds(ctx.db.frame)["labor_share"]
    assert clipped["factor_drift.labor_share"]["used"] == pytest.approx(-bound)
    assert clipped["resid_scale"]["used"] == 2.0 and clipped["r_star"]["used"] == -3.0
    ov = out["spec"]["phases"][0]["overrides"]
    assert set(ov["asset_drift"]) == {"nasdaq"} and set(ov["factor_drift"]) == {"labor_share"}
    assert "r_us_equity" not in {v["var"] for v in out["spec"]["phases"][0]["views"]}
    warnings = "\n".join(out["validation"]["warnings"])
    for token in ("r_us_equity", "bogus", "cash", "截断"):
        assert token in warnings
    assert out["raw"]["phases"][0]["views"][1]["mean"] == 40.0          # raw is kept for the UI diff


def test_refusal_raises_translator_error(ctx):
    refusal = message(stop_reason="refusal", content=[],
                      stop_details={"type": "refusal", "category": "general_harms", "explanation": "政策限制"})
    with pytest.raises(tr.TranslatorError, match="拒绝") as err:
        tr.translate(NARRATIVE, ctx, client=FakeClient(refusal))
    assert "general_harms" in str(err.value) and "政策限制" in str(err.value)
    bare = message(stop_reason="refusal", content=[], stop_details=None)
    with pytest.raises(tr.TranslatorError, match="拒绝"):
        tr.translate(NARRATIVE, ctx, client=FakeClient(bare))


def test_max_tokens_raises_translator_error(ctx):
    truncated = message('{"id": "oil", "name": "截断的', stop_reason="max_tokens")
    with pytest.raises(tr.TranslatorError, match="max_tokens"):
        tr.translate(NARRATIVE, ctx, client=FakeClient(truncated))


def test_bad_outputs_raise_translator_error(ctx):
    for reply, pattern in ((message("这不是 JSON"), "JSON"), (message("[1, 2]"), "不是对象"),
                           (message(content=[]), "没有返回")):
        with pytest.raises(tr.TranslatorError, match=pattern):
            tr.translate(NARRATIVE, ctx, client=FakeClient(reply))
    with pytest.raises(tr.TranslatorError, match="不能为空"):
        tr.translate("   ", ctx, client=FakeClient(message(spec_json())))
    with pytest.raises(tr.TranslatorError, match="时长"):
        tr.translate(NARRATIVE, ctx, horizon_months=6, client=FakeClient(message(spec_json())))


def test_sdk_errors_map_to_chinese_messages(ctx):
    req = httpx2.Request("POST", "https://api.anthropic.test/v1/messages")
    cases = [
        (anthropic.RateLimitError("rate limited", body=None,
                                  response=httpx2.Response(429, headers={"retry-after": "30"}, request=req)), "30 秒"),
        (anthropic.AuthenticationError("bad key", body=None, response=httpx2.Response(401, request=req)), "401"),
        (anthropic.NotFoundError("no model", body=None, response=httpx2.Response(404, request=req)), "MSL_LLM_MODEL"),
        (anthropic.InternalServerError("boom", body=None, response=httpx2.Response(500, request=req)), "暂时不可用"),
        (anthropic.APITimeoutError(request=req), "超时"),
        (anthropic.APIConnectionError(request=req), "无法连接"),
    ]
    for error, pattern in cases:
        with pytest.raises(tr.TranslatorError, match=pattern):
            tr.translate(NARRATIVE, ctx, client=FakeClient(error))


def test_status_requires_credentials(monkeypatch, ctx):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tr, "_profile_configured", lambda: False)
    st = tr.status()
    assert st == {"available": False, "model": tr.DEFAULT_MODEL, "reason": st["reason"]}
    assert "ANTHROPIC_API_KEY" in st["reason"]
    with pytest.raises(tr.TranslatorError, match="ANTHROPIC_API_KEY"):
        tr.translate(NARRATIVE, ctx)                                    # no client, no key: no network call
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert not tr.status()["available"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("MSL_LLM_MODEL", "claude-sonnet-5-5")
    st = tr.status()
    assert st["available"] and st["model"] == "claude-sonnet-5-5" and "sk-ant-test" not in st["reason"]
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr(tr, "_profile_configured", lambda: True)
    assert tr.status()["available"]
    monkeypatch.setattr(tr, "anthropic", None)
    assert not tr.status()["available"]


def test_real_sdk_request_on_the_wire(ctx):
    """The installed SDK accepts every parameter and sends the beta header, fallbacks and schema."""
    seen = {}
    fallback_reply = {
        "id": "msg_wire", "type": "message", "role": "assistant", "model": "claude-opus-4-8",
        "content": [{"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-4-8"},
                     "trigger": {"type": "refusal", "category": None}},
                    {"type": "text", "text": spec_json()}],
        "stop_reason": "end_turn", "stop_details": None,
        "usage": {"input_tokens": 50, "output_tokens": 2000, "cache_creation_input_tokens": 9000,
                  "cache_read_input_tokens": 0,
                  "iterations": [{"type": "message", "input_tokens": 50, "output_tokens": 0,
                                  "cache_creation_input_tokens": 9000, "cache_read_input_tokens": 0},
                                 {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 50,
                                  "output_tokens": 2000, "cache_creation_input_tokens": 9000,
                                  "cache_read_input_tokens": 0}]}}

    def handler(request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=fallback_reply)

    client = anthropic.Anthropic(api_key="sk-ant-test", base_url="https://api.anthropic.test", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    out = tr.translate(NARRATIVE, ctx, client=client)
    body = seen["body"]
    assert tr.FALLBACK_BETA in seen["headers"]["anthropic-beta"]
    assert body["fallbacks"] == "default" and body["max_tokens"] == tr.MAX_TOKENS
    assert body["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": SPEC_JSON_SCHEMA}}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["text"] == tr.build_system_prompt(ctx)
    for key in FORBIDDEN_PARAMS:
        assert key not in body
    assert out["model"] == "claude-opus-4-8" and out["requested_model"] == tr.DEFAULT_MODEL
    assert out["usage"]["fallback_used"] is True
    assert out["spec"]["id"] == "llm_oil_war"


def test_request_kwargs_fall_back_to_extra_body_for_older_sdks():
    def old_create(*, model, max_tokens, messages, system=None, extra_headers=None, extra_body=None):
        return None

    kwargs = tr.request_kwargs(old_create, "m", "sys", "user")
    assert "fallbacks" not in kwargs and "output_config" not in kwargs and "betas" not in kwargs
    assert kwargs["extra_body"]["fallbacks"] == "default"
    assert kwargs["extra_body"]["output_config"]["format"]["schema"] is SPEC_JSON_SCHEMA
    assert kwargs["extra_headers"] == {"anthropic-beta": tr.FALLBACK_BETA}
    direct = tr.request_kwargs(FakeMessages(None).create, "m", "sys", "user")
    assert direct["fallbacks"] == "default" and "extra_body" not in direct


def test_repair_fixes_unambiguous_slips():
    raw = {"id": "Oil Shock!", "prior_prob": 7, "phases": [{"name": "冲击", "months": 12, "views": [
        {"var": "s_cpi_yoy", "mean": None, "mean_annual": 6.0, "shift_sd": None, "why": "通胀 6%"},
        {"var": "f_oil", "mean": None, "mean_annual": 0.5, "shift_sd": 1.0, "why": "油价上涨"},
        {"var": "f_oil", "mean": None, "mean_annual": 0.9, "vol_mult": 1.5, "why": "重复"},
        {"var": "s_unrate", "mean": 5.0, "mean_annual": 0.3, "why": "失业"},
        {"var": "f_valuation", "mean_annual": -0.2, "why": "泡沫破裂"},
    ], "overrides": {"factor_drift": [{"factor": "ip", "annual": -0.01, "why": ""}], "asset_drift": [],
                     "resid_scale": None, "r_star": None}, "confidence": 90}]}
    spec, notes = tr.repair(raw)
    assert spec["id"] == "llm_oil_shock" and spec["source"] == "llm"
    assert spec["prior_prob"] == pytest.approx(0.07) and spec["phases"][0]["confidence"] == pytest.approx(0.9)
    views = {v["var"]: v for v in spec["phases"][0]["views"]}
    assert views["s_cpi_yoy"]["mean"] == 6.0 and "mean_annual" not in views["s_cpi_yoy"]
    assert views["f_oil"] == {"var": "f_oil", "mean_annual": 0.5, "why": "油价上涨", "vol_mult": 1.5}
    assert views["s_unrate"] == {"var": "s_unrate", "mean": 5.0, "why": "失业"}
    assert views["f_valuation"]["mean_annual"] == -0.2                  # kept, but flagged
    text = "\n".join(notes)
    for token in ("s_cpi_yoy", "出现多次", "shift_sd", "理由", "百分数", "f_valuation"):
        assert token in text
    assert raw["phases"][0]["views"][0]["mean_annual"] == 6.0          # input untouched
    assert tr.repair({"id": "llm_x", "prior_prob": 1.2})[0] == {"id": "llm_x", "prior_prob": 1.2, "source": "llm"}


def test_system_prompt_is_deterministic_and_complete(ctx):
    prompt = tr.build_system_prompt(ctx)
    assert prompt == tr.build_system_prompt(ctx)
    for v in allowed_variables(ctx.db.frame):
        assert f"| {v['var']} |" in prompt
    assert f"{config.HORIZON_MONTHS} 个月" in prompt and ctx.anchor.date.strftime("%Y-%m") in prompt
    for era in config.ERAS:
        assert era["id"] in prompt
    for fid in tr.factor_drift_bounds(ctx.db.frame):
        assert f"| {fid} |" in prompt
    blocks = prompt.split("```json\n")[1:]
    assert 1 <= len(blocks) <= 2
    for block in blocks:
        example = json.loads(block.split("\n```")[0])
        clean, report = validate_spec(example, ctx.db.frame, config.HORIZON_MONTHS)
        assert report["ok"] and not report["clipped"]
        assert sum(p["months"] for p in example["phases"]) == config.HORIZON_MONTHS
        assert all(set(v) == {"var", *tr.VIEW_KEYS, "why"} for p in example["phases"] for v in p["views"])
    short = tr.build_system_prompt(ctx, 120)
    assert "120 个月" in short
    for block in short.split("```json\n")[1:]:
        assert sum(p["months"] for p in json.loads(block.split("\n```")[0])["phases"]) == 120


def test_prompt_describes_the_engines_residual_draw(ctx, monkeypatch):
    """Under config.RESIDUALS partial 0.5 a block keeps its own months' residuals only half the time."""
    prompt = tr.build_system_prompt(ctx)
    assert "被抽中月份自身的残差" not in prompt                     # the coupled-only wording is gone
    mode, c = config.RESIDUALS["resid_mode"], config.RESIDUALS["resid_coupling"]
    if mode == "partial":
        assert f"每个 {config.BLOCK_MONTHS} 个月区块以 {c:.0%} 的概率沿用被抽中月份自己的残差" in prompt
        assert "独立抽取的另一段残差" in prompt
    assert "resid_scale：该阶段全部资产残差" in prompt
    monkeypatch.setattr(config, "RESIDUALS", {"resid_mode": "partial", "resid_coupling": 0.25})
    assert "以 25% 的概率" in tr.residual_rule() and "约 25%" in tr.residual_rule()
    monkeypatch.setattr(config, "RESIDUALS", {"resid_mode": "coupled", "resid_coupling": 1.0})
    assert "被抽中月份自身的残差" in tr.residual_rule()


def test_lab_examples_are_the_prompt_examples_with_whole_narratives(ctx):
    """The lab's 载入示例: the few-shot examples in the LLM form, narratives not cut, runnable as they are."""
    prompt_examples = tr.few_shot_examples(ctx)
    lab = tr.lab_examples(ctx)
    assert [e["id"] for e in lab] == [e["id"] for e in prompt_examples]
    specs = tr._builtin_specs()
    for ex, short in zip(lab, prompt_examples):
        assert {k: v for k, v in ex.items() if k != "narrative"} == {k: v for k, v in short.items() if k != "narrative"}
        if ex["id"] in specs:
            assert ex["narrative"] == str(specs[ex["id"]]["narrative"])[:6000]
        assert set(ex) == set(SPEC_JSON_SCHEMA["required"])
        clean, report = validate_spec(ex, ctx.db.frame, config.HORIZON_MONTHS)
        assert report["ok"] and not report["clipped"]
    long = {"id": "x", "name": "x", "narrative": "长" * 3000, "phases": []}
    assert tr.llm_form(long, narrative_chars=None)["narrative"] == "长" * 3000
    assert tr.llm_form(long)["narrative"] == "长" * tr.EXAMPLE_NARRATIVE_CHARS + "……"


def test_examples_prefer_builtin_yaml(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "SPECS_DIR", tmp_path)
    assert [e["id"] for e in tr.few_shot_examples(ctx)] == ["oil_war"]      # inline fallback
    k_shaped = {"id": "k_shaped", "name": "K 型分化", "summary": "s", "narrative": "n" * 3000, "tags": ["分化"],
                "phases": [{"name": "分化", "months": 60,
                            "views": [{"var": "f_labor_share", "mean_annual": -0.01},
                                      {"var": "s_unrate", "min_annual": 60.0}],
                            "overrides": {"factor_drift": {"labor_share": -0.01}, "asset_drift": {"nasdaq": 0.02}}}]}
    other = {"id": "aaa", "name": "其他", "phases": [{"name": "p", "months": 12, "views": [{"var": "s_ff", "mean": 2}]}]}
    import yaml
    (tmp_path / "k_shaped.yaml").write_text(yaml.safe_dump(k_shaped, allow_unicode=True), encoding="utf-8")
    (tmp_path / "aaa.yaml").write_text(yaml.safe_dump(other, allow_unicode=True), encoding="utf-8")
    (tmp_path / "broken.yaml").write_text("phases: [unclosed", encoding="utf-8")
    examples = tr.few_shot_examples(ctx)
    assert [e["id"] for e in examples] == ["k_shaped", "aaa"]
    ex = examples[0]
    assert len(ex["narrative"]) <= tr.EXAMPLE_NARRATIVE_CHARS + 2
    ov = ex["phases"][0]["overrides"]
    assert ov["factor_drift"] == [{"factor": "labor_share", "annual": -0.01, "why": tr.DEFAULT_OVERRIDE_WHY}]
    assert ov["asset_drift"][0]["asset"] == "nasdaq" and ov["resid_scale"] is None
    unrate = ex["phases"][0]["views"][1]
    assert unrate["min"] == pytest.approx(5.0) and unrate["mean"] is None
    for e in examples:
        assert json.loads(tr.example_json(e)) == e


def test_drawn_weights_match_the_block_sampler():
    from scenariolab.engine.sampler import sample_phase
    rng = np.random.default_rng(7)
    w = rng.gamma(0.3, size=240)
    w /= w.sum()
    for n, block in ((6, 6), (20, 6), (3, 6), (1, 6), (37, 4)):
        eff = tr.drawn_weights(w, n, block)
        idx = sample_phase(w, n, 100000, block, np.random.default_rng(1))
        freq = np.bincount(idx.ravel(), minlength=w.size) / idx.size
        assert eff.sum() == pytest.approx(1.0) and eff.min() >= 0
        assert np.abs(freq - eff).max() < 2e-3


def test_preview_reports_what_the_sampler_draws(ctx):
    out = tr.translate(NARRATIVE, ctx, client=FakeClient(message(spec_json())))
    warnings = out["validation"]["warnings"]
    for k, ph in enumerate(out["preview"], start=1):
        for v in ph["views"]:
            assert np.isfinite(v["drawn_mean"])
            if "realized_share" in v:
                expected = (v["drawn_mean"] - v["prior_mean"]) / (v["target"] - v["prior_mean"])
                assert v["realized_share"] == pytest.approx(expected)
                if ph["shrink"] == 1.0:              # block pooling: the drawn months meet the view exactly
                    assert v["realized_share"] == pytest.approx(ph["confidence"], abs=1e-6)
        shrunk = ph["shrink"] < 1.0
        assert shrunk == any(w.startswith(f"阶段 {k}「") and "λ=" in w for w in warnings)
    cpi = next(v for v in out["preview"][0]["views"] if v["var"] == "s_cpi_yoy")
    assert cpi["realized_share"] > 0.8


def test_translated_spec_runs_through_the_real_engine(ctx):
    from scenariolab.scenarios.runner import headline, run_spec
    out = tr.translate(NARRATIVE, ctx, client=FakeClient(message(spec_json())))
    detail, sim = run_spec(ctx, out["spec"], n_paths=400, seed=3, source="llm")
    stag = out["preview"][1]                           # the preview mirrors what simulate() draws
    x = ctx.db.frame["s_cpi_yoy"].to_numpy(float)[sim.idx[:, stag["start_month"]:stag["start_month"] + stag["months"]]]
    drawn = next(v["drawn_mean"] for v in stag["views"] if v["var"] == "s_cpi_yoy")
    assert abs(x.mean() - drawn) < 0.6
    assert detail["source"] == "llm" and detail["id"] == "llm_oil_war"
    assert len(detail["assets"]["crude_oil"]["candles_monthly"]) == config.HORIZON_MONTHS
    h = headline(detail)
    assert all(np.isfinite(v) for v in h.values())
    oil_p50 = detail["macro"]["oil"]["fan"]["p50"]
    assert oil_p50[6] > oil_p50[0]                     # the shock phase lifts the engine's oil path
    cpi = np.array(detail["macro"]["cpi_yoy"]["fan"]["p50"])
    assert cpi[6:24].mean() > cpi[0]                   # ... and inflation follows through the engine


def test_run_writes_reference_file(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "RESULT_PATH", tmp_path / "llm_translator.json")
    out = tr.run(ctx)
    saved = json.loads((tmp_path / "llm_translator.json").read_text(encoding="utf-8"))
    assert saved["prompt"]["sha256"] == out["prompt"]["sha256"] and saved["translation"] is None
    assert len(saved["variables"]) == len(allowed_variables(ctx.db.frame))
    assert saved["schema"] == SPEC_JSON_SCHEMA and saved["request"]["fallbacks"] == "default"
    assert "cash" not in saved["asset_drift"]["assets"]
    assert [e["id"] for e in saved["examples"]] == saved["prompt"]["examples"]     # the lab's 载入示例
    with_translation = tr.run(ctx, NARRATIVE, client=FakeClient(message(spec_json())), write=False)
    assert with_translation["translation"]["spec"]["id"] == "llm_oil_war"


def test_spec_to_yaml_roundtrips(ctx):
    import yaml
    out = tr.translate(NARRATIVE, ctx, client=FakeClient(message(spec_json())))
    again, report = validate_spec(yaml.safe_load(tr.spec_to_yaml(out["spec"])), ctx.db.frame)
    assert again["phases"] == out["spec"]["phases"] and not report["clipped"]


def test_cli(ctx, monkeypatch, capsys, tmp_path):
    from scenariolab import context
    monkeypatch.setattr(context.EngineContext, "load", classmethod(lambda cls, **k: ctx))
    assert tr.main(["--print-prompt"]) == 0
    assert "## 可用变量" in capsys.readouterr().out
    monkeypatch.setattr(tr, "status", lambda: {"available": False, "model": "m", "reason": "未配置凭据"})
    assert tr.main([NARRATIVE]) == 1
    assert "未配置凭据" in capsys.readouterr().err
    monkeypatch.setattr(tr, "status", lambda: {"available": True, "model": "m", "reason": ""})
    monkeypatch.setattr(tr.anthropic, "Anthropic", lambda: FakeClient(message(spec_json())))
    out_yaml = tmp_path / "spec.yaml"
    assert tr.main([NARRATIVE, "--yaml", str(out_yaml), "--run", "--paths", "100"]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed.split("\n\n模型")[0])["id"] == "llm_oil_war"
    assert "美股大盘" in printed and out_yaml.exists()
