"""The claude_code backend (llm/claude_code.py) and the translator's backend selection, status, response
cache and concurrency. Never a live call: subprocess.run is conftest.FakeCLI, and without `fake_cli` every
Claude Code process is blocked (conftest.no_live_llm). Needs neither data nor the anthropic package, except
the two end-to-end tests marked needs_panel."""
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenariolab import build, config
from scenariolab.env import dotenv_problems
from scenariolab.llm import claude_code as cc
from scenariolab.llm import translator as tr

SYSTEM = "你是测试用的系统提示词。"
NARRATIVE = "中东战争升级，霍尔木兹海峡航运受阻半年，油价急涨，随后经济滑向滞胀。"
USER = f"请把下面的叙事翻译成 180 个月的情景规格。\n\n<narrative>\n{NARRATIVE}\n</narrative>"
SPEC = {"id": "oil", "name": "油价冲击", "phases": []}
MODEL = "claude-opus-5-5"
TOKEN = "sk-ant-oat01-" + "Q7" * 40                 # a fake setup-token: no message may contain it
API_KEY = "sk-ant-api03-" + "K9" * 30
SENTINEL = "sentinel-value-ZX81"                     # an environment value: no message may contain it
SIGNED_OUT = {"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"}
needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    monkeypatch.setattr(tr, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(tr, "dotenv_problems", lambda *a, **k: [])
    for var in ("MSL_LLM_BACKEND", "MSL_LLM_MODEL", "MSL_LLM_TIMEOUT", "MSL_LLM_CONCURRENCY", cc.TOKEN_VAR):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MSL_TEST_SENTINEL", SENTINEL)


@pytest.fixture(scope="module")
def ctx():
    if not config.PANEL_PATH.exists():
        pytest.skip("panel not built")
    from scenariolab.context import EngineContext
    return EngineContext.load(selection=False)


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def assert_clean(message: str):
    for secret in (TOKEN, API_KEY, SENTINEL, "sk-ant-"):
        assert secret not in message


# ---------------------------------------------------------------- the call: arguments, cwd, environment

def test_print_mode_argument_list_without_a_shell(fake_cli):
    text, _ = cc.run(SYSTEM, USER, MODEL, timeout=123)
    ((args, kw),) = fake_cli.print_calls
    system_file = args[args.index("--system-prompt-file") + 1]
    assert args == [fake_cli.binary, "-p", "--model", MODEL, "--system-prompt-file", system_file, "--tools", "",
                    "--strict-mcp-config", "--setting-sources", "project", "--no-session-persistence",
                    "--output-format", "json"]
    assert "--bare" not in args and "--json-schema" not in args
    assert "--effort" not in args and "--safe-mode" not in args            # the translator's call is unchanged
    assert kw["shell"] is False and kw["input"] == USER and kw["timeout"] == 123
    assert kw["stdin"] == kw["stdout"] == kw["stderr"] == subprocess.PIPE
    assert kw["text"] is True and kw["encoding"] == "utf-8"
    assert kw["start_new_session"] is (os.name == "posix")                  # its own process group
    assert all(NARRATIVE not in a and SYSTEM not in a for a in args)       # stdin and a file, never argv
    assert text == fake_cli.result


def test_the_call_runs_in_a_fresh_empty_temporary_directory(fake_cli):
    seen = {}

    def look(args, kw):
        cwd = Path(kw["cwd"])
        system_file = Path(args[args.index("--system-prompt-file") + 1])
        seen.update(cwd=cwd, listing=os.listdir(cwd), file=system_file,
                    system=system_file.read_text(encoding="utf-8"))

    fake_cli.on_print = look
    cc.run(SYSTEM, USER, MODEL)
    first = seen["cwd"]
    assert seen["listing"] == [] and seen["system"] == SYSTEM
    assert seen["file"].parent == first.parent and seen["file"].parent != first       # beside the cwd, not in it
    assert Path(tempfile.gettempdir()).resolve() in first.resolve().parents
    assert config.ROOT.resolve() not in first.resolve().parents                     # never the project
    assert not first.exists() and not seen["file"].exists()                          # removed afterwards
    cc.run(SYSTEM, USER, MODEL)
    assert seen["cwd"] != first and seen["listing"] == []                            # a new one per call


def test_child_env_drops_api_credentials_and_the_parent_session():
    kept = {"PATH": "/usr/bin:/bin", "HOME": "/home/u", "HTTPS_PROXY": "http://proxy:3128", "LANG": "zh_CN.UTF-8",
            "MSL_PORT": "8765", "CLAUDE_CONFIG_DIR": "/home/u/.claude", "MAX_THINKING_TOKENS": "8000",
            "TOKENIZERS_PARALLELISM": "false", "SSH_AUTH_SOCK": "/tmp/agent.sock", "AWS_ACCESS_KEY_ID": "AKIA"}
    base = {**kept,
            "ANTHROPIC_API_KEY": API_KEY, "ANTHROPIC_AUTH_TOKEN": "auth-token", "ANTHROPIC_BASE_URL": "https://gw.test",
            "CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "claude-desktop", "CLAUDE_CODE_SESSION_ID": "s",
            "CLAUDE_CODE_SIMPLE": "1", "CLAUDE_CODE_USE_BEDROCK": "1", "CLAUDE_EFFORT": "max",
            "CLAUDE_AGENT_SDK_VERSION": "0.2",
            # credential-named variables the CLI never needs (FRED_API_KEY comes from the app's own .env)
            "FRED_API_KEY": "f" * 32, "GITHUB_TOKEN": "ghp_x", "AWS_SECRET_ACCESS_KEY": "s", "OPENAI_API_KEY": "o",
            "DB_PASSWORD": "p", "GOOGLE_APPLICATION_CREDENTIALS": "/k.json", "API_KEY": "k", "hf_token": "h"}
    assert cc.child_env(base) == kept
    assert cc.child_env({**base, cc.TOKEN_VAR: f"  {TOKEN}\n"})[cc.TOKEN_VAR] == TOKEN
    for empty in ("", "   "):
        assert cc.TOKEN_VAR not in cc.child_env({**base, cc.TOKEN_VAR: empty})
    assert base["ANTHROPIC_API_KEY"] == API_KEY                                      # the input is untouched


def test_the_call_and_the_status_checks_get_the_sanitized_environment(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "claude-desktop")
    cc.run(SYSTEM, USER, MODEL)
    env = fake_cli.print_calls[0][1]["env"]
    assert not any(k.startswith("ANTHROPIC_") for k in env) and "CLAUDECODE" not in env
    assert cc.TOKEN_VAR not in env and env["PATH"] == os.environ["PATH"] and env["MSL_TEST_SENTINEL"] == SENTINEL
    monkeypatch.setenv(cc.TOKEN_VAR, TOKEN)                   # a setup-token passes through
    cc.run(SYSTEM, USER, MODEL)
    assert fake_cli.print_calls[1][1]["env"][cc.TOKEN_VAR] == TOKEN
    fake_cli.calls.clear()
    tr.status(force=True)
    assert len(fake_cli.calls) == 2 and all(kw["env"] == cc.child_env() for _, kw in fake_cli.calls)


# ---------------------------------------------------------------- the envelope

def test_envelope_text_and_metadata(fake_cli):
    fake_cli.result = json.dumps(SPEC, ensure_ascii=False)
    text, meta = cc.run(SYSTEM, USER, MODEL)
    assert json.loads(text) == SPEC
    assert meta == {"backend": "claude_code", "model": MODEL, "models": [MODEL], "duration_ms": 41234,
                    "duration_api_ms": 40000, "num_turns": 1, "stop_reason": "end_turn",
                    "usage": {"input_tokens": 12, "output_tokens": 2400, "cache_creation_input_tokens": 9000,
                              "cache_read_input_tokens": 0},
                    "usage_by_model": {MODEL: {"inputTokens": 12, "outputTokens": 2400, "cacheReadInputTokens": 0,
                                               "cacheCreationInputTokens": 9000, "costUSD": 0.42}},
                    "cost_usd": 0.42, "effort": None, "safe_mode": False, "wall_s": meta["wall_s"]}
    fake_cli.envelope = fake_cli.success(fake_cli.result, modelUsage={
        "claude-haiku-5": {"outputTokens": 30}, "claude-opus-5-5": {"outputTokens": 2400}})
    assert cc.run(SYSTEM, USER, MODEL)[1]["model"] == MODEL                          # the one that answered
    fake_cli.envelope = "warning: something on stdout first\n" + json.dumps(fake_cli.success("{}"))
    assert cc.run(SYSTEM, USER, MODEL)[0] == "{}"


FAILURES = [  # (envelope, exit code, stderr, setup-token set, expected text)
    ({"is_error": True, "api_error_status": 401,
      "result": "Failed to authenticate. API Error: 401 OAuth access token is invalid."}, 1, "", False, "claude auth login"),
    ({"is_error": True, "api_error_status": 401,
      "result": "Failed to authenticate. API Error: 401 OAuth access token is invalid."}, 1, "", True, "setup-token"),
    ({"is_error": True, "result": "Not logged in · Please run /login"}, 1, "", False, "claude auth login"),
    ({"is_error": True, "api_error_status": 429, "result": "Claude AI usage limit reached|1767225600"}, 1, "", False,
     "用量上限（5 小时或每周额度）（预计"),
    ({"is_error": True, "result": "5-hour limit reached ∙ resets 3pm (America/Los_Angeles)"}, 1, "", False,
     "3pm (America/Los_Angeles) 重置"),
    ({"is_error": True, "api_error_status": 429, "result": "API Error: 429 rate_limit_error"}, 1, "", False, "过于频繁"),
    ({"is_error": True, "api_error_status": 404,
      "result": "There's an issue with the selected model (claude-x). It may not exist."}, 1, "", False, "MSL_LLM_MODEL"),
    ({"is_error": True, "api_error_status": 529, "result": "API Error: 529 overloaded_error"}, 1, "", False, "繁忙"),
    ({"is_error": True, "api_error_status": 500, "result": "API Error: 500"}, 1, "", False, "暂时不可用（500）"),
    ({"is_error": True, "subtype": "error_max_turns", "num_turns": 3}, 1, "", False, "3 轮内"),
    ({"is_error": True, "result": f"weird failure {TOKEN} {API_KEY}"}, 1, "", True, "调用失败（退出码 1）：weird failure ***"),
    ({"is_error": False}, 1, "", False, "调用失败（退出码 1）"),
    ("Error: not json at all", 1, "boom on stderr", False, "无法解析（退出码 1）：boom on stderr"),
    ("Not logged in · Please run /login", 1, "", False, "claude auth login"),          # plain text on stdout
    (f"fatal: {TOKEN}", 1, "", True, "无法解析（退出码 1）：fatal: ***"),
    ("you hit your usage limit, said the model", 0, "", False, "无法解析（退出码 0）：you hit"),    # exit 0: quoted only
    ({"stop_reason": "refusal", "result": ""}, 0, "", False, "拒绝"),
    ({"stop_reason": "max_tokens", "result": "{\"id\": \"cut"}, 0, "", False, "长度上限"),
    ({"result": "   "}, 0, "", False, "结果为空"),
]


@pytest.mark.parametrize("envelope, code, stderr, token, expected", FAILURES)
def test_failures_raise_chinese_messages_without_secrets(fake_cli, monkeypatch, envelope, code, stderr, token, expected):
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    if token:
        monkeypatch.setenv(cc.TOKEN_VAR, TOKEN)
    fake_cli.envelope = envelope if isinstance(envelope, str) else fake_cli.success(**{"result": None, **envelope})
    fake_cli.returncode, fake_cli.stderr = code, stderr
    with pytest.raises(cc.ClaudeCodeError) as err:
        cc.run(SYSTEM, USER, MODEL)
    assert expected in str(err.value)
    assert_clean(str(err.value))
    with pytest.raises(tr.TranslatorError) as terr:                  # the translator's error, same message
        tr.complete(SYSTEM, USER, MODEL)
    assert str(terr.value) == str(err.value)
    assert not list(tr.CACHE_DIR.glob("*.json"))                     # a failure is never cached


def test_redact_catches_tokens_outside_the_environment_and_short_keys_inside_it():
    login_token = "sk-ant-oat01-" + "W3" * 20                  # a keychain login token: never in the environment
    fred = "0123456789abcdef0123456789abcdef"                  # 32 characters: not token-shaped
    environ = {"FRED_API_KEY": fred, "GITHUB_TOKEN": "ghp_" + "a" * 20, "MSL_PORT": "87654321", "SHORT_TOKEN": "abc"}
    text = cc.redact(f"auth {login_token} failed; fred={fred} gh={environ['GITHUB_TOKEN']} port=87654321 abc",
                     environ=environ)
    assert login_token not in text and fred not in text and "ghp_" not in text
    assert "port=87654321" in text and text.endswith(" abc")            # other values and short ones stay
    assert cc.redact("x" * 39) == "x" * 39 and cc.redact("x" * 40) == "***"


def test_a_lone_surrogate_in_the_narrative_is_not_a_crash(fake_cli):
    """JSON allows "\\ud800" (a half-pasted emoji); it used to raise UnicodeEncodeError in cache_key: an HTTP 500."""
    user = "油价\ud800冲击"
    assert tr.cache_key("claude_code", MODEL, SYSTEM, user) != tr.cache_key("claude_code", MODEL, SYSTEM, "油价?冲击")
    fake_cli.result = json.dumps(SPEC)
    assert tr.complete(SYSTEM, user, MODEL)[0] == SPEC and tr.complete(SYSTEM, user, MODEL)[1]["cached"]
    key = tr.cache_key("claude_code", MODEL, SYSTEM, "an answer no encoder takes")
    tr.cache_store(key, "答\ud800", {"backend": "claude_code"})          # not cached, and nothing left behind
    assert tr.cache_load(key) is None and not list(tr.CACHE_DIR.glob(f"*{key}*"))


def test_timeout_and_a_missing_binary(fake_cli, monkeypatch):
    monkeypatch.setenv(cc.TOKEN_VAR, TOKEN)
    fake_cli.error = subprocess.TimeoutExpired(cmd=["claude", "-p"], timeout=7, output=TOKEN)
    with pytest.raises(cc.ClaudeCodeError, match="7 秒内没有完成") as err:
        cc.run(SYSTEM, USER, MODEL, timeout=7)
    assert "MSL_LLM_TIMEOUT" in str(err.value) and err.value.__cause__ is None and err.value.__suppress_context__
    assert_clean(str(err.value))
    monkeypatch.setenv("MSL_LLM_TIMEOUT", "42")                      # the translator passes MSL_LLM_TIMEOUT on
    with pytest.raises(tr.TranslatorError, match="42 秒内没有完成"):
        tr.complete(SYSTEM, USER, MODEL)
    assert fake_cli.print_calls[-1][1]["timeout"] == 42.0
    fake_cli.error = PermissionError(13, "Permission denied")
    with pytest.raises(cc.ClaudeCodeError, match="无法启动 Claude Code.*Permission denied"):
        cc.run(SYSTEM, USER, MODEL)
    monkeypatch.setenv("MSL_CLAUDE_BIN", str(Path(fake_cli.binary).parent / "gone"))
    with pytest.raises(cc.ClaudeCodeError, match="MSL_CLAUDE_BIN"):
        cc.run(SYSTEM, USER, MODEL)


def test_settings_from_the_environment(monkeypatch):
    assert (tr.timeout_s(), tr.concurrency()) == (600.0, 2)
    for raw, expected in (("90", 90.0), ("-5", 600.0), ("abc", 600.0), ("inf", 600.0), ("nan", 600.0), ("0.5", 600.0)):
        monkeypatch.setenv("MSL_LLM_TIMEOUT", raw)
        assert tr.timeout_s() == expected, raw
    for raw, expected in (("3", 3), ("1", 1), ("0", 2), ("-1", 2), ("2.5", 2), ("x", 2)):
        monkeypatch.setenv("MSL_LLM_CONCURRENCY", raw)
        assert tr.concurrency() == expected, raw


# ---------------------------------------------------------------- response cache and concurrency

def test_a_cache_hit_skips_the_subprocess(fake_cli, monkeypatch):
    monkeypatch.setenv(cc.TOKEN_VAR, TOKEN)
    fake_cli.result = "```json\n" + json.dumps(SPEC, ensure_ascii=False) + "\n```"     # a fenced answer parses too
    raw, meta = tr.complete(SYSTEM, USER, MODEL)
    assert raw == SPEC and meta["cached"] is False and meta["num_turns"] == 1 and len(fake_cli.print_calls) == 1
    again, hit = tr.complete(SYSTEM, USER, MODEL)
    assert again == SPEC and hit["cached"] is True and hit["duration_ms"] == 41234 and hit["cached_at"]
    assert len(fake_cli.print_calls) == 1                                             # no second call
    (path,) = tr.CACHE_DIR.glob("*.json")
    assert path.stem == tr.cache_key("claude_code", MODEL, SYSTEM, USER)
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert set(entry) == {"created", "text", "meta"} and entry["text"] == fake_cli.result      # verbatim
    content = path.read_text(encoding="utf-8")
    assert SYSTEM not in content and NARRATIVE not in content                          # no prompt, no narrative
    assert_clean(content)
    tr.complete(SYSTEM, USER, MODEL, refresh=True)                                     # refresh asks again
    assert len(fake_cli.print_calls) == 2
    tr.complete(SYSTEM, USER, "claude-sonnet-5-5")                                     # each key part matters
    tr.complete(SYSTEM + "。", USER, MODEL)
    tr.complete(SYSTEM, USER + "。", MODEL)
    assert len(fake_cli.print_calls) == 5 and len(list(tr.CACHE_DIR.glob("*.json"))) == 4
    assert tr.cache_key("api", MODEL, SYSTEM, USER) != tr.cache_key("claude_code", MODEL, SYSTEM, USER)
    fake_cli.result = "这不是 JSON"                                                     # not a spec: not cached
    for _ in range(2):
        with pytest.raises(tr.TranslatorError, match="不是合法的 JSON"):
            tr.complete(SYSTEM, "另一段叙事", MODEL)
    assert len(fake_cli.print_calls) == 7 and len(list(tr.CACHE_DIR.glob("*.json"))) == 4
    path.write_text("{broken", encoding="utf-8")                                       # a broken entry is replaced
    fake_cli.result = json.dumps(SPEC)
    assert tr.complete(SYSTEM, USER, MODEL)[1]["cached"] is False and tr.complete(SYSTEM, USER, MODEL)[1]["cached"]


def test_at_most_msl_llm_concurrency_calls_at_once(fake_cli, monkeypatch):
    monkeypatch.setenv("MSL_LLM_CONCURRENCY", "2")
    fake_cli.result = json.dumps(SPEC)
    lock, release = threading.Lock(), threading.Event()
    state = {"now": 0, "max": 0}

    def busy(args, kw):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        release.wait(10)
        with lock:
            state["now"] -= 1

    fake_cli.on_print = busy
    threads = [threading.Thread(target=tr.complete, args=(SYSTEM, f"{USER}{k}", MODEL)) for k in range(5)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 10
    while state["now"] < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.3)                                       # room for a third call to (wrongly) start
    assert state["now"] == 2
    release.set()
    for t in threads:
        t.join(10)
    assert state["max"] == 2 and len(fake_cli.print_calls) == 5


def test_the_api_backend_through_complete_with_an_injected_client(fake_cli):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stop_reason="end_turn", model="claude-opus-5-5",
                               content=[SimpleNamespace(type="text", text=json.dumps(SPEC))],
                               usage=SimpleNamespace(input_tokens=5, output_tokens=7, cache_creation_input_tokens=0,
                                                     cache_read_input_tokens=0, iterations=None))

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    raw, meta = tr.complete(SYSTEM, USER, MODEL, backend="api", client=client)
    assert raw == SPEC and meta["backend"] == "api" and meta["usage"]["output_tokens"] == 7 and len(calls) == 1
    assert fake_cli.calls == []                                                       # no CLI for the api backend
    assert tr.complete(SYSTEM, USER, MODEL, backend="api", client=client)[1]["cached"] and len(calls) == 1
    with pytest.raises(tr.TranslatorError, match="不认识"):
        tr.complete(SYSTEM, USER, MODEL, backend="openai")


# ---------------------------------------------------------------- status

def test_status_reports_binary_version_and_auth_without_spending_usage(fake_cli):
    st = tr.status()
    assert st == {"backend": "claude_code", "label": "本机 Claude Code（使用你的订阅额度）", "available": True,
                  "model": MODEL, "binary": fake_cli.binary, "version": "2.1.278", "auth": "Claude Code 登录",
                  "reason": st["reason"]}
    assert sorted(tuple(a[1:]) for a, _ in fake_cli.calls) == [("--version",), ("auth", "status", "--json")]
    assert all(Path(kw["cwd"]).exists() is False for _, kw in fake_cli.calls)       # a temporary cwd, removed
    fake_cli.auth, fake_cli.auth_rc = SIGNED_OUT, 1          # signed out: JSON on stdout and exit code 1
    st = tr.status(force=True)
    assert not st["available"] and st["auth"] == "未登录" and "claude auth login" in st["reason"]
    assert "claude setup-token" in st["reason"] and "CLAUDE_CODE_OAUTH_TOKEN" in st["reason"]


@pytest.mark.parametrize("auth, available, label", [
    ({"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"}, True, "CLAUDE_CODE_OAUTH_TOKEN"),
    ({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "max"}, True,
     "Claude Code 登录"),
    ({"loggedIn": True, "authMethod": "api_key", "apiKeySource": "apiKeyHelper", "apiProvider": "firstParty"}, False,
     "API Key（按 API 计费）"),
    ({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "bedrock"}, False, "bedrock"),
    (SIGNED_OUT, False, "未登录"),
    ("not json", False, "未登录"),
])
def test_status_auth_modes(fake_cli, monkeypatch, auth, available, label):
    monkeypatch.setenv(cc.TOKEN_VAR, TOKEN)
    fake_cli.auth = auth
    st = tr.status()
    assert st["available"] is available and st["auth"] == label and st["reason"]
    assert_clean(json.dumps(st, ensure_ascii=False))
    if isinstance(auth, dict) and auth.get("subscriptionType"):
        assert st["subscription"] == "max"
    if auth == "not json":
        assert "没有输出 JSON" in st["reason"]


def test_status_is_reused_for_a_minute_and_an_unavailable_one_for_seconds(fake_cli, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(tr, "_now", lambda: clock[0])
    tr.status()
    n = len(fake_cli.calls)
    clock[0] += tr.STATUS_TTL_S - 1
    tr.status()
    assert len(fake_cli.calls) == n                          # reused
    clock[0] += 2
    tr.status()
    assert len(fake_cli.calls) == n + 2                      # checked again
    tr.status(force=True)
    assert len(fake_cli.calls) == n + 4
    fake_cli.auth, fake_cli.auth_rc = SIGNED_OUT, 1
    assert not tr.status(force=True)["available"]
    m = len(fake_cli.calls)
    clock[0] += tr.STATUS_RETRY_S - 1
    tr.status()
    assert len(fake_cli.calls) == m
    fake_cli.auth, fake_cli.auth_rc = {"loggedIn": True, "authMethod": "claude.ai"}, 0     # signed in meanwhile
    clock[0] += 2
    assert tr.status()["available"] and len(fake_cli.calls) == m + 2
    monkeypatch.setenv("MSL_LLM_MODEL", "claude-sonnet-5-5")                              # another model: new check
    assert tr.status()["model"] == "claude-sonnet-5-5" and len(fake_cli.calls) == m + 4


def test_status_flags_a_token_split_over_two_lines(fake_cli, monkeypatch):
    monkeypatch.setenv(cc.TOKEN_VAR, TOKEN[:30])
    fake_cli.auth = {"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"}
    monkeypatch.setattr(tr, "dotenv_problems", lambda *a, **k: [(5, "ANTHROPIC_API_KEY")])
    assert tr.status(force=True)["available"]                               # another key's line: not this token
    monkeypatch.setattr(tr, "dotenv_problems", lambda *a, **k: [(7, cc.TOKEN_VAR)])
    st = tr.status(force=True)
    assert not st["available"] and "第 7 行" in st["reason"] and "合并成一行" in st["reason"]
    assert TOKEN[:30] not in st["reason"]
    monkeypatch.setattr(tr, "dotenv_problems", lambda *a, **k: [])
    monkeypatch.setenv(cc.TOKEN_VAR, "sk-ant-oat01-abc def")
    assert "空格或换行" in tr.status(force=True)["reason"]


def test_status_without_a_binary_runs_nothing(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(cc, "_exec", lambda args, **kw: started.append(args))
    monkeypatch.setenv("MSL_CLAUDE_BIN", str(tmp_path / "missing" / "claude"))
    st = tr.status()
    assert st["backend"] == "claude_code" and not st["available"] and st["binary"] is None
    assert "MSL_CLAUDE_BIN" in st["reason"] and started == []


def test_live_claude_code_is_blocked_in_tests():
    with pytest.raises(FileNotFoundError, match="tests never start Claude Code"):
        cc._exec(["claude", "--version"])


# ---------------------------------------------------------------- binary and backend selection

@pytest.mark.skipif(os.name != "posix", reason="POSIX paths and executable bits")
def test_binary_resolution_order(monkeypatch, tmp_path):
    custom = executable(tmp_path / "custom" / "claude-dev")
    on_path = executable(tmp_path / "pathdir" / "claude")
    fallback = executable(tmp_path / "home" / ".local" / "bin" / "claude")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(on_path.parent))
    monkeypatch.setenv("MSL_CLAUDE_BIN", str(custom))
    assert cc.resolve_binary() == (str(custom), None)                       # 1. MSL_CLAUDE_BIN
    monkeypatch.setenv("MSL_CLAUDE_BIN", str(tmp_path / "nope"))
    path, problem = cc.resolve_binary()
    assert path is None and "MSL_CLAUDE_BIN" in problem                     # set but wrong: no silent fallback
    monkeypatch.delenv("MSL_CLAUDE_BIN")
    assert cc.resolve_binary() == (str(on_path), None)                      # 2. claude on PATH
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert cc.resolve_binary() == (str(fallback), None)                     # 3. ~/.local/bin/claude
    fallback.unlink()
    path, problem = cc.resolve_binary()
    assert path is None and "claude auth login" in problem and "MSL_CLAUDE_BIN" in problem


def test_backend_selection(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)                  # an API key alone never selects the API
    assert tr.backend_name() == "claude_code" and tr.status()["backend"] == "claude_code"
    for raw in (" API ", "api"):
        monkeypatch.setenv("MSL_LLM_BACKEND", raw)
        fake_cli.calls.clear()
        st = tr.status()
        assert st["backend"] == "api" and st["label"] == tr.API_LABEL and fake_cli.calls == []
        assert set(st) == {"backend", "label", "available", "model", "reason"}
    monkeypatch.setenv("MSL_LLM_BACKEND", "claude-code")
    assert tr.backend_name() == "claude_code"
    monkeypatch.setenv("MSL_LLM_BACKEND", "openai")
    st = tr.status()
    assert st["backend"] == "openai" and not st["available"] and "不认识" in st["reason"]
    assert tr.request_info("openai", MODEL) == {"backend": "openai"}
    assert tr.request_info("claude_code", MODEL)["argv"] == cc.build_args("claude", MODEL, "<system prompt file>")
    assert tr.request_info("api", MODEL)["fallbacks"] == "default"


def test_parse_spec_text_tolerates_a_wrapped_answer():
    for text in (json.dumps(SPEC), "```json\n" + json.dumps(SPEC) + "\n```", "```\n" + json.dumps(SPEC) + "```",
                 "规格如下：\n" + json.dumps(SPEC, ensure_ascii=False) + "\n以上。"):
        assert tr.parse_spec_text(text) == SPEC
    with pytest.raises(tr.TranslatorError, match="不是合法的 JSON"):
        tr.parse_spec_text("这不是 JSON")
    with pytest.raises(tr.TranslatorError, match="不是对象"):
        tr.parse_spec_text("[1, 2]")
    with pytest.raises(tr.TranslatorError, match="没有返回"):
        tr.parse_spec_text("  ")


def test_dotenv_problems_names_the_key_before_a_wrapped_value(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# local secrets\nFRED_API_KEY=abc\n\nCLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-first-half\n"
                   " second-half-of-the-token\nMSL_PORT=8765\nexport X=1\n", encoding="utf-8")
    assert dotenv_problems(env) == [(5, "CLAUDE_CODE_OAUTH_TOKEN"), (7, "MSL_PORT")]
    assert dotenv_problems(tmp_path / "missing.env") == []


# ---------------------------------------------------------------- server, build, translate end to end

def test_server_status_and_translate_pass_through(fake_cli, monkeypatch):
    from fastapi.testclient import TestClient

    from scenariolab import context, server
    client = TestClient(server.app, base_url="http://127.0.0.1:8765", raise_server_exceptions=False)
    body = client.get("/api/llm/status").json()
    assert body["backend"] == "claude_code" and body["auth"] == "Claude Code 登录" and body["available"]
    seen = []
    monkeypatch.setattr(context, "get_context", lambda: "ctx")
    monkeypatch.setattr(tr, "translate", lambda narrative, ctx, horizon_months, refresh: seen.append(refresh) or {"ok": 1})
    for extra, refresh in (({}, False), ({"refresh": True}, True), ({"refresh": "yes"}, False)):
        r = client.post("/api/llm/translate", json={"narrative": NARRATIVE, **extra})
        assert r.status_code == 200 and seen[-1] is refresh
    for base in ("http://localhost:8765", "http://[::1]:8765"):
        assert TestClient(server.app, base_url=base).post("/api/llm/translate", json={"narrative": NARRATIVE}).status_code == 200
    calls = len(fake_cli.calls)
    for base in ("http://rebound.example:8765", "http://127.0.0.1.example", "http://testserver"):
        # another site's page whose DNS name points at 127.0.0.1 (DNS rebinding) sends its own Host: refused
        # before the status check or a translation runs, so it cannot spend the user's plan
        r = TestClient(server.app, base_url=base).post("/api/llm/translate", json={"narrative": NARRATIVE})
        assert r.status_code == 403 and "本机" in r.json()["detail"]
    assert len(seen) == 5 and len(fake_cli.calls) == calls
    fake_cli.auth, fake_cli.auth_rc = SIGNED_OUT, 1
    tr.status(force=True)
    r = client.post("/api/llm/translate", json={"narrative": NARRATIVE})
    assert r.status_code == 503 and "claude auth login" in r.json()["detail"] and len(seen) == 5
    assert fake_cli.print_calls == []


def test_the_build_llm_step_passes_no_narrative(monkeypatch, tmp_path):
    """build.py's llm step is translator.run(ctx) with nothing else: no narrative, so no translation call."""
    seen = {}
    cached = tmp_path / "panel_monthly.csv"
    cached.write_text("date\n", encoding="utf-8")
    monkeypatch.setattr(build.config, "PANEL_PATH", cached)
    monkeypatch.setattr(build.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(build.EngineContext, "load", classmethod(lambda cls, **kw: object()))
    monkeypatch.setattr(build, "_run", lambda module, ctx, **kw: seen.setdefault(module, kw) or {})
    assert build.main(["--only", "llm"]) == 0
    assert seen == {"llm.translator": {}}


@needs_panel
def test_translate_end_to_end_through_claude_code(fake_cli, ctx):
    fake_cli.result = json.dumps(tr.INLINE_EXAMPLE, ensure_ascii=False)
    seen = {}

    def look(args, kw):
        seen["system"] = Path(args[args.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        seen["stdin"] = kw["input"]

    fake_cli.on_print = look
    out = tr.translate(NARRATIVE, ctx)
    assert out["backend"] == "claude_code" and out["cached"] is False and out["model"] == MODEL
    assert out["spec"]["id"] == "llm_oil_war" and out["validation"]["ok"] and out["raw"] == tr.INLINE_EXAMPLE
    assert out["meta"]["num_turns"] == 1 and out["meta"]["duration_ms"] == 41234 and out["usage"]["output_tokens"] == 2400
    assert seen["system"] == tr.system_prompt(ctx, backend="claude_code")
    assert seen["system"] == tr.build_system_prompt(ctx) + "\n\n" + tr.schema_section()
    assert NARRATIVE in seen["stdin"] and seen["stdin"] == tr._user_message(NARRATIVE, config.HORIZON_MONTHS)
    again = tr.translate(NARRATIVE, ctx)
    assert again["cached"] is True and again["spec"] == out["spec"] and len(fake_cli.print_calls) == 1
    assert tr.translate(NARRATIVE, ctx, refresh=True)["cached"] is False and len(fake_cli.print_calls) == 2
    fake_cli.auth, fake_cli.auth_rc = SIGNED_OUT, 1
    tr.status(force=True)
    with pytest.raises(tr.TranslatorError, match="claude auth login"):
        tr.translate(NARRATIVE, ctx)                                  # not signed in: not even the cache is read
    assert len(fake_cli.print_calls) == 2


@needs_panel
def test_translate_replaces_a_lone_surrogate_before_anything_encodes_it(fake_cli, ctx):
    fake_cli.result = json.dumps(tr.INLINE_EXAMPLE, ensure_ascii=False)
    out = tr.translate("油价\ud800冲击，随后经济滑向滞胀。", ctx)
    assert out["validation"]["ok"] and "油价?冲击" in fake_cli.print_calls[0][1]["input"]


@needs_panel
def test_the_llm_step_never_calls_a_model(fake_cli, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "RESULT_PATH", tmp_path / "llm_translator.json")
    out = tr.run(ctx)                                                # what the build's llm step runs
    assert out["translation"] is None and fake_cli.print_calls == []
    assert {tuple(a[1:]) for a, _ in fake_cli.calls} == {("--version",), ("auth", "status", "--json")}
    saved = json.loads((tmp_path / "llm_translator.json").read_text(encoding="utf-8"))
    assert saved["backend"] == "claude_code" and saved["status"]["auth"] == "Claude Code 登录"
    assert saved["request"]["argv"][:2] == ["claude", "-p"] and saved["request"]["timeout_s"] == 600.0
    assert saved["prompt"]["system"].endswith(tr.schema_section()) and saved["prompt"]["chars"] == len(saved["prompt"]["system"])


# ---------------------------------------------------------------- error kinds, flags, in-flight sharing, cancellation

FAILURE_KINDS = ["login", "token", "login", "limit", "limit", "rate", "model", "overloaded", "server", "max_turns",
                 "failed", "failed", "unparseable", "login", "unparseable", "unparseable", "refusal", "max_tokens",
                 "empty"]                                       # one per FAILURES row, in order


@pytest.mark.parametrize("case, kind", list(zip(FAILURES, FAILURE_KINDS)))
def test_every_failure_carries_a_machine_readable_kind(fake_cli, monkeypatch, case, kind):
    envelope, code, stderr, token, _ = case
    if token:
        monkeypatch.setenv(cc.TOKEN_VAR, TOKEN)
    fake_cli.envelope = envelope if isinstance(envelope, str) else fake_cli.success(**{"result": None, **envelope})
    fake_cli.returncode, fake_cli.stderr = code, stderr
    with pytest.raises(cc.ClaudeCodeError) as err:
        cc.run(SYSTEM, USER, MODEL)
    assert err.value.kind == kind and err.value.kind in cc.KINDS
    assert_clean(err.value.detail)
    if "|1767225600" in json.dumps(envelope):
        assert err.value.reset_at == 1767225600.0


def test_timeout_cancel_and_start_failures_have_kinds(fake_cli):
    fake_cli.hang = True
    with pytest.raises(cc.ClaudeCodeError) as err:
        cc.run(SYSTEM, USER, MODEL, timeout=0.05)
    assert err.value.kind == "timeout" and fake_cli.kills == 1                 # the hanging call was killed
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(cc.ClaudeCodeError) as err:
        cc.run(SYSTEM, USER + "c", MODEL, timeout=30, cancel=cancel)
    assert err.value.kind == "cancelled" and fake_cli.kills == 2 and time.monotonic() - t0 < 5
    with pytest.raises(cc.ClaudeCodeError) as err:                            # already cancelled: nothing starts
        cc.run(SYSTEM, USER + "d", MODEL, cancel=cancel)
    assert err.value.kind == "cancelled" and len(fake_cli.print_calls) == 2
    fake_cli.hang, fake_cli.error = False, PermissionError(13, "Permission denied")
    with pytest.raises(cc.ClaudeCodeError) as err:
        cc.run(SYSTEM, USER, MODEL)
    assert err.value.kind == "cli_missing"


def test_effort_and_safe_mode_are_passed_only_when_the_cli_lists_them(fake_cli):
    text, meta = cc.run(SYSTEM, USER, MODEL, effort="medium", safe_mode=True)
    args = fake_cli.print_calls[-1][0]
    assert args[-3:] == ["--effort", "medium", "--safe-mode"] and meta["effort"] == "medium" and meta["safe_mode"]
    helps = [c for c in fake_cli.calls if c[0][1:] == ["--help"]]
    assert len(helps) == 1 and helps[0][1]["env"] == cc.child_env()            # probed once, sanitized env
    cc.run(SYSTEM, USER + "2", MODEL, effort="ultra")                           # a level the help does not list
    assert "--effort" not in fake_cli.print_calls[-1][0]
    assert len([c for c in fake_cli.calls if c[0][1:] == ["--help"]]) == 1      # the probe is cached
    cc._FLAGS.clear()
    fake_cli.help = "Usage: claude [options]\n  --model <model>  Model\n"
    text, meta = cc.run(SYSTEM, USER + "3", MODEL, effort="high", safe_mode=True)
    assert "--effort" not in fake_cli.print_calls[-1][0] and "--safe-mode" not in fake_cli.print_calls[-1][0]
    assert meta["effort"] is None and meta["safe_mode"] is False and len(meta["flag_notes"]) == 2
    assert cc.parse_help(FAKE_HELP_2_1_278) == {"effort": True, "effort_levels": ["low", "medium", "high", "xhigh",
                                                                                   "max"], "safe_mode": True}
    assert cc.parse_help("") == {"effort": False, "effort_levels": [], "safe_mode": False}
    assert cc.build_args("claude", MODEL, "f", effort="low", safe_mode=True)[-3:] == ["--effort", "low", "--safe-mode"]


FAKE_HELP_2_1_278 = """  --effort <level>                      Effort level for the current session
                                        (low, medium, high, xhigh, max)
  --safe-mode                           Start with all customizations
"""


def test_two_identical_requests_share_one_call(fake_cli):
    release, started = threading.Event(), threading.Event()

    def busy(args, kw):
        started.set()
        release.wait(10)

    fake_cli.on_print = busy
    results = []
    threads = [threading.Thread(target=lambda: results.append(cc.run(SYSTEM, USER, MODEL))) for _ in range(3)]
    for t in threads:
        t.start()
    started.wait(5)
    time.sleep(0.2)
    release.set()
    for t in threads:
        t.join(10)
    assert len(fake_cli.print_calls) == 1 and len(results) == 3
    assert sorted(bool(m.get("shared_inflight")) for _, m in results) == [False, True, True]
    assert len({text for text, _ in results}) == 1 and len(cc._INFLIGHT) == 0
    cc.run(SYSTEM, USER, MODEL)                                               # not in flight any more: a new call
    assert len(fake_cli.print_calls) == 2


def _until(cond, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_a_cancelled_leader_hands_over_and_other_errors_are_shared():
    flight = cc.SingleFlight()
    gate, calls, out = threading.Event(), [], {}

    def leader():
        calls.append("leader")
        gate.wait(5)
        raise cc.ClaudeCodeError(cc.E_CANCELLED, "cancelled")

    t1 = threading.Thread(target=_swallow, args=(flight.do, "k", leader))
    t1.start()
    _until(lambda: len(flight) == 1)
    t2 = threading.Thread(target=lambda: out.update(f=flight.do("k", lambda: calls.append("follower") or ("x", {}))))
    t2.start()
    time.sleep(0.1)
    gate.set()
    t1.join(5)
    t2.join(5)
    assert calls == ["leader", "follower"] and out["f"] == ("x", {})          # the follower asked again itself
    gate2, seen = threading.Event(), {}

    def failing():
        gate2.wait(5)
        raise cc.ClaudeCodeError("繁忙", "overloaded", detail="d")

    def follow():
        try:
            flight.do("k2", lambda: seen.setdefault("called", ("never", {})))
        except cc.ClaudeCodeError as err:
            seen["err"] = err

    t3 = threading.Thread(target=_swallow, args=(flight.do, "k2", failing))
    t3.start()
    _until(lambda: len(flight) == 1)
    t4 = threading.Thread(target=follow)
    t4.start()
    time.sleep(0.1)
    gate2.set()
    t3.join(5)
    t4.join(5)
    assert seen["err"].kind == "overloaded" and "called" not in seen and len(flight) == 0


def _swallow(fn, *args):
    try:
        fn(*args)
    except cc.ClaudeCodeError:
        pass


# ---------------------------------------------------------------- real processes, but of the fake executable only

def test_the_fake_executable_round_trip(fake_claude_bin, monkeypatch, tmp_path):
    log = tmp_path / "argv.jsonl"
    monkeypatch.setenv("MSL_FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    st = cc.status(MODEL)
    assert st["available"] and st["version"] == "2.1.278" and st["binary"] == str(fake_claude_bin)
    flags = cc.supported_flags(str(fake_claude_bin))
    assert flags["effort"] and flags["safe_mode"] and flags["effort_levels"][-1] == "max"
    from scenariolab.llm import agent_prompts as P
    from agent_llm_cases import ctx_root, fed
    schema = P.actor_output_schema(fed())
    text, meta = cc.run(P.actor_system(schema), P.actor_user(ctx_root(), fed()), MODEL, timeout=30, effort="medium",
                        safe_mode=True)
    assert json.loads(text)["actor"] == "fed" and meta["backend"] == "claude_code" and meta["effort"] == "medium"
    entries = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    printed = next(e for e in entries if "-p" in e["argv"])
    assert printed["argv"][-3:] == ["--effort", "medium", "--safe-mode"] and printed["cwd_listing"] == []
    assert not any(k.startswith("ANTHROPIC_") for k in printed["env"])         # the API key never reaches the CLI
    monkeypatch.setenv("MSL_FAKE_CLAUDE_FLAGS", "none")
    cc._FLAGS.clear()
    text, meta = cc.run("系统", "用户", MODEL, timeout=30, effort="medium", safe_mode=True)
    assert "--effort" not in json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["argv"]
    assert meta["effort"] is None and meta["flag_notes"]


@pytest.mark.parametrize("mode, kind", [("limit", "limit"), ("login", "login"), ("nojson", "unparseable"),
                                        ("exit:3", "unparseable"), ("refusal", "refusal")])
def test_the_fake_executables_failure_modes(fake_claude_bin, monkeypatch, mode, kind):
    monkeypatch.setenv("MSL_FAKE_CLAUDE_MODE", mode)
    with pytest.raises(cc.ClaudeCodeError) as err:
        cc.run("系统", "用户", MODEL, timeout=30)
    assert err.value.kind == kind
    if mode == "limit":
        assert err.value.reset_at and "预计" in str(err.value)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                              # a zombie is dead for our purposes
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False)
        return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")
    except OSError:
        return True


def _wait_dead(pid: int, seconds: float = 8.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("how", ["timeout", "cancel"])
def test_timeout_and_cancel_kill_the_whole_process_group(fake_claude_bin, monkeypatch, tmp_path, how):
    pidfile = tmp_path / "child.pid"
    monkeypatch.setenv("MSL_FAKE_CLAUDE_MODE", "sleep:30")
    monkeypatch.setenv("MSL_FAKE_CLAUDE_CHILD_PIDFILE", str(pidfile))
    cancel = threading.Event()
    if how == "cancel":
        def fire():
            deadline = time.monotonic() + 10
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            cancel.set()
        threading.Thread(target=fire, daemon=True).start()
    t0 = time.monotonic()
    with pytest.raises(cc.ClaudeCodeError) as err:
        cc.run("系统", "用户", MODEL, timeout=3.0 if how == "timeout" else 30, cancel=cancel)
    assert err.value.kind == ("timeout" if how == "timeout" else "cancelled") and time.monotonic() - t0 < 15
    child = int(pidfile.read_text(encoding="utf-8"))
    assert _wait_dead(child), "the CLI's own child process outlived the call"


def test_a_request_that_waited_for_a_slot_reuses_the_answer_it_waited_for(fake_cli, monkeypatch):
    monkeypatch.setenv("MSL_LLM_CONCURRENCY", "1")
    fake_cli.result = json.dumps(SPEC)
    release, started = threading.Event(), threading.Event()

    def busy(args, kw):
        started.set()
        release.wait(10)

    fake_cli.on_print = busy
    out = []
    first = threading.Thread(target=lambda: out.append(tr.complete(SYSTEM, USER, MODEL)))
    first.start()
    started.wait(5)
    second = threading.Thread(target=lambda: out.append(tr.complete(SYSTEM, USER, MODEL)))
    second.start()
    time.sleep(0.2)                                       # the second request now waits for the only slot
    release.set()
    first.join(10)
    second.join(10)
    assert len(fake_cli.print_calls) == 1 and sorted(m["cached"] for _, m in out) == [False, True]
