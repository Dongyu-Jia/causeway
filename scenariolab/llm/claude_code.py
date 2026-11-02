"""The translator's default backend: the local Claude Code CLI in print mode, on the user's own
subscription login (本机 Claude Code（使用你的订阅额度）).

    claude -p --model <id> --system-prompt-file <file> --tools "" --strict-mcp-config \
        --setting-sources project --no-session-persistence --output-format json   < user message

* The user message goes in on stdin and the system prompt in a temporary file; no shell is involved.
* The child runs in a fresh, empty temporary directory, so no project CLAUDE.md, hooks, settings or
  .mcp.json load. --tools "" leaves the model no tools, --strict-mcp-config no MCP servers,
  --setting-sources project no user or local settings (an empty directory has no project ones), and
  --no-session-persistence keeps no transcript. --bare is not used: it never reads the subscription login.
* Its environment has no ANTHROPIC_* variable (in print mode an ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN
  always wins and bills the API), none of a parent Claude Code session (CLAUDECODE, CLAUDE_*) and no other
  credential-named variable (…_API_KEY, …_TOKEN, …_SECRET, …_PASSWORD, such as the app's own FRED_API_KEY), except
  CLAUDE_CONFIG_DIR and a non-empty CLAUDE_CODE_OAUTH_TOKEN (a `claude setup-token` token, the alternative to
  `claude auth login`).
* The CLI has no timeout flag: the subprocess has one (MSL_LLM_TIMEOUT, 600 s by default).

Every call counts against the plan's five-hour and weekly limits. status() never makes one: it runs only
`claude --version` and `claude auth status --json`. Personal local use only: each user runs their own
signed-in Claude Code; nothing here proxies requests for anyone else or stores or forwards credentials.

Output: the text path, not --json-schema. SPEC_JSON_SCHEMA would be accepted there (plain types, required,
enum, anyOf, additionalProperties false; no $ref), and the CLI would return a validated `structured_output`,
but Claude Code does structured output as a tool call that it validates and retries on a mismatch (result
subtype error_max_structured_output_retries): more requests, and so more plan usage, per translation than
one text answer, and nothing here could check it without spending usage. The schema goes into the prompt
instead (translator.schema_section), and the answer takes the API path's parse -> repair -> validate_spec;
the schema holds no numeric bounds by design, so validate_spec is the guard rail either way.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

LABEL = "本机 Claude Code（使用你的订阅额度）"
AUTH_LOGIN = "Claude Code 登录"
AUTH_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"
AUTH_NONE = "未登录"
AUTH_API_KEY = "API Key（按 API 计费）"
TOKEN_VAR = "CLAUDE_CODE_OAUTH_TOKEN"
KEEP_CLAUDE_VARS = ("CLAUDE_CONFIG_DIR", TOKEN_VAR)       # the CLI's own config dir and the setup-token
SECRET_VARS = (TOKEN_VAR, "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
DEFAULT_TIMEOUT_S = 600.0
CHECK_TIMEOUT_S = 15.0                                   # claude --version, claude auth status --json
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
EXCERPT_CHARS = 200

INSTALL_HINT = ("找不到 Claude Code 命令行（claude）：先安装 Claude Code，并在终端运行 claude auth login 登录；"
                "claude 不在 PATH 里时，用 MSL_CLAUDE_BIN 写它的完整路径。")
LOGIN_HINT = ("Claude Code 未登录（或登录已过期）：在终端运行 claude auth login，用你自己的 Claude 订阅账号登录后重试；"
              "也可以运行 claude setup-token，把生成的令牌完整地写成 .env 里的一行 CLAUDE_CODE_OAUTH_TOKEN=…，再重启服务。")
E_LOGIN = "Claude Code 未登录或登录已过期（401）：在终端运行 claude auth login 重新登录后重试。"
E_TOKEN = ("CLAUDE_CODE_OAUTH_TOKEN 无效或已过期（401）：用 claude setup-token 重新生成令牌，完整地写在 .env 的一行里并重启服务；"
           "或者删掉它，改用 claude auth login 登录。")
E_LIMIT = "已达到 Claude 订阅的用量上限（5 小时或每周额度）{reset}：等额度恢复后再试。"
E_RATE = "请求过于频繁（429），请稍后重试。"
E_FORBIDDEN = "当前 Claude 账号无权使用模型 {model}（403）：检查 MSL_LLM_MODEL，或确认你的订阅包含这个模型。"
E_MODEL = "Claude Code 无法使用模型 {model}：检查 MSL_LLM_MODEL（模型 ID 是否正确、你的订阅能否使用它）。"
E_OVERLOADED = "Claude 服务繁忙（529），请稍后重试。"
E_SERVER = "Claude 服务暂时不可用（{status}），请稍后重试。"
E_CONTEXT = "输入超出模型上下文窗口：请缩短叙事。"
E_MAX_TOKENS = "模型输出达到长度上限，情景规格不完整：请缩短或简化叙事后重试。"
E_REFUSAL = "模型拒绝了这个请求：请换一种表述，聚焦宏观经济机制后重试。"
E_EMPTY = "Claude Code 没有返回情景规格（结果为空）。"
E_TIMEOUT = ("Claude Code 在 {seconds:g} 秒内没有完成（MSL_LLM_TIMEOUT）：模型可能仍在长时间思考，"
             "请稍后重试、简化叙事，或调大 MSL_LLM_TIMEOUT。")

# credential-shaped strings, removed from any CLI text a message quotes
TOKENISH = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}|[A-Za-z0-9_\-]{40,}")
# credential-named variables (FRED_API_KEY, GITHUB_TOKEN, AWS_SECRET_ACCESS_KEY, ...): never passed to the CLI,
# which needs none of them, and their values are redacted like SECRET_VARS (MAX_THINKING_TOKENS does not match)
SECRET_NAME = re.compile(r"(?:^|_)(?:API_?KEY|ACCESS_KEY|SECRET(?:_KEY)?|PRIVATE_KEY|PASSWORD|PASSWD|PASSPHRASE|TOKEN"
                         r"|CREDENTIALS?)$", re.I)


class ClaudeCodeError(Exception):
    """A failed Claude Code call; the message is user-facing (Chinese) and never holds a token or the environment."""


# ---------------------------------------------------------------- binary, environment, arguments

def fallback_bin() -> Path:
    """Where the native installer links the CLI (often missing from the PATH a server is started with)."""
    return Path(os.path.expanduser("~/.local/bin/claude"))


def _executable(path) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def resolve_binary() -> tuple[str | None, str | None]:
    """(path, None), or (None, a Chinese reason): MSL_CLAUDE_BIN, else `claude` on PATH, else ~/.local/bin/claude."""
    explicit = os.environ.get("MSL_CLAUDE_BIN", "").strip()
    if explicit:
        path = os.path.expanduser(explicit)
        found = path if _executable(path) else shutil.which(path)
        if found:
            return found, None
        return None, f"MSL_CLAUDE_BIN 指向的 {explicit} 不存在或不可执行。"
    found = shutil.which("claude")
    if found:
        return found, None
    fallback = fallback_bin()
    if _executable(fallback):
        return str(fallback), None
    return None, INSTALL_HINT


def child_env(environ=None) -> dict[str, str]:
    """The environment the CLI runs with: the parent's, minus every ANTHROPIC_* variable (an API key or
    auth token there makes print mode bill the API), every variable of a parent Claude Code session
    (CLAUDECODE, CLAUDE_*) and every other credential-named one (SECRET_NAME: the .env's FRED_API_KEY, a
    GITHUB_TOKEN, ...), keeping CLAUDE_CONFIG_DIR and CLAUDE_CODE_OAUTH_TOKEN when it is not empty."""
    src = os.environ if environ is None else environ
    env = {k: v for k, v in src.items()
           if not (k.startswith("ANTHROPIC_") or k == "CLAUDECODE"
                   or (k.startswith("CLAUDE_") and k not in KEEP_CLAUDE_VARS)
                   or (k != TOKEN_VAR and SECRET_NAME.search(k)))}
    token = str(src.get(TOKEN_VAR) or "").strip()
    if token:
        env[TOKEN_VAR] = token
    else:
        env.pop(TOKEN_VAR, None)
    return env


def build_args(binary, model: str, system_file) -> list[str]:
    """The print-mode argument list (no shell; the user message goes in on stdin)."""
    return [str(binary), "-p", "--model", model, "--system-prompt-file", str(system_file),
            "--tools", "", "--strict-mcp-config", "--setting-sources", "project",
            "--no-session-persistence", "--output-format", "json"]


def _exec(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """The one place this module starts a process: an argument list, never a shell (tests replace it)."""
    return subprocess.run(args, shell=False, check=False, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kwargs)


# ---------------------------------------------------------------- output envelope and errors

def parse_json_object(text) -> dict | None:
    """The JSON object a CLI command printed (the whole output, else its last line that is one), or None.
    A list (stream or verbose output) gives its last {"type": "result"} element."""
    s = str(text or "").strip()
    if not s:
        return None
    candidates = [s] + ([s[s.index("{"):]] if "{" in s else []) + [line for line in reversed(s.splitlines()) if line.strip()]
    for c in candidates:
        try:
            obj = json.loads(c)
        except ValueError:
            continue
        if isinstance(obj, list):
            obj = next((x for x in reversed(obj) if isinstance(x, dict) and x.get("type") == "result"), None)
        if isinstance(obj, dict):
            return obj
    return None


def redact(text, environ=None) -> str:
    """Text with the values of the credential variables (SECRET_VARS and every credential-named one, such as a
    32-character FRED_API_KEY that TOKENISH would miss) and any credential-shaped string replaced by ***."""
    out = str(text or "")
    src = os.environ if environ is None else environ
    names = set(SECRET_VARS) | {k for k in src if SECRET_NAME.search(k)}
    for value in sorted({str(src.get(k) or "").strip() for k in names}, key=len, reverse=True):
        if len(value) >= 8:
            out = out.replace(value, "***")
    return TOKENISH.sub("***", out)


def _excerpt(*texts) -> str:
    """A short single-line, redacted quote of the first non-empty text."""
    for t in texts:
        line = " ".join(redact(t).split())
        if line:
            return line if len(line) <= EXCERPT_CHARS else line[:EXCERPT_CHARS].rstrip() + "…"
    return "没有输出"


def _int(x) -> int | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return int(x)


def _reset_hint(text: str) -> str:
    """When the limit resets, from the CLI's message: "…|<epoch seconds>" or "resets <time>"."""
    m = re.search(r"\|(\d{9,11})\b", text)
    if m:
        return f"（预计 {time.strftime('%m-%d %H:%M', time.localtime(int(m.group(1))))} 恢复）"
    m = re.search(r"resets?\s+(?:at\s+)?([^\n|·∙]{1,40})", text, re.I)
    if m:
        return f"（{_excerpt(m.group(1)).rstrip('.。')} 重置）"
    return ""


def failure_message(envelope: dict | None, returncode, stderr: str, model: str, token_used: bool,
                    stdout: str = "") -> str:
    """A Chinese message for a failed call, from the envelope's api_error_status / result and stderr. When stdout
    held no envelope it is quoted too (a plain-text CLI error), and classified only after a non-zero exit (with
    exit code 0 it could be the model's own words)."""
    env = envelope or {}
    status = _int(env.get("api_error_status"))
    stdout = stdout if envelope is None else ""
    text = " ".join(str(x) for x in (env.get("result"), env.get("error"), stderr, stdout if returncode != 0 else "")
                    if x)
    low = text.lower()
    if status == 401 or any(k in low for k in ("failed to authenticate", "invalid api key", "not logged in",
                                               "/login", "authentication_error")) \
            or ("oauth" in low and any(k in low for k in ("invalid", "expired", "revoked"))):
        return E_TOKEN if token_used else E_LOGIN
    rate = "rate limit" in low or "rate_limit" in low
    if any(k in low for k in ("usage limit", "hit your limit", "reached your limit", "5-hour limit", "weekly limit")) \
            or (not rate and ("limit reached" in low or ("limit" in low and "resets" in low))):
        return E_LIMIT.format(reset=_reset_hint(text))
    if status == 429 or rate:
        return E_RATE
    if status == 403:
        return E_FORBIDDEN.format(model=model)
    if status == 404 or ("model" in low and any(k in low for k in ("not found", "not_found", "does not exist",
                                                                    "issue with the selected model"))):
        return E_MODEL.format(model=model)
    if status == 529 or "overloaded" in low:
        return E_OVERLOADED
    if status is not None and status >= 500:
        return E_SERVER.format(status=status)
    if any(k in low for k in ("prompt is too long", "context window", "context_length")):
        return E_CONTEXT
    if "output token" in low and any(k in low for k in ("maximum", "exceed")):
        return E_MAX_TOKENS
    subtype = str(env.get("subtype") or "")
    if subtype == "error_max_turns":
        return f"Claude Code 在 {_int(env.get('num_turns')) or '若干'} 轮内没有给出结果（{subtype}）。"
    where = f"退出码 {returncode}" + (f"，{subtype}" if subtype and subtype != "success" else "")
    if envelope is None:
        return f"Claude Code 的输出无法解析（{where}）：{_excerpt(stderr, stdout)}"
    return f"Claude Code 调用失败（{where}）：{_excerpt(env.get('result'), env.get('error'), stderr)}"


def metadata(envelope: dict, requested_model: str, wall_s: float) -> dict:
    """What a translation records about the call: the model, duration, turns, stop reason and token usage."""
    usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
    tokens = {k: _int(usage[k]) for k in USAGE_KEYS if _int(usage.get(k)) is not None}
    per_model = envelope.get("modelUsage") if isinstance(envelope.get("modelUsage"), dict) else {}

    def output_tokens(m: str) -> int:
        v = per_model[m]
        return (_int(v.get("outputTokens")) or 0) if isinstance(v, dict) else 0

    main = max(per_model, key=output_tokens) if per_model else None
    return {"backend": "claude_code", "model": main or requested_model, "models": sorted(per_model),
            "duration_ms": _int(envelope.get("duration_ms")), "duration_api_ms": _int(envelope.get("duration_api_ms")),
            "num_turns": _int(envelope.get("num_turns")), "stop_reason": envelope.get("stop_reason"),
            "usage": tokens, "wall_s": round(wall_s, 2)}


# ---------------------------------------------------------------- the call

def run(system: str, user: str, model: str, timeout: float = DEFAULT_TIMEOUT_S) -> tuple[str, dict]:
    """One print-mode call: (the answer's text, metadata). ClaudeCodeError on any failure (not signed in, an
    invalid token, the usage limit, a timeout, ...), with a message that holds no token and no environment."""
    binary, problem = resolve_binary()
    if problem:
        raise ClaudeCodeError(problem)
    env = child_env()
    with tempfile.TemporaryDirectory(prefix="causeway-claude-") as tmp:
        cwd = Path(tmp) / "cwd"
        cwd.mkdir()                                       # empty: the system prompt file sits beside it
        system_file = Path(tmp) / "system-prompt.md"
        system_file.write_text(system, encoding="utf-8")
        t0 = time.perf_counter()
        try:
            proc = _exec(build_args(binary, model, system_file), input=user, cwd=str(cwd), env=env,
                         timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ClaudeCodeError(E_TIMEOUT.format(seconds=timeout)) from None
        except OSError as err:
            raise ClaudeCodeError(f"无法启动 Claude Code（{binary}：{err.strerror or type(err).__name__}）。"
                                  f"{INSTALL_HINT}") from None
        wall = time.perf_counter() - t0
    envelope = parse_json_object(proc.stdout)
    if (proc.returncode != 0 or envelope is None or envelope.get("is_error")
            or str(envelope.get("subtype") or "success") != "success"):
        raise ClaudeCodeError(failure_message(envelope, proc.returncode, proc.stderr, model, TOKEN_VAR in env,
                                              stdout=proc.stdout))
    stop = envelope.get("stop_reason")
    if stop == "refusal":
        raise ClaudeCodeError(E_REFUSAL)
    if stop == "max_tokens":
        raise ClaudeCodeError(E_MAX_TOKENS)
    if stop == "model_context_window_exceeded":
        raise ClaudeCodeError(E_CONTEXT)
    text = envelope.get("result")
    if not isinstance(text, str) or not text.strip():
        raise ClaudeCodeError(E_EMPTY)
    return text, metadata(envelope, model, wall)


# ---------------------------------------------------------------- status (no model call, no usage)

def status(model: str) -> dict:
    """{"backend", "label", "available", "model", "binary", "version", "auth", "reason"} from `claude --version`
    and `claude auth status --json`, run with the call's own environment in an empty temporary directory.

    auth: "Claude Code 登录" / "CLAUDE_CODE_OAUTH_TOKEN" / "未登录"; an API-key login, or a third-party cloud
    provider, is reported but not available (it would not use the subscription). A token is only checked by
    the first call: `claude auth status` reports one without contacting the server.
    """
    out = {"backend": "claude_code", "label": LABEL, "available": False, "model": model,
           "binary": None, "version": None, "auth": AUTH_NONE, "reason": ""}
    binary, problem = resolve_binary()
    if problem:
        out["reason"] = problem
        return out
    out["binary"] = binary
    env = child_env()
    with tempfile.TemporaryDirectory(prefix="causeway-claude-status-") as cwd:
        try:
            ver = _exec([binary, "--version"], input="", cwd=cwd, env=env, timeout=CHECK_TIMEOUT_S)
            auth = _exec([binary, "auth", "status", "--json"], input="", cwd=cwd, env=env, timeout=CHECK_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            out["reason"] = f"{binary} 在 {CHECK_TIMEOUT_S:g} 秒内没有响应（查询版本和登录状态时）。"
            return out
        except OSError as err:
            out["reason"] = f"无法运行 {binary}（{err.strerror or type(err).__name__}）。{INSTALL_HINT}"
            return out
    m = re.search(r"\d+\.\d+\.\d+[^\s()]*", ver.stdout or "")
    out["version"] = m.group(0) if m else None
    info = parse_json_object(auth.stdout)                 # signed out: JSON with loggedIn false, exit code 1
    if info is None:
        out["reason"] = (f"无法读取 Claude Code 的登录状态（claude auth status --json 没有输出 JSON，"
                         f"退出码 {auth.returncode}）。")
        return out
    method = str(info.get("authMethod") or "none")
    provider = str(info.get("apiProvider") or "firstParty")
    if info.get("subscriptionType"):
        out["subscription"] = str(info["subscriptionType"])
    if not info.get("loggedIn") or method == "none":
        out["reason"] = LOGIN_HINT
    elif provider != "firstParty":
        out.update(auth=provider, reason=f"Claude Code 配置为第三方云服务（{provider}），不走你的 Claude 订阅额度："
                                         "在终端运行 claude auth login，用 Claude 订阅账号登录。")
    elif method == "api_key":
        out.update(auth=AUTH_API_KEY, reason="Claude Code 当前用的是 API Key（按 API 计费，不用订阅额度）："
                                             "在终端运行 claude auth login，选择 Claude 订阅账号登录。")
    elif method == "oauth_token":
        out.update(available=True, auth=AUTH_TOKEN,
                   reason="使用 CLAUDE_CODE_OAUTH_TOKEN（claude setup-token 生成的令牌；它是否有效要到第一次翻译时才知道）")
    else:
        out.update(available=True, auth=AUTH_LOGIN, reason="使用 Claude Code 自己的登录（claude auth login）")
    return out
