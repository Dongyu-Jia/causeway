"""The translator's default backend: the local Claude Code CLI in print mode, on the user's own
subscription login (本机 Claude Code（使用你的订阅额度）).

    claude -p --model <id> --system-prompt-file <file> --tools "" --strict-mcp-config \
        --setting-sources project --no-session-persistence --output-format json   < user message
        [--effort <level>] [--safe-mode]        (only when asked for and `claude --help` lists them)

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
* The CLI has no timeout flag: the subprocess has one (MSL_LLM_TIMEOUT, 600 s by default). The child runs in its
  own process group (a new session), and a timeout or a cancel (a threading.Event) kills the whole group, so no
  helper process the CLI started outlives the call.
* Two identical requests at once (same model, flags, system prompt and user message) make one call: the second
  waits for the first and gets its answer (meta["shared_inflight"]), so it spends no usage.

Failures raise ClaudeCodeError with a Chinese message and a machine-readable `kind` (KINDS), so callers choose
between retry, repair, fallback and pause without parsing the message. The agents' runner (scenariolab/agents/
llm_io.py) maps the kinds; the translator only shows the message.

Every call counts against the plan's five-hour and weekly limits. status() never makes one: it runs only
`claude --version` and `claude auth status --json`; supported_flags() runs only `claude --help`. Personal local
use only: each user runs their own signed-in Claude Code; nothing here proxies requests for anyone else or stores
or forwards credentials.

Output: the text path, not --json-schema. SPEC_JSON_SCHEMA would be accepted there (plain types, required,
enum, anyOf, additionalProperties false; no $ref), and the CLI would return a validated `structured_output`,
but Claude Code does structured output as a tool call that it validates and retries on a mismatch (result
subtype error_max_structured_output_retries): more requests, and so more plan usage, per translation than
one text answer, and nothing here could check it without spending usage. The schema goes into the prompt
instead (translator.schema_section), and the answer takes the API path's parse -> repair -> validate_spec;
the schema holds no numeric bounds by design, so validate_spec is the guard rail either way.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

LABEL = "本机 Claude Code（使用你的订阅额度）"
AUTH_LOGIN = "Claude Code 登录"
AUTH_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"
AUTH_NONE = "未登录"
AUTH_API_KEY = "API Key（按 API 计费）"
TOKEN_VAR = "CLAUDE_CODE_OAUTH_TOKEN"
IDENTITY_NAME = re.compile(r"USER_?AGENT|E_?MAIL", re.I)   # SEC_USER_AGENT (the user's email): sec.gov only
KEEP_CLAUDE_VARS = ("CLAUDE_CONFIG_DIR", TOKEN_VAR)       # the CLI's own config dir and the setup-token
SECRET_VARS = (TOKEN_VAR, "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
DEFAULT_TIMEOUT_S = 600.0
CHECK_TIMEOUT_S = 15.0                                   # claude --version, claude auth status --json, claude --help
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
EXCERPT_CHARS = 200
DETAIL_CHARS = 500                                       # ClaudeCodeError.detail: the CLI's own words, redacted
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")  # `claude --help` 2.1.278
KILL_WAIT_S = 5.0
_POSIX = os.name == "posix"

# Machine-readable failure classes (ClaudeCodeError.kind). The agents' runner maps them: retry (timeout, rate,
# overloaded, server), repair or fallback (empty, unparseable, failed, max_turns), prior-only (refusal,
# max_tokens, context), pause (limit, login, token, cli_missing, model, forbidden) and cancelled.
KINDS = ("limit", "login", "token", "rate", "overloaded", "server", "model", "forbidden", "context", "max_tokens",
         "refusal", "timeout", "empty", "cli_missing", "cancelled", "max_turns", "unparseable", "failed")

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
E_CANCELLED = "Claude Code 调用已取消（进程已结束，没有结果）。"

# credential-shaped strings, removed from any CLI text a message quotes
TOKENISH = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}|[A-Za-z0-9_\-]{40,}")
# credential-named variables (FRED_API_KEY, GITHUB_TOKEN, AWS_SECRET_ACCESS_KEY, ...): never passed to the CLI,
# which needs none of them, and their values are redacted like SECRET_VARS (MAX_THINKING_TOKENS does not match)
SECRET_NAME = re.compile(r"(?:^|_)(?:API_?KEY|ACCESS_KEY|SECRET(?:_KEY)?|PRIVATE_KEY|PASSWORD|PASSWD|PASSPHRASE|TOKEN"
                         r"|CREDENTIALS?)$", re.I)


class ClaudeCodeError(Exception):
    """A failed Claude Code call; the message is user-facing (Chinese) and never holds a token or the environment.

    kind: one of KINDS ("failed" when nothing more specific applies); reset_at: epoch seconds when the CLI said a
    usage limit resets (kind "limit" only, else None); detail: a short redacted excerpt of the CLI's own failure
    text, for the first-failure log (never shown as the message)."""

    def __init__(self, message: str, kind: str = "failed", *, reset_at: float | None = None, detail: str = ""):
        super().__init__(message)
        self.kind = kind if kind in KINDS else "failed"
        self.reset_at = reset_at
        self.detail = detail


class _Cancelled(Exception):
    """Raised by _exec when the caller's cancel event fired; the process group was killed."""


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
    (CLAUDECODE, CLAUDE_*), every other credential-named one (SECRET_NAME: the .env's FRED_API_KEY, a
    GITHUB_TOKEN, ...) and every identity-bearing one (IDENTITY_NAME: SEC_USER_AGENT holds the user's email and goes
    only to sec.gov), keeping CLAUDE_CONFIG_DIR and CLAUDE_CODE_OAUTH_TOKEN when it is not empty."""
    src = os.environ if environ is None else environ
    env = {k: v for k, v in src.items()
           if not (k.startswith("ANTHROPIC_") or k == "CLAUDECODE"
                   or (k.startswith("CLAUDE_") and k not in KEEP_CLAUDE_VARS)
                   or (k != TOKEN_VAR and SECRET_NAME.search(k))
                   or IDENTITY_NAME.search(k))}
    token = str(src.get(TOKEN_VAR) or "").strip()
    if token:
        env[TOKEN_VAR] = token
    else:
        env.pop(TOKEN_VAR, None)
    return env


def build_args(binary, model: str, system_file, *, effort: str | None = None, safe_mode: bool = False) -> list[str]:
    """The print-mode argument list (no shell; the user message goes in on stdin). `effort` and `safe_mode` add
    --effort <level> and --safe-mode; run() passes them only when the installed CLI lists them."""
    args = [str(binary), "-p", "--model", model, "--system-prompt-file", str(system_file),
            "--tools", "", "--strict-mcp-config", "--setting-sources", "project",
            "--no-session-persistence", "--output-format", "json"]
    if effort:
        args += ["--effort", effort]
    if safe_mode:
        args.append("--safe-mode")
    return args


def _popen(args: list[str], **kwargs) -> subprocess.Popen:
    """The one place this module starts a process: an argument list, never a shell (tests replace it). The child
    gets its own session, so its process group can be killed as a whole."""
    return subprocess.Popen(args, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            start_new_session=_POSIX, **kwargs)


def _kill_group(proc) -> None:
    """SIGKILL the child's process group (the CLI and anything it started), else the child alone."""
    pid = getattr(proc, "pid", None)
    if _POSIX and isinstance(pid, int) and pid > 0:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _reap(proc) -> tuple[str, str]:
    """After a kill: collect what the child wrote and close its pipes, without hanging on an escaped grandchild."""
    try:
        out, err = proc.communicate(timeout=KILL_WAIT_S)
        return out or "", err or ""
    except (subprocess.TimeoutExpired, ValueError, OSError):
        for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None), getattr(proc, "stdin", None)):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=KILL_WAIT_S)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return "", ""


def _exec(args: list[str], *, input: str | None = None, timeout: float | None = None,
          cancel: threading.Event | None = None, **kwargs) -> subprocess.CompletedProcess:
    """Run `args` to completion: (returncode, stdout, stderr) like subprocess.run, but the child is in its own
    process group, which is killed on a timeout (subprocess.TimeoutExpired is raised) or when `cancel` is set
    (_Cancelled is raised). kwargs: cwd, env."""
    proc = _popen(args, **kwargs)
    done = threading.Event()
    fired: list[bool] = []
    watcher = None
    if cancel is not None:
        def watch():
            while not done.is_set():
                if cancel.wait(0.05):
                    if not done.is_set():
                        fired.append(True)
                        _kill_group(proc)
                    return

        watcher = threading.Thread(target=watch, name="claude-cancel-watch", daemon=True)
        watcher.start()
    try:
        try:
            out, err = proc.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            _reap(proc)
            raise
        except BaseException:
            _kill_group(proc)
            _reap(proc)
            raise
    finally:
        done.set()
        if watcher is not None:
            watcher.join(1.0)
    if fired:
        raise _Cancelled()
    return subprocess.CompletedProcess(args, proc.returncode, out or "", err or "")


# ---------------------------------------------------------------- `claude --help`: optional flags

_FLAGS: dict[tuple, dict] = {}            # (binary, mtime_ns, size) -> supported flags
_FLAGS_LOCK = threading.Lock()


def parse_help(text: str) -> dict:
    """{"effort": bool, "effort_levels": [...], "safe_mode": bool} from `claude --help` output."""
    s = str(text or "")
    effort = re.search(r"(?m)^\s*--effort\b", s)
    levels: list[str] = []
    if effort:
        m = re.search(r"\(([a-z, ]+)\)", s[effort.end():effort.end() + 400])
        if m:
            levels = [x.strip() for x in m.group(1).split(",") if x.strip()]
        levels = [x for x in (levels or list(EFFORT_LEVELS)) if re.fullmatch(r"[a-z]+", x)]
    return {"effort": bool(effort), "effort_levels": levels if effort else [],
            "safe_mode": bool(re.search(r"(?m)^\s*--safe-mode\b", s))}


def supported_flags(binary: str | None = None, force: bool = False) -> dict:
    """The optional flags the installed CLI lists in `claude --help` (no model call, no usage), probed once per
    binary version (path, mtime, size): {"effort", "effort_levels", "safe_mode", "probed", "reason"}. Nothing is
    used that the help text does not list; an unreadable help means none."""
    none = {"effort": False, "effort_levels": [], "safe_mode": False, "probed": False, "reason": ""}
    if binary is None:
        binary, problem = resolve_binary()
        if problem:
            return {**none, "reason": problem}
    try:
        st = os.stat(binary)
        key = (str(binary), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(binary), 0, 0)
    with _FLAGS_LOCK:
        if not force and key in _FLAGS:
            return dict(_FLAGS[key])
    with tempfile.TemporaryDirectory(prefix="causeway-claude-help-") as cwd:
        try:
            proc = _exec([str(binary), "--help"], input="", cwd=cwd, env=child_env(), timeout=CHECK_TIMEOUT_S)
        except (subprocess.TimeoutExpired, OSError, _Cancelled) as err:
            return {**none, "reason": f"无法读取 claude --help（{type(err).__name__}），不使用可选参数。"}
    out = {**parse_help(proc.stdout), "probed": proc.returncode == 0, "reason": ""}
    if proc.returncode != 0:
        out.update(effort=False, effort_levels=[], safe_mode=False,
                   reason=f"claude --help 退出码 {proc.returncode}，不使用可选参数。")
    with _FLAGS_LOCK:
        _FLAGS[key] = dict(out)
    return out


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


def _excerpt(*texts, limit: int = EXCERPT_CHARS) -> str:
    """A short single-line, redacted quote of the first non-empty text."""
    for t in texts:
        line = " ".join(redact(t).split())
        if line:
            return line if len(line) <= limit else line[:limit].rstrip() + "…"
    return "没有输出"


def _int(x) -> int | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return int(x)


def _num(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


_RESET_AT = re.compile(r"resets?\s+(?:at\s+)?(?:(mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?\s+)?(\d{1,2})(?::(\d{2}))?"
                       r"\s*(am|pm)?(?:\s*\(([A-Za-z_]+(?:/[A-Za-z_+\-]+){0,2})\))?", re.I)
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _reset_epoch(text: str, now: float | None = None) -> float | None:
    """The epoch seconds a usage-limit message names: "…|<epoch seconds>", or "resets 3pm (America/Los_Angeles)",
    "resets Mon 9am", "resets 11:30pm" (the next such time, in the named zone or the local one); None otherwise."""
    m = re.search(r"\|(\d{9,11})\b", text)
    if m:
        return float(m.group(1))
    m = _RESET_AT.search(text or "")
    if not m:
        return None
    day, hh, mm, ampm, zone = m.groups()
    if not ampm and mm is None:
        return None                                   # "resets 3" alone is ambiguous
    hour, minute = int(hh), int(mm or 0)
    if ampm:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if ampm.lower() == "pm" else 0)
    if hour > 23 or minute > 59:
        return None
    import datetime as _dt
    tz = None
    if zone:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(zone)
        except Exception:  # noqa: BLE001 - an unknown zone: the local time zone
            tz = None
    t0 = _dt.datetime.fromtimestamp(time.time() if now is None else float(now), tz=tz or _dt.datetime.now().astimezone().tzinfo)
    cand = t0.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day:
        ahead = (_WEEKDAYS.index(day.lower()[:3]) - cand.weekday()) % 7
        cand = cand + _dt.timedelta(days=ahead)
        if cand <= t0:
            cand = cand + _dt.timedelta(days=7)
    elif cand <= t0:
        cand = cand + _dt.timedelta(days=1)
    return float(cand.timestamp())


def _reset_hint(text: str) -> str:
    """When the limit resets, from the CLI's message: "…|<epoch seconds>" or "resets <time>"."""
    m = re.search(r"\|(\d{9,11})\b", text)
    if m:
        epoch = float(m.group(1))
        return f"（预计 {time.strftime('%m-%d %H:%M', time.localtime(epoch))} 恢复）"
    m = re.search(r"resets?\s+(?:at\s+)?([^\n|·∙]{1,40})", text, re.I)
    if m:
        return f"（{_excerpt(m.group(1)).rstrip('.。')} 重置）"
    return ""


def classify_failure(envelope: dict | None, returncode, stderr: str, model: str, token_used: bool,
                     stdout: str = "") -> tuple[str, str, float | None]:
    """(kind, Chinese message, reset_at) for a failed call, from the envelope's api_error_status / result and
    stderr. When stdout held no envelope it is quoted too (a plain-text CLI error), and classified only after a
    non-zero exit (with exit code 0 it could be the model's own words)."""
    env = envelope or {}
    status = _int(env.get("api_error_status"))
    stdout = stdout if envelope is None else ""
    text = " ".join(str(x) for x in (env.get("result"), env.get("error"), stderr, stdout if returncode != 0 else "")
                    if x)
    low = text.lower()
    if status == 401 or any(k in low for k in ("failed to authenticate", "invalid api key", "not logged in",
                                               "/login", "authentication_error")) \
            or ("oauth" in low and any(k in low for k in ("invalid", "expired", "revoked"))):
        return ("token", E_TOKEN, None) if token_used else ("login", E_LOGIN, None)
    rate = "rate limit" in low or "rate_limit" in low
    if any(k in low for k in ("usage limit", "hit your limit", "reached your limit", "5-hour limit", "weekly limit")) \
            or (not rate and ("limit reached" in low or ("limit" in low and "resets" in low))):
        return "limit", E_LIMIT.format(reset=_reset_hint(text)), _reset_epoch(text)
    if status == 429 or rate:
        return "rate", E_RATE, None
    if status == 403:
        return "forbidden", E_FORBIDDEN.format(model=model), None
    if status == 404 or ("model" in low and any(k in low for k in ("not found", "not_found", "does not exist",
                                                                    "issue with the selected model"))):
        return "model", E_MODEL.format(model=model), None
    if status == 529 or "overloaded" in low:
        return "overloaded", E_OVERLOADED, None
    if status is not None and status >= 500:
        return "server", E_SERVER.format(status=status), None
    if any(k in low for k in ("prompt is too long", "context window", "context_length")):
        return "context", E_CONTEXT, None
    if "output token" in low and any(k in low for k in ("maximum", "exceed")):
        return "max_tokens", E_MAX_TOKENS, None
    subtype = str(env.get("subtype") or "")
    if subtype == "error_max_turns":
        return "max_turns", f"Claude Code 在 {_int(env.get('num_turns')) or '若干'} 轮内没有给出结果（{subtype}）。", None
    where = f"退出码 {returncode}" + (f"，{subtype}" if subtype and subtype != "success" else "")
    if envelope is None:
        return "unparseable", f"Claude Code 的输出无法解析（{where}）：{_excerpt(stderr, stdout)}", None
    return "failed", f"Claude Code 调用失败（{where}）：{_excerpt(env.get('result'), env.get('error'), stderr)}", None


def failure_message(envelope: dict | None, returncode, stderr: str, model: str, token_used: bool,
                    stdout: str = "") -> str:
    """The Chinese message of classify_failure."""
    return classify_failure(envelope, returncode, stderr, model, token_used, stdout)[1]


def failure_error(envelope: dict | None, returncode, stderr: str, model: str, token_used: bool,
                  stdout: str = "") -> ClaudeCodeError:
    """The ClaudeCodeError for a failed call (kind, message, reset time and a redacted excerpt of the CLI text)."""
    kind, message, reset_at = classify_failure(envelope, returncode, stderr, model, token_used, stdout)
    env = envelope or {}
    detail = _excerpt(env.get("result"), env.get("error"), stderr, stdout, limit=DETAIL_CHARS)
    return ClaudeCodeError(message, kind, reset_at=reset_at, detail=detail)


def metadata(envelope: dict, requested_model: str, wall_s: float) -> dict:
    """What a call records: the model, duration, turns, stop reason, token usage (total and per model) and the
    envelope's API-equivalent cost figure (cost_usd: for the user's information only, the plan is not billed)."""
    usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
    tokens = {k: _int(usage[k]) for k in USAGE_KEYS if _int(usage.get(k)) is not None}
    per_model = envelope.get("modelUsage") if isinstance(envelope.get("modelUsage"), dict) else {}

    def output_tokens(m: str) -> int:
        v = per_model[m]
        return (_int(v.get("outputTokens")) or 0) if isinstance(v, dict) else 0

    by_model = {m: {k: v for k, v in per_model[m].items() if _num(v) is not None}
                for m in sorted(per_model) if isinstance(per_model[m], dict)}
    main = max(per_model, key=output_tokens) if per_model else None
    return {"backend": "claude_code", "model": main or requested_model, "models": sorted(per_model),
            "duration_ms": _int(envelope.get("duration_ms")), "duration_api_ms": _int(envelope.get("duration_api_ms")),
            "num_turns": _int(envelope.get("num_turns")), "stop_reason": envelope.get("stop_reason"),
            "usage": tokens, "usage_by_model": by_model, "cost_usd": _num(envelope.get("total_cost_usd")),
            "wall_s": round(wall_s, 2)}


# ---------------------------------------------------------------- one request at a time per key

def request_key(model: str, system: str, user: str, effort: str | None = None, safe_mode: bool = False) -> str:
    """sha256 of everything that shapes an answer (model, flags, system prompt, user message)."""
    blob = json.dumps([model, effort, bool(safe_mode), system, user], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()


class _Flight:
    __slots__ = ("done", "result", "error")

    def __init__(self):
        self.done = threading.Event()
        self.result: tuple[str, dict] | None = None
        self.error: ClaudeCodeError | None = None


class SingleFlight:
    """Per-key in-flight lock: while a request runs, an identical one waits for it and shares its outcome (the
    text and metadata, or the error), so two identical requests never both spend usage. A leader that was
    cancelled does not cancel its followers: the first of them asks again."""

    def __init__(self):
        self._lock = threading.Lock()
        self._flights: dict[str, _Flight] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._flights)

    def clear(self) -> None:
        with self._lock:
            self._flights.clear()

    def do(self, key: str, fn, cancel: threading.Event | None = None) -> tuple[str, dict]:
        while True:
            with self._lock:
                flight = self._flights.get(key)
                leader = flight is None
                if leader:
                    flight = self._flights[key] = _Flight()
            if leader:
                try:
                    flight.result = fn()
                    return flight.result
                except ClaudeCodeError as err:
                    flight.error = err
                    raise
                finally:
                    with self._lock:
                        self._flights.pop(key, None)
                    flight.done.set()
            while not flight.done.wait(0.05):
                if cancel is not None and cancel.is_set():
                    raise ClaudeCodeError(E_CANCELLED, "cancelled")
            err = flight.error
            if err is not None and err.kind != "cancelled":
                raise ClaudeCodeError(str(err), err.kind, reset_at=err.reset_at, detail=err.detail)
            if err is None and flight.result is not None:
                text, meta = flight.result
                return text, {**meta, "shared_inflight": True}
            # the leader was cancelled (or died without an answer): ask again, as the next leader


_INFLIGHT = SingleFlight()


# ---------------------------------------------------------------- the call

def _flags_for(binary: str, effort: str | None, safe_mode: bool) -> tuple[str | None, bool, list[str]]:
    """The effort level and safe mode actually passed (only what `claude --help` lists), plus Chinese notes."""
    if not effort and not safe_mode:
        return None, False, []
    flags = supported_flags(binary)
    notes = []
    use_effort = None
    if effort:
        if flags["effort"] and effort in (flags["effort_levels"] or EFFORT_LEVELS):
            use_effort = effort
        else:
            notes.append(f"本机 Claude Code 不支持 --effort {effort}，已省略")
    use_safe = bool(safe_mode and flags["safe_mode"])
    if safe_mode and not use_safe:
        notes.append("本机 Claude Code 不支持 --safe-mode，已省略")
    return use_effort, use_safe, notes


def _run_once(binary: str, system: str, user: str, model: str, timeout: float, effort: str | None,
              safe_mode: bool, cancel: threading.Event | None, notes: list[str]) -> tuple[str, dict]:
    env = child_env()
    with tempfile.TemporaryDirectory(prefix="causeway-claude-") as tmp:
        cwd = Path(tmp) / "cwd"
        cwd.mkdir()                                       # empty: the system prompt file sits beside it
        system_file = Path(tmp) / "system-prompt.md"
        system_file.write_text(system, encoding="utf-8")
        t0 = time.perf_counter()
        try:
            proc = _exec(build_args(binary, model, system_file, effort=effort, safe_mode=safe_mode), input=user,
                         cwd=str(cwd), env=env, timeout=timeout, cancel=cancel)
        except subprocess.TimeoutExpired:
            raise ClaudeCodeError(E_TIMEOUT.format(seconds=timeout), "timeout") from None
        except _Cancelled:
            raise ClaudeCodeError(E_CANCELLED, "cancelled") from None
        except OSError as err:
            raise ClaudeCodeError(f"无法启动 Claude Code（{binary}：{err.strerror or type(err).__name__}）。"
                                  f"{INSTALL_HINT}", "cli_missing") from None
        wall = time.perf_counter() - t0
    envelope = parse_json_object(proc.stdout)
    if (proc.returncode != 0 or envelope is None or envelope.get("is_error")
            or str(envelope.get("subtype") or "success") != "success"):
        raise failure_error(envelope, proc.returncode, proc.stderr, model, TOKEN_VAR in env, stdout=proc.stdout)
    stop = envelope.get("stop_reason")
    if stop == "refusal":
        raise ClaudeCodeError(E_REFUSAL, "refusal")
    if stop == "max_tokens":
        raise ClaudeCodeError(E_MAX_TOKENS, "max_tokens")
    if stop == "model_context_window_exceeded":
        raise ClaudeCodeError(E_CONTEXT, "context")
    text = envelope.get("result")
    if not isinstance(text, str) or not text.strip():
        raise ClaudeCodeError(E_EMPTY, "empty")
    meta = metadata(envelope, model, wall)
    meta.update(effort=effort, safe_mode=safe_mode)
    if notes:
        meta["flag_notes"] = notes
    return text, meta


def run(system: str, user: str, model: str, timeout: float = DEFAULT_TIMEOUT_S, *, effort: str | None = None,
        safe_mode: bool = False, cancel: threading.Event | None = None) -> tuple[str, dict]:
    """One print-mode call: (the answer's text, metadata). ClaudeCodeError on any failure (not signed in, an
    invalid token, the usage limit, a timeout, a cancel, ...), with a `kind` and a message that holds no token
    and no environment.

    effort ("low" … "max") and safe_mode add --effort and --safe-mode, each only if `claude --help` lists it (the
    metadata says what was used); the defaults leave the translator's argument list unchanged. cancel: a
    threading.Event; setting it kills the call's process group (kind "cancelled"). An identical request already
    in flight is shared, not repeated (meta["shared_inflight"])."""
    binary, problem = resolve_binary()
    if problem:
        raise ClaudeCodeError(problem, "cli_missing")
    if cancel is not None and cancel.is_set():
        raise ClaudeCodeError(E_CANCELLED, "cancelled")
    use_effort, use_safe, notes = _flags_for(binary, effort, safe_mode)
    key = request_key(model, system, user, use_effort, use_safe)
    return _INFLIGHT.do(key, lambda: _run_once(binary, system, user, model, timeout, use_effort, use_safe, cancel,
                                               notes), cancel)


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
