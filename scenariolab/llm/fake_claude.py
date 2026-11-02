"""A fake `claude` executable for tests and offline demos: never the real CLI, never a model call.

    python -m scenariolab.llm.fake_claude <the arguments claude_code passes>

It speaks the subset of Claude Code's command line that claude_code.py uses: `--version`, `--help` (listing
--effort and --safe-mode unless MSL_FAKE_CLAUDE_FLAGS=none), `auth status --json`, and print mode (`-p … --system-
prompt-file <file> --output-format json` with the user message on stdin), answering through MockClaude and
printing a JSON envelope like Claude Code 2.1's. MSL_FAKE_CLAUDE_MODE picks a failure instead:

    ok (default) · sleep:<seconds> (then ok; with MSL_FAKE_CLAUDE_CHILD_PIDFILE it first starts a grandchild
    `sleep` and writes its pid, so a test can check that a timeout kills the whole process group) · malformed ·
    refusal · limit · login · nojson · exit:<code>

MSL_FAKE_CLAUDE_LOG=<file> appends one JSON line per invocation (argv, the working directory's listing, the
credential-like variable names it can see), so tests can check what the CLI was given.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

VERSION = "2.1.278 (Claude Code)"
HELP_HEAD = """Usage: claude [options] [command] [prompt]

Claude Code - starts an interactive session by default, use -p/--print for non-interactive output

Options:
  -p, --print                           Print response and exit (useful for pipes)
  --model <model>                       Model for the current session
  --system-prompt-file <file>           System prompt file
  --tools <tools...>                    Specify the list of available tools
  --strict-mcp-config                   Only use MCP servers from --mcp-config
  --setting-sources <sources>           Comma-separated list of setting sources
  --no-session-persistence              Disable session persistence
  --output-format <format>              Output format (only works with --print)
"""
HELP_EFFORT = """  --effort <level>                      Effort level for the current session
                                        (low, medium, high, xhigh, max)
"""
HELP_SAFE = """  --safe-mode                           Start with all customizations disabled
"""


def _log(argv: list[str]) -> None:
    path = os.environ.get("MSL_FAKE_CLAUDE_LOG")
    if not path:
        return
    try:
        listing = sorted(os.listdir(os.getcwd()))
    except OSError:
        listing = None
    seen = sorted(k for k in os.environ if k.startswith(("ANTHROPIC_", "CLAUDE")) or k.endswith(("_API_KEY", "_TOKEN")))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"argv": argv, "cwd_listing": listing, "env": seen}, ensure_ascii=False) + "\n")


def _after(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def _envelope(result, **extra) -> dict:
    env = {"type": "result", "subtype": "success", "is_error": False, "result": result, "num_turns": 1,
           "duration_ms": 1200, "duration_api_ms": 1000, "total_cost_usd": 0.0,
           "usage": {"input_tokens": 10, "output_tokens": 20, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 0},
           "modelUsage": {}, "api_error_status": None, "stop_reason": "end_turn"}
    env.update(extra)
    return env


def _print_mode(argv: list[str]) -> int:
    system_file = _after(argv, "--system-prompt-file")
    model = _after(argv, "--model") or "claude-opus-5-5"
    system = open(system_file, encoding="utf-8").read() if system_file else ""
    user = sys.stdin.read()
    mode = os.environ.get("MSL_FAKE_CLAUDE_MODE", "ok").strip() or "ok"
    if mode.startswith("sleep:"):
        pidfile = os.environ.get("MSL_FAKE_CLAUDE_CHILD_PIDFILE")
        if pidfile:
            child = subprocess.Popen(["sleep", "60"])
            with open(pidfile, "w", encoding="utf-8") as fh:
                fh.write(str(child.pid))
        time.sleep(float(mode.split(":", 1)[1]))
        mode = "ok"
    if mode.startswith("exit:"):
        print("boom", file=sys.stderr)
        return int(mode.split(":", 1)[1])
    if mode == "nojson":
        print("Error: not json")
        return 1
    if mode == "limit":
        print(json.dumps(_envelope(f"Claude AI usage limit reached|{int(time.time()) + 3600}", is_error=True,
                                   api_error_status=429)))
        return 1
    if mode == "login":
        print(json.dumps(_envelope("Failed to authenticate. API Error: 401 OAuth access token is invalid.",
                                   is_error=True, api_error_status=401)))
        return 1
    if mode == "refusal":
        print(json.dumps(_envelope("", stop_reason="refusal")))
        return 0
    from .mock import MockClaude                       # imported late: --version and --help stay instant
    text, meta = MockClaude().run(system, user, model)
    if mode == "malformed":
        text = text[:max(10, len(text) // 2)]
    usage = meta.get("usage") or {}
    print(json.dumps(_envelope(text, usage=usage, modelUsage={model: {"inputTokens": usage.get("input_tokens", 0),
                                                                      "outputTokens": usage.get("output_tokens", 0)}}),
                     ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _log(argv)
    if argv[:1] == ["--version"]:
        print(VERSION)
        return 0
    if argv[:1] == ["--help"]:
        flags = os.environ.get("MSL_FAKE_CLAUDE_FLAGS", "all").strip()
        print(HELP_HEAD + ("" if flags == "none" else HELP_EFFORT + HELP_SAFE), end="")
        return 0
    if argv[:3] == ["auth", "status", "--json"]:
        print(json.dumps({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}))
        return 0
    if "-p" in argv:
        return _print_mode(argv)
    print(f"fake claude: unexpected arguments {argv}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
