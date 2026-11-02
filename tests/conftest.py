"""Shared fixtures. No test ever starts the real Claude Code CLI or spends plan usage: `no_live_llm` makes
every Claude Code process fail as if the CLI were not installed, and `fake_cli` (the only way back) answers
for it with a fake subprocess.run. The translator's response cache and status cache are per test."""
import errno
import json
import subprocess

import pytest

from scenariolab.llm import claude_code as cc
from scenariolab.llm import translator as tr

REAL_EXEC = cc._exec


def _no_live_cli(args, **kwargs):
    raise FileNotFoundError(errno.ENOENT, "tests never start Claude Code (tests/conftest.py)")


@pytest.fixture(autouse=True)
def no_live_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_exec", _no_live_cli)
    monkeypatch.setattr(tr, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(tr, "_STATUS", {})
    monkeypatch.setattr(tr, "_SLOTS", {})


def success_envelope(result: str, **extra) -> dict:
    """A print-mode JSON result as Claude Code 2.1 writes it (the keys observed on a real run)."""
    env = {"type": "result", "subtype": "success", "is_error": False, "result": result, "num_turns": 1,
           "duration_ms": 41234, "duration_api_ms": 40000, "total_cost_usd": 0.42,
           "usage": {"input_tokens": 12, "cache_creation_input_tokens": 9000, "cache_read_input_tokens": 0,
                     "output_tokens": 2400, "service_tier": "standard"},
           "modelUsage": {"claude-opus-5-5": {"inputTokens": 12, "outputTokens": 2400, "cacheReadInputTokens": 0,
                                              "cacheCreationInputTokens": 9000, "costUSD": 0.42}},
           "api_error_status": None, "session_id": "00000000-0000-0000-0000-000000000000",
           "stop_reason": "end_turn", "terminal_reason": "completed", "permission_denials": []}
    env.update(extra)
    return env


class FakeCLI:
    """Stands in for subprocess.run while the module runs `claude`: records every (args, kwargs); --version,
    `auth status --json` and print mode answer from the attributes below."""

    def __init__(self, binary: str):
        self.binary = binary
        self.calls: list[tuple[list, dict]] = []
        self.version = "2.1.278 (Claude Code)"
        self.auth = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}
        self.auth_rc = 0
        self.result = '{"id": "fake", "phases": []}'
        self.envelope = None          # dict or str: replaces success_envelope(self.result)
        self.returncode = 0
        self.stderr = ""
        self.error = None             # raised by print mode, e.g. subprocess.TimeoutExpired
        self.on_print = None          # callback(args, kwargs), while the temporary files still exist

    @staticmethod
    def success(result: str | None = None, **extra) -> dict:
        return success_envelope(result, **extra)

    def __call__(self, args, **kwargs):
        args = list(args)
        self.calls.append((args, kwargs))
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, self.version + "\n", "")
        if args[1:] == ["auth", "status", "--json"]:
            out = self.auth if isinstance(self.auth, str) else json.dumps(self.auth, indent=2)
            return subprocess.CompletedProcess(args, self.auth_rc, out, "")
        if args[1:2] == ["-p"]:
            if self.on_print is not None:
                self.on_print(args, kwargs)
            if self.error is not None:
                raise self.error
            env = success_envelope(self.result) if self.envelope is None else self.envelope
            out = env if isinstance(env, str) else json.dumps(env, ensure_ascii=False)
            return subprocess.CompletedProcess(args, self.returncode, out, self.stderr)
        raise AssertionError(f"unexpected command: {args}")

    @property
    def print_calls(self) -> list[tuple[list, dict]]:
        return [(a, k) for a, k in self.calls if a[1:2] == ["-p"]]


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    """The claude_code backend against a FakeCLI: a dummy executable as MSL_CLAUDE_BIN (never run), the real
    _exec with subprocess.run faked, and no .env, model, backend, timeout or token settings."""
    binary = tmp_path / "bin" / "claude"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    binary.chmod(0o755)
    for var in ("MSL_LLM_BACKEND", "MSL_LLM_MODEL", "MSL_LLM_TIMEOUT", "MSL_LLM_CONCURRENCY", cc.TOKEN_VAR):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MSL_CLAUDE_BIN", str(binary))
    monkeypatch.setattr(tr, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(tr, "dotenv_problems", lambda *a, **k: [])
    fake = FakeCLI(str(binary))
    monkeypatch.setattr(cc, "_exec", REAL_EXEC)
    monkeypatch.setattr(subprocess, "run", fake)
    return fake
