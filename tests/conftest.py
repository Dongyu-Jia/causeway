"""Shared fixtures. No test ever starts the real Claude Code CLI or spends plan usage.

* `no_live_llm` (every test): MSL_LLM_BACKEND=mock, so the translator, the agent tree and the server answer from
  MockClaude (scenariolab/llm/mock.py), and MSL_AGENT_MODEL=stub, so the agent tree prices with the Gaussian stub
  (tests of the real quarterly model opt in and skip without data/); every Claude Code process fails as if the CLI were not installed
  (claude_code._exec and _popen are replaced); the mock's fixture queue, the CLI flag probe, the in-flight lock
  table and the translator's response and status caches are per test.
* `mock_llm`: the MockClaude instance, to queue fixtures (malformed JSON, refusals, timeouts, usage limits, …).
* `fake_cli` (the only way back to the claude_code backend): answers for the CLI with a fake subprocess.Popen,
  never a process.
* `fake_claude_bin`: a real subprocess, but of the fake executable scenariolab/llm/fake_claude.py (for the
  process-group kill and the --help probe); any other binary is refused.
* `worldsim_isolated` (every test): WorldSim keeps its data under the test's tmp_path, never fetches the model
  list (WORLDSIM_OFFLINE), its mock answers at once, and its OpenRouter transport refuses every request: a test
  that needs OpenRouter passes its own fake transport. No test can reach openrouter.ai.
"""
import errno
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from scenariolab.llm import backend as llm_backend
from scenariolab.llm import claude_code as cc
from scenariolab.llm import mock as llm_mock
from scenariolab.llm import translator as tr
from worldsim import openrouter as ws_openrouter

REAL_EXEC = cc._exec
REAL_POPEN = cc._popen
ROOT = Path(__file__).resolve().parent.parent
MOCK_VARS = ("MSL_LLM_MOCK_MODE", "MSL_LLM_MOCK_SEED", "MSL_LLM_MOCK_DELAY_S", "MSL_LLM_MOCK_FIXTURE",
             "MSL_LLM_MOCK_CASSETTES", "MSL_FAKE_CLAUDE_MODE", "MSL_FAKE_CLAUDE_FLAGS", "MSL_FAKE_CLAUDE_LOG",
             "MSL_FAKE_CLAUDE_CHILD_PIDFILE")


def _no_live_cli(args, **kwargs):
    raise FileNotFoundError(errno.ENOENT, "tests never start Claude Code (tests/conftest.py)")


@pytest.fixture(autouse=True)
def no_live_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_exec", _no_live_cli)
    monkeypatch.setattr(cc, "_popen", _no_live_cli)
    monkeypatch.setenv("MSL_LLM_BACKEND", "mock")
    # the agent tree prices with the Gaussian stub in tests (scenariolab.agents.model_adapter.load_adapter would
    # otherwise pick the quarterly stack when data/ exists); tests of the real model opt in and skip without data
    monkeypatch.setenv("MSL_AGENT_MODEL", "stub")
    for var in MOCK_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cc, "_FLAGS", {})
    monkeypatch.setattr(cc, "_INFLIGHT", cc.SingleFlight())
    monkeypatch.setattr(tr, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(tr, "_STATUS", {})
    monkeypatch.setattr(tr, "_SLOTS", {})
    llm_backend.clear_caches()
    llm_mock.reset()
    yield
    llm_mock.reset()
    llm_backend.clear_caches()


def _no_network(method, url, headers, body, timeout):
    raise AssertionError(f"tests never reach the network (WorldSim {method} {url})")


@pytest.fixture(autouse=True)
def worldsim_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_DATA_DIR", str(tmp_path / "worldsim"))
    monkeypatch.setenv("WORLDSIM_OFFLINE", "1")
    monkeypatch.setenv("WORLDSIM_MOCK_DELAY_S", "0")
    monkeypatch.delenv("WORLDSIM_LLM", raising=False)
    monkeypatch.delenv("WORLDSIM_PORT", raising=False)
    monkeypatch.delenv("WORLDSIM_MOCK_SCRIPT", raising=False)
    ws_openrouter.set_transport(_no_network)
    yield
    ws_openrouter.set_transport(_no_network)


@pytest.fixture
def mock_llm():
    """The process-wide MockClaude: mock_llm.push("usage_limit", match="synth"), mock_llm.calls, …"""
    return llm_mock.MOCK


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


FAKE_HELP = """Usage: claude [options]
  --effort <level>                      Effort level for the current session
                                        (low, medium, high, xhigh, max)
  --safe-mode                           Start with all customizations disabled
"""


class FakeProcess:
    """What subprocess.Popen returns under FakeCLI: communicate() answers once (FakeCLI.respond); kill() ends a
    hanging answer; there is never a real pid, so claude_code kills it with kill() instead of killpg()."""

    def __init__(self, fake, args, kwargs):
        self.fake, self.args, self.kwargs = fake, args, kwargs
        self.pid = None
        self.returncode = None
        self.killed = threading.Event()
        self._answered = False
        self.stdin = self.stdout = self.stderr = None

    def communicate(self, input=None, timeout=None):
        if self._answered or self.killed.is_set():
            return "", ""
        self._answered = True
        self.kwargs.update(input=input, timeout=timeout)
        out = self.fake.respond(self.args, self.kwargs, self, timeout)
        self.returncode = out.returncode
        return out.stdout, out.stderr

    def kill(self):
        self.killed.set()
        self.fake.kills += 1
        if self.returncode is None:
            self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


class FakeCLI:
    """Stands in for subprocess.Popen while the module runs `claude`: records every (args, kwargs) (kwargs include
    the stdin `input` and the `timeout`); --version, --help, `auth status --json` and print mode answer from the
    attributes below."""

    def __init__(self, binary: str):
        self.binary = binary
        self.calls: list[tuple[list, dict]] = []
        self.version = "2.1.278 (Claude Code)"
        self.help = FAKE_HELP
        self.auth = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}
        self.auth_rc = 0
        self.result = '{"id": "fake", "phases": []}'
        self.envelope = None          # dict or str: replaces success_envelope(self.result)
        self.returncode = 0
        self.stderr = ""
        self.error = None             # raised by print mode: an OSError when starting, else from communicate()
        self.on_print = None          # callback(args, kwargs), while the temporary files still exist
        self.hang = False             # print mode blocks until killed (or its timeout passes)
        self.kills = 0

    @staticmethod
    def success(result: str | None = None, **extra) -> dict:
        return success_envelope(result, **extra)

    def popen(self, args, **kwargs):
        args = list(args)
        self.calls.append((args, kwargs))
        if args[1:2] == ["-p"] and isinstance(self.error, OSError):
            raise self.error
        return FakeProcess(self, args, kwargs)

    __call__ = popen

    def respond(self, args, kwargs, proc, timeout):
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, self.version + "\n", "")
        if args[1:] == ["--help"]:
            return subprocess.CompletedProcess(args, 0, self.help, "")
        if args[1:] == ["auth", "status", "--json"]:
            out = self.auth if isinstance(self.auth, str) else json.dumps(self.auth, indent=2)
            return subprocess.CompletedProcess(args, self.auth_rc, out, "")
        if args[1:2] == ["-p"]:
            if self.on_print is not None:
                self.on_print(args, kwargs)
            if self.error is not None:
                raise self.error
            if self.hang:
                if not proc.killed.wait(timeout if timeout is not None else 30):
                    raise subprocess.TimeoutExpired(args, timeout)
                return subprocess.CompletedProcess(args, -9, "", "")
            env = success_envelope(self.result) if self.envelope is None else self.envelope
            out = env if isinstance(env, str) else json.dumps(env, ensure_ascii=False)
            return subprocess.CompletedProcess(args, self.returncode, out, self.stderr)
        raise AssertionError(f"unexpected command: {args}")

    @property
    def print_calls(self) -> list[tuple[list, dict]]:
        return [(a, k) for a, k in self.calls if a[1:2] == ["-p"]]


def _clean_backend_env(monkeypatch):
    for var in ("MSL_LLM_BACKEND", "MSL_LLM_MODEL", "MSL_LLM_TIMEOUT", "MSL_LLM_CONCURRENCY", cc.TOKEN_VAR):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tr, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(tr, "dotenv_problems", lambda *a, **k: [])
    monkeypatch.setattr(llm_backend, "load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    """The claude_code backend against a FakeCLI: a dummy executable as MSL_CLAUDE_BIN (never run), the real
    _exec and _popen with subprocess.Popen faked, and no .env, model, backend, timeout or token settings."""
    binary = tmp_path / "bin" / "claude"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    binary.chmod(0o755)
    _clean_backend_env(monkeypatch)
    monkeypatch.setenv("MSL_CLAUDE_BIN", str(binary))
    fake = FakeCLI(str(binary))
    monkeypatch.setattr(cc, "_exec", REAL_EXEC)
    monkeypatch.setattr(cc, "_popen", REAL_POPEN)
    monkeypatch.setattr(subprocess, "Popen", fake.popen)
    return fake


@pytest.fixture
def fake_claude_bin(monkeypatch, tmp_path):
    """MSL_CLAUDE_BIN -> a shell wrapper of `python -m scenariolab.llm.fake_claude`: real processes, the fake CLI.
    Returns the wrapper's path; MSL_FAKE_CLAUDE_* settings pass through the child environment."""
    if os.name != "posix":
        pytest.skip("POSIX shell wrapper")
    binary = tmp_path / "fakebin" / "claude"
    binary.parent.mkdir()
    binary.write_text(f'#!/bin/sh\nPYTHONPATH="{ROOT}" exec "{sys.executable}" -m scenariolab.llm.fake_claude "$@"\n',
                      encoding="utf-8")
    binary.chmod(0o755)
    _clean_backend_env(monkeypatch)
    monkeypatch.setenv("MSL_CLAUDE_BIN", str(binary))

    def only_the_fake(args, **kwargs):
        assert str(args[0]) == str(binary), f"tests start only the fake CLI, not {args[0]}"
        return REAL_POPEN(args, **kwargs)

    monkeypatch.setattr(cc, "_exec", REAL_EXEC)
    monkeypatch.setattr(cc, "_popen", only_the_fake)
    return binary
