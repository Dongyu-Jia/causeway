"""The build CLI and the data sources it starts from, without data, the engine or the network.

Every step of build.main() is faked, so these run on a fresh clone; the Shiller download is a mocked
requests.get (nothing is ever downloaded here).
"""
import pytest

from scenariolab import build
from scenariolab.data import panel
from scenariolab.data import sources as src


# ---------------------------------------------------------------- build.main(): skipped vs failed steps

@pytest.fixture
def fake_build(monkeypatch, tmp_path):
    """build.main() with the panel present, a dummy context and every step module faked.

    Returns {module: exception}: a step whose module is listed raises that exception.
    """
    def no_panel(refresh=False):
        raise AssertionError("the panel must not be built (or downloaded) in a test")

    cached = tmp_path / "panel_monthly.csv"
    cached.write_text("date\n", encoding="utf-8")
    monkeypatch.setattr(build, "save_panel", no_panel)
    monkeypatch.setattr(build.config, "PANEL_PATH", cached)
    monkeypatch.setattr(build.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(build.EngineContext, "load", classmethod(lambda cls, **kw: object()))
    raises: dict = {}

    def run(module, ctx, **kw):
        if module in raises:
            raise raises[module]
        return {"ok": True}

    monkeypatch.setattr(build, "_run", run)
    return raises


def test_missing_optional_anthropic_package_skips_the_step(fake_build, capsys):
    fake_build["llm.translator"] = ModuleNotFoundError("No module named 'anthropic'", name="anthropic")
    assert build.main(["--only", "llm", "eras"]) == 0
    out = capsys.readouterr().out
    assert "[build] llm skipped" in out and "FAILED" not in out and "[build] eras done" in out


def test_missing_anthropic_submodule_also_skips(fake_build, capsys):
    fake_build["llm.translator"] = ModuleNotFoundError("No module named 'anthropic.types'", name="anthropic.types")
    assert build.main(["--only", "llm"]) == 0
    assert "skipped" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["scipy", "sklearn.linear_model", "anthropicx", None])
def test_any_other_missing_module_fails_the_build(fake_build, capsys, name):
    fake_build["analysis.eras"] = ModuleNotFoundError(f"No module named {name!r}", name=name)
    assert build.main(["--only", "eras", "causality"]) == 1
    out = capsys.readouterr().out
    assert "[build] eras FAILED" in out and "failed: ['eras']" in out
    assert "[build] causality done" in out                     # the build goes on after a failure


def test_an_ordinary_error_fails_the_build(fake_build, capsys):
    fake_build["analysis.eras"] = ValueError("bad data")
    assert build.main(["--only", "eras"]) == 1
    assert "eras FAILED" in capsys.readouterr().out


@pytest.fixture
def step_calls(fake_build, monkeypatch):
    """The step modules build.main() runs, in order (meta, which is built in place, is faked too)."""
    calls: list[str] = []
    inner = build._run

    def run(module, ctx, **kw):
        calls.append(module)
        return inner(module, ctx, **kw)

    monkeypatch.setattr(build, "_run", run)
    monkeypatch.setattr(build, "build_meta", lambda ctx: {})
    return calls


def test_regimes_runs_after_scenarios_and_before_forecast_and_two_steps_are_opt_in():
    s = build.STEPS
    assert s.index("scenarios") < s.index("regimes") < s.index("forecast") < s.index("llm")
    assert build.OPT_IN == {"ablation", "regime_forecast"}
    assert s[-2:] == ["ablation", "regime_forecast"]


def test_the_default_build_runs_regimes_but_not_the_opt_in_steps(step_calls):
    assert build.main(["--skip", "panel"]) == 0
    assert "analysis.regimes" in step_calls
    assert step_calls.index("scenarios.library") < step_calls.index("analysis.regimes") < step_calls.index("analysis.forecast")
    assert "analysis.ablation" not in step_calls and "analysis.regime_forecast" not in step_calls


def test_all_includes_both_opt_in_steps(step_calls, tmp_path):
    assert build.main(["--all", "--skip", "panel"]) == 0
    assert step_calls[-2:] == ["analysis.ablation", "analysis.regime_forecast"]
    assert {"analysis.regimes", "analysis.forecast"} <= set(step_calls)
    assert (tmp_path / "regimes.json").exists() and (tmp_path / "regime_forecast.json").exists()


def test_only_regime_forecast_runs_that_step_alone(step_calls):
    assert build.main(["--only", "regime_forecast"]) == 0
    assert step_calls == ["analysis.regime_forecast"]


def test_rebuilding_scenarios_alone_reminds_to_rebuild_regimes(step_calls, tmp_path, capsys):
    assert build.main(["--only", "scenarios"]) == 0
    assert "--only regimes" not in capsys.readouterr().out          # nothing to be stale yet
    (tmp_path / "regimes.json").write_text("{}", encoding="utf-8")
    assert build.main(["--only", "scenarios"]) == 0
    assert "--only regimes" in capsys.readouterr().out
    assert build.main(["--only", "scenarios", "regimes"]) == 0
    assert "--only regimes" not in capsys.readouterr().out


def test_optional_missing_names_only_the_llm_extra():
    assert build.optional_missing(ModuleNotFoundError(name="anthropic"))
    assert build.optional_missing(ModuleNotFoundError(name="anthropic.lib.credentials"))
    assert not build.optional_missing(ModuleNotFoundError(name="anthropic_tools"))
    assert not build.optional_missing(ModuleNotFoundError(name="numpy"))
    assert not build.optional_missing(ModuleNotFoundError("no name"))


# ---------------------------------------------------------------- meta.json sources

def test_meta_sources_list_shiller_and_name_gold_properly():
    old = {"sources": {
        "FRED": "https://fred.stlouisfed.org",
        "Gold (datasets/gold-prices)": "https://github.com/datasets/gold-prices",   # an older panel build
        "Some new source": "https://example.org/data",
    }}
    out = panel.meta_sources(old)
    assert "Gold (datasets/gold-prices)" not in out
    assert list(out)[:len(panel.SOURCES)] == list(panel.SOURCES) and out["Some new source"] == "https://example.org/data"
    assert src.SHILLER_URL in out.values() and "https://github.com/datasets/gold-prices" in out.values()
    assert any("ie_data.xls" in name and "CAPE" in name for name in out)
    assert panel.meta_sources({}) == panel.SOURCES and panel.meta_sources({"sources": None}) == panel.SOURCES
    assert all(url.startswith("http") for url in panel.SOURCES.values())


def test_user_agent_names_the_repository():
    assert src.USER_AGENT.endswith("(+https://github.com/Dongyu-Jia/causeway)")


# ---------------------------------------------------------------- Shiller: downloaded when missing, never re-downloaded

class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


@pytest.fixture
def fake_net(monkeypatch, tmp_path):
    """sources with RAW_DIR in tmp_path, no sleeping, and requests.get recorded; set .body or .error to answer."""
    class Net:
        body = src.XLS_MAGIC + b"\x00" * 64       # looks like an .xls workbook (it does not parse: shiller() -> None)
        error = None
        calls: list = []

        def get(self, url, **kw):
            self.calls.append((url, kw))
            if self.error is not None:
                raise self.error
            return FakeResponse(self.body)

    net = Net()
    net.calls = []
    monkeypatch.setattr(src, "RAW_DIR", tmp_path)
    monkeypatch.setattr(src.time, "sleep", lambda s: None)
    monkeypatch.setattr(src.requests, "get", net.get)
    net.dest = tmp_path / "shiller" / "ie_data.xls"
    return net


def test_shiller_is_downloaded_when_the_cache_is_missing(fake_net):
    assert src.shiller() is None                    # the fake body is not a real workbook: no CAPE, no failure
    assert len(fake_net.calls) == 1 and fake_net.calls[0][0] == src.SHILLER_URL
    assert fake_net.calls[0][1]["headers"]["User-Agent"] == src.USER_AGENT
    assert fake_net.dest.read_bytes() == fake_net.body          # cached for the next build


def test_a_cached_shiller_file_is_never_downloaded_again_unless_refresh(fake_net):
    fake_net.dest.parent.mkdir(parents=True)
    fake_net.dest.write_bytes(src.XLS_MAGIC + b"cached")
    src.shiller()
    src.shiller(refresh=False)
    assert fake_net.calls == []
    assert fake_net.dest.read_bytes() == src.XLS_MAGIC + b"cached"
    src.shiller(refresh=True)
    assert len(fake_net.calls) == 1 and fake_net.dest.read_bytes() == fake_net.body


def test_download_false_reads_the_cache_only(fake_net):
    assert src.shiller(download=False) is None
    assert src.shiller(refresh=True, download=False) is None
    assert fake_net.calls == [] and not fake_net.dest.exists()


def test_a_body_that_is_not_a_workbook_is_not_cached(fake_net):
    fake_net.body = b"<!doctype html><html>moved</html>"
    assert src.shiller() is None
    assert len(fake_net.calls) == src.SHILLER_ATTEMPTS and not fake_net.dest.exists()
    assert not list(fake_net.dest.parent.glob("*.part"))


def test_no_network_returns_none_and_caches_nothing(fake_net):
    fake_net.error = src.requests.ConnectionError("offline")
    assert src.shiller() is None
    assert len(fake_net.calls) == src.SHILLER_ATTEMPTS and not fake_net.dest.exists()
    assert all(kw["timeout"] == src.SHILLER_TIMEOUT for _, kw in fake_net.calls)


def test_the_required_sources_still_retry_four_times(fake_net):
    fake_net.error = src.requests.Timeout("slow")
    with pytest.raises(RuntimeError, match="download failed"):
        src.gold()
    assert len(fake_net.calls) == 4 and all(kw["timeout"] == 90 for _, kw in fake_net.calls)
