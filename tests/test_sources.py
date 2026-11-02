"""data/sources.py and data/import_raw.py without the network (FACTORS_DESIGN §8, tests 1-5).

Every request is answered by a fake requests.get and the FRED key is a fake one, so these run on a fresh
clone; the checks of the imported research files themselves are skipped when data/raw lacks them.
install_synthetic_sources() (also used by test_panel.py) serves a synthetic FRED API and synthetic publisher
data, so the whole panel can be built offline.
"""
import datetime as dt
import hashlib
import io
import json
import logging
import re
import traceback
import zlib

import numpy as np
import pandas as pd
import pytest
import requests

from scenariolab import config, jsonio
from scenariolab.data import import_raw, panel
from scenariolab.data import sources as src

FAKE_KEY = "f0e1d2c3b4a5968778695a4b3c2d1e0f"          # shaped like a FRED key (32 hex digits); never a real one
API_OBS = "https://api.stlouisfed.org/fred/series/observations"
MONTHS = ["2020-01-01", "2020-02-01", "2020-03-01"]
needs_raw = pytest.mark.skipif(not (config.RAW_DIR / "SOURCES.json").exists(),
                               reason="research downloads not imported (python -m scenariolab.data.import_raw)")


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def fred_body(dates, values=None, **extra) -> bytes:
    """A FRED API observations response (a full, unpaged request unless `extra` says otherwise)."""
    values = values if values is not None else [f"{100 + 0.5 * i:.3f}" for i in range(len(dates))]
    obs = [{"realtime_start": "2026-11-01", "realtime_end": "2026-11-01", "date": d, "value": v}
           for d, v in zip(dates, values)]
    body = {"realtime_start": "2026-11-01", "realtime_end": "2026-11-01", "observation_start": "1600-01-01",
            "observation_end": "9999-12-31", "units": "lin", "output_type": 1, "file_type": "json",
            "order_by": "observation_date", "sort_order": "asc", "count": len(obs), "offset": 0, "limit": 100000,
            "observations": obs}
    body.update(extra)
    return json.dumps(body).encode()


class Net:
    """A fake requests.get (answer(url, params) returns a FakeResponse or an exception to raise) and clock."""

    def __init__(self, root):
        self.dir = root
        self.calls = []                 # (url, params, kwargs, clock)
        self.sleeps = []
        self.clock = 1000.0
        self.answer = lambda url, params: requests.ConnectionError("offline")

    def get(self, url, params=None, **kw):
        self.calls.append((url, dict(params or {}), kw, self.clock))
        out = self.answer(url, dict(params or {}))
        if isinstance(out, Exception):
            raise out
        return out

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds


def fake_network(monkeypatch, root) -> Net:
    """sources with RAW_DIR in `root`, the fake key, a fake clock and requests.get answered by net.answer."""
    n = Net(root)
    monkeypatch.setattr(src, "RAW_DIR", root)
    monkeypatch.setattr(src.requests, "get", n.get)
    monkeypatch.setattr(src.time, "sleep", n.sleep)
    monkeypatch.setattr(src, "_now", lambda: n.clock)
    monkeypatch.setattr(src, "_LAST_CALL", [None])
    monkeypatch.setenv("FRED_API_KEY", FAKE_KEY)
    return n


@pytest.fixture
def net(monkeypatch, tmp_path):
    return fake_network(monkeypatch, tmp_path)


# ---------------------------------------------------------------- 1. the API client and its cache names

def test_with_a_key_fred_calls_the_api_and_writes_the_research_cache_names(net):
    net.answer = lambda url, p: FakeResponse(fred_body(MONTHS, ["1.5", ".", "2.25"]))
    s = src.fred("TESTM")
    url, params, kw, _ = net.calls[0]
    assert url == API_OBS and params == {"series_id": "TESTM", "api_key": FAKE_KEY, "file_type": "json"}
    assert kw["allow_redirects"] is False and kw["headers"]["User-Agent"] == src.USER_AGENT
    assert list(s.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01")] and s.iloc[-1] == 2.25
    raw = src.read_fred_json(net.dir / "fred" / "TESTM.json")
    assert np.isnan(raw.loc["2020-02-01"]) and raw.loc["2020-01-01"] == 1.5         # "." parses to NaN
    src.fred("TESTM", agg="eop")
    src.fred("TESTM", agg="avg")
    assert [(c[1]["frequency"], c[1]["aggregation_method"]) for c in net.calls[1:]] == [("m", "eop"), ("m", "avg")]
    assert {p.name for p in (net.dir / "fred").iterdir()} == {"TESTM.json", "TESTM_m_eop.json", "TESTM_m_avg.json"}
    src.fred("TESTM", eop=True)                       # eop=True is agg="eop": read from the cache
    src.fred("TESTM")
    assert len(net.calls) == 3
    with pytest.raises(ValueError):
        src.fred("TESTM", agg="sum")


def test_monthly_dates_are_month_starts_and_weekly_series_keep_their_dates(net):
    weekly = ["2020-01-03", "2020-01-10", "2020-01-31", "2020-02-07"]
    net.answer = lambda url, p: FakeResponse(fred_body(weekly if p["series_id"] == "WK" else ["2020-01-01", "2020-04-01"]))
    w, q = src.fred("WK"), src.fred("QQ")
    assert list(w.index.strftime("%Y-%m-%d")) == weekly
    assert panel.last_in_month(w).to_dict() == {pd.Timestamp("2020-01-01"): w.iloc[2], pd.Timestamp("2020-02-01"): w.iloc[3]}
    assert panel.mean_in_month(w).loc["2020-01-01"] == pytest.approx(w.iloc[:3].mean())
    assert list(q.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")]


# ---------------------------------------------------------------- 2. graph CSV without a key; JSON beats CSV

def test_without_a_key_fred_uses_the_graph_csv(net, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY")
    net.answer = lambda url, p: FakeResponse(f"observation_date,{p['id']}\n2020-01-01,1.0\n2020-02-01,\n2020-03-01,3.0\n".encode())
    s = src.fred("TESTC")
    url, params, _, _ = net.calls[0]
    assert url == src.FRED_URL and params == {"id": "TESTC"} and (net.dir / "fred" / "TESTC.csv").exists()
    assert s.to_dict() == {pd.Timestamp("2020-01-01"): 1.0, pd.Timestamp("2020-03-01"): 3.0}
    src.fred("TESTC", eop=True)
    assert net.calls[1][1] == {"id": "TESTC", "fq": "Monthly", "fam": "eop"} and (net.dir / "fred" / "TESTC_eop.csv").exists()
    assert not any("api_key" in c[1] or c[0].startswith(src.FRED_API) for c in net.calls)
    assert src.fred_meta("TESTC") == {}               # no metadata without a key (provenance only)


def test_a_cached_json_is_read_offline_and_beats_the_csv(net, monkeypatch):
    fdir = net.dir / "fred"
    fdir.mkdir()
    (fdir / "BOTH.json").write_bytes(fred_body(["2020-01-01"], ["7.0"]))
    (fdir / "BOTH.csv").write_bytes(b"observation_date,BOTH\n2020-01-01,1.0\n")
    (fdir / "ONLYCSV.csv").write_bytes(b"observation_date,ONLYCSV\n2020-01-01,2.0\n")
    assert src.fred("BOTH").iloc[0] == 7.0 and src.fred("ONLYCSV").iloc[0] == 2.0
    monkeypatch.delenv("FRED_API_KEY")
    assert src.fred("BOTH").iloc[0] == 7.0 and src.fred("ONLYCSV").iloc[0] == 2.0
    assert net.calls == []


def test_a_keyless_refresh_reads_the_new_csv_and_moves_the_older_api_copy_aside(net, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY")
    fdir = net.dir / "fred"
    fdir.mkdir()
    (fdir / "OLD.json").write_bytes(fred_body(["2020-01-01"], ["7.0"]))
    net.answer = lambda url, p: FakeResponse(b"observation_date,OLD\n2020-01-01,8.0\n")
    assert src.fred("OLD", refresh=True).iloc[0] == 8.0
    assert src.fred("OLD").iloc[0] == 8.0 and (fdir / "superseded" / "OLD.json").exists()


def test_an_html_page_is_not_cached_as_a_graph_csv(net, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY")
    net.answer = lambda url, p: FakeResponse(b"<!doctype html><html>moved</html>")
    with pytest.raises(RuntimeError, match="download failed"):
        src.fred("MOVED")
    assert not (net.dir / "fred" / "MOVED.csv").exists()


# ---------------------------------------------------------------- 3. key hygiene

def _api_calls(net) -> list:
    return [c for c in net.calls if c[0].startswith(src.FRED_API)]


def _keyless(net) -> bool:
    """Every request that did not go to the API carried no key (the graph CSV fallback)."""
    return all("api_key" not in c[1] and FAKE_KEY not in c[0] for c in net.calls if not c[0].startswith(src.FRED_API))


def test_a_failing_request_never_shows_the_key(net, caplog):
    leaked = f"{API_OBS}?series_id=BAD&api_key={FAKE_KEY}&file_type=json"
    net.answer = lambda url, p: requests.ConnectionError(
        f"HTTPSConnectionPool(host='api.stlouisfed.org', port=443): Max retries exceeded with url: {leaked}")
    caplog.set_level(logging.DEBUG)
    with pytest.raises(RuntimeError) as exc:
        src.fred("BAD")
    err = exc.value
    shown = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    assert FAKE_KEY not in shown and "api_key=" not in str(err)
    assert "BAD" in str(err) and "ConnectionError" in str(err) and err.__cause__ is None and err.__context__ is None
    assert len(_api_calls(net)) == config.FRED["attempts"] and net.sleeps           # retried with backoff
    assert len(net.calls) > len(_api_calls(net)) and _keyless(net)                 # then the keyless graph CSV
    assert FAKE_KEY not in caplog.text and "api_key=REDACTED" in caplog.text
    assert not (net.dir / "fred").exists() or not any((net.dir / "fred").iterdir())


def test_other_http_errors_fail_at_once_and_odd_bodies_are_failures(net):
    net.answer = lambda url, p: FakeResponse(b'{"error_code": 400, "error_message": "Bad Request."}', 400)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        src.fred("NOPE")
    assert len(_api_calls(net)) == 1
    net.answer = lambda url, p: FakeResponse(b'{"seriess": []}')
    with pytest.raises(RuntimeError, match="UnexpectedBody"):
        src.fred("ODD")
    assert len(_api_calls(net)) == 1 + config.FRED["attempts"] and _keyless(net)
    net.answer = lambda url, p: FakeResponse(fred_body(MONTHS) + FAKE_KEY.encode())       # a body echoing the key
    with pytest.raises(RuntimeError):
        src.fred("ECHO")
    assert not (net.dir / "fred" / "ECHO.json").exists() and not (net.dir / "fred" / "ECHO.csv").exists()


def test_a_rejected_key_or_an_api_outage_falls_back_to_the_graph_csv(net, caplog):
    """With FRED_API_KEY set, a key FRED rejects (HTTP 400) or an API outage does not stop the build: the keyless
    graph CSV is tried; the warning names the series and the status only."""
    caplog.set_level(logging.DEBUG)

    def answer(url, p):
        if url.startswith(src.FRED_API):
            return FakeResponse(b'{"error_code": 400, "error_message": "Bad Request.  The value for variable api_key '
                                b'is not registered."}', 400) if p["series_id"] == "REJ" else requests.ConnectionError("down")
        return FakeResponse(f"observation_date,{p['id']}\n2020-01-01,4.0\n2020-02-01,5.0\n".encode())
    net.answer = answer
    assert src.fred("REJ").to_dict() == {pd.Timestamp("2020-01-01"): 4.0, pd.Timestamp("2020-02-01"): 5.0}
    assert len(_api_calls(net)) == 1                                   # a 400 is not retried
    assert src.fred("OUT", eop=True).iloc[-1] == 5.0
    assert len(_api_calls(net)) == 1 + config.FRED["attempts"]         # an outage is retried, then the CSV
    assert {p.name for p in (net.dir / "fred").iterdir()} == {"REJ.csv", "OUT_eop.csv"} and _keyless(net)
    assert [c[1] for c in net.calls if c[0] == src.FRED_URL] == [{"id": "REJ"}, {"id": "OUT", "fq": "Monthly", "fam": "eop"}]
    assert "series_id=REJ&api_key=REDACTED" in caplog.text and "HTTP 400" in caplog.text
    assert "FRED API failed for REJ; trying the keyless graph CSV" in caplog.text and FAKE_KEY not in caplog.text
    assert src.fred("REJ").iloc[0] == 4.0 and len(_api_calls(net)) == 1 + config.FRED["attempts"]   # now cached


def test_the_key_goes_to_api_stlouisfed_org_only(net, monkeypatch):
    monkeypatch.setattr(src, "FRED_API", "https://example.org/fred")
    with pytest.raises(RuntimeError, match="api.stlouisfed.org"):
        src.fred("X")
    assert not any(c[0].startswith("https://example.org") for c in net.calls) and _keyless(net)
    assert src.redacted_url("/series/observations", {"series_id": "X", "api_key": FAKE_KEY}) == (
        "https://example.org/fred/series/observations?series_id=X&api_key=REDACTED&file_type=json")


def test_fred_meta_is_cached_with_a_copyright_flag(net):
    meta = {"seriess": [{"id": "WBAA", "title": "Moody's Seasoned Baa", "units": "Percent", "frequency": "Weekly",
                         "notes": "© 2017, Moody's Corporation. All rights reserved."}]}
    net.answer = lambda url, p: FakeResponse(json.dumps(meta).encode())
    out = src.fred_meta("WBAA")
    assert net.calls[0][0] == "https://api.stlouisfed.org/fred/series" and out["copyright"] is True
    assert (net.dir / "fred" / "WBAA_meta.json").exists() and src.fred_meta("WBAA")["title"] == "Moody's Seasoned Baa"
    assert len(net.calls) == 1


def test_no_file_of_a_fake_build_holds_the_key(net, monkeypatch, tmp_path_factory, caplog):
    caplog.set_level(logging.DEBUG)
    install_synthetic_sources(monkeypatch, net)
    out = net.dir / "processed"
    monkeypatch.setattr(config, "PROCESSED_DIR", out)
    monkeypatch.setattr(config, "PANEL_PATH", out / "panel_monthly.csv")
    monkeypatch.setattr(config, "PANEL_META_PATH", out / "panel_meta.json")
    panel.save_panel()
    meta = panel.load_meta()
    jsonio.dump({"sources": panel.meta_sources(meta), "notes": meta["notes"]}, net.dir / "results" / "meta.json")
    research = make_research(tmp_path_factory.mktemp("factors"), key=FAKE_KEY)     # a record with a raw key
    import_raw.run(research / "raw", dest_root=net.dir)
    files = [p for p in net.dir.rglob("*") if p.is_file()]
    assert any(p.name.endswith(".json") and "fred" in p.parts for p in files)        # the API cache was written
    assert [p for p in files if FAKE_KEY in p.name or FAKE_KEY.encode() in p.read_bytes()] == []
    assert FAKE_KEY not in caplog.text
    text = (net.dir / "SOURCES.json").read_text(encoding="utf-8")
    assert set(re.findall(r"api_key=[^&\"\s]*", text)) == {"api_key=REDACTED"}
    assert all(url.startswith("http") and "api_key" not in url for url in meta["sources"].values())


# ---------------------------------------------------------------- 4. politeness

def test_api_calls_are_spaced_and_a_429_is_retried_once(net):
    first = iter([FakeResponse(b"", 429)])

    def answer(url, p):
        if p["series_id"] == "A":
            return next(first, None) or FakeResponse(fred_body(MONTHS))
        return FakeResponse(fred_body(MONTHS))
    net.answer = answer
    for sid in ("A", "B", "C"):
        src.fred(sid)
    times = [c[3] for c in net.calls]
    assert [c[1]["series_id"] for c in net.calls] == ["A", "A", "B", "C"]
    assert all(b - a >= config.FRED["min_interval_s"] - 1e-9 for a, b in zip(times, times[1:]))
    assert 1.5 in net.sleeps                                                     # backoff 1.5 s after the 429


# ---------------------------------------------------------------- 5. publisher parsers and content checks

EBP_CSV = "date,gz_spread,ebp,est_prob\n1973-01-01,1.13,-0.05,0.18\n1973-02-01,1.08,-0.06,0.18\n"
FMHPI_CSV = ("Year,Month,GEO_Type,GEO_Name,GEO_Code,Index_NSA,Index_SA\n1975,1,State,AK,.,34.8,35.2\n"
             "1975,1,US,USA,.,24.23,24.19\n1975,2,US,USA,.,24.37,24.33\n")
HTML = b"<!doctype html><html><body>This page has moved</body></html>"


def test_content_checks_reject_html_pages():
    assert src.is_ebp_csv(EBP_CSV.encode()) and src.is_fmhpi_csv(FMHPI_CSV.encode())
    assert src.is_xls(src.XLS_MAGIC + b"\x00") and src.is_zip(src.ZIP_MAGIC + b"\x00")
    assert not any(check(HTML) for check in (src.is_ebp_csv, src.is_fmhpi_csv, src.is_xls, src.is_zip))


def test_ebp_acm_fmhpi_gpr_and_fig31_parsers():
    ebp = src.parse_ebp_csv(EBP_CSV)
    assert list(ebp.columns) == ["gz_spread", "ebp"] and ebp.loc["1973-02-01", "gz_spread"] == 1.08
    rows = [["DATE", "ACMY10", "ACMTP10"], ["30-Jun-1961", 3.9, 0.51], ["31-Jul-1961", 3.95, 0.55], ["", "", ""]]
    acm = src.parse_acm_rows(rows)
    assert acm.to_dict() == {pd.Timestamp("1961-06-01"): 0.51, pd.Timestamp("1961-07-01"): 0.55}
    fm = src.parse_fmhpi_csv(io.StringIO(FMHPI_CSV))
    assert fm.to_dict() == {pd.Timestamp("1975-01-01"): 24.19, pd.Timestamp("1975-02-01"): 24.33}
    gpr = src.parse_gpr_rows([["month", "GPR", "GPRH"], [dt.datetime(1900, 1, 1), "", 87.9],
                              [dt.datetime(1900, 2, 1), "", 64.7]])
    assert gpr.to_dict() == {pd.Timestamp("1900-01-01"): 87.9, pd.Timestamp("1900-02-01"): 64.7}
    fig = [[""] * 9 for _ in range(7)] + [[1952.0] * 7 + [1952.0, 9.0], [0.0] * 7 + [1953.0 + 0.5 / 12, 10.0],
                                          [0.0] * 7 + [1953.0 + 11.5 / 12, 11.0]]
    assert src.parse_fig31_rows(fig).to_dict() == {pd.Timestamp("1953-01-01"): 10.0, pd.Timestamp("1953-12-01"): 11.0}
    with pytest.raises(ValueError):
        src.parse_acm_rows([["DATE", "ACMY10"], ["30-Jun-1961", 3.9]])


def pink_xlsx(path, oil_col=1, metal_col=15):
    """A small Pink Sheet workbook: labels where the World Bank puts them, columns movable."""
    wb = pytest.importorskip("openpyxl").Workbook()
    ws = wb.active
    ws.title = "Monthly Prices"
    for line in ("World Bank Commodity Price Data", "monthly prices", "(monthly series)", "Updated on October 02, 2026"):
        ws.append([line])
    head = [None] * 6
    head[oil_col], head[oil_col + 1] = "Crude oil, average", "Crude oil, Brent"
    ws.append(head)
    ws.append([None, "($/bbl)"])
    for m, v in (("1960M01", 2.0), ("1960M02", "…"), ("1960M03", 2.5)):
        row = [m] + [1.0] * 5
        row[oil_col] = v
        ws.append(row)
    wi = wb.create_sheet("Monthly Indices")
    for _ in range(5):
        wi.append([None])
    labels = [[None] * 18 for _ in range(4)]
    labels[0][1], labels[1][14], labels[2][metal_col] = "Total Index", "Metals  & Minerals", "Base Metals (ex. iron ore)"
    for r in labels:
        wi.append(r)
    for m, v in (("1960M01", 10.0), ("1960M02", 11.0), ("1960M03", 12.0)):
        row = [m] + [1.0] * 17
        row[metal_col] = v
        wi.append(row)
    wb.save(path)


def test_pink_sheet_columns_are_found_by_their_labels(net):
    path = net.dir / "worldbank" / src.PINK_FILE
    path.parent.mkdir()
    pink_xlsx(path, oil_col=2, metal_col=12)
    out = src.worldbank_pinksheet()
    assert out.loc["1960-01-01"].tolist() == [2.0, 10.0] and np.isnan(out.loc["1960-02-01", "crude_avg"])
    assert net.calls == []                                                     # cached: nothing downloaded


def test_pink_sheet_link_is_discovered_on_worldbank_org_only(net, tmp_path):
    book = tmp_path / "book.xlsx"
    pink_xlsx(book)
    page = b'<a href="https://evil.example.com/CMO-Historical-Data-Monthly.xlsx">x</a>'
    net.answer = lambda url, p: FakeResponse(page)
    with pytest.raises(RuntimeError, match="必需的数据源不可用"):
        src.worldbank_pinksheet()
    good = b'<a href="https://thedocs.worldbank.org/en/doc/abc/related/CMO-Historical-Data-Monthly.xlsx">Monthly</a>'
    net.answer = lambda url, p: FakeResponse(good if url == src.PINK_PAGE_URL else book.read_bytes())
    assert src.worldbank_pinksheet().loc["1960-03-01", "base_metals"] == 12.0
    assert net.calls[-1][0].startswith("https://thedocs.worldbank.org/")


def test_epu_and_hlw_workbooks(net):
    openpyxl = pytest.importorskip("openpyxl")
    pu = net.dir / "policyuncertainty"
    pu.mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Main News Index"
    for row in (["Year", "Month", "News_Based_Policy_Uncert_Index"], [1900, 1, 139.9], [1900, 2, 120.1],
                ["Source: 'Measuring Economic Policy Uncertainty'", None, None]):
        ws.append(row)
    wb.save(pu / "US_Policy_Uncertainty_Data.xlsx")
    assert src.epu_news().to_dict() == {pd.Timestamp("1900-01-01"): 139.9, pd.Timestamp("1900-02-01"): 120.1}
    ny = net.dir / "nyfed"
    ny.mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "HLW Estimates"
    for row in (["This spreadsheet"], ["by Holston, Laubach and Williams"], ["Final data point: 1961Q2"], [None],
                [None, None, "Trend Growth (g)", None, None, None, "Natural Rate (r*)", None],
                ["Date", None, "US", "Canada", "Euro Area", None, "Canada", "US"],
                [dt.datetime(1961, 1, 1), None, 4.9, 3.2, "NA", None, 3.5, 5.48],
                [dt.datetime(1961, 4, 1), None, 5.1, 4.9, "NA", None, 5.6, 5.70]):
        ws.append(row)
    wb.save(ny / "Holston_Laubach_Williams_current_estimates.xlsx")
    assert src.nyfed_hlw().to_dict() == {pd.Timestamp("1961-01-01"): 5.48, pd.Timestamp("1961-04-01"): 5.70}


def test_a_required_source_without_a_cache_raises_and_an_optional_one_returns_none(net):
    net.answer = lambda url, p: FakeResponse(HTML)                            # a moved file answers with a page
    with pytest.raises(RuntimeError, match="必需的数据源不可用"):
        src.fed_ebp()
    with pytest.raises(RuntimeError, match="必需的数据源不可用"):
        src.nyfed_acm()
    assert src.freddiemac_fmhpi() is None and src.iacoviello_gpr() is None and src.shiller_home() is None
    assert src.epu_news() is None and src.nyfed_hlw() is None
    assert not [p for p in net.dir.rglob("*") if p.is_file()]                 # nothing was cached
    net.answer = lambda url, p: FakeResponse(EBP_CSV.encode())
    assert src.fed_ebp().index[0] == pd.Timestamp("1973-01-01")
    net.answer = lambda url, p: requests.ConnectionError("offline")
    assert src.fed_ebp(refresh=True).index[0] == pd.Timestamp("1973-01-01")  # a failed refresh keeps the cache


def test_splice_without_a_primary_keeps_the_fallback():
    fb = pd.Series([1.0, 2.0], index=pd.to_datetime(["2000-01-01", "2000-02-01"]))
    assert src.splice(pd.Series(dtype=float), fb).tolist() == [1.0, 2.0]
    assert src.splice_level(pd.Series(dtype=float), fb).tolist() == [1.0, 2.0]


# ---------------------------------------------------------------- the import of the research downloads

def make_research(root, key=None):
    """A small research folder: raw/ with FRED responses and a publisher file, risk/sources.json with
    provenance (one URL carrying `key` unredacted, as a careless record might)."""
    raw = root / "raw"
    (raw / "fred").mkdir(parents=True)
    (raw / "fred" / "XQ.json").write_bytes(fred_body(MONTHS))
    (raw / "fred" / "XQ_obs.json").write_bytes(fred_body(MONTHS))                # the same request
    (raw / "fred" / "PAGED.json").write_bytes(fred_body(MONTHS, count=5000, limit=3))
    (raw / "fred" / "XQ_m_eop.json").write_bytes(fred_body(MONTHS))
    (raw / "fred" / "XQ_meta.json").write_text(json.dumps({"seriess": [{"id": "XQ", "frequency": "Monthly",
                                                                         "notes": "Copyright, 2016, Freddie Mac."}]}))
    (raw / "fred" / "XQ_rt20200101.json").write_bytes(fred_body(MONTHS, realtime_start="2020-01-01"))
    (raw / "fred" / "SEARCH_xq.json").write_text('{"seriess": []}')
    ebp = raw / "federalreserve" / "ebp_20261101" / "ebp_csv.csv"
    ebp.parent.mkdir(parents=True)
    ebp.write_text(EBP_CSV)
    (raw / "cboe" / "vix_20261101").mkdir(parents=True)
    (raw / "cboe" / "vix_20261101" / "VIX_History.csv").write_text("DATE,CLOSE\n")
    (root / "risk").mkdir()
    url = f"{API_OBS}?series_id=XQ&api_key={key or 'REDACTED'}&file_type=json"
    (root / "risk" / "sources.json").write_text(json.dumps({
        "family": "risk", "access_date": "2026-11-01",
        "files": [{"local_path": str(ebp), "sha256": hashlib.sha256(ebp.read_bytes()).hexdigest(),
                   "url": src.EBP_URL, "publisher": "Federal Reserve Board", "first_month": "1973-01",
                   "revised": "yes", "terms": "free with citation"}],
        "fred": [{"fred_id": "XQ", "url_observations": url, "publisher": "Freddie Mac via FRED",
                  "revised": "no", "cached_files": [str(raw / "fred" / "XQ_obs.json")]}]}))
    return root


def test_import_copies_bytes_routes_fred_files_and_records_redacted_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", FAKE_KEY)
    research = make_research(tmp_path / "factors", key=FAKE_KEY)
    dest = tmp_path / "data_raw"
    log = import_raw.run(research / "raw", dest_root=dest)
    rf = research / "raw" / "fred"
    assert (dest / "fred" / "XQ.json").read_bytes() == (rf / "XQ.json").read_bytes()
    assert (dest / "fred" / "XQ_m_eop.json").exists() and (dest / "fred" / "XQ_meta.json").exists()
    assert (dest / "fred" / "alfred" / "PAGED.json").exists() and not (dest / "fred" / "PAGED.json").exists()
    assert (dest / "fred" / "alfred" / "XQ_rt20200101.json").exists() and (dest / "fred" / "searches" / "SEARCH_xq.json").exists()
    assert (dest / "federalreserve" / "ebp_csv.csv").read_text() == EBP_CSV and not (dest / "cboe").exists()
    skipped = {r["source"]: r["reason"] for r in log["files"] if r["action"] == "skipped"}
    assert "same request" in skipped["fred/XQ_obs.json"] and "cboe/vix_20261101/VIX_History.csv" in skipped
    files = jsonio.load(dest / "SOURCES.json")["files"]
    assert files["fred/XQ.json"]["url"] == f"{API_OBS}?series_id=XQ&api_key=REDACTED&file_type=json"
    assert files["fred/XQ_m_eop.json"]["url"].endswith("frequency=m&aggregation_method=eop&api_key=REDACTED&file_type=json")
    assert files["fred/XQ.json"]["publisher"] == "Freddie Mac via FRED" and files["fred/XQ.json"]["first_month"] == "2020-01"
    assert files["fred/XQ.json"]["copyright"] == "Copyright, 2016, Freddie Mac."
    ebp = files["federalreserve/ebp_csv.csv"]
    assert ebp["sha256"] == hashlib.sha256(EBP_CSV.encode()).hexdigest() and ebp["access_date"] == "2026-11-01"
    assert {k for k in import_raw.FIELDS} <= set(ebp) and ebp["families"] == ["risk"]
    for name in ("SOURCES.json", "import_log.json"):
        text = (dest / name).read_text(encoding="utf-8")
        assert FAKE_KEY not in text and set(re.findall(r"api_key=[^&\"\s]*", text)) <= {"api_key=REDACTED"}
    again = import_raw.run(research / "raw", dest_root=dest)
    assert again["counts"]["copied"] == 0 and again["counts"]["unchanged"] == log["counts"]["copied"]


def test_import_refuses_a_different_file_without_force_and_a_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(import_raw, "load_dotenv", lambda *a, **k: None)      # never the real .env in a test
    research = make_research(tmp_path / "factors")
    dest = tmp_path / "data_raw"
    (dest / "fred").mkdir(parents=True)
    (dest / "fred" / "XQ.json").write_bytes(fred_body(MONTHS, ["9", "9", "9"]))
    log = import_raw.run(research / "raw", dest_root=dest)
    assert {r["dest"]: r["action"] for r in log["files"] if r.get("dest")}["fred/XQ.json"] == "conflict"
    assert import_raw.main([str(research / "raw"), "--dest", str(dest)]) == 1
    forced = import_raw.run(research / "raw", dest_root=dest, force=True)
    assert [r for r in forced["files"] if r.get("dest") == "fred/XQ.json"][0]["replaced"] is True
    (research / "raw" / "federalreserve" / "ebp_20261101" / "ebp_csv.csv").write_text(EBP_CSV + "1973-03-01,1,0,0\n")
    bad = import_raw.run(research / "raw", dest_root=tmp_path / "other")
    assert [r["action"] for r in bad["files"] if r.get("dest") == "federalreserve/ebp_csv.csv"] == ["rejected"]
    assert not (tmp_path / "other" / "federalreserve" / "ebp_csv.csv").exists()


def test_admissible_wants_a_full_unpaged_untransformed_request():
    assert import_raw.admissible(fred_body(MONTHS)) == (True, "")
    for extra in ({"observation_start": "2000-01-01"}, {"units": "pch"}, {"count": 99}, {"limit": 2},
                  {"offset": 3}, {"output_type": 4}, {"realtime_end": "9999-12-31"}):
        assert import_raw.admissible(fred_body(MONTHS, **extra))[0] is False, extra
    assert import_raw.admissible(b"<html>")[0] is False


# ---------------------------------------------------------------- the imported research files (real data)

@pytest.fixture
def offline(monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("no download in a test")
    monkeypatch.setattr(src.requests, "get", no_network)


@needs_raw
def test_the_imported_publisher_files_parse(offline):
    ebp = src.fed_ebp()
    assert ebp.index[0] == pd.Timestamp("1973-01-01") and ebp["gz_spread"].loc["2026-09-01"] == pytest.approx(0.92591, abs=1e-5)
    acm = src.nyfed_acm()
    assert acm.index[0] == pd.Timestamp("1961-06-01") and acm.index[-1] >= pd.Timestamp("2026-09-01")
    pink = src.worldbank_pinksheet()
    assert pink.index[0] == pd.Timestamp("1960-01-01") and pink.loc["1960-01-01":"2026-09-01"].notna().all().all()
    assert src.freddiemac_fmhpi().index[0] == pd.Timestamp("1975-01-01")
    assert src.shiller_home().index[0] == pd.Timestamp("1953-01-01")
    assert src.iacoviello_gpr().index[0] == pd.Timestamp("1900-01-01")
    assert src.epu_news().index[0] == pd.Timestamp("1900-01-01")
    hlw = src.nyfed_hlw()
    assert set(hlw.index.month) == {1, 4, 7, 10} and hlw.index[0] == pd.Timestamp("1961-01-01")


@needs_raw
def test_imported_json_and_the_projects_csv_agree_except_recent_revisions():
    fdir = config.RAW_DIR / "fred"
    pairs = [(c, fdir / (c.stem[:-4] + "_m_eop.json" if c.stem.endswith("_eop") else c.stem + ".json"))
             for c in sorted(fdir.glob("*.csv"))]
    pairs = [(c, j) for c, j in pairs if j.exists()]
    assert len(pairs) >= 15
    for c, j in pairs:
        a, b = src.read_fred_csv(c).dropna(), src.read_fred_json(j).dropna()
        common = a.index.intersection(b.index)[:-6]
        assert len(common) > 100 and np.allclose(a[common], b[common], rtol=0, atol=1e-9), c.name


@needs_raw
def test_sources_json_records_every_imported_file_with_redacted_urls():
    doc = jsonio.load(config.RAW_DIR / "SOURCES.json")
    files = doc["files"]
    assert files["fred/WBAA.json"]["copyright"] and "Moody" in files["fred/WBAA.json"]["copyright"]
    assert "Freddie Mac" in (files["freddiemac/fmhpi_master_file.csv"]["copyright"] or "")
    assert files["nyfed/ACMTermPremium.xls"]["sha256"] == hashlib.sha256(
        (config.RAW_DIR / "nyfed" / "ACMTermPremium.xls").read_bytes()).hexdigest()
    text = (config.RAW_DIR / "SOURCES.json").read_text(encoding="utf-8")
    assert set(re.findall(r"api_key=[^&\"\s]*", text)) == {"api_key=REDACTED"}


# ---------------------------------------------------------------- synthetic sources (also used by test_panel.py)

QUARTERLY = {"PRS85006173", "CP", "GDP", "PNFI", "TNWBSHNO", "DPI", "OPHNFB", "ULCNFB", "A679RC1Q027SBEA",
             "B235RC1Q027SBEA", "A255RC1Q027SBEA", "FGDEF", "FYGFGDQ188S", "GDPC1", "PCECC96"}
WEEKLY = {"WBAA", "WGS10YR", "MORTGAGE30US", "ICSA"}
SYN_END = "2026-09-01"


def _synthetic(name: str, index: pd.DatetimeIndex, level: float = 100.0, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return pd.Series(level * np.exp(np.cumsum(rng.normal(0.002, vol, len(index)))), index=index, name=name)


def synthetic_fred_answer(url, params):
    """A FRED API response for any series: monthly, quarterly or weekly (by id), deterministic values."""
    sid = params["series_id"]
    if url.endswith("/series"):
        return FakeResponse(json.dumps({"seriess": [{"id": sid, "notes": ""}]}).encode())
    if sid in WEEKLY and "frequency" not in params:
        idx = pd.date_range("1960-01-01", "2026-10-30", freq="W-FRI")
    elif sid in QUARTERLY:
        idx = pd.date_range("1947-01-01", "2026-04-01", freq="QS")
    else:
        idx = pd.date_range("1947-01-01", SYN_END, freq="MS")
    s = _synthetic(sid, idx, level=5.0 if sid in WEEKLY else 100.0)
    return FakeResponse(fred_body(list(idx.strftime("%Y-%m-%d")), [f"{v:.6f}" for v in s]))


def install_synthetic_sources(monkeypatch, net, fmhpi: bool = False):
    """Serve every panel input offline: the FRED API through `net` (with the fake key) and the other
    publishers as synthetic frames. FMHPI is unavailable unless fmhpi=True (an optional source's note)."""
    net.answer = synthetic_fred_answer
    m = pd.date_range("1926-07-01", SYN_END, freq="MS")
    inds = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]

    def french_table(name, title, refresh=False):
        rng = np.random.default_rng(zlib.crc32(title.encode()))     # the same draw with and without dividends
        if title.startswith("Average Value Weighted Returns"):
            base = rng.normal(1.0, 4.0, (len(m), len(inds)))
            return pd.DataFrame(base - (0.3 if "Wout_Div" in name else 0.0), index=m, columns=inds)
        return pd.DataFrame(rng.uniform(50, 500, (len(m), len(inds))), index=m, columns=inds)

    def french_factors(refresh=False):
        rng = np.random.default_rng(1)
        return pd.DataFrame({"Mkt-RF": rng.normal(0.007, 0.045, len(m)), "SMB": 0.0, "HML": 0.0, "RF": 0.003}, index=m)

    def months(start, end=SYN_END):
        return pd.date_range(start, end, freq="MS")
    r = months("1972-01-01")
    monkeypatch.setattr(src, "french_table", french_table)
    monkeypatch.setattr(src, "french_factors", french_factors)
    monkeypatch.setattr(src, "nareit_equity_reits", lambda refresh=False: pd.DataFrame(
        {"total": 0.01, "price": 0.007, "income": 0.003, "dividend_yield": 0.04}, index=r))
    monkeypatch.setattr(src, "gold", lambda refresh=False: _synthetic("gold", months("1950-01-01"), 35.0, 0.03))
    monkeypatch.setattr(src, "shiller", lambda refresh=False, download=True: None)
    e = months("1973-01-01")
    monkeypatch.setattr(src, "fed_ebp", lambda refresh=False: pd.DataFrame(
        {"gz_spread": _synthetic("gz", e, 2.0, 0.05), "ebp": _synthetic("ebp", e, 1.0, 0.05) - 1.0}, index=e))
    monkeypatch.setattr(src, "nyfed_acm", lambda refresh=False: _synthetic("acm", months("1961-06-01"), 1.0, 0.05))
    w = months("1960-01-01")
    monkeypatch.setattr(src, "worldbank_pinksheet", lambda refresh=False: pd.DataFrame(
        {"crude_avg": _synthetic("crude", w, 2.0, 0.05), "base_metals": _synthetic("metals", w, 10.0, 0.04)}, index=w))
    monkeypatch.setattr(src, "freddiemac_fmhpi", lambda refresh=False: (
        _synthetic("fmhpi", months("1975-01-01", "2026-08-01"), 24.0, 0.005) if fmhpi else None))
    monkeypatch.setattr(src, "shiller_home", lambda refresh=False: _synthetic("home", months("1953-01-01", "2023-01-01"), 3.0))
    monkeypatch.setattr(src, "iacoviello_gpr", lambda refresh=False: _synthetic("gpr", months("1900-01-01"), 90.0, 0.1))
    monkeypatch.setattr(src, "epu_news", lambda refresh=False: _synthetic("epu", months("1900-01-01"), 100.0, 0.1))
    monkeypatch.setattr(src, "nyfed_hlw", lambda refresh=False: _synthetic(
        "hlw", pd.date_range("1961-01-01", "2026-04-01", freq="QS"), 3.0, 0.02))


def test_import_brings_in_the_shiller_workbook_and_replaces_an_older_copy_only_with_force(tmp_path, monkeypatch):
    """P0.3: the research round's Shiller 2026 workbook goes to the path sources.shiller reads."""
    monkeypatch.setattr(import_raw, "load_dotenv", lambda *a, **k: None)
    research = make_research(tmp_path / "factors")
    book = research / "raw" / "shiller" / "ie_data_shillerdata_20261101" / "ie_data.xls"
    book.parent.mkdir(parents=True)
    book.write_bytes(src.XLS_MAGIC + b"2026 workbook")
    dest = tmp_path / "data_raw"
    (dest / "shiller").mkdir(parents=True)
    (dest / "shiller" / "ie_data.xls").write_bytes(src.XLS_MAGIC + b"2023 copy")
    log = import_raw.run(research / "raw", dest_root=dest)
    row = next(r for r in log["files"] if r.get("dest") == "shiller/ie_data.xls")
    assert row["action"] == "conflict" and (dest / "shiller" / "ie_data.xls").read_bytes().endswith(b"2023 copy")
    forced = import_raw.run(research / "raw", dest_root=dest, force=True)
    row = next(r for r in forced["files"] if r.get("dest") == "shiller/ie_data.xls")
    assert row["action"] == "copied" and row["replaced"] is True
    assert (dest / "shiller" / "ie_data.xls").read_bytes() == book.read_bytes()
    rec = jsonio.load(dest / "SOURCES.json")["files"]["shiller/ie_data.xls"]
    assert rec["research_path"] == "shiller/ie_data_shillerdata_20261101/ie_data.xls" and rec["kind"] == "publisher"
