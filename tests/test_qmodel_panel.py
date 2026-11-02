"""The quarterly stack's data layer and contract (SIM_DESIGN §14, P0.5 and P1).

Exit tests of P1: a quarter's L0 field equals the window sum of the monthly factor changes (<= 1e-12); the
start-of-quarter states and __known columns respect release lags (truncation test); V-PRICE y_us_equity = y_mult +
y_fund (<= 1e-10); card <-> quarterly panel <-> QTable schema one-to-one; the policy reaction functions reproduce the
research round's R² (u_ff 0.540 / 0.506 / 0.457). P0.5: the project schema carries the x_oil_cause enum and the CAPE
y_mult, and the blind schema view has no year token and no footprint_quarters.

The card, schema and view tests run anywhere (committed files). The panel tests need the built quarterly panel
(python -m scenariolab.qmodel.panel) and skip without it; the truncation test also needs the raw inputs (data/raw).
Nothing downloads and nothing calls an LLM.
"""
import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.qmodel import card as qcard
from scenariolab.qmodel import config_q as cq
from scenariolab.qmodel import panel as qpanel
from scenariolab.qmodel import sources_q

needs_qpanel = pytest.mark.skipif(not (cq.PANEL_Q_PATH.exists() and cq.PANEL_Q_META_PATH.exists()),
                                  reason="quarterly panel not built (python -m scenariolab.qmodel.panel)")
RAW_NEEDED = ["fred/CLF16OV.json", "fred/TWEXBPA.json", "philfed/bos_dif.csv", "umich/tbmpx1px5.csv",
              "policyuncertainty/Categorical_EPU_Data.xlsx"]
needs_qraw = pytest.mark.skipif(
    not (config.PANEL_PATH.exists() and all((config.RAW_DIR / p).exists() for p in RAW_NEEDED)),
    reason="raw inputs of the quarterly panel not imported (python -m scenariolab.qmodel.sources_q import ...)")


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"tools_{name}", config.ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def card():
    return qcard.load_card()


@pytest.fixture(scope="module")
def schema():
    return qcard.load_schema()


@pytest.fixture(scope="module")
def built():
    return qpanel.load_panel(), qpanel.load_meta()


# ---------------------------------------------------------------- P0.5: the project schema and the blind view

def test_schema_carries_the_oil_cause_enum_and_the_cape_multiple(schema):
    F = {f["id"]: f for f in schema["fields"]}
    assert len(F) == 94 and schema["version"].startswith("schema-revise-2 3.0")
    oc = F["x_oil_cause"]
    assert oc["enum"] == ["supply", "demand", "mixed"] and oc["unit"] != "0/1"
    assert oc["model_value"] == cq.OIL_CAUSE == {"demand": 0.0, "mixed": 0.5, "supply": 1.0}
    assert oc["superseded"]["unit"] == "0/1"
    ym = F["y_mult"]
    assert "CAPE" in ym["zh"] and "us_equity_lm_cape" in ym["definition"] and "P/Ê" in ym["superseded"]["zh"]
    vp = next(v for v in schema["valuation_identities"] if v["id"] == "V-PRICE")
    assert "y_mult + y_fund" in vp["formula"] and "y_eps + y_mult" in vp["superseded"]["formula"]
    assert {c["field"] for c in schema["changes"]} >= {"x_oil_cause", "y_mult", "valuation_identities.V-PRICE"}


def test_blind_schema_view_is_up_to_date_and_blind(schema, card):
    tool = _tool("schema_view")
    view = tool.build_view(schema, card)
    assert tool.render(view) == cq.SCHEMA_VIEW_PATH.read_text(encoding="utf-8")
    text = cq.SCHEMA_VIEW_PATH.read_text(encoding="utf-8")
    assert not re.search(r"(?<![0-9])(19|20)[0-9]{2}(?![0-9])", text)          # no 19xx / 20xx token
    assert "footprint_quarters" not in text and '"measured"' not in text
    assert tool.blind_problems(view) == []
    # the source does have them: the view is what removes them
    raw = cq.SCHEMA_PATH.read_text(encoding="utf-8")
    assert "footprint_quarters" in raw and len(qcard.year_tokens(raw)) > 300
    assert [t["id"] for t in view["templates"]] == [t["id"] for t in schema["templates"]]
    signs = {s for t in view["templates"] for p in t["phases"] for s in p["signs"].values()}
    assert signs <= {"+", "-", "0", "±", "+-", "?", "narrative", "supply", "demand"}
    assert {f["id"] for f in view["fields"] if f["group"] == "x"} == set(qcard.x_fields(card))
    assert tool.scrub("WTI 季末对数变动（1986 年前为月均）") == "WTI 季末对数变动"
    assert tool.blind_problems({"note": "2020Q2 单季"}) and tool.blind_problems({"footprint_quarters": []})


# ---------------------------------------------------------------- the card (SIM_DESIGN Appendix A)

def test_card_is_generated_from_the_schema(schema, card):
    tool = _tool("build_model_card")
    assert tool.render(tool.build_card(schema)) == cq.CARD_PATH.read_text(encoding="utf-8")
    assert qcard.validate_card(card) == []
    assert len(card["rows"]) == 101 and card["schema_version"] == schema["version"]
    assert [c["key"] for c in card["columns"]] == [k for k, _ in qcard.COLUMNS]
    order = [r["group"] for r in card["rows"]]
    assert order == sorted(order, key=["x", "u", "s", "c", "y", "v"].index)           # Appendix A's order


def test_card_rows_follow_the_design_decisions(card):
    r = {x["id"]: x for x in card["rows"]}
    assert {i for i in r if r[i]["l0"]} == set(cq.L0_FIELDS) and r["d_metals_oil"]["route"] == "L0 专用（能源）"
    assert r["d_tp10"]["route"].endswith("只作 TP×1{核心>4}")
    assert r["d_gz"]["fc"] == r["d_spread"]["fc"] == "金融条件（必填其一）"
    assert sum(x["factor_core"] == "是" for x in card["rows"]) == 10
    assert r["y_mult"]["slot"] == "输出（掩码）" and "CAPE" in r["y_mult"]["zh"]
    assert r["s_pe"]["slot"] == r["s_fcfy"]["slot"] == "s_q（只展示）"
    assert r["x_oil_cause"]["enum"] == ["supply", "demand", "mixed"]
    assert all(r[u]["permission"] == "intervention_only" and r[u]["raw_field"] in r for u in cq.SHOCKS)
    stated = qcard.stated_only_fields(card)
    assert len(stated) == 24 and all(r[i]["endo_code"] in ("E1", "E1c", "E2") and r[i]["keywords"] for i in stated)
    assert "GZ" in r["d_gz"]["keywords"] and "信用利差" in r["d_gz"]["keywords"]
    assert {x["permission"] for x in card["rows"] if x["group"] == "s"} == {"data"}
    assert {x["permission"] for x in card["rows"] if x["group"] in ("y", "v")} <= {"never", "read_only"}


def test_card_and_qtable_schema_are_one_to_one(card):
    qs = qcard.qtable_json_schema(card)
    f = qcard.qtable_fields(qs)
    assert f["x"] == set(qcard.x_fields(card)) and len(f["x"]) == 52
    assert f["u"] == set(qcard.shocks(card)) == set(cq.SHOCKS)
    anchors = set(qcard.ids(card, "c"))
    assert f["anchors"] == anchors | set(cq.QTABLE_ANCHOR_STATE)
    assert all(state in qcard.ids(card, "s") for state in cq.QTABLE_ANCHOR_STATE.values())
    rp = qs["properties"]["rows"]["items"]["properties"]
    assert rp["x"]["properties"]["x_oil_cause"]["properties"]["value"]["enum"] == ["supply", "demand", "mixed"]
    assert rp["fc"]["required"] == ["stance"] and rp["fc"]["properties"]["stance"]["enum"] == cq.FC_STANCES
    entry = rp["x"]["properties"]["d_ff"]
    assert entry["required"] == ["src", "quote"] and entry["properties"]["size"]["enum"] == list(cq.SIZE_BINS)
    cqp = qs["properties"]["c_q"]["properties"]
    assert set(cqp["announced"]["items"]["properties"]["field"]["enum"]) == f["x"] - {"x_oil_cause"}
    events = set(qs["properties"]["events"]["items"]["properties"]["channels"]["items"]["enum"])
    assert events == f["x"] | f["u"]
    never = {r["id"] for r in card["rows"] if r["group"] in ("y", "v", "s")}
    assert not never & (f["x"] | f["u"])                         # outputs, views and states are never inputs


def test_prompt_field_section_is_blind_in_test_mode(card):
    test = qcard.prompt_fields(card, "test")
    prod = qcard.prompt_fields(card, "production")
    assert not qcard.year_tokens(test) and qcard.year_tokens(prod)        # notes carry dates in production only
    for fid in qcard.x_fields(card) + qcard.shocks(card) + qcard.ids(card, "c"):
        assert re.search(rf"(?m)^{fid} \|", test), fid
    assert "明显收紧" in test and "温和 = 0.5 sd" in test
    with pytest.raises(ValueError):
        qcard.prompt_fields(card, "oracle")


def test_config_ties_the_quarterly_constants_together(card):
    assert set(cq.WINDOW_SUM_FIELDS) == set(cq.L0_FIELDS) | {"d_metals_oil"}
    assert set(cq.L0_FACTOR) == set(cq.WINDOW_SUM_FIELDS)
    sel = json.loads((config.RESULTS_DIR / "model_selection.json").read_text()) \
        if (config.RESULTS_DIR / "model_selection.json").exists() else None
    if sel:                                  # the factor round's F: the L0 fields feed exactly its factors
        assert set(cq.L0_FACTOR.values()) - {"metals_oil"} == set(sel["spec"]["factors"])
    tp = next(f for f in config.FACTORS if f["id"] == "tp10_x_core4")["interaction"]
    assert tp["above"] == cq.CORE4
    for field, spec in cq.REACTION.items():
        assert field in qcard.x_fields(card) and spec["shock"] in cq.SHOCKS
        assert qcard.row(card, spec["shock"])["raw_field"] == field
    assert cq.SERIES_LAG["fred:CP"] == config.RELEASE_LAGS["CP"] - 2 == 3     # 5 after the first month = 3 after the end
    assert qpanel.series_lag("panel:p_us_equity") == 0 and qpanel.series_lag("french:bm:BusEq") == 0
    with pytest.raises(KeyError):
        qpanel.series_lag("fred:NOT_A_SERIES")


# ---------------------------------------------------------------- helpers (synthetic data, run anywhere)

def test_quarter_helpers_and_known_lags():
    m = pd.Series(np.arange(1.0, 13.0), index=pd.date_range("2001-01-01", periods=12, freq="MS"))
    assert qpanel.q_sum(m).tolist() == [6.0, 15.0, 24.0, 33.0]
    assert np.isnan(qpanel.q_sum(m.drop(m.index[4])).iloc[1])                  # a missing month: no quarter sum
    assert qpanel.q_end(m).tolist() == [3.0, 6.0, 9.0, 12.0]
    s = qpanel.at_start(m, 0)
    assert s[pd.Period("2001Q2", "Q")] == 3.0 and s[pd.Period("2001Q4", "Q")] == 9.0      # the month before q
    assert qpanel.at_start(m, 1)[pd.Period("2001Q2", "Q")] == 2.0                          # lag 1: two months back
    assert [qpanel.quarters_back(k) for k in (0, 1, 2, 3, 4)] == [1, 2, 2, 2, 3]
    q = pd.Series([1.0, 2.0, 3.0], index=pd.period_range("2001Q1", periods=3, freq="Q"))
    assert qpanel.known_quarterly(q, 3)[pd.Period("2001Q3", "Q")] == 1.0
    inputs = qpanel.Inputs(series={"panel:f_ff": m, "fred:GDP": pd.Series(
        [1.0, 2.0, 3.0, 4.0], index=pd.date_range("2001-01-01", periods=4, freq="QS"))},
        freq={"panel:f_ff": "M", "fred:GDP": "Q"})
    cut = qpanel.truncate_inputs(inputs, "2001-09")
    assert cut["panel:f_ff"].index.max() == pd.Timestamp("2001-09-01")
    assert cut["fred:GDP"].index.max() == pd.Timestamp("2001-04-01")    # Q3 ends in 2001-09, published a month later
    a = pd.Series([10.0, 11.0], index=pd.to_datetime(["2005-12-01", "2006-01-01"]))
    b = pd.Series([1.0, 1.2], index=pd.to_datetime(["2006-01-01", "2006-02-01"]))
    ch = qpanel.chain_forward(a, b, "2006-01")
    assert ch.tolist() == [10.0, 11.0, pytest.approx(13.2)]                # the old level is kept


def test_fred_reader_never_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="sources_q import"):
        sources_q.fred("NOT_CACHED")
    ok, why = sources_q.fred_full_request(json.dumps(
        {"observation_start": "1947-01-01", "observation_end": "9999-12-31", "units": "lin", "count": 1,
         "limit": 100000, "offset": 0, "output_type": 1, "realtime_start": "2026-11-01",
         "realtime_end": "2026-11-01", "observations": [{"date": "1973-01-01", "value": "1"}]}).encode())
    assert ok, why
    bad, _ = sources_q.fred_full_request(json.dumps(
        {"observation_start": "2000-01-01", "observation_end": "9999-12-31", "units": "lin", "count": 1,
         "limit": 100000, "observations": [{"date": "1973-01-01", "value": "1"}]}).encode())
    assert not bad


def test_research_import_copies_bytes_checks_hashes_and_never_overwrites(monkeypatch, tmp_path):
    import hashlib
    scratch, raw = tmp_path / "scratch", tmp_path / "raw"
    good = json.dumps({"observation_start": "1600-01-01", "observation_end": "9999-12-31", "units": "lin",
                       "count": 2, "limit": 100000, "offset": 0, "output_type": 1, "realtime_start": "2026-11-01",
                       "realtime_end": "2026-11-01", "observations": [{"date": "1990-01-01", "value": "1.5"},
                                                                     {"date": "1990-04-01", "value": "."}]})
    (scratch / "dl").mkdir(parents=True)
    (scratch / "dl" / "XSERIES.json").write_text(good)
    (scratch / "pub").mkdir()
    (scratch / "pub" / "file.csv").write_bytes(b"a,b\n1,2\n")
    monkeypatch.setattr(sources_q, "FRED_IMPORTS", {"XSERIES": ("dl", "Test publisher", "about 1 month"),
                                                    "MISSING": ("dl", "Test publisher", "0")})
    monkeypatch.setattr(sources_q, "PUBLISHER_IMPORTS", [
        ("pub/file.csv", "pub/file.csv", "sha256", hashlib.sha256(b"a,b\n1,2\n").hexdigest(), {"url": "u"}),
        ("pub/file.csv", "pub/other.csv", "sha1", "0" * 40, {"url": "u"})])
    log = sources_q.run_import(scratch, raw)
    assert log["counts"] == {"copied": 2, "unchanged": 0, "conflict": 0, "rejected": 1, "missing": 1}
    assert (raw / "fred" / "XSERIES.json").read_text() == good
    doc = json.loads((raw / "SOURCES.json").read_text())
    rec = doc["files"]["fred/XSERIES.json"]
    assert rec["first_month"] == "1990-01" and "api_key=REDACTED" in rec["url"] and rec["sha256"]
    assert sources_q.run_import(scratch, raw)["counts"]["unchanged"] == 2
    (raw / "pub" / "file.csv").write_bytes(b"changed")
    assert sources_q.run_import(scratch, raw)["counts"]["conflict"] == 1
    assert (raw / "pub" / "file.csv").read_bytes() == b"changed"                # never replaced without --force
    assert sources_q.run_import(scratch, raw, force=True)["counts"]["copied"] == 1


# ---------------------------------------------------------------- the built quarterly panel

@needs_qpanel
def test_l0_fields_are_the_window_sums_of_the_monthly_factor_changes(built):
    Q, meta = built
    P = pd.read_csv(config.PANEL_PATH, index_col="date", parse_dates=["date"])
    last = pd.Period(meta["info"]["last_data_quarter"], "Q")
    for field, (col, scale) in cq.WINDOW_SUM_FIELDS.items():
        g = P[col].groupby(P.index.to_period("Q"))
        window = g.sum(min_count=3).where(g.count() == 3).reindex(Q.index).loc[:last]
        mine = (Q[field] / scale).loc[:last]
        assert (mine.isna() == window.isna()).all(), field
        assert float((mine - window).abs().max()) <= 1e-12, field
        assert meta["columns"][field]["sources"] == [f"panel:{col}"]
    for a in cq.ASSETS:                                   # the targets: sums of the monthly log price returns
        col = "r_cash" if a == "cash" else f"p_{a}"
        g = np.log1p(P[col]).groupby(P.index.to_period("Q"))
        window = g.sum(min_count=3).where(g.count() == 3).reindex(Q.index).loc[:last]
        assert float((Q[f"y_{a}"].loc[:last] - window).abs().max()) <= 1e-12, a


@needs_qpanel
def test_v_price_identity_on_the_cape_multiple(built):
    Q, _ = built
    gap = (Q["y_us_equity"] - Q["y_mult"] - Q["y_fund"]).dropna()
    assert len(gap) >= 200 and float(gap.abs().max()) <= 1e-10
    assert Q["y_mult"].notna().sum() == len(gap)
    # the multiple carries almost all of the price variance; the fundamentals leg is smooth (SIM_DESIGN §6.1)
    both = Q[["y_us_equity", "y_mult", "y_fund"]].dropna()
    assert both["y_mult"].corr(both["y_us_equity"]) > 0.95 and both["y_fund"].std() < 0.3 * both["y_mult"].std()
    bust = Q.loc["2007Q3":"2009Q2"]
    assert bust["y_us_equity"].sum() == pytest.approx(-0.468, abs=0.01)
    assert bust["y_mult"].sum() == pytest.approx(-0.492, abs=0.01)


@needs_qpanel
def test_spec_history_check_values(built):
    """The numbers SIM_DESIGN takes from the research panels (design d01, d05) come out of this panel."""
    Q, _ = built
    Q = Q.loc[:"2026Q3"]
    m = Q["y_mult"]
    # §4.4: training-period CAPE re-rating, % a year from 1973 to each end
    for end, want in {"1994Q4": 1.18, "2004Q4": 1.80, "2014Q4": 1.47, "2026Q3": 1.89}.items():
        assert 400.0 * m.loc["1973Q2":end].mean() == pytest.approx(want, abs=0.005), end
    # §6.3: re-rating size bins 温和 / 明显 / 极端 = p50 / p75 / p95 of |Σ Δlog M| on 1973-2014
    bins = {4: (0.118, 0.204, 0.335), 8: (0.147, 0.286, 0.492), 12: (0.210, 0.301, 0.526), 20: (0.297, 0.437, 0.618)}
    for span, want in bins.items():
        s = m.loc["1973Q2":"2014Q4"].rolling(span).sum().dropna().abs()
        assert [round(float(s.quantile(p)), 3) for p in (0.5, 0.75, 0.95)] == list(want), span
    # §5.5: the 80-quarter unemployment anchor at the train ends
    u = Q["s_unrate"].shift(-1)
    for end, want in {"1994Q4": 7.05, "2004Q4": 5.73, "2014Q4": 6.01, "2026Q2": 5.77}.items():
        assert u.loc[:end].iloc[-80:].mean() == pytest.approx(want, abs=0.005), end
    # §2 E23 / §11.2 (T18): credit and equity co-movement
    for lo, field, want, n in [("1973Q1", "d_gz", -0.557, 214), ("1985Q1", "d_gz", -0.582, 167),
                               ("1973Q1", "d_spread", -0.401, 215), ("1985Q1", "d_spread", -0.599, 167)]:
        d = Q.loc[lo:, [field, "y_us_equity"]].dropna()
        assert len(d) == n and d.corr().iloc[0, 1] == pytest.approx(want, abs=5e-4), (lo, field)
    # §6.1: the CAPE multiple and P/D legs against the price (n 214 from 1973Q2)
    d = Q.loc["1973Q2":, ["y_us_equity", "y_mult", "y_pd"]].dropna()
    share = lambda c: np.cov(d["y_us_equity"], d[c])[0, 1] / d["y_us_equity"].var()      # noqa: E731
    assert d["y_mult"].corr(d["y_us_equity"]) == pytest.approx(0.990, abs=5e-4)
    assert share("y_mult") == pytest.approx(1.011, abs=5e-4) and share("y_pd") == pytest.approx(0.973, abs=5e-4)
    assert Q["s_unrate"].min() == 3.5                       # §5.4: history's quarterly minimum since 1973


@needs_qpanel
def test_x_fields_reproduce_the_schema_display_sd(built, card, schema):
    """Every x field's 1975Q2-2019Q4 sd equals the schema's sd_1975_2019 (the schema rounds' panels), except the three
    fields whose research versions carried a splice error this port fixes."""
    Q, _ = built
    F = {f["id"]: f for f in schema["fields"]}
    fixed = {"d_epu": "1985Q1 splice of two EPU scales", "d_usd_em": "2006Q1 splice without chaining",
             "d_lf": "January population controls (the schema used the unadjusted labour force)"}
    stats = qpanel.field_stats(Q, qcard.x_fields(card))
    checked = 0
    for f in qcard.x_fields(card):
        want = F[f].get("sd_1975_2019")
        if want is None or f in fixed:
            continue
        assert stats.loc[f, "sd_1975_2019"] == pytest.approx(want, abs=1e-4, rel=1e-6), f
        checked += 1
    assert checked == 47
    for f in fixed:                                      # the fixes lower the sd (the jump is gone)
        assert stats.loc[f, "sd_1975_2019"] < F[f]["sd_1975_2019"]
    assert stats.loc["d_ff", "n"] == 215 and stats.loc["d_ff", "p01"] < 0 < stats.loc["d_ff", "p99"]


@needs_qpanel
def test_core_levels_map_to_the_schema_units(built):
    Q, _ = built
    for field, level in [("d_ff", "lvl_ff"), ("d_y10", "lvl_y10"), ("d_spread", "lvl_spread"), ("d_gz", "lvl_gz"),
                         ("d_usd", "lvl_usd"), ("d_unrate", "lvl_unrate"), ("d_tp10", "lvl_tp10")]:
        assert float((Q[field] - Q[level].diff()).abs().max()) <= 1e-10, field
    # d_oil = Δ(100 log real oil) + 100 Δlog CPI; d_cpi = 4 x Δ(100 log CPI) (SIM_DESIGN §5.1)
    assert float((Q["d_oil"] - Q["lvl_oil_real"].diff() - Q["lvl_cpi"].diff()).abs().max()) <= 1e-10
    assert float((Q["d_cpi"] - 4.0 * Q["lvl_cpi"].diff()).abs().max()) <= 1e-10


@needs_qpanel
def test_card_panel_and_qtable_schema_are_one_to_one(built, card):
    Q, meta = built
    cols = meta["columns"]
    assert list(cols) == list(Q.columns)
    want = qcard.panel_columns(card)                        # card row -> panel columns
    have = qpanel.card_columns(meta)
    assert have == want
    assert {c for c, m in cols.items() if m["role"] in qpanel.CARD_ROLES} == {c for v in want.values() for c in v}
    roles = {"x": "x", "s": "s", "y": "y", "v": "view"}
    for r in card["rows"]:
        for c in r["panel"]:
            assert cols[c]["role"] == roles[r["group"]], c
    xq = set(qcard.qtable_fields(qcard.qtable_json_schema(card))["x"])
    assert {c for c, m in cols.items() if m["role"] == "x"} == xq == set(qcard.x_fields(card))
    for c in xq:                                              # every x field has its release-lagged lags
        assert f"{c}_l1__known" in cols and f"{c}_l2__known" in cols
    assert not set(cq.SHOCKS) & set(Q.columns)                # shocks come from the reaction functions (P3)
    assert meta["info"]["groups"] == {k: v for k, v in want.items() if len(v) > 1}


@needs_qpanel
def test_the_today_row_holds_start_of_quarter_states_only(built):
    Q, meta = built
    info, cols = meta["info"], meta["columns"]
    assert Q.index[0] == pd.Period(cq.FIRST_QUARTER, "Q") and str(Q.index[-1]) == info["today"]
    assert pd.Period(info["today"], "Q") == pd.Period(info["last_data_quarter"], "Q") + 1
    today = Q.iloc[-1]
    realized = [c for c, m in cols.items() if m["role"] in qpanel.REALIZED_ROLES]
    assert today[realized].isna().all()
    if info["today"] == "2026Q4":                   # SIM_DESIGN §6.2: known at the start of 2026Q4
        assert np.exp(today["s_cape"]) == pytest.approx(41.2, abs=0.05)
        assert np.exp(today["s_pd"]) == pytest.approx(97.4, abs=0.05)
        assert today["s_fcfy"] == pytest.approx(1.96, abs=0.01)
        assert info["inputs"]["shiller"]["last_earnings"] == "2026-06"


@needs_qpanel
@needs_qraw
def test_start_of_quarter_columns_respect_release_lags():
    """The truncation test: rebuild the panel from the raw inputs as published by the end of the month before a
    quarter (each series cut at its release lag); every start-of-quarter column of that quarter is unchanged."""
    inputs = qpanel.load_inputs()
    Q, meta = qpanel.build(inputs)
    start_cols = qpanel.start_columns(meta)
    assert len(start_cols) > 150
    for origin in ["1979Q4", "1990Q3", "2001Q1", "2008Q4", "2020Q2", meta["info"]["today"]]:
        q = pd.Period(origin, "Q")
        asof = (q.start_time - pd.DateOffset(months=1)).strftime("%Y-%m")
        Qt, _ = qpanel.build(qpanel.truncate_inputs(inputs, asof))
        assert Qt.index[-1] == q
        a, b = Q.loc[q, start_cols], Qt.loc[q, start_cols]
        same = (a.isna() & b.isna()) | ((a - b).abs() <= 1e-12)
        assert same.all(), (origin, list(same.index[~same])[:10])
        # power: one month less of data changes the market states, so the lags are not padded
        Qe, _ = qpanel.build(qpanel.truncate_inputs(inputs, (q.start_time - pd.DateOffset(months=2)).strftime("%Y-%m")))
        early = Qe.reindex(Q.index).loc[q, ["s_ff", "s_spread", "s_pd"]]
        assert early.isna().all() or (early - Q.loc[q, ["s_ff", "s_spread", "s_pd"]]).abs().max() > 0


@needs_qpanel
def test_known_columns_mask_quarters_released_after_the_quarter_starts(built):
    Q, meta = built
    cols = meta["columns"]
    q = Q.index[-1]
    for c, m in cols.items():
        if m["role"] != "known":
            continue
        field, k = c[:-len("__known")].rsplit("_l", 1)
        lag = cols[field]["lag_months"]
        if 3 * (int(k) - 1) >= lag:
            assert Q.loc[q, c] == Q[field].shift(int(k)).loc[q] or (
                np.isnan(Q.loc[q, c]) and np.isnan(Q[field].shift(int(k)).loc[q])), c
        else:
            assert Q[c].isna().all(), c
    assert cols["d_ff"]["lag_months"] == 0 and cols["d_rgdp"]["lag_months"] == 1
    assert cols["d_profits"]["lag_months"] == 3 and Q["d_profits_l2__known"].notna().sum() > 200


@needs_qpanel
def test_policy_reaction_functions_reproduce_the_research_r2(built):
    Q, _ = built
    for field, spec in cq.REACTION.items():
        for end, want in spec["check_r2"].items():
            r2, n = qpanel.reaction_r2(Q, field, end)
            assert n == spec["check_n"][end], (field, end, n)
            assert abs(r2 - want) <= 5e-4 + 1e-9, (field, end, r2, want)      # the research JSON has 3 decimals
    # the headline numbers of SIM_DESIGN §5.1
    assert [round(qpanel.reaction_r2(Q, "d_ff", e)[0], 2) for e in cq.SPLIT_ENDS] == [0.54, 0.51, 0.46]


@needs_qpanel
@needs_qraw
def test_saved_panel_equals_a_fresh_build():
    Q, meta = qpanel.build_panel()
    saved = qpanel.load_panel()
    stale = "the quarterly panel is older than its inputs: rebuild it (python -m scenariolab.qmodel.panel)"
    assert list(saved.columns) == list(Q.columns) and saved.index.equals(Q.index), stale
    diff = (saved - Q).abs().max().max()
    assert np.isnan(diff) or diff <= 1e-12, stale
    assert (saved.isna() == Q.isna()).all().all(), stale
    assert qpanel.load_meta()["info"]["inputs_digest"] == meta["info"]["inputs_digest"], stale
