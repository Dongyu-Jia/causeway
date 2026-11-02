"""data/panel.py after the missing-factor round (FACTORS_DESIGN §8, tests 6-11): the credit-spread timing fix,
quarterly series dated at their release, no look-ahead, the high-core-inflation term premium, coverage and
column_info, and the white-collar Törnqvist index.

The helpers and an offline build from synthetic sources run anywhere; the real-data checks need the built panel
(needs_panel) and the imported research downloads (needs_raw), and never touch the network.
"""
import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.data import panel
from scenariolab.data import sources as src
from scenariolab.engine import factor_model
from test_sources import fake_network, install_synthetic_sources

needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
needs_raw = pytest.mark.skipif(not (config.RAW_DIR / "SOURCES.json").exists(),
                               reason="research downloads not imported (python -m scenariolab.data.import_raw)")
ALIGNMENTS = {"reference_month", "release_dated", "month_end", "monthly_average", "last_week", "mixed"}
INFO_KEYS = {"sources", "alignment", "lag_months", "first", "last", "late", "full_history", "revised", "copyright",
             "role"}
MODEL_INPUTS = ["f_gz", "f_tp10", "f_tp10_x_core4", "f_metals_oil"]
AUDIT_COLUMNS = ["f_ebp", "f_tp10_x_infl4", "x_spread_mixed", "x_labor_share_interp"]
VIEW_COLUMNS = [f"s_{s}" for s in config.VIEW_STATE_IDS]
NEW_COLUMNS = MODEL_INPUTS + AUDIT_COLUMNS + VIEW_COLUMNS


def ms(*dates) -> list:
    return [pd.Timestamp(d) for d in dates]


@pytest.fixture
def offline(monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("no download in a test")
    monkeypatch.setattr(src.requests, "get", no_network)


# ---------------------------------------------------------------- configuration ties

def test_config_ties_the_thresholds_and_lags_together():
    tp = next(f for f in config.FACTORS if f["id"] == "tp10_x_core4")["interaction"]
    assert tp == {"factor": "tp10", "state": "core_yoy", "lag": 1, "above": 4.0}
    assert tp["above"] == factor_model.HIGH_INFLATION == config.REGIMES["params"]["high_inflation_pct"]
    assert config.VALUATION["release_lag_months"] == config.RELEASE_LAGS["CP"]
    assert config.AUDIT_FACTOR_IDS == {"ebp", "tp10_x_infl4"}
    assert not any(f["core"] for f in config.FACTORS if f["id"] in {"gz", "tp10", "tp10_x_core4", "metals_oil"})
    assert next(f for f in config.FACTORS if f["id"] == "metals_oil")["dedicated"] == ["energy"]
    assert config.FRED == {"api_url": "https://api.stlouisfed.org/fred", "key_env": "FRED_API_KEY",
                           "min_interval_s": 0.6, "attempts": 4}


def test_every_view_variable_has_a_concept_a_source_and_column_info():
    views = [s for s in config.STATES if s.get("concept")]
    assert len(views) == 30 and {s["id"] for s in views if s.get("late")} == {"tips10", "mich1y"}
    assert all(s["concept"] in config.VIEW_CONCEPTS and s["source"] and s["unit"] for s in views)
    assert set(config.VIEW_CONCEPTS) == {s["concept"] for s in views}
    assert all(c in panel.COLUMN_INFO for c in NEW_COLUMNS)
    assert all(panel.COLUMN_INFO[c][1] in ALIGNMENTS for c in panel.COLUMN_INFO)
    assert all(panel.COLUMN_INFO[c][3] in config.RELEASE_LAGS for c in panel.COLUMN_INFO
               if panel.COLUMN_INFO[c][1] == "release_dated")


# ---------------------------------------------------------------- 7. release dating (synthetic)

def test_a_quarter_appears_at_its_release_month_and_holds_until_the_next_one():
    q = pd.Series([1.0, 2.0, 3.0], index=ms("2000-01-01", "2000-04-01", "2000-07-01"))
    idx = pd.date_range("1999-10-01", "2001-06-01", freq="MS")
    out = panel.release_dated(q, idx, lag_months=4, hold_months=3)
    assert out.loc[:"2000-04-01"].isna().all()                              # absent before the first release
    assert out.loc["2000-05-01":"2000-07-01"].tolist() == [1.0] * 3          # first visible at quarter start + lag
    assert out.loc["2000-08-01":"2000-10-01"].tolist() == [2.0] * 3          # constant until the next release
    assert out.loc["2000-11-01":"2001-01-01"].tolist() == [3.0] * 3
    assert out.loc["2001-02-01":].isna().all()                              # an overdue release is not carried on
    assert panel.release_dated(q, idx, 4).loc["2001-06-01"] == 3.0           # (the uncapped hold of the old helper)
    ls = np.log(out).diff()
    assert set(ls[ls.abs() > 0].index.month) == {8, 11}                      # changes only in release months


def test_quarterly_yoy_compares_the_same_quarter_a_year_earlier():
    q = pd.Series(np.arange(1.0, 10.0), index=pd.date_range("2000-01-01", periods=9, freq="QS"))
    y = panel._q_yoy(q)
    assert y.iloc[:4].isna().all() and y.loc["2001-01-01"] == pytest.approx(400.0) and y.loc["2002-01-01"] == pytest.approx(80.0)


# ---------------------------------------------------------------- 9. the high-core-inflation term premium (synthetic)

def test_high_inflation_term_uses_the_previous_months_state():
    idx = pd.date_range("2000-01-01", periods=5, freq="MS")
    change = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)
    state = pd.Series([np.nan, 5.0, 4.0, 4.1, 3.0], index=idx)
    out = panel.high_inflation_term(change, state, lag=1, above=4.0)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])                  # NaN where last month's state is unknown
    assert out.iloc[2:].tolist() == [0.3, 0.0, 0.5]                          # strictly above 4 the month before


# ---------------------------------------------------------------- 11. white-collar Törnqvist index (synthetic)

def test_tornqvist_index_equals_a_brute_force_chain():
    rng = np.random.default_rng(7)
    idx = pd.date_range("1963-10-01", periods=30, freq="MS")
    ahe = pd.DataFrame(np.exp(np.cumsum(rng.normal(0.003, 0.01, (30, 3)), axis=0)) * [20, 25, 30], index=idx,
                       columns=list("abc"))
    emp = pd.DataFrame(rng.uniform(500, 3000, (30, 3)), index=idx, columns=list("abc"))
    ahe.iloc[:3, 2] = np.nan                                                 # the third industry starts later
    out = panel.tornqvist(np.log(ahe).diff(), ahe * emp)
    brute, level = [], np.nan
    for t in range(30):
        share = (ahe.iloc[t] * emp.iloc[t]) / (ahe.iloc[t] * emp.iloc[t]).sum()
        if ahe.iloc[t].isna().any():
            brute.append(np.nan)
            continue
        if np.isnan(level):
            level = 1.0
        else:
            prev = (ahe.iloc[t - 1] * emp.iloc[t - 1]) / (ahe.iloc[t - 1] * emp.iloc[t - 1]).sum()
            w = 0.5 * (share + prev)
            level *= float(np.prod((ahe.iloc[t] / ahe.iloc[t - 1]) ** w))
        brute.append(level)
    np.testing.assert_allclose(out.to_numpy(), np.array(brute), rtol=1e-12)
    assert out.iloc[3] == 1.0 and out.iloc[:3].isna().all()
    ahe.iloc[10, 0] = np.nan                                                 # a gap after the start ends the index
    assert panel.tornqvist(np.log(ahe).diff(), ahe * emp).iloc[10:].isna().all()


def test_monthly_helpers_and_the_house_price_splice():
    w = pd.Series([1.0, 2.0, 3.0, 4.0], index=ms("2020-01-03", "2020-01-31", "2020-02-07", "2020-02-28"))
    assert panel.last_in_month(w).tolist() == [2.0, 4.0] and panel.mean_in_month(w).tolist() == [1.5, 3.5]
    a = pd.Series(1.0, index=pd.date_range("1950-01-01", "1990-12-01", freq="MS"))
    b = pd.Series(2.0, index=pd.date_range("1950-01-01", "1990-12-01", freq="MS"))
    seg = panel.segments((a, None), (b, "1962-01"), (a * 3, "1986-01"))
    assert seg.loc["1961-12-01"] == 1.0 and seg.loc["1962-01-01"] == 2.0 and seg.loc["1986-01-01"] == 3.0
    rng = np.random.default_rng(3)
    shiller = pd.Series(np.exp(np.cumsum(rng.normal(0.004, 0.01, 852))), index=pd.date_range("1953-01-01", periods=852, freq="MS"))
    fm = pd.Series(np.exp(np.cumsum(rng.normal(0.005, 0.005, 24))) * 24.0, index=pd.date_range("1975-01-01", periods=24, freq="MS"))
    h = panel.house_price_index(fm, shiller)
    assert h.index[0] == pd.Timestamp("1972-01-01") and h.loc["1975-01-01":].equals(fm)
    assert h.loc["1974-12-01"] == pytest.approx(fm.iloc[0])                  # the 1975-01 change is 0 (documented)
    dl = np.log(shiller).diff().loc["1953-02":"1974-12"]
    sa = dl - dl.index.month.map(dl.groupby(dl.index.month).mean() - dl.mean()).to_numpy()
    assert np.log(h).diff().loc["1972-02":"1974-12"].to_numpy() == pytest.approx(sa.loc["1972-02":].to_numpy())
    assert panel.house_price_index(fm, None).equals(fm)


# ---------------------------------------------------------------- an offline build from synthetic sources

@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    net = fake_network(monkeypatch, tmp_path)
    install_synthetic_sources(monkeypatch, net)
    return panel.build_panel()


def test_the_panel_builds_offline_with_every_new_column_described(synthetic):
    P, meta = synthetic
    info = meta["column_info"]
    assert set(NEW_COLUMNS) <= set(P.columns) and "lvl_y10s" in P and "lvl_metals_oil" in P
    assert set(info) == {c for c in P.columns if c.startswith(("s_", "f_", "x_"))}
    assert all(set(v) == INFO_KEYS and v["alignment"] in ALIGNMENTS for v in info.values())
    assert {c: info[c]["role"] for c in ("f_gz", "f_ebp", "x_spread_mixed", "s_gz", "s_cpi_yoy", "f_cpi")} == {
        "f_gz": "model", "f_ebp": "audit", "x_spread_mixed": "audit", "s_gz": "view", "s_cpi_yoy": "state",
        "f_cpi": "state"}
    assert info["s_labor_share"]["lag_months"] == 4 and info["s_wealth_income"]["lag_months"] == 6
    assert info["s_tariff_rate"]["lag_months"] == 3 and info["s_rstar"]["lag_months"] == 5
    assert info["s_cpi_yoy"]["lag_months"] is None and info["s_spread"]["copyright"] == "Moody's"
    moves = P["f_labor_share"].loc["1950":].pipe(lambda s: s[s.abs() > 0])
    assert set(moves.index.month) == {2, 5, 8, 11}
    assert meta["factor_start"]["gz"] == "1973-02-01" and meta["release_lags"] == config.RELEASE_LAGS
    assert P["s_hpi_yoy"].isna().all()                                       # FMHPI unavailable here
    assert "Freddie Mac 房价指数（FMHPI）不可用，房价同比为空。" in meta["notes"]
    assert P.loc["1973-02":, [f"f_{f['id']}" for f in config.FACTORS if not f.get("optional")]].notna().all().all()


# ---------------------------------------------------------------- 6. spread timing (real data)

@needs_panel
@needs_raw
def test_spread_legs_share_the_time_of_month(offline):
    P = panel.load_panel()
    dbaa, dgs10 = src.fred("DBAA", eop=True), src.fred("DGS10", eop=True)
    wbaa, wy10 = panel.last_in_month(src.fred("WBAA")), panel.last_in_month(src.fred("WGS10YR"))
    late = P.loc["1986-01":, "s_spread"]
    np.testing.assert_allclose(late, (dbaa - dgs10).reindex(late.index), atol=1e-12)
    mid = P.loc["1962-01":"1985-12", "s_spread"]
    np.testing.assert_allclose(mid, (wbaa - wy10).reindex(mid.index), atol=1e-12)
    old = src.splice_level(dbaa, src.fred("BAA")).reindex(P.index) - P["lvl_y10"]
    np.testing.assert_allclose(P.loc["1953-04":, "x_spread_mixed"], old.loc["1953-04":], atol=1e-12)
    np.testing.assert_allclose(P["f_spread"].loc["1953-05":], P["s_spread"].diff().loc["1953-05":], atol=1e-12)
    np.testing.assert_allclose(P.loc["1986-01":, "x_spread_mixed"], late, atol=1e-12)     # the same from 1986
    assert P.loc["1973":"1985", ["s_spread", "x_spread_mixed"]].corr().iloc[0, 1] == pytest.approx(0.969, abs=0.001)


# ---------------------------------------------------------------- 7. release dating (real data)

@needs_panel
def test_the_release_dated_labor_share_moves_only_in_release_months():
    P = panel.load_panel()
    moves = P["f_labor_share"].loc["1948":].pipe(lambda s: s[s.abs() > 0])
    assert set(moves.index.month) == {2, 5, 8, 11}
    assert np.allclose(P["s_profit_share"].dropna(), P["s_profit_share_rt"].dropna())
    for col in ("f_profit_share", "f_capex_share", "f_wealth"):
        nonzero = P[col].loc["1953":].pipe(lambda s: s[s.abs() > 0])
        assert set(nonzero.index.month) <= {3, 6, 9, 12, 1, 4, 7, 10}, col          # lag 5 or 6: one month a quarter
        assert nonzero.groupby(nonzero.index.year).size().max() <= 4, col


# ---------------------------------------------------------------- 8. no look-ahead (real data, property)

SOURCE_FUNCS = ("fred", "french_table", "french_factors", "nareit_equity_reits", "gold", "shiller", "fed_ebp",
                "nyfed_acm", "worldbank_pinksheet", "freddiemac_fmhpi", "shiller_home", "iacoviello_gpr",
                "epu_news", "nyfed_hlw")
# A quarterly input is cut at its publication month (quarter start + its release lag), so a quarter the panel
# showed before that month would fail the property. GDP's advance estimate (4 months after the quarter starts)
# enters the IT-investment share at that share's lag 4; its own entry (5) is the NIPA lag of the profit ratios.
CUT_LAGS = {**config.RELEASE_LAGS, "GDP": config.RELEASE_LAGS["A679RC1Q027SBEA"], "nyfed_hlw": config.RELEASE_LAGS["HLW"]}


def _cut(obj, end: pd.Timestamp, lag: int | None = None):
    """Everything known only after the month before `end`: a monthly value from its month, a quarter from its
    publication month (quarter start + `lag`), or, without a lag, from its last month (it is complete then)."""
    if not isinstance(obj, (pd.Series, pd.DataFrame)) or not isinstance(obj.index, pd.DatetimeIndex) or not len(obj):
        return obj
    idx = obj.index
    quarterly = len(idx) > 8 and set(idx.month) <= {1, 4, 7, 10} and (idx.day == 1).all() and \
        np.median(np.diff(idx.year * 12 + idx.month)) == 3
    known = idx + pd.DateOffset(months=2 if lag is None else lag) if quarterly else idx
    return obj[known < end]


def test_the_cut_dates_a_quarter_at_its_publication():
    q = pd.Series(np.arange(12.0), index=pd.date_range("2000-01-01", periods=12, freq="QS"))
    end = pd.Timestamp("2001-06-01")                                         # truncated at the end of 2001-05
    assert _cut(q, end).index.max() == pd.Timestamp("2001-01-01")            # complete by 2001-03
    assert _cut(q, end, lag=4).index.max() == pd.Timestamp("2001-01-01")     # published 2001-05
    assert _cut(q, end, lag=5).index.max() == pd.Timestamp("2000-10-01")     # 2001-Q1 published only in 2001-06


@needs_raw
def test_truncating_every_input_leaves_earlier_months_unchanged(monkeypatch, offline):
    full, _ = panel.build_panel()
    cut, memo = [None], {}
    for name in SOURCE_FUNCS:
        def wrapped(*a, _fn=getattr(src, name), _name=name, **k):
            key = (_name, a, tuple(sorted(k.items())))
            if key not in memo:
                memo[key] = _fn(*a, **k)
            lag = CUT_LAGS.get(a[0] if _name == "fred" and a else _name)
            return memo[key] if cut[0] is None else _cut(memo[key], cut[0], lag)
        monkeypatch.setattr(src, name, wrapped)
    cols = [c for c in full.columns if c.startswith(("s_", "f_")) and c != "s_real_oil"]
    draws = np.random.default_rng(20261101).choice(full.loc["1973-02":].index, 20, replace=False)
    for m in sorted(pd.Timestamp(d) for d in draws):
        cut[0] = m + pd.DateOffset(months=1)
        P, _ = panel.build_panel()
        assert P.index.max() <= m and P.index.max() >= m - pd.DateOffset(months=2)
        # the 1972-74 house-price splice uses seasonal factors fitted on 1953-74, as any seasonally adjusted series
        skip = {"s_hpi_yoy"} if m < pd.Timestamp("1975-01-01") else set()
        for c in cols:
            if c not in skip:
                np.testing.assert_allclose(P[c], full.loc[P.index, c], rtol=1e-9, atol=1e-12, err_msg=f"{c}, cut {m:%Y-%m}")
        ratio = (P["s_real_oil"] / full.loc[P.index, "s_real_oil"]).dropna()      # deflated to the last month's CPI
        assert ratio.max() / ratio.min() - 1.0 < 1e-9


# ---------------------------------------------------------------- 9. the term-premium interaction (real data)

@needs_panel
def test_the_term_premium_term_follows_last_months_core_cpi():
    P = panel.load_panel()
    prev = P["s_core_yoy"].shift(1)
    expect = (P["f_tp10"] * (prev > 4.0).astype(float)).where(prev.notna())
    np.testing.assert_array_equal(P["f_tp10_x_core4"].to_numpy(), expect.to_numpy())
    db = P.loc["1973-02":]
    on = (prev > 4.0).loc["1973-02":]
    assert int(on.sum()) == 231 and int((db["f_tp10_x_core4"] != 0).sum()) == 231
    assert on[on].index.max() == pd.Timestamp("2023-12-01") and not on.iloc[-1]
    expect4 = (P["f_tp10"] * (P["s_cpi_yoy"].shift(1) > 4.0).astype(float)).where(P["s_cpi_yoy"].shift(1).notna())
    np.testing.assert_array_equal(P["f_tp10_x_infl4"].to_numpy(), expect4.to_numpy())


# ---------------------------------------------------------------- 10. coverage and column_info (real data)

@needs_panel
def test_model_inputs_cover_every_monthdb_month_and_column_info_is_complete():
    P, meta = panel.load_panel(), panel.load_meta()
    info = meta["column_info"]
    win = P.loc["1973-02":]
    inputs = [f"f_{f['id']}" for f in config.FACTORS if not f.get("optional")] + ["f_cpi"]
    assert {c: int(win[c].isna().sum()) for c in inputs if win[c].isna().any()} == {}
    assert all(info[c]["full_history"] for c in inputs)
    assert set(info) == {c for c in P.columns if c.startswith(("s_", "f_", "x_"))}
    assert all(set(info[c]) == INFO_KEYS and info[c]["alignment"] in ALIGNMENTS for c in NEW_COLUMNS)
    late = {c for c in VIEW_COLUMNS if info[c]["late"]}
    assert late == {"s_tips10", "s_mich1y"} and (info["s_tips10"]["first"], info["s_mich1y"]["first"]) == ("2003-01", "1978-01")
    early = [c for c in VIEW_COLUMNS if c not in late and c != "s_usd_em_yoy"]
    assert all(info[c]["first"] <= "1973-02" for c in early) and info["s_usd_em_yoy"]["first"] == "1974-01"
    assert {c: info[c]["role"] for c in MODEL_INPUTS} == dict.fromkeys(MODEL_INPUTS, "model")
    assert {c: info[c]["role"] for c in AUDIT_COLUMNS} == dict.fromkeys(AUDIT_COLUMNS, "audit")
    assert info["s_mich1y"]["copyright"] == "University of Michigan"
    assert info["s_hpi_yoy"]["copyright"] == "Freddie Mac; Robert J. Shiller"
    assert all("api_key" not in url for url in meta["sources"].values())


@needs_panel
def test_the_monthdb_keeps_every_month_from_1973_02():
    from scenariolab.context import load_panel_with_shocks
    from scenariolab.engine.factor_model import AssetFactorModel, ModelSpec
    from scenariolab.engine.simulate import MonthDB
    P = load_panel_with_shocks()
    months = pd.date_range("1973-02-01", P.index.max(), freq="MS")
    core = [f for f in config.CORE_FACTOR_IDS if f != "labor_share"]
    for spec in (ModelSpec(), ModelSpec.from_dict({**ModelSpec().to_dict(), "factors": core + ["gz", "tp10_x_core4"]})):
        db = MonthDB(P, AssetFactorModel(spec).fit(P, config.MODEL_START, None))
        assert db.dates.equals(months)
    if P.index.max() == pd.Timestamp("2026-09-01"):
        assert len(months) == 644


# ---------------------------------------------------------------- P0.3: s_cape is Shiller's own CAPE

def _shiller_frame(cape_end: str = "2026-10-01", e_end: str = "2026-06-01") -> pd.DataFrame:
    idx = pd.date_range("1871-01-01", cape_end, freq="MS")
    rng = np.random.default_rng(3)
    out = pd.DataFrame({"price": np.exp(np.cumsum(rng.normal(0.003, 0.04, len(idx)))),
                        "earnings": np.exp(np.cumsum(rng.normal(0.002, 0.01, len(idx)))),
                        "cpi": np.exp(np.cumsum(rng.normal(0.002, 0.003, len(idx)))),
                        "cape": 15.0 + np.cumsum(rng.normal(0.0, 0.3, len(idx)))}, index=idx)
    out.loc[out.index > pd.Timestamp(e_end), ["earnings"]] = np.nan
    return out


def test_s_cape_is_shillers_column_with_no_extension():
    sh = _shiller_frame(cape_end="2020-06-01", e_end="2020-03-01")
    idx = pd.date_range("1947-01-01", "2020-12-01", freq="MS")
    cape = panel.shiller_cape(sh, idx)
    assert cape.index.equals(idx)
    assert np.array_equal(cape.loc[:"2020-06"].to_numpy(), sh["cape"].loc["1947-01":"2020-06"].to_numpy())
    assert cape.loc["2020-07":].isna().all()                               # after his last CAPE: empty, not extended
    assert panel.shiller_cape(None, idx).isna().all() and len(panel.shiller_cape(None, idx)) == len(idx)
    assert not hasattr(panel, "spliced_cape")                              # the NIPA splice is retired


def test_the_shiller_notes_name_his_last_cape_and_the_earnings_caveat():
    sh = _shiller_frame()
    sh.loc["2025-12-01", "earnings"], sh.loc["2026-06-01", "earnings"] = 240.6, 295.4
    lines = panel.shiller_notes(sh)
    assert lines[0].startswith("席勒 CAPE 直接取自 Shiller 的 ie_data.xls（缓存，CAPE 到 2026-10）") and "不再用 NIPA 利润外推" in lines[0]
    assert "盈利 E 到 2026-06：最近两个季度从 240.6（2025-12）变为 295.4" in lines[1]
    assert "E 的最近两个季度可能是估计值" in lines[1]
    assert panel.shiller_notes(None) == ["席勒 CAPE 取自 Shiller 的 ie_data.xls（缓存）；缺少该文件时 CAPE 为空。"]
    assert set(panel.shiller_notes(None)) <= set(panel._notes([]))


def test_the_offline_build_without_shiller_has_no_cape(synthetic):
    P, meta = synthetic
    assert P["s_cape"].isna().all() and P["f_cape"].isna().all() and meta["cape_source"] is None
    assert meta["column_info"]["s_cape"]["sources"] == ["shiller_ie_data"]


@needs_panel
@needs_raw
def test_s_cape_equals_shillers_cape_where_both_exist():
    """P0.3 exit test: to 1e-6 on every month of the panel where the cached workbook has a CAPE."""
    sh = src.shiller(download=False)
    if sh is None:
        pytest.skip("Shiller workbook not cached")
    P = panel.load_panel()
    both = pd.concat([P["s_cape"], sh["cape"].rename("shiller")], axis=1, join="inner").dropna()
    assert len(both) > 900 and float((both["s_cape"] - both["shiller"]).abs().max()) <= 1e-6
    later = P.loc[P.index > sh["cape"].last_valid_index(), "s_cape"]
    assert later.isna().all()                                               # no extension past his last CAPE
    meta = panel.load_meta()
    assert meta["cape_source"] == "shiller" and meta["column_info"]["s_cape"]["sources"] == ["shiller_ie_data"]
    if sh["earnings"].last_valid_index() >= pd.Timestamp("2026-06-01"):     # the 2026 workbook (import_raw)
        assert both.index.max() == P.index.max() and float(P["s_cape"].loc["2026-09-01"]) == pytest.approx(40.58, abs=0.01)
