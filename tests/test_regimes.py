"""Macro regimes (engine/regimes.py): the classifier, the rg_ data contract, MonthDB as-of columns, the
conventions every consumer shares and the engine's unconditional reference run."""
import gc
import json
import re

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.engine import regimes as R


def _panel_month() -> str | None:
    try:
        return str(json.loads(config.PANEL_META_PATH.read_text(encoding="utf-8"))["end"])[:7]
    except (OSError, ValueError, KeyError, TypeError):
        return None


needs_panel = pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
# exact check values (labels, counts, episodes, NBER scores, the reference run) were computed on the panel
# that ends 2026-09; a panel with later months or revised data skips them instead of failing
CHECK_MONTH = "2026-09"
check_values = pytest.mark.skipif(_panel_month() != CHECK_MONTH,
                                  reason=f"check values are for the panel ending {CHECK_MONTH}")


@pytest.fixture(scope="module")
def panel():
    from scenariolab.context import load_panel_with_shocks
    return load_panel_with_shocks()


@pytest.fixture(scope="module")
def ctx():
    from scenariolab.context import EngineContext
    return EngineContext.load(selection=True)


def _ids(codes) -> list[str]:
    return [R.ID_OF[int(c)] for c in codes]


def _synthetic(T_hist=120, months_down=12, ip_step=-1.5, u_step=0.4, cpi=2.0, T_after=48):
    """Steady growth, a contraction of `months_down` months, then a recovery."""
    ip = [100.0 * 1.002 ** t for t in range(T_hist)]
    u = [4.0] * T_hist
    for _ in range(months_down):
        ip.append(ip[-1] * (1 + ip_step / 100))
        u.append(u[-1] + u_step)
    for _ in range(T_after):
        ip.append(ip[-1] * 1.006)
        u.append(max(u[-1] - 0.05, 4.0))
    return np.full(len(ip), cpi), np.array(u), np.array(ip), T_hist


def _walks(seed: int, n: int = 3, T: int = 96):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ip = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.012, T)))
        ur = np.clip(5 + np.cumsum(rng.normal(0, 0.15, T)), 2, 15)
        cpi = 3 + np.cumsum(rng.normal(0, 0.3, T))
        out.append((cpi, ur, ip))
    return out


# ---------------------------------------------------------------- 1. synthetic cases

def test_a_deep_low_inflation_contraction_is_depression_from_the_month_after_the_peak_through_the_trough():
    c, u, ip, th = _synthetic()
    r = R.classify(c, u, ip)
    labs = _ids(r["label"])
    peak, trough = th - 1, th + 11                   # the last growth month is the IP peak
    assert labs[peak] == "expansion"
    assert all(x == "depression" for x in labs[peak + 1: trough + 1])
    assert labs[trough + 1] == "expansion"
    assert r["episode_ip_decline"][trough] == pytest.approx(100 * (1 - 0.985 ** 12), abs=1e-6)
    assert r["contraction"][peak + 1: trough + 1].all() and r["severe"][peak + 1: trough + 1].all()


def test_the_realtime_contraction_switches_on_and_off_on_the_rule():
    c, u, ip, th = _synthetic()
    r = R.classify(c, u, ip)
    on = np.flatnonzero(r["contraction_rt"])
    trough = th + 11
    assert on[0] == th + 2                           # month 2: IP -2.98% (no); month 3: -4.4%, u3 +0.8 (yes)
    assert on[-1] == trough + 4                      # 1.006^5 = +3.04% confirms the trough 5 months after it
    assert _ids(r["label_rt"])[th + 2] == "recession"           # depth so far 4.4% < 15%
    assert _ids(r["label_rt"])[th + 11] == "depression"         # depth so far 16.6%


def test_the_same_contraction_at_six_percent_inflation_is_stagflation_however_deep():
    c, u, ip, th = _synthetic()
    labs = _ids(R.classify(np.full_like(c, 6.0), u, ip)["label"])
    assert all(x == "stagflation" for x in labs[th: th + 12])
    assert labs[th - 1] == "inflation"


def test_the_episode_average_inflation_decides_the_label():
    c, u, ip, th = _synthetic()
    c2 = c.copy()
    c2[th: th + 4] = 5.0                             # 5% for the first 4 months, 1% after: average 2.33%
    c2[th + 4:] = 1.0
    r = R.classify(c2, u, ip)
    assert all(x == "depression" for x in _ids(r["label"])[th: th + 12])
    assert _ids(r["label_rt"])[th + 2] == "stagflation"          # real time: the average so far is 5%


def test_a_mild_contraction_is_recession():
    c, u, ip, th = _synthetic(months_down=10, ip_step=-0.5, u_step=0.15)
    assert set(_ids(R.classify(c, u, ip)["label"])[th: th + 10]) == {"recession"}


def test_an_ip_fall_without_an_unemployment_rise_is_no_contraction():
    c, u, ip, _ = _synthetic(u_step=0.0)
    r = R.classify(c, u, ip)
    assert not r["contraction"].any() and not r["contraction_rt"].any()


def test_negative_cpi_without_contraction_is_deflation():
    c, u, ip, _ = _synthetic(months_down=0)
    r = R.classify(np.full_like(c, -1.0), u, ip)
    assert set(_ids(r["label"])) == {"deflation"} and r["cpi_neg"].all()


def test_paths_with_history_prepended():
    c1, u1, ip1, th = _synthetic()
    c2, u2, ip2, _ = _synthetic(u_step=0.0)
    r2 = R.classify(np.vstack([c1, c2]), np.vstack([u1, u2]), np.vstack([ip1, ip2]), n_hist=th)
    full = R.classify(c1, u1, ip1)
    assert r2["label"].shape == (2, len(c1) - th)
    for k in ("label", "label_rt", "contraction", "episode", "episode_ip_decline", "sahm_gap"):
        assert np.array_equal(r2[k][0], full[k][th:], equal_nan=True), k
    assert not r2["contraction"][1].any()
    assert all(v.base is None for v in r2.values())  # copies: the prepended history is not kept alive


# ---------------------------------------------------------------- 2. real time = the rule on data through t

def test_the_realtime_label_is_the_rule_on_data_through_t_at_every_month():
    paths = [_synthetic()[:3]] + _walks(0) + _walks(11)
    contractions = 0
    for cc, uu, ii in paths:
        full = R.classify(cc, uu, ii)
        contractions += int(full["contraction_rt"].any())
        for t in range(len(ii)):
            cut = R.classify(cc[: t + 1], uu[: t + 1], ii[: t + 1])
            assert cut["label"][t] == full["label_rt"][t], t
            assert cut["contraction"][t] == full["contraction_rt"][t], t
            assert cut["severe"][t] == full["severe_rt"][t], t
        assert full["label"][-1] == full["label_rt"][-1]
    assert contractions >= 3                         # the walks do contract, so the check has content


@needs_panel
def test_the_realtime_label_is_the_rule_on_the_panel_cut_at_t(panel):
    cpi, ur, ip = (panel[c].to_numpy(float) for c in R.INPUTS.values())
    full = R.classify(cpi, ur, ip)
    for t in list(range(0, len(ip), 11)) + [len(ip) - 1]:
        cut = R.classify(cpi[: t + 1], ur[: t + 1], ip[: t + 1])
        assert cut["label"][t] == full["label_rt"][t], panel.index[t]
        assert cut["contraction"][t] == full["contraction_rt"][t], panel.index[t]
    assert full["label"][-1] == full["label_rt"][-1]


# ---------------------------------------------------------------- 3. the own-month unknown mask, flag names

def test_a_month_without_its_own_reading_is_unknown():
    c, u, ip, th = _synthetic()
    for arr in (c, u, ip):
        saved = arr[50]
        arr[50] = np.nan
        r = R.classify(c, u, ip)
        assert r["label"][50] == -1 and r["label_rt"][50] == -1
        assert r["label"][49] >= 0 and r["label"][51] >= 0    # the 3-month mean would have labelled it
        arr[50] = saved


def test_classify_flags_are_renamed_and_core_low_is_history_only():
    c, u, ip, _ = _synthetic()
    r = R.classify(c, u, ip, core_yoy=np.full_like(c, 0.5))
    assert "deflation" not in r and "deflation_risk" not in r
    assert {"cpi_neg", "core_low", "infl_rising"} <= set(r) and r["core_low"].all()
    assert "core_low" not in R.classify(c, u, ip)
    assert R.FLAG_IDS == ("contraction", "severe", "high_infl", "cpi_neg", "core_low", "contraction_rt", "severe_rt")
    assert [f["id"] for f in R.flag_defs()] == list(R.FLAG_IDS)


# Labels of the panel ending 2026-09 (1948-01..2026-09) as the classifier study recorded them: each label
# run as its first month plus a letter, and each flag as its runs of months.
LETTER = {"E": "expansion", "I": "inflation", "D": "deflation", "R": "recession", "P": "depression",
          "S": "stagflation"}
LABEL_RUNS = {
    "label": (
        "1948-01I 1948-08R 1949-11D 1950-07E 1950-12I 1952-02E 1953-08R 1954-05E 1954-09D 1955-09E 1957-05R "
        "1958-05E 1960-03R 1961-01E 1968-05I 1970-01S 1970-12I 1971-10E 1973-03I 1973-12S 1975-06I 1980-03S "
        "1980-08I 1981-09S 1983-01E 1983-04I 1983-05E 1984-01I 1985-01E 1987-08I 1988-02E 1988-07I 1990-10S "
        "1991-04I 1991-08E 2001-01R 2002-01E 2005-09I 2005-11E 2006-01I 2006-02E 2006-06I 2006-08E 2007-11I "
        "2008-01P 2009-07D 2009-11E 2015-01D 2015-05E 2020-03P 2020-05E 2021-04I 2023-06E 2026-05I 2026-06E"),
    "label_rt": (
        "1948-01I 1948-12E 1949-02S 1949-04R 1949-12D 1950-07E 1950-12I 1952-02E 1953-11R 1954-11D 1955-09E "
        "1957-10R 1958-06E 1959-11R 1959-12E 1960-10R 1961-05E 1968-05I 1970-03S 1971-01I 1971-10E 1973-03I "
        "1974-11S 1975-09I 1980-05S 1980-10I 1981-12S 1983-04I 1983-05E 1984-01I 1985-01E 1987-08I 1988-02E "
        "1988-07I 1991-02S 1991-09E 2001-06R 2002-06E 2005-09I 2005-11E 2006-01I 2006-02E 2006-06I 2006-08E "
        "2007-11I 2008-03E 2008-05I 2008-08S 2008-12R 2009-03P 2009-09D 2009-11E 2015-01D 2015-05E 2020-04P "
        "2020-06E 2021-04I 2023-06E 2026-05I 2026-06E"),
}
FLAG_RUNS = {
    "contraction": ("1948-08..1949-10 1953-08..1954-04 1957-05..1958-04 1960-03..1960-12 1970-01..1970-11 "
                    "1973-12..1975-05 1980-03..1980-07 1981-09..1982-12 1990-10..1991-03 2001-01..2001-12 "
                    "2008-01..2009-06 2020-03..2020-04"),
    "severe": "2008-01..2009-06 2020-03..2020-04",
    "high_infl": ("1948-01..1948-11 1950-12..1952-01 1968-05..1971-09 1973-03..1982-11 1983-04 1984-01..1984-12 "
                  "1987-08..1988-01 1988-07..1991-07 2005-09..2005-10 2006-01 2006-06..2006-07 2007-11..2008-02 "
                  "2008-05..2008-09 2021-04..2023-05 2026-05"),
    "cpi_neg": "1949-05..1950-06 1954-09..1955-08 2008-12..2009-01 2009-03..2009-10 2015-01..2015-04",
    "core_low": "1960-09 1960-12..1961-06 1961-10 1962-11 1963-01 1963-05 2010-04..2011-01",
    "contraction_rt": ("1949-02..1949-11 1953-11..1954-10 1957-10..1958-05 1959-11 1960-10..1961-04 "
                       "1970-03..1970-12 1974-11..1975-08 1980-05..1980-09 1981-12..1983-03 1991-02..1991-08 "
                       "2001-06..2002-05 2008-08..2009-08 2020-04..2020-05"),
    "severe_rt": "2009-03..2009-08 2020-04..2020-05",
}


def _expand_labels(text: str, months: list[str]) -> np.ndarray:
    pos = {m: i for i, m in enumerate(months)}
    starts = [(pos[tok[:7]], LETTER[tok[7]]) for tok in text.split()]
    out = np.empty(len(months), object)
    for (a, lab), (b, _) in zip(starts, starts[1:] + [(len(months), None)]):
        out[a:b] = lab
    return out


def _expand_runs(text: str, months: list[str]) -> np.ndarray:
    pos = {m: i for i, m in enumerate(months)}
    out = np.zeros(len(months), int)
    for tok in text.split():
        a, _, b = tok.partition("..")
        out[pos[a]: pos[b or a] + 1] = 1
    return out


@needs_panel
@check_values
def test_labels_of_the_panel_equal_the_recorded_labels_in_every_month(panel):
    L = R.classify_panel(panel)
    known = L[L["label"] != ""]
    months = known.index.strftime("%Y-%m").tolist()
    assert len(known) == 945 and months[0] == "1948-01" and months[-1] == CHECK_MONTH
    assert (L.loc[L.index < "1948-01-01", "label"] == "").all()
    for key, text in LABEL_RUNS.items():
        got, want = known[key].to_numpy(object), _expand_labels(text, months)
        assert (got == want).all(), (key, [months[i] for i in np.flatnonzero(got != want)[:5]])
    for key, text in FLAG_RUNS.items():
        got, want = known[key].to_numpy(int), _expand_runs(text, months)
        assert (got == want).all(), (key, [months[i] for i in np.flatnonzero(got != want)[:5]])


# ---------------------------------------------------------------- 4. nowcast

def _append_month(panel: pd.DataFrame, missing: str) -> pd.DataFrame:
    extra = panel.iloc[[-1]].copy()
    extra.index = pd.DatetimeIndex([panel.index[-1] + pd.offsets.MonthBegin(1)])
    extra[missing] = np.nan
    return pd.concat([panel, extra])


@needs_panel
@check_values
def test_nowcast_of_the_panel(panel):
    nc = R.nowcast(panel)
    assert (nc["month"], nc["known_from"], nc["label"], nc["name"], nc["label_expost"]) == (
        "2026-09", "2026-10", "expansion", "扩张", "expansion")
    checks = {c["id"]: c for c in nc["checks"]}
    assert list(checks) == ["high_infl", "cpi_neg", "ip_leg", "u_rise", "severe", "core_low"]
    want = {"high_infl": (3.717, 4.0, -0.283), "cpi_neg": (3.717, 0.0, 3.717), "ip_leg": (0.0, 3.0, -3.0),
            "u_rise": (0.0, 0.5, -0.5), "severe": (0.0, 15.0, -15.0), "core_low": (2.566, 1.0, 1.566)}
    for cid, (reading, threshold, distance) in want.items():
        c = checks[cid]
        assert c["reading"] == pytest.approx(reading, abs=1e-3) and c["threshold"] == threshold
        assert c["distance"] == pytest.approx(distance, abs=1e-3) and c["on"] is False
        assert c["name"] and c["rule"]
    readings = {"infl_trend_12m": 0.695, "infl_trend_6m": 0.431, "unrate": 4.2, "unrate_3m": 4.133, "sahm_gap": 0.0,
                "misery": 7.917, "ip_yoy": 2.245, "ip_drawdown_24m": 0.0}
    for k, v in readings.items():
        assert nc["readings"][k] == pytest.approx(v, abs=1e-3), k
    hot = nc["last_high_infl"]
    assert hot["month"] == "2026-05" and hot["cpi_yoy"] == pytest.approx(4.167, abs=1e-3)
    assert nc["provisional_from"] is None and nc["missing_after"] == {}
    assert len(nc["recent"]) == 24 and nc["recent"][-1]["month"] == "2026-09"
    assert "单边、修订后数据" in nc["note"]


@needs_panel
@pytest.mark.parametrize("col, key", [("s_cpi_yoy", "cpi_yoy"), ("lvl_unrate", "unrate"), ("lvl_ip", "ip")])
def test_nowcast_reads_the_last_complete_month_and_lists_what_is_missing_after_it(panel, col, key):
    last = panel.index[-1]
    nc = R.nowcast(_append_month(panel, col))                    # no exception on a trailing gap
    assert nc["month"] == last.strftime("%Y-%m")
    assert nc["missing_after"] == {key: [(last + pd.offsets.MonthBegin(1)).strftime("%Y-%m")]}
    assert nc["label"] == R.nowcast(panel)["label"]


def test_nowcast_without_a_complete_month_raises_a_clear_error():
    idx = pd.date_range("2000-01-01", periods=6, freq="MS")
    P = pd.DataFrame({"s_cpi_yoy": np.nan, "lvl_unrate": 4.0, "lvl_ip": 100.0}, index=idx)
    with pytest.raises(ValueError):
        R.nowcast(P)


def test_an_empty_panel_has_no_labels_no_provisional_months_and_no_nowcast():
    full = R.classify(np.ones(3), np.full(3, 4.0), np.full(3, 100.0))
    r = R.classify(np.array([]), np.array([]), np.array([]))
    assert set(r) == set(full) and all(r[k].shape == (0,) and r[k].dtype == full[k].dtype for k in r)
    assert R.classify(np.zeros((2, 0)), np.zeros((2, 0)), np.zeros((2, 0)))["label_rt"].shape == (2, 0)
    P = pd.DataFrame({"s_cpi_yoy": [], "lvl_unrate": [], "lvl_ip": [], "s_core_yoy": []},
                     index=pd.DatetimeIndex([]), dtype=float)
    assert R.provisional_from(P) is None
    assert R.classify_panel(P).empty and R.regime_columns(P).empty
    with pytest.raises(ValueError):
        R.nowcast(P)


def _synthetic_panel():
    c, u, ip, th = _synthetic()
    idx = pd.date_range("1990-01-01", periods=len(ip), freq="MS")
    return pd.DataFrame({"s_cpi_yoy": c, "lvl_unrate": u, "lvl_ip": ip, "s_core_yoy": c - 1.5}, index=idx), th


def test_nowcast_inside_a_contraction_and_after_a_trailing_gap():
    P, th = _synthetic_panel()
    cut = P.iloc[: th + 6]                           # six months into the fall: confirmed, 8.7% below the peak
    nc = R.nowcast(cut)
    assert (nc["month"], nc["label"], nc["label_expost"]) == ("2000-06", "recession", "recession")
    assert nc["provisional_from"] == "2000-01"       # the down-leg's months stay open
    checks = {c["id"]: c for c in nc["checks"]}
    assert list(checks) == ["high_infl", "cpi_neg", "ip_leg", "u_rise", "severe", "infl_episode", "core_low"]
    assert checks["ip_leg"]["on"] and checks["ip_leg"]["distance"] is None
    assert checks["u_rise"]["on"] and checks["u_rise"]["distance"] == pytest.approx(1.5)
    assert checks["severe"]["reading"] == pytest.approx(100 * (1 - 0.985 ** 6)) and not checks["severe"]["on"]
    assert not checks["infl_episode"]["on"] and checks["core_low"]["on"]
    gap = cut.copy()
    gap.loc[pd.Timestamp("2000-07-01")] = [np.nan, 5.0, 90.0, np.nan]
    later = R.nowcast(gap)
    assert later["month"] == "2000-06" and later["missing_after"] == {"cpi_yoy": ["2000-07"]}
    prose = [c["name"] + c["rule"] for c in nc["checks"]] + [nc["note"]]
    assert not [t for t in prose if re.search(r"[A-Za-z]\w*_\w+", t)]


def test_regime_columns_on_a_synthetic_panel():
    P, th = _synthetic_panel()
    P.iloc[50, P.columns.get_loc("lvl_unrate")] = np.nan
    cols = R.regime_columns(P)
    assert list(cols.columns) == R.COLUMNS and cols.iloc[50].isna().all()
    known = cols.drop(index=P.index[50])
    assert (known[[f"rg_{i}" for i in R.LABEL_IDS]].sum(axis=1) == 1.0).all()
    assert (cols.iloc[th: th + 12]["rg_depression"] == 1.0).all() and cols.iloc[th + 3]["rg_rt_recession"] == 1.0
    assert R.label_codes(cols)[th] == R.CODE["depression"] and R.label_codes(cols)[50] == -1


# ---------------------------------------------------------------- 5. provisional months

@needs_panel
@check_values
def test_provisional_from(panel):
    assert R.provisional_from(panel) is None                      # IP is at its running peak
    for end, first in (("2008-06", "2008-01"), ("2020-03", "2018-10"), ("2024-09", "2022-05")):
        assert R.provisional_from(panel.loc[:end]) == pd.Timestamp(first), end


def test_provisional_months_are_the_months_later_data_can_still_relabel():
    c, u, ip, th = _synthetic()
    for t in (th - 1, th + 3, th + 11, th + 30):
        first = R.provisional_from(pd.DataFrame({"lvl_unrate": u[: t + 1], "lvl_ip": ip[: t + 1]}))
        cut = R.classify(c[: t + 1], u[: t + 1], ip[: t + 1])["label"]
        full = R.classify(c, u, ip)["label"]
        final = len(cut) if first is None else first
        assert np.array_equal(cut[:final], full[:final]), t      # months before it never change
    first = R.provisional_from(pd.DataFrame({"lvl_unrate": u[: th + 6], "lvl_ip": ip[: th + 6]}))
    assert first == th                                           # inside the down-leg: the leg's months


# ---------------------------------------------------------------- 6. rg_ columns

@needs_panel
def test_regime_columns_one_hot_and_unknown_months(panel):
    cols = R.regime_columns(panel)
    assert list(cols.columns) == R.COLUMNS and cols.index.equals(panel.index)
    before = cols.index < pd.Timestamp(config.REGIMES["first_month"])
    assert before.any() and cols[before].isna().all().all()
    for pre in ("rg_", "rg_rt_"):
        s = cols.loc[~before, [f"{pre}{i}" for i in R.LABEL_IDS]].sum(axis=1)
        assert (s == 1.0).all()
    vals = cols.loc[~before].to_numpy(float)
    assert np.isin(vals, (0.0, 1.0)).all()
    end = "2008-06"
    cut = R.regime_columns(panel, end=end)
    assert cut.loc[:end].equals(R.regime_columns(panel.loc[:end])) and cut.loc["2008-07":].isna().all().all()


@needs_panel
def test_realtime_columns_are_one_sided(panel):
    full = R.regime_columns(panel)[R.REALTIME_COLUMNS]
    for end in panel.index[24::37].append(panel.index[-2:]):
        cut = R.regime_columns(panel.loc[:end])[R.REALTIME_COLUMNS]
        assert np.array_equal(cut.to_numpy(float), full.loc[:end].to_numpy(float), equal_nan=True), end


@needs_panel
def test_rg_columns_are_computed_on_load_and_never_stored(panel):
    from scenariolab.data.panel import load_panel
    stored = pd.read_csv(config.PANEL_PATH, nrows=0).columns
    assert not any(c.startswith("rg_") for c in stored)
    assert not any(c.startswith("rg_") for c in load_panel().columns)
    assert all(c in panel.columns for c in R.COLUMNS)
    assert panel[R.COLUMNS].equals(R.regime_columns(load_panel()))


def test_a_panel_without_the_inputs_loads_without_rg_columns(monkeypatch, tmp_path):
    from scenariolab import context
    idx = pd.date_range("2000-01-01", periods=4, freq="MS")
    monkeypatch.setattr(context, "load_panel", lambda: pd.DataFrame({"s_other": 1.0}, index=idx))
    monkeypatch.setattr(context, "SHOCKS_PATH", tmp_path / "none.csv")
    assert list(context.load_panel_with_shocks().columns) == ["s_other"]


# ---------------------------------------------------------------- 7. MonthDB as-of columns

@needs_panel
def test_monthdb_that_ends_early_sees_regime_labels_as_of_its_end(ctx):
    from scenariolab.engine.simulate import MonthDB
    full = ctx.panel
    for end in ("2004-12", "2008-06", "2009-03", "2020-03"):
        db = MonthDB(full, ctx.model, end=end)
        asof = R.regime_columns(full.loc[:end]).loc[db.dates]
        f = db.frame
        assert f[R.EXPOST_COLUMNS].equals(asof[R.EXPOST_COLUMNS]), end
        assert f[R.REALTIME_COLUMNS].equals(full.loc[db.dates, R.REALTIME_COLUMNS]), end
    # on the whole panel these windows would already see the contraction that later data date back
    for end, months in (("2008-06", pd.date_range("2008-01-01", "2008-06-01", freq="MS")),
                        ("2020-03", pd.DatetimeIndex(["2020-03-01"]))):
        f = MonthDB(full, ctx.model, end=end).frame
        differs = f.index[(f["rg_contraction"] != full.loc[f.index, "rg_contraction"]).to_numpy()]
        assert differs.equals(months), end
        assert (full.loc[months, "rg_contraction"] == 1.0).all() and (f.loc[months, "rg_contraction"] == 0.0).all()
    # a window that ends with the panel keeps the full-sample columns
    db = MonthDB(full, ctx.model, end=full.index[-1])
    assert db.frame[R.EXPOST_COLUMNS].equals(full.loc[db.dates, R.EXPOST_COLUMNS])


@needs_panel
def test_the_asof_cache_returns_the_same_object_and_forgets_a_dead_panel(panel):
    a = R.asof_columns(panel, "2008-06")
    assert R.asof_columns(panel, "2008-06") is a
    assert R.asof_columns(panel, pd.Timestamp("2008-06-01")) is a                # same last month, same entry
    assert R.asof_columns(panel, "2009-03") is not a
    other = panel.copy()
    b = R.asof_columns(other, "2008-06")
    assert b is not a and b.equals(a)                                           # a new panel object, its own entry
    other.loc["2008-03-01", "lvl_ip"] *= 0.5                                    # changed in place before the end
    assert R.asof_columns(other, "2008-06") is not b
    key = id(other)
    assert key in R._ASOF
    del other, b
    gc.collect()
    assert key not in R._ASOF


@needs_panel
def test_the_asof_cache_keeps_a_bounded_number_of_windows(panel, monkeypatch):
    other = panel.copy()
    monkeypatch.setattr(R, "_ASOF_MAX", 2)
    first = R.asof_columns(other, "2001-06")
    R.asof_columns(other, "2002-06")
    R.asof_columns(other, "2003-06")                                           # the oldest window goes
    assert len(R._ASOF[id(other)][1]) == 2
    again = R.asof_columns(other, "2001-06")
    assert again is not first and again.equals(first)


# ---------------------------------------------------------------- 8. invariants: nothing simulated changes

@needs_panel
def test_rg_columns_change_no_month_and_no_simulated_number(ctx, monkeypatch):
    from scenariolab.context import EngineContext
    from scenariolab.scenarios import library, runner
    monkeypatch.setattr(R, "has_inputs", lambda panel: False)                  # load the panel without rg_
    bare = EngineContext.load(selection=True)
    monkeypatch.undo()
    assert not any(c.startswith("rg_") for c in bare.panel.columns)
    assert ctx.db.dates.equals(bare.db.dates)
    assert set(ctx.db.frame.columns) - set(bare.db.frame.columns) == set(R.COLUMNS)
    specs = {s["id"]: s for s in library.resolved_specs(ctx)}
    # the regime readout switch, once run_spec has one, must not change a simulated number either
    switch = [True, False] if hasattr(runner, "REGIME_READOUT") else [None]
    for sid in ("baseline", "stagflation", "financial_crisis"):
        seed = library.seed_for(sid)
        _, ref = runner.run_spec(bare, specs[sid], n_paths=600, seed=seed, attribution=False)
        for on in switch:
            if on is not None:
                monkeypatch.setattr(runner, "REGIME_READOUT", on)
            _, sim = runner.run_spec(ctx, specs[sid], n_paths=600, seed=seed, attribution=False)
            assert np.array_equal(sim.total, ref.total) and np.array_equal(sim.idx, ref.idx), (sid, on)
            assert np.array_equal(sim.income, ref.income), (sid, on)
            assert sim.macro.keys() == ref.macro.keys()
            assert all(np.array_equal(sim.macro[k], ref.macro[k]) for k in ref.macro), (sid, on)


@needs_panel
@check_values
def test_monthdb_months(ctx):
    assert len(ctx.db) == 644
    assert (ctx.db.dates[0].strftime("%Y-%m"), ctx.db.dates[-1].strftime("%Y-%m")) == ("1973-02", "2026-09")


# ---------------------------------------------------------------- 9. NBER validation

PER_RECESSION = {   # NBER first month -> flagged run (ex post, real time)
    "1948-12": ("1948-08..1949-10", "1949-02..1949-11"), "1953-08": ("1953-08..1954-04", "1953-11..1954-10"),
    "1957-09": ("1957-05..1958-04", "1957-10..1958-05"), "1960-05": ("1960-03..1960-12", "1960-10..1961-04"),
    "1970-01": ("1970-01..1970-11", "1970-03..1970-12"), "1973-12": ("1973-12..1975-05", "1974-11..1975-08"),
    "1980-02": ("1980-03..1980-07", "1980-05..1980-09"), "1981-08": ("1981-09..1982-12", "1981-12..1983-03"),
    "1990-08": ("1990-10..1991-03", "1991-02..1991-08"), "2001-04": ("2001-01..2001-12", "2001-06..2002-05"),
    "2008-01": ("2008-01..2009-06", "2008-08..2009-08"), "2020-03": ("2020-03..2020-04", "2020-04..2020-05"),
}


def test_nber_months():
    idx = pd.date_range("2007-11-01", "2009-08-01", freq="MS")
    m = R.nber_months(idx)
    assert m.sum() == 18 and idx[m.astype(bool)][0] == pd.Timestamp("2008-01-01")
    assert len(R.NBER_PEAK_TROUGH) == 12


@needs_panel
@check_values
def test_nber_validation(panel):
    v = R.validate_nber(panel)
    assert v["window"] == list(config.REGIMES["nber_window"])
    ex, rt = v["expost"], v["realtime"]
    assert ex["balanced_accuracy"] >= 0.955 and ex["balanced_accuracy"] == pytest.approx(0.960, abs=5e-4)
    assert ex["detected"] == ex["recessions"] == 12 and ex["false_alarms"] == []
    assert rt["detected"] == rt["recessions"] == 12 and rt["false_alarms"] == [["1959-11", "1959-11"]]
    assert rt["balanced_accuracy"] == pytest.approx(0.786, abs=5e-4)
    assert v["months_flagged_after_window"] == {"expost": 0, "realtime": 0}
    for name, k in (("expost", 0), ("realtime", 1)):
        got = {r["first_month"]: f"{r['start']}..{r['end']}" for r in v[name]["per_recession"]}
        assert got == {m: runs[k] for m, runs in PER_RECESSION.items()}, name


# ---------------------------------------------------------------- 10. conventions

def test_annualized_real_log_returns_equal_the_chain_linked_product():
    rng = np.random.default_rng(5)
    r, f = rng.normal(0.008, 0.04, 37), rng.normal(0.003, 0.002, 37)
    chain = np.prod((1 + r) * np.exp(-f)) ** (12 / 37) - 1
    assert R.annualized(R.real_log_return(r, f).sum(), 37) == pytest.approx(chain, rel=1e-12)
    assert np.isnan(R.annualized(0.0, 0))


def test_runs_on_edge_masks():
    assert R.runs([]) == [] and R.runs([False, False]) == []
    assert R.runs([True] * 5) == [(0, 4)]
    assert R.runs([True, False, True]) == [(0, 0), (2, 2)]
    assert R.runs([False, True, True, False, True]) == [(1, 2), (4, 4)]
    assert R.runs(np.array([1.0, np.nan, 1.0, 0.0])) == [(0, 0), (2, 2)]      # NaN (unknown) is not in the run


def test_rng_for_is_order_independent():
    a = R.rng_for("assets", "1973", "label:expansion", "us_equity", "ann_real").random(5)
    R.rng_for("assets", "1973", "label:inflation", "us_equity", "ann_real").random(100)   # another cell in between
    b = R.rng_for("assets", "1973", "label:expansion", "us_equity", "ann_real").random(5)
    c = R.rng_for("assets", "1973", "label:inflation", "us_equity", "ann_real").random(5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    import zlib
    seed = config.REGIMES["bootstrap"]["seed"]
    want = np.random.default_rng([seed, zlib.crc32(b"forward|1948|all|cash|mean")]).random(3)
    assert np.array_equal(R.rng_for("forward", "1948", "all", "cash", "mean").random(3), want)


def test_window_masks_and_shares():
    idx = pd.date_range("1947-01-01", "1975-12-01", freq="MS")
    db_dates = pd.date_range("1973-02-01", "1975-12-01", freq="MS")
    m73 = R.window_mask(idx, "1973", db_dates)
    assert idx[m73].equals(db_dates)
    m48 = R.window_mask(idx, "1948", db_dates)
    assert idx[m48][0] == pd.Timestamp("1948-01-01") and idx[m48][-1] == db_dates[-1]
    assert R.window_mask(idx, "1948").sum() == len(idx) - 12
    with pytest.raises(ValueError):
        R.window_mask(idx, "1973")
    s = R.shares(np.array([[0, 1, 1, -1], [5, 5, 5, 0]]))
    assert list(s) == R.LABEL_IDS and sum(s.values()) == pytest.approx(1.0)
    assert s["stagflation"] == pytest.approx(3 / 7) and s["deflation"] == 0.0


def test_label_codes_from_one_hot_columns():
    frame = pd.DataFrame({f"rg_{i}": 0.0 for i in R.LABEL_IDS} | {f"rg_rt_{i}": 0.0 for i in R.LABEL_IDS},
                         index=range(3))
    frame.loc[0, "rg_inflation"] = 1.0
    frame.loc[1, "rg_stagflation"] = 1.0
    frame.loc[2, :] = np.nan
    frame.loc[0, "rg_rt_expansion"] = 1.0
    assert R.label_codes(frame).tolist() == [1, 5, -1]
    assert R.label_codes(frame, realtime=True).tolist() == [0, -1, -1]


# ---------------------------------------------------------------- 11. check values

def _count(mask_window, values, ids):
    n = int(mask_window.sum())
    return {i: {"months": int((mask_window & (values == i)).sum()),
                "share": round(float((mask_window & (values == i)).sum() / n), 4),
                "runs": len(R.runs(mask_window & (values == i)))} for i in ids}


LABEL_COUNTS = {   # (window, version): {label: (months, share, runs)}
    ("1948", "label"): {"expansion": (569, 0.6021, 21), "inflation": (214, 0.2265, 18), "deflation": (28, 0.0296, 4),
                        "recession": (58, 0.0614, 5), "depression": (20, 0.0212, 2), "stagflation": (56, 0.0593, 5)},
    ("1973", "label"): {"expansion": (396, 0.6149, 15), "inflation": (163, 0.2531, 14), "deflation": (8, 0.0124, 2),
                        "recession": (12, 0.0186, 1), "depression": (20, 0.0311, 2), "stagflation": (45, 0.0699, 4)},
    ("1948", "label_rt"): {"expansion": (574, 0.6074, 22), "inflation": (235, 0.2487, 18),
                           "deflation": (23, 0.0243, 4), "recession": (51, 0.054, 7), "depression": (8, 0.0085, 2),
                           "stagflation": (54, 0.0571, 7)},
    ("1973", "label_rt"): {"expansion": (394, 0.6118, 15), "inflation": (179, 0.278, 14),
                           "deflation": (6, 0.0093, 2), "recession": (15, 0.0233, 2), "depression": (8, 0.0124, 2),
                           "stagflation": (42, 0.0652, 5)},
}
FLAG_COUNTS = {   # flag: ((months, runs) since 1948, (months, runs) in the MonthDB months)
    "contraction": ((134, 12), (77, 7)), "severe": ((20, 2), (20, 2)), "high_infl": ((280, 15), (214, 12)),
    "cpi_neg": ((40, 5), (14, 3)), "core_low": ((22, 7), (10, 1)), "contraction_rt": ((113, 13), (65, 7)),
    "severe_rt": ((8, 2), (8, 2)),
}
EPISODES = [   # label, start, end, IP decline, CPI mean, real equity: total, to the low, low month, after the low,
    #            months after the low; real Treasuries; worst nominal equity window (start, months, return)
    ("recession", "1948-08", "1949-10", 10.1, 1.4, 11.8, -5.5, "1948-11", 18.2, 11, None, ("1948-08", 12, -0.9)),
    ("recession", "1953-08", "1954-04", 9.5, 0.9, 19.8, -4.6, "1953-08", 25.6, 8, 7.6, ("1953-08", 9, 20.1)),
    ("recession", "1957-05", "1958-04", 13.6, 3.4, -5.7, -13.1, "1957-12", 8.5, 4, 5.2, ("1957-05", 12, -2.2)),
    ("recession", "1960-03", "1960-12", 8.6, 1.5, 5.4, -3.9, "1960-10", 9.7, 2, 7.5, ("1960-03", 10, 6.9)),
    ("stagflation", "1970-01", "1970-11", 7.0, 5.9, -10.2, -24.9, "1970-06", 19.5, 5, 12.2, ("1970-01", 11, -5.7)),
    ("stagflation", "1973-12", "1975-05", 13.1, 10.8, -12.8, -39.1, "1974-09", 43.2, 8, -12.0, ("1974-01", 12, -27.8)),
    ("stagflation", "1980-03", "1980-07", 6.7, 14.2, 4.5, -12.9, "1980-03", 20.0, 4, 11.9, ("1980-03", 5, 9.3)),
    ("stagflation", "1981-09", "1982-12", 9.3, 7.1, 16.2, -14.4, "1982-07", 35.7, 5, 48.5, ("1981-09", 12, 1.4)),
    ("stagflation", "1990-10", "1991-03", 4.2, 5.8, 24.9, -1.9, "1990-10", 27.4, 5, 7.8, ("1990-10", 6, 27.1)),
    ("recession", "2001-01", "2001-12", 5.4, 2.8, -12.6, -22.7, "2001-09", 13.0, 3, 3.7, ("2001-01", 12, -11.2)),
    ("depression", "2008-01", "2009-06", 17.2, 2.4, -34.6, -48.0, "2009-02", 25.7, 4, 7.7, ("2008-03", 12, -42.6)),
    ("depression", "2020-03", "2020-04", 18.8, 0.9, -0.2, -12.9, "2020-03", 14.5, 1, 6.2, ("2020-03", 2, -1.5)),
]


@needs_panel
@check_values
def test_counts_and_agreement(ctx):
    P = ctx.panel
    L = R.classify_panel(P)
    windows = {w: R.window_mask(P.index, w, ctx.db.dates) for w in ("1948", "1973")}
    assert {w: int(m.sum()) for w, m in windows.items()} == {"1948": 945, "1973": 644}
    for (w, version), want in LABEL_COUNTS.items():
        got = _count(windows[w], L[version].to_numpy(object), R.LABEL_IDS)
        assert got == {i: dict(zip(("months", "share", "runs"), v)) for i, v in want.items()}, (w, version)
    for flag, (w48, w73) in FLAG_COUNTS.items():
        x = L[flag].to_numpy() == 1
        for w, (months, n_runs) in (("1948", w48), ("1973", w73)):
            m = x & windows[w]
            assert (int(m.sum()), len(R.runs(m))) == (months, n_runs), (flag, w)
    agree = {w: round(float((L["label"].to_numpy(object)[m] == L["label_rt"].to_numpy(object)[m]).mean()), 4)
             for w, m in windows.items()}
    assert agree == {"1948": 0.8942, "1973": 0.9053}
    eps = R.episodes(L.loc[windows["1948"], "label"])
    assert len(eps) == 55 and int((eps["months"] <= 2).sum()) == 8
    trend = L["infl_trend_12m"].to_numpy(float)
    for w, (n, falling) in (("1948", (207, 59)), ("1973", (163, 48))):
        m = windows[w] & (L["label"].to_numpy(object) == "inflation") & np.isfinite(trend)
        assert (int(m.sum()), int((m & (trend <= 0)).sum())) == (n, falling), w


@needs_panel
@check_values
def test_contraction_episode_table(panel):
    L = R.classify_panel(panel)
    idx = panel.index
    r_eq = panel["r_us_equity"].to_numpy(float)
    eq = R.real_log_return(r_eq, panel["f_cpi"].to_numpy(float))
    bd = R.real_log_return(panel["r_treasury_10y"].to_numpy(float), panel["f_cpi"].to_numpy(float))
    got = []
    for s, e in R.runs(L["contraction"].to_numpy() == 1):
        cum = np.concatenate([[0.0], np.cumsum(eq[s: e + 1])])      # 0 = the end of the month before the start
        k = int(np.argmin(cum))
        cs = np.concatenate([[0.0], np.cumsum(np.log1p(r_eq[s: e + 1]))])
        w = min(12, e - s + 1)
        j = int(np.argmin(cs[w:] - cs[:-w]))
        pct = lambda x: round(100 * float(np.expm1(x)), 1)          # noqa: E731
        got.append((L["label"].iloc[s], idx[s].strftime("%Y-%m"), idx[e].strftime("%Y-%m"),
                    round(float(L["episode_ip_decline"].iloc[s]), 1), round(float(L["episode_cpi_mean"].iloc[s]), 1),
                    pct(cum[-1]), pct(cum[k]), idx[s + k - 1 if k else s - 1].strftime("%Y-%m"), pct(cum[-1] - cum[k]),
                    int(e - (s + k - 1)) if k else e - s + 1,
                    pct(bd[s: e + 1].sum()) if np.isfinite(bd[s: e + 1]).all() else None,
                    (idx[s + j].strftime("%Y-%m"), w, pct(cs[j + w] - cs[j]))))
    assert got == EPISODES


# ---------------------------------------------------------------- 13. thresholds and texts

def test_high_inflation_threshold_matches_the_transmission():
    from scenariolab.engine import factor_model
    assert config.REGIMES["params"]["high_inflation_pct"] == factor_model.HIGH_INFLATION


def test_definitions_follow_the_parameters_and_hold_no_code_ids():
    assert R.LABEL_IDS == ["expansion", "inflation", "deflation", "recession", "depression", "stagflation"]
    assert [d["name"] for d in R.LABELS] == ["扩张", "通胀", "物价负增长", "衰退", "深度衰退", "滞胀"]
    assert {d["id"]: d["term"] for d in R.LABELS if d["term"]} == {"deflation": "通缩", "depression": "萧条"}
    assert sorted(R.PRECEDENCE) == sorted(R.LABEL_IDS) and R.PRECEDENCE[0] == "stagflation"
    p = config.REGIMES["params"]
    desc = {d["id"]: d["desc"] for d in R.LABELS}
    assert f"{p['high_inflation_pct']:g}%" in desc["inflation"]
    assert f"{p['severe_ip_decline_pct']:g}%" in desc["depression"]
    texts = R.rule_text()
    assert list(texts) == ["growth", "severity", "inflation", "precedence", "realtime", "data", "timing"]
    assert f"{p['ip_swing_pct']:g}%" in texts["growth"] and f"{p['u_rise_pp']:g} 个百分点" in texts["growth"]
    changed = R.rule_text({"ip_swing_pct": 4.0, "severe_ip_decline_pct": 12.0})
    assert "4%" in changed["growth"] and "12%" in changed["severity"]
    prose = list(texts.values()) + [f["name"] + f["desc"] for f in R.flag_defs()] + list(desc.values())
    code_id = re.compile(r"[A-Za-z]\w*_\w+|\.py\b|\w+\(\)")
    assert not [t for t in prose if code_id.search(t)]


# ---------------------------------------------------------------- simulated paths and the reference run

@needs_panel
def test_classify_sim_prepends_the_history_and_reads_the_anchor(ctx):
    P = ctx.panel
    H = P.loc[:"2008-05"]                            # IP 2% below its 2007-12 peak, contraction not yet confirmed
    T1 = 13
    ip0, u0, c0 = (H[c].iloc[-1] for c in ("lvl_ip", "lvl_unrate", "s_cpi_yoy"))
    macro = {"ip": np.vstack([ip0 * np.linspace(1, 0.90, T1), ip0 * np.linspace(1, 1.04, T1),
                              ip0 * np.linspace(1, 0.95, T1), ip0 * np.linspace(1, 1.02, T1)]),
             "unrate": np.vstack([np.linspace(u0, 7.5, T1), np.full(T1, u0), np.linspace(u0, 6.5, T1),
                                  np.full(T1, u0)]),
             "cpi_yoy": np.full((4, T1), c0)}
    r = R.classify_sim(macro, P, anchor="2008-05")
    assert r["label"].shape == (4, T1)
    assert all(np.array_equal(r[k], v, equal_nan=True) for k, v in R.classify_sim(macro, H).items())
    # the anchor's real-time label is the same on every path; its ex-post label depends on the path
    assert R.sim_anchor(r) == {"label_rt": "inflation", "label_expost_mix": {"inflation": 0.5, "stagflation": 0.5}}
    hist = H.iloc[:-1]
    one = R.classify(*(np.concatenate([hist[c].to_numpy(float), macro[k][0]]) for k, c in R.INPUTS.items()))
    assert np.array_equal(r["label"][0], one["label"][len(hist):])


@needs_panel
def test_reference_simulation_is_the_unconditional_run_spec_and_is_cached(ctx, monkeypatch):
    from scenariolab.scenarios import runner
    ref = runner.reference_simulation(ctx)
    assert runner.reference_simulation(ctx) is ref
    n, seed = config.REGIMES["reference"]["n_paths"], config.REGIMES["reference"]["seed"]
    assert (ref["n_paths"], ref["seed"], ref["months"]) == (n, seed, config.HORIZON_MONTHS)
    spec = {"id": "unconditional", "name": "无条件", "prior_prob": 0.0,
            "phases": [{"name": "回归常态", "months": config.HORIZON_MONTHS, "views": []}]}
    _, sim = runner.run_spec(ctx, spec, n_paths=n, seed=seed, attribution=False)
    s = ref["sim"]
    assert np.array_equal(s.total, sim.total) and np.array_equal(s.idx, sim.idx)
    assert np.array_equal(s.income, sim.income)
    assert all(np.array_equal(s.macro[k], sim.macro[k]) for k in sim.macro)
    r = ref["regimes"]
    assert tuple(r) == R.LABEL_KEYS and r["label"].shape == (n, config.HORIZON_MONTHS + 1)
    full = R.classify_sim(s.macro, ctx.panel, anchor=ctx.anchor.date)
    assert all(np.array_equal(r[k], full[k], equal_nan=True) for k in R.LABEL_KEYS)
    assert np.array_equal(ref["drawn"], R.label_codes(ctx.db.frame)[s.idx])
    with pytest.raises(ValueError):
        s.total[0, 0, 0] = 0.0                       # shared by every caller: read-only
    monkeypatch.setitem(runner.RESIDUALS, "resid_coupling", 1.0)                # a study that swaps the residuals
    other = runner.reference_simulation(ctx)
    assert other is not ref and not np.array_equal(other["sim"].total, s.total)
    monkeypatch.undo()
    assert runner.reference_simulation(ctx) is ref


@needs_panel
@check_values
def test_reference_simulation_check_values(ctx):
    from scenariolab.scenarios import runner
    ref = runner.reference_simulation(ctx)
    r = ref["regimes"]
    lab, contr = r["label"][:, 1:], r["contraction"][:, 1:]
    assert R.sim_anchor(r) == {"label_rt": "expansion", "label_expost_mix": {"expansion": 1.0}}
    year1 = {k: round(v, 4) for k, v in R.shares(lab[:, :12]).items()}
    assert year1 == {"expansion": 0.5441, "inflation": 0.3502, "deflation": 0.0019, "recession": 0.0345,
                     "depression": 0.0083, "stagflation": 0.061}
    assert {k: round(v, 4) for k, v in R.shares(ref["drawn"][:, :12]).items()} == {
        "expansion": 0.616, "inflation": 0.2498, "deflation": 0.0123, "recession": 0.019, "depression": 0.029,
        "stagflation": 0.074}
    assert round(float(contr[:, :12].any(axis=1).mean()), 4) == 0.1915
    assert round(float(contr[:, 24:].any(axis=1).mean()), 4) == 0.92
    assert round(float(contr[:, 12:48].any(axis=1).mean()), 4) == 0.469        # the baseline's second phase
    cpi = ref["sim"].macro["cpi_yoy"][:, 1:].astype(float)
    assert round(float((cpi[:, 24:] > 4).mean()), 4) == 0.4077
    lengths = [e - s + 1 for i in range(contr.shape[0]) for s, e in R.runs(contr[i])]
    assert round(len(lengths) / contr.shape[0], 3) == 2.116 and round(float(np.mean(lengths)), 2) == 11.92
