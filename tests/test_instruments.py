"""The instrument layer on synthetic cores (no data, no network): INSTRUMENTS_DESIGN §8 T5-T12, T14, T15, T19 and
the four defect fixes. The replays on history are in test_instruments_validation.py (they need data/)."""
import ast
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.instruments import ESTIMATED, STRUCTURAL, TICKERS
from scenariolab.instruments import data as D
from scenariolab.instruments import distributions as dist
from scenariolab.instruments import estimated as est
from scenariolab.instruments import fees, layer
from scenariolab.instruments import params as prm
from scenariolab.instruments import structural as st
from scenariolab.instruments import vol as V
from scenariolab.instruments.core import CoreView

PKG = Path(layer.__file__).resolve().parent
ASSETS = list(config.ASSET_IDS)


def synthetic_core(n=200, T=180, seed=0, freq="M", shocks=True, key="synthetic", **kw) -> CoreView:
    """Plausible monthly (or quarterly) core paths for the 13 assets, starting at the 2026-09 anchor."""
    rng = np.random.default_rng(seed)
    ppy = 12 if freq == "M" else 4
    mu = {"us_equity": 0.06, "nasdaq": 0.07, "energy": 0.05, "industrials": 0.06, "staples": 0.06, "healthcare": 0.06,
          "utilities": 0.05, "financials": 0.06, "real_estate": 0.05, "gold": 0.03, "crude_oil": 0.0}
    sd = {"us_equity": 0.15, "nasdaq": 0.22, "energy": 0.25, "industrials": 0.18, "staples": 0.13,
          "healthcare": 0.15, "utilities": 0.15, "financials": 0.2, "real_estate": 0.18, "gold": 0.15,
          "crude_oil": 0.3}
    inc = {"us_equity": 0.011, "nasdaq": 0.005, "energy": 0.03, "industrials": 0.015, "staples": 0.028,
           "healthcare": 0.015, "utilities": 0.03, "financials": 0.02, "real_estate": 0.035}
    z_mkt = rng.standard_normal((n, T)) if shocks else np.zeros((n, T))
    total = np.zeros((n, T, len(ASSETS)))
    income = np.zeros_like(total)
    y10 = np.full((n, T + 1), 5.29)
    ff = np.full((n, T + 1), 3.75)
    spread = np.full((n, T + 1), 1.47)
    if shocks:
        y10 = 5.29 + np.concatenate([np.zeros((n, 1)), np.cumsum(0.2 * rng.standard_normal((n, T)) / np.sqrt(ppy / 12),
                                                                    axis=1)], axis=1) * 0.3
        ff = np.clip(3.75 + (y10 - 5.29) * 0.8, 0, None)
        spread = np.clip(1.47 - np.concatenate([np.zeros((n, 1)), np.cumsum(0.05 * z_mkt, axis=1)], axis=1), 0.3, None)
    for k, a in enumerate(ASSETS):
        if a in mu:
            z = (0.8 * z_mkt + 0.6 * rng.standard_normal((n, T))) if shocks else np.zeros((n, T))
            price = mu[a] / ppy + sd[a] / np.sqrt(ppy) * z
            income[:, :, k] = inc.get(a, 0.0) / ppy
            total[:, :, k] = price + income[:, :, k]
        elif a == "treasury_10y":
            income[:, :, k] = y10[:, :-1] / 100 / ppy
            total[:, :, k] = income[:, :, k] - 8.0 * np.diff(y10, axis=1) / 100
        elif a == "cash":
            income[:, :, k] = total[:, :, k] = np.maximum(ff[:, :-1] + 0.19, 0) / 100 / ppy
    labels = [str(p) for p in pd.period_range("2026-09", periods=T + 1, freq="M" if freq == "M" else "Q")]
    if freq == "Q":
        labels = [str(pd.Period(p, "Q").asfreq("M", how="end")) for p in labels]
    cpi = 100 * np.exp(np.cumsum(np.full((n, T + 1), 0.025 / ppy), axis=1))
    return CoreView.from_arrays(freq, total, income, {"cpi": cpi, "y10": y10, "ff": ff, "spread": spread}, labels,
                                ASSETS, scenario_id=key, **kw)


@pytest.fixture(scope="module")
def core():
    return synthetic_core()


@pytest.fixture(scope="module")
def out(core):
    return layer.run_layer(core)


# ---------------------------------------------------------------- shape, conventions, T6

def test_every_ticker_comes_out_with_the_stored_shapes(core, out):
    assert list(out.paths) == list(TICKERS) and len(TICKERS) == 16
    for t, p in out.paths.items():
        assert p.total.shape == (core.n, core.T) and p.income.shape == (core.n, core.T), t
        assert p.level.shape == (core.n, core.T + 1), t
        assert np.isfinite(p.total).all() and (p.total >= -1.0).all(), t
        assert (p.income >= 0).all(), t
        assert (p.level >= 0).all(), t


def test_payout_rules_only_split_the_total_return(core, out):
    """T6: price growth + cash = total growth every period; changing the payout rule leaves totals unchanged."""
    for t, p in out.paths.items():
        lv = p.level
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = (lv[:, 1:] + p.income * lv[:, :-1]) / lv[:, :-1] - 1.0
        ok = lv[:, :-1] > 0
        assert np.allclose(implied[ok], p.total[ok], atol=1e-9), t
    alt = layer.run_layer(core, ["QQQI"], params={"QQQI": {"dist_rule": "premium_plus_dividend"}})
    assert np.array_equal(alt.paths["QQQI"].total, out.paths["QQQI"].total)
    assert not np.allclose(alt.paths["QQQI"].income, out.paths["QQQI"].income)
    d = prm.estimated()["tickers"]["SCHD"]["dist"]
    lr = np.log1p(out.paths["SCHD"].total).reshape(core.n, -1, 3).sum(axis=2)
    a = dist.sticky_quarterly(lr, {**d, "gamma": -0.5}, prm.packaged_anchors()["estimated"]["SCHD"],
                              np.zeros_like(lr), np.full(core.n, 0.03))
    b = dist.sticky_quarterly(lr, d, prm.packaged_anchors()["estimated"]["SCHD"], np.zeros_like(lr),
                              np.full(core.n, 0.03))
    gross_a = (a["price_q"][:, 1:] + a["dps_q"]) / a["price_q"][:, :-1]
    gross_b = (b["price_q"][:, 1:] + b["dps_q"]) / b["price_q"][:, :-1]
    assert np.allclose(gross_a, np.exp(lr)) and np.allclose(gross_b, np.exp(lr))


def test_year_factors_follow_the_withdrawal_convention(out):
    tot, price, inc = out.year_factors(["QQQ", "MAIN"])
    assert tot.shape == (out.core.n, 15, 2)
    g = np.prod(1 + out.paths["QQQ"].total[:, :12], axis=1)
    assert np.allclose(tot[:, 0, 0], g)
    assert (inc >= 0).all() and (price > 0).all()


def test_attribution_parts_add_up_to_the_log_return(out):
    for t, p in out.paths.items():
        s = sum(p.parts.values())
        lr = np.log1p(np.maximum(p.total, -0.999999))
        ok = np.isfinite(s)
        assert np.allclose(s[ok], lr[ok], atol=1e-9), t


# ---------------------------------------------------------------- leveraged funds (T7)

def test_leverage_identities():
    x = np.array([[0.02, -0.03, 0.01]])
    rv = np.array([[0.002, 0.004, 0.001]])
    assert np.allclose(st.letf_log_return(x, rv, 0 * x, 1.0, 0.0), x)          # L = 1, no carry: the index
    assert np.allclose(st.letf_log_return(x, rv, rv * 0.01, 1.0, 0.0, rv * rv), x)   # every higher term vanishes
    carry = np.full_like(x, 0.001)
    assert np.allclose(st.letf_log_return(x, 0 * rv, 0 * x, 3.0, carry), 3 * x + np.log1p(carry))   # RV = 0
    grid = np.linspace(0.0, 0.02, 9)
    vals = [float(st.letf_log_return(0.01, r, 0.0, 3.0, 0.0)) for r in grid]
    assert all(b < a for a, b in zip(vals, vals[1:]))                            # decreasing in RV
    assert st.breakeven_total_return(3, 0.24, 0.049, 0.0078, 0.0018) == pytest.approx(0.049 + 4 * 0.0576 / 2 + 0.003)


def test_fourth_order_formula_tracks_exact_daily_compounding():
    """Review finding 5: on synthetic daily paths (t5 days, vol around 22%) the period formula with the true
    RS3 and RS4 tracks exact daily-reset compounding to < 1 bp a month on average (the third-order formula was
    biased up by 0.4 / 3.2 bp a month for L = 2 / 3), and the financing leg accrues ACT/360."""
    rng = np.random.default_rng(12345)
    n, days = 20000, 21
    sd = np.exp(rng.normal(np.log(0.22), 0.45, n)) / np.sqrt(252)
    x = np.maximum(0.0003 + sd[:, None] * rng.standard_t(5, (n, days)) / np.sqrt(5 / 3), -0.25)
    for L in (2.0, 3.0):
        exact = np.log1p(L * np.expm1(x)).sum(axis=1)
        X, RV, R3, R4 = x.sum(1), (x * x).sum(1), (x ** 3).sum(1), (x ** 4).sum(1)
        third = st.letf_log_return(X, RV, R3, L, 0.0)
        fourth = st.letf_log_return(X, RV, R3, L, 0.0, R4)
        assert abs((fourth - exact).mean()) < 1e-4 and abs((fourth - exact).mean()) < abs((third - exact).mean()), L
    assert st.c4(1.0) == 0.0 and st.c4(2.0) == pytest.approx(-1.0833, abs=1e-4) and st.c4(3.0) == pytest.approx(-9.25)
    c = st.letf_carry(3.0, 0.0, 0.036, 0.0, 0.0, 12)
    assert c == pytest.approx(-2 * 0.036 * 365 / 360 / 12)
    v = prm.structural()["vol"]
    assert st.expected_rs4(0.004, "M") == pytest.approx(v["rs4_M"]["k"] * 0.004 ** 2 / v["days_M"])


def test_ruin_draws_match_the_tail_probability(core):
    """T7: the share of wiped-out periods matches ruin_probability; a fund never goes below zero."""
    hot = V.VolState(freq="M", source="fixed", rv_ndx=np.full((20000, 12), 0.8 ** 2 / 12),
                     rv_mkt=np.full((20000, 12), 0.5 ** 2 / 12), sigma_atm=np.full((20000, 12), 0.6),
                     dd_ndx=np.zeros((20000, 12)), dd_mkt=np.zeros((20000, 12)))
    c = synthetic_core(20000, 12, seed=3)
    p = st.leveraged(c, hot, {}, 77.96, "TQQQ")
    pr = float(st.ruin_probability(0.8 ** 2 / 12, 3, "M"))
    dead = ~p.alive[:, -1]
    expect = 1 - (1 - pr) ** 12
    assert abs(dead.mean() - expect) < 4 * np.sqrt(expect * (1 - expect) / 20000)
    assert (p.level >= 0).all() and (p.total >= -1).all()
    assert (p.total[~p.alive] <= 0).all()


def test_leveraged_funds_use_the_fee_table_and_the_spread_options(core, out):
    a = out.paths["TQQQ"].meta
    assert a["fee"] == fees.fee("TQQQ") and a["s_eff"] == prm.structural()["letf"]["TQQQ"]["s_eff"]["mean_2019_2026"]
    hi = layer.run_layer(core, ["TQQQ"], params={"TQQQ": {"s_eff": "latest", "ruin": False}})
    lo = layer.run_layer(core, ["TQQQ"], params={"TQQQ": {"ruin": False}})
    assert (np.log1p(hi.paths["TQQQ"].total) < np.log1p(lo.paths["TQQQ"].total)).all()


# ---------------------------------------------------------------- covered calls (T8)

def test_covered_call_limits():
    R = np.array([0.95, 1.0, 1.02, 1.08])
    tot0, _ = st.overlay_month(R, 0.006, 0.04, 0.2, ((1.019, 0.0), (1.034, 0.0)), 0.0068)
    assert np.allclose(tot0, R - 1 + 0.006 / 12 - 0.0068 / 12)                   # no coverage: index + q - fee
    tot1, parts = st.overlay_month(R, 0.0, 0.0, 1e-6, ((1.02, 1.0),), 0.0)
    assert np.allclose(tot1, np.minimum(R, 1.02) - 1)                            # sigma -> 0, k > 1: capped
    c = synthetic_core(50, 24, seed=5)
    v = V.vol_state(c)
    q = st.qqqi(c, v, {}, 54.94)
    assert np.allclose(q.income, prm.structural()["qqqi"]["distribution_rate_annual"] / 12)   # NAV rate


def test_quarterly_covered_call_matches_the_monthly_overlay_on_average():
    """T10: the quarterly bridge split, compounded, against the monthly recipe on the same paths (tolerance)."""
    m = synthetic_core(4000, 12, seed=7)
    qv = m.quarterly()
    vm, vq = V.vol_state(m, noise=False), V.vol_state(qv, noise=False)
    am = np.log1p(st.qqqi(m, vm, {}, 1.0).total).reshape(4000, 4, 3).sum(axis=2)
    aq = np.log1p(st.qqqi(qv, vq, {}, 1.0).total)
    assert abs(am.mean() - aq.mean()) * 4 < 0.01                                 # < 1%/yr on average


def _quarterly_vol(vm, n):
    """The monthly volatility state seen quarterly: RV summed, IV at each quarter's start, drawdowns at its end."""
    return V.VolState(freq="Q", source="model", rv_ndx=vm.rv_ndx.reshape(n, -1, 3).sum(axis=2),
                      rv_mkt=vm.rv_mkt.reshape(n, -1, 3).sum(axis=2), sigma_atm=vm.sigma_atm[:, ::3],
                      dd_ndx=vm.dd_ndx[:, 2::3], dd_mkt=vm.dd_mkt[:, 2::3])


def test_quarterly_core_keeps_qqqi_and_the_leveraged_funds_within_t3():
    """Review finding 4: the same paths seen monthly and quarterly give QQQI and the leveraged funds the same mean
    annual log return within T3's tolerances (QQQI 0.5, QLD 0.5, TQQQ 1.0 %/yr; the ruin draw off). The bridge's
    month variance is rho x RV_Q / 3 with rho the monthly return variance per unit of RV (0.85): with RV_Q / 3 the
    bridged months were ~18% too volatile and QQQI about 1%/yr lower on a quarterly core."""
    n = 3000
    m = synthetic_core(n, 120, seed=31)
    qv = m.quarterly()
    vm = V.vol_state(m)
    vq = _quarterly_vol(vm, n)
    ann = lambda ip: float(np.log1p(ip.total).sum(axis=1).mean() / 10)          # noqa: E731
    q_m, q_q = ann(st.qqqi(m, vm, {}, 1.0)), ann(st.qqqi(qv, vq, {}, 1.0))
    assert abs(q_m - q_q) < 0.005, (q_m, q_q)
    assert 0.8 < st.bridge_var_ratio() < 0.9
    old = ann(st.qqqi(qv, replace(vq, rv_ndx=vq.rv_ndx / st.bridge_var_ratio()), {}, 1.0))
    assert q_q - old > 0.003                                     # the unscaled bridge was the gap
    for t, tol in (("QLD", 0.005), ("TQQQ", 0.010)):
        lm = ann(st.leveraged(m, vm, {"ruin": False}, 1.0, t))
        lq = ann(st.leveraged(qv, vq, {"ruin": False}, 1.0, t))
        assert abs(lm - lq) < tol, (t, lm, lq)
    for t, tol in (("QLD", 0.005), ("TQQQ", 0.010), ("QQQI", 0.005)):          # each core's own vol model
        f = (lambda c, v: st.qqqi(c, v, {}, 1.0)) if t == "QQQI" else (lambda c, v, t=t: st.leveraged(c, v, {"ruin": False}, 1.0, t))
        a, b = ann(f(m, vm)), ann(f(qv, V.vol_state(qv)))
        assert abs(a - b) < 2 * tol, (t, a, b)


# ---------------------------------------------------------------- TLT (T9)

def test_tlt_duration_convexity_and_carry():
    S = prm.structural()["tlt"]
    y = np.linspace(0.0, 0.10, 21)
    D_, C_ = np.polyval(S["D_of_y"], y), np.polyval(S["C_of_y"], y)
    assert (D_ > 0).all() and (C_ > 0).all() and (np.diff(D_) < 0).all() and (np.diff(C_) < 0).all()
    path = np.full((1, 13), 0.05)
    r = st.tlt_returns(path, 12, fees.fee("TLT"), 0.857, 7.0)
    assert np.allclose(r["total"], 0.05 / 12 - fees.fee("TLT") / 12)              # dy = 0: carry less the fee


def test_tlt_quarterly_steps_monthly():
    q = synthetic_core(30, 8, seed=11, freq="Q")
    p = st.tlt(q, V.vol_state(q), {"noise": False}, 76.0, prm.packaged_anchors()["rates"])
    assert p.total.shape == (30, 8) and np.isfinite(p.total).all()


# ---------------------------------------------------------------- estimated: defect 1 (T5), T10

def test_residual_draw_is_mean_zero_given_the_vol():
    """T5: the recentered draw has mean |.| < 0.1%/yr overall and |E[eps | v]| < 0.3%/yr in 5 vol buckets."""
    pool = est.pool_dates()
    rng = np.random.default_rng(1)
    n, Q = 4000, 40
    v = np.exp(rng.normal(np.log(0.14), 0.45, (n, Q)))
    idx = est.draw_dates(n, Q, len(pool), rng)
    for t in ESTIMATED:
        r = est.residual_draw(t, pool, idx, v, "fixed")
        eps = r["eps"]
        # exact: the conditional mean over the pool is 0 for every v
        sc = est.scaled_residuals(np.array([prm.estimated()["tickers"][t]["resid"][d] for d in pool]) -
                                  np.mean([prm.estimated()["tickers"][t]["resid"][d] for d in pool]),
                                  np.array([prm.estimated()["tickers"][t]["rv_at"][d] for d in pool]),
                                  prm.estimated()["tickers"][t]["infl"], v[:50, :5])
        assert np.abs(sc.mean(axis=2) - sc.mean(axis=2)).max() == 0
        assert abs(4 * eps.mean()) < 0.001 + 4 * 3 * eps.std() / np.sqrt(eps.size), t
        qs = np.quantile(v, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (v >= lo) & (v <= hi)
            assert abs(4 * eps[m].mean()) < 0.003 + 4 * 3 * eps[m].std() / np.sqrt(m.sum()), (t, lo)


def test_the_old_draw_was_biased_and_the_fix_removes_it():
    """Defect 1: the raw residuals have mean alpha_hat - alpha (BRK.B +5.1%/yr in the design) and the vol
    scaling adds a conditional mean; the fixed draw has neither."""
    pool = est.pool_dates()
    rng = np.random.default_rng(2)
    v = np.full((3000, 40), 0.14)
    idx = est.draw_dates(3000, 40, len(pool), rng)
    asis = est.residual_draw("BRK.B", pool, idx, v, "asis")["eps"]
    fixed = est.residual_draw("BRK.B", pool, idx, v, "fixed")["eps"]
    assert 4 * asis.mean() > 0.01 and abs(4 * fixed.mean()) < 0.002


def test_monthly_estimated_paths_sum_to_the_quarterly_recipe():
    """T10: on the same residual draws and vol, the monthly path's quarterly sums equal the quarterly recipe."""
    m = synthetic_core(60, 36, seed=13)
    q = m.quarterly()
    vq = V.vol_state(m).mkt_vol_quarterly()
    a = est.simulate(m, vq, ESTIMATED)
    b = est.simulate(q, vq, ESTIMATED)
    for t in ESTIMATED:
        lm = np.log1p(a[t].total).reshape(60, 12, 3).sum(axis=2)
        assert np.allclose(lm, np.log1p(b[t].total), atol=1e-10), t
        assert a[t].dps.shape == (60, 12) and "bridge_month" in a[t].flags and "bridge_month" not in b[t].flags


def test_monthly_payers_pay_every_month(out):
    main, arcc = out.paths["MAIN"].income, out.paths["ARCC"].income
    assert (main[:, :3] > 0).all()
    assert (arcc[:, 0] == 0).all() and (arcc[:, 2] > 0).all()


def test_alpha_variants_differ_by_the_alpha_only(core):
    a = est.simulate(core, V.vol_state(core).mkt_vol_quarterly(), ["MAIN"], alpha_mode="shrunk")
    b = est.simulate(core, V.vol_state(core).mkt_vol_quarterly(), ["MAIN"], alpha_mode="half")
    r = prm.estimated()["tickers"]["MAIN"]["alpha_variants_q"]
    d = np.log1p(a["MAIN"].total) - np.log1p(b["MAIN"].total)
    assert np.allclose(d, (r["shrunk"] - r["half"]) / 3, atol=1e-12)


# ---------------------------------------------------------------- defect 4: c_LR from the running core

STICKY = ("SCHD", "MO", "MAIN", "ARCC", "CSWC", "EPD", "MPLX", "BME")


def _no_shock_core(drift: float):
    c = synthetic_core(2, 480, seed=0, shocks=False)
    tot = c.total.copy()
    tot[:, :, :11] += drift / 12                                              # every risky asset's drift moves
    return replace(c, total=tot)


def test_yield_stays_at_normal_on_paths_without_shocks():
    """Defect 4: with c_LR from the running core, a yield that starts at its normal level is still there
    (+-0.5pp) after 40 years whatever the core's drift (-2%, 0, +3% a year on every risky asset)."""
    for drift in (0.0, -0.02, 0.03):
        c = _no_shock_core(drift)
        for t in STICKY:
            d = prm.estimated()["tickers"][t]["dist"]
            st0 = dict(prm.packaged_anchors()["estimated"][t])
            st0["D_last_q"] = d["normal_yield"] * st0["price"] / 4
            r = est.simulate(c, np.full((2, 160), 0.14), [t], {}, {t: st0}, alpha_uncertainty=False,
                             residual_mode="none")[t]
            y = 4 * r.dps / r.level[:, :-1:3]
            gap = np.abs(y - d["normal_yield"])
            if t == "MPLX":     # slow error correction (|gamma| - y = 1.5%/yr): converging, not drifting
                assert (gap[:, -1] < gap[:, -41]).all() and (gap[:, -1] < 0.02).all(), drift
                fast = est.simulate(c, np.full((2, 160), 0.14), [t], {t: {"dist": {"gamma": -0.20}}},
                                    {t: st0}, alpha_uncertainty=False, residual_mode="none")[t]
                assert (np.abs(4 * fast.dps[:, -1] / fast.level[:, -4] - d["normal_yield"]) < 0.005).all()
                continue
            assert gap[:, -1].max() < 0.005, (t, drift)


def test_yield_returns_toward_normal_from_todays_start():
    """From today's yields the gap to the normal yield shrinks within 40 years and ends inside +-0.5pp, except
    MPLX: its error correction (gamma -0.10) is barely faster than its 8.5% normal yield (the yield's own rate is
    |gamma| - y, a half-life of about 47 years), so it closes only part of its 1.4pp gap (reported)."""
    for drift in (0.0, -0.02, 0.03):
        c = _no_shock_core(drift)
        for t in STICKY:
            d = prm.estimated()["tickers"][t]["dist"]
            st0 = prm.packaged_anchors()["estimated"][t]
            r = est.simulate(c, np.full((2, 160), 0.14), [t], alpha_uncertainty=False, residual_mode="none")[t]
            gap0 = abs(st0["yield_ttm"] - d["normal_yield"])
            gap40 = float(np.abs(4 * r.dps[:, -1] / r.level[:, -4] - d["normal_yield"]).max())
            assert gap40 < max(gap0, 0.003), (t, drift)
            if t != "MPLX":
                assert gap40 < 0.005, (t, drift)


@pytest.fixture(scope="module")
def shocked_40y():
    """400 shocked synthetic paths x 40 years and the sticky tickers on them with the default draws."""
    c = synthetic_core(400, 480, seed=17)
    vq = V.vol_state(c).mkt_vol_quarterly()
    return c, est.simulate(c, vq, STICKY)


def _yearly_yield(ip) -> np.ndarray:
    lv = ip.level
    cash = ip.income * lv[:, :-1]
    Y = cash.shape[1] // 12
    return cash[:, :Y * 12].reshape(-1, Y, 12).sum(axis=2) / lv[:, 0:Y * 12:12]


def test_payouts_never_drain_a_holding_on_shocked_cores(shocked_40y):
    """Review finding 1: with shocks, the sticky rule must not pay a holding's whole value out (MPLX, MO and EPD
    did on 5-16% of the scenario paths): on 40 shocked years the share of paths with a year whose cash exceeds half
    of the start-of-year value, or a quarter where the 2% price floor capped the payout, stays under 1%, and every
    number is finite (no 0 / 0 after a price that underflowed)."""
    _, e = shocked_40y
    for t in STICKY:
        ip = e[t]
        assert np.isfinite(ip.total).all() and np.isfinite(ip.income).all() and np.isfinite(ip.level).all(), t
        assert (ip.level > 0).all(), t
        drained = (_yearly_yield(ip) > 0.5).any(axis=1)
        assert drained.mean() < 0.01, (t, drained.mean())
        assert ip.meta["floor_share_paths"] < 0.01, t


def test_yield_distribution_stays_stationary_over_40_years(shocked_40y):
    """Defect 4's aim on shocked cores: the yield over the normal yield keeps roughly the same distribution at
    years 10, 20 and 40 (median 0.6-1.4 x normal, 5-95% band inside 0.2-2.5 x, the median moving < 0.25)."""
    _, e = shocked_40y
    for t in STICKY:
        yn = prm.estimated()["tickers"][t]["dist"]["normal_yield"]
        yy = _yearly_yield(e[t]) / yn
        med = {y: float(np.median(yy[:, y - 1])) for y in (10, 20, 40)}
        for y in (10, 20, 40):
            lo, hi = np.percentile(yy[:, y - 1], [5, 95])
            assert 0.6 < med[y] < 1.4 and lo > 0.2 and hi < 2.5, (t, y, med[y], lo, hi)
        assert abs(med[40] - med[10]) < 0.25, (t, med)


def test_the_explicit_cut_resets_the_payout_to_the_normal_yield():
    """A payout that would put the yield on the smoothed price above cut_multiple x normal is cut back to the
    normal yield on that price; without the cut (None) the same quarter keeps its sticky payout."""
    d = dict(prm.estimated()["tickers"]["MPLX"]["dist"], cut_multiple=2.0)
    st0 = {"price": 60.0, "D_last_q": 60.0 * d["normal_yield"] / 4}
    lr = np.array([[-0.7, -0.7, 0.0, 0.0, 0.0, 0.0]])
    tb3 = np.zeros_like(lr)
    a = dist.sticky_quarterly(lr, d, st0, tb3, np.zeros(1))
    b = dist.sticky_quarterly(lr, {**d, "cut_multiple": None}, st0, tb3, np.zeros(1))
    assert a["cut_q"].any() and not b["cut_q"].any()
    q = int(np.argmax(a["cut_q"][0]))
    anchor = a["price_q"][0, max(0, q + 1 - d["anchor_quarters"]):q + 1].mean()
    assert a["dps_q"][0, q] == pytest.approx(d["normal_yield"] * anchor / 4)
    assert a["dps_q"][0, q] < b["dps_q"][0, q]
    assert np.allclose((a["price_q"][:, 1:] + a["dps_q"]) / a["price_q"][:, :-1], np.exp(lr))     # T6


def test_c_lr_includes_each_rows_own_alpha_draw():
    """c_LR is the scenario's systematic mean plus the row's own alpha: a row drawn 1%/q above another gets a long-
    run payout growth 4%/yr higher, so the yield does not drift with the alpha draw."""
    x = np.full((3, 12), 0.01)
    c = dist.c_lr_running(x, np.array([0.01, 0.02, 0.0]), 0.05, 4, np.array(["a", "a", "a"]))
    assert c == pytest.approx([0.04 + 0.04 - 0.05, 0.04 + 0.08 - 0.05, 0.04 - 0.05])
    core = synthetic_core(60, 36, seed=8)
    r = est.simulate(core, np.full((60, 12), 0.14), ["MPLX"])["MPLX"]
    assert r.meta["c_lr"] == pytest.approx(float(np.mean(c_lr_rows(core, "MPLX"))), abs=1e-12)


def c_lr_rows(core, t):
    r = prm.estimated()["tickers"][t]
    from scenariolab.instruments.streams import row_normals
    a = r["alpha_variants_q"]["shrunk"] + r["alpha_sd_q"] * row_normals(core.row_namespace(), core.groups(),
                                                                         core.indices(), "alpha", t)
    gz, _ = core.gz_path()
    x = sum(est.systematic(core, r, gz).values())
    return dist.c_lr_running(x, a, r["dist"]["normal_yield"], core.ppy, core.groups())


def test_frozen_c_lr_drifts_where_the_running_one_does_not():
    c = synthetic_core(2, 480, seed=0, shocks=False)
    vq = np.full((2, 160), 0.14)
    d = prm.estimated()["tickers"]["SCHD"]["dist"]
    run = est.simulate(c, vq, ["SCHD"], alpha_uncertainty=False, residual_mode="none")["SCHD"]
    frz = est.simulate(c, vq, ["SCHD"], alpha_uncertainty=False, residual_mode="none", c_lr_mode="frozen")["SCHD"]
    y = lambda r: 4 * r.dps[0, -1] / r.level[0, -4]                               # noqa: E731
    assert abs(y(run) - d["normal_yield"]) < 0.005 and abs(y(frz) - d["normal_yield"]) > abs(y(run) - d["normal_yield"])


# ---------------------------------------------------------------- defect 2: the GZ bridge

def test_monthly_bridge_sums_to_the_quarterly_bridge(core):
    gm, src = core.gz_path()
    gq, _ = core.quarterly().gz_path()
    assert src.startswith("bridge:spread_eq")
    assert np.allclose(gm.reshape(core.n, -1, 3).sum(axis=2), gq, atol=1e-12)


def test_bridge_is_the_fitted_formula_for_every_row(core):
    """Review finding 2: the default bridge is D15's formula as fitted (0.019 + 1.003 d(Baa - 10y) - 0.771 r_eq a
    quarter), the same map for every row: a row's d_gz never depends on the other rows of its view."""
    q = core.quarterly()
    mdl = prm.bridges()["d_gz"]["models"]["spread_eq"]
    a, b, c_ = mdl["coef"]
    g, _ = q.gz_path()
    assert np.allclose(g, a + b * q.dspread() + c_ * q.lr("us_equity"), atol=1e-12)
    assert (a, b, c_) == pytest.approx((0.019, 1.003, -0.771), abs=6e-4)
    sub, _ = core.rows(0, 7).gz_path()
    full, _ = core.gz_path()
    assert np.array_equal(sub, full[:7])
    assert np.corrcoef(g.ravel(), q.lr("us_equity").ravel())[0, 1] < -0.3


def test_lower_equity_returns_widen_gz_at_least_as_much(core):
    """A scenario whose equity returns are uniformly lower gets at least as much GZ widening as the baseline (the
    withdrawn per-view centring gave ai_bust less widening than the baseline), and a node where every particle's
    market falls 20% in one quarter gets a positive d_gz."""
    j = core.assets.index("us_equity")
    tot = core.total.copy()
    tot[:, :, j] -= 0.03 / 12
    low = replace(core, total=tot)
    g0, _ = core.gz_path()
    g1, _ = low.gz_path()
    assert g1.sum(axis=1).mean() > g0.sum(axis=1).mean()
    assert (g1 >= g0 - 1e-12).all()
    node = core.rows(0, 50).quarterly()
    t = node.total.copy()
    t[:, 0, j] = 0.8 - 1.0 + node.income[:, 0, j]
    crash = replace(node, total=t, spread=np.full_like(node.spread, 1.47))       # the market term alone
    g, _ = crash.gz_path()
    a, _b, c_ = prm.bridges()["d_gz"]["models"]["spread_eq"]["coef"]
    assert np.allclose(g[:, 0], a + c_ * np.log1p(t[:, 0, j])) and (g[:, 0] > 0.15).all()


def test_core_supplied_gz_is_used_and_drops_the_credit_link_chip(core):
    c = replace(core, d_gz=np.zeros((core.n, core.T)))
    o = layer.run_layer(c, ["MAIN", "XLG"])
    assert "credit_link" not in o.paths["MAIN"].flags
    assert "credit_link" in layer.run_layer(core, ["MAIN"]).paths["MAIN"].flags


def test_diagnostics_flag_the_credit_link(out):
    d = out.diagnostics()
    assert set(d) >= {"corr_dspread_us_equity", "corr_dgz_us_equity", "ticker_corr_us_equity", "warnings"}
    assert d["ticker_corr_us_equity"]["MAIN"]["hist"] == pytest.approx(0.748, abs=0.01)


def test_credit_link_chip_only_where_the_gap_exceeds_the_tolerance():
    """Review finding 7 / T18: the bad chip only for a ticker whose simulated correlation is more than 0.15 below
    history in some scenario; within the tolerance everywhere it becomes the neutral credit_link_ok."""
    epd = layer.credit_link_status({"baseline": 0.52, "ai_bust": 0.579}, 0.555)
    main = layer.credit_link_status({"baseline": 0.55, "ai_bust": 0.43}, 0.748)
    assert not epd["bad"] and epd["n_over"] == 0 and (epd["sim_lo"], epd["sim_hi"]) == (0.52, 0.579)
    assert main["bad"] and main["n_over"] == 2 and main["over"] == ["ai_bust", "baseline"]
    flags = ["estimated", "credit_link", "tax"]
    assert layer.resolve_credit_link(flags, epd) == ["estimated", "credit_link_ok", "tax"]
    assert layer.resolve_credit_link(flags, main) == flags
    assert layer.credit_link_status({}, 0.6)["bad"] and layer.credit_link_status({"a": 0.1}, None)["bad"]
    chips = prm.catalog()["chips"]
    assert "0.3–0.4" not in chips["credit_link"] and chips["credit_link_ok"]
    k = prm.estimated()["payout_cut"]["multiple"]
    assert f"{k:g} 倍" in chips["payout_cut"]


# ---------------------------------------------------------------- volatility state (T11 part)

def test_vol_state_shock_correlation_and_mean_path():
    c = synthetic_core(3000, 24, seed=21)
    v = V.vol_state(c)
    m = V.model_paths(c.x("nasdaq"), c.x("us_equity"), c.dspread(), "M", noise=False)
    S, J = prm.structural()["vol"]["rv_M"], prm.vol_joint()["monthly"]
    # the shock in period 0 (same lags for every path) is the deviation from the mean recursion's first step
    a = prm.packaged_anchors()["vol"]
    x0n, x0m, d0 = c.x("nasdaq")[:, 0], c.x("us_equity")[:, 0], c.dspread()[:, 0]
    bn, bm = np.asarray(S["coef"]), np.asarray(J["mkt"]["coef"])
    mn = bn[0] + bn[1] * a["lrv_ndx"] + bn[2] * abs(x0n) + bn[3] * np.minimum(x0n, 0) + bn[4] * m["dd_ndx"][:, 0] + bn[5] * d0
    mm = bm[0] + bm[1] * a["lrv_mkt"] + bm[2] * abs(x0m) + bm[3] * np.minimum(x0m, 0) + bm[4] * m["dd_mkt"][:, 0] + bm[5] * d0
    rho = np.corrcoef(np.log(v.rv_ndx[:, 0]) - mn, np.log(v.rv_mkt[:, 0]) - mm)[0, 1]
    assert rho == pytest.approx(J["rho"], abs=0.03)
    assert (v.sigma_atm[:, 0] == pytest.approx(prm.structural()["vol"]["iv_atm_multiplier_on_vxn"] * a["vxn"]))
    assert v.mkt_vol_quarterly().shape == (3000, 8)


def test_fixed_vol_is_a_flagged_sensitivity(core):
    o = layer.run_layer(core, ["TQQQ"], vol_source="fixed", fixed_vol=0.25)
    assert "fixed_vol" in o.paths["TQQQ"].flags
    assert np.allclose(o.vol.rv_ndx, 0.25 ** 2 / 12)
    with pytest.raises(ValueError):
        layer.run_layer(core, ["TQQQ"], vol_source="drawn")                      # needs the drawn months


# ---------------------------------------------------------------- reproducibility and isolation (T19)

def test_same_seed_same_numbers_and_ticker_isolation(core, out):
    again = layer.run_layer(core)
    for t in TICKERS:
        assert np.array_equal(again.paths[t].total, out.paths[t].total), t
    tweak = layer.run_layer(core, params={"TQQQ": {"s_eff": 0.03}, "QQQI": {"coverage": 0.4},
                                          "MAIN": {"alpha_mode": "raw"}})
    for t in TICKERS:
        same = np.array_equal(tweak.paths[t].total, out.paths[t].total)
        assert same == (t not in ("TQQQ", "QQQI", "MAIN")), t
    sub = layer.run_layer(core, ["ARCC", "QLD"])
    assert np.array_equal(sub.paths["ARCC"].total, out.paths["ARCC"].total)
    assert np.array_equal(sub.paths["QLD"].total, out.paths["QLD"].total)


def test_alpha_is_fixed_per_row_across_the_seam():
    """A row keeps its alpha draw in its scenario and in the continuation chunk that follows it."""
    scen = synthetic_core(50, 36, seed=1, key="baseline")
    ids = ["baseline"]
    g, i = layer.rows_for_chunk(ids, 50, 10, 20)
    assert list(i) == list(range(10, 30)) and set(g) == {"baseline"}
    cont = replace(synthetic_core(20, 36, seed=99, key="chunk0", set_id="continuation"), row_group=g, row_index=i,
                   row_set_id="builtin")
    vq_s = np.full((50, 12), 0.14)
    vq_c = np.full((20, 12), 0.14)
    a = est.simulate(scen, vq_s, ["MAIN"])["MAIN"]
    b = est.simulate(cont, vq_c, ["MAIN"])["MAIN"]
    alpha_a = a.parts["alpha"][10:30, 0]
    alpha_b = b.parts["alpha"][:, 0]
    assert np.allclose(alpha_a, alpha_b)


def test_quarterly_core_runs_every_recipe():
    q = synthetic_core(40, 20, seed=4, freq="Q")
    o = layer.run_layer(q)
    for t in TICKERS:
        assert o.paths[t].total.shape == (40, 20), t
    assert "bridge_month" in o.paths["QQQI"].flags


def test_from_quarterly_maps_the_schema_fields():
    n, Q = 5, 8
    rng = np.random.default_rng(0)
    draws = {"y_assets": rng.normal(0.015, 0.05, (n, Q, 13)), "s_ff": 3.75, "d_ff": np.zeros((n, Q)),
             "s_y10": 5.29, "d_y10": np.zeros((n, Q)), "s_spread": 1.47, "d_spread": np.zeros((n, Q)),
             "d_gz": np.zeros((n, Q)), "infl_q": np.full((n, Q), 0.006)}
    c = CoreView.from_quarterly(draws, labels=[str(p) for p in pd.period_range("2026-09", periods=Q + 1, freq="3M")])
    assert c.freq == "Q" and c.y10.shape == (n, Q + 1) and np.allclose(c.lr("us_equity"), draws["y_assets"][:, :, 0])
    o = layer.run_layer(c, ["QQQ", "MAIN", "TLT"])
    assert "credit_link" not in o.paths["MAIN"].flags


def test_unknown_ticker_is_refused(core):
    with pytest.raises(KeyError, match="未知标的"):
        layer.run_layer(core, ["QQQ", "NOPE"])


# ---------------------------------------------------------------- the complete-month rule (T12)

def _daily(start, end):
    return pd.Series(1.0, index=pd.bdate_range(start, end))


def test_complete_month_rule_needs_every_source_on_the_last_trading_day():
    ndx = _daily("2026-08-03", "2026-10-29")             # FRED NDX stops on Thursday 10-29
    nav = _daily("2026-08-03", "2026-10-30")             # ProShares has Friday 10-30
    assert D.last_trading_day("2026-10") == pd.Timestamp("2026-10-30")
    months = D.complete_months(ndx, nav)
    assert pd.Period("2026-09", "M") in months and pd.Period("2026-10", "M") not in months
    assert pd.Period("2026-10", "M") in D.complete_months(_daily("2026-08-03", "2026-10-30"), nav)
    # month-end holidays: Good Friday 2024-03-29, Memorial Day 2021-05-31
    assert D.last_trading_day("2024-03") == pd.Timestamp("2024-03-28")
    assert D.last_trading_day("2021-05") == pd.Timestamp("2021-05-28")
    assert D.month_complete(_daily("2024-03-01", "2024-03-28"), "2024-03")


def test_letf_october_returns_differ_by_vintage():
    """T12 fixture: the same fund's 2026-10 return to 10-29 and to 10-30 (QLD -7.2% / -13.7% in the data)."""
    nav = pd.Series([95.6425, 88.7331, 82.5643], index=pd.to_datetime(["2026-09-30", "2026-10-29", "2026-10-30"]))
    assert nav.loc["2026-10-29"] / nav.loc["2026-09-30"] - 1 == pytest.approx(-0.0722, abs=1e-4)
    assert nav.iloc[-1] / nav.iloc[0] - 1 == pytest.approx(-0.1367, abs=1e-4)
    ndx = pd.Series(1.0, index=pd.bdate_range("2026-09-01", "2026-10-29"))
    assert pd.Period("2026-10", "M") not in D.complete_months(nav.reindex(pd.bdate_range("2026-09-01", "2026-10-30"))
                                                             .ffill().bfill(), ndx)


# ---------------------------------------------------------------- the fee table (T14)

def test_fee_table_schedules():
    assert fees.fee("QQQ") == 0.0018 and fees.fee("QQQ", "2025-12-19") == 0.0020
    assert fees.fee("TQQQ", "2021-05-31") == 0.0095 and fees.fee("TQQQ", "2022-03-01") == 0.0086
    assert fees.fee("QLD") == 0.0089 and fees.fee("QLD", "2025-05-31") == 0.0095
    assert list(fees.fee_path("TQQQ", pd.period_range("2025-05", "2025-06", freq="M"))) == [0.0082, 0.0078]


def test_no_fee_literal_outside_the_fee_table():
    """T14: no number of the fee table appears as a literal in the package's code."""
    vals = set()
    for f in fees.table()["funds"].values():
        for e in f["schedule"]:
            vals.add(round(e["fee"], 6))
    bad = []
    for p in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and round(node.value, 6) in vals:
                bad.append(f"{p.name}:{node.lineno} {node.value}")
    assert not bad, bad
    for name in ("structural", "estimated", "vol_joint", "bridges"):
        doc = json.dumps(prm.load(name))
        assert '"fee"' not in doc and "fee_latest" not in doc, name


# ---------------------------------------------------------------- sources and secrets (T15)

def test_no_secret_in_the_package():
    key_like = re.compile(r"api_key=(?!REDACTED)[A-Za-z0-9]{8,}")
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    for p in list(PKG.glob("*.py")) + list((PKG / "params").glob("*.json")):
        text = p.read_text(encoding="utf-8")
        assert not key_like.search(text), p.name
        assert not email.search(text), p.name


def test_import_refuses_excluded_sources_and_secrets(tmp_path, monkeypatch):
    from scenariolab.instruments import import_raw as IR
    ins = tmp_path / "instruments"
    (ins / "instr-structural" / "dl_cboe").mkdir(parents=True)
    (ins / "instr-structural" / "dl_cboe" / "VXN_History.csv").write_text("DATE,CLOSE\n01/02/2026,20\n")
    (ins / "instr-structural" / "dl_cboe" / "VIX_History.csv").write_text("DATE,CLOSE\nsecret-token-123456\n")
    (ins / "instr-structural" / "dl_proshares" / "distributions").mkdir(parents=True)
    (ins / "instr-structural" / "dl_proshares" / "etf_splits.csv").write_text("Symbol\n")
    monkeypatch.setenv("SEC_USER_AGENT", "secret-token-123456")
    log = IR.run(ins, None, tmp_path / "dest")
    acts = {r["source"]: r["action"] for r in log["files"]}
    assert acts["ins:instr-structural/dl_cboe/VXN_History.csv"] == "copied"
    assert acts["ins:instr-structural/dl_cboe/VIX_History.csv"] == "rejected"
    doc = json.loads((tmp_path / "dest" / "SOURCES.json").read_text())
    assert "cboe/VXN_History.csv" in doc["files"] and "cboe/VIX_History.csv" not in doc["files"]
    assert doc["files"]["cboe/VXN_History.csv"]["source"] == "cboe" and doc["files"]["cboe/VXN_History.csv"]["sha256"]
    assert set(doc["excluded"]) >= {"ishares_blackrock", "invesco"}
    assert IR._excluded("instr-mapped/dl_blackrock_bme/x.json", None)
    assert IR._excluded("x/y.csv", "https://www.ishares.com/us/products/239454/")
    assert IR._excluded("x/y.csv", "https://dng-api.invesco.com/cache/v1/x")
    assert IR._excluded("instr-structural/dl_cboe/VXN_History.csv", "https://cdn.cboe.com/x") is None


def test_loaders_read_only_recorded_sources(tmp_path, monkeypatch):
    root = tmp_path / "instruments"
    (root / "cboe").mkdir(parents=True)
    (root / "cboe" / "VXN_History.csv").write_text("DATE,CLOSE\n01/02/2026,20\n")
    (root / "cboe" / "BXN_History.csv").write_text("DATE,BXN\n01/02/2026,20\n")
    doc = {"files": {"cboe/VXN_History.csv": {"source": "cboe"}, "cboe/BXN_History.csv": {"source": "invesco"}},
           "excluded": ["invesco"]}
    (root / "SOURCES.json").write_text(json.dumps(doc))
    monkeypatch.setattr(D, "ROOT", root)
    D.clear_cache()
    try:
        assert D.cboe("VXN").iloc[0] == 20
        with pytest.raises(PermissionError):
            D.path("cboe/BXN_History.csv")
        with pytest.raises(D.DataMissing):
            D.path("cboe/VIX_History.csv")
    finally:
        D.clear_cache()


def test_sec_requests_go_only_to_sec_gov(monkeypatch):
    from scenariolab.instruments import refresh as RF
    monkeypatch.setenv("SEC_USER_AGENT", "x")
    with pytest.raises(ValueError, match="sec.gov"):
        RF.sec_get("https://example.com/file.zip")
    calls = []

    def fake_get(url, headers=None, timeout=None, allow_redirects=None):
        calls.append((url, headers, allow_redirects))

        class R:
            status_code, headers_, content = 302, None, b""
            headers = {"Location": "https://evil.example.com/x"}
        return R()
    monkeypatch.setattr(RF.requests, "get", fake_get)
    with pytest.raises(ValueError, match="sec.gov"):
        RF.sec_get("https://www.sec.gov/files/x.zip")
    assert calls and calls[0][2] is False and calls[0][1]["User-Agent"] == "x"


def test_refresh_never_downloads_an_existing_file(monkeypatch, tmp_path):
    from scenariolab.instruments import refresh as RF
    monkeypatch.setattr(RF, "ROOT", tmp_path)
    (tmp_path / "cboe").mkdir()
    for n in ("VXN", "VIX", "BXN", "BXNT", "BXM"):
        (tmp_path / "cboe" / f"{n}_History.csv").write_text("DATE,CLOSE\n")
    monkeypatch.setattr(RF, "_get_public", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert RF.refresh_cboe() == []


# ---------------------------------------------------------------- the build's reader

def test_read_paths_derives_levels_and_cash(tmp_path):
    from scenariolab.instruments import build
    total = np.full((1, 2, 3, 2), 0.01, np.float32)
    income = np.zeros_like(total)
    income[..., 1] = 0.005
    np.savez_compressed(tmp_path / "p.npz", tickers=np.array(["QQQ", "MAIN"]), ids=np.array(["a"]),
                        probs=np.array([1.0]), labels=np.array(["2026-09", "2026-10", "2026-11", "2026-12"]),
                        start=np.array([700.0, 50.0]), total=total, income=income, alive=np.ones((1, 2, 3, 2), bool))
    r = build.read_paths(tmp_path / "p.npz", ["MAIN"])
    assert r["level"].shape == (1, 2, 4, 1) and r["level"][0, 0, 0, 0] == 50.0
    assert r["level"][0, 0, 1, 0] == pytest.approx(50 * 1.005)
    assert r["cash"][0, 0, 1, 0] == pytest.approx(0.005 * 50 * 1.005)


def test_build_statistics_use_the_api_definition_of_the_first_years_cash(out):
    """Review finding 10: the overview's 第一年现金收益率 is year-1 cash per share over the start price, as the
    project endpoint and its table compute it, not a sum of 12 monthly income yields."""
    from scenariolab.instruments import build
    for t in ("QQQI", "MAIN", "SCHD"):
        ip = out.paths[t]
        cash = ip.income * ip.level[:, :-1]
        assert np.allclose(build.income_y1(ip), cash[:, :12].sum(axis=1) / ip.level[:, 0])
    q = out.paths["QQQI"]
    assert np.median(build.income_y1(q)) < np.median(q.income[:, :12].sum(axis=1))      # the NAV shrinks in year 1


def test_packaged_parameters_are_complete():
    assert set(prm.estimated()["tickers"]) == set(ESTIMATED)
    assert set(prm.catalog()["tickers"]) >= set(TICKERS) | {"GOLD"}
    a = prm.packaged_anchors()
    assert a["month"] == "2026-09" and set(a["prices"]) == set(TICKERS) and set(a["estimated"]) == set(ESTIMATED)
    assert a["prices"]["ARCC"]["price"] == pytest.approx(19.10) and a["estimated"]["MO"]["D_last_q"] == 1.11
    assert set(STRUCTURAL) | set(ESTIMATED) == set(TICKERS)
    for t in TICKERS:
        assert prm.catalog()["tickers"][t]["name"]
