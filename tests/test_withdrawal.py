import numpy as np
import pytest

from scenariolab.portfolio import withdrawal as wd

STRATS = list(wd.STRATEGIES)


def paths(T: int, A: int = 2, P: int = 1, cpi_growth: float = 0.0):
    """Zero monthly returns / income and a CPI growing `cpi_growth` per year."""
    total = np.zeros((P, T, A))
    income = np.zeros((P, T, A))
    cpi = 100.0 * (1.0 + cpi_growth) ** (np.arange(T + 1) / 12.0) * np.ones((P, 1))
    return total, income, cpi


def run(total, income, cpi, w, rate, strategy, initial=1000.0, indexed=False):
    f = wd.year_factors(total, income, cpi)
    return wd.simulate(f, np.array(w, float), initial, rate, strategy, indexed, record_holdings=True)


def test_year_factors_compound_price_only_inside_the_year():
    total, income, cpi = paths(12, A=1)
    total[0, 0, 0], income[0, 0, 0] = 0.11, 0.01      # +10% price, 1% income in month 0
    total[0, 1, 0], income[0, 1, 0] = 0.01, 0.01      # income only in month 1
    f = wd.year_factors(total, income, cpi)
    assert f.total[0, 0, 0] == pytest.approx(1.11 * 1.01)
    assert f.price[0, 0, 0] == pytest.approx(1.10)
    # month 0 income on a value of 1, month 1 income on 1.10 (after month 0's price gain)
    assert f.income[0, 0, 0] == pytest.approx(0.01 + 0.01 * 1.10)
    assert f.cpi.shape == (1, 2) and f.cpi[0, 0] == 1.0


def _two_asset_year():
    """Year 1: asset 0 +8% price & 2% income in month 5, +20% price in month 11; asset 1 2% income in month 5."""
    total, income, cpi = paths(24)
    total[0, 5] = [0.10, 0.02]
    income[0, 5] = [0.02, 0.02]
    total[0, 11, 0] = 0.20
    return total, income, cpi


def test_a_pro_rata_reinvests_income_and_sells_proportionally():
    total, income, cpi = _two_asset_year()
    res = run(total, income, cpi, [0.5, 0.5], 0.1, "pro_rata")
    grown = np.array([500 * 1.10 * 1.20, 500 * 1.02])         # income stays invested: 660, 510
    after1 = grown * (1 - 100 / grown.sum())                  # every holding gives up 100/1170
    assert np.allclose(res.holdings[0, 1], after1)
    assert res.wealth[0, 1] == pytest.approx(1170 - 100)
    assert np.allclose(res.holdings[0, 2], after1 * (1 - 100 / 1070))   # flat year 2
    assert res.ruin_year[0] == 0


def test_b_div_then_pro_rata_uses_income_first():
    total, income, cpi = _two_asset_year()
    price = np.array([500 * 1.08 * 1.20, 500.0])              # 648, 500: income was paid out
    pot = 0.02 * 500 + 0.02 * 500                             # 20, on start-of-month-5 values
    # shortfall: 100 - 20 = 80 sold pro-rata
    res = run(total, income, cpi, [0.5, 0.5], 0.1, "div_then_pro_rata")
    assert np.allclose(res.holdings[0, 1], price * (1 - 80 / 1148))
    assert res.wealth[0, 1] == pytest.approx(1148 + pot - 100)
    # income exactly covers the withdrawal: nothing is sold or bought
    res = run(total, income, cpi, [0.5, 0.5], 0.02, "div_then_pro_rata")
    assert np.allclose(res.holdings[0, 1], price)
    # surplus of 10 is reinvested pro-rata
    res = run(total, income, cpi, [0.5, 0.5], 0.01, "div_then_pro_rata")
    assert np.allclose(res.holdings[0, 1], price * (1 + 10 / 1148))


def test_c_div_then_winners_sells_best_total_return_first():
    total, income, cpi = paths(24)
    # year 1: asset 0 total +15% (5% price + 10% income) beats asset 1's +10% (all price)
    total[0, 11], income[0, 11] = [0.15, 0.10], [0.10, 0.0]
    # year 2: asset 0 flat, asset 1 -10%
    total[0, 23, 1] = -0.10
    # rate 10%: pot 10 (10% of 100), shortfall 90 comes out of asset 0 (105 -> 15)
    res = run(total, income, cpi, [0.1, 0.9], 0.1, "div_then_winners")
    assert np.allclose(res.holdings[0, 1], [15.0, 990.0])
    # rate 20%: shortfall 190 exhausts asset 0 (105) and takes 85 from asset 1
    res = run(total, income, cpi, [0.1, 0.9], 0.2, "div_then_winners")
    assert np.allclose(res.holdings[0, 1], [0.0, 905.0])
    # year 2: the empty winner is skipped, all 200 comes from asset 1 after its -10%
    assert np.allclose(res.holdings[0, 2], [0.0, 905 * 0.9 - 200])
    # same year with pro-rata selling, for contrast
    res = run(total, income, cpi, [0.1, 0.9], 0.2, "div_then_pro_rata")
    assert np.allclose(res.holdings[0, 1], np.array([105.0, 990.0]) * (1 - 190 / 1095))


def test_d_ruin_year_detection_and_summary():
    total, income, cpi = paths(60, P=2)
    total[1] = 0.03                                          # path 1 grows 3%/month and survives
    for s in STRATS:
        res = run(total, income, cpi, [0.5, 0.5], 0.3, s)
        # path 0: 1000 -> 700 -> 400 -> 100, then 100 < 300 in year 4
        assert res.ruin_year.tolist() == [4, 0]
        assert np.allclose(res.wealth[0], [1000, 700, 400, 100, 0, 0])
        assert np.allclose(res.holdings[0, 4:], 0.0)
        out = wd.summarize(res, wd.year_factors(total, income, cpi).cpi)
        assert out["ruin_prob"] == 0.5
        assert out["survival"] == [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
        assert out["depletion_year_median"] == 4
        assert out["terminal_real_p5"] == 0.0 and out["terminal_real_p95"] > 1000


def test_d_income_counts_toward_covering_the_withdrawal():
    total, income, cpi = paths(60, A=1)
    total[0, 11::12], income[0, 11::12] = 0.10, 0.10       # 10% income once a year, flat price
    for s in STRATS:
        res = run(total, income, cpi, [1.0], 0.3, s)
        # 1000 -> 800 -> 580 -> 338 -> 71.8; year 5 has 78.98 < 300
        assert np.allclose(res.wealth[0, :5], [1000, 800, 580, 338, 71.8])
        assert res.ruin_year[0] == 5 and res.wealth[0, 5] == 0


def test_e_inflation_indexing_uses_the_path_cpi():
    total, income, cpi = paths(24, cpi_growth=0.10)
    res = run(total, income, cpi, [0.5, 0.5], 0.1, "pro_rata", indexed=True)
    assert np.allclose(res.wealth[0], [1000, 1000 - 110, 1000 - 110 - 121])
    res_nominal = run(total, income, cpi, [0.5, 0.5], 0.1, "pro_rata", indexed=False)
    assert np.allclose(res_nominal.wealth[0], [1000, 900, 800])
    out = wd.summarize(res, wd.year_factors(total, income, cpi).cpi)
    assert out["terminal_real_p50"] == pytest.approx(769 / 1.21)
    assert out["terminal_nominal_p50"] == pytest.approx(769)


def _random_paths(P=300, years=15, A=3, seed=0):
    rng = np.random.default_rng(seed)
    T = 12 * years
    price = rng.normal([0.004, 0.002, 0.0005], [0.045, 0.02, 0.003], size=(P, T, A))
    income = np.broadcast_to(np.array([0.0015, 0.003, 0.0025]), (P, T, A)).copy()
    cpi = 100.0 * np.exp(np.cumsum(np.c_[np.zeros(P), rng.normal(0.0025, 0.003, (P, T))], axis=1))
    return price + income, income, cpi


def reference(total, income, cpi, w, initial, rate, strategy, indexed):
    """Month-by-month loop written straight from the strategy definitions."""
    P, T, A = total.shape
    Y = T // 12
    wealth, ruin = np.zeros((P, Y + 1)), np.zeros(P, int)
    for p in range(P):
        h = initial * np.asarray(w, float)
        wealth[p, 0] = h.sum()
        for y in range(Y):
            pot, growth = 0.0, np.ones(A)
            for m in range(12 * y, 12 * y + 12):
                growth *= 1 + total[p, m]
                if strategy == "pro_rata":
                    h = h * (1 + total[p, m])
                else:
                    pot += (h * income[p, m]).sum()
                    h = h * (1 + total[p, m] - income[p, m])
            need = rate * initial * (cpi[p, 12 * (y + 1)] / cpi[p, 0] if indexed else 1.0)
            if h.sum() + pot < need:
                ruin[p] = ruin[p] or y + 1
                h = np.zeros(A)
            elif pot >= need:
                h = h * (1 + (pot - need) / h.sum())
            elif strategy == "div_then_winners":
                short = need - pot
                for a in np.argsort(-growth, kind="stable"):
                    take = min(short, h[a])
                    h[a] -= take
                    short -= take
            else:
                h = h * (1 - (need - pot) / h.sum())
            wealth[p, y + 1] = h.sum()
    return wealth, ruin


@pytest.mark.parametrize("strategy", STRATS)
@pytest.mark.parametrize("indexed", [True, False])
def test_vectorized_simulation_matches_reference_loop(strategy, indexed):
    total, income, cpi = _random_paths(P=60, years=6)
    w = [0.5, 0.3, 0.2]
    f = wd.year_factors(total, income, cpi)
    for rate in (0.03, 0.17):
        res = wd.simulate(f, np.array(w), 1e6, rate, strategy, indexed)
        ref_w, ref_r = reference(total, income, cpi, w, 1e6, rate, strategy, indexed)
        assert np.allclose(res.wealth, ref_w, rtol=1e-9, atol=1e-6)
        assert (res.ruin_year == ref_r).all()
    assert (ref_r > 0).any() and (ref_r == 0).any()       # the high rate ruins some paths, not all


@pytest.mark.parametrize("strategy", STRATS)
def test_f_ruin_probability_monotone_zero_at_zero_and_safe_rate(strategy):
    total, income, cpi = _random_paths()
    f = wd.year_factors(total, income, cpi)
    w = np.array([0.6, 0.3, 0.1])
    rates = np.linspace(0.0, 0.2, 41)
    ruin = [float((wd.simulate(f, w, 1.0, r, strategy, True).ruin_year > 0).mean()) for r in rates]
    assert ruin[0] == 0.0
    assert np.all(np.diff(ruin) >= 0)
    crit = wd.critical_rates(f, w, strategy, True)
    # the critical rates reproduce the simulated ruin probabilities
    assert np.allclose([wd.ruin_probability(crit, r) for r in rates[1:]], ruin[1:], atol=1 / 300 + 1e-12)
    s = wd.safe_rate(crit)
    assert 0.0 < s < 0.2
    at = (wd.simulate(f, w, 1.0, s, strategy, True).ruin_year > 0).mean()
    above = (wd.simulate(f, w, 1.0, s + 0.0005, strategy, True).ruin_year > 0).mean()
    assert at <= wd.SAFE_RUIN_PROB < above
    # weighted: giving the most fragile paths more weight lowers the safe rate
    fragile = crit < np.median(crit)
    assert wd.safe_rate(crit, np.where(fragile, 3.0, 1.0)) <= s


def test_closed_form_critical_rate_is_the_survival_boundary():
    total, income, cpi = _random_paths(P=50)
    f = wd.year_factors(total, income, cpi)
    w = np.array([0.6, 0.3, 0.1])
    for strategy in wd.LINEAR_STRATEGIES:
        for indexed in (True, False):
            crit = wd.critical_rates(f, w, strategy, indexed)
            inside = crit < wd.RATE_BOUNDS[1]
            ok = wd.simulate(f, w, 1.0, crit * (1 - 1e-7), strategy, indexed).ruin_year == 0
            bad = wd.simulate(f, w, 1.0, crit * (1 + 1e-7), strategy, indexed).ruin_year > 0
            assert ok.all() and bad[inside].all()


def test_critical_rate_hand_computed():
    # flat prices, no income, 10 years: survive iff rate * sum_k c_k <= 1 (c_k = CPI_k / CPI_0 or 1)
    total, income, cpi = paths(120, A=2, cpi_growth=0.02)
    f = wd.year_factors(total, income, cpi)
    expected = {False: 0.1, True: 1.0 / sum(1.02 ** k for k in range(1, 11))}      # 0.0895 indexed
    for strategy in STRATS:
        for indexed, r in expected.items():
            crit = wd.critical_rates(f, np.array([0.5, 0.5]), strategy, indexed)[0]
            assert r - 1e-5 <= crit <= r + 1e-12
    # 10% income once a year on a flat price: every strategy grows 1.1x a year before the withdrawal,
    # so the boundary is the 10-year annuity rate at 10% (the pot counts toward the withdrawal)
    total, income, cpi = paths(120, A=1)
    total[0, 11::12], income[0, 11::12] = 0.10, 0.10
    f = wd.year_factors(total, income, cpi)
    r = 1.0 / sum(1.1 ** -k for k in range(1, 11))                                  # 0.16275
    for strategy in STRATS:
        crit = wd.critical_rates(f, np.array([1.0]), strategy, False)[0]
        assert r - 1e-5 <= crit <= r + 1e-12


def _head_and_tail(P=80, head_years=5, tail_paths=30, tail_years=7, seed=4):
    total, income, cpi = _random_paths(P=P, years=head_years, seed=seed)
    head = wd.year_factors(total, income, cpi)
    t_total, t_income, t_cpi = _random_paths(P=tail_paths, years=tail_years, seed=seed + 1)
    tail = wd.year_factors(t_total, t_income, 3.7 * t_cpi)          # any CPI base: chain() uses its ratios
    rows = np.arange(P) % tail_paths
    return head, tail, rows, (t_total, t_income, t_cpi)


def test_chain_keeps_the_head_bit_identical_and_chains_cpi():
    head, tail, rows, _ = _head_and_tail()
    f = head.chain(tail, rows)
    Yh = head.years
    assert f.years == Yh + tail.years and f.total.shape[0] == head.total.shape[0]
    for k in ("total", "price", "income", "order"):
        assert np.array_equal(getattr(f, k)[:, :Yh], getattr(head, k))           # bit-identical, not just close
        assert np.array_equal(getattr(f, k)[:, Yh:], getattr(tail, k)[rows])
    assert np.array_equal(f.cpi[:, :Yh + 1], head.cpi)
    # the tail's inflation continues from each path's own year-Yh CPI level
    assert np.allclose(f.cpi[:, Yh + 1:] / f.cpi[:, Yh:Yh + 1], tail.cpi[rows, 1:] / tail.cpi[rows, :1], rtol=1e-13)
    # every withdrawal statistic up to year Yh is exactly what the head alone gives
    w = np.array([0.6, 0.3, 0.1])
    for s in STRATS:
        for rate in (0.04, 0.12):
            long = wd.simulate(f, w, 1e6, rate, s, True)
            short = wd.simulate(head, w, 1e6, rate, s, True)
            assert np.array_equal(long.wealth[:, :Yh + 1], short.wealth)
            assert np.array_equal(long.head(Yh).ruin_year, short.ruin_year)
    with pytest.raises(ValueError):
        head.chain(tail)                                    # 80 head paths, 30 tail paths, no row map


def test_chain_matches_year_factors_of_the_concatenated_months():
    """Appending the tail's months to each path and computing year factors gives the same factors."""
    total, income, cpi = _random_paths(P=40, years=4, seed=8)
    t_total, t_income, t_cpi = _random_paths(P=40, years=6, seed=9)
    rows = np.arange(40)[::-1]
    month_cpi = np.concatenate([cpi, cpi[:, -1:] * t_cpi[rows, 1:] / t_cpi[rows, :1]], axis=1)
    whole = wd.year_factors(np.concatenate([total, t_total[rows]], axis=1),
                            np.concatenate([income, t_income[rows]], axis=1), month_cpi)
    f = wd.year_factors(total, income, cpi).chain(wd.year_factors(t_total, t_income, t_cpi), rows)
    for k in ("total", "price", "income", "cpi"):
        assert np.allclose(getattr(f, k), getattr(whole, k), rtol=1e-12, atol=0)
    assert np.array_equal(f.order, whole.order)


@pytest.mark.parametrize("strategy", STRATS)
def test_head_equals_a_run_of_that_length(strategy):
    total, income, cpi = _random_paths(P=120, years=12, seed=3)
    f = wd.year_factors(total, income, cpi)
    w = np.array([0.5, 0.3, 0.2])
    long = wd.simulate(f, w, 1.0, 0.09, strategy, True)
    for k in (1, 6, 11):
        short = wd.simulate(f, w, 1.0, 0.09, strategy, True, k)
        cut = long.head(k)
        assert np.array_equal(cut.wealth, short.wealth) and np.array_equal(cut.ruin_year, short.ruin_year)
        assert wd.summarize(cut, f.cpi) == wd.summarize(short, f.cpi)
    assert (long.ruin_year > 6).any()                       # some paths are ruined after year 6 only
    assert "wealth_fan_real" not in wd.summarize(long, f.cpi, with_fan=False)


@pytest.mark.parametrize("strategy", STRATS)
def test_ruin_rises_and_safe_rate_falls_with_horizon(strategy):
    head, tail, rows, _ = _head_and_tail(P=200, head_years=5, tail_paths=200, tail_years=10, seed=12)
    f = head.chain(tail, rows)
    w = np.array([0.6, 0.3, 0.1])
    rates = np.round(np.arange(0.03, 0.0801, 0.01), 4)
    ruin = np.array([[float((wd.simulate(f, w, 1.0, r, strategy, True, Y).ruin_year > 0).mean()) for r in rates]
                     for Y in (5, 10, 15)])
    assert np.all(np.diff(ruin, axis=1) >= 0)               # more withdrawal, more ruin
    assert np.all(np.diff(ruin, axis=0) >= 0)               # longer horizon, more ruin
    assert ruin[-1, -1] > ruin[0, 0]
    crit = [wd.critical_rates(f, w, strategy, True, Y) for Y in (5, 10, 15)]
    assert np.all(crit[1] <= crit[0]) and np.all(crit[2] <= crit[1])     # per path, bisection included
    safe = [wd.safe_rate(c) for c in crit]
    assert safe[2] <= safe[1] <= safe[0] and safe[2] < safe[0]


def test_safe_rate_quantile_rule_and_cap():
    crit = np.linspace(0.01, 0.20, 100)
    # 5% of 100 paths may fail: the 6th smallest rate is still safe (5 paths below it)
    assert wd.safe_rate(crit) == pytest.approx(np.floor(crit[5] * 1e4) / 1e4)
    assert wd.safe_rate(np.full(10, 0.2)) == 0.2
    assert wd.safe_rate(np.array([])) is None
    total, income, cpi = paths(24)
    f = wd.year_factors(total + 0.01, income, cpi)        # strong growth: never ruined below 20%
    assert wd.critical_rates(f, np.array([0.5, 0.5]), "div_then_winners", True).tolist() == [0.2]


# ---------------------------------------------------------------- cash flows (INSTRUMENTS_DESIGN §7.3)

def test_cash_flows_pot_need_and_alive_hand_computed():
    total, income, cpi = paths(36, A=2, cpi_growth=0.10)
    total[0, 11::12, 0], income[0, 11::12, 0] = 0.05, 0.05      # asset 0 pays 5% in cash once a year, flat price
    f = wd.year_factors(total, income, cpi)
    w = np.array([0.6, 0.4])
    res = wd.simulate(f, w, 1000.0, 0.1, "div_then_pro_rata", True, record_holdings=True)
    fl = wd.cash_flows(res, f, 100.0, True)
    # year 1: 600 in asset 0 pays 30 against 100 x 1.1; the holdings then shrink by (1030 - 110) / 1000
    assert fl["pot"][0, 0] == pytest.approx(30.0) and fl["need"][0, 0] == pytest.approx(110.0)
    assert fl["pot"][0, 1] == pytest.approx(600 * 0.92 * 0.05) and fl["need"][0, 1] == pytest.approx(121.0)
    assert fl["alive"].all() and fl["pot"].shape == fl["need"].shape == (1, 3)
    # pro_rata reinvests the cash: what the start-of-year holdings pay is still reported
    res = wd.simulate(f, w, 1000.0, 0.1, "pro_rata", True, record_holdings=True)
    fl = wd.cash_flows(res, f, 100.0, False)
    assert fl["pot"][0, 1] == pytest.approx(630 * (920 / 1030) * 0.05) and np.allclose(fl["need"], 100.0)
    # a ruined path: alive in its ruin year, not after; nothing is paid after it
    res = wd.simulate(f, w, 1000.0, 0.6, "div_then_pro_rata", False, record_holdings=True)
    assert res.ruin_year[0] == 2
    fl = wd.cash_flows(res, f, 600.0, False)
    assert fl["alive"][0].tolist() == [True, True, False] and fl["pot"][0, 2] == 0.0
    with pytest.raises(ValueError):
        wd.cash_flows(wd.simulate(f, w, 1000.0, 0.1, "pro_rata"), f, 100.0)


def test_cut_flags_compare_each_years_cash_per_unit_with_year_one():
    P, Y = 3, 10
    price = np.ones((P, Y, 1))
    income = np.full((P, Y, 1), 0.05)
    price[1, 1, 0] = 0.7                      # path 1: the price falls 30% in year 2, the yield stays 5%
    price[2, 4, 0] = 0.85                     # path 2: a 15% fall: a cut of 10% but not of 20%
    income[0, :, 0] = [0.05] + [0.0] * 9      # path 0 stops paying after year 1
    f = wd.YearFactors(price + income, price, income, np.ones((P, Y + 1)), np.zeros((P, Y, 1), dtype=int))
    assert wd.cut_flags(f, 0, 10, 0.2).tolist() == [1.0, 1.0, 0.0]
    assert wd.cut_flags(f, 0, 10, 0.1).tolist() == [1.0, 1.0, 1.0]
    assert wd.cut_flags(f, 0, 5, 0.1).tolist() == [1.0, 1.0, 0.0]      # path 2's cut comes in year 6
    none = wd.YearFactors(price, price, np.zeros_like(price), np.ones((P, Y + 1)), np.zeros((P, Y, 1), dtype=int))
    assert np.isnan(wd.cut_flags(none, 0)).all()                       # a holding that pays nothing has no cut
