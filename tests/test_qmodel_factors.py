"""The factor layer (SIM_DESIGN §5, §14 P3): qmodel/factors.py, qmodel/policy.py, qmodel/factor_checks.py.

P3 exit tests: the OLS limit of the BVAR (<= 1e-6); hard statements held to 1e-8 on every path; the Gaussian Monte
Carlo mean equals the analytic conditional mean (|z| < 3); a stated u_ff leaves same-quarter E0 macro exactly
unchanged; the observed policy rate never below 0.1; determinism by seed; T-F F1-F10 and T-C computed and stored in
data/results/qmodel.json with pass/fail shown. P4b (long-horizon stability): the bounded (logit) scales invert
exactly and keep their levels inside; the draw rule tests the system the cycle module simulates; 80-quarter paths stay
finite and bounded; the observed 10-year has its floor; statements beyond a ceiling are delivered inside it.

The BVAR, covariance and sampler pieces run on synthetic data anywhere. The model tests need the quarterly panel
(python -m scenariolab.qmodel.panel) and skip without it; the stored-results test also needs
python -m scenariolab.qmodel.factor_checks. Nothing downloads and nothing calls an LLM.
"""
import math

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.qmodel import config_q as cq
from scenariolab.qmodel import factors as F

S = F.Statement
needs_qpanel = pytest.mark.skipif(not (cq.PANEL_Q_PATH.exists() and config.PANEL_PATH.exists()),
                                  reason="quarterly panel not built (python -m scenariolab.qmodel.panel)")


# ================================================================ synthetic: the BVAR pieces

def _var_data(T=240, n=3, seed=0):
    rng = np.random.default_rng(seed)
    A1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.6, 0.2], [0.1, 0.0, 0.4]])[:n, :n]
    A2 = 0.1 * np.eye(n)
    c = np.array([0.2, -0.1, 0.3])[:n]
    Y = np.zeros((T, n))
    for t in range(2, T):
        Y[t] = c + A1 @ Y[t - 1] + A2 @ Y[t - 2] + rng.standard_normal(n) * np.array([1.0, 0.5, 2.0])[:n]
    return pd.DataFrame(Y, columns=F.CORE[:n])


def test_bvar_ols_limit():
    """A diffuse Minnesota prior (lambda -> inf), no sum-of-coefficients prior and Gaussian weights: the posterior
    mean is the OLS estimate with a constant (<= 1e-6)."""
    D = _var_data()
    fit = F.fit_core(D, lam_grid=(1e7,), nu_grid=(float("inf"),), soc=False)
    yl, X = F.lagmat(D.to_numpy(float), 2)
    ols = np.linalg.lstsq(X, yl, rcond=None)[0]
    assert np.max(np.abs(fit.B - ols)) < 1e-6
    assert np.allclose(fit.w, 1.0)


def test_chib_identity_equals_the_closed_form_marginal_likelihood():
    """For the Gaussian likelihood without missing data Chib's identity is exact: the density code (inverse
    Wishart, matrix normal, the t family at nu = inf) reproduces the closed-form marginal likelihood."""
    D = _var_data(seed=3)
    r = F._em_once(D.to_numpy(float), 2, (0.2,), float("inf"), True, 5, 1e-12)
    assert abs(F._chib(r, D.to_numpy(float), 2, float("inf")) - r["ml"]) < 1e-6


def test_student_t_weights_downweight_an_outlier_by_rule():
    D = _var_data(seed=1)
    D.iloc[150] = D.iloc[150] + np.array([25.0, -15.0, 40.0])
    fit = F.fit_core(D, lam_grid=(0.2,), nu_grid=(4.0, float("inf")))
    assert fit.nu == 4.0                                          # the marginal likelihood prefers fat tails
    w = fit.w
    assert w[150 - 2] < 0.1 < np.median(w)                        # the outlier row gets a small weight


def test_missing_entries_are_imputed_inside_the_em():
    D = _var_data(seed=2)
    truth = D.iloc[100:110, 1].copy()
    D.iloc[100:110, 1] = np.nan
    fit = F.fit_core(D, lam_grid=(0.5,), nu_grid=(float("inf"),), soc=False)
    imp = fit.Y[100:110, 1]
    assert np.all(np.isfinite(imp)) and fit.missing[100:110, 1].all()
    assert np.mean(np.abs(imp - truth.to_numpy())) < 1.0          # near the truth (innovation sd 0.5 per step)


def test_steady_state_prior_binds_with_a_tight_sd_and_not_with_a_diffuse_one():
    D = _var_data(seed=4)
    fit = F.fit_core(D, lam_grid=(0.3,), nu_grid=(float("inf"),))
    A, c = F.split_B(fit.B, 3, 2)
    mu_free, c_free = F.steady_state(A, fit.Sigma, fit.Y, fit.w, np.zeros(3), np.full(3, 1e6))
    mu_tight, c_tight = F.steady_state(A, fit.Sigma, fit.Y, fit.w, np.array([5.0, 5.0, 5.0]), np.full(3, 1e-4))
    assert np.allclose(mu_tight, 5.0, atol=1e-3)
    assert np.allclose(c_tight, (np.eye(3) - sum(A)) @ mu_tight)
    assert np.max(np.abs(c_free - c)) < 1e-6 * 10                 # diffuse: the data's own constant


def test_factor_covariance_em_recovers_a_low_rank_structure_with_missing_data():
    rng = np.random.default_rng(0)
    L = rng.standard_normal((8, 2))
    X = rng.standard_normal((400, 2)) @ L.T + 0.3 * rng.standard_normal((400, 8))
    X[:120, 5:] = np.nan                                           # late series, masked before they start
    S = F.em_cov(X, np.ones(400), "fa", k=2, iters=80)
    true = L @ L.T + 0.09 * np.eye(8)
    assert np.max(np.abs(S - true)) / np.max(np.abs(true)) < 0.25
    Slw = F.em_cov(X, np.ones(400), "lw", alpha=0.5, iters=30)
    assert np.all(np.linalg.eigvalsh(Slw) > 0)


def test_exponential_tilt_hits_the_target_share():
    share = np.r_[np.zeros(80), np.full(15, 0.5), np.ones(25)]
    w = F._tilt(share, 0.5)
    assert abs(float(w @ share) - 0.5) < 1e-6 and abs(w.sum() - 1) < 1e-12


def test_truncated_gibbs_respects_the_box():
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    lo, hi = np.array([1.5, -np.inf]), np.array([2.0, 0.0])
    x = F._gibbs_tmvn(np.zeros(2), cov, lo, hi, 2000, rng)
    assert np.all(x[:, 0] >= 1.5 - 1e-12) and np.all(x[:, 0] <= 2.0 + 1e-12) and np.all(x[:, 1] <= 1e-12)


def test_bounded_scales_invert_exactly_and_keep_levels_inside():
    """P4b: on the logit scale lo < level < hi for every x, value() inverts level() inside the bounds, dlevel is the
    derivative and the Gauss-Hermite moments match Monte Carlo; the 'log' kind is P3's lo + exp(x)."""
    b = F.Bounded(2.0, 30.0)
    x = np.linspace(-12.0, 12.0, 241)
    y = b.level(x)
    assert np.all((y > 2.0) & (y < 30.0))
    assert np.max(np.abs(b.value(y) - x)) < 1e-9
    h = 1e-6
    assert np.allclose(b.dlevel(x), (b.level(x + h) - b.level(x - h)) / (2 * h), rtol=1e-5, atol=1e-9)
    assert np.isnan(b.value(np.array([2.0, 30.0, 31.0, 1.0]))).all()            # data outside the bounds: missing
    assert 2.0 < float(b.level(b.value(np.array([31.0]), clip=True))[0]) < 30.0   # a target: kept just inside
    rng = np.random.default_rng(0)
    m, s_ = np.array([-2.3, 0.4]), 0.6
    mu, sd = b.moments(m, s_)
    mc = b.level(m[:, None] + s_ * rng.standard_normal((2, 400000)))
    assert np.allclose(mu, mc.mean(axis=1), rtol=2e-3) and np.allclose(sd, mc.std(axis=1), rtol=5e-3)
    lg = F.Bounded(2.0, math.inf, "log")
    assert np.allclose(lg.level(x[:60]), 2.0 + np.exp(x[:60])) and np.allclose(lg.value(lg.level(x[:60])), x[:60])
    assert lg.moments(np.array([0.3]), 0.5)[0][0] == pytest.approx(2.0 + math.exp(0.3 + 0.125))
    tf = F.transforms()
    assert set(tf.vars) == {"lu", "sp", "gz"} and set(tf.obs) == {"d_unrate", "unrate", "d_spread", "spread", "d_gz",
                                                                   "gz"}
    assert all((bb.lo, bb.hi) == cq.FACTOR_BOUNDS[v] and bb.kind == "logit" for v, bb in tf.vars.items())
    assert all(bb.kind == "log" and not math.isfinite(bb.hi) for bb in F.transforms("log").vars.values())


def test_the_cycle_system_drops_ips_feedback_and_the_rule_tests_it():
    """P4b: under the cycle module IP growth is exogenous, so the error correction that runs through IP's equation is
    gone and the innovation regression on IP feeds it back with the opposite sign. A VAR whose unemployment is
    explosive on its own but reverts through an IP rebound is stationary; its cycle system is not, and stable()
    rejects it. The root limit is the 40-year half-life."""
    n = F.NC
    ip, lu = F.IDX["ip"], F.IDX["lu"]
    A1, A2 = 0.5 * np.eye(n), np.zeros((n, n))
    A1[lu, lu], A1[lu, ip], A1[ip, lu] = 1.01, -0.02, 2.0
    Sig = np.eye(n)
    Sig[ip, lu] = Sig[lu, ip] = -0.3
    r_var, r_cyc = F.roots([A1, A2], Sig)
    At = F.cycle_A([A1, A2], Sig)
    assert np.allclose(At[0][ip], 0.0) and np.allclose(At[1][ip], 0.0)
    assert r_var < cq.FACTOR_MAX_ROOT < 1.0 < r_cyc and not F.stable([A1, A2], Sig)
    assert cq.FACTOR_MAX_ROOT ** cq.FACTOR_HALF_LIFE_MAX_Q == pytest.approx(0.5)


# ================================================================ the fitted model (real data)

@pytest.fixture(scope="module")
def Q():
    from scenariolab.qmodel import panel as qpanel
    return qpanel.load_panel()


@pytest.fixture(scope="module")
def P():
    from scenariolab.data import panel as monthly_panel
    return monthly_panel.load_panel()


@pytest.fixture(scope="module")
def model(Q, P):
    return F.fit(Q, "2026Q3", anchors="forward", P=P, seed=0)


@pytest.fixture(scope="module")
def init(model, Q):
    return model.initial_state(Q, "2026Q4")


@needs_qpanel
def test_fit_summary(model):
    info = model.info
    assert info["sample"] == ["1973Q1", "2026Q3"] and info["nu"] in cq.FACTOR_NU_GRID
    assert info["lambda"] in cq.FACTOR_LAMBDA_GRID
    assert len(model.draws) == cq.FACTOR_DRAWS and 0 <= info["rejection_rate"] < 1
    assert all(F.stable(d[0], d[2]) for d in model.draws)               # the VAR and the cycle system (P4b)
    assert len(model.sats.fields) == 42 and model.n == 52
    # the zero-bound quarters' shadow rate is imputed by the EM, never the floor
    zlb = model.core.missing[:, F.IDX["rff"]]
    assert zlb.sum() >= 30 and np.all(np.isfinite(model.core.Y[zlb, F.IDX["rff"]]))
    # extreme quarters get small t weights by the rule
    smallest = [q for q, _, _ in info["smallest_weights"][:3]]
    assert "2020Q2" in smallest


@needs_qpanel
def test_fit_uses_only_quarters_up_to_the_train_end(Q, P):
    m1 = F.fit(Q, "2004Q4", anchors="train_end", P=P, seed=0, satellites=False)
    Q2 = Q.copy()
    later = Q2.index > pd.Period("2004Q4", "Q")
    num = Q2.select_dtypes("number").columns
    Q2.loc[later, num] = Q2.loc[later, num] * 1.7 + 3.0
    P2 = P.copy()
    late_m = P2.index > pd.Timestamp("2004-12-01")
    P2.loc[late_m, P2.select_dtypes("number").columns] = P2.loc[late_m, P2.select_dtypes("number").columns] * 2.0
    m2 = F.fit(Q2, "2004Q4", anchors="train_end", P=P2, seed=0, satellites=False)
    assert np.allclose(m1.core.B, m2.core.B) and np.allclose(m1.c, m2.c)
    assert m1.anchors["r_star"] == pytest.approx(m2.anchors["r_star"])


@needs_qpanel
def test_train_end_anchors_follow_the_engine_rule(Q, P):
    from scenariolab.engine import anchors as eng
    m = F.fit(Q, "1994Q4", anchors="train_end", P=P, seed=0, satellites=False)
    a = eng.train_end_anchors(P, "1994-12")
    assert m.anchors["r_star"] == pytest.approx(a["r_star"]) and m.anchors["unrate"] == pytest.approx(a["unrate"])
    assert m.anchors["r_star"] == pytest.approx(2.19, abs=0.01) and m.anchors["term"] == pytest.approx(1.67, abs=0.01)
    assert m.mu_sd[F.IDX["rff"]] == cq.STEADY_STATE_SD["train_end"]
    assert m.mu_star[F.IDX["ry10"]] == pytest.approx(a["r_star"] + a["term"])


@needs_qpanel
def test_hard_statements_hold_on_every_path(model, init):
    """Every kind of hard statement to 1e-8 on every path: satellites, raw policy (chained from the start), y/y,
    an intervention, an unchained unemployment change (Newton), levels, a sum, and a soft one alongside."""
    st = [S("d_ff", (1,), -0.25), S("d_ff", (2,), -0.25), S("d_gz", (1,), 0.4), S("d_epu", (2,), 15.0),
          S("cpi_yoy", (4,), 3.2), S("u_ff", (3,), -0.3), S("d_unrate", (5,), 0.3), S("unrate", (6,), 5.0),
          S("d_ip", (2, 3), 0.4, agg="sum"), S("y10", (7,), 4.5), S("d_spread", (8,), 0.2, sd=0.1)]
    res = F.simulate(model, init, 10, n_paths=600, seed=11, statements=st)
    for d in res.delivery:
        if d["kind"] == "hard":
            assert d["max_abs_error"] < 1e-8, d
    soft = [d for d in res.delivery if d["kind"] == "soft"][0]
    assert 0.6 < soft["share_within_1sd"] < 0.76


@needs_qpanel
def test_soft_statements_are_met_within_their_sd(model, init):
    st = [S("d_gz", (1,), 0.5, sd=0.2), S("d_unrate", (2,), 0.4, sd=0.1), S("d_y2", (3,), -0.3, sd=0.15)]
    res = F.simulate(model, init, 4, n_paths=3000, seed=5, statements=st)
    for d in res.delivery:
        assert abs(d["share_within_1sd"] - 0.6827) < 0.04, d


@needs_qpanel
def test_policy_rate_floor(model, init):
    """Observed ff = max(shadow, 0.1) on every path; a cut to the floor is a range on the shadow rate; a hike stated
    after an unstated quarter is delivered on the observed rate even from the floor."""
    res = F.simulate(model, init, 40, n_paths=1000, seed=3)
    ff = res.levels["ff"]
    assert ff.min() >= cq.FF_FLOOR - 1e-12
    assert np.allclose(ff, np.maximum(res.levels["ff_shadow"], cq.FF_FLOOR))
    st = [S("d_ff", (1,), -2.0), S("d_ff", (2,), -2.5)]               # 3.75 -> 1.75 -> floor
    r2 = F.simulate(model, init, 4, n_paths=500, seed=4, statements=st)
    assert np.allclose(r2.levels["ff"][:, 1], cq.FF_FLOOR) and r2.levels["ff_shadow"][:, 1].max() <= cq.FF_FLOOR + 1e-9
    r3 = F.simulate(model, init, 4, n_paths=500, seed=4, statements=[S("d_ff", (3,), 0.5)])
    assert np.max(np.abs(r3.x["d_ff"][:, 2] - 0.5)) < 1e-8


@needs_qpanel
def test_monte_carlo_mean_equals_the_analytic_conditional_mean(model, init):
    st = [S("d_ff", (1,), 0.5), S("d_gz", (1,), -0.3), S("pc", (3,), 4.0), S("d_y2", (2,), 0.2)]
    mean, cov = F.conditional_moments(model, init, 4, st)
    n = 20000
    res = F.simulate(model, init, 4, n_paths=n, seed=21, statements=st, gaussian=True, param_uncertainty=False,
                     cycle=False, saturate=False)
    Y = np.concatenate([res.core, res.sats], axis=2)
    sd = np.sqrt(np.diag(cov)).reshape(4, model.n)
    for h in range(4):
        for v in ["pc", "ip", "lu", "rff", "ry10", "sp", "gz", "usd"]:
            j = F.IDX[v]
            if sd[h, j] < 1e-10:
                assert abs(Y[:, h, j].mean() - mean[h, j]) < 1e-8
                continue
            z = (Y[:, h, j].mean() - mean[h, j]) / (sd[h, j] / math.sqrt(n))
            assert abs(z) < 3.0, (h, v, z)
    # and the Monte Carlo covariance of an unstated entry matches the analytic one
    j = F.IDX["sp"]
    assert np.std(Y[:, 3, j]) == pytest.approx(sd[3, j], rel=0.05)


@needs_qpanel
def test_stated_u_ff_leaves_same_quarter_e0_unchanged(model, init, Q):
    from scenariolab.qmodel import policy
    usd = policy.shock_sd(policy.fit(Q, "2026Q3"))["u_ff"]
    kw = dict(n_paths=400, seed=9, cycle=False)
    base = F.simulate(model, init, 8, **kw)
    ease = F.simulate(model, init, 8, statements=[S("u_ff", (1,), -usd)], **kw)
    cls = F.field_classes()
    e0 = [f for f in base.x if cls.get(f) == "E0"] + ["d_cpi"]
    assert len(e0) > 20
    for f in e0:
        assert np.max(np.abs(ease.x[f][:, 0] - base.x[f][:, 0])) < 1e-10, f
    # it is a structural shock: delivered exactly, markets move on impact, E0 later
    assert np.max(np.abs(ease.shocks["u_ff"][:, 0] + usd)) < 1e-8
    assert np.mean(ease.x["d_y2"][:, 0] - base.x["d_y2"][:, 0]) < 0
    assert np.max(np.abs(ease.x["d_ip"][:, 1] - base.x["d_ip"][:, 1])) > 1e-6


@needs_qpanel
def test_stated_u_bs_alone_moves_nothing_but_the_balance_sheet(model, init):
    """§5.3: QE cannot be learned, so u_bs is ordered last and d_bs feeds no other equation: a stated u_bs moves d_bs
    and nothing else (a QE statement must state its intervention paths on d_tp10, d_y2 and d_gz)."""
    kw = dict(n_paths=200, seed=3, cycle=False)
    base = F.simulate(model, init, 6, **kw)
    qe = F.simulate(model, init, 6, statements=[S("u_bs", (1,), 1.0)], **kw)
    assert np.max(np.abs(qe.shocks["u_bs"][:, 0] - 1.0)) < 1e-8            # the stated shock is the shock
    assert not np.allclose(qe.x["d_bs"][:, 0], base.x["d_bs"][:, 0])
    for f in base.x:
        if f != "d_bs":
            assert np.max(np.abs(qe.x[f] - base.x[f])) < 1e-9, f


@needs_qpanel
def test_determinism_by_seed(model, init):
    st = [S("d_gz", (1,), 0.3, sd=0.1), S("d_unrate", (2,), lo=-0.2, hi=0.4)]
    a = F.simulate(model, init, 6, n_paths=300, seed=17, statements=st)
    b = F.simulate(model, init, 6, n_paths=300, seed=17, statements=st)
    c = F.simulate(model, init, 6, n_paths=300, seed=18, statements=st)
    for f in ("d_ff", "d_gz", "d_rgdp", "x_oil_cause"):
        assert np.array_equal(a.x[f], b.x[f])
    assert np.array_equal(a.drawn, b.drawn) and not np.allclose(a.x["d_gz"], c.x["d_gz"])


@needs_qpanel
def test_schema_mapping_identities(model, init):
    """§5.1: d_ff = Δ observed ff; d_y10 = Δ(ry10 + pi_trend); d_spread, d_gz, d_usd, d_unrate the level changes;
    d_ip = ip / 4; d_oil = Δro + 100 Δlog CPI; goods inflation from the CPI identity."""
    res = F.simulate(model, init, 12, n_paths=200, seed=1, usd_hold=False)
    lv, x, k = res.levels, res.x, res.core

    def d(a, first):
        return np.diff(np.concatenate([np.full((a.shape[0], 1), first), a], axis=1), axis=1)
    z1 = init.z[1]
    assert np.allclose(x["d_ff"], d(lv["ff"], init.ff0), atol=1e-12)
    y10 = np.maximum(k[:, :, F.IDX["ry10"]] + lv["pi_trend"], cq.Y10_FLOOR)       # the observed 10-year (P4b floor)
    assert np.allclose(lv["y10"], y10)
    assert np.allclose(x["d_y10"], d(y10, max(z1[F.IDX["ry10"]] + init.pi0, cq.Y10_FLOOR)))
    assert np.allclose(x["d_unrate"], d(lv["unrate"], init.u0))
    assert np.allclose(lv["unrate"], model.tf.level("lu", k[:, :, F.IDX["lu"]]))
    assert np.allclose(x["d_ip"], k[:, :, F.IDX["ip"]] / 4)
    assert np.allclose(x["d_oil"], d(k[:, :, F.IDX["ro"]], z1[F.IDX["ro"]]) + np.diff(
        np.concatenate([np.full((200, 1), init.lvl_cpi0), lv["cpi"]], axis=1), axis=1))
    assert np.allclose(model.wg * x["d_cpi_goods"] + model.ws * x["d_cpi_serv"], x["d_cpi"])
    assert np.allclose(x["d_usd"], d(k[:, :, F.IDX["usd"]], z1[F.IDX["usd"]]))
    for f, v, lvl in (("d_spread", "sp", "spread"), ("d_gz", "gz", "gz")):
        level = model.tf.level(v, k[:, :, F.IDX[v]])
        level0 = model.tf.level(v, z1[F.IDX[v]])
        assert np.allclose(lv[lvl], level) and np.allclose(x[f], d(level, level0))
    for lvl, v in (("unrate", "lu"), ("spread", "sp"), ("gz", "gz")):          # inside the bounded scales (P4b)
        lo, hi = cq.FACTOR_BOUNDS[v]
        assert lo < lv[lvl].min() and lv[lvl].max() < hi


@needs_qpanel
def test_usd_is_held_at_no_change_when_unstated(model, init):
    res = F.simulate(model, init, 8, n_paths=500, seed=2)
    assert np.max(np.abs(res.x["d_usd"].mean(axis=0))) < 1e-10
    res2 = F.simulate(model, init, 8, n_paths=500, seed=2, statements=[S("d_usd", (1,), 3.0)])
    assert np.max(np.abs(res2.x["d_usd"][:, 0] - 3.0)) < 1e-8


def test_the_dollar_hold_ignores_non_finite_paths(model, init):
    """The hold subtracts the cross-path mean: a non-finite path must not turn every path's dollar into NaN."""
    res = F.simulate(model, init, 4, n_paths=200, seed=3, usd_hold=False)
    Y = np.concatenate([res.core, res.sats], axis=2).astype(float)
    Y[0, :, F.IDX["usd"]] = np.inf
    x, _, _ = F.outputs(model, init, Y, np.zeros_like(Y), np.asarray(res.draw_index), 4, usd_hold=True)
    assert np.isfinite(x["d_usd"][1:]).all()
    assert np.max(np.abs(x["d_usd"][1:].mean(axis=0))) < 1e-9


@needs_qpanel
def test_step_and_path_mode_agree_on_the_first_quarter(model, init):
    st = [S("d_ip", (1,), 0.5), S("d_ff", (1,), -0.25), S("d_gz", (1,), 0.2, sd=0.1)]
    p = F.simulate(model, init, 6, n_paths=500, seed=4, statements=st)
    s = F.step(model, init, n_paths=500, seed=4, statements=st)
    for f in ("d_gz", "d_y2", "d_rgdp", "d_cpi_goods", "d_epu"):
        assert np.max(np.abs(p.x[f][:, 0] - s.x[f][:, 0])) < 1e-9, f


@needs_qpanel
def test_cycle_module_runs_only_where_growth_is_unstated(model, init):
    on = F.simulate(model, init, 6, n_paths=300, seed=8, statements=[S("d_ip", (2,), 1.0)])
    off = F.simulate(model, init, 6, n_paths=300, seed=8, statements=[S("d_ip", (2,), 1.0)], cycle=False)
    assert on.info["cycle_quarters"] == 5 and off.info["cycle_quarters"] == 0
    assert not np.allclose(on.x["d_ip"][:, 0], off.x["d_ip"][:, 0])
    assert np.max(np.abs(on.x["d_ip"][:, 1] - 1.0)) < 1e-8
    tilt = F.simulate(model, init, 8, n_paths=1000, seed=8, cycle_odds={"contraction": 0.8})
    assert tilt.x["d_ip"].mean() < on.x["d_ip"].mean()


@needs_qpanel
def test_ranges_by_accept_reject_and_by_gibbs(model, init):
    wide = F.simulate(model, init, 4, n_paths=400, seed=6, statements=[S("d_gz", (1, 2), lo=-0.5, hi=0.6)])
    assert wide.info["ranges"]["method"] == "accept_reject"
    assert all(d["share_inside"] == 1.0 for d in wide.delivery)
    narrow = F.simulate(model, init, 4, n_paths=400, seed=6, statements=[S("d_gz", (1,), lo=1.5, hi=1.6),
                                                                         S("d_y2", (2,), lo=-0.05, hi=0.0)])
    assert narrow.info["ranges"]["method"] == "gibbs"
    assert all(d["share_inside"] == 1.0 for d in narrow.delivery)


@needs_qpanel
def test_saturation_clips_an_absurd_value_with_a_note(model, init):
    res = F.simulate(model, init, 2, n_paths=200, seed=1, statements=[S("d_epu", (1,), 900.0), S("d_oil", (1,),
                                                                                                    150.0)])
    notes = [n for n in res.notes if "used" in n]
    assert [n["var"] for n in notes] == ["d_epu"] and F.CLIP_TEXT in notes[0]["text"]
    assert np.max(np.abs(res.x["d_oil"][:, 0] - 150.0)) < 1e-8          # mechanical input passes through
    assert np.max(np.abs(res.x["d_epu"][:, 0] - notes[0]["used"])) < 1e-8


@needs_qpanel
def test_rarity_flags_an_extreme_combination(model, init):
    mild = F.rarity(model, init, 2, [S("d_ff", (1,), -0.25)])
    wild = F.rarity(model, init, 2, [S("d_gz", (1,), -2.5), S("d_unrate", (1,), 2.0), S("d_ip", (1,), 8.0)])
    assert mild["percentile"] < 0.99 < wild["percentile"] and wild["warnings"]
    assert "极少见" in wild["warnings"][0]


@needs_qpanel
def test_initial_state_imputes_unpublished_fields(model, init):
    assert init.z.shape == (2, model.n) and np.all(np.isfinite(init.z))
    assert any(i.endswith("@2026Q3") for i in init.imputed)            # Z.1 / profits not published yet
    assert init.ff0 == pytest.approx(3.75, abs=0.5)


@needs_qpanel
def test_unknown_field_raises_in_chinese(model, init):
    with pytest.raises(F.FactorError, match="字段不存在"):
        F.simulate(model, init, 2, n_paths=10, seed=0, statements=[S("d_nonsense", (1,), 1.0)])


@needs_qpanel
def test_the_draw_rule_holds_for_the_simulated_system(model):
    """P4b: every posterior draw is stationary as a VAR and under the cycle module, with every mode halving within
    the longest simulated horizon; the fit records the rule, the roots and the bounded scales."""
    info = model.info
    assert info["root_limit"] == pytest.approx(cq.FACTOR_MAX_ROOT) and info["draw_fallbacks"] == 0
    for A, _, Sig in model.draws:
        assert max(F.roots(A, Sig)) < cq.FACTOR_MAX_ROOT
    assert info["cycle_root_mean"] < cq.FACTOR_MAX_ROOT and 0.0 < info["rejection_rate"] < 1.0
    assert info["transform"]["lu"] == {"kind": "logit", "lo": 2.0, "hi": 30.0}
    assert len(model.draw_mu_ip) == len(model.draws)
    if cq.CYCLE_CENTER == "training_mean":                                      # P3's level (the default)
        assert model.cycle_shift(0) == 0.0 and model.cycle_shift(None) == 0.0
    else:
        assert model.cycle_shift(0) == pytest.approx(model.draw_mu_ip[0] - np.mean(model.ip_hist))


@needs_qpanel
def test_the_cycle_level_and_pool_winsorization_options(model, init, monkeypatch):
    """P4b options, off by default (config_q.CYCLE_CENTER, FACTOR_POOL_WINSOR_SD): the cycle module's IP blocks
    shifted to each draw's steady-state IP growth, and the bootstrap pool winsorized at a multiple of each coordinate's
    sd (frequency kept, size cut)."""
    import copy
    monkeypatch.setattr(cq, "CYCLE_CENTER", "steady_state")
    shift = model.cycle_shift(None)
    assert shift == pytest.approx(model.mu[F.IDX["ip"]] - np.mean(model.ip_hist))
    assert model.cycle_shift(3) == pytest.approx(model.draw_mu_ip[3] - np.mean(model.ip_hist))
    a = F.simulate(model, init, 8, n_paths=300, seed=2, param_uncertainty=False)
    monkeypatch.setattr(cq, "CYCLE_CENTER", "training_mean")
    b = F.simulate(model, init, 8, n_paths=300, seed=2, param_uncertainty=False)
    assert np.allclose(a.core[:, :, F.IDX["ip"]] - b.core[:, :, F.IDX["ip"]], shift)    # same blocks, new level
    monkeypatch.setattr(cq, "FACTOR_POOL_WINSOR_SD", 4.0)
    m2 = copy.copy(model)
    cut = F._build_pool(m2, model.seed)
    assert np.all(np.abs(m2.pool) <= 4.0 * model.pool.std(axis=0) + 1e-12)
    assert cut["core_entries"] > 0 and "2020Q2" in cut["rows"] and np.array_equal(m2.pool_labels, model.pool_labels)


@needs_qpanel
def test_long_horizon_paths_stay_finite_and_bounded(model, init):
    """P4b, permanent: 80 quarters of unconditional paths (the agent tree's snapshot length) stay finite and inside
    the bounded scales, the 10-year at or above its floor, the policy rate at or above its own. Far tails are rare:
    P3's log scales put 29% of 80-quarter paths beyond u 20, GZ 12 or Baa - 10y 10 (GZ up to 1.7 million)."""
    res = F.simulate(model, init, 80, n_paths=400, seed=8)
    lv = res.levels
    for k in ("unrate", "gz", "spread", "y10", "cpi_yoy", "ff"):
        assert np.isfinite(lv[k]).all(), k
    assert all(np.isfinite(v).all() for v in res.x.values())
    for lvl, v in (("unrate", "lu"), ("spread", "sp"), ("gz", "gz")):
        lo, hi = cq.FACTOR_BOUNDS[v]
        assert lo < lv[lvl].min() and lv[lvl].max() < hi, lvl
    assert lv["y10"].min() >= cq.Y10_FLOOR and lv["ff"].min() >= cq.FF_FLOOR - 1e-12
    far = (lv["unrate"] > 20).any(axis=1) | (lv["gz"] > 12).any(axis=1) | (lv["spread"] > 10).any(axis=1)
    assert far.mean() < 0.15 and (lv["cpi_yoy"] > 15).mean() < 0.01


@needs_qpanel
def test_the_ten_year_has_an_effective_lower_bound(model, init):
    """P4b: the observed 10-year is max(shadow, Y10_FLOOR) with ry10 the shadow real rate (the policy rate's
    treatment): a statement on the shadow below the floor holds on the shadow and the observed yield sits at the
    floor; d_y10 is the observed change."""
    st = [S("y10_shadow", (3,), -1.5), S("y10_shadow", (4,), -1.0)]
    res = F.simulate(model, init, 6, n_paths=300, seed=3, statements=st, saturate=False)
    lv, x = res.levels, res.x
    assert np.max(np.abs(lv["y10_shadow"][:, 2] + 1.5)) < 1e-8 and np.max(np.abs(lv["y10_shadow"][:, 3] + 1.0)) < 1e-8
    assert np.allclose(lv["y10"][:, 2:4], cq.Y10_FLOOR) and lv["y10"].min() >= cq.Y10_FLOOR
    obs0 = max(float(init.z[1][F.IDX["ry10"]] + init.pi0), cq.Y10_FLOOR)
    assert np.allclose(x["d_y10"], np.diff(np.concatenate([np.full((300, 1), obs0), lv["y10"]], axis=1), axis=1))


@needs_qpanel
def test_bounded_statements_are_delivered_inside_the_bounds(model, init):
    """P4b: a level stated beyond its ceiling is delivered just inside it; a level and an unchained unemployment
    change on the logit scale (Newton in the space the linear rows leave free) hold to 1e-8 on every path."""
    over = F.simulate(model, init, 3, n_paths=200, seed=4, statements=[S("gz", (2,), 25.0)], saturate=False)
    assert np.allclose(over.levels["gz"][:, 1], cq.FACTOR_BOUNDS["gz"][1] - F._LEVEL_EPS, atol=1e-6)
    st = [S("unrate", (2,), 14.0), S("d_unrate", (4,), -1.0), S("d_spread", (3,), 1.5)]
    res = F.simulate(model, init, 5, n_paths=300, seed=4, statements=st, saturate=False)
    assert np.max(np.abs(res.levels["unrate"][:, 1] - 14.0)) < 1e-8
    assert np.max(np.abs(res.x["d_unrate"][:, 3] + 1.0)) < 1e-8 and np.max(np.abs(res.x["d_spread"][:, 2] - 1.5)) < 1e-8
    assert res.info["projection"]["newton_err"] < 1e-8


@needs_qpanel
def test_policy_reaction_functions_and_shock_conventions(Q):
    from scenariolab.qmodel import policy
    for end in ("1994Q4", "2004Q4", "2014Q4", "2026Q3"):
        fits = policy.fit(Q, end)
        for f, ft in fits.items():
            ck = cq.REACTION[f]["check_r2"].get(end)
            if ck is not None:
                assert ft.r2 == pytest.approx(ck, abs=5e-4) and ft.n == cq.REACTION[f]["check_n"][end], (f, end)
    fits = policy.fit(Q, "2026Q3")
    U = policy.shocks(Q, fits)
    zlb = policy.zlb_quarters(Q).to_numpy(bool)
    assert np.all(U.loc[zlb, "u_ff"].fillna(0) == 0)
    assert np.all(U.loc[U.index < pd.Period(cq.BS_POLICY_START, "Q"), "u_bs"].fillna(0) == 0)
    assert set(policy.shock_sd(fits)) == set(cq.SHOCKS)


@needs_qpanel
def test_tf_history_values_reproduce_the_design(Q, P):
    from scenariolab.qmodel import factor_checks as FC
    h = FC.history(Q, P)
    ck = cq.TF_HISTORY_CHECK
    assert h["cpi_gt4"] == pytest.approx(ck["cpi_gt4"], abs=5e-4)
    assert np.allclose(h["cpi_acf4_windows"], ck["cpi_acf4_windows"], atol=5e-3)
    assert h["high_infl_spell_q"] == pytest.approx(ck["high_infl_spell_q"], abs=0.05)
    assert h["contractions_per_15y"] == pytest.approx(ck["contractions_per_15y"], abs=0.01)
    assert h["contraction_mean_q"] == pytest.approx(ck["contraction_mean_q"], abs=0.05)
    assert tuple(h["contractions_gt6q"]) == ck["contractions_gt6q"]
    assert h["bounce_dd"] == pytest.approx(ck["bounce_dd"], abs=5e-3)
    assert h["dff_on_du"] == pytest.approx(ck["dff_on_du"], abs=5e-3)
    assert h["okun"] == pytest.approx(ck["okun"], abs=5e-4)
    assert h["corr_dgz_useq"] == pytest.approx(ck["corr_dgz_useq"], abs=5e-3)
    assert h["corr_dspread_useq"] == pytest.approx(ck["corr_dspread_useq"], abs=5e-3)


needs_results = pytest.mark.skipif(not cq.QMODEL_RESULTS_PATH.exists(),
                                   reason="factor checks not run (python -m scenariolab.qmodel.factor_checks)")


@needs_results
def test_tf_and_tc_are_stored_with_pass_fail():
    from scenariolab import jsonio
    doc = jsonio.load(cq.QMODEL_RESULTS_PATH)
    fl = doc.get("factor_layer")
    if fl is None or fl.get("quick"):
        pytest.skip("qmodel.json has no full factor-layer run")
    tf = fl["T_F"]
    assert set(tf["fits"]) == set(cq.TF_ENDS)
    last = tf["fits"][cq.TF_ENDS[-1]]
    for k in ("F1", "F9", "F10"):
        assert isinstance(last[k]["pass"], bool)
    for end, r in tf["fits"].items():
        for k in ("F2_cpi_gt4", "F2_cpi_acf4", "F2_spell", "F3_per_15y", "F3_mean_q", "F3_gt6q", "F4_bounce",
                  "F5_dff_on_du", "F5_okun", "F6_bounds", "F6_tails", "F6_max13_unrate", "F6_max13_gz",
                  "F6_max13_spread"):
            assert k in r["ppc"] and "pass" in r["ppc"][k], (end, k)
        assert r["ppc"]["F6_tails"]["nonfinite"] == 0, end                      # P4b: no non-finite value
    assert "ppc_alt_transform_log" in last                                      # §15 Q5: the log arm alongside
    assert isinstance(tf["F7"]["pass"], bool) and isinstance(tf["F8"]["pass"], bool)
    assert last["F1"]["hard_max_abs_error"] < 1e-8                      # F1 hard delivery holds by construction
    assert last["F9"]["same_quarter_E0_unchanged"] is True
    tc = fl["T_C"]
    assert set(tc["test"]) == {"a", "b", "c"} and isinstance(tc["gate"]["pass"], bool)
    assert tc["selection"]["completion_w"] in cq.COMPLETION_W_GRID


# ================================================================ the fix round (statistics and engineering reviews)

@needs_qpanel
def test_the_bootstrap_pool_is_centred_so_the_simulated_mean_is_the_models(model, init):
    """The pool's whitened rows have mean 0 (the residuals at the posterior-mean model do not: (I - A1 - A2)^-1 made
    their small mean a long-run drift of CPI +1.0 and real oil +11 log% at the forward fit). The bootstrap's mean path
    then equals the model's zero-shock path within Monte Carlo error at 40 quarters, and the posterior-mean system's
    fixed point is the steady state mu."""
    assert np.max(np.abs(model.pool.mean(axis=0))) < 1e-12
    assert model.info["pool_winsorized"]["centred"] is True
    H, n = 40, 4000
    res = F.simulate(model, init, H, n_paths=n, seed=21, cycle=False, param_uncertainty=False, usd_hold=False,
                     saturate=False)
    mp = model.mean_path(init, H, model.system())[:, :F.NC]
    Y = res.core                                                        # (n, H, NC)
    z = (Y.mean(axis=0) - mp) / (Y.std(axis=0, ddof=1) / math.sqrt(n))
    assert np.max(np.abs(z)) < 4.5, np.unravel_index(np.argmax(np.abs(z)), z.shape)
    A1, A2 = model.A
    fixed = np.linalg.solve(np.eye(F.NC) - A1 - A2, model.c)
    assert np.allclose(fixed, model.mu, atol=1e-8)


@needs_qpanel
@pytest.mark.parametrize("var", ["d_unrate", "d_spread", "d_gz"])
def test_sums_and_means_of_changes_on_the_bounded_variables_are_delivered(model, init, var):
    """compile_statements passed no model to _urow for a mean, or a sum whose first quarter's previous level is not
    pinned: every such statement raised TypeError. Delivered to 1e-8 on every path, from q1 and from q > 1; the
    linear form (a sum from q1) also agrees with conditional_moments; the per-path forms are not linear and
    conditional_moments refuses them in Chinese."""
    v = {"d_unrate": 0.2, "d_spread": 0.3, "d_gz": 0.4}[var]
    cases = [S(var, (1, 2, 3), v, agg="mean"), S(var, (2, 3), 2 * v, agg="sum"), S(var, (1, 2), 2 * v, agg="sum"),
             S(var, (2, 3, 4), v, agg="mean")]
    for st in cases:
        res = F.simulate(model, init, 4, n_paths=300, seed=3, statements=[st], saturate=False)
        hard = [d for d in res.delivery if d["kind"] == "hard"]
        assert hard and hard[0]["max_abs_error"] < 1e-8, (st, hard)
    lin = S(var, (1, 2), 2 * v, agg="sum")                               # previous level pinned: a linear row
    mean, cov = F.conditional_moments(model, init, 2, [lin])
    g = F.simulate(model, init, 2, n_paths=4000, seed=5, statements=[lin], cycle=False, gaussian=True,
                   param_uncertainty=False, usd_hold=False, saturate=False)
    k = F.IDX[{"d_unrate": "lu", "d_spread": "sp", "d_gz": "gz"}[var]]
    se = np.sqrt(np.maximum(np.diag(cov).reshape(2, -1)[:, k], 0.0) / 4000)    # quarter 2 is pinned: se 0
    assert np.all(np.abs(g.core[:, :, k].mean(axis=0) - mean[:, k]) < 4.5 * se + 1e-8)
    with pytest.raises(F.FactorError, match="线性"):
        F.conditional_moments(model, init, 3, [S(var, (2, 3), v, agg="sum")])


@needs_qpanel
def test_saturation_is_four_training_sd_from_the_conditional_mean(model, init):
    """§3.5 as written: the gap between a stated value and its conditional mean saturates at ±4 training sd (the
    field's sd over the training quarters), not ±4 sd of a 400-path Gaussian prior with the t-weighted Sigma (which
    clipped 1-6% of the model's own draws). Deterministic; d_oil passes through; sums and levels use their own
    training sd."""
    mp = model.mean_path(init, 2, model.system())[None]
    x, lv, sh = F.outputs(model, init, mp, np.zeros_like(mp), np.full(1, -1), 2, usd_hold=False)
    for f in ("d_unrate", "d_ip", "d_epu", "d_profits"):
        st = S(f, (1,), 0.0)
        mu, sd = float(F.evaluate(st, x, lv, sh)[0]), F.statement_sd(model, st)
        train = Q_train_sd(model, f)
        assert sd == pytest.approx(train, rel=1e-12)
        vals, notes = F.saturation(model, init, 1, [S(f, (1,), mu + 3.9 * sd), S(f, (1,), mu - 7.0 * sd)])
        assert vals[0] == pytest.approx(mu + 3.9 * sd) and vals[1] == pytest.approx(mu - 4.0 * sd)
        assert [n["statement"] for n in notes] == [1] and F.CLIP_TEXT in notes[0]["text"]
    vals, notes = F.saturation(model, init, 1, [S("d_oil", (1,), 400.0)])
    assert vals == [400.0] and notes == []
    s2 = F.statement_sd(model, S("d_unrate", (1, 2), 0.0, agg="sum"))
    assert s2 == pytest.approx(float(model.train_frame["d_unrate"].rolling(2).sum().std()), rel=1e-12)
    lev = F.statement_sd(model, S("unrate", (3,), 5.0))
    assert lev == pytest.approx(float((model.train_frame["unrate"].shift(-3) - model.train_frame["unrate"]).std()))
    a, _ = F.saturation(model, init, 1, [S("d_unrate", (1,), 2.0)], seed=1)
    b, _ = F.saturation(model, init, 1, [S("d_unrate", (1,), 2.0)], seed=2)
    assert a == b                                                       # no Monte Carlo


def Q_train_sd(model, f):
    return float(model.train_frame[f].dropna().std())


@needs_qpanel
def test_the_initial_state_uses_only_data_published_by_the_origin(Q, P):
    """§3.2: at a backtest origin the previous quarter's CPI, IP and unemployment (with them pi_trend, the real rates
    and real oil) and the NIPA / Z.1 / profit satellites are not published yet. The initial state built from the
    full panel ('auto' = 'published' at a past origin) equals the one built from a panel rebuilt from the inputs as
    published at the end of the month before the origin (whose last row is that origin's 'today'), for a zero-bound
    and an ordinary origin; the asset layer's start state too."""
    from scenariolab.qmodel import assets as qa
    from scenariolab.qmodel import panel as qpanel
    m = F.fit(Q, "2014Q4", anchors="train_end", P=P, seed=0)
    inputs = qpanel.load_inputs()
    al = qa.AssetLayer.fit("2014Q4", panel=P.loc[:"2014-12"], Q=Q, n_boot=0)
    for origin, asof in (("2015Q1", "2014-12"), ("2016Q3", "2016-06")):
        Qt, _ = qpanel.build(qpanel.truncate_inputs(inputs, asof))
        assert Qt.index[-1] == pd.Period(origin, "Q")
        a, b = m.initial_state(Q, origin), m.initial_state(Qt, origin)
        assert a.information == "published" and b.information == "available"
        assert np.array_equal(a.z, b.z) and sorted(a.imputed) == sorted(b.imputed)
        for k in ("pi0", "ff0", "u0", "lvl_cpi0"):
            assert getattr(a, k) == getattr(b, k), k
        assert np.allclose(a.pc_hist, b.pc_hist, equal_nan=True) and np.allclose(a.core_hist, b.core_hist,
                                                                                 equal_nan=True)
        real = m.initial_state(Q, origin, information="available")
        assert len(a.imputed) > 30 and len(real.imputed) <= 4             # the realized values are not used
        assert not np.allclose(a.z[1], real.z[1])
        sa, sb = al.start_state(Q, origin), al.start_state(Qt, origin)
        assert set(sa) == set(sb)
        for k in sa:
            assert np.allclose(np.ravel(sa[k]), np.ravel(sb[k]), rtol=0, atol=0, equal_nan=True), k
    with pytest.raises(ValueError):
        m.initial_state(Q, "2015Q1", information="realized")


@needs_results
def test_qmodel_json_names_the_code_and_data_it_was_computed_from():
    """The stored T-F / T-C carry the content identity of the code and data they come from (qmodel.spec); a stale
    file (code or data changed since) fails here instead of passing unnoticed."""
    from scenariolab import jsonio
    from scenariolab.qmodel import factor_checks as fc
    fl = jsonio.load(cq.QMODEL_RESULTS_PATH).get("factor_layer") or {}
    if fl.get("quick"):
        pytest.skip("qmodel.json has no full factor-layer run")
    assert fl.get("version") == F.VERSION
    assert fl.get("identity") == fc.identity(), ("data/results/qmodel.json is stale: rebuild it (python -m "
                                                "scenariolab.build --only qmodel)")


# ================================================================ the identification of u_ff (the uff round)

def test_the_do_order_moves_only_the_named_variables_ahead_of_rff():
    """config_q.FACTOR_DO_ORDER moves variables ahead of rff in the do-operator's order only (their class is kept);
    today's recursive identification is the default and the response rule is off."""
    names = F.CORE + ["d_core", "d_deficit", "d_infexp1", "d_y2", "d_bs"]
    classes = [F.CORE_CLASS[v] for v in F.CORE] + ["E0", "E0p", "E2", "E1", "last"]
    rec = [names[i] for i in F.do_order_of(names, classes, "recursive")]
    assert rec == ["pc", "ps", "ip", "lu", "d_core", "rff", "d_deficit", "ro", "ry10", "sp", "gz", "usd", "d_infexp1",
                   "d_y2", "d_bs"]
    oil = [names[i] for i in F.do_order_of(names, classes, "oil_first")]
    assert oil.index("d_core") < oil.index("ro") < oil.index("rff")
    assert [v for v in oil if v != "ro"] == [v for v in rec if v != "ro"]
    exp = [names[i] for i in F.do_order_of(names, classes, "oil_expectations_first")]
    assert exp.index("ro") < exp.index("rff") and exp.index("d_infexp1") < exp.index("rff")
    assert cq.FACTOR_DO_ORDER == "recursive" and cq.FACTOR_IRF_RULE is None
    with pytest.raises(ValueError):
        F.do_order_of(names, classes, "nope")


def test_the_response_helpers_on_a_toy_system():
    """shock_response scales the impact to `size` and runs the lags; core y/y's response is the mean of the last
    four quarterly (annualized) responses."""
    r = np.zeros((6, 2))
    r[:, 1] = [4.0, 4.0, 4.0, 4.0, 0.0, 0.0]
    assert np.allclose(F.core_yoy_response(r, 1), [1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
    Pm = np.array([[2.0, 0.0], [1.0, 1.0]])
    sysm = {"Phi": [np.array([[0.5, 0.0], [0.2, 0.5]]), np.zeros((2, 2))], "P": Pm, "Sigma": Pm @ Pm.T}
    out = F.shock_response(sysm, 0, 3, size=1.0)
    assert np.allclose(out[0], [1.0, 0.5]) and np.allclose(out[1], [0.5, 0.45])


@needs_qpanel
def test_the_identification_switch_moves_only_the_structural_impact(model, init):
    """Moving real oil ahead of rff changes P only: Sigma_u, Phi and every reduced-form simulation stay the same, the
    u_ff column no longer moves oil on impact, and the fiscal shocks' columns do not change. A stated u_ff then leaves
    same-quarter real oil unmoved, as it leaves E0 macro."""
    import dataclasses
    s0, s1 = model.system(), model.system(identification="oil_first")
    assert np.array_equal(s0["Sigma"], s1["Sigma"]) and all(np.array_equal(a, b) for a, b in zip(s0["Phi"], s1["Phi"]))
    for sm in (s0, s1):
        assert np.allclose(sm["P"] @ sm["P"].T, sm["Sigma"], atol=1e-8)
    rff, ro = F.IDX["rff"], F.IDX["ro"]
    assert abs(s1["P"][ro, rff]) < 1e-12 < abs(s0["P"][ro, rff])
    j = model.names.index("d_deficit")
    assert np.allclose(s0["P"][:, j], s1["P"][:, j], atol=1e-10)
    m1 = dataclasses.replace(model, identification="oil_first", _sd_cache={})
    for stmts in ((), [S("d_ip", (1,), 0.5), S("d_cpi", (1, 2), 3.0), S("d_gz", (3,), 0.3, sd=0.1)]):
        a = F.simulate(model, init, 6, n_paths=200, seed=4, statements=stmts)
        b = F.simulate(m1, init, 6, n_paths=200, seed=4, statements=stmts)
        assert all(np.array_equal(a.x[f], b.x[f]) for f in a.x)
    kw = dict(n_paths=200, seed=4, cycle=False)
    base = F.simulate(m1, init, 4, **kw)
    hike = F.simulate(m1, init, 4, statements=[S("u_ff", (1,), 0.5)], **kw)
    assert np.max(np.abs(hike.x["d_oil"][:, 0] - base.x["d_oil"][:, 0])) < 1e-9
    assert np.max(np.abs(hike.x["d_y2"][:, 0] - base.x["d_y2"][:, 0])) > 1e-3


@needs_qpanel
def test_the_u_ff_response_at_the_forward_fit_and_the_response_rule(model, Q, P):
    """The uff round's finding at the forward fit: today's identification has the price puzzle in its lag
    coefficients (core CPI y/y rises after a +50bp u_ff, and ordering oil first does not change it), with the right
    impact signs (the 2-year yield and the dollar rise, same-quarter E0 macro does not move). The pre-registered
    response rule (FACTOR_IRF_RULE 'no_price_puzzle', the round's candidate a, not adopted) keeps only puzzle-free
    draws and leaves the point model alone."""
    pos, jc = model.names.index("rff"), model.names.index("d_core")
    for ident in ("recursive", "oil_first"):
        r = F.shock_response(model.system(identification=ident), pos, 16, 0.5)
        c = F.core_yoy_response(r, jc)
        assert abs(c[0]) < 1e-12 and c[7] > 0.0                       # the puzzle the agent tree's overlay corrects
        assert r[0, model.names.index("d_y2")] > 0 and r[0, F.IDX["usd"]] > 0
        assert np.allclose(r[0, [F.IDX[v] for v in ("pc", "ps", "ip", "lu")]], 0.0, atol=1e-12)
    assert not F.price_puzzle_free(model.system(), model.names)
    ma = F.fit(Q, "2026Q3", anchors="forward", P=P, seed=0, n_draws=6, irf_rule="no_price_puzzle")
    assert ma.info["irf_rule"] == "no_price_puzzle" and ma.info["draw_fallbacks"] == 0
    assert 0 < ma.info["irf_rule_acceptance"] < 0.2
    assert all(F.price_puzzle_free(ma.system(d), ma.names) for d in range(len(ma.draws)))
    assert np.array_equal(ma.A[0], model.A[0]) and np.array_equal(ma.Sigma, model.Sigma)
