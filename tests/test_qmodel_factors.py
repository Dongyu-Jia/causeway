"""The factor layer (SIM_DESIGN §5, §14 P3): qmodel/factors.py, qmodel/policy.py, qmodel/factor_checks.py.

P3 exit tests: the OLS limit of the BVAR (<= 1e-6); hard statements held to 1e-8 on every path; the Gaussian Monte
Carlo mean equals the analytic conditional mean (|z| < 3); a stated u_ff leaves same-quarter E0 macro exactly
unchanged; the observed policy rate never below 0.1; determinism by seed; T-F F1-F10 and T-C computed and stored in
data/results/qmodel.json with pass/fail shown.

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
    assert all(F.max_root(d[0]) < cq.FACTOR_MAX_ROOT for d in model.draws)
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
    assert np.allclose(x["d_y10"], d(k[:, :, F.IDX["ry10"]] + lv["pi_trend"], z1[F.IDX["ry10"]] + init.pi0))
    assert np.allclose(x["d_unrate"], d(lv["unrate"], init.u0))
    assert np.allclose(lv["unrate"], 2.0 + np.exp(k[:, :, F.IDX["lu"]]))
    assert np.allclose(x["d_ip"], k[:, :, F.IDX["ip"]] / 4)
    assert np.allclose(x["d_oil"], d(k[:, :, F.IDX["ro"]], z1[F.IDX["ro"]]) + np.diff(
        np.concatenate([np.full((200, 1), init.lvl_cpi0), lv["cpi"]], axis=1), axis=1))
    assert np.allclose(model.wg * x["d_cpi_goods"] + model.ws * x["d_cpi_serv"], x["d_cpi"])
    assert np.allclose(x["d_usd"], d(k[:, :, F.IDX["usd"]], z1[F.IDX["usd"]]))
    for f, v, lvl in (("d_spread", "sp", "spread"), ("d_gz", "gz", "gz")):
        shift = F.EXP_SHIFT.get(v)
        level = shift + np.exp(k[:, :, F.IDX[v]]) if shift is not None else k[:, :, F.IDX[v]]
        level0 = shift + np.exp(z1[F.IDX[v]]) if shift is not None else z1[F.IDX[v]]
        assert np.allclose(lv[lvl], level) and np.allclose(x[f], d(level, level0))
    assert lv["unrate"].min() > cq.FACTOR_U_SHIFT
    if "gz" in F.EXP_SHIFT:
        assert lv["gz"].min() > cq.SPREAD_SHIFT and lv["spread"].min() > cq.SPREAD_SHIFT


@needs_qpanel
def test_usd_is_held_at_no_change_when_unstated(model, init):
    res = F.simulate(model, init, 8, n_paths=500, seed=2)
    assert np.max(np.abs(res.x["d_usd"].mean(axis=0))) < 1e-10
    res2 = F.simulate(model, init, 8, n_paths=500, seed=2, statements=[S("d_usd", (1,), 3.0)])
    assert np.max(np.abs(res2.x["d_usd"][:, 0] - 3.0)) < 1e-8


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
                  "F5_dff_on_du", "F5_okun", "F6_bounds"):
            assert k in r["ppc"] and "pass" in r["ppc"][k], (end, k)
    assert isinstance(tf["F7"]["pass"], bool) and isinstance(tf["F8"]["pass"], bool)
    assert last["F1"]["hard_max_abs_error"] < 1e-8                      # F1 hard delivery holds by construction
    assert last["F9"]["same_quarter_E0_unchanged"] is True
    tc = fl["T_C"]
    assert set(tc["test"]) == {"a", "b", "c"} and isinstance(tc["gate"]["pass"], bool)
    assert tc["selection"]["completion_w"] in cq.COMPLETION_W_GRID
