"""Rollout and adapters of the quarterly stack (SIM_DESIGN §3.6, §3.7, §11; §14 P4): qmodel/rollout.py,
qmodel/adapter.py, qmodel/api.py.

P4 exit tests: the agent tree's validate_adapter passes (its own function when the agents package is importable,
else the replica in adapter.py); step / path consistency (KS on 10,000 draws per field, Bonferroni over the fields as
the factor layer's F10); the state-roll identities (the roll reproduces the factor layer's own stacked paths and the
asset layer's states, and the accounting identities hold); the T18 / F5 statistics (history inside the simulated
per-path [p05, p95]); the performance budgets (step mode N = 500 <= 1 s, path mode 40Q x 1,000 paths <= 10 s).

The step engine's algebra runs on synthetic systems anywhere. The model tests need the quarterly panel (python -m
scenariolab.qmodel.panel) and the monthly panel, and skip without them. Nothing downloads and nothing calls an LLM.
"""
import math
import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from scenariolab import config
from scenariolab.qmodel import adapter as QA
from scenariolab.qmodel import config_q as cq
from scenariolab.qmodel import factors as F
from scenariolab.qmodel import rollout as R

S = F.Statement
needs_qpanel = pytest.mark.skipif(not (cq.PANEL_Q_PATH.exists() and config.PANEL_PATH.exists()),
                                  reason="quarterly panel not built (python -m scenariolab.qmodel.panel)")
HIST_T18 = {"d_gz": -0.582, "d_spread": -0.599}          # quarterly corr with us_equity, 1985-2026 (n 167) [d01]


# ================================================================ synthetic: the step engine's algebra

def _system(n=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Sig = A @ A.T + n * np.eye(n)
    order = np.array([2, 0, 1, 5, 3, 4])[:n]
    Pd = np.linalg.cholesky(Sig[np.ix_(order, order)])
    P = np.zeros((n, n))
    P[np.ix_(order, order)] = Pd
    return {"P": P, "Pinv": np.linalg.inv(P), "Sigma": Sig, "L": np.linalg.cholesky(Sig), "order": order}


def test_reduced_rows_are_the_waggoner_zha_projection_in_the_sigma_metric():
    sysm = _system()
    rng = np.random.default_rng(1)
    m, n = 7, 6
    pred = rng.standard_normal((m, n))
    E = rng.standard_normal((m, n)) @ sysm["L"].T
    R1, R2 = rng.standard_normal(n), rng.standard_normal(n)
    t1, t2 = rng.standard_normal(m), rng.standard_normal(m)
    y = R._solve(sysm, pred, E, [(R1, t1), (R2, t2)], [], [])
    assert np.max(np.abs(y @ R1 - t1)) < 1e-10 and np.max(np.abs(y @ R2 - t2)) < 1e-10
    Rm = np.stack([R1, R2])
    Sg = sysm["Sigma"]
    y0 = pred + E
    wz = y0 + (np.column_stack([t1, t2]) - y0 @ Rm.T) @ np.linalg.inv(Rm @ Sg @ Rm.T) @ Rm @ Sg
    assert np.max(np.abs(y - wz)) < 1e-10


def test_an_intervention_moves_only_its_structural_shock_and_what_is_ordered_after_it():
    sysm = _system()
    order = sysm["order"]
    rng = np.random.default_rng(2)
    m, n = 5, 6
    pred = rng.standard_normal((m, n))
    E = rng.standard_normal((m, n)) @ sysm["L"].T
    pos = int(order[3])                                   # the 4th variable in the recursion
    o = np.zeros(n)
    o[pos] = 1.0
    tau = rng.standard_normal(m)
    y = R._solve(sysm, pred, E, [], [(pos, o, tau)], [])
    assert np.max(np.abs(y[:, pos] - tau)) < 1e-10
    before = order[:3]                                    # ordered before it: unchanged exactly
    assert np.array_equal(y[:, before], (pred + (E @ sysm["Pinv"].T) @ sysm["P"].T)[:, before])
    eps0, eps1 = E @ sysm["Pinv"].T, (y - pred) @ sysm["Pinv"].T
    other = [k for k in range(n) if k != pos]
    assert np.max(np.abs(eps1[:, other] - eps0[:, other])) < 1e-10


def test_intervention_and_observation_together_hold_exactly():
    sysm = _system(seed=3)
    order = sysm["order"]
    rng = np.random.default_rng(4)
    m, n = 6, 6
    pred = rng.standard_normal((m, n))
    E = rng.standard_normal((m, n)) @ sysm["L"].T
    pos = int(order[2])
    o = np.zeros(n)
    o[pos] = 1.0
    Rr = rng.standard_normal(n)
    td, tr = rng.standard_normal(m), rng.standard_normal(m)
    shock_pos = int(order[4])
    sv = rng.standard_normal(m)
    y = R._solve(sysm, pred, E, [(Rr, tr)], [(pos, o, td)], [(shock_pos, sv)])
    eps = (y - pred) @ sysm["Pinv"].T
    assert np.max(np.abs(y[:, pos] - td)) < 1e-10
    assert np.max(np.abs(y @ Rr - tr)) < 1e-10
    assert np.max(np.abs(sysm["P"][shock_pos, shock_pos] * eps[:, shock_pos] - sv)) < 1e-10


def test_censored_moments_match_monte_carlo():
    rng = np.random.default_rng(5)
    for mu, sd, a in ((0.3, 1.0, 0.1), (2.0, 0.5, 0.1), (-1.0, 0.7, 0.1)):
        x = np.maximum(mu + sd * rng.standard_normal(400_000), a)
        m1, s1 = R._censored_moments(np.array(mu), np.array(sd), a)
        assert abs(float(m1) - x.mean()) < 5e-3 and abs(float(s1) - x.std()) < 5e-3


def test_given_and_draws_replicas_take_and_concat():
    g = QA.Given.from_dict({"b": np.arange(4.0), "a": 1.0}, 4)
    assert g.ids == ("a", "b") and g.n == 4 and np.array_equal(g.take([3, 1]).values[:, 1], [3.0, 1.0])
    assert QA.Given.empty(3).values.shape == (3, 0)
    d = QA.Draws({"x": np.arange(4.0), "_q": np.arange(8.0).reshape(4, 2)}, {"a": np.ones(4)}, {"a": np.zeros(4)},
                 np.zeros(4), np.array(list("abcd"), dtype=object))
    both = QA.Draws.concat([d.take([0, 1]), d.take([2, 3])])
    assert both.n == 4 and np.array_equal(both.x["_q"], d.x["_q"]) and list(both.drawn) == list("abcd")


def test_agent_keys_follow_the_agent_design_tables():
    assert QA.agent_keys("d_ff", "E0p") == {"cls": "instrument", "synth": "none", "axis": None,
                                            "channel_class": "well", "lag_discipline": False, "key": False}
    assert QA.agent_keys("d_gz", "E1")["axis"] == "fin_conditions_axis"
    assert QA.agent_keys("d_gz", "E1")["synth"] == "partition"
    assert QA.agent_keys("d_metals_oil", "E1c")["cls"] == "event_signature"
    assert QA.agent_keys("d_gpr", "E2")["cls"] == "event"
    assert QA.agent_keys("d_unrate", "E0")["lag_discipline"] is True
    assert QA.agent_keys("d_tariff", "E0")["channel_class"] == "thin"
    assert QA.agent_keys("d_tips10", "E1")["cls"] == "financial_price"


# ================================================================ the model: fixtures

@pytest.fixture(scope="module")
def stack():
    return R.get_stack()


@pytest.fixture(scope="module")
def adapter(stack):
    return QA.QAdapter(stack)


# ================================================================ the agent tree's contract

@needs_qpanel
def test_validate_adapter_passes(adapter):
    assert QA.validate_adapter(adapter) == []
    assert adapter.stub is False and adapter.capabilities == frozenset({"structural", "rarity", "impute"})
    from scenariolab.qmodel import card as qcard
    assert list(adapter.x_fields) == qcard.x_fields(qcard.load_card()) and len(adapter.x_fields) == 52
    assert list(adapter.assets) == list(config.ASSET_IDS)
    assert QA.make_adapter().version == QA.make_adapter().version          # cached


@needs_qpanel
def test_field_info_has_the_contract_keys(adapter):
    fi = adapter.field_info()
    for f in adapter.x_fields:
        e = fi[f]
        for k in ("zh", "unit", "endo", "cls", "synth", "axis", "channel_class", "lag_discipline", "key", "hist"):
            assert k in e, (f, k)
        for k in ("p01", "p05", "p50", "p95", "p99", "min", "max", "sd_q", "sd_q_1985_2019"):
            assert math.isfinite(e["hist"][k]), (f, k)
    assert fi["x_oil_cause"]["model_value"] == {"demand": 0.0, "mixed": 0.5, "supply": 1.0}
    assert fi["d_ff"]["given_mode"] == "do" and fi["d_ff"]["structural_shock"] == "u_ff"
    assert fi["d_ff"]["level_floor"] == cq.FF_FLOOR and fi["d_tariff"]["given_mode"] == "do_last"
    for k in ("s_ff", "s_unrate", "s_core_yoy", "s_spread", "s_term", "s_infexp1", "s_rstar", "s_sahm", "s_cape"):
        assert fi[k]["cls"] == "state" and fi[k]["hist"] is not None, k
    assert adapter.sd("d_ff") > 0


@needs_qpanel
def test_root_state_is_the_factor_layers_start(stack, adapter):
    n = 300
    s = adapter.root_state(n)
    for k in QA.REQUIRED_STATE:
        assert k in s and np.asarray(s[k]).shape == (n,), k
    assert all(np.asarray(v).shape[0] == n for v in s.values())
    init, Q = stack.init, stack.Q
    ql = pd.Period(stack.origin, "Q") - 1
    assert np.allclose(s["s_ff"], init.ff0) and np.allclose(s["s_unrate"], init.u0)
    assert np.allclose(s["s_y10"], Q.at[ql, "lvl_y10"]) and np.allclose(s["s_spread"], Q.at[ql, "lvl_spread"])
    assert np.allclose(s["_z"][:, 1, :], init.z[1]) and np.allclose(s["_pi"], init.pi0)
    assert np.allclose(s["d_ff_l1"], Q.at[ql, "d_ff"]) and np.allclose(s["us_equity_l1"], Q.at[ql, "yt_us_equity"])
    counts = np.bincount(s["_draw"].astype(int), minlength=stack.n_draws)
    assert counts.max() - counts.min() <= 3                                   # balanced posterior draws
    s2 = adapter.root_state(n)
    assert all(np.array_equal(np.asarray(s[k]), np.asarray(s2[k])) for k in s)


@needs_qpanel
def test_pins_come_back_unchanged_and_draws_are_reproducible(adapter):
    n = 200
    s = adapter.root_state(n)
    rng = np.random.default_rng(1)
    pins = {"d_ff": -0.25 + 0.05 * rng.standard_normal(n), "d_tariff": np.full(n, 0.5),
            "d_metals_oil": np.full(n, -12.0), "d_unrate": np.full(n, 0.3), "x_oil_cause": np.ones(n),
            "d_gz": np.where(np.arange(n) % 2 == 0, 0.4, np.nan), "d_nonexistent": np.full(n, 7.0)}
    g = QA.GivenClass.from_dict(pins, n)
    d1 = adapter.complete(s, g, np.random.default_rng(9))
    d2 = adapter.complete(s, g, np.random.default_rng(9))
    for k, v in pins.items():
        ok = np.isfinite(v)
        assert np.max(np.abs(d1.x[k][ok] - v[ok])) < 1e-12, k
    assert all(np.array_equal(d1.y[a], d2.y[a]) for a in adapter.assets)
    assert all(np.array_equal(d1.x[f], d2.x[f]) for f in adapter.x_fields)
    assert "mult" in d1.y and d1.x["_q"].shape == (n, len(R.PRIV)) and len(d1.drawn) == n
    assert any("原样返回" in nt["text"] for nt in d1.notes)                    # the unknown id passed through
    s1 = adapter.update_state(s, d1)
    assert np.allclose(s1["s_ff"], s["s_ff"] + pins["d_ff"]) and np.allclose(s1["d_tariff_l1"], 0.5)
    assert np.allclose(s1["s_unrate"], s["s_unrate"] + 0.3)


@needs_qpanel
def test_tree_draws_take_and_concat_keep_the_state_roll(adapter):
    n = 120
    s = adapter.root_state(n)
    d = adapter.complete(s, None, np.random.default_rng(3))
    idx = np.arange(0, n, 3)
    sub = adapter.update_state({k: np.asarray(v)[idx] for k, v in s.items()}, d.take(idx))
    full = adapter.update_state(s, QA.DrawsClass.concat([d.take(np.arange(0, 60)), d.take(np.arange(60, n))]))
    for k, v in sub.items():
        assert np.allclose(np.asarray(v), np.asarray(full[k])[idx], equal_nan=True), k


# ================================================================ interventions, bounds, saturation

@needs_qpanel
def test_an_intervention_on_the_policy_rate_leaves_same_quarter_e0_unchanged(stack, adapter):
    n = 2000
    s = adapter.root_state(n)
    fi = adapter.field_info()
    e0 = [f for f in adapter.x_fields if fi[f]["endo"] == "E0"]
    free = adapter.complete(s, None, np.random.default_rng(11))
    cut = adapter.complete(s, QA.GivenClass.from_dict({"d_ff": np.full(n, -0.5)}, n), np.random.default_rng(11))
    assert max(float(np.max(np.abs(cut.x[f] - free.x[f]))) for f in e0) == 0.0
    assert np.max(np.abs(cut.x["d_ff"] + 0.5)) < 1e-12
    assert np.mean(cut.x["d_y2"] - free.x["d_y2"]) < 0                     # markets respond on impact
    assert np.mean(cut.y["us_equity"] - free.y["us_equity"]) > 0
    red = QA.QAdapter(stack, given_modes={})                                 # the same pin as an observation
    obs = red.complete(s, QA.GivenClass.from_dict({"d_ff": np.full(n, -0.5)}, n), np.random.default_rng(11))
    assert max(float(np.max(np.abs(obs.x[f] - free.x[f]))) for f in e0) > 0.01   # ... imputes macro


@needs_qpanel
def test_do_last_sets_a_lever_without_same_quarter_effects(adapter):
    n = 500
    s = adapter.root_state(n)
    free = adapter.complete(s, None, np.random.default_rng(12))
    tar = adapter.complete(s, QA.GivenClass.from_dict({"d_tariff": np.full(n, 2.0)}, n), np.random.default_rng(12))
    assert np.allclose(tar.x["d_tariff"], 2.0)
    assert all(np.array_equal(tar.x[f], free.x[f]) for f in adapter.x_fields if f != "d_tariff")
    assert all(np.array_equal(tar.y[a], free.y[a]) for a in tar.y)


@needs_qpanel
@pytest.mark.parametrize("mode", ["reduced", "do"])
def test_a_policy_rate_pinned_below_the_floor_is_delivered_at_it(stack, mode):
    stp = R.Stepper(stack, saturate=False)
    n = 400
    s = stp.root(n)
    d = stp.draw(s, {"d_ff": -s["s_ff"]}, np.random.default_rng(5), modes={"d_ff": mode})
    s1 = stp.roll(s, d)
    assert np.allclose(d.x["d_ff"], cq.FF_FLOOR - s["s_ff"]) and np.allclose(s1["s_ff"], cq.FF_FLOOR)
    assert np.all(s1["_z"][:, 1, F.IDX["rff"]] + s1["_pi"] <= cq.FF_FLOOR + 1e-9)    # the shadow rate at or below
    assert any(nt["text"] == R.FLOOR_NOTE for nt in d.notes) and np.all(d.priv[:, R.PV["floor_bound"]] == 1)


@needs_qpanel
def test_bounds_and_saturation(stack):
    stp = R.Stepper(stack, saturate=False)
    n = 300
    s = stp.root(n)
    d = stp.draw(s, {"d_unrate": np.full(n, -3.0)}, np.random.default_rng(6), modes={})
    assert np.allclose(stp.roll(s, d)["s_unrate"], cq.PIN_U_MIN) and any(nt["text"] == R.BOUND_NOTE for nt in d.notes)
    sat = R.Stepper(stack)
    a = sat.draw(s, {"d_gz": np.full(n, 10.0)}, np.random.default_rng(7), modes={})
    b = stp.draw(s, {"d_gz": np.full(n, 10.0)}, np.random.default_rng(7), modes={})
    assert np.allclose(a.x["d_gz"], 10.0) and any(F.CLIP_TEXT in nt["text"] for nt in a.notes)
    assert a.x["d_spread"].mean() < 0.5 * b.x["d_spread"].mean()           # completion used the clipped value


@needs_qpanel
def test_rarity_equals_the_factor_layers_and_is_chi_square_under_the_model(stack, adapter):
    stp = R.Stepper(stack, saturate=False)
    pins = {"d_cpi_goods": 5.0, "d_unrate": 0.4, "d_gz": 0.5, "d_ff": -0.5, "d_sent": -8.0}
    p3 = F.rarity(stack.factors, stack.init, 1, [S(f, (1,), v) for f, v in pins.items()])
    mine = stp.rarity(stp.root(2), {k: np.full(2, v) for k, v in pins.items()}, modes={})
    assert np.allclose(mine["m2"], p3["m2"], rtol=1e-8) and np.all(mine["df"] == 5) and p3["df"] == 5
    n = 4000
    s = stp.root(n)
    base = R.Stepper(stack, param_uncertainty=False, cycle=False, gaussian=True, usd_hold=False, saturate=False)
    d = base.draw(s, {}, np.random.default_rng(13))
    own = {f: d.x[f] for f in ("d_cpi_serv", "d_ip", "d_gz", "d_sent")}
    m2 = stp.rarity(s, own, modes={})["m2"]
    assert abs(float(np.mean(m2)) - 4.0) < 0.25                            # chi2(4) when the pins are the model's own
    assert np.all(adapter.rarity(adapter.root_state(3), None) == 0.0)        # no pins: nothing to explain


@needs_qpanel
def test_impute_redraws_only_the_unpublished_root_fields(stack):
    stp = R.Stepper(stack)
    s = stp.root(3000)
    m = stack.factors
    q_last = str(pd.Period(stack.origin, "Q") - 1)
    pos = [m.names.index(v.split("@")[0]) for v in stack.init.imputed
           if v.endswith(f"@{q_last}") and v.split("@")[0] in m.names and not v.startswith("rff")]
    out = stp.impute(s, np.random.default_rng(8))
    other = [i for i in range(m.n) if i not in pos]
    assert np.array_equal(out["_z"][:, 1, other], s["_z"][:, 1, other])
    if pos:
        z = out["_z"][:, 1, pos]
        assert np.all(z.std(axis=0) > 0)
        assert np.all(np.abs(z.mean(axis=0) - stack.init.z[1][pos]) < 4 * z.std(axis=0) / math.sqrt(3000) + 1e-9)


# ================================================================ step / path consistency (§3.7, F10)

def _ks_all(a: dict, b: dict) -> dict:
    out = {}
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        if np.std(x) < 1e-9 and np.std(y) < 1e-9:
            assert abs(x.mean() - y.mean()) < 1e-9, k
            continue
        out[k] = float(ks_2samp(x, y).pvalue)
    return out


@needs_qpanel
def test_path_mode_with_quarter_one_statements_matches_step_mode(stack):
    """Path mode with statements only in quarter 1 gives quarter 1 the same distribution as the step engine (KS on
    10,000 draws per field, p > 0.01 after Bonferroni over the fields, as the factor layer's F10)."""
    n = 10_000
    pins = {"d_cpi_goods": 3.0, "d_oil": 10.0, "d_unrate": 0.2, "d_gz": 0.3, "d_tariff": 0.3}
    roll = R.run_path(stack, [S(f, (1,), v) for f, v in pins.items()], H=4, n_paths=n, seed=11, states=False)
    stp = R.Stepper(stack)
    d = stp.draw(stp.root(n), {k: np.full(n, v) for k, v in pins.items()}, np.random.default_rng(5), modes={})
    a = {f: roll.fx.x[f][:, 0] for f in roll.fx.x}
    a.update({f"y:{k}": roll.ax.total[:, 0, R.IA[k]] for k in R.ASSETS})
    a["y:mult"] = roll.ax.y_mult[:, 0]
    b = dict(d.x)
    b.update({f"y:{k}": v for k, v in d.y.items()})
    b.pop("_q", None)
    ps = _ks_all(a, b)
    assert len(ps) >= 60
    assert min(ps.values()) > 0.01 / len(ps), sorted(ps.items(), key=lambda kv: kv[1])[:3]


@needs_qpanel
def test_chained_steps_match_path_mode_over_eight_quarters(stack):
    """Unconditional: the step engine chained eight quarters (its cycle and residual blocks carried in the state)
    against path mode, per field, cumulative returns and states (the dollar hold off in both: path mode holds the
    cross-path mean, step mode each particle's own conditional mean)."""
    n, H = 6000, 8
    roll = R.run_path(stack, [], H=H, n_paths=n, seed=31, usd_hold=False)
    stp = R.Stepper(stack, usd_hold=False)
    s = stp.root(n)
    rng = np.random.default_rng(32)
    cum = {k: 0.0 for k in R.ASSETS}
    for _ in range(H):
        d = stp.draw(s, {}, rng)
        s = stp.roll(s, d)
        for k in R.ASSETS:
            cum[k] = cum[k] + d.y[k]
    a = {f: roll.fx.x[f][:, -1] for f in roll.fx.x}
    a.update({f"cum:{k}": roll.ax.total[:, :, R.IA[k]].sum(axis=1) for k in R.ASSETS})
    a.update({f"state:{k}": roll.states[k][:, -1] for k in ("s_ff", "s_unrate", "s_cape", "s_vol", "s_spread")})
    b = dict(d.x)
    b.update({f"cum:{k}": v for k, v in cum.items()})
    b.update({f"state:{k}": s[k] for k in ("s_ff", "s_unrate", "s_cape", "s_vol", "s_spread")})
    b.pop("_q", None)
    ps = _ks_all(a, b)
    assert min(ps.values()) > 0.01 / len(ps), sorted(ps.items(), key=lambda kv: kv[1])[:3]


# ================================================================ the state roll Φ

@needs_qpanel
def test_state_roll_identities_in_path_mode(stack):
    stm = [S("d_ff", (1,), -0.25), S("d_gz", (2,), 0.4, sd=0.1), S("d_cpi_goods", (3,), 4.0)]
    vst = [{"quarters": [3, 4, 5, 6], "mode": "rerate", "direction": "-", "size": "明显"}]
    roll = R.run_path(stack, stm, vst, H=8, n_paths=600, seed=21)
    fx, ax, st = roll.fx, roll.ax, roll.states
    stp = R.Stepper(stack)
    s = R.root_state(stack, roll.n, draw_index=fx.draw_index, param_index=ax.param_index)
    z_prev = s["_z"][:, 1, :]
    for k in range(roll.T):                       # the roll reproduces the factor layer's own stacked paths
        s = stp.roll(s, R.path_draw(stack, fx, ax, k, z_prev))
        z_prev = s["_z"][:, 1, :]
        assert np.max(np.abs(z_prev - np.concatenate([fx.core[:, k], fx.sats[:, k]], axis=1))) < 1e-9
    d = lambda k: np.diff(st[k], axis=1)                                     # noqa: E731
    for lvl, f in (("s_ff", "d_ff"), ("s_unrate", "d_unrate"), ("s_spread", "d_spread"), ("s_gz", "d_gz"),
                   ("s_y10", "d_y10")):
        assert np.max(np.abs(d(lvl) - fx.x[f])) < 1e-10, lvl
    assert np.max(np.abs(st["s_term"] - (st["s_y10"] - st["s_ff"]))) < 1e-12
    assert np.max(np.abs(st["s_real_y10"] - (st["s_y10"] - st["s_infl_trend"]))) < 1e-12
    pi, pc = st["s_infl_trend"], fx.x["d_cpi"]
    assert np.max(np.abs(pi[:, 1:] - (pi[:, :-1] + F.ALPHA * (pc - pi[:, :-1])))) < 1e-10
    assert np.array_equal(st["s_zlb"][:, 1:], (st["s_ff"][:, 1:] < cq.ZLB_FF).astype(float))
    assert np.array_equal(st["s_core4"][:, 1:], (st["s_core_yoy"][:, 1:] > cq.CORE4).astype(float))
    assert np.max(np.abs(st["s_cpi_yoy"][:, 1:] - fx.levels["cpi_yoy"])) < 1e-9
    assert np.max(np.abs(st["s_cape"] - ax.log_cape)) < 1e-12 and np.max(np.abs(st["s_pd"] - ax.log_pd)) < 1e-12
    assert np.max(np.abs(st["s_vol"][:, :-1] - ax.s_vol)) < 1e-12
    assert np.max(np.abs(st["s_tb3"][:, 1:] - np.maximum(st["s_ff"][:, 1:] + stack.bill_spread, 0))) < 1e-12
    eq = R.IA["us_equity"]
    assert np.max(np.abs(ax.price[:, :, eq] - (ax.y_mult + ax.y_fund))) < 1e-12              # V-PRICE
    tot = np.log(np.exp(ax.price) + ax.income)
    tot[:, :, R.IA["cash"]] = ax.price[:, :, R.IA["cash"]]
    assert np.max(np.abs(tot - ax.total)) < 1e-12
    inc_us = stack.assets.income_k["us_equity"] * np.exp(-st["s_pd"][:, :-1]) / 4.0
    assert np.max(np.abs(inc_us - ax.income[:, :, eq])) < 1e-12
    assert np.max(np.abs(np.diff(st["s_oil_real"], axis=1) - (fx.x["d_oil"] - pc / 4) / 100)) < 1e-12
    assert np.max(np.abs(st["real_oil_d1"][:, 1:] - np.diff(st["s_oil_real"], axis=1))) < 1e-12
    y_n = ax.total[:, :, R.IA["nasdaq"]]
    assert np.max(np.abs(st["nasdaq_4q"][:, 4:] - np.stack([y_n[:, k - 4:k].sum(1) for k in range(4, 9)], 1))) < 1e-12
    att = R.attribution(roll)
    assert att["identity_max_abs"] < 1e-10 and att["assets"]["us_equity"]["valuation"]["mean"] < 0


@needs_qpanel
def test_state_roll_identities_in_step_mode(adapter):
    n = 400
    s = adapter.root_state(n)
    rng = np.random.default_rng(14)
    for _ in range(6):
        g = QA.GivenClass.from_dict({"d_ff": -0.1 + 0.2 * rng.standard_normal(n), "d_gz": np.full(n, 0.1)}, n)
        d = adapter.complete(s, g, rng)
        s1 = adapter.update_state(s, d)
        for lvl, f in (("s_ff", "d_ff"), ("s_unrate", "d_unrate"), ("s_spread", "d_spread"), ("s_gz", "d_gz"),
                       ("s_y10", "d_y10")):
            assert np.max(np.abs(s1[lvl] - s[lvl] - d.x[f])) < 1e-10, lvl
        z1 = s1["_z"][:, 1, :]
        sh = F.EXP_SHIFT
        assert np.max(np.abs(cq.FACTOR_U_SHIFT + np.exp(z1[:, F.IDX["lu"]]) - s1["s_unrate"])) < 1e-9
        assert np.max(np.abs(sh["gz"] + np.exp(z1[:, F.IDX["gz"]]) - s1["s_gz"])) < 1e-9
        assert np.max(np.abs(z1[:, F.IDX["ry10"]] + s1["_pi"] - s1["s_y10"])) < 1e-10
        assert np.max(np.abs(s1["s_cape"] - s["s_cape"] - d.y["mult"])) < 1e-12
        assert np.max(np.abs(s1["_lcpi"] - s["_lcpi"] - 100.0 * d.cpi_q)) < 1e-10
        assert np.max(np.abs(s1["us_equity_l1"] - d.y["us_equity"])) < 1e-15
        assert np.max(np.abs(s1["d_unrate_l2"] - s["d_unrate_l1"])) < 1e-15
        s = s1


@needs_qpanel
def test_the_asset_step_is_the_asset_layer_at_one_quarter(stack):
    stp = R.Stepper(stack)
    n = 300
    s = stp.root(n)
    d = stp.draw(s, {}, np.random.default_rng(3))
    al = stack.assets
    rng = np.random.default_rng(77)
    pidx = rng.integers(0, len(al.params), size=n)
    idx = al.pool.draw_index(n, 1, rng, cq.RESID_BLOCK)
    sim = al.simulate(stp.asset_frame(s, d.x), al.start_state(stack.Q, stack.origin), n_paths=n, seed=77,
                      cma=cq.CMA_FORWARD, block=cq.RESID_BLOCK, params=True)
    _, y, inc, ap = stp._assets(s, d.x, idx[:, 0], pidx)
    assert np.max(np.abs(np.column_stack([y[a] for a in R.ASSETS]) - sim.total[:, 0])) < 1e-12
    assert np.max(np.abs(np.column_stack([inc[a] for a in R.ASSETS]) - sim.income[:, 0])) < 1e-12
    assert np.max(np.abs(y["mult"] - sim.y_mult[:, 0])) < 1e-12 and np.max(np.abs(ap["y_dg"] - sim.y_dg[:, 0])) < 1e-12


# ================================================================ T18 / F5 and the budgets

@needs_qpanel
def test_t18_f5_history_inside_the_simulated_band(stack, adapter):
    """Quarterly corr(d_gz, us_equity) and corr(d_spread, us_equity) per simulated 13-year path: history (1985-2026)
    inside [p05, p95], in path mode and through the adapter's chained steps (the instruments' T18 as F5)."""
    roll = R.run_path(stack, [], H=52, n_paths=1000, seed=18, states=False)
    c = QA.correlations(roll.ax.price[:, :, R.IA["us_equity"]], roll.fx.x["d_gz"], roll.fx.x["d_spread"])
    for k, h in HIST_T18.items():
        assert c[k]["p05"] <= h <= c[k]["p95"], (k, c[k])
        assert c[k]["median"] < 0
    s = adapter.root_state(1000)
    rng = np.random.default_rng(19)
    us, gz, sp = [], [], []
    for _ in range(52):
        d = adapter.complete(s, None, rng)
        s = adapter.update_state(s, d)
        us.append(s["_y_eq_l1"])
        gz.append(d.x["d_gz"])
        sp.append(d.x["d_spread"])
    c2 = QA.correlations(np.column_stack(us), np.column_stack(gz), np.column_stack(sp))
    for k, h in HIST_T18.items():
        assert c2[k]["p05"] <= h <= c2[k]["p95"], (k, c2[k])


@needs_qpanel
def test_performance_budgets(stack, adapter):
    n = 500
    s = adapter.root_state(n)
    g = QA.GivenClass.from_dict({"d_ff": np.full(n, -0.25), "d_tariff": np.full(n, 0.3), "d_gpr": np.full(n, 20.0),
                                 "d_metals_oil": np.full(n, -10.0), "d_transfers": np.full(n, 0.2)}, n)
    adapter.complete(s, g, np.random.default_rng(0))                          # warm caches
    t0 = time.perf_counter()
    d = adapter.complete(s, g, np.random.default_rng(1))
    adapter.update_state(s, d)
    assert time.perf_counter() - t0 <= 1.0
    stm = [S("d_ff", (q,), -0.25) for q in range(1, 5)] + [S("d_gz", (q,), 0.3, sd=0.1) for q in (1, 2)] + \
          [S("d_cpi_goods", (q,), 3.0, sd=1.0) for q in range(1, 9)] + [S("d_unrate", (q,), 0.1) for q in (1, 2, 3)]
    t0 = time.perf_counter()
    roll = R.run_path(stack, stm, H=40, n_paths=1000, seed=2)
    assert time.perf_counter() - t0 <= 10.0
    assert roll.ax.total.shape == (1000, 40, 13) and roll.states["s_ff"].shape == (1000, 41)


# ================================================================ path mode, continuation and export

@needs_qpanel
def test_continuation_runs_the_model_on_without_statements(stack):
    roll = R.run_path(stack, [S("d_ff", (q,), -0.25) for q in (1, 2)], H=4, n_paths=300, seed=4, continuation=8)
    assert roll.T == 12 and roll.labels[0] == stack.origin and roll.ax.total.shape[1] == 12
    assert np.max(np.abs(roll.fx.x["d_ff"][:, :2] + 0.25)) < 1e-9 and np.std(roll.fx.x["d_ff"][:, 8]) > 0.05
    with pytest.raises(R.RolloutError):
        R.run_path(stack, [S("d_ff", (6,), 0.1)], H=4, n_paths=10, continuation=4)
    s, draws = R.continue_steps(stack, roll.end_state, 3, seed=1)
    assert len(draws) == 3 and np.asarray(s["s_ff"]).shape == (300,)


@needs_qpanel
def test_core_view_draws_map_onto_coreview_from_quarterly(stack):
    roll = R.run_path(stack, [], H=6, n_paths=200, seed=5)
    cv = QA.core_view_draws(roll)
    n, T = 200, 6
    assert cv["y_assets"].shape == (n, T, 13) and cv["inc_assets"].shape == (n, T, 13) and len(cv["labels"]) == T + 1
    total = np.expm1(cv["y_assets"])                                         # the mapping from_quarterly applies
    price = np.log1p(total - cv["inc_assets"])                               # (cash: all of its return is income)
    keep = [R.IA[a] for a in R.ASSETS if a != "cash"]
    assert np.max(np.abs(price[:, :, keep] - roll.ax.price[:, :, keep])) < 1e-9
    for lvl, dk in (("s_y10", "d_y10"), ("s_ff", "d_ff"), ("s_spread", "d_spread")):
        path = np.concatenate([cv[lvl][:, None], cv[lvl][:, None] + np.cumsum(cv[dk], axis=1)], axis=1)
        assert np.max(np.abs(path - roll.states[lvl])) < 1e-10, lvl
    assert np.allclose(cv["cpi"][:, 1:], np.exp(np.cumsum(cv["infl_q"], axis=1))) and cv["drawn"].shape == (n, T)
    assert cv["labels"][0] == (pd.Period(stack.origin, "Q") - 1).end_time.strftime("%Y-%m")
    try:
        core = QA.core_view(roll)
    except ImportError:
        return                                                               # instruments not merged yet
    assert core.freq == "Q" and core.n == n and core.resid_drawn.shape == (n, T)   # pragma: no cover


@needs_qpanel
def test_monthly_disaggregation_keeps_quarterly_totals(stack):
    roll = R.run_path(stack, [], H=4, n_paths=150, seed=6)
    cv = QA.core_view_draws(roll)
    mo = R.disaggregate(cv, stack, seed=1)
    assert mo["label"] == cq.DISAGG_LABEL and mo["total"].shape == (150, 12, 13) and len(mo["labels"]) == 13
    lm = np.log1p(mo["total"]).reshape(150, 4, 3, 13).sum(axis=2)
    assert np.max(np.abs(lm - cv["y_assets"])) < 1e-12
    assert np.max(np.abs(mo["income"].reshape(150, 4, 3, 13).sum(axis=2) - cv["inc_assets"])) < 1e-12
    assert np.allclose(mo["cpi"][:, ::3], cv["cpi"]) and np.allclose(mo["ff"][:, ::3], roll.states["s_ff"])
    assert all(len(m) == 7 for m in np.ravel(mo["drawn"])[:10])


# ================================================================ the QTable reading and the API

QT = {"schema": "qtable/1", "n_quarters": 4, "rows": [
    {"q": 1, "x": {"d_oil": {"value": 26, "sd": 8, "src": "stated", "quote": "油价一个季度内上涨约三成"},
                   "x_oil_cause": {"value": "supply", "src": "stated_qual", "quote": "产油国卷入战争，供应中断"},
                   "d_gpr": {"sign": "+", "size": "大幅", "src": "stated_qual", "quote": "地缘政治风险急剧上升"}},
     "fc": {"stance": "收紧", "quote": "信用利差走阔"}},
    {"q": 2, "x": {"d_ff": {"value": 0.5, "src": "stated", "quote": "央行加息 50 个基点"}}, "fc": {"stance": "未说明"}},
    {"q": 3, "x": {}, "u": {"u_ff": {"sign": "-", "size": "明显", "src": "stated_qual", "quote": "央行意外降息"}},
     "fc": {"stance": "中性", "quote": "金融条件大体平稳"}},
    {"q": 4, "x": {}, "fc": {"stance": "未说明"}}],
    "c_q": {"constraints": [{"kind": "range", "field": "d_unrate", "quarters": [1, 2, 3, 4], "min": -0.2,
                             "max": 0.3, "quote": "就业基本稳定"}]}}


@needs_qpanel
def test_from_qtable_reads_the_appendix_b_example(stack):
    r = R.from_qtable(QT, stack)
    st = {(s.var, s.quarters[0]): s for s in r["statements"]}
    assert r["H"] == 4 and st[("x_oil_cause", 1)].value == 1.0 and st[("d_oil", 1)].sd == 8
    assert st[("d_ff", 2)].value == 0.5 and st[("d_ff", 2)].sd is None
    assert st[("u_ff", 3)].value < 0 and st[("d_gz", 1)].value > 0 and st[("d_gz", 3)].value == 0.0
    # the W-FC warning is one note listing the quarters whose credit stance the model completes (P6: qtable.py)
    assert [n["quarters"] for n in r["notes"] if n.get("code") == "W-FC"] == [[2, 4]]
    assert any(s.var == "d_unrate" and s.lo == -0.2 and s.hi == 0.3 for s in r["statements"])
    with pytest.raises(R.RolloutError):
        R.from_qtable({**QT, "c_q": {"valuation": [{"quarters": [1], "mode": "hold"}]}}, stack, "test")
    with pytest.raises(R.RolloutError):
        R.from_qtable({"rows": [{"q": 1, "x": {"d_gz": {"sign": "+", "size": "巨大"}}}]}, stack)


@needs_qpanel
def test_api_card_and_run(stack):
    from fastapi.testclient import TestClient

    from scenariolab.server import app
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/q/card")
    assert r.status_code == 200 and r.json()["label"] == cq.TRIAL_LABEL and len(r.json()["rows"]) == 101
    r = client.post("/api/q/run", json={"qtable": QT, "n_paths": 300, "seed": 3})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    for _ in range(400):
        s = client.get(f"/api/q/run/{jid}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done", s
    res = s["result"]
    assert res["label"] == cq.TRIAL_LABEL and len(res["quarters"]) == 4
    assert res["factors"]["d_ff"]["p50"][1] == pytest.approx(0.5, abs=1e-9)
    assert res["attribution"]["identity_max_abs"] < 1e-10
    assert client.post("/api/q/run", json={"qtable": QT, "n_paths": 300, "seed": 3}).json()["id"] == jid
    assert client.post("/api/q/run", json={"qtable": QT, "mode": "nope"}).status_code == 400
    assert client.post("/api/q/run", json={"qtable": QT, "n_paths": 5}).status_code == 400
    bad = client.post("/api/q/run", json={"qtable": {"rows": [{"q": 1, "x": {"d_nope": {"value": 1}}}]}})
    assert bad.status_code == 400 and "字段不存在" in bad.json()["detail"]
    assert client.get("/api/q/run/0123456789ab").status_code == 404
    assert client.get("/api/q/run/BAD").status_code == 400
    step = client.post("/api/q/run", json={"qtable": QT, "n_paths": 200, "mode": "step"}).json()["id"]
    for _ in range(400):
        s = client.get(f"/api/q/run/{step}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done" and len(s["result"]["quarters"]) == 1


@needs_qpanel
def test_the_api_bands_leave_out_the_paths_the_model_flags_as_divergent(stack):
    """One rule on every page: the fans and tables of the scenario detail and the lab use the paths the forecast tab's
    mixture uses (tail_flags removed), and say how many were removed; clip notes name the field in Chinese."""
    roll = R.run_path(stack, [F.Statement("d_gz", (1,), 2.5)], [], H=8, n_paths=400, seed=3)
    keep, bands = R.band_mask(roll)
    tails = R.tail_flags(roll.fx.levels["gz"], roll.fx.levels["unrate"],
                         roll.ax.total[:, :, R.IA["us_equity"]]).any(axis=1)
    assert (keep == ~tails).all() and bands["n_used"] == int((~tails).sum())
    out = R.summary(roll)
    cum = np.cumsum(roll.ax.total[:, :, R.IA["us_equity"]], axis=1)[keep]
    assert np.allclose(out["assets"]["us_equity"]["cum_total"]["p50"], np.round(np.nanpercentile(cum, 50, axis=0), 6))
    assert out["info"]["bands"]["excludes_tail"] == bool(tails.any())
    for n in out["notes"]:
        f = n.get("field") or n.get("var")
        assert not (f and str(n.get("text", "")).startswith(f"{f}：")), n
