from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scenariolab import config, jsonio
from scenariolab.analysis import causality as C

# rows: oil, ip, cpi, ff, spread; columns: oil_supply, demand, monetary, financial, other
B_TRUE = np.array([
    [5.0, 3.0, -1.0, -0.5, 0.5],
    [-0.5, 0.6, -0.3, -0.4, 0.1],
    [0.4, 0.3, -0.2, -0.05, -0.5],
    [0.05, 0.15, 0.3, -0.2, -0.05],
    [0.02, -0.03, 0.08, 0.25, -0.05],
])


def _synthetic_var(T=2000, seed=0, rho=0.3):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((T, 5))
    y = np.zeros((T, 5))
    for t in range(1, T):
        y[t] = rho * y[t - 1] + B_TRUE @ e[t]
    idx = pd.date_range("1800-01-01", periods=T, freq="MS")
    return pd.DataFrame(y, index=idx, columns=C.VAR_IDS), pd.DataFrame(e, index=idx)


@pytest.fixture(scope="module")
def synthetic():
    data, e = _synthetic_var()
    var = C.fit_var(data, p=1, outliers=None)
    draws = C.sample_svar(var, np.random.default_rng(1), n_accept=200, horizon=12, param_uncertainty=False)
    impact = draws["impact"][C.median_target(draws["irf"])]
    return var, draws, impact, e


def test_true_impact_satisfies_restrictions_and_no_other_column_does():
    psi = np.stack([0.3 ** h * np.eye(5) for h in range(6)])
    R = C.to_reported(np.einsum("hij,jk->hik", psi, B_TRUE))[None]
    signs = C.column_signs(R, C.sign_masks())[0]          # (K, n_cols)
    assert (np.diag(signs[:, :4]) == 1).all()
    assert (signs[:, 4] == 0).all()                       # the unrestricted column fits no pattern


def test_to_reported_cumulates_flows_only():
    irf = np.ones((4, 5, 1))
    out = C.to_reported(irf)
    assert np.allclose(out[:, 0, 0], [1, 2, 3, 4])                   # oil: cumulative
    assert np.allclose(out[:, 2, 0], np.array([1, 2, 3, 4]) / 12)    # cpi: annualized -> price level
    assert np.allclose(out[:, 3, 0], 1.0)                            # ff: level


def test_sign_identification_recovers_pattern_and_shocks(synthetic):
    var, draws, impact, e = synthetic
    assert draws["n_accepted"] == 200 and 0 < draws["acceptance_rate"] < 1
    med0 = np.median(draws["irf"], axis=0)[0]                         # (n, K) at h = 0
    S = C.sign_masks()[:, 0, :]                                       # (K, n)
    for k in range(4):
        active = S[k] != 0
        assert (np.sign(med0[active, k]) == S[k, active]).all()
    assert med0[0, 2] < 0                                             # unrestricted: oil falls after tightening
    E = C.structural_shocks(var, impact)
    true = e.loc[E.index]
    corr = np.array([np.corrcoef(E[j], true[j])[0, 1] for j in range(4)])
    assert (corr > 0.4).all() and corr.mean() > 0.6


def test_structural_shocks_unit_variance_and_uncorrelated(synthetic):
    var, _, impact, _ = synthetic
    E = C.structural_shocks(var, impact).to_numpy()
    assert np.allclose(E.std(axis=0), 1.0, atol=1e-8)
    assert np.abs(np.corrcoef(E, rowvar=False) - np.eye(5)).max() < 1e-8


def test_outlier_rows_blank_in_shocks_and_split_in_decomposition():
    data, _ = _synthetic_var(T=400, seed=3)
    out = (str(data.index[200].date())[:7], str(data.index[203].date())[:7])
    var = C.fit_var(data, p=2, outliers=out)
    assert len(var.Y_all) - len(var.dates) == 6                      # 4 months + 2 rows using them as lags
    impact = np.linalg.cholesky(var.sigma)
    E = C.structural_shocks(var, impact)
    assert E.isna().any(axis=1).sum() == 6
    assert np.allclose(E.dropna().std(ddof=0), 1.0)
    raw = C.structural_shocks(var, impact, standardize=False)
    special = ~np.isin(raw.index, var.dates)
    hd = C.historical_decomposition(var, impact, raw, special=special)
    recon = hd["baseline"] + hd["contrib"].sum(axis=0)
    assert np.allclose(recon, hd["actual"], atol=1e-9)
    first = np.flatnonzero(special)[0]
    assert np.abs(hd["contrib"][-1][:first]).max() == 0               # outlier bucket starts at the outliers
    assert np.abs(hd["contrib"][-1][first]).max() > 0


def test_ma_coefs_and_stability():
    coef = np.zeros((1 + 5 * 2, 5))
    coef[1:6] = 0.5 * np.eye(5)
    psi = C.ma_coefs(coef, 2, 4)
    assert np.allclose(psi[3], 0.125 * np.eye(5))
    assert abs(C.max_root(coef, 2) - 0.5) < 1e-9


def _synthetic_quarterly(T=400, seed=5, beta=(0.5, 0.3, 0.2)):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1900-01-01", periods=T, freq="QS")
    s = rng.standard_normal(T)
    ff = np.zeros(T)
    for t in range(1, T):
        ff[t] = 0.9 * ff[t - 1] + 0.3 * rng.standard_normal()
    dc = beta[0] * s + 0.3 * rng.standard_normal(T)
    for j, b in enumerate(beta[1:], start=1):
        dc[j:] += b * s[:-j]
    Q = pd.DataFrame({
        "labor_share": np.cumsum(s + 0.2 * rng.standard_normal(T)),
        "gdp": np.cumsum(0.5 + rng.standard_normal(T)),
        "cpi": np.cumsum(0.5 + 0.3 * rng.standard_normal(T)),
        "ff": ff,
        "consumption": np.cumsum(dc),
    }, index=idx)
    return Q, pd.Series(s, index=idx)


def test_local_projection_recovers_known_cumulative_response():
    Q, s = _synthetic_quarterly()
    shock = C.labor_share_shock(Q, "1901-01", "1999-12", outliers=())
    assert np.corrcoef(shock, s.loc[shock.index])[0, 1] > 0.9
    r = C.local_projection(Q, shock, "consumption", horizon=6, outliers=())
    truth = np.array([0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.all(np.abs(r["coef"] - truth) < 3.5 * r["se"] + 0.05)
    assert np.all(np.isfinite(r["se"])) and r["n"][0] > r["n"][-1]


def test_quarter_mean_drops_partial_quarters():
    idx = pd.date_range("2000-01-01", "2000-10-01", freq="MS")          # Q4 has one month only
    q = C.quarter_mean(pd.Series(np.arange(len(idx), dtype=float), index=idx))
    assert list(q.index.strftime("%Y-%m")) == ["2000-01", "2000-04", "2000-07", "2000-10"]
    assert np.allclose(q.iloc[:3], [1.0, 4.0, 7.0]) and np.isnan(q.iloc[3])


def test_restriction_note_flags_imposed_signs():
    assert "识别约束" in C._restriction_note("monetary", "ip") and "0-5" in C._restriction_note("monetary", "ip")
    assert "未施加约束" in C._restriction_note("oil_supply", "ff")


def test_newey_west_zero_lags_is_white():
    rng = np.random.default_rng(7)
    X = np.column_stack([np.ones(300), rng.standard_normal(300)])
    y = X @ [1.0, 2.0] + rng.standard_normal(300) * (1 + np.abs(X[:, 1]))
    b, se = C.newey_west(y, X, 0)
    u = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    white = np.sqrt(np.diag(XtX_inv @ (X.T * u ** 2) @ X @ XtX_inv))
    assert np.allclose(se, white)


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built")
def test_run_on_real_panel_small():
    from scenariolab.data.panel import load_panel

    out = C.run(SimpleNamespace(panel=load_panel()), n_accept=60, write=False)
    sv = out["svar"]
    assert set(sv["n_accepted"]) == {p["id"] for p in C.PERIODS}
    assert all(v == 60 for v in sv["n_accepted"].values())
    for per in sv["irf"].values():
        for k, shock in enumerate(C.SHOCKS):
            for var, sign, h0, _ in shock["restrictions"]:
                assert np.sign(per[shock["id"]][var]["p50"][h0]) == sign
    hd = out["historical_decomposition"]
    total = np.asarray(hd["baseline"]) + sum(np.asarray(v) for v in hd["contrib"].values())
    assert np.allclose(total, hd["actual"], atol=1e-8)
    assert 5.0 < max(hd["actual"]) < 16.0                             # 1970s / 1980 inflation peak
    dist = out["distribution"]
    for per in dist["periods"]:
        ls = dist["responses"][per["id"]]["labor_share"]["coef"]
        assert abs(ls[0] + 1.0) < 1e-6                                # scaled to a 1% decline on impact
    b = out["asset_betas"]
    assert np.asarray(b["beta_std"]["full"]).shape == (len(b["assets"]), len(b["factors"]))
    assert b["r2"]["full"]["energy"] > 0.2
    assert len(out["chains"]) == 5
    for ch in out["chains"]:
        ids = {n["id"] for n in ch["nodes"]}
        for e in ch["edges"]:
            assert e["from"] in ids and e["to"] in ids and e["evidence"] and "n/a" not in e["evidence"]
    jsonio.dumps(out)
