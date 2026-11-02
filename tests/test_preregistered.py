"""The pre-registered records T-A, T-C and T-S and their runner (SIM_DESIGN §4.2, §4.6, §5.8, §14 P5).

Exit tests of P5:
  * the record exists before the run: its frozen part carries frozen_at and a hash, every result started after it,
    and, once the records are committed, the first committed version has no result (commit order checked here);
  * results are appended, never edited (per-entry hashes; the frozen part's hash; git history when committed);
  * the default asset layer is the one the T-A rule set (config_q.TA_DEFAULT = the record's decision);
  * the NN row is recorded (selection and test cells, its comparison with L0 and G_app, the re-entry rule).
The lifecycle and NN tests are synthetic and run anywhere; the record tests need the committed records (they skip
before P5 has frozen them); the data tests need the panels. Nothing downloads and nothing calls an LLM.
"""
import importlib.util
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from scenariolab import config
from scenariolab.qmodel import config_q as cq

P5_DATE = "2026-11-02"
needs_panels = pytest.mark.skipif(not (config.PANEL_PATH.exists() and cq.PANEL_Q_PATH.exists()),
                                  reason="monthly or quarterly panel not built")
RECORDS = {rid: cq.PREREG_DIR / f"{P5_DATE}_{rid}.json" for rid in ("TA", "TC", "TS")}
needs_records = pytest.mark.skipif(not all(p.exists() for p in RECORDS.values()),
                                   reason="the P5 records have not been frozen")


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location("tools_run_preregistered",
                                                  config.ROOT / "tools" / "run_preregistered.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_protocols(date):
    return {"TA": {"id": "T-A", "x": [1, 2.5, None]}, "TC": {"id": "T-C"}, "TS": {"id": "T-S", "date": date}}


# ================================================================ the record lifecycle (synthetic)

def test_canonical_hash_ignores_key_order_and_maps_nan(rp):
    assert rp.sha256_obj({"a": 1, "b": [1, 2]}) == rp.sha256_obj({"b": [1, 2], "a": 1})
    assert rp.jsonable({"x": float("nan"), "y": np.float64(2.0), "z": (1, 2)}) == {"x": None, "y": 2.0, "z": [1, 2]}


def test_freeze_writes_once_and_runs_append_exactly_once(rp, tmp_path):
    paths = rp.freeze("1999-01-01", tmp_path, _tiny_protocols("1999-01-01"))
    assert set(paths) == {"TA", "TC", "TS"}
    frozen_bytes = paths["TA"].read_bytes()
    rec = rp.load_record(paths["TA"])
    assert rec["results"] == [] and rp.check_record(rec) == []
    assert rec["frozen"]["frozen_at"] == rec["frozen_at"]
    assert rec["frozen"]["inputs"]["code"]["files"]["tools/run_preregistered.py"]
    with pytest.raises(SystemExit):                                   # a record is written once
        rp.freeze("1999-01-01", tmp_path, _tiny_protocols("1999-01-01"))
    e = rp.append_result(paths["TA"], "run", {"score": 0.5, "rows": [1, 2]}, rp.now(), {"matches": True})
    assert e["n"] == 1 and e["frozen_sha256"] == rec["frozen_sha256"]
    rec = rp.load_record(paths["TA"])
    assert rp.check_record(rec) == []
    assert rp.frozen_copy(rec).encode("utf-8") == frozen_bytes        # the frozen file rebuilt byte for byte
    with pytest.raises(SystemExit):                                   # each test runs once
        rp.append_result(paths["TA"], "run", {"score": 0.6}, rp.now(), {"matches": True})
    with pytest.raises(SystemExit):                                   # T-S takes status entries before P9
        rp.append_result(paths["TS"], "run", {"x": 1}, rp.now(), {})
    rp.append_result(paths["TS"], "status", {"evaluable": False}, rp.now(), {})
    rp.append_result(paths["TS"], "status", {"evaluable": False, "later": True}, rp.now(), {})
    assert rp.check_record(rp.load_record(paths["TS"])) == []


def test_edits_to_the_frozen_part_or_a_result_are_detected(rp, tmp_path):
    paths = rp.freeze("1999-01-01", tmp_path, _tiny_protocols("1999-01-01"))
    rp.append_result(paths["TC"], "run", {"w": 0.25}, rp.now(), {"matches": True})
    rec = rp.load_record(paths["TC"])
    rec["results"][0]["result"]["w"] = 1.0
    assert any("result_sha256" in p for p in rp.check_record(rec))
    rec = rp.load_record(paths["TC"])
    rec["frozen"]["protocol"]["id"] = "edited"
    assert any("frozen_sha256" in p for p in rp.check_record(rec))
    rec = rp.load_record(paths["TC"])
    rec["results"][0]["started_at"] = "1990-01-01T00:00:00+00:00"   # a result cannot predate the freeze
    rec["results"][0]["result_sha256"] = rp.sha256_obj(rec["results"][0]["result"])
    assert any("before the record was frozen" in p for p in rp.check_record(rec))
    with pytest.raises(SystemExit):                                   # a broken record takes no result
        p = paths["TA"]
        bad = rp.load_record(p)
        bad["frozen"]["protocol"]["x"] = [0]
        p.write_text(rp.dumps_record(bad), encoding="utf-8")
        rp.append_result(p, "run", {"y": 1}, rp.now(), {})


def test_the_inputs_check_names_a_changed_file(rp, tmp_path):
    inputs = rp.inputs_block()
    assert rp.verify_inputs(inputs) == []
    fake = {g: {"digest": v["digest"], "files": dict(v["files"])} for g, v in inputs.items() if g != "data"}
    fake["data"] = {"digest": inputs["data"]["digest"], "files": dict(inputs["data"]["files"])}
    fake["code"]["files"]["tools/run_preregistered.py"] = "0" * 64
    bad = rp.verify_inputs(fake)
    assert [b["file"] for b in bad] == ["tools/run_preregistered.py"]


# ================================================================ the NN port (synthetic)

def _tiny_nn(rp, din_s=3, din_x=4, dout_r=2, hidden=5):
    nn = rp.MaskedNN(hidden=hidden)
    nn.dsk = din_s + din_x
    nn.din = din_s + 2 * din_x
    nn.dout = dout_r + din_x
    return nn


def test_nn_gradient_matches_finite_differences(rp):
    nn = _tiny_nn(rp)
    rng = np.random.default_rng(0)
    th = nn._init(rng) + 0.1 * rng.standard_normal(sum(int(np.prod(s)) for _, s in nn._shapes()))
    X = rng.standard_normal((30, nn.din))
    Y = rng.standard_normal((30, nn.dout))
    W = (rng.random((30, nn.dout)) > 0.3).astype(float)
    f0, g = nn._loss_grad(th, X, Y, W, 0.3)
    eps = 1e-6
    num = np.array([(nn._loss_grad(th + eps * e, X, Y, W, 0.3)[0] - nn._loss_grad(th - eps * e, X, Y, W, 0.3)[0])
                    / (2 * eps) for e in np.eye(len(th))[::7]])
    assert np.allclose(num, g[::7], rtol=1e-4, atol=1e-7)


def test_nn_training_fits_a_linear_signal(rp):
    nn = _tiny_nn(rp, hidden=4)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, nn.din))
    B = rng.standard_normal((nn.dsk, nn.dout))
    Y = X[:, :nn.dsk] @ B
    th = nn._train(X, Y, np.ones_like(Y), 1e-4, np.random.default_rng(2))
    out, _ = nn._forward(nn._unpack(th), X)
    assert 1 - np.sum((out - Y) ** 2) / np.sum((Y - Y.mean(0)) ** 2) > 0.99   # the linear skip carries it


def test_nn_masks_keep_unavailable_fields_masked(rp):
    nn = _tiny_nn(rp)
    nn.e1 = np.array([False, True, True, False])
    nn.e1o = np.array([False, True, False, False])
    avail = np.ones((500, 4), bool)
    avail[:100, 3] = False
    M = nn._masks(avail, np.random.default_rng(3))
    assert not M[:100, 3].any() and M.shape == avail.shape
    assert 0.2 < M.mean() < 0.9                                       # random rates, never all or nothing


def test_nn_pool_scales_draws_with_volatility(rp):
    rng = np.random.default_rng(4)
    sv = np.exp(rng.normal(0, 0.5, 300))
    E = rng.standard_normal((300, 3)) * sv[:, None]
    pool = rp.NNPool(E, sv, [0, 1])
    assert 0.5 < pool.b <= 1.0
    lo = pool.draw(4000, np.array([np.median(sv) / 3]), rng)
    hi = pool.draw(4000, np.array([np.median(sv) * 3]), rng)
    assert lo.shape == (4000, 1, 3) and hi[..., 0].std() > 2 * lo[..., 0].std()


# ================================================================ T-A's models on data (no score is read)

@needs_panels
def test_joint_layer_is_the_asset_layer_for_one_candidate(rp):
    from scenariolab.data import panel as mpanel
    from scenariolab.qmodel import assets as qa
    from scenariolab.qmodel import panel as qpanel
    P, Q = mpanel.load_panel().loc[:"1998-12"], qpanel.load_panel().loc[:"2000Q4"]
    X = qa.oracle_inputs(Q, pd.period_range("1999Q1", "2000Q4", freq="Q"))
    J = rp.make_joint_layer()
    for c in ("C1", "C2"):
        a = qa.AssetLayer.fit("1998Q4", panel=P, Q=Q, n_boot=0, candidates=(c,))
        b = J.fit("1998Q4", panel=P, Q=Q, n_boot=0, candidates=(c,))
        st = a.start_state(Q, pd.Period("1999Q1", "Q"))
        sa = a.simulate(X, st, n_paths=20, seed=3, cma="historical", params=False).systematic
        sb = b.simulate(X, st, n_paths=20, seed=3, cma="historical", params=False).systematic
        assert np.array_equal(sa, sb)
    u = J.fit("1998Q4", panel=P, Q=Q, n_boot=0, candidates=("C1", "C2"))
    assert {k[0] for k in u.cand_fits} == {rp.JOINT}                    # one ridge on the union of terms
    assert {"d_igrea", "s_vol", "y10_x_pd"} <= set(u.cand_fits[(rp.JOINT, "us_equity")].names)
    base = J.fit("1998Q4", panel=P, Q=Q, n_boot=0)
    assert np.allclose(base.mean(X, cma="historical"), qa.AssetLayer.fit("1998Q4", panel=P, Q=Q, n_boot=0)
                       .mean(X, cma="historical"))


@needs_panels
def test_c3_under_the_harness_rule_sees_no_lead_after_the_window(rp):
    from scenariolab.data import panel as mpanel
    from scenariolab.evaluation import harness as eh
    from scenariolab.qmodel import panel as qpanel
    P, Q = mpanel.load_panel().loc[:"1998-12"], qpanel.load_panel().loc[:"2000Q4"]
    layer = rp.make_joint_layer().fit("1998Q4", panel=P, Q=Q, n_boot=0, candidates=("C3",))
    fill = rp.lead_fill_of(layer)
    w = rp.make_window_lead_fitted()(layer, Q, fill, cma="historical", name="t")
    o = eh.QuarterlyFitted(layer, Q, cma="historical")
    for months in (3, 12):
        Xw = w._inputs(pd.Timestamp("1999-01-01"), months)[2]
        Xo = o._inputs(pd.Timestamp("1999-01-01"), months)[2]
        assert Xw["d_ip_lead"].iloc[-1] == fill["d_ip_lead"] and Xw["d_unrate_lead"].iloc[-1] == fill["d_unrate_lead"]
        assert np.array_equal(Xw["d_ip_lead"].iloc[:-1], Xo["d_ip_lead"].iloc[:-1])     # inside: realized
        assert Xo["d_ip_lead"].iloc[-1] == Q.at[pd.Period("1999Q1", "Q") + months // 3, "d_ip"]


@needs_panels
def test_nn_fits_on_quarters_up_to_the_train_end_and_reads_no_returns_from_the_origin_on(rp):
    from scenariolab.qmodel import panel as qpanel
    Q = qpanel.load_panel()
    te, q0 = pd.Period("1994Q4", "Q"), pd.Period("1995Q1", "Q")
    a = rp.MaskedNN().fit(Q, te)
    Qp = Q.copy()
    after = Qp.index > te
    num = [c for c in Qp.columns if pd.api.types.is_numeric_dtype(Qp[c])]
    Qp.loc[after, num] = Qp.loc[after, num] * 1.7 + 0.3                  # everything after the train end changes
    b = rp.MaskedNN().fit(Qp, te)
    for ta, tb in zip(a.thetas, b.thetas):
        assert np.array_equal(ta, tb)
    NNF = rp.make_nn_fitted()
    base = NNF(a).simulate_paths(None, "1995-01-01", 12, 50, 5)
    # realized returns and asset-derived states from the origin on never reach the window's paths
    Qr = Q.copy()
    asset_cols = [c for c in Q.columns if c.startswith(("y_", "yt_", "inc_"))]
    later = [c for c in ("s_pd", "s_cape", "s_fcfy", "s_vol", "s_ecyc") + tuple(f"m1_{x}" for x in cq.ASSETS)
             + tuple(f"m4_{x}" for x in cq.ASSETS)]
    Qr.loc[Qr.index >= q0, asset_cols] = 9.0
    Qr.loc[Qr.index > q0, later] = -9.0
    c = rp.MaskedNN().fit(Qr, te)
    assert set(c.eq4_coef) == {"u_def", "u_ptax"}
    D_r, _ = rp.with_shocks(Qr, te)
    assert not np.allclose(D_r.loc["1995Q2":"1995Q4", "u_def"], a.D.loc["1995Q2":"1995Q4", "u_def"])
    again = NNF(c).simulate_paths(None, "1995-01-01", 12, 50, 5)
    # the realized shocks moved with the realized returns; the window's inputs did not (recomputed from the path)
    assert np.allclose(base, again) and np.isfinite(base).all()


# ================================================================ the P5 records

@needs_records
def test_every_record_is_intact(rp):
    out = rp.check_all()
    assert {f"{P5_DATE}_{rid}.json" for rid in RECORDS} <= set(out)
    assert {k: v["problems"] for k, v in out.items() if v["problems"]} == {}


@needs_records
def test_ta_and_tc_ran_once_after_freezing_and_ts_waits_for_p7(rp):
    for rid in ("TA", "TC"):
        rec = rp.load_record(RECORDS[rid])
        runs = [e for e in rec["results"] if e["kind"] == "run"]
        assert len(runs) == 1 and runs[0]["started_at"] > rec["frozen_at"]
        assert runs[0]["inputs_check"]["matches"] is True
        assert runs[0]["inputs_check"]["code_digest"] == rec["frozen"]["inputs"]["code"]["digest"]
    ts = rp.load_record(RECORDS["TS"])
    st = [e for e in ts["results"] if e["kind"] == "status"]
    assert st and all(e["result"]["evaluable"] is False and e["result"]["switch"] is False for e in st)
    assert any("P7" in m for m in st[-1]["result"]["missing"])


@needs_records
def test_the_default_asset_layer_is_the_t_a_decision(rp):
    import inspect

    from scenariolab.evaluation import harness as eh
    from scenariolab.qmodel import assets as qa
    run = [e for e in rp.load_record(RECORDS["TA"])["results"] if e["kind"] == "run"][0]["result"]
    decided = tuple(run["decision"]["ta_default"])
    assert tuple(cq.TA_DEFAULT) == decided
    assert inspect.signature(qa.AssetLayer.fit).parameters["candidates"].default == decided
    assert inspect.signature(eh.quarterly_model_fn).parameters["candidates"].default == decided
    gate = run["selection"]["gate"]
    assert set(gate) == {"C1", "C2", "C3"}
    assert run["decision"]["selected"] == [c for c in ("C1", "C2", "C3") if gate[c]["pass"]]
    if not run["decision"]["selected"]:
        assert decided == () and run["test"]["evaluation"]["adopted"] is False


@needs_records
def test_the_nn_row_is_recorded(rp):
    run = [e for e in rp.load_record(RECORDS["TA"])["results"] if e["kind"] == "run"][0]["result"]
    nn = run["nn_row"]
    assert nn["adopted"] is False and isinstance(nn["qualifies_for_reentry"], bool)
    for dec in ("1995-2004", "2005-2014"):
        for h in ("1q", "4q"):
            assert nn["selection"][dec][h]["r2"] is not None
            assert nn["vs_L0"]["selection"][dec][h]["lo90"] is not None
            assert nn["vs_G_app"]["selection"][dec][h]["lo90"] is not None
    assert nn["test"]["overlapping"]["2015-2026"]["1q"]["r2"] is not None
    assert nn["test"]["nonoverlap_4q"]["2015-2026"]["4q"]["n"] == cq.TA_TEST["n_4q_nonoverlap"]
    assert len(nn["fits"]) == 26                                       # 25 yearly selection fits + the test fit
    assert set(run["not_run"]) == {"NN-res", "C4"}


@needs_records
def test_the_frozen_constants_still_agree_with_config_q(rp):
    ta = rp.load_record(RECORDS["TA"])["frozen"]["protocol"]
    for k, v in cq.TA_GATE.items():
        assert ta["gate"][k] == v
    for k, v in cq.TA_TEST_RULE.items():
        assert ta["test_rule"][k] == rp.jsonable(v)
    assert ta["bootstrap"] == rp.jsonable(cq.TA_BOOT)
    assert {k: ta["harness"][k] for k in cq.TA_HARNESS} == rp.jsonable(cq.TA_HARNESS)
    assert ta["seeds"]["paired_bootstrap"] == cq.PREREG_SEED
    tc = rp.load_record(RECORDS["TC"])["frozen"]["protocol"]
    assert tc["seed"] == cq.TF_SEED and tc["draws"] == cq.TC_DRAWS and tc["w_grid"] == list(cq.COMPLETION_W_GRID)
    ts = rp.load_record(RECORDS["TS"])["frozen"]["protocol"]
    assert {c["margin"] for k, c in ts["conditions"].items() if k != "1"} == set(cq.TS_MARGINS.values())


@needs_records
def test_tc_records_its_selection_without_applying_it(rp):
    run = [e for e in rp.load_record(RECORDS["TC"])["results"] if e["kind"] == "run"][0]["result"]
    d = run["decision"]
    assert d["completion_w"] in cq.COMPLETION_W_GRID and isinstance(d["regime_cov_adopted"], bool)
    assert d["applied_to_config_q"] is False
    assert isinstance(run["tc"]["gate"]["pass"], bool) and run["tc"]["n_test_origins"] == 47


def _git(*args, cwd=None):
    try:
        r = subprocess.run(["git", *args], cwd=cwd or config.ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


SHALLOW = ("the clone is shallow: the commit-order check needs the full history (CI: actions/checkout with "
           "fetch-depth: 0; never squash-merge the record commits)")


def history_problems(root, relp: str, current: dict) -> list[str] | None:
    """Problems of a record's git history (None: the file is not committed): the first committed version must hold
    no result, and every committed version must keep the frozen part and hold a prefix of the current results. On a
    shallow clone (CI's default checkout has depth 1) the boundary commit looks like a root that adds every file, so
    the check reports that explicitly instead of a false "results first"."""
    # no --follow: a record is never renamed, and rename detection would take a similar record for this one
    revs = (_git("log", "--format=%H", "--", relp, cwd=root) or "").split()
    if not revs:
        return None
    if (_git("rev-parse", "--is-shallow-repository", cwd=root) or "").strip() == "true":
        return [f"{relp}: {SHALLOW}"]
    probs = []
    texts = [_git("show", f"{h}:{relp}", cwd=root) for h in reversed(revs)]
    if any(t is None for t in texts):
        return [f"{relp}: a commit deleted the record"]
    versions = [json.loads(t) for t in texts]
    if versions[0]["results"]:
        probs.append(f"{relp}: the first commit already holds results (commit the frozen copy first)")
    for v in versions:
        if v["frozen"] != current["frozen"] or v["frozen_sha256"] != current["frozen_sha256"]:
            probs.append(f"{relp}: the frozen part changed between commits")
        if v["results"] != current["results"][:len(v["results"])]:
            probs.append(f"{relp}: a committed result was edited or removed")
    return probs


needs_git = pytest.mark.skipif(_git("--version", cwd=".") is None, reason="git not available")


@needs_git
def test_the_history_check_accepts_frozen_first_and_rejects_results_first(rp, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "t@example.invalid"), ("config", "user.name", "t")):
        assert _git(*args, cwd=repo) is not None
    paths = rp.freeze("1999-01-01", repo, _tiny_protocols("1999-01-01"))
    rel = paths["TA"].name
    assert history_problems(repo, rel, rp.load_record(paths["TA"])) is None              # not committed: no verdict
    _git("add", rel, cwd=repo)
    _git("commit", "-q", "-m", "frozen", cwd=repo)
    rp.append_result(paths["TA"], "run", {"score": 1.0}, rp.now(), {"matches": True})
    _git("commit", "-q", "-am", "results", cwd=repo)
    assert history_problems(repo, rel, rp.load_record(paths["TA"])) == []                 # the right order
    rel_tc = paths["TC"].name                                                             # results in the first commit
    rp.append_result(paths["TC"], "run", {"w": 0.25}, rp.now(), {"matches": True})
    _git("add", rel_tc, cwd=repo)
    _git("commit", "-q", "-m", "tc with results", cwd=repo)
    assert any("first commit" in m for m in history_problems(repo, rel_tc, rp.load_record(paths["TC"])))
    rec = rp.load_record(paths["TA"])                                                    # an edited result
    rec["results"][0]["result"]["score"] = 2.0
    assert any("edited" in m for m in history_problems(repo, rel, rec))


@needs_git
def test_the_history_check_says_a_shallow_clone_cannot_decide(rp, tmp_path):
    """CI's default checkout (depth 1) would make every record look committed with its results at once: the check must
    say that it needs the full history rather than report a false 'results first'."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "t@example.invalid"), ("config", "user.name", "t")):
        assert _git(*args, cwd=repo) is not None
    paths = rp.freeze("1999-01-01", repo, _tiny_protocols("1999-01-01"))
    rel = paths["TA"].name
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "frozen", cwd=repo)
    rp.append_result(paths["TA"], "run", {"score": 1.0}, rp.now(), {"matches": True})
    _git("commit", "-q", "-am", "results", cwd=repo)
    assert history_problems(repo, rel, rp.load_record(paths["TA"])) == []
    shallow = tmp_path / "shallow"
    if _git("clone", "-q", "--depth", "1", f"file://{repo}", str(shallow), cwd=tmp_path) is None:
        pytest.skip("git cannot make a shallow clone here")
    probs = history_problems(shallow, rel, rp.load_record(shallow / rel))
    assert probs == [f"{rel}: {SHALLOW}"]


@needs_records
def test_commit_order_frozen_first_then_results_appended(rp):
    """Once committed, the first committed version of each record holds no result and every later committed version
    keeps the frozen part and extends the results (CI checks the commit order; untracked records skip)."""
    checked = 0
    for rid, p in RECORDS.items():
        relp = str(p.relative_to(config.ROOT))
        probs = history_problems(config.ROOT, relp, rp.load_record(p))
        if probs is None:
            continue
        assert probs == []
        checked += 1
    if not checked:
        pytest.skip("the records are not committed yet (land them frozen first: tools/run_preregistered.py "
                    "frozen-copy)")


# ================================================================ the fix round: the record machinery

def test_the_code_a_record_hashes_is_the_runs_import_closure(rp):
    """Not the whole package: files the T-A / T-C runs never import (the server, the translator, the build) cannot
    invalidate a record when they change (§12's expected conflicts)."""
    files = {rp.rel(p) for p in rp.code_files()}
    assert {"scenariolab/evaluation/protocol.py", "scenariolab/evaluation/harness.py",
            "scenariolab/qmodel/factor_checks.py", "scenariolab/qmodel/config_q.py",
            "tools/run_preregistered.py"} <= files
    assert not files & {"scenariolab/server.py", "scenariolab/llm/translator.py", "scenariolab/scenarios/schema.py",
                        "scenariolab/build.py"}


def test_verify_compares_the_closure_and_names_new_code(monkeypatch):
    from scenariolab.evaluation import protocol as pr
    inputs = pr.inputs_block()
    assert pr.verify_inputs(inputs) == []
    fake = {g: {"digest": v["digest"], "files": dict(v["files"])} for g, v in inputs.items() if g in ("code", "card",
                                                                                                    "data")}
    fake["code"]["files"]["scenariolab/server.py"] = "0" * 64           # frozen, but the runs do not import it
    assert pr.verify_inputs(fake) == []
    extra = config.ROOT / "scenariolab" / "build.py"
    real = pr.code_files
    monkeypatch.setattr(pr, "code_files", lambda root=config.ROOT: real(root) + [extra])
    bad = pr.verify_inputs(fake)
    assert [(b["file"], b["status"]) for b in bad] == [("scenariolab/build.py", "new")]


def test_timestamps_compare_as_instants_not_strings(rp):
    fa = "2026-11-02T20:00:00+00:00"
    frozen = {"record": "TC", "date": "1999-01-01", "frozen_at": fa}
    res = {"w": 0.25}
    entry = {"n": 1, "kind": "run", "started_at": "2026-11-02T12:30:00-08:00",      # 20:30 UTC: after the freeze
             "finished_at": "2026-11-02T12:31:00-08:00", "frozen_sha256": rp.sha256_obj(frozen), "result": res,
             "result_sha256": rp.sha256_obj(res)}
    rec = {"schema": rp.SCHEMA_ID, "record": "TC", "date": "1999-01-01", "title": "t", "frozen_at": fa,
           "frozen": frozen, "frozen_sha256": rp.sha256_obj(frozen), "results": [entry]}
    assert rp.check_record(rec) == []
    early = dict(entry, started_at="2026-11-02T11:59:00-08:00")            # 19:59 UTC: before it
    assert any("before the record was frozen" in p for p in rp.check_record({**rec, "results": [early]}))


def test_a_note_is_appended_and_counts_as_no_run(rp, tmp_path):
    paths = rp.freeze("1999-01-01", tmp_path, _tiny_protocols("1999-01-01"))
    frozen_bytes = paths["TA"].read_bytes()
    rp.append_result(paths["TA"], "run", {"score": 1.0}, rp.now(), {"matches": True})
    rp.append_result(paths["TA"], "note", {"text": "a disclosure"}, rp.now(), {"matches": None})
    rec = rp.load_record(paths["TA"])
    assert rp.check_record(rec) == [] and [e["kind"] for e in rec["results"]] == ["run", "note"]
    assert rp.frozen_copy(rec).encode("utf-8") == frozen_bytes
    with pytest.raises(SystemExit):
        rp.append_result(paths["TA"], "run", {"score": 2.0}, rp.now(), {"matches": True})


def test_one_record_can_be_frozen_again_and_names_the_one_it_supersedes(rp, tmp_path):
    old = rp.freeze("1999-01-01", tmp_path, _tiny_protocols("1999-01-01"))
    rp.append_result(old["TC"], "run", {"w": 0.25}, rp.now(), {"matches": True})
    new = rp.freeze("1999-01-01", tmp_path, {"TC": {"id": "T-C", "revision": 2}}, only=["TC"], tag="r2",
                    supersedes="1999-01-01_TC.json")
    assert set(new) == {"TC"} and new["TC"].name == "1999-01-01-r2_TC.json"
    rec = rp.load_record(new["TC"])
    orec = rp.load_record(old["TC"])
    assert rec["frozen"]["supersedes"]["record"] == "1999-01-01_TC.json"
    assert rec["frozen"]["supersedes"]["frozen_sha256"] == orec["frozen_sha256"] and rp.check_record(rec) == []
    assert "1999-01-01-r2_TC.json" in rp.check_all(tmp_path)
    with pytest.raises(SystemExit):                                     # a record supersedes one of its own kind
        rp.freeze("1999-01-01", tmp_path, {"TA": {"id": "T-A"}}, only=["TA"], tag="r3",
                  supersedes="1999-01-01_TC.json")
    with pytest.raises(SystemExit):                                     # and is written once
        rp.freeze("1999-01-01", tmp_path, {"TC": {"id": "T-C"}}, only=["TC"], tag="r2")


RECORD_TC2 = cq.PREREG_DIR / f"{P5_DATE}-r2_TC.json"


@pytest.mark.skipif(not RECORD_TC2.exists(), reason="the fix round's T-C record has not been frozen")
def test_the_fix_rounds_t_c_record_supersedes_the_frozen_one_and_ran_once(rp):
    rec = rp.load_record(RECORD_TC2)
    old = rp.load_record(RECORDS["TC"])
    assert rp.check_record(rec) == [] and rec["record"] == "TC"
    sup = rec["frozen"]["supersedes"]
    assert sup["record"] == RECORDS["TC"].name and sup["frozen_sha256"] == old["frozen_sha256"]
    proto = rec["frozen"]["protocol"]
    assert proto["scoring"] == "exact" and proto["revision"] == 2 and proto["seed"] == cq.TF_SEED
    assert proto["w_grid"] == list(cq.COMPLETION_W_GRID) and "published" in proto["information"]
    runs = [e for e in rec["results"] if e["kind"] == "run"]
    if not runs:
        pytest.skip("frozen, not run yet")
    assert len(runs) == 1 and rp.instant(runs[0]["started_at"]) > rp.instant(rec["frozen_at"])
    assert runs[0]["inputs_check"]["matches"] is True
    res = runs[0]["result"]
    assert res["decision"]["scoring"] == "exact" and res["decision"]["applied_to_config_q"] is False
    assert set(res["tc"]["gate"]["mc_se"].values()) == {0.0} and isinstance(res["tc"]["gate"]["pass"], bool)
    assert res["tc"]["n_test_origins"] == 47


@needs_records
def test_the_t_a_record_carries_the_fix_rounds_disclosure_note(rp):
    rec = rp.load_record(RECORDS["TA"])
    notes = [e for e in rec["results"] if e["kind"] == "note"]
    if not notes:
        pytest.skip("no note appended yet")
    body = notes[-1]["result"]
    assert "cov80" in body["text"] and body["known_before_freeze"]["cov80_1q_L0q_test"] == pytest.approx(0.8894,
                                                                                                         abs=1e-3)
