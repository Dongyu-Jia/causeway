"""The agent tree engine, snapshot mode, jobs and endpoints (AGENT_DESIGN §6–§9, §11–§12; E0 invariants).

Every LLM interaction goes through the mock backend (tests/conftest.py: MSL_LLM_BACKEND=mock, the real CLI blocked).
The quarterly model is the Gaussian stub (scenariolab/agents/stub_model.json)."""
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from scenariolab.agents import accounts as AC
from scenariolab.agents import jobs
from scenariolab.agents import model_adapter as MA
from scenariolab.agents import snapshot as S
from scenariolab.agents import synthesizer as SY
from scenariolab.agents import tree as T
from scenariolab.llm import mock as llm_mock

HAS_ROOT = AC.load_root() is not None
needs_root = pytest.mark.skipif(not HAS_ROOT, reason="needs data/results/agents_accounts_root.json "
                                                     "(python -m scenariolab.build --only agents_accounts)")


def small(**kw) -> T.TreeSettings:
    base = dict(depth=3, B=3, n_particles=120, max_calls=400, concurrency=1, prior_trees=False)
    base.update(kw)
    return T.TreeSettings.coerce(base)


def run_tree(tmp: Path, name: str, st: T.TreeSettings, cache: Path | None = None, **kw):
    store = T.Store(tmp / name)
    run = T.TreeRun(st, store, cache_dir=cache or (tmp / "cache"), **kw)
    return run, store, run.run()


def records(store: T.Store) -> dict[int, list]:
    return {d: store.read_level(d)["nodes"] for d in store.levels()}


# ---------------------------------------------------------------------------------------------------- the adapter

def test_stub_adapter_contract_and_saturation():
    m = MA.stub()
    assert MA.validate_adapter(m) == []
    s = m.root_state(200)
    assert all(k in s for k in MA.REQUIRED_STATE)
    g = MA.Given.from_dict({"d_ff": np.full(200, -0.25)}, 200)
    d1 = m.complete(s, g, np.random.default_rng(3))
    d2 = m.complete(s, g, np.random.default_rng(3))
    assert np.allclose(d1.x["d_ff"], -0.25)
    assert np.array_equal(d1.y["us_equity"], d2.y["us_equity"])
    # a pin 40 sd away is saturated at 4 sd in the conditional mean (no absurd linear extrapolation)
    big = m.complete(s, MA.Given.from_dict({"d_tariff": np.full(200, 5.0)}, 200), np.random.default_rng(3))
    assert abs(float(np.mean(big.y["us_equity"]))) < 0.15
    s1 = m.update_state(s, d1)
    assert np.allclose(s1["s_ff"], np.maximum(s["s_ff"] - 0.25, 0))
    assert np.allclose(s1["d_ff_l1"], -0.25)


def test_validate_adapter_reports_problems():
    class Broken:
        version, stub, capabilities, x_fields, assets = "b", True, frozenset({"telepathy"}), ("d_ff",), ("us_equity",)

        def field_info(self):
            return {}

        def root_state(self, n):
            return {"s_ff": np.zeros(n)}

        def complete(self, state, given, rng, *, structural=None):
            n = len(state["s_ff"])        # ignores the pin and the rng it is given
            return MA.Draws({"d_ff": np.zeros(n)}, {"us_equity": np.random.default_rng().standard_normal(n)})

        def update_state(self, state, draws):
            return state
    probs = MA.validate_adapter(Broken())
    assert any("未知的能力" in p for p in probs)
    assert any("起点状态缺少" in p for p in probs)
    assert any("固定的字段" in p for p in probs)
    assert any("可复现" in p for p in probs)


# ---------------------------------------------------------------------------------------------------- numerics

def test_allocation_resampling_and_forward_selection():
    c = T.allocate([0.5, 0.3, 0.2], 101)
    assert c.sum() == 101 and list(c) == [51, 30, 20]
    rng = np.random.default_rng(0)
    assert np.array_equal(T.resample_rows(np.arange(7), 7, rng), np.arange(7))   # equal weights: identity
    idx = T.systematic(np.array([0.0, 1.0, 0.0]), 5, rng)
    assert set(idx.tolist()) == {1}
    p = np.array([0.4, 0.3, 0.2, 0.1])
    D = np.abs(np.subtract.outer(np.array([0.0, 0.1, 5.0, 5.2]), np.array([0.0, 0.1, 5.0, 5.2])))
    J, a = T.forward_select(p, D, 2)
    assert J == [1, 2] and list(a) == [1, 1, 2, 2]      # the far, light cluster survives (mass × distance)


def _toy_nodes(n_particles=50, k=6):
    m = MA.stub()
    out = []
    for i in range(k):
        st = m.root_state(n_particles)
        st["s_unrate"] = st["s_unrate"] + (0.0 if i < 2 else 0.6 * i)
        cl = T.Cloud(st, None, {}, {}, {"us_equity": np.full(n_particles, 0.01 * i)}, np.zeros(n_particles), None, {})
        nd = T.Node(key=f"{i:016x}", parent="0" * 16, level=1, quarter="2026Q4", p_path=1.0 / k, cloud=cl,
                    facts={"oil_disruption_ongoing": True})
        out.append(nd)
    return out


def test_reduction_conserves_mass_and_reassigns_particles():
    nodes = _toy_nodes()
    kept, pruned, stats = T.reduce_level(nodes, 3, eps=0.25, fact_penalty=3.0, n=50, seed=1)
    assert len(kept) == 3 and len(pruned) == 3
    assert abs(sum(n.p_path for n in kept) - 1.0) < 1e-12
    assert abs(stats["mass"] - 1.0) < 1e-12 and stats["transport_error"] >= 0
    assert stats["recombined"] >= 1                         # nodes 0 and 1 are identical: recombined
    keys = {n.key for n in kept}
    for p in pruned:
        assert p.absorbed_into in keys and p.status == "pruned" and p.reason and p.cloud is None
    for n in kept:
        assert n.cloud.n == 50
        assert abs(n.p_path - n.own_mass - sum(a["p"] for a in n.absorbed)) < 1e-6
    # differing facts are never recombined
    nodes = _toy_nodes(k=2)
    nodes[1].facts = {}
    kept, pruned, stats = T.reduce_level(nodes, 2, eps=0.25, fact_penalty=3.0, n=50, seed=1)
    assert len(kept) == 2 and stats["recombined"] == 0


# ---------------------------------------------------------------------------------------------------- the cell and the tree

def test_agent_tree_invariants(tmp_path):
    run, store, res = run_tree(tmp_path, "a", small())
    recs = records(store)
    assert sorted(recs) == [0, 1, 2, 3]
    for d, rs in recs.items():
        kept = [r for r in rs if r["status"] != "pruned"]
        assert abs(sum(r["p_path"] for r in kept) - 1.0) < 1e-9, d              # every level's mass is 1
        for r in kept:
            if d < 3:
                assert r["children"] and abs(sum(c["p_cond"] for c in r["children"]) - 1.0) < 1e-6
                assert {c["status"] for c in r["children"]} <= {"expanded", "leaf", "pruned"}
            if HAS_ROOT and d > 0:
                assert r["summary"]["identities"]["ok"], (d, r["key"], r["summary"]["identities"])
                assert r["summary"]["accounts"], "the books move with the particles along every branch"
        for r in rs:
            if r["status"] == "pruned":
                assert r["absorbed_into"] and r["reason"]
    # the root: separate calls (one per active LLM actor) and the synthesizer, all from the mock
    det = store.read_level(0, "detail")["details"]
    root_det = next(iter(det.values()))
    actor_calls = [c for c in root_det["calls"] if c["kind"] == "actor"]
    assert len(actor_calls) == len(root_det["next"]) >= 6
    assert all(c["backend"] == "mock" for c in root_det["calls"])
    assert any(c["kind"] == "synth" for c in root_det["calls"])
    assert all(a["actor"] in root_det["packets"] for a in root_det["next"])     # every actor saw its packet
    if HAS_ROOT:
        assert "你的账户" in root_det["packets"]["fed"]["text"]
    # below the root: one merged call per expanded node; private notes never rendered there
    lv1 = store.read_level(1, "detail")["details"]
    for d1 in lv1.values():
        assert [c["kind"] for c in d1["calls"]].count("node") == 1
    assert [c for c in llm_mock.calls() if c.get("kind") == "node"]
    assert res["calls_new"] == run.llm.stats["calls_new"] > 0


@needs_root
def test_books_follow_each_particles_own_actions(tmp_path, monkeypatch):
    """Every particle's books are stepped with its own options' account actions (review E12): per particle, the
    actions the books receive equal actors.account_actions of that particle's own options."""
    from scenariolab.agents import actors as A
    seen = {"children": 0, "particles": 0, "keys": set()}
    orig = T.Engine.advance

    def advance(self, nd, d, src, assign, attr, dec):
        acts = AC.per_particle_actions(self.account_columns(nd, dec, assign, attr), len(src))
        for i in range(0, len(src), 3):
            pays = []
            for aid in dec["order"]:
                ab = dec["actors"][aid]
                pays.append(self._opt_payload(aid, ab["options"][int(assign[aid][i])], nd))
                for dd in ab["attributes"]:
                    k = (aid, dd["domain"])
                    if k in attr:
                        pays.append(self.payload(aid, dd["options"][int(attr[k][i])]["id"]))
            own = A.account_actions(pays)
            booked = {k: v[i] for k, v in acts.items() if v[i] is not None}
            assert booked == own, (i, booked, own)
            seen["particles"] += 1
            seen["keys"] |= set(own)
        seen["children"] += 1
        return orig(self, nd, d, src, assign, attr, dec)
    monkeypatch.setattr(T.Engine, "advance", advance)
    run_tree(tmp_path, "books", small(depth=2, n_particles=200))
    assert seen["children"] > 20 and seen["particles"] > 600
    assert {"bt_capex_plan_pct", "opec_quota_chg_mbd", "fed_bs"} <= seen["keys"]


def test_accounts_step_per_particle_equals_stepping_each_group():
    root = AC.load_root()
    if root is None:
        pytest.skip("needs the root account snapshot")
    n = 6
    acc = AC.Accounts.from_root(root, n)
    m = MA.stub()
    d = m.complete(m.root_state(n), MA.Given.empty(n), np.random.default_rng(1))
    y = {a: d.y[a] for a in AC.ACCOUNT_ASSETS if a in d.y}
    groups = [{"fed_bs": "qe", "bt_capex_plan_pct": 15.0, "bt_new_guidance": True, "debt_limit_raise_bn": 2500.0},
              {"spr_release_mbd": 0.75, "cny_weaker": True, "bt_buyback_mult": 0.5},
              {}]
    code = np.array([0, 1, 2, 0, 1, 2])
    per = AC.per_particle_actions([(code, groups)], n)
    a1 = acc.take(np.arange(n))
    a1.step(d.x, y, per)
    for g in range(3):
        ix = np.flatnonzero(code == g)
        a2 = acc.take(ix)
        a2.step({k: v[ix] for k, v in d.x.items()}, {k: v[ix] for k, v in y.items()}, groups[g])
        for k, v in a2.s.items():
            assert np.allclose(a1.s[k][ix], v), k
    assert a1.ok()


def test_discrete_split_keeps_minority_states(tmp_path):
    """Review E7: in the election quarter the voter does not branch, yet the children carry 'House holds' with its
    pooled mass (before the fix every child carried the majority option's discrete state)."""
    run, store, _ = run_tree(tmp_path, "split", small(depth=1, n_particles=500))
    root = store.read_level(0)["nodes"][0]
    det = store.read_level(0, "detail")["details"][root["key"]]
    recs = {r["key"]: r for r in store.read_level(1)["nodes"]}
    voter = next(a for a in det["next"] if a["actor"] == "voter")
    assert not voter["branching"]
    flips = next(o["p_used"] for o in voter["options"] if o["id"] == "house_flips")
    got = sum(c["p_cond"] for c in root["children"] if recs[c["key"]]["political"].get("house_next") == "flip")
    assert 0.2 < flips < 0.9 and abs(got - flips) < 0.03, (got, flips)
    assert all(any(sp["actor"] == "voter" for sp in cl["split"]) for cl in det["clusters"])
    # every split column, every actor: Σ child mass by realized discrete class is close to the pooled class mass
    nd = T.Node.from_record(root)
    for a in det["next"]:
        split_everywhere = all(any(sp["actor"] == a["actor"] and sp["domain"] is None for sp in cl["split"])
                               for cl in det["clusters"])
        if not split_everywhere or a["branching"]:
            continue
        effs = [T.canonical(run.engine.discrete_effect(a["actor"], o, nd, root["quarter"][:4] + "Q4")) for o in a["options"]]
        ids = [o["id"] for o in a["options"]]
        for e in set(effs):
            want = sum(o["p_used"] for o, ee in zip(a["options"], effs) if ee == e)
            got = sum(c["p_cond"] for c in root["children"]
                      if effs[ids.index(recs[c["key"]]["cluster"]["realized"][a["actor"]])] == e)
            if want >= 0.05:
                assert abs(got - want) < 0.05, (a["actor"], e, got, want)
    # the misstatement that remains is reported per child and per level
    d0 = recs[root["children"][0]["key"]]["summary"]["discrete"]
    assert {"split", "misstated", "max_misstated", "cell_share"} <= set(d0)
    lv = run.level_stats[0]
    assert "discrete_misstated" in lv and lv["discrete_split_candidates"] > 0


def test_follower_shown_probability_is_the_realized_one(tmp_path, monkeypatch):
    """Review E7 (followers): the shown 采用概率 of a follower is the marginal of its reaction rows under the leader's
    adopted probabilities, and the particle allocation realizes exactly that, whichever actor branches."""
    blocks = []
    orig_alloc, orig_expand = T.Engine.allocate_cluster, T.Engine.expand
    log: dict = {}

    def alloc(self, nd, cl, dec, a0, rng):
        out = orig_alloc(self, nd, cl, dec, a0, rng)
        log.setdefault(nd.key, []).append((float(cl["p"]), {k: v.copy() for k, v in out["assign"].items()}, dec))
        return out

    def expand(self, nd):
        res = orig_expand(self, nd)
        for p, asg, dec in log.get(nd.key, [])[:1]:
            for aid in dec["order"]:
                ab = dec["actors"][aid]
                la = ab.get("follower_of")
                if not la or not ab.get("cond"):
                    continue
                pl = {o["id"]: o["p_used"] for o in dec["actors"][la]["options"]}
                for o in ab["options"]:
                    marg = sum(w * ab["cond"][lo].get(o["id"], 0.0) for lo, w in pl.items())
                    assert abs(marg - o["p_used"]) < 1e-6
                real = np.zeros(len(ab["options"]))
                for pc, a2, _ in log[nd.key]:
                    real += pc * np.bincount(a2[aid], minlength=len(ab["options"])) / len(a2[aid])
                shown = np.array([o["p_used"] for o in ab["options"]])
                blocks.append(float(0.5 * np.abs(real - shown).sum()))
        return res
    monkeypatch.setattr(T.Engine, "allocate_cluster", alloc)
    monkeypatch.setattr(T.Engine, "expand", expand)
    run_tree(tmp_path, "fol", small(depth=3, n_particles=300))
    assert len(blocks) >= 5 and max(blocks) < 0.01, blocks


def test_follower_rows_apply_the_commitment_keep_rule():
    from scenariolab.agents import actors as A
    eng = T.Engine(small(depth=1), root_accounts=AC.load_root())
    nd = eng.root()
    q = T.A.quarter_add(nd.quarter, 1)
    nd.commitments = [{"actor": "bigtech", "option": "capex_raise_large", "zh": "大幅上调", "due_quarter": q,
                       "write": {"field": "d_itinv", "role": "x", "kind": "shift", "a": 0.5, "b": 0.0, "dq": 0},
                       "cancellable": True}]
    a0 = eng.model.complete(nd.cloud.state, MA.Given.empty(nd.cloud.n), T.seed_for(1, nd.key, "a0"))
    gates = AC.gates(nd.cloud.acc) if nd.cloud.acc is not None else {}
    dec = eng.decide(nd, q, a0, gates, [], [])
    bt = dec["actors"].get("bigtech")
    if bt is None or not bt.get("follower_of") or "as_committed" not in [o["id"] for o in bt["options"]]:
        pytest.skip("bigtech is not a follower with a commitment due at this node")
    for row in bt["cond_prior"].values():
        assert row["as_committed"] == pytest.approx(T.COMMITMENT_KEEP, abs=0.02)
    shown = next(o for o in bt["options"] if o["id"] == "as_committed")
    assert shown["p_prior"] == pytest.approx(T.COMMITMENT_KEEP, abs=0.02)
    assert A.REACTIONS


def test_identical_feature_profiles_never_leave_an_empty_cluster(monkeypatch):
    """Two branching profiles with identical features tie at distance 0: before the fix the second head got no
    member and allocate_cluster crashed (the first real snapshot stopped at 2033Q2). Every cluster now has members
    and every particle slot gets a profile."""
    eng = T.Engine(small(depth=1), root_accounts=AC.load_root())
    nd = eng.root()
    q = T.A.quarter_add(nd.quarter, 1)
    a0 = eng.model.complete(nd.cloud.state, MA.Given.empty(nd.cloud.n), T.seed_for(1, nd.key, "a0"))
    gates = AC.gates(nd.cloud.acc) if nd.cloud.acc is not None else {}
    dec = eng.decide(nd, q, a0, gates, [], [])
    # options that write something all look alike; status-quo options write nothing: actors branch (their
    # options differ in impact) and the non-status-quo profiles tie at distance 0
    real_features = T.Engine.features
    monkeypatch.setattr(T.Engine, "features", lambda self, payload: {"x": 1.0} if real_features(self, payload) else {})

    def two_heads(p, D, keep, forced=()):               # selection keeps two profiles at distance 0
        pairs = [(a, b) for a in range(len(p)) for b in range(a + 1, len(p)) if D[a, b] == 0]
        if not pairs:
            pytest.skip("no two profiles with identical features at this node")
        return list(pairs[0]), np.zeros(len(p))
    monkeypatch.setattr(T, "forward_select", two_heads)
    eng.st = dataclasses.replace(eng.st, p_min=0.0, K=3)
    cls = eng.clusters(nd, dec)
    assert cls and all(c["members"] for c in cls)
    assert abs(sum(c["p"] for c in cls) - 1.0) < 1e-9
    for c in cls:
        alloc = eng.allocate_cluster(nd, c, dec, a0, T.seed_for(1, nd.key, f"alloc|{c['id']}"))
        assert all(len(v) == nd.cloud.n for v in alloc["assign"].values())


def test_mock_fixtures_cover_mechanism_and_nature_branches(tmp_path, mock_llm):
    """The UI's 机制 rows and the comic's 自然事件 panel have mock runs to show (review: kept kinds were only central
    and partition): a synthesizer fixture with a mechanism branch, and stress mode for explicit nature events."""
    mock_llm.push("synth_mechanism", match="synth", times=None)
    _, store, _ = run_tree(tmp_path, "mech", small(depth=1, n_particles=200, synth={"mode": "every", "depth": 4}))
    kinds = {r["branch"]["kind"] for r in store.read_level(1)["nodes"] if r.get("branch")}
    assert "mechanism" in kinds, kinds
    mock_llm.reset()
    _, store, _ = run_tree(tmp_path, "nat", small(depth=1, n_particles=200, stress_mode=True), cache=tmp_path / "c2")
    kinds = {r["branch"]["kind"] for r in store.read_level(1)["nodes"] if r.get("branch")}
    assert "nature_event" in kinds, kinds


def test_rerun_on_cache_makes_no_call_and_two_caches_agree(tmp_path):
    _, _, r1 = run_tree(tmp_path, "a", small(depth=2))
    _, _, r2 = run_tree(tmp_path, "b", small(depth=2))
    assert r2["calls_new"] == 0 and r2["tree_hash"] == r1["tree_hash"]
    _, _, r3 = run_tree(tmp_path, "c", small(depth=2), cache=tmp_path / "other_cache")
    assert r3["calls_new"] == r1["calls_new"] and r3["tree_hash"] == r1["tree_hash"]   # deterministic mock


def test_concurrency_never_changes_the_tree(tmp_path):
    _, _, r1 = run_tree(tmp_path, "s", small(depth=2, concurrency=1), cache=tmp_path / "c1")
    _, _, r2 = run_tree(tmp_path, "p", small(depth=2, concurrency=4), cache=tmp_path / "c2")
    assert r1["tree_hash"] == r2["tree_hash"]


def test_c0_tree_equals_the_model_tree(tmp_path):
    """With every pool weight 0 (actors, nature, c_s, c_m) and the model's own priors, the agent tree is the model
    tree, and it makes no LLM call."""
    syn = {"mode": "gated", "c_s": 0.0, "c_m": 0.0}
    _, _, a = run_tree(tmp_path, "agent_c0", small(c_scale=0.0, prior_source="model", synth=syn))
    _, _, m = run_tree(tmp_path, "model", small(tree_kind="model"))
    assert a["calls_new"] == 0
    assert a["tree_hash"] == m["tree_hash"]


def test_c0_agent_tree_equals_the_base_rate_tree(tmp_path):
    """At c = 0 the LLM's answers cannot move a probability: the agent tree is the 历史基率树."""
    _, _, a = run_tree(tmp_path, "agent_c0", small(c_scale=0.0, synth={"mode": "off"}))
    _, _, b = run_tree(tmp_path, "base", small(tree_kind="base_rate"))
    assert a["tree_hash"] == b["tree_hash"]


def test_model_tree_reproduces_the_model(tmp_path):
    """Neutrality (§9.4, E0): the model tree's leaf mixture matches the stub run alone, over 6 seeds."""
    D, N = 3, 300
    zs, ratios = [], []
    for seed in range(6):
        run, _, _ = run_tree(tmp_path, f"m{seed}", small(depth=D, n_particles=N, seed=seed, tree_kind="model"))
        V, W = [], []
        for nd in run.leaves:
            V.append(np.stack([nd.cloud.cum["us_equity"], nd.cloud.state["s_unrate"]], axis=1))
            W.append(np.full(nd.cloud.n, nd.p_path / nd.cloud.n))
        V, W = np.concatenate(V), np.concatenate(W)
        W = W / W.sum()
        mu = (W[:, None] * V).sum(0)
        sd = np.sqrt((W[:, None] * (V - mu) ** 2).sum(0))
        m = MA.stub()
        s = m.root_state(20000)
        rng = np.random.default_rng(100 + seed)
        cum = np.zeros(20000)
        for _ in range(D):
            d = m.complete(s, MA.Given.empty(20000), rng)
            s = m.update_state(s, d)
            cum += d.y["us_equity"]
        P = np.stack([cum, s["s_unrate"]], axis=1)
        zs.append((mu - P.mean(0)) / (P.std(0) / np.sqrt(N)))
        ratios.append(sd / P.std(0))
    zs, ratios = np.array(zs), np.array(ratios)
    assert np.all(np.abs(zs) < 3.0), zs
    assert np.all((ratios.mean(0) > 0.95) & (ratios.mean(0) < 1.05)), ratios


def test_synthesizer_calculus_is_neutral_at_zero_weight():
    rng = np.random.default_rng(0)
    x = {"d_gz": rng.standard_normal(2000), "d_sent": rng.standard_normal(2000)}

    class Ans:
        axis = "d_gz"
        branches = [{"id": "lo", "kind": "partition", "p_llm": 0.6, "region": {"lo_z": None, "hi_z": -1.0}},
                    {"id": "c", "kind": "central", "p_llm": 0.3, "region": {"lo_z": -1.0, "hi_z": 1.0}},
                    {"id": "hi", "kind": "partition", "p_llm": 0.1, "region": {"lo_z": 1.0, "hi_z": None}},
                    {"id": "m1", "kind": "mechanism", "p_llm": 0.4, "shifts": {"d_sent": {"mean": -3.0, "sd": 0.5}}}]
    axes = [("fin_conditions", "d_gz"), ("sentiment", "d_sent")]
    br, _ = SY.consequences(Ans(), x, written=[], causes=["president.tariff_up_large"], classes=["thin"],
                            regime=False, axes=axes, c_s=0.0, c_m=0.0)
    assert all(b["kind"] != "mechanism" for b in br)                     # π = 0 at c_m = 0
    for b in br:
        assert abs(b["p"] - b["m"]) < 1e-9                               # q = m at c_s = 0
    br, rep = SY.consequences(Ans(), x, written=[], causes=["president.tariff_up_large"], classes=["thin"],
                              regime=False, axes=axes)
    mech = [b for b in br if b["kind"] == "mechanism"]
    assert mech and abs(mech[0]["p"] - 0.2) < 1e-9                        # min(c_m · 0.4, 0.25)
    assert abs(sum(b["p"] for b in br) - 1.0) < 1e-9
    assert abs(mech[0]["shifts"]["d_sent"]["mean"] - 0.8 * -3.0) < 1e-9 or \
        abs(mech[0]["shifts"]["d_sent"]["mean"]) <= 2.5 * np.std(x["d_sent"]) + 1e-9
    # well-identified channels in normal states: mechanisms refused (the model already expresses them)
    br, rep = SY.consequences(Ans(), x, written=[], causes=["fed.hike_25"], classes=["well"], regime=False, axes=axes)
    assert not [b for b in br if b["kind"] == "mechanism"] and any("机制不合格" in r for r in rep)


def test_default_partition_rule():
    n = 1000
    calm = {"d_unrate": np.zeros(n), "d_spread": np.zeros(n)}
    assert len(SY.default_partition(calm, {"us_equity": np.zeros(n)}, 1.0)) == 1
    y = {"us_equity": np.where(np.arange(n) < 400, -0.3, 0.0)}
    br = SY.default_partition(calm, y, 1.0)
    assert [b["id"] for b in br] == ["calm", "stress"] and abs(br[1]["p"] - 0.4) < 1e-9


# ---------------------------------------------------------------------------------------------------- pause and resume

def test_usage_limit_pause_then_resume_equals_uninterrupted(tmp_path, mock_llm):
    st = small(depth=3)
    _, _, full = run_tree(tmp_path, "full", st, cache=tmp_path / "c_full")
    mock_llm.push("usage_limit", match="node", times=1)
    store = T.Store(tmp_path / "paused")
    run = T.TreeRun(st, store, cache_dir=tmp_path / "c_paused")
    with pytest.raises(T.Paused) as ei:
        run.run()
    assert ei.value.reason == "usage_limit" and ei.value.reset_at
    assert store.load_checkpoint() is not None                 # every finished level is checkpointed
    res = T.TreeRun(st, store, cache_dir=tmp_path / "c_paused").run(resume=True)
    assert res["tree_hash"] == full["tree_hash"]


def test_checkpoint_resume_is_exact(tmp_path):
    st = small(depth=3)
    _, _, full = run_tree(tmp_path, "full", st, cache=tmp_path / "c1")
    store = T.Store(tmp_path / "cut")
    cancel = __import__("threading").Event()
    run = T.TreeRun(st, store, cache_dir=tmp_path / "c2", cancel=cancel)
    with pytest.raises(T.Paused):
        run.run(on_level=lambda level, stats: cancel.set() if level == 1 else None)
    level, frontier, _ = store.load_checkpoint()
    assert level == 1 and frontier and all(nd.cloud is not None for nd in frontier)
    res = T.TreeRun(st, store, cache_dir=tmp_path / "c2").run(resume=True)
    assert res["tree_hash"] == full["tree_hash"]


def _second_node_call():
    n = {"node": 0}

    def match(kind, system, user):
        if kind == "node":
            n["node"] += 1
            return n["node"] == 2
        return False
    return match


def test_pause_and_resume_with_a_binding_cap_equal_the_uninterrupted_run(tmp_path, mock_llm):
    """Review E4: the run's paid-calls ledger counts the interrupted level's calls (now cache hits) against the
    budget again, so a pause and resume spends and decides exactly as an uninterrupted run, cap included."""
    base = small(depth=3, B=2, n_particles=80)
    _, _, free = run_tree(tmp_path, "free", base, cache=tmp_path / "c_free")
    cap = free["calls_paid"] - 3
    st = small(depth=3, B=2, n_particles=80, max_calls=cap)
    full_run, _, full = run_tree(tmp_path, "full", st, cache=tmp_path / "c_full")
    assert full_run.llm.stats["budget_hits"] > 0 and full["calls_paid"] == cap
    mock_llm.push("usage_limit", match=_second_node_call(), times=1)
    store = T.Store(tmp_path / "cut")
    with pytest.raises(T.Paused):
        T.TreeRun(st, store, cache_dir=tmp_path / "c_cut").run()
    resumed = T.TreeRun(st, store, cache_dir=tmp_path / "c_cut")
    res = resumed.run(resume=True)
    assert res["tree_hash"] == full["tree_hash"]
    assert resumed.llm.stats["budget_hits"] == full_run.llm.stats["budget_hits"]
    assert res["calls_paid"] == full["calls_paid"] == T.ledger_size(tmp_path / "cut")
    ok_calls = [c for c in llm_mock.calls() if c.get("fixture") is None]
    assert len(ok_calls) >= res["calls_paid"]


def test_the_hard_cap_pauses_with_reason_budget(tmp_path):
    st = small(depth=3, B=2, n_particles=80, max_calls=None, hard_cap=5)
    store = T.Store(tmp_path / "hard")
    with pytest.raises(T.Paused) as ei:
        T.TreeRun(st, store, cache_dir=tmp_path / "c").run()
    assert ei.value.reason == "budget" and "1.2" in ei.value.message
    assert T.ledger_size(tmp_path / "hard") <= 5
    cap = T.hard_cap_for(st, spent=T.ledger_size(tmp_path / "hard"))
    res = T.TreeRun(T.TreeSettings.coerce({**st.to_dict(), "hard_cap": cap}), store,
                    cache_dir=tmp_path / "c").run(resume=True)
    assert res["calls_paid"] <= cap
    est = T.estimate(small(depth=8, B=4))
    assert est["hard_cap"] == max(T.HARD_CAP_MIN, -(-12 * est["calls_new"] // 10))


def test_a_pause_stops_the_rest_of_the_level(tmp_path, mock_llm):
    """Review E8(d): after one node of a level pauses (persistent usage limit), no other node of the level starts a
    call: one attempt, not one per frontier node."""
    st = small(depth=3, B=6, n_particles=80, concurrency=2)
    run_tree(tmp_path, "warm", small(depth=1, B=6, n_particles=80, concurrency=2), cache=tmp_path / "c")
    mock_llm.push("usage_limit", match="node", times=50)
    before = len([c for c in llm_mock.calls() if c.get("kind") == "node"])
    with pytest.raises(T.Paused) as ei:
        T.TreeRun(st, T.Store(tmp_path / "p"), cache_dir=tmp_path / "c").run()
    assert ei.value.reason == "usage_limit"
    attempts = len([c for c in llm_mock.calls() if c.get("kind") == "node"]) - before
    assert attempts <= 2, attempts                      # concurrency 2: at most the two calls already started


# ---------------------------------------------------------------------------------------------------- snapshot mode

def _core_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and "overlay" not in p.relative_to(root).parts:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_snapshot_build_load_expand_overlay(tmp_path):
    st = S.default_settings(4, 6, n_particles=100, concurrency=2)
    man = S.build("t1", st, base=tmp_path / "snaps", cache_dir=tmp_path / "cache")
    assert man["status"] == "done" and man["stub"] is True and man["tree_hash"]
    assert not (tmp_path / "snaps" / "t1" / "checkpoint").exists()
    for key in ("settings", "versions", "calls", "data_end", "created", "levels"):
        assert key in man
    root = tmp_path / "snaps" / "t1"
    before = _core_digest(root)
    ex = S.Explorer(root, cache_dir=tmp_path / "cache")
    t0 = time.perf_counter()
    summ = ex.summary()
    assert time.perf_counter() - t0 < 2.0 and summ["read_only"]
    assert sorted(summ["levels"]) == list(range(7))
    for d, v in summ["levels"].items():
        assert abs(v["mass_kept"] - 1.0) < 1e-6
    lv = ex.level(3)["nodes"]
    pruned = [r for r in lv if r["status"] == "pruned"]
    assert pruned, "a beam of 4 prunes candidates at depth 3"
    parent = ex.record(pruned[0]["parent"])
    st_of = {c["key"]: c["status"] for c in parent["children"]}
    assert st_of[pruned[0]["key"]] == "pruned" and set(st_of.values()) <= {"expanded", "pruned", "leaf"}
    ch = next(c for c in parent["children"] if c["key"] == pruned[0]["key"])
    assert ch["reason"] and ch["own_mass"] > 0
    ent = ex.expand(pruned[0]["key"], quarters=1)
    assert ent["detail"] is not None and ent["replayed"] is True
    assert ent["record"]["children"] and abs(sum(c["p_cond"] for c in ent["record"]["children"]) - 1) < 1e-6
    assert _core_digest(root) == before                         # the frozen core is never written
    rec = ex.record(pruned[0]["key"])
    assert rec["source"] == "snapshot" and rec["overlay_children"]
    deeper = ex.expand(ent["record"]["children"][0]["key"], quarters=2)
    assert deeper["detail"] is not None and len(ex.overlay_index()) > len(ent["record"]["children"]) + 1
    leaf = next(r for r in ex.level(6)["nodes"] if r["status"] == "leaf")
    ext = ex.expand(leaf["key"], quarters=1)                    # a saved leaf cloud: no replay needed
    assert ext["replayed"] is False and ext["record"]["children"]
    assert _core_digest(root) == before
    br = ex.branches()
    assert abs(br["mass"] - 1.0) < 1e-6 and br["mixture_cum"]["us_equity"]


@pytest.mark.parametrize("books", [True, False])
def test_snapshot_pruned_nodes_replay_exactly_from_the_frontier(tmp_path, monkeypatch, books):
    """A survivor's cloud is pooled with the nodes it absorbed and resampled to N, which a replay from the root cannot
    rebuild: the build saves every kept node's cloud (frontier/), and a pruned child replays one quarter from it."""
    if not books:
        monkeypatch.setattr(S.AC, "load_root", lambda path=None: None)
        monkeypatch.setattr(T.AC, "load_root", lambda path=None: None)
    st = S.default_settings(3, 5, n_particles=120, concurrency=2)
    S.build("rp", st, base=tmp_path / "snaps", cache_dir=tmp_path / "cache")
    root = tmp_path / "snaps" / "rp"
    kept = [r["key"] for d in range(5) for r in S.Explorer(root).level(d)["nodes"] if r["status"] != "pruned"]
    assert sorted(p.stem for p in (root / "frontier").glob("*.npz")) == sorted(kept)
    ex = S.Explorer(root, cache_dir=tmp_path / "cache")
    n = 0
    for d in range(1, 6):
        for r in ex.level(d)["nodes"]:
            if r["status"] == "pruned":
                nd, replayed = ex.node_with_cloud(r["key"], ex._engine())
                assert replayed and nd.replay_exact and nd.cloud.n == st.n_particles
                n += 1
    assert n > 20


def test_snapshot_usage_limit_pause_and_resume(tmp_path, mock_llm):
    st = S.default_settings(3, 3, n_particles=100, concurrency=1)
    full = S.build("full", st, base=tmp_path / "a", cache_dir=tmp_path / "ca")
    mock_llm.push("usage_limit_text", match="node", times=1)
    with pytest.raises(S.BuildPaused) as ei:
        S.build("cut", st, base=tmp_path / "b", cache_dir=tmp_path / "cb")
    assert ei.value.reason == "usage_limit"
    man = S.read_manifest(tmp_path / "b" / "cut")
    assert man["status"] == "paused" and man["pause"]["reason"] == "usage_limit" and "reset_at" in man["pause"]
    with pytest.raises(ValueError):
        S.build("cut", st, base=tmp_path / "b", cache_dir=tmp_path / "cb")            # needs --resume
    done = S.build("cut", st, resume=True, base=tmp_path / "b", cache_dir=tmp_path / "cb")
    assert done["status"] == "done" and done["tree_hash"] == full["tree_hash"]
    with pytest.raises(ValueError):
        S.build("cut", st, resume=True, base=tmp_path / "b", cache_dir=tmp_path / "cb")  # frozen


def test_size_and_memory_estimates_and_a_deep_build(tmp_path):
    def sz(B, D):
        return S.size_estimate(S.default_settings(B, D, n_particles=100))["bytes"]
    # affine in depth at a fixed beam: equal increments (a fixed part for the root, the synthesizer's levels, leaves)
    d1, d2 = sz(8, 40) - sz(8, 20), sz(8, 60) - sz(8, 40)
    assert d1 > 0 and abs(d2 / d1 - 1.0) < 0.02
    # growing in the beam, less than linearly: the discrete split's cells per branch shrink as B grows
    assert sz(8, 40) < sz(12, 40) < sz(16, 40) < 2 * sz(8, 40)
    m1 = T.memory_estimate(S.default_settings(10, 80))
    m2 = T.memory_estimate(S.default_settings(20, 80))
    assert abs(m2["cloud_bytes"] / m1["cloud_bytes"] - 2.0) < 1e-6           # linear in the beam
    assert m2["peak_bytes"] - m1["peak_bytes"] == pytest.approx(T.MEMORY_OVERHEAD * m1["cloud_bytes"], rel=1e-6)
    big = S.full_estimate(S.default_settings(30, 80))
    assert 2000 < big["calls_new"] < 3200 and big["levels"][-1] == 30 and big["expanded_nodes"] == sum(big["levels"])
    # mostly the kept nodes' frontier clouds (about 0.5 MB each at N 500), which make pruned branches replay exactly
    assert 600 < big["size"]["mb"] < 2000 and big["size"]["frontier_mb"] > 0.8 * big["size"]["mb"], big["size"]
    assert big["memory"]["peak_mb"] < 4096
    first = S.full_estimate(S.default_settings(2, 80))   # the first real snapshot (beam 2 × 80 quarters)
    assert first["calls_new"] < 0.1 * big["calls_new"] and first["size"]["mb"] < 150
    # measured on a build 24 quarters deep (B 2): the level files match the estimate's records within a factor 2
    st = S.default_settings(2, 24, n_particles=100)
    man = S.build("d24", st, base=tmp_path / "s", cache_dir=tmp_path / "c")
    root = tmp_path / "s" / "d24"
    levels = sum(p.stat().st_size for p in (root / "levels").glob("*.gz"))
    est = S.size_estimate(st)
    est_levels = est["expanded_nodes"] * (S.BYTES_KEPT_RECORD + S.BYTES_DETAIL) + est["pruned_nodes"] * S.BYTES_PRUNED_RECORD
    assert 0.5 < levels / est_levels < 2.0, (levels, est_levels)
    pruned = sum(1 for p in sorted((root / "levels").glob("*.nodes.json.gz"))
                 for r in T.read_json(p)["nodes"] if r["status"] == "pruned")
    assert 0.4 < pruned / est["pruned_nodes"] < 2.5, (pruned, est["pruned_nodes"])
    assert man["levels_done"] == 24 and len(man["levels"]) == 24


def test_snapshot_cli_dry_run_and_confirmation(tmp_path, monkeypatch, capsys):
    assert S.main(["--beam", "30", "--quarters", "80", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "快照估计" in out and "大模型调用" in out and "快照大小" in out
    from scenariolab.llm import backend
    monkeypatch.setattr(backend, "backend_name", lambda: "claude_code")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert S.main(["--beam", "2", "--quarters", "2", "--id", "never"]) == 3            # no --yes: refused
    assert not (T.SNAPSHOTS_DIR / "never").exists()


# ---------------------------------------------------------------------------------------------------- jobs and endpoints

@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from scenariolab import server
    jobs.reset_manager(tmp_path / "agents", auto_resume=False)
    yield TestClient(server.app, base_url="http://127.0.0.1:8765")
    jobs.reset_manager(None)


def _wait_done(client, rid, timeout=120):
    t_end = time.time() + timeout
    while time.time() < t_end:
        pr = client.get(f"/api/agents/runs/{rid}/progress").json()
        if pr["status"] in ("done", "failed", "paused", "cancelled"):
            return pr
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def test_run_endpoints(client, tmp_path):
    st = {"depth": 2, "B": 2, "n_particles": 80, "prior_trees": True, "concurrency": 1}
    est = client.post("/api/agents/estimate", json={"settings": st}).json()
    assert est["calls_new"] > 0 and est["latency_basis"] in ("default", "measured")
    r = client.post("/api/agents/runs", json={"settings": st, "name": "测试"})
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    pr = _wait_done(client, rid)
    assert pr["status"] == "done", pr
    types = {e["type"] for e in pr["events"]}
    assert {"run_started", "level_started", "level_done", "run_done", "call_done", "prior_trees_done"} <= types
    run = client.get(f"/api/agents/runs/{rid}").json()
    assert run["status"] == "done" and run["root"] and run["prior_trees"].keys() == {"base_rate", "model"}
    lv = client.get(f"/api/agents/runs/{rid}/levels/1").json()
    assert lv["nodes"]
    node = client.get(f"/api/agents/runs/{rid}/nodes/{run['root']}").json()
    assert node["detail"]["next"] and node["record"]["children"]
    assert client.get(f"/api/agents/runs/{rid}/branches").json()["leaves"]
    for tree in ("base_rate", "model"):
        assert client.get(f"/api/agents/runs/{rid}/prior_trees/{tree}/branches").json()["leaves"]
    leaf = client.get(f"/api/agents/runs/{rid}/branches").json()["leaves"][0]["key"]
    story = client.post(f"/api/agents/runs/{rid}/nodes/{leaf}/story", json={})
    assert story.status_code == 200 and story.json()["story"]["story"]
    assert client.get("/api/agents/runs").json()["runs"][0]["id"] == rid
    assert client.patch(f"/api/agents/runs/{rid}", json={"name": "改名"}).json()["name"] == "改名"
    assert client.post(f"/api/agents/runs/{rid}/export_set", json={}).status_code == 409
    ext = client.post(f"/api/agents/runs/{rid}/extend", json={"quarters": 1})
    assert ext.status_code == 200, ext.text
    assert _wait_done(client, rid)["status"] == "done"
    assert client.get(f"/api/agents/runs/{rid}").json()["levels_done"] == 3
    with client.stream("GET", f"/api/agents/runs/{rid}/events") as resp:
        body = b"".join(resp.iter_bytes()).decode()
    assert "data:" in body and "event: end" in body
    assert client.delete(f"/api/agents/runs/{rid}").json() == {"deleted": rid}


def test_agent_routes_guard_off_host_and_confirmation(client, monkeypatch):
    from fastapi.testclient import TestClient

    from scenariolab import server
    evil = TestClient(server.app, base_url="http://evil.example")
    for method, path, body in (("post", "/api/agents/runs", {"settings": {}}), ("put", "/api/agents/institutions", {}),
                               ("post", "/api/agents/snapshots/x/nodes/0123456789abcdef/expand", {}),
                               ("post", "/api/agents/runs/r_x/resume", {}), ("delete", "/api/agents/runs/r_x", None),
                               ("post", "/api/agents/snapshots", {"id": "x"})):
        r = getattr(evil, method)(path, json=body) if body is not None else getattr(evil, method)(path)
        assert r.status_code == 403, (path, r.status_code)
    # the real CLI: a start without confirm is refused with the estimate, nothing runs
    monkeypatch.setattr(jobs, "real_backend", lambda: True)
    r = client.post("/api/agents/runs", json={"settings": {"depth": 2, "B": 2}})
    assert r.status_code == 409 and r.json()["needs_confirm"] is True and r.json()["estimate"]["calls_new"] > 0
    assert client.get("/api/agents/runs").json()["runs"] == []
    # existing routes are not shadowed; there is no bare /api/agents
    assert client.get("/api/agents").status_code == 404
    assert client.get("/api/agents/config").json()["model"]["stub"] is True
    assert client.get("/api/agents/schema").json()["x_fields"]
    assert "available" in client.get("/api/agents/llm_status").json()
    assert client.get("/api/agents/sets").json()["enabled"] is False


GUARDED = [("post", "/api/agents/runs"), ("put", "/api/agents/institutions"), ("post", "/api/agents/estimate"),
           ("patch", "/api/agents/runs/r_x"), ("delete", "/api/agents/runs/r_x"), ("post", "/api/agents/runs/r_x/cancel"),
           ("post", "/api/agents/runs/r_x/resume"), ("post", "/api/agents/runs/r_x/extend"),
           ("post", "/api/agents/runs/r_x/intervene"), ("post", "/api/agents/runs/r_x/nodes/0123456789abcdef/expand"),
           ("post", "/api/agents/runs/r_x/nodes/0123456789abcdef/story"), ("post", "/api/agents/runs/r_x/export_set"),
           ("post", "/api/agents/snapshots"), ("post", "/api/agents/snapshots/s1/nodes/0123456789abcdef/expand"),
           ("post", "/api/agents/snapshots/s1/nodes/0123456789abcdef/story"), ("post", "/api/agents/snapshots/s1/cancel")]


def test_every_agents_route_refuses_cross_site_requests(client, monkeypatch):
    """Review E1: a page on another site may POST to 127.0.0.1 with a local Host header (a "simple" request with
    text/plain or no body); the Origin / Sec-Fetch-Site check and the JSON-body requirement refuse every one, before
    anything runs."""
    from scenariolab.agents import snapshot as S2
    calls = []
    monkeypatch.setattr(S2.Explorer, "submit", lambda *a, **k: calls.append(a) or {"id": "x", "status": "queued"})
    evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
    for method, path in GUARDED:
        variants = [{"headers": evil}, {"headers": {"Origin": "https://evil.example"}}]
        if method != "delete":
            variants.append({"content": b"", "headers": {**evil, "Content-Type": "text/plain"}})
        for kw in variants:
            r = client.request(method.upper(), path, **kw)
            assert r.status_code == 403, (method, path, kw, r.status_code)
        if method in ("post", "put", "patch"):        # same origin but no JSON body: also refused
            r = client.request(method.upper(), path, content=b"", headers={"Content-Type": "text/plain"})
            assert r.status_code == 403, (method, path, r.status_code)
    r = client.get("/api/agents/llm_status", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert calls == []
    own = {"Origin": "http://127.0.0.1:8765", "Sec-Fetch-Site": "same-origin"}
    assert client.post("/api/agents/estimate", json={"settings": {"depth": 2, "B": 2}}, headers=own).status_code == 200


def test_status_checks_are_cached(client, monkeypatch):
    from scenariolab.agents import llm_io
    from scenariolab.llm import backend
    llm_io.clear_status_cache()
    n = {"status": 0}
    real = backend.status

    def counting(*a, **k):
        n["status"] += 1
        return real(*a, **k)
    monkeypatch.setattr(backend, "status", counting)
    for _ in range(5):
        assert client.get("/api/agents/llm_status").status_code == 200
    assert n["status"] == 1                                    # one CLI check a minute, not two per request
    llm_io.clear_status_cache()


def test_run_branch_generation_needs_confirmation_with_the_real_cli(client, tmp_path, monkeypatch):
    """Review E1/E5: generating a 未生成 branch of a saved run spends usage, so with the real CLI it needs confirm:
    true (409 needs_confirm with the estimate); a snapshot's branch needs none (the user's decision)."""
    st = {"depth": 2, "B": 2, "n_particles": 80, "prior_trees": False, "concurrency": 1}
    rid = client.post("/api/agents/runs", json={"settings": st}).json()["run_id"]
    assert _wait_done(client, rid)["status"] == "done"
    lv = client.get(f"/api/agents/runs/{rid}/levels/1").json()["nodes"]
    key = next(r["key"] for r in lv if r["status"] == "pruned")
    from scenariolab.agents import snapshot as S2
    submitted = []
    monkeypatch.setattr(S2.Explorer, "submit", lambda self, k, **kw: submitted.append((self.kind, kw)) or
                        {"id": "j", "status": "queued", "node": k})
    monkeypatch.setattr(jobs, "real_backend", lambda: True)
    r = client.post(f"/api/agents/runs/{rid}/nodes/{key}/expand", json={"quarters": 2})
    assert r.status_code == 409 and r.json()["needs_confirm"] and r.json()["estimate"]["calls_new"] >= 1
    assert submitted == []
    r = client.post(f"/api/agents/runs/{rid}/nodes/{key}/expand", json={"quarters": 2, "confirm": True})
    assert r.status_code == 200 and submitted == [("run", {"quarters": 2, "confirmed": True})]
    base = tmp_path / "agents" / "snapshots"
    S.build("s9", S.default_settings(2, 2, n_particles=60), base=base, cache_dir=tmp_path / "agents" / "cache")
    skey = next(r["key"] for r in client.get("/api/agents/snapshots/s9/levels/1").json()["nodes"] if r["status"] == "pruned")
    r = client.post(f"/api/agents/snapshots/s9/nodes/{skey}/expand", json={})
    assert r.status_code == 200 and submitted[-1] == ("snapshot", {"quarters": 1, "confirmed": True})


def test_snapshot_builds_from_the_server_validate_record_and_cancel(client, tmp_path, monkeypatch):
    """Review E8/E13: bad ids, bad settings and existing ids are refused before anything runs; a resume uses the
    manifest's own settings; a worker error is recorded; a build can be cancelled; a restart leaves no build
    'building'."""
    assert client.post("/api/agents/snapshots", json={"id": "我的 快照", "settings": {}}).status_code == 400
    assert client.post("/api/agents/snapshots", json={"id": "ok", "settings": {"B": 0}}).status_code == 400
    base = tmp_path / "agents" / "snapshots"
    S.build("done1", S.default_settings(2, 1, n_particles=60), base=base, cache_dir=tmp_path / "agents" / "cache")
    r = client.post("/api/agents/snapshots", json={"id": "done1", "settings": {"B": 2, "depth": 1}})
    assert r.status_code == 400 and "已存在" in r.json()["detail"]
    r = client.post("/api/agents/snapshots", json={"id": "nope", "settings": {}, "resume": True})
    assert r.status_code == 404
    # a paused build with a non-default option resumes from its own manifest (the request's B/depth are ignored)
    st = S.default_settings(2, 3, n_particles=60)
    cancel = __import__("threading").Event()
    with pytest.raises(S.BuildPaused):
        S.build("p1", st, base=base, cache_dir=tmp_path / "agents" / "cache", cancel=cancel,
                listener=lambda ev: cancel.set() if ev.get("type") == "level_done" else None)
    r = client.post("/api/agents/snapshots", json={"id": "p1", "settings": {"B": 2, "depth": 80}, "resume": True})
    assert r.status_code == 200, r.text
    t_end = time.time() + 60
    while time.time() < t_end and (S.read_manifest(base / "p1") or {}).get("status") != "done":
        time.sleep(0.2)
    man = S.read_manifest(base / "p1")
    assert man["status"] == "done" and man["levels_done"] == 3 and man["settings"]["n_particles"] == 60
    assert man["spent_calls"] == T.ledger_size(base / "p1")
    # a worker error is recorded in the manifest
    monkeypatch.setattr(S, "full_estimate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert client.post("/api/agents/snapshots", json={"id": "f1", "settings": {"B": 2, "depth": 1}}).status_code == 200
    t_end = time.time() + 30
    while time.time() < t_end and (S.read_manifest(base / "f1") or {}).get("status") in (None, "building"):
        time.sleep(0.1)
    m = S.read_manifest(base / "f1")
    assert m["status"] == "failed" and "boom" in m["error"]
    # cancel: an idle paused build stays paused; a restart turns "building" into paused
    assert client.post("/api/agents/snapshots/f1/cancel", json={}).json()["status"] == "paused"
    (base / "zz").mkdir()
    S.write_manifest(base / "zz", {"id": "zz", "status": "building", "settings": {}})
    mgr = jobs.JobManager(tmp_path / "agents", auto_resume=False)
    assert S.read_manifest(base / "zz")["status"] == "paused" and \
        S.read_manifest(base / "zz")["pause"]["reason"] == "server_restart"
    assert mgr.active is None


def test_usage_limits_without_a_reset_time_are_not_retried(tmp_path):
    mgr = jobs.JobManager(tmp_path / "agents", auto_resume=True)
    d = mgr.run_dir("r_20261102_000000_beef")
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"id": d.name, "status": "paused", "settings": {}}), encoding="utf-8")
    mgr._schedule_resume(d.name, None)
    assert mgr._timers == {}                              # no hourly retry loop
    mgr._schedule_resume(d.name, time.time() + 3600)
    assert d.name in mgr._timers
    mgr.cancel(d.name)
    assert mgr._timers == {}


def test_snapshot_endpoints(client, tmp_path):
    base = tmp_path / "agents" / "snapshots"
    S.build("s1", S.default_settings(3, 3, n_particles=80), base=base, cache_dir=tmp_path / "agents" / "cache")
    snaps = client.get("/api/agents/snapshots").json()["snapshots"]
    assert snaps[0]["id"] == "s1" and snaps[0]["status"] == "done"
    summ = client.get("/api/agents/snapshots/s1").json()
    assert summ["read_only"] and summ["root"]
    lv = client.get("/api/agents/snapshots/s1/levels/2").json()
    pruned = [r for r in lv["nodes"] if r["status"] == "pruned"]
    assert pruned
    key = pruned[0]["key"]
    node = client.get(f"/api/agents/snapshots/s1/nodes/{key}").json()
    assert node["record"]["status"] == "pruned" and node["path"][-1] == key and node["detail"] is None
    r = client.post(f"/api/agents/snapshots/s1/nodes/{key}/expand", json={"wait": 60})
    assert r.status_code == 200, r.text
    assert r.json()["job"]["status"] == "done" and r.json()["entry"]["detail"]
    node = client.get(f"/api/agents/snapshots/s1/nodes/{key}").json()
    assert node["record"]["overlay_children"] and node["detail"]
    again = client.post(f"/api/agents/snapshots/s1/nodes/{key}/expand", json={})
    assert again.status_code == 200 and again.json()["status"] == "done"
    lv = client.get("/api/agents/snapshots/s1/levels/2").json()
    assert any(r.get("overlay") == "expanded" for r in lv["nodes"]) and lv["overlay"] == []
    assert client.get(f"/api/agents/snapshots/s1/path/{key}").json()["path"][0]["level"] == 0
    br = client.get("/api/agents/snapshots/s1/branches").json()
    assert br["leaves"] and len(br["action_shares"]) == 3
    for row in br["action_shares"]:              # shares of the level's mass (an actor may act in some nodes only)
        for opts in row["actors"].values():
            assert sum(opts.values()) <= 1.0 + 1e-4
        assert abs(sum(row["actors"]["fed"].values()) - 1.0) < 0.01
    # the comic's path: an unknown node is a 404 (not an empty path); a node's best leaf is at the deepest level
    assert client.get("/api/agents/snapshots/s1/path/0123456789abcdef").status_code == 404
    root = summ["root"]
    leaf = client.get(f"/api/agents/snapshots/s1/best_leaf/{root}").json()["leaf"]
    deepest = client.get("/api/agents/snapshots/s1/levels/3").json()["nodes"]
    best = max((r for r in deepest if r["status"] != "pruned"), key=lambda r: r["p_path"])
    assert leaf == best["key"]
    mid = client.get("/api/agents/snapshots/s1/levels/1").json()["nodes"]
    for r in [x for x in mid if x["status"] != "pruned"]:
        lk = client.get(f"/api/agents/snapshots/s1/best_leaf/{r['key']}").json()["leaf"]
        assert r["key"] in client.get(f"/api/agents/snapshots/s1/nodes/{lk}").json()["path"]
    assert client.get("/api/agents/snapshots/nope").status_code == 404
    assert client.get("/api/agents/snapshots/s1/nodes/not-a-key").status_code == 400


def test_job_manager_marks_runs_paused_after_restart(tmp_path):
    d = tmp_path / "agents" / "runs" / "r_20261102_000000_abcd"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"id": d.name, "status": "running", "settings": {}}), encoding="utf-8")
    mgr = jobs.JobManager(tmp_path / "agents", auto_resume=False)
    assert mgr.read(d.name)["status"] == "paused" and mgr.read(d.name)["pause"]["reason"] == "server_restart"


def test_export_arrays_layout(tmp_path):
    run, _, _ = run_tree(tmp_path, "x", small(depth=2, keep_paths="full"))
    arr = T.export_arrays(run.leaves, run.adapter.assets)
    S_, N, T_, A_ = arr["total"].shape
    assert S_ == len(run.leaves) and T_ == 2 and A_ == len(run.adapter.assets)
    assert abs(arr["probs"].sum() - 1.0) < 1e-9 and int(arr["step_months"]) == 3
    assert list(arr["labels"][:3]) == ["2026-09", "2026-12", "2027-03"]
    assert T.EXPORT_ENABLED is False
