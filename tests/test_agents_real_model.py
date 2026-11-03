"""Phase P14: the agent tree on the history-estimated quarterly model instead of the Gaussian stub.

The first part runs everywhere (the stub, fakes built on it, and the committed overlay): the adapter contract (R-A4
Draws.drawn / notes, the default adapter and its fallback, model_info), the R-A1 overlay on schema 3.0, the tail rule
(non-finite or absurd particles never crash a build or reach books, clouds or charts), safe float32 casts, the
policy-rate floor and the room to cut, the A0 bins (midpoint rule, the pinned first meeting), the routing of pins on a
structural adapter, the export decision and the size / memory estimates. The second part needs the quarterly panel
(data/ is git-ignored) and is skipped without it: the real adapter as the default, do / reduced / structural pins with
common random numbers, the 0.1 floor, a mock snapshot on the real model, and the model tree's neutrality.

Every LLM interaction is the mock backend (tests/conftest.py); no test starts Claude Code or spends tokens."""
import warnings

import numpy as np
import pytest

from scenariolab.agents import accounts as AC
from scenariolab.agents import actors as A
from scenariolab.agents import model_adapter as MA
from scenariolab.agents import overlay as OV
from scenariolab.agents import snapshot as S
from scenariolab.agents import tree as T

HAS_ROOT = AC.load_root() is not None
READY, WHY = MA.quarterly_data_ready()
needs_quarterly = pytest.mark.skipif(not READY, reason=f"needs the quarterly panel: {WHY}")


def small(**kw) -> T.TreeSettings:
    base = dict(depth=2, B=2, n_particles=120, max_calls=400, concurrency=1, prior_trees=False)
    base.update(kw)
    return T.TreeSettings.coerce(base)


# ================================================================================================ the contract

def test_draws_carry_drawn_and_notes_through_take_and_concat():
    x = {"d_ff": np.arange(4.0), "_q": np.arange(8.0).reshape(4, 2)}
    d = MA.Draws(x, {"us_equity": np.zeros(4)}, {}, np.zeros(4), np.array(["1990Q1", "1991Q2", "2001Q3", "2008Q4"],
                                                                         dtype=object),
                 [{"field": "d_ff", "particles": 1, "text": "floor"}])
    t = d.take([3, 0, 0])
    assert list(t.drawn) == ["2008Q4", "1990Q1", "1990Q1"] and t.notes == d.notes and t.x["_q"].shape == (3, 2)
    bare = MA.Draws({"d_ff": np.ones(2), "_q": np.ones((2, 2))}, {"us_equity": np.ones(2)}, {}, np.ones(2))
    c = MA.Draws.concat([d, bare])
    assert c.n == 6 and list(c.drawn[-2:]) == ["", ""] and len(c.notes) == 1
    assert MA.as_draws(d) is d and MA.Draws.concat([bare, bare]).drawn is None


def test_the_stub_is_pinned_in_tests_and_the_default_falls_back_with_a_reason(monkeypatch):
    assert MA.load_adapter() is MA.stub()                       # tests/conftest.py: MSL_AGENT_MODEL=stub
    info = MA.model_info(MA.stub())
    assert info["stub"] and info["label"] == MA.STUB_LABEL and "MSL_AGENT_MODEL=stub" in info["why_stub"]
    with pytest.raises(ValueError):
        MA.load_adapter("not-a-spec")
    monkeypatch.delenv("MSL_AGENT_MODEL")
    monkeypatch.setitem(MA._FALLBACK, "key", None)
    monkeypatch.setattr(MA, "quarterly_data_ready", lambda: (False, "缺少季度模型的数据：panel_quarterly.csv"))
    monkeypatch.setattr(MA, "_data_key", lambda: ("no-data",))
    assert MA.load_adapter() is MA.stub()
    assert "panel_quarterly" in MA.model_info(MA.stub())["why_stub"]
    # the data exist but the model cannot be loaded: the labelled stub, and the reason
    monkeypatch.setitem(MA._FALLBACK, "key", None)
    monkeypatch.setattr(MA, "quarterly_data_ready", lambda: (True, ""))
    monkeypatch.setattr(MA, "_data_key", lambda: ("broken",))

    def boom(name):
        raise RuntimeError("fit failed")
    monkeypatch.setattr(MA, "_from_spec", boom)
    assert MA.load_adapter() is MA.stub()
    assert "季度模型无法加载" in MA.model_info(MA.stub())["why_stub"]


def test_model_kind_families_and_the_floor_of_field_info():
    assert MA.same_model_kind("qadapter-1/a", "qadapter-1/b") and MA.same_model_kind("gaussian-stub-1", "gaussian-stub-1")
    assert not MA.same_model_kind("gaussian-stub-1", "qadapter-1/a")
    assert MA.field_floor({"d_ff": {"level_floor": 0.1}}, "d_ff") == 0.1 and MA.field_floor({}, "d_ff", 0.0) == 0.0
    assert MA.adapter_kind(MA.stub()) == "stub"


# ================================================================================================ R-A1: the overlay

def test_the_overlay_is_schema_3_and_its_new_entries():
    ov = OV.load()
    assert ov and ov["fields"] == OV.build()["fields"], "overlay.json is stale: python -m scenariolab.agents.overlay"
    f = ov["fields"]
    assert len(f) == 94 and ov["schema_version"].startswith("schema-revise-2 3.0")
    assert f["x_oil_cause"]["cls"] == "event_label" and f["x_oil_cause"]["enum"] == {"demand": 0.0, "mixed": 0.5,
                                                                                      "supply": 1.0}
    assert f["y_mult"]["synth"] == "partition_optional" and "CAPE" in f["y_mult"]["zh"]
    assert f["s_pe"]["display_only"] and f["y_eps"]["display_only"] and f["s_ecyc"]["display_only"]
    assert f["d_tips10"]["cls"] == "financial_price" and f["d_ff"]["cls"] == "instrument"
    assert f["d_itinv"]["channel_class"] == "thin" and f["d_unrate"]["lag_discipline"]
    merged = OV.merge(MA.stub().field_info())
    assert "CAPE" in merged["mult"]["zh"] and merged["s_pe"]["display_only"] and merged["d_ff"]["hist"]
    assert OV.enum_label("x_oil_cause", 0.9) == "供给推动" and OV.enum_label("x_oil_cause", 0.4) == "供需混合"
    assert A.validate_catalogue(A.FieldResolver.load()) == []


def test_the_oil_composite_labels_the_cause_as_supply():
    r = A.FieldResolver(fields=["d_metals_oil", "x_oil_cause", "d_gpr"])
    ev = A.combine_events([A.option_payload("belligerents", "escalate", resolver=r)], resolver=r)
    lab = [w for w in ev if w["field"] == "x_oil_cause"]
    assert lab and lab[0]["a"] == OV.OIL_CAUSE["supply"]


# ================================================================================================ the tail rule

class TailStub(MA.GaussianStub):
    """The stub with a heavy tail: some particles' draws are not finite or leave the bounds (as the quarterly
    stack's long-horizon tails do until its P4b)."""

    def __init__(self, share: float = 0.12):
        super().__init__()
        self.version = "gaussian-stub-tail"
        self.share = share

    def complete(self, state, given, rng, *, structural=None):
        d = super().complete(state, given, rng)
        n = d.n
        k = max(1, int(self.share * n))
        u = np.random.default_rng(int(np.sum(np.asarray(state["s_ff"]) * 1000)) % 2**31).permutation(n)
        a, b, c = u[:k // 3], u[k // 3:2 * k // 3], u[2 * k // 3:k]
        d.y["us_equity"][a] = np.inf
        d.x["d_unrate"][b] = np.nan
        d.x["d_gz"][c] = 40.0
        return d


def test_tail_particles_never_reach_books_clouds_or_charts(tmp_path):
    ad = TailStub()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np_err = np.seterr(all="warn")
        try:
            run = T.TreeRun(small(depth=3), T.Store(tmp_path / "t"), adapter=ad, cache_dir=tmp_path / "c")
            res = run.run()
        finally:
            np.seterr(**np_err)
    ours = [w for w in caught if issubclass(w.category, RuntimeWarning) and "scenariolab" in str(w.filename)
            and "model_adapter.py" not in str(w.filename)]
    assert not ours, [f"{w.filename}:{w.lineno} {w.message}" for w in ours[:5]]
    assert res["leaves"]
    for d in run.store.levels():
        for r in run.store.read_level(d)["nodes"]:
            if d and r["status"] != "pruned":
                tail = (r.get("summary") or {}).get("tail") or {}
                assert tail.get("share", 0) > 0.05, (d, tail)
                if r.get("summary", {}).get("identities"):
                    assert r["summary"]["identities"]["ok"], r["summary"]["identities"]
    for nd in run.leaves:
        cl = nd.cloud
        assert not MA.state_tail_mask(cl.state).any()
        assert np.all(np.isfinite(cl.cum["us_equity"])) and np.all(np.abs(cl.y["us_equity"]) <= 1.0)
        assert np.all(cl.state["s_gz"] <= MA.TAIL_BOUNDS["s_gz"] + 1e-9)
        if cl.acc is not None:
            assert all(np.all(np.isfinite(np.asarray(v, float))) for k, v in cl.acc.s.items()
                       if np.asarray(v).dtype.kind == "f")
    stats = run.level_stats
    assert all(s["tail_share"] > 0.05 for s in stats) and all(s["tail_max"] >= s["tail_share"] for s in stats)
    assert any(e["type"] == "tail_particles" and e["rule"] == MA.TAIL_RULE for e in run.journal.events)
    det = run.store.read_level(0, "detail")["details"]
    assert all(c["tail"]["share"] > 0 for d in det.values() for c in d["clusters"])


def test_all_tail_clusters_are_clipped_and_flagged(tmp_path):
    run = T.TreeRun(small(depth=1), T.Store(tmp_path / "t"), adapter=TailStub(share=1.0), cache_dir=tmp_path / "c")
    run.run()
    for nd in run.leaves:
        assert (nd.summary.get("tail") or {}).get("all") or (nd.summary.get("tail") or {}).get("share") == 1.0
        assert np.all(np.isfinite(nd.cloud.cum["us_equity"]))


def test_float32_casts_clip_instead_of_overflowing():
    v = np.array([1.0, 1e300, -1e300, np.nan, np.inf])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = MA.to_float32(v)
    assert out.dtype == np.float32 and out[1] == np.float32(MA.FLOAT32_MAX) and np.isnan(out[3]) and np.isinf(out[4])
    assert MA.to_float32(np.arange(3)).dtype.kind == "i"


# ================================================================================================ the Fed and the bins

def test_the_room_to_cut_follows_the_models_floor():
    base = {"s_ff": 0.45, "s_core_yoy": 2.5, "s_infexp1": 3.0, "s_unrate": 4.5}
    inst = A.institution_values(A.institution_defaults())

    def opts(floor):
        c = A.NodeContext(quarter="2027Q2", state={**base, "ff_floor": floor}, facts={}, political={},
                          institutions=inst, gates={}, commitments=[], stress_mode=False, resolver=None,
                          leader_choices={})
        m = A.menu("fed", c)
        return [o["id"] for o in m["main"]["options"]], m["disabled"]
    with_floor, why = opts(0.1)
    assert "cut_50" not in with_floor and "cut_25" in with_floor
    assert any(d["id"] == "cut_50" and "下限" in d["why"] for d in why)
    assert "cut_50" in opts(0.0)[0]                              # the stub's floor 0: 0.45 − 0.5 rounds to the floor


def test_bin_edges_split_at_the_nearest_decision():
    full = T.Engine.bin_edges("fed", [o for o, _, _ in A.instrument_bins("fed")])
    assert full == {o: (lo, hi) for o, lo, hi in [(o, -np.inf if lo == -np.inf else lo, hi)
                                                  for o, lo, hi in A.instrument_bins("fed")]}
    trimmed = T.Engine.bin_edges("fed", ["cut_25", "hold", "hike_25", "hike_75plus"])
    assert trimmed["cut_25"] == (-np.inf, -0.125) and trimmed["hike_25"] == (0.125, 0.5)
    assert trimmed["hike_75plus"] == (0.5, np.inf)
    tar = T.Engine.bin_edges("president", ["tariff_down", "tariff_hold", "tariff_up_small", "tariff_up_large"])
    assert tar["tariff_hold"] == (-0.25, 0.25) and tar["tariff_up_small"] == (0.25, 1.5)


def test_a_pinned_first_meeting_renormalizes_the_a0_prior():
    eng = T.Engine(small(n_particles=50), adapter=MA.stub(), root_accounts=None)
    v = np.array([-0.3] * 10 + [0.0] * 20 + [0.3] * 30 + [0.55] * 25 + [1.2] * 15)
    pr = eng.a0_bins("fed", ["cut_25", "hold", "hike_25", "hike_75plus"], {"d_ff": v},
                     pinned={"meeting": "fomc_2026_10", "move": 0.0, "option": "hold"})
    assert pr["p"]["hike_75plus"] == 0.0 and pytest.approx(sum(pr["p"].values())) == 1.0
    assert pr["p"] == pytest.approx({"cut_25": 10 / 60, "hold": 20 / 60, "hike_25": 30 / 60, "hike_75plus": 0.0})
    assert pr["renormalized_mass"] == pytest.approx(0.6) and "重新归一" in pr["label"]
    free = eng.a0_bins("fed", ["cut_25", "hold", "hike_25", "hike_75plus"], {"d_ff": v})
    assert free["p"]["hike_75plus"] == pytest.approx(0.4) and free["p"]["hike_25"] == pytest.approx(0.3)
    ok = np.ones(100, bool)
    ok[-15:] = False                                              # tail particles are not counted
    assert eng.a0_bins("fed", ["hold", "hike_25", "hike_75plus"], {"d_ff": v}, ok=ok)["p"]["hike_75plus"] == \
        pytest.approx(25 / 85)


def test_a_ranged_options_default_size_is_the_models_median_in_its_bin():
    eng = T.Engine(small(n_particles=50), adapter=MA.stub(), root_accounts=None)
    o = {"id": "hike_75plus", "param": {"lo": 0.75, "hi": 1.5, "default": 1.0, "step": 0.25}}
    many = {"edges": {"hike_75plus": (0.625, np.inf)}, "values": np.r_[np.full(20, 1.2), np.full(20, 1.3), [0.1]]}
    assert eng.a0_bin_magnitude("fed", o, many) == pytest.approx(1.25)
    few = {"edges": {"hike_75plus": (0.625, np.inf)}, "values": np.r_[np.full(T.A0_MAGNITUDE_MIN - 1, 1.4), [0.1]]}
    assert eng.a0_bin_magnitude("fed", o, few) is None            # a handful of particles is not a typical size


# ================================================================================================ structural routing

class RecordingStub(MA.GaussianStub):
    """The stub declaring "structural" and recording how every completion was pinned."""
    capabilities = frozenset({"structural"})

    def __init__(self):
        super().__init__()
        self.version = "gaussian-stub-recording"
        self.calls = []

    def field_info(self):
        info = super().field_info()
        for f, mode in (("d_ff", "do"), ("d_bs", "do"), ("d_tariff", "do_last"), ("d_lf", "do_last")):
            info.setdefault(f, {})["given_mode"] = mode
        return info

    def complete(self, state, given, rng, *, structural=None):
        self.calls.append({"given": tuple(getattr(given, "ids", ())), "structural": tuple(sorted(structural or {}))})
        return super().complete(state, given, rng)


def test_a_structural_adapter_gets_interventions_through_structural(tmp_path):
    """P14's lever handling (channels "off": the response table stands in for transfers; the template channels'
    counterpart is tests/test_agents_channels.py)."""
    ad = RecordingStub()
    eng = T.Engine(small(n_particles=200, first_quarter="2027Q1", channels="off"), adapter=ad,
                   root_accounts=AC.load_root())
    root = eng.root()
    a0 = eng.a0(root)
    dec = eng.decide(root, "2027Q1", a0, {}, [], [])
    cong = dec["actors"].get("congress")
    assert cong is not None
    stim = A.option_summary("congress", A.ACTORS["congress"].find("stimulus_small")[1])
    cong["options"] = [{"id": "stimulus_small", "label": stim["zh"], "p_used": 1.0, "p_prior": 1.0, "magnitude": 0.8,
                        "param": stim["param"], "bin": None, "status_quo": False, "source": "menu"}]
    cong["attributes"] = []
    cl = eng.clusters(root, dec)[0]
    alloc = eng.allocate_cluster(root, cl, dec, a0, T.seed_for(1, root.key, "alloc"))
    bk = eng.build_bk(root, cl, dec, a0, alloc, T.seed_for(1, root.key, "bk"))
    assert "d_transfers" in bk["structural"].ids                 # do by u_tr, not a reduced-form observation
    # this stub ignores structural= (an intervention reaches nothing): the response table stands in (§6.3)
    assert "d_transfers" in eng.inert and "d_cons_goods" in bk["given"].ids
    assert not set(SY_LAG()) & set(bk["given"].ids)              # no two-step protocol
    assert "d_ff" in bk["given"].ids                             # the Fed's decision: do through given_modes
    held = set(bk["structural"].ids) & {"d_metals_oil", "d_gpr"}
    assert held                                                  # the hazard fields an action did not write: do_last
    assert bk["crn"] and ad.calls[-1]["structural"]
    # common random numbers: the stub's Matheron completion returns A0's draw where nothing is pinned
    rows = bk["rows"]
    free = np.ones(len(rows), bool)
    for g in (bk["given"], bk["structural"]):
        free &= ~np.isfinite(g.values).any(axis=1)
    if free.any():
        assert np.allclose(bk["draws"].y["us_equity"][free], np.asarray(a0.y["us_equity"])[rows][free])


def SY_LAG():
    from scenariolab.agents import synthesizer as SY
    return SY.LAG_PROTECTED


def test_the_stub_keeps_the_two_step_protocol(tmp_path):
    eng = T.Engine(small(n_particles=120), adapter=MA.stub(), root_accounts=AC.load_root())
    assert not eng.structural and eng.ff_floor == 0.0
    root = eng.root()
    children, det = eng.expand(root)
    assert children and all(c["crn"] for c in det["clusters"])
    assert det["model"]["kind"] == "stub" and det["model"]["ff_floor"] == 0.0


# ================================================================================================ export and estimates

def test_the_hand_off_stays_disabled_and_says_why():
    stub = T.export_status(MA.model_info(MA.stub()), "none")
    assert not stub["enabled"] and any("桩模型" in r for r in stub["reasons"])
    real = T.export_status({"stub": False, "trial": True, "kind": "quarterly"}, "none")
    assert not real["enabled"] and T.EXPORT_DISABLED_WHY in real["why"] and "试运行" in real["why"]
    assert any("逐季路径" in r for r in real["reasons"]) and "keep_paths" not in real["why"]
    assert "逐季路径" not in T.export_status({"stub": False, "trial": True}, "full")["why"]
    assert "step_months" not in real["why"]


def test_the_estimate_measures_the_pricing_models_bytes_per_particle(tmp_path):
    st = S.default_settings(2, 12, n_particles=200)
    est = S.full_estimate(st, adapter=MA.stub())
    mem = est["memory"]
    assert est["pricing_model"]["stub"] and est["size"]["measured"]
    assert 600 < mem["frontier_bytes_per_particle"] < 1600 and mem["probe"]["quarters"] == T.PROBE_QUARTERS
    assert mem["peak_mb"] > T.MEMORY_BASE_BYTES["stub"] / 2**20
    text = S.format_estimate(est, st)
    assert "定价模型：" + MA.STUB_LABEL in text and "本机计算约" in text
    man = S.build("e12", st, base=tmp_path / "s", cache_dir=tmp_path / "c", adapter=MA.stub())
    assert man["pricing_model"]["stub"] and man["export"]["enabled"] is False
    size = sum(p.stat().st_size for p in (tmp_path / "s" / "e12").rglob("*") if p.is_file())
    assert 0.75 < size / est["size"]["bytes"] < 1.25, (size, est["size"]["bytes"])


# ================================================================================================ the real quarterly model

@needs_quarterly
def test_the_quarterly_model_is_the_default_when_its_data_exist(monkeypatch):
    monkeypatch.delenv("MSL_AGENT_MODEL")
    monkeypatch.setitem(MA._FALLBACK, "key", None)
    ad = MA.load_adapter()
    info = MA.model_info(ad)
    assert info["kind"] == "quarterly" and not info["stub"] and info["structural"] and info["trial"]
    assert info["label"] == MA.QUARTERLY_LABEL and not info["why_stub"]
    assert MA.validate_adapter(ad) == []
    fi = OV.merge(ad.field_info())
    assert fi["x_oil_cause"]["cls"] == "event_label" and fi["x_oil_cause"]["enum"]["supply"] == 1.0
    assert fi["d_ff"]["level_floor"] == pytest.approx(0.1) and fi["d_ff"]["given_mode"] == "do"
    for k, sd in T.key_scales(fi):                                # node distances read the model's own sds
        assert sd == pytest.approx((fi[k]["hist"] or {}).get("sd_q_1985_2019"))


@pytest.fixture(scope="module")
def qad():
    if not READY:
        pytest.skip(f"needs the quarterly panel: {WHY}")
    return MA.load_adapter(MA.QUARTERLY_SPEC)


@needs_quarterly
def test_real_model_do_pins_leave_same_quarter_macro_unmoved(qad):
    n = 400
    s0 = qad.root_state(n)
    a0 = qad.complete(s0, MA.Given.empty(n), np.random.default_rng(5))
    e0 = ("d_unrate", "d_ip", "d_cpi_serv", "d_cpi_goods")
    hike = qad.complete(s0, MA.Given.from_dict({"d_ff": a0.x["d_ff"] + 0.5}, n), np.random.default_rng(5))
    assert all(np.allclose(hike.x[k], a0.x[k]) for k in e0)       # do: E0 unmoved (same stream)
    assert np.mean(hike.x["d_y2"] - a0.x["d_y2"]) > 0.05             # markets respond on impact
    v = a0.x["d_transfers"] + 2.0
    fiscal = qad.complete(s0, MA.Given.empty(n), np.random.default_rng(5), structural={"d_transfers": v})
    assert all(np.allclose(fiscal.x[k], a0.x[k]) for k in e0) and np.allclose(fiscal.x["d_transfers"], v)
    reduced = qad.complete(s0, MA.Given.from_dict({"d_transfers": v}, n), np.random.default_rng(5))
    assert not np.allclose(reduced.x["d_rgdp"], a0.x["d_rgdp"])      # an observation imputes its association
    last = qad.complete(s0, MA.Given.empty(n), np.random.default_rng(5), structural={"d_itinv": a0.x["d_itinv"] + 2.4})
    assert all(np.allclose(last.x[k], a0.x[k]) for k in e0 + ("d_capex", "d_rgdp"))


@needs_quarterly
def test_real_model_floor_and_the_engine(qad):
    n = 200
    s = qad.root_state(n)
    s["s_ff"] = np.full(n, 0.35)
    d = qad.complete(s, MA.Given.from_dict({"d_ff": np.full(n, -0.5)}, n), np.random.default_rng(1))
    assert np.allclose(s["s_ff"] + d.x["d_ff"], 0.1) and any(nt.get("field") == "d_ff" for nt in d.notes)
    eng = T.Engine(small(n_particles=100), adapter=qad, root_accounts=None)
    assert eng.structural and eng.ff_floor == pytest.approx(0.1)


@needs_quarterly
@pytest.mark.skipif(not HAS_ROOT, reason="needs data/results/agents_accounts_root.json")
def test_a_mock_snapshot_on_the_real_model_builds_clean_and_expands(tmp_path, qad):
    st = S.default_settings(2, 3, n_particles=150)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        man = S.build("q3", st, base=tmp_path / "s", cache_dir=tmp_path / "c", adapter=qad)
    bad = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not bad, [str(w.message) for w in bad[:3]]
    assert man["status"] == "done" and not man["stub"] and man["pricing_model"]["kind"] == "quarterly"
    assert all("tail_share" in lv for lv in man["levels"]) and man["export"]["enabled"] is False
    root = tmp_path / "s" / "q3"
    ex = S.Explorer(root, adapter=None, cache_dir=tmp_path / "c")
    assert ex.model() is qad or MA.same_model_kind(ex.model().version, qad.version)
    det = ex.level(0, "detail")["details"]
    d0 = next(iter(det.values()))
    fed = next(a for a in d0["next"] if a["actor"] == "fed")
    assert fed["prior"]["source"] == "a0_bins" and fed["prior"]["pinned"]          # 2026Q4: the pinned meeting
    assert d0["model"]["structural"] and d0["model"]["ff_floor"] == pytest.approx(0.1)
    pruned = [r for r in ex.level(2)["nodes"] if r["status"] == "pruned"]
    assert pruned
    ent = ex.expand(pruned[0]["key"])
    assert ent["detail"] is not None and ent["record"]["children"] and ent["model_changed"] is False
    for nd_key in list(ex.overlay_index())[:3]:
        cl = ex._saved_cloud(nd_key)
        if cl is not None:
            assert not MA.state_tail_mask(cl.state).any()


@needs_quarterly
@pytest.mark.skipif(not HAS_ROOT, reason="needs data/results/agents_accounts_root.json")
def test_real_model_tree_reproduces_the_models_own_distribution(tmp_path, qad):
    """Neutrality (§9.4): the model tree's leaf mixture against the model's step rollout with the same tail rule.
    Tolerance: means within 3.5 Monte Carlo standard errors per seed (σ/√N), medians within 0.1 sd, and the spread
    as the 10–90% width, whose ratio's mean over the seeds lies in [0.9, 1.1]. Not the sd: unemployment's right tail
    is heavy, and a leaf mixture of B · N resampled particles understates a heavy-tailed sd (8 seeds: unemployment's
    sd ratio 0.75–1.18, mean 0.91, the same with the tail rule of P14 or the wider one of its review)."""
    D, N, n_ref = 3, 300, 6000
    s = qad.root_state(n_ref)
    rng = np.random.default_rng(99)
    cum = np.zeros(n_ref)
    for _ in range(D):
        d = qad.complete(s, MA.Given.empty(n_ref), rng)
        bad = MA.tail_mask(s, d, end_state=qad.update_state(s, d))
        if bad.any():
            good = np.flatnonzero(~bad)
            idx = np.arange(n_ref)
            idx[bad] = good[rng.integers(0, len(good), int(bad.sum()))]
            s, d, cum = MA.take_state(s, idx), MA.as_draws(d).take(idx), cum[idx]
        s = qad.update_state(s, d)
        cum = cum + d.y["us_equity"]
    ref = np.stack([cum, s["s_unrate"], s["s_ff"], s["s_core_yoy"]], axis=1)
    rm, rs, rmed = ref.mean(0), ref.std(0), np.median(ref, axis=0)
    rw = np.percentile(ref, 90, axis=0) - np.percentile(ref, 10, axis=0)
    ratios = []
    for seed in range(4):
        st = T.TreeSettings.coerce(dict(depth=D, B=3, n_particles=N, seed=seed, tree_kind="model", prior_trees=False))
        run = T.TreeRun(st, T.Store(tmp_path / f"m{seed}"), adapter=qad, cache_dir=tmp_path / "c")
        run.run()
        V = np.concatenate([np.stack([nd.cloud.cum["us_equity"], nd.cloud.state["s_unrate"], nd.cloud.state["s_ff"],
                                      nd.cloud.state["s_core_yoy"]], axis=1) for nd in run.leaves])
        W = np.concatenate([np.full(nd.cloud.n, nd.p_path / nd.cloud.n) for nd in run.leaves])
        W = W / W.sum()
        mu = (W[:, None] * V).sum(0)
        sd = np.sqrt((W[:, None] * (V - mu) ** 2).sum(0))
        assert np.all(np.abs((mu - rm) / (rs / np.sqrt(N))) < 3.5), (seed, (mu - rm) / (rs / np.sqrt(N)))
        med = np.array([T.wquantiles(V[:, j], W, (0.5,))[0] for j in range(V.shape[1])])
        assert np.all(np.abs(med - rmed) / rs < 0.1), (seed, (med - rmed) / rs)
        q = np.array([T.wquantiles(V[:, j], W, (0.1, 0.9)) for j in range(V.shape[1])])
        ratios.append((q[:, 1] - q[:, 0]) / rw)
        assert np.all(sd / rs < 1.3), (seed, sd / rs)
    r = np.mean(ratios, axis=0)
    assert np.all((r > 0.9) & (r < 1.1)), r


# ================================================================================================ the P14 review
# Stubs that behave like the quarterly stack in the respect each test needs: InertStub's satellite levers never feed
# back (structural= values and d_bs / d_tariff / d_lf are set after the draw, nothing responds to them; d_gz is a
# core variable it refuses as an intervention), as on the stack, where only the ten core variables carry lags.

class InertStub(MA.GaussianStub):
    capabilities = frozenset({"structural"})
    LAST = ("d_bs", "d_tariff", "d_lf")

    def __init__(self):
        super().__init__()
        self.version = "gaussian-stub-inert"
        self.calls = []

    def field_info(self):
        info = super().field_info()
        for f, mode in (("d_ff", "do"), ("d_bs", "do"), ("d_tariff", "do_last"), ("d_lf", "do_last")):
            info.setdefault(f, {})["given_mode"] = mode
        return info

    def complete(self, state, given, rng, *, structural=None):
        self.calls.append({"given": tuple(getattr(given, "ids", ())), "structural": tuple(sorted(structural or {}))})
        last = {}
        if given is not None and len(given.ids):
            keep = [j for j, f in enumerate(given.ids) if f not in self.LAST]
            last.update({f: given.values[:, j] for j, f in enumerate(given.ids) if f in self.LAST})
            given = MA.Given(tuple(given.ids[j] for j in keep), given.values[:, keep])
        for f, v in (structural or {}).items():
            if f == "d_gz":
                raise ValueError("d_gz：没有对应的政策冲击，不能写成干预")
            last[f] = np.asarray(v, float)
        d = super().complete(state, given, rng)
        for f, v in last.items():
            if f in d.x:
                d.x[f] = np.where(np.isfinite(v), v, d.x[f])
        return d


def _menu_option(aid, oid, p=1.0, magnitude=None):
    o = A.option_summary(aid, A.ACTORS[aid].find(oid)[1])
    return {"id": oid, "label": o["zh"], "p_used": p, "p_prior": p, "magnitude": magnitude, "param": o["param"],
            "bin": tuple(o["bin"]) if o.get("bin") else None, "status_quo": o["status_quo"], "source": "menu"}


def _bk(eng, nd, dec, q):
    cl = eng.clusters(nd, dec)[0]
    alloc = eng.allocate_cluster(nd, cl, dec, eng.a0(nd), T.seed_for(1, nd.key, "alloc"))
    a0 = eng.a0(nd)
    bk = eng.build_bk(nd, cl, dec, a0, alloc, T.seed_for(1, nd.key, "bk"))
    bk["alloc"] = alloc
    return cl, a0, bk


def test_levers_the_model_does_not_transmit_get_absent_channels_the_response_table_and_qe_paths():
    """P14 (channels "off"): inert levers keep the response table and QE's G08 paths; with template channels on
    (the default) both are replaced (tests/test_agents_channels.py)."""
    ad = InertStub()
    eng = T.Engine(small(n_particles=200, first_quarter="2027Q1", channels="off"), adapter=ad,
                   root_accounts=AC.load_root())
    tr = eng.transmission
    assert {"d_bs", "d_transfers", "d_deficit", "d_perstax", "d_tariff", "d_lf", "d_itinv"} <= set(eng.inert)
    assert "d_ff" not in eng.inert and tr["levers"]["d_ff"]["max_z"] > 0.05      # the stub conditions on d_ff
    assert eng.meta["d_transfers"]["channel_class"] == "absent" and eng.meta["d_bs"]["inert"]
    assert eng.meta["d_transfers"]["channel_class_model"] == "moderate"
    root = eng.root()
    q = "2027Q1"
    dec = eng.decide(root, q, eng.a0(root), {}, [], [])
    dec["actors"]["congress"]["options"] = [_menu_option("congress", "stimulus_small", 1.0, 0.8)]
    dec["actors"]["congress"]["attributes"] = []
    fed = dec["actors"]["fed"]
    fed["attributes"] = [{"domain": "balance_sheet", "zh": "资产负债表", "options": [{"id": "bs_qe", "label": "量化宽松", "p": 1.0}]}]
    cl, a0, bk = _bk(eng, root, dec, q)
    assert "d_transfers" in bk["structural"].ids and "d_cons_goods" in bk["given"].ids      # §6.3's response table
    assert any(r["field"] == "d_cons_goods" and r["R"] == 2.1 for r in bk["response"])
    assert {"d_tp10", "d_y2"} <= set(bk["structural"].ids) and "d_gz" in bk["given"].ids      # G08: set last / observed
    assert bk["qe_particles"] == len(bk["rows"]) and "d_bs" in bk["written"]                   # the attribute wrote
    sd_tp = eng.sd["d_tp10"]
    tp = np.asarray(bk["draws"].x["d_tp10"]) - np.asarray(a0.x["d_tp10"])[bk["rows"]]
    assert -1.5 * sd_tp < np.mean(tp) < -0.5 * sd_tp
    assert any(n.get("code") == "qe_paths" for n in bk["notes"])
    assert "G1" in eng.gate(root, cl, bk, dec, None)                       # the absent channel opens the synthesizer
    lines = T.transmission_zh(tr, eng.meta)
    assert lines and "卫星变量" in lines[0] and "量化宽松" in lines[0] and "G08" not in lines[0]


def test_attribute_options_write_like_main_options():
    eng = T.Engine(small(n_particles=150, first_quarter="2027Q1"), adapter=MA.stub(), root_accounts=AC.load_root())
    root = eng.root()
    dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
    dec["actors"]["fed"]["attributes"] = [{"domain": "balance_sheet", "zh": "资产负债表",
                                           "options": [{"id": "bs_runoff", "label": "恢复缩表", "p": 1.0}]}]
    pres = dec["actors"]["president"]
    pres["attributes"] = [{"domain": "immigration", "zh": "移民", "options": [{"id": "imm_tighten", "label": "收紧", "p": 1.0}]}]
    _, _, bk = _bk(eng, root, dec, "2027Q1")
    assert {"d_bs", "d_lf"} <= set(bk["given"].ids) and {"d_bs", "d_lf"} <= set(bk["written"])
    j = bk["given"].ids.index("d_bs")
    assert -0.5 - 4 * 0.2 < np.nanmean(bk["given"].values[:, j]) < -0.5 + 4 * 0.2      # N(−0.5, 0.2) per particle


def test_a_due_commitment_is_routed_like_the_option_that_made_it():
    r = A.FieldResolver.load()
    cms = A.commitments_of(A.option_payload("congress", "stimulus_small", 0.8, resolver=r), "2026Q4")
    assert cms and all(cm["write"]["protocol"] == "response_table" for cm in cms)
    old = [{**cm, "write": {k: v for k, v in cm["write"].items() if k != "protocol"}} for cm in cms]
    for commitments in (cms, old):                      # old: saved before protocols travelled with their writes
        eng = T.Engine(small(n_particles=150, first_quarter="2027Q1", channels="off"), adapter=InertStub(),
                       root_accounts=AC.load_root())
        root = eng.root()
        root.commitments = [dict(cm, due_quarter="2027Q1") for cm in commitments]
        dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
        cong = dec["actors"]["congress"]
        o = next(o for o in cong["options"] if o["id"] == "as_committed")
        cong["options"] = [{**o, "p_used": 1.0, "p_prior": 1.0}]
        cong["attributes"] = []
        _, _, bk = _bk(eng, root, dec, "2027Q1")
        assert "d_transfers" in bk["structural"].ids and "d_transfers" not in bk["given"].ids
        assert "response_table" in bk["protocols"] and "d_cons_goods" in bk["given"].ids
    # the stub keeps the two-step protocol for a due commitment too (the hyperscaler fallback is thin)
    eng = T.Engine(small(n_particles=150, first_quarter="2027Q1"), adapter=MA.stub(), root_accounts=AC.load_root())
    root = eng.root()
    big = A.option_payload("bigtech", "capex_raise_large", 15.0, resolver=eng.resolver)
    root.commitments = [dict(cm, due_quarter="2027Q1") for cm in A.commitments_of(big, "2026Q4")
                        if cm["due_quarter"] == "2027Q1"]
    dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
    bt = dec["actors"]["bigtech"]
    o = next(o for o in bt["options"] if o["id"] == "as_committed")
    bt["options"] = [{**o, "p_used": 1.0, "p_prior": 1.0}]
    _, _, bk = _bk(eng, root, dec, "2027Q1")
    assert "thin" in bk["protocols"] and "d_itinv" in bk["given"].ids and "d_rgdp" in bk["given"].ids


def test_natures_mixture_keeps_every_intervention_with_its_particle():
    ad = InertStub()
    eng = T.Engine(small(n_particles=500, first_quarter="2027Q1"), adapter=ad, root_accounts=AC.load_root())
    root = eng.root()
    q = "2027Q1"
    a0 = eng.a0(root)
    dec = eng.decide(root, q, a0, {}, [], [])
    cong = dec["actors"]["congress"]
    cong["options"] = [_menu_option("congress", "stimulus_small", 0.12, 0.8), _menu_option("congress", "no_new_law", 0.88)]
    cong["attributes"] = []
    cl, a0, bk = _bk(eng, root, dec, q)
    bk = eng.sanitize(root, q, cl, bk)
    hz = {h: dict(v) for h, v in dec["hazards"].items()}
    live = [h for h, v in hz.items() if 0 < v["m"] < 0.3]
    assert live, hz
    for h in live:
        hz[h]["p"] = min(0.045, 2.5 * hz[h]["m"])
    out = eng.apply_nature(root, cl, bk, {**dec, "hazards": hz}, {})
    assert not np.array_equal(out["rows"], bk["rows"])                        # the mixture resampled the particles
    j = out["structural"].ids.index("d_transfers")
    code = np.asarray(out["alloc"]["assign"]["congress"])
    stim = [o["id"] for o in cong["options"]].index("stimulus_small")
    assert np.array_equal(np.isfinite(out["structural"].values[:, j]), code == stim)
    # the do_last holds of the hazard fields are the particle's own values, and the pins match the draw
    jm = out["structural"].ids.index("d_metals_oil")
    ok = np.isfinite(out["structural"].values[:, jm])
    assert np.allclose(out["structural"].values[ok, jm], np.asarray(out["draws"].x["d_metals_oil"])[ok])
    assert np.allclose(out["structural"].values[code == stim, j], np.asarray(out["draws"].x["d_transfers"])[code == stim])


def test_the_fed_is_pinned_in_the_models_units_and_the_target_is_a_state():
    eng = T.Engine(small(n_particles=120), adapter=MA.stub(), root_accounts=AC.load_root())
    root = eng.root()
    assert np.allclose(root.cloud.state["ff_target"], T.ROOT_FF_TARGET)
    u = eng.fed_units(root.cloud.state, "2026Q4", eng.fed_pin("2026Q4"))
    f4 = T.FOMC_LAST_MONTH_SHARE[4]
    assert np.allclose(u["carry"], T.ROOT_FF_TARGET - 3.75) and u["b"] == pytest.approx(f4) and u["a"] == 0.0
    free = eng.fed_units(root.cloud.state, "2027Q1")
    assert free["b"] == pytest.approx(0.5 * (1 + T.FOMC_LAST_MONTH_SHARE[1]))
    mv = np.array([-0.25, 0.0, 0.25])
    assert np.allclose(eng.implied_moves(u, eng.model_units(u, np.tile(mv, 40))), np.tile(mv, 40))
    dec = eng.decide(root, "2026Q4", eng.a0(root), {}, [], [])
    dec["actors"]["fed"]["options"] = [_menu_option("fed", "hold")]
    dec["actors"]["fed"]["attributes"] = []
    _, _, bk = _bk(eng, root, dec, "2026Q4")
    j = bk["given"].ids.index("d_ff")
    slopes = bk["given"].values[:, j] - (T.ROOT_FF_TARGET - 3.75)
    assert abs(float(np.median(slopes))) < 0.05          # a hold is the carry-over of September's hike, not 0
    ch = eng.expand(root)[0][0]
    assert "ff_target" in ch.cloud.state and ch.summary["rep"].get("ff_target") is not None
    assert "目标" in eng.state_table(root) and "3.75–4.00" in eng.state_table(root)


def test_a_cut_the_floor_cannot_deliver_is_flagged_and_the_notes_are_kept():
    eng = T.Engine(small(n_particles=120, first_quarter="2027Q1"), adapter=MA.stub(), root_accounts=AC.load_root())
    root = eng.root()
    root.cloud.state["s_ff"][:] = 0.2
    root.cloud.state["ff_target"][:] = 0.2
    dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
    dec["actors"]["fed"]["options"] = [_menu_option("fed", "cut_25")]
    _, _, bk = _bk(eng, root, dec, "2027Q1")
    assert "floor_short" in bk["flags"] and any(n.get("code") == "floor_engine" for n in bk["notes"])
    j = bk["given"].ids.index("d_ff")
    assert np.all(root.cloud.state["s_ff"] + bk["given"].values[:, j] >= -1e-9)
    merged = T.merge_notes([{"field": "d_ff", "particles": 3, "text": "x"}, {"field": "d_ff", "particles": 4, "text": "x"},
                            {"code": "tail", "paths": 2, "text": "t"}])
    assert merged[0]["particles"] == 7 and T.note_kind({"text": "政策利率钉在下限 0.1% 以下：按下限交付"}) == "floor"
    assert T.note_kind({"text": "d_gpr：超出历史范围，按 ±4 个标准差截断"}) == "saturation"


def test_the_tail_rule_reads_levels_cpi_and_every_asset():
    n = 6
    st = {"s_ff": np.full(n, 3.0), "s_gz": np.full(n, 1.0), "s_unrate": np.full(n, 4.0), "s_y10": np.full(n, 4.0)}
    x = {"d_ff": np.zeros(n), "d_gz": np.zeros(n), "d_unrate": np.zeros(n), "d_y10": np.zeros(n)}
    y = {a: np.zeros(n) for a in ("us_equity", "gold", "crude_oil", "treasury_10y")}
    x["d_ff"][0] = 18.0                     # policy rate 21
    x["d_y10"][1] = -4.5                    # 10-year yield −0.5
    y["gold"][2] = 1.1                      # beyond ±1 for gold
    y["crude_oil"][3] = 1.2                 # crude oil's bound is 1.5
    end = {"s_ff": st["s_ff"] + x["d_ff"], "s_core_yoy": np.r_[2.0, 2.0, 2.0, 2.0, 16.0, 2.0]}
    why = MA.tail_reasons(st, MA.Draws(x, y, {}, np.zeros(n)), end_state=end)
    bad = MA.tail_mask(st, MA.Draws(x, y, {}, np.zeros(n)), end_state=end)
    assert list(np.flatnonzero(bad)) == [0, 1, 2, 4]
    assert why["s_ff"][0] and why["s_y10"][1] and why["returns"][2] and not why["returns"][3] and why["s_core_yoy"][4]


def test_the_tail_share_is_cumulated_along_each_path(tmp_path):
    run = T.TreeRun(small(depth=3), T.Store(tmp_path / "t"), adapter=TailStub(), cache_dir=tmp_path / "c")
    run.run()
    for nd in run.leaves:
        tail = nd.summary["tail"]
        assert tail["cum"] >= tail["share"] - 1e-9 and tail["cum"] > 0.2 and "why" in tail
    lv = run.level_stats
    assert all(b["tail_cum"] >= a["tail_cum"] - 1e-9 for a, b in zip(lv, lv[1:]))
    assert all("ident_fail" in s for s in lv)


@pytest.mark.skipif(not HAS_ROOT, reason="needs data/results/agents_accounts_root.json")
def test_the_books_deficit_returns_to_its_baseline_and_debt_stays_positive():
    acc = AC.Accounts.from_root(AC.load_root(), 400, resolver=A.FieldResolver.load())
    base0 = float(acc.s["def_gdp"][0])
    rng = np.random.default_rng(3)
    x0 = {k: np.zeros(400) for k in ("d_ff", "d_y10", "d_y2", "d_rgdp", "d_core", "d_cpi_goods", "d_cpi_serv",
                                      "d_unrate", "d_cons_goods", "d_cons_serv", "d_capex", "d_oil", "d_usd", "d_hpi",
                                      "d_profits", "d_chargeoff", "d_tariff")}
    x0["d_rgdp"][:] = 2.0
    x0["d_cpi_goods"][:] = x0["d_cpi_serv"][:] = 2.5
    y0 = {a: np.zeros(400) for a in AC.ACCOUNT_ASSETS}
    for t in range(80):
        act = np.where(np.arange(400) < 200, -0.3, 0.0) if t == 0 else np.zeros(400)
        x = {**x0, "d_deficit": 1.4 * rng.standard_normal(400) + act, "_agent_deficit_action": act}
        acc.step(x, y0, {"debt_limit_suspend_q": 80.0} if t == 0 else {})     # the limit follows the debt
        assert np.all(acc.s["dhp"] >= 0) and acc.ok(), acc.resid
    acc.s["dhp"][:3] = 0.0
    acc.s["fh"][:3] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.all(np.isfinite(acc.views()["s_foreign_hold"]))
    gap = acc.s["def_gdp"] - acc.s["def_base"]
    assert np.percentile(np.abs(gap), 99) < 16.0                 # a random walk would be about 1.4 × √80 = 12.5 sd
    assert np.allclose(acc.s["def_base"][:200], base0 - 0.3) and np.allclose(acc.s["def_base"][200:], base0)


def test_identity_follows_the_datas_bytes_not_its_file_time(tmp_path, monkeypatch):
    import os
    import types

    from scenariolab.qmodel import assets as AS
    from scenariolab.qmodel import config_q as cq
    from scenariolab.qmodel import panel as qpanel
    pq, sel = tmp_path / "panel_quarterly.csv", tmp_path / "selection.json"
    pq.write_text("q,x\n2026Q3,1\n", encoding="utf-8")
    sel.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cq, "PANEL_Q_PATH", pq)
    monkeypatch.setattr(AS, "SELECTION_PATH", sel)
    monkeypatch.setattr(qpanel, "load_meta", lambda: {"info": {"inputs_digest": "i", "card_digest": "c"}})
    fake = types.SimpleNamespace(stub=False, capabilities=frozenset(),
                                 stack=types.SimpleNamespace(version="qmodel-r1/2026Q3/forward/s0/b200/abcdef123456"),
                                 version="qadapter-1/qmodel-r1/2026Q3/forward/s0/b200/abcdef123456/m1/o1")
    ident = MA.identity(fake)
    assert "abcdef123456" not in ident and ident.startswith("qadapter-1/qmodel-r1/2026Q3")
    st = pq.stat()
    os.utime(pq, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10**9))      # a byte-identical copy: a new file time
    assert MA.identity(fake) == ident and MA.same_version(ident, fake) and MA.same_version(fake.version, fake)
    pq.write_text("q,x\n2026Q3,2\n", encoding="utf-8")                 # new bytes: a new identity
    assert MA.identity(fake) != ident and not MA.same_version(ident, fake)
    assert T.versions(fake)["model_adapter"] == MA.identity(fake)
    assert MA.short_version(fake.version) == "2026Q3/abcdef123456"


def test_a_resumed_snapshot_keeps_the_model_it_was_priced_with(tmp_path, monkeypatch):
    st = S.default_settings(1, 2, n_particles=80, hard_cap=1)
    with pytest.raises(S.BuildPaused):
        S.build("rp", st, base=tmp_path / "s", cache_dir=tmp_path / "c", adapter=MA.stub())
    root = tmp_path / "s" / "rp"
    man = S.read_manifest(root)
    assert man["status"] == "paused" and man["pause"]["reason"] == "budget" and man["backend"] == "mock"

    def no_default(*a, **k):
        raise AssertionError("a resume must not load the environment's default model")
    monkeypatch.setattr(MA, "load_adapter", no_default)
    # another fit of the same family is refused (it would mix two fits in one tree) unless allowed
    T.write_json(root / "manifest.json", {**man, "versions": {**man["versions"], "model_adapter": "gaussian-stub-0"}})
    with pytest.raises(ValueError, match="混用两次拟合"):
        S.build("rp", st, resume=True, base=tmp_path / "s", cache_dir=tmp_path / "c", budget_confirmed=True)
    assert S.read_manifest(root)["status"] == "paused"
    T.write_json(root / "manifest.json", man)
    done = S.build("rp", st, resume=True, base=tmp_path / "s", cache_dir=tmp_path / "c", budget_confirmed=True)
    assert done["status"] == "done" and len(done["pricing_history"]) == 1
    ev = [e for e in T.Journal(root / "journal.jsonl").events if e["type"] in ("run_started", "run_resumed")]
    assert len(ev) >= 2 and all(e["model"]["kind"] == "stub" for e in ev)


def test_a_resume_after_a_refit_is_recorded_when_allowed(tmp_path):
    st = S.default_settings(1, 2, n_particles=80, hard_cap=1)
    with pytest.raises(S.BuildPaused):
        S.build("rf", st, base=tmp_path / "s", cache_dir=tmp_path / "c", adapter=MA.stub())
    root = tmp_path / "s" / "rf"
    man = S.read_manifest(root)
    T.write_json(root / "manifest.json", {**man, "versions": {**man["versions"], "model_adapter": "gaussian-stub-0"},
                                          "pricing_history": [{"from_level": 0, "identity": "gaussian-stub-0"}]})
    done = S.build("rf", st, resume=True, base=tmp_path / "s", cache_dir=tmp_path / "c", budget_confirmed=True,
                   allow_model_change=True)
    hist = done["pricing_history"]
    assert len(hist) == 2 and hist[-1]["identity"] == MA.identity(MA.stub()) and hist[-1]["from_level"] >= 0
    assert any(e["type"] == "model_changed" for e in T.Journal(root / "journal.jsonl").events)


def test_a_budget_pause_resumes_from_the_server(tmp_path):
    import time

    from scenariolab.agents import jobs
    jm = jobs.JobManager(tmp_path / "agents", auto_resume=False)

    def wait(states):
        for _ in range(600):
            m = S.read_manifest(jm.snaps_dir / "bp") or {}
            if m.get("status") in states and jm.active is None:
                return m
            time.sleep(0.1)
        raise AssertionError(f"snapshot still {m.get('status')}")
    jm.start_snapshot("bp", {"B": 1, "depth": 2, "n_particles": 80, "hard_cap": 1}, confirm=True)
    m = wait(("paused", "failed", "done"))
    assert m["status"] == "paused" and m["pause"]["reason"] == "budget", m.get("error")
    jm.start_snapshot("bp", None, confirm=True, resume=True)
    m = wait(("paused", "failed", "done"))
    assert m["status"] == "done", (m.get("pause"), m.get("error"))
    assert m.get("resume_pause") is None and m["hard_cap"] > 1


def test_real_calls_are_never_priced_by_a_fallback_stub(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "real_backend", lambda: True)
    S.refuse_fallback_stub(MA.stub())                      # MSL_AGENT_MODEL=stub (tests/conftest.py): explicit
    monkeypatch.delenv("MSL_AGENT_MODEL")
    monkeypatch.setitem(MA._FALLBACK, "why", "缺少季度模型的数据：panel_quarterly.csv")
    with pytest.raises(ValueError, match="回退到了桩模型"):
        S.refuse_fallback_stub(MA.stub())
    assert MA.model_info(MA.stub())["fallback"]
    monkeypatch.setattr(MA, "load_adapter", lambda *a, **k: MA.stub())
    assert S.main(["--beam", "1", "--quarters", "1", "--id", "fb", "--yes"]) == 2
    from scenariolab.agents import jobs
    jm = jobs.JobManager(tmp_path / "agents", auto_resume=False)
    with pytest.raises(ValueError, match="回退到了桩模型"):
        jm.start_snapshot("fb", {"B": 1, "depth": 1}, confirm=True)
    monkeypatch.undo()
    monkeypatch.setenv("MSL_AGENT_MODEL", "scenariolab.no_such_module:make")
    with pytest.raises(ValueError, match="无法加载"):
        MA.load_adapter()


def test_cached_levels_need_the_same_model_versions_and_backend(tmp_path):
    from scenariolab.agents import jobs
    jm = jobs.JobManager(tmp_path / "agents", auto_resume=False)
    st = T.TreeSettings.coerce(dict(depth=3, B=2))
    d = jm.runs_dir / "r_x"
    d.mkdir(parents=True)
    base = {"settings": st.to_dict(), "levels_done": 3, "versions": T.versions(MA.stub()), "backend": "mock"}
    T.write_json(d / "run.json", base)
    assert jm.cached_levels(st, adapter=MA.stub()) == 3
    for bad in ({**base, "versions": {**base["versions"], "model_adapter": "qadapter-1/x"}},
                {**base, "backend": "claude_code"}, {k: v for k, v in base.items() if k != "backend"}):
        T.write_json(d / "run.json", bad)
        assert jm.cached_levels(st, adapter=MA.stub()) == 0


def test_the_model_tree_menu_ignores_the_pinned_meeting_and_tariffs_use_the_decision_table():
    agent = T.Engine(small(n_particles=100, prior_source="a0"), adapter=MA.stub(), root_accounts=None)
    model = T.Engine(small(n_particles=100, tree_kind="model"), adapter=MA.stub(), root_accounts=None)
    root_a, root_m = agent.root(), model.root()
    da = agent.decide(root_a, "2026Q4", agent.a0(root_a), {}, [], [])
    dm = model.decide(root_m, "2026Q4", model.a0(root_m), {}, [], [])
    ids_a = {o["id"] for o in da["actors"]["fed"]["options"]}
    ids_m = {o["id"] for o in dm["actors"]["fed"]["options"]}
    assert "cut_50" not in ids_a and {"cut_50", "hike_50"} <= ids_m          # the model tree: plain bins of d_ff
    assert dm["actors"]["fed"]["prior"]["source"] == "a0_bins" and not dm["actors"]["fed"]["prior"].get("pinned")
    assert da["actors"]["president"]["prior"]["source"] != "a0_bins"           # tariffs: the decision record
    assert dm["actors"]["president"]["prior"]["source"] == "a0_bins"           # the model tree reads every instrument


def test_bin_actors_report_how_many_particles_left_their_bin():
    eng = T.Engine(small(n_particles=200, first_quarter="2027Q1", tree_kind="base_rate", prior_source="a0"),
                   adapter=MA.stub(), root_accounts=AC.load_root())
    root = eng.root()
    dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
    assert eng.is_bin_actor(dec["actors"]["fed"])
    _, _, bk = _bk(eng, root, dec, "2027Q1")
    assert "fed" in bk["bin_drift"] and 0.0 <= bk["bin_drift"]["fed"] <= 1.0


def test_the_size_estimate_follows_the_level_and_the_width():
    assert T.level_bytes_factor(0, 1, 3, 1) < 0.2 and T.level_bytes_factor(2, 2, 3, 2) == pytest.approx(1.0)
    assert T.level_bytes_factor(40, 2, 3, 2) == pytest.approx(T.DEPTH_BYTES_FACTOR)
    assert T.level_bytes_factor(1, 1, 3, 30) < 0.5 * T.level_bytes_factor(1, 1, 3, 2)
    wide = S.size_estimate(S.default_settings(30, 4), probe={"frontier_bytes_per_particle": 1440})
    assert wide["frontier_mb"] < 45.0                                  # B 30 × 4 measured 32 MB (was estimated 50)


def test_an_overlay_branch_priced_by_another_fit_says_so(tmp_path):
    st = S.default_settings(1, 2, n_particles=80)
    S.build("ov", st, base=tmp_path / "s", cache_dir=tmp_path / "c", adapter=MA.stub())
    root = tmp_path / "s" / "ov"
    ex = S.Explorer(root, cache_dir=tmp_path / "c")
    pruned = [r["key"] for d in ex.store.levels() for r in (ex.level(d) or {}).get("nodes", []) if r["status"] == "pruned"]
    assert len(pruned) >= 2
    ex.expand(pruned[0])
    assert ex.record(pruned[0])["overlay"]["model_changed"] is False
    man = S.read_manifest(root)
    T.write_json(root / "manifest.json", {**man, "versions": {**man["versions"], "model_adapter": "gaussian-stub-0"}})
    ex2 = S.Explorer(root, cache_dir=tmp_path / "c")
    assert ex2.model_changed()
    ex2.expand(pruned[1])
    ov = ex2.record(pruned[1])["overlay"]
    assert ov["model_changed"] is True and ov["model_fit"] is not None


@needs_quarterly
def test_real_model_satellite_levers_are_measured_inert_and_the_price_puzzle_disclosed(qad):
    eng = T.Engine(small(n_particles=100), adapter=qad, root_accounts=None)
    assert {"d_bs", "d_transfers", "d_deficit", "d_perstax", "d_tariff", "d_lf", "d_itinv"} <= set(eng.inert)
    assert "d_ff" not in eng.inert and eng.transmission["levers"]["d_ff"]["max_z"] > 0.05
    ff = eng.transmission["ff_irf"]
    assert ff["unrate_q12"] > 0                         # a hike raises unemployment later
    lines = T.transmission_zh(eng.transmission, eng.meta)
    assert any("卫星变量" in ln for ln in lines) and (not ff["puzzle"] or any("价格之谜" in ln for ln in lines))
    assert not eng._intervenes("d_gz") and eng._intervenes("d_tp10")
    u = eng.fed_units(eng.root().cloud.state, "2026Q4", eng.fed_pin("2026Q4"))
    assert np.allclose(u["carry"], T.ROOT_FF_TARGET - 3.75)        # the model's s_ff is September's average


@needs_quarterly
@pytest.mark.skipif(not HAS_ROOT, reason="needs data/results/agents_accounts_root.json")
def test_real_model_books_keep_their_identities_over_20_quarters(tmp_path, qad):
    st = S.default_settings(2, 20, n_particles=150)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        man = S.build("q20", st, base=tmp_path / "s", cache_dir=tmp_path / "c", adapter=qad)
    bad = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not bad, [f"{w.filename}:{w.lineno} {w.message}" for w in bad[:3]]
    assert man["status"] == "done" and man["transmission"]["inert"]
    assert all(lv["ident_fail"] == 0 for lv in man["levels"]), [(lv["level"], lv["ident_max"]) for lv in man["levels"]]
    ex = S.Explorer(tmp_path / "s" / "q20", adapter=qad)
    for d in ex.store.levels():
        for r in (ex.level(d) or {}).get("nodes", []):
            if r["status"] != "pruned" and (r.get("summary") or {}).get("identities"):
                assert r["summary"]["identities"]["ok"], (d, r["summary"]["identities"])
    for key in man["leaves"]:
        cl = ex.store.load_leaf_cloud(key)
        assert np.all(cl.acc.s["dhp"] >= 0) and np.all(np.isfinite(cl.state["ff_target"]))
