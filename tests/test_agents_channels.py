"""Template channels (the policy-channels round; scenariolab/agents/channels.py and channels.json).

The first part runs everywhere (the table, the per-particle layer, the engine on the Gaussian stub and on stubs that
behave like the quarterly stack): coverage of every catalogue option, template ids against the quarterly schema, the
judge's statements at default intensity, monotone domains, the protocol's rules (holds, displays, the growth-field
rule, two passes for IP, room, caps, habituation, standing facts, commitments, the queue), the replaced response
table and QE paths, the synthesizer's gate and mechanisms, the model tree's neutrality, saved settings, the price-
puzzle overlay's deconvolution. The second part needs the quarterly panel (data/ is git-ignored) and is skipped
without it: every mapped option measured on the stack against the judge's readings, QE at the root, the two-pass
rule, the overlay's acceptance chain.

Every LLM interaction is the mock backend (tests/conftest.py); no test starts Claude Code or spends tokens."""
import copy
import math
from collections import defaultdict

import numpy as np
import pytest

from scenariolab.agents import accounts as AC
from scenariolab.agents import actors as A
from scenariolab.agents import channels as CH
from scenariolab.agents import model_adapter as MA
from scenariolab.agents import priors as P
from scenariolab.agents import snapshot as S
from scenariolab.agents import synthesizer as SY
from scenariolab.agents import tree as T

READY, WHY = MA.quarterly_data_ready()
HAS_ROOT = AC.load_root() is not None


def small(**kw) -> T.TreeSettings:
    base = dict(depth=2, B=2, n_particles=120, max_calls=400, concurrency=1, prior_trees=False)
    base.update(kw)
    return T.TreeSettings.coerce(base)


def catalogue_ids() -> dict:
    return {aid: [o.id for d in a.domains for o in d.options] for aid, a in A.ACTORS.items()}


# ================================================================================================ the table

def test_the_table_covers_every_catalogue_option_and_its_templates_resolve():
    assert CH.problems(catalogue=catalogue_ids()) == []
    t = CH.table()
    opts = t["options"]
    assert len(opts) == 89 and sum(1 for e in opts.values() if e["status"] == "statements") == 46
    assert {"xdate_passed", "pandemic"} <= set(t["events"])
    for k in ("nature.oil_disruption", "nature.gpr_episode", "nature.financial_accident"):   # D8: no footprint
        assert opts[k]["status"] == "no_statement" and opts[k]["zh"]
    names = CH.template_names()
    assert names["G05"].startswith("关税战") and names["S10"] and names["P08"]
    for e in opts.values():                        # every entry says why, in Chinese, for the UI
        assert e.get("zh"), e["option"]
    fed = [k for k in opts if k.startswith("fed.") and not k.startswith("fed.bs_")]
    assert len(fed) == 7 and all(opts[k]["kind"] == "model" for k in fed)        # the rate transmits: no channel
    never = set(t["protocol"]["never"])
    assert all(s["field"] not in never for e in opts.values() for s in CH.statements_of(e))


def _judge_rows(entry_judge) -> dict:
    rows = defaultdict(float)
    for f, mode, q, size, _scope, _cond in entry_judge:
        a, _, b = str(q).partition("-")
        for k in range(int(a), int(b or a) + 1):
            rows[(f, mode, k)] += 0.0 if mode == "hold" else float(size)
    return rows


def _our_rows(key: str, e: dict, levers: dict, oil: dict) -> dict:
    aid, oid = key.split(".")
    rows = defaultdict(float)
    pay = A.option_payload(aid, oid, None, resolver=A.FieldResolver.load())

    def add(stmts, factor=1.0, shift=0):
        for s in stmts:
            for q in s["q"]:
                v = 0.0 if s["mode"] == "hold" else (float(s.get("value", 0.0)) if s["mode"] == "label" else s["size"] * factor)
                if s["mode"] == "display" and s.get("cap") is not None:
                    v = math.copysign(min(abs(v), float(s["cap"])), v)
                rows[(s["field"], s["mode"], q + shift)] += v
    add(e["statements"])
    for w in pay["writes"]:
        if e.get("lever") == w["field"] and w["kind"] == "point" and int(w["dq"]) == 0:
            add(levers[w["field"]]["up" if w["a"] > 0 else "down"], factor=float(w["a"]))
        if w["field"] in (e.get("writes") or {}):
            add(e["writes"][w["field"]], factor=float(w["a"]), shift=int(w["dq"]))
    if e.get("oil"):
        z = sum(float(ev["a"]) for ev in pay["events"]["oil_z"])
        add(oil["per_z"], factor=z)
        rows[("d_metals_oil", "shift", 0)] += oil["signature"]["size_per_z"] * z
        if abs(z) >= A.OIL_COMPOSITE["cause_label_min_abs_z"]:
            rows[("x_oil_cause", "label", 0)] += 1.0
    return rows


TARIFF_UNITS = 0.0553 / 0.041          # the review's unit fix of the tariff pass-through (0.21% of the level per pp)


def test_the_table_reproduces_the_judges_statements_at_default_intensity():
    """channels.json normalizes the judge's statements per unit of each trigger (tariff pp by sign, fiscal and
    balance-sheet writes, the fallback's d_itinv, the oil composite z, onsets at x0); at every option's default
    intensity (the catalogue's own default payload, commitments included except a tariff commitment, which the judge
    states only when executed) they sum to the judge's statements per field, mode and quarter, with the review's two
    documented changes: the tariff goods pass-through in its stated units (×0.0553/0.041) and the core-goods
    increments on d_core (P16), whose size is the protocol factor times the goods statement's."""
    t = CH.table()
    k_core = t["protocol"]["core_goods"]["factor_sd"]
    checked = 0
    for key, e in t["options"].items():
        if e["status"] != "statements":
            continue
        a, b = _judge_rows(e["judge"]), _our_rows(key, e, t["levers"], t["oil"])
        if e.get("lever") == "d_tariff":
            a = defaultdict(float, {k: v * (TARIFF_UNITS if k[0] == "d_cpi_goods" and k[1] == "shift" else 1.0)
                                    for k, v in a.items()})
        for k in [k for k in b if k[0] == "d_core"]:
            goods = b.get(("d_cpi_goods", "shift", k[2]), 0.0)
            if not (e.get("statements") and any(s.get("cond") for s in e["statements"] if s["field"] == "d_core")):
                assert b[k] == pytest.approx(goods * k_core, abs=0.006), (key, k)
            b.pop(k)
        for k in set(a) | set(b):
            assert abs(a.get(k, 0.0) - b.get(k, 0.0)) <= 0.006, (key, k, a.get(k), b.get(k))
        checked += 1
    assert checked == 46
    for ev in t["events"].values():
        a = _judge_rows(ev["judge"])
        b = defaultdict(float)
        for s in ev["statements"]:
            if s["field"] == "d_core":
                continue
            for q in s["q"]:
                b[(s["field"], s["mode"], q)] += 0.0 if s["mode"] == "hold" else (
                    float(s.get("value", 0.0)) if s["mode"] == "label" else s["size"])
        assert all(abs(a.get(k, 0.0) - b.get(k, 0.0)) <= 1e-6 for k in set(a) | set(b)), ev["key"]


def _size(key, field, mode="shift", q=0):
    e = CH.option(key)
    return sum(s["size"] for s in e["statements"] if s["field"] == field and s["mode"] == mode and q in s["q"]
               and s.get("cond") != "while_fact")


def test_monotone_domains_and_documented_asymmetries():
    ladder = ["president.pressure_public", "president.nominate_dovish", "president.attempt_removal", "courts.allow_removal"]
    for f, sign in (("d_y10", 1), ("d_usd", -1), ("d_gz", 1)):
        sizes = [sign * _size(k, f) for k in ladder]
        assert sizes == sorted(sizes) and sizes[0] > 0, (f, sizes)
        assert sizes[-1] == pytest.approx(4 * sizes[0])                    # 0.25 / 0.5 / 0.75 / 1.0 of G29
    assert 0 < CH.option("president.tariff_up_small")["tpu"] < CH.option("president.tariff_up_large")["tpu"]
    assert _size("bigtech.capex_cut", "d_gz") < _size("bigtech.capex_slash", "d_gz")
    qe = CH.option("fed.bs_qe")["writes"]["d_bs"]
    qt = CH.option("fed.bs_runoff")["writes"]["d_bs"]
    y = lambda ws: next(s["size"] for s in ws if s["field"] == "d_y10")  # noqa: E731
    assert y(qt) == pytest.approx(0.5 * y(qe), rel=0.02)                  # QT at half of QE per unit
    up, dn = (next(s["size"] for s in CH.table()["levers"]["d_tariff"][side] if s["field"] == "d_cpi_goods")
              for side in ("up", "down"))
    assert dn == pytest.approx(0.5 * up)                                  # cuts pass through at half
    raise_ = CH.option("bigtech.capex_raise_large")
    assert not any(s["field"] == "d_gz" for s in CH.statements_of(raise_))   # no spread narrowing in booms (G06)


# ================================================================================================ the layer

SD = {"d_cpi_goods": 5.0, "d_ip": 1.3, "d_oil": 19.7, "d_gz": 0.47, "d_epu": 25.0, "d_sent": 6.2, "x_supply_chain": 7.6,
      "d_exports": 8.1, "d_y10": 0.54, "d_usd": 3.4, "d_cpi_serv": 1.2, "d_tp10": 0.41, "d_infexp_lr": 0.12,
      "d_unrate": 0.3, "d_rgdp": 2.25, "d_core": 1.13, "d_cons_goods": 4.06, "d_wage": 0.96, "d_y2": 0.58,
      "d_metals_oil": 16.3, "x_oil_cause": 0.5, "d_cpbill": 0.2, "d_capex": 7.3, "d_lshare": 1.0, "d_profits": 7.3,
      "d_igrea": 28.8, "d_usd_em": 3.3, "d_cons_serv": 1.56, "d_lf": 0.97}


def layer(kernel=None) -> CH.Layer:
    return CH.Layer(SD, SD.keys(), facts_exit={f: v.get("exit") for f, v in A.FACTS.items()}, kernel=kernel)


def a0x(m, rng=None):
    rng = rng or np.random.default_rng(0)
    return {f: rng.standard_normal(m) * sd for f, sd in SD.items()}


def base_state(L, m, **levels):
    s = L.empty_state(m)
    s.update({"s_gz": np.full(m, levels.get("gz", 1.5)), "s_unrate": np.full(m, levels.get("u", 5.0)),
              "s_spread": np.full(m, levels.get("sp", 2.0))})
    return s


def choice(aid, oid, m, magnitude=None, writes=(), rows=None):
    pay = A.option_payload(aid, oid, magnitude, resolver=A.FieldResolver.load())
    return CH.Choice(actor=aid, option=oid, rows=np.arange(m) if rows is None else rows, magnitude=pay.get("magnitude"),
                     writes=list(writes), facts=[f["id"] for f in pay["facts"]], clears=list(pay["clears"]))


def tariff_write(m, pp, option="tariff_up_large"):
    return {"field": "d_tariff", "kind": "point", "value": np.full(m, float(pp)), "a": float(pp), "option": option}


def test_a_tariff_writes_flows_holds_displays_and_a_news_step():
    L, m = layer(), 400
    st = base_state(L, m)
    res = L.step(st, a0x(m), [choice("president", "tariff_up_large", m, 2.0, [tariff_write(m, 2.0)])],
                 np.random.default_rng(1))
    d, mk = res.shift["d_cpi_goods"]
    assert mk.all() and np.mean(d) / SD["d_cpi_goods"] == pytest.approx(0.1106, rel=0.08)  # 0.0553 sd per pp
    core, mkc = res.core                                    # the core-goods pass-through: an increment on core (P16)
    k_core = CH.table()["protocol"]["core_goods"]["factor_sd"]
    assert mkc.all() and np.mean(core) / SD["d_core"] == pytest.approx(0.1106 * k_core, rel=0.08)
    assert "d_core" not in res.shift and "d_core" in res.fields
    assert res.hold["d_oil"].all()                                                           # not an oil shock
    gz, _ = res.shift["d_gz"]
    assert np.mean(gz) / SD["d_gz"] == pytest.approx(0.3, rel=0.1)                           # TPU 1.2 units
    assert "x_supply_chain" in res.display and "d_epu" in res.display and res.ip is None      # IP falls only later
    q = res.next[CH.QUEUE_KEY]
    j = L.slot_ix[("d_ip", "shift")]
    assert np.all(q[:, :4, j].mean(axis=0) < 0)                                             # IP loss at q1–4
    # the uncertainty tail's give-back (decay 8 after its last quarter, q3): quarters 5–11 positive, then nothing; the
    # lever's efficiency loss (supply side) stays in the level
    assert np.all(q[:, 4:11, j].mean(axis=0) > 0) and np.all(q[:, 11:, j] == 0)
    assert "d_tariff" in res.covered
    # the next quarter applies the due entries, also after resampling (the queue is a per-particle state)
    idx = np.random.default_rng(2).integers(0, m, m)
    st2 = {k: np.asarray(v)[idx] for k, v in res.next.items()}
    st2.update({k: np.asarray(st[k])[idx] for k in ("s_gz", "s_unrate", "s_spread")})
    res2 = L.step(st2, a0x(m), [], np.random.default_rng(3))
    ip, mk = res2.ip
    assert mk.all() and np.allclose(ip, q[idx, 0, j]) and "d_gz" not in res2.shift
    assert res2.hold["d_oil"].all() and "d_cpi_goods" in res2.shift                          # q1 of the q0–2 flows


def test_status_quo_options_write_nothing():
    L, m = layer(), 50
    st = base_state(L, m)
    hold = choice("president", "tariff_hold", m, writes=[{"field": "d_tariff", "kind": "bin", "value": np.zeros(m),
                                                            "a": None, "option": None}])
    res = L.step(st, a0x(m), [hold, choice("congress", "no_new_law", m), choice("fed", "hold", m)], np.random.default_rng(1))
    assert not res.any() and not res.shift and not res.hold and not res.display and "d_tariff" in res.covered
    for key, e in CH.table()["options"].items():
        if e["status"] == "no_statement":
            assert not e.get("statements") and not e.get("writes"), key


def test_news_habituates_by_one_over_k_and_flows_do_not():
    L, m = layer(), 300
    st = base_state(L, m)
    rows = []
    for _ in range(3):
        res = L.step(st, a0x(m), [choice("president", "tariff_up_large", m, 2.0, [tariff_write(m, 2.0)])],
                     np.random.default_rng(5))
        rows.append((np.mean(res.shift["d_gz"][0]), np.mean(res.shift["d_cpi_goods"][0])))
        st = {**st, **res.next}
    gz = [r[0] for r in rows]
    assert gz[1] == pytest.approx(gz[0] / 2, rel=0.02) and gz[2] == pytest.approx(gz[0] / 3, rel=0.02)
    goods0 = rows[0][1]
    assert rows[2][1] > goods0                       # flows never habituate (and the queued q1–2 add up)


def test_news_habituates_within_a_group_so_alternating_escalations_still_repeat():
    L, m = layer(), 300
    st = base_state(L, m)
    res = L.step(st, a0x(m), [choice("president", "tariff_up_large", m, 2.0, [tariff_write(m, 2.0)])],
                 np.random.default_rng(5))
    first = np.mean(res.shift["d_gz"][0]) / 0.3
    st = {**st, **res.next}
    res = L.step(st, a0x(m), [choice("president", "tariff_up_small", m, 0.6, [tariff_write(m, 0.6, "tariff_up_small")])],
                 np.random.default_rng(6))
    second = np.mean(res.shift["d_gz"][0]) / 0.15
    assert second == pytest.approx(first / 2, rel=0.05)           # the small escalation is the group's second repeat
    assert L.news_group["president.deal"] == L.news_group["president.accept_lapse"]


def ladder_layer() -> CH.Layer:
    return CH.Layer(SD, SD.keys(), facts_exit={}, kernel=None)    # no exits: every rung lives


def test_the_independence_ladder_writes_levels_not_sums():
    """Climbed in sequence (public pressure, a dovish nominee, a removal attempt, the court allowing removal), the
    rungs' onsets add up to the full template once (y10 +0.8, dollar −1.0, GZ +0.3 sd), and the services drift is the
    highest live rung's (0.15 sd a quarter), not the sum of every live rung's; it runs while the ruling stands."""
    L, m = ladder_layer(), 400
    st = base_state(L, m)
    tot = defaultdict(float)
    steps = [("president", "pressure_public"), ("president", "nominate_dovish"), ("president", "attempt_removal"),
             ("courts", "allow_removal")]
    for i, (aid, oid) in enumerate(steps):
        res = L.step(st, a0x(m), [choice(aid, oid, m)], np.random.default_rng(10 + i))
        for f in ("d_y10", "d_usd", "d_gz"):
            if f in res.shift:
                tot[f] += float(np.mean(res.shift[f][0])) / SD[f]
        st = {**st, **res.next}
    assert tot["d_y10"] == pytest.approx(0.8, rel=0.03) and tot["d_usd"] == pytest.approx(-1.0, rel=0.03)
    assert tot["d_gz"] == pytest.approx(0.3, rel=0.03)
    drifts = []
    for k in range(12):                                            # past the protocol's 8 quarters
        res = L.step(st, a0x(m), [], np.random.default_rng(30 + k))
        d, mk = res.shift["d_cpi_serv"]
        drifts.append(float(np.mean(d[mk])) / SD["d_cpi_serv"])
        st = {**st, **res.next}
    assert all(x == pytest.approx(0.15, rel=0.1) for x in drifts), drifts
    # a lower rung after a higher one adds nothing
    L2, st2 = ladder_layer(), base_state(ladder_layer(), m)
    r = L2.step(st2, a0x(m), [choice("president", "attempt_removal", m)], np.random.default_rng(1))
    r2 = L2.step({**st2, **r.next}, a0x(m), [choice("president", "pressure_public", m)], np.random.default_rng(2))
    assert "d_y10" not in r2.shift and "d_usd" not in r2.shift


def test_demand_side_activity_is_given_back_and_supply_side_is_not():
    L, m = layer(), 200
    st = base_state(L, m)
    w = {"field": "d_deficit", "kind": "shift", "value": np.full(m, 1.0), "a": 1.0, "option": "stimulus_large"}
    res = L.step(st, a0x(m), [CH.Choice(actor="congress", option="as_committed", rows=np.arange(m),
                                        writes=[{**w, "option": "stimulus_large"}])], np.random.default_rng(1))
    ip0 = float(np.mean(res.ip[0]))
    q = res.next[CH.QUEUE_KEY][:, :, L.slot_ix[("d_ip", "shift")]]
    back = q.mean(axis=0)
    assert ip0 > 0 and np.all(back[:12] < 0) and back.sum() == pytest.approx(-ip0, rel=1e-3)   # 12 quarters, all of it
    # the oil composite's activity lag (supply side) stays in the level: no give-back
    res = L.step(st, a0x(m), [], np.random.default_rng(1), oil_z=np.full(m, 1.0))
    q = res.next[CH.QUEUE_KEY][:, :, L.slot_ix[("d_ip", "shift")]].mean(axis=0)
    assert np.all(q[:4] < 0) and np.all(q[4:] == 0)


def test_the_overlay_is_an_increment_and_not_a_channel_field():
    ker = {f: np.r_[0.0, np.full(CH.OVERLAY_K, -0.1)] for f in CH.OVERLAY_FIELDS}
    L, m = layer(kernel=ker), 100
    st = base_state(L, m)
    st[CH.FFS_KEY][:, 0] = 0.5
    res = L.step(st, a0x(m), [], np.random.default_rng(1), surprise=np.zeros(m), overlay=True)
    assert set(res.overlay_inc) == set(CH.OVERLAY_FIELDS) and res.overlay.all()
    assert not (set(CH.OVERLAY_FIELDS) & res.fields) and not res.shift and res.core is None
    d, mk = res.overlay_inc["d_core"]
    assert mk.all() and np.allclose(d, -0.05)


def test_the_channel_tables_version_carries_a_hash_of_its_bytes():
    v = CH.version()
    assert v.startswith(CH.table()["version"]) and "sha256:" in v and len(v.split("sha256:")[1]) == 12
    cov = T.channel_coverage()
    assert cov["dormant"] == sum(1 for e in CH.table()["options"].values() if e["actor"] == "treasury")
    assert cov["template"] == 44                                    # the Treasury's two dormant options left out


def test_standing_facts_fire_once_per_onset_and_their_flows_follow_the_particle():
    L, m = layer(), 500
    st = base_state(L, m)
    res = L.step(st, a0x(m), [choice("president", "pressure_public", m)], np.random.default_rng(1))
    assert "d_y10" in res.shift and res.hold["d_cpi_serv"].all()           # onset: 1/4 of G29, prices held
    j = L.facts.index("fed_pressure_public")
    assert np.all(res.next[CH.FACT_KEY][:, j] == 1)
    st = {**st, **res.next}
    res2 = L.step(st, a0x(m), [choice("president", "pressure_public", m)], np.random.default_rng(2))
    assert "d_y10" not in res2.shift                                        # re-choosing a live fact: no new onset
    drift, mk = res2.shift["d_cpi_serv"]
    assert mk.all() and np.mean(drift) / SD["d_cpi_serv"] == pytest.approx(0.0375, rel=0.15)   # the drift at age 1
    alive = np.mean(res2.next[CH.FACT_KEY][:, j] >= 1)
    assert 0.15 < alive < 0.45                                              # exit 0.7 per particle
    # immigration: flows each live quarter at the latest magnitude, unemployment held, output down
    L2 = layer()
    st = base_state(L2, m)
    w = {"field": "d_lf", "kind": "shift", "value": np.full(m, -1.0), "a": -1.0, "option": "imm_tighten"}
    res = L2.step(st, a0x(m), [choice("president", "imm_tighten", m, -1.0, [w])], np.random.default_rng(1))
    ip, _ = res.ip
    assert np.mean(ip) / SD["d_ip"] == pytest.approx(-0.114, rel=0.1) and res.hold["d_unrate"].all()
    res = L2.step({**st, **res.next}, a0x(m), [], np.random.default_rng(2))
    assert res.ip is not None and np.mean(res.ip[0]) < 0                    # the fact lives on (exit 0.2)


def test_conditions_on_live_facts_and_the_start_of_quarter_state():
    L, m = layer(), 200
    st = base_state(L, m)
    res = L.step(st, a0x(m), [choice("courts", "uphold_protection", m)], np.random.default_rng(1))
    assert not res.any()                                                    # no attempt: the status quo
    res = L.step(st, a0x(m), [choice("president", "attempt_removal", m)], np.random.default_rng(1))
    st2 = {**st, **res.next}
    res2 = L.step(st2, a0x(m), [choice("courts", "uphold_protection", m)], np.random.default_rng(2))
    assert np.mean(res2.shift["d_y10"][0]) < 0                              # half the premium reversed
    j = L.facts.index("fed_removal_attempt")
    assert np.all(res2.next[CH.FACT_KEY][:, j] == 0)                        # and the attempt's drift ends
    # the tight-market term of a stimulus only where unemployment starts below 4.5
    w = {"field": "d_transfers", "kind": "shift", "value": np.full(m, 2.0), "a": 2.0, "option": "stimulus_small"}
    tight = L.step(base_state(L, m, u=4.2), a0x(m), [choice("congress", "stimulus_small", m, 0.8, [w])],
                   np.random.default_rng(3))
    slack = L.step(base_state(L, m, u=5.2), a0x(m), [choice("congress", "stimulus_small", m, 0.8, [w])],
                   np.random.default_rng(3))
    assert "d_cpi_goods" in tight.shift and "d_cpi_goods" not in slack.shift and slack.hold["d_cpi_goods"].all()
    assert tight.ip is not None and np.mean(tight.ip[0]) / SD["d_ip"] == pytest.approx(0.092, rel=0.1)


def test_room_caps_and_beyond_history_templates():
    L, m = layer(), 300
    w = {"field": "d_bs", "kind": "normal", "value": np.full(m, 3.0), "a": 3.0, "option": "bs_qe"}
    near = L.step(base_state(L, m, gz=0.93), a0x(m), [choice("fed", "bs_qe", m, writes=[w])], np.random.default_rng(1))
    far = L.step(base_state(L, m, gz=1.5), a0x(m), [choice("fed", "bs_qe", m, writes=[w])], np.random.default_rng(1))
    ratio = np.mean(near.shift["d_gz"][0]) / np.mean(far.shift["d_gz"][0])
    assert ratio == pytest.approx((0.93 - 0.65) / (1.50 - 0.65), rel=0.02)   # falls scaled by the room to p01
    assert np.mean(far.shift["d_y10"][0]) == pytest.approx(np.mean(near.shift["d_y10"][0]))   # yields: no room rule
    # beyond history: the blockade's +2 sd GZ is exempt from the 3 sd cap and flagged; its IP loss is per quarter
    res = L.step(base_state(L, m), a0x(m), [choice("china", "tw_blockade", m)], np.random.default_rng(2))
    assert res.beyond.all() and any(nt.get("code") == "beyond_history" for nt in res.notes)
    assert np.mean(res.ip[0]) / SD["d_ip"] == pytest.approx(-0.761, rel=0.1)
    assert res.label["x_oil_cause"][0] == 1.0                                 # supply
    # an in-history pile-up is capped at 3 sd per field-quarter
    big = [choice("president", "tariff_up_large", m, 5.4, [tariff_write(m, 60.0)])]
    res = L.step(base_state(L, m), a0x(m), big, np.random.default_rng(3))
    assert np.max(np.abs(res.shift["d_cpi_goods"][0])) <= 3.0 * SD["d_cpi_goods"] + 1e-9
    assert any(nt.get("code") == "channel_cap" for nt in res.notes)


def test_commitments_fire_their_writes_flows_when_executed_and_custom_options_get_the_lever_channel():
    L, m = layer(), 200
    st = base_state(L, m)
    w = tariff_write(m, -0.5, option="deal")
    res = L.step(st, a0x(m), [CH.Choice(actor="president", option="as_committed", rows=np.arange(m), writes=[w])],
                 np.random.default_rng(1))
    assert np.mean(res.shift["d_cpi_goods"][0]) < 0 and "d_gz" not in res.shift   # cut flows, no TPU news again
    cu = CH.Choice(actor="president", option="custom", rows=np.arange(m),
                   writes=[{"field": "d_tariff", "kind": "point", "value": np.full(m, 1.0), "a": 1.0, "option": None}])
    res = L.step(st, a0x(m), [cu], np.random.default_rng(1))
    assert np.mean(res.shift["d_cpi_goods"][0]) > 0 and "d_gz" not in res.shift and "d_tariff" in res.covered


def test_the_oil_composite_states_oil_at_its_own_elasticity_and_an_activity_lag():
    L, m = layer(), 200
    z = np.full(m, 1.5)
    res = L.step(base_state(L, m), a0x(m), [choice("belligerents", "escalate", m)], np.random.default_rng(1), oil_z=z)
    assert np.mean(res.shift["d_oil"][0]) == pytest.approx(0.555 * 1.5 * SD["d_oil"], rel=1e-6)
    j = L.slot_ix[("d_ip", "shift")]
    assert np.all(res.next[CH.QUEUE_KEY][:, :4, j] < 0)                    # z carries the noise: no soft draw


def test_events_xdate_and_pandemic_fire_once_per_onset():
    L, m = layer(), 100
    st = base_state(L, m)
    x = np.zeros(m, bool)
    x[:40] = True
    # a TGA below the floor without a live standoff is bookkeeping (Congress raised the limit or never fought it)
    res0 = L.step(st, a0x(m), [], np.random.default_rng(1), xdate=x)
    assert "d_gz" not in res0.shift and not res0.next[CH.FACT_KEY][:, L.facts.index("xdate_passed")].any()
    st[CH.FACT_KEY][:, L.facts.index("debt_limit_standoff")] = 1.0            # the standoff kept through the deadline
    res = L.step(st, a0x(m), [], np.random.default_rng(1), xdate=x)
    d, mk = res.shift["d_gz"]
    assert mk[:40].all() and not mk[40:].any() and res.beyond[:40].all()
    res2 = L.step({**st, **res.next}, a0x(m), [], np.random.default_rng(2), xdate=x)
    assert "d_gz" not in res2.shift                                         # still past the X-date: no new onset
    # the first time only: a TGA that recovers and falls below the floor again does not repeat the shock
    res3 = L.step({**st, **res2.next}, a0x(m), [], np.random.default_rng(3), xdate=np.zeros(m, bool))
    x2 = np.zeros(m, bool)
    x2[30:60] = True
    res4 = L.step({**st, **res3.next}, a0x(m), [], np.random.default_rng(4), xdate=x2)
    d4, mk4 = res4.shift["d_gz"]
    assert not mk4[:40].any() and mk4[40:60].all() and not mk4[60:].any()
    res = L.step(st, a0x(m), [], np.random.default_rng(1), node_facts={"pandemic": True})
    assert np.mean(res.shift["d_unrate"][0]) / SD["d_unrate"] == pytest.approx(6.0, rel=0.05)
    assert np.mean(res.ip[0]) / SD["d_ip"] == pytest.approx(-5.0, rel=0.05)
    res2 = L.step({**st, **res.next}, a0x(m), [], np.random.default_rng(2), node_facts={"pandemic": True})
    assert np.mean(res2.ip[0]) > 0 and np.mean(res2.shift["d_unrate"][0]) < 0     # the restart, queued


# ================================================================================================ the engine (stubs)

class InertStub(MA.GaussianStub):
    """Like the quarterly stack in the respect these tests need: satellite levers set by structural= and d_bs,
    d_tariff, d_lf are set after the draw (nothing responds to them); d_gz cannot be an intervention."""
    capabilities = frozenset({"structural"})
    LAST = ("d_bs", "d_tariff", "d_lf")

    def __init__(self):
        super().__init__()
        self.version = "gaussian-stub-inert-ch"

    def field_info(self):
        info = super().field_info()
        for f, mode in (("d_ff", "do"), ("d_bs", "do"), ("d_tariff", "do_last"), ("d_lf", "do_last")):
            info.setdefault(f, {})["given_mode"] = mode
        return info

    def complete(self, state, given, rng, *, structural=None):
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


class StackLikeStub(MA.GaussianStub):
    """The stub declaring "structural" (d_ff and d_bs "do", d_tariff and d_lf "do_last") and recording every
    completion; satellites set by structural= come back as set (do_last)."""
    capabilities = frozenset({"structural"})

    def __init__(self):
        super().__init__()
        self.version = "gaussian-stub-stacklike"
        self.calls = []

    def field_info(self):
        info = super().field_info()
        for f, mode in (("d_ff", "do"), ("d_bs", "do"), ("d_tariff", "do_last"), ("d_lf", "do_last")):
            info.setdefault(f, {})["given_mode"] = mode
        return info

    def complete(self, state, given, rng, *, structural=None):
        rec = {"given": dict(given.to_dict()) if given is not None else {},
               "structural": {k: np.asarray(v).copy() for k, v in (structural or {}).items()}}
        d = super().complete(state, given, rng)
        for f, v in (structural or {}).items():
            if f in d.x:
                d.x[f] = np.where(np.isfinite(v), v, d.x[f])
        rec["d_ip"] = np.asarray(d.x["d_ip"], float).copy()
        self.calls.append(rec)
        return d


def _menu_option(aid, oid, p=1.0, magnitude=None):
    o = A.option_summary(aid, A.ACTORS[aid].find(oid)[1])
    return {"id": oid, "label": o["zh"], "p_used": p, "p_prior": p, "magnitude": magnitude, "param": o["param"],
            "bin": tuple(o["bin"]) if o.get("bin") else None, "status_quo": o["status_quo"], "source": "menu"}


def _force(dec, aid, oid, magnitude=None, attrs=None):
    ab = dec["actors"][aid]
    ab["options"] = [_menu_option(aid, oid, 1.0, magnitude)]
    ab["attributes"] = attrs or []
    ab["branching"] = False


def _bk(eng, nd, dec):
    a0 = eng.a0(nd)
    cl = eng.clusters(nd, dec)[0]
    alloc = eng.allocate_cluster(nd, cl, dec, a0, T.seed_for(1, nd.key, "alloc"))
    bk = eng.build_bk(nd, cl, dec, a0, alloc, T.seed_for(1, nd.key, "bk"))
    bk["alloc"] = alloc
    return cl, a0, bk


def _quiet(dec, keep=()):
    """Every actor at its status quo except `keep` (main and attribute domains)."""
    for aid, ab in dec["actors"].items():
        if aid in keep:
            continue
        sq = next((o for o in ab["options"] if o.get("status_quo")), ab["options"][0])
        ab["options"] = [{**sq, "p_used": 1.0}]
        ab["attributes"] = []
        ab["cond"] = None
        ab["follower_of"] = None


def test_channels_replace_the_response_table_and_qe_paths_and_displays_go_last():
    eng = T.Engine(small(n_particles=200, first_quarter="2027Q1"), adapter=StackLikeStub(), root_accounts=AC.load_root())
    assert eng.channels_on and eng.layer is not None
    root = eng.root()
    assert CH.QUEUE_KEY in root.cloud.state
    dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
    _quiet(dec, keep=("congress", "fed"))
    _force(dec, "congress", "stimulus_small", 0.8)
    _force(dec, "fed", "hold", attrs=[{"domain": "balance_sheet", "zh": "资产负债表",
                                      "options": [{"id": "bs_qe", "label": "量化宽松", "p": 1.0}]}])
    cl, a0, bk = _bk(eng, root, dec)
    assert bk["response"] == [] and bk["qe_particles"] == 0 and not any(n.get("code") == "qe_paths" for n in bk["notes"])
    assert "d_cons_goods" in bk["structural"].ids and "d_cons_goods" not in bk["given"].ids    # the table, as display
    assert {"d_y10", "d_gz", "d_usd", "d_ip"} <= set(bk["given"].ids)                          # G08 and FIS-TR
    assert {"d_tp10", "d_y2"} <= set(bk["structural"].ids)                                    # G08 displays
    assert {"d_bs", "d_transfers"} <= set(bk["template_levers"])
    assert set(bk["channel_fields"]) <= set(bk["written"]) and "d_gz" not in bk["lever_written"]
    assert len(eng.adapter_calls if hasattr(eng, "adapter_calls") else eng.model.calls) >= 2   # two passes for IP
    rows = {(r["source"], r["field"], r["mode"]) for r in bk["channels"]}
    assert ("fed.bs_qe", "d_gz", "shift") in rows and ("congress.stimulus_small", "d_ip", "shift") in rows
    # the private next-state columns ride with the draw and become the child's state
    assert all(CH.NEXT_PREFIX + k in bk["draws"].x for k in CH.STATE_KEYS)
    ch = eng.expand(root)[0][0]
    assert all(k in ch.cloud.state for k in CH.STATE_KEYS)


def test_two_passes_make_an_ip_statement_an_increment_on_the_quarters_other_pins():
    """capex_slash: pass 1 completes the quarter with the GZ statement and without the IP statement (IP free);
    pass 2 pins IP at pass 1's IP plus the statement's increment (+0.0236 sd per pp of the −5.6 pp writes)."""
    ad = StackLikeStub()
    eng = T.Engine(small(n_particles=150, first_quarter="2027Q1"), adapter=ad, root_accounts=AC.load_root())
    root = eng.root()
    dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
    _quiet(dec, keep=("bigtech",))
    _force(dec, "bigtech", "capex_slash", -35.0)
    ad.calls.clear()
    cl, a0, bk = _bk(eng, root, dec)
    passes = [c for c in ad.calls if "d_gz" in c["given"]]
    assert len(passes) >= 2
    p1, p2 = passes[0], passes[1]
    assert "d_ip" not in p1["given"] or not np.isfinite(p1["given"]["d_ip"]).any()
    assert np.all(np.isfinite(p2["given"]["d_ip"])) and np.allclose(bk["draws"].x["d_ip"], p2["given"]["d_ip"])
    inc = np.asarray(p2["given"]["d_ip"]) - p1["d_ip"]
    want = 0.0236 * eng.sd["d_ip"] * 0.16 * -35.0
    assert float(np.mean(inc)) == pytest.approx(want, rel=0.2)


def test_the_growth_field_rule_holds_ip_whenever_a_channel_writes_unemployment_or_gdp():
    eng = T.Engine(small(n_particles=160, first_quarter="2027Q1"), adapter=StackLikeStub(), root_accounts=AC.load_root())
    root = eng.root()
    for attrs, field, mode in (
            ([{"domain": "workforce", "zh": "用工", "options": [{"id": "layoffs", "label": "裁员潮", "p": 1.0}]}],
             "d_unrate", "given"),):
        dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
        _quiet(dec, keep=("bigtech",))
        _force(dec, "bigtech", "capex_hold", 0.0, attrs=attrs)
        _, a0, bk = _bk(eng, root, dec)
        g = bk["given"].to_dict()
        assert np.isfinite(g[field]).all() and np.isfinite(g["d_ip"]).all()
    # a shutdown's GDP display (set last) comes with the IP hold too
    nd = eng.root()
    nd.quarter = "2027Q3"                                                  # the next quarter is 2027Q4: a deadline
    dec = eng.decide(nd, "2027Q4", eng.a0(nd), {}, [], [])
    cg = dec["actors"]["congress"]
    _quiet(dec, keep=("congress",))
    cg["attributes"] = [{"domain": "deadline", "zh": "拨款", "options": [{"id": "shutdown", "label": "停摆", "p": 1.0}]}]
    _, a0, bk = _bk(eng, nd, dec)
    assert "d_rgdp" in bk["structural"].ids
    j = bk["given"].ids.index("d_ip")
    assert np.allclose(bk["given"].values[:, j], np.asarray(a0.x["d_ip"])[bk["rows"]])       # held at A0


def test_g1_opens_only_for_levers_no_channel_covers():
    """A satellite lever the model does not transmit opens the gate (G1) with channels off; with channels on its
    template channel exists and the lever's class is "template" (mechanisms from it are refused in normal states)."""
    for chan, expect in (("off", True), ("on", False)):
        eng = T.Engine(small(n_particles=150, first_quarter="2027Q1", channels=chan), adapter=InertStub(),
                       root_accounts=AC.load_root())
        root = eng.root()
        dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
        _quiet(dec, keep=("president",))
        _force(dec, "president", "tariff_up_small", 0.6)
        cl, a0, bk = _bk(eng, root, dec)
        assert ("G1" in eng.gate(root, cl, bk, dec, None)) is expect, chan
        if chan == "on":
            assert eng.lever_class("d_tariff", bk) == "template" and eng.fact_covered("fed_pressure_public")
    assert SY.CHANNEL_W["template"] == SY.CHANNEL_W["moderate"] and SY.CLASS_RANK["template"] < SY.CLASS_RANK["regime"]


def test_a_template_levers_rarity_in_the_models_history_does_not_open_g1():
    """The template's support is the literature's episodes, not the satellite's own history: a +2pp tariff outside
    d_tariff's 1–99% range (罕见) opens G1 only where no template covers the lever; a value beyond the history's
    extreme (超出, clipped) still does, and so does a cluster at least half of whose particles are beyond history."""
    rare = "d_tariff = 2 在 1–99% 区间之外：罕见"
    clipped = "d_tariff = 40 超出历史极值，截断到 9（超出可测历史，原值留给合成器）"
    other = "d_itinv = 9 在 1–99% 区间之外：罕见"
    assert T.Engine.support_flags({"flags": [rare, other], "template_levers": ["d_tariff"]}) == [other]
    assert T.Engine.support_flags({"flags": [rare], "template_levers": []}) == [rare]
    assert T.Engine.support_flags({"flags": [clipped], "template_levers": ["d_tariff"]}) == [clipped]
    assert T.Engine.support_flags({"flags": ["floor_short", "custom"]}) == []
    eng = T.Engine(small(n_particles=150, first_quarter="2027Q1"), adapter=InertStub(), root_accounts=AC.load_root())
    root = eng.root()
    dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
    _quiet(dec, keep=("president",))
    _force(dec, "president", "tariff_up_small", 0.6)
    cl, a0, bk = _bk(eng, root, dec)
    bk = dict(bk, flags=sorted(set(bk["flags"]) | {"d_tariff = 0.6 在 1–99% 区间之外：罕见"}))
    assert "d_tariff" in bk["template_levers"] and "G1" not in eng.gate(root, cl, bk, dec, None)


class _RareStub(InertStub):
    """The stub with a rarity capability: a particle is rare when its pinned d_gz exceeds `cut` or its d_tariff
    pin exceeds 5 (an action rare by itself); counts the particles each call tests."""
    capabilities = InertStub.capabilities | {"rarity"}
    tested: list = []
    cut = 1.0

    def rarity_detail(self, state, given, *, structural=None):
        n = len(np.asarray(state["s_ff"]))
        g = given.to_dict()
        s = dict(structural or {})
        gz = np.nan_to_num(np.asarray(g.get("d_gz", np.full(n, np.nan)), float), nan=0.0)
        tar = np.nan_to_num(np.asarray(s.get("d_tariff", g.get("d_tariff", np.full(n, np.nan))), float), nan=0.0)
        type(self).tested.append(n)
        return {"percentile": np.where((gz > self.cut) | (tar > 5.0), 0.999, 0.5)}


def test_the_rarity_rule_scales_channels_back_only_where_they_make_the_draw_rare():
    """P10 on a model with the rarity capability: channels that push a particle past the 99th percentile step back;
    a particle rare by its actions' own pins keeps its channels; channel displacements under one sd are not
    tested, and each step re-tests only the particles still rare."""
    ad = _RareStub()
    eng = T.Engine(small(n_particles=200, first_quarter="2027Q1"), adapter=ad, root_accounts=AC.load_root())
    m = 200
    sd = eng.sd["d_gz"]
    d = np.zeros(m)
    ad.cut = sd                            # rare above one sd of d_gz
    d[:50] = 1.4 * sd                      # rare at full size (A0 0 + 1.4 sd)
    d[50:100] = 0.5 * sd                   # under one sd: not tested
    res = CH.Result(m=m)
    res.shift = {"d_gz": (d, d != 0)}
    res.beyond = np.zeros(m, bool)
    res.hold, res.label, res.display, res.ip = {}, {}, {}, None
    a0x = {f: np.zeros(m) for f in ad.x_fields}
    lever = {"d_tariff": np.where(np.arange(m) < 25, 6.0, 1.0)}           # 25 particles rare by the action itself
    state = ad.root_state(m)
    type(ad).tested = []
    scale, n_scaled = eng._rarity_scale(None, np.arange(m), state, res, {}, lever, a0x, set(), m)
    assert type(ad).tested[0] == 50                                          # only the ≥ 1 sd displacements
    assert all(t <= 50 for t in type(ad).tested[1:])
    assert np.all(scale[:25] == 1.0)                                         # rare by the tariff: channels kept
    assert np.all(scale[25:50] == 0.5)                                      # 0.75 is still rare, 0.5 is not
    assert np.all(scale[50:] == 1.0) and n_scaled == 25


def test_mechanisms_never_shift_a_channel_field_nor_against_a_channel():
    class Ans:
        status, axis = "ok", None
        branches = [{"id": "c", "kind": "central", "p_llm": 0.6, "region": None},
                    {"id": "m1", "kind": "mechanism", "p_llm": 0.4,
                     "shifts": {"d_gz": {"mean": 1.0, "sd": 0.1}, "d_ip": {"mean": 2.0, "sd": 0.1},
                                "d_sent": {"mean": -6.0, "sd": 0.5}}}]
    rng = np.random.default_rng(0)
    x = {f: rng.standard_normal(300) for f in ("d_gz", "d_ip", "d_sent", "d_unrate")}
    br, rep = SY.consequences(Ans(), x, written=["d_gz"], causes=["x.y"], classes=["thin"], regime=False, axes=[],
                              signs={"d_ip": -1.0})
    mech = [b for b in br if b["kind"] == "mechanism"]
    assert mech and set(mech[0]["shifts"]) == {"d_sent"}
    assert any("方向与本组的模板通道相反" in r for r in rep)
    # the guard refuses mechanism shifts on the cluster's channel fields (the prompt lists them)
    from scenariolab.llm import agent_guard as G
    from scenariolab.llm.agent_prompts import ClusterSpec, SynthRequest
    spec = ClusterSpec(id="c1", actions=[], written_fields=["d_tariff"], channel_fields=["d_sent"],
                       channels=[{"text": "d_sent 展示"}])
    req = SynthRequest(ctx={"quarter_next": "2027Q1"}, clusters=[spec],
                       fields=[{"id": "d_sent", "mechanism": True}, {"id": "d_unrate", "mechanism": True, "partition": True}])
    raw = {"clusters": [{"cluster_id": "c1", "status": "ok", "axis": None, "branches": [
        {"id": "c", "kind": "central", "prob": 0.7, "region": {"lo_z": None, "hi_z": None}},
        {"id": "m", "kind": "mechanism", "prob": 0.3, "shifts": {"d_sent": {"mean": -3.0, "sd": 1.0},
                                                                   "d_unrate": {"mean": 0.1, "sd": 0.05}}}]}]}
    ans = G.guard_synth(raw, req)["c1"]
    m = next(b for b in ans.branches if b["kind"] == "mechanism")
    assert set(m["shifts"]) == {"d_unrate"} and any("模板通道" in r for r in ans.report)
    from scenariolab.llm.agent_prompts import synth_user
    text = synth_user(req)
    assert "模板通道：已由代码写入（模板与文献）" in text and "本季已由模板通道写入的全部字段" in text


def test_channels_run_in_the_agent_tree_only():
    model = T.Engine(small(tree_kind="model"), adapter=MA.stub(), root_accounts=None)
    assert not model.channels_on and model.layer is None and CH.QUEUE_KEY not in model.root().cloud.state
    # the base-rate tree draws history's actions at history's frequencies: the model's residuals contain their
    # average effects already, so channels there would count them twice (P20)
    base = T.Engine(small(tree_kind="base_rate"), adapter=MA.stub(), root_accounts=None)
    assert not base.channels_on and base.layer is None
    assert T.Engine(small(), adapter=MA.stub(), root_accounts=None).channels_on
    off = T.Engine(small(channels="off"), adapter=MA.stub(), root_accounts=None)
    assert not off.channels_on and CH.QUEUE_KEY not in off.root().cloud.state


def test_saved_settings_default_to_the_p14_behaviour_of_each_new_field():
    """Only the defaults: a folder's settings without the round's fields read as P14's behaviour (no channels, no
    overlay, the old big-hike gate). Continuing such a folder with this code is refused anyway (the engine version
    differs: test_resume_and_extend_refuse_another_engines_folder)."""
    new = ("channels", "puzzle_overlay", "puzzle_literature_arm", "big_hike_gate")
    old = {k: v for k, v in small().to_dict().items() if k not in new}
    assert T.TreeSettings.saved(old).channels == "off" and T.TreeSettings.saved(old).puzzle_overlay == "off"
    assert T.TreeSettings.saved(old).big_hike_gate == "p14" and T.TreeSettings.coerce(old).big_hike_gate == "fixed"
    assert T.TreeSettings.coerce(old).channels == "on"                        # a new request: channels on
    assert T.TreeSettings.saved(small().to_dict()).channels == "on"
    assert small(channels="off").hash() != small().hash()                     # the protocol is in the hash
    assert any("channels" in p for p in small(channels="maybe").problems())
    assert T.versions(MA.stub())["channels"] == CH.version()


def test_a_tree_with_channels_runs_end_to_end_and_its_children_carry_the_queue(tmp_path):
    st = small(depth=3, n_particles=150)
    run = T.TreeRun(st, T.Store(tmp_path / "t"), cache_dir=tmp_path / "c")
    res = run.run()
    assert res["leaves"]
    det = run.store.read_level(0, "detail")["details"]
    d0 = next(iter(det.values()))
    assert all("channels" in c and "template_levers" in c for c in d0["clusters"])
    kinds = {o["channel"]["kind"] for a in d0["next"] for o in a["options"]}
    assert "template" in kinds and "model" in kinds
    for lf in run.leaves:
        assert lf.cloud.state[CH.QUEUE_KEY].shape[1:] == (run.engine.layer.H, len(run.engine.layer.slots))
    # kept children carry the deferred channel attribution, and its parts still add up to the total
    recs = [r for r in run.store.read_level(2)["nodes"] if r.get("status") != "pruned"]
    atts = [((r.get("summary") or {}).get("attribution") or {}) for r in recs]
    assert recs and any(atts)
    for att in atts:
        for a, row in att.items():
            assert "channels" in row and "overlay" in row, (a, row)
            parts = sum(float(v) for k, v in row.items() if k not in ("total",) and isinstance(v, (int, float)))
            if "total" in row:
                assert parts == pytest.approx(float(row["total"]), abs=1e-3), row


def test_the_catalogues_gpr_write_is_set_last_when_channels_carry_geopolitics():
    """With channels on (a structural model) the Taiwan exercise's GPR write goes through structural= (do_last on a
    satellite: display only, as mapped); with channels off it stays the reduced-form observation P14 had."""
    for chan, where in (("on", "structural"), ("off", "given")):
        eng = T.Engine(small(n_particles=120, first_quarter="2027Q1", channels=chan), adapter=StackLikeStub(),
                       root_accounts=None)
        root = eng.root()
        dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
        _quiet(dec, keep=("china",))
        dom = next(d for d in A.ACTORS["china"].domains if any(o.id == "tw_large_exercise" for o in d.options))
        ab = dec["actors"]["china"]
        if dom.role == "attribute":
            ab["attributes"] = [{"domain": dom.id, "zh": dom.zh, "options": [{"id": "tw_large_exercise", "label": "演习",
                                                                             "p": 1.0}]}]
        else:
            _force(dec, "china", "tw_large_exercise")
        _, _, bk = _bk(eng, root, dec)
        ids = set(bk["structural"].ids) if where == "structural" else set(bk["given"].ids)
        other = set(bk["given"].ids) if where == "structural" else set(bk["structural"].ids)
        assert "d_gpr" in ids and "d_gpr" not in other, (chan, bk["given"].ids, bk["structural"].ids)


def test_the_big_hike_gate_is_the_folders_protocol():
    c = A.NodeContext(state={"s_core_yoy": 2.45, "s_infexp1": 4.6, "s_unrate": 4.2})
    assert not A._gate_big_hike(c)[0]                               # the fix: Michigan alone does not open it
    c.big_hike_gate = "p14"
    assert A._gate_big_hike(c)[0]                                   # a P14 folder keeps the gate it was built with
    eng = T.Engine(small(big_hike_gate="p14"), adapter=MA.stub(), root_accounts=None)
    assert eng.node_ctx(eng.root(), "2026Q4").big_hike_gate == "p14"


def test_base_rates_resolve_a_standoff_before_the_x_date_and_the_tariff_regime_ends():
    ids = ["debt_limit_raise", "debt_limit_suspend", "debt_limit_standoff", "no_new_law", "consolidation"]
    far = P.prior("congress", ids, {"quarter": "2027Q1", "gates": {"congress": {"p_xdate_next_q": 0.0}}})["p"]
    near = P.prior("congress", ids, {"quarter": "2027Q1", "gates": {"congress": {"p_xdate_next_q": 0.8}}})["p"]
    assert far["debt_limit_standoff"] > 0.4 and near["debt_limit_standoff"] < 0.05         # 0 defaults in 78
    tids = ["tariff_hold", "tariff_up_small", "tariff_up_large", "tariff_down", "deal"]
    pol = {"trade_regime": "active"}
    now = P.prior("president", tids, {"quarter": "2026Q4", "political": pol})["p"]
    later = P.prior("president", tids, {"quarter": "2031Q4", "political": pol})["p"]
    assert max(now, key=now.get) == "tariff_up_large" and later["tariff_hold"] > 0.8 and later["tariff_up_large"] < 0.05


@pytest.mark.skipif(not HAS_ROOT, reason="needs data/results/agents_accounts_root.json")
def test_the_base_rate_tree_does_not_default(tmp_path):
    """The review measured a sovereign default (the X-date passed) in 83-86% of the base-rate tree's particles within
    10 years; history has 0 in 78 limit actions. With the deadline-aware base rate and the books' hard deadline the
    tree's X-date share (mass-weighted over its leaves) is 0 and arrears stay under 5% after 6 years."""
    st = small(tree_kind="base_rate", depth=24, n_particles=100)
    run = T.TreeRun(st, T.Store(tmp_path / "t"), adapter=MA.stub(), root_accounts=AC.load_root(), cache_dir=tmp_path / "c")
    run.run()
    tot = sum(lf.p_path for lf in run.leaves)
    xd = sum(lf.p_path * lf.cloud.acc.fact_shares()["xdate_passed"] for lf in run.leaves) / tot
    ar = sum(lf.p_path * lf.cloud.acc.fact_shares()["arrears"] for lf in run.leaves) / tot
    assert xd < 0.02 and ar < 0.05, (xd, ar)


def test_resume_and_extend_refuse_a_folder_built_under_other_rules(tmp_path):
    st = S.default_settings(1, 2, n_particles=80, hard_cap=1)
    with pytest.raises(S.BuildPaused):
        S.build("vd", st, base=tmp_path / "s", cache_dir=tmp_path / "c", adapter=MA.stub())
    root = tmp_path / "s" / "vd"
    man = S.read_manifest(root)
    T.write_json(root / "manifest.json", {**man, "versions": {**man["versions"], "channels": "channels-1 (old) sha256:0"}})
    with pytest.raises(ValueError, match="模板通道表"):
        S.build("vd", st, resume=True, base=tmp_path / "s", cache_dir=tmp_path / "c", budget_confirmed=True)
    m2 = S.read_manifest(root)
    assert m2["status"] == "paused" and "模板通道表" in m2["error"] and "channels-1" not in m2["error"]
    T.write_json(root / "manifest.json", man)
    done = S.build("vd", st, resume=True, base=tmp_path / "s", cache_dir=tmp_path / "c", budget_confirmed=True)
    assert done["status"] == "done"
    from scenariolab.agents import jobs
    jm = jobs.JobManager(tmp_path / "agents", auto_resume=False)
    rid = "r-old"
    jm.run_dir(rid).mkdir(parents=True)
    vers = {**T.versions(MA.stub()), "engine": "agent-tree-4 (P14)"}
    T.write_json(jm.run_dir(rid) / "run.json", {"id": rid, "status": "paused", "settings": small().to_dict(),
                                                "versions": vers, "stub": True, "levels_done": 1})
    with pytest.raises(ValueError, match="推演引擎"):
        jm.resume(rid)
    with pytest.raises(ValueError, match="推演引擎"):
        jm.extend(rid, 2)


def test_the_engine_exposes_each_options_channel_for_the_actions_tab():
    eng = T.Engine(small(), adapter=MA.stub(), root_accounts=None)
    c = eng.option_channel("president", "tariff_up_large")
    assert c["kind"] == "template" and [t["id"] for t in c["templates"]] == ["G05", "G21"] and c["basis"]
    assert eng.option_channel("fed", "hike_25")["kind"] == "model"
    assert eng.option_channel("president", "ec_tighten")["kind"] == "display"
    assert eng.option_channel("china", "tw_blockade")["beyond_history"]
    assert eng.option_channel("congress", "no_new_law")["kind"] == "none"
    assert eng.option_channel("congress", "as_committed")["kind"] == "commitment"


def test_the_overlay_runs_only_while_the_probe_finds_the_puzzle(monkeypatch):
    eng = T.Engine(small(n_particles=100), adapter=InertStub(), root_accounts=None)
    monkeypatch.setitem(eng.transmission, "ff_irf", {"puzzle": False})
    eng._overlay = None
    assert not eng.overlay_on
    monkeypatch.setitem(eng.transmission, "ff_irf", {"puzzle": True})
    eng._overlay = None
    assert eng.overlay_on and eng.layer.kernel is not None                 # measured on this adapter, cached
    stub = T.Engine(small(n_particles=100), adapter=MA.stub(), root_accounts=None)
    assert not stub.overlay_on                                             # no structural capability: no probe
    off = T.Engine(small(n_particles=100, puzzle_overlay="off"), adapter=InertStub(), root_accounts=None)
    monkeypatch.setitem(off.transmission, "ff_irf", {"puzzle": True})
    off._overlay = None
    assert not off.overlay_on


def _overlay_chain(ad, ker, extra, H=16, n=1500, pulses=8):
    """+extra pp on each particle's own A0 policy-rate draw in the first `pulses` quarters (do), with the overlay's
    statements on the three inflation flows from the surprise history; returns core CPI y/y means per quarter."""
    L = CH.Layer({f: 1.0 for f in ad.x_fields}, ad.x_fields, facts_exit={}, kernel=ker)
    s = ad.root_state(n)
    s.update(L.empty_state(n))
    out = []
    for h in range(H):
        a0 = MA.as_draws(ad.complete(s, MA.Given.empty(n), np.random.default_rng(700 + h)))
        a0x = {k: np.asarray(v, float) for k, v in a0.x.items() if not str(k).startswith("_")}
        x = float(extra) if h < pulses else 0.0
        e = np.full(n, x)
        res = L.step(s, a0x, [], np.random.default_rng(1), surprise=e, overlay=ker is not None)
        pins = {"d_ff": a0x["d_ff"] + x} if x else {}
        d = MA.as_draws(ad.complete(s, MA.Given.from_dict(pins, n), np.random.default_rng(700 + h))) if pins else a0
        if res.overlay_inc:                       # the inflation pass (P16): increments on the quarter's values
            on = np.zeros(n, bool)
            for _, mk in res.overlay_inc.values():
                on |= mk
            for f in CH.INFLATION_FIELDS:
                inc = res.overlay_inc.get(f, (np.zeros(n), None))[0]
                pins[f] = np.where(on, np.asarray(d.x[f], float) + inc, np.nan)
            d = MA.as_draws(ad.complete(s, MA.Given.from_dict(pins, n), np.random.default_rng(700 + h)))
        s = ad.update_state(s, d)
        s.update(res.next)
        out.append(float(np.mean(s["s_core_yoy"])))
    return np.array(out)


def test_the_overlay_kernel_offsets_a_sustained_surprise_on_core_inflation():
    """The deconvolution machinery (channels.overlay_kernel and the layer's surprise history) on the stub: +25bp on
    the model's own draw for 8 quarters raises core CPI y/y (the stub has a price puzzle too); the overlay removes
    most of it. The stub reads a policy-rate pin as an association, so a kernel measured with the rate free cannot be
    exact while the rate is pinned; the stack's acceptance (±0.05 pp through 16 quarters) is the real-model test."""
    ad = MA.stub()
    ker = CH.overlay_kernel(ad, "stub-test", n=1500)
    assert ker is not None and set(ker) <= set(CH.OVERLAY_FIELDS)
    assert all(len(v) == CH.OVERLAY_K + 1 and v[0] == 0.0 for v in ker.values())
    free = _overlay_chain(ad, None, 0.0)
    hawk = _overlay_chain(ad, ker, 0.25)
    raw = _overlay_chain(ad, None, 0.25)
    assert np.max(raw[:8] - free[:8]) > 0.05                                 # the stub's own puzzle
    assert np.all((hawk - raw)[1:8] < 0)                                    # the overlay leans against it
    # on the stub a pinned rate also moves same-quarter inflation by association, which the increments keep (they
    # never erase a quarter's own inflation): only the lagged part is removed
    assert abs(hawk[3] - free[3]) < 0.8 * abs(raw[3] - free[3])


# ================================================================================================ the real model

@pytest.fixture(scope="module")
def qad():
    if not READY:
        pytest.skip(f"needs the quarterly panel: {WHY}")
    return MA.load_adapter(MA.QUARTERLY_SPEC)


@pytest.fixture(scope="module")
def qeng(qad):
    return T.Engine(small(n_particles=100), adapter=qad, root_accounts=None)


IMPACT_TOL = 0.25            # % of an impact reading; smaller judge readings are not sign-tested
LEVEL_TOL = 0.08             # % IP level / pp


def test_real_model_every_mapped_option_transmits_or_is_display_only_by_design(qad, qeng):
    """The engine's own channel layer at default intensity on the merged stack (CRN against a placebo, the agent
    tree's configuration): every entry with statements moves a response field or an asset by TRANSMISSION_TOL, except
    the display-only ones and the shutdown (core-neutral by design); where the judge's reading is material, the sign
    of the impact on US equity and Treasuries, the IP level at quarter 4 and headline CPI y/y at quarter 4 agree."""
    t = CH.table()
    keys = [k for k, e in t["options"].items() if e["status"] == "statements"] + ["event:xdate_passed", "event:pandemic"]
    for k in keys:
        r = T.channel_probe(qad, qeng, k, n=300)
        assert "error" not in r, (k, r)
        e = t["options"].get(k) or {}
        jr = (e.get("reading") if e else (t["events"][k.split(":")[1]].get("reading"))) or {}
        if e.get("display_only"):
            # displays transmit nothing; the Taiwan exercise keeps the catalogue's own GPR +1 sd observation, which
            # the stack reads weakly (crude about −0.45%, IP −0.03 sd): not a channel
            lim = 0.1 if k == "china.tw_large_exercise" else T.TRANSMISSION_TOL
            assert r["max_abs_3q_sd"] < lim, (k, r["max_abs_3q_sd"])
            continue
        if k == "congress.shutdown":
            # only its own GDP display moves; the IP hold at each particle's own draw re-shapes the stack's dollar
            # hold a little (a pin is not neutral there even at A0), so "neutral" is within 0.03 sd and 0.05%
            core = ("d_ip", "d_unrate", "d_cpi_goods", "d_cpi_serv", "d_gz", "d_y10", "d_usd", "d_oil", "d_spread")
            assert all(abs(r["impact_sd"][f]) < 0.03 for f in core), r["impact_sd"]
            assert all(abs(v) < 0.05 for v in r["impact_pct"].values()) and abs(r["ip_level_pct"]["q4"]) < 0.01
            assert r["impact_sd"]["d_rgdp"] < -0.1
            continue
        if k.startswith("treasury."):
            continue                                                          # dormant (no bill-share field)
        assert r["transmits"], (k, r["max_abs_3q_sd"])
        for a in ("us_equity", "treasury_10y"):
            jv = (jr.get("impact_pct") or {}).get(a)
            if jv is not None and abs(jv) >= IMPACT_TOL:
                assert np.sign(r["impact_pct"][a]) == np.sign(jv), (k, a, r["impact_pct"][a], jv)
        jv = (jr.get("ip_level_pct") or {}).get("q4")
        if jv is not None and abs(jv) >= LEVEL_TOL:
            assert np.sign(r["ip_level_pct"]["q4"]) == np.sign(jv), (k, "ip q4", r["ip_level_pct"]["q4"], jv)
        jv = (jr.get("h4_pp") or {}).get("cpi_yoy")
        if jv is not None and abs(jv) >= LEVEL_TOL:
            assert np.sign(r["h4"]["s_cpi_yoy"]) == np.sign(jv), (k, "cpi h4", r["h4"]["s_cpi_yoy"], jv)


def test_real_model_qe_at_the_root_is_bounded_and_not_rare(qad, qeng):
    r = T.channel_probe(qad, qeng, "fed.bs_qe", n=300)
    assert abs(r["impact_sd"]["d_y10"]) < 1.0 and r["impact_pct"]["us_equity"] > 0
    n = 300
    s = qad.root_state(n)
    a0 = MA.as_draws(qad.complete(s, MA.Given.empty(n), np.random.default_rng(3)))
    L = qeng.layer
    st = {**s, **L.empty_state(n)}
    w = {"field": "d_bs", "kind": "normal", "value": np.full(n, 3.0), "a": 3.0, "option": "bs_qe"}
    res = L.step(st, {k: np.asarray(v, float) for k, v in a0.x.items() if not str(k).startswith("_")},
                 [CH.Choice(actor="fed", option="bs_qe", rows=np.arange(n), writes=[w])], np.random.default_rng(1))
    pins = {f: np.asarray(a0.x[f], float) + d for f, (d, mk) in res.shift.items()}
    pins["d_bs"] = np.full(n, 3.0)
    det = qad.rarity_detail(s, MA.Given.from_dict(pins, n))
    assert float(np.median(det["percentile"])) <= 0.99
    gz_end = np.asarray(s["s_gz"], float) + np.asarray(res.shift["d_gz"][0])
    assert np.all(gz_end >= 0.65 - 1e-9)                                     # the room rule: never below p01


def test_real_model_two_passes_keep_the_credit_shocks_own_activity_effect(qad, qeng):
    """capex_slash's IP statements are increments on the IP its GZ statement already gives (P7): with the GZ news
    the IP level at quarter 4 is about −1.4%, against −0.7% for the IP statements alone (same particles, same seed;
    a one-pass pin at A0 + Δ gives the two the same −0.7%)."""
    key = "bigtech.capex_slash"
    T._CHANNEL_PROBE.clear()
    full = T.channel_probe(qad, qeng, key, n=300, seed=4242)
    orig = qeng.layer.options[key]
    try:
        alt = copy.deepcopy(orig)
        alt["statements"] = [s for s in alt["statements"] if s["field"] != "d_gz"]
        qeng.layer.options = {**qeng.layer.options, key: alt}       # a new table object: a new probe cache key
        alone = T.channel_probe(qad, qeng, key, n=300, seed=4242)
    finally:
        qeng.layer.options = {**qeng.layer.options, key: orig}
        T._CHANNEL_PROBE.clear()
    assert full["ip_level_pct"]["q4"] <= alone["ip_level_pct"]["q4"] - 0.3, (full["ip_level_pct"], alone["ip_level_pct"])


def test_real_model_the_running_overlay_keeps_other_actions_inflation(qad, qeng):
    """With the overlay running (a tiny surprise history, so its own increments are about 2e-4), an oil escalation and
    the court's removal ruling move inflation as with the overlay silent: the overlay is an increment on the quarter's
    inflation, never a pin at A0 (the review: escalate's CPI y/y h4 +0.36 silent against −0.03 running)."""
    for key in ("belligerents.escalate", "courts.allow_removal"):
        silent = T.channel_probe(qad, qeng, key, n=300, H=8, fed="own")
        running = T.channel_probe(qad, qeng, key, n=300, H=8, fed="own", overlay_history=0.001)
        for f in ("d_cpi_goods", "d_cpi_serv", "d_core"):
            assert running["impact_sd"][f] == pytest.approx(silent["impact_sd"][f], abs=0.01), (key, f)
        for k in ("s_cpi_yoy", "s_core_yoy"):
            for h in (3, 7):
                assert running["s_path"][k][h] == pytest.approx(silent["s_path"][k][h], abs=0.02), (key, k, h)
    esc = T.channel_probe(qad, qeng, "belligerents.escalate", n=300, H=8, fed="own", overlay_history=0.001)
    assert esc["impact_sd"]["d_core"] > 0.1 and esc["s_path"]["s_cpi_yoy"][3] > 0.2


def test_real_model_tariffs_reach_core_inflation(qad, qeng):
    """The core-goods pass-through (P16): a tariff raises core CPI y/y at quarter 4 by at least half of headline / 0.8
    (all of the goods move is in core goods), not the 0.05 per unit the d_core satellite reads off headline."""
    r = T.channel_probe(qad, qeng, "president.tariff_up_large", n=300, H=8, fed="own")
    head, core = r["s_path"]["s_cpi_yoy"][3], r["s_path"]["s_core_yoy"][3]
    assert head > 0.1 and core >= 0.5 * head / 0.8, (head, core)


def test_real_model_independence_raises_inflation_while_the_ruling_stands(qad, qeng):
    """Allowed removal (the ladder's top, no exit): with a Fed that never surprises, core CPI y/y is above the
    placebo at quarters 20 and 40 (the drift runs while the ruling stands; the review measured −0.29 at both with
    an 8-quarter cap and the overlay pinning core at A0)."""
    r = T.channel_probe(qad, qeng, "courts.allow_removal", n=300, H=40, fed="own")
    assert r["s_path"]["s_core_yoy"][19] >= 0 and r["s_path"]["s_core_yoy"][39] >= 0, r["s_path"]["s_core_yoy"]


def test_real_model_demand_shocks_do_not_leave_a_permanent_level_of_output(qad, qeng):
    """A large stimulus's IP level falls back: at quarter 20 under 30% of its peak (P17; the review: +0.62% at h8
    and h40 without the give-back)."""
    r = T.channel_probe(qad, qeng, "congress.stimulus_large", n=300, H=20)
    path = np.asarray(r["ip_level_path"])
    peak = float(np.max(np.abs(path)))
    assert peak > 1.0 and abs(path[19]) < 0.3 * peak, path


def test_real_model_overlay_kernel_and_its_acceptance_chain(qad, qeng):
    assert qeng.overlay_on, "the probe should still find the price puzzle on this stack"
    ker = qeng.layer.kernel
    ref = CH.table()["price_puzzle"]["overlay_spec"]["kernel_reference"]
    for f in CH.OVERLAY_FIELDS:
        for k in (1, 2, 4, 8):
            assert abs(float(ker[f][k]) - float(ref[str(k)][f])) < 0.02, (f, k, ker[f][k], ref[str(k)][f])
    free = _overlay_chain(qad, None, 0.0, n=1200)
    hawk = _overlay_chain(qad, ker, 0.25, n=1200)
    assert np.max(np.abs(hawk[:16] - free[:16])) < 0.05, hawk - free


@pytest.mark.skipif(not HAS_ROOT, reason="needs data/results/agents_accounts_root.json")
def test_real_model_branches_with_opposite_actions_differ_as_the_mapping_says(qad, tmp_path):
    """One node, the President's tariff decision forced both ways in two clusters with everything else at the
    status quo: the escalation's cluster has higher goods inflation and lower equities than the cut's (B_k)."""
    eng = T.Engine(small(n_particles=300, first_quarter="2027Q1"), adapter=qad, root_accounts=AC.load_root())
    root = eng.root()
    outs = {}
    for oid, mag in (("tariff_up_large", 2.0), ("tariff_down", -1.0)):
        dec = eng.decide(root, "2027Q1", eng.a0(root), {}, [], [])
        _quiet(dec, keep=("president",))
        _force(dec, "president", oid, mag)
        _, a0, bk = _bk(eng, root, dec)
        outs[oid] = {"goods": float(np.mean(bk["draws"].x["d_cpi_goods"])), "eq": float(np.mean(bk["draws"].y["us_equity"]))}
    assert outs["tariff_up_large"]["goods"] > outs["tariff_down"]["goods"]
    assert outs["tariff_up_large"]["eq"] < outs["tariff_down"]["eq"]
