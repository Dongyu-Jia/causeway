"""The actor registry (scenariolab/agents/actors.py) and the history priors (scenariolab/agents/priors.py):
role binding through the alias table, calendars and gates, payloads on schema fields, the oil-supply composite,
reactions, signals, the pinned facts (制度现状) and every prior table. No network and no LLM; the tests marked
`needs_local_data` re-estimate the priors from the project's local files and are skipped on a fresh clone."""
import json
import math

import numpy as np
import pytest

from scenariolab import config, jsonio
from scenariolab.agents import actors as A
from scenariolab.agents import priors as P

SCHEMA_V3 = [   # field ids of schema-revise-2 3.0 that the catalogue binds to (a subset is enough)
    "d_ff", "d_bs", "d_deficit", "d_transfers", "d_defense", "d_corptax", "d_perstax", "d_tariff", "d_metals_oil",
    "x_oil_cause", "d_itinv", "d_lf", "d_gpr", "d_capex", "d_unrate", "d_core", "s_ff", "y_mult"]
HIST = {"d_tariff": {"p01": -1.36, "p99": 1.49, "min": -3.21, "max": 5.44, "sd_q_1985_2019": 0.13},
        "d_gpr": {"p01": -44.2, "p99": 61.9, "min": -96.0, "max": 104.3, "sd_q_1985_2019": 22.15}}
ROOT_STATE = {"s_ff": 3.63, "s_core_yoy": 2.82, "s_unrate": 4.3, "s_rstar": 1.09, "us_equity_l1": 0.015,
              "d_spread_l1": -0.06, "d_unrate_l1": -0.1, "d_unrate_l2": 0.0, "d_ff_l1": 0.25}


def resolver(fields=SCHEMA_V3, **kw):
    return A.FieldResolver(fields=fields, **kw)


def ctx(**kw):
    base = {"quarter": "2026Q4", "state": dict(ROOT_STATE), "facts": {"oil_disruption_ongoing": True},
            "institutions": A.institution_values(A.institution_defaults()), "resolver": resolver()}
    base.update(kw)
    return base


# ---------------------------------------------------------------- the registry and the alias table

def test_the_catalogue_validates_with_zero_problems_on_schema_v3_and_without_a_schema():
    assert A.validate_catalogue(resolver()) == []
    assert A.validate_catalogue(A.FieldResolver()) == []
    cat = A.catalogue(resolver())
    jsonio.dumps(cat)
    assert [a["id"] for a in cat["actors"]] == A.STAGE_ORDER
    assert A.STAGE_ORDER[-1] == "fed" and A.ACTORS["fed"].stage == 4 and cat["problems"] == []


def test_the_final_actor_set_and_its_pool_weights():
    assert set(A.ACTORS) == {"voter", "courts", "belligerents", "president", "congress", "treasury", "china", "opec",
                             "bigtech", "fed"}
    c = {a: A.ACTORS[a].pool_c for a in A.ACTORS}
    assert c == {"voter": 0.2, "courts": 0.3, "belligerents": 0.4, "president": 0.7, "congress": 0.6, "treasury": 0.5,
                 "china": 0.6, "opec": 0.5, "bigtech": 0.6, "fed": 0.5}
    assert A.ACTORS["president"].custom_cap == A.ACTORS["china"].custom_cap == 0.15
    assert all(A.ACTORS[a].custom_cap == 0 for a in A.ACTORS if a not in ("president", "china"))
    assert "CEO、CFO" in A.ACTORS["bigtech"].naming and A.NATURE_POOL_C == 0.2
    for a in A.ACTORS.values():                          # exactly one main domain with one status-quo option
        mains = [d for d in a.domains if d.role == "main"]
        assert mains and all(sum(o.status_quo for o in d.options) == 1 for d in a.domains)


def test_aliases_resolve_old_ids_and_a_dropped_id_fails_with_a_chinese_reason():
    r = resolver(fields=["d_ff", "d_lf", "s_spread", "y_mult"])
    assert r.canonical("d_policy") == "d_ff" and r.canonical("d_lfpr") == "d_lf" and r.canonical("s_ebp") == "s_spread"
    with pytest.raises(KeyError, match="EBP"):
        r.canonical("d_ebp")
    v1 = resolver(fields=["d_policy", "d_lfpr"])           # an old schema: the role still binds through the alias
    assert v1.field("policy_rate_chg") is None or v1.field("policy_rate_chg") == "d_ff"
    with pytest.raises(KeyError):
        r.field("no_such_role")


def test_unresolved_levers_fall_back_as_the_catalogue_says():
    r = resolver()
    assert r.field("hyperscaler_capex_g") is None and r.field("it_invest_g") == "d_itinv"
    assert r.field("oil_supply_shock") is None and r.field("oil_supply_signature") == "d_metals_oil"
    assert r.field("bill_share_chg") is None
    probs = r.problems()
    assert probs == ["杠杆角色「bill_share_chg」在输入表里没有字段，也没有声明后备写法"]


def test_an_overlay_dict_is_accepted_as_the_resolver():
    ov = {"fields": {f: {"cls": "instrument" if f in ("d_ff", "d_bs", "d_tariff") else "quantity"} for f in SCHEMA_V3},
          "roles": {"policy_rate_chg": {"field": "d_ff", "candidates": ["d_ff", "d_policy"]}},
          "aliases": {"d_policy": "d_ff"}}
    r = A.FieldResolver.from_overlay(ov)
    assert r.field("policy_rate_chg") == "d_ff" and A.validate_catalogue(ov) == []
    bad = json.loads(json.dumps(ov))
    bad["fields"]["d_metals_oil"] = {"cls": "financial_price"}
    assert any("financial_price" in p for p in A.validate_catalogue(bad))


# ---------------------------------------------------------------- calendars and activity

def test_voters_act_only_in_federal_election_quarters_and_not_after_a_pinned_result():
    assert A.actor_activity("voter", ctx())[0]
    assert not A.actor_activity("voter", ctx(quarter="2027Q1"))[0]
    assert A.actor_activity("voter", ctx(quarter="2028Q4"))[0]
    m = A.menu("voter", ctx(quarter="2028Q4"))
    assert m["main"]["domain"] == "presidential"
    inst = {**A.institution_values(A.institution_defaults()), "election_2026_result": "house_flips"}
    on, why = A.actor_activity("voter", ctx(institutions=inst))
    assert not on and "锁定" in why


def test_courts_act_only_with_a_pending_ruling_in_the_quarter():
    assert not A.actor_activity("courts", ctx())[0]
    inst = {**A.institution_values(A.institution_defaults()), "ruling_tariff_quarter": "2026Q4"}
    on, why = A.actor_activity("courts", ctx(institutions=inst))
    assert on and "tariff_authority" in why
    m = A.menu("courts", ctx(institutions=inst))
    assert [o["id"] for o in m["main"]["options"]] == ["uphold", "strike_down", "strike_down_refunds"]
    assert not A.actor_activity("courts", ctx(institutions=inst, quarter="2027Q1"))[0]


def test_belligerents_need_the_oil_disruption_fact_and_treasury_needs_a_bill_share_field():
    assert A.actor_activity("belligerents", ctx())[0]
    assert not A.actor_activity("belligerents", ctx(facts={}))[0]
    assert not A.actor_activity("treasury", ctx())[0]
    r = resolver(fields=SCHEMA_V3 + ["x_bill_share_chg"])
    assert A.actor_activity("treasury", ctx(resolver=r))[0]
    assert A.root_state(A.institution_defaults())["facts"]["oil_disruption_ongoing"] is True


def test_the_root_has_nine_active_llm_actors_plus_nature():
    rows = A.active_actors(ctx())
    active = [r["id"] for r in rows if r["active"]]
    assert active == ["nature", "voter", "belligerents", "president", "congress", "china", "opec", "bigtech", "fed"]
    assert next(r for r in rows if r["id"] == "treasury")["why"] == "输入表无短债占比字段"


def test_quarter_helpers():
    assert A.quarter_add("2026Q4", 1) == "2027Q1" and A.quarter_add("2027Q1", -1) == "2026Q4"
    assert A.quarter_of("2026-11-10") == "2026Q4" and A.quarters_between("2026Q4", "2028Q4") == 8
    with pytest.raises(ValueError):
        A.quarter_add("2026-4", 1)


# ---------------------------------------------------------------- menus and gates

def test_fed_emergency_and_big_hike_options_open_only_when_their_gates_hold():
    m = A.menu("fed", ctx())
    ids = [o["id"] for o in m["main"]["options"]]
    assert "cut_75plus" not in ids and "hike_75plus" not in ids and "hold" in ids
    assert {d["id"] for d in m["disabled"]} >= {"cut_75plus", "hike_75plus", "bs_qe"}
    crash = ctx(state={**ROOT_STATE, "us_equity_l1": math.log(0.8)})
    assert "cut_75plus" in [o["id"] for o in A.menu("fed", crash)["main"]["options"]]
    accident = ctx(facts={"financial_accident": True})
    assert "cut_75plus" in [o["id"] for o in A.menu("fed", accident)["main"]["options"]]
    hot = ctx(state={**ROOT_STATE, "s_core_yoy": 4.4})
    assert "hike_75plus" in [o["id"] for o in A.menu("fed", hot)["main"]["options"]]
    elb = ctx(state={**ROOT_STATE, "s_ff": 0.1})
    bs = next(d for d in A.menu("fed", elb)["attributes"] if d["domain"] == "balance_sheet")
    assert "bs_qe" in [o["id"] for o in bs["options"]]


def test_the_debt_limit_gate_makes_its_options_required():
    m = A.menu("congress", ctx())
    assert not any(o["id"].startswith("debt_limit") for o in m["main"]["options"])
    gates = {"congress": {"require_options": ["debt_limit_raise", "debt_limit_suspend", "debt_limit_standoff"]}}
    m = A.menu("congress", ctx(gates=gates))
    ids = [o["id"] for o in m["main"]["options"]]
    assert {"debt_limit_raise", "debt_limit_suspend", "debt_limit_standoff"} <= set(ids)
    assert m["main"]["required"] == ["debt_limit_raise", "debt_limit_suspend", "debt_limit_standoff"]
    assert any("账户约束门" in n for n in m["notes"])
    assert next(d for d in m["attributes"] if d["domain"] == "deadline")      # 2026Q4 is a deadline quarter


def test_stimulus_options_need_a_recession_signal():
    ids = [o["id"] for o in A.menu("congress", ctx())["main"]["options"]]
    assert "stimulus_small" not in ids and "stimulus_large" not in ids
    soft = ctx(state={**ROOT_STATE, "s_sahm": 0.35})
    assert "stimulus_small" in [o["id"] for o in A.menu("congress", soft)["main"]["options"]]
    deep = ctx(state={**ROOT_STATE, "s_sahm": 0.6, "d_unrate_l1": 0.6, "d_unrate_l2": 0.5})
    assert "stimulus_large" in [o["id"] for o in A.menu("congress", deep)["main"]["options"]]
    assert "stimulus_large" in [o["id"] for o in A.menu("congress", ctx(facts={"pandemic": True}))["main"]["options"]]


def test_a_struck_down_tariff_authority_replaces_the_trade_menu():
    m = A.menu("president", ctx(facts={"tariff_authority": "struck_down", "oil_disruption_ongoing": True}))
    assert m["main"]["domain"] == "trade_substitute"
    assert [o["id"] for o in m["main"]["options"]] == ["replace_122", "replace_301_232", "accept_lapse"]
    assert m["notes"] and "替代授权" in m["notes"][0]
    assert A.menu("president", ctx())["main"]["domain"] == "trade"


def test_a_blockade_branches_only_in_stress_mode_and_spr_needs_a_disruption():
    tw = lambda c: next(d for d in A.menu("china", c)["attributes"] if d["domain"] == "taiwan")   # noqa: E731
    assert "tw_blockade" not in [o["id"] for o in tw(ctx())["options"]]
    assert "tw_blockade" in [o["id"] for o in tw(ctx(stress_mode=True))["options"]]
    energy = lambda c: next(d for d in A.menu("president", c)["attributes"] if d["domain"] == "energy")  # noqa: E731
    assert "spr_release" in [o["id"] for o in energy(ctx())["options"]]
    assert "spr_release" not in [o["id"] for o in energy(ctx(facts={}))["options"]]


def test_a_commitment_due_adds_the_as_committed_option():
    cm = [{"actor": "bigtech", "option": "capex_raise_large", "zh": "资本开支上调 15%", "due_in": 0}]
    m = A.menu("bigtech", ctx(commitments=cm))
    assert m["main"]["options"][0]["id"] == "as_committed" and "as_committed" in m["main"]["required"]


def test_trim_menu_by_prior_keeps_the_status_quo_and_covers_ninety_percent():
    prior = P.FED_RESEARCH["prior_2026Q4_macro"]
    ids = list(prior)
    kept = A.trim_menu_by_prior(ids, prior, keep=["hold"])
    assert "hold" in kept and len(kept) <= 4 and sum(prior[k] for k in kept) >= 0.85
    assert kept == ["cut_50", "cut_25", "hold", "hike_25"]


# ---------------------------------------------------------------- payloads on schema fields

def test_fed_payloads_are_points_rounded_to_25bp_inside_their_range():
    p = A.option_payload("fed", "cut_75plus", -1.3, resolver=resolver())
    assert p["writes"] == [{**p["writes"][0], "field": "d_ff", "kind": "point", "a": -1.25}]
    assert any("取整" in f for f in p["flags"])
    p = A.option_payload("fed", "hike_75plus", 3.0, resolver=resolver())
    assert p["writes"][0]["a"] == 1.5 and any("截断" in f for f in p["flags"])
    hold = A.option_payload("fed", "hold", resolver=resolver())
    assert hold["writes"][0]["a"] == 0.0 and hold["protocol"] == "lagged"


def test_tariff_magnitudes_are_checked_against_the_fields_history():
    big = A.option_payload("president", "tariff_up_large", 2.0, resolver=resolver(), support=HIST)
    assert big["writes"][0]["a"] == 2.0 and any("罕见" in f for f in big["flags"])
    assert not big["writes"][0]["beyond_history"]
    hold = A.option_payload("president", "tariff_hold", resolver=resolver())
    assert hold["writes"][0]["kind"] == "bin" and (hold["writes"][0]["lo"], hold["writes"][0]["hi"]) == (-0.25, 0.25)
    deal = A.option_payload("president", "deal", -2.0, resolver=resolver())
    assert [w["dq"] for w in deal["writes"]] == [0, 1] and deal["writes"][1]["a"] == -1.0
    assert deal["facts"][0]["id"] == "trade_truce"
    assert [c["due_quarter"] for c in A.commitments_of(deal, "2026Q4")] == ["2027Q1"]


def test_a_value_beyond_the_historical_extreme_is_clipped_and_kept_for_the_synthesizer():
    hist = {"d_tariff": {"p01": -1.36, "p99": 1.49, "min": -3.21, "max": 4.0}}
    p = A.option_payload("president", "tariff_up_large", 5.4, resolver=resolver(), support=hist)
    w = p["writes"][0]
    assert w["a"] == 4.0 and w["beyond_history"] and w["requested"] == 5.4


def test_congress_writes_fiscal_components_with_their_reversals():
    p = A.option_payload("congress", "stimulus_small", 0.8, resolver=resolver())
    assert [(w["field"], w["kind"], round(w["a"], 6), round(w["b"], 6), w["dq"]) for w in p["writes"]] == [
        ("d_transfers", "shift", 2.0, 1.0, 0), ("d_transfers", "shift", -2.0, 1.0, 1)]
    big = A.option_payload("congress", "stimulus_large", 4.0, resolver=resolver())
    assert {w["field"] for w in big["writes"]} == {"d_transfers", "d_deficit"} and max(w["dq"] for w in big["writes"]) == 5
    raise_ = A.option_payload("congress", "debt_limit_raise", 3000, resolver=resolver())
    assert raise_["writes"] == [] and raise_["account"] == {"debt_limit_raise_bn": 3000.0}
    assert "debt_limit_binding" in raise_["clears"]
    assert A.option_payload("congress", "shutdown", 35, resolver=resolver())["facts"][0]["value"] == 35.0


def test_the_hyperscaler_fallback_writes_it_investment_for_four_quarters():
    p = A.option_payload("bigtech", "capex_raise_large", 15, resolver=resolver())
    assert [(w["field"], w["dq"], round(w["a"], 6)) for w in p["writes"]] == [("d_itinv", k, 2.4) for k in range(4)]
    assert p["account"] == {"bt_capex_plan_pct": 15.0}
    assert A.account_actions([p]) == {"bt_capex_plan_pct": 15.0, "bt_new_guidance": True}
    with_field = resolver(fields=SCHEMA_V3 + ["hyperscaler_capex_g"])
    p2 = A.option_payload("bigtech", "capex_cut", -10, resolver=with_field)
    assert p2["writes"] == [{**p2["writes"][0], "field": "hyperscaler_capex_g", "kind": "point", "a": -10.0}]


def test_the_oil_supply_composite_sums_contributions_into_the_schemas_own_signature():
    pays = [A.option_payload("belligerents", "escalate", resolver=resolver()),
            A.option_payload("president", "spr_release", 1.0, resolver=resolver()),
            A.option_payload("opec", "quota_raise", 1.0, resolver=resolver())]
    ev = A.combine_events(pays, resolver=resolver(), support=HIST)
    oil = next(w for w in ev if w["field"] == "d_metals_oil")
    z = 1.5 - 0.32 - 0.32
    assert oil["kind"] == "shift" and abs(oil["a"] - (-6.8 * z)) < 1e-9 and oil["source"].startswith("event:")
    assert next(w for w in ev if w["field"] == "x_oil_cause")["a"] == 1.0
    gpr = next(w for w in ev if w["field"] == "d_gpr")
    assert abs(gpr["a"] - 0.5 * 22.15) < 1e-9
    calm = A.combine_events([A.option_payload("belligerents", "stalemate", resolver=resolver())], resolver=resolver())
    assert calm == []
    struct = resolver(fields=SCHEMA_V3 + ["x_oil_supply_shock"])
    ev2 = A.combine_events(pays, resolver=struct)
    assert next(w for w in ev2 if w["role"] == "oil_supply_shock")["kind"] == "normal"
    assert not any(w["field"] == "d_metals_oil" for w in ev2)
    nat = A.combine_events([A.hazard_payload("oil_disruption")], resolver=resolver())
    assert abs(next(w for w in nat if w["field"] == "d_metals_oil")["a"] + 6.8 * 4.0) < 1e-9


def test_a_blockade_beyond_the_gpr_history_is_flagged():
    ev = A.combine_events([A.option_payload("china", "tw_blockade", resolver=resolver())], resolver=resolver(),
                          support={"d_gpr": {"p01": -44.2, "p99": 61.9, "min": -96.0, "max": 60.0,
                                             "sd_q_1985_2019": 22.15}})
    assert ev[0]["beyond_history"] and ev[0]["a"] == 60.0 and abs(ev[0]["requested"] - 3 * 22.15) < 1e-9


def test_options_without_a_field_still_carry_facts_and_political_updates():
    p = A.option_payload("china", "retaliate_export_controls", resolver=resolver())
    assert p["writes"] == [] and {f["id"] for f in p["facts"]} == {"cn_retaliation", "export_controls_cn"}
    assert "trade_truce" in p["clears"]
    rem = A.option_payload("president", "attempt_removal", resolver=resolver())
    assert rem["political"] == {"independence_risk": 2}
    with pytest.raises(KeyError):
        A.option_payload("fed", "no_such_option")


# ---------------------------------------------------------------- reactions and lagged rules

def test_reaction_rows_sum_to_one_and_china_follows_the_president():
    for rr in A.REACTIONS:
        if rr["kind"] == "rows":
            assert all(abs(sum(r.values()) - 1) < 1e-12 for r in rr["rows"].values())
    rows = A.reaction_rows("china", {"president": "tariff_up_large"})
    assert rows[0]["row"]["retaliate_tariffs"] == 0.50 and rows[0]["llm_may_adjust"]
    assert A.reaction_rows("china", {"president": "unknown_option"}) == []
    assert A.reaction_rows("opec", {"belligerents": "escalate"})[0]["row"]["quota_raise"] == 0.45
    menu_rule = A.reaction_rows("president", {"courts": "strike_down"})
    assert menu_rule[0]["kind"] == "menu" and menu_rule[0]["row"] == "trade_substitute"
    assert len(A.reaction_pairs()) == len(A.REACTIONS)


def test_lagged_rules_fire_from_last_quarters_choices_and_markets():
    rules = A.lagged_rules({"choices": {"voter": "house_flips", "courts": "structural"}, "us_equity_q": math.log(0.85)})
    names = {r["rule"] for r in rules}
    assert {"new_congress", "antitrust_payout", "market_retreat"} <= names
    assert A.lagged_rules({}) == []
    assert any(r["rule"] == "pandemic_fiscal" for r in A.lagged_rules({"facts": {"pandemic": True}}))


# ---------------------------------------------------------------- public signals

def test_default_signals_and_signal_validation():
    s = A.default_signal("fed", "hike_25", resolver=resolver())
    assert s == {"actor": "fed", "option": "hike_25", "text": "进一步收紧可能合适", "instrument": "d_ff", "direction": 1,
                 "horizon_q": 2, "condition": None, "commit": False}
    clean, flags = A.validate_signal({"text": "x" * 60, "direction": 3, "horizon_q": 9, "condition": "ebp_up_0.5"})
    assert len(clean["text"]) == 40 and clean["direction"] == 1 and clean["horizon_q"] == 4 and clean["condition"] is None
    assert len(flags) == 2
    assert "ebp_up_0.5" not in A.SIGNAL_CONDITIONS and "spread_up_0.5" in A.SIGNAL_CONDITIONS


# ---------------------------------------------------------------- pinned facts (制度现状)

def test_every_shipped_pinned_fact_is_unconfirmed_with_a_reason():
    items = A.institution_defaults()
    ids = {d["id"] for d in items}
    assert {"fomc_2026_10", "fed_chair", "ieepa_status", "refunds_remaining_bn", "china_truce_expiry", "opec_spare_mbd",
            "belligerents_active", "house_flip_odds", "cr_expiry", "em_capacity_bn", "tga_target_bn",
            "hyperscaler_guidance", "chip_controls", "antitrust_pending", "ruling_tariff_quarter", "approval",
            "china_growth_target"} <= ids
    assert all(d["status"] == "unconfirmed" and d["default_reason"] for d in items)
    fomc = next(d for d in items if d["id"] == "fomc_2026_10")
    assert fomc["value"] == "hold" and "3.90%" in fomc["default_reason"]
    assert next(d for d in items if d["id"] == "opec_spare_mbd")["visibility"] == "opec"
    v = A.institution_values(items)
    assert v["em_capacity_bn"] == 350.0 and v["tga_target_bn"] == 850.0


def test_pinned_fact_edits_are_typed_validated_and_recorded(tmp_path):
    path = tmp_path / "institutions.json"
    assert A.ensure_institutions_file(path) == path and path.exists()
    before = path.read_bytes()
    A.ensure_institutions_file(path)
    assert path.read_bytes() == before                     # an existing file is never touched
    items = A.save_institutions({"house_flip_odds": 0.62, "fomc_2026_10": "cut_25", "cr_expiry": "2027-01-30"}, path)
    got = {d["id"]: d for d in items}
    assert got["house_flip_odds"]["value"] == 0.62 and got["house_flip_odds"]["status"] == "confirmed"
    assert got["fomc_2026_10"]["edited_at"] and A.load_institutions(path)[0]["status"] in ("confirmed", "unconfirmed")
    for bad, msg in (({"house_flip_odds": 1.5}, "概率"), ({"fomc_2026_10": "cut_100"}, "只能是"),
                     ({"cr_expiry": "next week"}, "日期"), ({"no_such_item": 1}, "没有名为"),
                     ({"belligerents_active": "yes"}, "true 或 false"), ({"em_capacity_bn": "lots"}, "必须是数")):
        with pytest.raises(ValueError, match=msg):
            A.save_institutions(bad, path)
    assert {d["id"]: d["value"] for d in A.load_institutions(path)}["house_flip_odds"] == 0.62


def test_root_state_reads_the_pinned_facts():
    st = A.root_state(A.institution_defaults())
    assert st["facts"]["trade_truce_expiry_q"] == "2026Q4" and st["political"]["tariff_authority"] == "broad"
    items = A.institution_defaults()
    for d in items:
        if d["id"] == "belligerents_active":
            d["value"] = False
        if d["id"] == "ieepa_status":
            d["value"] = "struck_down"
    st = A.root_state(items)
    assert "oil_disruption_ongoing" not in st["facts"] and st["political"]["tariff_authority"] == "struck_down"


# ---------------------------------------------------------------- the Fed's ordered probit

def test_the_frozen_probit_reproduces_the_research_prior_for_2026q4():
    p = P.FED_PROBIT_MACRO.prior(P.FED_RESEARCH["today_x_2026Q4"])
    assert all(abs(p[k] - v) < 0.0015 for k, v in P.FED_RESEARCH["prior_2026Q4_macro"].items())
    assert abs(sum(p.values()) - 1) < 1e-12 and P.FED_PROBIT_MACRO.counts == (12, 15, 10, 40, 17, 15, 8)
    s = P.FED_PROBIT_SCHEMA.prior({k: v for k, v in P.FED_RESEARCH["today_x_2026Q4"].items() if k != "gap"})
    assert abs(s["hold"] - 0.471) < 0.002 and max(s, key=s.get) == "hold"


def test_a_pinned_october_hold_renormalises_to_the_december_meeting():
    p = P.fed_pinned_renormalize(P.FED_RESEARCH["prior_2026Q4_macro"], "hold")
    assert set(p) == {"cut_25", "hold", "hike_25"}
    assert abs(p["cut_25"] - 0.122) < 0.002 and abs(p["hold"] - 0.634) < 0.002 and abs(p["hike_25"] - 0.244) < 0.002
    cut = P.fed_pinned_renormalize(P.FED_RESEARCH["prior_2026Q4_macro"], "cut_25")
    assert set(cut) == {"cut_50", "cut_25", "hold"}


def test_fed_prior_uses_the_state_covariates_and_falls_back_to_the_frozen_prior():
    full = P.fed_prior(ROOT_STATE)
    assert full["source"] == "fed_probit_schema" and abs(sum(full["p"].values()) - 1) < 1e-12
    cov = P.fed_covariates(ROOT_STATE)
    assert abs(cov["rgap"] - (3.63 - 2.82 - 1.09)) < 1e-12 and cov["du2"] == -0.1 and cov["lag"] == 0.25
    frozen = P.fed_prior({})
    assert frozen["p"] == P.FED_RESEARCH["prior_2026Q4_macro"] and "冻结" in frozen["label"]
    pinned = P.fed_prior({}, quarter="2026Q4", pinned_first_meeting="hold")
    assert set(pinned["p"]) == {"cut_25", "hold", "hike_25"}
    hot = P.fed_prior({**ROOT_STATE, "d_ff_l1": 0.5, "s_core_yoy": 5.0})
    assert sum(hot["p"][k] for k in ("hike_25", "hike_50", "hike_75plus")) > sum(
        full["p"][k] for k in ("hike_25", "hike_50", "hike_75plus"))


def test_the_ordered_probit_fit_recovers_known_parameters():
    from scipy import stats
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 2))
    lat = X @ np.array([0.8, -0.5]) + rng.normal(size=n)
    cuts = np.array([-1.0, 0.0, 1.2])
    y = np.searchsorted(cuts, lat)
    fit = P.fit_ordered_probit(X, y, 4, ["a", "b"])
    assert abs(fit.coef[0] - 0.8) < 0.08 and abs(fit.coef[1] + 0.5) < 0.08
    assert np.allclose(fit.cuts, cuts, atol=0.1) and fit.n == n
    p = fit.probs({"a": 0.0, "b": 0.0})
    assert abs(p[0] - stats.norm.cdf(-1.0)) < 0.05
    rt = P.OrderedProbit.from_dict(json.loads(json.dumps(fit.to_dict())))
    assert rt == fit


# ---------------------------------------------------------------- the prior of every menu

def _menu_ids(actor, c=None, domain=None):
    m = A.menu(actor, c or ctx())
    d = m["main"] if domain is None else next(x for x in m["attributes"] if x["domain"] == domain)
    return [o["id"] for o in d["options"]]


@pytest.mark.parametrize("actor", ["voter", "belligerents", "president", "congress", "china", "opec", "bigtech", "fed"])
def test_every_main_menu_gets_a_normalised_prior_with_a_source(actor):
    ids = _menu_ids(actor)
    pr = P.prior(actor, ids, ctx(), tables=P.frozen_priors())
    assert set(pr["p"]) == set(ids) and abs(sum(pr["p"].values()) - 1) < 1e-9
    assert all(v >= 0 for v in pr["p"].values()) and pr["source"] and pr["label"]
    assert pr["kind"] in ("history", "judgment", "market", "uniform")


@pytest.mark.parametrize("actor,domain", [("fed", "balance_sheet"), ("president", "energy"), ("congress", "deadline"),
                                          ("china", "taiwan"), ("china", "fx"), ("bigtech", "payout")])
def test_attribute_priors(actor, domain):
    ids = _menu_ids(actor, domain=domain)
    pr = P.prior(actor, ids, ctx(), domain=domain, tables=P.frozen_priors())
    assert abs(sum(pr["p"].values()) - 1) < 1e-9 and pr["kind"] == "judgment"


def test_market_odds_make_the_voter_prior_only_and_base_rates_are_labelled_otherwise():
    inst = {**A.institution_values(A.institution_defaults()), "house_flip_odds": 0.7}
    pr = P.prior("voter", ["house_holds", "house_flips"], ctx(institutions=inst), tables=P.frozen_priors())
    assert pr["kind"] == "market" and abs(pr["p"]["house_flips"] - 0.7) < 1e-12
    base = P.prior("voter", ["house_holds", "house_flips"], ctx(), tables=P.frozen_priors())
    assert abs(base["p"]["house_flips"] - 7 / 12) < 1e-3 and "待核" in base["label"]


def test_debt_limit_and_crash_and_bust_rows():
    t = P.frozen_priors()
    ids = ["no_new_law", "debt_limit_raise", "debt_limit_suspend", "debt_limit_standoff"]
    pr = P.prior("congress", ids, ctx(), tables=t)
    hz = 1 - 0.5 ** (91.25 / 135.5)
    assert abs(pr["p"]["debt_limit_standoff"] / (pr["p"]["debt_limit_raise"] + pr["p"]["debt_limit_suspend"])
               - (1 - hz) / hz) < 0.05
    crash = P.prior("opec", ["quota_hold", "quota_cut", "quota_raise"], ctx(state={"real_oil_d2": -0.4}, facts={}), tables=t)
    assert abs(crash["p"]["quota_cut"] - 0.625) < 1e-9 and crash["source"] == "opec_record:crash"
    lead = P.prior("opec", ["quota_hold", "quota_cut", "quota_raise"], ctx(leader_choices={"belligerents": "escalate"}), tables=t)
    assert abs(lead["p"]["quota_raise"] - 0.45) < 1e-9
    bust = P.prior("bigtech", ["capex_raise_large", "capex_hold", "capex_cut", "capex_slash"],
                   ctx(state={"nasdaq_4q": math.log(0.75)}), tables=t)
    assert bust["source"] == "bigtech:bust" and abs(bust["p"]["capex_cut"] - 0.40) < 1e-9
    g7 = P.prior("bigtech", ["capex_raise_large", "capex_hold", "capex_cut", "capex_slash"],
                 ctx(gates={"bigtech": {"tilt": {"capex_raise_large": 0.5}}}), tables=t)
    assert g7["p"]["capex_raise_large"] < P.prior("bigtech", ["capex_raise_large", "capex_hold", "capex_cut", "capex_slash"],
                                                 ctx(), tables=t)["p"]["capex_raise_large"]
    ec = P.prior("bigtech", ["capex_raise_large", "capex_hold", "capex_cut", "capex_slash"],
                 ctx(leader_choices={"president": "ec_tighten"}), tables=t)
    assert any("×1.5" in n for n in ec["notes"])


def test_china_follows_the_presidents_choice_and_the_truce_expiry():
    ids = ["no_response", "retaliate_tariffs", "retaliate_export_controls", "negotiate_truce"]
    t = P.frozen_priors()
    lead = P.prior("china", ids, ctx(leader_choices={"president": "tariff_up_large"}), tables=t)
    assert abs(lead["p"]["retaliate_tariffs"] - 0.5) < 1e-9 and lead["source"] == "reaction:president.tariff_up_large"
    exp = P.prior("china", ids, ctx(facts={"trade_truce_expiry_q": "2026Q4"}), tables=t)
    assert exp["p"]["negotiate_truce"] > P.prior("china", ids, ctx(quarter="2027Q2"), tables=t)["p"]["negotiate_truce"]


def test_courts_without_odds_are_uniform_and_labelled():
    pr = P.prior("courts", ["uphold", "strike_down", "strike_down_refunds"], ctx(), domain="tariff_authority",
                 tables=P.frozen_priors())
    assert pr["kind"] == "uniform" and "无可靠基率" in pr["label"]
    inst = {**A.institution_values(A.institution_defaults()), "ruling_tariff_odds": 0.6}
    pr = P.prior("courts", ["uphold", "strike_down", "strike_down_refunds"], ctx(institutions=inst),
                 domain="tariff_authority", tables=P.frozen_priors())
    assert pr["kind"] == "market" and abs(pr["p"]["uphold"] - 0.4) < 1e-12


def test_the_frozen_tables_carry_the_research_counts():
    t = P.frozen_priors()
    assert t["fiscal"]["stimulus_given_sahm"]["k"] == 4 and t["fiscal"]["stimulus_given_sahm"]["n"] == 9
    assert t["fiscal"]["shutdown"]["k"] == 7 and t["fiscal"]["shutdown"]["n"] == 36
    assert t["tariff"]["normal_big_move"]["k"] == 11 and t["opec"]["cut_given_crash"]["k"] == 5
    assert t["belligerents"]["exit"]["k"] == 13 and t["china"]["retaliation"]["k"] == 8
    assert t["nature"]["oil_disruption"]["k"] == 9 and t["nature"]["financial_accident"]["k"] == 5
    assert t["voter"]["since_1994"] == {**t["voter"]["since_1994"], "k": 5, "n": 6}
    assert t["china"]["taiwan_blockade"]["k"] == 0 and t["china"]["taiwan_blockade"]["jeffreys_mean"] < 0.002
    assert P.jeffreys(13, 31) == [0.259, 0.594] and P.jeffreys(0, 10)[0] == 0.0
    jsonio.dumps(t)
    hz = P.hazard_table(t)
    assert [h["id"] for h in hz] == ["oil_disruption", "gpr_episode", "financial_accident", "pandemic"]
    assert P.priors_summary(t)["fed_today"]["hold"] == 0.479


def test_load_priors_prefers_the_built_file(tmp_path):
    p = tmp_path / "agents_priors.json"
    assert P.load_priors(p)["built"] is False
    jsonio.dump({**P.frozen_priors(), "built": True}, p)
    assert P.load_priors(p)["built"] is True
    jsonio.dump({"version": "old"}, p)
    assert P.load_priors(p)["built"] is False


# ---------------------------------------------------------------- reproduction from local data (skipped on a fresh clone)

needs_local_data = pytest.mark.skipif(
    not (config.PANEL_PATH.exists() and (config.RAW_DIR / "fred" / "FEDFUNDS.csv").exists()
         and (config.RAW_DIR / "nyfed" / "Holston_Laubach_Williams_current_estimates.xlsx").exists()),
    reason="local data not built")


@needs_local_data
def test_every_prior_is_reproduced_from_local_data():
    t = P.build_priors()
    if t["fed"].get("next_decision", {}).get("quarter") != "2026Q4":
        pytest.skip("the local data run past 2026Q3: the research numbers are for the 2026Q4 decision")
    assert t["fed"]["reproduced"], t["fed"]["why"]
    assert all(v for v in t["summary"].values()), t["summary"]
    assert abs(t["fed"]["prior_next_macro"]["hold"] - 0.479) < 0.0015


# ---------------------------------------------------------------- instrument bins and custom options

def test_instrument_bins_tile_the_line_for_the_prior_only_trees():
    fed = A.instrument_bins("fed")
    assert [b[0] for b in fed] == list(P.FED_OPTIONS)
    assert fed[0][1] == -math.inf and fed[-1][2] == math.inf
    assert all(a[2] == b[1] for a, b in zip(fed, fed[1:]))               # contiguous, no gaps
    assert [lo for _, lo, _ in fed[1:]] == list(P.FED_EDGES[1:-1])
    tr = A.instrument_bins("president")
    assert [b[0] for b in tr] == ["tariff_down", "tariff_hold", "tariff_up_small", "tariff_up_large"]
    assert all(a[2] == b[1] for a, b in zip(tr, tr[1:]))


def test_custom_options_may_write_only_own_levers_and_own_facts_under_the_cap():
    rules = A.custom_option_rules("president", resolver())
    assert rules["allowed"] and rules["cap"] == 0.15 and rules["writable_fields"] == ["d_lf", "d_tariff"]
    clean, flags = A.validate_custom_option(
        "president", {"writes": {"d_tariff": 1.0, "d_ff": -0.5, "d_lf": "x"}, "prob": 0.3,
                      "facts": ["trade_truce", "pandemic"]}, resolver())
    assert clean["writes"] == {"d_tariff": 1.0} and clean["prob"] == 0.15 and clean["facts"] == ["trade_truce"]
    assert len(flags) == 4
    china = A.custom_option_rules("china", resolver())
    assert china["writable_fields"] == [] and "taiwan_exercise" in china["facts"]
    assert A.validate_custom_option("fed", {"writes": {}}, resolver()) == (None, ["美联储没有自定义选项"])


def test_the_pinned_october_meeting_trims_the_2026q4_fed_menu():
    m = A.menu("fed", ctx())
    assert [o["id"] for o in m["main"]["options"]] == ["cut_25", "hold", "hike_25"]
    assert {"cut_50", "hike_50"} <= {d["id"] for d in m["disabled"]} and any("锁定" in n for n in m["notes"])
    inst = {**A.institution_values(A.institution_defaults()), "fomc_2026_10": "cut_25"}
    assert [o["id"] for o in A.menu("fed", ctx(institutions=inst))["main"]["options"]] == ["cut_50", "cut_25", "hold"]
    assert [o["id"] for o in A.menu("fed", ctx(quarter="2027Q1"))["main"]["options"]] == [
        "cut_50", "cut_25", "hold", "hike_25", "hike_50"]
    pr = P.prior("fed", ["cut_25", "hold", "hike_25"], ctx(state={}), tables=P.frozen_priors())
    assert abs(pr["p"]["hold"] - 0.634) < 0.002 and abs(pr["p"]["hike_25"] - 0.244) < 0.002


def test_the_ample_reserves_gate_tilts_the_balance_sheet_prior():
    ids = ["bs_hold", "bs_runoff", "bs_reserve_purchases"]
    base = P.prior("fed", ids, ctx(), domain="balance_sheet", tables=P.frozen_priors())
    low = P.prior("fed", ids, ctx(gates={"fed": {"tilt": {"bs_runoff": 0.5}}}), domain="balance_sheet",
                  tables=P.frozen_priors())
    assert low["p"]["bs_runoff"] < base["p"]["bs_runoff"] and any("G-L5" in n for n in low["notes"])


def test_a_context_free_call_means_the_root_node(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "INSTITUTIONS_PATH", tmp_path / "missing.json")      # the shipped defaults
    c = A.NodeContext.of(None)
    assert c.quarter == "2026Q4" and c.facts["oil_disruption_ongoing"] and c.political["tariff_authority"] == "broad"
    assert A.menu("belligerents")["active"] and A.menu("fed")["main"]["options"][1]["id"] == "hold"
