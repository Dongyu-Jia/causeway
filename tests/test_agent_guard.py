"""Guardrails on parsed agent answers (scenariolab/llm/agent_guard.py): the PROMPTS T0 fixtures on the schema-v2
option shape, the structural part of the synthesizer's S0 suite, and the root translation. No model, no data."""
import copy

import pytest

from agent_llm_cases import STATE, china, fed, hazards, president, synth_request, tariff_pair
from scenariolab.llm import agent_guard as G

ALLOWED = G.allowed_numbers(STATE)


def block(*options, **kw):
    return {"status": "ok", "options": [dict(o) for o in options], "vs_base_rate": kw.pop("vs", "依据 s_core_yoy"), **kw}


def opt(oid, p, **kw):
    return {"id": oid, "prob": p, "rationale": kw.pop("rationale", "依据 s_ff 与 s_core_yoy"), **kw}


VALID = block(opt("hold", 0.55), opt("hike_25", 0.30), opt("cut_25", 0.15))


def probs(ans):
    return {o["id"]: o["p_llm"] for o in ans.options}


def test_a_valid_answer_passes_untouched():
    ans = G.guard_actor(VALID, fed(), allowed=ALLOWED)
    assert ans.problems == [] and ans.report == [] and not ans.fallback
    assert probs(ans) == {"hold": 0.55, "hike_25": 0.30, "cut_25": 0.15}
    assert ans.options[0]["label"] == "按兵不动" and ans.options[0]["source"] == "llm"
    assert sum(probs(ans).values()) == pytest.approx(1.0, abs=1e-9)


def test_percentages_and_small_sum_errors_are_fixed_and_large_ones_need_a_repair():
    ans = G.guard_actor(block(opt("hold", 55), opt("hike_25", 30), opt("cut_25", 15)), fed(), allowed=ALLOWED)
    assert probs(ans) == {"hold": 0.55, "hike_25": 0.30, "cut_25": 0.15} and any("百分数" in r for r in ans.report)
    ans = G.guard_actor(block(opt("hold", 0.5), opt("hike_25", 0.3), opt("cut_25", 0.15)), fed(), allowed=ALLOWED)
    assert sum(probs(ans).values()) == pytest.approx(1.0) and any("已归一化" in r for r in ans.report)
    ans = G.guard_actor(block(opt("hold", 0.10), opt("hike_25", 0.30), opt("cut_25", 0.22)), fed(), allowed=ALLOWED)
    assert ans.problems == ["fed：概率之和 0.62 偏离 1 太多"] and ans.options == []
    ans = G.guard_actor(block(opt("hold", None), opt("hike_25", 0.5)), fed(), allowed=ALLOWED)
    assert ans.problems and "缺少概率" in ans.problems[0]


def test_powers_unknown_options_and_writes_on_catalogue_options_are_dropped():
    raw = block(opt("hold", 0.6, writes={"d_ff": -0.5}), opt("qe_infinity", 0.2), opt("hike_25", 0.4))
    ans = G.guard_actor(raw, fed(), allowed=ALLOWED)
    assert set(probs(ans)) == {"hold", "hike_25"} and all(o["writes"] == {} for o in ans.options)
    assert any("不是本节点菜单中的选项" in r for r in ans.report) and any("已删除 writes" in r for r in ans.report)
    dup = G.guard_actor(block(opt("hold", 0.4), opt("hold", 0.2), opt("hike_25", 0.4)), fed(), allowed=ALLOWED)
    assert probs(dup) == {"hold": 0.6, "hike_25": 0.4} and any("重复" in r for r in dup.report)


def test_magnitudes_are_clipped_rounded_and_unit_slips_fixed():
    def mag(v):
        ans = G.guard_actor(block(opt("cut_75plus", 0.2, magnitude=v), opt("hold", 0.8)), fed(), allowed=ALLOWED)
        return ans.options[0]["magnitude"], ans.report
    assert mag(None) == (-1.0, [])                                        # the catalogue default
    assert mag(-1.25) == (-1.25, [])
    v, rep = mag(-100)                                                    # basis points in a pp field
    assert v == -1.0 and "看起来是基点" in rep[0]
    v, rep = mag(-3.0)
    assert v == -1.75 and "超出范围" in rep[0]
    v, rep = mag(-0.8)
    assert v == -0.75 and "取整" in rep[0]
    ans = G.guard_actor(block(opt("hold", 0.8, magnitude=0.25), opt("hike_25", 0.2)), fed(), allowed=ALLOWED)
    assert ans.options[0]["magnitude"] is None and "没有幅度参数" in ans.report[0]


def test_the_three_percent_rule_merges_into_the_nearest_option_and_conserves_mass():
    ans = G.guard_actor(block(opt("hold", 0.50), opt("cut_25", 0.30), opt("cut_50", 0.02), opt("hike_25", 0.18)),
                        fed(), allowed=ALLOWED)
    assert probs(ans) == {"hold": 0.50, "cut_25": 0.32, "hike_25": 0.18}     # cut_50 joins cut_25, not hold
    assert any("并入最接近的选项 cut_25" in r for r in ans.report)
    two = G.guard_actor(block(opt("hold", 0.99), opt("hike_25", 0.01)), fed(), allowed=ALLOWED)
    assert probs(two) == {"hold": 0.97, "hike_25": 0.03}
    five = G.guard_actor(block(opt("hold", 0.4), opt("hike_25", 0.2), opt("hike_50", 0.1), opt("cut_25", 0.2),
                               opt("cut_50", 0.1)), fed(), allowed=ALLOWED)
    assert len(five.options) == 4 and sum(probs(five).values()) == pytest.approx(1.0)


def test_status_quo_and_option_count_are_ensured():
    ans = G.guard_actor(block(opt("hike_25", 0.7), opt("hike_50", 0.3)), fed(), allowed=ALLOWED)
    assert probs(ans)["hold"] == 0.03 and sum(probs(ans).values()) == pytest.approx(1.0)
    assert any("缺少维持现状选项" in r for r in ans.report)
    one = G.guard_actor(block(opt("hold", 1.0)), fed(), allowed=ALLOWED)
    assert probs(one) == {"hold": 0.97, "hike_25": 0.03}                  # the top base-rate option joins
    assert G.guard_actor(block(), fed(), allowed=ALLOWED).problems == ["fed：没有有效的选项"]


def test_custom_options_write_only_own_levers_inside_history_and_under_the_cap():
    raw = block(opt("tariff_hold", 0.6), opt("custom", 0.4, label="对华加征", writes={"d_tariff": 12.0, "d_ff": -1.0},
                                             facts=["export_controls_level", "invented_fact"],
                                             beyond_history="2025 年式的大幅加征"))
    ans = G.guard_actor(raw, president(), allowed=ALLOWED)
    custom = next(o for o in ans.options if o["custom"])
    assert custom["writes"] == {"d_tariff": 5.4383} and custom["facts"] == ["export_controls_level"]
    assert custom["p_llm"] == pytest.approx(0.15) and probs(ans)["tariff_hold"] == pytest.approx(0.85)
    report = " ".join(ans.report)
    assert "越权字段 d_ff" in report and "超出历史极值" in report and "理由：2025 年式的大幅加征" in report
    assert "invented_fact 不允许" in report and "截为上限" in report
    rare = G.guard_actor(block(opt("tariff_hold", 0.9), opt("custom", 0.1, writes={"d_tariff": 2.0})), president(),
                         allowed=ALLOWED)
    assert any("罕见" in r for r in rare.report)
    empty = G.guard_actor(block(opt("custom", 0.1, writes={"d_ff": 1}), opt("tariff_up_small", 0.9)), president(),
                          allowed=ALLOWED)
    assert probs(empty) == {"tariff_hold": 0.1, "tariff_up_small": 0.9}      # an empty custom becomes the status quo
    fed_custom = G.guard_actor(block(opt("custom", 0.1, writes={"d_ff": -0.5}), opt("hold", 0.9)), fed(),
                               allowed=ALLOWED)
    assert "custom" not in probs(fed_custom)                                   # the Fed has no custom option


def test_price_sentences_are_deleted_and_restatements_kept():
    texts = {
        "加息后美股预计将下跌 12%，纳指跌至 18000 点。核心通胀 2.57% 仍高。": "核心通胀 2.57% 仍高。",
        "10 年期收益率 s_y10 5.29% 已上行；失业率 4.2%。": "10 年期收益率 s_y10 5.29% 已上行；失业率 4.2%。",
        "收益率 5.29% 将升至 6%。": "",
        "油价同比 53.8% 推高通胀，议息会议与国会日程不变。": "油价同比 53.8% 推高通胀，议息会议与国会日程不变。",
        "联邦公开市场委员会关注利差 1.47 的位置。": "",
    }
    for raw, kept in texts.items():
        rep = []
        assert G.scrub_prices(raw, ALLOWED, rep, "t") == kept, raw
        assert bool(rep) == (raw != kept)
    # forecasts hidden in future words without digits, Chinese numerals or full-width digits (review E11): deleted
    for raw in ("关税可能推动股价下跌。", "美股将大幅下跌。", "预计油价会继续上涨。", "纳指可能跌到一万五千点。",
                "标普５００指数年底前回落到５２００点。", "十年期国债收益率年底升到百分之五。",
                "黄金价格料将在明年突破每盎司三千美元。", "股价下跌削弱财富效应。"):
        rep = []
        assert G.scrub_prices(raw, ALLOWED, rep, "t") == "", raw
        assert rep and "删除涉及价格的句子" in rep[0], raw
    assert G.normalize_numbers("一万五千点；百分之五；５２００；三点五") == "15000点;5%;5200;3.5"
    assert G.normalize_numbers("两党一致同意万一") == "两党一致同意万一"
    rep = []
    kept = "国会两党一致通过拨款；失业率 4.2%。"                               # no price term: untouched
    assert G.scrub_prices(kept, ALLOWED, rep, "t") == kept and not rep
    ans = G.guard_actor(block(opt("hold", 0.7, rationale="加息后美股预计将下跌 12%。核心 2.57% 高于目标"),
                              opt("hike_25", 0.3)), fed(), allowed=ALLOWED)
    assert ans.options[0]["rationale"] == "核心 2.57% 高于目标" and any("删除涉及价格的句子" in r for r in ans.report)


def test_numbers_that_restate_nothing_are_flagged_and_lengths_capped():
    ans = G.guard_actor(block(opt("hold", 0.7, rationale="失业率 6.3% 上升，第 3 季，近 8 个季度，s_y10"),
                              opt("hike_25", 0.3, rationale="长" * 100, label="一个很长很长很长很长的名称")),
                        fed(), allowed=ALLOWED)
    flagged = [r for r in ans.report if "引用的数值" in r]
    assert flagged == ["fed/hold 理由：引用的数值 6.3 不在状态表中（请核对）"]   # ordinals, counts, ids are not values
    assert len(ans.options[1]["rationale"]) == 80 and len(ans.options[1]["label"]) == 12
    assert any("超过 80 字" in r for r in ans.report)


def test_signals_are_validated_and_scrubbed():
    sig = {"text": "进一步收紧可能合适", "instrument": "d_ff", "direction": 1, "horizon_q": 9, "condition": "ebp_up_0.5",
           "commit": 1}
    ans = G.guard_actor(block(opt("hold", 0.6, signal=sig), opt("hike_25", 0.4, signal={"direction": 3})),
                        fed(), allowed=ALLOWED)
    assert ans.options[0]["signal"] == {"text": "进一步收紧可能合适", "instrument": "d_ff", "direction": 1,
                                        "horizon_q": 4, "condition": None, "commit": True}
    assert ans.options[1]["signal"] is None
    other = G.guard_actor(block(opt("hold", 1.0, signal={**sig, "instrument": "d_tariff", "horizon_q": 2}),
                                opt("hike_25", 0.0)), fed(), allowed=ALLOWED)
    assert other.options[0]["signal"]["instrument"] is None


def test_base_rate_departures_need_a_reason_and_required_options_are_reported():
    a = fed()
    a.options[2].required = True                                            # cut_25 must be offered
    ans = G.guard_actor(block(opt("hold", 0.9), opt("hike_25", 0.1), vs=None), a, allowed=ALLOWED)
    assert any("与历史基率相差超过 0.15" in r for r in ans.report) and ans.missing_required == ["cut_25"]


def test_declines_and_fallbacks():
    ans = G.guard_actor({"status": "decline", "decline_reason": "无法评估", "options": []}, fed(), allowed=ALLOWED)
    assert ans.fallback and ans.declined and ans.prior_only and ans.status == "fallback"
    assert [o["id"] for o in ans.options] == ["hold", "hike_25", "cut_50", "hike_50"]
    assert [o["p_fallback"] for o in ans.options] == pytest.approx([0.545, 0.205, 0.125, 0.125], abs=1e-3)
    assert all(o["p_llm"] is None and o["source"] == "base_rate" for o in ans.options)
    uniform = G.fallback_answer(china(), "超时")
    assert [o["p_fallback"] for o in uniform.options] == pytest.approx([1 / 3] * 3, abs=1e-3)
    assert "无历史先例" in uniform.report[0]
    assert G.guard_actor(None, fed()).problems == ["fed：缺少行为者的回答"]


def test_follower_rows():
    pair = tariff_pair()
    raw = {"status": "ok", "options": [opt("no_response", None), opt("retaliate_tariffs", None)],
           "reaction": {"leader": "president", "rows": [
               {"leader_option": "tariff_up_small", "follower_probs": [{"option": "no_response", "prob": 40},
                                                                       {"option": "retaliate_tariffs", "prob": 60}]},
               {"leader_option": "tariff_hold", "follower_probs": [{"option": "no_response", "prob": 0.5},
                                                                   {"option": "invented", "prob": 0.5}]},
               {"leader_option": "tariff_down", "follower_probs": [{"option": "negotiate_truce", "prob": 0.3}]},
               {"leader_option": "not_a_leader_option", "follower_probs": []}]}}
    ans = G.guard_actor(raw, china(), allowed=ALLOWED, follower=pair)
    assert ans.reaction == {"tariff_up_small": {"no_response": 0.4, "retaliate_tariffs": 0.6}}
    assert all(o["p_llm"] is None for o in ans.options) and ans.problems == []
    report = " ".join(ans.report)
    assert "invented 无效" in report and "偏离 1 太多" in report and "不在菜单中" in report and "百分数" in report


def test_merged_reactions_and_hazard_tilts():
    rep = []
    rows = [{"leader_option": "tariff_hold", "follower_probs": [{"option": "no_response", "prob": 0.9},
                                                                {"option": "negotiate_truce", "prob": 0.1}]}]
    out = G.guard_reactions([{"leader": "president", "follower": "china", "rows": rows},
                             {"leader": "president", "follower": "china", "rows": rows},
                             {"leader": "fed", "follower": "china", "rows": rows}], [tariff_pair()], rep)
    assert out == {"president": {"china": {"tariff_hold": {"no_response": 0.9, "negotiate_truce": 0.1}}}}
    assert any("重复" in r for r in rep) and any("不在本节点的行为者对中" in r for r in rep)
    rep = []
    tilts = G.guard_hazards([{"hazard": "oil_disruption", "prob": 0.5, "why": "航运受威胁"},
                             {"hazard": "financial_accident", "prob": 0.05},
                             {"hazard": "alien_invasion", "prob": 0.5}], hazards(), rep)
    assert tilts == {"oil_disruption": pytest.approx(0.12), "financial_accident": pytest.approx(0.04)}
    assert any("截断" in r for r in rep) and any("2 倍且没有依据" in r for r in rep) and any("不在目录" in r for r in rep)


# ---------------------------------------------------------------- synthesizer (structural)

def branch(bid, kind, p, lo=None, hi=None, **kw):
    region = None if kind == "mechanism" else {"lo_z": lo, "hi_z": hi}
    return {"id": bid, "label": kw.pop("label", bid), "kind": kind, "prob": p, "region": region,
            "shifts": kw.pop("shifts", {}), "cause": kw.pop("cause", {"options": []}), "mechanism": kw.pop("mech", ""),
            "anchors": kw.pop("anchors", []), "unintended": False, "facts": kw.pop("facts", []),
            "memory": kw.pop("memory", "各方维持现状"), **kw}


PARTITION = [branch("good", "partition", 0.15, None, -1), branch("central", "central", 0.65, -1, 1),
             branch("weak", "partition", 0.20, 1, None)]


def synth(*clusters):
    return {"schema_version": "agents-synth-1", "status": "ok", "clusters": list(clusters)}


def test_a_valid_synthesizer_answer_passes():
    req = synth_request()
    mech = branch("squeeze", "mechanism", 0.2, shifts={"x_supply_chain": {"mean": 8, "sd": 4}},
                  cause={"options": ["president.tariff_up_small"]}, anchors=["A-TAR-1"], memory="交货延迟 8 加剧")
    out = G.guard_synth(synth({"cluster_id": "k1", "status": "ok", "axis": "d_unrate", "branches": PARTITION},
                              {"cluster_id": "k2", "status": "ok", "axis": "d_gz",
                               "branches": [branch("central", "central", 0.8), mech]}), req, allowed=ALLOWED)
    k1, k2 = out["k1"], out["k2"]
    assert k1.status == k2.status == "ok" and k1.report == [] and k2.report == []
    assert [b["p_llm"] for b in k1.branches] == [0.15, 0.65, 0.20]
    assert k2.branches[1]["shifts"] == {"x_supply_chain": {"mean": 8.0, "sd": 4.0}}
    assert k2.branches[1]["memory"] == "交货延迟 8 加剧"                       # the number is this branch's shift


def test_synthesizer_powers_and_written_fields():
    req = synth_request()
    mech = branch("m", "mechanism", 0.2, shifts={"d_gz": {"mean": 0.5, "sd": 0.1}, "d_ff": {"mean": 1, "sd": 0},
                                                 "d_core": {"mean": 0.4, "sd": 0.2}},
                  cause={"options": ["fed.hike_25", "china.invented"]})
    out = G.guard_synth(synth({"cluster_id": "k2", "status": "ok", "axis": None,
                               "branches": [branch("central", "central", 0.8), mech]}), req, allowed=ALLOWED)
    b = out["k2"].branches[1]
    assert b["shifts"] == {"d_core": {"mean": 0.4, "sd": 0.2}} and b["cause"]["options"] == ["fed.hike_25"]
    assert sum("不能对" in r for r in out["k2"].report) == 2
    written = G.guard_synth(synth({"cluster_id": "k1", "status": "ok", "axis": "d_tariff", "branches": PARTITION}),
                            req, allowed=ALLOWED)
    assert written["k1"].status == "default" and "分区轴 d_tariff 不允许" in written["k1"].report[0]
    no_shift = G.guard_synth(synth({"cluster_id": "k1", "status": "ok", "axis": "d_unrate", "branches": [
        *PARTITION[:2], {**PARTITION[2], "prob": 0.1}, branch("m", "mechanism", 0.1, shifts={"d_ff": {"mean": 1}})]}),
        req, allowed=ALLOWED)
    assert [x["id"] for x in no_shift["k1"].branches] == ["good", "central", "weak"]
    assert no_shift["k1"].branches[1]["p_llm"] == pytest.approx(0.75)          # the empty mechanism joined the centre


@pytest.mark.parametrize("branches, why", [
    ([branch("a", "partition", 0.3, None, -1), branch("c", "central", 0.7, -0.5, None)], "首尾相接"),
    ([branch("a", "partition", 0.3, None, -1.2), branch("c", "central", 0.7, -1.2, None)], "网格"),
    ([branch("a", "central", 0.3, None, 0), branch("c", "central", 0.7, 0, None)], "只有一个模型中枢"),
    ([branch("a", "partition", 0.3, -2, -1), branch("c", "central", 0.7, -1, None)], "覆盖整条轴"),
    ([branch("a", "partition", 0.3, 1, -1), branch("c", "central", 0.7, -1, None)], "下端不小于上端"),
])
def test_partition_geometry_errors_fall_back_to_the_default_partition(branches, why):
    out = G.guard_synth(synth({"cluster_id": "k1", "status": "ok", "axis": "d_unrate", "branches": branches}),
                        synth_request(), allowed=ALLOWED)
    assert out["k1"].status == "default" and any(why in r for r in out["k1"].report)


def test_synthesizer_texts_numbers_facts_and_probabilities():
    req = synth_request()
    bs = copy.deepcopy(PARTITION)
    bs[0]["memory"] = "就业韧性，失业率降到 3.1%，美股预计将上涨 10%。"
    bs[1]["memory"] = "疫情爆发，供应链中断"
    bs[1]["facts"] = [{"text": "稀土供应紧张", "exit_hazard": 0.99}, {"text": "第二条", "exit_hazard": None},
                      {"text": "第三条", "exit_hazard": 0.5}]
    out = G.guard_synth(synth({"cluster_id": "k1", "status": "ok", "axis": "d_unrate", "branches": bs}), req,
                        allowed=ALLOWED)
    k1 = out["k1"]
    assert "美股" not in k1.branches[0]["memory"] and "3.1" not in k1.branches[0]["memory"]
    assert k1.branches[1]["memory"] == ""                                       # a new exogenous event, deleted
    assert k1.branches[1]["facts"] == [{"text": "稀土供应紧张", "exit_hazard": 0.9},
                                        {"text": "第二条", "exit_hazard": 0.5}]
    assert out["k2"].status == "default" and "缺少该动作组" in out["k2"].reason
    allowed_event = G.guard_synth(synth({"cluster_id": "k1", "status": "ok", "axis": "d_unrate", "branches": bs}),
                                  synth_request(nature_events=["石油供给中断"]), allowed=ALLOWED)
    assert allowed_event["k1"].branches[1]["memory"] == "疫情爆发，供应链中断"
    pct = copy.deepcopy(PARTITION)
    for b, p in zip(pct, (15, 65, 20)):
        b["prob"] = p
    assert [b["p_llm"] for b in G.guard_synth(synth({"cluster_id": "k1", "status": "ok", "axis": "d_unrate",
                                                     "branches": pct}), req)["k1"].branches] == [0.15, 0.65, 0.20]
    off = copy.deepcopy(PARTITION)
    off[1]["prob"] = 0.2
    assert G.guard_synth(synth({"cluster_id": "k1", "status": "ok", "axis": "d_unrate", "branches": off}),
                         req)["k1"].problems
    declined = G.guard_synth(synth({"cluster_id": "k1", "status": "decline", "decline_reason": "无法分析",
                                    "branches": []}), req)
    assert declined["k1"].status == "default" and declined["k1"].declined


# ---------------------------------------------------------------- root translation

def test_root_forced_fields_need_the_users_own_words():
    levers = [{"id": "d_tariff", "zh": "有效关税率变动", "hist": {"min": -3.21, "max": 5.44, "p01": -1.36, "p99": 1.49}}]
    narrative = "总统宣布对全部进口商品加征 10% 关税，下季度生效。我认为美联储一定会降息。"
    payload = {"schema_version": "agents-root-1", "status": "ok",
               "h0": {"headline": "关税升级", "summary": "总统宣布加征关税。油价将升至 150 美元。",
                      "facts_used": [{"field": "s_ff", "value": 3.75}, {"field": "s_y10", "value": 7.0}]},
               "forced": [{"field": "d_tariff", "value": 9.0, "quote": "对全部进口商品加征 10% 关税"},
                          {"field": "d_tariff", "value": 1.0, "quote": "加征"},
                          {"field": "d_ff", "value": -0.25, "quote": "美联储一定会降息"},
                          {"field": "d_tariff", "value": 2.0, "quote": "用户没说过的话"}],
               "user_views": ["我认为美联储一定会降息。"], "unmapped": ["长" * 100]}
    out = G.guard_root(payload, narrative, levers, allowed=ALLOWED)
    assert out["ok"] and out["forced"] == [{"field": "d_tariff", "value": 5.44, "quote": "对全部进口商品加征 10% 关税"}]
    assert out["h0"]["summary"] == "总统宣布加征关税。" and out["user_views"] == ["我认为美联储一定会降息。"]
    assert len(out["unmapped"][0]) == 80
    report = " ".join(out["report"])
    assert "不是可设定的字段" in report and "重复" in report and "引文不是用户原文" in report
    assert "s_y10=7.0 与状态表不符" in report and "超出历史极值" in report
    assert not G.guard_root({"status": "decline", "decline_reason": "不处理"}, narrative, levers)["ok"]
