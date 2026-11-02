"""Hand-written specs shared by the agent-LLM tests (illustrative menus and numbers, not the production catalogue or
priors; scenariolab/agents/actors.py and priors.py own those)."""
from scenariolab.llm.agent_prompts import ActorSpec, HazardSpec, NodeContext, ReactionSpec, SynthRequest

STATE = "\n".join([
    "# 一、状态", "## 1. 硬状态（季末）",
    "| 字段 | 名称 | 单位 | 本季末 | 一年前 | 1973 年以来分位 |", "|---|---|---|---|---|---|",
    "| s_ff | 联邦基金利率 | % | 3.75 | 4.22 | 40% |", "| s_core_yoy | 核心 CPI 同比 | % | 2.57 | 3.02 | 42% |",
    "| s_y10 | 10 年期国债收益率 | % | 5.29 | 4.16 | 48% |", "| s_unrate | 失业率 | % | 4.2 | 4.4 | 13% |",
    "| s_oil_yoy | 油价同比 | % | 53.8 | -8.1 | 88% |"])

D_TARIFF = {"id": "d_tariff", "zh": "有效关税率变动", "unit": "pp/季",
            "hist": {"p01": -1.3559, "p50": -0.0058, "p99": 1.485, "min": -3.2147, "max": 5.4383, "sd_q_1985_2019": 0.1322}}


def fed(**kw) -> ActorSpec:
    d = dict(id="fed", zh="美联储（FOMC）", instruments=["d_ff"], base_rate_source="有序 probit（示意）",
             packet="## 你的账户：美联储（2026-10-28）\n资产：总资产 6,746｜国债 4,577", private="私下：年内不再加息",
             options=[
                 {"id": "cut_75plus", "label": "紧急降息", "base_rate": 0.03, "writes": "d_ff = 幅度", "point": {"d_ff": -1.0},
                  "magnitude": {"default": -1.0, "lo": -1.75, "hi": -0.75, "unit": "pp", "step": 0.25},
                  "note": "仅在金融事故后开放"},
                 {"id": "cut_50", "label": "降息 50bp", "base_rate": 0.11, "writes": "d_ff = -0.5", "point": {"d_ff": -0.5}},
                 {"id": "cut_25", "label": "降息 25bp", "base_rate": 0.09, "writes": "d_ff = -0.25", "point": {"d_ff": -0.25}},
                 {"id": "hold", "label": "按兵不动", "base_rate": 0.48, "status_quo": True, "writes": "d_ff = 0",
                  "point": {"d_ff": 0.0}},
                 {"id": "hike_25", "label": "加息 25bp", "base_rate": 0.18, "writes": "d_ff = 0.25", "point": {"d_ff": 0.25}},
                 {"id": "hike_50", "label": "加息 50bp", "base_rate": 0.11, "writes": "d_ff = 0.5", "point": {"d_ff": 0.5}}])
    d.update(kw)
    return ActorSpec(**d)


def president(**kw) -> ActorSpec:
    d = dict(id="president", zh="美国总统与白宫", custom=True, levers=[D_TARIFF], facts=["export_controls_level"],
             instruments=["d_tariff"],
             options=[{"id": "tariff_hold", "label": "关税不变", "base_rate": 0.70, "status_quo": True},
                      {"id": "tariff_up_small", "label": "小幅加征", "base_rate": 0.15,
                       "magnitude": {"default": 0.5, "lo": 0.25, "hi": 1.5, "unit": "pp"}},
                      {"id": "tariff_down", "label": "下调", "base_rate": 0.15,
                       "magnitude": {"default": -0.5, "lo": -3.0, "hi": -0.25, "unit": "pp"}}])
    d.update(kw)
    return ActorSpec(**d)


def china(**kw) -> ActorSpec:
    d = dict(id="china", zh="中国", options=[{"id": "no_response", "label": "不回应", "status_quo": True},
                                            {"id": "retaliate_tariffs", "label": "关税反制"},
                                            {"id": "negotiate_truce", "label": "谈判休战"}])
    d.update(kw)
    return ActorSpec(**d)


def ctx_root(**kw) -> NodeContext:
    d = dict(quarter_next="2026Q4", depth=1, state=STATE, news=["10 月议息会议维持利率不变", "国会通过持续决议"],
             board=[{"actor": "fed", "actor_zh": "美联储", "text": "进一步收紧可能合适", "instrument": "d_ff",
                     "direction": 1, "horizon_q": 2, "commit": False, "credibility": 0.5, "kept": 1, "total": 2}],
             facts=[{"text": "石油供给中断持续", "exit_hazard": 0.42}],
             institutions=[{"zh": "10 月议息结果", "value": "维持", "status": "未确认", "date": "2026-10-28"}],
             calendar=["11 月 3 日中期选举", "12 月 8–9 日议息会议"], as_of="2026-11-01", order_seed="")
    d.update(kw)
    return NodeContext(**d)


def ctx_deep(**kw) -> NodeContext:
    d = dict(quarter_next="2027Q1", depth=2, quarter_done="2026Q4", state=STATE,
             memory="2026Q4：美联储按兵不动；关税不变（本支条件概率 0.62）", order_seed="")
    d.update(kw)
    return NodeContext(**d)


def tariff_pair() -> ReactionSpec:
    return ReactionSpec(leader="president", follower="china", leader_zh="美国总统", follower_zh="中国",
                        leader_options=["tariff_hold", "tariff_up_small", "tariff_down"],
                        follower_options=["no_response", "retaliate_tariffs", "negotiate_truce"],
                        prior={"tariff_hold": {"no_response": 0.80, "retaliate_tariffs": 0.05, "negotiate_truce": 0.15},
                               "tariff_up_small": {"no_response": 0.40, "retaliate_tariffs": 0.50, "negotiate_truce": 0.10},
                               "tariff_down": {"no_response": 0.20, "negotiate_truce": 0.80}})


def hazards() -> list:
    return [HazardSpec(id="oil_disruption", zh="新的石油供给中断", mass=0.04, table_rate=0.042),
            HazardSpec(id="financial_accident", zh="金融事故", mass=0.02, table_rate=0.023)]


FIELDS = [
    {"id": "d_unrate", "zh": "失业率变动", "unit": "pp/季", "partition": True, "mechanism": True, "p01": -1.07, "p50": 0.0,
     "p99": 1.39},
    {"id": "d_core", "zh": "核心通胀", "unit": "%年化", "partition": True, "mechanism": True, "p01": 0.05, "p50": 2.93,
     "p99": 13.6},
    {"id": "x_supply_chain", "zh": "供应链压力", "unit": "指数", "partition": True, "mechanism": True},
    {"id": "d_cpi_goods", "zh": "商品通胀", "unit": "%年化", "partition": False, "mechanism": True},
    {"id": "d_gz", "zh": "GZ 信用利差变动", "unit": "pp/季", "partition": True, "mechanism": False},
    {"id": "d_cpbill", "zh": "融资市场压力", "unit": "pp/季", "partition": True, "mechanism": False},
]
THREE = [{"lo": None, "hi": -1, "mass": 0.158, "name": "好于中枢"}, {"lo": -1, "hi": 1, "mass": 0.684, "name": "中枢"},
         {"lo": 1, "hi": None, "mass": 0.158, "name": "走弱"}]
GZ = [{"lo": None, "hi": -1, "mass": 0.12, "name": "宽松"}, {"lo": -1, "hi": 1, "mass": 0.70, "name": "中枢"},
      {"lo": 1, "hi": 2.5, "mass": 0.17, "name": "收紧"}, {"lo": 2.5, "hi": None, "mass": 0.01, "name": "恐慌"}]


def synth_request(**kw) -> SynthRequest:
    d = dict(ctx=ctx_root(), fields=FIELDS, sector_summary="家庭储蓄率 3.75%，银行未实现损失收窄",
             anchors=[{"id": "A-MON-1", "text": "加息 → 信用条件收紧（分区）", "now": "+", "q4": "+", "source": "示意"},
                      {"id": "A-TAR-1", "text": "关税 → 商品 CPI +", "now": "+", "q4": "+", "source": "示意"}],
             channels=[{"channel": "关税", "cls": "thin", "w": 0.8, "cap": 2.5}],
             signatures=[{"key": "x_supply_chain↑", "need": "d_cpi_goods 不降；d_capex 不升"}],
             clusters=[{"id": "k1", "actions": [{"actor": "fed", "actor_zh": "美联储", "option": "hold", "label": "按兵不动",
                                                 "writes": "d_ff = 0"},
                                                {"actor": "president", "actor_zh": "美国总统", "option": "tariff_hold",
                                                 "label": "关税不变"}],
                        "written_fields": ["d_ff", "d_tariff"],
                        "summary": [{"field": "d_unrate", "zh": "失业率变动", "unit": "pp", "median": -0.02, "p10": -0.36,
                                     "p90": 0.30, "sd": 0.26}],
                        "masses": {"d_unrate": THREE, "d_gz": GZ}, "stress_share": 0.13},
                       {"id": "k2", "actions": [{"actor": "fed", "actor_zh": "美联储", "option": "hike_25", "label": "加息"},
                                                {"actor": "president", "actor_zh": "美国总统", "option": "tariff_up_small",
                                                 "label": "小幅加征", "writes": "d_tariff = 0.5"}],
                        "written_fields": ["d_ff", "d_tariff"], "support": ["d_tariff 罕见（1%–99% 之外）"],
                        "masses": {"d_gz": GZ}, "regime": ["核心通胀高于 4%"]}])
    d.update(kw)
    return SynthRequest(**d)
