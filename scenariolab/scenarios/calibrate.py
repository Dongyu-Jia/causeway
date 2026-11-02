"""Income distribution -> consumption calculator (人员收入对宏观经济的影响).

A transparent back-of-envelope model of how labour-income shocks to three
income groups and a change in equity wealth move aggregate consumption, real
growth, unemployment, the labour share and services inflation. Every
intermediate number is returned for the UI. All shocks are cumulative over
the scenario phase they apply to.

Per group g (bottom 50% / 50-90% / top 10%), in shares of consumption C:
    labour channel   dC_g / C_g = MPC_g * shock_g * (labour income_g / C_g)
    wealth channel   dC_g / C   = wealth_mpc * dE * own_g * (equity wealth / C)
    aggregate        dC / C     = sum_g share_g * labour_g + sum_g wealth_g
    real growth      dY / Y     = (C / Y) * dC / C        (first round, no multiplier)
    unemployment     du = direct job losses + Okun (-okun * dY/Y)
    labour share     dlog LS = log(1 + dLI / LI) - log(1 + dY / Y)

How the result enters the engine:
  * demand effects become entropy-pooling views on persistent states
    (s_ip_yoy, s_services_yoy), which choose the historical months drawn;
  * the Okun part becomes a view on the drawn months' unemployment trend
    (f_unrate). Without it the analog months bring their own cycle (e.g. the
    2009-14 profit recovery also carries a falling unemployment rate) and the
    engine contradicts the calculator. The view carries vol_mult: a mean view
    tilts toward the blocks with the most extreme means, and the 2020 lockdown
    blocks (+10.4 pp in April 2020) are by far the most extreme even at their
    0.25 prior weight (config.RARE_MONTHS), so without a bound on the drawn
    months' variance a few paths would replay the lockdown (UNRATE_VOL_MULT);
  * the direct distribution effects (job losses from displacement, the
    labour-share shift) have no historical analog, so they are added as
    bounded factor drifts on top of the drawn months. A view of zero on the
    drawn months' labour-share trend (f_labor_share) goes with the drift, for
    the same reason as the Okun view: the months these views select (mostly the
    2010s) bring a labour-share decline of their own (-0.2% to -0.65% a year in
    the built-in specs), which on top of the drift would count the shift twice.
"""
from __future__ import annotations

import copy
import math

GROUPS = ("bottom50", "mid40", "top10")
GROUP_NAMES = {"bottom50": "收入后 50%", "mid40": "收入 50-90%", "top10": "收入前 10%"}

# Defaults and where they come from (shown in the UI next to each number).
DEFAULTS: dict = {
    "consumption_share": {"bottom50": 0.19, "mid40": 0.32, "top10": 0.49},
    "labor_income_shock": {"bottom50": 0.0, "mid40": 0.0, "top10": 0.0},
    "mpc_labor": {"bottom50": 0.6, "mid40": 0.4, "top10": 0.2},
    "labor_income_ratio": {"bottom50": 0.66, "mid40": 1.04, "top10": 0.57},
    "equity_ownership": {"bottom50": 0.01, "mid40": 0.12, "top10": 0.87},
    "worker_share": {"bottom50": 0.5, "mid40": 0.4, "top10": 0.1},
    "wealth_mpc": 0.03,
    "equity_wealth_change": 0.0,
    "equity_wealth_to_consumption": 2.7,
    "consumption_gdp_share": 0.68,
    "job_loss_fraction": 0.5,
    "okun": 0.5,
    "ip_beta": 1.7,
    "services_slope": 0.2,
    "months": 48,
}
# Chinese display names of the parameters (the UI's parameter table shows these, not the keys).
PARAM_NAMES = {
    "consumption_share": "消费占比", "labor_income_shock": "劳动收入冲击", "mpc_labor": "劳动收入的边际消费倾向",
    "labor_income_ratio": "劳动收入 / 消费", "equity_ownership": "持股份额", "worker_share": "就业人口占比",
    "wealth_mpc": "财富效应系数", "equity_wealth_change": "股票财富变化", "equity_wealth_to_consumption": "股票财富 / 消费",
    "consumption_gdp_share": "消费占 GDP", "job_loss_fraction": "失业比例", "okun": "奥肯系数",
    "ip_beta": "工业产出弹性", "services_slope": "服务通胀斜率", "months": "阶段月数",
}
PARAM_NOTES = {
    "consumption_share": "各组占全美消费支出的比例；Moody's 估计 2025 年收入前 10% 约占 49.7%",
    "labor_income_shock": "本阶段累计的劳动收入变化（小数，-0.05 = 下降 5%）；白领替代主要冲击 50-90% 和前 10%",
    "mpc_labor": "劳动收入的边际消费倾向：低收入组几乎花掉每一元，高收入组大部分储蓄",
    "labor_income_ratio": "各组劳动收入 / 各组消费：由 DINA 量级的劳动收入分布 17%/45%/38%、"
                          "雇员报酬/个人消费 0.74（FRED COE/PCE，2026Q3）和消费占比推出",
    "equity_ownership": "各组持有的股票与基金份额（美联储 DFA）",
    "worker_share": "各组在就业人口中的大致占比（按收入分组的人数）",
    "wealth_mpc": "股票财富效应：每 1 美元财富变化带来的年度消费变化（约 3 美分）",
    "equity_wealth_change": "本阶段累计的股票财富变化（小数）",
    "equity_wealth_to_consumption": "家庭股票财富 / 个人消费：直接持股 52.1 万亿 + 共同基金 14.4 万亿中约 60%（约 8.6 万亿），"
                                    "合计约 60.8 万亿（2026Q2），为 2026 年三季度个人消费（年化约 22.3 万亿）的约 2.7 倍"
                                    "（FRED HNOCEAQ027S、HNOMFAQ027S、PCE）",
    "consumption_gdp_share": "个人消费占 GDP 的比例（FRED PCE/GDP ≈ 0.68）",
    "job_loss_fraction": "收入下降中由失业（而不是降薪、减工时）造成的比例",
    "okun": "奥肯系数：实际增长每低 1 个百分点，失业率约升 0.5 个百分点（1973-2026 年回归 0.51）",
    "ip_beta": "工业产出同比对实际 GDP 同比的弹性（1973-2026 年季度回归 1.69，R²=0.74）",
    "services_slope": "增长缺口每 1 个百分点，服务通胀一年后约变化 0.2 个百分点（1985-2026 年回归 0.22）",
    "months": "本阶段的月数（由情景阶段自动填入）",
}
DEFAULT_BASE = {"ip_yoy": 1.8, "services_yoy": 3.2}
# vol_mult of the Okun view: a bound on the drawn months' monthly variance around the view mean
# (ep.apply_views "__sq" constraint), in units of the prior's monthly sd of f_unrate. That prior sd is
# 0.281 pp (months weighted as an unconditional path draws them, 2020-03..08 at 0.25x); without 2020
# it is 0.175 pp. 0.62 x 0.281 = 0.175: the drawn months move about as much as an ordinary month since
# 1973 (the 1973-75 and 2007-09 recessions: 0.19-0.27), and the lockdown blocks drop out (their share
# of the draws falls from 0.5-2% under a plain mean view to under 0.1%).
UNRATE_VOL_MULT = 0.62


def _merge(params: dict | None) -> tuple[dict, list[str]]:
    p = copy.deepcopy(DEFAULTS)
    ignored = []
    for key, val in (params or {}).items():
        if key not in DEFAULTS:
            ignored.append(str(key))
            continue
        if isinstance(DEFAULTS[key], dict):
            if isinstance(val, dict):
                p[key].update({g: float(v) for g, v in val.items() if g in GROUPS})
                ignored += [f"{key}.{g}" for g in val if g not in GROUPS]
            else:
                p[key] = {g: float(val) for g in GROUPS}
        else:
            p[key] = float(val)
    p["months"] = max(int(round(p["months"])), 1)
    return p, ignored


def distribution_impact(params: dict | None = None) -> dict:
    """Every intermediate number of the calculator (decimals unless the key says pp)."""
    p, ignored = _merge(params)
    share, shock, mpc = p["consumption_share"], p["labor_income_shock"], p["mpc_labor"]
    ratio, own, workers = p["labor_income_ratio"], p["equity_ownership"], p["worker_share"]
    ew = p["equity_wealth_to_consumption"]
    groups = []
    labor_total = wealth_total = 0.0
    li_level = sum(share[g] * ratio[g] for g in GROUPS)
    li_change = 0.0
    direct_u = 0.0
    for g in GROUPS:
        labor_g = mpc[g] * shock[g] * ratio[g]                       # % of the group's own consumption
        wealth_agg = p["wealth_mpc"] * p["equity_wealth_change"] * own[g] * ew   # % of aggregate consumption
        wealth_g = wealth_agg / share[g] if share[g] > 0 else 0.0
        contrib_labor = share[g] * labor_g
        labor_total += contrib_labor
        wealth_total += wealth_agg
        li_change += share[g] * ratio[g] * shock[g]
        job_loss = workers[g] * p["job_loss_fraction"] * max(-shock[g], 0.0)
        direct_u += 100.0 * job_loss
        groups.append({
            "id": g, "name": GROUP_NAMES[g], "consumption_share": share[g], "labor_income_shock": shock[g],
            "mpc_labor": mpc[g], "labor_income_ratio": ratio[g], "equity_ownership": own[g],
            "worker_share": workers[g],
            "labor_income_share": share[g] * ratio[g] / li_level if li_level > 0 else 0.0,
            "consumption_change_labor": labor_g, "consumption_change_wealth": wealth_g,
            "consumption_change": labor_g + wealth_g,
            "contribution_labor": contrib_labor, "contribution_wealth": wealth_agg,
            "contribution": contrib_labor + wealth_agg,
            "unemployment_direct_pp": 100.0 * job_loss,
        })
    dc = labor_total + wealth_total
    dy = p["consumption_gdp_share"] * dc
    dli = li_change / li_level if li_level > 0 else 0.0
    dlog_ls = math.log1p(dli) - math.log1p(dy)
    okun_u = -p["okun"] * 100.0 * dy
    years = p["months"] / 12.0
    return {
        "params": p, "ignored_params": ignored, "groups": groups,
        "consumption_change_labor": labor_total, "consumption_change_wealth": wealth_total,
        "consumption_change": dc, "real_gdp_change": dy,
        "real_growth_per_year_pp": 100.0 * dy / years,
        "labor_income_change": dli, "labor_share_log_change": dlog_ls,
        "unemployment_direct_pp": direct_u, "unemployment_okun_pp": okun_u,
        "unemployment_change_pp": direct_u + okun_u,
        "years": years,
    }


def _pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:+.{digits}f}%"


def distribution_views(params: dict | None = None, base: dict | None = None) -> tuple[list[dict], dict]:
    """(views, explanation): EP views for one phase plus every number behind them.

    base: the no-shock levels the views are measured from, {"ip_yoy", "services_yoy"}
    in percent (library.py passes today's recent averages).
    explanation["overrides"]["factor_drift"] holds the direct distribution
    effects (annual, in factor units: pp for unrate, log for labor_share).
    """
    imp = distribution_impact(params)
    b = {**DEFAULT_BASE, **{k: float(v) for k, v in (base or {}).items() if k in DEFAULT_BASE}}
    p, years = imp["params"], imp["years"]
    growth_pp = imp["real_growth_per_year_pp"]
    ip_target = b["ip_yoy"] + p["ip_beta"] * growth_pp
    services_target = b["services_yoy"] + p["services_slope"] * 0.5 * 100.0 * imp["real_gdp_change"]
    unrate_drift = imp["unemployment_direct_pp"] / years
    okun_rate = imp["unemployment_okun_pp"] / years
    ls_drift = imp["labor_share_log_change"] / years
    views = [
        {"var": "s_ip_yoy", "mean": round(ip_target, 3),
         "why": f"收入分配计算器：实际增长每年 {growth_pp:+.2f} 个百分点 × 工业产出弹性 {p['ip_beta']:.2f}，"
                f"在 {b['ip_yoy']:.2f}% 的基础上"},
        {"var": "s_services_yoy", "mean": round(services_target, 3),
         "why": f"收入分配计算器：阶段平均需求缺口 {_pct(0.5 * imp['real_gdp_change'])} × 斜率 "
                f"{p['services_slope']:.2f}，在 {b['services_yoy']:.2f}% 的基础上"},
        {"var": "f_unrate", "mean_annual": round(okun_rate, 4), "vol_mult": UNRATE_VOL_MULT,
         "why": f"收入分配计算器：奥肯部分每年 {okun_rate:+.2f} 个百分点，让抽到的月份不带自己的复苏或衰退趋势；"
                f"波动约束（{UNRATE_VOL_MULT} 倍历史标准差，即不含 2020 年时的月度波动）排除 2020 年 4 月这类单月 +10 个百分点的封锁月份"},
        {"var": "f_labor_share", "mean_annual": 0.0,
         "why": "收入分配计算器：劳动收入占比的变化全部由因子漂移给出，让抽到的月份不带自己的劳动收入占比趋势"
                "（内置情景里这些月份每年自带约 −0.2% 到 −0.65%），以免重复计算"},
    ]
    overrides = {"factor_drift": {"unrate": round(unrate_drift, 4), "labor_share": round(ls_drift, 5)}}
    drift_notes = {
        "unrate": f"直接失业 {imp['unemployment_direct_pp']:+.2f} 个百分点分摊到 {years:.1f} 年；"
                  f"奥肯部分（{imp['unemployment_okun_pp']:+.2f}）由失业率月度变化视图约束抽到的历史月份",
        "labor_share": f"劳动收入 {_pct(imp['labor_income_change'])}、实际 GDP {_pct(imp['real_gdp_change'])}，"
                       f"劳动收入占比累计 {_pct(imp['labor_share_log_change'])}，分摊到 {years:.1f} 年",
    }
    top = max(imp["groups"], key=lambda g: abs(g["contribution"]))
    text = (
        f"劳动收入冲击使消费变化 {_pct(imp['consumption_change_labor'])}，股票财富变化 "
        f"{_pct(p['equity_wealth_change'], 0)} 通过财富效应使消费变化 {_pct(imp['consumption_change_wealth'])}，"
        f"合计 {_pct(imp['consumption_change'])}；按消费占 GDP {p['consumption_gdp_share']:.2f} 折算，"
        f"实际 GDP 一阶影响 {_pct(imp['real_gdp_change'])}（约每年 {growth_pp:+.2f} 个百分点）。"
        f"贡献最大的是{top['name']}（{_pct(top['contribution'])}）。失业率累计 "
        f"{imp['unemployment_change_pp']:+.2f} 个百分点，劳动收入占比累计 {_pct(imp['labor_share_log_change'])}。"
    )
    explanation = {
        "kind": "distribution", "title": "人员收入对宏观经济的影响",
        "text": text, "base": b, "impact": imp, "views": views,
        "overrides": overrides, "override_notes": drift_notes,
        "param_names": PARAM_NAMES, "param_notes": PARAM_NOTES,
        "formulae": [
            "各组消费变化（劳动渠道）= 边际消费倾向 × 劳动收入冲击 × 劳动收入/消费",
            "各组消费变化（财富渠道）= 财富效应系数 × 股票财富变化 × 持股份额 × 股票财富/消费",
            "总消费变化 = Σ 消费占比 × 劳动渠道 + Σ 财富渠道",
            "实际 GDP 一阶影响 = 消费占 GDP × 总消费变化（不含乘数和政策反应）",
            "失业率变化 = Σ 就业占比 × 失业比例 × 收入降幅 − 奥肯系数 × 实际 GDP 影响",
            "劳动收入占比 log 变化 = log(1 + 劳动收入变化) − log(1 + 实际 GDP 影响)",
        ],
        "engine_mapping": [
            "需求效应 → 熵池视图：工业产出同比、服务通胀（持续性强的状态变量，决定抽哪些历史月份）",
            "奥肯部分 → 失业率月度变化视图（带波动约束），使抽到的月份的失业率趋势与需求效应一致",
            "分配的直接效应 → 有界因子漂移：白领失业、劳动收入占比（历史上没有对应月份，叠加在抽到的月份上）；"
            "同时用劳动收入占比月度变化视图（均值为零）让抽到的月份不带自己的劳动收入占比趋势",
        ],
        "caveats": [
            "这是一阶的会计推算，不含乘数、价格和政策反应；只用于设定情景的视图和漂移，所有资产数字仍由引擎从历史月份生成。",
            "财富效应只算股票，不含房产；高收入组的边际消费倾向和持股比例都有较大不确定性。",
        ],
    }
    return views, explanation
