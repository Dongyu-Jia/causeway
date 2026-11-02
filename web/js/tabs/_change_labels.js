// The walk-forward's change ids in words: the candidates of diagnostics.json (and the iterations' picks that
// valuation.json's factor_walkforward repeats). One map for the diagnostics and valuation pages, so the two never
// drift apart. Stage-B ids (a change to one asset's equation) carry the option the factor round adopted; the
// diagnostics page prefers the build's own stage-B row label, which names the option this build kept.
export const CHANGE_LABELS = {
  add_valuation: '+估值变动', add_sentiment: '+消费者信心', add_profit_share: '+利润占比', add_capex_share: '+资本开支占比',
  add_wealth: '+家庭净值', add_white_collar: '+白领就业占比', add_wage: '+普通工资', add_real_y10: '+实际收益率',
  add_profit_growth: '+利润增长', add_fcf_share: '+利润减资本开支', add_cape: '+CAPE 变动', lags1: '1 个月滞后', lags3: '1–3 个月滞后',
  regime_inflation: '通胀状态交互', shocks_add: '+结构冲击', shocks_replace_oil: '结构冲击替代油价', 'ridge_0.01': '岭回归 0.01',
  'ridge_0.05': '岭回归 0.05', window_15: '只用近 15 年', window_25: '只用近 25 年', mlp16: '神经网络 16', mlp32: '神经网络 32',
  // the missing-factor round (FACTORS_DESIGN §2.1): the adoptable changes and the audit's controls
  add_gz: '+GZ 信用利差', add_tp10_x_core4: '期限溢价×高核心通胀', drop_labor_share: '去掉劳动收入占比',
  add_tp10_x_infl4: '期限溢价×高通胀（对照）', add_tp10: '+期限溢价（对照）', add_ebp: '+超额债券溢价（对照）',
  // stage B (one asset's own equation)
  energy_metals_oil: '能源股 +基础金属相对油价变动', gold_window: '黄金从 1982 年起估计',
};
