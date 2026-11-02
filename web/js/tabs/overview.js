// Overview: today's macro state (with the real-time macro regime from regimes.json, optional), which
// historical eras it resembles, the scenario set, and the next four quarters as a scenario-conditional simulation
// (forecast_mix.json, optional: the default engine at the default probabilities; mixQuantile from the forecast tab).
import { mixQuantile } from './forecast.js';
import { loadSwitch, quarterlyDefault } from './_switch.js';
const KEY_STATES = ['cpi_yoy', 'core_yoy', 'goods_yoy', 'ff', 'y10', 'spread', 'unrate', 'real_oil',
  'labor_share', 'profit_share', 'div_yield', 'sentiment'];

function percentileRank(series, value) {
  const xs = (series || []).filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  if (!xs.length || value === null || value === undefined) return null;
  return xs.filter(v => v <= value).length / xs.length;
}

export async function render(root, { api, charts, ui, meta, alive = () => root.isConnected }) {
  const { h, card } = ui;
  // how many built-in scenarios there are is known once scenarios.json has loaded
  const nScen = () => h('span', { class: 'ov-nscen' }, '十几');
  root.append(h('p', { class: 'lead' },
    '这是模拟，不是预测：预测只在情景假说（及其概率）里，情景推因子、因子推资产。只解释季度和更长的期限。',
    '从历史数据出发：先看清每个时代的宏观“指纹”和因果传导，再用回测检验模型能否复现实际走势，',
    '最后从今天出发生成 ', nScen(), ' 个情景，评估组合和提取策略。所有情景数字都来自模型，大模型只负责把文字翻译成情景表。'));

  const [eras, scen, fc, rg, mix, sw] = await Promise.all([
    api.optional('eras', meta),
    api.optional('scenarios', meta),
    api.optional('forecast', meta),
    api.optional('regimes', meta),
    api.optional('forecast_mix', meta),
    loadSwitch(),
  ]);
  if (!alive()) return;                  // the user moved on while the results loaded
  const paintCount = () => {
    const n = (scen?.scenarios || []).length;
    if (n) for (const el of root.querySelectorAll('.ov-nscen')) el.textContent = String(n);
  };
  paintCount();
  // the missing-factor audit (optional; skipped when /api/meta's `results` says it is absent): one line below
  const faP = api.optional('factor_audit', meta);

  // ---- today's state
  const today = meta?.today?.states || {};
  const tl = eras?.timeline?.series || {};
  const tiles = KEY_STATES.filter(s => today[s] !== undefined && today[s] !== null).map(s => {
    const st = ui.META.states[s] || { name: s, unit: '' };
    const v = today[s];
    const pr = percentileRank(tl[s], v);
    const shown = st.unit === '%' || st.unit === 'pp' ? `${v.toFixed(2)}%` : st.unit === '$' ? `$${v.toFixed(0)}` : v.toFixed(2);
    // no 1955+ series for this state (e.g. goods CPI, the policy rate): no percentile line, not a bare unit
    const sub = pr === null ? null : `历史分位 ${(pr * 100).toFixed(0)}%（1955 年以来）`;
    return ui.kpi(st.name, shown, sub, pr !== null && (pr > 0.9 || pr < 0.1) ? 'is-bad' : '');
  });
  root.append(h('div', { class: 'section-title' }, `今天的宏观状态（${meta?.today?.date || ''}，单月读数）`,
    tiles.some(t => t.querySelector('.kpi-value.is-bad'))
      ? h('span', { class: 'muted', style: { fontWeight: 400, marginLeft: '8px' } }, '红色数字 = 处在 1955 年以来最高或最低的 10%') : null));
  root.append(h('div', { class: 'grid cols-4' }, tiles));
  // the real-time macro regime of the last complete month, and how far CPI y/y is from the high-inflation line
  const nc = rg?.nowcast;
  if (nc?.month && nc.name) {
    const hi = (nc.checks || []).find(c => c.id === 'high_infl');
    const d = hi && Number.isFinite(hi.distance) ? Math.abs(hi.distance).toFixed(2) : null;
    const thr = hi && Number.isFinite(hi.threshold) ? `${Number(hi.threshold.toFixed(2))}%` : '';
    const lab = (rg.definitions?.labels || []).find(l => l.id === nc.label);
    root.append(h('p', { style: { margin: '10px 0 0', fontSize: '13px' } },
      h('a', { href: '#regimes?today', style: { color: 'inherit' } },
        lab?.color ? h('span', { style: { display: 'inline-block', width: '10px', height: '10px', borderRadius: '2px', background: lab.color, marginRight: '6px', verticalAlign: '-1px' } }) : null,
        h('b', {}, '宏观体制：'), `${nc.name}（实时，${nc.month}；单边、修订后数据）`,
        d !== null ? ` · CPI 同比${hi.on ? `高出 ${thr} 门槛` : `距 ${thr} 门槛`} ${d} 个百分点` : '', ' →')));
  }
  // valuation is a state, not a factor: one line to the page that shows the evidence
  const pdNow = meta?.cma?.pd_now, pdMed = meta?.cma?.pd_median;
  if (typeof pdNow === 'number' && Number.isFinite(pdNow)) {
    root.append(h('p', { class: 'muted', style: { margin: '10px 0 0', fontSize: '13px' } },
      `美股 P/D（价格 ÷ 过去 12 个月股息）今天 ${pdNow.toFixed(0)} 倍`,
      typeof pdMed === 'number' && Number.isFinite(pdMed) ? `，1973 年以来中位数 ${pdMed.toFixed(0)} 倍` : '',
      '。市盈率、ROIC、自由现金流要不要当宏观因子、估值怎样进入模型：',
      h('a', { href: '#valuation' }, '估值与盈利 →')));
  }
  // credit and term: today's GZ spread and term premium (factor_audit.json), and whether the term-premium term is on
  const fa = await faP;
  if (!alive()) return;
  const ft = fa?.today;
  if (ft?.gz && ft?.tp10 && typeof ft.tp_term_on === 'boolean') {
    const ok = (x) => typeof x === 'number' && Number.isFinite(x);
    const rank = (p) => (ok(p) ? `第 ${Math.round(p * 100)} 百分位` : '分位未知');
    root.append(h('p', { style: { margin: '6px 0 0', fontSize: '13px' } },
      h('a', { href: '#diagnostics?audit', style: { color: 'inherit' } }, h('b', {}, '信用与期限：'),
        `GZ 信用利差 ${ok(ft.gz.value) ? ft.gz.value.toFixed(2) : '—'}（1973 年以来${rank(ft.gz.pct_since_1973)}）· `,
        `ACM 期限溢价 ${ok(ft.tp10.value) ? ft.tp10.value.toFixed(2) : '—'}（${rank(ft.tp10.pct_since_1973)}）· `,
        `期限溢价项${ft.tp_term_on ? '打开' : '关闭'}`, tpWhy(ft, meta), ' →')));
  }

  // ---- resemblance to eras (the full matching is on 历史时代画像, #eras?today)
  root.append(h('div', { class: 'section-title' }, h('a', { href: '#eras?today', title: '在“历史时代画像”页查看完整的匹配' }, '今天像哪些历史时代 →')));
  const sim = eras?.today_similarity;
  if (sim?.weights) {
    const entries = Object.entries(sim.weights).sort((a, b) => b[1] - a[1]);
    const box = ui.chartBox();
    const residual = (sim.residual || []).slice().sort((a, b) => Math.abs(b.gap_z) - Math.abs(a.gap_z));
    const more = h('a', { class: 'btn', href: '#eras?today' }, '查看完整匹配');
    root.append(h('div', { class: 'grid cols-2 text' },
      card({ title: '历史时代权重', actions: more,
        sub: `用熵池（Entropy Pooling）给历史月份重新加权，让加权平均匹配今天的 ${sim.views?.length || ''} 个关键状态（各取过去 12 个月均值）；有效月份数 ${sim.ens ? sim.ens.toFixed(0) : '—'}；点击一个时代查看它的画像` },
        box, sim.interpretation ? h('p', { class: 'muted', style: { margin: '8px 0 0' } }, sim.interpretation) : null),
      card({ title: '历史拼不出来的部分',
        sub: `匹配后仍然偏离的状态：差距越大，越说明今天是“新形势”。这里的“今天”是过去 12 个月${(sim.window || []).length === 2 ? `（${sim.window.join(' 至 ')}）` : ''}的均值，所以和上面的单月读数不同` },
        ui.table([
          { key: 'state', label: '状态', fmt: (v, r) => r.name || ui.stateName(String(v).replace(/^s_/, '')) },
          { key: 'today', label: '今天（近 12 个月均值）', fmt: (v) => ui.fmtNum(v, 2) },
          { key: 'matched', label: '历史拼出的', fmt: (v) => ui.fmtNum(v, 2) },
          { key: 'gap_z', label: '差距（标准差）', fmt: (v) => ui.fmtNum(v, 2), cls: (v) => (Math.abs(v) > 1 ? 'is-bad' : '') },
        ], residual, { stack: true }))));
    const eraChart = charts.bars(box, { x: entries.map(([e]) => ui.eraName(e)), series: [{ name: '权重', data: entries.map(([, w]) => +w.toFixed(4)) }], horizontal: true, fmt: 'pct', legend: false });
    eraChart.on('click', (p) => { const e = entries[p.dataIndex]?.[0]; if (e) location.hash = `eras?${e}`; });
  } else {
    root.append(ui.notice('还没有时代画像结果。', 'info'));
  }

  // ---- scenarios
  root.append(h('div', { class: 'section-title' }, '从今天出发的情景'));
  const list = scen?.scenarios || [];
  if (list.length) {
    // one default basis on every page: the hand-set probability (U-9); the tilt by today's state is a comparison
    const sorted = list.slice().sort((a, b) => (b.prior_prob ?? 0) - (a.prior_prob ?? 0) || (b.prob ?? 0) - (a.prob ?? 0));
    const box = ui.chartBox('tall');
    root.append(h('div', { class: 'grid cols-2 text' },
      card({ title: '情景概率', sub: '蓝条是手设的主观先验（默认口径：下面的情景加权和“未来一年”页都用它），灰条是按今天状态倾斜后的概率（主观先验 × 与今天状态的吻合度，只作对照：倾斜还没有通过预注册的检验）。点击条形查看该情景。' }, box),
      card({ title: '情景一览', sub: '美股大盘 15 年年化回报中位数（名义 / 实际）与平均通胀；点击查看详情' },
        ui.table([
          { key: 'name', label: '情景', wrap: true },
          { key: 'prior_prob', label: '手设概率', fmt: (v) => ui.fmtPct(v, 1) },
          { key: 'prob', label: '倾斜后（对照）', fmt: (v) => ui.fmtPct(v, 1) },
          { value: (r) => r.headline?.us_equity_cagr_p50, label: '美股名义', fmt: (v) => ui.fmtPct(v, 1), cls: (v) => ui.signCls(v) },
          { value: (r) => r.headline?.us_equity_real_cagr_p50, label: '美股实际', fmt: (v) => ui.fmtPct(v, 1), cls: (v) => ui.signCls(v) },
          { value: (r) => r.headline?.cpi_avg, label: '平均通胀', fmt: (v) => ui.fmtPP(v, 1) },
        ], sorted, { onRowClick: (r) => { location.hash = `scenarios?${r.id}`; }, stack: true }))));
    const probChart = charts.bars(box, {
      x: sorted.map(s => s.name),
      series: [{ name: '手设概率（默认）', data: sorted.map(s => +(s.prior_prob ?? 0).toFixed(4)) },
        { name: '按今天状态倾斜（对照）', data: sorted.map(s => +(s.prob ?? 0).toFixed(4)), color: '#adb5bd' }],
      horizontal: true, fmt: 'pct',
    });
    probChart.on('click', (p) => { const s = sorted[p.dataIndex]; if (s) location.hash = `scenarios?${s.id}`; });
  } else {
    root.append(ui.notice('还没有情景结果。', 'info'));
  }

  // ---- the next four quarters: the scenario mixture (default engine, default probabilities). The default engine
  // follows the live switch (as the forecast tab does), not the copy the build wrote into forecast_mix.json.
  const mixKey = quarterlyDefault(sw) && mix?.engines?.quarterly ? 'quarterly' : mix?.engines?.monthly ? 'monthly' : null;
  const mixEng = mixKey ? mix.engines[mixKey] : null;
  if (mixEng && Array.isArray(mix.grid) && (mix.quarters || []).length) {
    const P = mix.grid, k = mix.quarters.length - 1;
    const probs = mix.probabilities?.sets?.[mix.probabilities?.default]?.probs || {};
    const excluded = new Set(mixEng.excluded || []);
    const nUsed = (mix.scenarios || []).filter(s => !excluded.has(s.id) && Object.values(mixEng.scenarios?.[s.id]?.assets || {})
      .some(x => Array.isArray(x?.grid?.[k]))).length;
    const at = (a) => {
      const ids = (mix.scenarios || []).map(s => s.id).filter(i => !excluded.has(i) && Array.isArray(mixEng.scenarios?.[i]?.assets?.[a]?.grid?.[k]) && probs[i] > 0);
      const grids = ids.map(i => mixEng.scenarios[i].assets[a].grid[k]);
      const w = ids.map(i => probs[i]);
      const W = w.reduce((t, x) => t + x, 0);
      if (!grids.length || !(W > 0)) return null;
      return { p10: mixQuantile(grids, w, P, 0.1), p50: mixQuantile(grids, w, P, 0.5), p90: mixQuantile(grids, w, P, 0.9),
        loss: ids.reduce((t, i, j) => t + (w[j] / W) * (mixEng.scenarios[i].assets[a].prob_loss?.[k] ?? 0), 0) };
    };
    const keys = ['us_equity', 'nasdaq', 'energy', 'industrials', 'staples', 'real_estate', 'gold', 'crude_oil', 'treasury_10y', 'cash'];
    const rows = keys.filter(a => (mix.assets || []).includes(a)).map(a => ({ a, m: at(a), ref: mixEng.reference?.assets?.[a] }));
    root.append(h('div', { class: 'section-title' }, `未来一年：情景条件模拟（到 ${mix.quarters[k]} 末，按情景概率加权）`));
    root.append(card({
      title: '四个季度后的累计总回报：情景加权',
      sub: `这不是预测：${nUsed} 个情景在未来 ${mix.quarters.length} 个季度的模拟，按手设的情景概率加权；预测只在情景概率里。`
        + `引擎：${mixKey === 'quarterly' && !quarterlyDefault(sw) ? '季度模型（试运行）' : mixKey === 'quarterly' ? '季度模型' : (mixEng.name || '月度引擎')}。点击查看每个情景的季度中位线、它要你相信什么，以及改概率。`,
    }, ui.table([
      { key: 'a', label: '资产', fmt: (v) => ui.assetName(v) },
      { value: (r) => r.m?.p50, label: '情景加权 中位', fmt: (v) => ui.fmtPct(v, 1), cls: (v) => ui.signCls(v) },
      { value: (r) => r.m, label: '10%–90%', fmt: (b) => (b ? `${ui.fmtPct(b.p10, 0)} ~ ${ui.fmtPct(b.p90, 0)}` : '—') },
      { value: (r) => r.m?.loss, label: 'P(亏损)', fmt: (v) => ui.fmtPct(v, 0) },
      { value: (r) => r.ref?.p50?.[k], label: '无情景基准 中位', fmt: (v) => ui.fmtPct(v, 1), cls: (v) => ui.signCls(v) },
    ], rows, { onRowClick: () => { location.hash = 'forecast'; }, stack: true })));
  }

  // ---- next 12 months: the analog method's one-year record (历史回放), when there is no scenario mixture
  const live = fc?.live?.assets;
  if (live && !mixEng) {
    const an = fc.scores?.all?.analog || {};
    root.append(h('div', { class: 'section-title' }, `相似起点法：从 ${fc.live.origin} 起的 12 个月（回放口径）`));
    const keys = ['us_equity', 'nasdaq', 'energy', 'industrials', 'staples', 'real_estate', 'gold', 'crude_oil', 'treasury_10y', 'cash'];
    root.append(card({
      title: '下一年的回报区间（相似起点法与情景加权）',
      sub: `相似起点法与情景加权的中位数和 90% 区间（名义总回报）；P(亏损) 是相似起点法下一年亏损的概率。过去 ${(fc.origins || []).length} 次一年期预测的 90% 区间覆盖率 `
        + `${ui.fmtPct(an.risk_avg?.coverage90, 0)}，跨资产排序相关 ${an.rank_ic?.mean?.toFixed(2) ?? '—'}；点击查看成绩单和偏差归因`,
    }, ui.table([
      { key: 'a', label: '资产', fmt: (v) => ui.assetName(v) },
      { value: (r) => live[r.a]?.analog?.p50, label: '相似起点 中位', fmt: (v) => ui.fmtPct(v, 1), cls: (v) => ui.signCls(v) },
      { value: (r) => live[r.a]?.analog, label: '90% 区间', fmt: (b) => (b ? `${ui.fmtPct(b.p5, 0)} ~ ${ui.fmtPct(b.p95, 0)}` : '—') },
      { value: (r) => live[r.a]?.scenarios?.p50, label: '情景加权 中位', fmt: (v) => ui.fmtPct(v, 1), cls: (v) => ui.signCls(v) },
      { value: (r) => live[r.a]?.scenarios, label: '情景加权 90% 区间', fmt: (b) => (b ? `${ui.fmtPct(b.p5, 0)} ~ ${ui.fmtPct(b.p95, 0)}` : '—') },
      { value: (r) => live[r.a]?.analog?.prob_loss, label: 'P(亏损)', fmt: (v) => ui.fmtPct(v, 0) },
    ], keys.map(a => ({ a })), { onRowClick: () => { location.hash = 'forecast'; }, stack: true })));
  }

  // ---- navigation
  root.append(h('div', { class: 'section-title' }, '从哪里开始'));
  const links = [
    ['eras', '历史时代画像', '每个时代的因子指纹、资产表现和传导特征'],
    ['regimes', '宏观体制', '你问的通胀、通缩、滞胀、萧条在数据里对应什么，今天处在哪一种'],
    ['causality', '因果传导', '石油、需求、货币、金融冲击和劳动收入冲击怎么传导'],
    ['forecast', '未来一年：情景条件模拟', '情景在未来四个季度的模拟按概率加权：模拟，不是预测；相似起点法的一年期回放在页尾'],
    ['backtest', '回测：模拟 vs 实际', '如果当时知道宏观情景，引擎能不能复现资产走势'],
    ['diagnostics', '诊断与迭代', '哪里不准、为什么、改了什么、在新时段上效果如何'],
    ['valuation', '估值与盈利', '市盈率、ROIC、自由现金流要不要当宏观因子：回测证据与结论'],
    ['scenarios', '未来情景', [nScen(), ' 个情景的 K 线、扇形图和因果链']],
    ['lab', '情景实验室', '写一段话，大模型翻译成季度情景表（每个值带原文引文），模型算出结果'],
    ['portfolio', '组合评估', '组合在所有情景下的收益、风险和夏普'],
    ['withdrawal', '提取与破产', '三种提取方式的破产概率与安全提取率'],
  ];
  root.append(h('div', { class: 'grid cols-4' }, links.map(([id, t, d]) =>
    h('a', { href: `#${id}`, class: 'card', style: { color: 'inherit', textDecoration: 'none' } },
      h('div', { class: 'card-title' }, t), h('div', { class: 'card-sub' }, d)))));
  paintCount();
}

// Why the term-premium term is on or off today: the reading that decided it (factor_audit.json today: the state
// `lag` months before the latest month) and the switch's own rule (meta.json, the selected model's interaction
// factor; the audit file's copy for an older meta).
function tpWhy(ft, meta) {
  const ok = (x) => typeof x === 'number' && Number.isFinite(x);
  const F = Object.fromEntries((meta?.factors || []).map(f => [f.id, f]));
  const rule = (meta?.model?.spec?.factors || []).map(id => F[id]?.interaction).find(x => x && ok(x.above)) || ft.tp_term_rule;
  const lag = Number(rule?.lag ?? 1);
  const when = lag === 1 ? '上月' : `${lag} 个月前`;
  const name = { core_yoy: '核心 CPI 同比', cpi_yoy: 'CPI 同比' }[rule?.state] || '核心 CPI 同比';
  const state = /^[A-Za-z]/.test(name) ? ` ${name}` : name;      // 「上月 CPI 同比」, 「上月核心 CPI 同比」
  const thr = ok(Number(rule?.above)) ? `${String(Number(rule.above))}%` : '门槛';
  if (ok(ft.core_yoy_prev)) {
    return `（${when}${state} ${ft.core_yoy_prev.toFixed(2)}%，${ft.core_yoy_prev_month || ''}；${when}高于 ${thr} 时打开）`;
  }
  return ok(ft.core_yoy) ? `（最新${state} ${ft.core_yoy.toFixed(2)}%，${ft.core_yoy_month || ''}；${when}高于 ${thr} 时打开）` : '';
}
