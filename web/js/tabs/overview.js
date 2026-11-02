// Overview: today's macro state, which historical eras it resembles, and the scenario set.
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
    '从历史月度数据出发：先看清每个时代的宏观“指纹”和因果传导，再用回测检验引擎能否复现实际走势，',
    '最后从今天出发生成 ', nScen(), ' 个情景，评估组合和提取策略。所有情景数字都来自引擎，LLM 只负责把文字翻译成约束。'));

  const [eras, scen, fc] = await Promise.all([
    api.get('eras').catch(() => null),
    api.get('scenarios').catch(() => null),
    api.get('forecast').catch(() => null),
  ]);
  if (!alive()) return;                  // the user moved on while the results loaded
  const paintCount = () => {
    const n = (scen?.scenarios || []).length;
    if (n) for (const el of root.querySelectorAll('.ov-nscen')) el.textContent = String(n);
  };
  paintCount();
  // the missing-factor audit (optional; skipped when /api/meta's `results` says it is absent): one line below
  const faP = Array.isArray(meta?.results) && !meta.results.includes('factor_audit') ? Promise.resolve(null)
    : api.get('factor_audit').catch(() => null);

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
        `期限溢价项${ft.tp_term_on ? '打开' : '关闭'}`,
        ok(ft.core_yoy) ? `（核心 CPI 同比 ${ft.core_yoy.toFixed(2)}%，${ft.core_yoy_month || ''}；上月高于 4% 时打开）` : '', ' →')));
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
    root.append(ui.notice('还没有时代画像结果（eras.json）。', 'info'));
  }

  // ---- scenarios
  root.append(h('div', { class: 'section-title' }, '从今天出发的情景'));
  const list = scen?.scenarios || [];
  if (list.length) {
    const sorted = list.slice().sort((a, b) => (b.prob ?? 0) - (a.prob ?? 0));
    const box = ui.chartBox('tall');
    root.append(h('div', { class: 'grid cols-2 text' },
      card({ title: '情景概率', sub: `${scen.probability_method || '主观先验 × 与当前状态的相似度'}（点击条形查看该情景）` }, box),
      card({ title: '情景一览', sub: '美股大盘 15 年年化回报中位数（名义 / 实际）与平均通胀；点击查看详情' },
        ui.table([
          { key: 'name', label: '情景', wrap: true },
          { key: 'prob', label: '概率', fmt: (v) => ui.fmtPct(v, 1) },
          { value: (r) => r.headline?.us_equity_cagr_p50, label: '美股名义', fmt: (v) => ui.fmtPct(v, 1), cls: (v) => ui.signCls(v) },
          { value: (r) => r.headline?.us_equity_real_cagr_p50, label: '美股实际', fmt: (v) => ui.fmtPct(v, 1), cls: (v) => ui.signCls(v) },
          { value: (r) => r.headline?.cpi_avg, label: '平均通胀', fmt: (v) => ui.fmtPP(v, 1) },
        ], sorted, { onRowClick: (r) => { location.hash = `scenarios?${r.id}`; }, stack: true }))));
    const probChart = charts.bars(box, {
      x: sorted.map(s => s.name),
      series: [{ name: '后验概率', data: sorted.map(s => +(s.prob ?? 0).toFixed(4)) },
        { name: '主观先验', data: sorted.map(s => +(s.prior_prob ?? 0).toFixed(4)), color: '#adb5bd' }],
      horizontal: true, fmt: 'pct',
    });
    probChart.on('click', (p) => { const s = sorted[p.dataIndex]; if (s) location.hash = `scenarios?${s.id}`; });
  } else {
    root.append(ui.notice('还没有情景结果（scenarios.json）。', 'info'));
  }

  // ---- next 12 months
  const live = fc?.live?.assets;
  if (live) {
    const an = fc.scores?.all?.analog || {};
    root.append(h('div', { class: 'section-title' }, `未来 12 个月（从 ${fc.live.origin} 起的一年期预测）`));
    const keys = ['us_equity', 'nasdaq', 'energy', 'industrials', 'staples', 'real_estate', 'gold', 'crude_oil', 'treasury_10y', 'cash'];
    root.append(card({
      title: '下一年的回报分布',
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
    ['causality', '因果传导', '石油、需求、货币、金融冲击和劳动收入冲击怎么传导'],
    ['forecast', '一年期预测', '每年只用当时的数据预测下一年：准不准、为什么不准'],
    ['backtest', '回测：模拟 vs 实际', '如果当时知道宏观情景，引擎能不能复现资产走势'],
    ['diagnostics', '诊断与迭代', '哪里不准、为什么、改了什么、在新时段上效果如何'],
    ['valuation', '估值与盈利', '市盈率、ROIC、自由现金流要不要当宏观因子：回测证据与结论'],
    ['scenarios', '未来情景', [nScen(), ' 个情景的 K 线、扇形图和因果链']],
    ['lab', '情景实验室', '写一段话，LLM 翻译成冲击，引擎算出 K 线'],
    ['portfolio', '组合评估', '组合在所有情景下的收益、风险和夏普'],
    ['withdrawal', '提取与破产', '三种提取方式的破产概率与安全提取率'],
  ];
  root.append(h('div', { class: 'grid cols-4' }, links.map(([id, t, d]) =>
    h('a', { href: `#${id}`, class: 'card', style: { color: 'inherit', textDecoration: 'none' } },
      h('div', { class: 'card-title' }, t), h('div', { class: 'card-sub' }, d)))));
  paintCount();
}
