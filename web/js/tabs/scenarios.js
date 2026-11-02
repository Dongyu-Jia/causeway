// Future scenarios: the built-in library ranked by posterior probability, a cross-scenario
// comparison of 15-year median returns, and the full detail of the selected scenario
// (shared renderer in _scenario_detail.js), plus each scenario's year-1 macro regime mix from the
// index. Deep link: #scenarios?<id> (built-in or custom).
import {
  renderScenarioDetail, disposeCharts, ensureStyle, ensureDetailStyle, shortName, errorText, isNum, pct, pp, esc,
  REGIME_LABELS, regimeBars,
} from './_scenario_detail.js';

const CSS = `
.scenarios-list { grid-template-columns: repeat(auto-fill, minmax(255px, 1fr)); }
.scenarios-list .scen-item { display: flex; flex-direction: column; gap: 5px; }
.scenarios-list .scen-item:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.scenarios-item-top { display: flex; gap: 8px; align-items: baseline; }
.scenarios-rank { color: var(--ink-3); font-size: 12px; font-variant-numeric: tabular-nums; min-width: 22px; }
.scenarios-probrow { display: grid; grid-template-columns: 30px 1fr 44px; gap: 6px; align-items: center; font-size: 11px; color: var(--ink-3); }
.scenarios-probrow .bar-inline > span.prior { background: #adb5bd; }
.scenarios-probrow b { color: var(--ink); font-variant-numeric: tabular-nums; text-align: right; font-weight: 600; }
.scenarios-line { font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.scenarios-tags { display: flex; gap: 4px; flex-wrap: wrap; }
.scenarios-tags .chip { font-size: 11px; padding: 0 6px; }
.scenarios-groupname { font-size: 12px; font-weight: 600; color: var(--ink-2); margin: 14px 0 8px; }
.scenarios-switch { position: sticky; top: 63px; z-index: 4; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  background: rgba(244, 245, 248, .96); padding: 8px 0 10px; margin-bottom: 10px; border-bottom: 1px solid var(--line); }
.scenarios-switch select { flex: 1 1 220px; max-width: 460px; min-width: 0; }
.scenarios-switch .btn, .scenarios-switch a { white-space: nowrap; }
.scenarios-switch .muted { font-size: 12px; }
.scenarios-anchor { scroll-margin-top: 72px; }
.scenarios-scroll { overflow-x: auto; overflow-y: hidden; }
@media (max-width: 760px) { .scenarios-switch { top: 55px; } }      /* the phone top bar is 56 px tall */
`;

// The detail's picks (asset, K-line period, fan, correlation view) live for the session, so the
// K-line colour toggle (which re-renders the page) or a visit to another tab keeps them.
const DETAIL_STATE = { asset: 'us_equity', kline: 'yearly', fan: 'nominal', corr: 'post' };

const MODES = [
  { value: 'real', label: '实际年化', key: 'real_cagr_p50', fmt: (x) => pct(x, 1),
    what: '15 年实际（扣除通胀）年化回报的中位数' },
  { value: 'nominal', label: '名义年化', key: 'cagr_p50', fmt: (x) => pct(x, 1),
    what: '15 年名义年化回报的中位数' },
  { value: 'dd', label: '最大回撤', key: 'max_dd_p50', fmt: (x) => pct(x, 0),
    what: '15 年内最大回撤的中位数' },
];

// Scenarios x assets heatmap; colours follow the K-line convention (up colour = positive).
function compareChart(el, charts, { rows, cols, values, fmt, tip, onRow }) {
  el.style.height = `${rows.length * 30 + 76}px`;
  const chart = charts.mount(el);
  const data = [];
  let bound = 0;
  values.forEach((row, r) => row.forEach((v, c) => {
    data.push([c, r, isNum(v) ? v : '-']);
    if (isNum(v)) bound = Math.max(bound, Math.abs(v));
  }));
  const width = el.clientWidth || 900;
  const cell = (width - 190) / Math.max(cols.length, 1);         // rough cell width after the row labels
  const tiny = cell < 34;                                         // drop the % sign and shrink the font
  // flat labels while they fit, 50° on mid-size cells, upright on narrow ones (50° labels collide there)
  const rotate = cell >= 62 ? 0 : cell >= 40 ? 50 : 90;
  chart.setOption({
    tooltip: { confine: true, formatter: (p) => tip(p.data[1], p.data[0], p.data[2]) },
    grid: { left: 8, right: 12, top: 8, bottom: 8, containLabel: true },
    xAxis: { type: 'category', data: cols, position: 'top', axisTick: { show: false }, axisLine: { show: false },
      axisLabel: { color: '#4a5468', fontSize: 11, interval: 0, rotate } },
    yAxis: { type: 'category', data: rows.map(r => r.label), inverse: true, axisTick: { show: false }, axisLine: { show: false },
      axisLabel: { color: '#4a5468', fontSize: 12, interval: 0,
        formatter: (v, i) => (rows[i]?.summary ? `{b|${v}}` : v), rich: { b: { fontWeight: 600, color: '#1d2433', fontSize: 12 } } } },
    visualMap: { min: -(bound || 1), max: bound || 1, show: false, calculable: false, dimension: 2,
      inRange: { color: charts.klineDiverging() } },
    series: [{ type: 'heatmap', data, itemStyle: { borderColor: '#fff', borderWidth: 1 },
      label: { show: true, fontSize: tiny ? 8 : 10, color: '#1d2433',
        formatter: (p) => (isNum(p.data[2]) ? (tiny ? fmt(p.data[2]).replace('%', '') : fmt(p.data[2])) : '') },
      emphasis: { itemStyle: { borderColor: '#1d2433', borderWidth: 1 } } }],
  });
  chart.on('click', (p) => { const row = rows[p.data?.[1]]; if (row?.id) onRow(row.id); });
  return chart;
}

export async function render(root, ctx) {
  const { api, charts, ui, param } = ctx;
  const alive = ctx.alive || (() => root.isConnected);
  const { h, card } = ui;
  ensureDetailStyle();
  ensureStyle('scenarios', CSS);

  const timeout = (ms) => new Promise(resolve => setTimeout(() => resolve(null), ms));
  const [index, custom] = await Promise.all([
    api.get('scenarios'),                                          // 404 -> app.js shows the build hint
    Promise.race([api.get('scenarios/custom').catch(() => null), timeout(1500)]),
  ]);
  if (!alive()) return null;             // the user moved on while the index loaded
  const list = (index.scenarios || []).slice().sort((a, b) => (b.prob ?? 0) - (a.prob ?? 0));
  const customs = (custom?.scenarios || []).filter(s => s && s.id);
  const builtinIds = new Set(list.map(s => s.id));

  root.append(h('p', { class: 'lead' },
    `从今天（${index.start || '—'}）出发的 ${list.length} 个情景。每个情景是一组分阶段的“视图”：它只改变每个真实历史月份被抽中的概率，`,
    '再按今天的利率、估值和股息率把抽到的月份拼成 15 年路径。K 线、扇形图和年化回报全部由引擎计算；',
    '概率是透明的启发式（主观先验 × 与今天状态的吻合度），不是预测。'));
  if (!list.length) {
    root.append(ui.notice('情景库是空的。在项目根目录运行 python -m scenariolab.build。', 'info'));
    return;
  }

  // ---- scenario list
  const maxProb = Math.max(...list.map(s => Math.max(s.prob ?? 0, s.prior_prob ?? 0)), 1e-9);
  const width = (x) => `${Math.max(1.5, Math.min(100, ((x || 0) / maxProb) * 100)).toFixed(1)}%`;
  const items = new Map();          // id -> list card (built-in and custom)
  const keyOpen = (id) => (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(id, { scroll: true }); } };
  const builtinItems = list.map((s, i) => {
    const hl = s.headline || {};
    const el = h('div', { class: 'scen-item', tabindex: '0', role: 'button', title: s.summary || s.name,
      onclick: () => select(s.id, { scroll: true }), onkeydown: keyOpen(s.id) },
      h('div', { class: 'scenarios-item-top' }, h('span', { class: 'scenarios-rank' }, `#${i + 1}`), h('span', { class: 't' }, s.name)),
      h('div', { class: 'scenarios-probrow' }, '后验', h('div', { class: 'bar-inline' }, h('span', { style: { width: width(s.prob) } })), h('b', {}, pct(s.prob, 1))),
      h('div', { class: 'scenarios-probrow' }, '先验', h('div', { class: 'bar-inline' }, h('span', { class: 'prior', style: { width: width(s.prior_prob) } })), h('b', {}, pct(s.prior_prob, 1))),
      h('div', { class: 'scenarios-line' }, '美股年化中位数：名义 ', h('b', { class: ui.signCls(hl.us_equity_cagr_p50) }, pct(hl.us_equity_cagr_p50, 1)),
        ' · 实际 ', h('b', { class: ui.signCls(hl.us_equity_real_cagr_p50) }, pct(hl.us_equity_real_cagr_p50, 1))),
      h('div', { class: 'scenarios-line' }, `平均 CPI 同比 ${pp(hl.cpi_avg, 2)} · 期末利率 ${pp(hl.ff_end_p50, 2)}`),
      isNum(hl.pd_end_p50) ? h('div', { class: 'scenarios-line', title: '每条路径记录的 P/D（价格 ÷ 过去 12 个月股息）的中位数；只记录，不做估值回归' },
        `15 年后 P/D 中位数 ${hl.pd_end_p50.toFixed(0)} 倍${isNum(index.cma?.pd_now) ? `（今天 ${index.cma.pd_now.toFixed(0)}）` : ''}`) : null,
      h('div', { class: 'scenarios-tags' }, (s.tags || []).slice(0, 4).map(t => ui.chip(t)), s.calibrated ? ui.chip('收入分配计算器', 'good') : null));
    items.set(s.id, el);
    return el;
  });
  const customItems = customs.map(s => {
    const el = h('div', { class: 'scen-item', tabindex: '0', role: 'button', title: s.summary || s.name,
      onclick: () => select(s.id, { scroll: true }), onkeydown: keyOpen(s.id) },
      h('div', { class: 't' }, s.name || s.id),
      h('div', { class: 's' }, `${s.source === 'llm' ? 'LLM 翻译' : '手动编辑'} · ${s.created || '—'} · 先验 ${pct(s.prior_prob, 0)}`),
      s.summary ? h('div', { class: 'scenarios-line' }, s.summary) : null);
    items.set(s.id, el);
    return el;
  });

  root.append(h('div', { class: 'section-title' }, '情景一览（按后验概率排序）'));
  root.append(card({ title: '情景库', sub: '蓝条是后验概率、灰条是主观先验，长度按最大值缩放；点击卡片查看下方详情。' },
    h('div', { class: 'scen-list scenarios-list' }, builtinItems),
    customItems.length ? h('div', {}, h('div', { class: 'scenarios-groupname' }, `自定义情景（来自情景实验室，不参与概率排序）· ${customItems.length} 个`),
      h('div', { class: 'scen-list scenarios-list' }, customItems)) : null,
    index.probability_method ? h('details', { class: 'scenarios-more' }, h('summary', {}, '概率是怎么算的'),
      h('p', { class: 'muted', style: { margin: 0 } }, index.probability_method)) : null));

  // ---- cross-scenario comparison: per-asset stats from the index (scenarios.json "assets"); an
  // index built before that field existed falls back to loading every detail in the background
  let mode = 'real';
  // 13 x 13 cells stay legible down to ~560 px; a narrower card scrolls the matrix sideways
  const cmpBox = h('div', { class: 'chart', style: { minWidth: '560px' } });
  cmpBox.style.height = `${(list.length + 4) * 30 + 76}px`;
  const cmpNote = h('p', { class: 'scenarios-note' });
  const cmpTitle = h('span', {}, '各情景 × 资产：15 年实际年化回报中位数');
  root.append(h('div', { class: 'section-title' }, '跨情景对比'));
  root.append(card({ title: cmpTitle, cls: 'scenarios-card',
    sub: '每格是一个情景下一类资产的中位数结果，颜色按 K 线配色（涨色为正）：同一列横向比较，就能看出哪类资产依赖哪种宏观传导；底部三行是按后验概率加权、情景间最差和最好。',
    actions: ui.seg(MODES.map(m => ({ value: m.value, label: m.label })), mode, (v) => { mode = v; drawCompare(); }) },
  h('div', { class: 'scenarios-scroll' }, cmpBox), cmpNote));
  cmpBox.append(ui.loading('正在载入全部情景的结果…'));

  // ---- each scenario's year-1 macro regime mix: index fields only (scenarios[].regimes, regimes)
  const rgList = list.filter(s => s.regimes?.year1);
  const rgIdx = index.regimes || {};
  const rgBox = h('div', { class: 'chart' });
  const rgNote = h('p', { class: 'scenarios-note' });
  const nRef = rgIdx.reference?.n_paths;
  root.append(h('div', { style: { height: '16px' } }), card({ title: '各情景第 1 年的宏观体制', cls: 'scenarios-card',
    sub: `每一行是一个情景第 1 年模拟路径的体制构成（事后标签，和历史月份用同一套规则）；“按概率加权”按后验概率平均，`
      + `“引擎无条件”是不加任何视图的引擎参考运行${isNum(nRef) ? `（${nRef} 条路径）` : ''}。悬停查看第 1 年出现收缩的路径占比，点击条形查看该情景。`,
    actions: h('a', { href: '#regimes' }, '宏观体制 →') },
  rgList.length ? [rgBox, rgNote]
    : ui.notice('情景库里没有宏观体制读数（由旧版本生成）：运行 python -m scenariolab.build --only scenarios regimes 后可见。', 'info')));
  if (rgList.length) {
    const rows = rgList.map(s => ({ label: `${shortName(s.name)}  ${pct(s.prob, 1)}`, mix: s.regimes.year1, p: s.regimes.p_contraction_year1, s }));
    if (rgIdx.year1_weighted) rows.push({ label: '按概率加权', mix: rgIdx.year1_weighted, p: rgIdx.p_contraction_year1_weighted });
    if (rgIdx.reference?.year1) rows.push({ label: '引擎无条件', mix: rgIdx.reference.year1, p: rgIdx.reference.p_contraction_year1 });
    const rgChart = regimeBars(rgBox, { rows: rows.map(r => r.label), mixes: rows.map(r => r.mix), labels: REGIME_LABELS, rowH: 26,
      extra: (i) => [rows[i]?.s ? `后验概率 ${pct(rows[i].s.prob, 1)}` : '',
        isNum(rows[i]?.p) ? `第 1 年出现收缩的路径 <b>${pct(rows[i].p, 1)}</b>` : '', rows[i]?.s ? '点击查看这个情景' : ''] });
    rgChart.on('click', (p) => { const s = rows[p.dataIndex]?.s; if (s) select(s.id, { scroll: true }); });
    const hist = rgIdx.history?.p_contraction_12m_window;
    rgNote.textContent = `第 1 年出现收缩的路径：按概率加权 ${pct(rgIdx.p_contraction_year1_weighted, 1)}，引擎无条件 ${pct(rgIdx.reference?.p_contraction_year1, 1)}`
      + `${isNum(hist) ? `，历史上任意 12 个月窗口里含收缩月份的比例 ${pct(hist, 1)}` : ''}。`
      + '引擎无条件抽样的 CPI 同比本身偏高，模拟路径里“通胀”的占比因此系统性偏大，比较时请对照“引擎无条件”这一行。';
  }

  // ---- the P/D each scenario's paths imply 15 years on (recorded at κ=0, no valuation pull; the
  // evidence for that is on 估值与盈利)
  const pdList = list.filter(s => isNum(s.headline?.pd_end_p50));
  if (pdList.length) {
    const cma = index.cma || ui.META.raw?.cma || {};
    const den = pdList.reduce((t, s) => t + (s.prob ?? 0), 0);
    const pw = den > 0 ? pdList.reduce((t, s) => t + (s.prob ?? 0) * s.headline.pd_end_p50, 0) / den : null;
    const n0 = (x) => (isNum(x) ? x.toFixed(0) : '—');
    const pdBox = h('div', { class: 'chart' });
    pdBox.style.height = `${Math.max(300, pdList.length * 26 + 70)}px`;
    root.append(h('div', { style: { height: '16px' } }), card({ title: '各情景 15 年后的美股 P/D（中位数）', cls: 'scenarios-card',
      sub: `每条路径记录的 P/D（价格 ÷ 过去 12 个月股息）：κ=0，只记录、不做估值回归，不影响上面的回报。竖线是今天（${n0(cma.pd_now)} 倍）和 1973 年以来的中位数（${n0(cma.pd_median)} 倍）；`
        + `按后验概率加权约 ${n0(pw)} 倍。点击条形查看该情景。`,
      actions: h('a', { href: '#valuation?scenarios' }, '估值与盈利 →') }, pdBox));
    const pdChart = charts.bars(pdBox, { x: pdList.map(s => shortName(s.name)), horizontal: true, legend: false, fmt: (v) => Number(v).toFixed(0),
      series: [{ name: '15 年后 P/D 中位数', data: pdList.map(s => +s.headline.pd_end_p50.toFixed(2)), color: '#3b5bdb' }],
      markLine: [isNum(cma.pd_now) ? { xAxis: cma.pd_now, name: `今天 ${n0(cma.pd_now)}` } : null,
        isNum(cma.pd_median) ? { xAxis: cma.pd_median, name: `中位数 ${n0(cma.pd_median)}` } : null].filter(Boolean) });
    pdChart.setOption({ tooltip: { formatter: (ps) => {
      const s = pdList[(Array.isArray(ps) ? ps[0] : ps)?.dataIndex];
      return s ? `<b>${esc(s.name)}</b><br/>后验概率 ${pct(s.prob, 1)}<br/>15 年后 P/D 中位数 <b>${s.headline.pd_end_p50.toFixed(1)} 倍</b><br/>点击查看这个情景` : '';
    } } });
    pdChart.on('click', (p) => { const s = pdList[p.dataIndex]; if (s) select(s.id, { scroll: true }); });
  }
  let details = null;          // id -> {asset: {cagr_p50, real_cagr_p50, max_dd_p50, ...}}, once known
  // label rotation and cell font depend on the width: redraw when it changes noticeably
  let drawnWidth = 0;
  const widthWatch = new ResizeObserver(() => {
    if (!cmpBox.isConnected) { widthWatch.disconnect(); return; }
    if (!details || !details.size || !drawnWidth || Math.abs(cmpBox.clientWidth - drawnWidth) < 40) return;
    requestAnimationFrame(() => { try { drawCompare(); } catch (err) { console.error(err); } });
  });
  widthWatch.observe(cmpBox);
  ctx.signal?.addEventListener('abort', () => widthWatch.disconnect());

  function drawCompare() {
    if (!details) return;
    drawnWidth = cmpBox.clientWidth;
    const m = MODES.find(x => x.value === mode) || MODES[0];
    cmpTitle.textContent = `各情景 × 资产：${m.what}`;
    const loaded = list.filter(s => details.has(s.id));
    const assetIds = [];
    for (const s of loaded) for (const a of Object.keys(details.get(s.id) || {})) if (!assetIds.includes(a)) assetIds.push(a);
    const val = (s, a) => details.get(s.id)?.[a]?.[m.key];
    const values = loaded.map(s => assetIds.map(a => val(s, a)));
    const weighted = [], worst = [], best = [];
    assetIds.forEach((a, c) => {
      let num = 0, den = 0, lo = null, hi = null;
      loaded.forEach((s, r) => {
        const v = values[r][c];
        if (!isNum(v)) return;
        const p = s.prob ?? 0;
        num += p * v; den += p;
        if (!lo || v < lo.v) lo = { v, s };
        if (!hi || v > hi.v) hi = { v, s };
      });
      weighted.push(den > 0 ? num / den : null);
      worst.push(lo); best.push(hi);
    });
    const rows = [
      ...loaded.map(s => ({ id: s.id, label: `${shortName(s.name)}  ${pct(s.prob, 1)}`, s })),
      { label: '' },
      { label: '概率加权', summary: true }, { label: '情景间最差', summary: true }, { label: '情景间最好', summary: true },
    ];
    const grid = [...values, assetIds.map(() => null), weighted, worst.map(x => x?.v ?? null), best.map(x => x?.v ?? null)];
    const n = loaded.length;
    compareChart(cmpBox, charts, {
      rows, cols: assetIds.map(a => ui.assetName(a)), values: grid, fmt: m.fmt,
      onRow: (id) => select(id, { scroll: true }),
      tip: (r, c, v) => {
        const asset = esc(ui.assetName(assetIds[c]));
        const shown = isNum(v) ? m.fmt(v) : '—';
        if (r < n) return `<b>${esc(loaded[r].name)}</b><br/>后验概率 ${pct(loaded[r].prob, 1)}<br/>${asset}：<b>${shown}</b><br/>点击查看这个情景`;
        if (r === n + 1) return `<b>概率加权</b>：各情景中位数按后验概率的加权平均<br/>${asset}：<b>${shown}</b>`;
        const pick = r === n + 2 ? worst[c] : r === n + 3 ? best[c] : null;
        return pick ? `<b>${r === n + 2 ? '情景间最差' : '情景间最好'}</b><br/>${asset}：<b>${shown}</b>（${esc(pick.s.name)}）` : '';
      },
    });
    const missing = list.length - loaded.length;
    const tiny = ((cmpBox.clientWidth || 900) - 190) / Math.max(assetIds.length, 1) < 34;
    cmpNote.textContent = `${loaded.length} 个情景，概率合计 ${pct(loaded.reduce((t, s) => t + (s.prob ?? 0), 0), 0)}。` +
      (tiny ? '格子较窄，数值省略了 % 号（单位均为 %）。' : '') +
      '注意“概率加权”是各情景中位数的加权平均，不是混合分布的中位数；完整的组合层面加权（含分位数、夏普、回撤）见“组合评估”页。' +
      (missing ? ` 有 ${missing} 个情景的详情没有载入，未计入。` : '');
  }

  // ---- detail of the selected scenario, with a sticky switcher
  const anchor = h('div', { class: 'section-title scenarios-anchor' }, '情景详情');
  const picker = h('select', { 'aria-label': '选择情景', onchange: (e) => select(e.target.value) },
    h('optgroup', { label: '内置情景（按后验概率）' }, list.map((s, i) => h('option', { value: s.id }, `#${i + 1} ${s.name}（${pct(s.prob, 1)}）`))),
    customs.length ? h('optgroup', { label: '自定义情景' }, customs.map(s => h('option', { value: s.id }, s.name || s.id))) : null);
  const order = [...list.map(s => s.id), ...customs.map(s => s.id)];
  const step = (k) => { const i = order.indexOf(current); select(order[(i + k + order.length) % order.length]); };
  const switcher = h('div', { class: 'scenarios-switch' },
    h('span', { class: 'muted' }, '当前情景'), picker,
    h('button', { class: 'btn', onclick: () => step(-1), title: '上一个情景' }, '‹ 上一个'),
    h('button', { class: 'btn', onclick: () => step(1), title: '下一个情景' }, '下一个 ›'),
    h('a', { href: '#scenarios', onclick: (e) => { e.preventDefault(); root.scrollIntoView({ behavior: 'smooth', block: 'start' }); } }, '↑ 回到情景列表'));
  // min-height until the first detail arrives: a deep link scrolls to the anchor before the
  // detail has loaded, and a page that ends right below it would stop the scroll short
  const detailRoot = h('div', { style: { minHeight: '100vh' } });
  root.append(anchor, switcher, detailRoot);

  const state = DETAIL_STATE;            // asset / K-line period / fan / correlation picks, kept across re-renders
  let current = null, token = 0;

  function mark(id) {
    for (const [k, el] of items) el.classList.toggle('on', k === id);
    if (![...picker.options].some(o => o.value === id)) picker.append(h('option', { value: id }, id));
    picker.value = id;
  }

  async function select(id, { scroll = false, note = null } = {}) {
    if (!id) return;
    const seq = ++token;
    current = id;
    mark(id);
    ui.setRoute('scenarios', encodeURIComponent(id), alive);
    detailRoot.style.minHeight = `${detailRoot.offsetHeight}px`;     // keep the scroll position while re-rendering
    disposeCharts(detailRoot);
    detailRoot.innerHTML = '';
    detailRoot.append(ui.loading('载入情景详情…'));
    if (scroll) anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
    try {
      const d = await api.get(`scenarios/${id}`);
      if (seq !== token || !alive()) return;
      renderScenarioDetail(detailRoot, d, ctx, { state, labLink: true });
      if (note) detailRoot.prepend(ui.notice(note, 'error'), h('div', { style: { height: '12px' } }));
      if (!builtinIds.has(id)) {
        detailRoot.prepend(ui.notice(`这是情景实验室里保存的自定义情景（${d.source === 'llm' ? 'LLM 翻译' : '手动编辑'}），不参与情景库的概率归一化；在“组合评估”页可以把它加进来。`, 'info'),
          h('div', { style: { height: '12px' } }));
      }
    } catch (err) {
      if (seq !== token) return;
      detailRoot.innerHTML = '';
      if ((err.status === 404 || err.status === 400) && !builtinIds.has(id)) {
        select(list[0].id, { note: `没有找到情景「${id}」（可能已被删除），下面显示的是概率最高的情景。` });
        return;
      } else {
        detailRoot.append(ui.notice(`无法载入情景「${id}」：${errorText(err)}`, 'error'));
      }
    } finally {
      if (seq === token) requestAnimationFrame(() => { detailRoot.style.minHeight = ''; });
    }
  }

  const first = param || list[0].id;
  select(first);

  const showCompare = () => {
    cmpBox.innerHTML = '';
    try { drawCompare(); } catch (err) {
      console.error(err);
      charts.unmount(cmpBox);
      cmpBox.innerHTML = '';
      cmpBox.style.height = 'auto';
      cmpBox.append(ui.notice(`对比图渲染失败：${err.message}`, 'error'));
    }
  };
  const statsOf = (assets) => Object.fromEntries(Object.entries(assets || {}).map(([a, x]) => [a, x?.stats || x || {}]));
  if (list.every(s => s.assets && typeof s.assets === 'object' && Object.keys(s.assets).length)) {
    details = new Map(list.map(s => [s.id, statsOf(s.assets)]));
    showCompare();
  } else {
    // an older index without per-asset stats: every built-in detail (also warms the cache for switching)
    Promise.allSettled(list.map(s => api.get(`scenarios/${s.id}`))).then((res) => {
      if (!alive() || !cmpBox.isConnected) return;
      details = new Map();
      res.forEach((r, i) => { if (r.status === 'fulfilled' && r.value) details.set(list[i].id, statsOf(r.value.assets)); });
      if (!details.size) {
        const err = res.find(r => r.status === 'rejected')?.reason;
        cmpBox.innerHTML = '';
        cmpBox.style.height = 'auto';
        cmpBox.append(ui.notice(`无法载入情景详情：${errorText(err)}`, 'error'));
        return;
      }
      showCompare();
    });
  }
  return param ? anchor : null;          // a deep link opens at the selected scenario
}
