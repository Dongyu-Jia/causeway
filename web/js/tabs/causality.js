// Causal transmission: curated causal chains and their evidence, sign-restricted SVAR impulse
// responses by period, the historical decomposition of inflation, labor-share local projections
// and the asset factor betas by period.
const INK = '#1d2433', INK3 = '#7a8396', LINE = '#e3e6ec';
const AX = {
  axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false },
  axisLabel: { color: INK3, fontSize: 11 }, splitLine: { lineStyle: { color: '#eef0f4' } },
};
const DIVERGING = ['#1c7ed6', '#a5d8ff', '#f8f9fa', '#ffc9c9', '#e03131'];

const CHAIN_SHORT = { oil_supply_chain: '石油供给', monetary_chain: '货币紧缩', labor_share_chain: '劳动收入占比',
  financial_chain: '金融冲击', ai_capex_chain: 'AI 资本开支' };
// SVAR shock behind each identified chain, and the SVAR variable each chain node stands for
const CHAIN_SHOCK = { oil_supply_chain: 'oil_supply', monetary_chain: 'monetary', financial_chain: 'financial' };
const NODE_VAR = { oil: 'oil', cpi: 'cpi', fed: 'ff', ff: 'ff', output: 'ip', spread: 'spread' };
const GROUP_STYLE = {
  shock: { color: '#fff4e6', borderColor: '#e8590c' }, price: { color: '#fff0f6', borderColor: '#c2255c' },
  policy: { color: '#f3f0ff', borderColor: '#7048e8' }, macro: { color: '#e7f5ff', borderColor: '#1c7ed6' },
  finance: { color: '#fff9db', borderColor: '#f08c00' }, asset: { color: '#ebfbee', borderColor: '#2f9e44' },
  distribution: { color: '#e3fafc', borderColor: '#1098ad' }, capital: { color: '#f8f0fc', borderColor: '#ae3ec9' },
};
const GROUP_NAME = { shock: '冲击', price: '价格', policy: '政策', macro: '宏观', finance: '金融', asset: '资产',
  distribution: '收入分配', capital: '资本' };
const SIGN_COLOR = { '+': '#3b5bdb', '-': '#e8590c', '±': '#868e96' };
// an edge's sign is the direction of its END variable after the original shock (or the sign of a β),
// so it reads as "the target goes up / down", not "same / opposite direction as the source box"
const SIGN_TEXT = { '+': '↑ 上升', '-': '↓ 下降', '±': '↕ 各时期不一' };
const SIGN_ARROW = { '+': '↑', '-': '↓', '±': '↕' };
// the variable whose impact response says how big a 1-sd shock is
const DEFINING = { oil_supply: 'oil', demand: 'ip', monetary: 'ff', financial: 'spread' };
const SHOCK_COLORS = { oil_supply: '#e8590c', demand: '#3b5bdb', monetary: '#7048e8', financial: '#c2255c',
  other: '#adb5bd', covid: '#495057' };
const PERIOD_COLORS = ['#3b5bdb', '#e8590c', '#2f9e44', '#c2255c'];
const EPISODES = [
  { year: '1974', label: '1974 石油禁运' }, { year: '1980', label: '1980 伊朗革命与沃尔克' },
  { year: '2008', label: '2008 油价高峰' }, { year: '2022', label: '2022 疫情后通胀' },
];

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
function signed(v, d = 2) {
  if (!isNum(v)) return '—';
  const s = v.toFixed(d);
  return Number(s) === 0 ? (0).toFixed(d) : `${v > 0 ? '+' : ''}${s}`;
}
const unitFmt = (unit, d = 2) => (v) => (isNum(v) ? `${signed(v, d)}${unit === 'pp' ? 'pp' : '%'}` : '—');
// band bound: a bound that rounds to 0.00 but is not zero keeps a third decimal, so "excludes 0" stays visible
const bound = (v) => (isNum(v) && v !== 0 && Math.abs(v) < 0.005 ? signed(v, 3) : signed(v));
const compact = (v) => `${Number(v.toFixed(2))}`;
// fixed decimals without a sign, never "-0.00"
const fixed = (v, d = 2) => (Number(v.toFixed(d)) === 0 ? (0).toFixed(d) : v.toFixed(d));

const CSS = `
.causality-toolbar { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-bottom: 10px; }
.causality-toolbar .lbl { font-size: 12px; color: var(--ink-3); }
.causality-multiples { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(min(200px, 100%), 1fr)); }
.causality-multiples.wide { grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); }
.causality-mini { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px 4px; background: var(--panel); min-width: 0; }
.causality-mini-title { font-size: 13px; font-weight: 600; }
.causality-mini-sub { font-size: 11px; color: var(--ink-3); }
.causality-mini .chart { height: 190px; }
.causality-legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--ink-2); margin: 10px 0 4px; align-items: center; }
.causality-swatch { display: inline-block; width: 14px; height: 9px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; border: 1px solid transparent; }
.causality-line { display: inline-block; width: 18px; height: 0; margin-right: 5px; vertical-align: 3px; border-top: 2px solid; }
.causality-evidence { white-space: pre-line; font-size: 12px; color: var(--ink-2); min-width: 260px; line-height: 1.6; }
.causality-kind-note { font-size: 11px; color: var(--ink-3); margin-top: 4px; white-space: normal; max-width: 220px; }
.causality-grid2 { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(440px, 100%), 1fr)); }
.causality-info { font-size: 12px; color: var(--ink-2); margin: 0 0 10px; }
.causality-details { margin-top: 14px; border-top: 1px dashed var(--line); padding-top: 10px; }
.causality-details > summary { cursor: pointer; font-weight: 600; color: var(--ink-2); font-size: 13px; }
.causality-note { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; }
.causality-gap { height: 16px; }
.causality-cell-link { min-width: 150px; }
.causality-cell-kind { min-width: 150px; }
.causality-cell-factor { min-width: 112px; }
.causality-nowrap { white-space: nowrap; }
.causality-group { display: inline-flex; flex-wrap: wrap; gap: 6px 8px; align-items: center; }
.causality-t { display: block; font-size: 11px; color: var(--ink-3); }
/* wide tables collapse into labelled blocks when their card is narrow */
.causality-stackbox { container-type: inline-size; }
@container (max-width: 820px) {
  .causality-stack table.tbl thead { display: none; }
  .causality-stack table.tbl, .causality-stack table.tbl tbody, .causality-stack table.tbl td { display: block; width: auto; }
  /* short cells flow two per line, the first cell and long (wrapping) cells take the full width */
  .causality-stack table.tbl tr { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(230px, 100%), 1fr));
    column-gap: 18px; padding: 10px 2px; border-bottom: 1px solid var(--line); }
  .causality-stack table.tbl td { border: 0; padding: 2px 0; text-align: left; white-space: normal; }
  .causality-stack table.tbl td.wrap, .causality-stack table.tbl td:first-child { grid-column: 1 / -1; }
  .causality-stack table.tbl td::before { content: attr(data-label); display: inline-block; min-width: 76px;
    margin-right: 8px; font-size: 11px; color: var(--ink-3); vertical-align: top; }
  .causality-stack table.tbl td:first-child { font-weight: 600; }
  .causality-stack table.tbl td:first-child::before { display: none; }
  .causality-stack .causality-cell-link, .causality-stack .causality-cell-kind,
  .causality-stack .causality-evidence { min-width: 0; }
  .causality-stack .causality-kind-note { max-width: none; }
}
`;

function injectStyle() {
  let el = document.getElementById('style-fe1');
  if (!el) { el = document.createElement('style'); el.id = 'style-fe1'; document.head.append(el); }
  if (!el.dataset.causality) { el.append(document.createTextNode(CSS)); el.dataset.causality = '1'; }
}

// Charts inside a container that is about to be emptied (unmount also stops the re-layout observers).
function disposeCharts(charts, el) {
  el.querySelectorAll('.chart').forEach((c) => charts.unmount(c));
}

// Copies the header text onto every cell (data-label) so the table can collapse into blocks.
function stackable(ui, wrap) {
  const heads = [...wrap.querySelectorAll('thead th')].map(th => th.textContent);
  wrap.querySelectorAll('tbody tr').forEach(tr => [...tr.children].forEach((td, i) => { td.dataset.label = heads[i] || ''; }));
  wrap.classList.add('causality-stack');
  return ui.h('div', { class: 'causality-stackbox' }, wrap);
}

function loadError(ui, err) {
  const { h } = ui;
  if (err.status === 404) {
    return h('div', {}, ui.notice('还没有因果传导结果（causality.json）。', 'info'), ui.buildHint(err));
  }
  return ui.notice(`读取因果传导结果失败：${err.message}`, 'error');
}

// Impulse responses: one shaded band + median per entry of `bands`, extra comparison lines,
// and shaded `zones` for horizons whose sign was imposed by the identification.
function bandChart(charts, el, { x, bands, lines = [], zones = [], fmt, xEvery = 6, xUnit = '月' }) {
  const chart = charts.mount(el);
  const clean = (a) => (a || []).map(v => (isNum(v) ? v : null));
  const series = [];
  bands.forEach((b, i) => {
    const lo = clean(b.lo), hi = clean(b.hi);
    series.push({ type: 'line', data: lo, stack: `band${i}`, stackStrategy: 'all', symbol: 'none', silent: true,
      lineStyle: { opacity: 0 }, emphasis: { disabled: true } });
    series.push({ type: 'line', data: hi.map((v, k) => (v !== null && lo[k] !== null ? v - lo[k] : null)), stack: `band${i}`,
      stackStrategy: 'all', symbol: 'none', silent: true, lineStyle: { opacity: 0 }, emphasis: { disabled: true },
      areaStyle: { color: b.color, opacity: 0.16 } });
  });
  const first = series.length;
  bands.forEach(b => series.push({ name: b.name, type: 'line', data: clean(b.mid), symbol: 'none',
    lineStyle: { width: 2, color: b.color }, itemStyle: { color: b.color } }));
  lines.forEach(l => series.push({ name: l.name, type: 'line', data: clean(l.data), symbol: 'none',
    lineStyle: { width: 1.3, type: l.dashed ? 'dashed' : 'solid', color: l.color }, itemStyle: { color: l.color } }));
  if (series[first]) {
    series[first].markLine = { silent: true, symbol: 'none', label: { show: false },
      lineStyle: { color: '#adb5bd', type: 'solid', width: 1 }, data: [{ yAxis: 0 }] };
    if (zones.length) {
      series[first].markArea = { silent: true, data: zones.map(z => [{ xAxis: String(z.from),
        itemStyle: { color: 'rgba(73, 80, 87, 0.13)' },
        label: { show: true, formatter: z.label, position: 'insideTopLeft', color: '#343a40', fontSize: 10 } }, { xAxis: String(z.to) }]) };
    }
  }
  chart.setOption({
    animation: false,
    grid: { left: 6, right: 12, top: 16, bottom: 4, containLabel: true },
    tooltip: { trigger: 'axis', confine: true, formatter: (ps) => {
      const i = ps[0].dataIndex;
      const rows = [`<b>冲击后第 ${x[i]} ${xUnit}</b>`];
      bands.forEach(b => rows.push(`${b.name}：<b>${fmt(b.mid?.[i])}</b>（${fmt(b.lo?.[i])} ~ ${fmt(b.hi?.[i])}）`));
      lines.forEach(l => rows.push(`${l.name}：${fmt(l.data?.[i])}`));
      return rows.join('<br/>');
    } },
    xAxis: { type: 'category', data: x.map(String), boundaryGap: false, ...AX, splitLine: { show: false },
      axisLabel: { ...AX.axisLabel, interval: (idx) => idx % xEvery === 0 } },
    yAxis: { type: 'value', ...AX, splitNumber: 4, axisLabel: { ...AX.axisLabel, formatter: compact } },
    series,
  });
  return chart;
}

// Null-safe heatmap with per-cell styles; labelFmt(v, r, c) and tip(r, c, v) are callbacks.
function heat(charts, el, { rows, cols, values, min, max, labelFmt, tip, cellStyle, height }) {
  if (height) el.style.height = `${height}px`;
  const chart = charts.mount(el);
  const data = [];
  values.forEach((row, r) => row.forEach((v, c) => {
    const ok = isNum(v);
    const item = { value: [c, r, ok ? v : '-'] };
    const st = cellStyle && cellStyle(r, c, ok ? v : null);
    if (st) item.itemStyle = st;
    data.push(item);
  }));
  const rowChars = Math.max(...rows.map(s => String(s).length), 2);
  const cellW = ((el.clientWidth || 600) - rowChars * 11 - 30) / Math.max(cols.length, 1);
  // 35° labels collide once the cells are narrower than ~24 px (a phone): stand them upright
  const rotate = cellW >= 24 ? 35 : 90;
  chart.setOption({
    animation: false,
    tooltip: { confine: true, formatter: (p) => { const [c, r, v] = p.value; return tip(r, c, isNum(v) ? v : null); } },
    grid: { left: 8, right: 16, top: 8, bottom: 8, containLabel: true },
    xAxis: { type: 'category', data: cols, ...AX, splitLine: { show: false }, axisLabel: { ...AX.axisLabel, rotate, interval: 0 } },
    yAxis: { type: 'category', data: rows, inverse: true, ...AX, splitLine: { show: false }, axisLabel: { ...AX.axisLabel, interval: 0, color: INK } },
    visualMap: { min, max, show: false, calculable: false, dimension: 2, inRange: { color: DIVERGING } },
    series: [{ type: 'heatmap', data,
      label: { show: cellW >= 34, fontSize: 10, color: INK,
        formatter: (p) => (isNum(p.value[2]) ? labelFmt(p.value[2], p.value[1], p.value[0]) : '') },
      emphasis: { itemStyle: { borderColor: INK, borderWidth: 1 } } }],
  });
  return chart;
}

// Median with a 68% whisker per row (normalized pass-through across periods).
function rangeChart(charts, el, { rows, fmt, color }) {
  el.style.height = `${rows.length * 36 + 36}px`;
  const chart = charts.mount(el);
  chart.setOption({
    animation: false,
    grid: { left: 8, right: 22, top: 8, bottom: 4, containLabel: true },
    tooltip: { trigger: 'item', confine: true, formatter: (p) => {
      const r = rows[p.value[0]];
      return `<b>${r.label}</b><br/>中位数 <b>${fmt(r.p50)}</b><br/>68% 区间 ${fmt(r.p16)} ~ ${fmt(r.p84)}`;
    } },
    xAxis: { type: 'value', ...AX, splitNumber: 4, axisLabel: { ...AX.axisLabel, formatter: compact } },
    yAxis: { type: 'category', data: rows.map(r => r.label), inverse: true, ...AX, splitLine: { show: false },
      axisLabel: { ...AX.axisLabel, color: INK } },
    series: [{ type: 'custom', data: rows.map((r, i) => [i, r.p16, r.p50, r.p84]), encode: { x: [1, 2, 3], y: 0 },
      renderItem: (params, api) => {
        const y = api.value(0);
        const a = api.coord([api.value(1), y]), m = api.coord([api.value(2), y]), b = api.coord([api.value(3), y]);
        const st = { stroke: color, lineWidth: 2 };
        return { type: 'group', children: [
          { type: 'line', shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style: st },
          { type: 'line', shape: { x1: a[0], y1: a[1] - 6, x2: a[0], y2: a[1] + 6 }, style: st },
          { type: 'line', shape: { x1: b[0], y1: b[1] - 6, x2: b[0], y2: b[1] + 6 }, style: st },
          { type: 'circle', shape: { cx: m[0], cy: m[1], r: 5 }, style: { fill: color, stroke: '#fff', lineWidth: 1.5 } },
        ] };
      },
      markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: '#adb5bd', type: 'solid' }, data: [{ xAxis: 0 }] } }],
  });
  return chart;
}

// Contributions of each shock to 12-month CPI inflation (stacked, pp) and actual minus baseline.
function hdChart(charts, el, hd, episodes) {
  const chart = charts.mount(el);
  const keys = Object.keys(hd.contrib || {});
  const names = hd.names || {};
  const n = hd.dates.length;
  const dev = hd.dates.map((_, i) => (isNum(hd.actual?.[i]) && isNum(hd.baseline?.[i]) ? +(hd.actual[i] - hd.baseline[i]).toFixed(4) : null));
  const series = keys.map(k => ({ name: names[k] || k, type: 'bar', stack: 'hd', barCategoryGap: '0%',
    data: (hd.contrib[k] || []).map(v => (isNum(v) ? v : null)), itemStyle: { color: SHOCK_COLORS[k] || '#868e96' },
    emphasis: { focus: 'series' } }));
  // the inflation level itself, for orientation: its gap to the black line is the baseline
  series.push({ name: '实际 12 个月通胀（水平）', type: 'line', data: (hd.actual || []).map(v => (isNum(v) ? v : null)),
    symbol: 'none', z: 9, lineStyle: { width: 1.2, type: 'dashed', color: '#868e96' }, itemStyle: { color: '#868e96' } });
  series.push({ name: '实际通胀 − 基线', type: 'line', data: dev, symbol: 'none', z: 10,
    lineStyle: { width: 1.6, color: INK }, itemStyle: { color: INK },
    // labels alternate in height so neighbouring episodes (1974 / 1980) never collide
    markLine: { silent: true, symbol: 'none', lineStyle: { color: INK3, type: 'dashed' },
      label: { color: INK, fontSize: 11, position: 'end', formatter: (p) => p.name },
      data: episodes.map((ep, k) => ({ name: ep.label, xAxis: ep.date, label: { distance: k % 2 ? 26 : 10 } })) } });
  // a plain legend wraps onto several rows on narrow cards (a scrolling one hides the two lines);
  // estimate its rows so the plot and the episode labels start below it
  const width = el.clientWidth || 900;
  let rows = 1, run = 0;
  for (const s of series.map(x => x.name)) {
    // CJK characters are about 12px wide at this font size, digits and Latin about 6.6px
    const w = [...String(s)].reduce((a, ch) => a + (/[⺀-鿿＀-￯]/.test(ch) ? 12 : 6.6), 0) + 34;
    if (run > 0 && run + w > width - 16) { rows += 1; run = 0; }
    run += w;
  }
  chart.setOption({
    animation: false,
    legend: { top: 0, left: 'center', textStyle: { color: INK, fontSize: 12 }, itemWidth: 14, itemHeight: 8, itemGap: 12 },
    tooltip: { trigger: 'axis', confine: true, axisPointer: { type: 'line' }, formatter: (ps) => {
      const i = ps[0].dataIndex;
      const parts = keys.map(k => ({ k, v: hd.contrib[k]?.[i] })).filter(p => isNum(p.v)).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      return [`<b>${hd.dates[i]}</b>`,
        `实际 12 个月通胀 <b>${isNum(hd.actual?.[i]) ? hd.actual[i].toFixed(2) : '—'}%</b>，基线 ${isNum(hd.baseline?.[i]) ? hd.baseline[i].toFixed(2) : '—'}%`,
        ...parts.map(p => `<span style="color:${SHOCK_COLORS[p.k] || INK}">■</span> ${names[p.k] || p.k}：${signed(p.v)}pp`),
        `实际 − 基线：<b>${signed(dev[i])}pp</b>`].join('<br/>');
    } },
    grid: { left: 44, right: 28, top: 10 + rows * 22 + 54, bottom: 60 },
    xAxis: { type: 'category', data: hd.dates, ...AX, splitLine: { show: false } },
    yAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: (v) => `${v}pp` } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }],
    series,
  });
  chart.zoomTo = (i, w = 48) => chart.dispatchAction({ type: 'dataZoom', startValue: Math.max(0, i - w), endValue: Math.min(n - 1, i + w) });
  return chart;
}

export async function render(root, { api, charts, ui, param, alive = () => root.isConnected }) {
  const { h, card } = ui;
  injectStyle();
  root.append(h('p', { class: 'lead' },
    '这一页回答“为什么”：把每个月的宏观变化拆成可识别的结构冲击（石油供给、总需求、货币政策、金融风险），',
    '估计它们怎么一步步传到通胀、产出、利率和资产，以及传导在不同时代怎么变；劳动收入占比的冲击单独用季度数据估计。',
    '符号约束只规定冲击后最初几个月的方向，之后的走势由数据决定，哪些是假设、哪些是数据，下面逐条标出。'));

  let data;
  try {
    data = await api.get('causality');
  } catch (err) {
    root.append(loadError(ui, err));
    return;
  }
  if (!alive()) return;                  // the user moved on while the results loaded
  const svar = data.svar || null;
  const hd = data.historical_decomposition || null;
  const dist = data.distribution || null;
  const betas = data.asset_betas || null;
  const chains = data.chains || [];
  const shocks = svar?.shocks || [];
  const periods = svar?.periods || [];
  const periodLabel = (p) => (p.id === 'full' ? `全样本 ${String(p.start).slice(0, 4)}–${String(p.end).slice(0, 4)}`
    : `${String(p.start).slice(0, 4)}–${String(p.end).slice(0, 4)}`);
  const restrictionOf = (shock, v) => shocks.find(s => s.id === shock)?.restrictions?.find(r => r.var === v) || null;
  const replaceHash = (v) => ui.setRoute('causality', v, alive);

  // ======================================================== causal chains
  root.append(h('div', { class: 'section-title' }, '因果链：从冲击到资产'));
  const chainWrap = h('div');
  root.append(chainWrap);
  let chainSel = chains.some(c => c.id === param) ? param
    : (Object.entries(CHAIN_SHOCK).find(([, s]) => s === param)?.[0] || chains[0]?.id);

  const isDescriptive = (ch) => ch.id === 'ai_capex_chain' || /描述性/.test(ch.title || '');
  function edgeKind(ch, e) {
    const ev = String(e.evidence || '');
    if (isDescriptive(ch)) {
      // the last AI edge carries the sign of the asset's β to the 10-year yield, not of "valuation"
      const beta = ev.includes('β') && ev.includes('10 年期收益率');
      return { label: '描述性', cls: '', note: beta
        ? '方向取自证据里的 β：10 年期收益率上升时这类资产下跌（利率敏感），不是“估值抬升”本身的因果效应。'
        : '没有可识别的外生冲击：这里是现状数据和相关历史估计，不是因果效应。' };
    }
    const m = ev.match(/第\s*(\d+)\s*-\s*(\d+)\s*月(上升|下降)是识别约束/);
    if (m) return { label: '约束 + 数据', cls: 'accent', note: `第 ${m[1]}–${m[2]} 月${m[3]}是识别假设，之后的方向和大小由数据决定。` };
    if (ev.includes('未施加约束')) return { label: '数据决定', cls: 'good', note: '这个变量没有施加符号约束，方向完全由数据决定。' };
    const shock = CHAIN_SHOCK[ch.id], v = NODE_VAR[e.to];
    const r = shock && e.from === 'shock' && v ? restrictionOf(shock, v) : null;
    if (r) {
      return { label: '识别约束', cls: 'warn', note: `冲击的定义：第 ${r.horizon[0]}–${r.horizon[1]} 月${r.sign === '+' ? '上升' : '下降'}是假设，不是估计结果；只有幅度来自数据。` };
    }
    if (ev.includes('β')) return { label: '回归 β', cls: 'good', note: '资产模型的同月回归系数，分时期估计，方向由数据决定。' };
    if (ev.includes('90% 区间')) return { label: '局部投影', cls: 'good', note: '季度局部投影，方向由数据决定（劳动收入占比只能滞后一季影响宏观）。' };
    return { label: '数据决定', cls: 'good', note: '' };
  }
  const evidenceText = (ev) => String(ev || '—').replace(/；/g, '；\n').replace(/。(?=.)/g, '。\n');

  function renderChain() {
    disposeCharts(charts, chainWrap);
    chainWrap.innerHTML = '';
    const ch = chains.find(c => c.id === chainSel);
    if (!ch) { chainWrap.append(ui.notice('causality.json 里没有因果链。', 'info')); return; }
    const labelOf = Object.fromEntries((ch.nodes || []).map(n => [n.id, n.label]));
    const groupOf = Object.fromEntries((ch.nodes || []).map(n => [n.id, n.group]));
    const edges = (ch.edges || []).map(e => ({ ...e, kind: edgeKind(ch, e) }));
    const box = ui.chartBox();
    const groups = [...new Set((ch.nodes || []).map(n => n.group))];
    chainWrap.append(card({ title: ch.title,
      sub: '方框是传导环节，箭头上是终点变量的方向与时滞：蓝 ↑ = 冲击后上升、橙 ↓ = 下降、灰 ↕ = 各时期方向不一致；鼠标悬停箭头看数据证据，表格逐条说明方向来自识别假设还是数据。' },
    h('div', { class: 'causality-toolbar' }, h('span', { class: 'causality-group' }, h('span', { class: 'lbl' }, '选择因果链'),
      ui.seg(chains.map(c => ({ value: c.id, label: CHAIN_SHORT[c.id] || c.title })), chainSel, (v) => { chainSel = v; renderChain(); replaceHash(v); }))),
    isDescriptive(ch) ? ui.notice('这条链是描述性的：AI 资本开支和利润占比繁荣还没有可识别的外生冲击，证据栏给出的是现状数据和相关的历史估计（如劳动收入占比冲击的局部投影、利率 β），不能读成因果效应。') : null,
    box,
    h('div', { class: 'causality-legend' },
      groups.map(g => h('span', {}, h('span', { class: 'causality-swatch', style: { background: GROUP_STYLE[g]?.color, borderColor: GROUP_STYLE[g]?.borderColor } }), GROUP_NAME[g] || g)),
      Object.entries(SIGN_TEXT).map(([s, t]) => h('span', {}, h('span', { class: 'causality-line', style: { borderColor: SIGN_COLOR[s] } }), t))),
    h('p', { class: 'causality-note', style: { margin: '0 0 8px' } }, isDescriptive(ch)
      ? '读法：方向是证据里现状数据或历史估计的方向，箭头只是推测的传导顺序。'
      : '读法：脉冲响应和局部投影类证据都是“最初那个冲击”发生后箭头终点变量的总变化（↑ 上升 / ↓ 下降），箭头表示推测的传导顺序，中间环节没有单独识别；回归 β 类证据是起点因子上升时资产的同月方向。'),
    stackable(ui, ui.table([
      { value: (e) => `${labelOf[e.from] || e.from} → ${labelOf[e.to] || e.to}`, label: '传导环节', wrap: true,
        fmt: (v) => h('div', { class: 'causality-cell-link' }, v) },
      { key: 'sign', label: '终点方向', fmt: (v) => h('span', { class: 'causality-nowrap', style: { color: SIGN_COLOR[v] || INK, fontWeight: 600 } }, SIGN_TEXT[v] || v || '—') },
      { key: 'label', label: '时滞', fmt: (v) => v || '—' },
      { key: 'kind', label: '方向从哪来', wrap: true,
        fmt: (k) => h('div', { class: 'causality-cell-kind' }, ui.chip(k.label, k.cls), k.note ? h('div', { class: 'causality-kind-note' }, k.note) : null) },
      { key: 'evidence', label: '数据证据', wrap: true, fmt: (v) => h('div', { class: 'causality-evidence' }, evidenceText(v)) },
    ], edges))));

    // charts.dag lays the chain out in pixels (left to right on wide cards, top to bottom on narrow ones),
    // cuts every edge at the box borders so the arrowheads show, and lays it out again on resize
    const esc = ui.esc;
    charts.dag(box, {
      nodes: (ch.nodes || []).map(n => ({ ...n, style: { ...(GROUP_STYLE[n.group] || {}), borderWidth: 1.3 } })),
      edges,
      edgeColor: (e) => SIGN_COLOR[e.sign] || INK3,
      edgeLabel: (e) => `${SIGN_ARROW[e.sign] || ''} ${e.label || ''}`.trim(),
      edgeTip: (e) => `<b>${esc(labelOf[e.from] || e.from)} → ${esc(labelOf[e.to] || e.to)}</b><br/>`
        + `终点方向 ${esc(SIGN_TEXT[e.sign] || e.sign || '—')}，时滞 ${esc(e.label || '—')}，${esc(e.kind?.label || '')}<br/>`
        + evidenceText(e.evidence).split('\n').map(esc).join('<br/>'),
      nodeTip: (n) => `<b>${esc(labelOf[n.id] || n.id)}</b><br/>${esc(GROUP_NAME[groupOf[n.id]] || '')}`,
    });
  }

  // ======================================================== SVAR impulse responses
  root.append(h('div', { class: 'section-title' }, '结构冲击的脉冲响应（符号约束 SVAR）'));
  const irfInfo = h('div');
  const irfGrid = h('div', { class: 'causality-multiples' });
  const irfLegend = h('div', { class: 'causality-legend' });
  const normWrap = h('div');
  let shockSel = shocks.some(s => s.id === param) ? param : (CHAIN_SHOCK[chainSel] || shocks[0]?.id);
  let periodSel = periods.some(p => p.id === 'full') ? 'full' : periods[0]?.id;
  let overlay = false;
  if (svar?.irf) {
    root.append(card({ title: '1σ 结构冲击之后 36 个月的响应',
      sub: '油价、工业产出、CPI 是累计变化（%），利率与利差是水平变化（pp）；实线是中位数、阴影是 16–84% 区间，灰色底纹是被符号约束强制方向的月份（识别假设），其余都由数据决定。' },
    h('div', { class: 'causality-toolbar' },
      h('span', { class: 'causality-group' }, h('span', { class: 'lbl' }, '冲击'),
        ui.seg(shocks.map(s => ({ value: s.id, label: s.name })), shockSel, (v) => { shockSel = v; renderIrf(); replaceHash(v); })),
      h('span', { class: 'causality-group' }, h('span', { class: 'lbl' }, '估计时期'),
        ui.seg(periods.map(p => ({ value: p.id, label: periodLabel(p) })), periodSel, (v) => { periodSel = v; renderIrf(); })),
      ui.seg([{ value: 'off', label: '只看本时期' }, { value: 'on', label: '叠加其他时期中位数' }], 'off', (v) => { overlay = v === 'on'; renderIrf(); })),
    irfInfo, irfGrid, irfLegend),
    h('div', { class: 'causality-gap' }),
    card({ title: '油价冲击的标准化传导：每 +10% 油价',
      sub: '各时期 1σ 石油冲击大小不同，按冲击当月油价涨 10% 换算后才能比较：圆点是中位数，横线是 68% 区间，看油价传到通胀和产出的力度怎么随时代变化。' },
    normWrap));
  } else {
    root.append(ui.notice('causality.json 里没有 SVAR 脉冲响应。', 'info'));
  }

  function renderIrf() {
    disposeCharts(charts, irfGrid);
    irfGrid.innerHTML = ''; irfInfo.innerHTML = ''; irfLegend.innerHTML = '';
    const shock = shocks.find(s => s.id === shockSel);
    const per = periods.find(p => p.id === periodSel);
    const irf = svar.irf?.[periodSel]?.[shockSel];
    if (!shock || !per || !irf) { irfGrid.append(ui.notice('这个时期没有该冲击的估计结果。', 'info')); return; }
    const dr = svar.draws?.[periodSel] || {};
    const defVar = DEFINING[shockSel];
    const defUnit = (svar.variables || []).find(v => v.id === defVar)?.unit;
    const size = defVar && isNum(irf[defVar]?.p50?.[0])
      ? `这个时期 1σ 冲击当月：${(svar.variables || []).find(v => v.id === defVar)?.name || defVar} ${unitFmt(defUnit)(irf[defVar].p50[0])}。` : '';
    irfInfo.append(h('p', { class: 'causality-info' },
      h('b', {}, `${shock.name}的识别约束：`), shock.text || '—', '。', size,
      `估计样本 ${per.start} 至 ${per.end}（${per.name}）${isNum(dr.n_obs) ? `，${dr.n_obs} 个月` : ''}；`,
      `接受 ${svar.n_accepted?.[periodSel] ?? '—'} 次旋转${isNum(dr.acceptance_rate) ? `（接受率 ${(dr.acceptance_rate * 100).toFixed(1)}%）` : ''}`,
      isNum(dr.max_root_ols) ? `；VAR 最大特征根 ${dr.max_root_ols.toFixed(3)}（小于 1 才稳定）。` : '。'));
    const x = Array.from({ length: (svar.horizon ?? 36) + 1 }, (_, i) => i);
    const others = periods.filter(p => p.id !== periodSel);
    // one fixed colour per period, so a period keeps its colour whichever one is selected
    const colorOf = (p) => PERIOD_COLORS[Math.max(0, periods.indexOf(p)) % PERIOD_COLORS.length];
    for (const v of svar.variables || []) {
      const r = irf[v.id];
      if (!r) continue;
      const box = ui.chartBox();
      irfGrid.append(h('div', { class: 'causality-mini' },
        h('div', { class: 'causality-mini-title' }, v.name), h('div', { class: 'causality-mini-sub' }, v.unit === 'pp' ? '水平变化，百分点' : '累计变化，%'), box));
      const lines = [];
      const mt = svar.median_target;
      if (periodSel === mt?.period && mt?.irf?.[shockSel]?.[v.id]) {
        lines.push({ name: '点模型（中位数目标）', data: mt.irf[shockSel][v.id], color: '#212529', dashed: true });
      }
      if (overlay) {
        others.forEach((p) => {
          const o = svar.irf?.[p.id]?.[shockSel]?.[v.id]?.p50;
          if (o) lines.push({ name: `${periodLabel(p)} 中位数`, data: o, color: colorOf(p) });
        });
      }
      const rs = restrictionOf(shockSel, v.id);
      bandChart(charts, box, { x, fmt: unitFmt(v.unit), xEvery: 6, xUnit: '月', lines,
        bands: [{ name: periodLabel(per), lo: r.p16, mid: r.p50, hi: r.p84, color: colorOf(per) }],
        zones: rs ? [{ from: rs.horizon[0], to: rs.horizon[1], label: `约束${rs.sign === '+' ? '↑' : '↓'}` }] : [] });
    }
    irfLegend.append(...[
      h('span', {}, h('span', { class: 'causality-line', style: { borderColor: colorOf(per) } }), `${periodLabel(per)} 中位数`),
      h('span', {}, h('span', { class: 'causality-swatch', style: { background: colorOf(per), opacity: 0.25 } }), '16–84% 区间'),
      h('span', {}, h('span', { class: 'causality-swatch', style: { background: 'rgba(73, 80, 87, 0.22)' } }), '符号约束月份（假设）'),
      periodSel === svar.median_target?.period ? h('span', {}, h('span', { class: 'causality-line', style: { borderColor: '#212529', borderTopStyle: 'dashed' } }), '点模型（Fry-Pagan 中位数目标，用于历史分解）') : null,
      ...(overlay ? others.map((p) => h('span', {}, h('span', { class: 'causality-line', style: { borderColor: colorOf(p) } }), `${periodLabel(p)} 中位数`)) : []),
      h('span', { class: 'muted' }, '各时期 1σ 冲击大小不同，跨时期比较力度请看下面的标准化传导。'),
    ].filter(Boolean));
  }

  function renderNorm() {
    const norm = svar.normalized || {};
    const rows = periods.filter(p => norm[p.id]).map(p => ({ id: p.id, label: periodLabel(p), name: p.name,
      cpi: norm[p.id].cpi12_per_oil10 || {}, ip: norm[p.id].ip24_per_oil10 || {} }));
    if (!rows.length) { normWrap.append(ui.notice('没有标准化传导结果。', 'info')); return; }
    const cpiBox = ui.chartBox(), ipBox = ui.chartBox();
    // median with its 68% band underneath, so the table fits a narrow card
    const cellOf = (q, cls) => (isNum(q?.p50) ? h('span', { class: cls || '' }, `${signed(q.p50)}%`,
      h('span', { class: 'causality-t' }, isNum(q.p16) ? `68%：${signed(q.p16)} ~ ${signed(q.p84)}` : '')) : '—');
    // strongest / weakest CPI pass-through among the sub-periods, stated from the data
    const sub = rows.filter(r => r.id !== 'full' && isNum(r.cpi.p50)).sort((a, b) => b.cpi.p50 - a.cpi.p50);
    if (sub.length > 1) {
      const hi = sub[0], lo = sub[sub.length - 1];
      normWrap.append(h('p', { class: 'causality-info' }, h('b', {}, '读法：'),
        `油价涨 10%，12 个月后 CPI 累计多涨：${hi.label}（${hi.name}）最强 ${signed(hi.cpi.p50)}%，${lo.label}（${lo.name}）最弱 ${signed(lo.cpi.p50)}%；`,
        `24 个月后工业产出：${rows.filter(r => r.id !== 'full').map(r => `${r.label} ${isNum(r.ip.p50) ? `${signed(r.ip.p50)}%` : '—'}`).join('，')}。区间都很宽，说明各时期的差别只有中位数层面的参考意义。`));
    }
    normWrap.append(
      ui.table([
        { key: 'label', label: '估计时期', fmt: (v, r) => h('span', {}, v, r.id === 'full' ? null : h('span', { class: 'causality-t' }, r.name)) },
        { value: (r) => r.cpi, label: 'CPI（12 个月后累计）', fmt: (q) => cellOf(q) },
        { value: (r) => r.ip, label: '工业产出（24 个月后累计）', fmt: (q) => cellOf(q, ui.signCls(q?.p50)) },
      ], rows),
      h('div', { class: 'causality-grid2', style: { marginTop: '12px' } },
        h('div', {}, h('div', { class: 'causality-mini-title' }, '12 个月后 CPI 累计（%）'), cpiBox),
        h('div', {}, h('div', { class: 'causality-mini-title' }, '24 个月后工业产出累计（%）'), ipBox)));
    const toRows = (key) => rows.map(r => ({ label: r.label, ...r[key] }));
    rangeChart(charts, cpiBox, { rows: toRows('cpi'), fmt: (v) => `${signed(v)}%`, color: '#c2255c' });
    rangeChart(charts, ipBox, { rows: toRows('ip'), fmt: (v) => `${signed(v)}%`, color: '#1c7ed6' });
  }

  // ======================================================== historical decomposition
  root.append(h('div', { class: 'section-title' }, '通胀的历史分解：每段通胀是哪种冲击推高的'));
  const hdBox = ui.chartBox('xtall');
  const hdTable = h('div');
  let hdChartInst = null;
  const episodes = [];
  if (hd?.dates?.length && hd.contrib) {
    for (const ep of EPISODES) {
      let best = -1;
      hd.dates.forEach((d, i) => { if (d.startsWith(ep.year) && isNum(hd.actual?.[i]) && (best < 0 || hd.actual[i] > hd.actual[best])) best = i; });
      if (best >= 0) episodes.push({ ...ep, i: best, date: hd.dates[best] });
    }
    const base = hd.baseline?.[hd.baseline.length - 1];
    root.append(card({ title: '12 个月 CPI 通胀的历史分解',
      sub: `柱子 = 各结构冲击对通胀的贡献（百分点），相对“没有任何冲击”的基线（${isNum(base) ? `近年约 ${base.toFixed(1)}%` : '初始条件与常数项'}）；柱子加总 = 黑线（实际通胀减基线），灰色虚线是实际通胀水平，两条线的差就是基线；可拖动下方滑块或滚轮缩放。`,
      actions: h('button', { class: 'btn causality-nowrap', onclick: () => hdChartInst?.dispatchAction({ type: 'dataZoom', start: 0, end: 100 }) }, '显示全部') },
    hdBox, hdTable,
    h('p', { class: 'causality-note' }, `${hd.unit || ''}。贡献为正 = 这类冲击把通胀推高：货币政策冲击按“紧缩”定义，它的贡献为正说明当时政策比平常的反应更宽松（负的紧缩冲击），为负说明更紧。`
      + `${hd.names?.covid || '疫情月份残差'}：疫情月份不参与估计，它们的残差不拆分到四种冲击里；“${hd.names?.other || '其他'}”是 VAR 里没有被约束识别的剩余冲击。`)));
  } else {
    root.append(ui.notice('causality.json 里没有历史分解。', 'info'));
  }

  // ======================================================== labor-share local projections
  root.append(h('div', { class: 'section-title' }, '劳动收入占比冲击：人员收入怎么影响宏观'));
  const lpGrid = h('div', { class: 'causality-multiples wide' });
  const lpTable = h('div');
  const altBody = h('div', { class: 'causality-multiples wide' });
  let lpH = 8;
  const lpPeriods = dist?.periods || [];
  if (dist?.responses) {
    const sd = lpPeriods.map(p => `${p.name} ${isNum(dist.shock_sd_pct?.[p.id]) ? `${dist.shock_sd_pct[p.id].toFixed(2)}%` : '—'}`).join('，');
    const details = h('details', { class: 'causality-details' },
      h('summary', {}, `替代识别：${dist.alternative?.name || '只控制滞后项'}（为什么不用它）`),
      h('p', { class: 'causality-info', style: { marginTop: '8px' } }, dist.alternative?.note || '—',
        isNum(dist.alternative?.shock_sd_pct?.[lpPeriods[0]?.id]) ? `这种冲击的 1σ：${lpPeriods.map(p => `${p.name} ${dist.alternative.shock_sd_pct[p.id]?.toFixed(2) ?? '—'}%`).join('，')}。` : ''),
      altBody);
    let altDrawn = false;
    details.addEventListener('toggle', () => { if (details.open && !altDrawn) { altDrawn = true; drawLp(altBody, dist.alternative?.responses || {}); } });
    root.append(card({ title: `${dist.shock || '劳动收入占比下降 1%'}之后 0–${dist.horizon_q ?? 12} 个季度`,
      sub: `季度局部投影：${lpPeriods.map((p, k) => `${['蓝', '橙', '绿', '紫'][k] || ''} = ${p.name}`).join('，')}，阴影是 90% 区间；比较两个时期，看“收入向资本倾斜”对消费、物价、利率和股价的影响有没有变。` },
    h('div', { class: 'causality-legend' }, lpPeriods.map((p, k) => h('span', {},
      h('span', { class: 'causality-line', style: { borderColor: PERIOD_COLORS[k % PERIOD_COLORS.length] } }), `${p.name}（冲击样本 ${p.shock_start} 至 ${p.shock_end}）`))),
    lpGrid,
    h('div', { class: 'causality-toolbar', style: { marginTop: '14px' } }, h('span', { class: 'causality-group' }, h('span', { class: 'lbl' }, '表格看第几季'),
      ui.seg([4, 8, 12].filter(q => q <= (dist.horizon_q ?? 12)).map(q => ({ value: q, label: `${q} 季后` })), lpH, (v) => { lpH = v; renderLpTable(); }))),
    lpTable,
    h('p', { class: 'causality-note' }, dist.method || '', ` 1σ 冲击约为劳动收入占比下降 ${sd}，图中统一换算成下降 1%。`),
    details));
  } else {
    root.append(ui.notice('causality.json 里没有劳动收入占比的局部投影。', 'info'));
  }

  function drawLp(grid, responses) {
    const hq = dist.horizon_q ?? 12;
    const x = Array.from({ length: hq + 1 }, (_, i) => i);
    for (const v of dist.variables || []) {
      const box = ui.chartBox();
      grid.append(h('div', { class: 'causality-mini' }, h('div', { class: 'causality-mini-title' }, v.name),
        h('div', { class: 'causality-mini-sub' }, `${v.unit === 'pp' ? '百分点' : '累计 %'}${v.id === 'labor_share' ? '（冲击本身的持续性）' : ''}`), box));
      const bands = lpPeriods.map((p, k) => {
        const r = responses[p.id]?.[v.id] || {};
        return { name: p.name, lo: r.lo, mid: r.coef, hi: r.hi, color: PERIOD_COLORS[k % PERIOD_COLORS.length] };
      });
      bandChart(charts, box, { x, bands, fmt: unitFmt(v.unit), xEvery: 2, xUnit: '季' });
    }
  }

  function renderLpTable() {
    lpTable.innerHTML = '';
    const cell = (r, unit) => {
      if (!r || !isNum(r.coef?.[lpH])) return '—';
      const sig = isNum(r.lo?.[lpH]) && isNum(r.hi?.[lpH]) && (r.lo[lpH] > 0 || r.hi[lpH] < 0);
      return h('span', { class: sig ? '' : 'muted' }, unitFmt(unit)(r.coef[lpH]),
        h('span', { class: 'muted', style: { fontSize: '11px' } }, `（${bound(r.lo?.[lpH])} ~ ${bound(r.hi?.[lpH])}）`), sig ? ' ★' : '');
    };
    lpTable.append(ui.table([
      { key: 'name', label: '变量', wrap: true, minWidth: 108 },        // the names wrap, but not one word per line on a phone
      ...lpPeriods.map(p => ({ value: (v) => ({ r: dist.responses[p.id]?.[v.id], unit: v.unit }), label: p.name, fmt: (o) => cell(o.r, o.unit) })),
    ], (dist.variables || []).filter(v => v.id !== 'labor_share')),
    h('p', { class: 'causality-note' }, `${lpH} 季后的累计响应；括号内为 ${Math.round((dist.band ?? 0.9) * 100)}% 区间，★ = 区间不含 0，灰色 = 不显著。`));
  }

  // ======================================================== asset betas by period
  root.append(h('div', { class: 'section-title' }, '资产对宏观因子的敏感度（分时期 β）'));
  const betaBox = ui.chartBox();
  const flipWrap = h('div');
  const r2Line = h('p', { class: 'causality-note' });
  const bPeriods = betas?.periods || [];
  let betaSel = bPeriods.some(p => p.id === 'full') ? 'full' : bPeriods[0]?.id;
  const subs = bPeriods.filter(p => p.id !== 'full');
  const bAssets = betas?.assets || [], bFactors = betas?.factors || [];
  const bAt = (pid, ai, fi) => betas?.beta_std?.[pid]?.[ai]?.[fi];
  const tAt = (pid, ai, fi) => betas?.tstat?.[pid]?.[ai]?.[fi];
  // regime changes across the sub-periods: significant sign flips, weaker sign changes, fading and emerging links
  function classify(ai, fi) {
    const bs = subs.map(p => bAt(p.id, ai, fi)), ts = subs.map(p => tAt(p.id, ai, fi));
    if (!bs.every(isNum) || !ts.every(isNum) || bs.length < 2) return null;
    const sig = (k, tmin, bmin) => Math.abs(ts[k]) >= tmin && Math.abs(bs[k]) >= bmin;
    const idx = bs.map((_, k) => k);
    // flip: significant on both sides (one at 5%, the other at least at 10%)
    for (const k of idx) {
      if (sig(k, 1.96, 0.05) && idx.some(j => Math.sign(bs[j]) === -Math.sign(bs[k]) && sig(j, 1.65, 0.05))) return 'flip';
    }
    for (const k of idx) {
      if (!sig(k, 1.96, 0.1)) continue;
      if (idx.some(j => j !== k && Math.sign(bs[j]) !== Math.sign(bs[k]) && sig(j, 1.0, 0.05))) return 'turn';
    }
    const last = bs.length - 1;
    const early = idx.slice(0, last);
    const earlyMax = Math.max(...early.map(k => Math.abs(bs[k])));
    if (early.some(k => sig(k, 1.96, 0.15)) && Math.abs(ts[last]) < 1 && Math.abs(bs[last]) < 0.5 * earlyMax) return 'fade';
    if (sig(last, 2.5, 0.15) && early.every(k => Math.abs(ts[k]) < 1.65)) return 'emerge';
    return null;
  }
  const lastYear = String(subs[subs.length - 1]?.start || '').slice(0, 4);
  const KIND = { flip: ['显著反号', 'bad'], turn: ['方向改变', 'warn'],
    fade: [`${lastYear} 后消失`, 'accent'], emerge: [`${lastYear} 后出现`, 'good'] };
  const flags = [];
  bAssets.forEach((a, ai) => bFactors.forEach((f, fi) => { const k = classify(ai, fi); if (k) flags.push({ ai, fi, asset: a, factor: f, kind: k }); }));
  const flagAt = (ai, fi) => flags.find(x => x.ai === ai && x.fi === fi)?.kind;

  if (betas?.beta_std) {
    root.append(card({ title: '标准化 β：因子动 1 个标准差，资产当月价格回报动几个标准差',
      sub: '红 = 同向，蓝 = 反向，* 表示 |t| ≥ 1.96（5% 显著）；黑实线框 = 在不同时期显著反号，虚线框 = 方向改变：同一个因子对资产的作用会随时代翻转，这正是只用一个全样本模型会出错的地方。' },
    h('div', { class: 'causality-toolbar' }, h('span', { class: 'causality-group' }, h('span', { class: 'lbl' }, '估计时期'),
      ui.seg(bPeriods.map(p => ({ value: p.id, label: periodLabel(p) })), betaSel, (v) => { betaSel = v; drawBetas(); }))),
    ui.wideChart(betaBox, 520), r2Line, flipWrap));          // 13 asset columns: a phone scrolls sideways
  } else {
    root.append(ui.notice('causality.json 里没有资产 β。', 'info'));
  }

  function drawBetas() {
    const per = bPeriods.find(p => p.id === betaSel);
    const rows = bFactors.map(f => ui.factorName(f)), cols = bAssets.map(a => ui.assetName(a));
    const values = bFactors.map((f, fi) => bAssets.map((a, ai) => bAt(betaSel, ai, fi) ?? null));
    heat(charts, betaBox, { rows, cols, values, min: -0.5, max: 0.5, height: rows.length * 32 + 110,
      labelFmt: (v, r, c) => `${fixed(v)}${Math.abs(tAt(betaSel, c, r) ?? 0) >= 1.96 ? '*' : ''}`,
      cellStyle: (r, c) => {
        const k = flagAt(c, r);
        if (k === 'flip') return { borderColor: INK, borderWidth: 2 };
        if (k === 'turn') return { borderColor: '#495057', borderWidth: 1.5, borderType: 'dashed' };
        return null;
      },
      tip: (r, c, v) => {
        const k = flagAt(c, r);
        const across = subs.map(p => `${periodLabel(p)} ${signed(bAt(p.id, c, r))}（t ${signed(tAt(p.id, c, r), 1)}）`).join('<br/>');
        const r2 = betas.r2?.[betaSel]?.[bAssets[c]];
        return `<b>${cols[c]}</b> ← ${rows[r]}<br/>${per ? periodLabel(per) : ''}：β <b>${signed(v)}</b>，t = ${signed(tAt(betaSel, c, r), 1)}`
          + `<br/>${across}${isNum(r2) ? `<br/>该资产模型 R² ${r2.toFixed(2)}` : ''}${k ? `<br/><b>${KIND[k][0]}</b>` : ''}`;
      } });
    const r2 = betas.r2?.[betaSel] || {};
    r2Line.textContent = `模型解释力 R²（${per ? periodLabel(per) : ''}）：` + bAssets.map(a => `${ui.assetName(a)} ${isNum(r2[a]) ? r2[a].toFixed(2) : '—'}`).join('，')
      + '。原油和 10 年期美债几乎就是油价和收益率本身（R² 接近 1），属于机械关系。';
  }

  function renderFlips() {
    const byMag = (a, b) => Math.max(...subs.map(p => Math.abs(bAt(p.id, b.ai, b.fi)))) - Math.max(...subs.map(p => Math.abs(bAt(p.id, a.ai, a.fi))));
    const pick = (kinds) => flags.filter(x => kinds.includes(x.kind)).sort((a, b) => kinds.indexOf(a.kind) - kinds.indexOf(b.kind) || byMag(a, b));
    const cols = [
      { value: (x) => x, label: '资产 / 类型', fmt: (x) => h('div', {}, ui.assetName(x.asset),
        h('div', { style: { marginTop: '3px' } }, ui.chip(KIND[x.kind][0], KIND[x.kind][1]))) },
      { value: (x) => ui.factorName(x.factor), label: '因子', wrap: true, fmt: (v) => h('div', { class: 'causality-cell-factor' }, v) },
      ...subs.map(p => ({ value: (x) => ({ b: bAt(p.id, x.ai, x.fi), t: tAt(p.id, x.ai, x.fi) }), label: periodLabel(p),
        fmt: (o) => h('span', { class: Math.abs(o.t) >= 1.96 ? ui.signCls(o.b) : 'muted' }, signed(o.b),
          h('span', { class: 'causality-t' }, `t ${signed(o.t, 1)}`)) })),
    ];
    // the 10-year yield row of the broad equity assets: how rates hit stocks in each period
    const y10 = bFactors.indexOf('y10');
    const eq = ['us_equity', 'nasdaq'].map(a => bAssets.indexOf(a)).filter(i => i >= 0);
    const verdict = (ai) => {
      const k = flagAt(ai, y10);
      return k ? `（${KIND[k][0]}）` : '';
    };
    const parts = [h('div', { class: 'section-title', style: { margin: '16px 0 8px' } }, '跨时期的符号变化')];
    if (y10 >= 0 && eq.length) {
      parts.push(h('p', { class: 'causality-info' }, h('b', {}, '利率对股票的作用：'),
        eq.map(ai => `${ui.assetName(bAssets[ai])}对 10 年期收益率的 β ${subs.map(p => `${periodLabel(p)} ${signed(bAt(p.id, ai, y10))}（t ${signed(tAt(p.id, ai, y10), 1)}）`).join(' → ')}${verdict(ai)}`).join('；'),
        '。股债相关性怎么随时代变化见“历史时代画像”页。'));
    }
    const main = pick(['flip', 'turn']), other = pick(['fade', 'emerge']);
    parts.push(main.length ? ui.table(cols, main)
      : ui.notice(betas.tstat ? '各时期之间没有显著的符号反转。' : 'causality.json 里没有 t 统计量，无法判断符号变化是否显著。', 'info'));
    if (other.length) {
      parts.push(h('details', { class: 'causality-details' },
        h('summary', {}, `关系出现或消失（${other.length} 条）：早期显著、${lastYear} 年后消失，或早期不显著、${lastYear} 年后才出现`),
        ui.table(cols, other)));
    }
    flipWrap.append(...parts);
  }

  // ======================================================== notes
  const notes = [];
  if (svar?.method) notes.push(svar.method);
  if (shocks.length) notes.push(h('span', {}, '识别约束只规定冲击后最初几个月的方向：', shocks.map(s => `${s.name}——${s.text}`).join('；'), '。满足约束的旋转有很多个（集合识别），所以区间同时包含参数不确定性和识别不确定性。'));
  if (svar?.median_target) notes.push(`点模型取 Fry-Pagan 中位数目标：全样本 ${svar.median_target.n_rotation_draws ?? '—'} 次可接受旋转中最接近逐点中位数的一个；结构冲击序列和历史分解都来自它。`);
  if (Array.isArray(svar?.shock_corr)) {
    const off = svar.shock_corr.flatMap((r, i) => r.filter((_, j) => j !== i)).filter(isNum).map(Math.abs);
    if (off.length) notes.push(`${shocks.length || '各'} 个结构冲击两两相关系数最大 ${Math.max(...off).toFixed(2)}：构造上正交，所以历史分解里各冲击的贡献可以相加。`);
  }
  if (svar?.excluded) notes.push(`${svar.excluded.note || ''}（剔除的月份：${(svar.excluded.rows || []).join('、') || '—'}）`);
  {
    // how big a 1-sd tightening is on impact in each sub-period (zero lower bound after 2008)
    const ffImpact = periods.filter(p => p.id !== 'full')
      .map(p => ({ p, v: svar?.irf?.[p.id]?.monetary?.ff?.p50?.[0] })).filter(o => isNum(o.v));
    notes.push(`1σ 冲击在不同时期大小不同${ffImpact.length ? `（例如紧缩冲击当月联邦基金利率：${ffImpact.map(o => `${periodLabel(o.p)} ${signed(o.v)}pp`).join('，')}；零利率下限时期利率本身几乎不动）` : ''}，跨时期比较力度要用标准化传导。`);
  }
  if (dist?.alternative?.note) notes.push(`劳动收入占比冲击的识别：${dist.alternative.note}因此基准识别把劳动收入占比排在同季度产出、通胀和利率之后。`);
  if (chains.some(isDescriptive)) notes.push('AI 资本开支链是描述性的：还没有可识别的外生冲击，证据只是现状和相关的历史估计。');
  notes.push('所有估计都是历史上的平均因果关系；情景引擎用它们决定“改哪些月份的概率、改多少”，不直接外推。');
  root.append(h('div', { class: 'section-title' }, '识别假设与局限'));
  root.append(card({}, h('ul', { class: 'list' }, notes.map(n => h('li', { class: 'muted' }, n)))));

  // ======================================================== charts after the DOM exists
  renderChain();
  if (svar?.irf) { renderIrf(); renderNorm(); }
  if (hd?.dates?.length && hd.contrib) {
    hdChartInst = hdChart(charts, hdBox, hd, episodes);
    if (episodes.length) {
      const keys = Object.keys(hd.contrib);
      const short = (k) => (hd.names?.[k] || k).replace(/（.*$/, '');
      hdTable.append(h('p', { class: 'causality-note' }, `${episodes.length} 次通胀高峰当月的分解（百分点；红 = 推高通胀 1pp 以上，绿 = 压低 1pp 以上；点击一行放大图表到前后 4 年）：`),
        stackable(ui, ui.table([
          { key: 'label', label: '事件' },
          { key: 'date', label: '峰值月份' },
          { value: (e) => hd.actual[e.i], label: '实际通胀', fmt: (v) => (isNum(v) ? `${v.toFixed(2)}%` : '—') },
          { value: (e) => hd.baseline?.[e.i], label: '基线', fmt: (v) => (isNum(v) ? `${v.toFixed(2)}%` : '—') },
          { value: (e) => { const k = keys.slice().sort((a, b) => Math.abs(hd.contrib[b]?.[e.i] ?? 0) - Math.abs(hd.contrib[a]?.[e.i] ?? 0))[0];
            return k && isNum(hd.contrib[k]?.[e.i]) ? `${short(k)} ${signed(hd.contrib[k][e.i])}pp` : '—'; }, label: '最大推手' },
          ...keys.map(k => ({ value: (e) => hd.contrib[k]?.[e.i], label: short(k),
            fmt: (v) => (isNum(v) ? signed(v) : '—'), cls: (v) => (isNum(v) && Math.abs(v) >= 1 ? ui.goodCls(v < 0) : '') })),
        ], episodes, { onRowClick: (e) => hdChartInst?.zoomTo(e.i) })));
    }
  }
  if (dist?.responses) { drawLp(lpGrid, dist.responses); renderLpTable(); }
  if (betas?.beta_std) { drawBetas(); renderFlips(); }
}
