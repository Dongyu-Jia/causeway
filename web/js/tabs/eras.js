// Historical eras: each era's macro fingerprint, asset returns and transmission features,
// a cross-era comparison, how unusual today's income distribution is, and which eras today resembles.
// With results/regimes.json (optional) each era also shows its macro regime mix (ex post / real time).
import { REGIME_LABELS, regimeBars, roundShares } from './_scenario_detail.js';

const INK = '#1d2433', INK3 = '#7a8396', LINE = '#e3e6ec';
const AX = {
  axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false },
  axisLabel: { color: INK3, fontSize: 11 }, splitLine: { lineStyle: { color: '#eef0f4' } },
};
const DIVERGING = ['#1c7ed6', '#a5d8ff', '#f8f9fa', '#ffc9c9', '#e03131'];
const UP = '#e03131', DOWN = '#1c7ed6', TODAY = '#e8590c';

// KPI tiles of the selected era (era means). States outside the fingerprint have no 1955+ z.
const KPI_STATES = ['cpi_yoy', 'core_yoy', 'ff', 'real_ff', 'y10', 'spread', 'unrate', 'ip_yoy',
  'labor_share', 'profit_share', 'capex_share', 'white_collar_share', 'real_wage_yoy', 'wealth_income',
  'div_yield', 'real_oil'];

// The new situation: who gets the income (labor vs capital) and how hot the capital side runs.
const NEW_STATES = [
  ['labor_share', '劳动者拿到的份额：非农企业部门劳动报酬 / 产出（指数，2017=100）'],
  ['profit_share', '资本拿到的份额：企业税后利润 / GDP'],
  ['white_collar_share', '白领岗位：专业与商业服务、信息、金融业就业占非农就业'],
  ['real_wage_yoy', '普通工人的购买力：生产与非管理岗时薪同比减 CPI 同比'],
  ['wealth_income', '资产价格相对收入：家庭净值 / 可支配收入'],
  ['capex_share', '资本开支强度：私人非住宅固定投资 / GDP'],
  ['div_yield', '估值：美股股息率越低，估值越高'],
  ['sentiment', '居民的感受：密歇根消费者信心指数'],
];

// Timeline presets: states sharing a unit go on the left axis, one odd unit on the right.
const TIMELINE_GROUPS = [
  { id: 'inflation', label: '通胀与利率', left: ['cpi_yoy', 'core_yoy', 'real_ff', 'y10'] },
  { id: 'distribution', label: '收入分配', left: ['profit_share', 'white_collar_share'], right: ['labor_share'] },
  { id: 'capital', label: '资本与估值', left: ['capex_share', 'div_yield', 'saving_rate'], right: ['wealth_income'] },
  { id: 'cycle', label: '增长与就业', left: ['ip_yoy', 'unrate', 'real_wage_yoy', 'productivity_yoy'], right: ['sentiment'] },
  { id: 'risk', label: '信用利差与油价', left: ['spread'], right: ['real_oil'] },
];
const ERA_SHORT = { golden_age: '黄金期', inflation_onset: '通胀起步', great_inflation: '大通胀', disinflation: '去通胀',
  tech_boom: '科技繁荣', china_credit: '中国冲击', stagnation_qe: 'QE', covid_reflation: '疫情', ai_capex: 'AI' };
const MIN_OIL_MOVES = 24;      // analysis/eras.py: fewer months with an oil-price move -> no pass-through
const PHILLIPS_LABEL = { wage_yoy: '工资增速', core_yoy: '核心通胀', cpi_yoy: '通胀' };
const LINE_COLORS = ['#3b5bdb', '#e8590c', '#2f9e44', '#c2255c', '#7048e8'];

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
// "+1.23" / "-1.23"; values that round to zero print as "0.00", never "-0.00"
function signed(v, d = 2) {
  if (!isNum(v)) return '—';
  const s = v.toFixed(d);
  return Number(s) === 0 ? (0).toFixed(d) : `${v > 0 ? '+' : ''}${s}`;
}
// decimal -> signed percent ("+12.6%"), with "0.0%" instead of "-0.0%"
const pctS = (v, d = 1) => (isNum(v) ? `${signed(v * 100, d)}%` : '—');
// band bound: a bound that rounds to 0.00 but is not zero keeps a third decimal, so "excludes 0" stays visible
const bound = (v) => (isNum(v) && v !== 0 && Math.abs(v) < 0.005 ? signed(v, 3) : signed(v));
// fixed decimals without a sign, never "-0.00"
const fixed = (v, d = 2) => (Number(v.toFixed(d)) === 0 ? (0).toFixed(d) : v.toFixed(d));
const signOf = (v, d = 1) => (isNum(v) && Number((v * 100).toFixed(d)) !== 0 ? (v > 0 ? 'pos' : 'neg') : '');
const DEC = { '%': 2, pp: 2, $: 1, '2017=100': 1, '倍': 2, '指数': 1 };
function fmtState(unit, v, d) {
  if (!isNum(v)) return '—';
  const n = v.toFixed(d ?? DEC[unit] ?? 2);
  // a spread level (unit 'pp') is quoted in % like the overview and the scenario fans; changes keep "pp"
  if (unit === '%' || unit === 'pp') return `${n}%`;
  if (unit === '$') return `${n} 美元`;
  if (unit === '倍') return `${n} 倍`;
  return n;
}
const axisFmt = (unit) => (v) => (unit === '%' ? `${v}%` : `${v}`);

const CSS = `
.eras-picker { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(210px, 100%), 1fr)); gap: 8px; margin-bottom: 16px; }
.eras-tile { font: inherit; text-align: left; border: 1px solid var(--line); border-radius: 8px; background: var(--panel);
  padding: 9px 11px; cursor: pointer; color: var(--ink); }
.eras-tile:hover { border-color: #c6cbd6; }
.eras-tile.on { border-color: var(--accent); background: var(--accent-soft); }
.eras-tile-name { font-weight: 600; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.eras-tile-sub { font-size: 12px; color: var(--ink-3); margin-top: 2px; }
.eras-tile-sig { font-size: 12px; color: var(--ink-2); margin-top: 4px; }
.eras-detail { display: grid; gap: 16px; }
.eras-anchor { scroll-margin-top: 76px; }
.eras-narr { margin: 0 0 10px; color: var(--ink-2); }
.eras-chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chip.eras-chip-hi { background: #fff0f0; border-color: #ffc9c9; color: #c92a2a; }
.chip.eras-chip-lo { background: #e7f5ff; border-color: #a5d8ff; color: #1864ab; }
.eras-kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(168px, 100%), 1fr)); gap: 10px; }
.eras-kpis .kpi { padding: 10px 12px; }
.eras-kpis .kpi-value { font-size: 19px; }
.eras-hi { color: #c92a2a; } .eras-lo { color: #1864ab; }
.eras-grid2 { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(440px, 100%), 1fr)); }
.eras-grid3 { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); }
.eras-grid4 { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(min(300px, 100%), 1fr)); }
.eras-metric { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: var(--panel-2); }
.eras-metric + .eras-metric { margin-top: 10px; }
.eras-metric-title { font-size: 12px; color: var(--ink-3); }
.eras-metric-value { font-size: 20px; font-weight: 650; font-variant-numeric: tabular-nums; margin: 2px 0; }
.eras-metric-text { font-size: 13px; color: var(--ink-2); }
.eras-note { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; }
.eras-sub { font-size: 11px; color: var(--ink-3); }
.eras-callout { margin: 0 0 12px; }
.eras-callout li { margin: 3px 0; }
.eras-mini .card-head { margin-bottom: 4px; }
.eras-detail .card-head .chip, .eras-tile .chip { white-space: nowrap; flex-shrink: 0; }
.eras-runs { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.eras-runs .chip { font-size: 11px; padding: 1px 7px; font-variant-numeric: tabular-nums; }
.eras-runs .chip i, .eras-rgmix i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; }
.eras-rgmix { font-size: 12px; color: var(--ink-2); margin: 4px 0 0; line-height: 1.7; }
.eras-rgmix span { white-space: nowrap; margin-right: 8px; }
.eras-toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; }
/* a wide table collapses into labelled blocks when its card is narrow */
.eras-stackbox { container-type: inline-size; }
@container (max-width: 760px) {
  .eras-stack table.tbl thead { display: none; }
  .eras-stack table.tbl, .eras-stack table.tbl tbody, .eras-stack table.tbl td { display: block; width: auto; }
  .eras-stack table.tbl tr { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(220px, 100%), 1fr));
    column-gap: 18px; padding: 10px 2px; border-bottom: 1px solid var(--line); }
  .eras-stack table.tbl td { border: 0; padding: 2px 0; text-align: left; white-space: normal; }
  .eras-stack table.tbl td:first-child { grid-column: 1 / -1; font-weight: 600; }
  .eras-stack table.tbl td::before { content: attr(data-label); display: inline-block; min-width: 92px;
    margin-right: 8px; font-size: 11px; color: var(--ink-3); vertical-align: top; }
  .eras-stack table.tbl td:first-child::before { display: none; }
  .eras-stack table.tbl td > div { display: inline-block; vertical-align: top; }
}
`;

function injectStyle() {
  let el = document.getElementById('style-fe1');
  if (!el) { el = document.createElement('style'); el.id = 'style-fe1'; document.head.append(el); }
  if (!el.dataset.eras) { el.append(document.createTextNode(CSS)); el.dataset.eras = '1'; }
}

// Copies the header text onto every cell (data-label) so the table can collapse into blocks.
function stackable(ui, wrap) {
  const heads = [...wrap.querySelectorAll('thead th')].map(th => th.textContent);
  wrap.querySelectorAll('tbody tr').forEach(tr => [...tr.children].forEach((td, i) => { td.dataset.label = heads[i] || ''; }));
  wrap.classList.add('eras-stack');
  return ui.h('div', { class: 'eras-stackbox' }, wrap);
}

// replaceChildren that skips null / false / undefined (the DOM would print them as text).
function fill(el, ...kids) { el.replaceChildren(...kids.flat().filter(k => k !== null && k !== undefined && k !== false)); return el; }

// Charts inside a container that is about to be emptied.
function disposeCharts(el) {
  el.querySelectorAll('.chart').forEach((c) => {
    if (c._ro) c._ro.disconnect();
    if (c._chart) c._chart.dispose();
    c._chart = null; c._ro = null;
  });
}

function loadError(ui, err) {
  const { h } = ui;
  if (err.status === 404) {
    return h('div', {}, ui.notice('还没有时代画像结果。', 'info'), ui.buildHint(err));
  }
  return ui.notice(`读取时代画像失败：${err.message}`, 'error');
}

// Null-safe heatmap with per-cell styles and a custom tooltip. values[r][c].
function heat(charts, el, { rows, cols, values, min, max, labelFmt, tip, cellStyle, height, colors = DIVERGING, boldRow, minCell = 26 }) {
  if (height) el.style.height = `${height}px`;
  const chart = charts.mount(el);
  const data = [];
  let lo = Infinity, hi = -Infinity;
  values.forEach((row, r) => row.forEach((v, c) => {
    const ok = isNum(v);
    if (ok) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const item = { value: [c, r, ok ? v : '-'] };
    const st = cellStyle && cellStyle(r, c, ok ? v : null);
    if (st) item.itemStyle = st;
    data.push(item);
  }));
  const bound = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const rowChars = Math.max(...rows.map(s => String(s).length), 2);
  const cellW = ((el.clientWidth || 600) - rowChars * 11 - 30) / Math.max(cols.length, 1);
  const colPx = Math.max(10, ...cols.map(s => [...String(s)].reduce((t, ch) => t + (/[\x00-\xff]/.test(ch) ? 6.6 : 11), 0)));
  const rotate = colPx <= cellW - 4 ? 0 : cellW >= 24 ? 40 : 90;
  const interval = rotate === 90 && cellW < 13 ? Math.ceil(13 / Math.max(cellW, 1)) - 1 : 0;
  chart.setOption({
    animation: false,
    tooltip: { confine: true, formatter: (p) => {
      const [c, r, v] = p.value;
      return tip ? tip(r, c, isNum(v) ? v : null) : `${rows[r]} × ${cols[c]}<br/><b>${isNum(v) ? labelFmt(v) : '无数据'}</b>`;
    } },
    grid: { left: 8, right: 16, top: 8, bottom: 8, containLabel: true },
    // narrow columns get steeper column labels, otherwise neighbouring labels run into each other
    xAxis: { type: 'category', data: cols, ...AX, splitLine: { show: false },
      axisLabel: { ...AX.axisLabel, rotate, interval } },
    yAxis: { type: 'category', data: rows, inverse: true, ...AX, splitLine: { show: false },
      axisLabel: { ...AX.axisLabel, interval: 0, color: INK,
        formatter: (v, i) => (i === boldRow ? `{b|${v}}` : v), rich: { b: { fontWeight: 'bold', color: TODAY, fontSize: 11 } } } },
    visualMap: { min: min ?? -bound, max: max ?? bound, show: false, calculable: false, dimension: 2, inRange: { color: colors } },
    series: [{ type: 'heatmap', data,
      label: { show: cellW >= minCell, fontSize: 10, color: INK, formatter: (p) => (isNum(p.value[2]) ? labelFmt(p.value[2]) : '') },
      emphasis: { itemStyle: { borderColor: INK, borderWidth: 1 } } }],
  });
  return chart;
}

// Horizontal bars with per-bar colours and value labels (so the value axis is hidden), an
// optional dashed reference line and a custom tooltip. text(i) is the label of bar i (default
// fmt(value)), so bars can show a deviation while the label shows the level. The axis is padded
// so labels of negative bars do not run into the category labels.
function hbars(charts, el, { labels, values, colors, fmt, text, ref, tip, nullText, height }) {
  if (height) el.style.height = `${height}px`;
  const chart = charts.mount(el);
  const lab = (i) => String(text ? text(i) : fmt(values[i]));
  const data = values.map((v, i) => (isNum(v)
    ? { value: v, itemStyle: { color: colors?.[i] || '#3b5bdb', borderRadius: 2 },
      label: { show: true, position: v < 0 ? 'left' : 'right', formatter: lab(i), color: INK3, fontSize: 10 } }
    : { value: 0, itemStyle: { color: 'transparent' },
      label: { show: true, position: 'right', formatter: (typeof nullText === 'function' ? nullText(i) : nullText) || '—', color: INK3, fontSize: 10 } }));
  // reserve room (in axis units) for the value labels beyond the bar ends, sized in pixels
  const xs = values.filter(isNum).concat(isNum(ref) ? [ref] : []);
  let lo = Math.min(0, ...xs), hi = Math.max(0, ...xs);
  const span = (hi - lo) || 1;
  const labelPx = Math.max(4, ...values.map((v, i) => (isNum(v) ? lab(i).length : 0))) * 6.3 + 8;
  const catPx = Math.max(2, ...labels.map(s => String(s).length)) * 11 + 18;
  const plotPx = Math.max(80, (el.clientWidth || 320) - catPx - 24);
  const sides = (lo < 0 ? 1 : 0) + (hi > 0 ? 1 : 0);
  const f = Math.min(0.42, labelPx / plotPx);
  const total = span / Math.max(0.16, 1 - f * sides);
  if (lo < 0) lo -= f * total;
  hi += hi > 0 ? f * total : 0.04 * total;
  chart.setOption({
    animation: false,
    grid: { left: 8, right: 12, top: 8, bottom: 6, containLabel: true },
    tooltip: { trigger: 'item', confine: true,
      formatter: (p) => (tip ? tip(p.dataIndex) : `${labels[p.dataIndex]}<br/><b>${isNum(values[p.dataIndex]) ? lab(p.dataIndex) : '—'}</b>`) },
    xAxis: { type: 'value', min: lo, max: hi, ...AX, axisLabel: { show: false }, splitNumber: 4 },
    yAxis: { type: 'category', data: labels, inverse: true, ...AX, splitLine: { show: false },
      axisLabel: { ...AX.axisLabel, interval: 0, color: INK } },
    series: [{ type: 'bar', data, barMaxWidth: 16,
      markLine: isNum(ref) ? { silent: true, symbol: 'none', lineStyle: { color: INK3, type: 'dashed' },
        label: { show: false }, data: [{ xAxis: ref }] } : undefined }],
  });
  return chart;
}

// Monthly state lines since 1955 with era bands; zMode standardizes every line by its 1955+ mean / sd.
function timelineChart(charts, el, { dates, lines, bands, zMode, unitLeft, unitRight, eraAt }) {
  const chart = charts.mount(el);
  const hasRight = !zMode && lines.some(l => l.right);
  const series = lines.map(l => ({
    name: hasRight && l.right ? `${l.name}（右轴）` : l.name, type: 'line', data: l.data, yAxisIndex: hasRight && l.right ? 1 : 0, symbol: 'none', showSymbol: false,
    lineStyle: { width: 1.5, color: l.color }, itemStyle: { color: l.color },
  }));
  if (series.length) {
    series[0].markArea = { silent: true, data: bands.map((b, i) => [{
      name: ERA_SHORT[b.id] || b.name, xAxis: b.start,
      itemStyle: { color: i % 2 ? 'rgba(122, 131, 150, 0.08)' : 'rgba(59, 91, 219, 0.07)' },
      label: { show: true, color: INK3, fontSize: 10, position: 'insideTopLeft', offset: [2, i % 2 ? 14 : 0] },
    }, { xAxis: b.end }]) };
    if (zMode) {
      series[0].markLine = { silent: true, symbol: 'none', lineStyle: { color: INK3, type: 'dashed' },
        label: { formatter: '1955 年以来均值', color: INK3, fontSize: 10, position: 'insideStartTop' }, data: [{ yAxis: 0 }] };
    }
  }
  const yAxis = [{ type: 'value', name: zMode ? '标准差（z）' : unitLeft, nameTextStyle: { color: INK3, fontSize: 11 },
    scale: true, ...AX, axisLabel: { ...AX.axisLabel, formatter: zMode ? (v) => `${v}` : axisFmt(unitLeft) } }];
  if (hasRight) {
    yAxis.push({ type: 'value', name: unitRight, position: 'right', scale: true, nameTextStyle: { color: INK3, fontSize: 11 },
      ...AX, splitLine: { show: false } });
  }
  chart.setOption({
    animation: false,
    legend: { top: 0, type: 'scroll', textStyle: { color: INK, fontSize: 12 }, itemWidth: 14, itemHeight: 8 },
    tooltip: { trigger: 'axis', confine: true, formatter: (ps) => {
      const i = ps[0].dataIndex;
      const head = `<b>${dates[i]}</b>　${eraAt(dates[i]) || ''}`;
      return [head, ...lines.map(l => {
        const raw = fmtState(l.unit, l.raw[i]);
        return `${l.name}：${zMode ? `z ${signed(l.data[i])}（${raw}）` : raw}`;
      })].join('<br/>');
    } },
    grid: { left: 56, right: hasRight ? 60 : 24, top: 52, bottom: 56 },
    xAxis: { type: 'category', data: dates, boundaryGap: false, ...AX, splitLine: { show: false } },
    yAxis,
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }],
    series,
  });
  return chart;
}

// Percentile (mid-rank), extremes and their months of value v within a series.
function histStats(series, dates, v) {
  let n = 0, below = 0, eq = 0, mn = Infinity, mx = -Infinity, imn = -1, imx = -1;
  (series || []).forEach((x, i) => {
    if (!isNum(x)) return;
    n += 1;
    if (x < v) below += 1; else if (x === v) eq += 1;
    if (x < mn) { mn = x; imn = i; }
    if (x > mx) { mx = x; imx = i; }
  });
  if (!n || !isNum(v)) return null;
  return { pct: (below + 0.5 * eq) / n, min: mn, max: mx, minDate: dates[imn], maxDate: dates[imx],
    isMin: v <= mn + 1e-9, isMax: v >= mx - 1e-9 };
}

export async function render(root, { api, charts, ui, meta, param, alive = () => root.isConnected }) {
  const { h, card } = ui;
  injectStyle();
  root.append(h('p', { class: 'lead' },
    '每个时代都有自己的宏观“指纹”：通胀、利率、就业、收入分配和估值的组合不同，资产的赢家与输家、股债关系、油价对通胀的传导也随之改变。',
    '先选一个时代看细节，再横向对比所有时代，最后看今天哪些地方是 1955 年以来从未出现过的。'));

  let data;
  const regP = api.optional('regimes', meta);
  try {
    data = await api.get('eras');
  } catch (err) {
    root.append(loadError(ui, err));
    return;
  }
  const rg = await regP;
  if (!alive()) return;                  // the user moved on while the results loaded
  const eras = data.eras || [];
  if (!eras.length) { root.append(ui.notice('时代画像结果里没有时代记录。', 'info')); return; }
  // regime mixes per era (regimes.json eras; ids match eras.json, plus 1948–54 before the first era)
  const rgEras = Array.isArray(rg?.eras) ? rg.eras : [];
  const rgById = Object.fromEntries(rgEras.map(x => [x.id, x]));
  const rgLabels = Array.isArray(rg?.definitions?.labels) && rg.definitions.labels.length ? rg.definitions.labels : REGIME_LABELS;
  const rgName = (id) => rgLabels.find(l => l.id === id)?.name || id;
  const rgColor = (id) => rgLabels.find(l => l.id === id)?.color || '#adb5bd';
  const rgByName = Object.fromEntries(rgLabels.map(l => [l.name, l]));
  const pct0 = (x) => `${Math.round(x * 100)}%`;
  // a mix in whole percents that add up to 100 (largest remainder), labels from 1% up
  const mixLine = (m) => {
    const n = roundShares(m, rgLabels.map(l => l.id));
    return rgLabels.filter(l => n[l.id] >= 1)
      .map(l => h('span', {}, h('i', { style: { background: l.color, marginRight: '4px' } }), `${l.name} ${n[l.id]}%`));
  };
  const byId = Object.fromEntries(eras.map(e => [e.id, e]));
  const fp = data.fingerprint || null;
  const tl = data.timeline || null;
  const today = meta?.today || null;

  // 1955+ reference mean / sd per fingerprint state (z = (x - mean) / sd)
  const ref = {};
  (fp?.cols || []).forEach((c, j) => {
    ref[c] = { mean: fp.ref_mean?.[j], sd: fp.ref_sd?.[j], unit: fp.col_units?.[j], name: fp.col_names?.[j] };
  });
  const unitOf = (s) => ref[s]?.unit ?? ui.META.states[s]?.unit ?? '';
  const nameOf = (s) => ref[s]?.name || tl?.names?.[s] || ui.stateName(s);
  const zOf = (s, v) => (isNum(v) && isNum(ref[s]?.mean) && ref[s].sd > 0 ? (v - ref[s].mean) / ref[s].sd : null);
  const eraAt = (d) => (eras.find(e => d >= e.start && d <= e.end) || {}).name;
  const assetIds = (ui.META.raw?.assets || []).map(a => a.id);
  const allAssets = assetIds.length ? assetIds : [...new Set(eras.flatMap(e => Object.keys(e.assets || {})))];

  // ======================================================== selected era
  root.append(h('div', { class: 'section-title' }, '选择一个时代'));
  let sel = byId[param] ? param : (eras.find(e => e.ongoing) || eras[eras.length - 1]).id;
  const picker = h('div', { class: 'eras-picker eras-anchor' });
  const detail = h('div', { class: 'eras-detail eras-anchor' });
  root.append(picker, detail);

  function drawPicker() {
    picker.innerHTML = '';
    for (const e of eras) {
      const sig = (e.signature || [])[0];
      picker.append(h('button', { type: 'button', class: `eras-tile${e.id === sel ? ' on' : ''}`, onclick: () => select(e.id) },
        h('div', { class: 'eras-tile-name' }, e.name, e.ongoing ? ui.chip('进行中', 'warn') : null),
        h('div', { class: 'eras-tile-sub' }, `${e.start} 至 ${e.ongoing ? '今' : e.end}（${e.n_months} 个月）`),
        sig ? h('div', { class: 'eras-tile-sig' }, String(sig.text || '').split('，')[0]) : null));
    }
  }

  function select(id, scroll = false) {
    if (!byId[id]) return;
    sel = id;
    drawPicker();
    renderEra(byId[id]);
    ui.setRoute('eras', id, alive);
    if (scroll) picker.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function sigChip(s) {
    const phrase = String(s.text || s.name || s.state).split('，')[0];
    return h('span', { class: `chip ${s.z > 0 ? 'eras-chip-hi' : 'eras-chip-lo'}`, title: s.text || '' }, `${phrase} · z ${signed(s.z, 1)}`);
  }

  function eraKpi(e, sid) {
    const st = e.states?.[sid];
    const unit = unitOf(sid);
    if (!st) return ui.kpi(nameOf(sid), '—', '该时期数据覆盖不足');
    const z = zOf(sid, st.mean);
    const sub = h('div', {}, `期初 ${fmtState(unit, st.start)} → 期末 ${fmtState(unit, st.end)}`,
      z === null ? null : h('br'), z === null ? null : `z ${signed(z)}（长期均值 ${fmtState(unit, ref[sid].mean)}）`);
    return ui.kpi(nameOf(sid), fmtState(unit, st.mean), sub, z !== null && Math.abs(z) >= 1 ? (z > 0 ? 'eras-hi' : 'eras-lo') : '');
  }

  function metric(title, value, text, cls = '') {
    return h('div', { class: 'eras-metric' }, h('div', { class: 'eras-metric-title' }, title),
      h('div', { class: `eras-metric-value ${cls}` }, value), h('div', { class: 'eras-metric-text' }, text));
  }

  // why the oil -> CPI pass-through is missing (mirrors analysis/eras.py)
  function passthroughWhy(e) {
    const moves = e.oil_price_moves;
    if (isNum(moves) && moves < MIN_OIL_MOVES) {
      const why = moves < 0.5 * e.n_months
        ? `这一时期 WTI 是管制挂牌价，${e.n_months} 个月里只有 ${moves} 个月油价有变动`
        : `油价有变动的月份太少（${moves} 个月）`;
      return `${why}；至少需要 ${MIN_OIL_MOVES} 个月，否则回归只由一两次调价事件决定。`;
    }
    return '样本太短，无法可靠估计。';
  }

  // median factor volatility across eras, the yardstick for "this era was volatile in X"
  const medVol = {};
  for (const e of eras) {
    for (const [f, v] of Object.entries(e.factors || {})) if (isNum(v.vol_annual)) (medVol[f] = medVol[f] || []).push(v.vol_annual);
  }
  for (const f of Object.keys(medVol)) {
    const xs = medVol[f].sort((a, b) => a - b);
    medVol[f] = xs.length % 2 ? xs[(xs.length - 1) / 2] : 0.5 * (xs[xs.length / 2 - 1] + xs[xs.length / 2]);
  }
  const factorName = (id) => (id === 'cpi' ? 'CPI 通胀' : ui.factorName(id));
  function fmtFactor(id, v, sgn) {
    if (!isNum(v)) return '—';
    const unit = id === 'cpi' ? 'Δlog' : ui.META.factors[id]?.unit;
    if (unit === 'Δlog') return sgn ? pctS(v) : ui.fmtPct(v, 1);
    if (unit === 'Δpp') return `${sgn ? signed(v) : v.toFixed(2)}pp`;
    return sgn ? signed(v) : v.toFixed(2);
  }

  // ---- transmission attribution: each era's average annual price return of one asset, split along the
  // transmission the engine uses (full-sample betas x the era's actual factor changes), the intercept and
  // the residual (what that transmission does not explain in the era), plus income (analysis/eras.py)
  const ainfo = data.attribution_info || {};
  const attrEras = eras.filter(x => x.attribution && Object.keys(x.attribution.assets || {}).length);
  const noAttr = eras.filter(x => !attrEras.includes(x));
  const ATTR_COLS = [
    { key: 'oil', label: '油价' }, { key: 'ip', label: '工业产出' }, { key: 'inflation', label: '通胀', parts: ['cpi_goods', 'cpi_services'] },
    { key: 'ff', label: '政策利率' }, { key: 'y10', label: '10 年期收益率' }, { key: 'spread', label: '信用利差' },
    { key: 'usd', label: '美元' }, { key: 'unrate', label: '失业率' }, { key: 'labor_share', label: '劳动收入占比' },
  ];
  const ATTR_TAIL = [
    { key: 'factor_sum', label: '＝ 因子合计', total: true },
    { key: 'intercept', label: '常数项' }, { key: 'unattributed', label: '未归因的传导' },
    { key: 'residual', label: '未解释部分', resid: true },
    { key: 'price_mean', label: '＝ 价格回报', total: true }, { key: 'income', label: '收入' },
  ];
  // the engine's factor list may differ from the core ten (a future selected model): extra factors get their own column
  const extraFactors = [...new Set(attrEras.flatMap(x => Object.values(x.attribution.assets || {})
    .flatMap(r => Object.keys(r.terms || {}))))]
    .filter(k => !['intercept', 'unattributed', 'residual', 'income', 'cpi_goods', 'cpi_services'].includes(k) && !ATTR_COLS.some(c => c.key === k));
  const attrCols = [...ATTR_COLS, ...extraFactors.map(k => ({ key: k, label: k.startsWith('e_') ? ui.varName(k) : ui.factorName(k) })), ...ATTR_TAIL]
    .filter(c => c.key === 'factor_sum' || c.key === 'price_mean'
      || attrEras.some(x => Object.values(x.attribution.assets || {}).some(r => (c.parts || [c.key]).some(p => isNum(r.terms?.[p])))));
  const cellOf = (r, c) => {
    if (!r) return null;
    if (c.key === 'factor_sum' || c.key === 'price_mean') return isNum(r[c.key]) ? r[c.key] : null;
    const vs = (c.parts || [c.key]).map(p => r.terms?.[p]).filter(isNum);
    return vs.length ? vs.reduce((a, b) => a + b, 0) : null;
  };
  // the model the eras step used vs the one the scenarios run (meta.model)
  const specKey = (s) => (s ? JSON.stringify([s.factors, s.lags, s.regime, s.kind, s.ridge, s.window_years, s.target, s.shock_cols,
    s.asset_factors || {}, s.asset_start || {}]) : '');
  const modelMismatch = ainfo.model?.spec && meta?.model?.spec && specKey(ainfo.model.spec) !== specKey(meta.model.spec);
  let attrAsset = 'us_equity';

  function attributionCard(e) {
    const box = ui.chartBox();
    const text = h('div');
    const assetsWith = allAssets.filter(a => attrEras.some(x => x.attribution.assets?.[a]));
    if (!assetsWith.includes(attrAsset)) attrAsset = assetsWith.includes('us_equity') ? 'us_equity' : assetsWith[0];
    const sub = '每格 = 这一时代该资产平均每年的价格回报里，有多少来自这一列（%/年，月均 ×12，一行严格相加）：因子列 = 引擎所用传导模型的全样本β × 该时代因子的实际平均变化；'
      + '“未解释部分”是这套传导在该时代解释不了的回报（泡沫、崩盘、估值重估都在这里），如实列出；价格回报加上收入是总回报。'
      + '颜色按 K 线配色（涨色为正）。';
    if (!attrEras.length || !assetsWith.length) {
      return { el: card({ title: '传导归因：各时代的价格回报从哪条渠道来', sub },
        ui.notice('时代画像结果里还没有传导归因：运行 python -m scenariolab.build --only eras 后可见。', 'info')), draw: () => {} };
    }
    const pick = ui.select(assetsWith.map(a => ({ value: a, label: ui.assetName(a) })), attrAsset, (v) => { attrAsset = v; draw(); });
    const el = card({ title: '传导归因：各时代的价格回报从哪条渠道来', sub, cls: 'eras-anchor',
      actions: h('div', { class: 'eras-toolbar', style: { marginBottom: 0 } }, h('span', { class: 'muted', style: { fontSize: '12px' } }, '资产'), pick) },
    modelMismatch ? ui.notice(`时代归因用的传导模型（${ainfo.model.describe || '—'}）和当前情景引擎（${meta.model.describe || '—'}）不同：运行 python -m scenariolab.build --only eras 让两者一致。`, 'error') : null,
    text, ui.wideChart(box, 760),
    h('p', { class: 'eras-note' }, `传导模型：${ainfo.model?.describe || '—'}，用 ${ainfo.model?.train_start || '1973'} 至 ${ainfo.model?.train_end || '今'} 的全部月份估计（全样本β）。`
      + (noAttr.length ? `${noAttr.map(x => x.name).join('、')}不做归因：${noAttr[0].attribution?.reason || '缺少因子数据'}。` : '')
      + '时期内的复利与波动修正（复利的抬升减去波动拖累，可正可负）见悬停提示。'));

    function draw() {
      const rows = attrEras.map(x => x.attribution.assets?.[attrAsset] || null);
      const values = rows.map(r => attrCols.map(c => cellOf(r, c)));
      const selRow = attrEras.findIndex(x => x.id === e.id);
      // betas of the scenario model (meta.model); only used to explain a cell when it is the eras' model too
      const coef = modelMismatch ? {} : (meta?.model?.coef?.[attrAsset] || {});
      const fmtF = (k) => (ui.META.factors[k] ? (v) => fmtFactor(k, v, true) : (v) => signed(v, 3));
      heat(charts, box, {
        rows: attrEras.map(x => ERA_SHORT[x.id] || x.name), cols: attrCols.map(c => c.label), values, min: -0.1, max: 0.1,
        height: attrEras.length * 30 + 110, labelFmt: (v) => pctS(v), minCell: 40, colors: charts.klineDiverging(),
        boldRow: selRow >= 0 ? selRow : undefined,
        cellStyle: (ri, ci) => (attrCols[ci].total ? { borderColor: '#1d2433', borderWidth: 1 }
          : attrCols[ci].resid ? { borderColor: '#868e96', borderWidth: 1, borderType: 'dashed' } : null),
        tip: (ri, ci, v) => {
          const x = attrEras[ri], r = rows[ri], c = attrCols[ci], at = x.attribution;
          if (!r) return `<b>${x.name}</b><br/>${ui.assetName(attrAsset)}在这一时期没有完整数据`;
          const head = `<b>${x.name}</b> · ${ui.assetName(attrAsset)} · ${c.label}<br/><b>${isNum(v) ? pctS(v) : '—'}/年</b>`;
          const span = `<br/><span style="color:${INK3}">${at.months ? `${at.months[0]} 至 ${at.months[1]}，` : ''}${r.n_months} 个月</span>`;
          // contribution = beta x the era's mean annual change over the attributed months (= contribution / beta)
          const parts = (c.parts || [c.key]).filter(k => isNum(r.terms?.[k]) && isNum(coef[k]) && Math.abs(coef[k]) > 1e-12);
          const why = parts.length && !c.total && c.key !== 'intercept' && c.key !== 'residual'
            ? `<br/>${parts.map(k => `${ui.factorName(k)}：β ${signed(coef[k], 3)} × 时期平均变化 ${fmtF(k)(r.terms[k] / coef[k])}/年 = ${pctS(r.terms[k])}`).join('<br/>')}` : '';
          const extra = c.key === 'price_mean'
            ? `<br/>+ 收入 ${pctS(r.terms?.income)} = 总回报（算术）${pctS(r.total_mean)}<br/>复利与波动修正 ${pctS(r.reconciliation)} → 年化回报 ${pctS(r.cagr)}`
            : c.key === 'residual' ? '<br/>这套传导在该时期解释不了的部分（残差的时期均值）' : c.key === 'intercept' ? '<br/>所有因子变动为零时模型给出的价格回报（每个时期都一样）' : '';
          return head + why + extra + span;
        },
      });
      // the selected era in words
      const at = e.attribution || {};
      const r = at.assets?.[attrAsset];
      if (!r) {
        fill(text, h('p', { class: 'eras-narr' }, at.reason ? `${e.name}不做传导归因：${at.reason}。`
          : `${e.name}：${ui.assetName(attrAsset)}在这一时期没有完整数据，无法归因。`));
        return;
      }
      const b = (v) => h('b', { class: signOf(v) }, pctS(v));
      const head = `${e.name} · ${ui.assetName(attrAsset)}（${at.months ? `${at.months[0]} 至 ${at.months[1]}，` : ''}${r.n_months} 个月）：`;
      // fewer months than the era (大通胀: 1973-01 has no dollar index): the CAGR is not the asset table's
      const tab = e.assets?.[attrAsset];
      const cover = r.n_months < e.n_months && isNum(tab?.cagr)
        ? `（归因只用因子齐全的 ${r.n_months} 个月；上面资产表按全部 ${e.n_months} 个月算的年化回报是 ${pctS(tab.cagr)}）` : '';
      if (r.kind === 'cash') {
        fill(text, h('p', { class: 'eras-narr' }, head, '没有价格变动，回报全部是利息收入，平均每年 ', b(r.terms?.income),
          `（算术），复利 ${pctS(r.reconciliation)}，年化回报 `, b(r.cagr), `${cover}。`));
        return;
      }
      const chAll = attrCols.filter(c => !ATTR_TAIL.includes(c)).map(c => ({ label: c.label, v: cellOf(r, c) }))
        .filter(c => isNum(c.v)).sort((x, y) => Math.abs(y.v) - Math.abs(x.v));
      // the three largest factor channels, and the rest, so the list adds up to the factor total
      const ch = chAll.filter(c => Math.abs(c.v) >= 0.0005).slice(0, 3);
      const rest = isNum(r.factor_sum) ? r.factor_sum - ch.reduce((s, c) => s + c.v, 0) : null;
      const chText = ch.map(c => `${c.label} ${pctS(c.v)}`).concat(isNum(rest) && Math.abs(rest) >= 0.0005 ? [`其余 ${pctS(rest)}`] : []).join('、');
      const ob = r.own_beta;
      const linear = isNum(r.factor_sum);
      // treasury and crude oil: the scenario engine builds them from the yield / oil path, not from this regression
      const structural = { treasury_10y: ' 10 年期收益率', crude_oil: '油价' }[attrAsset];
      fill(text, h('p', { class: 'eras-narr' },
        head, '价格回报平均每年 ', b(r.price_mean),
        linear ? [' ＝ 因子 ', b(r.factor_sum), ch.length ? `（${chText}）` : '', ' ＋ 常数项 ', b(r.terms?.intercept)]
          : [' ＝ 传导（未归因） ', b(r.terms?.unattributed)],
        ' ＋ 未解释部分 ', b(r.terms?.residual), '；加上收入 ', b(r.terms?.income), ' 为总回报 ', b(r.total_mean),
        `（算术），复利与波动修正 ${pctS(r.reconciliation)}，年化回报 `, b(r.cagr), `${cover}。`,
        isNum(r.r2_in_era) ? `全样本传导解释了这一时期月度价格波动的 ${Math.max(0, r.r2_in_era * 100).toFixed(0)}%（R² ${fixed(r.r2_in_era)}）` : '',
        ob ? `；用这一时期自己的β，R² ${isNum(ob.r2) ? fixed(ob.r2) : '—'}，因子合计 ${pctS(ob.factor_sum)}、常数项 ${pctS(ob.intercept)}（时期内的常数项会吸收因子的平均效应，这个拆分只作对照）。`
          : (isNum(r.r2_in_era) ? `；这一时期太短（${r.n_months} 个月），不单独估计它自己的β。` : ''),
        structural ? `${ui.assetName(attrAsset)}在情景引擎里不用这套回归，而是按模拟的${structural}路径结构化构建，这里的β拆分只作历史对照。` : ''));
    }
    return { el, draw };
  }

  function renderEra(e) {
    disposeCharts(detail);
    detail.innerHTML = '';
    detail.append(card({
      title: `${e.name}（${e.start} 至 ${e.end}）`,
      sub: `${e.n_months} 个月${e.ongoing ? '，时代仍在进行，统计截至最新数据月份' : ''}；下方 z 值 = 时期均值比 1955 年以来均值高或低几个标准差，绝对值越大越有“时代特色”。`,
      actions: e.ongoing ? ui.chip('进行中', 'warn') : null,
    },
    h('p', { class: 'eras-narr' }, e.narrative || e.summary || '—'),
    h('div', { class: 'eras-chips' }, h('span', { class: 'muted', style: { fontSize: '12px' } }, '数据指纹（最突出的状态）：'),
      // the release-month copy of a quarterly state (profit_share_rt) next to the state itself is the same fact twice
      (e.signature || []).filter((s, _, all) => !(/_rt$/.test(s?.state || '') && all.some(o => o?.state === s.state.replace(/_rt$/, ''))))
        .map(sigChip))));
    detail.append(h('div', { class: 'eras-kpis' }, KPI_STATES.map(s => eraKpi(e, s))));

    // ---- assets
    const rows = Object.entries(e.assets || {}).map(([id, s]) => ({ id, ...s }))
      .sort((a, b) => (isNum(b.real_cagr) ? b.real_cagr : -9) - (isNum(a.real_cagr) ? a.real_cagr : -9));
    const missing = allAssets.filter(a => !(a in (e.assets || {})));
    const administered = e.end < '1973-01';
    const barBox = ui.chartBox('tall');
    detail.append(h('div', { class: 'eras-grid2' },
      card({ title: '资产表现', sub: '含收入的总回报；实际 = 用 CPI 平减，看的是购买力，按实际年化从高到低排列。' },
        rows.length ? ui.table([
          { key: 'id', label: '资产', fmt: (v) => ui.assetName(v) },
          { key: 'cagr', label: '名义年化', fmt: (v) => pctS(v), cls: (v) => signOf(v) },
          { key: 'real_cagr', label: '实际年化', fmt: (v) => pctS(v), cls: (v) => signOf(v) },
          { key: 'vol', label: '波动率', fmt: (v) => ui.fmtPct(v, 1) },
          { key: 'max_dd', label: '最大回撤', fmt: (v) => ui.fmtPct(v, 1) },
          { key: 'avg_income', label: '收入率', fmt: (v, r) => (ui.META.assets[r.id]?.income === 'none' ? '—' : ui.fmtPct(v, 1)) },
        ], rows, { stack: true }) : ui.notice('这一时期没有完整的资产数据。', 'info'),
        missing.length ? h('p', { class: 'eras-note' }, `没有每个月数据、不参与统计：${missing.map(ui.assetName).join('、')}。`) : null,
        administered ? h('p', { class: 'eras-note' }, '1973 年以前黄金（布雷顿森林）和原油（挂牌价）价格受管制，波动率和回撤都很小，不能和后来直接比较。') : null),
      card({ title: '实际年化回报', sub: '每根柱子 = 扣除通胀后的年化总回报：这一时代的宏观组合让谁赢、让谁输。' }, barBox)));

    // ---- macro regimes of this era (regimes.json eras[]; omitted without the file)
    const re = rgById[e.id];
    const rgBox = ui.chartBox();
    if (re) {
      const runs = (re.runs_expost || []).map((t) => {
        const lab = rgByName[String(t).split(' ').pop()];
        return h('span', { class: 'chip' }, lab ? h('i', { style: { background: lab.color } }) : null, String(t));
      });
      const fl = re.flags || {};
      const flagText = isNum(fl.contraction)
        ? `收缩月份 ${pct0(fl.contraction)}（实时 ${isNum(fl.contraction_rt) ? pct0(fl.contraction_rt) : '—'}）`
          + `${isNum(fl.high_infl) ? `，高通胀月份 ${pct0(fl.high_infl)}` : ''}${isNum(fl.cpi_neg) && fl.cpi_neg > 0 ? `，CPI 同比为负的月份 ${pct0(fl.cpi_neg)}` : ''}。` : '';
      detail.append(card({ title: '这一时代的宏观体制',
        sub: `每个月按同一套规则归入一种体制：${re.months} 个月里各体制的占比。事后 = 用全部数据判断；实时 = 只用截至当月的数据（单边、修订后数据）。`,
        actions: h('a', { href: '#regimes?eras' }, '宏观体制 →') },
      rgBox,
      h('div', { class: 'eras-rgmix' }, h('b', {}, '事后：'), mixLine(re.expost)),
      h('div', { class: 'eras-rgmix' }, h('b', {}, '实时：'), mixLine(re.realtime)),
      flagText ? h('p', { class: 'eras-note' }, flagText) : null,
      runs.length ? h('div', { class: 'eras-sub', style: { marginTop: '10px' } }, '事后体制的各段（起止月份）') : null,
      runs.length ? h('div', { class: 'eras-runs' }, runs) : null));
    }

    // ---- transmission attribution (eras[].attribution)
    const attrCard = attributionCard(e);
    detail.append(attrCard.el);

    // ---- correlations and transmission
    const corrBox = ui.chartBox();
    const corrAssets = e.corr?.assets || [];
    const sbc = e.stock_bond_corr, pt = e.oil_cpi_passthrough, se = e.oil_cpi_passthrough_se, ps = e.phillips_slope;
    const pLabel = PHILLIPS_LABEL[e.phillips_measure] || '通胀';
    detail.append(h('div', { class: 'eras-grid2' },
      card({ title: '时代内资产相关性', sub: '月度总回报的相关系数：红 = 同涨同跌，蓝 = 此消彼长；看哪些资产在这个时代能互相分散风险。' },
        corrAssets.length > 1 ? corrBox : ui.notice('资产太少，无法计算相关性。', 'info')),
      card({ title: '传导特征', sub: '同一个冲击在不同时代的传导力度不同：这三个数刻画了这个时代的“因果脾气”。' },
        metric('股债相关性（美股大盘 vs 10 年期美债，月度）', signed(sbc),
          !isNum(sbc) ? '缺少股票或美债的完整数据。'
            : sbc > 0.1 ? '同涨同跌：股票跌时债券也跌，债券难以对冲股票（通胀风险主导时常见）。'
              : sbc < -0.1 ? '反向波动：股票跌时债券往往涨，股债组合能互相对冲。' : '基本不相关：债券对股票几乎没有对冲作用。',
          isNum(sbc) && Math.abs(sbc) > 0.1 ? (sbc > 0 ? 'eras-hi' : 'eras-lo') : ''),
        metric('油价 → CPI 传导（6 个月累计弹性）', isNum(pt) ? `${pt.toFixed(3)}${isNum(se) ? ` ± ${se.toFixed(3)}` : ''}` : '无法估计',
          isNum(pt) ? `油价涨 10%，6 个月内 CPI 约多涨 ${(pt * 10).toFixed(2)}%${isNum(se) && se > Math.abs(pt) / 2 ? '（估计误差较大）' : ''}；± 后是标准误。`
            : passthroughWhy(e)),
        metric(`菲利普斯曲线斜率（${pLabel}对失业率）`, isNum(ps) ? signed(ps) : '—',
          isNum(ps) ? `失业率每高 1 个百分点，${pLabel}约${ps < 0 ? '低' : '高'} ${Math.abs(ps).toFixed(2)} 个百分点。`
            + (ps > -0.2 ? '斜率接近 0 或为正：劳动力市场松紧对工资和物价的约束很弱。'
              : `斜率明显为负：劳动力市场越紧（失业率越低），${pLabel}涨得越快，就业对物价的传导较强。`)
            + (e.phillips_measure && e.phillips_measure !== 'wage_yoy' ? `（工资序列 1964 年起，这一时期改用${pLabel}。）` : '')
            : '数据不足，无法估计。'))));

    // ---- factor drift and volatility
    const facRows = Object.entries(e.factors || {}).map(([id, f]) => ({ id, ...f, ratio: isNum(f.vol_annual) && medVol[id] > 0 ? f.vol_annual / medVol[id] : null }));
    detail.append(card({ title: '因子的年化漂移与波动', sub: '月度因子变化 ×12 是漂移、×√12 是波动；“相对各时代中位数”大于 1 说明这个时代该因子格外动荡，资产回报更多由它驱动。' },
      ui.table([
        { key: 'id', label: '因子', fmt: (v) => factorName(v) },
        { key: 'mean_annual', label: '年化漂移', fmt: (v, r) => fmtFactor(r.id, v, true),
          cls: (v, r) => { const s = fmtFactor(r.id, v, true); return s.startsWith('+') ? 'pos' : s.startsWith('-') ? 'neg' : ''; } },
        { key: 'vol_annual', label: '年化波动', fmt: (v, r) => fmtFactor(r.id, v, false) },
        { key: 'ratio', label: '波动相对各时代中位数', fmt: (v) => (isNum(v) ? `${v.toFixed(2)} 倍` : '—'),
          cls: (v) => (isNum(v) && v >= 1.5 ? 'eras-hi' : isNum(v) && v <= 0.6 ? 'eras-lo' : '') },
      ], facRows, { stack: true }),
      h('p', { class: 'eras-note' }, '劳动收入占比、利润、投资、净值等季度数据按发布月对齐（不插值），只在发布月变化，月度变化集中在这些月份：它们的波动只适合跨时代相对比较。')));

    // charts after the DOM exists
    charts.bars(barBox, {
      x: rows.map(r => ui.assetName(r.id)),
      series: [{ name: '实际年化', data: rows.map(r => ({ value: r.real_cagr, itemStyle: { color: charts.assetColor(r.id), borderRadius: 2 } })) }],
      horizontal: true, fmt: 'pct', legend: false,
    });
    attrCard.draw();
    if (re) regimeBars(rgBox, { rows: ['事后', '实时'], mixes: [re.expost || {}, re.realtime || {}], labels: rgLabels, rowH: 34,
      extra: (i) => (i === 1 ? '只用截至当月的数据（单边、修订后数据）' : '') });
    if (corrAssets.length > 1) {
      const names = corrAssets.map(ui.assetName);
      heat(charts, corrBox, { rows: names, cols: names, values: e.corr.matrix, min: -1, max: 1, height: names.length * 24 + 110,
        labelFmt: (v) => fixed(v),
        tip: (r, c, v) => `${names[r]} × ${names[c]}<br/>相关系数 <b>${isNum(v) ? fixed(v) : '—'}</b>` });
    }
  }

  // ======================================================== cross-era comparison
  root.append(h('div', { class: 'section-title eras-anchor' }, '跨时代对比'));
  const fpBox = ui.chartBox(), cagrBox = ui.chartBox();
  const sbcBox = ui.chartBox(), ptBox = ui.chartBox(), psBox = ui.chartBox();
  const tlBox = ui.chartBox('tall');
  // eras whose Phillips slope uses an inflation measure instead of wage growth (no wage data yet)
  const fallback = eras.filter(e => e.phillips_measure && e.phillips_measure !== 'wage_yoy');
  const phillipsNote = fallback.length
    ? `工资序列 1964 年才开始，${fallback.map(e => `${e.name}的斜率用${PHILLIPS_LABEL[e.phillips_measure] || '通胀'}代替工资增速`).join('；')}，和其他时代不完全可比。`
    : '';
  const groupWrap = h('div', { class: 'eras-toolbar' });
  let tlGroup = 'inflation', tlZ = false;
  // each era's regime mix (regimes.json; the card is left out without it)
  const rgCmpBox = ui.chartBox();
  let rgSide = 'expost';
  const drawRgCompare = () => {
    const rows = rgEras.map(x => String(x.name).split('（')[0]);      // "1948–1954（…仅供参考）": the note goes to the tooltip
    const chart = regimeBars(rgCmpBox, { rows, mixes: rgEras.map(x => x[rgSide] || {}), labels: rgLabels, rowH: 28,
      extra: (i) => [rows[i] !== rgEras[i].name ? ui.esc(rgEras[i].name) : '', `${rgEras[i].start} 至 ${rgEras[i].end}，${rgEras[i].months} 个月`, rgSide === 'realtime' ? '实时（单边、修订后数据）' : '事后',
        byId[rgEras[i].id] ? '点击查看这个时代' : ''] });
    chart.off('click');
    chart.on('click', (p) => { const id = rgEras[p.dataIndex]?.id; if (byId[id]) select(id, true); });
  };
  const rgCompare = card({ title: '各时代的宏观体制构成',
    sub: '每个时代的月份按同一套规则归入扩张、通胀、物价负增长、衰退、深度衰退、滞胀；第一行早于时代划分，只作参考。切到“实时”看只用截至当月数据的版本（单边、修订后数据）。点击一行查看该时代。',
    actions: h('div', { style: { display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' } },
      ui.seg([{ value: 'expost', label: '事后' }, { value: 'realtime', label: '实时' }], rgSide, (v) => { rgSide = v; drawRgCompare(); }),
      h('a', { href: '#regimes?eras' }, '宏观体制 →')) }, rgCmpBox);
  root.append(
    card({ title: '时代指纹（z 值）', sub: `每格 = 该时代均值相对 1955 年以来均值偏离几个标准差：红 = 偏高，蓝 = 偏低；${today?.states ? '最后一行是今天（最新月份），' : ''}点击一行查看该时代详情。` },
      fp ? ui.wideChart(fpBox, 620) : ui.notice('时代画像结果里没有时代指纹。', 'info')),
    h('div', { style: { height: '16px' } }),
    card({ title: '各时代的实际年化回报', sub: '扣除通胀后的年化总回报：同一类资产在不同宏观组合下的表现可以相差 20 个百分点以上，空格表示该时期没有完整数据；颜色按 K 线配色（涨色为正）。' }, ui.wideChart(cagrBox, 520)),
    h('div', { style: { height: '16px' } }),
    card({ title: '传导特征怎么随时代变化', sub: '股债相关性 > 0 时债券对冲不了股票；油价弹性 ×10 = 油价涨 10% 时 6 个月内 CPI 多涨几 %；菲利普斯斜率越负，就业松紧对工资越敏感。同一个冲击在不同时代传导力度不同，这也是引擎分时代估计的原因。' },
      h('div', { class: 'eras-grid3' },
        h('div', {}, h('div', { class: 'eras-metric-title' }, '股债相关性（>0 同涨同跌）'), sbcBox),
        h('div', {}, h('div', { class: 'eras-metric-title' }, '油价 → CPI 6 个月弹性'), ptBox),
        h('div', {}, h('div', { class: 'eras-metric-title' }, '菲利普斯曲线斜率'), psBox)),
      phillipsNote ? h('p', { class: 'eras-note' }, phillipsNote) : null),
    h('div', { style: { height: '16px' } }),
    ...(rgEras.length ? [rgCompare, h('div', { style: { height: '16px' } })] : []),
    card({ title: '1955 年以来的关键状态', sub: '月度数据，底纹是时代划分；切到“标准化”可把不同单位的指标放在同一把尺子上（0 = 1955 年以来均值），看每个时代偏离常态的方向。' },
      tl ? groupWrap : null, tl ? tlBox : ui.notice('时代画像结果里没有状态时间线。', 'info')));

  // ======================================================== the new situation
  const newSec = h('div', { class: 'section-title eras-anchor' }, '新形势：白领工资被压制、资本端火热、收入向资本倾斜');
  root.append(newSec);
  root.append(h('p', { class: 'lead' },
    '数据里没有单独的“白领工资”序列，这里用四组数字刻画：劳动者拿到的份额（劳动收入占比）、资本拿到的份额（利润占比）、',
    '白领岗位（白领行业就业占比）和普通工人的实际工资；资本端看投资强度、家庭净值相对收入和估值。每个指标把今天和每个时代的均值放在一起比较。'));
  const newKpis = h('div', { class: 'grid cols-4' });
  const newGrid = h('div', { class: 'eras-grid4' });
  const newTable = h('div');
  const lpBox = h('div', {}, ui.loading('读取因果传导结果…'));
  const engineBox = h('div');
  root.append(newKpis, h('div', { style: { height: '16px' } }), newGrid, h('div', { style: { height: '16px' } }),
    card({ title: '今天在历史上的位置', sub: '历史分位 = 1955 年以来有多少比例的月份低于今天；分位接近 0 或 100% 就是历史上罕见、甚至从未出现过的状态。' }, newTable),
    h('div', { style: { height: '16px' } }),
    h('div', { class: 'eras-grid2' },
      card({ title: '收入分配怎么传导到宏观', sub: '劳动收入占比下降 1% 之后 8 个季度的局部投影：比较 1960–1999 和 2000 年以来，看“收入向资本倾斜”对消费、物价、利率和股价的影响有没有变。' }, lpBox),
      card({ title: '引擎怎么刻画这种新形势', sub: '情景只改变历史月份被抽中的概率，所以历史上没有的组合要靠因子、观点和有上限的覆盖参数注入。' }, engineBox)));

  // ======================================================== today's analogs
  const simSec = h('div', { class: 'section-title eras-anchor' }, '今天像哪个时代');
  root.append(simSec);
  const sim = data.today_similarity;
  const wBox = ui.chartBox(), gapBox = ui.chartBox();
  const simWrap = h('div');
  root.append(simWrap);
  if (sim?.weights) {
    const exc = (sim.excluded_eras || []).map(ui.eraName);
    const top = sim.top_months || [];
    const residual = (sim.residual || []).slice().sort((a, b) => Math.abs(b.gap_z) - Math.abs(a.gap_z));
    simWrap.append(
      h('div', { class: 'eras-grid2' },
        card({ title: '历史时代权重', sub: '用熵池（Entropy Pooling）给历史月份重新加权，让加权平均尽量接近今天的关键状态；条形 = 各时代月份拿到的权重之和。' },
          wBox,
          h('dl', { class: 'kv', style: { marginTop: '8px' } },
            h('dt', {}, '今天（观点窗口）'), h('dd', {}, (sim.window || []).join(' 至 ') || sim.date || '—', '，取过去 12 个月均值'),
            h('dt', {}, '历史数据库'), h('dd', {}, `${(sim.database || []).join(' 至 ')}，${sim.n_months_db ?? '—'} 个月`),
            h('dt', {}, '有效月份数 ENS'), h('dd', {}, `${isNum(sim.ens) ? sim.ens.toFixed(0) : '—'}（下限 ${sim.ens_min ?? '—'}；权重越集中越小）`),
            h('dt', {}, 'KL 散度'), h('dd', {}, `${isNum(sim.kl) ? sim.kl.toFixed(2) : '—'}（离均匀权重多远，越大说明今天越不寻常）`),
            h('dt', {}, '软化参数 τ'), h('dd', {}, `${isNum(sim.tau) ? sim.tau.toFixed(3) : '—'}（无法同时满足的观点被放松的程度）`)),
          exc.length ? h('div', { style: { marginTop: '10px' } }, ui.notice(
            `匹配时排除了：${exc.join('、')}。今天就处在这个时代里，用它自己的月份匹配今天没有信息量；数据库也只用 24 个月以前的数据。`, 'info')) : null),
        card({ title: '解读与最相似的历史月份', sub: '权重最高的 12 个历史月份；“相对平均”= 权重是均匀权重的几倍。' },
          sim.interpretation ? h('p', { style: { margin: '0 0 12px' } }, sim.interpretation) : null,
          top.length ? ui.table([
            { key: 'date', label: '月份' },
            { key: 'era', label: '所属时代', fmt: (v) => ui.eraName(v) },
            { key: 'weight', label: '权重', fmt: (v) => ui.fmtPct(v, 2) },
            { key: 'weight', label: '相对平均', fmt: (v) => (isNum(v) && sim.n_months_db ? `${(v * sim.n_months_db).toFixed(0)}×` : '—') },
          ], top, { stack: true }) : ui.notice('没有最相似的历史月份。', 'info'))),
      h('div', { style: { height: '16px' } }),
      h('div', { class: 'eras-grid2' },
        card({ title: '历史拼不出来的部分', sub: '加权后仍然存在的差距（标准差）：右 = 今天更高，左 = 今天更低；差距越大越说明今天是“新形势”，情景推演要额外考虑。' }, gapBox),
        card({ title: '差距明细', sub: '“今天”是过去 12 个月均值；“作为观点”的状态参与了匹配，其余只是事后对照。' },
          ui.table([
            { key: 'name', label: '状态', wrap: true, minWidth: 96, fmt: (v, r) => v || nameOf(r.state) },
            { key: 'today', label: '今天', fmt: (v, r) => fmtState(r.unit ?? unitOf(r.state), v) },
            { key: 'matched', label: '历史拼出的', fmt: (v, r) => fmtState(r.unit ?? unitOf(r.state), v) },
            { key: 'gap_z', label: h('span', {}, '差距', h('br'), '（标准差）'), fmt: (v) => signed(v),
              cls: (v) => (isNum(v) && Math.abs(v) >= 0.75 ? (v > 0 ? 'eras-hi' : 'eras-lo') : '') },
            { key: 'in_view', label: h('span', {}, '作为', h('br'), '观点'), fmt: (v) => (v ? ui.chip('是', 'accent') : h('span', { class: 'muted' }, '否')) },
          ], residual, { stack: true }))));
  } else {
    simWrap.append(ui.notice('时代画像结果里没有“今天像哪个时代”的匹配。', 'info'));
  }

  // ======================================================== notes
  if ((data.notes || []).length) {
    root.append(h('div', { class: 'section-title' }, '口径说明'));
    root.append(card({}, h('ul', { class: 'list' }, data.notes.map(n => h('li', { class: 'muted' },
      String(n).replace(/列在 residual 中/, '列在上面的“历史拼不出来的部分”'))))));
  }

  // ======================================================== charts
  drawPicker();
  renderEra(byId[sel]);
  if (rgEras.length) {
    try { drawRgCompare(); } catch (err) { console.error(err); rgCmpBox.replaceChildren(ui.notice(`图表渲染失败：${err.message}`, 'error')); }
  }

  // fingerprint, with today's single-month values on the same 1955+ yardstick
  if (fp) {
    const rowsN = [...(fp.row_names || fp.rows)], rowIds = [...fp.rows];
    const zs = fp.z.map(r => r.slice()), vals = fp.values.map(r => r.slice());
    let todayRow;
    if (today?.states) {
      todayRow = rowsN.length;
      rowsN.push(`今天（${today.date}）`); rowIds.push(null);
      zs.push(fp.cols.map(c => zOf(c, today.states[c])));
      vals.push(fp.cols.map(c => (isNum(today.states[c]) ? today.states[c] : null)));
    }
    const fpChart = heat(charts, fpBox, { rows: rowsN, cols: fp.col_names, values: zs, min: -2.5, max: 2.5,
      height: rowsN.length * 30 + 120, labelFmt: (v) => signed(v, 1), boldRow: todayRow,
      tip: (r, c, v) => {
        const sid = fp.cols[c], unit = fp.col_units[c];
        return `<b>${rowsN[r]}</b> × ${fp.col_names[c]}<br/>${rowIds[r] ? '时期均值' : '今天（单月）'} <b>${fmtState(unit, vals[r][c])}</b>`
          + `<br/>1955 年以来均值 ${fmtState(unit, ref[sid]?.mean)}，标准差 ${fmtState(unit, ref[sid]?.sd)}<br/>z = ${signed(v)}`;
      } });
    fpChart.on('click', (p) => { const id = rowIds[p.value?.[1]]; if (id) select(id, true); });
  }

  // real CAGR eras x assets
  {
    const vals = eras.map(e => allAssets.map(a => e.assets?.[a]?.real_cagr ?? null));
    const ch = heat(charts, cagrBox, { rows: eras.map(e => e.name), cols: allAssets.map(ui.assetName), values: vals, min: -0.15, max: 0.15,
      height: eras.length * 30 + 120, labelFmt: (v) => pctS(v), minCell: 42, colors: charts.klineDiverging(),      // returns: K-line colours
      tip: (r, c, v) => {
        const s = eras[r].assets?.[allAssets[c]];
        if (!s) return `${eras[r].name} × ${ui.assetName(allAssets[c])}<br/>该时期没有完整数据`;
        return `<b>${eras[r].name}</b> × ${ui.assetName(allAssets[c])}<br/>实际年化 <b>${pctS(v)}</b>　名义 ${pctS(s.cagr)}`
          + `<br/>波动率 ${ui.fmtPct(s.vol, 1)}　最大回撤 ${ui.fmtPct(s.max_dd, 1)}`;
      } });
    ch.on('click', (p) => { const e = eras[p.value?.[1]]; if (e) select(e.id, true); });
  }

  // transmission metrics across eras
  {
    const names = eras.map(e => e.name);
    const colorOf = (v) => (isNum(v) && v < 0 ? DOWN : UP);
    const sb = eras.map(e => e.stock_bond_corr), ptv = eras.map(e => e.oil_cpi_passthrough), psv = eras.map(e => e.phillips_slope);
    const hgt = eras.length * 26 + 40;
    hbars(charts, sbcBox, { labels: names, values: sb, colors: sb.map(colorOf), fmt: (v) => signed(v), ref: 0, height: hgt,
      tip: (i) => `<b>${names[i]}</b><br/>股债相关性 ${signed(sb[i])}` });
    hbars(charts, ptBox, { labels: names, values: ptv, colors: ptv.map(() => '#e8590c'), fmt: (v) => v.toFixed(3), height: hgt,
      nullText: (i) => (eras[i].oil_price_moves < MIN_OIL_MOVES ? '管制油价，无法估计' : '无法估计'),
      tip: (i) => (isNum(ptv[i])
        ? `<b>${names[i]}</b><br/>弹性 ${ptv[i].toFixed(3)}${isNum(eras[i].oil_cpi_passthrough_se) ? ` ± ${eras[i].oil_cpi_passthrough_se.toFixed(3)}` : ''}<br/>油价涨 10%，6 个月内 CPI 约多涨 ${(ptv[i] * 10).toFixed(2)}%`
        : `<b>${names[i]}</b><br/>无法估计：${passthroughWhy(eras[i])}`) });
    hbars(charts, psBox, { labels: names, values: psv, colors: psv.map(colorOf), fmt: (v) => signed(v), ref: 0, height: hgt,
      tip: (i) => `<b>${names[i]}</b><br/>斜率 ${signed(psv[i])}（${PHILLIPS_LABEL[eras[i].phillips_measure] || '通胀'}对失业率）` });
  }

  // timeline
  function drawTimeline() {
    if (!tl) return;
    const g = TIMELINE_GROUPS.find(x => x.id === tlGroup) || TIMELINE_GROUPS[0];
    const ids = [...g.left, ...(g.right || [])].filter(s => tl.series?.[s]);
    const lines = ids.map((s, i) => {
      const raw = tl.series[s];
      return { id: s, name: nameOf(s), unit: unitOf(s), right: (g.right || []).includes(s), color: LINE_COLORS[i % LINE_COLORS.length], raw,
        data: tlZ ? raw.map(v => { const z = zOf(s, v); return z === null ? null : +z.toFixed(3); }) : raw };
    });
    timelineChart(charts, tlBox, { dates: tl.dates, lines, bands: tl.bands || eras.map(e => ({ id: e.id, name: e.name, start: e.start, end: e.end })),
      zMode: tlZ, unitLeft: unitOf(g.left[0]) === 'pp' ? '%' : unitOf(g.left[0]),
      unitRight: g.right ? ({ '2017=100': '指数（2017=100）', '倍': '倍', '指数': '指数', $: '美元' }[unitOf(g.right[0])] || unitOf(g.right[0])) : '',
      eraAt });
  }
  groupWrap.append(
    ui.seg(TIMELINE_GROUPS.map(g => ({ value: g.id, label: g.label })), tlGroup, (v) => { tlGroup = v; drawTimeline(); }),
    ui.seg([{ value: 'raw', label: '原始值' }, { value: 'z', label: '标准化' }], 'raw', (v) => { tlZ = v === 'z'; drawTimeline(); }));
  drawTimeline();

  // ---- the new situation: KPIs, today vs each era, position in history
  {
    const dates = tl?.dates || [];
    const done = eras.filter(e => !e.ongoing);
    const items = NEW_STATES.map(([sid, why]) => {
      const v = today?.states?.[sid];
      const series = tl?.series?.[sid];
      const st = histStats(series, dates, v);
      const lastIdx = series ? series.length - 1 : -1;
      const chg = series && lastIdx >= 12 && isNum(series[lastIdx]) && isNum(series[lastIdx - 12]) ? series[lastIdx] - series[lastIdx - 12] : null;
      let closest = null;
      for (const e of done) {
        const m = e.states?.[sid]?.mean;
        if (isNum(m) && isNum(v) && (!closest || Math.abs(m - v) < Math.abs(closest.mean - v))) closest = { name: e.name, mean: m };
      }
      return { sid, why, v, st, chg, z: zOf(sid, v), closest, unit: unitOf(sid), name: nameOf(sid) };
    });
    const posText = (it) => {
      if (!it.st) return '—';
      if (it.st.isMin) return '1955 年以来最低';
      if (it.st.isMax) return '1955 年以来最高';
      return `历史分位 ${(it.st.pct * 100).toFixed(1)}%`;
    };
    // 12-month change in the state's own unit (percent states change in pp)
    const chgText = (it) => (isNum(it.chg)
      ? `${signed(it.chg)}${it.unit === '%' || it.unit === 'pp' ? 'pp' : it.unit === '倍' ? ' 倍' : it.unit === '$' ? ' 美元' : ''}` : '—');
    newKpis.append(...items.map(it => ui.kpi(it.name, fmtState(it.unit, it.v),
      `${posText(it)}${isNum(it.chg) ? ` · 12 个月变化 ${chgText(it)}` : ''}`,
      it.st && (it.st.pct >= 0.95 || it.st.isMax) ? 'eras-hi' : it.st && (it.st.pct <= 0.05 || it.st.isMin) ? 'eras-lo' : '')));

    // what has never happened before
    const extreme = items.filter(it => it.st && (it.st.isMin || it.st.isMax || it.st.pct <= 0.02 || it.st.pct >= 0.98));
    if (extreme.length) {
      newTable.append(h('div', { class: 'notice eras-callout' },
        h('b', {}, `今天有 ${extreme.length} 个指标处在 1955 年以来的极端位置：`),
        h('ul', { class: 'list' }, extreme.map(it => h('li', {}, `${it.name} ${fmtState(it.unit, it.v)}：${posText(it)}`
          + (it.st.isMin || it.st.isMax ? '' : `（历史区间 ${fmtState(it.unit, it.st.min)} ~ ${fmtState(it.unit, it.st.max)}）`)
          + (it.closest ? `，最接近的已结束时代是${it.closest.name}（均值 ${fmtState(it.unit, it.closest.mean)}）` : ''))))));
    }
    // two-line cells (value over a muted detail) keep the table narrow enough for a 900px screen
    const two = (main, sub, cls = '') => h('div', {}, h('div', { class: cls }, main), sub ? h('div', { class: 'eras-sub' }, sub) : null);
    newTable.append(stackable(ui, ui.table([
      { key: 'name', label: '指标' },
      { key: 'v', label: '今天 / 12 个月变化', fmt: (v, r) => two(fmtState(r.unit, v), isNum(r.chg) ? chgText(r) : '') },
      { value: (r) => ref[r.sid]?.mean, label: '1955 年以来均值 / 今天的 z', fmt: (v, r) => two(fmtState(r.unit, v), isNum(r.z)
        ? h('span', { class: Math.abs(r.z) >= 1.5 ? (r.z > 0 ? 'eras-hi' : 'eras-lo') : '' }, `z ${signed(r.z)}`) : '') },
      { value: (r) => r.st?.pct, label: '历史分位', fmt: (v, r) => (r.st ? (r.st.isMin ? h('b', { class: 'eras-lo' }, '最低') : r.st.isMax ? h('b', { class: 'eras-hi' }, '最高') : `${(v * 100).toFixed(1)}%`) : '—') },
      { value: (r) => r.st, label: '历史区间（月份）', fmt: (s, r) => (s ? h('div', {}, h('div', {}, `最低 ${fmtState(r.unit, s.min)}（${s.minDate}）`), h('div', {}, `最高 ${fmtState(r.unit, s.max)}（${s.maxDate}）`)) : '—') },
      { value: (r) => r.closest, label: '最接近的已结束时代', fmt: (c, r) => (c ? two(c.name, `均值 ${fmtState(r.unit, c.mean)}`) : '—') },
    ], items)));

    // small multiples: each era's mean vs today, drawn as the gap to the 1955+ mean (the dashed
    // zero line) so a value beyond every era stands out; the labels are the levels themselves
    const boxes = items.map(it => {
      const box = ui.chartBox();
      newGrid.append(card({ title: it.name, sub: `${it.why}。条形 = 相对 1955 年以来均值（虚线，${fmtState(it.unit, ref[it.sid]?.mean)}）的高低，数字是实际水平；灰 = 各时代均值，橙 = 今天。`, cls: 'eras-mini' }, box));
      return box;
    });
    items.forEach((it, k) => {
      const labels = [...eras.map(e => e.name), `今天（${today?.date || '—'}）`];
      const levels = [...eras.map(e => e.states?.[it.sid]?.mean ?? null), isNum(it.v) ? it.v : null];
      const mean = ref[it.sid]?.mean;
      const values = isNum(mean) ? levels.map(v => (isNum(v) ? v - mean : null)) : levels;
      const colors = [...eras.map(e => (e.ongoing ? '#91a7ff' : '#adb5bd')), TODAY];
      hbars(charts, boxes[k], { labels, values, colors, height: labels.length * 22 + 24, fmt: (v) => fmtState(it.unit, v),
        text: (i) => fmtState(it.unit, levels[i]), ref: isNum(mean) ? 0 : undefined, nullText: '数据不足',
        tip: (i) => `<b>${labels[i]}</b><br/>${i < eras.length ? '时期均值' : '今天'} ${fmtState(it.unit, levels[i])}`
          + (isNum(mean) && isNum(levels[i]) ? `<br/>比 1955 年以来均值 ${fmtState(it.unit, mean)} ${levels[i] >= mean ? '高' : '低'} ${fmtState(it.unit, Math.abs(levels[i] - mean))}` : '')
          + (isNum(zOf(it.sid, levels[i])) ? `<br/>z ${signed(zOf(it.sid, levels[i]))}` : '') });
    });

    // how the engine represents it (from meta: which distribution / capital factors exist and which are used)
    const facs = (ui.META.raw?.factors || []).filter(f => f.group === 'distribution' || f.group === 'capital');
    const used = new Set(ui.META.raw?.model?.spec?.factors || []);
    const facList = (pick) => facs.filter(pick).map(f => f.name).join('、') || '无';
    engineBox.append(h('ul', { class: 'list' },
      h('li', {}, h('b', {}, '当前资产模型在用的分配 / 资本因子：'), facList(f => used.has(f.id)), '。'),
      h('li', {}, h('b', {}, '候选因子（诊断与迭代页决定是否加入）：'), facList(f => !used.has(f.id) && !f.endogenous), '。'),
      h('li', {}, h('b', {}, '内生因子：'), facList(f => f.endogenous), '，它们由资产价格本身构成，不能用来解释资产回报，对它们设观点等于直接指定市场。'),
      h('li', {}, '情景可以直接给劳动收入占比、利润占比、白领就业占比等宏观状态和因子设定目标均值（“观点”），引擎在历史月份里找最接近的组合；历史上拼不出来的部分（见下方“历史拼不出来的部分”）由有上限的覆盖参数补足，并在情景页标出。'),
      h('li', {}, '“收入向资本倾斜”的宏观后果用季度局部投影估计（见“收入分配怎么传导到宏观”表），AI 资本开支没有可识别的外生冲击，在因果传导页里是描述性的链条。'),
      h('li', {}, h('a', { href: '#causality?labor_share_chain' }, '查看劳动收入占比的因果链与脉冲响应 →'))));
  }

  // labor-share local projections from causality.json; optional, never blocks the page
  const lpDone = api.get('causality').then((c) => {
    lpBox.innerHTML = '';
    const dist = c?.distribution;
    if (!dist?.responses) { lpBox.append(ui.notice('因果传导结果里没有劳动收入占比的局部投影。', 'info')); return; }
    const periods = dist.periods || [];
    const hq = Math.min(8, dist.horizon_q ?? 8);
    const vars = (dist.variables || []).filter(v => v.id !== 'labor_share');
    const cell = (r) => {
      if (!r || !isNum(r.coef?.[hq])) return '—';
      const u = r.unit === 'pp' ? 'pp' : '%';
      const sig = isNum(r.lo?.[hq]) && isNum(r.hi?.[hq]) && (r.lo[hq] > 0 || r.hi[hq] < 0);
      return h('div', { class: sig ? '' : 'muted' }, h('div', {}, `${signed(r.coef[hq])}${u}`, sig ? ' ★' : ''),
        h('div', { class: 'muted', style: { fontSize: '11px' } }, `${bound(r.lo?.[hq])} ~ ${bound(r.hi?.[hq])}`));
    };
    lpBox.append(ui.table([
      { key: 'name', label: `${hq} 季后的累计响应`, wrap: true, minWidth: 108 },     // names wrap, not one word per line
      ...periods.map(p => ({ value: (v) => ({ ...dist.responses[p.id]?.[v.id], unit: v.unit }), label: p.name, fmt: (r) => cell(r) })),
    ], vars, { stack: true }), h('p', { class: 'eras-note' }, `冲击 = ${dist.shock || '劳动收入占比下降 1%'}；数值下方为 ${Math.round((dist.band ?? 0.9) * 100)}% 区间，★ = 区间不含 0；灰色 = 不显著。`));
  }).catch((err) => {
    lpBox.innerHTML = '';
    lpBox.append(ui.notice(err.status === 404 ? '因果传导结果还没有生成，运行 python -m scenariolab.build 后可见。'
      : `读取因果传导结果失败：${err.message}`, 'info'));
  });

  // today's analogs
  if (sim?.weights) {
    const entries = Object.entries(sim.weights).sort((a, b) => b[1] - a[1]);
    hbars(charts, wBox, { labels: entries.map(([e]) => ui.eraName(e)), values: entries.map(([, w]) => w),
      colors: entries.map(() => '#3b5bdb'), fmt: (v) => ui.fmtPct(v, 1), height: entries.length * 28 + 40 });
    const residual = (sim.residual || []).slice().sort((a, b) => Math.abs(b.gap_z) - Math.abs(a.gap_z));
    hbars(charts, gapBox, { labels: residual.map(r => r.name || nameOf(r.state)), values: residual.map(r => r.gap_z),
      colors: residual.map(r => (Math.abs(r.gap_z) < 0.4 ? '#ced4da' : r.gap_z > 0 ? UP : DOWN)), fmt: (v) => signed(v), ref: 0,
      height: residual.length * 24 + 40,
      tip: (i) => { const r = residual[i]; const u = r.unit ?? unitOf(r.state); return `<b>${r.name || nameOf(r.state)}</b><br/>今天 ${fmtState(u, r.today)}　历史拼出 ${fmtState(u, r.matched)}<br/>差距 ${signed(r.gap_z)} 个标准差${r.in_view ? '（参与匹配）' : ''}`; } });
  }

  // deep links to a section: #eras?new, #eras?today
  const target = param === 'new' ? newSec : param === 'today' ? simSec : null;
  if (target) {
    let landed = null;
    const go = () => { if (!alive()) return; target.scrollIntoView({ block: 'start' }); landed = window.scrollY; };
    setTimeout(go, 60);
    // the labor-share table above "today" arrives later and pushes the section down: scroll again,
    // unless the reader has already scrolled somewhere else
    lpDone.then(() => setTimeout(() => { if (landed === null || window.scrollY === landed) go(); }, 90));
  }
}
