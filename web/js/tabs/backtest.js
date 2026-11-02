// Backtest: hold-out windows in which the factor model, the month library and the starting point see
// only the training months. The engine's paths are set against what actually happened, and the gap is
// split into a scenario, an engine and a transmission part.
// SIM_DESIGN §10.5 / §10.1: the conditional fan comes in two arms (宏观 + 金融条件（含市场价格）, the old fan_cond, and
// 只有宏观, fan_cond_macro); the long-run anchors are fitted to each train end (shown per window); the headline's
// transmission tile is quarterly and annual (blockR2 on transmission.series), the monthly R² is a reference.
const ORDER = ['us_equity', 'nasdaq', 'energy', 'industrials', 'staples', 'real_estate', 'gold',
  'healthcare', 'utilities', 'financials', 'crude_oil', 'treasury_10y', 'cash'];
const KEY = new Set(ORDER.slice(0, 7));              // the classes the user follows most closely
const TICKER = { us_equity: 'SPY', nasdaq: 'QQQ' };
const SHORT = { us_equity: '美股大盘', nasdaq: '纳指100', energy: '能源股', industrials: '工业股', staples: '必需消费',
  healthcare: '医疗', utilities: '公用事业', financials: '金融股', real_estate: 'REITs', gold: '黄金',
  crude_oil: '原油', treasury_10y: '10年美债', cash: '现金' };
const INK = '#1d2433', INK3 = '#7a8396', LINE = '#e3e6ec', SPLIT = '#eef0f4';
const COL = { actual: INK, cond: '#3b5bdb', uncond: '#868e96', oracle: '#e8590c', va: '#2f9e44' };
const PARTS = [
  { key: 'scenario', name: '情景部分', color: '#1098ad',
    text: 'log(不知道宏观的中位数) − log(已知宏观的中位数)：知道每 24 个月的真实宏观之后，模拟中位数被推动了多少；它是“知道宏观”带来的修正而不是误差来源，修正过头时与另两部分符号相反。' },
  { key: 'engine', name: '引擎部分', color: '#d97706',
    text: 'log(已知宏观的中位数) − log(oracle)：同样知道宏观，引擎靠抽取历史月份、熵池化重加权和再锚定得到的结果，与“因子模型直接吃真实因子”之间的差距。' },
  { key: 'transmission', name: '传导部分', color: '#7048e8',
    text: 'log(oracle) − log(实际)：把真实宏观因子原样喂给训练期因子模型仍解释不了的部分，也就是模型缺的传导机制或这个时代特有的定价（例如估值扩张）。' },
];
const MACRO = [
  { key: 'cpi_yoy', sub: '条件模拟把每 24 个月的 CPI 同比均值设成真实值；同比是过去 12 个月的累计，抽中的历史月份“已经在目标附近”，所以模拟路径会滞后于实际拐点。' },
  { key: 'ff', sub: '视图约束政策利率的月度变化均值，水平从起点累加并向“中性利率 + 通胀”回归；看蓝色区间能否罩住真实的加息、降息路径。' },
  { key: 'y10', sub: '同样约束月度变化；10 年期收益率既决定美债回报，也是股票和 REITs 的贴现率，这里的偏离会传导到风险资产的误差上。' },
  { key: 'unrate', sub: '约束失业率的月度变化；失业率和工业产出一起刻画需求与劳动收入，主要影响周期股、金融股和消费股的传导。' },
];
// Exogenous shocks (config.EXOGENOUS_SHOCKS, backtest.json `shocks`): shaded on every path chart.
const SHOCK_NAME = '外生冲击';
const SHOCK_FILL = 'rgba(240,140,0,0.15)', SHOCK_KEY = 'rgba(240,140,0,0.45)', SHOCK_INK = '#b86e00';
// Custom windows (POST /api/backtest/run): presets, the bounds when backtest.json predates them, and the
// windows run during this page load (kept across re-renders and visits; the server caches them too).
const CUSTOM_PRESETS = [
  { te: '1987-12', m: 180, label: '1987-12 起 15 年', hint: '最早可选' },
  { te: '1999-12', m: 60, label: '1999-12 起 5 年', hint: '泡沫破裂' },
  { te: '2007-06', m: 36, label: '2007-06 起 3 年', hint: '金融危机' },
  { te: '2019-12', m: 24, label: '2019-12 起 2 年', hint: '疫情' },
  { te: '2021-12', m: 36, label: '2021-12 起 3 年', hint: '通胀与加息' },
];
const CUSTOM_DEFAULTS = { paths: 200, paths_range: [100, 400], months_range: [12, 180], earliest_train_end: '1987-12' };
const CUSTOM_KEEP = 4;                         // custom windows kept as pills
const CUSTOM_RUNS = [];                        // newest first: the server's window JSON + id
const CUSTOM_RE = /^custom=(\d{4}-\d{2}):(\d{1,3})(?::(\d{1,3}))?(?::(\d{1,10}))?$/;
const customId = (c) => `custom_${c.train_end}_${c.months}_${c.n_paths}_${c.seed}`;
const SHRINK = { infeasible: '视图无法同时满足', thin: '历史支撑太薄' };
const SHRINK_WHY = {
  infeasible: '训练期里找不到能同时满足全部视图的 6 个月区块组合',
  thin: '原样执行时有效月份或不同的 12 个月历史片段太少（模拟会反复抽同几段历史）',
};

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
// Fixed-point text without a "-0.00" and, when signed, with "+" only on values that round above zero.
function fixed(x, d, signed = false) {
  let s = x.toFixed(d);
  if (Number(s) === 0) s = (0).toFixed(d);
  return signed && Number(s) > 0 ? `+${s}` : s;
}
const pct = (v, d = 1, signed = false) => (isNum(v) ? `${fixed(v * 100, d, signed)}%` : '—');
const num = (v, d = 2, signed = false) => (isNum(v) ? fixed(v, d, signed) : '—');
const idx = (v) => (!isNum(v) ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : String(+v.toPrecision(3)));
const lvl = (v) => (isNum(v) ? `${v.toFixed(2)}%` : '—');
const span = (p, n = 7) => (Array.isArray(p) && p.length === 2 ? `${String(p[0]).slice(0, n)}–${String(p[1]).slice(0, n)}` : '—');
const escHtml = (t) => String(t ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const lineKey = (color, dash) => `<span style="display:inline-block;width:14px;border-top:2px ${dash ? 'dashed' : 'solid'} ${color};vertical-align:middle;margin-right:6px"></span>`;
// Years and months of a test length: 36 -> "3 年", 30 -> "30 个月".
const monthsText = (m) => (m % 12 === 0 ? `${m / 12} 年` : `${m} 个月`);
const addMonths = (ym, k) => {
  const [y, m] = String(ym).split('-').map(Number);
  const t = y * 12 + (m - 1) + k;
  return `${Math.floor(t / 12)}-${String((t % 12) + 1).padStart(2, '0')}`;
};
const monthsBetween = (a, b) => {
  const [ya, ma] = String(a).split('-').map(Number), [yb, mb] = String(b).split('-').map(Number);
  return (yb - ya) * 12 + (mb - ma);
};

// Shocks overlapping the category axis x ("YYYY-MM" labels), as shaded spans that include their last month.
function shockSpans(x, shocks) {
  if (!Array.isArray(shocks) || !Array.isArray(x) || !x.length) return [];
  const first = String(x[0]), last = String(x[x.length - 1]);
  return shocks.filter(s => s && s.start <= last && s.end >= first).map(s => {
    const from = s.start < first ? first : s.start;
    const end = s.end > last ? last : s.end;
    const i = x.indexOf(end);
    return { from, to: i >= 0 ? x[Math.min(i + 1, x.length - 1)] : null, name: s.name, start: s.start, end: s.end };
  }).filter(a => a.to && x.includes(a.from));
}
const boxKey = (color) => `<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${color};vertical-align:middle;margin-right:6px"></span>`;

const CSS = `/* backtest */
.backtest-bar { display: grid; gap: 8px; padding: 12px 16px; }
@media (min-width: 1101px) { .backtest-bar { position: sticky; top: var(--backtest-top, 62px); z-index: 4; } }
.backtest-bar-row { display: flex; gap: 10px; align-items: flex-start; }
.backtest-bar-label { flex: 0 0 60px; font-size: 12px; color: var(--ink-3); padding-top: 5px; }
.backtest-pills { display: flex; flex-wrap: wrap; gap: 6px; min-width: 0; }
.backtest-pill { font: inherit; font-size: 12px; border: 1px solid var(--line); background: var(--panel); color: var(--ink-2);
  border-radius: 999px; padding: 3px 11px; cursor: pointer; }
.backtest-pill:hover { border-color: #c6cbd6; }
.backtest-pill.key { border-color: #bac8ff; }
.backtest-pill.on { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
.backtest-pill small { opacity: .75; margin-left: 4px; font-size: 11px; }
.backtest-anchor { scroll-margin-top: 230px; }
.backtest-split { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 16px; align-items: start; }
.backtest-2col { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
@media (max-width: 1100px) { .backtest-split, .backtest-2col { grid-template-columns: minmax(0, 1fr); } }
.backtest-kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.backtest-kpis .kpi { padding: 10px 12px; box-shadow: none; }
.backtest-kpis .kpi-value { font-size: 18px; }
.backtest-mini { cursor: pointer; padding: 10px 12px 6px; }
.backtest-mini:hover { border-color: #c6cbd6; }
.backtest-mini.on { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.backtest-mini-head { display: grid; gap: 1px; font-size: 12px; }
.backtest-mini-row { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; white-space: nowrap; }
.backtest-mini-row b { font-size: 13px; font-weight: 600; }
.backtest-mini-row span, .backtest-mini-row.muted { color: var(--ink-3); }
.backtest-mini .chart { height: 140px; }
.backtest-parts { display: grid; gap: 6px; margin: 12px 0 0; padding: 0; list-style: none; font-size: 13px; color: var(--ink-2); }
.backtest-parts li { display: flex; gap: 8px; align-items: baseline; }
.backtest-parts i { flex: 0 0 10px; height: 10px; border-radius: 2px; display: inline-block; transform: translateY(1px); }
.backtest-text { color: var(--ink-2); font-size: 13px; margin: 10px 0 0; line-height: 1.75; }
.backtest-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.backtest-chips .chip, .backtest-kv-chip { white-space: nowrap; }
table.tbl td.backtest-wide { min-width: 200px; }
.backtest-sub { font-size: 12px; font-weight: 600; color: var(--ink-2); margin: 14px 0 6px; }
table.tbl tr.backtest-sel td, .tbl-stackbox table.tbl-stack tr.backtest-sel { background: var(--accent-soft); }
.backtest-macro .chart { height: 250px; }
.backtest-pill.custom { border-style: dashed; border-color: #e8590c; }
.backtest-pill.custom.on { background: #e8590c; border-color: #e8590c; border-style: solid; }
.backtest-custom { display: grid; gap: 10px; }
.backtest-custom .card-head { margin-bottom: 0; }
.backtest-custom-form { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: flex-end; }
.backtest-field { display: grid; gap: 4px; font-size: 12px; color: var(--ink-3); }
.backtest-month, .backtest-num { font: inherit; font-size: 14px; color: var(--ink); background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px; padding: 5px 9px; font-variant-numeric: tabular-nums; }
.backtest-month { width: 150px; }
.backtest-num { width: 96px; }
.backtest-more { font-size: 12px; color: var(--ink-2); align-self: center; }
.backtest-more summary { cursor: pointer; padding: 6px 0; }
.backtest-more[open] { flex-basis: 100%; }
.backtest-more .backtest-custom-form { margin-top: 6px; }
.backtest-hint { font-size: 12px; }
.backtest-custom-status { font-size: 13px; color: var(--ink-2); display: flex; flex-wrap: wrap; gap: 6px; align-items: center; min-height: 22px; }
.backtest-custom-status .notice { flex: 1 1 100%; }
.backtest-copy { font-size: 12px; padding: 2px 10px; }
.backtest-shock-note { font-size: 12px; margin-top: 8px; }
@media (max-width: 480px) { .backtest-month, .backtest-num { width: 100%; } .backtest-field { flex: 1 1 140px; } }
`;

function ensureStyle() {
  let el = document.getElementById('style-fe2r');
  if (!el) { el = document.createElement('style'); el.id = 'style-fe2r'; document.head.append(el); }
  if (!el.textContent.includes('/* backtest */')) el.textContent += CSS;
}

function failure(ui, err) {
  const { h } = ui;
  if (err?.status === 404) {
    return h('div', {}, ui.notice('回测结果（backtest.json）还没有生成。', 'info'), ui.buildHint(err));
  }
  return ui.notice(`读取回测结果失败：${err?.message || err}`, 'error');
}

// Wrapping button group (a seg control that does not overflow on narrow screens).
function pills(h, options, value, onChange) {
  const wrap = h('div', { class: 'backtest-pills' });
  const paint = (v) => wrap.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  for (const o of options) {
    wrap.append(h('button', { class: `backtest-pill ${o.cls || ''}`, 'data-v': o.value, title: o.title || null,
      onclick: () => { paint(o.value); onChange(o.value); } }, o.label, o.hint ? h('small', {}, o.hint) : null));
  }
  paint(value);
  wrap.set = paint;
  return wrap;
}

// One percentile band drawn as a single polygon: unlike stacked areas it also works on a log axis.
function bandSeries(name, lo, hi, color, opacity) {
  return {
    name, type: 'custom', silent: true, clip: true, z: 1, tooltip: { show: false },
    itemStyle: { color, opacity }, data: [[0, lo.find(isNum) ?? 1]],
    renderItem: (params, api) => {
      const top = [], bottom = [];
      hi.forEach((v, i) => {
        if (isNum(v) && isNum(lo[i])) { top.push(api.coord([i, v])); bottom.push(api.coord([i, lo[i]])); }
      });
      return { type: 'polygon', silent: true, shape: { points: top.concat(bottom.reverse()) }, style: { fill: color, opacity } };
    },
  };
}

// Invisible line so the value axis covers the band.
const ghost = (name, data) => ({ name, type: 'line', data, symbol: 'none', silent: true, tooltip: { show: false },
  lineStyle: { opacity: 0 }, itemStyle: { opacity: 0 }, z: 0 });

let measureCtx = null;
function textWidth(s, font = '12px sans-serif') {
  try {
    measureCtx = measureCtx || document.createElement('canvas').getContext('2d');
    measureCtx.font = font;
    return measureCtx.measureText(String(s)).width;
  } catch (_) { return [...String(s)].reduce((w, ch) => w + (ch.charCodeAt(0) > 255 ? 12 : 7), 0); }
}

// Number of rows a plain ECharts legend wraps into at the element's width: each item is the icon, a 5px gap and
// the text (measured in the legend's own 12px font), items 10px apart, 5px padding on both sides.
function legendRows(el, names, itemWidth = 16) {
  const w = (el.clientWidth || 600) - 10;
  let rows = 1, x = 0;
  for (const n of names) {
    const iw = itemWidth + 5 + textWidth(n);
    if (x > 0 && x + iw > w) { rows += 1; x = 0; }
    x += iw + 10;
  }
  return rows;
}

// Category labels that wrap inside their own slot instead of running into the neighbours.
function wrapTicks(chart, el, n, labels = []) {
  const width = Math.max(48, ((el.clientWidth || 500) - 90) / Math.max(n, 1) - 8);
  // room for the tallest wrapped label (a window name takes three lines on a phone)
  const px = (t) => [...String(t)].reduce((w, ch) => w + (/[\x00-\xff]/.test(ch) ? 6.6 : 11), 0);
  const lines = Math.max(1, ...labels.map(t => Math.ceil(px(t) / width)));
  chart.setOption({ grid: { bottom: Math.max(48, lines * 14 + 26) }, xAxis: { axisLabel: { interval: 0, rotate: 0, width, overflow: 'break', lineHeight: 14 } } });
}

// Replace a chart container's content with a message, disposing the chart first.
function clearChart(el, node) {
  if (el._chart) { el._chart.dispose(); el._chart = null; }
  el.style.height = 'auto';
  el.replaceChildren(node);
  el._cleared = true;
}

// Undo clearChart() before drawing into the box again (drop the message, restore the CSS height).
function fresh(el) {
  if (el._cleared) { el.replaceChildren(); el.style.height = ''; el._cleared = false; }
  return el;
}

// Paths on a monthly axis: optional percentile band, lines, shaded periods and exogenous shocks
// (`shocks`: config.EXOGENOUS_SHOCKS entries; an orange span each, named in the tooltip); linear or log axis.
function pathChart(charts, el, { x, band, lines, log = false, fmt = idx, tipFmt, areas, compact = false, zoom = false, shocks }) {
  const tf = tipFmt || fmt;
  fresh(el);
  const series = [];
  const spans = shockSpans(x, shocks);
  if (band) {
    series.push(ghost('_lo', band.p5), ghost('_hi', band.p95), bandSeries(`${band.label} 90%`, band.p5, band.p95, band.color, 0.11));
    if (band.p25 && band.p75) series.push(bandSeries(`${band.label} 50%`, band.p25, band.p75, band.color, 0.2));
  }
  lines.forEach((l, i) => series.push({
    name: l.name, type: 'line', data: l.data, symbol: 'none', showSymbol: false, z: l.z ?? 3, connectNulls: false,
    lineStyle: { width: l.width || (compact ? 1.5 : 2), color: l.color, type: l.dash || 'solid' }, itemStyle: { color: l.color },
    markArea: i === 0 && areas?.length ? {
      silent: true, label: { color: INK3, fontSize: 10, position: 'insideTop' },
      data: areas.map(a => [{ name: a.label, xAxis: a.from, itemStyle: { color: a.color || 'rgba(0,0,0,0)' } }, { xAxis: a.to }]),
    } : undefined,
  }));
  if (spans.length && lines.length) {
    // an invisible copy of the first line carries the shock spans, so the legend can switch them off
    series.push({ name: SHOCK_NAME, type: 'line', data: lines[0].data, symbol: 'none', showSymbol: false, silent: true, z: 0,
      lineStyle: { opacity: 0 }, itemStyle: { color: SHOCK_KEY }, tooltip: { show: false },
      markArea: { silent: true, itemStyle: { color: SHOCK_FILL }, label: { show: false },
        data: spans.map(a => [{ xAxis: a.from }, { xAxis: a.to }]) } });
  }
  const names = [...(band ? [`${band.label} 90%`, ...(band.p25 ? [`${band.label} 50%`] : [])] : []), ...lines.map(l => l.name),
    ...(spans.length && lines.length ? [SHOCK_NAME] : [])];
  const n = x.length;
  const rows = compact ? 0 : legendRows(el, names);
  if (!compact) {
    if (!el.dataset.baseH) el.dataset.baseH = String(el.clientHeight || 320);
    el.style.height = `${Number(el.dataset.baseH) + Math.max(0, rows - 1) * 22}px`;
  }
  const chart = charts.mount(el);
  chart.setOption({
    animation: !compact,
    legend: compact ? undefined : { top: 0, left: 0, right: 0, textStyle: { color: INK, fontSize: 12 }, itemWidth: 16, itemHeight: 8,
      data: names.map(n => (n === SHOCK_NAME ? { name: n, icon: 'rect', itemStyle: { color: SHOCK_KEY } } : n)),
      selected: Object.fromEntries(lines.filter(l => l.hidden).map(l => [l.name, false])) },
    tooltip: {
      trigger: 'axis', confine: true,
      formatter: (params) => {
        const ps = (params || []).filter(p => p.seriesType === 'line' && !String(p.seriesName).startsWith('_') && p.seriesName !== SHOCK_NAME);
        if (!ps.length) return '';
        const i = ps[0].dataIndex;
        const out = [`<b>${x[i]}</b>`];
        for (const p of ps) {
          const l = lines.find(q => q.name === p.seriesName);
          out.push(`${lineKey(p.color, l?.dash)}<b>${tf(l?.data?.[i])}</b> ${p.seriesName}`);
        }
        if (band) {
          if (band.p25) out.push(`${boxKey(band.color)}50%：${tf(band.p25[i])} – ${tf(band.p75[i])}`);
          out.push(`${boxKey(band.color)}90%：${tf(band.p5[i])} – ${tf(band.p95[i])}`);
        }
        const hit = spans.filter(a => a.start <= x[i] && x[i] <= a.end);
        if (hit.length) out.push(...hit.map(a => `<span style="color:${SHOCK_INK}">${boxKey(SHOCK_KEY)}${SHOCK_NAME}：${escHtml(a.name)}</span>`));
        return out.join('<br/>');
      },
    },
    grid: compact ? { left: 38, right: 14, top: 8, bottom: 20 }
      : { left: 52, right: 22, top: 12 + 22 * rows, bottom: zoom ? 58 : 30 },
    xAxis: {
      type: 'category', data: x, boundaryGap: false, axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false },
      axisLabel: compact ? { color: INK3, fontSize: 10, interval: (i) => i === 0 || i === n - 1, formatter: (v) => String(v).slice(0, 4) }
        : { color: INK3, fontSize: 11 },
    },
    yAxis: {
      type: log ? 'log' : 'value', logBase: 2, scale: !log, splitNumber: compact ? 3 : 5,
      axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: SPLIT } },
      axisLabel: { color: INK3, fontSize: compact ? 10 : 11, formatter: (v) => fmt(v) },
    },
    dataZoom: zoom ? [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }] : undefined,
    series,
  });
  return chart;
}

// Stacked scenario / engine / transmission parts per asset, with the total as a marker.
function decompChart(charts, el, ui, win, assets) {
  const chart = charts.mount(fresh(el));
  const dec = (a) => win.assets[a]?.decomposition || {};
  const val = (v) => (isNum(v) ? v : null);
  chart.setOption({
    legend: { top: 0, textStyle: { color: INK, fontSize: 12 }, itemWidth: 12, itemHeight: 8 },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true,
      formatter: (params) => {
        const a = assets[params?.[0]?.dataIndex];
        if (!a) return '';
        const d = dec(a);
        return [`<b>${ui.assetName(a)}</b>（年化对数误差，正数 = 预测偏高）`,
          ...PARTS.map(p => `${boxKey(p.color)}<b>${pct(d[`${p.key}_annual`], 1, true)}</b> ${p.name}`),
          `◆ <b>${pct(d.total_annual, 1, true)}</b> 总误差（不知道宏观 vs 实际）`].join('<br/>');
      },
    },
    grid: { left: 8, right: 28, top: 30, bottom: 24, containLabel: true },
    xAxis: { type: 'value', axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: SPLIT } },
      axisLabel: { color: INK3, fontSize: 11, formatter: (v) => pct(v, 0) } },
    yAxis: { type: 'category', inverse: true, data: assets.map(a => SHORT[a] || ui.assetName(a)),
      axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false }, axisLabel: { color: INK, fontSize: 12 } },
    series: [
      ...PARTS.map(p => ({ name: p.name, type: 'bar', stack: 'parts', barMaxWidth: 16,
        itemStyle: { color: p.color, borderColor: '#fff', borderWidth: 1 },
        data: assets.map(a => val(dec(a)[`${p.key}_annual`])) })),
      { name: '总误差', type: 'scatter', symbol: 'diamond', symbolSize: 11, z: 5,
        itemStyle: { color: INK, borderColor: '#fff', borderWidth: 1.5 },
        data: assets.map((a, i) => [val(dec(a).total_annual), i]) },
    ],
  });
  return chart;
}

// Realized | posterior | prior correlation matrices side by side, one colour scale.
function corrChart(charts, el, ui, corr) {
  const assets = corr.assets || [];
  const n = assets.length;
  const names = assets.map(a => SHORT[a] || ui.assetName(a));
  const mats = [
    { key: 'realized', title: '实际（测试期）', sub: '测试期真实月度收益' },
    { key: 'posterior', title: '后验（引擎模拟）', sub: `与实际的 MAE ${num(corr.mae_posterior)}` },
    { key: 'prior', title: '先验（训练期历史）', sub: `与实际的 MAE ${num(corr.mae_prior)}` },
  ].filter(m => Array.isArray(corr[m.key]));
  fresh(el);
  const w = el.clientWidth || 900;
  const L = (Math.min(92, Math.max(68, w * 0.09)) / w) * 100, G = 2.4, R = 1;
  const W = (100 - L - R - G * (mats.length - 1)) / mats.length;
  const cellPx = (w * W / 100) / Math.max(n, 1);
  const top = 46, bottom = 96;
  el.style.height = `${Math.round(top + n * Math.max(16, Math.min(cellPx, 32)) + bottom)}px`;
  const chart = charts.mount(el);
  const at = (m, r, c) => corr[m]?.[r]?.[c];
  const axis = (g, show) => ({
    gridIndex: g, type: 'category', data: names, axisTick: { show: false }, axisLine: { show: false }, splitArea: { show: false },
    axisLabel: { show, color: INK3, fontSize: 10, interval: 0 },
  });
  chart.setOption({
    title: mats.map((m, g) => ({ text: m.title, subtext: m.sub, left: `${L + g * (W + G)}%`, top: 0, itemGap: 3,
      textStyle: { fontSize: 12, fontWeight: 600, color: INK }, subtextStyle: { fontSize: 11, color: INK3 } })),
    tooltip: {
      confine: true,
      formatter: (p) => {
        const [c, r] = p.data.value || p.data;
        return `<b>${names[r]} × ${names[c]}</b><br/>实际 <b>${num(at('realized', r, c))}</b>　后验 <b>${num(at('posterior', r, c))}</b>　先验 <b>${num(at('prior', r, c))}</b>`;
      },
    },
    grid: mats.map((m, g) => ({ left: `${L + g * (W + G)}%`, width: `${W}%`, top, bottom })),
    xAxis: mats.map((m, g) => ({ ...axis(g, true), axisLabel: { color: INK3, fontSize: 10, interval: 0, rotate: 90 } })),
    yAxis: mats.map((m, g) => ({ ...axis(g, g === 0), inverse: true })),
    visualMap: { min: -1, max: 1, seriesIndex: mats.map((m, g) => g), orient: 'horizontal', left: 'center', bottom: 0,
      itemWidth: 10, itemHeight: 160, text: ['+1', '−1'], textStyle: { color: INK3, fontSize: 10 }, calculable: false,
      inRange: { color: ['#1c7ed6', '#a5d8ff', '#f1f3f5', '#ffc9c9', '#e03131'] } },
    series: mats.map((m, g) => ({
      type: 'heatmap', xAxisIndex: g, yAxisIndex: g,
      data: corr[m.key].flatMap((row, r) => row.map((v, c) => ({ value: [c, r, isNum(v) ? v : '-'] }))),
      label: { show: cellPx >= 27, fontSize: 9, color: INK, formatter: (p) => (isNum(p.value[2]) ? p.value[2].toFixed(2) : '') },
      itemStyle: { borderColor: '#fff', borderWidth: 1 },
      emphasis: { itemStyle: { borderColor: INK, borderWidth: 1 } },
    })),
  });
  return chart;
}

// "#backtest?<window>/<asset>/log" or "#backtest?custom=YYYY-MM:months[:paths[:seed]]/<asset>".
function parseParam(param, wins, assets) {
  const out = {};
  for (const tok of String(param || '').split(/[/&,]/)) {
    const c = CUSTOM_RE.exec(tok.trim());
    if (c) out.custom = { train_end: c[1], months: Number(c[2]), n_paths: c[3] ? Number(c[3]) : null, seed: c[4] ? Number(c[4]) : null };
    else if (/^custom=/.test(tok.trim())) out.badCustom = tok.trim();        // a shared link that does not parse: say so
    else if (wins.some(w => w.id === tok)) out.win = tok;
    else if (assets.includes(tok)) out.asset = tok;
    else if (tok === 'log') out.log = true;
  }
  return out;
}

// The deep-link token of a custom window: paths and seed only when they differ from the defaults.
function customParam(c, defPaths, defSeed) {
  let s = `custom=${c.train_end}:${c.months}`;
  if (c.n_paths !== defPaths || c.seed !== defSeed) s += `:${c.n_paths}`;
  if (c.seed !== defSeed) s += `:${c.seed}`;
  return s;
}

export async function render(root, { api, charts, ui, param, alive = () => root.isConnected, signal }) {
  const { h, card } = ui;
  ensureStyle();
  root.append(h('p', { class: 'lead' },
    '回测把历史切成训练期和测试期：因子模型、历史月份库和起点都只用训练期数据，然后在测试期比较四条线——',
    '实际走势；oracle（训练期因子模型直接喂入测试期真实的宏观因子，检验传导）；已知宏观的条件扇形图',
    '（引擎在每 24 个月真实宏观均值的约束下抽样，检验引擎；分两档：“宏观 + 金融条件（含市场价格）”和“只有宏观”）；不知道宏观的无条件中位数（基准）。',
    '它们之间的差距把误差拆成情景、引擎、传导三部分，告诉我们不准时该改哪一层。长期锚（中性利率、期限利差、信用利差、失业率）按每个窗口训练期末可知的规则估计。'));
  const wait = ui.loading('读取回测结果…');
  root.append(wait);
  let bt;
  try { bt = await api.get('backtest'); } catch (err) { wait.remove(); root.append(failure(ui, err)); return; }
  wait.remove();
  if (!alive()) return;                  // the user moved on while the results loaded

  const builtins = (bt.windows || []).filter(w => w && w.id && w.assets);
  if (!builtins.length) { root.append(ui.notice('backtest.json 里没有回测窗口，请重新运行 python -m scenariolab.build。', 'info')); return; }
  const wins = [...builtins, ...CUSTOM_RUNS];          // the window pills: built-in windows, then this page load's custom ones
  const present = new Set(Object.keys(builtins[0].assets));
  const assets = [...ORDER.filter(a => present.has(a)), ...[...present].filter(a => !ORDER.includes(a))];
  const structural = new Set(bt.transmission?.structural || ['treasury_10y', 'crude_oil', 'cash']);
  const nameOf = (a) => ui.META.assets[a]?.name || bt.asset_names?.[a] || a;
  const labelOf = (a) => `${nameOf(a)}${TICKER[a] ? `（${TICKER[a]}）` : ''}`;
  const CB = { ...CUSTOM_DEFAULTS, ...(bt.custom || {}) };
  const defSeed = isNum(bt.seed) ? bt.seed : 11;
  const fill = (c) => (c ? { ...c, n_paths: c.n_paths ?? CB.paths, seed: c.seed ?? defSeed } : null);
  const keyOf = (c) => `${c.train_end}:${c.months}:${c.n_paths}:${c.seed}`;
  const parsed = parseParam(param, wins, assets);
  const linked = fill(parsed.custom);
  const known = linked && CUSTOM_RUNS.find(w => keyOf(w.custom) === keyOf(linked));
  const st = { win: builtins[0].id, asset: present.has('us_equity') ? 'us_equity' : assets[0], log: false, phase: 0,
    ...parsed, pending: linked && !known ? linked : null };
  if (known) st.win = known.id;
  delete st.custom;
  delete st.badCustom;
  const curWin = () => wins.find(w => w.id === st.win) || wins[0];
  const sync = () => {
    const w = curWin();
    const head = st.pending ? customParam(st.pending, CB.paths, defSeed) : w.custom ? customParam(w.custom, CB.paths, defSeed) : w.id;
    ui.setRoute('backtest', `${head}/${st.asset}${st.log ? '/log' : ''}`, alive);
  };
  const shocksOf = (w) => (Array.isArray(w?.shocks) ? w.shocks : (bt.shocks || []));
  const tile = (label, value, sub, cls = '') => ui.kpi(label, value, sub, cls);
  const readable = (t) => (t ? String(t).replace(/\b[sf]_[a-z0-9_]+\b/g, (m) => ui.varName(m)) : '');
  const titled = (el) => el.querySelector('.card-title');
  const subtitled = (el) => el.querySelector('.card-sub');

  // ---- overall verdict across windows
  const S = bt.summary || {};
  const T = bt.transmission || {};
  root.append(h('div', { class: 'section-title' }, `总体结论（${builtins.length} 个回测窗口，风险资产平均）`));
  const arm = bt.arm_names || { cond: '宏观 + 金融条件（含市场价格）', cond_macro: '只有宏观' };
  const rq = blockR2(T, bt.risk_assets || [], 3), ry = blockR2(T, bt.risk_assets || [], 12);
  root.append(h('div', { class: 'grid cols-4' },
    tile(`已知${arm.cond}时的年化误差`, pct(S.cagr_mae_cond_risk), `${arm.cond_macro} ${pct(S.cagr_mae_cond_macro_risk)} · 不知道宏观 ${pct(S.cagr_mae_uncond_risk)}（年化收益中位数与实际之差的绝对值）`,
      isNum(S.cagr_mae_cond_risk) && isNum(S.cagr_mae_uncond_risk) && S.cagr_mae_cond_risk > S.cagr_mae_uncond_risk ? 'is-bad' : ''),
    tile('90% 区间覆盖率', pct(S.coverage_cond_risk, 0), `${arm.cond_macro} ${pct(S.coverage_cond_macro_risk, 0)} · 不知道宏观 ${pct(S.coverage_uncond_risk, 0)}；校准良好时约 90%`),
    tile('终值落在 5%/95% 尾部之外', pct(S.pit_tail_share_cond, 0), `${arm.cond_macro} ${pct(S.pit_tail_share_cond_macro, 0)}；${S.n_pit ?? '—'} 个资产终值；校准良好时约 10%`),
    tile('因子解释的季度 / 年度价格波动 R²', `${pct(rq.r2, 0)} / ${pct(ry.r2, 0)}`,
      `样本外 ${pct(rq.oos, 0)} / ${pct(ry.oos, 0)}（${rq.n} 个季度、${ry.n} 个年度区块；以之前的平均为基准）。月度（参考）：样本内 ${pct(T.r2_risk_avg, 0)}，样本外 ${pct(T.r2_oos_risk_avg, 0)}`)));
  root.append(h('div', { style: { height: '16px' } }), anchorsCard(h, card, ui, bt, builtins));
  if (S.text) root.append(h('div', { style: { height: '16px' } }), card({}, h('p', { class: 'backtest-text', style: { marginTop: 0 } }, S.text)));

  // ---- custom window: train up to any month, test the next 1-15 years (POST /api/backtest/run)
  const lastMonth = CB.last_month || T.end || builtins[0]?.test?.[1] || null;
  const [minM, maxM] = CB.months_range;
  const latestTe = lastMonth ? addMonths(lastMonth, -minM) : null;
  const custom = { paths: CB.paths };
  const startTe = st.pending?.train_end || '2007-06', startM = st.pending?.months || 36;
  const teIn = h('input', { type: 'month', class: 'backtest-month', value: startTe, min: CB.earliest_train_end, max: latestTe,
    placeholder: 'YYYY-MM', 'aria-label': '训练期截止月份（YYYY-MM）' });
  const mIn = h('input', { type: 'number', class: 'backtest-num', value: String(startM), min: String(minM), max: String(maxM), step: '1',
    'aria-label': '测试期月数' });
  const seedIn = h('input', { type: 'number', class: 'backtest-num', value: String(st.pending?.seed ?? defSeed), min: '0', step: '1',
    'aria-label': '随机种子' });
  if (st.pending?.n_paths) custom.paths = st.pending.n_paths;
  const pathOpts = [100, 200, 400].filter(n => n >= CB.paths_range[0] && n <= CB.paths_range[1]);
  if (!pathOpts.includes(custom.paths)) pathOpts.push(custom.paths);
  const pathsSeg = ui.seg(pathOpts.sort((a, b) => a - b).map(n => ({ value: String(n), label: `${n} 条` })), String(custom.paths),
    (v) => { custom.paths = Number(v); });
  const mHint = h('div', { class: 'muted backtest-hint' });
  const runBtn = h('button', { class: 'btn primary', type: 'button', onclick: () => runCustom({ select: true, scroll: true }) }, '运行回测');
  const status = h('div', { class: 'backtest-custom-status', role: 'status', 'aria-live': 'polite' });
  const presetPills = pills(h, CUSTOM_PRESETS.map(p => ({ value: `${p.te}:${p.m}`, label: p.label, hint: p.hint,
    title: `训练 1973-01 至 ${p.te}，测试之后 ${monthsText(p.m)}` })), null, (v) => {
    const [te, m] = v.split(':');
    teIn.value = te; mIn.value = m;
    paintHint();
    runCustom({ select: true, scroll: true });
  });
  function paintHint() {
    const te = String(teIn.value || '').trim(), m = Number(mIn.value);
    let txt = '';
    if (/^\d{4}-\d{2}$/.test(te) && Number.isInteger(m) && m > 0) {
      const room = lastMonth ? monthsBetween(te, lastMonth) : null;
      txt = `训练 1973-01 至 ${te}，测试 ${addMonths(te, 1)} 至 ${addMonths(te, m)}（${monthsText(m)}）`
        + (room !== null && room >= minM ? `；从 ${te} 起最多 ${Math.min(room, maxM)} 个月` : '');
    }
    mHint.textContent = txt;
    presetPills.set(`${te}:${m}`);
  }
  for (const el of [teIn, mIn, seedIn]) {
    el.addEventListener('input', paintHint);
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter') runCustom({ select: true, scroll: true }); });
  }
  const field = (label, ...body) => h('label', { class: 'backtest-field' }, h('span', {}, label), ...body);
  root.append(h('div', { class: 'section-title' }, '自选时间段：拿一段新的历史来回测'),
    card({ title: '自选时间段', cls: 'backtest-custom',
      sub: `选训练期的截止月份和测试期长度：因子模型、历史月份库和起点只用截止月及以前的数据，然后在之后的月份上与实际比较，`
        + '口径与 4 个内置窗口完全相同（已知宏观的条件扇形图把测试期真实的宏观均值当作视图）。测试期 3–5 年（12–20 个季度）的结果每个十年只有 2–3 个独立窗口，只能当个案。结果作为「自选窗口」加入下面的窗口选择，'
        + `所有图表随之切换；地址栏的链接可以直接分享（#backtest?custom=2007-06:36）。训练期至少 15 年（最早 ${CB.earliest_train_end}），`
        + `测试期 ${minM}–${maxM} 个月，且不能超过数据的最后一个月${lastMonth ? `（${lastMonth}）` : ''}。` },
    h('div', { class: 'backtest-bar-row' }, h('div', { class: 'backtest-bar-label' }, '常用'), presetPills),
    h('div', { class: 'backtest-custom-form' },
      field('训练期截止', teIn), field('测试期（月）', mIn),
      h('details', { class: 'backtest-more' }, h('summary', {}, '路径数与随机种子'),
        h('div', { class: 'backtest-custom-form' }, field('每组路径数', pathsSeg), field('随机种子', seedIn))),
      runBtn),
    mHint, status));
  paintHint();
  if (parsed.badCustom) {
    status.replaceChildren(ui.notice(`链接里的「${parsed.badCustom}」无法识别：自选窗口的格式是 custom=YYYY-MM:测试期月数（可选 :路径数:随机种子），`
      + '例如 custom=2007-06:36，所以下面显示的是内置窗口。', 'error'));
  }

  let busy = false, tick = null;
  if (signal) signal.addEventListener('abort', () => clearInterval(tick));
  async function runCustom({ select = true, scroll = false, req = null, fromLink = false } = {}) {
    const body = req || { train_end: String(teIn.value || '').trim(), months: Number(mIn.value), n_paths: custom.paths,
      seed: seedIn.value === '' ? defSeed : Number(seedIn.value) };
    if (!/^\d{4}-\d{2}$/.test(body.train_end)) {
      status.replaceChildren(ui.notice('训练期截止月份的格式应为 YYYY-MM，例如 2007-06。', 'error'));
      return;
    }
    const done = CUSTOM_RUNS.find(w => keyOf(w.custom) === keyOf(body));
    if (done) { addCustom(done, { select, scroll, secs: null }); return; }
    if (busy) { status.append(h('div', { class: 'muted' }, '上一个回测还在计算，完成后再试。')); return; }
    busy = true;
    runBtn.disabled = true;
    const t0 = performance.now();
    const what = `训练到 ${body.train_end}、测试之后 ${Number.isFinite(body.months) ? monthsText(body.months) : '—'}，4 组 × ${body.n_paths} 条路径`;
    const paint = () => status.replaceChildren(h('span', { class: 'spinner' }),
      ` 正在回测（${what}）… ${Math.round((performance.now() - t0) / 1000)} 秒`,
      h('span', { class: 'muted' }, '；15 年的窗口约需 10 秒，算过的窗口会直接返回'));
    paint();
    clearInterval(tick);
    tick = setInterval(() => { if (!alive()) { clearInterval(tick); return; } paint(); }, 1000);
    let res = null, err = null;
    try { res = await api.post('backtest/run', body); } catch (e) { err = e; }
    clearInterval(tick);
    busy = false;
    runBtn.disabled = false;
    if (!alive()) return;
    if (err || !res?.assets || !res?.custom) {
      const msg = err ? (err.status === 400 ? err.message : `回测失败（${err.status || '无法连接本地服务'}）：${err.message}`) : '回测结果不完整';
      status.replaceChildren(ui.notice(msg, 'error'));
      if (fromLink) { st.pending = null; sync(); }
      return;
    }
    res.id = customId(res.custom);
    res.name = '自选窗口';
    const rest = CUSTOM_RUNS.filter(w => w.id !== res.id).slice(0, CUSTOM_KEEP - 1);
    CUSTOM_RUNS.length = 0;
    CUSTOM_RUNS.push(res, ...rest);
    addCustom(res, { select: select && (!fromLink || st.pending !== null), scroll, secs: (performance.now() - t0) / 1000 });
  }

  function addCustom(win, { select, scroll, secs }) {
    wins.splice(builtins.length, wins.length - builtins.length, ...CUSTOM_RUNS);
    const next = windowPills();
    winPills.replaceWith(next);
    winPills = next;
    if (select) { st.win = win.id; st.phase = 0; }
    st.pending = null;
    if (select) renderWindow(); else winPills.set(st.win);
    sync();
    const c = win.custom;
    const where = `${win.label || win.name}（训练 ${span(win.train)}，测试 ${span(win.test)}，${c.n_paths} 条路径，种子 ${c.seed}）`;
    const took = secs === null ? '本页已经算过，直接显示' : isNum(win.elapsed_s) && secs > win.elapsed_s * 0.5
      ? `服务器计算 ${win.elapsed_s.toFixed(1)} 秒` : '服务器已缓存，直接返回';
    const copy = h('button', { class: 'btn backtest-copy', type: 'button', onclick: async () => {
      try { await navigator.clipboard.writeText(location.href); copy.textContent = '已复制'; } catch (_) { copy.textContent = '请从地址栏复制'; }
    } }, '复制链接');
    status.replaceChildren(h('span', { class: 'chip good' }, '完成'), ` ${where}；${took}。`, copy);
    if (scroll) winTitle.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ---- controls: one row scoping everything below
  const windowPills = () => pills(h, wins.map(w => (w.custom
    ? { value: w.id, label: '自选窗口', hint: `${w.custom.train_end} 起 ${monthsText(w.custom.months)}`, cls: 'custom',
      title: `训练 ${span(w.train)}，测试 ${span(w.test)}，${w.custom.n_paths} 条路径，种子 ${w.custom.seed}` }
    : { value: w.id, label: w.name || w.id, hint: span(w.test, 4), title: `训练 ${span(w.train)}，测试 ${span(w.test)}` })),
  st.win, (v) => { st.win = v; st.phase = 0; st.pending = null; renderWindow(); sync(); });
  let winPills = windowPills();
  const assetPills = pills(h, assets.map(a => ({ value: a, label: SHORT[a] || nameOf(a), hint: TICKER[a], cls: KEY.has(a) ? 'key' : '',
    title: nameOf(a) })), st.asset, (v) => selectAsset(v));
  const scaleSeg = ui.seg([{ value: 'linear', label: '线性' }, { value: 'log', label: '对数' }], st.log ? 'log' : 'linear',
    (v) => { st.log = v === 'log'; renderMain(); renderMinis(); sync(); });
  const bar = card({ cls: 'backtest-bar' },
    h('div', { class: 'backtest-bar-row' }, h('div', { class: 'backtest-bar-label' }, '回测窗口'), winPills),
    h('div', { class: 'backtest-bar-row' }, h('div', { class: 'backtest-bar-label' }, '资产'), assetPills),
    h('div', { class: 'backtest-bar-row' }, h('div', { class: 'backtest-bar-label' }, '纵轴'), scaleSeg,
      h('span', { class: 'muted', style: { fontSize: '12px', paddingTop: '4px' } }, '蓝框资产为重点关注的大类；对数坐标下等距 = 等比例涨跌')));
  root.append(h('div', { class: 'section-title' }, '选择窗口与资产'), bar);
  bar.style.setProperty('--backtest-top', `${document.querySelector('.topbar')?.offsetHeight || 62}px`);

  // ---- window overview
  const winTitle = h('div', { class: 'section-title backtest-anchor' });
  const winKpis = h('div', { class: 'grid cols-4' });
  root.append(winTitle, winKpis);

  // ---- main chart + asset KPIs
  const mainBox = ui.chartBox('tall');
  const shockNote = h('p', { class: 'backtest-text backtest-shock-note' });
  const mainCard = card({ title: ' ', cls: 'backtest-anchor',
    sub: '黑线是实际累计总回报；橙色虚线 oracle = 训练期因子模型喂入真实宏观因子；蓝色扇形 = 引擎在知道每 24 个月真实宏观均值时的模拟（深色 50%、浅色 90%），灰色虚线 = 不知道宏观时的中位数——黑线偏离橙线是传导没抓住，橙线偏离蓝线是引擎的抽样与重加权问题。橙色竖带 = 外生冲击（战争、疫情、石油禁运等，任何宏观因子都无法预先知道）。' },
  mainBox, shockNote);
  const assetKpis = h('div', { class: 'backtest-kpis' });
  root.append(h('div', { class: 'section-title' }, '单个资产：模拟 vs 实际'),
    h('div', { class: 'backtest-split' }, mainCard, card({ title: '这个资产的回测指标', sub: '年化 = 测试期终值折年（名义总回报，含股息，未扣通胀）；“实际”指真实发生的走势；覆盖率 = 实际值落在模拟 5%–95% 区间内的月份占比；PIT = 模拟终值低于实际终值的路径占比' }, assetKpis)));

  // ---- small multiples + table twin
  const minis = assets.map(a => {
    const head = h('div', { class: 'backtest-mini-head' });
    const box = h('div', { class: 'chart' });
    const el = h('div', { class: `card backtest-mini${KEY.has(a) ? ' key' : ''}`, title: '点击在上方主图查看', onclick: () => selectAsset(a, true) }, head, box);
    return { a, el, head, box };
  });
  const miniCard = card({ title: ' ', sub: '每格：蓝色为已知宏观时的模拟中位数与 50% / 90% 区间，黑线为实际；黑线越常跑出区间，说明这类资产在这个时代的传导越难复现（点击切换上方主图）。' },
    h('div', { class: 'grid cols-4' }, minis.map(m => m.el)));
  const tableBox = h('div');
  const tableCard = card({ title: ' ', sub: '与小图对应的数字（年化均为名义总回报）：误差 = 模拟中位数年化 − 实际年化；“主要误差来源”取引擎部分与传导部分中绝对值较大者（点击行切换资产）。' }, tableBox);
  root.append(h('div', { class: 'section-title' }, '所有资产一览'), miniCard, h('div', { style: { height: '16px' } }), tableCard);

  // ---- error decomposition + valuation-adjusted CMA
  const decBox = ui.chartBox('tall');
  const decText = h('p', { class: 'backtest-text' });
  const decCard = card({ title: '误差拆解：情景 / 引擎 / 传导（年化对数误差）',
    sub: '把“不知道宏观时的中位数 vs 实际”的终值误差拆成三段，三段之和 = ◆ 总误差；正数 = 预测偏高。哪一段长，就说明该改哪一层：引擎段长改抽样与重加权，传导段长改因子模型。' },
  decBox,
  h('ul', { class: 'backtest-parts' }, PARTS.map(p => h('li', {}, h('i', { style: { background: p.color } }), h('span', {}, h('b', {}, `${p.name}：`), p.text)))),
  h('p', { class: 'backtest-text', style: { fontSize: '12px' } }, '怎么改：传导部分（因子模型）的候选改动与样本外检验见',
    h('a', { href: '#diagnostics?loop' }, '「诊断与迭代 · 走步迭代」'), '，引擎部分（抽样、重加权、CMA）的消融试验见',
    h('a', { href: '#diagnostics?ablation' }, '「诊断与迭代 · 引擎消融」'), '。'),
  h('div', { class: 'backtest-sub' }, '本窗口解读'), decText);
  const vaBox = h('div');
  const vaCard = card({ title: '估值调整 CMA 有没有帮助', sub: '同一组视图，把漂移从“训练期历史平均”换成“按起点估值向历史中位数部分回归”（黄金不做估值调整）；数值越小越好，覆盖率越接近 90% 越好。' }, vaBox);
  root.append(h('div', { class: 'section-title' }, '误差从哪里来'), h('div', { class: 'backtest-split' }, decCard, vaCard));

  // ---- correlations
  const corrBox = h('div', { class: 'chart' });
  const corrNote = h('p', { class: 'backtest-text' });
  const pairBox = h('div');
  root.append(h('div', { class: 'section-title' }, '后验相关系数'),
    card({ title: '资产相关：实际 vs 后验（引擎模拟）vs 先验（训练期）',
      sub: '左为测试期真实月度收益的相关，中为引擎在知道宏观后模拟路径的相关（后验），右为训练期历史相关（先验）；MAE = 非对角元素的平均绝对差，后验 MAE 小于先验才说明情景重加权让资产联动更接近现实。' },
    ui.wideChart(corrBox, 640), corrNote, h('div', { class: 'backtest-sub' }, '后验与实际差距最大的资产对'), pairBox));    // three 12 x 12 matrices

  // ---- macro check
  const macro = MACRO.map(m => {
    const box = h('div', { class: 'chart' });
    return { ...m, box, el: card({ title: `${ui.stateName(m.key)}（%）：实际 vs 模拟`, sub: m.sub, cls: 'backtest-macro' }, box) };
  });
  root.append(h('div', { class: 'section-title' }, '宏观核对：引擎抽到的宏观路径像不像真实的'),
    h('div', { class: 'backtest-2col' }, macro.map(m => m.el)));

  // ---- phases and analog months
  const phaseBox = h('div');
  const viewsBox = h('div');
  const viewsCard = card({ title: ' ', sub: '真实 = 测试期该阶段的实际均值；目标 = 收缩后交给熵池化的视图；期望 = 熵池化权重下抽中月份的期望均值；抽中 = 模拟实际抽到的月份均值；模拟 = 模拟路径上的值（抽到的变化再经利率按水平缩放、向中性利率和油价中位数回归之后，也就是资产真正面对的宏观）；状态变量为水平（%），因子为月度变化折成每年；偏离以训练期 6 个月均值的标准差 σ 计。' }, viewsBox);
  root.append(h('div', { class: 'section-title' }, '阶段与类比月份：引擎用了哪些真实历史月份'),
    card({ title: '每 24 个月一个阶段的熵池化结果',
      sub: '熵池化把训练期月份重新加权，使抽中月份的 8 个宏观变量均值等于该阶段的真实值；ENS = 有效月份数（括号内为占训练期月份的比例，越小越集中），λ < 1 表示视图在训练期无法同时满足而向先验收缩，右侧是权重最高的历史月份（× = 权重是先验的倍数）；点击行查看该阶段的视图。' },
    phaseBox), h('div', { style: { height: '16px' } }), viewsCard);

  // ---- transmission: actual vs factor-fitted over the whole sample
  const transBox = ui.chartBox('tall');
  const transChips = h('div', { class: 'backtest-chips' });
  const transCard = card({ title: ' ', actions: transChips,
    sub: `把真实的月度因子变化喂给因子模型（对数坐标）：蓝线为全样本估计（样本内），橙线从 ${T.oos_start || '—'} 起每 ${T.refit_every || 60} 个月只用之前的数据重估、预测下一段（样本外）；两线贴近黑线说明宏观因子抓住了长期走势，样本外越偏说明传导随时代改变，阴影为当前回测窗口的测试期。` },
  transBox);
  const r2Box = ui.chartBox('tall');
  const r2Table = h('div');
  root.append(h('div', { class: 'section-title' }, '传导：因子模型能解释多少'), transCard, h('div', { style: { height: '16px' } }),
    h('div', { class: 'backtest-2col' },
      card({ title: '因子模型解释的风险资产月度价格波动（R²）',
        sub: '样本内用全样本估计；样本外以当时可得的训练期均值为基准（Campbell–Thompson），大于 0 才表示真实因子变化好过历史均值；样本外比同期样本内低得多，说明传导系数随时代在变。' }, r2Box),
      card({ title: 'R² 明细', sub: '“同期样本内”是全样本模型在样本外那批月份上的 R²，与样本外对比才是同口径的差距；国债和原油的“因子”就是收益率与油价本身，R² 接近 1 不代表传导更好。' }, r2Table)));

  // ---- cross-window summary
  const crossBox = h('div');
  const crossMae = ui.chartBox();
  const crossParts = ui.chartBox();
  const partsCard = card({ title: '各窗口的误差来源（年化绝对值平均）',
    sub: '情景部分衡量知道宏观让中位数移动了多少，引擎部分和传导部分才是剩下的误差：引擎柱高就改抽样与重加权，传导柱高就是因子模型缺了渠道。' }, crossParts);
  root.append(h('div', { class: 'section-title' }, '跨窗口汇总'),
    card({ title: `${builtins.length} 个内置回测窗口汇总（风险资产平均）`,
      sub: '条件 = 已知每 24 个月的真实宏观均值，无条件 = 不知道宏观；年化误差是年化收益中位数与实际之差的绝对值平均；点击行切换到该窗口。' }, crossBox),
    h('div', { style: { height: '16px' } }),
    h('div', { class: 'backtest-2col' },
      card({ title: '各窗口的年化误差：已知宏观 / 不知道宏观 / oracle',
        sub: '蓝柱低于灰柱说明知道宏观有用；橙柱是只看传导（因子模型吃真实宏观）的误差，蓝柱与橙柱的差主要是引擎造成的。' }, crossMae),
      partsCard));

  // ---- notes (the JSON's wording names code fields; say what they are on this page instead)
  const units = bt.units || {};
  const UNIT_LABEL = { index: '累计指数', cagr: '年化收益', coverage: '覆盖率', pit: 'PIT', pit_mad: 'PIT 偏离', decomposition: '误差拆解',
    r2: 'R²', macro: '宏观单位', shrink: 'λ 收缩', view_gap_sd: '视图偏离 σ' };
  const PLAIN = [[/（fitted、fitted_oos、oracle）/g, '（样本内拟合、样本外拟合、oracle）'], [/\bfitted_oos\b/g, '样本外拟合'],
    [/\bfitted\b/g, '样本内拟合'], [/（r2_oos、r2_oracle）/g, '（样本外 R²、oracle 的 R²）'], [/\bep_gap_sd\b/g, '熵池化偏离'],
    [/\bdrawn_gap_sd\b/g, '抽中偏离'], [/\bsim_gap_sd\b/g, '模拟偏离'], [/\*_annual 为每年/g, '“年化”为每年'],
    [/只有不到 MIN_ENS 个有效月份支撑时/g, '支撑它的有效月份少于下限时'],
    [/不到 MIN_ENS = (\d+) 个有效月份或 MIN_EPISODES = (\d+) 段不同的 12 个月历史片段/g, '有效月份不到 $1 个，或不同的 12 个月历史片段不到 $2 段'],
    [/\bMIN_ENS\b/g, '有效月份下限'], [/\bMIN_EPISODES\b/g, '历史片段下限'],
    [/结果在 summary\.valuation_adjusted/g, '结果见“估值调整 CMA 有没有帮助”'], [/（见 ep\[\*\]\.shrink）/g, '（见阶段表中的 λ）'],
    [/\btransmission 里/g, '“传导”图里'], [/cpi_yoy、ff、y10、unrate/g, 'CPI 同比、联邦基金利率、10 年期收益率、失业率']];
  const plain = (t) => PLAIN.reduce((s, [re, to]) => s.replace(re, to), String(t ?? ''));
  root.append(h('div', { class: 'section-title' }, '口径与说明'),
    h('div', { class: 'grid cols-2 text' },
      card({ title: '口径' }, h('dl', { class: 'kv' }, Object.entries(units).flatMap(([k, v]) => [h('dt', {}, UNIT_LABEL[k] || k), h('dd', {}, plain(v))]))),
      card({ title: '说明' }, h('ul', { class: 'list' }, (bt.notes || []).map(n => h('li', { class: 'muted' }, plain(n)))))));

  // ================================================================ renderers
  function renderWindow() {
    const win = curWin();
    const s = win.summary || {};
    winPills.set(win.id);
    winTitle.textContent = `窗口：${win.name || win.id}（训练 ${span(win.train)}，测试 ${span(win.test)}${isNum(s.years) ? `，${s.years.toFixed(1)} 年` : ''}）`;
    const better = isNum(s.cagr_mae_cond_risk) && isNum(s.cagr_mae_uncond_risk) && s.cagr_mae_cond_risk < s.cagr_mae_uncond_risk;
    winKpis.replaceChildren(
      tile('风险资产年化误差（已知宏观）', pct(s.cagr_mae_cond_risk), `不知道宏观 ${pct(s.cagr_mae_uncond_risk)} · 只看传导（oracle）${pct(s.cagr_mae_oracle_risk)}`,
        isNum(s.cagr_mae_cond_risk) && isNum(s.cagr_mae_uncond_risk) ? ui.goodCls(better) : ''),
      tile('90% 区间覆盖率', pct(s.coverage_cond_risk, 0), `不知道宏观 ${pct(s.coverage_uncond_risk, 0)}；校准良好约 90%`),
      tile('终值 PIT 平均偏离 |PIT − 0.5|', num(s.pit_cond_mad), `不知道宏观 ${num(s.pit_uncond_mad)}；校准良好约 0.25`),
      tile('相关矩阵 MAE（后验 / 先验）', `${num(s.corr_mae_posterior)} / ${num(s.corr_mae_prior)}`,
        isNum(s.corr_mae_posterior) && isNum(s.corr_mae_prior) && s.corr_mae_posterior < s.corr_mae_prior - 0.01 ? '情景重加权让资产联动更接近实际' : '情景重加权没有让联动明显更接近实际'));
    titled(miniCard).textContent = `全部 ${assets.length} 类资产（${win.name || win.id}，测试 ${span(win.test, 4)}）`;
    titled(tableCard).textContent = `各资产回测指标（${win.name || win.id}）`;
    renderMain(); renderMinis(); renderTable(); renderDecomp(); renderVA(); renderCorr(); renderMacro(); renderPhases(); renderTransmission();
  }

  function selectAsset(a, scroll = false) {
    st.asset = a;
    assetPills.set(a);
    renderMain(); renderTable(); renderTransmission();
    minis.forEach(m => m.el.classList.toggle('on', m.a === a));
    sync();
    if (scroll) mainCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function phaseAreas(win, label = true) {
    const d = win.dates || [];
    let s = 0;
    return (win.phases || []).map((m, k) => {
      const a = { from: d[s], to: d[Math.min(s + m, d.length - 1)], label: label ? `阶段 ${k + 1}` : '', color: k % 2 === 0 ? 'rgba(59,91,219,0.05)' : null };
      s += m;
      return a;
    }).filter(a => a.from && a.to);
  }

  function renderMain() {
    const win = curWin();
    const A = win.assets[st.asset];
    titled(mainCard).textContent = `${labelOf(st.asset)}：实际 vs 模拟（${win.name || win.id}，测试 ${span(win.test, 4)}）`;
    if (!A) {
      clearChart(mainBox, ui.notice('这个窗口没有该资产的数据。', 'info'));
      assetKpis.replaceChildren(ui.notice('这个窗口没有该资产的数据。', 'info'));
      return;
    }
    const fc = A.fan_cond || {}, fu = A.fan_uncond || {}, va = A.fan_cond_va || {}, fm = A.fan_cond_macro || {};
    const shocks = shocksOf(win);
    const inTest = shockSpans(win.dates || A.dates, shocks);
    shockNote.textContent = inTest.length
      ? `测试期内的外生冲击（橙色竖带）：${inTest.map(s => `${s.name}（${s.start} 至 ${s.end}）`).join('、')}。这些月份的偏离不算模型错误：任何宏观因子都无法事先知道它们。`
      : '测试期内没有外生冲击（战争、疫情、恐怖袭击、石油禁运等）。';
    pathChart(charts, mainBox, {
      x: A.dates || win.dates, log: st.log, areas: phaseAreas(win), shocks,
      band: Array.isArray(fc.p5) && Array.isArray(fc.p95) ? { label: '宏观 + 金融条件', color: COL.cond, ...fc } : null,
      lines: [
        { name: '实际', data: A.actual, color: COL.actual, width: 2.4, z: 4 },
        { name: '宏观 + 金融条件·中位数', data: fc.p50, color: COL.cond },
        { name: '只有宏观·中位数', data: fm.p50, color: '#7048e8', dash: 'dashed' },
        { name: '不知道宏观·中位数', data: fu.p50, color: COL.uncond, dash: 'dashed' },
        { name: 'oracle（因子模型×真实宏观）', data: A.oracle, color: COL.oracle, dash: 'dashed' },
        { name: '估值调整 CMA·中位数', data: va.p50, color: COL.va, dash: 'dotted', hidden: true },
      ].filter(l => Array.isArray(l.data)),
    });
    const d = A.decomposition || {};
    const main = ['engine', 'transmission'].reduce((b, k) => (Math.abs(d[`${k}_annual`] ?? 0) > Math.abs(d[`${b}_annual`] ?? 0) ? k : b), 'engine');
    const err = (v) => (isNum(v) && isNum(A.cagr_actual) ? v - A.cagr_actual : null);
    const tail = (p) => (isNum(p) && (p < 0.05 || p > 0.95) ? 'is-bad' : '');
    assetKpis.replaceChildren(
      tile('实际走势·年化', pct(A.cagr_actual), `测试期 ${span(win.test, 4)} 名义累计总回报折年`, ui.signCls(A.cagr_actual)),
      tile('宏观 + 金融条件·中位数', pct(A.cagr_pred_cond_p50), `误差 ${pct(err(A.cagr_pred_cond_p50), 1, true)}/年`),
      tile('只有宏观·中位数', pct(A.cagr_pred_cond_macro_p50), `误差 ${pct(err(A.cagr_pred_cond_macro_p50), 1, true)}/年 · 覆盖 ${pct(A.coverage_cond_macro, 0)}`),
      tile('不知道宏观·中位数', pct(A.cagr_pred_uncond_p50), `误差 ${pct(err(A.cagr_pred_uncond_p50), 1, true)}/年`),
      tile('oracle', pct(A.cagr_oracle), structural.has(st.asset) ? '模拟器的结构规则（收益率、油价、利率）'
        : `传导误差 ${pct(err(A.cagr_oracle), 1, true)}/年 · 测试期月度 R² ${pct(A.r2_oracle, 0)}${isNum(A.r2_oracle) && A.r2_oracle < 0 ? '（< 0 = 不如历史均值）' : ''}`),
      tile('90% 区间覆盖', pct(A.coverage_cond, 0), `不知道宏观 ${pct(A.coverage_uncond, 0)}；目标约 90%`,
        isNum(A.coverage_cond) && A.coverage_cond < 0.6 ? 'is-bad' : ''),
      tile('终值 PIT', num(A.pit_terminal_cond), `不知道宏观 ${num(A.pit_terminal_uncond)}；0.5 = 实际正好在中位数`, tail(A.pit_terminal_cond)),
      tile('估值调整 CMA·中位数', pct(A.cagr_pred_cond_va_p50), `覆盖 ${pct(A.coverage_cond_va, 0)} · PIT ${num(A.pit_terminal_cond_va)}`),
      tile('主要误差来源', PARTS.find(p => p.key === main).name, `${pct(d[`${main}_annual`], 1, true)}/年（情景部分 ${pct(d.scenario_annual, 1, true)}）`));
  }

  function renderMinis() {
    const win = curWin();
    for (const m of minis) {
      const A = win.assets[m.a];
      m.el.classList.toggle('on', m.a === st.asset);
      if (!A) { clearChart(m.box, ui.notice('这个窗口没有该资产。', 'info')); m.head.replaceChildren(h('div', { class: 'backtest-mini-row' }, h('b', {}, SHORT[m.a] || nameOf(m.a)))); continue; }
      m.head.replaceChildren(
        h('div', { class: 'backtest-mini-row' }, h('b', {}, `${SHORT[m.a] || nameOf(m.a)}${TICKER[m.a] ? ` ${TICKER[m.a]}` : ''}`),
          h('span', { title: '实际落在模拟 90% 区间内的月份占比' }, `90% 覆盖 ${pct(A.coverage_cond, 0)}`)),
        h('div', { class: 'backtest-mini-row muted', title: '实际年化 / 已知宏观时的模拟中位数年化' },
          `实际 ${pct(A.cagr_actual)} · 模拟中位数 ${pct(A.cagr_pred_cond_p50)}`));
      const fc = A.fan_cond || {};
      pathChart(charts, m.box, {
        x: A.dates || win.dates, log: st.log, compact: true, shocks: shocksOf(win),
        band: Array.isArray(fc.p5) && Array.isArray(fc.p95) ? { label: '已知宏观', color: COL.cond, ...fc } : null,
        lines: [
          { name: '实际', data: A.actual, color: COL.actual, width: 1.8, z: 4 },
          { name: '已知宏观·中位数', data: fc.p50, color: COL.cond },
        ].filter(l => Array.isArray(l.data)),
      });
    }
  }

  function renderTable() {
    const win = curWin();
    const rows = assets.filter(a => win.assets[a]).map(a => {
      const A = win.assets[a], d = A.decomposition || {};
      const main = ['engine', 'transmission'].reduce((b, k) => (Math.abs(d[`${k}_annual`] ?? 0) > Math.abs(d[`${b}_annual`] ?? 0) ? k : b), 'engine');
      return { id: a, ...A, err_cond: isNum(A.cagr_pred_cond_p50) && isNum(A.cagr_actual) ? A.cagr_pred_cond_p50 - A.cagr_actual : null,
        main, main_val: d[`${main}_annual`] };
    });
    const big = (v) => (isNum(v) && Math.abs(v) > 0.03 ? 'is-bad' : '');
    const tbl = ui.table([
      { key: 'id', label: '资产', fmt: (v) => h('span', {}, labelOf(v), KEY.has(v) ? h('span', { class: 'chip accent', style: { marginLeft: '6px' } }, '重点') : null) },
      { key: 'cagr_actual', label: '实际走势·年化', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
      { key: 'cagr_pred_cond_p50', label: '已知宏观·中位数', fmt: (v) => pct(v) },
      { key: 'err_cond', label: '误差', fmt: (v) => pct(v, 1, true), cls: big },
      { key: 'cagr_pred_uncond_p50', label: '不知道宏观·中位数', fmt: (v) => pct(v) },
      { key: 'cagr_oracle', label: 'oracle', fmt: (v) => pct(v) },
      { key: 'coverage_cond', label: '90% 覆盖（已知）', fmt: (v) => pct(v, 0) },
      { key: 'coverage_uncond', label: '90% 覆盖（不知道）', fmt: (v) => pct(v, 0) },
      { key: 'pit_terminal_cond', label: '终值 PIT', fmt: (v) => num(v), cls: (v) => (isNum(v) && (v < 0.05 || v > 0.95) ? 'is-bad' : '') },
      { key: 'main', label: '主要误差来源', fmt: (v, r) => `${PARTS.find(p => p.key === v).name} ${pct(r.main_val, 1, true)}/年` },
    ], rows, { onRowClick: (r) => selectAsset(r.id, true), stack: true });
    tbl.querySelectorAll('tbody tr').forEach((tr, i) => tr.classList.toggle('backtest-sel', rows[i].id === st.asset));
    tableBox.replaceChildren(tbl);
  }

  function renderDecomp() {
    const win = curWin();
    const list = assets.filter(a => win.assets[a]?.decomposition);
    decBox.style.height = `${Math.max(300, list.length * 30 + 70)}px`;
    decompChart(charts, decBox, ui, win, list);
    decText.textContent = readable(win.summary?.interpretation) || '这个窗口没有文字解读。';
  }

  function renderVA() {
    const va = curWin().summary?.valuation_adjusted;
    if (!va) { vaBox.replaceChildren(ui.notice('这个窗口没有估值调整 CMA 的对比结果。', 'info')); return; }
    const row = (label, a, b, f, lowerBetter = true) => ({ label, hist: a, va: b, f,
      better: isNum(a) && isNum(b) ? (lowerBetter ? b < a - 0.001 : Math.abs(b - 0.9) < Math.abs(a - 0.9) - 0.001) : null });
    const rows = [
      row('终值误差（已知宏观，年化）', va.terminal_err_cond_historical, va.terminal_err_cond_va, (v) => pct(v)),
      row('年化误差 MAE（已知宏观）', va.cagr_mae_cond_historical, va.cagr_mae_cond_va, (v) => pct(v)),
      row('90% 覆盖率（已知宏观）', va.coverage_cond_historical, va.coverage_cond_va, (v) => pct(v, 0), false),
      row('终值误差（不知道宏观，年化）', va.terminal_err_uncond_historical, va.terminal_err_uncond_va, (v) => pct(v)),
      row('年化误差 MAE（不知道宏观）', va.cagr_mae_uncond_historical, va.cagr_mae_uncond_va, (v) => pct(v)),
      row('90% 覆盖率（不知道宏观）', va.coverage_uncond_historical, va.coverage_uncond_va, (v) => pct(v, 0), false),
    ];
    vaBox.replaceChildren(ui.table([
      { key: 'label', label: '指标', wrap: true },
      { key: 'hist', label: '历史漂移', fmt: (v, r) => r.f(v) },
      { key: 'va', label: '估值调整', fmt: (v, r) => r.f(v), cls: (v, r) => ui.goodCls(r.better) },
    ], rows), h('p', { class: 'muted', style: { fontSize: '12px', margin: '8px 0 0' } },
      `范围：${(va.assets || []).map(a => SHORT[a] || nameOf(a)).join('、') || '—'}。绿色 = 估值调整更好，红色 = 更差。`));
  }

  function renderCorr() {
    const corr = curWin().corr || {};
    const A = corr.assets || [];
    if (!A.length || !Array.isArray(corr.realized)) {
      clearChart(corrBox, ui.notice('这个窗口没有相关矩阵。', 'info'));
      corrNote.textContent = '';
      pairBox.replaceChildren();
      return;
    }
    corrChart(charts, corrBox, ui, corr);
    const i = A.indexOf('us_equity'), j = A.indexOf('treasury_10y');
    const re = corr.realized?.[i]?.[j], po = corr.posterior?.[i]?.[j], pr = corr.prior?.[i]?.[j];
    let verdict = '';
    if (isNum(re) && isNum(po)) {
      verdict = Math.abs(po - re) <= 0.15 ? '后验与实际接近：熵池化把权重放到了股债联动相似的历史月份上。'
        : `${Math.sign(po) !== Math.sign(re) ? '后验连正负号都没对上' : '后验与实际差距较大'}：股债相关的正负号是各个时代最鲜明的特征之一，引擎只能从训练期月份里抽样，时代切换时（如 1990 年代末股债相关转负、2020 年后重新转正）后验会跟不上。`;
    }
    corrNote.textContent = i >= 0 && j >= 0 ? `股债相关（美股大盘 × 10年期美债）：实际 ${num(re)}，后验 ${num(po)}，先验 ${num(pr)}。${verdict}` : '';
    const pairs = [];
    for (let r = 0; r < A.length; r++) {
      for (let c = r + 1; c < A.length; c++) {
        const re = corr.realized?.[r]?.[c], po = corr.posterior?.[r]?.[c];
        if (isNum(re) && isNum(po)) pairs.push({ pair: `${SHORT[A[r]] || nameOf(A[r])} × ${SHORT[A[c]] || nameOf(A[c])}`, re, po, pr: corr.prior?.[r]?.[c], gap: po - re });
      }
    }
    pairs.sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap));
    pairBox.replaceChildren(ui.table([
      { key: 'pair', label: '资产对' },
      { key: 're', label: '实际', fmt: (v) => num(v) },
      { key: 'po', label: '后验', fmt: (v) => num(v) },
      { key: 'pr', label: '先验', fmt: (v) => num(v) },
      { key: 'gap', label: '后验 − 实际', fmt: (v) => num(v, 2, true), cls: (v) => (Math.abs(v) > 0.3 ? 'is-bad' : '') },
    ], pairs.slice(0, 6), { stack: true }));
  }

  function renderMacro() {
    const win = curWin();
    for (const m of macro) {
      const M = win.macro?.[m.key];
      if (!M || !Array.isArray(M.actual)) { clearChart(m.box, ui.notice('这个窗口没有该宏观变量的模拟。', 'info')); continue; }
      const fc = M.cond || {};
      pathChart(charts, m.box, {
        x: win.dates, areas: phaseAreas(win, false), shocks: shocksOf(win), fmt: (v) => (isNum(v) ? `${+v.toFixed(1)}%` : '—'), tipFmt: lvl,
        band: Array.isArray(fc.p5) && Array.isArray(fc.p95) ? { label: '已知宏观', color: COL.cond, ...fc } : null,
        lines: [
          { name: '实际', data: M.actual, color: COL.actual, width: 2.2, z: 4 },
          { name: '已知宏观·中位数', data: fc.p50, color: COL.cond },
          { name: '不知道宏观·中位数', data: M.uncond_p50, color: COL.uncond, dash: 'dashed' },
        ].filter(l => Array.isArray(l.data)),
      });
    }
  }

  function viewFmt(v, val) {
    if (!isNum(val)) return '—';
    if (String(v).startsWith('s_')) return `${val.toFixed(2)}%`;
    const unit = ui.META.factors[String(v).replace(/^f_/, '')]?.unit;
    const y = val * 12;
    if (unit === 'Δlog') return `${fixed(y * 100, 1, true)}%/年`;
    if (unit === 'Δpp') return `${fixed(y, 2, true)} pp/年`;
    return val.toFixed(4);
  }

  function renderPhases() {
    const win = curWin();
    const eps = win.ep || [];
    if (!eps.length) { phaseBox.replaceChildren(ui.notice('这个窗口没有阶段权重报告。', 'info')); viewsBox.replaceChildren(); return; }
    const d = win.dates || [];
    let s = 0;
    const ranges = (win.phases || []).map(m => { const r = [d[s + 1], d[s + m]]; s += m; return r; });
    const chips = (list) => h('div', { class: 'backtest-chips' }, list);
    const hasEpisodes = eps.some(e => isNum(e.episodes));
    const shrinkChip = (v, r) => {
      if (!(isNum(v) && v < 0.999)) return ui.chip('λ 1 · 原样使用', 'good backtest-kv-chip');
      const c = ui.chip(`λ ${v.toFixed(2)} · ${SHRINK[r.shrink_reason] || r.shrink_reason || '收缩'}`, 'warn backtest-kv-chip');
      c.title = `${SHRINK_WHY[r.shrink_reason] || '视图向训练期均值收缩'}${isNum(r.ens_requested) ? `；原样执行约 ${r.ens_requested.toFixed(0)} 个有效月份` : ''}`;
      return c;
    };
    const tbl = ui.table([
      { key: 'phase', label: '阶段', fmt: (v) => `第 ${v} 阶段` },
      { value: (r) => ranges[r.phase - 1], label: '月份', fmt: (v) => (v && v[0] ? `${v[0]} 至 ${v[1]}` : '—') },
      { key: 'ens', label: '有效月份 ENS', fmt: (v, r) => `${num(v, 0)}（${pct(r.ens_ratio, 1)}）` },
      ...(hasEpisodes ? [{ key: 'episodes', label: '有效历史片段', fmt: (v) => (isNum(v) ? `${v.toFixed(1)} 段` : '—'),
        cls: (v) => (isNum(v) && v < 6 ? 'is-bad' : '') }] : []),
      { key: 'shrink', label: '视图收缩 λ', fmt: shrinkChip },
      { key: 'era_weights', label: '时代权重', wrap: true, cls: 'backtest-wide', fmt: (v) => chips(Object.entries(v || {}).filter(([, w]) => isNum(w) && w >= 0.01)
        .sort((a, b) => b[1] - a[1]).map(([e, w]) => ui.chip(`${ui.eraName(e)} ${pct(w, 0)}`))) },
      { key: 'top_months', label: '权重最高的历史月份', wrap: true, cls: 'backtest-wide', fmt: (v) => chips((v || []).map(m => ui.chip(`${m.date} ×${num(m.weight_ratio, 1)}`, 'accent'))) },
      { key: 'max_abs_drawn_gap_sd', label: '最大视图偏离', fmt: (v, r) => h('span', {}, isNum(v) ? `${v.toFixed(2)}σ` : '—',
        r.clipped?.length ? ui.chip(`截断 ${r.clipped.map(x => ui.varName(x)).join('、')}`, 'bad') : null), cls: (v) => (isNum(v) && v > 0.5 ? 'is-bad' : '') },
    ], eps, { onRowClick: (r) => { st.phase = Math.max(0, eps.indexOf(r)); renderPhases(); }, stack: true });
    st.phase = Math.min(st.phase, eps.length - 1);
    tbl.querySelectorAll('tbody tr').forEach((tr, i) => tr.classList.toggle('backtest-sel', i === st.phase));
    phaseBox.replaceChildren(tbl);
    const ep = eps[st.phase];
    const rg = ranges[st.phase] || [];
    titled(viewsCard).textContent = `第 ${ep.phase} 阶段（${rg[0] || '—'} 至 ${rg[1] || '—'}）的宏观视图：真实 vs 目标 vs 抽中`;
    const clipped = new Set(ep.clipped || []);
    const views = ep.views || [];
    const hasSim = views.some(v => isNum(v.simulated_mean));
    const gapCell = (v) => (isNum(v) ? `${fixed(v, 2, true)}σ` : '—');
    const gapCls = (v) => (isNum(v) && Math.abs(v) > 0.5 ? 'is-bad' : '');
    const vg = win.view_gap_sd || {};
    const nViews = views.length || (win.ep?.[0]?.views || []).length;
    viewsBox.replaceChildren(
      h('div', { class: 'backtest-chips', style: { marginBottom: '10px' } },
        eps.map((e, i) => h('button', { class: `backtest-pill${i === st.phase ? ' on' : ''}`, onclick: () => { st.phase = i; renderPhases(); } }, `阶段 ${e.phase}`))),
      ui.table([
        { key: 'var', label: '宏观变量', fmt: (v) => h('span', {}, ui.varName(v), clipped.has(v) ? ui.chip('截断到 2%–98% 分位', 'bad') : null) },
        { key: 'realized', label: '真实均值', fmt: (v, r) => viewFmt(r.var, v) },
        { key: 'target_used', label: '视图目标（收缩后）', fmt: (v, r) => viewFmt(r.var, v) },
        { key: 'posterior_mean', label: '熵池化期望', fmt: (v, r) => viewFmt(r.var, v) },
        { key: 'drawn_mean', label: '实际抽中', fmt: (v, r) => viewFmt(r.var, v) },
        { key: 'drawn_gap_sd', label: '抽中 − 真实', fmt: gapCell, cls: gapCls },
        ...(hasSim ? [
          { key: 'simulated_mean', label: '模拟路径（再锚定后）', fmt: (v, r) => viewFmt(r.var, v) },
          { key: 'sim_gap_sd', label: '模拟 − 真实', fmt: gapCell, cls: gapCls },
        ] : []),
        { key: 'prior_mean', label: '训练期均值（先验）', fmt: (v, r) => viewFmt(r.var, v) },
      ], views, { stack: true }),
      h('p', { class: 'muted', style: { fontSize: '12px', margin: '8px 0 0' } },
        `ENS ${num(ep.ens, 0)}${isNum(ep.episodes) ? `（约 ${ep.episodes.toFixed(1)} 段不同的 12 个月历史）` : ''}，KL 散度 ${num(ep.kl, 2)}，${ep.converged ? '熵池化已收敛' : '熵池化未完全收敛'}；`,
        isNum(ep.shrink) && ep.shrink < 0.999 ? `${SHRINK_WHY[ep.shrink_reason] || `${nViews} 个视图在训练期无法满足`}，所以 ${nViews} 个视图全部按 λ = ${ep.shrink.toFixed(2)} 向训练期均值收缩（目标 = 先验 + λ ×（真实 − 先验））。` : '视图按真实均值原样使用。'),
      isNum(vg.cond) || isNum(vg.cond_simulated) ? h('p', { class: 'muted', style: { fontSize: '12px', margin: '4px 0 0' } },
        `整个窗口（全部阶段、全部视图）的平均偏离：抽中月份 ${isNum(vg.cond) ? `${vg.cond.toFixed(2)}σ` : '—'}，`
        + `模拟路径 ${isNum(vg.cond_simulated) ? `${vg.cond_simulated.toFixed(2)}σ` : '—'}。`
        + (isNum(vg.cond) && isNum(vg.cond_simulated) && vg.cond_simulated > vg.cond + 0.02
          ? '模拟路径比抽中月份离真实更远，多出来的部分来自再锚定（利率按今天的水平缩放、向中性利率回归、油价回归），不是抽样。'
          : '再锚定没有让宏观路径明显偏离抽中的月份。')) : null);
  }

  function renderTransmission() {
    const win = curWin();
    const ser = T.series?.[st.asset];
    titled(transCard).textContent = `${labelOf(st.asset)}：实际 vs 因子模型拟合（${T.start || '—'} 至 ${T.end || '—'}）`;
    const r2 = T.r2?.[st.asset], r2o = T.r2_oos?.[st.asset];
    transChips.replaceChildren(ui.chip(`样本内 R² ${pct(r2, 1)}`, 'accent'), ui.chip(`样本外 R² ${pct(r2o, 1)}`, isNum(r2o) && r2o < 0 ? 'bad' : ''));
    if (!ser || !Array.isArray(T.dates)) { clearChart(transBox, ui.notice('没有该资产的传导拟合。', 'info')); return; }
    const t0 = win.test?.[0], t1 = win.test?.[1];
    pathChart(charts, transBox, {
      x: T.dates, log: true, zoom: true, shocks: bt.shocks || [],
      areas: t0 && t1 ? [{ from: t0, to: t1, label: `${win.name || win.id} 测试期`, color: 'rgba(232,89,12,0.07)' }] : [],
      lines: [
        { name: '实际', data: ser.actual, color: COL.actual, width: 2, z: 4 },
        { name: '因子模型·样本内', data: ser.fitted, color: COL.cond, width: 1.6 },
        { name: '因子模型·样本外（滚动重估）', data: ser.fitted_oos, color: COL.oracle, width: 1.6, dash: 'dashed' },
      ].filter(l => Array.isArray(l.data)),
    });
    if (st.asset === 'cash') subtitled(transCard).textContent = '现金只有利息收入，没有需要解释的价格波动，三条线重合。';
    else if (structural.has(st.asset)) subtitled(transCard).textContent = `${nameOf(st.asset)}在模拟里由收益率 / 油价路径按结构规则生成；这里的拟合线只是线性因子回归，仅作参考（原油价格收益对油价对数变化是凸的，线性回归的累计线会偏低）。`;
    else subtitled(transCard).textContent = `把真实的月度因子变化喂给因子模型（对数坐标）：蓝线为全样本估计（样本内），橙线从 ${T.oos_start || '—'} 起每 ${T.refit_every || 60} 个月只用之前的数据重估、预测下一段（样本外）；两线贴近黑线说明宏观因子抓住了长期走势，样本外越偏说明传导随时代改变，阴影为当前回测窗口的测试期。`;
  }

  function renderStatic() {
    // R² by asset
    const r2Assets = assets.filter(a => !structural.has(a) && (isNum(T.r2?.[a]) || isNum(T.r2_oos?.[a])));
    if (r2Assets.length) {
      const r2Chart = charts.bars(r2Box, {
        x: r2Assets.map(a => SHORT[a] || nameOf(a)), horizontal: true, fmt: 'pct',
        series: [
          { name: '样本内（全样本）', data: r2Assets.map(a => T.r2?.[a] ?? null), color: COL.cond },
          { name: '同期样本内', data: r2Assets.map(a => T.r2_in_sample_oos_months?.[a] ?? null), color: '#91a7ff' },
          { name: '样本外', data: r2Assets.map(a => T.r2_oos?.[a] ?? null), color: COL.oracle },
        ],
      });
      r2Chart.setOption({ xAxis: { splitNumber: 4, axisLabel: { formatter: (v) => pct(v, 0) } } });
    } else clearChart(r2Box, ui.notice('没有 R² 结果。', 'info'));
    r2Table.replaceChildren(ui.table([
      { key: 'a', label: '资产', fmt: (v) => labelOf(v) },
      { key: 'r2', label: '样本内', fmt: (v) => pct(v, 1) },
      { key: 'same', label: '同期样本内', fmt: (v) => pct(v, 1) },
      { key: 'oos', label: '样本外', fmt: (v) => pct(v, 1), cls: (v) => (isNum(v) && v < 0 ? 'is-bad' : '') },
    ], [...assets.map(a => ({ a, r2: T.r2?.[a], same: T.r2_in_sample_oos_months?.[a], oos: T.r2_oos?.[a] })),
      { a: '风险资产平均', r2: T.r2_risk_avg, same: T.r2_in_sample_oos_months_risk_avg, oos: T.r2_oos_risk_avg }]));

    // cross-window table and charts
    const rows = (S.windows || []).map(r => ({ ...r, win: wins.find(w => w.id === r.id) }));
    if (rows.length) {
      const agg = { id: null, name: '平均', test: null, ...Object.fromEntries(Object.entries(S).filter(([, v]) => isNum(v))) };
      const tbl = ui.table([
        { key: 'name', label: '窗口' },
        { key: 'test', label: '测试期', fmt: (v) => (v ? span(v) : '—') },
        { key: 'cagr_mae_cond_risk', label: '年化误差 已知 / 不知道宏观', fmt: (v, r) => `${pct(v)} / ${pct(r.cagr_mae_uncond_risk)}`,
          cls: (v, r) => (isNum(v) && isNum(r.cagr_mae_uncond_risk) && v > r.cagr_mae_uncond_risk ? 'is-bad' : '') },
        { key: 'coverage_cond_risk', label: '90% 覆盖 已知 / 不知道', fmt: (v, r) => `${pct(v, 0)} / ${pct(r.coverage_uncond_risk, 0)}` },
        { key: 'scenario_abs_annual', label: '情景部分', fmt: (v) => pct(v) },
        { key: 'engine_abs_annual', label: '引擎部分', fmt: (v) => pct(v) },
        { key: 'transmission_abs_annual', label: '传导部分', fmt: (v) => pct(v) },
        { key: 'terminal_err_cond_historical', label: '终值误差 历史漂移 / 估值调整', fmt: (v, r) => `${pct(v)} / ${pct(r.terminal_err_cond_va)}` },
        { key: 'view_gap_sd_drawn', label: '视图偏离 抽中 / 模拟', fmt: (v, r) => `${isNum(v) ? `${v.toFixed(2)}σ` : '—'} / ${isNum(r.view_gap_sd_simulated) ? `${r.view_gap_sd_simulated.toFixed(2)}σ` : '—'}` },
      ], [...rows, agg], { onRowClick: (r) => { if (r.id) { st.win = r.id; st.phase = 0; st.pending = null; renderWindow(); sync(); } }, stack: true });
      const trs = tbl.querySelectorAll('tbody tr');
      if (trs.length) trs[trs.length - 1].style.fontWeight = '600';
      crossBox.replaceChildren(tbl);
      const names = rows.map(r => r.name || r.id);
      const maeChart = charts.bars(crossMae, {
        x: names, fmt: 'pct',
        series: [
          { name: '已知宏观', data: rows.map(r => r.cagr_mae_cond_risk ?? null), color: COL.cond },
          { name: '不知道宏观', data: rows.map(r => r.cagr_mae_uncond_risk ?? null), color: COL.uncond },
          { name: 'oracle', data: rows.map(r => r.win?.summary?.cagr_mae_oracle_risk ?? null), color: COL.oracle },
        ],
      });
      wrapTicks(maeChart, crossMae, names.length, names);
      const partsChart = charts.bars(crossParts, {
        x: names, fmt: 'pct',
        series: PARTS.map(p => ({ name: p.name, data: rows.map(r => r[`${p.key}_abs_annual`] ?? null), color: p.color })),
      });
      wrapTicks(partsChart, crossParts, names.length, names);
      // Say which part dominates and whether the transmission part grows over the windows, from the data.
      const tr = rows.filter(r => isNum(r.transmission_abs_annual));
      const big = PARTS.slice(1).map(p => ({ p, n: rows.filter(r => isNum(r[`${p.key}_abs_annual`])
        && PARTS.slice(1).every(q => !isNum(r[`${q.key}_abs_annual`]) || r[`${p.key}_abs_annual`] >= r[`${q.key}_abs_annual`])).length }))
        .sort((a, b) => b.n - a.n)[0];
      const t0 = tr[0], t1 = tr[tr.length - 1];
      const trend = tr.length >= 2 && t1.transmission_abs_annual > t0.transmission_abs_annual + 0.005
        ? `传导部分从「${t0.name || t0.id}」的 ${pct(t0.transmission_abs_annual)} 升到「${t1.name || t1.id}」的 ${pct(t1.transmission_abs_annual)}，近年的资产定价越来越不由模型里的宏观因子决定。`
        : tr.length >= 2 && t1.transmission_abs_annual < t0.transmission_abs_annual - 0.005 ? '传导部分在后面的窗口变小。' : '';
      subtitled(partsCard).textContent = `情景部分衡量知道宏观让中位数移动了多少，引擎部分和传导部分才是剩下的误差；${big?.n
        ? `${rows.length} 个窗口中有 ${big.n} 个以${big.p.name}为主。` : ''}${trend}`;
    } else {
      crossBox.replaceChildren(ui.notice('没有跨窗口汇总。', 'info'));
    }
  }

  renderStatic();
  renderWindow();
  sync();
  if (st.pending) runCustom({ req: st.pending, fromLink: true, select: true });     // a shared #backtest?custom=… link
}

// Quarterly / annual R² of the transmission (SIM_DESIGN §10.1: the headline tile becomes quarterly and annual): the
// cumulative series of transmission.series (actual, fitted = in-sample, fitted_oos = expanding-window refits) cut into
// non-overlapping blocks of `step` months that end on calendar quarter-ends (step 3) or year-ends (step 12). In-sample:
// 1 - SSE / SST around the sample mean; out of sample: the blocks from the first refit on, against the mean of the
// earlier blocks (what was known). Averaged over the risk assets, as the monthly r2_risk_avg.
function blockR2(T, risk, step) {
  const dates = T?.dates || [];
  const endMonths = step === 12 ? ['12'] : ['03', '06', '09', '12'];
  const start = dates.findIndex(d => endMonths.includes(String(d).slice(5, 7)));
  const oosStart = T?.oos_start || null;
  const ins = [], oos = [];
  let nBlocks = 0;
  for (const a of risk) {
    const s = T?.series?.[a];
    if (!s || !Array.isArray(s.actual)) continue;
    const blocks = [];
    for (let i = start; i >= 0 && i + step < dates.length; i += step) {
      const a0 = s.actual[i], a1 = s.actual[i + step];
      if (!(a0 > 0 && a1 > 0)) continue;
      const f0 = s.fitted?.[i], f1 = s.fitted?.[i + step], o0 = s.fitted_oos?.[i], o1 = s.fitted_oos?.[i + step];
      blocks.push({ end: dates[i + step], y: Math.log(a1 / a0), f: f0 > 0 && f1 > 0 ? Math.log(f1 / f0) : null,
        o: o0 > 0 && o1 > 0 ? Math.log(o1 / o0) : null });
    }
    const fit = blocks.filter(b => isNum(b.f));
    if (fit.length > 3) {
      const m = fit.reduce((t, b) => t + b.y, 0) / fit.length;
      const sst = fit.reduce((t, b) => t + (b.y - m) ** 2, 0), sse = fit.reduce((t, b) => t + (b.y - b.f) ** 2, 0);
      if (sst > 0) ins.push(1 - sse / sst);
      nBlocks = Math.max(nBlocks, fit.length);
    }
    let sse = 0, sst = 0, k = 0;
    blocks.forEach((b, j) => {
      if (!isNum(b.o) || (oosStart && b.end < oosStart) || j < 2) return;
      const prev = blocks.slice(0, j).map(x => x.y);
      const bench = prev.reduce((t, x) => t + x, 0) / prev.length;
      sse += (b.y - b.o) ** 2; sst += (b.y - bench) ** 2; k += 1;
    });
    if (k > 3 && sst > 0) oos.push(1 - sse / sst);
  }
  const avg = (xs) => (xs.length ? xs.reduce((t, x) => t + x, 0) / xs.length : null);
  return { r2: avg(ins), oos: avg(oos), n: nBlocks };
}

// The long-run anchors each window simulated with: fitted to its train end by the pre-registered rule (SIM_DESIGN
// §5.5), next to today's fixed values that every window used before.
function anchorsCard(h, card, ui, bt, wins) {
  const rows = wins.filter(w => w.anchors).map(w => ({ name: w.name || w.id, ...w.anchors }));
  if (!rows.length) return h('div');
  const rules = bt.anchor_rules || {};
  const f = (x) => (isNum(x) ? x.toFixed(2) : '—');
  const d = rows[0].default || {};
  return card({ title: '各窗口的长期锚：按训练期末可知的规则估计',
    sub: `规则预先登记、不调参：中性实际利率 = ${rules.r_star || '—'}；期限利差 = ${rules.term || '—'}；信用利差 = ${rules.spread || '—'}；失业率 = ${rules.unrate || '—'}。`
      + `以前每个窗口都用今天的判断值（${f(d.r_star)} / ${f(d.term_premium)} / ${f(d.spread_level)} / ${f(d.unrate_level)}），那是 2008 年以后才知道的水平；前瞻情景仍用判断值。` },
  ui.table([
    { key: 'name', label: '窗口' },
    { key: 'end', label: '训练期末' },
    { key: 'r_star', label: '中性实际利率', fmt: f },
    { key: 'term', label: '期限利差', fmt: f },
    { key: 'spread', label: '信用利差', fmt: f },
    { key: 'unrate', label: '失业率', fmt: f },
  ], rows, { stack: true }));
}
