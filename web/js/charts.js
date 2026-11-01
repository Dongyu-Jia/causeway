// ECharts helpers shared by all tabs. Every helper takes a container element
// (from ui.chartBox()) and returns the chart instance.
//
//   mount(el)            init (or re-init) an ECharts instance on el, resized with el
//   unmount(el)          dispose it and disconnect every observer the helpers attached
//   disposeWithin(root)  unmount every chart inside root (and any that left the page)
//   line / fan / candles / heatmap / bars / dag / scatter   ready-made charts
//
// Options are only ever added: existing calls keep working.
/* global echarts */

const INK = '#1d2433', INK2 = '#4a5468', INK3 = '#7a8396', LINE = '#e3e6ec';
// 13 colours, one per asset class in config.ASSET_IDS; palette(i) cycles through them, so a 14th
// series repeats the first colour.
const PALETTE = ['#3b5bdb', '#e8590c', '#2f9e44', '#c2255c', '#7048e8', '#1098ad', '#f08c00',
  '#5c940d', '#d6336c', '#495057', '#0b7285', '#a61e4d', '#364fc7'];
const ASSET_COLORS = {
  us_equity: '#3b5bdb', nasdaq: '#7048e8', energy: '#e8590c', industrials: '#495057', staples: '#2f9e44',
  healthcare: '#1098ad', utilities: '#0b7285', financials: '#364fc7', real_estate: '#c2255c',
  gold: '#f08c00', crude_oil: '#5f3dc4', treasury_10y: '#2b8a3e', cash: '#868e96',
};
export const assetColor = (id, i = 0) => ASSET_COLORS[id] || PALETTE[i % PALETTE.length];
export const palette = (i) => PALETTE[i % PALETTE.length];

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
// ECharts tooltips are HTML: escape anything that may come from data, a user or an LLM.
export const esc = (t) => String(t ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
// Fixed decimals without "-0.0".
const fixed = (x, d) => { const s = x.toFixed(d); return Number(s) === 0 ? (0).toFixed(d) : s; };

// ---------------------------------------------------------------- K-line colours

let scheme = (() => { try { return localStorage.getItem('kline-color') || 'cn'; } catch (_) { return 'cn'; } })();
if (scheme !== 'us') scheme = 'cn';
export function setKlineScheme(s) { scheme = s === 'us' ? 'us' : 'cn'; try { localStorage.setItem('kline-color', scheme); } catch (_) { /* ignore */ } }
export const getKlineScheme = () => scheme;       // 'cn' = 红涨绿跌 (default), 'us' = 绿涨红跌
export function klineColors() {
  const red = '#d9363e', green = '#1a9c5b';
  return scheme === 'cn' ? { up: red, down: green } : { up: green, down: red };
}
// hex colour mixed toward white by t (0 = unchanged, 1 = white)
function lighten(hex, t) {
  const n = parseInt(String(hex).slice(1), 16);
  const mix = (c) => Math.round(c + (255 - c) * t);
  return `rgb(${mix((n >> 16) & 255)}, ${mix((n >> 8) & 255)}, ${mix(n & 255)})`;
}
// Diverging colours for signed returns (heatmap visualMap, low -> high): losses in the K-line
// down colour, gains in the up colour, so return heatmaps follow the CN / US toggle like .pos / .neg.
export function klineDiverging() {
  const { up, down } = klineColors();
  return [down, lighten(down, 0.62), '#f8f9fa', lighten(up, 0.62), up];
}

export function fmtValue(v, fmt) {
  if (v === null || v === undefined || v === '' || v === '-' || !Number.isFinite(Number(v))) return '—';
  if (typeof fmt === 'function') return fmt(v);
  const x = Number(v);
  if (fmt === 'pct') return `${fixed(x * 100, 1)}%`;
  if (fmt === 'pct2') return `${fixed(x * 100, 2)}%`;
  if (fmt === 'pp') return `${fixed(x, 2)}%`;
  if (fmt === 'x') return `${fixed(x, 2)}x`;
  if (fmt === 'int') return Math.round(x).toLocaleString('zh-CN');
  return fixed(x, Math.abs(x) >= 100 ? 0 : 2);
}

// ---------------------------------------------------------------- lifecycle

// Elements holding a chart, so a page change can dispose every instance (ECharts keeps them all
// in a global registry until dispose()) and disconnect their observers.
const mounted = new Set();
const OBSERVERS = ['_ro', '_dagRO', '_chainRO', '_fitRO', '_abRo'];

// Without the ECharts library (web/vendor/echarts/echarts.min.js missing or blocked) the chart box
// gets a notice and mount() returns an inert stand-in: every method does nothing, properties a tab
// assigns are kept, so the page's tables and text still render.
const NO_ECHARTS = '图表库 ECharts 没有加载，这里的图表无法显示（表格和文字不受影响）。'
  + '请确认 web/vendor/echarts/echarts.min.js 存在，然后刷新页面。';
let warnedNoEcharts = false;
function inertChart(el) {
  if (!warnedNoEcharts) { warnedNoEcharts = true; console.warn(`[charts] ${NO_ECHARTS}`); }
  if (!el.querySelector(':scope > .chart-missing')) {
    const note = document.createElement('div');
    note.className = 'notice chart-missing';
    note.setAttribute('role', 'note');
    note.textContent = NO_ECHARTS;
    Object.assign(note.style, { boxSizing: 'border-box', height: '100%', minHeight: '64px', display: 'flex',
      alignItems: 'center', justifyContent: 'center', textAlign: 'center' });
    el.append(note);
  }
  const own = { isDisposed: () => false };
  return new Proxy(own, {
    get: (t, k) => (k in t ? t[k] : typeof k === 'symbol' || k === 'then' ? undefined : () => undefined),
    set: (t, k, v) => { t[k] = v; return true; },
  });
}

export function mount(el) {
  if (el._chart) { try { el._chart.dispose(); } catch (_) { /* already gone */ } }
  const lib = globalThis.echarts;
  if (!lib || typeof lib.init !== 'function') {
    el._chart = inertChart(el);
    mounted.add(el);
    return el._chart;
  }
  const chart = lib.init(el, null, { renderer: 'canvas' });
  el._chart = chart;
  mounted.add(el);
  if (!el._ro) {
    el._ro = new ResizeObserver(() => { const c = el._chart; if (c && !c.isDisposed()) c.resize(); });
    el._ro.observe(el);
  }
  return chart;
}

// Dispose the chart on el and disconnect the observers the helpers (and tabs) attach to it.
export function unmount(el) {
  if (!el) return;
  if (el._chart) { try { el._chart.dispose(); } catch (_) { /* already gone */ } el._chart = null; }
  for (const k of OBSERVERS) {
    if (el[k]) { try { el[k].disconnect(); } catch (_) { /* gone */ } el[k] = null; }
  }
  mounted.delete(el);
}

// Unmount every chart inside root, plus any whose element has left the document.
export function disposeWithin(root) {
  for (const el of [...mounted]) if (!el.isConnected || !root || root.contains(el)) unmount(el);
}

const baseGrid = (extra = {}) => ({ left: 56, right: 24, top: 36, bottom: 40, containLabel: false, ...extra });
const axisCommon = {
  axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false },
  axisLabel: { color: INK3, fontSize: 11 }, splitLine: { lineStyle: { color: '#eef0f4' } },
};
const legendCommon = { top: 0, textStyle: { color: INK, fontSize: 12 }, itemWidth: 14, itemHeight: 8, type: 'scroll' };

// areas: [{from, to, label?, color?}] on a category axis.
function markAreaOf(areas) {
  if (!areas || !areas.length) return undefined;
  return {
    silent: true,
    itemStyle: { color: 'rgba(59, 91, 219, 0.06)' },
    label: { color: INK3, fontSize: 10, position: 'insideTop' },
    data: areas.filter(a => a && a.from !== undefined && a.to !== undefined)
      .map(a => [{ name: a.label || '', xAxis: a.from, itemStyle: a.color ? { color: a.color } : undefined }, { xAxis: a.to }]),
  };
}

// ECharts rounds a mark line's position to `precision` decimals (default 2: a line at 0.055 lands on 0.06).
// Its label shows the item's name, else the value in the chart's format (0.1 on a % axis reads "10.0%");
// position (e.g. 'insideEndTop') keeps it inside the plot instead of past the grid edge.
const markLineOf = (data, extra = {}, fmt, position) => ({ silent: true, symbol: 'none', precision: 10,
  lineStyle: { color: INK3, type: 'dashed' },
  label: { color: INK3, fontSize: 10, ...(position ? { position } : {}),
    formatter: (p) => (p.name ? p.name : typeof p.value === 'number' ? fmtValue(p.value, fmt) : String(p.value ?? '')) },
  ...extra, data });

// ---------------------------------------------------------------- line

// series: [{name, data, color?, dashed?, width?, area?, yAxisIndex?, step?, symbol?}]
export function line(el, { x, series, yName = '', yFmt, log = false, legend = true, zoom = false,
  markAreas, yAxis2, yMin, yMax, tooltipFmt, markLines } = {}) {
  const chart = mount(el);
  const yAxes = [{ type: log ? 'log' : 'value', name: yName, nameTextStyle: { color: INK3, fontSize: 11 }, scale: !log,
    min: yMin, max: yMax, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, yFmt) } }];
  if (yAxis2) yAxes.push({ type: 'value', name: yAxis2.name || '', position: 'right', scale: true, ...axisCommon,
    splitLine: { show: false }, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, yAxis2.fmt) } });
  chart.setOption({
    color: PALETTE,
    legend: legend ? legendCommon : undefined,
    tooltip: { trigger: 'axis', confine: true, valueFormatter: (v) => fmtValue(v, tooltipFmt || yFmt) },
    grid: baseGrid({ right: yAxis2 ? 56 : 24, bottom: zoom ? 56 : 32 }),
    xAxis: { type: 'category', data: x, boundaryGap: false, ...axisCommon, splitLine: { show: false } },
    yAxis: yAxes,
    dataZoom: zoom ? [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }] : undefined,
    series: series.map((s, i) => ({
      name: s.name, type: 'line', data: s.data, showSymbol: false, symbol: s.symbol || 'none', step: s.step,
      yAxisIndex: s.yAxisIndex || 0, smooth: false,
      lineStyle: { width: s.width || 1.6, type: s.dashed ? 'dashed' : 'solid', color: s.color || palette(i) },
      itemStyle: { color: s.color || palette(i) },
      areaStyle: s.area ? { opacity: 0.12 } : undefined,
      markArea: i === 0 ? markAreaOf(markAreas) : undefined,
      markLine: i === 0 && markLines ? markLineOf(markLines, {}, yFmt,
        markLines.every(m => m && m.yAxis !== undefined) ? 'insideEndTop' : undefined) : undefined,
    })),
  });
  return chart;
}

// ---------------------------------------------------------------- fan

// One percentile band as polygons on a custom series (one polygon per run of finite values, so
// missing values leave gaps). Unlike stacked areas it also works on a log axis. The whole band is
// drawn from the first data item in view; the item values [i, lo, hi] keep the axis covering it.
function polygonBand(lo, hi, color, opacity) {
  return {
    type: 'custom', silent: true, clip: true, z: 1, tooltip: { show: false }, emphasis: { disabled: true },
    encode: { x: 0, y: [1, 2] },
    data: lo.map((v, i) => [i, isNum(v) && isNum(hi[i]) ? v : '-', isNum(v) && isNum(hi[i]) ? hi[i] : '-']),
    renderItem: (params, api) => {
      if (params.dataIndexInside !== 0) return null;
      const children = [];
      let top = [], bottom = [];
      const flush = () => {
        if (top.length > 1) children.push({ type: 'polygon', silent: true, shape: { points: top.concat(bottom.reverse()) }, style: { fill: color, opacity } });
        top = []; bottom = [];
      };
      lo.forEach((l, i) => {
        const h = hi[i];
        if (isNum(l) && isNum(h)) { top.push(api.coord([i, h])); bottom.push(api.coord([i, l])); } else flush();
      });
      flush();
      return { type: 'group', children };
    },
  };
}

// Percentile fan: bands = {p5, p25?, p50, p75?, p95}; optional actual path and extra series.
// Missing values (null, undefined, NaN) are gaps. log: true draws the bands as polygons on a log
// axis (non-positive values become gaps). fmt formats the tooltip, axisFmt the axis (default fmt).
export function fan(el, { x, bands, actual, actualName = '实际', name = '中位数', color = '#3b5bdb', fmt, axisFmt, extra = [],
  markAreas, legend = true, yName = '', zoom = false, log = false, logBase = 2 } = {}) {
  const chart = mount(el);
  const clean = (a) => (Array.isArray(a) ? a : []).map(v => (isNum(v) && (!log || v > 0) ? v : null));
  const b = bands || {};
  const p5 = clean(b.p5), p25 = clean(b.p25), p50 = clean(b.p50), p75 = clean(b.p75), p95 = clean(b.p95);
  const inner = p25.some(isNum) && p75.some(isNum);
  const series = [];
  if (log) {
    series.push(polygonBand(p5, p95, color, inner ? 0.12 : 0.16));
    if (inner) series.push(polygonBand(p25, p75, color, 0.16));
  } else {
    // Stacked areas: each band is stacked on the one below. stackStrategy 'all' stacks regardless of
    // sign (the default 'samesign' drops a positive band's base when p5 < 0 and draws the band from 0).
    const diff = (a, lo) => a.map((v, i) => (isNum(v) && isNum(lo[i]) ? v - lo[i] : null));
    const stack = `fan${Math.random().toString(36).slice(2, 8)}`;
    const band = (nm, data, opacity) => ({ name: nm, type: 'line', data, stack, stackStrategy: 'all', symbol: 'none', showSymbol: false,
      silent: true, lineStyle: { opacity: 0 }, emphasis: { disabled: true }, tooltip: { show: false },
      areaStyle: opacity ? { color, opacity } : undefined });
    series.push(band('_p5', p5, 0));
    if (inner) {
      series.push(band('_p5_25', diff(p25, p5), 0.12), band('_p25_75', diff(p75, p25), 0.26), band('_p75_95', diff(p95, p75), 0.12));
    } else {
      series.push(band('_p5_95', diff(p95, p5), 0.16));
    }
  }
  series.push({ name, type: 'line', data: p50, symbol: 'none', showSymbol: false, z: 3, lineStyle: { width: 2, color }, itemStyle: { color },
    markArea: markAreaOf(markAreas) });
  const act = actual ? clean(actual) : null;
  if (act) series.push({ name: actualName, type: 'line', data: act, symbol: 'none', showSymbol: false, z: 4, lineStyle: { width: 2, color: INK }, itemStyle: { color: INK } });
  const ext = (extra || []).filter(Boolean);
  ext.forEach((s, i) => series.push({ type: 'line', symbol: 'none', showSymbol: false, z: 3, ...s, data: clean(s.data),
    lineStyle: { width: 1.5, type: s.dashed ? 'dashed' : 'solid', color: s.color || palette(i + 3) },
    itemStyle: { color: s.color || palette(i + 3) } }));
  const f = (v) => esc(fmtValue(v, fmt));
  chart.setOption({
    legend: legend ? { ...legendCommon, data: [name, ...(act ? [actualName] : []), ...ext.map(s => s.name)] } : undefined,
    tooltip: {
      trigger: 'axis', confine: true,
      formatter: (params) => {
        const ps = Array.isArray(params) ? params : [params];
        const i = ps.find(p => Number.isInteger(p?.dataIndex))?.dataIndex;
        if (i === undefined) return '';
        const rows = [`<b>${esc(x?.[i])}</b>`, `95%: ${f(p95[i])}`];
        if (inner) rows.push(`75%: ${f(p75[i])}`);
        rows.push(`<b>${esc(name)}: ${f(p50[i])}</b>`);
        if (inner) rows.push(`25%: ${f(p25[i])}`);
        rows.push(`5%: ${f(p5[i])}`);
        if (act) rows.push(`<b>${esc(actualName)}: ${f(act[i])}</b>`);
        ext.forEach(s => rows.push(`${esc(s.name)}: ${f(Array.isArray(s.data) ? s.data[i] : null)}`));
        return rows.join('<br/>');
      },
    },
    grid: baseGrid({ bottom: zoom ? 56 : 32 }),
    xAxis: { type: 'category', data: x, boundaryGap: false, ...axisCommon, splitLine: { show: false } },
    yAxis: { type: log ? 'log' : 'value', logBase: log ? logBase : undefined, scale: !log, name: yName,
      nameTextStyle: { color: INK3, fontSize: 11 }, ...axisCommon,
      axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, axisFmt || fmt) } },
    dataZoom: zoom ? [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }] : undefined,
    series,
  });
  return chart;
}

// ---------------------------------------------------------------- candles

function movingAverage(closes, n) {
  const out = [];
  for (let i = 0; i < closes.length; i++) {
    if (i < n - 1) { out.push(null); continue; }
    let sum = 0, ok = true;
    for (let k = i - n + 1; k <= i; k++) { if (!isNum(closes[k])) { ok = false; break; } sum += closes[k]; }
    out.push(ok ? +(sum / n).toFixed(3) : null);
  }
  return out;
}

// data: [[label, open, close, low, high], ...]
// freq: 'monthly' | 'yearly' (default: yearly when every label is a 4-digit year). The default
// moving average follows it (12 months on monthly candles, none on yearly ones: a 12-period MA
// of yearly candles would be a 12-year average); pass ma: [n, ...] to choose.
// fmt formats the tooltip; axisFmt the price axis (default: whole numbers when the range spans 20+).
// markAreas: [{from, to, label?, color?}] shaded spans on the label axis.
export function candles(el, { data, name = 'K 线', ma, fmt, axisFmt, log = false, zoomStart = 0, freq, markAreas } = {}) {
  const chart = mount(el);
  const rows = (data || []).filter(d => Array.isArray(d));
  const up = klineColors().up, down = klineColors().down;
  const labels = rows.map(d => d[0]);
  const yearly = freq ? freq === 'yearly' : labels.length > 0 && labels.every(l => /^\d{4}$/.test(String(l)));
  const periods = Array.isArray(ma) ? ma : (yearly ? [] : [12]);
  const ohlc = rows.map(d => [d[1], d[2], d[3], d[4]]);
  const closes = rows.map(d => d[2]);
  const maSeries = periods.filter(n => Number.isInteger(n) && n > 1 && n < rows.length).map((n, i) => ({
    name: `${n}${yearly ? '年' : '月'}均线`, type: 'line', data: movingAverage(closes, n), symbol: 'none', smooth: true,
    lineStyle: { width: 1.2, color: ['#f08c00', '#7048e8', '#1098ad'][i % 3] }, itemStyle: { color: ['#f08c00', '#7048e8', '#1098ad'][i % 3] },
  }));
  const lows = rows.map(d => d[3]).filter(isNum), highs = rows.map(d => d[4]).filter(isNum);
  const span = lows.length && highs.length ? Math.max(...highs) - Math.min(...lows) : 0;
  const yFmt = axisFmt || (span >= 20 ? 'int' : fmt);
  chart.setOption({
    legend: { ...legendCommon, data: [name, ...maSeries.map(s => s.name)] },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'cross' }, confine: true,
      formatter: (params) => {
        const ps = Array.isArray(params) ? params : [params];
        const k = ps.find(p => p.seriesType === 'candlestick');
        if (!k) return '';
        const [o, c, l, hgh] = (Array.isArray(k.data) ? k.data : k.value || []).slice(-4);
        const chg = isNum(o) && o !== 0 && isNum(c) ? (c / o - 1) : null;
        const mas = ps.filter(p => p.seriesType === 'line' && isNum(p.value)).map(p => `${esc(p.seriesName)} ${esc(fmtValue(p.value, fmt))}`);
        return [`<b>${esc(k.name)}</b>`, `开 ${esc(fmtValue(o, fmt))}　收 ${esc(fmtValue(c, fmt))}`,
          `低 ${esc(fmtValue(l, fmt))}　高 ${esc(fmtValue(hgh, fmt))}`, `涨跌 ${chg === null ? '—' : `${chg > 0 ? '+' : ''}${fixed(chg * 100, 2)}%`}`,
          ...mas].join('<br/>');
      },
    },
    grid: baseGrid({ bottom: 56 }),
    xAxis: { type: 'category', data: labels, ...axisCommon, splitLine: { show: false } },
    yAxis: { type: log ? 'log' : 'value', scale: true, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, yFmt) } },
    dataZoom: [{ type: 'inside', start: zoomStart, end: 100 }, { type: 'slider', height: 16, bottom: 8, start: zoomStart, end: 100 }],
    series: [{ name, type: 'candlestick', data: ohlc, markArea: markAreaOf(markAreas),
      itemStyle: { color: up, color0: down, borderColor: up, borderColor0: down } }, ...maSeries],
  });
  return chart;
}

// ---------------------------------------------------------------- heatmap

// Approximate rendered width of a label (CJK ~fontSize, Latin and digits ~0.6 x fontSize).
const textPx = (s, size = 11) => [...String(s ?? '')].reduce((t, ch) => t + (/[\x00-\xff]/.test(ch) ? size * 0.6 : size), 0);

// values[r][c]; diverging palette centred at 0 by default. Missing values (null, undefined, NaN)
// are blank cells. Optional:
//   cellStyle(r, c, v) -> itemStyle for one cell (e.g. outline flagged cells), v is null when missing
//   tip(r, c, v)       -> tooltip HTML (escape your own text), v is null when missing
//   labelFmt(v, r, c)  -> cell label text (default fmt)
//   colors, labelShow, rowLabelWidth
// Column labels turn steeper (and thin out) as the columns narrow.
export function heatmap(el, { rows, cols, values, min, max, fmt, rowLabels, colLabels, diverging = true, height,
  cellStyle, tip, labelFmt, colors, labelShow, rowLabelWidth } = {}) {
  if (height) el.style.height = `${height}px`;
  const chart = mount(el);
  const rl = rowLabels || rows || [], cl = colLabels || cols || [];
  const data = [];
  let lo = Infinity, hi = -Infinity;
  (values || []).forEach((row, r) => (row || []).forEach((v, c) => {
    const ok = isNum(v);
    if (ok) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const item = { value: [c, r, ok ? v : '-'] };
    const st = cellStyle ? cellStyle(r, c, ok ? v : null) : null;
    if (st) item.itemStyle = st;
    data.push(item);
  }));
  if (lo > hi) { lo = -1; hi = 1; }
  const bound = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const width = el.clientWidth || 600;
  const rowW = Math.round(rowLabelWidth ?? Math.min(240, Math.max(80, width * 0.3)));
  const rowPx = Math.min(rowW, Math.max(16, ...rl.map(s => textPx(s))));
  const cellW = (width - rowPx - 40) / Math.max(cl.length, 1);
  const cellH = ((el.clientHeight || height || 300) - 90) / Math.max(rl.length, 1);
  const colPx = Math.max(10, ...cl.map(s => textPx(s)));
  const rotate = colPx <= cellW - 6 ? 0 : cellW >= 24 ? 40 : 90;
  const interval = rotate === 90 && cellW < 13 ? Math.ceil(13 / Math.max(cellW, 1)) - 1 : 0;
  const lf = (v, r, c) => (labelFmt ? labelFmt(v, r, c) : fmtValue(v, fmt));
  const show = labelShow ?? (data.length <= 400 && cellW >= 30 && cellH >= 13);
  const fontSize = cellW < 40 ? 9 : 10;
  // a label wider than its cell drops its "%" (the unit is in the card text), and is hidden if still too wide
  const fit = (t) => {
    const s = String(t ?? '');
    if (textPx(s, fontSize) <= cellW - 4) return s;
    if (s.endsWith('%') && textPx(s.slice(0, -1), fontSize) <= cellW - 4) return s.slice(0, -1);
    return '';
  };
  chart.setOption({
    tooltip: {
      confine: true,
      formatter: (p) => {
        const [c, r, v] = p.value || p.data?.value || [];
        const val = isNum(v) ? v : null;
        if (tip) return tip(r, c, val);
        return `${esc(rl[r])} × ${esc(cl[c])}<br/><b>${val === null ? '无数据' : esc(fmtValue(val, fmt))}</b>`;
      },
    },
    grid: { left: 8, right: 20, top: 10, bottom: 8, containLabel: true },
    xAxis: { type: 'category', data: cl, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, rotate, interval }, splitArea: { show: false } },
    yAxis: { type: 'category', data: rl, inverse: true, ...axisCommon,
      axisLabel: { ...axisCommon.axisLabel, interval: 0, width: rowW, overflow: 'truncate' } },
    visualMap: {
      min: min ?? (diverging ? -bound : lo), max: max ?? (diverging ? bound : hi), calculable: false, show: false, dimension: 2,
      inRange: { color: colors || (diverging ? ['#1c7ed6', '#a5d8ff', '#f8f9fa', '#ffc9c9', '#e03131'] : ['#f8f9fa', '#bac8ff', '#3b5bdb']) },
    },
    series: [{ type: 'heatmap', data,
      label: { show, fontSize, color: INK,
        formatter: (p) => { const [c, r, v] = p.value; return isNum(v) ? fit(lf(v, r, c)) : ''; } },
      emphasis: { itemStyle: { borderColor: INK, borderWidth: 1 } } }],
  });
  return chart;
}

// ---------------------------------------------------------------- bars

// series: [{name, data, color?, stack?}]
// wrap: category labels wrap inside their own slot instead of being cut (horizontal: a number sets
// the label width in px, default up to 190); markLine: [{xAxis | yAxis, label?}].
// Labels adapt to the box: horizontal bars give the category labels at most ~38% of the width and
// thin out the value ticks; vertical bars keep labels flat while they fit their slot, then turn
// them 40° or 90° and show every k-th one when even turned labels would collide.
export function bars(el, { x, series, horizontal = false, fmt, legend = true, markLine, wrap = false } = {}) {
  const chart = mount(el);
  const n = Math.max(x.length, 1);
  const width = el.clientWidth || 500;
  let catLabel, bottom = 32, splitNumber;
  if (horizontal) {
    const lw = typeof wrap === 'number' ? wrap : Math.round(Math.min(190, Math.max(64, width * 0.38)));
    catLabel = wrap ? { interval: 0, rotate: 0, width: lw, overflow: 'break', lineHeight: 14 }
      : { interval: 0, rotate: 0, width: lw, overflow: 'truncate' };
    splitNumber = Math.max(2, Math.min(5, Math.floor((width - lw - 50) / 64)));
  } else if (wrap) {
    const lw = Math.max(40, (width - 90) / n - 8);
    catLabel = { interval: 0, rotate: 0, width: lw, overflow: 'break', lineHeight: 14 };
    const lines = Math.max(1, ...x.map(t => Math.ceil(textPx(t) / lw)));      // room for the tallest wrapped label
    bottom = Math.max(40, Math.min(110, lines * 14 + 24));
  } else {
    const slot = (width - 80) / n;
    const longest = Math.max(10, ...x.map(s => textPx(s)));
    let rotate = 0, interval = 0;
    if (longest > slot - 6) {
      rotate = slot >= 22 ? 40 : 90;
      const need = rotate === 90 ? 14 : 22;         // px between neighbouring turned labels
      interval = slot >= need ? 0 : Math.ceil(need / Math.max(slot, 1)) - 1;
    }
    catLabel = { interval, rotate };
    if (rotate) bottom = Math.min(120, Math.round(longest * Math.sin((rotate * Math.PI) / 180) + 24));
  }
  const cat = { type: 'category', data: x, ...axisCommon, splitLine: { show: false }, axisLabel: { ...axisCommon.axisLabel, ...catLabel } };
  const val = { type: 'value', splitNumber, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, hideOverlap: true, formatter: (v) => fmtValue(v, fmt) } };
  chart.setOption({
    legend: legend && series.length > 1 ? legendCommon : undefined,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true, valueFormatter: (v) => fmtValue(v, fmt) },
    grid: horizontal ? { left: 8, right: 30, top: series.length > 1 ? 30 : markLine ? 24 : 10, bottom: 24, containLabel: true }
      : baseGrid({ bottom }),
    xAxis: horizontal ? val : cat,
    yAxis: horizontal ? { ...cat, inverse: true } : val,
    series: series.map((s, i) => ({ name: s.name, type: 'bar', data: s.data, stack: s.stack, barMaxWidth: 26,
      itemStyle: { color: s.color || palette(i), borderRadius: 2 },
      // a vertical line's label goes above the plot (the inverted category axis starts at the top),
      // a horizontal one's inside the plot next to the value axis, so neither is cut at the edge
      markLine: i === 0 && markLine ? markLineOf(markLine, {}, fmt, horizontal ? 'start' : 'insideStartTop') : undefined })),
  });
  return chart;
}

// ---------------------------------------------------------------- causal graph

const SIGN_COLOR = { '+': '#3b5bdb', '-': '#e8590c', '−': '#e8590c', '±': '#868e96' };
const SIGN_WORD = { '+': '同向', '-': '反向', '−': '反向', '±': '方向不定' };
const SIGN_CHAR = { '-': '−' };

// At most `lines` lines of about `perLine` CJK characters (Latin letters count about half).
function fitLabel(text, perLine, lines = 2) {
  const s = String(text ?? '');
  let units = 0, out = '';
  for (const ch of s) {
    units += /[\x00-\xff]/.test(ch) ? 0.55 : 1;
    if (units > perLine * lines - 0.5) return `${out}…`;
    out += ch;
  }
  return s;
}

// Layered layout in pixels: longest-path layers, nodes inside a layer ordered by the average
// position of their causes and effects (barycenter sweeps, removes most crossings); left to right
// when the layers fit with room for an edge label between columns (boxes narrow from 150 to 112
// px first), otherwise top to bottom. Top to bottom, every connected part of the graph (a chain
// that shares no box with the others) is its own block of rows and the blocks sit side by side
// while they fit (then on further shelves), so separate chains never wrap into each other's rows;
// boxes narrow to 64 px before a layer wider than the card wraps onto an extra row.
// rank = column (or row) of each node. Top to bottom, an edge that skips rows gets a free slot in
// every row it crosses (a placeholder in the ordering), so it never has to run under a box.
export function dagLayout(nodes, edges, width, { nodeH = 40 } = {}) {
  const ids = nodes.map(n => n.id);
  const layer = new Map(ids.map(id => [id, 0]));
  for (let iter = 0; iter < ids.length; iter++) {
    let moved = false;
    for (const e of edges) {
      if (!layer.has(e.from) || !layer.has(e.to) || e.from === e.to) continue;
      const want = layer.get(e.from) + 1;
      if (layer.get(e.to) < want && want < ids.length) { layer.set(e.to, want); moved = true; }
    }
    if (!moved) break;
  }
  // layers ordered by barycenter sweeps (removes most crossings)
  const ordered = (allIds, layerOf, links) => {
    const byLayer = [];
    allIds.forEach(id => { const l = layerOf.get(id); (byLayer[l] = byLayer[l] || []).push(id); });
    const L = byLayer.filter(Boolean);
    const lidx = new Map();
    L.forEach((list, l) => list.forEach(id => lidx.set(id, l)));
    const preds = new Map(), succs = new Map();
    for (const e of links) {
      if (!lidx.has(e.from) || !lidx.has(e.to) || e.from === e.to) continue;
      preds.set(e.to, [...(preds.get(e.to) || []), e.from]);
      succs.set(e.from, [...(succs.get(e.from) || []), e.to]);
    }
    // centred slot of each node inside its layer: comparable across layers of different sizes
    const slot = () => { const m = new Map(); L.forEach(list => list.forEach((id, i) => m.set(id, i - (list.length - 1) / 2))); return m; };
    const sweep = (l, nbrs, before) => {
      const m = slot();
      const key = (id) => {
        const xs = (nbrs.get(id) || []).filter(x => (before ? lidx.get(x) < l : lidx.get(x) > l));
        return xs.length ? xs.reduce((t, x) => t + m.get(x), 0) / xs.length : m.get(id);
      };
      L[l] = L[l].map((id, i) => ({ id, k: key(id), i })).sort((a, b) => a.k - b.k || a.i - b.i).map(o => o.id);
    };
    for (let pass = 0; pass < 2; pass++) {
      for (let l = 1; l < L.length; l++) sweep(l, preds, true);
      for (let l = L.length - 2; l >= 0; l--) sweep(l, succs, false);
    }
    for (let l = 1; l < L.length; l++) sweep(l, preds, true);
    return L;
  };
  const L = ordered(ids, layer, edges);

  const NODE_W = 150, PAD_X = 10, PAD_Y = 26, pos = {}, rank = {};
  const maxN = Math.max(1, ...L.map(l => l.length));
  let wH = 0;
  for (let w = NODE_W; w >= 112; w -= 2) {
    if (L.length <= 1 || (width - 2 * PAD_X - w) / (L.length - 1) >= w + 56) { wH = w; break; }
  }
  if (wH) {
    const rowGap = nodeH + 26;
    const height = Math.max(200, (maxN - 1) * rowGap + nodeH + 2 * PAD_Y);
    const x0 = PAD_X + wH / 2, x1 = width - PAD_X - wH / 2;
    const colGap = L.length > 1 ? Math.min(280, (x1 - x0) / (L.length - 1)) : 0;
    const left = L.length > 1 ? x0 + (x1 - x0 - colGap * (L.length - 1)) / 2 : width / 2;
    L.forEach((list, l) => list.forEach((id, i) => { pos[id] = [left + l * colGap, height / 2 + (i - (list.length - 1) / 2) * rowGap]; rank[id] = l; }));
    return { pos, height, w: wH, h: nodeH, rank, vertical: false };
  }
  // ---- top to bottom. A forward edge that skips rows is split into a chain through placeholders
  // (one per row it crosses), which take a slot like a box but are not drawn.
  const DUMMY = '\u0001';
  const vLayer = new Map(layer), vIds = [...ids], vEdges = [];
  let nDummy = 0;
  for (const e of edges) {
    if (!layer.has(e.from) || !layer.has(e.to) || e.from === e.to) continue;
    const a = layer.get(e.from), b = layer.get(e.to);
    if (b - a <= 1) { vEdges.push(e); continue; }
    let prev = e.from;
    for (let j = a + 1; j < b; j++) {
      const id = `${DUMMY}${nDummy++}`;
      vIds.push(id); vLayer.set(id, j); vEdges.push({ from: prev, to: id });
      prev = id;
    }
    vEdges.push({ from: prev, to: e.to });
  }
  const LV = nDummy ? ordered(vIds, vLayer, vEdges) : L;
  // connected parts in order of first appearance, each keeping the layer order
  const comp = new Map(), adj = new Map(vIds.map(id => [id, []]));
  for (const e of vEdges) {
    if (adj.has(e.from) && adj.has(e.to) && e.from !== e.to) { adj.get(e.from).push(e.to); adj.get(e.to).push(e.from); }
  }
  let nComp = 0;
  for (const id of vIds) {
    if (comp.has(id)) continue;
    const todo = [id];
    comp.set(id, nComp);
    while (todo.length) for (const v of adj.get(todo.pop())) if (!comp.has(v)) { comp.set(v, nComp); todo.push(v); }
    nComp += 1;
  }
  const parts = [...Array(nComp).keys()].map(c => LV.map(list => list.filter(id => comp.get(id) === c)).filter(l => l.length));
  const widest = Math.max(1, ...parts.flatMap(p => p.map(l => l.length)));
  // several parts: boxes narrow enough for two of them side by side (when the card allows it)
  const fitCols = Math.max(widest, nComp > 1 ? 2 : 1);
  const inner = width - 2 * PAD_X + 12;
  const nodeW = Math.min(NODE_W, Math.max(64, Math.floor(inner / fitCols) - 12));
  const colGap = nodeW + 12, rowGap = nodeH + 54, SHELF_GAP = 12;     // a row gap with room for the labels of a fan-out
  const perRow = Math.max(1, Math.floor(inner / colGap));
  const blocks = parts.map(p => {
    const rows = [];
    p.forEach(list => { for (let k = 0; k < list.length; k += perRow) rows.push(list.slice(k, k + perRow)); });
    return { rows, cols: Math.max(1, ...rows.map(r => r.length)) };
  });
  const shelves = [];
  for (const b of blocks) {
    const cur = shelves[shelves.length - 1];
    if (cur && cur.cols + b.cols <= perRow) { cur.blocks.push(b); cur.cols += b.cols; cur.rows = Math.max(cur.rows, b.rows.length); }
    else shelves.push({ blocks: [b], cols: b.cols, rows: b.rows.length });
  }
  let y = PAD_Y + nodeH / 2, r0 = 0;
  for (const sh of shelves) {
    let x = width / 2 - (sh.cols * colGap - 12) / 2;            // the shelf is centred
    for (const b of sh.blocks) {
      const cx = x + (b.cols * colGap - 12) / 2;
      b.rows.forEach((row, r) => row.forEach((id, i) => {
        if (String(id).startsWith(DUMMY)) return;                 // a free slot for a long edge
        pos[id] = [cx + (i - (row.length - 1) / 2) * colGap, y + r * rowGap]; rank[id] = r0 + r;
      }));
      x += b.cols * colGap;
    }
    y += sh.rows * rowGap + SHELF_GAP; r0 += sh.rows + 1;
  }
  const height = Math.max(200, Math.round(y - rowGap - SHELF_GAP + nodeH / 2 + PAD_Y));
  return { pos, height, w: nodeW, h: nodeH, rank, vertical: true };
}

// Layered causal graph laid out in pixels on a hidden cartesian grid, so boxes keep their size
// and never overlap; laid out again when the box gets much wider or narrower.
//   nodes: [{id, label?, group?, style?: {color, borderColor}}]; ids only named by edges are added
//   edges: [{from, to, sign?: '+' | '-' | '±', lag?, label?, mechanism?, evidence?}]
// Edges are drawn as lines cut at the box borders, so every arrowhead is visible; an edge bends
// when it would otherwise run under a box it does not connect (and in a two-way pair), and its
// label sits next to it, off the boxes and the other edges. Options:
//   height                 minimum height in px (the layout's own height is used when larger)
//   nodeStyle(n)           -> {color, borderColor, borderWidth} for a node (default: n.style or blue)
//   edgeColor(e)           -> line colour (default by sign: blue +, orange −, grey ±)
//   edgeLabel(e)           -> short text on the edge (default: sign and lag)
//   edgeTip(e, label)      -> tooltip HTML for an edge; label(id) gives a node's label (escape it)
//   nodeTip(n)             -> tooltip HTML for a node
// Tooltips name nodes by their labels and escape every text by default (it may come from an LLM).
export function dag(el, opts = {}) {
  const { nodes = [], edges = [], height, nodeH = 40, nodeStyle, edgeColor, edgeLabel, edgeTip, nodeTip } = opts;
  const width = el.clientWidth || 800;
  const known = new Map();
  for (const n of nodes) if (n && n.id !== undefined && n.id !== null && !known.has(n.id)) known.set(n.id, n);
  for (const e of edges) {
    for (const id of [e?.from, e?.to]) if (id !== undefined && id !== null && id !== '' && !known.has(id)) known.set(id, { id, label: String(id) });
  }
  const all = [...known.values()];
  const list = edges.filter(e => e && known.has(e.from) && known.has(e.to) && e.from !== e.to);
  const lay = dagLayout(all, list, width, { nodeH });
  const { pos, w, rank } = lay;
  el.style.height = `${Math.max(lay.height, Number(height) || 0)}px`;
  const H = Math.max(lay.height, Number(height) || 0);
  const dy0 = (H - lay.height) / 2;                    // centre a layout shorter than the box
  for (const id of Object.keys(pos)) pos[id] = [pos[id][0], pos[id][1] + dy0];
  const chart = mount(el);

  const labelOf = (id) => String(known.get(id)?.label ?? id ?? '');
  const colorOf = (e) => (edgeColor ? edgeColor(e) : SIGN_COLOR[e.sign] || '#3b5bdb');
  const textOf = (e) => (edgeLabel ? String(edgeLabel(e) ?? '') : `${SIGN_CHAR[e.sign] || e.sign || ''} ${e.lag || ''}`.trim());
  const clipAt = (c, o, gap) => {
    const dx = o[0] - c[0], dy = o[1] - c[1];
    const tx = dx ? (w / 2 + gap) / Math.abs(dx) : Infinity, ty = dy ? (nodeH / 2 + gap) / Math.abs(dy) : Infinity;
    const t = Math.min(tx, ty, 0.45);
    return [c[0] + dx * t, c[1] + dy * t];
  };
  const pairs = new Set(list.map(e => `${e.from}\u0000${e.to}`));
  // Routing: an edge never runs under a box it does not connect (it would read as passing through
  // that box) and a label never covers a box or another label. Each edge takes the first
  // curvature that misses every other box (straight if possible), each label the first spot along
  // its edge, on either side, that is free. The control point is the one ECharts' `lines` uses.
  const boxes = Object.keys(pos).map(id => ({ id, x: pos[id][0], y: pos[id][1] }));
  const boxRects = boxes.map(b => ({ id: b.id, l: b.x - w / 2 - 2, r: b.x + w / 2 + 2, t: b.y - nodeH / 2 - 2, b: b.y + nodeH / 2 + 2 }));
  const ctrl = (p0, p1, c) => [(p0[0] + p1[0]) / 2 - (p0[1] - p1[1]) * c, (p0[1] + p1[1]) / 2 - (p1[0] - p0[0]) * c];
  const bezAt = (p0, cp, p1, t) => [0, 1].map(k => (1 - t) ** 2 * p0[k] + 2 * (1 - t) * t * cp[k] + t * t * p1[k]);
  const hits = (p0, p1, c, skip) => {
    const cp = ctrl(p0, p1, c);
    let n = 0;
    for (let k = 1; k < 32; k++) {
      const [x, y] = bezAt(p0, cp, p1, k / 32);
      if (x < 2 || x > width - 2 || y < 2 || y > H - 2) n += 1;                 // off the chart
      for (const r of boxRects) if (!skip.has(r.id) && x > r.l - 2 && x < r.r + 2 && y > r.t - 2 && y < r.b + 2) n += 1;
    }
    return n;
  };
  const CURVES = [0, 0.18, -0.18, 0.3, -0.3, 0.45, -0.45, 0.6, -0.6, 0.8, -0.8];
  const links = list.map(e => {
    const a = pos[e.from], b = pos[e.to];
    const p0 = clipAt(a, b, 2), p1 = clipAt(b, a, 3);
    const span = Math.abs((rank[e.to] ?? 0) - (rank[e.from] ?? 0));
    const twoWay = pairs.has(`${e.to}\u0000${e.from}`);
    // a two-way pair keeps one sign (the two edges then bend to opposite sides)
    const base = twoWay ? 0.25 : span > 1 ? 0.18 : 0;
    const cands = twoWay ? [0.25, 0.4, 0.55, 0.7] : [base, ...CURVES.filter(c => c !== base)];
    const skip = new Set([String(e.from), String(e.to)]);
    let best = null;
    for (const c of cands) {
      const n = hits(p0, p1, c, skip);
      if (!best || n < best.n) best = { c, n };
      if (n === 0) break;
    }
    return { e, p0, p1, curv: best.c, span, text: textOf(e) };
  });
  const placed = [];
  const overlap = (a, b) => Math.max(0, Math.min(a.r, b.r) - Math.max(a.l, b.l)) * Math.max(0, Math.min(a.b, b.b) - Math.max(a.t, b.t));
  // points along every routed edge: a label also stays off the other edges, so it cannot be read
  // as belonging to a neighbouring arrow
  for (const l of links) {
    const cp = ctrl(l.p0, l.p1, l.curv);
    l.pts = [];
    for (let k = 0; k <= 24; k++) l.pts.push(bezAt(l.p0, cp, l.p1, k / 24));
  }
  const onEdges = (rect, self) => {
    let n = 0;
    for (const o of links) {
      if (o === self) continue;
      for (const [x, y] of o.pts) if (x > rect.l - 2 && x < rect.r + 2 && y > rect.t - 2 && y < rect.b + 2) n += 1;
    }
    return n;
  };
  for (const l of links) {
    if (!l.text) continue;
    const { p0, p1, curv, span } = l;
    const cp = ctrl(p0, p1, curv);
    const tw = textPx(l.text, 10) + 8, th = 16;
    // the label of a long edge starts in its last gap, not on the boxes it passes
    const ts = span > 1 ? [1 - 0.5 / span, 0.5, 0.65, 0.35, 0.8, 0.2] : [0.5, 0.4, 0.6, 0.3, 0.7, 0.22, 0.78, 0.84, 0.16];
    let best = null;
    for (const t of ts) {
      const [x, y] = bezAt(p0, cp, p1, t);
      const tx = 2 * (1 - t) * (cp[0] - p0[0]) + 2 * t * (p1[0] - cp[0]), ty = 2 * (1 - t) * (cp[1] - p0[1]) + 2 * t * (p1[1] - cp[1]);
      const len = Math.hypot(tx, ty) || 1;
      // preferred side: above a flat edge, to the right of a steep one
      let nx = -ty / len, ny = tx / len;
      if (Math.abs(tx) >= Math.abs(ty) ? ny > 0 : nx < 0) { nx = -nx; ny = -ny; }
      for (const side of [1, -1]) {
        for (const extra of [0, 8, 16]) {
          const d = Math.abs(nx) * tw / 2 + Math.abs(ny) * th / 2 + 2 + extra;
          const cx = x + side * nx * d, cy = y + side * ny * d;
          const rect = { l: cx - tw / 2, r: cx + tw / 2, t: cy - th / 2, b: cy + th / 2 };
          let score = 30 * onEdges(rect, l);
          for (const r of boxRects) score += overlap(rect, r);
          for (const r of placed) score += 2 * overlap(rect, r);
          score += Math.max(0, -rect.l) * th + Math.max(0, rect.r - width) * th + Math.max(0, -rect.t) * tw + Math.max(0, rect.b - H) * tw;
          if (!best || score < best.score) best = { score, at: [cx, cy], rect };
          if (score === 0) break;
        }
        if (best.score === 0) break;
      }
      if (best.score === 0) break;
    }
    l.at = best.at;
    placed.push(best.rect);
  }
  const defaultEdgeTip = (e) => {
    const why = [e.mechanism || e.label, e.evidence].filter(Boolean).map(t => esc(t)).join('<br/>');
    return `<b>${esc(labelOf(e.from))} → ${esc(labelOf(e.to))}</b><br/>${SIGN_WORD[e.sign] || '方向不定'}`
      + `${e.lag ? `，时滞 ${esc(e.lag)}` : ''}${why ? `<br/>${why}` : ''}`;
  };
  const perLine = Math.max(4, Math.floor((w - 12) / 12));
  chart.setOption({
    animation: false,
    tooltip: {
      trigger: 'item', confine: true, extraCssText: 'max-width: 380px; white-space: normal;',
      formatter: (p) => {
        const d = p.data || {};
        if (d.edge) return edgeTip ? edgeTip(d.edge, labelOf) : defaultEdgeTip(d.edge);
        if (d.node) return nodeTip ? nodeTip(d.node) : `<b>${esc(labelOf(d.node.id))}</b>`;
        return '';
      },
    },
    grid: { left: 0, right: 0, top: 0, bottom: 0 },
    xAxis: { type: 'value', min: 0, max: width, show: false },
    yAxis: { type: 'value', min: 0, max: H, inverse: true, show: false },
    series: [
      { type: 'lines', coordinateSystem: 'cartesian2d', z: 1, symbol: ['none', 'arrow'], symbolSize: 9,
        data: links.map(l => ({ coords: [l.p0, l.p1], edge: l.e,
          lineStyle: { color: colorOf(l.e), width: 1.6, opacity: 0.9, curveness: l.curv } })),
        emphasis: { lineStyle: { width: 2.6 } } },
      { type: 'graph', coordinateSystem: 'cartesian2d', z: 2, symbol: 'roundRect', symbolSize: [w, nodeH],
        label: { show: true, color: INK, fontSize: w < 120 ? 11 : 12, width: w - 12, overflow: 'break', lineHeight: 15 },
        itemStyle: { color: '#f3f6ff', borderColor: '#bac8ff', borderWidth: 1 },
        emphasis: { scale: false },
        // a function formatter, so "{…}" in a (possibly LLM-written) label is not read as a template
        data: all.map(n => {
          const text = fitLabel(labelOf(n.id), perLine);
          const st = nodeStyle ? nodeStyle(n) : n.style;
          return { name: String(n.id), value: pos[n.id], node: n, itemStyle: st || undefined, label: { formatter: () => text } };
        }),
        links: [] },
      { type: 'scatter', z: 3, symbolSize: 1, itemStyle: { color: 'rgba(0, 0, 0, 0)' },
        data: links.filter(l => l.text).map(l => ({ value: l.at, edge: l.e, text: l.text })),
        label: { show: true, position: 'inside', fontSize: 10, color: INK2, backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: [1, 3], borderRadius: 3, formatter: (p) => p.data.text },
        emphasis: { scale: false } },
    ],
  });
  // lay out again when the box gets much wider or narrower (window resize, sidebar collapse)
  el._dagOpts = opts;
  el._dagWidth = width;
  if (!el._dagRO) {
    let raf = 0;
    el._dagRO = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        if (!el.isConnected || !el._chart) return;
        if (Math.abs((el.clientWidth || 0) - el._dagWidth) > 40) dag(el, el._dagOpts);
      });
    });
    el._dagRO.observe(el);
  }
  return chart;
}

// ---------------------------------------------------------------- scatter

// points: [{x, y, name?}]
export function scatter(el, { points, xName = '', yName = '', fmtX, fmtY, diagonal = false } = {}) {
  const chart = mount(el);
  const pts = (points || []).filter(p => isNum(p.x) && isNum(p.y));
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const lo = pts.length ? Math.min(...xs, ...ys) : 0, hi = pts.length ? Math.max(...xs, ...ys) : 1;
  chart.setOption({
    tooltip: { confine: true, formatter: (p) => `${esc(p.data[2] || '')}<br/>${esc(xName)}: ${esc(fmtValue(p.data[0], fmtX))}<br/>${esc(yName)}: ${esc(fmtValue(p.data[1], fmtY))}` },
    grid: baseGrid({ bottom: 44 }),
    xAxis: { type: 'value', name: xName, nameLocation: 'middle', nameGap: 26, scale: true, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, fmtX) } },
    yAxis: { type: 'value', name: yName, scale: true, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, fmtY) } },
    series: [{ type: 'scatter', symbolSize: 7, data: pts.map(p => [p.x, p.y, p.name]), itemStyle: { color: '#3b5bdb', opacity: 0.7 } },
      ...(diagonal ? [{ type: 'line', data: [[lo, lo], [hi, hi]], symbol: 'none', lineStyle: { color: INK3, type: 'dashed' } }] : [])],
  });
  return chart;
}
