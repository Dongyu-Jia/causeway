// ECharts helpers shared by all tabs. Every helper takes a container element
// (from ui.chartBox()) and returns the chart instance.
/* global echarts */

const INK = '#1d2433', INK3 = '#7a8396', LINE = '#e3e6ec';
const PALETTE = ['#3b5bdb', '#e8590c', '#2f9e44', '#c2255c', '#7048e8', '#1098ad', '#f08c00',
  '#5c940d', '#d6336c', '#495057', '#0b7285', '#a61e4d', '#364fc7'];
const ASSET_COLORS = {
  us_equity: '#3b5bdb', nasdaq: '#7048e8', energy: '#e8590c', industrials: '#495057', staples: '#2f9e44',
  healthcare: '#1098ad', utilities: '#0b7285', financials: '#364fc7', real_estate: '#c2255c',
  gold: '#f08c00', crude_oil: '#5f3dc4', treasury_10y: '#2b8a3e', cash: '#868e96',
};
export const assetColor = (id, i = 0) => ASSET_COLORS[id] || PALETTE[i % PALETTE.length];
export const palette = (i) => PALETTE[i % PALETTE.length];

let klineScheme = localStorage.getItem('kline-color') || 'cn';
export function setKlineScheme(s) { klineScheme = s; try { localStorage.setItem('kline-color', s); } catch (_) { /* ignore */ } }
export function klineColors() {
  const red = '#d9363e', green = '#1a9c5b';
  return klineScheme === 'cn' ? { up: red, down: green } : { up: green, down: red };
}

export function fmtValue(v, fmt) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (typeof fmt === 'function') return fmt(v);
  if (fmt === 'pct') return `${(v * 100).toFixed(1)}%`;
  if (fmt === 'pct2') return `${(v * 100).toFixed(2)}%`;
  if (fmt === 'pp') return `${Number(v).toFixed(2)}%`;
  if (fmt === 'x') return `${Number(v).toFixed(2)}x`;
  if (fmt === 'int') return Math.round(v).toLocaleString('zh-CN');
  return Number(v).toFixed(Math.abs(v) >= 100 ? 0 : 2);
}

export function mount(el) {
  if (el._chart) { el._chart.dispose(); }
  const chart = echarts.init(el, null, { renderer: 'canvas' });
  el._chart = chart;
  if (!el._ro) {
    el._ro = new ResizeObserver(() => el._chart && el._chart.resize());
    el._ro.observe(el);
  }
  return chart;
}

const baseGrid = (extra = {}) => ({ left: 56, right: 24, top: 36, bottom: 40, containLabel: false, ...extra });
const axisCommon = {
  axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false },
  axisLabel: { color: INK3, fontSize: 11 }, splitLine: { lineStyle: { color: '#eef0f4' } },
};
const legendCommon = { top: 0, textStyle: { color: INK, fontSize: 12 }, itemWidth: 14, itemHeight: 8, type: 'scroll' };

function markAreaOf(areas) {
  if (!areas || !areas.length) return undefined;
  return {
    silent: true,
    itemStyle: { color: 'rgba(59, 91, 219, 0.06)' },
    label: { color: INK3, fontSize: 10, position: 'insideTop' },
    data: areas.map(a => [{ name: a.label || '', xAxis: a.from, itemStyle: a.color ? { color: a.color } : undefined }, { xAxis: a.to }]),
  };
}

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
    tooltip: { trigger: 'axis', valueFormatter: (v) => fmtValue(v, tooltipFmt || yFmt) },
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
      markLine: i === 0 && markLines ? { silent: true, symbol: 'none', lineStyle: { color: INK3, type: 'dashed' },
        label: { color: INK3, fontSize: 10 }, data: markLines } : undefined,
    })),
  });
  return chart;
}

// Percentile fan: bands = {p5, p25, p50, p75, p95}; optional actual path and extra series.
export function fan(el, { x, bands, actual, actualName = '实际', name = '中位数', color = '#3b5bdb', fmt, extra = [],
  markAreas, legend = true, yName = '', zoom = false } = {}) {
  const chart = mount(el);
  const p5 = bands.p5, p25 = bands.p25, p50 = bands.p50, p75 = bands.p75, p95 = bands.p95;
  const diff = (a, b) => a.map((v, i) => (v === null || b[i] === null ? null : v - b[i]));
  const series = [
    { name: 'p5', type: 'line', data: p5, stack: 'fan', symbol: 'none', lineStyle: { opacity: 0 }, tooltip: { show: false } },
    { name: '5–25%', type: 'line', data: diff(p25, p5), stack: 'fan', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color, opacity: 0.12 } },
    { name: '25–75%', type: 'line', data: diff(p75, p25), stack: 'fan', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color, opacity: 0.26 } },
    { name: '75–95%', type: 'line', data: diff(p95, p75), stack: 'fan', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color, opacity: 0.12 } },
    { name, type: 'line', data: p50, symbol: 'none', lineStyle: { width: 2, color }, itemStyle: { color }, markArea: markAreaOf(markAreas) },
  ];
  if (actual) series.push({ name: actualName, type: 'line', data: actual, symbol: 'none', lineStyle: { width: 2, color: INK }, itemStyle: { color: INK } });
  extra.forEach((s, i) => series.push({ type: 'line', symbol: 'none', ...s,
    lineStyle: { width: 1.5, type: s.dashed ? 'dashed' : 'solid', color: s.color || palette(i + 3) },
    itemStyle: { color: s.color || palette(i + 3) } }));
  chart.setOption({
    legend: legend ? { ...legendCommon, data: [name, ...(actual ? [actualName] : []), ...extra.map(s => s.name)] } : undefined,
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const i = params[0].dataIndex;
        const rows = [`<b>${x[i]}</b>`,
          `95%: ${fmtValue(p95[i], fmt)}`, `75%: ${fmtValue(p75[i], fmt)}`, `<b>中位数: ${fmtValue(p50[i], fmt)}</b>`,
          `25%: ${fmtValue(p25[i], fmt)}`, `5%: ${fmtValue(p5[i], fmt)}`];
        if (actual) rows.push(`<b>${actualName}: ${fmtValue(actual[i], fmt)}</b>`);
        extra.forEach(s => rows.push(`${s.name}: ${fmtValue(s.data[i], fmt)}`));
        return rows.join('<br/>');
      },
    },
    grid: baseGrid({ bottom: zoom ? 56 : 32 }),
    xAxis: { type: 'category', data: x, boundaryGap: false, ...axisCommon, splitLine: { show: false } },
    yAxis: { type: 'value', scale: true, name: yName, nameTextStyle: { color: INK3, fontSize: 11 }, ...axisCommon,
      axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, fmt) } },
    dataZoom: zoom ? [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }] : undefined,
    series,
  });
  return chart;
}

function movingAverage(closes, n) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < closes.length; i++) {
    sum += closes[i];
    if (i >= n) sum -= closes[i - n];
    out.push(i >= n - 1 ? +(sum / n).toFixed(3) : null);
  }
  return out;
}

// data: [[label, open, close, low, high], ...]
export function candles(el, { data, name = 'K 线', ma = [12], fmt, log = false, zoomStart = 0 } = {}) {
  const chart = mount(el);
  const { up, down } = klineColors();
  const labels = data.map(d => d[0]);
  const ohlc = data.map(d => [d[1], d[2], d[3], d[4]]);
  const closes = data.map(d => d[2]);
  const maSeries = ma.filter(n => n < data.length).map((n, i) => ({
    name: `MA${n}`, type: 'line', data: movingAverage(closes, n), symbol: 'none', smooth: true,
    lineStyle: { width: 1.2, color: ['#f08c00', '#7048e8', '#1098ad'][i % 3] },
  }));
  chart.setOption({
    legend: { ...legendCommon, data: [name, ...maSeries.map(s => s.name)] },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'cross' },
      formatter: (params) => {
        const k = params.find(p => p.seriesType === 'candlestick');
        if (!k) return '';
        const [o, c, l, hgh] = k.data.slice(1);
        const chg = o ? (c / o - 1) : 0;
        return `<b>${k.name}</b><br/>开 ${fmtValue(o, fmt)}　收 ${fmtValue(c, fmt)}<br/>低 ${fmtValue(l, fmt)}　高 ${fmtValue(hgh, fmt)}<br/>涨跌 ${(chg * 100).toFixed(2)}%`;
      },
    },
    grid: baseGrid({ bottom: 56 }),
    xAxis: { type: 'category', data: labels, ...axisCommon, splitLine: { show: false } },
    yAxis: { type: log ? 'log' : 'value', scale: true, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, fmt) } },
    dataZoom: [{ type: 'inside', start: zoomStart, end: 100 }, { type: 'slider', height: 16, bottom: 8, start: zoomStart, end: 100 }],
    series: [{ name, type: 'candlestick', data: ohlc,
      itemStyle: { color: up, color0: down, borderColor: up, borderColor0: down } }, ...maSeries],
  });
  return chart;
}

// values[r][c]; diverging palette centred at 0 by default.
export function heatmap(el, { rows, cols, values, min, max, fmt, rowLabels, colLabels, diverging = true, height } = {}) {
  if (height) el.style.height = `${height}px`;
  const chart = mount(el);
  const data = [];
  let lo = Infinity, hi = -Infinity;
  values.forEach((row, r) => row.forEach((v, c) => {
    data.push([c, r, v === null ? '-' : v]);
    if (v !== null && v !== undefined) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }));
  const bound = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  chart.setOption({
    tooltip: { formatter: (p) => `${(rowLabels || rows)[p.data[1]]} × ${(colLabels || cols)[p.data[0]]}<br/><b>${fmtValue(p.data[2], fmt)}</b>` },
    grid: { left: 130, right: 20, top: 10, bottom: 80 },
    xAxis: { type: 'category', data: colLabels || cols, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, rotate: 40, interval: 0 }, splitArea: { show: false } },
    yAxis: { type: 'category', data: rowLabels || rows, inverse: true, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, interval: 0 } },
    visualMap: {
      min: min ?? (diverging ? -bound : lo), max: max ?? (diverging ? bound : hi), calculable: false, show: false,
      inRange: { color: diverging ? ['#1c7ed6', '#a5d8ff', '#f8f9fa', '#ffc9c9', '#e03131'] : ['#f8f9fa', '#bac8ff', '#3b5bdb'] },
    },
    series: [{ type: 'heatmap', data, label: { show: (rows.length * cols.length) <= 260, fontSize: 10, color: INK,
      formatter: (p) => (p.data[2] === '-' ? '' : fmtValue(p.data[2], fmt)) },
      emphasis: { itemStyle: { borderColor: INK, borderWidth: 1 } } }],
  });
  return chart;
}

// series: [{name, data, color?, stack?}]
export function bars(el, { x, series, horizontal = false, fmt, legend = true, markLine } = {}) {
  const chart = mount(el);
  const cat = { type: 'category', data: x, ...axisCommon, splitLine: { show: false },
    axisLabel: { ...axisCommon.axisLabel, interval: 0, rotate: horizontal ? 0 : (x.length > 8 ? 35 : 0) } };
  const val = { type: 'value', ...axisCommon, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, fmt) } };
  chart.setOption({
    legend: legend && series.length > 1 ? legendCommon : undefined,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => fmtValue(v, fmt) },
    grid: horizontal ? { left: 130, right: 30, top: series.length > 1 ? 30 : 10, bottom: 24 } : baseGrid({ bottom: x.length > 8 ? 70 : 32 }),
    xAxis: horizontal ? val : cat,
    yAxis: horizontal ? { ...cat, inverse: true } : val,
    series: series.map((s, i) => ({ name: s.name, type: 'bar', data: s.data, stack: s.stack, barMaxWidth: 26,
      itemStyle: { color: s.color || palette(i), borderRadius: 2 },
      markLine: i === 0 && markLine ? { silent: true, symbol: 'none', lineStyle: { color: INK3, type: 'dashed' }, data: markLine } : undefined })),
  });
  return chart;
}

// Layered causal graph. nodes: [{id, label, group?}], edges: [{from, to, sign, lag, label?}]
export function dag(el, { nodes, edges, height } = {}) {
  if (height) el.style.height = `${height}px`;
  const chart = mount(el);
  const ids = nodes.map(n => n.id);
  const layer = Object.fromEntries(ids.map(id => [id, 0]));
  for (let iter = 0; iter < ids.length; iter++) {
    let changed = false;
    for (const e of edges) {
      if (layer[e.from] === undefined || layer[e.to] === undefined) continue;
      if (layer[e.to] < layer[e.from] + 1 && layer[e.from] + 1 < ids.length) { layer[e.to] = layer[e.from] + 1; changed = true; }
    }
    if (!changed) break;
  }
  const byLayer = {};
  ids.forEach(id => { (byLayer[layer[id]] = byLayer[layer[id]] || []).push(id); });
  const pos = {};
  Object.entries(byLayer).forEach(([l, list]) => list.forEach((id, i) => { pos[id] = [Number(l) * 210, (i - (list.length - 1) / 2) * 78]; }));
  const signColor = (s) => (s === '-' ? '#e8590c' : s === '±' ? '#868e96' : '#3b5bdb');
  chart.setOption({
    tooltip: { formatter: (p) => (p.dataType === 'edge' ? `${p.data.source} → ${p.data.target}<br/>${p.data.tip || ''}` : p.data.name) },
    series: [{
      type: 'graph', layout: 'none', roam: true, symbol: 'roundRect', symbolSize: [150, 40],
      label: { show: true, color: INK, fontSize: 12, width: 140, overflow: 'break' },
      itemStyle: { color: '#f3f6ff', borderColor: '#bac8ff', borderWidth: 1 },
      edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 8,
      edgeLabel: { show: true, fontSize: 10, color: INK3, formatter: (p) => p.data.short || '' },
      data: nodes.map(n => ({ name: n.id, value: n.label, x: pos[n.id][0], y: pos[n.id][1], label: { formatter: n.label } })),
      links: edges.filter(e => pos[e.from] && pos[e.to]).map(e => ({
        source: e.from, target: e.to, short: `${e.sign || ''} ${e.lag || ''}`.trim(),
        tip: [e.mechanism || e.label || '', e.evidence || ''].filter(Boolean).join('<br/>'),
        lineStyle: { color: signColor(e.sign), width: 1.6, curveness: 0.12 },
      })),
    }],
  });
  return chart;
}

// points: [{x, y, name?}]
export function scatter(el, { points, xName = '', yName = '', fmtX, fmtY, diagonal = false } = {}) {
  const chart = mount(el);
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const lo = Math.min(...xs, ...ys), hi = Math.max(...xs, ...ys);
  chart.setOption({
    tooltip: { formatter: (p) => `${p.data[2] || ''}<br/>${xName}: ${fmtValue(p.data[0], fmtX)}<br/>${yName}: ${fmtValue(p.data[1], fmtY)}` },
    grid: baseGrid({ bottom: 44 }),
    xAxis: { type: 'value', name: xName, nameLocation: 'middle', nameGap: 26, scale: true, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, fmtX) } },
    yAxis: { type: 'value', name: yName, scale: true, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, formatter: (v) => fmtValue(v, fmtY) } },
    series: [{ type: 'scatter', symbolSize: 7, data: points.map(p => [p.x, p.y, p.name]), itemStyle: { color: '#3b5bdb', opacity: 0.7 } },
      ...(diagonal ? [{ type: 'line', data: [[lo, lo], [hi, hi]], symbol: 'none', lineStyle: { color: INK3, type: 'dashed' } }] : [])],
  });
  return chart;
}
