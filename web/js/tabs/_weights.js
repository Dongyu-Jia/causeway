// Shared inputs for the 组合评估 and 提取与破产概率 tabs (a helper module, not a tab).
//   weightsEditor  13 asset weights: sliders + number inputs, presets, normalization, remembered in localStorage
//   probEditor     per-scenario probability sliders, renormalized, reset to the model posterior
//   customPicker   which saved custom scenarios to include
//   setPicker      which scenario set the paths come from: the monthly engine's 13 built-ins, the quarterly stack's
//                  (step_months = 3, 试运行 until the switch gate T-S passes) or an agent-tree export; the default and
//                  the gate's reasons come from GET /api/portfolio/sets (portfolio/service.py sets_report)
// plus a floating status pill, a debounced / coalescing request runner, a range chart and
// formatting helpers. What the two tabs share for this page load lives in `shared`.
import * as ui from '../ui.js';
import * as charts from '../charts.js';
import * as api from '../api.js';
import { switchNotice } from './_switch.js';

const STORE_KEY = 'msl-weights';
const INK2 = '#4a5468', INK3 = '#7a8396', LINE = '#e3e6ec';
// used only when meta.json is missing (results not built), so the editor still reads in Chinese
const FALLBACK_NAMES = { us_equity: '美股大盘', nasdaq: '纳斯达克100', energy: '石油能源股', industrials: '工业制造股',
  staples: '必需消费股', healthcare: '医疗保健股', utilities: '公用事业股', financials: '金融股', real_estate: '房地产 REITs',
  gold: '黄金', crude_oil: '原油', treasury_10y: '10年期美债', cash: '现金' };
const FALLBACK_ASSETS = Object.keys(FALLBACK_NAMES);
export const assetLabel = (id) => ui.META.assets[id]?.name || FALLBACK_NAMES[id] || id;
export const FALLBACK_PRESETS = [{ id: 'classic_60_40', name: '经典股债 60/40',
  weights: { us_equity: 0.6, treasury_10y: 0.4 }, description: '60% 美股大盘 + 40% 10 年期美债。' }];
export const DEFAULT_CUSTOM_PROB = 0.05;     // portfolio/service.py: prior of a custom scenario without one

// weights: {asset: percent}; probs: {scenario id: relative weight in percent} or null (= model posterior);
// probMode: which probability set the editor shows — 'posterior' (probs null), 'calibrated' (the set calibrated
// to the one-year forecast, forecast.json live.scenario_probs_calibrated) or 'manual';
// include: saved custom scenario ids to evaluate with the built-ins; set: the scenario set id (null until the sets
// list has loaded: the server's default, i.e. the switch's).
export const shared = { weights: null, probs: null, probMode: 'posterior', include: [], set: null };

// ---------------------------------------------------------------- formatting

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// "AI 泡沫破裂：2000-02 年式" -> "AI 泡沫破裂" for chart axes; tables keep the full name.
export function shortName(name) {
  const s = String(name ?? '');
  const i = s.search(/[：:]/);
  return (i > 0 ? s.slice(0, i) : s).trim() || s;
}

// Probabilities that are often tiny (ruin): keep two decimals below 1%.
export function fmtProb(v) {
  if (!isNum(v)) return '—';
  if (v <= 0) return '0%';
  if (v < 0.0005) return '<0.05%';
  return `${(v * 100).toFixed(v < 0.01 ? 2 : 1)}%`;
}
export const fmtMult = (v) => (isNum(v) ? `${v.toFixed(2)} 倍` : '—');
export function fmtShare(w) {
  if (!isNum(w)) return '—';
  const p = w * 100;
  return `${Math.abs(p - Math.round(p)) < 0.05 ? p.toFixed(0) : p.toFixed(1)}%`;
}
export function weightsLabel(weights) {
  return Object.entries(weights || {}).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1])
    .map(([id, w]) => `${assetLabel(id)} ${fmtShare(w)}`).join(' + ') || '—';
}

// The service's notes name a few response fields; say them the way the page does.
const NOTE_TERMS = [[/（可用 scenario_probs 修改）/g, '（可在下方“情景概率”里修改）'], [/scenario_probs/g, '情景概率'],
  [/max_dd_p95 ?(是回撤的坏尾部)?/g, '回撤坏尾部'], [/cvar5_terminal/g, 'CVaR 5%'], [/terminal_real ?/g, '期末实际财富'],
  [/\breal\b ?/g, '“实际”']];
export const plainNote = (text) => NOTE_TERMS.reduce((s, [re, to]) => s.replace(re, to), String(text ?? ''));

// Colour class for a value shown with `d` decimals as a percent: one that rounds to 0.0% stays neutral.
export const signOf = (v, d = 1) => (isNum(v) && Math.abs(v) >= 0.5 * 10 ** -(d + 2) ? (v > 0 ? 'pos' : 'neg') : '');

// Error notice with a retry button: an unchanged input would otherwise never be sent again.
export function errorNotice(err, retry) {
  const n = ui.notice(errorText(err), 'error');
  if (retry) n.append(' ', ui.h('button', { class: 'btn weights-retry', onclick: retry }, '重试'));
  return n;
}

// Round axis bounds and a step for a value axis spanning [lo, hi] (decimals), ~4-8 ticks.
export function niceAxis(lo, hi) {
  const span = Math.max(hi - lo, 1e-4);
  const step = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1].find(s => span / s <= 7) || 0.1;
  const min = Math.floor((lo - span * 0.04) / step + 1e-9) * step;
  return { min: Math.max(0, +min.toFixed(6)), max: +(Math.ceil((hi + span * 0.04) / step - 1e-9) * step).toFixed(6), interval: step };
}

// What to tell the user for an API error (404 = not built, 400 = bad input, 503 = paths not readable).
export function errorText(err) {
  if (!err) return '';
  const msg = err.message || String(err);
  const hint = /scenariolab\.build/.test(msg) ? '' : '在项目根目录运行 python -m scenariolab.build 生成结果后刷新页面。';
  if (err.status === 404) return `结果文件还没有生成：${msg}。${hint}`;
  if (err.status === 400) return `参数有误：${msg}`;
  if (err.status === 503) return `情景路径暂时不可用：${msg}`;
  if (!err.status) return `无法连接本地服务（${msg}）：请确认 python -m scenariolab.server 仍在运行。`;
  return `计算失败（${err.status}）：${msg}`;
}

// ---------------------------------------------------------------- requests and status

export function debounce(fn, ms = 400) {
  let t = null;
  const d = (...args) => { clearTimeout(t); t = setTimeout(() => { t = null; fn(...args); }, ms); };
  d.flush = (...args) => { clearTimeout(t); t = null; fn(...args); };
  d.cancel = () => { clearTimeout(t); t = null; };
  return d;
}

// At most one request in flight: inputs that change meanwhile are sent once it returns,
// and the stale result is dropped, so the previous result stays on screen until the latest one lands.
export function liveRequest(run, { start, done, error } = {}) {
  let busy = false, next = null;
  async function fire(params) {
    if (busy) { next = params; return; }
    busy = true;
    if (start) start();
    let res = null, err = null;
    try { res = await run(params); } catch (e) { err = e; }
    busy = false;
    if (next !== null) { const p = next; next = null; fire(p); return; }
    if (err) { if (error) error(err); } else if (done) done(res);
  }
  return fire;
}

// Floating pill at the bottom right: visible wherever the user has scrolled to.
export function statusPill() {
  const { h } = ui;
  const el = h('div', { class: 'weights-pill', role: 'status', 'aria-live': 'polite' });
  let timer = null;
  const show = (cls, nodes, ms) => {
    clearTimeout(timer);
    el.className = `weights-pill show ${cls}`;
    el.replaceChildren(...nodes);
    if (ms) timer = setTimeout(() => { el.className = 'weights-pill'; }, ms);
  };
  return {
    el,
    busy: (text = '重新计算中…') => show('', [h('span', { class: 'spinner' }), text]),
    done: (text) => show('ok', [text], 1800),
    fail: (text) => show('err', [text], 6000),
  };
}

// Route parameter after "?": "income&rate=0.035" -> {tokens: ['income'], kv: {rate: '0.035'}}.
export function parseParam(param) {
  const out = { tokens: [], kv: {} };
  for (const part of String(param || '').split('&')) {
    const s = part.trim();
    if (!s) continue;
    const i = s.indexOf('=');
    if (i < 0) out.tokens.push(s);
    else out.kv[s.slice(0, i).trim()] = s.slice(i + 1).trim();
  }
  return out;
}
// "us_equity:60,treasury_10y:40" -> {us_equity: 60, treasury_10y: 40}
export function parseWeightsParam(text) {
  const out = {};
  for (const part of String(text || '').split(/[,;]/)) {
    const [k, v] = part.split(':');
    const x = Number(v);
    if (k && k.trim() && Number.isFinite(x) && x >= 0) out[k.trim()] = x;
  }
  return out;
}

// Rows of the probability editor: built-ins from scenarios.json (or, for another scenario set, that set's own list
// and default probabilities, with scenarios.json's posterior beside them), then the included custom ones.
export function scenarioRows(scen, customs = [], set = null) {
  const own = set && set.id !== 'monthly' && Array.isArray(set.scenarios) && set.scenarios.length ? set.scenarios : null;
  const post = Object.fromEntries((scen?.scenarios || []).map(s => [s.id, s]));
  const builtin = own
    ? own.map(s => ({ id: s.id, name: s.name || post[s.id]?.name || s.id, prior: isNum(s.prior_prob) ? s.prior_prob : post[s.id]?.prior_prob,
      likelihood: post[s.id]?.likelihood, post: post[s.id]?.prob, def: isNum(s.prob) ? s.prob : 0 }))
    : (scen?.scenarios || []).map(s => ({ id: s.id, name: s.name || s.id, prior: s.prior_prob,
      likelihood: s.likelihood, post: s.prob, def: isNum(s.prob) ? s.prob : (isNum(s.prior_prob) ? s.prior_prob : 0) }));
  const taken = new Set(builtin.map(r => r.id));
  const extra = shared.include.map(cid => {
    const c = customs.find(x => x.id === cid) || { id: cid, name: cid };
    const prior = isNum(c.prob) ? c.prob : isNum(c.prior_prob) ? c.prior_prob : DEFAULT_CUSTOM_PROB;
    // the service renames a custom scenario that collides with a built-in id to "custom:<id>"
    return { id: taken.has(cid) ? `custom:${cid}` : cid, name: c.name || cid, prior, likelihood: null, post: null,
      def: prior, custom: true };
  });
  return [...builtin, ...extra];
}

// ---------------------------------------------------------------- chart helpers

export function labelWidth(el, frac = 0.3, min = 64, max = 170) {
  return Math.round(Math.max(min, Math.min(max, (el.clientWidth || 600) * frac)));
}
// Height of a chart with one row per scenario: ~26 px a row, so saved custom scenarios never squeeze the labels.
export const rowsHeight = (n, min = 420) => Math.max(min, 64 + n * 26);

// 0 = best (green) ... n-1 = worst (red), grey in between: the same colour for a scenario in every chart.
export function rankColor(i, n) {
  const t = n > 1 ? i / (n - 1) : 0.5;
  const stops = [[26, 156, 91], [134, 142, 150], [217, 54, 62]];
  const [a, b, u] = t < 0.5 ? [stops[0], stops[1], t * 2] : [stops[1], stops[2], (t - 0.5) * 2];
  return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * u)).join(',')})`;
}

// Horizontal range plot: one row per item, a band from lo to hi and a dot at mid.
// rows: [{label, lo, mid, hi, color?, size?, tip?}]; marks: [{x, label, color?, type?}] vertical lines.
export function rangeChart(el, { rows, fmt, marks = [], height } = {}) {
  if (height) el.style.height = `${height}px`;
  const chart = charts.mount(el);
  const val = (v) => (isNum(v) ? v : '-');
  chart.setOption({
    tooltip: { trigger: 'item', formatter: (p) => rows[p.dataIndex]?.tip || esc(rows[p.dataIndex]?.label) },
    grid: { left: 8, right: 28, top: 22, bottom: 8, containLabel: true },
    xAxis: { type: 'value', axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false },
      splitLine: { lineStyle: { color: '#eef0f4' } }, splitNumber: (el.clientWidth || 600) < 480 ? 4 : 6,
      axisLabel: { color: INK3, fontSize: 11, hideOverlap: true, formatter: (v) => charts.fmtValue(v, fmt) } },
    yAxis: { type: 'category', inverse: true, data: rows.map(r => r.label), axisTick: { show: false },
      axisLine: { lineStyle: { color: LINE } },
      axisLabel: { color: INK2, fontSize: 11, interval: 0, width: labelWidth(el), overflow: 'truncate' } },
    series: [
      { type: 'custom', data: rows.map((r, i) => [i, val(r.lo), val(r.hi)]), encode: { x: [1, 2], y: 0 },
        renderItem: (params, a) => {
          const lo = a.value(1), hi = a.value(2);
          if (!isNum(lo) || !isNum(hi)) return null;
          const p = a.coord([lo, a.value(0)]), q = a.coord([hi, a.value(0)]);
          return { type: 'line', shape: { x1: p[0], y1: p[1], x2: q[0], y2: q[1] },
            style: { stroke: rows[params.dataIndex]?.color || '#3b5bdb', lineWidth: 7, opacity: 0.25, lineCap: 'round' } };
        } },
      { type: 'scatter', z: 3,
        data: rows.map((r, i) => ({ value: [val(r.mid), i], symbolSize: r.size || 10,
          itemStyle: { color: r.color || '#3b5bdb', borderColor: '#fff', borderWidth: 1 } })),
        markLine: marks.length ? { silent: true, symbol: 'none', animation: false, precision: 10,
          // precision: ECharts rounds a mark line's position to 2 decimals by default (0.0213 -> 0.02);
          // the category axis is inverted, so the line's start is at the top of the plot
          data: marks.filter(m => isNum(m.x)).map(m => ({ xAxis: m.x,
            lineStyle: { color: m.color || INK3, type: m.type || 'dashed', width: 1 },
            label: { formatter: m.label || '', color: m.color || INK3, fontSize: 10, position: 'start' } })) } : undefined },
    ],
  });
  return chart;
}

// ---------------------------------------------------------------- weights editor

function readStore() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch (_) { return null; }
}
function writeStore(w) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(w)); } catch (_) { /* storage unavailable: keep in memory */ }
}

// {id: number} -> {id: value >= 0} over `ids` (unknown ids dropped), or null when nothing is positive.
function clean(obj, ids, cap = 100) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return null;
  const out = {};
  let s = 0;
  for (const id of ids) {
    const v = Number(obj[id]);
    out[id] = Number.isFinite(v) && v > 0 ? Math.min(cap, v) : 0;
    s += out[id];
  }
  return s > 0 ? out : null;
}
// Any positive map -> percents summing to exactly 100 (0.1 steps; the rounding residual goes to the largest).
function toPercent(obj, ids) {
  const c = clean(obj, ids, Infinity);
  if (!c) return null;
  const s = ids.reduce((t, id) => t + c[id], 0);
  const out = Object.fromEntries(ids.map(id => [id, Math.round((c[id] / s) * 1000) / 10]));
  const diff = Math.round((100 - ids.reduce((t, id) => t + out[id], 0)) * 10) / 10;
  if (diff) {
    const top = ids.reduce((a, b) => (out[b] > out[a] ? b : a));
    out[top] = Math.round((out[top] + diff) * 10) / 10;
  }
  return out;
}
const fmtIn = (v) => (Number.isInteger(v) ? String(v) : String(Math.round(v * 10) / 10));

// Card with one slider + number input per asset. onChange(weights) fires ~`delay` ms after the last edit
// with normalized decimal weights (only assets > 0); presets fire at once.
export function weightsEditor({ assets, presets, initial, onChange, delay = 400, cls = '' } = {}) {
  injectStyle();
  const { h } = ui;
  const list = (assets && assets.length ? assets : FALLBACK_ASSETS.map(id => ({ id })))
    .map(a => ({ id: a.id, name: a.name || assetLabel(a.id), proxy: a.proxy || '' }));
  const ids = list.map(a => a.id);
  const w = toPercent(initial, ids) || clean(shared.weights, ids) || clean(readStore(), ids)
    || toPercent(FALLBACK_PRESETS[0].weights, ids) || Object.fromEntries(ids.map((id, i) => [id, i ? 0 : 100]));
  let presetList = presets && presets.length ? presets : FALLBACK_PRESETS;

  const total = () => ids.reduce((s, id) => s + w[id], 0);
  function get() {
    const s = total();
    if (!(s > 0)) return null;
    const out = {};
    for (const id of ids) if (w[id] > 0) out[id] = w[id] / s;
    return out;
  }

  const sumChip = h('span', { class: 'chip' });
  const bar = h('div', { class: 'weights-bar', 'aria-hidden': 'true' });
  const presetBox = h('div', { class: 'weights-presets' });
  const grid = h('div', { class: 'weights-grid' });
  const rows = {};
  for (const a of list) {
    const slider = h('input', { type: 'range', min: 0, max: 100, step: 0.5, value: String(w[a.id]), 'aria-label': `${a.name}权重` });
    const input = h('input', { type: 'number', min: 0, max: 100, step: 'any', value: fmtIn(w[a.id]), 'aria-label': `${a.name}权重（%）` });
    const share = h('span', { class: 'weights-share' });
    const row = h('div', { class: 'weights-row' },
      h('span', { class: 'weights-dot', style: { background: charts.assetColor(a.id) } }),
      h('span', { class: 'weights-name', title: a.proxy ? `${a.name}：${a.proxy}` : a.name }, a.name),
      slider, h('span', { class: 'weights-in' }, input, h('span', { class: 'muted' }, '%')), share);
    slider.addEventListener('input', () => { w[a.id] = Number(slider.value); input.value = fmtIn(w[a.id]); changed(); });
    input.addEventListener('input', () => {
      const v = input.value === '' ? 0 : Number(input.value);
      if (!Number.isFinite(v)) return;
      w[a.id] = Math.min(100, Math.max(0, v));
      slider.value = String(w[a.id]);
      changed();
    });
    input.addEventListener('change', () => { input.value = fmtIn(w[a.id]); });
    rows[a.id] = { row, slider, input, share };
    grid.append(row);
  }

  // Re-evaluate only when the weights moved by more than the 0.1% display rounding since the last
  // evaluation, so 归一化 (which only rewrites the inputs) does not trigger a request.
  let lastSent = get();
  const near = (a, b) => !!a && !!b && ids.every(id => Math.abs((a[id] || 0) - (b[id] || 0)) < 6e-4);
  const fire = debounce(() => {
    const n = get();
    if (!n || near(n, lastSent)) return;
    lastSent = n;
    if (onChange) onChange(n);
  }, delay);
  function changed(now = false) {
    paint();
    shared.weights = { ...w };
    writeStore(w);
    if (now) fire.flush(); else fire();
  }
  const sync = (id) => { rows[id].slider.value = String(w[id]); rows[id].input.value = fmtIn(w[id]); };

  function paint() {
    const s = total();
    for (const a of list) {
      rows[a.id].share.textContent = s > 0 ? fmtShare(w[a.id] / s) : '—';
      rows[a.id].row.classList.toggle('zero', !(w[a.id] > 0));
    }
    bar.replaceChildren(...(s > 0 ? list.filter(a => w[a.id] > 0).map(a => h('span', {
      style: { width: `${(w[a.id] / s) * 100}%`, background: charts.assetColor(a.id) },
      title: `${a.name} ${fmtShare(w[a.id] / s)}` })) : []));
    const ok = Math.abs(s - 100) < 0.05;
    sumChip.className = `chip ${s <= 0 ? 'bad' : ok ? 'good' : 'warn'}`;
    sumChip.textContent = s <= 0 ? '合计 0%：请至少给一个资产设权重'
      : ok ? '合计 100%' : `合计 ${fmtIn(Math.round(s * 10) / 10)}%，按比例归一化`;
    const n = get();
    for (const b of presetBox.querySelectorAll('button')) {
      b.classList.toggle('on', !!n && !!b._w && ids.every(id => Math.abs((n[id] || 0) - (b._w[id] || 0)) < 5e-4));
    }
  }

  function presetTip(p) {
    const lines = [p.description || p.name];
    const pf = p.portfolio;
    if (pf) lines.push(`加权实际年化回报中位数 ${ui.fmtPct(pf.real_cagr_p50, 1)} · 最大回撤中位数 ${ui.fmtPct(pf.max_dd_p50, 1)}`);
    // safe rates per horizon (portfolio_default.json withdrawal_by_horizon: 15 and 30 years); safe_rate_weighted
    // alone is at withdrawal_horizon_years (15), not at the withdrawal page's default of 30
    const by = p.withdrawal_by_horizon || {};
    const hz = Object.keys(by).map(Number).filter(y => isNum(by[y]?.safe_rate_weighted?.pro_rata)).sort((a, b) => a - b);
    if (hz.length) lines.push(`安全提取率（按比例提取）${hz.map(y => `${y} 年 ${ui.fmtPct(by[y].safe_rate_weighted.pro_rata, 2)}`).join(' · ')}`);
    else if (p.safe_rate_weighted) lines.push(`安全提取率（按比例提取，${p.withdrawal_horizon_years || 15} 年）${ui.fmtPct(p.safe_rate_weighted.pro_rata, 2)}`);
    return lines.join('\n');
  }
  function renderPresets() {
    presetBox.replaceChildren(h('span', { class: 'weights-presets-label' }, '预设'),
      ...presetList.map(p => {
        const b = h('button', { class: 'btn', title: presetTip(p), onclick: () => set(p.weights) }, p.name);
        const c = clean(p.weights, ids, Infinity);
        if (c) { const s = ids.reduce((t, id) => t + c[id], 0); b._w = Object.fromEntries(ids.map(id => [id, c[id] / s])); }
        return b;
      }));
  }

  function set(obj, { now = true } = {}) {
    const p = toPercent(obj, ids);
    if (!p) return;
    Object.assign(w, p);
    ids.forEach(sync);
    changed(now);
  }
  const normBtn = h('button', { class: 'btn', title: '把输入框改写成合计 100% 的百分比（结果不变）',
    onclick: () => { const p = toPercent(w, ids); if (p) { Object.assign(w, p); ids.forEach(sync); changed(); } } }, '归一化');
  const clearBtn = h('button', { class: 'btn', title: '全部清零后再逐个设置',
    onclick: () => { ids.forEach(id => { w[id] = 0; sync(id); }); changed(); } }, '清零');

  renderPresets();
  paint();
  shared.weights = { ...w };
  const el = ui.card({ title: '组合权重', cls: `weights-card ${cls}`,
    sub: `${list.length} 类资产的目标权重（%）。合计不必正好是 100%，计算时按比例归一化；停手约 0.4 秒后自动重算，并记住上次的组合。`,
    actions: [sumChip, normBtn, clearBtn] }, presetBox, bar, grid);
  return {
    el, get, set,
    setPresets(next) { presetList = next && next.length ? next : FALLBACK_PRESETS; renderPresets(); paint(); },
  };
}

// ---------------------------------------------------------------- scenario probability editor

// The probability set calibrated to the scored one-year forecast (forecast.json): what the editor offers as
// 「按一年期预测校准」. {available, probs, kl, ens, ens_prior, text, reason, prior}.
function calibrationOf(fc) {
  const c = fc?.live?.calibration, p = fc?.live?.scenario_probs_calibrated;
  if (!c) return { available: false, reason: '这份一年期预测结果（forecast.json）还没有校准概率：运行 python -m scenariolab.build --only forecast' };
  return { available: !!c.available && !!p && Object.keys(p).length > 0, probs: p || {}, kl: c.kl, ens: c.ens,
    ens_prior: c.ens_prior, text: c.text || '', reason: c.reason || '', prior: c.prior || fc.live.scenario_probs || {} };
}

// Sliders are relative weights (percent of the model posterior's scale); the share actually used is
// each weight divided by their sum. Three sets to start from: the model posterior (no scenario_probs sent
// at all), the set calibrated to the one-year forecast (built-ins only; included custom scenarios keep their
// prior weight) and hand-set weights (any slider or 等权). Changes fire onChange ~`delay` ms later.
export function probEditor({ onChange, delay = 400 } = {}) {
  injectStyle();
  const { h } = ui;
  let rows = [];
  let used = null;                    // {id: prob} the last response used
  let cal = null;                     // calibrationOf(forecast.json), once loaded
  // what the set's defaults are: the monthly set's model posterior, or a quarterly set's hand-set priors (§7.5)
  let basis = 'posterior';
  const basisWord = () => (basis === 'prior' ? '手设先验' : '模型后验');
  const chip = h('span', { class: 'chip' });
  const list = h('div', { class: 'weights-prob' });
  const foot = h('div', { class: 'weights-prob-foot' });
  const details = h('div');
  const intro = h('p', { class: 'weights-prob-intro' });
  const calLine = h('p', { class: 'weights-prob-cal' }, '正在读取一年期预测的校准概率…');
  const cur = (r) => { const v = shared.probs?.[r.id]; return isNum(v) ? v : r.def * 100; };
  const total = () => rows.reduce((s, r) => s + cur(r), 0);
  const adjusted = () => !!shared.probs && rows.some(r => Math.abs(cur(r) - r.def * 100) > 1e-9);
  const mode = () => (!shared.probs ? 'posterior' : shared.probMode === 'calibrated' ? 'calibrated' : 'manual');
  const fire = debounce(() => { if (total() > 0 && onChange) onChange(); }, delay);
  // stale: forecast.json was built on other posterior probabilities than the scenarios.json rows show
  const stale = () => !!cal?.available && rows.some(r => !r.custom && isNum(r.post) && isNum(cal.prior?.[r.id])
    && Math.abs(r.post - cal.prior[r.id]) > 1e-4);

  // Segmented choice of the probability set (ui.seg cannot be repainted from outside).
  const modeBox = h('div', { class: 'seg weights-mode', role: 'group', 'aria-label': '情景概率的来源' });
  const MODES = [{ value: 'posterior', label: '模型后验' }, { value: 'calibrated', label: '按一年期预测校准' }, { value: 'manual', label: '手动' }];
  function paintMode() {
    modeBox.replaceChildren(...MODES.map(o => h('button', { type: 'button', class: o.value === mode() ? 'on' : '',
      disabled: o.value === 'calibrated' && !cal?.available && mode() !== 'calibrated' ? true : null,
      title: o.value === 'calibrated' ? (cal?.available ? cal.text : (cal?.reason || '正在读取一年期预测…')) : null,
      onclick: () => choose(o.value) }, o.value === 'posterior' ? basisWord() : o.label)));
  }
  function choose(v) {
    if (!rows.length) return;          // nothing to weight yet (scenarios.json missing)
    if (v === 'posterior') { shared.probs = null; shared.probMode = 'posterior'; render(); fire.flush(); return; }
    if (v === 'calibrated') {
      if (!cal?.available) { paint(); return; }
      shared.probs = Object.fromEntries(rows.map(r => [r.id,
        !r.custom && isNum(cal.probs[r.id]) ? cal.probs[r.id] * 100 : r.def * 100]));
      shared.probMode = 'calibrated';
      render();
      fire.flush();
      return;
    }
    if (!shared.probs) shared.probs = Object.fromEntries(rows.map(r => [r.id, cur(r)]));
    shared.probMode = 'manual';        // same weights as before: nothing to recompute
    render();
  }

  function paintCal() {
    if (!cal) { calLine.textContent = '正在读取一年期预测的校准概率…'; return; }
    const why = String(cal.reason || '');
    calLine.replaceChildren(...[h('b', {}, '按一年期预测校准：'),
      cal.available ? cal.text : `暂不可用。${why}${why && !/[。！？.]$/.test(why) ? '。' : ''}`,
      stale() ? h('span', { class: 'weights-prob-stale' },
        ' 注意：一年期预测（forecast.json）与情景概率（scenarios.json）不是同一次构建，校准概率可能已过期：运行 python -m scenariolab.build --only forecast。') : null]
      .filter(x => x !== null));
  }

  function paint() {
    paintMode();
    paintCal();
    if (!rows.length) {
      chip.className = 'chip';
      chip.textContent = '暂无情景';
      foot.replaceChildren();
      return;
    }
    const s = total();
    const m = mode();
    for (const r of rows) {
      const share = adjusted() || !used || !isNum(used[r.id]) ? (s > 0 ? cur(r) / s : null) : used[r.id];
      if (r._share) r._share.textContent = ui.fmtPct(share, 1);
      if (r._raw) r._raw.textContent = fmtIn(Math.round(cur(r) * 10) / 10);
    }
    chip.className = `chip ${s <= 0 ? 'bad' : m === 'calibrated' ? 'good' : adjusted() ? 'warn' : 'accent'}`;
    chip.textContent = s <= 0 ? '概率不能全为 0' : m === 'calibrated' ? '按一年期预测校准'
      : adjusted() ? '已手动调整' : m === 'manual' ? '手动（尚未改动）' : basisWord();
    const custom = rows.some(r => r.custom);
    foot.replaceChildren(h('span', { class: 'muted' }, m === 'calibrated'
      ? `当前使用按一年期预测校准的概率（KL ${isNum(cal?.kl) ? cal.kl.toFixed(3) : '—'}）${custom ? '；自定义情景保持它的先验权重' : ''}，计算时按比例归一化为 100%。`
      : adjusted()
        ? `相对权重合计 ${fmtIn(Math.round(s * 10) / 10)}，计算时按比例归一化为 100%。`
        : `当前使用${basisWord()}概率（已归一化）。`),
      h('span', { class: 'muted' }, `${rows.length} 个情景`));
  }

  function render() {
    if (!rows.length) {
      list.replaceChildren(ui.notice('还没有情景列表（scenarios.json）：运行 python -m scenariolab.build 生成情景后，这里可以调整每个情景的概率。', 'info'));
      paint();
      return;
    }
    const head = h('div', { class: 'weights-prob-row head' },
      h('span', {}, '情景'), h('span', { class: 'opt' }, '主观先验'), h('span', { class: 'opt' }, '似然'),
      h('span', {}, '模型后验'), h('span', { class: 'opt', title: '按一年期预测校准的概率' }, '一年期校准'),
      h('span', {}, '你的相对权重'), h('span', {}, '计算所用'));
    list.replaceChildren(head, ...rows.map(r => {
      const slider = h('input', { type: 'range', min: 0, max: 50, step: 0.5, value: String(Math.min(50, cur(r))),
        'aria-label': `${r.name}的相对权重` });
      slider.addEventListener('input', () => {
        if (!shared.probs) shared.probs = Object.fromEntries(rows.map(x => [x.id, cur(x)]));
        shared.probs[r.id] = Number(slider.value);
        shared.probMode = 'manual';
        paint();
        fire();
      });
      r._share = h('span', { class: 'share' });
      r._raw = h('span', { class: 'raw' });
      const c = !r.custom && cal?.available ? cal.probs[r.id] : null;
      return h('div', { class: 'weights-prob-row' },
        h('span', { class: 'name', title: r.name }, r.custom ? ui.chip('自定义', 'accent') : null, ' ', r.name),
        h('span', { class: 'opt' }, ui.fmtPct(r.prior, 1)),
        h('span', { class: 'opt' }, isNum(r.likelihood) ? r.likelihood.toFixed(2) : '—'),
        h('span', {}, r.custom ? '—' : ui.fmtPct(r.post, 1)),
        h('span', { class: 'opt' }, isNum(c) ? ui.fmtPct(c, 1) : '—'),
        h('span', { class: 'slide' }, slider, r._raw),
        r._share);
    }));
    paint();
  }
  const setAll = (fn) => {
    if (!rows.length) return;          // nothing to weight yet (scenarios.json missing)
    shared.probs = Object.fromEntries(rows.map(r => [r.id, fn(r)]));
    shared.probMode = 'manual';
    render();
    fire.flush();
  };
  const equalBtn = h('button', { class: 'btn', title: '每个情景同样重要', onclick: () => setAll(() => 10) }, '等权');
  const resetBtn = h('button', { class: 'btn', onclick: () => choose('posterior') }, '重置为模型后验');

  api.get('forecast').then((fc) => { cal = calibrationOf(fc); render(); }, (err) => {
    cal = { available: false, reason: err?.status === 404
      ? '还没有一年期预测结果（forecast.json）：运行 python -m scenariolab.build --only forecast'
      : `无法读取一年期预测：${err?.message || err}` };
    paint();
  });

  const el = ui.card({ title: '情景概率', cls: 'span-all weights-anchor',
    sub: '加权汇总用的权重：每个情景的路径按这里的概率混合。可以从模型后验或按一年期预测校准的概率出发，也可以拖动滑块表达你自己的判断。',
    actions: [chip, equalBtn, resetBtn] }, h('div', { class: 'weights-mode-row' }, modeBox), calLine, intro, list, foot, details);
  return {
    el,
    // rows: scenarioRows(); info: {method, gamma} from scenarios.json
    // info: {method, gamma} from scenarios.json; basis 'prior' when the set's defaults are hand-set priors
    setRows(next, info = {}) {
      rows = next || [];
      basis = info.basis === 'prior' ? 'prior' : 'posterior';
      resetBtn.textContent = `重置为${basisWord()}`;
      const g = isNum(info.gamma) ? info.gamma : null;
      intro.textContent = basis === 'prior'
        ? '这个情景集默认用手设先验：每个情景的主观先验，归一化到 100%（按今天状态的后验倾斜默认关闭）。'
          + '“模型后验”一栏是月度情景集的默认，只作对照。滑块是相对权重：调整后所有情景按比例重新归一化，组合和提取的加权结果随之更新；'
          + '自定义情景默认用它的先验。'
        : `模型后验 = 主观先验 × 似然${g !== null ? `^${g}` : '^γ'}，再归一化到 100%。先验是对情景本身的判断；`
        + '似然衡量今天的宏观状态（通胀、增长、失业、利率、利差、油价）与情景开局有多吻合'
        + `${g !== null && g < 1 ? `，γ = ${g} < 1 让数据只能微调、不能推翻先验` : ''}。`
        + '滑块是相对权重：调整后所有情景按比例重新归一化，组合和提取的加权结果随之更新；自定义情景默认用它的先验。';
      details.replaceChildren(...(info.method ? [h('details', { class: 'weights-details' },
        h('summary', {}, '概率的完整算法'), h('p', {}, info.method))] : []));
      used = null;
      render();
    },
    overrides() {
      if (!shared.probs || !rows.length || !(total() > 0)) return null;
      return Object.fromEntries(rows.map(r => [r.id, cur(r) / 100]));
    },
    sync(perScenario) { used = Object.fromEntries((perScenario || []).map(e => [e.id, e.prob])); paint(); },
    adjusted,
    // 'posterior' | 'calibrated' | 'manual', and the words for it (for a tab's own status line)
    mode,
    label: () => (mode() === 'calibrated' ? `按一年期预测校准（KL ${isNum(cal?.kl) ? cal.kl.toFixed(3) : '—'}）`
      : adjusted() ? '已手动调整（自动归一化）' : basis === 'prior' ? '手设先验（这个情景集的默认）' : '模型后验（主观先验 × 似然）'),
    hasRows: () => rows.length > 0,
  };
}

// ---------------------------------------------------------------- custom scenario picker

export function customPicker({ onChange } = {}) {
  injectStyle();
  const { h } = ui;
  let items = [];
  const box = h('div', { class: 'weights-custom' }, h('span', { class: 'muted' }, '正在读取已保存的自定义情景…'));
  function paint() {
    if (!items.length) {
      box.replaceChildren(h('span', { class: 'muted' }, '还没有保存的自定义情景：可以在 ',
        h('a', { href: '#lab' }, '情景实验室（LLM）'), ' 里生成并保存，再回到这里纳入评估。'));
      return;
    }
    box.replaceChildren(...items.map(it => {
      const prior = isNum(it.prior_prob) ? it.prior_prob : DEFAULT_CUSTOM_PROB;
      const cb = h('input', { type: 'checkbox', checked: shared.include.includes(it.id) ? true : null });
      cb.addEventListener('change', () => {
        shared.include = cb.checked ? [...new Set([...shared.include, it.id])] : shared.include.filter(x => x !== it.id);
        if (onChange) onChange(shared.include.slice());
      });
      return h('label', { class: 'weights-check', title: it.summary || it.name },
        cb, h('span', { class: 'weights-check-name' }, it.name || it.id),
        h('span', { class: 'muted' }, `先验 ${ui.fmtPct(prior, 1)}`));
    }));
  }
  async function load() {
    try {
      const r = await api.get('scenarios/custom', { fresh: true });
      items = Array.isArray(r?.scenarios) ? r.scenarios.filter(s => s && s.id) : [];
    } catch (err) {
      items = [];
      shared.include = [];
      box.replaceChildren(h('span', { class: 'muted' }, `无法读取自定义情景：${err.message}`));
      return items;
    }
    const known = new Set(items.map(i => i.id));
    shared.include = shared.include.filter(id => known.has(id));     // deleted since last visit
    paint();
    return items;
  }
  return { el: box, load, items: () => items };
}

// ---------------------------------------------------------------- scenario set picker

const SET_RE = /^(monthly|quarterly|agent:[A-Za-z0-9_-]{1,80})$/;

// The scenario sets the two tabs can run on (GET /api/portfolio/sets): a segmented choice (an unavailable set is
// disabled, its reason in the tooltip), the switch gate's notice (the default and why) and a line on the chosen set.
// want: a set id from a deep link (#portfolio?set=quarterly), used when that set is available.
export function setPicker({ onChange, want = null } = {}) {
  injectStyle();
  const { h } = ui;
  let rep = null;
  const seg = h('div', { class: 'seg weights-sets', role: 'group', 'aria-label': '情景集' });
  const note = h('div', { class: 'portfolio-help' });
  const sw = h('div');
  const box = h('div', { class: 'weights-setbox' }, seg, note, sw);
  const entry = (id) => (rep?.sets || []).find(x => x.id === id) || null;
  const label = (x) => `${x.name}${x.stub ? '（桩模型）' : x.trial ? '（试运行）' : ''}${x.id === rep?.default ? '（默认）' : ''}`;
  function paint() {
    if (!rep) { seg.replaceChildren(h('span', { class: 'muted' }, '正在读取情景集…')); return; }
    seg.replaceChildren(...rep.sets.map(x => h('button', { type: 'button', class: x.id === shared.set ? 'on' : '',
      disabled: x.available ? null : true, title: x.available ? (x.text || x.name) : (x.issue || '还没有生成'),
      onclick: () => { if (shared.set === x.id || !x.available) return; shared.set = x.id; paint(); if (onChange) onChange(x.id); } },
    label(x))));
    const cur = entry(shared.set);
    note.replaceChildren(...[cur ? (cur.id === 'monthly'
      ? '每步一个月：月度引擎的 13 个内置情景（与「未来情景」页同一批路径）。'
      : `每步 ${cur.step_months} 个月。${cur.text || cur.name}`) : '',
    cur && cur.step_months > 1 ? ' 波动率按季度收益 × √4 年化，回撤是季末口径，标的（QQQ、BDC 等）需要月度路径，这里不能用。' : '',
    rep.default_note ? ` ${rep.default_note}` : ''].filter(Boolean));
    sw.replaceChildren(switchNotice(rep.switch, { where: '组合与提取的默认情景集', compact: true }));
  }
  async function load() {
    try {
      rep = await api.get('portfolio/sets', { fresh: true });
    } catch (err) {
      rep = { default: 'monthly', sets: [{ id: 'monthly', name: '13 个内置情景 · 月度引擎', available: true, step_months: 1 }], switch: null };
      note.textContent = `读不出情景集列表（${err.message}）：用月度引擎的情景集。`;
    }
    const ok = (id) => !!entry(id)?.available;
    if (want && SET_RE.test(want) && ok(want)) shared.set = want;
    else if (!shared.set || !ok(shared.set)) shared.set = ok(rep.default) ? rep.default : 'monthly';
    paint();
    return rep;
  }
  paint();
  return { el: box, load, current: () => shared.set, entry: () => entry(shared.set),
    stepMonths: () => entry(shared.set)?.step_months || 1, report: () => rep };
}

// ---------------------------------------------------------------- styles (both tabs)

export function injectStyle() {
  if (document.getElementById('style-fe4')) return;
  const css = `
.weights-presets { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 2px 0 10px; }
.weights-presets-label { font-size: 12px; color: var(--ink-3); margin-right: 2px; }
.weights-presets .btn, .weights-card .card-actions .btn, .weights-anchor .card-actions .btn { font-size: 12px; padding: 3px 10px; }
.weights-presets .btn.on { background: var(--accent-soft); border-color: #cdd7ff; color: var(--accent); font-weight: 600; }
.weights-bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--panel-2); border: 1px solid var(--line); margin-bottom: 10px; }
.weights-bar span { height: 100%; }
.weights-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(274px, 100%), 1fr)); gap: 2px 18px; }
.weights-row { display: grid; grid-template-columns: 10px 80px minmax(40px, 1fr) 72px 40px; gap: 8px; align-items: center; min-height: 32px; font-size: 13px; }
.weights-row.zero .weights-name, .weights-row.zero .weights-share { color: var(--ink-3); }
.weights-dot { width: 10px; height: 10px; border-radius: 3px; }
.weights-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.weights-row input[type=range], .weights-prob-row input[type=range] { width: 100%; min-width: 0; margin: 0; }
.weights-in { display: flex; gap: 4px; align-items: center; }
.weights-in input[type=number] { width: 56px; padding: 3px 6px; text-align: right; font-variant-numeric: tabular-nums; }
.weights-pair { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(400px, 100%), 1fr)); }
@media (max-width: 1100px) { .weights-side { grid-column: 1 / -1; } }
@media (max-width: 760px) { .weights-card .card-head, .weights-results .card-head, .weights-anchor .card-head { flex-wrap: wrap; } }
.weights-share { text-align: right; font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.weights-prob-intro { margin: 0 0 10px; color: var(--ink-2); font-size: 13px; max-width: 980px; }
.weights-prob { display: grid; }
.weights-prob-row { display: grid; grid-template-columns: minmax(150px, 1.5fr) 64px 52px 64px 72px minmax(140px, 2fr) 64px; gap: 12px; align-items: center; min-height: 34px; font-size: 13px; font-variant-numeric: tabular-nums; border-bottom: 1px solid var(--line); }
.weights-mode-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 6px; }
.weights-mode button:disabled { opacity: .5; cursor: not-allowed; }
.weights-prob-cal { margin: 0 0 10px; color: var(--ink-2); font-size: 12px; max-width: 980px; }
.weights-prob-stale { color: var(--warn); }
.weights-prob-row.head { min-height: 28px; font-size: 12px; font-weight: 600; color: var(--ink-3); background: var(--panel-2); }
.weights-prob-row > span:not(:first-child) { text-align: right; }
.weights-prob-row .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.weights-prob-row .slide { display: grid; grid-template-columns: 1fr 38px; gap: 8px; align-items: center; }
.weights-prob-row .raw { font-size: 12px; color: var(--ink-3); }
.weights-prob-row .share { font-weight: 600; }
.weights-prob-foot { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; }
.weights-details { margin-top: 8px; font-size: 12px; color: var(--ink-3); }
.weights-details summary { cursor: pointer; color: var(--ink-2); }
.weights-details p { margin: 6px 0 0; max-width: 980px; }
.weights-custom { display: grid; gap: 2px; font-size: 13px; }
.weights-check { display: flex; gap: 8px; align-items: center; padding: 3px 0; cursor: pointer; }
.weights-check-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.weights-anchor { scroll-margin-top: 84px; }
.weights-card .card-actions, .weights-results .card-actions, .weights-anchor .card-actions { flex-shrink: 0; }
.weights-card .seg button, .weights-results .seg button, .weights-anchor .seg button { white-space: nowrap; }
.weights-err { display: grid; gap: 8px; }
.weights-setbox { display: grid; gap: 6px; min-width: 0; }
.weights-sets { flex-wrap: wrap; max-width: 100%; }
.weights-sets button { white-space: normal !important; text-align: left; }
.weights-sets button:disabled { opacity: .5; cursor: not-allowed; }
.weights-rebal button:disabled { opacity: .5; cursor: not-allowed; }
.weights-err:not(:empty) { margin: 14px 0 2px; }
.weights-results.weights-busy { opacity: .55; }
.weights-pill { position: fixed; right: 24px; bottom: 24px; z-index: 30; display: flex; gap: 8px; align-items: center; background: #121a2b; color: #fff; padding: 8px 14px; border-radius: 999px; font-size: 12px; box-shadow: 0 4px 14px rgba(16, 24, 40, .2); opacity: 0; transform: translateY(6px); transition: opacity .2s, transform .2s; pointer-events: none; max-width: calc(100vw - 48px); }
.weights-pill.show { opacity: 1; transform: none; }
.weights-pill.err { background: #8a1c24; }
.weights-pill .spinner { width: 14px; height: 14px; border-color: rgba(255, 255, 255, .35); border-top-color: #fff; }
.weights-focus td { background: var(--accent-soft) !important; }
@media (max-width: 900px) {
  .weights-prob-row { grid-template-columns: minmax(110px, 1.3fr) 56px minmax(100px, 2fr) 56px; gap: 8px; }
  .weights-prob-row .opt { display: none; }
}
@media (max-width: 480px) {
  .weights-prob-row { grid-template-columns: minmax(0, 1.3fr) 48px minmax(72px, 1.6fr) 46px; gap: 6px; }
}
.portfolio-help { font-size: 12px; color: var(--ink-3); margin: 6px 0 0; }
.portfolio-label { font-size: 12px; font-weight: 600; color: var(--ink-2); margin: 16px 0 6px; }
.portfolio-label:first-child { margin-top: 0; }
.portfolio-summary { font-size: 12px; color: var(--ink-3); margin: -4px 0 10px; }
.portfolio-callout { border-left: 3px solid var(--line); background: var(--panel-2); border-radius: 0 8px 8px 0; padding: 8px 12px; font-size: 13px; margin-top: 10px; }
.portfolio-callout.bad { border-left-color: var(--bad); background: #fdf3f4; }
.portfolio-callout.good { border-left-color: var(--good); background: #eef8f2; }
.portfolio-row-worst td { background: #fdf3f4; }
.portfolio-row-best td { background: #eef8f2; }
.portfolio-table td:first-child { min-width: 220px; }
.portfolio-compare table.tbl th, .portfolio-compare table.tbl td { padding: 7px 7px; }
.withdrawal-form { display: grid; gap: 14px 24px; grid-template-columns: repeat(auto-fit, minmax(min(260px, 100%), 1fr)); align-items: start; }
.withdrawal-field { display: grid; gap: 6px; font-size: 13px; }
.withdrawal-field > .t { font-size: 12px; font-weight: 600; color: var(--ink-2); }
.withdrawal-field > .seg { justify-self: start; }
.withdrawal-kpi-unit { font-size: 12px; font-weight: 500; color: var(--ink-3); margin-left: 6px; }
.weights-retry { font-size: 12px; padding: 1px 10px; margin-left: 6px; }
.withdrawal-inline { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.withdrawal-inline input[type=range] { flex: 1; min-width: 110px; }
.withdrawal-inline input[type=number] { width: 92px; text-align: right; font-variant-numeric: tabular-nums; }
.withdrawal-def { margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--line); color: var(--ink-2); }
.kpi-value.withdrawal-warn, table.tbl td.withdrawal-warn { color: var(--warn); }
.withdrawal-toolbar { position: sticky; top: 63px; z-index: 4; display: flex; gap: 10px 18px; align-items: center; flex-wrap: wrap; background: rgba(244, 245, 248, .95); padding: 10px 0; margin-top: 8px; border-bottom: 1px solid var(--line); font-size: 12px; color: var(--ink-3); }
.withdrawal-toolbar select { max-width: 280px; }
.withdrawal-table td:first-child { min-width: 200px; }
.withdrawal-extremes td:first-child { min-width: 150px; }
`;
  document.head.append(ui.h('style', { id: 'style-fe4' }, css));
}
