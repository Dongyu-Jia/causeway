// Shared renderer for one scenario detail (built-in, LLM-translated or hand-edited spec):
// narrative and causal chain, the phases and the real months they draw, K-lines and fans
// per asset, macro paths, correlations, the income-distribution calculator, probability
// and the validation report. Used by tabs/scenarios.js and tabs/lab.js; the leading
// underscore keeps the router from treating this module as a tab.
//
//   renderScenarioDetail(root, detail, ctx, { state, labLink })
//   detail = GET /api/scenarios/<id> (scenarios/runner.run_spec + library.py extras)
import * as ui from '../ui.js';
import * as charts from '../charts.js';
import * as api from '../api.js';

const { h, card } = ui;
const STYLE_ID = 'style-fe3';

const ERA_COLORS = {
  golden_age: '#5c940d', inflation_onset: '#f08c00', great_inflation: '#e8590c', disinflation: '#1098ad',
  tech_boom: '#7048e8', china_credit: '#c2255c', stagnation_qe: '#3b5bdb', covid_reflation: '#0ca678',
  ai_capex: '#862e9c',
};
const PHASE_COLORS = ['#3b5bdb', '#e8590c', '#2f9e44', '#7048e8', '#1098ad', '#c2255c', '#f08c00', '#5c940d'];
const IDLE_COLOR = '#adb5bd';     // a phase without views draws history unconditionally
const SOURCE_NAMES = { builtin: '内置情景', llm: 'LLM 翻译', custom: '手动编辑' };
const MACRO = [
  { id: 'cpi_yoy', title: 'CPI 同比', unit: '%',
    sub: '中位数与 5–95% 区间；通胀把名义回报折成实际回报，并通过政策利率传导到债券和估值。' },
  { id: 'ff', title: '联邦基金利率', unit: '%',
    sub: '按今天与历史的利率水平比例缩放并向中性利率回归；是“通胀 → 利率 → 资产定价”的中间一环。' },
  { id: 'y10', title: '10 年期美债收益率', unit: '%',
    sub: '决定 10 年期美债的重定价和股票的贴现率；上行时长债和高估值资产承压。' },
  { id: 'unrate', title: '失业率', unit: '%',
    sub: '反映需求和劳动收入冲击；上升的阶段通常伴随盈利下滑和信用利差走阔。' },
  { id: 'oil', title: 'WTI 油价（美元/桶）', unit: '$',
    sub: '供给冲击的入口：经商品通胀传到 CPI，并直接决定能源股和原油的走势。' },
  { id: 'spread', title: '信用利差（BAA − 10 年期）', unit: '%',
    sub: '衡量金融条件；走阔时股票、REITs 和金融股的回撤加深。' },
  // shown only when the engine tracks the valuation state (macro.pd, with its fair value macro.pd_fair)
  { id: 'pd', title: '美股估值 P/D（价格 ÷ 过去 12 个月股息）', unit: '倍', optional: true,
    sub: '估值路径：P/D 越高、股息率越低；灰色虚线是 1973 年以来的中位数，估值向锚回归的快慢决定美股长期回报里“估值变化”那一块。' },
];
const PROB_STATES = { oil_yoy: '油价同比' };   // states the meta list may not name

const CSS = `
.scenarios-detail { scroll-margin-top: 120px; }
.scenarios-head { display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
.scenarios-name { font-size: 20px; font-weight: 650; margin: 0; line-height: 1.35; }
.scenarios-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; align-items: center; }
.scenarios-summary { color: var(--ink-2); margin: 10px 0 0; max-width: 1040px; }
.scenarios-kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)); gap: 10px; margin-top: 14px; }
.scenarios-kpis .kpi { box-shadow: none; background: var(--panel-2); padding: 10px 12px; }
.scenarios-kpis .kpi-value { font-size: 19px; }
.scenarios-kpis4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.scenarios-kpis2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
table.tbl tr.scenarios-vrow.has-why td { border-bottom: 0; padding-bottom: 1px; }
table.tbl tr.scenarios-why td { white-space: normal; text-align: left; color: var(--ink-3); font-size: 12px; padding-top: 0; line-height: 1.5; }
table.tbl tr.scenarios-why:hover td, table.tbl tr.scenarios-vrow:hover td { background: transparent; }
/* view tables sit in a narrower grid column: the name and the request wrap, the numbers do not */
table.tbl.scenarios-vtable td:first-child, table.tbl td.scenarios-wrap { white-space: normal; }
table.tbl.scenarios-vtable td:first-child { min-width: 112px; }
.scenarios-var .mono { overflow-wrap: anywhere; }
@media (max-width: 1100px) { .scenarios-kpis4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.scenarios-detail .card-head > div:first-child, .scenarios-card .card-head > div:first-child { min-width: 0; }
.scenarios-detail .card-actions, .scenarios-card .card-actions { flex: 0 0 auto; }
.scenarios-detail .seg button, .scenarios-card .seg button { white-space: nowrap; }
.scenarios-narr p { margin: 0 0 8px; color: var(--ink-2); line-height: 1.75; }
.scenarios-narr p:last-child { margin-bottom: 0; }
.scenarios-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 1280px) { .scenarios-pair { grid-template-columns: minmax(0, 1fr); } }
.scenarios-stack { display: grid; gap: 16px; }
.scenarios-timeline { display: flex; flex-wrap: wrap; gap: 3px; margin: 2px 0 10px; }
.scenarios-timeline > div { min-width: 92px; padding: 6px 8px; border-radius: 6px; color: #fff; font-size: 12px; line-height: 1.35; overflow: hidden; }
.scenarios-timeline .n { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.scenarios-timeline .m { opacity: .92; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.scenarios-phase { border-left: 4px solid var(--phase, var(--accent)); }
.scenarios-phase-grid { display: grid; grid-template-columns: minmax(0, 1.75fr) minmax(250px, 1fr); gap: 18px; }
@media (max-width: 1180px) { .scenarios-phase-grid { grid-template-columns: minmax(0, 1fr); } }
.scenarios-facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 12px; }
.scenarios-facts .l { color: var(--ink-3); font-size: 12px; }
.scenarios-facts .v { font-weight: 600; font-variant-numeric: tabular-nums; }
.scenarios-facts .s { color: var(--ink-3); font-size: 11px; }
.scenarios-sub { font-size: 12px; font-weight: 600; color: var(--ink-2); margin: 12px 0 6px; }
.scenarios-ranges { display: flex; flex-direction: column; gap: 4px; }
.scenarios-range { font-size: 12px; display: flex; gap: 6px; align-items: baseline; flex-wrap: wrap; color: var(--ink-2); }
.scenarios-range b { font-variant-numeric: tabular-nums; color: var(--ink); }
.scenarios-eramix { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--panel-2); margin: 2px 0 6px; }
.scenarios-eramix > span { display: block; height: 100%; }
.scenarios-legend { display: flex; flex-wrap: wrap; gap: 3px 10px; font-size: 11px; color: var(--ink-3); }
.scenarios-legend i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; }
.scenarios-assetpick { display: flex; flex-wrap: wrap; gap: 6px; }
.scenarios-assetpick button { font: inherit; font-size: 12px; border: 1px solid var(--line); background: var(--panel); border-radius: 999px; padding: 3px 10px; cursor: pointer; color: var(--ink-2); }
.scenarios-assetpick button:hover { border-color: #c6cbd6; }
.scenarios-assetpick button.on { background: var(--accent-soft); border-color: #cdd7ff; color: var(--accent); font-weight: 600; }
.scenarios-note { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; }
table.tbl tr.scenarios-row-on td { background: var(--accent-soft); }
table.tbl td.scenarios-left, table.tbl th.scenarios-left { text-align: left; }
details.scenarios-more { margin-top: 10px; }
details.scenarios-more > summary { cursor: pointer; color: var(--accent); font-size: 13px; }
details.scenarios-more[open] > summary { margin-bottom: 8px; }
.scenarios-ovr { margin: 4px 0 0; padding-left: 18px; font-size: 13px; color: var(--ink-2); }
.scenarios-ovr li { margin: 2px 0; }
.scenarios-var { display: flex; flex-direction: column; }
.scenarios-var .mono { color: var(--ink-3); font-size: 11px; }
.scenarios-changed { color: var(--warn); font-weight: 600; }
.scenarios-gloss { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 6px 18px; font-size: 12px; color: var(--ink-3); margin-top: 6px; }
.scenarios-gloss b { color: var(--ink-2); }
.scenarios-oil { border-left: 3px solid #f08c00; padding: 2px 0 2px 8px; color: var(--ink-2); }
.scenarios-pdgrid { display: grid; grid-template-columns: minmax(0, 2fr) minmax(240px, 1fr); gap: 18px; align-items: center; }
@media (max-width: 1180px) { .scenarios-pdgrid { grid-template-columns: minmax(0, 1fr); } }
.scenarios-attr-pick { display: flex; gap: 6px; align-items: center; font-size: 12px; }
.scenarios-attr-pick select { max-width: 100%; }
.scenarios-attr-phases { margin: 0 0 8px; }
.scenarios-attr-read { margin: 4px 0 6px; color: var(--ink-2); line-height: 1.7; max-width: 1040px; }
.scenarios-attr-read b { font-variant-numeric: tabular-nums; }
/* phase names wrap in the header, so four phases plus 全程 fit a 900 px screen without a side scroll */
.scenarios-attr-table table.tbl th { white-space: normal; vertical-align: bottom; }
.scenarios-edge-num { color: var(--accent); font-size: 12px; margin-top: 3px; }
`;

// ---------------------------------------------------------------- small shared helpers

// One <style id="style-fe3"> for every module of this group; each module appends its block once.
export function ensureStyle(key, css) {
  let el = document.getElementById(STYLE_ID);
  if (!el) { el = document.createElement('style'); el.id = STYLE_ID; document.head.append(el); }
  const marker = `/* ${key} */`;
  if (!el.textContent.includes(marker)) el.textContent += `\n${marker}\n${css}`;
}

// The detail stylesheet; tabs call it before using any scenarios-* class outside the detail.
export function ensureDetailStyle() { ensureStyle('detail', CSS); }

// ECharts keeps every instance until it is disposed: call before replacing rendered charts
// (charts.unmount also disconnects the resize and re-layout observers).
export function clearBox(el) {
  charts.unmount(el);
  el.innerHTML = '';
}
export function disposeCharts(root) {
  if (root) root.querySelectorAll('.chart').forEach(clearBox);
}

// replaceChildren that skips null / false (the DOM would print them as text).
export function fill(el, ...kids) { el.replaceChildren(...kids.flat().filter(k => k !== null && k !== undefined && k !== false)); return el; }

export const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
const meanOf = (xs) => { const v = (xs || []).filter(isNum); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const minOf = (xs) => { const v = (xs || []).filter(isNum); return v.length ? Math.min(...v) : null; };
const last = (xs) => (Array.isArray(xs) && xs.length ? xs[xs.length - 1] : null);
const fx = (x, d) => (Math.abs(x) < 0.5 / 10 ** d ? 0 : x).toFixed(d);          // never "-0.00"
const sfx = (x, d) => { const t = fx(x, d); return Number(t) > 0 ? `+${t}` : t; };   // signed
// ui.fmtPct / ui.fmtPP without the "-0.0%" of tiny negatives (decimals / percent units)
export const pct = (x, d = 1, signed = false) => (isNum(x) ? `${signed ? sfx(x * 100, d) : fx(x * 100, d)}%` : '—');
export const pp = (x, d = 1, signed = false) => (isNum(x) ? `${signed ? sfx(x, d) : fx(x, d)}%` : '—');
export function rgba(hex, a) {
  const n = parseInt(String(hex).slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
// ECharts tooltips are HTML: escape spec text (it may come from an LLM or a user).
export const esc = (t) => String(t ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export function shortName(name) { const s = String(name || ''); return s.split(/[：:]/)[0].trim() || s; }
export function monthAdd(ym, k) {
  const [y, m] = String(ym).slice(0, 7).split('-').map(Number);
  if (!y || !m) return '';
  const t = y * 12 + (m - 1) + k;
  return `${Math.floor(t / 12)}-${String((t % 12) + 1).padStart(2, '0')}`;
}
export function eraOf(ym) {
  const d = String(ym).slice(0, 7);
  const era = (ui.META.raw?.eras || []).find(e => e.start <= d && d <= (e.end || '9999-12'));
  return era ? era.id : null;
}

// User-facing text for an API error. notBuilt: a 404 means the results were never built.
export function errorText(err, { notBuilt = true } = {}) {
  const msg = err?.message || String(err);
  if (err?.status === undefined && /Failed to fetch|NetworkError|Load failed/i.test(msg)) {   // fetch() itself failed
    return `无法连接本地服务（${msg}）：请确认 python -m scenariolab.server 仍在运行。`;
  }
  // the server explains its 404s in Chinese, with the command that builds the missing file
  if (err?.status === 404 && notBuilt && !/python -m|[一-鿿]/.test(msg)) {
    return `结果文件还没有生成（${msg}）。在项目根目录运行 python -m scenariolab.build`;
  }
  return msg;
}

// ui.table plus left-aligned headers for columns marked {left: true}.
export function table(columns, rows, opts) {
  const cols = columns.map(c => (c.left ? { ...c, cls: (v, r) => `scenarios-left ${typeof c.cls === 'function' ? c.cls(v, r) : (c.cls || '')}` } : c));
  const el = ui.table(cols, rows, opts);
  const ths = el.querySelectorAll('thead th');
  columns.forEach((c, i) => { if (c.left && ths[i]) ths[i].classList.add('scenarios-left'); });
  return el;
}

// Correlations and z-scores: never print "-0.00".
const f2 = (x) => (isNum(x) ? (Math.abs(x) < 0.005 ? 0 : x).toFixed(2) : '—');

// ---------------------------------------------------------------- variables and views

const SHOCK_UNIT = 'σ';
function unitOf(v) {
  const id = String(v).slice(2);
  if (v.startsWith('s_')) return ui.META.states[id]?.unit || '';
  if (v.startsWith('f_')) return id === 'cpi' ? 'Δlog' : (ui.META.factors[id]?.unit || '');
  if (v.startsWith('e_')) return SHOCK_UNIT;
  return '';
}

// A log change as the equivalent simple percentage change ("+0.74 log" -> "+109.6%").
const logPct = (y) => { const p = (Math.exp(y) - 1) * 100; return `${sfx(p, Math.abs(p) < 1 ? 2 : 1)}%`; };

// A value of engine variable v in its native units. f_ factors are monthly changes and are
// shown annualized (x12) unless annual=false: log changes (Δlog) as the equivalent annual
// percentage change, pp changes in percentage points, index changes in points; e_ shocks are
// in standard deviations.
export function fmtVar(v, x, { annual = true } = {}) {
  if (!isNum(x)) return '—';
  const unit = unitOf(v);
  if (v.startsWith('f_')) {
    const y = annual ? x * 12 : x, per = annual ? '/年' : '/月';
    if (unit === 'Δpp') return `${sfx(y, 2)}pp${per}`;
    if (unit === 'Δlog') return `${logPct(y)}${per}`;
    if (unit === 'Δ') return `${sfx(y, 1)} 点${per}`;
    return `${sfx(y, 3)}${per}`;
  }
  if (v.startsWith('e_')) return `${sfx(x, 2)}σ`;
  // a spread level (BAA − 10Y, 10Y − 2Y) is quoted in % like every other rate level (overview,
  // macro fans); "pp" is kept for changes
  if (unit === '%' || unit === 'pp') return `${fx(x, 2)}%`;
  if (unit === '$') return `$${x.toFixed(1)}`;
  if (unit === '倍') return `${x.toFixed(2)} 倍`;
  return x.toFixed(1);
}

// How fmtVar shows a variable, in words (under the variable name in view tables).
const UNIT_HINT = { Δlog: '年化涨跌幅', Δpp: '年化百分点变化', Δ: '年化点数变化' };
export function varCell(v) {
  const unit = unitOf(v);
  const hint = v.startsWith('f_') ? (UNIT_HINT[unit] || '年化变化') : v.startsWith('e_') ? '标准差 σ' : unit === 'pp' ? '%' : unit;
  return h('div', { class: 'scenarios-var' }, ui.varName(v) || v, h('span', { class: 'mono' }, hint ? `${v} · ${hint}` : v));
}

// What a spec view asks for, in words (keys as in schema.TARGET_KEYS).
export function viewSpecText(view) {
  const v = view.var || '';
  const parts = [];
  if (isNum(view.mean)) parts.push(`均值 ${fmtVar(v, view.mean)}`);
  if (isNum(view.mean_annual)) parts.push(`年化均值 ${fmtVar(v, view.mean_annual / 12)}`);
  if (isNum(view.shift_sd)) parts.push(`比历史均值 ${sfx(view.shift_sd, 2)} 个标准差`);
  if (isNum(view.min)) parts.push(`均值 ≥ ${fmtVar(v, view.min)}`);
  if (isNum(view.max)) parts.push(`均值 ≤ ${fmtVar(v, view.max)}`);
  if (isNum(view.min_annual)) parts.push(`均值 ≥ ${fmtVar(v, view.min_annual / 12)}`);
  if (isNum(view.max_annual)) parts.push(`均值 ≤ ${fmtVar(v, view.max_annual / 12)}`);
  if (isNum(view.vol_mult)) parts.push(`波动 × ${view.vol_mult}`);
  return parts.join('，') || '—';
}

// An annual factor drift (overrides.factor_drift, in the factor's own units).
export function driftText(fid, x) {
  const unit = fid === 'cpi' ? 'Δlog' : (ui.META.factors[fid]?.unit || '');
  if (!isNum(x)) return '—';
  if (unit === 'Δpp') return `${sfx(x, 2)}pp/年`;
  if (unit === 'Δlog') return `${logPct(x)}/年（对数 ${sfx(x, 3)}）`;
  if (unit === 'Δ') return `${sfx(x, 1)} 点/年`;
  return `${sfx(x, 3)}/年`;
}
const factorLabel = (fid) => (fid === 'cpi' ? 'CPI 总体' : ui.factorName(fid));

function asDict(x, key) {   // overrides may arrive in the LLM list form
  if (Array.isArray(x)) return Object.fromEntries(x.filter(d => d && d[key]).map(d => [d[key], d.annual]));
  return x && typeof x === 'object' ? x : {};
}

export function overrideItems(ov) {
  const out = [];
  if (!ov || typeof ov !== 'object') return out;
  for (const [f, x] of Object.entries(asDict(ov.factor_drift, 'factor'))) {
    out.push(`因子额外漂移：${factorLabel(f)} ${driftText(f, x)}——叠加在抽到的历史月份上，表达历史上没有先例的部分`);
  }
  for (const [a, x] of Object.entries(asDict(ov.asset_drift, 'asset'))) {
    out.push(`资产额外漂移：${ui.assetName(a)} ${pct(x, 1, true)}/年（价格漂移，上限 ±8%）`);
  }
  if (isNum(ov.resid_scale)) out.push(`残差缩放 × ${ov.resid_scale}：放大或缩小因子解释不了的那部分波动`);
  if (isNum(ov.r_star)) out.push(`中性实际利率 r* = ${ov.r_star}%：政策利率向“r* + 通胀”回归`);
  return out;
}

// Consecutive top months -> ranges, strongest first: "2009-08 → 2010-02, up to 4.5x".
export function monthRanges(top) {
  const k = (d) => { const [y, m] = d.split('-').map(Number); return y * 12 + m; };
  const items = (top || []).filter(t => t && typeof t.date === 'string').map(t => ({ ...t, k: k(t.date) })).sort((a, b) => a.k - b.k);
  const out = [];
  for (const t of items) {
    const prev = out[out.length - 1];
    if (prev && t.k - prev.k2 <= 1) { prev.k2 = t.k; prev.to = t.date; prev.ratio = Math.max(prev.ratio, t.weight_ratio || 0); prev.n += 1; }
    else out.push({ from: t.date, to: t.date, k2: t.k, ratio: t.weight_ratio || 0, n: 1 });
  }
  return out.sort((a, b) => b.ratio - a.ratio);
}

export function eraMix(weights) {
  const entries = Object.entries(weights || {}).filter(([, w]) => isNum(w) && w > 0.002).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return ui.h('div', { class: 'muted' }, '—');
  return h('div', {},
    h('div', { class: 'scenarios-eramix' }, entries.map(([e, w]) => h('span', {
      style: { width: `${(w * 100).toFixed(2)}%`, background: ERA_COLORS[e] || '#868e96' }, title: `${ui.eraName(e)} ${pct(w, 1)}` }))),
    h('div', { class: 'scenarios-legend' }, entries.slice(0, 5).map(([e, w]) =>
      h('span', {}, h('i', { style: { background: ERA_COLORS[e] || '#868e96' } }), `${ui.eraName(e)} ${pct(w, 0)}`))));
}

// ---------------------------------------------------------------- phases

export function phaseColor(ph, k) {
  const idle = !(ph.views || []).length && !overrideItems(ph.overrides).length;
  return idle ? IDLE_COLOR : PHASE_COLORS[k % PHASE_COLORS.length];
}

function phaseSpan(ph, labels, startYm) {
  const a = ph.start_month ?? 0, m = ph.months ?? 0;
  const at = (i) => labels[i] || (startYm ? monthAdd(startYm, i) : '');
  return { from: at(a + 1), to: at(a + m), a, m };
}

// Band label that fits: the phase name, else its number, else nothing (px = band width).
const bandLabel = (name, k, px) => (px >= String(name || '').length * 10 + 12 ? name : px >= 16 ? String(k + 1) : '');

// Phase markAreas on a monthly axis (labels[0] is the anchor month); width = plot width in px.
function phaseAreas(phases, labels, { names = true, width = 900 } = {}) {
  const total = Math.max(labels.length - 1, 1);
  return phases.map((ph, k) => {
    const { a, m } = phaseSpan(ph, labels);
    const from = labels[a], to = labels[Math.min(a + m, labels.length - 1)];
    if (!from || !to) return null;
    return { from, to, label: names ? bandLabel(ph.name, k, (width * m) / total) : '', color: rgba(phaseColor(ph, k), 0.07) };
  }).filter(Boolean);
}

// One phase: views (what is asked, what the engine used, what the drawn months average),
// overrides, and the evidence behind the weights (ENS, lambda, era mix, analog months).
// ph: {name, start_month, months, views, overrides, confidence, ep: ep_report} — the lab
// passes translator previews normalized to the same shape.
export function phaseCard(ph, k, { labels = [], startYm = null } = {}) {
  const ep = ph.ep || {};
  const { from, to, m } = phaseSpan(ph, labels, startYm);
  const views = Array.isArray(ph.views) ? ph.views : [];
  const epViews = new Map((ep.views || []).map(v => [v.var, v]));
  const color = phaseColor(ph, k);
  const changed = (a, b) => isNum(a) && isNum(b) && Math.abs(a - b) > 1e-6 * Math.max(1, Math.abs(a));

  const viewRows = views.map(v => {
    const e = epViews.get(v.var) || {};
    return { v, e, drawn: isNum(e.drawn_mean) ? e.drawn_mean : e.posterior_mean };
  });
  const used = (r) => {
    const x = r.e.target_used;
    if (!isNum(x)) return '—';
    const moved = changed(x, r.e.target) || r.e.clipped;
    return h('span', { class: moved ? 'scenarios-changed' : '', title: moved ? `要求 ${fmtVar(r.v.var, r.e.target)}，引擎实际使用 ${fmtVar(r.v.var, x)}` : '' }, fmtVar(r.v.var, x));
  };
  const viewTable = viewRows.length ? viewsTable(viewRows, used)
    : ui.notice('这一阶段没有视图：按 1973 年以来的历史先验抽样（每个 6 个月区块的机会基本均等，只有 2020 年封锁月份被降权）。', 'info');

  const ovr = overrideItems(ph.overrides);
  const ranges = views.length ? monthRanges(ep.top_months).slice(0, 4) : [];
  const shrink = isNum(ep.shrink) ? ep.shrink : 1;
  const conf = isNum(ph.confidence) ? ph.confidence : 1;
  const ensSub = isNum(ep.ens_ratio) ? `占可抽月份的 ${pct(ep.ens_ratio, 0)}` : '';
  // a phase without views draws from the prior: λ and the confidence blend have nothing to act on
  const noViews = !views.length;
  const facts = [
    ['有效月份数 ENS', isNum(ep.ens) ? ep.ens.toFixed(0) : '—', ensSub],
    isNum(ep.episodes) ? ['有效历史片段', `${ep.episodes.toFixed(1)} 段`, '权重来自几段不同的 12 个月历史'] : null,
    noViews ? ['视图收缩 λ', '—', '没有视图，不涉及收缩'] : ['视图收缩 λ', shrink.toFixed(2), shrinkText(ep, shrink)],
    noViews ? ['置信度', '—', '没有视图：权重就是历史先验']
      : ['置信度', conf.toFixed(2), conf >= 1 ? '权重全部来自视图' : `视图占 ${pct(conf, 0)}，其余回到历史先验`],
    ['抽样区块', isNum(ep.block) ? `${ep.block} 个月` : '—', ep.converged === false ? '熵池未完全收敛' : '连续月份一起抽'],
  ].filter(Boolean);
  const oil = oilNote(ph);

  const right = h('div', {},
    h('div', { class: 'scenarios-facts' }, facts.map(([l, v, s]) =>
      h('div', {}, h('div', { class: 'l' }, l), h('div', { class: 'v' }, v), s ? h('div', { class: 's' }, s) : null))),
    h('div', { class: 'scenarios-sub' }, '抽到的月份来自哪些时代'),
    eraMix(ep.era_weights),
    h('div', { class: 'scenarios-sub' }, '这一阶段像历史上的哪些月份'),
    ranges.length ? h('div', { class: 'scenarios-ranges' }, ranges.map(r => {
      const era = eraOf(r.from);
      return h('div', { class: 'scenarios-range' },
        h('b', {}, r.from === r.to ? r.from : `${r.from} → ${r.to}`),
        era ? h('span', {}, ui.eraName(era)) : null,
        h('span', { class: 'muted' }, `被抽中的概率最高达历史平均的 ${r.ratio.toFixed(1)} 倍`));
    })) : h('div', { class: 'muted', style: { fontSize: '12px' } }, noViews
      ? '没有视图：除 2020 年封锁月份降权外，各段历史被抽中的机会均等。' : '没有哪段历史明显被多抽。'));

  const sub = `${m} 个月（${from || '—'} → ${to || '—'}）：视图决定这一阶段多抽哪些真实历史月份；同时列出权重的证据（ENS、收缩 λ、时代构成、最像的历史月份）。`;
  return h('div', { class: 'card scenarios-phase', style: { '--phase': color } },
    h('div', { class: 'card-head' }, h('div', {},
      h('h3', { class: 'card-title' }, `阶段 ${k + 1} · ${ph.name || ''}`), h('div', { class: 'card-sub' }, sub))),
    h('div', { class: 'scenarios-phase-grid' },
      h('div', {}, viewTable,
        oil ? h('p', { class: 'scenarios-note scenarios-oil' }, oil) : null,
        ovr.length ? h('div', {}, h('div', { class: 'scenarios-sub' }, '覆盖项（超出历史支撑、有上限的部分）'),
          h('ul', { class: 'scenarios-ovr' }, ovr.map(t => h('li', {}, t)))) : null),
      right));
}

// Why a phase's views were shrunk toward history (ep.shrink_reason from engine/pooling.py).
function shrinkText(ep, shrink) {
  if (shrink >= 1) return '视图原样执行';
  if (ep.shrink_reason === 'infeasible') return '历史上找不到能同时满足所有视图的月份组合，已收缩';
  if (ep.shrink_reason === 'thin') {
    return `历史支撑太薄${isNum(ep.ens_requested) ? `（原样只有约 ${ep.ens_requested.toFixed(0)} 个有效月份）` : '（有效月份或历史片段太少）'}，已收缩`;
  }
  return '已向历史均值收缩';
}

// A phase that states its own oil path (a view on these, or an oil drift) gets no oil reversion
// in the engine (simulate.with_oil_lock / OIL_VIEW_VARS).
const OIL_VIEW_VARS = ['f_oil', 's_oil_yoy', 's_real_oil'];
export function oilNote(ph) {
  const views = Array.isArray(ph?.views) ? ph.views : [];
  if (views.some(v => OIL_VIEW_VARS.includes(v?.var))) return '该阶段给了油价视图，油价按视图走，不做均值回归。';
  const drift = asDict(ph?.overrides?.factor_drift, 'factor');
  if (isNum(drift.oil)) return '该阶段给了油价漂移，油价按漂移走，不做均值回归。';
  return null;
}

// Views with their reason on a second row: [{v: spec view, e: ep report view, drawn}].
// numbers=false drops the engine columns (used for the calculator's views).
export function viewsTable(rows, used, { numbers = true } = {}) {
  const heads = numbers ? ['变量', '情景设定', '历史均值', '引擎采用', '抽到的月份平均'] : ['变量', '情景设定'];
  return h('div', { class: 'table-wrap' }, h('table', { class: 'tbl scenarios-vtable' },
    h('thead', {}, h('tr', {}, heads.map((t, i) => h('th', { class: i < 2 ? 'scenarios-left' : '' }, t)))),
    h('tbody', {}, rows.flatMap(r => [
      h('tr', { class: `scenarios-vrow${r.v.why ? ' has-why' : ''}` },
        h('td', {}, varCell(r.v.var)), h('td', { class: 'scenarios-left scenarios-wrap' }, viewSpecText(r.v)),
        numbers ? [h('td', {}, fmtVar(r.v.var, r.e?.prior_mean)), h('td', {}, used ? used(r) : fmtVar(r.v.var, r.e?.target_used)),
          h('td', {}, fmtVar(r.v.var, r.drawn))] : null),
      r.v.why ? h('tr', { class: 'scenarios-why' }, h('td', { colspan: String(heads.length) }, `原因：${r.v.why}`)) : null,
    ].filter(Boolean)))));
}

// Phases as proportional blocks; short phases show only their length (the full name and the
// dates are in the tooltip and in the phase cards below).
function timelineBar(phases, labels, startYm) {
  const total = phases.reduce((t, p) => t + (p.months || 0), 0) || 1;
  return h('div', { class: 'scenarios-timeline' }, phases.map((ph, k) => {
    const { from, to, m } = phaseSpan(ph, labels, startYm);
    const wide = m / total >= 0.2 && from;
    return h('div', { style: { flex: `${Math.max(m, 1)} 1 0`, background: phaseColor(ph, k) }, title: `${k + 1}. ${ph.name || ''}：${m} 个月（${from || '—'} → ${to || '—'}）` },
      h('div', { class: 'n' }, `${k + 1}. ${ph.name || ''}`), h('div', { class: 'm' }, wide ? `${m} 个月 · ${from} 起` : `${m} 个月`));
  }));
}

export const GLOSSARY = [
  ['视图', '对一个阶段里抽到的历史月份的平均值提出要求（例如 CPI 平均 5.5%），而不是直接规定资产涨跌'],
  ['熵池（Entropy Pooling）', '在满足视图的前提下，找与历史分布最接近的月份权重'],
  ['有效月份数 ENS', '权重相当于均匀抽取多少个历史月份；越小说明情景越极端、越依赖少数时期'],
  ['有效历史片段', '权重相当于来自多少段不同的 12 个月历史；相邻月份高度重叠，所以它比 ENS 更能看出情景是不是只在重复几段历史'],
  ['视图收缩 λ', '1 = 视图原样执行；< 1 表示历史上凑不出足够的月份（或片段），视图按 λ 向历史均值收缩'],
  ['置信度', '把视图后验和历史先验按这个比例混合，越低越接近无条件历史'],
  ['抽样区块', '每次抽连续 6 个月，保留通胀惯性和波动聚集'],
];

// ---------------------------------------------------------------- causal chain

// The chain is drawn by charts.dag: a layered pixel layout (left to right when the layers fit,
// otherwise top to bottom), edges cut at the box borders so every arrowhead shows, re-laid out
// when the width changes, and escaped tooltips that name the boxes by their labels.
// num(e) -> {text, tip} | null: the engine's number for an edge (see edgeNumber), appended to
// the edge's "sign lag" label and to its tooltip.
const SIGN_WORD = { '+': '同向（+）', '-': '反向（−）', '±': '方向不定（±）' };
const SIGN_CHAR = { '-': '−' };
function drawChain(el, nodes, edges, num = null) {
  return charts.dag(el, {
    nodes, edges,
    edgeLabel: (e) => {
      const base = `${SIGN_CHAR[e.sign] || e.sign || ''} ${e.lag || ''}`.trim();
      const x = num ? num(e) : null;
      return x ? `${base} · ${x.text}` : base;
    },
    edgeTip: (e, label) => {
      const x = num ? num(e) : null;
      return `<b>${esc(label(e.from))} → ${esc(label(e.to))}</b><br/>${SIGN_WORD[e.sign] || '方向不定'}，时滞 ${esc(e.lag || '—')}`
        + `${e.mechanism || e.label ? `<br/>${esc(e.mechanism || e.label)}` : ''}`
        + (x ? `<br/><b>引擎归因</b>：${x.tip}` : '');
    },
  });
}

// ---------------------------------------------------------------- correlation heatmap

// Correlation matrix with cells whose posterior-minus-prior gap is large outlined.
function corrChart(el, { names, post, prior, mode }) {
  const chart = charts.mount(el);
  const n = names.length, data = [];
  let maxDiff = 0;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const a = post?.[r]?.[c], b = prior?.[r]?.[c];
      const diff = isNum(a) && isNum(b) ? a - b : null;
      if (diff !== null && r !== c) maxDiff = Math.max(maxDiff, Math.abs(diff));
      const v = mode === 'diff' ? diff : mode === 'prior' ? b : a;
      const hot = mode === 'post' && diff !== null && r !== c && Math.abs(diff) >= 0.1;
      data.push({ value: [c, r, isNum(v) ? v : '-'], a, b, diff,
        itemStyle: hot ? { borderColor: '#1d2433', borderWidth: 2 } : { borderColor: '#fff', borderWidth: 1 } });
    }
  }
  const bound = mode === 'diff' ? Math.max(0.2, Math.ceil(maxDiff * 10) / 10) : 1;
  // cell size after the row labels: upright column labels and no cell numbers once cells get small
  const rowPx = Math.max(...names.map(t => [...String(t)].reduce((w, ch) => w + (/[\x00-\xff]/.test(ch) ? 6.6 : 11), 0)), 20) + 12;
  const cell = ((el.clientWidth || 600) - rowPx - 20) / Math.max(n, 1);
  chart.setOption({
    tooltip: {
      confine: true,
      formatter: (p) => `${esc(names[p.data.value[1]])} × ${esc(names[p.data.value[0]])}<br/>情景（后验）<b>${f2(p.data.a)}</b>　历史（先验）${f2(p.data.b)}` +
        `<br/>变化 ${isNum(p.data.diff) ? sfx(p.data.diff, 2) : '—'}`,
    },
    grid: { left: 8, right: 10, top: 8, bottom: 8, containLabel: true },
    xAxis: { type: 'category', data: names, axisTick: { show: false }, axisLine: { lineStyle: { color: '#e3e6ec' } },
      axisLabel: { color: '#7a8396', fontSize: 11, rotate: cell >= 24 ? 45 : 90, interval: 0 } },
    yAxis: { type: 'category', data: names, inverse: true, axisTick: { show: false }, axisLine: { lineStyle: { color: '#e3e6ec' } },
      axisLabel: { color: '#7a8396', fontSize: 11, interval: 0 } },
    visualMap: { min: -bound, max: bound, show: false, calculable: false, dimension: 2,
      inRange: { color: ['#1c7ed6', '#a5d8ff', '#f8f9fa', '#ffc9c9', '#e03131'] } },
    series: [{ type: 'heatmap', data,
      label: { show: cell >= 26, fontSize: 9, color: '#1d2433', formatter: (p) => (isNum(p.data.value[2]) ? f2(p.data.value[2]) : '') },
      emphasis: { itemStyle: { borderColor: '#1d2433', borderWidth: 1 } } }],
  });
  return chart;
}

// ---------------------------------------------------------------- sections

function headerCard(d, opts) {
  const eq = d.assets?.us_equity?.stats || {};
  const phases = Array.isArray(d.phases) ? d.phases : [];
  const cpiAvg = meanOf((d.macro?.cpi_yoy?.fan?.p50 || []).slice(1));
  const ffEnd = last(d.macro?.ff?.fan?.p50), y10End = last(d.macro?.y10?.fan?.p50);
  const ensMin = minOf(phases.map(p => p.ep?.ens));
  const epiMin = minOf(phases.filter(p => (p.views || []).length).map(p => p.ep?.episodes));
  const prob = d.probability || {};
  const probKpi = isNum(d.prob)
    ? ui.kpi('后验概率', pct(d.prob, 1), `主观先验 ${pct(prob.prior ?? d.prior_prob, 1)} · 似然 ${isNum(prob.likelihood) ? prob.likelihood.toFixed(2) : '—'}`)
    : ui.kpi('情景概率', '—', `自定义情景不参与归一化；主观先验 ${pct(d.prior_prob, 0)}`);
  const kpis = [
    probKpi,
    ui.kpi('美股大盘 名义年化（中位数）', pct(eq.cagr_p50, 1), `5%–95%：${pct(eq.cagr_p5, 1)} ~ ${pct(eq.cagr_p95, 1)}`, ui.signCls(eq.cagr_p50)),
    ui.kpi('美股大盘 实际年化（中位数）', pct(eq.real_cagr_p50, 1), '扣除每条路径自己的通胀', ui.signCls(eq.real_cagr_p50)),
    ui.kpi('美股最大回撤（中位数）', pct(eq.max_dd_p50, 0), '15 年内从高点的最大跌幅', 'neg'),
    ui.kpi('平均 CPI 同比', pp(cpiAvg, 2), '中位数路径 15 年平均'),
    ui.kpi('期末联邦基金利率', pp(ffEnd, 2), `起点 ${pp(d.macro?.ff?.fan?.p50?.[0], 2)}`),
    ui.kpi('期末 10 年期收益率', pp(y10End, 2), `起点 ${pp(d.macro?.y10?.fan?.p50?.[0], 2)}`),
    ui.kpi('最少有效月份数', isNum(ensMin) ? ensMin.toFixed(0) : '—',
      `各阶段 ENS 的最小值${isNum(epiMin) ? `，历史片段最少 ${epiMin.toFixed(1)} 段` : ''}；越小越极端`, isNum(ensMin) && ensMin < 60 ? 'is-bad' : ''),
  ];
  const src = d.source || 'builtin';
  const chips = [
    ui.chip(SOURCE_NAMES[src] || src, src === 'builtin' ? '' : 'accent'),
    ...(d.tags || []).map(t => ui.chip(t)),
    d.calibration ? ui.chip('含收入分配计算器', 'good') : null,
    d.created ? ui.chip(`保存于 ${d.created}`) : null,
    h('span', { class: 'muted', style: { fontSize: '12px' } },
      `${d.n_paths || '—'} 条路径 · 起点 ${String(d.start || d.labels?.[0] || '').slice(0, 7)} · ${Math.max((d.labels || []).length - 1, 0)} 个月`),
  ];
  const lab = opts.labLink ? h('a', { class: 'btn', href: src === 'builtin' ? `#lab?tpl=${d.id}` : `#lab?${d.id}` }, '在实验室里修改这个情景') : null;
  return h('div', { class: 'card' },
    h('div', { class: 'scenarios-head' },
      h('div', { style: { minWidth: 0, flex: '1 1 480px' } },
        h('h2', { class: 'scenarios-name' }, d.name || d.id || '未命名情景'),
        h('div', { class: 'scenarios-chips' }, chips)),
      lab),
    d.summary ? h('p', { class: 'scenarios-summary' }, d.summary) : null,
    h('div', { class: 'scenarios-kpis scenarios-kpis4' }, kpis));
}

export function validationCard(v) {
  const val = v || {};
  const errors = val.errors || [], warnings = val.warnings || [], clipped = val.clipped || [];
  const sub = '规格校验（validate_spec）是叙事和引擎之间的护栏：不认识的变量被忽略，超出历史 2%–98% 区间的目标被截断，阶段补齐到 180 个月，所有改动都列在这里。';
  if (!v) return card({ title: '校验与截断', sub }, ui.notice('这个结果没有附带校验报告。', 'info'));
  if (!errors.length && !warnings.length && !clipped.length) {
    return card({ title: '校验与截断', sub }, ui.notice('校验通过：所有视图和覆盖项都在历史可支撑的范围内，没有被忽略、截断或收缩。', 'info'));
  }
  const varLabel = (x) => {
    const s = String(x || '');
    if (s.startsWith('factor_drift.')) return `因子漂移 · ${factorLabel(s.slice(13))}`;
    if (s.startsWith('asset_drift.')) return `资产漂移 · ${ui.assetName(s.slice(12))}`;
    if (s === 'resid_scale') return '残差缩放';
    if (s === 'r_star') return '中性实际利率 r*';
    return `${ui.varName(s) || s}（${s}）`;
  };
  return card({ title: '校验与截断', sub },
    errors.length ? ui.notice(`错误：${errors.join('；')}`, 'error') : null,
    warnings.length ? h('ul', { class: 'list', style: { margin: '6px 0 0' } }, warnings.map(w => h('li', {}, w))) : null,
    clipped.length ? h('div', { style: { marginTop: '10px' } },
      h('div', { class: 'scenarios-sub' }, `被截断的数值（${clipped.length} 处）`),
      table([
        { key: 'phase', label: '阶段', fmt: (x) => (x ? `阶段 ${x}` : '—') },
        { key: 'var', label: '变量 / 参数', fmt: varLabel, left: true },
        { key: 'key', label: '键', fmt: (x) => x || '—', left: true },
        { key: 'requested', label: '要求值', fmt: (x) => (isNum(x) ? String(Number(x.toPrecision(4))) : '—') },
        { key: 'used', label: '实际使用', fmt: (x) => (isNum(x) ? String(Number(x.toPrecision(4))) : '—'), cls: 'scenarios-changed' },
      ], clipped)) : null);
}

function narrativeCard(d) {
  const paras = String(d.narrative || '').split(/\n+/).map(s => s.trim()).filter(Boolean);
  const analogs = (d.analogs || []).map(a => {
    const [f, t] = String(a).split(':');
    // a range that crosses an era boundary ("2023-01:2026-09") names both eras, not only the first
    const era = eraOf(f), era2 = t ? eraOf(t) : null;
    const eras = [era, era2 !== era ? era2 : null].filter(Boolean).map(ui.eraName).join(' → ');
    return ui.chip(`${f} → ${t || f}${eras ? ` · ${eras}` : ''}`);
  });
  return card({ title: '情景叙事', sub: '为什么会发生、通过哪些渠道传导；历史类比区间是规格作者认为最相似的真实时期。' },
    h('div', { class: 'scenarios-narr' }, paras.length ? paras.map(p => h('p', {}, p)) : h('p', { class: 'muted' }, '这个情景没有叙事说明。')),
    analogs.length ? h('div', { style: { marginTop: '10px' } }, h('div', { class: 'scenarios-sub' }, '历史类比区间'), h('div', { class: 'scenarios-chips' }, analogs)) : null);
}

function chainCard(d, later, state, bus) {
  const nodes = d.causal_chain?.nodes || [], edges = d.causal_chain?.edges || [];
  const numbered = edges.filter(e => edgeTerm(d, e)).length;
  const sub = '箭头是“因 → 果”：蓝色同向（+）、橙色反向（−）、灰色方向不定（±），线上标注符号和时滞，悬停看机制；这是解释图，引擎通过下面各阶段的视图把它落到真实历史月份上。'
    + (numbered ? `指向资产的 ${numbered} 条边另标了引擎的传导归因（该因子对该资产的年化贡献，阶段跟随下面「传导归因」卡片的选择）。` : '');
  if (!edges.length) return card({ title: '因果传导链', sub }, ui.notice('这个情景没有给出因果链。', 'info'));
  const box = h('div', { class: 'chart' });
  const draw = () => drawChain(box, nodes, edges, numbered ? (e) => edgeNumber(d, e, state.attr?.phase ?? 'all') : null);
  if (numbered) bus.redrawChain = () => { if (box.isConnected) { try { draw(); } catch (err) { console.error(err); } } };
  later.push([box, draw]);
  const signText = (s) => ({ '+': '+ 同向', '-': '− 反向', '±': '± 不定' }[s] || s || '—');
  return card({ title: '因果传导链', sub }, box,
    h('details', { class: 'scenarios-more', open: true }, h('summary', {}, `每条传导的机制（${edges.length} 条）`),
      table([
        { key: 'from', label: '因' },
        { key: 'sign', label: '方向', fmt: signText, left: true },
        { key: 'to', label: '果', left: true },
        { key: 'lag', label: '时滞', fmt: (x) => x || '—', left: true },
        // the engine's number under the mechanism where the edge runs from a factor box to an asset box
        { key: 'mechanism', label: '机制', wrap: true, fmt: (x, e) => {
          const s = numbered ? edgeSummary(d, e) : null;
          return s ? h('div', {}, x || '—', h('div', { class: 'scenarios-edge-num' }, `引擎归因（年化贡献）：${s}`)) : (x || '—');
        } },
      ], edges, { stack: true })));          // on a narrow card each edge becomes a labelled block
}

// ---------------------------------------------------------------- transmission attribution (传导归因)
//
// detail.assets[a].attribution (scenarios/runner.attribution_summary): per phase and over the whole
// path, the mean over paths of every additive part of the asset's monthly return, annualized (x12):
// one term per transmission factor (beta x the SIMULATED factor change), the intercept, the CMA
// valuation adjustment, the scenario's asset drift, the residual and income; they add up to
// `arithmetic`, and `reconciliation` (compounding, volatility drag, median vs mean) takes that to the
// median CAGR shown in the stats table. Structural assets: the 10-year bond = coupon + repricing on
// the yield change (term y10); cash = interest; crude oil = the oil path (term oil).

const FACTOR_SHORT = { oil: '油价', ip: '工业产出', cpi_goods: '商品通胀', cpi_services: '服务通胀', ff: '政策利率',
  y10: '10 年期收益率', spread: '信用利差', usd: '美元', unrate: '失业率', labor_share: '劳动收入占比' };
const ATTR_NAMES = { intercept: '常数项', unattributed: '未归因的传导', cma: '估值调整', overrides: '情景覆盖',
  valuation_pull: '估值回归', residual: '未解释部分', income: '收入', floor: '−95% 截断' };
const ATTR_HINT = {
  intercept: '所有因子变动为零时传导模型给出的价格回报（1973 年以来全样本回归的常数）；每个情景都一样',
  unattributed: '传导模型不是同月因子的线性组合（神经网络、滞后或通胀状态交互项），系统性部分不拆分',
  cma: '资本市场假设：去掉历史上估值抬升的顺风，并向中位数估值部分回归；每个情景都一样',
  overrides: '情景的资产额外漂移（asset_drift，超出历史支撑、有上限）；因子漂移通过因子路径体现在因子项里',
  valuation_pull: '估值状态模式下价格被拉向合理估值的部分',
  residual: '抽到的历史月份里因子解释不了的部分（残差部分耦合：一半来自同一批月份，一半来自独立抽样）',
  income: '股息或利息收入（按今天的收益率）',
  floor: '单月总回报不低于 −95% 的截断',
  reconciliation: '算术年化（月均 ×12）换算成几何年化：复利、波动拖累，以及“路径中位数 ≠ 均值”的差别；短阶段里后者常占主导，可正可负',
};
// The step from the arithmetic sum to the median CAGR. Not called a "drag": in short phases the median
// of a skewed distribution can sit well above its mean (financial_crisis 金融股 phase 1: +5.0 pp).
const RECON_LABEL = '复利、波动与中位数修正';
// what the CMA term is for one asset (capital_market_assumptions): the market's valuation adjustment;
// another equity's is that scaled by its beta minus half its historical drift relative to the market;
// gold's removes its sample real appreciation
function cmaHint(aid) {
  if (aid === 'us_equity') return '资本市场假设：去掉历史上估值抬升的顺风，并向中位数估值部分回归；每个情景都一样';
  if (aid === 'gold') return '资本市场假设：去掉样本里黄金的实际升值（主要是 1970 年代的一次性重估），长期只跟上通胀；每个情景都一样';
  return '资本市场假设：大盘的估值调整按该资产对大盘的β缩放，再把它相对大盘的历史超额漂移减半；每个情景都一样';
}
// causal-chain boxes that clearly name an engine factor (or the CMA term) and boxes that name an asset;
// an edge from the first kind to the second gets the engine's number. Labels must match exactly.
const NODE_TERM = {
  油价: 'oil', 国际油价: 'oil', 工业产出: 'ip', 通胀: 'inflation', 'CPI 通胀': 'inflation', CPI: 'inflation',
  商品通胀: 'cpi_goods', 服务通胀: 'cpi_services', 政策利率: 'ff', 联邦基金利率: 'ff',
  '10 年期收益率': 'y10', '10年期收益率': 'y10', 长期国债收益率: 'y10', 长期收益率: 'y10', 信用利差: 'spread',
  美元: 'usd', 美元走弱: 'usd', 美元走强: 'usd', 失业率: 'unrate', 劳动收入占比: 'labor_share', 高估值: 'cma',
};
const NODE_ASSET = {
  美股: 'us_equity', 标普: 'us_equity', 标普500: 'us_equity', '标普 500': 'us_equity', 股市: 'us_equity',
  股票长期回报: 'us_equity', 纳斯达克: 'nasdaq', 科技股: 'nasdaq', 能源股: 'energy', 金融股: 'financials',
  银行股: 'financials', 公用事业股: 'utilities', 工业股: 'industrials', 医疗股: 'healthcare', REITs: 'real_estate',
  黄金: 'gold', 金价: 'gold', 原油: 'crude_oil', 长期国债价格: 'treasury_10y', 美债: 'treasury_10y', 现金: 'cash',
};
const nodeKey = (s) => String(s ?? '').trim().replace(/\s+/g, ' ');
function nodeAsset(label) {
  const k = nodeKey(label);
  if (NODE_ASSET[k]) return NODE_ASSET[k];
  return Object.values(ui.META.assets).find(a => a.name === k)?.id || null;     // 美股大盘, 石油能源股, …
}

const isFactorTerm = (k) => !(k in ATTR_NAMES);
function attrName(k, kind) {
  if (kind === 'treasury') return { y10: '收益率变动（债券重新定价）', income: '票息收入' }[k] || ATTR_NAMES[k] || k;
  if (kind === 'cash' && k === 'income') return '利息收入';
  if (k === 'inflation') return '通胀（商品 + 服务）';
  if (ATTR_NAMES[k]) return ATTR_NAMES[k];
  if (FACTOR_SHORT[k]) return FACTOR_SHORT[k];
  return k.startsWith('e_') ? ui.varName(k) : ui.factorName(k);
}
// the block of the selected phase: 'all' = the whole path, else the phase index
const attrBlock = (att, sel) => (sel === 'all' ? att?.total : att?.phases?.[Number(sel)]) || null;
const phaseLabel = (att, sel) => {
  if (sel === 'all') return `${Math.round((att?.total?.months || 180) / 12)} 年全程`;
  const p = att?.phases?.[Number(sel)];
  return p ? `「${p.name}」阶段，${p.months} 个月` : '';
};
// the value of a term or a combined channel (inflation = goods + services) in a block
function termValue(blk, key) {
  const t = blk?.terms || {};
  if (key === 'inflation') return isNum(t.cpi_goods) || isNum(t.cpi_services) ? (t.cpi_goods || 0) + (t.cpi_services || 0) : null;
  return isNum(t[key]) ? t[key] : null;
}

// The engine's number for a causal edge "factor box -> asset box", or null.
function edgeTerm(d, e) {
  const term = NODE_TERM[nodeKey(e?.from)], aid = nodeAsset(e?.to);
  if (!term || !aid) return null;
  // the CMA term is the valuation channel only for the market itself (see cmaHint): another asset's also
  // holds its halved relative drift, gold's its real-price drift, so "高估值 → 黄金" would be mislabelled
  if (term === 'cma' && aid !== 'us_equity') return null;
  const att = d.assets?.[aid]?.attribution;
  return att && termValue(att.total, term) !== null ? { term, aid, att } : null;
}
// "全程 +0.4%；冲击期 +18.7%；…": the edge's number over the whole path and in every phase (plain text)
function edgeSummary(d, e) {
  const m = edgeTerm(d, e);
  if (!m) return null;
  return [['全程', m.att.total], ...(m.att.phases || []).map(p => [p.name, p])]
    .map(([n, b]) => `${n} ${pct(termValue(b, m.term), 1, true)}`).join('；');
}
function edgeNumber(d, e, sel) {
  const m = edgeTerm(d, e);
  if (!m) return null;
  const blk = attrBlock(m.att, sel) || m.att.total;
  const v = termValue(blk, m.term);
  if (!isNum(v)) return null;
  return { text: `${pct(v, 1, true)}/年`,
    tip: `${esc(attrName(m.term))}对${esc(ui.assetName(m.aid))}的年化贡献，${esc(sel === 'all' ? '全程' : (attrBlock(m.att, sel)?.name || ''))} ${pct(v, 1, true)}/年`
      + `<br/>${esc(edgeSummary(d, e))}（%/年）` };
}

// Waterfall rows of one block: [{key, label, value, total?, tip?}] (inflation shown as one row).
function attrRows(blk, kind, aid) {
  const t = blk?.terms || {};
  const keys = Object.keys(t).filter(k => isNum(t[k]));
  const rows = [];
  const factorKeys = kind === 'model' ? keys.filter(isFactorTerm) : [];
  for (const k of factorKeys) {
    if (k === 'cpi_services' && isNum(t.cpi_goods)) continue;
    if (k === 'cpi_goods' && isNum(t.cpi_services)) {
      rows.push({ key: 'inflation', label: attrName('inflation'), value: t.cpi_goods + t.cpi_services,
        tip: `商品通胀 ${pct(t.cpi_goods, 2, true)} + 服务通胀 ${pct(t.cpi_services, 2, true)}` });
      continue;
    }
    rows.push({ key: k, label: attrName(k), value: t[k] });
  }
  if (factorKeys.length && isNum(blk.factor_sum)) {
    const band = Array.isArray(blk.factor_band) ? blk.factor_band : [];
    rows.push({ key: 'factor_sum', label: '＝ 因子合计', value: blk.factor_sum, total: true,
      tip: band.length === 2 ? `各路径因子合计的 10%–90% 区间：${pct(band[0], 1, true)} ~ ${pct(band[1], 1, true)}` : '' });
  }
  for (const k of keys.filter(k => kind !== 'model' || !isFactorTerm(k))) {
    rows.push({ key: k, label: attrName(k, kind), value: t[k], tip: k === 'cma' ? cmaHint(aid) : undefined });
  }
  rows.push({ key: 'arithmetic', label: '＝ 算术年化合计', value: blk.arithmetic, total: true, tip: '上面各项相加：每月算术回报的均值 ×12' });
  rows.push({ key: 'reconciliation', label: RECON_LABEL, value: blk.reconciliation });
  rows.push({ key: 'cagr_p50', label: '＝ 中位数年化回报', value: blk.cagr_p50, total: true, final: true,
    tip: '各路径几何年化总回报（CAGR）的中位数' });
  return rows.filter(r => isNum(r.value));
}

// Horizontal waterfall: each term floats from the running sum, totals start at 0. Contributions take
// the K-line up / down colours (the sign of a number), totals a neutral colour.
function drawWaterfall(el, rows) {
  const { up, down } = charts.klineColors();
  let cum = 0;
  const spans = rows.map(r => {
    if (r.total) { cum = r.value; return [0, r.value]; }
    const s = cum; cum += r.value;
    return [s, cum];
  });
  el.style.height = `${rows.length * 26 + 34}px`;
  const chart = charts.mount(el);
  const color = (r) => (r.final ? '#3b5bdb' : r.total ? '#5c677d' : r.value >= 0 ? up : down);
  const lab = (r) => `${pct(r.value, 1, true)}`;
  // room for the value labels beyond the bar ends (wider on a phone, where the plot is narrow)
  const narrow = (el.clientWidth || 600) < 520;
  const hasNeg = rows.some((r, i) => r.value < 0 && Math.min(...spans[i]) <= Math.min(0, ...spans.flat()) + 1e-12);
  const gap = [hasNeg ? (narrow ? '28%' : '16%') : '6%', narrow ? '24%' : '14%'];
  chart.setOption({
    animation: false,
    grid: { left: 8, right: 16, top: 6, bottom: 22, containLabel: true },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true,
      formatter: (ps) => {
        const i = (Array.isArray(ps) ? ps : [ps]).find(p => Number.isInteger(p?.dataIndex))?.dataIndex;
        const r = rows[i];
        if (!r) return '';
        const [s, e] = spans[i];
        const hint = r.tip || ATTR_HINT[r.key] || '';
        return `<b>${esc(r.label)}</b>：${pct(r.value, 2, true)}/年`
          + (r.total ? '' : `<br/>累计 ${pct(s, 1, true)} → ${pct(e, 1, true)}`)
          + (hint ? `<br/><span style="color:#7a8396">${esc(hint)}</span>` : '');
      },
    },
    xAxis: { type: 'value', boundaryGap: gap, axisLine: { show: false }, axisTick: { show: false },
      splitLine: { lineStyle: { color: '#eef0f4' } },
      axisLabel: { color: '#7a8396', fontSize: 11, hideOverlap: true, formatter: (v) => pct(v, 0) } },
    yAxis: { type: 'category', inverse: true, data: rows.map(r => r.label), axisTick: { show: false },
      axisLine: { lineStyle: { color: '#e3e6ec' } },
      axisLabel: { color: '#1d2433', fontSize: 11, interval: 0,
        formatter: (v, i) => (rows[i]?.total ? `{b|${v}}` : v), rich: { b: { fontWeight: 'bold', color: '#1d2433', fontSize: 11 } } } },
    series: [
      { name: '_base', type: 'bar', stack: 'w', stackStrategy: 'all', silent: true, barMaxWidth: 18,
        itemStyle: { color: 'transparent' }, emphasis: { disabled: true }, tooltip: { show: false },
        data: rows.map((r, i) => (r.total ? 0 : Math.min(...spans[i]))) },
      { name: '贡献', type: 'bar', stack: 'w', stackStrategy: 'all', barMaxWidth: 18,
        data: rows.map((r, i) => ({ value: r.total ? r.value : Math.abs(spans[i][1] - spans[i][0]),
          itemStyle: { color: color(r), borderRadius: 2 },
          label: { show: true, position: r.value < 0 ? 'left' : 'right', formatter: lab(r), fontSize: 10,
            color: r.total ? '#1d2433' : '#4a5468', fontWeight: r.total ? 'bold' : 'normal' } })),
        markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: '#adb5bd', type: 'dashed' }, data: [{ xAxis: 0 }] } },
    ],
  });
  return chart;
}

// The one-line reading of a block, built from its numbers.
function attrReading(aid, att, sel) {
  const blk = attrBlock(att, sel);
  if (!blk) return null;
  const name = ui.assetName(aid), where = phaseLabel(att, sel);
  const num = (v, d = 1) => h('b', { class: ui.signCls(v) }, pct(v, d, true));
  // the reconciliation is added, whatever its sign (a positive one is mostly "median above the mean")
  const tail = [`，各项相加为算术年化 `, num(blk.arithmetic), '，再计入复利、波动拖累和“中位数 ≠ 均值”的差别（', num(blk.reconciliation),
    '），即中位数年化回报 ', num(blk.cagr_p50), '。'];
  const t = blk.terms || {};
  // the parts that are the same in every scenario (intercept, CMA, today's income yield), named so the
  // sentence adds up to the arithmetic sum
  const fixedPart = () => {
    const ks = ['intercept', 'cma', 'income'].filter(k => isNum(t[k]));
    if (!ks.length) return [];
    const v = ks.reduce((s, k) => s + t[k], 0);
    const names = ks.map(k => ATTR_NAMES[k]);
    const list = names.length > 1 ? `${names.slice(0, -1).join('、')}和${names[names.length - 1]}合计` : names[0];
    return [`，${list} `, num(v), '/年'];
  };
  const extra = () => [
    ...(isNum(t.overrides) && Math.abs(t.overrides) >= 0.0005 ? ['，情景覆盖 ', num(t.overrides), '/年'] : []),
    ...(isNum(t.valuation_pull) && Math.abs(t.valuation_pull) >= 0.0005 ? ['，估值回归 ', num(t.valuation_pull), '/年'] : []),
    ...(isNum(t.floor) && Math.abs(t.floor) >= 0.0005 ? ['，−95% 截断 ', num(t.floor), '/年'] : []),
    ...fixedPart()];
  if (att.kind === 'cash') return h('p', { class: 'scenarios-attr-read' }, `${name}（${where}）的回报全部是利息收入`, tail);
  if (att.kind === 'treasury') {
    return h('p', { class: 'scenarios-attr-read' }, `${name}（${where}）每年约 `, num(t.income), ' 来自票息、', num(t.y10),
      ' 来自收益率变动带来的重新定价', tail);
  }
  if (att.kind === 'crude_oil') return h('p', { class: 'scenarios-attr-read' }, `${name}（${where}）的回报全部来自油价路径`, tail);
  if (att.method !== 'linear') {
    return h('p', { class: 'scenarios-attr-read' }, `${name}（${where}）：传导部分整块未归因 `, num(t.unattributed), '/年，未解释部分 ',
      num(t.residual), '/年', extra(), tail);
  }
  // channels for the sentence: the two rates read as one "利率", inflation as one "通胀"
  const ch = [];
  const add = (label, ...keys) => {
    const vs = keys.map(k => t[k]).filter(isNum);
    if (vs.length) ch.push({ label, v: vs.reduce((a, b) => a + b, 0) });
  };
  add('油价', 'oil'); add('工业产出', 'ip'); add('通胀', 'cpi_goods', 'cpi_services'); add('利率', 'ff', 'y10');
  add('信用利差', 'spread'); add('美元', 'usd'); add('失业率', 'unrate'); add('劳动收入占比', 'labor_share');
  const known = new Set(['oil', 'ip', 'cpi_goods', 'cpi_services', 'ff', 'y10', 'spread', 'usd', 'unrate', 'labor_share']);
  for (const k of Object.keys(t).filter(k => isFactorTerm(k) && !known.has(k))) add(attrName(k), k);
  // the largest channels by size, then the rest, so the named channels add up to the factor total (the
  // largest gain and the largest loss alone could name a -0.1% channel and skip a -0.8% one)
  const sig = ch.filter(c => Math.abs(c.v) >= 0.0005).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const lead = sig.slice(0, 3);
  const rest = isNum(blk.factor_sum) ? blk.factor_sum - lead.reduce((s, c) => s + c.v, 0) : null;
  const band = Array.isArray(blk.factor_band) && blk.factor_band.length === 2 ? blk.factor_band : null;
  const parts = [`${name}（${where}）`];
  if (lead.length) {
    parts.push('每年约 ');
    lead.forEach((c, i) => parts.push(i ? '、' : '', num(c.v), ` 来自${c.label}`));
    if (isNum(rest) && Math.abs(rest) >= 0.0005) parts.push('，其余因子合计 ', num(rest));
  } else parts.push('几乎没有来自因子的贡献');
  parts.push('；因子合计 ', num(blk.factor_sum), '/年', band ? `（10%–90% 的路径在 ${pct(band[0], 1, true)} ~ ${pct(band[1], 1, true)}）` : '',
    '，未解释部分 ', num(t.residual), '/年', ...extra());
  return h('p', { class: 'scenarios-attr-read' }, parts, tail);
}

function attributionCard(d, state, later, bus) {
  const assets = Object.keys(d.assets || {}).filter(a => d.assets[a]?.attribution);
  const title = '传导归因：每个资产的回报从哪条渠道来';
  const sub = '引擎把每条路径每个月的回报拆成可以相加的几块：β × 模拟的因子变动（已含水平缩放、均值回归、油价回归或锁定、因子漂移）、常数项、估值调整、情景覆盖、未解释部分和收入；'
    + `下面是所有路径的平均，单位 %/年（月均 ×12，严格相加）；最后一项“${RECON_LABEL}”把算术合计换算到上表的中位数年化回报（复利、波动拖累，以及路径中位数和均值的差别，短阶段里可正可负）。`;
  if (!assets.length) {
    return card({ title, sub }, ui.notice('这个结果没有传导归因（由旧版本生成）：内置情景运行 python -m scenariolab.build --only scenarios 后可见，自定义情景在情景实验室里重新运行一次即可。', 'info'));
  }
  const st = state.attr = state.attr || {};
  if (st.for !== d.id) {                       // a new scenario: phase back to the whole path, asset unless picked
    st.for = d.id;
    st.phase = 'all';
    if (!st.picked) st.asset = null;
  }
  if (!assets.includes(st.asset)) {
    const linked = (d.causal_chain?.edges || []).map(e => edgeTerm(d, e)?.aid).find(Boolean);
    st.asset = linked && assets.includes(linked) ? linked : assets.includes('us_equity') ? 'us_equity' : assets[0];
  }
  const nPh = d.assets[st.asset].attribution.phases?.length || 0;
  if (st.phase !== 'all' && !(Number(st.phase) >= 0 && Number(st.phase) < nPh)) st.phase = 'all';

  const box = h('div', { class: 'chart' });
  const reading = h('div'), note = h('p', { class: 'scenarios-note' }), tableBox = h('div', { class: 'scenarios-attr-table' });
  const titleEl = h('span');
  const phaseSeg = h('span');

  const draw = () => {
    const att = d.assets[st.asset].attribution;
    const blk = attrBlock(att, st.phase);
    titleEl.textContent = `${ui.assetName(st.asset)} · ${st.phase === 'all' ? '全程' : (blk?.name || '')}：回报拆分（%/年）`;
    clearBox(box);
    box.style.height = '';
    fill(reading, attrReading(st.asset, att, st.phase));
    if (!blk) { box.append(ui.notice('这个阶段没有归因数据。', 'info')); return; }
    drawWaterfall(box, attrRows(blk, att.kind, st.asset));
    const t = att.total?.terms || {};
    note.textContent = att.kind === 'model'
      ? (att.method === 'linear'
        ? `常数项（${pct(t.intercept, 1, true)}/年）和估值调整（${pct(t.cma, 1, true)}/年）在每个情景、每个阶段都一样；情景之间算术年化的差别来自因子项、情景覆盖${isNum(t.valuation_pull) ? '、估值回归' : ''}和未解释部分。`
          + '因子项用的是引擎模拟出来的因子路径，不是抽到的原始历史变化。'
        : `${att.reason || '传导模型不是同月因子的线性组合'}；常数项包含在“未归因的传导”里。`)
      : att.kind === 'treasury' ? '10 年期美债是结构化资产：每月按上月收益率买入平价债券、按本月收益率重新定价，票息按上月收益率计。'
        : att.kind === 'cash' ? '现金是结构化资产：回报 = 模拟政策利率减今天的利差（联邦基金利率 − 3 个月国库券），没有价格变动。'
          : '原油是结构化资产：回报 = 模拟油价的月度涨跌（不含期货展期收益），由油价视图、漂移和均值回归决定。';
    drawTable(att);
  };

  const drawTable = (att) => {
    const blocks = [['全程', att.total], ...(att.phases || []).map((p, k) => [`${k + 1}. ${shortName(p.name)}`, p])];
    const keys = [];
    for (const [, b] of blocks) for (const k of Object.keys(b?.terms || {})) if (!keys.includes(k)) keys.push(k);
    const factorKeys = att.kind === 'model' ? keys.filter(isFactorTerm) : [];
    const rows = [
      ...factorKeys.map(k => ({ label: attrName(k), get: (b) => b?.terms?.[k] })),
      ...(factorKeys.length ? [{ label: '＝ 因子合计', get: (b) => b?.factor_sum, bold: true },
        { label: '因子合计的路径 10–90% 区间', get: (b) => b?.factor_band, band: true }] : []),
      ...keys.filter(k => !factorKeys.includes(k)).map(k => ({ label: attrName(k, att.kind), get: (b) => b?.terms?.[k] })),
      { label: '＝ 算术年化合计', get: (b) => b?.arithmetic, bold: true },
      { label: RECON_LABEL, get: (b) => b?.reconciliation },
      { label: '＝ 中位数年化回报', get: (b) => b?.cagr_p50, bold: true },
    ];
    const cell = (r, b) => {
      const v = r.get(b);
      if (r.band) return Array.isArray(v) && v.length === 2 && isNum(v[0]) && isNum(v[1]) ? `${sfx(v[0] * 100, 1)}~${sfx(v[1] * 100, 1)}` : '—';
      return h(r.bold ? 'b' : 'span', { class: ui.signCls(v) }, isNum(v) ? pct(v, 1, true) : '—');
    };
    fill(tableBox, table([
      { key: 'label', label: '项目（%/年）', left: true, wrap: true, minWidth: 96 },
      ...blocks.map(([n, b]) => ({ value: (r) => r, label: n, fmt: (r) => cell(r, b) })),
    ], rows, { stack: true }));          // on a phone each row becomes a block of labelled phases
  };

  const pickAsset = ui.select(assets.map(a => ({ value: a, label: ui.assetName(a) })), st.asset, (v) => {
    st.asset = v; st.picked = true; drawPhaseSeg(); draw();
  });
  const drawPhaseSeg = () => {
    const att = d.assets[st.asset].attribution;
    fill(phaseSeg, ui.seg([{ value: 'all', label: '全程' },
      ...(att.phases || []).map((p, k) => ({ value: String(k), label: `${k + 1}. ${shortName(p.name)}` }))], st.phase, (v) => {
      st.phase = v; draw();
      if (bus.redrawChain) bus.redrawChain();
    }));
  };
  drawPhaseSeg();
  later.push([box, draw]);
  return card({ title, sub, actions: h('div', { class: 'scenarios-attr-pick' }, h('span', { class: 'muted' }, '资产'), pickAsset) },
    h('div', { class: 'scenarios-attr-phases' }, phaseSeg),
    reading,
    h('div', { class: 'scenarios-sub' }, titleEl),
    box, note,
    h('details', { class: 'scenarios-more' }, h('summary', {}, '各阶段明细表（%/年）'), tableBox));
}

function phaseSection(d, labels, later) {
  const phases = Array.isArray(d.phases) ? d.phases : [];
  if (!phases.length) return [ui.notice('这个情景没有阶段信息。', 'info')];
  const eraBox = h('div', { class: 'chart' });
  eraBox.style.height = `${(phases.length + 1) * 46 + 80}px`;
  later.push([eraBox, () => {
    const rows = phases.map((p, k) => `阶段 ${k + 1} ${p.name || ''}`).concat(['全程 15 年']);
    const mixes = phases.map(p => p.ep?.era_weights || {}).concat([d.era_composition || {}]);
    const order = (ui.META.raw?.eras || []).map(e => e.id);
    for (const m of mixes) for (const e of Object.keys(m)) if (!order.includes(e)) order.push(e);
    const eras = order.filter(e => mixes.some(m => (m[e] || 0) > 0.0005));
    // a wrapping legend (a scrolling one hides eras at narrow widths): estimate its rows
    const legendW = eras.reduce((t, e) => t + String(ui.eraName(e)).length * 12 + 34, 0);
    const legendRows = Math.max(1, Math.ceil(legendW / Math.max((eraBox.clientWidth || 600) - 40, 200)));
    eraBox.style.height = `${(phases.length + 1) * 46 + 56 + legendRows * 22}px`;
    const chart = charts.bars(eraBox, {
      x: rows, horizontal: true, fmt: 'pct',
      series: eras.map((e, i) => ({ name: ui.eraName(e), stack: 'era', color: ERA_COLORS[e] || charts.palette(i),
        data: mixes.map(m => +(m[e] || 0).toFixed(4)) })),
    });
    chart.setOption({ xAxis: { max: 1 }, grid: { top: 12 + legendRows * 22 }, legend: { type: 'plain' } });
  }]);
  return [
    card({ title: '阶段时间线', sub: '情景被切成几个阶段，每个阶段对少数“因”和关键宏观结果给出视图；灰色阶段没有视图，回到历史常态。' },
      timelineBar(phases, labels),
      h('div', { class: 'scenarios-gloss' }, GLOSSARY.map(([t, x]) => h('div', {}, h('b', {}, t), '：', x)))),
    ...phases.map((ph, k) => phaseCard(ph, k, { labels })),
    card({ title: '各阶段抽到的历史月份来自哪些时代', sub: '每一行是一个阶段（最后一行是整条 15 年路径）的历史月份按时代的占比：视图越强，占比越偏离历史平均，这就是情景“像哪个时代”的定量回答。' }, eraBox),
  ];
}

function assetSection(d, state, labels, later) {
  const assets = Object.keys(d.assets || {});
  if (!assets.length) return [ui.notice('结果里没有资产路径。', 'info')];
  if (!assets.includes(state.asset)) state.asset = assets.includes('us_equity') ? 'us_equity' : assets[0];
  state.kline = state.kline === 'monthly' ? 'monthly' : 'yearly';
  state.fan = state.fan === 'real' ? 'real' : 'nominal';

  const phases = Array.isArray(d.phases) ? d.phases : [];
  const klineBox = h('div', { class: 'chart tall' }), fanBox = h('div', { class: 'chart tall' });
  const plotWidth = (el) => Math.max((el.clientWidth || 600) - 80, 100);
  const klineTitle = h('span'), fanTitle = h('span');
  const klineNote = h('p', { class: 'scenarios-note' }), fanNote = h('p', { class: 'scenarios-note' });
  const pick = h('div', { class: 'scenarios-assetpick' });
  let statsBody = null;

  const drawKline = () => {
    const a = d.assets[state.asset] || {};
    const yearly = state.kline === 'yearly';
    const data = yearly ? a.candles_yearly : a.candles_monthly;
    klineTitle.textContent = `${ui.assetName(state.asset)} · 15 年${yearly ? '年线' : '月线'} K 线（一条代表路径）`;
    clearBox(klineBox);
    if (!Array.isArray(data) || !data.length) { klineBox.append(ui.notice('没有 K 线数据。', 'info')); return; }
    const at = (lab) => (yearly ? String(lab).slice(0, 4) : lab);
    const cats = data.map(row => row[0]);
    const marks = phases.map((ph, k) => {
      const { a: s, m } = phaseSpan(ph, labels);
      const from = at(labels[s + 1]), to = at(labels[Math.min(s + m, labels.length - 1)]);
      if (!from || !to) return null;
      const span = Math.max(cats.indexOf(to) - cats.indexOf(from) + 1, 1);
      return { from, to, label: bandLabel(ph.name, k, (plotWidth(klineBox) * span) / Math.max(cats.length, 1)), color: rgba(phaseColor(ph, k), 0.07) };
    }).filter(Boolean);
    // yearly candles get no moving average (a 12-period MA would be a 12-year one); the price axis is whole numbers
    charts.candles(klineBox, { data, name: '代表路径', freq: yearly ? 'yearly' : 'monthly', ma: yearly ? [] : [12],
      fmt: (v) => Number(v).toFixed(1), axisFmt: 'int', markAreas: marks });
    const end = last(data)?.[2];
    const n = d.n_paths ? ` ${d.n_paths} 条` : '全部';
    klineNote.textContent = `这是编号 #${isNum(d.representative_path) ? d.representative_path : '—'} 的模拟路径：在美股大盘的${n}路径里，它的 15 年终值和最大回撤的排名都最接近中位数；所有资产用同一条路径（同一组历史月份）。` +
      `期初 100，期末 ${isNum(end) ? end.toFixed(0) : '—'}（总回报，股息/利息再投资）；${yearly ? '首尾两根年线只覆盖起止年份的部分月份；' : ''}` +
      '影线由布朗桥按该资产的典型月波动补出。K 线只是众多可能中的一条，分布看扇形图。';
  };

  const drawFan = () => {
    const a = d.assets[state.asset] || {};
    const real = state.fan === 'real';
    const src = real ? a.fan_real : a.fan;
    fanTitle.textContent = `${ui.assetName(state.asset)} · ${real ? '实际（扣除通胀）' : '名义'}财富分布扇形图`;
    clearBox(fanBox);
    if (!src || !Array.isArray(src.p50)) { fanBox.append(ui.notice('没有扇形图数据。', 'info')); return; }
    const scale = (xs) => (xs || []).map(v => (isNum(v) ? +(v * 100).toFixed(2) : null));
    const bands = { p5: scale(src.p5), p25: scale(src.p25), p50: scale(src.p50), p75: scale(src.p75), p95: scale(src.p95) };
    const extra = [];
    const cm = a.candles_monthly;
    if (!real && Array.isArray(cm) && cm.length === labels.length - 1) {
      extra.push({ name: 'K 线代表路径', data: [100, ...cm.map(c => c[2])], dashed: true, color: '#495057' });
    }
    charts.fan(fanBox, { x: labels, bands, extra, fmt: (v) => Number(v).toFixed(0), markAreas: phaseAreas(phases, labels, { width: plotWidth(fanBox) }),
      color: charts.assetColor(state.asset), name: '中位数' });
    const s = a.stats || {};
    const lo = last(bands.p5), mid = last(bands.p50), hi = last(bands.p95);
    fanNote.textContent = `期初 = 100。15 年后（${last(labels) || ''}）中位数 ${isNum(mid) ? mid.toFixed(0) : '—'}，5%–95% 区间 ${isNum(lo) ? lo.toFixed(0) : '—'} ~ ${isNum(hi) ? hi.toFixed(0) : '—'}；` +
      (real ? `实际年化 p5 / p50 / p95：${pct(s.real_cagr_p5, 1)} / ${pct(s.real_cagr_p50, 1)} / ${pct(s.real_cagr_p95, 1)}。实际值按每条路径自己的 CPI 折算，代表路径只画在名义图上。`
        : `名义年化 p5 / p50 / p95：${pct(s.cagr_p5, 1)} / ${pct(s.cagr_p50, 1)} / ${pct(s.cagr_p95, 1)}。浅色带是阶段。`);
  };

  const markRow = () => {
    if (!statsBody) return;
    [...statsBody.querySelectorAll('tbody tr')].forEach((tr, i) => tr.classList.toggle('scenarios-row-on', assets[i] === state.asset));
  };
  const drawPick = () => {
    pick.innerHTML = '';
    for (const a of assets) {
      pick.append(h('button', { class: a === state.asset ? 'on' : '', onclick: () => choose(a) }, ui.assetName(a)));
    }
  };
  function choose(a) { state.asset = a; drawPick(); markRow(); drawKline(); drawFan(); }
  drawPick();

  const rows = assets.map(a => ({ asset: a, s: d.assets[a]?.stats || {} }));
  const p1 = (x) => pct(x, 1);
  statsBody = ui.table([
    { key: 'asset', label: '资产', fmt: (v) => ui.assetName(v) },
    { value: (r) => r.s.cagr_p5, label: '名义 p5', fmt: p1, cls: ui.signCls },
    { value: (r) => r.s.cagr_p50, label: '名义 p50', fmt: p1, cls: ui.signCls },
    { value: (r) => r.s.cagr_p95, label: '名义 p95', fmt: p1, cls: ui.signCls },
    { value: (r) => r.s.real_cagr_p5, label: '实际 p5', fmt: p1, cls: ui.signCls },
    { value: (r) => r.s.real_cagr_p50, label: '实际 p50', fmt: p1, cls: ui.signCls },
    { value: (r) => r.s.real_cagr_p95, label: '实际 p95', fmt: p1, cls: ui.signCls },
    { value: (r) => r.s.vol, label: '年化波动', fmt: p1 },
    { value: (r) => r.s.max_dd_p50, label: '最大回撤中位数', fmt: (x) => pct(x, 0), cls: (x) => (isNum(x) && x < 0 ? 'neg' : '') },
    { value: (r) => r.s.income_yield, label: '收入回报', fmt: (x) => pct(x, 2) },
  ], rows, { onRowClick: (r) => choose(r.asset) });
  markRow();

  later.push([klineBox, drawKline], [fanBox, drawFan]);
  return [
    card({ title: '选择资产', sub: 'K 线、扇形图和下表都对应同一批模拟路径；切换资产不会重新模拟。' }, pick),
    h('div', { class: 'scenarios-pair' },
      card({ title: klineTitle, sub: 'K 线跟随一条代表路径（不是平均值），用来感受 15 年里涨跌的节奏；浅色底是各阶段，阶段的视图决定这段时间抽哪些历史月份。',
        actions: ui.seg([{ value: 'yearly', label: '年线' }, { value: 'monthly', label: '月线' }], state.kline, (v) => { state.kline = v; drawKline(); }) },
      klineBox, klineNote),
      card({ title: fanTitle, sub: '所有路径的 5%–95% 分位带（深色为 25%–75%），期初 = 100：分布宽度就是情景内部的不确定性，虚线是 K 线图里那条代表路径。',
        actions: ui.seg([{ value: 'nominal', label: '名义' }, { value: 'real', label: '实际' }], state.fan, (v) => { state.fan = v; drawFan(); }) },
      fanBox, fanNote)),
    card({ title: '各资产 15 年年化回报（CAGR）分布', sub: 'p5 / p50 / p95 是所有路径的 5% / 50% / 95% 分位；实际 = 按每条路径自己的 CPI 折算；收入回报是股息或利息的年化平均。点击一行切换上方图表。' },
      statsBody),
  ];
}

function macroSection(d, labels, later) {
  const phases = Array.isArray(d.phases) ? d.phases : [];
  const areas = phaseAreas(phases, labels, { names: false });
  const cards = MACRO.filter(m => !m.optional || Array.isArray(d.macro?.[m.id]?.fan?.p50)).map(m => {
    const fanData = d.macro?.[m.id]?.fan;
    if (!fanData || !Array.isArray(fanData.p50)) return card({ title: m.title, sub: m.sub }, ui.notice('没有这条宏观路径。', 'info'));
    // P/D without a fair value: the state is only recorded (κ=0), see pdCard
    if (m.id === 'pd' && !Array.isArray(d.macro?.pd_fair?.fan?.p50)) return pdCard(fanData, labels, areas, later);
    const box = h('div', { class: 'chart short' });
    const fmt = m.unit === '$' ? (v) => `$${Number(v).toFixed(0)}` : m.unit === '倍' ? (v) => `${Number(v).toFixed(0)}`
      : (v) => `${Number(v).toFixed(1)}%`;
    const fmtTip = m.unit === '$' ? (v) => `$${Number(v).toFixed(1)}` : m.unit === '倍' ? (v) => `${Number(v).toFixed(1)} 倍`
      : (v) => `${Number(v).toFixed(2)}%`;
    // P/D: the engine's fair value P/D* (moves with the simulated real yield, profit share and expected
    // inflation; prices are pulled toward it) and the 1973+ median the CMA anchors to
    const fair = m.id === 'pd' && Array.isArray(d.macro?.pd_fair?.fan?.p50) ? d.macro.pd_fair.fan.p50 : null;
    const med = m.id === 'pd' ? ui.META.raw?.cma?.pd_median : null;
    const extra = [
      fair ? { name: '合理估值 P/D*（中位数）', data: fair, dashed: true, color: '#e8590c' } : null,
      isNum(med) ? { name: '1973 年以来中位数', data: labels.map(() => med), dashed: true, color: '#868e96' } : null,
    ].filter(Boolean);
    later.push([box, () => {
      charts.fan(box, { x: labels, bands: fanData, fmt: fmtTip, axisFmt: fmt, markAreas: areas, legend: false, color: '#3b5bdb', extra })
        .setOption({ grid: { top: 16, left: 52 } });
    }]);
    const p0 = fanData.p50[0], pend = last(fanData.p50);
    const sub = fair ? `${m.sub}橙色虚线是引擎的合理估值 P/D*：随模拟的实际利率、利润占比和通胀预期变化，股价按一定速度被拉向它。` : m.sub;
    return card({ title: m.title, sub }, box,
      h('p', { class: 'scenarios-note' }, `起点 ${isNum(p0) ? fmtTip(p0) : '—'} → 15 年后中位数 ${isNum(pend) ? fmtTip(pend) : '—'}` +
        `（5%–95%：${isNum(last(fanData.p5)) ? fmtTip(last(fanData.p5)) : '—'} ~ ${isNum(last(fanData.p95)) ? fmtTip(last(fanData.p95)) : '—'}）` +
        (fair && isNum(fair[0]) && isNum(last(fair)) ? `；合理估值 P/D* 从 ${fmtTip(fair[0])} 到 ${fmtTip(last(fair))}` : '')));
  });
  return [h('div', { class: 'grid cols-3' }, cards)];
}

// The P/D the paths imply. The engine tracks the valuation state at κ=0 (config.VALUATION: track on,
// pull off): each path's P/D follows from its own price and dividends, nothing pulls it toward a fair
// value, so it changes no return; there is no macro.pd_fair. Today's level and the 1973+ median come
// from meta.cma; the percentiles of both ends against the 1973+ history come from valuation.json
// (fetched once and cached by api.js; without it the card shows the levels only).
function pdCard(fanData, labels, areas, later) {
  const cma = ui.META.raw?.cma || {};
  const since = String(ui.META.raw?.model_start || '1973').slice(0, 4);
  const med = isNum(cma.pd_median) ? cma.pd_median : null;
  const p0 = fanData.p50[0], pend = last(fanData.p50), lo = last(fanData.p5), hi = last(fanData.p95);
  const years = labels.length > 1 ? (labels.length - 1) / 12 : 15;
  const tip = (v) => `${Number(v).toFixed(1)} 倍`;
  const times = (v, dd = 1) => (isNum(v) ? `${v.toFixed(dd)} 倍` : '—');
  const box = h('div', { class: 'chart' });
  later.push([box, () => {
    charts.fan(box, { x: labels, bands: fanData, fmt: tip, axisFmt: (v) => `${Number(v).toFixed(0)}`, markAreas: areas, legend: false, color: '#3b5bdb',
      extra: med === null ? [] : [{ name: `${since} 年以来中位数`, data: labels.map(() => med), dashed: true, color: '#868e96' }] })
      .setOption({ grid: { top: 16, left: 52 } });
  }]);
  const pctNow = h('div', { class: 's' }), pctEnd = h('div', { class: 's' });
  const fact = (l, v, s) => h('div', {}, h('div', { class: 'l' }, l), h('div', { class: 'v' }, v), s || null);
  const ann = isNum(p0) && isNum(pend) && p0 > 0 && pend > 0 ? Math.log(pend / p0) / years : null;
  const facts = h('div', { class: 'scenarios-facts' },
    fact('今天（起点）', times(p0), pctNow),
    fact(`${Math.round(years)} 年后中位数`, times(pend), pctEnd),
    fact(`${Math.round(years)} 年后 5%–95%`, isNum(lo) && isNum(hi) ? `${lo.toFixed(0)} ~ ${hi.toFixed(0)} 倍` : '—'),
    fact(`${since} 年以来中位数`, times(med)),
    fact('估值变化（按中位数，年化）', ann === null ? '—' : pct(ann, 1, true),
      h('div', { class: 's' }, `ln(${Math.round(years)} 年后中位数 ÷ 今天) ÷ ${Math.round(years)}：估值对年化价格回报的大致贡献`)));
  // percentiles: the share of months since the fair-value regression's start with a lower P/D
  api.get('valuation').then((val) => {
    if (!box.isConnected) return;
    const D = val?.data || {};
    const dates = Array.isArray(D.dates) ? D.dates : [];
    const i0 = Math.max(0, dates.indexOf(val?.regression?.sample?.[0]));
    const hist = (D.pd || []).slice(i0).filter(isNum);
    if (!hist.length) return;
    const rank = (x) => (isNum(x) ? `${since} 年以来 ${pct(hist.filter(y => y < x).length / hist.length, 0)} 分位` : '');
    pctNow.textContent = rank(p0);
    pctEnd.textContent = rank(pend);
  }).catch(() => {});
  return card({ cls: 'span-all', title: '美股估值 P/D：这个情景的路径隐含的估值',
    sub: '引擎在每条路径上用模拟的价格和股息记录 P/D（价格 ÷ 过去 12 个月股息）：κ=0，只记录，不把价格拉向任何公允值，所以它不改变任何回报。'
      + `它回答的是：这个情景的价格路径和股息路径合起来，意味着 ${Math.round(years)} 年后估值有多高。浅色带是各阶段，灰色虚线是 ${since} 年以来的中位数。`,
    actions: h('span', { class: 'scenarios-chips', style: { marginTop: 0 } }, ui.chip('仅记录，不做估值回归', 'accent'),
      h('a', { href: '#valuation?scenarios' }, '估值与盈利 →')) },
  h('div', { class: 'scenarios-pdgrid' }, box, facts));
}

function corrSection(d, state, later) {
  const names0 = d.corr?.assets || [], post = d.corr?.matrix;
  if (!names0.length || !Array.isArray(post)) return [ui.notice('结果里没有相关性矩阵。', 'info')];
  // align the prior to the posterior's asset order
  const pa = d.corr_prior?.assets || [], pm = d.corr_prior?.matrix;
  const idx = names0.map(a => pa.indexOf(a));
  const prior = Array.isArray(pm) ? names0.map((_, r) => names0.map((__, c) => (idx[r] >= 0 && idx[c] >= 0 ? pm[idx[r]]?.[idx[c]] : null))) : null;
  const names = names0.map(a => ui.assetName(a));
  state.corr = state.corr === 'diff' ? 'diff' : 'post';
  const postBox = h('div', { class: 'chart' }), priorBox = h('div', { class: 'chart' });
  postBox.style.height = priorBox.style.height = '470px';
  const drawPost = () => corrChart(postBox, { names, post, prior, mode: state.corr });
  later.push([postBox, drawPost]);
  if (prior) later.push([priorBox, () => corrChart(priorBox, { names, post, prior, mode: 'prior' })]);

  const pairs = [];
  if (prior) {
    for (let r = 0; r < names0.length; r++) {
      for (let c = r + 1; c < names0.length; c++) {
        const a = post[r]?.[c], b = prior[r]?.[c];
        if (isNum(a) && isNum(b)) pairs.push({ pair: `${names[r]} × ${names[c]}`, prior: b, post: a, diff: a - b });
      }
    }
    pairs.sort((x, y) => Math.abs(y.diff) - Math.abs(x.diff));
  }
  return [
    h('div', { class: 'scenarios-pair' },
      card({ title: state.corr === 'diff' ? '相关性变化（情景 − 历史）' : '情景下的资产相关性（后验）',
        sub: '模拟月度回报之间的相关系数；粗框标出与历史相差 ≥ 0.10 的资产对，切到“差值”直接看情景把哪些联动改强或改弱。',
        actions: ui.seg([{ value: 'post', label: '后验' }, { value: 'diff', label: '差值' }], state.corr, (v) => {
          state.corr = v;
          const t = postBox.closest('.card')?.querySelector('.card-title');
          if (t) t.textContent = v === 'diff' ? '相关性变化（情景 − 历史）' : '情景下的资产相关性（后验）';
          drawPost();
        }) }, ui.wideChart(postBox, 520)),          // 13 x 13 cells: a phone scrolls the matrix sideways
      card({ title: '历史相关性（先验，1973 年以来）', sub: '所有历史月份等权的相关系数，作为对照：两图的差别来自情景抽到的历史月份不同（例如通胀时期股债同涨同跌）。' },
        prior ? ui.wideChart(priorBox, 520) : ui.notice('结果里没有先验相关性（corr_prior）。', 'info'))),
    prior ? card({ title: '情景改变最多的资产联动', sub: '按 |后验 − 先验| 排序；股债相关由负转正意味着债券在这个情景里不再是股票的对冲。' },
      table([
        { key: 'pair', label: '资产对' },
        { key: 'prior', label: '历史（先验）', fmt: f2 },
        { key: 'post', label: '情景（后验）', fmt: f2 },
        { key: 'diff', label: '变化', fmt: (x) => (isNum(x) ? sfx(x, 2) : '—'), cls: (x) => (Math.abs(x) >= 0.1 ? 'scenarios-changed' : '') },
      ], pairs.slice(0, 8))) : null,
  ];
}

function calibrationSection(d, later) {
  const list = Array.isArray(d.calibration) ? d.calibration : d.calibration ? [d.calibration] : [];
  const out = [];
  for (const c of list) {
    const imp = c?.impact;
    if (!imp) continue;
    const groups = Array.isArray(imp.groups) ? imp.groups : [];
    const pct2 = (x) => pct(x, 2, true);
    const phaseNo = isNum(c.phase) ? c.phase + 1 : '—';
    out.push(h('div', { class: 'section-title' }, `${c.title || '人员收入对宏观经济的影响'} · 阶段 ${phaseNo}「${c.phase_name || ''}」（${c.months || '—'} 个月）`));

    const box = h('div', { class: 'chart short' });
    later.push([box, () => {
      const chart = charts.bars(box, {
        x: groups.map(g => g.name || g.id), horizontal: true, fmt: 'pct2',
        series: [
          { name: '劳动收入渠道（工资与岗位）', stack: 'c', color: '#e8590c', data: groups.map(g => +(g.contribution_labor || 0).toFixed(5)) },
          { name: '股票财富渠道（财富效应）', stack: 'c', color: '#3b5bdb', data: groups.map(g => +(g.contribution_wealth || 0).toFixed(5)) },
        ],
        markLine: [{ xAxis: 0 }],
      });
      chart.setOption({ grid: { top: 34 }, series: [{ markLine: { label: { show: false } } }] });
    }]);

    const kpis = h('div', { class: 'scenarios-kpis scenarios-kpis2' },
      ui.kpi('总消费变化', pct2(imp.consumption_change), `劳动 ${pct2(imp.consumption_change_labor)} · 财富 ${pct2(imp.consumption_change_wealth)}`, ui.signCls(imp.consumption_change)),
      ui.kpi('实际 GDP 一阶影响', pct2(imp.real_gdp_change), `约每年 ${isNum(imp.real_growth_per_year_pp) ? sfx(imp.real_growth_per_year_pp, 2) : '—'} 个百分点`, ui.signCls(imp.real_gdp_change)),
      ui.kpi('失业率累计变化', isNum(imp.unemployment_change_pp) ? `${sfx(imp.unemployment_change_pp, 2)}pp` : '—',
        `直接失业 ${isNum(imp.unemployment_direct_pp) ? sfx(imp.unemployment_direct_pp, 2) : '—'} · 奥肯 ${isNum(imp.unemployment_okun_pp) ? sfx(imp.unemployment_okun_pp, 2) : '—'}`,
        isNum(imp.unemployment_change_pp) ? ui.goodCls(imp.unemployment_change_pp <= 0) : ''),
      ui.kpi('劳动收入占比（对数变化）', pct2(imp.labor_share_log_change), `劳动收入 ${pct2(imp.labor_income_change)}`, ui.signCls(imp.labor_share_log_change)));

    const base = c.base || {};
    const summaryCard = card({ title: '计算结果', sub: '三组家庭的劳动收入冲击和股票财富变化 → 总消费、增长、失业和劳动收入占比（一阶会计推算）。' },
      c.text ? h('p', { style: { margin: 0, color: 'var(--ink-2)', lineHeight: 1.7 } }, c.text) : null, kpis,
      h('p', { class: 'scenarios-note' }, `无冲击基准（最近 12 个月均值）：工业产出同比 ${pp(base.ip_yoy, 2)}，服务通胀 ${pp(base.services_yoy, 2)}；阶段长度 ${isNum(imp.years) ? imp.years.toFixed(1) : '—'} 年。`));
    const chartCard = card({ title: '各收入组对总消费变化的贡献', sub: '橙色是工资和岗位（劳动收入渠道），蓝色是股票财富效应：K 型结构下前 10% 的财富效应可以抵消中层白领收入下降对消费的拖累，但也让消费依赖股市。' }, box);

    const groupTable = ui.table([
      { key: 'name', label: '收入组' },
      { key: 'consumption_share', label: '消费占比', fmt: (x) => pct(x, 0) },
      { key: 'labor_income_shock', label: '劳动收入冲击', fmt: pct2, cls: ui.signCls },
      { key: 'mpc_labor', label: '边际消费倾向', fmt: (x) => ui.fmtNum(x, 2) },
      { key: 'labor_income_ratio', label: '劳动收入/消费', fmt: (x) => ui.fmtNum(x, 2) },
      { key: 'equity_ownership', label: '持股份额', fmt: (x) => pct(x, 0) },
      { key: 'consumption_change_labor', label: '本组消费：劳动渠道', fmt: pct2, cls: ui.signCls },
      { key: 'consumption_change_wealth', label: '本组消费：财富渠道', fmt: pct2, cls: ui.signCls },
      { key: 'contribution', label: '对总消费的贡献', fmt: pct2, cls: ui.signCls },
      { key: 'unemployment_direct_pp', label: '直接失业', fmt: (x) => (isNum(x) ? `${sfx(x, 2)}pp` : '—') },
    ], groups);

    const ec = c.engine_check || {};
    const checkRows = [
      { item: '股票财富变化（本阶段累计）', assumed: pct2(ec.equity_wealth_change_assumed),
        engine: `美股价格回报中位数 ${pct2(ec.us_equity_price_return_p50)}（总回报 ${pct2(ec.us_equity_total_return_p50)}）` },
      { item: '失业率变化', assumed: isNum(ec.unemployment_change_assumed_pp) ? `${sfx(ec.unemployment_change_assumed_pp, 2)}pp` : '—',
        engine: `中位数 ${isNum(ec.unemployment_change_p50_pp) ? sfx(ec.unemployment_change_p50_pp, 2) : '—'}pp（均值 ${isNum(ec.unemployment_change_mean_pp) ? sfx(ec.unemployment_change_mean_pp, 2) : '—'}pp）` },
      { item: '劳动收入占比（对数变化）', assumed: pct2(ec.labor_share_log_change_assumed), engine: `中位数 ${pct2(ec.labor_share_log_change_p50)}` },
    ];
    const notes = c.override_notes || {};
    const ovDrift = asDict(c.overrides?.factor_drift, 'factor');
    const mapCard = card({ title: '计算结果怎么进入引擎', sub: '需求效应变成熵池视图（决定抽哪些历史月份），没有历史先例的分配效应变成有上限的因子漂移；资产价格仍全部由引擎生成。' },
      h('ul', { class: 'list' }, (c.engine_mapping || []).map(t => h('li', {}, t))),
      (c.views || []).length ? h('div', {}, h('div', { class: 'scenarios-sub' }, '写入该阶段的视图'),
        viewsTable(c.views.map(v => ({ v })), null, { numbers: false })) : null,
      Object.keys(ovDrift).length ? h('div', {}, h('div', { class: 'scenarios-sub' }, '写入该阶段的因子漂移'),
        h('ul', { class: 'scenarios-ovr' }, Object.entries(ovDrift).map(([f, x]) => h('li', {}, `${factorLabel(f)} ${driftText(f, x)}${notes[f] ? `：${notes[f]}` : ''}`)))) : null);
    const checkCard = card({ title: '计算器假设 vs 引擎模拟', sub: '同一阶段里引擎实际模拟出来的数值（含抽到的历史月份自带的背景波动）；两者接近说明视图和漂移把计算器的结论落实了。' },
      ui.table([
        { key: 'item', label: '指标' },
        { key: 'assumed', label: '计算器假设' },
        { key: 'engine', label: '引擎模拟', wrap: true },
      ], checkRows));

    const params = imp.params || {}, pnotes = c.param_notes || {};
    const paramRows = Object.entries(params).map(([k, v]) => ({
      key: k,
      value: v && typeof v === 'object' ? Object.entries(v).map(([g, x]) => `${groups.find(gg => gg.id === g)?.name || g} ${x}`).join('；') : String(v),
      note: pnotes[k] || '',
    }));
    out.push(h('div', { class: 'scenarios-stack' }, h('div', { class: 'scenarios-pair' }, summaryCard, chartCard),
      card({ title: '各收入组的冲击与传导', sub: '劳动渠道 = 边际消费倾向 × 劳动收入冲击 × 劳动收入/消费；财富渠道 = 财富效应 × 股票财富变化 × 持股份额 × 股票财富/消费。' },
      groupTable,
      h('details', { class: 'scenarios-more' }, h('summary', {}, '公式、参数来源与局限'),
        h('ul', { class: 'list' }, (c.formulae || []).map(t => h('li', {}, t))),
        paramRows.length ? table([
          { key: 'key', label: '参数', fmt: (x) => h('span', { class: 'mono' }, x) },
          { key: 'value', label: '取值', left: true },
          { key: 'note', label: '说明', wrap: true, fmt: (x) => x || '—' },
        ], paramRows) : null,
        (c.caveats || []).length ? h('ul', { class: 'list', style: { marginTop: '8px' } }, c.caveats.map(t => h('li', { class: 'muted' }, t))) : null)),
      h('div', { class: 'scenarios-pair' }, mapCard, checkCard)));
  }
  return out;
}

function probabilitySection(d, later) {
  const p = d.probability;
  const sub = '情景概率 = 主观先验 × 似然^γ 再归一化；似然只看今天的状态与情景第一阶段会抽到的月份是否吻合，是透明的启发式，不是预测模型。';
  if (!p || !p.z || !Object.keys(p.z).length) {
    const custom = (d.source || 'builtin') !== 'builtin';
    return [card({ title: '概率从哪里来', sub },
      ui.notice(custom ? `这是自定义情景：概率只对内置情景库统一计算（先验 × 似然再归一化），这里没有后验概率。规格里的主观先验 prior_prob = ${pct(d.prior_prob, 0)}。`
        : `结果里没有概率明细（probability.z）：后验概率 ${pct(d.prob, 1)}，主观先验 ${pct(p?.prior ?? d.prior_prob, 1)}。`, 'info'))];
  }
  const states = Object.keys(p.z);
  const name = (s) => ui.META.states[s]?.name || PROB_STATES[s] || s;
  const box = h('div', { class: 'chart' });
  box.style.height = `${states.length * 34 + 60}px`;
  const bound = Math.max(1.5, Math.ceil(Math.max(...states.map(s => Math.abs(p.z[s] || 0))) * 2) / 2);
  later.push([box, () => {
    const chart = charts.bars(box, {
      x: states.map(name), horizontal: true, legend: false, fmt: (v) => Number(v).toFixed(1),
      series: [{ name: 'z', data: states.map(s => ({ value: +(p.z[s] || 0).toFixed(3), itemStyle: { color: Math.abs(p.z[s]) >= 1 ? '#e8590c' : '#3b5bdb' } })) }],
      markLine: [{ xAxis: -1 }, { xAxis: 1 }],
    });
    chart.setOption({ xAxis: { min: -bound, max: bound, interval: 0.5 }, series: [{ markLine: { label: { show: false } } }],
      tooltip: { valueFormatter: (v) => f2(v) } });
  }]);
  const rows = states.map(s => ({ state: s, today: p.today?.[s], mean: p.phase_means?.[s], z: p.z[s] }));
  return [h('div', { class: 'scenarios-pair' },
    card({ title: '概率从哪里来', sub },
      h('div', { class: 'scenarios-kpis' },
        ui.kpi('主观先验', pct(p.prior, 1), '规格作者给的先验（已在情景库内归一化）'),
        ui.kpi('似然 L', isNum(p.likelihood) ? p.likelihood.toFixed(3) : '—', `L = exp(−½·平均 z²)，距离 ${isNum(p.distance) ? p.distance.toFixed(2) : '—'}`),
        ui.kpi('后验概率', pct(p.posterior, 1), '先验 × L^γ，再在情景库内归一化')),
      table([
        { key: 'state', label: '状态', fmt: name },
        { key: 'today', label: '今天（近 6 个月均值）', fmt: (x) => pp(x, 2) },
        { key: 'mean', label: '情景第一阶段均值', fmt: (x) => pp(x, 2) },
        { key: 'z', label: 'z', fmt: (x) => (isNum(x) ? sfx(x, 2) : '—'), cls: (x) => (Math.abs(x) >= 1 ? 'scenarios-changed' : '') },
      ], rows),
      p.method ? h('details', { class: 'scenarios-more' }, h('summary', {}, '完整的计算说明'), h('p', { class: 'muted', style: { margin: 0 } }, p.method)) : null),
    card({ title: '今天与情景开局的距离 z（按状态）', sub: 'z = (今天 − 情景第一阶段均值) / 历史标准差；虚线是 ±1，橙色条说明今天在这个维度上不太像该情景的开局，会压低它的概率。' }, box))];
}

function specSection(d) {
  if (!d.spec) return [];
  let text = '';
  try { text = JSON.stringify(d.spec, null, 2); } catch (_) { text = String(d.spec); }
  return [h('details', { class: 'scenarios-more card' },
    h('summary', {}, '查看引擎实际运行的情景规格（validate_spec 校验之后的 JSON）'),
    h('pre', { class: 'code' }, text))];
}

// ---------------------------------------------------------------- entry point

// Render one scenario detail into root. ctx is the tab context (helpers come from the same
// ui/charts modules, so it is optional); opts.state keeps the asset and toggles across
// re-renders; opts.labLink adds a "modify in the lab" button.
export function renderScenarioDetail(root, d, ctx = {}, opts = {}) {
  ensureStyle('detail', CSS);
  disposeCharts(root);
  root.innerHTML = '';
  root.classList.add('scenarios-detail');
  if (!d || typeof d !== 'object') { root.append(ui.notice('没有情景数据。', 'info')); return; }
  const state = opts.state || {};
  const labels = Array.isArray(d.labels) ? d.labels : [];
  const later = [];
  const bus = {};                        // the attribution card's phase pick re-labels the causal chain
  const section = (t) => h('div', { class: 'section-title' }, t);

  root.append(
    headerCard(d, opts),
    h('div', { style: { height: '16px' } }),
    validationCard(d.validation),
    section('叙事与因果传导'),
    h('div', { class: 'scenarios-stack' }, narrativeCard(d), chainCard(d, later, state, bus), attributionCard(d, state, later, bus)),
    section('分阶段视图：每一阶段像历史上的哪些月份'),
    h('div', { class: 'scenarios-stack' }, phaseSection(d, labels, later)),
    section('资产：K 线与分布（全部由引擎从真实历史月份生成）'),
    h('div', { class: 'scenarios-stack' }, assetSection(d, state, labels, later)),
    section('宏观路径'),
    ...macroSection(d, labels, later),
    section('相关性：情景如何改变资产之间的联动'),
    h('div', { class: 'scenarios-stack' }, corrSection(d, state, later)),
    ...calibrationSection(d, later),
    section('概率'),
    ...probabilitySection(d, later),
    h('div', { style: { height: '16px' } }),
    ...specSection(d),
  );

  // charts after the DOM exists, so every box has its size; one failure stays local
  for (const [el, draw] of later) {
    try { draw(); } catch (err) {
      console.error(err);
      clearBox(el);
      el.style.height = 'auto';
      el.append(ui.notice(`图表渲染失败：${err.message}`, 'error'));
    }
  }
}
