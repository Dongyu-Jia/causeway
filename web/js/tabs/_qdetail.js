// The quarterly stack's scenario detail (季度模型，试运行; SIM_DESIGN §10.3, §4.4, §6.3): what the scenario stated and what
// the model completed, quarter by quarter (badges 文字数字 / 文字定性 / 推算 / 模型补全 / 锚 / 干预), the assets' quarter-end
// fans, the valuation block, the attribution bars, the financial-conditions block, the regime readout on quarterly
// paths, the delivery of the stated values and the rarity statistic. Used by the scenario detail (built-ins), the
// lab (QTable runs) and the forecast tab (badges). The leading underscore keeps the router from treating it as a tab.
//
//   renderQuarterlyDetail(root, run, ctx, opts)   run = GET /api/q/run/<id>'s result
//   runQuarterly(body, ctx, onProgress)           POST /api/q/run, then poll until done (aborts with ctx.signal)
//   badge(item) / cellNode(cell, median)          the shared badge chips
//
// Every text that may come from a narrative, a quote or an LLM is set as text (h(), textContent), never as HTML;
// ECharts tooltips escape every string (ui.esc).
import * as ui from '../ui.js';
import * as charts from '../charts.js';
import * as api from '../api.js';
import { lastSwitch, quarterlyDefault } from './_switch.js';   // 试运行 until the switch gate T-S passes (P9)

const { h, card } = ui;
const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
const fx = (x, d) => { const s = Number(x).toFixed(d); return Number(s) === 0 ? (0).toFixed(d) : s; };
const sfx = (x, d) => { const s = fx(x, d); return Number(s) > 0 ? `+${s}` : s; };
// replaceChildren without the null / undefined / false children a conditional leaves (they would print as text)
export const put = (el, ...kids) => { el.replaceChildren(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false)); return el; };
export const pctS = (x, d = 1) => (isNum(x) ? `${sfx(x * 100, d)}%` : '—');            // a simple return, signed
export const pctLog = (x, d = 1) => (isNum(x) ? `${sfx(Math.expm1(x) * 100, d)}%` : '—');   // a log return as simple %
export const lvl = (x, d = 2) => (isNum(x) ? `${fx(x, d)}%` : '—');
export const prob = (x, d = 0) => (isNum(x) ? `${fx(x * 100, d)}%` : '—');

export const BADGE = { stated: '文字数字', stated_qual: '文字定性', derived: '推算', completed: '模型补全', anchor: '锚', intervention: '干预' };
export const BADGE_HINT = {
  stated: '文字里的数字（或情景作者写下的约束）：模型精确或在给定的标准差内满足它',
  stated_qual: '文字的定性说法，按历史标准差的幅度档换算（小 / 温和 / 明显 / 大幅 / 极端）',
  derived: '对文字里数字的推算（例如把同比路径换成季度变化）',
  completed: '情景没有说：由联合模型从历史联合分布里补全（不是情景的假设）',
  anchor: '长期假设（锚）：这一版季度模型记录它，但稳态先验在拟合时固定',
  intervention: '文字写明的政策干预（do 算子）：同季的宏观不随它补全，市场价格当季反应',
};
export const TRIAL = '试运行';
export const ANECDOTE = '每个十年只有 2–3 个独立窗口，只能当个案';
export const VAL_WORDING = '相当于直接假设股价';

const CSS = `
.qb { display: inline-block; font-size: 11px; line-height: 17px; padding: 0 7px; border-radius: 999px; border: 1px solid;
  margin: 1px 4px 1px 0; white-space: nowrap; cursor: default; }
.qb-stated { background: #e8edff; border-color: #9fb2f5; color: #2f45b0; }
.qb-stated_qual { background: #fff; border-color: #9fb2f5; color: #2f45b0; }
.qb-derived { background: #f3e8ff; border-color: #c4a3f0; color: #6b3fb5; }
.qb-completed { background: #f1f3f5; border-color: #ced4da; color: #6c757d; }
.qb-anchor { background: #fff4e6; border-color: #f0c48a; color: #9a5b00; }
.qb-intervention { background: #343a40; border-color: #343a40; color: #fff; }
.qb-legend { display: flex; flex-wrap: wrap; gap: 4px 10px; font-size: 12px; color: var(--ink-3); margin: 4px 0 8px; }
.qb-legend span.qb { margin-right: 4px; }
.qd-cell { min-width: 0; }
.qd-med { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--ink); }
.qd-med small { font-weight: 400; color: var(--ink-3); }
.qd-badges { margin-top: 2px; line-height: 1.2; }
.qd-trial { border-left: 4px solid #f0c48a; }
.qd-head { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 8px; }
.qd-kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(190px, 100%), 1fr)); gap: 10px; }
.qd-belief { margin: 0; padding-left: 18px; font-size: 13px; }
.qd-belief li { margin: 5px 0; }
.qd-belief .qd-quote { display: block; color: var(--ink-3); font-size: 12px; margin-top: 2px; }
.qd-bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 10px; }
.qd-note { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; }
.qd-stack { display: grid; gap: 16px; }
.qd-two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr)); gap: 16px; }
.qd-status { font-size: 12px; color: var(--ink-3); display: inline-flex; gap: 8px; align-items: center; }
.qd-status.err { color: var(--bad); }
.qd-t table.tbl th { white-space: normal; line-height: 1.35; vertical-align: bottom; }
.qd-t table.tbl td { vertical-align: top; }
details.qd-more > summary { cursor: pointer; color: var(--accent); font-size: 13px; }
.qd-maint { margin-top: 8px; font-size: 12px; color: var(--ink-3); }
.qd-maint summary { cursor: pointer; }
.qd-maint code, .qd-maint .qd-ver { overflow-wrap: anywhere; word-break: break-all; }
.page[data-tab="lab"] .card-actions select, .page[data-tab="scenarios"] .qd-trial .card-actions select, .qd-scope .card-actions select { max-width: 100%; min-width: 0; }
`;

export function ensureQStyle() {
  let el = document.getElementById('style-qdetail');
  if (!el) { el = document.createElement('style'); el.id = 'style-qdetail'; document.head.append(el); }
  if (!el.textContent.includes('/* qdetail */')) el.textContent += `/* qdetail */\n${CSS}`;
}

// ---------------------------------------------------------------- badges and cells

// One stated item as a chip: the badge word, its statement and quote on hover (text, never HTML).
export function badge(item) {
  const kind = item?.badge && BADGE[item.badge] ? item.badge : 'completed';
  const title = [item?.text, item?.quote ? `原文：「${item.quote}」` : '', BADGE_HINT[kind]].filter(Boolean).join('\n');
  return h('span', { class: `qb qb-${kind}`, title }, BADGE[kind]);
}
export const completedBadge = (why) => h('span', { class: 'qb qb-completed', title: why || BADGE_HINT.completed }, BADGE.completed);

export function badgeLegend(kinds = ['stated', 'stated_qual', 'derived', 'completed', 'anchor', 'intervention']) {
  return h('div', { class: 'qb-legend' }, kinds.map(k => h('span', { title: BADGE_HINT[k] },
    h('span', { class: `qb qb-${k}` }, BADGE[k]), BADGE_HINT[k].split('：')[0].split('（')[0])));
}

// A channel's cell: the engine's median on top, then the badge of every stated item (or 模型补全).
export function cellNode(cell, median, sub) {
  const items = cell?.items || [];
  return h('div', { class: 'qd-cell' },
    h('div', { class: 'qd-med' }, median ?? '—', sub ? h('small', {}, ` ${sub}`) : null),
    h('div', { class: 'qd-badges' }, items.length ? items.map(badge)
      : completedBadge(cell?.stance === '未说明' ? '金融条件未说明：信用利差由模型补全。历史检验里补全的信用利差与无条件抽样相当，而资产结果主要取决于它' : null)));
}

// 「你必须相信什么」: the distinct stated items of the quarters, with their badges and quotes.
export function beliefList(rows, { max = 40 } = {}) {
  const seen = new Map();
  for (const r of rows || []) {
    for (const [key, c] of Object.entries(r.cells || {})) {
      for (const it of c.items || []) {
        const k = `${key}|${it.text}`;
        if (!seen.has(k)) seen.set(k, { ...it, qs: [r.q] });
        else seen.get(k).qs.push(r.q);
      }
    }
    for (const it of [...(r.other || []), ...(r.anchors || [])]) {
      const k = `other|${it.text}`;
      if (!seen.has(k)) seen.set(k, { ...it, qs: [r.q] });
      else seen.get(k).qs.push(r.q);
    }
  }
  const list = [...seen.values()].slice(0, max);
  if (!list.length) return h('p', { class: 'muted', style: { margin: 0 } }, '这几个季度情景什么都没有说：所有因子都由模型补全。');
  return h('ul', { class: 'qd-belief' }, list.map(it => h('li', {}, badge(it),
    it.constraint || /第 [\d–、]+ 季/.test(it.text) ? it.text : `第 ${span(it.qs)} 季：${it.text}`,
    it.quote ? h('span', { class: 'qd-quote' }, `原文：「${it.quote}」`) : null)));
}

export function span(qs) {
  const s = [...new Set(qs)].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < s.length; i++) {
    let j = i;
    while (j + 1 < s.length && s[j + 1] === s[j] + 1) j++;
    out.push(i === j ? `${s[i]}` : `${s[i]}–${s[j]}`);
    i = j;
  }
  return out.join('、');
}

// ---------------------------------------------------------------- the quarter fan (p10 / p50 / p90, optional reference)

// x: labels (the first is the start, value 0); band: {p10, p50, p90} arrays of the same length as x; ref: the same
// for a reference band (dashed); fmt: value -> text for the axis and tooltip.
export function quarterFan(el, { x, band, ref, name = '情景加权', refName = '无情景基准', fmt = (v) => pctS(v, 0), color = '#3b5bdb' }) {
  const chart = charts.mount(el);
  const lo = band.p10, mid = band.p50, hi = band.p90;
  const diff = lo.map((v, i) => (isNum(v) && isNum(hi[i]) ? hi[i] - v : null));
  const series = [
    { name: '_lo', type: 'line', data: lo, stack: 'qfan', symbol: 'none', lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
    { name: '_band', type: 'line', data: diff, stack: 'qfan', stackStrategy: 'all', symbol: 'none', lineStyle: { opacity: 0 }, silent: true,
      areaStyle: { color, opacity: 0.16 }, tooltip: { show: false } },
    { name, type: 'line', data: mid, symbol: x.length > 14 ? 'none' : 'circle', symbolSize: 5, lineStyle: { width: 2, color }, itemStyle: { color }, z: 3 },
  ];
  if (ref) {
    series.push({ name: refName, type: 'line', data: ref.p50, symbol: 'none', lineStyle: { width: 1.4, type: 'dashed', color: '#868e96' }, itemStyle: { color: '#868e96' } },
      { name: `${refName} 10%`, type: 'line', data: ref.p10, symbol: 'none', lineStyle: { width: 1, type: 'dotted', color: '#adb5bd' }, itemStyle: { color: '#adb5bd' } },
      { name: `${refName} 90%`, type: 'line', data: ref.p90, symbol: 'none', lineStyle: { width: 1, type: 'dotted', color: '#adb5bd' }, itemStyle: { color: '#adb5bd' } });
  }
  const t = (v) => ui.esc(isNum(v) ? fmt(v) : '—');
  chart.setOption({
    legend: { top: 0, data: [name, ...(ref ? [refName] : [])], textStyle: { fontSize: 12 } },
    grid: { left: 52, right: 30, top: 32, bottom: 28 },
    tooltip: { trigger: 'axis', confine: true, formatter: (ps) => {
      const i = (Array.isArray(ps) ? ps : [ps]).find(p => Number.isInteger(p?.dataIndex))?.dataIndex;
      if (i === undefined) return '';
      const rows = [`<b>${ui.esc(x[i])}</b>`, `${ui.esc(name)}：90% ${t(hi[i])} · <b>中位 ${t(mid[i])}</b> · 10% ${t(lo[i])}`];
      if (ref) rows.push(`${ui.esc(refName)}：90% ${t(ref.p90[i])} · 中位 ${t(ref.p50[i])} · 10% ${t(ref.p10[i])}`);
      return rows.join('<br/>');
    } },
    xAxis: { type: 'category', data: x, boundaryGap: false, axisLabel: { color: '#7a8396', fontSize: 11 }, axisLine: { lineStyle: { color: '#e3e6ec' } } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: '#7a8396', fontSize: 11, formatter: (v) => fmt(v) }, splitLine: { lineStyle: { color: '#eef0f4' } } },
    series,
  });
  return chart;
}

// ---------------------------------------------------------------- running the quarterly stack

// POST /api/q/run, then poll GET /api/q/run/<id> until it is done; onProgress(text) gets the status line.
export async function runQuarterly(body, { signal } = {}, onProgress = () => {}) {
  const t0 = Date.now();
  const job = await api.post('q/run', body);
  let st = job;
  while (!signal?.aborted) {
    if (st.status === 'done' && st.result) return st.result;
    if (st.status === 'error') throw new Error(st.error || '季度模型运行出错');
    onProgress(`${st.status === 'queued' ? '排队中' : '季度模型运行中'}（${Math.round((Date.now() - t0) / 1000)} 秒）`);
    await new Promise(r => setTimeout(r, 700));
    if (signal?.aborted) break;
    st = await api.get(`q/run/${job.id}`, { fresh: true });
  }
  throw new DOMException('aborted', 'AbortError');
}

let VALUATION = null;          // GET /api/q/valuation, once per page load
export function valuationView() {
  if (!VALUATION) VALUATION = api.get('q/valuation').catch(() => null);
  return VALUATION;
}
// The percentile (0..1) of a CAPE value in the CAPE history's percentile table (cape_q: percentiles 0..100).
export function capePercentile(capeQ, v) {
  if (!Array.isArray(capeQ) || capeQ.length < 2 || !isNum(v)) return null;
  if (v <= capeQ[0]) return 0;
  if (v >= capeQ[capeQ.length - 1]) return 1;
  for (let k = 1; k < capeQ.length; k++) {
    if (v <= capeQ[k]) {
      const lo = capeQ[k - 1], hi = capeQ[k];
      return ((k - 1) + (hi > lo ? (v - lo) / (hi - lo) : 0)) / (capeQ.length - 1);
    }
  }
  return 1;
}

// ---------------------------------------------------------------- the detail

const CH_MEDIAN = {
  policy: (r, k) => lvl(r.levels?.ff?.p50?.[k]),
  growth: (r, k) => lvl(r.levels?.unrate?.p50?.[k]),
  inflation: (r, k) => lvl(r.levels?.cpi_yoy?.p50?.[k]),
  oil: (r, k) => (isNum(r.factors?.d_oil?.p50?.[k]) ? `${sfx(r.factors.d_oil.p50[k], 1)}%` : '—'),
  fc: (r, k) => (isNum(r.levels?.spread?.p50?.[k]) ? fx(r.levels.spread.p50[k], 2) : '—'),
  rates: (r, k) => lvl(r.levels?.y10?.p50?.[k]),
  valuation: (r, k) => pctLog(r.valuation?.y_mult?.p50?.[k]),
};
const CH_SUB = { policy: '联邦基金利率', growth: '失业率', inflation: 'CPI 同比', oil: '本季油价（对数）', fc: 'Baa − 10 年', rates: '10 年期', valuation: '本季倍数' };
const PART_COLOR = { stated: '#3b5bdb', completed: '#868e96', drift: '#e8590c', valuation: '#7048e8', residual: '#adb5bd' };
const PART_NAME = { stated: '情景说了什么', completed: '模型补了什么', drift: '历史常态漂移', valuation: '估值假设（相当于直接假设股价）', residual: '残差（中位数）' };
const REGIME_NAME = { expansion: '扩张', inflation: '通胀', deflation: '物价负增长', recession: '衰退', depression: '深度衰退', stagflation: '滞胀' };
const REGIME_COLOR = { expansion: '#a5d8ff', inflation: '#fab005', deflation: '#15aabf', recession: '#9775fa', depression: '#343a40', stagflation: '#b35c00' };
const FOCUS = 40;              // the 40-quarter focus of a 60-quarter scenario (U-16)

const QSTATE = { asset: 'us_equity', span: 8, horizon: '4' };

export function renderQuarterlyDetail(root, run, ctx = {}, opts = {}) {
  ensureQStyle();
  charts.disposeWithin(root);
  root.replaceChildren();
  if (!run || typeof run !== 'object') { root.append(ui.notice('没有季度模型的结果。', 'info')); return; }
  const state = opts.state || QSTATE;
  const later = [];
  const quarters = Array.isArray(run.quarters) ? run.quarters : [];
  const H = run.scenario_quarters || quarters.length;
  const assets = Object.keys(run.assets || {});
  const assetName = (a) => ui.assetName(a);
  if (!assets.includes(state.asset)) state.asset = assets[0] || 'us_equity';
  const section = (t) => h('div', { class: 'section-title' }, t);
  const eq = run.assets?.us_equity?.cum_total || {};
  const at = (arr, k) => (Array.isArray(arr) && k < arr.length ? arr[k] : null);
  const k4 = Math.min(3, quarters.length - 1), kF = Math.min(FOCUS - 1, quarters.length - 1);
  const ann = (logCum, n) => (isNum(logCum) && n > 0 ? Math.expm1(logCum * 4 / n) : null);
  const rar = run.rarity || {};
  const tail = (run.notes || []).find(n => n.code === 'tail');

  // ---- A. header
  const chips = [quarterlyDefault(lastSwitch()) ? ui.chip('季度模型 · 默认', 'good') : ui.chip(`季度模型 · ${TRIAL}`, 'warn'),
    ui.chip('情景条件模拟，不是预测', 'accent'),
    h('span', { class: 'muted', style: { fontSize: '12px' } },
      `${run.info?.n_paths ?? '—'} 条路径 · 起点 ${quarters[0] || '—'} · ${H} 个季度${run.info?.continuation ? `（另加 ${run.info.continuation} 季无陈述的延续）` : ''} · 长期回报不外推历史估值抬升`)];
  const kpis = [
    ui.kpi(`美股大盘 第 ${k4 + 1} 季末累计（中位数）`, pctLog(at(eq.p50, k4)), `80% 区间 ${pctLog(at(eq.p10, k4), 0)} ~ ${pctLog(at(eq.p90, k4), 0)}`,
      ui.signCls(at(eq.p50, k4) !== null ? Math.expm1(at(eq.p50, k4)) : null)),
    quarters.length > FOCUS - 1 ? ui.kpi(`美股大盘 ${FOCUS} 季年化（中位数）`, pctS(ann(at(eq.p50, kF), FOCUS)), `主要看前 ${FOCUS} 个季度（10 年）`, ui.signCls(ann(at(eq.p50, kF), FOCUS))) : null,
    isNum(rar.percentile) ? ui.kpi('陈述组合的稀有度', `第 ${rar.percentile < 0.1 ? fx(rar.percentile * 100, 1) : fx(rar.percentile * 100, 0)} 百分位`,
      `约需每条约束 ${fx(rar.sd_per_constraint ?? NaN, 2)} 个标准差的冲击（χ² 分位；高于 99% 时历史上极少见）`, rar.percentile > 0.99 ? 'is-bad' : '') : null,
    tail ? ui.kpi('长期尾部发散的路径', prob(tail.share, 1), run.info?.bands?.excludes_tail
      ? '这些路径远在历史之外，不可信：下面的区间和表格不含它们' : '这些路径远在历史之外，不可信（区间含这些路径）', tail.share > 0.05 ? 'is-bad' : '') : null,
  ].filter(Boolean);
  const finite = (arr) => (Array.isArray(arr) ? arr : []).some(isNum);
  const brokenAssets = assets.filter(a => !finite(run.assets?.[a]?.cum_total?.p50));
  if (brokenAssets.length) {
    root.append(ui.notice(`这个情景有 ${brokenAssets.length} 个资产的路径这次没有算出来（显示为 —）：这是季度模型的已知问题，待修正；`
      + '宏观路径和情景的陈述仍然可以看。', 'error'), h('div', { style: { height: '12px' } }));
  }
  root.append(card({ cls: 'qd-trial', title: opts.title || '季度模型（试运行）：情景条件模拟',
    sub: '同一个联合模型：情景写了的值精确（或在给定标准差内）满足，没写的从联合分布补全；资产按同期映射得到。在切换关 T-S 通过之前，内置情景、组合与提取页仍用月度引擎。' },
  h('div', { class: 'qd-head' }, chips), h('div', { class: 'qd-kpis' }, kpis)));

  // ---- B. validation and notes
  const v = run.validation || {};
  const warn = [...(v.refused || []).map(x => ({ ...x, refused: true })), ...(v.warnings || [])];
  const notes = [...(run.notes || []), ...(run.qtable_notes || []).filter(n => !(v.warnings || []).some(w => w.code === n.code && w.text === n.text))];
  if (warn.length || notes.length) {
    const rows = warn.map(w => ({ code: w.code, text: w.text, where: w.where, evidence: w.evidence, refused: w.refused }));
    root.append(h('div', { style: { height: '12px' } }), card({ title: '校验与提示', sub: '拒绝的条目已从表里删除；“由模型补全”的提示附上叙事里的证据句（没有就写明没有）。' },
      rows.length ? h('ul', { class: 'list', style: { margin: 0 } }, rows.map(w => h('li', {},
        w.refused ? ui.chip('已拒绝', 'bad') : null, w.text,
        w.evidence ? h('div', { class: 'qd-note', style: { margin: '2px 0 0' } }, `证据句：${w.evidence}`) : null))) : null,
      notes.map(n => ui.notice(noteText(n), n.code === 'tail' ? 'error' : 'info'))));
  }

  // ---- C. stated / completed
  const cells = Array.isArray(run.cells) ? run.cells : [];
  const channels = Array.isArray(run.channels) ? run.channels : [];
  if (cells.length && channels.length) {
    const tblBox = h('div', { class: 'qd-t' });
    const rarQ = Object.fromEntries((rar.per_quarter || []).map(x => [x.q, x]));
    const draw = () => {
      const n = state.span === 'all' ? cells.length : Math.min(Number(state.span) || 8, cells.length);
      const rows = cells.slice(0, n);
      tblBox.replaceChildren(ui.table([
        { value: (r) => r, label: '季度', fmt: (r) => h('span', {}, h('b', {}, `第 ${r.q} 季`), h('br'), h('small', { class: 'muted' }, quarters[r.q - 1] || '')) },
        ...channels.map(ch => ({ value: (r) => r, label: ch.zh, fmt: (r) => cellNode(r.cells?.[ch.key], CH_MEDIAN[ch.key]?.(run, r.q - 1), r.q === 1 ? CH_SUB[ch.key] : '') })),
        { value: (r) => r, label: '其他陈述', fmt: (r) => ((r.other || []).length || (r.anchors || []).length
          ? h('div', {}, [...(r.other || []), ...(r.anchors || [])].map(it => h('div', {}, badge(it), h('small', {}, it.text)))) : '—') },
        { value: (r) => rarQ[r.q], label: '稀有度', fmt: (x) => (x && x.df > 0 ? h('span', { title: `需要约 ${fx(x.sd, 2)} 个标准差的冲击（自由度 ${x.df}）`, class: x.percentile > 0.99 ? 'is-bad' : '' }, `${prob(x.percentile, 0)} 分位`) : '—') },
      ], rows, { stack: true }));
    };
    const spanSeg = ui.seg([{ value: '4', label: '前 4 季' }, { value: '8', label: '前 8 季' }, { value: 'all', label: `全部 ${cells.length} 季` }],
      String(state.span), (x) => { state.span = x; draw(); });
    draw();
    root.append(section('情景说了什么 / 模型补了什么'), card({
      title: '逐季：情景的陈述与模型的补全',
      sub: '每格上面是模型模拟的中位数（第 1 季下方注明是哪个量），下面是来源：情景写了的条目按来源标出，没写的标“模型补全”。悬停在标签上看原文。',
      actions: spanSeg }, badgeLegend(), tblBox,
    h('details', { class: 'qd-more', style: { marginTop: '10px' } }, h('summary', {}, '你必须相信什么：这个情景写下的全部陈述'),
      h('div', { style: { marginTop: '8px' } }, beliefList(cells)))));
  }

  // ---- D. assets
  const fanBox = ui.chartBox();
  const assetSel = ui.select(assets.map(a => ({ value: a, label: assetName(a) })), state.asset, (x) => { state.asset = x; drawFan(); });
  const drawFan = () => {
    const c = run.assets?.[state.asset]?.cum_total || {};
    const toS = (arr) => [0, ...(arr || []).map(x => (isNum(x) ? Math.expm1(x) : null))];
    quarterFan(fanBox, { x: ['起点', ...quarters], band: { p10: toS(c.p10), p50: toS(c.p50), p90: toS(c.p90) }, name: assetName(state.asset), fmt: (y) => pctS(y, 0) });
  };
  const horizon = (k) => (k < quarters.length ? k : null);
  const cols = [3, FOCUS - 1, quarters.length - 1].map(horizon).filter((k, i, a) => k !== null && a.indexOf(k) === i);
  const assetRows = assets.map(a => ({ a, c: run.assets[a]?.cum_total || {} }));
  root.append(section('资产：季末累计总回报（情景条件模拟）'), card({ title: '累计总回报的分布', actions: assetSel,
    sub: '中位数与 10%–90% 区间；名义总回报（含股息 / 票息）。原油、10 年期国债和现金由各自的价格或收益率路径按恒等式得到。'
      + (run.info?.bands?.text ? `${run.info.bands.text}。` : '') }, fanBox,
  ui.table([
    { key: 'a', label: '资产', fmt: (x) => assetName(x) },
    ...cols.map(k => ({ value: (r) => r.c, label: `第 ${k + 1} 季末 中位（10%–90%）`,
      fmt: (c) => h('span', { class: ui.signCls(at(c.p50, k) !== null ? Math.expm1(at(c.p50, k)) : null) }, pctLog(at(c.p50, k)),
        h('small', { class: 'muted' }, ` ${pctLog(at(c.p10, k), 0)} ~ ${pctLog(at(c.p90, k), 0)}`)) })),
  ], assetRows, { stack: true })));
  later.push(drawFan);

  // ---- E. valuation
  root.append(section('估值：状态与隐含路径'), valuationBlock(run, later, state));

  // ---- F. attribution
  root.append(section(`归因：情景说了什么 / 模型补了什么 / 历史常态漂移 / 估值假设（${VAL_WORDING}）`), attributionBlock(run, later, state, assetName, H));

  // ---- G. financial conditions
  root.append(section('金融条件'), fcBlock(run, cells, quarters, later));

  // ---- H. regimes
  const rg = run.regimes;
  if (rg && Array.isArray(rg.years) && rg.years.length) {
    const box = ui.chartBox();
    const years = rg.years.slice(0, Math.min(rg.years.length, 10));
    root.append(section('宏观体制：季度路径处在什么状态'), card({ title: '模拟路径每一年的体制构成',
      sub: `${rg.text || ''}；“含收缩的路径”是这一年里至少一个季度处于收缩的路径占比。`, actions: h('a', { href: '#regimes' }, '宏观体制 →') }, box,
    ui.table([
      { key: 'year', label: '年', fmt: (y, r) => `第 ${y} 年（${(r.quarters || []).join(' 至 ')}）` },
      ...rg.labels.map(l => ({ value: (r) => r.mix?.[l], label: REGIME_NAME[l] || l, fmt: (x) => prob(x, 0) })),
      { key: 'paths_with_contraction', label: '含收缩的路径', fmt: (x) => prob(x, 0) },
    ], years, { stack: true })));
    later.push(() => charts.bars(box, { x: years.map(y => `第 ${y.year} 年`), valueMax: 1,
      series: rg.labels.map(l => ({ name: REGIME_NAME[l] || l, data: years.map(y => y.mix?.[l] ?? 0), stack: 'rg', color: REGIME_COLOR[l] })), fmt: 'pct' }));
  }

  // ---- I. delivery
  const del = Array.isArray(run.delivery) ? run.delivery : [];
  if (del.length) {
    root.append(section('陈述的交付：写下的值是否被满足'), card({ title: '每条陈述的交付',
      sub: '硬约束在每条路径上精确满足；软约束看路径落在 ±1 个给定标准差之内的比例（理想约 68%）。超过历史 ±4 个标准差的陈述按 ±4 个标准差截断（油价和 10 年期收益率作为自身价格不截断）。' },
    ui.table([
      { value: (r) => r, label: '陈述', fmt: (r) => deliveryName(r) },
      { key: 'quarters', label: '季度', fmt: (q) => (Array.isArray(q) ? `第 ${span(q)} 季` : '—') },
      { key: 'kind', label: '类型', fmt: (k) => ({ hard: '硬约束', soft: '软约束', range: '范围' }[k] || k || '—') },
      { value: (r) => [r, r.value], label: '写下的值', fmt: ([r, x]) => deliveredValue(r, x) },
      { value: (r) => [r, r.used], label: '实际使用', fmt: ([r, x]) => deliveredValue(r, x) },
      { value: (r) => [r, r.mean], label: '路径均值', fmt: ([r, x]) => deliveredValue(r, x, true) },
      { value: (r) => r, label: '交付', fmt: (r) => (r.kind === 'hard'
        ? (isNum(r.max_abs_error) ? (r.max_abs_error < 1e-6 ? '精确满足' : `最大偏差 ${fx(r.max_abs_error, 4)}`) : '—')
        : isNum(r.share_within_1sd) ? `${prob(r.share_within_1sd, 0)} 在 ±1 个标准差内` : '—') },
    ], del.slice(0, 60), { stack: true }),
    del.length > 60 ? h('p', { class: 'qd-note' }, `只列出前 60 条（共 ${del.length} 条）。`) : null));
  }

  // ---- J. method and limits
  root.append(h('div', { style: { height: '16px' } }), limitsCard(run));

  for (const draw of later) {
    try { draw(); } catch (err) { console.error(err); }
  }
}

// The oil-cause enum in words (the delivery stores its float: demand 0, mixed 0.5, supply 1)
const OIL_CAUSE_ZH = (x) => (x >= 0.75 ? '供给冲击' : x <= 0.25 ? '需求推动' : '混合');
function deliveredValue(r, x, mean = false) {
  if (!isNum(x)) return '—';
  if (r?.var === 'x_oil_cause') return mean ? `${OIL_CAUSE_ZH(x)}（${fx(x, 2)}）` : OIL_CAUSE_ZH(x);
  return fx(x, 2);
}
// A note for a page: a clip note written as "<field id>：…" gets the field's Chinese name (never a code id).
export function noteText(n) {
  const t = String(n?.text ?? '');
  const f = n?.field || n?.var;
  if (f && t.startsWith(`${f}：`)) return `${fieldName(f)}${t.slice(String(f).length)}`;
  return t;
}

// The card's Chinese name of a field id (the run's delivery names statements by field; level observables in words).
const FIELD_ZH = { ff: '政策利率水平', unrate: '失业率水平', cpi_yoy: 'CPI 同比', core_yoy: '核心 CPI 同比', spread: 'Baa − 10 年利差', gz: 'GZ 信用利差',
  y10: '10 年期收益率', usd: '美元', real_oil: '实际油价', d_ff: '政策利率变动', d_unrate: '失业率变动', d_ip: '工业产出增长', d_rgdp: '实际 GDP 增长',
  d_cpi_goods: '商品通胀', d_cpi_serv: '服务通胀', d_core: '核心通胀', d_oil: '油价变动', d_gz: 'GZ 信用利差变动', d_spread: 'Baa − 10 年利差变动',
  d_y10: '10 年期收益率变动', d_y2: '2 年期收益率变动', d_tp10: '期限溢价变动', d_usd: '美元变动', d_sent: '消费者信心变动', x_oil_cause: '油价变动的原因',
  d_wage: '工资增长', d_lprod: '劳动生产率增长', u_ff: '政策利率冲击', u_bs: '资产负债表冲击' };
let CARD_NAMES = null;
export function setCardNames(card) {
  CARD_NAMES = Object.fromEntries((card?.rows || []).map(r => [r.id, r.zh]));
}
export const fieldName = (id) => CARD_NAMES?.[id] || FIELD_ZH[id] || '（未命名字段）';
// A delivered statement in words: the stance and regime statements carry their own label ("fc:收紧", "regime:
// contraction:0.6"); the others are named by their field.
function deliveryName(r) {
  const lab = String(r?.label || '');
  if (lab.startsWith('fc:')) return `金融条件「${lab.slice(3)}」→ ${fieldName(r.var)}`;
  if (lab.startsWith('regime:')) {
    const [, id, share] = lab.split(':');
    return `体制请求「${REGIME_NAME[id] || { contraction: '收缩', high_infl: '高通胀' }[id] || '体制'}」${isNum(Number(share)) ? ` ${prob(Number(share), 0)}` : ''} → ${fieldName(r.var)}`;
  }
  return fieldName(r?.var);
}

function valuationBlock(run, later, state) {
  const val = run.valuation || {};
  const sts = Array.isArray(run.valuation_statements) ? run.valuation_statements : [];
  const quarters = run.quarters || [];
  const box = ui.chartBox();
  const which = { v: state.valFan || 'cape' };
  const holder = h('div');
  const body = card({ title: '隐含的 CAPE 与 P/D 路径', sub: '估值是状态，不是预测量：路径由模拟的价格回报和基本面腿按恒等式推出（倍数 = 价格 ÷ 基本面）；没有向均值拉回。',
    actions: ui.seg([{ value: 'cape', label: 'CAPE' }, { value: 'pd', label: 'P/D' }], which.v, (x) => { which.v = x; state.valFan = x; draw(); }) }, box, holder);
  const draw = () => {
    const f = val[which.v] || {};
    quarterFan(box, { x: quarters, band: { p10: f.p10 || [], p50: f.p50 || [], p90: f.p90 || [] }, name: which.v === 'cape' ? 'CAPE' : 'P/D',
      fmt: (y) => (isNum(y) ? fx(y, 1) : '—'), color: '#7048e8' });
  };
  later.push(draw);
  const modeCard = card({ title: '估值模式', sub: '默认是“由宏观与信用隐含”：倍数随模拟的宏观和信用变化，残差倍数零漂移，不回归。' },
    sts.length ? h('ul', { class: 'list', style: { margin: 0 } }, sts.map(s => {
      const dir = s.direction > 0 || s.direction === '+' ? '上调' : '下调';
      const what = s.mode === 'hold' ? '倍数保持不变' : s.mode === 'rerate' ? `倍数${dir}（${s.size || '—'}）` : '向历史中位数回归';
      return h('li', {}, h('b', {}, `估值假设（${VAL_WORDING}）：${what}。`),
        `美股价格在这 ${s.n} 个季度（第 ${span(s.quarters)} 季）里 ${pctLog(s.price_p50)}（倍数腿 ${pctLog(s.mult_p50)}，基本面腿 ${pctLog(s.fund_p50)}；三者各自是路径的中位数，不能直接相加）。`);
    })) : h('p', { style: { margin: 0 } }, '没有估值陈述：由宏观与信用隐含（默认）。'));
  // 「你必须相信的估值」: CAPE at the 40-quarter focus (or the last quarter) against today's and history
  const believe = h('div');
  valuationView().then((vv) => {
    if (!vv || !believe.isConnected) return;
    const capeNow = (vv.today?.states || []).find(s => s.key === 'cape');
    const k = Math.min(FOCUS, quarters.length) - 1;
    const c50 = val.cape?.p50?.[k];
    const pctl = capePercentile(vv.cape_q, c50);
    const when = k + 1 < 4 ? `第 ${k + 1} 季末` : `第 ${k + 1} 季末（约 ${fx((k + 1) / 4, (k + 1) % 4 ? 1 : 0)} 年后）`;
    believe.replaceChildren(h('p', { style: { margin: '10px 0 0' } }, h('b', {}, '你必须相信的估值：'),
      `${when} CAPE 约 ${isNum(c50) ? fx(c50, 1) : '—'}`,
      isNum(pctl) ? `（历史第 ${fx(pctl * 100, 1)} 百分位）` : '',
      capeNow ? `；今天 ${fx(capeNow.value, 1)}，${vv.cape_since || '1973Q1'} 以来第 ${fx((capePercentile(vv.cape_q, capeNow.value) ?? capeNow.percentile ?? 0) * 100, 1)} 百分位（同一张历史分位表）。` : '。'),
    h('p', { class: 'qd-note' }, 'CAPE 用 Shiller 2026 版工作簿；E 的最近两个季度可能是估计值。纳指的估值只能用全市场 CAPE 近似。'));
  });
  holder.append(believe);
  return h('div', { class: 'qd-two' }, body, modeCard);
}

function attributionBlock(run, later, state, assetName, H) {
  const byH = run.attribution_by_h || {};
  const opts = [...Object.keys(byH).map(k => ({ value: k, label: `前 ${k} 季` })), { value: 'all', label: `全部 ${H} 季` }];
  if (!opts.some(o => o.value === state.horizon)) state.horizon = opts[0].value;
  const box = ui.chartBox();
  const tbl = h('div');
  const parts = ['stated', 'completed', 'drift', 'valuation', 'residual'];
  const pick = () => (state.horizon === 'all' ? run.attribution?.assets : byH[state.horizon]) || {};
  const draw = () => {
    const A = pick();
    const risk = ['us_equity', 'nasdaq', 'energy', 'industrials', 'staples', 'healthcare', 'utilities', 'financials', 'real_estate', 'gold'].filter(a => A[a]);
    charts.bars(box, { x: risk.map(assetName), horizontal: true, fmt: (v) => pctS(v, 1),
      series: parts.slice(0, 4).map(p => ({ name: PART_NAME[p], data: risk.map(a => A[a]?.[p]?.median ?? null), stack: 'attr', color: PART_COLOR[p] })) });
    tbl.replaceChildren(ui.table([
      { key: 'a', label: '资产', fmt: (a) => assetName(a) },
      ...parts.map(p => ({ value: (r) => A[r.a]?.[p]?.median, label: PART_NAME[p], fmt: (x) => pctS(x, 1) })),
    ], risk.map(a => ({ a })), { stack: true }));
  };
  later.push(draw);
  return card({ title: '收益拆解（对数，路径中位数）',
    sub: '每条路径的季度对数价格收益线性拆成：情景写了的字段、模型补全的字段、历史常态漂移（截距与长期回报假设）、估值陈述的移位和残差（均值为 0）。'
      + `其中“估值假设”${VAL_WORDING}。`
      + '在历史上的大跌片段里，中位数的大部分来自历史常态漂移而不是情景本身，所以单独列出。',
    actions: ui.seg(opts, state.horizon, (x) => { state.horizon = x; draw(); }) }, box, tbl);
}

function fcBlock(run, cells, quarters, later) {
  const box = ui.chartBox('short');
  const n = Math.min(cells.length, 12);
  const stances = cells.slice(0, n).map(r => r.cells?.fc?.stance || '未说明');
  const unstated = stances.filter(s => s === '未说明').length;
  const lv = run.levels || {};
  later.push(() => {
    const x = quarters.slice(0, n);
    charts.line(box, { x, yFmt: (v) => fx(v, 2), series: [
      { name: 'Baa − 10 年利差（中位数）', data: (lv.spread?.p50 || []).slice(0, n), color: '#3b5bdb' },
      { name: 'GZ 信用利差（中位数）', data: (lv.gz?.p50 || []).slice(0, n), color: '#e8590c' },
    ] });
  });
  return card({ title: '金融条件：每一季的信用立场',
    sub: '金融条件是每个阶段的必填项：历史检验里，“已知宏观”对资产的样本外解释力几乎全部来自信用利差，写它接近写股权风险溢价——这是情景作者的假设。'
      + '立场按历史标准差的档换成对 GZ 信用利差和 Baa − 10 年利差的软约束（收紧 = +1 个标准差，明显收紧 = +2）。' },
  h('div', { class: 'qd-bar' }, stances.map((s, i) => h('span', { title: `第 ${i + 1} 季` },
    s === '未说明' ? completedBadge() : h('span', { class: 'qb qb-stated_qual' }, s)))),
  unstated ? ui.notice(`前 ${n} 季里有 ${unstated} 季没有说明金融条件：信用利差由模型补全。历史检验里，补全的信用利差与无条件抽样相当，而资产结果主要取决于它。`, 'info') : null,
  box);
}

export function limitsCard(run) {
  const li = (t) => h('li', {}, t);
  return h('details', { class: 'card qd-more' }, h('summary', { style: { fontWeight: 600 } }, '计算口径与局限'),
    h('ul', { class: 'list', style: { fontSize: '13px', marginTop: '8px' } },
      li('这是情景条件模拟，不是预测：预测只在情景假说（及其概率）里；同一个情景下的区间是“如果情景成立，模型认为会怎样”。'),
      li('因子层是一个季度联合模型（贝叶斯 VAR 核心 + 卫星方程）：情景写了的值精确或在给定标准差内满足，没写的从联合分布补全；政策干预用 do 算子；零下限用影子利率。'),
      li('资产层默认用因子轮的模型 F：月度估计的 β 加总到季度。宏观本身对资产几乎没有样本外信息，已知宏观的能力几乎全部来自信用利差，所以金融条件必填。'),
      li('神经网络做过（输入是状态、被掩码的情景字段和掩码本身），三个十年都输给结构化的线性高斯模型，只作对照。'),
      li('长期回报不外推训练期的 CAPE 倍数抬升（各训练期的数字见“估值与盈利”第 9 节）：那等于一个没说出口的估值判断。'),
      li(`季度模型在切换关 T-S 通过之前只是${TRIAL}：内置情景、未来一年页的默认数字、组合与提取仍用月度引擎。`),
      li(`12–20 个季度的检验统计${ANECDOTE}；这里的长期扇形图是模拟，不是检验结果。`),
      li('模拟里的极端季度按它们在历史上出现的频率出现：估计时用不按日期的稳健权重，不按日期降低它们的权重。'),
      isNum(run?.elapsed_s) ? li(`这次运行耗时 ${fx(run.elapsed_s, 1)} 秒。`) : null),
    run?.info?.version ? h('details', { class: 'qd-maint' }, h('summary', {}, '给维护者'),
      h('div', { class: 'qd-ver' }, `模型版本：${run.info.version}`)) : null);
}
