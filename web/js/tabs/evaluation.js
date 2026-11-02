// 情景条件检验 (SIM_DESIGN §8.6, §8.7, §10.1): how far the simulation lands from what happened once a scenario is
// given, and whether a wrong narrative does worse. It is not a test of predictive skill: prediction lives only in the
// scenario probabilities. Sections (deep link #evaluation?<key>; #evaluation?run=<run id> selects a narrative run):
//   conclusion  A. 结论: the status of every pre-registered test plus the narrative test's own headline (built by the
//               report from the pre-registered claims only; a mock run is never quoted here)
//   oracle      B. 已知因子路径时能复现多少: T-A's table (L0, G if adopted, NN report-only), the evaluation step's
//               conditional-path table, and the research round's ablation as the explanation card
//   narrative   C. 叙事检验: run selector, claims, scoreboard, paired differences, decomposition, discrimination,
//               small multiples, blindness panel, strata; the estimate-and-confirm run button
//   valuation   D. 估值账本: the identity by window, predicted against realized multiple with the placebo row, today
//   engine      E. 因子引擎检验: T-F (computed), delivery, fill-in skill, coverage, the do-operator, T-C
//   ledger      F. 前瞻账本: the forward ledger's state, linked to the forecast tab
//   records     G. 预注册与选择记录: every frozen record (date, hash, what was selected on which data, the result)
// Data: GET /api/evaluation (scenariolab/evaluation/evalview.py), /api/evaluation/narrative[/runs/<id>] (P7's
// report.json), /api/q/valuation. Every number comes from those files; 12-20Q figures carry ANECDOTE; no text from a
// model or a record reaches innerHTML (h(), textContent, <pre> for raw JSON, SVG nodes for the small multiples).
import * as ui from '../ui.js';
import * as charts from '../charts.js';
import * as api from '../api.js';

const { h, card } = ui;
const SVG = 'http://www.w3.org/2000/svg';
export const ANECDOTE = '每个十年只有 2–3 个独立窗口，只能当个案';
const TRIAL = '试运行';
// the switch gate's status (evaluation/switch.py state(), P9)
const GATE_STATUS = { passed: ['已通过', 'good'], failed: ['未通过', 'bad'], not_evaluated: ['尚未评估', 'warn'],
  stale: ['结果过期', 'warn'], invalid: ['结果无效', 'bad'] };
const SECTIONS = [['conclusion', 'A. 结论'], ['oracle', 'B. 已知因子路径时能复现多少'], ['narrative', 'C. 叙事检验'],
  ['valuation', 'D. 估值账本'], ['engine', 'E. 因子引擎检验'], ['ledger', 'F. 前瞻账本'], ['records', 'G. 预注册与选择记录']];
const STATE = { run: null, engine: null, cma: null, set: null, h: 'end', group: 'us_equity', asset: 'us_equity', arms: null };
const H_ZH = { '1Q': '1 个季度', '2Q': '2 个季度', '4Q': '4 个季度', '8Q': '8 个季度', end: '窗口末', endc: '共同窗口末',
  '12Q': '12 个季度', '20Q': '20 个季度' };                   // 12Q and 20Q are always shown with ANECDOTE (r2Table)
const H_ORDER = ['1Q', '2Q', '4Q', '8Q', 'end', 'endc'];
const Q_ORDER = ['1Q', '2Q', '4Q', '8Q', '12Q', '20Q'];       // T-A and the conditional tables (12Q, 20Q: ANECDOTE)
const ARM_SHORT = { U: '无叙事', OA: '已知宏观', OAp: '加已知商品价格', OB: '加已知金融市场', OF: '全部字段已知',
  V: '加已知估值倍数（恒等式上界）', N1: '叙事：宏观与政策', N2: '叙事：加金融条件', N3: '叙事：不含数字', NW: '错叙事',
  NS: '合成片段', HS: '美联储当时的预测', HS0: '美联储预测 + 金融条件中性' };
const RESEARCH_ARM = { U: '无叙事', cond: '已知 8 个视图变量的阶段均值', oracle: '已知因子路径', N1: '叙事 N1', N2: '叙事 N2（加金融条件）',
  N3: '叙事 N3（不含数字）' };
const SET_ZH = { clean: '干净集', original11: '原来的 11 个片段', short: '训练期短', bust: '大跌', ordinary: '普通', synth: '合成片段' };
const GROUP_ZH = { us_equity: '美股', gold: '黄金', crude: '原油', rates: '国债与现金', all13: '全部 13 个资产' };
const ASSET_ZH = { us_equity: '美股', nasdaq: '纳指', gold: '黄金', treasury_10y: '10 年期国债' };
const STATUS = { missing: ['尚无记录', ''], registered: ['已预注册，尚无结果', 'accent'], done: ['已有结果', 'good'], unreadable: ['读不了', 'bad'] };
const ENGINE_SHORT = { Q: '季度模型', E: '月度引擎', E0: '研究轮引擎', F: '因子轮 F', G: '研究轮季度模型 G', VB: '研究轮估值模块' };
const COLORS = ['#3b5bdb', '#e8590c', '#2f9e44', '#7048e8', '#c2255c', '#1098ad'];
let NAMES = {};                                    // card id -> 中文名 (GET /api/q/card), for the blindness panel

const CSS = `
.ev-anchor { scroll-margin-top: 80px; }
.ev-toc { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 0 0 6px; }
.ev-toc a.chip:hover { text-decoration: none; border-color: #c6cbd6; }
.ev-hero { border: 1px solid #cdd7ff; border-left: 4px solid var(--accent); background: linear-gradient(135deg, #f3f6ff, #fff 70%);
  border-radius: var(--radius); padding: 14px 16px; margin-bottom: 12px; }
.ev-hero h2 { margin: 0 0 8px; font-size: 16px; font-weight: 650; line-height: 1.55; color: #1f2f7a; }
.ev-concl { display: grid; gap: 8px; }
.ev-concl p { margin: 0; font-size: 14px; line-height: 1.7; color: var(--ink-2); }
.ev-concl p.warn { color: #7a4d00; }
.ev-concl p a { font-size: 12px; margin-left: 4px; }
.ev-status { display: flex; flex-wrap: wrap; gap: 6px 10px; margin-top: 10px; font-size: 12px; color: var(--ink-3); align-items: center; }
.ev-stack { display: grid; gap: 16px; }
.ev-two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(380px, 100%), 1fr)); gap: 16px; }
.ev-anec { color: var(--ink-3) !important; background: repeating-linear-gradient(135deg, #fafbfc, #fafbfc 6px, #f3f4f7 6px, #f3f4f7 12px); }
.ev-cell { font-variant-numeric: tabular-nums; }
.ev-cell small { display: block; color: var(--ink-3); font-size: 11px; white-space: normal; }
.ev-note { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; }
.ev-bar { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin: 0 0 10px; }
.ev-bar label { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 12px; color: var(--ink-3); min-width: 0; max-width: 100%; }
.ev-bar select { max-width: 100%; min-width: 0; }
.ev-bar .seg { max-width: 100%; }
.ev-labels { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 8px; }
.ev-labels .chip { white-space: normal; }
.ev-head { font-size: 14px; line-height: 1.7; color: var(--ink-2); margin: 0 0 10px; }
.ev-sm { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); gap: 12px; }
.ev-sm figure { margin: 0; border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: var(--panel); min-width: 0; }
.ev-sm figcaption { font-size: 12px; color: var(--ink-2); margin-bottom: 4px; }
.ev-sm figcaption small { color: var(--ink-3); }
.ev-sm svg { width: 100%; height: auto; display: block; }
.ev-legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; color: var(--ink-3); margin: 0 0 8px; }
.ev-legend i { display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 4px; border-radius: 2px; }
.ev-blind { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(260px, 100%), 1fr)); gap: 12px; }
.ev-blind > div { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: var(--panel-2); min-width: 0; }
.ev-blind h4 { margin: 0 0 6px; font-size: 13px; }
.ev-blind p { margin: 0; font-size: 12px; color: var(--ink-2); line-height: 1.6; }
.ev-run { border: 1px solid #f0c48a; background: #fff8ec; border-radius: 8px; padding: 10px 12px; margin-top: 10px; font-size: 13px; }
.ev-run label.tick { display: flex; gap: 8px; align-items: center; margin: 8px 0; }
.ev-form { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(200px, 100%), 1fr)); gap: 8px 12px; align-items: end; margin-bottom: 8px; }
.ev-form label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; color: var(--ink-3); min-width: 0; }
.ev-form input[type=text], .ev-form select { width: 100%; min-width: 0; }
.ev-checks { display: flex; flex-wrap: wrap; gap: 4px 10px; }
.ev-checks label { flex-direction: row; align-items: center; gap: 4px; color: var(--ink-2); }
.ev-msg { font-size: 12px; color: var(--ink-3); display: inline-flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.ev-msg.err { color: var(--bad); }
pre.code.ev-wrap { white-space: pre-wrap; word-break: break-word; max-height: 420px; overflow: auto; }
details.ev-more > summary { cursor: pointer; color: var(--accent); font-size: 13px; margin: 6px 0; }
details.ev-maint { margin: 6px 0 0; font-size: 12px; color: var(--ink-3); min-width: 0; }
details.ev-maint > summary { cursor: pointer; }
.ev-maint-line { overflow-wrap: anywhere; word-break: break-all; margin: 2px 0; }
.ev-t table.tbl td { vertical-align: top; }
.ev-t table.tbl th { white-space: normal; line-height: 1.35; vertical-align: bottom; }
.ev-kv { display: grid; grid-template-columns: max-content 1fr; gap: 3px 12px; font-size: 12px; margin: 0; }
.ev-kv dt { color: var(--ink-3); } .ev-kv dd { margin: 0; color: var(--ink); word-break: break-word; }
.ev-rec td { white-space: normal !important; }
`;

function ensureStyle() {
  let el = document.getElementById('style-eval');
  if (!el) { el = document.createElement('style'); el.id = 'style-eval'; document.head.append(el); }
  if (!el.textContent.includes('/* evaluation */')) el.textContent += `/* evaluation */\n${CSS}`;
}

// ---------------------------------------------------------------- formatting

const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
const fx = (x, d) => { const s = Number(x).toFixed(d); return Number(s) === 0 ? (0).toFixed(d) : s; };
const sg = (x, d = 3) => (isNum(x) ? `${Number(fx(x, d)) > 0 ? '+' : ''}${fx(x, d)}` : '—');
const nn = (x, d = 3) => (isNum(x) ? fx(x, d) : '—');
const pc = (x, d = 0) => (isNum(x) ? `${fx(x * 100, d)}%` : '—');
const ci = (lo, hi, d = 3) => (isNum(lo) && isNum(hi) ? `[${sg(lo, d)}, ${sg(hi, d)}]` : '');
const ciU = (lo, hi, d = 3) => (isNum(lo) && isNum(hi) ? `[${nn(lo, d)}, ${nn(hi, d)}]` : '');      // a non-negative statistic
const put = (el, ...kids) => { el.replaceChildren(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false)); return el; };
const passChip = (p, yes = '通过', no = '未通过') => (p === true ? ui.chip(yes, 'good') : p === false ? ui.chip(no, 'bad') : ui.chip('未评估'));
const short = (sha) => (sha ? String(sha).slice(0, 12) : '—');
const hOrder = (a, b) => H_ORDER.indexOf(a) - H_ORDER.indexOf(b);
const ANEC_H = new Set(['12Q', '20Q', '12', '20']);        // the horizons greyed with ANECDOTE (U-13)
const isAnecH = (hz) => ANEC_H.has(String(hz));
// one section failing to render shows its error and leaves the rest of the page intact
const safe = (name, fn) => {
  try { return fn(); } catch (err) {
    console.error(err);
    return [ui.notice(`「${name}」这一节渲染失败：${err.message}`, 'error')];
  }
};

function svg(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  for (const c of kids) if (c) el.append(typeof c === 'string' ? document.createTextNode(c) : c);
  return el;
}

// ---------------------------------------------------------------- names of arms, engines and variants

function legendOf(rep) {
  const L = rep?.legend || {};
  const research = rep?.kind === 'research';
  const arm = (id) => {
    if (research && RESEARCH_ARM[id]) return { name: RESEARCH_ARM[id], zh: L.research_arms?.[id] || RESEARCH_ARM[id] };
    const a = L.arms?.[id];
    return { name: a ? `${a.name} · ${ARM_SHORT[id] || ''}` : (ARM_SHORT[id] || '其他臂'), zh: a?.zh || ARM_SHORT[id] || '' };
  };
  const engine = (id) => L.engines?.[id] || { Q: '季度模型（试运行）', E: '月度引擎' }[id] || '其他引擎';
  const variant = (v) => { const [e, a] = String(v).split(':'); return `${ENGINE_SHORT[e] || engine(e)} · ${arm(a).name}`; };
  const cma = (id) => L.cma?.[id] || { historical: '训练期历史漂移', no_rerate: '扣除历史估值抬升' }[id] || '其他漂移口径';
  const assets = (id) => L.asset_sets?.[id] || GROUP_ZH[id] || '其他资产组';
  return { arm, engine, variant, cma, assets, armOrder: Object.keys(research ? RESEARCH_ARM : (L.arms || ARM_SHORT)) };
}

// ================================================================ A. 结论

function sectionA(ev, rep) {
  const parts = ev.conclusion?.parts || [];
  const out = [];
  const intro = parts.find(p => p.key === 'intro');
  if (intro) out.push(h('p', {}, intro.text));
  if (rep && rep.headline) {
    const when = rep.kind === 'real' ? `叙事检验（${String(rep.created || '').slice(0, 10)}，${rep.backend || ''}）：` : '';
    out.push(h('p', {}, when, rep.headline, ' ', h('a', { href: '#evaluation?narrative' }, '看 C 节 →')));
  } else if (ev.narrative?.headline_run) {
    out.push(h('p', { class: 'muted' }, '叙事检验的报告读不了：C 节会显示原因。'));
  }
  for (const p of parts) {
    if (p.key === 'intro') continue;
    out.push(h('p', { class: p.kind === 'warn' ? 'warn' : '' }, p.text, p.href ? h('a', { href: p.href }, '查看 →') : null));
  }
  const tests = ev.tests || {};
  const chips = ['TA', 'TC', 'TS', 'TN'].map((t) => {
    const s = tests[t] || {};
    const [word, kind] = STATUS[s.status] || STATUS.missing;
    return h('span', {}, `${s.zh || t}：`, ui.chip(`${word}${s.date ? `（${s.date}）` : ''}`, kind));
  });
  const gate = ev.switch_gate || {};
  const conds = gate.conditions || [];
  const reasons = gate.reasons || [];
  const cmds = gate.commands || [];
  const on = gate.status === 'passed' && gate.passed === true;
  const [gWord, gKind] = GATE_STATUS[gate.status] || GATE_STATUS.not_evaluated;
  return h('div', { class: 'ev-hero' },
    h('h2', {}, '结论：由预注册的声明和记录里的数字生成'),
    h('div', { class: 'ev-concl' }, out),
    gate.default_text ? h('p', { class: 'ev-note' }, h('b', {}, gate.default_text)) : null,
    conds.length ? h('ul', { class: 'list ev-note', style: { marginTop: '8px' } }, conds.map(c => h('li', {},
      c.evaluated === false ? ui.chip('不能评估', 'warn') : passChip(c.passed, '满足', '未满足'), ' ',
      `切换关条件：${c.text}`, c.detail ? `（${c.detail}）` : ''))) : null,
    !conds.length && reasons.length ? h('ul', { class: 'list ev-note', style: { marginTop: '8px' } },
      reasons.map(r => h('li', {}, r))) : null,
    !on && !gate.date && cmds.length ? h('details', { class: 'ev-note', style: { marginTop: '6px' } },
      h('summary', {}, '给维护者：怎样评估切换关'), gate.command_note ? h('p', {}, gate.command_note) : null,
      h('pre', { class: 'code ev-wrap' }, cmds.join('\n'))) : null,
    h('div', { class: 'ev-status' }, chips,
      h('span', {}, '切换关：', ui.chip(gWord, gKind)),
      h('span', {}, '季度模型：', ui.chip(on ? '默认' : TRIAL, on ? 'good' : 'warn'))));
}

// ================================================================ B. 已知因子路径时能复现多少

// rows: [{model, model_zh, horizon, r2, lo90, hi90, crpss, cov80, n_indep, report_only, anecdote}] -> model x horizon
function r2Table(rows) {
  const hs = [...new Set(rows.map(r => r.horizon))];
  hs.sort((a, b) => Q_ORDER.indexOf(a) - Q_ORDER.indexOf(b));
  const models = [...new Map(rows.map(r => [r.model, r])).values()];
  const cell = (m, hz) => rows.find(r => r.model === m.model && r.horizon === hz);
  const cols = [
    { value: (m) => m, label: '模型', wrap: true, fmt: (m) => h('div', {}, m.model_zh || '模型', m.report_only ? h('small', { class: 'muted', style: { display: 'block' } }, '只作对照，不进模型') : null) },
    ...hs.map(hz => ({ value: (m) => cell(m, hz), label: isAnecH(hz) ? `${H_ZH[hz] || hz}（${ANECDOTE}）` : (H_ZH[hz] || hz),
      cls: (c) => (isAnecH(hz) ? 'ev-anec' : ''),
      fmt: (c) => (c ? h('div', { class: 'ev-cell', title: isAnecH(hz) ? ANECDOTE : null }, `R² ${sg(c.r2)} ${ci(c.lo90, c.hi90)}`,
        h('small', {}, [isNum(c.crpss) ? `CRPSS ${sg(c.crpss)}` : null, isNum(c.cov80) ? `80% 覆盖 ${pc(c.cov80)}` : null,
          c.n_indep !== null && c.n_indep !== undefined ? `独立窗口 ${c.n_indep}` : null].filter(Boolean).join(' · '))) : '—') })),
  ];
  return h('div', { class: 'ev-t' }, ui.table(cols, models, { stack: true }));
}

function ablationCard(abl) {
  const hs = (abl.horizons || [1, 4, 12, 20]).map(String);
  const hLabel = (hq) => {
    const n = abl.models?.[0]?.cells?.[hq]?.n_indep;
    const base = `${hq} 个季度${n ? `（${n} 个独立窗口）` : ''}`;
    return isAnecH(hq) ? `${base}：${ANECDOTE}` : base;
  };
  const cols = [
    { value: (m) => m, label: '模型', wrap: true, fmt: (m) => h('div', {}, h('b', {}, m.short), h('small', { class: 'muted', style: { display: 'block' } }, m.zh)) },
    ...hs.map(hq => ({ value: (m) => m.cells?.[hq], label: hLabel(hq), cls: () => (isAnecH(hq) ? 'ev-anec' : ''),
      fmt: (c) => (c ? h('div', { class: 'ev-cell', title: isAnecH(hq) ? ANECDOTE : null }, sg(c.r2), h('small', {}, ci(c.lo90, c.hi90))) : '—') })),
  ];
  const pairs = abl.paired || [];
  const bs = [...new Set(pairs.map(p => p.b))];
  const shortOf = Object.fromEntries((abl.models || []).map(m => [m.id, m.short]));
  const pcols = [
    { value: (b) => b, label: '完整模型 F 减去', fmt: (b) => shortOf[b] || '对照模型' },
    ...hs.map(hq => ({ value: (b) => pairs.find(p => p.b === b && String(p.h) === hq), label: isAnecH(hq) ? `${hq} 个季度（${ANECDOTE}）` : `${hq} 个季度`,
      cls: () => (isAnecH(hq) ? 'ev-anec' : ''),
      fmt: (p) => (p ? h('div', { class: 'ev-cell' }, sg(p.diff), h('small', {}, ci(p.lo90, p.hi90))) : '—') })),
  ];
  return card({ title: abl.title || '研究轮的消融', sub: `${abl.label || '研究轮'}；${abl.protocol || ''}` },
    h('p', { class: 'ev-head' }, abl.text || ''),
    h('div', { class: 'ev-t' }, ui.table(cols, abl.models || [], { stack: true })),
    pairs.length ? [h('div', { class: 'section-title', style: { margin: '14px 0 6px' } }, 'R² 的成对差（块自助法 90% 区间）'),
      h('div', { class: 'ev-t' }, ui.table(pcols, bs, { stack: true }))] : null,
    h('ul', { class: 'list ev-note' }, (abl.notes || []).map(n => h('li', {}, n)),
      h('li', {}, `区间：${[...new Set((abl.models || []).map(m => m.interval))].join('；')}。`)),
    abl.spec ? maintainer(`出处：${abl.spec}`) : null);
}

// file names, phase and section references, commands and model versions: collapsed, for maintainers only
function maintainer(...lines) {
  return h('details', { class: 'ev-maint' }, h('summary', {}, '给维护者'),
    ...lines.filter(Boolean).map(t => h('div', { class: 'ev-maint-line' }, t)));
}
const zhSd = (t) => String(t ?? '').replace(/(\d)\s?sd\b/g, '$1 个标准差');

function sectionB(ev) {
  const al = ev.asset_layer;
  const ta = ev.tests?.TA || {};
  const out = [];
  if (al) {
    out.push(card({ title: 'T-A：资产层候选的预注册检验', sub: `${String(al.date || '').slice(0, 10)}；2015–2026 年检验期（预注册的重测，不是新的留出样本）。12 和 20 个季度：${ANECDOTE}。` },
      al.text ? h('p', { class: 'ev-head' }, al.text) : null,
      h('p', { class: 'ev-head' }, `资产层：${al.asset_layer || (al.adopted?.length ? `L0 加 ${al.adopted.join('、')}` : 'L0')}。`),
      al.rows?.length ? r2Table(al.rows) : ui.notice('记录里没有逐模型的表。', 'info'),
      al.conditions?.length ? h('ul', { class: 'list ev-note' }, al.conditions.map(c => h('li', {}, passChip(c.passed), ' ', c.text, c.detail ? `：${c.detail}` : ''))) : null));
  } else {
    out.push(card({ title: 'T-A：资产层候选的预注册检验' },
      ui.notice(ta.status === 'registered'
        ? `T-A 已预注册（${ta.date || ''}），还没有结果。运行一次之后，这里显示 L0、G（如果采纳）和神经网络（只作对照）在 1 个和 4 个季度的 R²（90% 区间）、CRPSS、80% 覆盖率与独立窗口数。`
        : '尚无记录：T-A 的预注册记录还没有落地。记录冻结、运行一次之后，这里显示 L0、G（如果采纳）和神经网络（只作对照）在 1 个和 4 个季度的 R²（90% 区间）、CRPSS、80% 覆盖率与独立窗口数。在那之前资产层用默认的 L0（因子轮 F）。', 'info')));
  }
  const cond = ev.conditional;
  out.push(card({ title: '已知因子路径的条件模拟（O-A、O-A′、O-B、O-full）', sub: `滚动起点合并；12 和 20 个季度：${ANECDOTE}。` },
    cond?.rows?.length ? [cond.text ? h('p', { class: 'ev-head' }, cond.text) : null, r2Table(cond.rows)]
      : ui.notice('尚无记录：这张表由“情景条件检验”这一步构建（重算预注册检验与条件路径表，约 10–20 分钟）生成，还没有生成。叙事检验 C 节里的 O-A … O-full 行是同一组臂在历史片段上的成绩。', 'info')));
  if (ev.ablation) out.push(ablationCard(ev.ablation));
  return h('div', { class: 'ev-stack' }, out);
}

// ================================================================ C. 叙事检验

function runLabel(r) {
  const when = String(r.created || '').replace('T', ' ').slice(0, 16);
  const kind = { research: '研究轮', mock: '模拟后端', real: '真实运行' }[r.kind] || '运行';
  const st = r.status && r.status !== 'done' ? `，${{ running: '运行中', paused: '已暂停', failed: '失败', cancelled: '已取消', translated: '已翻译、待评分', queued: '排队中' }[r.status] || r.status}` : '';
  const label = r.label || '未命名运行';
  const dated = when && label.includes(when.slice(0, 10));
  const tail = dated ? st.replace(/^，/, '') : `${when}${st}`;
  return `${label.startsWith(kind) ? '' : `${kind}：`}${label}${tail ? `（${tail}）` : ''}`;
}

function claimsTable(rep, Lg) {
  const rows = rep.claims || [];
  if (!rows.length) return null;
  const mock = rep.kind === 'mock';               // a pipeline check on the mock backend: no claim is concluded
  return h('div', { class: 'ev-t' }, ui.table([
    { value: (c) => c, label: '声明', wrap: true, fmt: (c) => h('div', {}, h('b', {}, c.id), c.claim && !c.always && c.evaluated && !mock ? h('small', { class: 'muted', style: { display: 'block' } }, `待检验的说法：${c.claim}`) : null) },
    { value: (c) => c, label: '结论', fmt: (c) => (c.always ? ui.chip('总是显示') : !c.evaluated ? ui.chip(`${c.id} 未评估`)
      : mock ? ui.chip(c.passed ? '流程跑通：条件满足' : '流程跑通：条件不满足') : passChip(c.passed, '成立', '不成立')) },
    { value: (c) => c, label: mock ? '流程检验（不作结论）' : '可以说的话', wrap: true, fmt: (c) => h('div', {},
      c.always ? (c.text || '') : !c.evaluated ? `${c.id} 未评估（缺少需要的臂或片段）`
        : mock ? (c.id === 'C2' ? '只用已知因子路径的臂（不涉及翻译）；模拟后端的运行不作结论' : '模拟后端的运行不作结论') : (c.text || ''),
      c.mde && c.evaluated && !mock ? h('small', { class: 'muted', style: { display: 'block' } }, `最小可检出差异约 ${nn(c.mde)}`) : null) },
    { value: (c) => c.conditions || [], label: '条件（股票类 9 个资产，窗口末 MAE 的成对差，90% 区间）', wrap: true,
      fmt: (cs) => (cs.length ? h('div', {}, cs.map(x => h('div', { class: 'ev-cell' }, `${Lg.variant(x.a)} − ${Lg.variant(x.b)}：`,
        x.evaluated === false ? '未评估（缺少需要的臂或片段）' : `${sg(x.diff)} ${ci(x.lo90, x.hi90)}，n ${x.n ?? '—'}`, ' ', x.evaluated === false ? null : passChip(x.passed, '上界 < 0', '上界 ≥ 0')))) : '—') },
  ], rows, { stack: true }));
}

function scoreTable(rows, Lg, { horizons } = {}) {
  const hs = horizons || [...new Set(rows.map(r => r.horizon))].sort(hOrder);
  const arms = [...new Set(rows.map(r => r.variant))];
  const cell = (v, hz) => rows.find(r => r.variant === v && r.horizon === hz);
  const oneEngine = new Set(arms.map(v => String(v).split(':')[0])).size === 1;
  return h('div', { class: 'ev-t' }, ui.table([
    { value: (v) => v, label: '臂', wrap: true, fmt: (v) => { const a = Lg.arm(String(v).split(':')[1]); return h('div', { title: a.zh }, oneEngine ? a.name : Lg.variant(v)); } },
    ...hs.map(hz => ({ value: (v) => cell(v, hz), label: H_ZH[hz] || hz, cls: (c) => (c?.anecdote ? 'ev-anec' : ''),
      fmt: (c) => (c ? h('div', { class: 'ev-cell', title: c.anecdote ? `只有 ${c.n_episodes} 个片段到这个期限：只能当个案` : null },
        `MAE ${nn(c.mae)} ${ciU(c.mae_lo90, c.mae_hi90)}`,
        h('small', {}, [`覆盖 ${pc(c.cov80)}`, isNum(c.crps) ? `CRPS ${nn(c.crps)}` : null, `n ${c.n_episodes ?? '—'}`, c.anecdote ? '只能当个案' : null].filter(Boolean).join(' · '))) : '—') })),
  ], arms, { stack: true }));
}

function pairedTable(pairs, Lg) {
  return h('div', { class: 'ev-t' }, ui.table([
    { value: (p) => p, label: '比较（a − b）', wrap: true, fmt: (p) => `${Lg.variant(p.a)} − ${Lg.variant(p.b)}` },
    { value: (p) => p, label: 'MAE 差 [90% 区间]', fmt: (p) => h('span', { class: p.favourable ? 'is-good' : p.unfavourable ? 'is-bad' : '' }, `${sg(p.diff)} ${ci(p.lo90, p.hi90)}`) },
    { key: 'n_episodes', label: '片段' },
    { key: 'mde', label: '最小可检出', fmt: (x) => nn(x) },
    { key: 'mse_skill', label: 'MSE 技能', fmt: (x) => sg(x, 2) },
    { key: 'a_wins_share', label: 'a 更近的片段', fmt: (x) => pc(x) },
    { value: (p) => p, label: '判定', fmt: (p) => (p.favourable ? ui.chip('有利（上界 < 0）', 'good') : p.unfavourable ? ui.chip('不利（下界 > 0）', 'bad') : ui.chip('区间含 0')) },
  ], pairs, { stack: true }));
}

function decompositionChart(dec, Lg) {
  const keys = Object.keys(dec || {}).filter(k => dec[k] && isNum(dec[k].hypothesis_mean) && isNum(dec[k].simulation_mean));
  if (!keys.length) return null;
  keys.sort((a, b) => a.localeCompare(b));
  const label = (k) => { const [arm, hz] = k.split('|'); return `${Lg.arm(arm).name}（${H_ZH[hz] || hz}）`; };
  const box = ui.chartBox('');
  box.style.height = `${Math.max(220, keys.length * 34 + 60)}px`;
  const draw = () => charts.bars(box, {
    x: keys.map(label), horizontal: true, wrap: true,
    series: [{ name: '假说与翻译误差（叙事 − 全部已知）', data: keys.map(k => dec[k].hypothesis_mean), stack: 'd', color: '#e8590c' },
      { name: '模拟误差（全部已知 − 实现）', data: keys.map(k => dec[k].simulation_mean), stack: 'd', color: '#3b5bdb' }],
    fmt: (v) => sg(v, 3),
  });
  const tbl = h('div', { class: 'ev-t' }, ui.table([
    { value: (k) => label(k), label: '臂与期限', wrap: true },
    { value: (k) => dec[k].n_episodes, label: '片段' },
    { value: (k) => dec[k].total_mabs, label: '总误差的平均绝对值', fmt: (x) => nn(x) },
    { value: (k) => dec[k].hypothesis_mabs, label: '假说与翻译部分', fmt: (x) => nn(x) },
    { value: (k) => dec[k].simulation_mabs, label: '模拟部分', fmt: (x) => nn(x) },
  ], keys, { stack: true }));
  return { box, draw, tbl };
}

function discriminationBlock(dis) {
  if (!dis || !(dis.episodes || []).length) return null;
  const rm = dis.right_minus_wrong || {};
  return [h('p', { class: 'ev-head' }, `右叙事（本片段的 N1）比错叙事（配对片段的 N1）离实际更近的片段占 ${pc(dis.share_right_better)}（${dis.n ?? dis.episodes.length} 个）；`
    + `窗口末股票类 MAE 的差（右 − 错）${sg(rm.diff)} ${ci(rm.lo90, rm.hi90)}，最小可检出差异约 ${nn(rm.mde)}。负数表示右叙事更近。`),
  h('div', { class: 'ev-t' }, ui.table([
    { key: 'code', label: '片段' },
    { key: 'right_mae', label: '右叙事 MAE', fmt: (x) => nn(x) },
    { key: 'wrong_mae', label: '错叙事 MAE', fmt: (x) => nn(x) },
    { key: 'right_better', label: '右叙事更近', fmt: (x) => (x ? ui.chip('是', 'good') : ui.chip('否', 'bad')) },
  ], dis.episodes, { stack: true }))];
}

// a small multiple: realized (black) against each selected arm's median (colour) and the first arm's 80% band
function smallMultiple(ep, variants, asset, Lg) {
  const W = 240, Hh = 132, L = 30, R = 6, T = 8, B = 18;
  const pts = [];
  const series = [];
  let xs = null;
  if (ep.series) {
    for (const v of variants) {
      const s = ep.series?.[v]?.[asset];
      if (!s) continue;
      series.push({ v, p10: s.p10 || [], p50: s.p50 || [], p90: s.p90 || [], y: s.realized || [] });
      xs = xs || (s.p50 || []).map((_, i) => `${i + 1}`);
    }
  } else if (ep.points) {                          // the research run: points at the scored horizons
    const hz = H_ORDER.filter(k => k !== 'endc' && variants.some(v => ep.points?.[v]?.[k]));
    xs = hz.map(k => H_ZH[k] || k);
    for (const v of variants) {
      const p = ep.points?.[v];
      if (!p) continue;
      series.push({ v, p10: hz.map(k => p[k]?.q10 ?? null), p50: hz.map(k => p[k]?.q50 ?? null), p90: hz.map(k => p[k]?.q90 ?? null), y: hz.map(k => p[k]?.y ?? null) });
    }
  }
  if (!series.length || !xs || !xs.length) return null;
  const all = series.flatMap(s => [...s.p10, ...s.p90, ...s.p50, ...s.y]).filter(isNum);
  if (!all.length) return null;
  let lo = Math.min(...all, 0), hi = Math.max(...all, 0);
  if (hi - lo < 1e-9) { lo -= 0.1; hi += 0.1; }
  const n = xs.length;
  const X = (i) => L + (n === 1 ? (W - L - R) / 2 : (i / (n - 1)) * (W - L - R));
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (Hh - T - B);
  const path = (arr) => { let d = '', pen = false; arr.forEach((v, i) => { if (!isNum(v)) { pen = false; return; } d += `${pen ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)} `; pen = true; }); return d.trim(); };
  const kids = [];
  kids.push(svg('line', { x1: L, x2: W - R, y1: Y(0).toFixed(1), y2: Y(0).toFixed(1), stroke: '#ced4da', 'stroke-dasharray': '2 2' }));
  const b = series[0];
  const top = b.p90.map((v, i) => (isNum(v) && isNum(b.p10[i]) ? `${X(i).toFixed(1)},${Y(v).toFixed(1)}` : null)).filter(Boolean);
  const bot = b.p10.map((v, i) => (isNum(v) && isNum(b.p90[i]) ? `${X(i).toFixed(1)},${Y(v).toFixed(1)}` : null)).filter(Boolean).reverse();
  if (top.length > 1) kids.push(svg('polygon', { points: [...top, ...bot].join(' '), fill: COLORS[0], 'fill-opacity': 0.13, stroke: 'none' }));
  else if (top.length === 1) kids.push(svg('line', { x1: X(0), x2: X(0), y1: Y(b.p90[0]), y2: Y(b.p10[0]), stroke: COLORS[0], 'stroke-opacity': 0.4, 'stroke-width': 6 }));
  series.forEach((s, k) => {
    const d = path(s.p50);
    if (d) kids.push(svg('path', { d, fill: 'none', stroke: COLORS[k % COLORS.length], 'stroke-width': 1.6 }, svg('title', {}, `${Lg.variant(s.v)}：中位数`)));
    s.p50.forEach((v, i) => { if (isNum(v) && n <= 6) kids.push(svg('circle', { cx: X(i), cy: Y(v), r: 2, fill: COLORS[k % COLORS.length] })); });
  });
  const y = series[0].y;
  const dy = path(y);
  if (dy) kids.push(svg('path', { d: dy, fill: 'none', stroke: '#1d2433', 'stroke-width': 1.8 }, svg('title', {}, '实现值')));
  y.forEach((v, i) => { if (isNum(v)) kids.push(svg('circle', { cx: X(i), cy: Y(v), r: 2.2, fill: '#1d2433' }, svg('title', {}, `${xs[i]}：实现 ${sg(v)}`))); });
  kids.push(svg('text', { x: 2, y: T + 8, 'font-size': 9, fill: '#7a8396' }, sg(hi, 2)));
  kids.push(svg('text', { x: 2, y: Hh - B, 'font-size': 9, fill: '#7a8396' }, sg(lo, 2)));
  kids.push(svg('text', { x: L, y: Hh - 4, 'font-size': 9, fill: '#7a8396' }, xs[0]));
  kids.push(svg('text', { x: W - R, y: Hh - 4, 'font-size': 9, fill: '#7a8396', 'text-anchor': 'end' }, xs[n - 1]));
  const w = ep.window || {};
  const span = w.start ? `${w.start}–${w.end || '…'}` : (w.months ? w.months.join(' – ') : '');
  const tags = [w.bust ? '大跌' : null, w.set === 'short' ? '训练期短' : null, ep.kind === 'synth' || w.set === 'synth' ? '合成' : null].filter(Boolean);
  return h('figure', {},
    h('figcaption', {}, h('b', {}, `片段 ${ep.code}`), ' ', h('small', {}, [span, ...tags].filter(Boolean).join(' · '))),
    svg('svg', { viewBox: `0 0 ${W} ${Hh}`, role: 'img', 'aria-label': `片段 ${ep.code} 的累计对数回报：实现值与各臂的中位数` }, ...kids));
}

function blindnessPanel(b, rep) {
  if (!b) return null;
  const items = [];
  const rec = b.recognition || {};
  if (rec.n) {
    items.push(h('div', {}, h('h4', {}, '识别率'), h('p', {}, rep?.kind === 'mock'
      ? `模拟探针总回答“不知道”，不是测量（${rec.recognised}/${rec.n}）。` : `探针认出了年代：${rec.recognised}/${rec.n}（${pc(rec.rate)}）。`,
      rec.rate >= 1 ? '识别出年代的回放（不是盲测）：仍能检查翻译的保真度和整条链，但不能检查事前能力。' : '探针的回答只与封存的答案比对，从不回传给翻译。',
      rec.source ? ` 来源：${rec.source}。` : '')));
  }
  if (b.prompts) {
    items.push(h('div', {}, h('h4', {}, '提示词扫描'), h('p', {}, `${b.prompts.length} 个提示词：`, b.prompt_clean ? ui.chip('干净', 'good') : ui.chip('有问题', 'bad'),
      ' 不含年份、季度标签、月数和日期化的足迹；提示词的哈希记录在运行清单里。')));
  }
  if (b.narratives) {
    const bad = b.narratives.filter(n => !n.clean);
    items.push(h('div', {}, h('h4', {}, '叙事扫描'), h('p', {}, `${b.narratives.length} 篇叙事：`, b.narratives_clean ? ui.chip('干净', 'good') : ui.chip(`${bad.length} 篇有标记`, 'warn'),
      ' 扫描年份、不含数字的 N3 里的数字、人名地名和禁用词。', bad.length ? ` 有标记的：${[...new Set(bad.map(n => `${n.code} ${String(n.narrative || '').toUpperCase()}`))].slice(0, 8).join('、')}。` : '')));
  }
  const con = b.contamination?.overall;
  if (con?.n) {
    items.push(h('div', {}, h('h4', {}, '污染指数'), h('p', {}, `翻译值与实现值的中位 |z| ${nn(con.median_abs_z, 2)}（n ${con.n}）；基线：历史均值 ${nn(con.baseline_prior, 2)}，模型无条件补全 ${nn(con.baseline_uncond, 2)}。`
      + '越接近 0、离基线越远，越像是“记得”实现的路径。')));
  }
  if ((b.recall || []).length) {
    items.push(h('div', {}, h('h4', {}, '回忆标记'), h('p', {}, '不含数字的叙事复原了实现路径（相关 > 0.9）：',
      b.recall.map(r => `${r.code} ${ARM_SHORT[r.arm] || '叙事'}的「${NAMES[r.field] || '一个字段'}」（相关 ${nn(r.corr, 3)}）`).join('；'), '。')));
  }
  const del = b.delivery?.overall;
  if (del?.n) items.push(h('div', {}, h('h4', {}, '交付'), h('p', {}, `陈述的值有 ${pc(del.share_within_half_sd)} 在 ±0.5 个标准差以内（${del.n} 个）。`)));
  const tr = b.translations;
  if (tr?.n) {
    items.push(h('div', {}, h('h4', {}, '翻译'), h('p', {}, `${tr.n} 次翻译：`, Object.entries(tr.status || {}).map(([k, v]) => `${{ ok: '完成', repaired: '修复后完成', fallback: '退回空表', standin: '研究轮代译' }[k] || '其他'} ${v}`).join('，'),
      Object.keys(tr.refused || {}).length ? `；被拒绝的值 ${Object.values(tr.refused).reduce((a, x) => a + x, 0)} 个（超出历史范围等）` : '', '。')));
  }
  if (b.manifest) items.push(h('div', {}, h('h4', {}, '运行清单'), h('p', {}, `${b.manifest.calls ?? 0} 次调用；发给真实后端的原文存档 ${b.manifest.sent_files ?? 0} 份。`)));
  return [(b.labels || []).length ? h('div', { class: 'ev-labels' }, b.labels.map(l => ui.chip(l, 'warn'))) : null, h('div', { class: 'ev-blind' }, items)];
}

function renderReport(box, rep, ctx, summary) {
  if (rep.status && rep.status !== 'done') {
    put(box, ui.notice(`这次运行的状态：${{ running: '运行中', paused: '已暂停（用量上限之后可以继续）', failed: '失败', cancelled: '已取消', translated: '已翻译、待评分', queued: '排队中' }[rep.status] || rep.status}`
      + `${rep.progress ? `（${rep.progress.phase || ''} ${rep.progress.done ?? ''}/${rep.progress.total ?? ''}）` : ''}${rep.error ? `：${rep.error}` : ''}`, rep.status === 'failed' ? 'error' : 'info'));
    return;
  }
  const Lg = legendOf(rep);
  let paintStrata = null;                          // the strata follow the engine chosen in the scoreboard
  const out = [];
  out.push(h('div', { class: 'ev-labels' }, (rep.labels || []).map(l => ui.chip(l, 'warn'))));
  out.push(h('p', { class: 'muted', style: { fontSize: '12px', margin: '0 0 8px' } },
    `后端：${rep.backend === 'mock' || rep.kind === 'mock' ? '模拟后端（没有调用模型）' : rep.backend === 'claude_code' ? '本机 Claude Code' : (rep.backend || '—')}`
    + `${rep.model && rep.backend !== 'mock' && rep.kind !== 'mock' ? ` · 模型 ${rep.model}` : ''} · 叙事版本 ${rep.narratives || '—'} · 引擎 ${(rep.engines || []).map(Lg.engine).join('、') || '—'}`
    + `${rep.tn ? ` · 预注册记录（${(String(rep.tn.date || rep.tn.file || '').match(/\d{4}-\d{2}-\d{2}/) || ['已'])[0]} 冻结）` : ''}`),
    rep.tn ? maintainer(`预注册记录 ${rep.tn.file || rep.tn.id}（SHA-256 ${short(rep.tn.sha256)}）`) : null);
  out.push(h('p', { class: 'ev-head' }, rep.headline || ''));
  const claims = claimsTable(rep, Lg);
  if (claims) out.push(card({ title: '预注册的声明（C1–C4）', sub: '生成的文字不会比通过的声明说得更强；没有通过时只说“没有可检出的改进（最小可检出差异约 X）”。' }, claims));

  // ---- the scoreboard: arms x horizons (equity9), engine and CMA arm chosen here
  const sets = [['scoreboard', rep.scoreboard], ['scoreboard_clean', rep.scoreboard_clean]].filter(([, s]) => s && (s.rows || []).length);
  if (sets.length) {
    const sbBox = h('div');
    const ctl = h('div', { class: 'ev-bar' });
    const paintSb = () => {
      const [, sb] = sets.find(([k]) => k === STATE.set) || sets[0];
      const rows = sb.rows || [];
      const engines = [...new Set(rows.map(r => r.engine))];
      const cmas = [...new Set(rows.map(r => r.cma).filter(Boolean))];
      if (!engines.includes(STATE.engine)) STATE.engine = engines[0];
      if (!cmas.includes(STATE.cma)) STATE.cma = cmas[0];
      put(ctl,
        sets.length > 1 ? h('label', {}, '片段', ui.seg(sets.map(([k, s]) => ({ value: k, label: SET_ZH[s.set] || s.set })), STATE.set || sets[0][0], (v) => { STATE.set = v; paintSb(); })) : null,
        engines.length > 1 ? h('label', {}, '引擎', ui.seg(engines.map(e => ({ value: e, label: Lg.engine(e) })), STATE.engine, (v) => { STATE.engine = v; paintSb(); })) : null,
        cmas.length > 1 ? h('label', {}, '漂移', ui.seg(cmas.map(c => ({ value: c, label: Lg.cma(c) })), STATE.cma, (v) => { STATE.cma = v; paintSb(); })) : null);
      const sel = rows.filter(r => r.engine === STATE.engine && (!STATE.cma || r.cma === STATE.cma));
      const order = Lg.armOrder;
      sel.sort((a, b) => order.indexOf(a.arm) - order.indexOf(b.arm));
      put(sbBox, scoreTable(sel, Lg));
      if (paintStrata) paintStrata();
    };
    paintSb();
    const sb0 = sets[0][1];
    out.push(card({ title: '成绩表：股票类 9 个资产的累计对数价格回报',
      sub: 'MAE 是中位数与实现值之差的绝对值，方括号是按片段自助的 90% 区间；覆盖 = 实现值落在 80% 区间里的比例。少于 5 个片段到达的期限灰显，只能当个案。' },
    ctl, sbBox, rep.n_per_horizon ? h('p', { class: 'ev-note' }, `到达各期限的片段数：${Object.entries((rep.n_per_horizon[sb0.set] || rep.n_per_horizon.clean || {})).map(([k, v]) => `${H_ZH[k] || k} ${v}`).join('，')}。`) : null));
  }

  // ---- asset groups at 4Q and the window end
  const groups = Object.entries(rep.groups || {}).filter(([, rows]) => (rows || []).length);
  if (groups.length) {
    const gBox = h('div');
    const paintG = () => {
      const [, rows] = groups.find(([k]) => k === STATE.group) || groups[0];
      const eng = rows.some(r => r.engine === STATE.engine) ? STATE.engine : rows[0]?.engine;
      const sel = rows.filter(r => r.engine === eng && (!STATE.cma || !r.cma || r.cma === STATE.cma));
      sel.sort((a, b) => Lg.armOrder.indexOf(a.arm) - Lg.armOrder.indexOf(b.arm));
      put(gBox, scoreTable(sel, Lg, { horizons: ['4Q', 'end'].filter(hz => sel.some(r => r.horizon === hz)) }));
    };
    out.push(card({ title: '其他资产组（4 个季度与窗口末）', sub: '原油、国债与现金是机械资产（它们的输入就是自己的价格），单独打分。' },
      h('div', { class: 'ev-bar' }, ui.seg(groups.map(([k]) => ({ value: k, label: GROUP_ZH[k] || Lg.assets(k) })), groups.some(([k]) => k === STATE.group) ? STATE.group : groups[0][0], (v) => { STATE.group = v; paintG(); })),
      gBox));
    paintG();
  }

  // ---- paired differences
  const paired = rep.paired || [];
  if (paired.length) {
    const hs = [...new Set(paired.map(p => p.horizon))].sort(hOrder);
    if (!hs.includes(STATE.h)) STATE.h = hs.includes('end') ? 'end' : hs[hs.length - 1];
    const pBox = h('div');
    const paintP = () => put(pBox, pairedTable(paired.filter(p => p.horizon === STATE.h && p.assets === 'equity9'), Lg));
    const ch = rep.chance;
    out.push(card({ title: '成对差：每个片段的 MAE 差，再对片段取平均', sub: '负数 = a 离实际更近。片段是独立单位；区间是按片段自助的 90% 区间。' },
      h('div', { class: 'ev-bar' }, ui.seg(hs.map(x => ({ value: x, label: H_ZH[x] || x })), STATE.h, (v) => { STATE.h = v; paintP(); })),
      pBox,
      ch ? h('p', { class: 'ev-note' }, `与偶然比：${ch.tests} 个叙事对无叙事的成对区间里，${ch.favourable} 个有利、${ch.unfavourable} 个不利；纯属偶然时每侧约 ${nn(ch.expected_each_side, 2)} 个。${ch.source ? `（${ch.source}）` : ''}`) : null,
      rep.mde?.text ? h('p', { class: 'ev-note' }, `${rep.mde.text}：差异小于它的改进检不出来。`) : null));
    paintP();
  }

  // ---- decomposition and discrimination
  const dec = decompositionChart(rep.decomposition, Lg);
  if (dec) {
    out.push(card({ title: '误差拆分：假说与翻译误差 + 模拟误差', sub: '总误差（叙事 − 实现）=（叙事 − 全部字段已知）+（全部字段已知 − 实现）。前一段是情景假说写错了多少，后一段是模型在情景写对时还错多少。' },
      dec.box, h('details', { class: 'ev-more' }, h('summary', {}, '平均绝对值'), dec.tbl)));
    ctx.later.push(dec.draw);
  }
  const dis = discriminationBlock(rep.discrimination);
  if (dis) out.push(card({ title: '右叙事对错叙事', sub: '每个干净片段配一个窗口长度相近、起点相隔至少 5 年的片段；错叙事是配对片段的 N1，按本片段的提示词翻译、从本片段的起点运行。' }, dis));

  // ---- small multiples
  const eps = (rep.episodes || []).filter(e => !e.skipped && (e.series || e.points));
  if (eps.length) {
    const vset = new Set();
    eps.forEach(e => Object.keys(e.series || e.points || {}).forEach(v => vset.add(v)));
    const variants = [...vset].sort();
    const defaults = rep.kind === 'research' ? ['E0:U', 'E0:oracle', 'E0:N2'] : ['Q:U', 'Q:OB', 'Q:N2', 'Q:NS'];
    if (!STATE.arms || !STATE.arms.some(v => vset.has(v))) STATE.arms = defaults.filter(v => vset.has(v)).slice(0, 3);
    if (!STATE.arms.length) STATE.arms = variants.slice(0, 2);
    const assets = [...new Set(eps.flatMap(e => Object.values(e.series || {}).flatMap(s => Object.keys(s || {}))))];
    if (assets.length && !assets.includes(STATE.asset)) STATE.asset = assets[0];
    const grid = h('div', { class: 'ev-sm' });
    const legend = h('div', { class: 'ev-legend' });
    const checks = h('div', { class: 'ev-checks' }, variants.map(v => h('label', {}, h('input', { type: 'checkbox', checked: STATE.arms.includes(v) ? true : null,
      onchange: (e) => { STATE.arms = e.target.checked ? [...STATE.arms, v].slice(-4) : STATE.arms.filter(x => x !== v); paintSm(); syncChecks(); } }), Lg.variant(v))));
    const syncChecks = () => checks.querySelectorAll('input').forEach((el, i) => { el.checked = STATE.arms.includes(variants[i]); });
    const paintSm = () => {
      put(legend, STATE.arms.map((v, k) => h('span', {}, h('i', { style: { background: COLORS[k % COLORS.length] } }), `${Lg.variant(v)}${k === 0 ? '（含 80% 区间）' : ''}`)),
        h('span', {}, h('i', { style: { background: '#1d2433' } }), '实现值'));
      const figs = eps.map(e => smallMultiple(e, STATE.arms, STATE.asset, Lg)).filter(Boolean);
      put(grid, figs.length ? figs : ui.notice('选中的臂在这些片段里没有序列。', 'info'));
    };
    paintSm();
    out.push(card({ title: '逐片段：实现值与各臂的中位数（累计对数回报，按季度）', sub: rep.kind === 'research' ? '研究轮只存了打分期限上的点（1、2、4、8 个季度和窗口末）。' : '第一条选中的臂画出 80% 区间。' },
      h('div', { class: 'ev-bar' }, assets.length > 1 ? h('label', {}, '资产', ui.seg(assets.map(a => ({ value: a, label: ASSET_ZH[a] || ui.assetName(a) })), STATE.asset, (v) => { STATE.asset = v; paintSm(); })) : null),
      h('details', { class: 'ev-more' }, h('summary', {}, '选择要画的臂（最多 4 个）'), checks),
      legend, grid));
  }

  // ---- the blindness panel, strata and the research extras
  const bl = blindnessPanel(rep.blindness, rep);
  if (bl) out.push(card({ title: '盲测面板', sub: '识别率、扫描、污染指数与回忆标记随每次运行一起量；识别率是 100% 时，结果只是识别出年代的回放。' }, bl));
  const strata = Object.entries(rep.strata || {}).filter(([, rows]) => (rows || []).length);
  if (strata.length) {
    const stBox = h('div', { class: 'ev-two' });
    const stSub = h('span');
    paintStrata = () => {
      stSub.textContent = `${Lg.engine(STATE.engine)}（成绩表里选的引擎）。`;
      put(stBox, strata.map(([k, rows]) => h('div', {}, h('div', { class: 'section-title', style: { margin: '0 0 6px' } }, SET_ZH[k] || k),
        h('div', { class: 'ev-t' }, ui.table([
          { value: (r) => Lg.arm(String(r.variant).split(':')[1]).name, label: '臂', wrap: true },
          { key: 'mae', label: 'MAE', fmt: (x) => nn(x) },
          { key: 'cov80', label: '覆盖', fmt: (x) => pc(x) },
          { key: 'n_episodes', label: '片段' },
        ], rows.filter(r => !STATE.engine || String(r.variant).split(':')[0] === STATE.engine)
          .sort((a, b) => Lg.armOrder.indexOf(String(a.variant).split(':')[1]) - Lg.armOrder.indexOf(String(b.variant).split(':')[1])), { stack: true })))));
    };
    paintStrata();
    out.push(card({ title: '分层（股票类 9 个资产，窗口末）' },
      h('p', { class: 'ev-note', style: { margin: '0 0 10px' } }, stSub, '训练期短的片段和合成片段单独报告，不进标题；大跌对普通是预注册的次要分层（大跌的定义用了结果，只能作次要分层）。'),
      stBox));
  }
  if (rep.kind === 'research') {
    out.push(card({ title: '研究轮的重算与批评的更正' },
      rep.checks ? h('p', { class: 'ev-head' }, `从存下的评分行重算发表的合并数字：${rep.checks.n} 个数字，`, rep.checks.ok ? ui.chip('全部一致', 'good') : ui.chip(`${(rep.checks.failed || []).length} 个不一致`, 'bad'),
        `（最大偏差 ${nn(rep.checks.max_dev, 4)}）。`) : null,
      h('ul', { class: 'list' }, (rep.corrections || []).map(c => h('li', {}, c.text))),
      (rep.critic || []).length ? h('details', { class: 'ev-more' }, h('summary', {}, `批评的成对差原文（${rep.critic.length} 行）`),
        h('pre', { class: 'code ev-wrap' }, rep.critic.map(c => `${c.source}  ${c.line}`).join('\n'))) : null));
  }
  if ((rep.notes || []).length && rep.kind !== 'research') out.push(h('ul', { class: 'list ev-note' }, rep.notes.map(n => h('li', {}, n))));
  put(box, h('div', { class: 'ev-stack' }, out));
  for (const f of ctx.later.splice(0)) { try { f(); } catch (err) { console.error(err); } }
}

// the estimate -> tick -> confirm: true flow (POST /api/evaluation/narrative/estimate, then /run)
function runPanel(summary, ctx, onStarted) {
  const panel = h('div', { class: 'ev-run', hidden: true });
  const open = h('button', { class: 'btn', type: 'button', onclick: () => { panel.hidden = !panel.hidden; } }, '用真实大模型重跑叙事检验…');
  const d = summary?.defaults || {};
  const backend = summary?.backend || 'mock';
  const epIn = h('input', { type: 'text', value: Array.isArray(d.episodes) ? d.episodes.join(',') : (d.episodes || 'all'), 'aria-label': '片段' });
  const narSel = ui.select([{ value: 'v2', label: 'v2（盲测规范 v2）' }, { value: 'v1', label: 'v1（研究轮的叙事）' }], d.narratives || 'v2', () => {});
  const armBoxes = ['N1', 'N2', 'N3', 'NW', 'NS'].map(a => h('label', {}, h('input', { type: 'checkbox', value: a, checked: (d.arms || []).includes(a) ? true : null }), ARM_SHORT[a]));
  const probe = h('input', { type: 'checkbox', checked: d.probe !== false ? true : null });
  const engBoxes = ['Q', 'E'].map(e => h('label', {}, h('input', { type: 'checkbox', value: e, checked: (d.engines || ['Q', 'E']).includes(e) ? true : null }), { Q: '季度模型（试运行）', E: '月度引擎' }[e]));
  const estBtn = h('button', { class: 'btn primary', type: 'button', onclick: () => estimate() }, '估算（不调用模型）');
  const msg = h('span', { class: 'ev-msg' });
  const estBox = h('div');
  put(panel,
    h('p', { style: { margin: '0 0 8px' } }, backend === 'mock'
      ? '当前后端：模拟后端（不调用模型，不消耗订阅额度）。在这里运行只是流程检验：翻译是研究轮的代译、合成片段的参考翻译或空表。'
      : '当前后端：本机 Claude Code，计入你自己的订阅额度。先看估算，勾选确认之后才会开始；到用量上限会暂停，之后可以继续。'),
    h('div', { class: 'ev-form' },
      h('label', {}, '片段：干净集写 clean，全部写 all，训练期短的一组写 short；也可以写片段编号，用逗号分开（例如 X03,X05）', epIn),
      h('label', {}, '叙事版本', narSel),
      h('label', {}, '翻译的臂', h('div', { class: 'ev-checks' }, armBoxes)),
      h('label', {}, '引擎', h('div', { class: 'ev-checks' }, engBoxes)),
      h('label', { style: { flexDirection: 'row', alignItems: 'center', gap: '6px' } }, probe, '识别探针（每个片段问一次是哪段时期）')),
    h('div', { class: 'ev-bar' }, estBtn, msg), estBox);
  const picked = (boxes) => boxes.map(l => l.querySelector('input')).filter(i => i.checked).map(i => i.value);
  async function estimate() {
    put(msg, ui.loading('估算中…'));
    put(estBox);
    let est;
    try {
      est = await api.post('evaluation/narrative/estimate', { episodes: epIn.value.trim() || 'all', narratives: narSel.value, arms: picked(armBoxes),
        probe: probe.checked, engines: picked(engBoxes) });
    } catch (err) {
      put(msg, h('span', { class: 'ev-msg err' }, `估算失败：${err.message}`));
      return;
    }
    if (!ctx.alive()) return;
    put(msg);
    const c = est.calls || {};
    const tick = h('input', { type: 'checkbox' });
    const go = h('button', { class: 'btn primary', type: 'button', disabled: true, onclick: () => start(est) }, '开始运行');
    tick.addEventListener('change', () => { go.disabled = !tick.checked; });
    const runMsg = h('span', { class: 'ev-msg' });
    put(estBox,
      h('dl', { class: 'ev-kv' },
        h('dt', {}, '后端'), h('dd', {}, est.backend_label || est.billing || est.backend || '—'),
        h('dt', {}, '调用'), h('dd', {}, `${c.total ?? 0} 次（翻译 ${c.translations ?? 0}，探针 ${c.probes ?? 0}；新调用 ${c.new ?? 0}，缓存 ${c.cached ?? 0}，研究轮代译 ${c.standin ?? 0}）`),
        h('dt', {}, '用量'), h('dd', {}, `输入约 ${est.input_tokens ?? 0} tokens，输出 ${(est.output_tokens || [0, 0]).join('–')} tokens；约 ${(est.minutes || [0, 0]).map(m => nn(m, 0)).join('–')} 分钟`),
        h('dt', {}, '计费'), h('dd', {}, est.billing || '—'),
        h('dt', {}, '有效期'), h('dd', {}, `估算编号 ${est.id}，${Math.round((est.fresh_s || 600) / 60)} 分钟内有效；计划改动后要重新估算`)),
      (est.missing || []).length ? h('details', { class: 'ev-more' }, h('summary', {}, `${est.missing.length} 个翻译缺叙事，不会运行`),
        h('ul', { class: 'list ev-note' }, est.missing.slice(0, 60).map(m => h('li', {}, `${m.code} ${ARM_SHORT[m.arm] || m.arm}：${m.text}`)))) : null,
      h('label', { class: 'tick' }, tick, est.real_calls ? `我确认：这会调用本机 Claude Code，计入我的订阅额度（约 ${c.new ?? c.total} 次新调用）。` : '我确认：用模拟后端跑一遍流程检验（不调用模型）。'),
      h('div', { class: 'ev-bar' }, go, runMsg));
    async function start(e) {
      go.disabled = true;
      put(runMsg, ui.loading('启动中…'));
      try {
        const res = await api.post('evaluation/narrative/run', { estimate_id: e.id, confirm: true });
        put(runMsg, `已开始：${res.run_id}`);
        onStarted(res.run_id);
      } catch (err) {
        put(runMsg, h('span', { class: 'ev-msg err' }, `没有开始：${err.message}`));
        go.disabled = false;
      }
    }
  }
  return { open, panel };
}

function sectionC(ev, ctx) {
  const box = h('div', {}, ui.loading('读取叙事检验…'));
  const reportBox = h('div');
  const selBox = h('div', { class: 'ev-bar' });
  const progress = h('div');
  let summary = null;
  let poll = 0;
  ctx.signal?.addEventListener('abort', () => clearTimeout(poll));
  const load = async (rid, picked = false) => {
    STATE.run = rid;
    if (picked) ui.setRoute('evaluation', `run=${encodeURIComponent(rid)}`, ctx.alive);
    put(reportBox, ui.loading('读取报告…'));
    try {
      const rep = await api.get(`evaluation/narrative/runs/${encodeURIComponent(rid)}`);
      if (!ctx.alive() || STATE.run !== rid) return;
      renderReport(reportBox, rep, ctx, summary);
    } catch (err) {
      if (ctx.alive()) put(reportBox, ui.notice(`报告读不了：${err.message}`, 'error'));
    }
  };
  const watch = async (rid) => {
    clearTimeout(poll);
    try {
      const p = await api.get(`evaluation/narrative/runs/${encodeURIComponent(rid)}/progress`, { fresh: true });
      if (!ctx.alive()) return;
      const pr = p.progress || {};
      put(progress, ui.notice(`运行 ${rid}：${{ running: '运行中', queued: '排队中', paused: '已暂停', failed: '失败', cancelled: '已取消', done: '完成', translated: '已翻译、待评分' }[p.status] || p.status}`
        + `${pr.phase ? `（${pr.phase} ${pr.done ?? ''}/${pr.total ?? ''}）` : ''}${p.error ? `：${p.error}` : ''}`, p.status === 'failed' ? 'error' : 'info'),
      p.status === 'running' || p.status === 'queued' ? h('button', { class: 'btn', type: 'button', style: { marginTop: '6px' }, onclick: async () => {
        try { await fetch(`/api/evaluation/narrative/runs/${encodeURIComponent(rid)}/cancel`, { method: 'POST' }); } catch (_) { /* the next poll shows it */ }
      } }, '取消这次运行') : null);
      if (p.status === 'running' || p.status === 'queued') { poll = setTimeout(() => watch(rid), 3000); return; }
      api.invalidate('evaluation/narrative');
      await paintRuns(rid);
    } catch (err) {
      if (ctx.alive()) put(progress, ui.notice(`进度读不了：${err.message}`, 'error'));
    }
  };
  async function paintRuns(selectId) {
    try {
      summary = await api.get('evaluation/narrative', { fresh: !!selectId });
    } catch (err) {
      if (ctx.alive()) put(box, ui.notice(`叙事检验读不了：${err.message}`, 'error'));
      return;
    }
    if (!ctx.alive()) return;
    const runs = summary.runs || [];
    if (!runs.length) {
      put(box, ui.notice('还没有任何运行：研究轮的存档不在，也还没有模拟或真实的运行。', 'info'));
      return;
    }
    const want = selectId || STATE.run || ev.narrative?.headline_run || runs[0].id;
    const cur = runs.some(r => r.id === want) ? want : runs[0].id;
    const sel = ui.select(runs.map(r => ({ value: r.id, label: runLabel(r) })), cur, (v) => load(v, true));
    sel.setAttribute('aria-label', '叙事检验的运行');
    const { open, panel } = runPanel(summary, ctx, (rid) => { put(progress, ui.notice(`运行 ${rid} 已开始。`, 'info')); watch(rid); });
    put(selBox, h('label', {}, '运行', sel), open);
    const act = summary.active;
    put(box, selBox, panel, progress,
      summary.registry?.sealed === false ? ui.notice(`片段登记表还没有封存：${summary.registry.hint || ''}`, 'info') : null,
      summary.sep && !summary.sep.available ? h('div', { class: 'ev-note' }, h('p', { style: { margin: 0 } }, `美联储当时的预测（SEP）这一臂：${summary.sep.text || '数据还没有下载'}。`),
        summary.sep.command ? maintainer('下载命令（在项目根目录手动运行，下载前会再问你）：', h('pre', { class: 'code ev-wrap' }, summary.sep.command)) : null) : null,
      reportBox);
    if (act?.rid) watch(act.rid);
    await load(cur);
  }
  paintRuns(null);
  return box;
}

// ================================================================ D. 估值账本

function sectionD(ev) {
  if (ev.panel === false) {
    return ui.notice('季度面板还没有生成，估值账本暂缺：在项目根目录运行 python -m scenariolab.qmodel.panel。', 'info');
  }
  const box = h('div', {}, ui.loading('读取估值口径…'));
  api.get('q/valuation').then((v) => {
    if (!box.isConnected) return;
    const id = v.identity || {};
    const all = id.all || {};
    const anat = ev.anatomy;
    const fmtVal = (r) => (isNum(r.value) ? `${fx(r.value, r.digits ?? 1)}${r.unit === '%' ? '%' : r.unit === '倍' ? ' 倍' : ''}` : '—');
    put(box, h('div', { class: 'ev-stack' },
      card({ title: '每个窗口：实现的价格 = 倍数腿 + 基本面腿（对数）', sub: '倍数腿是 CAPE 倍数，基本面腿是 CPI × 10 年平均实际盈利。P/D 与 P/Ê 只作对照：P/Ê 在两次大跌里因为减记反而上升。' },
        h('div', { class: 'ev-t' }, ui.table([
          { value: (r) => r, label: '窗口', fmt: (r) => `${r.zh || ''}（${r.from}–${r.to}）`, wrap: true },
          { key: 'price', label: '价格', fmt: (x) => sg(x) },
          { key: 'mult', label: 'CAPE 倍数', fmt: (x) => sg(x) },
          { key: 'fund', label: '基本面', fmt: (x) => sg(x) },
          { key: 'pd', label: 'P/D', fmt: (x) => sg(x) },
          { key: 'pe', label: 'P/Ê（含减记）', fmt: (x) => sg(x) },
        ], id.windows || [], { stack: true })),
        all.n ? h('p', { class: 'ev-note' }, `全部 ${all.n} 个季度：与价格的相关 / 占价格方差的份额——CAPE 倍数 ${all.mult?.corr ?? '—'} / ${all.mult?.share ?? '—'}，P/D ${all.pd?.corr ?? '—'} / ${all.pd?.share ?? '—'}，P/Ê ${all.pe?.corr ?? '—'} / ${all.pe?.share ?? '—'}。`
          + '倍数几乎承载了全部价格方差：拆分是记账，不是证据。') : null),
      card({ title: '模拟的倍数变化对实现的倍数变化（含安慰剂行）', sub: '不用 corr(误差, Δ倍数)²：那是恒等式，一个常数在各片段上也能得到 0.85。这里报告模拟与实现的相关、实现对模拟的斜率和相对零的 R²，每张表都带安慰剂行（常数）。' },
        anat?.rows?.length ? h('div', { class: 'ev-t' }, ui.table([
          { value: (r) => r, label: '窗口组', fmt: (r) => `${r.group || ''}`, wrap: true },
          { key: 'h_q', label: '期限', fmt: (x) => (isAnecH(String(x)) ? `${x} 个季度（${ANECDOTE}）` : `${x} 个季度`) },
          { key: 'row', label: '行', fmt: (x) => (x === 'placebo' ? '安慰剂（常数）' : '模型') },
          { key: 'n', label: 'n' },
          { key: 'corr_pred_dm', label: '相关（模拟, 实现）', fmt: (x) => sg(x, 2) },
          { key: 'slope_dm_on_pred', label: '斜率', fmt: (x) => sg(x, 2) },
          { key: 'r2_dm_vs_zero', label: '相对零的 R²', fmt: (x) => sg(x, 2) },
        ], anat.rows, { stack: true })) : ui.notice('尚无记录：这张表由“情景条件检验”这一步构建（安慰剂校正拆分）生成，还没有生成。', 'info')),
      card({ title: `今天的估值状态（${v.today?.quarter || '—'} 季初）`, sub: `分位按季度面板 ${v.cape_since || '1973Q1'} 以来的历史；只作状态和初始条件，不作预测量。` },
        h('div', { class: 'ev-t' }, ui.table([
          { key: 'zh', label: '状态', wrap: true },
          { value: (r) => r, label: '今天', fmt: (r) => fmtVal(r) },
          { key: 'percentile', label: '历史分位', fmt: (x) => pc(x, 1) },
          { key: 'use', label: '用途与注意', wrap: true },
        ], v.today?.states || [], { stack: true })),
        h('ul', { class: 'list ev-note' }, (v.notes || []).map(n => h('li', {}, n))),
        h('p', { class: 'ev-note' }, h('a', { href: '#valuation?quarterly' }, '估值与盈利 → 季度模型的估值口径 →')))));
  }).catch((err) => { if (box.isConnected) put(box, ui.notice(`估值口径读不了：${err.message}`, 'info')); });
  return box;
}

// ================================================================ E. 因子引擎检验

function tfTable(fl) {
  const ends = (fl.fits || []).map(f => f.end);
  const fmtH = (r) => {
    if (r.id === 'F3_gt6q' && Array.isArray(r.history)) return `${r.history[0]} / ${r.history[1]} 次`;
    if (Array.isArray(r.history)) return r.history.map(x => nn(x, 2)).join(' / ');
    return isNum(r.history) ? nn(r.history, 3) : '—';
  };
  const cellOf = (r, e) => {
    const c = r.fits?.[e];
    if (!c) return '—';
    if (r.id === 'F6_bounds') {
      return h('div', { class: 'ev-cell' }, passChip(c.pass), h('small', {}, `政策利率最低 ${nn(c.ff_min, 2)}；失业率最低 ${nn(c.u_min, 2)}，低于 3% 的季度 ${pc(c.u_below3, 1)}；CPI 同比 > 15% 的季度 ${pc(c.cpi_gt15, 1)}`));
    }
    if (r.id === 'F3_gt6q') return h('div', { class: 'ev-cell' }, passChip(c.pass), h('small', {}, `模拟里长于 6 季的占 ${pc(c.share)}`));
    return h('div', { class: 'ev-cell' }, passChip(c.pass), h('small', {}, `${nn(c.p50, 3)} [${nn(c.p05, 3)}, ${nn(c.p95, 3)}]`));
  };
  return h('div', { class: 'ev-t' }, ui.table([
    { value: (r) => r, label: '统计量', wrap: true, fmt: (r) => h('div', {}, r.zh) },
    { value: (r) => r, label: '历史', fmt: fmtH },
    ...ends.map((e) => {
      const f = (fl.fits || []).find(x => x.end === e) || {};
      return { value: (r) => r, label: `训练到 ${e}（${f.anchors_zh || ''}）`, fmt: (r) => cellOf(r, e) };
    }),
  ], fl.tf_rows || [], { stack: true }));
}

function sectionE(ev, ctx) {
  const fl = ev.factor_layer;
  const out = [];
  if (!fl) {
    out.push(ui.notice('因子引擎检验尚无结果：在项目根目录运行 python -m scenariolab.build --only qmodel，生成 T-F 与 T-C 的计算值。', 'info'));
  } else {
    out.push(card({ title: 'T-F：因子层的模拟能否复现历史的统计特征', sub: `${fl.label}。每个统计量按模拟的 13 年路径逐条计算，历史值要落在模拟的 5%–95% 之间；格子里是模拟的中位数 [5%, 95%]。${fl.created ? `计算于 ${fl.created}。` : ''}` },
      tfTable(fl),
      h('ul', { class: 'list ev-note' }, (fl.notes || []).map(n => h('li', {}, n)))));
    const f1 = fl.f1;
    if (f1) {
      const soft = (f1.delivery || []).filter(d => d.kind === 'soft' && isNum(d.share_within_1sd));
      const hard = (f1.delivery || []).filter(d => d.kind === 'hard');
      const rng = (f1.delivery || []).filter(d => d.kind === 'range');
      const chart = ui.chartBox('');
      chart.style.height = `${Math.max(180, soft.length * 30 + 60)}px`;
      out.push(card({ title: 'F1 交付：陈述的值有没有被满足', sub: `${f1.n_paths} 条路径。硬约束的最大误差 ${f1.hard_max_abs_error_sci || '—'}（要求 < 1e-8）；软约束落在 ±1 个标准差以内的比例应在 68% ± 5%。` },
        soft.length ? chart : null,
        h('ul', { class: 'list ev-note' },
          hard.length ? h('li', {}, `硬约束 ${hard.length} 条，逐条最大误差：${hard.map(d => `${zhSd(d.label || d.field)}（${d.max_abs_error_sci || '—'}）`).join('；')}。`) : null,
          rng.length ? h('li', {}, `范围约束 ${rng.length} 条：落在范围内的路径比例 ${[...new Set(rng.map(d => pc(d.share_inside)))].join('、')}。`) : null,
          f1.rarity ? h('li', {}, `稀有度：这组陈述需要的冲击约为每条约束 ${nn(f1.rarity.sd_per_constraint, 2)} 个标准差（χ² 第 ${pc(f1.rarity.percentile)} 百分位）。`) : null),
        h('p', { class: 'ev-note' }, passChip(f1.pass), ' 交付检验')));
      if (soft.length) {
        ctx.later.push(() => charts.bars(chart, { x: soft.map(d => (d.label && d.field && d.label !== d.field ? `${d.label}（${d.field}）` : d.label || d.field)), horizontal: true, wrap: true,
          series: [{ name: '在 ±1 个标准差以内的路径比例', data: soft.map(d => d.share_within_1sd), color: '#3b5bdb' }],
          fmt: (v) => pc(v), markLine: [{ xAxis: 0.68, name: '目标 68%' }], valueMax: 1 }));
      }
    }
    const f7 = fl.f7;
    if (f7) {
      const hs = Object.keys(f7).sort((a, b) => Number(a) - Number(b));
      const keys = [...new Set(hs.flatMap(k => (f7[k].rows || []).map(r => r.key)))];
      const byKey = (hk, key) => (f7[hk].rows || []).find(r => r.key === key);
      out.push(card({ title: 'F7 补全技能：已知 CPI、工业产出和油价时，没说的因子补得怎么样', sub: `滚动起点 ${f7[hs[0]]?.first || ''}–${f7[hs[0]]?.last || ''}；R² 相对“不变”。规则：失业率与 Baa 利差不低于月度引擎，其余不低于 −0.05；美元没说时保持不变（哪里都没有技能）。` },
        h('div', { class: 'ev-t' }, ui.table([
          { value: (k) => byKey(hs[0], k)?.zh || k, label: '因子', wrap: true },
          ...hs.map(hk => ({ value: (k) => byKey(hk, k), label: isAnecH(hk) ? `${hk} 个季度（${ANECDOTE}）` : `${hk} 个季度（独立窗口 ${f7[hk].n_indep ?? '—'}）`,
            cls: () => (isAnecH(hk) ? 'ev-anec' : ''),
            fmt: (r) => (r ? h('div', { class: 'ev-cell' }, `R² ${sg(r.r2, 2)}`, h('small', {}, [isNum(r.engine) ? `月度引擎 ${sg(r.engine, 2)}` : null, r.check === true ? '达标' : r.check === false ? '未达标' : null].filter(Boolean).join(' · '))) : '—') })),
        ], keys, { stack: true })),
        h('p', { class: 'ev-note' }, hs.flatMap(hk => [passChip(f7[hk].pass), ` ${hk} 个季度${isAnecH(hk) ? `（${ANECDOTE}）` : ''}　`]))));
    }
    const f8 = fl.f8;
    if (f8) {
      out.push(card({ title: 'F8 覆盖率：4 个季度的因子变化落在 80% 区间里的比例', sub: `${f8.windows} 个不重叠窗口（${f8.first}–${f8.last}）；要求在 72%–88% 之间。合计 ${pc(f8.coverage)}。` },
        h('div', { class: 'ev-t' }, ui.table([
          { key: 'zh', label: '因子', wrap: true },
          { key: 'coverage', label: '80% 区间覆盖', fmt: (x) => h('span', { class: ui.goodCls(isNum(x) ? x >= 0.72 && x <= 0.88 : null) }, pc(x)) },
        ], f8.per_field || [], { stack: true })),
        h('p', { class: 'ev-note' }, passChip(f8.pass), ' 合计覆盖率')));
    }
    const f9 = fl.f9, f10 = fl.f10;
    if (f9 || f10) {
      const s = f9?.signs || {};
      out.push(card({ title: 'F9 干预（do 算子）与 F10 逐季 / 整段一致', sub: '写明的 −1 个标准差意外降息：同季的外生宏观必须完全不动；股票当季上涨、2 年期收益率和美元当季下跌；失业率的反应在 4–8 个季度达到峰值。' },
        h('ul', { class: 'list' },
          f9 ? h('li', {}, passChip(f9.same_quarter_E0_unchanged, '不动', '动了'), ` 同季外生宏观的最大变化 ${f9.same_quarter_E0_max_abs_change_sci || '—'}`) : null,
          f9 ? h('li', {}, `当季反应：股票 ${sg(s.equity_impact)}，2 年期收益率 ${sg(s.d_y2_impact)}，美元 ${sg(s.usd_impact)}；失业率反应的峰值在第 ${s.unrate_peak_q ?? '—'} 个季度。`, passChip(f9.signs_as_literature, '符号与文献一致', '与文献不一致')) : null,
          f9 ? h('li', {}, passChip(f9.imputes_nothing, '没有乱补', '补出了宏观'), ' 只写降息、不写衰退时，宏观不随它补全') : null,
          f10 ? h('li', {}, passChip(f10.pass), ` 只在第 1 季陈述时，整段模式与逐季模式第 1 季的分布一致（${f10.ks_fields} 个字段，KS 最小 p ${nn(f10.ks_min_p, 3)}，${f10.n_draws} 次抽样）`) : null)));
    }
    const tc = fl.tc;
    if (tc) {
      const byW = Object.entries(tc.by_w || {});
      const pats = Object.entries(tc.test || {});
      out.push(card({ title: 'T-C：文字没说的字段怎样补全', sub: `${tc.validation || ''}；检验期到 ${tc.test_end || ''} 之后（${tc.n_test_origins ?? '—'} 个起点）。计算值；冻结的选择记录见 G 节。` },
        h('p', { class: 'ev-head' }, `补全收缩权重 w 按验证期的 CRPS 选为 ${tc.completion_w ?? '—'}（1 = 不收缩）；体制协方差${tc.regime_cov_adopted ? '采纳' : '没有采纳'}。`),
        byW.length ? h('div', { class: 'ev-t' }, ui.table([
          { value: (x) => x[0].replace('w=', 'w = '), label: '收缩权重' },
          { value: (x) => x[1], label: '相对无条件的 CRPS（越小越好）', fmt: (x) => nn(x, 3) },
        ], byW, { stack: true })) : null,
        h('div', { class: 'section-title', style: { margin: '14px 0 6px' } }, `关卡：${tc.gate_rule}`),
        h('div', { class: 'ev-t' }, ui.table([
          { key: 'zh', label: '金融市场字段', wrap: true },
          { key: 'r2', label: '补全 R²（相对无条件）', fmt: (x) => h('span', { class: ui.goodCls(isNum(x) ? x >= -0.02 : null) }, sg(x, 3)) },
        ], tc.gate_r2 || [], { stack: true })),
        h('p', { class: 'ev-note' }, passChip(tc.gate_pass), ' 关卡'),
        pats.length ? h('details', { class: 'ev-more' }, h('summary', {}, '三种缺失方式下每个字段的补全（检验期）'),
          pats.flatMap(([, p]) => [h('div', { class: 'section-title', style: { margin: '12px 0 6px' } }, p.zh),
            h('div', { class: 'ev-t' }, ui.table([
              { key: 'zh', label: '字段', wrap: true },
              { key: 'cls_zh', label: '内生性' },
              { key: 'r2', label: '补全 R²', fmt: (x) => sg(x, 2) },
              { key: 'crps_ratio', label: 'CRPS / 无条件', fmt: (x) => nn(x, 3) },
              { key: 'cov80', label: '80% 覆盖', fmt: (x) => pc(x) },
              { key: 'n', label: 'n' },
            ], p.rows || [], { stack: true }))])) : null));
    }
  }
  const dv = ev.delivery;
  if (dv) {
    const chart = ui.chartBox('');
    out.push(card({ title: '内置情景的季度情景表：陈述的视图交付了多少', sub: `${dv.n_views} 个视图里 ${dv.n_ok} 个落在月度引擎自己的交付容差以内（${dv.statistic}，${dv.n_paths} 条路径）。月度引擎的熵池收缩只交付了叙事里宏观路径的一部分；季度模型用条件模拟精确或在给定的标准差内满足陈述。` },
      chart,
      h('div', { class: 'ev-t' }, ui.table([
        { key: 'scenario', label: '情景', wrap: true },
        { value: (s) => `${s.ok}/${s.views}`, label: '达标的视图' },
        { key: 'median_gap_tol', label: '季度模型的缺口 / 容差（中位）', fmt: (x) => nn(x, 2) },
        { key: 'median_engine_gap_tol', label: '月度引擎的缺口 / 容差（中位）', fmt: (x) => nn(x, 2) },
        { key: 'dropped', label: '换算时放弃的视图' },
      ], dv.scenarios || [], { stack: true }))));
    const pts = (dv.views || []).filter(v => isNum(v.target) && isNum(v.quarterly)).map(v => ({ x: v.target, y: v.quarterly, name: `${v.scenario} · ${v.phase || ''}` }));
    if (pts.length) ctx.later.push(() => charts.scatter(chart, { points: pts, xName: '情景陈述的目标', yName: '季度模型交付的值', diagonal: true, fmtX: (v) => nn(v, 2), fmtY: (v) => nn(v, 2) }));
  }
  return h('div', { class: 'ev-stack' }, out);
}

// ================================================================ F. 前瞻账本, G. 预注册与选择记录

function sectionF(ev) {
  const led = ev.ledger;
  return card({ title: '前瞻账本：唯一干净的留出样本', sub: '每次构建把情景加权的季度分位原样追加，只追加、不改写；季度过去之后按实际结果打分（覆盖率、PIT、CRPS）。' },
    led ? h('p', { class: 'ev-head' }, `已存 ${led.n_entries} 条（起点 ${(led.origins || []).join('、') || '—'}；引擎 ${(led.engines || []).map(e => ({ monthly: '月度引擎', quarterly: '季度模型（试运行）' }[e] || '其他')).join('、')}），已到期可以打分的季度 ${led.matured} 个。${led.text || ''}`)
      : ui.notice('尚无记录：前瞻账本由一年期页的构建步骤写入（python -m scenariolab.build --only forecast）。', 'info'),
    h('p', { class: 'ev-note' }, h('a', { href: '#forecast' }, '未来一年：情景条件模拟 → 前瞻账本 →')));
}

function recordDetail(rec) {
  const box = h('div', {}, ui.loading('读取记录…'));
  api.get(`evaluation/records/${encodeURIComponent(rec.id)}`).then((full) => {
    if (!box.isConnected) return;
    const kv = [['文件', rec.file], ['SHA-256', rec.sha256], ['版本', rec.version], ['依据', rec.spec], ['冻结说明', rec.frozen],
      ['选了什么、用哪段数据', rec.selection], ['随机种子', rec.seeds ? JSON.stringify(rec.seeds) : null],
      ['数据 / 卡片 / 代码的哈希', rec.hashes ? Object.entries(rec.hashes).map(([k, v]) => `${{ data: '数据', card: '卡片', code: '代码' }[k] || k} ${short(v)}`).join('；') : null]]
      .filter(([, v]) => v !== null && v !== undefined && v !== '');
    put(box,
      h('dl', { class: 'ev-kv' }, kv.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, String(v))])),
      (rec.candidates || []).length ? h('ul', { class: 'list ev-note' }, rec.candidates.map(c => h('li', {}, `${c.id}：${c.text || ''}${c.report_only ? '（只作对照）' : ''}`))) : null,
      rec.last_result ? h('p', { class: 'ev-note' }, `最近一次结果（${rec.last_result.date || '—'}）：`, passChip(rec.last_result.passed), ' ', rec.last_result.text || '') : null,
      h('details', { class: 'ev-more' }, h('summary', {}, '原始记录（JSON）'), h('pre', { class: 'code ev-wrap' }, JSON.stringify(full.record, null, 2))),
      (full.results || []).length ? h('details', { class: 'ev-more' }, h('summary', {}, `追加的结果（${full.results.length} 次）`), h('pre', { class: 'code ev-wrap' }, JSON.stringify(full.results, null, 2))) : null);
  }).catch((err) => { if (box.isConnected) put(box, ui.notice(`记录读不了：${err.message}`, 'error')); });
  return box;
}

function sectionG(ev) {
  const recs = ev.records || [];
  const tests = ev.tests || {};
  const missing = Object.entries(tests).filter(([, t]) => t.status === 'missing').map(([id, t]) => ({ id: null, test: id, test_zh: t.zh, what: t.what, status: 'missing' }));
  const fl = ev.factor_layer;
  const rows = [...recs, ...missing, ...(fl ? [{ id: null, test: 'TF', test_zh: '因子引擎检验（T-F）与补全检验（T-C）的计算值', what: '因子层拟合时算出的通过 / 未通过，不是冻结的预注册记录', status: 'computed', date: fl.created }] : [])];
  const detailBox = h('div');
  const tbl = h('div', { class: 'ev-t ev-rec' }, ui.table([
    { value: (r) => r, label: '检验', wrap: true, fmt: (r) => h('div', {}, h('b', {}, r.test_zh || r.test), r.title ? h('small', { class: 'muted', style: { display: 'block' } }, r.title) : null) },
    { key: 'date', label: '日期', fmt: (x) => h('span', { style: { whiteSpace: 'nowrap' } }, x ? String(x).slice(0, 10) : '—') },
    { key: 'sha256', label: '哈希', fmt: (x) => (x ? h('span', { class: 'mono', title: x }, short(x)) : '—') },
    { value: (r) => r.selection || r.what || '—', label: '选了什么、用哪段数据', wrap: true },
    { value: (r) => r, label: '结果', wrap: true, fmt: (r) => {
      if (r.status === 'computed') return ui.chip('计算值（未冻结）');
      const [word, kind] = STATUS[r.status] || STATUS.missing;
      return h('div', {}, ui.chip(word, kind), r.last_result ? h('small', { style: { display: 'block' } }, r.last_result.text || '') : null);
    } },
    { value: (r) => r, label: '', fmt: (r) => (r.id && r.status !== 'unreadable' ? h('button', { class: 'btn', type: 'button', style: { whiteSpace: 'nowrap' }, onclick: () => { put(detailBox, card({ title: `${r.test_zh}：${r.id}` }, recordDetail(r))); detailBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } }, '查看记录') : null) },
  ], rows, { stack: true }));
  return h('div', { class: 'ev-stack' },
    card({ title: '冻结的预注册记录', sub: '每份记录在它的任何一次运行之前写定，结果只追加、不改写；改动只能另立新版本。哈希是记录文件本身的 SHA-256。' }, tbl, detailBox),
    card({ title: '污染说明', sub: '哪些数据已经被之前的选择看过' }, h('ul', { class: 'list' }, (ev.contamination_notes || []).map(n => h('li', {}, n)))));
}

// ================================================================ the page

export async function render(root, ctx) {
  ensureStyle();
  const { param } = ctx;
  const alive = ctx.alive || (() => root.isConnected);
  const later = [];
  const c2 = { ...ctx, alive, later };
  if (!ctx.rerender) Object.assign(STATE, { run: null, engine: null, cma: null, set: null, h: 'end', group: 'us_equity', asset: 'us_equity', arms: null });
  const runParam = /^run=([A-Za-z0-9_]{1,80})$/.exec(param || '');
  if (runParam) STATE.run = runParam[1];
  const sections = {};
  const anchor = (key, title) => (sections[key] = h('div', { class: 'section-title ev-anchor', id: `ev-${key}` }, title));
  const go = (key) => (e) => {
    const el = sections[key];
    if (!el) return;
    e.preventDefault();
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    ui.setRoute('evaluation', key, alive);
  };
  root.append(h('div', { class: 'ev-toc' }, SECTIONS.map(([k, t]) => h('a', { class: 'chip', href: `#evaluation?${k}`, onclick: go(k) }, t))));
  const boxA = h('div', {}, ui.loading('读取检验记录…'));
  root.append(anchor('conclusion', 'A. 结论'), boxA);
  const namesP = api.get('q/card').then(c => Object.fromEntries((c.rows || []).map(r => [r.id, r.zh]))).catch(() => ({}));
  let ev;
  try {
    ev = await api.get('evaluation');
  } catch (err) {
    if (!alive()) return null;
    put(boxA, ui.notice(`检验记录读不了：${err.message}`, 'error'));
    return null;
  }
  if (!alive()) return null;
  // the headline run's report: section A quotes its headline (C loads the same file from the cache)
  let rep = null;
  const head = STATE.run || ev.narrative?.headline_run;
  if (ev.narrative?.headline_run) {
    try { rep = await api.get(`evaluation/narrative/runs/${encodeURIComponent(ev.narrative.headline_run)}`); } catch (_) { rep = null; }
    if (!alive()) return null;
  }
  NAMES = await namesP;
  if (!alive()) return null;
  put(boxA, safe('结论', () => sectionA(ev, rep)));
  if (head && !STATE.run) STATE.run = head;
  root.append(anchor('oracle', 'B. 已知因子路径时能复现多少'), safe('已知因子路径', () => sectionB(ev)));
  root.append(anchor('narrative', 'C. 叙事检验'), safe('叙事检验', () => sectionC(ev, c2)));
  root.append(anchor('valuation', 'D. 估值账本'), safe('估值账本', () => sectionD(ev)));
  root.append(anchor('engine', 'E. 因子引擎检验'), safe('因子引擎检验', () => sectionE(ev, c2)));
  root.append(anchor('ledger', 'F. 前瞻账本'), safe('前瞻账本', () => sectionF(ev)));
  root.append(anchor('records', 'G. 预注册与选择记录'), safe('预注册记录', () => sectionG(ev)));
  root.append(h('ul', { class: 'list ev-note', style: { marginTop: '16px' } }, (ev.notes || []).map(n => h('li', {}, n))));
  for (const f of later.splice(0)) { try { f(); } catch (err) { console.error(err); } }
  const key = runParam ? 'narrative' : param;
  return key && sections[key] ? sections[key] : null;
}
