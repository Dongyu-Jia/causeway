// 情景实验室：季度情景表（QTable/1; SIM_DESIGN §10.4, §7, Appendix B). Narrative -> QTable (the LLM only translates; every
// value carries a quote from the text) -> validation report (refusals, warnings, the R1-R5 requirements left to the model
// with the narrative's evidence sentence) -> run on the quarterly stack (试运行) or on the monthly engine (the same QTable
// converted to a phase spec; the dropped views are listed) -> results.
// Editing: per-quarter cells (any of the card's x fields and policy shocks), the financial-conditions stance of every
// quarter, a valuation-mode control and, after a quarterly run, the rarity statistic per quarter. A cell the user fills
// is the user's own statement: its note is its quote and is appended to the narrative under 【我的设定】, so every value
// keeps a textual basis.
// A real translation with the local Claude Code spends the user's plan: the server answers 409 with the estimate, and
// the page asks for an explicit tick before it sends confirm: true. Nothing an LLM wrote is set as HTML.
// Deep link: #lab?q=<built-in id> loads that built-in's reviewed QTable.
import { put, renderQuarterlyDetail, runQuarterly, badge, completedBadge, badgeLegend, ensureQStyle, setCardNames, span, TRIAL, VAL_WORDING } from './_qdetail.js';
import { renderScenarioDetail, ensureDetailStyle } from './_scenario_detail.js';
import { fieldLabel } from './card.js';             // the 变量卡片 tooltip on the QTable's column headers (P8a)
import { loadSwitch } from './_switch.js';

const DRAFT_KEY = 'msl-qlab-draft';
const draft = (() => {
  try { const d = JSON.parse(sessionStorage.getItem(DRAFT_KEY) || 'null'); return d && typeof d === 'object' ? d : {}; } catch (_) { return {}; }
})();
function saveDraft(patch) {
  Object.assign(draft, patch);
  try { sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draft)); } catch (_) { /* private mode: keep it in memory */ }
}

const EXAMPLES = [
  { title: '油价供给冲击后加息', text: '产油国卷入战争，供应中断，油价一个季度内上涨约三成，商品价格明显上涨，地缘政治风险急剧上升，信用利差走阔。下一个季度央行加息 50 个基点。之后战事缓和，油价逐步回落，金融条件大体平稳，失业率基本稳定。' },
  { title: '银行业恐慌与紧急降息', text: '几家中型银行接连出现挤兑，融资市场压力骤升，信用利差明显走阔。央行在两次议息会议之间紧急降息 50 个基点，并宣布购买国债以压低期限溢价和 2 年期收益率。之后两个季度信贷收紧，失业率上升约 0.5 个百分点，通胀回落。' },
  { title: '需求放缓、温和衰退', text: '消费和企业投资同时放缓，工业产出连续两个季度下降，失业率每季上升约 0.3 个百分点，服务通胀回落。央行随之降息，金融条件温和收紧，之后经济在第四个季度企稳。' },
];
const MAIN_FIELDS = ['d_ff', 'd_unrate', 'd_ip', 'd_cpi_goods', 'd_cpi_serv', 'd_oil', 'x_oil_cause', 'd_y10', 'd_gz'];
const STANCES = ['明显收紧', '收紧', '中性', '宽松', '明显宽松', '未说明'];
const SIZES = ['小', '温和', '明显', '大幅', '极端'];
const VAL_SIZES = ['温和', '明显', '极端'];
const OIL_CAUSE = { supply: '供给冲击', demand: '需求推动', mixed: '供给与需求混合' };
const KIND_ZH = { mean: '平均', sum: '合计', level: '水平', yoy: '同比', range: '范围' };
const LEVEL_ZH = { s_ff: '政策利率水平', s_unrate: '失业率水平', s_cpi_yoy: 'CPI 同比', s_core_yoy: '核心 CPI 同比' };
const REGIME_ZH = { expansion: '扩张', inflation: '通胀', deflation: '物价负增长', recession: '衰退', depression: '深度衰退', stagflation: '滞胀', contraction: '收缩', high_infl: '高通胀' };
const ANCHOR_ZH = { r_star: '中性实际利率 r*', pi_lr: '长期通胀', a_tp: '期限溢价锚', a_spread: '信用利差锚', a_ustar: '失业率锚', a_oil: '实际油价锚移位', a_usd: '实际美元锚移位' };
const STATUS_ZH = { ok: '翻译完成', repaired: '翻译完成（格式修复过一次）', fallback: '没有得到可用的翻译：退回到空白情景表（全部由模型补全）' };
const ERR_KIND = { refusal: '模型拒绝回答', timeout: '超时', overloaded: '服务繁忙', max_tokens: '输出被截断', empty: '空回答', invalid: '格式无效' };

const CSS = `
.ql-hero { border: 1px solid #cdd7ff; border-left: 4px solid var(--accent); background: linear-gradient(135deg, #f3f6ff, #fff 70%);
  border-radius: var(--radius); padding: 14px 16px; margin-bottom: 16px; }
.ql-hero h2 { margin: 0 0 10px; font-size: 16px; font-weight: 650; line-height: 1.55; color: #1f2f7a; }
.ql-flow { display: flex; flex-wrap: wrap; gap: 6px; align-items: stretch; }
.ql-step { flex: 1 1 150px; min-width: 0; background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: 12px; color: var(--ink-2); line-height: 1.5; }
.ql-step b { display: block; color: var(--ink); font-size: 13px; margin-bottom: 2px; }
.ql-bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 10px; }
.ql-bar label { display: flex; gap: 6px; align-items: center; font-size: 12px; color: var(--ink-3); }
.ql-bar input[type=number] { width: 90px; }
.ql-bar select { max-width: 100%; min-width: 0; }
.ql-msg { font-size: 12px; color: var(--ink-3); display: inline-flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.ql-msg.ok { color: var(--good); } .ql-msg.err { color: var(--bad); }
.ql-ex { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; align-items: center; }
.ql-ex button { font: inherit; font-size: 12px; border: 1px dashed #c6cbd6; background: var(--panel-2); border-radius: 999px; padding: 3px 10px; cursor: pointer; color: var(--ink-2); }
textarea.ql-narr { min-height: 110px; }
textarea.ql-json { font-family: var(--mono); font-size: 12px; min-height: 280px; line-height: 1.45; }
.ql-cellbtn { font: inherit; font-size: 12px; width: 100%; min-height: 30px; text-align: left; border: 1px dashed transparent; background: transparent;
  border-radius: 6px; padding: 2px 4px; cursor: pointer; color: var(--ink); }
.ql-cellbtn:hover { border-color: var(--accent); background: var(--accent-soft); }
.ql-cellbtn.on { border-color: var(--accent); border-style: solid; }
.ql-cellbtn .v { display: block; font-variant-numeric: tabular-nums; }
.ql-cellbtn .empty { color: var(--ink-3); }
.ql-grid table.tbl th { white-space: normal; line-height: 1.35; vertical-align: bottom; cursor: help; }
.ql-grid table.tbl td { vertical-align: top; padding: 4px 6px; }
.ql-grid select { font-size: 12px; padding: 3px 6px; max-width: 100%; }
.ql-editor { border: 1px solid #cdd7ff; background: #f8faff; border-radius: 8px; padding: 12px; margin-top: 12px; }
.ql-editor h4 { margin: 0 0 4px; font-size: 14px; }
.ql-editor .def { font-size: 12px; color: var(--ink-3); margin: 0 0 8px; }
.ql-form { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(170px, 100%), 1fr)); gap: 8px 12px; align-items: end; }
.ql-form label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; color: var(--ink-3); min-width: 0; }
.ql-form input, .ql-form select { width: 100%; min-width: 0; }
.ql-form .wide { grid-column: 1 / -1; }
.ql-list { margin: 0; padding-left: 18px; font-size: 13px; }
.ql-list li { margin: 4px 0; }
.ql-list .btn { padding: 1px 8px; font-size: 12px; margin-left: 6px; }
.ql-quote { display: block; color: var(--ink-3); font-size: 12px; }
.ql-check ul { margin: 4px 0 0; padding-left: 18px; }
.ql-check li { margin: 3px 0; }
.ql-ev { display: block; font-size: 12px; color: var(--ink-3); }
.ql-confirm { border: 1px solid #f0c48a; background: #fff8ec; border-radius: 8px; padding: 10px 12px; margin-top: 10px; font-size: 13px; }
.ql-confirm label { display: flex; gap: 8px; align-items: center; margin: 8px 0; }
pre.code.ql-wrap { white-space: pre-wrap; word-break: break-word; max-height: 360px; overflow: auto; }
`;
function ensureStyle() {
  let el = document.getElementById('style-qlab');
  if (!el) { el = document.createElement('style'); el.id = 'style-qlab'; document.head.append(el); }
  if (!el.textContent.includes('/* qlab */')) el.textContent += `/* qlab */\n${CSS}`;
}

const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
const g = (v) => (isNum(v) ? String(Number(v.toFixed(3))) : '—');
const sg = (v) => (isNum(v) ? `${v > 0 ? '+' : ''}${Number(v.toFixed(3))}` : '—');
const clone = (x) => JSON.parse(JSON.stringify(x));
const minimal = (H) => ({ schema: 'qtable/1', n_quarters: H, rows: Array.from({ length: H }, (_, i) => ({ q: i + 1, x: {}, fc: { stance: '未说明' } })),
  c_q: {}, events: [], notes: [] });

export async function renderQLab(root, ctx) {
  loadSwitch();                   // the switch gate's state, for the result header's 试运行 / 默认 chip (P9)
  const { api, ui, param } = ctx;
  const alive = ctx.alive || (() => root.isConnected);
  const { h, card } = ui;
  ensureStyle();
  ensureQStyle();
  ensureDetailStyle();

  // ---- the card (field names, units, definitions, permissions) and the built-ins
  let cardRows = {}, xFields = [], uFields = [];
  const cardP = api.get('q/card').then((c) => {
    setCardNames(c);
    cardRows = Object.fromEntries((c.rows || []).map(r => [r.id, r]));
    xFields = (c.rows || []).filter(r => r.group === 'x').map(r => r.id);
    uFields = (c.rows || []).filter(r => r.group === 'u').map(r => r.id);
    return c;
  }).catch(() => null);
  const builtinsP = api.get('q/scenarios').catch(() => null);
  const zh = (f) => LEVEL_ZH[f] || cardRows[f]?.zh || '（未命名字段）';
  // the monthly engine's notes name its view variables by id (f_ff, s_cpi_yoy): their names instead
  const readable = (t) => String(t ?? '').replace(/\b[fs]_[a-z0-9_]+\b/g, (m) => (ui.varName(m) && ui.varName(m) !== m ? `「${ui.varName(m)}」` : m));
  const unit = (f) => cardRows[f]?.unit || '';

  // ---- state
  const S = {
    narr: typeof draft.narr === 'string' ? draft.narr : '',
    nq: Number(draft.nq) || 8,
    name: draft.name || '我的季度情景',
    qt: draft.qt && Array.isArray(draft.qt.rows) ? draft.qt : null,
    origin: draft.origin || 'blank',            // llm | builtin | user | blank
    builtinId: draft.builtinId || null,
    edited: !!draft.edited,
    notes: Array.isArray(draft.notes) ? draft.notes : [],
    extra: Array.isArray(draft.extra) ? draft.extra : [],     // extra columns the user added
    engine: draft.engine || 'quarterly',
    paths: Number(draft.paths) || 1000,
    cell: null,                                  // {q, f} being edited
    run: null,
    span: draft.span || '12',                    // rows shown: '8' | '12' | 'all'
  };
  const keep = () => saveDraft({ narr: S.narr, nq: S.nq, name: S.name, qt: S.qt, origin: S.origin, builtinId: S.builtinId,
    edited: S.edited, notes: S.notes, extra: S.extra, engine: S.engine, paths: S.paths, span: S.span });
  const fullNarrative = () => (S.notes.length ? `${S.narr}\n【我的设定】${S.notes.join('；')}` : S.narr);
  const source = () => (S.edited || S.origin === 'user' || S.origin === 'blank' ? 'user' : S.origin === 'builtin' ? 'builtin' : 'translator');

  // ---- hero
  const step = (t, x) => h('div', { class: 'ql-step' }, h('b', {}, t), x);
  root.append(h('div', { class: 'ql-hero' },
    h('h2', {}, '大模型只把文字翻译成季度情景表：每个值都带原文引文；文字没说的留空，由模型补全并标出来。所有数字由模型计算。'),
    h('div', { class: 'ql-flow' },
      step('① 叙事', '发生了什么、持续几个季度、怎么传导；金融条件（信用利差的方向）每个阶段都要说'),
      step('② 季度情景表', '每季每个字段：数值、方向与幅度档或范围，带来源和引文；没写的留空'),
      step('③ 校验', '拒绝没有文字依据的值、市场价格和估值数字；列出由模型补全的部分和证据句'),
      step('④ 运行', `季度模型（${TRIAL}）或月度引擎（同一张表换算成阶段规格）`),
      step('⑤ 结果', '逐季的陈述与补全、资产区间、估值、归因、金融条件'))));

  // ---- ① narrative
  const narr = h('textarea', { class: 'ql-narr', placeholder: '例如：油价一个季度内上涨约三成，央行加息 50 个基点，信用利差走阔……写清楚每个季度发生什么，以及金融条件的方向。',
    oninput: () => { S.narr = narr.value; keep(); scheduleValidate(); } });
  narr.value = S.narr;
  const nqIn = h('input', { type: 'number', min: '1', max: '60', step: '1', value: String(S.nq), onchange: () => { S.nq = Math.min(Math.max(Math.round(Number(nqIn.value)) || 8, 1), 60); nqIn.value = String(S.nq); keep(); } });
  const nameIn = h('input', { type: 'text', value: S.name, style: { width: '180px', maxWidth: '100%' }, oninput: () => { S.name = nameIn.value.slice(0, 60); keep(); } });
  const estBtn = h('button', { class: 'btn', type: 'button', onclick: () => estimate() }, '估算调用');
  const trBtn = h('button', { class: 'btn primary', type: 'button', onclick: () => translate(false) }, '翻译成季度情景表');
  const trMsg = h('span', { class: 'ql-msg' });
  const confirmBox = h('div');
  const backendLine = h('p', { class: 'muted', style: { fontSize: '12px', margin: '8px 0 0' } });
  const tplSel = h('select', { 'aria-label': '内置情景' }, h('option', { value: '' }, '载入内置情景…'));
  root.append(card({ title: '① 写下情景叙事', sub: '点一个例子或自己写；也可以直接载入一个内置情景的季度情景表，或从空白表开始手填。' },
    h('div', { class: 'ql-ex' }, h('span', { class: 'muted', style: { fontSize: '12px' } }, '例子：'),
      EXAMPLES.map(ex => h('button', { type: 'button', title: ex.text, onclick: () => { narr.value = ex.text; S.narr = ex.text; keep(); scheduleValidate(); } }, ex.title))),
    narr,
    h('div', { class: 'ql-bar' }, h('label', {}, '季度数（1–60）', nqIn), h('label', {}, '情景名称', nameIn), estBtn, trBtn, trMsg),
    h('div', { class: 'ql-bar' }, tplSel, h('button', { class: 'btn', type: 'button', onclick: () => loadBuiltin(tplSel.value) }, '载入'),
      h('button', { class: 'btn', type: 'button', onclick: () => blank() }, '空白情景表')),
    confirmBox, backendLine));
  const report = h('div');
  root.append(report);

  // ---- ② the table
  const gridBox = h('div', { class: 'ql-grid' });
  const editorBox = h('div');
  const cqBox = h('div');
  const valBox = h('div');
  const addSel = h('select', { 'aria-label': '添加字段' }, h('option', { value: '' }, '添加一列字段…'));
  const allFc = ui.select([{ value: '', label: '全部金融条件设为…' }, ...STANCES.map(s => ({ value: s, label: s }))], '', (v) => { if (v) setAllStances(v); });
  const jsonArea = h('textarea', { class: 'ql-json', spellcheck: 'false' });
  const jsonMsg = h('span', { class: 'ql-msg' });
  const tableCard = card({ title: '② 季度情景表（QTable）', sub: '每一行是一个季度；点一个格子在下方编辑（你填的值是你自己的设定，说明会作为它的引文）。空格 = 情景没有说，由模型补全。金融条件每一季都要选；“未说明”时信用利差由模型补全。' },
    h('div', { class: 'ql-bar', style: { marginTop: 0, marginBottom: '8px' } }, addSel,
      h('button', { class: 'btn', type: 'button', onclick: () => addCol() }, '添加'), allFc,
      h('button', { class: 'btn', type: 'button', onclick: () => resize(+1) }, '加一季'),
      h('button', { class: 'btn', type: 'button', onclick: () => resize(-1) }, '减一季'),
      ui.seg([{ value: '8', label: '前 8 季' }, { value: '12', label: '前 12 季' }, { value: 'all', label: '全部' }], S.span,
        (v) => { S.span = v; keep(); paintGrid(); })),
    badgeLegend(), gridBox, editorBox,
    h('div', { style: { height: '12px' } }),
    h('div', { class: 'section-title', style: { margin: '6px 0 6px' } }, '跨季度的陈述（约束、预告路径、体制请求、长期锚）'), cqBox,
    h('div', { class: 'section-title', style: { margin: '14px 0 6px' } }, '估值模式'), valBox,
    h('details', { class: 'qd-more', style: { marginTop: '12px' } }, h('summary', {}, '高级：直接编辑 JSON'),
      h('p', { class: 'muted', style: { fontSize: '12px', margin: '6px 0' } }, '季度情景表 QTable/1 的 JSON；应用之后上面的表格和校验随之更新。'),
      jsonArea, h('div', { class: 'ql-bar' }, h('button', { class: 'btn', type: 'button', onclick: () => applyJson() }, '应用 JSON'), jsonMsg)));
  root.append(h('div', { style: { height: '16px' } }), tableCard);

  // ---- ③ validation
  const checkBox = h('div', { class: 'ql-check' }, ui.notice('还没有情景表：翻译一段叙事、载入内置情景，或从空白表开始。', 'info'));
  root.append(h('div', { style: { height: '16px' } }), card({ title: '③ 校验报告', sub: '每一处拒绝、截断和“由模型补全”都列在这里；补全的部分附上叙事里的证据句（没有就写明没有）。' }, checkBox));

  // ---- ④ run
  const engSeg = ui.seg([{ value: 'quarterly', label: `季度模型（${TRIAL}）` }, { value: 'monthly', label: '月度引擎（换算成阶段规格）' }], S.engine,
    (v) => { S.engine = v; keep(); paintRunNote(); });
  const pathsIn = h('input', { type: 'number', min: '100', max: '4000', step: '100', value: String(S.paths), onchange: () => { S.paths = Math.min(Math.max(Math.round(Number(pathsIn.value)) || 1000, 100), 4000); pathsIn.value = String(S.paths); keep(); } });
  const runBtn = h('button', { class: 'btn primary', type: 'button', onclick: () => run() }, '运行');
  const runMsg = h('span', { class: 'ql-msg' });
  const runNote = h('div');
  root.append(h('div', { style: { height: '16px' } }), card({ title: '④ 运行', sub: '季度模型在切换关通过之前是试运行；月度引擎是现行默认，同一张表换算成它的阶段规格，换算不了的字段会列出来。' },
    h('div', { class: 'ql-bar', style: { marginTop: 0 } }, engSeg, h('label', {}, '路径数', pathsIn), runBtn, runMsg), runNote));
  const resultHead = h('div', { class: 'section-title', style: { scrollMarginTop: '72px' } }, '⑤ 结果');
  const resultBox = h('div', {}, ui.notice('运行之后，结果显示在这里。', 'info'));
  root.append(resultHead, resultBox);

  const say = (el, text, kind = '') => { el.className = `ql-msg ${kind}`; el.textContent = text; };
  const ticking = (el, text) => {
    const t0 = Date.now();
    const tick = () => { el.className = 'ql-msg'; el.replaceChildren(h('span', { class: 'spinner' }), `${text}（${Math.round((Date.now() - t0) / 1000)} 秒）`); };
    tick();
    const id = setInterval(() => { if (!el.isConnected) clearInterval(id); else tick(); }, 1000);
    return () => clearInterval(id);
  };

  // ---------------------------------------------------------------- the table: rendering
  const columns = () => {
    const present = new Set();
    for (const r of S.qt?.rows || []) { Object.keys(r.x || {}).forEach(f => present.add(f)); Object.keys(r.u || {}).forEach(f => present.add(f)); }
    return [...new Set([...MAIN_FIELDS, ...S.extra, ...present])];
  };
  const entryText = (f, e) => {
    if (!e) return null;
    if (f === 'x_oil_cause') return OIL_CAUSE[e.value] || String(e.value || '');
    if ('sign' in e) return `${{ '+': '上升', '-': '下降', 0: '不变', '0': '不变' }[e.sign] || e.sign}·${e.size || ''}`;
    if ('value' in e) return `${sg(e.value)}${isNum(e.sd) ? `（±${g(e.sd)}）` : '（硬）'}`;
    if ('min' in e || 'max' in e) return `${isNum(e.min) ? g(e.min) : '…'} ~ ${isNum(e.max) ? g(e.max) : '…'}`;
    return '—';
  };
  // the c_q items (constraints, announced paths) on a field that cover quarter q; level constraints show in their
  // change field's column (s_ff -> d_ff, s_unrate -> d_unrate), CPI y/y ones in a read-only column
  const LEVEL_COL = { s_ff: 'd_ff', s_unrate: 'd_unrate' };
  const cqItems = () => [...(S.qt?.c_q?.constraints || []), ...(S.qt?.c_q?.announced || [])];
  const covering = (q, col) => cqItems().filter(c => (LEVEL_COL[c.field] || c.field) === col && (c.quarters || []).includes(q));
  const roCols = () => [...new Set(cqItems().map(c => LEVEL_COL[c.field] || c.field).filter(f => !f.startsWith('d_') && !f.startsWith('x_')))];
  const covText = (c) => `${zh(c.field)} 第 ${span(c.quarters || [])} 季${'values' in c ? '预告路径' : KIND_ZH[c.kind] || ''} `
    + (isNum(c.value) ? g(c.value) : 'values' in c ? (c.values || []).map(g).join('、') : `${isNum(c.min) ? g(c.min) : '…'} ~ ${isNum(c.max) ? g(c.max) : '…'}`);
  const covNode = (q, col) => covering(q, col).map(c => h('div', { title: c.quote ? `原文：「${c.quote}」` : '' },
    badge({ badge: 'stated', text: covText(c), quote: c.quote }), h('small', { class: 'muted' }, '约束')));
  function paintGrid() {
    if (!S.qt) { put(gridBox, ui.notice('还没有情景表。', 'info')); put(cqBox); put(valBox); jsonArea.value = ''; return; }
    const cols = columns();
    const ro = roCols();
    const regime = S.qt.c_q?.regime || [];
    const shown = S.span === 'all' ? S.qt.rows : S.qt.rows.slice(0, Number(S.span) || 12);
    const rar = Object.fromEntries((S.run?.rarity?.per_quarter || []).map(x => [x.q, x]));
    const head = (f) => (cardRows[f] ? fieldLabel(f, zh(f), cardRows[f]) : h('span', {}, zh(f)));
    put(gridBox, ui.table([
      { key: 'q', label: '季度', fmt: (q) => h('b', {}, `第 ${q} 季`) },
      ...cols.map(f => ({ value: (r) => r, label: head(f), fmt: (r) => {
        const isU = f.startsWith('u_');
        const e = isU ? r.u?.[f] : r.x?.[f];
        const on = S.cell && S.cell.q === r.q && S.cell.f === f;
        const cov = covNode(r.q, f);
        return h('div', {}, h('button', { type: 'button', class: `ql-cellbtn${on ? ' on' : ''}`, title: e?.quote ? `原文：「${e.quote}」` : '点击填写（空 = 模型补全）',
          onclick: () => openCell(r.q, f) },
        e ? [h('span', { class: 'v' }, entryText(f, e)), badge({ ...e, badge: isU ? 'intervention' : e.src, text: entryText(f, e) })]
          : h('span', { class: 'empty' }, cov.length ? '＋（另有约束）' : '＋')), cov);
      } })),
      ...ro.map(f => ({ value: (r) => r, label: head(f), fmt: (r) => { const cov = covNode(r.q, f); return cov.length ? h('div', {}, cov) : h('span', { class: 'muted' }, '—'); } })),
      ...(regime.length ? [{ value: (r) => regime.filter(x => (x.quarters || []).includes(r.q)), label: '体制请求',
        fmt: (xs) => (xs.length ? h('div', {}, xs.map(x => h('div', { title: x.quote ? `原文：「${x.quote}」` : '' },
          badge({ badge: 'stated_qual', text: `${REGIME_ZH[x.label] || '体制'} ${Math.round((x.share || 0) * 100)}%`, quote: x.quote }),
          h('small', {}, `${REGIME_ZH[x.label] || '体制'} ${Math.round((x.share || 0) * 100)}%`)))) : '—') }] : []),
      { value: (r) => r, label: '金融条件', fmt: (r) => {
        const sel = ui.select(STANCES.map(s => ({ value: s, label: s })), r.fc?.stance || '未说明', (v) => setStance(r.q, v));
        sel.setAttribute('aria-label', `第 ${r.q} 季的金融条件`);
        return sel;
      } },
      ...(S.run ? [{ value: (r) => rar[r.q], label: '稀有度', fmt: (x) => (x && x.df > 0 ? h('span', { class: x.percentile > 0.99 ? 'is-bad' : '', title: `需要约 ${g(x.sd)} 个标准差的冲击` }, `${Math.round(x.percentile * 100)}% 分位`) : '—') }] : []),
    ], shown, { stack: true }),
    shown.length < S.qt.rows.length ? h('p', { class: 'muted', style: { fontSize: '12px', margin: '6px 0 0' } }, `只显示前 ${shown.length} 季（共 ${S.qt.rows.length} 季）；选“全部”看其余季度。`) : null);
    paintCq();
    paintValuation();
    jsonArea.value = JSON.stringify(S.qt, null, 2);
  }

  // c_q: constraints, announced paths, regime requests, anchors (readable, deletable)
  function paintCq() {
    const c = S.qt?.c_q || {};
    const items = [];
    (c.constraints || []).forEach((x, i) => items.push({ kind: 'constraints', i, text: `${zh(x.field)} 第 ${span(x.quarters || [])} 季${KIND_ZH[x.kind] || ''} `
      + (isNum(x.value) ? `${g(x.value)}${isNum(x.sd) ? `（±${g(x.sd)}）` : ''}` : `${isNum(x.min) ? g(x.min) : '…'} ~ ${isNum(x.max) ? g(x.max) : '…'}`), quote: x.quote, b: 'stated' }));
    (c.announced || []).forEach((x, i) => items.push({ kind: 'announced', i, text: `预告路径：${zh(x.field)} 第 ${span(x.quarters || [])} 季依次 ${(x.values || []).map(g).join('、')}`, quote: x.quote, b: 'stated' }));
    (c.regime || []).forEach((x, i) => items.push({ kind: 'regime', i, text: `体制请求：${REGIME_ZH[x.label] || '体制'} 占 ${Math.round((x.share || 0) * 100)}%（第 ${span(x.quarters || [])} 季）`, quote: x.quote, b: 'stated_qual' }));
    Object.entries(c.anchors || {}).forEach(([a, x]) => items.push({ kind: 'anchors', i: a, text: `${ANCHOR_ZH[a] || '长期锚'} ${g(x.value)}（第 ${x.from_q || 1} 季起）`, quote: x.quote, b: 'anchor' }));
    put(cqBox, items.length ? h('ul', { class: 'ql-list' }, items.map(it => h('li', {}, badge({ badge: it.b, text: it.text, quote: it.quote }), it.text,
      h('button', { class: 'btn', type: 'button', onclick: () => dropCq(it.kind, it.i) }, '删除'),
      it.quote ? h('span', { class: 'ql-quote' }, `原文：「${it.quote}」`) : null)))
      : h('p', { class: 'muted', style: { margin: 0, fontSize: '13px' } }, '没有跨季度的陈述（可以在“高级：直接编辑 JSON”里添加约束、预告路径、体制请求或长期锚）。'));
  }

  // the valuation-mode control: one statement over a span of quarters (more in the JSON)
  function paintValuation() {
    const list = S.qt?.c_q?.valuation || [];
    const v = list[0] || null;
    const H = S.qt?.n_quarters || S.qt?.rows?.length || 1;
    const mode = ui.select([{ value: 'inferred', label: '由宏观与信用隐含（默认）' }, { value: 'hold', label: '倍数保持不变' }, { value: 'rerate', label: '倍数重估' }],
      v?.mode || 'inferred', () => {});
    const dir = ui.select([{ value: '+', label: '上调' }, { value: '-', label: '下调' }], v?.direction === '-' || v?.direction === -1 ? '-' : '+', () => {});
    const size = ui.select(VAL_SIZES.map(s => ({ value: s, label: s })), v?.size || '明显', () => {});
    const qs = v?.quarters || [1, Math.min(4, H)];
    const from = h('input', { type: 'number', min: '1', max: String(H), value: String(Math.min(...qs)) });
    const to = h('input', { type: 'number', min: '1', max: String(H), value: String(Math.max(...qs)) });
    const note = h('input', { type: 'text', value: v?.quote || '', placeholder: '说明（会作为引文）' });
    put(valBox, 
      h('div', { class: 'ql-form' },
        h('label', {}, '模式', mode), h('label', {}, '方向（重估时）', dir), h('label', {}, '幅度（重估时）', size),
        h('label', {}, '从第几季', from), h('label', {}, '到第几季', to), h('label', { class: 'wide' }, '说明', note)),
      h('div', { class: 'ql-bar' }, h('button', { class: 'btn', type: 'button', onclick: () => {
        const m = mode.value;
        if (m === 'inferred') { if (S.qt.c_q) delete S.qt.c_q.valuation; } else {
          const a = Math.min(Math.max(Math.round(Number(from.value)) || 1, 1), H), b = Math.min(Math.max(Math.round(Number(to.value)) || a, a), H);
          const quote = note.value.trim() || `我设定：估值${m === 'hold' ? '倍数保持不变' : `倍数${dir.value === '-' ? '下调' : '上调'}（${size.value}）`}`;
          S.qt.c_q = S.qt.c_q || {};
          S.qt.c_q.valuation = [{ quarters: Array.from({ length: b - a + 1 }, (_, k) => a + k), mode: m, ...(m === 'rerate' ? { direction: dir.value, size: size.value } : {}), quote }];
          addNote(quote);
        }
        edited();
      } }, '应用估值模式'),
      ui.notice(`估值假设${VAL_WORDING}：写“倍数保持”或“重估”等于直接写出一段股价路径；默认的“由宏观与信用隐含”不加任何估值陈述，不回归。`, 'info')),
      list.length > 1 ? h('p', { class: 'muted', style: { fontSize: '12px' } }, `另有 ${list.length - 1} 条估值陈述（在 JSON 里编辑）。`) : null);
  }

  // ---------------------------------------------------------------- the table: editing
  function openCell(q, f) {
    S.cell = { q, f };
    paintGrid();
    const r = S.qt.rows.find(x => x.q === q);
    const isU = f.startsWith('u_');
    const e = (isU ? r.u?.[f] : r.x?.[f]) || null;
    const form = e ? ('sign' in e ? 'qual' : ('min' in e || 'max' in e) && !('value' in e) ? 'range' : 'value') : 'value';
    const kind = ui.select([{ value: 'value', label: '数值（可带标准差）' }, { value: 'qual', label: '方向与幅度档' }, { value: 'range', label: '范围' }], form, () => showFields());
    const value = h('input', { type: 'number', step: 'any', value: isNum(e?.value) ? String(e.value) : '' });
    const sd = h('input', { type: 'number', step: 'any', min: '0', value: isNum(e?.sd) ? String(e.sd) : '', placeholder: '不填 = 硬约束' });
    const sign = ui.select([{ value: '+', label: '上升' }, { value: '-', label: '下降' }, { value: '0', label: '不变' }], e?.sign || '+', () => {});
    const size = ui.select(SIZES.map(s => ({ value: s, label: s })), e?.size || '明显', () => {});
    const lo = h('input', { type: 'number', step: 'any', value: isNum(e?.min) ? String(e.min) : '' });
    const hi = h('input', { type: 'number', step: 'any', value: isNum(e?.max) ? String(e.max) : '' });
    const cause = ui.select(Object.entries(OIL_CAUSE).map(([k, v]) => ({ value: k, label: v })), e?.value || 'supply', () => {});
    const note = h('input', { type: 'text', value: e?.quote || '', placeholder: '说明（会作为这一格的引文；不填就用“我设定：…”）' });
    const fieldsBox = h('div', { class: 'ql-form' });
    const msg = h('span', { class: 'ql-msg' });
    function showFields() {
      const k = kind.value;
      fieldsBox.replaceChildren(...(f === 'x_oil_cause' ? [h('label', {}, '原因', cause)]
        : k === 'value' ? [h('label', {}, `数值（${unit(f) || '字段单位'}）`, value), h('label', {}, '标准差（可选）', sd)]
          : k === 'qual' ? [h('label', {}, '方向', sign), h('label', {}, '幅度档（历史标准差的倍数）', size)]
            : [h('label', {}, '下限', lo), h('label', {}, '上限', hi)]),
      h('label', { class: 'wide' }, '说明', note));
    }
    showFields();
    const row = cardRows[f] || {};
    editorBox.replaceChildren(h('div', { class: 'ql-editor' },
      h('h4', {}, `第 ${q} 季 · ${zh(f)}`),
      h('p', { class: 'def' }, [row.definition, unit(f) ? `单位 ${unit(f)}` : '', row.endo, row.translator ? `翻译器：${row.translator}` : ''].filter(Boolean).join(' · '),
        row.id ? [' ', h('a', { href: `#card?${encodeURIComponent(f)}` }, '变量卡片 →')] : null),
      f === 'x_oil_cause' ? null : h('div', { class: 'ql-bar', style: { marginTop: 0, marginBottom: '8px' } }, h('label', {}, '写法', kind)),
      fieldsBox,
      h('div', { class: 'ql-bar' },
        h('button', { class: 'btn primary', type: 'button', onclick: () => save() }, '保存这一格'),
        e ? h('button', { class: 'btn', type: 'button', onclick: () => { setEntry(q, f, null); close(); } }, '清空（交给模型补全）') : null,
        h('button', { class: 'btn', type: 'button', onclick: () => close() }, '取消'), msg),
      isU ? ui.notice('政策冲击只用于文字写明的干预（紧急或意外的政策动作、扩表缩表、退税支票、减税法案等）：同季的宏观不随它补全，市场价格当季反应。', 'info') : null));
    editorBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    function save() {
      let entry;
      const k = kind.value;
      if (f === 'x_oil_cause') entry = { value: cause.value, src: 'stated_qual' };
      else if (k === 'value') {
        const v = Number(value.value);
        if (value.value.trim() === '' || !Number.isFinite(v)) { say(msg, '请填一个数。', 'err'); return; }
        entry = { value: v, src: 'stated' };
        if (sd.value.trim() !== '') {
          const s = Number(sd.value);
          if (!(s > 0)) { say(msg, '标准差必须是正数（不填 = 硬约束）。', 'err'); return; }
          entry.sd = s;
        }
      } else if (k === 'qual') entry = { sign: sign.value, size: size.value, src: 'stated_qual' };
      else {
        const a = lo.value.trim() === '' ? null : Number(lo.value), b = hi.value.trim() === '' ? null : Number(hi.value);
        if (a === null && b === null) { say(msg, '下限和上限至少填一个。', 'err'); return; }
        if (a !== null && b !== null && a > b) { say(msg, '下限大于上限。', 'err'); return; }
        entry = { src: 'stated' };
        if (a !== null) entry.min = a;
        if (b !== null) entry.max = b;
      }
      const quote = note.value.trim() || `我设定：第 ${q} 季${zh(f)} ${entryText(f, entry)}`;
      entry.quote = quote.slice(0, 120);
      addNote(entry.quote);
      setEntry(q, f, entry);
      close();
    }
  }
  function close() { S.cell = null; editorBox.replaceChildren(); paintGrid(); }
  function addNote(quote) {
    if (!S.narr.includes(quote) && !S.notes.includes(quote)) S.notes.push(quote);
  }
  function setEntry(q, f, entry) {
    const r = S.qt.rows.find(x => x.q === q);
    if (!r) return;
    const slot = f.startsWith('u_') ? 'u' : 'x';
    r[slot] = r[slot] || {};
    if (entry) r[slot][f] = entry; else delete r[slot][f];
    if (slot === 'u' && !Object.keys(r.u).length) delete r.u;
    edited();
  }
  function setStance(q, v) {
    const r = S.qt.rows.find(x => x.q === q);
    if (!r) return;
    if (v === '未说明') r.fc = { stance: '未说明' };
    else {
      const quote = `我设定：第 ${q} 季金融条件${v}`;
      r.fc = { stance: v, quote };
      addNote(quote);
    }
    edited();
  }
  function setAllStances(v) {
    for (const r of S.qt?.rows || []) {
      if (v === '未说明') r.fc = { stance: '未说明' };
      else { const quote = `我设定：第 ${r.q} 季金融条件${v}`; r.fc = { stance: v, quote }; addNote(quote); }
    }
    edited();
  }
  function dropCq(kind, i) {
    const c = S.qt.c_q || {};
    if (kind === 'anchors') delete c.anchors[i];
    else if (Array.isArray(c[kind])) c[kind].splice(i, 1);
    if (kind !== 'anchors' && !c[kind]?.length) delete c[kind];
    if (kind === 'anchors' && !Object.keys(c.anchors || {}).length) delete c.anchors;
    edited();
  }
  function resize(d) {
    if (!S.qt) return;
    const H = Math.min(Math.max((S.qt.n_quarters || S.qt.rows.length) + d, 1), 60);
    if (d > 0) for (let q = S.qt.rows.length + 1; q <= H; q++) S.qt.rows.push({ q, x: {}, fc: { stance: '未说明' } });
    else S.qt.rows = S.qt.rows.filter(r => r.q <= H);
    S.qt.n_quarters = H;
    S.nq = H; nqIn.value = String(H);
    edited();
  }
  function addCol() {
    const f = addSel.value;
    if (!f) return;
    if (!S.extra.includes(f)) S.extra.push(f);
    keep();
    paintGrid();
  }
  function edited() {
    S.edited = true;
    S.run = null;
    keep();
    paintGrid();
    scheduleValidate();
  }
  function applyJson() {
    let qt;
    try { qt = JSON.parse(jsonArea.value); } catch (err) { say(jsonMsg, `JSON 语法错误：${err.message}`, 'err'); return; }
    if (!qt || typeof qt !== 'object' || !Array.isArray(qt.rows)) { say(jsonMsg, '季度情景表必须是对象，并有 rows 列表。', 'err'); return; }
    S.qt = qt;
    S.nq = qt.n_quarters || qt.rows.length; nqIn.value = String(S.nq);
    say(jsonMsg, '已应用。', 'ok');
    edited();
  }

  // ---------------------------------------------------------------- sources: translate, built-in, blank
  async function estimate() {
    const text = narr.value.trim();
    if (!text) { say(trMsg, '请先写一段情景叙事。', 'err'); return; }
    try {
      const est = await api.post('q/translate/estimate', { narrative: text, n_quarters: S.nq });
      showEstimate(est, false);
    } catch (err) { say(trMsg, `估算失败：${err.message}`, 'err'); }
  }
  function showEstimate(est, needConfirm) {
    const tok = isNum(est.input_tokens) ? `输入约 ${est.input_tokens.toLocaleString('zh-CN')} token，输出 ${(est.output_tokens || []).join('–')} token` : '';
    const sec = Array.isArray(est.seconds) ? `约 ${est.seconds.join('–')} 秒` : '';
    backendLine.textContent = `后端：${est.backend_label || est.backend || '—'}；模型 ${est.model || '—'}。`;
    if (!needConfirm) {
      say(trMsg, est.cached ? '这段叙事之前翻译过：直接用本地缓存，不调用模型。'
        : est.backend === 'mock' ? '模拟后端：不调用模型，不消耗额度。' : `需要调用 ${est.calls ?? 1} 次（格式有问题时最多再修复 1 次）；${tok}；${sec}。`, '');
      put(confirmBox);
      return;
    }
    const tick = h('input', { type: 'checkbox' });
    const go = h('button', { class: 'btn primary', type: 'button', disabled: true, onclick: () => { put(confirmBox); translate(true); } }, '确认并翻译');
    tick.addEventListener('change', () => { go.disabled = !tick.checked; });
    put(confirmBox, h('div', { class: 'ql-confirm' },
      h('div', {}, h('b', {}, '这一步会调用真实的大模型。'), ` ${est.backend_label || ''}：调用 ${est.calls ?? 1} 次（最多 ${est.max_calls ?? 2} 次）；${tok}；${sec}。`),
      h('label', {}, tick, '我确认：这会计入我的 Claude 订阅用量。'), go));
  }
  async function translate(confirm) {
    const text = narr.value.trim();
    if (!text) { say(trMsg, '请先写一段情景叙事。', 'err'); narr.focus(); return; }
    trBtn.disabled = true;
    const stop = ticking(trMsg, '正在把叙事翻译成季度情景表');
    try {
      const res = await fetch('/api/q/translate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ narrative: text, n_quarters: S.nq, ...(confirm ? { confirm: true } : {}) }) });
      const body = await res.json().catch(() => ({}));
      stop();
      if (!alive()) return;
      if (res.status === 409 && body.estimate) { say(trMsg, '需要你确认之后才会调用模型。', ''); showEstimate(body.estimate, true); return; }
      if (res.status === 503) {
        say(trMsg, `暂停：${body.detail || '模型暂时不可用'}${body.reset_at ? `（额度在 ${body.reset_at} 恢复）` : ''}`, 'err'); return;
      }
      if (!res.ok) { say(trMsg, `翻译失败：${body.detail || res.statusText}`, 'err'); return; }
      S.qt = body.qtable; S.origin = 'llm'; S.edited = false; S.notes = []; S.builtinId = null; S.run = null; S.narr = text;
      S.nq = body.qtable?.n_quarters || S.nq; nqIn.value = String(S.nq);
      keep();
      renderReport(body);
      say(trMsg, STATUS_ZH[body.status] || '完成', body.status === 'fallback' ? 'err' : 'ok');
      paintGrid();
      validate();
      ui.setRoute('lab', '', alive);
    } catch (err) {
      stop();
      say(trMsg, `翻译失败：${err.message}`, 'err');
    } finally {
      trBtn.disabled = false;
    }
  }
  function renderReport(res) {
    const u = res.usage || {};
    put(report, h('div', { style: { height: '16px' } }), card({ title: '翻译报告', sub: '大模型只给出情景表；下面的校验报告说明哪些值被拒绝、哪些部分交给模型补全。' },
      h('dl', { class: 'kv' },
        h('dt', {}, '状态'), h('dd', {}, STATUS_ZH[res.status] || res.status || '—', res.error ? `（${ERR_KIND[res.error.kind] || '出错'}：${res.error.message || ''}）` : ''),
        h('dt', {}, '后端'), h('dd', {}, res.backend === 'mock' ? '模拟后端（不调用模型，不消耗额度）' : res.backend === 'claude_code' ? '本机 Claude Code（你的订阅额度）' : res.backend || '—'),
        h('dt', {}, '模型'), h('dd', {}, res.model || '—'),
        h('dt', {}, '调用'), h('dd', {}, `${res.calls ?? 0} 次${res.cached ? '（本地缓存，没有调用模型）' : ''}${isNum(res.elapsed_s) ? `，${res.elapsed_s.toFixed(1)} 秒` : ''}`),
        h('dt', {}, 'token'), h('dd', {}, isNum(u.input_tokens) ? `输入 ${u.input_tokens.toLocaleString('zh-CN')} · 输出 ${(u.output_tokens ?? 0).toLocaleString('zh-CN')}` : '—')),
      res.raw ? h('details', { class: 'qd-more', style: { marginTop: '8px' } }, h('summary', {}, '模型的原始输出（修复和校验之前）'),
        h('pre', { class: 'code ql-wrap' }, JSON.stringify(res.raw, null, 2))) : null));
  }
  async function loadBuiltin(id) {
    if (!id) { say(trMsg, '先在下拉框里选一个内置情景。', 'err'); return; }
    try {
      const env = await api.get(`q/scenarios/${id}`);
      if (!alive()) return;
      S.qt = clone(env.qtable); S.origin = 'builtin'; S.builtinId = id; S.edited = false; S.notes = []; S.run = null;
      S.narr = env.text || ''; narr.value = S.narr;
      S.name = env.name || id; nameIn.value = S.name;
      S.nq = S.qt.n_quarters || S.qt.rows.length; nqIn.value = String(S.nq);
      keep();
      put(report, h('div', { style: { height: '16px' } }), card({ title: `内置情景「${env.name || id}」的季度情景表`, sub: '由情景库的规格确定性地换算，经过人工审阅；引文来自它的叙事原文。' },
        h('p', { style: { margin: 0, fontSize: '13px' } }, `换算记录 ${(env.conversion || []).length} 条（其中 ${(env.conversion || []).filter(c => c.status === 'dropped').length} 条没法写进情景表）；审阅记录 ${(env.review || []).length} 条。`),
        (env.review || []).length ? h('ul', { class: 'list', style: { fontSize: '13px' } }, env.review.map(r => h('li', {}, r.why || r.text || ''))) : null));
      say(trMsg, `已载入「${env.name || id}」。`, 'ok');
      paintGrid();
      validate();
      ui.setRoute('lab', `q=${encodeURIComponent(id)}`, alive);
    } catch (err) { say(trMsg, `载入失败：${err.message}`, 'err'); }
  }
  function blank() {
    S.qt = minimal(S.nq); S.origin = 'blank'; S.builtinId = null; S.edited = false; S.notes = []; S.run = null;
    keep();
    put(report);
    say(trMsg, `空白情景表：${S.nq} 个季度，全部由模型补全，直到你填写。`, 'ok');
    paintGrid();
    validate();
    ui.setRoute('lab', '', alive);
  }

  // ---------------------------------------------------------------- validation
  let vTimer = 0, vSeq = 0;
  function scheduleValidate() { clearTimeout(vTimer); vTimer = setTimeout(validate, 400); }
  async function validate() {
    if (!S.qt) return;
    const seq = ++vSeq;
    try {
      const rep = await api.post('q/validate', { qtable: S.qt, narrative: fullNarrative(), source: source(), mode: 'production' });
      if (seq !== vSeq || !alive()) return;
      paintCheck(rep);
    } catch (err) {
      if (seq !== vSeq || !alive()) return;
      put(checkBox, ui.notice(`校验失败：${err.message}`, 'error'));
    }
  }
  function paintCheck(rep) {
    const refused = rep.refused || [], errors = rep.errors || [], warnings = (rep.warnings || []).filter(w => !/^W-R\d|^W-FC|^W-OIL/.test(w.code || ''));
    const missing = rep.missing || [];
    const c = rep.counts || {};
    put(checkBox, 
      h('div', { class: 'ql-bar', style: { marginTop: 0 } },
        rep.ok ? ui.chip('可以运行', 'good') : ui.chip(errors.length ? '有错误' : '有被拒绝的条目', 'bad'),
        h('span', { class: 'muted', style: { fontSize: '12px' } }, `${rep.n_quarters ?? '—'} 个季度 · ${c.entries ?? 0} 个条目（文字数字 ${c.stated ?? 0}、文字定性 ${c.stated_qual ?? 0}、推算 ${c.derived ?? 0}）· 约束 ${c.constraints ?? 0} · 体制请求 ${c.regime ?? 0} · 估值陈述 ${c.valuation ?? 0}`)),
      errors.length ? h('div', {}, h('b', {}, '错误'), h('ul', {}, errors.map(e => h('li', { class: 'is-bad' }, e.where ? `${e.where}：${e.text}` : e.text)))) : null,
      refused.length ? h('div', {}, h('b', {}, `拒绝（${refused.length}，运行前要改掉或删掉）`), h('ul', {}, refused.map(e => h('li', {}, e.where ? `${e.where}：${e.text}` : e.text,
        e.quote ? h('span', { class: 'ql-ev' }, `引文：「${e.quote}」`) : null)))) : null,
      missing.length ? h('div', {}, h('b', {}, '由模型补全的部分（R1–R5）'), h('ul', {}, missing.map(m => h('li', {}, m.text,
        m.evidence ? h('span', { class: 'ql-ev' }, `证据句：${m.evidence}`) : null)))) : h('p', { style: { margin: '6px 0' } }, '每个季度的政策、增长、通胀、油价和金融条件都有陈述。'),
      warnings.length ? h('div', {}, h('b', {}, '其他提示'), h('ul', {}, warnings.map(w => h('li', {}, w.text)))) : null,
      (rep.clipped || []).length ? h('p', { class: 'muted', style: { fontSize: '12px' } }, `${rep.clipped.length} 个值超出历史范围的 1.5 倍，已按边界处理。`) : null);
  }

  // ---------------------------------------------------------------- run
  function paintRunNote() {
    put(runNote, S.engine === 'quarterly'
      ? ui.notice(`季度模型（${TRIAL}）：同一个联合模型，情景写了的值精确或在给定标准差内满足，没写的从联合分布补全；结果不保存，换页后需要重新运行。`, 'info')
      : ui.notice('月度引擎：情景表按季度换算成阶段规格（相同陈述的连续季度合成一个阶段，每个字段换成月度视图）；月度引擎没有的字段会被丢掉并列出。结果保存为自定义情景，可以在组合评估和提取页使用。', 'info'));
  }
  async function run() {
    if (!S.qt) { say(runMsg, '还没有情景表。', 'err'); return; }
    runBtn.disabled = true;
    const stop = ticking(runMsg, S.engine === 'quarterly' ? '季度模型运行中' : '换算并用月度引擎运行');
    try {
      if (S.engine === 'quarterly') {
        const res = await runQuarterly({ qtable: S.qt, narrative: fullNarrative(), source: source(), n_paths: S.paths, seed: 0, mode: 'path' },
          { signal: ctx.signal }, (t) => { runMsg.textContent = t; });
        stop();
        if (!alive()) return;
        S.run = res;
        say(runMsg, `完成：${res.info?.n_paths ?? S.paths} 条路径，${res.scenario_quarters ?? '—'} 个季度，用时 ${isNum(res.elapsed_s) ? res.elapsed_s.toFixed(1) : '—'} 秒。`, 'ok');
        paintGrid();
        renderQuarterlyDetail(resultBox, res, ctx, { title: `「${S.name}」：季度模型（${TRIAL}）` });
      } else {
        const conv = await api.post('q/phase_spec', { qtable: S.qt, narrative: fullNarrative(), source: source() });
        const spec = { ...conv.spec, id: `custom_q_${(conv.spec.id || '').replace(/^qtable_/, '') || Date.now().toString(36)}`, name: S.name || conv.spec.name,
          summary: '由季度情景表换算的月度规格（情景实验室）' };
        const d = await api.post('scenarios/custom/run', { spec, n_paths: Math.min(S.paths, 2000), save: true, source: S.origin === 'llm' && !S.edited ? 'llm' : 'custom' });
        stop();
        if (!alive()) return;
        api.invalidate('scenarios/custom');
        api.invalidate(`scenarios/${d.id}`);
        say(runMsg, `完成并已保存为自定义情景「${d.name}」。`, 'ok');
        resultBox.replaceChildren();
        const dropped = conv.report?.dropped || [];
        resultBox.append(card({ title: '换算成月度阶段规格', sub: '相同陈述的连续季度合成一个阶段；月度引擎没有的字段在这里列出，它们没有进入这次运行。' },
          h('ul', { class: 'list', style: { margin: 0 } }, (conv.report?.phases || []).map(p => h('li', {}, `${p.name}：${p.months} 个月，${(p.views || []).length} 个视图`))),
          dropped.length ? h('div', { style: { marginTop: '8px' } }, h('b', {}, `丢掉的陈述（${dropped.length}）`),
            h('ul', { class: 'list' }, dropped.map(x => h('li', {}, `${zh(x.field)}（第 ${span(x.quarters || [])} 季）：${x.text}`))))
            : h('p', { style: { margin: '8px 0 0' } }, '没有丢掉任何陈述。'),
          (conv.report?.notes || []).map(n => ui.notice(readable(typeof n === 'string' ? n : n.text), 'info'))),
        h('div', { style: { height: '16px' } }));
        const box = h('div');
        resultBox.append(box);
        renderScenarioDetail(box, d, ctx, { state: {}, labLink: false });
      }
      resultHead.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      stop();
      if (err?.name === 'AbortError') return;
      say(runMsg, `运行失败：${err.message}`, 'err');
    } finally {
      runBtn.disabled = false;
    }
  }

  // ---------------------------------------------------------------- start
  paintRunNote();
  const [, bl] = await Promise.all([cardP, builtinsP]);
  if (!alive()) return;
  for (const s of bl?.scenarios || []) tplSel.append(h('option', { value: s.id }, s.name || s.id));
  const opts = [...xFields, ...uFields].filter(f => !MAIN_FIELDS.includes(f)).sort((a, b) => zh(a).localeCompare(zh(b), 'zh-CN'));
  for (const f of opts) addSel.append(h('option', { value: f }, `${zh(f)}${f.startsWith('u_') ? '（干预）' : ''}`));
  api.post('q/translate/estimate', { narrative: '估算', n_quarters: 1 }).then((est) => {
    if (alive()) backendLine.textContent = `大模型后端：${est.backend_label || est.backend || '—'}${est.backend === 'claude_code' ? '：真实调用前会先显示估算并要你确认' : ''}。`;
  }).catch(() => {});
  const m = /^q=([a-z0-9_]{1,60})$/.exec(param || '');
  if (m && !ctx.rerender) await loadBuiltin(m[1]);
  else {
    paintGrid();
    if (S.qt) validate();
  }
}
