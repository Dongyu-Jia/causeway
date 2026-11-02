// 变量卡片 (SIM_DESIGN §3.3, §10.2, Appendix A): every variable of the quarterly model, one row each, with the model
// card's columns (中文名, 同期/前定, 时点, 内生性, 到资产的路径, 翻译器, 单位, 起始, 发布滞后, 注), today's value, its
// percentile and a sparkline from the quarterly panel, and the links to each row's evidence. Searchable; filters by
// 槽位, 同期/前定, 时点, 内生性 and 到资产的路径. Data: GET /api/q/card/view (scenariolab/qmodel/cardview.py), whose texts
// are already readable (card ids replaced by Chinese names).
//
// Deep link: #card?<field> opens that row (the id lives only in the address and in the search, never on the page).
// The same component gives the lab's QTable headers their tooltips: fieldLabel(id, text) / attachCardTip(el, id).
// Every text is set with h() / textContent; sparklines are SVG elements built node by node.
import * as api from '../api.js';
import * as ui from '../ui.js';

const { h } = ui;
const SVG = 'http://www.w3.org/2000/svg';
const ID_RE = /^[a-z][a-z0-9_]{1,40}$/;
const STATE = { q: '', group: '', lead: '', timing: '', endo: '', route: '', open: new Set(), all: false };
const isNum = (x) => typeof x === 'number' && Number.isFinite(x);

const CSS = `
.cd-hero { border: 1px solid #cdd7ff; border-left: 4px solid var(--accent); background: linear-gradient(135deg, #f3f6ff, #fff 70%);
  border-radius: var(--radius); padding: 14px 16px; margin-bottom: 16px; }
.cd-hero h2 { margin: 0 0 6px; font-size: 16px; font-weight: 650; line-height: 1.55; color: #1f2f7a; }
.cd-hero p { margin: 0; color: var(--ink-2); font-size: 13px; }
.cd-kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(104px, 100%), 1fr)); gap: 8px; margin: 0 0 16px; }
.cd-kpis .kpi { padding: 8px 10px; cursor: pointer; }
.cd-kpis .kpi-label { font-size: 11px; }
.cd-kpis .kpi.on { border-color: var(--accent); background: var(--accent-soft); }
.cd-kpis .kpi-value { font-size: 18px; }
.cd-bar { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: flex-end; }
.cd-bar label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; color: var(--ink-3); min-width: 0; }
.cd-bar select, .cd-bar input[type=text] { max-width: 100%; min-width: 0; }
.cd-bar input[type=text] { width: 260px; }
.cd-count { font-size: 12px; color: var(--ink-3); margin-left: auto; }
.cd-legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(260px, 100%), 1fr)); gap: 10px 18px; margin-top: 10px; }
.cd-legend dl { margin: 0; display: grid; grid-template-columns: max-content 1fr; gap: 3px 10px; font-size: 12px; }
.cd-legend dt { color: var(--ink); font-weight: 600; } .cd-legend dd { margin: 0; color: var(--ink-3); }
.cd-legend h4 { margin: 0 0 6px; font-size: 12px; color: var(--ink-2); }
details.cd-more > summary { cursor: pointer; color: var(--accent); font-size: 13px; }
.cd-t table.tbl td { vertical-align: top; }
.cd-t table.tbl th, .cd-t table.tbl td { padding: 6px 8px; }
.cd-t table.tbl th:nth-child(-n+4), .cd-t table.tbl td:nth-child(-n+4) { text-align: left; }
.cd-t table.tbl tr.cd-row { cursor: pointer; scroll-margin-top: 80px; }
.cd-t table.tbl tr.cd-row.on td { background: #f5f7ff; }
.cd-t table.tbl tr.cd-row:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.cd-name { font-weight: 600; color: var(--ink); white-space: normal; }
.cd-name small { display: block; font-weight: 400; color: var(--ink-3); font-size: 11px; }
.cd-name .cd-note { color: #7a4d00; }
.cd-lag { display: block; font-size: 11px; color: var(--ink-3); margin-top: 2px; }
.cd-chips { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 3px; }
.cd-chips .chip { font-size: 11px; padding: 0 6px; }
.cd-today { font-variant-numeric: tabular-nums; white-space: nowrap; }
.cd-today small { display: block; color: var(--ink-3); font-size: 11px; }
.cd-pct { display: inline-flex; align-items: center; gap: 6px; font-variant-numeric: tabular-nums; }
.cd-pct .bar-inline { width: 42px; display: inline-block; }
.cd-spark { display: block; }
.cd-detail td { background: #fbfcff; white-space: normal !important; text-align: left !important; font-weight: 400 !important; }
.cd-detail .cd-dgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(260px, 100%), 1fr)); gap: 12px 20px; padding: 4px 2px 8px;
  contain: inline-size; width: 0; min-width: 100%; }
.cd-detail svg.cd-spark { max-width: 100%; height: auto; }
.cd-endo { white-space: nowrap; }
.cd-endo small { display: block; color: var(--ink-3); font-size: 11px; }
.cd-detail h5 { margin: 0 0 4px; font-size: 12px; color: var(--ink-3); font-weight: 600; }
.cd-detail p { margin: 0 0 8px; font-size: 13px; line-height: 1.6; color: var(--ink-2); }
.cd-detail dl { margin: 0; display: grid; grid-template-columns: max-content 1fr; gap: 3px 10px; font-size: 12px; }
.cd-detail dt { color: var(--ink-3); } .cd-detail dd { margin: 0; color: var(--ink); }
.cd-ev { margin: 0; padding-left: 18px; font-size: 12px; }
.cd-ev li { margin: 4px 0; color: var(--ink-2); }
.cd-group td { background: var(--panel-2); font-weight: 600; color: var(--ink-2); font-size: 12px; }
.cd-empty { color: var(--ink-3); text-align: center; padding: 24px 8px; }
.ct-anchor { cursor: help; border-bottom: 1px dotted #9aa5c0; }
.ct-anchor:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.ct-pop { position: fixed; z-index: 50; background: #fff; border: 1px solid #cdd7ff; border-radius: 10px; box-shadow: 0 6px 24px rgba(16, 24, 40, .16);
  padding: 10px 12px; font-size: 12px; line-height: 1.55; color: var(--ink-2); max-height: min(420px, 70vh); overflow: auto; }
.ct-pop b { color: var(--ink); font-size: 13px; }
.ct-pop .ct-k { color: var(--ink-3); }
.ct-pop .ct-row { margin-top: 4px; }
.ct-pop a { font-size: 12px; }
`;

function ensureStyle() {
  let el = document.getElementById('style-card');
  if (!el) { el = document.createElement('style'); el.id = 'style-card'; document.head.append(el); }
  if (!el.textContent.includes('/* card */')) el.textContent += `/* card */\n${CSS}`;
}

const put = (el, ...kids) => { el.replaceChildren(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false)); return el; };
const pctText = (p) => (isNum(p) ? `${Math.round(p * 100)}%` : '—');
// a long text cut for a table cell (the full text is its tooltip and sits in the expanded row)
const clip = (t, n) => { const a = [...String(t || '')]; return a.length > n ? `${a.slice(0, n).join('')}…` : a.join(''); };

// ---------------------------------------------------------------- the sparkline (SVG nodes, no markup strings)

function svg(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  for (const c of kids) if (c) el.append(c);
  return el;
}

// values: quarterly numbers (null = gap); the last finite value gets a dot. A zero line when the range crosses 0.
export function sparkline(values, { width = 96, height = 26, label = '', big = false } = {}) {
  const vals = (values || []).map(v => (isNum(v) ? v : null));
  const fin = vals.filter(isNum);
  if (fin.length < 2) return h('span', { class: 'muted' }, '—');
  let lo = Math.min(...fin), hi = Math.max(...fin);
  if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
  const pad = big ? 6 : 2;
  const X = (i) => pad + (i / Math.max(vals.length - 1, 1)) * (width - 2 * pad);
  const Y = (v) => pad + (1 - (v - lo) / (hi - lo)) * (height - 2 * pad);
  let d = '';
  let pen = false;
  vals.forEach((v, i) => {
    if (!isNum(v)) { pen = false; return; }
    d += `${pen ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)} `;
    pen = true;
  });
  let last = vals.length - 1;
  while (last >= 0 && !isNum(vals[last])) last -= 1;
  const kids = [];
  if (lo < 0 && hi > 0) kids.push(svg('line', { x1: pad, x2: width - pad, y1: Y(0).toFixed(1), y2: Y(0).toFixed(1), stroke: '#ced4da', 'stroke-width': 1, 'stroke-dasharray': '2 2' }));
  kids.push(svg('path', { d: d.trim(), fill: 'none', stroke: '#3b5bdb', 'stroke-width': big ? 1.4 : 1.1, 'stroke-linejoin': 'round' }));
  if (last >= 0) kids.push(svg('circle', { cx: X(last).toFixed(1), cy: Y(vals[last]).toFixed(1), r: big ? 3 : 2, fill: '#1d2433' }));
  const el = svg('svg', { class: 'cd-spark', width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': label || '历史走势' },
    svg('title', {}, document.createTextNode(label || '历史走势')), ...kids);
  return el;
}

// ---------------------------------------------------------------- the tooltip component (also used by the lab)

let VIEW = null;
function viewRows() {
  if (!VIEW) VIEW = api.get('q/card/view').then(v => Object.fromEntries((v.rows || []).map(r => [r.id, r]))).catch(() => ({}));
  return VIEW;
}
let POP = null, POP_FOR = null, HIDE = 0;
function popEl() {
  const host = document.getElementById('view') || document.body;
  if (!POP || !POP.isConnected || POP.parentElement !== host) {
    POP = h('div', { class: 'ct-pop', role: 'tooltip', id: 'ct-pop', hidden: true });
    POP.addEventListener('mouseenter', () => clearTimeout(HIDE));
    POP.addEventListener('mouseleave', () => hideTip());
    host.append(POP);
  }
  return POP;
}
function place(anchor) {
  const pop = popEl();
  const r = anchor.getBoundingClientRect();
  const vw = document.documentElement.clientWidth || window.innerWidth;
  const w = Math.min(340, vw - 16);
  pop.style.width = `${w}px`;
  pop.style.left = `${Math.max(8, Math.min(r.left, vw - w - 8))}px`;
  const below = window.innerHeight - r.bottom;
  pop.style.top = '0px';
  const ph = Math.min(pop.offsetHeight || 200, window.innerHeight * 0.7);
  pop.style.top = `${below >= ph + 12 || r.top < ph + 12 ? Math.min(r.bottom + 6, window.innerHeight - ph - 8) : r.top - ph - 6}px`;
}
function tipBody(row) {
  const r = row || {};
  const t = r.today;
  return [
    h('b', {}, r.zh || '（未命名字段）'),
    r.definition ? h('div', { class: 'ct-row' }, r.definition) : null,
    h('div', { class: 'ct-row' }, [r.lead_kind || r.lead, r.timing, r.endo].filter(Boolean).join(' · ')),
    r.unit ? h('div', { class: 'ct-row' }, h('span', { class: 'ct-k' }, '单位 '), r.unit, r.lag ? [h('span', { class: 'ct-k' }, '　发布滞后 '), r.lag] : null) : null,
    r.translator ? h('div', { class: 'ct-row' }, h('span', { class: 'ct-k' }, '翻译器 '), r.translator) : null,
    r.route ? h('div', { class: 'ct-row' }, h('span', { class: 'ct-k' }, '到资产的路径 '), r.route) : null,
    t && t.text ? h('div', { class: 'ct-row' }, h('span', { class: 'ct-k' }, `今天（${t.when || ''}）`), ` ${t.text}`, isNum(t.pct) ? `，历史分位 ${pctText(t.pct)}` : '') : null,
    r.note ? h('div', { class: 'ct-row ct-k' }, r.note) : null,
    r.id ? h('div', { class: 'ct-row' }, h('a', { href: `#card?${encodeURIComponent(r.id)}` }, '在变量卡片里查看 →')) : null,
  ];
}
function showTip(anchor, id, fallback) {
  clearTimeout(HIDE);
  const pop = popEl();
  POP_FOR = anchor;
  anchor.setAttribute('aria-describedby', 'ct-pop');
  put(pop, h('b', {}, fallback?.zh || '（未命名字段）'), h('div', { class: 'ct-row ct-k' }, '读取变量卡片…'));   // the readable row follows
  pop.hidden = false;
  place(anchor);
  viewRows().then((rows) => {
    if (POP_FOR !== anchor || pop.hidden) return;
    put(pop, rows[id] ? tipBody(rows[id]) : [h('b', {}, fallback?.zh || '（未命名字段）'), h('div', { class: 'ct-row' }, h('a', { href: `#card?${encodeURIComponent(id)}` }, '在变量卡片里查看 →'))]);
    place(anchor);
  });
}
export function hideTip() {
  HIDE = setTimeout(() => { if (POP) POP.hidden = true; POP_FOR = null; }, 120);
}

// Attach the card tooltip to an element: hover or focus shows it, a tap toggles it (phones), Escape hides it.
// fallback: the row the caller already has (/api/q/card): its name shows until the readable view (/api/q/card/view)
// arrives, so the raw card texts (which name other fields by id) never reach the reader.
export function attachCardTip(el, id, fallback = null) {
  if (!el || !ID_RE.test(String(id || ''))) return el;
  ensureStyle();
  el.classList.add('ct-anchor');
  if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '0');
  el.addEventListener('mouseenter', () => showTip(el, id, fallback));
  el.addEventListener('mouseleave', () => hideTip());
  el.addEventListener('focus', () => showTip(el, id, fallback));
  el.addEventListener('blur', () => hideTip());
  el.addEventListener('click', (e) => {
    e.stopPropagation();
    if (POP && !POP.hidden && POP_FOR === el) { POP.hidden = true; POP_FOR = null; } else showTip(el, id, fallback);
  });
  el.addEventListener('keydown', (e) => { if (e.key === 'Escape' && POP) { POP.hidden = true; POP_FOR = null; } });
  return el;
}

// A field's name with the card tooltip (the lab's QTable column headers).
export function fieldLabel(id, text, fallback = null) {
  return attachCardTip(h('span', {}, text || fallback?.zh || '（未命名字段）'), id, fallback);
}

// ---------------------------------------------------------------- the tab

const LEAD_CHIP = { 同期: 'accent', 前定: '', 假设: 'warn' };
const ROUTE_CHIP = { L0: 'good', candidate: 'warn', factor: '', shock: '', state: '', identity: '', anchor: '', output: '', none: '' };

function detailRow(r, ncol) {
  const s = r.series;
  const t = r.today;
  const kv = [
    ['槽位', r.slot], ['同期 / 前定', r.lead], ['时点', r.timing], ['内生性', r.endo], ['单位', r.unit], ['起始', r.first],
    ['发布滞后', r.lag], ['到资产的路径', r.route], ['翻译器', r.translator], ['核心因子层', r.factor_core ? '是' : '否'],
    ['金融条件', r.fc || '—'], ['家族', r.family],
  ].filter(([, v]) => v !== null && v !== undefined && v !== '');
  const td = h('td', { colspan: String(ncol) },
    h('div', { class: 'cd-dgrid' },
      h('div', {},
        h('h5', {}, '定义'), h('p', {}, r.definition || '—'),
        r.note ? [h('h5', {}, '注'), h('p', {}, r.note)] : null,
        r.keywords?.length && r.permission === 'stated_only' ? [h('h5', {}, '引文必须点名这个变量'),
          h('p', {}, `翻译器写它的时候，引文里要有这些词之一：${r.keywords.join(' / ')}`)] : null,
        r.enum?.length ? [h('h5', {}, '取值'), h('p', {}, r.enum.join(' / '))] : null),
      h('div', {}, h('h5', {}, '卡片'), h('dl', {}, kv.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]))),
      h('div', {},
        h('h5', {}, '今天与历史'),
        t ? h('p', {}, `${t.when || ''}：${t.text}`, isNum(t.pct) ? `；在它自己 ${s?.n || ''} 个季度的历史里分位 ${pctText(t.pct)}（${s?.pct_since || ''} 起）` : '', s?.label ? `。显示的是：${s.label}` : '') : h('p', {}, '没有数据序列（政策冲击是反应函数的残差，锚是情景的长期假设）。'),
        s ? sparkline(s.values, { width: 300, height: 72, big: true, label: `${s.first}–${s.last_q} 的季度走势` }) : null,
        s ? h('div', { class: 'muted', style: { fontSize: '11px', display: 'flex', justifyContent: 'space-between', maxWidth: '300px' } }, h('span', {}, s.first), h('span', {}, s.last_q)) : null),
      h('div', {},
        h('h5', {}, '证据'),
        r.evidence?.length ? h('ul', { class: 'cd-ev' }, r.evidence.map(e => h('li', {}, e.text, ' ', e.href ? h('a', { href: e.href }, '查看 →') : null)))
          : h('p', { class: 'muted' }, '这个变量没有单独的检验记录：它的作用只经由联合模型。'))));
  return h('tr', { class: 'cd-detail' }, td);
}

export async function render(root, ctx) {
  ensureStyle();
  const { param } = ctx;
  const alive = ctx.alive || (() => root.isConnected);
  if (!ctx.rerender) { Object.assign(STATE, { q: '', group: '', lead: '', timing: '', endo: '', route: '', all: false }); STATE.open = new Set(); }
  const want = param && ID_RE.test(param) ? param : null;
  if (want) STATE.open.add(want);

  root.append(h('div', { class: 'cd-hero' },
    h('h2', {}, '变量卡片：季度模型的每一个变量，一行一个。'),
    h('p', {}, '大模型写的季度情景表、模型的输入和这一页用的是同一张卡片。“同期”的输入与资产回报在同一个季度实现，模型不说它们领先资产；'
      + '“前定”的状态在季初已经知道；“假设”是情景给的长期锚。内生性决定翻译器能不能写它：市场价格和调查值只在文字明说时才写。')));
  const box = h('div', {}, ui.loading('读取变量卡片…'));
  root.append(box);
  let data;
  try {
    data = await api.get('q/card/view');
  } catch (err) {
    if (!alive()) return null;
    put(box, ui.notice(`变量卡片读不了：${err.message}`, 'error'), err.status === 404 ? ui.buildHint(err) : null);
    return null;
  }
  if (!alive()) return null;
  const rows = data.rows || [];
  const byId = Object.fromEntries(rows.map(r => [r.id, r]));
  const F = data.filters || {};

  // ---- KPIs: one tile per group (a click filters the table to it)
  const kpis = h('div', { class: 'cd-kpis' });
  const paintKpis = () => put(kpis, h('div', { class: `kpi${STATE.group ? '' : ' on'}`, role: 'button', tabindex: '0',
    onclick: () => { STATE.group = ''; sync(); } }, h('div', { class: 'kpi-label' }, '全部变量'), h('div', { class: 'kpi-value' }, String(rows.length))),
  (data.groups || []).map(g => h('div', { class: `kpi${STATE.group === g.id ? ' on' : ''}`, role: 'button', tabindex: '0',
    onclick: () => { STATE.group = STATE.group === g.id ? '' : g.id; sync(); },
    onkeydown: (e) => { if (e.key === 'Enter') { STATE.group = STATE.group === g.id ? '' : g.id; sync(); } } },
  h('div', { class: 'kpi-label' }, g.zh), h('div', { class: 'kpi-value' }, String(g.n)))));

  // ---- controls
  const opt = (list, all) => [{ value: '', label: all }, ...(list || []).map(o => ({ value: o.value, label: `${o.label}（${o.n}）` }))];
  const search = h('input', { type: 'text', placeholder: '搜索中文名、定义、注或关键词', 'aria-label': '搜索变量', value: STATE.q });
  let tmr = 0;
  search.addEventListener('input', () => { clearTimeout(tmr); tmr = setTimeout(() => { STATE.q = search.value.trim(); paint(); }, 120); });
  const sels = {};
  const mk = (key, label, list, all) => {
    const s = ui.select(opt(list, all), STATE[key], (v) => { STATE[key] = v; sync(); });
    s.setAttribute('aria-label', label);
    sels[key] = s;
    return h('label', {}, label, s);
  };
  const count = h('span', { class: 'cd-count' });
  const allBox = h('input', { type: 'checkbox', checked: STATE.all ? true : null, onchange: (e) => { STATE.all = e.target.checked; paint(); } });
  const reset = h('button', { class: 'btn', type: 'button', onclick: () => {
    Object.assign(STATE, { q: '', group: '', lead: '', timing: '', endo: '', route: '' }); search.value = ''; sync();
  } }, '清除筛选');
  const ctrl = ui.card({ title: '筛选', sub: data.today ? `数据截至 ${data.today.data_end_month || '—'}；今天 = ${data.today.quarter} 季初。点一行展开它的定义、注、今天的值与证据。` : '季度面板还没有生成：今天的值、分位与走势暂缺（python -m scenariolab.qmodel.panel）。' },
    h('div', { class: 'cd-bar' },
      h('label', {}, '搜索', search),
      mk('group', '槽位', F.group, '全部'), mk('lead', '同期 / 前定', F.lead, '全部'), mk('timing', '时点', F.timing, '全部'),
      mk('endo', '内生性', F.endo, '全部'), mk('route', '到资产的路径', F.route, '全部'),
      h('label', { style: { flexDirection: 'row', alignItems: 'center', gap: '6px' } }, allBox, '展开全部说明'),
      reset, count),
    h('details', { class: 'cd-more', style: { marginTop: '10px' } }, h('summary', {}, '怎么读这些列'),
      h('div', { class: 'cd-legend' },
        h('div', {}, h('h4', {}, '同期 / 前定 / 假设'), h('dl', {}, Object.entries(data.legend?.lead || {}).flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]))),
        h('div', {}, h('h4', {}, '内生性'), h('dl', {}, Object.entries(data.legend?.endo || {}).flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]))),
        h('div', {}, h('h4', {}, '到资产的路径'), h('dl', {}, (F.route || []).flatMap(o => [h('dt', {}, o.label), h('dd', {}, data.legend?.route?.[o.value] || '')])))),
      h('ul', { class: 'list', style: { marginTop: '10px', fontSize: '12px', color: 'var(--ink-3)' } }, (data.notes || []).map(n => h('li', {}, n)))));

  // ---- the table
  const tableBox = h('div', { class: 'cd-t' });
  const COLS = ['变量', '时点', '内生性 · 翻译器', '到资产的路径', '今天', '历史分位', '走势'];
  const hay = (r) => [r.zh, r.definition, r.note, r.family, r.slot, r.timing, r.endo, r.route, r.translator, r.id, ...(r.keywords || [])].join(' ').toLowerCase();
  const HAY = Object.fromEntries(rows.map(r => [r.id, hay(r)]));
  const match = (r) => {
    if (STATE.group && r.group !== STATE.group) return false;
    if (STATE.lead && r.lead_kind !== STATE.lead) return false;
    if (STATE.timing && r.timing !== STATE.timing) return false;
    if (STATE.endo && r.endo_code !== STATE.endo) return false;
    if (STATE.route && r.route_kind !== STATE.route) return false;
    if (STATE.q) return STATE.q.toLowerCase().split(/\s+/).every(w => HAY[r.id].includes(w));
    return true;
  };
  const rowEls = {};
  function rowNode(r) {
    const t = r.today;
    const chips = [
      r.factor_core ? ui.chip('核心因子层', 'accent') : null,
      r.fc_role === 'required_one' ? ui.chip('金融条件：每季必填其一', 'warn') : r.fc_role === 'optional' ? ui.chip('金融条件（可选）') : null,
    ].filter(Boolean);
    const cells = [
      h('div', { class: 'cd-name' }, r.zh, h('small', {}, r.unit ? `单位 ${r.unit}` : ''),
        r.note ? h('small', { class: 'cd-note', title: r.note }, `注：${clip(r.note, 34)}`) : null,
        chips.length ? h('div', { class: 'cd-chips' }, chips) : null),
      h('div', {}, ui.chip(r.lead_kind || '—', LEAD_CHIP[r.lead_kind] || ''), h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '2px' } }, r.timing),
        r.lag && r.lag !== '—' ? h('span', { class: 'cd-lag', title: `发布滞后：${r.lag}` }, `发布滞后 ${clip(r.lag, 14)}`) : null),
      h('div', { class: 'cd-endo' }, r.endo_short || r.endo, h('small', {}, r.translator_short)),
      ui.chip(r.route_short, ROUTE_CHIP[r.route_kind] || ''),
      t ? h('div', { class: 'cd-today' }, t.text, h('small', {}, t.when || '')) : h('span', { class: 'muted' }, '—'),
      t && isNum(t.pct) ? h('span', { class: 'cd-pct', title: `这个序列自己的历史里，不高于今天的季度占 ${pctText(t.pct)}` }, pctText(t.pct)) : h('span', { class: 'muted' }, '—'),
      r.series ? sparkline(r.series.values, { width: 84, label: `${r.zh}：${r.series.first}–${r.series.last_q} 的季度走势` }) : h('span', { class: 'muted' }, '—'),
    ];
    const tr = h('tr', { class: 'cd-row', tabindex: '0', 'aria-expanded': 'false', 'data-id': r.id },
      cells.map((c, i) => h('td', { 'data-label': COLS[i], class: i <= 1 ? 'wrap' : null }, c)));
    const toggle = () => {
      if (STATE.open.has(r.id)) STATE.open.delete(r.id); else STATE.open.add(r.id);
      paint();
    };
    tr.addEventListener('click', (e) => { if (e.target.closest('a')) return; toggle(); });
    tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    return tr;
  }
  function paint() {
    const shown = rows.filter(match);
    count.textContent = `显示 ${shown.length} / ${rows.length} 个变量`;
    const body = [];
    let lastGroup = null;
    for (const r of shown) {
      if (r.group !== lastGroup && !STATE.group) {
        body.push(h('tr', { class: 'cd-group' }, h('td', { colspan: String(COLS.length) }, (data.groups || []).find(g => g.id === r.group)?.zh || r.group_zh)));
        lastGroup = r.group;
      }
      const tr = rowNode(r);
      rowEls[r.id] = tr;
      const open = STATE.all || STATE.open.has(r.id);
      if (open) { tr.classList.add('on'); tr.setAttribute('aria-expanded', 'true'); }
      body.push(tr);
      if (open) body.push(detailRow(r, COLS.length));
    }
    const tbl = h('table', { class: 'tbl tbl-stack' },
      h('thead', {}, h('tr', {}, COLS.map((c, i) => h('th', { class: i <= 1 ? 'wrap' : null }, c)))),
      h('tbody', {}, body.length ? body : h('tr', {}, h('td', { colspan: String(COLS.length), class: 'cd-empty' }, '没有符合条件的变量：换个关键词或清除筛选。'))));
    put(tableBox, h('div', { class: 'tbl-stackbox' }, h('div', { class: 'table-wrap' }, tbl)));
  }
  function sync() {
    for (const [k, s] of Object.entries(sels)) s.value = STATE[k];
    paintKpis();
    paint();
  }
  put(box, kpis, ctrl, h('div', { style: { height: '16px' } }), ui.card({ title: '模型卡片', sub: `${rows.length} 个变量。季度模型通过切换关之前只作${data.label || '试运行'}。` }, tableBox));
  sync();
  if (want && byId[want]) {
    if (!match(byId[want])) { Object.assign(STATE, { q: '', group: '', lead: '', timing: '', endo: '', route: '' }); search.value = ''; sync(); }
    return rowEls[want] || null;
  }
  return null;
}
