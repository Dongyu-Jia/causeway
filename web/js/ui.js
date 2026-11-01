// Small DOM helpers shared by every tab. Tabs build their own layout from these.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === undefined || c === null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// Text for HTML strings (ECharts tooltips): escape anything that may come from data, a user or an LLM.
export const esc = (t) => String(t ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function card({ title, sub, actions, cls = '' } = {}, ...body) {
  const head = (title || actions) ? h('div', { class: 'card-head' },
    h('div', { class: 'card-titles' }, title ? h('h3', { class: 'card-title' }, title) : null, sub ? h('div', { class: 'card-sub' }, sub) : null),
    actions ? h('div', { class: 'card-actions' }, actions) : null) : null;
  return h('div', { class: `card ${cls}` }, head, ...body);
}

export function chartBox(cls = '') { return h('div', { class: `chart ${cls}` }); }

// A chart box that needs at least minWidth px to stay legible (a dense matrix): on a narrower card
// it scrolls sideways inside its own wrapper instead of squeezing; the page never scrolls sideways.
// Returns the wrapper; draw into `box` as usual.
export function wideChart(box, minWidth) {
  box.style.minWidth = `${minWidth}px`;
  return h('div', { class: 'chart-scroll' }, box);
}

export function kpi(label, value, sub, valueCls = '') {
  return h('div', { class: 'kpi' }, h('div', { class: 'kpi-label' }, label),
    h('div', { class: `kpi-value ${valueCls}` }, value), sub ? h('div', { class: 'kpi-sub' }, sub) : null);
}

const ALIGN = { left: 'left', right: 'right', center: 'center', l: 'left', r: 'right', c: 'center' };

// columns: [{key | value(row), label, fmt?(raw, row), cls?, wrap?, align?, minWidth?}], rows: array of objects
//   cls      string (applied to the header cell too) or function (raw, row) -> class of a body cell
//   align    'left' | 'right' | 'center' for the header and body cells
//   minWidth minimum column width (number = px, or any CSS length), set on the header cell
// options:
//   onRowClick(row)  makes rows clickable
//   rowClass(row)    class of a body row
//   dataLabels       copy each column's label onto its cells as data-label (for CSS that stacks
//                    the table into labelled blocks on narrow cards)
//   stack            dataLabels + the shared stacking style: below ~480px of card width (a phone)
//                    every row becomes a labelled block (true, or a class to add to the wrapper)
export function table(columns, rows, { onRowClick, rowClass, dataLabels = false, stack = false } = {}) {
  const labelText = (c) => (typeof c.label === 'string' ? c.label : c.label instanceof Node ? c.label.textContent : String(c.label ?? ''));
  const alignOf = (c) => ALIGN[c.align] || null;
  const thead = h('thead', {}, h('tr', {}, columns.map(c => {
    const cls = [c.wrap ? 'wrap' : '', typeof c.cls === 'string' ? c.cls : ''].join(' ').trim();
    const style = {};
    if (alignOf(c)) style.textAlign = alignOf(c);
    if (c.minWidth !== undefined && c.minWidth !== null) style.minWidth = typeof c.minWidth === 'number' ? `${c.minWidth}px` : String(c.minWidth);
    return h('th', { class: cls || null, style: Object.keys(style).length ? style : null }, c.label);
  })));
  const labels = dataLabels || stack ? columns.map(labelText) : null;
  const tbody = h('tbody', {}, (rows || []).map(r => {
    const rc = rowClass ? rowClass(r) : '';
    const tr = h('tr', { class: rc || null, style: onRowClick ? { cursor: 'pointer' } : null },
      columns.map((c, i) => {
        const raw = typeof c.value === 'function' ? c.value(r) : r[c.key];
        const shown = c.fmt ? c.fmt(raw, r) : raw;
        const cls = [c.wrap ? 'wrap' : '', typeof c.cls === 'function' ? c.cls(raw, r) : (c.cls || '')].join(' ').trim();
        const td = h('td', { class: cls || null, style: alignOf(c) ? { textAlign: alignOf(c) } : null },
          shown instanceof Node ? shown : (shown ?? '—'));
        if (labels) td.dataset.label = labels[i];
        return td;
      }));
    if (onRowClick) tr.addEventListener('click', () => onRowClick(r));
    return tr;
  }));
  const tbl = h('table', { class: `tbl${stack ? ' tbl-stack' : ''}` }, thead, tbody);
  if (!stack) return h('div', { class: 'table-wrap' }, tbl);
  return h('div', { class: `tbl-stackbox${typeof stack === 'string' ? ` ${stack}` : ''}` }, h('div', { class: 'table-wrap' }, tbl));
}

// Segmented control. options: [{value, label}]
export function seg(options, value, onChange) {
  const wrap = h('div', { class: 'seg' });
  const render = (v) => {
    wrap.innerHTML = '';
    for (const o of options) {
      wrap.append(h('button', { type: 'button', class: o.value === v ? 'on' : '', onclick: () => { render(o.value); onChange(o.value); } }, o.label));
    }
  };
  render(value);
  return wrap;
}

export function select(options, value, onChange) {
  const el = h('select', { onchange: (e) => onChange(e.target.value) },
    options.map(o => h('option', { value: o.value, selected: o.value === value ? true : null }, o.label)));
  return el;
}

// Rewrite the "?param" part of the address for the page `tab` without navigating, but only while
// that page is still the one in the address bar and still alive: a slow render or a late result
// (an engine run finishing after the reader left) must not drag the URL, and the next hashchange,
// back to a page the reader has left. Returns whether the address was changed.
export function setRoute(tab, param, alive = () => true) {
  const cur = location.hash.slice(1).split('?')[0] || 'overview';
  if (cur !== tab || !alive()) return false;
  try { history.replaceState(null, '', `#${tab}${param ? `?${param}` : ''}`); } catch (_) { return false; }   // file:// or sandboxed
  return true;
}

// What to run when a result file is missing (a 404): the server's message names the command that
// writes that file (python -m scenariolab.build --only <step>, or ... --only ablation for the opt-in study).
export function buildHint(err) {
  const cmds = String(err?.message || '').match(/python -m [a-z_.]+(?: --only [a-z_]+)?/g) || [];
  const only = cmds.find(c => c.includes('--only')) || cmds[0] || 'python -m scenariolab.build';
  const full = only.startsWith('python -m scenariolab.build --only') && !only.endsWith('ablation');
  return h('p', { class: 'lead', style: { marginTop: '12px' } }, '在项目根目录运行 ', h('code', { class: 'mono' }, only),
    full ? ['（缺的不止这一个文件时，运行完整的 ', h('code', { class: 'mono' }, 'python -m scenariolab.build'), '）'] : null,
    '，生成结果后刷新本页。');
}

export function loading(text = '加载中…') { return h('div', { class: 'loading' }, h('span', { class: 'spinner' }), text); }
export function notice(text, kind = '') { return h('div', { class: `notice ${kind}` }, text); }
export function chip(text, kind = '') { return h('span', { class: `chip ${kind}` }, text); }

// A finite number from a number or a numeric string; null otherwise.
function num(x) {
  if (x === null || x === undefined || x === '' || typeof x === 'boolean') return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}
// Fixed decimals, rounded first, so a value that rounds to zero prints "0.0", never "-0.0".
function fixedText(n, d) { const s = n.toFixed(d); return Number(s) === 0 ? (0).toFixed(d) : s; }

export function fmtPct(x, d = 1, signed = false) {     // decimals: 0.052 -> "5.2%"
  const n = num(x);
  if (n === null) return '—';
  const s = fixedText(n * 100, d);
  return `${signed && Number(s) > 0 ? '+' : ''}${s}%`;
}
export function fmtPP(x, d = 1, signed = false) {   // values already in percent units
  const n = num(x);
  if (n === null) return '—';
  const s = fixedText(n, d);
  return `${signed && Number(s) > 0 ? '+' : ''}${s}%`;
}
export function fmtNum(x, d = 2) {
  const n = num(x);
  if (n === null) return '—';
  const r = Number(fixedText(n, d));
  return r.toLocaleString('zh-CN', { minimumFractionDigits: d, maximumFractionDigits: d });
}
export function fmtMoney(x) {
  const n = num(x);
  if (n === null) return '—';
  const a = Math.abs(n);
  if (a >= 1e8) return `${fixedText(n / 1e8, 2)} 亿`;
  if (a >= 1e4) return `${fixedText(n / 1e4, 1)} 万`;
  return fixedText(n, 0);
}

// Colour class of a signed number: 'pos' (up colour) / 'neg' (down colour) follow the K-line
// setting (default 红涨绿跌); a value that rounds to 0 at `d` decimals of a percent (the way
// fmtPct shows decimals) stays neutral. Safe to pass straight as a table column's cls (the second
// argument is then the row and is ignored). For good / bad judgements use goodCls.
export function signCls(x, d = 1) {
  const n = num(x);
  if (n === null) return '';
  const digits = typeof d === 'number' && d >= 0 ? d : 1;
  if (Math.abs(n) < 0.5 * 10 ** -(digits + 2)) return '';
  return n > 0 ? 'pos' : 'neg';
}
// Judgement colour, independent of the K-line setting: true -> green 'is-good', false -> red 'is-bad'.
export const goodCls = (ok) => (ok === true ? 'is-good' : ok === false ? 'is-bad' : '');

// Lookups filled by app.js from /api/meta
export const META = { assets: {}, factors: {}, states: {}, eras: {}, raw: null };
export function setMeta(meta) {
  META.raw = meta;
  for (const a of meta.assets || []) META.assets[a.id] = a;
  for (const f of meta.factors || []) META.factors[f.id] = f;
  for (const s of meta.states || []) META.states[s.id] = s;
  for (const e of meta.eras || []) META.eras[e.id] = e;
}
export const assetName = (id) => META.assets[id]?.name || id;
export const factorName = (id) => META.factors[id]?.name || id;
export const stateName = (id) => META.states[id]?.name || id;
export const eraName = (id) => META.eras[id]?.name || id;
export function varName(col) {
  if (!col) return '';
  const [pre, ...rest] = col.split('_');
  const id = rest.join('_');
  if (pre === 's') return stateName(id);
  if (pre === 'f') return id === 'cpi' ? 'CPI 月度变化' : factorName(id);
  if (pre === 'e') return { oil_supply: '石油供给冲击', demand: '总需求冲击', monetary: '货币政策冲击', financial: '金融/风险冲击' }[id] || col;
  return col;
}
