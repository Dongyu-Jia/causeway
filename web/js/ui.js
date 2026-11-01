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

export function card({ title, sub, actions, cls = '' } = {}, ...body) {
  const head = (title || actions) ? h('div', { class: 'card-head' },
    h('div', {}, title ? h('h3', { class: 'card-title' }, title) : null, sub ? h('div', { class: 'card-sub' }, sub) : null),
    actions ? h('div', { class: 'card-actions' }, actions) : null) : null;
  return h('div', { class: `card ${cls}` }, head, ...body);
}

export function chartBox(cls = '') { return h('div', { class: `chart ${cls}` }); }

export function kpi(label, value, sub, valueCls = '') {
  return h('div', { class: 'kpi' }, h('div', { class: 'kpi-label' }, label),
    h('div', { class: `kpi-value ${valueCls}` }, value), sub ? h('div', { class: 'kpi-sub' }, sub) : null);
}

// columns: [{key, label, fmt?, cls?, wrap?}], rows: array of objects
export function table(columns, rows, { onRowClick } = {}) {
  const thead = h('thead', {}, h('tr', {}, columns.map(c => h('th', { class: c.wrap ? 'wrap' : '' }, c.label))));
  const tbody = h('tbody', {}, rows.map(r => {
    const tr = h('tr', { style: onRowClick ? { cursor: 'pointer' } : null },
      columns.map(c => {
        const raw = typeof c.value === 'function' ? c.value(r) : r[c.key];
        const shown = c.fmt ? c.fmt(raw, r) : raw;
        const cls = [c.wrap ? 'wrap' : '', typeof c.cls === 'function' ? c.cls(raw, r) : (c.cls || '')].join(' ');
        return h('td', { class: cls }, shown instanceof Node ? shown : (shown ?? '—'));
      }));
    if (onRowClick) tr.addEventListener('click', () => onRowClick(r));
    return tr;
  }));
  return h('div', { class: 'table-wrap' }, h('table', { class: 'tbl' }, thead, tbody));
}

// Segmented control. options: [{value, label}]
export function seg(options, value, onChange) {
  const wrap = h('div', { class: 'seg' });
  const render = (v) => {
    wrap.innerHTML = '';
    for (const o of options) {
      wrap.append(h('button', { class: o.value === v ? 'on' : '', onclick: () => { render(o.value); onChange(o.value); } }, o.label));
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

export function loading(text = '加载中…') { return h('div', { class: 'loading' }, h('span', { class: 'spinner' }), text); }
export function notice(text, kind = '') { return h('div', { class: `notice ${kind}` }, text); }
export function chip(text, kind = '') { return h('span', { class: `chip ${kind}` }, text); }

export function fmtPct(x, d = 1, signed = false) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  const v = (x * 100).toFixed(d);
  return `${signed && x > 0 ? '+' : ''}${v}%`;
}
export function fmtPP(x, d = 1, signed = false) {   // values already in percent units
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return `${signed && x > 0 ? '+' : ''}${Number(x).toFixed(d)}%`;
}
export function fmtNum(x, d = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Number(x).toLocaleString('zh-CN', { minimumFractionDigits: d, maximumFractionDigits: d });
}
export function fmtMoney(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  const a = Math.abs(x);
  if (a >= 1e8) return `${(x / 1e8).toFixed(2)} 亿`;
  if (a >= 1e4) return `${(x / 1e4).toFixed(1)} 万`;
  return x.toFixed(0);
}
export const signCls = (x) => (x > 0 ? 'pos' : x < 0 ? 'neg' : '');

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
