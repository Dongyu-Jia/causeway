// App shell: hash router + lazy-loaded tabs. Each tab module exports
//   async function render(root, ctx)
// root  a fresh container for this visit inside #view; it is detached when the user navigates
//       away, so anything a slow render appends afterwards never shows up on the next tab
// ctx   {api, charts, ui, meta, param, route, signal, alive, rerender}
//   param    the part of the route after "?" (a deep link), decoded, or null
//   signal   AbortSignal, aborted on the next navigation (or re-render): stop timers and observers
//   alive()  false once the user has moved on: drop stale responses
//   rerender true when the same page is drawn again in place (the K-line toggle): keep what the
//            reader was doing (drafts, picks) instead of re-applying the deep link
// The page starts at the top; render() may return an Element (a deep-linked section) to scroll to.
import * as api from './api.js';
import * as charts from './charts.js';
import * as ui from './ui.js';

const TABS = [
  { group: '历史与因果' },
  { id: 'overview', name: '总览' },
  { id: 'eras', name: '历史时代画像' },
  { id: 'regimes', name: '宏观体制' },
  { id: 'causality', name: '因果传导' },
  { group: '验证' },
  { id: 'forecast', name: '一年期预测' },
  { id: 'backtest', name: '回测：模拟 vs 实际' },
  { id: 'diagnostics', name: '诊断与迭代' },
  { id: 'valuation', name: '估值与盈利' },
  { group: '未来' },
  { id: 'scenarios', name: '未来情景' },
  { id: 'lab', name: '情景实验室（LLM）' },
  { id: 'instruments', name: '标的推演' },
  { id: 'portfolio', name: '组合评估' },
  { id: 'withdrawal', name: '提取与破产概率' },
  { group: '说明' },
  { id: 'method', name: '方法与数据' },
];

const nav = document.getElementById('nav');
const view = document.getElementById('view');
const title = document.getElementById('page-title');
let meta = null;
let current = null;          // the visit being shown: {ctrl}
let releaseHold = null;          // ends the scroll hold of the last K-line re-render

function buildNav(active) {
  nav.innerHTML = '';
  let n = 0;
  for (const t of TABS) {
    if (t.group) { nav.append(ui.h('div', { class: 'nav-group' }, t.group)); continue; }
    n += 1;
    nav.append(ui.h('a', { href: `#${t.id}`, class: t.id === active ? 'active' : '' },
      ui.h('span', { class: 'nav-num' }, String(n)), t.name));
  }
}

// "#scenarios?oil_war" -> {id: 'scenarios', param: 'oil_war'}
function parseRoute(route) {
  const s = String(route || '');
  const k = s.indexOf('?');
  const id = k < 0 ? s : s.slice(0, k);
  let param = k < 0 ? null : s.slice(k + 1);
  if (param !== null) { try { param = decodeURIComponent(param); } catch (_) { /* keep it as typed */ } }
  return { id, param: param || null };
}

// .pos / .neg and every CSS up / down colour follow the K-line setting through a body class.
function paintScheme() {
  const s = charts.getKlineScheme();
  document.body.classList.toggle('kline-us', s === 'us');
  document.body.classList.toggle('kline-cn', s !== 'us');
}

// A re-render in place (the K-line toggle) keeps the reader where they were. The page comes back in two steps:
// its controls at once, its results a moment later above the sections below them. Until the page is as tall as it
// was (or 15 s pass) #view keeps its old height (so the scroll position stays reachable) and opts out of scroll
// anchoring: the browser would otherwise anchor on a section that sits below the still-empty results and push the
// reader thousands of pixels down when they arrive.
function holdPlace(page, height, pageHeight) {
  if (releaseHold) releaseHold();
  view.style.minHeight = `${height}px`;
  view.style.overflowAnchor = 'none';
  let timer = 0;
  const ro = typeof ResizeObserver === 'function'
    ? new ResizeObserver(() => { if (page.offsetHeight >= pageHeight - 40) release(); }) : null;
  function release() {
    ro?.disconnect();
    clearTimeout(timer);
    view.style.minHeight = '';
    view.style.overflowAnchor = '';
    if (releaseHold === release) releaseHold = null;
  }
  ro?.observe(page);
  timer = setTimeout(release, 15000);
  releaseHold = release;
}

// keepScroll: re-render the same page where the reader is (the K-line toggle), instead of from the top.
async function show(route, { keepScroll = false } = {}) {
  const { id, param } = parseRoute(route);
  const tab = TABS.find(t => t.id === id) || TABS[1];
  if (current) current.ctrl.abort();
  const visit = { ctrl: new AbortController() };
  current = visit;
  const alive = () => current === visit && !visit.ctrl.signal.aborted;
  const y = window.scrollY;
  const height = view.offsetHeight;
  const pageHeight = view.querySelector(':scope > .page')?.offsetHeight || 0;
  buildNav(tab.id);
  title.textContent = tab.name;
  document.title = `${tab.name} · Causeway`;
  if (releaseHold) releaseHold();
  if (keepScroll) { view.style.minHeight = `${height}px`; view.style.overflowAnchor = 'none'; }   // until holdPlace
  charts.disposeWithin(view);            // every ECharts instance and observer of the previous page
  view.replaceChildren(ui.loading());
  if (!keepScroll) window.scrollTo(0, 0);
  try {
    const mod = await import(`./tabs/${tab.id}.js`);
    if (!alive()) return;
    const page = ui.h('div', { class: 'page', 'data-tab': tab.id });
    view.replaceChildren(page);
    if (keepScroll) holdPlace(page, height, pageHeight);
    const target = await mod.render(page, { api, charts, ui, meta, param, route: tab.id, signal: visit.ctrl.signal, alive, rerender: keepScroll });
    if (!alive()) return;
    if (keepScroll) window.scrollTo(0, y);            // a re-render stays where the reader was
    else if (target instanceof Element && target.isConnected) ui.pinIntoView(target, { block: 'start' });   // a deep link
  } catch (err) {
    if (!alive()) return;
    console.error(err);
    charts.disposeWithin(view);
    view.replaceChildren(ui.notice(`页面加载失败：${err.message}`, 'error'));
    if (err.status === 404) view.append(ui.buildHint(err));          // the command that builds the missing file
  }
}

async function boot() {
  const pref = document.getElementById('kline-color');
  pref.value = charts.getKlineScheme();
  paintScheme();
  pref.addEventListener('change', () => {
    charts.setKlineScheme(pref.value);
    paintScheme();
    show(location.hash.slice(1), { keepScroll: true });
  });
  try {
    meta = await api.get('meta');
    ui.setMeta(meta);
    document.getElementById('data-date').textContent = `数据截至 ${String(meta.data_end || '').slice(0, 7)}`;
    document.getElementById('build-info').textContent = `构建于 ${String(meta.built_at || '').replace('T', ' ').replace('Z', ' UTC')}`;
  } catch (err) {
    document.getElementById('data-date').textContent = '尚未构建';
  }
  ui.watchTables(view);                // tables wider than their card wrap, then stack (ui.fitTable)
  window.addEventListener('hashchange', () => show(location.hash.slice(1)));
  show(location.hash.slice(1) || 'overview');
}

boot();
