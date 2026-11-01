// App shell: hash router + lazy-loaded tabs. Each tab module exports
//   async function render(root, ctx)   where ctx = {api, charts, ui, meta}
import * as api from './api.js';
import * as charts from './charts.js';
import * as ui from './ui.js';

const TABS = [
  { group: '历史与因果' },
  { id: 'overview', name: '总览' },
  { id: 'eras', name: '历史时代画像' },
  { id: 'causality', name: '因果传导' },
  { group: '验证' },
  { id: 'backtest', name: '回测：模拟 vs 实际' },
  { id: 'diagnostics', name: '诊断与迭代' },
  { group: '未来' },
  { id: 'scenarios', name: '未来情景' },
  { id: 'lab', name: '情景实验室（LLM）' },
  { id: 'portfolio', name: '组合评估' },
  { id: 'withdrawal', name: '提取与破产概率' },
  { group: '说明' },
  { id: 'method', name: '方法与数据' },
];

const nav = document.getElementById('nav');
const view = document.getElementById('view');
const title = document.getElementById('page-title');
let meta = null;

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

async function show(id) {
  const tab = TABS.find(t => t.id === id) || TABS[1];
  buildNav(tab.id);
  title.textContent = tab.name;
  view.innerHTML = '';
  view.append(ui.loading());
  try {
    const mod = await import(`./tabs/${tab.id}.js`);
    view.innerHTML = '';
    await mod.render(view, { api, charts, ui, meta });
  } catch (err) {
    console.error(err);
    view.innerHTML = '';
    view.append(ui.notice(`页面加载失败：${err.message}`, 'error'));
    if (err.status === 404) {
      view.append(ui.h('p', { class: 'lead', style: { marginTop: '12px' } },
        '结果文件还没有生成。在项目根目录运行：', ui.h('code', { class: 'mono' }, 'python -m scenariolab.build')));
    }
  }
  window.scrollTo(0, 0);
}

async function boot() {
  const pref = document.getElementById('kline-color');
  pref.value = charts.klineColors().up === '#d9363e' ? 'cn' : 'us';
  pref.addEventListener('change', () => { charts.setKlineScheme(pref.value); show(location.hash.slice(1)); });
  try {
    meta = await api.get('meta');
    ui.setMeta(meta);
    document.getElementById('data-date').textContent = `数据截至 ${meta.data_end.slice(0, 7)}`;
    document.getElementById('build-info').textContent = `构建于 ${meta.built_at.replace('T', ' ').replace('Z', ' UTC')}`;
  } catch (err) {
    document.getElementById('data-date').textContent = '尚未构建';
  }
  window.addEventListener('hashchange', () => show(location.hash.slice(1)));
  show(location.hash.slice(1) || 'overview');
}

boot();
