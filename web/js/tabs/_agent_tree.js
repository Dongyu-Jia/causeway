// Browser of a snapshot or a run (a helper module of agents.js): metadata, the probability tree (an overview of every
// quarter and node, drawn on a canvas; a two-tier drawing of action clusters and consequence branches when the window
// is narrow enough; an outline by quarter), filters by actor and event, the node detail, the path view, the branch
// comparison (three trees, attribution, factor paths, asset fans) and on-demand generation of branches the snapshot
// did not expand (未生成 → 即时生成, saved in the overlay; no confirmation dialog, the user's decision).
import * as C from './_agent_common.js';
import * as D from './_agent_data.js';
import { TreeStore } from './_agent_data.js';
import { emblem, sprite } from './_agent_art.js';

const { h } = C;
const INK = '#1d2433', INK3 = '#7a8396', LINE = '#e3e6ec', ACCENT = '#3b5bdb';

const CSS = `
.agents-heat { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
.agents-heat th, .agents-heat td { padding: 4px 8px; border: 1px solid var(--line); text-align: right; white-space: nowrap; }
.agents-heat th:first-child { text-align: left; }
.agents-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 6px 18px; font-size: 13px; }
.agents-meta div { display: flex; gap: 8px; min-width: 0; }
.agents-meta dt { color: var(--ink-3); flex: 0 0 auto; }
.agents-meta dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.agents-ov { position: relative; width: 100%; }
.agents-ov canvas { display: block; width: 100%; cursor: pointer; touch-action: manipulation; }
.agents-tip { position: absolute; pointer-events: none; z-index: 3; background: rgba(29, 36, 51, .95); color: #fff; font-size: 12px; line-height: 1.5;
  padding: 7px 10px; border-radius: 6px; max-width: 300px; box-shadow: 0 4px 12px rgba(0,0,0,.18); }
.agents-tip b { color: #fff; } .agents-tip .m { color: #c9d1e3; }
.agents-ctrl { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-bottom: 10px; font-size: 12px; color: var(--ink-3); }
.agents-ctrl select { font-size: 12px; padding: 4px 8px; max-width: 100%; }
.agents-ctrl .seg button { font-size: 12px; padding: 4px 10px; }
.agents-ctrl input[type=range] { width: 160px; }
.agents-legend { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; font-size: 12px; color: var(--ink-3); margin-top: 6px; }
.agents-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: -1px; margin-right: 4px; }
.agents-legend .grad { width: 90px; height: 8px; border-radius: 4px; display: inline-block; vertical-align: 0; }
.agents-loadbar { font-size: 12px; color: var(--ink-3); display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }
.agents-loadbar .agents-progress { flex: 1 1 auto; max-width: 260px; height: 6px; }
.agents-outline details { border-bottom: 1px solid var(--line); }
.agents-outline summary { cursor: pointer; padding: 7px 4px; font-size: 13px; display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.agents-outline summary .q { font-weight: 650; min-width: 64px; }
.agents-outline summary .s { color: var(--ink-3); font-size: 12px; }
.agents-orow { display: grid; grid-template-columns: 56px minmax(0, 1fr) auto; gap: 8px; align-items: center; padding: 5px 6px 5px 18px; font-size: 13px;
  border-radius: 6px; cursor: pointer; }
.agents-orow:hover, .agents-orow:focus-visible { background: var(--accent-soft); outline: none; }
.agents-orow.on { background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); }
.agents-orow .p { font-variant-numeric: tabular-nums; color: var(--accent); font-weight: 600; }
.agents-orow .l { min-width: 0; overflow-wrap: anywhere; }
.agents-orow .r { font-size: 12px; white-space: nowrap; }
.agents-crumb { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; font-size: 12px; color: var(--ink-3); margin-bottom: 8px; }
.agents-crumb button { border: 0; background: none; padding: 1px 4px; border-radius: 4px; color: var(--accent); cursor: pointer; font: inherit; }
.agents-crumb button:hover { background: var(--accent-soft); }
.agents-tabs { margin: 6px 0 12px; }
.agents-tabs .seg button { font-size: 12px; padding: 5px 11px; }
.agents-actorblock { border: 1px solid var(--line); border-left: 4px solid var(--c, var(--line)); border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; background: var(--panel); }
.agents-actorhead { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; margin-bottom: 6px; }
.agents-actorhead .agents-emblem { width: 34px; height: 34px; flex: 0 0 auto; color: var(--ink); }
.agents-actorhead b { font-size: 14px; }
.agents-signal { margin-top: 6px; font-size: 13px; background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: 6px 10px; display: inline-block; max-width: 100%; }
.agents-why { font-size: 12px; color: var(--ink-2); line-height: 1.5; }
.agents-chosen td { font-weight: 650; background: #f6f8ff; }
.agents-fan { display: grid; gap: 10px; }
.agents-cluster { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
.agents-cluster-h { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 13px; margin-bottom: 6px; }
.agents-diamond { width: 10px; height: 10px; transform: rotate(45deg); display: inline-block; flex: 0 0 auto; }
.agents-kid { display: grid; grid-template-columns: 20px minmax(0, 1fr) 70px auto; gap: 8px; align-items: center; padding: 4px 0; font-size: 13px; }
.agents-kid .dotk { width: 12px; height: 12px; border-radius: 50%; background: var(--accent); justify-self: center; }
.agents-kid.ghost .dotk { background: transparent; border: 2px dashed var(--ink-3); }
.agents-kid.overlay .dotk { background: #fff; border: 2px dashed var(--accent); }
.agents-kid.ghost { color: var(--ink-2); }
.agents-kid .btn { padding: 2px 9px; font-size: 12px; }
.agents-gen { border: 1px dashed var(--accent); background: #f6f8ff; border-radius: 8px; padding: 8px 12px; font-size: 13px; margin: 8px 0; display: flex;
  gap: 8px; align-items: center; flex-wrap: wrap; }
.agents-gen.err { border-color: #f6c7cc; background: #fdf1f2; color: #8a1c24; }
.agents-gen.warn { border-color: #f5d9a8; background: #fff8eb; color: #7a4d00; }
.agents-pre { white-space: pre-wrap; word-break: break-word; background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
  font-size: 12px; line-height: 1.6; max-height: 380px; overflow: auto; margin: 6px 0 0; color: var(--ink); }
.agents-hist { position: relative; height: 8px; border-radius: 4px; background: linear-gradient(90deg, #eef0f4, #dfe4f0 50%, #eef0f4); min-width: 80px; }
.agents-hist i { position: absolute; top: -3px; width: 3px; height: 14px; background: var(--accent); border-radius: 2px; transform: translateX(-1px); }
.agents-range { position: relative; height: 14px; min-width: 120px; }
.agents-range .t { position: absolute; top: 6px; left: 0; right: 0; height: 2px; background: var(--line); }
.agents-range .b { position: absolute; top: 3px; height: 8px; border-radius: 4px; background: rgba(59, 91, 219, .22); }
.agents-range .m { position: absolute; top: 1px; width: 3px; height: 12px; background: var(--ink); border-radius: 2px; transform: translateX(-1px); }
.agents-range .z { position: absolute; top: 0; width: 1px; height: 14px; background: var(--ink-3); }
.agents-attr { display: grid; grid-template-columns: 96px minmax(0, 1fr) 64px; gap: 6px 10px; align-items: center; font-size: 12px; }
.agents-attr .bar { position: relative; height: 14px; background: var(--panel-2); border-radius: 3px; }
.agents-attr .bar span { position: absolute; top: 0; height: 14px; }
.agents-attr .bar .zero { position: absolute; top: -2px; width: 1px; height: 18px; background: var(--ink-3); left: 50%; }
.agents-tri { display: grid; grid-template-columns: 92px minmax(0, 1fr) 150px; gap: 4px 10px; align-items: center; font-size: 12px; }
.agents-tri .name { color: var(--ink-2); }
.agents-tri .vals { font-variant-numeric: tabular-nums; color: var(--ink-2); text-align: right; }
.agents-tri .asset { grid-column: 1 / -1; font-weight: 650; font-size: 13px; margin-top: 8px; }
.agents-pathwrap { max-height: 520px; overflow: auto; }
.agents-pathwrap table.tbl th { position: sticky; top: 0; z-index: 1; }
.agents-ev { display: flex; gap: 1px; align-items: flex-end; height: 34px; margin-top: 4px; }
.agents-ev span { flex: 1 1 0; background: var(--accent); opacity: .85; min-height: 1px; border-radius: 1px 1px 0 0; }
.agents-runbar { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; font-size: 13px; }
@media (max-width: 760px) {
  .agents-kid { grid-template-columns: 18px minmax(0, 1fr) auto; }
  .agents-kid .pc { display: none; }
  .agents-tri { grid-template-columns: 76px minmax(0, 1fr); }
  .agents-tri .vals { grid-column: 2; text-align: left; }
  .agents-attr { grid-template-columns: 80px minmax(0, 1fr) 54px; }
}
`;

const COLOR_MODES = [
  { value: 'eq', label: '美股累计回报' }, { value: 'regime', label: '宏观体制' }, { value: 's_ff', label: '政策利率' },
  { value: 's_core_yoy', label: '核心通胀' }, { value: 's_unrate', label: '失业率' }, { value: 'none', label: '不着色' },
];
const ATTR_PARTS = [
  { id: 'actions', zh: '行动经模型', color: '#9775fa' }, { id: 'partition', zh: '后果分区', color: '#74c0fc' },
  { id: 'fin_conditions', zh: '金融条件分段（情景假说）', color: '#ffc078' }, { id: 'equity_leg', zh: '股票腿分段', color: '#e599f7' },
  { id: 'mechanism', zh: '机制', color: '#adb5bd' },
];
// level colouring (政策利率, 核心通胀, 失业率): a neutral sand-to-umber ramp (blue is probability; red/green are returns)
const SEQ_LO = '#f6efe2', SEQ_HI = '#7c4a12';
const MAIN_ASSETS = ['us_equity', 'nasdaq', 'treasury_10y', 'gold', 'crude_oil'];
const ASSET_ZH = { us_equity: '美股大盘', nasdaq: '纳斯达克100', treasury_10y: '10年期美债', gold: '黄金', crude_oil: '原油', real_estate: '房地产 REITs',
  cash: '现金', energy: '石油能源股', industrials: '工业制造股', staples: '必需消费股', healthcare: '医疗保健股', utilities: '公用事业股', financials: '金融股' };
export const assetZh = (id, ui) => ui?.META?.assets?.[id]?.name || ASSET_ZH[id] || '资产';
const FACTORS = [
  { id: 's_ff', zh: '政策利率', unit: '%' }, { id: 's_unrate', zh: '失业率', unit: '%' }, { id: 's_core_yoy', zh: '核心 CPI 同比', unit: '%' },
  { id: 's_y10', zh: '10 年期收益率', unit: '%' }, { id: 's_spread', zh: '信用利差', unit: '个百分点' }, { id: 's_term', zh: '期限利差', unit: '个百分点' },
  { id: 's_gz', zh: 'GZ 信用利差', unit: '个百分点' }, { id: 's_cape', zh: '估值（log CAPE）', unit: '' },
];

// ================================================================ entry
export async function renderBrowser(root, env) {
  C.ensureStyle('agents-tree', CSS);
  const { ui } = env.ctx;
  const { AG, tree, source: src } = env;
  C.put(root, ui.loading('载入…'));
  try {
    await tree.init();
  } catch (err) {
    if (!env.alive()) return;
    C.fill(root, ui.notice(`读取${src.kind === 'snapshot' ? '快照' : '推演'}失败：${D.errorText(err)}`, 'error'));
    return;
  }
  if (!env.alive()) return;
  C.fill(root);
  if (!document.getElementById('agents-sprite')) { const sp = sprite(); sp.id = 'agents-sprite'; C.put(document.body, sp); }

  const sum = tree.summary;
  let metaEl = metaCard(env, sum);
  const runBar = h('div', { class: 'agents-sticky' });
  C.put(root, runBar, metaEl, honestyCard(env));
  const paintRunBar = () => {
    const m = src.kind === 'snapshot' ? (tree.summary?.manifest || {}) : (tree.summary || {});
    const st = m.settings || {};
    const lv = (src.kind === 'snapshot' ? m.levels : tree.summary?.levels) || [];
    const last = Array.isArray(lv) && lv.length ? lv[lv.length - 1] : null;
    const done = m.levels_done ?? (Array.isArray(lv) ? lv.length : 0);
    C.fill(runBar, h('div', { class: 'agents-runbar' },
      h('b', {}, src.kind === 'snapshot' ? `快照 ${m.id || src.id}` : `推演 ${m.name || m.id || src.id}`),
      h('span', {}, `第 ${done} / ${st.depth ?? '—'} 季 · 束宽 ${st.B ?? '—'}`),
      last && C.isNum(last.own_mass) ? h('span', { title: '最后一层保留的节点自身的质量；其余是并入的剪枝质量' }, `末层覆盖 ${C.prob(last.own_mass)}`) : null,
      C.isNum(last?.discrete_misstated) ? h('span', { title: '粒子的离散状态（事实、政治状态、承诺）与所在节点不一致的比例（按质量加权，取最不一致的一列）' },
        `离散近似 ${C.prob(last.discrete_misstated)}`) : null,
      m.stub ? C.chip('桩模型', 'warn') : null));
  };
  paintRunBar();

  const treeCard = h('div');
  const detailBox = h('div', { style: { marginTop: '16px', scrollMarginTop: '72px' } });
  const pathBox = h('div', { style: { marginTop: '16px' } });
  const cmpBox = h('div', { style: { marginTop: '16px' } });
  C.put(root, h('div', { class: 'section-title' }, '情景树'), treeCard, detailBox, pathBox, h('div', { class: 'section-title' }, '分支对比'), cmpBox);

  const B = {
    env, src, tree, treeCard, detailBox, pathBox, cmpBox,
    select: (key, { scroll = false } = {}) => {
      AG.node = key;
      env.setRoute();
      ov?.fitWindow();
      ov?.draw();
      B.onWindow?.();
      drawOutline();
      drawDetail(B, key, { scroll });
    },
  };
  env.browser = B;
  B.refreshMeta = () => { if (!env.alive()) return; const n = metaCard(env, tree.summary); metaEl.replaceWith(n); metaEl = n; paintRunBar(); };

  // ---- tree card
  const ctrl = h('div', { class: 'agents-ctrl' });
  const loadBar = h('div', { class: 'agents-loadbar' });
  const ovBox = h('div', { class: 'agents-ov' });
  const evStrip = h('div');
  const legend = h('div', { class: 'agents-legend' });
  const outline = h('div', { class: 'agents-outline' });
  const narrow = () => (root.clientWidth || window.innerWidth) < 640;
  if (!AG.treeView) AG.treeView = narrow() ? 'outline' : 'overview';
  if (!AG.colour) AG.colour = 'eq';
  if (!AG.filters) AG.filters = { actor: '', option: '', event: '' };
  if (!AG.win) AG.win = 'all';
  const viewSeg = ui.seg([{ value: 'overview', label: '全景图' }, { value: 'outline', label: '按季度大纲' }], AG.treeView, (v) => {
    AG.treeView = v; showView();
  });
  const body = h('div');
  C.put(treeCard, ui.card({
    title: src.kind === 'snapshot' ? `快照「${src.id}」的情景树` : `推演「${sum.name || src.id}」的情景树`,
    sub: '横轴是季度，每个圆点是一个季末状态；面积 ∝ 到达它的路径概率，线的粗细 ∝ 流过的概率。窗口不超过 10 季时画成两层：◇ 是动作簇（这一季的联合动作），'
      + '其后的圆点是后果分支。被剪掉的分支（未生成）在选中节点后以虚线圆圈出现，点击即时生成。',
    actions: viewSeg,
  }, loadBar, ctrl, body));
  let ov = null;

  const actors = C.actorIds();
  const actorSel = h('select', { 'aria-label': '按行为者筛选' }, h('option', { value: '' }, '全部行为者'),
    actors.map(a => h('option', { value: a, selected: AG.filters.actor === a ? true : null }, C.actorName(a))));
  const optSel = h('select', { 'aria-label': '按动作筛选' });
  const fillOpts = () => {
    const a = C.catalogueActor(AG.filters.actor);
    C.fill(optSel, h('option', { value: '' }, a ? '任一动作' : '先选行为者'));
    optSel.disabled = !a;
    for (const d of (a?.domains || []).filter(d => d.role === 'main')) {
      C.put(optSel, h('optgroup', { label: d.zh }, (d.options || []).map(o => h('option', { value: o.id, selected: AG.filters.option === o.id ? true : null }, o.zh))));
    }
  };
  fillOpts();
  actorSel.onchange = () => { AG.filters.actor = actorSel.value; AG.filters.option = ''; fillOpts(); filtersChanged(); };
  optSel.onchange = () => { AG.filters.option = optSel.value; filtersChanged(); };
  const evSel = h('select', { 'aria-label': '按事件筛选' });
  const fillEvents = () => {
    const facts = new Map();
    for (const n of tree.byKey.values()) for (const [k, t] of C.liveFacts(n.facts)) facts.set(k, t);
    const cur = AG.filters.event;
    C.fill(evSel, h('option', { value: '' }, '全部事件'),
      h('optgroup', { label: '季度特征' }, [['stress', '压力季度'], ['kind:nature_event', '自然事件分支'], ['kind:mechanism', '机制分支'],
        ['kind:partition', '分区分支'], ['overlay', '即时生成']].map(([v, t]) => h('option', { value: v }, t))),
      h('optgroup', { label: '宏观体制（简化规则）' }, Object.entries(C.REGIMES).map(([id, r]) => h('option', { value: `regime:${id}` }, r.name))),
      facts.size ? h('optgroup', { label: '持续事实' }, [...facts].sort((a, b) => a[1].localeCompare(b[1], 'zh')).map(([k, t]) => h('option', { value: `fact:${k}` }, t))) : null);
    evSel.value = cur;
  };
  fillEvents();
  evSel.onchange = () => { AG.filters.event = evSel.value; filtersChanged(); };
  const colourSel = h('select', { 'aria-label': '圆点颜色' }, COLOR_MODES.map(m => h('option', { value: m.value, selected: AG.colour === m.value ? true : null }, m.label)));
  colourSel.onchange = () => { AG.colour = colourSel.value; ov?.layout(); ov?.draw(); drawLegend(); };
  const winSeg = ui.seg([{ value: 'all', label: '全部' }, { value: '24', label: '24 季' }, { value: '10', label: '10 季' }, { value: '5', label: '5 季' }],
    AG.win, (v) => { AG.win = v; ov?.fitWindow(); ov?.draw(); syncWinRange(); });
  const winRange = h('input', { type: 'range', min: 0, max: Math.max(0, tree.depth), value: 0, 'aria-label': '窗口起点（季）',
    oninput: (e) => { ov?.setStart(Number(e.target.value)); } });
  const winText = h('span', { class: 'agents-wintext' });
  const syncWinRange = () => {
    const span = AG.win === 'all' ? tree.depth + 1 : Number(AG.win);
    const qa = tree.nodesAt(ov?.a || 0)[0]?.quarter, qb = tree.nodesAt(ov?.b ?? tree.depth)[0]?.quarter;
    winText.textContent = AG.win === 'all' || !qa ? '' : `${qa} – ${qb || ''}`;
    winRange.max = String(Math.max(0, tree.depth + 1 - span));
    winRange.value = String(ov?.a || 0);
    winRange.disabled = AG.win === 'all' || span >= tree.depth + 1;
    winRange.style.display = AG.treeView === 'overview' && !winRange.disabled ? '' : 'none';
  };
  C.put(ctrl, h('span', {}, '筛选'), actorSel, optSel, evSel, h('span', {}, '颜色'), colourSel,
    h('span', { class: 'agents-ovonly' }, '窗口'), winSeg, winRange, winText);

  function filtersChanged() {
    ov?.draw();
    drawEventStrip();
    drawOutline();
  }

  // ---- matching
  const regimeMemo = new Map();
  const regimeOfNode = (n) => {
    if (regimeMemo.has(n.key)) return regimeMemo.get(n.key);
    const us = [];
    let cur = n;
    for (let i = 0; i < 5 && cur; i++) { us.push(cur.rep?.s_unrate); cur = cur.parent ? tree.get(cur.parent) : null; }
    const rg = C.regimeOf(n.rep, n.gdp, us);
    regimeMemo.set(n.key, rg);
    return rg;
  };
  B.regimeOfNode = regimeOfNode;
  const matches = (n) => {
    const f = AG.filters;
    if (f.actor) {
      const pick = n.cl?.realized?.[f.actor] ?? n.cl?.head?.[f.actor];
      if (f.option ? pick !== f.option : !pick) return false;
    }
    if (f.event) {
      const [k, v] = f.event.split(':');
      if (k === 'stress' && !n.stress) return false;
      if (k === 'overlay' && n.source !== 'overlay') return false;
      if (k === 'kind' && n.br?.kind !== v) return false;
      if (k === 'fact' && !n.facts?.[v]) return false;
      if (k === 'regime' && regimeOfNode(n).id !== v) return false;
    }
    return true;
  };
  const filtering = () => !!(AG.filters.actor || AG.filters.event);
  B.matches = matches;

  // ---- views
  function showView() {
    C.fill(body);
    ctrl.querySelectorAll('.agents-ovonly').forEach(x => { x.style.display = AG.treeView === 'overview' ? '' : 'none'; });
    winSeg.style.display = AG.treeView === 'overview' ? '' : 'none';
    if (AG.treeView === 'overview') {
      C.put(body, ovBox, evStrip, legend);
      if (!ov) ov = new Overview(ovBox, B);
      ov.resize();
      syncWinRange();
      drawLegend();
      drawEventStrip();
    } else {
      C.put(body, evStrip, outline);
      drawEventStrip();
      drawOutline();
    }
  }
  B.onWindow = () => syncWinRange();
  B.onLayout = () => drawLegend();
  B.redrawTree = () => { ov?.layout(); ov?.draw(); fillEvents(); drawEventStrip(); drawOutline(); };

  function drawLegend() {
    C.fill(legend);
    const m = AG.colour;
    if (m === 'eq') {
      const { up, down } = env.ctx.charts.klineColors();
      const g = h('span', { class: 'grad' });
      g.style.background = `linear-gradient(90deg, ${down}, #f1f3f5, ${up})`;
      C.put(legend, h('span', {}, '颜色：到该季末的美股累计回报中位数 ', g, ` ±${Math.round((ov?.eqBound || 0.3) * 100)}%`));
    } else if (m === 'regime') {
      C.put(legend, ...Object.values(C.REGIMES).map(r => { const i = h('i'); i.style.background = r.color; return h('span', { title: C.REGIME_RULE }, i, r.name); }),
        h('span', {}, '（简化规则，悬停看说明）'));
    } else if (m === 'none') {
      C.put(legend, h('span', {}, '圆点不着色'));
    } else {
      const g = h('span', { class: 'grad' });
      g.style.background = `linear-gradient(90deg, ${SEQ_LO}, ${SEQ_HI})`;
      const r = ov?.seqMode === m ? ov.seqRange : null;
      C.put(legend, h('span', {}, `颜色：${FACTORS.find(f => f.id === m)?.zh || ''}（季末中位数）`, g, r ? `${C.num(r[0], 2)} – ${C.num(r[1], 2)}%` : '（载入中）'));
    }
    const ring = h('i'); ring.style.cssText = 'background:#fff;border:2px dashed #3b5bdb';
    const ghost = h('i'); ghost.style.cssText = 'background:#fff;border:2px dashed #7a8396';
    C.put(legend, h('span', {}, ring, '即时生成'), h('span', {}, ghost, '未生成（选中节点后出现，点击生成）'));
  }

  function drawEventStrip() {
    C.fill(evStrip);
    if (!filtering()) return;
    const bars = [];
    let any = 0;
    for (let d = 0; d <= tree.depth; d++) {
      const nodes = tree.nodesAt(d).filter(TreeStore.shown);
      const tot = nodes.reduce((s, n) => s + (n.p || 0), 0) || 1;
      const m = nodes.filter(n => matches(n)).reduce((s, n) => s + (n.p || 0), 0) / tot;
      any = Math.max(any, m);
      const sp = h('span', { title: `${nodes[0]?.quarter || ''}：${C.prob(m)}` });
      sp.dataset.m = String(m);
      bars.push(sp);
    }
    for (const sp of bars) sp.style.height = `${any > 0 ? Math.max(Number(sp.dataset.m) > 0 ? 2 : 0, (Number(sp.dataset.m) / any) * 34) : 0}px`;
    C.put(evStrip, h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '8px' } },
      `符合筛选的节点在各季的概率质量：柱高按最高的一季（${C.prob(any)}）缩放；即时生成的节点按它们自己的路径概率计入`), h('div', { class: 'agents-ev' }, bars));
  }

  function drawOutline() {
    if (AG.treeView !== 'outline') return;
    const open = new Set([...outline.querySelectorAll('details[open]')].map(d => Number(d.dataset.level)));
    const sel = AG.node ? tree.get(AG.node) : null;
    if (sel) open.add(sel.level);
    if (!outline.children.length && !sel) { open.add(0); open.add(1); }
    if (!outline.children.length && sel) open.delete(0);
    C.fill(outline);
    for (let d = 0; d <= tree.depth; d++) {
      const nodes = tree.nodesAt(d).filter(TreeStore.shown).sort((a, b) => (b.p || 0) - (a.p || 0));
      const shown = filtering() ? nodes.filter(matches) : nodes;
      const info = tree.levelInfo?.[d] || tree.levelInfo?.[String(d)] || {};
      const det = h('details', { 'data-level': String(d) });
      if (open.has(d)) det.open = true;
      const top = shown[0];
      C.put(det, h('summary', {}, h('span', { class: 'q' }, d === 0 ? `${nodes[0]?.quarter || info.quarter || ''} 末` : (nodes[0]?.quarter || info.quarter || `第 ${d} 季`)),
        h('span', { class: 's' }, tree.loaded.has(d) ? `${filtering() ? `${shown.length} / ` : ''}${nodes.length} 个节点` : '载入中…'),
        top && d > 0 ? h('span', { class: 's' }, `最可能：${C.cut(C.diffLabel(top.cl?.label, nodes.map(n => n.cl?.label)), 18)} ${C.prob(top.p)}`) : null));
      const fill = () => {
        if (det.dataset.filled) return;
        det.dataset.filled = '1';
        for (const n of shown) C.put(det, outlineRow(n, nodes));
        if (!shown.length) C.put(det, h('div', { class: 'muted', style: { padding: '4px 18px', fontSize: '12px' } }, '没有符合筛选的节点'));
      };
      det.addEventListener('toggle', () => { if (det.open) fill(); });
      if (det.open) fill();
      C.put(outline, det);
    }
  }
  function outlineRow(n, siblings) {
    const eq = n.assets?.us_equity?.c?.[1];
    const row = h('div', { class: `agents-orow ${AG.node === n.key ? 'on' : ''}`, tabindex: '0', role: 'button',
      'aria-label': `${n.quarter} 路径概率 ${C.prob(n.p)}`,
      onclick: () => B.select(n.key, { scroll: true }),
      onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); B.select(n.key, { scroll: true }); } } },
    h('span', { class: 'p' }, C.prob(n.p)),
    h('span', { class: 'l', title: C.humanize(n.cl?.label || '') }, n.level === 0 ? '起点' : C.cut(C.humanize(C.diffLabel(n.cl?.label, siblings.map(s => s.cl?.label))), 30),
      n.br ? h('span', { class: 'muted' }, ` · ${n.br.label || C.BRANCH_KIND[n.br.kind] || ''}`) : null,
      n.source === 'overlay' ? ' ' : null, n.source === 'overlay' ? C.chip('即时生成', 'accent') : null,
      n.stress ? ' ' : null, n.stress ? C.chip('压力', 'warn') : null),
    h('span', { class: `r ${C.retCls(eq)}` }, n.level ? `美股 ${C.retText(eq)}` : ''));
    return row;
  }

  // ---- loading
  const paintLoad = () => {
    const done = tree.loaded.size, tot = tree.depth + 1;
    if (done >= tot && (!tree.loading || paintLoad.final)) { C.fill(loadBar); return; }
    const bar = h('span'); bar.style.width = `${((done / tot) * 100).toFixed(0)}%`;
    C.fill(loadBar, h('span', { class: 'spinner', style: { width: '12px', height: '12px' } }), `载入情景树：${done} / ${tot} 层`,
      h('div', { class: 'agents-progress' }, bar));
  };
  let lastPaint = 0;
  const off = tree.on((what) => {
    if (!env.alive()) { off(); return; }
    if (what === 'done') paintLoad.final = true;
    paintLoad();
    const now = Date.now();
    if (what === 'done' || what === 'overlay' || now - lastPaint > 250) {
      lastPaint = now;
      regimeMemo.clear();
      ov?.layout(); ov?.draw();
      if (what === 'done') { fillEvents(); drawEventStrip(); drawLegend(); syncWinRange(); }
      if (AG.treeView === 'outline') drawOutline();
      if (what === 'done') drawCompare(B);
    }
  });
  env.signal?.addEventListener('abort', () => { off(); ov?.destroy(); }, { once: true });
  showView();
  paintLoad();
  // a deep-linked node: load its level first
  if (AG.node) drawDetail(B, AG.node, { scroll: true });
  else drawDetail(B, null);
  tree.loadAll({ signal: env.signal }).then(() => {
    if (!env.alive()) return;
    if (!AG.node) {
      const best = tree.bestLeaf();
      if (best) B.select(best.key, { scroll: false });
    }
  }).catch(err => { if (err.name !== 'AbortError') console.error(err); });
  if (tree.complete) drawCompare(B);
  // a run still running: refresh as levels finish
  if (src.kind === 'run' && ['running', 'queued', 'cancelling'].includes(sum.status)) {
    const t = setInterval(async () => {
      if (!env.alive()) { clearInterval(t); return; }
      try {
        await tree.refresh();
        B.refreshMeta();
        if (!['running', 'queued', 'cancelling'].includes(tree.summary.status)) { clearInterval(t); drawCompare(B); }
      } catch (_) { /* keep polling */ }
    }, 4000);
    env.signal?.addEventListener('abort', () => clearInterval(t), { once: true });
  }
}

// ================================================================ honesty (计算口径与局限; AGENT_DESIGN §13 and UI §14)
export function honestyCard(env) {
  const { ui } = env.ctx;
  const li = (t) => h('li', {}, t);
  return h('details', { class: 'card', style: { marginTop: '12px' } },
    h('summary', { style: { cursor: 'pointer', fontWeight: 600 } }, '计算口径与局限'),
    h('ul', { class: 'list', style: { fontSize: '13px', marginTop: '8px' } },
      li('行为者只选动作，后果由合成器给出，价格只来自季度模型。'),
      li('账户是记账，不写价格，也不是模型的学习输入；每个粒子的账户按它自己的动作更新。'),
      li('大模型知道历史：任何历史回放都会泄漏信息（状态本身就能定年），只能当管道测试；能干净检验的只有护栏与恒等式、不调用大模型的先验树、合成状态上的变形测试，以及从今天起的前瞻账本，而账本要好几年才能分辨中等程度的偏差。'),
      li('概率是蓝色，涨跌颜色跟随 K 线配色；每个行为者的大模型权重、上限与精度权重都是判断值，尚未校准。'),
      li('束搜索每层只保留 B 个节点：被剪掉的分支的质量与粒子并入最近的保留节点，不会丢掉；一个子节点只能带一种离散状态（事实、政治状态、承诺），没有单独成支的少数状态在节点详情里标出比例。'),
      li('第 D 季之后没有智能体：按历史平均反应函数推演；回撤是季末口径。'),
      env.stub ? li('现在用的是高斯桩模型（不是资产模型）：资产价格只用于检查流程，不能当作结果，也不能导出。') : null),
    ui.notice('前瞻 · 未经验证：从今天出发的推演要等这些季度过去才能检验。', 'info'));
}

// ================================================================ metadata
function metaCard(env, sum) {
  const { ui } = env.ctx;
  const src = env.source;
  const m = src.kind === 'snapshot' ? (sum.manifest || {}) : sum;
  const st = m.settings || {};
  const calls = src.kind === 'snapshot' ? (m.calls || {}) : (m.stats || {});
  const ver = m.versions || {};
  const facts = m.institutions || [];
  const unconf = facts.filter(f => f.status !== 'confirmed').length;
  const lvKeys = Object.keys(sum.levels_view || (Array.isArray(sum.levels) ? null : sum.levels) || {}).map(Number);
  const lastLv = lvKeys.length ? Math.max(...lvKeys) : 0;
  const status = m.status;
  const pause = m.pause;
  const row = (k, v) => (v === null || v === undefined || v === '' ? null : h('div', {}, h('dt', {}, k), h('dd', {}, v)));
  const usage = calls.usage || {};
  const title = src.kind === 'snapshot' ? `快照「${m.id || src.id}」` : `推演「${m.name || m.id || src.id}」`;
  const chips = h('div', { class: 'agents-chips' },
    status === 'done' ? C.chip(src.kind === 'snapshot' ? '已冻结（只读）' : '完成', 'good') : status === 'paused' ? C.chip(`已暂停：${pause?.message || pause?.reason || ''}`, 'warn')
      : C.chip({ building: '构建中', running: '运行中', queued: '排队中', failed: '失败', cancelled: '已取消', cancelling: '正在停止' }[status] || status || '—', status === 'failed' ? 'bad' : 'accent'),
    m.stub ? C.chip('桩模型（不是资产模型）', 'warn', '资产价格来自高斯桩模型，只能检查流程') : null,
    C.chip('前瞻 · 未经验证', 'accent'),
    unconf ? C.chip(`构建时 ${unconf} 项锁定事实未确认`, 'warn', '快照与推演保留它们开始时的事实；之后的修改不改变它们') : C.chip('锁定事实全部已确认', 'good'),
    Object.keys(sum.overlay || {}).length ? C.chip(`覆盖层：${Object.keys(sum.overlay).length} 个即时生成的节点`, 'accent') : null);
  const exportWhy = `组合评估与提取服务尚不支持季度情景集${m.stub ? '；桩模型的结果也不能导出' : ''}`;
  return ui.card({ title, sub: src.kind === 'snapshot' ? '快照是一次束搜索的冻结结果：打开即用，不再运行模型。' : '推演的设置、版本与调用。',
    actions: h('div', { class: 'agents-export' }, h('button', { class: 'btn', type: 'button', disabled: true, title: exportWhy }, '导出到组合评估'),
      h('span', { class: 'why' }, `暂不可用：${exportWhy}`)) },
  chips,
  h('dl', { class: 'agents-meta', style: { marginTop: '10px' } },
    row('规模', `束宽 ${st.B ?? '—'}（每层保留的节点）× ${st.depth ?? '—'} 季（每节点 ${st.K ?? '—'} 个动作簇 × 最多 ${st.M_max ?? '—'} 个后果）`),
    row('完成层数', `${m.levels_done ?? lastLv} / ${st.depth ?? '—'}`),
    row(src.kind === 'snapshot' ? '构建于' : '创建于', C.when(m.created)),
    row('完成于', C.when(m.finished)),
    row('数据截至', m.data_end || '2026-09'),
    row('起点', `${env.cfg.starts?.today?.root_quarter || '2026Q3'} 末，第一季 ${st.first_quarter || '2026Q4'}`),
    row('模型', `${m.model || st.model || 'claude-opus-5-5'}，力度 ${C.EFFORT_ZH[m.effort || 'medium'] || '中'}`),
    row('大模型调用', `新 ${m.spent_calls ?? m.calls_paid ?? calls.calls_paid ?? calls.calls_new ?? 0} · 缓存 ${calls.cached ?? 0} · 失败 ${calls.failed ?? 0} · 修复 ${calls.repairs ?? 0} · 回退 ${calls.fallbacks ?? 0}${calls.budget_hits ? ` · 达到上限 ${calls.budget_hits}` : ''}`),
    row('用量上限', st.hard_cap ? `硬上限 ${st.hard_cap} 次新调用（预计的 1.2 倍）${st.max_calls ? `；软上限 ${st.max_calls} 次（之后只用历史基率）` : ''}` : null),
    row('token', usage.input_tokens !== undefined ? `输入 ${(usage.input_tokens / 1e6).toFixed(2)}M · 输出 ${(usage.output_tokens / 1e6).toFixed(2)}M` : null),
    row('后果合成器', st.synth ? `${{ off: '关闭', gated: '门控', every: '每个节点' }[st.synth.mode] || st.synth.mode}，前 ${st.synth.depth} 季` : null),
    row('大小', C.isNum(m.size_mb) ? `${m.size_mb.toFixed(1)} MB` : null),
    row('引擎版本', C.versionLabel(ver.engine)), row('提示词版本', C.versionLabel(ver.prompts)), row('行为者目录', C.versionLabel(ver.catalogue)),
    row('资产模型', C.versionLabel(ver.model_adapter)), row('账户', C.versionLabel(ver.accounts)), row('先验', C.versionLabel(ver.priors)),
    row('树哈希', m.tree_hash ? h('code', { class: 'mono' }, m.tree_hash) : null)));
}

// ================================================================ the overview (canvas)
class Overview {
  constructor(box, B) {
    this.box = box; this.B = B; this.tree = B.tree; this.AG = B.env.AG;
    this.canvas = h('canvas', { role: 'img', 'aria-label': '情景树全景图：横轴是季度，圆点是节点，面积表示概率' });
    this.tip = h('div', { class: 'agents-tip', style: { display: 'none' } });
    C.fill(box, this.canvas, this.tip);
    this.a = 0; this.b = this.tree.depth;
    this.pos = new Map();
    this.hits = [];
    this.ro = new ResizeObserver(() => { const w = this.box.clientWidth; if (w && w !== this.w) this.resize(); });
    this.ro.observe(box);
    this.canvas.addEventListener('mousemove', (e) => this.hover(e));
    this.canvas.addEventListener('mouseleave', () => { this.tip.style.display = 'none'; this.hot = null; this.draw(); });
    this.canvas.addEventListener('click', (e) => this.click(e));
    this.fitWindow();
    this.layout();
  }
  destroy() { try { this.ro.disconnect(); } catch (_) { /* gone */ } }
  get span() { return this.AG.win === 'all' ? this.tree.depth + 1 : Math.min(this.tree.depth + 1, Number(this.AG.win)); }
  fitWindow() {
    const span = this.span;
    const sel = this.AG.node ? this.tree.get(this.AG.node) : null;
    let a = this.a;
    if (sel) {
      if (sel.level < a + 1 || sel.level > a + span - 2) a = sel.level - Math.floor(span / 2);
    }
    this.a = C.clamp(a, 0, Math.max(0, this.tree.depth + 1 - span));
    this.b = Math.min(this.tree.depth, this.a + span - 1);
  }
  setStart(a) { this.a = C.clamp(a, 0, Math.max(0, this.tree.depth + 1 - this.span)); this.b = Math.min(this.tree.depth, this.a + this.span - 1); this.draw(); this.B.onWindow?.(); }
  resize() {
    const w = this.box.clientWidth || 600;
    this.w = w;
    let rows = 1;
    for (const arr of this.tree.levels.values()) rows = Math.max(rows, arr.filter(TreeStore.shown).length);
    this.h = Math.round(C.clamp(rows * 15 + 70, 200, 470));
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(this.h * dpr);
    this.canvas.style.height = `${this.h}px`;
    this.ctx = this.canvas.getContext('2d');
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.layout();
    this.draw();
  }
  // rows: every level spread over the height, children ordered under their parents (stable for the whole tree)
  layout() {
    const T = this.tree;
    const top = 26, bottom = 34;
    const H = (this.h || 300) - top - bottom;
    this.pos.clear();
    let prevY = new Map();
    let maxP = new Map();
    let eqB = 0.05;
    const seq = { lo: Infinity, hi: -Infinity };
    for (let d = 0; d <= T.depth; d++) {
      const nodes = T.nodesAt(d).filter(TreeStore.shown);
      nodes.sort((x, y) => {
        const px = prevY.get(x.parent) ?? 0.5, py = prevY.get(y.parent) ?? 0.5;
        return px - py || (x.source === 'overlay') - (y.source === 'overlay') || (y.p || 0) - (x.p || 0);
      });
      const yMap = new Map();
      nodes.forEach((n, i) => {
        const f = (i + 0.5) / nodes.length;
        yMap.set(n.key, f);
        this.pos.set(n.key, { f, n });
        const eq = C.simple(n.assets?.us_equity?.c?.[1]);
        if (C.isNum(eq)) eqB = Math.max(eqB, Math.abs(eq));
        const m = this.AG.colour;
        if (m && m.startsWith('s_')) { const v = n.rep?.[m]; if (C.isNum(v)) { seq.lo = Math.min(seq.lo, v); seq.hi = Math.max(seq.hi, v); } }
      });
      maxP.set(d, Math.max(1e-9, ...nodes.map(n => (n.source === 'overlay' ? 0 : n.p || 0))));
      prevY = yMap;
    }
    this.maxP = maxP;
    this.top = top; this.H = H;
    this.eqBound = Math.min(1, eqB);
    this.seqRange = seq.lo <= seq.hi ? [seq.lo, seq.hi] : [0, 1];
    this.seqMode = this.AG.colour;
    this.B.onLayout?.();
  }
  colX(d) {
    const left = 30, right = 24;
    const n = this.b - this.a + 1;
    const cw = (this.w - left - right) / Math.max(1, n);
    return left + (d - this.a + 0.5) * cw;
  }
  get colW() { return (this.w - 54) / Math.max(1, this.b - this.a + 1); }
  yOf(key) { const p = this.pos.get(key); return p ? this.top + p.f * this.H : null; }
  colour(n) {
    const m = this.AG.colour;
    if (m === 'none') return ACCENT;
    if (m === 'regime') return this.B.regimeOfNode(n).color;
    if (m === 'eq') {
      const v = C.simple(n.assets?.us_equity?.c?.[1]);
      if (!C.isNum(v)) return '#adb5bd';
      const { up, down } = this.B.env.ctx.charts.klineColors();
      return mix('#f1f3f5', v >= 0 ? up : down, Math.min(1, Math.abs(v) / (this.eqBound || 0.3)) * 0.92 + 0.08);
    }
    const v = n.rep?.[m];
    if (!C.isNum(v)) return '#adb5bd';
    const [lo, hi] = this.seqRange;
    return mix(SEQ_LO, SEQ_HI, hi > lo ? (v - lo) / (hi - lo) : 0.5);
  }
  draw() {
    const g = this.ctx;
    if (!g) return;
    const T = this.tree, AG = this.AG;
    g.clearRect(0, 0, this.w, this.h);
    this.hits = [];
    const sel = AG.node ? T.get(AG.node) : null;
    const onPath = new Set(sel ? T.pathKeys(sel.key) : []);
    const filt = !!(AG.filters?.actor || AG.filters?.event);
    const twoTier = this.b - this.a + 1 <= 10;
    const cw = this.colW;
    // quarter axis
    g.font = '11px -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
    g.fillStyle = INK3; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    const every = Math.max(1, Math.ceil(46 / cw));
    const gap = 50;
    let lastX = -Infinity;
    this._axisRight = -Infinity;
    const xEnd = this.colX(this.b);
    for (let d = this.a; d <= this.b; d++) {
      const x = this.colX(d);
      const isLast = d === this.b;
      if (!isLast && ((d - this.a) % every || x - lastX < gap || xEnd - x < gap)) continue;
      const q = T.nodesAt(d)[0]?.quarter || T.levelInfo?.[d]?.quarter || '';
      const lab = d === 0 ? `${q}末` : (cw < 52 ? q.replace(/^20(\d\d)Q/, '$1Q') : q);
      const align = isLast && d !== this.a ? 'right' : d === this.a ? 'left' : 'center';
      const tx = isLast && d !== this.a ? Math.min(this.w - 2, x + 6) : d === this.a ? Math.max(2, x - 6) : x;
      const wd = g.measureText(lab).width;
      const left = align === 'left' ? tx : align === 'right' ? tx - wd : tx - wd / 2;
      if (left < (this._axisRight ?? -Infinity) + 8 && d !== this.a) continue;   // overlapping the previous label
      g.textAlign = align;
      g.fillText(lab, tx, this.h - 10);
      this._axisRight = left + wd;
      lastX = x;
    }
    if (!T.loaded.size) return;
    // edges (two-tier: parent -> cluster diamond -> child)
    const diamonds = new Map();
    for (let d = Math.max(1, this.a); d <= this.b; d++) {
      for (const n of T.nodesAt(d)) {
        if (!TreeStore.shown(n)) continue;
        const p = n.parent ? T.get(n.parent) : null;
        if (!p || p.level < this.a) continue;
        const x0 = this.colX(p.level), y0 = this.yOf(p.key), x1 = this.colX(d), y1 = this.yOf(n.key);
        if (y0 === null || y1 === null) continue;
        const w = 0.6 + 3.2 * Math.sqrt((n.p || 0) / (this.maxP.get(d) || 1));
        const hl = onPath.has(n.key) && onPath.has(p.key);
        const dim = filt && !this.B.matches(n);
        g.globalAlpha = dim ? 0.12 : hl ? 1 : n.source === 'overlay' ? 0.9 : 0.45;
        g.strokeStyle = hl ? INK : n.source === 'overlay' ? ACCENT : '#9aa3b5';
        g.lineWidth = hl ? Math.max(2, w) : w;
        g.setLineDash(n.source === 'overlay' ? [4, 3] : []);
        if (twoTier && n.cl) {
          const dk = `${p.key}|${n.cl.id || n.cl.label}`;
          if (!diamonds.has(dk)) diamonds.set(dk, { p, x: (x0 + x1) / 2, ys: [], kids: [], cl: n.cl, hl: false, dim: true });
          const dm = diamonds.get(dk);
          dm.ys.push(y1); dm.kids.push(n); dm.hl = dm.hl || hl; dm.dim = dm.dim && dim;
        } else {
          g.beginPath(); g.moveTo(x0, y0);
          const mx = (x0 + x1) / 2;
          g.bezierCurveTo(mx, y0, mx, y1, x1, y1);
          g.stroke();
        }
      }
    }
    g.setLineDash([]);
    if (twoTier) {
      this.dmCount = new Map();
      for (const dm of diamonds.values()) this.dmCount.set(dm.p.level, (this.dmCount.get(dm.p.level) || 0) + 1);
      for (const dm of diamonds.values()) {
        const x0 = this.colX(dm.p.level), y0 = this.yOf(dm.p.key);
        const yd = dm.ys.reduce((s, y) => s + y, 0) / dm.ys.length;
        dm.y = yd;
        const head = Object.keys(dm.cl.head || {})[0];
        const pc = dm.cl.p;
        g.globalAlpha = dm.dim ? 0.12 : dm.hl ? 1 : 0.7;
        g.strokeStyle = head ? C.actorColor(head) : '#9aa3b5';
        g.lineWidth = 1 + 4 * Math.sqrt(C.isNum(pc) ? pc : 0.3);
        g.beginPath(); g.moveTo(x0, y0); g.bezierCurveTo((x0 + dm.x) / 2, y0, (x0 + dm.x) / 2, yd, dm.x, yd); g.stroke();
        for (const k of dm.kids) {
          const x1 = this.colX(k.level), y1 = this.yOf(k.key);
          const hl = dm.hl && onPath.has(k.key);
          g.globalAlpha = dm.dim ? 0.12 : hl ? 1 : 0.5;
          g.strokeStyle = hl ? INK : k.source === 'overlay' ? ACCENT : '#9aa3b5';
          g.lineWidth = hl ? 2 : 0.8 + 2.5 * Math.sqrt((k.pc || 0) / Math.max(1e-6, pc || 1));
          g.setLineDash(k.source === 'overlay' ? [4, 3] : []);
          g.beginPath(); g.moveTo(dm.x, yd); g.bezierCurveTo((dm.x + x1) / 2, yd, (dm.x + x1) / 2, y1, x1, y1); g.stroke();
        }
        g.setLineDash([]);
        // the diamond
        g.globalAlpha = dm.dim ? 0.2 : 1;
        g.fillStyle = head ? C.actorColor(head) : '#9aa3b5';
        g.beginPath(); g.moveTo(dm.x, yd - 5); g.lineTo(dm.x + 5, yd); g.lineTo(dm.x, yd + 5); g.lineTo(dm.x - 5, yd); g.closePath(); g.fill();
        this.hits.push({ x: dm.x, y: yd, r: 7, kind: 'cluster', dm });
        const perGap = this.dmCount?.get(dm.p.level) || 1;
        if (cw >= 120 && this.H / perGap >= 26) {
          const sibs = [...diamonds.values()].filter(o => o.p === dm.p).map(o => o.cl.label);
          g.fillStyle = INK3; g.textAlign = 'center'; g.font = '10px -apple-system, "PingFang SC", sans-serif';
          g.fillText(C.cut(C.diffLabel(dm.cl.label, sibs).replace(/^[^：]*：/, ''), Math.max(4, Math.floor(cw / 22))), dm.x, yd - 8);
        }
      }
    }
    g.globalAlpha = 1;
    // nodes
    for (let d = this.a; d <= this.b; d++) {
      const mp = this.maxP.get(d) || 1;
      for (const n of T.nodesAt(d)) {
        if (!TreeStore.shown(n)) continue;
        const x = this.colX(d), y = this.yOf(n.key);
        if (y === null) continue;
        const r = C.clamp(2 + 7 * Math.sqrt((n.p || 0) / mp), 2, Math.max(2.4, Math.min(9, cw / 2.4)));
        const dim = filt && !this.B.matches(n);
        g.globalAlpha = dim ? 0.15 : 1;
        g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
        g.fillStyle = n.source === 'overlay' ? '#fff' : this.colour(n);
        g.fill();
        const isSel = sel && sel.key === n.key;
        if (n.source === 'overlay') { g.setLineDash([3, 2]); g.strokeStyle = ACCENT; g.lineWidth = 1.6; g.stroke(); g.setLineDash([]); }
        else if (isSel || onPath.has(n.key)) { g.strokeStyle = INK; g.lineWidth = isSel ? 2.4 : 1.4; g.stroke(); }
        else if (filt && !dim) { g.strokeStyle = ACCENT; g.lineWidth = 1.6; g.stroke(); }
        else { g.strokeStyle = 'rgba(29,36,51,.25)'; g.lineWidth = 0.6; g.stroke(); }
        if (isSel) { g.beginPath(); g.arc(x, y, r + 4, 0, Math.PI * 2); g.strokeStyle = ACCENT; g.lineWidth = 1.5; g.stroke(); }
        const rows = T.nodesAt(d).length || 1;
        if (twoTier && cw >= 90 && this.H / rows >= 15 && (n.p || 0) >= 0.02 && !dim) {
          g.fillStyle = INK; g.textAlign = 'left'; g.font = '10px -apple-system, "PingFang SC", sans-serif';
          g.fillText(C.prob(n.p), x + r + 3, y + 3);
        }
        this.hits.push({ x, y, r: Math.max(r, 6), kind: 'node', n });
      }
    }
    // the selected node's children that are not in the snapshot (未生成)
    if (sel && sel.level < this.b + 1 && sel.level >= this.a) {
      const ghosts = T.ghosts(sel.key).slice(0, GHOSTS_SHOWN);
      this.ghostMore = Math.max(0, T.ghosts(sel.key).length - ghosts.length);
      const xg = this.colX(sel.level) + Math.min(cw * 0.55, 70);
      const y0 = this.yOf(sel.key);
      const kids = T.children(sel.key).map(k => this.yOf(k.key)).filter(C.isNum);
      let yy = Math.max(this.top + 6, (kids.length ? Math.max(...kids) : y0) + 14);
      const step = 15;
      const room = Math.floor((this.top + this.H - yy) / step);
      const list = ghosts.slice(0, Math.max(0, Math.min(ghosts.length, room > 0 ? room : 0)));
      if (list.length < ghosts.length && ghosts.length) yy = Math.max(this.top + 6, this.top + this.H - ghosts.length * step);
      g.globalAlpha = 1;
      for (const gh of ghosts.slice(0, Math.max(list.length, Math.min(ghosts.length, Math.floor(this.H / step))))) {
        g.strokeStyle = 'rgba(122,131,150,.7)'; g.lineWidth = 1; g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(this.colX(sel.level), y0); g.bezierCurveTo((this.colX(sel.level) + xg) / 2, y0, (this.colX(sel.level) + xg) / 2, yy, xg, yy); g.stroke();
        g.beginPath(); g.arc(xg, yy, 4.5, 0, Math.PI * 2); g.fillStyle = '#fff'; g.fill(); g.strokeStyle = INK3; g.lineWidth = 1.3; g.stroke();
        g.setLineDash([]);
        if (cw >= 110) { g.fillStyle = INK3; g.textAlign = 'left'; g.font = '10px -apple-system, "PingFang SC", sans-serif'; g.fillText(`未生成 ${C.prob(gh.pc)}`, xg + 7, yy + 3); }
        this.hits.push({ x: xg, y: yy, r: 6, kind: 'ghost', gh, parent: sel });
        yy += step;
        if (yy > this.top + this.H + 10) break;
      }
      if (this.ghostMore) {
        g.fillStyle = INK3; g.textAlign = 'left'; g.font = '10px -apple-system, "PingFang SC", sans-serif';
        g.fillText(`另有 ${this.ghostMore} 个未生成（见节点详情）`, xg - 4, Math.min(yy + 3, this.top + this.H + 18));
      }
    }
    g.globalAlpha = 1;
    if (this.hot) {
      g.beginPath(); g.arc(this.hot.x, this.hot.y, this.hot.r + 3, 0, Math.PI * 2);
      g.strokeStyle = ACCENT; g.lineWidth = 1.2; g.stroke();
    }
  }
  pick(e) {
    const rc = this.canvas.getBoundingClientRect();
    const x = e.clientX - rc.left, y = e.clientY - rc.top;
    let best = null, bd = Infinity;
    const rank = { node: 0, cluster: 1, ghost: 2 };       // a kept node under the pointer wins over a ghost circle
    for (const hh of this.hits) {
      const d = Math.hypot(hh.x - x, hh.y - y);
      if (d > hh.r + 4) continue;
      if (!best || rank[hh.kind] < rank[best.kind] || (rank[hh.kind] === rank[best.kind] && d < bd)) { bd = d; best = hh; }
    }
    return { best, x, y };
  }
  hover(e) {
    const { best, x, y } = this.pick(e);
    if (best !== this.hot) { this.hot = best; this.draw(); }
    if (!best) { this.tip.style.display = 'none'; return; }
    const B = this.B;
    const lines = [];
    if (best.kind === 'node') {
      const n = best.n;
      const eq = n.assets?.us_equity?.c?.[1];
      lines.push(h('b', {}, `${n.level === 0 ? `${n.quarter} 末（起点）` : n.quarter} · 路径概率 ${C.prob(n.p)}`));
      if (n.level) lines.push(h('div', {}, `条件概率 ${C.prob(n.pc)}${n.source === 'overlay' ? ' · 即时生成' : ''}${n.stress ? ' · 压力季度' : ''}`));
      if (n.cl?.label) lines.push(h('div', { class: 'm' }, C.cut(C.humanize(n.cl.label), 60)));
      if (n.br) lines.push(h('div', {}, `后果：${n.br.label || C.BRANCH_KIND[n.br.kind] || ''}${n.br.axis ? `（${C.axisName(n.br.axis) || C.fieldName(n.br.axis)}）` : ''}`));
      lines.push(h('div', {}, `美股累计 ${C.retText(eq)} · 利率 ${C.pp(n.rep?.s_ff)} · 失业 ${C.pp(n.rep?.s_unrate)} · 核心 ${C.pp(n.rep?.s_core_yoy)}`));
      lines.push(h('div', { class: 'm' }, `体制：${B.regimeOfNode(n).name}`));
    } else if (best.kind === 'cluster') {
      const dm = best.dm;
      lines.push(h('b', {}, `动作簇 · 概率 ${C.prob(dm.cl.p)}`));
      for (const [a, o] of C.splitLabel(C.humanize(dm.cl.label)).slice(0, 6)) lines.push(h('div', {}, `${a}：${o}`));
      lines.push(h('div', { class: 'm' }, `其后 ${dm.kids.length} 个后果分支在快照里`));
    } else {
      const gh = best.gh;
      lines.push(h('b', {}, `未生成 · 条件概率 ${C.prob(gh.pc)}`));
      if (gh.label) lines.push(h('div', { class: 'm' }, C.cut(C.humanize(gh.label), 60)));
      if (gh.branch) lines.push(h('div', {}, `后果：${gh.branch}`));
      lines.push(h('div', {}, `剪掉的原因：${C.cut(C.humanize(gh.reason || '前向选择未保留'), 50)}`));
      lines.push(h('div', {}, needsConfirm(B.env) ? '点击：先确认用量，再即时生成它的下一季' : '点击：即时生成它的下一季（快照分支不需要确认）'));
    }
    C.fill(this.tip, ...lines);
    this.tip.style.display = 'block';
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    let left = x + 14, top = y + 12;
    if (left + tw > this.w - 4) left = Math.max(4, x - tw - 14);
    if (top + th > this.h) top = Math.max(0, y - th - 10);
    this.tip.style.left = `${left}px`; this.tip.style.top = `${top}px`;
  }
  click(e) {
    const { best } = this.pick(e);
    if (!best) return;
    if (best.kind === 'node') this.B.select(best.n.key, { scroll: false });
    else if (best.kind === 'cluster') this.B.select(best.dm.kids[0].key, { scroll: false });
    else if (best.kind === 'ghost') {
      this.B.select(best.parent.key, { scroll: true });
      generate(this.B.env, best.gh.key, { quarters: 1, parentLevel: best.parent.level, onDone: (k) => this.B.select(k, { scroll: true }) });
    }
  }
}

function mix(a, b, t) {
  const pa = rgb(a), pb = rgb(b);
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * C.clamp(t, 0, 1)));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
function rgb(hex) {
  const s = String(hex);
  if (s.startsWith('rgb')) return s.match(/\d+/g).slice(0, 3).map(Number);
  const n = parseInt(s.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// ================================================================ on-demand generation (未生成 → 即时生成)
// A snapshot's branch: no confirmation dialog (the user's decision); a small notice says how many calls and how long,
// then a progress line. A saved run's branch with the real Claude Code spends usage on a run, so it asks first
// (inline, with the server's estimate), like every other action that spends usage on a run.
const GEN = new Map();          // `${source}:${key}` -> promise (one generation per node at a time)
const GHOSTS_SHOWN = 12;
export const needsConfirm = (env) => env.source?.kind === 'run' && !env.mock;
export function expectedCalls(env, levelOfParent, quarters = 1) {
  const st = (env.tree?.summary?.manifest || env.tree?.summary || {}).settings || {};
  const synthDepth = st.synth?.mode === 'off' ? 0 : (st.synth?.depth ?? 4);
  let nodes = 1, calls = 0;
  for (let i = 0; i < quarters; i++) {
    const level = levelOfParent + 1 + i;
    calls += nodes * (1 + (level < synthDepth ? 1 : 0));
    nodes = Math.min(4, nodes * 3);
  }
  return calls;
}
export function genNotice(env, calls) {
  const sec = env.llm?.latency?.p50_s;
  const lo = calls * (sec || 20), hi = calls * (env.llm?.latency?.p90_s || 60);
  return env.mock ? `约 ${calls} 次大模型调用（模拟后端：不调用模型，几秒内完成）`
    : `约 ${calls} 次大模型调用，约 ${C.seconds(lo)}–${C.seconds(hi)}（每次 ${sec ? '实测' : '假设'} ${Math.round(sec || 20)}–${Math.round(env.llm?.latency?.p90_s || 60)} 秒），计入订阅用量`;
}

export function generate(env, key, { quarters = 1, onDone, onStatus, box, parentLevel = null, confirmed = false } = {}) {
  const src = env.source;
  const id = `${src.skey}:${key}`;
  if (GEN.has(id)) return GEN.get(id);
  const tree = env.tree;
  const node = tree.get(key);
  // the node being expanded sits at its own level (a 未生成 ghost: its parent's level + 1)
  const level = node ? node.level : (C.isNum(parentLevel) ? parentLevel + 1 : 0);
  const calls = expectedCalls(env, level - 1, quarters);
  if (needsConfirm(env) && !confirmed) {
    const el = box || env.browser?.genBox;
    const ask = (resolve, reject) => {
      if (!el || !el.isConnected) {
        if (window.confirm(`在已保存的推演里生成这一支会调用本机 Claude Code，约 ${calls} 次（计入订阅额度）。确认生成？`)) resolve(); else reject(new Error('cancelled'));
        return;
      }
      const ok = h('input', { type: 'checkbox' });
      const go = h('button', { class: 'btn primary', type: 'button', disabled: true }, '确认生成');
      ok.onchange = () => { go.disabled = !ok.checked; };
      go.onclick = () => resolve();
      el.className = 'agents-gen warn';
      el.style.display = '';
      C.fill(el, h('div', {}, h('b', {}, `在已保存的推演里生成这一支：${genNotice(env, calls)}。`),
        h('label', { class: 'inl', style: { margin: '6px 0' } }, ok, '我确认：这会计入我的 Claude 订阅用量'),
        h('span', { class: 'agents-rowbtns' }, go, h('button', { class: 'btn', type: 'button', onclick: () => { C.fill(el); el.style.display = 'none'; reject(new Error('cancelled')); } }, '取消'))));
    };
    const p = new Promise(ask).then(() => generate(env, key, { quarters, onDone, onStatus, box, parentLevel, confirmed: true }));
    p.catch(() => {});
    return p;
  }
  const t0 = Date.now();
  const say = (text, kind = '') => {
    onStatus?.(text, kind);
    if (env.browser) env.browser.lastGen = { text, kind, at: Date.now() };
    const el = box || env.browser?.genBox;
    if (el && el.isConnected) {
      el.className = `agents-gen ${kind}`;
      C.fill(el, kind === '' ? h('span', { class: 'spinner', style: { width: '14px', height: '14px' } }) : null, text);
      el.style.display = '';
    }
  };
  const p = (async () => {
    say(`即时生成：${genNotice(env, calls)}。`);
    let r;
    try {
      r = await src.expand(key, quarters, undefined, confirmed);
    } catch (err) {
      say(`没有生成：${D.errorText(err)}`, 'err');
      throw err;
    }
    let job = r.job;
    if (r.status === 'done' || job?.status === 'done') {
      // already in the overlay
    } else {
      for (;;) {
        await new Promise(res => setTimeout(res, 700));
        if (env.signal?.aborted && !box) break;
        try { job = await src.job(job.id); } catch (err) { say(`查询生成进度失败：${D.errorText(err)}`, 'err'); throw err; }
        const el = (Date.now() - t0) / 1000;
        if (job.status === 'queued' || job.status === 'running') { say(`即时生成中（${job.status === 'queued' ? '排队' : '运行'} ${C.seconds(el)}）：${genNotice(env, calls)}。`); continue; }
        break;
      }
      if (job.status === 'paused') {
        const reset = job.reset_at ? new Date(job.reset_at * 1000).toLocaleString('zh-CN', { hour12: false }) : null;
        say(`暂停：${job.reason === 'usage_limit' ? `达到 Claude 订阅的用量上限${reset ? `，预计 ${reset} 恢复` : ''}` : job.error || job.reason}。之后再点一次即可继续。`, 'warn');
        throw new Error(job.error || 'paused');
      }
      if (job.status === 'failed') { say(`生成失败：${job.error || '未知错误'}`, 'err'); throw new Error(job.error || 'failed'); }
    }
    src.forget(key);
    if (node?.parent) src.forget(node.parent);
    await tree.loadOverlay({ fresh: true });
    env.browser?.refreshMeta?.();
    say(`已即时生成并存入覆盖层（${C.seconds((Date.now() - t0) / 1000)}）。新的分支标为“即时生成”。`, 'ok');
    onDone?.(key);
    return key;
  })().finally(() => GEN.delete(id));
  GEN.set(id, p);
  p.catch(() => {});
  return p;
}

// ================================================================ node detail
const DETAIL_TABS = [
  { value: 'overview', label: '概览' }, { value: 'actions', label: '动作' }, { value: 'conseq', label: '后果' },
  { value: 'accounts', label: '账户' }, { value: 'packets', label: '观察包' }, { value: 'inputs', label: '输入字段' },
  { value: 'assets', label: '资产' }, { value: 'news', label: '新闻' }, { value: 'calls', label: '调用' },
];

async function drawDetail(B, key, { scroll = false } = {}) {
  const { env, tree, src, detailBox } = B;
  const { ui } = env.ctx;
  key = key || tree.root;
  if (!key) { C.fill(detailBox); return; }
  const token = (B.detailToken = (B.detailToken || 0) + 1);
  C.fill(detailBox, ui.card({ title: '节点详情' }, ui.loading('读取节点…')));
  let cur, par = null;
  try {
    cur = await src.node(key, { signal: env.signal });
    if (cur?.record?.parent) par = await src.node(cur.record.parent, { signal: env.signal }).catch(() => null);
  } catch (err) {
    if (err.name === 'AbortError' || token !== B.detailToken) return;
    C.fill(detailBox, ui.card({ title: '节点详情' }, ui.notice(`读取节点失败：${D.errorText(err)}`, 'error')));
    return;
  }
  if (token !== B.detailToken || !env.alive()) return;
  const rec = cur.record, det = cur.detail;
  if (!tree.get(rec.key)) tree.add(D.skeleton(rec, rec.source === 'overlay' || rec.status === 'pruned' ? 'overlay' : 'snapshot'));
  const sk = tree.get(rec.key);
  const pdet = par?.detail || null;
  const AG = env.AG;
  if (!AG.detailTab) AG.detailTab = 'overview';
  const body = h('div');
  const genBox = h('div', { class: 'agents-gen', style: { display: 'none' }, role: 'status', 'aria-live': 'polite' });
  B.genBox = genBox;
  if (B.lastGen && Date.now() - B.lastGen.at < 20000) {
    genBox.className = `agents-gen ${B.lastGen.kind}`;
    genBox.style.display = '';
    C.fill(genBox, B.lastGen.kind === '' ? h('span', { class: 'spinner', style: { width: '14px', height: '14px' } }) : null, B.lastGen.text);
  }
  const tabs = h('div', { class: 'agents-tabs' }, ui.seg(DETAIL_TABS, AG.detailTab, (v) => { AG.detailTab = v; paint(); }));
  const ctx = { B, rec, det, par, pdet, sk, body };
  const paint = () => {
    env.ctx.charts.disposeWithin(body);
    C.fill(body);
    const fn = { overview: tabOverview, actions: tabActions, conseq: tabConseq, accounts: tabAccounts, packets: tabPackets,
      inputs: tabInputs, assets: tabAssets, news: tabNews, calls: tabCalls }[AG.detailTab] || tabOverview;
    try { fn(ctx); } catch (err) { console.error(err); C.put(body, ui.notice(`这一栏显示出错：${err.message}`, 'error')); }
  };
  const crumb = h('div', { class: 'agents-crumb' });
  const pathKeys = cur.path || tree.pathKeys(rec.key);
  const drawCrumb = (all) => {
    C.fill(crumb);
    const fold = !all && pathKeys.length > 12;
    pathKeys.forEach((k, i) => {
      if (fold && i >= 3 && i < pathKeys.length - 5) {
        if (i === 3) C.put(crumb, h('span', {}, '›'), h('button', { type: 'button', title: '展开整条路径', onclick: () => drawCrumb(true) }, `…（${pathKeys.length - 8} 季）`));
        return;
      }
      const n = tree.get(k);
      if (i) C.put(crumb, h('span', {}, '›'));
      C.put(crumb, k === rec.key ? h('b', {}, i === 0 ? '起点' : (n?.quarter || rec.quarter)) :
        h('button', { type: 'button', onclick: () => B.select(k, { scroll: false }) }, i === 0 ? '起点' : (n?.quarter || `第 ${i} 季`)));
    });
  };
  drawCrumb(false);
  const statusChip = rec.status === 'pruned' ? (sk?.expandedOnDemand ? C.chip('即时生成（原为未生成）', 'accent') : C.chip('未生成（被剪掉）', ''))
    : rec.source === 'overlay' ? C.chip('即时生成', 'accent') : rec.status === 'leaf' ? C.chip('末端', '') : C.chip(rec.level === 0 ? '起点' : '已展开', 'good');
  const rg = rec.level ? B.regimeOfNode(sk || D.skeleton(rec)) : null;
  const flagsZh = { prior_only: '只用历史基率', synth_default: '后果未经大模型判断', intervened: '人工干预', recombined: '重组合并', reassigned: '吸收了剪掉的质量', accounts_missing: '缺账户', stub_model: '桩模型' };
  const head = h('div', {},
    crumb,
    h('div', { class: 'agents-chips' }, statusChip, rg ? C.regimeBadge(rg) : null, rec.stress ? C.chip('压力季度', 'warn') : null,
      ...(rec.flags || []).map(f => C.chip(flagsZh[f] || '标记', f === 'synth_default' || f === 'prior_only' ? 'warn' : '')),
      rec.overlay?.replayed ? (rec.overlay.replay_exact === false
        ? C.chip('近似重放', 'warn', '粒子云由父节点保存的粒子云重放一个季度得到；保存精度（float32）让个别粒子落到分区边界另一侧，这里取离散状态相同的格子')
        : C.chip('由父节点重放', '', '粒子云由父节点保存的粒子云（构建时并入了被剪节点、重抽样到 N）重放一个季度得到')) : null),
    h('div', { class: 'agents-actions', style: { marginTop: '8px' } },
      h('button', { class: 'btn primary', type: 'button', onclick: () => env.go('story', { node: rec.key, page: rec.level }) }, '在连环画里播放'),
      h('button', { class: 'btn', type: 'button', onclick: () => B.pathBox.scrollIntoView({ behavior: 'smooth', block: 'start' }) }, '整条路径'),
      rec.parent ? h('button', { class: 'btn', type: 'button', onclick: () => B.select(rec.parent, { scroll: false }) }, '← 父节点') : null),
    genBox);
  const title = rec.level === 0 ? `起点 · ${rec.quarter} 末` : `${rec.quarter} · 路径概率 ${C.prob(sk?.p ?? rec.p_path)}`;
  C.fill(detailBox, ui.card({ title: `节点详情：${title}`,
    sub: '这一季的动作、后果、账户和模型输入都来自快照（或即时生成的覆盖层），这里只读取、不重新计算。' }, head, tabs, body));
  paint();
  drawPath(B, rec.key);
  if (scroll) detailBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function kidsOf(B, rec) {
  const { tree } = B;
  const list = [...(rec.children || []), ...(rec.overlay_children || [])];
  const seen = new Set();
  const items = [];
  for (const c of list) {
    if (seen.has(c.key)) continue;
    seen.add(c.key);
    const sk = tree.get(c.key);
    const ghost = c.status === 'pruned' && c.overlay !== 'expanded' && !(sk && sk.expandedOnDemand);
    const overlay = !ghost && (c.overlay === 'expanded' || c.overlay === 'child' || sk?.source === 'overlay' || (rec.overlay_children || []).some(o => o.key === c.key));
    items.push({ key: c.key, label: c.label, branch: c.branch, kind: c.kind, pc: c.p_cond, p: sk?.p, status: ghost ? 'ghost' : overlay ? 'overlay' : 'kept',
      reason: c.reason, into: c.absorbed_into, sk });
  }
  // group by cluster label (a cluster = a joint action), clusters by total conditional probability
  const groups = new Map();
  for (const it of items) {
    const k = String(it.label || '').split('｜')[0] || '（无动作簇）';     // the action cluster (cells of a split share it)
    if (!groups.has(k)) groups.set(k, { label: k, items: [], p: 0, head: it.sk?.cl?.head || null });
    const g = groups.get(k);
    g.items.push(it); g.p += it.pc || 0;
    if (!g.head && it.sk?.cl?.head) g.head = it.sk.cl.head;
  }
  const out = [...groups.values()].sort((a, b) => b.p - a.p);
  for (const g of out) g.items.sort((a, b) => (b.pc || 0) - (a.pc || 0));
  return { groups: out };
}

function fanView(B, rec, { onPick } = {}) {
  const { env } = B;
  const { groups } = kidsOf(B, rec);
  if (!groups.length) return null;
  const labels = groups.map(g => g.label);
  return h('div', { class: 'agents-fan' }, groups.map(g => {
    const named = C.splitLabel(g.label)[0]?.[0];
    const head = g.head ? Object.keys(g.head)[0] : C.actorIds().find(a => C.actorName(a) === named) || null;
    const dia = h('span', { class: 'agents-diamond' });
    dia.style.background = head ? C.actorColor(head) : '#9aa3b5';
    const row = (it) => {
        const btn = it.status === 'ghost'
          ? h('button', { class: 'btn', type: 'button', title: needsConfirm(env) ? '即时生成（先确认用量）' : '即时生成（快照分支不需要确认）', onclick: (e) => {
            e.target.disabled = true;
            generate(env, it.key, { quarters: B.genQuarters || 1, parentLevel: rec.level, onDone: (k) => (onPick ? onPick(k) : B.select(k, { scroll: false })) })
              .catch(() => { e.target.disabled = false; });
          } }, '即时生成')
          : h('button', { class: 'btn', type: 'button', onclick: () => (onPick ? onPick(it.key) : B.select(it.key, { scroll: false })) }, '查看');
        const split = it.sk?.cl?.split || (it.label && it.label.includes('｜') ? [it.label.split('｜')[1]] : null);
        return h('div', { class: `agents-kid ${it.status}` }, h('span', { class: 'dotk' }),
          h('span', {}, it.status === 'ghost' ? h('span', { class: 'muted' }, '未生成 · ') : it.status === 'overlay' ? C.chip('即时生成', 'accent') : null,
            ' ', it.branch || C.BRANCH_KIND[it.kind] || '后果', h('span', { class: 'muted' }, it.kind ? `（${C.BRANCH_KIND[it.kind] || ''}）` : ''),
            split ? h('span', { class: 'muted', style: { fontSize: '11px' } }, ` · ${C.cut(C.humanize(Array.isArray(split) ? split.join('；') : split), 30)}`) : null),
          h('span', { class: 'pc muted', title: '条件概率' }, C.prob(it.pc)), btn);
    };
    // a cluster's children (consequence branches × discrete cells): the six most probable, the rest folded
    const shown = g.items.slice(0, 6), rest = g.items.slice(6);
    return h('div', { class: 'agents-cluster' },
      h('div', { class: 'agents-cluster-h' }, dia, h('b', {}, C.humanize(C.diffLabel(g.label, labels))), h('span', { class: 'muted' }, `动作簇概率 ${C.prob(g.p)}`)),
      shown.map(row),
      rest.length ? h('details', {}, h('summary', { class: 'muted', style: { fontSize: '12px', cursor: 'pointer' } },
        `其余 ${rest.length} 个分支（合计 ${C.prob(rest.reduce((t, it) => t + (it.pc || 0), 0))}）`), rest.map(row)) : null);
  }));
}

function tabOverview(x) {
  const { B, rec, det, sk, body } = x;
  const { env } = B;
  const { ui } = env.ctx;
  const s = rec.summary || {};
  const rep = s.rep || {};
  const eq = s.assets?.us_equity?.cum?.[1];
  C.put(body, h('div', { class: 'grid cols-4', style: { marginBottom: '12px' } },
    ui.kpi('路径概率', C.prob(sk?.p ?? rec.p_path), rec.level ? `条件 ${C.prob(rec.p_cond)} · 自身 ${C.prob(rec.own_mass)}` : '起点'),
    ui.kpi('美股累计（中位数）', C.retText(eq), `5–95%：${C.retText(s.assets?.us_equity?.cum?.[0])} ~ ${C.retText(s.assets?.us_equity?.cum?.[2])}`, C.retCls(eq)),
    ui.kpi('政策利率 · 失业率', `${C.pp(rep.s_ff)} · ${C.pp(rep.s_unrate)}`, `核心 CPI 同比 ${C.pp(rep.s_core_yoy)}`),
    ui.kpi('10 年期 · 信用利差', `${C.pp(rep.s_y10)} · ${C.num(rep.s_spread, 2)}`, `期限利差 ${C.num(rep.s_term, 2)}`)));
  if (rec.memory_line) C.put(body, h('p', { class: 'agents-why', style: { fontSize: '13px' } }, C.humanize(rec.memory_line)));
  if (rec.status === 'pruned' && !sk?.expandedOnDemand) {
    C.put(body, ui.notice(`这个分支没有进入快照：${C.humanize(rec.reason || '前向选择未保留')}。它的质量与粒子并入了保留的节点。`, 'info'));
  }
  if ((rec.absorbed || []).length) {
    C.put(body, h('details', { style: { margin: '8px 0' } }, h('summary', {}, `吸收了 ${rec.absorbed.length} 个被剪掉的分支（合计 ${C.prob(rec.absorbed.reduce((t, a) => t + (a.p || 0), 0))}）`),
      ui.table([{ key: 'label', label: '动作簇', fmt: (v) => C.cut(C.humanize(v), 40), wrap: true }, { key: 'branch', label: '后果' },
        { key: 'p', label: '质量', fmt: C.prob }, { key: 'distance', label: '距离', fmt: (v) => C.num(v, 2) },
        { key: 'how', label: '方式', fmt: (v) => ({ reassigned: '并入（前向选择）', recombined: '重合合并' }[v] || '并入') }], rec.absorbed, { stack: true })));
  }
  C.put(body, h('div', { class: 'section-title', style: { marginTop: '14px' } }, '下一季的分叉：动作簇 ◇ → 后果分支 ●'));
  const fan = fanView(B, rec);
  const qsel = h('select', { 'aria-label': '即时生成的深度', onchange: (e) => { B.genQuarters = Number(e.target.value); } },
    [1, 2, 4, 8].map(q => h('option', { value: q, selected: (B.genQuarters || 1) === q ? true : null }, `${q} 季`)));
  const unexpanded = !det;
  if (fan) C.put(body, fan);
  if (unexpanded) {
    const calls = expectedCalls(env, rec.level, B.genQuarters || 1);
    const go = h('button', { class: 'btn primary', type: 'button' }, '即时生成下一季');
    go.onclick = () => { go.disabled = true; generate(env, rec.key, { quarters: B.genQuarters || 1, onDone: (k) => B.select(k, { scroll: false }) }).catch(() => { go.disabled = false; }); };
    C.put(body, h('div', { class: 'agents-gen', style: { borderStyle: 'dashed' } },
      rec.status === 'leaf' ? '这是末端：' : '这个节点没有展开：', needsConfirm(env) ? '点击后先确认用量，再即时生成，结果存入覆盖层。' : '点击即时生成（快照分支不需要确认），结果存入覆盖层。',
      h('span', { class: 'muted' }, genNotice(env, calls)), h('label', { class: 'inl' }, '深度', qsel), go));
  } else if (fan) {
    C.put(body, h('div', { class: 'agents-legend' }, h('span', {}, needsConfirm(env) ? '未生成的分支点“即时生成”：先确认用量，结果存入覆盖层；生成深度'
      : '未生成的分支点“即时生成”：快照分支不需要确认，结果存入覆盖层；生成深度'), qsel));
  }
}

function actorBlock(B, a, { chosen, signal, title, leaderChoice = null } = {}) {
  const { ui } = B.env.ctx;
  const color = C.actorColor(a.actor);
  const blk = h('div', { class: 'agents-actorblock' });
  blk.style.setProperty('--c', color);
  const em = emblem(a.actor, { size: 34, tone: chosen ? C.toneOf(chosen, a.options?.find(o => o.id === chosen)?.status_quo) : 'neutral' });
  const maxP = Math.max(0.01, ...(a.options || []).map(o => o.p_used || 0));
  C.put(blk, h('div', { class: 'agents-actorhead' }, em, h('b', {}, a.zh || C.actorName(a.actor)),
    a.branching ? C.chip('分叉动作', 'accent', '这一季在它的选项上分叉') : C.chip('不分叉', '', '它的选项在粒子里混合'),
    a.follower_of ? C.chip(`跟随${C.actorName(typeof a.follower_of === 'string' ? a.follower_of : a.follower_of[0])}`, '', '它的选项按领先者的选择给出条件概率（反应函数）') : null,
    a.fallback ? C.chip('回退：历史基率', 'warn') : null,
    C.isNum(a.c_used ?? a.c) ? C.chip(`大模型权重 ${C.num(a.c_used ?? a.c, 2)}（判断值）`, '') : null,
    title ? h('span', { class: 'muted', style: { fontSize: '12px' } }, title) : null));
  const lead = typeof a.follower_of === 'string' ? a.follower_of : null;
  const lo = lead && leaderChoice ? leaderChoice : null;
  const row = lo && a.cond?.[lo] ? a.cond[lo] : null;
  const rows = (a.options || []).map(o => (row ? { ...o, p_marg: o.p_used, p_used: row[o.id] ?? 0 } : o))
    .sort((x, y) => (y.p_used || 0) - (x.p_used || 0));
  const maxR = Math.max(0.01, ...rows.map(o => o.p_used || 0));
  const tbl = ui.table([
    { key: 'label', label: '选项', wrap: true, fmt: (v, o) => h('span', {}, o.id === chosen ? '▶ ' : '', C.humanize(v), o.status_quo ? ' ' : '', o.status_quo ? C.chip('现状', '') : null, o.custom ? C.chip('自定义', 'accent') : null) },
    { key: 'p_used', label: row ? `在${C.actorName(lead)}选“${C.optionLabel(lead, lo)}”时` : '采用概率', fmt: (v) => h('span', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center', minWidth: '110px' } }, C.probBar(v, { max: row ? maxR : maxP }), C.prob(v)) },
    ...(row ? [{ key: 'p_marg', label: '采用概率（不分情形）', fmt: (v) => C.prob(v) }] : []),
    { key: 'p_llm', label: '大模型判断', fmt: (v) => C.prob(v) },
    { key: 'p_prior', label: '历史基率', fmt: (v, o) => { const t = C.prob(v); const far = C.isNum(v) && C.isNum(o.p_llm) && Math.abs(v - o.p_llm) > 0.3; return far ? h('span', {}, t, ' ', C.chip('与大模型相差 >30 个百分点', 'warn')) : t; } },
    { key: 'writes', label: '写入', wrap: true, fmt: (v) => (v ? C.humanize(v) : h('span', { class: 'muted' }, '不写字段')) },
  ], rows, { stack: true, rowClass: (o) => (o.id === chosen ? 'agents-chosen' : '') });
  C.put(blk, tbl);
  const ch = rows.find(o => o.id === chosen);
  if (ch?.rationale) C.put(blk, h('div', { class: 'agents-why', style: { marginTop: '6px' } }, `理由：${C.humanize(ch.rationale)}`));
  if (a.prior?.label) C.put(blk, h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '4px' } }, `历史基率来源：${C.sourceLabel(C.humanize(a.prior.label))}`));
  if ((a.attributes || []).length) {
    C.put(blk, h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '4px' } },
      a.attributes.map(d => `${d.zh.replace(/（.*$/, '')}：${(d.options || []).filter(o => (o.p || 0) >= 0.05).map(o => `${o.label} ${C.prob(o.p)}`).join(' · ')}`).join('；')));
  }
  const leader = typeof a.follower_of === 'string' ? a.follower_of : Array.isArray(a.follower_of) ? a.follower_of[0] : null;
  if (leader && a.cond && typeof a.cond === 'object' && Object.keys(a.cond).length) {
    const optLabel = (id) => (a.options || []).find(o => o.id === id)?.label || C.optionLabel(a.actor, id);
    const cols = [...new Set(Object.values(a.cond).flatMap(r => Object.keys(r || {})))];
    const rows = Object.entries(a.cond).map(([lo, r]) => ({ lo, ...r }));
    C.put(blk, h('details', { style: { marginTop: '6px' } }, h('summary', { style: { fontSize: '12px', cursor: 'pointer' } }, `反应函数：按${C.actorName(leader)}的选择给出的条件概率`),
      ui.table([{ key: 'lo', label: `${C.actorName(leader)}选`, fmt: (v) => C.optionLabel(leader, v) },
        ...cols.map(c => ({ key: c, label: optLabel(c), fmt: (v) => C.prob(v) }))], rows, { stack: true })));
  }
  if (signal?.text) C.put(blk, h('div', { class: 'agents-signal' }, `公开表态：“${C.humanize(signal.text)}”`, signal.commit ? ' ' : '', signal.commit ? C.chip('承诺', 'accent') : null));
  return blk;
}

function tabActions(x) {
  const { B, rec, det, pdet, body } = x;
  const { ui } = B.env.ctx;
  const realized = rec.cluster?.realized || {};
  const sigs = new Map((rec.signals || []).map(sg => [sg.actor, sg]));
  if (rec.level > 0) {
    C.put(body, h('div', { class: 'section-title', style: { marginTop: '4px' } }, `进入本节点的动作（${rec.quarter}）`));
    if (rec.cluster?.label) C.put(body, h('p', { class: 'agents-why' }, `动作簇（概率 ${C.prob(rec.cluster.p)}）：${C.humanize(rec.cluster.label)}。下方“▶”是本分支里每个行为者最常见的选项。`));
    if (pdet?.next?.length) {
      const order = pdet.next.slice().sort((a, b) => C.actorStage(a.actor) - C.actorStage(b.actor));
      for (const a of order) {
        const lead = typeof a.follower_of === 'string' ? a.follower_of : null;
        C.put(body, actorBlock(B, a, { chosen: realized[a.actor] ?? rec.cluster?.head?.[a.actor], signal: sigs.get(a.actor),
          leaderChoice: lead ? (realized[lead] ?? rec.cluster?.head?.[lead]) : null }));
      }
    } else C.put(body, ui.notice('父节点没有保存决策细节。', 'info'));
  }
  C.put(body, h('div', { class: 'section-title' }, `各行为者对下一季（${det?.quarter_next || '—'}）的选项：决定子节点`));
  if (det?.next?.length) {
    const order = det.next.slice().sort((a, b) => C.actorStage(a.actor) - C.actorStage(b.actor));
    for (const a of order) C.put(body, actorBlock(B, a, {}));
    if (B.src.kind === 'run') C.put(body, interveneForm(B, rec, det));
  } else C.put(body, ui.notice('这个节点没有展开：没有下一季的决策。可在“概览”里即时生成。', 'info'));
}

// 假如……: force one actor's choice at this node; the result is a new child run (the original stays as it is)
function interveneForm(B, rec, det) {
  const { env } = B;
  const actors = (det.next || []).filter(a => (a.options || []).length > 1);
  if (!actors.length) return null;
  const aSel = h('select', { 'aria-label': '行为者' }, actors.map(a => h('option', { value: a.actor }, a.zh || C.actorName(a.actor))));
  const oSel = h('select', { 'aria-label': '选项' });
  const fillOpts = () => {
    const a = actors.find(x => x.actor === aSel.value) || actors[0];
    C.fill(oSel, (a.options || []).map(o => h('option', { value: o.id }, `${o.label}（原采用 ${C.prob(o.p_used)}）`)));
  };
  aSel.onchange = fillOpts;
  fillOpts();
  const pIn = h('input', { type: 'number', min: 1, max: 100, value: 100, style: { width: '70px' }, 'aria-label': '指定概率（%）' });
  const msg = h('span', { class: 'agents-msg' });
  const body = () => ({ node: rec.key, actor: aSel.value, option: oSel.value, prob: C.clamp(Number(pIn.value) || 100, 1, 100) / 100 });
  const dry = h('button', { class: 'btn', type: 'button' }, '预估');
  const go = h('button', { class: 'btn primary', type: 'button' }, env.mock ? '开始（模拟后端）' : '开始…');
  dry.onclick = async () => {
    msg.className = 'agents-msg'; msg.textContent = '估算中…';
    try {
      const r = await D.post(`agents/runs/${encodeURIComponent(B.src.id)}/intervene?dry=1`, body());
      msg.textContent = `子推演约 ${r.estimate?.calls_new ?? '—'} 次调用（相同的前缀命中缓存），${C.minutesRange(r.estimate?.minutes)}`;
    } catch (err) { msg.className = 'agents-msg err'; msg.textContent = D.errorText(err); }
  };
  const start = async (confirm) => {
    msg.className = 'agents-msg'; msg.textContent = '提交中…';
    try {
      const r = await D.post(`agents/runs/${encodeURIComponent(B.src.id)}/intervene`, { ...body(), confirm });
      env.AG.activeRun = r.run_id;
      await env.refreshLists();
      env.go('setup');
    } catch (err) {
      if (err.status === 409 && err.body?.needs_confirm) {
        const e = err.body.estimate || {};
        if (window.confirm(`这个“假如”会新开一次推演，调用本机 Claude Code 约 ${e.calls_new ?? '—'} 次（计入订阅用量）。确认开始？`)) start(true);
        else { msg.textContent = '没有开始。'; }
        return;
      }
      msg.className = 'agents-msg err'; msg.textContent = D.errorText(err);
    }
  };
  go.onclick = () => start(false);
  return h('div', { class: 'agents-cluster', style: { marginTop: '12px' } },
    h('div', { class: 'agents-cluster-h' }, h('b', {}, `假如……在 ${det.quarter_next || '下一季'}`),
      h('span', { class: 'muted' }, '指定一个行为者的选择，从这里向后重新推演；结果是一个新的子推演，原推演不变，标为“人工干预”。')),
    h('div', { class: 'row' }, aSel, '选择', oSel, h('label', { class: 'inl' }, '概率', pIn, '%'), dry, go), msg);
}

function branchRows(ui, branches, mine) {
  return ui.table([
    { key: 'label', label: '后果', wrap: true, fmt: (v, b) => h('span', {}, b.id === mine ? '▶ ' : '', v || C.BRANCH_KIND[b.kind] || '—',
      b.beyond_history ? ' ' : '', b.beyond_history ? C.chip('超出可测历史', 'warn') : null, b.unintended ? ' ' : '', b.unintended ? C.chip('非预期', '') : null) },
    { key: 'kind', label: '类型', fmt: (v, b) => `${C.BRANCH_KIND[v] || '—'}${b.axis ? ` · ${C.axisName(b.axis) || C.fieldName(b.axis)}` : ''}${b.axis === 'd_gz' ? '（情景假说：金融条件）' : ''}` },
    { key: 'region', label: '区间（所分字段的标准差）', fmt: (v, b) => (v ? `${b.axis ? `${C.fieldName(b.axis)} ` : ''}${C.isNum(v.lo_z) ? `${C.num(v.lo_z, 1)}σ` : '−∞'} ~ ${C.isNum(v.hi_z) ? `${C.num(v.hi_z, 1)}σ` : '+∞'}` : '—') },
    { key: 'p', label: '采用', fmt: C.prob }, { key: 'm', label: '模型自身质量', fmt: C.prob }, { key: 'p_llm', label: '大模型', fmt: C.prob },
    { key: 'mechanism', label: '说明', wrap: true, fmt: (v, b) => C.humanize(v || b.memory || '') },
  ], branches || [], { stack: true, rowClass: (b) => (b.id === mine ? 'agents-chosen' : '') });
}
const SYNTH_ZH = { called: '大模型判断', gated_off: '门控关闭（用默认分区）', failed: '调用失败（用默认分区）', no_llm: '没有调用大模型', declined: '大模型拒绝' };

function tabConseq(x) {
  const { B, rec, det, pdet, body } = x;
  const { ui } = B.env.ctx;
  if (rec.level > 0) {
    C.put(body, h('div', { class: 'section-title', style: { marginTop: '4px' } }, `本季的后果（${rec.quarter}，在这一动作簇下）`));
    const cons = (pdet?.consequences || []).find(c => c.cluster === rec.cluster?.id);
    if (cons) {
      C.put(body, h('div', { class: 'agents-chips', style: { marginBottom: '6px' } }, C.chip(SYNTH_ZH[cons.synth_status] || '后果合成', cons.synth_status === 'called' ? 'accent' : 'warn'),
        ...(cons.gates || []).map(g => C.chip(`门控：${C.gateName(g)}`, ''))));
      C.put(body, branchRows(ui, cons.branches, rec.branch?.id));
      if ((cons.report || []).length) C.put(body, h('ul', { class: 'list', style: { fontSize: '12px' } }, cons.report.map(t => h('li', {}, C.prose(t)))));
    } else if (rec.branch) {
      C.put(body, branchRows(ui, [rec.branch], rec.branch.id));
    }
    const br = rec.branch || {};
    if (br.cause && ((br.cause.options || []).length || (br.cause.facts || []).length)) {
      C.put(body, h('p', { class: 'agents-why' }, `原因：${[...(br.cause.options || []), ...(br.cause.state || []), ...(br.cause.facts || [])].map(String).map(C.humanize).join('；')}`));
    }
  }
  C.put(body, h('div', { class: 'section-title' }, '下一季各动作簇的后果（本节点的后果合成）'));
  if (det?.consequences?.length) {
    for (const c of det.consequences) {
      const cl = (det.clusters || []).find(k => k.id === c.cluster);
      C.put(body, h('div', { class: 'agents-cluster', style: { marginBottom: '10px' } },
        h('div', { class: 'agents-cluster-h' }, h('b', {}, C.humanize(cl?.label || '动作簇')), h('span', { class: 'muted' }, `概率 ${C.prob(cl?.p)}`),
          C.chip(SYNTH_ZH[c.synth_status] || '后果合成', c.synth_status === 'called' ? 'accent' : 'warn')),
        branchRows(ui, c.branches, null)));
    }
  } else C.put(body, ui.notice('这个节点没有展开：没有下一季的后果。', 'info'));
  const hz = det?.hazards || {};
  if (Object.keys(hz).length) {
    C.put(body, h('div', { class: 'section-title' }, '自然事件的风险率（模型自身的尾部质量）'),
      ui.table([{ key: 'zh', label: '事件', wrap: true }, { key: 'm', label: '模型尾部质量', fmt: C.prob }, { key: 'p', label: '采用', fmt: C.prob },
        { key: 'table_rate', label: '历史频率', fmt: C.prob }], Object.values(hz), { stack: true }));
  }
}

function tabAccounts(x) {
  const { B, rec, det, body } = x;
  const { ui } = B.env.ctx;
  const lines = det?.accounts?.lines;
  if (!lines) {
    const acc = rec.summary?.accounts || [];
    C.put(body, ui.notice('这个节点没有展开，只保存了几行账户摘要（季末，本分支中位数与 10–90%）。', 'info'));
    if (acc.length) C.put(body, ui.table([{ key: 'zh', label: '科目' }, { key: 'median', label: '中位数', fmt: (v, r) => C.accountValue(v, C.accountLine(r.line).unit) },
      { key: 'p10', label: '10–90%', fmt: (v, r) => `${C.accountValue(r.p10, C.accountLine(r.line).unit)} ~ ${C.accountValue(r.p90, C.accountLine(r.line).unit)}` }], acc, { stack: true }));
    return;
  }
  C.put(body, h('p', { class: 'agents-why' }, `${rec.quarter} 季末，本分支所有粒子的中位数与 10–90% 区间。账户是记账：按存量—流量恒等式逐粒子更新，不写价格，也不是模型的学习输入；行为者每季看到的就是自己的这几张表。`));
  const books = new Map();
  for (const [k, v] of Object.entries(lines)) {
    const [book] = k.split('.');
    if (!books.has(book)) books.set(book, []);
    books.get(book).push([k, v]);
  }
  const order = ['fed', 'fiscal', 'president', 'china', 'opec', 'bigtech', 'voter', 'households', 'nfc', 'banks', 'investors', 'row', 'sectoral', 'views'];
  const ids = [...books.keys()].sort((a, b) => (order.indexOf(a) + 99 * (order.indexOf(a) < 0)) - (order.indexOf(b) + 99 * (order.indexOf(b) < 0)));
  for (const book of ids) {
    const rows = books.get(book).map(([k, v]) => ({ k, ...C.accountLine(k), p10: v[0], med: v[1], p90: v[2] }));
    const byStmt = ['b', 'i', 'c', 'o'].map(sc => [sc, rows.filter(r => r.stmt === sc)]).filter(([, r]) => r.length);
    const d = h('details', { class: 'agents-cluster', style: { marginBottom: '8px' } },
      h('summary', { style: { cursor: 'pointer', fontWeight: 600 } }, C.ACCOUNT_BOOKS[book] || (book === 'views' ? '视图（账户比率）' : '其他'), h('span', { class: 'muted', style: { fontWeight: 400 } }, ` · ${rows.length} 行`)));
    for (const [sc, r] of byStmt) {
      C.put(d, h('div', { class: 'muted', style: { fontSize: '12px', margin: '8px 0 2px' } }, C.STATEMENTS[sc]),
        ui.table([{ key: 'zh', label: '科目', wrap: true }, { key: 'med', label: '中位数', fmt: (v, o) => C.accountValue(v, o.unit) },
          { key: 'p10', label: '10–90%', fmt: (v, o) => (Math.abs((o.p90 ?? 0) - (o.p10 ?? 0)) < 1e-9 ? '（无离散）' : `${C.accountValue(o.p10, o.unit)} ~ ${C.accountValue(o.p90, o.unit)}`) }], r, { stack: true }));
    }
    if (['fed', 'fiscal', 'bigtech', 'china'].includes(book)) d.open = true;
    C.put(body, d);
  }
  const idn = det.accounts.identities || {};
  if (Object.keys(idn).length) {
    const bad = Object.entries(idn).filter(([, v]) => Math.abs(v) > 1);
    C.put(body, h('p', { class: bad.length ? 'is-bad' : 'is-good', style: { fontSize: '13px' } },
      bad.length ? `恒等式残差偏大：${bad.map(([k, v]) => `${k} ${C.num(v, 2)}`).join('，')}` : `恒等式全部成立（最大残差 ${C.num(Math.max(...Object.values(idn).map(Math.abs)), 3)}）`));
  } else if (rec.summary?.identities) {
    C.put(body, h('p', { class: rec.summary.identities.ok ? 'is-good' : 'is-bad', style: { fontSize: '13px' } },
      rec.summary.identities.ok ? `恒等式全部成立（最大残差 ${C.num(rec.summary.identities.max, 3)}）` : `恒等式残差 ${C.num(rec.summary.identities.max, 3)}`));
  }
  const gates = det.accounts.gates || {};
  const notes = Object.entries(gates).filter(([, g]) => g && g.note).map(([a, g]) => h('li', {}, h('b', {}, `${C.actorName(a) === '未命名行为者' ? '合成器' : C.actorName(a)}：`), C.humanize(g.note)));
  if (notes.length) C.put(body, h('details', {}, h('summary', {}, '账户门（账户如何改变下一季的选项与先验）'), h('ul', { class: 'list', style: { fontSize: '12px' } }, notes)));
}

function tabPackets(x) {
  const { B, rec, det, pdet, body } = x;
  const { ui } = B.env.ctx;
  const AG = B.env.AG;
  if (!AG.packetSide) AG.packetSide = rec.level ? 'parent' : 'self';
  const box = h('div');
  const seg = ui.seg([{ value: 'parent', label: `本季决策时（${rec.level ? rec.quarter : '—'}）` }, { value: 'self', label: `下一季决策时（${det?.quarter_next || '—'}）` }],
    AG.packetSide, (v) => { AG.packetSide = v; draw(); });
  const draw = () => {
    const d = AG.packetSide === 'parent' ? pdet : det;
    C.fill(box);
    const pk = d?.packets || {};
    if (!Object.keys(pk).length) { C.put(box, ui.notice(AG.packetSide === 'parent' ? (rec.level ? '父节点没有保存观察包。' : '起点没有上一季。') : '这个节点没有展开：没有下一季的观察包。', 'info')); return; }
    C.put(box, h('p', { class: 'agents-why' }, '每个行为者做决定时看到的全部内容：自己的账户（三张表的中位数与区间）、共享的宏观状态、代码写的新闻摘要和各方的公开表态。原文照录。'));
    for (const [a, p] of Object.entries(pk).sort((x1, y1) => C.actorStage(x1[0]) - C.actorStage(y1[0]))) {
      C.put(box, h('details', { class: 'agents-cluster', style: { marginBottom: '8px' } },
        h('summary', { style: { cursor: 'pointer' } }, h('b', {}, C.actorName(a)), h('span', { class: 'muted' }, ` · ${p.chars ?? '—'} 字，约 ${p.tokens_est ?? '—'} token`)),
        h('pre', { class: 'agents-pre' }, p.text || '')));
    }
  };
  C.put(body, seg, h('div', { style: { height: '8px' } }), box);
  draw();
}

function histPos(f, v) {
  const hst = f?.hist || {};
  const pts = [[0.01, hst.p01], [0.05, hst.p05], [0.5, hst.p50], [0.95, hst.p95], [0.99, hst.p99]].filter(([, x]) => C.isNum(x));
  if (!C.isNum(v) || pts.length < 2) return null;
  if (v <= pts[0][1]) return pts[0][0];
  for (let i = 1; i < pts.length; i++) {
    const [q0, x0] = pts[i - 1], [q1, x1] = pts[i];
    if (v <= x1) return x1 > x0 ? q0 + (q1 - q0) * (v - x0) / (x1 - x0) : q1;
  }
  return pts[pts.length - 1][0];
}
const ENDO_ZH = { E0: '数量', E0p: '政策工具', E1: '金融价格', E1c: '商品价格', E2: '调查与新闻', S: '状态' };
const SRC_ZH = { actor: '行为者写入', 'synth:partition': '后果合成 · 分区', 'synth:mechanism': '后果合成 · 机制', a0_pinned: '两步法固定（季前运行）', model: '模型推断' };

function tabInputs(x) {
  const { B, rec, body } = x;
  const { ui } = B.env.ctx;
  const inp = rec.summary?.inputs;
  if (!inp) { C.put(body, ui.notice(rec.level === 0 ? '起点没有本季输入。' : '这个节点没有保存输入字段（被剪掉的节点只保存摘要）。', 'info')); return; }
  C.put(body, h('p', { class: 'agents-why' }, '季度模型这一季的输入：行为者写入的杠杆、后果合成器的分区或机制、两步法里固定的季前运行值，其余由模型按历史一致性推断。数值是本分支粒子的中位数与 10–90%；历史位置是该字段 1985 年以来季度分布中的分位。'));
  const rows = Object.entries(inp).map(([fid, v]) => ({ fid, f: C.fieldInfo(fid), p10: v[0], p50: v[1], p90: v[2], src: v[3] }));
  const order = ['actor', 'synth:mechanism', 'synth:partition', 'a0_pinned', 'model'];
  rows.sort((a, b) => order.indexOf(a.src) - order.indexOf(b.src) || (a.f?.zh || '').localeCompare(b.f?.zh || '', 'zh'));
  C.put(body, ui.table([
    { key: 'fid', label: '字段', wrap: true, fmt: (v, r) => h('span', {}, C.fieldName(v), r.f?.endo ? ' ' : '', r.f?.endo ? C.chip(ENDO_ZH[r.f.endo] || '其他', '', `类别 ${r.f.endo}：谁可以写这个字段由类别决定（数量与政策工具可由行为者或合成器写；金融价格只由模型给出）`) : null) },
    { key: 'p50', label: '中位数', fmt: (v, r) => `${C.num(v, 2)} ${r.f?.unit || ''}` },
    { key: 'p10', label: '10–90%', fmt: (v, r) => `${C.num(r.p10, 2)} ~ ${C.num(r.p90, 2)}` },
    { key: 'src', label: '来源', fmt: (v) => C.chip(SRC_ZH[v] || '模型推断', v === 'actor' ? 'accent' : v?.startsWith('synth') ? 'warn' : '') },
    { key: 'pos', label: '历史位置', fmt: (v, r) => { const q = histPos(r.f, r.p50); if (q === null) return '—'; const i = h('i'); i.style.left = `${(q * 100).toFixed(0)}%`; return h('span', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center' } }, h('span', { class: 'agents-hist', style: { width: '90px' } }, i), `${Math.round(q * 100)}%`); } },
  ], rows, { stack: true }));
}

function rangeBar(lo, mid, hi, bound) {
  const pos = (v) => `${(50 + C.clamp(v / bound, -1, 1) * 50).toFixed(1)}%`;
  const b = h('span', { class: 'b' }); b.style.left = pos(Math.min(lo, hi)); b.style.width = `${(Math.abs(C.clamp(hi / bound, -1, 1) - C.clamp(lo / bound, -1, 1)) * 50).toFixed(1)}%`;
  const m = h('span', { class: 'm' }); m.style.left = pos(mid);
  const z = h('span', { class: 'z' }); z.style.left = '50%';
  return h('div', { class: 'agents-range' }, h('span', { class: 't' }), z, b, m);
}

function attributionBars(attr, assets, ui) {
  if (!attr) return null;
  const rows = assets.filter(a => attr[a]);
  if (!rows.length) return null;
  const bound = Math.max(0.005, ...rows.map(a => ATTR_PARTS.reduce((s, p) => s + Math.abs(attr[a][p.id] || 0), 0)));
  const out = h('div', { class: 'agents-attr' });
  for (const a of rows) {
    const bar = h('div', { class: 'bar' }, h('span', { class: 'zero' }));
    let pos = 0, neg = 0;
    for (const p of ATTR_PARTS) {
      const v = attr[a][p.id] || 0;
      if (Math.abs(v) < 1e-6) continue;
      const w = (Math.abs(v) / bound) * 50;
      const sp = h('span', { title: `${p.zh}：${C.pct(v, 2, true)}（对数回报）` });
      sp.style.background = p.color;
      sp.style.width = `${w.toFixed(2)}%`;
      if (v > 0) { sp.style.left = `${(50 + pos).toFixed(2)}%`; pos += w; } else { neg += w; sp.style.left = `${(50 - neg).toFixed(2)}%`; }
      C.put(bar, sp);
    }
    const tot = ATTR_PARTS.reduce((s, p) => s + (attr[a][p.id] || 0), 0);
    C.put(out, h('span', {}, assetZh(a, ui)), bar, h('span', { class: ui.signCls(tot), style: { textAlign: 'right' } }, C.pct(tot, 1, true)));
  }
  return h('div', {}, out, h('div', { class: 'agents-legend' }, ATTR_PARTS.map(p => { const i = h('i'); i.style.background = p.color; i.style.borderRadius = '2px'; return h('span', {}, i, p.zh); })));
}

function tabAssets(x) {
  const { B, rec, body } = x;
  const { ui } = B.env.ctx;
  const as = rec.summary?.assets || {};
  const ids = Object.keys(as);
  if (!ids.length) { C.put(body, ui.notice('没有资产分布。', 'info')); return; }
  const bound = Math.max(0.05, ...ids.flatMap(a => (as[a].cum || []).map(v => Math.abs(C.simple(v) || 0))));
  C.put(body, h('p', { class: 'agents-why' }, `季度模型给出的${rec.level ? `${rec.quarter} 季末` : '起点'}分布：本季回报与从起点算起的累计回报（总回报，含收入），5% / 中位数 / 95%。条形是累计回报的 5–95% 区间，竖线是中位数。`));
  C.put(body, ui.table([
    { key: 'a', label: '资产', fmt: (v) => assetZh(v, ui) },
    { key: 'q', label: '本季 5% / 中位数 / 95%', fmt: (v) => (v ? h('span', {}, h('span', { class: C.retCls(v[0]) }, C.retText(v[0])), ' / ', h('b', { class: C.retCls(v[1]) }, C.retText(v[1])), ' / ', h('span', { class: C.retCls(v[2]) }, C.retText(v[2]))) : '—') },
    { key: 'c', label: '累计中位数', fmt: (v) => h('b', { class: C.retCls(v?.[1]) }, C.retText(v?.[1])) },
    { key: 'bar', label: '累计 5–95%', fmt: (v, r) => (r.c ? rangeBar(C.simple(r.c[0]), C.simple(r.c[1]), C.simple(r.c[2]), bound) : '—') },
  ], ids.map(a => ({ a, q: as[a].q, c: as[a].cum })), { stack: true }));
  const attr = rec.summary?.attribution;
  if (attr) {
    C.put(body, h('div', { class: 'section-title' }, '本季回报的来源（中位数分解）'),
      h('p', { class: 'agents-why' }, '行动经模型 = 行为者的动作通过季度模型带来的变化；后果分区 = 合成器选中模型自身分布的哪一段；金融条件分段是情景假说；机制是有上限的增量。单位：对数回报。'),
      attributionBars(attr, MAIN_ASSETS, ui));
  }
}

function tabNews(x) {
  const { B, rec, body } = x;
  const { ui } = B.env.ctx;
  const groups = { action: '要闻：动作', consequence: '后果', data: '数据', fact: '持续事实', signal: '表态' };
  const by = new Map();
  for (const [k, t] of rec.digest || []) { if (!by.has(k)) by.set(k, []); by.get(k).push(t); }
  C.put(body, h('p', { class: 'agents-why' }, '代码写的新闻摘要：只记已经发生的事，没有任何预测。下一季的行为者在观察包里读到的就是这一段。'));
  if (!by.size) C.put(body, ui.notice('这个节点没有新闻摘要。', 'info'));
  for (const [k, list] of by) C.put(body, h('div', { class: 'agents-sec', style: { marginTop: '8px' } }, groups[k] || '其他'), h('ul', { class: 'list' }, list.map(t => h('li', {}, C.humanize(t)))));
  const facts = C.liveFacts(rec.facts);
  if (facts.length) C.put(body, h('div', { class: 'agents-sec', style: { marginTop: '10px' } }, '季末的持续事实'), h('div', { class: 'agents-chips' }, facts.map(([, t]) => C.chip(t))));
  const pol = C.politicalRows(rec.political);
  if (pol.length) C.put(body, h('div', { class: 'agents-sec', style: { marginTop: '10px' } }, '政治格局'), h('dl', { class: 'agents-meta' }, pol.map(([k, v]) => h('div', {}, h('dt', {}, k), h('dd', {}, v)))));
  if ((rec.commitments || []).length) C.put(body, h('div', { class: 'agents-sec', style: { marginTop: '10px' } }, '承诺'), h('ul', { class: 'list' }, rec.commitments.map(c => h('li', {}, C.humanize(typeof c === 'string' ? c : `${C.actorName(c.actor)}：${c.text || ''}`)))));
}

const ERROR_ZH = { timeout: '超时', rate: '请求过于频繁', overloaded: '服务繁忙', server: '服务暂时不可用', failed: '调用失败',
  unparseable: '输出无法解析', empty: '结果为空', refusal: '模型拒绝', max_tokens: '输出过长', context: '输入过长',
  limit: '用量上限', login: '未登录', token: '令牌无效', cli_missing: '找不到 Claude Code', model: '模型不可用', cancelled: '已取消',
  invalid_output: '输出无效', unconfirmed: '未确认' };
function tabCalls(x) {
  const { B, det, pdet, rec, body } = x;
  const { ui } = B.env.ctx;
  const kind = { node: '合并（每节点一次）', actor: '行为者（分开）', synth: '后果合成', repair: '修复', story: '补写叙事' };
  const show = (d, title) => {
    C.put(body, h('div', { class: 'section-title' }, title));
    const calls = d?.calls || [];
    if (!calls.length) { C.put(body, h('div', { class: 'muted', style: { fontSize: '12px' } }, '没有调用记录。')); return; }
    C.put(body, ui.table([
      { key: 'kind', label: '类型', fmt: (v) => kind[v] || '调用' },
      { key: 'cached', label: '缓存', fmt: (v) => (v ? '命中' : '新调用') },
      { key: 'ok', label: '结果', fmt: (v, r) => (r.error_kind === 'budget' ? C.chip('达到上限·只用历史基率', 'warn')
        : C.chip(v ? '成功' : `失败${r.error_kind ? `（${ERROR_ZH[r.error_kind] || '其他原因'}）` : ''}`, v ? 'good' : 'bad')) },
      { key: 'seconds', label: '用时', fmt: (v) => (C.isNum(v) ? `${v.toFixed(v < 1 ? 2 : 1)} 秒` : '—') },
      { key: 'model', label: '模型' }, { key: 'effort', label: '力度', fmt: (v) => C.EFFORT_ZH[v] || (v ? '其他' : '—') },
      { key: 'backend', label: '后端', fmt: (v) => ({ mock: '模拟后端', claude_code: '本机 Claude Code' }[v] || v || '—') },
      { key: 'usage', label: 'token（入/出）', fmt: (v) => (v ? `${v.input_tokens ?? '—'} / ${v.output_tokens ?? '—'}` : '—') },
    ], calls, { stack: true }));
    if ((d.validation_report || []).length) {
      C.put(body, h('details', { style: { marginTop: '6px' } }, h('summary', {}, `护栏报告（${d.validation_report.length} 条）`),
        h('ul', { class: 'list', style: { fontSize: '12px' } }, d.validation_report.map(t => h('li', {}, C.prose(t))))));
    }
  };
  if (rec.level > 0) show(pdet, `本季（${rec.quarter}）的调用：在父节点`);
  show(det, `下一季（${det?.quarter_next || '—'}）的调用：在本节点`);
}

// ================================================================ the path view
async function drawPath(B, key) {
  const { env, src, pathBox } = B;
  const { ui } = env.ctx;
  const token = (B.pathToken = (B.pathToken || 0) + 1);
  C.fill(pathBox, ui.card({ title: '整条路径' }, ui.loading('读取路径…')));
  let recs;
  try { recs = await src.pathRecords(key, { signal: env.signal }); } catch (err) {
    if (err.name === 'AbortError' || token !== B.pathToken) return;
    C.fill(pathBox, ui.card({ title: '整条路径' }, ui.notice(`读取路径失败：${D.errorText(err)}`, 'error')));
    return;
  }
  if (token !== B.pathToken || !env.alive()) return;
  const charts = env.ctx.charts;
  const xs = recs.map(r => (r.level === 0 ? `${r.quarter}末` : r.quarter));
  const lineBox = ui.chartBox();
  const fanBox = ui.chartBox();
  const assetSel = h('select', { 'aria-label': '资产' }, MAIN_ASSETS.map(a => h('option', { value: a }, assetZh(a, ui))));
  const us = [];
  const rows = recs.map((r, i) => {
    us.push(r.summary?.rep?.s_unrate);
    const rg = i ? C.regimeOf(r.summary?.rep, r.summary?.bands?.d_rgdp?.[1], us.slice(-5)) : null;
    return { r, rg };
  });
  const table = ui.table([
    { key: 'q', label: '季度', fmt: (v, o) => h('button', { class: 'btn', type: 'button', style: { padding: '1px 8px', fontSize: '12px' }, onclick: () => B.select(o.r.key, { scroll: true }) }, o.r.level === 0 ? '起点' : o.r.quarter) },
    { key: 'a', label: '动作簇', wrap: true, minWidth: 180, fmt: (v, o) => (o.r.level ? C.humanize(o.r.cluster?.label || '—') : '—') },
    { key: 'b', label: '后果', fmt: (v, o) => (o.r.level ? `${o.r.branch?.label || C.BRANCH_KIND[o.r.branch?.kind] || '—'} · ${C.prob(o.r.p_cond)}` : '—') },
    { key: 'g', label: '体制', fmt: (v, o) => (o.rg ? C.regimeBadge(o.rg) : '—') },
    { key: 'f', label: '利率 / 失业 / 核心', fmt: (v, o) => `${C.pp(o.r.summary?.rep?.s_ff)} / ${C.pp(o.r.summary?.rep?.s_unrate)} / ${C.pp(o.r.summary?.rep?.s_core_yoy)}` },
    { key: 'e', label: '美股本季 / 累计', fmt: (v, o) => { const a = o.r.summary?.assets?.us_equity; return a ? h('span', {}, h('span', { class: C.retCls(a.q?.[1]) }, C.retText(a.q?.[1])), ' / ', h('b', { class: C.retCls(a.cum?.[1]) }, C.retText(a.cum?.[1]))) : '—'; } },
    { key: 't', label: '10年期美债累计', fmt: (v, o) => { const a = o.r.summary?.assets?.treasury_10y; return a ? h('span', { class: C.retCls(a.cum?.[1]) }, C.retText(a.cum?.[1])) : '—'; } },
    { key: 'acc', label: '账户（中位数）', wrap: true, minWidth: 170, fmt: (v, o) => (o.r.summary?.accounts || []).slice(0, 3).map(a => `${a.zh} ${C.accountValue(a.median, C.accountLine(a.line).unit)}`).join('；') || '—' },
  ], rows, { stack: true });
  const last = recs[recs.length - 1];
  if (recs.length < 2) {
    C.fill(pathBox, ui.card({ title: '整条路径', sub: '选中起点之后的节点，这里逐季列出从起点到它的动作、后果、因子、资产与账户。' },
      h('div', { class: 'muted' }, '起点本身没有路径。点击情景树上的任一节点。')));
    return;
  }
  C.fill(pathBox, ui.card({
    title: `整条路径：起点 → ${last?.quarter || ''}（${recs.length - 1} 季）`,
    sub: '逐季的动作、后果、因子、资产与账户；点季度按钮跳到那个节点。上图是这条路径上的季末中位数；下图是所选资产的累计回报扇形（5–95%，起点 = 100）。',
    actions: h('div', { class: 'agents-rowbtns' },
      h('button', { class: 'btn', type: 'button', title: '让大模型把这条路径写成一段叙事（1 次调用，结果缓存）', onclick: (e) => writeStory(B, key, e.target) }, '补写叙事'),
      h('button', { class: 'btn', type: 'button', onclick: () => env.go('story', { node: key, page: 1 }) }, '从头播放连环画')),
  }, h('div', { class: 'grid cols-2' }, h('div', {}, h('div', { class: 'agents-sec' }, '因子路径'), lineBox),
    h('div', {}, h('div', { class: 'agents-sec', style: { display: 'flex', gap: '8px', alignItems: 'center' } }, '资产扇形', assetSel), fanBox)),
  h('div', { class: 'agents-pathwrap', style: { marginTop: '12px' } }, table)));
  charts.line(lineBox, { x: xs, yFmt: 'pp', series: [
    { name: '政策利率', data: recs.map(r => r.summary?.rep?.s_ff ?? null), color: C.actorColor('fed') },
    { name: '失业率', data: recs.map(r => r.summary?.rep?.s_unrate ?? null), color: '#f08c00' },
    { name: '核心 CPI 同比', data: recs.map(r => r.summary?.rep?.s_core_yoy ?? null), color: '#c2255c' },
    { name: '10 年期收益率', data: recs.map(r => r.summary?.rep?.s_y10 ?? null), color: '#0b7285', dashed: true },
  ] });
  const drawFan = () => {
    const a = assetSel.value;
    const idx = (k) => recs.map(r => { const c = r.summary?.assets?.[a]?.cum; return C.isNum(c?.[k]) ? 100 * Math.exp(c[k]) : (r.level === 0 ? 100 : null); });
    charts.fan(fanBox, { x: xs, bands: { p5: idx(0), p50: idx(1), p95: idx(2) }, name: `${assetZh(a, ui)}（中位数）`, fmt: (v) => C.num(v, 1), color: env.ctx.charts.assetColor(a) });
  };
  assetSel.onchange = drawFan;
  drawFan();
}

async function writeStory(B, key, btn, confirm = false) {
  const { env, src, pathBox } = B;
  const box = pathBox.querySelector('.agents-storyout') || C.put(pathBox.querySelector('.card') || pathBox, h('div', { class: 'agents-storyout', style: { marginTop: '10px' } })).lastChild;
  btn.disabled = true;
  C.fill(box, h('span', { class: 'agents-msg' }, h('span', { class: 'spinner', style: { width: '12px', height: '12px' } }), env.mock ? '补写中（模拟后端）…' : '补写中…'));
  try {
    const r = await D.post(`${src.base}/nodes/${key}/story`, { confirm });
    const st = r.story || {};
    C.fill(box, h('div', { class: 'agents-cluster' }, h('div', { class: 'agents-cluster-h' }, h('b', {}, C.humanize(st.headline || '这条路径')), C.chip('大模型叙事：只复述路径，不是预测', 'warn')),
      h('p', { class: 'agents-why', style: { fontSize: '13px', whiteSpace: 'pre-wrap' } }, C.humanize(st.story || ''))));
  } catch (err) {
    if (err.status === 409 && err.body?.needs_confirm && !confirm) {
      btn.disabled = false;
      if (window.confirm('补写叙事会调用一次本机 Claude Code（计入订阅用量，结果缓存）。确认？')) return writeStory(B, key, btn, true);
      C.fill(box);
      return;
    }
    C.fill(box, h('div', { class: 'agents-msg err' }, `没有补写：${D.errorText(err)}`));
  }
  btn.disabled = false;
}

// ================================================================ branch comparison
async function drawCompare(B) {
  const { env, src, tree, cmpBox } = B;
  const { ui } = env.ctx;
  const charts = env.ctx.charts;
  const token = (B.cmpToken = (B.cmpToken || 0) + 1);
  const d = tree.maxLoadedLevel();
  if (d < 1) { C.fill(cmpBox, ui.card({ title: '分支对比' }, h('div', { class: 'muted' }, '至少完成一季后才能比较分支。'))); return; }
  C.fill(cmpBox, ui.card({ title: '分支对比' }, ui.loading('读取末端分支…')));
  let mine, base = null, model = null;
  try {
    mine = await src.branches(undefined, { signal: env.signal });
    const pt = tree.summary?.prior_trees || {};
    if (src.kind === 'run' && tree.summary?.status === 'done' && (pt.base_rate || pt.model)) {
      [base, model] = await Promise.all([pt.base_rate ? src.priorBranches('base_rate').catch(() => null) : null,
        pt.model ? src.priorBranches('model').catch(() => null) : null]);
    }
  } catch (err) {
    if (err.name === 'AbortError' || token !== B.cmpToken) return;
    C.fill(cmpBox, ui.card({ title: '分支对比' }, ui.notice(`读取末端分支失败：${D.errorText(err)}`, 'error')));
    return;
  }
  if (token !== B.cmpToken || !env.alive()) return;
  const leaves = (mine.leaves || []).slice(0, 30);
  const tot = leaves.reduce((s, l) => s + (l.p || 0), 0) || 1;
  const top = leaves.slice(0, 6);

  // three trees
  const tri = h('div', { class: 'agents-tri' });
  const trees = [['智能体树', mine, ACCENT], ['历史基率树', base, '#868e96'], ['模型树', model, INK]];
  const avail = trees.filter(([, t]) => t?.mixture_cum && Object.keys(t.mixture_cum).length);
  const bound = Math.max(0.05, ...avail.flatMap(([, t]) => Object.values(t.mixture_cum).flatMap(q => [Math.abs(C.simple(q[0])), Math.abs(C.simple(q[4]))])));
  for (const a of Object.keys(mine.mixture_cum || {})) {
    C.put(tri, h('div', { class: 'asset' }, assetZh(a, ui)));
    for (const [name, t, color] of trees) {
      const q = t?.mixture_cum?.[a];
      if (!q) continue;
      const s = q.map(C.simple);
      const pos = (v) => `${(50 + C.clamp(v / bound, -1, 1) * 50).toFixed(1)}%`;
      const wid = (lo, hi) => `${((C.clamp(hi / bound, -1, 1) - C.clamp(lo / bound, -1, 1)) * 50).toFixed(1)}%`;
      const box = h('div', { class: 'agents-range', style: { height: '16px' } }, h('span', { class: 't' }));
      const outer = h('span', { class: 'b' }); outer.style.left = pos(s[0]); outer.style.width = wid(s[0], s[4]); outer.style.background = `${color}22`;
      const inner = h('span', { class: 'b' }); inner.style.left = pos(s[1]); inner.style.width = wid(s[1], s[3]); inner.style.background = `${color}66`;
      const m = h('span', { class: 'm' }); m.style.left = pos(s[2]); m.style.background = color;
      const z = h('span', { class: 'z' }); z.style.left = '50%';
      C.put(box, z, outer, inner, m);
      C.put(tri, h('span', { class: 'name' }, name), box, h('span', { class: 'vals' }, `${C.pct(s[0], 0, true)} / `, h('b', { class: ui.signCls(s[2]) }, C.pct(s[2], 1, true)), ` / ${C.pct(s[4], 0, true)}`));
    }
  }
  // attribution along each leaf's path, weighted by the leaf's probability
  const attr = {};
  let wsum = 0;
  for (const l of leaves) {
    const keys = tree.pathKeys(l.key);
    if (keys.length < 2) continue;
    wsum += l.p || 0;
    for (const k of keys) {
      const at = tree.get(k)?.attr;
      if (!at) continue;
      for (const a of Object.keys(at)) {
        attr[a] = attr[a] || {};
        for (const p of ATTR_PARTS) attr[a][p.id] = (attr[a][p.id] || 0) + (l.p || 0) * (at[a][p.id] || 0);
      }
    }
  }
  if (wsum > 0) for (const a of Object.keys(attr)) for (const p of ATTR_PARTS) attr[a][p.id] /= wsum;

  // factor paths and fans of the top leaves
  const fSel = h('select', { 'aria-label': '因子' }, FACTORS.map(f => h('option', { value: f.id }, f.zh)));
  const aSel = h('select', { 'aria-label': '资产' }, MAIN_ASSETS.map(a => h('option', { value: a }, assetZh(a, ui))));
  const lSel = h('select', { 'aria-label': '分支' }, top.map((l, i) => h('option', { value: l.key }, `#${i + 1} ${C.prob(l.p / tot)}`)));
  const lineBox = ui.chartBox();
  const fanBox = ui.chartBox();
  const paths = top.map(l => tree.pathKeys(l.key).map(k => tree.get(k)));
  const xs = (paths[0] || []).map((n, i) => (n ? (i === 0 ? `${n.quarter}末` : n.quarter) : ''));
  const colours = ['#3b5bdb', '#e8590c', '#7048e8', '#0b7285', '#c2255c', '#f08c00'];
  const drawLines = () => {
    const f = fSel.value;
    charts.line(lineBox, { x: xs, yFmt: f === 's_cape' ? undefined : 'pp', series: paths.map((p, i) => ({ name: `#${i + 1} ${C.prob(top[i].p / tot)}`, data: p.map(n => n?.rep?.[f] ?? null), color: colours[i % colours.length] })) });
  };
  const drawFans = () => {
    const a = aSel.value;
    const i = Math.max(0, top.findIndex(l => l.key === lSel.value));
    const idx = (p, k) => p.map(n => { const c = n?.assets?.[a]?.c; return C.isNum(c?.[k]) ? 100 * Math.exp(c[k]) : (n?.level === 0 ? 100 : null); });
    const p = paths[i] || [];
    charts.fan(fanBox, { x: xs, bands: { p5: idx(p, 0), p50: idx(p, 1), p95: idx(p, 2) }, name: `#${i + 1} 中位数`, color: colours[i % colours.length],
      fmt: (v) => C.num(v, 1), extra: paths.map((q, j) => (j === i ? null : { name: `#${j + 1}`, data: idx(q, 1), dashed: true, color: colours[j % colours.length] })).filter(Boolean) });
  };
  fSel.onchange = drawLines; aSel.onchange = drawFans; lSel.onchange = drawFans;

  const leafRows = leaves.map((l, i) => ({ ...l, i }));
  // the KPI: the agent tree against the base-rate tree at quarter 8 (or the last one), probability-weighted medians
  const kd = Math.min(8, d);
  const wmed = (nodes) => {
    let w = 0, v = 0;
    for (const n of nodes) { const c = n?.assets?.us_equity?.c?.[1] ?? n?.assets?.us_equity?.cum?.[1]; if (C.isNum(c) && C.isNum(n.p)) { w += n.p; v += n.p * c; } }
    return w > 0 ? v / w : null;
  };
  const agentK = wmed(tree.nodesAt(kd).filter(n => TreeStore.shown(n) && n.source !== 'overlay'));
  let baseK = null;
  if (base) {
    try {
      const bk = await src.priorBranches('base_rate', kd, { signal: env.signal });
      baseK = wmed((bk.leaves || []).map(l => ({ p: l.p, assets: { us_equity: l.assets?.us_equity } })));
    } catch (_) { baseK = null; }
  }
  if (token !== B.cmpToken || !env.alive()) return;
  const kpi = h('div', { class: 'grid cols-3', style: { marginBottom: '12px' } },
    ui.kpi(`第 ${kd} 季美股累计（智能体树）`, C.retText(agentK), '各节点中位数按路径概率加权（近似）', C.retCls(agentK)),
    ui.kpi(`第 ${kd} 季美股累计（历史基率树）`, base ? C.retText(baseK) : '—', base ? '大模型权重 0 的同一棵树' : '这次没有对照树', C.retCls(baseK)),
    ui.kpi('大模型的作用', C.isNum(agentK) && C.isNum(baseK) ? `${((Math.expm1(agentK) - Math.expm1(baseK)) * 100).toFixed(1)} 个百分点` : '—',
      '智能体树减历史基率树；判断值的权重下，未经校准'));
  // action shares per quarter (before reduction: exact under the tree model)
  const shares = mine.action_shares || [];
  const shareActors = [...new Set(shares.flatMap(r => Object.keys(r.actors || {})))].sort((a, b) => C.actorStage(a) - C.actorStage(b));
  const shareSel = h('select', { 'aria-label': '行为者' }, shareActors.map(a => h('option', { value: a }, C.actorName(a))));
  const shareBox = h('div');
  const drawShares = () => {
    const a = shareSel.value;
    const opts = [...new Set(shares.flatMap(r => Object.keys(r.actors?.[a] || {})))];
    const rows = shares.filter(r => r.actors?.[a]).map(r => {
      const vals = Object.fromEntries(opts.map(o => [o, r.actors[a][o] ?? 0]));
      return { quarter: r.quarter, ...vals, _idle: Math.max(0, 1 - Object.values(vals).reduce((t, v) => t + v, 0)) };
    });
    const idle = rows.some(r => r._idle > 0.005);
    const bar = (v) => h('span', { style: { display: 'inline-flex', gap: '4px', alignItems: 'center', minWidth: '70px' } }, C.probBar(v), C.prob(v));
    C.fill(shareBox, rows.length ? ui.table([{ key: 'quarter', label: '季度' },
      ...opts.map(o => ({ key: o, label: C.optionLabel(a, o), fmt: bar })), ...(idle ? [{ key: '_idle', label: '不行动', fmt: bar }] : [])],
    rows.slice(0, 24), { stack: true }) : h('div', { class: 'muted' }, '没有这个行为者的决策。'));
  };
  shareSel.onchange = drawShares;
  // branch × asset: the top leaves' cumulative medians, coloured by sign (K-line colours)
  const { up, down } = charts.klineColors();
  const heat = h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'agents-heat' },
    h('thead', {}, h('tr', {}, h('th', {}, '分支'), ...MAIN_ASSETS.map(a => h('th', {}, assetZh(a, ui))))),
    h('tbody', {}, top.map((l, i) => h('tr', {}, h('th', {}, `#${i + 1} ${C.prob(l.p / tot)}`), ...MAIN_ASSETS.map(a => {
      const v = C.simple(l.assets?.[a]?.cum?.[1]);
      const td = h('td', { class: ui.signCls(v) }, C.pct(v, 1, true));
      if (C.isNum(v)) td.style.background = mix('#ffffff', v >= 0 ? up : down, Math.min(1, Math.abs(v) / 0.4) * 0.35);
      return td;
    }))))));
  C.fill(cmpBox, ui.card({
    title: `末端分支（${mine.quarter || ''}，第 ${mine.depth} 季）`,
    sub: `${leaves.length} 个末端节点，概率合计 ${C.prob(mine.mass ?? tot)}（剪掉的质量已并入保留的节点）。混合分布是这些末端粒子按路径概率加权的结果。`,
  },
  ui.table([
    { key: 'i', label: '#', fmt: (v) => `#${v + 1}` },
    { key: 'p', label: '概率', fmt: (v) => h('span', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center', minWidth: '96px' } }, C.probBar(v, { max: leaves[0]?.p || 1 }), C.prob(v)) },
    { key: 'label', label: '最后一季的动作簇', wrap: true, fmt: (v) => C.cut(C.humanize(v || '—'), 48) },
    { key: 'branch', label: '后果' },
    { key: 'assets', label: '美股累计中位数', fmt: (v) => h('b', { class: C.retCls(v?.us_equity?.cum?.[1]) }, C.retText(v?.us_equity?.cum?.[1])) },
    { key: 'key', label: '', fmt: (v) => h('span', { class: 'agents-rowbtns' }, h('button', { class: 'btn', type: 'button', onclick: () => B.select(v, { scroll: true }) }, '查看'),
      h('button', { class: 'btn', type: 'button', onclick: () => env.go('story', { node: v, page: 1 }) }, '连环画')) },
  ], leafRows.slice(0, 12), { stack: true }),
  h('div', { class: 'grid cols-2 text', style: { marginTop: '14px' } },
    h('div', {}, h('div', { class: 'agents-sec' }, src.kind === 'run' ? '三棵树：智能体树、历史基率树（大模型权重 0）、模型树（只有季度模型）' : '末端混合分布（快照没有对照树）'),
      h('p', { class: 'agents-why' }, '累计回报 5% / 中位数 / 95%；深色是 25–75%。大模型的作用 = 智能体树与历史基率树之差。'), tri,
      src.kind === 'run' && !base ? h('div', { class: 'muted', style: { fontSize: '12px' } }, '这次推演没有对照树（关闭或尚未完成）。') : null),
    h('div', {}, h('div', { class: 'agents-sec' }, '路径回报的来源（按末端概率加权）'),
      h('p', { class: 'agents-why' }, '每个末端节点沿路径把各季的中位数分解相加，再按末端概率加权；近似值，单位为对数回报。'),
      attributionBars(attr, MAIN_ASSETS, ui) || h('div', { class: 'muted' }, '树还在载入，或节点没有分解。'))),
  h('div', { class: 'grid cols-2', style: { marginTop: '14px' } },
    h('div', {}, h('div', { class: 'agents-sec', style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' } }, '前 6 个分支的因子路径', fSel), lineBox),
    h('div', {}, h('div', { class: 'agents-sec', style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' } }, '资产扇形', aSel, lSel), fanBox)),
  h('div', { class: 'grid cols-2 text', style: { marginTop: '14px' } },
    h('div', {}, h('div', { class: 'agents-sec' }, '分支 × 资产：前 6 个末端分支的累计回报中位数'), heat),
    h('div', {}, h('div', { class: 'agents-sec', style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' } }, '各季动作分布（剪枝之前，按采用概率）', shareSel), shareBox))));
  const cardEl = cmpBox.firstElementChild;
  const headEl = cardEl?.querySelector('.card-head');
  if (headEl) headEl.after(kpi); else if (cardEl) C.pre(cardEl, kpi);
  if (shareActors.length) drawShares();
  if (paths.length && paths[0].every(Boolean)) { drawLines(); drawFans(); }
  else { C.fill(lineBox, h('div', { class: 'muted', style: { padding: '20px' } }, '树载入完成后显示。')); }
}
