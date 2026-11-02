// 智能体推演: actors choose actions each quarter, a consequence synthesizer turns joint actions into consequences,
// and the history-estimated quarterly model sets prices. This tab browses frozen snapshots (beam search, read-only,
// instant), plays any path as a comic (连环画), starts new runs (pinned facts, settings, estimate, explicit
// confirmation for the real Claude Code), shows their progress, and lists saved runs and snapshots.
//
// Deep links (#agents?…): view=browse|story|setup|saved, snap=<id> or run=<id>, node=<key>, q=<page>, leaf=<key> (the
// comic's path is root → its leaf, so a reload or a shared link plays the same path).
// Module state (AG) survives re-renders (the K-line toggle) and visits within the page load.
import * as C from './_agent_common.js';
import * as D from './_agent_data.js';
import { renderBrowser } from './_agent_tree.js';
import { renderStory } from './_agent_story.js';

const { h } = C;

const AG = {
  view: null, src: null, node: null, page: null, leaf: null,
  draft: null, snapDraft: null, facts: null, factEdits: {}, factOpen: new Set(),
  activeRun: null, lastRun: null, filters: null,
};
export const agentState = AG;

const VIEWS = [
  { value: 'browse', label: '快照与推演' },
  { value: 'story', label: '连环画' },
  { value: 'setup', label: '新推演' },
  { value: 'saved', label: '已保存' },
];

const CSS = `
.agents-top { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; margin-bottom: 14px; }
.agents-top .seg button { font-size: 13px; padding: 6px 14px; }
.agents-src { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; min-width: 0; }
.agents-src select { max-width: min(460px, 100%); }
.agents-src label { font-size: 12px; color: var(--ink-3); }
.agents-chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.agents-banner { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin-bottom: 12px; }
.agents-hero { border: 1px solid #cdd7ff; border-left: 4px solid var(--accent); background: linear-gradient(135deg, #f3f6ff, #fff 70%);
  border-radius: var(--radius); padding: 14px 18px; margin-bottom: 16px; }
.agents-hero h2 { margin: 0 0 8px; font-size: 16px; font-weight: 650; color: #1f2f7a; line-height: 1.5; }
.agents-flow { display: flex; flex-wrap: wrap; gap: 6px; align-items: stretch; }
.agents-step { flex: 1 1 150px; background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 7px 10px; font-size: 12px; color: var(--ink-2); }
.agents-step b { display: block; color: var(--ink); font-size: 13px; }
.agents-step.llm { border-color: #f5d9a8; background: #fffaf0; }
.agents-step.engine { border-color: #bfe8d3; background: #f3fbf7; }
.agents-step.code { border-color: #d0d7e2; background: #f8f9fb; }
.agents-arrow { align-self: center; color: var(--ink-3); }
.agents-form { display: grid; gap: 14px; }
.agents-form .row { gap: 10px 16px; }
.agents-form label.inl { display: inline-flex; gap: 6px; align-items: center; font-size: 13px; color: var(--ink-2); }
.agents-form input[type=number] { width: 84px; }
.agents-form input[type=range] { width: min(220px, 100%); }
.agents-form textarea:disabled { background: var(--panel-2); color: var(--ink-3); cursor: not-allowed; }
.agents-sec { font-size: 12px; font-weight: 600; color: var(--ink-2); margin-bottom: 6px; }
.agents-actorgrid { display: flex; flex-wrap: wrap; gap: 6px; }
.agents-actor { display: inline-flex; gap: 6px; align-items: flex-start; border: 1px solid var(--line); border-radius: 8px; padding: 6px 9px;
  font-size: 12px; background: var(--panel); max-width: 100%; cursor: pointer; }
.agents-actor input { margin: 2px 0 0; }
.agents-actor .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-top: 4px; flex: 0 0 auto; }
.agents-actor b { font-weight: 600; color: var(--ink); }
.agents-actor small { display: block; color: var(--ink-3); font-size: 11px; line-height: 1.4; max-width: 260px; }
.agents-actor.off { opacity: .6; }
.agents-kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 12px; font-size: 13px; }
.agents-kv dt { color: var(--ink-3); } .agents-kv dd { margin: 0; font-variant-numeric: tabular-nums; }
.agents-big { font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; }
.agents-levels { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 4px; }
.agents-levels span { min-width: 18px; height: 18px; border-radius: 4px; background: var(--accent-soft); color: var(--accent); font-size: 10px;
  display: inline-grid; place-items: center; padding: 0 3px; }
.agents-levels span.done { background: var(--accent); color: #fff; }
.agents-levels span.synth { box-shadow: inset 0 -3px 0 #f5b84b; }
.agents-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 12px; }
.agents-msg { font-size: 12px; color: var(--ink-3); display: inline-flex; gap: 6px; align-items: center; }
.agents-msg.ok { color: var(--good); } .agents-msg.err { color: var(--bad); }
.agents-confirm { border: 1px solid #f5d9a8; background: #fffaf0; border-radius: 8px; padding: 12px 14px; margin-top: 12px; font-size: 13px; }
.agents-confirm label { display: flex; gap: 8px; align-items: flex-start; margin: 8px 0; }
.agents-facts { display: grid; gap: 8px; grid-template-columns: repeat(auto-fill, minmax(min(430px, 100%), 1fr)); }
.agents-fact { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 12px; align-items: start; border: 1px solid var(--line);
  border-radius: 8px; padding: 8px 12px; background: var(--panel); }
.agents-fact.unconfirmed { border-left: 4px solid var(--warn); background: #fffdf7; }
.agents-fact.confirmed { border-left: 4px solid var(--good); }
.agents-fact.edited { box-shadow: 0 0 0 2px var(--accent-soft); }
.agents-fact .t { font-weight: 600; font-size: 13px; }
.agents-fact .val { font-size: 12px; color: var(--ink-2); overflow-wrap: anywhere; }
.agents-fact.compact { padding: 6px 10px; align-items: center; }
.agents-fact .why { font-size: 12px; color: var(--ink-3); grid-column: 1 / -1; line-height: 1.5; }
.agents-fact .ed { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; min-width: 0; grid-column: 1 / -1; }
.agents-fact .ed input[type=text] { flex: 1 1 260px; min-width: 0; }
.agents-fact .ed input[type=number] { width: 140px; }
.agents-fact .ed select { max-width: 100%; }
.agents-fact .st { display: flex; gap: 6px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
.agents-fact .btn { padding: 2px 9px; font-size: 12px; }
@media (max-width: 760px) {
  .agents-fact { grid-template-columns: 1fr; }
  .agents-fact .st { justify-content: flex-start; }
}
.agents-progress { height: 10px; border-radius: 5px; background: var(--panel-2); overflow: hidden; border: 1px solid var(--line); }
.agents-progress > span { display: block; height: 100%; background: var(--accent); transition: width .3s ease; }
.agents-calls { font-size: 12px; }
.agents-calls td, .agents-calls th { padding: 4px 8px !important; }
.agents-statusdot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; background: var(--accent); margin-right: 4px; }
.agents-statusdot.live { animation: agents-pulse 1.2s ease-in-out infinite; }
@keyframes agents-pulse { 50% { opacity: .25; } }
.agents-rowbtns { display: flex; flex-wrap: wrap; gap: 4px; justify-content: flex-end; }
.agents-rowbtns .btn { padding: 2px 9px; font-size: 12px; }
.agents-name { display: inline-flex; gap: 6px; align-items: center; }
.agents-name button { border: 0; background: none; cursor: pointer; color: var(--ink-3); font-size: 12px; padding: 0 2px; }
.agents-export { display: inline-flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.agents-export .why { font-size: 12px; color: var(--ink-3); }
.agents-pbar { width: 100%; min-width: 40px; }
.agents-regime { display: inline-block; white-space: nowrap; font-weight: 700; font-size: 12px; border-radius: 999px; padding: 1px 9px; border: 1.5px solid rgba(29, 36, 51, .55); }
code.agents-cmd { font-family: var(--mono); font-size: 12px; background: var(--panel-2); border: 1px solid var(--line); border-radius: 6px; padding: 2px 6px;
  word-break: break-all; }
.agents-sticky { position: sticky; top: 64px; z-index: 4; background: var(--bg); padding: 6px 0; }
.agents-runbar { display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: center; font-size: 12px; color: var(--ink-2);
  border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; background: var(--panel); }
.agents-runbar b { color: var(--ink); font-size: 13px; }
@media (max-width: 760px) { .agents-sticky { top: 55px; } }
`;

// ---------------------------------------------------------------- entry
export async function render(root, ctx) {
  const { ui } = ctx;
  C.ensureStyle('agents', CSS);
  const alive = ctx.alive || (() => root.isConnected);
  const signal = ctx.signal;

  C.put(root, ui.loading('载入智能体推演…'));
  let cfg, schema, llm, snaps, runs;
  try {
    [cfg, schema, llm, snaps, runs] = await Promise.all([
      ctx.api.get('agents/config'), ctx.api.get('agents/schema'),
      D.get('agents/llm_status').catch(err => ({ available: false, reason: D.errorText(err) })),
      D.get('agents/snapshots').catch(() => ({ snapshots: [] })),
      D.get('agents/runs').catch(() => ({ runs: [] })),
    ]);
  } catch (err) {
    if (!alive()) return null;
    C.fill(root, ui.notice(`智能体推演的设置读取失败：${D.errorText(err)}`, 'error'));
    if (err.status === 404) C.put(root, ui.buildHint(err));
    return null;
  }
  if (!alive()) return null;
  C.setCatalogue(cfg);
  C.setSchema(schema);
  C.fill(root);

  const env = { ctx, cfg, schema, llm, snaps: snaps.snapshots || [], runs: runs.runs || [], AG, alive, signal, root };
  env.mock = !!llm?.mock || llm?.backend === 'mock';
  env.stub = !!cfg.model?.stub;
  env.refreshLists = async () => {
    const [s, r] = await Promise.all([D.get('agents/snapshots').catch(() => null), D.get('agents/runs').catch(() => null)]);
    if (s) env.snaps = s.snapshots || [];
    if (r) env.runs = r.runs || [];
    syncActive(env);
  };

  if (!ctx.rerender) applyRoute(env, ctx.param);
  // a source remembered from an earlier visit that has since been deleted
  if (AG.src && !(AG.src.kind === 'snapshot' ? env.snaps.some(s => s.id === AG.src.id) : env.runs.some(r => r.id === AG.src.id))) {
    AG.src = null; AG.node = null; AG.page = null;
  }
  if (!AG.view) AG.view = defaultView(env);
  if (!AG.src) AG.src = defaultSource(env);
  syncActive(env);

  // ---- header
  C.put(root, h('p', { class: 'lead' },
    '每个季度，美联储、国会、总统、中国、欧佩克+、超大型科技公司等行为者各自给出几个可能的动作和概率；后果合成器把联合动作变成几种可能的后果；',
    '资产价格只由按历史估计的季度模型给出，再作为下一季的起点。分支 = 动作簇 × 后果，概率沿路径相乘。预测只发生在情景层。'));
  const chips = h('div', { class: 'agents-chips', style: { marginBottom: '12px' } },
    env.stub ? C.chip(cfg.model?.label || '桩模型（不是资产模型）', 'warn', '季度资产模型还在设计中：现在用的是高斯桩模型，资产价格只用于检查流程，不能当作结果') : null,
    C.chip('前瞻 · 未经验证', 'accent', '从今天出发的推演要等这些季度过去才能检验'),
    env.mock ? C.chip('模拟后端：不调用模型，不消耗额度', 'good', llm?.reason || '')
      : llm?.available ? C.chip('本机 Claude Code', '', `${llm.model || ''} · ${llm.cli_version || ''}`)
        : C.chip('本机 Claude Code 不可用', 'bad', llm?.reason || ''));
  C.put(root, chips);

  const banner = h('div', { class: 'agents-banner' });
  const top = h('div', { class: 'agents-top' });
  const body = h('div', { class: 'agents-body' });
  C.put(root, top, banner, body);

  const go = (view, patch = {}) => {
    // a new node or source means a new comic path (its own best leaf), unless the caller names the leaf
    if (('node' in patch || 'src' in patch) && !('leaf' in patch)) patch = { ...patch, leaf: null };
    if (('node' in patch || 'src' in patch) && !('page' in patch)) patch = { ...patch, page: null };
    Object.assign(AG, patch, { view });
    draw();
  };
  env.go = go;
  env.setRoute = () => ctx.ui.setRoute('agents', routeOf(), alive);

  // Each view gets its own abort signal (linked to the tab's): switching views inside the tab stops the previous
  // view's player, listeners, intervals and requests, as leaving the tab does.
  let viewCtl = null;
  let drawToken = 0;
  async function draw() {
    if (!alive()) return null;
    viewCtl?.abort();
    const ctl = new AbortController();
    viewCtl = ctl;
    signal?.addEventListener('abort', () => ctl.abort(), { once: true });
    // a run or build the tab last saw as live may have finished meanwhile: read the lists before drawing
    const token = ++drawToken;
    if (AG.activeRun || token > 1) await env.refreshLists();     // the first draw has just read them
    if (token !== drawToken || ctl.signal.aborted || !alive()) return null;
    const venv = Object.create(env);
    venv.signal = ctl.signal;
    venv.alive = () => alive() && !ctl.signal.aborted;
    ctx.charts.disposeWithin(body);
    C.fill(body);
    C.fill(top, ctx.ui.seg(VIEWS, AG.view, (v) => go(v)));
    syncActive(env);
    drawBanner(env, banner);
    env.setRoute();
    const target = h('div');
    C.put(body, target);
    if (AG.view === 'setup') return renderSetup(target, venv);
    if (AG.view === 'saved') return renderSaved(target, venv);
    // browse / story need a source
    const picker = sourcePicker(env, () => draw());
    C.put(top, picker);
    if (!AG.src) {
      C.put(target, emptyState(env));
      return null;
    }
    const src = new D.Source(AG.src.kind, AG.src.id);
    env.source = src;
    env.tree = D.treeOf(src);
    venv.source = src;
    venv.tree = env.tree;
    if (AG.view === 'story') return renderStory(target, venv);
    return renderBrowser(target, venv);
  }
  env.redraw = draw;

  const out = draw();
  if (out && typeof out.then === 'function') await out.catch(err => console.error(err));
  return null;
}

// which run or snapshot build is live, from the lists the tab just read (never a stale memory of one)
function syncActive(env) {
  const run = env.runs.find(r => ['running', 'queued', 'cancelling'].includes(r.status));
  const snap = env.snaps.find(s => ['building', 'cancelling'].includes(s.status));
  AG.activeRun = run?.id || (snap ? `snapshot:${snap.id}` : null);
  AG.activeName = run ? (run.name || run.id) : snap ? snap.id : null;
}

function routeOf() {
  const p = new URLSearchParams();
  p.set('view', AG.view || 'browse');
  if (AG.src && (AG.view === 'browse' || AG.view === 'story')) p.set(AG.src.kind === 'snapshot' ? 'snap' : 'run', AG.src.id);
  if (AG.node && (AG.view === 'browse' || AG.view === 'story')) p.set('node', AG.node);
  if (AG.view === 'story' && Number.isFinite(AG.page)) p.set('q', String(AG.page));
  if (AG.view === 'story' && AG.leaf) p.set('leaf', AG.leaf);
  return p.toString();
}

function applyRoute(env, param) {
  if (!param) return;
  const p = new URLSearchParams(param.includes('=') ? param : `view=${param}`);
  const view = p.get('view');
  if (VIEWS.some(v => v.value === view)) AG.view = view;
  if (p.get('snap')) AG.src = { kind: 'snapshot', id: p.get('snap') };
  else if (p.get('run')) AG.src = { kind: 'run', id: p.get('run') };
  const node = p.get('node');
  if (node && /^[0-9a-f]{16}$/.test(node)) AG.node = node;
  else if (AG.src) AG.node = null;
  const q = Number(p.get('q'));
  if (Number.isFinite(q) && q >= 0) AG.page = q;
  const leaf = p.get('leaf');
  AG.leaf = leaf && /^[0-9a-f]{16}$/.test(leaf) ? leaf : null;
  if (AG.src) {
    const known = AG.src.kind === 'snapshot' ? env.snaps.some(s => s.id === AG.src.id) : env.runs.some(r => r.id === AG.src.id);
    if (!known) { env.missing = AG.src; AG.src = null; AG.node = null; }
  }
}

function defaultView(env) {
  if (env.snaps.some(s => s.status === 'done') || env.runs.length) return 'browse';
  return 'setup';
}
function defaultSource(env) {
  const done = env.snaps.filter(s => s.status === 'done').sort((a, b) => String(b.finished || b.created).localeCompare(String(a.finished || a.created)));
  if (done.length) return { kind: 'snapshot', id: done[0].id };
  if (env.snaps.length) return { kind: 'snapshot', id: env.snaps[0].id };
  const run = env.runs.find(r => r.status === 'done') || env.runs[0];
  return run ? { kind: 'run', id: run.id } : null;
}

function snapLabel(s) {
  const st = { done: '完成', building: '构建中', paused: '已暂停' }[s.status] || s.status || '';
  return `快照 ${s.id} · 束宽 ${s.B ?? '—'} × ${s.depth ?? '—'} 季 · ${st}${s.stub ? ' · 桩模型' : ''}`;
}
function runLabel(r) {
  return `推演 ${r.name || r.id} · ${r.depth ?? '—'} 季 · 束宽 ${r.B ?? '—'} · ${STATUS_ZH[r.status] || r.status || ''}`;
}
const STATUS_ZH = { queued: '排队中', running: '运行中', cancelling: '正在停止', paused: '已暂停', done: '完成', failed: '失败', cancelled: '已取消' };

function sourcePicker(env, onChange) {
  const opts = [];
  const sel = h('select', { 'aria-label': '选择快照或推演',
    onchange: (e) => {
      const [kind, ...rest] = e.target.value.split(':');
      AG.src = { kind, id: rest.join(':') };
      AG.node = null; AG.page = null; AG.leaf = null;
      onChange();
    } });
  if (env.snaps.length) {
    C.put(sel, h('optgroup', { label: '快照（冻结，只读）' }, env.snaps.map(s => h('option', { value: `snapshot:${s.id}` }, snapLabel(s)))));
  }
  if (env.runs.length) {
    C.put(sel, h('optgroup', { label: '推演' }, env.runs.map(r => h('option', { value: `run:${r.id}` }, runLabel(r)))));
  }
  if (AG.src) sel.value = `${AG.src.kind}:${AG.src.id}`;
  opts.push(sel);
  if (!env.snaps.length && !env.runs.length) return h('span');
  return h('div', { class: 'agents-src' }, h('label', {}, '查看'), ...opts);
}

function drawBanner(env, el) {
  C.fill(el);
  if (env.missing) {
    C.put(el, env.ctx.ui.notice(`没有找到${env.missing.kind === 'snapshot' ? '快照' : '推演'}「${env.missing.id}」（可能已被删除），下面显示其余的。`, 'error'));
    env.missing = null;
  }
  if (AG.activeRun && AG.view !== 'setup') {
    const snap = AG.activeRun.startsWith('snapshot:');
    C.put(el, h('div', { class: 'notice info', style: { display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' } },
      h('span', { class: 'agents-statusdot live' }), snap ? `快照「${AG.activeName}」正在构建。` : `推演「${AG.activeName || AG.activeRun}」正在运行。`,
      h('button', { class: 'btn', type: 'button', onclick: () => env.go('setup') }, '查看进度')));
  }
}

function emptyState(env) {
  const step = (cls, t, d) => h('div', { class: `agents-step ${cls}` }, h('b', {}, t), d);
  const arrow = () => h('span', { class: 'agents-arrow' }, '→');
  return h('div', {},
    h('div', { class: 'agents-hero' },
      h('h2', {}, '还没有快照或推演。先在「新推演」里设置并开始，或用命令行构建一个快照。'),
      h('div', { class: 'agents-flow' },
        step('code', '起点状态', '2026Q3 末的真实数据与锁定事实'), arrow(),
        step('llm', '行为者各出 2–4 个动作', '大模型给概率，与历史基率混合'), arrow(),
        step('llm', '后果合成器', '模型自身分布的分区或有上限的机制'), arrow(),
        step('engine', '季度资产模型', '季末联合分布（历史估计）'), arrow(),
        step('code', '下一季起点', '账户按恒等式更新')),
      h('p', { style: { margin: '10px 0 0', fontSize: '13px', color: 'var(--ink-2)' } }, '快照命令（先看估计，不调用大模型）：',
        h('code', { class: 'agents-cmd' }, `python -m scenariolab.agents.snapshot --beam ${env.cfg.snapshot_defaults?.B ?? 2} --quarters 80 --dry-run`))),
    h('button', { class: 'btn primary', type: 'button', onclick: () => env.go('setup') }, '去设置新推演'));
}

// ================================================================ setup: pinned facts, settings, estimate, progress
const ENUM_ZH = { hold: '维持利率', cut_25: '降息 25bp', cut_50: '降息 50bp', hike_25: '加息 25bp', in_force: '仍然有效',
  pending: '待最高法院裁决', struck_down: '被裁定无效（不退税）', struck_down_refunds: '被裁定无效并退税', unknown: '未知（尚未揭晓）',
  house_holds: '执政党保住众议院', house_flips: '执政党失去众议院', none: '没有待决裁决', pending_this_quarter: '本季有待决裁决' };
const FACT_STATUS = { confirmed: ['已确认', 'good'], unconfirmed: ['未确认', 'warn'], default: ['默认', ''] };

async function renderSetup(root, env) {
  C.put(root, h('div', { class: 'section-title' }, '锁定事实与制度现状'));
  const factsBox = h('div');
  const accBox = h('div', { style: { marginTop: '12px' } });
  C.put(root, factsBox, accBox);
  renderFacts(factsBox, env);
  renderAccountsSnapshot(accBox, env);

  C.put(root, h('div', { class: 'section-title' }, '运行设置与预计用量'));
  const grid = h('div', { class: 'grid cols-3' });
  C.put(root, grid);
  const progressBox = h('div', { style: { marginTop: '16px' } });
  C.put(root, progressBox);
  const snapBox = h('div', { style: { marginTop: '16px' } });
  C.put(root, snapBox);
  renderSettings(grid, env, progressBox);
  if (AG.activeRun || AG.lastRun) renderProgress(progressBox, env, AG.activeRun || AG.lastRun);
  renderSnapshotBuilder(snapBox, env);
}

// the actor grid's activity notes read the pinned facts: redraw them when the facts arrive or change
function factsChanged(env) {
  try { env.redrawActors?.(); } catch (err) { console.error(err); }
}

// ---------------------------------------------------------------- A0 the root account snapshot: data dates, stale rows
const ACCOUNTS_CMD = 'python -m scenariolab.build --only agents_accounts';
async function renderAccountsSnapshot(box, env) {
  const { ui } = env.ctx;
  let snap;
  try {
    snap = await D.get('agents/accounts/root', { signal: env.signal });
  } catch (err) {
    if (err.name === 'AbortError' || !env.alive()) return;
    C.fill(box, ui.card({ title: '起点账户', sub: '各行为者的资产负债表、利润表与现金流量表（真实数据）' },
      ui.notice(`还没有起点账户（${D.errorText(err)}）：推演会在没有账户块的情况下进行。`, ''),
      h('div', { style: { fontSize: '13px', marginTop: '6px' } }, '构建命令：', h('code', { class: 'agents-cmd' }, ACCOUNTS_CMD))));
    return;
  }
  if (!env.alive()) return;
  const dates = [];
  for (const [book, b] of Object.entries(snap.books || {})) {
    for (const [lid, ln] of Object.entries(b.lines || {})) if (ln?.as_of) dates.push({ book, lid, as_of: String(ln.as_of) });
  }
  dates.sort((a, b) => a.as_of.localeCompare(b.as_of));
  const latest = dates.length ? dates[dates.length - 1].as_of : '—';
  const oldest = dates.length ? dates[0].as_of : '—';
  const stale = (snap.stale || []).map(x => (typeof x === 'string' ? x : x.key || x.line || JSON.stringify(x)));
  const missing = snap.missing || [];
  C.fill(box, ui.card({
    title: '起点账户',
    sub: '推演开始时各行为者看到的三张表（真实数据，按数据日期）；之后每个粒子的账户按它自己的动作与模型的季度结果更新。',
    actions: h('div', { class: 'agents-chips' }, stale.length ? C.chip(`未刷新 ${stale.length} 行`, 'warn') : C.chip('全部刷新', 'good'),
      missing.length ? C.chip(`缺 ${missing.length} 行`, 'warn') : null),
  },
  h('dl', { class: 'agents-kv' },
    h('dt', {}, '数据截至'), h('dd', {}, `${latest}（最早的一行 ${oldest}）`),
    h('dt', {}, '构建于'), h('dd', {}, snap.as_of_run || '—'),
    h('dt', {}, '行数'), h('dd', {}, `${dates.length} 行，${Object.keys(snap.books || {}).length} 本账`)),
  stale.length ? h('details', { style: { marginTop: '6px' } }, h('summary', {}, `未刷新的行（下载失败，用的是缓存）：${stale.length} 行`),
    h('ul', { class: 'list', style: { fontSize: '12px' } }, stale.slice(0, 40).map(t => h('li', {}, C.accountLine(t)?.[0] || C.humanize(t))))) : null,
  missing.length ? h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '6px' } }, `缺少的行：${missing.slice(0, 12).map(t => C.accountLine(t)?.[0] || t).join('、')}`) : null,
  h('div', { style: { fontSize: '13px', marginTop: '8px' } }, '刷新命令（下载 FRED、财政部、纽约联储、SEC 等免费数据）：', h('code', { class: 'agents-cmd' }, ACCOUNTS_CMD))));
}

// ---------------------------------------------------------------- A0 pinned facts
async function renderFacts(box, env) {
  const { ui } = env.ctx;
  C.fill(box, ui.loading('读取锁定事实…'));
  let items;
  try {
    items = (await D.get('agents/institutions', { signal: env.signal })).items || [];
  } catch (err) {
    if (err.name === 'AbortError') return;
    C.fill(box, ui.notice(`锁定事实读取失败：${D.errorText(err)}`, 'error'));
    return;
  }
  if (!env.alive()) return;
  AG.facts = items;
  drawFacts(box, env);
  factsChanged(env);
}

function factValueText(it, v) {
  if (v === null || v === undefined || v === '') return '（空）';
  if (it.type === 'enum') return ENUM_ZH[v] || String(v);
  if (it.type === 'bool') return v ? '是' : '否';
  if (it.type === 'number') return `${v}${it.unit ? ` ${it.unit}` : ''}`;
  return String(v);
}

function drawFacts(box, env) {
  const { ui } = env.ctx;
  const items = AG.facts || [];
  const edits = AG.factEdits;
  const unconf = items.filter(i => i.status !== 'confirmed');
  const reqUn = unconf.filter(i => i.required);
  const msg = h('span', { class: 'agents-msg' });
  const saveBtn = h('button', { class: 'btn primary', type: 'button', disabled: !Object.keys(edits).length },
    Object.keys(edits).length ? `保存修改（${Object.keys(edits).length} 项）` : '保存修改');
  const save = async (payload, label) => {
    saveBtn.disabled = true;
    msg.className = 'agents-msg';
    C.fill(msg, h('span', { class: 'spinner' }), '保存中…');
    try {
      const r = await D.put('agents/institutions', { edits: payload });
      AG.facts = r.items || AG.facts;
      for (const k of Object.keys(payload)) { delete AG.factEdits[k]; AG.factOpen.delete(k); }
      drawFacts(box, env);
      factsChanged(env);
      const m = box.querySelector('.agents-msg');
      if (m) { m.className = 'agents-msg ok'; m.textContent = `${label}：已保存，状态改为已确认。新的推演从这些事实出发（已有的推演与快照保留它们开始时的事实）。`; }
    } catch (err) {
      msg.className = 'agents-msg err';
      msg.textContent = `没有保存：${D.errorText(err)}`;
      saveBtn.disabled = false;
    }
  };
  saveBtn.onclick = () => save({ ...edits }, `${Object.keys(edits).length} 项修改`);

  const row = (it) => {
    const has = Object.prototype.hasOwnProperty.call(edits, it.id);
    const cur = has ? edits[it.id] : it.value;
    const set = (v) => { AG.factEdits[it.id] = v; drawFacts(box, env); };
    const open = has || AG.factOpen.has(it.id);
    if (!open) {
      // one line per fact: the value it will use, its status, and the two actions
      const [stText, stKind] = FACT_STATUS[it.status] || ['未确认', 'warn'];
      return h('div', { class: `agents-fact compact ${it.status === 'confirmed' ? 'confirmed' : 'unconfirmed'}`,
        title: it.status === 'confirmed' ? '' : `默认：${it.default_reason || it.suggested?.why || ''}` },
        h('div', {}, h('div', { class: 't' }, it.zh), h('div', { class: 'val' }, `${it.status === 'confirmed' ? '已确认' : '按默认'}：${factValueText(it, it.value)}`)),
        h('div', { class: 'st' }, it.required ? C.chip('必填', '') : null, C.chip(stText, stKind),
          h('button', { class: 'btn', type: 'button', onclick: () => { AG.factOpen.add(it.id); drawFacts(box, env); } }, '修改'),
          it.status !== 'confirmed' ? h('button', { class: 'btn', type: 'button', title: '不改值，只把这一项标为已确认',
            onclick: () => save({ [it.id]: it.value }, `「${it.zh}」`) }, '确认当前值') : null));
    }
    let ed;
    if (it.type === 'enum') {
      ed = h('select', { 'aria-label': it.zh, onchange: (e) => set(e.target.value) },
        (it.options || []).map(o => h('option', { value: o, selected: o === cur ? true : null }, ENUM_ZH[o] || C.optionLabel('', o))));
    } else if (it.type === 'bool') {
      ed = h('label', { class: 'inl' }, h('input', { type: 'checkbox', checked: cur ? true : null, onchange: (e) => set(e.target.checked) }), cur ? '是' : '否');
    } else if (it.type === 'number') {
      ed = h('span', { class: 'ed' }, h('input', { type: 'number', step: 'any', 'aria-label': it.zh, value: cur ?? '', placeholder: '未填',
        onchange: (e) => set(e.target.value === '' ? null : Number(e.target.value)) }), it.unit ? h('span', { class: 'muted' }, it.unit) : null);
    } else if (it.type === 'date') {
      ed = h('input', { type: 'date', 'aria-label': it.zh, value: cur || '', onchange: (e) => set(e.target.value || null) });
    } else {
      ed = h('input', { type: 'text', 'aria-label': it.zh, value: cur ?? '', placeholder: '未填', onchange: (e) => set(e.target.value) });
    }
    const [stText, stKind] = FACT_STATUS[it.status] || ['未确认', 'warn'];
    const affects = (it.affects || []).map(a => C.actorName(a)).join('、');
    return h('div', { class: `agents-fact ${it.status === 'confirmed' ? 'confirmed' : 'unconfirmed'} ${has ? 'edited' : ''}` },
      h('div', {}, h('div', { class: 't' }, it.zh), h('div', { class: 'muted', style: { fontSize: '12px' } },
        `影响：${affects || '—'}${it.visibility && it.visibility !== 'public' ? '（只有该行为者可见）' : ''}`)),
      h('div', { class: 'st' }, it.required ? C.chip('必填', '') : null, C.chip(has ? '已修改，未保存' : stText, has ? 'accent' : stKind),
        it.status !== 'confirmed' && !has ? h('button', { class: 'btn', type: 'button', title: '不改值，只把这一项标为已确认',
          onclick: () => save({ [it.id]: it.value }, `「${it.zh}」`) }, '确认当前值') : null,
        has ? h('button', { class: 'btn', type: 'button', onclick: () => { delete AG.factEdits[it.id]; AG.factOpen.delete(it.id); drawFacts(box, env); } }, '撤销')
          : h('button', { class: 'btn', type: 'button', onclick: () => { AG.factOpen.delete(it.id); drawFacts(box, env); } }, '收起')),
      h('div', { class: 'ed' }, ed),
      h('div', { class: 'why' }, it.status === 'confirmed' ? `已确认${it.edited_at ? `（${it.edited_at}）` : ''}：${factValueText(it, it.value)}` : `默认：${it.default_reason || it.suggested?.why || '—'}`,
        it.source ? ` · 来源：${it.source}` : '', it.date ? ` · ${it.date}` : ''));
  };
  const required = items.filter(i => i.required);
  const optional = items.filter(i => !i.required);
  C.fill(box, ui.card({
    title: '锁定事实与制度现状',
    sub: '推演开始时各行为者看到的制度现状，只由你编辑，从不由大模型编辑。未确认的项按“默认”一行处理；改动后保存，状态变为已确认。',
    actions: h('div', { class: 'agents-chips' },
      reqUn.length ? C.chip(`必填未确认 ${reqUn.length} 项`, 'warn') : C.chip('必填项都已确认', 'good'),
      C.chip(`未确认 ${unconf.length} / ${items.length}`, unconf.length ? 'warn' : 'good')),
  },
  reqUn.length ? ui.notice(`有 ${reqUn.length} 项必填事实还是未确认：推演会按默认值进行，并在结果里标注。`, '') : null,
  h('div', { class: 'agents-facts', style: { marginTop: '10px' } }, required.map(row)),
  optional.length ? h('details', { style: { marginTop: '10px' } }, h('summary', {}, `可选事实（${optional.length} 项）`),
    h('div', { class: 'agents-facts', style: { marginTop: '8px' } }, optional.map(row))) : null,
  h('div', { class: 'agents-actions' }, saveBtn, msg)));
}

// ---------------------------------------------------------------- A settings + estimate
function defaultDraft(cfg) {
  const d = cfg.defaults || {};
  return { name: '', depth: d.depth ?? 8, B: d.B ?? 4, K: d.K ?? 3, M_max: d.M_max ?? 3, actors: null,
    synth: { mode: d.synth?.mode || 'gated', depth: d.synth?.depth ?? 4, axes: { ...(d.synth?.axes || {}) } },
    c_scale: null, stress_mode: !!d.stress_mode, n_particles: d.n_particles ?? 500, seed: d.seed ?? 7,
    max_calls: d.max_calls ?? 60, concurrency: d.concurrency ?? 3, prior_trees: d.prior_trees !== false };
}
function settingsOf(dr) {
  const out = { depth: dr.depth, B: dr.B, K: dr.K, M_max: dr.M_max, synth: dr.synth, stress_mode: dr.stress_mode,
    n_particles: dr.n_particles, seed: dr.seed, max_calls: dr.max_calls, concurrency: dr.concurrency, prior_trees: dr.prior_trees };
  if (Array.isArray(dr.actors)) out.actors = dr.actors;
  if (C.isNum(dr.c_scale)) out.c_scale = dr.c_scale;
  return out;
}

function activityNote(a, env) {
  const ins = AG.facts || [];
  const v = (id) => ins.find(i => i.id === id)?.value;
  if (a.id === 'voter') {
    const e = C.elections()[env.cfg.starts?.today?.quarter || '2026Q4'];
    return e ? `本季：${e.zh}` : '只在联邦选举季度';
  }
  if (a.id === 'courts') return v('ruling_tariff_quarter') || v('ruling_fed_removal_quarter') ? '有待决裁决' : '没有待决裁决：不行动';
  if (a.id === 'belligerents') return v('belligerents_active') === false ? '石油供给中断未开启：不行动' : '石油供给中断持续：开启';
  if (a.id === 'treasury') return '输入表没有短债占比字段：不行动';
  return a.note ? C.cut(a.note, 40) : '每季行动';
}

function renderSettings(grid, env, progressBox) {
  const { ui } = env.ctx;
  const cfg = env.cfg;
  if (!AG.draft) AG.draft = defaultDraft(cfg);
  const dr = AG.draft;
  const limits = cfg.limits || {};
  const estBox = h('div');
  const runEstimate = C.debounce(() => estimate(), 400);
  const changed = () => runEstimate();

  const numIn = (key, lo, hi, step = 1, label, sub) => {
    const inp = h('input', { type: 'number', min: lo, max: hi, step, value: dr[key], 'aria-label': label,
      onchange: (e) => { const x = Number(e.target.value); if (Number.isFinite(x)) { dr[key] = C.clamp(Math.round(x / step) * step, lo, hi); e.target.value = dr[key]; changed(); } } });
    return h('label', { class: 'inl' }, label, inp, sub ? h('span', { class: 'muted', style: { fontSize: '12px' } }, sub) : null);
  };
  const depthRange = h('input', { type: 'range', min: 1, max: limits.max_depth || 80, value: dr.depth, 'aria-label': '推演深度',
    oninput: (e) => { dr.depth = Number(e.target.value); depthNum.value = dr.depth; depthSpan.textContent = span(); changed(); } });
  const depthNum = h('input', { type: 'number', min: 1, max: limits.max_depth || 80, value: dr.depth, 'aria-label': '推演深度（季）',
    onchange: (e) => { dr.depth = C.clamp(Math.round(Number(e.target.value) || 1), 1, limits.max_depth || 80); e.target.value = dr.depth; depthRange.value = dr.depth; depthSpan.textContent = span(); changed(); } });
  const q0 = cfg.starts?.today?.quarter || '2026Q4';
  const qAdd = (q, k) => { const m = /^(\d{4})Q([1-4])$/.exec(q); if (!m) return q; const t = Number(m[1]) * 4 + Number(m[2]) - 1 + k; return `${Math.floor(t / 4)}Q${(t % 4) + 1}`; };
  const span = () => `${q0} – ${qAdd(q0, dr.depth - 1)}`;
  const depthSpan = h('span', { class: 'muted', style: { fontSize: '12px' } }, span());

  // actors grouped by kind
  const actors = (cfg.catalogue?.actors || []);
  const allIds = actors.map(a => a.id);
  const isOn = (id) => !Array.isArray(dr.actors) || dr.actors.includes(id);
  const toggle = (id, on) => {
    const cur = new Set(Array.isArray(dr.actors) ? dr.actors : allIds);
    if (on) cur.add(id); else cur.delete(id);
    dr.actors = cur.size === allIds.length ? null : allIds.filter(x => cur.has(x));
    drawActors();
    changed();
  };
  const actorBox = h('div');
  const drawActors = () => {
    const groups = new Map();
    for (const a of actors) { const k = a.kind || 'policy'; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(a); }
    C.fill(actorBox, ...[...groups].map(([k, list]) => h('div', { style: { marginBottom: '6px' } },
      h('div', { class: 'muted', style: { fontSize: '12px', margin: '2px 0 4px' } }, C.KIND_ZH[k] || '其他'),
      h('div', { class: 'agents-actorgrid' }, list.map(a => {
        const on = isOn(a.id);
        const dot = h('span', { class: 'dot' });
        dot.style.background = C.actorColor(a.id);
        return h('label', { class: `agents-actor ${on ? '' : 'off'}`, title: a.note || '' },
          h('input', { type: 'checkbox', checked: on ? true : null, onchange: (e) => toggle(a.id, e.target.checked) }), dot,
          h('span', {}, h('b', {}, a.zh), h('small', {}, activityNote(a, env))));
      })))));
  };
  drawActors();
  env.redrawActors = drawActors;

  const synthSeg = ui.seg([{ value: 'off', label: '关闭' }, { value: 'gated', label: '门控（默认）' }, { value: 'every', label: '每个节点' }],
    dr.synth.mode, (v) => { dr.synth.mode = v; changed(); });
  const axisBox = (key, label) => h('label', { class: 'inl' }, h('input', { type: 'checkbox', checked: dr.synth.axes[key] ? true : null,
    onchange: (e) => { dr.synth.axes[key] = e.target.checked; changed(); } }), label);
  const cIn = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: C.isNum(dr.c_scale) ? dr.c_scale : 1, 'aria-label': '大模型权重倍数',
    oninput: (e) => { dr.c_scale = Number(e.target.value); cOut.textContent = cText(); changed(); } });
  const cText = () => (C.isNum(dr.c_scale) ? `× ${dr.c_scale.toFixed(2)}${dr.c_scale === 0 ? '（行为者只用历史基率，不调用大模型；后果合成器另算）' : ''}` : '× 1（各行为者的默认权重）');
  const cOut = h('span', { class: 'muted', style: { fontSize: '12px' } }, cText());

  const form = h('div', { class: 'agents-form' },
    h('div', {}, h('div', { class: 'agents-sec' }, '起点'),
      h('div', { style: { fontSize: '13px' } }, `今天：${cfg.starts?.today?.root_quarter || '2026Q3'} 末的状态（数据截至 ${cfg.starts?.today?.data_end || '2026-09'}），第一季推演 ${q0}。`,
        h('span', { class: 'muted' }, ` ${q0} 已过去约一个月，这部分真实数据没有纳入。`))),
    h('div', {}, h('div', { class: 'agents-sec' }, '推演深度与分叉'),
      h('div', { class: 'row' }, h('label', { class: 'inl' }, '深度', depthNum, '季'), depthRange, depthSpan),
      h('div', { class: 'row', style: { marginTop: '6px' } },
        numIn('B', 1, limits.max_B || 30, 1, '每层保留', '个节点'), numIn('K', 1, limits.max_K || 4, 1, '每个节点', '个动作簇'),
        numIn('M_max', 1, limits.max_M || 4, 1, '每簇最多', '个后果分支')),
      h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '4px' } },
        '束搜索：每层把候选节点按概率与距离合并，保留 B 个；被剪掉的质量和粒子并入最近的保留节点，不会丢掉。')),
    h('div', {}, h('div', { class: 'agents-sec' }, '行为者（关掉的不做决策，它负责的字段交给模型按历史一致性推断）'), actorBox),
    h('div', {}, h('div', { class: 'agents-sec' }, '后果合成器'),
      h('div', { class: 'row' }, synthSeg, h('label', { class: 'inl' }, '只在前', h('input', { type: 'number', min: 0, max: 80, value: dr.synth.depth,
        'aria-label': '后果合成器只在前几季调用',
        onchange: (e) => { dr.synth.depth = C.clamp(Math.round(Number(e.target.value) || 0), 0, 80); e.target.value = dr.synth.depth; changed(); } }), '季调用')),
      h('div', { class: 'row', style: { marginTop: '6px' } }, axisBox('fin_conditions', '金融条件分段'), axisBox('funding', '融资市场分段'), axisBox('equity_leg', '股票腿分段'))),
    h('div', {}, h('div', { class: 'agents-sec' }, '大模型权重（判断值）'),
      h('div', { class: 'row' }, cIn, cOut),
      h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '4px' } }, '每个行为者的概率 = 权重 × 大模型判断 + (1 − 权重) × 历史基率。'
        + '×0 时行为者不调用大模型；后果合成器有自己的权重，仍会调用，把它也关闭才等于历史基率树。')),
    h('div', {}, h('div', { class: 'agents-sec' }, '开局叙事（可选）'),
      h('textarea', { disabled: true, rows: 2, placeholder: '例如：“2026Q4 美国对华关税再提高 25 个百分点……”', style: { minHeight: '56px' } }),
      h('div', { class: 'muted', style: { fontSize: '12px' } }, '推演引擎还没有接入开局叙事（把一段文字翻译成第一季的事实与动作）：这一栏暂不可用，开始推演时不会发送任何文字。')),
    h('details', {}, h('summary', {}, '高级'),
      h('div', { class: 'row', style: { marginTop: '8px' } },
        numIn('n_particles', 50, 5000, 50, '每节点粒子', '个'), numIn('seed', 0, 999999, 1, '种子', ''),
        numIn('max_calls', 1, 5000, 1, '新调用上限', '次'), numIn('concurrency', 1, 8, 1, '并发', '个'),
        h('label', { class: 'inl' }, h('input', { type: 'checkbox', checked: dr.stress_mode ? true : null, onchange: (e) => { dr.stress_mode = e.target.checked; changed(); } }), '压力模式'),
        h('label', { class: 'inl' }, h('input', { type: 'checkbox', checked: dr.prior_trees ? true : null, onchange: (e) => { dr.prior_trees = e.target.checked; changed(); } }), '同时生成两棵对照树（0 次调用）'))),
    h('div', { class: 'row' }, h('label', { class: 'inl' }, '名称', h('input', { type: 'text', value: dr.name, placeholder: '可选',
      oninput: (e) => { dr.name = e.target.value.slice(0, 80); } }))),
    h('div', { class: 'agents-chips' }, C.chip('两步法条件化', '', '季度模型还不能按结构条件化：先跑一次不含行为者杠杆的模型，再把动作写入的字段固定'),
      env.stub ? C.chip(cfg.model?.label || '桩模型（不是资产模型）', 'warn') : null));

  C.put(grid, ui.card({ title: '运行设置', sub: '每一项改动都会立刻重算右侧的预计用量（不调用大模型）。', cls: 'span-2' }, form),
    ui.card({ title: '预计用量', sub: '运行前估算：调用次数、轮次、时间与额度，不调用大模型。' }, estBox));

  async function estimate() {
    C.fill(estBox, ui.loading('估算中…'));
    let est;
    try {
      est = await D.post('agents/estimate', { settings: settingsOf(dr) }, { signal: env.signal });
    } catch (err) {
      if (err.name === 'AbortError' || !env.alive()) return;
      C.fill(estBox, h('div', { class: 'agents-msg err' }, D.errorText(err)));
      return;
    }
    if (!env.alive()) return;
    drawEstimate(estBox, env, est, dr, progressBox);
  }
  estimate();
}

function drawEstimate(box, env, est, dr, progressBox) {
  const { ui } = env.ctx;
  const llm = env.llm || {};
  const real = !env.mock;
  const lat = est.latency_s || [];
  const lv = (est.per_level || []).map((p, i) => {
    const sp = h('span', { class: p.synth_calls ? 'synth' : '', title: `${p.quarter}：${p.nodes} 个节点，行为者调用 ${p.actor_calls}、后果合成 ${p.synth_calls}（${p.mode === 'separate' ? '分开调用' : '合并调用'}）` }, String(i + 1));
    return sp;
  });
  const startMsg = h('span', { class: 'agents-msg' });
  const confirmBox = h('div');
  const startBtn = h('button', { class: 'btn primary', type: 'button', disabled: (!env.mock && !llm.available) || !!AG.activeRun },
    env.mock ? '开始推演（模拟后端，不消耗额度）' : '开始推演…');
  const start = async (confirm) => {
    startBtn.disabled = true;
    startMsg.className = 'agents-msg';
    C.fill(startMsg, h('span', { class: 'spinner' }), '提交中…');
    try {
      const r = await D.post('agents/runs', { settings: settingsOf(dr), name: dr.name || undefined, confirm: !!confirm });
      AG.activeRun = r.run_id; AG.lastRun = r.run_id;
      startMsg.className = 'agents-msg ok';
      startMsg.textContent = `已开始：${r.run_id}`;
      C.fill(confirmBox);
      await env.refreshLists();
      renderProgress(progressBox, env, r.run_id);
      progressBox.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    } catch (err) {
      startBtn.disabled = false;
      startMsg.className = 'agents-msg err';
      if (err.status === 409 && err.body?.active_run) {
        startMsg.textContent = '';
        AG.activeRun = err.body.active_run;
        C.put(startMsg, `已有一个推演在运行（${err.body.active_run}）：同一时间只能运行一个。`);
        renderProgress(progressBox, env, err.body.active_run);
      } else if (err.status === 409 && err.body?.needs_confirm) {
        startMsg.textContent = '这次推演需要你确认用量。';
        showConfirm(err.body.estimate || est);
      } else {
        startMsg.textContent = D.errorText(err);
      }
    }
  };
  const showConfirm = (e) => {
    const ok = h('input', { type: 'checkbox' });
    const go = h('button', { class: 'btn primary', type: 'button', disabled: true }, '确认开始');
    ok.onchange = () => { go.disabled = !ok.checked; };
    go.onclick = () => start(true);
    C.fill(confirmBox, h('div', { class: 'agents-confirm', role: 'alertdialog', 'aria-label': '确认用量' },
      h('b', {}, `这次推演会调用本机 Claude Code 最多约 ${e.calls_new} 次（${e.model || 'claude-opus-5-5'}，力度 medium），${C.minutesRange(e.minutes)}；达到 ${e.hard_cap ?? '—'} 次时暂停。`),
      h('div', { style: { marginTop: '4px' } }, e.usage_note || ''),
      h('label', {}, ok, h('span', {}, '我确认：这会计入我的 Claude 订阅用量。')),
      h('div', { class: 'agents-actions', style: { marginTop: '4px' } }, go,
        h('button', { class: 'btn', type: 'button', onclick: () => C.fill(confirmBox) }, '取消'))));
  };
  startBtn.onclick = () => (env.mock ? start(false) : showConfirm(est));

  const kv = (k, v) => [h('dt', {}, k), h('dd', {}, v)];
  C.fill(box, 
    h('div', { class: 'agents-big' }, `最多约 ${est.calls_new} 次新调用`),
    h('div', { class: 'muted', style: { fontSize: '12px' } }, `上限估计：行为者 ${est.actor_calls} 次，后果合成 ${est.synth_calls} 次`,
      est.calls_cached ? `；已有 ${est.calls_cached} 次在缓存里（前 ${est.cached_levels} 季与已保存的推演或快照相同），不再调用` : '',
      est.actors_ask === false ? '；大模型权重 ×0：行为者不调用大模型' : ''),
    est.over_cap ? ui.notice(`超过新调用上限 ${est.calls_cap} 次：推演会在达到上限后停下，剩余节点只用历史基率。`, '') : null,
    h('dl', { class: 'agents-kv', style: { marginTop: '10px' } },
      kv('预计耗时', `${C.minutesRange(est.minutes)}`),
      kv('轮次', `${est.rounds} 轮（并发 ${est.concurrency}）`),
      kv('单次用时', `${lat[0] ?? '—'}–${lat[lat.length - 1] ?? '—'} 秒（${est.latency_basis === 'measured' ? '实测' : '假设'}）`),
      kv('token', `输入约 ${(est.tokens?.input / 1e6).toFixed(2)}M，输出约 ${(est.tokens?.output / 1e6).toFixed(2)}M（假设的大小）`),
      kv('模型', `${est.model || 'claude-opus-5-5'}，力度 ${est.effort || 'medium'}（每一步都相同）`),
      kv('硬上限', `${est.hard_cap ?? '—'} 次新调用（预计的 1.2 倍）：达到后暂停，确认后才继续`),
      kv('后端', env.mock ? '模拟后端：所有调用由本地模拟回答，不调用模型' : `本机 Claude Code（${llm.model || '—'}）`)),
    h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '8px' } }, '每季展开的节点（橙色底边 = 本季调用后果合成器）：'),
    h('div', { class: 'agents-levels' }, lv),
    h('details', { style: { marginTop: '10px' } }, h('summary', {}, '用量说明'),
      h('p', { style: { fontSize: '12px', color: 'var(--ink-2)', margin: '6px 0 0' } },
        '推演与后果合成通过本机 Claude Code 的非交互模式（claude -p）调用大模型，计入你的 Claude 订阅用量，与你平时在 Claude / Claude Code 里的使用共用额度，不另外计费。',
        '达到用量上限时推演会暂停：已完成的调用保存在本地缓存里，额度恢复后继续不会重复计入。相同设置重跑全部命中缓存，不调用大模型。'),
      h('p', { class: 'muted', style: { fontSize: '12px', margin: '6px 0 0' } }, est.basis || '')),
    !env.mock && !llm.available ? ui.notice(llm.reason || '本机 Claude Code 不可用：安装并登录后刷新本页；已保存的推演和快照仍可查看。', 'error') : null,
    !env.mock && llm.limited ? ui.notice(`Claude 订阅的用量暂时达到上限：现在开始会立即暂停。${llm.limited_reason || ''}`, '') : null,
    h('div', { class: 'agents-actions' }, startBtn, startMsg), confirmBox);
}

// ---------------------------------------------------------------- B progress
const PROG = { es: null, timer: 0, rid: null };
const CALL_KIND = { actor: '行为者（分开）', node: '合并', merged: '合并', synth: '后果合成', repair: '修复', story: '补写叙事', p4: '开局叙事' };

function stopProgress() {
  if (PROG.es) { try { PROG.es.close(); } catch (_) { /* gone */ } PROG.es = null; }
  clearInterval(PROG.timer); PROG.timer = 0;
}

async function renderProgress(box, env, rid) {
  const { ui } = env.ctx;
  stopProgress();
  PROG.rid = rid;
  env.signal?.addEventListener('abort', stopProgress, { once: true });
  const st = { seq: 0, calls: [], newCalls: 0, cached: 0, failed: 0, budget: 0, retries: 0, levels: 0, status: 'queued', t0: null, pause: null, run: null, nodes: 0 };
  const bar = h('span', { style: { width: '0%' } });
  const head = h('div', { class: 'agents-msg' });
  const counts = h('div', { style: { fontSize: '13px', marginTop: '6px' } });
  const pauseBox = h('div');
  const callsBox = h('div', { style: { marginTop: '10px' } });
  const btns = h('div', { class: 'agents-actions' });
  C.fill(box, ui.card({ title: '推演进度', sub: '每一层（季度）的节点全部完成后才开始下一层；同一层的调用并行。离开本页推演会继续，回来时自动接上。' },
    head, h('div', { class: 'agents-progress', style: { marginTop: '8px' } }, bar), counts, pauseBox, callsBox, btns));

  if (String(rid).startsWith('snapshot:')) {           // a snapshot build holds the slot: its own progress line
    const sid = String(rid).slice('snapshot:'.length);
    const msg = h('span', { class: 'agents-msg' });
    C.fill(box, ui.card({ title: '正在构建快照', sub: '同一时间只能运行一个推演或快照构建：它们共用同一份订阅额度。' }, msg));
    watchSnapshot(env, sid, msg);
    return;
  }
  let run = null;
  try { run = await D.get(`agents/runs/${rid}`); } catch (err) {
    C.fill(box, ui.notice(`读取推演失败：${D.errorText(err)}`, 'error'));
    if (err.status === 404 && AG.activeRun === rid) AG.activeRun = null;
    return;
  }
  st.run = run;
  const depth = run.settings?.depth || 1;
  const estCalls = run.estimate?.calls_new || 0;

  const paint = () => {
    if (!box.isConnected) return;
    const frac = Math.min(1, st.levels / depth);
    bar.style.width = `${(frac * 100).toFixed(1)}%`;
    const live = ['running', 'queued', 'cancelling'].includes(st.status);
    C.fill(head, live ? h('span', { class: 'agents-statusdot live' }) : null,
      `「${run.name || run.id}」 · ${STATUS_ZH[st.status] || st.status} · 第 ${Math.min(st.levels + (live ? 1 : 0), depth)} / ${depth} 季`);
    const el = st.t0 ? (Date.now() - st.t0) / 1000 : null;
    C.fill(counts, `新调用 ${st.newCalls}（这次预计最多 ${estCalls}${run.estimate?.calls_cached ? `，另有 ${run.estimate.calls_cached} 次命中缓存` : ''}） · 缓存命中 ${st.cached} · 失败 ${st.failed} · 重试 ${st.retries}`,
      st.budget ? h('span', { class: 'agents-msg', style: { marginLeft: '6px' } }, ` · 达到调用上限：${st.budget} 次只用历史基率`) : null,
      el !== null && live ? ` · 已用 ${C.seconds(el)}` : '');
    C.fill(pauseBox);
    const p = st.pause;
    if (st.status === 'paused' && p) {
      const reset = p.reset_at ? new Date(p.reset_at * 1000).toLocaleString('zh-CN', { hour12: false }) : null;
      const text = p.reason === 'usage_limit'
        ? `已达到 Claude 订阅的用量上限，推演暂停在第 ${(p.level ?? st.levels) + 1} 季。${reset ? `额度预计 ${reset} 恢复，之后会自动继续；` : 'Claude Code 没有给出恢复时间：额度恢复后点“继续推演”；'}已完成的部分不会重复计入。`
        : p.reason === 'server_restart' ? `推演在第 ${(p.level ?? st.levels) + 1} 季被中断（服务重启）。已完成的节点在缓存里，继续不会重复计入。`
          : p.reason === 'budget' ? `${p.message || '已达到这次推演的用量上限（预计调用的 1.2 倍）'}`
            : `推演已暂停：${p.message || p.reason}`;
      C.put(pauseBox, ui.notice(text, ''));
    } else if (st.status === 'failed') {
      C.put(pauseBox, ui.notice(`推演出错：${run.error || '未知错误'}。已完成的节点已保存，可以继续。`, 'error'));
    } else if (st.status === 'cancelled') {
      C.put(pauseBox, ui.notice(`已取消：完成了第 1–${st.levels} 季。已完成的节点保留在缓存里，继续推演会从第 ${st.levels + 1} 季接着算。`, 'info'));
    } else if (st.status === 'done') {
      C.put(pauseBox, ui.notice(`推演完成：新调用 ${st.newCalls} 次，缓存命中 ${st.cached} 次。${st.budget ? `达到新调用上限后，${st.budget} 次决策只用了历史基率（节点标“只用历史基率”）。` : ''}`, 'info'));
    }
    const rows = st.calls.slice(-8).reverse();
    C.fill(callsBox, rows.length ? ui.table([
      { key: 't', label: '时间', fmt: (v) => String(v || '').slice(11, 19) },
      { key: 'kind', label: '类型' },
      { key: 'state', label: '状态', fmt: (v, r) => C.chip(v, r.budget ? 'warn' : r.ok === false ? 'bad' : r.cached ? '' : 'good') },
      { key: 'sec', label: '用时', fmt: (v) => (C.isNum(v) ? `${v.toFixed(1)} 秒` : '—') },
    ], rows, { stack: true }) : h('div', { class: 'muted', style: { fontSize: '12px' } }, '等待第一次调用…'));
    callsBox.querySelector('table')?.classList.add('agents-calls');
    C.fill(btns, 
      live ? h('button', { class: 'btn', type: 'button', disabled: st.status === 'cancelling', onclick: async (e) => {
        e.target.disabled = true; e.target.textContent = '正在停止…';
        try { await D.post(`agents/runs/${rid}/cancel`, {}); } catch (err) { C.put(pauseBox, ui.notice(D.errorText(err), 'error')); }
      } }, st.status === 'cancelling' ? '正在停止…' : '取消') : null,
      ['paused', 'cancelled', 'failed'].includes(st.status) ? h('button', { class: 'btn primary', type: 'button', onclick: () => resumeRun(env, rid, box) }, '继续推演') : null,
      st.status === 'done' || st.levels > 0 ? h('button', { class: 'btn', type: 'button', onclick: () => env.go('browse', { src: { kind: 'run', id: rid }, node: null }) }, '打开结果') : null,
      st.status === 'done' ? h('button', { class: 'btn', type: 'button', onclick: () => env.go('story', { src: { kind: 'run', id: rid }, node: null, page: null }) }, '看连环画') : null);
  };

  const onEvent = (ev) => {
    if (!ev || typeof ev !== 'object') return;
    if (C.isNum(ev.seq)) st.seq = Math.max(st.seq, ev.seq);
    if (!st.t0 && ev.t) st.t0 = Date.parse(ev.t);
    switch (ev.type) {
      case 'run_started': case 'run_resumed': st.status = 'running'; break;
      case 'call_done': {
        const budget = ev.ok === false && ev.error === 'budget';
        if (ev.cached) st.cached += 1; else if (ev.ok !== false) st.newCalls += 1;
        if (budget) st.budget += 1; else if (ev.ok === false) st.failed += 1;
        if (ev.retry) st.retries += 1;
        st.calls.push({ t: ev.t, kind: CALL_KIND[ev.call] || '调用', state: budget ? '达到上限·只用历史基率' : ev.ok === false ? '失败' : ev.cached ? '缓存' : '完成',
          sec: ev.seconds, ok: ev.ok, cached: ev.cached, budget });
        if (st.calls.length > 60) st.calls.shift();
        break;
      }
      case 'level_done': st.levels = Math.max(st.levels, ev.level || 0); break;
      case 'run_paused': st.status = ev.reason === 'cancelled' ? 'cancelled' : 'paused'; st.pause = { reason: ev.reason, message: ev.message, reset_at: ev.reset_at, level: ev.level }; break;
      case 'run_done': st.levels = depth; break;
      case 'prior_trees_done': st.status = 'done'; break;
      case 'run_failed': st.status = 'failed'; run.error = ev.error; break;
      default: break;
    }
  };

  st.status = run.status;
  st.pause = run.pause;
  st.levels = run.levels_done || 0;
  let finished = false;
  const finish = async () => {
    if (finished) return;
    finished = true;
    stopProgress();
    try { const r = await D.get(`agents/runs/${rid}`); st.status = r.status; st.pause = r.pause; run.error = r.error; st.levels = r.levels_done || st.levels; } catch (_) { /* keep */ }
    if (!['running', 'queued', 'cancelling'].includes(st.status)) {
      if (AG.activeRun === rid) AG.activeRun = null;
      AG.lastRun = rid;
      D.dropTree(new D.Source('run', rid));
      await env.refreshLists();
    }
    paint();
  };
  // replay the journal once (counts), then follow live events
  try {
    const pr = await D.get(`agents/runs/${rid}/progress?since=0`);
    for (const ev of pr.events || []) onEvent(ev);
    st.status = pr.status || st.status;
    st.pause = pr.pause || st.pause;
  } catch (_) { /* the stream below replays it */ }
  paint();
  if (!['running', 'queued', 'cancelling'].includes(st.status)) {
    if (AG.activeRun === rid) { AG.activeRun = null; await env.refreshLists(); paint(); }
    return;
  }

  // Two channels: the event stream (instant) and a slow poll of the journal (a stalled or buffered stream never leaves
  // the card behind). Events carry a sequence number, so neither channel counts one twice.
  const pollOnce = async () => {
    if (!box.isConnected || env.signal?.aborted) { stopProgress(); return; }
    try {
      const pr = await D.get(`agents/runs/${rid}/progress?since=${st.seq}`);
      for (const ev of pr.events || []) if (!(C.isNum(ev.seq) && ev.seq <= st.seq)) onEvent(ev);
      st.status = pr.status; st.pause = pr.pause;
      paint();
      if (!['running', 'queued', 'cancelling'].includes(pr.status)) finish();
    } catch (_) { /* try again */ }
  };
  const poll = (ms) => { clearInterval(PROG.timer); PROG.timer = setInterval(pollOnce, ms); };
  if (typeof EventSource !== 'function') { poll(2000); return; }
  const es = new EventSource(`/api/agents/runs/${rid}/events`);
  PROG.es = es;
  poll(4000);
  let painted = 0, fails = 0;
  es.onmessage = (m) => {
    let ev = null;
    try { ev = JSON.parse(m.data); } catch (_) { return; }
    if (C.isNum(ev.seq) && ev.seq <= st.seq) return;      // already counted
    onEvent(ev);
    const now = Date.now();
    if (now - painted > 300 || ev.type !== 'call_done') { painted = now; paint(); }
  };
  es.addEventListener('end', () => finish());
  es.onerror = () => {
    fails += 1;
    if (fails >= 2) { try { es.close(); } catch (_) { /* gone */ } PROG.es = null; poll(2000); }
  };
}

async function resumeRun(env, rid, box, confirm = false) {
  const { ui } = env.ctx;
  try {
    await D.post(`agents/runs/${rid}/resume`, { confirm });
    AG.activeRun = rid;
    renderProgress(box, env, rid);
  } catch (err) {
    if (err.status === 409 && err.body?.needs_confirm) {
      const e = err.body.estimate || {};
      if (window.confirm(`继续推演会调用本机 Claude Code，最多约 ${e.calls_new ?? '—'} 次（已完成的部分命中缓存，不重复计入）。确认继续？`)) {
        resumeRun(env, rid, box, true);
      }
      return;
    }
    C.pre(box, ui.notice(`没有继续：${D.errorText(err)}`, 'error'));
  }
}

// ---------------------------------------------------------------- snapshot builder
function renderSnapshotBuilder(box, env) {
  const { ui } = env.ctx;
  if (!AG.snapDraft) AG.snapDraft = { B: env.cfg.snapshot_defaults?.B ?? 2, depth: env.cfg.snapshot_defaults?.depth ?? 80, id: '' };
  const sd = AG.snapDraft;
  const estBox = h('div');
  const msg = h('span', { class: 'agents-msg' });
  const confirmBox = h('div');
  const idIn = h('input', { type: 'text', value: sd.id || `b${sd.B}_q${sd.depth}`, 'aria-label': '快照 id', pattern: '[a-z0-9_\\-]{1,60}',
    oninput: (e) => { sd.id = e.target.value.trim(); } });
  const run = C.debounce(() => estimate(), 400);
  const nIn = (key, lo, hi, label) => h('label', { class: 'inl' }, label, h('input', { type: 'number', min: lo, max: hi, value: sd[key],
    onchange: (e) => { sd[key] = C.clamp(Math.round(Number(e.target.value) || lo), lo, hi); e.target.value = sd[key]; if (!sd.id) idIn.value = `b${sd.B}_q${sd.depth}`; run(); } }));
  const cmd = () => `python -m scenariolab.agents.snapshot --beam ${sd.B} --quarters ${sd.depth}`;
  const cmdEl = h('code', { class: 'agents-cmd' });
  const build = async (confirm) => {
    const id = (sd.id || idIn.value || '').trim();
    msg.className = 'agents-msg';
    C.fill(msg, h('span', { class: 'spinner' }), '提交中…');
    try {
      await D.post('agents/snapshots', { id, settings: { B: sd.B, depth: sd.depth }, confirm: !!confirm });
      msg.className = 'agents-msg ok';
      msg.textContent = `快照「${id}」开始构建：完成后在「快照与推演」里打开。`;
      C.fill(confirmBox);
      AG.activeRun = `snapshot:${id}`;
      AG.activeName = id;
      env.refreshLists().catch(() => {});
      watchSnapshot(env, id, msg);
    } catch (err) {
      msg.className = 'agents-msg err';
      if (err.status === 409 && err.body?.needs_confirm) {
        msg.textContent = '需要确认用量。';
        const ok = h('input', { type: 'checkbox' });
        const go = h('button', { class: 'btn primary', type: 'button', disabled: true, onclick: () => build(true) }, '确认构建');
        ok.onchange = () => { go.disabled = !ok.checked; };
        const e = err.body.estimate || {};
        C.fill(confirmBox, h('div', { class: 'agents-confirm' },
          h('b', {}, `这个快照会调用本机 Claude Code 最多约 ${e.calls_new ?? '—'} 次（${e.model || 'claude-opus-5-5'}，力度 medium），约 ${e.hours?.[0] ?? '—'}–${e.hours?.[2] ?? '—'} 小时；达到 ${e.hard_cap ?? '—'} 次时暂停。`),
          h('label', {}, ok, h('span', {}, '我确认：这会计入我的 Claude 订阅用量；达到上限时自动暂停，额度恢复后继续。')),
          h('div', { class: 'agents-actions' }, go, h('button', { class: 'btn', type: 'button', onclick: () => C.fill(confirmBox) }, '取消'))));
      } else msg.textContent = D.errorText(err);
    }
  };
  async function estimate() {
    cmdEl.textContent = `${cmd()} --dry-run`;
    C.fill(estBox, ui.loading('估算中…'));
    try {
      const e = await D.post('agents/estimate', { settings: { B: sd.B, depth: sd.depth }, snapshot: true }, { signal: env.signal });
      if (!env.alive()) return;
      const time = (e.hours?.[2] ?? 0) < 1 ? C.minutesRange(e.minutes) : `约 ${e.hours?.[0]}–${e.hours?.[2]} 小时`;
      C.fill(estBox, h('dl', { class: 'agents-kv' },
        h('dt', {}, '大模型调用'), h('dd', {}, `最多约 ${e.calls_new} 次（行为者 ${e.actor_calls}，后果合成 ${e.synth_calls}）${e.calls_cached ? `；另有 ${e.calls_cached} 次命中缓存` : ''}`),
        h('dt', {}, '模型'), h('dd', {}, `${e.model || 'claude-opus-5-5'}，力度 ${e.effort || 'medium'}（每一步都相同）`),
        h('dt', {}, '硬上限'), h('dd', {}, `${e.hard_cap ?? '—'} 次新调用（预计的 1.2 倍）：达到后暂停，确认后继续`),
        h('dt', {}, '耗时'), h('dd', {}, `${time}（${e.rounds} 轮，每次 ${e.latency_s?.[0]}–${e.latency_s?.[2]} 秒的假设）`),
        h('dt', {}, 'token'), h('dd', {}, `输入约 ${(e.tokens?.input / 1e6).toFixed(1)}M，输出约 ${(e.tokens?.output / 1e6).toFixed(1)}M（假设）`),
        h('dt', {}, '内存峰值'), h('dd', {}, `约 ${e.memory?.peak_mb ?? '—'} MB`),
        h('dt', {}, '快照大小'), h('dd', {}, `约 ${e.size?.mb ?? '—'} MB（扩展 ${e.size?.expanded_nodes ?? '—'} 个节点，剪枝 ${e.size?.pruned_nodes ?? '—'} 个）`)));
    } catch (err) {
      if (err.name !== 'AbortError') C.fill(estBox, h('div', { class: 'agents-msg err' }, D.errorText(err)));
    }
  }
  const btn = h('button', { class: 'btn', type: 'button', disabled: !!AG.activeRun }, env.mock ? '在界面里构建（模拟后端）' : '在界面里构建…');
  btn.onclick = () => build(false);
  C.fill(box, h('details', { class: 'card' },
    h('summary', { style: { cursor: 'pointer', fontWeight: 600 } }, '构建快照（束搜索，冻结后供浏览与连环画）'),
    h('div', { class: 'agents-form', style: { marginTop: '10px' } },
      h('div', { class: 'row' }, nIn('B', 1, 30, '每层保留'), nIn('depth', 1, 80, '季数'), h('label', { class: 'inl' }, '快照 id', idIn)),
      h('div', { class: 'muted', style: { fontSize: '12px' } }, '第一个真实快照先用束宽 2 × 80 季：检查之后再改成 15 或 30。'),
      estBox,
      h('div', { style: { fontSize: '13px' } }, '也可以在终端运行（先用 --dry-run 看估计；使用本机 Claude Code 时加 --yes 确认）：', h('br'), cmdEl),
      h('div', { class: 'muted', style: { fontSize: '12px' } }, '构建期间每完成一层存一个检查点；达到用量上限时暂停，额度恢复后继续。快照完成后只读，点击没有生成的分支会即时生成并存入覆盖层。'),
      h('div', { class: 'agents-actions' }, btn, msg), confirmBox)));
  estimate();
}

function watchSnapshot(env, sid, msg) {
  let t = 0, misses = 0;
  const ZH = { building: '构建中', paused: '已暂停', done: '完成', failed: '失败', cancelling: '正在停止' };
  const stop = () => clearInterval(t);
  const tick = async () => {
    if (env.signal?.aborted || !msg.isConnected) { stop(); return; }
    let s;
    try {
      s = await D.get(`agents/snapshots/${encodeURIComponent(sid)}`);
    } catch (err) {
      misses += 1;                           // the folder may not exist for a moment; a lasting 404 is an error
      if (err.status === 404 && misses < 5) return;
      stop();
      msg.className = 'agents-msg err';
      msg.textContent = `快照「${sid}」：${D.errorText(err)}`;
      return;
    }
    const m = s.manifest || {};
    const status = m.status || 'building';  // not written yet: still starting
    const done = m.levels_done || 0, depth = m.settings?.depth;
    msg.className = `agents-msg ${status === 'failed' ? 'err' : status === 'done' ? 'ok' : ''}`;
    const cancel = ['building', 'cancelling'].includes(status) ? h('button', { class: 'btn', type: 'button', disabled: status === 'cancelling',
      onclick: async (e) => {
        e.target.disabled = true;
        try { await D.post(`agents/snapshots/${encodeURIComponent(sid)}/cancel`, {}); } catch (err) { window.alert(D.errorText(err)); }
      } }, status === 'cancelling' ? '正在停止…' : '取消构建') : null;
    C.fill(msg, status === 'building' ? h('span', { class: 'spinner' }) : null,
      `快照「${sid}」：${ZH[status] || status}${depth ? `，第 ${done} / ${depth} 层` : ''}`,
      status === 'failed' && m.error ? `：${m.error}` : '', status === 'paused' && m.pause ? `（${m.pause.message || m.pause.reason}）` : '',
      cancel ? ' ' : '', cancel);
    if (!['building', 'cancelling'].includes(status)) {
      stop();
      await env.refreshLists();
      if (status === 'done') C.put(msg, ' ', h('button', { class: 'btn', type: 'button', onclick: () => env.go('browse', { src: { kind: 'snapshot', id: sid }, node: null }) }, '打开'));
    }
  };
  t = setInterval(tick, 2000);
  env.signal?.addEventListener('abort', stop, { once: true });
  tick();
}

// ================================================================ saved runs and snapshots
async function renderSaved(root, env) {
  const { ui } = env.ctx;
  await env.refreshLists();
  if (!env.alive()) return;
  let sets = null;
  try { sets = await D.get('agents/sets'); } catch (_) { sets = null; }
  const exportWhy = `${sets?.reason || '组合评估与提取服务尚不支持季度情景集'}${env.stub ? '；桩模型的推演也不能导出' : ''}`;

  // snapshots
  const snapRows = env.snaps.map(s => ({ ...s }));
  C.put(root, ui.card({ title: '快照', sub: '束搜索的冻结结果：只读，打开即用，不再运行模型；点击没有生成的分支会即时生成并存入覆盖层（核心文件不变）。' },
    snapRows.length ? ui.table([
      { key: 'id', label: '快照', fmt: (v, r) => h('span', {}, h('b', {}, v), r.stub ? ' ' : null, r.stub ? C.chip('桩模型', 'warn') : null) },
      { key: 'status', label: '状态', fmt: (v, r) => C.chip({ done: '完成', building: '构建中', failed: '失败', cancelling: '正在停止',
        paused: `已暂停${{ usage_limit: '（用量上限）', budget: '（达到调用上限）', server_restart: '（服务重启）', cancelled: '（已取消）' }[r.pause?.reason] || ''}` }[v] || v,
      v === 'done' ? 'good' : v === 'paused' ? 'warn' : v === 'failed' ? 'bad' : 'accent') },
      { key: 'B', label: '规模', fmt: (v, r) => `束宽 ${v ?? '—'} × ${r.depth ?? '—'} 季` },
      { key: 'levels_done', label: '完成层数', fmt: (v, r) => `${v ?? 0} / ${r.depth ?? '—'}` },
      { key: 'size_mb', label: '大小', fmt: (v) => (C.isNum(v) ? `${v.toFixed(1)} MB` : '—') },
      { key: 'finished', label: '完成于', fmt: (v, r) => C.when(v || r.created) },
      { key: 'id', label: '操作', fmt: (v, r) => h('div', { class: 'agents-rowbtns' },
        h('button', { class: 'btn', type: 'button', onclick: () => env.go('browse', { src: { kind: 'snapshot', id: v }, node: null }) }, '打开'),
        r.status === 'done' ? h('button', { class: 'btn', type: 'button', onclick: () => env.go('story', { src: { kind: 'snapshot', id: v }, node: null, page: null }) }, '连环画') : null,
        ['paused', 'failed'].includes(r.status) ? h('button', { class: 'btn', type: 'button', onclick: async (e) => {
          e.target.disabled = true;
          try {                                // the server resumes with the snapshot's own stored settings
            await D.post('agents/snapshots', { id: v, resume: true, confirm: env.mock });
            e.target.textContent = '已继续';
          } catch (err) {
            if (err.status === 409 && err.body?.needs_confirm && window.confirm(`继续构建会调用本机 Claude Code，剩余部分最多约 ${err.body.estimate?.calls_new ?? '—'} 次（已完成的层命中缓存）。确认继续？`)) {
              await D.post('agents/snapshots', { id: v, resume: true, confirm: true }).catch(e2 => window.alert(D.errorText(e2)));
            } else if (!(err.status === 409 && err.body?.needs_confirm)) window.alert(D.errorText(err));
            e.target.disabled = false;
          }
        } }, '继续构建') : null,
        ['building', 'cancelling'].includes(r.status) ? h('button', { class: 'btn', type: 'button', onclick: async (e) => {
          e.target.disabled = true;
          try { await D.post(`agents/snapshots/${encodeURIComponent(v)}/cancel`, {}); e.target.textContent = '正在停止…'; } catch (err) { window.alert(D.errorText(err)); e.target.disabled = false; }
        } }, '取消构建') : null) },
    ], snapRows, { stack: true }) : h('div', { class: 'muted' }, '还没有快照。在「新推演」底部可以估算并构建，或在终端运行 ',
      h('code', { class: 'agents-cmd' }, `python -m scenariolab.agents.snapshot --beam ${env.cfg.snapshot_defaults?.B ?? 2} --quarters 80 --dry-run`), '。')));

  // runs
  const runsBox = h('div');
  const filter = h('input', { type: 'text', placeholder: '按名称或 id 筛选', 'aria-label': '筛选推演', oninput: () => drawRuns() });
  C.put(root, h('div', { style: { height: '16px' } }), ui.card({
    title: '已保存的推演',
    sub: '每次推演都保存在本地（data/agents/runs/）。相同设置重跑全部命中缓存，不调用大模型；“加深”只为新的季度调用大模型。删除推演不删缓存。',
    actions: filter,
  }, runsBox, h('div', { class: 'agents-export', style: { marginTop: '12px' } },
    h('button', { class: 'btn', type: 'button', disabled: true, title: exportWhy }, '交给组合评估 / 提取与破产概率'),
    h('span', { class: 'why' }, `暂不可用：${exportWhy}。`))));

  const drawRuns = () => {
    const q = filter.value.trim();
    const rows = env.runs.filter(r => !q || (r.name || '').includes(q) || r.id.includes(q));
    if (!rows.length) { C.fill(runsBox, h('div', { class: 'muted' }, q ? '没有匹配的推演。' : '还没有推演。')); return; }
    C.fill(runsBox, ui.table([
      { key: 'name', label: '名称', fmt: (v, r) => nameCell(env, r, drawRuns) },
      { key: 'status', label: '状态', fmt: (v, r) => statusChip(v, r) },
      { key: 'depth', label: '规模', fmt: (v, r) => `${v ?? '—'} 季 · 束宽 ${r.B ?? '—'}${r.tree_kind && r.tree_kind !== 'agent' ? (r.tree_kind === 'model' ? ' · 模型树' : ' · 历史基率树') : ''}` },
      { key: 'calls_new', label: '调用（新/缓存）', fmt: (v, r) => `${v ?? 0} / ${r.calls_cached ?? 0}` },
      { key: 'created', label: '创建', fmt: (v) => C.when(v) },
      { key: 'id', label: '操作', fmt: (v, r) => runButtons(env, r, drawRuns) },
    ], rows, { stack: true }));
  };
  drawRuns();
}

function statusChip(v, r) {
  const map = { running: ['运行中', 'accent'], queued: ['排队中', 'accent'], cancelling: ['正在停止', 'accent'], done: ['完成', 'good'],
    cancelled: ['已取消', ''], failed: ['失败', 'bad'],
    paused: [r?.pause?.reason === 'usage_limit' ? '已暂停（用量上限）' : r?.pause?.reason === 'server_restart' ? '中断（服务重启）' : '已暂停', 'warn'] };
  const [t, k] = map[v] || [v || '—', ''];
  return C.chip(t, k);
}

function nameCell(env, r, redraw) {
  const wrap = h('span', { class: 'agents-name' }, h('b', {}, r.name || r.id), r.stub ? C.chip('桩模型', 'warn') : null,
    r.parent_run ? C.chip('人工干预', 'accent', `来自 ${r.parent_run}`) : null);
  const edit = h('button', { type: 'button', title: '重命名', 'aria-label': '重命名' }, '✎');
  edit.onclick = () => {
    const inp = h('input', { type: 'text', value: r.name || '', style: { width: '160px' } });
    const save = async () => {
      try { await D.patch(`agents/runs/${r.id}`, { name: inp.value }); r.name = inp.value; } catch (err) { window.alert(D.errorText(err)); }
      redraw();
    };
    inp.onkeydown = (e) => { if (e.key === 'Enter') save(); if (e.key === 'Escape') redraw(); };
    inp.onblur = save;
    C.fill(wrap, inp);
    inp.focus();
  };
  C.put(wrap, edit);
  return wrap;
}

function runButtons(env, r, redraw) {
  const live = ['running', 'queued', 'cancelling'].includes(r.status);
  const b = (label, fn, title) => h('button', { class: 'btn', type: 'button', title, onclick: fn }, label);
  const extendBox = h('span');
  return h('div', { class: 'agents-rowbtns' },
    b('打开', () => env.go('browse', { src: { kind: 'run', id: r.id }, node: null })),
    r.levels_done ? b('连环画', () => env.go('story', { src: { kind: 'run', id: r.id }, node: null, page: null })) : null,
    live ? b('进度', () => { AG.activeRun = r.id; env.go('setup'); }) : null,
    live ? b('取消', async () => { try { await D.post(`agents/runs/${r.id}/cancel`, {}); } catch (err) { window.alert(D.errorText(err)); } await env.refreshLists(); redraw(); }) : null,
    ['paused', 'cancelled', 'failed'].includes(r.status) ? b('继续', async () => {
      try { await D.post(`agents/runs/${r.id}/resume`, { confirm: env.mock }); AG.activeRun = r.id; env.go('setup'); } catch (err) {
        if (err.status === 409 && err.body?.needs_confirm) {
          if (window.confirm(`继续推演会调用本机 Claude Code，最多约 ${err.body.estimate?.calls_new ?? '—'} 次（已完成的部分不重复计入）。确认继续？`)) {
            try { await D.post(`agents/runs/${r.id}/resume`, { confirm: true }); AG.activeRun = r.id; env.go('setup'); } catch (e2) { window.alert(D.errorText(e2)); }
          }
        } else window.alert(D.errorText(err));
      }
    }) : null,
    r.status === 'done' ? b('加深', () => {
      const n = h('input', { type: 'number', min: 1, max: 80, value: 2, style: { width: '56px' }, 'aria-label': '再推几季' });
      const go = h('button', { class: 'btn primary', type: 'button' }, '开始');
      go.onclick = async () => {
        const quarters = C.clamp(Math.round(Number(n.value) || 1), 1, 80 - (r.depth || 0));
        try {
          await D.post(`agents/runs/${r.id}/extend`, { quarters, confirm: env.mock });
          AG.activeRun = r.id; env.go('setup');
        } catch (err) {
          if (err.status === 409 && err.body?.needs_confirm) {
            if (window.confirm(`加深 ${quarters} 季会调用本机 Claude Code，约 ${err.body.estimate?.calls_new ?? '—'} 次（已完成的部分命中缓存）。确认？`)) {
              try { await D.post(`agents/runs/${r.id}/extend`, { quarters, confirm: true }); AG.activeRun = r.id; env.go('setup'); } catch (e2) { window.alert(D.errorText(e2)); }
            }
          } else window.alert(D.errorText(err));
        }
      };
      C.fill(extendBox, h('span', { class: 'agents-name' }, '再推', n, '季', go));
    }, '在这次推演的末端再推几季（只为新的季度调用大模型）') : null,
    extendBox,
    !live ? b('删除', async () => {
      if (!window.confirm(`删除推演「${r.name || r.id}」？缓存里的节点会保留，相同设置重跑仍不调用大模型。`)) return;
      try { await D.del(`agents/runs/${r.id}`); D.dropTree(new D.Source('run', r.id)); if (AG.src?.id === r.id) AG.src = null; } catch (err) { window.alert(D.errorText(err)); }
      await env.refreshLists();
      redraw();
    }) : null);
}
