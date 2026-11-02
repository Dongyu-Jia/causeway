// 连环画 (storyboard) of one path through a snapshot or run (a helper module of agents.js). One page per quarter:
// a header (quarter, regime badge, macro gauges), one panel per acting actor (emblem → thinking → action card with
// the alternatives' probabilities → speech bubble with its public signal), a newspaper headline for the consequence,
// and an outcome panel (asset moves counting up, the example portfolio, the equity K-line growing by one quarter).
// Playback: play/pause, previous/next, speed, a scrubber, autoplay; branches can be switched at every decision point,
// and branches the snapshot did not expand (未生成) are generated on demand without a confirmation dialog.
// Inline SVG and CSS/Web Animations only; prefers-reduced-motion shows static panels.
import * as C from './_agent_common.js';
import * as D from './_agent_data.js';
import { emblem, sprite, gauge, thinkingDots, s as svgEl } from './_agent_art.js';
import { generate, expectedCalls, genNotice, assetZh, needsConfirm, pricingModel, pricingText } from './_agent_tree.js';

const { h } = C;

const CSS = `
.agents-story { outline: none; }
.agents-story:focus-visible .agents-stage { box-shadow: 0 0 0 3px var(--accent-soft); }
.agents-story-top { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin-bottom: 10px; font-size: 12px; color: var(--ink-3); }
.agents-story-top select { max-width: min(420px, 100%); font-size: 12px; }
.agents-stage { --ink: #1d2433; background-color: #e8ecf3; border-radius: 14px; padding: 12px; position: relative; overflow: hidden;
  background-image: radial-gradient(rgba(59, 91, 219, .10) 1px, transparent 1.2px); background-size: 9px 9px; }
.agents-page { display: grid; gap: 10px; }
.agents-panel { --c: #3b5bdb; position: relative; background: #fff; border: 2.5px solid var(--ink); border-radius: 7px;
  box-shadow: 4px 4px 0 rgba(29, 36, 51, .2); overflow: hidden; color: var(--ink); min-width: 0; }
.agents-panel::before { content: ''; position: absolute; inset: 0; pointer-events: none;
  background-image: radial-gradient(color-mix(in srgb, var(--c) 30%, transparent) 1.15px, transparent 1.35px); background-size: 7px 7px;
  -webkit-mask-image: linear-gradient(135deg, #000 0%, rgba(0,0,0,.35) 28%, transparent 52%); mask-image: linear-gradient(135deg, #000 0%, rgba(0,0,0,.35) 28%, transparent 52%); }
.agents-cap { position: absolute; top: 0; left: 0; z-index: 2; background: var(--ink); color: #fff; font-weight: 700; font-size: 12px; letter-spacing: .5px;
  padding: 3px 10px 4px 9px; border-bottom-right-radius: 7px; }
.agents-cap em { font-style: normal; color: #ffd43b; margin-right: 6px; }
/* header */
.agents-head { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 8px 18px; align-items: center; padding: 26px 16px 12px; }
.agents-head .big { font-size: 34px; font-weight: 800; line-height: 1; letter-spacing: 1px; }
.agents-head .big small { display: block; font-size: 13px; font-weight: 600; letter-spacing: 0; color: #4a5468; margin-top: 6px; }
.agents-head .mid { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.agents-head .mid .line { font-size: 13px; color: #4a5468; overflow-wrap: anywhere; }
.agents-gauges { display: flex; gap: 6px 10px; flex-wrap: wrap; justify-content: flex-end; }
.agents-gauge { text-align: center; font-size: 11px; color: #4a5468; width: 80px; }
.agents-gauge b { display: block; font-size: 15px; color: var(--ink); font-variant-numeric: tabular-nums; }
.agents-gauge .chg { font-size: 11px; }
.agents-gauge svg { color: var(--ink); display: block; margin: 0 auto; }
.agents-regime { display: inline-block; font-weight: 700; font-size: 12px; border-radius: 999px; padding: 2px 10px; border: 2px solid var(--ink); }
/* actors */
.agents-cast { display: grid; grid-template-columns: repeat(auto-fill, minmax(232px, 1fr)); gap: 10px; }
.agents-actorp { padding: 28px 10px 10px; display: grid; grid-template-columns: 74px minmax(0, 1fr); gap: 6px 10px; align-items: start; }
.agents-actorp .agents-emblem { color: var(--ink); width: 74px; height: 74px; position: relative; z-index: 1; }
.agents-actorp .side { position: relative; min-height: 74px; min-width: 0; display: grid; }
.agents-actorp .side > * { grid-area: 1 / 1; }
.agents-actorp.branching .agents-cap { background: var(--c); }
.agents-think { align-self: start; justify-self: start; background: #fff; border: 2px solid var(--ink); border-radius: 22px; padding: 6px 12px; font-size: 12px;
  color: #4a5468; position: relative; max-width: 100%; z-index: 1; }
.agents-think::before, .agents-think::after { content: ''; position: absolute; background: #fff; border: 2px solid var(--ink); border-radius: 50%; }
.agents-think::before { width: 10px; height: 10px; left: -8px; bottom: -4px; }
.agents-think::after { width: 6px; height: 6px; left: -16px; bottom: -9px; }
.agents-dots { color: var(--ink); display: block; margin-bottom: 2px; }
.agents-dots circle { animation: agents-dot 1s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
.agents-emblem.blink .agents-eyes ellipse { animation: agents-blink 3.2s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
.agents-emblem.blink .agents-eyes ellipse + ellipse { animation-delay: .05s; }
@keyframes agents-blink { 0%, 90%, 100% { transform: scaleY(1); } 95% { transform: scaleY(.15); } }
@keyframes agents-dot { 0%, 100% { opacity: .25; transform: scale(.75); } 50% { opacity: 1; transform: scale(1); } }
.agents-card { background: #fff; border: 2px solid var(--ink); border-radius: 8px; overflow: hidden; z-index: 1; min-width: 0; }
.agents-card .stripe { height: 6px; background: var(--c); }
.agents-card .in { padding: 6px 9px 8px; }
.agents-card .k { font-size: 11px; color: #7a8396; }
.agents-card .v { font-size: 16px; font-weight: 800; line-height: 1.3; overflow-wrap: anywhere; }
.agents-card .pp { font-size: 11px; color: #4a5468; margin-top: 2px; }
.agents-opts { grid-column: 1 / -1; display: grid; gap: 3px; margin-top: 2px; }
.agents-opt { display: grid; grid-template-columns: minmax(0, 1fr) minmax(48px, 38%) 38px; gap: 6px; align-items: center; font-size: 11px; color: #4a5468; border-radius: 4px;
  cursor: pointer; padding: 1px 2px; border: 0; background: none; text-align: left; font-family: inherit; }
.agents-opt:hover, .agents-opt:focus-visible { background: var(--accent-soft); outline: none; }
.agents-opt .n { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.agents-opt .bar { height: 7px; background: #eef1f7; border-radius: 4px; overflow: hidden; }
.agents-opt .bar span { display: block; height: 100%; background: #b8c6f5; border-radius: 4px; transform-origin: left center; }
.agents-opt.on { color: var(--ink); font-weight: 700; }
.agents-opt.on .bar span { background: #3b5bdb; }
.agents-opt .p { text-align: right; font-variant-numeric: tabular-nums; }
.agents-say { grid-column: 1 / -1; position: relative; background: #fff; border: 2px solid var(--ink); border-radius: 16px; padding: 6px 12px; font-size: 13px;
  font-weight: 600; line-height: 1.45; margin-top: 8px; min-height: 1.45em; overflow-wrap: anywhere; }
.agents-say::before { content: ''; position: absolute; top: -11px; left: 26px; border: 10px solid transparent; border-bottom-color: var(--ink); border-top: 0; }
.agents-say::after { content: ''; position: absolute; top: -7px; left: 28px; border: 8px solid transparent; border-bottom-color: #fff; border-top: 0; }
.agents-say.quiet { font-weight: 400; color: #7a8396; font-style: normal; }
.agents-why2 { grid-column: 1 / -1; font-size: 11px; color: #7a8396; line-height: 1.45; }
/* newspaper */
.agents-lower { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); gap: 10px; align-items: start; }
.agents-news { background: #f3f4f1; padding: 30px 16px 14px; font-family: "Songti SC", "STSong", "SimSun", "Noto Serif CJK SC", "Source Han Serif SC", serif; }
.agents-news::before { display: none; }
.agents-news .mast { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 3px double var(--ink); padding-bottom: 4px; gap: 8px; flex-wrap: wrap; }
.agents-news .mast b { font-size: 22px; font-weight: 900; letter-spacing: 4px; }
.agents-news .mast span { font-size: 11px; color: #4a5468; font-family: var(--font); }
.agents-news .hl { font-size: 26px; font-weight: 900; line-height: 1.25; margin: 10px 0 4px; }
.agents-news .deck { font-size: 14px; color: #2b3242; line-height: 1.55; border-bottom: 1px solid #c9ced8; padding-bottom: 8px; }
.agents-news .cols { columns: 2 220px; column-gap: 18px; column-rule: 1px solid #d5d9e0; font-size: 13px; line-height: 1.6; margin-top: 8px; }
.agents-news .cols p { margin: 0 0 6px; break-inside: avoid; }
.agents-news .cols b { font-family: var(--font); font-size: 11px; color: #7a8396; display: block; }
.agents-news .hlrow { display: flex; gap: 10px 16px; align-items: center; justify-content: space-between; flex-wrap: wrap; margin: 10px 0 4px; }
.agents-news .hlrow .hl { margin: 0; flex: 1 1 220px; }
.agents-stamp { flex: 0 0 auto; border: 3px solid #3b5bdb; color: #3b5bdb; border-radius: 9px; padding: 3px 10px; font-weight: 800;
  font-size: 15px; transform: rotate(-9deg); background: rgba(255, 255, 255, .85); font-family: var(--font); text-align: center; line-height: 1.25; }
.agents-stamp small { display: block; font-size: 10px; font-weight: 600; }
.agents-news.nature .hl::before { content: '自然事件 · '; color: #c77a00; }
/* outcome */
.agents-out { padding: 28px 14px 12px; }
.agents-out .rows { display: grid; gap: 6px; margin-top: 4px; }
.agents-orow2 { display: grid; grid-template-columns: 82px minmax(0, 1fr) 62px 20px; gap: 8px; align-items: center; font-size: 12px; }
.agents-orow2 .nm { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.agents-orow2 .track { position: relative; height: 14px; background: #f1f3f7; border-radius: 3px; }
.agents-orow2 .track .zero { position: absolute; left: 50%; top: -2px; width: 2px; height: 18px; background: var(--ink); opacity: .55; }
.agents-orow2 .track .fill { position: absolute; top: 2px; height: 10px; border-radius: 2px; transform-origin: left center; }
.agents-orow2 .num { text-align: right; font-weight: 800; font-variant-numeric: tabular-nums; font-size: 14px; }
.agents-orow2 .arw { font-size: 14px; text-align: center; }
.agents-orow2 .cum { grid-column: 2 / -1; font-size: 11px; color: #7a8396; margin-top: -4px; }
.agents-port { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; margin-top: 10px; padding-top: 8px; border-top: 2px dashed #c9ced8; }
.agents-port .lab { font-size: 12px; color: #4a5468; }
.agents-port .val { font-size: 26px; font-weight: 800; font-variant-numeric: tabular-nums; }
.agents-port .chg { font-size: 14px; font-weight: 700; }
.agents-kline { width: 100%; height: 150px; display: block; margin-top: 8px; }
.agents-kline-note { font-size: 10px; color: #7a8396; line-height: 1.4; }
/* forks */
.agents-forks { padding: 26px 12px 10px; }
.agents-forks .list { display: flex; flex-wrap: wrap; gap: 6px; }
.agents-fork { display: inline-flex; gap: 6px; align-items: center; border: 2px solid var(--ink); background: #fff; border-radius: 999px; padding: 3px 10px 3px 6px;
  font-size: 12px; cursor: pointer; font-family: inherit; color: var(--ink); max-width: 100%; text-align: left; }
.agents-fork:hover, .agents-fork:focus-visible { background: var(--accent-soft); outline: none; }
.agents-fork .pc { color: #3b5bdb; font-weight: 700; font-variant-numeric: tabular-nums; }
.agents-fork .t { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 300px; }
.agents-fork i { width: 10px; height: 10px; border-radius: 50%; background: #3b5bdb; display: inline-block; flex: 0 0 auto; }
.agents-fork.ghost { border-style: dashed; color: #4a5468; }
.agents-fork.ghost i { background: #fff; border: 2px dashed #7a8396; }
.agents-fork.overlay i { background: #fff; border: 2px solid #3b5bdb; }
.agents-fork.hit { box-shadow: 0 0 0 3px #ffd43b; }
.agents-genline { font-size: 12px; margin-top: 6px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
/* cover */
.agents-cover { padding: 30px 16px 16px; }
.agents-cover h2 { margin: 0 0 4px; font-size: 26px; font-weight: 800; }
.agents-cover .sub { color: #4a5468; font-size: 13px; margin-bottom: 12px; }
.agents-castlist { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.agents-castlist .m { display: grid; grid-template-columns: 48px minmax(0, 1fr); gap: 8px; align-items: center; font-size: 12px; }
.agents-castlist .m b { display: block; font-size: 13px; }
.agents-castlist .m span { color: #7a8396; line-height: 1.35; display: block; }
.agents-castlist .agents-emblem { width: 48px; height: 48px; color: var(--ink); }
/* controls */
.agents-controls { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin-top: 10px; padding: 8px 10px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 10px; position: sticky; bottom: 0; z-index: 3; box-shadow: 0 -2px 8px rgba(16, 24, 40, .06); }
.agents-controls .btn { padding: 5px 10px; min-width: 36px; }
.agents-controls .btn.play { min-width: 74px; font-weight: 700; }
.agents-controls input[type=range] { flex: 1 1 160px; min-width: 120px; }
.agents-controls .pos { font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; min-width: 128px; }
.agents-story-set { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; margin-bottom: 10px; font-size: 12px; color: var(--ink-2); }
.agents-story-set .seg button { padding: 4px 9px; font-size: 12px; }
.agents-story-set label { display: inline-flex; gap: 4px; align-items: center; }
.agents-story-set .btn { padding: 3px 10px; font-size: 12px; }
.agents-keys { color: var(--ink-3); font-size: 11px; }
.agents-hold { font-size: 11px; color: var(--ink-3); }
.agents-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.agents-pre-hidden { opacity: 0; }
@media (max-width: 900px) { .agents-lower { grid-template-columns: 1fr; } }
@media (max-width: 760px) {
  .agents-stage { padding: 8px; }
  .agents-head { grid-template-columns: 1fr; padding-top: 24px; }
  .agents-head .big { font-size: 28px; }
  .agents-gauges { justify-content: flex-start; }
  .agents-cast { grid-template-columns: 1fr; }
  .agents-news .hl { font-size: 21px; }
  .agents-controls { gap: 6px 8px; padding: 6px 8px; }
  .agents-controls .btn { padding: 4px 8px; min-width: 32px; }
  .agents-controls .btn.play { min-width: 64px; }
  .agents-controls input[type=range] { flex-basis: 100%; order: 9; }
  .agents-controls .pos { min-width: 0; order: 8; }
  .agents-keys { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  .agents-dots circle { animation: none; opacity: .8; }
  .agents-emblem.blink .agents-eyes ellipse { animation: none; }
}
`;

const SPEEDS = [{ value: '0.5', label: '0.5×' }, { value: '1', label: '1×' }, { value: '2', label: '2×' }, { value: '4', label: '4×' }];
const OUT_ASSETS = ['us_equity', 'nasdaq', 'treasury_10y', 'gold', 'crude_oil'];
const GAUGES = [
  { id: 's_ff', zh: '政策利率', min: 0, max: 10 }, { id: 's_unrate', zh: '失业率', min: 2, max: 12 },
  { id: 's_core_yoy', zh: '核心通胀', min: -1, max: 10 }, { id: 's_y10', zh: '10 年期', min: 0, max: 10 },
];
const STORY = { speed: 1, autoplay: true };       // module state: kept across re-renders and visits
const EASE = 'cubic-bezier(.22, 1, .36, 1)', POP = 'cubic-bezier(.34, 1.56, .64, 1)';

// ================================================================ entry
export async function renderStory(root, env) {
  C.ensureStyle('agents-story', CSS);
  const { ui } = env.ctx;
  const { AG, tree, source: src } = env;
  C.put(root, ui.loading('准备连环画…'));
  try { await tree.init(); } catch (err) {
    if (!env.alive()) return;
    C.fill(root, ui.notice(`读取失败：${D.errorText(err)}`, 'error'));
    return;
  }
  if (!env.alive()) return;
  if (!document.getElementById('agents-sprite')) { const sp = sprite(); sp.id = 'agents-sprite'; C.put(document.body, sp); }
  // the path first (one request); the rest of the tree (alternatives' descendants, the path picker) loads afterwards.
  // The path is identified by its leaf (AG.leaf, &leaf= in the route), so a re-render, a view switch or a reload plays
  // the same path; entering at an interior node plays it through to the most probable kept descendant at the deepest
  // level its subtree reaches (never a greedy descent that stops at a beam-search dead end).
  const isKey = (k) => typeof k === 'string' && /^[0-9a-f]{16}$/.test(k);
  let leaf = isKey(AG.leaf) ? AG.leaf : null;
  let start = isKey(AG.node) ? AG.node : null;
  if (!leaf && !start) start = tree.root;
  if (!env.alive()) return;
  const story = new Story(root, env);
  env.story = story;
  await story.setPath(start || leaf, { page: Number.isFinite(AG.page) ? AG.page : null, leaf });
  if (!env.alive()) return;
  tree.loadAll({ signal: env.signal }).then(() => { if (!story.dead) story.drawTop(); }).catch(() => {});
}

// a short label that tells leaves apart: the realized options that differ from the other leaves' (most first)
function pathTag(tree, leaf, leaves) {
  const mine = leaf.cl?.realized || {};
  const others = leaves.filter(l => l !== leaf).map(l => l.cl?.realized || {});
  const diff = Object.entries(mine).filter(([a, o]) => others.some(r => r[a] !== o))
    .sort((x, y) => others.filter(r => r[y[0]] !== y[1]).length - others.filter(r => r[x[0]] !== x[1]).length);
  const eq = leaf.assets?.us_equity?.c?.[1];
  const parts = diff.slice(0, 2).map(([a, o]) => `${C.actorName(a)}${C.cut(C.optionLabel(a, o), 6)}`);
  return `${parts.join('，') || C.cut(C.humanize(C.diffLabel(leaf.cl?.label, leaves.map(x => x.cl?.label))), 16)}${C.isNum(eq) ? ` · 美股 ${C.retText(eq)}` : ''}`;
}

// ================================================================ the player
class Story {
  constructor(root, env) {
    this.root = root; this.env = env; this.tree = env.tree; this.src = env.source;
    this.ui = env.ctx.ui;
    this.reduced = C.reducedMotion();
    this.path = []; this.recs = []; this.dets = new Map();
    this.page = 0; this.playing = false; this.clock = 0; this.beats = []; this.anims = []; this.tickers = [];
    this.raf = 0; this.endAt = 0; this.timer = 0;
    this.speed = STORY.speed;
    this.build();
    const stop = () => this.destroy();
    env.signal?.addEventListener('abort', stop, { once: true });
    this.onVis = () => { if (document.hidden && (this.playing || this.running)) this.pause(); };
    document.addEventListener('visibilitychange', this.onVis);
    this.portfolio = { name: '经典股债 60/40', weights: { us_equity: 0.6, treasury_10y: 0.4 } };
    env.ctx.api.optional('portfolio_default', env.ctx.meta).then(pd => {
      const p = (pd?.presets || []).find(x => x.id === 'classic_60_40') || (pd?.presets || [])[0];
      if (p?.weights) this.portfolio = { name: p.name || this.portfolio.name, weights: p.weights };
    }).catch(() => {});
  }
  destroy() {
    cancelAnimationFrame(this.raf); clearTimeout(this.timer);
    for (const a of this.anims) { try { a.cancel(); } catch (_) { /* gone */ } }
    document.removeEventListener('visibilitychange', this.onVis);
    document.removeEventListener('keydown', this.onKey);
    this.dead = true;
  }

  // ---------------------------------------------------------------- layout
  build() {
    const { ui } = this;
    this.top = h('div', { class: 'agents-story-top' });
    this.stage = h('div', { class: 'agents-stage', 'aria-roledescription': '连环画' });
    this.live = h('div', { class: 'agents-sr', 'aria-live': 'polite' });
    this.posEl = h('span', { class: 'pos' });
    this.scrub = h('input', { type: 'range', min: 0, max: 1, value: 0, 'aria-label': '跳到第几季', oninput: (e) => this.go(Number(e.target.value), { animate: false, keepPlaying: false }) });
    const btn = (label, aria, fn) => h('button', { class: 'btn', type: 'button', 'aria-label': aria, title: aria, onclick: fn }, label);
    this.playBtn = h('button', { class: 'btn primary play', type: 'button', onclick: () => this.toggle() }, '▶ 播放');
    this.auto = h('input', { type: 'checkbox', checked: STORY.autoplay ? true : null, onchange: (e) => { STORY.autoplay = e.target.checked; } });
    this.holdEl = h('span', { class: 'agents-hold' });
    this.controls = h('div', { class: 'agents-controls', role: 'toolbar', 'aria-label': '连环画播放控制' },
      btn('⏮', '第一页（封面）', () => this.go(0, { animate: false, keepPlaying: false })),
      btn('◀', '上一季（←）', () => this.go(this.page - 1, { animate: !this.reduced, keepPlaying: false })),
      this.playBtn,
      btn('▶|', '下一季（→）', () => this.go(this.page + 1, { animate: true, keepPlaying: this.playing })),
      btn('⏭', '最后一季', () => this.go(this.path.length - 1, { animate: false, keepPlaying: false })),
      this.scrub, this.posEl, this.holdEl);
    // playback settings sit above the stage (the sticky bar stays short on a phone)
    this.settings = h('div', { class: 'agents-story-set' }, h('span', {}, '速度'),
      ui.seg(SPEEDS, String(this.speed), (v) => this.setSpeed(Number(v))),
      h('label', {}, this.auto, '自动连播'),
      h('button', { class: 'btn', type: 'button', title: '把当前这一页存成图片', onclick: () => this.exportPng() }, '存为图片'),
      h('span', { class: 'agents-keys' }, '键盘：空格 播放/暂停 · ← → 翻页 · 1–4 速度'));
    this.wrap = h('div', { class: 'agents-story', tabindex: '0', 'aria-label': '连环画：空格播放或暂停，左右方向键翻页' }, this.top, this.settings, this.stage, this.controls, this.live);
    this.onKey = (e) => {
      if (this.dead || !this.wrap.isConnected) return;
      const a = document.activeElement;
      const mine = !a || a === document.body || this.wrap.contains(a);
      if (!mine || e.altKey || e.ctrlKey || e.metaKey) return;
      this.key(e);
    };
    document.addEventListener('keydown', this.onKey);
    C.fill(this.root, this.wrap);
  }
  // the page does not turn while the reader is choosing a branch (pointer over the fork panel) or has keyboard focus
  // inside the comic; the current page's own animation keeps going
  get holding() {
    const a = document.activeElement;
    return !!this.forkHover || (!!a && a !== this.wrap && this.stage.contains(a));
  }
  paintHold() { this.holdEl.textContent = this.holding && this.playing ? '正在选择分支：暂不翻页' : ''; }
  key(e) {
    if (e.target?.matches?.('input, select, textarea')) return;
    if ((e.key === ' ' || e.key === 'Enter') && e.target?.matches?.('button, a, summary')) return;    // the focused control's own action
    if (e.key === ' ' || e.key === 'k') { e.preventDefault(); this.toggle(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); this.go(this.page + 1, { animate: true, keepPlaying: this.playing }); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); this.go(this.page - 1, { animate: !this.reduced, keepPlaying: false }); }
    else if (e.key === 'Home') { e.preventDefault(); this.go(0, { animate: false, keepPlaying: false }); }
    else if (e.key === 'End') { e.preventDefault(); this.go(this.path.length - 1, { animate: false, keepPlaying: false }); }
    else if (['1', '2', '3', '4'].includes(e.key)) this.setSpeed([0.5, 1, 2, 4][Number(e.key) - 1]);
  }

  // ---------------------------------------------------------------- path
  async setPath(key, { page = null, leaf = null } = {}) {
    C.fill(this.stage, this.ui.loading('读取路径…'));
    let recs;
    try {
      let last = leaf;
      if (!last) {
        const r = await this.src.bestLeaf(key, { signal: this.env.signal });
        last = r?.leaf || key;
      }
      recs = await this.src.pathRecords(last, { signal: this.env.signal });
      if (leaf && key && !recs.some(r => r.key === key)) {
        // the page's node is not on the remembered path (a stale link): play the node's own best path
        const r = await this.src.bestLeaf(key, { signal: this.env.signal });
        recs = await this.src.pathRecords(r?.leaf || key, { signal: this.env.signal });
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      C.fill(this.stage, this.ui.notice(err.status === 404 ? `找不到这个节点（${key || leaf}）：它可能不在这个快照或推演里。请从路径菜单选一条路径。`
        : `读取路径失败：${D.errorText(err)}`, 'error'));
      return;
    }
    if (this.dead) return;
    if (!recs.length) {
      C.fill(this.stage, this.ui.notice('这条路径是空的：请从上方的路径菜单选一条路径。', 'error'));
      return;
    }
    this.recs = recs;
    this.path = recs.map(r => r.key);
    this.focusKey = key || this.path[this.path.length - 1];
    this.env.AG.leaf = this.path[this.path.length - 1];
    for (const r of recs) if (!this.tree.get(r.key)) this.tree.add(D.skeleton(r, r.source === 'overlay' || r.status === 'pruned' ? 'overlay' : 'snapshot'));
    this.computeScale();
    this.drawTop();
    this.scrub.max = String(Math.max(0, this.path.length - 1));
    const at = this.path.indexOf(key);
    const p = page !== null && page !== undefined ? C.clamp(page, 0, this.path.length - 1) : (at > 0 ? at : 0);
    this.go(p, { animate: !this.reduced && p > 0, keepPlaying: false });
  }
  computeScale() {
    // a stable scale for the whole path, from the candles (open, close, the quarter's 5–95% wicks)
    let lo = 100, hi = 100;
    for (let k = 1; k < this.recs.length; k++) {
      const a = this.recs[k].summary?.assets?.us_equity;
      const pa = this.recs[k - 1].summary?.assets?.us_equity;
      if (!a) continue;
      const open = k === 1 ? 100 : 100 * Math.exp(pa?.cum?.[1] ?? 0);
      for (const v of [open, 100 * Math.exp(a.cum?.[1] ?? 0), open * Math.exp(a.q?.[0] ?? 0), open * Math.exp(a.q?.[2] ?? 0)]) {
        if (C.isNum(v) && v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      }
    }
    this.kRange = [lo * 0.97, hi * 1.03];
  }
  drawTop() {
    const tree = this.tree;
    const leaves = tree.nodesAt(tree.maxLoadedLevel()).filter(D.TreeStore.shown).sort((a, b) => (b.p || 0) - (a.p || 0)).slice(0, 30);
    const last = this.recs[this.recs.length - 1];
    const sel = h('select', { 'aria-label': '选择路径', onchange: (e) => { const k = e.target.value; if (k) { this.env.AG.node = k; this.env.AG.leaf = k; this.env.AG.page = 1; this.env.setRoute(); this.setPath(k, { page: 1, leaf: k }); } } },
      h('option', { value: '' }, `当前路径：到 ${last?.quarter || ''}，末端概率 ${C.prob(tree.get(last?.key)?.p ?? last?.p_path)}`),
      leaves.length ? h('optgroup', { label: '换一条（末端节点，按概率）' }, leaves.map((l, i) => h('option', { value: l.key }, `#${i + 1} ${l.quarter || ''} 末端 ${C.prob(l.p)} · ${pathTag(tree, l, leaves)}`))) : null);
    C.fill(this.top, h('span', {}, '路径'), sel,
      h('button', { class: 'btn', type: 'button', onclick: () => this.env.go('browse', { node: this.path[this.page] || this.focusKey }) }, '在情景树里看这一页'),
      ((pm) => (pm.stub ? C.chip('桩模型：价格只用于检查流程', 'warn') : C.chip(`${pm.label || '季度模型'}定价`, 'accent', pricingText(pm))))(
        pricingModel(this.env, this.src?.kind === 'snapshot' ? (tree.summary?.manifest || {}) : (tree.summary || {}))),
      this.reduced ? C.chip('已按系统设置减少动画：静态画面', '') : null);
  }

  // ---------------------------------------------------------------- playback
  setSpeed(v) {
    this.speed = v; STORY.speed = v;
    for (const a of this.anims) { try { a.playbackRate = v; } catch (_) { /* gone */ } }
    const seg = this.settings.querySelector('.seg');
    seg?.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.textContent === `${v}×`));
  }
  toggle() { if (this.playing) this.pause(); else this.play(); }
  // playing = the reader wants the pages to keep turning; running = the current page's animation clock is ticking
  play() {
    this.playing = true;
    this.playBtn.textContent = '❚❚ 暂停';
    this.paintHold();
    if (this.done && this.page >= this.path.length - 1) { this.go(Math.min(1, this.path.length - 1), { animate: true, keepPlaying: true }); return; }
    if (this.done) this.maybeAdvance(true);
    else this.resume();
  }
  pause() {
    this.playing = false;
    this.running = false;
    this.playBtn.textContent = '▶ 播放';
    cancelAnimationFrame(this.raf); clearTimeout(this.timer);
    for (const a of this.anims) { try { if (a.playState === 'running') a.pause(); } catch (_) { /* gone */ } }
    this.wrap.querySelectorAll('.agents-dots circle, .agents-emblem.blink .agents-eyes ellipse').forEach(d => { d.style.animationPlayState = 'paused'; });
    this.paintHold();
  }
  resume() {
    this.running = true;
    for (const a of this.anims) { try { if (a.playState === 'paused') a.play(); } catch (_) { /* gone */ } }
    this.wrap.querySelectorAll('.agents-dots circle, .agents-emblem.blink .agents-eyes ellipse').forEach(d => { d.style.animationPlayState = 'running'; });
    this.loop();
  }
  loop() {
    cancelAnimationFrame(this.raf);
    let last = performance.now();
    const step = (now) => {
      if (this.dead || !this.running) return;
      const dt = Math.min(100, now - last); last = now;
      this.clock += dt * this.speed;
      while (this.beats.length && this.beats[0].at <= this.clock) {
        const b = this.beats.shift();
        try { b.fn(); } catch (err) { console.error(err); }
      }
      for (const t of this.tickers) t.tick(this.clock);
      this.tickers = this.tickers.filter(t => !t.done);
      if (!this.beats.length && this.clock >= this.endAt && !this.tickers.length) { this.finished(); return; }
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }
  finished() {
    this.running = false;
    this.done = true;
    this.maybeAdvance();
  }
  maybeAdvance(now = false) {
    clearTimeout(this.timer);
    if (this.dead || !this.wrap.isConnected) { this.destroy(); return; }
    if (!this.playing || !this.done) return;
    if (this.page >= this.path.length - 1) { this.pause(); this.say(`第 ${this.page} 季，路径到此为止。`); return; }
    if (!STORY.autoplay) { this.pause(); return; }
    const wait = now ? 200 : (this.reduced ? 3800 : 1300) / this.speed;
    this.timer = setTimeout(() => {
      if (!this.playing || this.dead) return;
      if (this.holding) { this.paintHold(); this.maybeAdvance(); return; }
      this.paintHold();
      this.go(this.page + 1, { animate: true, keepPlaying: true });
    }, wait);
  }
  // an animation that follows the player's clock (pause, speed)
  anim(el, frames, opts) {
    if (!el || typeof el.animate !== 'function') return null;
    const a = el.animate(frames, { fill: 'both', easing: EASE, ...opts });
    a.playbackRate = this.speed;
    if (!this.running) a.pause();
    this.anims.push(a);
    return a;
  }
  ticker(dur, fn) {
    const t0 = this.clock;
    const tk = { done: false, tick: (c) => { const f = C.clamp((c - t0) / dur, 0, 1); fn(1 - Math.pow(1 - f, 3)); if (f >= 1) tk.done = true; } };
    this.tickers.push(tk);
    return tk;
  }
  beat(at, fn) { this.beats.push({ at, fn }); }

  go(i, { animate = true, keepPlaying = true } = {}) {
    if (this.dead || !this.wrap.isConnected) { this.destroy(); return; }
    if (!this.path.length) return;
    i = C.clamp(Math.round(i), 0, this.path.length - 1);
    cancelAnimationFrame(this.raf); clearTimeout(this.timer);
    for (const a of this.anims) { try { a.cancel(); } catch (_) { /* gone */ } }
    this.anims = []; this.tickers = []; this.beats = []; this.clock = 0; this.done = false; this.running = false;
    this.page = i;
    this.env.AG.page = i;
    this.env.AG.node = this.path[i];
    this.env.setRoute();
    this.scrub.value = String(i);
    const r = this.recs[i];
    this.posEl.textContent = i === 0 ? `封面 · 共 ${this.path.length - 1} 季` : `第 ${i} / ${this.path.length - 1} 季 · ${r?.quarter || ''}`;
    if (!keepPlaying) this.playing = false;
    this.playBtn.textContent = this.playing ? '❚❚ 暂停' : '▶ 播放';
    const anim = animate && !this.reduced;
    const token = (this.goToken = (this.goToken || 0) + 1);
    this.renderPage(i, anim).then(() => {
      if (this.dead || this.page !== i || token !== this.goToken) return;
      this.say(i === 0 ? '封面：登场人物' : `第 ${i} 季，${C.quarterZh(r?.quarter)}`);
      if (!anim) { this.done = true; if (this.playing) this.maybeAdvance(); return; }
      this.resume();
    });
    // prefetch the decisions of this page and the next
    for (const k of [this.path[i - 1], this.path[i], this.path[i + 1]]) if (k && !this.dets.has(k)) this.detail(k).catch(() => {});
  }
  say(t) { this.live.textContent = t; }
  async detail(key) {
    if (this.dets.has(key)) return this.dets.get(key);
    const v = await this.src.node(key, { signal: this.env.signal });
    this.dets.set(key, v?.detail || null);
    return v?.detail || null;
  }

  // ---------------------------------------------------------------- pages
  async renderPage(i, animate) {
    const rec = this.recs[i];
    const prev = i > 0 ? this.recs[i - 1] : null;
    let pdet = null;
    if (prev) { try { pdet = await this.detail(prev.key); } catch (_) { pdet = null; } }
    if (this.dead || this.page !== i) return;
    const page = h('div', { class: 'agents-page' });
    C.fill(this.stage, page);
    if (i === 0) { this.cover(page, rec, animate); return; }
    this.header(page, rec, prev, i, animate);
    const cast = h('div', { class: 'agents-cast' });
    C.put(page, cast);
    const order = (pdet?.next || []).slice().sort((a, b) => C.actorStage(a.actor) - C.actorStage(b.actor));
    const realized = rec.cluster?.realized || {};
    const actors = order.length ? order : Object.keys(realized).map(a => ({ actor: a, zh: C.actorName(a), options: [{ id: realized[a], label: C.optionLabel(a, realized[a]), p_used: null }] }));
    const stages = [...new Set(actors.map(a => C.actorStage(a.actor)))];
    const t0 = 450;
    const stageGap = 1150;
    actors.forEach((a) => {
      const st = stages.indexOf(C.actorStage(a.actor));
      this.actorPanel(cast, a, rec, st + 1, animate ? t0 + st * stageGap : null);
    });
    let t = t0 + Math.max(1, stages.length) * stageGap + 300;
    if (rec.branch?.kind === 'nature_event') this.naturePanel(cast, rec, animate ? t - 600 : null);
    const lower = h('div', { class: 'agents-lower' });
    C.put(page, lower);
    this.headline(lower, rec, i, animate ? t : null);
    this.outcome(lower, rec, prev, i, animate ? t + 1000 : null);
    this.forks(page, rec, prev, i, animate ? t + 2600 : null);
    this.endAt = t + 3400;
  }

  panel(cls, color, cap) {
    const p = h('div', { class: `agents-panel ${cls}` });
    if (color) p.style.setProperty('--c', color);
    if (cap) C.put(p, h('div', { class: 'agents-cap' }, cap));
    return p;
  }
  enter(el, at) {
    if (at === null) return;
    el.classList.add('agents-pre-hidden');
    this.beat(at, () => { el.classList.remove('agents-pre-hidden'); this.anim(el, [{ opacity: 0, transform: 'translateY(12px)' }, { opacity: 1, transform: 'none' }], { duration: 380 }); });
  }

  cover(page, rec, animate) {
    const last = this.recs[this.recs.length - 1];
    const leafP = this.tree.get(last?.key)?.p ?? last?.p_path;
    const eq = last?.summary?.assets?.us_equity?.cum?.[1];
    const p = this.panel('agents-cover', '#3b5bdb', '封面');
    const cast = (this.env.cfg.catalogue?.actors || []).slice().sort((a, b) => (a.stage ?? 9) - (b.stage ?? 9));
    const list = h('div', { class: 'agents-castlist' }, cast.map(a => {
      const main = (a.domains || []).find(d => d.role === 'main');
      return h('div', { class: 'm' }, emblem(a.id, { size: 48, tone: 'neutral', label: a.zh }), h('div', {}, h('b', {}, a.zh), h('span', {}, main ? main.zh : '')));
    }));
    C.put(p, h('h2', {}, `从 ${rec.quarter} 末出发的 ${this.recs.length - 1} 个季度`),
      h('div', { class: 'sub' }, `这条路径到 ${last?.quarter || ''}，末端概率 ${C.prob(leafP)}；美股累计回报中位数 `,
        h('b', { class: C.retCls(eq) }, C.retText(eq)), '。每一页是一个季度：行为者依次决定，后果合成器给出后果，季度模型给出季末市场。'),
      h('div', { class: 'agents-sec', style: { margin: '6px 0 8px', fontWeight: 700 } }, '登场人物'), list,
      h('p', { class: 'agents-why2', style: { marginTop: '12px' } }, '图形都是通用的示意徽记，不代表任何真实的人、机构标志或国家。概率是蓝色；涨跌颜色跟随右上角的 K 线配色。'));
    C.put(page, p);
    this.header(page, rec, null, 0, animate);
    if (animate) this.enter(p, 0);
    this.endAt = 900;
  }

  header(page, rec, prev, i, animate) {
    const rep = rec.summary?.rep || {};
    const prep = prev?.summary?.rep || {};
    const us = this.recs.slice(Math.max(0, i - 4), i + 1).map(r => r.summary?.rep?.s_unrate);
    const rg = i ? C.regimeOf(rep, rec.summary?.bands?.d_rgdp?.[1], us) : null;
    const p = this.panel('agents-headp', '#3b5bdb', i ? '本季' : '起点');
    const gauges = GAUGES.map(g => {
      const gg = gauge({ value: rep[g.id], prev: prep[g.id], min: g.min, max: g.max });
      const d = C.isNum(rep[g.id]) && C.isNum(prep[g.id]) ? rep[g.id] - prep[g.id] : null;
      const val = h('b', {}, C.pp(rep[g.id], 2));
      const box = h('div', { class: 'agents-gauge', title: `${g.zh}：季末中位数` }, gg.svg, val,
        h('div', {}, g.zh, d !== null && Math.abs(d) >= 0.005 ? h('span', { class: 'chg' }, ` ${d > 0 ? '+' : ''}${d.toFixed(2)}`) : null));
      if (animate !== null && animate) {
        gg.needle.style.transform = `rotate(${gg.from}deg)`;
        this.beat(250, () => this.anim(gg.needle, [{ transform: `rotate(${gg.from}deg)` }, { transform: `rotate(${gg.to}deg)` }], { duration: 900, easing: POP }));
        if (C.isNum(prep[g.id]) && C.isNum(rep[g.id])) this.beat(250, () => this.ticker(900, (f) => { val.textContent = C.pp(prep[g.id] + (rep[g.id] - prep[g.id]) * f, 2); }));
      }
      return box;
    });
    const pathP = this.tree.get(rec.key)?.p ?? rec.p_path;
    C.put(p, h('div', { class: 'agents-head' },
      h('div', { class: 'big' }, i ? `第 ${i} 季` : '起点', h('small', {}, C.quarterZh(rec.quarter))),
      h('div', { class: 'mid' },
        h('div', { class: 'agents-chips' }, rg ? h('span', { class: 'agents-regime', style: { background: rg.color, color: rg.ink }, title: C.REGIME_RULE }, rg.name) : null,
          rec.stress ? C.chip('压力季度', 'warn') : null,
          rec.source === 'overlay' || rec.status === 'pruned' ? C.chip('即时生成', 'accent', '快照没有生成这一支：它是点击后由大模型即时生成的，存在覆盖层里') : null,
          i ? C.chip(`到达这一页的概率 ${C.prob(pathP)}`, 'accent') : null),
        i ? h('div', { class: 'line' }, C.humanize(C.liveFacts(rec.facts).map(([, t]) => t).slice(0, 4).join('；') || '没有持续的特殊事实')) : h('div', { class: 'line' }, '真实数据的最后一个完整季度；下一页开始推演。')),
      h('div', { class: 'agents-gauges' }, gauges)));
    C.put(page, p);
    if (animate) this.enter(p, 0);
  }

  actorPanel(cast, a, rec, step, at) {
    const color = C.actorColor(a.actor);
    const chosen = rec.cluster?.realized?.[a.actor] ?? rec.cluster?.head?.[a.actor] ?? null;
    // a follower's probabilities on this branch are its reaction row under the leader's realized option
    const lead = typeof a.follower_of === 'string' ? a.follower_of : null;
    const lo = lead ? (rec.cluster?.realized?.[lead] ?? rec.cluster?.head?.[lead]) : null;
    const row = lead && lo && a.cond?.[lo] ? a.cond[lo] : null;
    const opts = (a.options || []).map(o => (row ? { ...o, p_marg: o.p_used, p_used: row[o.id] ?? 0 } : o))
      .sort((x, y) => (y.p_used || 0) - (x.p_used || 0));
    const ch = opts.find(o => o.id === chosen) || null;
    const branching = !!a.branching;
    const p = this.panel(`agents-actorp ${branching ? 'branching' : ''}`, color, null);
    C.put(p, h('div', { class: 'agents-cap' }, h('em', {}, `第 ${step} 步`), a.zh || C.actorName(a.actor), branching ? ' · 分叉' : ''));
    const tone = C.toneOf(chosen, ch?.status_quo);
    const em = emblem(a.actor, { size: 74, tone: at === null ? tone : 'think', label: a.zh || C.actorName(a.actor), blink: at !== null && !this.reduced });
    const side = h('div', { class: 'side' });
    const top2 = opts.slice(0, 2).map(o => `${C.cut(o.label, 7)}？`).join(' ');
    const think = h('div', { class: 'agents-think' }, thinkingDots(), top2 || '……');
    const maxP = Math.max(0.01, ...opts.map(o => o.p_used || 0));
    const shown = opts.slice(0, 4);
    if (ch && !shown.includes(ch)) shown[shown.length - 1] = ch;
    const bars = shown.map(o => {
      const span = h('span');
      span.style.width = `${((o.p_used || 0) / maxP * 100).toFixed(1)}%`;
      const row = h('button', { class: `agents-opt ${o.id === chosen ? 'on' : ''}`, type: 'button', title: `${o.label}：采用概率 ${C.prob(o.p_used)}（点击：找这一选项的分支）`,
        onclick: () => this.highlightOption(a.actor, o) },
      h('span', { class: 'n' }, `${o.id === chosen ? '▶ ' : ''}${o.label}`), h('span', { class: 'bar' }, span), h('span', { class: 'p' }, C.prob(o.p_used)));
      row._bar = span;
      return row;
    });
    const rest = opts.filter(o => !shown.includes(o)).reduce((t, o) => t + (o.p_used || 0), 0);
    const card = h('div', { class: 'agents-card' }, h('div', { class: 'stripe' }),
      h('div', { class: 'in' }, h('div', { class: 'k' }, branching ? '分叉的选择' : '本支最常见的选择'),
        h('div', { class: 'v' }, ch ? C.humanize(ch.label) : C.optionLabel(a.actor, chosen)),
        ch && C.isNum(ch.p_used) ? h('div', { class: 'pp' }, row
          ? `在${C.actorName(lead)}选“${C.optionLabel(lead, lo)}”时 ${C.prob(ch.p_used)} · 不分情形 ${C.prob(ch.p_marg)}`
          : `采用 ${C.prob(ch.p_used)} · 大模型 ${C.prob(ch.p_llm)} · 基率 ${C.prob(ch.p_prior)}`) : null));
    const optsEl = h('div', { class: 'agents-opts', 'aria-label': '各选项的采用概率' }, bars,
      rest > 0.0005 ? h('div', { class: 'pp', style: { fontSize: '11px', color: '#7a8396' } }, `其余选项合计 ${C.prob(rest)}`) : null);
    C.put(side, think, card);
    const sig = (rec.signals || []).find(sg => sg.actor === a.actor);
    const text = sig?.text ? C.humanize(sig.text) : '';
    const say = h('div', { class: `agents-say ${text ? '' : 'quiet'}` }, text ? `“${text}”` : '（没有公开表态）');
    const why = ch?.rationale ? h('div', { class: 'agents-why2' }, `理由：${C.cut(C.humanize(ch.rationale), 80)}`) : null;
    C.put(p, em, side, optsEl, say, why);
    C.put(cast, p);
    this.panels = this.panels || [];
    if (at === null) { think.style.display = 'none'; return; }
    // the beat: enter → think → card → speech
    p.classList.add('agents-pre-hidden');
    card.style.visibility = 'hidden';
    optsEl.style.visibility = 'hidden';
    say.style.visibility = 'hidden';
    if (why) why.style.visibility = 'hidden';
    this.beat(at, () => {
      p.classList.remove('agents-pre-hidden');
      this.anim(p, [{ opacity: 0, transform: 'translateY(14px)' }, { opacity: 1, transform: 'none' }], { duration: 380 });
      this.anim(think, [{ opacity: 0, transform: 'scale(.6)' }, { opacity: 1, transform: 'scale(1)' }], { duration: 300, delay: 180, easing: POP });
      this.anim(em, [{ transform: 'translateY(0)' }, { transform: 'translateY(-3px)' }, { transform: 'translateY(0)' }], { duration: 700, delay: 200, iterations: 1 });
    });
    this.beat(at + 650, () => {
      this.anim(think, [{ opacity: 1, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(.7)' }], { duration: 200 });
      card.style.visibility = '';
      optsEl.style.visibility = '';
      this.anim(card, [{ opacity: 0, transform: 'scale(.85)' }, { opacity: 1, transform: 'scale(1)' }], { duration: 420, easing: POP });
      em.setTone(tone);
      this.anim(em.face, [{ transform: 'scale(1.25)' }, { transform: 'scale(1)' }], { duration: 300, easing: POP, composite: 'add' });
      bars.forEach((b, k) => this.anim(b._bar, [{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }], { duration: 520, delay: 120 + k * 60 }));
    });
    this.beat(at + 1150, () => {
      say.style.visibility = '';
      if (why) why.style.visibility = '';
      this.anim(say, [{ opacity: 0, transform: 'translateY(-4px)' }, { opacity: 1, transform: 'none' }], { duration: 260 });
      if (text) {
        const full = `“${text}”`;
        const n = [...full].length;
        say.textContent = '';
        this.ticker(Math.min(900, 40 * n), (f) => { say.textContent = [...full].slice(0, Math.max(1, Math.round(f * n))).join(''); });
      }
    });
    this.beat(at + 1150 + 20, () => think.remove());
  }

  naturePanel(cast, rec, at) {
    const p = this.panel('agents-actorp', C.actorColor('nature'), null);
    C.put(p, h('div', { class: 'agents-cap' }, h('em', {}, '自然'), '事件'));
    const em = emblem('nature', { size: 74, tone: 'worried', label: '自然事件' });
    C.put(p, em, h('div', { class: 'side' }, h('div', { class: 'agents-card' }, h('div', { class: 'stripe' }),
      h('div', { class: 'in' }, h('div', { class: 'k' }, '模型自身尾部的一段'), h('div', { class: 'v' }, C.humanize(rec.branch?.label || '自然事件')),
        h('div', { class: 'pp' }, `条件概率 ${C.prob(rec.branch?.p)}`)))));
    C.put(cast, p);
    if (at !== null) this.enter(p, at);
  }

  headline(lower, rec, i, at) {
    const br = rec.branch || {};
    const ax = br.axis ? (C.axisName(br.axis) || C.fieldName(br.axis)) : '';
    const kind = br.kind;
    let hl;
    if (kind === 'central') hl = ax ? `${ax}落在模型中枢` : '后果落在模型中枢';
    else if (kind === 'partition') hl = ax ? `${ax}：${br.label || ''}` : (br.label || '分区后果');
    else if (kind === 'default') hl = br.label ? `默认：${br.label}` : '默认分区';
    else hl = br.label || C.BRANCH_KIND[kind] || '本季后果';
    if (br.axis === 'd_gz') hl += '（情景假说）';
    const deck = C.humanize(br.mechanism || br.memory || '后果合成器把这一季的联合动作变成几种可能的后果；资产价格只来自季度模型。');
    const p = this.panel(`agents-news ${kind === 'nature_event' ? 'nature' : ''}`, null, '新闻头条');
    const items = { action: [], consequence: [], data: [], fact: [] };
    for (const [k, t] of rec.digest || []) (items[k] || (items[k] = [])).push(C.humanize(t));
    const col = (title, list) => (list?.length ? h('p', {}, h('b', {}, title), list.join('；')) : null);
    const stamp = h('div', { class: 'agents-stamp', title: '在这一动作簇之下，这个后果的概率' }, h('small', {}, '后果概率'), C.prob(br.p ?? rec.p_cond));
    C.put(p, h('div', { class: 'mast' }, h('b', {}, '季度快报'), h('span', {}, `${C.quarterZh(rec.quarter)} · 第 ${i} 期 · ${C.BRANCH_KIND[kind] || '后果'}`)),
      h('div', { class: 'hlrow' }, h('div', { class: 'hl' }, hl), stamp),
      h('div', { class: 'deck' }, deck, br.beyond_history ? ' （超出可测历史）' : ''),
      h('div', { class: 'cols' }, col('本季动作', items.action), col('季末数据', items.data), col('持续事实', items.fact),
        h('p', {}, h('b', {}, '这一支'), `动作簇概率 ${C.prob(rec.cluster?.p)} × 后果概率 ${C.prob(br.p)} = 条件概率 ${C.prob(rec.p_cond)}`)));
    C.put(lower, p);
    if (at === null) return;
    p.classList.add('agents-pre-hidden');
    stamp.style.visibility = 'hidden';
    this.beat(at, () => {
      p.classList.remove('agents-pre-hidden');
      const narrow = (this.stage.clientWidth || window.innerWidth) < 640;
      this.anim(p, narrow ? [{ opacity: 0, transform: 'scale(.6)' }, { opacity: 1, transform: 'scale(1)' }]
        : [{ opacity: 0, transform: 'rotate(-540deg) scale(.15)' }, { opacity: 1, transform: 'rotate(0deg) scale(1)' }], { duration: narrow ? 520 : 850 });
    });
    this.beat(at + 900, () => {
      stamp.style.visibility = '';
      this.anim(stamp, [{ opacity: 0, transform: 'rotate(-9deg) scale(1.8)' }, { opacity: 1, transform: 'rotate(-9deg) scale(1)' }], { duration: 260, easing: 'cubic-bezier(.5, 0, .75, 0)' });
    });
  }

  portfolioValue(rec) {
    const as = rec?.summary?.assets || {};
    let v = 0, w = 0;
    for (const [a, wt] of Object.entries(this.portfolio.weights || {})) {
      const c = as[a]?.cum?.[1];
      if (C.isNum(c)) { v += wt * Math.exp(c); w += wt; }
    }
    return w > 0 ? (100 * v) / w : null;
  }

  outcome(lower, rec, prev, i, at) {
    const p = this.panel('agents-out', '#3b5bdb', '季末市场');
    const as = rec.summary?.assets || {};
    const { up, down } = this.env.ctx.charts.klineColors();
    const bound = 0.2;
    const rows = OUT_ASSETS.filter(a => as[a]).map(a => {
      const q = C.simple(as[a].q?.[1]);
      const cum = C.simple(as[a].cum?.[1]);
      const fill = h('span', { class: 'fill' });
      const w = Math.min(1, Math.abs(q || 0) / bound) * 50;
      fill.style.background = (q || 0) >= 0 ? up : down;
      fill.style.width = `${w.toFixed(1)}%`;
      fill.style.left = (q || 0) >= 0 ? '50%' : `${(50 - w).toFixed(1)}%`;
      if ((q || 0) < 0) fill.style.transformOrigin = 'right center';
      const num = h('span', { class: `num ${this.ui.signCls(q)}` }, C.pct(q, 1, true));
      const arw = h('span', { class: `arw ${this.ui.signCls(q)}` }, Math.abs(q || 0) < 0.0005 ? '–' : q > 0 ? '▲' : '▼');
      const row = h('div', { class: 'agents-orow2' }, h('span', { class: 'nm' }, assetZh(a, this.ui)),
        h('span', { class: 'track' }, h('span', { class: 'zero' }), fill), num, arw,
        h('span', { class: 'cum' }, `从起点累计 `, h('span', { class: this.ui.signCls(cum) }, C.pct(cum, 1, true)),
          `（5–95%：${C.pct(C.simple(as[a].cum?.[0]), 0, true)} ~ ${C.pct(C.simple(as[a].cum?.[2]), 0, true)}）`));
      return { row, fill, num, arw, q };
    });
    const v1 = this.portfolioValue(rec), v0 = prev ? this.portfolioValue(prev) : 100;
    const chg = C.isNum(v1) && C.isNum(v0) && v0 ? v1 / v0 - 1 : null;
    const val = h('span', { class: 'val' }, C.isNum(v1) ? v1.toFixed(1) : '—');
    const chgEl = h('span', { class: `chg ${this.ui.signCls(chg)}` }, C.pct(chg, 1, true));
    const kl = this.kline(i);
    C.put(p, h('div', { class: 'rows' }, rows.map(r => r.row)),
      h('div', { class: 'agents-port' }, h('span', { class: 'lab' }, `示例组合（${this.portfolio.name}，起点 = 100）`), val, chgEl,
        h('span', { class: 'lab', style: { flexBasis: '100%' } }, '买入持有，按各资产累计回报的中位数加权，是近似值，不是组合的中位数。')),
      kl.svg,
      h('div', { class: 'agents-kline-note' }, '美股大盘指数（起点 = 100）：蜡烛 = 季初到季末的中位数；影线 = 本季回报 5–95%（不是季内最高最低）；浅色带 = 累计 5–95%。'));
    C.put(lower, p);
    if (at === null) return;
    p.classList.add('agents-pre-hidden');
    for (const r of rows) { r.fill.style.transform = 'scaleX(0)'; r.num.textContent = '0.0%'; r.num.className = 'num'; r.arw.style.visibility = 'hidden'; }
    val.textContent = C.isNum(v0) ? v0.toFixed(1) : '—';
    chgEl.style.visibility = 'hidden';
    kl.prep();
    this.beat(at, () => {
      p.classList.remove('agents-pre-hidden');
      this.anim(p, [{ opacity: 0, transform: 'translateY(14px)' }, { opacity: 1, transform: 'none' }], { duration: 380 });
    });
    this.beat(at + 300, () => {
      rows.forEach((r, k) => {
        this.anim(r.fill, [{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }], { duration: 800, delay: k * 70 });
        this.ticker(800 + k * 70, (f) => { r.num.textContent = C.pct((r.q || 0) * f, 1, true); });
      });
      if (C.isNum(v0) && C.isNum(v1)) this.ticker(1000, (f) => { val.textContent = (v0 + (v1 - v0) * f).toFixed(1); });
      kl.play();
    });
    this.beat(at + 1250, () => {
      rows.forEach(r => { r.num.className = `num ${this.ui.signCls(r.q)}`; r.arw.style.visibility = ''; this.anim(r.arw, [{ transform: 'scale(0)' }, { transform: 'scale(1)' }], { duration: 260, easing: POP }); });
      chgEl.style.visibility = '';
    });
  }

  // the equity index K-line along the path, one candle per quarter up to page i (SVG, drawn in code)
  kline(i) {
    const W = 520, H = 150, padL = 34, padR = 6, padT = 6, padB = 10;
    const n = Math.max(1, this.recs.length - 1);
    const [lo, hi] = this.kRange;
    const y = (v) => padT + (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (H - padT - padB);
    const slot = (W - padL - padR) / n;
    const x = (k) => padL + (k - 0.5) * slot;
    const { up, down } = this.env.ctx.charts.klineColors();
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'agents-kline', preserveAspectRatio: 'none', role: 'img', 'aria-label': '美股大盘指数的季度蜡烛图' });
    // axis: 100 line and labels
    for (const v of [lo, 100, hi]) {
      const yy = y(v);
      C.put(svg, svgEl('line', { x1: padL, x2: W - padR, y1: yy, y2: yy, stroke: v === 100 ? '#9aa3b5' : '#e3e6ec', 'stroke-dasharray': v === 100 ? '3 3' : null, 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }));
      C.put(svg, svgEl('text', { x: padL - 4, y: yy + 3, 'text-anchor': 'end', 'font-size': 9, fill: '#7a8396' }, v.toFixed(0)));
    }
    // fan band up to i
    const top = [], bot = [];
    for (let k = 0; k <= i; k++) {
      const c = this.recs[k]?.summary?.assets?.us_equity?.cum;
      const xx = k === 0 ? padL : x(k);
      top.push(`${xx.toFixed(1)},${y(k === 0 ? 100 : 100 * Math.exp(c?.[2] ?? 0)).toFixed(1)}`);
      bot.unshift(`${xx.toFixed(1)},${y(k === 0 ? 100 : 100 * Math.exp(c?.[0] ?? 0)).toFixed(1)}`);
    }
    const band = svgEl('polygon', { points: [...top, ...bot].join(' '), fill: 'rgba(59, 91, 219, .13)', stroke: 'none' });
    C.put(svg, band);
    let newest = null;
    for (let k = 1; k <= i; k++) {
      const r = this.recs[k], pr = this.recs[k - 1];
      const a = r?.summary?.assets?.us_equity;
      if (!a) continue;
      const open = k === 1 ? 100 : 100 * Math.exp(pr?.summary?.assets?.us_equity?.cum?.[1] ?? 0);
      const close = 100 * Math.exp(a.cum?.[1] ?? 0);
      const lowW = open * Math.exp(a.q?.[0] ?? 0), highW = open * Math.exp(a.q?.[2] ?? 0);
      const colr = close >= open ? up : down;
      const bw = Math.max(1.2, Math.min(14, slot * 0.62));
      const g = svgEl('g', {});
      const wick = svgEl('line', { x1: x(k), x2: x(k), y1: y(highW), y2: y(lowW), stroke: colr, 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' });
      const y0 = y(Math.max(open, close)), y1 = y(Math.min(open, close));
      const body = svgEl('rect', { x: x(k) - bw / 2, y: y0, width: bw, height: Math.max(1, y1 - y0), fill: colr });
      C.put(g, wick, body);
      C.put(svg, g);
      if (k === i) newest = { g, wick, body, open: y(open), y0, h: Math.max(1, y1 - y0), up: close >= open };
    }
    const self = this;
    return {
      svg,
      prep() {
        if (!newest) return;
        newest.wick.style.opacity = '0';
        newest.body.setAttribute('height', '0');
        newest.body.setAttribute('y', String(newest.open));
        band.style.clipPath = `inset(0 ${(((W - x(Math.max(1, i - 1)) - slot / 2) / W) * 100).toFixed(2)}% 0 0)`;
      },
      play() {
        if (!newest) return;
        self.anim(band, [{ clipPath: band.style.clipPath || 'inset(0 0 0 0)' }, { clipPath: 'inset(0 0 0 0)' }], { duration: 700 });
        self.ticker(700, (f) => {
          const hh = newest.h * f;
          newest.body.setAttribute('height', String(Math.max(0.5, hh)));
          newest.body.setAttribute('y', String(newest.up ? newest.open - hh : newest.open));
        });
        self.anim(newest.wick, [{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: 500 });
      },
    };
  }

  // the decision point: other branches of this quarter (kept: switch; 未生成: generate at once, no dialog)
  forks(page, rec, prev, i, at) {
    const p = this.panel('agents-forks', '#3b5bdb', '分叉点');
    const kids = [...(prev?.children || []), ...(prev?.overlay_children || [])];
    const seen = new Set();
    const alts = [];
    for (const c of kids) {
      if (seen.has(c.key) || c.key === rec.key) continue;
      seen.add(c.key);
      const sk = this.tree.get(c.key);
      const ghost = c.status === 'pruned' && c.overlay !== 'expanded' && !(sk && sk.expandedOnDemand);
      alts.push({ ...c, ghost, overlay: !ghost && (c.overlay || sk?.source === 'overlay') });
    }
    alts.sort((a, b) => (b.p_cond || 0) - (a.p_cond || 0));
    const labels = alts.map(a => a.label).concat(rec.cluster?.label);
    const genLine = h('div', { class: 'agents-genline', role: 'status', 'aria-live': 'polite' });
    const showN = 8;
    const list = h('div', { class: 'list' });
    const chipOf = (a) => h('button', { class: `agents-fork ${a.ghost ? 'ghost' : ''} ${a.overlay ? 'overlay' : ''}`, type: 'button', 'data-key': a.key, 'data-label': a.label || '',
      title: `${C.humanize(a.label || '')}${a.branch ? ` · 后果：${a.branch}` : ''}${a.ghost ? (needsConfirm(this.env) ? '（未生成：点击后先确认用量，再即时生成）' : '（未生成：点击即时生成，快照分支不需要确认）') : ''}`,
      onclick: () => this.switchTo(a, i, genLine) },
    h('i'), h('span', { class: 'pc' }, C.prob(a.p_cond)),
    h('span', { class: 't' }, `${a.ghost ? '未生成 · ' : a.overlay ? '即时生成 · ' : ''}${C.cut(C.humanize(C.diffLabel(a.label, labels)), 16)}${a.branch ? ` · ${a.branch}` : ''}`));
    alts.slice(0, showN).forEach(a => C.put(list, chipOf(a)));
    if (alts.length > showN) {
      const more = h('button', { class: 'agents-fork', type: 'button', onclick: () => { more.remove(); alts.slice(showN).forEach(a => C.put(list, chipOf(a))); } },
        h('span', { class: 't' }, `其余 ${alts.length - showN} 个`));
      C.put(list, more);
    }
    const isLast = i === this.recs.length - 1;
    const nextBtn = isLast ? h('button', { class: 'agents-fork ghost', type: 'button', title: needsConfirm(this.env) ? '这一支到此为止：即时生成下一季（先确认用量）' : '这一支在快照里到此为止：即时生成下一季（不需要确认）',
      onclick: () => this.extendEnd(rec, genLine) }, h('i'), h('span', { class: 't' }, '路径到此为止 · 即时生成下一季')) : null;
    C.put(p, h('div', { class: 'agents-why2', style: { marginBottom: '6px' } },
      alts.length ? `这一季还可能走向的 ${alts.length} 个分支（条件概率；点击切换。虚线圈是快照没有生成的分支，点击会立即调用大模型生成：${genNotice(this.env, expectedCalls(this.env, rec.level - 1, 1))}）` : '这一季没有其他分支。'),
    list, nextBtn ? h('div', { class: 'list', style: { marginTop: '6px' } }, nextBtn) : null, genLine);
    C.put(page, p);
    this.forkList = list;
    this.forkHover = false;
    p.addEventListener('pointerenter', () => { this.forkHover = true; this.paintHold(); });
    p.addEventListener('pointerleave', () => { this.forkHover = false; this.paintHold(); });
    if (at !== null) this.enter(p, at);
  }
  highlightOption(actor, o) {
    if (!this.forkList) return;
    const name = C.actorName(actor);
    let hits = 0;
    this.forkList.querySelectorAll('.agents-fork').forEach(b => {
      const lab = b.dataset.label || '';
      const sk = this.tree.get(b.dataset.key);
      const pick = sk?.cl?.realized?.[actor] ?? sk?.cl?.head?.[actor];
      const hit = pick ? pick === o.id : lab.includes(`${name}：${o.label}`);
      b.classList.toggle('hit', hit);
      if (hit) hits += 1;
    });
    this.say(hits ? `${name}选“${o.label}”的分支有 ${hits} 个，已用黄色圈出` : `这一季没有${name}选“${o.label}”的分支`);
    const line = this.stage.querySelector('.agents-forks .agents-why2');
    if (line) line.dataset.note = '';
    this.forkList.scrollIntoView({ behavior: this.reduced ? 'auto' : 'smooth', block: 'nearest' });
  }
  async switchTo(a, i, genLine) {
    if (a.ghost) {
      const btn = this.forkList?.querySelector(`[data-key="${a.key}"]`);
      if (btn) btn.disabled = true;
      try {
        await generate(this.env, a.key, { quarters: 1, box: genLine, parentLevel: this.recs[i - 1]?.level });
      } catch (_) { if (btn) btn.disabled = false; return; }
    }
    if (this.dead) return;
    this.dets.clear();
    this.env.AG.leaf = null;                       // the new branch's own best path
    await this.setPath(a.key, { page: i });
    this.env.AG.node = a.key;
    this.env.setRoute();
  }
  async extendEnd(rec, genLine) {
    try { await generate(this.env, rec.key, { quarters: 1, box: genLine }); } catch (_) { return; }
    if (this.dead) return;
    this.src.forget(rec.key);
    this.dets.clear();
    this.env.AG.leaf = null;
    await this.setPath(rec.key, { page: this.page + 1 });
  }

  // ---------------------------------------------------------------- export the current page as PNG
  async exportPng() {
    const was = this.playing;
    if (was) this.pause();
    this.go(this.page, { animate: false, keepPlaying: false });
    await new Promise(r => setTimeout(r, 120));
    const page = this.stage;
    const w = Math.ceil(page.scrollWidth), hgt = Math.ceil(page.scrollHeight);
    const clone = page.cloneNode(true);
    inlineStyles(page, clone);
    // emblems reference the sprite: copy each symbol's shapes in place
    clone.querySelectorAll('use').forEach(u => {
      const id = (u.getAttribute('href') || '').slice(1);
      const sym = document.getElementById(id);
      if (!sym) return;
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      for (const c of sym.children) C.put(g, c.cloneNode(true));
      u.replaceWith(g);
    });
    clone.querySelectorAll('button').forEach(b => { b.disabled = false; });
    const xml = new XMLSerializer().serializeToString(clone);
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${hgt}"><foreignObject x="0" y="0" width="100%" height="100%">${xml}</foreignObject></svg>`;
    const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
    const name = `连环画_${this.recs[this.page]?.quarter || 'cover'}`;
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      const scale = Math.min(2, window.devicePixelRatio || 1);
      const cv = document.createElement('canvas');
      cv.width = Math.round(w * scale); cv.height = Math.round(hgt * scale);
      const g = cv.getContext('2d');
      g.fillStyle = '#e8ecf3'; g.fillRect(0, 0, cv.width, cv.height);
      g.scale(scale, scale);
      g.drawImage(img, 0, 0);
      const blob = await new Promise((res, rej) => { try { cv.toBlob(b => (b ? res(b) : rej(new Error('toBlob'))), 'image/png'); } catch (err) { rej(err); } });
      download(URL.createObjectURL(blob), `${name}.png`, true);
    } catch (_) {
      download(url, `${name}.svg`, false);         // the browser refused the canvas: the page as SVG
    }
    if (was) this.play();
  }
}

const STYLE_PROPS = ['display', 'position', 'box-sizing', 'width', 'height', 'min-width', 'min-height', 'max-width', 'top', 'left', 'right', 'bottom',
  'margin-top', 'margin-right', 'margin-bottom', 'margin-left', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
  'border-top', 'border-right', 'border-bottom', 'border-left', 'border-radius', 'background-color', 'background-image', 'background-size',
  'color', 'font-family', 'font-size', 'font-weight', 'font-style', 'line-height', 'letter-spacing', 'text-align', 'white-space', 'overflow',
  'text-overflow', 'overflow-wrap', 'word-break', 'vertical-align', 'opacity', 'transform', 'transform-origin', 'box-shadow', 'z-index',
  'flex-direction', 'flex-wrap', 'flex-grow', 'flex-shrink', 'flex-basis', 'align-items', 'align-self', 'justify-content', 'justify-self', 'gap', 'row-gap', 'column-gap',
  'grid-template-columns', 'grid-template-rows', 'grid-column', 'grid-row', 'grid-area', 'columns', 'column-count', 'column-width', 'column-rule',
  'fill', 'stroke', 'stroke-width', 'stroke-dasharray', 'visibility', 'clip-path', 'mask-image', '-webkit-mask-image', 'font-variant-numeric'];
function inlineStyles(src, dst) {
  const a = [src, ...src.querySelectorAll('*')];
  const b = [dst, ...dst.querySelectorAll('*')];
  for (let k = 0; k < a.length && k < b.length; k++) {
    const cs = getComputedStyle(a[k]);
    const keep = b[k].getAttribute && b[k].getAttribute('style');
    let css = '';
    for (const p of STYLE_PROPS) { const v = cs.getPropertyValue(p); if (v) css += `${p}:${v};`; }
    for (let j = 0; j < cs.length; j++) { const p = cs[j]; if (p.startsWith('--')) css += `${p}:${cs.getPropertyValue(p)};`; }
    b[k].setAttribute('style', `${css}${keep || ''}`);
    b[k].removeAttribute?.('class');
  }
  // pseudo-elements (bubble tails, halftone corners) are not copied: the export is a flat page
}
function download(url, name, revoke) {
  const a = document.createElement('a');
  a.href = url; a.download = name;
  C.put(document.body, a); a.click(); a.remove();
  if (revoke) setTimeout(() => URL.revokeObjectURL(url), 4000);
}
