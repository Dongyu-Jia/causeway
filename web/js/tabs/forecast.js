// 未来一年：情景条件模拟 (SIM_DESIGN §9). This is not a forecast: the built-in scenarios' next four quarters, simulated,
// mixed by the scenario probabilities; the prediction lives only in the probabilities (Layer 1). forecast_mix.json
// (analysis/scenario_mix.py, the forecast build step) carries, per engine and scenario, each asset's quarter-end
// cumulative total return as a quantile grid; the page mixes the grids with the probabilities the reader sets (the
// mixture CDF sum_i w_i F_i, inverted by bisection: mixQuantile, the same rule as scenario_mix.mix_quantile).
// Sections: 1 一句话, 2 情景加权的季度区间, 3 每个情景的季度中位线 + 「你必须相信什么」, 4 概率 (inputs and each asset's
// sensitivity), 5 无情景基准, 6 情景组合隐含的宏观基率, 7 前瞻账本; then the rolling one-year record of the analog
// method (forecast.json) as 历史回放（相似起点法）.
// Engines: the monthly engine aggregated to quarter-ends is the default until the switch gate T-S passes; the
// quarterly stack is shown as 试运行. The default follows the live switch (GET /api/evaluation/switch, _switch.js),
// not the copy the build wrote, and the gate's reasons show under the engine choice. The tab id stays `forecast`, so
// #forecast deep links still work.
import { put, badge, completedBadge, badgeLegend, beliefList, quarterFan, ensureQStyle, pctS, lvl, prob, span, TRIAL, VAL_WORDING } from './_qdetail.js';
import { loadSwitch, switchNotice, quarterlyDefault } from './_switch.js';

const METHODS = ['analog', 'uncond', 'naive'];
const METHOD_CN = { analog: '相似起点', uncond: '无条件', naive: '历史均值', scenarios: '情景加权', scenarios_calibrated: '情景加权（校准）' };
const METHOD_COLOR = { analog: '#3b5bdb', uncond: '#868e96', naive: '#adb5bd', scenarios: '#e8590c' };
const CAUSE = { shock: ['外生冲击', 'warn'], macro: ['宏观意外', 'bad'], pricing: ['资产定价偏离', 'accent'] };
const STATE_CN = { cpi_yoy: 'CPI 同比', unrate: '失业率', real_ff: '实际政策利率', term_spread: '期限利差',
  spread: '信用利差', div_yield: '股息率（估值）', ip_yoy: '工业产出同比', oil_yoy: '油价同比' };
const MACRO_ROWS = [['unrate', '失业率变化', 'pp'], ['ff', '政策利率变化', 'pp'], ['cpi_yoy', 'CPI 同比变化', 'pp'],
  ['spread', '信用利差变化', 'pp'], ['oil', '油价涨跌', 'log']];
const NEXT_ASSETS = ['us_equity', 'nasdaq', 'treasury_10y', 'gold', 'energy', 'real_estate'];

// label of the 12 months a forecast made at `origin` covers (Dec origins -> the next calendar year)
const yearLabel = (o) => (o.date.endsWith('-12') ? String(Number(o.date.slice(0, 4)) + 1) : `${o.end}`);

// Tables of this page: headers wrap and cells are a little denser (.fc-t); the widest ones stack into labelled
// blocks below ~1000px of card width (.fc-wide), the mid-size ones below ~800px (.fc-mid), the rest at the
// shared 480px (app.css), so no table scrolls sideways inside its card.
// the shared phone layout of app.css (every row a block of labelled cells), for a wider container
const stacked = (sel) => `
  ${sel} table.tbl-stack thead { display: none; }
  ${sel} table.tbl-stack, ${sel} table.tbl-stack tbody, ${sel} table.tbl-stack td { display: block; width: auto; }
  ${sel} table.tbl-stack tr { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(200px, 100%), 1fr));
    column-gap: 16px; padding: 8px 2px; border-bottom: 1px solid var(--line); }
  ${sel} table.tbl-stack td { border: 0; padding: 2px 0; text-align: left !important; white-space: normal; }
  ${sel} table.tbl-stack td:first-child { grid-column: 1 / -1; font-weight: 600; }
  ${sel} table.tbl-stack td::before { content: attr(data-label); display: inline-block; min-width: 84px;
    margin-right: 8px; font-size: 11px; color: var(--ink-3); vertical-align: top; }
  ${sel} table.tbl-stack td:first-child::before { display: none; }`;
const CSS = `
.fc-t table.tbl th { white-space: normal; line-height: 1.35; vertical-align: bottom; }
.fc-t table.tbl th, .fc-t table.tbl td { padding: 7px 8px; }
.fc-t table.tbl td small { font-size: 11px; color: var(--ink-3); }
@container (max-width: 1000px) {${stacked('.fc-wide')}
}
@container (max-width: 800px) {${stacked('.fc-mid')}
}
@container (max-width: 480px) { .fc-t table.tbl-stack td { padding: 2px 0; } }
.fx-hero { border: 1px solid #cdd7ff; border-left: 4px solid var(--accent); background: linear-gradient(135deg, #f3f6ff, #fff 70%);
  border-radius: var(--radius); padding: 14px 16px; margin-bottom: 14px; }
.fx-hero h2 { margin: 0 0 6px; font-size: 16px; font-weight: 650; line-height: 1.55; color: #1f2f7a; }
.fx-hero p { margin: 0; font-size: 13px; color: var(--ink-2); }
.fx-bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 0 0 12px; }
.fx-bar .muted { font-size: 12px; }
.fx-probs { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(260px, 100%), 1fr)); gap: 6px 14px; }
.fx-prob { display: grid; grid-template-columns: minmax(0, 1fr) 76px; gap: 8px; align-items: center; font-size: 13px;
  padding: 4px 0; border-bottom: 1px solid var(--line); }
.fx-prob .nm { min-width: 0; overflow-wrap: anywhere; }
.fx-prob .nm small { display: block; color: var(--ink-3); font-size: 11px; }
.fx-prob input { width: 76px; text-align: right; }
.fx-sum { font-size: 13px; font-variant-numeric: tabular-nums; }
.fx-sum.bad { color: var(--bad); }
.fx-med { font-variant-numeric: tabular-nums; font-weight: 600; }
.fx-med small { font-weight: 400; color: var(--ink-3); }
.fx-anchor { scroll-margin-top: 72px; }
.fx-note { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; }
.fx-replay { border-top: 2px dashed var(--line); margin-top: 28px; padding-top: 6px; }
.page[data-tab="forecast"] .card-actions select, .page[data-tab="forecast"] .fx-bar select { max-width: 100%; min-width: 0; }
.qb-assume { background: #fff0f0; border-color: #f0a3a3; color: #b02a2a; }
.fx-phase { margin: 10px 0 0; font-size: 13px; }
.fx-phase > b { display: block; margin-bottom: 2px; }
.fx-phase ul { margin: 2px 0 0; padding-left: 18px; }
.fx-phase li { margin: 4px 0; overflow-wrap: anywhere; }
.fx-phase .fx-why { display: block; color: var(--ink-3); font-size: 12px; margin-top: 2px; }
`;

// ---------------------------------------------------------------- the monthly engine's own phase specs (section 3)
// Under the monthly engine, section 3 shows what the monthly engine ran: the scenario library's phase specs (their
// views and overrides: GET /api/scenarios/<id> spec.phases), not the QTable, which is their conversion for the
// quarterly stack (GET /api/q/scenarios/<id> conversion[]: the records it could not carry are listed).
const MROW = {
  policy: ['f_ff', 's_ff', 's_real_ff'],
  growth: ['f_unrate', 's_unrate', 'f_ip', 's_ip_yoy'],
  inflation: ['s_cpi_yoy', 's_core_yoy', 'f_cpi_goods', 'f_cpi_services', 's_goods_yoy', 's_services_yoy'],
  oil: ['f_oil', 's_oil_yoy', 's_real_oil'],
  fc: ['f_spread', 's_spread', 'f_gz', 's_gz', 'f_ebp', 's_ebp'],
  valuation: ['s_div_yield', 'f_valuation', 'f_cape', 's_cape'],
};
const VALUATION_VIEWS = new Set(MROW.valuation);
const MSPEC = new Map();                         // sid -> {phases, dropped} | {error}
const fmtNum = (x, d = 2) => { const t = Number(x).toFixed(d); return Number(t) === 0 ? (0).toFixed(d) : t; };
const sgnNum = (x, d = 2) => { const t = fmtNum(x, d); return Number(t) > 0 ? `+${t}` : t; };

function monthlyText(ui) {
  const vName = (v0) => {
    const v = String(v0 || '').replace(/^overrides\./, '');
    if (v === 'asset_drift') return '资产漂移覆盖';
    if (v === 'factor_drift') return '因子漂移覆盖';
    if (v === 'r_star') return '中性实际利率 r*';
    if (v === 'term_premium') return '期限溢价锚';
    if (v === 'resid_scale') return '残差缩放';
    const n = ui.varName(v);
    return n && n !== v ? n : '其他变量';
  };
  const unitOf = (v) => {
    const [pre, ...rest] = String(v).split('_');
    const id = rest.join('_');
    return (pre === 's' ? ui.META.states[id]?.unit : ui.META.factors[id]?.unit) || '';
  };
  const change = (x, unit) => (unit === 'Δlog' ? `${sgnNum(x * 100, 1)}%` : unit === 'Δpp' ? `${sgnNum(x, 2)} 个百分点` : sgnNum(x, 2));
  const level = (x, unit) => (unit === '%' ? `${fmtNum(x, 2)}%` : unit === 'pp' ? `${fmtNum(x, 2)} 个百分点` : unit === '倍' ? `${fmtNum(x, 1)} 倍` : fmtNum(x, 2));
  const view = (v) => {
    const unit = unitOf(v.var);
    const lvl = String(v.var).startsWith('s_');
    const parts = [];
    if (isNumber(v.mean)) parts.push(lvl ? `平均 ${level(v.mean, unit)}` : `每月 ${change(v.mean, unit)}`);
    if (isNumber(v.mean_annual)) parts.push(`每年 ${change(v.mean_annual, unit)}`);
    if (isNumber(v.min)) parts.push(`不低于 ${lvl ? level(v.min, unit) : change(v.min, unit)}`);
    if (isNumber(v.max)) parts.push(`不高于 ${lvl ? level(v.max, unit) : change(v.max, unit)}`);
    if (isNumber(v.vol_mult)) parts.push(`波动按历史的 ${fmtNum(v.vol_mult, 2)} 倍`);
    return `${vName(v.var)}：${parts.join('，') || '有视图'}`;
  };
  return { vName, view };
}

// [{badge kind, label, text, why, row}] of one phase: its views and its overrides
function phaseItems(ph, ui) {
  const { vName, view } = monthlyText(ui);
  const rowOf = (v) => Object.keys(MROW).find(k => MROW[k].includes(v)) || null;
  const out = [];
  for (const v of ph.views || []) {
    const val = VALUATION_VIEWS.has(v.var);
    out.push({ cls: val ? 'qb-assume' : 'qb-stated', label: val ? '估值假设' : '情景视图',
      text: val ? `${view(v)}（估值假设，${VAL_WORDING}）` : view(v), why: v.why, row: rowOf(v.var) });
  }
  const ov = ph.overrides || {};
  if (ov.asset_drift && typeof ov.asset_drift === 'object') {
    const txt = Object.entries(ov.asset_drift).filter(([, x]) => isNumber(x)).map(([a, x]) => `${ui.assetName(a)}每年 ${sgnNum(x * 100, 1)}%`).join('、');
    out.push({ cls: 'qb-assume', label: '资产漂移覆盖', text: `资产漂移覆盖：${txt}（直接改资产的漂移，${VAL_WORDING}）`, row: 'valuation' });
  }
  if (ov.factor_drift && typeof ov.factor_drift === 'object') {
    for (const [f, x] of Object.entries(ov.factor_drift)) {
      if (!isNumber(x)) continue;
      out.push({ cls: 'qb-derived', label: '因子漂移', text: `因子漂移覆盖：${vName(`f_${f}`)}每年 ${sgnNum(x, 2)}`, row: rowOf(`f_${f}`) });
    }
  }
  if (isNumber(ov.r_star)) out.push({ cls: 'qb-anchor', label: '锚', text: `中性实际利率 r* = ${fmtNum(ov.r_star, 2)}%（这个阶段的长期锚）`, row: 'policy' });
  if (isNumber(ov.term_premium)) out.push({ cls: 'qb-anchor', label: '锚', text: `期限溢价锚 = ${fmtNum(ov.term_premium, 2)} 个百分点`, row: null });
  if (ph.regime && typeof ph.regime === 'object') out.push({ cls: 'qb-stated_qual', label: '体制请求', text: `体制请求：${Object.keys(ph.regime).length} 项（见情景详情）`, row: 'growth' });
  return out;
}

function loadMonthlySpec(api, sid) {
  if (MSPEC.has(sid)) return MSPEC.get(sid);
  const p = Promise.all([api.get(`scenarios/${sid}`), api.get(`q/scenarios/${sid}`).catch(() => null)]).then(([d, q]) => {
    let start = 0;
    const phases = ((d?.spec?.phases || d?.phases) || []).map(ph => { const o = { ...ph, start }; start += Number(ph.months) || 0; return o; });
    const dropped = (q?.conversion || []).filter(c => c.status === 'dropped');
    const v = { phases, dropped };
    MSPEC.set(sid, v);
    return v;
  }, (err) => { const v = { error: err?.message || '读不出来' }; MSPEC.set(sid, v); return v; });
  MSPEC.set(sid, p);
  return p;
}
// phases overlapping quarter k (0-based): months [3k, 3k + 3)
const phasesAt = (phases, k) => phases.filter(ph => ph.start < 3 * (k + 1) && ph.start + (Number(ph.months) || 0) > 3 * k);

// ui.table's `stack` for this page: stacked below 480px (T), 800px (MID) or 1000px (WIDE) of card width
const T = 'fc-t', MID = 'fc-t fc-mid', WIDE = 'fc-t fc-wide';

function ensureStyle() {
  let el = document.getElementById('style-forecast');
  if (!el) { el = document.createElement('style'); el.id = 'style-forecast'; document.head.append(el); }
  if (!el.textContent.includes('/* forecast */')) el.textContent += `/* forecast */\n${CSS}`;
}

// ---------------------------------------------------------------- the mixture (same rule as scenario_mix.py)

const isNumber = (x) => typeof x === 'number' && Number.isFinite(x);
// F(x) of the distribution whose quantiles at the grid percentiles `P` (0..100) are `g`, linear between points
export function gridCdf(g, P, x) {
  if (x < g[0]) return 0;
  if (x >= g[g.length - 1]) return 1;
  let k = 1;
  while (k < g.length && g[k] <= x) k++;
  const lo = g[k - 1], hi = g[k];
  const p0 = P[k - 1] / 100, p1 = P[k] / 100;
  return hi > lo ? p0 + (p1 - p0) * ((x - lo) / (hi - lo)) : p1;
}
// the q-quantile (0..1) of sum_i w_i F_i, exactly: the mixture CDF is linear between the sorted union of the grids'
// knots, so a binary search over the knots finds the segment that reaches q and the quantile is interpolated in it
// (a divergent path can put a grid's maximum at 1e40, where a bisection on the values would not converge)
export function mixQuantile(grids, weights, P, q) {
  const pairs = grids.map((g, i) => [g, weights[i]]).filter(([g, w]) => Array.isArray(g) && g.length === P.length && w > 0);
  if (!pairs.length) return null;
  const W = pairs.reduce((t, [, w]) => t + w, 0);
  const F = (x) => pairs.reduce((t, [g, w]) => t + w * gridCdf(g, P, x), 0) / W;
  const knots = [...new Set(pairs.flatMap(([g]) => g))].sort((a, b) => a - b);
  if (knots.length === 1) return knots[0];
  let lo = 0, hi = knots.length - 1;               // smallest j with F(knots[j]) >= q
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (F(knots[mid]) >= q) hi = mid; else lo = mid + 1;
  }
  if (lo === 0) return knots[0];
  const x0 = knots[lo - 1], x1 = knots[lo], f0 = F(x0), f1 = F(x1);
  return f1 > f0 ? x0 + ((q - f0) * (x1 - x0)) / (f1 - f0) : x1;
}

// Kept across re-renders (the K-line toggle) and visits within the page load.
const MIX_STATE = { engine: null, asset: 'us_equity', scen: null, set: null, probs: null };

export async function render(root, ctx) {
  const { api, ui, meta } = ctx;
  const alive = ctx.alive || (() => root.isConnected);
  const { h } = ui;
  ensureStyle();
  ensureQStyle();
  const [mix, fcRes, scen, sw] = await Promise.all([
    api.optional('forecast_mix', meta),
    api.get('forecast').then(d => ({ d }), e => ({ e })),
    api.optional('scenarios', meta),
    loadSwitch(),
  ]);
  if (!alive()) return;
  if (mix) {
    try { renderMix(root, mix, ctx, sw); } catch (err) {
      console.error(err);
      root.append(ui.notice(`情景加权这一部分渲染失败：${err.message}`, 'error'));
    }
  } else {
    root.append(h('div', { class: 'fx-hero' },
      h('h2', {}, '这不是预测：情景条件模拟按情景概率加权，预测只在情景概率里。'),
      h('p', {}, '情景加权的季度结果（forecast_mix.json）还没有生成。')),
    ui.notice('在项目根目录运行 python -m scenariolab.build --only forecast（或只重算情景加权：python -m scenariolab.analysis.scenario_mix），生成后刷新本页。', 'info'));
  }
  const replay = h('div', { class: 'fx-replay fx-anchor', id: 'fx-replay' });
  root.append(replay);
  replay.append(h('div', { class: 'section-title' }, '历史回放（相似起点法）'),
    h('p', { class: 'lead' },
      '下面是另一套、不依赖情景的方法：每年年底只用当时能看到的数据，找状态与当时相近的历史起点，看它们之后一年发生了什么。',
      '多年期的路径只有在给出情景主题之后才有意义；没有主题时能认真打分的最长期限是下一年。战争、疫情这类外生冲击造成的偏差单独标出，不算模型错误。'));
  if (fcRes.e) {
    const err = fcRes.e;
    if (err.status === 404) replay.append(ui.notice('一年期回放结果（forecast.json）还没有生成。', 'info'), ui.buildHint(err));
    else replay.append(ui.notice(`读取一年期回放结果失败：${err.message}`, 'error'));
    return;
  }
  renderReplay(replay, fcRes.d, scen, ctx);
}

// ---------------------------------------------------------------- sections 1-7

function renderMix(root, mix, ctx, sw) {
  const { ui, charts } = ctx;
  const { h, card } = ui;
  const P = mix.grid || [];
  const engines = Object.keys(mix.engines || {});
  // the live switch decides the default (the build's copy may be older than the latest evaluation)
  const switched = quarterlyDefault(sw);
  const defEngine = switched && engines.includes('quarterly') ? 'quarterly'
    : engines.includes('monthly') ? 'monthly' : (mix.default_engine || engines[0]);
  const trialOf = (e) => e === 'quarterly' && !switched;
  const engName = (e) => (e === 'quarterly' ? (switched ? '季度模型' : `季度模型（${TRIAL}）`) : (mix.engines[e]?.name || e));
  if (!MIX_STATE.engine || !engines.includes(MIX_STATE.engine)) MIX_STATE.engine = defEngine;
  const scenarios = mix.scenarios || [];
  const ids = scenarios.map(s => s.id);
  const nameOf = (id) => scenarios.find(s => s.id === id)?.name || '（未命名情景）';
  const assets = (mix.assets || []).filter(a => a !== 'cash').concat((mix.assets || []).includes('cash') ? ['cash'] : []);
  const assetName = (a) => mix.asset_names?.[a] || ui.assetName(a);
  if (!assets.includes(MIX_STATE.asset)) MIX_STATE.asset = assets[0];
  if (!MIX_STATE.scen || !ids.includes(MIX_STATE.scen)) MIX_STATE.scen = ids[0];
  const sets = mix.probabilities?.sets || {};
  if (!MIX_STATE.set || !(MIX_STATE.set in sets || MIX_STATE.set === 'custom')) MIX_STATE.set = mix.probabilities?.default || Object.keys(sets)[0];
  if (!MIX_STATE.probs) MIX_STATE.probs = { ...(sets[MIX_STATE.set]?.probs || {}) };
  const quarters = mix.quarters || [];
  const nQ = quarters.length;
  const eng = () => mix.engines[MIX_STATE.engine] || {};
  const later = [];

  // the scenarios the selected engine has a usable result for (the others are left out of its mixture)
  const usable = (i) => {
    if ((eng().excluded || []).includes(i)) return false;
    const a = eng().scenarios?.[i]?.assets || {};
    return Object.values(a).some(x => Array.isArray(x?.grid?.[nQ - 1]));
  };
  const heroH = h('h2');
  const paintHero = () => {
    const n = ids.filter(usable).length;
    heroH.textContent = `这不是预测：下面是 ${n} 个情景在未来 ${nQ} 个季度的模拟，按情景概率加权。预测只在情景概率里。`
      + (n < ids.length ? `（另有 ${ids.length - n} 个情景这个引擎没有可用路径，未计入。）` : '');
  };
  paintHero();

  // ---- 1. 一句话
  root.append(h('div', { class: 'fx-hero' },
    heroH,
    h('p', {}, '每个情景是一个带概率的假说：如果它成立，模型认为资产会怎样；模型本身不判断哪个情景会发生。',
      `起点 ${mix.origin || '—'}（训练期到 ${mix.train_end || '—'}）；概率可以在第 4 节改，改完全页按新的概率重算。`)));

  const engineSeg = ui.seg(engines.map(e => ({ value: e, label: `${engName(e)}${e === defEngine ? '（默认）' : ''}` })),
    MIX_STATE.engine, (v) => { MIX_STATE.engine = v; redrawAll(); });
  const engineNote = h('div');
  root.append(...[h('div', { class: 'fx-bar' }, h('span', { class: 'muted' }, '引擎：'), engineSeg),
    switchNotice(sw, { where: '未来一年页' }),
    engineNote,
    (mix.notes || []).length ? h('div', { style: { margin: '0 0 12px' } }, mix.notes.map(n => ui.notice(n, 'info'))) : null].filter(Boolean));
  const paintEngineNote = () => {
    const e = eng();
    const trial = trialOf(MIX_STATE.engine);
    const etext = switched ? String(e.text || '').replace(`季度模型（${TRIAL}）`, '季度模型') : (e.text || '');   // a build before the switch
    put(engineNote, ui.notice(`${etext}。${trial ? `${TRIAL}：在切换关 T-S 通过之前，这个引擎的数字只作对照。` : ''}`, trial ? 'error' : 'info'),
      ...(e.notes || []).map(n => ui.notice(n, 'info')));
  };

  // ---- probabilities (normalised over the scenarios that have a result for this asset and quarter)
  const weightsFor = (a, k) => {
    const sc = eng().scenarios || {};
    const avail = ids.filter(i => Array.isArray(sc[i]?.assets?.[a]?.grid?.[k]));
    const raw = avail.map(i => Math.max(Number(MIX_STATE.probs[i]) || 0, 0));
    const tot = raw.reduce((t, x) => t + x, 0);
    return { ids: avail, w: tot > 0 ? raw.map(x => x / tot) : raw };
  };
  const mixAt = (a, k) => {
    const sc = eng().scenarios || {};
    const { ids: av, w } = weightsFor(a, k);
    const grids = av.map(i => sc[i].assets[a].grid[k]);
    if (!grids.length || !(w.reduce((t, x) => t + x, 0) > 0)) return null;
    const mean = av.reduce((t, i, j) => t + w[j] * (sc[i].assets[a].mean?.[k] ?? 0), 0);
    const loss = av.reduce((t, i, j) => t + w[j] * (sc[i].assets[a].prob_loss?.[k] ?? 0), 0);
    return { p10: mixQuantile(grids, w, P, 0.1), p50: mixQuantile(grids, w, P, 0.5), p90: mixQuantile(grids, w, P, 0.9), mean, loss, n: grids.length };
  };

  // ---- 2. 情景加权的季度区间
  const fanBox = ui.chartBox();
  const fanTable = h('div');
  const assetSel = ui.select(assets.map(a => ({ value: a, label: assetName(a) })), MIX_STATE.asset, (v) => { MIX_STATE.asset = v; drawFan(); });
  root.append(h('div', { class: 'section-title fx-anchor' }, '情景加权的季度区间'),
    card({ title: '各资产季末累计总回报：情景加权的 10%–90% 区间', actions: assetSel,
      sub: '名义总回报（含股息 / 票息），从今天累计到每个季末；灰色虚线是不加任何情景的无情景基准（第 5 节）。区间是情景假说之下的模拟，不是对会发生什么的判断。' },
    fanBox, fanTable));
  const drawFan = () => {
    const a = MIX_STATE.asset;
    const m = quarters.map((_, k) => mixAt(a, k));
    const ref = eng().reference?.assets?.[a];
    quarterFan(fanBox, { x: ['今天', ...quarters], band: { p10: [0, ...m.map(x => x?.p10 ?? null)], p50: [0, ...m.map(x => x?.p50 ?? null)], p90: [0, ...m.map(x => x?.p90 ?? null)] },
      ref: ref ? { p10: [0, ...(ref.p10 || [])], p50: [0, ...(ref.p50 || [])], p90: [0, ...(ref.p90 || [])] } : null,
      name: '情景加权', refName: '无情景基准', fmt: (v) => pctS(v, 0) });
    const rows = assets.map(x => ({ a: x, m: quarters.map((_, k) => mixAt(x, k)) }));
    put(fanTable, ui.table([
      { key: 'a', label: '资产', fmt: (x) => assetName(x) },
      ...quarters.map((_, k) => ({ value: (r) => r.m[k], label: `${quarters[k]} 末`, align: 'right',
        fmt: (x) => (x ? h('span', { class: `fx-med ${ui.signCls(x.p50)}` }, pctS(x.p50), h('br'), h('small', {}, `${pctS(x.p10, 0)} ~ ${pctS(x.p90, 0)}`)) : '—') })),
      { value: (r) => r.m[nQ - 1]?.loss, label: `P(亏损)`, align: 'right', fmt: (x) => prob(x, 0) },
    ], rows, { stack: T }),
    h('p', { class: 'fx-note' }, `每格：情景加权的中位数，下方小字是 10%–90% 区间；P(亏损) 是到 ${quarters[nQ - 1] || '第 4 季'} 末累计总回报为负的概率（按概率加权）。`));
  };

  // ---- 3. 每个情景的季度中位线 + 「你必须相信什么」
  const scenSel = ui.select(ids.map(i => ({ value: i, label: nameOf(i) })), MIX_STATE.scen, (v) => { MIX_STATE.scen = v; drawScen(); });
  const scenBox = h('div');
  const overview = h('div');
  root.append(h('div', { class: 'section-title fx-anchor' }, '每个情景的季度中位线：你必须相信什么'),
    card({ title: '这个情景说了什么，模型补了什么', actions: scenSel,
      sub: '每格上面是所选引擎模拟的季末中位数，下面是这个引擎在这个季度读到的假设（悬停看原文或说明），没写的标“模型补全”。'
        + '月度引擎运行的是情景库原来的阶段规格（视图与覆盖项）；季度模型运行的是由它换算的季度情景表（QTable），两者不完全相同。' },
    scenBox),
    h('div', { style: { height: '16px' } }),
    card({ title: '所有情景一览：第 4 季末的美股大盘与宏观中位数', sub: '点击一行查看这个情景的逐季表。' }, overview));
  const MACROS = [
    ['policy', '政策利率', 'ff', (v) => lvl(v)], ['growth', '失业率', 'unrate', (v) => lvl(v)], ['inflation', 'CPI 同比', 'cpi_yoy', (v) => lvl(v)],
    ['oil', '油价（累计）', 'oil', (v) => pctS(v, 0)], ['fc', '信用利差（Baa − 10 年）', 'spread', (v) => (isNumber(v) ? v.toFixed(2) : '—')],
  ];
  const cma = ctx.meta?.cma || {};
  const cmaCore = () => (isNumber(cma.reversion) && isNumber(cma.horizon_years)
    ? `月度引擎的资本市场假设让美股的长期漂移含估值回归（今天市价 / 股息 ${isNumber(cma.pd_now) ? fmtNum(cma.pd_now, 0) : '—'} 倍，`
      + `假设 ${fmtNum(cma.horizon_years, 0)} 年内向历史中位数 ${isNumber(cma.pd_median) ? fmtNum(cma.pd_median, 0) : '—'} 倍回归 ${fmtNum(cma.reversion * 100, 0)}%，`
      + `约 ${isNumber(cma.reversion_term) ? sgnNum(cma.reversion_term * 100, 2) : '—'}%/年）；这是所有情景共用的假设，不是这个情景写的`
    : '月度引擎的资本市场假设里含估值回归（所有情景共用，不是这个情景写的）');
  const cmaText = (hasViews = false) => (hasViews ? `此外，${cmaCore()}` : `没有估值视图：倍数不由情景决定，但${cmaCore()}`);
  const drawScen = () => {
    const sid = MIX_STATE.scen;
    const monthly = MIX_STATE.engine === 'monthly';
    let ms = null;
    if (monthly) {
      const got = loadMonthlySpec(ctx.api, sid);
      if (got && typeof got.then === 'function') {
        put(scenBox, h('p', { class: 'muted' }, '正在读取月度引擎的阶段规格…'));
        got.then(() => { if (MIX_STATE.scen === sid && MIX_STATE.engine === 'monthly' && scenBox.isConnected) drawScen(); });
        return;
      }
      ms = got;
    } else {
      loadMonthlySpec(ctx.api, sid);              // the QTable's dropped records, for the note below
    }
    const st = mix.statements?.[sid] || [];
    const blk = eng().scenarios?.[sid] || {};
    const macro = blk.macro || {};
    const rows = [...MACROS.map(([key, zh, mk, f]) => ({ key, zh, mk, f })), { key: 'valuation', zh: '估值模式' }];
    const eqRow = { key: 'equity', zh: '美股大盘（累计）' };
    const phases = ms?.phases || [];
    const mBadges = (k, row) => phasesAt(phases, k).flatMap(ph => phaseItems(ph, ui).filter(it => it.row === row))
      .map(it => h('span', { class: `qb ${it.cls}`, title: [it.text, it.why ? `说明：${it.why}` : ''].filter(Boolean).join('\n') }, it.label));
    const mCompleted = (row) => completedBadge(row === 'fc'
      ? '这个季度的阶段规格没有写信用利差：由月度引擎从历史月份里补全'
      : '这个季度的阶段规格没有写它：由月度引擎从历史月份里补全（不是情景的假设）');
    const known = MSPEC.get(sid);
    const dropped = known && !known.then ? (known.dropped || []) : [];
    const { vName } = monthlyText(ui);
    put(scenBox, ...[
      monthly
        ? ui.notice(ms?.error ? `月度引擎的阶段规格读不出来（${ms.error}）：下面只有中位数。`
          : `月度引擎运行的是情景库原来的阶段规格：下面每格的标记是这个季度所在阶段的视图和覆盖项。季度情景表是它的换算${dropped.length ? `，其中 ${dropped.length} 条没法写进情景表（列在最后）` : ''}。`, 'info')
        : ui.notice(`季度模型（${switched ? '默认' : TRIAL}）运行的是季度情景表：下面每格的标记是情景表这个季度写了的条目。${dropped.length ? `情景表是情景库阶段规格的换算，有 ${dropped.length} 条没法写进去（它们只在月度引擎里起作用）。` : ''}`, 'info'),
      monthly ? h('div', { class: 'qb-legend' },
        h('span', {}, h('span', { class: 'qb qb-stated' }, '情景视图'), '阶段规格里的视图（挑选历史月份或加漂移）'),
        h('span', {}, h('span', { class: 'qb qb-assume' }, '估值假设'), VAL_WORDING),
        h('span', {}, h('span', { class: 'qb qb-derived' }, '因子漂移'), '直接改因子的漂移'),
        h('span', {}, h('span', { class: 'qb qb-anchor' }, '锚'), '这个阶段的长期锚'),
        h('span', {}, h('span', { class: 'qb qb-completed' }, '模型补全'), '没写：由引擎补全')) : badgeLegend(),
      blk.assets && (blk.finite ?? 1) >= 0.5 ? null : ui.notice(`所选引擎没有这个情景的可用结果${blk.error ? '（运行出错）' : ''}：下面只有情景写下的假设。`, 'error'),
      ui.table([
        { key: 'zh', label: '渠道' },
        ...quarters.map((_, k) => ({ value: (r) => r, label: `${quarters[k]} 末`, fmt: (r) => {
          if (r.key === 'equity') {
            const g = blk.assets?.us_equity?.grid?.[k];
            const med = Array.isArray(g) ? g[P.indexOf(50)] : null;
            return h('span', { class: `fx-med ${ui.signCls(med)}` }, pctS(med));
          }
          if (monthly) {
            const bs = mBadges(k, r.key);
            if (r.key === 'valuation') {
              return h('div', {}, h('div', { class: 'fx-med' }, bs.length ? `估值假设（${VAL_WORDING}）` : '资本市场假设的估值回归'),
                h('div', {}, bs.length ? bs : completedBadge(cmaText())));
            }
            const med = (macro[r.mk] || {}).p50?.[k];
            return h('div', {}, h('div', { class: 'fx-med' }, r.f(med)), h('div', {}, bs.length ? bs : mCompleted(r.key)));
          }
          const cell = st[k]?.cells?.[r.key];
          if (r.key === 'valuation') {
            const items = cell?.items || [];
            return h('div', {}, h('div', { class: 'fx-med' }, items.length ? `估值假设（${VAL_WORDING}）` : '由宏观与信用隐含'),
              h('div', {}, items.length ? items.map(badge) : completedBadge('没有估值陈述：季度模型里倍数随模拟的宏观和信用变化，残差倍数零漂移，不回归')));
          }
          const band = macro[r.mk] || {};
          const med = band.p50?.[k];
          return h('div', {}, h('div', { class: 'fx-med' }, r.f(med)),
            h('div', {}, (cell?.items || []).length ? cell.items.map(badge) : completedBadge(r.key === 'fc'
              ? '金融条件未说明：信用利差由模型补全。历史检验里补全的信用利差与无条件抽样相当，而资产结果主要取决于它' : null)));
        } })),
      ], [eqRow, ...rows], { stack: T }),
      monthly
        ? h('div', { style: { marginTop: '12px' } }, h('div', { style: { fontSize: '13px', fontWeight: 600, margin: '0 0 6px' } }, '你必须相信什么（月度引擎运行的阶段规格）'),
          ...phases.filter(ph => ph.start < 3 * nQ).map(ph => {
            const q0 = Math.floor(ph.start / 3) + 1, q1 = Math.min(nQ, Math.ceil((ph.start + (Number(ph.months) || 0)) / 3));
            const items = phaseItems(ph, ui);
            return h('div', { class: 'fx-phase' }, h('b', {}, `第 ${q0 === q1 ? q0 : `${q0}–${q1}`} 季：${ph.name || '（未命名阶段）'}（这个阶段共 ${ph.months} 个月）`),
              items.length ? h('ul', {}, items.map(it => h('li', {}, h('span', { class: `qb ${it.cls}` }, it.label), it.text,
                it.why ? h('span', { class: 'fx-why' }, `说明：${it.why}`) : null)))
                : h('p', { class: 'muted', style: { margin: '2px 0' } }, '这个阶段没有视图：所有因子由月度引擎从历史月份里补全。'));
          }),
          h('p', { class: 'fx-note' }, `${cmaText(phases.filter(ph => ph.start < 3 * nQ).some(ph => phaseItems(ph, ui).some(it => it.row === 'valuation')))}。`),
          dropped.length ? h('div', { class: 'fx-phase' }, h('b', {}, `季度情景表没法写进的 ${dropped.length} 条（季度模型不含它们）`),
            h('ul', {}, dropped.map(c => h('li', {}, `${vName(c.var)}（第 ${span(c.quarters || [])} 季${c.phase_name ? `，${c.phase_name}` : ''}）：${c.note || '没有对应的季度字段'}`)))) : null)
        : h('div', { style: { marginTop: '12px' } }, h('div', { style: { fontSize: '13px', fontWeight: 600, margin: '0 0 6px' } }, '你必须相信什么（季度情景表写下的陈述）'),
          beliefList(st))].filter(Boolean));
    const sc = eng().scenarios || {};
    put(overview, ui.table([
      { key: 'id', label: '情景', fmt: (i) => nameOf(i), wrap: true },
      { value: (r) => r.id, label: '这个引擎里的权重', fmt: (i) => (usable(i) ? prob(Math.max(Number(MIX_STATE.probs[i]) || 0, 0) / Math.max(sumUsable(), 1e-12), 1)
        : h('span', { class: 'muted' }, '未计入（没有可用路径）')) },
      { value: (r) => sc[r.id]?.assets?.us_equity?.grid?.[nQ - 1], label: '美股大盘中位', fmt: (g) => {
        const v = Array.isArray(g) ? g[P.indexOf(50)] : null;
        return h('span', { class: ui.signCls(v) }, pctS(v));
      } },
      { value: (r) => sc[r.id]?.assets?.us_equity?.prob_loss?.[nQ - 1], label: 'P(亏损)', fmt: (x) => prob(x, 0) },
      ...MACROS.filter(([key]) => key !== 'oil').map(([, zh, mk, f]) => ({ value: (r) => sc[r.id]?.macro?.[mk]?.p50?.[nQ - 1], label: zh, fmt: (v) => f(v) })),
      ...(MIX_STATE.engine === 'monthly' ? [] : [{ value: (r) => (mix.statements?.[r.id] || []).slice(0, nQ).map(q => q.cells?.fc?.stance), label: '情景表的金融条件（四季）',
        fmt: (s) => h('span', {}, (s || []).map((x, k) => (x && x !== '未说明' ? h('span', { class: 'qb qb-stated_qual', title: `第 ${k + 1} 季` }, x) : completedBadge()))) }]),
    ], ids.map(id => ({ id })), { stack: T, onRowClick: (r) => { MIX_STATE.scen = r.id; scenSel.value = r.id; drawScen(); scenBox.scrollIntoView({ behavior: 'smooth', block: 'center' }); } }));
  };

  // ---- 4. 概率
  const sumProbs = () => ids.reduce((t, i) => t + Math.max(Number(MIX_STATE.probs[i]) || 0, 0), 0);
  const sumUsable = () => ids.filter(usable).reduce((t, i) => t + Math.max(Number(MIX_STATE.probs[i]) || 0, 0), 0);
  const probBox = h('div', { class: 'fx-probs' });
  const sumEl = h('span', { class: 'fx-sum' });
  const setSel = ui.select([...Object.entries(sets).map(([k, s]) => ({ value: k, label: s.name || k })), { value: 'custom', label: '自己设定' }],
    MIX_STATE.set, (v) => { MIX_STATE.set = v; if (v !== 'custom') MIX_STATE.probs = { ...(sets[v]?.probs || {}) }; paintProbs(); scheduleRedraw(); });
  const inputs = new Map();
  const paintProbs = () => {
    probBox.replaceChildren(...ids.map(i => {
      const inp = h('input', { type: 'number', min: '0', max: '100', step: '0.5', value: String(+((MIX_STATE.probs[i] || 0) * 100).toFixed(2)),
        'aria-label': `${nameOf(i)} 的概率（%）`,
        oninput: () => { MIX_STATE.probs[i] = Math.max(Number(inp.value) || 0, 0) / 100; MIX_STATE.set = 'custom'; setSel.value = 'custom'; paintSum(); scheduleRedraw(); } });
      inputs.set(i, inp);
      const dflt = sets[mix.probabilities?.default]?.probs?.[i];
      return h('label', { class: 'fx-prob' }, h('span', { class: 'nm' }, nameOf(i), h('small', {}, `默认 ${prob(dflt, 1)}`)), inp);
    }));
    paintSum();
  };
  const paintSum = () => {
    const s = sumProbs();
    sumEl.className = `fx-sum${Math.abs(s - 1) > 0.005 ? ' bad' : ''}`;
    sumEl.textContent = s > 0 ? `合计 ${(s * 100).toFixed(1)}%${Math.abs(s - 1) > 0.005 ? '：计算时按比例归一到 100%' : ''}` : '合计为 0：没有可以加权的情景';
  };
  const normalize = () => { const s = sumProbs(); if (s > 0) for (const i of ids) MIX_STATE.probs[i] = (MIX_STATE.probs[i] || 0) / s; MIX_STATE.set = 'custom'; setSel.value = 'custom'; paintProbs(); scheduleRedraw(); };
  const reset = () => { MIX_STATE.set = mix.probabilities?.default; setSel.value = MIX_STATE.set; MIX_STATE.probs = { ...(sets[MIX_STATE.set]?.probs || {}) }; paintProbs(); scheduleRedraw(); };
  const sensBox = ui.chartBox('tall');
  root.append(h('div', { class: 'section-title fx-anchor' }, '概率：唯一带判断的一层'),
    card({ title: `${ids.length} 个情景的概率`, actions: h('div', { class: 'row' }, setSel,
      h('button', { class: 'btn', type: 'button', onclick: normalize }, '归一到 100%'), h('button', { class: 'btn', type: 'button', onclick: reset }, '恢复默认')),
    sub: `${sets[mix.probabilities?.default]?.text || ''}。改一个数，全页立刻按新的概率重算；概率不会自动随今天的状态变化（领先宏观状态还没有通过预注册的预测能力检验）。` },
    probBox, h('div', { style: { marginTop: '8px' } }, sumEl)),
    h('div', { style: { height: '16px' } }),
    card({ title: '每个资产对每个情景概率的敏感度', sub: `第 4 季末累计总回报：该情景的均值 − 情景加权的均值（百分点）。正数 = 调高这个情景的概率会把加权均值往上拉；颜色按 K 线配色。` },
      ui.wideChart(sensBox, 720)));
  const drawSens = () => {
    const sc = eng().scenarios || {};
    const cols = assets.filter(a => a !== 'cash');
    const k = nQ - 1;
    const means = Object.fromEntries(cols.map(a => [a, mixAt(a, k)?.mean]));
    const values = ids.map(i => cols.map(a => {
      const m = sc[i]?.assets?.[a]?.mean?.[k];
      return isNumber(m) && isNumber(means[a]) ? m - means[a] : null;
    }));
    charts.heatmap(sensBox, { rows: ids, cols, values, rowLabels: ids.map(i => `${nameOf(i)}（${prob((MIX_STATE.probs[i] || 0) / Math.max(sumProbs(), 1e-12), 0)}）`),
      colLabels: cols.map(assetName), fmt: 'pct', height: Math.max(380, 28 * ids.length + 90), colors: charts.klineDiverging(), rowLabelWidth: 210,
      tip: (r, c, v) => `<b>${ui.esc(nameOf(ids[r]))}</b> × ${ui.esc(assetName(cols[c]))}<br/>该情景均值 − 加权均值：<b>${ui.esc(pctS(v))}</b>` });
  };

  // ---- 5. 无情景基准（历史常态）
  const refBox = h('div');
  root.append(h('div', { class: 'section-title fx-anchor' }, '无情景基准（历史常态）'),
    card({ title: '不加任何情景时的季末区间', sub: '同一个引擎、不写任何情景陈述（季度模型）或不加任何视图（月度引擎）：历史常态的参照线，也画在第 2 节的图里。' }, refBox));
  const drawRef = () => {
    const ref = eng().reference;
    const an = mix.analog;
    const k = nQ - 1;
    const rows = assets.map(a => ({ a, ref: ref?.assets?.[a], m: mixAt(a, k) }));
    const diff = (an?.differs || []).filter(a => an.assets?.[a]);
    const t = an?.paired_t;
    put(refBox, 
      ref ? ui.table([
        { key: 'a', label: '资产', fmt: (x) => assetName(x) },
        { value: (r) => r.ref, label: `无情景基准（${quarters[k]} 末）`, fmt: (x) => (x ? h('span', { class: 'fx-med' }, pctS(x.p50?.[k]), h('br'), h('small', {}, `${pctS(x.p10?.[k], 0)} ~ ${pctS(x.p90?.[k], 0)}`)) : '—') },
        { value: (r) => r.m, label: '情景加权 中位', fmt: (x) => (x ? h('span', { class: ui.signCls(x.p50) }, pctS(x.p50)) : '—') },
        { value: (r) => (r.m && r.ref ? r.m.p50 - r.ref.p50?.[k] : null), label: '差（百分点）', fmt: (x) => (isNumber(x) ? h('span', { class: ui.signCls(x) }, pctS(x)) : '—') },
      ], rows, { stack: T }) : ui.notice(`${eng().name || '这个引擎'}没有无情景基准（构建时没有算出）。`, 'info'),
      h('p', { class: 'fx-note' }, `${ref?.name || ''}${ref?.n_paths ? `，${ref.n_paths} 条路径` : ''}。`),
      an ? h('div', { style: { marginTop: '10px' } },
        h('div', { style: { fontSize: '13px', fontWeight: 600, margin: '0 0 4px' } }, '相似起点法（12 个月，回放口径）'),
        h('p', { style: { margin: '0 0 6px', fontSize: '13px' } }, '相似起点法与无条件抽样相当：',
          t ? `逐起点配对的风险资产 CRPS 差在随机误差之内（${an.n_seeds} 组随机种子，t 在 ${t[0]} 到 ${t[1]} 之间）；` : '',
          diff.length ? `只在区间明显不同（任一分位差 ≥ 2 个百分点）的资产上列出它：` : '它的 12 个月区间与无条件抽样几乎相同，所以不单列。'),
        diff.length ? ui.table([
          { key: 'a', label: '资产', fmt: (x) => assetName(x) },
          { value: (r) => an.assets[r.a].analog, label: '相似起点 5% / 中位 / 95%', fmt: (b) => `${pctS(b.p5, 0)} / ${pctS(b.p50, 0)} / ${pctS(b.p95, 0)}` },
          { value: (r) => an.assets[r.a].uncond, label: '无条件 5% / 中位 / 95%', fmt: (b) => `${pctS(b.p5, 0)} / ${pctS(b.p50, 0)} / ${pctS(b.p95, 0)}` },
        ], diff.map(a => ({ a })), { stack: T }) : null) : null);
  };

  // ---- 6. 情景组合隐含的宏观基率
  const rateBox = h('div');
  root.append(h('div', { class: 'section-title fx-anchor' }, '情景组合隐含的宏观基率'),
    card({ title: '按概率加权之后，未来一年的宏观基率', sub: '每个情景的比例按概率加权；与历史（1973 年以来每个季度作起点）和无情景基准对照。差别大说明情景组合整体偏向某一种宏观状态——这是概率的判断，需要单独说服自己。' }, rateBox));
  const drawRates = () => {
    const sc = eng().scenarios || {};
    const names = mix.rate_names || {};
    const keys = Object.keys(names);
    const hist = mix.history_rates || {};
    const refR = eng().reference?.rates || {};
    const excluded = new Set(eng().excluded || []);
    const rows = keys.map(key => {
      const av = ids.filter(i => !excluded.has(i) && isNumber(sc[i]?.rates?.[key]));
      const raw = av.map(i => Math.max(Number(MIX_STATE.probs[i]) || 0, 0));
      const tot = raw.reduce((t, x) => t + x, 0);
      const v = tot > 0 ? av.reduce((t, i, j) => t + (raw[j] / tot) * sc[i].rates[key], 0) : null;
      return { key, name: names[key], mix: v, hist: hist[key], ref: refR[key], n: av.length };
    });
    put(rateBox, ui.table([
      { key: 'name', label: '基率', wrap: true },
      { key: 'mix', label: '情景组合', fmt: (x) => h('b', {}, prob(x, 0)) },
      { key: 'hist', label: `历史（${hist.first || '1973Q1'} 起）`, fmt: (x) => prob(x, 0) },
      { key: 'ref', label: '无情景基准', fmt: (x) => prob(x, 0) },
      { key: 'n', label: '有结果的情景', fmt: (x) => `${x} 个` },
    ], rows, { stack: T }),
    h('p', { class: 'fx-note' }, `${hist.note || ''}。月度引擎的收缩比例来自它自己的月度体制标签，季度模型用季末序列的同一套规则；两者口径接近但不完全相同。`),
    ui.notice(mix.sep?.text || 'SEP 暂无。', 'info'));
  };

  // ---- 7. 前瞻账本
  const led = mix.ledger || {};
  root.append(h('div', { class: 'section-title fx-anchor' }, '前瞻账本'),
    card({ title: '前瞻账本：每次构建存档的情景加权分布', sub: '每次构建把当时（默认概率下）的情景加权季度分位原样追加到前瞻账本，'
      + '带模型、变量卡片和概率的版本；只追加、不改写。季度过去之后按实际结果打分（80% 区间覆盖率、PIT、CRPS）。' },
    ui.notice(led.text || '需要很多年才能分辨中等程度的偏差。', 'info'),
    h('p', { class: 'fx-note' }, `账本里有 ${led.n_entries ?? 0} 条记录。`),
    (led.recent || []).length ? ui.table([
      { key: 'origin', label: '起点' },
      { key: 'engine', label: '引擎', fmt: (e) => mix.engines?.[e]?.name || (e === 'quarterly' ? '季度模型（试运行）' : '月度引擎') },
      { key: 'created', label: '记录于（UTC）', fmt: (x) => String(x || '').replace('T', ' ').replace('Z', '') },
      { value: (r) => r.us_equity?.[3] || r.us_equity?.[r.us_equity.length - 1], label: '美股大盘 第 4 季末 中位（10%–90%）',
        fmt: (b) => (b ? `${pctS(b.p50)}（${pctS(b.p10, 0)} ~ ${pctS(b.p90, 0)}）` : '—') },
      { value: (r) => (led.entries || []).find(e => e.origin === r.origin && e.engine === r.engine && e.created === r.created), label: '已到期 / 打分',
        fmt: (s) => (s?.matured ? `${s.matured} 季：覆盖 ${prob(s.coverage80, 0)}，PIT 偏离 ${s.pit_mad?.toFixed(2)}` : '待到期') },
    ], led.recent, { stack: T }) : null,
    Object.keys(led.summary || {}).length ? ui.table([
      { key: 'e', label: '引擎', fmt: (e) => mix.engines?.[e]?.name || e },
      { key: 'n', label: '已打分的季度 × 资产' },
      { key: 'coverage80', label: '80% 区间覆盖率', fmt: (x) => prob(x, 0) },
      { key: 'pit_mad', label: 'PIT 偏离（理想 0.25）', fmt: (x) => (isNumber(x) ? x.toFixed(2) : '—') },
      { key: 'crps', label: 'CRPS', fmt: (x) => (isNumber(x) ? x.toFixed(4) : '—') },
    ], Object.entries(led.summary).map(([e, s]) => ({ e, ...s })), { stack: T }) : h('p', { class: 'fx-note' }, '还没有到期的记录：第一个季度过去之后，下一次构建会给出第一批打分。')));

  // ---- 计算口径与局限
  root.append(h('div', { style: { height: '16px' } }), h('details', { class: 'card qd-more' }, h('summary', { style: { fontWeight: 600 } }, '计算口径与局限'),
    h('ul', { class: 'list', style: { fontSize: '13px', marginTop: '8px' } },
      h('li', {}, `情景加权 = 各情景分布按概率的混合：混合分布的累计分布函数是各情景累计分布函数的加权和，它在各网格点之间是线性的，分位数在网格点的并集上精确反解；各情景的分布由构建时的分位网格表示（${P.length} 个分位点，含最小值和最大值，之间线性插值）。季度模型里模型自己标为不可信的发散路径不进入区间。`),
      h('li', {}, '月度引擎的数字是“未来情景”页同一批路径的前 12 个月按季末汇总；季度模型的数字是每个内置情景的季度情景表在路径模式下的前四季（后面季度的陈述也约束前四季）。'),
      h('li', {}, `季度模型在切换关 T-S 通过之前只是${TRIAL}；切换之前，默认引擎是月度引擎。`),
      h('li', {}, '“你必须相信什么”只列情景写下的陈述；其余因子由模型补全，补全不是情景的假设。'),
      h('li', {}, `预测只在情景概率里。概率是手设的主观判断，${scenarios.length} 个互相重叠的长期情景只有一个实现，无法校准；能检查的只有概率合计、隐含的宏观基率和前瞻账本。`),
      h('li', {}, '所有结果都是假设性的模拟，不构成投资建议。'))));

  // ---- drawing
  let timer = 0;
  const scheduleRedraw = () => { clearTimeout(timer); timer = setTimeout(redrawData, 150); };
  const safe = (fn) => { try { fn(); } catch (err) { console.error(err); } };
  function redrawData() { safe(drawFan); safe(drawScen); safe(drawSens); safe(drawRef); safe(drawRates); }
  function redrawAll() { paintEngineNote(); paintHero(); redrawData(); }
  paintEngineNote();
  paintProbs();
  later.push(redrawData);
  for (const fn of later) safe(fn);
}

// ---------------------------------------------------------------- 历史回放（相似起点法）: forecast.json
function renderReplay(root, fc, scen, { charts, ui }) {
  const { h, card } = ui;
  const pct = (x, d = 1) => ui.fmtPct(x, d);
  const names = fc.asset_names || {};
  const assetName = (a) => names[a] || ui.assetName(a);
  const live = fc.live || {};

  // ---------------------------------------------------------------- next 12 months
  root.append(h('div', { class: 'section-title' }, `相似起点法：从今天（${live.origin || '—'}）出发的 12 个月`));
  const liveMethods = ['analog', 'uncond', 'scenarios'].filter(m => fc.assets.some(a => live.assets?.[a]?.[m]));
  const boxEl = ui.chartBox('tall');
  root.append(card({
    title: '各资产未来 12 个月总回报的分布',
    sub: '箱体为 25%–75%，须线为 5%–95%，中线为中位数。相似起点 = 历史上状态与今天相近的起点之后那一年；'
      + '情景加权 = 情景库每个情景的第一年按概率加权。三者都经引擎按今天的利率、估值和股息率重新锚定。',
  }, boxEl));

  const liveRows = fc.assets.map(a => ({ a, ...live.assets?.[a] }));
  root.append(card({ title: '分位数一览', sub: '名义总回报（含股息/票息）；“实际”扣除同一路径的通胀；P(亏损) 为相似起点法下 12 个月名义收益为负的概率' },
    ui.table([
      { key: 'a', label: '资产', fmt: (v) => assetName(v) },
      ...liveMethods.flatMap(m => [
        { value: (r) => r[m]?.p5, label: `${METHOD_CN[m]} 5%`, fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
        { value: (r) => r[m]?.p50, label: `${METHOD_CN[m]} 中位`, fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
        { value: (r) => r[m]?.p95, label: `${METHOD_CN[m]} 95%`, fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
      ]),
      { value: (r) => r.analog?.real_p50, label: '相似起点 实际中位', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
      { value: (r) => r.analog?.prob_loss, label: 'P(亏损)', fmt: (v) => pct(v, 0) },
    ], liveRows, { stack: WIDE })));

  const zEl = ui.chartBox();
  const macroLive = live.macro || {};
  root.append(h('div', { class: 'grid cols-2 text' },
    card({ title: '今天的状态离历史均值多远', sub: '以 1973 年以来的均值和标准差衡量；相似起点法按这 8 个状态找历史上相近的月份。'
      + '偏态变量（如股息率）的极端程度会被标准差低估，历史分位见总览' }, zEl),
    card({ title: '宏观变量未来 12 个月的变化', sub: '模拟路径第 12 个月相对今天的变化（百分点；油价为涨跌幅），中位数与 90% 区间' },
      ui.table([
        { key: 'label', label: '变量' },
        ...['analog', 'uncond'].flatMap(m => [
          { value: (r) => r[m]?.p50, label: `${METHOD_CN[m]} 中位`, fmt: (v, r) => fmtMacro(v, r.kind) },
          { value: (r) => r[m], label: `${METHOD_CN[m]} 90% 区间`, fmt: (b, r) => (b ? `${fmtMacro(b.p5, r.kind)} ~ ${fmtMacro(b.p95, r.kind)}` : '—') },
        ]),
      ], MACRO_ROWS.map(([k, label, kind]) => ({ label, kind, ...macroLive[k] })), { stack: T }))));

  if ((live.analogs || []).length) {
    const share = isFin(live.analog_share) ? `可用起点的 ${shareText(live.analog_share)}，` : '';
    root.append(card({
      title: '今天最像的历史起点，以及之后一年发生了什么',
      sub: `按 8 个状态的标准化距离排序（只用于展示的窄核，有效起点约 ${Math.round(live.display_ens || 0)}）；区间本身用的是滚动验证选出的较宽加权（${share}有效起点约 ${Math.round(live.analog_ens || 0)}）`,
    }, ui.table([
      { key: 'date', label: '起点' },
      { key: 'distance', label: '距离', fmt: (v) => ui.fmtNum(v, 2) },
      ...NEXT_ASSETS.map(a => ({ value: (r) => r.next12?.[a], label: assetName(a), fmt: (v) => pct(v), cls: (v) => ui.signCls(v) })),
      { value: (r) => r.macro?.unrate, label: '失业率变化', fmt: (v) => fmtMacro(v, 'pp') },
      { value: (r) => r.macro?.ff, label: '政策利率变化', fmt: (v) => fmtMacro(v, 'pp') },
    ], live.analogs, { stack: MID })));
  }

  const scenMed = live.scenario_medians;
  let heatEl = null;
  if (scenMed && scen?.scenarios?.length) {
    heatEl = ui.chartBox('tall');
    root.append(card({ title: '每个情景的第一年（月度引擎）', sub: '各情景前 12 个月名义总回报的中位数，颜色按 K 线配色（涨色为正）；情景按概率从高到低排列。多年之后的分化见“未来情景”' }, heatEl));
  }

  // ---------------------------------------------------------------- reconciliation with the scenario library
  const drawRecon = renderReconciliation(root, { ui, charts, fc, pct, assetName });

  // ---------------------------------------------------------------- live ledger
  root.append(h('div', { class: 'section-title' }, '相似起点法的实时样本外账本'));
  root.append(renderLedger({ ui, fc, pct, assetName }));

  // ---------------------------------------------------------------- track record
  root.append(h('div', { class: 'section-title' }, '相似起点法的滚动一年期成绩单（回放）'));
  if (fc.text) root.append(h('p', { class: 'lead' }, fc.text));
  const kpiWrap = h('div', { class: 'grid cols-4' });
  let scope = 'all';
  // a CRPS difference is called better / worse only beyond the module's claim rule (forecast.json selection.claim)
  const claim = fc.selection?.claim || { min_skill: 0.005, min_t: 2 };
  const verdict = (x, t) => (isFin(x) && isFin(t)
    ? (x > claim.min_skill && t >= claim.min_t ? 1 : (x < -claim.min_skill && t <= -claim.min_t ? -1 : 0)) : 0);
  const renderKpis = () => {
    const s = fc.scores?.[scope] || {};
    const an = s.analog || {}, un = s.uncond || {}, nv = s.naive || {};
    const sk = s.crps_skill_risk || {}, pr = s.crps_paired_risk || {};
    const cov = an.risk_avg?.coverage90;
    const vNaive = verdict(sk.analog_vs_naive, pr.analog_vs_naive?.t);
    kpiWrap.innerHTML = '';
    kpiWrap.append(
      ui.kpi('90% 区间覆盖率（风险资产）', pct(cov, 0), `理想值 90%；无条件 ${pct(un.risk_avg?.coverage90, 0)}，历史均值 ${pct(nv.risk_avg?.coverage90, 0)}`,
        typeof cov === 'number' && Math.abs(cov - 0.9) > 0.08 ? 'is-bad' : ''),
      ui.kpi('分布准确度 CRPS 技能', fmtSkill(sk.analog_vs_naive),
        `相对历史均值基准（t ${ui.fmtNum(pr.analog_vs_naive?.t, 1)}）；相对无条件 ${fmtSkill(sk.analog_vs_uncond)}`
        + `（t ${ui.fmtNum(pr.analog_vs_uncond?.t, 1)}）。正 = 更好；|t| 不到 ${claim.min_t} 时在随机误差之内`,
        vNaive ? ui.goodCls(vNaive > 0) : ''),
      ui.kpi('超额收益方向命中率', pct(an.direction?.hit_rate, 0), `同期上涨年份占比 ${pct(an.direction?.base_rate_up, 0)}：只猜“涨”的水平`),
      ui.kpi('跨资产排序相关 IC', an.rank_ic?.mean === null || an.rank_ic?.mean === undefined ? '—' : an.rank_ic.mean.toFixed(2),
        an.rank_ic?.t ? `t = ${an.rank_ic.t.toFixed(1)}，${an.rank_ic.n} 个起点：预测能否排出明年哪类资产更好` : ''),
    );
  };
  root.append(card({
    title: '总体成绩', sub: `${(fc.origins || []).length} 个起点，其中 ${fc.n_shock_years ?? '—'} 个年份含外生冲击`,
    actions: ui.seg([{ value: 'all', label: '全部年份' }, { value: 'ex_shock', label: '剔除外生冲击' }], scope,
      (v) => { scope = v; renderKpis(); renderScoreTable(); }),
  }, kpiWrap));
  renderKpis();

  // selection of the analog kernel's concentration (a share of the starts available at each origin)
  const sel = fc.selection;
  let selEl = null;
  const gridLabel = (r) => r.label ?? String(r.ens ?? '—');           // older forecast.json: an ENS grid
  if (sel?.grid?.length) {
    selEl = ui.chartBox();
    // the analog's CRPS on top; below in small type its skill against the unconditional draw and the paired t
    const crpsCell = (a, u, t) => (isFin(a) ? h('span', {}, ui.fmtNum(a, 4), h('br'),
      h('small', {}, `技能 ${fmtSkill(isFin(u) && u > 0 ? 1 - a / u : null)}，t ${ui.fmtNum(t, 1)}`)) : '—');
    const g0 = sel.grid[0];
    root.append(card({ title: '“像哪段历史”该押多重：滚动验证的选择', sub: sel.text || '' }, selEl,
      ui.table([
        { value: (r) => r, label: '集中度（占可用起点）', fmt: (r) => (r.share === sel.chosen ? `${gridLabel(r)}（选中）` : gridLabel(r)) },
        { value: (r) => r, label: '有效起点（首个 → 最近）',
          fmt: (r) => (isFin(r.ens_first) ? `${Math.round(r.ens_first)} → ${Math.round(r.ens_last)}` : '—') },
        { value: (r) => r, label: `验证期 CRPS（≤${sel.split}）`, fmt: (r) => crpsCell(r.val_crps_analog, r.val_crps_uncond, r.val_t) },
        { value: (r) => r, label: '检验期 CRPS', fmt: (r) => crpsCell(r.test_crps_analog, r.test_crps_uncond, r.test_t) },
        { key: 'test_coverage', label: '检验期覆盖率', fmt: (v) => pct(v, 0) },
        { key: 'test_rank_ic', label: '检验期 IC', fmt: (v) => ui.fmtNum(v, 2) },
      ], sel.grid, { stack: T }),
      h('p', { class: 'muted', style: { fontSize: '12px', margin: '6px 0 0' } },
        `无条件抽样的 CRPS：验证期 ${ui.fmtNum(g0.val_crps_uncond, 4)}，检验期 ${ui.fmtNum(g0.test_crps_uncond, 4)}，每一行都一样`
        + (sel.common_random_numbers ? '（与相似起点法用同一组随机数）' : '')
        + '；小字是相似起点法相对它的 CRPS 技能（正 = 更好）和逐起点差别的配对 t 值。')));
  }

  // per-asset forecast vs realized
  let asset = 'us_equity', method = 'analog';
  const trackEl = ui.chartBox('tall');
  const assetSel = ui.select(fc.assets.map(a => ({ value: a, label: assetName(a) })), asset, (v) => { asset = v; renderTrack(); });
  const methodSeg = ui.seg(METHODS.map(m => ({ value: m, label: METHOD_CN[m] })), method, (v) => { method = v; renderTrack(); });
  root.append(card({
    title: '每年的一年期区间 vs 实际',
    sub: '每根“K 线”是一次相似起点法的回放：实体 = 25%–75%，影线 = 5%–95%；圆点是实际 12 个月回报（红色 = 落在 90% 区间之外）；橙色背景 = 含外生冲击的年份',
    actions: h('div', { class: 'row' }, assetSel, methodSeg),
  }, trackEl));

  const scoreWrap = h('div');
  root.append(card({ title: '分资产成绩', sub: '覆盖率越接近 90% 越好；PIT 偏离越接近 0.25 越好（越大说明实际常落在尾部）；CRPS 和中位数误差越小越好' }, scoreWrap));
  const renderScoreTable = () => {
    const s = fc.scores?.[scope] || {};
    const rows = fc.assets.map(a => ({ a, ...Object.fromEntries(METHODS.map(m => [m, s[m]?.assets?.[a]])) }));
    scoreWrap.innerHTML = '';
    scoreWrap.append(ui.table([
      { key: 'a', label: '资产', fmt: (v) => assetName(v) },
      ...METHODS.map(m => ({ value: (r) => r[m]?.coverage90, label: `${METHOD_CN[m]} 覆盖率`, fmt: (v) => pct(v, 0),
        cls: (v) => (typeof v === 'number' && Math.abs(v - 0.9) > 0.1 ? 'is-bad' : '') })),
      ...METHODS.map(m => ({ value: (r) => r[m]?.crps, label: `${METHOD_CN[m]} CRPS`, fmt: (v) => ui.fmtNum(v, 3) })),
      ...['analog', 'uncond'].map(m => ({ value: (r) => r[m]?.pit_mad, label: `${METHOD_CN[m]} PIT 偏离`, fmt: (v) => ui.fmtNum(v, 2) })),
      { value: (r) => r.analog?.bias_p50, label: '相似起点 中位数偏差', fmt: (v) => ui.fmtPct(v, 1, true),
        cls: (v) => (typeof v === 'number' && Math.abs(v) > 0.03 ? 'is-bad' : '') },
    ], rows, { stack: WIDE }));
  };
  renderScoreTable();

  const pitEl = ui.chartBox(), icEl = ui.chartBox();
  root.append(h('div', { class: 'grid cols-2' },
    card({ title: 'PIT 分布（校准检查）', sub: '实际值在相似起点法分布里的分位；校准良好时每一格都接近 10%，两端偏高 = 区间太窄，中间偏高 = 区间太宽' }, pitEl),
    card({ title: '每年的跨资产排序相关（IC）', sub: '相似起点法中位数的资产排序与实际排序的相关；长期为正说明能分辨明年哪些资产相对更好' }, icEl)));

  // misses
  const misses = fc.misses || [];
  if (misses.length) {
    const counts = Object.fromEntries(Object.keys(CAUSE).map(c => [c, misses.filter(m => m.cause === c).length]));
    root.append(card({
      title: '偏离最大的年份：为什么不准',
      sub: `外生冲击 ${counts.shock} 次（不算模型错误）；宏观意外 ${counts.macro} 次（失业率、政策利率、CPI、信用利差或油价的实际变化落在相似起点法 90% 区间之外，偏高偏低都算：一年期本来就难）；`
        + `资产定价偏离 ${counts.pricing} 次（宏观大体符合预期，资产自己走偏：估值或盈利，正是需要补的因子/传导机制）`,
    }, ui.table([
      { key: 'origin', label: '起点' },
      { key: 'asset', label: '资产', fmt: (v) => assetName(v) },
      { key: 'realized', label: '实际', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
      { value: (r) => `${pct(r.p5)} ~ ${pct(r.p95)}`, label: '90% 区间' },
      { key: 'cause', label: '原因', fmt: (v) => ui.chip(...(CAUSE[v] || [v, ''])) },
      { key: 'why', label: '说明', wrap: true },
    ], misses, { stack: T })));
  }

  if ((fc.notes || []).length) {
    root.append(card({ title: '说明' }, h('ul', { class: 'list' }, fc.notes.map(n => h('li', {}, n)))));
  }

  // ---------------------------------------------------------------- charts (after the DOM exists)
  renderLiveBoxes(charts, boxEl, fc, liveMethods, assetName);
  const z = live.z_today || {};
  charts.bars(zEl, {
    x: Object.keys(z).map(k => STATE_CN[k] || k), series: [{ name: '今天（标准差）', data: Object.values(z).map(v => +v.toFixed(2)) }],
    horizontal: true, legend: false, fmt: (v) => Number(v).toFixed(1),
    markLine: [{ xAxis: 2, name: '+2σ' }, { xAxis: -2, name: '−2σ' }],
  });
  if (drawRecon) drawRecon();
  if (heatEl) {
    const order = scen.scenarios.slice().sort((a, b) => (b.prob ?? 0) - (a.prob ?? 0)).filter(s => scenMed.us_equity?.[s.id] !== undefined);
    const cols = fc.assets.filter(a => a !== 'cash');
    charts.heatmap(heatEl, {
      rows: order.map(s => s.id), cols, values: order.map(s => cols.map(a => scenMed[a]?.[s.id] ?? null)),
      rowLabels: order.map(s => `${s.name}（${pct(s.prob, 0)}）`), colLabels: cols.map(assetName), fmt: 'pct',
      height: Math.max(360, 28 * order.length + 90), colors: charts.klineDiverging(),     // returns: K-line colours
    });
  }
  if (selEl) {
    charts.line(selEl, {
      x: sel.grid.map(gridLabel), yFmt: (v) => Number(v).toFixed(3),
      series: [
        { name: '验证期 相似起点', data: sel.grid.map(r => r.val_crps_analog), color: '#3b5bdb' },
        { name: '验证期 无条件', data: sel.grid.map(r => r.val_crps_uncond), color: '#3b5bdb', dashed: true },
        { name: '检验期 相似起点', data: sel.grid.map(r => r.test_crps_analog), color: '#e8590c' },
        { name: '检验期 无条件', data: sel.grid.map(r => r.test_crps_uncond), color: '#e8590c', dashed: true },
      ],
    });
  }
  const renderTrack = () => renderTrackChart(charts, trackEl, fc, asset, method);
  renderTrack();
  renderPit(charts, pitEl, fc);
  const ic = fc.scores?.all?.analog?.rank_ic?.by_origin || [];
  charts.bars(icEl, { x: (fc.origins || []).slice(0, ic.length).map(yearLabel), series: [{ name: 'IC', data: ic.map(v => +v.toFixed(3)) }],
    legend: false, fmt: (v) => Number(v).toFixed(2) });
}

const isFin = (v) => typeof v === 'number' && Number.isFinite(v);
// A share of the available starts as text, as forecast.share_label: 0.25 -> "1/4", else a percentage.
const shareText = (s) => {
  const k = 1 / s;
  return k >= 1 && Math.abs(k - Math.round(k)) < 1e-9 ? `1/${Math.round(k)}` : `${Math.round(s * 100)}%`;
};
// Chinese text followed by a Latin word gets a space before it ("只对齐 CPI 同比").
const latinGap = (t) => (/^[A-Za-z0-9]/.test(String(t)) ? ` ${t}` : String(t));
const subHead = (ui, text) => ui.h('div', { style: { fontSize: '12px', fontWeight: 600, color: 'var(--ink-2)', margin: '14px 0 6px' } }, text);
// Percentage points of a decimal difference: 0.013 -> "+1.3 pp".
const ppText = (v, d = 1) => (isFin(v) ? `${signedText(v * 100, d)} pp` : '—');
const MACRO_NAME = Object.fromEntries(MACRO_ROWS.map(([k, label]) => [k, label.replace(/(变化|涨跌)$/, '')]));

// The scenario library's first year against the scored analog forecast: the gap, which scenarios make it
// (per asset), the calibrated probabilities and what each macro view alone would do. Appends its cards and
// returns a function that draws the chart once they are in the page (null for an older forecast.json).
function renderReconciliation(root, { ui, charts, fc, pct, assetName }) {
  const { h, card } = ui;
  const live = fc.live || {};
  const rec = live.reconciliation, cal = live.calibration || {};
  if (!rec?.summary?.us_equity) {
    if (cal.reason) {                    // e.g. scenario paths from another start month: say why there is no reconciliation
      root.append(h('div', { class: 'section-title' }, '情景概率与相似起点法的对账'),
        card({}, ui.notice(`暂时无法对账：${cal.reason}。`, 'info')));
    }
    return null;
  }
  const skill = fc.macro_skill || {};
  const calOk = !!cal.available;
  const eq = rec.summary.us_equity;
  const cut = String(skill.split || '2004-12').slice(0, 4);
  root.append(h('div', { class: 'section-title' }, '情景概率与相似起点法的对账'));
  root.append(h('div', { class: 'grid cols-4' },
    ui.kpi('相似起点法：美股大盘第一年', pct(eq.analog.p50), `中位数；P(亏损) ${pct(eq.analog.prob_loss, 0)}。滚动打过分的方法`, ui.signCls(eq.analog.p50)),
    ui.kpi('情景加权（模型后验）', pct(eq.scenarios.p50), `P(亏损) ${pct(eq.scenarios.prob_loss, 0)}；比相似起点 ${ppText(eq.gap_p50)}`, ui.signCls(eq.scenarios.p50)),
    ui.kpi('情景加权（按一年期预测校准）', calOk ? pct(eq.calibrated?.p50) : '未校准',
      calOk ? `P(亏损) ${pct(eq.calibrated?.prob_loss, 0)}；比相似起点 ${ppText(eq.calibrated_gap_p50)}`
        : '没有宏观变量通过预测能力检验，见下方「校准」',
      calOk ? ui.signCls(eq.calibrated?.p50) : ''),
    ui.kpi('校准离模型后验多远', calOk ? `KL ${ui.fmtNum(cal.kl, 3)}` : '—',
      calOk ? `有效情景数 ${ui.fmtNum(cal.ens_prior, 1)} → ${ui.fmtNum(cal.ens, 1)}；只对齐${latinGap((cal.views || []).map(v => v.name).join('、'))}`
        : '未校准：情景概率保持模型后验')));
  root.append(h('div', { style: { height: '16px' } }), card({ title: '为什么情景加权的第一年更悲观',
    sub: `同一个起点（${live.origin || '—'}）、同一套引擎；情景各 ${rec.n_paths ?? '—'} 条路径（与「未来情景」同一批），相似起点法 ${fc.n_paths ?? '—'} 条` },
  h('ul', { class: 'list' }, (rec.text || []).map(t => h('li', { style: { margin: '6px 0' } }, t)))));

  const rows = (rec.assets || []).map(a => ({ a, ...rec.summary[a] }));
  root.append(h('div', { style: { height: '16px' } }), card({ title: '各资产第一年：相似起点 vs 情景加权',
    sub: '名义总回报中位数与 P(亏损)；均值差距 = 情景加权均值 − 相似起点均值（百分点），下图把它按情景拆开' },
  ui.table([
    { key: 'a', label: '资产', fmt: (v) => assetName(v) },
    { value: (r) => r.analog?.p50, label: '相似起点 中位', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
    { value: (r) => r.scenarios?.p50, label: '情景加权 中位', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
    ...(calOk ? [{ value: (r) => r.calibrated?.p50, label: '校准后 中位', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) }] : []),
    { value: (r) => r.analog?.prob_loss, label: '相似起点 P(亏损)', fmt: (v) => pct(v, 0) },
    { value: (r) => r.scenarios?.prob_loss, label: '情景加权 P(亏损)', fmt: (v) => pct(v, 0) },
    ...(calOk ? [{ value: (r) => r.calibrated?.prob_loss, label: '校准后 P(亏损)', fmt: (v) => pct(v, 0) }] : []),
    { key: 'gap_mean', label: '均值差距', fmt: (v) => ppText(v), cls: (v) => ui.signCls(v) },
  ], rows, { stack: T })));

  const heatBox = ui.chartBox('tall');
  root.append(h('div', { style: { height: '16px' } }), card({ title: '哪些情景造成了差距（按资产）',
    sub: '每格 = 情景概率 ×（该情景第一年均值 − 相似起点法均值），单位百分点；每一列加起来正好等于最下面一行的均值差距。'
      + '颜色按 K 线配色（涨色 = 把情景加权往上拉）；悬停可看 P(亏损) 差距的同样拆分' }, ui.wideChart(heatBox, 760)));

  // ---- front-loading: each scenario's first year against its own 15 years
  const scs = (rec.scenarios || []).slice().sort((x, y) => (y.prob ?? 0) - (x.prob ?? 0));
  const macroOf = (key) => Object.fromEntries(Object.entries(rec.macro || {}).map(([k, m]) => [k, m[key]]));
  const refs = [
    { name: '相似起点法（参照）', ref: true, year1: { us_equity: { p50: eq.analog.p50 } }, macro: macroOf('analog_mean') },
    { name: '情景加权（模型后验）', ref: true, year1: { us_equity: { p50: eq.scenarios.p50 } }, macro: macroOf('scenarios_mean') },
    ...(calOk ? [{ name: '情景加权（校准后）', ref: true, year1: { us_equity: { p50: eq.calibrated?.p50 } }, macro: macroOf('calibrated_mean') }] : []),
  ];
  root.append(h('div', { style: { height: '16px' } }), card({ title: '冲击放在第一年：每个情景的第一年 vs 它自己的 15 年',
    sub: '第一年 = 情景路径的前 12 个月（与情景加权同一批路径）；15 年年化 = 整个情景的年化中位数；宏观为第一年 12 个月的平均变化'
      + '（百分点，油价为涨跌幅）。冲击类情景的第一年远差于它自己的 15 年平均，失业率在第一年就上升' },
  ui.table([
    { key: 'name', label: '情景', wrap: true, minWidth: 150, fmt: (v, r) => (r.ref ? h('b', {}, v) : v) },
    { key: 'prob', label: '模型后验', fmt: (v) => pct(v, 1) },
    ...(calOk ? [{ key: 'prob_calibrated', label: '校准后', fmt: (v) => pct(v, 1) }] : []),
    { value: (r) => r.year1?.us_equity?.p50, label: '美股第一年中位数', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
    { value: (r) => r.cagr15_p50?.us_equity, label: '美股 15 年年化', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
    ...['unrate', 'cpi_yoy', 'ff', 'spread', 'oil'].map(k => ({ value: (r) => r.macro?.[k], label: MACRO_NAME[k],
      fmt: (v) => fmtMacro(v, k === 'oil' ? 'log' : 'pp') })),
  ], [...scs, ...refs], { stack: MID })));

  // ---- calibration: only forecastable macro variables
  const sv = Object.entries(skill.vars || {}).map(([k, v]) => ({ k, ...v }));
  // one comparison in one cell: all origins on top, then (up to the split / after, paired t) in small type
  const skillCell = (p) => (p?.all && isFin(p.all.skill)
    ? h('span', {}, h('b', { style: { fontWeight: 600 } }, fmtSkill(p.all.skill)), h('br'),
      h('small', { style: { opacity: 0.85 } }, `${fmtSkill(p.val?.skill)} / ${fmtSkill(p.test?.skill)}，t ${ui.fmtNum(p.all.t, 1)}`))
    : '—');
  const passCls = (ok) => (ok === true ? 'is-good' : ok === false ? 'is-bad' : '');
  const sens = (rec.sensitivity || []).map(s => ({ ...s, label: s.vars.length === Object.keys(rec.macro || {}).length
    ? `全部 ${s.vars.length} 个` : s.vars.map(k => MACRO_NAME[k] || k).join('、') }));
  root.append(h('div', { style: { height: '16px' } }), card({ title: '校准情景概率：只用相似起点法有预测能力的宏观变量',
    sub: `以模型后验为先验做熵池化：找离模型后验最近（KL 最小）的一组情景概率，使情景加权的第一年宏观变化等于相似起点法的预测。${skill.rule ? `规则：${skill.rule}。` : ''}` },
  h('div', { class: 'grid cols-2 text' },
    h('div', {},
      subHead(ui, `相似起点法的宏观预测能力（${skill.n_origins ?? '—'} 个起点；CRPS 技能 = 1 − 相似起点 CRPS / 基准 CRPS，正 = 更好；`
        + `小字是 ${cut} 年及以前 / ${Number(cut) + 1} 年起和配对 t 值；绿色 = 通过，红色 = 未通过）`),
      ui.table([
        { key: 'name', label: '宏观变量' },
        { value: (r) => r, label: '对无条件抽样', fmt: (r) => skillCell(r), cls: (r) => passCls(r?.beats?.uncond), wrap: true },
        { value: (r) => r.naive, label: '对历史均值基准', fmt: (p) => skillCell(p), cls: (p, r) => passCls(r?.beats?.naive), wrap: true },
        { key: 'skilled', label: '用于校准', fmt: (v) => ui.chip(v ? '是' : '否', v ? 'good' : '') },
      ], sv, { stack: T }),
      h('p', { class: 'muted', style: { fontSize: '12px', margin: '6px 0 0' } },
        '历史均值基准 = 训练期 12 个月变化的均值和标准差构成的正态分布（约等于“不变”）。两种引擎方法都直接抽历史月份的通胀而不重新锚定，'
        + 'CPI 同比的预测因此一起偏高；只胜过无条件抽样不说明预测本身可信，所以两个基准都要胜过。'),
      calOk ? [subHead(ui, '校准的视图（第一年 12 个月的平均变化）'),
        ui.table([
          { key: 'name', label: '变量' },
          { key: 'target_used', label: '相似起点法（目标）', fmt: (v, r) => fmtMacro(v, r.var === 'oil' ? 'log' : 'pp') },
          { key: 'prior_mean', label: '情景加权·模型后验', fmt: (v, r) => fmtMacro(v, r.var === 'oil' ? 'log' : 'pp') },
          { key: 'posterior_mean', label: '校准后', fmt: (v, r) => fmtMacro(v, r.var === 'oil' ? 'log' : 'pp') },
        ], cal.views || [], { stack: T })]
        : ui.notice(`未校准：${cal.reason || '没有可用于校准的宏观变量'}。${cal.verdicts ? `逐个看：${cal.verdicts}。` : ''}`, 'info')),
    h('div', {},
      calOk ? [subHead(ui, `概率怎么变（${(cal.shifts || []).length} 个情景，按变化大小排序）`),
        ui.table([
          { key: 'name', label: '情景', wrap: true, minWidth: 150 },
          { key: 'prior', label: '模型后验', fmt: (v) => pct(v, 1) },
          { key: 'calibrated', label: '校准后', fmt: (v) => pct(v, 1) },
          { key: 'shift', label: '变化', fmt: (v) => ppText(v) },
        ], cal.shifts || [], { stack: T })] : null,
      sens.length ? [subHead(ui, '对照：分别对齐每个宏观变量会怎样（没有预测能力的只作对照，不作为可选的概率）'),
        ui.table([
          { key: 'label', label: '对齐的变量' },
          { key: 'skilled', label: '有预测能力', fmt: (v) => (v ? '是' : '否') },
          { key: 'kl', label: 'KL', fmt: (v) => ui.fmtNum(v, 3) },
          { key: 'ens', label: '有效情景数', fmt: (v) => ui.fmtNum(v, 1) },
          { value: (r) => r.us_equity?.p50, label: '美股中位数', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
          { value: (r) => r.us_equity?.prob_loss, label: 'P(亏损)', fmt: (v) => pct(v, 0) },
        ], sens, { stack: T })] : null)),
  h('p', { class: 'muted', style: { fontSize: '12px', margin: '10px 0 0' } },
    calOk ? '「组合评估」和「提取与破产概率」的情景概率里可以选「按一年期预测校准」；默认仍是模型后验。'
      : '「组合评估」和「提取与破产概率」的情景概率里，「按一年期预测校准」这一项因此不可选（灰色），默认的模型后验不受影响。',
    rec.paths_reproduced === false ? ' 注意：重跑情景得到的路径与情景结果里存的路径不一致，请重新运行 python -m scenariolab.build --only scenarios forecast。' : '')));

  return () => {
    const cols = (rec.assets || []).filter(a => a !== 'cash');
    const cm = rec.contrib_mean || {}, cl = rec.contrib_prob_loss || {};
    const values = [...scs.map(s => cols.map(a => cm[a]?.[s.id] ?? null)), cols.map(a => rec.summary[a]?.gap_mean ?? null)];
    const rowLabels = [...scs.map(s => `${s.name}（${pct(s.prob, 0)}）`), '合计 = 均值差距'];
    charts.heatmap(heatBox, {
      rows: rowLabels, cols, values, rowLabels, colLabels: cols.map(assetName), colors: charts.klineDiverging(),
      height: Math.max(380, 26 * rowLabels.length + 110), labelFmt: (v) => signedText(v * 100, 1), rowLabelWidth: 210,
      cellStyle: (r) => (r === scs.length ? { borderColor: '#1d2433', borderWidth: 1 } : null),
      tip: (r, c, v) => {
        const a = cols[c];
        if (r === scs.length) {
          const s = rec.summary[a] || {};
          return `<b>${ui.esc(assetName(a))}</b><br/>均值差距 <b>${ppText(s.gap_mean)}</b>（情景加权 ${pct(s.scenarios?.mean)} − 相似起点 ${pct(s.analog?.mean)}）`
            + `<br/>P(亏损) 差距 ${ppText(s.gap_prob_loss, 0)}`;
        }
        const s = scs[r];
        return `<b>${ui.esc(s.name)}</b> × ${ui.esc(assetName(a))}<br/>均值差距贡献 <b>${ppText(v)}</b>`
          + `<br/>P(亏损) 差距贡献 ${ppText(cl[a]?.[s.id])}<br/>该情景第一年中位数 ${pct(s.year1?.[a]?.p50)}，均值 ${pct(s.year1?.[a]?.mean)}`
          + `<br/>概率 ${pct(s.prob, 1)}${calOk ? `（校准后 ${pct(s.prob_calibrated, 1)}）` : ''}`;
      },
    });
  };
}

// The live forecast ledger (forecast.json `ledger`): what each build forecast and, once its 12 months are
// in the data, how it turned out.
function renderLedger({ ui, fc, pct, assetName }) {
  const { h, card } = ui;
  const led = fc.ledger;
  if (!led) {
    return card({ title: '实时样本外账本' },
      ui.notice('这份 forecast.json 还没有预测账本：运行 python -m scenariolab.build --only forecast 后出现。', 'info'));
  }
  const names = led.methods || METHOD_CN;
  let asset = 'us_equity';
  const box = h('div');
  const stamp = (v) => (v ? String(v).replace('T', ' ').slice(0, 16) : '—');
  // each entry keeps the transmission it was recorded with; an entry from another model than today's (meta.json)
  // is not today's forecast of that origin, and the page says so
  const nowModel = ui.META.raw?.model?.describe || '';
  const otherModel = (led.entries || []).filter(e => e?.model && nowModel && e.model !== nowModel);
  const draw = () => {
    const rows = (led.entries || []).slice().reverse();          // newest first
    box.replaceChildren(rows.length ? ui.table([
      { key: 'origin', label: '起点' },
      { key: 'created', label: '记录于（UTC）', fmt: stamp },
      { key: 'matures', label: '到期（数据到）' },
      { key: 'model', label: '记录时的传导模型', wrap: true, minWidth: 110, fmt: (v) => v || '—' },
      { key: 'status', label: '状态', fmt: (v) => ui.chip(v === 'matured' ? '已评分' : '待到期', v === 'matured' ? 'good' : 'warn') },
      { value: (r) => r.methods?.analog?.[asset]?.p50, label: '相似起点 中位数', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
      { value: (r) => r.methods?.analog?.[asset], label: '相似起点 90% 区间', fmt: (b) => (b ? `${pct(b.p5)} ~ ${pct(b.p95)}` : '—') },
      { value: (r) => r.methods?.scenarios?.[asset]?.p50, label: '情景加权 中位数', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
      { value: (r) => r.methods?.scenarios_calibrated?.[asset]?.p50, label: '校准后 中位数', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
      { value: (r) => r.realized?.[asset], label: '实际', fmt: (v, r) => (r.status === 'matured' ? pct(v) : '待到期'), cls: (v) => ui.signCls(v) },
      { value: (r) => r.scores?.analog?.[asset], label: '落在 90% 区间', fmt: (s) => (s ? ui.chip(s.covered ? '是' : '否', s.covered ? 'good' : 'bad') : '—') },
      { value: (r) => r.scores?.analog?.[asset]?.pit, label: 'PIT', fmt: (v) => ui.fmtNum(v, 2) },
      { value: (r) => r.scores?.analog?.[asset]?.crps, label: 'CRPS', fmt: (v) => ui.fmtNum(v, 3) },
    ], rows, { stack: WIDE }) : ui.notice('账本里还没有记录。', 'info'));
  };
  const sel = ui.select(fc.assets.map(a => ({ value: a, label: assetName(a) })), asset, (v) => { asset = v; draw(); });
  draw();
  const sc = led.scores;
  const scoreRows = sc ? ['analog', 'uncond', 'scenarios', 'scenarios_calibrated', 'naive'].filter(m => sc[m])
    .map(m => ({ m, ...sc[m].risk_avg, n: sc[m].n_entries })) : [];
  return card({ title: '实时样本外账本：每次构建存档的一年期区间', actions: sel,
    sub: '下面的成绩单是事后回放（只用当时的数据，但模型和代码是今天的）；这里是每次构建当时真正做出的预测，原样存档、到期后评分，没有任何前视' },
  h('p', { class: 'lead', style: { marginBottom: '10px' } }, led.text || ''),
  !led.matured && led.first_maturity
    ? ui.notice(`还没有到期的记录：第一条在数据更新到 ${led.first_maturity} 之后到期，届时下一次构建会自动给出覆盖率、PIT 和 CRPS。`, 'info') : null,
  led.skipped ? ui.notice(`这次构建没有记账：${led.skipped}。`, 'info') : null,
  otherModel.length ? ui.notice(`${otherModel.map(e => e.origin).join('、')} 的记录用的是当时的传导模型（${[...new Set(otherModel.map(e => e.model))].join('、')}），`
    + `和这次构建的一年期预测（${nowModel}）不是同一个模型：同一起点的两组数字不同，账本按记录时的原样保留、到期后评分。`, 'info') : null,
  h('div', { style: { height: '10px' } }), box,
  scoreRows.length ? [subHead(ui, `已到期记录的成绩（风险资产平均；${led.matured} 条）`), ui.table([
    { key: 'm', label: '方法', fmt: (v) => names[v] || METHOD_CN[v] || v },
    { key: 'n', label: '记录数' },
    { key: 'coverage90', label: '90% 覆盖率', fmt: (v) => pct(v, 0) },
    { key: 'crps', label: 'CRPS', fmt: (v) => ui.fmtNum(v, 3) },
    { key: 'pit_mad', label: 'PIT 偏离', fmt: (v) => ui.fmtNum(v, 2) },
    { key: 'mae_p50', label: '中位数误差', fmt: (v) => pct(v) },
  ], scoreRows, { stack: T })] : null,
  (led.errors || []).length ? ui.notice(`有 ${led.errors.length} 个账本文件读不了：${led.errors.join('；')}`, 'error') : null,
  h('p', { class: 'muted', style: { fontSize: '12px', margin: '10px 0 0' } },
    '怎么记录：每次构建（完整构建或只重建一年期预测这一步）把当时的一年期预测原样存进账本目录，'
    + '每个起点一个文件，只追加、不覆盖；同一份数据重复构建不会重复记录；只按时间顺序追加（不补记更早的起点），'
    + '起点之后 3 个月以上才做的预测也不记（那时要预测的月份已经部分知道了）。数据更新到起点之后 12 个月时，下一次构建用新数据给它打分'
    + '（覆盖率、PIT、CRPS，口径同下面的滚动成绩）。'));
}

// Signed fixed decimals, rounded first: never "-0.00", "+" only on values that round above zero.
function signedText(x, d) {
  const t = Number(x).toFixed(d);
  return Number(t) === 0 ? (0).toFixed(d) : `${Number(t) > 0 ? '+' : ''}${t}`;
}

function fmtMacro(v, kind) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  if (kind === 'log') return `${signedText(Math.expm1(v) * 100, 0)}%`;
  return signedText(v, 2);
}

function fmtSkill(s) {
  if (typeof s !== 'number' || !Number.isFinite(s)) return '—';
  return `${signedText(s * 100, 1)}%`;
}

// grouped box plots: assets on the x axis, one series per method
function renderLiveBoxes(charts, el, fc, methods, assetName) {
  const chart = charts.mount(el);
  const live = fc.live?.assets || {};
  const assets = fc.assets.filter(a => a !== 'cash');
  // asset labels: flat while they fit their slot, 30° on mid-size slots, upright on a phone
  const names = assets.map(assetName);
  const slot = ((el.clientWidth || 600) - 76) / Math.max(assets.length, 1);
  const longest = Math.max(10, ...names.map(t => [...String(t)].reduce((w, ch) => w + (/[\x00-\xff]/.test(ch) ? 6.6 : 11), 0)));
  const rotate = longest <= slot - 6 ? 0 : slot >= 34 ? 30 : 90;
  const bottom = rotate === 90 ? Math.min(110, Math.round(longest + 20)) : rotate ? 70 : 36;
  chart.setOption({
    legend: { top: 0, data: methods.map(m => METHOD_CN[m]) },
    tooltip: {
      trigger: 'item',
      formatter: (p) => {
        const d = Array.isArray(p.value) ? p.value : [];
        const o = d.length === 6 ? 1 : 0;                 // ECharts may prepend the category index
        const v = (i) => (d[i + o] === undefined ? '—' : `${(d[i + o] * 100).toFixed(1)}%`);
        return `<b>${assetName(assets[p.dataIndex])}</b> · ${p.seriesName}<br/>95%: ${v(4)}<br/>75%: ${v(3)}<br/><b>中位数: ${v(2)}</b><br/>25%: ${v(1)}<br/>5%: ${v(0)}`;
      },
    },
    grid: { left: 56, right: 20, top: 36, bottom },
    xAxis: { type: 'category', data: names, axisLabel: { interval: 0, rotate, fontSize: 11, color: '#7a8396' },
      axisTick: { show: false }, axisLine: { lineStyle: { color: '#e3e6ec' } } },
    yAxis: { type: 'value', axisLabel: { formatter: (v) => `${Math.round(v * 100)}%`, color: '#7a8396', fontSize: 11 },
      splitLine: { lineStyle: { color: '#eef0f4' } } },
    series: methods.map(m => ({
      name: METHOD_CN[m], type: 'boxplot', boxWidth: [6, 18],
      itemStyle: { color: `${METHOD_COLOR[m]}22`, borderColor: METHOD_COLOR[m] },
      data: assets.map(a => { const b = live[a]?.[m]; return b ? [b.p5, b.p25, b.p50, b.p75, b.p95] : null; }),
      markLine: m === methods[0] ? { silent: true, symbol: 'none', label: { show: false },
        lineStyle: { color: '#adb5bd', type: 'dashed' }, data: [{ yAxis: 0 }] } : undefined,
    })),
  });
}

// interval candles (body p25-p75, wicks p5-p95), median line, realized dots, shock-year shading
function renderTrackChart(charts, el, fc, asset, method) {
  const chart = charts.mount(el);
  const rows = (fc.origins || []).filter(o => o.assets?.[asset]?.[method] && o.assets[asset].realized !== null);
  const labels = rows.map(yearLabel);
  const f = rows.map(o => o.assets[asset][method]);
  const real = rows.map(o => o.assets[asset].realized);
  const has25 = f.every(b => b.p25 !== undefined);
  const color = METHOD_COLOR[method];
  chart.setOption({
    legend: { top: 0, data: ['相似起点法区间', '相似起点法中位数', '实际'] },
    tooltip: {
      trigger: 'axis',
      formatter: (ps) => {
        const i = ps[0].dataIndex, o = rows[i], b = f[i];
        const p = (x) => (x === undefined ? '—' : `${(x * 100).toFixed(1)}%`);
        const out = [`<b>${o.date} 起的 12 个月</b>`, `实际: <b>${p(real[i])}</b>`, `90% 区间: ${p(b.p5)} ~ ${p(b.p95)}`, `中位数: ${p(b.p50)}`];
        if (o.shocks?.length) out.push(`外生冲击: ${o.shocks.join('、')}`);
        return out.join('<br/>');
      },
    },
    grid: { left: 56, right: 20, top: 36, bottom: 60 },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#7a8396', fontSize: 11 }, axisTick: { show: false },
      axisLine: { lineStyle: { color: '#e3e6ec' } } },
    yAxis: [
      { type: 'value', scale: true, axisLabel: { formatter: (v) => `${Math.round(v * 100)}%`, color: '#7a8396', fontSize: 11 },
        splitLine: { lineStyle: { color: '#eef0f4' } } },
      { type: 'value', min: 0, max: 1, show: false },
    ],
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }],
    series: [
      { name: '外生冲击', type: 'bar', yAxisIndex: 1, barWidth: '96%', silent: true, z: 0,
        itemStyle: { color: 'rgba(240, 140, 0, 0.13)' }, data: rows.map(o => (o.shocks?.length ? 1 : 0)), tooltip: { show: false } },
      { name: '相似起点法区间', type: 'candlestick',
        data: f.map(b => (has25 ? [b.p25, b.p75, b.p5, b.p95] : [b.p50, b.p50, b.p5, b.p95])),
        itemStyle: { color: `${color}55`, color0: `${color}55`, borderColor: color, borderColor0: color } },
      { name: '相似起点法中位数', type: 'line', data: f.map(b => b.p50), symbol: 'none', lineStyle: { width: 1.2, color, type: 'dashed' } },
      { name: '实际', type: 'scatter', symbolSize: 8, z: 5,
        data: real.map((y, i) => ({ value: y, itemStyle: { color: y < f[i].p5 || y > f[i].p95 ? '#d9363e' : '#1d2433' } })) },
    ],
  });
}

// pooled PIT histogram of the risk assets (10 bins) for the two engine methods
function renderPit(charts, el, fc) {
  const risk = fc.risk_assets || [];
  const bins = (m) => {
    const counts = new Array(10).fill(0);
    let n = 0;
    for (const o of fc.origins || []) {
      for (const a of risk) {
        const p = o.assets?.[a]?.[m]?.pit;
        if (p === undefined || p === null) continue;
        counts[Math.min(9, Math.floor(p * 10))] += 1;
        n += 1;
      }
    }
    return counts.map(c => (n ? c / n : 0));
  };
  charts.bars(el, {
    x: Array.from({ length: 10 }, (_, i) => `${i * 10}–${i * 10 + 10}%`),
    series: [{ name: METHOD_CN.analog, data: bins('analog'), color: METHOD_COLOR.analog },
      { name: METHOD_CN.uncond, data: bins('uncond'), color: METHOD_COLOR.uncond }],
    fmt: 'pct', markLine: [{ yAxis: 0.1 }],
  });
}
