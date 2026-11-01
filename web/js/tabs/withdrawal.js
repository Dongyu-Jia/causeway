// Withdrawals and ruin: the same scenario paths, a retiree taking rate × initial out at every year
// end under three strategies; ruin probability, safe withdrawal rate, survival and terminal real
// wealth, per scenario and probability-weighted, over a 10–40 year horizon (default 30). The scenario
// paths are 15 years long: from year 16 on there is no scenario theme and every path goes on with an
// unconditional engine path of its own (portfolio/service.py, continuation_paths.npz), so the figures at the
// chosen horizon are shown next to their 15-year (scenario-themed) counterparts (`scenario_horizon`).
// Inputs (_weights.js) are shared with 组合评估.
// Deep links: #withdrawal?income, #withdrawal?rate=3.5&horizon=10&nominal&strategy=div_then_winners&scenario=stagflation
import {
  shared, weightsEditor, probEditor, customPicker, statusPill, liveRequest, debounce, errorNotice, parseParam,
  parseWeightsParam, scenarioRows, rangeChart, labelWidth, rowsHeight, shortName, fmtProb, weightsLabel, isNum, esc,
  niceAxis, plainNote, FALLBACK_PRESETS, injectStyle,
} from './_weights.js';

const INK = '#1d2433', INK2 = '#4a5468', INK3 = '#7a8396', LINE = '#e3e6ec';
const STRATS = [
  { id: 'pro_rata', name: '按比例提取', short: '按比例提取', axis: '按比例提取',
    color: '#3b5bdb', dash: 'solid', symbol: 'circle',
    def: '分红和利息留在原资产里再投资；每年年末按各持仓市值的比例卖出，凑足当年的提取额。' },
  { id: 'div_then_pro_rata', name: '先用分红再按比例提取', short: '先分红·再按比例', axis: '先用分红\n再按比例提取',
    color: '#f08c00', dash: 'dashed', symbol: 'diamond',
    def: '年内收到的分红和利息先进现金池（不计息）；年末先用现金池支付，多出的按持仓比例再投资，不够的部分按持仓比例卖出。' },
  { id: 'div_then_winners', name: '先用分红、再按当年收益率从高到低卖出', short: '先分红·再卖高收益', axis: '先用分红、再按当年\n收益率从高到低卖出',
    color: '#7048e8', dash: 'dotted', symbol: 'triangle',
    def: '和上一种一样先用现金池；不够的部分从当年总收益率最高的资产开始卖，卖光了再卖下一个，相当于逢高减持。' },
];
// Heatmap colour bands around the 5% ruin threshold that defines the safe withdrawal rate.
const RUIN_PIECES = [
  { lte: 0.005, label: '≤0.5%', color: '#e6f4ea' },
  { gt: 0.005, lte: 0.02, label: '0.5–2%', color: '#fff3bf' },
  { gt: 0.02, lte: 0.05, label: '2–5%', color: '#ffd8a8' },
  { gt: 0.05, lte: 0.15, label: '5–15%', color: '#ffa8a8' },
  { gt: 0.15, label: '>15%', color: '#e03131' },
];
// Past the scenario years ruin is rarely below 2%, and above 15% it still differs a lot between scenarios.
const RUIN_PIECES_LONG = [
  { lte: 0.02, label: '≤2%', color: '#e6f4ea' },
  { gt: 0.02, lte: 0.05, label: '2–5%', color: '#ffd8a8' },
  { gt: 0.05, lte: 0.15, label: '5–15%', color: '#ffa8a8' },
  { gt: 0.15, lte: 0.3, label: '15–30%', color: '#ff6b6b' },
  { gt: 0.3, lte: 0.5, label: '30–50%', color: '#e03131' },
  { gt: 0.5, label: '>50%', color: '#a51d24' },
];
const RATE_CAP = 0.2;            // portfolio/withdrawal.py RATE_BOUNDS: the safe-rate search stops here
// portfolio/service.py HORIZON_BOUNDS and DEFAULT_HORIZON: the horizons the API accepts, and this page's default
const HORIZON_MIN = 10, HORIZON_MAX = 40, HORIZON_DEFAULT = 30;
const CONT_SHADE = 'rgba(122, 131, 150, 0.09)';    // the continuation years on the time charts

// inputs and view state kept while the page is open
const params = { initial: 1e6, rate: 0.04, indexed: true, horizon: HORIZON_DEFAULT };
let scope = 'weighted';          // 'weighted' or a scenario id
let strat = 'pro_rata';

const fmtRate = (v) => (!isNum(v) ? '—' : v >= RATE_CAP - 1e-9 ? '≥20%' : `${(v * 100).toFixed(2)}%`);
// the user's own rate, as entered: 0.04 -> "4%", 0.0425 -> "4.25%"
const rateTxt = (v) => (isNum(v) ? `${+(v * 100).toFixed(2)}%` : '—');
// compact money for chart axes and fan tooltips: 1234567 -> "123.5 万"
const moneyShort = (v) => (!isNum(v) ? '—' : Math.abs(v) >= 1e8 ? `${+(v / 1e8).toFixed(2)} 亿`
  : Math.abs(v) >= 1e4 ? `${+(v / 1e4).toFixed(1)} 万` : `${Math.round(v)}`);
// a judgement (red = above the 5% safety line), not the sign of a number: independent of the K-line colours
const ruinCls = (v) => (!isNum(v) ? '' : v > 0.05 ? 'is-bad' : v > 0.01 ? 'withdrawal-warn' : 'is-good');
// Top of a plot under a plain legend of `names` (12 px text, a marker iconW wide): one 22 px row per
// legend line, so a legend that wraps on a narrow card does not run into the plot.
function legendTop(el, names, iconW) {
  const px = (t) => [...String(t)].reduce((w, ch) => w + (/[\x00-\xff]/.test(ch) ? 7.2 : 12), 0);
  const avail = Math.max(120, (el.clientWidth || 600) - 20);
  let rows = 1, used = 0;
  for (const n of names) {
    const item = iconW + 5 + px(n) + 10;
    if (used && used + item > avail) { rows += 1; used = 0; }
    used += item;
  }
  return 16 + rows * 22;
}

// Styles of this page's own blocks (the shared ones come from _weights.js injectStyle).
function injectHorizonStyle(h) {
  if (document.getElementById('style-withdrawal-horizon')) return;
  document.head.append(h('style', { id: 'style-withdrawal-horizon' }, `
.withdrawal-cont { border-left: 3px solid var(--accent); background: var(--accent-soft); border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 13px; line-height: 1.6; color: var(--ink-2); margin: 0 0 12px; }
.withdrawal-cont.plain { border-left-color: var(--line); background: var(--panel-2); }
.withdrawal-cont-label { font-weight: 650; color: var(--ink); }
.withdrawal-cont p { margin: 4px 0 0; max-width: 1080px; }
.withdrawal-themed b { font-variant-numeric: tabular-nums; }
.withdrawal-themed b.is-good { color: var(--good); } .withdrawal-themed b.is-bad { color: var(--bad); }
.withdrawal-themed b.withdrawal-warn { color: var(--warn); }
.withdrawal-compare-card { margin-top: 16px; }
.withdrawal-compare td:first-child { min-width: 150px; }
/* the horizon-labelled headers wrap, so the columns are as wide as their numbers */
.withdrawal-compare table.tbl th, .withdrawal-table table.tbl th, .withdrawal-extremes table.tbl th { white-space: normal; vertical-align: bottom; line-height: 1.35; min-width: 56px; }
.withdrawal-pair, .withdrawal-nw { white-space: nowrap; font-variant-numeric: tabular-nums; }
.withdrawal-pair .is-good { color: var(--good); } .withdrawal-pair .is-bad { color: var(--bad); } .withdrawal-pair .withdrawal-warn { color: var(--warn); }
.withdrawal-arrow { color: var(--ink-3); }
/* wide detail tables scroll inside their card on a tablet: the scenario / strategy column stays put */
@container (min-width: 481px) {
  .withdrawal-table table.tbl td:first-child, .withdrawal-table table.tbl th:first-child,
  .withdrawal-extremes table.tbl td:first-child, .withdrawal-extremes table.tbl th:first-child {
    position: sticky; left: 0; z-index: 1; background: var(--panel); box-shadow: 1px 0 0 var(--line); }
  .withdrawal-table table.tbl th:first-child, .withdrawal-extremes table.tbl th:first-child { z-index: 2; background: var(--panel-2); }
}
`));
}

export async function render(root, { api, charts, ui, meta, param }) {
  injectStyle();
  const { h, card } = ui;
  injectHorizonStyle(h);
  const pct = (v, d = 1) => ui.fmtPct(v, d);
  // all amounts in 万 (亿 above 1e8) so ranges never mix units; a ruined path's wealth is exactly 0
  const money = (v) => (!isNum(v) ? '—' : v === 0 ? '0' : Math.abs(v) >= 1e8 ? `${(v / 1e8).toFixed(2)} 亿` : `${(v / 1e4).toFixed(1)} 万`);
  // years with a scenario theme (the path length): from meta until the first result says so
  let themedYears = Math.max(1, Math.round((meta?.horizon_months || 180) / 12));

  // ---- deep-link parameters
  const q = parseParam(param);
  const tokens = [...q.tokens];
  const linkNotes = [];
  const take = (pred) => { const i = tokens.findIndex(pred); return i < 0 ? null : tokens.splice(i, 1)[0]; };
  // rate=0.035 or rate=3.5: anything above the 20% cap as a decimal is read as percent
  const rate = Number(q.kv.rate);
  if (q.kv.rate !== undefined && Number.isFinite(rate) && rate >= 0) params.rate = Math.min(RATE_CAP, rate > RATE_CAP ? rate / 100 : rate);
  const init = Number(q.kv.initial);
  if (Number.isFinite(init) && init > 0) params.initial = init;
  const yrsRaw = q.kv.horizon ?? q.kv.years;
  const yrs = yrsRaw === undefined || String(yrsRaw).trim() === '' ? NaN : Number(yrsRaw);
  if (Number.isFinite(yrs)) {
    const y = Math.min(Math.max(HORIZON_MIN, Math.round(yrs)), HORIZON_MAX);
    if (y !== yrs) linkNotes.push(`链接里的期限 ${yrs} 年已改为 ${y} 年（可选 ${HORIZON_MIN}–${HORIZON_MAX} 年的整数）`);
    params.horizon = y;
  } else if (yrsRaw !== undefined) {
    linkNotes.push(`链接里的期限「${String(yrsRaw).slice(0, 24)}」不是数字，已使用默认的 ${HORIZON_DEFAULT} 年（可选 ${HORIZON_MIN}–${HORIZON_MAX} 年的整数）`);
  }
  params.horizon = Math.min(Math.max(HORIZON_MIN, params.horizon), HORIZON_MAX);
  if (take(t => t === 'nominal') || ['0', 'false'].includes(q.kv.indexed)) params.indexed = false;
  if (take(t => t === 'indexed') || ['1', 'true'].includes(q.kv.indexed)) params.indexed = true;
  const st = q.kv.strategy || take(t => STRATS.some(s => s.id === t));
  if (STRATS.some(s => s.id === st)) strat = st;
  let wantScope = q.kv.scenario || null;

  // `root` (#view) is reused by whichever tab opens next, so "is this page still showing" is
  // checked on a node of our own: the lead paragraph, later the results container.
  const lead = h('p', { class: 'lead' },
    '用同一批情景路径模拟退休提取：期初把初始资产按组合权重买入，之后每年年末按“提取率 × 初始资产”取一次钱，',
    '比较三种提取方式在每个情景里会不会破产（钱不够付当年的提取额）。破产概率按情景概率加权汇总，',
    '安全提取率是破产概率不超过 5% 的最高提取率；所有数字都由引擎在模拟路径上算出。',
    `期限默认 ${HORIZON_DEFAULT} 年（4% 法则说的就是 30 年）；情景路径只有 ${themedYears} 年，再往后没有情景主题，按历史月份无条件续抽，`,
    `所以每个数字旁边都给出前 ${themedYears} 年（情景主题内）的对照。`);
  root.append(lead);

  const defaultsP = api.get('portfolio_default').catch(err => ({ error: err }));
  const scenP = api.get('scenarios').catch(err => ({ error: err }));
  const presetsOf = (d) => (d && !d.error && Array.isArray(d.presets) && d.presets.length ? d.presets : FALLBACK_PRESETS);
  // a preset named in the link needs the preset list before the editor starts
  const defaults = tokens.length || q.kv.preset ? await defaultsP : null;
  if (!lead.isConnected) return;       // the user moved on while the preset list loaded
  const presetId = q.kv.preset || take(t => presetsOf(defaults).some(p => p.id === t));
  let initial = presetsOf(defaults).find(p => p.id === presetId)?.weights || null;
  if (q.kv.w && Object.keys(parseWeightsParam(q.kv.w)).length) initial = parseWeightsParam(q.kv.w);

  // ---- inputs
  let scen = null, probInfo = {};
  const pill = statusPill();
  const editor = weightsEditor({ assets: meta?.assets, presets: presetsOf(defaults), initial, cls: 'span-2',
    onChange: () => refresh() });
  const probs = probEditor({ onChange: () => refresh() });
  const picker = customPicker({ onChange: () => { probs.setRows(scenarioRows(scen, picker.items()), probInfo); refresh(); } });
  const schedule = debounce(() => refresh(), 400);

  const initialIn = h('input', { type: 'number', min: 0, step: 10, value: String(Math.round(params.initial / 100) / 100), 'aria-label': '初始资产（万）' });
  const rateRange = h('input', { type: 'range', min: 0, max: 10, step: 0.1, value: String(Math.min(10, params.rate * 100)), 'aria-label': '年提取率' });
  const rateIn = h('input', { type: 'number', min: 0, max: 20, step: 0.1, value: String(+(params.rate * 100).toFixed(2)), 'aria-label': '年提取率（%）' });
  const horizonRange = h('input', { type: 'range', min: HORIZON_MIN, max: HORIZON_MAX, step: 1, value: String(params.horizon), 'aria-label': '期限（年）' });
  const initialHint = h('span', { class: 'muted' });
  const rateHint = h('div', { class: 'muted' });
  const horizonVal = h('b', {});
  const horizonHint = h('div', { class: 'muted' });
  function paintHints() {
    initialHint.textContent = `= ${Math.round(params.initial).toLocaleString('zh-CN')}`;
    rateHint.textContent = `每年提取 ${money(params.rate * params.initial)}`
      + (params.indexed ? '（按期初购买力计，名义金额随 CPI 逐年上调）' : '（名义金额固定，购买力会被通胀侵蚀）');
    horizonVal.textContent = `${params.horizon} 年`;
    const S = themedYears;
    horizonHint.textContent = params.horizon > S
      ? `前 ${S} 年按各情景路径；第 ${S + 1}–${params.horizon} 年没有情景主题，按历史月份无条件续抽（${HORIZON_MIN}–${HORIZON_MAX} 年可选）`
      : `全部 ${params.horizon} 年都在情景路径之内（情景路径共 ${S} 年；${HORIZON_MIN}–${HORIZON_MAX} 年可选）`;
  }
  initialIn.addEventListener('input', () => {
    const v = Number(initialIn.value);
    if (!(Number.isFinite(v) && v > 0)) { initialHint.textContent = '请输入大于 0 的金额'; return; }
    params.initial = v * 1e4;
    paintHints();
    schedule();
  });
  rateRange.addEventListener('input', () => {
    params.rate = Number(rateRange.value) / 100;
    rateIn.value = String(Number(rateRange.value));
    paintHints();
    schedule();
  });
  rateIn.addEventListener('input', () => {
    const v = Number(rateIn.value);
    if (rateIn.value === '' || !Number.isFinite(v) || v < 0 || v > RATE_CAP * 100) return;
    params.rate = v / 100;
    rateRange.value = String(Math.min(10, v));
    paintHints();
    schedule();
  });
  // on leaving the field: clamp an out-of-range rate to 0–20% (the safe-rate search range) instead of
  // silently discarding it, then show the rate actually used
  rateIn.addEventListener('change', () => {
    const v = Number(rateIn.value);
    if (rateIn.value !== '' && Number.isFinite(v) && (v < 0 || v > RATE_CAP * 100)) {
      params.rate = v < 0 ? 0 : RATE_CAP;
      rateRange.value = String(Math.min(10, params.rate * 100));
      paintHints();
      schedule.flush();
    }
    rateIn.value = String(+(params.rate * 100).toFixed(2));
  });
  horizonRange.addEventListener('input', () => { params.horizon = Number(horizonRange.value); paintHints(); schedule(); });
  const probState = h('span', { class: 'muted' });
  const paintProbState = () => { probState.textContent = probs.label(); };   // posterior, calibrated or hand-set
  paintHints();
  paintProbState();

  const field = (title, ...body) => h('div', { class: 'withdrawal-field' }, h('div', { class: 't' }, title), ...body);
  root.append(h('div', { class: 'grid cols-3' }, editor.el,
    card({ title: '提取参数', sub: '破产概率只取决于提取率，初始金额只决定财富数字的大小', cls: 'weights-side' },
      h('div', { class: 'withdrawal-form' },
        field('初始资产（万）', h('div', { class: 'withdrawal-inline' }, initialIn, initialHint)),
        field('年提取率（占初始资产）', h('div', { class: 'withdrawal-inline' }, rateRange, rateIn, h('span', { class: 'muted' }, '%')), rateHint),
        field('通胀挂钩', ui.seg([{ value: 'real', label: '随通胀上调' }, { value: 'nominal', label: '名义固定' }],
          params.indexed ? 'real' : 'nominal', (v) => { params.indexed = v === 'real'; paintHints(); schedule.flush(); })),
        field('期限', h('div', { class: 'withdrawal-inline' }, horizonRange, horizonVal), horizonHint),
        ui.notice('每年年末提取一次（第一次在满 1 年时）；随通胀上调时，提取额按该路径自己的 CPI 逐年调高，始终保持期初的购买力。', 'info'),
        field('纳入已保存的自定义情景', picker.el),
        field('情景概率', h('div', { class: 'row' }, probState,
          h('button', { class: 'btn', onclick: () => probs.el.scrollIntoView({ behavior: 'smooth', block: 'start' }) }, '调整情景概率')))))));

  const errBox = h('div', { class: 'weights-err' });
  const results = h('div', { class: 'weights-results' }, ui.loading('正在模拟三种提取方式…'));
  const alive = () => results.isConnected;
  const notesList = h('ul', { class: 'list' }, h('li', { class: 'muted' }, '计算完成后，这里列出本次结果的计算口径。'));
  const notesStale = h('p', { class: 'muted', style: { display: 'none', margin: 0 } }, '组合权重合计为 0：没有结果，也就没有计算口径。');
  root.append(errBox, results,
    h('div', { class: 'section-title' }, '调整情景概率'), probs.el,
    h('div', { class: 'section-title' }, '计算口径'), card({}, notesList, notesStale),
    pill.el);

  // ---- results view (built on the first result)
  let view = null, shown = null, idx = null;
  function buildView() {
    const v = {
      head: h('div', { class: 'section-title' }), summary: h('div', { class: 'portfolio-summary' }),
      cont: h('div', {}), kpis: h('div', { class: 'grid cols-4' }), compare: h('div', { class: 'withdrawal-compare' }),
      heat: ui.chartBox('tall'), curve: ui.chartBox('tall'), safe: ui.chartBox('tall'), surv: ui.chartBox('tall'),
      fan: ui.chartBox('tall'), term: ui.chartBox('tall'), extremes: h('div', { class: 'withdrawal-extremes' }),
      table: h('div', { class: 'withdrawal-table' }),
      scopeSel: h('select', { 'aria-label': '图表范围', onchange: (e) => { scope = e.target.value; drawScoped(); } }),
      stratSeg: ui.seg(STRATS.map(s => ({ value: s.id, label: s.short })), strat, (x) => { strat = x; drawStrategy(); }),
    };
    const titled = (title, sub, box, cls) => {
      const c = card({ title, sub, cls }, box);
      return [c, c.querySelector('.card-title'), c.querySelector('.card-sub')];
    };
    let heatCard, curveCard, safeCard, survCard, fanCard, termCard, extremesCard, tableCard;
    [v.compareCard, v.compareTitle, v.compareSub] = titled('前 15 年与整个期限', '（计算后填入）', v.compare, 'withdrawal-compare-card');
    [heatCard, v.heatTitle] = titled('破产概率：情景 × 提取方式',
      '每格是该情景全部路径里破产的比例，按 5% 安全线分档着色：哪一行变红，说明那种宏观环境（高通胀、熊市、利率上升）最容易让提取计划失败；提示框里有前 15 年的数字。', v.heat);
    [curveCard, v.curveTitle] = titled('破产概率 vs 提取率',
      '横轴是每年提取初始资产的比例、纵轴是破产概率（圆点 = 与 5% 破产线相交的安全提取率，竖虚线 = 当前提取率），曲线越靠右，说明这个范围内的资产回报越能扛住通胀和提取；细线是只算前 15 年（情景主题内）的同一条曲线。', v.curve);
    [safeCard, v.safeTitle] = titled('安全提取率（破产概率 ≤ 5%）',
      '每个情景、每种提取方式下仍能把破产概率控制在 5% 以内的最高年提取率：点落在竖虚线左边，说明在那种宏观环境（通胀推高提取额、熊市侵蚀本金）下当前提取率已不安全；提示框里有前 15 年的安全提取率。', v.safe);
    [survCard, v.survTitle] = titled('存活曲线',
      '每年年末还没有破产的概率：曲线向下弯的那一年，就是熊市叠加通胀让提取额吃掉本金的时候（纵轴已按数据范围放大；灰底是没有情景主题的续抽年份）。', v.surv);
    [fanCard, v.fanTitle] = titled('实际财富扇形图',
      '每年年末提取后剩下的财富（按期初购买力，深色带 25%–75%、浅色带 5%–95% 的路径，灰线是期初本金，灰底是续抽年份）：通胀抬高提取额、熊市压低回报时扇形整体下移，跌到 0 即破产。', v.fan);
    [termCard, v.termTitle] = titled('期末实际财富：各情景',
      '期限结束时剩下的财富（期初购买力）：点 = 中位数、色带 = 5%–95% 的路径，在灰线右边表示提取之后购买力还增加了，行与行之间的差距就是宏观情景的影响。', v.term);
    [extremesCard, v.extremesTitle] = titled('每种提取方式的最坏与最好情景',
      '按破产概率排序（相同时比较期末实际财富）：最坏情景就是这个提取计划最怕的宏观环境。', v.extremes);
    [tableCard, v.tableTitle] = titled('各情景明细', '概率是计算所用的情景概率；耗尽年份 = 破产路径用光资产的年份中位数（没有路径破产时为 —）。', v.table);
    results.replaceChildren(
      v.head, v.summary, v.cont, v.kpis, v.compareCard,
      h('div', { class: 'withdrawal-toolbar' },
        h('span', {}, '图表范围'), v.scopeSel, h('span', {}, '财富图与明细表的策略'), v.stratSeg),
      h('div', { class: 'section-title' }, '破产概率从哪里来'),
      h('div', { class: 'weights-pair' }, heatCard, curveCard),
      h('div', { class: 'section-title' }, '安全提取率与存活曲线'),
      h('div', { class: 'weights-pair' }, safeCard, survCard),
      h('div', { class: 'section-title' }, '期末还剩多少（期初购买力）'),
      h('div', { class: 'weights-pair' }, fanCard, termCard),
      h('div', { class: 'section-title' }, '最坏 / 最好情景'), extremesCard,
      h('div', { class: 'section-title' }, '各情景明细'), tableCard);
    return v;
  }

  // ---- the scenario-themed part (the first `part.years` years) next to the chosen horizon
  const partOf = (res) => res.scenario_horizon || null;
  const signedPct = (v) => (isNum(v) ? ui.fmtPct(v, 2, true) : '—');
  const scenCell = (id) => (id ? h('span', { title: idx.name[id] || id }, idx.label[id] || id) : '—');
  function contNote(res) {
    const c = res.continuation, S = res.scenario_years ?? themedYears;
    if (!c) {
      return h('div', { class: 'withdrawal-cont plain' }, `期限 ${res.horizon_years} 年全部在情景路径之内（情景路径共 ${S} 年），`
        + '所有数字都只来自各情景自己的宏观路径。');
    }
    const at = c.at_start || {};
    const levels = [isNum(c.inflation_p50) ? `续抽期通胀中位数约每年 ${pct(c.inflation_p50)}` : null,
      isNum(at.y10) ? `10 年期收益率从约 ${at.y10.toFixed(1)}% 起步` : null].filter(Boolean).join('，');
    return h('div', { class: 'withdrawal-cont' },
      h('div', { class: 'withdrawal-cont-label' }, c.label || `第 ${S + 1} 年起没有情景主题：按历史月份无条件续抽`),
      h('p', {}, `前 ${S} 年是各情景自己的宏观路径；第 ${c.first_year ?? S + 1}–${res.horizon_years} 年接上引擎的无条件模拟：`
        + `同样从今天的状态出发、不加任何视图、按 ${c.block_months ?? 6} 个月一段抽历史月份，去掉前 ${S} 年，把之后的年份按路径序号接在情景路径后面。`
        + (c.shared ? `续抽路径共 ${c.n_paths ?? '—'} 条，部分情景路径按序号循环共用；`
          : `每条情景路径各接一条独立的续抽路径（共 ${c.n_paths ?? '—'} 条，抽法相同）；`)
        + `第 ${S} 年之后各情景的差别只来自前 ${S} 年留下的财富和持仓。`),
      h('p', {}, `为什么这样接：情景只描述未来 ${S} 年，再往后没有可依据的主题，就退回引擎对历史月份的无条件抽样，而不是把某个情景的叙事硬延长。`
        + `长期回报假设去掉了只为 ${S} 年设定的估值回归项（美股价格漂移每年 ${signedPct(c.market_adjustment)}，前 ${S} 年为 ${signedPct(c.scenario_market_adjustment)}），`
        + `既不再向历史估值中位数回归，也不加回 1973 年以来的估值抬升；CPI、分红和利息照常延续`
        + (levels ? `（${levels}，都不承接各情景第 ${S} 年末的水平）` : '') + '。'));
  }

  function drawCompare(res) {
    const part = partOf(res);
    view.compareCard.style.display = part ? '' : 'none';
    if (!part) { view.compare.replaceChildren(); return; }
    const Y = res.horizon_years, S = part.years;
    view.compareTitle.textContent = `前 ${S} 年（情景主题内）与 ${Y} 年对比`;
    view.compareSub.textContent = `前 ${S} 年的数字只来自各情景路径，就是在第 ${S} 年末截断的同一批提取路径（与只算 ${S} 年完全相同）；`
      + `${Y} 年的数字包含第 ${S + 1} 年起的无条件续抽。两者的差距就是“再多撑 ${Y - S} 年”的代价。`;
    const rows = STRATS.filter(s => res.strategies?.[s.id])
      .map(s => ({ s, a: aggOf(res, s.id), p: part.strategies?.[s.id]?.aggregate || {} }));
    // each cell reads "first S years -> whole horizon", so three columns hold the comparison
    const pair = (from, to, fmt, cls = () => '') => h('span', { class: 'withdrawal-pair' },
      h('span', { class: cls(from) || null }, fmt(from)), h('span', { class: 'withdrawal-arrow' }, ' → '), h('b', { class: cls(to) || null }, fmt(to)));
    const head = (what, span) => h('span', {}, what, ' ', h('br'), span);     // the space keeps the stacked label readable
    view.compare.replaceChildren(ui.table([
      { label: '提取方式', value: (r) => r.s.name, wrap: true },
      { label: head('破产概率', `前 ${S} 年 → ${Y} 年`), value: (r) => r,
        fmt: (r) => pair(r.p.ruin_prob_weighted, r.a.ruin_prob_weighted, fmtProb, ruinCls) },
      { label: head('安全提取率（破产 ≤ 5%）', `${S} 年 → ${Y} 年`), value: (r) => r,
        fmt: (r) => pair(part.safe_rate?.[r.s.id]?.weighted, res.safe_rate?.[r.s.id]?.weighted, fmtRate) },
      { label: head('实际财富中位数', `第 ${S} 年末 → 第 ${Y} 年末`), value: (r) => r,
        fmt: (r) => pair(r.p.terminal_real_p50_weighted, r.a.terminal_real_p50_weighted, money) },
    ], rows, { stack: true }));
  }
  // the continuation years on a chart whose x axis is res.dates (category axis)
  const contArea = (res, x) => {
    const part = partOf(res);
    return part && x.length > part.years ? [{ from: x[part.years], to: x[x.length - 1], label: '无条件续抽', color: CONT_SHADE }] : undefined;
  };

  // ---- lookups on one result
  function index(res) {
    const first = STRATS.find(s => res.strategies?.[s.id]);
    const list = first ? res.strategies[first.id].per_scenario || [] : [];
    const by = Object.fromEntries(STRATS.map(s => [s.id,
      Object.fromEntries((res.strategies?.[s.id]?.per_scenario || []).map(e => [e.id, e]))]));
    const count = {};
    list.forEach(e => { const k = shortName(e.name); count[k] = (count[k] || 0) + 1; });
    const label = Object.fromEntries(list.map(e => [e.id, count[shortName(e.name)] > 1 ? e.name : shortName(e.name)]));
    const name = Object.fromEntries(list.map(e => [e.id, e.name]));
    // chart rows: the weighted mixture first, then every scenario
    const rows = [{ id: null, label: '加权汇总', full: '加权汇总（按情景概率）' },
      ...list.map(e => ({ id: e.id, label: label[e.id], full: e.name }))];
    return { list, by, label, name, rows };
  }
  const aggOf = (res, sid) => res.strategies?.[sid]?.aggregate || {};
  const scopeName = () => (scope === 'weighted' ? '加权汇总' : idx.label[scope] || scope);

  function kpis(res) {
    const Y = res.horizon_years, part = partOf(res);
    const tiles = STRATS.filter(s => res.strategies?.[s.id]).map(s => {
      const a = aggOf(res, s.id);
      const p = part?.strategies?.[s.id]?.aggregate;
      const safeY = res.safe_rate?.[s.id]?.weighted;
      const nw = (text) => h('span', { class: 'withdrawal-nw' }, text);     // a figure and its label never split
      return ui.kpi(s.name, [fmtProb(a.ruin_prob_weighted), h('span', { class: 'withdrawal-kpi-unit' }, `${Y} 年破产概率`)], [
        p ? h('div', { class: 'withdrawal-themed' }, `前 ${part.years} 年（情景主题内）破产概率 `,
          h('b', { class: ruinCls(p.ruin_prob_weighted) }, fmtProb(p.ruin_prob_weighted))) : null,
        part ? h('div', {}, '安全提取率（破产 ≤ 5%）：', nw(`${Y} 年 ${fmtRate(safeY)}`), ' · ',
          nw(`${part.years} 年 ${fmtRate(part.safe_rate?.[s.id]?.weighted)}`))
          : h('div', {}, `安全提取率 ${fmtRate(safeY)}（破产 ≤ 5%）`),
        h('div', {}, `第 ${Y} 年末实际财富中位数 `, nw(money(a.terminal_real_p50_weighted)),
          p ? ['（', nw(`第 ${part.years} 年末 ${money(p.terminal_real_p50_weighted)}`), '）'] : null),
        h('div', {}, `5%–95%：${money(a.terminal_real_p5_weighted)} ~ ${money(a.terminal_real_p95_weighted)}`),
        h('div', {}, `最坏情景：${idx.label[a.worst_id] || '—'} ${fmtProb(a.ruin_prob_worst)}`),
        h('div', { class: 'withdrawal-def' }, s.def),
      ], ruinCls(a.ruin_prob_weighted));
    });
    const total = `${res.horizon_years} 年合计提取 ${money(res.annual_withdrawal * res.horizon_years)}`
      + (res.inflation_indexed ? '（期初购买力）' : '（名义）');
    tiles.push(ui.kpi('每年提取额', money(res.annual_withdrawal), [
      h('div', {}, `提取率 ${rateTxt(res.rate)} × 初始资产 ${money(res.initial)}`),
      h('div', {}, res.inflation_indexed ? '随通胀上调：每年按该路径的 CPI 调高，购买力不变' : '名义金额固定：购买力逐年被通胀侵蚀'),
      h('div', {}, total),
      h('div', { class: 'withdrawal-def' }, '破产 = 年末持仓加现金池不够付当年的提取额；之后财富记为 0。'),
    ]));
    return tiles;
  }

  function drawHeat(res) {
    const part = partOf(res);
    const themedOf = (r, s) => (r.id === null ? part?.strategies?.[s.id]?.aggregate?.ruin_prob_weighted
      : part?.strategies?.[s.id]?.per_scenario?.[r.id]?.ruin_prob);
    const rows = idx.rows.map(r => ({ ...r, values: STRATS.map(s => (r.id === null
      ? aggOf(res, s.id).ruin_prob_weighted : idx.by[s.id][r.id]?.ruin_prob)), themed: STRATS.map(s => themedOf(r, s)) }));
    view.heatTitle.textContent = `${res.horizon_years} 年破产概率：情景 × 提取方式`;
    view.heat.style.height = `${rowsHeight(rows.length)}px`;
    // column heads: the two-line full names while a column has room for them, else the short
    // names, broken inside their own column (on a phone the long ones ran into each other)
    const colW = ((view.heat.clientWidth || 600) - labelWidth(view.heat) - 24) / STRATS.length;
    const roomy = colW >= 124;
    const chart = charts.mount(view.heat);
    const pieces = part ? RUIN_PIECES_LONG : RUIN_PIECES;
    const dark = part ? 0.3 : 0.15;                 // above this the cell colour is dark enough for white text
    const narrow = (view.heat.clientWidth || 600) < 520;
    const data = [];
    rows.forEach((r, i) => STRATS.forEach((s, j) => {
      const v = r.values[j];
      data.push({ value: [j, i, isNum(v) ? v : '-'], label: isNum(v) && v > dark ? { color: '#fff' } : undefined });
    }));
    chart.setOption({
      tooltip: { confine: true, formatter: (p) => {
        const r = rows[p.value[1]], j = p.value[0];
        return `<b>${esc(r.full)}</b><br/>${esc(STRATS[j].name)}<br/>${res.horizon_years} 年破产概率 <b>${fmtProb(isNum(p.value[2]) ? p.value[2] : null)}</b>`
          + (part ? `<br/>前 ${part.years} 年（情景主题内）${fmtProb(r.themed[j])}` : '');
      } },
      grid: { left: 8, right: 8, top: 4, bottom: 34, containLabel: true },
      xAxis: { type: 'category', position: 'top', data: STRATS.map(s => (roomy ? s.axis : s.short)), axisLine: { show: false }, axisTick: { show: false },
        axisLabel: { interval: 0, color: INK2, fontSize: 11, lineHeight: 14, ...(roomy ? {} : { width: Math.max(40, colW - 6), overflow: 'break' }) },
        splitArea: { show: false } },
      yAxis: { type: 'category', inverse: true, data: rows.map(r => r.label), axisLine: { show: false }, axisTick: { show: false },
        axisLabel: { color: INK2, fontSize: 11, interval: 0, width: labelWidth(view.heat), overflow: 'truncate',
          formatter: (val, i) => (i === 0 ? `{b|${val}}` : val), rich: { b: { fontWeight: 600, color: INK } } } },
      visualMap: { type: 'piecewise', dimension: 2, orient: 'horizontal', left: 'center', bottom: 0,
        itemWidth: narrow ? 10 : 14, itemHeight: 10, itemGap: narrow ? 4 : 10, textGap: narrow ? 3 : 5,
        textStyle: { color: INK3, fontSize: narrow ? 10 : 11 }, pieces },
      series: [{ type: 'heatmap', data, itemStyle: { borderColor: '#fff', borderWidth: 2 },
        label: { show: true, fontSize: 11, color: INK, formatter: (p) => fmtProb(isNum(p.value[2]) ? p.value[2] : null) },
        emphasis: { itemStyle: { borderColor: INK, borderWidth: 1 } } }],
    });
  }

  function drawCurve(res) {
    const rates = res.ruin_curve?.rates || [];
    const xMax = rates.length ? rates[rates.length - 1] : 0.1;
    const list = STRATS.filter(s => res.ruin_curve?.[s.id]);
    const part = partOf(res);
    const curveOf = (s) => (scope === 'weighted' ? res.ruin_curve[s.id].weighted : res.ruin_curve[s.id].per_scenario?.[scope]) || [];
    const safeOf = (s) => (scope === 'weighted' ? res.safe_rate?.[s.id]?.weighted : res.safe_rate?.[s.id]?.per_scenario?.[scope]);
    const themedCurve = (s) => (scope === 'weighted' ? part?.ruin_curve?.[s.id]?.weighted : part?.ruin_curve?.[s.id]?.per_scenario?.[scope]) || [];
    const themedSafe = (s) => (scope === 'weighted' ? part?.safe_rate?.[s.id]?.weighted : part?.safe_rate?.[s.id]?.per_scenario?.[scope]);
    const limit = isNum(res.safe_ruin_prob) ? res.safe_ruin_prob : 0.05;
    const Y = res.horizon_years;
    // one legend entry switches the three thin scenario-themed curves together
    const themedName = part ? `前 ${part.years} 年（细线）` : null;
    view.curveTitle.textContent = `${Y} 年破产概率 vs 提取率：${scopeName()}`;
    const series = list.map(s => {
      const c = curveOf(s);
      return { name: s.short, id: `h_${s.id}`, type: 'line', showSymbol: false, data: rates.map((r, i) => [r, isNum(c[i]) ? c[i] : null]),
        lineStyle: { color: s.color, width: 2, type: s.dash }, itemStyle: { color: s.color } };
    });
    if (part) {
      list.forEach(s => {
        const c = themedCurve(s);
        series.push({ name: themedName, id: `s_${s.id}`, type: 'line', showSymbol: false, z: 1, data: rates.map((r, i) => [r, isNum(c[i]) ? c[i] : null]),
          lineStyle: { color: s.color, width: 1.2, type: s.dash, opacity: 0.55 }, itemStyle: { color: '#868e96' } });
      });
    }
    const pts = list.map(s => ({ s, v: safeOf(s) })).filter(p => isNum(p.v) && p.v <= xMax);
    series.push({ name: '安全提取率', type: 'scatter', symbolSize: 10, z: 5,
      data: pts.map(p => ({ value: [p.v, limit], itemStyle: { color: p.s.color, borderColor: '#fff', borderWidth: 1.5 } })) });
    if (series.length > 1) {      // precision: by default ECharts rounds the line's position to 2 decimals (5.5% -> 6%)
      series[0].markLine = { silent: true, symbol: 'none', animation: false, precision: 10, lineStyle: { color: INK3, type: 'dashed' },
        label: { color: INK3, fontSize: 10 },
        data: [{ yAxis: limit, label: { formatter: `破产概率 ${pct(limit, 0)}`, position: 'insideStartTop' } },
          ...(res.rate <= xMax ? [{ xAxis: res.rate, label: { formatter: `当前 ${rateTxt(res.rate)}`, position: 'end' } }] : [])] };
    }
    const names = [...list.map(s => s.short), ...(part ? [themedName] : [])];
    const curveTop = legendTop(view.curve, names, 18);
    const nameOf = (p) => {
      const id = String(p.seriesId || '');
      const s = STRATS.find(t => id.endsWith(`_${t.id}`));
      return id.startsWith('s_') ? `前 ${part.years} 年·${s?.short || ''}` : `${Y} 年·${s?.short || p.seriesName}`;
    };
    charts.mount(view.curve).setOption({
      legend: { top: 0, data: names, textStyle: { color: INK, fontSize: 12 }, itemWidth: 18, itemHeight: 8 },
      tooltip: { trigger: 'axis', confine: true, formatter: (ps) => {
        const line = ps.filter(p => p.seriesType === 'line');
        if (!line.length) return '';
        return [`年提取率 <b>${(line[0].value[0] * 100).toFixed(2)}%</b>`,
          ...line.map(p => `${p.marker}${esc(nameOf(p))}：破产概率 ${fmtProb(p.value[1])}`),
          ...list.map(s => `${esc(s.short)}的安全提取率：${Y} 年 ${fmtRate(safeOf(s))}`
            + (part ? ` · ${part.years} 年 ${fmtRate(themedSafe(s))}` : ''))].join('<br/>');
      } },
      grid: { left: 48, right: 24, top: curveTop + 6, bottom: 46 },
      xAxis: { type: 'value', min: 0, max: xMax, name: '年提取率（占初始资产）', nameLocation: 'middle', nameGap: 28,
        nameTextStyle: { color: INK3, fontSize: 11 }, axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false },
        splitLine: { lineStyle: { color: '#eef0f4' } }, axisLabel: { color: INK3, fontSize: 11, formatter: (x) => `${(x * 100).toFixed(0)}%` } },
      yAxis: { type: 'value', min: 0, max: 1, axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false },
        splitLine: { lineStyle: { color: '#eef0f4' } }, axisLabel: { color: INK3, fontSize: 11, formatter: (y) => `${(y * 100).toFixed(0)}%` } },
      series,
    });
  }

  function drawSafe(res) {
    const rows = idx.rows;
    const list = STRATS.filter(s => res.safe_rate?.[s.id]);
    const part = partOf(res);
    const themedSafe = (s, r) => (r?.id === null ? part?.safe_rate?.[s?.id]?.weighted : part?.safe_rate?.[s?.id]?.per_scenario?.[r?.id]);
    view.safeTitle.textContent = `安全提取率（破产概率 ≤ 5%，${res.horizon_years} 年）`;
    const vals = [];
    const series = list.map(s => ({ name: s.short, type: 'scatter', symbol: s.symbol, symbolSize: 11,
      itemStyle: { color: s.color, opacity: 0.9 },
      data: rows.map((r, i) => {
        const v = r.id === null ? res.safe_rate[s.id].weighted : res.safe_rate[s.id].per_scenario?.[r.id];
        if (isNum(v)) vals.push(v);
        return [isNum(v) ? v : '-', i];
      }) }));
    // round bounds on a fixed step, so the first and last labels sit on the same grid as the rest;
    // a narrow plot takes every k-th step (labels at least ~44 px apart)
    const ax = niceAxis(Math.min(...vals, res.rate), Math.max(...vals, res.rate));
    const plotPx = Math.max(80, (view.safe.clientWidth || 600) - labelWidth(view.safe) - 52);
    const steps = Math.max(1, Math.round((ax.max - ax.min) / ax.interval));
    const k = Math.max(1, Math.ceil(44 / (plotPx / steps)));
    if (k > 1) { ax.max = +(ax.min + Math.ceil(steps / k) * k * ax.interval).toFixed(6); ax.interval = +(ax.interval * k).toFixed(6); }
    const dec = ax.interval < 0.005 ? 2 : 1;
    const safeTop = legendTop(view.safe, list.map(s => s.short), 12);
    if (series.length) {          // inverted category axis: 'start' puts the label at the top
      series[0].markLine = { silent: true, symbol: 'none', animation: false, precision: 10, lineStyle: { color: INK3, type: 'dashed' },
        label: { color: INK3, fontSize: 10, formatter: `当前 ${rateTxt(res.rate)}`, position: 'start' }, data: [{ xAxis: res.rate }] };
    }
    view.safe.style.height = `${rowsHeight(rows.length)}px`;
    charts.mount(view.safe).setOption({
      legend: { top: 0, data: list.map(s => s.short), textStyle: { color: INK, fontSize: 12 }, itemWidth: 12, itemHeight: 10 },
      tooltip: { trigger: 'item', confine: true, formatter: (p) => `<b>${esc(rows[p.value[1]]?.full)}</b><br/>${p.marker}${esc(list[p.seriesIndex]?.name)}`
        + `<br/>${res.horizon_years} 年安全提取率 <b>${fmtRate(isNum(p.value[0]) ? p.value[0] : null)}</b>`
        + (part ? `<br/>前 ${part.years} 年（情景主题内）${fmtRate(themedSafe(list[p.seriesIndex], rows[p.value[1]]))}` : '') },
      grid: { left: 8, right: 30, top: safeTop, bottom: 8, containLabel: true },
      xAxis: { type: 'value', min: ax.min, max: ax.max, interval: ax.interval,
        axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false }, splitLine: { lineStyle: { color: '#eef0f4' } },
        axisLabel: { color: INK3, fontSize: 11, hideOverlap: true, formatter: (x) => `${(x * 100).toFixed(dec)}%` } },
      yAxis: { type: 'category', inverse: true, data: rows.map(r => r.label), axisTick: { show: false },
        axisLine: { lineStyle: { color: LINE } }, splitLine: { show: true, lineStyle: { color: '#f1f3f5' } },
        axisLabel: { color: INK2, fontSize: 11, interval: 0, width: labelWidth(view.safe), overflow: 'truncate',
          formatter: (val, i) => (i === 0 ? `{b|${val}}` : val), rich: { b: { fontWeight: 600, color: INK } } } },
      series,
    });
  }

  function drawSurv(res) {
    const x = res.dates || res.years || [];
    const list = STRATS.filter(s => res.strategies?.[s.id]);
    const survOf = (s) => (scope === 'weighted' ? aggOf(res, s.id).survival_weighted : idx.by[s.id][scope]?.survival) || [];
    const all = list.flatMap(s => survOf(s)).filter(isNum);
    const lo = all.length ? Math.min(...all) : 1;
    // zoom the axis to the data, in round steps (0.5% / 2% / 10%) so small differences stay readable
    const pad = Math.max(0.002, (1 - lo) * 0.25);
    const step = 1 - lo + pad < 0.02 ? 0.005 : 1 - lo + pad < 0.1 ? 0.02 : 0.1;
    const yMin = Math.max(0, Math.floor((lo - pad) / step + 1e-9) * step);
    view.survTitle.textContent = `存活曲线（${res.horizon_years} 年）：${scopeName()}`;
    const chart = charts.line(view.surv, { x, yFmt: 'pct', yMax: 1, yMin: +yMin.toFixed(4),
      tooltipFmt: (v) => `${(v * 100).toFixed(2)}%`, markAreas: contArea(res, x),
      series: list.map(s => ({ name: s.short, data: survOf(s), color: s.color, width: 2 })) });
    chart.setOption({ series: list.map(s => ({ lineStyle: { type: s.dash } })) });     // same dashes as the ruin curve
    if (all.length && lo >= 1) {
      chart.setOption({ graphic: [{ type: 'text', left: 'center', top: 'middle', silent: true,
        style: { text: `${scopeName()}：${res.horizon_years} 年内没有任何路径破产`, fill: INK3, fontSize: 13 } }] });
    }
  }

  function drawFan(res) {
    const x = res.dates || res.years || [];
    const s = STRATS.find(t => t.id === strat) || STRATS[0];
    const bands = scope === 'weighted' ? aggOf(res, s.id).wealth_fan_real_weighted : idx.by[s.id][scope]?.wealth_fan_real;
    view.fanTitle.textContent = `实际财富扇形图（${res.horizon_years} 年）：${s.short} · ${scopeName()}`;
    if (!bands) { charts.mount(view.fan).clear(); return; }
    charts.fan(view.fan, { x, bands, fmt: moneyShort, name: '中位数', color: s.color, markAreas: contArea(res, x),
      extra: [{ name: '期初本金', data: x.map(() => res.initial), dashed: true, color: '#adb5bd' }] });
  }

  function drawTerm(res) {
    const s = STRATS.find(t => t.id === strat) || STRATS[0];
    const a = aggOf(res, s.id);
    const Y = res.horizon_years, part = partOf(res), ps = part?.strategies?.[s.id];
    view.termTitle.textContent = `第 ${Y} 年末实际财富：${s.short}`;
    const tip = (full, e, ruin, themed) => `<b>${esc(full)}</b><br/>${esc(s.name)}<br/>第 ${Y} 年末实际财富中位数 <b>${money(e.p50)}</b>`
      + `<br/>5%–95%：${money(e.p5)} ~ ${money(e.p95)}<br/>${Y} 年破产概率 ${fmtProb(ruin)}`
      + (themed ? `<br/>第 ${part.years} 年末中位数 ${money(themed.terminal_real_p50 ?? themed.terminal_real_p50_weighted)}`
        + `，前 ${part.years} 年破产概率 ${fmtProb(themed.ruin_prob ?? themed.ruin_prob_weighted)}` : '');
    rangeChart(view.term, {
      height: rowsHeight(idx.rows.length),
      fmt: moneyShort,
      rows: idx.rows.map(r => {
        const one = idx.by[s.id][r.id] || {};
        const e = r.id === null
          ? { p5: a.terminal_real_p5_weighted, p50: a.terminal_real_p50_weighted, p95: a.terminal_real_p95_weighted, ruin: a.ruin_prob_weighted }
          : { p5: one.terminal_real_p5, p50: one.terminal_real_p50, p95: one.terminal_real_p95, ruin: one.ruin_prob };
        const themed = !ps ? null : r.id === null ? ps.aggregate : ps.per_scenario?.[r.id];
        return { label: r.label, lo: e.p5, mid: e.p50, hi: e.p95, color: r.id === null ? INK : s.color, size: r.id === null ? 12 : 9,
          tip: tip(r.full, e, e.ruin, themed) };
      }),
      marks: [{ x: res.initial, label: '期初本金', color: '#868e96' }],
    });
  }

  function drawExtremes(res) {
    const rows = STRATS.filter(s => res.strategies?.[s.id]).map(s => ({ s, a: aggOf(res, s.id) }));
    view.extremesTitle.textContent = `每种提取方式的最坏与最好情景（${res.horizon_years} 年）`;
    view.extremes.replaceChildren(ui.table([
      { label: '提取方式', value: (r) => r.s.name, wrap: true },
      { label: '加权破产概率', value: (r) => r.a.ruin_prob_weighted, fmt: (v) => fmtProb(v), cls: (v) => ruinCls(v) },
      { label: '最坏情景', value: (r) => r.a.worst_id, fmt: (v) => scenCell(v), cls: 'is-bad' },
      { label: '最坏：破产概率', value: (r) => r.a.ruin_prob_worst, fmt: (v) => fmtProb(v), cls: (v) => ruinCls(v) },
      { label: '最坏：安全提取率', value: (r) => res.safe_rate?.[r.s.id]?.per_scenario?.[r.a.worst_id], fmt: (v) => fmtRate(v) },
      { label: '最好情景', value: (r) => r.a.best_id, fmt: (v) => scenCell(v), cls: 'is-good' },
      { label: '最好：破产概率', value: (r) => r.a.ruin_prob_best, fmt: (v) => fmtProb(v), cls: (v) => ruinCls(v) },
      { label: '最好：期末实际财富中位数', value: (r) => idx.by[r.s.id][r.a.best_id]?.terminal_real_p50, fmt: (v) => money(v) },
    ], rows, { stack: true }));
  }

  function drawTable(res) {
    const s = STRATS.find(t => t.id === strat) || STRATS[0];
    const a = aggOf(res, s.id);
    const safe = res.safe_rate?.[s.id] || {};
    const Y = res.horizon_years, part = partOf(res), S = part?.years;
    const ps = part?.strategies?.[s.id], safeS = part?.safe_rate?.[s.id] || {};
    view.tableTitle.textContent = `各情景明细：${s.name}（${Y} 年）`;
    const rows = [{ id: null, name: '加权汇总（按情景概率）', prob: 1, ruin_prob: a.ruin_prob_weighted, safe: safe.weighted,
      terminal_real_p5: a.terminal_real_p5_weighted, terminal_real_p50: a.terminal_real_p50_weighted,
      terminal_real_p95: a.terminal_real_p95_weighted, terminal_nominal_p50: a.terminal_nominal_p50_weighted,
      depletion_year_median: a.depletion_year_median_weighted,
      ruin_s: ps?.aggregate?.ruin_prob_weighted, safe_s: safeS.weighted, real_s: ps?.aggregate?.terminal_real_p50_weighted },
    ...idx.list.map(e => ({ ...(idx.by[s.id][e.id] || e), safe: safe.per_scenario?.[e.id],
      ruin_s: ps?.per_scenario?.[e.id]?.ruin_prob, safe_s: safeS.per_scenario?.[e.id], real_s: ps?.per_scenario?.[e.id]?.terminal_real_p50 }))];
    const themed = (col) => (part ? [col] : []);
    const tbl = ui.table([
      { key: 'name', label: '情景', wrap: true, fmt: (v, r) => (r.id === null ? h('b', {}, v) : h('span', {},
        r.source === 'custom' ? v : h('a', { href: `#scenarios?${r.id}`, title: '查看这个情景的宏观设定与传导链' }, v),
        r.id === a.worst_id ? [' ', ui.chip('最坏', 'bad')] : null, r.id === a.best_id ? [' ', ui.chip('最好', 'good')] : null,
        r.source === 'custom' ? [' ', ui.chip('自定义', 'accent')] : null)) },
      { key: 'prob', label: '概率', fmt: (v) => pct(v) },
      { key: 'ruin_prob', label: part ? `破产概率 · ${Y} 年` : '破产概率', fmt: (v) => fmtProb(v), cls: (v) => ruinCls(v) },
      ...themed({ key: 'ruin_s', label: `破产概率 · 前 ${S} 年`, fmt: (v) => fmtProb(v), cls: (v) => ruinCls(v) }),
      { key: 'safe', label: part ? `安全提取率 · ${Y} 年` : '安全提取率', fmt: (v) => fmtRate(v) },
      ...themed({ key: 'safe_s', label: `安全提取率 · ${S} 年`, fmt: (v) => fmtRate(v) }),
      { key: 'terminal_real_p5', label: '期末实际 p5', fmt: (v) => money(v) },
      { key: 'terminal_real_p50', label: '期末实际 p50', fmt: (v) => money(v) },
      { key: 'terminal_real_p95', label: '期末实际 p95', fmt: (v) => money(v) },
      ...themed({ key: 'real_s', label: `第 ${S} 年末实际 p50`, fmt: (v) => money(v) }),
      { key: 'terminal_nominal_p50', label: '期末名义 p50', fmt: (v) => money(v) },
      { key: 'depletion_year_median', label: '耗尽年份', fmt: (v) => (isNum(v) ? `第 ${Math.round(v * 10) / 10} 年` : '—') },
    ], rows, { stack: true });
    tbl.querySelectorAll('tbody tr').forEach((tr, i) => { if (rows[i].id && rows[i].id === scope) tr.classList.add('weights-focus'); });
    view.table.replaceChildren(tbl);
  }

  const drawScoped = () => { if (!shown) return; drawCurve(shown); drawSurv(shown); drawFan(shown); drawTable(shown); };
  const drawStrategy = () => { if (!shown) return; drawFan(shown); drawTerm(shown); drawTable(shown); };

  // adjusted: which scenario probabilities the request that produced `res` carried: 'calibrated' (the set
  // calibrated to the one-year forecast), true (hand-set) or false (none: the model posterior)
  function show(res, { adjusted = false } = {}) {
    if (!view) view = buildView();
    shown = res;
    idx = index(res);
    if (wantScope && idx.name[wantScope]) { scope = wantScope; wantScope = null; }
    if (scope !== 'weighted' && !idx.name[scope]) scope = 'weighted';
    view.scopeSel.replaceChildren(h('option', { value: 'weighted' }, '加权汇总（全部情景）'),
      ...idx.list.map(e => h('option', { value: e.id }, `${e.name}（${pct(e.prob)}）`)));
    view.scopeSel.value = scope;
    if (isNum(res.scenario_years) && res.scenario_years !== themedYears) { themedYears = res.scenario_years; paintHints(); }
    const part = partOf(res);
    const custom = idx.list.filter(e => e.source === 'custom').length;
    view.head.textContent = `${res.horizon_years} 年内三种提取方式的破产概率（按情景概率加权）`;
    view.summary.textContent = [
      weightsLabel(res.weights), `初始 ${money(res.initial)}`, `每年提取 ${rateTxt(res.rate)}（${money(res.annual_withdrawal)}）`,
      res.inflation_indexed ? '随通胀上调' : '名义固定',
      part ? `${res.horizon_years} 年（前 ${part.years} 年情景 + ${res.horizon_years - part.years} 年无条件续抽）` : `${res.horizon_years} 年`,
      `${idx.list.length} 个情景${custom ? `（含 ${custom} 个自定义）` : ''}`,
      adjusted === 'calibrated' ? '按一年期预测校准的概率' : adjusted ? '手动调整的概率' : '模型后验概率',
    ].join(' · ');
    view.cont.replaceChildren(contNote(res));
    view.kpis.replaceChildren(...kpis(res));
    drawCompare(res);
    notesList.replaceChildren(...(res.notes || []).map(n => h('li', { class: 'muted' }, plainNote(n))));
    drawExtremes(res);
    drawTable(res);
    drawHeat(res);
    drawCurve(res);
    drawSafe(res);
    drawSurv(res);
    drawFan(res);
    drawTerm(res);
  }

  // ---- requests
  function body() {
    const weights = editor.get();
    if (!weights) return null;
    const b = { weights, initial: params.initial, rate: params.rate, strategy: 'all',
      inflation_indexed: params.indexed, horizon_years: params.horizon };
    if (shared.include.length) b.include_custom = shared.include.slice();
    const sp = probs.overrides();
    if (sp) b.scenario_probs = sp;
    return b;
  }
  // probability-editor rows from a response, when scenarios.json could not be read
  const idxRows = (res) => index(res).list.map(e => ({ id: e.id, name: e.name, prior: null, likelihood: null,
    post: e.prob, def: e.prob, custom: e.source === 'custom' }));
  // ---- every weight cleared (清零, or each input set to 0), as on 组合评估: the editor sends no change for a
  // portfolio it cannot evaluate, so the page watches the editor itself and hides the previous portfolio's
  // numbers (not shown as if they were this one's) until a valid portfolio is computed, or until the inputs
  // are back to exactly the request whose result is hidden.
  const ZERO = '组合权重合计为 0：请至少给一个资产设置权重。上一个组合的结果已隐藏，设置权重后会自动重算。';
  let stale = false, shownKey = null, pendingKey = null, staleTimer = 0;
  const keyOf = (b) => (b ? JSON.stringify(b) : null);
  function setStale(on) {
    stale = on;
    results.style.display = on ? 'none' : '';
    notesList.style.display = on ? 'none' : '';
    notesStale.style.display = on ? '' : 'none';
  }
  function watchWeights() {
    if (!alive()) return;
    if (!editor.get()) { if (!stale) refresh(); return; }
    if (!stale) return;
    errBox.replaceChildren();
    clearTimeout(staleTimer);
    const key = keyOf(body());
    if (shown && key === shownKey) { setStale(false); pill.done('已恢复'); return; }
    if (key === pendingKey) return;        // the editor's own change already asked for exactly this
    staleTimer = setTimeout(refresh, 450);
  }
  editor.el.addEventListener('input', watchWeights);
  editor.el.addEventListener('click', watchWeights);

  let live = false;
  const go = liveRequest(async ({ b, adjusted }) => ({ res: await api.post('withdrawal/evaluate', b), adjusted, key: keyOf(b) }), {
    start: () => { pill.busy('重新模拟提取中…'); results.classList.add('weights-busy'); },
    done: ({ res, adjusted, key }) => {
      if (!alive()) return;
      pendingKey = null;
      results.classList.remove('weights-busy');
      live = true;
      if (!probs.hasRows()) probs.setRows(idxRows(res), probInfo);
      show(res, { adjusted });
      shownKey = key;
      probs.sync(idx.list);
      if (!editor.get()) { setStale(true); pill.fail('权重合计为 0：未计算'); return; }   // cleared while it ran
      errBox.replaceChildren();
      setStale(false);
      pill.done('已更新');
    },
    error: async (err) => {
      if (!alive()) return;
      pendingKey = null;
      results.classList.remove('weights-busy');
      pill.fail('计算失败：见页面上方提示');
      errBox.replaceChildren(errorNotice(err, () => refresh()));
      if (live) return;                    // keep the last good result on screen
      const d = defaults || await defaultsP;
      if (!alive()) return;
      if (d && !d.error && d.withdrawal) {
        errBox.append(ui.notice(`下面先显示构建时预先计算的默认结果（60/40、100 万、每年 4%、随通胀上调、${d.withdrawal.horizon_years || HORIZON_DEFAULT} 年），`
          + '实时计算恢复后会自动替换。', 'info'));
        show(d.withdrawal);
        shownKey = null;
        setStale(!editor.get());
      } else if (!view) {
        results.replaceChildren();
      }
    },
  });
  function refresh() {
    schedule.cancel();
    if (!alive()) return;                  // a timer from a page the user has left
    clearTimeout(staleTimer);
    paintProbState();
    const b = body();
    if (!b) {
      setStale(true);
      errBox.replaceChildren(ui.notice(ZERO, 'error'));
      pill.fail('权重合计为 0：未计算');
      return;
    }
    if (!(params.initial > 0)) { errBox.replaceChildren(ui.notice('初始资产必须大于 0。', 'error')); return; }
    pendingKey = keyOf(b);
    go({ b, adjusted: !b.scenario_probs ? false : probs.mode() === 'calibrated' ? 'calibrated' : probs.adjusted() });
  }

  // ---- scenario list + saved custom scenarios, then the first evaluation; presets fill in later
  const [s, items] = await Promise.all([scenP, picker.load()]);
  if (!alive()) return;
  scen = s && !s.error ? s : null;
  probInfo = { method: scen?.probability_method, gamma: scen?.probability?.gamma };
  probs.setRows(scenarioRows(scen, items), probInfo);
  if (scen && !wantScope) wantScope = take(t => (scen.scenarios || []).some(x => x.id === t));
  if (tokens.length) {
    root.insertBefore(ui.notice(`无法识别的链接参数：${tokens.join('、')}（可用预设 id、情景 id、提取方式 id 或 nominal）。`, 'info'), errBox);
  }
  if (linkNotes.length) root.insertBefore(ui.notice(`${linkNotes.join('；')}。`, 'info'), errBox);
  refresh();
  const d = defaults || await defaultsP;
  if (alive() && d && !d.error) editor.setPresets(presetsOf(d));
}
