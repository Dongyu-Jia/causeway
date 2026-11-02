// Portfolio evaluation: one set of weights run through every scenario's simulated paths, the
// scenarios mixed by probability (weighted aggregate), next to the worst / best scenario ranked
// in real terms. The weights may hold asset classes and tickers (标的, _tickers.js); the inputs are
// shared with the withdrawal tab. Below the portfolio, 标的推演 shows one ticker on every scenario's
// paths (fans per scenario and mixed, payouts per share, attribution, correlations).
// Deep links: #portfolio?all_weather, #portfolio?preset=income&rebalance=annual&scenario=ai_bust,
// #portfolio?w=us_equity:50,gold:50, #portfolio?w=MAIN:10,QQQI:15,brk.b:5 (any case),
// #portfolio?preset=income_tickers, #portfolio?ticker=TQQQ (opens 标的推演 on TQQQ), #portfolio?TQQQ,
// #portfolio?set=quarterly (the quarterly stack's set; P9, SIM_DESIGN §11.3: a set with step_months = 3 rebalances
// every quarter or year, annualizes volatility by √4 and reports quarter-end drawdowns; the default set follows the
// switch gate T-S)
import {
  shared, probEditor, customPicker, setPicker, statusPill, liveRequest, errorNotice, parseParam,
  scenarioRows, rangeChart, rowsHeight, rankColor, shortName, fmtMult, isNum, esc,
  signOf, plainNote, injectStyle,
} from './_weights.js';
import {
  portfolioEditor, allPresets, loadCatalog, parseLinkWeights, resolveId, isTicker, holdingsCard, projection, weightsText, TK,
  TICKERS_Q, TICKERS_Q_ONLY,
} from './_tickers.js';

const REBALANCE = [
  { value: 'monthly', label: '每月再平衡', help: '每个月末把各资产调回目标权重：纪律最强，组合波动最低。' },
  { value: 'quarterly', label: '每季再平衡', help: '每个季度末调回一次目标权重，季内随涨跌漂移（季度情景集的每一步）。' },
  { value: 'annual', label: '每年再平衡', help: '每满 12 个月调回一次目标权重，年内随涨跌漂移。' },
  { value: 'none', label: '不再平衡', help: '期初买入后一直持有：涨得多的资产权重越来越大，组合会向赢家集中。' },
];
const NO_MONTHLY = '季度情景集每步一个季度，没有月内路径：不能每月再平衡';
const SORTS = [{ value: 'prob', label: '按概率' }, { value: 'real', label: '按实际回报' }, { value: 'sharpe', label: '按夏普' }];

// view state kept while the page is open
let rebalance = 'monthly';
let wealthView = 'real';
let sortBy = 'prob';

// Short chart labels, falling back to the full name where two scenarios would share one.
function labelsOf(per) {
  const count = {};
  per.forEach(e => { const s = shortName(e.name); count[s] = (count[s] || 0) + 1; });
  return Object.fromEntries(per.map(e => [e.id, count[shortName(e.name)] > 1 ? e.name : shortName(e.name)]));
}

export async function render(root, { api, charts, ui, meta, param, signal, rerender }) {
  injectStyle();
  const { h, card } = ui;
  const pct = (v, d = 1) => ui.fmtPct(v, d);
  const q = parseParam(param);
  const tokens = [...q.tokens];
  const take = (pred) => { const i = tokens.findIndex(pred); return i < 0 ? null : tokens.splice(i, 1)[0]; };
  const reb = q.kv.rebalance || take(t => REBALANCE.some(r => r.value === t));
  if (REBALANCE.some(r => r.value === reb)) rebalance = reb;
  const assetIds = (meta?.assets || []).map(a => a.id);
  const linkNotes = [];

  // `root` (#view) is reused by whichever tab opens next, so "is this page still showing" is
  // checked on a node of our own: the lead paragraph, later the results container.
  const lead = h('p', { class: 'lead' },
    '把同一个组合放进每个情景的全部模拟路径里逐月计算，再按情景概率把所有路径混合成“加权汇总”：',
    '它回答“综合所有可能的未来，这个组合大概会怎样”。最坏 / 最好情景按扣除通胀后的实际年化回报排序，',
    '因为高通胀情景的名义收益会显得很好看。所有数字都由引擎在同一批路径上算出，改动权重或概率后自动重算。',
    '组合里可以混合 13 类资产和 16 个标的（QQQ、TQQQ、QQQI、BDC、MLP 等）：标的是作用在同一批路径上的配方，',
    '在每条路径上与资产类别一致；页面下方的“标的推演”逐个查看标的在各情景里的价格、总回报和分配。');
  root.append(lead);

  const defaultsP = api.get('portfolio_default').catch(err => ({ error: err }));
  const scenP = api.get('scenarios').catch(err => ({ error: err }));
  const catP = loadCatalog();
  // a preset named in the link needs the preset list before the editor starts; the ticker rows need the catalog
  const [defaults, cat] = await Promise.all([tokens.length || q.kv.preset ? defaultsP : null, catP]);
  if (!lead.isConnected) return;       // the user moved on while the preset list loaded
  const presets = allPresets(defaults, cat);
  const presetId = q.kv.preset || take(t => presets.some(p => p.id === t));
  let initial = presets.find(p => p.id === presetId)?.weights || null;
  if (q.kv.w && !rerender) {
    const lw = parseLinkWeights(q.kv.w, assetIds.length ? assetIds : undefined);
    if (Object.keys(lw.weights).length) initial = lw.weights;
    if (lw.unknown.length) linkNotes.push(`链接里的权重有无法识别的资产或标的：${lw.unknown.join('、')}`);
  }
  let focus = q.kv.scenario || null;
  const tickerOf = (t) => { const id = resolveId(t, assetIds.length ? assetIds : undefined); return id && isTicker(id) ? id : id === 'gold' ? 'GOLD' : null; };
  let focusTicker = q.kv.ticker ? tickerOf(q.kv.ticker) : null;
  if (rerender) initial = null;        // the K-line toggle redraws the page: keep the reader's weights and ticker
  if (q.kv.ticker && !focusTicker) linkNotes.push(`链接里的标的「${String(q.kv.ticker).slice(0, 20)}」不在标的列表里`);
  if (!focusTicker) { const tk = take(t => !!tickerOf(t)); if (tk) focusTicker = tickerOf(tk); }

  // ---- inputs (built before any data arrives)
  let scen = null, probInfo = {};
  const pill = statusPill();
  // the scenario set (monthly engine / quarterly stack / an agent-tree export): the default follows the switch gate
  const sets = setPicker({ want: q.kv.set || null, onChange: () => { setChanged(); contextChanged(); } });
  const quarterlySet = () => sets.stepMonths() > 1;
  const editor = portfolioEditor({ assets: meta?.assets, presets, initial, cls: 'span-2',
    onChange: () => refresh() });
  editor.setCatalog(cat);
  // scenario probabilities and custom scenarios: the portfolio and 标的推演 both follow them
  const contextChanged = () => { refresh(); proj.refresh(); };
  const probs = probEditor({ onChange: contextChanged });
  const rowsNow = () => scenarioRows(scen, picker.items(), sets.entry());
  const infoNow = () => ({ ...probInfo, basis: sets.current() && sets.current() !== 'monthly' ? 'prior' : 'posterior' });
  const picker = customPicker({ onChange: () => { probs.setRows(rowsNow(), infoNow()); contextChanged(); } });
  if (focusTicker && !rerender) TK.pj.ticker = focusTicker;
  const proj = projection({ signal, focus: !!focusTicker && !rerender,
    getContext: () => ({ include_custom: shared.include, scenario_probs: probs.overrides() }) });

  const rebHelp = h('div', { class: 'portfolio-help' });
  const rebSeg = h('div', { class: 'seg weights-rebal', role: 'group', 'aria-label': '再平衡方式' });
  const paintReb = () => {
    const q3 = quarterlySet();
    rebSeg.replaceChildren(...REBALANCE.map(r => h('button', { type: 'button', class: r.value === rebalance ? 'on' : '',
      disabled: q3 && r.value === 'monthly' ? true : null, title: q3 && r.value === 'monthly' ? NO_MONTHLY : r.help,
      onclick: () => { if (rebalance === r.value) return; rebalance = r.value; paintReb(); refresh(); } }, r.label)));
    rebHelp.textContent = (REBALANCE.find(r => r.value === rebalance)?.help || '') + (q3 ? `（${NO_MONTHLY}。）` : '');
  };
  // a set change: its own scenarios and default probabilities (a quarterly set: hand-set priors), and no monthly
  // rebalancing on a quarterly set
  function setChanged() {
    shared.probs = null;
    shared.probMode = 'posterior';
    probs.setRows(rowsNow(), infoNow());
    if (quarterlySet() && rebalance === 'monthly') rebalance = 'quarterly';
    editor.lockTickers(quarterlySet() ? TICKERS_Q : null);
    paintReb();
    paintHorizon();
  }
  const paintHorizon = () => {
    const e = sets.entry();
    if (e && e.id !== 'monthly') {
      horizonNote.textContent = `期限 ${e.years || 15} 年，每个情景 ${e.n_paths || '—'} 条季度路径（${e.name}${e.trial ? '，试运行' : ''}）；`
        + '和「未来情景」页的月度路径不是同一批。';
    } else if (scen) {
      horizonNote.textContent = `期限 ${Math.round((scen.horizon_months || 180) / 12)} 年（${scen.start || '—'} 起），每个情景 ${scen.n_paths || '—'} 条模拟路径，`
        + '与「未来情景」页是同一批路径。';
    }
  };
  const probState = h('span', { class: 'muted' });
  const paintProbState = () => { probState.textContent = probs.label(); };   // posterior, calibrated or hand-set
  const horizonNote = h('div', { class: 'portfolio-help' });
  paintReb();
  paintProbState();
  root.append(h('div', { class: 'grid cols-3' }, editor.el,
    card({ title: '计算设置', sub: '情景集、再平衡方式、纳入哪些情景、按什么概率混合', cls: 'weights-side' },
      h('div', { class: 'portfolio-label' }, '情景集'), sets.el,
      h('div', { class: 'portfolio-label' }, '再平衡方式'),
      rebSeg,
      rebHelp,
      h('div', { class: 'portfolio-label' }, '纳入已保存的自定义情景'), picker.el,
      h('div', { class: 'portfolio-label' }, '情景概率'),
      h('div', { class: 'row' }, probState,
        h('button', { class: 'btn', onclick: () => probs.el.scrollIntoView({ behavior: 'smooth', block: 'start' }) }, '调整情景概率')),
      horizonNote,
      h('div', { class: 'portfolio-label' }, '标的'),
      h('div', { class: 'portfolio-help' }, '标的按构建时的默认参数计算（估计型标的的 α 用收缩后的版本，提取页可以切换）；',
        '情景概率和自定义情景同样用于下方的“标的推演”。'),
      h('div', { class: 'row', style: { marginTop: '6px' } },
        h('button', { class: 'btn', onclick: () => proj.el.scrollIntoView({ behavior: 'smooth', block: 'start' }) }, '查看标的推演')))));

  const errBox = h('div', { class: 'weights-err' });
  const results = h('div', { class: 'weights-results' }, ui.loading('正在用全部情景的路径计算组合…'));
  const alive = () => results.isConnected;
  const notesList = h('ul', { class: 'list' }, h('li', { class: 'muted' }, '计算完成后，这里列出本次结果的计算口径。'));
  const notesStale = h('p', { class: 'muted', style: { display: 'none', margin: 0 } }, '组合权重合计为 0：没有结果，也就没有计算口径。');
  root.append(errBox, results, proj.el,
    h('div', { class: 'section-title' }, '调整情景概率'), probs.el,
    h('div', { class: 'section-title' }, '计算口径'), card({}, notesList, notesStale),
    pill.el);

  // ---- results view (built on the first result, refilled on every later one)
  let view = null, shown = null, scrolled = false;
  function buildView() {
    const v = {
      summary: h('div', { class: 'portfolio-summary' }), kpis: h('div', { class: 'grid cols-4' }), holdings: h('div', { class: 'tk-holdings-box' }),
      compare: h('div', { class: 'portfolio-compare' }), rank: ui.chartBox('tall'), fan: ui.chartBox('tall'), lines: ui.chartBox('tall'),
      table: h('div', { class: 'portfolio-table' }),
    };
    const fanCard = card({ title: '加权财富扇形图',
      sub: '期初 1 元在所有情景按概率混合后的分布（深色带 25%–75%、浅色带 5%–95% 的路径，灰线是期初 1 元），红绿虚线是最差 / 最好情景的中位数，两者的差距就是宏观情景经通胀和资产回报传导出来的影响。',
      actions: ui.seg([{ value: 'real', label: '实际购买力' }, { value: 'nominal', label: '名义' }], wealthView,
        (x) => { wealthView = x; if (shown) drawWealth(shown); }) }, v.fan);
    v.fanTitle = fanCard.querySelector('.card-title');
    const linesCard = card({ title: '各情景的中位数财富路径',
      sub: '每条线是一个情景的中位数路径，颜色由绿到红 = 实际回报由高到低（与排名图一致），粗黑线是加权中位数，线尾标出最好 / 最差情景、悬停可逐条查看：线与线的距离就是宏观情景对这个组合的影响。' }, v.lines);
    v.linesTitle = linesCard.querySelector('.card-title');
    results.replaceChildren(
      h('div', { class: 'section-title' }, '加权汇总：按情景概率把全部路径混合'), v.summary, v.kpis, v.holdings,
      h('div', { class: 'section-title' }, '最坏 / 最好情景（按实际年化回报排序）'),
      h('div', { class: 'weights-pair' },
        card({ title: '加权平均 vs 等权 vs 最坏 / 最好',
          sub: '加权 = 各情景的路径按概率混合后再取分位数、等权 = 每个情景一样重要（均为中位数，实际亏损 = 期末购买力低于期初的概率），两者差得越多，结论越依赖你对情景概率的判断。' }, v.compare),
        card({ title: '情景排名：扣除通胀后的年化回报',
          sub: '点 = 实际 CAGR 中位数、色带 = 5%–95% 的路径区间、点越大情景概率越高；宏观冲击经通胀、利率和盈利传导到各类资产，排在最下面的就是这个组合最怕的宏观环境。' }, v.rank)),
      h('div', { class: 'section-title' }, '财富路径'),
      h('div', { class: 'weights-pair' }, fanCard, linesCard),
      h('div', { class: 'section-title' }, '各情景明细'),
      card({ title: '每个情景下的组合表现',
        sub: 'CAGR = 年化复合增长率；p5 / p50 / p95 是该情景全部路径的 5% / 50% / 95% 分位；实际 = 按各路径自己的 CPI 扣除通胀；夏普 = 相对现金的超额收益 ÷ 波动；回撤坏尾部 = 只有 5% 的路径比它更深；点击情景名可查看它的宏观设定与传导链。',
        actions: ui.seg(SORTS, sortBy, (x) => { sortBy = x; if (shown) drawTable(shown); }) }, v.table));
    return v;
  }

  function kpis(res) {
    const a = res.aggregate?.weighted || {};
    const yrs = Math.round((res.horizon_months || 180) / 12);
    const q3 = (res.step_months || 1) > 1;
    return [
      ui.kpi('名义年化回报 CAGR（中位数）', pct(a.cagr_p50, 2), `5%–95% 路径：${pct(a.cagr_p5)} ~ ${pct(a.cagr_p95)}`, signOf(a.cagr_p50, 2)),
      ui.kpi('实际年化回报（扣除通胀，中位数）', pct(a.real_cagr_p50, 2), `5%–95% 路径：${pct(a.real_cagr_p5)} ~ ${pct(a.real_cagr_p95)}`, signOf(a.real_cagr_p50, 2)),
      ui.kpi('年化波动率', pct(a.vol), q3 ? '季度收益的标准差 × √4，各路径平均' : '月度收益的标准差 × √12，各路径平均'),
      ui.kpi('夏普比率', ui.fmtNum(a.sharpe, 2), '相对现金的年化超额收益 ÷ 年化波动（中位数）'),
      ui.kpi('索提诺比率', ui.fmtNum(a.sortino, 2), q3 ? '分母只计跑输现金的季度（下行波动）' : '分母只计跑输现金的月份（下行波动）'),
      ui.kpi(q3 ? '最大回撤（中位数，季末口径）' : '最大回撤（中位数）', pct(a.max_dd_p50),
        `坏尾部：5% 的路径深于 ${pct(a.max_dd_p95)}${q3 ? '；季末口径会低估月内最深点（美股约 0.89 倍）' : ''}`, signOf(a.max_dd_p50)),
      ui.kpi(`${yrs} 年后实际亏损概率`, pct(a.prob_loss_real), '期末购买力低于期初的路径占比', a.prob_loss_real > 0.25 ? 'is-bad' : ''),
      ui.kpi('CVaR 5%：最差 5% 路径的期末财富', fmtMult(a.cvar5_terminal),
        `期初 = 1 倍（名义）；扣除通胀后 ${fmtMult(a.cvar5_terminal_real)}`, isNum(a.cvar5_terminal) && a.cvar5_terminal < 1 ? 'is-bad' : ''),
    ];
  }

  function compare(res) {
    const agg = res.aggregate || {};
    const rows = [
      { label: '加权汇总', s: agg.weighted },
      { label: '等权汇总', s: agg.equal_weight },
      { label: `最差：${shortName(agg.worst?.name) || '—'}`, s: agg.worst, cls: 'portfolio-row-worst' },
      { label: `最好：${shortName(agg.best?.name) || '—'}`, s: agg.best, cls: 'portfolio-row-best' },
    ].filter(r => r.s);
    const signed = (k) => ({ value: (r) => r.s[k], fmt: (v) => pct(v), cls: (v) => signOf(v) });
    const tbl = ui.table([
      { key: 'label', label: '口径', fmt: (v, r) => h('span', { title: r.s.name || v }, v) },
      { label: '名义 CAGR', ...signed('cagr_p50') },
      { label: '实际 CAGR', ...signed('real_cagr_p50') },
      { label: '波动率', value: (r) => r.s.vol, fmt: (v) => pct(v) },
      { label: '夏普', value: (r) => r.s.sharpe, fmt: (v) => ui.fmtNum(v, 2) },
      { label: '最大回撤', ...signed('max_dd_p50') },
      { label: '实际亏损', value: (r) => r.s.prob_loss_real, fmt: (v) => pct(v) },
    ], rows);
    tbl.querySelectorAll('tbody tr').forEach((tr, i) => { if (rows[i].cls) tr.classList.add(rows[i].cls); });

    const w = agg.weighted || {};
    const yrs = Math.round((res.horizon_months || 180) / 12);
    const callout = (e, kind, title) => {
      if (!e) return null;
      const d = isNum(e.real_cagr_p50) && isNum(w.real_cagr_p50) ? (e.real_cagr_p50 - w.real_cagr_p50) * 100 : null;
      const vs = d === null ? '' : `，比加权汇总${d < 0 ? '低' : '高'} ${Math.abs(d).toFixed(1)} 个百分点`;
      return h('div', { class: `portfolio-callout ${kind}` }, h('b', {}, `${title}「${e.name}」`),
        `（概率 ${pct(e.prob)}）：实际年化回报中位数 ${pct(e.real_cagr_p50, 2)}${vs}；`,
        `${pct(e.prob_loss_real, 0)} 的路径 ${yrs} 年后跑输通胀，最大回撤中位数 ${pct(e.max_dd_p50)}，夏普 ${ui.fmtNum(e.sharpe, 2)}。`);
    };
    return [tbl, callout(agg.worst, 'bad', '最差情景'), callout(agg.best, 'good', '最好情景')];
  }

  // Scenarios sorted by real median CAGR, best first; the colour of each follows its rank.
  const ranked = (res) => (res.per_scenario || []).slice()
    .sort((a, b) => (isNum(b.real_cagr_p50) ? b.real_cagr_p50 : -Infinity) - (isNum(a.real_cagr_p50) ? a.real_cagr_p50 : -Infinity));
  const colorsOf = (list) => Object.fromEntries(list.map((e, i) => [e.id, rankColor(i, list.length)]));

  function drawRank(res) {
    const per = ranked(res), colors = colorsOf(per), labels = labelsOf(per);
    const wid = res.aggregate?.worst?.id, bid = res.aggregate?.best?.id;
    rangeChart(view.rank, {
      height: rowsHeight(per.length),
      fmt: (v) => `${+(v * 100).toFixed(1)}%`,
      rows: per.map(e => ({ label: labels[e.id], lo: e.real_cagr_p5, mid: e.real_cagr_p50, hi: e.real_cagr_p95,
        color: colors[e.id], size: 7 + 28 * Math.sqrt(Math.max(0, e.prob || 0)),
        tip: `<b>${esc(e.name)}</b>${e.id === wid ? '（最差）' : e.id === bid ? '（最好）' : ''}<br/>概率 ${pct(e.prob)}`
          + `<br/>实际 CAGR 中位数 <b>${pct(e.real_cagr_p50, 2)}</b><br/>5%–95%：${pct(e.real_cagr_p5)} ~ ${pct(e.real_cagr_p95)}`
          + `<br/>名义 CAGR 中位数 ${pct(e.cagr_p50, 2)}` })),
      marks: [{ x: 0, color: '#adb5bd', type: 'solid' },
        { x: res.aggregate?.weighted?.real_cagr_p50, label: '加权中位数', color: '#3b5bdb' }],
    });
  }

  function drawWealth(res) {
    const real = wealthView === 'real';
    const key = real ? 'fan_real' : 'fan';
    const x = res.dates || res.years || [];
    const agg = res.aggregate || {};
    const per = ranked(res), colors = colorsOf(per), labels = labelsOf(per);
    view.fanTitle.textContent = `加权财富扇形图（${real ? '实际购买力' : '名义'}）`;
    const bands = agg.weighted?.[key];
    if (bands) {
      charts.fan(view.fan, { x, bands, fmt: 'x', name: '加权中位数',
        extra: [
          agg.worst?.[key] ? { name: `最差：${labels[agg.worst.id] || shortName(agg.worst.name)}`, data: agg.worst[key].p50, dashed: true, color: '#d9363e' } : null,
          agg.best?.[key] ? { name: `最好：${labels[agg.best.id] || shortName(agg.best.name)}`, data: agg.best[key].p50, dashed: true, color: '#1a9c5b' } : null,
          { name: '期初 1 元', data: x.map(() => 1), dashed: true, color: '#adb5bd' },
        ].filter(Boolean) });
    }
    // 13+ near-identical shades make a legend useless: label the best / worst / weighted lines at
    // their ends and list every scenario, highest first, in the tooltip.
    view.linesTitle.textContent = `各情景的中位数财富路径（${real ? '实际购买力' : '名义'}）`;
    const tagged = { [agg.best?.id]: '最好', [agg.worst?.id]: '最差' };
    const clip = (s) => (s.length > 8 ? `${s.slice(0, 8)}…` : s);
    const lines = [
      ...per.map(e => ({ name: labels[e.id], data: e[key]?.p50 || [], color: colors[e.id],
        width: tagged[e.id] ? 2.4 : 1.3, end: tagged[e.id] ? `${tagged[e.id]}：${clip(labels[e.id])}` : null })),
      { name: '加权中位数', data: bands?.p50 || [], color: '#1d2433', width: 3, end: '加权中位数' },
    ];
    const chart = charts.line(view.lines, { x, yFmt: 'x', legend: false, series: lines });
    const textW = (s) => [...s].reduce((t, c) => t + (c.charCodeAt(0) > 255 ? 11 : 6.5), 0);   // 11px labels
    const endW = Math.round(Math.max(72, ...lines.filter(l => l.end).map(l => textW(l.end) + 16)));
    chart.setOption({
      grid: { top: 16, right: endW },
      tooltip: { formatter: (ps) => [`<b>${esc(ps[0]?.axisValue)}</b>`,
        ...ps.filter(p => isNum(p.value)).sort((a, b) => b.value - a.value)
          .map(p => `${p.marker}${esc(p.seriesName)}：${charts.fmtValue(p.value, 'x')}`)].join('<br/>') },
      series: lines.map(l => ({ endLabel: l.end ? { show: true, formatter: l.end, color: l.color, fontSize: 11, distance: 6 } : { show: false },
        labelLayout: { moveOverlap: 'shiftY' }, emphasis: { focus: 'series' } })),
    });
  }

  function drawTable(res) {
    const wid = res.aggregate?.worst?.id, bid = res.aggregate?.best?.id;
    const by = { prob: (e) => e.prob, real: (e) => e.real_cagr_p50, sharpe: (e) => e.sharpe }[sortBy] || ((e) => e.prob);
    const per = (res.per_scenario || []).slice().sort((a, b) => (isNum(by(b)) ? by(b) : -Infinity) - (isNum(by(a)) ? by(a) : -Infinity));
    const signed = (key, label) => ({ key, label, fmt: (v) => pct(v), cls: (v) => signOf(v) });
    const tbl = ui.table([
      { key: 'name', label: '情景', wrap: true, fmt: (v, r) => h('span', {},
        r.source !== 'builtin' ? r.name : h('a', { href: `#scenarios?${r.id}`, title: '查看这个情景的宏观设定与传导链' }, r.name),
        r.id === wid ? [' ', ui.chip('最差', 'bad')] : null, r.id === bid ? [' ', ui.chip('最好', 'good')] : null,
        r.source === 'custom' ? [' ', ui.chip('自定义', 'accent')] : null) },
      { key: 'prob', label: '概率', fmt: (v) => pct(v) },
      signed('cagr_p5', '名义 p5'), signed('cagr_p50', '名义 p50'), signed('cagr_p95', '名义 p95'),
      signed('real_cagr_p50', '实际 p50'),
      { key: 'vol', label: '波动率', fmt: (v) => pct(v) },
      { key: 'sharpe', label: '夏普', fmt: (v) => ui.fmtNum(v, 2) },
      signed('max_dd_p50', '最大回撤 p50'), signed('max_dd_p95', '回撤坏尾部'),
      { key: 'prob_loss_real', label: '实际亏损概率', fmt: (v) => pct(v), cls: (v) => (v > 0.25 ? 'is-bad' : '') },
    ], per);
    tbl.querySelectorAll('tbody tr').forEach((tr, i) => {
      const id = per[i].id;
      if (id === wid) tr.classList.add('portfolio-row-worst');
      if (id === bid) tr.classList.add('portfolio-row-best');
      if (id === focus) tr.classList.add('weights-focus');
    });
    view.table.replaceChildren(tbl);
  }

  // adjusted: which scenario probabilities the request that produced `res` carried: 'calibrated' (the set
  // calibrated to the one-year forecast), true (hand-set) or false (none: the model posterior)
  function show(res, { adjusted = false } = {}) {
    if (!view) view = buildView();
    shown = res;
    const per = res.per_scenario || [];
    const yrs = Math.round((res.horizon_months || 180) / 12);
    const dates = res.dates || [];
    const custom = per.filter(e => e.source === 'custom').length;
    const ss = res.scenario_set;
    view.summary.textContent = [
      ss ? `${ss.name}${ss.label ? `（${ss.label}）` : ''}` : null,
      weightsText(res.weights), REBALANCE.find(r => r.value === res.rebalance)?.label || res.rebalance,
      `${yrs} 年${dates.length ? `（${dates[0]} → ${dates[dates.length - 1]}）` : ''}`,
      `${per.length} 个情景${custom ? `（含 ${custom} 个自定义）` : ''}`,
      adjusted === 'calibrated' ? '按一年期预测校准的概率' : adjusted ? '手动调整的概率'
        : ss && ss.id !== 'monthly' ? '手设先验概率' : '按今天状态倾斜的概率（模型后验）',
    ].filter(Boolean).join(' · ');
    view.kpis.replaceChildren(...kpis(res));
    const hc = holdingsCard(res, { scenarioYears: yrs, include: shared.include });
    view.holdings.replaceChildren(...(hc ? [hc] : []));
    view.holdings.style.marginTop = hc ? '16px' : '';
    view.compare.replaceChildren(...compare(res).filter(Boolean));
    notesList.replaceChildren(...(res.notes || []).map(n => h('li', { class: 'muted' }, plainNote(n))));
    drawTable(res);
    drawRank(res);
    drawWealth(res);
  }

  // ---- requests
  // the weights a request carries: on the quarterly set only the asset classes (never a request known to fail)
  const reqWeights = () => (quarterlySet() ? editor.getAssets() : editor.get());
  const tickerOnly = () => quarterlySet() && !!editor.get() && !editor.getAssets();
  const qNote = () => (quarterlySet() && editor.heldTickers().length ? [ui.notice(TICKERS_Q, 'info')] : []);
  function body() {
    const weights = reqWeights();
    if (!weights) return null;
    const b = { weights, rebalance };
    if (shared.set) b.scenario_set = shared.set;
    if (shared.include.length) b.include_custom = shared.include.slice();
    const sp = probs.overrides();
    if (sp) b.scenario_probs = sp;
    return b;
  }
  // probability-editor rows from a response, when scenarios.json could not be read
  const per0 = (res) => (res.per_scenario || []).map(e => ({ id: e.id, name: e.name, prior: null, likelihood: null,
    post: e.prob, def: e.prob, custom: e.source === 'custom' }));
  // ---- every weight cleared (清零, or each input set to 0). The editor sends no change for a portfolio it
  // cannot evaluate, so the page watches the editor itself: the previous portfolio's numbers are hidden
  // (not shown as if they were this one's) until a valid portfolio is computed, or until the inputs are
  // back to exactly the request whose result is hidden.
  const ZERO = '组合权重合计为 0：请至少给一个资产或标的设置权重。上一个组合的结果已隐藏，设置权重后会自动重算。';
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
    if (!reqWeights()) { if (!stale) refresh(); return; }
    if (!stale) return;
    errBox.replaceChildren(...qNote());
    clearTimeout(staleTimer);
    const key = keyOf(body());
    if (shown && key === shownKey) { setStale(false); pill.done('已恢复'); return; }
    if (key === pendingKey) return;        // the editor's own change already asked for exactly this
    // the editor fires its own change ~0.4 s after the last edit unless the weights equal the last ones it
    // sent (then another input, e.g. the rebalancing, may have moved meanwhile): recompute in that case too
    staleTimer = setTimeout(refresh, 450);
  }
  editor.el.addEventListener('input', watchWeights);
  editor.el.addEventListener('click', watchWeights);

  let live = false;
  const go = liveRequest(async ({ b, adjusted }) => ({ res: await api.post('portfolio/evaluate', b), adjusted, key: keyOf(b) }), {
    start: () => { pill.busy(); results.classList.add('weights-busy'); },
    done: ({ res, adjusted, key }) => {
      if (!alive()) return;
      pendingKey = null;
      results.classList.remove('weights-busy');
      live = true;
      if (!probs.hasRows()) probs.setRows(per0(res), probInfo);
      show(res, { adjusted });
      shownKey = key;
      probs.sync(res.per_scenario);
      if (!reqWeights()) { setStale(true); pill.fail('权重合计为 0：未计算'); return; }   // cleared while it ran
      errBox.replaceChildren(...qNote());
      setStale(false);
      pill.done('已更新');
      // a deep-linked row or section: kept in view while the results above it settle (tables stack a frame later)
      if (focus && !scrolled) { scrolled = true; const row = view.table.querySelector('.weights-focus'); if (row) ui.pinIntoView(row, { block: 'center' }); }
      else if (focusTicker && !rerender && !scrolled) { scrolled = true; ui.pinIntoView(proj.el, { block: 'start' }); }
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
      if (d && !d.error && d.portfolio && (!shared.set || shared.set === 'monthly')) {
        errBox.append(ui.notice('下面先显示构建时预先计算的默认组合（60/40、模型后验概率、每月再平衡），实时计算恢复后会自动替换。', 'info'));
        show(d.portfolio);
        shownKey = null;
        setStale(!reqWeights());
      } else if (!view) {
        results.replaceChildren();
      }
    },
  });
  function refresh() {
    if (!alive()) return;                  // a timer from a page the user has left
    clearTimeout(staleTimer);
    paintProbState();
    const b = body();
    if (!b) {
      setStale(true);
      errBox.replaceChildren(ui.notice(tickerOnly() ? TICKERS_Q_ONLY : ZERO, 'error'));
      pill.fail(tickerOnly() ? '季度情景集算不了标的：未计算' : '权重合计为 0：未计算');
      return;
    }
    errBox.replaceChildren(...qNote());
    pendingKey = keyOf(b);
    go({ b, adjusted: !b.scenario_probs ? false : probs.mode() === 'calibrated' ? 'calibrated' : probs.adjusted() });
  }

  // ---- scenario list + saved custom scenarios + the scenario sets, then the first evaluation; presets fill in later
  const [s] = await Promise.all([scenP, picker.load(), sets.load()]);
  if (!alive()) return;
  scen = s && !s.error ? s : null;
  probInfo = { method: scen?.probability_method, gamma: scen?.probability?.gamma };
  probs.setRows(rowsNow(), infoNow());
  if (quarterlySet() && rebalance === 'monthly') rebalance = 'quarterly';
  editor.lockTickers(quarterlySet() ? TICKERS_Q : null);
  paintReb();
  paintHorizon();
  if (scen) proj.setScenarios(scen.scenarios || []);
  if (scen) {
    const ids = (scen.scenarios || []).map(x => x.id);
    if (!focus) focus = take(t => ids.includes(t));
  }
  if (tokens.length) root.insertBefore(ui.notice(`无法识别的链接参数：${tokens.join('、')}（可用预设 id、情景 id、标的代码或 monthly / quarterly / annual / none）。`, 'info'), errBox);
  if (linkNotes.length) root.insertBefore(ui.notice(`${linkNotes.join('；')}。`, 'info'), errBox);
  refresh();
  const d = defaults || await defaultsP;
  if (alive() && d && !d.error) editor.setPresets(allPresets(d, cat));
  return focusTicker && !rerender && alive() ? proj.el : undefined;
}
