// One-year forecasts (forecast.json): the next 12 months from today, and the rolling
// track record of forecasts made each year-end with only the data known at the time.
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
`;

// ui.table's `stack` for this page: stacked below 480px (T), 800px (MID) or 1000px (WIDE) of card width
const T = 'fc-t', MID = 'fc-t fc-mid', WIDE = 'fc-t fc-wide';

function ensureStyle() {
  let el = document.getElementById('style-forecast');
  if (!el) { el = document.createElement('style'); el.id = 'style-forecast'; document.head.append(el); }
  if (!el.textContent.includes('/* forecast */')) el.textContent += `/* forecast */\n${CSS}`;
}

export async function render(root, { api, charts, ui, alive = () => root.isConnected }) {
  const { h, card } = ui;
  const pct = (x, d = 1) => ui.fmtPct(x, d);
  ensureStyle();
  root.append(h('p', { class: 'lead' },
    '多年期的路径只有在给出情景主题（以及因子方向）之后才有意义，见“未来情景”；没有主题时能认真打分的最长期限是下一年。',
    '这里每年年底只用当时能看到的数据预测下一年，统计区间准不准、为什么不准。战争、疫情这类外生冲击造成的偏差单独标出，不算模型错误。'));

  let fc, scen;
  try {
    [fc, scen] = await Promise.all([api.get('forecast'), api.get('scenarios').catch(() => null)]);
  } catch (err) {
    if (err.status === 404) root.append(ui.notice('一年期预测结果（forecast.json）还没有生成。', 'info'), ui.buildHint(err));
    else root.append(ui.notice(`读取一年期预测结果失败：${err.message}`, 'error'));
    return;
  }
  if (!alive()) return;                  // the user moved on while the results loaded
  const names = fc.asset_names || {};
  const assetName = (a) => names[a] || ui.assetName(a);
  const live = fc.live || {};

  // ---------------------------------------------------------------- next 12 months
  root.append(h('div', { class: 'section-title' }, `从今天（${live.origin || '—'}）出发的未来 12 个月`));
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
      sub: `按 8 个状态的标准化距离排序（只用于展示的窄核，有效起点约 ${Math.round(live.display_ens || 0)}）；预测本身用的是滚动验证选出的较宽加权（${share}有效起点约 ${Math.round(live.analog_ens || 0)}）`,
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
    root.append(card({ title: '每个情景的第一年', sub: '各情景前 12 个月名义总回报的中位数，颜色按 K 线配色（涨色为正）；情景按概率从高到低排列。多年之后的分化见“未来情景”' }, heatEl));
  }

  // ---------------------------------------------------------------- reconciliation with the scenario library
  const drawRecon = renderReconciliation(root, { ui, charts, fc, pct, assetName });

  // ---------------------------------------------------------------- live ledger
  root.append(h('div', { class: 'section-title' }, '实时样本外记录'));
  root.append(renderLedger({ ui, fc, pct, assetName }));

  // ---------------------------------------------------------------- track record
  root.append(h('div', { class: 'section-title' }, '滚动一年期预测的成绩单'));
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
    title: '每年的预测区间 vs 实际',
    sub: '每根“K 线”是一次预测：实体 = 25%–75%，影线 = 5%–95%；圆点是实际 12 个月回报（红色 = 落在 90% 区间之外）；橙色背景 = 含外生冲击的年份',
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
    card({ title: 'PIT 分布（校准检查）', sub: '实际值在预测分布里的分位；预测校准良好时每一格都接近 10%，两端偏高 = 区间太窄，中间偏高 = 区间太宽' }, pitEl),
    card({ title: '每年的跨资产排序相关（IC）', sub: '预测中位数的资产排序与实际排序的相关；长期为正说明能分辨明年哪些资产相对更好' }, icEl)));

  // misses
  const misses = fc.misses || [];
  if (misses.length) {
    const counts = Object.fromEntries(Object.keys(CAUSE).map(c => [c, misses.filter(m => m.cause === c).length]));
    root.append(card({
      title: '偏离最大的预测：为什么不准',
      sub: `外生冲击 ${counts.shock} 次（不算模型错误）；宏观意外 ${counts.macro} 次（失业率、政策利率、CPI、信用利差或油价的实际变化落在预测的 90% 区间之外，偏高偏低都算：一年期本来就难）；`
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
      root.append(h('div', { class: 'section-title' }, '情景概率与一年期预测的对账'),
        card({}, ui.notice(`暂时无法对账：${cal.reason}。`, 'info')));
    }
    return null;
  }
  const skill = fc.macro_skill || {};
  const calOk = !!cal.available;
  const eq = rec.summary.us_equity;
  const cut = String(skill.split || '2004-12').slice(0, 4);
  root.append(h('div', { class: 'section-title' }, '情景概率与一年期预测的对账'));
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
  root.append(h('div', { style: { height: '16px' } }), card({ title: '校准：只用相似起点法有预测能力的宏观变量',
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
    return card({ title: '实时样本外记录' },
      ui.notice('这份 forecast.json 还没有预测账本：运行 python -m scenariolab.build --only forecast 后出现。', 'info'));
  }
  const names = led.methods || METHOD_CN;
  let asset = 'us_equity';
  const box = h('div');
  const stamp = (v) => (v ? String(v).replace('T', ' ').slice(0, 16) : '—');
  const draw = () => {
    const rows = (led.entries || []).slice().reverse();          // newest first
    box.replaceChildren(rows.length ? ui.table([
      { key: 'origin', label: '起点' },
      { key: 'created', label: '记录于（UTC）', fmt: stamp },
      { key: 'matures', label: '到期（数据到）' },
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
  return card({ title: '实时样本外记录：每次构建存档的一年期预测', actions: sel,
    sub: '下面的成绩单是事后回放（只用当时的数据，但模型和代码是今天的）；这里是每次构建当时真正做出的预测，原样存档、到期后评分，没有任何前视' },
  h('p', { class: 'lead', style: { marginBottom: '10px' } }, led.text || ''),
  !led.matured && led.first_maturity
    ? ui.notice(`还没有到期的记录：第一条在数据更新到 ${led.first_maturity} 之后到期，届时下一次构建会自动给出覆盖率、PIT 和 CRPS。`, 'info') : null,
  led.skipped ? ui.notice(`这次构建没有记账：${led.skipped}。`, 'info') : null,
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
    '怎么记录：每次构建（完整构建或只重建一年期预测这一步）把当时的一年期预测原样存进账本目录（',
    h('code', { class: 'mono' }, `${led.dir || 'data/ledger'}/forecast_<起点>.json`), '），'
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
    legend: { top: 0, data: ['预测区间', '预测中位数', '实际'] },
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
      { name: '预测区间', type: 'candlestick',
        data: f.map(b => (has25 ? [b.p25, b.p75, b.p5, b.p95] : [b.p50, b.p50, b.p5, b.p95])),
        itemStyle: { color: `${color}55`, color0: `${color}55`, borderColor: color, borderColor0: color } },
      { name: '预测中位数', type: 'line', data: f.map(b => b.p50), symbol: 'none', lineStyle: { width: 1.2, color, type: 'dashed' } },
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
