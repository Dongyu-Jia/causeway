// Diagnostics: why the transmission model misses, what the walk-forward loop tried, which fixes survived
// the untouched test months, how the coefficients drift across eras, and (optional) the engine ablation.
const QUESTIONS = ['tuning', 'neural_net', 'missing_factor', 'missing_mechanism'];
const GROUP_COLOR = { tuning: '#3b5bdb', neural_net: '#c2255c', missing_factor: '#2f9e44', missing_mechanism: '#e8590c' };
const GROUP_LABEL = { tuning: '调参数', neural_net: '神经网络', missing_factor: '加因子', missing_mechanism: '加传导机制' };
const CAND = { add_valuation: '+估值变动', add_sentiment: '+消费者信心', add_profit_share: '+利润占比', add_capex_share: '+资本开支占比',
  add_wealth: '+家庭净值', add_white_collar: '+白领就业占比', add_wage: '+普通工资', add_real_y10: '+实际收益率',
  add_profit_growth: '+利润增长', add_fcf_share: '+利润减资本开支', add_cape: '+CAPE 变动', lags1: '1 个月滞后', lags3: '1–3 个月滞后',
  regime_inflation: '通胀状态交互', shocks_add: '+结构冲击', shocks_replace_oil: '结构冲击替代油价', 'ridge_0.01': '岭回归 0.01',
  'ridge_0.05': '岭回归 0.05', window_15: '只用近 15 年', window_25: '只用近 25 年', mlp16: '神经网络 16', mlp32: '神经网络 32' };
const SHORT = { us_equity: '美股大盘', nasdaq: '纳指100', energy: '能源股', industrials: '工业股', staples: '必需消费',
  healthcare: '医疗', utilities: '公用事业', financials: '金融股', real_estate: 'REITs', gold: '黄金',
  crude_oil: '原油', treasury_10y: '10年美债', cash: '现金' };
const SPAN = { train: { label: '训练', bg: '#dbe4ff', ink: '#364fc7' }, validate: { label: '验证', bg: '#ffe8cc', ink: '#c2410c' },
  test: { label: '测试', bg: '#d3f9d8', ink: '#2b8a3e' } };
const PRIORITY = { 高: 'bad', 中: 'warn', 低: '' };
const VERDICT = { supported: ['支持', 'good'], rejected: ['否定', 'bad'], inconclusive: ['不确定', 'warn'] };
const ROLE = { validation: '验证', test: '测试' };
// Engine ablation: one colour per hypothesis family, readable names for parameters and selection checks.
const HYP_COLOR = { H1: '#3b5bdb', H2: '#e8590c', H3: '#2f9e44', H4: '#7048e8', H5: '#c2255c', H6: '#1098ad' };
const PARAM = { resid_mode: '残差处理', resid_window: '滚动窗口（月）', resid_coupling: '残差保留比例', phase_months: '阶段月数',
  views: '视图集合', cma: 'CMA', min_ens: '最少有效月份' };
const PARAM_VALUE = { coupled: '耦合', decoupled: '解耦', detrended: '去趋势', trend_decoupled: '慢变解耦', partial: '部分耦合',
  recentered: '漂移收缩', full: '8 个变量', levels2: '2 个水平视图', levels8: '8 个水平视图', historical: '历史漂移',
  valuation_adjusted: '估值调整' };
const CHECK = { coverage: '覆盖率', forward_fan: '长期分布', gain: '改善幅度' };

// The ablation prose (ablation.json) names engine variants, scenarios and settings by id: "recentered_50（验证
// 3.40%…）", "变化最大的是 financial_crisis", "相对 historical 的改善", "H6 MIN_ENS". Show their names instead.
// An id the text puts in parentheses right after its name ("「部分耦合（50% 混合）」（partial_50）") is a deliberate
// cross-reference and stays; a space next to a replaced id is dropped where it would sit between Chinese characters.
function humanIds(variants, scen) {
  const names = new Map();
  // "baseline" is both the reference variant and a scenario id: the text's own wording is kept for it
  for (const v of variants || []) if (v?.id && v.name && v.id !== 'baseline') names.set(v.id, `「${v.name}」`);
  for (const s of scen?.scenarios || []) if (s?.id && s.name && s.id !== 'baseline' && !names.has(s.id)) names.set(s.id, `「${s.name}」`);
  for (const [k, x] of Object.entries({ historical: '历史漂移', valuation_adjusted: '估值调整', MIN_ENS: '最少有效月份', spec: '情景规格',
    'backtest.json': '回测页', 'config.RESIDUALS': '引擎的残差设定' })) {
    if (!names.has(k)) names.set(k, x);
  }
  if (!names.size) return (t) => t;
  const alts = [...names.keys()].sort((a, b) => b.length - a.length).map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  const rx = new RegExp(`( ?)(?<!\\w)(${alts})(?!\\w)( ?)`, 'g');
  const wide = (ch) => !!ch && /[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]/.test(ch);
  return (t) => (typeof t !== 'string' ? t : t.replace(rx, (m, sp1, id, sp2, off, str) => {
    const start = off + sp1.length, end = start + id.length;
    if (str[start - 1] === '（' && str[end] === '）') return m;          // "「name」（id）": already named
    return `${sp1 && !wide(str[off - 1]) ? ' ' : ''}${names.get(id)}${sp2 && !wide(str[off + m.length]) ? ' ' : ''}`;
  }).replace(/历史漂移漂移/g, '历史漂移'));                                // "historical 漂移"
}
const GAP_KEY = { f_valuation: '估值变化（内生）', f_profit_share: '利润占比', f_capex_share: '资本开支占比',
  all: '三者合计', unexplained: '仍未解释', note: '说明项' };
const INK = '#1d2433', INK3 = '#7a8396', LINE = '#e3e6ec', SPLIT = '#eef0f4';
// p-values per finding; the drift test reports a t statistic (normal approximation).
const TESTS = [
  { finding: 'missing_lags', label: '残差自相关 Ljung-Box', p: (x) => x.lb_p },
  { finding: 'missing_lags', label: '1–3 月滞后 F 检验', p: (x) => x.lag_p },
  { finding: 'regime', label: '通胀状态交互 F 检验', p: (x) => x.p },
  { finding: 'parameter_break', label: '训练/验证系数断点 Chow', p: (x) => x.p },
  { finding: 'nonlinearity', label: '非线性 RESET', p: (x) => x.p },
  { finding: 'drift_bias', label: '漂移偏差 t 检验', p: (x) => twoSidedP(x.t) },
];
const EVI = {
  base_oos_r2: '基准模型各样本外十年的 R²', last_selected: '最后一轮选中的模型', changes: '改动', oos_delta: '单独加入后样本外 R² 的变化',
  leakage: '与美股价格收益的相关（泄漏检查）', chow_reject: 'Chow 检验拒绝的资产', us_equity_y10_beta_pre1999: '美股对 10 年期收益率的 β（1999 年前）',
  us_equity_y10_beta_post2005: '美股对 10 年期收益率的 β（2005 年后）', reset_reject: 'RESET 显示非线性的资产数（各轮）',
  gold_acf1: '黄金残差一阶自相关', others_acf1: '其他风险资产平均一阶自相关', r2_risk_avg: '全样本风险资产平均 R²',
  bias: '验证期偏差最大的资产（年化）', robust: '每个十年都稳定提升的候选', n_reject: '交互项显著的资产数（各轮）',
};

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
// Fixed-point text without a "-0.00" and, when signed, with "+" only on values that round above zero.
function fixed(x, d, signed = false) {
  let s = x.toFixed(d);
  if (Number(s) === 0) s = (0).toFixed(d);
  return signed && Number(s) > 0 ? `+${s}` : s;
}
const pct = (v, d = 1, signed = false) => (isNum(v) ? `${fixed(v * 100, d, signed)}%` : '—');
const num = (v, d = 2, signed = false) => (isNum(v) ? fixed(v, d, signed) : '—');
const r2s = (v) => num(v, 3, true);
const period = (p) => (Array.isArray(p) && p[0] && p[1] ? `${String(p[0]).slice(0, 4)}–${String(p[1]).slice(0, 4)}` : '—');
const ymNum = (ym) => { const [y, m] = String(ym).split('-').map(Number); return y * 12 + ((m || 1) - 1); };
const mean = (xs) => { const v = xs.filter(isNum); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const pFmt = (p) => (!isNum(p) ? '—' : p < 0.001 ? '<0.001' : p.toFixed(3));
const boxKey = (color) => `<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${color};vertical-align:middle;margin-right:6px"></span>`;
// The months a spec is actually fitted on: a recent-window spec only uses the last `window_years` of the training span.
function fitPeriod(it) {
  const wy = it?.start?.spec?.window_years;
  if (!isNum(wy) || !Array.isArray(it.train)) return period(it?.train);
  const end = ymNum(it.train[1]);
  return `${Math.floor(Math.max(ymNum(it.train[0]), end - wy * 12 + 1) / 12)}–${String(it.train[1]).slice(0, 4)}`;
}
// Smallest of 1, 2, 2.5, 5 x 10^k that is >= x.
function niceCeil(x) {
  if (!isNum(x) || x <= 0) return 1;
  const mag = 10 ** Math.floor(Math.log10(x));
  return [1, 2, 2.5, 5, 10].map(m => m * mag).find(v => v >= x * 0.9999);
}
let measureCtx = null;
function textWidth(s, font = '11px sans-serif') {
  try {
    measureCtx = measureCtx || document.createElement('canvas').getContext('2d');
    measureCtx.font = font;
    return measureCtx.measureText(String(s)).width;
  } catch (_) { return [...String(s)].reduce((w, ch) => w + (ch.charCodeAt(0) > 255 ? 11 : 6.5), 0); }
}

// Two-sided normal p-value of a t statistic (Abramowitz–Stegun 7.1.26).
function twoSidedP(t) {
  if (!isNum(t)) return null;
  const x = Math.abs(t) / Math.SQRT2, k = 1 / (1 + 0.3275911 * x);
  const poly = ((((1.061405429 * k - 1.453152027) * k + 1.421413741) * k - 0.284496736) * k + 0.254829592) * k;
  return Math.min(1, Math.max(0, poly * Math.exp(-x * x)));
}

// R^2 change of one candidate alone on top of the base, per out-of-sample decade: the first validation
// period, then every test period (iteration k's test is iteration k+1's validation).
function oosDeltas(its, cid) {
  const abl = its.map(it => Object.fromEntries((it.test_ablation || []).map(r => [r.candidate, r])));
  if (!abl.length || !abl.every(a => a[cid])) return [];
  return [{ period: period(its[0].validate), delta: abl[0][cid].val_delta },
    ...its.map((it, i) => ({ period: period(it.test), delta: abl[i][cid].test_delta }))];
}

const CSS = `/* diagnostics */
.diagnostics-why { color: var(--ink-2); font-size: 14px; line-height: 1.8; margin: 0; }
.diagnostics-chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.diagnostics-verdict { font-size: 13px; color: var(--ink-2); margin: 0 0 6px; line-height: 1.65; }
.diagnostics-rec { border-top: 1px solid var(--line); padding: 12px 0 2px; }
.diagnostics-rec-head { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.diagnostics-rec-head b { font-size: 14px; }
.diagnostics-rec p { margin: 6px 0; font-size: 13px; color: var(--ink-2); line-height: 1.7; }
.diagnostics-evi { font-size: 12px; margin: 2px 0 8px; }
.diagnostics-evi summary { cursor: pointer; color: var(--accent); margin-bottom: 4px; }
.diagnostics-evi dl { margin: 0; }
.diagnostics-evi dt { color: var(--ink-3); margin-top: 6px; }
.diagnostics-evi dd { margin: 2px 0 0; min-width: 0; color: var(--ink-2); }
.diagnostics-evi table.tbl { font-size: 12px; width: auto; min-width: 70%; }
.diagnostics-evi table.tbl th, .diagnostics-evi table.tbl td { padding: 4px 10px; }
.diagnostics-q .card-actions .chip { white-space: nowrap; }
.diagnostics-tl { display: grid; gap: 8px; }
.diagnostics-tl-row { display: grid; grid-template-columns: 70px minmax(0, 1fr); gap: 10px; align-items: center; }
.diagnostics-tl-label { font-size: 12px; color: var(--ink-2); font-weight: 600; }
.diagnostics-tl-track { position: relative; height: 28px; background: var(--panel-2); border-radius: 6px; }
.diagnostics-tl-seg { position: absolute; top: 2px; bottom: 2px; border-radius: 4px; font-size: 11px; display: flex; align-items: center;
  justify-content: center; overflow: hidden; white-space: nowrap; padding: 0 4px; }
.diagnostics-tl-axis { position: relative; height: 16px; margin-left: 80px; font-size: 11px; color: var(--ink-3); }
.diagnostics-tl-axis span { position: absolute; transform: translateX(-50%); }
.diagnostics-legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--ink-2); margin-bottom: 8px; }
.diagnostics-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.diagnostics-legend s { display: inline-block; width: 16px; border-top: 1px dashed var(--ink-3); margin-right: 5px; vertical-align: 3px; }
.diagnostics-round h4 { margin: 0; font-size: 13px; font-weight: 600; }
.diagnostics-round .muted { font-size: 12px; min-height: 18px; }
.diagnostics-path { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 12px; margin-top: 12px; }
.diagnostics-path b { margin-left: 4px; }
.diagnostics-arrow { color: var(--ink-3); }
.diagnostics-carry { margin-top: 12px; }
table.diagnostics-heat td.c, table.diagnostics-heat th.c { text-align: center; }
table.diagnostics-heat tr.g td { background: var(--panel-2); font-weight: 600; color: var(--ink-2); text-align: left; }
table.diagnostics-heat td.h { white-space: normal; text-align: left; color: var(--ink-2); min-width: 200px; }
.diagnostics-finding { border-top: 1px solid var(--line); padding: 10px 0; }
.diagnostics-finding:first-child { border-top: 0; padding-top: 0; }
.diagnostics-finding p { margin: 6px 0 0; font-size: 13px; color: var(--ink-2); line-height: 1.65; }
.diagnostics-split, .diagnostics-2col { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
@media (max-width: 1100px) { .diagnostics-split, .diagnostics-2col { grid-template-columns: minmax(0, 1fr); } }
.diagnostics-kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.diagnostics-kpis .kpi { padding: 10px 12px; box-shadow: none; }
.diagnostics-kpis .kpi-value { font-size: 18px; }
.diagnostics-ctl { display: grid; gap: 8px; padding: 12px 16px; }
.diagnostics-ctl-row { display: flex; gap: 10px; align-items: flex-start; }
.diagnostics-ctl-label { flex: 0 0 60px; font-size: 12px; color: var(--ink-3); padding-top: 5px; }
.diagnostics-pills { display: flex; flex-wrap: wrap; gap: 6px; min-width: 0; }
.diagnostics-pill { font: inherit; font-size: 12px; border: 1px solid var(--line); background: var(--panel); color: var(--ink-2);
  border-radius: 999px; padding: 3px 11px; cursor: pointer; }
.diagnostics-pill:hover { border-color: #c6cbd6; }
.diagnostics-pill.on { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
.diagnostics-anchor { scroll-margin-top: 80px; }
.diagnostics-text { color: var(--ink-2); font-size: 13px; line-height: 1.75; margin: 10px 0 0; }
.diagnostics-select { border: 1px solid #cdd7ff; background: #f7f9ff; border-radius: 8px; padding: 12px 14px; }
table.tbl td.diagnostics-nowrap { white-space: nowrap; font-weight: 600; color: var(--ink); vertical-align: top; }
table.tbl td.diagnostics-top { vertical-align: top; }
.diagnostics-hyp { min-width: 320px; line-height: 1.7; color: var(--ink-2); }
.diagnostics-hyp details { margin-top: 4px; }
.diagnostics-hyp details p { margin: 4px 0; }
table.diagnostics-var td { vertical-align: top; }
/* long variant names wrap, so the table fits a 1440px screen without a sideways scrollbar */
table.diagnostics-var td:first-child { white-space: normal; min-width: 120px; }
table.diagnostics-var td.d { white-space: normal; text-align: left; min-width: 220px; max-width: 340px; font-size: 12px; color: var(--ink-2); line-height: 1.6; }
table.diagnostics-var td.d .mono { display: block; color: var(--ink-3); margin-top: 2px; }
table.diagnostics-var tr.g td { background: var(--panel-2); color: var(--ink-2); font-weight: 600; text-align: left; }
table.diagnostics-var tr.sel td { background: #f0fbf4; }
table.diagnostics-var td.s { white-space: normal; text-align: left; min-width: 124px; max-width: 170px; font-size: 12px; color: var(--ink-3); }
table.diagnostics-var td.s .chip { white-space: nowrap; margin-bottom: 2px; }
.diagnostics-var small { display: block; color: var(--ink-3); font-size: 11px; }
.diagnostics-var small.is-good { color: var(--good); } .diagnostics-var small.is-bad { color: var(--bad); }
.diagnostics-var td .is-bad > small { color: var(--bad); }
.diagnostics-checks { display: flex !important; flex-wrap: wrap; gap: 0 8px; margin-top: 4px; }
.diagnostics-var .chip + .chip { margin-left: 4px; }
`;

function ensureStyle() {
  let el = document.getElementById('style-fe2r');
  if (!el) { el = document.createElement('style'); el.id = 'style-fe2r'; document.head.append(el); }
  if (!el.textContent.includes('/* diagnostics */')) el.textContent += CSS;
}

function failure(ui, err) {
  const { h } = ui;
  if (err?.status === 404) {
    return h('div', {}, ui.notice('诊断结果（diagnostics.json）还没有生成。', 'info'), ui.buildHint(err));
  }
  return ui.notice(`读取诊断结果失败：${err?.message || err}`, 'error');
}

function pills(h, options, value, onChange) {
  const wrap = h('div', { class: 'diagnostics-pills' });
  const paint = (v) => wrap.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  for (const o of options) {
    wrap.append(h('button', { class: 'diagnostics-pill', 'data-v': o.value, title: o.title || null,
      onclick: () => { paint(o.value); onChange(o.value); } }, o.label));
  }
  paint(value);
  return wrap;
}

function clearChart(el, node) {
  if (el._chart) { el._chart.dispose(); el._chart = null; }
  el.style.height = 'auto';
  el.replaceChildren(node);
  el._cleared = true;
}

// Undo clearChart() before drawing into the box again (drop the message, restore the CSS height).
function fresh(el) {
  if (el._cleared) { el.replaceChildren(); el.style.height = ''; el._cleared = false; }
  return el;
}

// One walk-forward step: R^2 change of every candidate tried in `round`; rows are shared across steps.
function roundChart(charts, el, { rows, trials, round, lo, hi, minGain, label }) {
  const chart = charts.mount(fresh(el));
  const by = Object.fromEntries(trials.filter(t => t.round === round).map(t => [t.candidate, t]));
  const item = (t) => {
    const v = t.delta, c = Math.min(Math.max(v, lo), hi);
    const it = { value: c, t };
    if (t.eligible === false) it.itemStyle = { opacity: 0.35 };
    if (t.accepted) it.label = { show: true, position: 'right', formatter: `采纳 ${r2s(v)}`, color: INK, fontSize: 10, fontWeight: 600 };
    else if (c !== v) it.label = { show: true, position: v > 0 ? 'insideRight' : 'insideLeft', formatter: r2s(v), fontSize: 10,
      color: t.eligible === false ? INK : '#fff' };
    return it;
  };
  chart.setOption({
    animation: false,
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true,
      formatter: (params) => {
        const c = rows[params?.[0]?.dataIndex];
        const t = by[c];
        if (!t) return `${label(c)}<br/><span style="color:${INK3}">这一步没有尝试（已采纳、只作对照或不改变模型）</span>`;
        return [`<b>${label(c)}</b>（${GROUP_LABEL[t.group] || t.group || ''}${t.eligible === false ? '，内生，只作对照' : ''}）`,
          t.hypothesis || '', `验证期平均 R² <b>${r2s(t.val_score)}</b>（变化 <b>${r2s(t.delta)}</b>）`,
          `训练期平均 R² ${num(t.train_r2, 3)}${t.accepted ? '　<b>被采纳</b>' : ''}`].filter(Boolean).join('<br/>');
      },
    },
    grid: { left: 4, right: 52, top: 6, bottom: 22, containLabel: true },
    xAxis: { type: 'value', min: lo, max: hi, interval: hi, axisLine: { show: false }, axisTick: { show: false },
      splitLine: { lineStyle: { color: SPLIT } }, axisLabel: { color: INK3, fontSize: 10, formatter: (v) => String(+v.toFixed(3)) } },
    yAxis: { type: 'category', inverse: true, data: rows, axisTick: { show: false }, axisLine: { lineStyle: { color: LINE } },
      axisLabel: { fontSize: 11, color: INK, formatter: (c) => (by[c] ? label(c) : `{off|${label(c)}}`), rich: { off: { color: '#c1c7d0', fontSize: 11 } } } },
    series: QUESTIONS.map((q, qi) => ({
      name: GROUP_LABEL[q], type: 'bar', stack: 'd', barMaxWidth: 14, itemStyle: { color: GROUP_COLOR[q], borderRadius: 2 },
      data: rows.map(c => (by[c] && by[c].group === q && isNum(by[c].delta) ? item(by[c]) : null)),
      markLine: qi === 0 ? { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: INK3, type: 'dashed', width: 1 },
        data: [{ xAxis: minGain }] } : undefined,
    })),
  });
  return chart;
}

// Rows = tests, columns = assets, colour = significance band. Column labels stand upright and
// the p values shorten (".736", "<.001") on narrow cells, and give way to the colours (tooltip
// keeps the number) where even those do not fit; the box itself never gets narrower than 560 px
// (it scrolls sideways inside the card on a phone, see ui.wideChart).
const textW = (s, size = 11) => [...String(s ?? '')].reduce((t, ch) => t + (/[\x00-\xff]/.test(ch) ? size * 0.6 : size), 0);
function pvalChart(charts, el, rows, cols, colLabels) {
  fresh(el);
  const width = el.clientWidth || 600;
  const rowPx = Math.min(width * 0.45, Math.max(40, ...rows.map(r => textW(r.label))) + 12);
  const cellW = (width - rowPx - 24) / Math.max(cols.length, 1);
  const colPx = Math.max(10, ...colLabels.map(s => textW(s)));
  const rotate = colPx <= cellW - 4 ? 0 : 90;
  const top = 28 + (rotate ? colPx + 8 : 22);                      // the legend row, then the column labels
  const short = cellW < 38;
  const pText = (v) => (short ? (v < 0.001 ? '<.001' : v.toFixed(3).replace(/^0\./, '.')) : pFmt(v));
  el.style.height = `${rows.length * 26 + top + 12}px`;
  const chart = charts.mount(el);
  const data = [];
  rows.forEach((r, i) => cols.forEach((a, j) => {
    const p = r.values[a];
    data.push(isNum(p) ? { value: [j, i, p], label: p < 0.01 ? { color: '#fff' } : undefined } : { value: [j, i, '-'] });
  }));
  chart.setOption({
    tooltip: { confine: true, formatter: (p) => {
      const [j, i, v] = p.data.value;
      if (!isNum(v)) return '';
      return `<b>${rows[i].label}</b> · ${colLabels[j]}<br/>p = <b>${pFmt(v)}</b>${v < 0.05 ? '（显著：拒绝“没有问题”的原假设）' : '（不显著）'}`;
    } },
    grid: { left: 8, right: 12, top, bottom: 8, containLabel: true },
    xAxis: { type: 'category', position: 'top', data: colLabels, axisTick: { show: false }, axisLine: { show: false },
      axisLabel: { color: INK, fontSize: 11, interval: 0, rotate }, splitArea: { show: false } },
    yAxis: { type: 'category', inverse: true, data: rows.map(r => r.label), axisTick: { show: false }, axisLine: { show: false },
      axisLabel: { color: INK, fontSize: 11, interval: 0 } },
    visualMap: { type: 'piecewise', dimension: 2, orient: 'horizontal', left: 'center', top: 0, itemWidth: 14, itemHeight: 10,
      textStyle: { color: INK3, fontSize: 11 },
      pieces: [{ lt: 0.01, label: 'p < 0.01 强显著', color: '#c92a2a' }, { gte: 0.01, lt: 0.05, label: 'p < 0.05 显著', color: '#ff8787' },
        { gte: 0.05, lt: 0.1, label: 'p < 0.10 边缘', color: '#ffe3e3' }, { gte: 0.1, label: 'p ≥ 0.10 不显著', color: '#f1f3f5' }] },
    series: [{ type: 'heatmap', data, itemStyle: { borderColor: '#fff', borderWidth: 2 },
      label: { show: cellW >= 24, fontSize: short ? 9 : 10, color: INK, formatter: (p) => (isNum(p.value[2]) ? pText(p.value[2]) : '') },
      emphasis: { itemStyle: { borderColor: INK, borderWidth: 1 } } }],
  });
  return chart;
}

// Rolling beta of one asset to one factor with a +/- 2 standard-error band.
function betaChart(charts, el, { years, beta, se, color }) {
  const chart = charts.mount(fresh(el));
  const lo = beta.map((b, i) => (isNum(b) && isNum(se[i]) ? b - 2 * se[i] : null));
  const hi = beta.map((b, i) => (isNum(b) && isNum(se[i]) ? b + 2 * se[i] : null));
  const ghost = (name, data) => ({ name, type: 'line', data, symbol: 'none', silent: true, tooltip: { show: false }, lineStyle: { opacity: 0 }, z: 0 });
  chart.setOption({
    legend: { top: 0, data: ['β（每 1 个标准差）', '±2 倍标准误'], textStyle: { color: INK, fontSize: 12 }, itemWidth: 16, itemHeight: 8 },
    tooltip: { trigger: 'axis', confine: true, formatter: (params) => {
      const p = (params || []).find(q => q.seriesName === 'β（每 1 个标准差）');
      if (!p) return '';
      const i = p.dataIndex;
      return `<b>${years[i]}</b>（过去 10 年窗口）<br/>β <b>${pct(beta[i], 2, true)}</b><br/>±2 倍标准误：${pct(lo[i], 2, true)} ~ ${pct(hi[i], 2, true)}`;
    } },
    grid: { left: 56, right: 20, top: 34, bottom: 30 },
    xAxis: { type: 'category', data: years, boundaryGap: false, axisLine: { lineStyle: { color: LINE } }, axisTick: { show: false },
      axisLabel: { color: INK3, fontSize: 11 } },
    yAxis: { type: 'value', scale: true, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: SPLIT } },
      axisLabel: { color: INK3, fontSize: 11, formatter: (v) => pct(v, 1) } },
    series: [
      ghost('_lo', lo), ghost('_hi', hi),
      { name: '±2 倍标准误', type: 'custom', silent: true, clip: true, z: 1, tooltip: { show: false }, itemStyle: { color, opacity: 0.15 },
        data: [[0, lo.find(isNum) ?? 0]],
        renderItem: (params, api) => {
          const top = [], bottom = [];
          hi.forEach((v, i) => { if (isNum(v) && isNum(lo[i])) { top.push(api.coord([i, v])); bottom.push(api.coord([i, lo[i]])); } });
          return { type: 'polygon', silent: true, shape: { points: top.concat(bottom.reverse()) }, style: { fill: color, opacity: 0.15 } };
        } },
      { name: 'β（每 1 个标准差）', type: 'line', data: beta, symbol: 'circle', symbolSize: 5, showSymbol: false, z: 3,
        lineStyle: { width: 2, color }, itemStyle: { color },
        markLine: { silent: true, symbol: 'none', label: { show: false },
          lineStyle: { color: '#adb5bd', type: 'solid', width: 1 }, data: [{ yAxis: 0 }] } },
    ],
  });
  return chart;
}

export async function render(root, { api, charts, ui, param, alive = () => root.isConnected }) {
  const { h, card } = ui;
  ensureStyle();
  root.append(h('p', { class: 'lead' },
    '诊断回答四个问题：是不是要调参数、要不要训练小型神经网络、是不是少了一个因子、是不是缺一个传导机制。',
    '做法是走步验证（walk-forward：沿时间一段一段往前推，每一段都只用它之前的数据）：每一轮先诊断模型在训练期和验证期哪里不准，',
    '再把候选改动逐个加上去、只按验证期打分挑选，最后在从未参与挑选的测试期检验，能延续到测试期的改动才保留。',
    '打分 = 用真实因子变化预测 10 类风险资产月度价格收益的样本外 R²，衡量的是传导质量，而不是预测宏观。'));
  const ablation = api.get('ablation').then(d => ({ d }), e => ({ e }));       // optional file: never blocks the page
  const wait = ui.loading('读取诊断结果…');
  root.append(wait);
  let dg;
  try { dg = await api.get('diagnostics'); } catch (err) { wait.remove(); root.append(failure(ui, err)); return; }
  wait.remove();
  if (!alive()) return;                  // the user moved on while the results loaded

  const its = (dg.iterations || []).filter(it => it && Array.isArray(it.train) && Array.isArray(it.validate) && Array.isArray(it.test));
  const fin = dg.final || {};
  const ra = dg.residual_analysis || {};
  const recs = dg.recommendations || [];
  const cands = dg.candidates || [];
  const byId = Object.fromEntries(cands.map(c => [c.id, c]));
  const minGain = isNum(dg.min_gain) ? dg.min_gain : 0.003;
  const scoreAssets = dg.score_assets || [];
  const nA = scoreAssets.length || 10;
  const questions = dg.questions || {};
  const label = (id) => CAND[id] || byId[id]?.hypothesis || id || '—';
  const plus = (id) => (label(id).startsWith('+') ? label(id) : `+${label(id)}`);
  const nameOf = (a) => ui.META.assets[a]?.name || a;
  const short = (a) => SHORT[a] || nameOf(a);
  const deltas = Object.fromEntries(cands.map(c => [c.id, its.length ? oosDeltas(its, c.id) : []]));
  const robust = (id) => byId[id]?.eligible !== false && deltas[id]?.length > 0 && deltas[id].every(d => isNum(d.delta) && d.delta > minGain);
  const periods = [...new Set(Object.values(deltas).flatMap(ds => ds.map(d => d.period)))];
  const tile = (l, v, s, c = '') => ui.kpi(l, v, s, c);
  const sections = {};
  const itCharts = [];          // chart work queued by iterationBlock(), drawn once the DOM is in place
  // The page is long: an in-page table of contents (each entry is also a deep link, #diagnostics?<key>).
  const TOC = [['recs', '四个问题的回答'], ['loop', '走步迭代'], ['robust', '候选改动的稳健性'], ['tests', '诊断检验'],
    ['betas', '滚动 β'], ['mlp', '小型神经网络'], ['ablation', '引擎消融：为什么“知道宏观”反而更差']];
  root.append(h('div', { class: 'diagnostics-chips', style: { margin: '-6px 0 4px' } }, h('span', { class: 'muted', style: { fontSize: '12px' } }, '跳到：'),
    TOC.map(([k, t]) => h('a', { href: `#diagnostics?${k}`, class: 'chip', onclick: (e) => {
      if (!sections[k]) return;
      e.preventDefault();
      sections[k].scrollIntoView({ behavior: 'smooth', block: 'start' });
      ui.setRoute('diagnostics', k, alive);
    } }, t))));

  // ---- verdict
  const baseOos = its.length ? [{ period: period(its[0].validate), r2: its[0].baseline?.val_score },
    ...its.map(it => ({ period: period(it.test), r2: it.test_result?.base_all_months?.score }))] : [];
  const nRobust = cands.filter(c => robust(c.id)).length;
  const sum = ra.summary || {};
  root.append(h('div', { class: 'section-title' }, '结论：预测不准是为什么'));
  root.append(h('div', { class: 'grid cols-4' },
    tile('全样本风险资产 R²', num(sum.r2_risk_avg, 3), `宏观因子能解释约 ${pct(sum.r2_risk_avg, 0)} 的月度价格波动（样本内）`),
    ...baseOos.map(b => tile(`基准模型样本外 R²（${b.period}）`, r2s(b.r2), '只用之前的数据估计；> 0 才好过历史均值', isNum(b.r2) ? ui.goodCls(b.r2 >= 0) : ''))));
  const robustIds = cands.filter(c => robust(c.id)).map(c => c.id);
  const why = [
    `宏观因子在全样本只解释风险资产月度价格波动的 ${pct(sum.r2_risk_avg, 0)}；`,
    baseOos.length ? `只用之前的数据估计、再去预测之后的十年，基准模型的样本外 R² 依次为 ${baseOos.map(b => `${b.period} ${r2s(b.r2)}`).join('、')}（小于 0 表示还不如直接用历史均值）。` : '',
    isNum(sum.n_chow_reject) ? `全样本 Chow 检验显示 ${sum.n_chow_reject}/${nA} 个风险资产的传导系数在 ${String(ra.break_date || '').slice(0, 4) || '—'} 年前后显著不同。` : '',
    cands.length ? `试过的 ${cands.length} 个候选改动中，单独加在基准上、在每个样本外十年都提升 R² 超过 +${minGain} 的有 ${nRobust} 个${nRobust ? `（${robustIds.map(label).join('、')}）` : ''}。` : '',
    nRobust === 0 && sum.n_chow_reject >= 3 ? '所以不准的主因是传导关系随时代在变，而不是少了某个因子或缺了非线性；这靠改模型形式解决不了，更稳妥的是保留全样本系数，在情景里用熵池化把权重放到相似时代的月份上。' : '',
  ].join('');
  root.append(h('div', { style: { height: '16px' } }), card({}, h('p', { class: 'diagnostics-why' }, why)));

  const finChanges = fin.changes || [];
  // The last selection is scored on the months both it and the base can predict (structural shocks are blank in
  // the COVID months), so it is compared with the base on that same sample, not with the base's all-month score.
  const lastTr = its.length ? its[its.length - 1].test_result || {} : {};
  const lastBase = lastTr.base?.score;
  root.append(h('div', { style: { height: '16px' } }), card({
    title: `最终采用：${fin.is_base === true ? `基准线性模型（${fin.describe || '—'}）` : fin.describe || '—'}`,
    sub: '走步验证最后一轮选中的改动，必须在未参与挑选的测试期胜过基准才会被采用，否则保留基准；最终模型在全部数据上重新估计，供情景引擎使用。',
    actions: h('div', { class: 'diagnostics-chips' },
      fin.is_base === true ? ui.chip('保留基准模型', 'accent') : fin.is_base === false ? ui.chip('采用改进模型', 'good') : null,
      fin.test_used_for_choice === true ? ui.chip('测试期参与了“采用 / 不采用”的决定', 'warn')
        : fin.test_used_for_choice === false ? ui.chip('测试期只用于报告', 'good') : null),
  },
  h('div', { class: 'diagnostics-kpis' },
    tile('验证期 R²', r2s(fin.val_score), `选择所用验证期 ${period(fin.selected_on?.validate)}`),
    tile('测试期 R²', r2s(fin.test_score), `${period(fin.selected_on?.test)}，${fin.test_months ?? '—'} 个月`),
    tile('基准的测试期 R²', r2s(fin.base_test_score), '同一测试期'),
    tile('最后一轮选中', r2s(fin.last_selected?.test_score),
      isNum(lastBase) ? `测试期 R²；同一批 ${lastTr.n_months ?? '—'} 个月上基准为 ${r2s(lastBase)}` : `测试期 R²；验证期 ${r2s(fin.last_selected?.val_score)}`,
      isNum(fin.last_selected?.test_score) && isNum(lastBase ?? fin.base_test_score)
        && fin.last_selected.test_score < (lastBase ?? fin.base_test_score) ? 'is-bad' : '')),
  h('dl', { class: 'kv', style: { marginTop: '12px' } },
    h('dt', {}, '采用的改动'), h('dd', {}, finChanges.length ? finChanges.map(label).join('、') : '无（基准线性模型）'),
    h('dt', {}, '最后一轮选中'), h('dd', {}, `${fin.last_selected?.describe || '—'}（${(fin.last_selected?.changes || []).map(label).join(' → ') || '无改动'}；验证期 R² ${r2s(fin.last_selected?.val_score)}）`),
    h('dt', {}, '最终估计样本'), h('dd', {}, Array.isArray(fin.trained_on) ? `${fin.trained_on[0]} 至 ${fin.trained_on[1]}` : '—')),
  fin.reason ? h('p', { class: 'diagnostics-text' }, fin.reason) : null,
  fin.test_used_for_choice === true ? h('p', { class: 'diagnostics-text muted' },
    '披露：测试期决定了“最后一轮的选择要不要采用”，之后没有再留出的数据。',
    fin.is_base === true ? '因为最终保留的是事先固定的基准模型，它的测试期 R² 仍是无偏的样本外估计。' : '选中模型的测试期 R² 是在通过这道关卡的条件下观察到的，会略偏乐观。') : null));

  // ---- four questions (recommendations grouped by question)
  const recGrid = h('div', { class: 'diagnostics-2col' });
  sections.recs = h('div', { class: 'section-title diagnostics-anchor' }, '四个问题的回答与建议');
  root.append(sections.recs, recGrid);
  const groups = [...QUESTIONS, ...new Set(recs.map(r => r.group).filter(g => !QUESTIONS.includes(g)))];
  for (const q of groups) {
    const rs = recs.filter(r => r.group === q);
    const gc = cands.filter(c => c.group === q);
    if (!rs.length && !gc.length) continue;
    const adopted = finChanges.filter(id => byId[id]?.group === q);
    const rob = gc.filter(c => robust(c.id)).map(c => c.id);
    const verdict = adopted.length ? ui.chip(`结论：采用 ${adopted.map(label).join('、')}`, 'good')
      : rob.length ? ui.chip(`结论：值得一试 ${rob.map(label).join('、')}`, 'warn') : ui.chip('结论：不需要改模型', 'accent');
    recGrid.append(card({
      cls: 'diagnostics-q', title: `${questions[q] || rs[0]?.question || q}？`,
      sub: `判断标准：候选改动单独加在基准上，在每个样本外十年都让 R² 提升超过 +${minGain} 才算稳定有效。`, actions: gc.length ? verdict : null },
    gc.length ? h('p', { class: 'diagnostics-verdict' },
      `本组试过 ${gc.length} 个改动（${gc.map(c => `${label(c.id)}${c.eligible === false ? '·只作对照' : ''}`).join('、')}）；每个十年都稳定提升的：${rob.length ? rob.map(label).join('、') : '无'}。`) : null,
    rs.length ? rs.map(r => h('div', { class: 'diagnostics-rec' },
      h('div', { class: 'diagnostics-rec-head' }, ui.chip(`${r.priority || '—'}优先级`, PRIORITY[r.priority] ?? ''), h('b', {}, r.title || '—')),
      r.detail ? h('p', {}, r.detail) : null,
      evidence(r.evidence))) : h('p', { class: 'muted' }, '这一组没有单独的建议。')));
  }

  // ---- walk-forward loop
  sections.loop = h('div', { class: 'section-title diagnostics-anchor' }, '走步迭代：诊断 → 修正 → 在新时段检验 → 再来一轮');
  root.append(sections.loop);
  if (!its.length) root.append(ui.notice('diagnostics.json 里没有走步迭代记录。', 'info'));
  else {
    root.append(card({ title: '每一轮用哪段历史训练、挑选和检验',
      sub: '每轮在训练期估计、在验证期挑选改动、在从未参与挑选的测试期检验；第 k 轮的测试期就是第 k+1 轮的验证期，最后一轮之后在全部数据上重新估计最终模型。' },
    timeline()));
    its.forEach((it, k) => root.append(...iterationBlock(it, k)));
    if (fin.reason) {
      root.append(h('div', { style: { height: '16px' } }), ui.notice(
        `最终决定：${fin.reason}${fin.test_used_for_choice === true ? '（测试期参与了这次决定，详见本页顶部的披露。）' : ''}`, 'info'));
    }
  }

  // ---- robustness across decades
  sections.robust = h('div', { class: 'section-title diagnostics-anchor' }, '候选改动在每个样本外十年的效果');
  root.append(sections.robust, card({ title: '单独加在基准模型上：样本外 R² 的变化（按问题分组）',
    sub: `每个数 = 只加这一个改动后样本外平均 R² 的变化（${periods.join('、') || '—'}，都只用之前的数据估计）；绿色为提升、红色为变差，三个十年都高于 +${minGain} 才算稳定有效。` },
  robustTable()));

  // ---- diagnostic tests
  sections.tests = h('div', { class: 'section-title diagnostics-anchor' }, '诊断检验：哪里出了问题');
  root.append(sections.tests);
  const pvBox = h('div', { class: 'chart' });
  root.append(card({ title: '诊断检验的 p 值（行 = 检验，列 = 资产）',
    sub: 'p < 0.05 即拒绝“没有问题”：Ljung-Box 查残差自相关，滞后 F 查漏掉的滞后传导，通胀状态 F 查高/低通胀下系数是否不同，Chow 查系数是否断裂，RESET 查非线性，漂移 t 查系统性偏差（正态近似）；同一行 3 个以上资产显著才算问题。' },
  ui.wideChart(pvBox, 560)));
  const wyOf = (it) => it.start?.spec?.window_years;
  const findingCards = its.map(it => card({ title: `第 ${it.iter} 轮诊断（训练 ${period(it.train)}，验证 ${period(it.validate)}）`,
    sub: `对本轮起点模型做的检验${isNum(wyOf(it)) ? `（起点模型只用最近 ${wyOf(it)} 年，即 ${fitPeriod(it)} 估计）` : ''}；“提示问题”= 至少 3 个资产显著，并据此建议要尝试的候选改动。` },
  (it.findings || []).length ? (it.findings || []).map(f => h('div', { class: 'diagnostics-finding' },
    h('div', { class: 'diagnostics-chips' }, ui.chip(f.flagged ? '提示问题' : '未发现问题', f.flagged ? 'warn' : 'good'),
      ui.chip(f.question || '—'), h('b', { style: { fontSize: '13px' } }, f.test || f.id || '—')),
    f.text ? h('p', {}, f.text) : null,
    f.suggests?.length ? h('p', { class: 'muted', style: { fontSize: '12px' } }, `建议尝试：${f.suggests.map(label).join('、')}`) : null))
    : h('p', { class: 'muted' }, '这一轮没有诊断记录。')));
  if (findingCards.length) root.append(h('div', { style: { height: '16px' } }), h('div', { class: 'diagnostics-2col' }, findingCards));
  const overfit = its.map(it => ({ it, box: ui.chartBox('tall') }));
  if (overfit.length) {
    root.append(h('div', { style: { height: '16px' } }), h('div', { class: 'diagnostics-2col' }, overfit.map(o => card({
      title: `第 ${o.it.iter} 轮：训练期 R² vs 验证期 R²`,
      sub: `本轮起点模型在训练期（样本内，${fitPeriod(o.it)}${isNum(wyOf(o.it)) ? `，只用最近 ${wyOf(o.it)} 年` : ''}）与验证期（样本外，${period(o.it.validate)}）的月度 R²；验证期远低于训练期，说明过拟合或传导关系变了。` }, o.box))));
  }
  const mf = its.map(it => card({ title: `第 ${it.iter} 轮：残差里还藏着哪些因子？`,
    sub: '数字 = 残差与该序列显著相关（|t| > 2）的资产数：训练期和验证期都对 3 个以上资产显著才说明真的缺了这个因子；“重复 R²”是它能被现有因子线性解释的部分，“最相关”是验证期相关最强的资产。' },
  missingFactorTable(it)));
  if (mf.length) root.append(h('div', { style: { height: '16px' } }), h('div', { class: 'diagnostics-2col' }, mf));

  // ---- rolling betas
  sections.betas = h('div', { class: 'section-title diagnostics-anchor' }, '传导系数随时代变化：滚动 β');
  root.append(sections.betas);
  const rb = ra.rolling_betas;
  const betaBox = ui.chartBox();
  const betaHeat = h('div', { class: 'chart' });
  let betaCard = null, heatCard = null;
  const bst = { f: null, a: 'us_equity' };
  if (rb && Array.isArray(rb.years) && rb.years.length && rb.assets) {
    const fs = (rb.factors || []).filter(f => Object.values(rb.assets).some(x => x?.beta?.[f]));
    const as = Object.keys(rb.assets);
    bst.f = fs.includes('y10') ? 'y10' : fs[0];
    bst.a = as.includes('us_equity') ? 'us_equity' : as[0];
    root.append(card({ cls: 'diagnostics-ctl' },
      h('div', { class: 'diagnostics-ctl-row' }, h('div', { class: 'diagnostics-ctl-label' }, '因子'),
        pills(h, fs.map(f => ({ value: f, label: ui.factorName(f) })), bst.f, (v) => { bst.f = v; renderBetas(); })),
      h('div', { class: 'diagnostics-ctl-row' }, h('div', { class: 'diagnostics-ctl-label' }, '资产'),
        pills(h, as.map(a => ({ value: a, label: short(a), title: nameOf(a) })), bst.a, (v) => { bst.a = v; renderBetas(); }))));
    betaCard = card({ title: ' ', sub: ' ' }, betaBox);
    heatCard = card({ title: ' ', sub: '红 = 因子上升 1 个标准差时资产当月上涨，蓝 = 下跌；某一年附近整列换色，就是传导机制随时代改变的证据（例如 2000 年前后股票对利率的反应换号，对应股债相关性由正转负）。' }, betaHeat);
    root.append(h('div', { style: { height: '16px' } }), betaCard, h('div', { style: { height: '16px' } }), heatCard);
  } else root.append(ui.notice('诊断结果里没有滚动 β。', 'info'));

  // ---- neural net
  sections.mlp = h('div', { class: 'section-title diagnostics-anchor' }, '小型神经网络：在哪里有用、在哪里有害');
  root.append(sections.mlp);
  const nnIds = cands.filter(c => c.group === 'neural_net').map(c => c.id);
  const nnBox = ui.chartBox();
  const nnAssetBox = ui.chartBox('tall');
  const nnIt = its.find(it => it.start?.is_base && (it.trials || []).some(t => t.round === 1 && nnIds.includes(t.candidate) && t.per_asset));
  const nnCards = [];
  if (nnIds.some(id => deltas[id]?.length)) {
    const ok = nnIds.filter(id => deltas[id]?.length)
      .map(id => `${label(id)} ${deltas[id].filter(d => d.delta > minGain).length}/${deltas[id].length} 个十年提升`).join('，');
    nnCards.push(card({ title: '神经网络替换线性模型后，各样本外十年的 R² 变化',
      sub: `只有每个十年都高于虚线（+${minGain}）才值得用：${ok}；不稳定的增益说明它在拟合某个时代的噪声，而且神经网络没有可解释的 β，情景的因子归因会失去依据。` }, nnBox));
  }
  if (nnIt) {
    const r1 = Object.fromEntries((nnIt.trials || []).filter(t => t.round === 1 && nnIds.includes(t.candidate)).map(t => [t.candidate, t]));
    const base = nnIt.baseline?.val_per_asset || {};
    const wins = nnIds.filter(id => r1[id]).map(id => `${label(id)} 在 ${scoreAssets.filter(a => isNum(r1[id].per_asset?.[a]) && isNum(base[a]) && r1[id].per_asset[a] > base[a]).length}/${nA} 个资产上更好（训练期 R² ${num(r1[id].train_r2, 3)}，线性 ${num(nnIt.baseline?.train_r2, 3)}）`).join('；');
    nnCards.push(card({ title: `第 ${nnIt.iter} 轮验证期（${period(nnIt.validate)}）各资产 R²：线性 vs 神经网络`,
      sub: `同一训练期（${period(nnIt.train)}）、同一组因子，只换模型形式：${wins}；训练期更高而验证期更低就是过拟合。` }, nnAssetBox));
  }
  if (nnCards.length) root.append(h('div', { class: 'diagnostics-2col' }, nnCards));
  else root.append(ui.notice('诊断结果里没有神经网络候选的记录。', 'info'));

  // ---- engine ablation (optional file)
  sections.ablation = h('div', { class: 'section-title diagnostics-anchor' }, '引擎消融：误差来自引擎的哪一部分');
  const abBox = h('div', {}, ui.loading('读取引擎消融结果…'));
  root.append(sections.ablation, abBox);

  // ---- notes
  const units = dg.units || {};
  const UNIT = { score: '打分', bias_annual: '漂移偏差', beta_per_sd: 'β', p: 'p 值' };
  root.append(h('div', { class: 'section-title' }, '口径与说明'), h('div', { class: 'grid cols-2 text' },
    card({ title: '口径' }, h('dl', { class: 'kv' }, Object.entries(units).flatMap(([k, v]) => [h('dt', {}, UNIT[k] || k), h('dd', {}, v)]))),
    card({ title: '说明' }, h('ul', { class: 'list' }, [
      `采纳门槛：验证期平均 R² 至少提高 ${minGain}；每轮最多采纳 ${dg.max_accept ?? '—'} 个改动；${dg.require_test_gain ? '最终模型必须在测试期胜过基准。' : '测试期只用于报告。'}`,
      ...(dg.notes || [])].map(n => h('li', { class: 'muted' }, n))))));

  // ================================================================ charts (after the DOM exists)
  for (const e of itCharts) (e.rounds ? renderRounds(e.it, e.rounds) : renderTest(e.it, e.test));
  renderTests();
  overfit.forEach(o => {
    const pa = (o.it.findings || []).find(f => f.id === 'overfit')?.per_asset || {};
    const as = scoreAssets.filter(a => pa[a]);
    if (!as.length) { clearChart(o.box, ui.notice('这一轮没有过拟合检验。', 'info')); return; }
    charts.bars(o.box, { x: as.map(short), horizontal: true, fmt: (v) => num(v, 2),
      series: [{ name: '训练期 R²（样本内）', data: as.map(a => pa[a].train_r2 ?? null), color: '#3b5bdb' },
        { name: '验证期 R²（样本外）', data: as.map(a => pa[a].validate_r2 ?? null), color: '#e8590c' }] });
  });
  if (betaCard) renderBetas();
  if (nnCards.length) renderNN();
  // scenarios.json only names the scenarios the ablation text mentions by id
  Promise.all([ablation, api.get('scenarios').catch(() => null)]).then(([res, sc]) => { if (alive()) renderAblation(res, sc); });
  if (param && sections[param]) setTimeout(() => sections[param].scrollIntoView({ block: 'start' }), 80);

  // ================================================================ builders
  function evidence(ev) {
    const items = Object.entries(ev || {}).filter(([, v]) => v !== null && v !== undefined);
    if (!items.length) return null;
    return h('details', { class: 'diagnostics-evi', open: true }, h('summary', {}, '证据'),
      h('dl', {}, items.flatMap(([k, v]) => [h('dt', {}, EVI[k] || k), h('dd', {}, eviValue(k, v))])));
  }

  // An unnamed list of per-decade deltas: name it after the candidate whose own deltas it repeats.
  function matchCand(ds) {
    return cands.find(c => (deltas[c.id] || []).length === ds.length && deltas[c.id].every((d, i) => d.period === ds[i]?.period
      && isNum(d.delta) && isNum(ds[i]?.delta) && Math.abs(d.delta - ds[i].delta) < 1e-6))?.id || '';
  }

  function eviValue(k, v) {
    if (k === 'oos_delta') return deltaTable(Array.isArray(v) ? { [matchCand(v)]: v } : v);
    if (k === 'base_oos_r2' && Array.isArray(v)) return v.map(r => `${r.period} ${r2s(r.r2)}`).join('、');
    if (k === 'last_selected' && v && typeof v === 'object') {
      return `${v.describe || '—'}：验证期 R² ${r2s(v.val_score)}，测试期 ${r2s(v.test_score)}；改动 ${(v.changes || []).map(label).join(' → ') || '无'}`;
    }
    if (k === 'leakage' && v && typeof v === 'object') {
      return ui.table([{ key: 'f', label: '序列', fmt: (x) => ui.factorName(x) },
        { key: 'same_month', label: '当月', fmt: (x) => num(x) }, { key: 'next_month', label: '下月', fmt: (x) => num(x) },
        { key: 'prev_month', label: '上月', fmt: (x) => num(x) }], Object.entries(v).map(([f, x]) => ({ f, ...x })));
    }
    if (k === 'chow_reject' && Array.isArray(v)) return `${v.length}/${nA}：${v.map(short).join('、') || '无'}`;
    if ((k === 'robust' || k === 'changes') && Array.isArray(v)) return v.length ? v.map(label).join('、') : '无';
    if ((k === 'reset_reject' || k === 'n_reject') && Array.isArray(v)) return v.map((n, i) => `第 ${i + 1} 轮 ${n}/${nA}`).join('、');
    if (k === 'bias' && v && typeof v === 'object') {
      return Object.entries(v).map(([i, b]) => `第 ${i} 轮 ${nameOf(b?.asset)} ${pct(b?.bias_annual, 1, true)}/年`).join('；');
    }
    if (isNum(v)) return k.includes('beta') ? pct(v, 2, true) : k.includes('acf') ? num(v, 2) : k.startsWith('r2') ? pct(v, 0) : num(v, 3);
    if (typeof v === 'string') return v;
    if (Array.isArray(v)) return v.map(x => (isNum(x) ? num(x, 3) : typeof x === 'object' ? JSON.stringify(x) : String(x))).join('、');
    return Object.entries(v).map(([a, b]) => `${a}: ${isNum(b) ? num(b, 3) : typeof b === 'object' ? JSON.stringify(b) : b}`).join('；');
  }

  function deltaTable(map) {
    const rows = Object.entries(map || {}).filter(([, ds]) => Array.isArray(ds));
    const ps = [...new Set(rows.flatMap(([, ds]) => ds.map(d => d.period)))];
    const cls = (x) => (isNum(x) ? (x > minGain ? 'is-good' : x < -minGain ? 'is-bad' : 'muted') : '');
    return ui.table([{ key: 'name', label: '改动' },
      ...ps.map(p => ({ key: p, label: p, fmt: (x) => r2s(x), cls }))],
    rows.map(([id, ds]) => ({ name: id ? label(id) : '该改动', ...Object.fromEntries(ds.map(d => [d.period, d.delta])) })));
  }

  function timeline() {
    const starts = its.map(it => it.train[0]);
    const ends = its.map(it => it.test[1]).concat(fin.trained_on?.[1] || []);
    const t0 = Math.min(...starts.map(ymNum)), t1 = Math.max(...ends.map(ymNum)) + 1;
    const pos = (ym) => ((ymNum(ym) - t0) / (t1 - t0)) * 100;
    const seg = (kind, p, text) => {
      const left = pos(p[0]), width = pos(p[1]) - left + 100 / (t1 - t0);
      const s = SPAN[kind] || { bg: '#e9ecef', ink: INK, label: '' };
      return h('div', { class: 'diagnostics-tl-seg', title: `${s.label} ${p[0]} 至 ${p[1]}`,
        style: { left: `${left}%`, width: `${width}%`, background: s.bg, color: s.ink } }, text ?? (width > 14 ? `${s.label} ${period(p)}` : period(p)));
    };
    const rows = its.map(it => h('div', { class: 'diagnostics-tl-row' },
      h('div', { class: 'diagnostics-tl-label' }, `第 ${it.iter} 轮`),
      h('div', { class: 'diagnostics-tl-track' }, seg('train', it.train), seg('validate', it.validate), seg('test', it.test))));
    if (Array.isArray(fin.trained_on)) {
      rows.push(h('div', { class: 'diagnostics-tl-row' }, h('div', { class: 'diagnostics-tl-label' }, '最终模型'),
        h('div', { class: 'diagnostics-tl-track' }, seg('final', fin.trained_on, `全部数据 ${period(fin.trained_on)} 重新估计：${fin.describe || ''}`))));
    }
    const y0 = Math.ceil(t0 / 120) * 10;
    const ticks = [];
    for (let y = y0; y * 12 < t1; y += 10) ticks.push(h('span', { style: { left: `${((y * 12 - t0) / (t1 - t0)) * 100}%` } }, String(y)));
    return h('div', { class: 'diagnostics-tl' },
      h('div', { class: 'diagnostics-legend' }, Object.values(SPAN).map(s => h('span', {}, h('i', { style: { background: s.bg, border: `1px solid ${s.ink}` } }), `${s.label}期`))),
      ...rows, h('div', { class: 'diagnostics-tl-axis' }, ticks));
  }

  function iterationBlock(it, k) {
    const trials = (it.trials || []).filter(t => isNum(t.delta));
    const rounds = [...new Set(trials.map(t => t.round))].filter(isNum).sort((a, b) => a - b);
    const path = it.selected?.path || [];
    const errors = (it.trials || []).filter(t => t.error);
    const wy = it.start?.spec?.window_years;
    const head = h('div', { class: 'section-title' }, `第 ${it.iter} 轮：训练 ${period(it.train)} → 在 ${period(it.validate)} 挑选 → 在 ${period(it.test)} 检验`);
    const boxes = rounds.map(r => ({ r, box: h('div', { class: 'chart' }) }));
    itCharts.push({ it, rounds: boxes });
    const acceptedIn = (r) => trials.find(t => t.round === r && t.accepted);
    const startOf = (r) => path.find(p => p.round === r - 1);
    const pick = card({
      title: `逐步挑选：每一步把剩下的候选逐个加到当前模型上（验证期 ${period(it.validate)}）`,
      sub: `条形 = 加入该改动后验证期平均 R² 的变化，超过虚线（+${minGain}）且最大的被采纳、下一步从它出发；浅色条为内生候选（只作对照），灰色行名表示这一步没有尝试，超出坐标范围的条在末端标出真实值。` },
    h('div', { class: 'diagnostics-legend' },
      QUESTIONS.map(q => h('span', {}, h('i', { style: { background: GROUP_COLOR[q] } }), `${questions[q] || GROUP_LABEL[q]}`)),
      h('span', {}, h('s'), `采纳门槛 +${minGain}`)),
    rounds.length ? h('div', { class: 'grid cols-3' }, boxes.map(({ r, box }) => {
      const st = startOf(r), acc = acceptedIn(r);
      return h('div', { class: 'diagnostics-round' }, h('h4', {}, `第 ${r} 步`),
        h('div', { class: 'muted' }, `起点 R² ${r2s(st?.val_score)}${acc ? `，采纳「${label(acc.candidate)}」` : '，没有改动超过门槛，停止'}`), box);
    })) : ui.notice('这一轮没有可显示的尝试。', 'info'),
    h('div', { class: 'diagnostics-path' }, path.flatMap((p, i) => [i ? h('span', { class: 'diagnostics-arrow' }, '→') : null,
      ui.chip(h('span', {}, i === 0 ? `起点：${p.describe || '—'}` : plus(p.change), h('b', {}, r2s(p.val_score))), i === path.length - 1 && i > 0 ? 'good' : '')])),
    wy ? h('p', { class: 'muted', style: { fontSize: '12px', margin: '8px 0 0' } }, `本轮起点模型只用最近 ${wy} 年估计（承接上一轮的选择），实际训练窗口短于 ${period(it.train)}。`) : null,
    errors.length ? h('p', { class: 'muted', style: { fontSize: '12px', margin: '6px 0 0' } },
      `无法估计：${errors.map(t => `第 ${t.round} 步「${label(t.candidate)}」（${/not enough months/.test(t.error) ? '样本月份不足' : t.error}）`).join('；')}。`) : null);

    const tr = it.test_result || {};
    const assetBox = ui.chartBox('tall');
    itCharts.push({ it, test: assetBox });
    // The data's carry-over note always talks about "the next round"; after the last round the same gate picks
    // the final model instead, so say that.
    const last = k === its.length - 1;
    const carry = !last ? it.carry_over && `↓ 下一轮：${it.carry_over}`
      : typeof fin.is_base === 'boolean' ? `→ 最终：${tr.improved ? '选中模型在测试期胜过基准' : '选中模型没有在测试期胜过基准'}，`
        + `${fin.is_base ? '最终保留基准模型' : `最终采用「${fin.describe || '选中模型'}」`}，并在全部数据上重新估计（见下方“最终决定”）。`
        : it.carry_over && `→ ${it.carry_over}`;
    const test = card({
      title: `未参与挑选的测试期 ${period(it.test)}：基准 vs 本轮选中`,
      sub: '把训练期和验证期合并重新估计，再用测试期的真实因子逐月预测；改进只有在这里也成立才算数，并决定下一轮从哪个模型出发。' },
    h('div', { class: 'diagnostics-split' },
      h('div', {},
        h('div', { class: 'diagnostics-kpis' },
          tile('基准模型', r2s(tr.base?.score), `${it.baseline?.describe || '—'}，测试期 ${tr.n_months ?? '—'} 个月平均 R²`),
          tile('本轮选中', r2s(tr.selected?.score), it.selected?.describe || '—'),
          tr.start ? tile('本轮起点', r2s(tr.start.score), it.start?.describe || '—') : null,
          tile('选中 − 基准', r2s(tr.delta), tr.improved ? '改进经受住了样本外检验' : '改进没有延续到测试期', ui.goodCls(!!tr.improved))),
        tr.text ? h('p', { class: 'diagnostics-text' }, tr.text) : null,
        carry ? h('div', { class: 'diagnostics-carry' }, ui.notice(carry, tr.improved ? 'info' : '')) : null),
      h('div', {}, h('div', { class: 'muted', style: { fontSize: '12px', marginBottom: '4px' } }, '各资产测试期 R²（同一批月份）'), assetBox)));
    return [head, pick, h('div', { style: { height: '16px' } }), test];
  }

  // Shared rows and a shared, clipped value range so the steps of one iteration line up.
  function renderRounds(it, boxes) {
    const trials = (it.trials || []).filter(t => isNum(t.delta));
    const rows = [];
    for (const t of trials) if (!rows.includes(t.candidate)) rows.push(t.candidate);
    const elig = trials.filter(t => t.eligible !== false).map(t => t.delta);
    const raw = Math.max(Math.max(minGain * 2, ...elig) * 1.15, 0.005);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const hi = [1, 2, 2.5, 5, 10].map(m => m * mag).find(v => v >= raw);      // a round upper bound
    const lo = -2 * hi;
    for (const { r, box: el } of boxes) {
      el.style.height = `${rows.length * 21 + 34}px`;
      roundChart(charts, el, { rows, trials, round: r, lo, hi, minGain, label });
    }
  }

  function renderTest(it, el) {
    const tr = it.test_result || {};
    const ser = [['base', '基准模型', '#868e96'], ['selected', '本轮选中', '#3b5bdb'], ['start', '本轮起点', '#e8590c']]
      .filter(([key]) => tr[key]?.per_asset);
    if (!ser.length) { clearChart(el, ui.notice('没有分资产的测试结果。', 'info')); return; }
    charts.bars(el, { x: scoreAssets.map(short), horizontal: true, fmt: (v) => num(v, 2),
      series: ser.map(([key, name, color]) => ({ name, color, data: scoreAssets.map(a => tr[key].per_asset[a] ?? null) })) });
  }

  function robustTable() {
    if (!cands.length || !periods.length) return ui.notice('没有候选改动的样本外记录。', 'info');
    const accepted = (id) => its.filter(it => (it.selected?.changes || []).includes(id)).map(it => `第 ${it.iter} 轮`);
    const cell = (v) => {
      if (!isNum(v)) return h('td', { class: 'c' }, '—');
      const a = Math.min(Math.abs(v) / 0.05, 1) * 0.55 + 0.08;
      const bg = Math.abs(v) <= minGain ? '#f8f9fa' : v > 0 ? `rgba(47,158,68,${a.toFixed(2)})` : `rgba(224,49,49,${a.toFixed(2)})`;
      return h('td', { class: 'c', style: { background: bg, color: INK } }, r2s(v));
    };
    const body = [];
    for (const q of [...QUESTIONS, ...new Set(cands.map(c => c.group).filter(g => !QUESTIONS.includes(g)))]) {
      const gc = cands.filter(c => c.group === q);
      if (!gc.length) continue;
      body.push(h('tr', { class: 'g' }, h('td', { colspan: String(periods.length + 5) }, `${questions[q] || q}？`)));
      for (const c of gc) {
        const ds = Object.fromEntries((deltas[c.id] || []).map(d => [d.period, d.delta]));
        const acc = accepted(c.id);
        body.push(h('tr', {},
          h('td', { title: c.note || '' }, label(c.id)),
          h('td', { class: 'h' }, c.hypothesis || '—', c.eligible === false ? ui.chip('内生，只作对照', 'warn') : null),
          periods.map(p => cell(ds[p])),
          h('td', { class: 'c' }, c.eligible === false ? '—' : robust(c.id) ? ui.chip('是', 'good') : ui.chip('否')),
          h('td', { class: 'c' }, acc.length ? acc.join('、') : '—'),
          h('td', { class: 'c' }, finChanges.includes(c.id) ? ui.chip('采用', 'good') : '—')));
      }
    }
    return h('div', { class: 'table-wrap' }, h('table', { class: 'tbl diagnostics-heat' },
      h('thead', {}, h('tr', {}, h('th', {}, '改动'), h('th', { class: 'wrap' }, '假设'), periods.map(p => h('th', { class: 'c' }, p)),
        h('th', { class: 'c' }, '每个十年都提升'), h('th', { class: 'c' }, '走步中被采纳'), h('th', { class: 'c' }, '最终采用'))),
      h('tbody', {}, body)));
  }

  function missingFactorTable(it) {
    const f = (it.findings || []).find(x => x.id === 'missing_factor');
    if (!f?.table?.length) return h('p', { class: 'muted' }, '这一轮没有候选因子的残差相关检验。');
    return ui.table([
      { key: 'name', label: '候选序列', wrap: true, minWidth: 150, fmt: (v, r) => h('span', {}, v || r.column, r.eligible === false ? ui.chip('内生', 'warn') : null) },
      { key: 'n_sig_train', label: '训练期', fmt: (v) => `${v ?? '—'}/${nA}` },
      { key: 'n_sig_validate', label: '验证期', fmt: (v) => `${v ?? '—'}/${nA}` },
      { key: 'n_sig_both', label: '两期都', fmt: (v) => `${v ?? '—'}/${nA}`, cls: (v) => (v >= 3 ? 'is-bad' : '') },
      { key: 'spanned_r2', label: '重复 R²', fmt: (v) => pct(v, 0) },
      { key: 'best', label: '最相关', fmt: (v) => (v?.asset ? `${short(v.asset)} ${num(v.corr, 2, true)}` : '—') },
    ], f.table);
  }

  function renderTests() {
    const rows = [];
    for (const it of its) {
      const F = Object.fromEntries((it.findings || []).map(f => [f.id, f]));
      for (const t of TESTS) {
        const pa = F[t.finding]?.per_asset;
        if (!pa) continue;
        const values = {};
        for (const [a, x] of Object.entries(pa)) { const p = x ? t.p(x) : null; if (isNum(p)) values[a] = p; }
        if (Object.keys(values).length) rows.push({ label: `第 ${it.iter} 轮 · ${t.label}`, values });
      }
    }
    const RA = ra.assets || {};
    const lb = {}, ch = {};
    for (const [a, x] of Object.entries(RA)) { if (isNum(x?.ljung_box_p)) lb[a] = x.ljung_box_p; if (isNum(x?.chow_p)) ch[a] = x.chow_p; }
    if (Object.keys(lb).length) rows.push({ label: '全样本 · 残差自相关 Ljung-Box', values: lb });
    if (Object.keys(ch).length) rows.push({ label: `全样本 · ${String(ra.break_date || '').slice(0, 4)} 年前后断点 Chow`, values: ch });
    const cols = [...scoreAssets, ...Object.keys(RA).filter(a => !scoreAssets.includes(a))].filter(a => rows.some(r => isNum(r.values[a])));
    if (!rows.length || !cols.length) { clearChart(pvBox, ui.notice('没有诊断检验结果。', 'info')); return; }
    pvalChart(charts, pvBox, rows, cols, cols.map(short));
  }

  function renderBetas() {
    const f = bst.f, a = bst.a;
    const A = rb.assets?.[a];
    const beta = A?.beta?.[f], se = A?.se?.[f] || [];
    betaCard.querySelector('.card-title').textContent = `${nameOf(a)} 对「${ui.factorName(f)}」的滚动 β（${rb.window_months || 120} 个月窗口）`;
    const sd = rb.factor_sd?.[f], unit = ui.META.factors[f]?.unit;
    const sdTxt = !isNum(sd) ? '—' : unit === 'Δpp' ? `${num(sd, 2)} 个百分点` : unit === 'Δlog' ? pct(sd, 2) : num(sd, 4);
    betaCard.querySelector('.card-sub').textContent = `每个点用截至该年年底的过去 ${rb.window_months || 120} 个月估计，表示因子变动 1 个标准差（月度标准差 ${sdTxt}）时资产当月的价格收益；阴影为 ±2 倍标准误，跨过 0 线就是传导方向翻转。`;
    if (!Array.isArray(beta)) clearChart(betaBox, ui.notice('没有这个组合的滚动 β。', 'info'));
    else betaChart(charts, betaBox, { years: rb.years, beta, se, color: '#3b5bdb' });
    heatCard.querySelector('.card-title').textContent = `所有资产对「${ui.factorName(f)}」的滚动 β（年份 × 资产）`;
    const as = Object.keys(rb.assets).filter(x => Array.isArray(rb.assets[x]?.beta?.[f]));
    if (!as.length) { clearChart(betaHeat, ui.notice('没有滚动 β。', 'info')); return; }
    const values = as.map(x => rb.assets[x].beta[f].map(v => (isNum(v) ? v : null)));
    // Colour scale from the factor-model assets only: crude oil's beta to oil and the treasury's beta to the
    // 10-year yield are mechanical and several times larger, and would wash every other row out.
    const scaled = as.map((x, i) => (scoreAssets.length && !scoreAssets.includes(x) ? [] : values[i]));
    const abs = (scaled.flat().some(isNum) ? scaled : values).flat().filter(isNum).map(Math.abs).sort((p, q) => p - q);
    const bound = abs.length ? abs[Math.floor(abs.length * 0.95)] || abs[abs.length - 1] || 1 : 1;
    const capped = as.filter((x, i) => values[i].some(v => isNum(v) && Math.abs(v) > bound * 1.5) && !scoreAssets.includes(x));
    heatCard.querySelector('.card-sub').textContent = '红 = 因子上升 1 个标准差时资产当月上涨，蓝 = 下跌；某一年附近整列换色，就是传导机制随时代改变的证据'
      + '（例如 2000 年前后股票对利率的反应换号，对应股债相关性由正转负）。'
      + (capped.length ? `${capped.map(nameOf).join('、')}对这个因子的反应是机械关系、数值大得多，颜色已封顶（悬停看数值）。` : '');
    const hm = charts.heatmap(fresh(betaHeat), { rows: as, cols: rb.years, values, rowLabels: as.map(nameOf), colLabels: rb.years,
      min: -bound, max: bound, fmt: (v) => pct(v, 2, true), height: as.length * 24 + 100 });
    const plotW = Math.max(100, (betaHeat.clientWidth || 800) - 150);          // the helper reserves ~150px for labels
    hm.setOption({ xAxis: { axisLabel: { interval: Math.max(0, Math.ceil((rb.years.length * 22) / plotW) - 1) } } });
  }

  function renderNN() {
    const ids = nnIds.filter(id => deltas[id]?.length);
    if (ids.length) {
      const ps = deltas[ids[0]].map(d => d.period);
      charts.bars(nnBox, { x: ps, fmt: (v) => num(v, 3, true),
        markLine: [{ yAxis: minGain, label: { formatter: `采纳门槛 +${minGain}`, position: 'insideStartTop', color: INK3, fontSize: 10 } }],
        series: ids.map((id, i) => ({ name: label(id), color: ['#c2255c', '#f783ac', '#a61e4d'][i % 3],
          data: ps.map(p => deltas[id].find(d => d.period === p)?.delta ?? null) })) });
    }
    if (nnIt) {
      const r1 = Object.fromEntries((nnIt.trials || []).filter(t => t.round === 1 && nnIds.includes(t.candidate)).map(t => [t.candidate, t]));
      const base = nnIt.baseline?.val_per_asset || {};
      charts.bars(nnAssetBox, { x: scoreAssets.map(short), horizontal: true, fmt: (v) => num(v, 2),
        series: [{ name: '线性（基准）', color: '#868e96', data: scoreAssets.map(a => base[a] ?? null) },
          ...nnIds.filter(id => r1[id]).map((id, i) => ({ name: label(id), color: ['#c2255c', '#f783ac'][i % 2],
            data: scoreAssets.map(a => r1[id].per_asset?.[a] ?? null) }))] });
    }
  }

  // ---- engine ablation
  function renderAblation(res, scen = null) {
    abBox.replaceChildren();
    if (res.e) {
      abBox.append(res.e.status === 404
        ? ui.notice(/python -m/.test(String(res.e.message)) ? `引擎消融结果尚未生成：${res.e.message}`
          : '引擎消融结果尚未生成：在项目根目录运行 python -m scenariolab.build --only ablation（约 6 分钟，默认构建不包含这一步）。', 'info')
        : ui.notice(`读取引擎消融结果失败：${res.e.message || res.e}`, 'error'));
      return;
    }
    const ab = res.d && typeof res.d === 'object' && !Array.isArray(res.d) ? res.d : {};
    const V = (Array.isArray(ab.variants) ? ab.variants : []).filter(v => v && typeof v === 'object' && (v.id || v.name));
    let W = (Array.isArray(ab.windows) ? ab.windows : []).filter(w => w && w.id);
    if (!W.length && V.length) W = [...new Set(V.flatMap(v => Object.keys(v.windows || {})))].map(id => ({ id, name: id }));
    const H = Array.isArray(ab.hypotheses) ? ab.hypotheses.filter(x => x && typeof x === 'object') : [];
    const sel = ab.selection && typeof ab.selection === 'object' && !Array.isArray(ab.selection) ? ab.selection : null;
    const set = ab.settings && typeof ab.settings === 'object' ? ab.settings : {};
    const base = V.find(v => v.id === 'baseline') || null;
    const hyp = Object.fromEntries(H.filter(x => x.id).map(x => [x.id, x]));
    // the variant the engine runs now (the ablation's adopted choice); the baseline is then the old engine
    const engineId = [sel?.engine_variant, set.engine_variant].find(x => typeof x === 'string' && x) || null;
    const engineV = engineId ? V.find(v => v.id === engineId) || null : null;
    const switched = !!engineId && engineId !== 'baseline' && engineV !== base;
    const c = {
      W, V, sel, base, hyp, engineId, engineV, switched,
      baseName: switched ? '基准引擎（采用前）' : '现行引擎',
      wLabel: (w) => `${w.name || w.id}${ROLE[w.role] ? `（${ROLE[w.role]}）` : ''}`,
      vName: (id) => V.find(v => v.id === id)?.name || id || '—',
      color: (v) => (v === base ? '#495057' : HYP_COLOR[v.hypothesis] || '#868e96'),
      hypLabel: (id) => (id ? `${id}${hyp[id]?.name ? ` ${hyp[id].name}` : ''}` : '其他'),
      human: humanIds(V, scen),
    };
    const blocks = [];
    const gap = () => h('div', { style: { height: '16px' } });
    const nSeeds = Array.isArray(set.seeds) ? set.seeds.length : null;
    const reproduces = typeof set.engine_reproduces_backtest === 'boolean' ? set.engine_reproduces_backtest : set.baseline_reproduces_backtest;

    blocks.push(card({ title: typeof ab.question === 'string' && ab.question ? ab.question : '引擎消融：误差来自引擎的哪一部分？',
      sub: '在同一批回测窗口上逐个替换引擎的组成部分（残差怎么跟着抽中的月份走、阶段长度、视图集合、CMA、最少有效月份），看已知宏观时的年化误差和区间覆盖率怎么变；只在验证窗口上挑选，测试窗口只确认一次。' },
    typeof ab.text === 'string' && ab.text ? h('p', { class: 'diagnostics-text', style: { marginTop: 0 } }, c.human(ab.text)) : h('p', { class: 'muted' }, '没有文字说明。'),
    nSeeds ? h('p', { class: 'muted', style: { fontSize: '12px', margin: '8px 0 0' } },
      `每个数字是 ${nSeeds} 个随机种子的平均${isNum(set.n_paths) ? `（每个种子 ${set.n_paths} 条路径）` : ''}，各方案用同一批种子，差别只来自改动本身`
      + `${reproduces === true ? `；现行引擎${switched ? `「${c.vName(engineId)}」` : ''}的结果与回测页一致` : reproduces === false ? '；注意：现行引擎的结果与回测页不完全一致（可能是回测或消融其中一个还没有重新生成）' : ''}。`) : null,
    switched ? h('div', { style: { marginTop: '10px' } }, ui.notice(
      `引擎已经改用消融选出的「${c.vName(engineId)}」（${engineId}）：回测、未来情景和一年期预测都在用它。下面的“基准引擎（采用前）”是原来的设置，各方案的改善都相对它计算。`, 'info')) : null,
    W.length ? h('div', { class: 'diagnostics-chips', style: { marginTop: '10px' } },
      W.map(w => ui.chip(`${c.wLabel(w)} ${Array.isArray(w.test) ? `${w.test[0] || ''} 至 ${w.test[1] || ''}` : ''}`, w.role === 'test' ? 'good' : 'accent'))) : null));

    if (H.length) {
      blocks.push(gap(), card({ title: '假设与检验',
        sub: '每个假设对应一组改动或对照实验：“支持”= 证据站得住，“否定”= 改动没有带来改善，“不确定”= 窗口之间结论不一致；点开“怎么检验与证据”看具体数字。' },
      ui.table([
        { key: 'name', label: '假设', cls: 'diagnostics-nowrap', fmt: (v, r) => `${r.id ? `${r.id} ` : ''}${typeof v === 'string' ? v : ''}`.trim() || '—' },
        { key: 'verdict', label: '结论', cls: 'diagnostics-top', fmt: (v) => ui.chip(...(VERDICT[v] || [v ? String(v) : '—', ''])) },
        { key: 'result', label: '结果', wrap: true, fmt: (v, r) => h('div', { class: 'diagnostics-hyp' }, h('div', {}, c.human(show(v))),
          r.test || r.evidence ? h('details', { class: 'diagnostics-evi' }, h('summary', {}, '怎么检验与证据'),
            r.test ? h('p', {}, h('b', {}, '怎么检验：'), c.human(show(r.test))) : null,
            r.evidence ? h('p', {}, h('b', {}, '证据：'), c.human(show(r.evidence))) : null) : null) },
      ], H)));
    }

    const maeBox = h('div', { class: 'chart' }), covBox = h('div', { class: 'chart' });
    if (V.length && W.length) {
      const hypIds = [...new Set(V.filter(v => v !== base).map(v => v.hypothesis).filter(Boolean))];
      const picked = sel?.chosen && sel.chosen !== base?.id && V.some(v => v.id === sel.chosen);
      blocks.push(gap(),
        card({ title: '各方案在每个回测窗口的已知宏观年化误差（风险资产 MAE）',
          sub: `每个小图是一个回测窗口，每行一个方案，柱子越短越好：短于灰色实线（${c.baseName}）说明这个改动有帮助，短于黑色虚线（${c.baseName}不知道宏观时的误差）才说明“知道宏观”终于有用；鼠标悬停可看覆盖率和误差拆解。` },
        h('div', { class: 'diagnostics-legend' },
          base ? h('span', {}, h('i', { style: { background: c.color(base) } }), c.baseName) : null,
          hypIds.map(id => h('span', {}, h('i', { style: { background: HYP_COLOR[id] || '#868e96' } }), c.hypLabel(id))),
          h('span', {}, h('s', { style: { borderTop: `1.5px dashed ${INK}` } }), base ? `不知道宏观（${c.baseName}）` : '不知道宏观（各方案平均）'),
          base ? h('span', {}, h('s', { style: { borderTop: '1.5px solid #868e96' } }), `${c.baseName}·已知宏观`) : null,
          picked ? h('span', { style: { color: '#2b8a3e', fontWeight: 600 } },
            `★ 按规则选中${sel.chosen === engineId ? '，引擎已采用' : ''}：${c.vName(sel.chosen)}`) : null),
        maeBox),
        gap(),
        card({ title: '各方案的 90% 区间覆盖率（风险资产）',
          sub: `实际路径落在模拟 5%–95% 区间内的月份占比，越接近虚线 90% 越好：太低说明区间太窄或中枢偏了，太高说明区间过宽；选择规则要求覆盖率不能比${switched ? '基准引擎' : '现行引擎'}低 5 个百分点以上。` }, covBox),
        gap(),
        card({ title: '方案明细', sub: `按假设分组；“验证 / 测试 MAE”是对应角色窗口的平均，下面的小字是相对${c.baseName}的改善（± 种子间标准误）；“15 年分布宽度”是从今天出发、不加视图时各风险资产年化收益 5%–95% 区间相对${c.baseName}的宽度，取最窄的那个资产（收窄超过 5% 不能采用：回测窗口太短，看不出长期离散度）；选择规则只看验证窗口，测试窗口只用来确认一次。` },
          abVariantTable(c)));
    } else if (V.length) {
      blocks.push(gap(), ui.notice('消融结果里没有窗口信息，无法画分窗口的对比。', 'info'));
    }

    if (sel) {
      const elig = Array.isArray(sel.eligible) ? sel.eligible.map(c.vName).join('、') : null;
      blocks.push(gap(), h('div', { class: 'diagnostics-select' },
        h('div', { class: 'diagnostics-chips', style: { marginBottom: '8px' } }, h('b', {}, '选择结果'),
          sel.adopt === true ? ui.chip(`采用「${c.vName(sel.chosen)}」`, 'good')
            : sel.adopt === false ? ui.chip(`不采用，保留${switched ? '基准引擎' : '现行引擎'}`, 'warn') : ui.chip('未说明是否采用'),
          engineId ? ui.chip(switched ? `引擎现行设置：${c.vName(engineId)}` : '引擎现行设置：基准引擎', 'accent') : null),
        h('dl', { class: 'kv' },
          h('dt', {}, '选择规则'), h('dd', {}, c.human(show(sel.rule))),
          h('dt', {}, '选中方案'), h('dd', {}, c.vName(sel.chosen)),
          engineId ? [h('dt', {}, '引擎现在用的'), h('dd', {}, `${c.vName(engineId)}（${engineId}）`,
            switched ? '：回测、未来情景和一年期预测都已改用它' : '')] : null,
          elig !== null ? [h('dt', {}, '满足条件的候选'), h('dd', {}, elig || '无')] : null,
          h('dt', {}, '理由'), h('dd', {}, c.human(show(sel.reason))))));
    }

    const tg = ab.transmission_gap && typeof ab.transmission_gap === 'object' ? ab.transmission_gap : {};
    const TG = tg.windows && typeof tg.windows === 'object' && !Array.isArray(tg.windows) ? tg.windows : null;
    if (TG && Object.keys(TG).length) {
      const ids = Object.keys(TG);
      const winById = Object.fromEntries(W.map(w => [w.id, w]));
      const keys = [...new Set(ids.flatMap(id => Object.values(TG[id] || {})
        .flatMap(x => (x && typeof x.explained_by === 'object' && x.explained_by ? Object.keys(x.explained_by) : []))))];
      const tgBox = h('div');
      const text = typeof tg.text === 'string' ? c.human(tg.text.replace(/\bexplained_by\b/g, '各解释列').replace(/\bunexplained\b/g, '“仍未解释”')
        .replace(/\ball\b/g, '“三者合计”').replace(/\bf_valuation\b/g, '估值变化')) : '';
      blocks.push(gap(), card({ title: '传导缺口：oracle 与实际差在哪里',
        sub: 'oracle = 训练期因子模型吃测试期真实宏观的结果；传导缺口 =（ln 实际 − ln oracle）/ 年数，正数 = 因子模型漏掉的上涨。右侧各列 = 给训练期因子模型加入该候选因子后 oracle 每年上移多少（与缺口同单位）'
          + `${keys.includes('f_valuation') ? '；估值变化几乎就是股价本身（内生），只用来诊断' : ''}${keys.includes('unexplained') ? '；“仍未解释” = 缺口 − 三者合计' : ''}。` },
      h('div', { style: { marginBottom: '10px' } }, pills(h, ids.map(id => ({ value: id, label: winById[id] ? c.wLabel(winById[id]) : id })), ids[0],
        (id) => fillGap(tgBox, TG[id], keys))),
      tgBox,
      text ? h('p', { class: 'diagnostics-text muted', style: { fontSize: '12px' } }, text) : null));
      fillGap(tgBox, TG[ids[0]], keys);
    }

    if (blocks.length === 1 && !ab.text && !ab.question) {
      abBox.append(ui.notice('引擎消融结果是空的。', 'info'));
      return;
    }
    abBox.append(...blocks);
    if (V.length && W.length) { abMaeChart(maeBox, c); abCovChart(covBox, c); }
    if (param === 'ablation') setTimeout(() => sections.ablation.scrollIntoView({ block: 'start' }), 80);
  }

  function show(x) {
    if (x === null || x === undefined || x === '') return '—';
    if (isNum(x)) return Number.isInteger(x) ? String(x) : String(+x.toPrecision(4));
    if (typeof x === 'string') return x;
    if (typeof x !== 'object') return String(x);
    if (Array.isArray(x)) return x.map(show).join('、');
    return Object.entries(x).map(([k, v]) => `${k}: ${show(v)}`).join('；');
  }

  // Labels for the transmission-gap columns: candidate factor sets, their sum and the rest.
  function gapKey(k) {
    if (GAP_KEY[k]) return GAP_KEY[k];
    if (k.includes('+')) return k.split('+').map(gapKey).join(' + ');
    return /^[sfe]_/.test(k) ? ui.varName(k) : k;
  }

  function fillGap(box, rowsById, keys) {
    const rows = Object.entries(rowsById || {}).filter(([, x]) => x && typeof x === 'object').map(([a, x]) => ({ a, ...x }));
    if (!rows.length) { box.replaceChildren(ui.notice('这个窗口没有传导缺口数据。', 'info')); return; }
    const rank = (k) => { const i = ['f_valuation', 'f_profit_share', 'f_capex_share', 'f_profit_share+f_capex_share'].indexOf(k); return i < 0 ? 50 : i; };
    const order = keys.slice().sort((p, q) => (p === 'unexplained') - (q === 'unexplained') || (p === 'all') - (q === 'all') || rank(p) - rank(q));
    const ex = order.length && order.length <= 7
      ? order.map(k => ({ value: (r) => r.explained_by?.[k], label: gapKey(k), fmt: (x) => pct(x, 1, true),
        cls: (x) => (k === 'unexplained' && isNum(x) && Math.abs(x) > 0.03 ? 'is-bad' : '') }))
      : [{ key: 'explained_by', label: '主要由什么解释', wrap: true, fmt: (v) => (v && typeof v === 'object'
        ? Object.entries(v).filter(([, x]) => isNum(x)).sort((p, q) => Math.abs(q[1]) - Math.abs(p[1])).slice(0, 4)
          .map(([k, x]) => `${gapKey(k)} ${pct(x, 1, true)}`).join('；') || '—' : '—') }];
    box.replaceChildren(ui.table([
      { key: 'a', label: '资产', fmt: (v) => short(v) },
      { key: 'actual', label: '实际年化', fmt: (v) => pct(v) },
      { key: 'oracle', label: 'oracle 年化', fmt: (v) => pct(v) },
      { key: 'gap_annual', label: '传导缺口（年化）', fmt: (v) => pct(v, 1, true), cls: (v) => (isNum(v) && Math.abs(v) > 0.03 ? 'is-bad' : '') },
      ...ex,
    ], rows));
  }

  // Small multiples, one panel per window: variants as rows, conditional MAE as bars, the current engine's
  // unconditional (dashed) and conditional (solid) MAE as reference lines.
  function abMaeChart(el, c) {
    const { W, V, base } = c;
    const val = (v, w, k) => (isNum(v?.windows?.[w.id]?.[k]) ? v.windows[w.id][k] : null);
    if (!V.some(v => W.some(w => isNum(val(v, w, 'cagr_mae_cond_risk'))))) {
      clearChart(el, ui.notice('消融结果里没有分窗口的年化误差。', 'info'));
      return;
    }
    fresh(el);
    const width = el.clientWidth || 900;
    // The panels are laid out in pixels for this width: lay them out again when the card gets narrower or wider.
    el._abWidth = width;
    if (!el._abRo && typeof ResizeObserver === 'function') {
      el._abRo = new ResizeObserver(() => { if (el.isConnected && Math.abs((el.clientWidth || 0) - el._abWidth) > 40) abMaeChart(el, c); });
      el._abRo.observe(el);
    }
    const cols = width >= 860 ? Math.min(W.length, 4) : width >= 480 ? Math.min(W.length, 2) : 1;
    const nrows = Math.ceil(W.length / cols);
    const names = V.map(v => v.name || v.id);
    const chosen = V.findIndex(v => v.id === c.sel?.chosen && v !== base);
    const labelW = Math.min(230, Math.max(60, ...names.map((s, j) => textWidth(j === chosen ? `★ ${s}` : s, '600 11px sans-serif'))) + 12);
    const gapX = 32, titleH = 24, axisH = 22, rowH = 18, gapY = 18;
    const panelW = Math.max(110, (width - labelW - 14 - (cols - 1) * gapX) / cols);
    const plotH = V.length * rowH;
    el.style.height = `${nrows * (titleH + plotH + axisH) + (nrows - 1) * gapY + 4}px`;
    const unc = W.map(w => val(base, w, 'cagr_mae_uncond_risk') ?? mean(V.map(v => val(v, w, 'cagr_mae_uncond_risk'))));
    const cur = W.map(w => val(base, w, 'cagr_mae_cond_risk'));
    const all = [...V.flatMap(v => W.map(w => val(v, w, 'cagr_mae_cond_risk'))), ...unc, ...cur].filter(isNum);
    // Shared range for every panel, with at most ~3 round ticks so neighbouring panels' labels do not touch.
    const top = Math.max(...all) * 1.04;
    const step = [0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1].find(s => top / s <= 3.2) || niceCeil(top);
    const xmax = Math.ceil(top / step) * step;
    const pos = W.map((w, i) => {
      const r = Math.floor(i / cols), k = i % cols;
      return { left: labelW + k * (panelW + gapX), top: r * (titleH + plotH + axisH + gapY) + titleH, first: k === 0 };
    });
    const chart = charts.mount(el);
    chart.setOption({
      animation: false,
      // a panel title longer than its panel (one panel per row on a phone) starts further left, never past the edge
      title: W.map((w, i) => {
        const tw = textWidth(c.wLabel(w), '600 12px sans-serif');
        return { text: c.wLabel(w), left: Math.max(4, Math.min(pos[i].left, width - tw - 6)), top: pos[i].top - titleH + 2,
          textStyle: { fontSize: 12, fontWeight: 600, color: w.role === 'test' ? '#2b8a3e' : INK } };
      }),
      tooltip: {
        trigger: 'item', confine: true,
        formatter: (p) => {
          const v = V[p.dataIndex], w = W[p.seriesIndex];
          if (!v || !w) return '';
          const x = v.windows?.[w.id] || {};
          const d = v !== base && isNum(x.cagr_mae_cond_risk) && isNum(cur[p.seriesIndex]) ? x.cagr_mae_cond_risk - cur[p.seriesIndex] : null;
          return [`<b>${v.name || v.id}</b>${v.id === c.engineId && c.switched ? '（现行引擎）' : ''} · ${c.wLabel(w)}`,
            `已知宏观的年化误差 <b>${pct(x.cagr_mae_cond_risk, 2)}</b>${isNum(d) ? `（比${c.baseName}${d < 0 ? '低' : '高'} ${pct(Math.abs(d), 2)}）` : ''}`,
            `不知道宏观 ${pct(x.cagr_mae_uncond_risk, 2)}　90% 覆盖 ${pct(x.coverage_cond_risk, 0)}`,
            isNum(x.engine_abs_annual) || isNum(x.transmission_abs_annual)
              ? `误差拆解（年化绝对值）：引擎 ${pct(x.engine_abs_annual)}　传导 ${pct(x.transmission_abs_annual)}　情景 ${pct(x.scenario_abs_annual)}` : '',
            isNum(x.pit_tail_share) ? `终值落在 5%/95% 之外的资产占比 ${pct(x.pit_tail_share, 0)}` : ''].filter(Boolean).join('<br/>');
        },
      },
      grid: W.map((w, i) => ({ left: pos[i].left, top: pos[i].top, width: panelW, height: plotH })),
      xAxis: W.map((w, i) => ({ gridIndex: i, type: 'value', min: 0, max: xmax, interval: step,
        axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: SPLIT } },
        axisLabel: { color: INK3, fontSize: 10, hideOverlap: true, formatter: (x) => pct(x, step < 0.01 ? 1 : 0) } })),
      yAxis: W.map((w, i) => ({ gridIndex: i, type: 'category', inverse: true, data: names,
        axisTick: { show: false }, axisLine: { lineStyle: { color: LINE } },
        axisLabel: { show: pos[i].first, interval: 0, fontSize: 11, color: INK,
          formatter: (s, j) => (j === chosen ? `{sel|★ ${s}}` : V[j] === base ? `{base|${s}}` : s),
          rich: { sel: { fontWeight: 700, color: '#2b8a3e', fontSize: 11 }, base: { fontWeight: 700, color: INK, fontSize: 11 } } } })),
      series: W.map((w, i) => ({
        type: 'bar', xAxisIndex: i, yAxisIndex: i, barMaxWidth: 11,
        data: V.map((v, j) => ({ value: val(v, w, 'cagr_mae_cond_risk') ?? '-',
          itemStyle: { color: c.color(v), borderRadius: [0, 2, 2, 0], ...(j === chosen ? { borderColor: '#2b8a3e', borderWidth: 2 } : {}) } })),
        markLine: { silent: true, symbol: 'none', label: { show: false }, data: [
          isNum(unc[i]) ? { xAxis: unc[i], lineStyle: { color: INK, type: 'dashed', width: 1.4 } } : null,
          isNum(cur[i]) ? { xAxis: cur[i], lineStyle: { color: '#868e96', type: 'solid', width: 1.4 } } : null].filter(Boolean) },
      })),
    });
  }

  // Coverage per variant as dots (validation / test averages, or one dot per window without roles): a dot plot
  // can start the axis above zero without exaggerating differences the way a cut bar would.
  function abCovChart(el, c) {
    const { W, V } = c;
    const roles = ['validation', 'test'].filter(r => W.some(w => w.role === r));
    const groups = roles.length
      ? roles.map(r => ({ name: `${ROLE[r]}窗口平均`, color: r === 'validation' ? '#3b5bdb' : '#2f9e44', symbol: r === 'validation' ? 'circle' : 'diamond',
        get: (v) => (isNum(v[`${r}_coverage`]) ? v[`${r}_coverage`] : mean(W.filter(w => w.role === r).map(w => v.windows?.[w.id]?.coverage_cond_risk))) }))
      : W.map((w, i) => ({ name: c.wLabel(w), color: charts.palette(i), symbol: 'circle', get: (v) => v.windows?.[w.id]?.coverage_cond_risk }));
    const data = groups.map(g => V.map(v => { const x = g.get(v); return isNum(x) ? x : null; }));
    const vals = data.flat().filter(isNum);
    if (!vals.length) { clearChart(el, ui.notice('消融结果里没有覆盖率。', 'info')); return; }
    fresh(el);
    const names = V.map(v => v.name || v.id);
    const chosen = V.findIndex(v => v.id === c.sel?.chosen && v !== c.base);
    const lo = Math.max(0, Math.floor(Math.min(0.85, ...vals) * 20 - 1) / 20);
    el.style.height = `${V.length * 20 + 80}px`;
    const chart = charts.mount(el);
    chart.setOption({
      animation: false,
      legend: { top: 0, data: groups.map(g => g.name), textStyle: { color: INK, fontSize: 12 }, itemWidth: 12, itemHeight: 10 },
      tooltip: {
        trigger: 'item', confine: true,
        formatter: (p) => {
          const j = p.value?.[1];
          if (!isNum(j)) return '';
          return [`<b>${names[j]}</b>`, ...groups.map((g, k) => `${boxKey(g.color)}${g.name} <b>${pct(data[k][j], 1)}</b>`)].join('<br/>');
        },
      },
      // the target label sits just above the plot, below the legend; ticks thin out on a narrow plot
      grid: { left: 8, right: 30, top: 40, bottom: 24, containLabel: true },
      xAxis: { type: 'value', min: lo, max: 1, splitNumber: (el.clientWidth || 600) < 480 ? 2 : 4, axisLine: { show: false }, axisTick: { show: false },
        splitLine: { lineStyle: { color: SPLIT } }, axisLabel: { color: INK3, fontSize: 11, hideOverlap: true, formatter: (x) => pct(x, 0) } },
      yAxis: { type: 'category', inverse: true, data: names, axisTick: { show: false }, axisLine: { lineStyle: { color: LINE } },
        splitLine: { show: true, lineStyle: { color: '#f3f4f7' } },
        axisLabel: { interval: 0, fontSize: 11, color: INK, formatter: (s, j) => (j === chosen ? `{sel|★ ${s}}` : s),
          rich: { sel: { fontWeight: 700, color: '#2b8a3e', fontSize: 11 } } } },
      series: groups.map((g, k) => ({
        name: g.name, type: 'scatter', symbol: g.symbol, symbolSize: 9, z: 3, itemStyle: { color: g.color, opacity: 0.9 },
        data: data[k].map((x, j) => [isNum(x) ? x : '-', j]),
        markLine: k === 0 ? { silent: true, symbol: 'none', lineStyle: { color: INK3, type: 'dashed' },     // the y axis is inverted:
          label: { formatter: '目标 90%', color: INK3, fontSize: 10, position: 'start' }, data: [{ xAxis: 0.9 }] } : undefined,   // "start" is the top

      })),
    });
  }

  // Variants grouped by hypothesis: what changed, validation / test MAE with the gain over the baseline engine,
  // coverage, how much the 15-year distribution from today narrows, and how the selection rule treated each one
  // (the rule's checks: coverage, long-run distribution, size of the gain).
  function abVariantTable(c) {
    const { V, sel, base, hyp } = c;
    const diff = (v) => {
      if (v === base || !v.params || typeof v.params !== 'object') return '';
      const bp = base?.params && typeof base.params === 'object' ? base.params : null;
      return Object.entries(v.params).filter(([k, x]) => !bp || JSON.stringify(x) !== JSON.stringify(bp[k]))
        .map(([k, x]) => `${PARAM[k] || k} = ${PARAM_VALUE[x] ?? show(x)}`).join('，');
    };
    const mae = (x, gain, se) => h('span', {}, pct(x, 2), isNum(gain)
      ? h('small', { class: gain > 0 ? 'is-good' : gain < 0 ? 'is-bad' : '' },
        `${gain > 0 ? '改善' : gain < 0 ? '变差' : '持平'} ${pct(Math.abs(gain), 2)}${isNum(se) && se > 0 ? ` ± ${pct(se, 2)}` : ''}`) : null);
    const checksOf = (v) => { const ck = sel?.checks?.[v.id]; return ck && typeof ck === 'object' ? ck : null; };
    // the narrowest risk asset's 5%-95% width of 15-year annual returns from today, relative to the baseline
    const fanCell = (v) => {
      const f = v.forward_fan;
      if (!f || typeof f !== 'object' || !isNum(f.width_ratio_min)) return '—';
      const a = f.worst_asset, p5 = isNum(f.p5_change?.[a]) ? f.p5_change[a] : null;
      const failed = checksOf(v)?.forward_fan === false;
      return h('span', { class: failed ? 'is-bad' : '' }, v === base ? '100%（对照）' : pct(f.width_ratio_min, 0),
        v === base ? null : h('small', {}, `最窄：${a ? short(a) : '—'}${isNum(p5) && Math.abs(p5) >= 0.0005 ? `，5% 分位 ${pct(p5, 1, true)}` : ''}`));
    };
    const outcome = (v) => {
      if (v !== base && sel?.chosen === v.id) {
        return ui.chip(sel.adopt === true ? '选中并采用' : sel.adopt === false ? '选中，测试未通过' : '选中', sel.adopt === false ? 'warn' : 'good');
      }
      if (v === base) return ui.chip(c.switched ? '基准引擎（对照）' : '现行引擎（对照）', 'accent');
      if (v.adoptable === false) return ui.chip('只给结论');
      if (v.eligible === true) return ui.chip('满足条件', 'accent');
      if (v.eligible === false) return ui.chip('未通过');
      return null;
    };
    const status = (v) => {
      const ck = v === base ? null : checksOf(v);
      const marks = ck ? Object.entries(ck).filter(([, ok]) => typeof ok === 'boolean')
        .map(([k, ok]) => h('span', { class: ok ? 'is-good' : 'is-bad' }, `${ok ? '✓' : '✗'} ${CHECK[k] || k}`)) : [];
      return [outcome(v) || (marks.length ? null : '—'),
        v.id === c.engineId && c.switched ? ui.chip('现行引擎', 'good') : null,
        marks.length ? h('small', { class: 'diagnostics-checks' }, marks) : null];
    };
    const groups = [...new Set(V.map(v => v.hypothesis || ''))];
    const body = [];
    for (const g of groups) {
      const vs = V.filter(v => (v.hypothesis || '') === g);
      if (groups.length > 1) {
        const hv = hyp[g]?.verdict;
        body.push(h('tr', { class: 'g' }, h('td', { colspan: '8' }, h('span', { class: 'diagnostics-chips' }, c.hypLabel(g),
          hv ? ui.chip(...(VERDICT[hv] || [String(hv), ''])) : null,
          vs.some(v => v.adoptable === true) ? ui.chip('引擎改动，可以采用', 'accent')
            : vs.every(v => v.adoptable === false || v === base) && vs.some(v => v !== base) ? ui.chip('只给结论，不采用') : null))));
      }
      for (const v of vs) {
        const own = v === base;
        body.push(h('tr', { class: v !== base && sel?.chosen === v.id ? 'sel' : '' },
          h('td', {}, h('b', {}, v.name || v.id)),
          h('td', { class: 'd' }, typeof v.description === 'string' && v.description ? c.human(v.description) : '—', diff(v) ? h('span', { class: 'mono' }, diff(v)) : null),
          h('td', {}, mae(v.validation_mae, own ? null : v.validation_gain, v.validation_gain_se)),
          h('td', {}, mae(v.test_mae, own ? null : v.test_gain, v.test_gain_se)),
          h('td', {}, pct(v.validation_coverage, 0)),
          h('td', {}, pct(v.test_coverage, 0)),
          h('td', {}, fanCell(v)),
          h('td', { class: 's' }, status(v))));
      }
    }
    return h('div', { class: 'table-wrap' }, h('table', { class: 'tbl diagnostics-var' },
      h('thead', {}, h('tr', {}, ['方案', '改了什么', '验证 MAE', '测试 MAE', '验证覆盖', '测试覆盖', '15 年分布宽度', '选择结果']
        .map((t, i) => h('th', { class: i === 1 ? 'wrap' : '' }, t)))),
      h('tbody', {}, body)));
  }
}
