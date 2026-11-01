// 估值与盈利: should the P/E, ROIC or free cash flow of the S&P 500 / QQQ be macro factors, or does the
// model need more macro factors? The evidence in valuation.json (analysis/valuation.py), one section each:
//   answer      the question and the short answer (text_paragraphs, selection, factor candidates)
//   today       P/D against the fitted fair value: the gap and its percentile, CAPE, the drivers, the gap
//               under the alternative formulas, and the drag a pull would imply (hypothetical, not used)
//   history     P/D and fair P/D on a log axis, the gap, CAPE, real yield vs expected inflation, profits
//   regression  formula, coefficients, alternatives, stability (halves, rolling windows), theory check
//   kappa       how fast valuation reverts: in-sample vs real-time error correction, persistence
//   variants    the valuation-state variants against the current engine, and the selection rule
//   factors     the factor candidates (exogenous ones and the endogenous references)
//   scenarios   the P/D each scenario's paths imply after 15 years, and what each alternative mode changes
// Deep link: #valuation?<section>. The few numbers the page derives itself (the gap's percentile, half-lives,
// the probability-weighted P/D) come from the same file and the page says so.

const SECTIONS = [
  ['answer', '问题与回答'], ['today', '今天'], ['history', '历史'], ['regression', '公允值回归'],
  ['kappa', '拉力速度 κ'], ['variants', '变体检验'], ['factors', '因子候选'], ['scenarios', '情景隐含 P/D'],
];
// same era colours as the scenario detail (_scenario_detail.js)
const ERA_COLORS = {
  golden_age: '#5c940d', inflation_onset: '#f08c00', great_inflation: '#e8590c', disinflation: '#1098ad',
  tech_boom: '#7048e8', china_credit: '#c2255c', stagnation_qe: '#3b5bdb', covid_reflation: '#0ca678',
  ai_capex: '#862e9c',
};
const FEAT = { real_y10: '实际利率', profit_share: '利润占比', infl_expect: '通胀预期', y10: '名义收益率', const: '常数' };
const FEAT_COLOR = { real_y10: '#3b5bdb', profit_share: '#2f9e44', infl_expect: '#e8590c' };
const THEORY = { real_y10: '实际利率越高、估值越低', y10: '收益率越高、估值越低', profit_share: '利润占比越高、估值越高' };
// selection.checks keys, in the order of the rule's ① ② ③ ④
const CHECKS = [['coverage', '①覆盖率'], ['forward_fan', '②长期分布宽度'], ['gain', '③MAE 改善'], ['forecast_val', '④一年期预测']];
// walk-forward change ids (same wording as the diagnostics page)
const CHANGE = { add_valuation: '+估值变动', add_sentiment: '+消费者信心', add_profit_share: '+利润占比', add_capex_share: '+资本开支占比',
  add_wealth: '+家庭净值', add_white_collar: '+白领就业占比', add_wage: '+普通工资', add_real_y10: '+实际收益率',
  add_profit_growth: '+利润增长', add_fcf_share: '+利润减资本开支', add_cape: '+CAPE 变动', lags1: '1 个月滞后', lags3: '1–3 个月滞后',
  regime_inflation: '通胀状态交互', shocks_add: '+结构冲击', shocks_replace_oil: '结构冲击替代油价', 'ridge_0.01': '岭回归 0.01',
  'ridge_0.05': '岭回归 0.05', window_15: '只用近 15 年', window_25: '只用近 25 年', mlp16: '神经网络 16', mlp32: '神经网络 32' };
// meta.cma.method in words
const CMA_METHOD = { valuation_adjusted: '估值调整', historical: '历史漂移' };
// "% of GDP" (valuation.json data.units) in Chinese; "% of GDP（note）" -> "占 GDP 的 %；note"
const gdpUnit = (u, short = false) => {
  const s = String(u || '% of GDP');
  if (short) return s.replace(/^% of GDP.*$/, '占 GDP 的 %');
  return s.replace(/^% of GDP（(.*)）$/, '占 GDP 的 %；$1').replace(/% of GDP/g, '占 GDP 的 %');
};
const UNIT_NAMES = { mae: 'MAE', coverage: '覆盖率', gain: '改善', fan_width_ratio_min: '15 年分布宽度比',
  forecast_crps_skill: '一年期 CRPS 技能', gap_log: '对数缺口', implied_annual_drag: '拉力的年化影响', drivers: '驱动分解',
  variance_ratio: '方差比', theory_check: '理论检验', oos_r2_price_12m: '实时缺口的样本外 R²', delta: 'R² 变化' };
// the alternative mode shown per scenario in section 8, kept across re-renders (the K-line toggle)
const STATE = { mode: null };

const CSS = `
.val-anchor { scroll-margin-top: 80px; }
.val-toc { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: -4px 0 6px; }
.val-toc > .muted { font-size: 12px; }
.val-toc a.chip:hover { text-decoration: none; border-color: #c6cbd6; }
.val-q { border-left: 4px solid var(--accent); }
.val-q-label { font-size: 12px; color: var(--ink-3); }
.val-q-text { margin: 4px 0 0; font-size: 15px; font-weight: 600; line-height: 1.6; }
.val-answers { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr)); margin-top: 16px; }
.val-answer { display: flex; flex-direction: column; gap: 8px; }
.val-answer-head { display: flex; gap: 10px; align-items: flex-start; }
.val-num { flex: 0 0 24px; height: 24px; border-radius: 7px; background: var(--accent); color: #fff; display: grid;
  place-items: center; font-size: 12px; font-weight: 650; margin-top: 1px; }
.val-answer-title { font-size: 15px; font-weight: 600; line-height: 1.45; }
.val-answer p, .val-text { margin: 0; color: var(--ink-2); font-size: 13px; line-height: 1.75; }
.val-answer ol { margin: 0; padding-left: 20px; color: var(--ink-2); font-size: 13px; line-height: 1.75; }
.val-answer ol li + li { margin-top: 4px; }
.val-answer-foot { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; margin-top: auto; padding-top: 2px; font-size: 13px; }
.val-text + .val-text { margin-top: 8px; }
.val-pair { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(460px, 100%), 1fr)); }
.val-stack { display: grid; gap: 16px; }
.val-kpis { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(min(230px, 100%), 1fr)); }
.val-kpis .kpi-value { font-size: 21px; }
.card .val-kpis .kpi { box-shadow: none; background: var(--panel-2); }
.val-hypo { border-style: dashed; border-color: #f5d9a8; background: #fffcf5; }
.val-hypo .val-kpis .kpi { background: #fff; }
.val-legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 11px; color: var(--ink-3); margin: 0 0 10px; }
.val-legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.val-note { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; line-height: 1.6; }
.val-paras p { margin: 0 0 10px; color: var(--ink-2); font-size: 13px; line-height: 1.8; }
.val-paras p:last-child { margin-bottom: 0; }
details.val-more { margin-top: 12px; }
details.val-more > summary { cursor: pointer; color: var(--accent); font-size: 13px; }
details.val-more[open] > summary { margin-bottom: 10px; }
table.tbl tr.val-current td { background: #f0fbf4; }
table.tbl tr.val-current td:first-child { box-shadow: inset 3px 0 0 var(--good); }
table.tbl tr.val-current:hover td { background: #e7f7ef; }
table.tbl td.val-name { white-space: normal; text-align: left; min-width: 130px; max-width: 240px; line-height: 1.5; }
table.tbl td.val-formula-cell { white-space: normal; text-align: left; min-width: 150px; max-width: 300px; }
table.tbl td.val-wrap, table.tbl th.val-wrap { white-space: normal; text-align: left; }
table.tbl td.val-wrap { min-width: 200px; max-width: 440px; line-height: 1.6; }
table.tbl th.val-th { min-width: 58px; }
table.tbl td .chip { white-space: nowrap; }
/* every table of this page (wrapper .val-t): headers wrap, cells a little denser */
.val-t table.tbl th { white-space: normal; line-height: 1.35; vertical-align: bottom; }
.val-t table.tbl th, .val-t table.tbl td { padding: 7px 8px; }
/* the widest tables (.val-wide) stack into labelled blocks below ~1000px of card width, the others at the
   shared 480px (app.css); stacked cells drop the table-only widths and padding */
@container (max-width: 1000px) {
  .val-wide table.tbl-stack thead { display: none; }
  .val-wide table.tbl-stack, .val-wide table.tbl-stack tbody, .val-wide table.tbl-stack td { display: block; width: auto; }
  .val-wide table.tbl-stack tr { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(200px, 100%), 1fr));
    column-gap: 16px; padding: 8px 2px; border-bottom: 1px solid var(--line); }
  .val-wide table.tbl-stack td { border: 0; padding: 2px 0; text-align: left !important; white-space: normal; }
  .val-wide table.tbl-stack td:first-child { grid-column: 1 / -1; font-weight: 600; }
  .val-wide table.tbl-stack td::before { content: attr(data-label); display: inline-block; min-width: 84px;
    margin-right: 8px; font-size: 11px; color: var(--ink-3); vertical-align: top; }
  .val-wide table.tbl-stack td:first-child::before { display: none; }
  .val-wide table.tbl-stack td.val-name, .val-wide table.tbl-stack td.val-wrap { min-width: 0; max-width: none; }
  .val-wide table.tbl-stack td.val-wrap { grid-column: 1 / -1; }
  .val-wide table.tbl-stack tr.val-current td { padding-left: 10px; }
}
@container (max-width: 480px) {
  .val-t table.tbl-stack td { padding: 2px 0; }
  .val-t table.tbl-stack td.val-name, .val-t table.tbl-stack td.val-wrap, .val-t table.tbl-stack td.val-formula-cell { min-width: 0; max-width: none; }
  .val-t table.tbl-stack td.val-wrap { grid-column: 1 / -1; }
  .val-t table.tbl-stack tr.val-current td { padding-left: 10px; }
}
.val-small { display: block; font-size: 11px; color: var(--ink-3); font-weight: 400; line-height: 1.45; }
.val-strong { font-weight: 650; }
.val-chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.val-rule { margin: 6px 0 0; padding-left: 0; list-style: none; }
.val-rule li { font-size: 13px; color: var(--ink-2); line-height: 1.7; margin: 3px 0; padding-left: 1.3em; text-indent: -1.3em; }
.val-formula { font-family: var(--mono); font-size: 12px; background: var(--panel-2); border: 1px solid var(--line);
  border-radius: 6px; padding: 4px 8px; display: inline-block; overflow-wrap: anywhere; }
.val-warn-list { margin: 6px 0 0; padding-left: 18px; }
.val-warn-list li { margin: 3px 0; line-height: 1.65; }
.val-legend-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 8px; }
.val-legend-row select { max-width: 100%; min-width: 0; }
`;

function ensureStyle() {
  let el = document.getElementById('style-valuation');
  if (!el) { el = document.createElement('style'); el.id = 'style-valuation'; document.head.append(el); }
  if (!el.textContent.includes('/* valuation */')) el.textContent += `/* valuation */\n${CSS}`;
}

const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
// fixed decimals, never "-0.0"
const fixed = (x, d) => { const s = x.toFixed(d); return Number(s) === 0 ? (0).toFixed(d) : s; };
const f = (x, d = 2) => (isNum(x) ? fixed(x, d) : '—');
const sgn = (x, d = 2) => { if (!isNum(x)) return '—'; const s = fixed(x, d); return Number(s) > 0 ? `+${s}` : s; };
const times = (x, d = 1) => (isNum(x) ? `${fixed(x, d)} 倍` : '—');
const tt = (t) => (isNum(t) ? `t=${fixed(t, 1)}` : 't=—');
const bt = (c, d = 3) => (c && isNum(c.beta) ? `${sgn(c.beta, d)}（${tt(c.t)}）` : '—');
const isSig = (c) => !!c && isNum(c.t) && Math.abs(c.t) >= 2;
const sigWord = (t) => (isNum(t) ? (Math.abs(t) >= 2 ? '显著' : '不显著') : '—');
const median = (xs) => {
  const a = (xs || []).filter(isNum).sort((p, q) => p - q);
  if (!a.length) return null;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
};
// share of values strictly below x (the convention of today.pd_percentile)
const below = (xs, x) => { const a = (xs || []).filter(isNum); return a.length && isNum(x) ? a.filter(y => y < x).length / a.length : null; };
const halfLifeAR = (rho) => (isNum(rho) && rho > 0 && rho < 1 ? Math.log(0.5) / Math.log(rho) : null);   // AR(1) per year -> years
const halfLifeK = (k) => (isNum(k) && k > 0 ? Math.LN2 / k : null);
const shortName = (s) => String(s || '').split(/[：:]/)[0].trim() || String(s || '');
const shortFormula = (s) => String(s || '').replace(/^.*?=\s*a\s*\+\s*/, '').replace(/[b-e]·/g, '');
const textPx = (s, size = 10) => [...String(s ?? '')].reduce((t, ch) => t + (/[\x00-\xff]/.test(ch) ? size * 0.6 : size), 0);
const rgba = (hex, a) => { const n = parseInt(String(hex).slice(1), 16); return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`; };

export async function render(root, { api, charts, ui, meta, param, alive = () => root.isConnected }) {
  const { h, card } = ui;
  const esc = ui.esc;
  const pct = (x, d = 1, signed = false) => ui.fmtPct(x, d, signed);
  ensureStyle();
  root.append(h('p', { class: 'lead' },
    '这一页回答一个问题：标普500 / 纳指（QQQ）的市盈率、ROIC、自由现金流，要不要当成宏观因子放进模型？还需不需要再加别的宏观因子？',
    '先给结论，后面每一节是支撑它的证据，包括对结论不利的部分（公允值回归的系数符号、时期稳定性、拉力的速度）。数字全部来自估值分析（valuation.json）。'));

  const wait = ui.loading('读取估值分析结果…');
  root.append(wait);
  let v, scen;
  try {
    [v, scen] = await Promise.all([api.get('valuation'), api.get('scenarios').catch(() => null)]);
  } catch (err) {
    wait.remove();
    if (err.status === 404) root.append(ui.notice('估值分析结果（valuation.json）还没有生成。', 'info'), ui.buildHint(err));
    else root.append(ui.notice(`读取估值分析结果失败：${err.message}`, 'error'));
    return null;
  }
  if (!alive()) return null;                // the user moved on while the results loaded
  wait.remove();

  // ---------------------------------------------------------------- shared context
  const D = v.data || {};
  const dates = Array.isArray(D.dates) ? D.dates : [];
  const units = D.units || {};
  const eras = (D.eras || []).filter(e => e && e.start);
  const reg = v.regression || {};
  const coef = reg.coefficients || {};
  const stab = reg.stability || {};
  const tc = reg.theory_check || {};
  const today = v.today || {};
  const kap = v.kappa || {};
  const pers = kap.persistence || {};
  const rt = pers.ecm_real_time || {};
  const oos = pers.oos_r2_price_12m || {};
  const sel = v.selection || {};
  const settings = v.settings || {};
  const ev = settings.engine_valuation || {};
  // A valuation.json built before the analysis named its modes in Chinese has code ids in variant names and
  // descriptions ("参照：全部用 historical CMA"): say them in words (copies, so the cached response is not changed)
  const plainCma = (t) => (typeof t === 'string' ? t.replace(/historical CMA/g, '历史漂移 CMA').replace(/CMA historical（训练期历史漂移）/g, '训练期历史漂移')
    .replace(/\bCMA historical\b/g, '历史漂移 CMA').replace(/(?<![\w（])valuation_adjusted(?![\w）])/g, '估值调整 CMA') : t);
  // the notes also name the drift modes bare ("在 historical 上加拉力", "historical 或 valuation_state")
  // (an id in parentheses right after its name stays; a space next to it goes where it would sit between Chinese text)
  const wideCh = (ch) => !!ch && /[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]/.test(ch);
  const plainNote = (t) => (typeof t !== 'string' ? t : plainCma(t)
    .replace(/( ?)(?<!\w)(historical|valuation_state)(?!\w)( ?)/g, (m, a, id, b, off, str) => {
      const st = off + a.length, en = st + id.length;
      if (str[st - 1] === '（' && str[en] === '）') return m;
      const word = id === 'historical' ? '历史漂移' : '估值状态模式';
      return `${a && !wideCh(str[off - 1]) ? ' ' : ''}${word}${b && !wideCh(str[off + m.length]) ? ' ' : ''}`;
    }).replace(/历史漂移漂移/g, '历史漂移'));
  const variants = (Array.isArray(v.variants) ? v.variants : []).map(x => (x && typeof x === 'object'
    ? { ...x, name: plainCma(x.name), description: plainCma(x.description) } : x));
  const vById = new Map(variants.map(x => [x.id, x]));
  const base = vById.get('baseline') || null;
  const adoptable = variants.filter(x => x.adoptable);
  const references = variants.filter(x => !x.adoptable && x.id !== 'baseline');
  const cands = Array.isArray(v.factor_candidates) ? v.factor_candidates : [];
  const cById = new Map(cands.map(c => [c.id, c]));
  const fw = v.factor_walkforward || {};
  const minGain = isNum(fw.min_gain) ? fw.min_gain : 0.003;
  const se = v.scenario_effects || {};
  const cma = scen?.cma || meta?.cma || {};
  const paragraphs = Array.isArray(v.text_paragraphs) ? v.text_paragraphs : [];
  const regStart = reg.sample?.[0] || null, regEnd = reg.sample?.[1] || null;
  const since = String(regStart || '1973').slice(0, 4);
  const i0 = Math.max(0, regStart ? dates.indexOf(regStart) : 0);
  const pdHist = (D.pd || []).slice(i0).filter(isNum);
  const pdNow = isNum(today.pd) ? today.pd : pdHist[pdHist.length - 1];
  const pdMedian = isNum(today.pd_median) ? today.pd_median : median(pdHist);
  // the in-sample gap ln(P/D) - ln(fair P/D) per month (full-sample coefficients)
  const gapSeries = dates.map((_, i) => {
    const p = D.pd?.[i], q = D.pd_fair?.[i];
    return isNum(p) && isNum(q) && p > 0 && q > 0 ? Math.log(p / q) : null;
  });
  const gapHist = gapSeries.slice(i0).filter(isNum);
  // the gap's rank and its record come from the analysis (today.gap_percentile, today.gap_max: the
  // regression months, today's coefficients); a valuation.json without them is ranked here the same way
  const gapPct = isNum(today.gap_percentile) ? today.gap_percentile
    : gapHist.length ? below(gapHist, gapHist[gapHist.length - 1]) : null;
  let gapMax = isNum(today.gap_max?.value) ? { v: today.gap_max.value, d: today.gap_max.date || '—' } : null;
  if (!gapMax) gapSeries.forEach((g, i) => { if (i >= i0 && isNum(g) && (!gapMax || g > gapMax.v)) gapMax = { v: g, d: dates[i] }; });
  const capeHist = (D.pe_proxy || []).slice(i0).filter(isNum);
  const capeMed = median(capeHist);
  // where Shiller's own CAPE data ends (the notes say so; later months are extrapolated)
  const capeEnd = (v.notes || []).map(n => String(n).match(/CAPE[^。]*?结束（(\d{4}-\d{2})）/)).find(Boolean)?.[1] || null;
  const winName = Object.fromEntries((reg.by_window || []).map(w => [w.window, w.name]));
  const rollY = stab.rolling_years;
  // scenarios: the live index first, the analysis copy (scenario_effects.baseline) as a fallback
  let scenList = (scen?.scenarios || []).filter(s => s && isNum(s.headline?.pd_end_p50));
  if (!scenList.length) {
    scenList = Object.entries(se.baseline || {}).filter(([, x]) => isNum(x?.pd_end_p50))
      .map(([id, x]) => ({ id, name: id, prob: x.prob, headline: { pd_end_p50: x.pd_end_p50, us_equity_cagr_p50: x.us_equity_cagr_p50 } }));
  }
  scenList = scenList.slice().sort((a, b) => (b.prob ?? 0) - (a.prob ?? 0));
  const probSum = scenList.reduce((t, s) => t + (isNum(s.prob) ? s.prob : 0), 0);
  const pdPW = probSum > 0 ? scenList.reduce((t, s) => t + (isNum(s.prob) ? s.prob : 0) * s.headline.pd_end_p50, 0) / probSum : null;
  // The modes in scenario_effects were compared against the analysis's own copy of the scenario headlines
  // (scenario_effects.baseline). A later `build --only scenarios` changes scenarios.json but not valuation.json:
  // then the live index and that copy differ, and the mode comparison must stay on the copy.
  const seBase = se.baseline || {};
  const staleIds = scen?.scenarios ? (() => {
    const live = new Map((scen.scenarios || []).map(s => [s.id, s]));
    const off = (a, b, tol) => isNum(a) && isNum(b) && Math.abs(a - b) > tol;
    const changed = Object.entries(seBase).filter(([id, b]) => {
      const l = live.get(id);
      return !l || off(l.prob, b.prob, 5e-4) || off(l.headline?.pd_end_p50, b.pd_end_p50, 0.05)
        || off(l.headline?.us_equity_cagr_p50, b.us_equity_cagr_p50, 5e-5);
    }).map(([id]) => id);
    return [...changed, ...[...live.keys()].filter(id => !(id in seBase))];
  })() : [];
  const stale = Object.keys(seBase).length > 0 && staleIds.length > 0;
  const REBUILD = 'python -m scenariolab.build --only valuation';
  const variantName = (id) => vById.get(id)?.name || id;
  const decades = (c) => (c?.periods || []).map(p => `${p.period} ${sgn(p.delta, 3)}`).join('、') || '—';
  const stable = (c) => (c?.periods || []).length > 0 && c.periods.every(p => isNum(p.delta) && p.delta > minGain);

  const later = [];              // [box, draw]: charts are drawn once the DOM is in place
  const sections = {};
  const anchor = (key, title) => (sections[key] = h('div', { class: 'section-title val-anchor', id: `val-${key}` }, title));
  const go = (key) => (e) => {
    const el = sections[key];
    if (!el) return;
    e.preventDefault();
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    ui.setRoute('valuation', key, alive);
  };
  const jump = (key, text) => h('a', { href: `#valuation?${key}`, onclick: go(key) }, `${text} →`);
  // one section failing to render shows its error and leaves the rest of the page intact
  const safe = (name, fn) => {
    try { return fn(); } catch (err) {
      console.error(err);
      return [ui.notice(`「${name}」这一节渲染失败：${err.message}`, 'error')];
    }
  };
  const chartOrNote = (box, ok, text) => (ok ? box : ui.notice(text, 'info'));

  // ---------------------------------------------------------------- chart helpers
  // Era bands on the monthly axis; a band's name is drawn only where it fits (the legend lists them all).
  function eraAreas(width) {
    if (!dates.length) return [];
    const idx = new Map(dates.map((d, i) => [d, i]));
    const first = dates[0], last = dates[dates.length - 1];
    const plot = Math.max(width - 90, 100), n = Math.max(dates.length - 1, 1);
    return eras.map((e, k) => {
      if (e.start > last || (e.end && e.end < first)) return null;           // outside the data
      const from = dates.find(d => d >= e.start);                             // clipped to the data's months
      const to = e.end ? [...dates].reverse().find(d => d <= e.end) : last;
      if (!from || !to || from > to) return null;
      const px = ((idx.get(to) - idx.get(from)) / n) * plot;
      const name = e.name || e.id;
      return { from, to, label: px >= textPx(name) + 10 ? name : '', color: rgba(ERA_COLORS[e.id] || '#868e96', k % 2 ? 0.05 : 0.1) };
    }).filter(Boolean);
  }
  const eraLegend = () => h('div', { class: 'val-legend' }, eras.map(e => h('span', {},
    h('i', { style: { background: rgba(ERA_COLORS[e.id] || '#868e96', 0.5) } }),
    `${e.name}（${String(e.start).slice(0, 4)}–${e.end ? String(e.end).slice(0, 4) : '今'}）`)));
  // axis tooltip that lists only the series with a value in that month
  const tipLines = (fmt) => (ps) => {
    const arr = (Array.isArray(ps) ? ps : [ps]).filter(p => isNum(Array.isArray(p?.value) ? p.value[1] : p?.value));
    if (!arr.length) return '';
    return [`<b>${esc(arr[0].axisValueLabel ?? arr[0].name ?? '')}</b>`,
      ...arr.map(p => `${p.marker || ''}${esc(p.seriesName)}：<b>${esc(fmt(Array.isArray(p.value) ? p.value[1] : p.value))}</b>`)].join('<br/>');
  };
  // A monthly history chart: charts.line with era bands, mark lines and, for log: true, a base-2 axis
  // (P/D ticks read 16 / 32 / 64 / 128 instead of 10 / 100).
  function history(el, { series, fmt, tip = fmt, log = false, lines, yAxis2, zoom = false }) {
    const chart = charts.line(el, { x: dates, series, yFmt: fmt, tooltipFmt: tip, log, zoom, yAxis2,
      markAreas: eraAreas(el.clientWidth || 800), markLines: lines && lines.length ? lines : undefined });
    chart.setOption({ tooltip: { formatter: tipLines(tip) }, ...(log ? { yAxis: [{ logBase: 2 }] } : {}) });
    return chart;
  }
  const hasData = (arr) => Array.isArray(arr) && arr.some(isNum);
  // a mark line whose label sits inside the plot; `position` moves it off the most recent data (which is often
  // at an extreme) when the line crosses the series there
  const inside = (name, extra = {}, position = 'insideEndTop') => ({ name, label: { position }, ...extra });

  // ---------------------------------------------------------------- header: table of contents
  root.append(h('div', { class: 'val-toc' }, h('span', { class: 'muted' }, '跳到：'),
    SECTIONS.map(([k, t], i) => h('a', { href: `#valuation?${k}`, class: 'chip', onclick: go(k) }, `${i + 1} ${t}`))));

  // ================================================================ 1. the question and the short answer
  root.append(anchor('answer', '1. 问题与简短回答'));
  root.append(...safe('问题与简短回答', () => {
    const pdC = cById.get('valuation'), capeC = cById.get('cape');
    const exo = cands.filter(c => c.new && !c.endogenous);
    const nStable = exo.filter(stable).length;
    const ry = cById.get('real_y10'), pg = cById.get('profit_growth'), fcf = cById.get('fcf_share');
    const worseVal = base ? adoptable.filter(x => isNum(x.validation_mae) && x.validation_mae > base.validation_mae).length : null;
    const worseTest = base ? adoptable.filter(x => isNum(x.test_mae) && x.test_mae > base.test_mae).length : null;
    const best = vById.get(sel.best_candidate);
    const flips = (stab.sign_flips || []).map(k => FEAT[k] || k);
    // a wrong sign at |t| < 2 is weak evidence: say so next to it
    const wrongSign = Object.entries(tc).filter(([, x]) => x && x.consistent === false)
      .map(([k, x]) => `${FEAT[k] || k}（${tt(x.t)}，${x.significant_wrong ? '显著' : '不显著'}）`);
    const nVal = (settings.validation || []).length, nTest = (settings.test || []).length;
    const qqq = paragraphs.flatMap(p => String(p).split('。')).find(s => s.includes('QQQ') && /没有/.test(s));
    const usHist = cma.historical_price_drift?.us_equity, usTarget = cma.target_price_drift?.us_equity;
    const pass = (sel.eligible || []).length > 0 || sel.adopt === true;

    const answers = [
      { title: '市盈率不能当成同月的宏观因子', chip: [pdC?.verdict || '内生，只作对照', 'warn'], key: 'factors', link: '因子候选',
        body: [h('p', {},
          `股价 = 盈利 × 市盈率，市盈率这个月的变化几乎就是这个月的股价变化：市场 P/D 的月度变化与当月美股价格收益的相关是 ${f(pdC?.corr_same_month_equity)}，`,
          `席勒 CAPE 是 ${f(capeC?.corr_same_month_equity)}。把它们放进传导模型，样本外 R² 大涨（P/D：${decades(pdC)}），只是因为它们就是答案本身——用股价解释股价，`,
          '所以只作对照、不参与因子选择。', qqq ? `${qqq}。` : '')] },
      { title: nStable ? '新的外生候选里有值得一试的' : '不需要再加宏观因子', chip: [nStable ? '值得一试' : '不加入', nStable ? 'good' : 'accent'], key: 'factors', link: '因子候选',
        body: [h('p', {},
          `新的外生候选（${exo.map(c => c.name).join('、') || '—'}）单独加入基准模型，在三个样本外十年都让 R² 提升超过 +${minGain} 的有 ${nStable}/${exo.length} 个；`,
          `走步选择的最终模型仍是${fw.final?.is_base ? '基准的 ' : ' '}${fw.final?.describe || '—'}。`,
          ry && isNum(ry.spanned_r2) ? `实际利率的月度变化 ${pct(ry.spanned_r2, 0)} 已能由现有因子线性解释；` : '',
          pg && fcf ? `利润增长、利润减资本开支按发布月对齐（不前视）之后，与当月美股价格收益的相关只有 ${f(pg.corr_same_month_equity)} 和 ${f(fcf.corr_same_month_equity)}。` : '')] },
      { title: pass ? '估值作为“状态 + 回归拉力”通过了回测' : '估值作为“状态 + 回归拉力”：回测全部更差，拉力关闭',
        chip: [pass ? '采用' : `未采用 · 估值拉力${ev.enabled === false ? '关闭' : ev.enabled === true ? '开启' : '—'}`, pass ? 'good' : 'warn'],
        key: 'variants', link: '变体检验',
        body: [h('p', {},
          `把 P/D 当成状态放进引擎（按 −κ·(ln P/D − ln 公允 P/D) 拉回公允值，或只让股息率随估值路径变化）：`,
          `${adoptable.length} 个变体里 ${worseVal ?? '—'} 个在${nVal ? ` ${nVal} 个` : ''}验证窗口、${worseTest ?? '—'} 个在${nTest ? ` ${nTest} 个` : ''}测试窗口上比现行引擎差`,
          base ? `（现行引擎验证 MAE ${pct(base.validation_mae, 2)}${best ? `；最接近的「${best.name}」${pct(best.validation_mae, 2)}` : ''}）。` : '。',
          '公允值回归本身也站不稳：',
          wrongSign.length ? `${wrongSign.join('、')}的系数符号与理论相反，` : '',
          flips.length && stab.split ? `以 ${stab.split} 为界前后两段里${flips.join('、')}的系数符号相反，` : '',
          `样本内估出的拉力偏快——只用当时数据估计 κ≈${f(rt.kappa, 2)}/年（半衰期约 ${f(rt.half_life_years, 0)} 年，${tt(rt.t)}，${sigWord(rt.t)}），`,
          `实时缺口预测下一年价格收益的样本外 R² 为 ${f(oos.r2, 2)}。`)] },
      { title: '引擎实际怎么处理估值', chip: ['现行引擎', 'good'], key: 'scenarios', link: '情景隐含 P/D',
        body: [h('ol', {},
          h('li', {}, `长期漂移用常数的估值调整（资本市场假设 CMA：${CMA_METHOD[cma.method] || cma.method || '—'}）：今天 P/D ${times(cma.pd_now ?? pdNow, 0)}、${since} 年以来中位数 ${times(cma.pd_median ?? pdMedian, 0)}；`,
            `扣掉历史估值抬升（${pct(cma.rerating_tailwind, 2)}/年），再假设 ${f(cma.horizon_years, 0)} 年内向中位数回归 ${pct(cma.reversion, 0)}（${pct(cma.reversion_term, 2, true)}/年），`,
            `美股价格漂移从 ${pct(usHist, 2)}/年调到 ${pct(usTarget, 2)}/年。这是一个常数，不随路径变化。`),
          h('li', {}, `估值作为状态逐路径记录（${ev.track === false ? '当前未开启' : 'κ=0：只记录，不改变任何回报'}）：每个情景详情里都有它的路径隐含的 P/D；`,
            isNum(pdPW) ? `${scenList.length} 个情景各自 15 年后 P/D 的中位数按概率加权约 ${times(pdPW, 0)}，${pdPW > (cma.pd_median ?? pdMedian) ? '仍高于' : '低于'}历史中位数。` : ''))] },
    ];
    const answerCards = answers.map((a, i) => card({ cls: 'val-answer' },
      h('div', { class: 'val-answer-head' }, h('span', { class: 'val-num' }, String(i + 1)), h('div', { class: 'val-answer-title' }, a.title)),
      ...a.body,
      h('div', { class: 'val-answer-foot' }, ui.chip(a.chip[0], a.chip[1]), jump(a.key, a.link))));

    // what was asked, item by item
    const roic = v.roic_proxy || {};
    const vsDesc = (c) => (c ? `R² 变化 ${decades(c)}；与当月美股价格收益相关 ${f(c.corr_same_month_equity)}` : '—');
    const asked = [
      capeC && { item: '标普500 市盈率（席勒 CAPE）', how: 'CAPE 的月度变化作为候选因子，单独加入传导模型，看三个样本外十年的 R²', result: vsDesc(capeC), verdict: [capeC.verdict, 'warn'] },
      pdC && { item: '市场 P/D（股息口径的估值）', how: '同上；另外把 P/D 作为状态放进引擎，按不同速度拉回公允值（第 6 节）', result: `${vsDesc(pdC)}；带拉力的状态变体在回测里全部更差`, verdict: [pdC.verdict, 'warn'] },
      qqq && { item: '纳指（QQQ）市盈率', how: '—', result: `${qqq}。`, verdict: ['未检验（无数据）', ''] },
      isNum(roic.corr_with_profit_share) && { item: 'ROIC（投入资本回报率）',
        how: `ROIC 代理 = 利润 / 永续盘存法估算的非住宅资本存量（折旧率 ${pct(roic.depreciation, 0)}），换掉公允值回归里的利润占比`,
        result: `与利润占比相关 ${f(roic.corr_with_profit_share)}；回归系数 ${sgn(roic.coef_in_fair_value, 3)}（${tt(roic.t_in_fair_value)}），${Math.abs(roic.t_in_fair_value) >= 2 ? '显著' : '不显著'}；R² ${f(roic.r2)}`,
        verdict: ['不单独加入：用利润占比代表盈利能力', 'accent'] },
      fcf && { item: '自由现金流', how: '代理 = 税后利润 − 非住宅固定投资（占 GDP，按发布月对齐，不前视）；它的月度变化作为候选因子', result: vsDesc(fcf), verdict: [fcf.verdict, 'accent'] },
      pg && { item: '企业利润增长', how: '按发布月对齐的利润同比作为候选因子', result: vsDesc(pg), verdict: [pg.verdict, 'accent'] },
      ry && { item: '10 年期实际收益率', how: '它的月度变化作为候选因子', result: `${vsDesc(ry)}；月度变化 ${pct(ry.spanned_r2, 0)} 已能由现有因子线性解释`, verdict: [ry.verdict, 'accent'] },
      fw.final && { item: '还要不要再加宏观因子', how: '走步检验：每轮在验证期挑改动，在没参与挑选的测试期确认',
        result: `${(fw.iterations || []).map(it => `第 ${it.iter} 轮选中 ${(it.selected || []).map(x => CHANGE[x] || x).join('、')}，测试期 ${sgn(it.test_delta, 3)}`).join('；')}；最终模型：${fw.final.describe || '—'}`,
        verdict: [fw.final.is_base ? '不需要：保留基准模型' : '采用改进模型', fw.final.is_base ? 'accent' : 'good'] },
    ].filter(Boolean);
    const askedTable = ui.table([
      { key: 'item', label: '你问的', cls: 'val-name val-strong' },
      { key: 'how', label: '怎么检验的', cls: 'val-wrap' },
      { key: 'result', label: '结果', cls: 'val-wrap' },
      { key: 'verdict', label: '结论', align: 'left', fmt: (x) => ui.chip(x[0] || '—', x[1]) },
    ], asked, { stack: 'val-t val-wide' });
    // the analysis text as written, with what a reader of it should know first
    const paraNotes = [
      stale ? '这些段落写于估值分析运行时；情景结果之后重新生成过，其中的情景数字（概率加权的回报和 P/D）与第 8 节、“未来情景”页现在的数字不同。' : '',
    ].filter(Boolean);

    return [
      card({ cls: 'val-q' }, h('div', { class: 'val-q-label' }, '问题'), h('p', { class: 'val-q-text' }, v.question || '要不要把市盈率、ROIC、自由现金流当成宏观因子？')),
      h('div', { class: 'val-answers' }, answerCards),
      h('div', { style: { height: '16px' } }),
      card({ title: '你问到的每一项', sub: '每一项都用同一套标准检验：作为因子时看走步检验的样本外 R²（三个十年都要提升超过 +' + minGain + '），作为估值时看回测窗口和一年期预测。' },
        askedTable,
        paragraphs.length ? h('details', { class: 'val-more' }, h('summary', {}, `分析模块生成的完整结论（${paragraphs.length} 段）`),
          paraNotes.length ? h('p', { class: 'val-note', style: { margin: '0 0 10px' } }, `注：${paraNotes.join('')}`) : null,
          h('div', { class: 'val-paras' }, paragraphs.map(p => h('p', {}, p)))) : null),
    ];
  }));

  // ================================================================ 2. today
  root.append(anchor('today', `2. 今天（${today.date || '—'}）：P/D 对公允值`));
  root.append(...safe('今天', () => {
    const t = today;
    const lo = t.gap_log_range?.[0], hi = t.gap_log_range?.[1];
    const k = t.kappa;
    const out = [];
    out.push(h('div', { class: 'grid cols-4' },
      ui.kpi('今天的 P/D（价格 ÷ 过去 12 个月股息）', times(t.pd, 1), `${since} 年以来 ${pct(t.pd_percentile, 0)} 分位；中位数 ${times(t.pd_median, 1)}`),
      ui.kpi('公允 P/D（今天的公允值回归）', times(t.pd_fair, 1), `各变量取 ${since} 年以来均值时是 ${times(t.pd_fair_at_sample_means, 1)}`),
      ui.kpi('对数缺口 ln(P/D) − ln(公允 P/D)', sgn(t.gap_log, 2),
        `≈ 比公允值贵 ${pct(isNum(t.gap_log) ? Math.expm1(t.gap_log) : null, 0)}；${since} 年以来 ${pct(gapPct, 0)} 分位${gapMax ? `（最高 ${sgn(gapMax.v, 2)}，${gapMax.d}）` : ''}`),
      ui.kpi('席勒 CAPE', f(t.cape, 1), `${since} 年以来 ${pct(t.cape_percentile, 0)} 分位${capeEnd ? `；${capeEnd} 以后${t.date && t.date > capeEnd ? '（含今天的值）' : ''}为外推` : ''}`)));
    // the gap ranks lower than the P/D itself when the fair value's drivers "explain" part of today's P/D
    if (isNum(gapPct) && isNum(t.pd_percentile) && t.pd_percentile - gapPct >= 0.02) {
      const d0 = t.drivers || {};
      const lifts = Object.keys(d0).filter(x => isNum(d0[x]) && d0[x] > 0.01).sort((a, b) => d0[b] - d0[a]);
      const level = (x) => (isNum(t.levels?.[x]) && isNum(t.sample_means?.[x]) ? (t.levels[x] > t.sample_means[x] ? '较高的' : '较低的') : '');
      const weak = lifts.filter(x => coef[x] && !isSig(coef[x]));
      out.push(h('p', { class: 'val-note' },
        `P/D 本身在 ${pct(t.pd_percentile, 0)} 分位，缺口只在 ${pct(gapPct, 0)} 分位：`,
        lifts.length ? `今天 P/D 的一部分已被${lifts.map(x => `${level(x)}${FEAT[x] || x}`).join('和')}“解释”掉了（见下方驱动分解）`
          : '今天的公允值高于各变量取样本均值时的水平（见下方驱动分解）',
        weak.length ? `——而${weak.map(x => FEAT[x] || x).join('、')}这部分解释本身不可靠（系数不显著）。` : '。'));
    }

    // drivers of today's fair value relative to the sample means
    const drv = t.drivers || {};
    const feats = Object.keys(drv).filter(x => isNum(drv[x]));
    const sum = feats.reduce((s, x) => s + drv[x], 0);
    const topK = feats.slice().sort((a, b) => Math.abs(drv[b]) - Math.abs(drv[a]))[0];
    const drvBox = h('div', { class: 'chart short' });
    later.push([drvBox, () => {
      const { up, down } = charts.klineColors();
      const cats = [...feats.map(x => FEAT[x] || t.driver_names?.[x] || x), '合计：公允值相对均值', '缺口：P/D 相对公允值'];
      const vals = [...feats.map(x => drv[x]), sum, t.gap_log];
      const colors = [...feats.map(x => (drv[x] >= 0 ? up : down)), '#3b5bdb', '#868e96'];
      charts.bars(drvBox, { x: cats, horizontal: true, legend: false, fmt: (y) => sgn(y, 2),
        series: [{ name: '对数点', data: vals.map((y, i) => ({ value: isNum(y) ? +y.toFixed(4) : null, itemStyle: { color: colors[i] } })) }] });
    }]);
    const drvRows = feats.map(x => ({ x, today: t.levels?.[x], mean: t.sample_means?.[x], c: coef[x], d: drv[x] }));
    const drvTable = ui.table([
      { key: 'x', label: '变量', cls: 'val-name', fmt: (x, r) => h('div', { title: t.driver_names?.[x] || '' }, FEAT[x] || t.driver_names?.[x] || x,
        h('span', { class: 'val-chips', style: { marginTop: '3px' } },
          isSig(r.c) ? null : ui.chip('系数不显著', 'warn'), tc[x]?.consistent === false ? ui.chip('符号与理论相反', 'bad') : null)) },
      { key: 'today', label: '今天', fmt: (y) => (isNum(y) ? `${f(y, 2)}%` : '—') },
      { key: 'mean', label: `${since} 年以来均值`, fmt: (y) => (isNum(y) ? `${f(y, 2)}%` : '—') },
      { key: 'c', label: '系数（t）', fmt: (c) => bt(c) },
      { key: 'd', label: '贡献', fmt: (y) => sgn(y, 3), cls: (y) => ui.signCls(y, 1) },
    ], drvRows, { stack: 'val-t' });
    const drvCard = card({ title: `公允值为什么是 ${times(t.pd_fair, 0)}：驱动分解`,
      sub: `各变量相对 ${since} 年以来均值、对公允 ln(P/D) 的贡献（对数点）；柱色按 K 线配色表示正负。` },
    chartOrNote(drvBox, feats.length, '结果里没有驱动分解。'),
    h('p', { class: 'val-text' },
      `各变量取 ${since} 年以来均值时，公允 P/D 是 ${times(t.pd_fair_at_sample_means, 1)}；今天的状态把它推到 ${times(t.pd_fair, 1)}（合计 ${sgn(sum, 2)} 对数点）`,
      topK ? `，其中${FEAT[topK] || topK}贡献最大（${sgn(drv[topK], 2)}）${isSig(coef[topK]) ? '' : '，但它的系数并不显著，这部分最不可靠'}` : '',
      `。P/D 比公允值再高 ${sgn(t.gap_log, 2)} 对数点。`),
    h('div', { style: { height: '8px' } }), drvTable);

    // the gap under the alternative formulas
    const alts = (reg.alternatives || []).filter(a => isNum(a.gap_log_today));
    const gapRows = [{ formula: reg.formula, r2: reg.r2, fair: t.pd_fair, gap: t.gap_log, main: true },
      ...alts.map(a => ({ formula: a.formula, r2: a.r2, fair: a.pd_fair_today, gap: a.gap_log_today, feats: a.features }))].filter(r => isNum(r.gap));
    const gapBox = h('div', { class: 'chart short' });
    later.push([gapBox, () => {
      const chart = charts.bars(gapBox, { x: gapRows.map(r => `${r.main ? '现行：' : ''}${shortFormula(r.formula)}`), horizontal: true, legend: false,
        fmt: (y) => sgn(y, 2), wrap: true,
        series: [{ name: '今天的缺口', data: gapRows.map(r => ({ value: +r.gap.toFixed(4), itemStyle: { color: r.main ? '#3b5bdb' : '#91a7ff' } })) }] });
      chart.setOption({ tooltip: { formatter: (ps) => {
        const r = gapRows[(Array.isArray(ps) ? ps[0] : ps)?.dataIndex];
        if (!r) return '';
        return `<b>${esc(r.formula)}</b>${r.main ? '（现行）' : ''}<br/>今天的缺口 <b>${esc(sgn(r.gap, 2))}</b>（≈ 贵 ${esc(pct(Math.expm1(r.gap), 0))}）`
          + `<br/>今天的公允 P/D ${esc(times(r.fair, 1))}<br/>R² ${esc(f(r.r2, 2))}`;
      } } });
    }]);
    const mainIsMin = isNum(lo) && Math.abs(lo - t.gap_log) < 1e-9;
    const inflOnly = alts.find(a => Array.isArray(a.features) && a.features.length === 1 && a.features[0] === 'infl_expect');
    const gapCard = card({ title: '换一个公式，缺口是多少',
      sub: '同一个今天、同样的数据，公允值回归换成另外几种常见设定（系数见第 4 节）。悬停查看每个公式的 R² 和公允 P/D。' },
    chartOrNote(gapBox, gapRows.length, '结果里没有替代公式。'),
    h('p', { class: 'val-text' },
      `${gapRows.length} 种公式下今天的缺口在 ${sgn(lo, 2)} 到 ${sgn(hi, 2)} 之间（≈ 贵 ${pct(isNum(lo) ? Math.expm1(lo) : null, 0)} 到 ${pct(isNum(hi) ? Math.expm1(hi) : null, 0)}）：`,
      `都说今天偏贵，但贵多少取决于公式${mainIsMin ? '，现行公式给出的是最小的缺口' : ''}。`,
      inflOnly && isNum(inflOnly.r2) && isNum(reg.r2) && reg.r2 - inflOnly.r2 < 0.05
        ? `只用通胀预期一个变量的 R² 已有 ${f(inflOnly.r2, 2)}（现行 ${f(reg.r2, 2)}）：现行公式的解释力几乎全来自通胀预期。` : ''));

    out.push(h('div', { style: { height: '16px' } }), h('div', { class: 'val-pair' }, drvCard, gapCard));

    // the drag a pull would imply: hypothetical, the engine does not use it
    out.push(h('div', { style: { height: '16px' } }), card({ cls: 'val-hypo', title: '假设：如果打开估值拉力',
      sub: `只为说明今天的缺口意味着什么。回测不支持这个拉力（第 6 节），引擎没有使用它，任何情景的回报里都不含这一项；情景里的估值假设只有常数的 CMA 调整（美股 ${pct(cma.market_adjustment, 2, true)}/年，见第 1 节第 4 条）。`,
      actions: ui.chip('假设 · 引擎未采用', 'warn') },
    h('div', { class: 'val-kpis' },
      ui.kpi(`第一年对股价的年化影响（κ=${f(k, 2)}/年）`, pct(t.implied_annual_drag, 1, true), '= −κ × 今天的缺口', ui.signCls(t.implied_annual_drag)),
      ui.kpi(`${f(t.horizon_years, 0)} 年平均每年`, pct(t.implied_annual_drag_15y, 1, true), '缺口按 κ 逐年收窄后的年均影响', ui.signCls(t.implied_annual_drag_15y)),
      ui.kpi(`${gapRows.length} 种公式下的第一年影响`, isNum(k) && isNum(lo) && isNum(hi) ? `${pct(-k * hi, 1, true)} ~ ${pct(-k * lo, 1, true)}` : '—',
        `缺口 ${sgn(lo, 2)} ~ ${sgn(hi, 2)}，同样的 κ`, isNum(k) && isNum(lo) ? ui.signCls(-k * lo) : '')),
    h('p', { class: 'val-text', style: { marginTop: '10px' } },
      `参考拉力 κ=${f(k, 2)}/年对应半衰期约 ${f(halfLifeK(k), 0)} 年；只用当时数据估计的 κ 约 ${f(rt.kappa, 2)}/年（半衰期约 ${f(rt.half_life_years, 0)} 年，${tt(rt.t)}，${sigWord(rt.t)}）。`,
      `换成后者，第一年影响约 ${pct(isNum(rt.kappa) && isNum(t.gap_log) ? -rt.kappa * t.gap_log : null, 1, true)}。`)));
    return out;
  }));

  // ================================================================ 3. history
  root.append(anchor('history', `3. 历史：估值、利率与盈利（${String(dates[0] || '').slice(0, 4)} 年以来，按时代分段）`));
  root.append(...safe('历史', () => {
    const out = [eraLegend()];
    const pdBox = h('div', { class: 'chart tall' });
    const okPD = hasData(D.pd);
    if (okPD) {
      later.push([pdBox, () => history(pdBox, {
        series: [{ name: 'P/D', data: D.pd, color: '#3b5bdb', width: 1.8 },
          { name: '公允 P/D（今天的回归）', data: D.pd_fair, color: '#e8590c', dashed: true }],
        fmt: (y) => fixed(y, 0), tip: (y) => times(y, 1), log: true, zoom: true,
        lines: [inside(`${since} 年以来中位数 ${f(pdMedian, 0)}`, { yAxis: pdMedian }, 'insideMiddleTop'),
          ...(regStart ? [inside('回归样本起点', { xAxis: regStart })] : [])],
      })]);
    }
    out.push(card({ title: '美股 P/D 与公允 P/D（对数刻度）',
      sub: `P/D = 市价 ÷ 过去 12 个月股息（倍）。公允 P/D 把今天的回归系数（${regStart || '—'} 至 ${regEnd || '—'}）代入每个月的实际利率、利润占比和通胀预期；`
        + `${regStart || '回归样本起点'} 以前是回推（不在回归样本里）。对数刻度：同样的高度 = 同样的百分比差。拖动底部滑块可以放大某一段。` },
    chartOrNote(pdBox, okPD, '结果里没有 P/D 历史。')));

    const gapBox = h('div', { class: 'chart' }), capeBox = h('div', { class: 'chart' });
    const sd = reg.gap_sd;
    const okGap = gapSeries.some(isNum), okCape = hasData(D.pe_proxy);
    if (okGap) {
      later.push([gapBox, () => history(gapBox, {
        series: [{ name: '对数缺口', data: gapSeries.map(g => (isNum(g) ? +g.toFixed(4) : null)), color: '#495057', area: true }],
        fmt: (y) => sgn(y, 1), tip: (y) => sgn(y, 2),
        lines: [inside('公允值', { yAxis: 0 }, 'insideStartTop'),
          ...(isNum(sd) ? [inside(`+1σ（${f(sd, 2)}）`, { yAxis: sd }, 'insideStartTop'), inside('−1σ', { yAxis: -sd }, 'insideStartTop')] : [])],
      })]);
    }
    if (okCape) {
      later.push([capeBox, () => history(capeBox, {
        series: [{ name: '席勒 CAPE', data: D.pe_proxy, color: '#7048e8' }],
        fmt: (y) => fixed(y, 0), tip: (y) => f(y, 1),
        lines: [...(isNum(capeMed) ? [inside(`${since} 年以来中位数 ${f(capeMed, 1)}`, { yAxis: capeMed }, 'insideMiddleTop')] : []),
          ...(capeEnd ? [inside('此后外推', { xAxis: capeEnd })] : [])],
      })]);
    }
    out.push(h('div', { style: { height: '16px' } }), h('div', { class: 'val-pair' },
      card({ title: '缺口：P/D 相对公允值', sub: `ln(P/D) − ln(公允 P/D)，正数 = 比公允值贵。今天 ${sgn(isNum(today.gap_log) ? today.gap_log : gapHist[gapHist.length - 1], 2)}，${since} 年以来 ${pct(gapPct, 0)} 分位`
        + `${gapMax ? `，最高 ${sgn(gapMax.v, 2)}（${gapMax.d}）` : ''}。这是用全样本系数算的样本内缺口，按构造会回到 0，所以它看起来比实际回归得快（第 5 节）。` },
      chartOrNote(gapBox, okGap, '结果里没有公允值历史。')),
      card({ title: '席勒 CAPE（周期调整市盈率）', sub: `价格 ÷ 过去 10 年平均实际盈利（倍）；今天 ${f(today.cape, 1)}，${since} 年以来 ${pct(today.cape_percentile, 0)} 分位、中位数 ${f(capeMed, 1)}。`
        + `它和 P/D 讲的是同一件事；引擎用 P/D，因为它每个月都有、没有发布时滞，而且和引擎里的股息收入一致。${capeEnd ? `${capeEnd} 以后 Shiller 的数据结束，按利润和指数外推，只作展示。` : ''}` },
      chartOrNote(capeBox, okCape, '结果里没有 CAPE（缺少 Shiller 数据文件时为空）。'))));

    const ryBox = h('div', { class: 'chart' }), prBox = h('div', { class: 'chart' });
    const okRY = hasData(D.real_yield) || hasData(D.infl_expect);
    const okPR = hasData(D.profit_share) || hasData(D.fcf_proxy) || hasData(D.profit_growth_yoy);
    if (okRY) {
      later.push([ryBox, () => history(ryBox, {
        series: [{ name: '10 年期实际收益率', data: D.real_yield, color: '#3b5bdb' }, { name: '通胀预期代理', data: D.infl_expect, color: '#e8590c' }],
        fmt: (y) => `${fixed(y, 0)}%`, tip: (y) => `${f(y, 2)}%`, lines: [inside('0', { yAxis: 0 })],
      })]);
    }
    if (okPR) {
      later.push([prBox, () => history(prBox, {
        series: [{ name: `利润占比（${gdpUnit(units.profit_share, true)}）`, data: D.profit_share, color: '#2f9e44' },
          { name: '自由现金流代理（占 GDP 的 %）', data: D.fcf_proxy, color: '#1098ad' },
          { name: `利润同比（${units.profit_growth_yoy || '%'}，右轴）`, data: D.profit_growth_yoy, color: '#adb5bd', width: 1, yAxisIndex: 1 }],
        fmt: (y) => `${fixed(y, 0)}`, tip: (y) => `${f(y, 1)}%`, yAxis2: { name: '', fmt: (y) => `${fixed(y, 0)}%` },
        lines: [inside('0', { yAxis: 0 })],
      })]);
    }
    const sigFeats = FEAT_ORDER(coef).filter(x => isSig(coef[x])).map(x => FEAT[x] || x);
    out.push(h('div', { style: { height: '16px' } }), h('div', { class: 'val-pair' },
      card({ title: '实际收益率与通胀预期', sub: `实际收益率 = 10 年期收益率 − 通胀预期代理；通胀预期代理 = 过去 CPI 通胀的指数加权均值（半衰期 ${ev.infl_half_life_months ?? '—'} 个月），`
        + `模拟路径上用同样的递推。全样本回归里显著的只有${sigFeats.length ? sigFeats.join('、') : '—'}${tc.real_y10?.consistent === false ? '；实际利率的系数反而为正（与理论相反）' : ''}。单位 %。` },
      chartOrNote(ryBox, okRY, '结果里没有实际利率与通胀预期。')),
      card({ title: '盈利：利润占比、利润增长与自由现金流代理', sub: `左轴：利润占比、自由现金流代理（${gdpUnit(units.fcf_proxy)}）；右轴：利润同比（%）。`
        + `季度数据按发布月对齐（从季度第一个月起 ${ev.release_lag_months ?? '—'} 个月后才算已知），不前视。今天利润占比 ${f(today.profit_share, 1)}%，${since} 年以来均值 ${f(today.sample_means?.profit_share, 1)}%。` },
      chartOrNote(prBox, okPR, '结果里没有盈利序列。'))));
    return out;
  }));

  // ================================================================ 4. fair-value regression
  root.append(anchor('regression', '4. 公允值回归：系数、替代公式与稳定性'));
  root.append(...safe('公允值回归', () => {
    const out = [];
    const feats = FEAT_ORDER(coef);
    // theory check first: the coefficient signs that contradict theory
    const warns = Object.entries(tc).map(([k, x]) => {
      if (!x) return null;
      const name = FEAT[k] || k;
      const roll = isNum(x.rolling_n) && isNum(x.rolling_significant_wrong) && x.rolling_significant_wrong > 0
        ? `${x.rolling_n} 个 ${rollY ?? ''} 年滚动窗口中有 ${x.rolling_significant_wrong} 个显著地与理论相反` : '';
      if (x.consistent === false) {
        return `${name}的系数为 ${sgn(x.beta, 3)}（${tt(x.t)}），与理论（${THEORY[k] || x.expected}）相反${x.significant_wrong ? '且显著' : '，但不显著（|t| < 2）'}：按这个回归，${name}上升 1 个百分点，`
          + `公允 P/D 反而${x.fair_pd_change_per_unit >= 0 ? '上升' : '下降'} ${pct(Math.abs(x.fair_pd_change_per_unit), 1)}${roll ? `；${roll}` : ''}。`;
      }
      if (roll) return `${name}的系数符号与理论一致（${sgn(x.beta, 3)}，${tt(x.t)}${Math.abs(x.t) >= 2 ? '' : '，不显著'}），但${roll}。`;
      return null;
    }).filter(Boolean);
    const theoryRows = Object.entries(tc).map(([k, x]) => ({ k, ...x }));
    out.push(card({ title: '理论检验：系数符号对不对',
      sub: '理论：实际利率越高、估值越低（负号）；利润占比越高、估值越高（正号）。最后一列 = 该变量上升 1 个百分点时公允 P/D 的变化（按回归系数换算）。',
      actions: warns.length ? ui.chip('有系数与理论相反', 'bad') : ui.chip('与理论一致', 'good') },
    warns.length ? ui.notice(h('div', {}, h('b', {}, '注意：'), h('ul', { class: 'val-warn-list' }, warns.map(w => h('li', {}, w)))), '') : null,
    h('div', { style: { height: '10px' } }),
    ui.table([
      { key: 'k', label: '变量', fmt: (k) => FEAT[k] || k, cls: 'val-strong' },
      { key: 'expected', label: '理论符号', fmt: (s) => (s === '-' ? '−（负）' : s === '+' ? '+（正）' : '—') },
      { value: (r) => r, label: '全样本 β（t）', fmt: (r) => bt(r) },
      { key: 'consistent', label: '与理论', fmt: (ok) => (ok === true ? '一致' : ok === false ? '相反' : '—'), cls: (ok) => ui.goodCls(ok) },
      { key: 'significant_wrong', label: '全样本显著地相反', fmt: (x) => (x === true ? '是' : x === false ? '否' : '—'), cls: (x) => (x === true ? 'is-bad' : '') },
      { value: (r) => r, label: '滚动窗口里显著地相反', fmt: (r) => (isNum(r.rolling_n) ? `${r.rolling_significant_wrong ?? 0} / ${r.rolling_n}` : '—'),
        cls: (r) => (r.rolling_significant_wrong > 0 ? 'is-bad' : '') },
      { key: 'fair_pd_change_per_unit', label: '+1 个百分点时公允 P/D', fmt: (x) => pct(x, 1, true), cls: (x) => ui.signCls(x) },
    ], theoryRows, { stack: 'val-t' })));

    // the full-sample regression
    const capeC = reg.cape?.coefficients || {};
    const coefRows = ['const', ...feats].filter(k => coef[k]).map(k => ({ k, c: coef[k], cape: capeC[k] }));
    out.push(h('div', { style: { height: '16px' } }), card({ title: '公允值回归（全样本）',
      sub: `水平对水平的回归，变量都很持久：t 值用 Newey-West（${reg.hac_lags ?? '—'} 个月）修正，仍可能偏高。缺口 = 回归残差，标准差 ${f(reg.gap_sd, 2)}。` },
    h('div', { class: 'val-chips', style: { marginBottom: '10px' } }, h('span', { class: 'val-formula' }, reg.formula || '—'),
      ui.chip(`样本 ${regStart || '—'} 至 ${regEnd || '—'}，${reg.n_obs ?? '—'} 个月`), ui.chip(`R² ${f(reg.r2, 2)}`, 'accent'),
      reg.cape ? ui.chip(`因变量换成 ${reg.cape.dependent || 'ln CAPE'}：R² ${f(reg.cape.r2, 2)}`) : null),
    ui.table([
      { key: 'k', label: '变量', fmt: (k, r) => r.c?.name || FEAT[k] || k, cls: 'val-name' },
      { value: (r) => r.c?.beta, label: 'β', fmt: (x) => sgn(x, 3) },
      { value: (r) => r.c?.t, label: 't', fmt: (x) => f(x, 1), cls: (x) => (isNum(x) && Math.abs(x) >= 2 ? 'val-strong' : '') },
      { value: (r) => r, label: '显著（|t| ≥ 2）', fmt: (r) => (r.k === 'const' ? '—' : isSig(r.c) ? ui.chip('显著', 'accent') : '不显著') },
      { value: (r) => tc[r.k], label: '与理论', fmt: (x) => (!x ? '—' : x.consistent ? '一致' : '相反'), cls: (x) => (x ? ui.goodCls(x.consistent) : '') },
      { value: (r) => (r.k === 'const' ? null : r.c?.beta), label: '+1 个百分点时公允 P/D', fmt: (b) => (isNum(b) ? pct(Math.expm1(b), 1, true) : '—'), cls: (b) => (isNum(b) ? ui.signCls(Math.expm1(b)) : '') },
      { value: (r) => r.cape, label: '换成 ln CAPE：β（t）', fmt: (c) => bt(c) },
    ], coefRows, { stack: 'val-t' })));

    // the alternative formulas
    const altRows = [{ formula: reg.formula, r2: reg.r2, coefficients: coef, pd_fair_today: today.pd_fair, gap_log_today: today.gap_log, ecm: reg.ecm, main: true },
      ...(reg.alternatives || [])];
    // what the formulas agree on, read from the rows (R² and the gap are not monotonic in each other)
    const altOk = altRows.filter(r => r && isNum(r.r2) && isNum(r.gap_log_today));
    const mainRow = altOk.find(r => r.main);
    const allRich = altOk.length > 1 && altOk.every(r => r.gap_log_today > 0);
    const mainTop = !!mainRow && altOk.length > 1 && altOk.every(r => r.r2 <= mainRow.r2 + 1e-12 && r.gap_log_today >= mainRow.gap_log_today - 1e-12);
    out.push(h('div', { style: { height: '16px' } }), card({ title: '替代公式',
      sub: `同样的样本和方法，只换解释变量。${allRich ? '每个公式都说今天偏贵；' : ''}${mainTop ? 'R² 最高的现行公式给出的缺口最小，其余公式的缺口都更大；' : ''}`
        + '“今天贵多少”首先取决于你相信哪个公式。误差修正 κ 用样本内缺口估计（偏快，见第 5 节）。' },
    ui.table([
      { key: 'formula', label: '公式', cls: 'val-formula-cell', fmt: (x, r) => h('div', {}, h('span', { class: 'val-formula' }, x || '—'), r.main ? h('span', { class: 'val-small' }, '现行公式') : null) },
      { key: 'r2', label: 'R²', fmt: (x) => f(x, 2) },
      { key: 'coefficients', label: '系数（t）', cls: 'val-wrap',
        fmt: (c) => Object.entries(c || {}).filter(([k]) => k !== 'const').map(([k, x]) => `${FEAT[k] || k} ${bt(x)}`).join('；') || '—' },
      { key: 'pd_fair_today', label: '今天的公允 P/D', fmt: (x) => times(x, 1) },
      { key: 'gap_log_today', label: '今天的缺口', fmt: (x) => sgn(x, 2) },
      { value: (r) => r.ecm?.kappa, label: '误差修正 κ/年', fmt: (x) => f(x, 2) },
    ], altRows.filter(r => r && r.formula), { stack: 'val-t' })));

    // stability: halves, rolling windows, the backtests' expanding training samples
    const halves = stab.halves || [], rolling = stab.rolling || [];
    const counts = stab.rolling_counts || {};
    const flips = (stab.sign_flips || []);
    const stRow = (group, x) => ({ group, start: x.start || x.sample?.[0], end: x.end || x.sample?.[1], n: x.n_obs, r2: x.r2, c: x.coefficients || {} });
    const stRows = [
      { group: '全样本', start: regStart, end: regEnd, n: reg.n_obs, r2: reg.r2, c: coef },
      ...halves.map((x, i) => stRow(i === 0 ? `前段（至 ${stab.split || '—'}）` : '后段', x)),
      ...rolling.map(x => stRow(`${rollY ?? ''} 年滚动窗口`, x)),
      ...(reg.by_window || []).map(x => stRow(`回测训练期（${winName[x.window] || x.window}）`, x)),
    ];
    const cellCls = (k, c) => {
      const cls = [];
      if (isSig(c)) cls.push('val-strong');
      const th = tc[k]?.expected;
      if (isSig(c) && th && Math.sign(c.beta) !== (th === '-' ? -1 : 1)) cls.push('is-bad');
      return cls.join(' ');
    };
    const rbBox = h('div', { class: 'chart' });
    const okRoll = rolling.length > 0 && feats.length > 0;
    if (okRoll) {
      later.push([rbBox, () => {
        const labels = rolling.map(x => `${String(x.start).slice(0, 4)}–${String(x.end).slice(0, 4)}`);
        const chart = charts.bars(rbBox, { x: labels, fmt: (y) => sgn(y, 2),
          series: feats.map(k => ({ name: FEAT[k] || k, data: rolling.map(x => (isNum(x.coefficients?.[k]?.beta) ? +x.coefficients[k].beta.toFixed(4) : null)), color: FEAT_COLOR[k] })),
          markLine: [{ yAxis: 0, name: '', label: { show: false } }] });
        chart.setOption({ tooltip: { formatter: (ps) => {
          const arr = Array.isArray(ps) ? ps : [ps];
          const i = arr[0]?.dataIndex;
          const w = rolling[i];
          if (!w) return '';
          return [`<b>${esc(w.start)} 至 ${esc(w.end)}</b>（R² ${esc(f(w.r2, 2))}）`,
            ...feats.map(k => `${esc(FEAT[k] || k)}：<b>${esc(bt(w.coefficients?.[k]))}</b>`)].join('<br/>');
        } } });
      }]);
    }
    const countText = feats.filter(k => counts[k]).map(k => `${FEAT[k] || k} ${counts[k].sig_pos} 个显著为正、${counts[k].sig_neg} 个显著为负`).join('；');
    out.push(h('div', { style: { height: '16px' } }), card({ title: '稳定性：前后两段、滚动窗口与回测训练期',
      sub: '加粗 = 显著（|t| ≥ 2）；红色 = 显著且符号与理论相反。滚动窗口每个 ' + (rollY ?? '—') + ' 年、每 ' + (stab.rolling_step ?? '—') + ' 年一个；回测训练期都从 ' + (regStart || '—') + ' 开始。' },
    halves.length === 2 ? h('p', { class: 'val-text' },
      `以 ${stab.split || '—'} 为界：前段 R² ${f(halves[0].r2, 2)}，后段只有 ${f(halves[1].r2, 2)}`,
      flips.length ? `；${flips.map(k => FEAT[k] || k).join('、')}的系数在两段里符号相反` : '', '。',
      countText ? `${rolling.length} 个滚动窗口里：${countText}。` : '',
      '系数随时期变化很大，没有一个能跨时期稳定使用的公允值公式；第 6 节里所有估值拉力都没能改善回测，与此一致。') : null,
    okRoll ? h('div', {}, h('div', { class: 'val-note', style: { margin: '10px 0 0' } }, `${rollY ?? ''} 年滚动窗口的系数 β（悬停查看 t 值）`), rbBox) : null,
    h('div', { style: { height: '8px' } }),
    ui.table([
      { value: (r) => r, label: '样本', cls: 'val-name', fmt: (r) => h('div', {}, h('span', { class: 'val-small' }, r.group), `${r.start || '—'} 至 ${r.end || '—'}`) },
      { key: 'n', label: '月数', fmt: (x) => (isNum(x) ? String(x) : '—') },
      { key: 'r2', label: 'R²', fmt: (x) => f(x, 2) },
      // β with its t below it: three of these side by side stay narrow enough for a 900px window
      ...feats.map(k => ({ value: (r) => r.c?.[k], label: `${FEAT[k] || k} β（t）`, cls: (c) => cellCls(k, c),
        fmt: (c) => (c && isNum(c.beta) ? h('span', {}, sgn(c.beta, 3), h('span', { class: 'val-small' }, tt(c.t))) : '—') })),
    ], stRows, { stack: 'val-t' })));
    return out;
  }));

  // ================================================================ 5. pull speed
  root.append(anchor('kappa', '5. 拉力有多快：κ 与估值的持续性'));
  root.append(...safe('拉力速度', () => {
    const out = [];
    const ecm = kap.ecm_full_sample || reg.ecm || {};
    const lit = kap.literature || {};
    const vr = pers.variance_ratio || {};
    const engineVR = v.implied_pd?.valuation_adjusted?.variance_ratio;
    const litHL = halfLifeAR(lit.dp_ar1_annual);
    out.push(h('div', { class: 'grid cols-3' },
      ui.kpi('样本内误差修正 κ', `≈ ${f(ecm.kappa, 2)}/年`, `半衰期约 ${f(halfLifeK(ecm.kappa), 1)} 年；斜率 ${sgn(ecm.slope, 3)}（${tt(ecm.t)}），${ecm.n ?? '—'} 个月。缺口按全样本拟合，天然会回归：偏快`),
      ui.kpi('实时误差修正 κ（只用当时的数据）', `≈ ${f(rt.kappa, 2)}/年`, `半衰期约 ${f(rt.half_life_years, 1)} 年；斜率 ${sgn(rt.slope, 3)}（${tt(rt.t)}，${Math.abs(rt.t) >= 2 ? '显著' : '不显著'}），${rt.n ?? '—'} 个月，自 ${rt.first || '—'}`),
      ui.kpi('ln P/D 的年度自相关', f(pers.log_pd_ar1_annual, 3), `半衰期约 ${f(pers.log_pd_half_life_years_annual, 1)} 年（${pers.n_years ?? '—'} 年）；月度 ${f(pers.log_pd_ar1_monthly, 4)}，折合半衰期 ${f(pers.log_pd_half_life_years, 1)} 年`),
      ui.kpi('文献：美国股息率的年度自相关', f(lit.dp_ar1_annual, 3), `半衰期约 ${f(litHL, 1)} 年（本页按 AR(1) 换算）；${lit.source || '—'}`),
      ui.kpi('实时缺口预测下一年价格收益', `样本外 R² ${f(oos.r2, 2)}`, `${oos.n ?? '—'} 个月，相对当时的历史均值；负数 = 还不如历史均值`, isNum(oos.r2) ? ui.goodCls(oos.r2 > 0) : ''),
      ui.kpi('15 年方差比（历史）', f(vr.ratio, 2), `${vr.months ?? '—'} 个月对数收益的方差 /（${vr.months ?? 'N'} × 月度方差）；只有约 ${f(vr.independent_windows, 1)} 个独立窗口。现行引擎 ${f(engineVR, 2)}，随机游走 = 1`)));

    // half-lives by method
    const hl = [
      { name: '样本内缺口的自相关', y: pers.gap_half_life_years, c: '#e8590c', note: `月度 AR(1) ${f(pers.gap_ar1_monthly, 3)}；样本内缺口，偏快` },
      { name: '样本内误差修正', y: halfLifeK(ecm.kappa), c: '#e8590c', note: `κ≈${f(ecm.kappa, 2)}/年；样本内缺口，偏快` },
      { name: `参考设定 κ=${f(kap.value, 2)}`, y: kap.half_life_years, c: '#adb5bd', note: '只用来说明今天的缺口（第 2 节）' },
      { name: '实时误差修正', y: rt.half_life_years, c: '#3b5bdb', note: `κ≈${f(rt.kappa, 2)}/年；只用当时的数据，不显著` },
      { name: '文献（Cochrane 2008）', y: litHL, c: '#3b5bdb', note: `股息率年度 AR(1) ${f(lit.dp_ar1_annual, 3)}` },
      { name: 'ln P/D 年度自相关', y: pers.log_pd_half_life_years_annual, c: '#3b5bdb', note: `AR(1) ${f(pers.log_pd_ar1_annual, 3)}，${pers.n_years ?? '—'} 年` },
      { name: 'ln P/D 月度自相关', y: pers.log_pd_half_life_years, c: '#3b5bdb', note: `AR(1) ${f(pers.log_pd_ar1_monthly, 4)}` },
    ].filter(x => isNum(x.y));
    const hlBox = h('div', { class: 'chart' });
    if (hl.length) {
      later.push([hlBox, () => {
        const chart = charts.bars(hlBox, { x: hl.map(x => x.name), horizontal: true, legend: false, fmt: (y) => `${fixed(y, 0)} 年`,
          series: [{ name: '半衰期', data: hl.map(x => ({ value: +x.y.toFixed(2), itemStyle: { color: x.c } })) }] });
        chart.setOption({ tooltip: { formatter: (ps) => {
          const x = hl[(Array.isArray(ps) ? ps[0] : ps)?.dataIndex];
          return x ? `<b>${esc(x.name)}</b><br/>半衰期 <b>${esc(f(x.y, 1))} 年</b><br/>${esc(x.note)}` : '';
        } } });
      }]);
    }
    // did valuation return to the training-period fair value in the backtest windows?
    const wins = (reg.by_window || []).map(w => ({ ...w, further: isNum(w.gap_start) && isNum(w.gap_end_realized) ? Math.abs(w.gap_end_realized) > Math.abs(w.gap_start) : null }));
    const nFurther = wins.filter(w => w.further === true).length;
    const firstTrainEnd = wins.map(w => String(w.train_end || '')).filter(Boolean).sort()[0];
    const fromYear = firstTrainEnd ? Number(firstTrainEnd.slice(0, 4)) + (firstTrainEnd.slice(5, 7) === '12' ? 1 : 0) : null;
    const howCard = card({ title: 'κ 是怎么选的',
      actions: h('span', { class: 'val-chips' }, ...(kap.grid || []).map(g => ui.chip(isNum(g) ? `κ=${f(g, 2)}` : g === 'ecm' ? 'κ=训练期误差修正估计' : String(g)))) },
    h('p', { class: 'val-text' }, kap.how_chosen || '—'),
    h('p', { class: 'val-text' }, `引擎的设定：估值拉力${ev.enabled === false ? '关闭' : ev.enabled === true ? '开启' : '—'}（config.VALUATION['enabled'] = ${ev.enabled === false ? 'False' : ev.enabled === true ? 'True' : '—'}），`,
      `估值状态${ev.track ? '逐路径记录（track = True，κ=0，不影响回报）' : '不记录'}。`),
    Object.keys(kap.ecm_by_window || {}).length ? h('p', { class: 'val-note' }, '各回测训练期的误差修正 κ：',
      Object.entries(kap.ecm_by_window).map(([w, x]) => `「${winName[w] || w}」${f(x, 2)}${x >= 1 ? '（截断为 1）' : ''}`).join('、'), '。') : null);
    out.push(h('div', { style: { height: '16px' } }), h('div', { class: 'val-pair' },
      card({ title: '估值回归的半衰期：不同估计方法', sub: '缺口（或 ln P/D）回到一半所需的年数。橙色是用样本内缺口的估计（偏快），蓝色是只用当时数据的估计、文献和 ln P/D 本身的持续性。' },
        chartOrNote(hlBox, hl.length, '结果里没有持续性估计。'),
        h('p', { class: 'val-text' }, `只用当时数据的估计（${tt(rt.t)}，${sigWord(rt.t)}）和文献都指向十年量级的回归；样本内估计`,
          isNum(rt.half_life_years) && isNum(halfLifeK(ecm.kappa)) && rt.half_life_years >= 2 * halfLifeK(ecm.kappa) ? '快一倍以上' : '更快',
          '，因为样本内的缺口是按全样本拟合出来的，按构造就会回到 0。')),
      howCard));
    out.push(h('div', { style: { height: '16px' } }),
      card({ title: '回测窗口里，估值有没有回到公允值', sub: '每个回测窗口的起点用训练期（从 ' + (regStart || '—') + ' 到起点）重新估计公允值；窗口结束时的公允值按当时实际的状态计算。' },
        ui.table([
          { key: 'name', label: '窗口', cls: 'val-name', fmt: (x, r) => h('div', {}, x || r.window, h('span', { class: 'val-small' }, `训练期至 ${r.train_end || '—'}`)) },
          { value: (r) => r, label: '起点：P/D / 公允值', fmt: (r) => `${f(r.pd_start, 0)} / ${times(r.pd_fair_start, 0)}` },
          { key: 'gap_start', label: '起点缺口', fmt: (x) => sgn(x, 2) },
          { value: (r) => r, label: '结束：P/D / 公允值', fmt: (r) => `${f(r.pd_end_actual, 0)} / ${times(r.pd_fair_end_realized, 0)}` },
          { key: 'gap_end_realized', label: '结束缺口', fmt: (x) => sgn(x, 2) },
          { key: 'kappa_ecm', label: '训练期估出的 κ', fmt: (x) => (isNum(x) ? h('span', {}, f(x, 2), x >= 1 ? h('span', { class: 'val-small' }, '截断为 1') : null) : '—') },
          { key: 'further', label: '结果', fmt: (x) => (x === true ? '离公允值更远' : x === false ? '更接近公允值' : '—'), cls: (x) => ui.goodCls(x === null ? null : !x) },
        ], wins, { stack: 'val-t' }),
        wins.length ? h('p', { class: 'val-text', style: { marginTop: '10px' } },
          `${wins.length} 个窗口中有 ${nFurther} 个在结束时离公允值比开始时更远：${nFurther === wins.length ? `${fromYear ? `${fromYear} 年以来` : ''}估值并没有按训练期估计的速度回到公允值，拉力在这些窗口里主要是在加误差。` : '估值回到公允值的速度并不稳定。'}`) : null));
    return out;
  }));

  // ================================================================ 6. variant test
  root.append(anchor('variants', '6. 变体检验：把估值状态放进引擎，回测会不会更好'));
  root.append(...safe('变体检验', () => {
    const out = [];
    const tol = settings.forecast_tol ?? 0.01, icTol = settings.forecast_ic_tol ?? 0.05;
    const covDrop = settings.coverage_drop ?? 0.05, fanDrop = settings.fan_drop ?? 0.05;
    const split = String(settings.forecast_split || '2004-12').slice(0, 4);
    const best = vById.get(sel.best_candidate);
    const winList = (ids) => (ids || []).map(w => `「${winName[w] || w}」`).join('、') || '—';     // names contain 、
    out.push(card({ title: sel.adopt ? `结论：采用「${variantName(sel.chosen)}」` : '结论：没有变体通过，保留现行引擎',
      actions: h('span', { class: 'val-chips' }, sel.adopt ? ui.chip('采用', 'good') : ui.chip('估值拉力默认关闭', 'warn'),
        best ? ui.chip(`最接近：${best.name}`) : null) },
    h('p', { class: 'val-text' }, sel.reason || '—'),
    h('p', { class: 'val-note' }, `${settings.n_paths ?? '—'} 条路径 × ${(settings.seeds || []).length || '—'} 个随机种子；验证窗口：${winList(settings.validation)}；测试窗口：${winList(settings.test)}；`,
      `一年期预测 ${settings.n_origins ?? '—'} 个起点，以 ${settings.forecast_split || '—'} 为界分验证 / 测试。${settings.baseline_reproduces_backtest ? '现行引擎这一行与回测页的结果一致。' : ''}`)));

    // validation / test MAE against the current engine
    const dBox = h('div', { class: 'chart tall' });
    const okD = !!base && adoptable.length > 0;
    if (okD) {
      later.push([dBox, () => charts.bars(dBox, { x: adoptable.map(x => x.name), horizontal: true, fmt: (y) => pct(y, 2, true),
        wrap: Math.round(Math.min(250, Math.max(96, (dBox.clientWidth || 600) * 0.3))),
        series: [{ name: '验证窗口', data: adoptable.map(x => (isNum(x.validation_mae) ? +(x.validation_mae - base.validation_mae).toFixed(5) : null)), color: '#3b5bdb' },
          { name: '测试窗口', data: adoptable.map(x => (isNum(x.test_mae) ? +(x.test_mae - base.test_mae).toFixed(5) : null)), color: '#e8590c' }],
        markLine: [{ xAxis: 0, name: '', label: { show: false } }] })]);
    }
    const rel = (r) => r.id === 'baseline';
    const maeCls = (key) => (x, r) => (rel(r) || !base || !isNum(x) ? '' : x > base[key] + 1e-9 ? 'is-bad' : x < base[key] - 1e-9 ? 'is-good' : '');
    const outcome = (r) => {
      if (rel(r)) return sel.adopt ? '被替换' : '保留（现行引擎）';
      if (sel.adopt && sel.chosen === r.id) return ui.chip('采用', 'good');
      if (!r.adoptable) return '参照，只作比较';
      const ch = sel.checks?.[r.id] || {};
      const failed = CHECKS.filter(([k]) => ch[k] === false).map(([, l]) => l);
      const txt = failed.length ? `未通过：${failed.join('、')}` : r.eligible ? '合格，但没有被采用' : '未通过';
      return r.id === sel.best_candidate ? `${txt}（最接近）` : txt;
    };
    const vCols = [
      { key: 'name', label: '设定', cls: 'val-name', fmt: (x, r) => h('div', { title: r.description || '' }, h('span', { class: 'val-strong' }, x || r.id),
        rel(r) ? h('span', { class: 'val-small' }, '现行引擎') : null) },
      { key: 'validation_mae', label: '验证 MAE', cls: (x, r) => maeCls('validation_mae')(x, r), fmt: (x) => pct(x, 2) },
      { key: 'validation_gain', label: '验证改善 ± SE', fmt: (x, r) => (rel(r) ? '基准' : `${pct(x, 2, true)} ± ${pct(r.validation_gain_se, 2)}`),
        cls: (x, r) => (rel(r) || !isNum(x) ? '' : ui.goodCls(x > (settings.min_gain ?? 0.002) && x > 2 * (r.validation_gain_se || 0))) },
      { key: 'validation_coverage', label: '验证覆盖率', fmt: (x) => pct(x, 0),
        cls: (x, r) => (rel(r) || !base || !isNum(x) ? '' : x < base.validation_coverage - covDrop ? 'is-bad' : '') },
      { key: 'test_mae', label: '测试 MAE', cls: (x, r) => maeCls('test_mae')(x, r), fmt: (x) => pct(x, 2) },
      { key: 'test_coverage', label: '测试覆盖率', fmt: (x) => pct(x, 0),
        cls: (x, r) => (rel(r) || !base || !isNum(x) ? '' : x < base.test_coverage - covDrop ? 'is-bad' : '') },
      { key: 'fan_width_ratio_min', label: '15 年分布宽度（最窄的资产）', fmt: (x, r) => (rel(r) ? '100%' : pct(x, 0)),
        cls: (x, r) => (rel(r) || !isNum(x) ? '' : x < 1 - fanDrop ? 'is-bad' : '') },
      { value: (r) => r.forecast_skill, label: `一年期 CRPS 技能（≤${split} / 之后）`,
        fmt: (s, r) => (rel(r) ? '基准' : `${pct(s?.val?.crps_skill, 1, true)} / ${pct(s?.test?.crps_skill, 1, true)}`),
        cls: (s, r) => (rel(r) || !isNum(s?.val?.crps_skill) ? '' : s.val.crps_skill < -tol ? 'is-bad' : '') },
      { value: (r) => r.forecast_skill, label: `排序 IC 变化（≤${split} / 之后）`,
        fmt: (s, r) => (rel(r) ? '基准' : `${sgn(s?.val?.rank_ic_delta, 2)} / ${sgn(s?.test?.rank_ic_delta, 2)}`),
        cls: (s, r) => (rel(r) || !isNum(s?.val?.rank_ic_delta) ? '' : s.val.rank_ic_delta < -icTol ? 'is-bad' : '') },
      { key: 'eligible', label: '合格', fmt: (x, r) => (rel(r) || !r.adoptable ? '—' : ui.chip(x ? '是' : '否', x ? 'good' : 'bad')) },
      { value: (r) => r, label: '结果', cls: 'val-wrap', fmt: (r) => outcome(r) },
    ];
    // the table is wide: the headers of the numeric columns may wrap
    const vTable = (rows, opts = {}) => {
      const el = ui.table(vCols, rows, { stack: 'val-t val-wide', ...opts });
      el.querySelectorAll('thead th').forEach((th, i) => { if (i > 0 && i < vCols.length - 1) th.classList.add('val-th'); });
      return el;
    };

    const rule = ruleParts(sel.rule);
    out.push(h('div', { style: { height: '16px' } }), card({ title: '各变体相对现行引擎：验证窗口与测试窗口的 MAE 差',
      sub: '每根柱 = 该变体的 MAE − 现行引擎的 MAE；MAE = 风险资产条件中位数年化收益与实际之差的绝对值平均。0 的右边 = 比现行引擎更差；先只看验证窗口挑选，测试窗口只确认一次。' },
    chartOrNote(dBox, okD, '结果里没有变体。')));
    const nSeeds = (settings.seeds || []).length, nV = (settings.validation || []).length, nT = (settings.test || []).length;
    out.push(h('div', { style: { height: '16px' } }), card({ title: `${adoptable.length} 个估值状态变体 + 现行引擎`,
      sub: '绿色底色一行是现行引擎（不加估值拉力，回报不受估值影响；回测扇形图用历史漂移）。红色 = MAE 比现行引擎差，或违反选择规则 ①–④ 的那一项；绿色 = 比现行引擎好。'
        + `“± SE”只是 ${nSeeds || '—'} 个随机种子之间的模拟误差，不含历史抽样误差：验证、测试各只有 ${nV || '—'}、${nT || '—'} 个窗口，“更差”说的是这几段历史。悬停在设定名上查看它具体改了什么。` },
    vTable([base, ...adoptable].filter(Boolean), { rowClass: (r) => (rel(r) ? 'val-current' : '') }),
    h('details', { class: 'val-more' }, h('summary', {}, '每个变体具体改了什么'),
      h('ul', { class: 'list' }, [base, ...adoptable, ...references].filter(Boolean).map(x => h('li', {}, h('b', {}, x.name), `：${x.description || '—'}`)))),
    references.length ? h('details', { class: 'val-more' }, h('summary', {}, `参照行（${references.length} 个，只作比较、不参与选择）`),
      vTable(references)) : null));
    out.push(h('div', { style: { height: '16px' } }), card({ title: '选择规则', sub: '先只用验证窗口挑选，测试窗口只确认一次；不通过就不采用，不再换第二名（与残差消融同一套纪律）。' },
      rule.head ? h('p', { class: 'val-text' }, rule.head) : null,
      rule.items.length ? h('ul', { class: 'val-rule' }, rule.items.map(x => h('li', {}, x))) : null,
      rule.tail ? h('p', { class: 'val-text', style: { marginTop: '8px' } }, rule.tail) : null));

    // the P/D at the end of each backtest window: each variant's conditional median vs the actual
    const wins = Object.keys(adoptable[0]?.windows || {});
    if (wins.length) {
      const act = Object.fromEntries(wins.map(w => [w, adoptable[0].windows[w]?.pd?.actual_end]));
      out.push(h('details', { class: 'val-more card' }, h('summary', {}, '回测窗口结束时的 P/D：各变体的条件模拟中位数 vs 实际'),
        h('p', { class: 'val-note', style: { margin: '0 0 8px' } }, '条件模拟 = 给定当时真实的宏观路径；起点 P/D、训练期公允值和缺口见第 5 节。拉力越强，模拟的 P/D 越靠近训练期的公允值。'),
        ui.table([
          { key: 'name', label: '变体', cls: 'val-name' },
          ...wins.map(w => ({ value: (r) => r.windows?.[w]?.pd?.cond_end_p50, label: `${winName[w] || w}（实际 ${times(act[w], 0)}）`, fmt: (x) => times(x, 0) })),
        ], adoptable, { stack: 'val-t' })));
    }
    return out;
  }));

  // ================================================================ 7. factor candidates
  root.append(anchor('factors', '7. 因子候选：还要不要加宏观因子'));
  root.append(...safe('因子候选', () => {
    const out = [];
    const periods = [];
    for (const c of cands) for (const p of c.periods || []) if (!periods.includes(p.period)) periods.push(p.period);
    const exo = cands.filter(c => !c.endogenous);
    const fBox = h('div', { class: 'chart' });
    const okF = exo.length > 0 && periods.length > 0;
    if (okF) {
      later.push([fBox, () => charts.bars(fBox, { x: exo.map(c => c.name), wrap: true, fmt: (y) => sgn(y, 3),
        series: periods.map((p, i) => ({ name: p, data: exo.map(c => { const d = (c.periods || []).find(x => x.period === p)?.delta; return isNum(d) ? +d.toFixed(5) : null; }),
          color: ['#91a7ff', '#4c6ef5', '#364fc7'][i % 3] })),
        markLine: [{ yAxis: minGain, name: `门槛 +${minGain}` }] })]);
    }
    const kind = (c) => (c.endogenous ? '内生对照' : c.new ? '新的外生候选' : '已有候选');
    const kindChip = (c) => ui.chip(kind(c), c.endogenous ? 'warn' : c.new ? 'accent' : '');
    const why = (s) => String(s || '—').replace(/\bf_valuation\b/g, '估值变动因子');
    const order = [...cands.filter(c => !c.endogenous && c.new), ...cands.filter(c => !c.endogenous && !c.new), ...cands.filter(c => c.endogenous)];
    out.push(card({ title: '外生候选单独加入基准模型：三个样本外十年的 R² 变化',
      sub: `每根柱 = 只加这一个候选后样本外 R² 的变化（只用之前的数据估计）；三个十年都高于虚线（+${minGain}）才算稳定有效。内生的两个对照（CAPE、P/D）数值大得多，只在下表列出。` },
    chartOrNote(fBox, okF, '结果里没有外生候选。')));
    out.push(h('div', { style: { height: '16px' } }), card({ title: '全部候选',
      sub: '“现有因子能解释”= 它的月度变化能被现有 10 个因子线性解释的比例；按发布月对齐的序列只用当时已发布的数据。内生对照的 R² 增量没有意义：它们几乎就是股价本身。' },
    ui.table([
      { key: 'name', label: '候选', cls: 'val-name', fmt: (x, r) => h('div', {}, h('span', { class: 'val-strong' }, x || r.id),
        h('span', { class: 'val-chips', style: { marginTop: '3px' } }, kindChip(r)),
        h('span', { class: 'val-small' }, [r.release_dated && !String(x || '').includes('发布月') ? '按发布月' : '',
          r.accepted_in_walkforward ? '走步中曾被选中' : ''].filter(Boolean).join(' · '))) },
      ...periods.map(p => ({ value: (r) => (r.periods || []).find(x => x.period === p)?.delta, label: p, fmt: (x) => sgn(x, 3),
        cls: (x, r) => (r.endogenous || !isNum(x) ? 'muted' : x > minGain ? 'is-good' : x < 0 ? 'is-bad' : '') })),
      { key: 'corr_same_month_equity', label: '与当月美股相关', fmt: (x) => f(x, 2) },
      { key: 'spanned_r2', label: '现有因子能解释', fmt: (x) => (isNum(x) ? pct(x, 0) : '—') },
      { key: 'verdict', label: '结论', fmt: (x, r) => ui.chip(x || '—', r.endogenous ? 'warn' : stable(r) ? 'good' : 'accent') },
      { key: 'why', label: '原因', cls: 'val-wrap', fmt: (x) => why(x) },
    ], order, { stack: 'val-t val-wide' })));
    const fin = fw.final || {};
    out.push(h('div', { style: { height: '16px' } }), card({ title: `走步选择的最终模型：${fin.describe || '—'}`,
      actions: fin.is_base ? ui.chip('保留基准模型', 'accent') : fin.is_base === false ? ui.chip('采用改进模型', 'good') : null,
      sub: '和“诊断与迭代”页同一套走步检验：每轮只按验证期挑改动，最后在没参与挑选的测试期确认。' },
    (fw.iterations || []).length ? h('ul', { class: 'list' }, fw.iterations.map(it => h('li', {},
      `第 ${it.iter} 轮选中：${(it.selected || []).map(x => CHANGE[x] || x).join('、') || '无'}；测试期 R² 变化 ${sgn(it.test_delta, 3)}（${it.improved ? '有改善' : '没有改善'}）`))) : null,
    fin.reason ? h('p', { class: 'val-text', style: { marginTop: '8px' } }, fin.reason) : null,
    h('p', { class: 'val-note' }, h('a', { href: '#diagnostics?robust' }, '在“诊断与迭代”页查看全部候选改动 →'))));
    return out;
  }));

  // ================================================================ 8. scenarios
  root.append(anchor('scenarios', '8. 情景：每个情景的路径隐含的 15 年后 P/D'));
  root.append(...safe('情景隐含 P/D', () => {
    const out = [];
    const ip = v.implied_pd || {};
    const va = ip.valuation_adjusted || {}, hs = ip.historical || {};
    out.push(h('p', { class: 'val-text', style: { marginBottom: '12px' } },
      '引擎在每条路径上记录 P/D（价格 ÷ 过去 12 个月股息，κ=0：只记录，不把价格拉向任何公允值，所以不改变回报）。它说明的是：一个情景的价格路径和股息路径合起来，意味着 15 年后的估值有多高。'));
    if (stale) {
      out.push(ui.notice([`情景结果在这份估值分析之后重新生成过（${staleIds.length} 个情景的概率或结果变了）。本节上半部分按现在的情景结果计算，与“未来情景”页一致；`
        + '“换一种估值处理”的表和逐个情景对比来自估值分析，仍按它当时的情景结果比较。在项目根目录运行 ', h('code', { class: 'mono' }, REBUILD), ' 重新对齐。']),
      h('div', { style: { height: '12px' } }));
    }
    out.push(h('div', { class: 'grid cols-4' },
      ui.kpi('今天的 P/D', times(pdNow, 1), `${since} 年以来 ${pct(today.pd_percentile, 0)} 分位`),
      ui.kpi('情景按概率加权：15 年后 P/D 中位数', times(pdPW, 0), `${scenList.length} 个情景各自的中位数按后验概率加权（本页计算）`),
      ui.kpi('无视图模拟（现行估值调整 CMA）', times(va.pd_end?.p50, 0), `5%–95%：${f(va.pd_end?.p5, 0)} ~ ${f(va.pd_end?.p95, 0)} 倍`),
      ui.kpi(`${since} 年以来的中位数`, times(pdMedian, 1), `历史漂移（不做估值假设）时无视图模拟是 ${times(hs.pd_end?.p50, 0)}`)));

    // per scenario
    const sBox = h('div', { class: 'chart tall' });
    const okS = scenList.length > 0;
    if (okS) {
      sBox.style.height = `${Math.max(320, scenList.length * 28 + 70)}px`;
      later.push([sBox, () => {
        const chart = charts.bars(sBox, { x: scenList.map(s => shortName(s.name)), horizontal: true, legend: false, fmt: (y) => fixed(y, 0),
          series: [{ name: '15 年后 P/D 中位数', data: scenList.map(s => +s.headline.pd_end_p50.toFixed(2)), color: '#3b5bdb' }],
          markLine: [...(isNum(pdNow) ? [{ xAxis: pdNow, name: `今天 ${f(pdNow, 0)}` }] : []), ...(isNum(pdMedian) ? [{ xAxis: pdMedian, name: `中位数 ${f(pdMedian, 0)}` }] : [])] });
        chart.setOption({ tooltip: { formatter: (ps) => {
          const s = scenList[(Array.isArray(ps) ? ps[0] : ps)?.dataIndex];
          if (!s) return '';
          return `<b>${esc(s.name)}</b><br/>后验概率 ${esc(pct(s.prob, 1))}<br/>15 年后 P/D 中位数 <b>${esc(times(s.headline.pd_end_p50, 1))}</b>`
            + `<br/>美股名义年化中位数 ${esc(pct(s.headline.us_equity_cagr_p50, 1))}<br/>点击查看这个情景`;
        } } });
        chart.on('click', (p) => { const s = scenList[p.dataIndex]; if (s && scen) location.hash = `scenarios?${encodeURIComponent(s.id)}`; });
      }]);
    }
    const years = va.pd_p50_by_year?.length ? va.pd_p50_by_year.length - 1 : 15;
    const sRows = scenList.map(s => {
      const end = s.headline.pd_end_p50;
      return { ...s, end, chg: isNum(pdNow) ? end / pdNow - 1 : null, ann: isNum(pdNow) && end > 0 ? Math.log(end / pdNow) / years : null, rank: below(pdHist, end) };
    });
    out.push(h('div', { style: { height: '16px' } }), h('div', { class: 'val-pair' },
      card({ title: '各情景 15 年后的 P/D 中位数', sub: '按后验概率从高到低；竖线是今天和 ' + since + ' 年以来的中位数。点击条形查看该情景（详情里有完整的 P/D 路径）。' },
        chartOrNote(sBox, okS, '还没有情景结果（scenarios.json）。')),
      card({ title: '数字', sub: `“相对今天”是 15 年后中位数 ÷ 今天 − 1；“估值变化”= ln(15 年后 P/D ÷ 今天) ÷ ${years}，即按中位数算的估值变化对年化价格回报的大致贡献；分位 = 在 ${since} 年以来的历史 P/D 里的位置。点击一行查看该情景。` },
        ui.table([
          { key: 'name', label: '情景', cls: 'val-name', fmt: (x) => shortName(x) },
          { key: 'prob', label: '概率', fmt: (x) => pct(x, 1) },
          { key: 'end', label: '15 年后', fmt: (x) => times(x, 0) },
          { key: 'chg', label: '相对今天', fmt: (x) => pct(x, 0, true), cls: (x) => ui.signCls(x) },
          { key: 'ann', label: '估值变化/年', fmt: (x) => pct(x, 1, true), cls: (x) => ui.signCls(x) },
          { key: 'rank', label: '历史分位', fmt: (x) => pct(x, 0) },
        ], sRows, { stack: 'val-t', onRowClick: scen ? (r) => { location.hash = `scenarios?${encodeURIComponent(r.id)}`; } : undefined }))));

    // the unconditional path
    const pBox = h('div', { class: 'chart' });
    const okP = hasData(va.pd_p50_by_year) || hasData(hs.pd_p50_by_year);
    const y0 = Number(String(today.date || '').slice(0, 4)) || 0;
    if (okP) {
      const n = Math.max(va.pd_p50_by_year?.length || 0, hs.pd_p50_by_year?.length || 0);
      later.push([pBox, () => {
        const chart = charts.line(pBox, { x: Array.from({ length: n }, (_, i) => (y0 ? String(y0 + i) : `第 ${i} 年`)),
          series: [{ name: '估值调整 CMA（现行）', data: va.pd_p50_by_year || [], color: '#3b5bdb', width: 2 },
            { name: '历史漂移（不做估值假设）', data: hs.pd_p50_by_year || [], color: '#868e96', dashed: true }],
          yFmt: (y) => fixed(y, 0), tooltipFmt: (y) => times(y, 1),
          markLines: isNum(pdMedian) ? [{ yAxis: pdMedian, name: `${since} 年以来中位数 ${f(pdMedian, 0)}` }] : undefined });
        chart.setOption({ xAxis: { boundaryGap: false }, yAxis: [{ min: 0 }] });
      }]);
    }
    const histVR = pers.variance_ratio?.ratio;
    out.push(h('div', { style: { height: '16px' } }), card({ title: '没有情景视图时，P/D 的中位数路径',
      sub: '从今天出发、不加任何情景视图的模拟（无条件），按年取 P/D 的中位数：现行引擎的常数估值调整让它缓慢下降，而历史漂移会让它继续上升。' },
    chartOrNote(pBox, okP, '结果里没有无条件 P/D 路径。'),
    isNum(va.log_price_sd) ? h('p', { class: 'val-text', style: { marginTop: '8px' } },
      `无视图模拟里，15 年对数价格收益的标准差 ${f(va.log_price_sd, 2)}：估值变化部分 ${f(va.valuation_part_sd, 2)}、股息增长部分 ${f(va.dividend_part_sd, 2)}（两者相关 ${f(va.valuation_dividend_corr, 2)}）——估值像随机游走一样漂移。`,
      `15 年方差是“月度方差 × 180”的 ${pct(va.variance_ratio, 0)}（随机游走 = 100%），历史上是 ${pct(histVR, 0)}（只有约 ${f(pers.variance_ratio?.independent_windows, 1)} 个独立窗口，误差很大）。`,
      '这不一定是错（回测并不支持更快的估值回归），所以每个情景隐含的 P/D 都摆出来，由读者判断是否合理。') : null));

    // what each alternative mode would change
    const modes = se.modes || {};
    const modeIds = Object.keys(modes);
    const bpw = se.baseline_prob_weighted || {};
    if (modeIds.length) {
      const mRows = modeIds.map(id => ({ id, name: variantName(id), ...modes[id] }));
      const d = (k) => (r) => r.prob_weighted_delta?.[k];
      out.push(h('div', { style: { height: '16px' } }), card({ title: '换一种估值处理，情景结果会变多少',
        sub: `每一行 = 把 ${Object.keys(seBase).length || scenList.length || '—'} 个情景全部换成该模式重新模拟（${se.n_paths ?? '—'} 条路径），按概率加权后减去现行引擎；颜色按 K 线配色表示正负。`
          + `${stale ? '估值分析当时' : ''}现行引擎（估值调整 CMA）的概率加权：美股名义 ${pct(bpw.us_equity_cagr_p50, 1)}、实际 ${pct(bpw.us_equity_real_cagr_p50, 1)}、纳指 ${pct(bpw.nasdaq_cagr_p50, 1)}、`
          + `黄金 ${pct(bpw.gold_cagr_p50, 1)}，最大回撤中位数 ${pct(bpw.us_equity_max_dd_p50, 0)}。用历史漂移的模式同时去掉了“黄金只跟通胀”的假设，所以黄金也会变。` },
      ui.table([
        { key: 'name', label: '模式', cls: 'val-name' },
        { value: d('us_equity_cagr_p50'), label: 'Δ美股名义年化', fmt: (x) => pct(x, 1, true), cls: (x) => ui.signCls(x) },
        { value: d('us_equity_real_cagr_p50'), label: 'Δ美股实际年化', fmt: (x) => pct(x, 1, true), cls: (x) => ui.signCls(x) },
        { value: d('nasdaq_cagr_p50'), label: 'Δ纳指年化', fmt: (x) => pct(x, 1, true), cls: (x) => ui.signCls(x) },
        { value: d('gold_cagr_p50'), label: 'Δ黄金年化', fmt: (x) => pct(x, 1, true), cls: (x) => ui.signCls(x) },
        { value: d('us_equity_max_dd_p50'), label: 'Δ美股最大回撤', fmt: (x) => pct(x, 1, true), cls: (x) => ui.signCls(x) },
        { key: 'pd_end_p50_prob_weighted', label: '15 年后 P/D（概率加权）', fmt: (x) => times(x, 0) },
        { value: (r) => r, label: '单个情景最大 |Δ美股|', fmt: (r) => `${pct(r.max_abs_us_equity_delta, 1)}（${shortName(scenList.find(s => s.id === r.max_scenario)?.name || r.max_scenario || '—')}）` },
      ], mRows, { stack: 'val-t val-wide' })));

      // per scenario, for one mode
      if (!STATE.mode || !modes[STATE.mode]) STATE.mode = modes.vs_tw_k10 ? 'vs_tw_k10' : modeIds[0];
      const cBox = h('div', { class: 'chart' });
      cBox.style.height = `${Math.max(360, scenList.length * 38 + 80)}px`;
      // the modes were run against the analysis's copy of the scenarios: compare with that copy, not the live index
      const baseOf = (s) => (isNum(seBase[s.id]?.pd_end_p50) ? seBase[s.id].pd_end_p50 : s.headline.pd_end_p50);
      const baseName = stale ? '现行引擎（估值分析时）' : '现行引擎';
      const drawCompare = () => {
        const m = modes[STATE.mode] || {};
        const per = m.per_scenario || {};
        const chart = charts.bars(cBox, { x: scenList.map(s => shortName(s.name)), horizontal: true, fmt: (y) => fixed(y, 0),
          series: [{ name: baseName, data: scenList.map(s => +baseOf(s).toFixed(2)), color: '#3b5bdb' },
            { name: variantName(STATE.mode), data: scenList.map(s => (isNum(per[s.id]?.pd_end_p50) ? +per[s.id].pd_end_p50.toFixed(2) : null)), color: '#f08c00' }],
          markLine: isNum(pdNow) ? [{ xAxis: pdNow, name: `今天 ${f(pdNow, 0)}` }] : undefined });
        chart.setOption({ tooltip: { formatter: (ps) => {
          const s = scenList[(Array.isArray(ps) ? ps[0] : ps)?.dataIndex];
          if (!s) return '';
          const x = per[s.id] || {};
          return `<b>${esc(s.name)}</b><br/>15 年后 P/D：${esc(baseName)} ${esc(times(baseOf(s), 0))} → ${esc(variantName(STATE.mode))} ${esc(times(x.pd_end_p50, 0))}`
            + `<br/>Δ美股名义年化 <b>${esc(pct(x.us_equity_delta, 1, true))}</b>，Δ纳指 ${esc(pct(x.nasdaq_delta, 1, true))}`;
        } } });
      };
      const pick = ui.select(mRows.map(r => ({ value: r.id, label: r.name })), STATE.mode, (val) => { STATE.mode = val; drawCompare(); });
      pick.setAttribute('aria-label', '选择估值模式');
      out.push(h('div', { style: { height: '16px' } }), card({ title: '逐个情景：现行引擎 vs 所选模式的 15 年后 P/D',
        sub: '选择一种模式，看它把每个情景的 15 年后 P/D 改到哪里；悬停查看美股和纳指年化回报的变化。'
          + (stale ? '“现行引擎”一栏是估值分析当时的情景结果（各模式是和它比的），与上面按现在结果算的数字不同。' : '') },
      h('div', { class: 'val-legend-row' }, h('span', { class: 'muted', style: { fontSize: '12px' } }, '模式'), pick), okS ? cBox : ui.notice('还没有情景结果（scenarios.json）。', 'info')));
      if (okS) later.push([cBox, drawCompare]);
    }
    return out;
  }));

  // ================================================================ notes
  root.append(h('div', { class: 'section-title' }, '口径与说明'));
  root.append(h('div', { class: 'grid cols-2 text' },
    card({ title: '口径' }, h('dl', { class: 'kv' }, Object.entries(v.units || {}).flatMap(([k, x]) => [h('dt', {}, UNIT_NAMES[k] || k),
      h('dd', {}, String(x).replace(/\*_gain_se/g, '“± SE”').replace(/fair_pd_change_per_unit/g, '“+1 个百分点时公允 P/D”'))]))),
    card({ title: '说明' }, h('ul', { class: 'list' }, (v.notes || []).map(n => h('li', { class: 'muted' }, plainNote(n)))))));

  // ================================================================ charts, once the DOM exists
  for (const [el, draw] of later) {
    try { draw(); } catch (err) {
      console.error(err);
      charts.unmount(el);
      el.style.height = 'auto';
      el.replaceChildren(ui.notice(`图表渲染失败：${err.message}`, 'error'));
    }
  }
  return param && sections[param] ? sections[param] : null;
}

// The fair-value features in the regression's order (constant excluded).
function FEAT_ORDER(coef) { return Object.keys(coef || {}).filter(k => k !== 'const'); }

// selection.rule as {head, items (① ② ③ ④), tail}: the last item carries the confirmation step after
// its first full stop. Falls back to the whole text as the head.
function ruleParts(text) {
  const s = String(text || '');
  const parts = s.split(/(?=[①②③④⑤⑥⑦⑧])/);
  if (parts.length < 2) return { head: s, items: [], tail: '' };
  const head = parts.shift();
  let tail = '';
  const last = parts.length - 1;
  const k = parts[last].indexOf('。');
  if (k > 0 && k < parts[last].length - 1) { tail = parts[last].slice(k + 1); parts[last] = parts[last].slice(0, k + 1); }
  return { head, items: parts.map(p => p.trim().replace(/；$/, '')), tail };      // the ① ② … stay: the results column cites them
}
