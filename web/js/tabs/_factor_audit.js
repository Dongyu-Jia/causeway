// The 缺失因子审计 section of the diagnostics tab (a helper module, not a tab): factor_audit.json, written by
// analysis/factor_audit.py after the walk-forward. Every number in a sentence is read from that file when the
// page is drawn; numbers from the frozen research archive carry its tag (「研究轮存档（…）」), recomputed ones
// say so. Sub-anchors (deep links #diagnostics?audit-families, ?audit-assets, ?audit-concepts) are registered
// in the tab's `sections`.
import * as ui from '../ui.js';

const { h, card } = ui;
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
function fixed(x, d, signed = false) {
  let s = x.toFixed(d);
  if (Number(s) === 0) s = (0).toFixed(d);
  return signed && Number(s) > 0 ? `+${s}` : s;
}
const num = (v, d = 3, signed = false) => (isNum(v) ? fixed(v, d, signed) : '—');
const r2s = (v) => num(v, 3, true);
const pct = (v, d = 0) => (isNum(v) ? `${fixed(v * 100, d)}%` : '—');
const pFmt = (p) => (!isNum(p) ? '—' : p < 0.001 ? p.toExponential(1).replace('e-', 'e−') : p.toFixed(3));
const int = (v) => (isNum(v) ? v.toLocaleString('zh-CN') : '—');
const arr = (x) => (Array.isArray(x) ? x : []);
const obj = (x) => (x && typeof x === 'object' && !Array.isArray(x) ? x : {});
const BH_Q = 0.10;

// The research round's candidates (family:id) and forecast targets the archived tables list, by name.
const CAND = {
  'risk:spread_fixed': '利差计时修正', 'risk:gz': 'GZ 信用利差', 'rates:tp10_x_core4': '期限溢价 × 上月核心 CPI > 4%',
  'rates:tp10_x_infl4': '期限溢价 × 上月 CPI > 4%', 'rates:tp10_x_infl3': '期限溢价 × 上月 CPI > 3%',
  'rates:tp10_x_infl5': '期限溢价 × 上月 CPI > 5%', 'rates:tp10_x_cpi_level': '期限溢价 × CPI 同比水平',
  'rates:tips10_avg': '10 年期 TIPS 实际收益率（月均）', 'commod:metals_oil': '基础金属相对油价',
  'commod:usd_em': '美元对新兴市场货币', 'commod:usd_em_rel': '新兴市场相对主要货币的美元变化', 'risk:mortgage_spread': '房贷利差',
  'housing:defense_gdp': '国防支出/GDP', 'housing:shelter_cpi@news': '住房 CPI（按发布月）', 'housing:china_import_share': '自中国进口占比',
  'housing:debt_public_gdp': '公众持有国债/GDP', 'housing:exports_gdp': '出口/GDP', 'housing:ppi_foods': '加工食品 PPI',
  'housing:resinv_share': '住宅投资/GDP', 'labor:cap_income_share': '家庭资本收入占比', 'labor:earnings_mgmt_prof_premium': '管理与专业岗位的收入溢价',
  'labor:labor_share_rt': '劳动收入占比（按发布月）', 'labor:wage_prod_gap': '生产率–工资差', 'labor:real_pce_nondurables': '实际非耐用品消费',
  'rates:bank_credit__known_yoy': '银行信贷同比', 'rates:re_loans__known_yoy': '银行房地产贷款同比',
  'tech:comp_share.known_level': '电脑与外设投资/GDP', 'risk:epu': '经济政策不确定性',
};
const TARGET = { cpi_yoy: 'CPI 同比变化', y10: '10 年期收益率变化', oil: '油价变化', ff: '政策利率变化', ip_yoy: '工业产出同比变化',
  unrate: '失业率变化', spread: '信用利差变化' };
const KIND = { data_fix: '数据修正', dedicated: '单个资产' };
const VERDICT = { CONFIRMED: ['确认', 'good'], PLAUSIBLE: ['可信', 'warn'], REFUTED: ['否定', 'bad'] };
const STATUS = { earns: ['有作用', 'good'], partial: ['部分', 'accent'], null: ['测过但无效', ''], untested: ['有数据未测', 'warn'],
  none: ['没有允许的数据源', 'bad'] };
const STATUS_ORDER = ['earns', 'partial', 'null', 'untested', 'none'];
// column_info roles; a "model" column (a config factor) is 传导模型 only when the selected model uses it
const ROLE = { model: '候选因子', state: '状态', view: '情景变量', audit: '只作审计' };
const ALIGN = { reference_month: '参考月', release_dated: '按发布月', month_end: '月末', monthly_average: '月均', last_week: '每月最后一周',
  mixed: '分段（见说明）' };
const BASIS = { vs_B0: '数据修正：10 类资产样本外 R² 平均相对本轮之前的基准', alone_on_B1: '共同改动：单独加在修正数据后的基准上（走步检验里的消融）',
  asset_on_core: '单个资产：这个资产的 R² 在新核心之上的变化' };
// critique tables: model changes by candidate id (the diagnostics labels), factor changes and MLP seed sets by their research keys
const FACTOR_ROW = { 'spread_fixed replaces f_spread': '利差计时修正（替换旧利差）', 'GZ added': '加 GZ 信用利差',
  'GZ replaces f_spread': 'GZ 替换核心利差', 'tp10_x_infl4 added': '加 期限溢价 × 上月 CPI > 4%',
  'tp10_x_core4 added': '加 期限溢价 × 上月核心 CPI > 4%', 'tp10_x_infl3 added': '加 期限溢价 × 上月 CPI > 3%',
  'drop labor_share': '去掉劳动收入占比（插值版）', 'EBP added': '加 超额债券溢价', 'real_pce_nondurables added': '加 实际非耐用品消费',
  'U4 (spread_fixed, GZ, tp10_x_infl4, no labor share)': 'U4：利差修正 + GZ + 期限溢价 × 上月 CPI > 4%，去掉劳动收入占比',
  'metals_oil (energy only)': '基础金属相对油价（只进能源股）', 'usd_em (industrials only)': '美元对新兴市场货币（只进工业股）',
  'gold equation trained from 1982 (window)': '黄金方程从 1982 年起估计' };
// candidate sets of the penalised fits (their counts come from upper_bound.methods[].n_candidates)
const METHOD_SET = { A: (n) => `${n} 个全历史候选`, B: (n) => `${n} 个候选（含晚开始的）`, C: (n, b) => `再加 ${isNum(n) && isNum(b) ? n - b : '—'} 个价格序列（不可用，只作对照）` };
const METHOD_FIT = { ridge: '岭回归', lasso: 'Lasso', pcr: '主成分回归' };
// archived research text may name the one-year forecast's state list by its code name
const plain = (s) => (typeof s === 'string' ? s.replace(/ ?KEY_STATES ?/g, '一年期预测的现有状态') : s);
const ENGINE_ROW = { base: '现行（不加预测状态）', re_loans_yoy: '+ 银行房地产贷款同比', labor_share_rt: '+ 劳动收入占比（按发布月）',
  china_import_share: '+ 自中国进口占比' };

export const AUDIT_CSS = `/* diagnostics-audit */
.diagnostics-audit-kpis .kpi-value { font-size: 20px; }
.diagnostics-audit details.card > summary { cursor: pointer; list-style: none; display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center; }
.diagnostics-audit details.card > summary::-webkit-details-marker { display: none; }
.diagnostics-audit details.card > summary::before { content: '▸'; color: var(--ink-3); font-size: 12px; }
.diagnostics-audit details.card[open] > summary::before { content: '▾'; }
.diagnostics-audit details.card > summary b { font-size: 14px; }
.diagnostics-audit details.card[open] > summary { margin-bottom: 10px; }
.diagnostics-audit .card + .card, .diagnostics-audit details.card + details.card { margin-top: 12px; }
.diagnostics-audit ul.list li { line-height: 1.65; }
.diagnostics-audit .tri { white-space: nowrap; }
.diagnostics-audit .foot { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; line-height: 1.6; }
.diagnostics-audit table.tbl td.wrap { min-width: 120px; }
.diagnostics-audit table.tbl td.wrap .chip { white-space: normal; }
.diagnostics-audit .chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
`;

// R^2 changes: "+0.016 / +0.089 / +0.038", each in its sign colour (the K-line setting).
function tri(vs, d = 3) {
  const xs = arr(vs);
  if (!xs.length || !xs.some(isNum)) return '—';
  return h('span', { class: 'tri' }, xs.flatMap((v, i) => [i ? ' / ' : '', h('span', { class: ui.signCls(v, 1) }, num(v, d, true))]));
}
// R^2 levels, no colour: "0.319 → 0.470"
const triText = (vs, d = 3) => (arr(vs).some(isNum) ? arr(vs).map(v => num(v, d)).join(' / ') : '—');
const deltaCell = (v) => h('span', { class: ui.signCls(v, 1) }, r2s(v));

export function renderAudit(box, res, ctx) {
  const { dg = {}, scen = null, label = (id) => id, short = (a) => a, minGain = 0.003, nA = 10, sections = {}, param = null,
    finDescribe = null } = ctx || {};
  box.replaceChildren();
  if (res?.e) {
    box.append(res.e.status === 404
      ? ui.notice('这个构建没有缺失因子审计结果：运行 python -m scenariolab.build --only factor_audit（约 30 秒）', 'info')
      : ui.notice(`读取缺失因子审计结果失败：${res.e.message || res.e}`, 'error'));
    return;
  }
  const fa = obj(res?.d);
  const S = obj(fa.summary), score = obj(S.score);
  const tag = typeof fa.archive?.tag === 'string' && fa.archive.tag ? fa.archive.tag : '研究轮存档';
  const archChip = (short_ = false) => (short_ ? h('span', { class: 'chip accent', title: tag }, '研究轮存档') : ui.chip(tag, 'accent'));
  const redoneChip = () => ui.chip('本次构建重算', 'good');
  const srcChip = (src) => (src === 'archive' ? archChip(true) : src === 'recomputed' ? redoneChip() : null);
  const its = arr(dg.iterations);
  const DEC = its.length >= 2 ? [its[0].validate, ...its.map(it => it.test)].slice(0, 3).map(p => `${String(p[0]).slice(0, 4)}–${String(p[1]).slice(0, 4)}`)
    : ['1995–2004', '2005–2014', '2015–2026'];
  const T3 = DEC[2];
  const byCand = Object.fromEntries(arr(dg.candidates).map(c => [c.id, c]));
  const colName = Object.fromEntries(arr(fa.coverage).map(c => [c.col, c.name]));
  const vname = (col) => colName[col] || (ui.varName(col) !== col ? ui.varName(col) : col);
  const names = { ...CAND };
  for (const r of [...arr(fa.refutation), ...arr(fa.plausible), ...arr(fa.forecast_states?.plausible)]) {
    if (r?.candidate && r.name && !names[r.candidate]) names[r.candidate] = r.name;
  }
  const cand = (id) => names[id] || String(id || '—');
  const target = (t) => (TARGET[t] || (String(t).startsWith('ret_') ? `${short(String(t).slice(4))}回报` : t));
  // scenario names without their subtitle (「系统性金融危机：2008 年式」 -> 「系统性金融危机」)
  const scenName = Object.fromEntries(arr(scen?.scenarios).map(s => [s.id, String(s.name || s.id).split(/[：:]/)[0].trim() || s.id]));
  const anchor = (key, text) => { const el = h('div', { class: 'section-title diagnostics-anchor' }, text); sections[key] = el; return el; };
  const gap = () => h('div', { style: { height: '16px' } });
  const out = [];

  // ---------------------------------------------------------------- consistency with the page
  const model = obj(fa.model);
  const issues = arr(fa.issues).filter(x => typeof x === 'string' && x);
  if (finDescribe && model.describe && model.describe !== finDescribe) {
    issues.unshift(`审计按「${model.describe}」计算，诊断页的最终模型是「${finDescribe}」：两者不是同一次构建，重新运行 python -m scenariolab.build --only factor_audit。`);
  }
  if (issues.length) out.push(ui.notice(issues.join(' '), ''), gap());

  // ---------------------------------------------------------------- A. 结论
  const ch = obj(S.changes);
  const k0 = isNum(ch.data_fixes) && isNum(ch.data_fixes_changing_model) ? ch.data_fixes - ch.data_fixes_changing_model : null;
  const sB0 = arr(score.B0), sB1 = arr(score.B1), sF = arr(score.final);
  const bound = arr(S.honest_bound_2015_2026);
  const lead = [
    `预测不准，是不是因为少了专门的因子？研究轮用 ${S.families ?? '—'} 个家族测试了 ${int(S.n_transmission)} 个同月传导候选（含按发布月对齐的版本）和 ${int(S.n_predictive)} 个已知时点的预测状态。`,
    `跨家族多重检验（BH，q = ${BH_Q.toFixed(2)}）、反驳检验和走步验证之后，改变模型的是 ${ch.data_fixes_changing_model ?? '—'} 项数据修正、${ch.common ?? '—'} 项共同改动和 ${ch.dedicated ?? '—'} 项单个资产的改动`,
    k0 ? `（另 ${k0} 项数据修正只影响候选和情景视图）` : '',
    `：${T3} 年的样本外 R² 从 ${num(sB0[2])} 提高到 ${num(sF[2])}。`,
    bound.length === 2 ? `上限很低：把全部可用候选放进惩罚回归，${T3} 年也只有 ${num(bound[0], 2)}–${num(bound[1], 2)}；` : '',
    isNum(S.unexplained_share_2015_2026) ? `平均约 ${pct(S.unexplained_share_2015_2026)} 的月度价格波动解释不了，主要是市场自身的估值和风险偏好变化、行业新闻和危机月份。引擎用抽中月份的残差承载这一部分。` : '',
  ].join('');
  out.push(h('p', { class: 'lead', style: { maxWidth: 'none' } }, lead));
  out.push(h('div', { class: 'grid cols-4 diagnostics-audit-kpis' },
    ui.kpi('测试的候选', `${int(S.n_transmission)} + ${int(S.n_predictive)}`,
      `同月传导 ${int(S.n_transmission)} 个、预测状态 ${int(S.n_predictive)} 个，${S.families ?? '—'} 个家族`),
    ui.kpi('改变模型的改动', `${ch.data_fixes_changing_model ?? '—'} · ${ch.common ?? '—'} · ${ch.dedicated ?? '—'}`,
      `数据修正${k0 ? `（另有 ${k0} 项只影响候选）` : ''} · 共同改动 · 单个资产`),
    ui.kpi(`样本外 R²（${T3}）`, `${num(sB0[2])} → ${num(sF[2])}`,
      `本轮之前的基准 → 最终模型；修正数据后的基准 ${num(sB1[2])}${S.n_test_months_T3 ? `；${S.n_test_months_T3} 个月` : ''}`),
    ui.kpi('解释不了的部分', pct(S.unexplained_share_2015_2026), `${T3} 年 ${nA} 类平均：1 − max(最终模型, 研究轮按 ${T3} 年自己选惩罚的上限)`)));
  out.push(h('p', { class: 'foot' }, '候选与检验的数量、惩罚回归的上限',
    bound.length === 2 ? `（${num(bound[0], 2)}–${num(bound[1], 2)}）` : '', '和“解释不了的部分”所用的作弊上限来自', archChip(true),
    '；三个十年的 R² 由本次构建在当前面板上重算', fa.data_version?.panel_sha1 ? `（面板 ${fa.data_version.panel_sha1}）` : '', '。'));

  // ---------------------------------------------------------------- B. 改动清单
  const changes = arr(fa.changes);
  const fixes = Object.fromEntries(arr(fa.data_fixes).map(d => [d.id, d]));
  const typeOf = (c) => {
    if (KIND[c.kind]) return KIND[c.kind];
    const cc = byCand[c.walkforward_id || c.id];
    if (cc?.group === 'missing_mechanism') return '共同机制';
    if (cc?.change?.drop_factors) return '共同改动';
    return '共同因子';
  };
  const bhText = (c) => {
    if (c.id === 'release_dating') {
      const r = fixes[c.id]?.evidence?.look_ahead_corr_archive;
      return isNum(r) ? `前视已确认（季初月与本季变化相关 ${r.toFixed(6)}）` : '前视已确认';
    }
    const q = obj(c.replication?.q_archive);
    const pass = Object.entries(q).filter(([a, v]) => a !== 'score' && isNum(v) && v <= BH_Q).sort((p, r) => p[1] - r[1]);
    const sq = isNum(q.score) && q.score <= BH_Q ? q.score : null;
    if (!pass.length && sq === null) return c.kind === 'common' && c.replication == null ? '不显著' : c.replication ? '没有通过' : '—';
    if (pass.length <= 2 && sq === null) return `BH 通过（${pass.map(([a, v]) => `${short(a)} q ${num(v)}`).join('、')}）`;
    return `BH 通过（${pass.length} 类资产${sq !== null ? `；${nA} 类平均 q ${num(sq)}` : ''}）`;
  };
  const refText = (c) => {
    const r = c.refutation;
    if (r && typeof r === 'object') {
      return h('span', {}, `滚动窗口 ${r.rolling?.positive ?? '—'}/${r.rolling?.n ?? '—'}；剔除危机 `, tri(r.ex_crisis),
        `；安慰剂 p ${num(r.placebo_p)}`);
    }
    if (fixes[c.id]?.changes_model === false) return '只影响候选和视图';
    if (c.flagged) return '在同一批十年上发现';
    const loo = arr(c.loo_in_common).length ? c.loo_in_common : c.loo_in_final;
    if (c.kind === 'common' && arr(loo).length === 3) {
      const drops = !!byCand[c.walkforward_id || c.id]?.change?.drop_factors;
      return h('span', {}, `${drops ? '去掉它的贡献' : '它的贡献'}（留一法，共同模型里）：`,
        ...DEC.map((d, i) => [i ? '、' : '', `${d} `, deltaCell(loo[i])]));
    }
    return '—';
  };
  const statusOf = (c) => {
    if (c.status === 'fixed') return ui.chip('已修正', 'good');
    if (c.status === 'not_in_model') return ui.chip('未进入模型');
    if (c.status !== 'adopted') return ui.chip(c.status || '—');
    if (c.kind === 'dedicated') return ui.chip(`专门改动阶段采用${c.flagged ? '（标注）' : ''}`, c.flagged ? 'warn' : 'good');
    const wi = arr(c.walkforward?.iterations);
    const first = wi.find(x => x.selected);
    const rides = wi.some(x => isNum(x.ablation?.test_delta) && x.ablation.test_delta <= minGain);
    return ui.chip(`${first ? `第 ${first.iter} 轮` : ''}采用${rides ? '（随整体检验通过）' : ''}`, 'good');
  };
  const bases = [...new Set(changes.map(c => c.table?.basis).filter(b => BASIS[b]))];
  out.push(gap(), card({ title: '改动清单：哪些改动进了模型',
    sub: `每行一项改动；三个十年是样本外 R² 的变化（${DEC.join('、')}），颜色随 K 线配色表示升降；多重检验一列取自研究轮存档，其余由本次构建重算；反驳检验的“剔除危机”去掉 2008–09 与 2020 年，“安慰剂”是循环平移安慰剂。` },
  ui.table([
    { key: 'name', label: '改动', wrap: true, minWidth: 150, fmt: (v, c) => h('span', {}, v || c.id, c.flagged ? [' ', ui.chip('标注', 'warn')] : null) },
    { key: 'kind', label: '类型', fmt: (v, c) => typeOf(c) },
    ...DEC.map((d, i) => ({ value: (c) => c.table?.values?.[i], label: d, fmt: (v) => (isNum(v) ? deltaCell(v) : '—') })),
    { key: 'replication', label: h('span', {}, '多重检验 ', archChip(true)), wrap: true, minWidth: 150, fmt: (v, c) => bhText(c) },
    { key: 'refutation', label: '反驳检验', wrap: true, minWidth: 170, fmt: (v, c) => refText(c) },
    { key: 'status', label: '状态', wrap: true, minWidth: 96, fmt: (v, c) => statusOf(c) },
  ], changes, { stack: true }),
  bases.length ? h('p', { class: 'foot' }, `${bases.map(b => BASIS[b]).join('；')}。`) : null));

  // ---------------------------------------------------------------- C. 每项改动
  const perAsset = Object.fromEntries(arr(fa.per_asset).map(r => [r.asset, r]));
  const sbRows = arr(dg.stage_b?.iterations).flatMap(it => arr(it.rows).map(r => ({ ...r, iter: it.iter })));
  out.push(h('div', { class: 'section-title' }, '每项改动：定义、走步检验和注意事项'));
  for (const c of changes) out.push(changeCard(c));

  // ---------------------------------------------------------------- the plausible candidates that were not adopted
  const plaus = arr(fa.plausible).filter(p => p && (p.name || p.id));
  if (plaus.length) {
    out.push(gap(), card({ title: `可信但没有采用的 ${plaus.length} 个传导候选`, actions: archChip(true),
      sub: '通过了大部分检验，但没有通过跨家族多重检验或有具体的问题；这一轮只在审计里报告，不进入模型。' },
    ui.table([
      { key: 'name', label: '候选', wrap: true, minWidth: 140 },
      { key: 'series', label: '序列与时点', wrap: true, minWidth: 180 },
      { key: 'role_if_adopted', label: '如果采用', wrap: true, minWidth: 100 },
      { key: 'numbers', label: '研究轮的数字', wrap: true, minWidth: 200 },
      { key: 'why_not', label: '为什么没有采用', wrap: true, minWidth: 220 },
    ], plaus, { stack: true })));
  }

  function changeCard(c) {
    const fx = fixes[c.id];
    const kv = [['定义', c.definition], ['时点', c.alignment], ['覆盖', c.coverage],
      fx?.before ? ['修正前', fx.before] : null, fx?.after ? ['修正后', fx.after] : null].filter(x => x && x[1] && x[1] !== '—');
    const body = [h('dl', { class: 'kv' }, kv.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]))];
    // the walk-forward: stage A rows for common changes, stage B rows for dedicated ones
    const wf = obj(c.walkforward);
    const lines = [];
    if (c.kind === 'common') {
      let inStart = false;
      for (const it of arr(wf.iterations)) {
        const parts = [];
        if (it.selected) parts.push(`第 ${it.round} 步采纳（验证期平均 R² ${num(it.val_score)}）`);
        else parts.push(inStart ? '已在本轮的起点模型里' : '没有被选中');
        if (isNum(it.round1_delta) && it.selected) parts.push(`第 1 步单独尝试时 ${r2s(it.round1_delta)}`);
        if (it.ablation) parts.push(`单独加在基准上：验证期 ${r2s(it.ablation.val_delta)}、测试期 ${r2s(it.ablation.test_delta)}（${it.ablation.test_months ?? '—'} 个月）`);
        if (it.selected) {
          parts.push(`本轮测试：选中模型比起点 ${r2s(it.test_delta_vs_start)}，${it.test_improved ? '保留' : '没有保留'}`);
          if (it.test_improved) inStart = true;
        }
        lines.push(`第 ${it.iter} 轮：${parts.join('；')}`);
      }
    } else if (c.kind === 'dedicated') {
      for (const r of arr(wf.stage_b)) {
        const lab = sbRows.find(x => x.id === c.id && x.option === r.option && x.iter === r.iter)?.label || label(c.id);
        lines.push(`第 ${r.iter} 轮 · ${lab}：这个资产的验证期 R² ${r2s(r.val_gain)}、测试期 ${r2s(r.test_gain)}${r.accepted ? '，通过' : '，没有选这个选项'}`);
      }
    }
    if (lines.length) body.push(h('div', { class: 'muted', style: { fontSize: '12px', margin: '10px 0 4px' } }, '走步检验'), h('ul', { class: 'list' }, lines.map(t => h('li', {}, t))));
    // per asset: the assets the change touches
    let assets = [];
    if (c.kind === 'dedicated' && c.asset) assets = [c.asset];
    else if (c.kind === 'data_fix') assets = Object.keys(obj(fx?.per_asset));
    else assets = Object.keys(obj(c.replication?.per_asset));
    const rep = obj(c.replication?.per_asset);
    if (assets.length) {
      const cols = [{ key: 'a', label: '资产', fmt: (a) => short(a) }];
      if (c.kind === 'data_fix') cols.push({ value: (r) => fx.per_asset[r.a], label: '相对本轮之前的基准', fmt: (v) => tri(v) });
      else {
        cols.push({ value: (r) => perAsset[r.a]?.r2?.B1, label: '修正数据后的基准', fmt: (v) => triText(v) },
          { value: (r) => perAsset[r.a]?.r2?.final, label: '最终模型', fmt: (v) => triText(v) });
        if (c.kind === 'dedicated' && c.dedicated) cols.push({ value: () => c.dedicated.gain, label: '在新核心上的变化', fmt: (v) => tri(v) });
      }
      if (Object.keys(rep).length) {
        cols.push({ value: (r) => ['T1', 'T2', 'T3'].map(t => rep[r.a]?.[t]?.delta), label: h('span', {}, '同样本增益（复现研究轮）'), fmt: (v) => tri(v) });
      }
      body.push(h('div', { class: 'muted', style: { fontSize: '12px', margin: '10px 0 4px' } },
        `各资产样本外 R²（${DEC.join(' / ')}）`), ui.table(cols, assets.map(a => ({ a })), { stack: true }));
    }
    // leave one out: a dedicated change's is its own asset's R^2; the 10-asset average is a tenth of it
    if (arr(c.loo_in_final).length === 3 || arr(c.loo_in_common).length === 3) {
      const own = c.kind === 'dedicated';
      body.push(h('p', { class: 'diagnostics-text' },
        own ? `留一法（有这项改动减去没有它，${short(c.asset)}自己的 R² 变化）：` : `留一法（有这项改动减去没有它，${nA} 类资产平均）：`,
        arr(c.loo_in_final).length === 3 ? ['在最终模型里 ', tri(c.loo_in_final)] : null,
        arr(c.loo_in_final).length === 3 && arr(c.loo_in_common).length === 3 ? '；' : '',
        arr(c.loo_in_common).length === 3 ? ['在共同模型里 ', tri(c.loo_in_common)] : null, '。',
        own && arr(c.dedicated?.score_gain).length === 3 ? [`对 ${nA} 类资产平均的影响（加在新核心上）：`, tri(c.dedicated.score_gain), '。'] : null));
    }
    if (c.dedicated_alt && c.dedicated) {
      body.push(h('p', { class: 'diagnostics-text' }, '另一个预先登记的选项：', tri(c.dedicated_alt.gain), '（没有选）。'));
    }
    if (c.active_months) {
      const am = obj(c.active_months);
      const tr = am.T1?.train_active;
      body.push(h('p', { class: 'diagnostics-text' }, `高核心通胀月份：训练期 ${tr ?? '—'} 个；测试期 ${['T1', 'T2', 'T3'].map((t, i) => `${DEC[i]} ${am[t]?.test_active ?? '—'} 个`).join('、')}。`,
        todayTp()));
    }
    if (c.id === 'add_gz' && fa.today?.gz) {
      const g = fa.today.gz;
      body.push(h('p', { class: 'diagnostics-text' }, `今天（${g.month}）GZ 信用利差 ${num(g.value, 2)}，1973 年以来第 ${isNum(g.pct_since_1973) ? Math.round(g.pct_since_1973 * 100) : '—'} 百分位（最高 ${num(g.max, 2)}，${g.max_month}）。`));
    }
    // caveats; those quoting archived numbers (caveat_sources) carry the archive tag
    const cavSrc = arr(c.caveat_sources);
    const cav = arr(c.caveats).map((t, i) => [t, cavSrc[i]]).filter(([t]) => typeof t === 'string');
    if (cav.length) {
      body.push(h('div', { class: 'muted', style: { fontSize: '12px', margin: '10px 0 4px' } }, '注意'),
        h('ul', { class: 'list' }, cav.map(([t, s]) => h('li', {}, t, s === 'archive' ? [' ', archChip(true)] : null))));
    }
    return h('details', { class: 'card' },
      h('summary', {}, h('b', {}, c.name || c.id), ui.chip(typeOf(c)), statusOf(c), c.verdict && VERDICT[c.verdict] ? ui.chip(...VERDICT[c.verdict]) : null,
        c.table?.values ? h('span', { class: 'muted', style: { fontSize: '12px' } }, tri(c.table.values)) : null),
      ...body);
  }

  function todayTp() {
    const t = obj(fa.today);
    if (typeof t.tp_term_on !== 'boolean') return '';
    return `今天（${t.month}）期限溢价项${t.tp_term_on ? '打开' : '关闭'}（核心 CPI 同比 ${num(t.core_yoy, 2)}%，${t.core_yoy_month}）；模拟用的历史月份里 ${t.tp_term_active_months_monthdb ?? '—'}/${t.monthdb_months ?? '—'} 个打开，最近一次 ${t.tp_term_last_on || '—'}。`;
  }

  // ---------------------------------------------------------------- D. 六个家族
  const fams = arr(fa.families);
  const tv = obj(S.transmission), pv = obj(S.predictive);
  if (fams.length) {
    const vt = (f, k, ks) => { const n = f.tested_n?.[k]; const v = obj(f.verdicts?.[k]); return `${n ?? '—'}：${ks.map(x => v[x] ?? 0).join(' / ')}`; };
    out.push(gap(), anchor('audit-families', `${fams.length} 个家族测了什么`), card({ title: '研究家族与结论', actions: archChip(),
      sub: '同月传导：确认 / 可信 / 否定；预测状态：可信 / 否定。“确认”= 通过跨家族多重检验和全部反驳检验。' },
    ui.table([
      { key: 'name', label: '家族', wrap: true, minWidth: 90 },
      { key: 'tested', label: '测了什么', wrap: true, minWidth: 150 },
      { value: (f) => f, label: '同月传导（确认 / 可信 / 否定）', fmt: (f) => (f.total ? f.tr : vt(f, 'transmission', ['CONFIRMED', 'PLAUSIBLE', 'REFUTED'])) },
      { value: (f) => f, label: '预测状态（可信 / 否定）', fmt: (f) => (f.total ? f.pr : vt(f, 'predictive', ['PLAUSIBLE', 'REFUTED'])) },
      { key: 'headline', label: '主要结论', wrap: true, minWidth: 140 },
      { key: 'failures', label: '没通过的主要原因', wrap: true, minWidth: 140, fmt: (v) => (arr(v).length ? arr(v).join('；') : '—') },
    ], [...fams, { name: '合计', tested: '', total: true, headline: '', failures: [],
      tr: `${int(S.n_transmission)}：${['CONFIRMED', 'PLAUSIBLE', 'REFUTED'].map(k => tv[k] ?? 0).join(' / ')}`,
      pr: `${int(S.n_predictive)}：${['PLAUSIBLE', 'REFUTED'].map(k => pv[k] ?? 0).join(' / ')}` }], { stack: true })));
  }

  // ---------------------------------------------------------------- E. 按资产
  const pa = arr(fa.per_asset);
  if (pa.length) {
    out.push(gap(), anchor('audit-assets', '按资产：每类资产缺什么'), card({ title: `各资产 ${T3} 年的样本外 R²`,
      sub: `“解释不了的份额” = 1 − max(最终模型, 研究轮按 ${T3} 年自己选惩罚的上限)，${nA} 类平均 ${pct(S.unexplained_share_2015_2026)}；候选数、BH 通过的候选和还没测的驱动来自研究轮存档，R² 由本次构建重算${S.n_test_months_T3 ? `（${S.n_test_months_T3} 个月）` : ''}。`,
      actions: archChip(true) },
    ui.table([
      { key: 'asset', label: '资产', fmt: (a, r) => r.name || short(a) },
      { key: 'n_best', label: '以它为最佳资产的候选' },
      { key: 'bh_survivor_names', label: '通过 BH 的候选', wrap: true, minWidth: 160, fmt: (v) => (arr(v).length ? arr(v).join('、') : '无') },
      { value: (r) => r.r2, label: '本轮之前的基准 → 最终模型', fmt: (v) => `${num(v?.B0?.[2])} → ${num(v?.final?.[2])}`,
        cls: (v) => (isNum(v?.B0?.[2]) && isNum(v?.final?.[2]) ? ui.signCls(v.final[2] - v.B0[2], 1) : '') },
      { key: 'unexplained', label: '解释不了的份额', fmt: (v) => (!isNum(v) ? '—' : v >= 0.995 ? '≈ 1.0' : num(v, 2)) },
      { key: 'untested', label: '还没测的专门驱动', wrap: true, minWidth: 160, fmt: (v) => (arr(v).length ? arr(v).join('、') : '—') },
    ], pa, { stack: true })));
  }

  // ---------------------------------------------------------------- F. 多重检验
  const bh = obj(fa.bh);
  if (Object.keys(bh).length) {
    const tp = obj(bh.transmission_pairs), sc = obj(bh.score), raw = obj(bh.raw_dm_would_add), pr = obj(bh.prediction), cw = obj(bh.cw_size_on_placebos);
    const scRows = arr(sc.rows).filter(r => r.reject);
    const cal = arr(pr.calibrated_rows);
    out.push(gap(), h('div', { class: 'section-title' }, '多重检验：几千个检验里留下了什么'),
      h('div', { class: 'grid cols-2 text' },
        card({ title: '同月传导', actions: archChip(true) },
          h('p', { class: 'diagnostics-text', style: { marginTop: 0 } },
            `候选 × 资产：${int(tp.n)} 个检验（同样本 Diebold–Mariano，三个十年 Stouffer 合并，BH q = ${BH_Q.toFixed(2)}），通过 ${tp.rejections ?? '—'} 个；`,
            `${nA} 类资产平均：${int(sc.n)} 个，通过 ${sc.rejections ?? '—'} 个${scRows.length ? `（${scRows.map(r => `${cand(r.candidate)} q ${num(r.q)}`).join('、')}）` : ''}。`),
          isNum(raw.n) ? h('p', { class: 'diagnostics-text' },
            `如果用晚开始的序列直接对比全窗口估计的基准，会多出 ${raw.n} 个“通过”${isNum(raw.gold) ? `，其中 ${raw.gold} 个是黄金` : ''}：较短的训练窗口被算成了序列的功劳。审计一律用同样本比较。`) : null),
        card({ title: '预测状态', actions: archChip(true) },
          h('p', { class: 'diagnostics-text', style: { marginTop: 0 } },
            `候选 × 目标：${int(pr.n)} 个检验。Clark–West 通过 ${pr.cw ?? '—'} 个，均方误差检验（DM）${pr.dm ?? '—'} 个，`,
            `用 ${cw.n_placebos ?? '—'} 个持续性安慰剂校准后的 Clark–West ${pr.calibrated ?? '—'} 个`,
            cal.length ? `（${cal.map(r => `${cand(r.candidate)} → ${target(r.target)}，q ${num(r.q_calibrated)}`).join('；')}）` : '', '。'),
          isNum(cw.min) ? h('p', { class: 'diagnostics-text' },
            `在持续性回归元上，Clark–West 在 5% 水平的拒绝率是 ${pct(cw.min)}–${pct(cw.max)}（黄金 ${pct(cw.gold)}、医疗 ${pct(cw.healthcare)}、CPI ${pct(cw.cpi)}）：它偏大，所以要用安慰剂校准。`) : null)),
      gap(), card({ title: `同月传导里通过 BH 的 ${arr(tp.rows).length} 个候选 × 资产`, actions: archChip(true),
        sub: 'z = 各十年同样本 DM t 的 Stouffer 合并（训练月份至少 120 个的十年）；“平均增益”是同一批月份上的 R² 变化。' },
      ui.table([
        { key: 'candidate', label: '候选', wrap: true, minWidth: 150, fmt: (v) => cand(v) },
        { key: 'asset', label: '资产', fmt: (a) => (a === 'score' ? `${nA} 类平均` : short(a)) },
        { key: 'mean_delta_ss', label: '平均增益', fmt: (v) => deltaCell(v) },
        { key: 'decades_positive', label: '为正的十年', fmt: (v, r) => `${v ?? '—'}/${r.k ?? '—'}` },
        { key: 'z', label: 'z', fmt: (v) => num(v, 2) },
        { key: 'q', label: 'q', fmt: (v) => num(v) },
      ], arr(tp.rows), { stack: true })),
      arr(pr.rows).length ? h('details', { class: 'card', style: { marginTop: '12px' } },
        h('summary', {}, h('b', {}, `预测状态里 Clark–West 通过的 ${arr(pr.rows).length} 个候选 × 目标`), archChip(true)),
        ui.table([
          { key: 'candidate', label: '候选', wrap: true, minWidth: 150, fmt: (v) => cand(v) },
          { key: 'target', label: '目标', wrap: true, fmt: (v) => target(v) },
          { value: (r) => [r.r2_incr_all, r.r2_incr_h1, r.r2_incr_h2], label: '增量 R²（全部 / 前半 / 后半）', fmt: (v) => tri(v) },
          { key: 'q_cw', label: 'CW q', fmt: (v) => num(v) },
          { key: 'q_dm', label: 'DM q', fmt: (v) => num(v) },
          { key: 'q_calibrated', label: '校准后 q', fmt: (v) => num(v), cls: (v) => (isNum(v) && v <= BH_Q ? 'is-good' : '') },
        ], arr(pr.rows), { stack: true })) : null);
  }

  // ---------------------------------------------------------------- G. 反驳检验
  const rf = arr(fa.refutation);
  if (rf.length) {
    out.push(gap(), card({ title: '反驳检验：换个窗口、错一个月、打乱时间以后还在不在',
      sub: `${nA} 类资产平均（括号里的资产是只看它自己）；“早一个月”是前视安慰剂，提升应该消失；循环平移安慰剂 p 越小越好。复现的几行由本次构建重算，其余来自研究轮存档。` },
    ui.table([
      { key: 'name', label: '候选', wrap: true, minWidth: 150, fmt: (v, r) => h('span', {}, v || cand(r.candidate), ' ', srcChip(r.source)) },
      { key: 'decades', label: '三个十年', fmt: (v) => tri(v) },
      { key: 'ex_crisis', label: '剔除 2008–09 与 2020', fmt: (v) => tri(v) },
      { key: 'rolling', label: '滚动 10 年窗口为正', fmt: (v) => (v ? `${v.positive}/${v.n}` : '—') },
      { key: 'lag', label: '晚一个月', fmt: (v) => tri(v) },
      { key: 'lead', label: '早一个月', fmt: (v) => tri(v) },
      { key: 'placebo_p', label: '安慰剂 p', fmt: (v) => num(v) },
      { key: 'verdict', label: '判定', fmt: (v) => (VERDICT[v] ? ui.chip(...VERDICT[v]) : v || '—') },
    ], rf, { stack: true })));
  }

  // ---------------------------------------------------------------- H. 预测状态
  const fs = obj(fa.forecast_states);
  if (Object.keys(fs).length) {
    const eng = obj(fs.engine), base = obj(eng.base);
    const alt = Object.entries(eng).filter(([k, v]) => k !== 'base' && v && typeof v === 'object');
    const narrow = alt.filter(([, v]) => isNum(v.chosen_ens) && isNum(base.chosen_ens) && v.chosen_ens < base.chosen_ens);
    const maxT = Math.max(0, ...alt.flatMap(([, v]) => ['risk_crps', 'rank_ic'].map(k => Math.abs(v.paired_576?.[k]?.t)).filter(isNum)));
    // how far ahead house prices lead (the archived target's horizon, e.g. 「住房 CPI 的 18 个月变化」)
    const hpiLead = /(\d+) 个月/.exec(String(arr(fs.plausible).find(r => r.candidate === 'housing:hpi_fm')?.target_name || ''))?.[1];
    const text = [`没有一个预测状态进入一年期预测。加进一年期预测的现有状态后，按构建所用的带宽（有效起点 ${base.chosen_ens ?? '—'}），风险资产 CRPS 和排序 IC 都没有可测的变化（配对差的 |t| 最大 ${num(maxT, 2)}）`,
      narrow.length ? `；验证规则因此改选更窄的带宽（${[...new Set(narrow.map(([, v]) => v.chosen_ens))].join('、')}），检验期反而更差（${narrow.map(([, v]) => num(v.test_crps_chosen, 4)).join('、')} 对 ${num(base.test_crps_chosen, 4)}）` : '',
      `。用户关心的渠道（白领工资、资本端过热、家庭收入）在数据里通过通胀慢慢起作用：劳动收入占比领先一年后的 CPI 变化，房价领先${hpiLead ? ` ${hpiLead} 个月` : ''}后的住房和服务通胀；它们不在同一个月推动资产价格。现在它们是情景变量（见概念缺口）。`].join('');
    out.push(gap(), h('div', { class: 'section-title' }, '预测状态：能不能帮一年期预测'),
      card({ title: '可信的预测状态（没有一个进入一年期预测）', actions: archChip(true),
        sub: '增量 R² 是相对一年期预测现有状态的样本外提升，三个数是全部 / 前半 / 后半；校准 p 用持续性安慰剂校准 Clark–West。' },
      ui.table([
        { key: 'name', label: '预测状态', wrap: true, minWidth: 140 },
        { key: 'known_at', label: '已知时点', wrap: true, minWidth: 140 },
        { key: 'target_name', label: '目标', wrap: true, minWidth: 110 },
        { key: 'r2_incr', label: '增量 R²', fmt: (v) => tri(v) },
        { key: 'cw_p_calibrated', label: '校准 p', fmt: (v) => pFmt(v) },
        { key: 'robustness', label: '稳健性', wrap: true, minWidth: 200, fmt: (v) => plain(v) || '—' },
        { key: 'panel_column', label: '情景变量', wrap: true, fmt: (v) => (v ? vname(v) : '—') },
      ], arr(fs.plausible), { stack: true })),
      arr(fs.replicated).length ? h('div', { class: 'diagnostics-text' }, redoneChip(), ' 本次构建在月度起点上重算：',
        arr(fs.replicated).flatMap((r, i) => [i ? '；' : '', `${r.name}：`, tri(r.r2_incr), `（CW p ${pFmt(r.cw_p)}）`]), '。') : null,
      ...(alt.length ? [gap(), card({ title: '加进一年期预测以后', actions: archChip(true),
        sub: `每行把一个预测状态加进一年期预测的现有状态，重跑一年期预测；配对差是同一带宽（有效起点 ${base.chosen_ens ?? '—'}）下逐个起点的差（t 统计量）。` },
      ui.table([
        { key: 'id', label: '版本', wrap: true, fmt: (v) => ENGINE_ROW[v] || cand(v) },
        { key: 'chosen_ens', label: '选中的带宽' },
        { key: 'risk_crps', label: '风险资产 CRPS', fmt: (v) => num(v, 4) },
        { key: 'rank_ic', label: '排序 IC', fmt: (v) => num(v, 3) },
        { key: 'cpi_crps', label: 'CPI CRPS', fmt: (v) => num(v, 3) },
        { key: 'test_crps_chosen', label: '检验期 CRPS', fmt: (v) => num(v, 4) },
        { value: (r) => r.paired_576?.risk_crps, label: '配对差：CRPS', fmt: (v) => (v ? `${num(v.mean_diff, 5, true)}（t ${num(v.t, 2)}）` : '—') },
        { value: (r) => r.paired_576?.rank_ic, label: '配对差：IC', fmt: (v) => (v ? `${num(v.mean_diff, 4, true)}（t ${num(v.t, 2)}）` : '—') },
      ], [{ id: 'base', ...base }, ...alt.map(([id, v]) => ({ id, ...v }))], { stack: true }),
      h('p', { class: 'diagnostics-text' }, text))] : []));
  }

  // ---------------------------------------------------------------- I. 能解释多少
  const ub = obj(fa.upper_bound), rs = obj(fa.residuals);
  const ubRows = arr(ub.per_asset).filter(r => r && r.asset && r.asset !== 'mean');      // the mean row is recomputed below
  if (ubRows.length || Object.keys(rs).length) {
    out.push(gap(), h('div', { class: 'section-title' }, '能解释多少：上限和剩下的部分'));
    const meth = obj(ub.methods);
    const nB = meth.B_lasso?.n_candidates ?? meth.B_ridge?.n_candidates;
    if (ubRows.length) {
      const anat = obj(ub.residual_anatomy?.assets);
      const cols = [['base_T3', '原基准'], ['core_fixed_T3', '固定核心'], ['useful_U5_T3', '手选集合 U5'], ['A_lasso_common', '诚实上限（Lasso）'],
        ['oracle_AB', '作弊上限'], ['C_lasso_common(prices)', '加上市场价格']];
      const rows = ubRows.map(r => ({ ...r, pd: anat[r.asset]?.T3_bound_resid_pd_r2, fin: perAsset[r.asset]?.r2?.final?.[2] }));
      const mean = (k) => { const v = rows.map(r => r[k]).filter(isNum); return v.length === rows.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
      const meanRow = { asset: '_mean', ...Object.fromEntries([...cols.map(([k]) => k), 'pd', 'fin'].map(k => [k, mean(k)])) };
      // the honest range: the six honest fits (sets A and B x ridge, lasso, PCR) with the penalty chosen on the
      // 10-asset score; the mean row shows the archive's range of the 10-asset scores
      const range = (v, r) => (r.asset === '_mean' ? (bound.length === 2 ? `${num(bound[0], 2)}–${num(bound[1], 2)}` : '—')
        : arr(v).length === 2 && arr(v).every(isNum) ? `${r2s(v[0])}..${r2s(v[1])}` : '—');
      const ci = cols.findIndex(([k]) => k === 'A_lasso_common') + 1;
      const colDefs = cols.map(([k, l]) => ({ key: k, label: l, fmt: (v) => r2s(v) }));
      if (rows.some(r => arr(r.honest_range_T3).length === 2)) {
        colDefs.splice(ci, 0, { key: 'honest_range_T3', label: '诚实范围（6 种拟合）', fmt: range });
      }
      out.push(card({ title: `各资产 ${T3} 年：今天的因子最多能解释多少`, actions: archChip(true),
        sub: `“诚实上限”= 固定核心加上全部全历史候选，惩罚在前两个十年上选定、只评一次；“诚实范围”是两组候选 × 岭回归、Lasso、主成分回归这 6 种拟合（惩罚按 ${nA} 类平均选）；“作弊上限”按 ${T3} 年自己选惩罚；“加上市场价格”加入 P/D、CAPE、VIX、风格因子，只说明价格能解释价格；最后一列是本次构建的最终模型（重算）。` },
      ui.table([
        { key: 'asset', label: '资产', fmt: (a) => (a === '_mean' ? `${nA} 类平均` : short(a)) },
        ...colDefs,
        { key: 'pd', label: '上限残差的 P/D R²', fmt: (v) => num(v, 2) },
        { key: 'fin', label: '最终模型', fmt: (v) => r2s(v) },
      ], [...rows, meanRow], { stack: true }),
      h('p', { class: 'diagnostics-text' }, `把 ${nB ?? '全部'} 个候选一起放进去，找到的和付出的差不多：诚实惩罚下 ${T3} 年 ${bound.length === 2 ? `${num(bound[0], 2)}–${num(bound[1], 2)}` : '—'}，`,
        `固定核心 ${num(S.fixed_core_T3)}，手选集合 ${num(S.useful_set_T3)}，按 ${T3} 年自己选惩罚（作弊上限）也只有 ${num(S.oracle)}；本次的最终模型 ${num(sF[2])}。`)));
      const mrows = Object.entries(meth).filter(([, v]) => v && typeof v === 'object').map(([k, v]) => ({ k, ...v }));
      if (mrows.length) {
        out.push(h('details', { class: 'card', style: { marginTop: '12px' } }, h('summary', {}, h('b', {}, `${mrows.length} 种惩罚回归的结果`), archChip(true)),
          ui.table([
            { key: 'k', label: '候选集合 · 方法', wrap: true, minWidth: 160, fmt: (k, r) => { const [s, f] = String(k).split('_'); return `${s}（${METHOD_SET[s] ? METHOD_SET[s](r.n_candidates, meth.B_lasso?.n_candidates) : ''}）· ${METHOD_FIT[f] || f}`; } },
            { key: 'n_candidates', label: '候选数' },
            ...DEC.map((d, i) => ({ key: `T${i + 1}`, label: d, fmt: (v) => r2s(v) })),
            { key: 'T3_131', label: `${T3}（剔除 2020 年 3–12 月）`, fmt: (v) => r2s(v) },
          ], mrows, { stack: true })));
      }
    }
    const B0 = obj(rs.B0), FR = obj(rs.final);
    if (Object.keys(FR).length) {
      const top = Object.values(obj(FR.top10_share_2015_2026)).filter(isNum);
      const sse = obj(FR.sse_kept_vs_base);
      out.push(gap(), card({ title: '残差里剩下什么', actions: redoneChip(),
        sub: `每年 1 月重新估计、只用之前数据的样本外残差，${FR.first || '—'} 至 ${FR.last || '—'}（${FR.months ?? '—'} 个月）；${nA} 类资产标准化后做主成分。` },
      ui.table([
        { key: 'm', label: '指标', wrap: true, minWidth: 200 },
        { key: 'b', label: '本轮之前的基准' },
        { key: 'f', label: '最终模型' },
      ], [
        { m: '残差第一主成分的方差占比', b: num(B0.pc1_share), f: num(FR.pc1_share) },
        { m: '第一主成分与美股残差的相关', b: num(B0.pc1_corr_us_equity_resid), f: num(FR.pc1_corr_us_equity_resid) },
        { m: '第一主成分与当月 P/D 变化的相关', b: num(B0.pc1_corr_pd_change), f: num(FR.pc1_corr_pd_change) },
        { m: `当月 P/D 变化对美股残差的 R²（全部 / ${T3}）`, b: `${num(B0.pd_r2_us_equity)} / ${num(B0.pd_r2_us_equity_2015_2026)}`,
          f: `${num(FR.pd_r2_us_equity)} / ${num(FR.pd_r2_us_equity_2015_2026)}` },
      ], { stack: true }),
      Object.keys(sse).length ? h('p', { class: 'diagnostics-text' }, '最终模型保留的残差平方和（相对本轮之前的基准）：',
        Object.entries(sse).map(([a, v]) => `${short(a)} ${num(v, 2)}`).join('、'), '。') : null,
      h('p', { class: 'diagnostics-text' }, `剩下的主要是市场本身的估值和风险溢价变化：残差第一主成分仍占 ${pct(FR.pc1_share)}，和当月市场 P/D 变化的相关 ${num(FR.pc1_corr_pd_change, 2)}；`,
        `P/D 变化能解释美股残差的 ${pct(FR.pd_r2_us_equity)}，但它由价格本身构成，不能当因子。`,
        top.length ? `十个最大的月份占 ${T3} 年残差平方和的 ${pct(Math.min(...top))}–${pct(Math.max(...top))}。` : '',
        '引擎用抽中月份的残差把这一部分带进模拟。',
        arr(ub.beta_drift?.by_decade_range).length === 2 ? [`即使事后按十年重新估计系数，也只能再减少 ${pct(ub.beta_drift.by_decade_range[0])}–${pct(ub.beta_drift.by_decade_range[1])} 的残差（调整后 R²，研究轮手选集合 U5 的残差）`, archChip(true), '。'] : null),
      rs.note ? h('p', { class: 'foot' }, rs.note) : null));
    }
  }

  // ---------------------------------------------------------------- J. 模型还是因子
  const mvf = obj(fa.model_vs_factor), oou = obj(fa.out_of_universe), proto = obj(fa.protocol);
  if (Object.keys(mvf).length || Object.keys(oou).length) {
    out.push(gap(), h('div', { class: 'section-title' }, '改模型还是改因子'));
    const mb = obj(mvf.model_on_base), mu = obj(mvf.model_on_U4);
    const v3 = (x) => (x ? [x.T1, x.T2, x.T3] : null);
    const r05 = v3(mb['ridge_0.05']);
    const seeds = Object.entries(obj(mvf.mlp_seeds)).map(([k, v]) => {
      const m = /^(\w+?)_mlp(\d+)_seeds\((.*)\)$/.exec(k);
      return { k, base: m ? (m[1] === 'base' ? '本轮之前的基准' : m[1]) : k, hidden: m ? m[2] : '', seeds: m ? m[3] : '', v: v3(v) };
    });
    const u4 = seeds.filter(s => s.base === 'U4' && s.v);
    const later = u4.length && u4.every(s => s.v[1] > 0 && s.v[2] > 0);
    const t1mixed = u4.length && u4.some(s => s.v[0] > 0) && u4.some(s => s.v[0] < 0);
    const sent = [r05 ? `因子修正的作用远大于模型改动：最好的模型改动（${label('ridge_0.05')}）在本轮之前的基准上只有 ${r05.map(r2s).join(' / ')}` : '因子修正的作用远大于模型改动',
      later ? `；换上新因子之后，小型神经网络在 ${DEC[1]}、${DEC[2]} 年对所有种子都有提升${t1mixed ? `，但 ${DEC[0]} 年随种子正负不定` : ''}` : '',
      '，而且会让界面失去每个因子的 β，所以本次只作对照。'].join('');
    const left = [];
    if (Object.keys(mb).length) {
      left.push(card({ title: '模型改动：加在本轮之前的基准上 vs 加在新因子（U4）上', actions: archChip(true),
        sub: `${DEC.join(' / ')} 的 10 类平均样本外 R² 变化；U4 = 利差修正 + GZ + 期限溢价 × 上月 CPI > 4%，去掉劳动收入占比。` },
      ui.table([
        { key: 'k', label: '模型改动', fmt: (k) => label(k) },
        { value: (r) => v3(mb[r.k]), label: '在基准上', fmt: (v) => tri(v) },
        { value: (r) => v3(mu[r.k]), label: '在 U4 上', fmt: (v) => tri(v) },
      ], Object.keys(mb).map(k => ({ k })), { stack: true }), h('p', { class: 'diagnostics-text' }, sent)));
    }
    const fb = obj(mvf.factor_on_base);
    if (Object.keys(fb).length) {
      left.push(card({ title: '因子改动：单独加在本轮之前的基准上', actions: archChip(true) },
        ui.table([
          { key: 'k', label: '因子改动', wrap: true, minWidth: 180, fmt: (k) => FACTOR_ROW[k] || k },
          { value: (r) => v3(fb[r.k]), label: DEC.join(' / '), fmt: (v) => tri(v) },
        ], Object.keys(fb).map(k => ({ k })), { stack: true })));
    }
    if (left.length) out.push(h('div', { class: 'grid cols-2 text' }, left));
    const right = [];
    if (seeds.length) {
      right.push(card({ title: '小型神经网络换随机种子', actions: archChip(true), sub: '每行三个种子平均；同一个神经网络在不同种子下的正负不一致，说明增益不稳。' },
        ui.table([
          { key: 'base', label: '加在' },
          { key: 'hidden', label: '隐藏单元' },
          { key: 'seeds', label: '种子' },
          { key: 'v', label: DEC.join(' / '), fmt: (v) => tri(v) },
        ], seeds, { stack: true })));
    }
    const fc = obj(oou.final_common), inds = Object.entries(obj(fc.industries));
    if (inds.length) {
      right.push(card({ title: `模型没用过的 ${fc.n ?? inds.length} 个行业：新的共同模型是否也更好`, actions: redoneChip(),
        sub: `Fama–French 行业（总回报），本轮之前的基准 vs 新的共同模型（${fc.spec || '—'}），${DEC.join(' / ')}；没有用这些行业做过任何选择。` },
      ui.table([
        { value: (r) => r[1].name, label: '行业' },
        { value: (r) => r[1].base, label: '本轮之前的基准', fmt: (v) => triText(v) },
        { value: (r) => r[1].final_common, label: '新的共同模型', fmt: (v) => triText(v) },
      ], inds, { stack: true }),
      h('p', { class: 'diagnostics-text' }, `平均提升 `, tri(fc.delta_vs_base), `；更好的行业 ${arr(fc.n_better).map(n => `${n}/${fc.n ?? inds.length}`).join(' / ')}。`,
        oou.archive_U4 ? ['研究轮的 U4（期限溢价项用 CPI 而不是核心 CPI）：', tri(oou.archive_U4.delta_vs_base), ' ', archChip(true)] : null)));
    }
    if (right.length) out.push(gap(), h('div', { class: 'grid cols-2 text' }, right));
    const alts = arr(proto.alternatives);
    if (alts.length) {
      const protoCols = [
        { key: 'name', label: '协议', wrap: true, minWidth: 180 },
        { key: 'iteration2', label: '第 2 轮选中', wrap: true, minWidth: 160, fmt: (v) => (arr(v).length ? arr(v).map(label).join('、') : '—') },
        { key: 'final', label: '最终的共同模型', wrap: true, minWidth: 160 },
        { value: (r) => r, label: `${T3} 年相对起始模型`, fmt: (r) => (isNum(r.t3_vs_start)
          ? h('span', {}, deltaCell(r.t3_vs_start), `（${num(r.t3_selected)} 对 ${num(r.t3_start)}，${r.n_months ?? '—'} 个月）`) : r.kept === 'start' ? '保留起始模型' : '—') },
      ];
      const archAlts = arr(proto.archive_alternatives);
      out.push(gap(), card({ title: '换一种走步检验的协议会怎样', actions: srcChip(proto.source),
        sub: `本次的协议：${proto.gate === 'incremental' ? '新改动要胜过本轮起始模型才保留' : '新改动只和基准比'}，神经网络${proto.mlp_eligible ? '可以被选中' : '只作对照'}。${proto.note ? `${proto.note}。` : ''}` },
      ui.table(protoCols, alts, { stack: true }),
      proto.source === 'recomputed' && archAlts.length ? h('details', { style: { marginTop: '10px' } },
        h('summary', { style: { cursor: 'pointer', fontSize: '13px' } }, '研究轮当时的结果 ', archChip(true)),
        ui.table(protoCols, archAlts, { stack: true })) : null));
    }
  }

  // ---------------------------------------------------------------- K. 概念缺口
  const cs = arr(fa.concepts);
  if (cs.length) {
    const cc = obj(fa.concept_counts);
    out.push(gap(), anchor('audit-concepts', '概念缺口：情景想表达的东西，数据里有没有'), card({
      title: `${cs.length} 个概念`, actions: archChip(true),
      sub: '“依赖它的情景”是内置情景里用到这个概念的；“新增的情景变量”可以在情景实验室里直接给视图。' },
    h('div', { class: 'chips', style: { marginBottom: '10px' } },
      STATUS_ORDER.filter(k => isNum(cc[k])).map(k => ui.chip(`${STATUS[k][0]} ${cc[k]}`, STATUS[k][1]))),
    ui.table([
      { key: 'name', label: '概念', wrap: true, minWidth: 150, fmt: (v, r) => `${r.num ? `${r.num}. ` : ''}${v || '—'}` },
      { key: 'specs', label: '依赖它的情景', wrap: true, minWidth: 120, fmt: (v) => (arr(v).length ? arr(v).map(id => scenName[id] || id).join('、') : '—') },
      { key: 'series_tested', label: '测过的序列', wrap: true, minWidth: 150 },
      { key: 'verdict', label: '结论', wrap: true, minWidth: 180 },
      { key: 'status', label: '状态', fmt: (v, r) => ui.chip(r.status_name || STATUS[v]?.[0] || v || '—', STATUS[v]?.[1] || '') },
      { key: 'new_columns', label: '新增的情景变量', wrap: true, minWidth: 140,
        fmt: (v) => (arr(v).length ? h('a', { href: '#lab', title: '情景实验室的变量表' }, arr(v).map(vname).join('、')) : '—') },
    ], cs, { stack: true })));
  }

  // ---------------------------------------------------------------- L. 口径与已知问题
  const notes = arr(fa.notes).filter(x => typeof x === 'string');
  const untested = arr(fa.untested), noSrc = arr(fa.no_source);
  const cov = arr(fa.coverage);
  // the selected model's own inputs (common factors and each asset's extras) are the transmission's columns
  const modelCols = new Set([...arr(model.spec?.factors), ...Object.values(obj(model.spec?.asset_factors)).flatMap(arr)].map(f => `f_${f}`));
  const roleOf = (v, r) => (v === 'model' && modelCols.has(r.col) ? '传导模型' : ROLE[v] || v || '—');
  out.push(gap(), h('div', { class: 'section-title' }, '审计的口径与已知问题'), h('div', { class: 'grid cols-2 text' },
    card({ title: '口径' }, notes.length ? h('ul', { class: 'list' }, notes.map(n => h('li', { class: 'muted' }, n))) : h('p', { class: 'muted' }, '—')),
    card({ title: '还没做的' },
      untested.length ? [h('div', { class: 'muted', style: { fontSize: '12px' } }, '便宜、有机制、还没测的'), h('ul', { class: 'list' }, untested.map(n => h('li', {}, n)))] : null,
      noSrc.length ? [h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '8px' } }, '没有允许的数据源'), h('ul', { class: 'list' }, noSrc.map(n => h('li', {}, n)))] : null,
      arr(fa.archive?.docs).length ? h('p', { class: 'foot' }, `研究轮的报告（只有统计结果，没有原始数据）随项目文档一起保存，共 ${fa.archive.docs.length} 份。`) : null)));
  if (cov.length) {
    out.push(h('details', { class: 'card', style: { marginTop: '12px' } },
      h('summary', {}, h('b', {}, `新增和修正的 ${cov.length} 个面板列：覆盖与时点`), redoneChip()),
      ui.table([
        { key: 'name', label: '列', wrap: true, minWidth: 160 },
        { key: 'role', label: '用途', fmt: roleOf },
        { key: 'first', label: '起', fmt: (v, r) => h('span', {}, v || '—', r.late ? [' ', ui.chip('晚开始', 'warn')] : null) },
        { key: 'last', label: '止' },
        { key: 'share_since_1973_02', label: '1973-02 以来有值', fmt: (v) => pct(v) },
        { key: 'alignment', label: '时点', fmt: (v, r) => `${ALIGN[v] || v || '—'}${isNum(r.lag_months) ? `（滞后 ${r.lag_months} 个月）` : ''}` },
        { key: 'revised', label: '会修订', fmt: (v) => (v ? '是' : '否') },
        { key: 'copyright', label: '版权方', fmt: (v) => v || '—' },
      ], cov, { stack: true })));
  }

  box.append(h('div', { class: 'diagnostics-audit' }, out));
  if (param && String(param).startsWith('audit') && sections[param]) setTimeout(() => sections[param].scrollIntoView({ block: 'start' }), 80);
}
