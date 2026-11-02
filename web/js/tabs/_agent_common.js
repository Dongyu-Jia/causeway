// Shared helpers of the 智能体推演 tab (a helper module, not a tab): styles, number formats, actor colours and
// emblems, regime badges, field and fact names, account-line labels, small DOM bits.
// Conventions (UI.md §1.2): a signed return uses .pos / .neg (K-line colours), a probability is always the accent
// blue, a judgement uses .is-good / .is-bad or warn / bad chips. LLM and backend text reaches the DOM only as text.
import * as ui from '../ui.js';

export const { h } = ui;
// DOM append / replaceChildren turn null into the text "null": these skip null, undefined and booleans
const keep = (k) => k.flat(Infinity).filter(x => x !== null && x !== undefined && x !== false && x !== true);
export const put = (el, ...k) => { el.append(...keep(k)); return el; };
export const fill = (el, ...k) => { el.replaceChildren(...keep(k)); return el; };
export const pre = (el, ...k) => { el.prepend(...keep(k)); return el; };
export const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// ---------------------------------------------------------------- styles
// One <style id="style-agents"> for the agent modules; each module appends its block once.
export function ensureStyle(key, css) {
  let el = document.getElementById('style-agents');
  if (!el) { el = document.createElement('style'); el.id = 'style-agents'; document.head.append(el); }
  const marker = `/* ${key} */`;
  if (!el.textContent.includes(marker)) el.textContent += `\n${marker}\n${css}`;
}

// ---------------------------------------------------------------- numbers
const fixed = (x, d) => { const s = x.toFixed(d); return Number(s) === 0 ? (0).toFixed(d) : s; };
// decimals 0.052 -> "5.2%"; signed adds "+" to positive values
export const pct = (x, d = 1, signed = false) => {
  if (!isNum(x)) return '—';
  const s = fixed(x * 100, d);
  return `${signed && Number(s) > 0 ? '+' : ''}${s}%`;
};
// values already in percent (3.75 -> "3.75%")
export const pp = (x, d = 2, signed = false) => {
  if (!isNum(x)) return '—';
  const s = fixed(x, d);
  return `${signed && Number(s) > 0 ? '+' : ''}${s}%`;
};
export const num = (x, d = 2, signed = false) => {
  if (!isNum(x)) return '—';
  const s = fixed(x, Math.abs(x) >= 1000 ? 0 : d);
  const t = Number(s).toLocaleString('zh-CN', { minimumFractionDigits: Math.abs(x) >= 1000 ? 0 : d, maximumFractionDigits: Math.abs(x) >= 1000 ? 0 : d });
  return `${signed && Number(s) > 0 ? '+' : ''}${t}`;
};
// a probability: "23%", "4.2%", "0.4%", "<0.1%"
export function prob(p) {
  if (!isNum(p)) return '—';
  if (p <= 0) return '0%';
  if (p < 0.001) return '<0.1%';
  if (p < 0.1) return `${fixed(p * 100, 1)}%`;
  return `${fixed(p * 100, 0)}%`;
}
// The tree stores log total returns: the simple return is exp(x) - 1.
export const simple = (logret) => (isNum(logret) ? Math.expm1(logret) : null);
export const retText = (logret, d = 1) => pct(simple(logret), d, true);
export const retCls = (logret) => ui.signCls(simple(logret));
export function seconds(s) {
  if (!isNum(s)) return '—';
  if (s < 60) return `${Math.round(s)} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${Math.round(s % 60)} 秒`;
  return `${Math.floor(s / 3600)} 小时 ${Math.round((s % 3600) / 60)} 分`;
}
export function minutesRange(m) {
  if (!Array.isArray(m) || !m.length) return '—';
  const lo = m[0], hi = m[m.length - 1];
  if (hi >= 120) return `约 ${fixed(lo / 60, 1)}–${fixed(hi / 60, 1)} 小时`;
  return `约 ${Math.max(1, Math.round(lo))}–${Math.max(1, Math.round(hi))} 分钟`;
}
export const when = (t) => String(t || '').replace('T', ' ').slice(0, 16);
// "2027Q1" -> "2027 年一季度"
const QN = ['', '一', '二', '三', '四'];
export function quarterZh(q) {
  const m = /^(\d{4})Q([1-4])$/.exec(String(q || ''));
  return m ? `${m[1]} 年${QN[Number(m[2])]}季度` : String(q || '');
}

// ---------------------------------------------------------------- actors
// Colours from the app palette minus the two K-line hexes (an actor never reads as "up" or "down") and not the
// accent blue (that one means probability). Emblems are drawn in _agent_art.js.
// Actor colours stay away from the accent blue (probability) and the K-line red and green (returns).
export const ACTOR_META = {
  voter: { zh: '选民', color: '#f08c00', emblem: 'voter' },
  courts: { zh: '法院', color: '#862e9c', emblem: 'courts' },
  belligerents: { zh: '冲突方', color: '#8d6e3f', emblem: 'ship' },
  president: { zh: '总统与白宫', color: '#b8860b', emblem: 'podium' },
  congress: { zh: '国会', color: '#343a40', emblem: 'dome' },
  treasury: { zh: '财政部', color: '#0b7285', emblem: 'vault' },
  china: { zh: '中国', color: '#e8590c', emblem: 'crane' },
  opec: { zh: '欧佩克+', color: '#5f3dc4', emblem: 'derrick' },
  bigtech: { zh: '超大型科技公司', color: '#1098ad', emblem: 'chip' },
  fed: { zh: '美联储', color: '#6b7280', emblem: 'columns' },
  nature: { zh: '自然', color: '#868e96', emblem: 'cloud' },
};
const SPARE = ['#a07a1f', '#7048e8', '#5c5f66', '#8f5e3b', '#b04a9b', '#0b7285'];
const extra = new Map();
export function actorColor(id) {
  if (ACTOR_META[id]) return ACTOR_META[id].color;
  if (!extra.has(id)) extra.set(id, SPARE[extra.size % SPARE.length]);
  return extra.get(id);
}
export const emblemOf = (id) => ACTOR_META[id]?.emblem || 'columns';
const CAT = { actors: new Map(), facts: {}, synth: null, elections: {} };
const OPTIONS = new Map();                // option id -> Chinese label (any actor), for prose
export function setCatalogue(cfg) {
  const c = cfg?.catalogue || {};
  CAT.actors = new Map((c.actors || []).map(a => [a.id, a]));
  CAT.gates = cfg?.synth?.gates || {};
  OPTIONS.clear();
  for (const a of c.actors || []) for (const d of a.domains || []) for (const o of d.options || []) if (!OPTIONS.has(o.id)) OPTIONS.set(o.id, o.zh || o.id);
  CAT.facts = c.facts || {};
  CAT.elections = c.elections || {};
  CAT.synth = cfg?.synth || null;
  CAT.nature = c.nature || null;
}
export const catalogueActor = (id) => CAT.actors.get(id) || null;
export const actorName = (id) => CAT.actors.get(id)?.zh || ACTOR_META[id]?.zh || '未命名行为者';
export const actorStage = (id) => (isNum(CAT.actors.get(id)?.stage) ? CAT.actors.get(id).stage : 9);
export const actorIds = () => [...CAT.actors.keys()];
export const elections = () => CAT.elections || {};
// option id -> its catalogue label for an actor (fallback: the id is never shown, a neutral word is)
export function optionLabel(actor, opt) {
  const a = CAT.actors.get(actor);
  for (const d of a?.domains || []) for (const o of d.options || []) if (o.id === opt) return o.zh;
  return opt ? '（未列出的选项）' : '—';
}
export const KIND_ZH = { policy: '政策', calendar: '日历', state: '状态触发', corporate: '企业', code: '代码' };

// Expression of an emblem's face from the tone of the chosen option (decorative; the text carries the meaning).
export function toneOf(opt, statusQuo) {
  const s = String(opt || '');
  if (statusQuo || /hold|no_new_law|no_response|refunding_unchanged|status_quo|uphold|stalemate|_hold$/.test(s)) return 'neutral';
  if (/escalate|hike|slash|standoff|shutdown|blockade|retaliate|strike_down|attempt_removal|consolidation|cut$|quota_cut|tighten|layoffs|flips|opposition|price_war|lapse/.test(s)) return 'stern';
  if (/cut_|ceasefire|stimulus|truce|deal|raise|suspend|ease|monetize|expand|wins|holds|tariff_down|release/.test(s)) return 'happy';
  return 'neutral';
}

// ---------------------------------------------------------------- consequences, facts, fields
export const BRANCH_KIND = { central: '模型中枢', partition: '分区', mechanism: '机制', default: '默认', nature_event: '自然事件' };
const AXIS_GROUP = { growth: '增长', investment: '投资', demand: '消费', inflation: '通胀', expectations: '通胀预期',
  supply: '供应链', sentiment: '信心', fin_conditions: '金融条件', funding: '融资市场', equity_leg: '估值' };
export function axisName(axisField) {
  const axes = CAT.synth?.axes || {};
  for (const [id, ax] of Object.entries(axes)) if ((ax.fields || []).includes(axisField) || ax.alt === axisField) return AXIS_GROUP[id] || '';
  return '';
}
const ACCOUNT_FACTS = { arrears: '联邦出现逾期未付（技术性违约）', debt_limit_binding: '债务上限约束（非常规措施中）',
  xdate_passed: '越过 X 日期（逾期未付）' };
export function factLabel(id) {
  return CAT.facts?.[id]?.zh || ACCOUNT_FACTS[id] || null;
}
// standing facts of a record as [id, label] (metadata keys such as *_expiry_q are skipped)
export function liveFacts(facts) {
  return Object.entries(facts || {}).filter(([k, v]) => v && v !== 0 && !/_q$/.test(k))
    .map(([k, v]) => [k, factLabel(k) ? (typeof v === 'number' && v !== 1 && v !== true ? `${factLabel(k)}（${v}）` : factLabel(k)) : null])
    .filter(([, t]) => t);
}

const FIELDS = new Map();
export function setSchema(schema) { for (const f of schema?.fields || []) FIELDS.set(f.id, f); }
export const fieldInfo = (id) => FIELDS.get(id) || null;
const STATE_ZH = { s_ff: '政策利率', s_unrate: '失业率', s_core_yoy: '核心 CPI 同比', s_spread: '信用利差', s_term: '期限利差',
  s_cape: '估值（log CAPE）', s_y10: '10 年期收益率', s_gz: 'GZ 信用利差', s_infexp1: '1 年通胀预期', cum_us_equity: '美股累计' };
export function fieldName(id) {
  const f = FIELDS.get(id);
  if (f?.zh) return String(f.zh).replace(/（[^）]*）$/, '').trim() || f.zh;
  return STATE_ZH[id] || '（未命名字段）';
}
export const fieldUnit = (id) => FIELDS.get(id)?.unit || '';
// Backend and LLM text may carry field ids ("按模型分布分区（d_unrate）"): show their Chinese names instead.
export function humanize(text) {
  return String(text ?? '').replace(/\b([dsx])_[a-z][a-z0-9_]*\b/g, (m) => (FIELDS.has(m) || STATE_ZH[m] ? fieldName(m) : m));
}
// guard reports and engine notes in prose: field ids, "actor/option", "actor：" and bare option ids become names
export function prose(text) {
  let s = humanize(text);
  s = s.replace(/\b([a-z]+)\/([a-z][a-z0-9_]*)\b/g, (m, a, o) => (CAT.actors.has(a) ? `${actorName(a)}·${optionLabel(a, o)}` : m));
  s = s.replace(/\b([a-z]+)(?=\s*[：:])/g, (m, a) => (CAT.actors.has(a) ? actorName(a) : m));
  s = s.replace(/\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/g, (m) => OPTIONS.get(m) || m);
  return s;
}
export const EFFORT_ZH = { low: '低', medium: '中', high: '高', xhigh: '很高', max: '最高' };
// a prior's source label without the research-file ids (W_A06, A11, …)
export const sourceLabel = (t) => String(t || '').replace(/[（(][^（()）]*\bW_[A-Z]\d+[^（()）]*[）)]/g, '')
  .replace(/\bW_[A-Z]\d+[、,，]?\s*/g, '').replace(/[（(]\s*[）)]/g, '').trim();
export const gateName = (g) => CAT.gates?.[g] || '后果合成器的门控条件';
// "actors-v2.1 (2026-11-01; …)" -> "v2.1（2026-11-01）"
export function versionLabel(v) {
  const m = /([vV]?\d+(?:\.\d+)*)\s*\((\d{4}-\d{2}-\d{2})/.exec(String(v || ''));
  return m ? `${m[1].startsWith('v') ? m[1] : `第 ${m[1]} 版`}（${m[2]}）` : (v ? cut(String(v), 24) : null);
}

// ---------------------------------------------------------------- regimes (badge)
// The 宏观体制 tab's six labels and colours. The badge here is a simplified quarterly rule on a node's medians:
// contraction = real GDP growth below 0 or unemployment 0.5 pp above its low of the last 4 quarters; severe =
// growth at or below -4% or a 2 pp rise; inflation = core CPI year on year above 4% (below 0: falling prices).
export const REGIMES = {
  expansion: { name: '扩张', color: '#a5d8ff', ink: '#1d2433' },
  inflation: { name: '通胀', color: '#fab005', ink: '#1d2433' },
  deflation: { name: '物价负增长', color: '#15aabf', ink: '#fff' },
  recession: { name: '衰退', color: '#9775fa', ink: '#fff' },
  depression: { name: '深度衰退', color: '#343a40', ink: '#fff' },
  stagflation: { name: '滞胀', color: '#b35c00', ink: '#fff' },
};
export const REGIME_RULE = '体制徽章用季末中位数的简化规则：实际 GDP 增长为负，或失业率比近 4 季低点高 0.5 个百分点算收缩'
  + '（增长 ≤ −4% 或升 2 个百分点算深度收缩）；核心 CPI 同比高于 4% 算高通胀。与「宏观体制」页的月度规则不同，只作提示。';
// rec: a record or skeleton (rep, bands); unrates: unemployment medians of this and up to 4 earlier quarters
export function regimeOf(rep, gdp, unrates = []) {
  const u = rep?.s_unrate, core = rep?.s_core_yoy;
  const hist = unrates.filter(isNum);
  const low = hist.length ? Math.min(...hist) : u;
  const rise = isNum(u) && isNum(low) ? u - low : 0;
  const contraction = (isNum(gdp) && gdp < 0) || rise >= 0.5;
  const severe = (isNum(gdp) && gdp <= -4) || rise >= 2;
  let id = 'expansion';
  if (contraction && isNum(core) && core > 4) id = 'stagflation';
  else if (contraction && severe) id = 'depression';
  else if (contraction) id = 'recession';
  else if (isNum(core) && core < 0) id = 'deflation';
  else if (isNum(core) && core > 4) id = 'inflation';
  return { id, ...REGIMES[id] };
}
export function regimeBadge(rg, extra = '') {
  return h('span', { class: `agents-regime ${extra}`, style: { background: rg.color, color: rg.ink }, title: REGIME_RULE }, rg.name);
}

// ---------------------------------------------------------------- political state
const POLITICAL = { pres_party: '白宫', house: '众议院', senate: '参议院', senate_seats: '参议院席位', house_margin: '众议院席位差',
  chair_tilt: '美联储主席倾向', independence_risk: '央行独立性风险', trade_regime: '关税体制', tariff_authority: '关税授权',
  house_next: '下届众议院', pres_next: '下届白宫' };
const POLITICAL_VAL = { R: '共和党', D: '民主党', active: '活跃', calm: '平静', broad: '宽泛', struck_down: '被裁定无效',
  struck_down_refunds: '被裁定无效并退税', flip: '易手', hold: '保持' };
export function politicalRows(pol) {
  return Object.entries(pol || {}).filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => [POLITICAL[k] || '其他', POLITICAL_VAL[v] || String(v)]);
}

// ---------------------------------------------------------------- account lines (generated from accounts.LINES)
// statement codes: b 资产负债表, i 利润表, c 现金流量表, o 其他指标
export const STATEMENTS = { b: '资产负债表', i: '利润表', c: '现金流量表', o: '其他指标' };
export const ACCOUNT_BOOKS = {"fed": "美联储", "fiscal": "联邦政府（国会、财政部、总统共用）", "president": "总统与白宫（政治资本）", "china": "中国", "opec": "欧佩克+（以沙特为摆动产油国）", "bigtech": "超大型科技公司（五家合计，可分公司）", "voter": "选民（家庭部门的分布）", "courts": "法院（案卷，不是财务报表）", "belligerents": "冲突方（模板）", "households": "家庭部门", "nfc": "非金融企业", "banks": "银行", "investors": "投资者", "row": "外国部门", "sectoral": "部门余额（恒等式）"};
export const ACCOUNT_LINES = {"fed.total_assets":["总资产","b","$bn"],"fed.soma_treasuries":["持有国债","b","$bn"],"fed.soma_bills":["其中国库券","b","$bn"],"fed.soma_mbs":["持有 MBS","b","$bn"],"fed.other_assets":["其他资产（溢价、贷款等）","b","$bn"],"fed.currency":["流通中货币","b","$bn"],"fed.reserves":["银行准备金","b","$bn"],"fed.reverse_repo_total":["逆回购（外国官方 + 隔夜）","b","$bn"],"fed.tga":["财政部一般账户","b","$bn"],"fed.deferred_asset":["递延资产","b","$bn"],"fed.interest_income":["利息收入（扣除运营成本）","i","$bn/q"],"fed.interest_expense":["利息支出（准备金与逆回购）","i","$bn/q"],"fed.net_income":["净收入","i","$bn/q"],"fed.remittances":["上缴财政部","c","$bn/q"],"fed.rmp_purchases_q":["准备金管理购买","c","$bn/q"],"fed.inflation_gap":["通胀缺口","o","pp"],"fed.unemployment_gap":["就业缺口","o","pp"],"fiscal.deficit_nipa_gdp":["联邦净借款/GDP（NIPA，模型口径）","i","% GDP"],"fiscal.deficit_cash_q":["现金赤字（MTS）","i","$bn/q"],"fiscal.receipts_q":["收入","i","$bn/q"],"fiscal.outlays_q":["支出","i","$bn/q"],"fiscal.net_interest":["净利息","i","$bn/q"],"fiscal.customs_q":["关税收入","i","$bn/q"],"fiscal.customs_refunds":["关税退款","c","$bn/q"],"fiscal.fed_remittances":["美联储上缴","i","$bn/q"],"fiscal.debt_held_public":["公众持有债务","b","$bn"],"fiscal.intragov_holdings":["政府内部持有","b","$bn"],"fiscal.statutory_limit":["法定债务上限","b","$bn"],"fiscal.limit_headroom":["上限余量","b","$bn"],"fiscal.extraordinary_measures_used":["已用非常规措施","b","$bn"],"fiscal.tga":["TGA 现金","b","$bn"],"fiscal.arrears":["逾期未付（越过 X 日期后）","b","$bn"],"fiscal.avg_rate_marketable":["可交易国债平均利率","b","%"],"fiscal.bill_share":["国库券占比","b","%"],"fiscal.holders_fed":["美联储持有","b","$bn"],"fiscal.holders_foreign":["外国持有","b","$bn"],"fiscal.house_senate_control":["两院控制与席位差","o","seats"],"fiscal.appropriations_deadline":["拨款到期日","o","date"],"president.approval":["支持率","o","%"],"president.gasoline_price":["汽油零售价","o","$/gal"],"president.cpi_yoy":["CPI 同比","o","%"],"president.real_dpi_pc_yoy":["人均实际可支配收入同比","o","%"],"president.us_equity_drawdown":["美股距高点","o","%"],"president.effective_tariff_rate":["有效关税率","o","%"],"president.spr_inventory":["战略石油储备","o","mn bbl"],"president.tariff_authority":["关税授权状态","o","fact"],"china.fx_reserves":["外汇储备","b","$bn"],"china.us_treasury_holdings":["持有美债","b","$bn"],"china.current_account_gdp":["经常账户/GDP","i","% GDP"],"china.gg_deficit_gdp":["广义政府赤字/GDP","i","% GDP"],"china.gg_debt_gdp":["政府债务/GDP","b","% GDP"],"china.private_credit_gdp":["私人信贷/GDP","b","% GDP"],"china.real_house_price_vs_peak":["实际房价距峰值","b","%"],"china.growth_vs_target":["增长与目标","i","pp"],"china.cny_per_usd":["人民币中间价","o","CNY/USD"],"china.us_bilateral_trade":["对美贸易","i","$bn/12m"],"opec.brent":["布伦特","o","$/bbl"],"opec.production":["产量（沙特、集团）","o","mb/d"],"opec.spare_capacity_reachable":["可达市场的闲置产能","o","mb/d"],"opec.oil_revenue":["石油收入","i","$bn/q"],"opec.fiscal_balance":["财政余额（沙特）","i","% GDP"],"opec.fiscal_breakeven":["财政盈亏平衡油价","i","$/bbl"],"opec.reserves":["外汇储备（沙特）","b","$bn"],"opec.members_growth":["成员增长","i","%"],"opec.us_crude_production":["美国原油产量（竞争者）","o","mb/d"],"bigtech.revenue":["收入","i","$bn/q"],"bigtech.depreciation":["折旧","i","$bn/q"],"bigtech.operating_income":["经营利润","i","$bn/q"],"bigtech.interest":["利息支出","i","$bn/q"],"bigtech.cfo":["经营现金流","c","$bn/q"],"bigtech.capex":["资本开支","c","$bn/q"],"bigtech.lease_principal":["融资租赁本金","c","$bn/q"],"bigtech.fcf":["自由现金流","c","$bn/q"],"bigtech.buybacks":["回购","c","$bn/q"],"bigtech.dividends":["股息","c","$bn/q"],"bigtech.debt_issuance":["新增债务","c","$bn/q"],"bigtech.cash_securities":["现金与证券","b","$bn"],"bigtech.debt":["债务","b","$bn"],"bigtech.rpo":["未履约合同（RPO）","b","$bn"],"bigtech.headcount":["员工数","b","persons"],"voter.nw_distribution":["净资产分布","b","% NW"],"voter.real_dpi_pc_growth":["人均实际可支配收入增长","i","%"],"voter.saving_rate":["储蓄率","i","%"],"voter.debt_service":["偿债率","c","% DPI"],"voter.election_odds":["选举赔率","o","probability"],"courts.docket":["待决案件","o","list"],"courts.money_at_stake":["涉及金额","o","$bn"],"belligerents.oil_exports":["石油出口量与收入","i","mb/d, $bn/q"],"belligerents.fiscal_balance":["财政余额","i","% GDP"],"belligerents.reserves":["储备（含被冻结部分）","b","$bn"],"belligerents.military_spending":["军费","i","% GDP"],"belligerents.sanctions":["制裁与航道状态","o","fact"],"households.equities":["股票与基金","b","$bn"],"households.real_estate":["房地产","b","$bn"],"households.other_assets":["其他资产","b","$bn"],"households.liabilities":["负债","b","$bn"],"households.net_worth":["净资产","b","$bn"],"households.dpi":["可支配收入","i","$bn SAAR"],"households.pce":["消费","i","$bn SAAR"],"households.saving_rate":["储蓄率","i","%"],"households.debt_service_ratio":["偿债率","c","% DPI"],"nfc.profits_pbt":["税前利润","i","$bn SAAR"],"nfc.net_interest":["净利息","i","$bn SAAR"],"nfc.interest_coverage":["利息覆盖","i","x"],"nfc.capex":["资本开支","c","$bn SAAR"],"nfc.net_equity_issuance":["股票净发行","c","$bn SAAR"],"nfc.debt":["债务","b","$bn"],"banks.total_assets":["总资产","b","$bn"],"banks.reserves":["准备金","b","$bn"],"banks.securities":["证券","b","$bn"],"banks.unrealized_losses":["证券未实现损失","b","$bn"],"banks.loans":["贷款","b","$bn"],"banks.deposits":["存款","b","$bn"],"banks.nii":["净利息收入","i","$bn/q"],"banks.chargeoffs":["核销","i","% loans"],"investors.margin_debt":["保证金负债","b","$bn"],"investors.mmf_assets":["货币基金","b","$bn"],"investors.household_equity_share":["家庭股票占金融资产","b","%"],"investors.broker_dealer_assets":["券商资产","b","$bn"],"row.foreign_holdings":["外国持有美债","b","$bn"],"row.foreign_official":["其中官方","b","$bn"],"row.current_account_gdp":["经常账户/GDP","i","% GDP"],"row.niip":["国际投资净头寸","b","$bn"],"sectoral.government_nl":["政府净借出","o","$bn SAAR"],"sectoral.nation_nl":["国家对外净借出","o","$bn SAAR"],"sectoral.private_nl":["私人部门净借出","o","$bn SAAR"],"fed.iorb":["准备金利率","o","%"],"fed.unemployment":["失业率","o","%"],"fiscal.debt_gdp":["公众持有债务/GDP","b","% GDP"],"fiscal.interest_burden_nipa":["利息负担（NIPA 利息/GDP）","i","% GDP"],"fiscal.limit_binding_share":["债务上限约束的粒子占比","o","share"],"fiscal.xdate_flag":["越过 X 日期的粒子占比","o","share"],"china.fx_reserves_change_4q":["外汇储备 4 季变化","b","%"],"opec.brent_q_avg_last2":["布伦特近两季均价","o","$/bbl"],"opec.quota_change":["配额变动","o","mb/d"],"president.unemployment":["失业率","o","%"],"bigtech.net_cash":["净现金","b","$bn"],"bigtech.op_margin":["经营利润率","i","%"],"row.current_account_q":["经常账户（季度）","i","$bn/q"],"row.foreign_share":["外国持有占比","b","%"],"sectoral.private_nl_gdp":["私人部门净借出/GDP","o","% GDP"],"views.s_debt":["联邦债务/GDP（视图）","o","%"],"views.s_int_burden":["利息负担（视图）","o","%"],"views.s_hh_debt":["家庭债务/GDP（视图）","o","%"],"views.s_nfc_debt":["企业债务/GDP（视图）","o","%"],"views.s_foreign_hold":["外国持有美债占比（视图）","o","%"]};
export function accountLine(key) {
  const v = ACCOUNT_LINES[key];
  if (v) return { zh: v[0], stmt: v[1], unit: v[2] };
  return { zh: '（未命名科目）', stmt: 'o', unit: '' };
}
// money in $bn -> "1.2 万亿" / "3,450 亿" (the books are in billions of dollars)
export function bn(x) {
  if (!isNum(x)) return '—';
  const a = Math.abs(x);
  if (a >= 1000) return `${fixed(x / 1000, 2)} 万亿`;
  if (a >= 10) return `${Math.round(x * 10).toLocaleString('zh-CN')} 亿`;
  return `${fixed(x * 10, 1)} 亿`;
}
export function accountValue(x, unit) {
  if (!isNum(x)) return '—';
  if (unit === '$bn' || unit === '$bn/q' || unit === '$bn SAAR' || unit === '$bn/12m') return `${bn(x)}${unit === '$bn/q' ? '/季' : unit === '$bn SAAR' ? '（年化）' : ''}`;
  if (/%/.test(unit) || unit === 'pp') return `${fixed(x, 2)}${unit === 'pp' ? ' 个百分点' : unit === '%' ? '%' : ` ${unit}`}`;
  if (unit === 'share') return pct(x, 0);
  return `${num(x, 2)} ${unit || ''}`.trim();
}

// ---------------------------------------------------------------- small DOM bits
export function chip(text, kind = '', title) { const c = ui.chip(text, kind); if (title) c.title = title; return c; }
export function probBar(p, { max = 1, cls = '' } = {}) {
  const w = isNum(p) && max > 0 ? clamp(p / max, 0, 1) * 100 : 0;
  return h('div', { class: `bar-inline agents-pbar ${cls}` }, h('span', { style: { width: `${w.toFixed(1)}%` } }));
}
export function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(`agents:${key}`) || 'null');
    localStorage.setItem(`agents:${key}`, JSON.stringify(value));
  } catch (_) { /* private window or blocked storage: a convenience only */ }
  return value;
}
export function debounce(fn, ms = 400) {
  let t = 0;
  const d = (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  d.cancel = () => clearTimeout(t);
  return d;
}
export const reducedMotion = () => {
  try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (_) { return false; }
};
// "冲突方：停火或航道恢复；总统与白宫：广泛加征" -> [["冲突方", "停火或航道恢复"], ["总统与白宫", "广泛加征"]]
// a child's label is "<action cluster>｜<discrete cell>" when its branch was split by discrete state
export const clusterPart = (label) => String(label || '').split('｜')[0];
export function splitLabel(label) {
  return String(label || '').split(/[；｜]/).map(s => s.split('：')).filter(p => p.length >= 2).map(([a, ...b]) => [a.trim(), b.join('：').trim()]);
}
// the actions that differ between sibling cluster labels ("美联储：降息 25bp" when only the Fed differs)
export function diffLabel(label, siblings) {
  const mine = splitLabel(label);
  if (!mine.length) return String(label || '');
  const others = (siblings || []).filter(l => l && l !== label).map(splitLabel);
  if (!others.length) return mine.map(p => p.join('：')).join('；');
  const differs = mine.filter(([a, o]) => others.some(ps => !ps.some(([a2, o2]) => a2 === a && o2 === o)));
  return (differs.length ? differs : mine).map(p => p.join('：')).join('；');
}
// at most n CJK characters (Latin letters count about half)
export function cut(text, n) {
  const s = String(text ?? '');
  let units = 0, out = '';
  for (const ch of s) {
    units += /[\x00-\xff]/.test(ch) ? 0.55 : 1;
    if (units > n) return `${out}…`;
    out += ch;
  }
  return s;
}
