// Tickers (标的) on 组合评估 and 提取与破产概率 (a helper module, not a tab). The instrument layer turns every
// scenario path into 16 ticker columns next to the 13 asset classes (GET /api/instruments: the catalog;
// POST /api/instruments/project: one ticker on every path). What lives here:
//   portfolioEditor  one weights editor for asset classes and tickers, grouped (纳指系 · 债与金 · 红利与价值 · BDC ·
//                    MLP); 金价 is the gold asset class itself (one weight, two rows). By share or by amount (USD),
//                    presets, 导入持仓 (pasted holdings) and 我的组合 (saved in this browser only)
//   holdingsCard     the caveats of the tickers a result holds (chips), their start prices, wipe-outs
//   projection       标的推演: one ticker on every scenario's paths: all tickers × scenarios at a glance, the recipe
//                    card with its parameters, fans per scenario and mixed, the comparison table, cash per share and
//                    the probability of a cut, attribution, correlations
//   cashFlowView     提取页: year-1 cash against the withdrawal, coverage by year, payout-cut pressure
//   alphaCompare     提取页: the three α versions of the estimated tickers side by side
// Module state (TK) survives the K-line re-render and the switch between the two tabs.
import * as ui from '../ui.js';
import * as charts from '../charts.js';
import * as api from '../api.js';
import {
  debounce, liveRequest, errorText, isNum, esc, shortName, fmtProb, fmtShare, rankColor, rowsHeight, labelWidth,
  assetLabel, FALLBACK_PRESETS, injectStyle as injectWeightsStyle,
} from './_weights.js';

const INK = '#1d2433', INK2 = '#4a5468', INK3 = '#7a8396', LINE = '#e3e6ec', ACCENT = '#3b5bdb';
export const GOLD = 'GOLD';
// the quarterly set has no monthly paths: tickers cannot be evaluated on it (the server refuses them)
export const TICKERS_Q = '标的权重在季度情景集里不计入：标的（QQQ、TQQQ、BDC 等）需要月度路径，季度情景集没有。这里只按资产类别的权重计算（按比例归一化）；换回月度情景集即可算标的。';
export const TICKERS_Q_ONLY = '这个组合只有标的：季度情景集算不了标的（它们需要月度路径）。请换回月度情景集，或者给资产类别设权重。';
const ASSET_IDS = ['us_equity', 'nasdaq', 'energy', 'industrials', 'staples', 'healthcare', 'utilities', 'financials',
  'real_estate', 'gold', 'crude_oil', 'treasury_10y', 'cash'];
// used until (or when) GET /api/instruments cannot be read: the same ids, groups and names as its catalog
const FALLBACK_GROUPS = [
  { id: 'nasdaq', name: '纳指系', tickers: ['QQQ', 'QLD', 'TQQQ', 'QQQI'] },
  { id: 'bond_gold', name: '债与金', tickers: ['TLT', GOLD, 'GLD'] },
  { id: 'dividend_value', name: '红利与价值', tickers: ['XLG', 'SCHD', 'MO', 'BRK.B', 'BME'] },
  { id: 'bdc', name: 'BDC', tickers: ['MAIN', 'ARCC', 'CSWC'] },
  { id: 'mlp', name: 'MLP', tickers: ['EPD', 'MPLX'] },
];
const FALLBACK_NAMES = { QQQ: '纳指100 ETF（QQQ）', QLD: '两倍纳指（QLD，日度 2×）', TQQQ: '三倍纳指（TQQQ，日度 3×）',
  QQQI: '纳指备兑收益（QQQI）', TLT: '20 年以上美债（TLT）', [GOLD]: '黄金（金价）', GLD: '黄金 ETF（GLD）',
  XLG: '标普前 50 大（XLG）', SCHD: '美国红利（SCHD）', MO: '奥驰亚（MO）', 'BRK.B': '伯克希尔（BRK.B）',
  BME: '贝莱德健康科学信托（BME，封闭式基金）', MAIN: 'Main Street（MAIN，BDC）', ARCC: 'Ares Capital（ARCC，BDC）',
  CSWC: 'Capital Southwest（CSWC，BDC）', EPD: 'Enterprise Products（EPD，MLP）', MPLX: 'MPLX（MLP）' };
const STRUCTURAL = ['QQQ', 'QLD', 'TQQQ', 'QQQI', 'TLT', 'GLD'];
const LEVERAGED = ['QLD', 'TQQQ'];
const BDC = ['MAIN', 'ARCC', 'CSWC'], MLP = ['EPD', 'MPLX'];
// the short Chinese names in the editor rows and on chart axes (the ticker symbol is shown next to them)
const SHORT = { QQQ: '纳指100', QLD: '两倍纳指', TQQQ: '三倍纳指', QQQI: '纳指备兑', TLT: '长期美债',
  [GOLD]: '= 黄金', GLD: '黄金 ETF', XLG: '标普前50', SCHD: '美国红利', MO: '奥驰亚', 'BRK.B': '伯克希尔',
  BME: '健康信托', MAIN: 'Main St', ARCC: 'Ares', CSWC: 'Cap SW', EPD: 'Enterprise', MPLX: '中游 MLP' };
// categorical colours (none is a K-line up / down hex, so a holding never reads as "up" or "down")
const TICKER_COLORS = { QQQ: '#845ef7', QLD: '#9c36b5', TQQQ: '#5f3dc4', QQQI: '#cc5de8', TLT: '#0ca678', GLD: '#fab005',
  XLG: '#4263eb', SCHD: '#1c7ed6', MO: '#74b816', 'BRK.B': '#22b8cf', BME: '#15aabf', MAIN: '#e64980', ARCC: '#f783ac',
  CSWC: '#a61e4d', EPD: '#fd7e14', MPLX: '#d9480f' };
const KIND_NAMES = { structural: '结构推导', estimated: '估计型', alias: '别名' };
const GROUP_NOTES = { nasdaq: '按产品机制从纳指 100 推导（杠杆、期权都是算出来的）',
  bond_gold: 'TLT、GLD 按产品机制推导；金价就是资产类别「黄金」，两行是同一个权重',
  dividend_value: '2013 年起季末价格估计的风格组合 + 自身残差', bdc: '商业发展公司 · 2013 年起季末价格估计 · α 为贷款信用溢价',
  mlp: '能源中游合伙企业 · 2013 年起季末价格估计' };
// chip colour of each caveat: a judgement (warn / bad), never the K-line colours
const CHIP_KIND = { credit_link: 'bad', credit_link_ok: '', no_2008: 'warn', qqqi_short: 'warn', ruin_extrap: 'warn',
  rv_level: 'warn', long_yield_range: 'warn', gold_avg: 'warn', fixed_vol: 'warn', estimated: 'accent', payout_cut: '',
  start_est: '', bridge_month: '', tax: '' };
const CHIP_ORDER = ['credit_link', 'credit_link_ok', 'no_2008', 'estimated', 'qqqi_short', 'rv_level', 'ruin_extrap',
  'long_yield_range', 'gold_avg', 'fixed_vol', 'payout_cut', 'start_est', 'bridge_month', 'tax'];
const CHIP_FALLBACK = { estimated: '估计型 · 2013 年起的季末价格（13F）', bridge_month: '季内月度路径为布朗桥拆分（假设）',
  credit_link: '与股市的联动低于历史（核心问题）', credit_link_ok: '与股市的联动接近历史（模拟与历史相差不超过 0.15）',
  rv_level: '期内波动的水平是主要误差：只用月度输入时，样本外（2019–26）TQQQ 每年高估约 4.6%、QLD 约 1.6%',
  payout_cut: '收益率超过正常水平 2 倍时按正常收益率削减分配（假设）', gold_avg: '金价为月均价：月度波动和回撤被低估',
  ruin_extrap: '归零概率是外推（主要由 1987-10-19 一天决定）', qqqi_short: 'QQQI 只有 29 个月历史；不用 VXN 时约高估 2%/年',
  tax: '税前', start_est: '起点价格为估计（13F 2026-06-30 推一季）', no_2008: '样本里没有 2008 年',
  fixed_vol: '固定波动：敏感性，不是推演', long_yield_range: '收益率超出久期/凸性的拟合区间（1.4–5.3%）' };
const QQQI_ROC = '分配约 99.6% 为资本返还（Form 8937，2025 年 6–12 月）：分配高于期权费+股息时每股净值缩水';
// what each recipe cannot see (INSTRUMENTS_DESIGN §3), next to the catalog's own limits
const TICKER_NOTES = {
  QQQ: ['QQQ 是引擎纳指 100 资产减费率的直接映射：它在各情景之间的差别完全来自引擎。'],
  QLD: ['期内方差是唯一要建模的输入：杠杆损耗随它变化，“月收益不大但月内剧烈波动”的月份被低估。'],
  TQQQ: ['期内方差是唯一要建模的输入：杠杆损耗随它变化，“月收益不大但月内剧烈波动”的月份被低估。',
    '爆仓抽签：某一天纳指跌幅达到 1/3 时基金归零；概率的量级可信，具体数值不可信。'],
  QQQI: ['经济上，NAV 的 14.3% 分配里约一半是回报、一半是本金，所以每股净值长期向下；“先用股息”在 QQQI 上有一部分是在卖出本金。',
    '分配规则是基金的政策选择：默认按 NAV 定比例，也可以只分配期权费 + 股息，两者总回报相同，只是现金与净值缩水的拆分不同。'],
  TLT: ['起点收益率高（5.3–5.6%）而引擎的 10 年期路径缓慢回落，所以 TLT 的结果主要是票息加上缓慢的久期收益。'],
  GLD: ['金价为月均价：一阶自相关约 0.21，月度波动约低估 18%，最大/最小值和回撤被低估，时间上约滞后半个月。'],
  [GOLD]: ['金价为月均价：一阶自相关约 0.21，月度波动约低估 18%，最大/最小值和回撤被低估，时间上约滞后半个月。',
    '金价就是引擎的黄金资产本身：在引擎里黄金只跟通胀。'],
  XLG: ['几乎就是大盘：只用美股大盘的样本外 R² 也有 0.92 / 0.93 / 0.92。'],
  SCHD: ['长期回报里约 1.6%/年来自保留的 α，约 0.9%/年来自引擎保留的一半行业相对漂移（必需消费等），其余是这些行业较高的股息率。'],
  MO: ['2020 年以后的切分检验很弱（样本外 R² 0.19 / 0.01 / −0.43）。', '模拟波动高于历史（约 28% 对 22%），来自残差的波动率缩放和自由度膨胀。',
    '不预测诉讼等个股事件。'],
  'BRK.B': ['不分红。', '残差一阶自相关 −0.42；α 先验为 0（巴菲特 2025 年底卸任），8 个季度的检验里原始 α 更好，先验可能偏保守。'],
  MAIN: ['NAV 溢价不是状态变量：MAIN 市价约为 NAV 的 1.53 倍（2026-06-30），回落风险没有进入模型。'],
  CSWC: ['NAV 溢价 1.43 倍高于历史均值，回落风险没有进入模型。', '2015 年重组，残差只用 2016 年起的数据，载荷按 BDC 池化。'],
  EPD: ['产量没有核心变量，用能源股代理。', '模拟波动低于历史（约 20% 对 26%）；与美股的相关接近历史，所以不提示信用联动被低估。'],
  MPLX: ['2016 年以前是另一家公司（MarkWest 合并、IDR 买断），2020–26 年的重估（每年 +12–16%）不外推。',
    '模拟波动低于历史（约 23% 对 37%），部分来自信用联动被低估。', '载荷按 MLP 池化。'],
  BME: ['备兑看涨的封顶效应估不出来：加入非线性项后样本外崩溃。', '折价状态只有半年数据（2026-06-30 折价 2.9%）。'],
};
const BDC_NOTE = '预期总回报（约 8–9%）低于当前收益率（9–11%），价格或分配必须下行。样本（2013–2026）里没有一次 BDC 大幅削减分配，'
  + '削减频率无法检验，这里的削减概率可能偏悲观；α（贷款信用溢价）每年 ±1.6% 的不确定性对长期分配影响很大。';
const MLP_NOTE = 'MLP 的分配为粘性规则：滞后一年 log 总回报低于 −30% 时冻结 4 个季度（假设）。分配属于 K-1 收入，这里不区分税务性质。';
const VINTAGE_NAMES = { ndx: '纳指 100', vxn: 'VXN', us_market_daily: '美股日度', dgs20: '20 年期收益率', dgs30: '30 年期收益率',
  qld_nav: 'QLD 净值', tqqq_nav: 'TQQQ 净值', prices_13f: '13F 季末价', nport: 'N-PORT 持仓' };

// ---------------------------------------------------------------- module state

const STORE_KEY = 'msl-weights-v2';       // {values: {id: number}, mode: 'ratio' | 'amount'}
const LEGACY_KEY = 'msl-weights';          // the asset-only editor's store ({asset: percent})
const MINE_KEY = 'msl-my-portfolios';      // [{name, values, mode, saved}], this browser only
export const TK = {
  values: null, mode: 'ratio',             // the weights editor, shared by the two tabs for this page load
  open: null,                              // {assets, tickers} once the reader has toggled a section
  pj: { ticker: 'QQQ', measure: 'price', scope: 'weighted', freq: 'M', log: null, compare: null, params: {},
    vol: 'model', fixedVol: 25, alpha: 'shrunk', attr: 'median', corr: 'assets', allRows: false },
  alpha: 'shrunk',                         // 提取页: the estimated tickers' α
};

// ---------------------------------------------------------------- catalog and names

let CAT = null;                            // the last catalog read (names, groups, chips, diagnostics)
let catalogP = null;
// GET /api/instruments once per page load; {error} when it cannot be read (retried on the next call)
export function loadCatalog() {
  if (!catalogP) {
    catalogP = api.get('instruments').then((c) => { CAT = c; return c; },
      (err) => { catalogP = null; return { error: err }; });
  }
  return catalogP;
}
export const catalog = () => CAT;
export function tickerGroups() {
  const g = CAT?.groups;
  return Array.isArray(g) && g.length ? g.map(x => ({ id: x.id, name: x.name, tickers: (x.tickers || []).slice() })) : FALLBACK_GROUPS;
}
export const tickerOrder = () => tickerGroups().flatMap(g => g.tickers);
const storedTickers = () => tickerOrder().filter(t => t !== GOLD);
export const isTicker = (id) => id === GOLD || storedTickers().includes(id);
export const tickerName = (t) => CAT?.tickers?.[t]?.name || FALLBACK_NAMES[t] || t;
export const tickerKind = (t) => CAT?.tickers?.[t]?.kind || (t === GOLD ? 'alias' : STRUCTURAL.includes(t) ? 'structural' : 'estimated');
export const isEstimated = (t) => isTicker(t) && t !== GOLD && tickerKind(t) === 'estimated';
const shortOf = (t) => SHORT[t] || String(tickerName(t)).replace(/（.*$/, '');
// A holding (asset id or ticker id) as the prose names it: tickers by their symbol, assets by their Chinese name.
export const holdingLabel = (id) => (isTicker(id) ? id : assetLabel(id));
export const holdingName = (id) => (isTicker(id) ? tickerName(id) : assetLabel(id));
export const holdingColor = (id, i = 0) => TICKER_COLORS[id] || charts.assetColor(id, i);
export function weightsText(weights) {
  return Object.entries(weights || {}).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1])
    .map(([id, w]) => `${holdingLabel(id)} ${fmtShare(w)}`).join(' + ') || '—';
}
export const chipText = (key) => CAT?.chips?.[key] || CHIP_FALLBACK[key] || '';

// An id as typed (any case; "brk-b", "金价", "美股大盘", "奥驰亚") -> an asset id or a stored ticker id, else null.
// GOLD (金价) is the gold asset class itself.
export function resolveId(raw, assetIds = ASSET_IDS) {
  const s = String(raw ?? '').trim();
  if (!s) return null;
  if (assetIds.includes(s)) return s;
  const low = s.toLowerCase();
  const a = assetIds.find(x => x.toLowerCase() === low);
  if (a) return a;
  const up = s.toUpperCase().replace(/\s+/g, '');
  if (up === GOLD || s === '金价' || s === '黄金') return assetIds.includes('gold') ? 'gold' : null;
  const map = Object.fromEntries(storedTickers().map(t => [t.toUpperCase(), t]));
  if (map[up]) return map[up];
  const dotted = up.replace(/[-/_]/g, '.');
  if (map[dotted]) return map[dotted];
  if (up === 'BRKB') return map['BRK.B'] || null;
  for (const id of assetIds) if (assetLabel(id) === s) return id;
  for (const t of storedTickers()) if (shortOf(t) === s || String(tickerName(t)).replace(/（.*$/, '') === s) return t;
  return null;
}

// "MAIN:10,qqqi:15,us_equity:30" (a deep link) -> {weights: {id: value}, unknown: [typed ids]}
export function parseLinkWeights(text, assetIds = ASSET_IDS) {
  const weights = {}, unknown = [];
  for (const part of String(text || '').split(/[,;]/)) {
    const i = part.lastIndexOf(':');
    if (i <= 0) { if (part.trim()) unknown.push(part.trim()); continue; }
    const id = resolveId(part.slice(0, i), assetIds);
    const x = Number(part.slice(i + 1));
    if (!id || !Number.isFinite(x) || x < 0) { unknown.push(part.slice(0, i).trim() || part.trim()); continue; }
    weights[id] = (weights[id] || 0) + x;
  }
  return { weights, unknown };
}

// Pasted holdings, one per line ("MAIN 120000", "QQQI 15%", "BRK.B 5万", "美股大盘 30%") ->
// {values, mode: 'amount' | 'ratio', bad: [lines], ignored: [lines], count}. Amounts are USD (万 = 10,000,
// k = 1,000, m = 1,000,000). Percent lines next to amount lines are shares of the whole portfolio.
export function parseHoldings(text, assetIds = ASSET_IDS) {
  const pcts = {}, amts = {}, bad = [];
  let nP = 0, nA = 0;
  const lineRe = /^(.*?)[\s:：=,，]*([$＄]?\s*\d[\d,，]*(?:\.\d+)?)\s*(%|％|万|萬|[kKmMwW])?\s*(?:美元|usd|\$)?$/i;
  for (const raw of String(text || '').split(/\r?\n|[;；]/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const m = lineRe.exec(line);
    const id = m ? resolveId(m[1].replace(/[\s:：=,，]+$/, ''), assetIds) : null;
    const x = m ? Number(m[2].replace(/[$＄\s,，]/g, '')) : NaN;
    if (!id || !Number.isFinite(x) || x < 0) { bad.push(line); continue; }
    const unit = m[3] || '';
    if (unit === '%' || unit === '％') { pcts[id] = (pcts[id] || 0) + x; nP += 1; } else {
      const k = { 万: 1e4, 萬: 1e4, w: 1e4, W: 1e4, k: 1e3, K: 1e3, m: 1e6, M: 1e6 }[unit] || 1;
      amts[id] = (amts[id] || 0) + x * k; nA += 1;
    }
  }
  const P = Object.values(pcts).reduce((s, x) => s + x, 0), A = Object.values(amts).reduce((s, x) => s + x, 0);
  if (!nA || !(A > 0)) return { values: pcts, mode: 'ratio', bad, ignored: [], count: nP };
  if (!nP) return { values: amts, mode: 'amount', bad, ignored: [], count: nA };
  if (P >= 100) return { values: pcts, mode: 'ratio', bad, ignored: Object.keys(amts), count: nP };
  const whole = A / (1 - P / 100);                 // the amounts are the rest of the portfolio
  const values = { ...amts };
  for (const [id, p] of Object.entries(pcts)) values[id] = (values[id] || 0) + (p / 100) * whole;
  return { values, mode: 'amount', bad, ignored: [], count: nP + nA };
}

// ---------------------------------------------------------------- small helpers

const pct = (v, d = 1) => ui.fmtPct(v, d);
const readJSON = (key) => { try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch (_) { return null; } };
const writeJSON = (key, v) => { try { localStorage.setItem(key, JSON.stringify(v)); return true; } catch (_) { return false; } };
const num = (x) => (typeof x === 'number' ? x : Number(x));
function fmtUsd(v, d) {
  if (!isNum(v)) return '—';
  const dd = d ?? (Math.abs(v) >= 1000 ? 0 : Math.abs(v) >= 100 ? 1 : 2);
  return `${v.toLocaleString('zh-CN', { minimumFractionDigits: dd, maximumFractionDigits: dd })} 美元`;
}
const usdAxis = (v) => (!isNum(v) ? '' : Math.abs(v) >= 1000 ? `${Math.round(v).toLocaleString('zh-CN')}` : `${+v.toFixed(Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2)}`);
const multFmt = (v) => (isNum(v) ? `${v.toFixed(2)} 倍` : '—');
const signed = (v, d = 1) => h0('span', { class: ui.signCls(v, d) || null }, ui.fmtPct(v, d));
function h0(...a) { return ui.h(...a); }
// "13F 2026-06-30 price rolled to 2026-09-30 by ..." -> what the start price is, in Chinese
const ROLL_TEXT = [[/NDX price return/, '按纳指 100 的实际价格回报'],
  [/QQQI model/, '按 QQQI 的模型（对纳指的 beta × 纳指实际回报，减去按 NAV 比例的分配）'],
  [/duration model/, '按 TLT 收益率（0.513 × 20 年 + 0.487 × 30 年）上的久期模型 D(y)·Δy'],
  [/gold price/, '按伦敦金月均价之比（减费率）']];
export function basisText(b, short = false) {
  const s = String(b ?? '').trim();
  if (!s) return '';
  if (s === 'user') return '用户输入的起点价格';
  let m = /^13F (\d{4}-\d{2}-\d{2}) price rolled to (\d{4}-\d{2}-\d{2})/.exec(s);
  if (m && short) return `13F ${m[1]} 季末价，按配方的系统部分推到 ${m[2]}（残差未知）`;
  if (m) return `13F ${m[1]} 季末价，按配方在核心 ${m[2].slice(0, 4)} 年三季度实际回报上的系统部分推到 ${m[2]}（残差未知）`;
  m = /^13F (\d{4}-\d{2}-\d{2}) rolled to (\d{4}-\d{2}-\d{2})(?: by (.*))?$/.exec(s);
  if (m) {
    const how = ROLL_TEXT.find(([re]) => re.test(m[3] || ''))?.[1] || '按核心的回报';
    return `13F ${m[1]} 季末价，${how}推到 ${m[2]}`;
  }
  if (/ProShares NAV/i.test(s)) return 'ProShares 公布的基金净值（NAV）';
  m = /SEC XBRL market price (\d{4}-\d{2}-\d{2})/.exec(s);
  if (m) return `SEC XBRL 申报的 ${m[1]} 市场价`;
  return s.replace(/\s*gold\s*资产/g, '黄金资产');
}
export function vintageText(v) {
  return Object.entries(v || {}).filter(([, d]) => d).map(([k, d]) => `${VINTAGE_NAMES[k] || k} ${d}`).join(' · ');
}
// one segmented control that can be repainted from outside (ui.seg cannot)
function segBox(options, get, set, label) {
  const box = ui.h('div', { class: 'seg', role: 'group', 'aria-label': label || null });
  const paint = () => box.replaceChildren(...options.map(o => ui.h('button', { type: 'button', class: o.value === get() ? 'on' : '',
    disabled: o.disabled ? true : null, title: o.title || null, onclick: () => { if (o.value !== get()) { set(o.value); paint(); } } }, o.label)));
  paint();
  box.paint = paint;
  return box;
}

// ---------------------------------------------------------------- weights editor

// {any id: number} -> {resolved id: sum of the positive values}, or null when nothing is positive
function resolveMap(obj, aIds) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return null;
  const out = {};
  for (const [k, x] of Object.entries(obj)) {
    const id = resolveId(k, aIds);
    const n = num(x);
    if (id && Number.isFinite(n) && n > 0) out[id] = (out[id] || 0) + n;
  }
  return Object.keys(out).length ? out : null;
}
// a positive map over ids -> percents summing to exactly 100 (0.1 steps; the rounding residual goes to the largest)
function toPercent(obj, ids) {
  if (!obj) return null;
  const s = ids.reduce((t, id) => t + (num(obj[id]) > 0 ? num(obj[id]) : 0), 0);
  if (!(s > 0)) return null;
  const out = Object.fromEntries(ids.map(id => [id, num(obj[id]) > 0 ? Math.round((num(obj[id]) / s) * 1000) / 10 : 0]));
  const diff = Math.round((100 - ids.reduce((t, id) => t + out[id], 0)) * 10) / 10;
  if (diff) {
    const top = ids.reduce((a, b) => (out[b] > out[a] ? b : a));
    out[top] = Math.round((out[top] + diff) * 10) / 10;
  }
  return out;
}
// a positive map over ids -> exact percents summing to 100 (no rounding: a mode switch must not move the weights;
// the inputs show them rounded)
function exactPercent(obj, ids) {
  if (!obj) return null;
  const s = ids.reduce((t, id) => t + (num(obj[id]) > 0 ? num(obj[id]) : 0), 0);
  if (!(s > 0)) return null;
  return Object.fromEntries(ids.map(id => [id, num(obj[id]) > 0 ? (num(obj[id]) / s) * 100 : 0]));
}
// a stored map: every value finite and >= 0 over ids (unknown ids dropped), null when nothing is positive
function cleanMap(obj, ids, cap = Infinity) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return null;
  const out = {};
  let s = 0;
  for (const id of ids) {
    const x = num(obj[id]);
    out[id] = Number.isFinite(x) && x > 0 ? Math.min(cap, x) : 0;
    s += out[id];
  }
  return s > 0 ? out : null;
}

// Card with one row per asset class and per ticker (slider + number input in 按比例, a USD input in 按金额).
// onChange(weights) fires ~`delay` ms after the last edit with normalized decimal weights over every holding > 0
// (asset ids and ticker ids; 金价 is `gold`); presets, imports and saved portfolios fire at once.
// onAmount(totalUSD | null) after every change: the total in 按金额, null in 按比例.
// amountBase() gives the total that 按比例 -> 按金额 starts from (default 1,000,000 USD).
export function portfolioEditor({ assets, presets, initial, onChange, onAmount, amountBase, delay = 400, cls = '' } = {}) {
  injectWeightsStyle();
  injectStyles();
  const { h } = ui;
  const alist = (assets && assets.length ? assets : ASSET_IDS.map(id => ({ id })))
    .map(a => ({ id: a.id, name: a.name || assetLabel(a.id), proxy: a.proxy || '' }));
  const aIds = alist.map(a => a.id);
  const groups = tickerGroups().map(g => ({ ...g, tickers: g.tickers.filter(t => t === GOLD ? aIds.includes('gold') : true) }));
  const tIds = groups.flatMap(g => g.tickers).filter(t => t !== GOLD);
  const ids = [...aIds, ...tIds];
  const v = Object.fromEntries(ids.map(id => [id, 0]));
  let mode = 'ratio';
  let presetList = presets && presets.length ? presets : FALLBACK_PRESETS;

  // starting values: the deep link / preset > this page load (the other tab) > stored > the asset-only store > 60/40
  const linked = toPercent(resolveMap(initial, aIds), ids);
  const stored = readJSON(STORE_KEY);
  const mem = cleanMap(TK.values, ids);
  const kept = cleanMap(stored?.values, ids);
  if (linked) { Object.assign(v, linked); mode = 'ratio'; }
  else if (mem) { Object.assign(v, mem); mode = TK.mode === 'amount' ? 'amount' : 'ratio'; }
  else if (kept) { Object.assign(v, kept); mode = stored.mode === 'amount' ? 'amount' : 'ratio'; }
  else Object.assign(v, toPercent(cleanMap(readJSON(LEGACY_KEY), aIds, 100), ids) || toPercent(resolveMap(FALLBACK_PRESETS[0].weights, aIds), ids));
  if (mode === 'ratio') for (const id of ids) v[id] = Math.min(100, v[id]);

  const total = () => ids.reduce((s, id) => s + v[id], 0);
  function get() {
    const s = total();
    if (!(s > 0)) return null;
    const out = {};
    for (const id of ids) if (v[id] > 0) out[id] = v[id] / s;
    return out;
  }
  const fmtIn = (x) => (mode === 'amount' ? String(Math.round(x)) : Number.isInteger(x) ? String(x) : String(Math.round(x * 10) / 10));

  // ---- rows (the gold asset has two: 黄金 under 资产类别 and 金价 under 标的)
  const rows = {};
  function makeRow(id, { label, sub, title, color }) {
    const slider = h('input', { type: 'range', min: 0, max: 100, step: 0.5, 'aria-label': `${label}权重` });
    const input = h('input', { type: 'number', min: 0, step: 'any', 'aria-label': `${label}权重` });
    const unit = h('span', { class: 'muted tk-unit' });
    const share = h('span', { class: 'weights-share' });
    const row = h('div', { class: 'weights-row tk-row' },
      h('span', { class: 'weights-dot', style: { background: color } }),
      h('span', { class: 'weights-name tk-name', title }, sub !== undefined ? [h('b', {}, label), h('span', { class: 'tk-sub' }, sub)] : label),
      slider, h('span', { class: 'weights-in' }, input, unit), share);
    slider.addEventListener('input', () => { setVal(id, Number(slider.value), input); });
    input.addEventListener('input', () => {
      const x = input.value === '' ? 0 : Number(input.value);
      if (Number.isFinite(x)) setVal(id, x, input, true);
    });
    input.addEventListener('change', () => { input.value = fmtIn(v[id]); });
    (rows[id] = rows[id] || []).push({ row, slider, input, unit, share });
    return row;
  }
  function setVal(id, x, src, typed = false) {
    v[id] = Math.max(0, mode === 'ratio' ? Math.min(100, x) : Math.min(1e13, x));
    for (const r of rows[id] || []) {
      r.slider.value = String(Math.min(100, v[id]));
      if (r.input !== src || !typed) r.input.value = fmtIn(v[id]);
    }
    changed();
  }
  const syncAll = () => {
    for (const id of ids) {
      for (const r of rows[id] || []) {
        r.slider.value = String(Math.min(100, v[id]));
        r.input.value = fmtIn(v[id]);
        r.input.max = mode === 'ratio' ? '100' : '';
        r.unit.textContent = mode === 'ratio' ? '%' : '美元';
      }
    }
  };

  const assetGrid = h('div', { class: 'weights-grid tk-grid' }, alist.map(a => makeRow(a.id,
    { label: a.name, title: a.proxy ? `${a.name}：${a.proxy}` : a.name, color: charts.assetColor(a.id) })));
  const tickerBox = h('div', { class: 'tk-groups' }, groups.map(g => h('div', { class: 'tk-group' },
    h('div', { class: 'tk-group-head' }, h('b', {}, g.name), h('span', { class: 'muted' }, GROUP_NOTES[g.id] || '')),
    h('div', { class: 'weights-grid tk-grid' }, g.tickers.map(t => (t === GOLD
      ? makeRow('gold', { label: '金价', sub: '= 黄金', title: '金价就是资产类别「黄金」（伦敦金价月均价）：与上面的黄金是同一个权重，不另占一列',
        color: charts.assetColor('gold') })
      : makeRow(t, { label: t, sub: shortOf(t), title: `${tickerName(t)} · ${KIND_NAMES[tickerKind(t)] || ''}`, color: holdingColor(t) })))))));

  // ---- status of the ticker files (GET /api/instruments status)
  const statusBox = h('div', { class: 'tk-status' });
  function paintStatus(cat) {
    if (!cat) { statusBox.replaceChildren(); return; }
    if (cat.error) {
      const e = cat.error;
      statusBox.replaceChildren(ui.notice(e.status === 404
        ? '本地服务还没有标的目录：请确认代码已更新，并重启 python -m scenariolab.server。在那之前标的按内置名单列出，含标的的组合可能无法计算。'
        : `无法读取标的目录：${errorText(e)} 含标的的组合可能无法计算。`, 'error'));
      return;
    }
    const st = cat.status || {};
    const cmd = st.build || 'python -m scenariolab.build --only instruments';
    const issue = String(st.issue || '标的路径还没有生成').replace(/[。.]$/, '');
    statusBox.replaceChildren(...(st.available === false
      ? [ui.notice(`标的路径暂时不可用：${issue}。${issue.includes('scenariolab.build') ? '' : `在项目根目录运行 ${cmd}，`}生成后刷新本页；在那之前只能评估资产类别。`, 'error')]
      : []));
  }

  // ---- the summary of each section, the allocation bar, the sum chip
  const sumChip = h('span', { class: 'chip' });
  const bar = h('div', { class: 'weights-bar', 'aria-hidden': 'true' });
  const assetSum = h('span', { class: 'tk-sec-sum' });
  const tickerSum = h('span', { class: 'tk-sec-sum' });
  const presetBox = h('div', { class: 'weights-presets' });
  const presetNote = h('div', { class: 'tk-preset-note' });
  const amountLine = h('div', { class: 'tk-amount-line' });
  function paint() {
    const s = total();
    for (const id of ids) for (const r of rows[id] || []) {
      r.share.textContent = s > 0 ? fmtShare(v[id] / s) : '—';
      r.row.classList.toggle('zero', !(v[id] > 0));
    }
    const held = ids.filter(id => v[id] > 0);
    bar.replaceChildren(...(s > 0 ? held.map(id => h('span', {
      style: { width: `${(v[id] / s) * 100}%`, background: holdingColor(id) }, title: `${holdingLabel(id)} ${fmtShare(v[id] / s)}` })) : []));
    const sa = aIds.reduce((t, id) => t + v[id], 0), st = tIds.reduce((t, id) => t + v[id], 0);
    const na = aIds.filter(id => v[id] > 0).length, nt = tIds.filter(id => v[id] > 0).length;
    assetSum.textContent = s > 0 && na ? `已配置 ${na} 类 · 占 ${fmtShare(sa / s)}` : '未配置';
    tickerSum.textContent = s > 0 && nt ? `已配置 ${nt} 个 · 占 ${fmtShare(st / s)}` : '未配置';
    if (mode === 'amount') {
      sumChip.className = `chip ${s > 0 ? 'accent' : 'bad'}`;
      sumChip.textContent = s > 0 ? `合计 ${fmtUsd(s, 0)}` : '合计 0：请至少给一个资产或标的填金额';
      amountLine.textContent = s > 0 ? `按金额输入：计算时换算成比例（合计 ${fmtUsd(s, 0)} ≈ ${ui.fmtMoney(s)} 美元）；不需要价格，金额只决定比例。` : '';
    } else {
      const ok = Math.abs(s - 100) < 0.05;
      sumChip.className = `chip ${s <= 0 ? 'bad' : ok ? 'good' : 'warn'}`;
      sumChip.textContent = s <= 0 ? '合计 0%：请至少给一个资产或标的设权重'
        : ok ? '合计 100%' : `合计 ${fmtIn(Math.round(s * 10) / 10)}%，按比例归一化`;
      amountLine.textContent = '';
    }
    const n = get();
    let active = null;
    for (const b of presetBox.querySelectorAll('button')) {
      const on = !!n && !!b._w && ids.every(id => Math.abs((n[id] || 0) - (b._w[id] || 0)) < 5e-4);
      b.classList.toggle('on', on);
      if (on) active = b._p;
    }
    const hasT = active && Object.keys(active.weights || {}).some(k => isTicker(resolveId(k, aIds)));
    presetNote.replaceChildren(...(hasT ? [h('b', {}, active.name), ` —— ${active.description || '只为展示标的层，不是建议。'}`] : []));
    presetNote.style.display = hasT ? '' : 'none';
  }

  // ---- presets (the engine's 5 plus 示例：收息标的)
  function presetTip(p) {
    const lines = [p.description || p.name];
    const pf = p.portfolio;
    if (pf) lines.push(`加权实际年化回报中位数 ${pct(pf.real_cagr_p50)} · 最大回撤中位数 ${pct(pf.max_dd_p50)}`);
    const by = p.withdrawal_by_horizon || {};
    const hz = Object.keys(by).map(Number).filter(y => isNum(by[y]?.safe_rate_weighted?.pro_rata)).sort((a, b) => a - b);
    if (hz.length) lines.push(`安全提取率（按比例提取）${hz.map(y => `${y} 年 ${ui.fmtPct(by[y].safe_rate_weighted.pro_rata, 2)}`).join(' · ')}`);
    return lines.join('\n');
  }
  function renderPresets() {
    presetBox.replaceChildren(h('span', { class: 'weights-presets-label' }, '预设'),
      ...presetList.map(p => {
        const b = h('button', { class: 'btn', title: presetTip(p), onclick: () => set(p.weights) }, p.name);
        const c = toPercent(resolveMap(p.weights, aIds), ids);
        if (c) b._w = Object.fromEntries(ids.map(id => [id, c[id] / 100]));
        b._p = p;
        return b;
      }));
  }

  // ---- changes
  let lastSent = get();
  const near = (a, b) => !!a && !!b && ids.every(id => Math.abs((a[id] || 0) - (b[id] || 0)) < 6e-4);
  const fire = debounce(() => {
    const n = get();
    if (!n || near(n, lastSent)) return;
    lastSent = n;
    if (onChange) onChange(n);
  }, delay);
  function changed(now = false) {
    paint();
    TK.values = { ...v };
    TK.mode = mode;
    writeJSON(STORE_KEY, { values: { ...v }, mode });
    if (onAmount) onAmount(mode === 'amount' ? total() : null);
    if (now) fire.flush(); else fire();
  }
  // weights (any ids, relative) -> the editor; keeps 按金额 at its current total
  function set(obj, { now = true, asMode } = {}) {
    const resolved = resolveMap(obj, aIds);
    if (!resolved) return false;
    if (asMode) mode = asMode;
    if (mode === 'amount' && asMode !== 'amount') {
      const s = total() > 0 ? total() : base();
      const sum = Object.values(resolved).reduce((t, x) => t + x, 0);
      for (const id of ids) v[id] = resolved[id] ? Math.round((resolved[id] / sum) * s) : 0;
    } else if (mode === 'amount') {
      for (const id of ids) v[id] = resolved[id] ? Math.round(resolved[id]) : 0;
    } else {
      const p = toPercent(resolved, ids);
      for (const id of ids) v[id] = p[id];
    }
    syncAll();
    modeSeg.paint();
    paintMode();
    changed(now);
    return true;
  }
  const base = () => { const b = amountBase ? Number(amountBase()) : NaN; return Number.isFinite(b) && b > 0 ? b : 1e6; };
  function setMode(m) {
    if (m === mode) return;
    const s = total();
    if (m === 'amount') {
      const b = base();
      for (const id of ids) v[id] = s > 0 ? Math.round((v[id] / s) * b) : 0;
    } else {
      const p = exactPercent(v, ids);
      for (const id of ids) v[id] = p ? p[id] : 0;
    }
    mode = m;
    syncAll();
    paintMode();
    changed();                 // the shares are exactly unchanged (the inputs show them rounded): no request
  }
  const paintMode = () => { el.classList.toggle('tk-amount', mode === 'amount'); };     // el: the card, below

  const modeSeg = segBox([{ value: 'ratio', label: '按比例（%）' }, { value: 'amount', label: '按金额（美元）' }],
    () => mode, setMode, '输入方式');
  const normBtn = h('button', { class: 'btn', title: mode === 'amount' ? '' : '把输入框改写成合计 100% 的百分比（结果不变）',
    onclick: () => { if (mode !== 'ratio') { setMode('ratio'); return; } const p = exactPercent(v, ids); if (p) { Object.assign(v, p); syncAll(); changed(); } } }, '归一化');
  const clearBtn = h('button', { class: 'btn', title: '全部清零后再逐个设置',
    onclick: () => { ids.forEach(id => { v[id] = 0; }); syncAll(); changed(); } }, '清零');
  const clearTickers = h('button', { class: 'btn tk-mini-btn', title: '只清空标的，资产类别不变',
    onclick: (e) => { e.preventDefault(); tIds.forEach(id => { v[id] = 0; }); syncAll(); changed(); } }, '清空标的');

  // ---- 导入持仓
  const ta = h('textarea', { rows: 5, spellcheck: 'false', 'aria-label': '导入持仓：每行一个',
    placeholder: '每行一个持仓，例如：\nMAIN 120000\nQQQI 15%\nBRK.B 5万\n美股大盘 30%' });
  const importMsg = h('div', { class: 'tk-import-msg' });
  const importBox = h('div', { class: 'tk-import', style: { display: 'none' } },
    h('div', { class: 'tk-help' }, '每行一个：代码或中文名，后面跟金额（美元，可写 12万、50k）或百分比。只有百分比时按比例填入；只有金额时切换到按金额；',
      '两者混合时，百分比是占整个组合的比例。金价、GOLD、黄金都指资产类别黄金。'),
    ta,
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', onclick: applyImport }, '解析并填入'),
      h('button', { class: 'btn', onclick: () => { importBox.style.display = 'none'; } }, '收起')),
    importMsg);
  const importBtn = h('button', { class: 'btn', onclick: () => { importBox.style.display = importBox.style.display === 'none' ? '' : 'none'; if (importBox.style.display === '') ta.focus(); } }, '导入持仓');
  function applyImport() {
    const r = parseHoldings(ta.value, aIds);
    const kids = [];
    if (!r.count) kids.push(ui.chip('没有识别出任何持仓', 'bad'));
    else {
      for (const id of ids) v[id] = 0;
      mode = r.mode;
      set(r.values, { asMode: r.mode });
      kids.push(ui.chip(`已填入 ${Object.keys(r.values).length} 项（${r.mode === 'amount' ? '按金额' : '按比例'}）`, 'good'));
    }
    if (r.ignored.length) kids.push(ui.chip(`百分比合计已达 100%，金额行没有计入：${r.ignored.map(holdingLabel).join('、')}`, 'warn'));
    for (const line of r.bad.slice(0, 12)) kids.push(ui.chip(`无法识别：${line.slice(0, 40)}`, 'warn'));
    if (r.bad.length > 12) kids.push(ui.chip(`另有 ${r.bad.length - 12} 行无法识别`, 'warn'));
    importMsg.replaceChildren(...kids);
  }

  // ---- 我的组合 (localStorage only, never the server or the address)
  const mineSel = h('select', { 'aria-label': '我的组合' });
  const nameIn = h('input', { type: 'text', maxlength: '30', placeholder: '组合名称', 'aria-label': '组合名称' });
  const mineMsg = h('span', { class: 'muted tk-mine-msg' });
  const saveRow = h('span', { class: 'tk-save-row', style: { display: 'none' } }, nameIn,
    h('button', { class: 'btn primary', onclick: () => saveMine() }, '保存'), h('button', { class: 'btn', onclick: () => { saveRow.style.display = 'none'; } }, '取消'));
  const delBtn = h('button', { class: 'btn', title: '从本浏览器删除所选的组合', onclick: () => delMine() }, '删除');
  const mineList = () => { const l = readJSON(MINE_KEY); return Array.isArray(l) ? l.filter(x => x && typeof x.name === 'string' && x.values) : []; };
  function paintMine() {
    const list = mineList();
    mineSel.replaceChildren(h('option', { value: '' }, list.length ? `我的组合（${list.length}）` : '我的组合（未保存）'),
      ...list.map((x, i) => h('option', { value: String(i) }, `${x.name}${x.saved ? ` · ${x.saved}` : ''}`)));
    delBtn.disabled = !list.length;
  }
  mineSel.addEventListener('change', () => {
    const x = mineList()[Number(mineSel.value)];
    if (!x || mineSel.value === '') return;
    const vals = cleanMap(x.values, ids);
    if (!vals) { mineMsg.textContent = '这个组合里没有可识别的持仓'; return; }
    mode = x.mode === 'amount' ? 'amount' : 'ratio';
    set(vals, { asMode: mode });
    mineMsg.textContent = `已载入「${x.name}」`;
  });
  function saveMine() {
    const name = nameIn.value.trim() || `我的组合 ${mineList().length + 1}`;
    if (!get()) { mineMsg.textContent = '合计为 0，没有可保存的组合'; return; }
    const list = mineList().filter(x => x.name !== name);
    const d = new Date();
    const stamp = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    list.unshift({ name, values: { ...v }, mode, saved: stamp });
    const ok = writeJSON(MINE_KEY, list.slice(0, 20));
    mineMsg.textContent = ok ? `已保存「${name}」（只在本浏览器）` : '本浏览器无法保存（存储不可用）';
    saveRow.style.display = 'none';
    paintMine();
  }
  function delMine() {
    const i = Number(mineSel.value);
    const list = mineList();
    if (mineSel.value === '' || !list[i]) { mineMsg.textContent = '先在左边选一个已保存的组合'; return; }
    const [gone] = list.splice(i, 1);
    mineMsg.textContent = writeJSON(MINE_KEY, list) ? `已删除「${gone.name}」` : '本浏览器无法删除（存储不可用）';
    paintMine();
  }
  nameIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') saveMine(); });
  const saveBtn = h('button', { class: 'btn', title: '把当前组合保存在本浏览器（不上传、不进链接）',
    onclick: () => { saveRow.style.display = saveRow.style.display === 'none' ? '' : 'none'; if (saveRow.style.display === '') nameIn.focus(); } }, '保存当前');
  paintMine();

  // a section the reader has not toggled: open on a wide screen, or when it holds something
  const wide = window.innerWidth >= 760;
  const open0 = TK.open || { assets: wide || aIds.some(id => v[id] > 0) || !tIds.some(id => v[id] > 0), tickers: wide || tIds.some(id => v[id] > 0) };
  const assetSec = h('details', { class: 'tk-section', open: open0.assets ? true : null },
    h('summary', {}, h('span', { class: 'tk-sec-title' }, `资产类别（${alist.length} 类）`), assetSum), assetGrid);
  const lockBox = h('div', { class: 'tk-lock' });
  const tickerSec = h('details', { class: 'tk-section', open: open0.tickers ? true : null },
    h('summary', {}, h('span', { class: 'tk-sec-title' }, `标的（${tIds.length} 个 + 金价）`), tickerSum, clearTickers),
    lockBox, statusBox, tickerBox);
  // a scenario set the tickers cannot use (the quarterly set has no monthly paths): their rows are disabled, the
  // weights are kept for when the reader switches back, and the request carries the asset classes only
  let locked = null;
  function lockTickers(reason) {
    locked = reason || null;
    for (const id of tIds) for (const r of rows[id] || []) { r.slider.disabled = !!locked; r.input.disabled = !!locked; }
    clearTickers.disabled = !!locked;
    tickerBox.classList.toggle('tk-locked', !!locked);
    lockBox.replaceChildren(...(locked ? [ui.notice(locked, 'info')] : []));
  }
  // the asset-class weights alone, normalized (null when none): what a set without tickers can evaluate
  function getAssets() {
    const s = aIds.reduce((t, id) => t + v[id], 0);
    if (!(s > 0)) return null;
    const out = {};
    for (const id of aIds) if (v[id] > 0) out[id] = v[id] / s;
    return out;
  }
  const remember = () => { TK.open = { assets: assetSec.open, tickers: tickerSec.open }; };
  assetSec.addEventListener('toggle', remember);
  tickerSec.addEventListener('toggle', remember);

  renderPresets();
  syncAll();
  paint();
  TK.values = { ...v };
  TK.mode = mode;
  const el = ui.card({ title: '组合权重', cls: `weights-card tk-editor ${cls}`,
    sub: `${alist.length} 类资产和 ${tIds.length} 个标的可以混合持有。合计不必正好是 100%，计算时按比例归一化；停手约 0.4 秒后自动重算，并在本浏览器记住上次的组合。`,
    actions: [sumChip, normBtn, clearBtn] },
  h('div', { class: 'tk-toolbar' }, modeSeg, importBtn,
    h('span', { class: 'tk-mine' }, mineSel, saveBtn, delBtn), saveRow, mineMsg),
  importBox, presetBox, presetNote, amountLine, bar, assetSec, tickerSec);
  paintMode();
  return {
    el, get, set,
    setPresets(next) { presetList = next && next.length ? next : FALLBACK_PRESETS; renderPresets(); paint(); },
    setCatalog: paintStatus,
    mode: () => mode,
    amountTotal: () => (mode === 'amount' ? total() : null),
    held: () => ids.filter(id => v[id] > 0),
    heldTickers: () => tIds.filter(id => v[id] > 0),
    lockTickers, getAssets, tickersLocked: () => locked,
    open(section = 'tickers') { (section === 'assets' ? assetSec : tickerSec).open = true; },
  };
}

// The presets the editor offers: the engine's (portfolio_default.json presets), then the ticker examples
// (portfolio_default.json instrument_presets with their precomputed results, else the catalog's list).
export function allPresets(defaults, cat) {
  const base = defaults && !defaults.error && Array.isArray(defaults.presets) && defaults.presets.length ? defaults.presets : FALLBACK_PRESETS;
  const withStats = defaults && !defaults.error && Array.isArray(defaults.instrument_presets) ? defaults.instrument_presets : [];
  const plain = cat && !cat.error && Array.isArray(cat.presets) ? cat.presets : [];
  const seen = new Set(base.map(p => p.id));
  const extra = [];
  for (const p of [...withStats, ...plain]) if (p && p.id && !seen.has(p.id)) { seen.add(p.id); extra.push(p); }
  return [...base, ...extra];
}

// ---------------------------------------------------------------- caveats of the tickers a result holds

// The correlation of a ticker with US equity in the build's diagnostics: {lo, hi, hist} over the scenarios.
function corrRange(t) {
  const d = CAT?.diagnostics;
  if (!d || typeof d !== 'object') return null;
  const sims = [];
  let hist = null;
  for (const s of Object.values(d)) {
    const c = s?.ticker_corr_us_equity?.[t];
    if (isNum(c?.sim)) sims.push(c.sim);
    if (isNum(c?.hist)) hist = c.hist;
  }
  return sims.length ? { lo: Math.min(...sims), hi: Math.max(...sims), hist } : null;
}
const corrNums = (list) => list.map(t => {
  const r = corrRange(t);
  return r ? `${t} 模拟 ${r.lo.toFixed(2)}–${r.hi.toFixed(2)}${isNum(r.hist) ? `、历史 ${r.hist.toFixed(2)}` : ''}` : null;
}).filter(Boolean).join('；');

// Card listing every caveat (chip) of the tickers a portfolio / withdrawal result holds, with the tickers it
// concerns, the wipe-out probabilities, and the held tickers' start prices. null for a portfolio without tickers
// (and without gold). scenarioYears: the scenario paths' length (wipe-outs are counted at their end).
export function holdingsCard(res, { scenarioYears = 15, include = [] } = {}) {
  const { h, card } = ui;
  const ins = res?.instruments;
  const tickers = (ins?.tickers || []).filter(t => isTicker(t));
  const gold = (res?.weights?.gold || 0) > 0;
  if (!tickers.length && !gold) return null;
  const flags = ins?.flags || {};
  const by = {};
  for (const t of tickers) for (const f of flags[t] || []) (by[f] = by[f] || []).push(t);
  if (gold) (by.gold_avg = by.gold_avg || []).unshift('黄金');
  const items = [];
  for (const key of [...CHIP_ORDER, ...Object.keys(by).filter(k => !CHIP_ORDER.includes(k))]) {
    const held = by[key];
    if (!held || !held.length) continue;
    const text = ins?.chips?.[key] || chipText(key) || key;
    let extra = null;
    if (key === 'credit_link') {
      const n = corrNums(held.filter(t => t !== '黄金'));
      extra = `：BDC、MLP 主要靠信用利差这个因子，而月度引擎里信用利差与股市的联动只有历史的一半左右，所以含它们的组合分散效果被高估、风险被低估。${n ? `本次构建与美股的季度相关：${n}（至少一个情景低于历史超过 0.15）。` : ''}季度模型应当直接复现这个联动，在那之前这条提示一直显示。`;
    }
    if (key === 'credit_link_ok') {
      const n = corrNums(held.filter(t => t !== '黄金'));
      extra = `：信用利差由月度引擎的利差与股市回报桥接，但这些标的与美股的模拟相关在每个情景里都离历史不到 0.15${n ? `（${n}）` : ''}，所以不提示风险被低估。`;
    }
    if (key === 'payout_cut') extra = '：粘性分配的误差修正在高收益率时不稳定，这条规则让分配不会把持仓的价值整个付出去；倍数是假设，样本里没有可用来估计它的削减。';
    if (key === 'rv_level') extra = '：用实际的期内波动时，杠杆公式在 2006–18 年和 2019–26 年两个窗口都在 T3 容差内；误差来自只用月度回报推出的期内波动水平（2019–26 年被低估约 15%）。';
    if (key === 'tax') {
      const kinds = [tickers.includes('QQQI') ? 'QQQI 的资本返还' : null, tickers.some(t => MLP.includes(t)) ? 'MLP 的 K-1 收入' : null,
        tickers.some(t => BDC.includes(t)) ? 'BDC 的普通收入' : null].filter(Boolean);
      extra = kinds.length ? `：${kinds.join('、')}都不区分税务性质。` : '：分红、利息和价格回报都按税前计算。';
    }
    if (key === 'no_2008') extra = '：估计型标的只有 2013 年起的季末价格，2008 年那样的危机行为（BDC、MLP 的回撤与削减分配）无法检验。';
    items.push(h('li', {}, ui.chip(text, CHIP_KIND[key] ?? ''), ' ', h('span', { class: 'muted' }, held.join('、')), extra ? h('span', { class: 'tk-li-extra' }, extra) : null));
  }
  if (tickers.includes('QQQI')) {
    items.unshift(h('li', {}, ui.chip('QQQI 分配大部分是资本返还', 'warn'), ' ',
      h('span', { class: 'tk-li-extra' }, `${CAT?.tickers?.QQQI?.tax_note || QQQI_ROC}。“先用股息”用掉的 QQQI 现金里有一部分其实是在卖出本金。`)));
  }
  const wipe = ins?.wipeout_prob_scenario_end || {};
  for (const t of tickers.filter(x => LEVERAGED.includes(x))) {
    items.unshift(h('li', {}, ui.chip(`${t} 可能归零`, 'bad'), ' ',
      h('span', { class: 'tk-li-extra' }, `按情景概率加权，${scenarioYears} 年内某一天纳指跌幅达到 1/${t === 'QLD' ? 2 : 3} 而归零的概率 ${fmtProb(wipe[t])}（外推，主要由 1987-10-19 一天决定）；`
        + '按月再平衡的组合里归零的这部分权重之后按 0 回报计（相当于不计息的现金），提取时归零的持仓不再有价值。')));
  }
  if (include.length && tickers.some(isEstimated)) {
    items.push(h('li', {}, ui.chip('自定义情景', 'accent'), ' ', h('span', { class: 'tk-li-extra' },
      '纳入的自定义情景在第 16 年起接续抽路径时，估计型标的会换一次 α 抽样（内置情景按路径固定，续抽前后不变）。')));
  }
  const start = ins?.start || {};
  const rows = tickers.map(t => ({ t, w: res.weights?.[t], kind: KIND_NAMES[tickerKind(t)] || '', s: start[t] || {} }));
  const tbl = rows.length ? ui.table([
    { label: '标的', value: (r) => r.t, fmt: (t) => h('span', { title: tickerName(t) }, h('b', {}, t), ' ', h('span', { class: 'muted' }, shortOf(t))) },
    { label: '权重', value: (r) => r.w, fmt: (x) => fmtShare(x) },
    { label: '类型', value: (r) => r.kind },
    { label: '起点价格', value: (r) => r.s.price, fmt: (x) => fmtUsd(x) },
    { label: '起点价格的来源', value: (r) => r.s, wrap: true, fmt: (s) => h('span', { class: 'muted' }, basisText(s.basis, true) || s.asof || '—') },
  ], rows, { stack: true }) : null;
  if (!tickers.length) {                 // gold only: the asset class, no ticker
    return card({ title: '黄金的口径', cls: 'span-all tk-holdings', sub: '组合里没有标的；黄金是资产类别本身（伦敦金价月均价）。' },
      h('ul', { class: 'tk-caveats' }, items));
  }
  return card({ title: '所持标的：局限与起点', cls: 'span-all tk-holdings',
    sub: '每个提示条都是设计里列出的一项局限，后面是它涉及的标的；预测（带判断的假设）只发生在情景层，标的按配方在同一批路径上推出，不预测个股事件。' },
  h('ul', { class: 'tk-caveats' }, items), tbl);
}

// ---------------------------------------------------------------- 标的推演: one ticker on every scenario path

const MEASURES = [
  { value: 'price', label: '价格（美元）', key: 'price', what: '每股价格（美元，除息）', gold: '金价（美元/盎司，伦敦金月均价）' },
  { value: 'total', label: '总回报', key: 'total', what: '期初 1 美元的财富（分配全部再投资）' },
  { value: 'total_real', label: '实际总回报', key: 'total_real', what: '期初 1 美元的实际购买力（分配再投资，扣除各路径的 CPI）' },
];
const PART_COLORS = ['#3b5bdb', '#e8590c', '#7048e8', '#1098ad', '#fab005', '#c2255c', '#495057', '#0b7285', '#5c940d', '#a61e4d'];
// round bounds for the small multiples' shared axis: 0 .. a 1-2-2.5-5 step (linear), whole decades (log)
function niceBounds(lo, hi, log) {
  if (!(isNum(lo) && isNum(hi) && hi > lo)) return {};
  if (log) return { min: 10 ** Math.floor(Math.log10(Math.max(lo, 1e-6))), max: 10 ** Math.ceil(Math.log10(hi)) };
  const mag = 10 ** Math.floor(Math.log10(hi));
  const top = [1, 2, 2.5, 5, 10].map(k => k * mag).find(x => x >= hi - 1e-12) || 10 * mag;
  const min = lo >= 0 ? 0 : -[1, 2, 2.5, 5, 10].map(k => k * 10 ** Math.floor(Math.log10(-lo))).find(x => x >= -lo - 1e-12);
  // a round interval that divides the span (2-6 steps), so the top tick is the max itself (no 90 / 100 crowding)
  const span = top - min;
  const isNice = (x) => { const m = x / 10 ** Math.floor(Math.log10(x)); return [1, 2, 2.5, 5, 10].some(k => Math.abs(m - k) < 1e-9); };
  const step = [5, 4, 3, 6, 2].map(k => span / k).find(isNice) || span / 4;
  return { min, max: top, step };
}
// the recipes the volatility state enters (the leveraged funds' drag, QQQI's premiums, the estimated residuals)
const VOL_TICKERS = (t) => LEVERAGED.includes(t) || t === 'QQQI' || isEstimated(t);
const OPTION_LABELS = { mean_2019_2026: '2019–26 均值', latest: '最新', train_2006_2018: '2006–18 训练期', mean_2024_2026: '2024–26 均值' };
const PCT_PARAMS = new Set(['s_eff', 'coverage', 'dist_rate']);
const VOL_OPTIONS = [{ value: 'model', label: '模型推出' }, { value: 'drawn', label: '抽到的历史月' }, { value: 'fixed', label: '固定（敏感性）' }];
const ALPHA_OPTIONS = [{ value: 'shrunk', label: '收缩后（默认）' }, { value: 'half', label: '减半' }, { value: 'raw', label: '原始' }];
const NO_CASH = { QLD: '杠杆 ETF 的现金分配记为 0（实际每年 0–0.3%），对“先用股息”的提取几乎没有影响。',
  TQQQ: '杠杆 ETF 的现金分配记为 0（实际每年 0–1.9%），对“先用股息”的提取几乎没有影响。',
  'BRK.B': '伯克希尔不分红：它的回报全部体现在价格里。', GLD: '黄金不产生现金：GLD 的赞助费每天以黄金支付，体现在价格里。',
  [GOLD]: '金价没有现金分配。' };

// The 标的推演 section: build it once per page visit. getContext() -> {include_custom?, scenario_probs?} of the
// page; scenarioNames(list) gives it the scenario list of scenarios.json. Loads when it first scrolls into view
// (or at once with focus). signal: the page visit's AbortSignal.
export function projection({ getContext, signal, focus = false } = {}) {
  injectWeightsStyle();
  injectStyles();
  const { h, card } = ui;
  const S = TK.pj;
  if (!tickerOrder().includes(S.ticker)) S.ticker = 'QQQ';
  let res = null, started = false, builtFor = null, names = {}, scrolled = false;
  const el = h('div', { class: 'tk-proj', id: 'tk-proj' });
  const alive = () => el.isConnected && !signal?.aborted;

  // ---- A: picker (chips by group) + the overview heatmap
  const chips = h('div', { class: 'tk-picker' });
  const busyLine = h('div', { class: 'tk-busy muted' });
  const vintLine = h('div', { class: 'tk-vint muted' });
  const overBox = ui.chartBox('tk-over');
  const overWrap = ui.wideChart(overBox, 720);
  const overDetails = h('details', { class: 'tk-over-details', open: window.innerWidth >= 760 ? true : null },
    h('summary', {}, '全部标的 × 情景：10 年年化总回报中位数一览（点击一行即选中该标的）'), overWrap,
    h('div', { class: 'tk-help' }, '构建时的默认参数；颜色跟随 K 线配色（涨色为正回报），超过 ±15% 的格子颜色不再加深。概率加权的汇总不能由各情景中位数相加得到，选中标的后见下方的加权汇总。'));
  const errBox = h('div', { class: 'weights-err' });
  function paintChips() {
    chips.replaceChildren(...tickerGroups().map(g => h('div', { class: 'tk-pick-group' },
      h('span', { class: 'tk-pick-label' }, g.name),
      h('div', { class: 'tk-pick-row' }, g.tickers.map(t => h('button', {
        type: 'button', class: `tk-pick${t === S.ticker ? ' on' : ''}`, title: tickerName(t), 'aria-pressed': t === S.ticker ? 'true' : 'false',
        onclick: () => select(t) }, t === GOLD ? '金价' : t))))));
  }
  const pickCard = card({ title: '选择标的', sub: '每个标的是一份“配方”：在核心模拟的同一批路径上推出价格、总回报和现金分配。这是模拟，不是价格预测：唯一带判断的是情景本身。' },
    chips, busyLine, vintLine, overDetails, errBox);

  // ---- B: recipe card
  const recipeTitle = h('h3', { class: 'card-title' });
  const recipeChips = h('div', { class: 'tk-chips' });
  const recipeBody = h('div', { class: 'tk-recipe' });
  const recipeCard = h('div', { class: 'card tk-recipe-card' }, h('div', { class: 'card-head' }, h('div', { class: 'card-titles' }, recipeTitle,
    h('div', { class: 'card-sub' }, '配方的文字公式、参数、验证与局限；改动参数只在本页即时重算 15 年的情景路径。')), recipeChips), recipeBody);

  // ---- C: fan + compare chips; small multiples
  const fanBox = ui.chartBox('tk-fan');
  const fanTitle = h('h3', { class: 'card-title' });
  const fanSub = h('div', { class: 'card-sub' });
  const scopeSel = h('select', { 'aria-label': '扇形图的范围', onchange: (e) => { S.scope = e.target.value; drawScoped(); } });
  const logBox = h('input', { type: 'checkbox', 'aria-label': '对数坐标', onchange: (e) => { S.log = e.target.checked; drawFan(); drawMulti(); } });
  const measureSeg = segBox(MEASURES.map(m => ({ value: m.value, label: m.label })), () => S.measure, (x) => { S.measure = x; drawFan(); drawMulti(); }, '扇形图的量');
  const freqSeg = segBox([{ value: 'M', label: '月' }, { value: 'Q', label: '季末' }], () => S.freq, (x) => { S.freq = x; drawFan(); drawMulti(); }, '频率');
  const compareBox = h('div', { class: 'tk-compare' });
  const flagBox = h('div', { class: 'tk-flags' });
  const fanCard = h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('div', { class: 'card-titles' }, fanTitle, fanSub),
      h('div', { class: 'card-actions' }, measureSeg, scopeSel, freqSeg, h('label', { class: 'tk-check' }, logBox, '对数坐标'))),
    flagBox, fanBox, compareBox);
  const multiGrid = h('div', { class: 'tk-multi' });
  const multiNote = h('div', { class: 'muted tk-legend' });
  const multiDetails = h('details', { class: 'tk-multi-details', open: window.innerWidth >= 760 ? true : null },
    h('summary', {}, '每个情景一张小图（与上图同一个量，所有小图同一纵轴）'), multiNote, multiGrid);
  multiDetails.addEventListener('toggle', () => { if (multiDetails.open) drawMulti(); });

  // ---- D: table; E: payouts; F: attribution; G: correlations; H: notes
  const tableBox = h('div', { class: 'tk-table' });
  const tableTitle = h('h3', { class: 'card-title' });
  const tableCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('div', { class: 'card-titles' }, tableTitle,
    h('div', { class: 'card-sub' }, 'p10 / p50 / p90 是该情景全部路径的分位；回撤先按月度路径，估计型标的的季内月度是布朗桥拆分（假设），所以同时给出季末口径（季末回撤比月度浅）。“15 年”指情景路径的终点。'))), tableBox);
  const distTitle = h('h3', { class: 'card-title' });
  const distSub = h('div', { class: 'card-sub' });
  const distKpis = h('div', { class: 'grid cols-4 tk-kpis' });
  const dpsBox = ui.chartBox('tk-dps');
  const cutBox = ui.chartBox('tk-cut');
  const distNote = h('div', { class: 'tk-dist-note' });
  const distEmpty = h('div');
  const distCharts = h('div', { class: 'weights-pair tk-dist-charts' },
    h('div', {}, h('div', { class: 'portfolio-label' }, '每股年度分配（美元）'), dpsBox),
    h('div', {}, h('div', { class: 'portfolio-label' }, '10 年内削减的概率（某年现金比第一年低 10% / 20% 以上）'), cutBox));
  const distCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('div', { class: 'card-titles' }, distTitle, distSub)),
    distEmpty, distKpis, distNote, distCharts);
  const attrBox = ui.chartBox('tk-attr');
  const attrTitle = h('h3', { class: 'card-title' });
  const attrSeg = segBox([{ value: 'median', label: '中位路径' }, { value: 'mean', label: '路径平均' }], () => S.attr, (x) => { S.attr = x; drawAttr(); }, '分解口径');
  const attrCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('div', { class: 'card-titles' }, attrTitle,
    h('div', { class: 'card-sub' }, '每个情景一根横条：年化对数回报按配方拆成可加的几项（菱形 = 合计）。中位路径 = 总回报处于中位的那条路径；路径平均另含概率加权的汇总行。')),
  h('div', { class: 'card-actions' }, attrSeg)), attrBox);
  const corrBox = ui.chartBox('tk-corr');
  const corrTitle = h('h3', { class: 'card-title' });
  const corrHead = h('div', { class: 'tk-corr-head' });
  const corrSeg = segBox([{ value: 'assets', label: '资产类别' }, { value: 'tickers', label: '其他标的' }], () => S.corr, (x) => { S.corr = x; drawCorr(); }, '相关的对象');
  const corrCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('div', { class: 'card-titles' }, corrTitle,
    h('div', { class: 'card-sub' }, '季度对数总回报，逐条路径计算相关系数，再取中位数（加权汇总按情景概率）；范围跟随扇形图的选择。')),
  h('div', { class: 'card-actions' }, corrSeg)), corrHead, corrBox);
  const notesList = h('ul', { class: 'list' });
  const metaList = h('ul', { class: 'list' });
  const notesCard = card({ cls: 'tk-notes' }, h('details', {}, h('summary', {}, '计算口径与局限（展开）'),
    h('div', { class: 'portfolio-label' }, '本次推演'), notesList, h('div', { class: 'portfolio-label' }, '配方运行信息'), metaList));
  const resultBox = h('div', { class: 'tk-results', style: { display: 'none' } },
    fanCard, card({ cls: 'tk-multi-card' }, multiDetails), tableCard, distCard,
    h('div', { class: 'weights-pair' }, attrCard, corrCard), notesCard);

  el.append(h('div', { class: 'section-title', id: 'tk-proj-title' }, '标的推演：单个标的在每个情景里的价格、总回报与分配'),
    h('p', { class: 'lead tk-lead' }, ...(CAT?.intro?.length ? CAT.intro.map((t, i) => (i ? [h('br'), t] : t)) : ['每个标的都是一份配方：在核心模拟的同一批路径上推出它的价格、总回报和现金分配。'])),
    h('div', { class: 'grid tk-top' }, pickCard, recipeCard), resultBox);

  // ---------------------------------------------------------------- requests
  const specsOf = (t) => {
    const e = CAT?.tickers?.[t] || {};
    const list = [...(e.params || [])];
    return list.filter(p => p.id !== 'alpha_mode');
  };
  const isDefault = (spec, val) => {
    if (val === undefined || val === null || val === '') return true;
    if (val === spec.default) return true;
    if (typeof val === 'string' && spec.options && isNum(spec.default) && spec.options[val] === spec.default) return true;
    if (isNum(val) && isNum(spec.default) && Math.abs(val - spec.default) < 1e-9) return true;
    return false;
  };
  function body() {
    const t = S.ticker;
    const b = { ticker: t };
    const p = {};
    const mine = S.params[t] || {};
    for (const spec of specsOf(t)) if (!isDefault(spec, mine[spec.id])) p[spec.id] = mine[spec.id];
    if (isNum(mine.price) && mine.price > 0) p.price = mine.price;
    if (Object.keys(p).length) b.params = p;
    if (VOL_TICKERS(t) && S.vol !== 'model') {
      b.vol_source = S.vol;
      if (S.vol === 'fixed') b.fixed_vol = Math.min(150, Math.max(5, S.fixedVol)) / 100;
    }
    if (isEstimated(t) && S.alpha !== 'shrunk') b.alpha_mode = S.alpha;
    const c = getContext ? getContext() || {} : {};
    if (c.include_custom?.length) b.include_custom = c.include_custom.slice();
    if (c.scenario_probs) b.scenario_probs = c.scenario_probs;
    return b;
  }
  const go = liveRequest(async (b) => ({ r: await api.post('instruments/project', b), b }), {
    start: () => {
      busyLine.replaceChildren(h('span', { class: 'spinner' }), ` 正在计算 ${S.ticker === GOLD ? '金价' : S.ticker} 在全部情景路径上的推演…（第一次约 3–5 秒，之后有缓存）`);
      resultBox.classList.add('weights-busy');
    },
    done: ({ r }) => {
      if (!alive()) return;
      resultBox.classList.remove('weights-busy');
      busyLine.replaceChildren();
      errBox.replaceChildren();
      if (r.ticker !== S.ticker) { refresh(); return; }     // picked another ticker meanwhile
      show(r);
      if (focus && !scrolled) { scrolled = true; ui.pinIntoView(el, { block: 'start' }); }
    },
    error: (err) => {
      if (!alive()) return;
      resultBox.classList.remove('weights-busy');
      busyLine.replaceChildren();
      const n = ui.notice(`标的推演失败：${errorText(err)}`, 'error');
      n.append(' ', h('button', { class: 'btn weights-retry', onclick: () => refresh() }, '重试'));
      errBox.replaceChildren(n);
    },
  });
  const refresh = () => { if (started && alive()) go(body()); };
  const later = debounce(refresh, 600);
  function start() { if (started) return; started = true; refresh(); }
  function select(t) {
    if (!t || t === S.ticker) { start(); return; }
    S.ticker = t;
    S.compare = null;
    paintChips();
    buildRecipe();
    started = true;
    refresh();
  }

  // ---------------------------------------------------------------- B: the recipe card
  const pctIn = (val) => (isNum(val) ? String(+(val * 100).toFixed(3)) : '');
  // an input outside its range: say so next to it, keep computing with the value in force, and put that value back
  // in the field when the reader leaves it
  function rangeGuard(input, msg, check, active) {
    const err = h('div', { class: 'tk-input-err', role: 'alert' });
    const paint = () => {
      const bad = check();
      err.textContent = bad ? `${bad}；仍按 ${active()} 计算` : '';
      input.classList.toggle('is-bad', !!bad);
      return !bad;
    };
    input.addEventListener('change', () => { if (!paint()) { input.value = msg(); paint(); } });
    return { err, paint };
  }
  function paramControl(t, spec) {
    const mine = (S.params[t] = S.params[t] || {});
    const cur = mine[spec.id];
    const set = (val) => { if (val === undefined) delete mine[spec.id]; else mine[spec.id] = val; later(); };
    const label = h('div', { class: 't' }, spec.name || spec.id);
    if (!spec.options && !spec.range) {                       // a switch (the ruin draw)
      const cb = h('input', { type: 'checkbox', checked: (cur ?? spec.default) ? true : null });
      cb.addEventListener('change', () => set(cb.checked === spec.default ? undefined : cb.checked));
      return h('div', { class: 'withdrawal-field' }, label, h('label', { class: 'tk-check' }, cb,
        spec.id === 'ruin' ? `某一天纳指跌幅达到 1/${t === 'QLD' ? 2 : 3} 时基金归零（抽签）` : '开'));
    }
    const pctUnit = PCT_PARAMS.has(spec.id);
    const opts = spec.options || {};
    const numericOpts = Object.values(opts).length && Object.values(opts).every(isNum);
    if (Object.keys(opts).length) {
      const keyOf = (val) => (typeof val === 'string' && val in opts ? val
        : isNum(val) && numericOpts ? Object.keys(opts).find(k => Math.abs(opts[k] - val) < 1e-9) || 'custom' : null);
      const curKey = keyOf(cur ?? spec.default) || (spec.range ? 'custom' : Object.keys(opts)[0]);
      const sel = h('select', { 'aria-label': spec.name },
        Object.entries(opts).map(([k, x]) => h('option', { value: k, selected: k === curKey ? true : null },
          isNum(x) ? `${OPTION_LABELS[k] || k}（${pctUnit ? pct(x, 2) : x}）` : String(x))),
        spec.range ? h('option', { value: 'custom', selected: curKey === 'custom' ? true : null }, '自定义…') : null);
      const lo = spec.range?.[0], hi = spec.range?.[1];
      const custom = h('input', { type: 'number', step: 'any', min: isNum(lo) ? (pctUnit ? lo * 100 : lo) : null, max: isNum(hi) ? (pctUnit ? hi * 100 : hi) : null,
        value: curKey === 'custom' ? (pctUnit ? pctIn(cur ?? spec.default) : String(cur ?? spec.default)) : '', 'aria-label': `${spec.name}（自定义）` });
      const shownVal = () => { const v0 = mine[spec.id] ?? spec.default; const x = typeof v0 === 'string' ? opts[v0] : v0; return pctUnit ? pctIn(x) : String(x); };
      const guard = rangeGuard(custom, shownVal, () => {
        const x = Number(custom.value);
        if (custom.value === '' || !Number.isFinite(x)) return '请输入数字';
        const val = pctUnit ? x / 100 : x;
        return (isNum(lo) && val < lo) || (isNum(hi) && val > hi) ? `超出范围 ${pctUnit ? `${pct(lo, 1)}–${pct(hi, 1)}` : `${lo}–${hi}`}` : '';
      }, () => (pctUnit ? `${shownVal()}%` : shownVal()));
      const customWrap = h('span', { class: 'withdrawal-inline', style: { display: curKey === 'custom' ? '' : 'none' } }, custom, pctUnit ? h('span', { class: 'muted' }, '%') : null);
      sel.addEventListener('change', () => {
        customWrap.style.display = sel.value === 'custom' ? '' : 'none';
        if (sel.value === 'custom') { if (custom.value === '') custom.value = pctUnit ? pctIn(isNum(spec.default) ? spec.default : opts[spec.default]) : ''; readCustom(); return; }
        set(isDefault(spec, sel.value) ? undefined : sel.value);
      });
      const readCustom = () => {
        if (!guard.paint()) return;
        const x = Number(custom.value);
        const val = pctUnit ? x / 100 : x;
        set(isDefault(spec, val) ? undefined : val);
      };
      custom.addEventListener('input', readCustom);
      return h('div', { class: 'withdrawal-field' }, label, h('div', { class: 'withdrawal-inline' }, sel, customWrap),
        spec.range ? h('div', { class: 'muted' }, `自定义范围 ${pctUnit ? `${pct(lo, 1)}–${pct(hi, 1)}` : `${lo}–${hi}`}`) : null, guard.err);
    }
    const lo = spec.range[0], hi = spec.range[1];
    const input = h('input', { type: 'number', step: 'any', min: pctUnit ? lo * 100 : lo, max: pctUnit ? hi * 100 : hi,
      value: pctUnit ? pctIn(cur ?? spec.default) : String(cur ?? spec.default), 'aria-label': spec.name });
    const shownVal = () => (pctUnit ? pctIn(mine[spec.id] ?? spec.default) : String(mine[spec.id] ?? spec.default));
    const guard = rangeGuard(input, shownVal, () => {
      const x = Number(input.value);
      if (input.value === '' || !Number.isFinite(x)) return '请输入数字';
      const val = pctUnit ? x / 100 : x;
      return val < lo || val > hi ? `超出范围 ${pctUnit ? `${pct(lo, 1)}–${pct(hi, 1)}` : `${lo}–${hi}`}` : '';
    }, () => (pctUnit ? `${shownVal()}%` : shownVal()));
    input.addEventListener('input', () => {
      if (!guard.paint()) return;
      const x = Number(input.value);
      const val = pctUnit ? x / 100 : x;
      set(isDefault(spec, val) ? undefined : val);
    });
    const unit = pctUnit ? '%' : spec.id === 'cy_halflife_years' ? '年' : '';
    return h('div', { class: 'withdrawal-field' }, label, h('div', { class: 'withdrawal-inline' }, input, unit ? h('span', { class: 'muted' }, unit) : null),
      h('div', { class: 'muted' }, `范围 ${pctUnit ? `${pct(lo, 1)}–${pct(hi, 1)}` : `${lo}–${hi}`}，默认 ${pctUnit ? pct(spec.default, 2) : spec.default}`), guard.err);
  }
  const SAMPLE = { in: ['样本内', 'warn'], oos: ['样本外', 'accent'], full: ['全期', ''] };
  function validationLine(v) {
    const bits = [];
    if (isNum(v.actual) && isNum(v.model)) bits.push(`实际 ${pct(v.actual)}/年，模型 ${pct(v.model)}/年（差 ${ui.fmtPct(v.model - v.actual, 1, true)}）`);
    if (isNum(v.te_ann)) bits.push(`跟踪误差 ${pct(v.te_ann, 2)}/年`);
    if (isNum(v.rmse_monthly)) bits.push(`月 RMSE ${pct(v.rmse_monthly, 2)}`);
    if (isNum(v.cover80)) bits.push(`80% 区间覆盖 ${pct(v.cover80, 0)}`);
    if (Array.isArray(v.r2)) bits.push(`样本外 R² ${v.r2.map(x => (isNum(x) ? x.toFixed(2) : '—')).join(' / ')}`);
    if (Array.isArray(v.r2_market)) bits.push(`只用美股大盘 ${v.r2_market.map(x => (isNum(x) ? x.toFixed(2) : '—')).join(' / ')}`);
    const tag = SAMPLE[v.sample];
    const t3 = typeof v.t3_pass === 'boolean' ? ui.chip(v.t3_pass ? '达到 T3 容差' : '未达到 T3 容差', v.t3_pass ? 'good' : 'bad') : null;
    return [tag ? ui.chip(tag[0], tag[1]) : null, tag ? ' ' : null, `${v.what || ''}：${bits.join('；') || '—'}`, t3 ? ' ' : null, t3];
  }
  const startInfo = () => (res?.ticker === S.ticker ? res.start : null) || CAT?.tickers?.[S.ticker]?.start || {};
  const startDd = h('dd', {});
  function paintStart() {
    const st = startInfo();
    const unit = S.ticker === GOLD ? ' 美元/盎司' : '';
    const px = st.price ? (unit ? `${fmtUsd(st.price).replace(' 美元', '')}${unit}` : fmtUsd(st.price)) : '—';
    const built = st.build?.price ? `（构建时 ${fmtUsd(st.build.price)}：${basisText(st.build.basis, true)}）` : '';
    startDd.replaceChildren(px, st.asof ? ` · ${st.asof}` : '', st.basis ? ` · ${basisText(st.basis)}` : '', built);
  }
  function buildRecipe() {
    const t = S.ticker;
    builtFor = t;
    const e = CAT?.tickers?.[t] || {};
    const est = isEstimated(t);
    recipeTitle.textContent = `配方：${tickerName(t)}`;
    const kids = [];
    kids.push(h('p', { class: 'tk-formula' }, e.card || '（标的目录没有读到，配方说明暂缺）'));
    const st = startInfo();
    const fee = isNum(e.fee) ? h('span', { title: e.fee_source ? `费率出处（原文）：${e.fee_source}` : null }, `费率 ${pct(e.fee, 2)}/年`) : null;
    kids.push(h('dl', { class: 'kv tk-kv' },
      h('dt', {}, '现金分配'), h('dd', {}, e.cash || '—', fee ? ['；', fee] : null),
      h('dt', {}, '起点'), startDd,
      h('dt', {}, '类型'), h('dd', {}, KIND_NAMES[tickerKind(t)] || '—', e.native ? ` · 原生频率 ${{ M: '月', Q: '季', any: '任意' }[e.native] || e.native}` : '')));
    const params = h('div', { class: 'withdrawal-form tk-params' });
    for (const spec of specsOf(t)) params.append(paramControl(t, spec));
    if (est) {
      params.append(h('div', { class: 'withdrawal-field' }, h('div', { class: 't' }, 'α（相对风格组合的超额回报）'),
        segBox(ALPHA_OPTIONS, () => S.alpha, (x) => { S.alpha = x; paintRecipeChips(); later(); }, 'α'),
        h('div', { class: 'muted' }, CAT?.alpha_note || 'α 是历史上相对风格组合的超额回报。')));
    }
    if (VOL_TICKERS(t)) {                     // QQQ, TLT, GLD and 金价 do not use the volatility state
      const fixedIn = h('input', { type: 'number', min: 5, max: 150, step: 1, value: String(S.fixedVol), 'aria-label': '固定年化波动（%）' });
      const guard = rangeGuard(fixedIn, () => String(S.fixedVol), () => {
        const x = Number(fixedIn.value);
        if (fixedIn.value === '' || !Number.isFinite(x)) return '请输入数字';
        return x < 5 || x > 150 ? '超出范围 5%–150%' : '';
      }, () => `${S.fixedVol}%`);
      const fixedWrap = h('span', { class: 'withdrawal-inline', style: { display: S.vol === 'fixed' ? '' : 'none' } }, fixedIn, h('span', { class: 'muted' }, '% 年化'));
      fixedIn.addEventListener('input', () => {
        if (!guard.paint()) return;
        S.fixedVol = Number(fixedIn.value);
        later();
      });
      params.append(h('div', { class: 'withdrawal-field' }, h('div', { class: 't' }, '期内波动'),
        h('div', { class: 'withdrawal-inline' }, segBox(VOL_OPTIONS, () => S.vol, (x) => {
          S.vol = x; fixedWrap.style.display = x === 'fixed' ? '' : 'none'; paintRecipeChips(); paintFlags(); later();
        }, '期内波动'), fixedWrap), guard.err,
        h('div', { class: 'muted' }, '模型：由核心自己的回报路径推出期内波动（默认）；抽到的历史月只作对照；固定值在历史上有系统偏差，只作敏感性。'
          + '这个选择对本页所有用到期内波动的标的生效（杠杆 ETF、QQQI 和估计型标的；QQQ、TLT、GLD 与金价不用它）。')));
    }
    const mine = (S.params[t] = S.params[t] || {});
    const priceIn = h('input', { type: 'number', min: 0, step: 'any', value: isNum(mine.price) ? String(mine.price) : '',
      placeholder: st.price ? String(+st.price.toFixed(2)) : '', 'aria-label': '起点价格（美元）' });
    const priceGuard = rangeGuard(priceIn, () => (isNum(mine.price) ? String(mine.price) : ''), () => {
      if (priceIn.value === '') return '';
      const x = Number(priceIn.value);
      return Number.isFinite(x) && x > 0 ? '' : '起点价格必须是大于 0 的数字';
    }, () => (isNum(mine.price) ? `${mine.price} 美元` : '构建时的起点价格'));
    priceIn.addEventListener('input', () => {
      if (!priceGuard.paint()) return;
      if (priceIn.value === '') { delete mine.price; later(); return; }
      mine.price = Number(priceIn.value);
      later();
    });
    params.append(h('div', { class: 'withdrawal-field' }, h('div', { class: 't' }, '起点价格（美元，可改成今天的价格）'),
      h('div', { class: 'withdrawal-inline' }, priceIn, h('span', { class: 'muted' }, '美元')), priceGuard.err,
      h('div', { class: 'muted' }, est ? '只改起点收益率（现金与价格的拆分），不改总回报路径。' : '只把价格路径按比例换算到这个起点，总回报不变。')));
    const reset = h('button', { class: 'btn', onclick: () => { S.params[t] = {}; if (VOL_TICKERS(t)) S.vol = 'model'; S.alpha = 'shrunk'; buildRecipe(); refresh(); } }, '恢复默认参数');
    kids.push(h('div', { class: 'portfolio-label' }, '参数'), params, h('div', { class: 'row tk-reset' }, reset));
    if (e.breakeven) {
      const text = String(e.breakeven).replace(/fee_L/g, 'L 倍基金费率').replace(/fee_1/g, 'QQQ 费率');
      kids.push(h('div', { class: 'portfolio-callout' }, h('b', {}, '盈亏平衡：'), text,
        h('div', { class: 'muted tk-legend' }, 'μ = 纳指 100 的年化算术价格回报，q = 股息率，b = 联邦基金利率，s = 互换利差，σ = 期内年化波动，L = 杠杆倍数。')));
    }
    const limits = [...(e.limits || []), ...(est ? CAT?.estimated_common?.limits || [] : []), ...(TICKER_NOTES[t] || [])];
    if (BDC.includes(t)) limits.push(BDC_NOTE);
    if (MLP.includes(t)) limits.push(MLP_NOTE);
    const sources = [...(e.sources || []), ...(est ? CAT?.estimated_common?.sources || [] : [])]
      .map(x => String(x).replace(/datasets\/gold-prices/g, '伦敦金价月均价数据集'));
    kids.push(
      h('details', { class: 'tk-fold' }, h('summary', {}, `验证（${(e.validation || []).length || 0} 项，标明样本内 / 样本外）`),
        (e.validation || []).length ? h('ul', { class: 'list' }, e.validation.map(v => h('li', {}, validationLine(v))))
          : h('p', { class: 'muted' }, t === GOLD || t === 'GLD' ? '金价就是核心的黄金资产（GLD 再减费率），没有另外的拟合，长期水平与历史一致。' : '—')),
      h('details', { class: 'tk-fold' }, h('summary', {}, '数据来源'), h('ul', { class: 'list' },
        [...new Set(sources)].map(s => h('li', {}, s)),
        isNum(e.fee) ? h('li', { class: 'muted' }, `费率 ${pct(e.fee, 2)}/年只来自一张费率表（按财年，注明招募书或年报出处；把鼠标停在上面“费率”上可看原文出处）。`) : null,
        h('li', { class: 'muted' }, '条款禁止自动访问的发行人数据（BlackRock / iShares、Invesco）一律不用；SEC 请求只发往 sec.gov。'))),
      h('details', { class: 'tk-fold' }, h('summary', {}, `局限（${limits.length} 条）`), h('ul', { class: 'list' }, limits.map(x => h('li', {}, x)))),
      h('p', { class: 'tk-foot muted' }, '组合评估与提取页使用构建时的默认参数', defaultParamsText(t), '；这里的改动只在本页重算 15 年的情景路径，',
        '组合页和提取页暂时不能改用这些参数（需要带参数重新生成标的路径，目前还不支持）。'));
    recipeBody.replaceChildren(...kids);
    paintStart();
    paintRecipeChips();
    paintFlags();
  }
  function defaultParamsText(t) {
    const d = CAT?.built?.defaults || {};
    const e = CAT?.tickers?.[t] || {};
    const bits = [];
    for (const spec of e.params || []) {
      if (spec.id === 's_eff') { const k = d.letf_s_eff?.[t] || spec.default; const x = spec.options?.[k]; bits.push(`互换利差 ${isNum(x) ? pct(x, 2) : '—'}（${OPTION_LABELS[k] || k}）`); }
      if (spec.id === 'ruin') bits.push('爆仓抽签开');
      if (spec.id === 'coverage') bits.push(`覆盖率 ${pct(spec.default, 1)}`);
      if (spec.id === 'dist_rule') bits.push(`分配 ${spec.options?.[spec.default] || ''}`);
      if (spec.id === 'cy_halflife_years') bits.push(`当期收益率半衰期 ${spec.default} 年`);
    }
    if (isEstimated(t)) bits.push('α 收缩后（提取页可切换）');
    if (VOL_TICKERS(t)) bits.push('期内波动由模型推出');
    return bits.length ? `（${bits.join('、')}）` : '';
  }
  function paintRecipeChips() {
    const t = S.ticker;
    const kind = tickerKind(t);
    const list = [ui.chip(KIND_NAMES[kind] || kind, kind === 'estimated' ? 'warn' : 'accent')];
    if (VOL_TICKERS(t)) list.push(ui.chip(`期内波动：${S.vol === 'fixed' ? `固定 ${S.fixedVol}%（敏感性，不是推演）` : S.vol === 'drawn' ? '抽到的历史月（对照）' : '模型'}`, S.vol === 'fixed' ? 'warn' : ''));
    if (isEstimated(t)) list.push(ui.chip(`α：${ALPHA_OPTIONS.find(o => o.value === S.alpha)?.label || S.alpha}`, S.alpha === 'shrunk' ? '' : 'warn'));
    if (res?.ticker === t && res.recomputed) list.push(ui.chip('本页参数已即时重算', 'accent'));
    list.push(ui.chip('税前', ''));
    recipeChips.replaceChildren(...list);
  }

  // ---------------------------------------------------------------- helpers on a response
  const per = () => res?.per_scenario || [];
  const perOf = (id) => per().find(e => e.id === id);
  const W = () => res?.aggregate?.weighted || {};
  const labelOf = (e) => (e ? (names[e.id] ? shortName(names[e.id]) : shortName(e.name)) : '');
  const fansOf = (scope) => (scope === 'weighted' ? W().fans : perOf(scope)?.fans);
  const scopeName = () => (S.scope === 'weighted' ? '加权汇总' : labelOf(perOf(S.scope)) || '—');
  const measure = () => MEASURES.find(m => m.value === S.measure) || MEASURES[0];
  const tl = () => (S.ticker === GOLD ? '金价' : `${S.ticker} `);      // before 的 / 与: "金价的", "TLT 的"
  const useLog = () => (S.log === null ? LEVERAGED.includes(S.ticker) : !!S.log);
  // the small multiples share one axis: on a linear axis the largest p95 squeezes every median flat, so price and
  // total default to the log axis there (the reader's own log choice wins)
  const multiLog = () => (S.log === null ? S.measure !== 'total_real' || LEVERAGED.includes(S.ticker) : !!S.log);
  const fmtOf = () => (S.measure === 'price' ? (v) => fmtUsd(v) : multFmt);
  const axisOf = () => (S.measure === 'price' ? usdAxis : (v) => `${+(+v).toFixed(2)}`);
  const rankOrder = () => per().slice().sort((a, b) => (isNum(b.cagr10_p50) ? b.cagr10_p50 : -Infinity) - (isNum(a.cagr10_p50) ? a.cagr10_p50 : -Infinity));
  const colorOf = (id) => { const r = rankOrder(); const i = r.findIndex(e => e.id === id); return i < 0 ? INK3 : rankColor(i, r.length); };
  const compareIds = () => {
    if (Array.isArray(S.compare)) return S.compare.filter(id => perOf(id));
    const r = rankOrder();
    return r.length > 1 ? [r[0].id, r[r.length - 1].id] : [];
  };
  function sliceX() {
    const idx = S.freq === 'Q' ? res.quarter_index : null;
    const x = idx ? idx.map(i => res.labels[i]) : res.labels;
    const pick = (a) => (Array.isArray(a) ? (idx ? idx.map(i => a[i]) : a) : []);
    return { x, pick };
  }
  const startOf = () => (S.measure === 'price' ? (fansOf('weighted')?.price?.p50?.[0] ?? null) : 1);

  // ---------------------------------------------------------------- C: fans
  // the selected ticker's caveats, next to its results (INSTRUMENTS_DESIGN §7.5)
  function paintFlags() {
    const t = S.ticker;
    const mine = res?.ticker === t;
    const flags = new Set(mine ? res.flags || [] : CAT?.tickers?.[t]?.flags || []);
    if (t === GOLD || t === 'GLD') flags.add('gold_avg');
    if (S.vol === 'fixed' && VOL_TICKERS(t)) flags.add('fixed_vol');
    const ruinOff = LEVERAGED.includes(t) && (mine ? res.ruin === false : (S.params[t] || {}).ruin === false);
    if (ruinOff) flags.delete('ruin_extrap');
    const text = (k) => (mine ? res.chips?.[k] : null) || chipText(k) || k;
    const cl = mine ? res.credit_link : null;
    const tip = (k) => ((k === 'credit_link' || k === 'credit_link_ok') && cl && isNum(cl.sim_lo)
      ? `与美股的季度相关：模拟 ${cl.sim_lo.toFixed(2)}–${cl.sim_hi.toFixed(2)}、历史 ${cl.hist.toFixed(2)}；${cl.n_over}/${cl.n} 个情景低于历史超过 ${cl.tol}` : null);
    const list = [...CHIP_ORDER.filter(k => flags.has(k)), ...[...flags].filter(k => !CHIP_ORDER.includes(k))].map((k) => {
      const c = ui.chip(text(k), CHIP_KIND[k] ?? '');
      const tt = tip(k);
      if (tt) { c.title = tt; if (k === 'credit_link') c.append(` · ${cl.n_over}/${cl.n} 个情景`); }
      return c;
    });
    if (t === 'QQQI') list.unshift(ui.chip('分配约 99.6% 为资本返还', 'warn'));
    if (ruinOff) list.unshift(ui.chip('爆仓抽签已关闭：没有计算归零概率（敏感性）', 'warn'));
    else if (LEVERAGED.includes(t) && mine && isNum(W().wipeout_prob)) list.unshift(ui.chip(`15 年内归零的概率 ${fmtProb(W().wipeout_prob)}（加权，外推）`, 'bad'));
    flagBox.replaceChildren(...list);
  }
  function paintScope() {
    if (S.scope !== 'weighted' && !perOf(S.scope)) S.scope = 'weighted';
    scopeSel.replaceChildren(h('option', { value: 'weighted' }, '加权汇总（按情景概率）'),
      ...per().map(e => h('option', { value: e.id }, `${names[e.id] || e.name}（${pct(e.prob)}）`)));
    scopeSel.value = S.scope;
  }
  function paintCompare() {
    const on = new Set(compareIds());
    compareBox.replaceChildren(h('span', { class: 'muted' }, '虚线对比的情景（最多 4 个，默认最好与最差）：'),
      ...rankOrder().map(e => h('button', { type: 'button', class: `tk-cmp${on.has(e.id) ? ' on' : ''}`, title: e.name,
        style: on.has(e.id) ? { borderColor: colorOf(e.id), color: colorOf(e.id) } : null,
        onclick: () => {
          const cur = compareIds();
          S.compare = cur.includes(e.id) ? cur.filter(x => x !== e.id) : [...cur, e.id].slice(-4);
          paintCompare();
          drawFan();
        } }, labelOf(e))));
  }
  function drawFan() {
    if (!res) return;
    const m = measure();
    const bands0 = fansOf(S.scope)?.[m.key];
    if (!bands0) { charts.mount(fanBox).clear(); return; }
    const { x, pick } = sliceX();
    const bands = Object.fromEntries(Object.entries(bands0).map(([k, a]) => [k, pick(a)]));
    const extra = compareIds().filter(id => id !== S.scope).map(id => ({ name: labelOf(perOf(id)), data: pick(perOf(id).fans[m.key].p50), dashed: true, color: colorOf(id) }));
    if (S.scope !== 'weighted') extra.push({ name: '加权中位数', data: pick(W().fans[m.key].p50), dashed: true, color: INK });
    const s0 = startOf();
    if (isNum(s0)) extra.push({ name: '起点', data: x.map(() => s0), dashed: true, color: '#adb5bd' });
    logBox.checked = useLog();
    charts.fan(fanBox, { x, bands, fmt: fmtOf(), axisFmt: axisOf(), name: '中位数', color: ACCENT, extra, log: useLog(), logBase: 10 });
    const gold = S.ticker === GOLD;
    const t = gold ? '金价' : S.ticker;
    fanTitle.textContent = gold && S.measure === 'price' ? `金价（美元/盎司）：${scopeName()}` : `${t} ${m.label}：${scopeName()}`;
    const st = startInfo();
    const what = gold && m.gold ? m.gold : m.what;
    const at = gold ? `${res.labels?.[0] || ''} 月均价` : `${res.labels?.[0] || ''} 月末`;
    const px = S.measure === 'price' && st.price ? (gold ? ` ${fmtUsd(st.price).replace(' 美元', '')} 美元/盎司` : `的 ${fmtUsd(st.price)}`) : '';
    const basis = st.basis && S.measure === 'price' && !gold ? `（${basisText(st.basis)}）` : '';
    fanSub.textContent = `${what}。深色带 25%–75%、浅色带 5%–95% 的路径，实线是中位数，虚线是所选情景的中位数；起点是 ${at}${px}${basis}。`
      + `${S.freq === 'Q' ? '季末视图只取每季最后一个月。' : isEstimated(S.ticker) ? '估计型标的的季内月度路径为布朗桥拆分（假设），季末视图更接近它的原生频率。' : ''}`;
  }
  function drawMulti() {
    if (!res || !multiDetails.open) return;
    const m = measure();
    const { x, pick } = sliceX();
    const scopes = [{ id: 'weighted', name: '加权汇总', e: { cagr10_p50: W().cagr10_p50 } }, ...per().map(e => ({ id: e.id, name: labelOf(e), e }))];
    let lo = Infinity, hi = -Infinity;
    for (const s of scopes) {
      const b = fansOf(s.id)?.[m.key];
      for (const v of pick(b?.p5)) if (isNum(v) && (!useLog() || v > 0)) lo = Math.min(lo, v);
      for (const v of pick(b?.p95)) if (isNum(v)) hi = Math.max(hi, v);
    }
    const lg = multiLog();
    if (lg) { lo = Infinity; for (const s of scopes) for (const v of pick(fansOf(s.id)?.[m.key]?.p5)) if (isNum(v) && v > 0) lo = Math.min(lo, v); }
    const nb = niceBounds(lo, hi, lg);
    charts.disposeWithin(multiGrid);
    multiNote.textContent = lg ? '小图用对数纵轴（同一纵轴，便于比较中位数）；上方大图可切换。' : '';
    multiGrid.replaceChildren(...scopes.map(s => {
      const box = h('div', { class: 'chart tk-mini-chart' });
      const tile = h('div', { class: `tk-mini${s.id === S.scope ? ' on' : ''}`, title: '点击：在上面的大图里查看这个情景',
        onclick: () => { S.scope = s.id; scopeSel.value = s.id; drawScoped(); fanCard.scrollIntoView({ behavior: 'smooth', block: 'start' }); } },
      h('div', { class: 'tk-mini-head' }, h('b', {}, s.name), h('span', {}, '10 年中位 ', signed(s.e.cagr10_p50))), box);
      return tile;
    }));
    requestAnimationFrame(() => scopes.forEach((s, i) => {
      if (!alive()) return;
      const box = multiGrid.children[i]?.querySelector('.chart');
      const b0 = fansOf(s.id)?.[m.key];
      if (!box || !b0) return;
      const bands = Object.fromEntries(Object.entries(b0).map(([k, a]) => [k, pick(a)]));
      const chart = charts.fan(box, { x, bands, fmt: fmtOf(), axisFmt: axisOf(), name: '中位数', color: ACCENT, legend: false, log: lg, logBase: 10 });
      chart.setOption({ grid: { left: 4, right: 8, top: 10, bottom: 22, containLabel: true },
        yAxis: lg ? { min: nb.min, max: nb.max } : { min: nb.min, max: nb.max, interval: nb.step, splitNumber: undefined },
        xAxis: { axisLabel: { interval: Math.max(1, Math.floor(x.length / 3)), fontSize: 10 } } });
    }));
  }
  const drawScoped = () => { if (!res) return; paintScope(); drawFan(); drawTable(); drawDps(); drawCorr(); paintMultiOn(); };
  const paintMultiOn = () => { [...multiGrid.children].forEach((c, i) => c.classList.toggle('on', i === 0 ? S.scope === 'weighted' : per()[i - 1]?.id === S.scope)); };

  // ---------------------------------------------------------------- D: table
  function drawTable() {
    const t = S.ticker;
    const lev = LEVERAGED.includes(t);
    tableTitle.textContent = `${t === GOLD ? '金价' : t} 情景对比（${per().length} 个情景 + 加权汇总）`;
    const all = [{ id: null, name: '加权汇总（按情景概率）', prob: 1, ...W() },
      ...per().slice().sort((a, b) => (b.prob || 0) - (a.prob || 0))];
    const phone = (tableBox.clientWidth || window.innerWidth) < 560;
    const rows = phone && !S.allRows ? all.filter(r => r.id === null || r.id === S.scope) : all;
    const tri = (r) => h('span', { class: 'tk-tri' }, signed(r.cagr10_p10), ' · ', h('b', {}, signed(r.cagr10_p50)), ' · ', signed(r.cagr10_p90));
    const pair = (a, b, f) => h('span', { class: 'tk-pair' }, f(a), h('span', { class: 'muted' }, ' / '), f(b));
    const cols = [
      { label: '情景', value: (r) => r, wrap: true, fmt: (r) => (r.id === null ? h('b', {}, r.name) : h('span', {}, r.source === 'custom'
        ? [r.name, ' ', ui.chip('自定义', 'accent')] : h('a', { href: `#scenarios?${r.id}`, title: '查看这个情景的宏观设定与传导链' }, names[r.id] || r.name))) },
      { label: '概率', value: (r) => r.prob, fmt: (v) => pct(v) },
      { label: '10 年年化 p10 · p50 · p90', value: (r) => r, fmt: tri },
      { label: '15 年年化 名义 / 实际', value: (r) => r, fmt: (r) => pair(r.cagr15_p50, r.real_cagr15_p50, (v) => signed(v)) },
      { label: '每股价格 10 年年化', value: (r) => r.price_cagr10_p50, fmt: (v) => signed(v) },
      { label: '最大回撤中位 月 / 季末', value: (r) => r, fmt: (r) => pair(r.max_dd_p50, r.max_dd_q_p50, (v) => signed(v, 0)) },
      { label: '年化波动', value: (r) => r.vol, fmt: (v) => pct(v) },
      { label: '15 年亏损概率 名义 / 实际', value: (r) => r, fmt: (r) => pair(r.prob_loss15, r.prob_loss15_real, (v) => fmtProb(v)) },
      { label: '第一年现金收益率', value: (r) => r.income_y1_p50, fmt: (v) => pct(v) },
      { label: '10 年现金（美元，每 1 万）', value: (r) => r.cash10_per_10k_p50, fmt: (v) => (isNum(v) ? Math.round(v).toLocaleString('zh-CN') : '—') },
    ];
    if (lev) cols.push({ label: '15 年内归零', value: (r) => r.wipeout_prob, fmt: (v) => (res?.ruin === false ? '抽签已关闭' : fmtProb(v)), cls: (v) => (v > 0.01 ? 'is-bad' : '') });
    const tbl = ui.table(cols, rows, { stack: true, rowClass: (r) => (r.id === null ? 'tk-row-w' : r.id === S.scope ? 'weights-focus' : '') });
    const more = phone && all.length > rows.length
      ? h('button', { class: 'btn tk-more', onclick: () => { S.allRows = true; drawTable(); } }, `展开全部 ${all.length - 1} 个情景`)
      : phone && S.allRows ? h('button', { class: 'btn tk-more', onclick: () => { S.allRows = false; drawTable(); tableCard.scrollIntoView({ block: 'start' }); } }, '只看加权汇总与所选情景') : null;
    tableBox.replaceChildren(...[tbl, more].filter(Boolean));
  }

  // ---------------------------------------------------------------- E: payouts per share
  function drawDps() {
    if (!res) return;
    const t = S.ticker;
    const w = W();
    const payer = isNum(w.income_y1_p50) && w.income_y1_p50 > 1e-4;
    distTitle.textContent = `${tl()}的现金分配：每股路径与削减概率`;
    distSub.textContent = payer ? `分配规则：${CAT?.tickers?.[t]?.cash || '—'}。持有一股、分配不再投资；削减 = 前 10 年里某一年的现金比第一年低 10% / 20% 以上（${scopeName()}）。` : '';
    distEmpty.replaceChildren(...(payer ? [] : [ui.notice(NO_CASH[t] || '这个标的几乎不分配现金。', 'info')]));
    distKpis.style.display = payer ? '' : 'none';
    distCharts.style.display = payer ? '' : 'none';
    distNote.replaceChildren();
    if (!payer) { charts.unmount(dpsBox); charts.unmount(cutBox); return; }
    const src = S.scope === 'weighted' ? w : perOf(S.scope) || w;
    const st = startInfo();
    const dps = fansOf(S.scope)?.dps;
    const yoc10 = isNum(dps?.p50?.[9]) && isNum(st.price) && st.price > 0 ? dps.p50[9] / st.price : null;
    distKpis.replaceChildren(
      ui.kpi('第一年现金收益率（中位数）', pct(src.income_y1_p50, 1), isNum(dps?.p50?.[0]) ? `每股约 ${fmtUsd(dps.p50[0])}` : ''),
      ui.kpi('10 年内削减 ≥ 20% 的概率', fmtProb(src.cut20_prob), `削减 ≥ 10%：${fmtProb(src.cut10_prob)}`, 'tk-prob'),
      ui.kpi('第 10 年分配 ÷ 第 1 年（中位数）', isNum(src.dps10_over_1_p50) ? `${src.dps10_over_1_p50.toFixed(2)} 倍` : '—',
        isNum(yoc10) ? `第 10 年成本收益率约 ${pct(yoc10, 1)}（按起点价格）` : ''),
      ui.kpi('10 年累计现金（每 1 万美元）', isNum(src.cash10_per_10k_p50) ? `${Math.round(src.cash10_per_10k_p50).toLocaleString('zh-CN')} 美元` : '—',
        `p10–p90：${isNum(src.cash10_per_10k_p10) ? Math.round(src.cash10_per_10k_p10).toLocaleString('zh-CN') : '—'} ~ ${isNum(src.cash10_per_10k_p90) ? Math.round(src.cash10_per_10k_p90).toLocaleString('zh-CN') : '—'}；每股价格 10 年年化 ${pct(src.price_cagr10_p50, 1)}`));
    const notes = [];
    if (t === 'QQQI') notes.push(h('div', { class: 'portfolio-callout bad' }, h('b', {}, '资本返还：'), `${CAT?.tickers?.QQQI?.tax_note || QQQI_ROC}。`,
      `这里每股净值 10 年年化 ${pct(src.price_cagr10_p50, 1)}：分配里有一部分是本金，10 年后的每股分配中位数只有第一年的 ${isNum(src.dps10_over_1_p50) ? pct(src.dps10_over_1_p50, 0) : '—'}。`));
    if (BDC.includes(t)) notes.push(h('div', { class: 'portfolio-callout' }, h('b', {}, 'BDC：'), BDC_NOTE));
    if (MLP.includes(t)) notes.push(h('div', { class: 'portfolio-callout' }, h('b', {}, 'MLP：'), MLP_NOTE));
    if (t === 'TLT') notes.push(h('div', { class: 'portfolio-callout' }, h('b', {}, 'TLT：'), '现金是票息收益率（当期收益率约为到期收益率的 0.857 倍，按 7 年半衰期回到 1）；收益率下行时分配随之下降，这是削减概率的来源。'));
    distNote.replaceChildren(...notes);
    if (dps) {
      const n = dps.p50.length;
      const x = (res.dates || []).slice(1, n + 1);
      const y1 = dps.p50[0];
      charts.fan(dpsBox, { x: x.length === n ? x : dps.p50.map((_, i) => `第 ${i + 1} 年`), bands: dps, fmt: (v) => fmtUsd(v), axisFmt: usdAxis, name: '中位数', color: '#1098ad',
        extra: isNum(y1) ? [{ name: '第一年中位数 × 0.8', data: dps.p50.map(() => y1 * 0.8), dashed: true, color: '#adb5bd' }] : [] });
    }
    const rows = [{ id: null, label: '加权汇总', e: w }, ...rankOrder().map(e => ({ id: e.id, label: labelOf(e), e }))];
    cutBox.style.height = `${rowsHeight(rows.length, 300)}px`;
    const chart = charts.mount(cutBox);
    chart.setOption({
      legend: { top: 0, data: ['削减 ≥ 20%', '削减 ≥ 10%'], textStyle: { color: INK, fontSize: 12 }, itemWidth: 14, itemHeight: 8 },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true,
        formatter: (ps) => [`<b>${esc(rows[ps[0]?.dataIndex]?.label)}</b>`, ...ps.map(p => `${p.marker}${esc(p.seriesName)}：${fmtProb(p.value)}`)].join('<br/>') },
      grid: { left: 8, right: 26, top: 30, bottom: 8, containLabel: true },
      xAxis: { type: 'value', min: 0, max: 1, axisLine: { lineStyle: { color: LINE } }, splitLine: { lineStyle: { color: '#eef0f4' } },
        axisLabel: { color: INK3, fontSize: 11, formatter: (v) => `${Math.round(v * 100)}%` } },
      yAxis: { type: 'category', inverse: true, data: rows.map(r => r.label), axisTick: { show: false }, axisLine: { lineStyle: { color: LINE } },
        axisLabel: { color: INK2, fontSize: 11, interval: 0, width: labelWidth(cutBox, 0.32), overflow: 'truncate',
          formatter: (val, i) => (i === 0 ? `{b|${val}}` : val), rich: { b: { fontWeight: 600, color: INK } } } },
      series: [
        { name: '削减 ≥ 20%', type: 'bar', data: rows.map(r => (isNum(r.e.cut20_prob) ? r.e.cut20_prob : null)), barMaxWidth: 10, itemStyle: { color: ACCENT, borderRadius: 2 } },
        { name: '削减 ≥ 10%', type: 'bar', data: rows.map(r => (isNum(r.e.cut10_prob) ? r.e.cut10_prob : null)), barMaxWidth: 10, itemStyle: { color: '#91a7ff', borderRadius: 2 } },
      ],
    });
  }

  // ---------------------------------------------------------------- F: attribution
  function drawAttr() {
    if (!res) return;
    const pn = res.part_names || {};
    const parts = Object.keys(pn);
    const t = S.ticker === GOLD ? '金价' : S.ticker;
    attrTitle.textContent = `${t} 收益来源分解（年化对数回报，${S.attr === 'mean' ? '路径平均' : '中位路径'}）`;
    const rows = S.attr === 'mean'
      ? [{ label: '加权汇总', a: W().attribution?.mean || {} }, ...per().map(e => ({ label: labelOf(e), a: e.attribution?.mean || {} }))]
      : per().map(e => ({ label: labelOf(e), a: e.attribution?.median_path || {} }));
    const sums = rows.map(r => { const xs = parts.map(p => r.a[p]).filter(isNum); return xs.length ? xs.reduce((s, x) => s + x, 0) : null; });
    attrBox.style.height = `${rowsHeight(rows.length, 360) + 20}px`;
    const chart = charts.mount(attrBox);
    chart.setOption({
      color: PART_COLORS,
      legend: { top: 0, type: 'scroll', data: [...parts.map(p => pn[p]), '合计'], textStyle: { color: INK, fontSize: 12 }, itemWidth: 12, itemHeight: 8 },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true,
        formatter: (ps) => [`<b>${esc(rows[ps[0]?.dataIndex]?.label)}</b>`,
          ...ps.filter(p => isNum(p.value)).map(p => `${p.marker}${esc(p.seriesName)}：${ui.fmtPct(p.value, 2, true)}/年`)].join('<br/>') },
      grid: { left: 8, right: 30, top: 34, bottom: 8, containLabel: true },
      xAxis: { type: 'value', splitNumber: (attrBox.clientWidth || 600) < 480 ? 3 : 5, axisLine: { lineStyle: { color: LINE } }, splitLine: { lineStyle: { color: '#eef0f4' } },
        axisLabel: { color: INK3, fontSize: 11, hideOverlap: true, formatter: (v) => `${+(v * 100).toFixed(0)}%` } },
      yAxis: { type: 'category', inverse: true, data: rows.map(r => r.label), axisTick: { show: false }, axisLine: { lineStyle: { color: LINE } },
        axisLabel: { color: INK2, fontSize: 11, interval: 0, width: labelWidth(attrBox, 0.3), overflow: 'truncate' } },
      series: [...parts.map((p, i) => ({ name: pn[p], type: 'bar', stack: 'parts', barMaxWidth: 16, data: rows.map(r => (isNum(r.a[p]) ? r.a[p] : null)),
        itemStyle: { color: PART_COLORS[i % PART_COLORS.length] }, emphasis: { focus: 'series' } })),
      { name: '合计', type: 'scatter', symbol: 'diamond', symbolSize: 10, z: 5, data: sums, itemStyle: { color: INK, borderColor: '#fff', borderWidth: 1 } }],
    });
  }

  // ---------------------------------------------------------------- G: correlations
  function drawCorr() {
    if (!res) return;
    const c = res.corr || {};
    const t = S.ticker;
    const assets = (k) => ASSET_IDS.includes(k);
    const keys = Object.keys(c.with || {}).filter(k => (S.corr === 'assets' ? assets(k) : !assets(k)));
    const valOf = (k) => (S.scope === 'weighted' ? c.with[k]?.weighted : c.with[k]?.per_scenario?.[S.scope]);
    const rows = keys.map(k => ({ k, v: valOf(k), label: assets(k) ? assetLabel(k) : `${k} ${shortOf(k)}` }))
      .filter(r => isNum(r.v)).sort((a, b) => b.v - a.v);
    corrTitle.textContent = `${tl()}与其他${S.corr === 'assets' ? '资产类别' : '标的'}的相关：${scopeName()}`;
    const us = valOf('us_equity');
    const hist = c.hist_us_equity;
    const head = [h('span', {}, '与美股大盘的季度相关：模拟 ', h('b', {}, isNum(us) ? us.toFixed(2) : '—'),
      isNum(hist) ? ['，历史 ', h('b', {}, hist.toFixed(2)), '（2013–2026 季度）'] : '（历史对照只对估计型标的计算）')];
    if (isNum(us) && isNum(hist) && hist - us > 0.15) head.push(' ', ui.chip(chipText('credit_link') || '与股市的联动低于历史', 'bad'));
    const dg = CAT?.diagnostics?.[S.scope === 'weighted' ? 'baseline' : S.scope];
    if (dg?.corr_dspread_us_equity) {
      head.push(h('div', { class: 'muted tk-diag' }, `核心一致性诊断（${S.scope === 'weighted' ? '基准情景' : scopeName()}）：信用利差变化与美股的季度相关 模拟 ${dg.corr_dspread_us_equity.sim?.toFixed(2)}、历史 ${dg.corr_dspread_us_equity.hist?.toFixed(2)}`
        + (dg.corr_dgz_us_equity ? `；GZ 利差变化 模拟 ${dg.corr_dgz_us_equity.sim?.toFixed(2)}、历史 ${dg.corr_dgz_us_equity.hist?.toFixed(2)}` : '')
        + '。差距来自核心（月度引擎的信用利差与股市联动太弱），应由季度模型修正。'));
    }
    corrHead.replaceChildren(...head);
    corrBox.style.height = `${rowsHeight(rows.length, 300)}px`;
    charts.mount(corrBox).setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true,
        formatter: (ps) => `<b>${esc(rows[ps[0]?.dataIndex]?.label)}</b><br/>相关系数 ${isNum(ps[0]?.value) ? ps[0].value.toFixed(2) : '—'}` },
      grid: { left: 8, right: 30, top: isNum(hist) && S.corr === 'assets' ? 26 : 8, bottom: 8, containLabel: true },
      xAxis: { type: 'value', min: -1, max: 1, axisLine: { lineStyle: { color: LINE } }, splitLine: { lineStyle: { color: '#eef0f4' } },
        axisLabel: { color: INK3, fontSize: 11, formatter: (v) => (+v).toFixed(1) } },
      yAxis: { type: 'category', inverse: true, data: rows.map(r => r.label), axisTick: { show: false }, axisLine: { lineStyle: { color: LINE } },
        axisLabel: { color: INK2, fontSize: 11, interval: 0, width: labelWidth(corrBox, 0.34), overflow: 'truncate' } },
      series: [{ type: 'bar', data: rows.map(r => ({ value: r.v, itemStyle: { color: r.k === 'us_equity' ? ACCENT : '#91a7ff', borderRadius: 2 } })), barMaxWidth: 12,
        label: { show: true, position: 'right', fontSize: 10, color: INK2, formatter: (p) => (isNum(p.value) ? p.value.toFixed(2) : '') },
        markLine: isNum(hist) && S.corr === 'assets' ? { silent: true, symbol: 'none', precision: 10, lineStyle: { color: INK3, type: 'dashed' },
          label: { formatter: `历史（与美股）${hist.toFixed(2)}`, color: INK3, fontSize: 10, position: 'start' }, data: [{ xAxis: hist }] } : undefined }],
    });
  }

  // ---------------------------------------------------------------- H: notes, the overview heatmap
  function paintNotes() {
    notesList.replaceChildren(...(res.notes || []).map(n => h('li', { class: 'muted' }, n)),
      h('li', { class: 'muted' }, `相关系数：${res.corr?.method || '—'}。`),
      h('li', { class: 'muted' }, `构建于 ${res.built_at || '—'}；数据截至：${vintageText(res.vintages) || '—'}。`));
    const m = res.recipe_meta || {};
    const lines = [];
    if (isNum(m.alpha_q)) lines.push(`α：每季 ${pct(m.alpha_q, 2)}（年化约 ${pct(m.alpha_q * 4, 1)}）`);
    if (isNum(m.c_lr)) lines.push(`分配的长期增速（年化）${pct(m.c_lr, 1)}：由本次运行的核心路径（该情景的系统部分均值）加上每条路径自己的 α 求出，让收益率在几十年里保持平稳（各情景、各路径的平均）`);
    if (isNum(m.cut_multiple)) lines.push(`分配削减规则：收益率（按 4 个季末均价）超过正常水平的 ${m.cut_multiple} 倍时削减到正常收益率（假设）；`
      + `15 年内触发过的路径约 ${pct(m.cut_share_paths, 0)}（各情景平均）`);
    if (m.residual_mode === 'fixed') lines.push('残差：在实际被抽的池上去均值，再按模拟波动做条件重定心（按构造零均值）');
    if (isNum(m.resid_bucket_mean_max_ann)) {
      lines.push(`实际抽到的残差：均值每年 ${pct(m.resid_mean_ann, 2)}；按模拟波动分 ${m.resid_buckets || 5} 档，档内均值偏离 0 最多每年 ${pct(m.resid_bucket_mean_max_ann, 2)}`
        + `（一档的抽样误差约 ±${pct(m.resid_bucket_se_ann, 2)}；各情景平均）`);
    }
    if (typeof m.gz === 'string') lines.push(m.gz.includes('spread_eq') ? 'GZ 信用利差：月度引擎没有它，用“Baa−10 年利差变化 + 美股季度回报”桥接（不加噪声）' : `GZ 信用利差：${m.gz.startsWith('bridge') ? '桥接' : '核心直接给出'}`);
    if (isNum(m.s_eff)) lines.push(`互换利差 ${pct(m.s_eff, 2)}/年`);
    if (isNum(m.L)) lines.push(`杠杆倍数 ${m.L}`);
    if (isNum(m.coverage)) lines.push(`期权覆盖率 ${pct(m.coverage, 1)}`);
    if (isNum(m.dist_rate)) lines.push(`分配率 NAV 的 ${pct(m.dist_rate, 1)}/年`);
    if (isNum(m.fee)) lines.push(`费率 ${pct(m.fee, 2)}/年`);
    metaList.replaceChildren(...(lines.length ? lines : ['（这个配方没有额外的运行信息）']).map(x => h('li', { class: 'muted' }, x)));
  }
  function drawOverview() {
    if (!CAT || CAT.error) { overDetails.style.display = 'none'; return; }
    const rows = storedTickers().filter(t => CAT.tickers?.[t]?.stats && Object.keys(CAT.tickers[t].stats).length);
    const cols = rows.length ? Object.keys(CAT.tickers[rows[0]].stats) : [];
    if (!rows.length || !cols.length || !cols.every(id => names[id])) { overDetails.style.display = 'none'; return; }
    overDetails.style.display = '';
    if (!overDetails.open) return;
    const values = rows.map(t => cols.map(id => CAT.tickers[t].stats[id]?.cagr10_p50 ?? null));
    overBox.style.height = `${Math.max(380, 70 + rows.length * 24)}px`;
    const chart = charts.heatmap(overBox, { rows: rows.map(t => `${t} ${shortOf(t)}`), cols: cols.map(id => shortName(names[id])), values,
      min: -0.15, max: 0.15, colors: charts.klineDiverging(), fmt: 'pct', rowLabelWidth: 150,
      tip: (r, c, v) => `<b>${esc(tickerName(rows[r]))}</b><br/>${esc(names[cols[c]])}<br/>10 年年化中位数 <b>${v === null ? '—' : esc(ui.fmtPct(v, 1))}</b>`
        + `<br/>p10–p90：${esc(ui.fmtPct(CAT.tickers[rows[r]].stats[cols[c]]?.cagr10_p10, 1))} ~ ${esc(ui.fmtPct(CAT.tickers[rows[r]].stats[cols[c]]?.cagr10_p90, 1))}`
        + `<br/>第一年现金收益率 ${esc(ui.fmtPct(CAT.tickers[rows[r]].stats[cols[c]]?.income_y1_p50, 1))}` });
    chart.off('click');
    chart.on('click', (p) => { const t = rows[p.value?.[1]]; if (t) { select(t); recipeCard.scrollIntoView({ behavior: 'smooth', block: 'start' }); } });
  }
  overDetails.addEventListener('toggle', () => { if (overDetails.open) drawOverview(); });

  function show(r) {
    res = r;
    if (!Object.keys(names).length) { names = Object.fromEntries((r.per_scenario || []).map(e => [e.id, e.name])); drawOverview(); }
    if (builtFor !== S.ticker) buildRecipe();
    paintStart();
    paintRecipeChips();
    paintFlags();
    vintLine.textContent = `数据截至：${vintageText(r.vintages) || '—'}${r.built_at ? `；标的路径构建于 ${r.built_at}` : ''}${r.cached ? '' : `；本次计算 ${r.elapsed_s} 秒`}`;
    resultBox.style.display = '';
    paintScope();
    paintCompare();
    drawFan();
    drawTable();
    drawDps();
    drawAttr();
    drawCorr();
    paintNotes();
    drawMulti();
  }

  // ---------------------------------------------------------------- lifecycle
  paintChips();
  buildRecipe();
  if (focus) setTimeout(start, 0);                 // once the page has attached the section (alive() needs it)
  else if (typeof IntersectionObserver === 'function') {
    const io = new IntersectionObserver((entries) => { if (entries.some(e => e.isIntersecting)) { io.disconnect(); start(); } }, { rootMargin: '300px 0px' });
    io.observe(el);
    signal?.addEventListener('abort', () => io.disconnect());
  } else start();
  return {
    el,
    // scenario id -> full name (scenarios.json), for axis labels and the overview
    setScenarios(list) { names = Object.fromEntries((list || []).map(s => [s.id, s.name || s.id])); drawOverview(); if (res) { paintScope(); drawTable(); } },
    setCatalog() { paintChips(); buildRecipe(); drawOverview(); },
    refresh,
    select,
    loaded: () => !!res,
  };
}

// ---------------------------------------------------------------- 提取页: cash flow with real distributions

const CONT_SHADE = 'rgba(122, 131, 150, 0.09)';
const DIV_WHY = '熊市里多付出的现金出自价格回报，“先用分红”再把多余的现金按比例再投资，又把它分散到低收益的资产上';
// The reading of the cash pot against this page's own numbers: whether the two dividend-first ways do have the higher
// ruin probability here (they usually do for income tickers, not always for a plan of asset classes).
function divReading(res, strategies) {
  const r = (id) => res?.strategies?.[id]?.aggregate?.ruin_prob_weighted;
  const pro = r('pro_rata');
  const div = strategies.filter(s => s.id !== 'pro_rata').map(s => ({ s, v: r(s.id) })).filter(x => isNum(x.v));
  if (!isNum(pro) || !div.length) return '';
  const nums = div.map(x => `${x.s.short} ${fmtProb(x.v)}`).join('、');
  if (div.every(x => x.v > pro + 0.001)) {
    return `现金池在第一年就可能超过提取额，但这不等于更安全：本页两种“先用分红”的提取方式破产概率（${nums}）都高于按比例提取（${fmtProb(pro)}），因为${DIV_WHY}。`;
  }
  if (div.every(x => x.v < pro - 0.001)) {
    return `本页“先用分红”的提取方式破产概率（${nums}）低于按比例提取（${fmtProb(pro)}）：这个组合的分配大多没有出自价格回报。`;
  }
  return `本页三种提取方式的破产概率相近（按比例 ${fmtProb(pro)}；${nums}）：现金池的大小没有明显改变结果。`;
}

// The 现金流 section of the withdrawal page. strategies: the page's STRATS ([{id, name, short, color}]).
// show(res, stratId) fills it from a withdrawal response (res.cash_flow); drawCover(res, stratId) redraws the
// coverage fan for another strategy.
export function cashFlowView({ strategies } = {}) {
  injectWeightsStyle();
  injectStyles();
  const { h, card } = ui;
  const S = strategies || [];
  const kpis = h('div', { class: 'grid cols-4' });
  const reading = h('div', { class: 'portfolio-callout' });
  const potBox = ui.chartBox('tk-pot');
  const potNote = h('p', { class: 'tk-note' });
  const coverBox = ui.chartBox('tall');
  const cutBox = ui.chartBox('tk-cutp');
  const notes = h('ul', { class: 'list' });
  const titled = (title, sub, ...body) => {
    const c = card({ title, sub }, ...body);
    return [c, c.querySelector('.card-title'), c.querySelector('.card-sub')];
  };
  const [potCard, potTitle] = titled('第一年现金池构成（加权汇总）', '每个持仓第一年收到的现金（股息、利息、分配）÷ 第一年的提取额，取全部情景路径按概率加权的中位数：加起来就是第一年现金池能付提取额的几倍。', potBox, potNote);
  const [coverCard, coverTitle, coverSub] = titled('现金流覆盖：每年现金池 ÷ 当年提取额', '（计算后填入）', coverBox);
  const cutEmpty = h('div');
  const [cutCard, cutTitle] = titled('分配削减压力', '持有一份、分配不再投资时，前 10 年里某一年的现金比第一年低 10% / 20% 以上的概率（按情景概率加权）。资产类别按引擎口径（今天的股息率 × 当时的市值，随价格同比例变化），标的按各自的分配模型。', cutEmpty, cutBox);
  const notesCard = card({ title: '分配的口径', sub: '现金池里的钱从哪里来：标的的分配模型与资产类别的引擎口径。' }, notes);
  const empty = h('div');
  const el = h('div', { class: 'tk-cash' },
    h('div', { class: 'section-title' }, '现金流：真实分配能付多少提取额（按情景概率加权汇总）'),
    h('p', { class: 'muted tk-note tk-cash-scope' }, '这一节始终是全部情景按概率混合的结果，不跟随上面的“图表范围”。'), empty, kpis, reading,
    h('div', { class: 'weights-pair' }, potCard, coverCard),
    h('div', { class: 'weights-pair' }, cutCard, notesCard));

  function show(res, stratId) {
    const cf = res?.cash_flow;
    el.style.display = cf ? '' : 'none';
    if (!cf) return;
    const pot = cf.pot_share_y1;
    const tickers = (res.instruments?.tickers || []).filter(isTicker);
    const zero = !pot || !isNum(pot.total);
    empty.replaceChildren(...(zero ? [ui.notice('提取率为 0：没有提取额，现金池与覆盖无从计算；下面只列分配削减压力。', 'info')] : []));
    kpis.style.display = zero ? 'none' : '';
    reading.style.display = zero ? 'none' : '';
    potCard.style.display = zero ? 'none' : '';
    coverCard.style.display = zero ? 'none' : '';
    if (!zero) {
      kpis.replaceChildren(
        ui.kpi('第一年现金 ÷ 第一年提取额（中位数）', `${pot.total.toFixed(2)} 倍`,
          pot.total >= 1 ? '第一年的现金就够付提取额：“先用分红”的策略第一年不用卖出持仓' : '第一年的现金不够付提取额：不足的部分要卖出持仓'),
        ...S.filter(s => cf.strategies?.[s.id]).map(s => ui.kpi(`某年现金池不足提取额一半的概率（加权）· ${s.short}`, fmtProb(cf.strategies[s.id].prob_pot_below_half),
          '整个期限里至少有一年如此；只计当年年初还没破产的路径', 'tk-prob')));
      const rd = divReading(res, S);
      reading.textContent = rd;
      reading.style.display = rd ? '' : 'none';
      drawPot(res, tickers);
      drawCover(res, stratId);
    } else {
      charts.unmount(potBox);
      charts.unmount(coverBox);
    }
    drawCut(res, cf);
    const lines = [];
    if (tickers.length) lines.push(...(CAT?.withdrawal_note || []));
    else lines.push('资产类别按引擎口径：今天的股息率乘以当时的市值，熊市里随价格同比例下降（真实分红更有粘性）。', '全部为税前。');
    lines.push('按比例提取时，现金留在原资产里再投资；覆盖图仍按它产生的现金计算，以便与两种“先用分红”的方式对照。');
    if (tickers.includes('QQQI')) lines.push(`QQQI：${CAT?.tickers?.QQQI?.tax_note || QQQI_ROC}。`);
    notes.replaceChildren(...lines.map(x => h('li', {}, x)));
  }

  function drawPot(res, tickers) {
    const cols = res.cash_flow.pot_share_y1.columns || {};
    const rows = Object.entries(cols).filter(([, x]) => isNum(x)).sort((a, b) => b[1] - a[1]);
    const total = res.cash_flow.pot_share_y1.total;
    potTitle.textContent = `第一年现金池构成：合计为提取额的 ${pct(total, 0)}`;
    potBox.style.height = `${Math.max(220, 40 + rows.length * 26)}px`;
    charts.mount(potBox).setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true,
        formatter: (ps) => { const r = rows[ps[0]?.dataIndex]; return r ? `<b>${esc(holdingName(r[0]))}</b><br/>第一年现金 = 提取额的 ${esc(pct(r[1], 1))}<br/>权重 ${esc(fmtShare(res.weights?.[r[0]]))}` : ''; } },
      grid: { left: 8, right: 44, top: 8, bottom: 8, containLabel: true },
      xAxis: { type: 'value', min: 0, axisLine: { lineStyle: { color: LINE } }, splitLine: { lineStyle: { color: '#eef0f4' } },
        axisLabel: { color: INK3, fontSize: 11, hideOverlap: true, formatter: (v) => `${Math.round(v * 100)}%` } },
      yAxis: { type: 'category', inverse: true, data: rows.map(([id]) => holdingLabel(id)), axisTick: { show: false }, axisLine: { lineStyle: { color: LINE } },
        axisLabel: { color: INK2, fontSize: 11, interval: 0, width: labelWidth(potBox, 0.3), overflow: 'truncate' } },
      series: [{ type: 'bar', barMaxWidth: 14, data: rows.map(([id, x]) => ({ value: x, itemStyle: { color: holdingColor(id), borderRadius: 2 } })),
        label: { show: true, position: 'right', fontSize: 10, color: INK2, formatter: (p) => pct(p.value, 0) } }],
    });
    const top = rows[0];
    const bdc = tickers.filter(t => BDC.includes(t));
    const share = (ids) => ids.reduce((s, id) => s + (cols[id] || 0), 0);
    const bits = [];
    if (tickers.includes('QQQI') && bdc.length) bits.push(`高分配标的（QQQI、BDC）支付了大部分现金池（合计为提取额的 ${pct(share(['QQQI', ...bdc]), 0)}）；QQQI 的分配大部分是资本返还，相当于在卖出持仓。`);
    else if (tickers.includes('QQQI')) bits.push(`QQQI 一项就是提取额的 ${pct(cols.QQQI, 0)}；它的分配大部分是资本返还，相当于在卖出持仓。`);
    else if (bdc.length) bits.push(`BDC 合计为提取额的 ${pct(share(bdc), 0)}：它们的高收益率意味着价格或分配需要下行。`);
    if (top && !bits.length) bits.push(`最大的现金来源是 ${holdingLabel(top[0])}（提取额的 ${pct(top[1], 0)}）。`);
    potNote.textContent = bits.join('');
  }

  function drawCover(res, stratId) {
    const cf = res?.cash_flow;
    const s = S.find(x => x.id === stratId) || S[0];
    const fan = cf?.strategies?.[s?.id]?.coverage_fan;
    if (!fan) { charts.unmount(coverBox); return; }
    const n = fan.p50?.length || 0;
    const x = (res.dates || []).slice(1, n + 1);
    const xs = x.length === n ? x : fan.p50.map((_, i) => `第 ${i + 1} 年`);
    const SY = res.scenario_years;
    coverTitle.textContent = `现金流覆盖：每年现金池 ÷ 当年提取额（${s.short}，加权汇总）`;
    coverSub.textContent = '每年收到的现金（股息、利息、分配）是当年提取额的几倍：在灰线 1 倍以上，“先用分红”当年不用卖出持仓；跌破 0.5 倍虚线，一半以上的提取额要靠卖出。'
      + '纵轴按对数刻度（0.5、1、2、4 倍等距）；只计当年年初还没破产的路径；灰底是没有情景主题的续抽年份。'
      + (isNum(SY) && n > SY && (res.instruments?.tickers || []).some(t => t !== 'QQQ' && !LEVERAGED.includes(t) && t !== 'BRK.B' && t !== 'GLD')
        ? `注意：续抽年份里，标的的分配状态（粘性分配的收益率、TLT 的收益率）不承接各情景第 ${SY} 年末的水平，而是来自续抽路径自己的历史，所以覆盖在第 ${SY + 1} 年会跳一下、分布变宽；这是续抽拼接的已知局限。`
        : '');
    charts.fan(coverBox, { x: xs, bands: fan, fmt: (v) => `${(+v).toFixed(2)} 倍`, axisFmt: (v) => `${+(+v).toFixed(2)}`, name: '中位数', color: s.color || ACCENT,
      log: true, logBase: 2, markAreas: isNum(SY) && n > SY ? [{ from: xs[SY], to: xs[n - 1], label: '无条件续抽', color: CONT_SHADE }] : undefined,
      extra: [{ name: '= 提取额', data: xs.map(() => 1), dashed: true, color: '#868e96' },
        { name: '提取额的一半', data: xs.map(() => 0.5), dashed: true, color: '#d9480f' }] });
  }

  function drawCut(res, cf) {
    const cp = cf.cut_pressure || {};
    const rows = Object.entries(cp).map(([id, x]) => ({ id, w: res.weights?.[id], c10: x?.cut10_prob, c20: x?.cut20_prob }))
      .filter(r => isNum(r.c10) || isNum(r.c20))
      .sort((a, b) => (isNum(b.c20) ? b.c20 : -1) - (isNum(a.c20) ? a.c20 : -1));
    cutTitle.textContent = `分配削减压力（前 ${cf.cut_years || 10} 年）`;
    cutEmpty.replaceChildren(...(rows.length ? [] : [ui.notice('这个组合没有产生现金分配的持仓。', 'info')]));
    cutBox.style.display = rows.length ? '' : 'none';
    if (!rows.length) { charts.unmount(cutBox); return; }
    cutBox.style.height = `${Math.max(220, 56 + rows.length * 30)}px`;
    charts.mount(cutBox).setOption({
      legend: { top: 0, data: ['削减 ≥ 20%', '削减 ≥ 10%'], textStyle: { color: INK, fontSize: 12 }, itemWidth: 14, itemHeight: 8 },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true,
        formatter: (ps) => { const r = rows[ps[0]?.dataIndex]; return r ? [`<b>${esc(holdingName(r.id))}</b>（权重 ${esc(fmtShare(r.w))}）`,
          ...ps.map(q => `${q.marker}${esc(q.seriesName)}：${esc(fmtProb(q.value))}`)].join('<br/>') : ''; } },
      grid: { left: 8, right: 44, top: 28, bottom: 8, containLabel: true },
      xAxis: { type: 'value', min: 0, max: 1, axisLine: { lineStyle: { color: LINE } }, splitLine: { lineStyle: { color: '#eef0f4' } },
        axisLabel: { color: INK3, fontSize: 11, formatter: (v) => `${Math.round(v * 100)}%` } },
      yAxis: { type: 'category', inverse: true, data: rows.map(r => holdingLabel(r.id)), axisTick: { show: false }, axisLine: { lineStyle: { color: LINE } },
        axisLabel: { color: INK2, fontSize: 11, interval: 0, width: labelWidth(cutBox, 0.3), overflow: 'truncate' } },
      series: [
        { name: '削减 ≥ 20%', type: 'bar', barMaxWidth: 10, data: rows.map(r => (isNum(r.c20) ? r.c20 : null)), itemStyle: { color: ACCENT, borderRadius: 2 },
          label: { show: true, position: 'right', fontSize: 10, color: INK2, formatter: (q) => fmtProb(q.value) } },
        { name: '削减 ≥ 10%', type: 'bar', barMaxWidth: 10, data: rows.map(r => (isNum(r.c10) ? r.c10 : null)), itemStyle: { color: '#91a7ff', borderRadius: 2 } },
      ],
    });
  }

  return { el, show, drawCover };
}

// The three α versions of the estimated tickers on one withdrawal plan, side by side (INSTRUMENTS_DESIGN §7.4):
// the page fills each version as its request returns. strategies: the page's STRATS.
export function alphaCompare({ strategies } = {}) {
  injectStyles();
  const { h, card } = ui;
  const S = strategies || [];
  const st = { key: null, res: {}, busy: new Set(), err: null, current: 'shrunk' };
  const body = h('div');
  const el = card({ title: 'α 的影响：估计型标的的三个版本', cls: 'tk-alpha',
    sub: `${CAT?.alpha_note || 'α 是历史上相对风格组合的超额回报。'}三个版本只差 α 这一项：系统部分、残差抽样和随机数完全相同，所以行与行的差别就是 α 本身。` }, body);
  function paint() {
    const rows = ALPHA_OPTIONS.map(o => ({ o, r: st.res[o.value] }));
    const cell = (r, f) => (r ? f(r) : st.busy.size ? h('span', { class: 'muted' }, '计算中…') : '—');
    const cols = [{ label: '版本', value: (x) => x, fmt: (x) => h('span', {}, x.o.label, x.o.value === st.current ? [' ', ui.chip('当前', 'accent')] : null) }];
    for (const s of S) {
      cols.push({ label: `破产概率 · ${s.short}`, value: (x) => x, fmt: (x) => cell(x.r, (r) => {
        const v = r.strategies?.[s.id]?.aggregate?.ruin_prob_weighted;
        return h('span', { class: !isNum(v) ? '' : v > 0.05 ? 'is-bad' : v > 0.01 ? 'withdrawal-warn' : 'is-good' }, fmtProb(v));
      }) });
    }
    for (const s of S) {
      cols.push({ label: `安全提取率 · ${s.short}`, value: (x) => x, fmt: (x) => cell(x.r, (r) => {
        const v = r.safe_rate?.[s.id]?.weighted;
        return isNum(v) ? (v >= 0.2 - 1e-9 ? '≥20%' : `${(v * 100).toFixed(2)}%`) : '—';
      }) });
    }
    body.replaceChildren(...[ui.table(cols, rows, { stack: true, rowClass: (x) => (x.o.value === st.current ? 'weights-focus' : '') }),
      st.err ? ui.notice(`有一个版本没有算出来：${errorText(st.err)}`, 'error') : null,
      h('p', { class: 'tk-note muted' }, '默认用收缩后的 α：BDC 收缩到组均值（约 +6.3%/年）、MLP 约 +2.25%/年、单一公司收缩到 0 附近。减半与原始只作对照；',
        '在“提取参数”里切换后，本页的全部数字和图表都改用所选版本。')].filter(Boolean));
  }
  return {
    el,
    // key: the request without its α (a different plan clears the table); current: the α the page shows
    reset(key, current) { if (key !== st.key) { st.key = key; st.res = {}; st.err = null; st.busy.clear(); } st.current = current; paint(); },
    put(key, mode, res) { if (key !== st.key) return; st.res[mode] = res; st.busy.delete(mode); paint(); },
    busy(key, mode, on) { if (key !== st.key) return; if (on) st.busy.add(mode); else st.busy.delete(mode); paint(); },
    fail(key, mode, err) { if (key !== st.key) return; st.busy.delete(mode); st.err = err; paint(); },
    has: (key, mode) => key === st.key && !!st.res[mode],
    key: () => st.key,
  };
}

// ---------------------------------------------------------------- styles

function injectStyles() {
  if (document.getElementById('style-tickers')) return;
  document.head.append(ui.h('style', { id: 'style-tickers' }, CSS));
}
const CSS = `
.tk-toolbar { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin: 0 0 10px; }
.tk-toolbar .btn, .tk-proj .card-actions .btn, .tk-mini-btn { font-size: 12px; padding: 3px 10px; }
.tk-toolbar select { font-size: 12px; padding: 3px 8px; max-width: 220px; }
.tk-mine, .tk-save-row { display: inline-flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.tk-save-row input[type=text] { width: 150px; font-size: 12px; padding: 3px 8px; }
.tk-mine-msg { font-size: 12px; }
.tk-import { display: grid; gap: 8px; margin: 0 0 12px; padding: 10px 12px; border: 1px dashed var(--line); border-radius: 8px; background: var(--panel-2); }
.tk-import textarea { min-height: 96px; font-family: var(--mono); font-size: 12px; }
.tk-import-msg { display: flex; flex-wrap: wrap; gap: 6px; }
.tk-help { font-size: 12px; color: var(--ink-3); }
.tk-preset-note { font-size: 12px; color: var(--ink-2); margin: -4px 0 10px; padding: 4px 10px; border-left: 3px solid var(--warn); background: #fff8eb; border-radius: 0 6px 6px 0; }
.tk-amount-line { font-size: 12px; color: var(--ink-3); margin: -4px 0 8px; }
.tk-amount-line:empty { display: none; }
.tk-section { border-top: 1px solid var(--line); padding-top: 8px; margin-top: 10px; }
.tk-groups.tk-locked { opacity: 0.55; }
.tk-lock .notice { margin: 4px 0 8px; }
.tk-section > summary { cursor: pointer; display: flex; gap: 6px 10px; align-items: center; flex-wrap: wrap; list-style: none; padding: 2px 0 4px; }
.tk-section > summary::-webkit-details-marker { display: none; }
.tk-section > summary::before { content: '▸'; color: var(--ink-3); display: inline-block; transition: transform .15s; }
.tk-section[open] > summary::before { transform: rotate(90deg); }
.tk-sec-title { font-size: 13px; font-weight: 650; color: var(--ink); }
.tk-sec-sum { font-size: 12px; color: var(--ink-3); }
.tk-section > summary .tk-mini-btn { margin-left: auto; }
.tk-group { margin-top: 8px; }
.tk-group-head { display: flex; gap: 4px 10px; align-items: baseline; flex-wrap: wrap; font-size: 12px; margin: 2px 0 2px; }
.tk-group-head b { color: var(--ink); font-size: 12px; font-weight: 650; }
.tk-grid { grid-template-columns: repeat(auto-fill, minmax(min(290px, 100%), 1fr)); }
.weights-row.tk-row { grid-template-columns: 10px minmax(92px, 1.7fr) minmax(36px, 1fr) 72px 40px; }
.tk-name { display: flex; gap: 6px; align-items: baseline; min-width: 0; }
.tk-name b { font-weight: 600; flex-shrink: 0; }
.tk-sub { color: var(--ink-3); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
.weights-row.zero .tk-name b { color: var(--ink-3); font-weight: 500; }
.tk-unit { min-width: 12px; }
.tk-amount .weights-row.tk-row { grid-template-columns: 10px minmax(84px, 1fr) minmax(112px, 150px) 40px; }
.tk-amount .weights-row.tk-row .weights-in { display: flex; align-items: center; gap: 4px; min-width: 0; }
.tk-unit { white-space: nowrap; }
.tk-amount .weights-row.tk-row input[type=range] { display: none; }
.tk-amount .weights-row.tk-row .weights-in input[type=number] { width: 100%; min-width: 0; }
.tk-status:empty { display: none; }
.withdrawal-amount { display: flex; gap: 6px 10px; align-items: center; flex-wrap: wrap; font-size: 12px; }
.withdrawal-amount:empty { display: none; }
.withdrawal-amount .btn { font-size: 12px; padding: 2px 10px; }
.tk-status { margin: 4px 0 8px; }
@media (max-width: 480px) {
  .weights-row.tk-row { grid-template-columns: 10px minmax(0, 1.3fr) minmax(36px, 1fr) 62px 38px; gap: 6px; }
  .weights-row.tk-row .weights-in input[type=number] { width: 48px; }
  .tk-amount .weights-row.tk-row { grid-template-columns: 10px minmax(0, 1fr) minmax(104px, 128px) 38px; }
  .tk-amount .weights-row.tk-row .weights-in input[type=number] { width: auto; flex: 1 1 0; min-width: 0; }
}
.tk-caveats { list-style: none; margin: 0 0 12px; padding: 0; display: grid; gap: 8px; font-size: 13px; }
.tk-caveats li { line-height: 1.6; }
.tk-li-extra { color: var(--ink-2); }
.tk-holdings td:first-child { min-width: 150px; }
.tk-proj { scroll-margin-top: 84px; }
.tk-lead { margin-bottom: 12px; }
.tk-picker { display: grid; gap: 6px; margin-bottom: 8px; }
.tk-pick-group { display: flex; gap: 6px 8px; align-items: center; min-width: 0; }
.tk-pick-label { font-size: 12px; color: var(--ink-3); flex: 0 0 64px; word-break: keep-all; line-height: 1.3; }
.tk-pick-row { display: flex; flex-wrap: wrap; gap: 6px; min-width: 0; }
.tk-pick { font: inherit; font-size: 12px; border: 1px solid var(--line); background: var(--panel); color: var(--ink-2); border-radius: 999px; padding: 3px 10px; cursor: pointer; white-space: nowrap; }
.tk-pick:hover { border-color: #c6cbd6; }
.tk-pick.on { background: var(--accent-soft); border-color: #cdd7ff; color: var(--accent); font-weight: 650; }
@media (max-width: 760px) { .tk-pick-row { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: thin; padding-bottom: 2px; } }
.tk-busy { font-size: 12px; min-height: 18px; display: flex; gap: 6px; align-items: center; }
.tk-busy .spinner { width: 14px; height: 14px; }
.tk-vint { font-size: 12px; }
.tk-over-details, .tk-multi-details { margin-top: 10px; }
.tk-over-details > summary, .tk-multi-details > summary, .tk-fold > summary, .tk-notes details > summary { cursor: pointer; font-size: 12px; color: var(--ink-2); font-weight: 600; }
.tk-multi-details > summary { font-size: 13px; }
.chart.tk-over { height: 420px; }
.tk-chips { display: flex; flex-wrap: wrap; gap: 6px; max-width: 100%; }
.tk-recipe { display: grid; gap: 8px; font-size: 13px; min-width: 0; }
.tk-formula { margin: 0; background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; line-height: 1.65; overflow-wrap: anywhere; }
.tk-kv { font-size: 12px; }
.tk-kv dd { overflow-wrap: anywhere; }
.tk-params { grid-template-columns: repeat(auto-fill, minmax(min(250px, 100%), 1fr)); }
.tk-params select { max-width: 100%; }
.tk-top { gap: 16px; }
.tk-multi-card { padding-top: 12px; padding-bottom: 12px; }
.tk-multi-details { margin-top: 0; }
.tk-params input[type=number] { width: 96px; text-align: right; font-variant-numeric: tabular-nums; }
.tk-check { display: inline-flex; gap: 6px; align-items: center; font-size: 12px; color: var(--ink-2); cursor: pointer; }
.tk-reset .btn { font-size: 12px; padding: 3px 10px; }
.tk-fold { font-size: 12px; } .tk-fold ul { margin-top: 6px; }
.tk-foot { font-size: 12px; margin: 4px 0 0; }
.tk-results { display: grid; gap: 16px; margin-top: 16px; }
.tk-results.weights-busy { opacity: .55; }
.chart.tk-fan { height: 380px; }
.tk-proj .card-actions { min-width: 0; }
.tk-proj .card-actions select { max-width: min(100%, 300px); min-width: 0; text-overflow: ellipsis; }
.tk-compare { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 8px; font-size: 12px; }
.tk-cmp { font: inherit; font-size: 12px; border: 1px solid var(--line); background: var(--panel); color: var(--ink-3); border-radius: 999px; padding: 1px 9px; cursor: pointer; }
.tk-cmp.on { font-weight: 600; background: var(--panel-2); }
.tk-multi { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(min(260px, 100%), 1fr)); margin-top: 10px; }
.tk-mini { border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px 0; background: var(--panel); cursor: pointer; min-width: 0; }
.tk-mini.on { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
.tk-mini-head { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; }
.tk-mini-head b { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; font-weight: 600; }
.tk-mini-head span { white-space: nowrap; color: var(--ink-3); }
.chart.tk-mini-chart { height: 170px; }
.tk-table td:first-child { min-width: 170px; }
.tk-tri, .tk-pair { white-space: nowrap; font-variant-numeric: tabular-nums; }
.tk-more { font-size: 12px; padding: 3px 10px; margin-top: 8px; }
tr.tk-row-w td { background: var(--panel-2); font-weight: 600; }
.tk-kpis { margin-bottom: 10px; }
.kpi-value.tk-prob { color: var(--accent); }
.tk-dist-note { display: grid; gap: 8px; margin-bottom: 10px; }
.tk-dist-note:empty { display: none; }
.tk-dist-note .portfolio-callout { margin-top: 0; }
.chart.tk-dps { height: 300px; }
.tk-corr-head { font-size: 13px; margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.tk-diag { flex-basis: 100%; font-size: 12px; }
.tk-flags { display: flex; flex-wrap: wrap; gap: 6px; margin: -2px 0 8px; }
.tk-flags:empty { display: none; }
.tk-legend { font-size: 12px; margin-top: 4px; }
.tk-notes details > summary { font-size: 13px; }
@media (max-width: 480px) { .chart.tk-fan { height: 260px; } .chart.tk-mini-chart { height: 150px; } .chart.tk-dps { height: 240px; } .tk-pick-label { flex-basis: 64px; } }
.tk-input-err { font-size: 12px; color: var(--bad, #c92a2a); }
.tk-input-err:empty { display: none; }
.tk-params input.is-bad { border-color: var(--bad, #c92a2a); }
.tk-cash-scope { margin: -8px 0 0; }
.tk-note { font-size: 12px; color: var(--ink-2); margin: 8px 0 0; }
.tk-cash { display: grid; gap: 16px; }
.tk-cash > .section-title { margin-bottom: -4px; }
.tk-cash > .portfolio-callout { margin-top: -6px; }

.tk-alpha td:first-child { min-width: 130px; }
`;
