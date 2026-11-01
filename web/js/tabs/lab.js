// LLM scenario lab: narrative -> (LLM) scenario spec -> validate_spec -> engine.
// The LLM only translates a story into shocks, views, phases and a causal chain; every
// K-line and annualized return comes from the engine drawing real historical months.
// Without an API key the spec can still be loaded from a built-in, edited and run, and 载入示例 walks
// through narrative -> spec -> engine with a built-in's narrative and its hand-written spec (laid out as
// the LLM answers, llm_translator.json `examples`), labelled as not a translation.
// Deep links: #lab?<custom id> opens a saved scenario, #lab?tpl=<built-in id> loads a template.
import {
  renderScenarioDetail, disposeCharts, ensureStyle, ensureDetailStyle, phaseCard, validationCard,
  errorText, isNum, table, fmtVar, varCell, pct, fill, GLOSSARY, viewSpecText, overrideItems,
} from './_scenario_detail.js';

const DEFAULT_HORIZON = 180;    // the engine always simulates 15 years (config.HORIZON_MONTHS, meta.horizon_months)

const CSS = `
.lab-hero { border: 1px solid #cdd7ff; border-left: 4px solid var(--accent); background: linear-gradient(135deg, #f3f6ff, #fff 70%);
  border-radius: var(--radius); padding: 16px 18px; margin-bottom: 16px; }
.lab-hero h2 { margin: 0 0 12px; font-size: 17px; font-weight: 650; line-height: 1.55; color: #1f2f7a; }
.lab-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 6px; }
.lab-step { flex: 1 1 160px; background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: 12px; color: var(--ink-2); line-height: 1.5; }
.lab-step b { display: block; color: var(--ink); font-size: 13px; margin-bottom: 2px; }
.lab-step.llm { border-color: #f5d9a8; background: #fffaf0; }
.lab-step.engine { border-color: #bfe8d3; background: #f3fbf7; }
.lab-arrow { align-self: center; color: var(--ink-3); font-size: 14px; }
.lab-hero .lab-foot { margin: 10px 0 0; font-size: 12px; color: var(--ink-2); }
.lab-examples { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; align-items: center; }
.lab-examples button { font: inherit; font-size: 12px; border: 1px dashed #c6cbd6; background: var(--panel-2); border-radius: 999px; padding: 3px 10px; cursor: pointer; color: var(--ink-2); }
.lab-examples button:hover { border-color: var(--accent); color: var(--accent); }
.lab-examples.lab-demo button { border-style: solid; border-color: #bfe8d3; background: #f3fbf7; color: var(--ink); }
.lab-demo-steps { margin: 8px 0 4px; padding-left: 20px; font-size: 13px; color: var(--ink-2); line-height: 1.6; }
textarea.lab-narr { min-height: 120px; }
textarea.lab-editor { font-family: var(--mono); font-size: 12px; min-height: 460px; line-height: 1.5; tab-size: 2; }
.lab-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
.lab-bar label { display: flex; gap: 6px; align-items: center; font-size: 12px; color: var(--ink-3); }
.lab-bar input[type=number] { width: 96px; }
.lab-bar select { max-width: min(420px, 100%); }
.lab-msg { font-size: 12px; color: var(--ink-3); display: inline-flex; gap: 8px; align-items: center; }
.lab-msg.ok { color: var(--good); } .lab-msg.err { color: var(--bad); }
.lab-check { margin-top: 8px; font-size: 12px; color: var(--ink-2); }
.lab-check.err { color: var(--bad); }
.lab-check ul { margin: 4px 0 0; padding-left: 18px; color: var(--warn); }
.lab-note { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; }
.lab-status ol { margin: 6px 0 0; padding-left: 20px; }
.lab-status code, .lab-cheat code { font-family: var(--mono); font-size: 12px; background: rgba(0, 0, 0, .05); padding: 0 4px; border-radius: 4px; }
.lab-cheat { margin: 0; padding-left: 18px; font-size: 12px; color: var(--ink-2); }
.lab-cheat li { margin: 3px 0; }
.lab-result { scroll-margin-top: 72px; }
.lab-actions { display: flex; gap: 6px; justify-content: flex-end; }
.lab-actions .btn { padding: 3px 10px; font-size: 12px; }
.lab-filter { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
.lab-filter input[type=text] { width: 220px; }
.lab-new li { margin: 3px 0; }
pre.code.lab-wrap { white-space: pre-wrap; word-break: break-word; }
`;

const EXAMPLES = [
  { title: '台海冲突导致先进芯片断供 18 个月',
    text: '台海局势升级为海空封锁，台积电等先进制程芯片对全球断供 18 个月。电子产品、汽车和数据中心建设大面积停摆，AI 资本开支被迫中断；芯片和电子产品价格飙升，推高商品通胀，美联储在通胀和衰退之间左右为难。科技股估值大幅下修，工业产出下降，失业率上升，信用利差走阔。18 个月后美国、日本和欧洲的新产能陆续投产，供应链在更高的成本下恢复，之后逐步回到常态。' },
  { title: '美联储在 4% 通胀下被迫降息到零',
    text: '在政治压力和财政利息负担下，美联储在通胀仍约 4% 时连续降息，一年内把联邦基金利率降到接近零。实际利率深度为负，美元走弱，通胀预期上移；长端收益率因期限溢价上升而不降反升。黄金和实物资产走强，现金和长期国债的实际回报为负。两到三年后通胀升到 6% 以上，美联储被迫重新大幅加息，经济随之放缓。' },
  { title: 'AI 让入门级白领失业率上升 3 个百分点、利润占比继续创新高',
    text: 'AI 代理在三年内大规模替代入门级和中层白领岗位，白领失业使整体失业率上升约 3 个百分点，普通实际工资停滞，劳动收入占比继续下降；企业利润占 GDP 再创新高，股票财富进一步集中到收入前 10% 的家庭。高收入家庭的财富效应撑住一部分消费，但中低收入家庭消费走弱，服务通胀回落，美联储逐步降息。' },
  { title: '美国财政赤字失控、长端利率飙升',
    text: '美国财政赤字长期保持在 GDP 的 7% 以上，利息支出超过国防开支，国债拍卖接连遇冷。期限溢价飙升，10 年期收益率在一年内升到 7% 左右，信用利差走阔，房地产和长久期资产大幅下跌，美元先涨后跌。美联储起初维持利率不变，随后在金融市场动荡中被迫降息并重启购债，通胀维持在 4% 上下。' },
];

const BLANK = {
  id: 'custom_my_scenario',
  name: '我的情景',
  summary: '一句话概括这个情景',
  narrative: '发生了什么、为什么、通过哪些渠道传导。',
  tags: ['自定义'],
  prior_prob: 0.05,
  analogs: ['1990-08:1991-03'],
  causal_chain: [
    { from: '冲击', to: '油价', sign: '+', lag: '0-3月', mechanism: '说明这一步的传导机制' },
    { from: '油价', to: 'CPI', sign: '+', lag: '1-6月', mechanism: '能源成本进入商品价格' },
  ],
  phases: [
    { name: '冲击期', months: 12, confidence: 0.9, overrides: {}, views: [
      { var: 'f_oil', mean_annual: 0.5, why: '油价按年率约上涨 65%（mean_annual 是年化对数变化：e^0.5 − 1 ≈ 65%）' },
      { var: 's_cpi_yoy', mean: 5.0, why: '多抽通胀在 5% 左右的月份' },
    ] },
    { name: '回归常态', months: 168, views: [], overrides: {}, confidence: 1.0 },
  ],
};

// The draft (narrative, horizon, editor text and where it came from, path count) outlives one
// rendering of the tab: the K-line colour toggle re-renders the page and a visit to another tab
// rebuilds it, and neither may wipe what the user typed or what the LLM returned. Kept for the
// browser session (sessionStorage when the browser allows it, otherwise for this page load).
const DRAFT_KEY = 'msl-lab-draft';
const draft = (() => {
  try { const d = JSON.parse(sessionStorage.getItem(DRAFT_KEY) || 'null'); return d && typeof d === 'object' ? d : {}; } catch (_) { return {}; }
})();
function saveDraft(patch) {
  Object.assign(draft, patch);
  try { sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draft)); } catch (_) { /* private mode or quota: keep it in memory */ }
}

// Same rule as schema.slugify, so collisions with built-in ids are caught before saving.
const slugify = (t) => String(t || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40) || 'scenario';
const clampInt = (v, lo, hi, dflt) => { const n = Math.round(Number(v)); return Number.isFinite(n) ? Math.min(Math.max(n, lo), hi) : dflt; };

// The browser's JSON error descriptions (V8 wording; others fall through unchanged) in Chinese.
const JSON_ZH = [
  [/^Unterminated string/i, '字符串缺少结尾的英文双引号'],
  [/^Expected ',' or '}' after property value/i, '属性值后面缺少逗号或右花括号 }'],
  [/^Expected ',' or ']' after array element/i, '数组元素后面缺少逗号或右方括号 ]'],
  [/^Expected double-quoted property name/i, '属性名要用英文双引号（也可能是最后一项后面多了逗号）'],
  [/^Expected property name or '}'/i, '缺少属性名或右花括号 }'],
  [/^Expected ':' after property name/i, '属性名后面缺少英文冒号 :'],
  [/^Bad control character in string literal/i, '字符串里有换行等控制字符（换行请写成 \\n）'],
  [/^Bad escaped character/i, '字符串里有无效的反斜杠转义'],
  [/^Unexpected end of JSON input/i, '文本不完整：可能缺少右括号、右方括号或引号'],
  [/^Unexpected non-whitespace character after JSON/i, 'JSON 结束之后还有多余的内容'],
  [/^No number after minus sign/i, '负号后面缺少数字'],
  [/^Unterminated fractional number/i, '小数点后面缺少数字'],
  [/^Exponent part is missing a number/i, '指数部分缺少数字'],
];
const jsonZh = (s) => (JSON_ZH.find(([rx]) => rx.test(s)) || [null, s])[1];

// JSON.parse error -> Chinese message with line / column and the offset in the editor text.
// `lead` is the number of characters trimmed from the start before parsing. Browsers report the
// place differently: "at position N" (Chrome), "line N column M" (Firefox), or, for V8's
// "Unexpected token 'x', ..."<context>"... is not valid JSON", only a snippet around it.
function jsonError(err, text, lead = 0) {
  const raw = String(err.message || err);
  const src = text.slice(lead);
  // the browser's own position counts from the trimmed text: keep only its description
  let pos = null, what = raw.replace(/\s*\(line \d+ column \d+\)/, '').replace(/\s*(in JSON )?at position \d+/, '');
  let m;
  if ((m = /position (\d+)/.exec(raw))) pos = Number(m[1]);
  else if ((m = /line (\d+) column (\d+)/.exec(raw))) {
    pos = src.split('\n').slice(0, Number(m[1]) - 1).reduce((t, l) => t + l.length + 1, 0) + Number(m[2]) - 1;
  } else if ((m = /^Unexpected token '([\s\S])', (\.\.\.)?"([\s\S]*?)"(?:\.\.\.)? is not valid JSON$/.exec(raw))) {
    const at = src.indexOf(m[3]);
    // V8 keeps up to 10 characters before the bad token when it cuts the start of the context
    if (at >= 0) pos = at + (m[2] ? Math.min(10, m[3].length - 1) : Math.max(0, m[3].indexOf(m[1])));
    what = `意外的字符 '${m[1]}'`;
  }
  if (pos === null || pos < 0 || pos > src.length) {
    if (/end of JSON/i.test(raw)) return { msg: jsonZh('Unexpected end of JSON input'), pos: null };
    const zh = jsonZh(raw);
    return { msg: `${zh !== raw ? zh : raw.length > 160 ? `${raw.slice(0, 160)}…` : raw}（常见原因：多余的逗号、缺少引号、注释或中文标点）`, pos: null };
  }
  pos += lead;
  const before = text.slice(0, pos);
  const near = text.slice(pos, pos + 24).split('\n')[0];
  return { msg: `第 ${before.split('\n').length} 行第 ${pos - before.lastIndexOf('\n')} 列附近${near ? `（“${near}”）` : ''}：${jsonZh(what)}`, pos };
}

export async function render(root, ctx) {
  const { api, ui, meta, param } = ctx;
  const alive = ctx.alive || (() => root.isConnected);
  const { h, card } = ui;
  ensureDetailStyle();
  ensureStyle('lab', CSS);
  const startYm = meta?.today?.date || null;
  const HORIZON = Number(meta?.horizon_months) || DEFAULT_HORIZON;

  // ---- the rule, stated first
  const step = (cls, title, text) => h('div', { class: `lab-step ${cls}` }, h('b', {}, title), text);
  const arrow = () => h('span', { class: 'lab-arrow' }, '→');
  root.append(h('div', { class: 'lab-hero' },
    h('h2', {}, 'LLM 只负责把叙事翻译成冲击、视图和传导路径；所有 K 线和年化收益都由引擎用真实历史月份计算。'),
    h('div', { class: 'lab-flow' },
      step('', '① 你的叙事', '一段中文：发生了什么、持续多久、怎么传导'), arrow(),
      step('llm', '② LLM 翻译', '只输出情景规格：结构冲击、分阶段视图、有上限的覆盖项、因果链'), arrow(),
      step('', '③ 规格校验（validate_spec）','不认识的变量被忽略，数值截断到历史 2%–98% 区间，阶段补齐到 180 个月'), arrow(),
      step('engine', '④ 引擎', '熵池给 1973 年以来的真实月份重新赋权，按 6 个月区块抽样，经今天的传导模型生成资产回报'), arrow(),
      step('engine', '⑤ 结果', 'K 线、扇形图、年化收益、相关性：每个数字都来自真实历史月份')),
    h('p', { class: 'lab-foot' }, 'LLM 写出的数字只是对历史月份平均值的要求（视图），不是价格或收益率；每一处被截断、忽略或收缩的地方都会列在校验报告里。')));

  const statusBox = h('div', { class: 'lab-status' }, ui.loading('检查 LLM 是否可用…'));
  root.append(statusBox, h('div', { style: { height: '16px' } }));

  // ---- ① narrative
  const narr = h('textarea', { class: 'lab-narr', placeholder: '例如：中东冲突升级，霍尔木兹海峡航运中断半年，油价涨到 140 美元……写清楚冲击、持续时间和你认为的传导路径。',
    oninput: () => saveDraft({ narr: narr.value }) });
  const horizon = h('input', { type: 'number', min: '12', max: '600', step: '6', value: String(HORIZON),
    onchange: () => saveDraft({ horizon: horizon.value }) });
  if (typeof draft.narr === 'string') narr.value = draft.narr;
  if (draft.horizon) horizon.value = String(draft.horizon);
  const transBtn = h('button', { class: 'btn primary', disabled: true, onclick: () => translate() }, '用 LLM 翻译成情景规格');
  const transMsg = h('span', { class: 'lab-msg' });
  // 载入示例: filled once llm_translator.json has loaded (renderDemos)
  const demoRow = h('div', { class: 'lab-examples lab-demo' });
  root.append(card({ title: '① 写下情景叙事', sub: '点一个例子或自己写；LLM 会把它翻译成下面第 ② 步的情景规格（JSON），你可以再修改。' },
    h('div', { class: 'lab-examples' }, h('span', { class: 'muted', style: { fontSize: '12px' } }, '例子：'),
      EXAMPLES.map(ex => h('button', { title: ex.text, onclick: () => { narr.value = ex.text; saveDraft({ narr: narr.value }); narr.focus(); } }, ex.title))),
    demoRow,
    narr,
    h('div', { class: 'lab-bar' }, h('label', {}, '情景时长（月）', horizon), transBtn, transMsg),
    h('p', { class: 'lab-note' }, `时长只影响 LLM 如何安排阶段；引擎固定模拟 ${HORIZON} 个月（15 年），不足的部分自动补一个无视图的“回归常态”阶段，超出的部分从后往前截短。`)));

  const report = h('div', { class: 'lab-report' });
  root.append(report);

  // ---- ② spec editor
  const tplSelect = h('select', { 'aria-label': '内置情景模板' }, h('option', { value: '' }, '载入内置情景…'));
  const editor = h('textarea', { class: 'lab-editor', spellcheck: 'false', placeholder: '情景规格 JSON：可以由第 ① 步的 LLM 生成，也可以从内置情景载入模板或用空白模板手写。' });
  const check = h('div', { class: 'lab-check muted' }, '编辑器是空的。');
  const newSituation = h('div', {});
  root.append(h('div', { style: { height: '16px' } }), card({
    title: '② 情景规格（可编辑 JSON）',
    sub: '这是引擎唯一的输入：分阶段的视图（对抽到的历史月份平均值的要求）、有上限的覆盖项和因果链；运行前会先检查 JSON 格式。',
  },
  h('div', { class: 'lab-bar', style: { marginTop: 0, marginBottom: '10px' } },
    tplSelect, h('button', { class: 'btn', onclick: () => loadTemplate(tplSelect.value) }, '载入为模板'),
    h('button', { class: 'btn', onclick: () => setEditor(BLANK, 'custom') }, '空白模板'),
    h('button', { class: 'btn', onclick: () => formatEditor() }, '格式化 JSON')),
  editor, check, newSituation,
  h('details', { class: 'scenarios-more' }, h('summary', {}, '规格怎么写（字段速查）'),
    h('ul', { class: 'lab-cheat' },
      h('li', {}, h('code', {}, 'phases[].months'), `：各阶段月数，合计 ${HORIZON}（引擎固定模拟 15 年）；最多 8 个阶段，每阶段最多 12 个视图`),
      h('li', {}, h('code', {}, 'views[].var'), '：只能用 s_（状态水平，如 s_cpi_yoy）、f_（月度变化，如 f_oil）、e_（结构冲击，如 e_oil_supply）变量，见下方变量参考'),
      h('li', {}, '目标键（每个视图选一个）：', h('code', {}, 'mean'), ' 阶段平均（变量本身的单位，f_ 为月度）、', h('code', {}, 'mean_annual'),
        ' f_ 的年化平均（= 月度 × 12）、', h('code', {}, 'shift_sd'), ' 比历史均值偏移几个标准差（−3~3）、', h('code', {}, 'min'), ' / ', h('code', {}, 'max'),
        ' 平均值的下限 / 上限；可另加 ', h('code', {}, 'vol_mult'), ' 波动倍数（0.3~3）'),
      h('li', {}, h('code', {}, 'why'), '：中文理由，会显示在情景详情的阶段表里'),
      h('li', {}, h('code', {}, 'overrides'), '：', h('code', {}, 'factor_drift'), ' 因子额外年化漂移（有上限，见变量参考）、', h('code', {}, 'asset_drift'),
        ' 资产额外价格漂移（±8%/年）、', h('code', {}, 'resid_scale'), '（0.5~2）、', h('code', {}, 'r_star'), ' 中性实际利率（−3~3）——只用于历史上没有先例的部分'),
      h('li', {}, h('code', {}, 'confidence'), '：0.3~1，越低越接近无条件历史；', h('code', {}, 'causal_chain'), '：解释图，sign 取 "+"、"-" 或 "±"'),
      h('li', {}, h('code', {}, 'id'), '：小写字母、数字和下划线；与内置情景重名时运行前会自动加 custom_ 前缀，同 id 的自定义情景会被覆盖'))),
  h('div', { class: 'scenarios-gloss' }, GLOSSARY.map(([t, x]) => h('div', {}, h('b', {}, t), '：', x)))));

  // ---- variables reference (filled in the background)
  const refBox = h('div', {}, ui.loading('载入变量参考…'));
  root.append(h('div', { style: { height: '16px' } }),
    h('details', { class: 'card scenarios-more', style: { marginTop: 0 } },
      h('summary', {}, '变量参考：引擎认识的变量和视图的可设范围（展开）'), refBox));

  // ---- ③ run
  const nPaths = h('input', { type: 'number', min: '100', max: '2000', step: '100', value: String(draft.nPaths || 600),
    onchange: () => saveDraft({ nPaths: nPaths.value }) });
  const runBtn = h('button', { class: 'btn primary', onclick: () => run() }, '用引擎运行并保存');
  const runMsg = h('span', { class: 'lab-msg' });
  const runNotes = h('div', {});
  root.append(h('div', { style: { height: '16px' } }), card({
    title: '③ 用引擎运行',
    sub: '引擎按规格给真实历史月份重新赋权并模拟 15 年路径；结果保存为自定义情景，可以在“组合评估”和“提取与破产概率”页加入计算。',
  },
  h('div', { class: 'lab-bar', style: { marginTop: 0 } }, h('label', {}, '模拟路径数（100–2000）', nPaths), runBtn, runMsg),
  runNotes,
  h('p', { class: 'lab-note' }, '600 条路径通常需要 5–20 秒；路径越多分位数越稳定，耗时近似线性增加。')));

  // ---- saved custom scenarios
  const savedBox = h('div', {}, ui.loading());
  root.append(h('div', { style: { height: '16px' } }), card({
    title: '已保存的自定义情景',
    sub: '保存在 data/results/custom/；“打开”显示引擎结果，“载入编辑器”把它的规格放回第 ② 步继续修改。',
  }, savedBox));

  // ---- ④ result
  const resultHead = h('div', { class: 'section-title lab-result' }, '④ 引擎结果');
  const resultBox = h('div', {}, ui.notice('运行第 ③ 步或打开一个已保存的情景后，结果显示在这里。', 'info'));
  root.append(resultHead, resultBox);
  const resultState = { asset: 'us_equity', kline: 'yearly', fan: 'nominal', corr: 'post' };

  // ---------------------------------------------------------------- state + helpers
  let builtins = [];               // index rows
  let builtinIds = new Set();
  let llmAvailable = false;
  let origin = 'custom';           // where the editor text came from: 'llm' | 'custom'
  let baseline = '';               // editor text as loaded (unchanged LLM output keeps source "llm")
  let shownId = null;              // id of the scenario in the result area
  if (typeof draft.editor === 'string') {          // the draft of an earlier rendering of this tab
    editor.value = draft.editor;
    origin = draft.origin === 'llm' ? 'llm' : 'custom';
    baseline = typeof draft.baseline === 'string' ? draft.baseline : '';
  }
  const keepEditor = () => saveDraft({ editor: editor.value, origin, baseline });

  const timer = (el, text) => {
    const t0 = Date.now();
    const tick = () => { el.className = 'lab-msg'; fill(el, h('span', { class: 'spinner' }), `${text}（${Math.round((Date.now() - t0) / 1000)} 秒）`); };
    tick();
    const id = setInterval(() => { if (!el.isConnected) clearInterval(id); else tick(); }, 1000);
    return () => clearInterval(id);
  };
  const say = (el, text, kind = '') => { el.className = `lab-msg ${kind}`; el.textContent = text; };

  function setEditor(spec, from) {
    editor.value = typeof spec === 'string' ? spec : JSON.stringify(spec, null, 2);
    origin = from;
    baseline = editor.value;
    keepEditor();
    inspect();
  }

  function formatEditor() {
    const r = inspect();
    if (r.ok) {
      const unchanged = editor.value === baseline;
      editor.value = JSON.stringify(r.spec, null, 2);
      if (unchanged) baseline = editor.value;
      keepEditor();
      inspect();
    }
  }

  // Live check of the editor text: JSON syntax, shape, months, ids.
  function inspect() {
    const text = editor.value.trim();
    if (!text) { check.className = 'lab-check muted'; check.textContent = '编辑器是空的：先翻译一段叙事，或载入一个模板。'; return { ok: false, empty: true }; }
    let spec;
    try { spec = JSON.parse(text); } catch (err) {
      const e = jsonError(err, editor.value, editor.value.length - editor.value.trimStart().length);
      check.className = 'lab-check err';
      check.textContent = `JSON 语法错误：${e.msg}`;
      return { ok: false, msg: `JSON 语法错误：${e.msg}`, pos: e.pos };
    }
    const bad = (msg) => { check.className = 'lab-check err'; check.textContent = msg; return { ok: false, msg }; };
    if (!spec || typeof spec !== 'object' || Array.isArray(spec)) return bad('规格必须是一个 JSON 对象（{ … }）。');
    if (!Array.isArray(spec.phases) || !spec.phases.length) return bad('规格至少需要一个阶段：phases 必须是非空数组。');
    const months = spec.phases.reduce((t, p) => t + (Number(p?.months) || 0), 0);
    const nViews = spec.phases.reduce((t, p) => t + (Array.isArray(p?.views) ? p.views.length : 0), 0);
    const notes = [];
    if (months !== HORIZON) {
      notes.push(months < HORIZON ? `阶段合计 ${months} 个月：引擎会补一个 ${HORIZON - months} 个月的“回归常态”阶段`
        : `阶段合计 ${months} 个月：超出 ${HORIZON} 的部分会从后往前截短`);
    }
    const sid = slugify(spec.id || '');
    if (!spec.id) notes.push('没有 id：运行时自动生成');
    else if (builtinIds.has(sid)) notes.push(`id「${sid}」与内置情景重名：运行时改为 ${slugify(`custom_${sid}`)}，以免被内置情景遮住`);
    else if (sid !== spec.id) notes.push(`id 只能包含小写字母、数字和下划线（最多 40 个字符）：运行时改为 ${sid}`);
    if (!nViews) notes.push('没有任何视图：结果等同于无条件的历史基准');
    check.className = 'lab-check';
    fill(check, h('span', { class: 'is-good' }, '✓ JSON 有效'),
      ` · ${spec.name || '未命名'}（${sid}）· ${spec.phases.length} 个阶段 · 合计 ${months} 个月 · ${nViews} 个视图`,
      notes.length ? h('ul', {}, notes.map(n => h('li', {}, n))) : null);
    return { ok: true, spec };
  }
  let inspectTimer = null;
  editor.addEventListener('input', () => { keepEditor(); clearTimeout(inspectTimer); inspectTimer = setTimeout(inspect, 300); });

  function showResult(d) {
    shownId = d?.id || null;
    renderScenarioDetail(resultBox, d, ctx, { state: resultState, labLink: false });
    ui.setRoute('lab', encodeURIComponent(d.id), alive);        // a run that ends after the reader left changes nothing
    resultHead.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ---------------------------------------------------------------- actions
  async function translate() {
    const text = narr.value.trim();
    if (!text) { say(transMsg, '请先写一段情景叙事。', 'err'); narr.focus(); return; }
    const months = clampInt(horizon.value, 12, 600, HORIZON);
    horizon.value = String(months);
    transBtn.disabled = true;
    const stop = timer(transMsg, 'LLM 正在把叙事翻译成情景规格');
    try {
      const res = await api.post('llm/translate', { narrative: text, horizon_months: months });
      stop();
      if (!res?.spec) throw new Error('翻译结果里没有情景规格');
      setEditor(res.spec, 'llm');
      renderReport(res);
      say(transMsg, `翻译完成，用时 ${isNum(res.elapsed_s) ? res.elapsed_s.toFixed(1) : '—'} 秒；规格已放进第 ② 步。`, 'ok');
    } catch (err) {
      stop();
      say(transMsg, `翻译失败：${errorText(err, { notBuilt: false })}`, 'err');
    } finally {
      transBtn.disabled = !llmAvailable;
    }
  }

  // What the LLM produced, how validate_spec changed it, and what each phase will draw.
  function renderReport(res) {
    const u = res.usage || {};
    const specPhases = res.spec?.phases || [];
    const phases = (Array.isArray(res.preview) ? res.preview : []).map((p, k) => ({
      name: p.name, start_month: p.start_month, months: p.months, confidence: p.confidence,
      views: specPhases[k]?.views || [], overrides: specPhases[k]?.overrides || {},
      ep: { ens: p.ens, ens_ratio: p.ens_ratio, converged: p.converged, shrink: p.shrink, block: p.block,
        episodes: p.episodes, shrink_reason: p.shrink_reason, ens_requested: p.ens_requested,
        era_weights: p.era_weights, top_months: p.top_months, views: p.views },
    }));
    const tokens = [u.input_tokens, u.output_tokens].every(isNum) ? `输入 ${u.input_tokens.toLocaleString('zh-CN')} · 输出 ${u.output_tokens.toLocaleString('zh-CN')}` : '—';
    fill(report,
      h('div', { class: 'section-title' }, 'LLM 翻译报告'),
      card({ title: `「${res.spec?.name || '未命名'}」`, sub: res.spec?.summary || 'LLM 只给出规格；下面的预览是引擎按这份规格给历史月份赋权的结果，还没有模拟路径。' },
        h('dl', { class: 'kv' },
          h('dt', {}, '模型'), h('dd', {}, res.model || '—', res.requested_model && res.requested_model !== res.model ? `（请求 ${res.requested_model}）` : ''),
          h('dt', {}, '用时'), h('dd', {}, isNum(res.elapsed_s) ? `${res.elapsed_s.toFixed(1)} 秒` : '—'),
          h('dt', {}, 'tokens'), h('dd', {}, tokens, u.fallback_used ? '（使用了备用模型）' : ''),
          h('dt', {}, '情景 id'), h('dd', {}, h('span', { class: 'mono' }, res.spec?.id || '—')))),
      h('div', { style: { height: '16px' } }),
      validationCard(res.validation),
      phases.length ? h('div', { class: 'section-title' }, '预览：每个阶段会抽哪些历史月份（翻译时的规格；编辑后以运行结果为准）') : null,
      phases.length ? h('div', { class: 'scenarios-stack' }, phases.map((ph, k) => phaseCard(ph, k, { startYm }))) : null,
      res.raw ? h('details', { class: 'card scenarios-more' }, h('summary', {}, 'LLM 原始输出（修复和校验之前）'),
        h('pre', { class: 'code' }, JSON.stringify(res.raw, null, 2))) : null);
  }

  async function loadTemplate(id) {
    if (!id) { say(runMsg, '先在下拉框里选一个内置情景。', 'err'); tplSelect.focus(); return; }
    try {
      const d = await api.get(`scenarios/${id}`);
      if (!d?.spec) throw new Error('这个情景没有附带规格');
      const s = JSON.parse(JSON.stringify(d.spec));
      delete s.source;
      if (!s.reversion || !Object.keys(s.reversion).length) delete s.reversion;
      s.id = slugify(`custom_${s.id || id}`);
      s.name = `${s.name || id}（自定义）`;
      setEditor(s, 'custom');
      tplSelect.value = id;
      say(runMsg, `已载入「${d.name}」作为模板（id 改为 ${s.id}，不会覆盖内置情景）。改完后点“用引擎运行并保存”。`, 'ok');
    } catch (err) {
      const why = err.status === 404 || err.status === 400 ? `没有找到内置情景「${id}」` : errorText(err);
      say(runMsg, `载入模板失败：${why}`, 'err');
    }
  }

  // ---- 载入示例 (no API key needed): a built-in's narrative into ①, its hand-written spec into ②, the LLM's
  // output form when llm_translator.json carries `examples` (the prompt's few-shot examples), otherwise the
  // built-in's validated spec. Always labelled: this is not a translation.
  async function renderDemos(lt) {
    let list = (Array.isArray(lt?.examples) ? lt.examples : []).filter(x => x && x.id && Array.isArray(x.phases));
    let form = 'llm';
    if (!list.length) {
      form = 'engine';
      const ids = (Array.isArray(lt?.prompt?.examples) && lt.prompt.examples.length ? lt.prompt.examples : ['oil_war', 'k_shaped']).slice(0, 2);
      const got = await Promise.all(ids.map(id => api.get(`scenarios/${id}`)
        .then(d => (d?.spec ? { ...d.spec, narrative: d.narrative || d.spec.narrative || '' } : null)).catch(() => null)));
      list = got.filter(Boolean);
    }
    if (!alive()) return;
    if (!list.length) { demoRow.remove(); return; }
    fill(demoRow, h('span', { class: 'muted', style: { fontSize: '12px' } }, '没有 API key？载入示例：'),
      list.map(ex => h('button', { title: `把内置情景「${ex.name}」的叙事和手写规格分别放进第 ① ② 步（不是 LLM 翻译）`,
        onclick: () => loadDemo(ex, form) }, ex.name || ex.id)));
  }

  function loadDemo(ex, form) {
    const spec = JSON.parse(JSON.stringify(ex));
    delete spec.source;
    spec.id = slugify(`example_${ex.id}`);
    spec.name = `${ex.name || ex.id}（示例）`;
    narr.value = String(ex.narrative || '');
    saveDraft({ narr: narr.value });
    setEditor(spec, 'custom');
    renderDemo(ex, spec, form);
    say(transMsg, '已载入示例：叙事在上面，规格在第 ② 步。这不是 LLM 的实时翻译。', 'ok');
    report.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderDemo(ex, spec, form) {
    const phases = Array.isArray(spec.phases) ? spec.phases : [];
    const rows = phases.map((ph, k) => ({ k, ph }));
    const views = (ph) => (Array.isArray(ph.views) && ph.views.length
      ? h('ul', { class: 'lab-cheat' }, ph.views.map(v => h('li', {}, `${ui.varName(v.var) || v.var}：${viewSpecText(v)}`)))
      : h('span', { class: 'muted' }, '无视图：按无条件的历史抽样'));
    const overrides = (ph) => { const items = overrideItems(ph.overrides); return items.length ? h('ul', { class: 'lab-cheat' }, items.map(t => h('li', {}, t))) : '—'; };
    const runDemo = () => { runBtn.scrollIntoView({ behavior: 'smooth', block: 'center' }); run(); };
    const calibrated = !!builtins.find(b => b.id === ex.id)?.calibrated;
    fill(report,
      h('div', { class: 'section-title' }, '示例：叙事 → 规格（手写，不是 LLM 翻译）'),
      card({ title: `「${ex.name || ex.id}」`, sub: ex.summary || '' },
        ui.notice(h('div', {},
          h('b', {}, '这不是 LLM 的实时翻译，也没有调用任何 API。'),
          `第 ① 步的叙事和第 ② 步的规格都来自内置情景库（scenariolab/scenarios/specs/${ex.id}.yaml），规格是人工写好的`,
          form === 'llm'
            ? '，按 LLM 输出的 JSON 格式排列（与 LLM 在提示词里看到的示例相同）'
            : '（这份 llm_translator.json 较旧，规格按引擎内部的格式显示；运行 python -m scenariolab.build --only llm 后改为 LLM 的输出格式）',
          calibrated ? '；它的收入分配计算器（calibration）已经换算成视图和因子漂移，写在规格里。' : '。',
          h('ol', { class: 'lab-demo-steps' },
            h('li', {}, '叙事：发生了什么、怎么传导（LLM 的输入）；'),
            h('li', {}, '规格：分阶段的视图、有上限的覆盖项和因果链（LLM 唯一的输出，下表逐阶段列出）；'),
            h('li', {}, '引擎：按规格给 1973 年以来的真实月份重新赋权并模拟 15 年路径，K 线和年化收益都由它算出。'))), 'info'),
        h('div', { style: { height: '10px' } }),
        table([
          { key: 'k', label: '阶段', left: true, fmt: (k, r) => `${k + 1}. ${r.ph.name || ''}` },
          { value: (r) => r.ph.months, label: '月数' },
          { value: (r) => r, label: '视图（对抽到的历史月份平均值的要求）', left: true, wrap: true, cls: 'scenarios-wrap', fmt: (r) => views(r.ph) },
          { value: (r) => r, label: '覆盖项', left: true, wrap: true, cls: 'scenarios-wrap', fmt: (r) => overrides(r.ph) },
        ], rows, { stack: true }),
        h('div', { class: 'lab-bar' },
          h('button', { class: 'btn primary', onclick: runDemo }, '用引擎运行这个示例'),
          h('a', { class: 'btn', href: `#scenarios?${encodeURIComponent(ex.id)}` }, '查看这个内置情景的结果'))));
  }

  async function run() {
    const r = inspect();
    if (!r.ok) {
      say(runMsg, r.empty ? '编辑器是空的：先翻译一段叙事或载入一个模板。' : r.msg, 'err');
      if (isNum(r.pos)) { editor.focus(); editor.setSelectionRange(r.pos, Math.min(r.pos + 1, editor.value.length)); }
      return;
    }
    const spec = r.spec;
    const source = origin === 'llm' && editor.value === baseline ? 'llm' : 'custom';
    const notes = [];
    let sid = slugify(spec.id || '');
    if (!spec.id || sid === 'scenario') { sid = `custom_${Date.now().toString(36)}`; notes.push(`规格没有可用的 id，已自动设为 ${sid}`); }
    // slugify again: the server keeps at most 40 characters, so the saved id matches the editor
    if (builtinIds.has(sid)) { const to = slugify(`custom_${sid}`); notes.push(`id「${sid}」与内置情景重名，已改为 ${to}`); sid = to; }
    if (spec.id !== sid) {
      spec.id = sid;
      const text = JSON.stringify(spec, null, 2);
      editor.value = text;
      if (source === 'llm') baseline = text;
      keepEditor();
      inspect();
    }
    const paths = clampInt(nPaths.value, 100, 2000, 600);
    nPaths.value = String(paths);
    saveDraft({ nPaths: nPaths.value });
    runBtn.disabled = true;
    fill(runNotes, ...(notes.length ? [ui.notice(notes.join('；'), 'info')] : []));
    const stop = timer(runMsg, `引擎正在用真实历史月份模拟 ${paths} 条路径`);
    try {
      const d = await api.post('scenarios/custom/run', { spec, n_paths: paths, save: true, source });
      stop();
      api.invalidate('scenarios/custom');
      api.invalidate(`scenarios/${d.id}`);
      const w = d.validation?.warnings?.length || 0, c = d.validation?.clipped?.length || 0;
      say(runMsg, `完成并已保存为「${d.name}」（${d.id}）。校验：${w} 条提示，${c} 处截断。`, 'ok');
      showResult(d);
      loadSaved();
    } catch (err) {
      stop();
      say(runMsg, `运行失败：${errorText(err)}`, 'err');
    } finally {
      runBtn.disabled = false;
    }
  }

  async function openSaved(id, { intoEditor = false, scroll = true } = {}) {
    fill(resultBox, ui.loading(`载入情景 ${id}…`));
    try {
      const d = await api.get(`scenarios/${id}`, { fresh: true });
      if (intoEditor || !editor.value.trim()) {
        if (d.spec) setEditor(d.spec, d.source === 'llm' ? 'llm' : 'custom');
      }
      if (scroll) showResult(d);
      else { shownId = d.id; renderScenarioDetail(resultBox, d, ctx, { state: resultState, labLink: false }); }
    } catch (err) {
      disposeCharts(resultBox);
      fill(resultBox, ui.notice(err.status === 404 ? `没有找到自定义情景「${id}」（可能已被删除）。`
        : err.status === 400 ? `情景 id「${id}」不合法：只能包含小写字母、数字和下划线。`
          : `无法载入情景「${id}」：${errorText(err, { notBuilt: false })}`, 'error'));
    }
  }

  async function loadIntoEditor(id) {
    try {
      const d = await api.get(`scenarios/${id}`, { fresh: true });
      if (!d?.spec) throw new Error('这个情景没有附带规格');
      setEditor(d.spec, d.source === 'llm' ? 'llm' : 'custom');
      say(runMsg, `已把「${d.name}」的规格载入第 ② 步；再次运行会覆盖同 id 的结果。`, 'ok');
      editor.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (err) {
      say(runMsg, `载入失败：${errorText(err, { notBuilt: false })}`, 'err');
    }
  }

  async function removeSaved(item) {
    if (!window.confirm(`删除自定义情景「${item.name || item.id}」？结果文件会被删除，无法撤销。`)) return;
    try {
      await api.del(`scenarios/custom/${item.id}`);
      api.invalidate('scenarios/custom');
      api.invalidate(`scenarios/${item.id}`);
      if (shownId === item.id) {
        disposeCharts(resultBox);
        fill(resultBox, ui.notice(`自定义情景「${item.name || item.id}」已删除。`, 'info'));
        shownId = null;
        ui.setRoute('lab', '', alive);
      }
    } catch (err) {
      window.alert(`删除失败：${errorText(err, { notBuilt: false })}`);
    }
    loadSaved();
  }

  async function loadSaved() {
    try {
      const r = await api.get('scenarios/custom', { fresh: true });
      const items = (r?.scenarios || []).filter(s => s && s.id);
      if (!items.length) {
        fill(savedBox, ui.notice('还没有保存的自定义情景：运行第 ③ 步后会自动保存在这里。', 'info'));
        return;
      }
      fill(savedBox, table([
        { key: 'name', label: '名称', wrap: true, fmt: (x, s) => h('div', {}, h('div', { style: { fontWeight: 600 } }, x || s.id),
          s.summary ? h('div', { class: 'muted', style: { fontSize: '12px' } }, s.summary) : null) },
        { key: 'id', label: 'id', left: true, fmt: (x) => h('span', { class: 'mono' }, x) },
        { key: 'source', label: '来源', left: true, fmt: (x) => ui.chip(x === 'llm' ? 'LLM 翻译' : '手动编辑', x === 'llm' ? 'accent' : '') },
        { key: 'created', label: '保存时间', fmt: (x) => x || '—' },
        { key: 'prior_prob', label: '主观先验', fmt: (x) => pct(x, 0) },
        { value: (s) => s, label: '', fmt: (s) => h('div', { class: 'lab-actions' },
          h('button', { class: 'btn', onclick: () => openSaved(s.id) }, '打开'),
          h('button', { class: 'btn', onclick: () => loadIntoEditor(s.id) }, '载入编辑器'),
          h('button', { class: 'btn', onclick: () => removeSaved(s) }, '删除')) },
      ], items));
    } catch (err) {
      fill(savedBox, ui.notice(`无法读取自定义情景：${errorText(err, { notBuilt: false })}`, 'error'));
    }
  }

  // Variables the engine knows, with the range a phase's views can reach (6-month means).
  function renderReference(lt) {
    const vars = Array.isArray(lt?.variables) ? lt.variables : [];
    if (!vars.length) { fill(refBox, ui.notice('llm_translator.json 里没有变量表。', 'info')); return; }
    const kinds = { state: '状态', factor: '因子', shock: '冲击' };
    const outside = vars.filter(v => v.kind === 'state' && v.outside_support);
    if (outside.length) {
      fill(newSituation, ui.notice(h('div', {},
        h('b', {}, `今天有 ${outside.length} 个状态超出 1973 年以来单月的 2%–98% 区间（新形势）：`),
        h('ul', { class: 'lab-new', style: { margin: '6px 0 4px', paddingLeft: '18px' } }, outside.map(v =>
          h('li', {}, `${v.name}：今天 ${fmtVar(v.var, v.today)}，历史区间 ${fmtVar(v.var, v.p2)} ~ ${fmtVar(v.var, v.p98)}`))),
        '视图只能在历史上挑选月份，表达不了这些没有先例的水平；要刻画它们（例如白领工资被压制、利润占比创新高），请用有上限的 factor_drift 覆盖项，或接受被截断。'), 'info'));
      newSituation.style.marginTop = '10px';
    }
    let kind = 'all', q = '';
    const tableBox = h('div', {});
    // readable range, plus the numbers to type into the spec where they differ (f_: mean_annual)
    const num = (x, d) => (isNum(x) ? String(Number(x.toFixed(d))) : '—');
    const range = (v) => {
      const shown = `${fmtVar(v.var, v.block_p2)} ~ ${fmtVar(v.var, v.block_p98)}`;
      if (!v.var.startsWith('f_') || !isNum(v.block_p2) || !isNum(v.block_p98)) return shown;
      return h('div', { class: 'scenarios-var' }, shown,
        h('span', { class: 'mono' }, `mean_annual：${num(v.block_p2 * 12, 3)} ~ ${num(v.block_p98 * 12, 3)}`));
    };
    const draw = () => {
      const rows = vars.filter(v => (kind === 'all' || v.kind === kind) &&
        (!q || `${v.var} ${v.name}`.toLowerCase().includes(q.toLowerCase())));
      if (!rows.length) { fill(tableBox, ui.notice('没有匹配的变量。', 'info')); return; }
      const el = table([
        { key: 'var', label: '变量', fmt: (x) => varCell(x) },
        { key: 'kind', label: '类型', left: true, fmt: (x) => kinds[x] || x },
        { value: (v) => v, label: '视图可设范围（阶段平均）', left: true, cls: 'scenarios-wrap', fmt: (v) => range(v) },
        { key: 'median', label: '历史中位数', fmt: (x, v) => fmtVar(v.var, x) },
        { value: (v) => (v.kind === 'state' ? v.today : v.last12_mean), label: '近期值',
          fmt: (x, v) => (isNum(x) ? h('div', { class: v.outside_support ? 'scenarios-changed' : '' },
            fmtVar(v.var, x), v.outside_support ? h('div', { style: { fontSize: '11px' } }, '超出历史区间') : null) : '—') },
        { key: 'asset_driver', label: '资产模型', fmt: (x) => (x ? '是' : '—') },
      ], rows);
      el.querySelector('table')?.classList.add('scenarios-vtable');   // names and ranges wrap
      fill(tableBox, el);
    };
    const search = h('input', { type: 'text', placeholder: '搜索变量名或 id', oninput: (e) => { q = e.target.value.trim(); draw(); } });
    const bounds = lt.factor_drift_bounds || {};
    const boundText = (f, b) => {
      if (!isNum(b)) return '—';
      const unit = f === 'cpi' ? 'Δlog' : ui.META.factors[f]?.unit;
      const p = (y) => `${((Math.exp(y) - 1) * 100).toFixed(1)}%`;
      return unit === 'Δpp' ? `±${b.toFixed(2)}pp/年` : unit === 'Δlog' ? `±${b.toFixed(3)}/年（对数，约 ${p(-b)} ~ +${p(b)}）`
        : unit === 'Δ' ? `±${b.toFixed(1)} 点/年` : `±${b.toFixed(3)}/年`;
    };
    const driftRows = Object.entries(bounds).map(([f, b]) => ({ f, b }));
    const ad = lt.asset_drift || {};
    fill(refBox,
      h('p', { class: 'lab-note', style: { marginTop: '10px' } }, '视图约束的是一个阶段里抽到的月份的平均值，所以可设范围是 6 个月平均值的 2%–98% 分位（比单月窄得多），超出会被截断。',
        'f_ 因子是月度变化：对数变化换算成年化涨跌幅（%）显示，百分点变化按年化百分点显示；写进规格时用下面那行 mean_annual 的数值（年化对数变化或年化百分点）。',
        'e_ 冲击以标准差为单位，常用 shift_sd。近期值：状态取今天的值，因子和冲击取近 12 个月均值；“资产模型”列为“是”表示资产回报模型直接用这个变量传导到资产。'),
      h('div', { class: 'lab-filter' },
        ui.seg([{ value: 'all', label: '全部' }, { value: 'state', label: '状态 s_' }, { value: 'factor', label: '因子 f_' }, { value: 'shock', label: '冲击 e_' }],
          kind, (v) => { kind = v; draw(); }),
        search, h('span', { class: 'muted', style: { fontSize: '12px' } }, `${vars.length} 个变量`)),
      tableBox,
      driftRows.length ? h('div', { style: { marginTop: '14px' } },
        h('div', { class: 'scenarios-sub' }, '覆盖项的上限：factor_drift（每年，±）'),
        h('div', { style: { maxWidth: '640px' } }, table([
          { key: 'f', label: '因子', fmt: (f) => (f === 'cpi' ? 'CPI 总体（cpi）' : `${ui.factorName(f)}（${f}）`) },
          { key: 'b', label: '上限（每年 ±）', fmt: (b, r) => boundText(r.f, b) },
        ], driftRows)),
        h('p', { class: 'lab-note' }, `asset_drift：每年 ±${pct(ad.bound ?? 0.08, 0)} 的额外价格漂移，只能用于 ${(ad.assets || []).map(a => ui.assetName(a)).join('、') || '股票、REITs 和黄金'}。`)) : null,
      lt.prompt?.system ? h('details', { class: 'scenarios-more' },
        h('summary', {}, `LLM 看到的系统提示词（${(lt.prompt.chars || lt.prompt.system.length).toLocaleString('zh-CN')} 字，示例：${(lt.prompt.examples || []).join('、') || '—'}）`),
        h('pre', { class: 'code lab-wrap' }, lt.prompt.system)) : null);
    draw();
  }

  function renderStatus(st) {
    llmAvailable = !!st?.available;
    transBtn.disabled = !llmAvailable;
    transBtn.title = llmAvailable ? '' : '需要先配置 Anthropic API 凭据';
    if (llmAvailable) {
      fill(statusBox, ui.notice(`LLM 已就绪：模型 ${st.model || '—'}（${st.reason || '已配置凭据'}）。翻译一段叙事通常需要 20–90 秒。`, 'info'));
      return;
    }
    fill(statusBox, h('div', { class: 'notice' },
      h('b', {}, 'LLM 翻译暂不可用'), st?.reason ? `：${st.reason}` : '',
      h('ol', {},
        h('li', {}, '在项目根目录的 ', h('code', {}, '.env'), ' 文件里写入 ', h('code', {}, 'ANTHROPIC_API_KEY=sk-ant-…'), '（可以从 ', h('code', {}, '.env.example'), ' 复制）；'),
        h('li', {}, '安装可选依赖：', h('code', {}, 'pip install -e ".[llm]"'), '；'),
        h('li', {}, '重启服务：', h('code', {}, 'python -m scenariolab.server'), '。')),
      h('div', { style: { marginTop: '6px' } }, '没有 key 也能用：点第 ① 步的“载入示例”，看一遍叙事 → 规格 → 引擎的完整流程（规格是手写的，不是翻译）；',
        '或者在第 ② 步从内置情景载入模板、手写规格，直接用引擎运行。')));
  }

  // ---------------------------------------------------------------- boot (nothing below blocks the page)
  api.get('llm/status', { fresh: true }).then(renderStatus).catch((err) => {
    renderStatus(null);
    statusBox.prepend(ui.notice(`无法查询 LLM 状态：${errorText(err, { notBuilt: false })}`, 'error'));
  });
  const ltP = api.get('llm_translator');
  ltP.then(renderReference).catch((err) => {
    fill(refBox, ui.notice(`无法载入变量参考：${errorText(err)}`, err.status === 404 ? '' : 'error'));
  });
  ltP.then(renderDemos, () => renderDemos(null)).catch((err) => console.error(err));
  loadSaved();
  inspect();

  try {
    const index = await api.get('scenarios');
    builtins = (index.scenarios || []).slice().sort((a, b) => (b.prob ?? 0) - (a.prob ?? 0));
    builtinIds = new Set(builtins.map(s => s.id));
    builtins.forEach(s => tplSelect.append(h('option', { value: s.id }, `${s.name}（${pct(s.prob, 1)}）`)));
  } catch (err) {
    tplSelect.disabled = true;
    say(runMsg, `无法读取内置情景（模板不可用）：${errorText(err)}`, 'err');
  }

  // deep links. A re-render of the same page (the K-line toggle: ctx.rerender) keeps the editor
  // draft and the reader's scroll position instead of loading the linked spec over it again.
  const redraw = !!ctx.rerender && typeof draft.editor === 'string';
  if (param) {
    const tpl = param.startsWith('tpl=') ? param.slice(4) : builtinIds.has(param) ? param : null;
    if (tpl) {
      if (!redraw) {
        await loadTemplate(tpl);
        setTimeout(() => { if (editor.isConnected) editor.scrollIntoView({ block: 'center' }); }, 60);
      }
    } else if (redraw) {
      openSaved(param, { intoEditor: false, scroll: false });
    } else {
      openSaved(param, { intoEditor: true, scroll: false })
        .then(() => setTimeout(() => { if (resultHead.isConnected) resultHead.scrollIntoView({ block: 'start' }); }, 60));
    }
  }
}
