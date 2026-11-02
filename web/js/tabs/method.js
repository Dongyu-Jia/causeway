// Static methodology page: how the engine works, what it assumes, where data comes from.
// valuation.json (optional) adds the valuation-state settings to the assumptions; regimes.json
// (optional) fills the macro-regime step's thresholds, label names and episode counts.
// The quarterly chapter (SIM_DESIGN §0, §1, §3, §4.3, §6; 季度模型，试运行) comes first: simulation, not prediction;
// the three layers; the answer to "what are the model's factor inputs and how do they become network inputs"; the
// valuation block; why financial conditions are required; the neural network that lost. Its counts come from the
// model card (/api/q/card) and its valuation numbers from the quarterly panel (/api/q/valuation), both optional.
const CMA_METHOD = { valuation_adjusted: '估值调整', historical: '历史漂移' };
// The regime thresholds and labels as in config.REGIMES, used only when regimes.json is missing.
const REGIME_DEFAULTS = {
  params: { ip_swing_pct: 3, u_rise_pp: 0.5, severe_ip_decline_pct: 15, high_inflation_pct: 4, cpi_neg_pct: 0 },
  labels: [{ id: 'expansion', name: '扩张' }, { id: 'inflation', name: '通胀' }, { id: 'deflation', name: '物价负增长', term: '通缩' },
    { id: 'recession', name: '衰退' }, { id: 'depression', name: '深度衰退', term: '萧条' }, { id: 'stagflation', name: '滞胀' }],
};

export async function render(root, { api, ui, meta, alive = () => root.isConnected }) {
  const { h, card } = ui;
  const cma = meta?.cma || {};
  const pct = (x) => (x === undefined || x === null ? '—' : `${(x * 100).toFixed(2)}%`);
  // the missing-factor audit (optional, like the other two) fills step 12 and one limitation with its numbers
  const [val, rg, fa, qcard, qval] = api
    ? await Promise.all([api.optional('valuation', meta), api.optional('regimes', meta), api.optional('factor_audit', meta),
      api.get('q/card').catch(() => null), api.get('q/valuation').catch(() => null)])
    : [null, null, null, null, null];
  if (!alive()) return;

  root.append(h('p', { class: 'lead' },
    '这是模拟，不是预测。预测只在第一层：带概率的情景假说；第二层用一个内部一致的联合模型把情景没说的因子补齐，第三层把因子映射成资产。',
    '只解释季度和更长的期限。大模型只负责把文字翻译成情景表，所有数字都由模型计算。'));
  root.append(...quarterlyChapter(h, card, ui, meta, qcard, qval));

  root.append(h('div', { class: 'section-title' }, '月度引擎（现行默认，切换关通过之前）'));
  root.append(h('p', { class: 'lead' },
    '月度引擎不“发明”未来。它把历史上真实发生过的每个月当作积木：情景只改变每块积木被抽中的概率，',
    '因果模型决定改哪些概率、改多少有上限，最后按今天的利率、估值和股息率把积木重新拼成 15 年的路径。',
    '在季度模型通过切换关之前，内置情景、组合评估和提取页都用它。'));

  // step 7 reads the CMA settings the build used (meta.cma), so it cannot drift from them
  const num = (x, dflt) => (typeof x === 'number' && Number.isFinite(x) ? x : dflt);
  const rev = num(cma.reversion, 0.3), horizon = num(cma.horizon_years, 15), shrink = num(cma.shrink_relative, 0.5);
  const goldRev = num(cma.gold_reversion, 0);
  const step7 = `1973 年以来美股享受了估值大幅抬升的顺风；在今天的高估值下不能照搬。价格漂移扣掉历史估值抬升，并按部分均值回归（今天 P/D 向历史中位数回归 ${(rev * 100).toFixed(0)}%，分摊到 ${horizon.toFixed(0)} 年）调整；`
    + `行业相对大盘的超额漂移收缩 ${(shrink * 100).toFixed(0)}%。黄金没有现金流，它的“估值”就是实际金价：扣掉 1970 年代重新定价带来的历史实际涨幅，长期大致跟随通胀；`
    + (goldRev > 0 ? `并从今天的位置向历史中位数温和回归（${(goldRev * 100).toFixed(0)}%）。` : '今天的实际金价不再额外向历史中位数回归（这一项默认关闭）。');
  const steps = [
    ['1. 历史月度库', ...(panelStep(h, ui, meta, fa) || ['1947–至今的月度面板：宏观因子（油价、产出、商品/服务通胀、政策利率、10 年期收益率、信用利差、美元、失业率、劳动收入占比、利润占比、资本开支、家庭财富、白领就业、工资、信心、估值）和 13 类资产的总回报、价格回报与收入。模型从 1973 年（浮动汇率、黄金自由定价、OPEC 时代）开始估计。'])],
    ['2. 因果识别', '带符号约束的结构 VAR（Rubio-Ramírez–Waggoner–Zha 旋转算法）把每个月的变化拆成石油供给、总需求、货币政策、金融冲击；分时代估计脉冲响应，展示传导机制如何变化。劳动收入占比的冲击用季度局部投影（Jordà）估计它对消费、增长、利润和股价的影响。'],
    ['3. 情景 = 视图', '每个情景分阶段，每个阶段对少数“因”（结构冲击、油价、政策利率、利差、劳动收入占比……）和关键宏观结果给出均值约束，不直接对资产回报下判断。'],
    ['4. 熵池化（Entropy Pooling）', '在满足视图的前提下，找与历史分布 KL 距离最小的权重（Meucci 2008），解的形式是 q ∝ p·exp(λ·f)。视图约束的是“被抽中的 6 个月区块的平均值”，所以按区块起点求解：抽中月份的均值在期望上正好等于视图，不会被单月尖峰旁边的反弹抵消。有效月份数 ENS = exp(H(q)) 衡量“偏离历史多远”；视图组合找不到足够多同时满足的历史区块时，所有视图按同一比例 λ 向历史均值收缩并报告。'],
    ['5. 块状抽样 + 再锚定', '按权重抽连续 6 个月的块，保留通胀惯性和波动聚集；抽到的是变化量，按今天的水平重新拼接。利率变化按今天与当时的利率水平比例缩放（近零利率时期最多放大 1.5 倍；信用利差走阔时至少保留 85%），并向“中性利率 + 通胀”温和回归；实际油价扣掉 1973 年以来的长期上涨后向历史中位数回归（给了油价视图或油价漂移的阶段除外：那里油价按视图走，不做回归）。2020 年 3–8 月的封锁月份视为罕见事件，先验权重降为 1/4。'],
    ['6. 今天的传导', `${transmissionText(ui, meta)}残差“部分耦合”：每个 6 个月区块有一半概率沿用抽中月份自己的残差，否则换成按历史先验独立抽取的残差，免得视图把少数历史片段的残差漂移一起带进来（之前一次引擎消融选出并采用的设定；之后的消融结果和是否改用别的设定，见“诊断与迭代”的引擎消融）。10 年美债按收益率路径重定价，现金跟随政策利率，原油直接取油价变化。`,
      h('a', { href: '#diagnostics?ablation' }, '残差设定的消融证据见“诊断与迭代” →')],
    ['7. 估值锚定（CMA）', step7],
    ['8. K 线与提取', '每个情景 600 条路径：扇形图给出分位数；K 线展示一条“代表路径”（终值和最大回撤都接近中位数），日内高低点用布朗桥补出。组合与提取在同一批路径上计算，提取额随路径自己的 CPI 上调。提取期限超过 15 年时（最长 40 年），第 16 年起没有情景主题：每条情景路径各接一条不加视图的无条件引擎路径（长期回报假设去掉只为前 15 年设定的估值回归项）。'],
    ['9. 怎么检验', '季度和年度：情景条件检验按 1 个季度和 4 个季度报告；12–20 个季度的数字每个十年只有 2–3 个独立窗口，只能当个案。回测窗口把真实宏观路径当作情景，检查引擎能否复现资产，并把误差拆成情景、引擎和传导三部分；条件分两档：“宏观 + 金融条件（含市场价格）”和“只有宏观”。另有相似起点法的一年期回放（历史回放，不是情景结果）。'],
    ['10. 估值：状态与一条腿', '估值的月度变化几乎就是当月的股价变化，放进因子模型等于用股价解释股价，所以不作因子。它是状态和初始条件：月度引擎逐路径记录 P/D（κ=0，只记录、不改变回报），拉回公允值的设定在回测里更差，所以拉力关闭；'
      + '季度模型把股价拆成 CAPE 倍数和基本面两条腿（恒等式），情景可以明说“估值重估”，那相当于直接假设股价（见上面季度模型一章的“估值”）。',
      h('a', { href: '#valuation' }, '证据见“估值与盈利” →')],
  ];
  steps.push(regimeStep(h, rg, meta), auditStep(h, fa));
  root.append(h('div', { class: 'grid cols-2 text' }, steps.map(([t, d, more]) => card({ title: t }, h('p', { style: { margin: 0 } }, d),
    more ? h('p', { style: { margin: '8px 0 0' } }, more) : null))));

  root.append(h('div', { class: 'section-title' }, '当前假设'));
  const adj = cma.adjustment || {};
  root.append(h('div', { class: 'grid cols-2 text' },
    card({ title: '估值锚定参数', sub: '影响所有未来情景的基准漂移' },
      h('dl', { class: 'kv' },
        h('dt', {}, '方法'), h('dd', {}, CMA_METHOD[cma.method] || cma.method || '—'),
        h('dt', {}, '今天的 P/D'), h('dd', {}, cma.pd_now ? cma.pd_now.toFixed(1) : '—'),
        h('dt', {}, '历史中位 P/D'), h('dd', {}, cma.pd_median ? cma.pd_median.toFixed(1) : '—'),
        h('dt', {}, '历史估值顺风（年化）'), h('dd', {}, pct(cma.rerating_tailwind)),
        h('dt', {}, '均值回归项（年化）'), h('dd', {}, pct(cma.reversion_term)),
        h('dt', {}, '美股漂移调整（年化）'), h('dd', {}, pct(cma.market_adjustment)),
        h('dt', {}, '行业超额漂移收缩'), h('dd', {}, cma.shrink_relative !== undefined ? `${cma.shrink_relative * 100}%` : '—'),
        ...valuationRows(h, val)),
      val ? h('p', { style: { margin: '10px 0 0', fontSize: '13px' } }, h('a', { href: '#valuation' }, '估值与盈利：公允值回归、拉力速度和估值状态变体的回测 →')) : null),
    card({ title: '各资产漂移调整', sub: '目标价格漂移 = 历史价格漂移 + 调整' },
      ui.table([
        { key: 'asset', label: '资产', fmt: (v) => ui.assetName(v) },
        { key: 'hist', label: '历史价格漂移', fmt: (v) => pct(v) },
        { key: 'adj', label: '调整', fmt: (v) => pct(v), cls: (v) => ui.signCls(v) },
        { key: 'target', label: '目标', fmt: (v) => pct(v) },
      ], Object.keys(adj).map(a => ({ asset: a, hist: cma.historical_price_drift?.[a], adj: adj[a],
        target: (cma.target_price_drift || {})[a] ?? ((cma.historical_price_drift?.[a] || 0) + adj[a]) })), { stack: true }))));

  root.append(h('div', { class: 'section-title' }, '数据来源'));
  const src = meta?.sources || {};
  root.append(card({ title: '公开数据（构建时下载，不随代码分发）' },
    h('ul', { class: 'list' }, Object.entries(src).map(([k, v]) => h('li', {}, h('a', { href: v, target: '_blank', rel: 'noopener' }, k)))),
    h('ul', { class: 'list' }, (meta?.notes || []).map(n => h('li', { class: 'muted' }, n))),
    h('p', { class: 'muted', style: { margin: '10px 0 0', fontSize: '12px' } }, COPYRIGHT)));

  root.append(h('div', { class: 'section-title' }, '计算口径与局限'));
  root.append(card({},
    h('ul', { class: 'list' },
      h('li', {}, '情景结果由假设决定。这个工具的价值是内部一致和假设透明，不是预测哪种未来会发生；预测只在情景概率里，而概率是手设的判断。'),
      h('li', {}, '季度模型在切换关通过之前只是试运行：它的结果和月度引擎的结果并列显示，默认数字仍来自月度引擎。'),
      h('li', {}, '宏观本身对资产几乎没有样本外信息，已知宏观的能力几乎全部来自信用利差：情景对金融条件怎么说，几乎决定了资产结果，这是情景作者的假设。'),
      h('li', {}, '12–20 个季度的检验统计每个十年只有 2–3 个独立窗口，只能当个案；长期扇形图是模拟，不是检验结果。'),
      h('li', {}, '历史只覆盖美国 1973 年以来；美国是过去半个世纪的赢家，存在幸存者偏差。'),
      h('li', {}, '战争、地缘政治、AGI 在模型里都是外生冲击；超出历史支持的部分（如 AGI 生产率）通过有上限的覆盖参数注入，并在界面上标出。'),
      h('li', {}, '情景概率是第一层的判断（预测只在这里）：默认用手设的主观先验；按今天状态的倾斜（主观先验 × 与当前状态的相似度）默认关闭，只作对照，组合评估的月度情景集仍按它加权并标明。13 个互相重叠的长期情景只有一个实现，这些概率无法像统计预测那样校准。'),
      h('li', {}, '一年期的资产回报本来就很难预测：滚动检验显示“今天像哪段历史”只能轻微调整预测，方向判断不比“总是猜涨”更好；模型的价值在于区间是否可靠、跨资产排序和情景一致性。'),
      h('li', {}, regimeLimit(rg)),
      h('li', {}, auditLimit(fa)),
      h('li', {}, '所有结果都是假设性的模拟，不构成投资建议。'))));
}

// Step 11: the macro regimes. Thresholds and names come from regimes.json definitions (REGIME_DEFAULTS
// without it); every number in the text is a parameter, never typed here. When the selected model has the
// term-premium interaction (meta.json), its own high-inflation switch is named next to the label's in one
// sentence (FACTORS_DESIGN §7.3).
function regimeStep(h, rg, meta) {
  const def = rg?.definitions || {};
  const p = { ...REGIME_DEFAULTS.params, ...(def.params || {}) };
  const labels = Array.isArray(def.labels) && def.labels.length ? def.labels : REGIME_DEFAULTS.labels;
  const n = (x) => String(Number(Number(x).toFixed(2)));
  const names = labels.map(l => l.name).join('、');
  const termed = labels.filter(l => l.term);
  const terms = termed.map(l => `“${l.term}”对应${l.name}`).join('，');
  const F = Object.fromEntries((meta?.factors || []).map(f => [f.id, f]));
  const inter = (meta?.model?.spec?.factors || []).map(id => F[id]?.interaction).find(x => x && isNum(x.above));
  const STATE = { core_yoy: '核心 CPI 同比', cpi_yoy: 'CPI 同比' };
  const switches = inter
    ? `宏观体制的高通胀看当月 CPI 同比；传导模型的期限溢价项看${inter.lag === 1 ? '上月' : `${inter.lag} 个月前`}${STATE[inter.state] || '通胀'}；`
      + (Number(inter.above) === Number(p.high_inflation_pct) ? `门槛都是 ${n(p.high_inflation_pct)}%。`
        : `门槛分别是 ${n(p.high_inflation_pct)}% 和 ${n(inter.above)}%。`)
    : meta?.model?.spec?.regime === 'inflation'
      ? `宏观体制的高通胀看当月 CPI 同比；传导模型的整体通胀状态交互看上月 CPI 同比；门槛都是 ${n(p.high_inflation_pct)}%。`
      : `宏观体制的高通胀看当月 CPI 同比，门槛 ${n(p.high_inflation_pct)}%；传导模型没有通胀状态项。`;
  const text = `每个月按工业产出的方向（下跌 ${n(p.ip_swing_pct)}% 且失业率上升 ${n(p.u_rise_pp)} 个百分点 = 收缩）、`
    + `收缩深度（工业产出累计跌 ${n(p.severe_ip_decline_pct)}%）和 CPI 同比（${n(p.high_inflation_pct)}%、${n(p.cpi_neg_pct)}）`
    + `归入${names}之一；历史月份和模拟路径用同一套规则。情景阶段可以要求多抽某种体制的历史月份（写法见情景实验室的规格说明）。`
    + '体制不进入传导模型和一年期预测：两者都做过样本外检验，没有通过。' + switches
    + (terms ? `美国战后数据里没有真正的${termed.map(l => l.term).join('和')}，所以${terms}，都是较温和的近似。` : '');
  return ['11. 宏观体制', text, h('a', { href: '#regimes' }, '宏观体制 →')];
}

// The limitation line: the regime history is thin. Episode counts since 1973 from regimes.json counts.
function regimeLimit(rg) {
  const lab = rg?.counts?.['1973']?.expost?.labels || {};
  const names = Object.fromEntries((rg?.definitions?.labels || REGIME_DEFAULTS.labels).map(l => [l.id, l.name]));
  const ids = ['recession', 'depression', 'deflation'];
  const runs = ids.map(id => lab[id]?.runs).filter(x => typeof x === 'number');
  const lo = runs.length ? Math.min(...runs) : null, hi = runs.length ? Math.max(...runs) : null;
  return '宏观体制按今天的修订版数据计算，实时版只保证单边、按发布滞后'
    + (runs.length === ids.length ? `；${ids.map(id => names[id] || id).join('、')}在 1973 年以来只有 ${lo === hi ? lo : `${lo}–${hi}`} 段历史。` : '。');
}

// The valuation-state settings from valuation.json (dt / dd pairs): the engine records P/D on every path
// and does not pull it toward a fair value; the gap is today's distance from the fitted fair value.
function valuationRows(h, val) {
  if (!val) return [];
  const ev = val.settings?.engine_valuation || {};
  const tested = (val.variants || []).filter(x => x.adoptable).length;
  const passed = (val.selection?.eligible || []).length;
  const ok = (x) => typeof x === 'number' && Number.isFinite(x);
  const sg = (x) => { if (!ok(x)) return '—'; const s = x.toFixed(2); return Number(s) === 0 ? '0.00' : `${x > 0 ? '+' : ''}${s}`; };
  const range = val.today?.gap_log_range;
  return [
    h('dt', {}, '估值状态'), h('dd', {}, ev.track ? '逐路径记录 P/D（κ=0，不影响回报）' : '不记录'),
    h('dt', {}, '估值拉力'), h('dd', {}, `${ev.enabled ? '开启' : '关闭'}（回测了 ${tested} 个估值状态变体，通过 ${passed} 个）`),
    h('dt', {}, '今天相对公允值'), h('dd', {}, `对数缺口 ${sg(val.today?.gap_log)}`,
      Array.isArray(range) ? `（换公式 ${sg(range[0])} ~ ${sg(range[1])}）` : ''),
  ];
}

// ---- the missing-factor round (factor_audit.json and the selected model in meta.json; both optional)

const COPYRIGHT = '第三方版权：Moody\'s（Baa 公司债收益率，经 FRED）、Freddie Mac（30 年房贷利率 PMMS、房价指数 FMHPI）、'
  + '密歇根大学（消费者调查与通胀预期）、Robert J. Shiller（CAPE 与美国房价）。数据只在本地构建时下载，不随代码分发。';
const isNum = (x) => typeof x === 'number' && Number.isFinite(x);

// Step 1: the panel. The factor lists come from meta.json: the selected model's factors first, then the other
// macro factors (candidates and simulated paths), then the scenario view variables grouped by concept.
// Returns [text, details] or null (meta.json without a selected model: the fixed sentence is used).
function panelStep(h, ui, meta, fa) {
  const F = Object.fromEntries((meta?.factors || []).map(f => [f.id, f]));
  const spec = meta?.model?.spec || {};
  const inModel = (spec.factors || []).filter(id => F[id]);
  if (!inModel.length) return null;
  const nm = (id) => F[id]?.name || id;
  const extra = Object.entries(spec.asset_factors || {}).filter(([, ids]) => Array.isArray(ids) && ids.length);
  const extraIds = new Set(extra.flatMap(([, ids]) => ids));
  const others = (meta.factors || []).filter(f => !inModel.includes(f.id) && !extraIds.has(f.id) && f.role !== 'audit').map(f => f.name);
  const nAssets = (meta.assets || []).length;
  const text = `1947–至今的月度面板。传导模型的因子（走步检验选出）：${inModel.map(nm).join('、')}`
    + (extra.length ? `；${extra.map(([a, ids]) => `${ui.assetName(a)}另含${ids.map(nm).join('、')}`).join('；')}` : '')
    + `。其他宏观因子（候选和模拟路径）：${others.join('、') || '—'}。${nAssets ? `${nAssets} 类` : '各类'}资产的总回报、价格回报与收入。`
    + '季度数据按发布月对齐，不插值。模型从 1973 年（浮动汇率、黄金自由定价、OPEC 时代）开始估计。';
  // scenario view variables by concept: concept names from meta.json (view_concepts) or the audit's concept table
  const views = (meta.states || []).filter(s => s.concept);
  if (!views.length) return [text];
  const cname = { ...Object.fromEntries((fa?.concepts || []).filter(c => c.id && c.name).map(c => [c.id, c.name])), ...(meta.view_concepts || {}) };
  const order = [...Object.keys(meta.view_concepts || {}), ...views.map(s => s.concept)].filter((c, i, a) => a.indexOf(c) === i);
  const groups = order.map(c => [c, views.filter(s => s.concept === c)]).filter(([, ss]) => ss.length);
  return [text, h('details', {},
    h('summary', { style: { cursor: 'pointer', color: 'var(--accent)', fontSize: '13px' } }, `情景变量（不进传导模型，${views.length} 个，按概念）`),
    h('ul', { class: 'list', style: { fontSize: '13px', marginTop: '6px' } }, groups.map(([c, ss]) => h('li', {},
      cname[c] ? h('span', { class: 'muted' }, `${cname[c]}：`) : null, ss.map(s => `${s.name}${s.late ? '（晚开始）' : ''}`).join('、')))))];
}

// Step 6's first sentences from the selected model: inputs taken from the drawn months as they are, and
// equations fitted on a shorter window.
function transmissionText(ui, meta) {
  const spec = meta?.model?.spec;
  const desc = meta?.model?.describe;
  const head = `资产价格回报 = 走步检验选出的线性因子模型${desc ? `（${desc}）` : '（诊断与走步迭代选出的版本）'} × 模拟的因子变化 + 历史残差（因子解释不了的部分，保留资产之间的联动）。`;
  if (!spec || !Array.isArray(spec.factors)) return head;
  const F = Object.fromEntries((meta.factors || []).map(f => [f.id, f]));
  const own = [];
  if (spec.factors.includes('gz')) own.push('GZ 信用利差');
  if (spec.factors.some(id => F[id]?.interaction)) own.push('期限溢价项');
  for (const [a, ids] of Object.entries(spec.asset_factors || {})) {
    if (Array.isArray(ids) && ids.length) own.push(`${ui.assetName(a)}的${ids.map(id => (F[id]?.name || id).replace(/变动$/, '')).join('、')}`);
  }
  const starts = Object.entries(spec.asset_start || {}).filter(([, ym]) => typeof ym === 'string' && ym)
    .map(([a, ym]) => `${ui.assetName(a)}方程用 ${ym.slice(0, 4)} 年以来的月份估计`);
  // an interaction term switches on with the drawn month's own state (FACTORS_DESIGN §3.3): say what that means
  const inter = spec.factors.map(id => F[id]?.interaction).find(Boolean);
  const STATE = { core_yoy: '核心 CPI 同比', cpi_yoy: 'CPI 同比' };
  const interText = inter && isNum(inter.above)
    ? `期限溢价项只在抽中的历史月份本身处于高通胀时起作用（${inter.lag === 1 ? '上月' : `${inter.lag} 个月前`}${STATE[inter.state] || '通胀'}高于 ${inter.above}%：1973–92 年的大部分月份、2021–23 年）；`
      + '模拟路径的通胀升高不会自动打开它，要让它起作用，阶段视图里要多抽这些月份（例如给核心 CPI 同比或宏观体制的高通胀标志设视图）。'
    : '';
  return head + (own.length ? `${own.join('、')}取自抽中的历史月份本身，不按今天的利差水平缩放` : '')
    + (starts.length ? `${own.length ? '；' : ''}${starts.join('；')}` : '') + (own.length || starts.length ? '。' : '') + interText;
}

// Step 12 (the macro-regime layer's step is 11): the missing-factor audit.
function auditStep(h, fa) {
  const S = fa?.summary || {};
  const counts = isNum(S.n_transmission) && isNum(S.n_predictive)
    ? `研究轮的${isNum(S.families) ? ` ${S.families} 个` : ''}研究家族测试了 ${S.n_transmission} 个同月传导候选和 ${S.n_predictive} 个预测状态（研究轮存档的数字）`
    : '研究轮按家族测试了同月传导候选和预测状态';
  return ['12. 缺失因子审计', `${counts}；只有通过多重检验、反驳检验和走步验证的改动进入模型，其余的作为情景变量或只在审计里报告。`,
    h('a', { href: '#diagnostics?audit' }, '缺失因子审计 →')];
}

function auditLimit(fa) {
  const u = fa?.summary?.unexplained_share_2015_2026;
  return `${isNum(u) ? `约 ${(u * 100).toFixed(0)}% ` : '大部分'}的月度价格波动在样本外解释不了（市场自身的估值与风险偏好、行业新闻、危机月份），`
    + '引擎用抽中月份的残差承载这一部分；GZ 和期限溢价是今天的修订版和模型估计值。';
}

// ---- the quarterly chapter (SIM_DESIGN §0, §1, §3, §4.3, §6; 季度模型，试运行)
// Counts come from the model card (/api/q/card: groups x, u, s, c), the re-rating drift and today's CAPE from the quarterly
// panel (/api/q/valuation); without them the chapter keeps its sentences and leaves the numbers out.
function quarterlyChapter(h, card, ui, meta, qcard, qval) {
  const rows = qcard?.rows || [];
  const n = (g) => rows.filter(r => r.group === g).length || null;
  const nx = n('x'), nu = n('u'), ns = n('s'), nc = n('c');
  const cnt = (k, unit) => (k ? `${k} 个${unit}` : unit);
  const nAssets = (meta?.assets || []).length || null;
  const rer = (qval?.rerating || []).filter(r => typeof r.per_year === 'number');
  const rerText = rer.length ? `${(Math.min(...rer.map(r => r.per_year)) * 100).toFixed(1)}–${(Math.max(...rer.map(r => r.per_year)) * 100).toFixed(1)}%` : '';
  const capeNow = (qval?.today?.states || []).find(x => x.key === 'cape');
  const train = rer.find(r => r.to === '2014Q4');
  const steps = [
    ['Q1. 三层：情景 → 因子 → 资产', '第一层是带概率的情景假说，唯一带判断（预测）的一层；第二层是一个联合模型：情景写了的值精确或在给定标准差内满足，没写的从历史联合分布补全，政策可以写成“干预”（同季的宏观不随它补全，市场价格当季反应）；'
      + '第三层把补全的因子同期映射成资产（每个输入都标明是同期、前定还是假设）。情景错了是“假说误差”，前两层在情景正确时仍然算错是“模拟误差”，两者分开报告。'],
    ['Q2. 季度单元：三个输入', `输入是季初以前的全部数据（${cnt(ns, '期初状态')}，前定）、季前文字的翻译（${cnt(nc, '长期锚')}、预告路径、跨季约束、估值陈述、体制请求，假设）和季内文字的翻译（${cnt(nx, '字段')}和 ${cnt(nu, '政策冲击')}，同期）。`
      + `大模型只翻译两段文字；输出是季末${nAssets ? ` ${nAssets} 个` : ''}资产的联合分布。同一个模型一季一季往前滚，推 10–15 年（主要看 10 年）。`],
    ['Q3. 模型的因子输入是什么，怎样进入模型', `因子输入是变量卡片上的情景输入字段（${cnt(nx, '')}，按货币、财政、需求、供给、价格与产出、金融条件、风险和政策分组）加上 ${cnt(nu, '政策冲击')}；大模型写的是季度情景表：每季每个字段要么空着（没说），要么带一个数值、一个定性幅度或一个范围，并附原文引文。`
      + '模型读入的是 [期初状态, 字段值 × 掩码, 掩码]，掩码标明文字说了哪些字段；在高斯模型里掩码不需要学：说了的分量直接作条件，没说的积分掉。'],
    ['Q4. 神经网络：做了，但输了', `带掩码的小型神经网络（输入同上）在三个十年里都输给结构化的线性高斯模型，只作对照，只有在预注册检验里每个时段都赢才会启用。`
      + (train && nx && nu && nAssets ? `可能的原因：到 ${train.to} 的训练期只有约 ${train.n} 个季度，却要学 ${nx + nu} 个输入、${nAssets} 个输出。` : '可能的原因：训练期的季度太少，输入和输出太多。')],
    ['Q5. 金融条件必填', '把真实的宏观路径原样喂给传导模型，样本外几乎没有解释力；“已知宏观”的能力几乎全部来自信用利差。所以情景每个阶段都要说明金融条件（信用利差的方向）；没说时由模型补全，并标出“补全的信用利差在历史检验里与无条件抽样相当”。写它接近写股权风险溢价，界面会这样标明。'],
    ['Q6. 估值：状态，也是一条腿', `股价 = CAPE 倍数 × CPI × 10 年平均实际盈利（恒等式）：价格回报 = 倍数腿 + 基本面腿。倍数腿几乎承载了全部股价波动，所以写出一条倍数路径就等于写出一条股价路径。`
      + `起点估值对 1–20 个季度的回报没有样本外增益（其中 12–20 个季度每个十年只有 2–3 个独立窗口，只能当个案），任何“拉回均值”都更差：模型不回归；情景可以写“倍数保持”或“重估（温和 / 明显 / 极端）”，那相当于直接假设股价。`
      + (rerText ? `长期回报不外推训练期的倍数抬升（1973 年以来每年约 ${rerText}）。` : '长期回报不外推训练期的倍数抬升。')
      + (capeNow ? `今天的 CAPE ${capeNow.value.toFixed(1)}（${qval.cape_since || '1973Q1'} 以来第 ${((capeNow.percentile || 0) * 100).toFixed(1)} 百分位）。` : '')],
    ['Q7. 怎么检验、何时切换', '季度模型通过预注册的切换关之前，是试运行：内置情景、未来一年页的默认数字、组合与提取仍用月度引擎。切换关要求季度模型在已知因子路径和同一份翻译上都不比月度引擎差（差距的 90% 上界低于预先登记的门槛），且校准不比它差。叙事检验是常设功能：真实大模型只在你看过调用估算并确认之后才运行，测试和构建都用模拟后端。'],
  ];
  return [
    h('div', { class: 'section-title' }, '季度模型（试运行）：情景 → 因子 → 资产'),
    h('div', { class: 'grid cols-2 text' }, steps.map(([t, d]) => card({ title: t }, h('p', { style: { margin: 0 } }, d)))),
    h('p', { class: 'muted', style: { fontSize: '13px', margin: '10px 0 0' } }, '对应的页面：',
      h('a', { href: '#forecast' }, '未来一年：情景条件模拟'), '、', h('a', { href: '#lab' }, '情景实验室（季度情景表）'), '、',
      h('a', { href: '#scenarios' }, '未来情景（每个情景的季度模型一节）'), '。'),
  ];
}
