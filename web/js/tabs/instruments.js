// 标的推演 (INSTRUMENTS_DESIGN §7.1): one ticker on every scenario's paths, its own tab between 情景实验室 and
// 组合评估. The section itself is _tickers.js projection(), the same one 组合评估 shows below the portfolio; here it
// runs with the model's scenario probabilities and the custom scenarios picked on the portfolio pages.
// Deep links: #instruments?TQQQ, #instruments?ticker=MAIN (any case; 金价 / GOLD for the gold asset).
import { shared, parseParam, injectStyle } from './_weights.js';
import { loadCatalog, projection, resolveId, isTicker, TK } from './_tickers.js';

export async function render(root, { api, ui, meta, param, signal, rerender }) {
  injectStyle();
  const { h } = ui;
  const q = parseParam(param);
  const lead = h('p', { class: 'lead' },
    '每个标的是一份配方：在核心模拟的同一批情景路径上，用资产类别回报、宏观路径（利率、信用利差）和期内波动推出它的价格、总回报和现金分配。',
    '预测（带判断的假设）只发生在情景层；这里用模型的情景概率混合各情景，组合评估页下方是同一个推演，并跟随那里调整的概率。');
  root.append(lead);
  const cat = await loadCatalog();
  if (!lead.isConnected) return undefined;
  const assetIds = (meta?.assets || []).map(a => a.id);
  const want = q.kv.ticker || q.tokens[0] || null;
  const id = want ? resolveId(want, assetIds.length ? assetIds : undefined) : null;
  const ticker = id && isTicker(id) ? id : id === 'gold' ? 'GOLD' : null;
  if (want && !ticker) root.append(ui.notice(`链接里的标的「${String(want).slice(0, 20)}」不在标的列表里。`, 'info'));
  if (ticker && !rerender) TK.pj.ticker = ticker;
  if (cat?.error) root.append(ui.notice('无法读取标的目录：标的按内置名单列出。', 'error'));
  const proj = projection({ signal, focus: !!ticker && !rerender, getContext: () => ({ include_custom: shared.include }) });
  root.append(proj.el);
  api.get('scenarios').then((s) => { if (proj.el.isConnected && s && !s.error) proj.setScenarios(s.scenarios || []); }, () => {});
  return undefined;
}
