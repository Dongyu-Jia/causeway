// Macro regimes (regimes.json, optionally regime_forecast.json and the scenario index): what regime
// today is in, how the six labels are defined from three series, how well they date recessions,
// what assets did in each, how regimes change, and why they are not used in the one-year forecast
// or the transmission model. Every number in the text is read from the results at render time.
const INK = '#1d2433', INK3 = '#7a8396';
const SEQ = ['#f8f9fa', '#bac8ff', '#3b5bdb'];                        // probabilities and shares
const RT = '实时（单边、修订后数据）';                                  // before a real-time name or group
const RT_NOTE = '实时版只用截至当月的数据（单边、修订后数据）';          // a sentence in a card subtitle
// Shades for the contraction episode paths, per label (the label colour comes first).
const SHADES = {
  stagflation: ['#b35c00', '#d9480f', '#e67700', '#f08c00', '#ffa94d', '#8a4500'],
  depression: ['#343a40', '#868e96'],
  recession: ['#7048e8', '#9775fa', '#5f3dc4', '#b197fc', '#845ef7', '#d0bfff'],
};
const SERIES_NAME = { cpi_yoy: 'CPI 同比', unrate: '失业率', ip: '工业产出', core_yoy: '核心 CPI 同比',
  cpi: 'CPI', core: '核心 CPI' };
const RULE_NAME = { growth: '增长', severity: '深度', inflation: '通胀', precedence: '优先顺序', realtime: '实时版',
  data: '数据', timing: '时点' };
const HALF_NAME = { '1973_1999': '1973–99 年', '2000_end': '2000 年以后', '1948_1972': '1948–72 年', '1973_end': '1973 年以后' };
const SECTIONS = [
  ['today', '今天'], ['words', '四个词'], ['timeline', '时间线'], ['validation', '对比 NBER'], ['assets', '资产表现'],
  ['episodes', '收缩逐段'], ['factors', '宏观特征'], ['transitions', '转换'], ['forward', '之后一年'],
  ['eras', '各时代'], ['engine', '引擎里的体制'], ['forecast', '一年期预测'], ['conventions', '口径'],
];

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
// fixed decimals, never "-0.0"
const fx = (v, d) => { const s = v.toFixed(d); return Number(s) === 0 ? (0).toFixed(d) : s; };
const sg = (v, d = 1) => (isNum(v) ? `${Number(fx(v, d)) > 0 ? '+' : ''}${fx(v, d)}` : '—');      // signed number
const pc = (v, d = 1) => (isNum(v) ? `${fx(v * 100, d)}%` : '—');                                 // decimal -> %
const pcS = (v, d = 1) => (isNum(v) ? `${sg(v * 100, d)}%` : '—');                                // decimal -> signed %
const ppS = (v, d = 1) => (isNum(v) ? `${sg(v * 100, d)} 个百分点` : '—');                         // decimal difference
const lv = (v, d = 2) => (isNum(v) ? `${fx(v, d)}%` : '—');                                       // a level already in %
const nm = (v, d = 2) => (isNum(v) ? fx(v, d) : '—');
// a permutation p of exactly 0 means no shift was as extreme: "< 1/shifts"
const pv = (p, shifts) => (p === 0 && isNum(shifts) && shifts > 0 ? `< ${fx(1 / shifts, 3)}` : nm(p, 3));
const ci = (c, f = (x) => sg(x * 100, 1)) => (Array.isArray(c) && isNum(c[0]) && isNum(c[1]) ? `[${f(c[0])}, ${f(c[1])}]` : '');
const span = (a, b) => (!b || a === b ? a : `${a}..${b}`);
const yearsSpan = (a, b) => { const y1 = String(a).slice(0, 4), y2 = String(b).slice(0, 4); return y1 === y2 ? y1 : `${y1}–${y2.slice(2)}`; };
function rgba(hex, a) {
  const n = parseInt(String(hex).slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
// an ECharts pattern fill: diagonal strokes 8 px apart (a plain colour where canvas is unavailable)
function hatch(stroke) {
  const c = document.createElement('canvas');
  const g = c.getContext && c.getContext('2d');
  if (!g) return stroke;
  c.width = c.height = 8;
  g.strokeStyle = stroke;
  g.lineWidth = 1.5;
  g.beginPath();
  g.moveTo(-2, 10); g.lineTo(10, -2);              // the tile's diagonal, overdrawn so neighbouring tiles join
  g.moveTo(-2, 2); g.lineTo(2, -2);
  g.moveTo(6, 10); g.lineTo(10, 6);
  g.stroke();
  return { image: c, repeat: 'repeat' };
}
function nextMonth(m) {
  const [y, mo] = String(m).split('-').map(Number);
  if (!y || !mo) return m;
  return mo === 12 ? `${y + 1}-01` : `${y}-${String(mo + 1).padStart(2, '0')}`;
}
// [1973, 1974, 1979, 1980, 1981] -> "1973–74、1979–81"
function yearRanges(ys) {
  const out = [];
  [...ys].sort((a, b) => a - b).forEach((y) => {
    const last = out[out.length - 1];
    if (last && y === last[1] + 1) last[1] = y; else out.push([y, y]);
  });
  return out.map(([a, b]) => (a === b ? `${a}` : `${a}–${String(b).slice(2)}`)).join('、');
}
// runs of a 0/1 series (dates aligned), "YYYY-MM..YYYY-MM", kept when they end at or after `from`
function maskRuns(mask, dates, from = '') {
  const out = [];
  let s = -1;
  (mask || []).forEach((v, i) => {
    if (v && s < 0) s = i;
    if ((!v || i === mask.length - 1) && s >= 0) {
      const e = v ? i : i - 1;
      if (dates[e] >= from) out.push(span(dates[s], dates[e]));
      s = -1;
    }
  });
  return out;
}

const CSS = `
.rg-nav { display: flex; flex-wrap: wrap; gap: 6px; margin: -4px 0 14px; }
.rg-nav a { font-size: 12px; padding: 2px 9px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel); color: var(--ink-2); }
.rg-nav a:hover { border-color: var(--accent); color: var(--accent); text-decoration: none; }
.rg-anchor { scroll-margin-top: 76px; }
.rg-kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(180px, 100%), 1fr)); gap: 10px; margin-bottom: 16px; }
.rg-kpis .kpi { padding: 10px 12px; }
.rg-kpis .kpi-value { font-size: 19px; }
.rg-stack { display: grid; gap: 16px; }
.rg-two { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(520px, 100%), 1fr)); }
.rg-chip { gap: 5px; white-space: nowrap; }
.rg-chip .rg-dot, .rg-legend .rg-dot { width: 9px; height: 9px; border-radius: 3px; display: inline-block; flex-shrink: 0; }
.rg-legend { display: flex; flex-wrap: wrap; gap: 6px 12px; font-size: 12px; color: var(--ink-2); margin-top: 8px; align-items: center; }
.rg-legend span { display: inline-flex; gap: 5px; align-items: center; }
.rg-legend .rg-nber { width: 14px; height: 9px; border: 1px solid #868e96; background: rgba(0,0,0,.08); display: inline-block; }
.rg-thin { cursor: help; border-bottom: 1px dashed var(--ink-3); }
.rg-note { font-size: 12px; color: var(--ink-3); margin: 8px 0 0; }
.rg-text { margin: 0 0 10px; color: var(--ink-2); font-size: 13px; line-height: 1.7; }
.rg-text b { color: var(--ink); }
.rg-list { margin: 6px 0 0; padding-left: 18px; color: var(--ink-2); font-size: 13px; }
.rg-list li { margin: 4px 0; }
.rg-toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
table.rg-grid { border-collapse: separate; border-spacing: 4px; width: 100%; font-size: 13px; }
table.rg-grid th { font-size: 12px; color: var(--ink-3); font-weight: 600; padding: 4px 6px; text-align: center; }
table.rg-grid th.rg-rowhead { text-align: left; white-space: nowrap; }
table.rg-grid td { border-radius: 6px; padding: 9px 6px; text-align: center; font-weight: 600; color: var(--ink); }
.rg-terms { display: grid; gap: 10px; }
.rg-term { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: var(--panel-2); }
.rg-term-head { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 4px; }
.rg-term-word { font-weight: 650; font-size: 15px; }
.rg-term-text { font-size: 13px; color: var(--ink-2); line-height: 1.7; }
.rg-cell small { display: block; font-size: 11px; color: var(--ink-3); }
.rg-mono { font-variant-numeric: tabular-nums; }
.rg-verdict { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
`;

function ensureStyle() {
  let el = document.getElementById('style-regimes');
  if (!el) { el = document.createElement('style'); el.id = 'style-regimes'; document.head.append(el); }
  if (!el.textContent.includes('/* regimes */')) el.textContent += `/* regimes */\n${CSS}`;
}

export async function render(root, { api, charts, ui, meta, param, alive = () => root.isConnected }) {
  const { h, card } = ui;
  ensureStyle();
  root.append(h('p', { class: 'lead' },
    '把每个月归入一种宏观体制，只用三条序列：CPI 同比（通胀轴）、工业产出的方向并用失业率确认（增长轴）、工业产出从高点的跌幅（深度）。',
    '历史月份和每条模拟路径用同一套规则。体制用来描述历史、约束和解读情景；它不进入一年期预测和传导模型——两者都做过样本外检验，没有通过（见本页最后两节）。'));

  // the page's own file: when /api/meta says it is missing, the build hint shows without a request (no 404)
  if (Array.isArray(meta?.results) && !meta.results.includes('regimes')) {
    root.append(ui.notice('宏观体制结果还没有生成。', 'info'),
      ui.buildHint({ message: 'python -m scenariolab.build --only regimes' }));
    return null;
  }
  let R, RF, SC;
  try {
    [R, RF, SC] = await Promise.all([
      api.get('regimes'),
      api.optional('regime_forecast', meta),
      api.optional('scenarios', meta),
    ]);
  } catch (err) {
    if (!alive()) return null;
    if (err.status === 404) root.append(ui.notice('宏观体制结果还没有生成。', 'info'), ui.buildHint(err));
    else root.append(ui.notice(`读取宏观体制结果失败：${err.message}`, 'error'));
    return null;
  }
  if (!alive()) return null;
  // forecast.json: the current forecast's name, and what regime_forecast.json must match to be its test
  const FC = await api.optional('forecast', meta);
  if (!alive()) return null;
  // regime_forecast is an opt-in step, so its file can outlive the forecast it tested: it counts only when it
  // has this build's data month, forecast.json's last origin and the analog share forecast.json uses
  const rfStale = [];
  if (RF) {
    if (RF.data_end !== R.data_end) rfStale.push(`检验的数据截至 ${RF.data_end || '（文件里没有记录）'}，宏观体制结果截至 ${R.data_end}`);
    const fcLast = FC?.origins?.[FC.origins.length - 1]?.date, rfLast = (RF.settings?.origins || []).slice(-1)[0];
    if (FC && fcLast !== rfLast) rfStale.push(`检验的最后一个起点是 ${rfLast || '—'}，现行一年期预测的是 ${fcLast || '—'}`);
    if (FC && isNum(FC.analog_share) && !(isNum(RF.baseline?.share) && Math.abs(RF.baseline.share - FC.analog_share) < 1e-9)) {
      rfStale.push(`检验对照的相似起点集中度是${RF.baseline?.share_label ? `可用起点的 ${RF.baseline.share_label}` : '（文件里没有记录）'}，`
        + `现行一年期预测用的是可用起点的 ${FC.selection?.chosen_label || fx(FC.analog_share, 3)}`);
    }
  }
  const rfOk = !!RF && !rfStale.length;

  const D = R.definitions || {};
  const P = D.params || {};
  const LABELS = D.labels || [];
  const LIDS = LABELS.map(l => l.id);
  const LBL = Object.fromEntries(LABELS.map(l => [l.id, l]));
  const FLAG = Object.fromEntries((D.flags || []).map(f => [f.id, f]));
  const THIN = R.settings?.min_episodes ?? 4;
  const TL = R.timeline || { dates: [], runs: [], runs_rt: [], flags: {}, series: {} };
  const dates = TL.dates || [];
  const lname = (id) => LBL[id]?.name || id;
  const gName = (gid) => {
    const [kind, id] = String(gid).split(':');
    if (kind === 'label') return lname(id);
    if (kind === 'flag') return FLAG[id]?.name || id;
    return '全部月份';
  };
  // the episodes behind a group in a window, for † tooltips
  const groupEpisodes = (gid, from, rt = false) => {
    const [kind, id] = String(gid).split(':');
    if (kind === 'label') return ((rt ? TL.runs_rt : TL.runs) || []).filter(r => r.label === id && r.end >= from).map(r => span(r.start, r.end));
    if (kind === 'flag') return maskRuns(TL.flags?.[id], dates, from);
    return [];
  };
  const thinTitle = (n, eps) => `† 只有 ${n} 段历史${eps && eps.length ? `：${eps.join('、')}` : ''}`;
  const dag = (text, n, eps) => h('span', { class: 'rg-thin', title: thinTitle(n, eps) }, `${text} †`);
  const nameOrDag = (text, thin, n, eps) => (thin ? dag(text, n, eps) : text);
  // a label chip in the regime colour (never the K-line colours)
  const labelChip = (id, text) => {
    const L = LBL[id];
    if (!L) return h('span', { class: 'muted' }, text || '—');
    return h('span', { class: 'chip rg-chip', style: { background: rgba(L.color, 0.18), borderColor: rgba(L.color, 0.7), color: INK } },
      h('i', { class: 'rg-dot', style: { background: L.color } }), text ?? L.name);
  };
  const legend = (extra) => h('div', { class: 'rg-legend' },
    LABELS.map(l => h('span', {}, h('i', { class: 'rg-dot', style: { background: l.color } }), l.term ? `${l.name}（你说的“${l.term}”）` : l.name)), extra);
  // horizontal 100% bars of regime mixes in the label colours; the card carries legend() as wrapping chips
  // (the chart's own legend pages through the six labels on a phone)
  const mixBars = (box, rows, mixes) => {
    const ch = charts.bars(box, {
      x: rows, horizontal: true, fmt: 'pct', wrap: true, valueMax: 1, legend: false,
      series: LABELS.map(l => ({ name: l.name, stack: 'mix', color: l.color, data: mixes.map(m => m?.[l.id] ?? 0) })),
    });
    ch.setOption({ grid: { top: 10 } });
    return ch;
  };

  const sections = {};
  const anchor = (key, title) => (sections[key] = h('div', { class: 'section-title rg-anchor', id: `rg-${key}` }, title));
  const go = (key) => (e) => {
    const el = sections[key];
    if (!el) return;
    e.preventDefault();
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    ui.setRoute('regimes', key, alive);
  };
  const jump = (key, text) => h('a', { href: `#regimes?${key}`, onclick: go(key) }, text);
  // one section failing shows its error and leaves the rest of the page intact
  const block = (fn, name) => {
    const box = h('div', { class: 'rg-stack' });
    root.append(box);
    try { fn(box); } catch (err) {
      console.error(err);
      box.append(ui.notice(`「${name}」这一节渲染失败：${err.message}`, 'error'));
    }
    return box;
  };
  const draw = (box, fn) => {
    try { return fn(); } catch (err) {
      console.error(err);
      charts.unmount(box);
      box.replaceChildren(ui.notice(`图表绘制失败：${err.message}`, 'error'));
      return null;
    }
  };

  root.append(h('nav', { class: 'rg-nav' }, SECTIONS.map(([k, t]) => jump(k, t))));
  const scenStart = SC?.start || SC?.anchor;
  if (scenStart && R.data_end && String(scenStart).slice(0, 7) !== String(R.data_end).slice(0, 7)) {
    root.append(ui.notice(`宏观体制结果的数据截至 ${R.data_end}，情景库从 ${String(scenStart).slice(0, 7)} 出发：两者不是同一次构建，`
      + '请重新运行 python -m scenariolab.build --only regimes。', ''));
  }

  // ---------------------------------------------------------------- A. today
  root.append(anchor('today', '今天处在什么体制'));
  block((box) => {
    const nc = R.nowcast || {};
    const rd = nc.readings || {};
    const chk = Object.fromEntries((nc.checks || []).map(c => [c.id, c]));
    const ipBelow = isNum(rd.ip_below_peak_rt) ? lv(-rd.ip_below_peak_rt) : '—';
    box.append(h('div', { class: 'rg-kpis' },
      ui.kpi(`实时体制（${nc.month}，${nc.known_from} 起可知）`, labelChip(nc.label, nc.name), '单边、修订后数据'),
      ui.kpi('CPI 同比', lv(rd.cpi_yoy), `距 ${P.high_inflation_pct}% 高通胀线 ${sg(chk.high_infl?.distance, 2)} 个百分点`
        + (nc.last_high_infl ? `；上一次在线上：${nc.last_high_infl.month}（${lv(nc.last_high_infl.cpi_yoy)}）` : '')),
      ui.kpi('工业产出离本轮高点', ipBelow, `收缩要先跌到 ${P.ip_swing_pct}%`),
      ui.kpi('失业率（3 个月均值）比本轮低点', `${sg(rd.u_rise_rt, 2)} 个百分点`, `收缩还要升到 ${P.u_rise_pp}`),
      ui.kpi('核心 CPI 同比', lv(rd.core_yoy), `低于 ${P.core_low_pct}% 才算核心通胀过低`)));

    const checks = card({ title: '判断条件', sub: `${nc.month} 的读数，实时版（单边、修订后数据）；距离 = 读数 − 门槛` },
      ui.table([
        { key: 'name', label: '条件' },
        { key: 'rule', label: '规则', wrap: true },
        { key: 'reading', label: '读数', fmt: (v) => nm(v, 2) },
        { key: 'threshold', label: '门槛', fmt: (v) => nm(v, 1) },
        { key: 'distance', label: '距离', fmt: (v) => sg(v, 2) },
        { key: 'on', label: '成立', fmt: (v) => (v ? ui.chip('是', 'warn') : ui.chip('否')) },
      ], nc.checks || [], { stack: true }));

    // the last 36 months: CPI y/y and IP below its peak, non-expansion real-time months shaded
    const n = dates.length, k0 = Math.max(0, n - 36);
    const x = dates.slice(k0);
    const lab = (TL.label_rt || []).slice(k0);
    const areas = [];
    lab.forEach((l, i) => {
      if (!l || l === 'expansion' || !LBL[l]) return;
      const last = areas[areas.length - 1];
      if (last && last.id === l && last.end === i - 1) { last.end = i; return; }
      areas.push({ id: l, start: i, end: i });
    });
    const box36 = ui.chartBox();
    const recent = card({ title: '最近 36 个月', sub: `CPI 同比（左轴，${P.high_inflation_pct}% 线）和工业产出比本轮高点低多少（右轴，${P.ip_swing_pct}% 线）；`
      + '阴影是实时版（单边、修订后数据）不是扩张的月份' }, box36,
      areas.length ? h('div', { class: 'rg-legend' }, [...new Set(areas.map(a => a.id))].map(id => h('span', {}, h('i', { class: 'rg-dot', style: { background: LBL[id].color } }),
        `${lname(id)}：${areas.filter(a => a.id === id).map(a => span(x[a.start], x[a.end])).join('、')}`))) : null);
    box.append(h('div', { class: 'rg-two' }, checks, recent));
    draw(box36, () => {
      const ch = charts.line(box36, {
        x, legend: true, yFmt: 'pp', tooltipFmt: 'pp', yAxis2: { fmt: 'pp' },
        series: [
          { name: 'CPI 同比', data: (TL.series?.cpi_yoy || []).slice(k0), color: '#c2255c', width: 2 },
          { name: '工业产出比本轮高点低', data: (TL.series?.ip_below_peak_rt || []).slice(k0), color: '#495057', yAxisIndex: 1 },
        ],
        markLines: [{ yAxis: P.high_inflation_pct, name: `${P.high_inflation_pct}%` }],
        markAreas: areas.map(a => ({ from: x[a.start], to: x[Math.min(a.end + 1, x.length - 1)], color: rgba(LBL[a.id].color, 0.35) })),
      });
      // both threshold lines stay inside their axes (an axis scaled to the data alone would drop a line the
      // readings have not reached); tops rounded up to a half point
      const top = (line) => (v) => Math.ceil(Math.max(v.max, line * 1.1) * 2) / 2;
      // each line's label sits at the end of its own axis: CPI on the left, IP on the right
      ch.setOption({
        yAxis: [{ max: top(P.high_inflation_pct) }, { min: 0, max: top(P.ip_swing_pct) }],
        series: [{ markLine: { label: { position: 'insideStartTop', formatter: `CPI ${P.high_inflation_pct}%` } } },
          { markLine: { silent: true, symbol: 'none', precision: 10, lineStyle: { color: INK3, type: 'dotted' },
            label: { color: INK3, fontSize: 10, position: 'insideEndTop', formatter: `工业产出 ${P.ip_swing_pct}%` }, data: [{ yAxis: P.ip_swing_pct }] } }],
      });
    });

    // notes: provisional months, interpolations, a trailing gap
    const notes = [];
    const prov = nc.provisional_from || TL.provisional_from;
    if (prov) notes.push(ui.notice(`${prov} 以后的事后标签是暂定的：之后的数据确认拐点后可能改写。`, ''));
    else {
      const ipb = TL.series?.ip_below_peak_rt || [];
      let pk = -1;
      for (let i = ipb.length - 1; i >= 0; i--) if (isNum(ipb[i]) && ipb[i] === 0) { pk = i; break; }
      if (pk >= 0) {
        notes.push(ui.notice(`本轮工业产出${dates[pk] === nc.month ? `在 ${dates[pk]} 创新高` : `的高点在 ${dates[pk]}`}，在这之前所有月份的事后标签已经确定；`
          + `以后工业产出若从这里下跌 ${P.ip_swing_pct}%、失业率上升 ${P.u_rise_pp} 个百分点，${nextMonth(dates[pk])} 以后的月份才会被改写。`, 'info'));
      }
    }
    for (const it of D.interpolations || []) if (it.how) notes.push(ui.notice(it.how, 'info'));
    const miss = Object.entries(nc.missing_after || {}).filter(([, v]) => Array.isArray(v) && v.length);
    if (miss.length) {
      notes.push(ui.notice(`最新月份缺少 ${miss.map(([s, ms]) => `${SERIES_NAME[s] || s}（${ms.join('、')}）`).join('、')}，现况按 ${nc.month} 判断。`, ''));
    }
    if (nc.note) notes.push(h('p', { class: 'rg-note' }, nc.note));
    if (notes.length) box.append(h('div', { class: 'rg-stack', style: { gap: '8px' } }, notes));
  }, '今天');

  // ---------------------------------------------------------------- B. the four words
  root.append(anchor('words', '你问的四个词怎么对应到数据'));
  block((box) => {
    const g = D.grid || { rows: [], cols: [], cells: [] };
    const grid = h('table', { class: 'rg-grid' },
      h('thead', {}, h('tr', {}, h('th', {}, ''), (g.cols || []).map(c => h('th', {}, `CPI 同比 ${String(c).replace(/^CPI\s*/, '')}`)))),
      h('tbody', {}, (g.rows || []).map((r, i) => h('tr', {}, h('th', { class: 'rg-rowhead' }, r),
        (g.cells?.[i] || []).map(id => h('td', { style: { background: rgba(LBL[id]?.color || '#868e96', 0.3), boxShadow: `inset 0 0 0 1px ${rgba(LBL[id]?.color || '#868e96', 0.6)}` } }, lname(id)))))));
    // a rule text that opens with its own short heading ("收缩：…", "实时版：…") gets that heading in bold;
    // the others get the rule's name, so no item reads "实时版：实时版：…"
    const rules = h('ul', { class: 'rg-list' }, Object.entries(D.rule_text || {}).map(([k, t]) => {
      const m = /^([^：，。（）]{1,6})：/.exec(String(t));
      return m ? h('li', {}, h('b', {}, `${m[1]}：`), String(t).slice(m[0].length)) : h('li', {}, h('b', {}, `${RULE_NAME[k] || k}：`), t);
    }));
    const how = card({ title: '判断方法', sub: '增长看工业产出的方向（用失业率确认），深度看工业产出从高点的累计跌幅，通胀看 CPI 同比' },
      h('div', { class: 'table-wrap' }, grid), g.note ? h('p', { class: 'rg-note' }, g.note) : null, rules);

    const terms = card({ title: '四个词', sub: '你用的词（左）在数据里对应哪种体制（右侧色块），以及历史上有多少' },
      h('div', { class: 'rg-terms' }, (D.terms || []).map(t => {
        const labs = t.labels || [];
        const same = labs.length === 1 && lname(labs[0]) === t.term;
        return h('div', { class: 'rg-term' },
          h('div', { class: 'rg-term-head' }, h('span', { class: 'rg-term-word' }, `“${t.term}”`), h('span', { class: 'muted' }, same ? '数据里也叫' : '数据里叫'),
            labs.map(id => labelChip(id))),
          h('div', { class: 'rg-term-text' }, t.text));
      })));
    box.append(h('div', { class: 'rg-two' }, how, terms));

    const C = R.counts || {};
    const cell = (w, ver, id) => {
      const c = C[w]?.[ver]?.labels?.[id];
      if (!c) return '—';
      const txt = `${c.months} 个月，${c.runs} 段（${pc(c.share, 0)}）`;
      const from = R.assets?.windows?.[w]?.start || (w === '1973' ? R.monthdb?.first : `${w}-01`);
      return c.runs < THIN ? dag(txt, c.runs, groupEpisodes(`label:${id}`, from, ver === 'realtime')) : txt;
    };
    box.append(card({ title: '六种体制', sub: `按顺序取第一个成立的规则；1973 年以来 = 引擎的历史月份（${R.monthdb?.first}..${R.monthdb?.last}，${R.monthdb?.months} 个月）；${RT_NOTE}。† = 少于 ${THIN} 段历史，悬停看是哪几段` },
      ui.table([
        { key: 'id', label: '体制', fmt: (v, r) => h('span', {}, labelChip(v), r.term ? h('span', { class: 'muted', style: { fontSize: '12px', marginLeft: '4px' } }, `你说的“${r.term}”`) : null) },
        { key: 'desc', label: '规则', wrap: true },
        { label: '1948 年以来（月，段）', value: (r) => cell('1948', 'expost', r.id) },
        { label: '1973 年以来（月，段）', value: (r) => cell('1973', 'expost', r.id) },
        { label: '实时版 1973 年以来', value: (r) => cell('1973', 'realtime', r.id) },
      ], LABELS, { stack: true }),
      h('p', { class: 'rg-note' }, `实时版（单边、修订后数据）与事后版同月一致的比例：1948 年以来 ${pc(C.agreement_rt_expost?.['1948'], 0)}，1973 年以来 ${pc(C.agreement_rt_expost?.['1973'], 0)}。`
        + (C.runs_expost_1948 ? `1948 年以来事后版共 ${C.runs_expost_1948.runs} 段，其中 ${C.runs_expost_1948.le_2_months} 段不超过 2 个月。` : ''))));

    if ((D.names_changed || []).length) {
      box.append(card({ title: '为什么不叫“金发姑娘”“过热”', sub: '名称按数据里实际是什么来起' },
        h('ul', { class: 'rg-list' }, D.names_changed.map(n => h('li', {}, h('b', {}, `${n.from} → ${n.to}：`), n.why)))));
    }
  }, '四个词');

  // ---------------------------------------------------------------- C. timeline
  root.append(anchor('timeline', '1948 年以来的体制时间线'));
  block((box) => {
    let mode = 'expost', metric = 'cpi';
    const tlBox = ui.chartBox('tall');
    const V = R.validation || {};
    const METRIC = {
      cpi: { data: TL.series?.cpi_yoy, name: 'CPI 同比', line: P.high_inflation_pct },
      ip: { data: TL.series?.ip_below_peak_rt, name: '工业产出比本轮高点低', line: P.ip_swing_pct },
      u: { data: TL.series?.unrate_3m, name: '失业率（3 个月均值）', line: null },
    };
    const idx = new Map(dates.map((d, i) => [d, i]));
    const nextOf = (m) => dates[Math.min((idx.get(m) ?? 0) + 1, dates.length - 1)];
    const redraw = () => draw(tlBox, () => {
      const m = METRIC[metric];
      const runs = (mode === 'rt' ? TL.runs_rt : TL.runs) || [];
      const ch = charts.line(tlBox, {
        x: dates, legend: false, zoom: true, yFmt: 'pp',
        series: [{ name: m.name, data: m.data || [], color: INK, width: 1.4 }],
        markLines: isNum(m.line) ? [{ yAxis: m.line, name: `${m.line}%` }] : undefined,
        markAreas: runs.filter(r => LBL[r.label]).map(r => ({ from: r.start, to: nextOf(r.end), color: rgba(LBL[r.label].color, 0.30) })),
      });
      const extra = [{ name: 'NBER', type: 'line', data: [], silent: true, markArea: { silent: true,
        itemStyle: { color: 'rgba(0, 0, 0, 0.08)', borderColor: '#868e96', borderWidth: 1 },
        data: (TL.nber || []).map(nb => [{ xAxis: nb.first_month || nb.peak }, { xAxis: nb.trough }]) } }];
      const prov = TL.provisional_from;
      if (prov && idx.has(prov)) {
        // hatched: a markArea ignores decals, so the fill is a small repeated canvas of diagonal strokes
        extra.push({ name: '暂定', type: 'line', data: [], silent: true, markArea: { silent: true,
          itemStyle: { color: hatch('rgba(122, 131, 150, 0.55)'), borderColor: INK3, borderWidth: 1, borderType: 'dashed' },
          label: { show: true, color: INK3, fontSize: 10, position: 'insideTop', formatter: '暂定' },
          data: [[{ xAxis: prov }, { xAxis: dates[dates.length - 1] }]] } });
      }
      ch.setOption({
        series: [{}, ...extra],
        tooltip: { trigger: 'axis', confine: true, formatter: (ps) => {
          const i = (Array.isArray(ps) ? ps : [ps]).find(p => Number.isInteger(p?.dataIndex))?.dataIndex;
          if (i === undefined) return '';
          const S = TL.series || {};
          return [`<b>${ui.esc(dates[i])}</b>`,
            `事后：${ui.esc(lname(TL.label?.[i]) || '—')}　实时：${ui.esc(lname(TL.label_rt?.[i]) || '—')}`,
            `CPI 同比 ${ui.esc(lv(S.cpi_yoy?.[i]))}`,
            `工业产出比本轮高点低 ${ui.esc(lv(S.ip_below_peak_rt?.[i]))}`,
            `失业率（3 个月均值）比本轮低点 ${ui.esc(sg(S.u_rise_rt?.[i], 2))} 个百分点`].join('<br/>');
        } },
      });
    });
    const modeSeg = ui.seg([{ value: 'expost', label: '事后' }, { value: 'rt', label: '实时' }], mode, (v) => { mode = v; redraw(); });
    const metricSeg = ui.seg([{ value: 'cpi', label: 'CPI 同比' }, { value: 'ip', label: '工业产出离高点' }, { value: 'u', label: '失业率（3 个月均值）' }], metric, (v) => { metric = v; redraw(); });
    const lag = V.realtime || {};
    box.append(card({
      title: '每个月的体制',
      sub: `事后版把收缩追溯到真正的拐点；实时版只用截至当月的数据（单边、修订后数据），平均晚 ${nm(lag.mean_start_err, 1)} 个月亮、晚 ${nm(lag.mean_end_err, 1)} 个月熄。拖动下方滑块放大。`,
      actions: h('div', { class: 'rg-toolbar' }, modeSeg, metricSeg),
    }, tlBox, legend(h('span', {}, h('i', { class: 'rg-nber' }), 'NBER 衰退（只用于检验，不进入标签）'))));
    redraw();
  }, '时间线');

  // ---------------------------------------------------------------- D. validation
  root.append(anchor('validation', '和 NBER 衰退日期对比'));
  block((box) => {
    const V = R.validation || {};
    const ex = V.expost || {}, rt = V.realtime || {};
    const oos = V.out_of_sample?.loro || {};
    const hd = oos.heldout_detected || {};
    const fa = rt.false_alarms || [];
    const faMonths = fa.reduce((t, [a, b]) => t + (monthsBetween(a, b) + 1), 0);
    box.append(h('div', { class: 'rg-kpis' },
      ui.kpi('事后版平衡准确率（样本内）', nm(ex.balanced_accuracy, 3), `${(V.window || []).join('..')} 的 NBER 衰退月份`),
      ui.kpi('样本外（逐次留出一次衰退）', `${nm(oos.cv_ba_worst_tie, 3)}–${nm(oos.cv_ba_best_tie, 3)}`, `同一批窗口样本内 ${nm(oos.insample_ba_same_windows, 3)}`),
      ui.kpi('检出衰退', `${ex.detected} / ${ex.recessions}`, `样本外 ${hd.all_tied === hd.any_tied ? hd.all_tied : `${hd.all_tied}–${hd.any_tied}`}`),
      ui.kpi('实时版信号晚多少', `起点 ${sg(rt.mean_start_err, 1)} 个月`, `终点 ${sg(rt.mean_end_err, 1)} 个月；实时版：单边、修订后数据`),
      ui.kpi('误报', `实时 ${fa.length} 次`, `${fa.length ? `${fa.map(([a, b]) => span(a, b)).join('、')}，共 ${faMonths} 个月；` : ''}`
        + `事后版 ${(ex.false_alarms || []).length} 次；实时版：单边、修订后数据`)));

    const per = (ex.per_recession || []).map((r, i) => ({ ...r, rt: (rt.per_recession || [])[i] || {} }));
    const runTxt = (r) => (r?.detected ? `${span(r.start, r.end)}` : '未检出');
    const errTxt = (r) => (r?.detected ? `起点 ${sg(r.start_err, 0)} · 终点 ${sg(r.end_err, 0)}` : '');
    box.append(card({ title: '逐次衰退', sub: `NBER 高点→低谷；误差 = 规则的起点（终点）减 NBER 的第一个衰退月（低谷），单位：月。${RT_NOTE}` },
      ui.table([
        { key: 'nber', label: 'NBER' },
        { label: '事后版', value: (r) => h('span', { class: 'rg-cell' }, runTxt(r), h('small', {}, errTxt(r))) },
        { label: '实时版', value: (r) => h('span', { class: 'rg-cell' }, runTxt(r.rt), h('small', {}, errTxt(r.rt))) },
      ], per, { stack: true })));

    box.append(card({ title: '别的判定方法', sub: '同一批 NBER 月份上的平衡准确率；误报按段数和月数；“窗口之后”= 最后一次 NBER 衰退之后还亮着的月份；'
      + '类型为“实时”的方法只用截至当月的数据（单边、修订后数据）' },
      ui.table([
        { key: 'name', label: '方法', wrap: true },
        { key: 'kind', label: '类型', fmt: (v) => (v === 'realtime' ? '实时' : v === 'expost' ? '事后' : v) },
        { key: 'ba', label: '平衡准确率', fmt: (v) => nm(v, 3) },
        { label: '检出', value: (r) => `${r.detected}/${r.recessions}` },
        { key: 'start_err', label: '起点误差（月）', fmt: (v) => sg(v, 1) },
        { key: 'end_err', label: '终点误差（月）', fmt: (v) => sg(v, 1) },
        { label: '误报（段 / 月）', value: (r) => `${r.fa_runs} / ${r.fa_months}` },
        { key: 'months_after_window', label: '窗口之后' },
      ], V.alternatives || [], { stack: true })));

    const rev = V.revisions || {};
    const cases = rev.cases || [];
    const rng = (a, f = (x) => pc(x, 0)) => (Array.isArray(a) && a.length === 3 ? `${f(a[1])} [${f(a[0])}, ${f(a[2])}]` : '—');
    const revCard = card({ title: '数据修订会不会改变结论', sub: `把修订误差加到今天的数据上重算 ${rev.paths} 次；保持 = 与原标签同月一致的比例（中位数 [5%, 95%]），1948 年以来；${RT_NOTE}` },
      ui.table([
        { key: 'name', label: '修订情形', wrap: true },
        { label: '实时标签保持', value: (r) => rng(r.rt_kept_1948) },
        { label: '事后标签保持', value: (r) => rng(r.ex_kept_1948) },
        { label: '12 次全检出（事后 / 实时）', value: (r) => `${pc(r.share_12_of_12_ex, 0)} / ${pc(r.share_12_of_12_rt, 0)}` },
        { label: '2022–24 被判为收缩（事后 / 实时）', value: (r) => `${pc(r.spots?.['2022_24']?.expost, 0)} / ${pc(r.spots?.['2022_24']?.realtime, 0)}` },
      ], cases, { stack: true }));
    const nm2 = V.near_miss_2022_24 || {};
    const ipCases = cases.filter(c => isNum(c.ip_growth_error_sd_pp) && c.ip_growth_error_sd_pp > 0);
    const rangeOf = (key) => {
      const v = ipCases.map(c => c.spots?.['2022_24']?.[key]).filter(isNum);
      return v.length ? (v.length > 1 ? `${pc(Math.min(...v), 0)}–${pc(Math.max(...v), 0)}` : pc(v[0], 0)) : '—';
    };
    const sds = ipCases.map(c => c.ip_growth_error_sd_pp);
    const sahm = nm2.sahm_ge_05 || [];
    const near = card({ title: '2024 年：接近触发', sub: '没有被判为收缩，但离门槛不远' },
      h('p', { class: 'rg-text' },
        `工业产出 ${nm2.ip_peak} 见顶后最多跌 ${lv(nm2.max_decline_pct)}（${nm2.trough}），离 ${nm2.swing_pct ?? P.ip_swing_pct}% 线只差 ${nm(nm2.short_by_pp, 2)} 个百分点`
        + (sahm.length ? `；Sahm 规则在 ${span(sahm[0], sahm[sahm.length - 1])} 已经触发` : '') + '。'
        + (ipCases.length ? `工业产出修订 ${sds.length > 1 ? `${Math.min(...sds)}–${Math.max(...sds)}` : sds[0]} 个百分点时，${rangeOf('expost')}（事后）/ ${rangeOf('realtime')}（实时）的模拟历史会把这段判为收缩。` : '')));
    box.append(h('div', { class: 'rg-two' }, revCard, near));
    box.append(ui.notice(`参数是对着这些 NBER 日期选的，所以样本内的 ${nm(ex.balanced_accuracy, 3)} 偏乐观；NBER 日期只用来打分，不进入任何标签。`
      + (V.out_of_sample ? `参数网格 ${V.out_of_sample.grid_size} 个点里有 ${V.out_of_sample.grid_points_ba_ge_0955} 个的平衡准确率不低于 0.955。` : ''), 'info'));
  }, '对比 NBER');

  // ---------------------------------------------------------------- E. assets in each regime (same month)
  root.append(anchor('assets', '各体制下的资产表现（同月）'));
  block((box) => {
    const A = R.assets || {};
    let win = '1973', metric = 'real';
    const ROWS = [...LIDS.map(id => `label:${id}`), 'flag:contraction', 'flag:high_infl', 'flag:cpi_neg'];
    const heatBox = ui.chartBox('tall');
    const tableBox = h('div');
    const heatCard = card({ title: '年化实际回报（几何）', sub: ' ', actions: h('div', { class: 'rg-toolbar' },
      ui.seg([{ value: '1973', label: '1973 年以来（13 类资产）' }, { value: '1948', label: '1948 年以来（美股、美债、现金）' }], win, (v) => { win = v; redraw(); }),
      ui.seg([{ value: 'real', label: '年化实际回报' }, { value: 'vs', label: '相对其他月份' }], metric, (v) => { metric = v; redraw(); })) },
    ui.wideChart(heatBox, 560), tableBox);
    box.append(heatCard);
    const redraw = () => {
      const W = A.windows?.[win];
      if (!W) { heatBox.replaceChildren(ui.notice('没有这个窗口的结果。', 'info')); return; }
      const rows = ROWS.filter(g => W.groups?.[g]);
      if (metric === 'real') rows.unshift('all');
      const assets = W.assets || [];
      heatCard.querySelector('.card-title').textContent = metric === 'real' ? '年化实际回报（几何）' : '相对其他月份（年化实际回报之差）';
      heatCard.querySelector('.card-sub').textContent = `${W.start}..${W.end}，${W.months} 个月；同月回报。虚线框 = † 少于 ${THIN} 段历史；点线框 = 收缩期`
        + `（包含先跌后涨）。${metric === 'vs' ? '* = 与其他月份的差异在 90% 区间外。单位：%/年' : ''}`;
      const val = (g, a) => {
        const c = W.groups[g]?.assets?.[a];
        return metric === 'real' ? c?.ann_real?.value : c?.vs_rest?.value;
      };
      heatBox.style.height = `${Math.max(300, rows.length * 34 + 90)}px`;
      draw(heatBox, () => charts.heatmap(heatBox, {
        rows: rows.map(g => `${gName(g)}${W.groups[g]?.thin && g !== 'all' ? ' †' : ''}`),
        cols: assets.map(a => ui.assetName(a)),
        values: rows.map(g => assets.map(a => val(g, a))),
        colors: charts.klineDiverging(),
        labelFmt: (v, r, c) => {
          const cell = W.groups[rows[r]]?.assets?.[assets[c]];
          return metric === 'real' ? `${sg(v * 100, 1)}%` : `${sg(v * 100, 1)}${cell?.vs_rest?.survives ? '*' : ''}`;
        },
        cellStyle: (r) => {
          const G = W.groups[rows[r]] || {};
          if (G.thin && rows[r] !== 'all') return { borderColor: INK3, borderWidth: 1, borderType: 'dashed' };
          if (G.contraction_cell) return { borderColor: INK, borderWidth: 1, borderType: 'dotted' };
          return null;
        },
        tip: (r, c) => {
          const g = rows[r], G = W.groups[g] || {}, cell = G.assets?.[assets[c]] || {};
          const lines = [`<b>${ui.esc(gName(g))} × ${ui.esc(ui.assetName(assets[c]))}</b>`,
            `年化实际 ${ui.esc(pcS(cell.ann_real?.value))} ${ui.esc(ci(cell.ann_real?.ci90))}`,
            g === 'all' ? '' : `相对其他月份 ${ui.esc(pcS(cell.vs_rest?.value))} ${ui.esc(ci(cell.vs_rest?.ci90))}${cell.vs_rest?.survives ? ' *' : ''}`,
            `${G.episodes ?? '—'} 段 · ${cell.months ?? G.months ?? '—'} 个月`];
          if (G.thin && g !== 'all') lines.push(ui.esc(thinTitle(G.episodes, G.episode_list)));
          if (G.contraction_cell) lines.push('收缩期里股市先跌后涨，年化数字只适合在资产之间比较，<br/>不代表持有体验——看下面的逐段路径');
          if (g !== 'all') lines.push('* 与其他月份的差异在区间外');
          return lines.filter(Boolean).join('<br/>');
        },
      }));
      // the main assets, with intervals
      const MAIN = ['us_equity', 'treasury_10y', 'gold', 'crude_oil', 'real_estate', 'energy'].filter(a => assets.includes(a));
      const trows = ['all', ...ROWS].filter(g => W.groups?.[g]).map(g => ({ g, G: W.groups[g] }));
      tableBox.replaceChildren(h('div', { class: 'section-title', style: { margin: '14px 0 6px' } }, '主要资产，带 90% 区间'),
        ui.table([
          { label: '体制 / 标志', value: (r) => (r.g === 'all' ? '全部月份' : nameOrDag(gName(r.g), r.G.thin, r.G.episodes, r.G.episode_list)) },
          { label: '月 / 段', value: (r) => `${r.G.months} / ${r.g === 'all' ? '—' : r.G.episodes}` },
          ...MAIN.map(a => ({ label: ui.assetName(a), value: (r) => {
            const c = r.G.assets?.[a];
            if (!c) return '—';
            const v = c.ann_real?.value;
            return h('span', { class: 'rg-cell' }, h('span', { class: ui.signCls(v) }, pcS(v)), h('small', {}, ci(c.ann_real?.ci90)),
              r.g === 'all' ? null : h('small', {}, `Δ ${pcS(c.vs_rest?.value)}${c.vs_rest?.survives ? '*' : ''}`));
          } })),
        ], trows, { stack: true }),
        h('p', { class: 'rg-note' }, `每格：年化实际回报、90% 区间、Δ = 相对其他月份（* = 差异在区间外）。年化 = 月均实际对数回报 × 12 再取指数；区间按整段重抽样。${trows.filter(r => (r.G.survivors || []).length).map(r => `${gName(r.g)}：${r.G.survivors.map(a => ui.assetName(a)).join('、')}`).join('；') ? `与其他月份的差异在区间外的资产——${trows.filter(r => (r.G.survivors || []).length).map(r => `${gName(r.g)}：${r.G.survivors.map(a => ui.assetName(a)).join('、')}`).join('；')}。` : ''}`));
    };
    redraw();

    // inflation months by era: stocks and bonds
    const ERA = A.inflation_by_era || [];
    if (ERA.length) {
      const eraBox = ui.chartBox('short');
      // a block with too few months for a correlation (corr_sb null: fewer months than the page's correlation
      // rule) is drawn but kept out of the sentence
      const judged = ERA.filter(e => isNum(e.corr_sb)), few = ERA.filter(e => !isNum(e.corr_sb));
      const negB = judged.filter(e => isNum(e.bond_real) && e.bond_real < 0).map(e => e.span);
      const posB = judged.filter(e => isNum(e.bond_real) && e.bond_real > 0);
      const negE = judged.filter(e => isNum(e.equity_real) && e.equity_real < 0).length;
      const sentence = `通胀月份的股票实际回报在 ${judged.length} 个时代里有 ${negE} 个为负；债券只在 ${negB.join('、') || '—'} 为负`
        + (posB.length ? `，${posB.map(e => `${e.span} 年通胀月份的债券实际 ${pcS(e.bond_real)}/年`).join('、')}` : '') + '。'
        + (negB.length < judged.length ? '“通胀月份股债都跌”只对部分时代成立。' : '')
        + (few.length ? `${few.map(e => `${e.span} 只有 ${e.months} 个通胀月份`).join('、')}，太少，不计入这句结论。` : '');
      const eraEps = (e) => (TL.runs || []).filter(r => r.label === 'inflation' && r.end >= e.start && r.start <= e.end).map(r => span(r.start, r.end));
      box.append(h('div', { class: 'rg-two' },
        card({ title: '通胀月份的股债：分时代看', sub: `通胀体制的月份，按时代分块的年化实际回报；† = 少于 ${THIN} 段` },
          h('p', { class: 'rg-text' }, sentence), eraBox,
          h('p', { class: 'rg-note' }, ERA.flatMap((e, i) => [i ? '；' : '', nameOrDag(`${e.span}：${e.months} 个月、${e.episodes} 段`, e.thin, e.episodes, eraEps(e))]))),
        corrCard()));
      draw(eraBox, () => charts.bars(eraBox, {
        x: ERA.map(e => `${e.span}\n${e.months} 个月${e.thin ? ' †' : ''}`), fmt: 'pct', wrap: true,
        series: [{ name: ui.assetName('us_equity'), data: ERA.map(e => e.equity_real), color: '#3b5bdb' },
          { name: ui.assetName('treasury_10y'), data: ERA.map(e => e.bond_real), color: '#868e96' }],
      }));
    } else box.append(corrCard());

    function corrCard() {
      const W = A.windows?.['1973'] || { groups: {} };
      const h1 = A.corr_by_half?.['1973_1999'] || {}, h2 = A.corr_by_half?.['2000_end'] || {};
      const rows = [...LIDS.map(id => `label:${id}`), 'flag:contraction', 'flag:high_infl', 'all'].filter(g => W.groups?.[g]?.corr_stock_bond);
      const ids1 = LIDS.filter(id => isNum(h1[id]?.corr));
      const allPos = ids1.length > 0 && ids1.every(id => h1[id].corr > 0);
      const ids2 = LIDS.filter(id => isNum(h2[id]?.corr) && (h2[id].episodes ?? 0) >= THIN);
      const flip = allPos && isNum(h2.expansion?.corr) && h2.expansion.corr < 0;
      const sentence = `1973–99 年${allPos ? '有数据的每种体制里股债都是正相关' : '各体制的股债相关'}（${ids1.map(id => `${lname(id)} ${sg(h1[id].corr, 2)}`).join('、')}）；`
        + `2000 年以后 ${ids2.map(id => `${lname(id)} ${sg(h2[id].corr, 2)}`).join('、')}。${flip ? '体制之间的差别主要是时代差别。' : ''}`;
      const B = A.corr_by_cpi_bucket || {};
      return card({ title: '股债相关性', sub: `美股和 10 年期美债月度总回报的相关系数，1973 年以来；少于 9 个月不报告` },
        h('p', { class: 'rg-text' }, sentence),
        ui.table([
          { label: '体制 / 标志', value: (r) => { const c = W.groups[r].corr_stock_bond; return r === 'all' ? '全部月份' : nameOrDag(gName(r), c.thin, c.episodes, W.groups[r].episode_list); } },
          { label: '相关 [90%]', value: (r) => { const c = W.groups[r].corr_stock_bond; return isNum(c.value) ? h('span', { class: 'rg-cell' }, sg(c.value, 2), h('small', {}, ci(c.ci90, (x) => sg(x, 2)))) : '—'; } },
          { label: '相对其他月份', value: (r) => { const c = W.groups[r].corr_stock_bond?.vs_rest; return c && isNum(c.value) ? `${sg(c.value, 2)}${c.survives ? '*' : ''}` : '—'; } },
          { label: '1973–99', value: (r) => { const id = r.split(':')[1]; return isNum(h1[id]?.corr) ? `${sg(h1[id].corr, 2)}（${h1[id].months}）` : '—'; } },
          { label: '2000 年以后', value: (r) => { const id = r.split(':')[1]; return isNum(h2[id]?.corr) ? `${sg(h2[id].corr, 2)}（${h2[id].months}）` : '—'; } },
        ], rows, { stack: true }),
        (B.buckets || []).length ? h('p', { class: 'rg-note' }, `按${B.by || '上月 CPI 同比'}分组：${B.buckets.map(b => `${b}%：${sg(B.history?.[b]?.corr, 2)}（${B.history?.[b]?.months} 个月）`).join('，')}。`
          + '* = 与其他月份的差异在 90% 区间外；括号里是月数。') : null);
    }
  }, '资产表现');

  // ---------------------------------------------------------------- F. contraction episodes as paths (H1)
  root.append(anchor('episodes', '收缩期的逐段路径'));
  block((box) => {
    const EP = R.episodes || [];
    const ES = R.episodes_summary || {};
    const stag = EP.filter(e => e.label === 'stagflation');
    const worst = stag.filter(e => isNum(e.equity_real_to_low)).sort((a, b) => a.equity_real_to_low - b.equity_real_to_low)[0];
    const lead = ES.stagflation_low_before_trough_1973 || [];
    const after = ES.stagflation_after_low_1973 || {};
    const stagAnn = R.assets?.windows?.['1973']?.groups?.['label:stagflation']?.assets?.us_equity?.ann_real?.value;
    const ww = worst?.worst_window;
    const wwTxt = ww ? (ww.start.endsWith('-01') && ww.end.endsWith('-12') && ww.start.slice(0, 4) === ww.end.slice(0, 4) ? `${ww.start.slice(0, 4)} 全年` : span(ww.start, ww.end)) : '';
    const parts = [];
    if (lead.length === 2) parts.push(`收缩按工业产出低点结束，而股市低点通常更早：1973 年以来每一段滞胀里早 ${lead[0]}–${lead[1]} 个月`);
    if (worst) {
      parts.push(`所以“滞胀月份”里既有下跌也有反弹：${yearsSpan(worst.start, worst.end)} 年实际先跌 ${pc(-worst.equity_real_to_low)} 再涨 ${pc(worst.equity_real_after_low)}`
        + (ww && isNum(ww.value) ? `，${wwTxt}名义 ${pcS(ww.value)}` : ''));
    }
    let sentence = parts.length ? `${parts.join('，')}。` : '';
    if (isNum(after.of)) sentence += `1973 年以来 ${after.of} 个滞胀月份里有 ${after.after} 个在股市低点之后。`;
    if (isNum(stagAnn) && worst && worst.equity_real_to_low < 0 && worst.equity_real_after_low > 0) {
      sentence += `滞胀的平均年化回报（${pcS(stagAnn)}）不是“持有不亏”，而是一次大跌加一次大涨。`;
    }
    const pathBox = ui.chartBox('tall');
    const used = {};
    const series = EP.filter(e => Array.isArray(e.path) && e.path.length).map(e => {
      const k = used[e.label] = (used[e.label] ?? -1) + 1;
      const shades = SHADES[e.label] || [LBL[e.label]?.color || INK3];
      return { name: `${lname(e.label)} ${span(e.start, e.end)}`, data: e.path, color: shades[k % shades.length], width: e.label === 'stagflation' ? 2 : 1.5, ep: e };
    });
    const len = Math.max(0, ...series.map(s => s.data.length));
    box.append(card({ title: '每段收缩里的美股实际指数', sub: '横轴 = 收缩开始后第几个月（0 = 开始前一个月，指数 = 100）；画到工业产出低点为止。颜色：滞胀琥珀、深度衰退石板灰、衰退紫色' },
      h('p', { class: 'rg-text' }, sentence), pathBox,
      h('div', { class: 'rg-legend' }, series.map(s => h('span', {}, h('i', { class: 'rg-dot', style: { background: s.color } }), s.name)))));
    draw(pathBox, () => {
      const ch = charts.line(pathBox, { x: Array.from({ length: len }, (_, i) => String(i)), series, yFmt: (v) => fx(v, 0), legend: false });
      ch.setOption({ grid: { top: 16 }, tooltip: { trigger: 'axis', confine: true, formatter: (ps) => {
        const arr = (Array.isArray(ps) ? ps : [ps]).filter(p => isNum(p?.value));
        if (!arr.length) return '';
        return [`<b>收缩开始后第 ${ui.esc(arr[0].axisValue)} 个月</b>`, ...arr.map(p => {
          const e = series[p.seriesIndex]?.ep;
          return `${p.marker || ''}${ui.esc(p.seriesName)}：<b>${ui.esc(fx(p.value, 1))}</b>${e?.equity_low_month ? `（股市低点 ${ui.esc(e.equity_low_month)}）` : ''}`;
        })].join('<br/>');
      } } });
    });
    box.append(card({ title: '逐段明细', sub: '美股实际回报，从开始前一个月到收缩结束；最差 12 个月是这一段里的名义回报（不足 12 个月时用整段，括号里是月数）' },
      ui.table([
        { key: 'label', label: '体制', fmt: (v) => labelChip(v) },
        { label: '区间', value: (e) => h('span', { class: 'rg-cell' }, span(e.start, e.end), h('small', {}, `${e.months} 个月`)) },
        { key: 'ip_decline_pct', label: '工业产出跌幅', fmt: (v) => lv(isNum(v) ? -v : v, 1) },
        { key: 'cpi_mean', label: 'CPI 同比均值', fmt: (v) => lv(v, 1) },
        { label: '到股市低点', value: (e) => h('span', { class: 'rg-cell' }, h('span', { class: ui.signCls(e.equity_real_to_low) }, pcS(e.equity_real_to_low)), h('small', {}, `低点 ${e.equity_low_month}`)) },
        { label: '低点之后', value: (e) => h('span', { class: 'rg-cell' }, h('span', { class: ui.signCls(e.equity_real_after_low) }, pcS(e.equity_real_after_low)), h('small', {}, `${e.months_after_low} 个月`)) },
        { key: 'equity_real_total', label: '整段', fmt: (v) => h('span', { class: ui.signCls(v) }, pcS(v)) },
        { label: '最差 12 个月（名义）', value: (e) => (e.worst_window ? h('span', { class: 'rg-cell' }, h('span', { class: ui.signCls(e.worst_window.value) }, pcS(e.worst_window.value)),
          h('small', {}, e.worst_window.months < 12 ? `整段 ${e.worst_window.months} 个月` : span(e.worst_window.start, e.worst_window.end))) : '—') },
        { key: 'bond_real_total', label: '10 年美债整段', fmt: (v) => h('span', { class: ui.signCls(v) }, pcS(v)) },
      ], EP, { stack: true })));
  }, '收缩逐段');

  // ---------------------------------------------------------------- G. macro fingerprint
  root.append(anchor('factors', '体制的宏观特征'));
  block((box) => {
    const F = R.factors || {};
    const fm = F.factors_meta || {};
    const cols = Object.keys(fm).filter(f => fm[f].core || f === 'cpi');
    const rows = Object.keys(F.groups || {});
    const fBox = ui.chartBox('tall');
    fBox.style.height = `${Math.max(300, rows.length * 32 + 90)}px`;
    box.append(card({ title: '因子月度变化：z 值', sub: `z = (体制里的月均变化 − ${(F.window || []).join('..')} 的月均) / 标准差；红 = 偏高、蓝 = 偏低；* = 区间不含 0 且至少 ${THIN} 段；虚线框 = †` },
      ui.wideChart(fBox, 620),
      h('p', { class: 'rg-note' }, 'CPI、工业产出和失业率本身就是判定规则的一部分，它们在各体制里的均值部分是定义出来的；其余因子与定义无关。')));
    draw(fBox, () => charts.heatmap(fBox, {
      rows: rows.map(g => `${gName(g)}${F.groups[g].thin ? ' †' : ''}`),
      cols: cols.map(f => fm[f].name),
      values: rows.map(g => cols.map(f => F.groups[g].factors?.[f]?.z)),
      labelFmt: (v, r, c) => `${sg(v, 2)}${F.groups[rows[r]].factors?.[cols[c]]?.sig && !F.groups[rows[r]].thin ? '*' : ''}`,
      cellStyle: (r) => (F.groups[rows[r]].thin ? { borderColor: INK3, borderWidth: 1, borderType: 'dashed' } : null),
      tip: (r, c) => {
        const G = F.groups[rows[r]], x = G.factors?.[cols[c]] || {};
        return [`<b>${ui.esc(gName(rows[r]))} × ${ui.esc(fm[cols[c]].name)}</b>`, `z ${ui.esc(sg(x.z, 2))} ${ui.esc(ci(x.ci90, (y) => sg(y, 2)))}${x.sig && !G.thin ? ' *' : ''}`,
          `${G.months} 个月 · ${G.episodes} 段`, G.thin ? ui.esc(thinTitle(G.episodes, G.episode_list)) : ''].filter(Boolean).join('<br/>');
      },
    }));
    const sm = F.states_meta || {};
    const st = F.table_states || [];
    const ref = F.reference || {};
    const unit = (s, v) => (sm[s]?.unit === '%' || sm[s]?.unit === 'pp' ? lv(v, 2) : nm(v, 2));
    box.append(card({ title: '各体制的宏观状态', sub: '状态变量在体制月份里的平均水平（括号里是 z 值）；第一行是同一窗口所有月份的平均' },
      ui.table([
        { label: '体制 / 标志', value: (g) => (g === '_ref' ? '全部月份' : nameOrDag(gName(g), F.groups[g].thin, F.groups[g].episodes, F.groups[g].episode_list)) },
        ...st.map(s => ({ label: sm[s]?.name || s, value: (g) => {
          if (g === '_ref') return unit(s, ref[`s_${s}`]?.mean);
          const x = F.groups[g].states?.[s];
          return x ? h('span', { class: 'rg-cell' }, unit(s, x.mean), h('small', {}, `z ${sg(x.z, 2)}${x.sig && !F.groups[g].thin ? '*' : ''}`)) : '—';
        } })),
      ], ['_ref', ...rows], { stack: true })));
  }, '宏观特征');

  // ---------------------------------------------------------------- H. transitions
  root.append(anchor('transitions', '体制怎么转换'));
  block((box) => {
    const T = R.transitions || {};
    const T48 = T['1948'] || {}, T73 = T['1973'] || {};
    const dur = T48.durations || {};
    const durCard = card({ title: '每段持续多久', sub: `${T48.start}..${T48.end}，只用起点在窗口内、没有被样本两端截断的段，单位：月` },
      ui.table([
        { label: '体制', value: (id) => nameOrDag(lname(id), dur[id]?.thin, dur[id]?.runs, groupEpisodes(`label:${id}`, T48.start || '')) },
        { label: '完整段数', value: (id) => dur[id]?.complete_runs ?? '—' },
        { label: '平均', value: (id) => nm(dur[id]?.mean, 1) },
        { label: '中位数', value: (id) => nm(dur[id]?.median, 1) },
        { label: '最短', value: (id) => dur[id]?.min ?? '—' },
        { label: '最长', value: (id) => dur[id]?.max ?? '—' },
        { label: '下月不变的概率', value: (id) => pc(T48.monthly?.probs?.[LIDS.indexOf(id)]?.[LIDS.indexOf(id)], 1) },
      ], LIDS, { stack: true }));

    const M = T73.rt_to_expost_12m || {};
    const thinRows = new Set(M.thin_rows || []);
    const hm = ui.chartBox();
    const rowsIds = [...LIDS, '_base'];
    hm.style.height = `${rowsIds.length * 36 + 90}px`;
    const pi = LIDS.indexOf('inflation'), si = LIDS.indexOf('stagflation');
    const hits = T73.inflation_to_stagflation_hits || {};
    const pInfl = T73.p_contraction_12m?.by_label?.inflation, pAll = T73.p_contraction_12m?.all;
    const sentence = `从实时“通胀”出发，12 个月后 ${pc(M.probs?.[pi]?.[si], 1)} 是滞胀（基准 ${pc(T73.base_rate_12m?.stagflation, 1)}）`
      + (hits.hits ? `——但这 ${hits.hits} 次都发生在 ${yearRanges(hits.years || [])} 年` : '') + '。'
      + (pInfl && pAll ? `从实时“通胀”出发 12 个月后处在收缩的比例 ${pc(pInfl.p, 1)}，全部起点 ${pc(pAll.p, 1)}。` : '');
    // at least 480 px wide (it scrolls inside its card on a phone), so every cell keeps its percentage or count
    const hmCard = card({ title: '实时体制 → 12 个月后的事后体制', sub: `${T73.start} 以来的 ${M.pairs} 个起点；每行加起来是 100%；${RT_NOTE}。† 行只有不到 ${THIN} 段历史，格子里写的是次数` },
      h('p', { class: 'rg-text' }, sentence), ui.wideChart(hm, 480));
    box.append(h('div', { class: 'rg-two' }, hmCard, durCard));
    draw(hm, () => charts.heatmap(hm, {
      rows: rowsIds.map(id => (id === '_base' ? '基准（全部起点）' : `${lname(id)}${thinRows.has(id) ? ' †' : ''}（${(M.counts?.[LIDS.indexOf(id)] || []).reduce((t, v) => t + v, 0)}）`)),
      cols: LIDS.map(lname),
      values: rowsIds.map(id => (id === '_base' ? LIDS.map(c => T73.base_rate_12m?.[c]) : M.probs?.[LIDS.indexOf(id)] || [])),
      diverging: false, min: 0, max: 1, colors: SEQ,
      labelFmt: (v, r, c) => (thinRows.has(rowsIds[r]) ? String(M.counts?.[r]?.[c] ?? '') : pc(v, 0)),
      cellStyle: (r) => (thinRows.has(rowsIds[r]) ? { borderColor: INK3, borderWidth: 1, borderType: 'dashed' } : rowsIds[r] === '_base' ? { borderColor: INK, borderWidth: 1 } : null),
      tip: (r, c) => {
        const id = rowsIds[r];
        if (id === '_base') return `基准：12 个月后是${ui.esc(lname(LIDS[c]))}的比例 <b>${ui.esc(pc(T73.base_rate_12m?.[LIDS[c]], 1))}</b>`;
        const n = M.counts?.[r]?.[c];
        return [`<b>实时${ui.esc(lname(id))} → 12 个月后${ui.esc(lname(LIDS[c]))}</b>`, `${ui.esc(pc(M.probs?.[r]?.[c], 1))}（${n} 次）`,
          thinRows.has(id) ? ui.esc(thinTitle(M.row_episodes?.[id], groupEpisodes(`label:${id}`, T73.start || '', true))) : ''].filter(Boolean).join('<br/>');
      },
    }));

    // walk-forward: can the label predict a contraction?
    const WF = T.walk_forward || {};
    const halves = ['all', '1973_1999', '2000_end'];
    const row = (name, get) => ({ name, cells: halves.map(k => get(k)) });
    const rows = [
      row('恰好 12 个月后处在收缩', (k) => ({ ...(WF.exactly_t12?.[k]?.contraction_given_rt_label || {}), origins: WF.exactly_t12?.[k]?.origins })),
      row('12 个月内任何一个月处在收缩', (k) => ({ ...(WF.any_12m?.[k]?.rt_label || {}), origins: WF.any_12m?.[k]?.origins })),
      row('同上，只用实时收缩标志', (k) => ({ ...(WF.any_12m?.[k]?.contraction_rt || {}), origins: WF.any_12m?.[k]?.origins })),
      row('新的收缩（起点时实时未收缩）', (k) => ({ ...(WF.new_contraction?.[k]?.rt_label || {}), origins: WF.new_contraction?.[k]?.origins })),
    ];
    const cellTxt = (c) => (isNum(c?.brier_skill) ? h('span', { class: 'rg-cell' }, sg(c.brier_skill, 3), h('small', {}, `t ${sg(c.t_hac, 2)} · ${c.origins} 个起点`)) : '—');
    const nc = WF.new_contraction || {};
    const nAll = nc.all?.rt_label, n2000 = nc['2000_end']?.rt_label;
    const any = WF.any_12m?.all || {};
    const weak = isNum(nAll?.t_hac) && Math.abs(nAll.t_hac) < 2;
    const concl = `${weak ? '对新的收缩没有预测力' : '对新的收缩'}（Brier 技能 ${sg(nAll?.brier_skill, 2)}，t ${sg(nAll?.t_hac, 1)}；2000 年以后为 ${sg(n2000?.brier_skill, 2)}）；`
      + `“12 个月内”的技能（标签 ${sg(any.rt_label?.brier_skill, 2)}，t ${sg(any.rt_label?.t_hac, 1)}；实时收缩标志 ${sg(any.contraction_rt?.brier_skill, 2)}，t ${sg(any.contraction_rt?.t_hac, 1)}）来自已经在进行的收缩会持续。`;
    box.append(card({ title: '标签能不能预测收缩', sub: `走步检验：每个起点只用当时的数据重算标签，和历史基准率比 Brier 技能（正 = 更好）；HAC t 值；${RT_NOTE}` },
      ui.table([
        { key: 'name', label: '要预测的事', wrap: true },
        ...halves.map((k, i) => ({ label: k === 'all' ? `全部（${WF.exactly_t12?.all?.first}..${WF.exactly_t12?.all?.last}）` : HALF_NAME[k], value: (r) => cellTxt(r.cells[i]) })),
      ], rows, { stack: true }),
      h('p', { class: 'rg-text', style: { marginTop: '10px' } }, concl)));
  }, '转换');

  // ---------------------------------------------------------------- I. forward returns
  root.append(anchor('forward', '体制之后一年的资产回报'));
  block((box) => {
    const FW = R.forward || {};
    const W = FW.windows?.['1973'] || { groups: {} };
    const W48 = FW.windows?.['1948'] || { groups: {} };
    const G = W.groups || {};
    const lag = FW.lag_months ?? 1;
    const [m0, m1] = W.months_ahead || [lag + 1, lag + 12];
    const allEq = G.all?.assets?.us_equity || {};
    const from = allEq.first_origin || '';
    const gids = Object.keys(G).filter(g => g !== 'all');
    const eq = (g) => G[g]?.assets?.us_equity || {};
    const barIds = gids.filter(g => !eq(g).thin && isNum(eq(g).diff));
    const { up, down } = charts.klineColors();
    const fBox = ui.chartBox();
    fBox.style.height = `${Math.max(200, barIds.length * 34 + 60)}px`;
    const gp = W.global_placebo || {};
    const infl = eq('label:inflation');
    const mech = FW.mechanism || {};
    const verdictKnown = rfOk && Array.isArray(RF.verdict?.adopted);
    const sentences = [];
    if (isNum(infl.diff)) {
      sentences.push(`按发布时间（第 t 月的标签在 t+${lag} 月才知道，所以从 t+${m0} 月算起）：${RT}“通胀”月份之后一年，美股实际回报比全部起点平均${infl.diff < 0 ? '低' : '高'} ${fx(Math.abs(infl.diff) * 100, 1)} 个百分点`
        + `（90% 区间 ${ci(infl.ci90)}，置换检验 p ${pv(infl.placebo_p, gp.shifts).startsWith('<') ? '' : '= '}${pv(infl.placebo_p, gp.shifts)}，HAC t ${sg(infl.t_hac, 2)}）。`);
    }
    if (isNum(mech.all_minus_uncond) && isNum(mech.not_minus_uncond) && mech.all_minus_uncond !== 0) {
      const share = 1 - mech.not_minus_uncond / mech.all_minus_uncond;
      sentences.push(`其中约 ${pc(share, 0)} 来自衰退已经开始但还没有被确认的月份（${mech.origins} 个起点里有 ${mech.within_12m_of_contraction} 个 12 个月内出现收缩，之后一年平均 ${pcS(mech.mean_within)}；`
        + `其余 ${mech.origins - mech.within_12m_of_contraction} 个平均 ${pcS(mech.mean_not)}，比全部起点${mech.not_minus_uncond < 0 ? '低' : '高'} ${fx(Math.abs(mech.not_minus_uncond) * 100, 1)} 个百分点）。`);
    }
    if (isNum(gp.p)) {
      sentences.push(`${gp.tests} 项差异合在一起做置换检验 p = ${nm(gp.p, 2)}（超出各自 90% 范围的 ${gp.actual} 项，平移后平均 ${nm(gp.null_mean, 1)} 项）；`
        + `有足够历史段数（≥ ${THIN} 段）的 ${W.eligible_tests} 项里（偶然约 ${nm(W.expected_by_chance, 1)} 项），Holm 校正后${W.holm_discoveries ? `有 ${W.holm_discoveries} 项` : '没有一项'}显著，BH（10%）${(W.bh10_discoveries || []).length} 项。`);
    }
    sentences.push(verdictKnown ? (RF.verdict.adopted.length ? '把体制放进一年期预测的走步检验见下一节。' : '把它放进一年期预测的走步检验没有通过（见下一节）。')
      : RF ? '把它放进一年期预测的走步检验需要按现行预测重新运行（见下一节）。'
        : '把它放进一年期预测的走步检验在可选构建步骤里（见下一节）。');
    const ex48 = W48.groups?.['label:expansion']?.assets?.us_equity?.halves || {};
    const ex = eq('label:expansion');
    const expNote = isNum(ex.diff) && isNum(ex48['1948_1972']?.diff_vs_rest) && Math.sign(ex48['1948_1972'].diff_vs_rest) !== Math.sign(ex.diff)
      ? `扩张：${ppS(ex.diff)}只在 1973 年以后成立，1948–72 年相对其他起点 ${ppS(ex48['1948_1972'].diff_vs_rest)}，方向相反。` : '';
    const lateNote = ['label:stagflation', 'label:depression'].filter(g => G[g]).map(gName).join('、');
    const i48 = W48.groups?.['label:inflation']?.assets?.us_equity;
    const base = W.base?.groups?.['label:inflation']?.assets?.us_equity;
    box.append(card({ title: `${RT}体制之后一年的美股实际回报：相对全部起点`, sub: `第 t+${m0} 到 t+${m1} 个月的复利实际回报减去全部起点的平均（${pcS(allEq.mean)}，${allEq.origins} 个起点，${from}..${allEq.last_origin}）；只画至少 ${THIN} 段的分组` },
      h('div', { class: 'rg-text' }, sentences.map(s => h('p', { style: { margin: '0 0 6px' } }, s))),
      fBox,
      h('ul', { class: 'rg-list' },
        expNote ? h('li', {}, expNote) : null,
        lateNote ? h('li', {}, `${lateNote}：实时信号到来时股市往往已在底部附近。`) : null,
        i48 && isNum(i48.diff) ? h('li', {}, `1948 年以来（只有美股、美债、现金）：通胀之后一年 ${ppS(i48.diff)} ${ci(i48.ci90)}${i48.robust ? '，两半同号' : ''}；联合置换检验 p = ${nm(W48.global_placebo?.p, 3)}。`) : null,
        base && isNum(base.diff) ? h('li', {}, `敏感性（按数据月份，t+${W.base.months_ahead?.[0]}..t+${W.base.months_ahead?.[1]}）：通胀 ${ppS(base.diff)} ${ci(base.ci90)}，联合置换检验 p = ${nm(W.base.global_placebo?.p, 3)}。`) : null)));
    draw(fBox, () => {
      const ch = charts.bars(fBox, {
        x: barIds.map(gName), horizontal: true, fmt: 'pct', legend: false,
        series: [{ name: '相对全部起点', data: barIds.map(g => ({ value: eq(g).diff, itemStyle: { color: eq(g).diff >= 0 ? up : down } })) }],
      });
      ch.setOption({ tooltip: { trigger: 'item', formatter: (p) => {
        const c = eq(barIds[p.dataIndex]);
        const hv = c.halves || {};
        return [`<b>${ui.esc(gName(barIds[p.dataIndex]))}</b>（${c.origins} 个起点，${c.episodes} 段）`,
          `之后一年 ${ui.esc(pcS(c.mean))}，相对全部起点 ${ui.esc(ppS(c.diff))}`, `90% 区间 ${ui.esc(ci(c.ci90))}${c.survives ? ' *' : ''}`,
          `置换检验 p ${ui.esc(pv(c.placebo_p, gp.shifts))} · HAC t ${ui.esc(sg(c.t_hac, 2))}`,
          `相对其他起点：${Object.entries(hv).map(([k, v]) => `${HALF_NAME[k] || k} ${ppS(v?.diff_vs_rest)}`).map(ui.esc).join('，')}`].join('<br/>');
      } } });
    });
    const AS = ['us_equity', 'industrials', 'real_estate', 'treasury_10y'];
    const cellOf = (g, a) => {
      const c = G[g]?.assets?.[a];
      if (!c) return '—';
      if (g === 'all') return h('span', { class: ui.signCls(c.mean) }, pcS(c.mean));
      return h('span', { class: 'rg-cell' }, h('span', { class: ui.signCls(c.diff) }, `${sg(c.diff * 100, 1)}${c.survives ? '*' : ''}`), c.thin ? h('small', {}, `之后一年 ${pcS(c.mean)}`) : h('small', {}, ci(c.ci90)));
    };
    box.append(card({ title: '所有分组 × 四类资产', sub: `相对全部起点的差（个百分点）与 90% 区间；* = 区间不含 0；† 行只给均值，不给 p 值；稳健 = 区间不含 0 且 1973–99 与 2000 年以后同号。第一行是全部起点之后一年的平均实际回报。分组是实时版（单边、修订后数据）` },
      ui.table([
        { label: '实时分组', value: (g) => (g === 'all' ? '全部起点' : nameOrDag(gName(g), eq(g).thin, eq(g).episodes, groupEpisodes(g, from, true))) },
        { label: '起点 / 段', value: (g) => `${eq(g).origins ?? '—'} / ${g === 'all' ? '—' : eq(g).episodes ?? '—'}` },
        ...AS.map(a => ({ label: ui.assetName(a), value: (g) => cellOf(g, a) })),
        { label: '美股：置换 p · HAC t', value: (g) => (g === 'all' || eq(g).thin ? '—' : `${pv(eq(g).placebo_p, gp.shifts)} · ${sg(eq(g).t_hac, 2)}`) },
        { label: '美股：稳健', value: (g) => (g === 'all' || eq(g).thin ? '—' : eq(g).robust ? ui.chip('是', 'accent') : '否') },
      ], ['all', ...gids], { stack: true })));
  }, '之后一年');

  // ---------------------------------------------------------------- J. eras
  root.append(anchor('eras', '各时代的体制构成'));
  block((box) => {
    const ER = R.eras || [];
    let mode = 'expost';
    const eBox = ui.chartBox();
    eBox.style.height = `${Math.max(260, ER.length * 34 + 70)}px`;
    // the legend is HTML chips that wrap (the chart's own legend pages through the six labels on a phone)
    // "name\nyears": two short lines, so a phone never breaks a label inside its parentheses
    const redraw = () => draw(eBox, () => mixBars(eBox, ER.map(e => (/（.*）$/.test(e.name) ? e.name.replace(/（.*）$/, '')
      : `${e.name}\n${String(e.start).slice(0, 4)}–${String(e.end).slice(2, 4)}`)), ER.map(e => e[mode])));
    const pre = ER.find(e => /（/.test(e.name));
    box.append(card({ title: '每个时代里各体制的月份占比', sub: `历史时代画像页的时代划分；${pre ? `${pre.name.replace(/^.*（|）$/g, '')}：${pre.name.replace(/（.*）$/, '')}。` : ''}${RT_NOTE}`,
      actions: ui.seg([{ value: 'expost', label: '事后' }, { value: 'rt', label: '实时' }], mode, (v) => { mode = v === 'rt' ? 'realtime' : 'expost'; redraw(); }) },
    legend(), eBox, h('p', { class: 'rg-note' }, h('a', { href: '#eras' }, '历史时代画像 →'))));
    redraw();
  }, '各时代');

  // ---------------------------------------------------------------- K. the engine
  root.append(anchor('engine', '引擎里的体制'));
  block((box) => {
    const E = R.engine || {};
    const ref = E.reference || {}, hist = E.history || {};
    const RW = ref.windows || {};
    const ce = ref.contraction_episodes || {}, hce = hist.contraction_episodes || {};
    const longest = E.studies?.longest_contraction_months ?? 18;
    box.append(h('div', { class: 'rg-kpis' },
      ui.kpi(`模拟月份 CPI 同比 > ${P.high_inflation_pct}%`, pc(ref.cpi_gt4_share, 1), `历史 ${pc(hist.cpi_gt4_share, 1)}；${ref.window_months?.m25_180 || ''}`),
      ui.kpi('每 15 年的收缩次数', nm(ce.per_path, 2), `历史 ${nm(hce.per_15y, 2)}（${hce.episodes} 次）`),
      ui.kpi(`超过 ${longest} 个月的收缩`, pc(ce.share_longer_than_18m, 1), `历史 ${hce.episodes} 次里 ${hce.longer_than_18m} 次`),
      ui.kpi('深度收缩', pc(ce.share_severe, 1), `历史 ${hce.episodes} 次里 ${hce.severe} 次`)));

    const S = E.support || {};
    const sids = Object.keys(S);
    box.append(card({ title: '一个 12 个月的阶段最多能抽到多少', sub: '熵池化在历史月份上加权；“最多能抽到”= 要求 100% 时实际抽中的占比；视图打折 λ < 1 = 视图只按这个比例执行。'
      + '† = 这种体制本身的历史太薄（段数太少，或按抽样权重算的有效段数太少）。引擎自己的稀薄检查看的是全部抽中月份；这里的“历史段数”看的是体制本身' },
      ui.table([
        { label: '体制 / 标志', value: (id) => nameOrDag(S[id].name || id, !!S[id].thin, S[id].runs, (S[id].episodes || []).map(e => e.span ?? e)) },
        { label: '历史月数 / 段', value: (id) => `${S[id].months} / ${S[id].runs}` },
        { label: '最多能抽到', value: (id) => pc(S[id].max_delivered, 0) },
        { label: '视图打折 λ', value: (id) => nm(S[id].shrink, 2) },
        { label: '有效月份 ENS', value: (id) => nm(S[id].ens, 0) },
        { label: '有效历史段数（全部 / 体制本身）', value: (id) => `${nm(S[id].engine_episodes, 1)} / ${nm(S[id].regime_effective_episodes, 2)}` },
      ], sids, { stack: true })));

    const mixRows = [
      ['1973 年以来的历史', hist.shares],
      [`引擎无条件 · ${(ref.window_months?.m25_180 || '第 25–180 个月').replace(/（.*）$/, '')}`, RW.m25_180?.simulated],
      ['引擎无条件 · 第 1 年', RW.year1?.simulated],
      SC?.regimes?.year1_weighted ? [`${(SC.scenarios || []).length} 个情景 · 第 1 年（按概率加权）`, SC.regimes.year1_weighted] : null,
    ].filter(r => r && r[1]);
    const mBox = ui.chartBox('short');
    mBox.style.height = `${mixRows.length * 40 + 50}px`;
    const CM = E.comovement || {};
    // the history column's thin marks: the same-month table's correlations (assets, 1973 window)
    const G73 = R.assets?.windows?.['1973']?.groups || {};
    const cmRows = LIDS.map(id => ({ id, h: CM.history_by_label?.[id], d: CM.reference_by_drawn_label?.[id], s: CM.reference_by_sim_label?.[id],
      g: G73[`label:${id}`] || {} }));
    const corrTxt = (c) => (c && isNum(c.corr) ? `${sg(c.corr, 2)}` : '—');
    const hs = CM.history_by_label?.stagflation, ds = CM.reference_by_drawn_label?.stagflation;
    box.append(h('div', { class: 'rg-two' },
      card({ title: '体制构成：历史 vs 引擎', sub: `引擎无条件 = ${ref.n_paths} 条路径的无条件模拟（种子 ${ref.seed}），按同一套规则贴标签` },
        legend(), mBox, h('p', { class: 'rg-note' }, `模拟的 CPI 同比是从今天的水平累加出来的，所以模拟路径里的“通胀”比抽中的历史月份多：引擎无条件模拟的通胀月份有 ${pc(E.simulated_inflation_from_expansion_share, 0)} 抽自历史上的扩张月份。`)),
      card({ title: '股债相关性：历史 vs 引擎', sub: `美股和 10 年期美债月度回报的相关系数；引擎按抽中月份本身的标签和按模拟路径的标签各算一次；† = 历史少于 ${THIN} 段，括号里是月数` },
        ui.table([
          { label: '体制', value: (r) => { const c = r.g.corr_stock_bond || {}; return c.thin ? h('span', {}, labelChip(r.id), dag('', c.episodes, r.g.episode_list)) : labelChip(r.id); } },
          { label: '历史（1973 年以来）', value: (r) => `${corrTxt(r.h)}${r.h ? `（${r.h.months}）` : ''}` },
          { label: '引擎 · 按抽中月份', value: (r) => corrTxt(r.d) },
          { label: '引擎 · 按模拟标签', value: (r) => corrTxt(r.s) },
        ], cmRows, { stack: true }),
        h('p', { class: 'rg-note' }, `按上月 CPI 同比分组的相关系数，最高组减最低组：历史 ${sg(CM.history_bucket_spread, 2)}，引擎 ${sg(CM.reference_bucket_spread, 2)}。`),
        h('p', { class: 'rg-text', style: { marginTop: '8px' } }, `引擎按今天的利率水平缩放利率变化、只沿用部分历史残差，这两步是在回测里选出来的，代价是体制之间的股债联动被冲淡`
          + `（抽中的滞胀月份 ${corrTxt(ds)}，历史 ${corrTxt(hs)}）。这里只报告，不为了这个指标调参数。`))));
    draw(mBox, () => mixBars(mBox, mixRows.map(r => r[0]), mixRows.map(r => r[1])));

    const ST = E.studies || {};
    const DV = ST.drawn_vs_simulated || [];
    const durBox = ui.chartBox('short');
    const DU = ST.duration || [];
    box.append(h('div', { class: 'rg-two' },
      card({ title: '要求的体制，模拟出来是多少', sub: `单阶段情景，${ST.n_paths} 条路径（种子 ${ST.seed}）；抽中 = 抽到的历史月份本身的标签，模拟 = 模拟路径按规则贴的标签；占比都按所看的体制计；λ < 1 = 视图被打折执行` },
        ui.table([
          // a recipe whose name does not name the regime it is read for says which one (CPI −1%/yr + 收缩 0 -> 物价负增长)
          { label: '要求', wrap: true, value: (r) => {
            const tn = String(LBL[r.target]?.name || S[r.target]?.name || FLAG[r.target]?.name || r.target).replace(/（.*）$/, '');
            return r.target && !String(r.name).includes(tn) ? h('span', { class: 'rg-cell' }, r.name, h('small', {}, `看${tn}的占比`)) : r.name;
          } },
          { key: 'months', label: '阶段月数' },
          { key: 'drawn_share', label: '抽中', fmt: (v) => pc(v, 0) },
          { key: 'simulated_share', label: '模拟', fmt: (v) => pc(v, 0) },
          { key: 'lambda', label: 'λ', fmt: (v) => nm(v, 2) },
          { key: 'ip_change_p50', label: '工业产出变化中位数', fmt: (v) => lv(v, 1) },
          { key: 'unrate_end_p50', label: '期末失业率中位数', fmt: (v) => lv(v, 1) },
        ], DV, { stack: true })),
      card({ title: '收缩要求得越久，路径跌得越深', sub: `要求整段都是收缩（种子 ${ST.duration_seed}）：工业产出跌幅超出历史最深一次的路径占比；虚线 = ${longest} 个月（历史上最长的一次收缩）` }, durBox)));
    // months on a value axis, so 6, 12, 18, 24, 36 and 48 sit at their true distances
    draw(durBox, () => charts.line(durBox, {
      x: DU.map(d => d.expected_contraction_months), xValue: true, xFmt: (v) => `${v} 个月`, legend: false, yFmt: 'pct',
      series: [{ name: '超出历史最深跌幅的路径', data: DU.map(d => d.paths_ip_fall_beyond_hist), color: '#7048e8', width: 2, symbol: 'circle' }],
      markLines: [{ xAxis: longest, name: `${longest} 个月` }],
    }));
    const CX = ST.conflict_example;
    if (CX) {
      box.append(ui.notice(`要求和同一阶段的其他观点打架时，所有视图会被打折执行（λ < 1）：「${CX.spec_name}」的「${CX.phase_name}」阶段再要求${Object.entries(CX.regime || {}).map(([k, v]) => `${lname(k)} ${pc(v, 0)}`).join('、')}，`
        + `单独要求时 λ = ${nm(CX.lambda_alone, 2)}，和这个阶段原有的观点放在一起 λ = ${nm(CX.lambda, 2)}，实际抽中 ${pc(CX.drawn_share, 0)}；`
        + `去掉一个原有观点后的 λ：${(CX.leave_one_out || []).map(l => `${ui.varName(l.var)} ${nm(l.lambda_without, 2)}`).join('、')}。`, 'info'));
    }
  }, '引擎里的体制');

  // ---------------------------------------------------------------- L. the one-year forecast
  root.append(anchor('forecast', '一年期预测里用不用体制'));
  block((box) => {
    // the current forecast's name comes from forecast.json; the test file's own name only when it tested it
    const curName = FC?.selection?.baseline_name || '';
    const baseName = rfOk ? (RF.baseline?.name || curName) : curName;
    const tv = R.transmission?.walk_forward_verdict;
    const trans = tv?.oos_r2_delta?.length
      ? h('p', { class: 'rg-text' }, `传导模型不加通胀状态交互项：走步检验里样本外 R² 变化 ${tv.oos_r2_delta.map(d => `${d.period} ${sg(d.delta, 3)}`).join(' / ')}（`, h('a', { href: '#diagnostics' }, '诊断与迭代页'), '）。')
      : (R.model_use?.transmission?.reason ? h('p', { class: 'rg-text' }, R.model_use.transmission.reason) : null);
    const cmd = h('code', { class: 'mono' }, 'python -m scenariolab.build --only regime_forecast');
    if (!RF) {
      box.append(card({ title: '体制特征的样本外检验', sub: baseName ? `对照：${baseName}` : '' },
        h('p', { class: 'rg-text' }, '这项检验在可选构建步骤里：', cmd, '（约 2 分钟）。一年期预测按现行设定运行，不使用宏观体制。'), trans));
      return;
    }
    const adopted = RF.verdict?.adopted || [];
    const fams = RF.families || [];
    const best = fams.filter(f => isNum(f.risk_crps_rel?.test?.mean)).sort((a, b) => a.risk_crps_rel.test.mean - b.risk_crps_rel.test.mean)[0];
    const fr = best?.fragility_seed23;
    const seeds = RF.settings?.seed_bases || [];
    const relCls = (v) => (isNum(v) ? ui.goodCls(v < 0) : '');
    const relTxt = (x) => (isNum(x?.mean) ? h('span', { class: 'rg-cell' }, h('span', { class: relCls(x.mean) }, pcS(x.mean, 2)), h('small', {}, `${pcS(x.min, 2)}..${pcS(x.max, 2)}`)) : '—');
    const g = RF.guard_tests || {};
    // the largest test-half gain, with its seed-averaged t and the forecast module's claim rule
    // (settings.claim): within the noise unless beyond min_skill with |t| of at least min_t
    const claim = RF.settings?.claim;
    const tT = best?.t?.test;
    const gainOk = !!claim && isNum(tT) && best.risk_crps_rel.test.mean < -claim.min_skill && tT <= -claim.min_t;
    const tNote = isNum(tT) ? `（t = ${sg(tT, 1)}${gainOk ? '' : '，在随机误差之内'}）` : (gainOk ? '' : '（在随机误差之内）');
    const stale = rfStale.length
      ? ui.notice(h('span', {}, `这份检验不是现行一年期预测的：${rfStale.join('；')}。下面的数字和结论属于当时的预测，只作参考`
        + `${RF.baseline?.name ? `；当时的对照是${RF.baseline.name}` : ''}。按现行预测重新检验：`, cmd, '（约 2 分钟）。'), '') : null;
    box.append(card({ title: '体制特征的样本外检验', sub: `${RF.question || ''} 对照：${baseName || RF.baseline?.name || '—'}` },
      stale,
      h('div', { class: 'rg-verdict' }, rfOk ? null : ui.chip('旧结果', 'warn'),
        adopted.length ? ui.chip(`采用：${adopted.join('、')}`, 'good') : ui.chip('未采用', 'bad'), h('span', { class: 'rg-text', style: { margin: 0 } }, RF.verdict?.text || '')),
      ui.table([
        { key: 'name', label: '做法', wrap: true, minWidth: 170 },
        { label: '选中的设定', value: (f) => f.selected_text?.[String(seeds[0])] || '—', wrap: true, minWidth: 110 },
        { label: '验证期 CRPS（相对现行）', value: (f) => relTxt(f.risk_crps_rel?.val) },
        { label: '检验期 CRPS（相对现行）', value: (f) => relTxt(f.risk_crps_rel?.test) },
        { key: 'rank_ic_delta', label: '排序 IC 变化', fmt: (v) => sg(v, 3) },
        { label: '90% 覆盖率', value: (f) => h('span', { class: 'rg-cell' }, pc(f.coverage90, 1), h('small', {}, `现行 ${pc(f.coverage90_current, 1)}`)) },
        { label: `两半都更好的种子（共 ${seeds.length} 组）`, value: (f) => `${f.better_both_halves_seeds} 组` },
        { key: 'passes', label: '结论', fmt: (v) => (v ? ui.chip('达到', 'good') : ui.chip('未达到')) },
      ], fams, { stack: true }),
      h('p', { class: 'rg-note' }, `CRPS 是风险资产的连续分级概率评分，越低越好；相对现行 = 负数更好，小字是 ${seeds.length} 组随机种子的范围；选中的设定是验证期在种子 ${seeds[0]} 下选出的。验证期 = ${RF.settings?.split} 及以前的起点。设定：${(RF.settings?.share_labels || []).map(s => `可用起点的 ${s}`).join('、')}；λ ${(RF.settings?.lambdas || []).join('、')}。`),
      best && fr && best.risk_crps_rel.test.mean < 0 ? h('p', { class: 'rg-text' }, `检验期降得最多的是「${best.name}」：检验期 ${pcS(best.risk_crps_rel.test.mean, 2)}${tNote}，验证期 ${pcS(best.risk_crps_rel.val.mean, 2)}；`
        + `随机种子 ${seeds[0]} 下改进最多的是 ${(fr.best_origins || []).join('、')}，去掉这 ${(fr.best_origins || []).length} 个起点后检验期为 ${pcS(fr.test_without_best_rel, 2)}。`) : null,
      h('p', { class: 'rg-note' }, `防前视检查：截断到起点的数据重算标签 ${g.truncation?.months_checked} 个月${g.truncation?.pass ? '全部一致' : '有不一致'}；`
        + `打乱起点之后的数据，${g.scrambled_future?.origins} 个起点的权重${g.scrambled_future?.pass ? '都不变' : '有变化'}；`
        + `起点的实时体制（单边、修订后数据）：${Object.entries(RF.labels_at_origins || {}).map(([k, v]) => `${k} ${v}`).join('、')}。`),
      trans));
  }, '一年期预测');

  // ---------------------------------------------------------------- M. conventions
  root.append(anchor('conventions', '口径说明'));
  block((box) => {
    const CV = D.conventions || {};
    box.append(card({ title: '口径', sub: '本页所有数字共用的定义；页面上的回报和占比都以百分数显示' },
      h('ul', { class: 'rg-list' }, Object.values(CV).map(c => h('li', {}, h('b', {}, `${c.name}：`), c.text))),
      (R.notes || []).length ? h('ul', { class: 'rg-list' }, R.notes.map(n => h('li', {}, n))) : null,
      h('p', { class: 'rg-note' }, `数据截至 ${R.data_end}；引擎的历史月份 ${R.monthdb?.first}..${R.monthdb?.last}（${R.monthdb?.months} 个月）；自助法 ${R.settings?.bootstrap} 次。`)));
  }, '口径');

  return param && sections[param] ? sections[param] : null;
}

// whole months from a to b ("YYYY-MM")
function monthsBetween(a, b) {
  const [y1, m1] = String(a).split('-').map(Number), [y2, m2] = String(b).split('-').map(Number);
  return (y2 - y1) * 12 + (m2 - m1);
}
