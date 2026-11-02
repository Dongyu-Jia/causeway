// The switch gate T-S (SIM_DESIGN §4.6; scenariolab/evaluation/switch.py, P9): which engine is the default for the
// 13 built-in scenarios, the 未来一年 tab, 组合评估 and 提取与破产概率, and why. A helper module, not a tab.
//   loadSwitch()             -> Promise<state | null>   GET /api/evaluation/switch, once per page load
//   switchNotice(st, opts)   -> a notice: the default engine, the gate's status, the reasons (each with its numbers),
//                               how to evaluate it when it has not been, and a link to 情景条件检验
//   engineTag(st)            -> '默认' once T-S has passed, else '试运行' (the quarterly stack's label)
//   quarterlyDefault(st)     -> whether the quarterly stack is the default now
//   lastSwitch()             -> the state loadSwitch() last returned on this page (null before)
// Every text comes from the state (Chinese, from the server); nothing here reaches innerHTML.
import * as ui from '../ui.js';
import * as api from '../api.js';

export const STATUS_ZH = { passed: ['已通过', 'good'], failed: ['未通过', 'bad'], not_evaluated: ['尚未评估', 'warn'],
  stale: ['结果过期', 'warn'], invalid: ['结果无效', 'bad'] };

let LAST = null;                 // the last state loaded on this page (renderers that cannot wait read it)

export function loadSwitch() {
  return api.get('evaluation/switch').then((st) => { LAST = st; return st; }, () => null);
}

export const lastSwitch = () => LAST;

export const quarterlyDefault = (st) => !!(st && st.quarterly === true && st.status === 'passed');
export const engineTag = (st) => (quarterlyDefault(st) ? '默认' : '试运行');

function ensureStyle() {
  if (document.getElementById('style-switch-note')) return;
  document.head.append(ui.h('style', { id: 'style-switch-note' }, `
.sw-note { border-left: 3px solid var(--warn); background: var(--panel-2); border-radius: 0 8px 8px 0; padding: 10px 14px;
  font-size: 13px; line-height: 1.6; color: var(--ink-2); margin: 0 0 12px; min-width: 0; overflow-wrap: anywhere; }
.sw-note.on { border-left-color: var(--good); }
.sw-note.bad { border-left-color: var(--bad); }
.sw-head { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
.sw-head b { color: var(--ink); }
.sw-note p { margin: 4px 0 0; max-width: 1080px; }
.sw-note details { margin-top: 6px; }
.sw-note summary { cursor: pointer; color: var(--ink-2); font-size: 12px; }
.sw-note ul { margin: 4px 0 0; padding-left: 18px; }
.sw-note li { margin: 2px 0; }
.sw-note pre { margin: 6px 0 0; white-space: pre-wrap; word-break: break-all; font-size: 12px; }
.sw-note .sw-link { font-size: 12px; }
`));
}

// where: the page's own words for what the default decides here ('组合评估的默认情景集', ...)
export function switchNotice(st, { where = '', compact = false } = {}) {
  ensureStyle();
  const { h } = ui;
  if (!st) {
    return h('div', { class: 'sw-note' }, h('div', { class: 'sw-head' }, h('b', {}, '默认引擎：月度引擎'),
      ui.chip('读不出切换关', 'warn')), h('p', {}, '切换关 T-S 的状态读不出来：默认引擎是月度引擎，季度模型只作试运行。'));
  }
  const [word, kind] = STATUS_ZH[st.status] || STATUS_ZH.not_evaluated;
  const on = quarterlyDefault(st);
  const reasons = Array.isArray(st.reasons) ? st.reasons : [];
  const cmds = Array.isArray(st.commands) ? st.commands : [];
  const kids = [
    h('div', { class: 'sw-head' }, h('b', {}, `${where ? `${where}：` : ''}${st.default_text || (on ? '默认引擎：季度模型' : '默认引擎：月度引擎')}`),
      h('span', {}, '切换关 T-S ', ui.chip(word, kind))),
    compact ? null : h('p', {}, st.text || ''),
  ];
  if (reasons.length) {
    kids.push(h('details', { class: 'sw-why' }, h('summary', {}, on ? '切换关的依据' : `为什么没有切换（${reasons.length} 条）`),
      h('ul', {}, reasons.map(r => h('li', {}, r)))));
  }
  if (!on && !st.evaluated_at && cmds.length) {
    kids.push(h('details', {}, h('summary', {}, '给维护者：怎样评估切换关'), st.command_note ? h('p', {}, st.command_note) : null,
      h('pre', { class: 'code' }, cmds.join('\n'))));
  }
  kids.push(h('div', {}, h('a', { class: 'sw-link', href: '#evaluation?conclusion' }, '在「情景条件检验」里看切换关的全部条件 →')));
  return h('div', { class: `sw-note${on ? ' on' : st.status === 'failed' || st.status === 'invalid' ? ' bad' : ''}` }, kids);
}
