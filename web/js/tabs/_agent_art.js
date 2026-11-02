// Inline SVG art of the 智能体推演 storyboard (a helper module, not a tab): one emblem per kind of actor, drawn in
// code on a shared 120 × 120 view box, each with a face layer whose expression can change; small macro gauges.
// Generic emblems only: no logos, flags, seals, real people or stereotypes. Fills come from CSS custom properties
// (--emb: the actor colour, --emb-soft: its tint), strokes from currentColor (the page ink).
import { actorColor, emblemOf, isNum, clamp } from './_agent_common.js';

const NS = 'http://www.w3.org/2000/svg';
// custom properties (--emb) need setProperty: Object.assign on a style object drops them
export function setStyle(el, obj) {
  for (const [k, v] of Object.entries(obj || {})) {
    if (v === undefined || v === null) continue;
    if (k.startsWith('--')) el.style.setProperty(k, String(v));
    else el.style[k] = v;
  }
  return el;
}
export function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'style' && typeof v === 'object') setStyle(el, v);
    else if (k === 'class') el.setAttribute('class', v);
    else if ((k === 'fill' || k === 'stroke') && String(v).includes('var(')) el.style.setProperty(k, String(v));
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of kids.flat()) {
    if (c === undefined || c === null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// Shapes as [tag, attrs] lists; every shape gets the ink stroke unless it sets its own.
const SOFT = 'var(--emb-soft)', EMB = 'var(--emb)', PAPER = 'var(--emb-paper, #fff)';
const EMBLEMS = {
  columns: [                                           // a central bank: a columned building
    ['path', { d: 'M14 46 L60 16 L106 46 Z', fill: SOFT }],
    ['rect', { x: 16, y: 46, width: 88, height: 9, rx: 1.5, fill: EMB }],
    ...[24, 44, 64, 84].map(x => ['rect', { x, y: 55, width: 12, height: 37, fill: PAPER }]),
    ['rect', { x: 12, y: 92, width: 96, height: 8, rx: 1.5, fill: SOFT }],
    ['rect', { x: 6, y: 100, width: 108, height: 8, rx: 1.5, fill: EMB }],
  ],
  dome: [                                              // a legislature: a dome
    ['path', { d: 'M60 5 V13', 'stroke-width': 2.5 }],
    ['rect', { x: 54, y: 13, width: 12, height: 12, rx: 2, fill: SOFT }],
    ['path', { d: 'M24 64 C24 40 40 25 60 25 C80 25 96 40 96 64 Z', fill: SOFT }],
    ['path', { d: 'M40 33 C36 42 34 52 34 64 M80 33 C84 42 86 52 86 64', fill: 'none', 'stroke-width': 2 }],
    ['rect', { x: 20, y: 64, width: 80, height: 9, rx: 1.5, fill: EMB }],
    ...[26, 38, 50, 64, 76, 88].map(x => ['rect', { x, y: 73, width: 6, height: 18, fill: PAPER, 'stroke-width': 2 }]),
    ['rect', { x: 14, y: 91, width: 92, height: 9, rx: 1.5, fill: SOFT }],
    ['rect', { x: 8, y: 100, width: 104, height: 8, rx: 1.5, fill: EMB }],
  ],
  podium: [                                            // an executive: a generic figure at a seal-free podium
    ['path', { d: 'M32 64 C32 50 45 45 60 45 C75 45 88 50 88 64 Z', fill: EMB }],
    ['circle', { cx: 60, cy: 30, r: 15, fill: '#f1f3f5' }],
    ['path', { d: 'M50 58 L46 48 M70 58 L74 48', 'stroke-width': 2 }],
    ['circle', { cx: 46, cy: 47, r: 2.6, fill: 'currentColor' }],
    ['circle', { cx: 74, cy: 47, r: 2.6, fill: 'currentColor' }],
    ['path', { d: 'M30 62 H90 L84 106 H36 Z', fill: SOFT }],
    ['rect', { x: 25, y: 56, width: 70, height: 8, rx: 2, fill: EMB }],
    ['rect', { x: 47, y: 74, width: 26, height: 18, rx: 3, fill: PAPER, 'stroke-width': 2 }],
  ],
  vault: [                                             // a treasury: a vault door
    ['rect', { x: 9, y: 34, width: 9, height: 14, rx: 2, fill: EMB }],
    ['rect', { x: 9, y: 76, width: 9, height: 14, rx: 2, fill: EMB }],
    ['rect', { x: 16, y: 20, width: 88, height: 86, rx: 10, fill: SOFT }],
    ['circle', { cx: 60, cy: 62, r: 32, fill: PAPER }],
    ['circle', { cx: 60, cy: 62, r: 25, fill: 'none', 'stroke-width': 2, 'stroke-dasharray': '3 5' }],
    ['circle', { cx: 97, cy: 62, r: 4, fill: EMB }],
  ],
  crane: [                                             // a trade partner: a port crane and containers
    ['path', { d: 'M8 105 H112' }],
    ['path', { d: 'M24 105 V27 M36 105 V27', fill: 'none' }],
    ['path', { d: 'M24 41 L36 53 M36 41 L24 53 M24 65 L36 77 M36 65 L24 77 M24 89 L36 101 M36 89 L24 101', 'stroke-width': 2 }],
    ['rect', { x: 18, y: 18, width: 24, height: 10, rx: 2, fill: EMB }],
    ['path', { d: 'M30 22 L52 11 L106 22 Z', fill: 'none', 'stroke-width': 2 }],
    ['path', { d: 'M30 22 H106' }],
    ['rect', { x: 74, y: 20, width: 11, height: 6, rx: 1.5, fill: EMB }],
    ['path', { d: 'M79.5 26 V53', 'stroke-width': 2 }],
    ['rect', { x: 57, y: 53, width: 46, height: 30, rx: 2, fill: SOFT }],
    ['path', { d: 'M63 57 V79 M97 57 V79', 'stroke-width': 1.6 }],
    ['rect', { x: 52, y: 89, width: 54, height: 16, rx: 2, fill: EMB }],
  ],
  derrick: [                                           // an oil producer: a derrick and a drop of oil
    ['path', { d: 'M8 105 H112' }],
    ['path', { d: 'M22 105 L40 23 L58 105', fill: 'none' }],
    ['path', { d: 'M27 84 H53 M31 64 H49 M35 44 H45 M27 84 L49 64 M53 84 L31 64 M31 64 L45 44 M49 64 L35 44', 'stroke-width': 2 }],
    ['rect', { x: 33, y: 16, width: 14, height: 8, rx: 1.5, fill: EMB }],
    ['path', { d: 'M88 42 C88 42 68 66 68 79 C68 91 77 99 88 99 C99 99 108 91 108 79 C108 66 88 42 88 42 Z', fill: SOFT }],
    ['path', { d: 'M76 76 C76 71 79 67 82 64', fill: 'none', stroke: EMB, 'stroke-width': 2.5 }],
  ],
  chip: [                                              // a technology firm: a chip
    ...[38, 49, 60, 71, 82].flatMap(x => [['path', { d: `M${x} 20 V30`, 'stroke-width': 2.5 }], ['path', { d: `M${x} 90 V100`, 'stroke-width': 2.5 }]]),
    ...[38, 49, 60, 71, 82].flatMap(y => [['path', { d: `M20 ${y} H30`, 'stroke-width': 2.5 }], ['path', { d: `M90 ${y} H100`, 'stroke-width': 2.5 }]]),
    ['rect', { x: 30, y: 30, width: 60, height: 60, rx: 8, fill: EMB }],
    ['rect', { x: 39, y: 39, width: 42, height: 42, rx: 5, fill: SOFT }],
    ['circle', { cx: 36, cy: 36, r: 2, fill: PAPER, stroke: 'none' }],
  ],
  voter: [                                             // voters: a ballot box
    ['g', { transform: 'rotate(-8 60 38)' }, [
      ['rect', { x: 46, y: 16, width: 28, height: 38, rx: 2, fill: PAPER }],
      ['path', { d: 'M52 33 L58 39 L68 25', fill: 'none', stroke: EMB, 'stroke-width': 3.5 }],
    ]],
    ['path', { d: 'M18 56 H102 L95 65 H25 Z', fill: EMB }],
    ['rect', { x: 43, y: 53, width: 34, height: 5, rx: 2, fill: 'currentColor' }],
    ['rect', { x: 25, y: 65, width: 70, height: 40, rx: 4, fill: SOFT }],
  ],
  courts: [                                            // courts: a gavel
    ['rect', { x: 16, y: 93, width: 52, height: 12, rx: 3, fill: EMB }],
    ['path', { d: 'M62 52 L94 101', 'stroke-width': 8 }],
    ['path', { d: 'M62 52 L94 101', stroke: SOFT, 'stroke-width': 3.5 }],
    ['rect', { x: 20, y: 24, width: 76, height: 30, rx: 8, fill: SOFT }],
    ['rect', { x: 28, y: 20, width: 9, height: 38, rx: 2, fill: EMB }],
    ['rect', { x: 79, y: 20, width: 9, height: 38, rx: 2, fill: EMB }],
  ],
  ship: [                                              // a conflict: a ship (no flag, no markings)
    ['path', { d: 'M6 102 C16 96 26 108 36 102 C46 96 56 108 66 102 C76 96 86 108 96 102 C104 97 110 102 114 101', fill: 'none', 'stroke-width': 2.5 }],
    ['path', { d: 'M51 50 L57 27 L63 50', fill: 'none', 'stroke-width': 2.5 }],
    ['path', { d: 'M50 29 Q57 21 64 29', fill: 'none', 'stroke-width': 2.5 }],
    ['rect', { x: 40, y: 50, width: 34, height: 20, rx: 2, fill: EMB }],
    ['path', { d: 'M46 58 H68', stroke: PAPER, 'stroke-width': 2 }],
    ['rect', { x: 78, y: 56, width: 10, height: 14, rx: 1.5, fill: EMB }],
    ['path', { d: 'M10 70 H110 L96 95 H24 Z', fill: SOFT }],
  ],
  cloud: [                                             // nature: weather
    ['path', { d: 'M62 80 L54 97 H64 L58 113 L77 90 H67 L73 80 Z', fill: '#f59f00' }],
    ['path', { d: 'M30 80 C17 80 13 65 24 59 C22 45 38 36 48 44 C52 30 75 28 81 43 C95 41 105 53 99 65 C107 71 103 81 92 81 Z', fill: '#f8f9fa' }],
  ],
};
// where the face sits on each emblem (centre between the eyes) and its scale
const FACE_AT = {
  columns: [60, 35, 0.82], dome: [60, 47, 0.95], podium: [60, 31, 0.92], vault: [60, 61, 1.05], crane: [80, 68, 0.78],
  derrick: [88, 81, 0.8], chip: [60, 61, 0.92], voter: [60, 84, 0.95], courts: [58, 39, 0.85], ship: [58, 82, 0.85],
  cloud: [60, 61, 1],
};

function shapes(list) {
  return list.map(([tag, attrs, kids]) => {
    if (tag === 'g') return s('g', attrs, shapes(kids));
    const a = { stroke: 'currentColor', 'stroke-width': 3, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', fill: 'none', ...attrs };
    return s(tag, a);
  });
}

// The sprite: one hidden <svg> holding every emblem as a <symbol>; panels <use> them.
export function sprite() {
  return s('svg', { width: 0, height: 0, 'aria-hidden': 'true', focusable: 'false', style: { position: 'absolute', width: 0, height: 0, overflow: 'hidden' } },
    s('defs', {}, Object.entries(EMBLEMS).map(([id, list]) => s('symbol', { id: `agents-emb-${id}`, viewBox: '0 0 120 120' }, shapes(list)))));
}
export const spriteMarkup = () => sprite().outerHTML;

// expressions: mouth path, brow paths, eye offset
const FACES = {
  neutral: { mouth: 'M-5 1 L5 1', brows: null, eyes: [0, 0] },
  happy: { mouth: 'M-6.5 -0.5 Q0 6.5 6.5 -0.5', brows: null, eyes: [0, 0] },
  stern: { mouth: 'M-5.5 2 L5.5 2', brows: 'M-11 -8 L-4 -5.5 M11 -8 L4 -5.5', eyes: [0, 0.5] },
  worried: { mouth: 'M-5.5 3.5 Q0 -2 5.5 3.5', brows: 'M-11 -5.5 L-4 -8 M11 -5.5 L4 -8', eyes: [0, 0] },
  think: { mouth: 'M-3 2 Q0 4 3 2', brows: 'M-11 -8.5 L-4 -9 M4 -7 L11 -7.5', eyes: [1.6, -1.6] },
};
function rgbMix(hex, t) {
  const n = parseInt(String(hex).slice(1), 16);
  const mix = (c) => Math.round(c + (255 - c) * t);
  return `rgb(${mix((n >> 16) & 255)}, ${mix((n >> 8) & 255)}, ${mix(n & 255)})`;
}
export const softOf = (hex) => rgbMix(hex, 0.74);

// An actor emblem with a face. Returns the <svg> with .setTone(tone) to change the expression.
export function emblem(actor, { size = 72, tone = 'neutral', label, blink = false, kind } = {}) {
  const id = kind || emblemOf(actor);
  const color = actorColor(actor);
  const [fx, fy, fs] = FACE_AT[id] || FACE_AT.columns;
  const mouth = s('path', { d: FACES.neutral.mouth, fill: 'none', stroke: 'currentColor', 'stroke-width': 2.4, 'stroke-linecap': 'round', class: 'agents-mouth' });
  const brows = s('path', { d: '', fill: 'none', stroke: 'currentColor', 'stroke-width': 2.2, 'stroke-linecap': 'round', class: 'agents-brows' });
  const eye = (cx) => {
    return s('ellipse', { cx, cy: 0, rx: 2.5, ry: 3.1, fill: 'currentColor' });
  };
  const eyes = s('g', { class: 'agents-eyes' }, eye(-7), eye(7));
  const face = s('g', { class: 'agents-face', transform: `translate(${fx} ${fy}) scale(${fs})` }, eyes, brows, mouth);
  const svg = s('svg', { viewBox: '0 0 120 120', width: size, height: size, class: `agents-emblem${blink ? ' blink' : ''}`, role: 'img',
    'aria-label': label || '', style: { '--emb': color, '--emb-soft': softOf(color) } },
  label ? s('title', {}, label) : null,
  s('use', { href: `#agents-emb-${id}` }), face);
  svg.setTone = (t) => {
    const f = FACES[t] || FACES.neutral;
    mouth.setAttribute('d', f.mouth);
    brows.setAttribute('d', f.brows || '');
    eyes.setAttribute('transform', `translate(${f.eyes[0]} ${f.eyes[1]})`);
  };
  svg.face = face;
  svg.setTone(tone);
  return svg;
}

// A small semicircle gauge: track, value arc, needle. animateTo(prev -> value) is done by the caller with WAAPI on
// the returned .needle (a <g> rotated around the centre).
export function gauge({ value, min, max, prev, label, text, sub }) {
  const W = 76, cx = 38, cy = 38, r = 30;
  const ang = (v) => (isNum(v) ? clamp((v - min) / (max - min), 0, 1) : 0) * 180 - 90;      // -90 left, +90 right
  const pt = (a) => [cx + r * Math.sin((a * Math.PI) / 180), cy - r * Math.cos((a * Math.PI) / 180)];
  const arc = (a0, a1) => { const [x0, y0] = pt(a0), [x1, y1] = pt(a1); return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`; };
  const a = ang(value);
  const needle = s('g', { class: 'agents-needle', style: { transformOrigin: `${cx}px ${cy}px`, transformBox: 'view-box', transform: `rotate(${a}deg)` } },
    s('path', { d: `M${cx} ${cy + 3} L${cx} ${cy - r + 5}`, stroke: 'currentColor', 'stroke-width': 2.6, 'stroke-linecap': 'round' }));
  const svg = s('svg', { viewBox: `0 0 ${W} 46`, width: W, height: 46, class: 'agents-gauge-svg', 'aria-hidden': 'true' },
    s('path', { d: arc(-90, 90), fill: 'none', stroke: 'var(--line, #e3e6ec)', 'stroke-width': 7, 'stroke-linecap': 'round' }),
    s('path', { d: arc(-90, Math.max(-89.9, a)), fill: 'none', stroke: 'var(--accent, #3b5bdb)', 'stroke-width': 7, 'stroke-linecap': 'round', opacity: 0.85, class: 'agents-gauge-arc' }),
    needle, s('circle', { cx, cy, r: 3.6, fill: 'currentColor' }));
  return { svg, needle, from: ang(isNum(prev) ? prev : value), to: a, label, text, sub };
}

// The three dots of a thought bubble (CSS pulses them).
export function thinkingDots() {
  return s('svg', { viewBox: '0 0 44 14', width: 44, height: 14, class: 'agents-dots', 'aria-hidden': 'true' },
    [7, 22, 37].map((x, i) => s('circle', { cx: x, cy: 7, r: 4, fill: 'currentColor', style: { animationDelay: `${i * 0.18}s` } })));
}
