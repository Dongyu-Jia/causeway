// Data access of the 智能体推演 tab (a helper module, not a tab).
//
// A "source" is a frozen snapshot (data/agents/snapshots/<id>) or a saved run (data/agents/runs/<id>): both are read
// through the same explorer endpoints (levels, nodes, branches, expand, jobs). Level files are large (a level of a
// beam-30 snapshot is about 1–2.6 MB), so they are fetched without the page cache, turned into compact skeletons and
// dropped; full records and node details are fetched one node at a time and kept in a small LRU.
import { isNum } from './_agent_common.js';

// ---------------------------------------------------------------- requests
export async function request(method, path, body, { signal } = {}) {
  const res = await fetch(`/api/${path}`, {
    method, signal, headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  if (!res.ok) {
    const d = data?.detail;
    const err = new Error(typeof d === 'string' ? d : d ? JSON.stringify(d) : `${res.status} ${res.statusText}`);
    err.status = res.status;
    err.body = data;
    throw err;
  }
  return data;
}
export const get = (path, opts) => request('GET', path, undefined, opts);
export const post = (path, body = {}, opts) => request('POST', path, body, opts);
export const put = (path, body, opts) => request('PUT', path, body, opts);
export const patch = (path, body, opts) => request('PATCH', path, body, opts);
export const del = (path, opts) => request('DELETE', path, undefined, opts);

// "无法连接本地服务" for a network failure, the server's own message otherwise
export function errorText(err) {
  if (!err) return '未知错误';
  if (err.name === 'AbortError') return '已取消';
  if (err instanceof TypeError || /Failed to fetch|NetworkError|Load failed/.test(String(err.message))) {
    return '无法连接本地服务：请确认 python -m scenariolab.server 仍在运行，然后刷新本页。';
  }
  return String(err.message || err);
}

// ---------------------------------------------------------------- skeletons
const SKEL_ASSETS = ['us_equity', 'nasdaq', 'treasury_10y', 'gold', 'crude_oil', 'real_estate', 'cash'];

// The parts of a record the overview, the filters, the outline and the comparison need (about 1.5 KB).
export function skeleton(r, source = 'snapshot') {
  const s = r.summary || {};
  const assets = {};
  for (const a of SKEL_ASSETS) {
    const v = s.assets?.[a];
    if (v) assets[a] = { q: Array.isArray(v.q) ? v.q[1] : null, c: Array.isArray(v.cum) ? v.cum : null };
  }
  const ov = r.overlay && typeof r.overlay === 'object' ? r.overlay : null;
  const kids = (r.children || []).map(c => ({ key: c.key, status: c.status, pc: c.p_cond, own: c.own_mass, label: c.label,
    branch: c.branch, kind: c.kind, reason: c.reason, into: c.absorbed_into, overlay: c.overlay || null }));
  const okids = (r.overlay_children || []).map(c => ({ key: c.key, status: c.status, pc: c.p_cond, own: c.own_mass,
    label: c.label, branch: c.branch, kind: c.kind, reason: c.reason, into: c.absorbed_into, overlay: 'child' }));
  return {
    key: r.key, parent: r.parent || null, level: r.level, quarter: r.quarter, status: r.status,
    p: isNum(r.p_path_own) && source === 'overlay' ? r.p_path_own : r.p_path, pc: r.p_cond, own: r.own_mass,
    source: r.source === 'overlay' || source === 'overlay' ? 'overlay' : 'snapshot',
    expandedOnDemand: !!(ov && ov.expanded_at) || (r.overlay_children || []).length > 0,
    cl: r.cluster ? { id: r.cluster.id, label: r.cluster.label, p: r.cluster.p, head: r.cluster.head || {}, realized: r.cluster.realized || {} } : null,
    br: r.branch ? { id: r.branch.id, kind: r.branch.kind, label: r.branch.label, axis: r.branch.axis, p: r.branch.p } : null,
    facts: r.facts || {}, stress: !!r.stress, flags: r.flags || [],
    rep: s.rep || {}, gdp: Array.isArray(s.bands?.d_rgdp) ? s.bands.d_rgdp[1] : null, cpi: s.cpi_cum ?? null,
    assets, attr: s.attribution || null,
    acc: (s.accounts || []).map(a => [a.line, a.median]),
    kids, okids, absorbed: (r.absorbed || []).length, into: r.absorbed_into || null,
  };
}

// ---------------------------------------------------------------- sources
export class Source {
  constructor(kind, id) {
    this.kind = kind;                       // 'snapshot' | 'run'
    this.id = id;
    this.base = kind === 'snapshot' ? `agents/snapshots/${encodeURIComponent(id)}` : `agents/runs/${encodeURIComponent(id)}`;
    this.cache = new Map();                 // key -> {record, detail, path} (LRU)
  }
  get skey() { return `${this.kind}:${this.id}`; }
  summary(opts) { return get(this.base, opts); }
  level(d, opts) { return get(`${this.base}/levels/${d}`, opts); }
  async node(key, { fresh = false, signal } = {}) {
    if (!fresh && this.cache.has(key)) {
      const v = this.cache.get(key);
      this.cache.delete(key); this.cache.set(key, v);
      return v;
    }
    const v = await get(`${this.base}/nodes/${key}`, { signal });
    this.cache.set(key, v);
    while (this.cache.size > 240) this.cache.delete(this.cache.keys().next().value);
    return v;
  }
  forget(key) { if (key) this.cache.delete(key); else this.cache.clear(); }
  // full records root..key (one request for a snapshot; node by node for a run)
  async pathRecords(key, { signal } = {}) {
    if (this.kind === 'snapshot') {
      const r = await get(`${this.base}/path/${key}`, { signal });
      return (r.path || []).filter(Boolean);
    }
    const first = await this.node(key, { signal });
    const keys = first.path || [key];
    const out = await mapLimit(keys, 6, async (k) => (await this.node(k, { signal })).record);
    return out.filter(Boolean);
  }
  expand(key, quarters = 1, opts, confirm = false) {
    return post(`${this.base}/nodes/${key}/expand`, confirm ? { quarters, wait: 0, confirm: true } : { quarters, wait: 0 }, opts);
  }
  // the most probable kept descendant of `key` at the deepest level its subtree reaches (server-side, every level)
  bestLeaf(key, opts) { return get(`${this.base}/best_leaf/${key}`, opts); }
  job(jid, opts) { return get(`${this.base}/jobs/${encodeURIComponent(jid)}`, opts); }
  branches(depth, opts) { return get(`${this.base}/branches${isNum(depth) ? `?depth=${depth}` : ''}`, opts); }
  priorBranches(tree, depth, opts) {
    return get(`agents/runs/${encodeURIComponent(this.id)}/prior_trees/${tree}/branches${isNum(depth) ? `?depth=${depth}` : ''}`, opts);
  }
}

export async function mapLimit(items, n, fn) {
  const out = new Array(items.length);
  let i = 0;
  const worker = async () => { while (i < items.length) { const k = i++; out[k] = await fn(items[k], k); } };
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, worker));
  return out;
}

// ---------------------------------------------------------------- the tree of a source (skeletons)
// Shared by the browser and the storyboard, kept for the page load (module state), so going back to the tab or
// switching the K-line colours does not fetch the levels again.
const STORES = new Map();
export function treeOf(src) {
  let st = STORES.get(src.skey);
  if (!st) { st = new TreeStore(src); STORES.set(src.skey, st); }
  return st;
}
export function dropTree(src) { STORES.delete(src.skey); }

export class TreeStore {
  constructor(src) {
    this.src = src;
    this.summary = null;
    this.byKey = new Map();
    this.levels = new Map();           // level -> [skeleton] (kept core nodes and overlay nodes)
    this.loaded = new Set();           // levels whose core file has been read
    this.depth = 0;
    this.listeners = new Set();
    this.loading = null;
    this.error = null;
  }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(what) { for (const fn of [...this.listeners]) { try { fn(what, this); } catch (err) { console.error(err); } } }
  get root() { return this.summary?.root || null; }
  get complete() { return this.depth >= 0 && this.loaded.size >= this.depth + 1; }

  async init({ fresh = false } = {}) {
    if (this.summary && !fresh) return this.summary;
    this.summary = await this.src.summary();
    // a snapshot's summary has levels {d: …}; a run's has levels_view {d: …} (its `levels` is the list of level stats)
    const lv = this.summary.levels_view || (Array.isArray(this.summary.levels) ? null : this.summary.levels) || {};
    const ds = Object.keys(lv).map(Number).filter(Number.isFinite);
    this.depth = ds.length ? Math.max(...ds) : 0;
    this.levelInfo = lv;
    return this.summary;
  }

  add(sk) {
    const old = this.byKey.get(sk.key);
    if (old) {
      const arr = this.levels.get(old.level) || [];
      const i = arr.indexOf(old);
      if (i >= 0) arr.splice(i, 1);
    }
    this.byKey.set(sk.key, sk);
    if (!this.levels.has(sk.level)) this.levels.set(sk.level, []);
    this.levels.get(sk.level).push(sk);
  }

  async loadLevel(d, { signal } = {}) {
    const lv = await this.src.level(d, { signal });
    for (const r of lv.nodes || []) {
      if (r.status === 'pruned' && r.overlay !== 'expanded') continue;
      this.add(skeleton(r, r.status === 'pruned' ? 'overlay' : 'snapshot'));
    }
    this.loaded.add(d);
    return lv;
  }

  // every level, `concurrency` at a time in level order; listeners hear ('level', d) as each one lands
  loadAll({ signal, concurrency = 4 } = {}) {
    // a load started by a view that has since closed (its signal aborted) is not reused: it stops at its next level
    if (this.loading && !this.loadingSignal?.aborted) return this.loading;
    this.loadingSignal = signal || null;
    const mine = (this.loading = (async () => {
      await this.init();
      const todo = [];
      for (let d = 0; d <= this.depth; d++) if (!this.loaded.has(d)) todo.push(d);
      let i = 0;
      const worker = async () => {
        while (i < todo.length) {
          const d = todo[i++];
          if (signal?.aborted) return;
          try { await this.loadLevel(d, { signal }); } catch (err) {
            if (err.name === 'AbortError') return;
            this.error = err;
          }
          this.emit('level', d);
        }
      };
      await Promise.all(Array.from({ length: concurrency }, worker));
      if (signal?.aborted) return;
      await this.loadOverlay({ signal });
      this.emit('done');
    })().finally(() => { if (this.loading === mine) this.loading = null; }));
    return mine;
  }

  // overlay nodes (generated on demand) that the level files do not list
  async loadOverlay({ signal, fresh = false } = {}) {
    if (fresh) {
      this.summary = await this.src.summary();
    }
    const idx = this.summary?.overlay || {};
    const keys = Object.keys(idx).filter(k => fresh || !this.byKey.has(k) || this.byKey.get(k).source !== 'overlay' || idx[k].expanded);
    await mapLimit(keys, 6, async (k) => {
      try {
        const v = await this.src.node(k, { fresh, signal });
        if (v?.record) this.add(skeleton(v.record, v.record.source === 'overlay' || v.record.status === 'pruned' ? 'overlay' : 'snapshot'));
      } catch (err) { if (err.name !== 'AbortError') this.error = err; }
    });
    if (keys.length) this.emit('overlay');
  }

  // a run that is still growing: re-read the summary and any level not read yet
  async refresh({ signal } = {}) {
    await this.init({ fresh: true });
    for (let d = 0; d <= this.depth; d++) {
      if (!this.loaded.has(d) || d >= this.depth - 1) {
        try { await this.loadLevel(d, { signal }); } catch (err) { if (err.name !== 'AbortError') this.error = err; }
      }
    }
    this.emit('done');
  }

  get(key) { return this.byKey.get(key) || null; }
  nodesAt(d) { return this.levels.get(d) || []; }
  maxLoadedLevel() { let m = -1; for (const [d, arr] of this.levels) if (arr.length && d > m) m = d; return m; }

  // a node that is drawn as part of the tree: kept in the core, or generated on demand
  static shown(sk) { return !!sk && (sk.status !== 'pruned' || sk.expandedOnDemand); }
  // kept children (core and overlay) of a node, most probable first
  children(key) {
    const n = this.get(key);
    if (!n) return [];
    const out = [];
    const seen = new Set();
    for (const c of [...n.kids, ...n.okids]) {
      if (seen.has(c.key)) continue;
      seen.add(c.key);
      const sk = this.get(c.key);
      if (TreeStore.shown(sk) && sk.parent === key) out.push(sk);
    }
    return out.sort((a, b) => (b.p || 0) - (a.p || 0));
  }
  // children not in the snapshot (pruned and never generated): drawn as 未生成
  ghosts(key) {
    const n = this.get(key);
    if (!n) return [];
    const seen = new Set();
    const out = [];
    for (const c of [...n.kids, ...n.okids]) {
      if (seen.has(c.key)) continue;
      seen.add(c.key);
      if (c.status !== 'pruned' || c.overlay === 'expanded' || TreeStore.shown(this.get(c.key))) continue;
      out.push(c);
    }
    return out.sort((a, b) => (b.pc || 0) - (a.pc || 0));
  }
  pathKeys(key) {
    const out = [];
    let cur = this.get(key);
    const seen = new Set();
    while (cur && !seen.has(cur.key)) { seen.add(cur.key); out.push(cur.key); cur = cur.parent ? this.get(cur.parent) : null; }
    return out.reverse();
  }
  // follow the most probable kept child down to a leaf
  bestDescent(key) {
    const out = [key];
    let cur = key;
    for (let guard = 0; guard < 200; guard++) {
      const kids = this.children(cur);
      if (!kids.length) break;
      cur = kids[0].key;
      out.push(cur);
    }
    return out;
  }
  // the most probable node of the deepest loaded level (the default storyboard path ends there)
  bestLeaf() {
    const d = this.maxLoadedLevel();
    const arr = this.nodesAt(d).filter(n => n.source !== 'overlay' && TreeStore.shown(n));
    return (arr.length ? arr : this.nodesAt(d)).slice().sort((a, b) => (b.p || 0) - (a.p || 0))[0] || null;
  }
}
