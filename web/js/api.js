// Thin fetch wrapper around the local API. GET results are cached per page load.
const cache = new Map();

async function handle(res) {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) { /* not json */ }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export function get(path, { fresh = false } = {}) {
  if (!fresh && cache.has(path)) return cache.get(path);
  const p = fetch(`/api/${path}`).then(handle);
  cache.set(path, p);
  p.catch(() => cache.delete(path));
  return p;
}

export async function post(path, body) {
  const res = await fetch(`/api/${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  return handle(res);
}

export async function del(path) {
  const res = await fetch(`/api/${path}`, { method: 'DELETE' });
  return handle(res);
}

export function invalidate(prefix = '') {
  for (const k of [...cache.keys()]) if (k.startsWith(prefix)) cache.delete(k);
}
