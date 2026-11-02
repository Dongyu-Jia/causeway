// Thin fetch wrapper around the local API. GET results are cached per page load.
const cache = new Map();

// The analysis modules write prose with rounded numbers, and a tiny negative value rounds to
// "-0.00" / "-0.0%" there ("合计只解释 -0.0%"). Strings in every response are shown without that
// sign, the way ui.fmtPct shows numbers. Dates ("2020-01") and exponents ("3.6e-08") are untouched.
const NEG_ZERO = /(^|[^\d.\w])[-−](0(?:\.0+)?)(?![\d.])/g;
function tidy(x) {
  if (typeof x === 'string') return x.includes('0') && /[-−]0/.test(x) ? x.replace(NEG_ZERO, '$1$2') : x;
  if (Array.isArray(x)) { for (let i = 0; i < x.length; i++) if (typeof x[i] === 'string' || (x[i] && typeof x[i] === 'object')) x[i] = tidy(x[i]); return x; }
  if (x && typeof x === 'object') { for (const k of Object.keys(x)) { const v = x[k]; if (typeof v === 'string' || (v && typeof v === 'object')) x[k] = tidy(v); } }
  return x;
}

async function handle(res) {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) { /* not json */ }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return tidy(await res.json());
}

export function get(path, { fresh = false } = {}) {
  if (!fresh && cache.has(path)) return cache.get(path);
  const p = fetch(`/api/${path}`).then(handle);
  cache.set(path, p);
  p.catch(() => cache.delete(path));
  return p;
}

// An optional result file: null (and no request, so no 404 in the console) when /api/meta's `results` says
// it is missing, otherwise get() with any error turned into null.
export function optional(name, meta) {
  const have = meta?.results;
  if (Array.isArray(have) && !have.includes(name)) return Promise.resolve(null);
  return get(name).catch(() => null);
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
