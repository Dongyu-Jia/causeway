// Schema: the fields (add, edit, delete, reorder, groups), the initial state's values, and the initial text
// (narrative, key questions, attributes). The initial state is editable only while the world has no later steps.
const TYPES = ['number', 'percent', 'integer', 'text', 'enum', 'boolean'];
const NUMERIC = new Set(['number', 'percent', 'integer']);

export async function render(root, ctx) {
  const { api, ui, wid } = ctx;
  const h = ui.h;
  const world = ctx.world;
  const steps = (ctx.meta && ctx.meta.steps) || [0];
  const hasHistory = steps.length && steps[steps.length - 1] > 0;
  let fields = structuredClone(world.schema.fields);
  let vals = structuredClone(world.initial_state.values || {});
  let text = structuredClone(world.initial_state.text || { narrative: '', key_questions: [], attributes: [] });
  let period = world.initial_state.period || '';
  let editing = null;              // index of the field whose editor is open

  const status = h('span', { class: 'dirty' });
  const errors = h('div');
  const saveBtn = h('button', { class: 'btn primary', type: 'button', onclick: save }, hasHistory ? 'Save schema' : 'Save schema & initial state');
  const revertBtn = h('button', { class: 'btn', type: 'button', onclick: revert }, 'Revert');
  const fieldsBox = h('div');
  const initBox = h('div');
  const groupsList = h('datalist', { id: 'schema-groups' });

  root.append(
    h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'Schema'),
        h('button', { class: 'btn', type: 'button', onclick: addField }, '+ Add field')),
      h('p', { class: 'small muted' }, 'Every state has a value for each field. The synthesizer must return the same schema; '
        + 'a value outside the plausible min/max is kept but flagged. The tag (e.g. 未确认, judgment) is shown next to the field.'),
      hasHistory ? h('div', { class: 'banner info' }, `This world has ${steps[steps.length - 1]} step(s) after the initial state: `
        + 'fields can still be edited (a new field starts empty and the synthesizer fills it), but the initial state is read-only. '
        + 'Rewind to step 0 or fork the world to change it.') : null,
      groupsList, fieldsBox),
    h('div', { class: 'card' }, h('h2', null, 'Initial state (step 0)'), initBox),
    errors,
    h('div', { class: 'savebar' }, saveBtn, revertBtn, status));

  function setDirty(d) {
    ctx.setDirty(d);
    status.textContent = d ? 'Unsaved changes' : '';
    saveBtn.disabled = !d;
    revertBtn.disabled = !d;
  }
  setDirty(false);
  const touch = () => setDirty(true);

  function valueInput(f) {
    const v = vals[f.id];
    const dis = !!hasHistory;
    if (f.type === 'boolean') {
      const s = h('select', { disabled: dis, 'aria-label': `Initial value of ${f.label}` },
        h('option', { value: '' }, '—'), h('option', { value: 'true', selected: v === true }, 'true'),
        h('option', { value: 'false', selected: v === false }, 'false'));
      s.addEventListener('change', () => { vals[f.id] = s.value === '' ? null : s.value === 'true'; touch(); });
      return s;
    }
    if (f.type === 'enum') {
      const s = h('select', { disabled: dis, 'aria-label': `Initial value of ${f.label}` }, h('option', { value: '' }, '—'),
        (f.options || []).map((o) => h('option', { value: o, selected: v === o }, o)));
      s.addEventListener('change', () => { vals[f.id] = s.value === '' ? null : s.value; touch(); });
      return s;
    }
    const inp = h('input', { type: 'text', value: v === null || v === undefined ? '' : String(v), disabled: dis,
      inputmode: NUMERIC.has(f.type) ? 'decimal' : null, 'aria-label': `Initial value of ${f.label}`,
      placeholder: NUMERIC.has(f.type) ? (f.unit || 'number') : 'text' });
    inp.addEventListener('input', () => {
      const raw = inp.value.trim();
      if (NUMERIC.has(f.type)) {
        const n = Number(raw);
        vals[f.id] = raw === '' ? null : (Number.isFinite(n) ? n : raw);
        inp.classList.toggle('bad', raw !== '' && !Number.isFinite(n));
      } else vals[f.id] = inp.value;
      touch();
    });
    return inp;
  }

  function drawFields() {
    ui.clear(fieldsBox);
    ui.clear(groupsList);
    const groups = [...new Set(fields.map((f) => f.group))];
    groups.forEach((g) => groupsList.append(h('option', { value: g })));
    if (!fields.length) fieldsBox.append(h('div', { class: 'empty' }, 'No fields yet.'));
    let group = null;
    fields.forEach((f, i) => {
      if (f.group !== group) {
        group = f.group;
        fieldsBox.append(h('h3', { class: 'small muted' }, `${group} · ${fields.filter((x) => x.group === group).length}`));
      }
      const move = (d) => { const j = i + d; if (j < 0 || j >= fields.length) return;
        [fields[i], fields[j]] = [fields[j], fields[i]]; if (editing === i) editing = j; else if (editing === j) editing = i; touch(); drawFields(); };
      fieldsBox.append(h('div', { class: 'schema-row' },
        h('div', { class: 'ord' },
          h('button', { class: 'btn tiny', type: 'button', title: 'Move up', 'aria-label': 'Move up', onclick: () => move(-1) }, '↑'),
          h('button', { class: 'btn tiny', type: 'button', title: 'Move down', 'aria-label': 'Move down', onclick: () => move(1) }, '↓')),
        h('div', null, h('div', { class: 'fl-label' }, f.label || '(no label)', ' ', f.tag ? h('span', { class: 'badge tag' }, f.tag) : null),
          h('div', { class: 'fl-id' }, f.id)),
        h('div', { class: 's-type small' }, h('span', { class: 'badge' }, f.type), ' ', h('span', { class: 'muted' }, f.unit || '')),
        h('div', { class: 's-init' }, valueInput(f)),
        h('div', { class: 'tools' },
          h('button', { class: 'btn tiny', type: 'button', onclick: () => { editing = editing === i ? null : i; drawFields(); } },
            editing === i ? 'Close' : 'Edit'),
          h('button', { class: 'btn tiny danger', type: 'button', title: 'Delete field', 'aria-label': 'Delete field', onclick: () => removeField(i) }, '✕'))));
      if (editing === i) fieldsBox.append(fieldEditor(f, i));
    });
  }

  function fieldEditor(f, i) {
    const inp = (key, attrs = {}) => {
      const el = h('input', { type: 'text', value: f[key] ?? '', ...attrs });
      el.addEventListener('input', () => { f[key] = el.value; touch(); });
      return el;
    };
    const idInput = h('input', { type: 'text', value: f.id, class: ui.ID_RE.test(f.id) ? '' : 'bad' });
    idInput.addEventListener('change', () => {
      const old = f.id;
      const nid = idInput.value.trim();
      idInput.classList.toggle('bad', !ui.ID_RE.test(nid));
      if (nid !== old) {
        if (fields.some((x, j) => j !== i && x.id === nid)) { ui.toast(`The id ${nid} is already used`, 'bad'); idInput.value = old; return; }
        vals[nid] = vals[old]; delete vals[old];
        f.id = nid; touch(); drawFields();
      }
    });
    const label = inp('label');
    label.addEventListener('change', () => drawFields());
    const groupIn = h('input', { type: 'text', value: f.group, list: 'schema-groups' });
    groupIn.addEventListener('change', () => { f.group = groupIn.value.trim() || 'General'; touch(); drawFields(); });
    const type = h('select', null, TYPES.map((t) => h('option', { value: t, selected: f.type === t }, t)));
    type.addEventListener('change', () => {
      f.type = type.value;
      if (f.type === 'enum' && !(f.options || []).length) f.options = ['option a', 'option b'];
      vals[f.id] = null;
      touch(); drawFields();
    });
    const desc = h('textarea', { rows: 3 }); desc.value = f.description || '';
    desc.addEventListener('input', () => { f.description = desc.value; touch(); });
    const opts = h('textarea', { rows: 4, placeholder: 'one option per line' }); opts.value = (f.options || []).join('\n');
    opts.addEventListener('change', () => {
      f.options = opts.value.split('\n').map((s) => s.trim()).filter(Boolean);
      if (vals[f.id] !== null && !f.options.includes(vals[f.id])) vals[f.id] = null;
      touch(); drawFields();
    });
    const numIn = (key) => {
      const el = h('input', { type: 'text', inputmode: 'decimal', value: f[key] === null || f[key] === undefined ? '' : String(f[key]), placeholder: 'none' });
      el.addEventListener('input', () => {
        const raw = el.value.trim(); const n = Number(raw);
        f[key] = raw === '' ? null : (Number.isFinite(n) ? n : raw);
        el.classList.toggle('bad', raw !== '' && !Number.isFinite(n)); touch();
      });
      return el;
    };
    return h('div', { class: 'schema-edit form' },
      h('div', { class: 'inline-fields' }, ui.field('ID', idInput, 'lower-case letters, digits, _'), ui.field('Label', label),
        ui.field('Group', groupIn), ui.field('Type', type), ui.field('Unit', inp('unit', { placeholder: 'e.g. %, 十亿美元' })),
        ui.field('Tag', inp('tag', { placeholder: 'e.g. 未确认' }))),
      ui.field('Description (what it measures; shown to the synthesizer)', desc),
      f.type === 'enum' ? ui.field('Options', opts) : null,
      NUMERIC.has(f.type) ? h('div', { class: 'inline-fields' }, ui.field('Plausible min', numIn('min')), ui.field('Plausible max', numIn('max'))) : null,
      ui.field('Historical range note', inp('hist', { placeholder: 'e.g. 1954–2026: 0.05 … 19.1, median 4.3' })));
  }

  function addField() {
    const taken = new Set(fields.map((f) => f.id));
    const group = fields.length ? fields[fields.length - 1].group : 'General';
    fields.push({ id: ui.uniqueId('new_field', taken), label: 'New field', group, type: 'number', unit: '', description: '',
      options: [], min: null, max: null, hist: '', tag: '' });
    vals[fields[fields.length - 1].id] = null;
    editing = fields.length - 1;
    touch(); drawFields();
    fieldsBox.lastElementChild?.scrollIntoView({ block: 'nearest' });
  }

  async function removeField(i) {
    const f = fields[i];
    if (!(await ui.confirmBox({ title: `Delete the field ${f.label}?`, body: 'Saved states keep their old values, but the field is no longer shown or simulated.', ok: 'Delete', danger: true }))) return;
    fields.splice(i, 1);
    delete vals[f.id];
    editing = null;
    touch(); drawFields();
  }

  function drawInit() {
    ui.clear(initBox);
    const dis = !!hasHistory;
    const periodIn = h('input', { type: 'text', value: period, disabled: dis, placeholder: 'e.g. 2026Q4' });
    periodIn.addEventListener('input', () => { period = periodIn.value; touch(); });
    const narr = h('textarea', { class: 'tall', disabled: dis }); narr.value = text.narrative || '';
    narr.addEventListener('input', () => { text.narrative = narr.value; touch(); });
    const qBox = h('div', { class: 'list-editor' });
    const drawQ = () => {
      ui.clear(qBox);
      text.key_questions.forEach((q, i) => {
        const qi = h('textarea', { rows: 2, disabled: dis, placeholder: 'question', 'aria-label': 'Question' }); qi.value = q.question;
        qi.addEventListener('input', () => { q.question = qi.value; touch(); });
        const st = h('select', { disabled: dis, 'aria-label': 'Status' }, ['open', 'answered'].map((s) => h('option', { value: s, selected: q.status === s }, s)));
        st.addEventListener('change', () => { q.status = st.value; touch(); });
        const an = h('textarea', { rows: 2, disabled: dis, placeholder: 'current answer', 'aria-label': 'Answer' }); an.value = q.answer || '';
        an.addEventListener('input', () => { q.answer = an.value; touch(); });
        qBox.append(h('div', { class: 'q-row' }, qi, st, an, h('button', { class: 'btn tiny danger', type: 'button', disabled: dis,
          'aria-label': 'Remove question', onclick: () => { text.key_questions.splice(i, 1); touch(); drawQ(); } }, '✕')));
      });
      if (!dis) qBox.append(h('div', null, h('button', { class: 'btn tiny', type: 'button',
        onclick: () => { text.key_questions.push({ question: '', status: 'open', answer: '' }); touch(); drawQ(); } }, '+ Add question')));
    };
    const aBox = h('div', { class: 'list-editor' });
    const drawA = () => {
      ui.clear(aBox);
      text.attributes.forEach((a, i) => {
        const n = h('input', { type: 'text', value: a.name, disabled: dis, placeholder: 'name', 'aria-label': 'Attribute name' });
        n.addEventListener('input', () => { a.name = n.value; touch(); });
        const v = h('textarea', { rows: 1, disabled: dis, placeholder: 'value', 'aria-label': 'Attribute value' }); v.value = a.value;
        v.addEventListener('input', () => { a.value = v.value; touch(); });
        aBox.append(h('div', { class: 'a-row' }, n, v, h('button', { class: 'btn tiny danger', type: 'button', disabled: dis,
          'aria-label': 'Remove attribute', onclick: () => { text.attributes.splice(i, 1); touch(); drawA(); } }, '✕')));
      });
      if (!dis) aBox.append(h('div', null, h('button', { class: 'btn tiny', type: 'button',
        onclick: () => { text.attributes.push({ name: '', value: '' }); touch(); drawA(); } }, '+ Add attribute')));
    };
    drawQ(); drawA();
    initBox.append(h('div', { class: 'form' },
      h('div', { class: 'inline-fields' }, ui.field('Period of step 0', periodIn, 'e.g. 2026Q4, 2026-11, 2026 or Turn 1')),
      h('p', { class: 'small muted' }, 'The initial values are edited in the field list above.'),
      ui.field('Narrative', narr),
      h('div', { class: 'field' }, h('span', null, `Key questions (${text.key_questions.length})`), qBox),
      h('div', { class: 'field' }, h('span', null, `Attributes (${text.attributes.length})`), aBox)));
  }

  async function save() {
    ui.clear(errors);
    try {
      const body = { fields };
      if (!hasHistory) body.initial_values = vals;
      let res = await api.put(`worlds/${api.enc(wid)}/schema`, body);
      if (!hasHistory) {
        res = await api.put(`worlds/${api.enc(wid)}/initial`, { period, values: vals,
          text: { ...text, key_questions: text.key_questions.filter((q) => q.question.trim()),
            attributes: text.attributes.filter((a) => a.name.trim()) } });
      }
      Object.assign(world, res.world);
      revert();
      ui.toast('Schema saved', 'ok');
    } catch (e) {
      errors.append(ui.errorBox(e, 'Not saved.'));
      errors.scrollIntoView({ block: 'nearest' });
    }
  }

  function revert() {
    fields = structuredClone(world.schema.fields);
    vals = structuredClone(world.initial_state.values || {});
    text = structuredClone(world.initial_state.text);
    period = world.initial_state.period || '';
    editing = null;
    setDirty(false);
    drawFields(); drawInit();
  }

  drawFields();
  drawInit();
}
