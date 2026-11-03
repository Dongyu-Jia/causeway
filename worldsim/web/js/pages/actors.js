// Actors: the list and an editor (role, what they care about, action space, private info, visible fields, model).
// Edits stay in a draft until "Save actors" (one PUT of the whole list, validated by the server).
export async function render(root, ctx) {
  const { api, ui, wid } = ctx;
  const h = ui.h;
  const world = ctx.world;
  let draft = structuredClone(world.actors);
  let current = draft.length ? 0 : -1;
  let dirty = false;

  const listBox = h('div', { class: 'item-list' });
  const editor = h('div');
  const status = h('span', { class: 'dirty' });
  const errors = h('div');
  const saveBtn = h('button', { class: 'btn primary', type: 'button', onclick: save }, 'Save actors');
  const revertBtn = h('button', { class: 'btn', type: 'button', onclick: revert }, 'Revert');

  root.append(
    h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, `Actors (${draft.length})`),
      h('div', { class: 'row' },
        h('button', { class: 'btn', type: 'button', onclick: addActor }, '+ Add actor'))),
    h('p', { class: 'small muted' }, 'Each actor is called once per iteration (in parallel) with its role, what it cares about, '
      + 'its action space, its private information and the current state. Disabled actors sit out.')),
    errors,
    h('div', { class: 'split' }, h('div', { class: 'card' }, listBox), h('div', null, editor)),
    h('div', { class: 'savebar' }, saveBtn, revertBtn, status));

  function setDirty(d) {
    dirty = d;
    ctx.setDirty(d);
    status.textContent = d ? 'Unsaved changes' : '';
    saveBtn.disabled = !d;
    revertBtn.disabled = !d;
  }
  setDirty(false);

  function drawList() {
    ui.clear(listBox);
    if (!draft.length) listBox.append(h('div', { class: 'empty' }, 'No actors yet.'));
    draft.forEach((a, i) => {
      listBox.append(h('button', { type: 'button', class: `item${i === current ? ' active' : ''}${a.enabled ? '' : ' off'}`,
        onclick: () => { current = i; drawList(); drawEditor(); } },
      h('span', { class: 'nm' }, a.name || a.id || '(unnamed)'),
      h('span', { class: 'row' }, h('span', { class: 'badge' }, `${a.action_space.length} actions`),
        a.model ? h('span', { class: 'badge accent', title: a.model }, 'model') : null)));
    });
  }

  function touch() { setDirty(true); drawList(); }

  function drawEditor() {
    ui.clear(editor);
    const a = draft[current];
    if (!a) { editor.append(h('div', { class: 'card' }, h('div', { class: 'empty' }, 'Select or add an actor.'))); return; }
    const idInput = h('input', { type: 'text', value: a.id, class: ui.ID_RE.test(a.id) ? '' : 'bad' });
    idInput.addEventListener('input', () => { a.id = idInput.value.trim(); idInput.classList.toggle('bad', !ui.ID_RE.test(a.id)); touch(); });
    const nameInput = h('input', { type: 'text', value: a.name });
    nameInput.addEventListener('input', () => { a.name = nameInput.value; touch(); });
    const enabled = h('input', { type: 'checkbox', checked: a.enabled, onchange: (e) => { a.enabled = e.target.checked; touch(); } });
    const role = h('textarea', { rows: 5 }); role.value = a.role;
    role.addEventListener('input', () => { a.role = role.value; touch(); });
    const priv = h('textarea', { rows: 10, class: 'mono' }); priv.value = a.private_info;
    priv.addEventListener('input', () => { a.private_info = priv.value; touch(); });
    const free = h('input', { type: 'checkbox', checked: a.allow_free_action, onchange: (e) => { a.allow_free_action = e.target.checked; touch(); } });
    const model = ui.modelInput(a.model, { placeholder: world.model?.model || ctx.settings?.default_model || 'the default model',
      onchange: (v) => { a.model = v; touch(); } });

    const actionsBox = h('div', { class: 'list-editor' });
    const drawActions = () => {
      ui.clear(actionsBox);
      a.action_space.forEach((act, i) => {
        const id = h('input', { type: 'text', value: act.id, placeholder: 'id', 'aria-label': 'Action id',
          class: ui.ID_RE.test(act.id) && act.id !== 'free' ? '' : 'bad' });
        id.addEventListener('input', () => { act.id = id.value.trim(); id.classList.toggle('bad', !ui.ID_RE.test(act.id) || act.id === 'free'); touch(); });
        const label = h('input', { type: 'text', value: act.label, placeholder: 'label', 'aria-label': 'Action label' });
        label.addEventListener('input', () => {
          const was = ui.slugify(act.label, 'action');
          act.label = label.value;
          if (!act.id || act.id === was) { act.id = ui.slugify(label.value, 'action'); id.value = act.id; }
          touch();
        });
        const desc = h('textarea', { rows: 2, placeholder: 'description (what it means, size, conditions)', 'aria-label': 'Action description' });
        desc.value = act.description;
        desc.addEventListener('input', () => { act.description = desc.value; touch(); });
        const move = (d) => { const j = i + d; if (j < 0 || j >= a.action_space.length) return;
          [a.action_space[i], a.action_space[j]] = [a.action_space[j], a.action_space[i]]; touch(); drawActions(); };
        actionsBox.append(h('div', { class: 'action-row' }, id, label,
          h('div', { class: 'row-tools' },
            h('button', { class: 'btn tiny', type: 'button', title: 'Move up', 'aria-label': 'Move up', onclick: () => move(-1) }, '↑'),
            h('button', { class: 'btn tiny', type: 'button', title: 'Move down', 'aria-label': 'Move down', onclick: () => move(1) }, '↓'),
            h('button', { class: 'btn tiny danger', type: 'button', title: 'Remove', 'aria-label': 'Remove action',
              onclick: () => { a.action_space.splice(i, 1); touch(); drawActions(); } }, '✕')),
          desc));
      });
      actionsBox.append(h('div', null, h('button', { class: 'btn tiny', type: 'button', onclick: () => {
        const taken = new Set(a.action_space.map((x) => x.id));
        a.action_space.push({ id: ui.uniqueId('new_action', taken), label: 'New action', description: '' });
        touch(); drawActions();
      } }, '+ Add action')));
    };
    drawActions();

    // visible fields
    const visBox = h('div');
    const drawVis = () => {
      ui.clear(visBox);
      const all = a.visible_fields === null || a.visible_fields === undefined;
      visBox.append(h('div', { class: 'row' },
        h('label', { class: 'check' }, h('input', { type: 'radio', name: `vis-${current}`, checked: all,
          onchange: () => { a.visible_fields = null; touch(); drawVis(); } }), 'All fields'),
        h('label', { class: 'check' }, h('input', { type: 'radio', name: `vis-${current}`, checked: !all,
          onchange: () => { a.visible_fields = world.schema.fields.map((f) => f.id); touch(); drawVis(); } }), 'Only selected fields')));
      if (!all) {
        const set = new Set(a.visible_fields);
        const list = h('div', { class: 'vis-list' });
        let group = null;
        for (const f of world.schema.fields) {
          if (f.group !== group) { group = f.group; list.append(h('div', { class: 'grp' }, group)); }
          list.append(h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: set.has(f.id), onchange: (e) => {
            if (e.target.checked) set.add(f.id); else set.delete(f.id);
            a.visible_fields = world.schema.fields.map((x) => x.id).filter((x) => set.has(x)); touch();
          } }), f.label));
        }
        visBox.append(list, h('div', { class: 'hint small muted' }, `${set.size} of ${world.schema.fields.length} fields visible to this actor`));
      }
    };
    drawVis();

    editor.append(h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, a.name || '(unnamed actor)'),
        h('div', { class: 'row' },
          h('button', { class: 'btn', type: 'button', onclick: duplicate }, 'Duplicate'),
          h('button', { class: 'btn danger', type: 'button', onclick: remove }, 'Delete'))),
      h('div', { class: 'form' },
        h('div', { class: 'inline-fields' }, ui.field('Name', nameInput), ui.field('ID', idInput, 'lower-case letters, digits, _'),
          h('div', { class: 'field' }, h('span', null, 'Status'), h('label', { class: 'check' }, enabled, 'Enabled'))),
        ui.field('Role (who they are)', role),
        h('div', { class: 'field' }, h('span', null, 'Cares about (goals, interests, constraints)'),
          ui.textList(a.cares_about, { placeholder: 'something this actor cares about', addLabel: 'Add item',
            onchange: (list) => { a.cares_about = list; touch(); } })),
        h('div', { class: 'field' }, h('span', null, `Action space (${a.action_space.length})`),
          h('span', { class: 'hint' }, 'Put the status quo first. The actor picks one id per iteration.'), actionsBox,
          h('label', { class: 'check' }, free, 'Allow free actions (an action of its own, described in its answer)')),
        ui.field('Private information (only this actor sees it)', priv),
        h('div', { class: 'field' }, h('span', null, 'Visible state fields'), visBox),
        h('div', { class: 'field' }, h('span', null, 'Model override'), model.el))));
  }

  function addActor() {
    const taken = new Set(draft.map((a) => a.id));
    draft.push({ id: ui.uniqueId('new_actor', taken), name: 'New actor', enabled: true, role: '', cares_about: [],
      action_space: [{ id: 'status_quo', label: 'Keep the status quo', description: '' }], allow_free_action: true,
      private_info: '', visible_fields: null, model: '' });
    current = draft.length - 1;
    touch(); drawEditor();
  }

  function duplicate() {
    const a = structuredClone(draft[current]);
    const taken = new Set(draft.map((x) => x.id));
    a.id = ui.uniqueId(`${a.id}_copy`.slice(0, 48), taken);
    a.name = `${a.name} (copy)`;
    draft.splice(current + 1, 0, a);
    current += 1;
    touch(); drawEditor();
  }

  async function remove() {
    const a = draft[current];
    if (!(await ui.confirmBox({ title: `Delete ${a.name}?`, body: 'The actor is removed when you save. Past iterations keep its outputs.', ok: 'Delete', danger: true }))) return;
    draft.splice(current, 1);
    current = Math.min(current, draft.length - 1);
    touch(); drawEditor();
  }

  async function save() {
    ui.clear(errors);
    try {
      const res = await api.put(`worlds/${api.enc(wid)}/actors`, { actors: draft });
      world.actors = res.world.actors;
      draft = structuredClone(world.actors);
      current = Math.min(Math.max(current, 0), draft.length - 1);
      setDirty(false);
      drawList(); drawEditor();
      ui.toast('Actors saved', 'ok');
    } catch (e) {
      errors.append(ui.errorBox(e, 'Not saved.'));
      errors.scrollIntoView({ block: 'nearest' });
    }
  }

  function revert() {
    draft = structuredClone(world.actors);
    current = Math.min(Math.max(current, 0), draft.length - 1);
    setDirty(false);
    ui.clear(errors);
    drawList(); drawEditor();
  }

  drawList();
  drawEditor();
}
