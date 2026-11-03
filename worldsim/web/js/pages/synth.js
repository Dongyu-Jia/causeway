// Synthesizer: its instructions (how actions turn into consequences, consistency rules) and a model override.
export async function render(root, ctx) {
  const { api, ui, wid } = ctx;
  const h = ui.h;
  const world = ctx.world;
  const draft = { ...world.synthesizer };
  const status = h('span', { class: 'dirty' });
  const errors = h('div');
  const saveBtn = h('button', { class: 'btn primary', type: 'button', onclick: save }, 'Save synthesizer');

  const setDirty = (d) => { ctx.setDirty(d); status.textContent = d ? 'Unsaved changes' : ''; saveBtn.disabled = !d; };
  const instr = h('textarea', { class: 'tall', rows: 18, 'aria-label': 'Synthesizer instructions' });
  instr.value = draft.instructions || '';
  instr.addEventListener('input', () => { draft.instructions = instr.value; setDirty(true); });
  const model = ui.modelInput(draft.model, { placeholder: world.model?.model || ctx.settings?.default_model || 'the default model',
    onchange: (v) => { draft.model = v; setDirty(true); } });

  root.append(
    h('div', { class: 'card' },
      h('h2', null, 'Synthesizer'),
      h('p', { class: 'small muted' }, 'After every enabled actor has decided, the synthesizer sees the current state, the schema, '
        + 'all actors\' outputs (including their rationales) and these instructions, and writes the next state in the same shape: '
        + 'a value and a reason for every field, the narrative, the updated key questions and attributes, and a one-line summary. '
        + 'The fixed part of its prompt already asks for one JSON object, every field, a reason for each change and the world\'s language.'),
      h('div', { class: 'form' },
        ui.field('Instructions (world-specific rules)', instr),
        h('div', { class: 'field' }, h('span', null, 'Model override'), model.el))),
    errors,
    h('div', { class: 'savebar' }, saveBtn, status));
  setDirty(false);

  async function save() {
    ui.clear(errors);
    try {
      const res = await api.put(`worlds/${api.enc(wid)}/synthesizer`, draft);
      world.synthesizer = res.world.synthesizer;
      setDirty(false);
      ui.toast('Synthesizer saved', 'ok');
    } catch (e) { errors.append(ui.errorBox(e, 'Not saved.')); }
  }
}
