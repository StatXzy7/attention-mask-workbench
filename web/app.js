import { WorkbenchStore } from '../src/store.js';
import { renderScores, renderMask, renderWeights, renderOutputs, renderMetrics } from './render.js';

const store = new WorkbenchStore();
let selectedQuery = 0;

const $ = (id) => document.getElementById(id);
const els = {
  intervals: $('in-intervals'),
  budget: $('in-budget'),
  causal: $('in-causal'),
  shared: $('in-shared'),
  queryButtons: $('query-buttons'),
  reset: $('btn-reset'),
  error: $('error-banner'),
  metrics: $('metrics'),
  editors: $('editors'),
  scores: $('svg-scores'),
  mask: $('svg-mask'),
  weights: $('svg-weights'),
  outputs: $('output-table'),
};

function renderEditors(state) {
  const wasInvalidHead = state.lastErrorHead;
  els.editors.innerHTML = '';
  state.drafts.forEach((draft, h) => {
    const card = document.createElement('div');
    card.className = `head-editor${wasInvalidHead === h ? ' invalid' : ''}`;

    const heading = document.createElement('h3');
    heading.textContent = `Head ${h} — 手工编辑（空格/逗号分隔，回车换行；# 开头为注释）`;
    card.appendChild(heading);

    const fields = [
      ['q', 'Q 查询矩阵 (nq × dk)'],
      ['k', 'K 键矩阵 (nk × dk)'],
      ['v', 'V 值矩阵 (nk × dv)'],
    ];
    fields.forEach(([key, label]) => {
      const field = document.createElement('div');
      field.className = 'matrix-field';
      const lab = document.createElement('label');
      lab.textContent = label;
      const ta = document.createElement('textarea');
      ta.rows = draft[key].split('\n').length;
      ta.value = draft[key];
      ta.dataset.head = String(h);
      ta.dataset.matrix = key;
      ta.spellcheck = false;
      field.appendChild(lab);
      field.appendChild(ta);
      card.appendChild(field);
    });

    const apply = document.createElement('button');
    apply.type = 'button';
    apply.className = 'apply';
    apply.textContent = '应用 Head ' + h;
    apply.addEventListener('click', () => {
      const patch = {};
      card.querySelectorAll('textarea').forEach((ta) => {
        patch[ta.dataset.matrix] = ta.value;
      });
      store.editHead(h, patch);
    });
    card.appendChild(apply);

    const hint = document.createElement('span');
    hint.className = 'hint';
    hint.textContent = '非法编辑不会覆盖上次有效结果';
    card.appendChild(hint);

    els.editors.appendChild(card);
  });
}

function renderQueryButtons(state) {
  const nq = state.result ? state.result.model.nq : state.nq;
  if (selectedQuery >= nq) selectedQuery = 0;
  els.queryButtons.innerHTML = '';
  for (let i = 0; i < nq; i += 1) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = String(i);
    if (i === selectedQuery) btn.classList.add('active');
    btn.addEventListener('click', () => {
      selectedQuery = i;
      render(stateSafe());
    });
    els.queryButtons.appendChild(btn);
  }
}

function stateSafe() {
  return store.getState();
}

function renderError(state) {
  if (state.error) {
    els.error.classList.remove('hidden');
    els.error.textContent = '';
    const dismiss = document.createElement('span');
    dismiss.className = 'dismiss';
    dismiss.textContent = '✕ 保留旧结果并关闭';
    dismiss.addEventListener('click', () => store.dismissError());
    els.error.appendChild(dismiss);
    els.error.appendChild(document.createTextNode('编辑被拒绝：' + state.error));
  } else {
    els.error.classList.add('hidden');
    els.error.textContent = '';
  }
}

function render(state) {
  if (!state.result) return;
  const p = state.result.params;
  els.intervals.value = p.intervals;
  els.budget.value = p.budget;
  els.causal.checked = Boolean(p.causal);
  els.shared.checked = Boolean(p.shared);

  renderError(state);
  renderMetrics(els.metrics, state.result);
  renderQueryButtons(state);
  renderEditors(state);
  renderScores(els.scores, state.result, selectedQuery);
  renderMask(els.mask, state.result, selectedQuery);
  renderWeights(els.weights, state.result, selectedQuery);
  renderOutputs(els.outputs, state.result, selectedQuery);
}

els.intervals.addEventListener('change', () => store.setParam('intervals', els.intervals.value));
els.budget.addEventListener('change', () => store.setParam('budget', els.budget.value));
els.causal.addEventListener('change', () => store.setParam('causal', els.causal.checked));
els.shared.addEventListener('change', () => store.setParam('shared', els.shared.checked));
els.reset.addEventListener('click', () => {
  selectedQuery = 0;
  store.resetBuiltin();
});

store.subscribe((state) => render(state));
render(store.getState());
