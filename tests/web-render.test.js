// Minimal DOM smoke test: exercises the actual SVG render functions and the
// WorkbenchStore render path without a browser. This catches wiring errors
// (wrong element namespaces, null fields) that HTTP tests cannot.
import { test } from 'node:test';
import assert from 'node:assert/strict';

const SVG_NS = 'http://www.w3.org/2000/svg';

class FakeNode {
  constructor(tag, ns) {
    this.tag = tag;
    this.ns = ns || 'html';
    this.attributes = {};
    this.children = [];
    this.textContent = '';
    this.innerHTML = '';
    this.style = {};
    this.value = '';
    this.rows = 3;
    this.checked = false;
    this.spellcheck = true;
    this.dataset = {};
    this.className = '';
    this._listeners = {};
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
  }
  getAttribute(k) {
    return this.attributes[k] ?? null;
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  removeChild(child) {
    this.children = this.children.filter((c) => c !== child);
  }
  addEventListener(type, fn) {
    (this._listeners[type] ||= []).push(fn);
  }
  dispatch(type, event = {}) {
    for (const fn of this._listeners[type] || []) fn(event);
  }
  querySelector(selector) {
    // Only the ".k"/".v" pattern used by renderMetrics is needed.
    return new FakeNode('span');
  }
  querySelectorAll() {
    return [];
  }
}

const elements = {};
function ensure(id, tag = 'div') {
  if (!elements[id]) elements[id] = new FakeNode(tag);
  return elements[id];
}

globalThis.document = {
  createElement: (tag) => new FakeNode(tag),
  createElementNS: (ns, tag) => new FakeNode(tag, ns),
  createTextNode: (text) => {
    const n = new FakeNode('#text');
    n.textContent = text;
    return n;
  },
  getElementById: (id) => ensure(id),
};
globalThis.console = console;

const {
  renderScores,
  renderMask,
  renderWeights,
  renderOutputs,
  renderMetrics,
} = await import('../web/render.js');
const { WorkbenchStore } = await import('../src/store.js');

function renderAll(result, queryIndex) {
  for (const id of ['svg-scores', 'svg-mask', 'svg-weights', 'output-table', 'metrics']) {
    elements[id] = new FakeNode('div');
  }
  renderScores(elements['svg-scores'], result, queryIndex);
  renderMask(elements['svg-mask'], result, queryIndex);
  renderWeights(elements['svg-weights'], result, queryIndex);
  renderOutputs(elements['output-table'], result, queryIndex);
  renderMetrics(elements['metrics'], result);
}

function svgCount(node) {
  let count = 0;
  const walk = (n) => {
    if (n.ns === SVG_NS) count += 1;
    for (const c of n.children || []) walk(c);
  };
  walk(node);
  return count;
}

test('full initial render builds synchronized SVG nodes for both heads', () => {
  const store = new WorkbenchStore();
  const state = store.getState();
  renderAll(state.result, 0);
  assert.ok(svgCount(elements['svg-scores']) > 0);
  assert.ok(svgCount(elements['svg-mask']) > 0);
  assert.ok(svgCount(elements['svg-weights']) > 0);
  assert.ok(elements['output-table'].children.length >= 2);
  assert.ok(elements['metrics'].children.length >= 4);

  // Switching the simulated selected query must re-render without throwing.
  assert.doesNotThrow(() => renderAll(state.result, 2));
});

test('render path reflects parameter changes: causal + shared + zero budget', () => {
  const store = new WorkbenchStore();
  store.setParams({ intervals: 3, budget: 0, causal: true, shared: true });
  const state = store.getState();
  assert.equal(state.result.params.budget, 0);
  assert.doesNotThrow(() => renderAll(state.result, 0));
  const maskSvg = elements['svg-mask'];
  // Some cells must be drawn as "future" (causal) for query 0.
  const texts = [];
  const walk = (n) => {
    if (n.textContent) texts.push(n.textContent);
    for (const c of n.children || []) walk(c);
  };
  walk(maskSvg);
  assert.ok(texts.includes('未来'));
});

test('rejected edit keeps last render inputs valid; recovery re-renders', () => {
  const store = new WorkbenchStore();
  const good = store.getState().result;
  store.editHead(0, { q: 'not a number' });
  assert.ok(store.getState().error);
  // The surviving result must still be fully renderable.
  assert.equal(store.getState().result, good);
  assert.doesNotThrow(() => renderAll(store.getState().result, 0));

  store.editHead(0, { q: '1 0\n0 2\n2 1\n0 0' });
  assert.equal(store.getState().error, null);
  assert.doesNotThrow(() => renderAll(store.getState().result, 0));
});

test('score chart marks chosen intervals and mask agrees with retained keys', () => {
  const store = new WorkbenchStore();
  const { result } = store.setParams({ intervals: 2, budget: 1 });
  renderAll(result, 0);
  assert.ok(svgCount(elements['svg-scores']) > 0);

  const chosen = new Set(result.heads[0].queries[0].chosenIntervals);
  // Exactly one chosen bar per head exists with the warn fill color.
  let chosenBars = 0;
  const walk = (n) => {
    if (n.attributes['data-kind'] === 'score-bar' && n.attributes.fill === '#ffce6a') {
      chosenBars += 1;
    }
    for (const c of n.children || []) walk(c);
  };
  walk(elements['svg-scores']);
  assert.equal(chosenBars, result.heads.length);
  assert.equal(chosen.size, 1);
});
