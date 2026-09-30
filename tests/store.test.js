import { test } from 'node:test';
import assert from 'node:assert/strict';
import { WorkbenchStore } from '../src/store.js';
import { BUILTIN_MODEL } from '../src/data.js';

test('store starts from built-in data with default parameters and a valid result', () => {
  const store = new WorkbenchStore();
  const state = store.getState();
  assert.equal(state.error, null);
  assert.equal(state.heads, BUILTIN_MODEL.heads.length);
  assert.equal(state.params.intervals, 2);
  assert.equal(state.params.budget, 1);
  assert.equal(state.result.params.intervals, 2);
  assert.ok(Number.isFinite(state.result.metrics.relL2Error));
});

test('parameter changes recompute: interval count, budget, causal, shared', () => {
  const store = new WorkbenchStore();

  let state = store.setParam('intervals', 5);
  assert.equal(state.result.params.intervals, 5);

  state = store.setParam('budget', 2);
  // 5 single-key intervals + budget 5 -> zero error for the built-in data.
  state = store.setParam('budget', 5);
  assert.equal(state.result.params.budget, 5);
  assert.equal(state.result.metrics.maxAbsError, 0);
  assert.equal(state.result.metrics.relL2Error, 0);

  state = store.setParam('causal', true);
  assert.equal(state.result.params.causal, true);
  // Causal future keys are invisible even with a full per-head budget.
  const lastQuery = state.result.heads[0].queries.at(-1);
  assert.ok(lastQuery.visibleCount <= state.nk);

  state = store.setParam('shared', true);
  assert.equal(state.result.params.shared, true);
  // Shared cap is heads*intervals = 10 for the built-in 2 heads.
  assert.equal(state.result.params.budget, 5);
});

test('zero budget yields zero sparse vectors and empty-mask status via the store', () => {
  const store = new WorkbenchStore();
  const state = store.setParam('budget', 0);
  assert.equal(state.result.params.budget, 0);
  for (const headResult of state.result.heads) {
    for (const query of headResult.queries) {
      assert.equal(query.emptyMask, true);
      assert.deepEqual(query.retainedKeys, []);
      assert.ok(query.sparseOutput.every((v) => v === 0));
    }
  }
});

test('over-cap budgets are truncated and clamping is surfaced', () => {
  const store = new WorkbenchStore();
  const state = store.setParams({ intervals: 2, budget: 99, shared: false });
  assert.equal(state.result.params.budget, 2);
  assert.equal(state.result.params.budgetWasClamped, true);

  const shared = store.setParams({ budget: 99, shared: true });
  assert.equal(shared.result.params.budget, 4); // 2 heads x 2 intervals
  assert.equal(shared.result.params.budgetWasClamped, true);
});

test('an invalid edit is rejected and never overwrites the last valid result', () => {
  const store = new WorkbenchStore();
  const before = store.getState().result;
  const beforeText = store.getState().drafts[0].q;

  // Non-finite token.
  let state = store.editHead(0, { q: '1 0\nNaN 2' });
  assert.match(state.error, /not a finite number/);
  assert.equal(state.lastErrorHead, 0);
  assert.equal(state.result, before, 'last valid result object must survive');

  // Ragged rows.
  state = store.editHead(0, { q: '1 0 0\n2 2' });
  assert.match(state.error, /row 2 has 2 entries/);
  assert.equal(state.result, before);

  // Shape conflict against K/V of the still-valid model.
  state = store.editHead(0, { q: '1\n2\n3\n4\n5' }); // nq becomes 5 vs nk 5? k has 5 rows
  assert.ok(state.error !== null);
  assert.equal(state.result, before);

  // Draft text is preserved so the user can fix it, but the model is untouched.
  assert.equal(store.getState().drafts[0].q, '1\n2\n3\n4\n5');
  assert.equal(store.getState().nq, BUILTIN_MODEL.heads[0].q.length);

  // Editing a different head while head 0 draft is broken still works.
  state = store.editHead(1, { v: '9\n8\n7\n6\n5' });
  assert.equal(state.error, null);
  // Head 1 has dv=1; dense output is the softmax-weighted value, and only
  // positive V entries contribute, so it must be strictly positive.
  assert.ok(state.result.heads[1].queries[0].denseOutput[0] > 0);
  assert.notEqual(
    state.result.heads[1].queries[0].denseOutput[0],
    new WorkbenchStore().getState().result.heads[1].queries[0].denseOutput[0],
  );

  // Head 0 draft still holds the broken text; fixing it restores validity.
  state = store.editHead(0, { q: beforeText });
  assert.equal(state.error, null);
  assert.notEqual(state.result, before);
  assert.equal(state.lastErrorHead, null);
});

test('valid edits produce numerically changed sparse and dense outputs', () => {
  const store = new WorkbenchStore();
  const state = store.editHead(0, { v: '10 0\n0 10\n10 10\n20 0\n0 20' });
  assert.equal(state.error, null);
  const query = state.result.heads[0].queries[0];
  // Values were scaled x10 vs built-in head 0 V.
  assert.ok(Math.abs(query.denseOutput[0]) > 1);
  assert.ok(query.sparseOutput.some((v) => v !== 0));
});

test('dismissError clears the banner without mutating the valid result', () => {
  const store = new WorkbenchStore();
  store.editHead(0, { q: 'oops' });
  assert.ok(store.getState().error);
  const state = store.dismissError();
  assert.equal(state.error, null);
  assert.ok(state.result);
});

test('resetBuiltin restores defaults after edits and parameter changes', () => {
  const store = new WorkbenchStore();
  store.setParams({ intervals: 1, budget: 0, causal: true, shared: true });
  store.editHead(0, { q: '9 9\n9 9\n9 9\n9 9' });
  const state = store.resetBuiltin();
  assert.equal(state.params.intervals, 2);
  assert.equal(state.params.budget, 1);
  assert.equal(state.params.causal, false);
  assert.equal(state.params.shared, false);
  assert.equal(state.error, null);
  assert.deepEqual(
    state.result.heads[0].queries[0].denseOutput,
    new WorkbenchStore().getState().result.heads[0].queries[0].denseOutput,
  );
});

test('subscribers observe each recompute and rejected edits', () => {
  const store = new WorkbenchStore();
  const reasons = [];
  store.subscribe((_state, reason) => reasons.push(reason));
  store.setParam('budget', 2);
  store.editHead(0, { q: 'bad' });
  store.editHead(0, { q: '1 0\n0 2\n2 1\n0 0' });
  assert.deepEqual(reasons, ['param:budget', 'edit:parse-error', 'edit:ok']);
});
