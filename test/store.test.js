'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { computeWorkbench } = require('../src/core.js');
const { createStore, DEFAULT_DATA } = require('../src/store.js');

const good = { Q: DEFAULT_DATA.Q, K: DEFAULT_DATA.K, V: DEFAULT_DATA.V };

test('拒绝非有限数', () => {
  const badQ = JSON.parse(JSON.stringify(good.Q));
  badQ[0][0][0] = NaN;
  assert.throws(() => computeWorkbench(badQ, good.K, good.V, { numBlocks: 2, budget: 1 }), /非有限数/);
  const infK = JSON.parse(JSON.stringify(good.K));
  infK[1][2][1] = -Infinity;
  assert.throws(() => computeWorkbench(good.Q, infK, good.V, { numBlocks: 2, budget: 1 }), /非有限数/);
});

test('拒绝形状冲突', () => {
  const kWrongHeads = [good.K[0]];
  assert.throws(() => computeWorkbench(good.Q, kWrongHeads, good.V, { numBlocks: 2, budget: 1 }), /头数/);
  const qWrongRows = [[good.Q[0][0]]];
  assert.throws(() => computeWorkbench(qWrongRows, good.K, good.V, { numBlocks: 1, budget: 1 }), /位置数|头数/);
  const kWrongDim = good.K.map((m) => m.map((row) => row.slice(0, 2)));
  assert.throws(() => computeWorkbench(good.Q, kWrongDim, good.V, { numBlocks: 2, budget: 1 }), /特征维度/);
  assert.throws(() => computeWorkbench('nope', good.K, good.V, { numBlocks: 2, budget: 1 }), /三维数组/);
  assert.throws(() => computeWorkbench([], good.K, good.V, { numBlocks: 2, budget: 1 }), /三维数组/);
});

test('拒绝非法区间数与预算', () => {
  assert.throws(() => computeWorkbench(good.Q, good.K, good.V, { numBlocks: 0, budget: 1 }), /区间数/);
  assert.throws(() => computeWorkbench(good.Q, good.K, good.V, { numBlocks: 99, budget: 1 }), /超过键位置数/);
  assert.throws(() => computeWorkbench(good.Q, good.K, good.V, { numBlocks: 2.5, budget: 1 }), /区间数/);
  assert.throws(() => computeWorkbench(good.Q, good.K, good.V, { numBlocks: 2, budget: -1 }), /预算/);
  assert.throws(() => computeWorkbench(good.Q, good.K, good.V, { numBlocks: 2, budget: 1.5 }), /预算/);
});

test('页面参数变更会真实重算并影响保留数量', () => {
  const store = createStore();
  const before = store.getSnapshot().result.metrics.totalRetained;
  store.setParams({ numBlocks: 4, budget: 4 });
  const after = store.getSnapshot();
  assert.equal(after.result.config.numBlocks, 4);
  // 全量保留 => 稀疏与稠密一致
  assert.equal(after.result.metrics.maxAbsError < 1e-12, true);
  assert.equal(after.result.metrics.relativeL2Error < 1e-12, true);
  assert.notEqual(before, after.result.metrics.totalRetained);

  store.setParams({ causal: true });
  assert.equal(store.getSnapshot().result.config.causal, true);
  const retainedFirstRow = store.getSnapshot().result.retained[0][0];
  assert.deepEqual(retainedFirstRow, [0]); // 因果下首行仅自身

  store.setParams({ sharedBudget: true });
  assert.equal(store.getSnapshot().result.config.sharedBudget, true);
});

test('错误参数不覆盖上次有效结果, 之后可恢复继续修改', () => {
  const store = createStore();
  store.setParams({ numBlocks: 2, budget: 1 });
  const valid = store.getSnapshot().result;
  assert.throws(() => store.setParams({ numBlocks: 99 }), /超过/);
  // 与页面一致: 捕获后通过 reportError 展示, 结果仍是上次的
  store.reportError('区间数 99 超过键位置数 4');
  const snap = store.getSnapshot();
  assert.equal(snap.result, valid);
  assert.equal(snap.params.numBlocks, 2);
  assert.match(snap.error, /超过/);
  // 恢复: 再给合法参数
  const next = store.setParams({ numBlocks: 3 });
  assert.equal(next.error, null);
  assert.equal(next.result.config.numBlocks, 3);
});

test('零预算在页面状态下显示空掩码状态', () => {
  const store = createStore();
  const snap = store.setParams({ numBlocks: 4, budget: 0 });
  assert.equal(snap.result.metrics.totalRetained, 0);
  assert.equal(snap.result.metrics.emptyRows.length, snap.result.config.heads * snap.result.config.n);
  assert.deepEqual(snap.result.sparseOut, snap.result.denseOut.map((m) => m.map((row) => row.map(() => 0))));
});

test('错误编辑不覆盖上次有效结果, 全部三个张量必须一起成功', () => {
  const store = createStore();
  const valid = store.getSnapshot().result;
  // 非法 JSON
  assert.throws(() => store.applyEdits({
    Q: 'not json',
    K: JSON.stringify(good.K),
    V: JSON.stringify(good.V),
  }), /JSON 解析失败/);
  store.reportError('Q JSON 解析失败');
  assert.equal(store.getSnapshot().result, valid);

  // JSON 合法但含非有限数
  const nanQ = JSON.parse(JSON.stringify(good.Q));
  nanQ[0][1][1] = 1 / 0;
  assert.throws(() => store.applyEdits({
    Q: JSON.stringify(nanQ),
    K: JSON.stringify(good.K),
    V: JSON.stringify(good.V),
  }), /非有限数/);
  store.reportError('Q 包含非有限数');
  assert.equal(store.getSnapshot().result, valid);

  // 形状冲突: V 少一个位置
  const wrongV = good.V.map((m) => m.slice(0, 3));
  assert.throws(() => store.applyEdits({
    Q: JSON.stringify(good.Q),
    K: JSON.stringify(good.K),
    V: JSON.stringify(wrongV),
  }), /位置数|形状/);
  store.reportError('V 形状冲突');
  assert.equal(store.getSnapshot().result, valid);

  // 恢复: 合法编辑成功并真正生效
  const okSnap = store.applyEdits({
    Q: JSON.stringify(good.Q),
    K: JSON.stringify(good.K),
    V: JSON.stringify(good.V),
  });
  assert.equal(okSnap.error, null);
  assert.equal(okSnap.result.config.n, 4);
});

test('订阅器在参数与编辑后收到新快照', () => {
  const store = createStore();
  const calls = [];
  store.subscribe((snap) => calls.push(snap.params.numBlocks));
  store.setParams({ numBlocks: 2 });
  store.reportError('x');
  assert.deepEqual(calls, [2, 2]);
});
