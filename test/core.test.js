'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  phi,
  featureMapTensor,
  partitionBlocks,
  softmaxStable,
  computeWorkbench,
} = require('../src/core.js');

const approx = (actual, expected, tol = 1e-9) => {
  if (!Number.isFinite(expected)) {
    assert.equal(actual, expected);
  } else {
    assert.ok(Math.abs(actual - expected) <= tol,
      `期望 ${expected}, 实际 ${actual}, 容差 ${tol}`);
  }
};


// 独立复算的小型案例: 1 头 3 位置, dk=2, dv=2
const Q = [[[0, 2], [1, 1], [3, 0]]];
const K = [[[1, 0], [2, 0], [0, 1]]];
const V = [[[1, 1], [2, 0], [0, 3]]];
const SCALE = 1 / Math.sqrt(2);

test('特征映射固定为 max(x,0)+1', () => {
  assert.equal(phi(2), 3);
  assert.equal(phi(-5), 1);
  assert.equal(phi(0), 1);
  assert.deepEqual(featureMapTensor([[[-1, 2]]]), [[[1, 3]]]);
});

test('区间划分连续且覆盖全部位置', () => {
  const blocks = partitionBlocks(4, 3);
  assert.deepEqual(blocks.map((b) => [b.start, b.end]), [[0, 2], [2, 3], [3, 4]]);
  assert.equal(blocks[0].end, blocks[1].start);
  assert.equal(blocks[2].end, 4);
});

test('稳定 softmax 对大数值不溢出且和为 1', () => {
  const w = softmaxStable([1000, 1001, 999]);
  approx(w[0], 0.24472847105479764);
  approx(w[1], 0.6652409557748218);
  approx(w[2], 0.09003057317038046);
  approx(w[0] + w[1] + w[2], 1, 1e-12);
});

test('估计分数: phi(q) 与区间 phi(k) 均值的点积, 按区间和归一化', () => {
  const r = computeWorkbench(Q, K, V, { numBlocks: 2, budget: 1 });
  // 区间: b0=[0,2) 均值 phi(K0)=[2,1] 与 phi(K1)=[3,1] 的均值 [2.5,1]; b1=[2,3) 取 phi(K2)=[1,2]
  // phi(Q0)=[1,3]: 5.5 与 7
  approx(r.estRaw[0][0][0], 5.5);
  approx(r.estRaw[0][0][1], 7);
  approx(r.estNorm[0][0][0], 5.5 / 12.5);
  approx(r.estNorm[0][0][1], 7 / 12.5);
  // phi(Q2)=[4,1]: 11 与 6
  approx(r.estRaw[0][2][0], 11);
  approx(r.estRaw[0][2][1], 6);
  approx(r.estNorm[0][2][1], 6 / 17);
  // 每个查询的归一化分数和为 1
  for (let i = 0; i < 3; i++) {
    approx(r.estNorm[0][i][0] + r.estNorm[0][i][1], 1);
  }
});

test('掩码: 选中最高分区间并映射回原位置, 其余位置掩码为 0', () => {
  const r = computeWorkbench(Q, K, V, { numBlocks: 2, budget: 1 });
  // q0: b1 分数高 -> 块 1 -> 位置 2
  assert.deepEqual(r.selectedBlocks[0][0], [1]);
  assert.deepEqual(r.retained[0][0], [2]);
  // q1: 9/13.5 > 4.5/13.5 -> b0 -> 位置 0,1
  assert.deepEqual(r.retained[0][1], [0, 1]);
  // q2: 11/17 > 6/17 -> b0 -> 位置 0,1
  assert.deepEqual(r.retained[0][2], [0, 1]);
  // 稀疏权重只在保留位置非零
  assert.equal(r.sparseWeights[0][0][0], 0);
  approx(r.sparseWeights[0][0][2], 1);
});

test('因果模式: 区间估计不读未来键, 首个查询只用自身所在块的前缀', () => {
  const r = computeWorkbench(Q, K, V, { numBlocks: 2, budget: 2, causal: true });
  // q0 可见键只有 k0; b0=[0,2) 在估计时只取 k0
  approx(r.estRaw[0][0][0], 1 * 2 + 3 * 1); // 5, 不含 k1
  assert.equal(r.estNorm[0][0][1], 0); // b1 完全在未来
  assert.deepEqual(r.retained[0][0], [0]);
  // q1 可见 k0,k1: phi(Q1)=[2,2], 均值 [2.5,1] => 7
  approx(r.estRaw[0][1][0], 7);
  assert.equal(r.estNorm[0][1][1], 0);
  assert.deepEqual(r.retained[0][1], [0, 1]);
  // q2: b0 两键全可见均值点积 11, b1 取 k2 为 6
  approx(r.estRaw[0][2][0], 11);
  approx(r.estRaw[0][2][1], 6);
});

test('因果稠密/稀疏: 行内独立按 softmax 计算, 单行数据精确可复算', () => {
  const r = computeWorkbench(Q, K, V, { numBlocks: 2, budget: 2, causal: true });
  // q0 只看到 k0 => 输出 = V0, 前缀权重数组长度为 1
  assert.equal(r.denseWeights[0][0].length, 1);
  approx(r.denseWeights[0][0][0], 1);
  approx(r.denseOut[0][0][0], 1);
  approx(r.denseOut[0][0][1], 1);
  assert.deepEqual(r.retained[0][0], [0]);
  approx(r.sparseOut[0][0][0], 1);
  approx(r.sparseOut[0][0][1], 1);
  // q2 保留全部可见位置 => 稀疏与稠密一致, 行误差为 0
  const pe = r.metrics.perRowErrors[0][2];
  approx(pe.maxAbs, 0, 1e-12);
  approx(pe.relativeL2, 0, 1e-12);
});

test('非因果稠密注意力按缩放点积 softmax 独立复算', () => {
  const r = computeWorkbench(Q, K, V, { numBlocks: 2, budget: 1 });
  // q0 logits = scale * [0*1+2*0, 0*2+2*0, 0*0+2*1] = scale*[0,0,2]
  const e2 = Math.exp(SCALE * 2);
  const w2 = e2 / (2 + e2);
  const w0 = 1 / (2 + e2);
  approx(r.denseWeights[0][0][0], w0);
  approx(r.denseWeights[0][0][1], w0);
  approx(r.denseWeights[0][0][2], w2);
  // 稠密输出: w0*(V0+V1)+w2*V2
  approx(r.denseOut[0][0][0], w0 * (1 + 2) + w2 * 0);
  approx(r.denseOut[0][0][1], w0 * (1 + 0) + w2 * 3);
  // 稀疏只保留 k2: 输出 V2=[0,3], 误差真实非零
  approx(r.sparseOut[0][0][0], 0);
  approx(r.sparseOut[0][0][1], 3);
  assert.ok(r.metrics.maxAbsError > 0.01);
  assert.ok(r.metrics.relativeL2Error > 0);
  assert.ok(r.metrics.relativeL2Error < 1);
});

test('零预算: 零向量输出与空掩码状态', () => {
  const r = computeWorkbench(Q, K, V, { numBlocks: 2, budget: 0 });
  assert.equal(r.config.budget, 0);
  assert.equal(r.metrics.totalRetained, 0);
  assert.deepEqual(r.sparseOut, [[[0, 0], [0, 0], [0, 0]]]);
  assert.equal(r.metrics.emptyRows.length, 3);
  // 稠密仍被正常计算, 误差度量存在
  assert.ok(r.metrics.maxAbsError >= 0);
  assert.ok(Number.isFinite(r.metrics.relativeL2Error));
});

test('预算超上限截断到区间数, 映射后位置去重', () => {
  const r = computeWorkbench(Q, K, V, { numBlocks: 2, budget: 99 });
  assert.equal(r.config.requestedBudget, 99);
  assert.equal(r.config.budget, 2);
  // 非因果下保留全部位置, 且无重复
  assert.deepEqual(r.retained[0][0], [0, 1, 2]);
  assert.equal(new Set(r.retained[0][0]).size, 3);
  // 全保留 => 稀疏等于稠密
  approx(r.metrics.maxAbsError, 0, 1e-12);
  approx(r.metrics.relativeL2Error, 0, 1e-12);
});

test('分数并列按头与位置稳定排序', () => {
  // 所有 phi 值相同 => 所有区间分数相等, 预算 1 应选中块 0
  const q = [[[0, 0], [0, 0], [0, 0], [0, 0]]];
  const k = [[[0, 0], [0, 0], [0, 0], [0, 0]]];
  const v = [[[1], [2], [3], [4]]];
  const r = computeWorkbench(q, k, v, { numBlocks: 4, budget: 1 });
  assert.deepEqual(r.selectedBlocks[0][0], [0]);
  assert.deepEqual(r.retained[0][0], [0]);
});

test('跨头共享预算: 同一查询只在一个头上消费预算, 与逐头模式不同', () => {
  // n=2, 每块一个位置; 头0 两个区间同分, 头1 的块0 分数更高
  const q2 = [
    [[0], [0]],
    [[0], [0]],
  ];
  const k2 = [
    [[10], [10]],
    [[10], [0]],
  ];
  const v2 = [
    [[1], [2]],
    [[3], [4]],
  ];
  const shared = computeWorkbench(q2, k2, v2, { numBlocks: 2, budget: 1, sharedBudget: true });
  // 最高分为头1块0 (11/12), 头0 该行无选中
  assert.deepEqual(shared.selectedBlocks[0][0], []);
  assert.deepEqual(shared.selectedBlocks[1][0], [0]);
  assert.deepEqual(shared.retained[0][0], []);
  assert.deepEqual(shared.retained[1][0], [0]);
  // q0 行在共享模式下只保留 1 个位置
  assert.equal(shared.retained[0][0].length + shared.retained[1][0].length, 1);

  const perHead = computeWorkbench(q2, k2, v2, { numBlocks: 2, budget: 1, sharedBudget: false });
  assert.deepEqual(perHead.selectedBlocks[0][0], [0]);
  assert.deepEqual(perHead.selectedBlocks[1][0], [0]);
  // 同样的 q0 行在逐头模式下两个头各保留 1 个
  assert.equal(perHead.retained[0][0].length + perHead.retained[1][0].length, 2);
});

test('参考输出范数为零: 全零 V 时相对误差为 0; 稠密抵消但稀疏非零时为 Infinity', () => {
  const zeroV = [[[0], [0], [0]]];
  const r0 = computeWorkbench(Q, K, zeroV, { numBlocks: 2, budget: 1 });
  assert.equal(r0.metrics.maxAbsError, 0);
  assert.equal(r0.metrics.relativeL2Error, 0);
  assert.ok(r0.metrics.allReferenceZero);
  assert.equal(r0.metrics.referenceZeroRows.length, 3);

  // q0 与所有键正交 => 稠密均匀 [0.5,0.5], 取 V=[1,-1] => 参考 0
  // 估计分数 phi 下两块不等, 稀疏只选一块 => 稀疏非零
  const qe = [[[0, 1], [0, 1]]];
  const ke = [[[1, 0], [0, 0]]];
  const ve = [[[1], [-1]]];
  const re = computeWorkbench(qe, ke, ve, { numBlocks: 2, budget: 1 });
  approx(re.denseWeights[0][0][0], 0.5, 1e-12);
  approx(re.denseWeights[0][0][1], 0.5, 1e-12);
  approx(re.denseOut[0][0][0], 0, 1e-12);
  assert.notEqual(re.sparseOut[0][0][0], 0);
  assert.equal(re.metrics.relativeL2Error, Infinity);
});
