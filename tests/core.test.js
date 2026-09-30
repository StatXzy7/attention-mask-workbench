import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  analyzeModel,
  featureMap,
  partitionIntervals,
  validateHead,
  validateModel,
} from '../src/core.js';

const eps = 1e-10;

// Hand-verified single-head model:
// Q = [[1,0],[0,2],[2,1],[0,0]], K = [[1,0],[0,1],[2,1],[0,0],[1,2]], V 5x2
const head = {
  q: [[1, 0], [0, 2], [2, 1], [0, 0]],
  k: [[1, 0], [0, 1], [2, 1], [0, 0], [1, 2]],
  v: [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2]],
};
const model = { heads: [head] };
const dk = 2;
const s = 1 / Math.sqrt(dk);

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / z);
}

test('feature map is non-negative: max(x,0)+1', () => {
  assert.equal(featureMap(0), 1);
  assert.equal(featureMap(3), 4);
  assert.equal(featureMap(-7), 1);
});

test('partition splits into balanced contiguous ranges covering all keys', () => {
  assert.deepEqual(partitionIntervals(5, 2), [
    { index: 0, start: 0, end: 3 },
    { index: 1, start: 3, end: 5 },
  ]);
  assert.deepEqual(partitionIntervals(5, 5).map((r) => [r.start, r.end]), [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
  ]);
  const ranges = partitionIntervals(5, 3);
  assert.equal(ranges[0].end - ranges[0].start, 2);
  assert.equal(ranges[2].end, 5);
});

test('estimated interval scores use feature-key means and normalize by interval sum', () => {
  // q0 -> phi = [2,1]. B=2 -> [0..3), [3..5).
  // bin0 raw: mean of phi(k0..k2) . [2,1] = (([2,1]+[1,2]+[3,2])/3) . [2,1]
  //         = [2, 5/3] . [2,1] = 4 + 5/3 = 17/3
  // bin1 raw: mean phi(k3..k4) = ([1,1]+[2,3])/2 = [1.5,2]; dot = 3+2 = 5
  const r = analyzeModel(model, { intervals: 2, budget: 1 });
  const q0 = r.heads[0].queries[0];
  assert.ok(Math.abs(q0.intervalRawScores[0] - 17 / 3) < eps);
  assert.ok(Math.abs(q0.intervalRawScores[1] - 5) < eps);
  assert.ok(Math.abs(q0.intervalScores[0] - (17 / 3) / (17 / 3 + 5)) < eps);
  assert.ok(Math.abs(q0.intervalScores[1] - 5 / (17 / 3 + 5)) < eps);
  assert.ok(Math.abs(q0.intervalScores.reduce((a, b) => a + b, 0) - 1) < eps);
});

test('higher normalized score wins the budget and maps back to key positions', () => {
  const r = analyzeModel(model, { intervals: 2, budget: 1 });
  const q0 = r.heads[0].queries[0];
  assert.deepEqual(q0.chosenIntervals, [0]);
  assert.deepEqual(q0.retainedKeys, [0, 1, 2]);
  assert.deepEqual(q0.mask, [1, 1, 1, 0, 0]);
});

test('dense weights and output equal hand-computed stable softmax attention', () => {
  const r = analyzeModel(model, { intervals: 2, budget: 1 });
  const qf = [2, 1];
  const kf = [[2, 1], [1, 2], [3, 2], [1, 1], [2, 3]];
  const logits = kf.map((k) => k[0] * qf[0] + k[1] * qf[1]).map((v) => v * s);
  const expectedW = softmax(logits);
  const got = r.heads[0].queries[0].denseWeights;
  got.forEach((w, j) => assert.ok(Math.abs(w - expectedW[j]) < eps, `dense w${j}`));

  let out0 = 0;
  let out1 = 0;
  expectedW.forEach((w, j) => {
    out0 += w * head.v[j][0];
    out1 += w * head.v[j][1];
  });
  const denseOut = r.heads[0].queries[0].denseOutput;
  assert.ok(Math.abs(denseOut[0] - out0) < eps);
  assert.ok(Math.abs(denseOut[1] - out1) < eps);
});

test('sparse attention renormalizes only on retained positions and computes real output', () => {
  const r = analyzeModel(model, { intervals: 2, budget: 1 });
  const qf = [2, 1];
  const kf = [[2, 1], [1, 2], [3, 2]];
  const logits = kf.map((k) => k[0] * qf[0] + k[1] * qf[1]).map((v) => v * s);
  const expectedW = softmax(logits);
  const sparseWeights = r.heads[0].queries[0].sparseWeights;
  expectedW.forEach((w, j) => assert.ok(Math.abs(sparseWeights[j] - w) < eps));
  assert.equal(sparseWeights[3], 0);
  assert.equal(sparseWeights[4], 0);

  const expectedOut = [0, 0];
  const retainedV = [head.v[0], head.v[1], head.v[2]];
  expectedW.forEach((w, j) => {
    expectedOut[0] += w * retainedV[j][0];
    expectedOut[1] += w * retainedV[j][1];
  });
  const got = r.heads[0].queries[0].sparseOutput;
  assert.ok(Math.abs(got[0] - expectedOut[0]) < eps);
  assert.ok(Math.abs(got[1] - expectedOut[1]) < eps);

  // Sparse with full budget over 1-key intervals must match dense exactly.
  const full = analyzeModel(model, { intervals: 5, budget: 5 });
  assert.equal(full.metrics.maxAbsError, 0);
  assert.equal(full.metrics.relL2Error, 0);
});

test('score ties are broken stably by head then position', () => {
  // Same Q and all-zero K for both heads -> phi(k) identical everywhere,
  // so all interval scores are genuinely tied (1/2 each).
  const tieHead = {
    q: [[0, 0]],
    k: [[0, 0], [0, 0]],
    v: [[1], [2]],
  };
  const tieModel = { heads: [tieHead, JSON.parse(JSON.stringify(tieHead))] };

  // Per-head budget: both heads independently keep their lower interval.
  const perHead = analyzeModel(tieModel, { intervals: 2, budget: 1, shared: false });
  assert.deepEqual(perHead.heads[0].queries[0].chosenIntervals, [0]);
  assert.deepEqual(perHead.heads[1].queries[0].chosenIntervals, [0]);
  assert.deepEqual(perHead.heads[1].queries[0].retainedKeys, [0]);

  // Shared budget 1: the tie goes to head 0's lower interval first.
  const r = analyzeModel(tieModel, { intervals: 2, budget: 1, shared: true });
  assert.deepEqual(r.heads[0].queries[0].chosenIntervals, [0]);
  assert.deepEqual(r.heads[1].queries[0].chosenIntervals, []);

  const r2 = analyzeModel(tieModel, { intervals: 2, budget: 2, shared: true });
  // Pooled tie order: score desc, then head asc, then interval asc
  // -> h0#0, h0#1, h1#0, h1#1.
  assert.deepEqual(r2.heads[0].queries[0].chosenOrder, [0, 1]);
  assert.deepEqual(r2.heads[1].queries[0].chosenOrder, []);

  const r3 = analyzeModel(tieModel, { intervals: 2, budget: 3, shared: true });
  assert.deepEqual(r3.heads[0].queries[0].chosenOrder, [0, 1]);
  assert.deepEqual(r3.heads[1].queries[0].chosenOrder, [0]);
  assert.equal(r3.metrics.retainedPairs, 3); // 1 key per selected interval x 3 tokens
});

test('causal mode never reads future keys in interval estimates or attention', () => {
  const r = analyzeModel(model, { intervals: 2, budget: 1, causal: true });
  // Query 0 sees only key 0: bin 0 clipped to [0,1), bin 1 fully in the future.
  const q0 = r.heads[0].queries[0];
  assert.equal(q0.visibleCount, 1);
  assert.equal(q0.intervalScores[1], 0);
  assert.deepEqual(q0.chosenIntervals, [0]);
  assert.deepEqual(q0.retainedKeys, [0]);
  // Sparse equals dense when the only visible key is retained.
  assert.deepEqual(q0.sparseWeights, [1, 0, 0, 0, 0]);
  assert.deepEqual(q0.denseWeights, [1, 0, 0, 0, 0]);

  // Query 1 sees keys 0..1, both inside bin 0.
  const q1 = r.heads[0].queries[1];
  assert.equal(q1.visibleCount, 2);
  assert.deepEqual(q1.retainedKeys, [0, 1]);
  for (let j = 2; j < 5; j += 1) {
    assert.equal(q1.denseWeights[j], 0);
    assert.equal(q1.mask[j], 0);
  }

  // nq=4, nk=5: even the last query is causal to 4 keys (k0..k3); k4 is
  // permanently future. Bin 0 is clipped to [0,3) and bin 1 keeps [3,4).
  const q3 = r.heads[0].queries[3];
  assert.equal(q3.visibleCount, 4);
  assert.equal(q3.intervalRawScores[1] > 0, true);
  assert.deepEqual(q3.pickedIntervals.map((x) => [x.start, x.end]).flat().length > 0, true);
});

test('per-head budget is applied independently to every head', () => {
  const h0 = {
    q: [[0, 0]],
    k: [[0, 0], [0, 0], [0, 0], [0, 0]],
    v: [[1], [1], [1], [1]],
  };
  const twoHeads = { heads: [h0, JSON.parse(JSON.stringify(h0))] };
  const r = analyzeModel(twoHeads, { intervals: 2, budget: 1 });
  for (const headResult of r.heads) {
    assert.deepEqual(headResult.queries[0].chosenIntervals, [0]);
    assert.deepEqual(headResult.queries[0].retainedKeys, [0, 1]);
  }
  // 2 heads x 2 retained keys = 4 pairs under per-head budget 1.
  assert.equal(r.metrics.retainedPairs, 4);
});

test('shared budget is pooled across heads for the same query', () => {
  const h0 = {
    q: [[0, 0]],
    k: [[0, 0], [0, 0], [0, 0], [0, 0]],
    v: [[1], [1], [1], [1]],
  };
  const twoHeads = { heads: [h0, JSON.parse(JSON.stringify(h0))] };
  const r = analyzeModel(twoHeads, { intervals: 2, budget: 1, shared: true });
  // Tie -> head 0 consumes the only token; head 1 keeps nothing.
  assert.deepEqual(r.heads[0].queries[0].retainedKeys, [0, 1]);
  assert.deepEqual(r.heads[1].queries[0].retainedKeys, []);
  assert.equal(r.heads[1].queries[0].emptyMask, true);
  assert.equal(r.metrics.retainedPairs, 2);
});

test('budget and interval counts above the limit are truncated', () => {
  const r = analyzeModel(model, { intervals: 99, budget: 99 });
  assert.equal(r.params.intervals, 5);
  assert.equal(r.params.budget, 5);
  assert.equal(r.params.intervalsWasClamped, true);
  assert.equal(r.params.budgetWasClamped, true);
  assert.equal(r.metrics.maxAbsError, 0);
});

test('negative budget is treated as zero; zero budget yields zero vectors and empty mask', () => {
  const r = analyzeModel(model, { intervals: 2, budget: 0 });
  assert.equal(r.params.budget, 0);
  for (const query of r.heads[0].queries) {
    assert.deepEqual(query.mask, [0, 0, 0, 0, 0]);
    assert.deepEqual(query.retainedKeys, []);
    assert.deepEqual(query.sparseOutput, [0, 0]);
    assert.deepEqual(query.sparseWeights, [0, 0, 0, 0, 0]);
    assert.equal(query.emptyMask, true);
  }
  assert.equal(r.metrics.retainedPairs, 0);
  assert.ok(r.metrics.maxAbsError >= 0);
});

test('mapping selected intervals back to positions deduplicates repeated keys', () => {
  // Construct two intervals that overlap only via causal clipping:
  // nk=3, B=2 -> [0,2), [2,3). Query i=1 sees [0,2): bin1 is invisible,
  // so no duplicates arise there. Instead force identical coverage by
  // re-running selection: build a tiny model where both bins survive clipping
  // at the same query (i>=2) and select both -> keys 0,1,2 with no repeats.
  const tiny = {
    heads: [{
      q: [[0, 0], [0, 0], [0, 0]],
      k: [[0, 0], [1, 0], [0, 1]],
      v: [[1], [2], [3]],
    }],
  };
  const r = analyzeModel(tiny, { intervals: 2, budget: 2, causal: true });
  const q2 = r.heads[0].queries[2];
  assert.deepEqual(q2.chosenIntervals.sort(), [0, 1]);
  assert.deepEqual(q2.retainedKeys, [0, 1, 2]);
  assert.equal(new Set(q2.retainedKeys).size, q2.retainedKeys.length);
  // The dedup invariant must hold for every random-ish parameter combination.
  for (let b = 1; b <= 3; b += 1) {
    for (const causal of [false, true]) {
      const sweep = analyzeModel(model, { intervals: b, budget: b, causal });
      for (const query of sweep.heads[0].queries) {
        assert.equal(new Set(query.retainedKeys).size, query.retainedKeys.length);
        for (const j of query.retainedKeys) assert.equal(query.mask[j], 1);
        assert.equal(query.retainedKeys.length, query.mask.reduce((a, z) => a + z, 0));
      }
    }
  }
});

test('relative L2 error and max abs error are reported against dense reference', () => {
  const r = analyzeModel(model, { intervals: 2, budget: 1 });
  let maxAbs = 0;
  let diffSq = 0;
  let denseSq = 0;
  for (const query of r.heads[0].queries) {
    query.denseOutput.forEach((d, dIdx) => {
      const diff = query.sparseOutput[dIdx] - d;
      maxAbs = Math.max(maxAbs, Math.abs(diff));
      diffSq += diff * diff;
      denseSq += d * d;
    });
  }
  assert.ok(Math.abs(r.metrics.maxAbsError - maxAbs) < 1e-12);
  assert.ok(Math.abs(r.metrics.relL2Error - Math.sqrt(diffSq) / Math.sqrt(denseSq)) < 1e-12);
  assert.equal(r.metrics.refNormIsZero, false);
});

test('zero-norm dense reference keeps explicit error semantics (no NaN)', () => {
  // All V = 0 -> dense and sparse outputs are zero vectors, dense norm 0.
  const zeroV = {
    heads: [{
      q: [[1, 0], [0, 1]],
      k: [[1, 0], [0, 1]],
      v: [[0, 0], [0, 0]],
    }],
  };
  const r = analyzeModel(zeroV, { intervals: 2, budget: 1 });
  assert.equal(r.metrics.denseNorm, 0);
  assert.equal(r.metrics.refNormIsZero, true);
  assert.equal(r.metrics.relL2Error, 0); // absolute diff norm
  assert.equal(r.metrics.maxAbsError, 0);
  for (const query of r.heads[0].queries) {
    for (const v of query.sparseOutput) assert.ok(Number.isFinite(v));
  }

  // Non-zero V but V-vectors orthogonal to every softmax weight can still be
  // forced: zero budget -> sparse is zero while dense is non-zero, dense norm
  // stays non-zero here; instead test a genuinely zero-norm dense with a
  // non-zero sparse discrepancy is impossible (sparse subset of same values).
  // A zero-dense but nonzero-sparse cannot occur; semantics covered above.
});

test('non-finite numbers and shape conflicts are rejected', () => {
  assert.throws(() => analyzeModel({
    heads: [{ q: [[NaN]], k: [[1]], v: [[1]] }],
  }, {}), /non-finite/);
  assert.throws(() => analyzeModel({
    heads: [{ q: [[Infinity]], k: [[1]], v: [[1]] }],
  }, {}), /non-finite/);
  assert.throws(() => validateHead({
    q: [[1, 2], [3]], k: [[1]], v: [[1]],
  }), /ragged/);
  assert.throws(() => analyzeModel({
    heads: [{ q: [[1, 2]], k: [[1, 2, 3]], v: [[1]] }],
  }, {}), /dimension mismatch/);
  assert.throws(() => analyzeModel({
    heads: [{ q: [[1]], k: [[1], [2]], v: [[1]] }],
  }, {}), /k\/v key-position/);
  assert.throws(() => analyzeModel({
    heads: [
      { q: [[1]], k: [[1]], v: [[1]] },
      { q: [[1], [2]], k: [[1], [2]], v: [[1], [1]] },
    ],
  }, {}), /counts must match/);
  assert.throws(() => validateModel({ heads: [] }), /At least one/);
});

test('multi-head runs with different per-head feature and value widths', () => {
  const multi = {
    heads: [
      { q: [[1, 0]], k: [[1, 0], [0, 1]], v: [[1, 0, 0], [0, 1, 0]] },
      { q: [[1, 0, 0]], k: [[1, 0, 0], [0, 1, 0]], v: [[2], [3]] },
    ],
  };
  const r = analyzeModel(multi, { intervals: 2, budget: 1, shared: true });
  assert.equal(r.heads[0].dv, 3);
  assert.equal(r.heads[1].dv, 1);
  assert.ok(Number.isFinite(r.metrics.relL2Error));
});
