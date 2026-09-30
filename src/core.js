// Core sparse-attention workbench logic.
// Runs unchanged in Node (tests/server build checks) and in the browser.
// No third-party dependencies.

export const FEATURE_MAP_NAME = 'max(x, 0) + 1';

// Non-negative feature map, applied elementwise to queries and keys.
export function featureMap(x) {
  return Math.max(x, 0) + 1;
}

export function mapMatrix(rows) {
  return rows.map((row) => row.map(featureMap));
}

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function fail(message) {
  const err = new Error(message);
  err.name = 'WorkbenchValidationError';
  return err;
}

// Validate one head: { q: number[][], k: number[][], v: number[][] }.
export function validateHead(head, indexLabel) {
  const where = indexLabel === undefined ? '' : ` (head ${indexLabel})`;
  if (head === null || typeof head !== 'object') {
    throw fail(`Head${where} must be an object with q, k, v matrices.`);
  }
  for (const name of ['q', 'k', 'v']) {
    const matrix = head[name];
    if (!Array.isArray(matrix) || matrix.length === 0) {
      throw fail(`Matrix "${name}"${where} must be a non-empty array of rows.`);
    }
    for (const row of matrix) {
      if (!Array.isArray(row) || row.length === 0) {
        throw fail(`Every row of "${name}"${where} must be a non-empty array.`);
      }
      for (const value of row) {
        if (!isFiniteNumber(value)) {
          throw fail(`Matrix "${name}"${where} contains a non-finite or non-numeric value.`);
        }
      }
    }
    const width = matrix[0].length;
    for (const row of matrix) {
      if (row.length !== width) {
        throw fail(`Matrix "${name}"${where} has ragged rows (expected width ${width}).`);
      }
    }
  }
  const nq = head.q.length;
  const nk = head.k.length;
  const dk = head.q[0].length;
  const dv = head.v[0].length;
  if (head.k[0].length !== dk) {
    throw fail(`q/k feature dimension mismatch${where}: q has ${dk}, k has ${head.k[0].length}.`);
  }
  if (head.v.length !== nk) {
    throw fail(`k/v key-position count mismatch${where}: k has ${nk} rows, v has ${head.v.length}.`);
  }
  return { nq, nk, dk, dv };
}

export function validateModel(model) {
  if (model === null || typeof model !== 'object' || !Array.isArray(model.heads)) {
    throw fail('Model must be an object with a "heads" array.');
  }
  if (model.heads.length === 0) {
    throw fail('At least one attention head is required.');
  }
  let shape = null;
  const shapes = [];
  model.heads.forEach((head, h) => {
    const s = validateHead(head, h);
    shapes.push(s);
    if (shape === null) shape = s;
    if (s.nq !== shape.nq || s.nk !== shape.nk) {
      throw fail(
        `Head ${h} has ${s.nq} queries and ${s.nk} keys but head 0 has ` +
          `${shape.nq} queries and ${shape.nk} keys; query/key position counts must match across heads.`,
      );
    }
  });
  return { heads: shapes, nq: shape.nq, nk: shape.nk, dv: shape.dv };
}

function sanitizeParams(rawParams, nk, nq) {
  const p = rawParams || {};
  let intervals = Number(p.intervals);
  if (!Number.isFinite(intervals) || intervals < 1) intervals = 1;
  intervals = Math.floor(Math.min(intervals, nk));

  let budget = Number(p.budget);
  if (!Number.isFinite(budget) || budget < 0) budget = 0;
  budget = Math.floor(budget);

  return {
    intervals,
    rawBudget: budget,
    // Effective budget is finalized later (per-head cap = intervals;
    // shared cap = number of heads * intervals).
    budget,
    causal: Boolean(p.causal),
    shared: Boolean(p.shared),
    nk,
    nq,
    intervalsWasClamped: Number.isFinite(Number(p.intervals)) && Number(p.intervals) > nk,
    budgetWasClamped: false,
  };
}

// Partition length nk into at most numIntervals contiguous ranges.
// Sizes differ by at most one; larger ranges come first.
export function partitionIntervals(nk, numIntervals) {
  const count = Math.max(1, Math.min(numIntervals, nk));
  const base = Math.floor(nk / count);
  let extra = nk % count;
  const ranges = [];
  let start = 0;
  for (let i = 0; i < count; i += 1) {
    const size = base + (extra > 0 ? 1 : 0);
    if (extra > 0) extra -= 1;
    ranges.push({ index: i, start, end: start + size });
    start += size;
  }
  return ranges;
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) total += a[i] * b[i];
  return total;
}

// Numerically stable masked softmax attention for one query.
// allowedKeys lists which key positions are visible; the softmax is renormalized
// over exactly those positions (the sparse attention thus compares directly
// against the dense reference restricted to the mask).
function attentionOnKeys(queryF, keyF, values, allowedKeys, scale) {
  const logits = allowedKeys.map((j) => dot(queryF, keyF[j]) * scale);
  let maxLogit = -Infinity;
  for (const value of logits) if (value > maxLogit) maxLogit = value;
  const weights = logits.map((value) => Math.exp(value - maxLogit));
  let denom = 0;
  for (const value of weights) denom += value;
  const dv = values[0].length;
  const out = new Array(dv).fill(0);
  const weightByKey = new Array(allowedKeys.length);
  for (let t = 0; t < allowedKeys.length; t += 1) {
    const w = denom > 0 ? weights[t] / denom : 0;
    weightByKey[t] = w;
    const row = values[allowedKeys[t]];
    for (let d = 0; d < dv; d += 1) out[d] += w * row[d];
  }
  // Full-length weight vector with 0 at disallowed positions, for visualization.
  const fullWeights = new Array(keyF.length).fill(0);
  allowedKeys.forEach((keyIndex, t) => {
    fullWeights[keyIndex] = weightByKey[t];
  });
  return { output: out, weights: fullWeights };
}

// Scores that differ only by floating-point noise are treated as tied so the
// documented stable (head, position) ordering remains observable.
const SCORE_TIE_EPS = 1e-12;
function compareScoreDesc(aScore, bScore, tieBreak) {
  const diff = bScore - aScore;
  if (Math.abs(diff) <= SCORE_TIE_EPS * Math.max(1, Math.abs(aScore), Math.abs(bScore))) {
    return tieBreak();
  }
  return diff;
}

function allowedKeyRange(i, nk, causal) {
  const end = causal ? Math.min(nk, i + 1) : nk;
  return end; // visible keys are 0 .. end-1
}

function buildCandidates(head, headF, ranges, i, causal, nk) {
  const { q: qF, k: kF } = headF;
  const queryF = qF[i];
  const visibleEnd = allowedKeyRange(i, nk, causal);
  const candidates = [];
  for (const range of ranges) {
    const start = Math.max(range.start, 0);
    const end = Math.min(range.end, visibleEnd);
    if (end <= start) continue; // no visible key in this interval (causal future)
    let score = 0;
    for (let j = start; j < end; j += 1) score += dot(queryF, kF[j]);
    score /= end - start; // mean feature key over the visible slice
    candidates.push({
      intervalIndex: range.index,
      start,
      end, // clipped, visible [start, end)
      fullStart: range.start,
      fullEnd: range.end,
      rawScore: score,
    });
  }
  const sum = candidates.reduce((acc, c) => acc + c.rawScore, 0);
  for (const c of candidates) {
    c.normScore = sum > 0 ? c.rawScore / sum : 0;
  }
  // Stable ordering: score desc, then interval index asc.
  candidates.sort((a, b) =>
    compareScoreDesc(a.normScore, b.normScore, () => a.intervalIndex - b.intervalIndex),
  );
  return candidates;
}

// Choose intervals under per-head budgets. Tie-break inside sort is stable by
// (head, interval position): iterate heads in ascending order.
function selectPerHead(queryCandidates, budget) {
  // Each head receives its own independent budget; intervals never collide
  // across heads, so there is deliberately no cross-head dedup here.
  const details = queryCandidates.map((candidates) =>
    candidates.slice(0, Math.min(budget, candidates.length)).map((c) => c.intervalIndex),
  );
  return { details };
}

// Shared budget across heads for one query. Pooled entries keep a stable
// ordering: score desc, then head asc, then interval position asc.
function selectShared(queryCandidates, budget) {
  const pooled = [];
  queryCandidates.forEach((candidates, h) => {
    candidates.forEach((c) => pooled.push({ h, intervalIndex: c.intervalIndex, normScore: c.normScore }));
  });
  // Stable tie-break: score desc, then head asc, then interval asc.
  pooled.sort((a, b) =>
    compareScoreDesc(a.normScore, b.normScore, () =>
      a.h !== b.h ? a.h - b.h : a.intervalIndex - b.intervalIndex,
    ),
  );
  const chosen = new Set();
  // Details preserve pooled selection order; callers sort for presentation.
  const details = queryCandidates.map(() => []);
  let spent = 0;
  for (const entry of pooled) {
    if (spent >= budget) break;
    // Dedup key is scoped per head: head0#0 and head1#0 are distinct picks.
    const key = `${entry.h}#${entry.intervalIndex}`;
    if (chosen.has(key)) continue;
    chosen.add(key);
    details[entry.h].push(entry.intervalIndex);
    spent += 1;
  }
  return { chosen, details };
}

function flattenHeads(perHeadArrays) {
  const flat = [];
  for (const arr of perHeadArrays) for (const v of arr) flat.push(v);
  return flat;
}

export function analyzeModel(inputModel, rawParams) {
  validateModel(inputModel);
  const modelInfo = validateModel(inputModel);
  const { nk, nq } = modelInfo;
  const params = sanitizeParams(rawParams, nk, nq);
  const ranges = partitionIntervals(nk, params.intervals);

  const headsOut = inputModel.heads.map((head, h) => {
    const f = { q: mapMatrix(head.q), k: mapMatrix(head.k), v: head.v };
    const scale = 1 / Math.sqrt(f.q[0].length);
    const perQuery = [];
    for (let i = 0; i < nq; i += 1) {
      const visibleEnd = allowedKeyRange(i, nk, params.causal);
      const denseAllowed = [];
      for (let j = 0; j < visibleEnd; j += 1) denseAllowed.push(j);
      const dense = attentionOnKeys(f.q[i], f.k, f.v, denseAllowed, scale);
      perQuery.push({
        candidates: buildCandidates(head, f, ranges, i, params.causal, nk),
        denseOutput: dense.output,
        denseWeights: dense.weights,
        visibleCount: denseAllowed.length,
      });
    }
    return { head: head, f, scale, perQuery };
  });

  // Effective caps: per-head selection caps at intervals; a shared pool caps
  // at the total number of (head, interval) candidates.
  let budgetWasClamped = false;
  if (params.shared) {
    const cap = inputModel.heads.length * params.intervals;
    budgetWasClamped = params.rawBudget > cap;
    params.budget = Math.min(params.rawBudget, cap);
  } else {
    budgetWasClamped = params.rawBudget > params.intervals;
    params.budget = Math.min(params.rawBudget, params.intervals);
  }
  params.budgetWasClamped = budgetWasClamped;

  for (let i = 0; i < nq; i += 1) {
    const queryCandidates = headsOut.map((hd) => hd.perQuery[i].candidates);
    const selection = params.shared
      ? selectShared(queryCandidates, params.budget)
      : selectPerHead(queryCandidates, params.budget);

    headsOut.forEach((hd, h) => {
      const slot = hd.perQuery[i];
      // chosenOrder preserves pooled selection order (stable tie-break);
      // chosenIntervals is sorted for deterministic mapping/rendering.
      const chosenOrder = selection.details[h].slice();
      const chosenIntervals = chosenOrder.slice().sort((a, b) => a - b);
      const chosenSet = new Set(chosenIntervals);

      // Full-length estimated-score vector over intervals (0 for invisible ones).
      const intervalScores = new Array(ranges.length).fill(0);
      for (const c of slot.candidates) {
        intervalScores[c.intervalIndex] = c.normScore;
      }
      const intervalRawScores = new Array(ranges.length).fill(0);
      for (const c of slot.candidates) {
        intervalRawScores[c.intervalIndex] = c.rawScore;
      }

      // Map chosen intervals back to original key positions and deduplicate.
      const keySet = new Set();
      const picked = [];
      for (const c of slot.candidates) {
        if (!chosenSet.has(c.intervalIndex)) continue;
        picked.push({ intervalIndex: c.intervalIndex, start: c.start, end: c.end });
        for (let j = c.start; j < c.end; j += 1) keySet.add(j);
      }
      picked.sort((a, b) => a.intervalIndex - b.intervalIndex);
      const retainedKeys = Array.from(keySet).sort((a, b) => a - b);

      const mask = new Array(nk).fill(0);
      for (const j of retainedKeys) mask[j] = 1;

      let sparseOutput;
      let sparseWeights;
      if (retainedKeys.length === 0 || params.budget === 0) {
        sparseOutput = new Array(hd.head.v[0].length).fill(0);
        sparseWeights = new Array(nk).fill(0);
      } else {
        const sparse = attentionOnKeys(hd.f.q[i], hd.f.k, hd.f.v, retainedKeys, hd.scale);
        sparseOutput = sparse.output;
        sparseWeights = sparse.weights;
      }

      slot.chosenIntervals = chosenIntervals;
      slot.chosenOrder = chosenOrder;
      slot.pickedIntervals = picked;
      slot.intervalScores = intervalScores;
      slot.intervalRawScores = intervalRawScores;
      slot.retainedKeys = retainedKeys;
      slot.mask = mask;
      slot.sparseOutput = sparseOutput;
      slot.sparseWeights = sparseWeights;
      slot.emptyMask = retainedKeys.length === 0;
    });
  }

  // Metrics over the concatenated multi-head outputs.
  const denseFlat = [];
  const sparseFlat = [];
  let retainedPairs = 0;
  let totalPairs = 0;
  let allowedPairs = 0;
  let anyEmptyMask = false;
  for (let i = 0; i < nq; i += 1) {
    for (const hd of headsOut) {
      const slot = hd.perQuery[i];
      denseFlat.push(...slot.denseOutput);
      sparseFlat.push(...slot.sparseOutput);
      retainedPairs += slot.retainedKeys.length;
      totalPairs += nk;
      allowedPairs += slot.visibleCount;
      if (slot.emptyMask) anyEmptyMask = true;
    }
  }
  let maxAbsError = 0;
  let diffSq = 0;
  let denseSq = 0;
  for (let t = 0; t < denseFlat.length; t += 1) {
    const diff = sparseFlat[t] - denseFlat[t];
    const abs = Math.abs(diff);
    if (abs > maxAbsError) maxAbsError = abs;
    diffSq += diff * diff;
    denseSq += denseFlat[t] * denseFlat[t];
  }
  const denseNorm = Math.sqrt(denseSq);
  const diffNorm = Math.sqrt(diffSq);
  const REL_L2_EPS = 1e-12;
  let relL2Error;
  let refNormIsZero;
  if (denseNorm <= REL_L2_EPS) {
    // Explicit semantics for a vanishing reference: report the absolute error
    // norm instead of dividing by zero, and flag the state.
    relL2Error = diffNorm;
    refNormIsZero = true;
  } else {
    relL2Error = diffNorm / denseNorm;
    refNormIsZero = false;
  }

  return {
    featureMap: FEATURE_MAP_NAME,
    params,
    model: { nq, nk, numHeads: inputModel.heads.length },
    heads: headsOut.map((hd, h) => ({
      headIndex: h,
      dk: hd.head.q[0].length,
      dv: hd.head.v[0].length,
      queries: hd.perQuery.map((slot) => ({
        intervalScores: slot.intervalScores,
        intervalRawScores: slot.intervalRawScores,
        chosenIntervals: slot.chosenIntervals,
        chosenOrder: slot.chosenOrder,
        pickedIntervals: slot.pickedIntervals,
        mask: slot.mask,
        retainedKeys: slot.retainedKeys,
        denseWeights: slot.denseWeights,
        sparseWeights: slot.sparseWeights,
        denseOutput: slot.denseOutput,
        sparseOutput: slot.sparseOutput,
        visibleCount: slot.visibleCount,
        emptyMask: slot.emptyMask,
      })),
    })),
    metrics: {
      retainedPairs,
      totalPairs,
      allowedPairs,
      maxAbsError,
      diffNorm,
      denseNorm,
      relL2Error,
      refNormIsZero,
      anyEmptyMask,
    },
  };
}
