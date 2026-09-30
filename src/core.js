'use strict';

// 固定非负特征映射: phi(x) = max(x, 0) + 1
const FEATURE_BIAS = 1;

function phi(x) {
  return Math.max(x, 0) + FEATURE_BIAS;
}

function featureMapTensor(t) {
  return t.map((m) => m.map((row) => row.map(phi)));
}

function isFiniteNumber(x) {
  return typeof x === 'number' && Number.isFinite(x);
}

function toDense3D(value, name) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${name} 必须是非空的三维数组`);
  }
  const a = value.length;
  const b0 = value[0];
  if (!Array.isArray(b0) || b0.length === 0) {
    throw new Error(`${name} 必须是非空的三维数组`);
  }
  const b = b0.length;
  if (!Array.isArray(b0[0]) || b0[0].length === 0) {
    throw new Error(`${name} 必须是非空的三维数组`);
  }
  const c = b0[0].length;
  for (let h = 0; h < a; h++) {
    const m = value[h];
    if (!Array.isArray(m) || m.length !== b) {
      throw new Error(`${name} 形状冲突: 各头的行数必须一致`);
    }
    for (let i = 0; i < b; i++) {
      if (!Array.isArray(m[i]) || m[i].length !== c) {
        throw new Error(`${name} 形状冲突: 第 ${h} 头第 ${i} 行维度不正确`);
      }
      for (let j = 0; j < c; j++) {
        if (!isFiniteNumber(m[i][j])) {
          throw new Error(`${name} 包含非有限数 (头 ${h}, 行 ${i}, 列 ${j})`);
        }
      }
    }
  }
  return { tensor: value, heads: a, rows: b, cols: c };
}

// 将长度 n 划分为 nBlocks 个尽量等长的连续半开区间 [start, end)
function partitionBlocks(n, nBlocks) {
  if (!Number.isInteger(nBlocks) || nBlocks < 1) {
    throw new Error('区间数必须是不小于 1 的整数');
  }
  if (nBlocks > n) {
    throw new Error(`区间数 ${nBlocks} 超过键位置数 ${n}`);
  }
  const blocks = [];
  const base = Math.floor(n / nBlocks);
  let extra = n % nBlocks;
  let start = 0;
  for (let b = 0; b < nBlocks; b++) {
    const size = base + (extra > 0 ? 1 : 0);
    if (extra > 0) extra -= 1;
    const end = start + size;
    blocks.push({ block: b, start, end });
    start = end;
  }
  return blocks;
}

function dot(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += a[k] * b[k];
  return s;
}

function softmaxStable(xs) {
  if (xs.length === 0) return [];
  let m = xs[0];
  for (let i = 1; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  const exps = xs.map((x) => Math.exp(x - m));
  let sum = 0;
  for (const e of exps) sum += e;
  return exps.map((e) => e / sum);
}

function assertPositiveInt(value, label) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${label} 必须是不小于 1 的整数`);
  }
}

// 工作台主计算: 估计分数 -> 排序/预算选区间 -> 映射回原位置 -> 稀疏注意力
function computeWorkbench(qIn, kIn, vIn, options = {}) {
  const q = toDense3D(qIn, '查询 Q');
  const k = toDense3D(kIn, '键 K');
  const v = toDense3D(vIn, '值 V');
  const heads = q.heads;
  const n = q.rows;
  const dk = q.cols;
  if (k.heads !== heads || v.heads !== heads) {
    throw new Error('形状冲突: Q、K、V 的头数必须一致');
  }
  if (k.rows !== n || v.rows !== n) {
    throw new Error('形状冲突: Q、K、V 的位置数必须一致');
  }
  if (k.cols !== dk) {
    throw new Error(`形状冲突: K 的特征维度 ${k.cols} 与 Q 的 ${dk} 不一致`);
  }
  assertPositiveInt(v.cols, '值维度');
  const dv = v.cols;

  const numBlocks = options.numBlocks;
  const budget = options.budget;
  const causal = !!options.causal;
  const sharedBudget = !!options.sharedBudget;
  if (!Number.isInteger(numBlocks) || numBlocks < 1) {
    throw new Error('区间数必须是不小于 1 的整数');
  }
  if (numBlocks > n) {
    throw new Error(`区间数 ${numBlocks} 超过键位置数 ${n}`);
  }
  if (!Number.isInteger(budget) || budget < 0) {
    throw new Error('预算必须是不小于 0 的整数');
  }
  const blocks = partitionBlocks(n, numBlocks);
  // 预算超上限时截断到可用区间总数
  const effectiveBudget = Math.min(budget, numBlocks);

  const pq = featureMapTensor(q.tensor);
  const pk = featureMapTensor(k.tensor);

  // estRaw[h][i][b]: phi(q_i) · 区间内 phi(k) 的均值
  const estRaw = [];
  const estNorm = [];
  for (let h = 0; h < heads; h++) {
    const headRaw = [];
    const headNorm = [];
    for (let i = 0; i < n; i++) {
      const rowRaw = new Array(numBlocks).fill(0);
      const rowNorm = new Array(numBlocks).fill(0);
      const allowed = [];
      for (const blk of blocks) {
        // 因果: 区间估计不得读未来键, 完全位于未来的区间直接排除
        let count = 0;
        if (!causal || blk.start <= i) {
          const end = causal ? Math.min(blk.end, i + 1) : blk.end;
          let s = 0;
          for (let j = blk.start; j < end; j++) s += dot(pq[h][i], pk[h][j]);
          count = end - blk.start;
          rowRaw[blk.block] = count > 0 ? s / count : 0;
          allowed.push(blk.block);
        }
      }
      // 每头估计分数按区间和归一化
      let sum = 0;
      for (const b of allowed) sum += rowRaw[b];
      if (sum > 0) {
        for (const b of allowed) rowNorm[b] = rowRaw[b] / sum;
      }
      // sum === 0 时归一化分数保持 0, 候选仍按稳定次序参与选择
      headRaw.push(rowRaw);
      headNorm.push(rowNorm);
    }
    estRaw.push(headRaw);
    estNorm.push(headNorm);
  }

  const scale = 1 / Math.sqrt(dk);
  const selectedBlocks = [];
  const retained = [];
  const denseWeights = [];
  const sparseWeights = [];
  const denseOut = [];
  const sparseOut = [];
  for (let h = 0; h < heads; h++) {
    selectedBlocks.push(Array.from({ length: n }, () => []));
    retained.push(Array.from({ length: n }, () => []));
    denseWeights.push([]);
    sparseWeights.push([]);
    denseOut.push([]);
    sparseOut.push([]);
  }

  for (let i = 0; i < n; i++) {
    // 构造候选区间; 因果下只保留不读未来的区间
    const candidates = [];
    for (let h = 0; h < heads; h++) {
      for (const blk of blocks) {
        if (causal && blk.start > i) continue;
        candidates.push({
          h,
          b: blk.block,
          start: blk.start,
          end: blk.end,
          score: estNorm[h][i][blk.block],
        });
      }
    }
    // 同一估计分数下按头、再按位置(区间序号)稳定排序
    candidates.sort((x, y) => {
      if (y.score !== x.score) return y.score - x.score;
      if (x.h !== y.h) return x.h - y.h;
      return x.b - y.b;
    });

    if (sharedBudget) {
      // 同一查询跨头共享一份预算
      const chosen = candidates.slice(0, effectiveBudget);
      const byHead = new Map();
      for (const c of chosen) {
        if (!byHead.has(c.h)) byHead.set(c.h, []);
        byHead.get(c.h).push(c);
      }
      for (let h = 0; h < heads; h++) {
        selectedBlocks[h][i] = (byHead.get(h) || []).map((c) => c.b);
      }
    } else {
      // 逐头选择: 每头各自获得预算
      for (let h = 0; h < heads; h++) {
        const headCands = candidates.filter((c) => c.h === h);
        selectedBlocks[h][i] = headCands.slice(0, effectiveBudget).map((c) => c.b);
      }
    }

    // 区间映射回原始键位置, 映射重复去重
    for (let h = 0; h < heads; h++) {
      const seen = new Set();
      const pos = [];
      for (const b of selectedBlocks[h][i]) {
        const blk = blocks[b];
        const end = causal ? Math.min(blk.end, i + 1) : blk.end;
        for (let j = blk.start; j < end; j++) {
          if (!seen.has(j)) {
            seen.add(j);
            pos.push(j);
          }
        }
      }
      pos.sort((a, b2) => a - b2);
      retained[h][i] = pos;
    }
  }

  let totalRetained = 0;
  let totalPossible = 0;
  let maxAbs = 0;
  let relNumSq = 0;
  let relDenSq = 0;
  const refZeroRows = [];
  const emptyRows = [];
  const perRowErrors = [];

  for (let h = 0; h < heads; h++) {
    const hRowErrs = [];
    for (let i = 0; i < n; i++) {
      // 稠密参考: 稳定的缩放点积注意力
      const limit = causal ? i + 1 : n;
      const dLogits = [];
      for (let j = 0; j < limit; j++) dLogits.push(scale * dot(q.tensor[h][i], k.tensor[h][j]));
      const dw = softmaxStable(dLogits);
      const dRow = new Array(dv).fill(0);
      for (let j = 0; j < limit; j++) {
        for (let c = 0; c < dv; c++) dRow[c] += dw[j] * v.tensor[h][j][c];
      }
      denseWeights[h].push(dw);
      denseOut[h].push(dRow);

      // 稀疏: 仅在保留位置上计算注意力
      const pos = retained[h][i];
      const sRow = new Array(dv).fill(0);
      const sw = new Array(n).fill(0);
      if (pos.length > 0) {
        const sLogits = pos.map((j) => scale * dot(q.tensor[h][i], k.tensor[h][j]));
        const sp = softmaxStable(sLogits);
        pos.forEach((j, t) => {
          sw[j] = sp[t];
          for (let c = 0; c < dv; c++) sRow[c] += sp[t] * v.tensor[h][j][c];
        });
      }
      sparseWeights[h].push(sw);
      sparseOut[h].push(sRow);

      totalRetained += pos.length;
      totalPossible += limit;
      if (pos.length === 0) emptyRows.push({ h, i });

      let rowMax = 0;
      let numSq = 0;
      let denSq = 0;
      for (let c = 0; c < dv; c++) {
        const diff = sRow[c] - dRow[c];
        const ad = Math.abs(diff);
        if (ad > rowMax) rowMax = ad;
        numSq += diff * diff;
        denSq += dRow[c] * dRow[c];
      }
      if (rowMax > maxAbs) maxAbs = rowMax;
      relNumSq += numSq;
      relDenSq += denSq;
      const refNorm = Math.sqrt(denSq);
      const rowRel = refNorm === 0 ? null : Math.sqrt(numSq) / refNorm;
      hRowErrs.push({ maxAbs: rowMax, relativeL2: rowRel, refNorm });
      if (refNorm === 0) refZeroRows.push({ h, i });
    }
    perRowErrors.push(hRowErrs);
  }

  // 参考输出范数为零时: 差异为 0 返回相对误差 0, 否则返回 Infinity 以明确表达语义
  const relativeL2 = relDenSq === 0 ? (relNumSq === 0 ? 0 : Infinity) : Math.sqrt(relNumSq / relDenSq);
  const anyRefZero = relDenSq === 0 && refZeroRows.length > 0;

  return {
    config: { heads, n, dk, dv, numBlocks, budget: effectiveBudget, requestedBudget: budget, causal, sharedBudget, scale },
    blocks,
    estRaw,
    estNorm,
    selectedBlocks,
    retained,
    denseWeights,
    sparseWeights,
    denseOut,
    sparseOut,
    metrics: {
      totalRetained,
      totalPossible,
      maxAbsError: maxAbs,
      relativeL2Error: relativeL2,
      allReferenceZero: anyRefZero,
      referenceZeroRows: refZeroRows,
      emptyRows,
      perRowErrors,
    },
  };
}

(typeof module !== 'undefined' && module.exports
  ? (x) => Object.assign(module.exports, x)
  : (x) => Object.assign(globalThis, x))({
  FEATURE_BIAS,
  phi,
  featureMapTensor,
  toDense3D,
  partitionBlocks,
  softmaxStable,
  computeWorkbench,
});
