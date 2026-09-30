'use strict';

(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const CELL = 34;
  const ROW_H = 30;
  const LABEL_W = 74;
  const TRACK_GAP = 26;

  const els = {
    numBlocks: document.getElementById('numBlocks'),
    budget: document.getElementById('budget'),
    causal: document.getElementById('causal'),
    sharedBudget: document.getElementById('sharedBudget'),
    errorBox: document.getElementById('errorBox'),
    statusBox: document.getElementById('statusBox'),
    metricsView: document.getElementById('metricsView'),
    svgView: document.getElementById('svgView'),
    outputView: document.getElementById('outputView'),
    editQ: document.getElementById('editQ'),
    editK: document.getElementById('editK'),
    editV: document.getElementById('editV'),
    apply: document.getElementById('applyEdits'),
    reset: document.getElementById('resetData'),
  };

  let store = createStore();
  let lastAppliedData = JSON.stringify(store.getSnapshot().data);
  let statusTimer = null;

  function fmt(x, digits) {
    if (x === null || x === undefined) return '—';
    if (!Number.isFinite(x)) return x > 0 ? '∞' : String(x);
    const d = digits === undefined ? 4 : digits;
    if (x !== 0 && Math.abs(x) < 1e-4) return x.toExponential(2);
    return Number(x.toFixed(d)).toString();
  }

  function el(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function heatColor(v, max) {
    if (!(v > 0)) return '#111a2c';
    const t = Math.max(0, Math.min(1, max > 0 ? v / max : 0));
    const light = 18 + t * 47;
    const hue = 212;
    return `hsl(${hue}, 82%, ${light}%)`;
  }

  function drawTrack(svg, y0, title, n, valueAt, opts) {
    const titleNode = el('text', { x: 4, y: y0 + 14, fill: '#94a3b8', 'font-size': 12 }, title);
    svg.appendChild(titleNode);
    const gy = y0 + 22;
    let max = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = valueAt(i, j);
        if (v !== null && Number.isFinite(v) && v > max) max = v;
      }
    }
    for (let i = 0; i < n; i++) {
      svg.appendChild(el('text', {
        x: LABEL_W - 6, y: gy + i * ROW_H + ROW_H / 2 + 4,
        'text-anchor': 'end', fill: '#94a3b8', 'font-size': 11,
      }, `q${i}`));
      for (let j = 0; j < n; j++) {
        const v = valueAt(i, j);
        const fill = v === null ? '#0a0f1a' : heatColor(v, max);
        const rect = el('rect', {
          x: LABEL_W + j * CELL, y: gy + i * ROW_H,
          width: CELL - 2, height: ROW_H - 2,
          fill, stroke: '#243047', 'stroke-width': 0.5, rx: 3,
        });
        const t = v === null ? (opts.forbiddenText || '未来') : fmt(v, opts.digits === undefined ? 2 : opts.digits);
        const titleTip = el('title', {}, opts.tooltip ? opts.tooltip(i, j, v) : `${title} (q${i}, k${j}) = ${v === null ? '禁止' : fmt(v, 4)}`);
        rect.appendChild(titleTip);
        svg.appendChild(rect);
        svg.appendChild(el('text', {
          x: LABEL_W + j * CELL + CELL / 2 - 1,
          y: gy + i * ROW_H + ROW_H / 2 + 4,
          'text-anchor': 'middle', fill: v !== null && v > max * 0.55 ? '#0b1220' : '#c8d6ec',
          'font-size': 10, 'pointer-events': 'none',
        }, t));
      }
    }
    for (let j = 0; j <= n; j++) {
      if (j < n) {
        svg.appendChild(el('text', {
          x: LABEL_W + j * CELL + CELL / 2, y: gy - 6,
          'text-anchor': 'middle', fill: '#94a3b8', 'font-size': 11,
        }, `k${j}`));
      }
    }
    return gy + n * ROW_H;
  }

  function renderHeadSvg(container, r, h) {
    const { n, numBlocks, causal } = r.config;
    const width = LABEL_W + n * CELL + 10;
    const trackH = 22 + n * ROW_H;
    const height = 26 + trackH * 3 + TRACK_GAP * 2;
    const svg = el('svg', { width, height, viewBox: `0 0 ${width} ${height}` });

    const blockOf = new Array(n);
    r.blocks.forEach((b) => {
      for (let j = b.start; j < b.end; j++) blockOf[j] = b.block;
    });
    const retainedSet = r.retained[h].map((row) => new Set(row));

    let y = 4;
    y = drawTrack(svg, y, '估计分数（区间归一化，映射到位置）', n, (i, j) => {
      if (causal && j > i) return null;
      return r.estNorm[h][i][blockOf[j]];
    }, { digits: 2,
      tooltip(i, j, v) {
        const b = blockOf[j];
        const blk = r.blocks[b];
        return `头${h} q${i} 区间${b}[${blk.start},${blk.end}) 归一化=${fmt(r.estNorm[h][i][b], 4)} 原始=${fmt(r.estRaw[h][i][b], 4)}`;
      },
    }) + TRACK_GAP;

    y = drawTrack(svg, y, '实际稀疏掩码（1=保留位置）', n, (i, j) => {
      if (causal && j > i) return null;
      return retainedSet[i].has(j) ? 1 : 0;
    }, {
      digits: 0, forbiddenText: '×',
      tooltip(i, j) { return `头${h} q${i} → k${j}: ${retainedSet[i].has(j) ? '保留' : (causal && j > i ? '因果禁止' : '未选中')}`; },
    }) + TRACK_GAP;

    drawTrack(svg, y, '稠密注意力权重（参考）', n, (i, j) => {
      if (causal && j > i) return null;
      return r.denseWeights[h][i][j];
    }, {
      digits: 2,
      tooltip(i, j, v) { return `头${h} q${i} → k${j} 稠密权重=${fmt(v, 5)} 稀疏权重=${fmt(r.sparseWeights[h][i][j], 5)}`; },
    });
    container.appendChild(svg);
  }

  function renderSvg(snap) {
    els.svgView.innerHTML = '';
    const r = snap.result;
    const legend = document.createElement('p');
    legend.className = 'legend';
    legend.textContent = '每个头自上而下三行：估计分数、实际掩码、稠密权重；深色格表示未选/零值，斜格（“未来”/×）为因果禁止。悬停查看原始估计与稀疏权重。';
    els.svgView.appendChild(legend);
    for (let h = 0; h < r.config.heads; h++) {
      const wrap = document.createElement('div');
      wrap.className = 'head-block';
      wrap.appendChild(Object.assign(document.createElement('p'), { className: 'head-title', textContent: `头 ${h}` }));
      const div = document.createElement('div');
      renderHeadSvg(div, r, h);
      wrap.appendChild(div);
      els.svgView.appendChild(wrap);
    }
  }

  function renderMetrics(snap) {
    const m = snap.result.metrics;
    const c = snap.result.config;
    const allEmpty = m.emptyRows.length === c.heads * c.n;
    const items = [
      ['保留位置数', `${m.totalRetained} / ${m.totalPossible}`],
      ['最大绝对误差', Number.isFinite(m.maxAbsError) ? fmt(m.maxAbsError, 6) : '∞'],
      ['相对二范数误差', Number.isFinite(m.relativeL2Error) ? fmt(m.relativeL2Error, 6) : '∞'],
      ['有效预算 / 请求预算', `${c.budget} / ${c.requestedBudget}`],
    ];
    let html = '<dl class="metric-grid">';
    for (const [k, v] of items) html += `<dt>${k}</dt><dd>${v}</dd>`;
    html += '</dl>';
    if (allEmpty) {
      html += '<p><span class="badge bad">空掩码状态</span> 所有查询均无保留位置，稀疏输出为零向量。</p>';
    }
    if (m.emptyRows.length > 0 && !allEmpty) {
      html += `<p><span class="badge warn">部分空掩码</span> ${m.emptyRows.length} 个 (头,查询) 行无保留位置（含因果下无可用历史的行）。</p>`;
    }
    if (m.allReferenceZero) {
      html += '<p><span class="badge warn">参考范数为零</span> 稠密参考总范数为 0：稀疏也为 0 时相对误差记 0，否则记 ∞。</p>';
    } else if (m.referenceZeroRows.length > 0) {
      html += `<p><span class="badge warn">部分参考行范数为零</span> ${m.referenceZeroRows.length} 行按“零/零=0、非零/零=∞”处理。</p>`;
    }
    if (c.requestedBudget > c.budget) {
      html += `<p><span class="badge warn">预算截断</span> 请求预算 ${c.requestedBudget} 超过区间数，已截断为 ${c.budget}。</p>`;
    }
    els.metricsView.innerHTML = html;
  }

  function renderOutputs(snap) {
    const r = snap.result;
    const { heads, n, dv } = r.config;
    let html = '';
    for (let h = 0; h < heads; h++) {
      html += `<h3>头 ${h}</h3><table><thead><tr><th>查询</th><th>保留位置</th><th>稀疏输出</th><th>稠密参考</th><th>行最大绝对误差</th><th>行相对L2</th></tr></thead><tbody>`;
      for (let i = 0; i < n; i++) {
        const pe = r.metrics.perRowErrors[h][i];
        const refZero = pe.refNorm === 0;
        const sparse = r.sparseOut[h][i].map((x) => fmt(x, 4)).join(', ');
        const dense = r.denseOut[h][i].map((x) => fmt(x, 4)).join(', ');
        const rel = pe.relativeL2 === null ? (refZero && pe.maxAbs === 0 ? '0（参考为零）' : '∞（参考为零）') : fmt(pe.relativeL2, 4);
        html += `<tr><td>q${i}</td><td>[${r.retained[h][i].join(', ')}]${r.retained[h][i].length === 0 ? ' <span class="badge bad">空</span>' : ''}</td>`
          + `<td>[${sparse}]</td><td class="${refZero ? 'zero-ref' : ''}">[${dense}]</td>`
          + `<td>${fmt(pe.maxAbs, 6)}</td><td>${rel}</td></tr>`;
      }
      html += '</tbody></table>';
    }
    els.outputView.innerHTML = html;
  }

  function syncControls(snap) {
    els.numBlocks.value = snap.params.numBlocks;
    els.budget.value = snap.params.budget;
    els.causal.checked = snap.params.causal;
    els.sharedBudget.checked = snap.params.sharedBudget;
  }

  function syncEditors(snap) {
    const current = JSON.stringify(snap.data);
    if (current !== lastAppliedData || els.editQ.value === '') {
      els.editQ.value = JSON.stringify(snap.data.Q);
      els.editK.value = JSON.stringify(snap.data.K);
      els.editV.value = JSON.stringify(snap.data.V);
      lastAppliedData = current;
    }
  }

  function flashStatus(text) {
    els.statusBox.textContent = text;
    if (statusTimer) clearTimeout(statusTimer);
    statusTimer = setTimeout(() => { els.statusBox.textContent = ''; }, 1600);
  }

  function render(snap) {
    if (snap.error) {
      els.errorBox.hidden = false;
      els.errorBox.textContent = snap.error;
    } else {
      els.errorBox.hidden = true;
      els.errorBox.textContent = '';
    }
    if (snap.result) {
      renderMetrics(snap);
      renderSvg(snap);
      renderOutputs(snap);
      syncEditors(snap);
    }
  }

  function readParamPatch() {
    return {
      numBlocks: Number(els.numBlocks.value),
      budget: Number(els.budget.value),
      causal: els.causal.checked,
      sharedBudget: els.sharedBudget.checked,
    };
  }

  els.numBlocks.addEventListener('change', () => {
    try {
      const snap = store.setParams(readParamPatch());
      syncControls(snap);
      flashStatus('参数已更新');
    } catch (e) {
      store.reportError(e.message);
      syncControls(store.getSnapshot());
    }
  });
  ['budget', 'causal', 'sharedBudget'].forEach((id) => {
    els[id].addEventListener('change', () => {
      try {
        const snap = store.setParams(readParamPatch());
        syncControls(snap);
        flashStatus('参数已更新');
      } catch (e) {
        store.reportError(e.message);
        syncControls(store.getSnapshot());
      }
    });
  });

  els.apply.addEventListener('click', () => {
    try {
      const snap = store.applyEdits({ Q: els.editQ.value, K: els.editK.value, V: els.editV.value });
      lastAppliedData = JSON.stringify(snap.data);
      flashStatus('编辑已应用并重算');
    } catch (e) {
      store.reportError(e.message);
    }
  });

  els.reset.addEventListener('click', () => {
    store = createStore();
    store.subscribe(render);
    lastAppliedData = JSON.stringify(store.getSnapshot().data);
    syncControls(store.getSnapshot());
    syncEditors(store.getSnapshot());
    render(store.getSnapshot());
    flashStatus('已恢复内置数据');
  });

  store.subscribe(render);
  syncControls(store.getSnapshot());
  syncEditors(store.getSnapshot());
  render(store.getSnapshot());
})();
