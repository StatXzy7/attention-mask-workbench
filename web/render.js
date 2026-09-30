// SVG rendering: estimated scores, realized mask and dense/sparse weights.

const SVG_NS = 'http://www.w3.org/2000/svg';

export function svgEl(tag, attrs = {}, children = []) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  for (const child of children) el.appendChild(child);
  return el;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function fmt(v, digits = 3) {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(digits);
  return v.toFixed(digits + 1);
}

// Blue (0) -> accent -> yellow (1).
function heatColor(t) {
  const x = Math.max(0, Math.min(1, t));
  const stops = [
    [30, 40, 64],
    [92, 200, 255],
    [255, 206, 106],
  ];
  const seg = x <= 0.5 ? 0 : 1;
  const local = seg === 0 ? x * 2 : (x - 0.5) * 2;
  const a = stops[seg];
  const b = stops[seg + 1];
  const mix = a.map((av, i) => Math.round(av + (b[i] - av) * local));
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
}

const CELL_W = 64;
const CELL_H = 30;
const HEAD_LABEL_W = 72;
const ROW_GAP = 8;

function headRowSvg(parent, y, label, width, buildCells) {
  parent.appendChild(svgEl('text', { x: 6, y: y + CELL_H / 2 + 4, class: 'head-row-label' }, [
    document.createTextNode(label),
  ]));
  const inner = svgEl('g', { transform: `translate(${HEAD_LABEL_W},${y})` });
  buildCells(inner);
  parent.appendChild(inner);
  return y + CELL_H + ROW_GAP;
}

function colorFor(weight, maxWeight) {
  if (weight <= 0) return '#11182a';
  return heatColor(maxWeight > 0 ? weight / maxWeight : 0);
}

function textColor(weight, maxWeight) {
  if (weight <= 0) return 'light';
  const t = maxWeight > 0 ? weight / maxWeight : 0;
  return t > 0.62 ? '' : 'light';
}

export function renderScores(container, result, queryIndex) {
  clear(container);
  const { heads, model } = result;
  const numBins = result.params.intervals;
  const chartH = 64;
  const barAreaW = numBins * CELL_W;
  const width = HEAD_LABEL_W + barAreaW + 10;
  const heightPerHead = chartH + 34;
  const svg = svgEl('svg', {
    width,
    height: heads.length * heightPerHead + 24,
    role: 'img',
    'aria-label': 'estimated interval scores',
  });

  heads.forEach((head, h) => {
    const y0 = h * heightPerHead;
    const query = head.queries[queryIndex];
    const chosen = new Set(query.chosenIntervals);

    const label = svgEl('text', {
      x: 6,
      y: y0 + chartH / 2,
      class: 'head-row-label',
    });
    label.textContent = `Head ${h}`;
    svg.appendChild(label);

    const g = svgEl('g', { transform: `translate(${HEAD_LABEL_W},${y0})` });
    g.appendChild(
      svgEl('line', { x1: 0, y1: chartH, x2: barAreaW, y2: chartH, stroke: '#2a3448' }),
    );

    let maxScore = 0;
    for (const s of query.intervalScores) if (s > maxScore) maxScore = s;
    const top = maxScore > 0 ? maxScore : 1;

    query.intervalScores.forEach((score, bin) => {
      const x = bin * CELL_W;
      const barH = (score / top) * (chartH - 16);
      const isChosen = chosen.has(bin);
      const rect = svgEl('rect', {
        'data-kind': 'score-bar',
        x: x + 4,
        y: chartH - barH,
        width: CELL_W - 10,
        height: Math.max(barH, score === 0 ? 0 : 1),
        rx: 3,
        fill: isChosen ? '#ffce6a' : '#5cc8ff',
        opacity: score === 0 ? 0.25 : 1,
      });
      rect.appendChild(svgEl('title', {}, [document.createTextNode(`区间 ${bin} 归一化分数 ${fmt(score)} (原始 ${fmt(query.intervalRawScores[bin], 4)})`)]));
      g.appendChild(rect);

      if (isChosen) {
        g.appendChild(
          svgEl('rect', {
            x: x + 3,
            y: chartH - barH - 1,
            width: CELL_W - 8,
            height: barH + 2,
            class: 'chosen-outline',
            fill: 'none',
          }),
        );
      }
      const valueText = svgEl('text', {
        x: x + CELL_W / 2 - 1,
        y: chartH - barH - 4,
        'text-anchor': 'middle',
        class: 'axis-label',
        fill: isChosen ? '#ffce6a' : '#93a0b8',
      });
      valueText.textContent = fmt(score);
      g.appendChild(valueText);

      const binLabel = svgEl('text', {
        x: x + CELL_W / 2 - 1,
        y: chartH + 14,
        'text-anchor': 'middle',
        class: 'axis-label',
      });
      binLabel.textContent = `#${bin}`;
      g.appendChild(binLabel);
    });

    const note = svgEl('text', { x: 0, y: chartH + 28, class: 'axis-label' });
    const picked = query.chosenIntervals.map((b) => `#${b}`).join(', ') || '无';
    note.textContent = `已选区间: ${picked}（分数并列时按 头→位置 稳定排序）`;
    g.appendChild(note);

    svg.appendChild(g);
  });

  container.appendChild(svg);
}

export function renderMask(container, result, queryIndex) {
  clear(container);
  const { heads, model } = result;
  const nk = model.nk;
  const blockH = CELL_H + 22;
  const width = HEAD_LABEL_W + nk * CELL_W + 10;
  const svg = svgEl('svg', { width, height: heads.length * blockH + 8 });

  heads.forEach((head, h) => {
    const y = h * blockH;
    let cursorY = y;
    const label = svgEl('text', { x: 6, y: y + 14, class: 'head-row-label' });
    label.textContent = `Head ${h}`;
    svg.appendChild(label);

    const g = svgEl('g', { transform: `translate(${HEAD_LABEL_W},0)` });
    for (let j = 0; j < nk; j += 1) {
      const x = j * CELL_W;
      const keyLabel = svgEl('text', { x: x + CELL_W / 2, y: y + 14, 'text-anchor': 'middle', class: 'axis-label' });
      keyLabel.textContent = `k${j}`;
      g.appendChild(keyLabel);

      const retained = head.queries[queryIndex].mask[j] === 1;
      const visible = j < head.queries[queryIndex].visibleCount;
      let fill = '#11182a';
      let text = '未选';
      if (retained) {
        fill = '#58d39b';
        text = '保留';
      } else if (!visible) {
        fill = '#26221f';
        text = '未来';
      }
      const rect = svgEl('rect', {
        x: x + 3,
        y: y + 18,
        width: CELL_W - 6,
        height: CELL_H - 8,
        rx: 4,
        fill,
        stroke: retained ? '#58d39b' : visible ? '#2a3448' : '#5a4a33',
        'stroke-dasharray': visible ? undefined : '3 3',
      });
      rect.appendChild(svgEl('title', {}, [document.createTextNode(
        retained ? `保留位置 k${j}` : visible ? `可见但未保留 k${j}` : `因果遮蔽（未来）k${j}`,
      )]));
      g.appendChild(rect);
      const t = svgEl('text', {
        x: x + CELL_W / 2,
        y: y + 18 + (CELL_H - 8) / 2 + 3,
        'text-anchor': 'middle',
        class: retained ? 'cell-text' : 'cell-text light',
      });
      t.textContent = text;
      g.appendChild(t);
    }
    svg.appendChild(g);
  });

  container.appendChild(svg);
}

export function renderWeights(container, result, queryIndex) {
  clear(container);
  const { heads, model } = result;
  const nk = model.nk;
  const rowsPerHead = 2;
  const blockH = 18 + rowsPerHead * (CELL_H + ROW_GAP) + 10;
  const width = HEAD_LABEL_W + nk * CELL_W + 10;
  const svg = svgEl('svg', { width, height: heads.length * blockH + 8 });

  heads.forEach((head, h) => {
    const baseY = h * blockH;
    const query = head.queries[queryIndex];

    const label = svgEl('text', { x: 6, y: baseY + 14, class: 'head-row-label' });
    label.textContent = `Head ${h}`;
    svg.appendChild(label);

    const g = svgEl('g', { transform: `translate(${HEAD_LABEL_W},0)` });
    for (let j = 0; j < nk; j += 1) {
      const keyLabel = svgEl('text', {
        x: j * CELL_W + CELL_W / 2,
        y: baseY + 14,
        'text-anchor': 'middle',
        class: 'axis-label',
      });
      keyLabel.textContent = `k${j}`;
      g.appendChild(keyLabel);
    }

    const rows = [
      { tag: '稠密', weights: query.denseWeights },
      { tag: '稀疏', weights: query.sparseWeights },
    ];
    rows.forEach((rowSpec, r) => {
      const rowY = baseY + 18 + r * (CELL_H + ROW_GAP);
      let maxWeight = 0;
      for (const w of rowSpec.weights) if (w > maxWeight) maxWeight = w;
      rowSpec.weights.forEach((w, j) => {
        const x = j * CELL_W;
        const rect = svgEl('rect', {
          x: x + 3,
          y: rowY,
          width: CELL_W - 6,
          height: CELL_H,
          rx: 4,
          fill: colorFor(w, maxWeight),
          stroke: '#2a3448',
        });
        const sparseZero = rowSpec.tag === '稀疏' && w === 0 && query.denseWeights[j] > 0;
        rect.appendChild(svgEl('title', {}, [document.createTextNode(
          `${rowSpec.tag}权重 k${j} = ${fmt(w, 5)}${sparseZero ? '（稠密非零，被掩码）' : ''}`,
        )]));
        g.appendChild(rect);
        const text = svgEl('text', {
          x: x + CELL_W / 2,
          y: rowY + CELL_H / 2 + 3,
          'text-anchor': 'middle',
          class: `cell-text ${textColor(w, maxWeight)}`,
        });
        text.textContent = fmt(w, 2);
        g.appendChild(text);
      });
      const rowLabel = svgEl('text', {
        x: -HEAD_LABEL_W + 6,
        y: rowY + CELL_H / 2 + 4,
        class: 'axis-label',
      });
      rowLabel.textContent = rowSpec.tag;
      g.appendChild(rowLabel);
    });

    svg.appendChild(g);
  });

  container.appendChild(svg);
}

export function renderOutputs(container, result, queryIndex) {
  clear(container);
  result.heads.forEach((head) => {
    const query = head.queries[queryIndex];
    const wrap = document.createElement('div');
    const title = document.createElement('h3');
    title.style.cssText = 'margin:8px 0 4px;font-size:13px;color:var(--accent)';
    title.textContent = `Head ${head.headIndex}（dv=${head.dv}，保留 ${query.retainedKeys.length}/${query.visibleCount} 个可见键）`;
    wrap.appendChild(title);

    if (query.emptyMask) {
      const note = document.createElement('div');
      note.className = 'empty-state';
      note.textContent = '空掩码状态：零个保留位置，稀疏输出返回零向量。';
      wrap.appendChild(note);
    }

    const table = document.createElement('table');
    table.className = 'outputs';
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    ['维度', '稠密参考', '稀疏输出', '差值'].forEach((name) => {
      const th = document.createElement('th');
      th.textContent = name;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (let d = 0; d < head.dv; d += 1) {
      const dense = query.denseOutput[d];
      const sparse = query.sparseOutput[d];
      const tr = document.createElement('tr');
      const cells = [`d${d}`, fmt(dense, 5), fmt(sparse, 5), fmt(sparse - dense, 5)];
      cells.forEach((cell, idx) => {
        const td = document.createElement('td');
        td.textContent = cell;
        if (idx === 3 && Math.abs(sparse - dense) > 1e-10) td.style.color = 'var(--warn)';
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    container.appendChild(wrap);
  });
}

export function renderMetrics(container, result) {
  clear(container);
  const m = result.metrics;
  const p = result.params;
  const cards = [
    { k: '保留 (头·查询·键) 对', v: `${m.retainedPairs} / ${m.allowedPairs} 可见（总 ${m.totalPairs}）` },
    { k: '最大绝对误差', v: fmt(m.maxAbsError, 6), cls: m.maxAbsError === 0 ? 'good' : 'warn' },
    {
      k: m.refNormIsZero ? '相对二范数误差（参考范数≈0，回退为绝对差范数）' : '相对二范数误差 ‖Δ‖/‖稠密‖',
      v: fmt(m.relL2Error, 6),
      cls: m.relL2Error === 0 ? 'good' : 'warn',
    },
    { k: '稠密参考范数', v: fmt(m.denseNorm, 6) },
    { k: '生效参数', v: `B=${p.intervals}, 预算=${p.budget}${p.causal ? ', 因果' : ''}${p.shared ? ', 跨头共享' : ''}` },
  ];
  for (const card of cards) {
    const div = document.createElement('div');
    div.className = `metric ${card.cls || ''}`;
    div.innerHTML = `<span class="k"></span><span class="v"></span>`;
    div.querySelector('.k').textContent = card.k;
    div.querySelector('.v').textContent = card.v;
    container.appendChild(div);
  }

  const notices = [];
  if (p.budgetWasClamped) notices.push('预算超过区间数上限，已截断到区间数。');
  if (p.intervalsWasClamped) notices.push('区间数超过键位置数，已截断到键位置数。');
  if (p.budget === 0) notices.push('零预算：所有头返回零向量，掩码为空。');
  if (m.anyEmptyMask && p.budget > 0) notices.push('存在空掩码查询（因果模式下可见区间不足）。');
  if (m.refNormIsZero) notices.push('稠密参考范数为 0：相对误差按绝对差范数报告。');
  for (const text of notices) {
    const div = document.createElement('div');
    div.className = 'metric warn';
    div.innerHTML = `<span class="k">语义提示</span><span class="v" style="font-size:12px"></span>`;
    div.querySelector('.v').textContent = text;
    container.appendChild(div);
  }
}
