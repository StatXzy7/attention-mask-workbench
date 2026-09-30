// Built-in sample multi-head data and the hand-edit parser.

export const BUILTIN_MODEL = {
  heads: [
    {
      // Head 0: dk = 2, dv = 2
      q: [
        [1, 0],
        [0, 2],
        [2, 1],
        [0, 0],
      ],
      k: [
        [1, 0],
        [0, 1],
        [2, 1],
        [0, 0],
        [1, 2],
      ],
      v: [
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 0],
        [0, 2],
      ],
    },
    {
      // Head 1: dk = 3, dv = 1
      q: [
        [1, 0, 0],
        [0, 1, 1],
        [2, 0, 1],
        [0, 0, 0],
      ],
      k: [
        [1, 0, 1],
        [0, 2, 0],
        [1, 1, 0],
        [0, 0, 1],
        [2, 1, 0],
      ],
      v: [[1], [2], [0], [3], [1]],
    },
  ],
};

export const DEFAULT_PARAMS = {
  intervals: 2,
  budget: 1,
  causal: false,
  shared: false,
};

export function cloneModel(model) {
  return {
    heads: model.heads.map((head) => ({
      q: head.q.map((row) => row.slice()),
      k: head.k.map((row) => row.slice()),
      v: head.v.map((row) => row.slice()),
    })),
  };
}

export function matrixToText(matrix) {
  return matrix.map((row) => row.join(' ')).join('\n');
}

// Parse text like "1 0\n2 3" into a numeric matrix.
// Throws WorkbenchValidationError-like errors with row/column context.
export function parseMatrixText(text, label) {
  if (typeof text !== 'string') {
    throw new Error(`${label}: editor content must be text.`);
  }
  const lines = text.split(/\r?\n/);
  const rows = [];
  let width = null;
  lines.forEach((rawLine, lineIndex) => {
    const line = rawLine.trim();
    if (line === '') return;
    if (line.startsWith('#') || line.startsWith('//')) return; // comments
    const tokens = line.split(/[\s,;]+/).filter((t) => t !== '');
    if (tokens.length === 0) return;
    const row = [];
    tokens.forEach((token, column) => {
      const value = Number(token);
      if (!Number.isFinite(value)) {
        throw new Error(
          `${label}: "${token}" at row ${lineIndex + 1}, col ${column + 1} is not a finite number.`,
        );
      }
      row.push(value);
    });
    if (width === null) width = row.length;
    if (row.length !== width) {
      throw new Error(
        `${label}: row ${lineIndex + 1} has ${row.length} entries but row 1 has ${width}.`,
      );
    }
    rows.push(row);
  });
  if (rows.length === 0) {
    throw new Error(`${label}: matrix is empty.`);
  }
  return rows;
}

export function parseHeadFromText({ q, k, v }) {
  return {
    q: parseMatrixText(q, 'Q'),
    k: parseMatrixText(k, 'K'),
    v: parseMatrixText(v, 'V'),
  };
}
