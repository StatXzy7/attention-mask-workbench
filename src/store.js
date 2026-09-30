'use strict';

const core = (typeof require === 'function')
  ? require('./core.js')
  : globalThis;

const DEFAULT_PARAMS = { numBlocks: 3, budget: 1, causal: false, sharedBudget: false };

const DEFAULT_DATA = {
  // 2 头, 4 个位置, dk=3, dv=2; Q/K 均为非负输入
  Q: [
    [
      [2, 0, 1],
      [0, 3, 0],
      [1, 1, 2],
      [3, 0, 0],
    ],
    [
      [0, 1, 2],
      [2, 2, 0],
      [1, 0, 3],
      [0, 0, 1],
    ],
  ],
  K: [
    [
      [1, 0, 2],
      [0, 2, 1],
      [2, 1, 0],
      [0, 0, 3],
    ],
    [
      [2, 1, 0],
      [0, 3, 1],
      [1, 0, 2],
      [0, 2, 0],
    ],
  ],
  V: [
    [
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 0],
    ],
    [
      [0, 2],
      [1, 0],
      [0, 1],
      [2, 2],
    ],
  ],
};

function createStore(initial) {
  const data = initial && initial.data ? clone(initial.data) : clone(DEFAULT_DATA);
  const params = Object.assign({}, DEFAULT_PARAMS, initial && initial.params ? initial.params : {});
  let result = null;
  let error = null;
  const listeners = new Set();

  function clone(v) {
    return JSON.parse(JSON.stringify(v));
  }

  function notify() {
    for (const fn of listeners) fn(getSnapshot());
  }

  function recompute() {
    result = core.computeWorkbench(data.Q, data.K, data.V, params);
  }

  function setParams(patch) {
    const next = Object.assign({}, params, patch);
    // 先用新参数试算, 抛错则本次修改完全不生效
    const trial = core.computeWorkbench(data.Q, data.K, data.V, next);
    Object.assign(params, next);
    result = trial;
    error = null;
    notify();
    return getSnapshot();
  }

  function applyEdits(texts) {
    const parsed = {};
    for (const name of ['Q', 'K', 'V']) {
      let value;
      try {
        value = JSON.parse(texts[name]);
      } catch (e) {
        throw new Error(`${name} JSON 解析失败: ${e.message}`);
      }
      parsed[name] = value;
    }
    // 全部解析成功后再统一试算; 任何错误都不覆盖上次有效结果
    const trial = core.computeWorkbench(parsed.Q, parsed.K, parsed.V, params);
    data.Q = parsed.Q;
    data.K = parsed.K;
    data.V = parsed.V;
    result = trial;
    error = null;
    notify();
    return getSnapshot();
  }

  function reportError(message) {
    error = message;
    notify();
  }

  function getSnapshot() {
    return {
      data: clone(data),
      params: Object.assign({}, params),
      result,
      error,
    };
  }

  function subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }

  recompute();
  return { setParams, applyEdits, reportError, getSnapshot, subscribe };
}

(typeof module !== 'undefined' && module.exports
  ? (x) => Object.assign(module.exports, x)
  : (x) => Object.assign(globalThis, x))({ createStore, DEFAULT_DATA, DEFAULT_PARAMS });
