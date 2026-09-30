// Workbench state: drafts, last-valid model, parameters and the derived result.
// A failed edit never overwrites the previous valid result.

import { analyzeModel } from './core.js';
import {
  BUILTIN_MODEL,
  DEFAULT_PARAMS,
  cloneModel,
  matrixToText,
  parseHeadFromText,
} from './data.js';

function draftsFromModel(model) {
  return model.heads.map((head) => ({
    q: matrixToText(head.q),
    k: matrixToText(head.k),
    v: matrixToText(head.v),
  }));
}

function normalizeParams(raw) {
  return {
    intervals: Number(raw.intervals),
    budget: Number(raw.budget),
    causal: Boolean(raw.causal),
    shared: Boolean(raw.shared),
  };
}

export class WorkbenchStore {
  constructor() {
    this._listeners = new Set();
    this.resetBuiltin();
  }

  resetBuiltin() {
    this._model = cloneModel(BUILTIN_MODEL);
    this._drafts = draftsFromModel(this._model);
    this._params = { ...DEFAULT_PARAMS };
    this._error = null;
    this._lastErrorHead = null;
    this._recompute('builtin');
    return this.getState();
  }

  _recompute(reason) {
    try {
      this._result = analyzeModel(this._model, this._params);
      this._error = null;
      this._lastErrorHead = null;
    } catch (err) {
      // Defensive: model state should already be valid, but never crash the UI.
      this._result = this._result || null;
      this._error = err.message;
    }
    this._emit(reason);
  }

  // Update one control parameter. Invalid numeric input is tolerated here;
  // analyzeModel clamps it (and reports the clamping via result.params).
  setParam(key, value) {
    if (!(key in this._params)) return this.getState();
    if (key === 'causal' || key === 'shared') {
      this._params[key] = Boolean(value);
    } else {
      const num = Number(value);
      this._params[key] = Number.isFinite(num) ? num : this._params[key];
    }
    this._recompute(`param:${key}`);
    return this.getState();
  }

  setParams(next) {
    this._params = normalizeParams({ ...this._params, ...next });
    this._recompute('params');
    return this.getState();
  }

  // Replace a head's draft text. Parses and validates immediately.
  // On failure the previous model/result stay intact and an error is exposed.
  editHead(headIndex, patch) {
    const currentDraft = this._drafts[headIndex];
    if (!currentDraft) {
      this._error = `Head ${headIndex} does not exist.`;
      this._lastErrorHead = headIndex;
      this._emit('edit:bad-head');
      return this.getState();
    }
    const nextDraft = { ...currentDraft, ...patch };
    this._drafts[headIndex] = nextDraft; // keep text so the user can keep typing

    let candidateHead;
    try {
      candidateHead = parseHeadFromText(nextDraft);
    } catch (err) {
      this._error = err.message;
      this._lastErrorHead = headIndex;
      this._emit('edit:parse-error');
      return this.getState();
    }

    const candidateModel = {
      heads: this._model.heads.map((head, h) => (h === headIndex ? candidateHead : head)),
    };
    try {
      analyzeModel(candidateModel, this._params);
    } catch (err) {
      this._error = err.message;
      this._lastErrorHead = headIndex;
      this._emit('edit:validation-error');
      return this.getState();
    }

    this._model = candidateModel;
    this._recompute('edit:ok');
    return this.getState();
  }

  dismissError() {
    this._error = null;
    this._lastErrorHead = null;
    this._emit('error-dismissed');
    return this.getState();
  }

  getState() {
    return {
      drafts: this._drafts.map((d) => ({ ...d })),
      params: { ...this._params },
      result: this._result,
      error: this._error,
      lastErrorHead: this._lastErrorHead,
      heads: this._model.heads.length,
      nq: this._model.heads[0].q.length,
      nk: this._model.heads[0].k.length,
    };
  }

  subscribe(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  _emit(reason) {
    for (const fn of this._listeners) {
      try {
        fn(this.getState(), reason);
      } catch (listenerErr) {
        // A broken view must not break the store.
        console.error('store listener failed:', listenerErr);
      }
    }
  }
}
