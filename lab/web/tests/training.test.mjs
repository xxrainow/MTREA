import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

const source = await readFile(new URL('../src/training/model.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 } });
const { defaultConfig, initialTrainingState, trainingReducer: reduce, validateConfig, isTrainingActive } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const valid = { ...defaultConfig, seed: '0', dataset: 'demo/pick_place' };
function start() {
  return reduce(reduce({ ...initialTrainingState, config: valid }, { type: 'check' }), { type: 'start', id: 'run-1' });
}

test('seed is explicit; invalid numerical values and unsafe run names are rejected', () => {
  assert.deepEqual(validateConfig(valid, []), []);
  for (const patch of [{ seed: '' }, { seed: '-1' }, { seed: '1.5' }, { steps: '1000' }, { learningRate: 'Infinity' }, { batchSize: '32' }, { saveFreq: '0' }, { runName: '../other' }]) {
    assert.ok(validateConfig({ ...valid, ...patch }, []).length, JSON.stringify(patch));
  }
});
test('input changes invalidate validation and prevent stale starts', () => {
  let state = reduce({ ...initialTrainingState, config: valid }, { type: 'check' });
  assert.equal(state.checked, true);
  state = reduce(state, { type: 'configure', config: { ...valid, target: 'local' } });
  assert.equal(state.checked, false);
  assert.equal(reduce(state, { type: 'start', id: 'stale' }).run, null);
});
test('active run retains its config and blocks duplicate starts or edits', () => {
  const state = start();
  assert.notEqual(state.run.config, state.config);
  assert.equal(reduce(state, { type: 'start', id: 'duplicate' }), state);
  assert.equal(reduce(state, { type: 'configure', config: defaultConfig }), state);
  assert.equal(reduce(state, { type: 'new' }), state);
});
test('connection loss remains active and cannot falsely complete or stop', () => {
  const state = reduce(start(), { type: 'disconnect' });
  assert.ok(isTrainingActive(state.run));
  for (const type of ['advance', 'complete', 'stop']) assert.equal(reduce(state, { type }), state);
  assert.equal(reduce(state, { type: 'reconnect' }).run.id, 'run-1');
});
test('completion preserves intermediate snapshots and adds one final checkpoint', () => {
  const advanced = reduce(start(), { type: 'advance' });
  assert.equal(advanced.checkpoints.at(-1).step, 2000);
  const done = reduce(advanced, { type: 'complete' });
  assert.equal(done.run.status, 'completed');
  assert.deepEqual(done.checkpoints.map(c => c.step), Array.from({length:10}, (_, i) => (i+1)*1000));
  assert.equal(done.checkpoints.at(-1).id, 'run-1:10000');
  assert.equal(reduce(done, { type: 'complete' }), done);
  assert.deepEqual(reduce(done, { type: 'new' }).checkpoints, done.checkpoints);
  assert.ok(validateConfig(valid, done.checkpoints).length);
});
test('stop keeps saved snapshots without claiming to save the current step', () => {
  const advanced = reduce(start(), { type: 'advance' });
  const stopped = reduce(advanced, { type: 'stop' });
  assert.equal(stopped.run.status, 'stopped');
  assert.equal(stopped.checkpoints, advanced.checkpoints);
  assert.equal(reduce(start(), { type: 'stop' }).checkpoints.length, 0);
});
test('advance never implies completion and does not duplicate a saved interval', () => {
  let state = start();
  for (let i = 0; i < 20; i++) state = reduce(state, { type: 'advance' });
  assert.equal(state.run.step, 9999);
  assert.equal(state.run.status, 'running');
  assert.equal(new Set(state.checkpoints.map(c => c.id)).size, state.checkpoints.length);
});

test('automatic ticks generate bounded metrics, logs, interval checkpoints and one job record', () => {
  let state = start();
  for (let i = 0; i < 100; i++) state = reduce(state, { type: 'tick' });
  assert.equal(state.run.status, 'completed');
  assert.equal(state.run.metrics.length, 100);
  assert.equal(state.run.metrics.at(-1).step, 10000);
  assert.ok(Math.abs(state.run.metrics.at(-1).lr - Number(valid.learningRate) / 10) < 1e-12);
  assert.equal(state.checkpoints.length, 10);
  assert.equal(state.runs.length, 1);
  assert.equal(state.runs[0], state.run);
  state = reduce(state, { type: 'new' });
  assert.equal(state.run, null);
  assert.equal(state.runs[0].status, 'completed');
});
