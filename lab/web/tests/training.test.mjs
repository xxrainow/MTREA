import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

const source = await readFile(new URL('../src/training/model.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 } });
const model = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const {
  initialForm, formGroups, toRequest, requestKey, canStart, runNameFrom, displayValue, isActiveState, stateLabel,
  appendMetrics, sortCheckpoints, checkpointId, parseCheckpointId, progressPercent, formatDuration,
} = model;

// The shape of `scripts/train.sh --describe` (docs/TRAINING_JOBS.md section 1).
const param = (name, type, group, extra = {}) => ({
  name, flag: `--${name.replaceAll('_', '-')}`, type, group, required: false, default: null, choices: null, min: null,
  label: name, help: '', ...extra,
});
const describe = {
  run_name_template: '{method}_{dataset_name}_s{seed}_{yyyymmdd-HHMM}',
  methods: [{ name: 'h0', label: 'H0', description: '', max_batch_size_tested: 16, oom_batch_size: 32, seconds_per_step: { 16: 3.92 } }],
  fixed: { lerobot_flags: { 'policy.type': 'pi05' }, scheduler: 'cosine' },
  params: [
    param('method', 'string', 'basic', { required: true, choices: ['expert_only', 'h0'], label: 'Method' }),
    param('dataset', 'string', 'basic', { required: true }),
    param('dataset_root', 'string', 'basic'),
    param('seed', 'int', 'basic', { required: true, label: 'Seed' }),
    param('output', 'string', 'basic', { required: true }),
    param('steps', 'int', 'basic', { default: 10000, min: 1, label: 'Steps' }),
    param('base', 'string', 'fixed', { default: 'lerobot/pi05_base' }),
    param('lr', 'float', 'advanced', { default: 2.5e-5 }),
    param('keep_checkpoints', 'int_list', 'advanced', { default: [] }),
    param('wandb', 'bool', 'advanced', { default: false }),
    param('fake', 'flag', 'fake', { default: false }),
    param('fake_seconds_per_step', 'float', 'fake', { default: 0.01, min: 0 }),
  ],
};
const dataset = { backend: 'real', name: 'pick_v1' };
const valid = { ...initialForm(), dataset, runName: 'h0_pick_v1_s0', values: { method: 'h0', seed: '0' } };

test('lab-owned options never appear on the form; fake options only in fake mode', () => {
  const groups = formGroups(describe, false);
  const names = Object.values(groups).flat().map(p => p.name);
  for (const owned of ['dataset', 'dataset_root', 'output', 'fake']) assert.ok(!names.includes(owned), owned);
  assert.deepEqual(groups.fake, []);
  assert.deepEqual(formGroups(describe, true).fake.map(p => p.name), ['fake_seconds_per_step']);
  assert.deepEqual(groups.fixed.map(p => p.name), ['base']);
});

test('seed is explicit and has no default; untouched fields are not sent', () => {
  assert.equal(displayValue(describe.params[3], undefined), '');
  const { request, errors } = toRequest(describe, valid);
  assert.deepEqual(errors, []);
  assert.deepEqual(request.params, { method: 'h0', seed: 0 });
  const missing = toRequest(describe, { ...valid, values: { method: 'h0' } });
  assert.equal(missing.request, null);
  assert.ok(missing.errors.includes('Seed is required.'));
});

test('typed values are converted, and invalid ones rejected', () => {
  const { request } = toRequest(describe, { ...valid, values: { ...valid.values, steps: '600', lr: '1e-4', keep_checkpoints: '200, 400', wandb: true } });
  assert.deepEqual(request.params, { method: 'h0', seed: 0, steps: 600, lr: 1e-4, keep_checkpoints: [200, 400], wandb: true });
  for (const values of [{ seed: '-1' }, { seed: '1.5' }, { steps: '0' }, { lr: 'Infinity' }, { keep_checkpoints: 'a,b' }, { method: 'h9' }]) {
    assert.equal(toRequest(describe, { ...valid, values: { ...valid.values, ...values } }).request, null, JSON.stringify(values));
  }
  assert.equal(toRequest(describe, { ...valid, runName: '../other' }).request, null);
  assert.equal(toRequest(describe, { ...valid, dataset: null }).request, null);
});

test('this machine runs fake training only', () => {
  assert.equal(toRequest(describe, { ...valid, target: 'local', fake: false }).request, null);
  const { request } = toRequest(describe, { ...valid, target: 'local', fake: true, values: { ...valid.values, fake_seconds_per_step: '0.05' } });
  assert.deepEqual(request, { target: 'local', fake: true, dataset, run_name: 'h0_pick_v1_s0', params: { method: 'h0', seed: 0, fake_seconds_per_step: 0.05 } });
  // fake options are dropped, not sent, when fake is off
  assert.deepEqual(toRequest(describe, { ...valid, values: { ...valid.values, fake_seconds_per_step: '0.05' } }).request.params, { method: 'h0', seed: 0 });
});

test('Start needs a passing check of exactly the current inputs', () => {
  const key = requestKey(toRequest(describe, valid).request);
  const passed = { ok: true, checks: [] };
  assert.equal(canStart(passed, key, key), true);
  assert.equal(canStart({ ok: false, checks: [] }, key, key), false);
  assert.equal(canStart(null, key, key), false);
  const edited = requestKey(toRequest(describe, { ...valid, values: { ...valid.values, seed: '1' } }).request);
  assert.notEqual(edited, key);
  assert.equal(canStart(passed, key, edited), false);
  assert.equal(canStart(passed, '', ''), false);
});

test('run name follows the describe template', () => {
  const now = new Date(2026, 9, 31, 9, 5);
  assert.equal(runNameFrom(describe.run_name_template, 'h0', 'pick_v1', '0', now), 'h0_pick_v1_s0_20261031-0905');
  assert.equal(runNameFrom(describe.run_name_template, 'h0', 'lab/pick_v1', '', now), 'h0_pick_v1_sX_20261031-0905');
});

test('run states: only running or starting runs are active; failures say why', () => {
  for (const state of ['running', 'starting', 'queued']) assert.ok(isActiveState(state), state);
  for (const state of ['done', 'failed', 'stopped', 'ended_unknown', 'unknown']) assert.ok(!isActiveState(state), state);
  assert.equal(stateLabel('failed', 'oom'), 'Failed · Out of memory');
  assert.equal(stateLabel('ended_unknown'), 'Ended · status unknown');
});

test('metrics append by step without duplicates and stay bounded', () => {
  const m = step => ({ step, loss: 1 / step, lr: 1e-5 });
  let metrics = appendMetrics([], [m(10), m(20)]);
  metrics = appendMetrics(metrics, [m(20), m(30)]);
  assert.deepEqual(metrics.map(x => x.step), [10, 20, 30]);
  assert.equal(appendMetrics(metrics, []), metrics);
  assert.deepEqual(appendMetrics(metrics, [m(40), m(50)], 3).map(x => x.step), [30, 40, 50]);
});

test('checkpoints sort newest first and are identified by job id and step', () => {
  const entries = [{ step: 200 }, { step: 600 }, { step: 400 }];
  assert.deepEqual(sortCheckpoints(entries).map(c => c.step), [600, 400, 200]);
  assert.deepEqual(entries.map(c => c.step), [200, 600, 400]);
  const id = checkpointId('20261031-0905-train-abc123', 600);
  assert.deepEqual(parseCheckpointId(id), { jobId: '20261031-0905-train-abc123', step: 600 });
  assert.equal(parseCheckpointId('no-step'), null);
});

test('progress and durations', () => {
  assert.equal(progressPercent(150, 600), 25);
  assert.equal(progressPercent(undefined, 600), 0);
  assert.equal(progressPercent(700, 600), 100);
  assert.equal(formatDuration(null), '—');
  assert.equal(formatDuration(45), '45 s');
  assert.equal(formatDuration(39200), '10 h 53 min');
});
