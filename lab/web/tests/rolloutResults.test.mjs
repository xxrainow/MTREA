import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

const source = await readFile(new URL('../src/rollouts/results.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 } });
const { listRolloutResults, outcomeCounts, resultTargetSelection, formatElapsed } =
  await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);

function trial(id, instruction = 'Put the block in the box.', metadata = {}) {
  return { job: { id, submitted_at: '2026-11-26T09:00:00+09:00', spec: { label: instruction, metadata } },
    state: 'failed', owner_alive: false, status: { reason: 'time_limit' },
    session: { task: instruction, checkpoint: 'policy/legacy' } };
}

function validation(id, { instruction = 'Put the block in the box.', checkpoint = 'policy/h0', mode = 'real',
  created = '2026-11-26T01:00:00Z', attempts = [] } = {}) {
  return { id, state: 'completed', session: { configuration: { instruction }, checkpoint,
    server_mode: mode, created_at: created, target_count: 10 }, attempts };
}

test('each rollout is a separate entry even when instructions match', () => {
  const runs = listRolloutResults([
    validation('a'), validation('b', { checkpoint: 'policy/expert', mode: 'hold' }),
    validation('c', { instruction: 'Put the block in the box' }),
  ], []);
  assert.equal(runs.length, 3);
  assert.deepEqual(new Set(runs.map(run => run.id)), new Set(['a', 'b', 'c']));
  assert.equal(runs.filter(run => run.instruction.endsWith('.')).length, 2);
  assert.deepEqual(new Set(runs.map(run => run.checkpoint)), new Set(['policy/h0', 'policy/expert']));
  assert.deepEqual(new Set(runs.map(run => run.mode)), new Set(['real', 'hold']));
});

test('linked trials are not duplicated and missing validation records stay visible read-only', () => {
  const linked = trial('linked', undefined, { validation: { id: 'a' } });
  const orphan = trial('orphan', undefined, { validation: { id: 'missing' } });
  const legacy = trial('legacy');
  const grouped = validation('a', { attempts: [{ number: 1, rollout: linked, outcome: 'unjudged', notes: '' }] });
  const runs = listRolloutResults([grouped], [linked, orphan, legacy]);
  assert.equal(runs.length, 3);
  assert.equal(runs.find(run => run.id === 'orphan').kind, 'unlinked');
  assert.equal(runs.find(run => run.id === 'legacy').kind, 'standalone');
  assert.equal(runs.filter(run => run.key === 'standalone:linked').length, 0);
});

test('runs sort by actual time across recorded UTC offsets', () => {
  const runs = listRolloutResults([
    validation('older', { created: '2026-11-26T09:00:00+09:00' }),
    validation('newer', { created: '2026-11-26T01:00:00Z' }),
  ], []);
  assert.deepEqual(runs.map(run => run.id), ['newer', 'older']);
});

test('outcomes use explicit annotations and never infer failure from execution errors', () => {
  const failed = trial('failed');
  assert.deepEqual(outcomeCounts([
    { outcome: 'success', rollout: failed }, { outcome: 'failure', rollout: failed }, { outcome: 'unjudged', rollout: failed },
  ]), { success: 1, failure: 1, unjudged: 1 });
  assert.deepEqual(outcomeCounts([]), { success: 0, failure: 0, unjudged: 0 });
});

test('result links retain validation and attempt identity; absent legacy metrics are explicit', () => {
  assert.deepEqual(resultTargetSelection({ validationId: 'a', jobId: 'trial' }), { runKey: 'validation:a', jobId: 'trial' });
  assert.deepEqual(resultTargetSelection({ jobId: 'old' }), { runKey: 'standalone:old', jobId: 'old' });
  assert.deepEqual(resultTargetSelection({}), {});
  assert.equal(formatElapsed(undefined), '—');
  assert.equal(formatElapsed(null), '—');
  assert.equal(formatElapsed(1.25), '1.3 s');
});

test('multi-attempt validations stay together and every legacy rollout is one attempt', () => {
  const first = trial('first', undefined, { validation: { id: 'multi' } });
  const second = trial('second', undefined, { validation: { id: 'multi' } });
  const attempts = [first, second].map((rollout, index) => ({ number: index + 1, rollout, outcome: 'unjudged', notes: '' }));
  const runs = listRolloutResults([validation('multi', { attempts })], [first, second, trial('old-a'), trial('old-b')]);
  assert.equal(runs.length, 3);
  const multi = runs.find(run => run.id === 'multi');
  assert.equal(multi.targetCount, 10);
  assert.deepEqual(multi.attempts.map(item => item.rollout.job.id), ['first', 'second']);
  for (const id of ['old-a', 'old-b']) {
    const old = runs.find(run => run.id === id);
    assert.equal(old.targetCount, 1);
    assert.equal(old.attempts.length, 1);
    assert.equal(old.attempts[0].number, 1);
    assert.equal(old.attempts[0].rollout.job.id, id);
  }
});

test('merged bundles retain all attempts and leave their sources visible', () => {
  const legacy = trial('legacy');
  const attempt = { number: 1, rollout: legacy, outcome: 'unjudged', notes: '' };
  const merged = { id: 'merged', name: 'Combined report', created_at: '2026-11-26T12:00:00Z',
    condition: { instruction: 'Put the block in the box.', checkpoint: 'policy/legacy', mode: 'real' },
    target_count: 2, attempts: [attempt, { ...attempt, number: 2, rollout: trial('other') }] };
  const runs = listRolloutResults([], [legacy], [merged]);
  assert.equal(runs.length, 2);
  assert.equal(runs[0].kind, 'merged');
  assert.equal(runs[0].name, 'Combined report');
  assert.equal(runs[0].attempts.length, 2);
  assert.equal(runs[1].id, 'legacy');
});
